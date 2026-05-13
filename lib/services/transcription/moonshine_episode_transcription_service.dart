// Copyright 2020 Ben Hills and the project contributors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

import 'dart:async';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:anytime/core/utils.dart';
import 'package:anytime/entities/episode.dart';
import 'package:anytime/entities/transcript.dart';
import 'package:anytime/services/transcription/episode_transcription_service.dart';
import 'package:archive/archive_io.dart';
import 'package:crypto/crypto.dart';
import 'package:ffmpeg_kit_flutter_new_min/ffmpeg_kit.dart';
import 'package:ffmpeg_kit_flutter_new_min/return_code.dart';
import 'package:logging/logging.dart';
import 'package:path/path.dart' as path;
import 'package:path_provider/path_provider.dart';
import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa;
import 'package:synchronized/synchronized.dart';

/// On-device transcription service backed by Moonshine (v2 tiny, English-only,
/// int8) via `sherpa_onnx`. Selected via the Transcription provider setting.
class MoonshineEpisodeTranscriptionService implements EpisodeTranscriptionService {
  MoonshineEpisodeTranscriptionService();

  static const _modelArchiveUrl = 'https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/'
      'sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27.tar.bz2';
  // SHA-256 of the pinned model archive. Fails closed if the upstream asset
  // is swapped out so a tampered archive never reaches _extractTarBz2.
  static const _modelArchiveSha256 = '9ec31b342d8fa3240c3b81b8f82e1cf7e3ac467c93ca5a999b741d5887164f8d';
  static const _modelDirName = 'sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27';
  static const _encoderFile = 'encoder_model.ort';
  static const _decoderFile = 'decoder_model_merged.ort';
  static const _tokensFile = 'tokens.txt';
  static const _vadModelUrl = 'https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx';
  static const _vadModelSha256 = '9e2449e1087496d8d4caba907f23e0bd3f78d91fa552479bb9c23ac09cbb1fd6';
  static const _vadModelFileName = 'silero_vad.onnx';
  static const _vadProgressShare = 0.2;
  static const _vadSliceSeconds = 0.5;
  // Manual Android validation showed long VAD segments can trigger Moonshine
  // v2 ONNX broadcast errors and empty text. Keep decode windows at the known
  // safe prototype chunk length while still using VAD for natural boundaries
  // whenever speech pauses earlier.
  static const _vadMaxSpeechDuration = 5.0;
  static const _maxDecodeSegmentSeconds = 5.0;
  // Hard cap on the model archive download. The published asset is ~30 MB;
  // anything significantly larger is treated as an unexpected/replaced asset
  // and rejected to avoid filling storage or feeding a decompression bomb.
  static const _maxModelBytes = 100 * 1024 * 1024;
  static const _maxVadModelBytes = 10 * 1024 * 1024;

  static final _log = Logger('MoonshineEpisodeTranscriptionService');
  static bool _bindingsInitialized = false;

  final Lock _modelLock = Lock();

  @override
  Future<Transcript> transcribeDownloadedEpisode({
    required Episode episode,
    void Function(EpisodeTranscriptionProgress progress)? onProgress,
  }) async {
    if (!(Platform.isAndroid || Platform.isIOS || Platform.isMacOS)) {
      throw const EpisodeTranscriptionException(
        'Moonshine transcription is only supported on Android, iOS, and macOS.',
      );
    }

    onProgress?.call(const EpisodeTranscriptionProgress(
      stage: EpisodeTranscriptionStage.preparing,
      message: 'Preparing local audio...',
    ));

    final audioPath = await resolvePath(episode);
    if (!File(audioPath).existsSync()) {
      throw EpisodeTranscriptionException('Downloaded audio file is missing: $audioPath');
    }

    final modelPaths = await _ensureModel(onProgress: onProgress);
    final vadModelPath = await _ensureVadModel(onProgress: onProgress);
    _log.fine('Silero VAD model ready at $vadModelPath');

    onProgress?.call(const EpisodeTranscriptionProgress(
      stage: EpisodeTranscriptionStage.preparing,
      message: 'Preparing audio for Moonshine...',
    ));

    final workDir = await _freshWorkDir();
    try {
      final wavPath = await _renderTo16kMonoWav(audioPath, workDir);
      _ensureSherpaBindingsInitialized();

      // sherpa.readWave loads the full episode into memory. That is acceptable
      // only while Moonshine remains a dev-gated prototype.
      final wave = sherpa.readWave(wavPath);
      final sampleRate = wave.sampleRate;
      final samples = wave.samples;
      if (sampleRate <= 0 || samples.isEmpty) {
        throw const EpisodeTranscriptionException(
          'Audio preparation produced no samples for Moonshine.',
        );
      }

      final stopwatch = Stopwatch()..start();
      final detectedSegments = await _detectSpeechSegments(
        samples: samples,
        sampleRate: sampleRate,
        vadModelPath: vadModelPath,
        onProgress: onProgress,
      );
      final segments = _enforceMaxDecodeSegmentDuration(detectedSegments, sampleRate);
      if (segments.isEmpty) {
        throw const EpisodeTranscriptionException(
          'VAD found no speech segments for Moonshine.',
        );
      }
      _logVadSegmentDistribution('VAD', detectedSegments, sampleRate);
      _logVadSegmentDistribution('Moonshine decode', segments, sampleRate);

      final sherpa.OfflineRecognizer recognizer;
      try {
        recognizer = _buildRecognizer(modelPaths);
      } catch (err, stack) {
        _log.severe('Moonshine recognizer init failed', err, stack);
        rethrow;
      }
      _log.info('Moonshine recognizer built with '
          'encoder=${modelPaths.encoder} decoder=${modelPaths.decoder}');

      final subtitles = <Subtitle>[];

      try {
        for (var i = 0; i < segments.length; i++) {
          final segment = segments[i];
          final segmentStart = Duration(
            microseconds: (segment.start * Duration.microsecondsPerSecond) ~/ sampleRate,
          );
          // Yield to the event loop so the progress dialog can repaint
          // between sync FFI calls.
          await Future<void>.delayed(Duration.zero);

          final _ChunkResult result;
          try {
            result = _transcribeSamples(
              recognizer: recognizer,
              samples: segment.samples,
              sampleRate: sampleRate,
            );
          } catch (err, stack) {
            _log.severe(
              'Moonshine failed on VAD segment ${i + 1}/${segments.length} '
              'starting at ${segmentStart.inMilliseconds}ms',
              err,
              stack,
            );
            throw EpisodeTranscriptionException(
              'Moonshine segment ${i + 1}/${segments.length} failed: $err',
            );
          }

          onProgress?.call(EpisodeTranscriptionProgress(
            stage: EpisodeTranscriptionStage.transcribing,
            message: 'Transcribed segment ${i + 1}/${segments.length} '
                '(${_format(stopwatch.elapsed)} elapsed)',
            progress: _vadProgressShare + ((i + 1) / segments.length) * (1 - _vadProgressShare),
          ));

          if (result.text.isEmpty) continue;

          subtitles.add(Subtitle(
            index: subtitles.length + 1,
            start: segmentStart,
            end: segmentStart + result.duration,
            data: result.text,
          ));
        }
      } finally {
        recognizer.free();
      }

      if (subtitles.isEmpty) {
        throw const EpisodeTranscriptionException(
          'Moonshine returned no usable transcript segments.',
        );
      }

      onProgress?.call(EpisodeTranscriptionProgress(
        stage: EpisodeTranscriptionStage.completed,
        message: 'Transcript ready after ${_format(stopwatch.elapsed)}.',
        progress: 1.0,
      ));

      return Transcript(
        subtitles: List<Subtitle>.unmodifiable(subtitles),
        provenance: TranscriptProvenance.localAi,
        provider: 'moonshine',
      );
    } catch (error, stack) {
      _log.severe('Moonshine transcription aborted', error, stack);
      rethrow;
    } finally {
      await _cleanupDir(workDir);
    }
  }

  sherpa.OfflineRecognizer _buildRecognizer(_MoonshineModelPaths paths) {
    _ensureSherpaBindingsInitialized();
    final moonshine = sherpa.OfflineMoonshineModelConfig(
      encoder: paths.encoder,
      mergedDecoder: paths.decoder,
    );
    final modelConfig = sherpa.OfflineModelConfig(
      moonshine: moonshine,
      tokens: paths.tokens,
      debug: false,
      numThreads: 1,
    );
    return sherpa.OfflineRecognizer(sherpa.OfflineRecognizerConfig(model: modelConfig));
  }

  void _ensureSherpaBindingsInitialized() {
    if (_bindingsInitialized) return;
    sherpa.initBindings();
    _bindingsInitialized = true;
  }

  Future<List<sherpa.SpeechSegment>> _detectSpeechSegments({
    required Float32List samples,
    required int sampleRate,
    required String vadModelPath,
    void Function(EpisodeTranscriptionProgress progress)? onProgress,
  }) async {
    _ensureSherpaBindingsInitialized();

    final config = sherpa.VadModelConfig(
      sileroVad: sherpa.SileroVadModelConfig(
        model: vadModelPath,
        threshold: 0.5,
        minSilenceDuration: 0.25,
        minSpeechDuration: 0.25,
        maxSpeechDuration: _vadMaxSpeechDuration,
      ),
      sampleRate: sampleRate,
      numThreads: 1,
      debug: false,
    );
    final vad = sherpa.VoiceActivityDetector(
      config: config,
      bufferSizeInSeconds: 30.0,
    );
    final segments = <sherpa.SpeechSegment>[];

    try {
      final sliceSize = math.max(1, (sampleRate * _vadSliceSeconds).round());
      for (var offset = 0; offset < samples.length; offset += sliceSize) {
        final end = math.min(offset + sliceSize, samples.length);
        vad.acceptWaveform(Float32List.sublistView(samples, offset, end));
        _drainVadSegments(vad, segments);

        onProgress?.call(EpisodeTranscriptionProgress(
          stage: EpisodeTranscriptionStage.transcribing,
          message: 'Detecting speech for Moonshine...',
          progress: _vadProgressShare * (end / samples.length),
        ));

        // Yield after each synchronous VAD FFI call so the UI can repaint.
        await Future<void>.delayed(Duration.zero);
      }

      vad.flush();
      _drainVadSegments(vad, segments);
      return segments;
    } finally {
      vad.free();
    }
  }

  void _drainVadSegments(
    sherpa.VoiceActivityDetector vad,
    List<sherpa.SpeechSegment> segments,
  ) {
    while (!vad.isEmpty()) {
      segments.add(vad.front());
      vad.pop();
    }
  }

  List<sherpa.SpeechSegment> _enforceMaxDecodeSegmentDuration(
    List<sherpa.SpeechSegment> segments,
    int sampleRate,
  ) {
    if (sampleRate <= 0) return segments;

    final maxSamples = math.max(1, (sampleRate * _maxDecodeSegmentSeconds).round());
    final splitSegments = <sherpa.SpeechSegment>[];
    for (final segment in segments) {
      if (segment.samples.length <= maxSamples) {
        splitSegments.add(segment);
        continue;
      }

      for (var offset = 0; offset < segment.samples.length; offset += maxSamples) {
        final end = math.min(offset + maxSamples, segment.samples.length);
        splitSegments.add(sherpa.SpeechSegment(
          samples: Float32List.sublistView(segment.samples, offset, end),
          start: segment.start + offset,
        ));
      }
    }
    return splitSegments;
  }

  void _logVadSegmentDistribution(
    String label,
    List<sherpa.SpeechSegment> segments,
    int sampleRate,
  ) {
    if (segments.isEmpty || sampleRate <= 0) return;

    var minSamples = segments.first.samples.length;
    var maxSamples = minSamples;
    var totalSamples = 0;
    var overTenSeconds = 0;

    for (final segment in segments) {
      final length = segment.samples.length;
      minSamples = math.min(minSamples, length);
      maxSamples = math.max(maxSamples, length);
      totalSamples += length;
      if (length / sampleRate > 10.0) {
        overTenSeconds++;
      }
    }

    final minSeconds = minSamples / sampleRate;
    final avgSeconds = totalSamples / segments.length / sampleRate;
    final maxSeconds = maxSamples / sampleRate;
    _log.info(
      '$label segment distribution: count=${segments.length}; '
      'duration min=${minSeconds.toStringAsFixed(2)}s '
      'avg=${avgSeconds.toStringAsFixed(2)}s '
      'max=${maxSeconds.toStringAsFixed(2)}s '
      '>10s=$overTenSeconds.',
    );
  }

  _ChunkResult _transcribeSamples({
    required sherpa.OfflineRecognizer recognizer,
    required Float32List samples,
    required int sampleRate,
  }) {
    final stream = recognizer.createStream();
    try {
      stream.acceptWaveform(samples: samples, sampleRate: sampleRate);
      recognizer.decode(stream);
      final text = recognizer.getResult(stream).text.trim();
      final durationMicros = sampleRate == 0 ? 0 : (samples.length * Duration.microsecondsPerSecond) ~/ sampleRate;
      return _ChunkResult(text: text, duration: Duration(microseconds: durationMicros));
    } finally {
      stream.free();
    }
  }

  Future<String> _renderTo16kMonoWav(
    String inputPath,
    Directory outDir,
  ) async {
    final outPath = path.join(outDir.path, 'episode.wav');
    final args = <String>[
      '-y',
      '-i',
      inputPath,
      '-vn',
      '-ac',
      '1',
      '-ar',
      '16000',
      '-c:a',
      'pcm_s16le',
      outPath,
    ];

    final session = await FFmpegKit.executeWithArguments(args);
    final rc = await session.getReturnCode();
    if (!ReturnCode.isSuccess(rc)) {
      final output = (await session.getOutput())?.trim();
      _log.warning('ffmpeg render failed: ${args.join(' ')}\n$output');
      throw EpisodeTranscriptionException(
        'Failed to prepare audio for Moonshine.'
        '${output == null || output.isEmpty ? '' : ' $output'}',
      );
    }

    return outPath;
  }

  Future<_MoonshineModelPaths> _ensureModel({
    void Function(EpisodeTranscriptionProgress progress)? onProgress,
  }) {
    return _modelLock.synchronized(() async {
      final modelRoot = await _modelRoot();
      final modelDir = Directory(path.join(modelRoot.path, _modelDirName));
      final encoder = File(path.join(modelDir.path, _encoderFile));
      final decoder = File(path.join(modelDir.path, _decoderFile));
      final tokens = File(path.join(modelDir.path, _tokensFile));

      if (encoder.existsSync() && decoder.existsSync() && tokens.existsSync()) {
        return _MoonshineModelPaths(
          encoder: encoder.path,
          decoder: decoder.path,
          tokens: tokens.path,
        );
      }

      await modelRoot.create(recursive: true);

      onProgress?.call(const EpisodeTranscriptionProgress(
        stage: EpisodeTranscriptionStage.downloadingModel,
        message: 'Downloading Moonshine model (~30 MB)...',
        progress: 0.0,
      ));

      final tarballPath = path.join(modelRoot.path, 'moonshine.tar.bz2');
      await _downloadWithProgress(
        url: _modelArchiveUrl,
        destination: File(tarballPath),
        maxBytes: _maxModelBytes,
        label: 'Moonshine model',
        onProgress: (downloaded, total) {
          onProgress?.call(EpisodeTranscriptionProgress(
            stage: EpisodeTranscriptionStage.downloadingModel,
            message: 'Downloading Moonshine model...',
            progress: total > 0 ? downloaded / total : null,
          ));
        },
      );

      onProgress?.call(const EpisodeTranscriptionProgress(
        stage: EpisodeTranscriptionStage.preparing,
        message: 'Verifying Moonshine model...',
      ));

      try {
        await _verifyFileSha256(tarballPath, _modelArchiveSha256, label: 'Moonshine model archive');

        onProgress?.call(const EpisodeTranscriptionProgress(
          stage: EpisodeTranscriptionStage.preparing,
          message: 'Extracting Moonshine model...',
        ));

        await _extractTarBz2(tarballPath, modelRoot.path);

        if (!(encoder.existsSync() && decoder.existsSync() && tokens.existsSync())) {
          throw const EpisodeTranscriptionException(
            'Moonshine model archive is missing expected files after extraction.',
          );
        }

        return _MoonshineModelPaths(
          encoder: encoder.path,
          decoder: decoder.path,
          tokens: tokens.path,
        );
      } finally {
        final tarball = File(tarballPath);
        if (tarball.existsSync()) {
          try {
            await tarball.delete();
          } on FileSystemException {
            // Best-effort cleanup; preserve the original error if any.
          }
        }
      }
    });
  }

  Future<String> _ensureVadModel({
    void Function(EpisodeTranscriptionProgress progress)? onProgress,
  }) {
    return _modelLock.synchronized(() async {
      final modelRoot = await _modelRoot();
      final vadModel = File(path.join(modelRoot.path, _vadModelFileName));

      if (vadModel.existsSync()) {
        await _verifyFileSha256(vadModel.path, _vadModelSha256, label: 'Silero VAD model');
        return vadModel.path;
      }

      await modelRoot.create(recursive: true);

      onProgress?.call(const EpisodeTranscriptionProgress(
        stage: EpisodeTranscriptionStage.downloadingModel,
        message: 'Downloading Silero VAD model...',
        progress: 0.0,
      ));

      await _downloadWithProgress(
        url: _vadModelUrl,
        destination: vadModel,
        maxBytes: _maxVadModelBytes,
        label: 'Silero VAD model',
        onProgress: (downloaded, total) {
          onProgress?.call(EpisodeTranscriptionProgress(
            stage: EpisodeTranscriptionStage.downloadingModel,
            message: 'Downloading Silero VAD model...',
            progress: total > 0 ? downloaded / total : null,
          ));
        },
      );

      try {
        onProgress?.call(const EpisodeTranscriptionProgress(
          stage: EpisodeTranscriptionStage.preparing,
          message: 'Verifying Silero VAD model...',
        ));

        await _verifyFileSha256(vadModel.path, _vadModelSha256, label: 'Silero VAD model');
        return vadModel.path;
      } catch (_) {
        if (vadModel.existsSync()) {
          try {
            await vadModel.delete();
          } catch (_) {
            // Best-effort cleanup; preserve the original verification error.
          }
        }
        rethrow;
      }
    });
  }

  Future<void> _downloadWithProgress({
    required String url,
    required File destination,
    required int maxBytes,
    required String label,
    required void Function(int downloaded, int total) onProgress,
  }) async {
    final client = HttpClient()..userAgent = 'Anytime Podcast Player';
    HttpClientResponse? response;
    try {
      final request = await client.getUrl(Uri.parse(url));
      response = await request.close();
      if (response.statusCode < 200 || response.statusCode >= 300) {
        throw EpisodeTranscriptionException(
          '$label download failed with status ${response.statusCode}.',
        );
      }
      final total = response.contentLength;
      if (total > maxBytes) {
        throw EpisodeTranscriptionException(
          '$label download is $total bytes, exceeds cap of $maxBytes.',
        );
      }
      final sink = destination.openWrite();
      var downloaded = 0;
      var success = false;
      try {
        await for (final chunk in response) {
          downloaded += chunk.length;
          if (downloaded > maxBytes) {
            throw EpisodeTranscriptionException(
              '$label download exceeded cap of $maxBytes bytes.',
            );
          }
          sink.add(chunk);
          onProgress(downloaded, total);
        }
        success = true;
      } finally {
        await sink.close();
        if (!success && destination.existsSync()) {
          try {
            await destination.delete();
          } catch (_) {
            // Best-effort cleanup; ignore.
          }
        }
      }
    } finally {
      client.close(force: true);
    }
  }

  Future<void> _verifyFileSha256(String filePath, String expected, {required String label}) async {
    final digest = await sha256.bind(File(filePath).openRead()).first;
    final actual = digest.toString();
    if (actual != expected) {
      throw EpisodeTranscriptionException(
        '$label SHA-256 mismatch: expected $expected, got $actual.',
      );
    }
  }

  Future<void> _extractTarBz2(String tarballPath, String destDir) async {
    final bytes = await File(tarballPath).readAsBytes();
    final tarBytes = BZip2Decoder().decodeBytes(bytes);
    final archive = TarDecoder().decodeBytes(tarBytes);
    final destAbsolute = path.normalize(path.absolute(destDir));
    for (final entry in archive) {
      final outAbsolute = path.normalize(path.absolute(path.join(destDir, entry.name)));
      // Reject any entry whose normalized path escapes destDir (zip-slip).
      if (outAbsolute != destAbsolute && !path.isWithin(destAbsolute, outAbsolute)) {
        throw EpisodeTranscriptionException(
          'Refusing to extract tar entry outside model dir: ${entry.name}',
        );
      }
      if (entry.isFile) {
        final out = File(outAbsolute);
        await out.parent.create(recursive: true);
        await out.writeAsBytes(entry.content as List<int>);
      } else {
        await Directory(outAbsolute).create(recursive: true);
      }
    }
  }

  Future<Directory> _modelRoot() async {
    if (Platform.isAndroid) {
      return getApplicationSupportDirectory();
    }
    return getLibraryDirectory();
  }

  Future<Directory> _freshWorkDir() async {
    final tmp = await getTemporaryDirectory();
    final dir = Directory(path.join(
      tmp.path,
      'moonshine_${DateTime.now().microsecondsSinceEpoch}',
    ));
    await dir.create(recursive: true);
    return dir;
  }

  Future<void> _cleanupDir(Directory dir) async {
    if (!dir.existsSync()) return;
    try {
      await dir.delete(recursive: true);
    } catch (e) {
      _log.warning('Failed to clean up ${dir.path}: $e');
    }
  }

  String _format(Duration d) {
    final s = d.inSeconds;
    final h = s ~/ 3600;
    final m = (s % 3600) ~/ 60;
    final r = s % 60;
    if (h > 0) return '${h}h${m.toString().padLeft(2, '0')}m${r.toString().padLeft(2, '0')}s';
    if (m > 0) return '${m}m${r.toString().padLeft(2, '0')}s';
    return '${s}s';
  }
}

class _MoonshineModelPaths {
  final String encoder;
  final String decoder;
  final String tokens;
  const _MoonshineModelPaths({
    required this.encoder,
    required this.decoder,
    required this.tokens,
  });
}

class _ChunkResult {
  final String text;
  final Duration duration;
  const _ChunkResult({required this.text, required this.duration});
}
