// Copyright 2020 Ben Hills and the project contributors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

import 'dart:async';
import 'dart:io';

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

/// Prototype on-device transcription service backed by Moonshine (v2 tiny,
/// English-only, int8) via `sherpa_onnx`. Gated behind a dev-only file flag:
/// create `<app-support>/use_moonshine` to switch local transcription from
/// Whisper to this implementation. Not yet surfaced in the settings UI.
class MoonshineEpisodeTranscriptionService implements EpisodeTranscriptionService {
  MoonshineEpisodeTranscriptionService();

  static const _modelArchiveUrl =
      'https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/'
      'sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27.tar.bz2';
  // SHA-256 of the pinned model archive. Fails closed if the upstream asset
  // is swapped out so a tampered archive never reaches _extractTarBz2.
  static const _modelArchiveSha256 =
      '9ec31b342d8fa3240c3b81b8f82e1cf7e3ac467c93ca5a999b741d5887164f8d';
  static const _modelDirName = 'sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27';
  static const _encoderFile = 'encoder_model.ort';
  static const _decoderFile = 'decoder_model_merged.ort';
  static const _tokensFile = 'tokens.txt';
  static const _chunkSeconds = 5;
  // Hard cap on the model archive download. The published asset is ~30 MB;
  // anything significantly larger is treated as an unexpected/replaced asset
  // and rejected to avoid filling storage or feeding a decompression bomb.
  static const _maxModelBytes = 100 * 1024 * 1024;

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

    onProgress?.call(const EpisodeTranscriptionProgress(
      stage: EpisodeTranscriptionStage.preparing,
      message: 'Chunking audio for Moonshine...',
    ));

    final workDir = await _freshWorkDir();
    try {
      final chunkPaths = await _splitTo16kMonoWavChunks(audioPath, workDir);
      _log.info('Chunked audio into ${chunkPaths.length} files at ${workDir.path}');
      if (chunkPaths.isEmpty) {
        throw const EpisodeTranscriptionException(
          'Audio preparation produced no chunks for Moonshine.',
        );
      }

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
      final stopwatch = Stopwatch()..start();

      try {
        for (var i = 0; i < chunkPaths.length; i++) {
          final chunkStart = Duration(seconds: i * _chunkSeconds);
          // Yield to the event loop so the progress dialog can repaint
          // between sync FFI calls.
          await Future<void>.delayed(Duration.zero);

          final _ChunkResult result;
          try {
            result = _transcribeChunk(recognizer, chunkPaths[i]);
          } catch (err, stack) {
            _log.severe(
              'Moonshine failed on chunk ${i + 1}/${chunkPaths.length} '
                  '(${chunkPaths[i]})',
              err,
              stack,
            );
            throw EpisodeTranscriptionException(
              'Moonshine chunk ${i + 1}/${chunkPaths.length} failed: $err',
            );
          }

          onProgress?.call(EpisodeTranscriptionProgress(
            stage: EpisodeTranscriptionStage.transcribing,
            message: 'Transcribed chunk ${i + 1}/${chunkPaths.length} '
                '(${_format(stopwatch.elapsed)} elapsed)',
            progress: (i + 1) / chunkPaths.length,
          ));

          if (result.text.isEmpty) continue;

          subtitles.add(Subtitle(
            index: subtitles.length + 1,
            start: chunkStart,
            end: chunkStart + result.duration,
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
    if (!_bindingsInitialized) {
      sherpa.initBindings();
      _bindingsInitialized = true;
    }
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

  _ChunkResult _transcribeChunk(sherpa.OfflineRecognizer recognizer, String wavPath) {
    final stream = recognizer.createStream();
    try {
      final wave = sherpa.readWave(wavPath);
      stream.acceptWaveform(samples: wave.samples, sampleRate: wave.sampleRate);
      recognizer.decode(stream);
      final text = recognizer.getResult(stream).text.trim();
      final durationMicros = wave.sampleRate == 0
          ? 0
          : (wave.samples.length * Duration.microsecondsPerSecond) ~/ wave.sampleRate;
      return _ChunkResult(text: text, duration: Duration(microseconds: durationMicros));
    } finally {
      stream.free();
    }
  }

  /// Runs ffmpeg once to emit N 16kHz mono pcm_s16le WAV chunks of
  /// [_chunkSeconds] each using the segment muxer. Returns chunk paths in
  /// order.
  Future<List<String>> _splitTo16kMonoWavChunks(
    String inputPath,
    Directory outDir,
  ) async {
    final pattern = path.join(outDir.path, 'chunk_%03d.wav');
    final args = <String>[
      '-y',
      '-i', inputPath,
      '-vn',
      '-ac', '1',
      '-ar', '16000',
      '-c:a', 'pcm_s16le',
      '-f', 'segment',
      '-segment_time', '$_chunkSeconds',
      '-reset_timestamps', '1',
      pattern,
    ];

    final session = await FFmpegKit.executeWithArguments(args);
    final rc = await session.getReturnCode();
    if (!ReturnCode.isSuccess(rc)) {
      final output = (await session.getOutput())?.trim();
      _log.warning('ffmpeg chunking failed: ${args.join(' ')}\n$output');
      throw EpisodeTranscriptionException(
        'Failed to prepare audio chunks for Moonshine.'
        '${output == null || output.isEmpty ? '' : ' $output'}',
      );
    }

    final chunks = outDir
        .listSync()
        .whereType<File>()
        .where((f) => path.basename(f.path).startsWith('chunk_'))
        .map((f) => f.path)
        .toList()
      ..sort();
    return chunks;
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
        await _verifyArchiveSha256(tarballPath, _modelArchiveSha256);

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

  Future<void> _downloadWithProgress({
    required String url,
    required File destination,
    required int maxBytes,
    required void Function(int downloaded, int total) onProgress,
  }) async {
    final client = HttpClient()..userAgent = 'Anytime Podcast Player';
    HttpClientResponse? response;
    try {
      final request = await client.getUrl(Uri.parse(url));
      response = await request.close();
      if (response.statusCode < 200 || response.statusCode >= 300) {
        throw EpisodeTranscriptionException(
          'Moonshine model download failed with status ${response.statusCode}.',
        );
      }
      final total = response.contentLength;
      if (total > maxBytes) {
        throw EpisodeTranscriptionException(
          'Moonshine model download is $total bytes, exceeds cap of $maxBytes.',
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
              'Moonshine model download exceeded cap of $maxBytes bytes.',
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

  Future<void> _verifyArchiveSha256(String tarballPath, String expected) async {
    final digest = await sha256.bind(File(tarballPath).openRead()).first;
    final actual = digest.toString();
    if (actual != expected) {
      throw EpisodeTranscriptionException(
        'Moonshine model archive SHA-256 mismatch: expected $expected, got $actual.',
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
