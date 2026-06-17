// Copyright 2020 Ben Hills and the project contributors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

import 'dart:async';

import 'package:anytime/bloc/podcast/audio_bloc.dart';
import 'package:anytime/bloc/podcast/episode_bloc.dart';
import 'package:anytime/bloc/podcast/queue_bloc.dart';
import 'package:anytime/bloc/settings/settings_bloc.dart';
import 'package:anytime/entities/app_settings.dart';
import 'package:anytime/entities/episode.dart';
import 'package:anytime/entities/podcast.dart';
import 'package:anytime/entities/sleep.dart';
import 'package:anytime/entities/transcript.dart';
import 'package:anytime/l10n/L.dart';
import 'package:anytime/repository/repository.dart';
import 'package:anytime/services/analysis/background/background_analysis_scheduler.dart';
import 'package:anytime/services/analysis/episode_analysis_dto.dart';
import 'package:anytime/services/analysis/episode_analysis_service.dart';
import 'package:anytime/services/audio/audio_player_service.dart';
import 'package:anytime/services/podcast/podcast_service.dart';
import 'package:anytime/services/transcription/episode_transcription_service.dart';
import 'package:anytime/state/ad_skip_state.dart';
import 'package:anytime/state/episode_state.dart';
import 'package:anytime/state/library_state.dart';
import 'package:anytime/state/queue_event_state.dart';
import 'package:anytime/state/transcript_state_event.dart';
import 'package:anytime/ui/podcast/mini_player.dart';
import 'package:anytime/ui/podcast/now_playing.dart';
import 'package:anytime/ui/themes.dart';
import 'package:flutter/material.dart';
import 'package:flutter_localizations/flutter_localizations.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:provider/provider.dart';
import 'package:rxdart/rxdart.dart';

import '../unit/mocks/mock_notification_service.dart';
import '../unit/mocks/mock_settings_service.dart';

Episode _episode() => Episode(
      guid: 'ep-mini',
      pguid: 'pod-1',
      podcast: 'Mini Podcast',
      title: 'Mini Episode',
      author: 'Mini Author',
      imageUrl: '',
      contentUrl: 'https://cdn.example.com/mini.mp3',
      duration: 600,
    );

void main() {
  group('MiniPlayer', () {
    Widget tree({required AudioBloc audioBloc, required SettingsBloc settingsBloc}) {
      return MaterialApp(
        theme: Themes.lightTheme().themeData,
        localizationsDelegates: const [
          AnytimeLocalisationsDelegate(),
          GlobalMaterialLocalizations.delegate,
          GlobalWidgetsLocalizations.delegate,
          GlobalCupertinoLocalizations.delegate,
        ],
        supportedLocales: const [Locale('en')],
        home: Scaffold(
          body: MultiProvider(
            providers: [
              Provider<AudioBloc>.value(value: audioBloc),
              Provider<SettingsBloc>.value(value: settingsBloc),
            ],
            child: const MiniPlayer(),
          ),
        ),
      );
    }

    testWidgets('builds while playing and shows the title, AI cue, frosted card and artwork',
        (tester) async {
      final audio = _MiniAudio(playing: AudioState.playing, episode: _episode());
      final audioBloc = AudioBloc(audioPlayerService: audio);
      final settingsBloc = SettingsBloc(
        settingsService: MockSettingsService(),
        notificationService: MockNotificationService(),
        backgroundAnalysisScheduler: const NoopBackgroundAnalysisScheduler(),
      );
      addTearDown(() {
        audioBloc.dispose();
        settingsBloc.dispose();
        audio.dispose();
      });

      await tester.pumpWidget(tree(audioBloc: audioBloc, settingsBloc: settingsBloc));
      await tester.pumpAndSettle();

      expect(find.text('Mini Episode'), findsOneWidget);
      // AI ad-skip is enabled by default (prompt mode) → teal dot cue.
      expect(find.text('AI ad-skip on'), findsOneWidget);
      // Frosted card uses BackdropFilter.
      expect(find.byType(BackdropFilter), findsWidgets);
      // Artwork placeholder (empty url → asset placeholder image).
      expect(find.byType(Image), findsWidgets);
    });

    testWidgets('falls back to the author when ad-skip is disabled', (tester) async {
      final audio = _MiniAudio(playing: AudioState.playing, episode: _episode());
      final audioBloc = AudioBloc(audioPlayerService: audio);
      final settingsService = MockSettingsService()..adSkipMode = AdSkipMode.disabled;
      final settingsBloc = SettingsBloc(
        settingsService: settingsService,
        notificationService: MockNotificationService(),
        backgroundAnalysisScheduler: const NoopBackgroundAnalysisScheduler(),
      );
      addTearDown(() {
        audioBloc.dispose();
        settingsBloc.dispose();
        audio.dispose();
      });

      await tester.pumpWidget(tree(audioBloc: audioBloc, settingsBloc: settingsBloc));
      await tester.pumpAndSettle();

      expect(find.text('AI ad-skip on'), findsNothing);
      expect(find.text('Mini Author'), findsOneWidget);
    });

    testWidgets('renders nothing when playback is stopped', (tester) async {
      final audio = _MiniAudio(playing: AudioState.stopped, episode: _episode());
      final audioBloc = AudioBloc(audioPlayerService: audio);
      final settingsBloc = SettingsBloc(
        settingsService: MockSettingsService(),
        notificationService: MockNotificationService(),
        backgroundAnalysisScheduler: const NoopBackgroundAnalysisScheduler(),
      );
      addTearDown(() {
        audioBloc.dispose();
        settingsBloc.dispose();
        audio.dispose();
      });

      await tester.pumpWidget(tree(audioBloc: audioBloc, settingsBloc: settingsBloc));
      await tester.pumpAndSettle();

      expect(find.text('Mini Episode'), findsNothing);
    });
  });

  group('NowPlaying smoke', () {
    testWidgets('builds and shows the artwork card, frosted control card and NOW PLAYING header',
        (tester) async {
      final episode = _episode();
      final audio = _MiniAudio(playing: AudioState.playing, episode: episode);
      final audioBloc = AudioBloc(audioPlayerService: audio);
      final repository = _MiniRepository();
      final podcastService = _MiniPodcastService(repository: repository);
      final queueBloc = QueueBloc(audioPlayerService: audio, podcastService: podcastService);
      final episodeBloc = EpisodeBloc(
        podcastService: podcastService,
        audioPlayerService: audio,
        analysisService: _MiniAnalysisService(),
        settingsService: MockSettingsService(),
        transcriptionService: _MiniTranscriptionService(),
        analysisPollInterval: Duration.zero,
      );
      final settingsBloc = SettingsBloc(
        settingsService: MockSettingsService(),
        notificationService: MockNotificationService(),
        backgroundAnalysisScheduler: const NoopBackgroundAnalysisScheduler(),
      );
      addTearDown(() {
        audioBloc.dispose();
        queueBloc.dispose();
        episodeBloc.dispose();
        settingsBloc.dispose();
        audio.dispose();
      });

      await tester.pumpWidget(
        MaterialApp(
          theme: Themes.lightTheme().themeData,
          localizationsDelegates: const [
            AnytimeLocalisationsDelegate(),
            GlobalMaterialLocalizations.delegate,
            GlobalWidgetsLocalizations.delegate,
            GlobalCupertinoLocalizations.delegate,
          ],
          supportedLocales: const [Locale('en')],
          home: MultiProvider(
            providers: [
              Provider<AudioBloc>.value(value: audioBloc),
              Provider<QueueBloc>.value(value: queueBloc),
              Provider<EpisodeBloc>.value(value: episodeBloc),
              Provider<SettingsBloc>.value(value: settingsBloc),
            ],
            // Stub the transport controls so the smoke test stays focused on the
            // Ambient player surfaces (artwork / frosted card / AI cue).
            child: PlayerControlsBuilder(
              builder: (_) => (context) => const SizedBox(height: 48, child: Text('stub-transport')),
              child: const NowPlaying(),
            ),
          ),
        ),
      );
      // QueueBloc autosave debounce flush.
      await tester.pump(const Duration(seconds: 2));
      await tester.pumpAndSettle();

      expect(find.text('NOW PLAYING'), findsOneWidget);
      expect(find.byType(NowPlayingArtworkCard), findsOneWidget);
      expect(find.byType(FrostedControlCard), findsOneWidget);
      expect(find.byType(BackdropFilter), findsWidgets);
      expect(find.text('stub-transport'), findsOneWidget);
    });
  });
}

class _MiniAudio implements AudioPlayerService {
  _MiniAudio({required AudioState playing, Episode? episode})
      : _playing = BehaviorSubject<AudioState>.seeded(playing),
        _episode = BehaviorSubject<Episode?>.seeded(episode);

  final BehaviorSubject<AudioState> _playing;
  final BehaviorSubject<Episode?> _episode;
  final BehaviorSubject<PositionState> _position = BehaviorSubject<PositionState>.seeded(
    PositionState(position: Duration.zero, length: const Duration(seconds: 1), percentage: 0),
  );
  final BehaviorSubject<QueueListState> _queue = BehaviorSubject<QueueListState>.seeded(
    QueueListState(playing: null, queue: const <Episode>[]),
  );
  final BehaviorSubject<TranscriptState> _transcript =
      BehaviorSubject<TranscriptState>.seeded(TranscriptUnavailableState());
  final StreamController<AdSkipState> _adSkip = StreamController<AdSkipState>.broadcast();

  void dispose() {
    _playing.close();
    _episode.close();
    _position.close();
    _queue.close();
    _transcript.close();
    _adSkip.close();
  }

  @override
  Episode? nowPlaying;

  @override
  Stream<AudioState>? get playingState => _playing.stream;

  @override
  ValueStream<Episode?>? get episodeEvent => _episode.stream;

  @override
  ValueStream<PositionState>? get playPosition => _position.stream;

  @override
  Stream<QueueListState>? get queueState => _queue.stream;

  @override
  Stream<TranscriptState>? get transcriptEvent => _transcript.stream;

  @override
  Stream<AdSkipState>? get adSkipEvent => _adSkip.stream;

  @override
  Stream<int>? get playbackError => const Stream<int>.empty();

  @override
  Stream<Sleep>? get sleepStream => const Stream<Sleep>.empty();

  @override
  noSuchMethod(Invocation invocation) => super.noSuchMethod(invocation);
}

class _MiniPodcastService implements PodcastService {
  _MiniPodcastService({required this.repository});

  @override
  final Repository repository;

  @override
  Stream<Podcast?> get podcastListener => const Stream<Podcast?>.empty();

  @override
  Stream<EpisodeState> get episodeListener => const Stream<EpisodeState>.empty();

  @override
  Stream<LibraryState> get libraryListener => const Stream<LibraryState>.empty();

  @override
  Future<Episode> saveEpisode(Episode episode) async => episode;

  @override
  Future<void> saveQueue(List<Episode> episodes) async {}

  @override
  Future<List<Episode>> loadQueue() async => <Episode>[];

  @override
  noSuchMethod(Invocation invocation) => super.noSuchMethod(invocation);
}

class _MiniRepository implements Repository {
  @override
  Future<Episode?> findEpisodeByGuid(String guid) async => null;

  @override
  Stream<EpisodeState> get episodeListener => const Stream<EpisodeState>.empty();

  @override
  Stream<Podcast> get podcastListener => const Stream<Podcast>.empty();

  @override
  noSuchMethod(Invocation invocation) => super.noSuchMethod(invocation);
}

class _MiniAnalysisService implements EpisodeAnalysisService {
  @override
  Future<EpisodeAnalysisSubmitResponse> submit({
    required Episode episode,
    bool force = false,
    EpisodeAnalysisTranscriptPayload? transcript,
  }) async =>
      EpisodeAnalysisSubmitResponse(jobId: 'job-1', status: EpisodeAnalysisJobStatus.queued);

  @override
  Future<EpisodeAnalysisStatusResponse> poll({required String jobId}) async =>
      EpisodeAnalysisStatusResponse(
        jobId: jobId,
        status: EpisodeAnalysisJobStatus.completed,
        adSegments: const [],
      );

  @override
  void close() {}
}

class _MiniTranscriptionService implements EpisodeTranscriptionService {
  @override
  Future<Transcript> transcribeDownloadedEpisode({
    required Episode episode,
    void Function(EpisodeTranscriptionProgress progress)? onProgress,
  }) async =>
      Transcript(subtitles: const <Subtitle>[]);
}
