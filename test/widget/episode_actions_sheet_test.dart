// Copyright 2020 Ben Hills and the project contributors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

import 'dart:async';

import 'package:anytime/bloc/podcast/audio_bloc.dart';
import 'package:anytime/bloc/podcast/episode_bloc.dart';
import 'package:anytime/bloc/podcast/podcast_bloc.dart';
import 'package:anytime/bloc/podcast/queue_bloc.dart';
import 'package:anytime/bloc/settings/settings_bloc.dart';
import 'package:anytime/entities/ad_segment.dart';
import 'package:anytime/entities/episode.dart';
import 'package:anytime/entities/podcast.dart';
import 'package:anytime/entities/transcript.dart';
import 'package:anytime/l10n/L.dart';
import 'package:anytime/repository/repository.dart';
import 'package:anytime/services/analysis/background/background_analysis_scheduler.dart';
import 'package:anytime/services/analysis/episode_analysis_dto.dart';
import 'package:anytime/services/analysis/episode_analysis_service.dart';
import 'package:anytime/services/audio/audio_player_service.dart';
import 'package:anytime/services/download/download_manager.dart';
import 'package:anytime/services/download/download_service.dart';
import 'package:anytime/services/download/mobile_download_service.dart';
import 'package:anytime/services/podcast/podcast_service.dart';
import 'package:anytime/services/transcription/episode_transcription_service.dart';
import 'package:anytime/state/episode_state.dart';
import 'package:anytime/state/library_state.dart';
import 'package:anytime/state/queue_event_state.dart';
import 'package:anytime/ui/podcast/episode_actions_sheet.dart';
import 'package:anytime/ui/themes.dart';
import 'package:flutter/material.dart';
import 'package:flutter_localizations/flutter_localizations.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:provider/provider.dart';
import 'package:rxdart/rxdart.dart';

import '../unit/mocks/mock_notification_service.dart';
import '../unit/mocks/mock_settings_service.dart';

Episode _episode({List<AdSegment> adSegments = const <AdSegment>[]}) => Episode(
      guid: 'ep-sheet',
      pguid: 'pod-1',
      podcast: 'Test Podcast',
      title: 'Sheet Episode',
      contentUrl: 'https://cdn.example.com/sheet.mp3',
      duration: 600,
    )..adSegments = adSegments;

void main() {
  // PodcastBloc.dispose closes this static subject; reset it before each test
  // so repeated builds don't close an already-closed stream.
  setUp(() {
    MobileDownloadService.downloadProgress = BehaviorSubject<DownloadProgress>();
  });

  group('EpisodeActionsSheet', () {
    testWidgets('renders the action rows', (tester) async {
      final harness = await _SheetHarness.create();
      addTearDown(harness.dispose);

      await tester.pumpWidget(harness.treeFor(_episode()));
      await tester.pumpAndSettle();
      await _openSheet(tester);

      expect(find.text('Play next'), findsOneWidget);
      expect(find.text(L.of(tester.element(find.byType(Scaffold)))!.semantics_add_to_queue),
          findsOneWidget);
      expect(find.byIcon(Icons.play_circle_outline_rounded), findsOneWidget);
      expect(find.byIcon(Icons.playlist_play_rounded), findsOneWidget);
      expect(find.byIcon(Icons.playlist_add_rounded), findsOneWidget);
      expect(find.byIcon(Icons.check_circle_outline_rounded), findsOneWidget);
      expect(find.byIcon(Icons.ios_share_rounded), findsOneWidget);
    });

    testWidgets('shows the AI ad-skip heads-up when the episode has ad segments', (tester) async {
      final harness = await _SheetHarness.create();
      addTearDown(harness.dispose);

      final episode = _episode(adSegments: const <AdSegment>[
        AdSegment(startMs: 1000, endMs: 4000),
      ]);
      await tester.pumpWidget(harness.treeFor(episode));
      await tester.pumpAndSettle();
      await _openSheet(tester);

      expect(find.text('1 ad · 3s will be skipped'), findsOneWidget);
      expect(find.byIcon(Icons.auto_awesome), findsOneWidget);
    });

    testWidgets('shows a pluralised heads-up for multiple ad segments', (tester) async {
      final harness = await _SheetHarness.create();
      addTearDown(harness.dispose);

      final episode = _episode(adSegments: const <AdSegment>[
        AdSegment(startMs: 1000, endMs: 4000),
        AdSegment(startMs: 10000, endMs: 16000),
      ]);
      await tester.pumpWidget(harness.treeFor(episode));
      await tester.pumpAndSettle();
      await _openSheet(tester);

      // 3s + 6s = 9s total.
      expect(find.text('2 ads · 9s will be skipped'), findsOneWidget);
    });

    testWidgets('hides the AI ad-skip heads-up when there are no ad segments', (tester) async {
      final harness = await _SheetHarness.create();
      addTearDown(harness.dispose);

      await tester.pumpWidget(harness.treeFor(_episode()));
      await tester.pumpAndSettle();
      await _openSheet(tester);

      expect(find.textContaining('will be skipped'), findsNothing);
      // Falls back to the podcast name subtitle.
      expect(find.text('Test Podcast'), findsWidgets);
    });

    testWidgets('tapping "Play next" queues the episode at the front', (tester) async {
      final harness = await _SheetHarness.create();
      addTearDown(harness.dispose);

      await tester.pumpWidget(harness.treeFor(_episode()));
      await tester.pumpAndSettle();
      await _openSheet(tester);

      await tester.tap(find.text('Play next'));
      await tester.pumpAndSettle();

      expect(harness.audio.addUpNextEpisodeCalls, <String>['ep-sheet']);
    });

    testWidgets('tapping the play row plays the episode now', (tester) async {
      final harness = await _SheetHarness.create();
      addTearDown(harness.dispose);

      await tester.pumpWidget(harness.treeFor(_episode()));
      await tester.pumpAndSettle();
      await _openSheet(tester);

      await tester.tap(find.byIcon(Icons.play_circle_outline_rounded));
      await tester.pumpAndSettle();

      expect(harness.audio.playEpisodeCalls, <String>['ep-sheet']);
    });

    testWidgets('tapping "Add to Up next" appends to the queue', (tester) async {
      final harness = await _SheetHarness.create();
      addTearDown(harness.dispose);

      await tester.pumpWidget(harness.treeFor(_episode()));
      await tester.pumpAndSettle();
      await _openSheet(tester);

      await tester.tap(find.byIcon(Icons.playlist_add_rounded));
      await tester.pumpAndSettle();

      expect(harness.audio.addUpNextEpisodeCalls, <String>['ep-sheet']);
    });

    testWidgets('tapping the mark-as-played row toggles played via the bloc', (tester) async {
      final harness = await _SheetHarness.create();
      addTearDown(harness.dispose);

      await tester.pumpWidget(harness.treeFor(_episode()));
      await tester.pumpAndSettle();
      await _openSheet(tester);

      await tester.tap(find.byIcon(Icons.check_circle_outline_rounded));
      await tester.pumpAndSettle();

      expect(harness.podcast.togglePlayedCalls, <String>['ep-sheet']);
    });

    testWidgets('tapping Close dismisses the sheet without an action', (tester) async {
      final harness = await _SheetHarness.create();
      addTearDown(harness.dispose);

      await tester.pumpWidget(harness.treeFor(_episode()));
      await tester.pumpAndSettle();
      await _openSheet(tester);

      await tester.tap(find.text(L.of(tester.element(find.byType(Scaffold)))!.close_button_label));
      await tester.pumpAndSettle();

      expect(find.text('Play next'), findsNothing);
      expect(harness.audio.playEpisodeCalls, isEmpty);
      expect(harness.audio.addUpNextEpisodeCalls, isEmpty);
    });
  });
}

Future<void> _openSheet(WidgetTester tester) async {
  // QueueBloc subscribes to a debounced (2s) queue autosave stream at
  // construction. Flush that timer with virtual time so no Timer is left
  // pending at the end of the test.
  await tester.pump(const Duration(seconds: 2));

  await tester.tap(find.text('open'));
  await tester.pumpAndSettle();
}

class _SheetHarness {
  _SheetHarness(this.audio, this.podcast);

  final _RecordingAudioPlayerService audio;
  final _SheetPodcastService podcast;
  late final EpisodeBloc episodeBloc;
  late final PodcastBloc podcastBloc;
  late final QueueBloc queueBloc;
  late final AudioBloc audioBloc;
  late final SettingsBloc settingsBloc;

  static Future<_SheetHarness> create() async {
    final audio = _RecordingAudioPlayerService();
    final repository = _SheetRepository();
    final podcast = _SheetPodcastService(repository: repository);
    final notificationService = MockNotificationService();
    final settingsService = MockSettingsService();
    final downloadService = _SheetDownloadService();

    final harness = _SheetHarness(audio, podcast);

    harness.audioBloc = AudioBloc(audioPlayerService: audio);
    harness.queueBloc = QueueBloc(audioPlayerService: audio, podcastService: podcast);
    harness.episodeBloc = EpisodeBloc(
      podcastService: podcast,
      audioPlayerService: audio,
      analysisService: _SheetAnalysisService(),
      settingsService: settingsService,
      transcriptionService: _SheetTranscriptionService(),
      analysisPollInterval: Duration.zero,
    );
    harness.podcastBloc = PodcastBloc(
      podcastService: podcast,
      audioPlayerService: audio,
      downloadService: downloadService,
      notificationService: notificationService,
      settingsService: settingsService,
    );
    harness.settingsBloc = SettingsBloc(
      settingsService: settingsService,
      notificationService: notificationService,
      backgroundAnalysisScheduler: const NoopBackgroundAnalysisScheduler(),
    );

    return harness;
  }

  Widget treeFor(Episode episode) {
    return MaterialApp(
      theme: Themes.lightTheme().themeData,
      localizationsDelegates: const [
        AnytimeLocalisationsDelegate(),
        GlobalMaterialLocalizations.delegate,
        GlobalWidgetsLocalizations.delegate,
        GlobalCupertinoLocalizations.delegate,
      ],
      supportedLocales: const [Locale('en')],
      // Providers live above the Navigator so the modal bottom sheet can resolve them.
      builder: (context, child) => MultiProvider(
        providers: [
          Provider<AudioBloc>.value(value: audioBloc),
          Provider<EpisodeBloc>.value(value: episodeBloc),
          Provider<PodcastBloc>.value(value: podcastBloc),
          Provider<QueueBloc>.value(value: queueBloc),
          Provider<SettingsBloc>.value(value: settingsBloc),
        ],
        child: child,
      ),
      home: Scaffold(
        body: Builder(
          builder: (context) => Center(
            child: ElevatedButton(
              onPressed: () => showEpisodeActionsSheet(context, episode),
              child: const Text('open'),
            ),
          ),
        ),
      ),
    );
  }

  Future<void> dispose() async {
    audioBloc.dispose();
    queueBloc.dispose();
    episodeBloc.dispose();
    podcastBloc.dispose();
    settingsBloc.dispose();
  }
}

class _RecordingAudioPlayerService implements AudioPlayerService {
  _RecordingAudioPlayerService();

  final List<String> playEpisodeCalls = <String>[];
  final List<String> addUpNextEpisodeCalls = <String>[];

  final BehaviorSubject<QueueListState> _queueState =
      BehaviorSubject<QueueListState>.seeded(
    QueueListState(playing: null, queue: const <Episode>[]),
  );

  @override
  Future<void> playEpisode({required Episode episode, bool resume = true}) async {
    playEpisodeCalls.add(episode.guid);
  }

  @override
  Future<void> addUpNextEpisode(Episode episode) async {
    addUpNextEpisodeCalls.add(episode.guid);
  }

  @override
  Episode? nowPlaying;

  @override
  Stream<QueueListState>? get queueState => _queueState.stream;

  @override
  noSuchMethod(Invocation invocation) => super.noSuchMethod(invocation);
}

class _SheetPodcastService implements PodcastService {
  _SheetPodcastService({required this.repository});

  @override
  final Repository repository;

  final List<String> togglePlayedCalls = <String>[];

  final StreamController<EpisodeState> _episodeController =
      StreamController<EpisodeState>.broadcast();

  @override
  Stream<Podcast?> get podcastListener => const Stream<Podcast?>.empty();

  @override
  Stream<EpisodeState> get episodeListener => _episodeController.stream;

  @override
  Stream<LibraryState> get libraryListener => const Stream<LibraryState>.empty();

  @override
  Future<void> toggleEpisodePlayed(Episode episode) async {
    togglePlayedCalls.add(episode.guid);
  }

  @override
  Future<List<Episode>> loadDownloads() async => <Episode>[];

  @override
  Future<List<Episode>> loadEpisodes() async => <Episode>[];

  @override
  Future<void> saveQueue(List<Episode> episodes) async {}

  @override
  Future<List<Episode>> loadQueue() async => <Episode>[];

  @override
  Future<Episode> saveEpisode(Episode episode) async => episode;

  @override
  noSuchMethod(Invocation invocation) => super.noSuchMethod(invocation);
}

class _SheetRepository implements Repository {
  @override
  Future<Episode?> findEpisodeByGuid(String guid) async => null;

  @override
  Stream<EpisodeState> get episodeListener => const Stream<EpisodeState>.empty();

  @override
  Stream<Podcast> get podcastListener => const Stream<Podcast>.empty();

  @override
  noSuchMethod(Invocation invocation) => super.noSuchMethod(invocation);
}

class _SheetDownloadService implements DownloadService {
  @override
  Future<bool> downloadEpisode(Episode episode) async => true;

  @override
  Future<Episode?> findEpisodeByTaskId(String taskId) async => null;

  @override
  void dispose() {}
}

class _SheetAnalysisService implements EpisodeAnalysisService {
  @override
  Future<EpisodeAnalysisSubmitResponse> submit({
    required Episode episode,
    bool force = false,
    EpisodeAnalysisTranscriptPayload? transcript,
  }) async =>
      EpisodeAnalysisSubmitResponse(
        jobId: 'job-1',
        status: EpisodeAnalysisJobStatus.queued,
      );

  @override
  Future<EpisodeAnalysisStatusResponse> poll({required String jobId}) async =>
      EpisodeAnalysisStatusResponse(
        jobId: jobId,
        status: EpisodeAnalysisJobStatus.completed,
        adSegments: const <AdSegment>[],
      );

  @override
  void close() {}
}

class _SheetTranscriptionService implements EpisodeTranscriptionService {
  @override
  Future<Transcript> transcribeDownloadedEpisode({
    required Episode episode,
    void Function(EpisodeTranscriptionProgress progress)? onProgress,
  }) async =>
      Transcript(subtitles: const <Subtitle>[]);
}
