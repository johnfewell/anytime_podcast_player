// Copyright 2020 Ben Hills and the project contributors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

import 'dart:async';

import 'package:anytime/bloc/podcast/audio_bloc.dart';
import 'package:anytime/bloc/podcast/episode_bloc.dart';
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
import 'package:anytime/services/podcast/podcast_service.dart';
import 'package:anytime/services/transcription/episode_transcription_service.dart';
import 'package:anytime/state/episode_state.dart';
import 'package:anytime/state/library_state.dart';
import 'package:anytime/state/queue_event_state.dart';
import 'package:anytime/ui/podcast/episode_details.dart';
import 'package:anytime/ui/themes.dart';
import 'package:flutter/material.dart';
import 'package:flutter_localizations/flutter_localizations.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:provider/provider.dart';
import 'package:rxdart/rxdart.dart';

import '../unit/mocks/mock_notification_service.dart';
import '../unit/mocks/mock_settings_service.dart';

Episode _episode({List<AdSegment> adSegments = const <AdSegment>[]}) => Episode(
      guid: 'ep-details',
      pguid: 'pod-1',
      podcast: 'Test Podcast',
      title: 'Details Episode',
      description: 'A short description.',
      contentUrl: 'https://cdn.example.com/details.mp3',
      imageUrl: '',
      duration: 600,
    )..adSegments = adSegments;

void main() {
  group('EpisodeDetails ads-ahead chip', () {
    testWidgets('renders "N ads · DURATION" + "will be skipped" when adSegments present', (tester) async {
      final harness = await _DetailsHarness.create();
      addTearDown(harness.dispose);

      final episode = _episode(adSegments: const <AdSegment>[
        AdSegment(startMs: 1000, endMs: 4000),
      ]);

      await tester.pumpWidget(harness.treeFor(episode));
      // QueueBloc debounce flush.
      await tester.pump(const Duration(seconds: 2));
      await tester.pumpAndSettle();

      expect(find.text('1 ad · 3s'), findsOneWidget);
      expect(find.text('will be skipped'), findsOneWidget);
    });

    testWidgets('formats a multi-minute duration with padded seconds', (tester) async {
      final harness = await _DetailsHarness.create();
      addTearDown(harness.dispose);

      final episode = _episode(adSegments: const <AdSegment>[
        AdSegment(startMs: 0, endMs: 65000),
      ]);

      await tester.pumpWidget(harness.treeFor(episode));
      await tester.pump(const Duration(seconds: 2));
      await tester.pumpAndSettle();

      expect(find.text('1 ad · 1m 05s'), findsOneWidget);
    });

    testWidgets('pluralises to "ads" for multiple segments', (tester) async {
      final harness = await _DetailsHarness.create();
      addTearDown(harness.dispose);

      final episode = _episode(adSegments: const <AdSegment>[
        AdSegment(startMs: 1000, endMs: 4000),
        AdSegment(startMs: 10000, endMs: 16000),
      ]);

      await tester.pumpWidget(harness.treeFor(episode));
      await tester.pump(const Duration(seconds: 2));
      await tester.pumpAndSettle();

      // 3s + 6s = 9s total.
      expect(find.text('2 ads · 9s'), findsOneWidget);
      expect(find.text('will be skipped'), findsOneWidget);
    });

    testWidgets('hides the chip when there are no ad segments', (tester) async {
      final harness = await _DetailsHarness.create();
      addTearDown(harness.dispose);

      await tester.pumpWidget(harness.treeFor(_episode()));
      await tester.pump(const Duration(seconds: 2));
      await tester.pumpAndSettle();

      expect(find.textContaining('will be skipped'), findsNothing);
    });
  });
}

class _DetailsHarness {
  _DetailsHarness(this.audio);

  final _DetailsAudioPlayerService audio;
  late final AudioBloc audioBloc;
  late final EpisodeBloc episodeBloc;
  late final QueueBloc queueBloc;
  late final SettingsBloc settingsBloc;

  static Future<_DetailsHarness> create() async {
    final harness = _DetailsHarness(_DetailsAudioPlayerService());
    final repository = _DetailsRepository();
    final podcastService = _DetailsPodcastService(repository: repository);
    final settingsService = MockSettingsService();
    final notificationService = MockNotificationService();

    harness.audioBloc = AudioBloc(audioPlayerService: harness.audio);
    harness.queueBloc = QueueBloc(audioPlayerService: harness.audio, podcastService: podcastService);
    harness.episodeBloc = EpisodeBloc(
      podcastService: podcastService,
      audioPlayerService: harness.audio,
      analysisService: _DetailsAnalysisService(),
      settingsService: settingsService,
      transcriptionService: _DetailsTranscriptionService(),
      analysisPollInterval: Duration.zero,
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
      builder: (context, child) => MultiProvider(
        providers: [
          Provider<AudioBloc>.value(value: audioBloc),
          Provider<EpisodeBloc>.value(value: episodeBloc),
          Provider<QueueBloc>.value(value: queueBloc),
          Provider<SettingsBloc>.value(value: settingsBloc),
        ],
        child: child,
      ),
      home: Scaffold(
        body: EpisodeDetails(episode: episode),
      ),
    );
  }

  void dispose() {
    audioBloc.dispose();
    queueBloc.dispose();
    episodeBloc.dispose();
    settingsBloc.dispose();
  }
}

class _DetailsAudioPlayerService implements AudioPlayerService {
  _DetailsAudioPlayerService() {
    playingState = _playingStateController.stream;
    episodeEvent = _episodeController;
  }

  final BehaviorSubject<QueueListState> _queueState = BehaviorSubject<QueueListState>.seeded(
    QueueListState(playing: null, queue: const <Episode>[]),
  );
  final BehaviorSubject<AudioState> _playingStateController =
      BehaviorSubject<AudioState>.seeded(AudioState.stopped);
  final BehaviorSubject<Episode?> _episodeController = BehaviorSubject<Episode?>.seeded(null);

  @override
  Episode? nowPlaying;

  @override
  Stream<QueueListState>? get queueState => _queueState.stream;

  @override
  Stream<AudioState>? playingState;

  @override
  ValueStream<Episode?>? episodeEvent;

  @override
  noSuchMethod(Invocation invocation) => super.noSuchMethod(invocation);
}

class _DetailsPodcastService implements PodcastService {
  _DetailsPodcastService({required this.repository});

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

class _DetailsRepository implements Repository {
  @override
  Future<Episode?> findEpisodeByGuid(String guid) async => null;

  @override
  Stream<EpisodeState> get episodeListener => const Stream<EpisodeState>.empty();

  @override
  Stream<Podcast> get podcastListener => const Stream<Podcast>.empty();

  @override
  noSuchMethod(Invocation invocation) => super.noSuchMethod(invocation);
}

class _DetailsAnalysisService implements EpisodeAnalysisService {
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
        adSegments: const <AdSegment>[],
      );

  @override
  void close() {}
}

class _DetailsTranscriptionService implements EpisodeTranscriptionService {
  @override
  Future<Transcript> transcribeDownloadedEpisode({
    required Episode episode,
    void Function(EpisodeTranscriptionProgress progress)? onProgress,
  }) async =>
      Transcript(subtitles: const <Subtitle>[]);
}
