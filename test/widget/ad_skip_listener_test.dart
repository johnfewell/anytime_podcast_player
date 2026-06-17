// Copyright 2020 Ben Hills and the project contributors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

import 'dart:async';

import 'package:anytime/bloc/podcast/audio_bloc.dart';
import 'package:anytime/bloc/settings/settings_bloc.dart';
import 'package:anytime/entities/ad_segment.dart';
import 'package:anytime/entities/episode.dart';
import 'package:anytime/services/analysis/background/background_analysis_scheduler.dart';
import 'package:anytime/services/audio/audio_player_service.dart';
import 'package:anytime/state/ad_skip_state.dart';
import 'package:anytime/ui/podcast/ad_skip_listener.dart';
import 'package:anytime/ui/themes.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:provider/provider.dart';

import '../unit/mocks/mock_notification_service.dart';
import '../unit/mocks/mock_settings_service.dart';

/// Extended coverage for [AdSkipListener]. The single happy-path case already
/// lives in `ai_ad_skip_flow_test.dart`; these tests cover the countdown
/// auto-skip, the keep-playing dismissal, the cleared-while-prompted path, and
/// the confirmation-toast behaviour.
void main() {
  late _AdSkipAudioPlayerService audioService;

  setUp(() {
    audioService = _AdSkipAudioPlayerService();
  });

  Widget buildTree({
    required AudioBloc audioBloc,
    SettingsBloc? settingsBloc,
  }) {
    return MaterialApp(
      theme: Themes.lightTheme().themeData,
      home: MultiProvider(
        providers: [
          Provider<AudioBloc>.value(value: audioBloc),
          if (settingsBloc != null) Provider<SettingsBloc>.value(value: settingsBloc),
        ],
        child: const AdSkipListener(
          child: Scaffold(body: SizedBox.shrink()),
        ),
      ),
    );
  }

  Future<void> emitPrompt(WidgetTester tester, {int countdownSeconds = 3}) async {
    final episode = Episode(
      guid: 'ep-skip',
      podcast: 'Podcast',
      title: 'Skip Episode',
      contentUrl: 'https://cdn.example.com/skip.mp3',
    );
    const segment = AdSegment(startMs: 1000, endMs: 4000);

    audioService.emitAdSkipEvent(AdSkipPromptState(episode: episode, segment: segment));
    await tester.pump();
    await tester.pump(const Duration(milliseconds: 200));
  }

  testWidgets('shows the Ambient ad card with the expected copy', (tester) async {
    final audioBloc = AudioBloc(audioPlayerService: audioService);
    addTearDown(() {
      audioBloc.dispose();
      audioService.dispose();
    });

    await tester.pumpWidget(buildTree(audioBloc: audioBloc));
    await emitPrompt(tester);

    expect(find.text('Ad coming up'), findsOneWidget);
    expect(find.text('AI DETECTED'), findsOneWidget);
    expect(find.text('Skip ad'), findsOneWidget);
    expect(find.text('Keep playing'), findsOneWidget);
    expect(find.textContaining('Sponsor break'), findsOneWidget);
  });

  testWidgets('tapping "Skip ad" invokes audioPlayerService.skipActiveAd', (tester) async {
    final audioBloc = AudioBloc(audioPlayerService: audioService);
    addTearDown(() {
      audioBloc.dispose();
      audioService.dispose();
    });

    await tester.pumpWidget(buildTree(audioBloc: audioBloc));
    await emitPrompt(tester);

    await tester.tap(find.text('Skip ad'));
    await tester.pump();

    expect(audioService.skipActiveAdCallCount, 1);
  });

  testWidgets('tapping "Keep playing" dismisses the card without skipping', (tester) async {
    final audioBloc = AudioBloc(audioPlayerService: audioService);
    addTearDown(() {
      audioBloc.dispose();
      audioService.dispose();
    });

    await tester.pumpWidget(buildTree(audioBloc: audioBloc));
    await emitPrompt(tester);
    expect(find.text('Ad coming up'), findsOneWidget);

    await tester.tap(find.text('Keep playing'));
    await tester.pump();
    await tester.pump(const Duration(milliseconds: 100));

    expect(audioService.skipActiveAdCallCount, 0);
    expect(find.text('Ad coming up'), findsNothing);
  });

  testWidgets('AdSkipClearedState dismisses a visible card', (tester) async {
    final audioBloc = AudioBloc(audioPlayerService: audioService);
    addTearDown(() {
      audioBloc.dispose();
      audioService.dispose();
    });

    await tester.pumpWidget(buildTree(audioBloc: audioBloc));
    await emitPrompt(tester);
    expect(find.text('Ad coming up'), findsOneWidget);

    final episode = Episode(
      guid: 'ep-skip',
      podcast: 'Podcast',
      title: 'Skip Episode',
      contentUrl: 'https://cdn.example.com/skip.mp3',
    );
    const segment = AdSegment(startMs: 1000, endMs: 4000);
    audioService.emitAdSkipEvent(AdSkipClearedState(episode: episode, segment: segment));
    await tester.pump();
    await tester.pump(const Duration(milliseconds: 100));

    expect(find.text('Ad coming up'), findsNothing);
    expect(audioService.skipActiveAdCallCount, 0);
  });

  testWidgets('the countdown auto-skips after the configured seconds', (tester) async {
    final settingsService = MockSettingsService()..adSkipCountdownSeconds = 1;
    final settingsBloc = SettingsBloc(
      settingsService: settingsService,
      notificationService: MockNotificationService(),
      backgroundAnalysisScheduler: const NoopBackgroundAnalysisScheduler(),
    );
    final audioBloc = AudioBloc(audioPlayerService: audioService);
    addTearDown(() {
      settingsBloc.dispose();
      audioBloc.dispose();
      audioService.dispose();
    });

    await tester.pumpWidget(buildTree(audioBloc: audioBloc, settingsBloc: settingsBloc));
    await emitPrompt(tester, countdownSeconds: 1);

    expect(audioService.skipActiveAdCallCount, 0);

    // One countdown tick at 1s triggers the auto-skip.
    await tester.pump(const Duration(seconds: 1));
    await tester.pump(const Duration(milliseconds: 100));

    expect(audioService.skipActiveAdCallCount, 1);
    expect(find.text('Ad coming up'), findsNothing);
  });

  testWidgets('the "Skipped a N-sec ad break" toast appears when notify is on', (tester) async {
    final settingsService = MockSettingsService()..adSkipNotify = true;
    final settingsBloc = SettingsBloc(
      settingsService: settingsService,
      notificationService: MockNotificationService(),
      backgroundAnalysisScheduler: const NoopBackgroundAnalysisScheduler(),
    );
    final audioBloc = AudioBloc(audioPlayerService: audioService);
    addTearDown(() {
      settingsBloc.dispose();
      audioBloc.dispose();
      audioService.dispose();
    });

    await tester.pumpWidget(buildTree(audioBloc: audioBloc, settingsBloc: settingsBloc));
    await emitPrompt(tester);

    await tester.tap(find.text('Skip ad'));
    await tester.pump();
    await tester.pump(const Duration(milliseconds: 100));

    // Segment is 1000ms-4000ms = 3 seconds.
    expect(find.text('Skipped a 3-sec ad break'), findsOneWidget);
  });

  testWidgets('the toast is suppressed when notify is off', (tester) async {
    final settingsService = MockSettingsService()..adSkipNotify = false;
    final settingsBloc = SettingsBloc(
      settingsService: settingsService,
      notificationService: MockNotificationService(),
      backgroundAnalysisScheduler: const NoopBackgroundAnalysisScheduler(),
    );
    final audioBloc = AudioBloc(audioPlayerService: audioService);
    addTearDown(() {
      settingsBloc.dispose();
      audioBloc.dispose();
      audioService.dispose();
    });

    await tester.pumpWidget(buildTree(audioBloc: audioBloc, settingsBloc: settingsBloc));
    await emitPrompt(tester);

    await tester.tap(find.text('Skip ad'));
    await tester.pump();
    await tester.pump(const Duration(milliseconds: 100));

    expect(audioService.skipActiveAdCallCount, 1);
    expect(find.textContaining('Skipped a'), findsNothing);
  });

  testWidgets('without a SettingsBloc the card still uses the notify default', (tester) async {
    final audioBloc = AudioBloc(audioPlayerService: audioService);
    addTearDown(() {
      audioBloc.dispose();
      audioService.dispose();
    });

    await tester.pumpWidget(buildTree(audioBloc: audioBloc));
    await emitPrompt(tester);

    await tester.tap(find.text('Skip ad'));
    await tester.pump();
    await tester.pump(const Duration(milliseconds: 100));

    expect(audioService.skipActiveAdCallCount, 1);
    // Default notify is true, so the toast surfaces.
    expect(find.text('Skipped a 3-sec ad break'), findsOneWidget);
  });
}

class _AdSkipAudioPlayerService implements AudioPlayerService {
  final _adSkipController = StreamController<AdSkipState>.broadcast();

  int skipActiveAdCallCount = 0;

  void emitAdSkipEvent(AdSkipState event) => _adSkipController.add(event);

  void dispose() => _adSkipController.close();

  @override
  Future<void> skipActiveAd() async {
    skipActiveAdCallCount++;
  }

  @override
  Stream<AdSkipState>? get adSkipEvent => _adSkipController.stream;

  @override
  Episode? nowPlaying;

  @override
  noSuchMethod(Invocation invocation) => super.noSuchMethod(invocation);
}
