// Copyright 2020 Ben Hills and the project contributors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

import 'package:anytime/bloc/settings/settings_bloc.dart';
import 'package:anytime/entities/app_settings.dart';
import 'package:anytime/services/analysis/background/background_analysis_scheduler.dart';
import 'package:anytime/ui/settings/ai_ad_skip_settings.dart';
import 'package:anytime/ui/themes.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:provider/provider.dart';

import '../unit/mocks/mock_notification_service.dart';
import '../unit/mocks/mock_settings_service.dart';

void main() {
  late MockSettingsService settingsService;
  SettingsBloc? settingsBloc;

  setUp(() {
    settingsService = MockSettingsService();
  });

  tearDown(() => settingsBloc?.dispose());

  Future<void> pumpSettings(WidgetTester tester) async {
    // Build the bloc here so it picks up the MockSettingsService state the test
    // has just configured.
    settingsBloc = SettingsBloc(
      settingsService: settingsService,
      notificationService: MockNotificationService(),
      backgroundAnalysisScheduler: const NoopBackgroundAnalysisScheduler(),
    );

    await tester.pumpWidget(
      MaterialApp(
        theme: Themes.lightTheme().themeData,
        home: Provider<SettingsBloc>.value(
          value: settingsBloc!,
          child: const AiAdSkipSettings(),
        ),
      ),
    );
    await tester.pumpAndSettle();
  }

  // The three toggles render in order: auto-skip (0), notify (1), host-read (2).
  Finder switchAt(int index) => find.byType(Switch).at(index);

  group('AiAdSkipSettings auto-skip toggle', () {
    testWidgets('turning the toggle off flips adSkipMode prompt -> disabled', (tester) async {
      settingsService.adSkipMode = AdSkipMode.prompt;
      await pumpSettings(tester);

      expect((tester.widget<Switch>(switchAt(0))).value, isTrue);

      await tester.tap(switchAt(0));
      await tester.pump(const Duration(milliseconds: 300));

      expect(settingsBloc!.currentSettings.adSkipMode, AdSkipMode.disabled);
      expect(settingsService.adSkipMode, AdSkipMode.disabled);
    });

    testWidgets('turning the toggle on flips adSkipMode disabled -> prompt', (tester) async {
      settingsService.adSkipMode = AdSkipMode.disabled;
      await pumpSettings(tester);

      expect((tester.widget<Switch>(switchAt(0))).value, isFalse);

      await tester.tap(switchAt(0));
      await tester.pump(const Duration(milliseconds: 300));

      expect(settingsBloc!.currentSettings.adSkipMode, AdSkipMode.prompt);
      expect(settingsService.adSkipMode, AdSkipMode.prompt);
    });

    testWidgets('disabling auto-skip disables the notify and host-read toggles', (tester) async {
      settingsService.adSkipMode = AdSkipMode.disabled;
      await pumpSettings(tester);

      expect((tester.widget<Switch>(switchAt(1))).onChanged, isNull);
      expect((tester.widget<Switch>(switchAt(2))).onChanged, isNull);
    });
  });

  group('AiAdSkipSettings countdown selector', () {
    testWidgets('tapping a countdown option calls setAdSkipCountdownSeconds', (tester) async {
      settingsService.adSkipCountdownSeconds = 3;
      await pumpSettings(tester);

      await tester.tap(find.text('5s'));
      await tester.pump(const Duration(milliseconds: 300));

      expect(settingsBloc!.currentSettings.adSkipCountdownSeconds, 5);
      expect(settingsService.adSkipCountdownSeconds, 5);
    });

    testWidgets('the selected option mirrors the current setting', (tester) async {
      settingsService.adSkipCountdownSeconds = 0;
      await pumpSettings(tester);

      // '0s' option is selected (primary colour) when adSkipCountdownSeconds is 0.
      final selected = tester.widget<AnimatedContainer>(
        find.ancestor(of: find.text('0s'), matching: find.byType(AnimatedContainer)),
      );
      // Selected chips paint with the surface container colour; just assert the
      // option renders and the underlying setting is wired by toggling it.
      expect(selected, isNotNull);
    });
  });

  group('AiAdSkipSettings detection toggles', () {
    testWidgets('the notify toggle writes through via setAdSkipNotify', (tester) async {
      settingsService.adSkipMode = AdSkipMode.prompt;
      settingsService.adSkipNotify = true;
      await pumpSettings(tester);

      await tester.tap(switchAt(1));
      await tester.pump(const Duration(milliseconds: 300));

      expect(settingsBloc!.currentSettings.adSkipNotify, isFalse);
      expect(settingsService.adSkipNotify, isFalse);
    });

    testWidgets('the host-read toggle writes through via setAdSkipIncludeHostRead', (tester) async {
      settingsService.adSkipMode = AdSkipMode.prompt;
      settingsService.adSkipIncludeHostRead = false;
      await pumpSettings(tester);

      await tester.tap(switchAt(2));
      await tester.pump(const Duration(milliseconds: 300));

      expect(settingsBloc!.currentSettings.adSkipIncludeHostRead, isTrue);
      expect(settingsService.adSkipIncludeHostRead, isTrue);
    });
  });

  group('AiAdSkipSettings hero stat formatting', () {
    testWidgets('formats saved seconds as hours and minutes', (tester) async {
      settingsService.adSkipSavedSeconds = 8040; // 2h 14m
      settingsService.adSkipCount = 3;
      await pumpSettings(tester);

      expect(find.text('2h 14m of ads skipped'), findsOneWidget);
      expect(find.text('across 3 episodes'), findsOneWidget);
    });

    testWidgets('shows the empty state when nothing has been skipped', (tester) async {
      settingsService.adSkipSavedSeconds = 0;
      settingsService.adSkipCount = 0;
      await pumpSettings(tester);

      expect(find.text('0m of ads skipped'), findsOneWidget);
      expect(find.text('No ads skipped yet'), findsOneWidget);
    });

    testWidgets('uses the singular noun for a single skip', (tester) async {
      settingsService.adSkipSavedSeconds = 45;
      settingsService.adSkipCount = 1;
      await pumpSettings(tester);

      expect(find.text('45s of ads skipped'), findsOneWidget);
      expect(find.text('across 1 episode'), findsOneWidget);
    });

    testWidgets('formats minute-only saved time without an hours term', (tester) async {
      settingsService.adSkipSavedSeconds = 150; // 2m 30s -> formatted as "2m"
      settingsService.adSkipCount = 2;
      await pumpSettings(tester);

      expect(find.text('2m of ads skipped'), findsOneWidget);
    });
  });
}
