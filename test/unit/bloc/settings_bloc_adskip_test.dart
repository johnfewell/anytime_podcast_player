// Copyright 2020 Ben Hills and the project contributors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

import 'package:anytime/bloc/settings/settings_bloc.dart';
import 'package:anytime/services/analysis/background/background_analysis_scheduler.dart';
import 'package:flutter_test/flutter_test.dart';

import '../mocks/mock_notification_service.dart';
import '../mocks/mock_settings_service.dart';

void main() {
  late MockSettingsService settingsService;
  late MockNotificationService notificationService;

  setUp(() {
    settingsService = MockSettingsService();
    notificationService = MockNotificationService();
  });

  SettingsBloc buildBloc() => SettingsBloc(
        settingsService: settingsService,
        notificationService: notificationService,
        backgroundAnalysisScheduler: const NoopBackgroundAnalysisScheduler(),
      );

  group('SettingsBloc ad-skip setters', () {
    test('setAdSkipCountdownSeconds updates the settings stream and writes through', () async {
      final bloc = buildBloc();
      addTearDown(bloc.dispose);
      await Future<void>.delayed(Duration.zero);

      final emitted = <int>[];
      final subscription = bloc.settings.map((s) => s.adSkipCountdownSeconds).listen(emitted.add);
      addTearDown(subscription.cancel);

      bloc.setAdSkipCountdownSeconds(5);
      await Future<void>.delayed(Duration.zero);

      expect(bloc.currentSettings.adSkipCountdownSeconds, 5);
      expect(settingsService.adSkipCountdownSeconds, 5);
      expect(emitted, contains(5));
    });

    test('setAdSkipNotify updates the settings stream and writes through', () async {
      final bloc = buildBloc();
      addTearDown(bloc.dispose);
      await Future<void>.delayed(Duration.zero);

      expect(bloc.currentSettings.adSkipNotify, isTrue);

      bloc.setAdSkipNotify(false);
      await Future<void>.delayed(Duration.zero);

      expect(bloc.currentSettings.adSkipNotify, isFalse);
      expect(settingsService.adSkipNotify, isFalse);
    });

    test('setAdSkipIncludeHostRead updates the settings stream and writes through', () async {
      final bloc = buildBloc();
      addTearDown(bloc.dispose);
      await Future<void>.delayed(Duration.zero);

      expect(bloc.currentSettings.adSkipIncludeHostRead, isFalse);

      bloc.setAdSkipIncludeHostRead(true);
      await Future<void>.delayed(Duration.zero);

      expect(bloc.currentSettings.adSkipIncludeHostRead, isTrue);
      expect(settingsService.adSkipIncludeHostRead, isTrue);
    });

    test('ad-skip setters only mutate their own field on the emitted AppSettings', () async {
      final bloc = buildBloc();
      addTearDown(bloc.dispose);
      await Future<void>.delayed(Duration.zero);

      bloc.setAdSkipCountdownSeconds(8);
      bloc.setAdSkipNotify(false);
      bloc.setAdSkipIncludeHostRead(true);
      await Future<void>.delayed(Duration.zero);

      final settings = bloc.currentSettings;
      expect(settings.adSkipCountdownSeconds, 8);
      expect(settings.adSkipNotify, isFalse);
      expect(settings.adSkipIncludeHostRead, isTrue);
      // Untouched ad-skip fields keep their loaded values.
      expect(settings.adSkipSavedSeconds, settingsService.adSkipSavedSeconds);
      expect(settings.adSkipCount, settingsService.adSkipCount);
      expect(settings.adSkipMode, settingsService.adSkipMode);
    });
  });

  group('SettingsBloc live ad-skip stats', () {
    // The audio backend writes the running "time saved" / "ads skipped" totals
    // straight to the SettingsService, bypassing the bloc. The bloc must pick
    // these up via the settings listener so the AI ad-skip screen's hero stat
    // updates within the session rather than only after a restart.
    test('re-emits when adSkipSavedSeconds changes outside the bloc', () async {
      final bloc = buildBloc();
      addTearDown(bloc.dispose);
      await Future<void>.delayed(Duration.zero);

      final emitted = <int>[];
      final subscription = bloc.settings.map((s) => s.adSkipSavedSeconds).listen(emitted.add);
      addTearDown(subscription.cancel);

      // Simulate recordAdSkipStat accumulating a skipped break.
      settingsService.adSkipSavedSeconds = 42;
      await Future<void>.delayed(Duration.zero);

      expect(bloc.currentSettings.adSkipSavedSeconds, 42);
      expect(emitted, contains(42));
    });

    test('re-emits when adSkipCount changes outside the bloc', () async {
      final bloc = buildBloc();
      addTearDown(bloc.dispose);
      await Future<void>.delayed(Duration.zero);

      final emitted = <int>[];
      final subscription = bloc.settings.map((s) => s.adSkipCount).listen(emitted.add);
      addTearDown(subscription.cancel);

      settingsService.adSkipCount = 3;
      await Future<void>.delayed(Duration.zero);

      expect(bloc.currentSettings.adSkipCount, 3);
      expect(emitted, contains(3));
    });
  });
}
