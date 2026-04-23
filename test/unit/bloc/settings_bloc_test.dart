// Copyright 2020 Ben Hills and the project contributors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

import 'package:anytime/bloc/settings/settings_bloc.dart';
import 'package:anytime/entities/app_settings.dart';
import 'package:anytime/services/analysis/background/background_analysis_scheduler.dart';
import 'package:flutter_test/flutter_test.dart';

import '../mocks/mock_notification_service.dart';
import '../mocks/mock_settings_service.dart';

class _RecordingScheduler implements BackgroundAnalysisScheduler {
  int scheduleCalls = 0;
  int cancelCalls = 0;
  bool throwOnSchedule = false;
  bool throwOnCancel = false;

  @override
  Future<void> schedule() async {
    scheduleCalls++;
    if (throwOnSchedule) throw StateError('schedule failed');
  }

  @override
  Future<void> cancel() async {
    cancelCalls++;
    if (throwOnCancel) throw StateError('cancel failed');
  }

  @override
  Future<bool> isScheduled() async => false;
}

void main() {
  late MockSettingsService settingsService;
  late MockNotificationService notificationService;
  late _RecordingScheduler scheduler;

  setUp(() {
    settingsService = MockSettingsService();
    notificationService = MockNotificationService();
    scheduler = _RecordingScheduler();
  });

  SettingsBloc buildBloc() => SettingsBloc(
        settingsService: settingsService,
        notificationService: notificationService,
        backgroundAnalysisScheduler: scheduler,
      );

  group('SettingsBloc background analysis scheduler reconciliation', () {
    test('cancels schedule on startup when backgroundAnalysisEnabled is false', () async {
      settingsService.backgroundAnalysisEnabled = false;

      final bloc = buildBloc();
      addTearDown(bloc.dispose);

      // Reconcile runs asynchronously after construction.
      await Future<void>.delayed(Duration.zero);

      expect(scheduler.cancelCalls, 1);
      expect(scheduler.scheduleCalls, 0);
    });

    test('schedules on startup when backgroundAnalysisEnabled is true', () async {
      settingsService.backgroundAnalysisEnabled = true;

      final bloc = buildBloc();
      addTearDown(bloc.dispose);

      await Future<void>.delayed(Duration.zero);

      expect(scheduler.scheduleCalls, 1);
      expect(scheduler.cancelCalls, 0);
    });

    test('swallows scheduler errors during startup reconciliation', () async {
      settingsService.backgroundAnalysisEnabled = true;
      scheduler.throwOnSchedule = true;

      final bloc = buildBloc();
      addTearDown(bloc.dispose);

      await Future<void>.delayed(Duration.zero);

      // Construction should not propagate the error.
      expect(scheduler.scheduleCalls, 1);
    });

    test('defaults to NoopBackgroundAnalysisScheduler when none is injected', () async {
      settingsService.backgroundAnalysisEnabled = true;

      final bloc = SettingsBloc(
        settingsService: settingsService,
        notificationService: notificationService,
      );
      addTearDown(bloc.dispose);

      await Future<void>.delayed(Duration.zero);

      expect(bloc.backgroundAnalysisScheduler, isA<NoopBackgroundAnalysisScheduler>());
    });
  });

  group('SettingsBloc background analysis setters', () {
    test('toggling backgroundAnalysisEnabled updates settings and runs scheduler', () async {
      settingsService.backgroundAnalysisEnabled = false;
      final bloc = buildBloc();
      addTearDown(bloc.dispose);
      await Future<void>.delayed(Duration.zero);
      final startupCancels = scheduler.cancelCalls;

      bloc.setBackgroundAnalysisEnabled(true);
      await Future<void>.delayed(Duration.zero);

      expect(bloc.currentSettings.backgroundAnalysisEnabled, isTrue);
      expect(settingsService.backgroundAnalysisEnabled, isTrue);
      expect(scheduler.scheduleCalls, 1);

      bloc.setBackgroundAnalysisEnabled(false);
      await Future<void>.delayed(Duration.zero);

      expect(bloc.currentSettings.backgroundAnalysisEnabled, isFalse);
      expect(settingsService.backgroundAnalysisEnabled, isFalse);
      expect(scheduler.cancelCalls, startupCancels + 1);
    });

    test('scheduler errors from setBackgroundAnalysisEnabled do not propagate', () async {
      final bloc = buildBloc();
      addTearDown(bloc.dispose);
      await Future<void>.delayed(Duration.zero);

      scheduler.throwOnSchedule = true;
      bloc.setBackgroundAnalysisEnabled(true);
      await Future<void>.delayed(Duration.zero);

      expect(bloc.currentSettings.backgroundAnalysisEnabled, isTrue);
    });

    test('setBackgroundLocalModel updates settings and persists', () async {
      final bloc = buildBloc();
      addTearDown(bloc.dispose);
      await Future<void>.delayed(Duration.zero);

      bloc.setBackgroundLocalModel(BackgroundAnalysisLocalModel.gemma4E4B);
      await Future<void>.delayed(Duration.zero);

      expect(bloc.currentSettings.backgroundLocalModel, BackgroundAnalysisLocalModel.gemma4E4B);
      expect(settingsService.backgroundLocalModel, BackgroundAnalysisLocalModel.gemma4E4B);
    });

    test('setBackgroundAnalysisDiskCostAccepted updates settings and persists', () async {
      final bloc = buildBloc();
      addTearDown(bloc.dispose);
      await Future<void>.delayed(Duration.zero);

      bloc.setBackgroundAnalysisDiskCostAccepted(true);
      await Future<void>.delayed(Duration.zero);

      expect(bloc.currentSettings.backgroundAnalysisDiskCostAccepted, isTrue);
      expect(settingsService.backgroundAnalysisDiskCostAccepted, isTrue);
    });

    test('setOnDemandAnalysisEnabled updates settings and persists', () async {
      final bloc = buildBloc();
      addTearDown(bloc.dispose);
      await Future<void>.delayed(Duration.zero);

      bloc.setOnDemandAnalysisEnabled(false);
      await Future<void>.delayed(Duration.zero);

      expect(bloc.currentSettings.onDemandAnalysisEnabled, isFalse);
      expect(settingsService.onDemandAnalysisEnabled, isFalse);
    });

    test('setShowAnalysisHistory updates settings and persists', () async {
      final bloc = buildBloc();
      addTearDown(bloc.dispose);
      await Future<void>.delayed(Duration.zero);

      bloc.setShowAnalysisHistory(true);
      await Future<void>.delayed(Duration.zero);

      expect(bloc.currentSettings.showAnalysisHistory, isTrue);
      expect(settingsService.showAnalysisHistory, isTrue);
    });
  });

  test('initial currentSettings reflect SettingsService values', () async {
    settingsService.backgroundAnalysisEnabled = true;
    settingsService.backgroundLocalModel = BackgroundAnalysisLocalModel.gemma4E4B;
    settingsService.backgroundAnalysisDiskCostAccepted = true;
    settingsService.onDemandAnalysisEnabled = false;
    settingsService.showAnalysisHistory = true;

    final bloc = buildBloc();
    addTearDown(bloc.dispose);
    await Future<void>.delayed(Duration.zero);

    expect(bloc.currentSettings.backgroundAnalysisEnabled, isTrue);
    expect(bloc.currentSettings.backgroundLocalModel, BackgroundAnalysisLocalModel.gemma4E4B);
    expect(bloc.currentSettings.backgroundAnalysisDiskCostAccepted, isTrue);
    expect(bloc.currentSettings.onDemandAnalysisEnabled, isFalse);
    expect(bloc.currentSettings.showAnalysisHistory, isTrue);
  });
}
