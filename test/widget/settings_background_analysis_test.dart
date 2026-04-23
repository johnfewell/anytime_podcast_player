// Copyright 2020 Ben Hills and the project contributors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

import 'dart:async';

import 'package:anytime/bloc/settings/settings_bloc.dart';
import 'package:anytime/entities/app_settings.dart';
import 'package:anytime/l10n/L.dart';
import 'package:anytime/services/analysis/background/background_analysis_scheduler.dart';
import 'package:anytime/services/analysis/background/model_download_service.dart';
import 'package:anytime/services/secrets/secure_secrets_service.dart';
import 'package:anytime/ui/settings/settings.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_localizations/flutter_localizations.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:provider/provider.dart';

import '../unit/mocks/mock_notification_service.dart';
import '../unit/mocks/mock_settings_service.dart';

const _hfTokenTitle = 'HuggingFace token';
const _hfTokenSet = '•••••••• (set)';
const _downloadGemmaModel = 'Download Gemma model';
const _gemmaModelInstalled = 'Gemma model installed';
const _downloadFailed = 'Download failed';
const _gemmaTaskFilename = 'gemma.task';

void main() {
  late _FakeSecretsService secrets;
  late _FakeGemmaDownloadService gemma;
  late MockSettingsService settingsService;
  late SettingsBloc settingsBloc;

  setUp(() {
    debugBackgroundAnalysisSupportedOverride = () => true;
    secrets = _FakeSecretsService();
    gemma = _FakeGemmaDownloadService();
    settingsService = MockSettingsService()
      ..backgroundAnalysisEnabled = true
      ..backgroundAnalysisDiskCostAccepted = true
      ..backgroundLocalModel = BackgroundAnalysisLocalModel.gemma4E2B;
    settingsBloc = SettingsBloc(
      settingsService: settingsService,
      notificationService: MockNotificationService(),
      backgroundAnalysisScheduler: const NoopBackgroundAnalysisScheduler(),
    );
  });

  tearDown(() {
    debugBackgroundAnalysisSupportedOverride = null;
    settingsBloc.dispose();
  });

  Future<void> pumpSettings(WidgetTester tester) async {
    // Render the full settings tree on-screen so every tile built by the
    // ListView is realized and tappable, not lazy-built off-viewport.
    tester.view.physicalSize = const Size(1200, 6000);
    tester.view.devicePixelRatio = 1.0;
    addTearDown(() {
      tester.view.resetPhysicalSize();
      tester.view.resetDevicePixelRatio();
    });
    await tester.pumpWidget(
      MaterialApp(
        theme: ThemeData(
          appBarTheme: const AppBarTheme(
            systemOverlayStyle: SystemUiOverlayStyle.dark,
          ),
        ),
        localizationsDelegates: const [
          AnytimeLocalisationsDelegate(),
          GlobalMaterialLocalizations.delegate,
          GlobalWidgetsLocalizations.delegate,
          GlobalCupertinoLocalizations.delegate,
        ],
        supportedLocales: const [Locale('en')],
        home: MultiProvider(
          providers: [
            Provider<SettingsBloc>.value(value: settingsBloc),
            Provider<SecureSecretsService>.value(value: secrets),
            Provider<GemmaModelDownloadService>.value(value: gemma),
          ],
          child: const Settings(),
        ),
      ),
    );
    // FutureBuilders need at least one extra pump to settle.
    await tester.pumpAndSettle();
  }

  group('HuggingFace token tile', () {
    testWidgets('shows Loading subtitle while the token Future resolves', (tester) async {
      final completer = Completer<String?>();
      secrets.readFuture = completer.future;

      tester.view.physicalSize = const Size(1200, 6000);
      tester.view.devicePixelRatio = 1.0;
      addTearDown(() {
        tester.view.resetPhysicalSize();
        tester.view.resetDevicePixelRatio();
      });

      await tester.pumpWidget(
        MaterialApp(
          theme: ThemeData(
            appBarTheme: const AppBarTheme(
              systemOverlayStyle: SystemUiOverlayStyle.dark,
            ),
          ),
          localizationsDelegates: const [
            AnytimeLocalisationsDelegate(),
            GlobalMaterialLocalizations.delegate,
            GlobalWidgetsLocalizations.delegate,
            GlobalCupertinoLocalizations.delegate,
          ],
          supportedLocales: const [Locale('en')],
          home: MultiProvider(
            providers: [
              Provider<SettingsBloc>.value(value: settingsBloc),
              Provider<SecureSecretsService>.value(value: secrets),
              Provider<GemmaModelDownloadService>.value(value: gemma),
            ],
            child: const Settings(),
          ),
        ),
      );
      await tester.pump();

      expect(find.text('Loading…'), findsOneWidget);

      completer.complete('hf_real_token');
      await tester.pumpAndSettle();
    });

    testWidgets('renders the set-token subtitle when a token exists', (tester) async {
      secrets.values[huggingFaceAccessTokenSecret] = 'hf_existing';
      await pumpSettings(tester);

      expect(find.text(_hfTokenSet), findsOneWidget);
    });

    testWidgets('renders the empty subtitle when no token is stored', (tester) async {
      await pumpSettings(tester);

      expect(find.text('Optional — required only for gated model files'), findsOneWidget);
    });

    testWidgets('renders an error subtitle when the stored token cannot be read', (tester) async {
      secrets.readError = StateError('secure storage locked');
      await pumpSettings(tester);

      expect(find.text('Unable to read stored token — tap to retry'), findsOneWidget);
    });

    testWidgets('tapping the tile when storage is locked shows a snackbar, not the dialog', (tester) async {
      secrets.readError = StateError('secure storage locked');
      await pumpSettings(tester);

      await tester.tap(find.text(_hfTokenTitle));
      await tester.pump();

      expect(find.text('HuggingFace access token'), findsNothing);
      expect(
        find.text('Could not access secure storage to load the HuggingFace token.'),
        findsOneWidget,
      );
    });

    testWidgets('Save writes the token and refreshes the subtitle', (tester) async {
      await pumpSettings(tester);

      await tester.tap(find.text(_hfTokenTitle));
      await tester.pumpAndSettle();

      await tester.enterText(find.byType(TextField), 'hf_new_token');
      await tester.tap(find.text('Save'));
      await tester.pumpAndSettle();

      expect(secrets.values[huggingFaceAccessTokenSecret], 'hf_new_token');
      expect(find.text(_hfTokenSet), findsOneWidget);
    });

    testWidgets('Saving an empty value deletes the stored token', (tester) async {
      secrets.values[huggingFaceAccessTokenSecret] = 'hf_old';
      await pumpSettings(tester);

      await tester.tap(find.text(_hfTokenTitle));
      await tester.pumpAndSettle();

      // Empty input in the dialog's TextField.
      await tester.enterText(find.byType(TextField), '   ');
      await tester.tap(find.text('Save'));
      await tester.pumpAndSettle();

      expect(secrets.values.containsKey(huggingFaceAccessTokenSecret), isFalse);
    });

    testWidgets('Clear removes the token and refreshes the subtitle', (tester) async {
      secrets.values[huggingFaceAccessTokenSecret] = 'hf_existing';
      await pumpSettings(tester);

      await tester.tap(find.text(_hfTokenTitle));
      await tester.pumpAndSettle();

      await tester.tap(find.text('Clear'));
      await tester.pumpAndSettle();

      expect(secrets.values.containsKey(huggingFaceAccessTokenSecret), isFalse);
      expect(find.text(_hfTokenSet), findsNothing);
    });
  });

  group('Gemma install tile', () {
    testWidgets('renders the download call-to-action when model is not installed', (tester) async {
      gemma.installed = false;
      await pumpSettings(tester);

      expect(find.text(_downloadGemmaModel), findsOneWidget);
    });

    testWidgets('renders the installed state when model is already present', (tester) async {
      gemma.installed = true;
      await pumpSettings(tester);

      expect(find.text(_gemmaModelInstalled), findsOneWidget);
    });

    testWidgets('tapping the download tile starts a download and streams progress', (tester) async {
      gemma.installed = false;
      await pumpSettings(tester);

      await tester.tap(find.text(_downloadGemmaModel));
      await tester.pump();

      expect(gemma.downloadStartCount, 1);
      expect(gemma.lastHuggingFaceToken, isNull);

      gemma.emitProgress(const GemmaDownloadProgress(percent: 42, filename: _gemmaTaskFilename));
      await tester.pump();

      expect(find.textContaining('Downloading Gemma model (42%)'), findsOneWidget);
      expect(find.text(_gemmaTaskFilename), findsOneWidget);

      // Complete the download; tile should flip to installed state.
      gemma.completeDownload();
      await tester.pumpAndSettle();

      expect(find.text(_gemmaModelInstalled), findsOneWidget);
    });

    testWidgets('passes the stored HuggingFace token when starting a download', (tester) async {
      secrets.values[huggingFaceAccessTokenSecret] = 'hf_live';
      gemma.installed = false;
      await pumpSettings(tester);

      await tester.tap(find.text(_downloadGemmaModel));
      await tester.pump();

      expect(gemma.lastHuggingFaceToken, 'hf_live');
    });

    testWidgets('surfaces a stream error and offers a retry tap', (tester) async {
      gemma.installed = false;
      await pumpSettings(tester);

      await tester.tap(find.text(_downloadGemmaModel));
      await tester.pump();

      gemma.emitError('network down');
      await tester.pumpAndSettle();

      expect(find.text(_downloadFailed), findsOneWidget);
      expect(find.textContaining('Tap to retry'), findsOneWidget);

      // Retry — should kick off another download.
      await tester.tap(find.text(_downloadFailed));
      await tester.pump();
      expect(gemma.downloadStartCount, 2);
    });

    testWidgets('surfaces a token-read error without starting the download', (tester) async {
      secrets.readError = StateError('keyring locked');
      gemma.installed = false;
      await pumpSettings(tester);

      await tester.tap(find.text(_downloadGemmaModel));
      await tester.pumpAndSettle();

      expect(gemma.downloadStartCount, 0);
      expect(find.text(_downloadFailed), findsOneWidget);
    });

    testWidgets('cancel tile deletes the model and resets state', (tester) async {
      gemma.installed = false;
      await pumpSettings(tester);

      await tester.tap(find.text(_downloadGemmaModel));
      await tester.pumpAndSettle();
      gemma.emitProgress(const GemmaDownloadProgress(percent: 10, filename: _gemmaTaskFilename));
      await tester.pumpAndSettle();

      final downloadingTile = find.textContaining('Downloading Gemma model').first;
      final inkWell = find.ancestor(
        of: downloadingTile,
        matching: find.byType(InkWell),
      );
      await tester.tap(inkWell);
      // The broadcast-stream subscription's cancel Future resolves on the
      // outer zone, so flush it with runAsync before the assertions.
      await tester.runAsync(() async {
        await Future<void>.delayed(const Duration(milliseconds: 10));
      });
      await tester.pumpAndSettle();

      expect(gemma.deleteCalls, 1);
      expect(find.text(_downloadGemmaModel), findsOneWidget);
    });

    testWidgets('re-download confirmation dialog kicks off a new download', (tester) async {
      gemma.installed = true;
      await pumpSettings(tester);

      await tester.tap(find.text(_gemmaModelInstalled));
      await tester.pumpAndSettle();

      expect(find.text('Re-download model?'), findsOneWidget);

      await tester.tap(find.text('Re-download'));
      await tester.pumpAndSettle();

      expect(gemma.deleteCalls, 1);
      expect(gemma.downloadStartCount, 1);
    });

    testWidgets('re-download dialog can be cancelled without side effects', (tester) async {
      gemma.installed = true;
      await pumpSettings(tester);

      await tester.tap(find.text(_gemmaModelInstalled));
      await tester.pumpAndSettle();

      await tester.tap(find.text('Cancel'));
      await tester.pumpAndSettle();

      expect(gemma.deleteCalls, 0);
      expect(gemma.downloadStartCount, 0);
    });
  });
}

class _FakeSecretsService implements SecureSecretsService {
  final Map<String, String> values = <String, String>{};
  Future<String?>? readFuture;
  Object? readError;

  @override
  Future<String?> read(String key) {
    if (readError != null) {
      return Future<String?>.error(readError!);
    }
    if (readFuture != null) return readFuture!;
    return Future<String?>.value(values[key]);
  }

  @override
  Future<void> write({required String key, required String value}) async {
    values[key] = value;
  }

  @override
  Future<void> delete(String key) async {
    values.remove(key);
  }
}

class _FakeGemmaDownloadService implements GemmaModelDownloadService {
  bool installed = false;
  int downloadStartCount = 0;
  int deleteCalls = 0;
  String? lastHuggingFaceToken;
  StreamController<GemmaDownloadProgress>? _controller;

  @override
  Future<bool> isInstalled(BackgroundAnalysisLocalModel variant) async => installed;

  @override
  Stream<GemmaDownloadProgress> download(
    BackgroundAnalysisLocalModel variant, {
    String? huggingFaceToken,
  }) {
    downloadStartCount++;
    lastHuggingFaceToken = huggingFaceToken;
    _controller = StreamController<GemmaDownloadProgress>.broadcast();
    return _controller!.stream;
  }

  void emitProgress(GemmaDownloadProgress progress) {
    _controller?.add(progress);
  }

  void emitError(Object error) {
    _controller?.addError(error);
  }

  void completeDownload() {
    _controller?.close();
    _controller = null;
    installed = true;
  }

  @override
  Future<String?> resolveLocalPath(BackgroundAnalysisLocalModel variant) async => null;

  @override
  Future<void> delete(BackgroundAnalysisLocalModel variant) async {
    deleteCalls++;
    installed = false;
  }
}
