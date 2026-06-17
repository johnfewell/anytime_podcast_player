// Copyright 2020 Ben Hills and the project contributors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
import 'package:anytime/entities/app_settings.dart';
import 'package:anytime/services/settings/mobile_settings_service.dart';
import 'package:anytime/services/settings/settings_service.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:shared_preferences/shared_preferences.dart';

/// Covers persistence (get/set + defaults when unset) and the settings stream
/// notification for the five new AI ad-skip settings on [MobileSettingsService].
void main() {
  const int timeout = 500;
  final Map<String, Object> settings = <String, Object>{'dummy': 1};
  SettingsService? mobileSettingsService;
  late Stream<String>? settingsListener;

  TestWidgetsFlutterBinding.ensureInitialized();

  setUp(() async {
    SharedPreferences.setMockInitialValues(settings);
    mobileSettingsService = await MobileSettingsService.instance();
    settingsListener = mobileSettingsService?.settingsListener;
  });

  test('Test ad skip countdown seconds default then set', () async {
    expect(mobileSettingsService?.adSkipCountdownSeconds, 3);
    expectLater(settingsListener, emits('adSkipCountdownSeconds'));
    mobileSettingsService?.adSkipCountdownSeconds = 5;
    expect(mobileSettingsService?.adSkipCountdownSeconds, 5);
  }, timeout: const Timeout(Duration(milliseconds: timeout)));

  test('Test ad skip notify default then set', () async {
    expect(mobileSettingsService?.adSkipNotify, true);
    expectLater(settingsListener, emits('adSkipNotify'));
    mobileSettingsService?.adSkipNotify = false;
    expect(mobileSettingsService?.adSkipNotify, false);
  }, timeout: const Timeout(Duration(milliseconds: timeout)));

  test('Test ad skip include host read default then set', () async {
    expect(mobileSettingsService?.adSkipIncludeHostRead, false);
    expectLater(settingsListener, emits('adSkipIncludeHostRead'));
    mobileSettingsService?.adSkipIncludeHostRead = true;
    expect(mobileSettingsService?.adSkipIncludeHostRead, true);
  }, timeout: const Timeout(Duration(milliseconds: timeout)));

  test('Test ad skip saved seconds default then set', () async {
    expect(mobileSettingsService?.adSkipSavedSeconds, 0);
    expectLater(settingsListener, emits('adSkipSavedSeconds'));
    mobileSettingsService?.adSkipSavedSeconds = 8040;
    expect(mobileSettingsService?.adSkipSavedSeconds, 8040);
  }, timeout: const Timeout(Duration(milliseconds: timeout)));

  test('Test ad skip count default then set', () async {
    expect(mobileSettingsService?.adSkipCount, 0);
    expectLater(settingsListener, emits('adSkipCount'));
    mobileSettingsService?.adSkipCount = 3;
    expect(mobileSettingsService?.adSkipCount, 3);
  }, timeout: const Timeout(Duration(milliseconds: timeout)));

  test('ad-skip settings survive a fresh instance reading SharedPreferences', () async {
    mobileSettingsService?.adSkipCountdownSeconds = 7;
    mobileSettingsService?.adSkipNotify = false;
    mobileSettingsService?.adSkipIncludeHostRead = true;
    mobileSettingsService?.adSkipSavedSeconds = 4242;
    mobileSettingsService?.adSkipCount = 9;

    // Drop the in-memory singleton and rebuild from the same SharedPreferences.
    MobileSettingsService.resetInstanceForTesting();
    final restored = await MobileSettingsService.instance();
    addTearDown(MobileSettingsService.resetInstanceForTesting);

    expect(restored?.adSkipCountdownSeconds, 7);
    expect(restored?.adSkipNotify, isFalse);
    expect(restored?.adSkipIncludeHostRead, isTrue);
    expect(restored?.adSkipSavedSeconds, 4242);
    expect(restored?.adSkipCount, 9);
  }, timeout: const Timeout(Duration(milliseconds: timeout)));

  test('defaults match AppSettings.sensibleDefaults for the ad-skip fields', () {
    expect(mobileSettingsService?.adSkipMode, AdSkipMode.prompt);
    expect(mobileSettingsService?.adSkipCountdownSeconds, 3);
    expect(mobileSettingsService?.adSkipNotify, isTrue);
    expect(mobileSettingsService?.adSkipIncludeHostRead, isFalse);
    expect(mobileSettingsService?.adSkipSavedSeconds, 0);
    expect(mobileSettingsService?.adSkipCount, 0);
  }, timeout: const Timeout(Duration(milliseconds: timeout)));
}
