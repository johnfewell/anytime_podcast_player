// Copyright 2020 Ben Hills and the project contributors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

import 'package:anytime/entities/app_settings.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  group('AppSettings.sensibleDefaults ad-skip fields', () {
    final settings = AppSettings.sensibleDefaults();

    test('adSkipCountdownSeconds defaults to 3', () {
      expect(settings.adSkipCountdownSeconds, 3);
    });

    test('adSkipNotify defaults to true', () {
      expect(settings.adSkipNotify, isTrue);
    });

    test('adSkipIncludeHostRead defaults to false', () {
      expect(settings.adSkipIncludeHostRead, isFalse);
    });

    test('adSkipSavedSeconds defaults to 0', () {
      expect(settings.adSkipSavedSeconds, 0);
    });

    test('adSkipCount defaults to 0', () {
      expect(settings.adSkipCount, 0);
    });

    test('adSkipMode defaults to prompt', () {
      expect(settings.adSkipMode, AdSkipMode.prompt);
    });
  });

  group('AppSettings.copyWith ad-skip fields', () {
    final base = AppSettings.sensibleDefaults();

    test('updates adSkipCountdownSeconds and preserves the rest', () {
      final updated = base.copyWith(adSkipCountdownSeconds: 5);

      expect(updated.adSkipCountdownSeconds, 5);
      expect(updated.adSkipNotify, base.adSkipNotify);
      expect(updated.adSkipIncludeHostRead, base.adSkipIncludeHostRead);
      expect(updated.adSkipSavedSeconds, base.adSkipSavedSeconds);
      expect(updated.adSkipCount, base.adSkipCount);
      expect(updated.adSkipMode, base.adSkipMode);
    });

    test('updates adSkipNotify', () {
      final updated = base.copyWith(adSkipNotify: false);

      expect(updated.adSkipNotify, isFalse);
      expect(base.adSkipNotify, isTrue);
    });

    test('updates adSkipIncludeHostRead', () {
      final updated = base.copyWith(adSkipIncludeHostRead: true);

      expect(updated.adSkipIncludeHostRead, isTrue);
      expect(base.adSkipIncludeHostRead, isFalse);
    });

    test('updates adSkipSavedSeconds', () {
      final updated = base.copyWith(adSkipSavedSeconds: 8040);

      expect(updated.adSkipSavedSeconds, 8040);
    });

    test('updates adSkipCount', () {
      final updated = base.copyWith(adSkipCount: 12);

      expect(updated.adSkipCount, 12);
    });

    test('round-trips every ad-skip field back to its starting value', () {
      final mutated = base.copyWith(
        adSkipCountdownSeconds: 9,
        adSkipNotify: false,
        adSkipIncludeHostRead: true,
        adSkipSavedSeconds: 250,
        adSkipCount: 7,
      );

      final restored = mutated.copyWith(
        adSkipCountdownSeconds: base.adSkipCountdownSeconds,
        adSkipNotify: base.adSkipNotify,
        adSkipIncludeHostRead: base.adSkipIncludeHostRead,
        adSkipSavedSeconds: base.adSkipSavedSeconds,
        adSkipCount: base.adSkipCount,
      );

      expect(restored.adSkipCountdownSeconds, base.adSkipCountdownSeconds);
      expect(restored.adSkipNotify, base.adSkipNotify);
      expect(restored.adSkipIncludeHostRead, base.adSkipIncludeHostRead);
      expect(restored.adSkipSavedSeconds, base.adSkipSavedSeconds);
      expect(restored.adSkipCount, base.adSkipCount);
    });

    test('copyWith without arguments returns an equal ad-skip configuration', () {
      final copy = base.copyWith();

      expect(copy.adSkipCountdownSeconds, base.adSkipCountdownSeconds);
      expect(copy.adSkipNotify, base.adSkipNotify);
      expect(copy.adSkipIncludeHostRead, base.adSkipIncludeHostRead);
      expect(copy.adSkipSavedSeconds, base.adSkipSavedSeconds);
      expect(copy.adSkipCount, base.adSkipCount);
      expect(copy.adSkipMode, base.adSkipMode);
    });
  });
}
