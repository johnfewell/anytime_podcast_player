// Copyright 2020 Ben Hills and the project contributors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

import 'package:anytime/ui/themes.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  group('AmbientColors.fromBrightness', () {
    test('returns the expected light tokens', () {
      final light = AmbientColors.fromBrightness(Brightness.light);

      expect(light.aiTeal, const Color(0xff2a9d99));
      expect(light.night, const Color(0xff213183));
      expect(light.onNight, const Color(0xffffffff));
      expect(light.success, const Color(0xff2a9d99));
      expect(light.hairline, const Color(0xffe6e6e6));
      expect(light.frostedSurface, const Color(0xb8ffffff));
      expect(light.frostedBorder, const Color(0xd9ffffff));
    });

    test('returns the expected dark tokens', () {
      final dark = AmbientColors.fromBrightness(Brightness.dark);

      expect(dark.aiTeal, const Color(0xff3fc4b8));
      expect(dark.night, const Color(0xff243a86));
      expect(dark.success, const Color(0xff5fd3a0));
      expect(dark.hairline, const Color(0x12ffffff));
      expect(dark.frostedSurface, const Color(0xa8262a38));
    });

    test('light and dark variants are distinct on every token', () {
      final light = AmbientColors.fromBrightness(Brightness.light);
      final dark = AmbientColors.fromBrightness(Brightness.dark);

      expect(light.aiTeal, isNot(dark.aiTeal));
      expect(light.aiTealSurface, isNot(dark.aiTealSurface));
      expect(light.night, isNot(dark.night));
      expect(light.success, isNot(dark.success));
      expect(light.controlFill, isNot(dark.controlFill));
      expect(light.hairline, isNot(dark.hairline));
      expect(light.frostedSurface, isNot(dark.frostedSurface));
      expect(light.frostedBorder, isNot(dark.frostedBorder));
      // onNight is intentionally white in both variants.
      expect(light.onNight, dark.onNight);
    });
  });

  group('AmbientColors.copyWith', () {
    test('overrides only the supplied tokens', () {
      const base = AmbientColors(
        aiTeal: Color(0xff000000),
        aiTealSurface: Color(0x11000000),
        night: Color(0xff111111),
        onNight: Color(0xff222222),
        success: Color(0xff333333),
        controlFill: Color(0x0d000000),
        hairline: Color(0xff444444),
        frostedSurface: Color(0xb8ffffff),
        frostedBorder: Color(0xd9ffffff),
      );

      final copy = base.copyWith(aiTeal: const Color(0xffffffff), hairline: const Color(0xff999999));

      expect(copy.aiTeal, const Color(0xffffffff));
      expect(copy.hairline, const Color(0xff999999));
      // Untouched tokens are retained.
      expect(copy.aiTealSurface, base.aiTealSurface);
      expect(copy.night, base.night);
      expect(copy.onNight, base.onNight);
      expect(copy.success, base.success);
      expect(copy.controlFill, base.controlFill);
      expect(copy.frostedSurface, base.frostedSurface);
      expect(copy.frostedBorder, base.frostedBorder);
    });
  });

  group('AmbientColors.lerp', () {
    test('lerps every token between light and dark at the midpoint', () {
      final light = AmbientColors.fromBrightness(Brightness.light);
      final dark = AmbientColors.fromBrightness(Brightness.dark);
      const t = 0.5;

      final mid = light.lerp(dark, t);

      expect(mid.aiTeal, Color.lerp(light.aiTeal, dark.aiTeal, t));
      expect(mid.aiTealSurface, Color.lerp(light.aiTealSurface, dark.aiTealSurface, t));
      expect(mid.night, Color.lerp(light.night, dark.night, t));
      expect(mid.onNight, Color.lerp(light.onNight, dark.onNight, t));
      expect(mid.success, Color.lerp(light.success, dark.success, t));
      expect(mid.controlFill, Color.lerp(light.controlFill, dark.controlFill, t));
      expect(mid.hairline, Color.lerp(light.hairline, dark.hairline, t));
      expect(mid.frostedSurface, Color.lerp(light.frostedSurface, dark.frostedSurface, t));
      expect(mid.frostedBorder, Color.lerp(light.frostedBorder, dark.frostedBorder, t));
    });

    test('returns this when the other is null', () {
      final light = AmbientColors.fromBrightness(Brightness.light);

      expect(light.lerp(null, 0.5), same(light));
    });

    test('at t=0 returns the start values', () {
      final light = AmbientColors.fromBrightness(Brightness.light);
      final dark = AmbientColors.fromBrightness(Brightness.dark);

      final result = light.lerp(dark, 0);

      expect(result.aiTeal, light.aiTeal);
      expect(result.night, light.night);
    });
  });

  group('AmbientColors theme registration', () {
    test('is registered on Themes.lightTheme()', () {
      final theme = Themes.lightTheme().themeData;

      expect(theme.extension<AmbientColors>(), isNotNull);
      expect(theme.extension<AmbientColors>()!.aiTeal, AmbientColors.fromBrightness(Brightness.light).aiTeal);
    });

    test('is registered on Themes.darkTheme()', () {
      final theme = Themes.darkTheme().themeData;

      expect(theme.extension<AmbientColors>(), isNotNull);
      expect(theme.extension<AmbientColors>()!.aiTeal, AmbientColors.fromBrightness(Brightness.dark).aiTeal);
    });

    testWidgets('AmbientColors.of(context) resolves the registered extension', (tester) async {
      AmbientColors? resolved;
      final key = GlobalKey();

      await tester.pumpWidget(
        MaterialApp(
          theme: Themes.lightTheme().themeData,
          home: Builder(
            key: key,
            builder: (context) {
              resolved = AmbientColors.of(context);
              return const SizedBox.shrink();
            },
          ),
        ),
      );

      expect(resolved, isNotNull);
      expect(resolved!.aiTeal, Themes.lightTheme().themeData.extension<AmbientColors>()!.aiTeal);
    });

    testWidgets('AmbientColors.of falls back to brightness-based defaults without the extension',
        (tester) async {
      AmbientColors? resolved;
      final key = GlobalKey();

      await tester.pumpWidget(
        MaterialApp(
          theme: ThemeData(brightness: Brightness.dark),
          darkTheme: ThemeData(brightness: Brightness.dark),
          home: Builder(
            key: key,
            builder: (context) {
              resolved = AmbientColors.of(context);
              return const SizedBox.shrink();
            },
          ),
        ),
      );

      expect(resolved!.aiTeal, AmbientColors.fromBrightness(Brightness.dark).aiTeal);
    });
  });
}
