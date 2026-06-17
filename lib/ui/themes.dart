// Copyright 2020 Ben Hills and the project contributors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

// Notion Ambient brand face: Schibsted Grotesk — one humanist grotesque used
// for everything (heavy/tight headlines over calm muted body), matching the
// Player Ambient design. Inter is kept as a metric-compatible fallback.
const String _headlineFontFamily = 'SchibstedGrotesk';
const String _bodyFontFamily = 'SchibstedGrotesk';
const List<String> _fontFallback = <String>['Inter'];

final ThemeData _lightTheme = _buildTheme(
  colorScheme: _lightColorScheme,
  brightness: Brightness.light,
);
final ThemeData _darkTheme = _buildTheme(
  colorScheme: _darkColorScheme,
  brightness: Brightness.dark,
);

// ── Notion Ambient palette ──────────────────────────────────────────────────
// One tightly-scoped system: warm paper canvas, warm near-black ink, exactly one
// structural accent (Notion Blue #0075de) reserved for actions, a deep-indigo
// "night" band (#213183), and teal for the AI/ad-skip cue. See the design's
// tokens/colors.css.
const ColorScheme _lightColorScheme = ColorScheme(
  brightness: Brightness.light,
  primary: Color(0xff0075de), // Notion Blue — the only colour that paints an action
  onPrimary: Color(0xffffffff),
  primaryContainer: Color(0xffd9eafb),
  onPrimaryContainer: Color(0xff004a8d),
  primaryFixed: Color(0xffd9eafb),
  primaryFixedDim: Color(0xffa9cef2),
  onPrimaryFixed: Color(0xff001b3b),
  onPrimaryFixedVariant: Color(0xff005bab),
  secondary: Color(0xff213183), // Deep Indigo — the dark "night" band
  onSecondary: Color(0xffffffff),
  secondaryContainer: Color(0xffdfe2f5),
  onSecondaryContainer: Color(0xff121a4a),
  secondaryFixed: Color(0xffdfe2f5),
  secondaryFixedDim: Color(0xffb6bde6),
  onSecondaryFixed: Color(0xff0a1138),
  onSecondaryFixedVariant: Color(0xff35408f),
  tertiary: Color(0xff2a9d99), // Teal — AI / ad-skip accent
  onTertiary: Color(0xffffffff),
  tertiaryContainer: Color(0xffcdeceb),
  onTertiaryContainer: Color(0xff00413f),
  tertiaryFixed: Color(0xffcdeceb),
  tertiaryFixedDim: Color(0xff8fd0cd),
  onTertiaryFixed: Color(0xff00201f),
  onTertiaryFixedVariant: Color(0xff1f7a76),
  error: Color(0xffcb4a32),
  onError: Color(0xffffffff),
  errorContainer: Color(0xffffdad2),
  onErrorContainer: Color(0xff7a1c0a),
  surface: Color(0xfff6f5f4), // warm paper page canvas
  onSurface: Color(0xff181715), // warm near-black ink (~ink-95)
  onSurfaceVariant: Color(0xff615d59), // Stone — muted copy
  outline: Color(0xffa39e98), // Ash — captions / placeholders
  outlineVariant: Color(0xffe6e6e6), // hairline
  shadow: Color(0xff000000),
  scrim: Color(0xff141a30),
  inverseSurface: Color(0xff1f2a44), // dark toast / snackbar
  onInverseSurface: Color(0xffffffff),
  inversePrimary: Color(0xffa9cef2),
  surfaceTint: Color(0xff0075de),
  surfaceDim: Color(0xffe4e2df),
  surfaceBright: Color(0xffffffff),
  surfaceContainerLowest: Color(0xffffffff),
  surfaceContainerLow: Color(0xffffffff),
  surfaceContainer: Color(0xfff1f0ee),
  surfaceContainerHigh: Color(0xffeceae7),
  surfaceContainerHighest: Color(0xffe7e5e1),
);

// "After dark" — the design's dark variant. Not an inversion: warm near-black
// surfaces (#131318), lifted hairlines instead of shadows, a brighter action
// blue (#3b8bef), and teal (#3fc4b8) carrying the AI cue. Cards sit *lighter*
// than the page so the ambient artwork glow does the work.
const ColorScheme _darkColorScheme = ColorScheme(
  brightness: Brightness.dark,
  primary: Color(0xff3b8bef), // brighter action blue
  onPrimary: Color(0xffffffff),
  primaryContainer: Color(0xff1c2a52),
  onPrimaryContainer: Color(0xffd4e7fb),
  primaryFixed: Color(0xffd9eafb),
  primaryFixedDim: Color(0xffa9cef2),
  onPrimaryFixed: Color(0xff001b3b),
  onPrimaryFixedVariant: Color(0xff005bab),
  secondary: Color(0xff8f9ce0),
  onSecondary: Color(0xff1a2456),
  secondaryContainer: Color(0xff243a86), // night band
  onSecondaryContainer: Color(0xffe3e7fb),
  secondaryFixed: Color(0xffdfe2f5),
  secondaryFixedDim: Color(0xff8f9ce0),
  onSecondaryFixed: Color(0xff0a1138),
  onSecondaryFixedVariant: Color(0xff35408f),
  tertiary: Color(0xff3fc4b8), // teal — AI cue
  onTertiary: Color(0xff00322f),
  tertiaryContainer: Color(0xff134f4a),
  onTertiaryContainer: Color(0xffbff1ec),
  tertiaryFixed: Color(0xffbff1ec),
  tertiaryFixedDim: Color(0xff3fc4b8),
  onTertiaryFixed: Color(0xff00201f),
  onTertiaryFixedVariant: Color(0xff1f7a76),
  error: Color(0xffffb4a4),
  onError: Color(0xff5f1605),
  errorContainer: Color(0xff8a2e1a),
  onErrorContainer: Color(0xffffdad2),
  surface: Color(0xff131318), // warm near-black page
  onSurface: Color(0xffededec),
  onSurfaceVariant: Color(0xff9b9a97), // Stone, lifted for dark
  outline: Color(0xff86857f), // Ash
  outlineVariant: Color(0xff2b2b31), // lifted hairline
  shadow: Color(0xff000000),
  scrim: Color(0xff0c0e14),
  inverseSurface: Color(0xff26262d), // dark toast / snackbar chip
  onInverseSurface: Color(0xffededec),
  inversePrimary: Color(0xff0075de),
  surfaceTint: Color(0xff3b8bef),
  surfaceDim: Color(0xff0d0d10),
  surfaceBright: Color(0xff3a3b40),
  surfaceContainerLowest: Color(0xff202027), // cards sit lighter than the page
  surfaceContainerLow: Color(0xff23232a),
  surfaceContainer: Color(0xff1b1b20),
  surfaceContainerHigh: Color(0xff26262d),
  surfaceContainerHighest: Color(0xff33333b),
);

ThemeData _buildTheme({
  required ColorScheme colorScheme,
  required Brightness brightness,
}) {
  final base = ThemeData(
    useMaterial3: true,
    brightness: brightness,
    colorScheme: colorScheme,
  );
  final textTheme = _buildTextTheme(base.textTheme, brightness, colorScheme);
  final overlayStyle = _systemOverlayStyle(colorScheme, brightness);
  // The Ambient system uses a crisp 1px hairline rather than a faded divider.
  // In the dark, hairlines are *lifted* (a faint white line) rather than dark.
  final dividerColor = brightness == Brightness.light ? colorScheme.outlineVariant : const Color(0x12ffffff);

  return base.copyWith(
    extensions: <ThemeExtension<dynamic>>[
      AmbientColors.fromBrightness(brightness),
    ],
    colorScheme: colorScheme,
    textTheme: textTheme,
    primaryTextTheme: textTheme,
    scaffoldBackgroundColor: colorScheme.surface,
    canvasColor: colorScheme.surface,
    cardColor: colorScheme.surfaceContainerLowest,
    dividerColor: dividerColor,
    primaryColor: colorScheme.primary,
    secondaryHeaderColor: colorScheme.surfaceContainerLow,
    highlightColor: colorScheme.surfaceContainerHigh,
    splashColor: colorScheme.primary.withValues(alpha: 0.12),
    hintColor: colorScheme.outline,
    disabledColor: colorScheme.onSurface.withValues(alpha: 0.38),
    unselectedWidgetColor: colorScheme.onSurfaceVariant,
    iconTheme: IconThemeData(color: colorScheme.primary),
    primaryIconTheme: IconThemeData(color: colorScheme.primary),
    appBarTheme: AppBarTheme(
      backgroundColor: colorScheme.surface,
      foregroundColor: colorScheme.onSurface,
      elevation: 0,
      scrolledUnderElevation: 0,
      surfaceTintColor: Colors.transparent,
      titleSpacing: 0,
      centerTitle: false,
      systemOverlayStyle: overlayStyle,
      titleTextStyle: textTheme.headlineSmall,
      toolbarTextStyle: textTheme.bodyMedium,
      iconTheme: IconThemeData(color: colorScheme.onSurface),
      actionsIconTheme: IconThemeData(color: colorScheme.onSurface),
    ),
    bottomAppBarTheme: BottomAppBarThemeData(
      color: colorScheme.surface.withValues(alpha: 0.92),
      surfaceTintColor: Colors.transparent,
      elevation: 0,
      padding: EdgeInsets.zero,
    ),
    navigationBarTheme: NavigationBarThemeData(
      height: 74,
      backgroundColor: colorScheme.surface.withValues(alpha: 0.94),
      surfaceTintColor: Colors.transparent,
      shadowColor: Colors.transparent,
      labelBehavior: NavigationDestinationLabelBehavior.alwaysShow,
      iconTheme: WidgetStateProperty.resolveWith((states) {
        final selected = states.contains(WidgetState.selected);
        return IconThemeData(
          color: selected ? colorScheme.primary : colorScheme.onSurfaceVariant,
          size: 24,
        );
      }),
      labelTextStyle: WidgetStateProperty.resolveWith((states) {
        final selected = states.contains(WidgetState.selected);
        return textTheme.labelSmall!.copyWith(
          color: selected ? colorScheme.primary : colorScheme.onSurfaceVariant,
          fontWeight: selected ? FontWeight.w700 : FontWeight.w600,
          letterSpacing: 0.2,
        );
      }),
      indicatorColor: colorScheme.secondaryContainer,
    ),
    cardTheme: CardThemeData(
      color: colorScheme.surfaceContainerLowest,
      surfaceTintColor: Colors.transparent,
      shadowColor: colorScheme.primary.withValues(alpha: brightness == Brightness.light ? 0.08 : 0.18),
      elevation: 0,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(24)),
      margin: EdgeInsets.zero,
    ),
    chipTheme: base.chipTheme.copyWith(
      backgroundColor: colorScheme.surfaceContainerLow,
      selectedColor: colorScheme.primary,
      secondarySelectedColor: colorScheme.primary,
      disabledColor: colorScheme.surfaceContainer,
      side: BorderSide.none,
      labelStyle: textTheme.labelLarge!.copyWith(color: colorScheme.primary),
      secondaryLabelStyle: textTheme.labelLarge!.copyWith(color: colorScheme.onPrimary),
      shape: const StadiumBorder(),
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
      showCheckmark: false,
    ),
    dividerTheme: DividerThemeData(
      color: dividerColor,
      thickness: 1,
      space: 1,
    ),
    dialogTheme: DialogThemeData(
      backgroundColor: colorScheme.surfaceContainerLow,
      surfaceTintColor: Colors.transparent,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(28)),
      titleTextStyle: textTheme.titleLarge,
      contentTextStyle: textTheme.bodyMedium,
    ),
    bottomSheetTheme: BottomSheetThemeData(
      backgroundColor: colorScheme.surface.withValues(alpha: 0.96),
      surfaceTintColor: Colors.transparent,
      modalBackgroundColor: colorScheme.surface.withValues(alpha: 0.96),
      showDragHandle: false,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(28)),
      ),
    ),
    snackBarTheme: SnackBarThemeData(
      behavior: SnackBarBehavior.floating,
      backgroundColor: colorScheme.inverseSurface,
      contentTextStyle: textTheme.bodyMedium!.copyWith(color: colorScheme.onInverseSurface),
      actionTextColor: colorScheme.primaryFixedDim,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(18)),
    ),
    inputDecorationTheme: InputDecorationTheme(
      filled: true,
      fillColor: colorScheme.surfaceContainerLowest,
      hintStyle: textTheme.bodyLarge!.copyWith(color: colorScheme.outline),
      prefixIconColor: colorScheme.onSurfaceVariant,
      suffixIconColor: colorScheme.onSurfaceVariant,
      contentPadding: const EdgeInsets.symmetric(horizontal: 20, vertical: 18),
      border: OutlineInputBorder(
        borderRadius: BorderRadius.circular(28),
        borderSide: BorderSide.none,
      ),
      enabledBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(28),
        borderSide: BorderSide.none,
      ),
      focusedBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(28),
        borderSide: BorderSide(color: colorScheme.primary.withValues(alpha: 0.45), width: 1.5),
      ),
    ),
    listTileTheme: ListTileThemeData(
      iconColor: colorScheme.primary,
      textColor: colorScheme.onSurface,
      subtitleTextStyle: textTheme.bodySmall!.copyWith(color: colorScheme.onSurfaceVariant),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(22)),
      contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 4),
    ),
    sliderTheme: base.sliderTheme.copyWith(
      trackHeight: 6,
      activeTrackColor: colorScheme.primary,
      inactiveTrackColor: colorScheme.surfaceContainerHighest,
      secondaryActiveTrackColor: colorScheme.primaryContainer,
      thumbColor: colorScheme.primary,
      overlayColor: colorScheme.primary.withValues(alpha: 0.12),
      valueIndicatorColor: colorScheme.primaryContainer,
      valueIndicatorTextStyle: textTheme.labelMedium!.copyWith(color: colorScheme.onPrimary),
      trackShape: const RoundedRectSliderTrackShape(),
    ),
    progressIndicatorTheme: ProgressIndicatorThemeData(
      color: colorScheme.primary,
      circularTrackColor: colorScheme.surfaceContainerHigh,
      linearTrackColor: colorScheme.surfaceContainerHighest,
      linearMinHeight: 6,
    ),
    popupMenuTheme: PopupMenuThemeData(
      color: colorScheme.surfaceContainerLow,
      surfaceTintColor: Colors.transparent,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
      textStyle: textTheme.bodyMedium,
    ),
    switchTheme: SwitchThemeData(
      thumbColor: WidgetStateProperty.resolveWith((states) {
        if (states.contains(WidgetState.selected)) {
          return colorScheme.onPrimary;
        }
        return colorScheme.outlineVariant;
      }),
      trackColor: WidgetStateProperty.resolveWith((states) {
        if (states.contains(WidgetState.selected)) {
          return colorScheme.primary;
        }
        return colorScheme.surfaceContainerHighest;
      }),
      trackOutlineColor: const WidgetStatePropertyAll(Colors.transparent),
    ),
    tabBarTheme: TabBarThemeData(
      dividerColor: Colors.transparent,
      indicatorColor: colorScheme.primary,
      indicatorSize: TabBarIndicatorSize.label,
      labelColor: colorScheme.primary,
      unselectedLabelColor: colorScheme.onSurfaceVariant,
      labelStyle: textTheme.labelLarge,
      unselectedLabelStyle: textTheme.labelLarge,
    ),
    outlinedButtonTheme: OutlinedButtonThemeData(
      style: OutlinedButton.styleFrom(
        foregroundColor: colorScheme.primary,
        side: BorderSide(color: colorScheme.outlineVariant.withValues(alpha: 0.2)),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(999)),
        padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 14),
      ),
    ),
    elevatedButtonTheme: ElevatedButtonThemeData(
      style: ElevatedButton.styleFrom(
        backgroundColor: colorScheme.primary,
        foregroundColor: colorScheme.onPrimary,
        elevation: 0,
        shadowColor: Colors.transparent,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(999)),
        padding: const EdgeInsets.symmetric(horizontal: 22, vertical: 16),
        textStyle: textTheme.labelLarge,
      ),
    ),
    textButtonTheme: TextButtonThemeData(
      style: TextButton.styleFrom(
        foregroundColor: colorScheme.primary,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(999)),
        textStyle: textTheme.labelLarge,
      ),
    ),
    iconButtonTheme: IconButtonThemeData(
      style: IconButton.styleFrom(
        backgroundColor: Colors.transparent,
        foregroundColor: colorScheme.onSurface,
        hoverColor: colorScheme.surfaceContainer,
        highlightColor: colorScheme.surfaceContainerHigh,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(999)),
      ),
    ),
    textSelectionTheme: TextSelectionThemeData(
      cursorColor: colorScheme.primary,
      selectionColor: colorScheme.primaryFixedDim.withValues(alpha: 0.5),
      selectionHandleColor: colorScheme.primary,
    ),
  );
}

TextTheme _buildTextTheme(
  TextTheme base,
  Brightness brightness,
  ColorScheme colorScheme,
) {
  final body = base.apply(
    fontFamily: _bodyFontFamily,
    fontFamilyFallback: _fontFallback,
    bodyColor: colorScheme.onSurface,
    displayColor: colorScheme.onSurface,
  );

  TextStyle headline(TextStyle? style, {FontWeight weight = FontWeight.w700, double? letterSpacing}) {
    return style!.copyWith(
      fontFamily: _headlineFontFamily,
      fontFamilyFallback: _fontFallback,
      fontWeight: weight,
      letterSpacing: letterSpacing,
      color: colorScheme.onSurface,
    );
  }

  return body.copyWith(
    displayLarge: headline(body.displayLarge, weight: FontWeight.w800, letterSpacing: -1.6),
    displayMedium: headline(body.displayMedium, weight: FontWeight.w800, letterSpacing: -1.2),
    displaySmall: headline(body.displaySmall, weight: FontWeight.w800, letterSpacing: -0.9),
    headlineLarge: headline(body.headlineLarge, weight: FontWeight.w800, letterSpacing: -0.8),
    headlineMedium: headline(body.headlineMedium, weight: FontWeight.w700, letterSpacing: -0.6),
    headlineSmall: headline(body.headlineSmall, weight: FontWeight.w700, letterSpacing: -0.4),
    titleLarge: headline(body.titleLarge, weight: FontWeight.w800, letterSpacing: -0.4),
    titleMedium: headline(body.titleMedium, weight: FontWeight.w700, letterSpacing: -0.2),
    titleSmall: headline(body.titleSmall, weight: FontWeight.w700, letterSpacing: -0.1),
    bodyLarge: body.bodyLarge?.copyWith(
      fontWeight: FontWeight.w500,
      color: colorScheme.onSurface,
    ),
    bodyMedium: body.bodyMedium?.copyWith(
      fontWeight: FontWeight.w500,
      color: colorScheme.onSurface,
    ),
    bodySmall: body.bodySmall?.copyWith(
      fontWeight: FontWeight.w500,
      color: colorScheme.onSurfaceVariant,
    ),
    labelLarge: body.labelLarge?.copyWith(
      fontWeight: FontWeight.w700,
      letterSpacing: 0.2,
      color: colorScheme.onSurface,
    ),
    labelMedium: body.labelMedium?.copyWith(
      fontWeight: FontWeight.w600,
      letterSpacing: 0.2,
      color: colorScheme.onSurfaceVariant,
    ),
    labelSmall: body.labelSmall?.copyWith(
      fontWeight: brightness == Brightness.light ? FontWeight.w700 : FontWeight.w600,
      letterSpacing: 0.4,
      color: colorScheme.onSurfaceVariant,
    ),
  );
}

SystemUiOverlayStyle _systemOverlayStyle(
  ColorScheme colorScheme,
  Brightness brightness,
) {
  final iconBrightness = brightness == Brightness.light ? Brightness.dark : Brightness.light;

  return SystemUiOverlayStyle(
    systemNavigationBarColor: colorScheme.surface,
    systemNavigationBarIconBrightness: iconBrightness,
    statusBarColor: Colors.transparent,
    statusBarIconBrightness: iconBrightness,
    statusBarBrightness: brightness == Brightness.light ? Brightness.light : Brightness.dark,
  );
}

/// Ambient accent tokens that don't map cleanly onto Material's [ColorScheme].
///
/// Notion Blue lives on [ColorScheme.primary] (the one action colour); the
/// deep-indigo "night" band on [ColorScheme.secondary]; and the AI/ad-skip
/// teal on [ColorScheme.tertiary]. This extension carries the remaining
/// design tokens — the teal cue surface, the toast success green, the subtle
/// control fill, and the crisp hairline — so bespoke player surfaces can pull
/// them with `Theme.of(context).extension<AmbientColors>()`.
@immutable
class AmbientColors extends ThemeExtension<AmbientColors> {
  final Color aiTeal;
  final Color aiTealSurface;
  final Color night;
  final Color onNight;
  final Color success;
  final Color controlFill;
  final Color hairline;
  final Color frostedSurface;
  final Color frostedBorder;

  const AmbientColors({
    required this.aiTeal,
    required this.aiTealSurface,
    required this.night,
    required this.onNight,
    required this.success,
    required this.controlFill,
    required this.hairline,
    required this.frostedSurface,
    required this.frostedBorder,
  });

  factory AmbientColors.fromBrightness(Brightness brightness) {
    if (brightness == Brightness.light) {
      return const AmbientColors(
        aiTeal: Color(0xff2a9d99),
        aiTealSurface: Color(0x1f2a9d99), // rgba(42,157,153,0.12)
        night: Color(0xff213183),
        onNight: Color(0xffffffff),
        success: Color(0xff2a9d99),
        controlFill: Color(0x0d000000), // rgba(0,0,0,0.05)
        hairline: Color(0xffe6e6e6),
        frostedSurface: Color(0xb8ffffff), // rgba(255,255,255,0.72)
        frostedBorder: Color(0xd9ffffff), // rgba(255,255,255,0.85)
      );
    }

    return const AmbientColors(
      aiTeal: Color(0xff3fc4b8),
      aiTealSurface: Color(0x2e3fc4b8), // rgba(63,196,184,0.18)
      night: Color(0xff243a86),
      onNight: Color(0xffffffff),
      success: Color(0xff5fd3a0),
      controlFill: Color(0x14ffffff), // white 0.08
      hairline: Color(0x12ffffff), // lifted white 0.07
      frostedSurface: Color(0xa8262a38), // rgba(38,42,56,0.66)
      frostedBorder: Color(0x17ffffff), // white 0.09
    );
  }

  @override
  AmbientColors copyWith({
    Color? aiTeal,
    Color? aiTealSurface,
    Color? night,
    Color? onNight,
    Color? success,
    Color? controlFill,
    Color? hairline,
    Color? frostedSurface,
    Color? frostedBorder,
  }) {
    return AmbientColors(
      aiTeal: aiTeal ?? this.aiTeal,
      aiTealSurface: aiTealSurface ?? this.aiTealSurface,
      night: night ?? this.night,
      onNight: onNight ?? this.onNight,
      success: success ?? this.success,
      controlFill: controlFill ?? this.controlFill,
      hairline: hairline ?? this.hairline,
      frostedSurface: frostedSurface ?? this.frostedSurface,
      frostedBorder: frostedBorder ?? this.frostedBorder,
    );
  }

  @override
  AmbientColors lerp(ThemeExtension<AmbientColors>? other, double t) {
    if (other is! AmbientColors) {
      return this;
    }
    return AmbientColors(
      aiTeal: Color.lerp(aiTeal, other.aiTeal, t)!,
      aiTealSurface: Color.lerp(aiTealSurface, other.aiTealSurface, t)!,
      night: Color.lerp(night, other.night, t)!,
      onNight: Color.lerp(onNight, other.onNight, t)!,
      success: Color.lerp(success, other.success, t)!,
      controlFill: Color.lerp(controlFill, other.controlFill, t)!,
      hairline: Color.lerp(hairline, other.hairline, t)!,
      frostedSurface: Color.lerp(frostedSurface, other.frostedSurface, t)!,
      frostedBorder: Color.lerp(frostedBorder, other.frostedBorder, t)!,
    );
  }

  /// Convenience accessor used throughout the player surfaces.
  static AmbientColors of(BuildContext context) {
    return Theme.of(context).extension<AmbientColors>() ?? AmbientColors.fromBrightness(Theme.of(context).brightness);
  }
}

/// The deep-indigo "night" surface with a soft radial teal glow, used by the
/// AI ad-skip feature (the Settings featured row and the AI ad-skip hero stat).
/// Extracted so the band + glow live in one place.
class AmbientNightBand extends StatelessWidget {
  final Widget child;
  final EdgeInsetsGeometry padding;
  final double borderRadius;
  final double glowSize;
  final double glowTop;
  final double glowRight;

  const AmbientNightBand({
    super.key,
    required this.child,
    required this.padding,
    required this.borderRadius,
    required this.glowSize,
    this.glowTop = -24.0,
    this.glowRight = -20.0,
  });

  @override
  Widget build(BuildContext context) {
    final ambient = AmbientColors.of(context);

    return Container(
      padding: padding,
      decoration: BoxDecoration(
        color: ambient.night,
        borderRadius: BorderRadius.circular(borderRadius),
      ),
      child: Stack(
        children: [
          Positioned(
            top: glowTop,
            right: glowRight,
            child: Container(
              width: glowSize,
              height: glowSize,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                gradient: RadialGradient(
                  colors: [
                    ambient.aiTeal.withValues(alpha: 0.5),
                    ambient.aiTeal.withValues(alpha: 0.0),
                  ],
                ),
              ),
            ),
          ),
          child,
        ],
      ),
    );
  }
}

class Themes {
  final ThemeData themeData;

  Themes({required this.themeData});

  factory Themes.lightTheme() {
    return Themes(themeData: _lightTheme);
  }

  factory Themes.darkTheme() {
    return Themes(themeData: _darkTheme);
  }
}
