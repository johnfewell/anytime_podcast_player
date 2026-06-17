// Copyright 2020 Ben Hills and the project contributors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

import 'package:anytime/bloc/podcast/audio_bloc.dart';
import 'package:anytime/entities/ad_segment.dart';
import 'package:anytime/entities/episode.dart';
import 'package:anytime/entities/transcript.dart';
import 'package:anytime/services/audio/audio_player_service.dart';
import 'package:anytime/ui/podcast/transcript_view.dart';
import 'package:anytime/ui/themes.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:provider/provider.dart';

/// The inline ad marker (`AdSegmentMarker`) and the current-line highlight
/// treatment (`SubtitleWidget`) are the new transcript surfaces added by the
/// Ambient redesign. `_buildAdMarkers` is private and reaches the marker purely
/// through these public widgets, so we exercise them directly rather than
/// driving the whole scrollable transcript view.
void main() {
  group('AdSegmentMarker', () {
    Future<void> pumpMarker(WidgetTester tester, AdSegment segment) async {
      await tester.pumpWidget(
        MaterialApp(
          theme: Themes.lightTheme().themeData,
          home: Scaffold(body: AdSegmentMarker(segment: segment)),
        ),
      );
      await tester.pump();
    }

    testWidgets('renders the teal AI badge copy and the segment length', (tester) async {
      await pumpMarker(tester, const AdSegment(startMs: 1000, endMs: 4000));

      expect(find.byIcon(Icons.auto_awesome), findsOneWidget);
      expect(find.text('Sponsor segment · 3s'), findsOneWidget);
      expect(find.text('Detected by AI · skipped automatically'), findsOneWidget);
    });

    testWidgets('formats a longer segment as whole seconds', (tester) async {
      await pumpMarker(tester, const AdSegment(startMs: 0, endMs: 65000));

      expect(find.text('Sponsor segment · 65s'), findsOneWidget);
    });

    testWidgets('uses the Ambient teal surface tint', (tester) async {
      await pumpMarker(tester, const AdSegment(startMs: 1000, endMs: 4000));

      final ambient = AmbientColors.fromBrightness(Brightness.light);
      final container = tester.widget<Container>(find.byType(Container).first);
      final decoration = container.decoration as BoxDecoration;

      expect(decoration.color, ambient.aiTealSurface);
    });
  });

  group('SubtitleWidget highlight treatment', () {
    late AudioBloc audioBloc;

    setUp(() {
      audioBloc = AudioBloc(audioPlayerService: _NoopAudioPlayerService());
    });

    tearDown(() => audioBloc.dispose());

    Future<void> pumpSubtitle(WidgetTester tester, Subtitle subtitle, {bool highlight = false}) async {
      await tester.pumpWidget(
        Provider<AudioBloc>.value(
          value: audioBloc,
          child: MaterialApp(
            theme: Themes.lightTheme().themeData,
            home: Scaffold(body: SubtitleWidget(subtitle: subtitle, highlight: highlight)),
          ),
        ),
      );
    }

    testWidgets('renders the line and timestamp', (tester) async {
      final subtitle = Subtitle(
        index: 1,
        start: Duration.zero,
        end: const Duration(seconds: 2),
        data: 'Hello world',
      );

      await pumpSubtitle(tester, subtitle);

      expect(find.text('Hello world'), findsOneWidget);
      expect(find.textContaining('00:00:00'), findsOneWidget);
    });

    testWidgets('highlight paints the leading tick in the primary colour', (tester) async {
      final subtitle = Subtitle(
        index: 1,
        start: Duration.zero,
        end: const Duration(seconds: 1),
        data: 'Active line',
      );

      final theme = Themes.lightTheme().themeData;
      await pumpSubtitle(tester, subtitle, highlight: true);

      final tick = tester.widget<AnimatedContainer>(find.byType(AnimatedContainer));
      final decoration = tick.decoration as BoxDecoration;

      expect(decoration.color, theme.colorScheme.primary);
    });

    testWidgets('without highlight the leading tick is transparent', (tester) async {
      final subtitle = Subtitle(
        index: 1,
        start: Duration.zero,
        end: const Duration(seconds: 1),
        data: 'Idle line',
      );

      await pumpSubtitle(tester, subtitle, highlight: false);

      final tick = tester.widget<AnimatedContainer>(find.byType(AnimatedContainer));
      final decoration = tick.decoration as BoxDecoration;

      expect(decoration.color, Colors.transparent);
    });

    testWidgets('formats a speaker label when present', (tester) async {
      final subtitle = Subtitle(
        index: 2,
        start: const Duration(seconds: 5),
        end: const Duration(seconds: 8),
        data: 'Hello',
        speaker: 'Alice',
      );

      await pumpSubtitle(tester, subtitle);

      expect(find.textContaining('Alice'), findsOneWidget);
    });
  });
}

class _NoopAudioPlayerService implements AudioPlayerService {
  @override
  Episode? nowPlaying;

  @override
  noSuchMethod(Invocation invocation) => super.noSuchMethod(invocation);
}
