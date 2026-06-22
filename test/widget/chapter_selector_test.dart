// Copyright 2020 Ben Hills and the project contributors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

import 'package:anytime/bloc/podcast/audio_bloc.dart';
import 'package:anytime/entities/chapter.dart';
import 'package:anytime/entities/episode.dart';
import 'package:anytime/l10n/L.dart';
import 'package:anytime/services/audio/audio_player_service.dart';
import 'package:anytime/ui/podcast/chapter_selector.dart';
import 'package:anytime/ui/themes.dart';
import 'package:anytime/ui/widgets/platform_progress_indicator.dart';
import 'package:flutter/material.dart';
import 'package:flutter_localizations/flutter_localizations.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:provider/provider.dart';
import 'package:rxdart/rxdart.dart';

/// Covers the Chapters navigation UI (feature F-CHAP-01): table-of-contents
/// filtering, the loading state, and chapter numbering.
void main() {
  late _ChapterAudioPlayerService audio;
  late AudioBloc audioBloc;

  setUp(() {
    audio = _ChapterAudioPlayerService();
    audioBloc = AudioBloc(audioPlayerService: audio);
  });

  tearDown(() {
    audioBloc.dispose();
    audio.dispose();
  });

  Episode episodeWith(List<Chapter> chapters, {bool loading = false}) {
    final episode = Episode(
      guid: 'ep-chapters',
      pguid: 'pod-1',
      podcast: 'Chaptered Podcast',
      title: 'Chaptered Episode',
      author: 'Author',
      imageUrl: '',
      contentUrl: 'https://cdn.example.com/ep.mp3',
      duration: 600,
      chapters: chapters,
    );
    episode.chaptersLoading = loading;
    return episode;
  }

  Future<void> pump(WidgetTester tester, Episode episode, {bool settle = true}) async {
    audio.episodeEvent.add(episode);
    await tester.pumpWidget(
      Provider<AudioBloc>.value(
        value: audioBloc,
        child: MaterialApp(
          theme: Themes.lightTheme().themeData,
          localizationsDelegates: const [
            AnytimeLocalisationsDelegate(),
            GlobalMaterialLocalizations.delegate,
            GlobalWidgetsLocalizations.delegate,
            GlobalCupertinoLocalizations.delegate,
          ],
          supportedLocales: const [Locale('en')],
          home: Scaffold(
            body: SizedBox(height: 600, child: ChapterSelector(episode: episode)),
          ),
        ),
      ),
    );
    if (settle) {
      await tester.pumpAndSettle();
    } else {
      // The loading state shows a CircularProgressIndicator that animates
      // forever, so pumpAndSettle would time out — pump a couple of frames.
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 50));
    }
  }

  testWidgets('lists only table-of-contents chapters', (tester) async {
    final episode = episodeWith([
      Chapter(title: 'Welcome', imageUrl: null, startTime: 0.0),
      Chapter(title: 'Hidden meta', imageUrl: null, startTime: 30.0, toc: false),
      Chapter(title: 'Main topic', imageUrl: null, startTime: 60.0),
    ]);

    await pump(tester, episode);

    expect(find.text('Welcome'), findsOneWidget);
    expect(find.text('Main topic'), findsOneWidget);
    // toc=false chapters are metadata only and must not be displayed.
    expect(find.text('Hidden meta'), findsNothing);
  });

  testWidgets('numbers the visible chapters sequentially', (tester) async {
    final episode = episodeWith([
      Chapter(title: 'One', imageUrl: null, startTime: 0.0),
      Chapter(title: 'Two', imageUrl: null, startTime: 60.0),
    ]);

    await pump(tester, episode);

    expect(find.text('1.'), findsOneWidget);
    expect(find.text('2.'), findsOneWidget);
  });

  testWidgets('shows a progress indicator while chapters are loading', (tester) async {
    final episode = episodeWith(
      [Chapter(title: 'One', imageUrl: null, startTime: 0.0)],
      loading: true,
    );

    await pump(tester, episode, settle: false);

    expect(find.byType(PlatformProgressIndicator), findsOneWidget);
    expect(find.text('One'), findsNothing);
  });
}

class _ChapterAudioPlayerService implements AudioPlayerService {
  @override
  final BehaviorSubject<Episode?> episodeEvent = BehaviorSubject<Episode?>();

  final BehaviorSubject<PositionState> _playPosition = BehaviorSubject<PositionState>();

  @override
  ValueStream<PositionState> get playPosition => _playPosition;

  void dispose() {
    episodeEvent.close();
    _playPosition.close();
  }

  @override
  noSuchMethod(Invocation invocation) => super.noSuchMethod(invocation);
}
