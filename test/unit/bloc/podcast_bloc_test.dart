// Copyright 2020 Ben Hills and the project contributors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

import 'package:anytime/bloc/podcast/podcast_bloc.dart';
import 'package:anytime/entities/episode.dart';
import 'package:anytime/entities/feed.dart';
import 'package:anytime/entities/podcast.dart';
import 'package:anytime/services/audio/audio_player_service.dart';
import 'package:anytime/services/download/download_manager.dart';
import 'package:anytime/services/download/download_service.dart';
import 'package:anytime/services/download/mobile_download_service.dart';
import 'package:anytime/services/podcast/podcast_service.dart';
import 'package:anytime/state/episode_state.dart';
import 'package:anytime/state/library_state.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:rxdart/rxdart.dart';

import '../mocks/mock_notification_service.dart';
import '../mocks/mock_settings_service.dart';

/// Covers the "mark all played" podcast action (feature F-LIB-03), handled by
/// PodcastBloc via [PodcastEvent.markAllPlayed].
void main() {
  Episode ep(String guid, {bool played = false, int position = 1000}) => Episode(
        guid: guid,
        pguid: 'pod-1',
        podcast: 'Podcast',
        title: 'Episode $guid',
        contentUrl: 'https://cdn.example.com/$guid.mp3',
        played: played,
        position: position,
      );

  Podcast podcastWith(List<Episode> episodes) => Podcast(
        guid: 'pod-1',
        url: 'https://example.com/feed.xml',
        link: 'https://example.com',
        title: 'Podcast',
        episodes: episodes,
      );

  late _MarkPlayedPodcastService podcastService;
  late PodcastBloc bloc;

  PodcastBloc buildBloc(Podcast podcast) {
    // PodcastBloc.dispose() closes the static MobileDownloadService.downloadProgress
    // subject. Give each test a fresh one so disposing here can't break other
    // suites (e.g. episode_actions_sheet_test) that also build a PodcastBloc.
    MobileDownloadService.downloadProgress = BehaviorSubject<DownloadProgress>();
    podcastService = _MarkPlayedPodcastService(podcastToLoad: podcast);
    return PodcastBloc(
      podcastService: podcastService,
      audioPlayerService: _NoopAudioPlayerService(),
      downloadService: _NoopDownloadService(),
      notificationService: MockNotificationService(),
      settingsService: MockSettingsService(),
    );
  }

  tearDown(() {
    bloc.dispose();
    // Leave the static subject fresh and open for any later test in the suite.
    MobileDownloadService.downloadProgress = BehaviorSubject<DownloadProgress>();
  });

  Future<void> settle() => Future<void>.delayed(const Duration(milliseconds: 20));

  test('marks every unplayed episode as played and persists the change', () async {
    final episodes = [
      ep('a', played: false, position: 5000),
      ep('b', played: true, position: 0),
      ep('c', played: false, position: 1234),
    ];
    bloc = buildBloc(podcastWith(episodes));

    // Drive a load so the bloc has a current podcast to operate on.
    bloc.load(Feed(podcast: podcastWith(episodes)));
    await settle();

    bloc.podcastEvent(PodcastEvent.markAllPlayed);
    await settle();

    // Only the two previously-unplayed episodes are saved, both now played and
    // rewound to position 0.
    expect(podcastService.savedEpisodes, isNotNull);
    expect(podcastService.savedEpisodes!.map((e) => e.guid), ['a', 'c']);
    expect(podcastService.savedEpisodes!.every((e) => e.played), isTrue);
    expect(podcastService.savedEpisodes!.every((e) => e.position == 0), isTrue);
  });

  test('is a no-op (saves an empty set) when everything is already played', () async {
    final episodes = [ep('a', played: true, position: 0), ep('b', played: true, position: 0)];
    bloc = buildBloc(podcastWith(episodes));

    bloc.load(Feed(podcast: podcastWith(episodes)));
    await settle();

    bloc.podcastEvent(PodcastEvent.markAllPlayed);
    await settle();

    expect(podcastService.savedEpisodes, isEmpty);
  });
}

class _MarkPlayedPodcastService implements PodcastService {
  _MarkPlayedPodcastService({required this.podcastToLoad});

  final Podcast podcastToLoad;
  List<Episode>? savedEpisodes;

  @override
  Stream<LibraryState> get libraryListener => const Stream<LibraryState>.empty();

  @override
  Stream<EpisodeState> get episodeListener => const Stream<EpisodeState>.empty();

  @override
  Future<Podcast?> loadPodcast({
    required Podcast podcast,
    bool highlightNewEpisodes = false,
    bool ignoreCache = false,
  }) async =>
      podcastToLoad;

  @override
  Future<List<Episode>> saveEpisodes(List<Episode> episodes) async {
    savedEpisodes = episodes;
    return episodes;
  }

  @override
  noSuchMethod(Invocation invocation) => super.noSuchMethod(invocation);
}

class _NoopAudioPlayerService implements AudioPlayerService {
  @override
  noSuchMethod(Invocation invocation) => super.noSuchMethod(invocation);
}

class _NoopDownloadService implements DownloadService {
  @override
  void dispose() {}

  @override
  noSuchMethod(Invocation invocation) => super.noSuchMethod(invocation);
}
