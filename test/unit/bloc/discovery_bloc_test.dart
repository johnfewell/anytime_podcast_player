// Copyright 2020 Ben Hills and the project contributors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

import 'package:anytime/bloc/discovery/discovery_bloc.dart';
import 'package:anytime/bloc/discovery/discovery_state_event.dart';
import 'package:anytime/services/podcast/podcast_service.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:podcast_search/podcast_search.dart' as pcast;

/// Covers the Discovery charts BLoC (feature F-DISC-01) — specifically the
/// loading→populated lifecycle and the 30-minute results cache that lets the
/// Discover tab skip the spinner when revisited.
void main() {
  late _FakeDiscoveryPodcastService service;
  late DiscoveryBloc bloc;

  setUp(() {
    // 'Technology' sits at index 1 (>0) so the bloc retains it as the last
    // genre and the cache can match on a repeat request.
    service = _FakeDiscoveryPodcastService(genreList: ['All', 'Technology']);
    bloc = DiscoveryBloc(podcastService: service);
  });

  tearDown(() => bloc.dispose());

  Future<void> settle() => Future<void>.delayed(const Duration(milliseconds: 10));

  test('first request emits Loading then Populated and hits the network', () async {
    final states = <DiscoveryState>[];
    final sub = bloc.results!.listen(states.add);

    bloc.discover(DiscoveryChartEvent(count: 10, genre: 'Technology'));
    await settle();

    expect(states.whereType<DiscoveryLoadingState>(), hasLength(1));
    final populated = states.whereType<DiscoveryPopulatedState>().toList();
    expect(populated, hasLength(1));
    expect(populated.single.genre, 'Technology');
    expect(service.chartsCalls, 1);

    await sub.cancel();
  });

  test('a warm cache skips the loading spinner and avoids a second fetch', () async {
    bloc.discover(DiscoveryChartEvent(count: 10, genre: 'Technology'));
    await settle();
    expect(service.chartsCalls, 1);

    // Re-request the same genre: should serve from cache.
    final states = <DiscoveryState>[];
    final sub = bloc.results!.listen(states.add);

    bloc.discover(DiscoveryChartEvent(count: 10, genre: 'Technology'));
    await settle();

    // No spinner re-emitted, no second network call.
    expect(states.whereType<DiscoveryLoadingState>(), isEmpty);
    expect(states.whereType<DiscoveryPopulatedState>(), isNotEmpty);
    expect(service.chartsCalls, 1);

    await sub.cancel();
  });

  test('changing genre invalidates the cache and refetches', () async {
    bloc.discover(DiscoveryChartEvent(count: 10, genre: 'Technology'));
    await settle();
    expect(service.chartsCalls, 1);

    final states = <DiscoveryState>[];
    final sub = bloc.results!.listen(states.add);

    bloc.discover(DiscoveryChartEvent(count: 10, genre: 'All'));
    await settle();

    expect(states.whereType<DiscoveryLoadingState>(), hasLength(1));
    expect(service.chartsCalls, 2);

    await sub.cancel();
  });
}

class _FakeDiscoveryPodcastService implements PodcastService {
  _FakeDiscoveryPodcastService({required this.genreList});

  final List<String> genreList;
  int chartsCalls = 0;

  @override
  List<String> genres() => genreList;

  @override
  Future<pcast.SearchResult> charts({
    int size = 20,
    String? genre,
    String? countryCode,
    String? languageCode,
  }) async {
    chartsCalls++;
    return pcast.SearchResult(resultCount: 1);
  }

  @override
  noSuchMethod(Invocation invocation) => super.noSuchMethod(invocation);
}
