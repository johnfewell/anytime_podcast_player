// Copyright 2020 Ben Hills and the project contributors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

import 'package:anytime/bloc/bloc.dart';
import 'package:anytime/bloc/discovery/discovery_state_event.dart';
import 'package:anytime/services/podcast/podcast_service.dart';
import 'dart:async';

import 'package:logging/logging.dart';
import 'package:podcast_search/podcast_search.dart' as podcast_search;
import 'package:rxdart/rxdart.dart';

/// A BLoC to interact with the Discovery UI page and the [PodcastService] to
/// fetch the iTunes/PodcastIndex charts.
///
/// As charts will not change very frequently the results are cached for [cacheMinutes].
class DiscoveryBloc extends Bloc {
  static const cacheMinutes = 30;

  final log = Logger('DiscoveryBloc');
  final PodcastService podcastService;

  /// Takes an event which triggers a loading of chart data from the selected provider.
  final _discoveryInput = BehaviorSubject<DiscoveryEvent>();

  /// A stream of genres from the selected provider.
  final _genres = PublishSubject<List<String>>();

  /// The last genre to be passed in a [DiscoveryEvent].
  final _selectedGenre = BehaviorSubject<SelectedGenre>(sync: true);

  /// The latest discovery state. A [BehaviorSubject] so a freshly-mounted
  /// Discover tab immediately receives the last populated state instead of
  /// re-running the loading spinner — this is what kills the layout shift when
  /// you return to Discover, and lets us preload it in the background at startup.
  final _discoveryOutput = BehaviorSubject<DiscoveryState>();

  StreamSubscription<DiscoveryState>? _chartsSubscription;

  /// To save bandwidth we cache the results.
  podcast_search.SearchResult? _resultsCache;

  String _lastGenre = '';
  int _lastIndex = 0;

  DiscoveryBloc({required this.podcastService}) {
    _init();
  }

  void _init() {
    _chartsSubscription = _discoveryInput
        .switchMap<DiscoveryState>((DiscoveryEvent event) => _charts(event))
        .listen(_discoveryOutput.add);
    _selectedGenre.value = SelectedGenre(index: 0, genre: '');
    _genres.onListen = _loadGenres;
  }

  void _loadGenres() {
    _genres.sink.add(podcastService.genres());
  }

  Stream<DiscoveryState> _charts(DiscoveryEvent event) async* {
    if (event is DiscoveryChartEvent) {
      final cacheValid = _resultsCache != null &&
          event.genre == _lastGenre &&
          DateTime.now().difference(_resultsCache!.processedTime).inMinutes <= cacheMinutes;

      // Only surface the loading spinner when we genuinely have to hit the
      // network. A warm cache (including the startup preload) goes straight to
      // the populated state, so switching back to Discover doesn't flash.
      if (!cacheValid) {
        yield DiscoveryLoadingState();

        _lastGenre = event.genre;
        _lastIndex = podcastService.genres().indexOf(_lastGenre);

        if (_lastIndex > 0) {
          _selectedGenre.add(SelectedGenre(index: _lastIndex, genre: _lastGenre));
        } else {
          /// Must have changed provider
          _lastGenre = '';
          _selectedGenre.add(SelectedGenre(index: 0, genre: ''));
        }
        _resultsCache = await podcastService.charts(
          size: event.count,
          genre: event.genre,
          countryCode: event.countryCode,
          languageCode: event.languageCode,
        );
      }

      yield DiscoveryPopulatedState<podcast_search.SearchResult>(
        genre: event.genre,
        index: podcastService.genres().indexOf(event.genre),
        results: _resultsCache,
      );
    }
  }

  @override
  void dispose() {
    _chartsSubscription?.cancel();
    _discoveryInput.close();
    _discoveryOutput.close();
    _genres.close();
    _selectedGenre.close();
  }

  void Function(DiscoveryEvent) get discover => _discoveryInput.add;

  Stream<DiscoveryState>? get results => _discoveryOutput.stream;

  Stream<List<String>> get genres => _genres.stream;

  SelectedGenre get selectedGenre => _selectedGenre.value;
}

class SelectedGenre {
  final int index;
  final String genre;

  SelectedGenre({
    required this.index,
    required this.genre,
  });
}
