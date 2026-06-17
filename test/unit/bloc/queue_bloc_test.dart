// Copyright 2020 Ben Hills and the project contributors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

import 'package:anytime/bloc/podcast/queue_bloc.dart';
import 'package:anytime/entities/episode.dart';
import 'package:anytime/entities/sleep.dart';
import 'package:anytime/services/audio/audio_player_service.dart';
import 'package:anytime/services/podcast/podcast_service.dart';
import 'package:anytime/state/ad_skip_state.dart';
import 'package:anytime/state/queue_event_state.dart';
import 'package:anytime/state/transcript_state_event.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:rxdart/rxdart.dart';

Episode _episode(String guid) => Episode(
      guid: guid,
      podcast: 'Podcast',
      title: guid,
      contentUrl: 'https://cdn.example.com/$guid.mp3',
    );

void main() {
  late _QueueAudioPlayerService audioService;
  late _NoopPodcastService podcastService;

  setUp(() {
    audioService = _QueueAudioPlayerService();
    podcastService = _NoopPodcastService();
  });

  QueueBloc buildBloc() => QueueBloc(
        audioPlayerService: audioService,
        podcastService: podcastService,
      );

  group('QueueBloc QueueAddEvent', () {
    test('"Play next" (position:0) moves the episode to the front', () async {
      // Pre-seed the up-next queue with two episodes.
      audioService.seed(<Episode>[_episode('a'), _episode('b')]);

      final bloc = buildBloc();
      addTearDown(bloc.dispose);

      final added = _episode('c');
      bloc.queueEvent(QueueAddEvent(episode: added, position: 0));

      // Drain the async event handler (addUpNextEpisode then a moveUpNextEpisode).
      await _flush();
      await _flush();

      // addUpNextEpisode ran, and because the appended episode landed at the
      // back, the bloc repositioned it to the requested index 0.
      expect(audioService.addCalls, <String>['c']);
      expect(audioService.moveCalls, <_Move>[const _Move(guid: 'c', oldIndex: 2, newIndex: 0)]);

      expect(audioService.queueGuids, <String>['c', 'a', 'b']);
    });

    test('an add without a position appends and never invokes move', () async {
      audioService.seed(<Episode>[_episode('a'), _episode('b')]);

      final bloc = buildBloc();
      addTearDown(bloc.dispose);

      bloc.queueEvent(QueueAddEvent(episode: _episode('c')));

      await _flush();
      await _flush();

      expect(audioService.addCalls, <String>['c']);
      expect(audioService.moveCalls, isEmpty);
      expect(audioService.queueGuids, <String>['a', 'b', 'c']);
    });

    test('"Play next" onto an empty queue appends without a move', () async {
      final bloc = buildBloc();
      addTearDown(bloc.dispose);

      bloc.queueEvent(QueueAddEvent(episode: _episode('only'), position: 0));

      await _flush();
      await _flush();

      expect(audioService.addCalls, <String>['only']);
      // from (0) equals target (0), so no move is requested.
      expect(audioService.moveCalls, isEmpty);
      expect(audioService.queueGuids, <String>['only']);
    });
  });
}

/// Pump a couple of microtask turns so the bloc's async queue event handler
/// can finish its await chain.
Future<void> _flush() => Future<void>.delayed(Duration.zero);

class _Move {
  const _Move({required this.guid, required this.oldIndex, required this.newIndex});

  final String guid;
  final int oldIndex;
  final int newIndex;

  @override
  bool operator ==(Object other) =>
      other is _Move && guid == other.guid && oldIndex == other.oldIndex && newIndex == other.newIndex;

  @override
  int get hashCode => Object.hash(guid, oldIndex, newIndex);

  @override
  String toString() => '_Move($guid $oldIndex->$newIndex)';
}

/// A queue-focused [AudioPlayerService] fake that keeps an in-memory queue and
/// records the add/move operations the [QueueBloc] drives.
class _QueueAudioPlayerService implements AudioPlayerService {
  final BehaviorSubject<QueueListState> _queueState = BehaviorSubject<QueueListState>();

  final List<String> addCalls = <String>[];
  final List<_Move> moveCalls = <_Move>[];

  List<Episode> _queue = <Episode>[];

  void seed(List<Episode> episodes) {
    _queue = List<Episode>.of(episodes);
    _emit();
  }

  List<String> get queueGuids => _queue.map((e) => e.guid).toList(growable: false);

  void _emit() {
    _queueState.add(QueueListState(playing: null, queue: List<Episode>.unmodifiable(_queue)));
  }

  @override
  Future<void> addUpNextEpisode(Episode episode) async {
    addCalls.add(episode.guid);
    _queue.add(episode);
    _emit();
  }

  @override
  Future<bool> moveUpNextEpisode(Episode episode, int oldIndex, int newIndex) async {
    moveCalls.add(_Move(guid: episode.guid, oldIndex: oldIndex, newIndex: newIndex));
    if (oldIndex < 0 || oldIndex >= _queue.length) {
      return false;
    }
    final moved = _queue.removeAt(oldIndex);
    final clamped = newIndex.clamp(0, _queue.length);
    _queue.insert(clamped, moved);
    _emit();
    return true;
  }

  @override
  Future<bool> removeUpNextEpisode(Episode episode) async {
    final index = _queue.indexWhere((e) => e.guid == episode.guid);
    if (index < 0) {
      return false;
    }
    _queue.removeAt(index);
    _emit();
    return true;
  }

  @override
  Future<void> clearUpNext() async {
    _queue = <Episode>[];
    _emit();
  }

  @override
  Stream<QueueListState>? get queueState => _queueState.stream;

  @override
  Episode? nowPlaying;

  @override
  Stream<AudioState>? playingState;

  @override
  ValueStream<PositionState>? playPosition;

  @override
  ValueStream<Episode?>? episodeEvent;

  @override
  Stream<TranscriptState>? transcriptEvent;

  @override
  Stream<AdSkipState>? adSkipEvent;

  @override
  Stream<int>? playbackError;

  @override
  Stream<Sleep>? sleepStream;

  @override
  noSuchMethod(Invocation invocation) => super.noSuchMethod(invocation);
}

class _NoopPodcastService implements PodcastService {
  @override
  Future<void> saveQueue(List<Episode> queue) async {}

  @override
  noSuchMethod(Invocation invocation) => super.noSuchMethod(invocation);
}
