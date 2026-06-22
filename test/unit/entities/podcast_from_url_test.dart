// Copyright 2020 Ben Hills and the project contributors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

import 'package:anytime/entities/podcast.dart';
import 'package:flutter_test/flutter_test.dart';

/// Covers `Podcast.fromUrl` — the transform behind the "Add RSS feed manually"
/// flow (feature F-FEED-01), which turns a user-entered URL into the Podcast
/// that is then loaded/subscribed.
void main() {
  group('Podcast.fromUrl', () {
    test('upgrades an http feed URL to https', () {
      final podcast = Podcast.fromUrl(url: 'http://example.com/feed.xml');

      expect(podcast.url, 'https://example.com/feed.xml');
    });

    test('leaves an https feed URL untouched', () {
      final podcast = Podcast.fromUrl(url: 'https://example.com/feed.xml');

      expect(podcast.url, 'https://example.com/feed.xml');
    });

    test('creates an otherwise-empty placeholder podcast', () {
      final podcast = Podcast.fromUrl(url: 'https://example.com/feed.xml');

      expect(podcast.guid, isEmpty);
      expect(podcast.title, isEmpty);
      expect(podcast.link, isEmpty);
      expect(podcast.funding, isEmpty);
      expect(podcast.persons, isEmpty);
      expect(podcast.episodes, isEmpty);
      // Not yet subscribed.
      expect(podcast.subscribed, isFalse);
    });
  });
}
