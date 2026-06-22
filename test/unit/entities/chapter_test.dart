// Copyright 2020 Ben Hills and the project contributors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

import 'package:anytime/entities/chapter.dart';
import 'package:flutter_test/flutter_test.dart';

/// Covers the Chapter data model (feature F-CHAP-01).
void main() {
  group('Chapter construction', () {
    test('upgrades http image and link URLs to https', () {
      final chapter = Chapter(
        title: 'Intro',
        imageUrl: 'http://example.com/art.png',
        url: 'http://example.com/notes',
        startTime: 0.0,
      );

      expect(chapter.imageUrl, 'https://example.com/art.png');
      expect(chapter.url, 'https://example.com/notes');
    });

    test('leaves https URLs untouched', () {
      final chapter = Chapter(
        title: 'Intro',
        imageUrl: 'https://example.com/art.png',
        startTime: 0.0,
      );

      expect(chapter.imageUrl, 'https://example.com/art.png');
    });

    test('null image/link URLs are tolerated', () {
      final chapter = Chapter(title: 'Intro', imageUrl: null, startTime: 0.0);

      expect(chapter.imageUrl, isNull);
      expect(chapter.url, isNull);
    });

    test('toc defaults to true and endTime to 0.0', () {
      final chapter = Chapter(title: 'Intro', imageUrl: null, startTime: 12.5);

      expect(chapter.toc, isTrue);
      expect(chapter.endTime, 0.0);
    });
  });

  group('Chapter serialization', () {
    test('round-trips through toMap/fromMap', () {
      final chapter = Chapter(
        title: 'Segment 2',
        imageUrl: 'https://example.com/2.png',
        url: 'https://example.com/2',
        startTime: 90.0,
        endTime: 180.0,
        toc: false,
      );

      final restored = Chapter.fromMap(chapter.toMap());

      expect(restored.title, 'Segment 2');
      expect(restored.imageUrl, 'https://example.com/2.png');
      expect(restored.url, 'https://example.com/2');
      expect(restored.startTime, 90.0);
      expect(restored.endTime, 180.0);
      expect(restored.toc, isFalse);
    });

    test('toMap encodes toc as a string boolean', () {
      final included = Chapter(title: 'A', imageUrl: null, startTime: 0.0, toc: true);
      final excluded = Chapter(title: 'B', imageUrl: null, startTime: 0.0, toc: false);

      expect(included.toMap()['toc'], 'true');
      expect(excluded.toMap()['toc'], 'false');
    });

    test('fromMap treats any non-"false" toc value as true', () {
      final map = <String, dynamic>{
        'title': 'A',
        'imageUrl': null,
        'url': null,
        'toc': 'true',
        'startTime': '0.0',
        'endTime': '0.0',
      };

      expect(Chapter.fromMap(map).toc, isTrue);
    });
  });

  group('Chapter equality', () {
    test('equal by title and startTime', () {
      final a = Chapter(title: 'A', imageUrl: null, startTime: 10.0, endTime: 20.0);
      final b = Chapter(title: 'A', imageUrl: null, startTime: 10.0, endTime: 99.0);

      expect(a, equals(b));
      expect(a.hashCode, b.hashCode);
    });

    test('differing startTime breaks equality', () {
      final a = Chapter(title: 'A', imageUrl: null, startTime: 10.0);
      final b = Chapter(title: 'A', imageUrl: null, startTime: 11.0);

      expect(a, isNot(equals(b)));
    });
  });
}
