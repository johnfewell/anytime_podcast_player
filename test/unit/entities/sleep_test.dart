// Copyright 2020 Ben Hills and the project contributors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

import 'package:anytime/entities/sleep.dart';
import 'package:flutter_test/flutter_test.dart';

/// Covers the Sleep timer data model (feature F-SLEEP-01).
void main() {
  group('Sleep construction', () {
    test('defaults to a zero duration when none is supplied', () {
      final sleep = Sleep(type: SleepType.none);

      expect(sleep.type, SleepType.none);
      expect(sleep.duration, Duration.zero);
    });

    test('endTime is computed as now + duration', () {
      const duration = Duration(minutes: 30);
      final before = DateTime.now();

      final sleep = Sleep(type: SleepType.time, duration: duration);

      final after = DateTime.now();
      // endTime must fall within [before+duration, after+duration].
      expect(sleep.endTime.isBefore(before.add(duration)), isFalse);
      expect(sleep.endTime.isAfter(after.add(duration)), isFalse);
    });

    test('timeRemaining is positive and within the configured duration', () {
      const duration = Duration(minutes: 15);

      final sleep = Sleep(type: SleepType.time, duration: duration);

      expect(sleep.timeRemaining.inSeconds, greaterThan(0));
      expect(sleep.timeRemaining.inSeconds, lessThanOrEqualTo(duration.inSeconds));
    });
  });

  group('Sleep equality', () {
    test('same type and duration are equal and share a hashCode', () {
      final a = Sleep(type: SleepType.time, duration: const Duration(minutes: 5));
      final b = Sleep(type: SleepType.time, duration: const Duration(minutes: 5));

      expect(a, equals(b));
      expect(a.hashCode, b.hashCode);
    });

    test('differing duration breaks equality', () {
      final a = Sleep(type: SleepType.time, duration: const Duration(minutes: 5));
      final b = Sleep(type: SleepType.time, duration: const Duration(minutes: 10));

      expect(a, isNot(equals(b)));
    });

    test('differing type breaks equality', () {
      final a = Sleep(type: SleepType.time, duration: const Duration(minutes: 5));
      final b = Sleep(type: SleepType.episode, duration: const Duration(minutes: 5));

      expect(a, isNot(equals(b)));
    });

    test('identical instance equals itself', () {
      final a = Sleep(type: SleepType.episode);

      expect(a, equals(a));
    });
  });
}
