// Copyright 2020 Ben Hills and the project contributors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

import 'package:anytime/entities/ad_segment.dart';
import 'package:anytime/services/audio/default_audio_player_service.dart';
import 'package:flutter_test/flutter_test.dart';

import '../../mocks/mock_settings_service.dart';

/// `DefaultAudioPlayerService.skipActiveAd` updates the running "time saved"
/// and "ads skipped" stats on the [SettingsService]. The bookkeeping lives in
/// the `@visibleForTesting` [recordAdSkipStat] helper, which we exercise here
/// (the surrounding `skipActiveAd` needs the live audio backend).
void main() {
  late MockSettingsService settingsService;

  setUp(() {
    settingsService = MockSettingsService();
  });

  group('recordAdSkipStat', () {
    test('increments adSkipSavedSeconds by the segment length in seconds', () {
      const segment = AdSegment(startMs: 1000, endMs: 4000);

      recordAdSkipStat(settingsService: settingsService, segment: segment);

      expect(settingsService.adSkipSavedSeconds, 3);
    });

    test('increments adSkipCount by one for a non-zero segment', () {
      const segment = AdSegment(startMs: 0, endMs: 5000);

      recordAdSkipStat(settingsService: settingsService, segment: segment);

      expect(settingsService.adSkipCount, 1);
    });

    test('accumulates across multiple skips', () {
      const first = AdSegment(startMs: 1000, endMs: 4000); // 3s
      const second = AdSegment(startMs: 60000, endMs: 75000); // 15s

      recordAdSkipStat(settingsService: settingsService, segment: first);
      recordAdSkipStat(settingsService: settingsService, segment: second);

      expect(settingsService.adSkipSavedSeconds, 18);
      expect(settingsService.adSkipCount, 2);
    });

    test('is a no-op when the segment has no length', () {
      const segment = AdSegment(startMs: 2000, endMs: 2000);

      recordAdSkipStat(settingsService: settingsService, segment: segment);

      expect(settingsService.adSkipSavedSeconds, 0);
      expect(settingsService.adSkipCount, 0);
    });

    test('rounds sub-second segments to whole seconds before tallying', () {
      const segment = AdSegment(startMs: 0, endMs: 1400); // 1.4s -> 1s

      recordAdSkipStat(settingsService: settingsService, segment: segment);

      expect(settingsService.adSkipSavedSeconds, 1);
      expect(settingsService.adSkipCount, 1);
    });
  });
}
