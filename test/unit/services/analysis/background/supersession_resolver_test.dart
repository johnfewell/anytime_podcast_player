// Copyright 2020 Ben Hills and the project contributors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

import 'package:anytime/entities/ad_segment.dart';
import 'package:anytime/entities/episode_analysis_record.dart';
import 'package:anytime/services/analysis/background/supersession_resolver.dart';
import 'package:flutter_test/flutter_test.dart';

EpisodeAnalysisRecord _record({
  required String provider,
  required bool active,
  int completedAtMs = 0,
  List<AdSegment> adSegments = const <AdSegment>[],
  String modelId = '',
}) {
  return EpisodeAnalysisRecord(
    provider: provider,
    modelId: modelId,
    completedAtMs: completedAtMs,
    adSegments: adSegments,
    active: active,
  );
}

void main() {
  group('SupersessionResolver.shouldActivate', () {
    test('first record always becomes active', () {
      final next = _record(provider: AnalysisProvider.legacyUnknown, active: false);
      expect(SupersessionResolver.shouldActivate(null, next), isTrue);
    });

    test('gemini -> whisper+gemma4 activates', () {
      final existing = _record(provider: AnalysisProvider.geminiAudio, active: true);
      final next = _record(provider: AnalysisProvider.whisperGemma4, active: false);
      expect(SupersessionResolver.shouldActivate(existing, next), isTrue);
    });

    test('whisper+gemma4 -> gemini does NOT activate', () {
      final existing = _record(provider: AnalysisProvider.whisperGemma4, active: true);
      final next = _record(provider: AnalysisProvider.geminiAudio, active: false);
      expect(SupersessionResolver.shouldActivate(existing, next), isFalse);
    });

    test('rerun of same provider activates', () {
      final existing = _record(provider: AnalysisProvider.whisperGemma4, active: true);
      final next = _record(provider: AnalysisProvider.whisperGemma4, active: false);
      expect(SupersessionResolver.shouldActivate(existing, next), isTrue);
    });

    test('legacy provider is always replaceable by any new provider', () {
      for (final legacy in <String>[
        AnalysisProvider.openAi,
        AnalysisProvider.grok,
        AnalysisProvider.backend,
        AnalysisProvider.legacyUnknown,
      ]) {
        final existing = _record(provider: legacy, active: true);
        expect(SupersessionResolver.shouldActivate(existing, _record(provider: AnalysisProvider.whisperGemma4, active: false)),
            isTrue,
            reason: '$legacy should be replaced by whisper+gemma4');
        expect(SupersessionResolver.shouldActivate(existing, _record(provider: AnalysisProvider.geminiAudio, active: false)),
            isTrue,
            reason: '$legacy should be replaced by gemini-audio');
      }
    });

    test('whisper+gemma4 active is NOT replaced by legacy providers', () {
      final existing = _record(provider: AnalysisProvider.whisperGemma4, active: true);
      for (final legacy in <String>[
        AnalysisProvider.openAi,
        AnalysisProvider.grok,
        AnalysisProvider.backend,
        AnalysisProvider.legacyUnknown,
      ]) {
        expect(SupersessionResolver.shouldActivate(existing, _record(provider: legacy, active: false)), isFalse,
            reason: 'whisper+gemma4 should not be demoted by $legacy');
      }
    });
  });

  group('SupersessionResolver.resolve', () {
    test('empty existing -> new record is active', () {
      final next = _record(provider: AnalysisProvider.geminiAudio, active: false, completedAtMs: 1);
      final updated = SupersessionResolver.resolve(existing: const [], newRecord: next);
      expect(updated, hasLength(1));
      expect(updated.single.active, isTrue);
      expect(updated.single.provider, AnalysisProvider.geminiAudio);
    });

    test('gemini -> whisper+gemma4: gemini demoted, whisper active, both retained', () {
      final gemini = _record(provider: AnalysisProvider.geminiAudio, active: true, completedAtMs: 100);
      final whisper = _record(provider: AnalysisProvider.whisperGemma4, active: false, completedAtMs: 200);

      final updated = SupersessionResolver.resolve(existing: [gemini], newRecord: whisper);

      expect(updated, hasLength(2));
      expect(updated[0].provider, AnalysisProvider.geminiAudio);
      expect(updated[0].active, isFalse);
      expect(updated[1].provider, AnalysisProvider.whisperGemma4);
      expect(updated[1].active, isTrue);
    });

    test('whisper+gemma4 active -> gemini arrives: whisper stays active, gemini inactive', () {
      final whisper = _record(provider: AnalysisProvider.whisperGemma4, active: true, completedAtMs: 100);
      final gemini = _record(provider: AnalysisProvider.geminiAudio, active: false, completedAtMs: 200);

      final updated = SupersessionResolver.resolve(existing: [whisper], newRecord: gemini);

      expect(updated, hasLength(2));
      expect(updated[0].active, isTrue);
      expect(updated[0].provider, AnalysisProvider.whisperGemma4);
      expect(updated[1].active, isFalse);
      expect(updated[1].provider, AnalysisProvider.geminiAudio);
    });

    test('rerun of same provider demotes the old record and activates the new', () {
      final firstRun = _record(provider: AnalysisProvider.whisperGemma4, active: true, completedAtMs: 100);
      final secondRun = _record(provider: AnalysisProvider.whisperGemma4, active: false, completedAtMs: 200);

      final updated = SupersessionResolver.resolve(existing: [firstRun], newRecord: secondRun);

      expect(updated, hasLength(2));
      expect(updated[0].active, isFalse, reason: 'prior active demoted');
      expect(updated[0].completedAtMs, 100);
      expect(updated[1].active, isTrue);
      expect(updated[1].completedAtMs, 200);
    });

    test('multiple prior records: only the one that was active gets demoted', () {
      final legacyA = _record(provider: AnalysisProvider.legacyUnknown, active: false, completedAtMs: 1);
      final gemini = _record(provider: AnalysisProvider.geminiAudio, active: true, completedAtMs: 2);
      final whisper = _record(provider: AnalysisProvider.whisperGemma4, active: false, completedAtMs: 3);

      final updated = SupersessionResolver.resolve(existing: [legacyA, gemini], newRecord: whisper);

      expect(updated, hasLength(3));
      expect(updated[0], legacyA, reason: 'inactive records are passed through unchanged');
      expect(updated[1].provider, AnalysisProvider.geminiAudio);
      expect(updated[1].active, isFalse);
      expect(updated[2].active, isTrue);
    });

    test('failure record never becomes active even when no prior active exists', () {
      final failure = EpisodeAnalysisRecord(
        provider: AnalysisProvider.grok,
        modelId: 'grok-4.3',
        completedAtMs: 5,
        adSegments: const <AdSegment>[],
        active: false,
        status: 'failed',
        error: 'Grok analysis timed out. Try again.',
      );

      final updated = SupersessionResolver.resolve(existing: const [], newRecord: failure);

      expect(updated, hasLength(1));
      expect(updated.single.active, isFalse, reason: 'failure records are diagnostic-only');
      expect(updated.single.error, 'Grok analysis timed out. Try again.');
    });

    test('same-provider failure does not displace prior successful active record', () {
      final prior = _record(
        provider: AnalysisProvider.grok,
        active: true,
        completedAtMs: 10,
        adSegments: const <AdSegment>[
          AdSegment(startMs: 0, endMs: 1000),
        ],
      );
      final failure = EpisodeAnalysisRecord(
        provider: AnalysisProvider.grok,
        modelId: 'grok-4.3',
        completedAtMs: 20,
        adSegments: const <AdSegment>[],
        active: false,
        status: 'failed',
        error: 'HTTP 429: rate limited',
      );

      final updated = SupersessionResolver.resolve(existing: [prior], newRecord: failure);

      expect(updated, hasLength(2));
      expect(updated[0].active, isTrue, reason: 'prior successful record stays active');
      expect(updated[0].adSegments, hasLength(1));
      expect(updated[1].active, isFalse);
      expect(updated[1].isFailure, isTrue);
    });

    test('returned list is unmodifiable', () {
      final next = _record(provider: AnalysisProvider.geminiAudio, active: false);
      final updated = SupersessionResolver.resolve(existing: const [], newRecord: next);
      expect(() => updated.add(next), throwsUnsupportedError);
    });
  });
}
