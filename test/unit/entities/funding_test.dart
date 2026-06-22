// Copyright 2020 Ben Hills and the project contributors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

import 'package:anytime/entities/funding.dart';
import 'package:flutter_test/flutter_test.dart';

/// Covers the Funding link data model (feature F-FUND-01).
void main() {
  group('Funding construction', () {
    test('upgrades an http funding URL to https', () {
      final funding = Funding(url: 'http://patreon.com/show', value: 'Support us');

      expect(funding.url, 'https://patreon.com/show');
      expect(funding.value, 'Support us');
    });

    test('leaves an https funding URL untouched', () {
      final funding = Funding(url: 'https://buymeacoffee.com/show', value: 'Coffee');

      expect(funding.url, 'https://buymeacoffee.com/show');
    });
  });

  group('Funding serialization', () {
    test('round-trips through toMap/fromMap', () {
      final funding = Funding(url: 'https://example.com/donate', value: 'Donate');

      final restored = Funding.fromMap(funding.toMap());

      expect(restored.url, 'https://example.com/donate');
      expect(restored.value, 'Donate');
    });

    test('fromMap re-applies the https upgrade to stored http URLs', () {
      final restored = Funding.fromMap(<String, dynamic>{
        'url': 'http://example.com/donate',
        'value': 'Donate',
      });

      expect(restored.url, 'https://example.com/donate');
    });
  });
}
