// Copyright 2020 Ben Hills and the project contributors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

import 'dart:io';

import 'package:anytime/services/transcription/episode_transcription_service.dart';
import 'package:anytime/services/transcription/moonshine_episode_transcription_service.dart';
import 'package:anytime/services/transcription/whisper_episode_transcription_service.dart';
import 'package:flutter/foundation.dart';
import 'package:logging/logging.dart';
import 'package:path/path.dart' as path;
import 'package:path_provider/path_provider.dart';

/// Dev-only flag. Touch `<app-support>/use_moonshine` to flip local
/// transcription from Whisper to Moonshine (sherpa_onnx v2 int8). No UI —
/// prototype only. Remove the file to revert.
const _moonshineFlagName = 'use_moonshine';

Future<EpisodeTranscriptionService> buildLocalTranscriptionService() async {
  final log = Logger('LocalTranscriptionEngine');

  if (kIsWeb || !(Platform.isAndroid || Platform.isIOS || Platform.isMacOS)) {
    return DisabledEpisodeTranscriptionService();
  }

  try {
    final supportDir = await getApplicationSupportDirectory();
    final flag = File(path.join(supportDir.path, _moonshineFlagName));
    if (flag.existsSync()) {
      log.info('use_moonshine flag found; selecting Moonshine');
      return MoonshineEpisodeTranscriptionService();
    }
  } catch (e, stackTrace) {
    log.warning(
      'Failed to check Moonshine flag, defaulting to Whisper',
      e,
      stackTrace,
    );
  }

  return WhisperEpisodeTranscriptionService();
}
