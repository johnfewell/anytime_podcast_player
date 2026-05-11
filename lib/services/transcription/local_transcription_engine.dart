// Copyright 2020 Ben Hills and the project contributors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

import 'dart:io';

import 'package:anytime/entities/app_settings.dart';
import 'package:anytime/services/transcription/episode_transcription_service.dart';
import 'package:anytime/services/transcription/moonshine_episode_transcription_service.dart';
import 'package:anytime/services/transcription/whisper_episode_transcription_service.dart';
import 'package:flutter/foundation.dart';

Future<EpisodeTranscriptionService> buildLocalTranscriptionService(
  TranscriptionProvider provider, {
  int Function()? moonshineChunkSeconds,
}) async {
  if (kIsWeb || !(Platform.isAndroid || Platform.isIOS || Platform.isMacOS)) {
    return DisabledEpisodeTranscriptionService();
  }

  if (provider == TranscriptionProvider.moonshine) {
    return MoonshineEpisodeTranscriptionService(
      chunkSeconds: moonshineChunkSeconds,
    );
  }

  return WhisperEpisodeTranscriptionService();
}
