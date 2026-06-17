// Copyright 2020 Ben Hills and the project contributors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

import 'package:anytime/bloc/podcast/audio_bloc.dart';
import 'package:anytime/bloc/podcast/episode_bloc.dart';
import 'package:anytime/bloc/podcast/podcast_bloc.dart';
import 'package:anytime/bloc/podcast/queue_bloc.dart';
import 'package:anytime/bloc/settings/settings_bloc.dart';
import 'package:anytime/core/utils.dart';
import 'package:anytime/entities/downloadable.dart';
import 'package:anytime/entities/episode.dart';
import 'package:anytime/l10n/L.dart';
import 'package:anytime/state/queue_event_state.dart';
import 'package:anytime/ui/podcast/episode_details.dart';
import 'package:anytime/ui/podcast/now_playing.dart';
import 'package:anytime/ui/themes.dart';
import 'package:anytime/ui/widgets/tile_image.dart';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

/// Shows the Ambient quick-action sheet for an episode: a tap on an episode
/// raises a focused action menu — Play now, Play next, Add to Up next, Download,
/// Mark as played, Share — with an AI ad-skip heads-up at the top. See the
/// Player Ambient design.
Future<void> showEpisodeActionsSheet(BuildContext context, Episode episode, {bool queued = false}) {
  return showModalBottomSheet<void>(
    context: context,
    barrierLabel: L.of(context)!.scrim_episode_details_selector,
    backgroundColor: Colors.transparent,
    isScrollControlled: true,
    builder: (sheetContext) {
      return EpisodeActionsSheet(episode: episode, queued: queued);
    },
  );
}

class EpisodeActionsSheet extends StatelessWidget {
  final Episode episode;
  final bool queued;

  const EpisodeActionsSheet({
    super.key,
    required this.episode,
    this.queued = false,
  });

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final colorScheme = theme.colorScheme;
    final ambient = AmbientColors.of(context);

    final audioBloc = Provider.of<AudioBloc>(context, listen: false);
    final episodeBloc = Provider.of<EpisodeBloc>(context, listen: false);
    final podcastBloc = Provider.of<PodcastBloc>(context, listen: false);
    final queueBloc = Provider.of<QueueBloc>(context, listen: false);
    final settings = Provider.of<SettingsBloc>(context, listen: false).currentSettings;

    final downloaded = episode.downloaded;
    final downloading = episode.downloadState == DownloadState.downloading ||
        episode.downloadState == DownloadState.queued;

    return SafeArea(
      top: false,
      child: Container(
        margin: const EdgeInsets.fromLTRB(8.0, 0.0, 8.0, 8.0),
        decoration: BoxDecoration(
          color: colorScheme.surface,
          borderRadius: BorderRadius.circular(24.0),
          border: Border.all(color: ambient.hairline),
        ),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            // Drag handle.
            Container(
              width: 40.0,
              height: 5.0,
              margin: const EdgeInsets.fromLTRB(0.0, 12.0, 0.0, 12.0),
              decoration: BoxDecoration(
                color: colorScheme.outlineVariant,
                borderRadius: BorderRadius.circular(3.0),
              ),
            ),
            // Episode header with the ad-skip heads-up.
            Padding(
              padding: const EdgeInsets.fromLTRB(20.0, 0.0, 20.0, 14.0),
              child: Row(
                children: [
                  TileImage(
                    url: episode.thumbImageUrl ?? episode.imageUrl ?? '',
                    size: 50.0,
                    highlight: false,
                  ),
                  const SizedBox(width: 13.0),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          episode.title ?? '',
                          maxLines: 1,
                          overflow: TextOverflow.ellipsis,
                          style: theme.textTheme.titleSmall,
                        ),
                        const SizedBox(height: 4.0),
                        _HeaderSubtitle(episode: episode),
                      ],
                    ),
                  ),
                ],
              ),
            ),
            Divider(height: 1.0, thickness: 1.0, color: ambient.hairline),
            const SizedBox(height: 6.0),
            _ActionRow(
              icon: Icons.play_circle_outline_rounded,
              label: L.of(context)!.play_button_label,
              emphasised: true,
              onTap: () {
                final navigator = Navigator.of(context);
                navigator.pop();
                audioBloc.play(episode);
                if (settings.autoOpenNowPlaying) {
                  navigator.push(
                    MaterialPageRoute<void>(
                      builder: (_) => const NowPlaying(),
                      settings: const RouteSettings(name: 'nowplaying'),
                    ),
                  );
                }
              },
            ),
            _ActionRow(
              icon: Icons.playlist_play_rounded,
              label: 'Play next',
              onTap: () {
                queueBloc.queueEvent(QueueAddEvent(episode: episode, position: 0));
                Navigator.pop(context);
              },
            ),
            if (!queued)
              _ActionRow(
                icon: Icons.playlist_add_rounded,
                label: L.of(context)!.semantics_add_to_queue,
                onTap: () {
                  queueBloc.queueEvent(QueueAddEvent(episode: episode));
                  Navigator.pop(context);
                },
              )
            else
              _ActionRow(
                icon: Icons.playlist_remove_rounded,
                label: L.of(context)!.semantics_remove_from_queue,
                onTap: () {
                  queueBloc.queueEvent(QueueRemoveEvent(episode: episode));
                  Navigator.pop(context);
                },
              ),
            _ActionRow(
              icon: downloaded ? Icons.delete_outline_rounded : Icons.download_rounded,
              label: downloaded
                  ? L.of(context)!.delete_episode_button_label
                  : (downloading
                      ? L.of(context)!.cancel_download_button_label
                      : L.of(context)!.download_episode_button_label),
              trailing: (!downloaded && !downloading && episode.length > 0) ? _formatBytes(episode.length) : null,
              onTap: () {
                if (downloaded || downloading) {
                  episodeBloc.deleteDownload(episode);
                } else {
                  podcastBloc.downloadEpisode(episode);
                }
                Navigator.pop(context);
              },
            ),
            _ActionRow(
              icon: episode.played ? Icons.unpublished_outlined : Icons.check_circle_outline_rounded,
              label: episode.played
                  ? L.of(context)!.semantics_mark_episode_unplayed
                  : L.of(context)!.semantics_mark_episode_played,
              onTap: () {
                episodeBloc.togglePlayed(episode);
                Navigator.pop(context);
              },
            ),
            _ActionRow(
              icon: Icons.ios_share_rounded,
              label: L.of(context)!.share_episode_option_label,
              onTap: () async {
                Navigator.pop(context);
                await shareEpisode(episode: episode);
              },
            ),
            _ActionRow(
              icon: Icons.notes_rounded,
              label: L.of(context)!.episode_details_button_label,
              onTap: () {
                Navigator.pop(context);
                showModalBottomSheet<void>(
                  context: context,
                  barrierLabel: L.of(context)!.scrim_episode_details_selector,
                  backgroundColor: theme.bottomAppBarTheme.color,
                  isScrollControlled: true,
                  shape: const RoundedRectangleBorder(
                    borderRadius: BorderRadius.only(
                      topLeft: Radius.circular(10.0),
                      topRight: Radius.circular(10.0),
                    ),
                  ),
                  builder: (_) => EpisodeDetails(episode: episode),
                );
              },
            ),
            // Cancel.
            Padding(
              padding: const EdgeInsets.fromLTRB(16.0, 10.0, 16.0, 14.0),
              child: SizedBox(
                width: double.infinity,
                height: 50.0,
                child: TextButton(
                  style: TextButton.styleFrom(
                    backgroundColor: ambient.controlFill,
                    foregroundColor: colorScheme.onSurfaceVariant,
                    shape: const StadiumBorder(),
                  ),
                  onPressed: () => Navigator.pop(context),
                  child: Text(
                    L.of(context)!.close_button_label,
                    style: theme.textTheme.titleSmall?.copyWith(color: colorScheme.onSurfaceVariant),
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  String _formatBytes(int bytes) {
    if (bytes >= 1024 * 1024 * 1024) {
      return '${(bytes / (1024 * 1024 * 1024)).toStringAsFixed(1)} GB';
    }
    if (bytes >= 1024 * 1024) {
      return '${(bytes / (1024 * 1024)).round()} MB';
    }
    return '${(bytes / 1024).round()} KB';
  }
}

/// Header line: the AI ad-skip heads-up when ads are detected, otherwise the
/// usual podcast / duration subtitle.
class _HeaderSubtitle extends StatelessWidget {
  final Episode episode;

  const _HeaderSubtitle({required this.episode});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final ambient = AmbientColors.of(context);

    if (episode.adSegments.isNotEmpty) {
      final count = episode.adSegments.length;
      final totalSeconds =
          episode.adSegments.fold<int>(0, (sum, s) => sum + ((s.endMs - s.startMs) / 1000).round());

      return Row(
        children: [
          Icon(Icons.auto_awesome, size: 13.0, color: ambient.aiTeal),
          const SizedBox(width: 6.0),
          Flexible(
            child: Text(
              '$count ${count == 1 ? 'ad' : 'ads'} · ${_formatDuration(totalSeconds)} will be skipped',
              overflow: TextOverflow.ellipsis,
              style: theme.textTheme.labelMedium?.copyWith(
                color: ambient.aiTeal,
                fontWeight: FontWeight.w600,
              ),
            ),
          ),
        ],
      );
    }

    final podcast = episode.podcast?.trim() ?? '';
    return Text(
      podcast,
      maxLines: 1,
      overflow: TextOverflow.ellipsis,
      style: theme.textTheme.bodySmall?.copyWith(color: theme.colorScheme.onSurfaceVariant),
    );
  }

  String _formatDuration(int totalSeconds) {
    final minutes = totalSeconds ~/ 60;
    final seconds = totalSeconds % 60;
    if (minutes > 0) {
      return '${minutes}m ${seconds.toString().padLeft(2, '0')}s';
    }
    return '${seconds}s';
  }
}

class _ActionRow extends StatelessWidget {
  final IconData icon;
  final String label;
  final String? trailing;
  final bool emphasised;
  final VoidCallback onTap;

  const _ActionRow({
    required this.icon,
    required this.label,
    required this.onTap,
    this.trailing,
    this.emphasised = false,
  });

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final colorScheme = theme.colorScheme;
    final color = emphasised ? colorScheme.primary : colorScheme.onSurface;

    return Material(
      color: Colors.transparent,
      child: InkWell(
        onTap: onTap,
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 22.0, vertical: 13.0),
          child: Row(
            children: [
              Icon(icon, size: 22.0, color: emphasised ? colorScheme.primary : colorScheme.onSurfaceVariant),
              const SizedBox(width: 15.0),
              Expanded(
                child: Text(
                  label,
                  style: theme.textTheme.titleMedium?.copyWith(
                    color: color,
                    fontWeight: emphasised ? FontWeight.w600 : FontWeight.w500,
                  ),
                ),
              ),
              if (trailing != null)
                Text(
                  trailing!,
                  style: theme.textTheme.bodySmall?.copyWith(color: colorScheme.outline),
                ),
            ],
          ),
        ),
      ),
    );
  }
}
