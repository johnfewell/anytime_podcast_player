// Copyright 2020 Ben Hills and the project contributors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

import 'package:anytime/bloc/podcast/audio_bloc.dart';
import 'package:anytime/entities/app_settings.dart';
import 'package:anytime/entities/downloadable.dart';
import 'package:anytime/entities/episode.dart';
import 'package:anytime/l10n/L.dart';
import 'package:anytime/services/audio/audio_player_service.dart';
import 'package:anytime/ui/podcast/episode_actions_sheet.dart';
import 'package:anytime/ui/podcast/now_playing.dart';
import 'package:anytime/ui/podcast/transport_controls.dart';
import 'package:anytime/ui/widgets/expressive_linear_progress_indicator.dart';
import 'package:anytime/ui/widgets/tile_image.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:intl/intl.dart' show DateFormat;
import 'package:provider/provider.dart';
import 'package:rxdart/rxdart.dart';

/// This class builds a tile for each episode in the podcast feed.
class EpisodeTile extends StatelessWidget {
  final Episode episode;
  final bool download;
  final bool play;
  final bool playing;
  final bool queued;

  const EpisodeTile({
    super.key,
    required this.episode,
    required this.download,
    required this.play,
    this.playing = false,
    this.queued = false,
  });

  @override
  Widget build(BuildContext context) {
    final screenReaderEnabled = MediaQuery.accessibleNavigationOf(context);

    if (screenReaderEnabled) {
      if (defaultTargetPlatform == TargetPlatform.iOS) {
        return _CupertinoAccessibleEpisodeTile(
          episode: episode,
          download: download,
          play: play,
          playing: playing,
          queued: queued,
        );
      } else {
        return _AndroidAccessibleEpisodeTile(
          episode: episode,
          download: download,
          play: play,
          playing: playing,
          queued: queued,
        );
      }
    } else {
      return ExpandableEpisodeTile(
        episode: episode,
        download: download,
        play: play,
        playing: playing,
        queued: queued,
      );
    }
  }
}

/// An EpisodeTitle is built with an [ExpansionTile] widget and displays the episode's
/// basic details, thumbnail and play button.
///
/// It can then be expanded to present addition information about the episode and further
/// controls.
///
/// TODO: Replace [Opacity] with [Container] with a transparent colour.
class ExpandableEpisodeTile extends StatefulWidget {
  final Episode episode;
  final bool download;
  final bool play;
  final bool playing;
  final bool queued;

  const ExpandableEpisodeTile({
    super.key,
    required this.episode,
    required this.download,
    required this.play,
    this.playing = false,
    this.queued = false,
  });

  @override
  State<ExpandableEpisodeTile> createState() => _ExpandableEpisodeTileState();
}

class _ExpandableEpisodeTileState extends State<ExpandableEpisodeTile> {
  @override
  Widget build(BuildContext context) {
    final textTheme = Theme.of(context).textTheme;

    return ListTile(
      contentPadding: const EdgeInsets.fromLTRB(16.0, 0.0, 8.0, 0.0),
      key: Key('PT${widget.episode.guid}'),
      onTap: () {
        showEpisodeActionsSheet(context, widget.episode, queued: widget.queued);
      },
      trailing: Opacity(
        opacity: widget.episode.played ? 0.5 : 1.0,
        child: EpisodeTransportControls(
          episode: widget.episode,
          download: widget.download,
          play: widget.play,
        ),
      ),
      leading: ExcludeSemantics(
        child: Stack(
          alignment: Alignment.bottomLeft,
          fit: StackFit.passthrough,
          children: <Widget>[
            Opacity(
              opacity: widget.episode.played ? 0.5 : 1.0,
              child: TileImage(
                url: widget.episode.thumbImageUrl ?? widget.episode.imageUrl!,
                size: 56.0,
                highlight: widget.episode.highlight,
              ),
            ),
            SizedBox(
              height: 5.0,
              width: 56.0 * (widget.episode.percentagePlayed / 100),
              child: Container(
                color: Theme.of(context).colorScheme.primary,
              ),
            ),
          ],
        ),
      ),
      subtitle: Opacity(
        opacity: widget.episode.played ? 0.5 : 1.0,
        child: EpisodeTileSubtitle(widget.episode),
      ),
      title: Opacity(
        opacity: widget.episode.played ? 0.5 : 1.0,
        child: Text(
          widget.episode.title!,
          overflow: TextOverflow.ellipsis,
          maxLines: 2,
          softWrap: false,
          style: textTheme.bodyMedium,
        ),
      ),
    );
  }
}

/// This is an accessible version of the episode tile that uses Apple theming.
/// When the tile is tapped, an iOS menu will appear with the relevant options.
class _CupertinoAccessibleEpisodeTile extends StatefulWidget {
  final Episode episode;
  final bool download;
  final bool play;
  final bool playing;
  final bool queued;

  const _CupertinoAccessibleEpisodeTile({
    required this.episode,
    required this.download,
    required this.play,
    this.playing = false,
    this.queued = false,
  });

  @override
  State<_CupertinoAccessibleEpisodeTile> createState() => _CupertinoAccessibleEpisodeTileState();
}

class _CupertinoAccessibleEpisodeTileState extends State<_CupertinoAccessibleEpisodeTile> {
  bool expanded = false;

  @override
  Widget build(BuildContext context) {
    final textTheme = Theme.of(context).textTheme;
    final audioBloc = Provider.of<AudioBloc>(context, listen: false);

    return StreamBuilder<_PlayerControlState>(
        stream: Rx.combineLatest2(audioBloc.playingState!, audioBloc.nowPlaying!,
            (AudioState audioState, Episode? episode) => _PlayerControlState(audioState, episode)),
        builder: (context, snapshot) {
          if (!snapshot.hasData) {
            return Container();
          }

          return Semantics(
            button: true,
            child: ListTile(
              key: Key('PT${widget.episode.guid}'),
              leading: ExcludeSemantics(
                child: Stack(
                  alignment: Alignment.bottomLeft,
                  fit: StackFit.passthrough,
                  children: <Widget>[
                    Opacity(
                      opacity: widget.episode.played ? 0.5 : 1.0,
                      child: TileImage(
                        url: widget.episode.thumbImageUrl ?? widget.episode.imageUrl!,
                        size: 56.0,
                        highlight: widget.episode.highlight,
                      ),
                    ),
                    SizedBox(
                      height: 5.0,
                      width: 56.0 * (widget.episode.percentagePlayed / 100),
                      child: Container(
                        color: Theme.of(context).colorScheme.primary,
                      ),
                    ),
                  ],
                ),
              ),
              subtitle: Opacity(
                opacity: widget.episode.played ? 0.5 : 1.0,
                child: EpisodeTileSubtitle(widget.episode),
              ),
              title: Opacity(
                opacity: widget.episode.played ? 0.5 : 1.0,
                child: Text(
                  widget.episode.title!,
                  overflow: TextOverflow.ellipsis,
                  maxLines: 2,
                  softWrap: false,
                  style: textTheme.bodyMedium,
                ),
              ),
              onTap: () {
                showEpisodeActionsSheet(context, widget.episode, queued: widget.queued);
              },
            ),
          );
        });
  }

  // /// If we have the 'show now playing upon play' option set to true, launch
  // /// the [NowPlaying] widget automatically.
  void optionalShowNowPlaying(BuildContext context, AppSettings settings) {
    if (settings.autoOpenNowPlaying) {
      Navigator.push(
        context,
        MaterialPageRoute<void>(
          builder: (context) => const NowPlaying(),
          settings: const RouteSettings(name: 'nowplaying'),
          fullscreenDialog: false,
        ),
      );
    }
  }
}

/// This is an accessible version of the episode tile that uses Android theming.
/// When the tile is tapped, an Android dialog menu will appear with the relevant
/// options.
class _AndroidAccessibleEpisodeTile extends StatefulWidget {
  final Episode episode;
  final bool download;
  final bool play;
  final bool playing;
  final bool queued;

  const _AndroidAccessibleEpisodeTile({
    required this.episode,
    required this.download,
    required this.play,
    this.playing = false,
    this.queued = false,
  });

  @override
  State<_AndroidAccessibleEpisodeTile> createState() => _AndroidAccessibleEpisodeTileState();
}

class _AndroidAccessibleEpisodeTileState extends State<_AndroidAccessibleEpisodeTile> {
  @override
  Widget build(BuildContext context) {
    final textTheme = Theme.of(context).textTheme;
    final audioBloc = Provider.of<AudioBloc>(context, listen: false);

    return StreamBuilder<_PlayerControlState>(
        stream: Rx.combineLatest2(audioBloc.playingState!, audioBloc.nowPlaying!,
            (AudioState audioState, Episode? episode) => _PlayerControlState(audioState, episode)),
        builder: (context, snapshot) {
          if (!snapshot.hasData) {
            return Container();
          }

          return ListTile(
            key: Key('PT${widget.episode.guid}'),
            onTap: () {
              showEpisodeActionsSheet(context, widget.episode, queued: widget.queued);
            },
            leading: ExcludeSemantics(
              child: Stack(
                alignment: Alignment.bottomLeft,
                fit: StackFit.passthrough,
                children: <Widget>[
                  Opacity(
                    opacity: widget.episode.played ? 0.5 : 1.0,
                    child: TileImage(
                      url: widget.episode.thumbImageUrl ?? widget.episode.imageUrl!,
                      size: 56.0,
                      highlight: widget.episode.highlight,
                    ),
                  ),
                  SizedBox(
                    height: 5.0,
                    width: 56.0 * (widget.episode.percentagePlayed / 100),
                    child: Container(
                      color: Theme.of(context).colorScheme.primary,
                    ),
                  ),
                ],
              ),
            ),
            subtitle: Opacity(
              opacity: widget.episode.played ? 0.5 : 1.0,
              child: EpisodeTileSubtitle(widget.episode),
            ),
            title: Opacity(
              opacity: widget.episode.played ? 0.5 : 1.0,
              child: Text(
                widget.episode.title!,
                overflow: TextOverflow.ellipsis,
                maxLines: 2,
                softWrap: false,
                style: textTheme.bodyMedium,
              ),
            ),
          );
        });
  }

  /// If we have the 'show now playing upon play' option set to true, launch
  /// the [NowPlaying] widget automatically.
  void optionalShowNowPlaying(BuildContext context, AppSettings settings) {
    if (settings.autoOpenNowPlaying) {
      Navigator.push(
        context,
        MaterialPageRoute<void>(
          builder: (context) => const NowPlaying(),
          settings: const RouteSettings(name: 'nowplaying'),
          fullscreenDialog: false,
        ),
      );
    }
  }
}

class EpisodeTransportControls extends StatelessWidget {
  final Episode episode;
  final bool download;
  final bool play;

  const EpisodeTransportControls({
    super.key,
    required this.episode,
    required this.download,
    required this.play,
  });

  @override
  Widget build(BuildContext context) {
    final buttons = <Widget>[];

    if (download) {
      buttons.add(Semantics(
        container: true,
        child: DownloadControl(
          episode: episode,
        ),
      ));
    }

    if (play) {
      buttons.add(Semantics(
        container: true,
        child: PlayControl(
          episode: episode,
        ),
      ));
    }

    return SizedBox(
      width: (buttons.length * 48.0),
      child: Row(
        children: <Widget>[...buttons],
      ),
    );
  }
}

class EpisodeTileSubtitle extends StatelessWidget {
  final Episode episode;

  const EpisodeTileSubtitle(this.episode, {super.key});

  @override
  Widget build(BuildContext context) {
    final isActiveDownload =
        episode.downloadState == DownloadState.queued || episode.downloadState == DownloadState.downloading;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      mainAxisSize: MainAxisSize.min,
      children: [
        EpisodeSubtitle(episode),
        if (isActiveDownload) _EpisodeDownloadProgress(episode: episode),
      ],
    );
  }
}

class _EpisodeDownloadProgress extends StatelessWidget {
  final Episode episode;

  const _EpisodeDownloadProgress({
    required this.episode,
  });

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final percentage = (episode.downloadPercentage ?? 0).clamp(0, 100);
    final showDeterminateProgress = episode.downloadState == DownloadState.downloading && percentage > 0;

    return Padding(
      padding: const EdgeInsets.only(top: 6.0),
      child: Row(
        children: [
          Expanded(
            child: ClipRRect(
              borderRadius: BorderRadius.circular(999.0),
              child: showDeterminateProgress
                  ? ExpressiveLinearProgressIndicator(
                      value: percentage / 100,
                      minHeight: 5.0,
                      animated: true,
                    )
                  : const LinearProgressIndicator(minHeight: 5.0),
            ),
          ),
          if (episode.downloadState == DownloadState.downloading) ...[
            const SizedBox(width: 8.0),
            Text(
              '$percentage%',
              style: theme.textTheme.bodySmall?.copyWith(
                color: theme.colorScheme.primary,
                fontWeight: FontWeight.w600,
              ),
            ),
          ],
        ],
      ),
    );
  }
}

/// This class builds the subtitle line for an episode. This consists of the publication date,
/// episode length, time remaining (if episode has been started) and file size.
class EpisodeSubtitle extends StatelessWidget {
  final Episode episode;
  final String date;
  final Duration length;

  EpisodeSubtitle(this.episode, {super.key})
      : date = episode.publicationDate == null
            ? ''
            : DateFormat(episode.publicationDate!.year == DateTime.now().year ? 'd MMM' : 'd MMM yyyy')
                .format(episode.publicationDate!),
        length = Duration(seconds: episode.duration);

  @override
  Widget build(BuildContext context) {
    final textTheme = Theme.of(context).textTheme;
    var timeRemaining = episode.timeRemaining;
    var dateLabel = date;
    var dateSemanticLabel = date;

    String title;
    String semanticTitle;

    // If publication is within 7 days, give friendlier date format.
    if (episode.publicationDate != null) {
      final now = DateTime.now();
      final diff = now.difference(episode.publicationDate!);

      if (diff.inDays < 7) {
        (dateLabel, dateSemanticLabel) = calculateTimeAgo(context, episode.publicationDate!, now);
      }
    }

    if (length.inSeconds > 0) {
      if (length.inSeconds < 60) {
        title = '$dateLabel • ${L.of(context)!.time_seconds(length.inSeconds)}';
        semanticTitle = '$dateSemanticLabel, ${L.of(context)!.time_semantic_seconds(length.inSeconds)}';
      } else {
        title = '$dateLabel • ${L.of(context)!.time_minutes(length.inMinutes)}';
        semanticTitle = '$dateSemanticLabel, ${L.of(context)!.time_semantic_minutes(length.inMinutes)}';
      }
    } else {
      title = dateLabel;
      semanticTitle = dateLabel;
    }

    if (timeRemaining.inSeconds > 0) {
      if (timeRemaining.inSeconds < 60) {
        title = '$title / ${L.of(context)!.episode_time_second_remaining(timeRemaining.inSeconds.toString())}';
        semanticTitle =
            '$semanticTitle / ${L.of(context)!.episode_semantic_time_second_remaining(timeRemaining.inSeconds.toString())}';
      } else {
        title = '$title / ${L.of(context)!.episode_time_minute_remaining(timeRemaining.inMinutes.toString())}';
        semanticTitle =
            '$semanticTitle / ${L.of(context)!.episode_semantic_time_minute_remaining(timeRemaining.inMinutes.toString())}';
      }
    }

    if (episode.length > 0) {
      final mb = (episode.length / (1024 * 1024)).toStringAsFixed(1);

      title = '$title • $mb${L.of(context)!.label_megabytes_abbr}';
      semanticTitle = '$semanticTitle, $mb ${L.of(context)!.label_megabytes}';
    }

    return Padding(
      padding: const EdgeInsets.only(top: 4.0),
      child: Text(
        title,
        semanticsLabel: semanticTitle,
        overflow: TextOverflow.ellipsis,
        softWrap: false,
        style: textTheme.bodySmall,
      ),
    );
  }

  (String, String) calculateTimeAgo(BuildContext context, DateTime d, DateTime n) {
    final difference = n.difference(d);
    var label = '';
    var semanticLabel = '';

    if ((difference.inDays / 7).floor() >= 1) {
      label = L.of(context)!.episode_time_weeks_ago(1);
      semanticLabel = L.of(context)!.episode_semantic_time_weeks_ago(1);
    } else if (difference.inDays >= 1) {
      label = L.of(context)!.episode_time_days_ago(difference.inDays);
      semanticLabel = L.of(context)!.episode_semantic_time_days_ago(difference.inDays);
    } else if (difference.inHours >= 1) {
      label = L.of(context)!.episode_time_hours_ago(difference.inHours);
      semanticLabel = L.of(context)!.episode_semantic_time_hours_ago(difference.inHours);
    } else if (difference.inMinutes >= 1) {
      label = L.of(context)!.episode_time_minutes_ago(difference.inMinutes);
      semanticLabel = L.of(context)!.episode_semantic_time_minutes_ago(difference.inMinutes);
    } else {
      label = L.of(context)!.episode_time_now;
      semanticLabel = L.of(context)!.episode_time_now;
    }

    return (label, semanticLabel);
  }
}

/// This class acts as a wrapper between the current audio state and
/// downloadables. Saves all that nesting of StreamBuilders.
class _PlayerControlState {
  final AudioState audioState;
  final Episode? episode;

  _PlayerControlState(this.audioState, this.episode);
}
