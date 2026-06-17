// Copyright 2020 Ben Hills and the project contributors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

import 'dart:async';

import 'package:anytime/bloc/podcast/audio_bloc.dart';
import 'package:anytime/bloc/settings/settings_bloc.dart';
import 'package:anytime/entities/ad_segment.dart';
import 'package:anytime/state/ad_skip_state.dart';
import 'package:anytime/ui/themes.dart';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

/// Wraps the player body and surfaces the signature AI ad-skip moment.
///
/// When an [AdSkipPromptState] is emitted, an elevated "Ambient" card rises over
/// a soft scrim: an AI-detected badge, the detected segment length, and a
/// countdown ring that auto-skips when it reaches zero (or the listener can tap
/// *Skip ad* / *Keep playing*). After a skip a brief confirmation toast appears.
///
/// This replaces the previous plain "Ad detected → Skip" snackbar. See the
/// Player Ambient design.
class AdSkipListener extends StatefulWidget {
  final Widget child;

  const AdSkipListener({
    super.key,
    required this.child,
  });

  @override
  State<AdSkipListener> createState() => _AdSkipListenerState();
}

class _AdSkipListenerState extends State<AdSkipListener> {
  StreamSubscription<AdSkipState>? _subscription;

  /// The ad segment currently being prompted, or null when no card is showing.
  AdSegment? _segment;

  /// Remaining whole seconds on the auto-skip countdown. -1 means no countdown
  /// (the listener configured 0s — wait for an explicit choice).
  int _countdown = 0;

  /// The countdown length the card started with (drives the ring fraction).
  int _countdownTotal = 0;

  Timer? _countdownTimer;

  /// The most recently skipped segment length, surfaced by the toast.
  int? _toastSeconds;
  Timer? _toastTimer;

  @override
  void initState() {
    super.initState();
    final audioBloc = Provider.of<AudioBloc>(context, listen: false);

    _subscription = audioBloc.adSkipEvents?.listen((event) {
      if (!mounted) {
        return;
      }

      if (event is AdSkipClearedState) {
        _dismissCard();
        return;
      }

      _showPrompt(event.segment);
    });
  }

  void _showPrompt(AdSegment segment) {
    _countdownTimer?.cancel();

    final countdown = _configuredCountdownSeconds();

    setState(() {
      _segment = segment;
      _countdownTotal = countdown;
      _countdown = countdown > 0 ? countdown : -1;
    });

    if (countdown > 0) {
      _countdownTimer = Timer.periodic(const Duration(seconds: 1), (timer) {
        if (!mounted) {
          timer.cancel();
          return;
        }

        final next = _countdown - 1;
        if (next <= 0) {
          timer.cancel();
          _skip();
        } else {
          setState(() => _countdown = next);
        }
      });
    }
  }

  void _dismissCard() {
    _countdownTimer?.cancel();
    if (_segment != null) {
      setState(() => _segment = null);
    }
  }

  void _skip() {
    final audioBloc = Provider.of<AudioBloc>(context, listen: false);
    final segment = _segment;

    audioBloc.skipActiveAd();
    _dismissCard();

    if (segment != null && _notifyOnSkip()) {
      _showToast(_segmentSeconds(segment));
    }
  }

  void _keepPlaying() {
    // Leaving the card simply dismisses it; the service has already recorded the
    // prompt for this segment and won't re-prompt for it.
    _dismissCard();
  }

  void _showToast(int seconds) {
    _toastTimer?.cancel();
    setState(() => _toastSeconds = seconds);
    _toastTimer = Timer(const Duration(milliseconds: 2400), () {
      if (mounted) {
        setState(() => _toastSeconds = null);
      }
    });
  }

  int _configuredCountdownSeconds() {
    final settings = _settings;
    return settings?.adSkipCountdownSeconds ?? 3;
  }

  bool _notifyOnSkip() {
    final settings = _settings;
    return settings?.adSkipNotify ?? true;
  }

  /// Reads the current settings if a [SettingsBloc] is available above us,
  /// falling back to sensible defaults (used in isolated tests).
  dynamic get _settings {
    try {
      return Provider.of<SettingsBloc>(context, listen: false).currentSettings;
    } catch (_) {
      return null;
    }
  }

  int _segmentSeconds(AdSegment segment) => ((segment.endMs - segment.startMs) / 1000).round();

  @override
  void dispose() {
    _subscription?.cancel();
    _countdownTimer?.cancel();
    _toastTimer?.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final segment = _segment;
    final toastSeconds = _toastSeconds;

    return Stack(
      children: [
        Positioned.fill(child: widget.child),
        if (segment != null)
          Positioned.fill(
            child: _AdSkipScrim(
              child: _AdSkipCard(
                seconds: _segmentSeconds(segment),
                countdown: _countdown,
                countdownTotal: _countdownTotal,
                onSkip: _skip,
                onKeep: _keepPlaying,
              ),
            ),
          ),
        if (toastSeconds != null)
          Positioned(
            top: 18.0,
            left: 0.0,
            right: 0.0,
            child: Center(child: _SkippedToast(seconds: toastSeconds)),
          ),
      ],
    );
  }
}

/// A soft, bottom-weighted scrim that dims the player behind the ad card.
class _AdSkipScrim extends StatelessWidget {
  final Widget child;

  const _AdSkipScrim({required this.child});

  @override
  Widget build(BuildContext context) {
    final scrim = Theme.of(context).colorScheme.scrim;

    return IgnorePointer(
      ignoring: false,
      child: DecoratedBox(
        decoration: BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
            stops: const [0.3, 0.6, 1.0],
            colors: [
              scrim.withValues(alpha: 0.0),
              scrim.withValues(alpha: 0.18),
              scrim.withValues(alpha: 0.42),
            ],
          ),
        ),
        child: SafeArea(
          child: Align(
            alignment: Alignment.bottomCenter,
            child: Padding(
              padding: const EdgeInsets.fromLTRB(16.0, 0.0, 16.0, 28.0),
              child: child,
            ),
          ),
        ),
      ),
    );
  }
}

class _AdSkipCard extends StatelessWidget {
  final int seconds;
  final int countdown;
  final int countdownTotal;
  final VoidCallback onSkip;
  final VoidCallback onKeep;

  const _AdSkipCard({
    required this.seconds,
    required this.countdown,
    required this.countdownTotal,
    required this.onSkip,
    required this.onKeep,
  });

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final colorScheme = theme.colorScheme;
    final ambient = AmbientColors.of(context);

    return TweenAnimationBuilder<double>(
      duration: const Duration(milliseconds: 280),
      curve: Curves.easeOutCubic,
      tween: Tween<double>(begin: 0.0, end: 1.0),
      builder: (context, t, child) {
        return Opacity(
          opacity: t,
          child: Transform.translate(offset: Offset(0, 24 * (1 - t)), child: child),
        );
      },
      child: Container(
        constraints: const BoxConstraints(maxWidth: 460.0),
        padding: const EdgeInsets.fromLTRB(22.0, 22.0, 22.0, 20.0),
        decoration: BoxDecoration(
          color: colorScheme.surface,
          borderRadius: BorderRadius.circular(24.0),
          border: Border.all(color: ambient.hairline),
          boxShadow: [
            BoxShadow(
              color: colorScheme.shadow.withValues(alpha: 0.18),
              blurRadius: 40.0,
              offset: const Offset(0, 18),
            ),
          ],
        ),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                const AiDetectedBadge(),
                const SizedBox(width: 9.0),
                Flexible(
                  child: Text(
                    'Sponsor break · ${seconds}s',
                    overflow: TextOverflow.ellipsis,
                    style: theme.textTheme.bodyMedium?.copyWith(
                      color: colorScheme.outline,
                      fontFeatures: const [FontFeature.tabularFigures()],
                    ),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 14.0),
            Text('Ad coming up', style: theme.textTheme.titleLarge),
            const SizedBox(height: 4.0),
            Text(
              'Skipping ahead to the conversation.',
              style: theme.textTheme.bodyMedium?.copyWith(color: colorScheme.onSurfaceVariant),
            ),
            const SizedBox(height: 18.0),
            Row(
              children: [
                Expanded(
                  child: _SkipAdButton(
                    countdown: countdown,
                    countdownTotal: countdownTotal,
                    onPressed: onSkip,
                  ),
                ),
                const SizedBox(width: 12.0),
                _KeepPlayingButton(onPressed: onKeep),
              ],
            ),
          ],
        ),
      ),
    );
  }
}

/// The teal "AI detected" pill used across the player surfaces.
class AiDetectedBadge extends StatelessWidget {
  final String label;

  const AiDetectedBadge({super.key, this.label = 'AI detected'});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final ambient = AmbientColors.of(context);

    return Container(
      height: 25.0,
      padding: const EdgeInsets.symmetric(horizontal: 11.0),
      decoration: BoxDecoration(
        color: ambient.aiTealSurface,
        borderRadius: BorderRadius.circular(999.0),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(Icons.auto_awesome, size: 12.0, color: ambient.aiTeal),
          const SizedBox(width: 6.0),
          Text(
            label.toUpperCase(),
            style: theme.textTheme.labelSmall?.copyWith(
              color: ambient.aiTeal,
              fontWeight: FontWeight.w700,
              letterSpacing: 0.4,
            ),
          ),
        ],
      ),
    );
  }
}

class _SkipAdButton extends StatelessWidget {
  final int countdown;
  final int countdownTotal;
  final VoidCallback onPressed;

  const _SkipAdButton({
    required this.countdown,
    required this.countdownTotal,
    required this.onPressed,
  });

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final colorScheme = theme.colorScheme;
    final showRing = countdown > 0 && countdownTotal > 0;

    return SizedBox(
      height: 50.0,
      child: FilledButton(
        onPressed: onPressed,
        style: FilledButton.styleFrom(
          backgroundColor: colorScheme.primary,
          foregroundColor: colorScheme.onPrimary,
          shape: const StadiumBorder(),
          padding: const EdgeInsets.symmetric(horizontal: 18.0),
        ),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.center,
          mainAxisSize: MainAxisSize.min,
          children: [
            if (showRing) ...[
              _CountdownRing(
                value: countdown / countdownTotal,
                label: '$countdown',
                color: colorScheme.onPrimary,
              ),
              const SizedBox(width: 10.0),
            ],
            const Text('Skip ad'),
          ],
        ),
      ),
    );
  }
}

/// A small ring with the remaining countdown number in its centre.
class _CountdownRing extends StatelessWidget {
  final double value;
  final String label;
  final Color color;

  const _CountdownRing({
    required this.value,
    required this.label,
    required this.color,
  });

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      width: 24.0,
      height: 24.0,
      child: Stack(
        alignment: Alignment.center,
        children: [
          TweenAnimationBuilder<double>(
            duration: const Duration(milliseconds: 900),
            curve: Curves.linear,
            tween: Tween<double>(begin: value, end: value),
            builder: (context, v, _) {
              return SizedBox(
                width: 24.0,
                height: 24.0,
                child: CircularProgressIndicator(
                  value: v,
                  strokeWidth: 2.4,
                  strokeCap: StrokeCap.round,
                  backgroundColor: color.withValues(alpha: 0.3),
                  valueColor: AlwaysStoppedAnimation<Color>(color),
                ),
              );
            },
          ),
          Text(
            label,
            style: TextStyle(
              color: color,
              fontWeight: FontWeight.w700,
              fontSize: 11.0,
              height: 1.0,
            ),
          ),
        ],
      ),
    );
  }
}

class _KeepPlayingButton extends StatelessWidget {
  final VoidCallback onPressed;

  const _KeepPlayingButton({required this.onPressed});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final colorScheme = theme.colorScheme;
    final ambient = AmbientColors.of(context);

    return SizedBox(
      height: 50.0,
      child: OutlinedButton(
        onPressed: onPressed,
        style: OutlinedButton.styleFrom(
          foregroundColor: colorScheme.onSurfaceVariant,
          backgroundColor: colorScheme.surface,
          side: BorderSide(color: ambient.hairline),
          shape: const StadiumBorder(),
          padding: const EdgeInsets.symmetric(horizontal: 20.0),
        ),
        child: const Text('Keep playing'),
      ),
    );
  }
}

/// The "Skipped a N-sec ad break" confirmation toast.
class _SkippedToast extends StatelessWidget {
  final int seconds;

  const _SkippedToast({required this.seconds});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final colorScheme = theme.colorScheme;
    final ambient = AmbientColors.of(context);

    return TweenAnimationBuilder<double>(
      duration: const Duration(milliseconds: 240),
      curve: Curves.easeOut,
      tween: Tween<double>(begin: 0.0, end: 1.0),
      builder: (context, t, child) {
        return Opacity(
          opacity: t,
          child: Transform.translate(offset: Offset(0, -10 * (1 - t)), child: child),
        );
      },
      child: Material(
        color: Colors.transparent,
        child: Container(
          height: 38.0,
          padding: const EdgeInsets.symmetric(horizontal: 16.0),
          decoration: BoxDecoration(
            color: colorScheme.inverseSurface,
            borderRadius: BorderRadius.circular(999.0),
            boxShadow: [
              BoxShadow(
                color: colorScheme.shadow.withValues(alpha: 0.24),
                blurRadius: 24.0,
                offset: const Offset(0, 8),
              ),
            ],
          ),
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              Icon(Icons.check_rounded, size: 16.0, color: ambient.success),
              const SizedBox(width: 8.0),
              Text(
                'Skipped a $seconds-sec ad break',
                style: theme.textTheme.bodyMedium?.copyWith(
                  color: colorScheme.onInverseSurface,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
