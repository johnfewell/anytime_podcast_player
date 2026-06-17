// Copyright 2020 Ben Hills and the project contributors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

import 'package:anytime/bloc/settings/settings_bloc.dart';
import 'package:anytime/entities/app_settings.dart';
import 'package:anytime/ui/themes.dart';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

/// The headline feature's own home: the AI ad-skip settings screen.
///
/// A deep-indigo "night" hero shows how much ad time has been skipped, followed
/// by the auto-skip toggle, the countdown-before-skip selector, and detection
/// preferences. See the Player Ambient design.
class AiAdSkipSettings extends StatelessWidget {
  const AiAdSkipSettings({super.key});

  @override
  Widget build(BuildContext context) {
    final settingsBloc = Provider.of<SettingsBloc>(context);
    final theme = Theme.of(context);

    return Scaffold(
      backgroundColor: theme.colorScheme.surface,
      appBar: AppBar(
        title: Text(
          'AI ad-skip',
          style: theme.textTheme.labelSmall?.copyWith(
            color: theme.colorScheme.onSurfaceVariant,
            fontWeight: FontWeight.w700,
            letterSpacing: 1.4,
          ),
        ),
        centerTitle: true,
      ),
      body: StreamBuilder<AppSettings>(
        stream: settingsBloc.settings,
        initialData: settingsBloc.currentSettings,
        builder: (context, snapshot) {
          final settings = snapshot.data!;
          final enabled = settings.adSkipMode != AdSkipMode.disabled;

          return ListView(
            padding: const EdgeInsets.fromLTRB(18.0, 12.0, 18.0, 32.0),
            children: [
              _AdSkipHeroStat(
                savedSeconds: settings.adSkipSavedSeconds,
                count: settings.adSkipCount,
              ),
              const SizedBox(height: 22.0),
              _AdSkipCard(
                children: [
                  _ToggleRow(
                    title: 'Auto-skip detected ads',
                    subtitle: 'Skip without asking',
                    value: enabled,
                    onChanged: (on) => settingsBloc.setAdSkipMode(on ? AdSkipMode.prompt : AdSkipMode.disabled),
                  ),
                  const _RowDivider(),
                  _CountdownRow(
                    seconds: settings.adSkipCountdownSeconds,
                    enabled: enabled,
                    onChanged: settingsBloc.setAdSkipCountdownSeconds,
                  ),
                  const _RowDivider(),
                  _ToggleRow(
                    title: 'Notify when skipped',
                    subtitle: 'Brief toast after each skip',
                    value: settings.adSkipNotify,
                    onChanged: enabled ? settingsBloc.setAdSkipNotify : null,
                  ),
                  const _RowDivider(),
                  _ToggleRow(
                    title: 'Include host-read ads',
                    subtitle: 'Detect in-content sponsorships',
                    value: settings.adSkipIncludeHostRead,
                    onChanged: enabled ? settingsBloc.setAdSkipIncludeHostRead : null,
                  ),
                ],
              ),
              const Padding(
                padding: EdgeInsets.fromLTRB(8.0, 14.0, 8.0, 0.0),
                child: _PrivacyNote(),
              ),
            ],
          );
        },
      ),
    );
  }
}

/// Deep-indigo hero band showing the running "ads skipped" stat.
class _AdSkipHeroStat extends StatelessWidget {
  final int savedSeconds;
  final int count;

  const _AdSkipHeroStat({required this.savedSeconds, required this.count});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final ambient = AmbientColors.of(context);
    final onNight = ambient.onNight;

    return AmbientNightBand(
      padding: const EdgeInsets.all(22.0),
      borderRadius: 20.0,
      glowSize: 120.0,
      glowTop: -28.0,
      glowRight: -24.0,
      child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Container(
                height: 26.0,
                padding: const EdgeInsets.symmetric(horizontal: 11.0),
                decoration: BoxDecoration(
                  color: Colors.white.withValues(alpha: 0.16),
                  borderRadius: BorderRadius.circular(999.0),
                ),
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Icon(Icons.auto_awesome, size: 12.0, color: ambient.aiTeal),
                    const SizedBox(width: 6.0),
                    Text(
                      'THIS MONTH',
                      style: theme.textTheme.labelSmall?.copyWith(
                        color: onNight,
                        fontWeight: FontWeight.w700,
                        letterSpacing: 0.4,
                      ),
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 14.0),
              Text(
                '${_formatSaved(savedSeconds)} of ads skipped',
                style: theme.textTheme.headlineMedium?.copyWith(
                  color: onNight,
                  fontWeight: FontWeight.w800,
                ),
              ),
              const SizedBox(height: 6.0),
              Text(
                count == 0 ? 'No ads skipped yet' : 'across $count ${count == 1 ? 'episode' : 'episodes'}',
                style: theme.textTheme.bodyMedium?.copyWith(
                  color: onNight.withValues(alpha: 0.66),
                ),
              ),
            ],
          ),
    );
  }

  String _formatSaved(int seconds) {
    if (seconds <= 0) {
      return '0m';
    }
    final hours = seconds ~/ 3600;
    final minutes = (seconds % 3600) ~/ 60;
    if (hours > 0) {
      return '${hours}h ${minutes}m';
    }
    if (minutes > 0) {
      return '${minutes}m';
    }
    return '${seconds}s';
  }
}

class _AdSkipCard extends StatelessWidget {
  final List<Widget> children;

  const _AdSkipCard({required this.children});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final ambient = AmbientColors.of(context);

    return Container(
      decoration: BoxDecoration(
        color: theme.colorScheme.surfaceContainerLowest,
        borderRadius: BorderRadius.circular(14.0),
        border: Border.all(color: ambient.hairline),
      ),
      child: Column(children: children),
    );
  }
}

class _RowDivider extends StatelessWidget {
  const _RowDivider();

  @override
  Widget build(BuildContext context) {
    return Divider(height: 1.0, thickness: 1.0, color: AmbientColors.of(context).hairline);
  }
}

class _ToggleRow extends StatelessWidget {
  final String title;
  final String subtitle;
  final bool value;
  final ValueChanged<bool>? onChanged;

  const _ToggleRow({
    required this.title,
    required this.subtitle,
    required this.value,
    required this.onChanged,
  });

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Padding(
      padding: const EdgeInsets.fromLTRB(16.0, 14.0, 16.0, 14.0),
      child: Row(
        children: [
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(title, style: theme.textTheme.titleSmall),
                const SizedBox(height: 2.0),
                Text(
                  subtitle,
                  style: theme.textTheme.bodySmall?.copyWith(color: theme.colorScheme.onSurfaceVariant),
                ),
              ],
            ),
          ),
          const SizedBox(width: 12.0),
          Switch.adaptive(value: value, onChanged: onChanged),
        ],
      ),
    );
  }
}

class _CountdownRow extends StatelessWidget {
  final int seconds;
  final bool enabled;
  final ValueChanged<int> onChanged;

  const _CountdownRow({
    required this.seconds,
    required this.enabled,
    required this.onChanged,
  });

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Padding(
      padding: const EdgeInsets.fromLTRB(16.0, 14.0, 16.0, 14.0),
      child: Row(
        children: [
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text('Countdown before skip', style: theme.textTheme.titleSmall),
                const SizedBox(height: 2.0),
                Text(
                  'Time to cancel a skip',
                  style: theme.textTheme.bodySmall?.copyWith(color: theme.colorScheme.onSurfaceVariant),
                ),
              ],
            ),
          ),
          const SizedBox(width: 12.0),
          _SegmentedSeconds(
            value: seconds,
            enabled: enabled,
            options: const [0, 3, 5],
            onChanged: onChanged,
          ),
        ],
      ),
    );
  }
}

class _SegmentedSeconds extends StatelessWidget {
  final int value;
  final bool enabled;
  final List<int> options;
  final ValueChanged<int> onChanged;

  const _SegmentedSeconds({
    required this.value,
    required this.enabled,
    required this.options,
    required this.onChanged,
  });

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final ambient = AmbientColors.of(context);

    return Container(
      padding: const EdgeInsets.all(3.0),
      decoration: BoxDecoration(
        color: ambient.controlFill,
        borderRadius: BorderRadius.circular(999.0),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: options.map((option) {
          final selected = option == value;
          return GestureDetector(
            onTap: enabled ? () => onChanged(option) : null,
            child: AnimatedContainer(
              duration: const Duration(milliseconds: 150),
              height: 26.0,
              padding: const EdgeInsets.symmetric(horizontal: 10.0),
              alignment: Alignment.center,
              decoration: BoxDecoration(
                color: selected ? theme.colorScheme.surfaceContainerLowest : Colors.transparent,
                borderRadius: BorderRadius.circular(999.0),
                boxShadow: selected
                    ? [
                        BoxShadow(
                          color: theme.colorScheme.shadow.withValues(alpha: 0.08),
                          blurRadius: 6.0,
                          offset: const Offset(0, 2),
                        ),
                      ]
                    : null,
              ),
              child: Text(
                '${option}s',
                style: theme.textTheme.labelLarge?.copyWith(
                  color: selected ? theme.colorScheme.primary : theme.colorScheme.outline,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ),
          );
        }).toList(),
      ),
    );
  }
}

class _PrivacyNote extends StatelessWidget {
  const _PrivacyNote();

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Text(
      'Detection runs on-device from the episode transcript. Audio never leaves your phone.',
      style: theme.textTheme.bodySmall?.copyWith(color: theme.colorScheme.outline, height: 1.45),
    );
  }
}
