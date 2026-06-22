# Defect Log — Anytime Podcast Player QA

Source-of-truth feature inventory: [`feature_spec.csv`](./feature_spec.csv)

## Iteration 1 — 2026-06-21 (branch `johnfewell/ambient-player-redesign`)

Baseline before work: `flutter analyze` = 98 issues (all `info`: deprecated
`withOpacity`, `prefer_const_constructors` in tests); `flutter test` = 335 pass.

### D-001 — AI ad-skip hero stat is stale within a session — FIXED
- **Feature:** F-AISKIP-04 (AI ad-skip settings + live hero stat)
- **Severity:** Medium (data integrity / UX of the headline feature)
- **Repro:** Play an episode → let an AI-detected ad auto-skip → open Settings ▸
  AI Ad-Skip. The "time saved" / "ads skipped" hero stat does not move.
- **Expected:** Hero stat reflects skips made during the current session.
- **Actual:** Stat only updated after an app restart.
- **Root cause:** `recordAdSkipStat` (in `default_audio_player_service.dart`)
  writes `adSkipSavedSeconds`/`adSkipCount` straight to `SettingsService`
  (SharedPreferences), bypassing `SettingsBloc`. The bloc loaded those values
  only once at init and never listened for external changes, so its `settings`
  stream (which the screen's `StreamBuilder` reads) never re-emitted.
- **Fix:** `SettingsBloc._init()` now subscribes to
  `settingsService.settingsListener` and, on `adSkipSavedSeconds`/`adSkipCount`
  keys, refreshes `_currentSettings` and re-emits. Subscription cancelled in
  `dispose()`. (`lib/bloc/settings/settings_bloc.dart`)
- **Tests:** `settings_bloc_adskip_test.dart` → group "SettingsBloc live
  ad-skip stats" (2 cases). Mock `settingsListener` made functional so it emits
  on stat writes (`test/unit/mocks/mock_settings_service.dart`).

### D-002 — `moveUpNextEpisode` always returns `false` — FIXED
- **Feature:** F-QUEUE-03 (Reorder queue)
- **Severity:** Low (latent; current callers ignore the return value)
- **Root cause:** `DefaultAudioPlayerService.moveUpNextEpisode` declared
  `var moved = false`, performed the move, then returned `moved` without ever
  setting it `true` — contract violation vs the `Future<bool>` signature.
- **Fix:** set `moved = true` after a successful reposition.
  (`lib/services/audio/default_audio_player_service.dart`)

### D-003 — Transcript view: uncancelled scroll listener can `setState` after dispose — FIXED
- **Feature:** F-TRANS-01 (Transcript view)
- **Severity:** Medium (potential crash)
- **Repro:** Open the transcript tab, then navigate away while a scroll-offset
  event is in flight. The `_scrollOffsetListener.changes` subscription was never
  cancelled and its callback called `setState` with no `mounted` guard →
  "setState() called after dispose()".
- **Additional issues in same `dispose()`:** `super.dispose()` was called
  *before* `_positionSubscription.cancel()`, and `_transcriptSearchController`
  (a `TextEditingController`) was never disposed (leak).
- **Fix:** capture the scroll-offset subscription, add a `mounted` guard to its
  `setState`, and in `dispose()` cancel both subscriptions and dispose the
  controller *before* `super.dispose()`. (`lib/ui/podcast/transcript_view.dart`)
- **Tests:** existing `transcript_view_test.dart` still passes; lifecycle is
  not directly unit-testable without a full widget host, verified by review +
  analyzer.

### Iteration 2 — adversarial UI review (no Critical/High introduced)
Full adversarial correctness review of the branch's changed UI files
(`now_playing.dart`, `mini_player.dart`, `episode_actions_sheet.dart`,
`transcript_view.dart`, `episode_details.dart`, `themes.dart`). Confirmed:
`episode_details.dart:942` `state.progress!` is safe (`isIndeterminate ⟺
progress == null`, verified in `episode_transcription_service.dart:27`);
`AmbientColors.of()` never returns null; `_buildAdMarkers` index math is bounded.
Only actionable finding was D-003 above.

### Iteration 3 — coverage backfill for previously UNTESTED features
Added automated tests for features that had no regression net (Phase 2/3 gap):
- `test/unit/entities/sleep_test.dart` (7) — F-SLEEP-01 Sleep model: defaults,
  endTime/timeRemaining, equality/hashCode.
- `test/unit/entities/chapter_test.dart` (10) — F-CHAP-01 Chapter model:
  http→https upgrade, toc default/encoding, toMap/fromMap round-trip, equality.
- `test/unit/entities/funding_test.dart` (6) — F-FUND-01 Funding model:
  URL upgrade + serialization round-trip.
- `test/widget/chapter_selector_test.dart` (3) — F-CHAP-01 Chapters UI:
  toc filtering (meta-only chapters hidden), sequential numbering, loading state.

Suite: 335 (baseline) → **360 pass**. No regressions.

Still relying on manual coverage (logged as residual risk, no observed defects):
F-LIB-03 (mark all played), F-DISC-01 (discovery charts), F-DL-02 (downloads
list), F-FEED-01 (manual RSS add), F-NOTIF-01 (notifications), plus the
sleep-timer ticker/stop path and the funding/sleep selector widgets.

### Iteration 5 — close out remaining UNTESTED features
Added automated coverage for the features the hook flagged as incomplete:
- `episode_bloc_test.dart` "EpisodeBloc downloads (F-DL-02)" (3) — fetchDownloads
  Loading→populated, empty set, delete-removes-and-refreshes. (Extended the
  shared `_FakePodcastService`/`_FakeAudioPlayerService` with `loadDownloads`,
  `deleteDownload`, `removeUpNextEpisode`.)
- `podcast_bloc_test.dart` (2) — F-LIB-03 mark-all-played: marks every unplayed
  episode played + position 0 and persists; all-already-played is a no-op.
  NOTE: PodcastBloc.dispose() closes the *static*
  `MobileDownloadService.downloadProgress`; the test resets it per-test so the
  global side effect can't leak into `episode_actions_sheet_test`.
- `podcast_from_url_test.dart` (3) — F-FEED-01 manual RSS add: `Podcast.fromUrl`
  http→https normalisation + empty placeholder fields.

WAIVED: **F-NOTIF-01** — `MobileNotificationService` is a thin pass-through to
the `AwesomeNotifications()` singleton (platform channels), with no business
logic of its own. A unit test would only assert "the plugin was called" via a
channel mock — brittle and low value. Appropriate coverage is `integration_test`
on a device. Explicitly waived per Phase 4 exit criteria.

Suite: 335 (baseline) → **371 pass** (+36). No regressions; analyzer clean on
all new files.

### Observations (not defects — logged for follow-up)
- O-1: `addUpNextEpisode` does not de-duplicate, so "Play next" on an episode
  already in the up-next list can create a duplicate (pre-existing behaviour,
  out of this branch's scope).
- O-2: `position_slider.dart` uses deprecated `withOpacity` (18 `info`-level
  analyzer hits). Cosmetic; no behaviour change.

After fixes: `flutter test` = 337 pass; `flutter analyze` of changed files = no
issues.
