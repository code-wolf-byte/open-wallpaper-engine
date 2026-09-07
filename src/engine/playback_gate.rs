//! Playback-pause conditions — the real Wallpaper Engine's own "pause
//! wallpaper rendering when..." general settings.
//!
//! Recovered from the Ghidra dump's import table (`~/Applications/
//! ghidra-dump/imports.txt`), not the usual `scene.json` property-name
//! scan: `SetWinEventHook`/`GetForegroundWindow`/`IsIconic` (focus/
//! maximized/fullscreen-app detection), `RegisterPowerSettingNotification`/
//! `GetSystemPowerStatus` (battery), `WTSRegisterSessionNotification`
//! (session lock/sleep) — cross-referenced against six setting-key strings
//! clustered together in `strings.txt` (`140476df0`-`140476e78`):
//! `playbackfocus`, `playbackmaximized`, `playbackfullscreen`,
//! `playbackonbattery`, `playbacksleep`, `playbackaudio`. See the report's
//! Follow-up (ff) for the full writeup.
//!
//! This is a real, unbuilt feature, not a formula-approximation gap — the
//! platform primitives it needs (foreground-window/fullscreen detection,
//! battery status, session lock state) all have Linux/macOS equivalents,
//! unlike shadows/fog/bloom's unrecoverable-D3D-bytecode wall.
//!
//! # What's implemented
//! - `on_battery` (`playbackonbattery`) — `platform::power::is_on_battery`.
//! - `on_lock_or_sleep` (`playbacksleep`) — `platform::power::is_locked`
//!   polls the session's `LockedHint` (a persistent property); the actual
//!   suspend-to-RAM transition (`PrepareForSleep`) is a one-shot D-Bus
//!   signal, not something worth polling for here — the whole process
//!   freezes with the machine during a real suspend anyway, so there's
//!   nothing this module could still be doing at that instant that pausing
//!   a frame earlier would meaningfully save.
//! - `on_fullscreen_app`/`on_maximized_app` (`playbackfullscreen`/
//!   `playbackmaximized`) — `platform::x11`'s EWMH query and
//!   `platform::wayland`'s `zwlr_foreign_toplevel_management` tracker.
//!
//! # What's NOT implemented (documented, not silently dropped)
//! - `playbackfocus` — ambiguous even as a concept for a background
//!   desktop wallpaper (which never "has focus" the way a normal window
//!   does); no confident reading of what WE's own checkbox means here.
//! - `playbackaudio` — likely "mute the wallpaper's own audio output when
//!   another app is also playing sound," a real WE feature, but a muting
//!   rule, not a *pause* condition — doesn't belong in this gate at all,
//!   and is its own separate, unscoped follow-on.
//!
//! # Defaults
//! WE's own real defaults for these six checkboxes aren't recoverable from
//! the binary (same "confirmed setting exists, exact default value isn't"
//! situation as `SceneObject::light_params`'s intensity/radius defaults
//! elsewhere in this codebase). [`PauseConditions::default`] picks
//! defaults by how unambiguously safe each one is (see its own doc), not a
//! verified match — overridable per-field via `WP_ENGINE_PAUSE_ON_*` env
//! vars, this codebase's existing toggle convention (`WP_ENGINE_FORCE_X11`,
//! `WP_ENGINE_SKIP_EFFECTS`, `WP_ENGINE_ADDITIVE_BRIGHTNESS`, …) rather than
//! new `settings.json`/CLI surface for a first cut.

/// Which conditions actually pause rendering — the enabled/disabled state
/// of WE's four implemented checkboxes (see the module doc for the other
/// two).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PauseConditions {
    pub on_battery: bool,
    pub on_lock_or_sleep: bool,
    pub on_fullscreen_app: bool,
    pub on_maximized_app: bool,
}

impl Default for PauseConditions {
    /// Fullscreen-app and lock/sleep default ON: both are unambiguously
    /// desirable (nothing is visible either way — a locked screen shows
    /// nothing, a fullscreen game covers the whole output) and match this
    /// project's own stated purpose as a lightweight alternative. Battery
    /// and maximized-window default OFF: pausing on battery trades away
    /// visual continuity a laptop user may still want on principle, and
    /// "maximized" (an ordinary browser window, say) is a much lower bar
    /// than "fullscreen" for a condition this aggressive — both are
    /// reasonable opt-in choices, not obviously-correct defaults.
    fn default() -> Self {
        Self {
            on_battery: false,
            on_lock_or_sleep: true,
            on_fullscreen_app: true,
            on_maximized_app: false,
        }
    }
}

impl PauseConditions {
    /// [`Self::default`], with each field overridable by its own
    /// `WP_ENGINE_PAUSE_ON_*` env var (`"1"` = on, `"0"` = off, anything
    /// else/unset = the default).
    pub fn from_env() -> Self {
        let d = Self::default();
        let flag = |name: &str, default: bool| -> bool {
            match std::env::var(name).ok().as_deref() {
                Some("1") => true,
                Some("0") => false,
                _ => default,
            }
        };
        Self {
            on_battery: flag("WP_ENGINE_PAUSE_ON_BATTERY", d.on_battery),
            on_lock_or_sleep: flag("WP_ENGINE_PAUSE_ON_LOCK", d.on_lock_or_sleep),
            on_fullscreen_app: flag("WP_ENGINE_PAUSE_ON_FULLSCREEN", d.on_fullscreen_app),
            on_maximized_app: flag("WP_ENGINE_PAUSE_ON_MAXIMIZED", d.on_maximized_app),
        }
    }
}

/// Live system state each condition above checks against. Platform code
/// (`platform::power`, `platform::x11`, `platform::wayland`) fills this
/// in each tick; this module itself stays pure combine logic with no I/O
/// of its own, so it's fully unit-testable without touching the OS.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PlaybackState {
    pub on_battery: bool,
    pub locked_or_sleeping: bool,
    pub fullscreen_app: bool,
    pub maximized_app: bool,
}

/// Whether rendering should pause right now, given which conditions are
/// enabled and the live system state — an enabled condition that isn't
/// currently true never contributes; any true condition that's enabled
/// pauses regardless of the others (WE's own checkboxes are independent
/// ORed triggers, not a priority list).
pub fn should_pause(cond: &PauseConditions, state: &PlaybackState) -> bool {
    (cond.on_battery && state.on_battery)
        || (cond.on_lock_or_sleep && state.locked_or_sleeping)
        || (cond.on_fullscreen_app && state.fullscreen_app)
        || (cond.on_maximized_app && state.maximized_app)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_conditions_enabled_never_pauses() {
        let cond = PauseConditions {
            on_battery: false,
            on_lock_or_sleep: false,
            on_fullscreen_app: false,
            on_maximized_app: false,
        };
        let state = PlaybackState {
            on_battery: true,
            locked_or_sleeping: true,
            fullscreen_app: true,
            maximized_app: true,
        };
        assert!(!should_pause(&cond, &state));
    }

    #[test]
    fn enabled_condition_not_currently_true_does_not_pause() {
        let cond = PauseConditions {
            on_battery: true,
            on_lock_or_sleep: true,
            on_fullscreen_app: true,
            on_maximized_app: true,
        };
        assert!(!should_pause(&cond, &PlaybackState::default()));
    }

    #[test]
    fn any_single_enabled_and_true_condition_pauses() {
        let cond = PauseConditions {
            on_battery: false,
            on_lock_or_sleep: false,
            on_fullscreen_app: true,
            on_maximized_app: false,
        };
        let state = PlaybackState {
            fullscreen_app: true,
            ..Default::default()
        };
        assert!(should_pause(&cond, &state));
    }

    #[test]
    fn default_conditions_favor_lock_and_fullscreen_over_battery_and_maximized() {
        let d = PauseConditions::default();
        assert!(d.on_lock_or_sleep);
        assert!(d.on_fullscreen_app);
        assert!(!d.on_battery);
        assert!(!d.on_maximized_app);
    }

    #[test]
    fn from_env_overrides_each_field_independently() {
        // SAFETY: test-only env mutation, no other thread reads these vars
        // during this test's lifetime (cargo test runs each test in its
        // own thread, but these names are unique to this module).
        unsafe {
            std::env::set_var("WP_ENGINE_PAUSE_ON_BATTERY", "1");
            std::env::set_var("WP_ENGINE_PAUSE_ON_LOCK", "0");
        }
        std::env::remove_var("WP_ENGINE_PAUSE_ON_FULLSCREEN");
        std::env::remove_var("WP_ENGINE_PAUSE_ON_MAXIMIZED");

        let cond = PauseConditions::from_env();
        assert!(cond.on_battery, "explicit 1 should override the false default");
        assert!(!cond.on_lock_or_sleep, "explicit 0 should override the true default");
        assert!(cond.on_fullscreen_app, "unset should keep the true default");
        assert!(!cond.on_maximized_app, "unset should keep the false default");

        unsafe {
            std::env::remove_var("WP_ENGINE_PAUSE_ON_BATTERY");
            std::env::remove_var("WP_ENGINE_PAUSE_ON_LOCK");
        }
    }
}
