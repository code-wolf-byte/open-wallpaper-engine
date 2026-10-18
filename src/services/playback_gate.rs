//! Playback policy — Wallpaper Engine's "while another application is…"
//! settings: for each condition, keep running, mute, pause, or stop (pause
//! and free the renderer's memory).
//!
//! The conditions match WE's own setting keys (recovered from the binary's
//! string table): `playbackfocus` (another application has focus),
//! `playbackmaximized`, `playbackfullscreen`, `playbackaudio` (another
//! application is playing audio), `playbackonbattery`, and `playbacksleep`
//! (session locked or asleep).
//!
//! This module is pure decision logic; the platform backends fill in
//! [`PlaybackState`] (EWMH / `zwlr_foreign_toplevel_management` for window
//! state, `platform::power` for battery and lock, MPRIS for other media).

use serde::{Deserialize, Serialize};

/// What to do while a condition holds. Ordered by strength: when several
/// conditions hold, the strongest action applies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum PlaybackAction {
    #[default]
    Run,
    Mute,
    Pause,
    /// Pause and release the wallpaper's renderer; it is rebuilt on resume.
    Stop,
}

impl PlaybackAction {
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "run" | "keep" | "keeprunning" | "0" | "off" | "false" => Some(Self::Run),
            "mute" => Some(Self::Mute),
            "pause" | "1" | "on" | "true" => Some(Self::Pause),
            "stop" => Some(Self::Stop),
            _ => None,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Run => "run",
            Self::Mute => "mute",
            Self::Pause => "pause",
            Self::Stop => "stop",
        }
    }

    pub const ALL: [PlaybackAction; 4] = [Self::Run, Self::Mute, Self::Pause, Self::Stop];

    /// Rendering halts (pause or stop).
    pub fn halts(self) -> bool {
        self >= Self::Pause
    }
}

/// The action for each condition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct PlaybackPolicy {
    pub other_focused: PlaybackAction,
    pub other_maximized: PlaybackAction,
    pub other_fullscreen: PlaybackAction,
    pub other_audio: PlaybackAction,
    pub on_battery: PlaybackAction,
    pub locked: PlaybackAction,
}

impl Default for PlaybackPolicy {
    /// Pause behind a fullscreen application and while the session is
    /// locked — nothing is visible either way. Everything else keeps
    /// running, as in WE's own defaults.
    fn default() -> Self {
        Self {
            other_focused: PlaybackAction::Run,
            other_maximized: PlaybackAction::Run,
            other_fullscreen: PlaybackAction::Pause,
            other_audio: PlaybackAction::Run,
            on_battery: PlaybackAction::Run,
            locked: PlaybackAction::Pause,
        }
    }
}

impl PlaybackPolicy {
    /// The saved policy (`settings.json`), then `WP_ENGINE_PAUSE_ON_*`
    /// overrides per condition (`run`/`mute`/`pause`/`stop`, or `0`/`1`).
    pub fn load() -> Self {
        Self::from_env(crate::settings::WpSettings::load().playback)
    }

    pub fn from_env(base: Self) -> Self {
        let get = |name: &str, current: PlaybackAction| {
            std::env::var(name)
                .ok()
                .and_then(|v| PlaybackAction::parse(&v))
                .unwrap_or(current)
        };
        Self {
            other_focused: get("WP_ENGINE_PAUSE_ON_FOCUS", base.other_focused),
            other_maximized: get("WP_ENGINE_PAUSE_ON_MAXIMIZED", base.other_maximized),
            other_fullscreen: get("WP_ENGINE_PAUSE_ON_FULLSCREEN", base.other_fullscreen),
            other_audio: get("WP_ENGINE_PAUSE_ON_AUDIO", base.other_audio),
            on_battery: get("WP_ENGINE_PAUSE_ON_BATTERY", base.on_battery),
            locked: get("WP_ENGINE_PAUSE_ON_LOCK", base.locked),
        }
    }

    /// Whether any condition needs the media (MPRIS) watcher.
    pub fn watches_audio(&self) -> bool {
        self.other_audio != PlaybackAction::Run
    }

    /// Set one condition by its settings name.
    pub fn set(&mut self, condition: &str, action: PlaybackAction) -> bool {
        let slot = match condition {
            "focus" | "other_focused" => &mut self.other_focused,
            "maximized" | "other_maximized" => &mut self.other_maximized,
            "fullscreen" | "other_fullscreen" => &mut self.other_fullscreen,
            "audio" | "other_audio" => &mut self.other_audio,
            "battery" | "on_battery" => &mut self.on_battery,
            "lock" | "locked" => &mut self.locked,
            _ => return false,
        };
        *slot = action;
        true
    }
}

/// Live system state, filled in by the platform backends.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PlaybackState {
    pub on_battery: bool,
    pub locked_or_sleeping: bool,
    /// The active window is fullscreen.
    pub fullscreen_app: bool,
    /// The active window is maximized.
    pub maximized_app: bool,
    /// Some application window is active (the desktop isn't focused).
    pub focused_app: bool,
    /// Another application is playing media.
    pub other_audio: bool,
}

/// The action to take right now: the strongest among the conditions that
/// currently hold.
pub fn decide(policy: &PlaybackPolicy, state: &PlaybackState) -> PlaybackAction {
    [
        (state.focused_app, policy.other_focused),
        (state.maximized_app, policy.other_maximized),
        (state.fullscreen_app, policy.other_fullscreen),
        (state.other_audio, policy.other_audio),
        (state.on_battery, policy.on_battery),
        (state.locked_or_sleeping, policy.locked),
    ]
    .into_iter()
    .filter(|(holds, _)| *holds)
    .map(|(_, action)| action)
    .max()
    .unwrap_or(PlaybackAction::Run)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nothing_holding_keeps_running() {
        let policy = PlaybackPolicy {
            other_focused: PlaybackAction::Stop,
            other_maximized: PlaybackAction::Stop,
            other_fullscreen: PlaybackAction::Stop,
            other_audio: PlaybackAction::Stop,
            on_battery: PlaybackAction::Stop,
            locked: PlaybackAction::Stop,
        };
        assert_eq!(
            decide(&policy, &PlaybackState::default()),
            PlaybackAction::Run
        );
    }

    #[test]
    fn strongest_holding_condition_wins() {
        let policy = PlaybackPolicy {
            other_focused: PlaybackAction::Mute,
            other_maximized: PlaybackAction::Pause,
            ..Default::default()
        };
        let state = PlaybackState {
            focused_app: true,
            maximized_app: true,
            ..Default::default()
        };
        assert_eq!(decide(&policy, &state), PlaybackAction::Pause);
        let state = PlaybackState {
            focused_app: true,
            ..Default::default()
        };
        assert_eq!(decide(&policy, &state), PlaybackAction::Mute);
    }

    #[test]
    fn defaults_pause_for_fullscreen_and_lock_only() {
        let d = PlaybackPolicy::default();
        assert_eq!(d.other_fullscreen, PlaybackAction::Pause);
        assert_eq!(d.locked, PlaybackAction::Pause);
        assert_eq!(d.other_focused, PlaybackAction::Run);
        assert_eq!(d.on_battery, PlaybackAction::Run);
        assert!(!d.watches_audio());
    }

    #[test]
    fn env_overrides_accept_actions_and_legacy_flags() {
        // SAFETY: test-only env mutation of names unique to this test.
        unsafe {
            std::env::set_var("WP_ENGINE_PAUSE_ON_BATTERY", "1");
            std::env::set_var("WP_ENGINE_PAUSE_ON_LOCK", "0");
            std::env::set_var("WP_ENGINE_PAUSE_ON_AUDIO", "mute");
        }
        let p = PlaybackPolicy::from_env(PlaybackPolicy::default());
        assert_eq!(p.on_battery, PlaybackAction::Pause);
        assert_eq!(p.locked, PlaybackAction::Run);
        assert_eq!(p.other_audio, PlaybackAction::Mute);
        assert_eq!(p.other_fullscreen, PlaybackAction::Pause);
        unsafe {
            std::env::remove_var("WP_ENGINE_PAUSE_ON_BATTERY");
            std::env::remove_var("WP_ENGINE_PAUSE_ON_LOCK");
            std::env::remove_var("WP_ENGINE_PAUSE_ON_AUDIO");
        }
    }

    #[test]
    fn set_by_condition_name() {
        let mut p = PlaybackPolicy::default();
        assert!(p.set("audio", PlaybackAction::Pause));
        assert!(!p.set("nope", PlaybackAction::Pause));
        assert_eq!(p.other_audio, PlaybackAction::Pause);
        assert!(PlaybackAction::Stop.halts() && !PlaybackAction::Mute.halts());
    }
}
