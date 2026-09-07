//! Persisted wp-engine settings: per-wallpaper property overrides, screen
//! assignments, and playlists — the wp-engine-native counterpart of the real
//! Wallpaper Engine client's `config.json` (`steamuser.wproperties`/
//! `wallpaperconfig.selectedwallpapers`/`general.playlists`).
//!
//! Not wire-compatible with that format, deliberately: the real client's
//! on-disk `wproperties` value encoding has no ground truth available to
//! verify against (`wproperties` is an empty object in every real
//! `config.json` this was checked against — see the Ghidra report's
//! settings-management follow-up; the vendored C++ reference doesn't read
//! or write it either, only `config.json`'s playlists/screen-assignment
//! sections for its own CLI flags). Guessing at an unverified per-property
//! value encoding would risk silently writing something the real client
//! can't read; a fully-specified wp-engine-native schema avoids that.
//!
//! Honest scope note: `properties`, `screens`, and `quality` are wired
//! end-to-end (`wp-engine set`/`set-file`/`run` load and apply them
//! automatically, merged under any matching CLI flags). `screens` genuinely
//! drives different content *and* independent render quality per output now
//! (see `render::ScreenContent`/`render::ScreenSettings`) — different
//! outputs can show different wallpapers at different quality levels.
//! `playlists` is still data-only: fully persisted and listable via
//! `wp-engine config`, but there's no playback-rotation timer anywhere in
//! the codebase yet, so only each playlist's first item is ever reachable
//! through `run` today. `volume` has no per-wallpaper override here at all
//! — `RenderSettings::volume` itself is documented as "not yet applied" (no
//! audio-mixing code reads it), so persisting a per-wallpaper value for a
//! setting that doesn't do anything yet would be silently misleading.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct WpSettings {
    /// Per-wallpaper user-property overrides, keyed by [`normalize_key`].
    #[serde(default)]
    pub properties: BTreeMap<String, BTreeMap<String, String>>,
    /// Named screen → wallpaper assignment (e.g. `"DP-1"` → workshop ID or
    /// path). See the module doc: persisted, not yet renderer-driven.
    #[serde(default)]
    pub screens: BTreeMap<String, String>,
    /// The wallpaper `wp-engine run` falls back to when no screen-specific
    /// assignment applies.
    #[serde(default)]
    pub default_background: Option<String>,
    /// Named playlists. See the module doc: persisted, not yet
    /// renderer-driven (no rotation timer exists yet).
    #[serde(default)]
    pub playlists: BTreeMap<String, Playlist>,
    /// Per-wallpaper render-quality override, keyed by [`normalize_key`] —
    /// one of `platform::RenderQuality::label()`'s values ("Ultra"/"High"/
    /// "Medium"/"Low"), stored as the label string rather than the enum so
    /// an older saved value someone hand-edited still round-trips even if
    /// the level names ever change.
    #[serde(default)]
    pub quality: BTreeMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Playlist {
    pub items: Vec<String>,
    #[serde(default = "default_delay_minutes")]
    pub delay_minutes: u32,
    /// `"timer"` (advance every `delay_minutes`) — the only mode with
    /// anywhere to plug into yet; mirrors the real client's/linux-
    /// wallpaperengine's `mode`/`order` fields structurally so a playback
    /// scheduler has a ready-made schema to consume later.
    #[serde(default = "default_mode")]
    pub mode: String,
    #[serde(default = "default_order")]
    pub order: String,
}

fn default_delay_minutes() -> u32 {
    60
}
fn default_mode() -> String {
    "timer".to_string()
}
fn default_order() -> String {
    "sequential".to_string()
}

/// `$XDG_CONFIG_HOME/wp-engine/settings.json` (`~/.config/wp-engine/...` by
/// convention when unset) via `dirs::config_dir()` — the same crate
/// `workshop::scan_wallpapers`'s Steam-path fallback already depends on, so
/// this resolves per-platform the same way (`~/Library/Application Support`
/// on macOS, `%APPDATA%` on Windows).
pub fn settings_path() -> PathBuf {
    dirs::config_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("wp-engine")
        .join("settings.json")
}

/// One wallpaper's stable key for `properties`/`screens`/playlist items: the
/// bare workshop ID when it resolves to one (so `wp-engine set 123 …` today
/// and a future run by the same ID hit the same saved entry regardless of
/// where the Workshop happens to have it on disk), else the canonicalized
/// absolute path so two different spellings of the same local file
/// (relative vs. absolute, `./x` vs `x`) still collide correctly.
pub fn normalize_key(id_or_path: &str) -> String {
    if let Some(w) = crate::workshop::find_by_id(id_or_path) {
        return w.workshop_id;
    }
    std::fs::canonicalize(id_or_path)
        .map(|p| p.display().to_string())
        .unwrap_or_else(|_| id_or_path.to_string())
}

impl WpSettings {
    /// Loads from [`settings_path`]. A missing file is the common case (a
    /// fresh install) — returns defaults silently. A present-but-corrupt
    /// file is a real problem, but still shouldn't crash the whole app over
    /// a hand-edited or half-written file: logs a warning and returns
    /// defaults instead.
    pub fn load() -> Self {
        Self::load_from(&settings_path())
    }

    pub fn load_from(path: &Path) -> Self {
        let Ok(data) = std::fs::read_to_string(path) else {
            return Self::default();
        };
        match serde_json::from_str(&data) {
            Ok(settings) => settings,
            Err(e) => {
                tracing::warn!(target: "settings", "failed to parse {}: {e} — using defaults", path.display());
                Self::default()
            }
        }
    }

    pub fn save(&self) -> anyhow::Result<()> {
        self.save_to(&settings_path())
    }

    pub fn save_to(&self, path: &Path) -> anyhow::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let data = serde_json::to_string_pretty(self)?;
        std::fs::write(path, data)?;
        Ok(())
    }

    /// Saved property overrides for one wallpaper, empty when it has none.
    pub fn property_overrides(&self, id_or_path: &str) -> std::collections::HashMap<String, String> {
        self.properties
            .get(&normalize_key(id_or_path))
            .map(|m| m.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
            .unwrap_or_default()
    }

    pub fn set_property(&mut self, id_or_path: &str, name: &str, value: &str) {
        self.properties
            .entry(normalize_key(id_or_path))
            .or_default()
            .insert(name.to_string(), value.to_string());
    }

    /// Returns whether a saved override actually existed to remove.
    pub fn unset_property(&mut self, id_or_path: &str, name: &str) -> bool {
        let key = normalize_key(id_or_path);
        let Some(m) = self.properties.get_mut(&key) else {
            return false;
        };
        let removed = m.remove(name).is_some();
        // Prune the now-empty inner map so a fully-cleared wallpaper
        // disappears from `properties` entirely instead of leaving a
        // dangling `"id": {}` behind.
        if m.is_empty() {
            self.properties.remove(&key);
        }
        removed
    }

    pub fn set_screen(&mut self, screen: &str, id_or_path: &str) {
        self.screens.insert(screen.to_string(), id_or_path.to_string());
    }

    /// Returns whether a screen assignment actually existed to remove.
    pub fn unset_screen(&mut self, screen: &str) -> bool {
        self.screens.remove(screen).is_some()
    }

    pub fn set_default_background(&mut self, id_or_path: &str) {
        self.default_background = Some(id_or_path.to_string());
    }

    /// The saved render-quality override for one wallpaper, if any and if
    /// it still parses as a known [`crate::platform::RenderQuality`] (a
    /// hand-edited or stale `settings.json` value is ignored, not fatal).
    pub fn quality_override(&self, id_or_path: &str) -> Option<crate::platform::RenderQuality> {
        self.quality
            .get(&normalize_key(id_or_path))
            .and_then(|s| crate::platform::RenderQuality::parse(s))
    }

    pub fn set_quality(&mut self, id_or_path: &str, quality: crate::platform::RenderQuality) {
        self.quality
            .insert(normalize_key(id_or_path), quality.label().to_string());
    }

    /// Returns whether a saved override actually existed to remove.
    pub fn unset_quality(&mut self, id_or_path: &str) -> bool {
        self.quality.remove(&normalize_key(id_or_path)).is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tempdir_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "wp-engine-settings-test-{name}-{}-{:?}",
            std::process::id(),
            std::time::SystemTime::now()
        ))
    }

    #[test]
    fn missing_file_loads_as_default() {
        let path = tempdir_path("missing").join("settings.json");
        let settings = WpSettings::load_from(&path);
        assert_eq!(settings, WpSettings::default());
    }

    #[test]
    fn corrupt_file_falls_back_to_default_instead_of_panicking() {
        let path = tempdir_path("corrupt");
        std::fs::create_dir_all(&path).unwrap();
        let file = path.join("settings.json");
        std::fs::write(&file, "{ not valid json").unwrap();
        let settings = WpSettings::load_from(&file);
        assert_eq!(settings, WpSettings::default());
    }

    #[test]
    fn save_then_load_round_trips_every_field() {
        let dir = tempdir_path("roundtrip");
        let file = dir.join("settings.json");

        let mut settings = WpSettings::default();
        // A fake ID that `workshop::find_by_id` won't resolve, so
        // `normalize_key` falls back to using the string as-is (no
        // dependency on this sandbox's real Workshop contents).
        settings.set_property("no-such-wallpaper-id", "speed", "5");
        settings.set_property("no-such-wallpaper-id", "color", "1 0 0");
        settings.set_screen("DP-1", "no-such-wallpaper-id");
        settings.set_default_background("no-such-wallpaper-id");
        settings.set_quality("no-such-wallpaper-id", crate::platform::RenderQuality::Medium);
        settings.playlists.insert(
            "morning".to_string(),
            Playlist {
                items: vec!["a".to_string(), "b".to_string()],
                delay_minutes: 30,
                mode: "timer".to_string(),
                order: "random".to_string(),
            },
        );

        settings.save_to(&file).unwrap();
        let loaded = WpSettings::load_from(&file);
        assert_eq!(loaded, settings);
    }

    #[test]
    fn property_overrides_are_per_wallpaper_and_independent() {
        let mut settings = WpSettings::default();
        settings.set_property("wallpaper-a", "speed", "1");
        settings.set_property("wallpaper-b", "speed", "2");

        assert_eq!(
            settings.property_overrides("wallpaper-a").get("speed"),
            Some(&"1".to_string())
        );
        assert_eq!(
            settings.property_overrides("wallpaper-b").get("speed"),
            Some(&"2".to_string())
        );
        assert!(settings.property_overrides("wallpaper-c").is_empty());
    }

    #[test]
    fn unset_property_removes_only_the_named_one() {
        let mut settings = WpSettings::default();
        settings.set_property("w", "speed", "1");
        settings.set_property("w", "color", "1 0 0");

        assert!(settings.unset_property("w", "speed"));
        assert!(!settings.unset_property("w", "speed")); // already gone
        assert!(!settings.unset_property("w", "nonexistent"));

        let overrides = settings.property_overrides("w");
        assert!(!overrides.contains_key("speed"));
        assert_eq!(overrides.get("color"), Some(&"1 0 0".to_string()));
    }

    #[test]
    fn unset_screen_reports_whether_it_existed() {
        let mut settings = WpSettings::default();
        settings.set_screen("DP-1", "123");
        assert!(settings.unset_screen("DP-1"));
        assert!(!settings.unset_screen("DP-1"));
        assert!(settings.screens.is_empty());
    }

    #[test]
    fn quality_override_round_trips_through_the_label_string() {
        use crate::platform::RenderQuality;

        let mut settings = WpSettings::default();
        assert_eq!(settings.quality_override("w"), None);

        settings.set_quality("w", RenderQuality::Low);
        assert_eq!(settings.quality_override("w"), Some(RenderQuality::Low));
        // Stored as the label string, not some opaque encoding — inspect it directly.
        assert_eq!(settings.quality.get(&normalize_key("w")), Some(&"Low".to_string()));

        settings.set_quality("w", RenderQuality::Ultra);
        assert_eq!(settings.quality_override("w"), Some(RenderQuality::Ultra));
    }

    #[test]
    fn quality_override_ignores_an_unparseable_saved_value() {
        let mut settings = WpSettings::default();
        settings
            .quality
            .insert(normalize_key("w"), "Blazing Fast".to_string());
        assert_eq!(settings.quality_override("w"), None);
    }

    #[test]
    fn unset_quality_removes_only_the_named_wallpaper() {
        use crate::platform::RenderQuality;

        let mut settings = WpSettings::default();
        settings.set_quality("a", RenderQuality::Low);
        settings.set_quality("b", RenderQuality::High);

        assert!(settings.unset_quality("a"));
        assert!(!settings.unset_quality("a")); // already gone
        assert_eq!(settings.quality_override("a"), None);
        assert_eq!(settings.quality_override("b"), Some(RenderQuality::High));
    }
}
