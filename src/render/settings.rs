use crate::platform::RenderQuality;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Render settings shared between the UI and the wallpaper thread.
#[derive(Debug)]
pub struct RenderSettings {
    /// GPU render quality / source down-sample factor.
    pub quality: RenderQuality,
    /// Volume in 0.0–1.0 (stored for future audio support; not yet applied).
    pub volume: f32,
}

impl Default for RenderSettings {
    fn default() -> Self {
        Self {
            quality: RenderQuality::Ultra,
            volume: 1.0,
        }
    }
}

/// Per-output `RenderSettings`: the twin of [`crate::render::ScreenContent`]
/// for live-adjustable settings rather than content. Content became
/// genuinely independent per output (different outputs can show different
/// wallpapers); quality/volume should follow the same shape rather than
/// stay one setting forcibly shared across every output in a `run` —
/// otherwise a heavy scene on one monitor would force every *other*
/// monitor's simple wallpaper down to the same quality level too.
///
/// Each `Arc<Mutex<RenderSettings>>` is independently live-mutable (the GUI
/// slider path keeps working exactly as before for the single-wallpaper
/// case, where `by_output` is empty and every output resolves to the same
/// `default`).
#[derive(Clone)]
pub struct ScreenSettings {
    pub by_output: HashMap<String, Arc<Mutex<RenderSettings>>>,
    pub default: Arc<Mutex<RenderSettings>>,
}

impl ScreenSettings {
    /// The common case: one settings object, shared by every output — what
    /// `set`/`set-file` and the GUI preview always use.
    pub fn single(settings: Arc<Mutex<RenderSettings>>) -> Self {
        Self {
            by_output: HashMap::new(),
            default: settings,
        }
    }

    /// The settings object to use for an output with this name (`None` when
    /// the platform can't identify the output).
    pub fn resolve(&self, output_name: Option<&str>) -> Arc<Mutex<RenderSettings>> {
        output_name
            .and_then(|name| self.by_output.get(name))
            .cloned()
            .unwrap_or_else(|| self.default.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_resolves_to_the_same_arc_for_any_output() {
        let settings = Arc::new(Mutex::new(RenderSettings::default()));
        let screen_settings = ScreenSettings::single(Arc::clone(&settings));
        assert!(Arc::ptr_eq(&screen_settings.resolve(None), &settings));
        assert!(Arc::ptr_eq(&screen_settings.resolve(Some("DP-1")), &settings));
    }

    #[test]
    fn resolve_prefers_the_named_output_entry_over_default() {
        let dp1 = Arc::new(Mutex::new(RenderSettings::default()));
        let default = Arc::new(Mutex::new(RenderSettings::default()));
        let mut by_output = HashMap::new();
        by_output.insert("DP-1".to_string(), Arc::clone(&dp1));
        let screen_settings = ScreenSettings {
            by_output,
            default: Arc::clone(&default),
        };

        assert!(Arc::ptr_eq(&screen_settings.resolve(Some("DP-1")), &dp1));
        assert!(Arc::ptr_eq(
            &screen_settings.resolve(Some("HDMI-1")),
            &default
        ));
        assert!(Arc::ptr_eq(&screen_settings.resolve(None), &default));
    }
}
