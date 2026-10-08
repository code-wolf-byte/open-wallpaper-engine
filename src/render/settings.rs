use crate::platform::RenderQuality;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Render settings shared between the UI and the wallpaper thread.
#[derive(Debug)]
pub struct RenderSettings {
    /// GPU render quality / source down-sample factor.
    pub quality: RenderQuality,
    /// Master volume in 0.0–1.0, applied to every wallpaper audio stream via
    /// `engine::audio::set_master_volume` each frame.
    pub volume: f32,
    /// Frame-rate cap for animated content.
    pub fps: u32,
    /// How video, image and web content fits an output of another aspect.
    pub scaling: ScaleMode,
}

/// How a frame of one aspect ratio fills an output of another.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ScaleMode {
    /// Cover the output, cropping the overflow (WE's default).
    #[default]
    Fill,
    /// Fit inside the output, with black bars.
    Fit,
    /// Stretch to the output, ignoring aspect.
    Stretch,
    /// Native size, centred (cropped or bordered).
    Center,
}

impl ScaleMode {
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "fill" | "cover" => Some(Self::Fill),
            "fit" | "contain" => Some(Self::Fit),
            "stretch" => Some(Self::Stretch),
            "center" | "centre" => Some(Self::Center),
            _ => None,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Fill => "fill",
            Self::Fit => "fit",
            Self::Stretch => "stretch",
            Self::Center => "center",
        }
    }

    /// `(uv_scale, uv_offset)` mapping output UV to source UV
    /// (`src = dst * scale + offset`); source UVs outside `[0,1]` are black.
    pub fn uv_transform(self, src: (u32, u32), dst: (u32, u32)) -> ([f32; 2], [f32; 2]) {
        let (sw, sh) = (src.0.max(1) as f32, src.1.max(1) as f32);
        let (dw, dh) = (dst.0.max(1) as f32, dst.1.max(1) as f32);
        let scale = match self {
            Self::Stretch => return ([1.0, 1.0], [0.0, 0.0]),
            Self::Fill => (dw / sw).max(dh / sh),
            Self::Fit => (dw / sw).min(dh / sh),
            Self::Center => 1.0,
        };
        let uv_scale = [dw / (sw * scale), dh / (sh * scale)];
        (
            uv_scale,
            [(1.0 - uv_scale[0]) * 0.5, (1.0 - uv_scale[1]) * 0.5],
        )
    }
}

impl ScaleMode {
    /// [`uv_transform`](Self::uv_transform) for a view showing only
    /// `region` (`[x, y, w, h]`, normalized) of a `canvas` the source is
    /// fitted to — one monitor's share of a spanned wallpaper.
    pub fn uv_transform_region(
        self,
        src: (u32, u32),
        canvas: (u32, u32),
        region: [f32; 4],
    ) -> ([f32; 2], [f32; 2]) {
        let (s, o) = self.uv_transform(src, canvas);
        (
            [s[0] * region[2], s[1] * region[3]],
            [o[0] + s[0] * region[0], o[1] + s[1] * region[1]],
        )
    }
}

/// The whole canvas, for [`ScaleMode::uv_transform_region`].
pub const FULL_REGION: [f32; 4] = [0.0, 0.0, 1.0, 1.0];

/// The frame-rate cap when none is configured.
pub const DEFAULT_FPS: u32 = 30;

impl Default for RenderSettings {
    fn default() -> Self {
        Self {
            quality: RenderQuality::Ultra,
            volume: 1.0,
            fps: DEFAULT_FPS,
            scaling: ScaleMode::default(),
        }
    }
}

#[cfg(test)]
mod scale_tests {
    use super::*;

    #[test]
    fn a_region_shows_its_part_of_the_canvas() {
        // A 32:9 video spanned over two 16:9 monitors: the left one shows
        // exactly the left half.
        let (s, o) =
            ScaleMode::Fill.uv_transform_region((3840, 1080), (3840, 1080), [0.0, 0.0, 0.5, 1.0]);
        assert_eq!((s, o), ([0.5, 1.0], [0.0, 0.0]));
        let (s, o) =
            ScaleMode::Fill.uv_transform_region((3840, 1080), (3840, 1080), [0.5, 0.0, 0.5, 1.0]);
        assert_eq!((s, o), ([0.5, 1.0], [0.5, 0.0]));
        // The full region is the plain transform.
        assert_eq!(
            ScaleMode::Fit.uv_transform_region((1920, 1080), (1600, 1200), FULL_REGION),
            ScaleMode::Fit.uv_transform((1920, 1080), (1600, 1200))
        );
    }

    #[test]
    fn fill_crops_and_fit_letterboxes() {
        // 16:9 video on a 4:3 output.
        let (s, o) = ScaleMode::Fill.uv_transform((1920, 1080), (1600, 1200));
        assert!((s[1] - 1.0).abs() < 1e-5 && s[0] < 1.0);
        assert!((o[0] - (1.0 - s[0]) / 2.0).abs() < 1e-6);
        let (s, _) = ScaleMode::Fit.uv_transform((1920, 1080), (1600, 1200));
        assert!((s[0] - 1.0).abs() < 1e-5 && s[1] > 1.0);
        assert_eq!(
            ScaleMode::Stretch.uv_transform((1, 1), (5, 5)),
            ([1.0, 1.0], [0.0, 0.0])
        );
        let (s, o) = ScaleMode::Center.uv_transform((100, 100), (200, 100));
        assert_eq!(s, [2.0, 1.0]);
        assert_eq!(o, [-0.5, 0.0]);
    }
}

impl RenderSettings {
    /// Minimum time between frames for the configured cap (1–240 fps).
    pub fn frame_interval(&self) -> std::time::Duration {
        std::time::Duration::from_secs_f64(1.0 / self.fps.clamp(1, 240) as f64)
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
        assert!(Arc::ptr_eq(
            &screen_settings.resolve(Some("DP-1")),
            &settings
        ));
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
