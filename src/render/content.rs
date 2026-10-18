use anyhow::{anyhow, Result};
use image::RgbaImage;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::workshop::{Wallpaper, WallpaperType};

/// The resolved content of a wallpaper, ready to hand to the renderer.
///
/// This is the primary extension point as new wallpaper types are supported:
/// add a new variant here, implement it in `from_wallpaper`/`from_path`, and
/// add a corresponding `FrameSource` variant in `frame.rs`.
///
/// `Clone` is cheap for every variant (an `Arc` clone or a `PathBuf` clone) —
/// needed so the same content value can seed independent per-output renderer
/// state (see [`ScreenContent`]) without re-resolving it from disk per output.
#[derive(Clone)]
pub enum WallpaperContent {
    /// A pre-loaded static RGBA image (PNG, JPEG, …).
    Static(Arc<RgbaImage>),
    /// A video file to be decoded frame-by-frame with FFmpeg.
    Video { path: PathBuf },
    /// A scene wallpaper — rendered by compositing decoded .tex layers.
    Scene { dir: PathBuf },
    /// A web wallpaper — an HTML entry point rendered off-screen by CEF.
    /// Only renderable when built with the `web` feature.
    Web { html: PathBuf },
    // Future variants (not yet implemented):
    // Application { exe: PathBuf },
}

impl WallpaperContent {
    /// The wallpaper folder user properties belong to, if any.
    pub fn project_dir(&self) -> Option<&std::path::Path> {
        match self {
            Self::Scene { dir } => Some(dir),
            Self::Web { html } => html.parent(),
            Self::Video { path } => path.parent(),
            Self::Static(_) => None,
        }
    }

    /// Parse a Workshop `Wallpaper` into the appropriate content variant.
    ///
    /// Returns `Err` for types that are not yet renderable (Scene, Web,
    /// Application) so callers get a clear diagnostic instead of a generic
    /// image-loading failure.
    pub fn from_wallpaper(w: &Wallpaper) -> Result<Self> {
        match w.wallpaper_type() {
            WallpaperType::Preset => Self::from_preset_dir(&w.path),
            WallpaperType::Scene => {
                let dir = w.path.clone();
                if dir.join("scene.json").exists() || dir.join("scene.pkg").exists() {
                    Ok(WallpaperContent::Scene { dir })
                } else {
                    Err(anyhow!(
                        "scene wallpaper missing scene.json and scene.pkg in {}",
                        dir.display()
                    ))
                }
            }
            WallpaperType::Web => {
                let html = w
                    .wallpaper_file()
                    .ok_or_else(|| anyhow!("web wallpaper has no file field in project.json"))?;
                if !html.exists() {
                    return Err(anyhow!(
                        "web wallpaper entry point not found: {}",
                        html.display()
                    ));
                }
                Ok(WallpaperContent::Web { html })
            }
            WallpaperType::Application => Err(anyhow!("application wallpapers are Windows-only")),
            WallpaperType::Video | WallpaperType::Unknown => {
                let path = w
                    .wallpaper_file()
                    .ok_or_else(|| anyhow!("wallpaper has no file field in project.json"))?;
                if !path.exists() {
                    return Err(anyhow!("wallpaper file not found: {}", path.display()));
                }
                Self::from_path(&path)
            }
        }
    }

    /// Resolve content from any filesystem path: a wallpaper directory
    /// (scene.json / scene.pkg / project.json) or a plain image/video file.
    #[tracing::instrument(target = "content", level = "debug", fields(path = %path.display()))]
    pub fn from_any_path(path: &Path) -> Result<Self> {
        if path.is_dir() && Self::is_preset_dir(path) {
            return Self::from_preset_dir(path);
        }
        if path.is_dir() {
            if path.join("scene.json").exists() || path.join("scene.pkg").exists() {
                tracing::debug!(target: "content", "resolved as scene wallpaper");
                return Ok(WallpaperContent::Scene {
                    dir: path.to_owned(),
                });
            }
            // A wallpaper directory with project.json but no scene data:
            // resolve the `file` entry (video/image wallpapers).
            let project_path = path.join("project.json");
            if project_path.exists() {
                let data = std::fs::read_to_string(&project_path)?;
                let project: serde_json::Value = serde_json::from_str(&data)?;
                if let Some(file) = project.get("file").and_then(|f| f.as_str()) {
                    return Self::from_path(&path.join(file));
                }
            }
            return Err(anyhow!(
                "directory {} contains no scene.json, scene.pkg, or project.json file entry",
                path.display()
            ));
        }
        Self::from_path(path)
    }

    fn is_preset_dir(dir: &Path) -> bool {
        crate::workshop::load_project(dir)
            .map(|p| p.wallpaper_type.is_none() && p.dependency.is_some() && p.preset.is_some())
            .unwrap_or(false)
    }

    /// `dir` itself, or — for a preset item — the directory of the item it
    /// builds on, after registering the preset's values as its overrides.
    pub fn preset_base_dir(dir: &Path) -> Result<PathBuf> {
        if !dir.is_dir() || !Self::is_preset_dir(dir) {
            return Ok(dir.to_path_buf());
        }
        let project = crate::workshop::load_project(dir)?;
        let dep = project
            .dependency
            .as_deref()
            .ok_or_else(|| anyhow!("preset {} has no dependency", dir.display()))?;
        let parent = crate::workshop::find_item_dir(dep, Some(dir)).ok_or_else(|| {
            anyhow!(
                "preset {} needs workshop item {dep}, which is not installed — subscribe to it first",
                dir.display()
            )
        })?;
        if let Some(preset) = &project.preset {
            crate::engine::properties::register_preset(&parent, dir, preset);
        }
        tracing::info!(target: "content", "preset {} → base item {}", dir.display(), parent.display());
        Ok(parent)
    }

    /// A preset item: the content of the item it builds on, with the
    /// preset's values registered as that item's property overrides.
    fn from_preset_dir(dir: &Path) -> Result<Self> {
        Self::from_any_path(&Self::preset_base_dir(dir)?)
    }

    /// Detect content type from a raw file path, guessing by extension.
    ///
    /// Video extensions are decoded with FFmpeg; everything else is opened
    /// by the `image` crate as a static image.
    pub fn from_path(path: &Path) -> Result<Self> {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();

        match ext.as_str() {
            "mp4" | "webm" | "mkv" | "avi" | "mov" | "flv" | "wmv" => {
                tracing::debug!(target: "content", %ext, "resolved as video wallpaper");
                Ok(WallpaperContent::Video {
                    path: path.to_owned(),
                })
            }
            "html" | "htm" => {
                tracing::debug!(target: "content", %ext, "resolved as web wallpaper");
                Ok(WallpaperContent::Web {
                    html: path.to_owned(),
                })
            }
            _ => {
                tracing::debug!(target: "content", %ext, "resolved as static image wallpaper");
                let img = image::open(path)
                    .map_err(|e| anyhow!("failed to load image {}: {}", path.display(), e))?
                    .into_rgba8();
                Ok(WallpaperContent::Static(Arc::new(img)))
            }
        }
    }
}

/// Per-output wallpaper content: what a [`crate::platform::DisplayPlatform`]
/// resolves for each display it discovers, so different outputs can show
/// different wallpapers (`wp-engine config set-screen`).
///
/// `by_output` is keyed by the platform's own output-name convention (a
/// Wayland `wl_output`/xdg-output name like `"DP-1"`, or an X11 RandR output
/// name like `"HDMI-1"`) — the same string `wp-engine config set-screen`
/// takes. Outputs with no entry (including every output on platforms/paths
/// that only ever discover one target, or when nothing was configured at
/// all) fall back to `default`.
#[derive(Clone)]
pub struct ScreenContent {
    pub by_output: std::collections::HashMap<String, WallpaperContent>,
    pub default: WallpaperContent,
    /// Span or clone show `default` from one shared renderer and ignore
    /// `by_output`.
    pub layout: ScreenLayout,
}

/// How a wallpaper covers several monitors (Wallpaper Engine's layout
/// setting).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ScreenLayout {
    /// Each monitor runs its own wallpaper (`screens`, else the default).
    #[default]
    PerScreen,
    /// One wallpaper stretched across the bounding box of all monitors.
    Span,
    /// One wallpaper rendered once and shown on every monitor.
    Clone,
}

impl ScreenLayout {
    pub const ALL: [Self; 3] = [Self::PerScreen, Self::Span, Self::Clone];

    pub fn parse(s: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|l| l.label() == s.trim())
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::PerScreen => "per-screen",
            Self::Span => "span",
            Self::Clone => "clone",
        }
    }
}

impl ScreenContent {
    /// The common case: one wallpaper, shown on every output — what `set`/
    /// `set-file` always use, and what `run` falls back to when no
    /// screen-specific assignment was ever configured.
    pub fn single(content: WallpaperContent) -> Self {
        Self {
            by_output: std::collections::HashMap::new(),
            default: content,
            layout: ScreenLayout::PerScreen,
        }
    }

    /// The content to show on an output with this name (`None` when the
    /// platform can't identify the output, e.g. no xdg-output support).
    pub fn resolve(&self, output_name: Option<&str>) -> WallpaperContent {
        if self.layout != ScreenLayout::PerScreen {
            return self.default.clone();
        }
        output_name
            .and_then(|name| self.by_output.get(name))
            .cloned()
            .unwrap_or_else(|| self.default.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scene(name: &str) -> WallpaperContent {
        WallpaperContent::Scene {
            dir: PathBuf::from(name),
        }
    }

    fn scene_dir(content: &WallpaperContent) -> &str {
        match content {
            WallpaperContent::Scene { dir } => dir.to_str().unwrap(),
            _ => panic!("expected a Scene variant, got a different one"),
        }
    }

    #[test]
    fn single_resolves_to_the_same_content_for_any_output() {
        let content = ScreenContent::single(scene("default"));
        assert_eq!(scene_dir(&content.resolve(None)), "default");
        assert_eq!(scene_dir(&content.resolve(Some("DP-1"))), "default");
        assert_eq!(scene_dir(&content.resolve(Some("HDMI-1"))), "default");
    }

    #[test]
    fn resolve_prefers_the_named_output_entry_over_default() {
        let mut by_output = std::collections::HashMap::new();
        by_output.insert("DP-1".to_string(), scene("dp1-wallpaper"));
        let content = ScreenContent {
            by_output,
            default: scene("default"),
            layout: ScreenLayout::PerScreen,
        };

        assert_eq!(scene_dir(&content.resolve(Some("DP-1"))), "dp1-wallpaper");
        // A different, unconfigured output name falls back to default.
        assert_eq!(scene_dir(&content.resolve(Some("HDMI-1"))), "default");
        // No output name at all (platform couldn't identify it) also falls
        // back to default.
        assert_eq!(scene_dir(&content.resolve(None)), "default");
    }

    #[test]
    fn resolve_handles_multiple_distinct_screens_independently() {
        let mut by_output = std::collections::HashMap::new();
        by_output.insert("DP-1".to_string(), scene("wallpaper-a"));
        by_output.insert("HDMI-1".to_string(), scene("wallpaper-b"));
        let content = ScreenContent {
            by_output,
            default: scene("default"),
            layout: ScreenLayout::PerScreen,
        };

        assert_eq!(scene_dir(&content.resolve(Some("DP-1"))), "wallpaper-a");
        assert_eq!(scene_dir(&content.resolve(Some("HDMI-1"))), "wallpaper-b");
    }

    #[test]
    fn span_and_clone_show_the_default_everywhere() {
        let mut content = ScreenContent::single(scene("default"));
        content.by_output.insert("DP-1".into(), scene("dp1"));
        content.layout = ScreenLayout::Span;
        assert_eq!(scene_dir(&content.resolve(Some("DP-1"))), "default");
        assert_eq!(ScreenLayout::parse("clone"), Some(ScreenLayout::Clone));
    }
}
