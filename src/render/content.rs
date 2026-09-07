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
    /// Parse a Workshop `Wallpaper` into the appropriate content variant.
    ///
    /// Returns `Err` for types that are not yet renderable (Scene, Web,
    /// Application) so callers get a clear diagnostic instead of a generic
    /// image-loading failure.
    pub fn from_wallpaper(w: &Wallpaper) -> Result<Self> {
        match w.wallpaper_type() {
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
}

impl ScreenContent {
    /// The common case: one wallpaper, shown on every output — what `set`/
    /// `set-file` always use, and what `run` falls back to when no
    /// screen-specific assignment was ever configured.
    pub fn single(content: WallpaperContent) -> Self {
        Self {
            by_output: std::collections::HashMap::new(),
            default: content,
        }
    }

    /// The content to show on an output with this name (`None` when the
    /// platform can't identify the output, e.g. no xdg-output support).
    pub fn resolve(&self, output_name: Option<&str>) -> WallpaperContent {
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
        };

        assert_eq!(scene_dir(&content.resolve(Some("DP-1"))), "wallpaper-a");
        assert_eq!(scene_dir(&content.resolve(Some("HDMI-1"))), "wallpaper-b");
    }
}
