//! What every wallpaper output does whatever way it reaches the screen,
//! shared by the Wayland and X11 backends: building an output's renderer,
//! applying live user-property edits, and deciding from the playback rules
//! (`engine::playback_gate`) whether to run, mute, pause or stop. The
//! backends keep only what is theirs: querying window states and getting
//! pixels onto the screen.

use std::path::{Path, PathBuf};
use std::sync::mpsc::SyncSender;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::Result;

use super::web::{WebInputEvent, WebOptions};
use super::{FrameSource, RenderSettings, WallpaperContent};
use crate::engine::gpu::renderer::GpuSceneInstance;
use crate::services::media::{MediaWatcher, PlaybackState as MediaState};
use crate::services::playback_gate::{self, PlaybackAction, PlaybackPolicy, PlaybackState};
use crate::services::power::{self, LockWatcher};

/// How often the playback conditions are re-checked. Each check is real I/O
/// (sysfs reads, a D-Bus call, a window-manager query), so this stays well
/// below frame rate.
pub const PAUSE_CHECK_INTERVAL: Duration = Duration::from_millis(750);

/// How an output's wallpaper produces pixels.
pub enum ContentRenderer {
    /// CPU frames: images, video, web pages, and the scene fallback loop.
    Frames(FrameSource),
    /// A scene rendered on the backend's own GPU device.
    Scene(Box<GpuSceneInstance>),
}

impl ContentRenderer {
    /// The renderer for one output's content. Scenes render on `device`
    /// unless `gpu_scene` is off; everything else, and a scene whose GPU
    /// setup fails, produces CPU frames.
    pub fn build(
        content: WallpaperContent,
        device: wgpu::Device,
        queue: wgpu::Queue,
        gpu_scene: bool,
        web: WebOptions,
    ) -> Result<Self> {
        Ok(match content {
            WallpaperContent::Scene { dir } if gpu_scene => {
                match GpuSceneInstance::with_device(device, queue, &dir) {
                    Ok(instance) => Self::Scene(Box::new(instance)),
                    Err(e) => {
                        tracing::warn!(target: "wallpaper", "GPU scene init failed ({e}); using frame-loop fallback");
                        Self::Frames(FrameSource::from_content(WallpaperContent::Scene { dir })?)
                    }
                }
            }
            other => Self::Frames(FrameSource::from_content_with(other, web)?),
        })
    }

    pub fn is_animated(&self) -> bool {
        match self {
            Self::Frames(fs) => fs.is_animated(),
            Self::Scene(_) => true,
        }
    }

    /// The page's input channel when this is a web wallpaper.
    pub fn web_input(&self) -> Option<&SyncSender<WebInputEvent>> {
        match self {
            Self::Frames(fs) => fs.web_input(),
            Self::Scene(_) => None,
        }
    }

    /// The next CPU frame: advances video and web content, renders a scene.
    pub fn next_frame(&mut self) -> Result<Arc<image::RgbaImage>> {
        match self {
            Self::Frames(fs) => {
                fs.try_advance();
                Ok(Arc::clone(fs.current_frame()))
            }
            Self::Scene(instance) => Ok(Arc::new(instance.render_rgba()?)),
        }
    }

    /// Show live edits of user properties `names` of the wallpaper in `dir`:
    /// scenes in place, web pages through `applyUserProperties`. Returns
    /// whether that worked; `false` means rebuild the renderer instead.
    pub fn apply_properties(&mut self, dir: &Path, names: &[String]) -> bool {
        match self {
            Self::Scene(scene) => scene.set_user_properties(names),
            Self::Frames(fs) => {
                if let Some(tx) = fs.web_input() {
                    let json = super::web::user_properties_subset(dir, names);
                    let _ = tx.try_send(WebInputEvent::Properties(json));
                }
                true
            }
        }
    }
}

/// Property edits made since `seen` that concern the wallpaper an output
/// shows (`content`): its folder and the changed names. Advances `seen`.
pub fn pending_property_edits(
    settings: &Mutex<RenderSettings>,
    seen: &mut usize,
    content: &WallpaperContent,
) -> Option<(PathBuf, Vec<String>)> {
    let settings = settings.lock().unwrap();
    if settings.property_edits.len() == *seen {
        return None;
    }
    let Some(dir) = content.project_dir() else {
        *seen = settings.property_edits.len();
        return None;
    };
    let names = settings.property_edits_since(seen, dir);
    (!names.is_empty()).then(|| (dir.to_path_buf(), names))
}

/// What the window manager reports about the active window, for the
/// playback rules.
#[derive(Debug, Clone, Copy, Default)]
pub struct WindowState {
    pub fullscreen: bool,
    pub maximized: bool,
    /// An application window (not the desktop or a panel) is active.
    pub focused: bool,
}

/// The playback rules and the system state they look at.
pub struct PlaybackMonitor {
    policy: PlaybackPolicy,
    /// MPRIS watcher for "another application is playing audio"; only
    /// started when the policy acts on it.
    media: Option<MediaWatcher>,
    media_playing: bool,
    lock: Option<LockWatcher>,
    state: PlaybackState,
    last_check: Option<Instant>,
}

impl Default for PlaybackMonitor {
    fn default() -> Self {
        Self::new()
    }
}

impl PlaybackMonitor {
    pub fn new() -> Self {
        let policy = PlaybackPolicy::load();
        Self {
            policy,
            media: policy.watches_audio().then(MediaWatcher::start).flatten(),
            media_playing: false,
            lock: LockWatcher::start(),
            state: PlaybackState::default(),
            last_check: None,
        }
    }

    /// The action the rules call for at `now`. Conditions are re-checked at
    /// most every [`PAUSE_CHECK_INTERVAL`]; `windows` is only called then.
    pub fn action(
        &mut self,
        now: Instant,
        windows: impl FnOnce() -> WindowState,
    ) -> PlaybackAction {
        if self
            .last_check
            .is_none_or(|t| now.duration_since(t) >= PAUSE_CHECK_INTERVAL)
        {
            self.last_check = Some(now);
            let windows = windows();
            while let Some(info) = self.media.as_ref().and_then(|m| m.try_recv()) {
                self.media_playing = info.playback_state == MediaState::Playing;
            }
            self.state = PlaybackState {
                on_battery: power::is_on_battery(),
                locked_or_sleeping: self.lock.as_mut().is_some_and(|l| l.is_locked()),
                fullscreen_app: windows.fullscreen,
                maximized_app: windows.maximized,
                focused_app: windows.focused,
                other_audio: self.media_playing,
            };
        }
        playback_gate::decide(&self.policy, &self.state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::content::settings::PropertyEdit;

    #[test]
    fn pending_edits_pick_this_wallpaper_and_advance() {
        let mine = std::env::temp_dir();
        let edit = |dir: &Path, name: &str| PropertyEdit {
            dir: std::fs::canonicalize(dir).unwrap(),
            name: name.into(),
        };
        let settings = Mutex::new(RenderSettings::default());
        let content = WallpaperContent::Scene { dir: mine.clone() };
        let mut seen = 0;
        assert!(pending_property_edits(&settings, &mut seen, &content).is_none());

        settings.lock().unwrap().property_edits = vec![
            edit(&mine, "speed"),
            edit(Path::new("/"), "other wallpaper"),
            edit(&mine, "color"),
        ];
        let (dir, names) = pending_property_edits(&settings, &mut seen, &content).unwrap();
        assert_eq!(dir, mine);
        assert_eq!(names, ["speed", "color"]);
        assert_eq!(seen, 3);
        assert!(pending_property_edits(&settings, &mut seen, &content).is_none());

        // Content without a folder just catches up.
        settings
            .lock()
            .unwrap()
            .property_edits
            .push(edit(&mine, "x"));
        let image = WallpaperContent::Static(Arc::new(image::RgbaImage::new(1, 1)));
        assert!(pending_property_edits(&settings, &mut seen, &image).is_none());
        assert_eq!(seen, 4);
    }
}
