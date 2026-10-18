pub mod frame;
pub mod settings;
// Shared by the Wayland and X11 backends; uses the MPRIS watcher, Linux-only.
#[cfg(target_os = "linux")]
pub mod session;
pub mod video;
pub mod wallpaper;
pub mod web;

pub use frame::FrameSource;
pub use settings::{RenderSettings, ScaleMode, ScreenSettings};
pub use wallpaper::{ScreenContent, ScreenLayout, WallpaperContent};
