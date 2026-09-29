//! Wire format between `wp-engine` clients and the daemon.
//!
//! One request per connection: the client writes a single JSON [`Request`]
//! line, the daemon answers with a single JSON [`Response`] line and closes.
//! Newline-delimited JSON keeps it debuggable with `socat`/`nc -U`.

use std::collections::HashMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::application::ApplicationContext;
use crate::platform::RenderQuality;
use crate::render::RenderSettings;

/// Everything an [`ApplicationContext`] carries, in a serializable shape.
/// Quality travels as its [`RenderQuality::label`] string.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct ApplySpec {
    pub background: PathBuf,
    #[serde(default)]
    pub screens: HashMap<String, PathBuf>,
    #[serde(default)]
    pub properties: HashMap<String, String>,
    #[serde(default)]
    pub quality: Option<String>,
    #[serde(default)]
    pub volume: Option<f32>,
    /// Per-screen quality, keyed like `screens`.
    #[serde(default)]
    pub screen_quality: HashMap<String, String>,
    /// Shown by `wp-engine status` / the GUI status bar.
    #[serde(default)]
    pub title: Option<String>,
}

impl ApplySpec {
    /// Snapshot a fully-built context (saved + CLI overrides already applied).
    /// Paths are made absolute: the daemon's working directory is not the
    /// client's.
    pub fn from_context(context: &ApplicationContext, title: Option<String>) -> Self {
        let absolute = |p: &PathBuf| std::path::absolute(p).unwrap_or_else(|_| p.clone());
        let (quality, volume) = {
            let s = context.settings.lock().unwrap();
            (s.quality, s.volume)
        };
        Self {
            background: absolute(&context.background),
            screens: context.screens.iter().map(|(k, p)| (k.clone(), absolute(p))).collect(),
            properties: context.properties.clone(),
            quality: Some(quality.label().to_string()),
            volume: Some(volume),
            screen_quality: context
                .screen_settings
                .iter()
                .map(|(screen, s)| (screen.clone(), s.lock().unwrap().quality.label().to_string()))
                .collect(),
            title,
        }
    }

    /// Rebuild the context on the daemon side. Unknown quality labels are
    /// ignored (left at the default) rather than failing the whole apply.
    pub fn into_context(self) -> ApplicationContext {
        let mut context = ApplicationContext::new(self.background);
        context.screens = self.screens;
        context.properties = self.properties;
        {
            let mut s = context.settings.lock().unwrap();
            if let Some(q) = self.quality.as_deref().and_then(RenderQuality::parse) {
                s.quality = q;
            }
            if let Some(v) = self.volume {
                s.volume = v.clamp(0.0, 1.0);
            }
        }
        for (screen, q) in self.screen_quality {
            if let Some(quality) = RenderQuality::parse(&q) {
                context.screen_settings.insert(
                    screen,
                    std::sync::Arc::new(std::sync::Mutex::new(RenderSettings {
                        quality,
                        ..Default::default()
                    })),
                );
            }
        }
        context
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "cmd", rename_all = "kebab-case")]
pub enum Request {
    /// Liveness check.
    Ping,
    /// Current daemon state.
    Status,
    /// Replace whatever is showing with this wallpaper.
    Apply(ApplySpec),
    /// Live-adjust the active wallpaper's default render quality.
    SetQuality { quality: String },
    /// Live-adjust the active wallpaper's volume (0.0–1.0).
    SetVolume { volume: f32 },
    /// Audio capture device for audio-reactive wallpapers; `None` =
    /// automatic. Takes effect on the next apply.
    SetAudioDevice { device: Option<String> },
    /// Remove the wallpaper but keep the daemon running.
    Clear,
    /// Remove the wallpaper and exit the daemon.
    Shutdown,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ActiveWallpaper {
    pub background: PathBuf,
    pub title: Option<String>,
    pub quality: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DaemonStatus {
    pub pid: u32,
    pub active: Option<ActiveWallpaper>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "result", rename_all = "kebab-case")]
pub enum Response {
    Ok,
    Status(DaemonStatus),
    Error { message: String },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_round_trips_as_one_json_line() {
        let req = Request::Apply(ApplySpec {
            background: "/w/123".into(),
            quality: Some("High".into()),
            title: Some("Lake".into()),
            ..Default::default()
        });
        let line = serde_json::to_string(&req).unwrap();
        assert!(!line.contains('\n'));
        assert_eq!(serde_json::from_str::<Request>(&line).unwrap(), req);
        assert_eq!(
            serde_json::from_str::<Request>(r#"{"cmd":"shutdown"}"#).unwrap(),
            Request::Shutdown
        );
    }

    #[test]
    fn apply_spec_survives_a_context_round_trip() {
        let mut context = ApplicationContext::new("/w/default".into());
        context.screens.insert("DP-1".into(), "/w/dp1".into());
        context.properties.insert("speed".into(), "2".into());
        context.settings.lock().unwrap().quality = RenderQuality::Low;
        context.screen_settings.insert(
            "DP-1".into(),
            std::sync::Arc::new(std::sync::Mutex::new(RenderSettings {
                quality: RenderQuality::Medium,
                ..Default::default()
            })),
        );

        let back = ApplySpec::from_context(&context, None).into_context();
        assert_eq!(back.background, context.background);
        assert_eq!(back.screens, context.screens);
        assert_eq!(back.properties, context.properties);
        assert_eq!(back.settings.lock().unwrap().quality, RenderQuality::Low);
        assert_eq!(
            back.screen_settings["DP-1"].lock().unwrap().quality,
            RenderQuality::Medium
        );
    }
}
