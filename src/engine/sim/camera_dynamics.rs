//! Per-frame camera dynamics: shake, fade-in, and parallax displacement.
//!
//! Mirrors linux-wallpaperengine's scene-level camera settings, which are all
//! user-settable general fields resolved through the property system:
//! `camerashake{,amplitude,roughness,speed}`, `camerafade`, and
//! `cameraparallax{,amount,delay,mouseinfluence}` (see WallpaperParser.cpp).

use crate::engine::scene::{parse_value_bool, parse_value_f32, Scene};

/// How long the camera-fade ramp lasts after the scene starts, in seconds.
const FADE_DURATION_SECS: f32 = 1.2;

/// Static camera-dynamics configuration parsed from scene.json `general`.
#[derive(Debug, Clone, Default)]
pub struct CameraDynamics {
    pub shake_enabled: bool,
    pub shake_amplitude: f32,
    pub shake_roughness: f32,
    pub shake_speed: f32,
    pub fade_enabled: bool,
    pub parallax_enabled: bool,
    pub parallax_amount: f32,
    pub parallax_delay: f32,
    pub parallax_mouse_influence: f32,
    parallax_displacement: [f32; 2],
}

/// Per-frame output of [`CameraDynamics::update`].
#[derive(Debug, Clone, Copy)]
pub struct CameraFrameDynamics {
    /// Camera shake offset as a fraction of scene size (applied to every layer).
    pub shake_offset: [f32; 2],
    /// Parallax displacement as a fraction of scene size; each layer scales
    /// this by its own `parallaxDepth`.
    pub parallax_displacement: [f32; 2],
    /// Global scene opacity from the camera fade-in ramp (1.0 once settled).
    pub fade: f32,
}

impl Default for CameraFrameDynamics {
    fn default() -> Self {
        Self {
            shake_offset: [0.0, 0.0],
            parallax_displacement: [0.0, 0.0],
            fade: 1.0,
        }
    }
}

impl CameraDynamics {
    pub fn from_scene(scene: &Scene) -> Self {
        let Some(general) = scene.general.as_ref() else {
            return Self::default();
        };
        let get_f32 = |v: &Option<serde_json::Value>, default: f32| -> f32 {
            v.as_ref().and_then(parse_value_f32).unwrap_or(default)
        };
        let get_bool = |v: &Option<serde_json::Value>| -> bool {
            v.as_ref().and_then(parse_value_bool).unwrap_or(false)
        };
        Self {
            shake_enabled: get_bool(&general.camera_shake),
            shake_amplitude: get_f32(&general.camera_shake_amplitude, 0.0),
            shake_roughness: get_f32(&general.camera_shake_roughness, 0.0),
            shake_speed: get_f32(&general.camera_shake_speed, 0.0),
            fade_enabled: get_bool(&general.camera_fade),
            parallax_enabled: get_bool(&general.camera_parallax),
            parallax_amount: get_f32(&general.camera_parallax_amount, 1.0),
            parallax_delay: get_f32(&general.camera_parallax_delay, 0.0),
            parallax_mouse_influence: get_f32(&general.camera_parallax_mouse_influence, 1.0),
            parallax_displacement: [0.0, 0.0],
        }
    }

    /// True when any dynamic behaviour is active (callers can skip per-frame
    /// work entirely for static cameras).
    pub fn is_active(&self) -> bool {
        (self.shake_enabled && self.shake_amplitude != 0.0)
            || self.fade_enabled
            || self.parallax_enabled
    }

    /// Advance the dynamics one frame.
    ///
    /// `time` is seconds since the scene started, `delta` is seconds since the
    /// previous frame, `mouse_norm` is the pointer position in [0,1]² (pass
    /// `[0.5, 0.5]` when no pointer input is available — parallax then rests
    /// at the neutral position).
    pub fn update(&mut self, time: f32, delta: f32, mouse_norm: [f32; 2]) -> CameraFrameDynamics {
        let mut out = CameraFrameDynamics::default();

        // Intentional deviation from linux-wallpaperengine: the reference parses
        // camerashake/amplitude/roughness/speed (WallpaperParser.cpp) and exposes
        // them as read-only JS getters (Scripting/SceneObject.cpp), but its
        // native Render/ code and builtins.js never actually apply a shake
        // offset anywhere — camerashake-enabled wallpapers render static on
        // that engine. We apply an approximated shake here on purpose, since
        // it matches the real Windows Wallpaper Engine's intent even though
        // this particular Linux port never implemented it. Don't "fix" this
        // to match the reference without discussing it first.
        if self.shake_enabled && self.shake_amplitude != 0.0 {
            out.shake_offset = self.shake_offset_at(time);
        }

        if self.parallax_enabled {
            // Port of CScene::renderFrame: displacement eases toward
            // (mouse - center) * amount * influence at `delay` speed.
            let centered = [mouse_norm[0] - 0.5, mouse_norm[1] - 0.5];
            let target = [
                centered[0] * self.parallax_amount * self.parallax_mouse_influence,
                centered[1] * self.parallax_amount * self.parallax_mouse_influence,
            ];
            // delay == 0 in the C++ formula would freeze the displacement at
            // its initial value; treat it as "no easing" instead.
            let t = if self.parallax_delay > 0.0 {
                (self.parallax_delay * delta).clamp(0.0, 1.0)
            } else {
                1.0
            };
            self.parallax_displacement = [
                self.parallax_displacement[0] + (target[0] - self.parallax_displacement[0]) * t,
                self.parallax_displacement[1] + (target[1] - self.parallax_displacement[1]) * t,
            ];
            out.parallax_displacement = self.parallax_displacement;
        }

        if self.fade_enabled {
            out.fade = (time / FADE_DURATION_SECS).clamp(0.0, 1.0);
            // Ease-out so the ramp doesn't end abruptly.
            out.fade = out.fade * (2.0 - out.fade);
        }

        out
    }

    /// Smooth pseudo-random shake using layered sines. Amplitude is expressed
    /// in Wallpaper Engine's editor units (~0..2) and mapped to a small
    /// fraction of the scene so shakes stay subtle like the original.
    fn shake_offset_at(&self, time: f32) -> [f32; 2] {
        let speed = if self.shake_speed > 0.0 {
            self.shake_speed
        } else {
            1.0
        };
        let rough = 1.0 + self.shake_roughness.max(0.0);
        let t = time * speed;
        // Two incommensurate frequencies per axis produce non-repeating motion;
        // roughness raises the high-frequency contribution.
        let x = (t * 1.1).sin() + 0.6 * (t * 2.7 * rough + 1.3).sin();
        let y = (t * 0.9 + 4.2).sin() + 0.6 * (t * 3.1 * rough + 0.7).sin();
        let scale = self.shake_amplitude * 0.005; // amplitude 1.0 → ±0.8% of scene
        [x * scale, y * scale]
    }
}
