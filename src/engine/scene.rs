use anyhow::{Context, Result};
use serde::Deserialize;

/// Top-level scene.json structure for Wallpaper Engine scene wallpapers.
///
/// Fields use `serde_json::Value` or `Option` liberally because Wallpaper
/// Engine's format varies wildly across versions and creators.
#[derive(Debug, Deserialize)]
pub struct Scene {
    pub camera: Option<Camera>,
    pub general: Option<General>,
    #[serde(default)]
    pub objects: Vec<SceneObject>,
}

impl Scene {
    /// True for genuine 3D scenes: no usable orthogonal projection anywhere
    /// (the key may be absent, JSON `null`, or missing width/height — WE
    /// writes `"orthogonalprojection": null` for perspective scenes) and a
    /// camera with parseable eye+center. The reference has no perspective
    /// path (CScene.cpp always calls setOrthogonalProjection), so these
    /// scenes are broken there too; we project them ourselves.
    pub fn is_perspective(&self) -> bool {
        let usable = |o: &Option<OrthogonalProjection>| {
            o.as_ref()
                .is_some_and(|p| (p.width.is_some() && p.height.is_some()) || p.auto == Some(true))
        };
        let has_ortho = self
            .camera
            .as_ref()
            .is_some_and(|c| usable(&c.orthogonal_projection))
            || self
                .general
                .as_ref()
                .is_some_and(|g| usable(&g.orthogonal_projection));
        !has_ortho
            && self
                .camera
                .as_ref()
                .is_some_and(|c| c.parsed_eye().is_some() && c.parsed_center().is_some())
    }

    /// Resolve every `light` scene object into a world-space light for the
    /// mesh3d lighting pass. `SceneObject::light_type()` (the `light`
    /// field's own string value — see its doc comment) is the primary
    /// type signal; cone-angle presence is a secondary fallback so content
    /// authoring the common `"light": true` boolean form (which never
    /// matches the type-discriminant table, so the real engine itself
    /// treats it as `Point`) can still resolve to `Spot` when the author's
    /// intent is otherwise clear from `innercone`/`outercone` being set.
    ///
    /// Every `light` scene object as `(light, casts_shadow, volumetrics,
    /// cookie_path)`. Paired rather than parallel vecs so a caller can
    /// never let a flag drift out of sync with the light it belongs to
    /// after filtering. `volumetrics` is `Some((density, exponent))` when
    /// `castvolumetrics` is set — see `SceneObject::volumetrics_params` and
    /// the Ghidra report's `_rt_volumetrics*` follow-up. `cookie_path` is
    /// `Some(path)` only for a Spot light with a resolvable
    /// `SceneObject::cookie_texture_path` — always `None` for every other
    /// light type.
    pub fn lights(
        &self,
    ) -> Vec<(
        crate::engine::lighting::Light,
        bool,
        Option<(f32, f32)>,
        Option<String>,
    )> {
        self.objects
            .iter()
            .filter(|o| o.image.is_none())
            .filter_map(|o| {
                let (radius, intensity, color) = o.light_params()?;
                let origin = o.parsed_origin();
                let origin = [origin[0] as f32, origin[1] as f32, origin[2] as f32];
                let casts_shadow = o
                    .castshadow
                    .as_ref()
                    .and_then(parse_value_bool)
                    .unwrap_or(false);
                let volumetrics = o.volumetrics_params();
                // Forward = -Z of the object's own rotation, shared by
                // Spot and Directional. Picked as the most common "facing"
                // convention (matching how `camera3d`'s eye→center
                // direction works) — unverified against real content, since
                // no downloaded Workshop `light` object was available
                // locally. See the `innercone`/`outercone` field docs.
                let angles = o
                    .angles
                    .as_ref()
                    .and_then(parse_value_vec3)
                    .map(|a| [a[0] as f32, a[1] as f32, a[2] as f32])
                    .unwrap_or([0.0; 3]);
                let direction = || crate::engine::camera3d::rotate_euler([0.0, 0.0, -1.0], angles);

                let is_spot = matches!(o.light_type(), LightType::Spot) || o.spot_cone_degrees().is_some();
                let light = if matches!(o.light_type(), LightType::Directional) {
                    crate::engine::lighting::Light::Directional {
                        direction: direction(),
                        color,
                        intensity,
                    }
                } else if is_spot {
                    // `pbr_spot`'s falloff is `cos_angle.powf(exponent)` (a
                    // Phong-style approximation — see lighting.rs's own
                    // module docs), not WE's recovered `smoothstep(inner,
                    // outer, cosAngle)` formula, so `innercone` isn't used
                    // yet: converting a smoothstep pair into a single Phong
                    // exponent faithfully needs both edges, and getting
                    // that wrong would be worse than an honest partial
                    // mapping. `exponent` is picked so the falloff reaches
                    // roughly zero brightness at `outercone` (defaulting to
                    // a narrow cone when cone angles aren't set at all —
                    // reached only via the `light_type() == Spot` string
                    // signal without cone fields, an edge case no real
                    // content was available to verify): reasonable, not
                    // verified against real content.
                    let outer = o.spot_cone_degrees().map(|(_, o)| o).unwrap_or(30.0);
                    let exponent =
                        (2.0 / (1.0 - outer.to_radians().cos()).max(1e-4)).clamp(1.0, 64.0);
                    crate::engine::lighting::Light::Spot {
                        origin,
                        direction: direction(),
                        exponent,
                        color,
                        intensity,
                        radius,
                        outer_cone_degrees: outer,
                    }
                } else if matches!(o.light_type(), LightType::Tube) {
                    let (origin_a, origin_b) = o.tube_endpoints();
                    crate::engine::lighting::Light::Tube {
                        origin_a,
                        origin_b,
                        color,
                        intensity,
                    }
                } else {
                    crate::engine::lighting::Light::Point {
                        origin,
                        color,
                        intensity,
                        radius,
                    }
                };
                let cookie_path = is_spot
                    .then(|| o.cookie_texture_path().map(str::to_string))
                    .flatten();
                Some((light, casts_shadow, volumetrics, cookie_path))
            })
            .collect()
    }
}

#[derive(Debug, Deserialize)]
pub struct Camera {
    pub center: Option<serde_json::Value>,
    pub eye: Option<serde_json::Value>,
    pub up: Option<serde_json::Value>,
    #[serde(rename = "parallaxamount")]
    pub parallax_amount: Option<f64>,
    #[serde(rename = "orthogonalprojection")]
    pub orthogonal_projection: Option<OrthogonalProjection>,
}

impl Camera {
    pub fn parsed_eye(&self) -> Option<[f64; 3]> {
        self.eye.as_ref().and_then(parse_value_vec3)
    }

    pub fn parsed_center(&self) -> Option<[f64; 3]> {
        self.center.as_ref().and_then(parse_value_vec3)
    }

    pub fn parsed_up(&self) -> Option<[f64; 3]> {
        self.up.as_ref().and_then(parse_value_vec3)
    }
}

#[derive(Debug, Deserialize)]
pub struct OrthogonalProjection {
    pub width: Option<u32>,
    pub height: Option<u32>,
    /// `{"auto": true}` — ortho sized automatically (gifscenes): still an
    /// orthogonal projection even with null width/height.
    pub auto: Option<bool>,
}

#[derive(Debug, Default, Deserialize)]
pub struct General {
    #[serde(rename = "supportsaudioprocessing")]
    pub supports_audio_processing: Option<bool>,
    pub speed: Option<f64>,
    #[serde(rename = "orthogonalprojection")]
    pub orthogonal_projection: Option<OrthogonalProjection>,
    #[serde(rename = "clearcolor")]
    pub clear_color: Option<serde_json::Value>,
    // Camera dynamics — all user-settable via project.json properties
    // (values arrive as {"user":..., "value":...} and are resolved by the
    // property pass before deserialization).
    #[serde(rename = "camerafade")]
    pub camera_fade: Option<serde_json::Value>,
    #[serde(rename = "cameraparallax")]
    pub camera_parallax: Option<serde_json::Value>,
    #[serde(rename = "cameraparallaxamount")]
    pub camera_parallax_amount: Option<serde_json::Value>,
    #[serde(rename = "cameraparallaxdelay")]
    pub camera_parallax_delay: Option<serde_json::Value>,
    #[serde(rename = "cameraparallaxmouseinfluence")]
    pub camera_parallax_mouse_influence: Option<serde_json::Value>,
    #[serde(rename = "camerashake")]
    pub camera_shake: Option<serde_json::Value>,
    #[serde(rename = "camerashakeamplitude")]
    pub camera_shake_amplitude: Option<serde_json::Value>,
    #[serde(rename = "camerashakeroughness")]
    pub camera_shake_roughness: Option<serde_json::Value>,
    #[serde(rename = "camerashakespeed")]
    pub camera_shake_speed: Option<serde_json::Value>,
    // Perspective projection (3D scenes; vertical FOV in degrees)
    pub fov: Option<serde_json::Value>,
    pub nearz: Option<serde_json::Value>,
    pub farz: Option<serde_json::Value>,
    // Scene-level bloom (user-settable)
    pub bloom: Option<serde_json::Value>,
    #[serde(rename = "bloomstrength")]
    pub bloom_strength: Option<serde_json::Value>,
    #[serde(rename = "bloomthreshold")]
    pub bloom_threshold: Option<serde_json::Value>,
    // HDR bloom (99/195 wallpapers). We run one SDR bloom chain, so when `hdr`
    // is on we feed it the HDR-tuned strength/threshold instead.
    pub hdr: Option<serde_json::Value>,
    #[serde(rename = "bloomhdrstrength")]
    pub bloom_hdr_strength: Option<serde_json::Value>,
    #[serde(rename = "bloomhdrthreshold")]
    pub bloom_hdr_threshold: Option<serde_json::Value>,
    // Camera zoom (105/195) — scales the whole scene about its center.
    pub zoom: Option<serde_json::Value>,
    // Whether the frame is cleared to clearcolor each frame (else transparent).
    pub clearenabled: Option<serde_json::Value>,
    // Scene ambient / sky lighting — applied as a global additive/multiplicative
    // tint (a minimal lighting approximation, no per-normal shading).
    pub ambientcolor: Option<serde_json::Value>,
    pub skylightcolor: Option<serde_json::Value>,
    // Scene fog (from the 2031-name property schema: g_FogDistance_*, g_FogHeight_*).
    // `fogdistance` = distance-fog density (0 = off); `fogheight` = height-fog density.
    // Colours and exponents come from the per-scene property surface.
    #[serde(rename = "fogdistance")]
    pub fog_distance: Option<serde_json::Value>,
    #[serde(rename = "fogheight")]
    pub fog_height: Option<serde_json::Value>,
    #[serde(rename = "fogdistancecolor")]
    pub fog_distance_color: Option<serde_json::Value>,
    #[serde(rename = "fogheightcolor")]
    pub fog_height_color: Option<serde_json::Value>,
    #[serde(rename = "fogheightexponent")]
    pub fog_height_exponent: Option<serde_json::Value>,
    #[serde(rename = "fogheightoffset")]
    pub fog_height_offset: Option<serde_json::Value>,
    // Bloom color tint.
    pub bloomtint: Option<serde_json::Value>,
    #[serde(rename = "bloomhdrfeather")]
    pub bloom_hdr_feather: Option<serde_json::Value>,
    #[serde(rename = "bloomhdrscatter")]
    pub bloom_hdr_scatter: Option<serde_json::Value>,
    #[serde(rename = "bloomhdriterations")]
    pub bloom_hdr_iterations: Option<serde_json::Value>,
    // Global particle forces (12/195).
    pub winddirection: Option<serde_json::Value>,
    pub windstrength: Option<serde_json::Value>,
    pub windenabled: Option<serde_json::Value>,
    pub gravitydirection: Option<serde_json::Value>,
    pub gravitystrength: Option<serde_json::Value>,
    // Per-scene FOV override (16/195).
    pub perspectiveoverridefov: Option<serde_json::Value>,
    // Transparency sort mode (2/195).
    pub transparentsorting: Option<serde_json::Value>,
}

impl General {
    /// Combined global particle acceleration in particle space (screen y-down):
    /// `gravitydirection*gravitystrength + winddirection*windstrength`. WE's
    /// directions are y-up, so y is negated. Returns `[0,0]` when unset.
    pub fn particle_force(&self) -> [f32; 2] {
        let term = |dir: &Option<serde_json::Value>, str: &Option<serde_json::Value>| {
            let d = dir.as_ref().and_then(parse_value_vec3).unwrap_or([0.0; 3]);
            let s = str.as_ref().and_then(parse_value_f32).unwrap_or(0.0);
            [d[0] as f32 * s, d[1] as f32 * s]
        };
        let wind_on = self
            .windenabled
            .as_ref()
            .and_then(|v| {
                v.as_bool()
                    .or_else(|| v.get("value").and_then(|i| i.as_bool()))
            })
            .unwrap_or(true);
        let g = term(&self.gravitydirection, &self.gravitystrength);
        let w = if wind_on {
            term(&self.winddirection, &self.windstrength)
        } else {
            [0.0, 0.0]
        };
        [g[0] + w[0], -(g[1] + w[1])]
    }

    /// Resolve the scene's fog parameters from the `g_Fog*` property surface.
    ///
    /// Returns `None` when both distance and height fog density are 0 / unset
    /// (fog off). The caller (render.rs / gpu_renderer.rs) should only run
    /// the fog pass when this returns `Some`.
    pub fn fog(&self) -> Option<crate::engine::lighting::Fog> {
        use crate::engine::lighting::Fog;
        let d = self.fog_distance.as_ref().and_then(parse_value_f32).unwrap_or(0.0);
        let h = self.fog_height.as_ref().and_then(parse_value_f32).unwrap_or(0.0);
        if d <= 0.0 && h <= 0.0 {
            return None;
        }
        Some(Fog {
            distance_density: d,
            distance_color: self
                .fog_distance_color
                .as_ref()
                .and_then(parse_value_vec3)
                .map(|c| [c[0] as f32, c[1] as f32, c[2] as f32])
                .unwrap_or([1.0; 3]),
            height_density: h,
            height_color: self
                .fog_height_color
                .as_ref()
                .and_then(parse_value_vec3)
                .map(|c| [c[0] as f32, c[1] as f32, c[2] as f32])
                .unwrap_or([1.0; 3]),
            height_exponent: self
                .fog_height_exponent
                .as_ref()
                .and_then(parse_value_f32)
                .unwrap_or(1.0),
            height_offset: self
                .fog_height_offset
                .as_ref()
                .and_then(parse_value_f32)
                .unwrap_or(0.0),
        })
    }

    /// Scene ambient color (`ambientcolor`), for the mesh3d lighting pass.
    /// `None` when unset — distinct from `Some([0,0,0])`, so the lighting pass
    /// can tell "no ambient authored" (skip the term) from "authored black"
    /// (matches real content's near-universal unauthored default: applying it
    /// unconditionally would darken every mesh3d scene, lit or not).
    pub fn ambient_color(&self) -> Option<[f32; 3]> {
        self.ambientcolor
            .as_ref()
            .and_then(parse_value_vec3)
            .map(|c| [c[0] as f32, c[1] as f32, c[2] as f32])
    }
}

/// A single object/layer in the scene.
///
/// Uses `serde_json::Value` for fields that have inconsistent types across
/// different Wallpaper Engine versions (e.g., `visible` can be bool or object).
#[derive(Debug, Deserialize)]
pub struct SceneObject {
    pub id: Option<i64>,
    pub name: Option<String>,
    pub visible: Option<serde_json::Value>,
    pub origin: Option<serde_json::Value>,
    pub angles: Option<serde_json::Value>,
    pub size: Option<serde_json::Value>,
    pub scale: Option<serde_json::Value>,
    pub alpha: Option<serde_json::Value>,
    pub color: Option<serde_json::Value>,
    #[serde(default)]
    pub brightness: Option<serde_json::Value>,
    #[serde(rename = "parallaxDepth")]
    pub parallax_depth: Option<serde_json::Value>,
    pub image: Option<String>,
    /// "top"/"bottom"/"left"/"right" (any combination, substring-matched like
    /// the reference) — shifts the object so its origin becomes an edge
    /// rather than its center. See `CImage.cpp` lines 242-256.
    #[serde(default)]
    pub alignment: Option<String>,
    pub particle: Option<serde_json::Value>,
    /// Per-instance overrides for a referenced particle preset (alpha, color,
    /// rate, size, speed) — e.g. `{"alpha":0.5,"color":"192 192 192","rate":0.16}`.
    #[serde(default)]
    pub instanceoverride: Option<serde_json::Value>,
    /// Presence of this field marks a text object (CText.cpp); the value is
    /// the initial/placeholder string (user-settable, possibly scripted —
    /// we only support the static case, matching the reference's own
    /// "Phase 1" text scope of no live script updates... actually the
    /// reference *does* support scripted text; we don't have a JS engine, so
    /// we render whatever the initial value is and leave it static).
    #[serde(default)]
    pub text: Option<serde_json::Value>,
    /// Font reference: a path into the wallpaper's own assets
    /// (`"fonts/VCR_OSD_MONO.ttf"`) or `"systemfont_*"` for a system font.
    #[serde(default)]
    pub font: Option<String>,
    #[serde(default)]
    pub pointsize: Option<serde_json::Value>,
    /// Vertical text alignment: "top"/"center"/"bottom". Horizontal alignment
    /// reuses `alignment`/`horizontalalign` — same field the reference falls
    /// back through for text objects (ObjectParser.cpp's `parseText`).
    #[serde(default)]
    pub verticalalign: Option<String>,
    #[serde(default)]
    pub horizontalalign: Option<String>,
    #[serde(default)]
    pub effects: Vec<Effect>,
    #[serde(rename = "colorBlendMode", default)]
    pub color_blend_mode: u32,
    #[serde(default)]
    pub copybackground: bool,
    // WE uses either integer indices or string names depending on wallpaper version.
    #[serde(default)]
    pub parent: Option<serde_json::Value>,
    #[serde(default)]
    pub dependencies: Vec<serde_json::Value>,
    // Text background box (rendered behind the glyphs when opaquebackground).
    #[serde(default)]
    pub opaquebackground: Option<serde_json::Value>,
    #[serde(default)]
    pub backgroundcolor: Option<serde_json::Value>,
    #[serde(default)]
    pub backgroundbrightness: Option<serde_json::Value>,
    #[serde(default)]
    pub padding: Option<serde_json::Value>,
    // ── Remaining scene.json object fields ───────────────────────────────
    // Declared so the value tree always parses (a strict/missing field is how
    // 31 scenes were rejected once before) and so callers can read them
    // without re-deriving the JSON. Runtime behaviour is wired only where the
    // semantics are actually determinable — see the notes on each.
    /// Gates whether this object's transform propagates to its children.
    /// Applied in `render::apply_parent_transforms`.
    #[serde(default)]
    pub disablepropagation: Option<serde_json::Value>,
    /// `"enabled"`/`"disabled"` — depth testing for 3D mesh objects.
    #[serde(default)]
    pub depthtest: Option<serde_json::Value>,
    /// Ellipsise rather than hard-truncate when `limitrows` clips text.
    #[serde(default)]
    pub limituseellipsis: Option<serde_json::Value>,
    /// Cursor hit-testing, not rendering: the C++ reference's only use is a
    /// commented-out `isSolid()` guard for "layer receives cursor events"
    /// (CImage.cpp). Parsed; no visual effect to apply.
    #[serde(default)]
    pub solid: Option<serde_json::Value>,
    /// Shadow casting. Parsed; there is no shadow subsystem to feed, and every
    /// occurrence in real content is `false` anyway.
    #[serde(default)]
    pub castshadow: Option<serde_json::Value>,
    /// Per-object time window. No reference implementation exists and real
    /// content pins these at 1.0/5.0; parsed for completeness.
    #[serde(default)]
    pub mintime: Option<serde_json::Value>,
    #[serde(default)]
    pub maxtime: Option<serde_json::Value>,
    /// Editor state: canvas-resize pinning, transform lock, editor mute,
    /// authoring labels and paths. No runtime meaning.
    #[serde(default)]
    pub locktransforms: Option<serde_json::Value>,
    #[serde(default)]
    pub muteineditor: Option<serde_json::Value>,
    #[serde(default)]
    pub username: Option<serde_json::Value>,
    #[serde(default)]
    pub config: Option<serde_json::Value>,
    #[serde(default)]
    pub path: Option<serde_json::Value>,
    /// LED-strip output (Wallpaper Engine's ambient-lighting feature), not
    /// screen rendering.
    #[serde(default)]
    pub ledsource: Option<serde_json::Value>,
    /// Particle/instancing extras. `instance` is script-driven in real
    /// content; `queuemode`/`shape`/`particlesrc` are constant at default.
    #[serde(default)]
    pub instance: Option<serde_json::Value>,
    #[serde(default)]
    pub queuemode: Option<serde_json::Value>,
    #[serde(default)]
    pub shape: Option<serde_json::Value>,
    #[serde(default)]
    pub particlesrc: Option<serde_json::Value>,
    /// Volumetric-lighting parameters — see `SceneObject::volumetrics_params`
    /// and `engine::volumetrics`.
    #[serde(default)]
    pub castvolumetrics: Option<serde_json::Value>,
    #[serde(default)]
    pub volumetricsexponent: Option<serde_json::Value>,
    #[serde(default)]
    pub density: Option<serde_json::Value>,
    #[serde(default)]
    pub exponent: Option<serde_json::Value>,
    #[serde(default)]
    pub cascadedistance0: Option<serde_json::Value>,
    #[serde(default)]
    pub cascadedistance1: Option<serde_json::Value>,
    #[serde(default)]
    pub cascadedistance2: Option<serde_json::Value>,

    /// Editor-only; deliberately not applied.
    ///
    /// `anchor` records which edge the editor pins an object to when the
    /// canvas resizes. It does not position anything at runtime: across the 21
    /// real objects that set it (all text), the value never correlates with
    /// `origin` — `anchor: "left"` appears at x=3052, `anchor: "center"` on
    /// origins scattered from 111 to 3100 — because `origin` is already the
    /// resolved absolute position. The C++ reference doesn't implement it
    /// either (its only `anchor` is Wayland surface placement). Applying it
    /// would move 21 clocks that currently sit correctly.
    ///
    /// Same conclusion for the object-level `perspective` flag, which we don't
    /// even declare: all 18 objects that set it sit at z=0 with zero angles in
    /// orthographic scenes (no camera to project through), and the reference
    /// uses `perspective` only for particles (CParticle.cpp, `flags & 4`),
    /// never for the text/image objects that carry it here.
    #[serde(default)]
    pub anchor: Option<serde_json::Value>,
    // Text wrapping / row limits. `maxwidth`/`maxrows` only apply when their
    // `limit*` gate is on — WE keeps the numbers around while the checkbox is
    // off, so applying them unconditionally wraps text that should run free.
    #[serde(default)]
    pub maxwidth: Option<serde_json::Value>,
    #[serde(default)]
    pub maxrows: Option<serde_json::Value>,
    #[serde(default)]
    pub limitwidth: Option<serde_json::Value>,
    #[serde(default)]
    pub limitrows: Option<serde_json::Value>,
    #[serde(default)]
    pub blockalign: Option<serde_json::Value>,
    // Sound objects (audio playback).
    #[serde(default)]
    pub sound: Option<serde_json::Value>,
    #[serde(default)]
    pub volume: Option<serde_json::Value>,
    #[serde(default)]
    pub playbackmode: Option<serde_json::Value>,
    #[serde(default)]
    pub startsilent: Option<serde_json::Value>,
    // Object-level texture clamp override.
    #[serde(default)]
    pub clampuvs: Option<serde_json::Value>,
    /// Light objects (radial glow approximation). Presence alone gates
    /// whether this object is a light at all (`light_params`) — but the
    /// *value* is also the real light-type discriminant, per the Ghidra
    /// report's directional-light-type-discriminant follow-up:
    /// `FUN_14025da80`'s property parser registers `"light"` itself (not a
    /// separate `"type"` field) with a string-enum setter
    /// (`FUN_14025eb40`/`FUN_14025e9e0`) that `memcmp`s the value against a
    /// 5-entry table — `"point"`, `"lpoint"`, `"lspot"`, `"ltube"`,
    /// `"ldirectional"` — falling back to the `"point"` slot when nothing
    /// matches (covers the common authored `"light": true` boolean form).
    /// See [`SceneObject::light_type`].
    #[serde(default)]
    pub light: Option<serde_json::Value>,
    #[serde(default)]
    pub radius: Option<serde_json::Value>,
    #[serde(default)]
    pub intensity: Option<serde_json::Value>,
    /// Spot-light cone angles (degrees). Confirmed field names from the
    /// Ghidra property dump (`~/Applications/ghidra-dump/REPORT.md`,
    /// Follow-up (f)). `Scene::lights()` now reads `light`'s own value as
    /// the primary type signal (see `light_type`); cone-angle presence is
    /// kept as a fallback for content that authors `"light": true` (a
    /// boolean never matches the string table, so the real engine itself
    /// falls back to point too — cone presence is wp-engine's own extra
    /// signal to still catch author intent in that case).
    #[serde(default)]
    pub innercone: Option<serde_json::Value>,
    #[serde(default)]
    pub outercone: Option<serde_json::Value>,
    /// A spot light's cookie-texture toggle and path — real field names
    /// from the Ghidra property dump (`usecookie`, `cookie`, adjacent to a
    /// literal built-in preset path `cookie/flashlight1` in `strings.txt`).
    /// See `SceneObject::cookie_texture_path`.
    #[serde(default)]
    pub usecookie: Option<serde_json::Value>,
    #[serde(default)]
    pub cookie: Option<serde_json::Value>,
    /// A tube light's second endpoint, relative to `origin` — see
    /// `SceneObject::tube_endpoints`. Confirmed a vec3 field from the Ghidra
    /// property dump (same type code and setter function as `color`, a
    /// known vec3), registered by the same Light property parser as
    /// `light`/`innercone`/etc; whether it's relative-to-origin or an
    /// absolute world position wasn't recoverable (no real content with a
    /// tube light was available locally to check against — see the report's
    /// `_rt_volumetrics*`/tube-lights follow-up) — relative was picked to
    /// match how every other per-object offset in this schema works.
    #[serde(default)]
    pub controlpoint: Option<serde_json::Value>,
    // Puppet animation layers (blended each frame).
    #[serde(default)]
    pub animationlayers: Option<serde_json::Value>,
    /// A static 3D mesh reference (`models/x/x.mdl`) — an alternative to
    /// `image` used by genuine 3D scenes for spheres/skyboxes/cylinders. NOT
    /// redundant with `image`: these objects carry no `image` at all, so
    /// without this they're skipped entirely. See `engine::mesh3d`.
    #[serde(default)]
    pub model: Option<serde_json::Value>,
}

impl SceneObject {
    /// The `model` field as a `.mdl` path, when this object is a static 3D
    /// mesh (`model` set, no `image`).
    pub fn mesh3d_path(&self) -> Option<&str> {
        if self.image.is_some() {
            return None;
        }
        let p = self.model.as_ref()?.as_str()?;
        p.to_lowercase().ends_with(".mdl").then_some(p)
    }

    /// Resolve `(radius, intensity, color)` for a `light` object — `None` when
    /// this isn't one. Shared by the 2D glow stand-in (`render.rs`) and the
    /// mesh3d per-pixel lighting pass (`Scene::lights`), so both treat a
    /// `light` object's fields identically.
    pub fn light_params(&self) -> Option<(f32, f32, [f32; 3])> {
        self.light.as_ref()?;
        let radius = self
            .radius
            .as_ref()
            .and_then(parse_value_f32)
            .unwrap_or(256.0)
            .max(1.0);
        let intensity = self
            .intensity
            .as_ref()
            .and_then(parse_value_f32)
            .unwrap_or(1.0)
            .clamp(0.0, 1.0);
        let color = self
            .color
            .as_ref()
            .and_then(parse_value_vec3)
            .map(|c| [c[0] as f32, c[1] as f32, c[2] as f32])
            .unwrap_or([1.0, 1.0, 1.0]);
        Some((radius, intensity, color))
    }

    /// `(innercone, outercone)` in degrees, when this light object sets
    /// both — the signal `Scene::lights()` uses to build a spot light
    /// instead of a point light. See the `innercone`/`outercone` field docs
    /// for the "no real content to verify a type field against" caveat.
    pub fn spot_cone_degrees(&self) -> Option<(f32, f32)> {
        let inner = self.innercone.as_ref().and_then(parse_value_f32)?;
        let outer = self.outercone.as_ref().and_then(parse_value_f32)?;
        Some((inner, outer))
    }

    /// A spot light's cookie-texture path, when `usecookie` is truthy and
    /// `cookie` names a non-empty path — `None` otherwise (the overwhelming
    /// common case: real content never sets `usecookie`, per the Ghidra
    /// report's property-schema survey). WE's own built-in cookie presets
    /// (e.g. `cookie/flashlight1`, seen as a literal string in the binary)
    /// live inside the original .exe, not in any wallpaper's own asset
    /// folder, so they're unreachable here — this only resolves to
    /// something real for a wallpaper that ships its own cookie image
    /// relative to its own directory (same resolution rules as any other
    /// texture reference — see `GpuSceneInstance::build` in gpu_renderer.rs).
    pub fn cookie_texture_path(&self) -> Option<&str> {
        let on = self
            .usecookie
            .as_ref()
            .and_then(parse_value_bool)
            .unwrap_or(false);
        if !on {
            return None;
        }
        self.cookie.as_ref()?.as_str().filter(|s| !s.is_empty())
    }

    /// The `light` field's own value as a type discriminant — see the
    /// field's doc comment for the decompiled evidence. `memcmp`-equivalent
    /// (case-sensitive, exact match against the binary's own table); any
    /// non-string value (`true`, a number) or unrecognized string falls
    /// back to `Point`, mirroring `FUN_14025eb40`'s own fallback-to-first-
    /// table-entry behavior exactly.
    pub fn light_type(&self) -> LightType {
        match self.light.as_ref().and_then(|v| v.as_str()) {
            Some("lspot") => LightType::Spot,
            Some("ltube") => LightType::Tube,
            Some("ldirectional") => LightType::Directional,
            _ => LightType::Point,
        }
    }

    /// `(density, exponent)` for a volumetric light shaft, when
    /// `castvolumetrics` is truthy — `None` otherwise. See
    /// `GpuSceneInstance::build_volumetrics`/`record_volumetrics` in
    /// gpu_renderer.rs for the consumer; defaults picked to give a
    /// visible-but-not-overwhelming shaft when a scene sets
    /// `castvolumetrics` without also tuning `density`/`volumetricsexponent`
    /// — unverified against real content, no downloaded Workshop light
    /// object with `castvolumetrics` set was available locally (see the
    /// Ghidra report's `_rt_volumetrics*` follow-up).
    pub fn volumetrics_params(&self) -> Option<(f32, f32)> {
        let on = self
            .castvolumetrics
            .as_ref()
            .and_then(parse_value_bool)
            .unwrap_or(false);
        if !on {
            return None;
        }
        let density = self
            .density
            .as_ref()
            .and_then(parse_value_f32)
            .unwrap_or(1.0)
            .clamp(0.0, 4.0);
        let exponent = self
            .volumetricsexponent
            .as_ref()
            .and_then(parse_value_f32)
            .unwrap_or(1.0)
            .max(0.01);
        Some((density, exponent))
    }

    /// World-space `(origin_a, origin_b)` for a tube light — `origin_a` is
    /// the object's own `origin`, `origin_b` is `origin + controlpoint`
    /// (see `controlpoint`'s field doc for the relative-vs-absolute
    /// caveat). Always returns a pair (falls back to `controlpoint`
    /// defaulting to zero, degenerating to a zero-length tube = a point
    /// light in all but name) rather than `None`, since every light object
    /// already has an `origin` regardless of type.
    pub fn tube_endpoints(&self) -> ([f32; 3], [f32; 3]) {
        let origin = self.parsed_origin();
        let origin_a = [origin[0] as f32, origin[1] as f32, origin[2] as f32];
        let offset = self
            .controlpoint
            .as_ref()
            .and_then(parse_value_vec3)
            .unwrap_or([0.0; 3]);
        let origin_b = [
            origin_a[0] + offset[0] as f32,
            origin_a[1] + offset[1] as f32,
            origin_a[2] + offset[2] as f32,
        ];
        (origin_a, origin_b)
    }
}

/// The light-type discriminant recovered from `SceneObject::light_type`.
/// `Tube` constructs `lighting::Light::Tube` via `SceneObject::tube_endpoints`
/// — see its doc comment for the relative-endpoint caveat (the one part of
/// this that's a guess rather than directly recovered; everything else
/// about tube-light *construction* follows straightforwardly once the type
/// discriminant and the endpoint field are both known). Not shadow-mapped
/// (matches Point/Spot's own staged rollout — shadows for spot lights only
/// shipped after point lights had already been working for a while) — no
/// tube-light case exists in `engine::shadow` yet.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LightType {
    Point,
    Spot,
    Tube,
    Directional,
}

impl SceneObject {
    pub fn parsed_origin(&self) -> [f64; 3] {
        self.origin
            .as_ref()
            .and_then(|v| parse_value_vec3(v))
            .unwrap_or([0.0; 3])
    }

    pub fn parsed_size(&self) -> [f64; 3] {
        self.size
            .as_ref()
            .and_then(|v| parse_value_vec3(v))
            .unwrap_or([0.0; 3])
    }

    pub fn is_visible(&self) -> bool {
        match &self.visible {
            None => true,
            Some(serde_json::Value::Bool(b)) => *b,
            Some(serde_json::Value::Object(m)) => {
                m.get("value").and_then(|v| v.as_bool()).unwrap_or(true)
            }
            _ => true,
        }
    }

    /// Inline SceneScript source driving `visible`, if the property was
    /// authored as `{"value": …, "script": "…"}`. An object with a visible
    /// script must not be dropped by the `is_visible()` load-time filter even
    /// when its authored value is `false` — the script may turn it on.
    pub fn visible_script(&self) -> Option<String> {
        match &self.visible {
            Some(serde_json::Value::Object(m)) => {
                m.get("script").and_then(|v| v.as_str()).map(str::to_string)
            }
            _ => None,
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct Effect {
    pub id: Option<i64>,
    pub file: Option<String>,
    pub name: Option<serde_json::Value>,
    pub visible: Option<serde_json::Value>,
    #[serde(default)]
    pub passes: Vec<Pass>,
}

impl Effect {
    pub fn name_string(&self) -> Option<String> {
        match &self.name {
            Some(serde_json::Value::String(s)) => Some(s.clone()),
            Some(value) if !value.is_null() => Some(value.to_string()),
            _ => None,
        }
    }

    pub fn is_visible(&self) -> bool {
        match &self.visible {
            None => true,
            Some(serde_json::Value::Bool(b)) => *b,
            Some(serde_json::Value::Object(m)) => {
                m.get("value").and_then(|v| v.as_bool()).unwrap_or(true)
            }
            _ => true,
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct Pass {
    pub material: Option<String>,
    #[serde(default)]
    pub combos: std::collections::HashMap<String, i32>,
    /// Positional: index N is the texture bound to shader slot N (`null`
    /// entries are real gaps, e.g. "leave this slot at its material
    /// default"), preserved as `None` rather than dropped — texture slots
    /// are matched to shader uniforms by array position, so collapsing
    /// `[null, "a", "b"]` down to `["a", "b"]` silently shifts every texture
    /// after a gap into the wrong slot (see [[wp_engine_project]] memory,
    /// nitro effect mask bug).
    #[serde(default, deserialize_with = "deserialize_textures")]
    pub textures: Vec<Option<TextureRef>>,
    #[serde(default)]
    pub constantshadervalues: std::collections::HashMap<String, serde_json::Value>,
}

#[derive(Debug, Deserialize)]
pub struct TextureRef {
    pub file: Option<String>,
    pub name: Option<String>,
}

fn deserialize_textures<'de, D>(
    deserializer: D,
) -> std::result::Result<Vec<Option<TextureRef>>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let items: Vec<Option<serde_json::Value>> = Vec::deserialize(deserializer)?;
    Ok(items
        .into_iter()
        .map(|v| {
            let v = v?;
            if v.is_null() {
                return None;
            }
            if let Some(s) = v.as_str() {
                return Some(TextureRef {
                    file: Some(s.to_string()),
                    name: Some(s.to_string()),
                });
            }
            serde_json::from_value(v).ok()
        })
        .collect())
}

impl Scene {
    pub fn from_json(json: &str) -> Result<Self> {
        serde_json::from_str(json).with_context(|| {
            let preview = if json.len() > 200 { &json[..200] } else { json };
            format!("failed to parse scene.json: {preview}...")
        })
    }

    /// Parse from an already-loaded JSON tree (used after user-property
    /// resolution has rewritten `{"user":...}` values in place).
    pub fn from_value(value: serde_json::Value) -> Result<Self> {
        serde_json::from_value(value).context("failed to parse scene.json value tree")
    }

    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        let text = std::str::from_utf8(data).context("scene.json is not valid UTF-8")?;
        Self::from_json(text)
    }

    pub fn visible_objects(&self) -> impl Iterator<Item = &SceneObject> {
        self.objects.iter().filter(|o| o.is_visible())
    }

    /// Per-object *effective* visibility, indexed by position in `self.objects`.
    ///
    /// Wallpaper Engine hides an entire subtree when any ancestor is hidden, so
    /// an object is drawn only if it AND every ancestor up its `parent` chain
    /// are visible. Our per-object `is_visible()` alone ignores the chain, which
    /// let hidden language-group children (clocks/dates) all draw at once — see
    /// the LonelyCat multi-language overlap. This walks the chain (same `parent`
    /// resolution the transform pass uses: numeric `id`, occasionally a name).
    pub fn visibility_mask(&self) -> Vec<bool> {
        use std::collections::HashMap;
        let id_to_idx: HashMap<i64, usize> = self
            .objects
            .iter()
            .enumerate()
            .filter_map(|(i, o)| o.id.map(|id| (id, i)))
            .collect();
        let name_to_idx: HashMap<&str, usize> = self
            .objects
            .iter()
            .enumerate()
            .filter_map(|(i, o)| o.name.as_deref().map(|n| (n, i)))
            .collect();
        let parent_of = |o: &SceneObject| -> Option<usize> {
            let p = o.parent.as_ref()?;
            if let Some(id) = p.as_i64() {
                return id_to_idx.get(&id).copied();
            }
            if let Some(id) = p.as_u64() {
                return id_to_idx.get(&(id as i64)).copied();
            }
            if let Some(name) = p.as_str() {
                return name_to_idx.get(name).copied();
            }
            None
        };
        (0..self.objects.len())
            .map(|i| {
                let mut cur = i;
                // Depth guard: a malformed scene can loop or nest absurdly; bail
                // to "visible" rather than spin (tolerant parsing).
                for _ in 0..64 {
                    // A visibility script decides per-frame whether the layer
                    // draws, so a statically-hidden but scripted object must
                    // survive load (upstream's per-object rule, folded into the
                    // hierarchical walk).
                    if !self.objects[cur].is_visible()
                        && self.objects[cur].visible_script().is_none()
                    {
                        return false;
                    }
                    match parent_of(&self.objects[cur]) {
                        Some(p) if p != cur => cur = p,
                        _ => return true,
                    }
                }
                true
            })
            .collect()
    }

    pub fn image_paths(&self) -> Vec<&str> {
        self.objects
            .iter()
            .filter_map(|o| o.image.as_deref())
            .collect()
    }

    pub fn topological_render_order_indices(&self) -> Vec<usize> {
        let mut name_to_index = std::collections::HashMap::new();
        for (i, obj) in self.objects.iter().enumerate() {
            if let Some(name) = &obj.name {
                name_to_index.insert(name.clone(), i);
            }
        }
        let mut visited = vec![false; self.objects.len()];
        let mut result = Vec::with_capacity(self.objects.len());
        for i in 0..self.objects.len() {
            if !visited[i] {
                Self::topo_dfs_indices(i, &self.objects, &mut visited, &mut result, &name_to_index);
            }
        }
        result
    }

    pub fn topological_render_order(&self) -> Vec<&SceneObject> {
        self.topological_render_order_indices()
            .into_iter()
            .map(|index| &self.objects[index])
            .collect()
    }

    fn topo_dfs_indices(
        index: usize,
        objects: &[SceneObject],
        visited: &mut Vec<bool>,
        result: &mut Vec<usize>,
        name_to_index: &std::collections::HashMap<String, usize>,
    ) {
        if visited[index] {
            return;
        }
        visited[index] = true;
        let obj = &objects[index];
        for dep in &obj.dependencies {
            let di = if let Some(n) = dep.as_u64() {
                // integer index
                Some(n as usize).filter(|&i| i < objects.len())
            } else if let Some(s) = dep.as_str() {
                // string name
                name_to_index.get(s).copied()
            } else {
                None
            };
            if let Some(di) = di {
                Self::topo_dfs_indices(di, objects, visited, result, name_to_index);
            }
        }
        if let Some(parent) = &obj.parent {
            let pi = if let Some(n) = parent.as_u64() {
                Some(n as usize).filter(|&i| i < objects.len())
            } else if let Some(s) = parent.as_str() {
                name_to_index.get(s).copied()
            } else {
                None
            };
            if let Some(pi) = pi {
                Self::topo_dfs_indices(pi, objects, visited, result, name_to_index);
            }
        }
        result.push(index);
    }

    pub fn texture_paths(&self) -> Vec<&str> {
        let mut paths: Vec<&str> = self.image_paths();
        for obj in &self.objects {
            for effect in &obj.effects {
                for pass in &effect.passes {
                    for tex in pass.textures.iter().flatten() {
                        if let Some(f) = tex.file.as_deref() {
                            paths.push(f);
                        }
                    }
                }
            }
        }
        paths
    }
}

fn parse_vec3(s: Option<&str>) -> [f64; 3] {
    let Some(s) = s else { return [0.0; 3] };
    let parts: Vec<f64> = s
        .split_whitespace()
        .filter_map(|p| p.parse().ok())
        .collect();
    [
        parts.first().copied().unwrap_or(0.0),
        parts.get(1).copied().unwrap_or(0.0),
        parts.get(2).copied().unwrap_or(0.0),
    ]
}

/// Unwrap a possibly `{"value": ...}`-wrapped JSON value to f32.
pub fn parse_value_f32(v: &serde_json::Value) -> Option<f32> {
    match v {
        serde_json::Value::Number(n) => n.as_f64().map(|f| f as f32),
        serde_json::Value::String(s) => s.trim().parse().ok(),
        serde_json::Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
        serde_json::Value::Object(m) => m.get("value").and_then(parse_value_f32),
        _ => None,
    }
}

/// Unwrap a possibly `{"value": ...}`-wrapped JSON value to bool.
pub fn parse_value_bool(v: &serde_json::Value) -> Option<bool> {
    match v {
        serde_json::Value::Bool(b) => Some(*b),
        serde_json::Value::Number(n) => Some(n.as_f64().unwrap_or(0.0) != 0.0),
        serde_json::Value::String(s) => match s.trim() {
            "true" | "1" | "yes" | "on" => Some(true),
            "false" | "0" | "no" | "off" => Some(false),
            _ => None,
        },
        serde_json::Value::Object(m) => m.get("value").and_then(parse_value_bool),
        _ => None,
    }
}

pub fn parse_value_vec3(v: &serde_json::Value) -> Option<[f64; 3]> {
    match v {
        serde_json::Value::String(s) => {
            let r = parse_vec3(Some(s));
            Some(r)
        }
        serde_json::Value::Array(arr) if arr.len() >= 3 => Some([
            arr[0].as_f64().unwrap_or(0.0),
            arr[1].as_f64().unwrap_or(0.0),
            arr[2].as_f64().unwrap_or(0.0),
        ]),
        serde_json::Value::Object(m) => {
            // SceneScript animated property: {"value": "x y z", "script": "..."}
            // Use the static default; runtime scripting is not yet supported.
            m.get("value").and_then(|inner| parse_value_vec3(inner))
        }
        _ => None,
    }
}

/// Unwrap a possibly `{"value": ...}`-wrapped JSON value to `[f64; 4]` (rgb + w).
/// Accepts a 4-component array/string, or a 3-component value (w defaults to 1.0).
/// Used for `g_L*_Color` / `g_L*_Origin` uniforms where w carries intensity.
pub fn parse_value_vec4(v: &serde_json::Value) -> Option<[f64; 4]> {
    match v {
        serde_json::Value::Array(arr) if arr.len() >= 3 => {
            let g = |i: usize| arr.get(i).and_then(|x| x.as_f64()).unwrap_or(if i == 3 { 1.0 } else { 0.0 });
            Some([g(0), g(1), g(2), g(3)])
        }
        serde_json::Value::Object(m) => m.get("value").and_then(|inner| parse_value_vec4(inner)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A `light` object with both innercone/outercone set must resolve to a
    // Spot light, not the default Point.
    #[test]
    fn lights_with_cone_angles_resolve_to_spot() {
        let scene = Scene::from_json(
            r#"{"objects": [
                {"id": 1, "light": true, "origin": "1 2 3", "radius": 100,
                 "intensity": 0.8, "color": "1 0.5 0.2",
                 "innercone": 20, "outercone": 45}
            ]}"#,
        )
        .unwrap();
        let lights = scene.lights();
        assert_eq!(lights.len(), 1);
        match &lights[0].0 {
            crate::engine::lighting::Light::Spot {
                origin,
                intensity,
                exponent,
                ..
            } => {
                assert_eq!(*origin, [1.0, 2.0, 3.0]);
                assert!((*intensity - 0.8).abs() < 1e-5);
                // Narrower-than-default cone should sharpen the falloff
                // above the widest (2.0) end of the clamp range.
                assert!(*exponent > 2.0, "exponent={exponent}");
            }
            other => panic!("expected Spot, got {other:?}"),
        }
    }

    // A `light` object with no cone angles must keep resolving to Point —
    // the pre-existing, still-most-common case must not regress.
    #[test]
    fn lights_without_cone_angles_stay_point() {
        let scene = Scene::from_json(
            r#"{"objects": [
                {"id": 1, "light": true, "origin": "0 0 0", "radius": 50, "intensity": 1.0}
            ]}"#,
        )
        .unwrap();
        let lights = scene.lights();
        assert_eq!(lights.len(), 1);
        assert!(matches!(lights[0].0, crate::engine::lighting::Light::Point { .. }));
    }

    // Only one of the two cone fields set is treated as "no cone data" —
    // half a spot-light definition shouldn't silently activate spot mode.
    #[test]
    fn spot_cone_degrees_requires_both_fields() {
        let scene = Scene::from_json(
            r#"{"objects": [
                {"id": 1, "light": true, "origin": "0 0 0", "outercone": 45}
            ]}"#,
        )
        .unwrap();
        assert!(matches!(
            scene.lights()[0].0,
            crate::engine::lighting::Light::Point { .. }
        ));
    }

    // `light: "ldirectional"` — the real type-discriminant value recovered
    // from FUN_14025eb40's string table — must resolve to Directional, with
    // direction derived from the object's own angles (the same convention
    // Spot already uses).
    #[test]
    fn light_string_ldirectional_resolves_to_directional() {
        let scene = Scene::from_json(
            r#"{"objects": [
                {"id": 1, "light": "ldirectional", "angles": "0 1.5707963 0",
                 "intensity": 0.6, "color": "1 1 1"}
            ]}"#,
        )
        .unwrap();
        let lights = scene.lights();
        assert_eq!(lights.len(), 1);
        match &lights[0].0 {
            crate::engine::lighting::Light::Directional { direction, intensity, .. } => {
                assert!((*intensity - 0.6).abs() < 1e-5);
                // angles.y = 90°: forward (-Z rotated by yaw) points toward -X.
                assert!(direction[0] < -0.9, "direction={direction:?}");
            }
            other => panic!("expected Directional, got {other:?}"),
        }
    }

    // `light: "lspot"` alone (no innercone/outercone) must still resolve to
    // Spot via the string discriminant — the cone-presence heuristic is a
    // fallback, not the only path.
    #[test]
    fn light_string_lspot_resolves_to_spot_without_cone_fields() {
        let scene = Scene::from_json(
            r#"{"objects": [{"id": 1, "light": "lspot", "origin": "0 0 0"}]}"#,
        )
        .unwrap();
        assert!(matches!(
            scene.lights()[0].0,
            crate::engine::lighting::Light::Spot { .. }
        ));
    }

    // `light: "lpoint"` (the table's OTHER accepted point spelling) and the
    // common authored `light: true` boolean must both land on Point, exactly
    // matching the real engine's own fallback-to-first-table-entry behavior.
    #[test]
    fn light_string_lpoint_and_bare_true_both_stay_point() {
        let scene = Scene::from_json(
            r#"{"objects": [
                {"id": 1, "light": "lpoint", "origin": "0 0 0"},
                {"id": 2, "light": true, "origin": "0 0 0"}
            ]}"#,
        )
        .unwrap();
        let lights = scene.lights();
        assert_eq!(lights.len(), 2);
        assert!(matches!(lights[0].0, crate::engine::lighting::Light::Point { .. }));
        assert!(matches!(lights[1].0, crate::engine::lighting::Light::Point { .. }));
    }

    // `castvolumetrics: true` must surface as `Some((density, exponent))`
    // in `Scene::lights()`'s third tuple slot, with the authored values
    // read through, not the defaults.
    #[test]
    fn volumetrics_params_read_when_castvolumetrics_set() {
        let scene = Scene::from_json(
            r#"{"objects": [
                {"id": 1, "light": true, "origin": "0 0 0",
                 "castvolumetrics": true, "density": 2.5, "volumetricsexponent": 0.5}
            ]}"#,
        )
        .unwrap();
        let lights = scene.lights();
        assert_eq!(lights.len(), 1);
        let (density, exponent) = lights[0].2.expect("must report volumetrics params");
        assert!((density - 2.5).abs() < 1e-5);
        assert!((exponent - 0.5).abs() < 1e-5);
    }

    // Without `castvolumetrics` (the overwhelmingly common case), the third
    // tuple slot must be `None` — not a light silently defaulting into a
    // volumetric shaft nobody asked for.
    #[test]
    fn volumetrics_params_none_by_default() {
        let scene = Scene::from_json(
            r#"{"objects": [{"id": 1, "light": true, "origin": "0 0 0"}]}"#,
        )
        .unwrap();
        assert_eq!(scene.lights()[0].2, None);
    }

    // `usecookie: true` + a non-empty `cookie` path on a spot light must
    // surface in `Scene::lights()`'s 4th tuple slot; every other case
    // (no `usecookie`, or a non-spot light) must stay `None`.
    #[test]
    fn cookie_texture_path_only_set_for_spot_with_usecookie() {
        let scene = Scene::from_json(
            r#"{"objects": [
                {"id": 1, "light": "lspot", "origin": "0 0 0",
                 "usecookie": true, "cookie": "cookie/flashlight1"},
                {"id": 2, "light": true, "origin": "0 0 0",
                 "usecookie": true, "cookie": "cookie/flashlight1"},
                {"id": 3, "light": "lspot", "origin": "0 0 0"}
            ]}"#,
        )
        .unwrap();
        let lights = scene.lights();
        assert_eq!(lights.len(), 3);
        assert_eq!(lights[0].3.as_deref(), Some("cookie/flashlight1"));
        assert_eq!(lights[1].3, None, "a Point light must never carry a cookie path");
        assert_eq!(lights[2].3, None, "usecookie absent must leave the cookie path None");
    }

    #[test]
    fn cookie_texture_path_requires_usecookie_true() {
        let scene = Scene::from_json(
            r#"{"objects": [{"id": 1, "light": "lspot", "origin": "0 0 0", "cookie": "x"}]}"#,
        )
        .unwrap();
        assert_eq!(scene.objects[0].cookie_texture_path(), None);
    }

    // `light: "ltube"` must resolve to Tube, with origin_b = origin +
    // controlpoint (the relative-offset interpretation — see
    // `SceneObject::tube_endpoints`'s doc comment for the caveat).
    #[test]
    fn light_string_ltube_resolves_to_tube_with_relative_controlpoint() {
        let scene = Scene::from_json(
            r#"{"objects": [
                {"id": 1, "light": "ltube", "origin": "1 2 3",
                 "controlpoint": "0 0 4", "intensity": 0.7, "color": "1 1 1"}
            ]}"#,
        )
        .unwrap();
        let lights = scene.lights();
        assert_eq!(lights.len(), 1);
        match &lights[0].0 {
            crate::engine::lighting::Light::Tube { origin_a, origin_b, intensity, .. } => {
                assert_eq!(*origin_a, [1.0, 2.0, 3.0]);
                assert_eq!(*origin_b, [1.0, 2.0, 7.0]);
                assert!((*intensity - 0.7).abs() < 1e-5);
            }
            other => panic!("expected Tube, got {other:?}"),
        }
    }

    // No `controlpoint` at all must degenerate to a zero-length tube
    // (origin_a == origin_b) rather than panicking or defaulting oddly.
    #[test]
    fn tube_endpoints_default_to_a_zero_length_segment() {
        let scene = Scene::from_json(
            r#"{"objects": [{"id": 1, "light": "ltube", "origin": "5 0 0"}]}"#,
        )
        .unwrap();
        match &scene.lights()[0].0 {
            crate::engine::lighting::Light::Tube { origin_a, origin_b, .. } => {
                assert_eq!(origin_a, origin_b);
            }
            other => panic!("expected Tube, got {other:?}"),
        }
    }

    // A visible text layer whose parent group is hidden must NOT draw — the WE
    // subtree-hide rule (regression for the LonelyCat multi-language overlap).
    #[test]
    fn visibility_mask_inherits_hidden_ancestor() {
        let scene = Scene::from_json(
            r#"{"objects": [
                {"id": 1, "name": "HiddenGroup", "visible": false},
                {"id": 2, "name": "Clock", "parent": 1, "visible": true, "text": "12:00"},
                {"id": 3, "name": "VisibleGroup", "visible": true},
                {"id": 4, "name": "Date", "parent": 3, "visible": true, "text": "Mon"}
            ]}"#,
        )
        .unwrap();
        let mask = scene.visibility_mask();
        assert_eq!(mask, vec![false, false, true, true]);
    }

    // A parent cycle must not spin; every node in the cycle just stays visible.
    #[test]
    fn visibility_mask_tolerates_parent_cycle() {
        let scene = Scene::from_json(
            r#"{"objects": [
                {"id": 1, "parent": 2, "visible": true},
                {"id": 2, "parent": 1, "visible": true}
            ]}"#,
        )
        .unwrap();
        assert_eq!(scene.visibility_mask(), vec![true, true]);
    }
}
