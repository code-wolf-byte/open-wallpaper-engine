//! A particle system's JSON configuration (`particles/*.json`, with the
//! instance overrides a scene object applies) and the helpers that read its
//! loosely typed values.

use super::*;

// Numeric/count fields use `Option<T>` rather than bare `T` with
// `#[serde(default)]`: real presets frequently write an explicit JSON `null`
// for unused fields (e.g. `"flags": null`), and `#[serde(default)]` only
// fills in *missing* keys — it still rejects an explicit `null` against a
// non-Option numeric type.
#[derive(Debug, Clone, Deserialize)]
pub struct ParticleConfig {
    #[serde(default)]
    pub emitter: Vec<Emitter>,
    #[serde(default)]
    pub maxcount: Option<u32>,
    #[serde(default)]
    pub initializer: Vec<Initializer>,
    #[serde(default)]
    pub operator: Vec<Operator>,
    #[serde(default)]
    pub renderer: Vec<serde_json::Value>,
    #[serde(default)]
    pub flags: Option<u32>,
    /// Real JSON key is singular `"controlpoint"` (an array), referenced by
    /// index (not `id`-keyed lookup at use-site) from operators/emitters like
    /// `controlpointattract` (CParticle.cpp/ObjectParser.cpp).
    #[serde(default, deserialize_with = "null_entries_as_default")]
    pub controlpoint: Vec<ControlPointConfig>,
    /// Delay in seconds before this system begins emitting. Present on nearly
    /// every real preset (610/610) so a burst ramps in instead of all
    /// particles appearing at frame 0.
    #[serde(default)]
    pub starttime: Option<f64>,
    /// Path to the material JSON providing this system's sprite texture
    /// (e.g. `"materials/presets/water_faucet.json"`). Resolved by callers
    /// (this module stays asset-agnostic) and passed into `render_onto`.
    #[serde(default)]
    pub material: Option<String>,
    /// How a particle's sprite-sheet frame advances over time: `"once"`
    /// (play through over the particle's lifetime), `"randomframe"` (pick
    /// one fixed random frame at spawn), or anything else/absent (loop
    /// continuously) — CParticle.cpp's animation-frame update.
    #[serde(default)]
    pub animationmode: Option<String>,
    /// Playback speed multiplier for sprite-sheet animation; `<= 0` (or
    /// absent) means 1.0, matching the reference's `sequenceMultiplier > 0
    /// ? sequenceMultiplier : 1.0`.
    #[serde(default)]
    pub sequencemultiplier: Option<f64>,
    /// Nested child particle presets (`"children"` array; often JSON `null`).
    /// Loaded/resolved by the caller (this module stays asset-agnostic) and
    /// attached via [`ParticleSystem::add_child`].
    #[serde(default)]
    pub children: Option<Vec<ChildRef>>,
}

/// One `children` entry: a nested particle preset attached to this system.
/// `type` values seen in real content: absent/`"static"` (a sub-system
/// running continuously at the parent object's origin), `"eventfollow"`
/// (an instance per parent particle, tracking it), `"eventspawn"` /
/// `"eventdeath"` (a burst where a parent particle appears/dies).
#[derive(Debug, Clone, Deserialize)]
pub struct ChildRef {
    pub name: String,
    #[serde(rename = "type", default)]
    pub child_type: Option<String>,
    #[serde(default)]
    pub probability: Option<f64>,
    #[serde(default)]
    pub maxcount: Option<u64>,
}

/// One resolved particle sprite: every sprite-sheet frame (already sliced
/// out of the `.tex` atlas by `TexFile::to_rgba_frames`) plus the total
/// loop duration in seconds (sum of each frame's `frametime`, 0 for a
/// single-frame/non-animated sprite), plus the material's
/// `ui_editor_properties_overbright` brightness multiplier
/// (genericparticle.frag's `color.rgb *= g_Overbright`, default 1.0).
#[derive(Clone)]
pub struct ParticleSprite {
    pub frames: Vec<RgbaImage>,
    pub duration: f32,
    pub overbright: f32,
}

impl ParticleSprite {
    pub fn single(image: RgbaImage) -> Self {
        Self {
            frames: vec![image],
            duration: 0.0,
            overbright: 1.0,
        }
    }
}

/// Parsed `animationmode` (defaults to `Loop` for anything unrecognized or
/// absent, matching the reference's `else` fallback).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub(super) enum AnimationMode {
    #[default]
    Loop,
    Once,
    RandomFrame,
}

impl AnimationMode {
    pub(super) fn from_config(mode: Option<&str>) -> Self {
        match mode {
            Some("once") => Self::Once,
            Some("randomframe") => Self::RandomFrame,
            _ => Self::Loop,
        }
    }
}

/// Real presets pad their `controlpoint` array with literal `null`s (unused
/// slots). Serde rejects the whole preset over one, so nulls become defaults —
/// the array is indexed positionally, so the padding has to stay in place.
pub(super) fn null_entries_as_default<'de, D, T>(d: D) -> Result<Vec<T>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Deserialize<'de> + Default,
{
    Ok(Vec::<Option<T>>::deserialize(d)?
        .into_iter()
        .map(Option::unwrap_or_default)
        .collect())
}

#[derive(Default, Debug, Clone, Deserialize)]
pub struct ControlPointConfig {
    #[serde(default)]
    pub id: Option<i64>,
    #[serde(default)]
    pub offset: Option<serde_json::Value>,
    #[serde(default)]
    pub flags: Option<u32>,
    /// Orientation of the control point's local frame (z used in 2D): rotates
    /// the `offset` about the object origin so oriented control points land in
    /// the right place.
    #[serde(default)]
    pub angles: Option<serde_json::Value>,
    /// `locktopointer`: the control point tracks the mouse cursor. Parsed;
    /// we have no live pointer input, so it resolves to its static offset.
    #[serde(default)]
    pub locktopointer: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Emitter {
    #[serde(default)]
    pub name: String,
    /// Emission rate per second. `None` (absent) takes the reference's
    /// default of 10; an explicit `0` stays 0 (instantaneous-only emitters).
    #[serde(default)]
    pub rate: Option<f64>,
    #[serde(default)]
    pub origin: Option<String>,
    #[serde(default)]
    pub directions: Option<String>,
    /// A scalar radius for sphere emitters, but a `"x y z"` box-extent vector
    /// for box emitters (e.g. `"distancemax": "1024 512 0"`).
    #[serde(default)]
    pub distancemin: Option<serde_json::Value>,
    #[serde(default)]
    pub distancemax: Option<serde_json::Value>,
    /// Optional emitter-driven radial speed range (rare; most presets set
    /// velocity via initializers).
    #[serde(default)]
    pub speedmin: Option<serde_json::Value>,
    #[serde(default)]
    pub speedmax: Option<serde_json::Value>,
    #[serde(default)]
    pub sign: Option<String>,
    #[serde(default)]
    pub id: Option<u32>,
    /// One-shot burst count: this many particles are emitted in a single frame
    /// when the emitter first activates (at `starttime`), then normal `rate`
    /// emission continues. `0`/absent = purely rate-driven.
    #[serde(default)]
    pub instantaneous: Option<f64>,
    /// Emitter lifetime in seconds: after `starttime + duration`, this emitter
    /// stops emitting. Absent = emits forever.
    #[serde(default)]
    pub duration: Option<f64>,
    /// Presence of `audioprocessingmode`/`bounds` makes emission react to
    /// captured audio; the rest shape the response (see `AudioResponse`).
    #[serde(default)]
    pub audioprocessingmode: Option<serde_json::Value>,
    #[serde(default)]
    pub audioprocessingbounds: Option<serde_json::Value>,
    #[serde(default)]
    pub audioprocessingexponent: Option<serde_json::Value>,
    #[serde(default)]
    pub audioprocessingfrequencystart: Option<serde_json::Value>,
    #[serde(default)]
    pub audioprocessingfrequencyend: Option<serde_json::Value>,
}

/// An emitter's audio response, with the defaults and clamping of WE's own
/// parser (wallpaper64.exe `FUN_1401c1e20`): `exponent` 2.0, `bounds`
/// "0.8 1.0", bands `0..=1` of the 16-band spectrum, both clamped to 15 and
/// swapped if reversed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct AudioResponse {
    pub(super) exponent: f32,
    pub(super) bounds: [f32; 2],
    pub(super) bands: [usize; 2],
}

impl AudioResponse {
    pub(super) fn from_emitter(e: &Emitter) -> Option<Self> {
        if e.audioprocessingmode.is_none() && e.audioprocessingbounds.is_none() {
            return None;
        }
        let band = |v: &Option<serde_json::Value>, d: usize| {
            v.as_ref()
                .and_then(|v| v.as_u64())
                .map_or(d, |b| b as usize)
                .min(15)
        };
        let (a, b) = (
            band(&e.audioprocessingfrequencystart, 0),
            band(&e.audioprocessingfrequencyend, 1),
        );
        let bounds = match &e.audioprocessingbounds {
            Some(serde_json::Value::String(s)) => {
                let p: Vec<f32> = s
                    .split_whitespace()
                    .filter_map(|x| x.parse().ok())
                    .collect();
                [
                    p.first().copied().unwrap_or(0.0),
                    p.get(1).copied().unwrap_or(0.0),
                ]
            }
            Some(v) => value_as_f32(v).map_or([0.8, 1.0], |x| [x, x]),
            None => [0.8, 1.0],
        };
        Some(Self {
            exponent: e
                .audioprocessingexponent
                .as_ref()
                .and_then(value_as_f32)
                .unwrap_or(2.0),
            bounds,
            bands: [a.min(b), a.max(b)],
        })
    }

    /// Rate multiplier for one 16-band spectrum (L/R averaged). Per WE's
    /// emitter docs the emitter is "only active when audio is playing" and
    /// `bounds` are the volume levels where the response starts and stops,
    /// so silence -> 0 and the rate ramps from `bounds[0]` to `bounds[1]`.
    // ponytail: WE's runtime consumer isn't decompiled; the exponent-then-
    // bounds order follows the docs' wording, and `mode` (left/right/center)
    // is treated only as an on-switch. Revisit if the consumer is recovered.
    pub(super) fn gain(&self, bands16: &[f32; 16]) -> f32 {
        let s = &bands16[self.bands[0]..=self.bands[1]];
        let level = (s.iter().sum::<f32>() / s.len() as f32)
            .clamp(0.0, 1.0)
            .powf(self.exponent);
        let [lo, hi] = self.bounds;
        if hi <= lo {
            return if level >= lo { 1.0 } else { 0.0 };
        }
        ((level - lo) / (hi - lo)).clamp(0.0, 1.0)
    }
}

/// `min`/`max` are untyped because real presets mix plain numbers
/// (`lifetimerandom`, `sizerandom`, `alpharandom`) with `"r g b"`/`"x y z"`
/// vector strings (`velocityrandom`, `colorrandom`, `angularvelocityrandom`) —
/// declaring these as `f64` makes serde reject any config using a vector
/// initializer.
#[derive(Debug, Clone, Deserialize)]
pub struct Initializer {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub max: Option<serde_json::Value>,
    #[serde(default)]
    pub min: Option<serde_json::Value>,
    /// Biases the uniform `random(min,max)` toward `min` (exponent > 1) or
    /// `max` (< 1): the sample `t` is raised to this power before the lerp.
    /// Absent = 1.0 (uniform).
    #[serde(default)]
    pub exponent: Option<f64>,
    /// `turbulentvelocityrandom` (CParticle.cpp
    /// createTurbulentVelocityRandomInitializer): a curl-noise-directed
    /// spawn velocity. `forward`/`right` are `"x y z"` vector strings;
    /// `offset` here is the scalar tilt angle around `right`, unrelated to
    /// an emitter's positional offset. `speedmin`/`speedmax` are untyped
    /// because turbulentvelocityrandom uses plain numbers but
    /// `mapsequencearoundcontrolpoint` uses `"x y z"` vector strings.
    #[serde(default)]
    pub speedmin: Option<serde_json::Value>,
    #[serde(default)]
    pub speedmax: Option<serde_json::Value>,
    /// `mapsequencearoundcontrolpoint`: spawn the Nth particle at the
    /// control point, launched at angle `(N % count) / count * 2pi`.
    #[serde(default)]
    pub controlpoint: Option<i64>,
    #[serde(default)]
    pub count: Option<f64>,
    /// `mapsequencebetweencontrolpoints`: `"mirror"` ping-pongs the
    /// sequence along the segment instead of wrapping.
    #[serde(default)]
    pub limitbehavior: Option<String>,
    #[serde(default)]
    pub offset: Option<f64>,
    #[serde(default)]
    pub scale: Option<f64>,
    #[serde(default)]
    pub forward: Option<serde_json::Value>,
    #[serde(default)]
    pub timescale: Option<f64>,
    #[serde(default)]
    pub phasemin: Option<f64>,
    #[serde(default)]
    pub phasemax: Option<f64>,
    #[serde(default)]
    pub right: Option<serde_json::Value>,
}

#[derive(Debug, Default, Clone, Deserialize)]
pub struct Operator {
    #[serde(default)]
    pub name: String,
    /// The real field on a WE "movement" operator (a constant acceleration
    /// vector), not a generic scalar.
    #[serde(default)]
    pub gravity: Option<String>,
    /// `alphafade`: fade in over `[0, fadeintime]`, hold, fade out over
    /// `[fadeouttime, 1]` of normalized lifetime (CParticle.cpp createAlphaFadeOperator).
    #[serde(default)]
    pub fadeintime: Option<f64>,
    #[serde(default)]
    pub fadeouttime: Option<f64>,
    /// `sizechange`/`alphachange`/`colorchange`: linear ramp from `startvalue`
    /// to `endvalue` over normalized-lifetime `[starttime, endtime]`
    /// (CParticle.cpp's generic `fadeValue`). `startvalue`/`endvalue` are
    /// untyped because `sizechange`/`alphachange` use a plain number but
    /// `colorchange` uses an `"r g b"` vector string (e.g. `"1 0.75 0"`).
    #[serde(default)]
    pub starttime: Option<f64>,
    #[serde(default)]
    pub endtime: Option<f64>,
    #[serde(default)]
    pub startvalue: Option<serde_json::Value>,
    #[serde(default)]
    pub endvalue: Option<serde_json::Value>,
    /// `oscillatealpha`/`oscillatesize`/`oscillateposition`: per-particle
    /// random frequency/scale/phase, resampled once at spawn.
    #[serde(default)]
    pub frequencymin: Option<f64>,
    #[serde(default)]
    pub frequencymax: Option<f64>,
    #[serde(default)]
    pub scalemin: Option<f64>,
    #[serde(default)]
    pub scalemax: Option<f64>,
    #[serde(default)]
    pub phasemin: Option<f64>,
    #[serde(default)]
    pub phasemax: Option<f64>,
    /// `controlpointattract`: pulls particles toward `controlpoint` (index
    /// into the particle config's `controlpoint` array) whenever they're
    /// within `threshold/2` of it (CParticle.cpp createControlPointAttractOperator).
    #[serde(default)]
    pub controlpoint: Option<i64>,
    #[serde(default)]
    pub origin: Option<serde_json::Value>,
    #[serde(default)]
    pub scale: Option<f64>,
    #[serde(default)]
    pub threshold: Option<f64>,
    /// `angularmovement`: integrates `rotation.z` from `angularvelocityrandom`'s
    /// sampled `angular_velocity`, itself accelerated by `force.z` and decayed
    /// by `drag` each step (CParticle.cpp createAngularMovementOperator).
    #[serde(default)]
    pub drag: Option<f64>,
    #[serde(default)]
    pub force: Option<serde_json::Value>,
    /// `turbulence` (CParticle.cpp createTurbulenceOperator): curl-noise
    /// acceleration. Shares `scale`/`phasemin`/`phasemax` above.
    #[serde(default)]
    pub speedmin: Option<f64>,
    #[serde(default)]
    pub speedmax: Option<f64>,
    #[serde(default)]
    pub timescale: Option<f64>,
    #[serde(default)]
    pub mask: Option<serde_json::Value>,
    /// `vortex` (CParticle.cpp createVortexOperator): tangential spin around
    /// a control point. `flags`: 1 = infinite axis, 2 = maintain distance,
    /// 4 = ring shape.
    #[serde(default)]
    pub flags: Option<serde_json::Value>,
    #[serde(default)]
    pub axis: Option<serde_json::Value>,
    #[serde(default)]
    pub offset: Option<serde_json::Value>,
    #[serde(default)]
    pub distanceinner: Option<f64>,
    #[serde(default)]
    pub distanceouter: Option<f64>,
    #[serde(default)]
    pub speedinner: Option<f64>,
    #[serde(default)]
    pub speedouter: Option<f64>,
    #[serde(default)]
    pub centerforce: Option<f64>,
    #[serde(default)]
    pub ringradius: Option<f64>,
    #[serde(default)]
    pub ringwidth: Option<f64>,
    #[serde(default)]
    pub ringpulldistance: Option<f64>,
    #[serde(default)]
    pub ringpullforce: Option<f64>,
    /// `remapvalue` (CParticle createValueRemapOperator, real retail-binary
    /// dispatch tables recovered in the Ghidra report's Follow-up (x)):
    /// reads `input` (a named particle field, or noise sampled at the
    /// particle position/time when unset), shapes it through
    /// `transformfunction`, remaps `[inputrangemin, inputrangemax]` into
    /// `[outputrangemin, outputrangemax]`, and combines it into `output` via
    /// `operation`. `outputrangemin`/`max` are `"x y z"` vec3 for
    /// vector-valued outputs (velocity/position/color), scalar otherwise.
    #[serde(default)]
    pub input: Option<String>,
    #[serde(default)]
    pub inputcomponent: Option<String>,
    #[serde(default)]
    pub output: Option<String>,
    #[serde(default)]
    pub outputcomponent: Option<String>,
    #[serde(default)]
    pub operation: Option<String>,
    #[serde(default)]
    pub inputrangemin: Option<serde_json::Value>,
    #[serde(default)]
    pub inputrangemax: Option<serde_json::Value>,
    #[serde(default)]
    pub outputrangemin: Option<serde_json::Value>,
    #[serde(default)]
    pub outputrangemax: Option<serde_json::Value>,
    #[serde(default)]
    pub transformfunction: Option<String>,
    #[serde(default)]
    pub transforminputscale: Option<f64>,
    /// `collisionsphere`/`collisionbox`/`collisionbounds`/`collisionquad`/
    /// `collisionplane` (real operator names, Ghidra `strings.txt` around
    /// `140490008`, alongside `boids`/`collisionbehavior`/`bouncefactor` —
    /// no C++ reference implements any of these, so field semantics below
    /// are a reasoned reconstruction, not byte-verified). A sphere/box/
    /// bounds/quad collider's half-size or radius; box/bounds/quad read
    /// `size`'s x/y, sphere reads `radius`.
    #[serde(default)]
    pub radius: Option<f64>,
    #[serde(default)]
    pub size: Option<serde_json::Value>,
    /// `slide` (zero the into-surface velocity component), `bounce`
    /// (reflect it, scaled by `bouncefactor`), or `delete` (kill the
    /// particle on contact). Defaults to `bounce`, the most common
    /// particle-collision behavior in comparable engines.
    #[serde(default)]
    pub collisionbehavior: Option<String>,
    #[serde(default)]
    pub bouncefactor: Option<f64>,
    /// `boids` flocking rules (separation/alignment/cohesion) — real field
    /// names from the same table (`variablestrength`, `separationthreshold`,
    /// `neighborthreshold`, `maxspeed`, `separationfactor`,
    /// `alignmentfactor`, `cohesionfactor`).
    #[serde(default)]
    pub separationthreshold: Option<f64>,
    #[serde(default)]
    pub neighborthreshold: Option<f64>,
    #[serde(default)]
    pub maxspeed: Option<f64>,
    #[serde(default)]
    pub separationfactor: Option<f64>,
    #[serde(default)]
    pub alignmentfactor: Option<f64>,
    #[serde(default)]
    pub cohesionfactor: Option<f64>,
    #[serde(default)]
    pub variablestrength: Option<f64>,
    /// Plane-collider orientation (reuses the same "z-axis rotates a default
    /// normal" convention `controlpoint`/vortex offsets already use).
    #[serde(default)]
    pub angles: Option<serde_json::Value>,
}

/// Scene.json's per-instance `instanceoverride` on a particle object —
/// overrides the referenced preset's rate/size/speed/alpha/color.
#[derive(Debug, Default, Clone, Deserialize)]
pub struct InstanceOverride {
    pub alpha: Option<f64>,
    pub color: Option<String>,
    pub rate: Option<f64>,
    pub size: Option<f64>,
    pub speed: Option<f64>,
    /// Multiplier on the preset's `maxcount` pool size
    /// (CParticle.cpp: `adjustedMaxCount = maxCount * count`).
    pub count: Option<f64>,
    /// Multiplier on every spawned particle's lifetime — the reference
    /// applies it both in `createLifetimeRandomInitializer` and the
    /// default-spawn path, so a single multiply at spawn covers both.
    pub lifetime: Option<f64>,
    /// Per-channel color multiplier ("r g b", 0–1 floats) applied to the
    /// `colorrandom` result (CParticle.cpp:715-719). Distinct from `color`.
    pub colorn: Option<serde_json::Value>,
    /// `false` disables the whole system (ObjectParser default true).
    pub enabled: Option<serde_json::Value>,
    /// Remaining override keys — notably `controlpointN` ("x y z" strings,
    /// absolute scene coordinates) that reposition a preset's control
    /// points per instance (e.g. the discharge preset's arc endpoint).
    #[serde(flatten)]
    pub extra: std::collections::HashMap<String, serde_json::Value>,
}

/// Initializer names are matched exactly — they're fixed strings in the
/// reference's ObjectParser, and substring matching let
/// `turbulentvelocityrandom`/`angularvelocityrandom` shadow
/// `velocityrandom` (or `angularmovement` shadow `movement`) depending on
/// declaration order.
/// The `exponent` (distribution bias) of a named initializer, or 1.0 (uniform).
pub(super) fn initializer_exponent(initializers: &[Initializer], name: &str) -> f32 {
    initializers
        .iter()
        .find(|i| i.name == name)
        .and_then(|i| i.exponent)
        .map(|e| e as f32)
        .filter(|e| *e > 0.0)
        .unwrap_or(1.0)
}

pub(super) fn scalar_range_from_initializers(
    initializers: &[Initializer],
    name: &str,
    default_min: f32,
    default_max: f32,
) -> (f32, f32) {
    let Some(init) = initializers.iter().find(|init| init.name == name) else {
        return (default_min, default_max);
    };
    (
        init.min
            .as_ref()
            .and_then(value_as_f32)
            .unwrap_or(default_min),
        init.max
            .as_ref()
            .and_then(value_as_f32)
            .unwrap_or(default_max),
    )
}

/// Returns `Some((min, max))` when the named initializer exists and both
/// bounds parse as a 3-component vector (e.g. `velocityrandom`/`colorrandom`).
/// Exact-name matching for the same reason as `scalar_range_from_initializers`.
/// The (min, max) of initializer `name`, each falling back to WE's default
/// when omitted (ObjectParser.cpp: `colorrandom` black..white,
/// `velocityrandom` -32..32). `None` when there is no such initializer.
pub(super) fn vec3_range_from_initializers(
    initializers: &[Initializer],
    name: &str,
    default_min: [f32; 3],
    default_max: [f32; 3],
) -> Option<([f32; 3], [f32; 3])> {
    let init = initializers.iter().find(|init| init.name == name)?;
    let min = init
        .min
        .as_ref()
        .and_then(value_as_vec3)
        .unwrap_or(default_min);
    let max = init
        .max
        .as_ref()
        .and_then(value_as_vec3)
        .unwrap_or(default_max);
    Some((min, max))
}

pub(super) fn value_as_f32(v: &serde_json::Value) -> Option<f32> {
    v.as_f64().map(|f| f as f32)
}

/// Public alias for callers outside this module (e.g. model::resolve's
/// control-point override plumbing).
pub fn value_as_vec3_pub(v: &serde_json::Value) -> Option<[f32; 3]> {
    value_as_vec3(v)
}

pub(super) fn value_as_vec3(v: &serde_json::Value) -> Option<[f32; 3]> {
    match v {
        serde_json::Value::String(s) => Some(parse_f32_vec3(Some(s))),
        serde_json::Value::Array(arr) => {
            let parts: Vec<f32> = arr
                .iter()
                .filter_map(|x| x.as_f64().map(|f| f as f32))
                .collect();
            (parts.len() >= 3).then(|| [parts[0], parts[1], parts[2]])
        }
        _ => None,
    }
}

pub(super) fn parse_f32_vec3(s: Option<&str>) -> [f32; 3] {
    let Some(s) = s else { return [0.0; 3] };
    let parts: Vec<f32> = s
        .split_whitespace()
        .filter_map(|p| p.parse().ok())
        .collect();
    [
        parts.first().copied().unwrap_or(0.0),
        parts.get(1).copied().unwrap_or(0.0),
        parts.get(2).copied().unwrap_or(0.0),
    ]
}
