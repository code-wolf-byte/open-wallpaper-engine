use image::RgbaImage;
use serde::Deserialize;

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
enum AnimationMode {
    #[default]
    Loop,
    Once,
    RandomFrame,
}

impl AnimationMode {
    fn from_config(mode: Option<&str>) -> Self {
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
fn null_entries_as_default<'de, D, T>(d: D) -> Result<Vec<T>, D::Error>
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
    /// Presence of any `audioprocessing*` field makes emission react to the
    /// captured audio level (rate scales with loudness). We use the mode as
    /// the trigger; band bounds are parsed but not yet split per-frequency.
    #[serde(default)]
    pub audioprocessingmode: Option<serde_json::Value>,
    #[serde(default)]
    pub audioprocessingbounds: Option<serde_json::Value>,
    #[serde(default)]
    pub audioprocessingexponent: Option<serde_json::Value>,
    #[serde(default)]
    pub audioprocessingfrequencyend: Option<serde_json::Value>,
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

/// Per-particle oscillator state for `oscillatealpha`/`oscillatesize`: a
/// random frequency/scale/phase drawn once at spawn (CParticle.cpp captures
/// these lazily on first use, but since we never re-use particle slots
/// across configs, drawing them at spawn is equivalent).
#[derive(Clone, Copy)]
struct Oscillator1D {
    frequency: f32,
    scale: f32,
    phase: f32,
}

/// Same idea as `Oscillator1D` but per-axis for `oscillateposition`, which
/// integrates a velocity (derivative of `scale*cos(w*t+phase)`) into position
/// each frame rather than overwriting an absolute value.
#[derive(Clone, Copy)]
struct OscillatorPos {
    frequency: [f32; 2],
    scale: [f32; 2],
    phase: [f32; 2],
}

#[derive(Clone)]
struct Particle {
    /// Stable identity across `retain` compaction — child `eventfollow`
    /// instances track their parent particle by this, not by index.
    id: u64,
    x: f32,
    y: f32,
    vx: f32,
    vy: f32,
    life: f32,
    max_life: f32,
    size: f32,
    alpha: f32,
    color: [u8; 3],
    /// Alpha/size/color as set by the initializers at spawn — `alphafade`/
    /// `sizechange`/`colorchange` scale *this*, not the live (already-modified)
    /// value. `initial_color` is 0..1 (matching `colorchange`'s multiplier).
    initial_alpha: f32,
    initial_size: f32,
    initial_color: [f32; 3],
    osc_alpha: Option<Oscillator1D>,
    osc_size: Option<Oscillator1D>,
    osc_pos: Option<OscillatorPos>,
    /// Screen-space (Z-axis) rotation angle, radians. Only visually relevant
    /// once a sprite texture is drawn (`render_onto`'s textured path) — a
    /// flat-color circle is rotation-invariant.
    rotation: f32,
    angular_velocity: f32,
    /// Current sprite-sheet frame (fractional, truncated at draw time);
    /// `-1.0` at spawn means "not yet assigned" (only meaningful for
    /// `AnimationMode::RandomFrame`, which assigns it once and freezes it).
    frame: f32,
}

/// Parsed `oscillatealpha`/`oscillatesize`/`oscillateposition` operator
/// config — random ranges each spawned particle draws its own values from.
#[derive(Clone, Copy)]
struct OscillateParams {
    freq_min: f32,
    freq_max: f32,
    scale_min: f32,
    scale_max: f32,
    phase_min: f32,
    phase_max: f32,
}

impl OscillateParams {
    fn from_operator(op: &Operator) -> Self {
        let freq_min = op.frequencymin.unwrap_or(0.0) as f32;
        let freq_max = op.frequencymax.unwrap_or(freq_min as f64) as f32;
        let scale_min = op.scalemin.or(op.scalemax).unwrap_or(0.0) as f32;
        let scale_max = op.scalemax.or(op.scalemin).unwrap_or(0.0) as f32;
        let phase_min = op.phasemin.unwrap_or(0.0) as f32;
        let phase_max = op.phasemax.unwrap_or(0.0) as f32;
        Self {
            freq_min,
            freq_max,
            scale_min,
            scale_max,
            phase_min,
            phase_max,
        }
    }

    /// CParticle.cpp always draws phase from `[phaseMin, phaseMax + 2*PI]` —
    /// even with no phase fields at all (both 0), this yields a fully random
    /// phase in `[0, 2*PI]`, which is what real presets (lacking phase
    /// fields entirely) actually rely on.
    fn sample(&self) -> Oscillator1D {
        Oscillator1D {
            frequency: self.freq_min + fastrand::f32() * (self.freq_max - self.freq_min),
            scale: self.scale_min + fastrand::f32() * (self.scale_max - self.scale_min),
            phase: self.phase_min
                + fastrand::f32() * (self.phase_max + std::f32::consts::TAU - self.phase_min),
        }
    }
}

pub struct ParticleSystem {
    particles: Vec<Particle>,
    emitters: Vec<EmitterState>,
    max_count: usize,
    life_min: f32,
    life_max: f32,
    size_min: f32,
    size_max: f32,
    /// `exponent` distribution bias for lifetime/size random sampling (1.0 = uniform).
    life_exp: f32,
    size_exp: f32,
    /// `alpharandom` initializer range (CParticle.cpp's `AlphaRandomInitializer`):
    /// defaults to `(1.0, 1.0)` — a no-op — when the config has no such
    /// initializer at all, matching `ParticleInstance::alpha`'s `1.0f` default.
    alpha_min: f32,
    alpha_max: f32,
    velocity_min: Option<[f32; 3]>,
    velocity_max: Option<[f32; 3]>,
    color_min: [f32; 3],
    color_max: [f32; 3],
    gravity: [f32; 2],
    /// Scene-global constant acceleration (`general.gravity*` + `wind*`),
    /// added to every particle's velocity each step. Set post-construction.
    scene_force: [f32; 2],
    /// Current audio loudness in ~[0,1], set per frame; audio-reactive emitters
    /// scale their emission rate by it.
    audio_level: f32,
    /// `movement.drag`: velocity decays by `1 - drag*dt` (clamped at 0)
    /// each step, matching the reference's reversal-preventing clamp.
    movement_drag: f32,
    /// `alphafade`: (fadeintime, fadeouttime), normalized-lifetime fractions.
    alphafade: Option<(f32, f32)>,
    /// `sizechange`/`alphachange`: (starttime, endtime, startvalue, endvalue).
    sizechange: Option<(f32, f32, f32, f32)>,
    alphachange: Option<(f32, f32, f32, f32)>,
    /// `colorchange`: (starttime, endtime, start_color, end_color), colors in
    /// 0..1 (unlike `colorrandom`'s 0..255 — matches the real presets, e.g.
    /// `"startvalue": "1 0.75 0"`).
    colorchange: Option<(f32, f32, [f32; 3], [f32; 3])>,
    oscillate_alpha: Option<OscillateParams>,
    oscillate_size: Option<OscillateParams>,
    oscillate_position: Option<OscillateParams>,
    alpha_mult: f32,
    size_mult: f32,
    speed_mult: f32,
    rate_mult: f32,
    /// `instanceoverride.lifetime` multiplier on spawned lifetimes.
    lifetime_mult: f32,
    /// `instanceoverride.colorn` per-channel multiplier (0-1) on the
    /// `colorrandom` result.
    colorn_mult: [f32; 3],
    color_override: Option<[u8; 3]>,
    /// Resolved once at construction: `spawn_center + offset` per control
    /// point, indexed positionally (falls back to declaration order when a
    /// config omits `id`). Nested parent-space transforms and mouse-linked
    /// (`locktopointer`)/world-space (`flags & 2`) control points are not
    /// modeled — same simplification already accepted for emitter origins.
    control_points: Vec<[f32; 3]>,
    /// `controlpointattract`: (control point index, origin offset, scale, threshold).
    control_point_attract: Option<(usize, [f32; 3], f32, f32)>,
    /// True when `config.renderer` names a `"rope"`/`"ropetrail"` renderer —
    /// draws a connected ribbon through living particles instead of
    /// independent circles (CParticle::renderRope).
    rope_mode: bool,
    rope_subdivision: usize,
    /// Rope UV controls (`uvscale`/`uvscrolling`/`uvsmoothing` renderer keys).
    rope_uv_scale: f32,
    rope_uv_scrolling: bool,
    rope_uv_smoothing: bool,
    /// `angularvelocityrandom` initializer range (z-axis component only).
    angular_velocity_min: f32,
    angular_velocity_max: f32,
    /// `angularmovement` operator (z-axis `force` component, plus `drag`).
    angular_force: f32,
    angular_drag: f32,
    /// Sprite-sheet animation: how many frames the resolved sprite has (0 if
    /// no sprite, or a plain single-frame one) and its total loop duration
    /// in seconds — set post-construction via `set_sprite_frames` once the
    /// caller has actually resolved the sprite texture (this module stays
    /// asset-agnostic, same convention as `material`/`render_onto`).
    sprite_frame_count: usize,
    sprite_duration: f32,
    animation_mode: AnimationMode,
    sequence_multiplier: f32,
    /// `rotationrandom` initializer range (z-axis component, radians) —
    /// `None` leaves spawn rotation at 0, matching the reference's absent-
    /// initializer default.
    rotation_range: Option<(f32, f32)>,
    /// `spritetrail` renderer parameters.
    sprite_trail: Option<SpriteTrailParams>,
    /// `mapsequencearoundcontrolpoint` initializer + its running sequence
    /// counter (shared across spawns, wraps at `count`).
    map_sequence: Option<MapSequenceParams>,
    map_sequence_index: u32,
    /// `mapsequencebetweencontrolpoints` initializer + its counter.
    map_seq_between: Option<MapSequenceBetweenParams>,
    map_seq_between_index: u32,
    /// `turbulentvelocityrandom` initializer parameters.
    turbulent_velocity: Option<TurbulentVelocityParams>,
    /// `turbulence` operator parameters.
    turbulence: Option<TurbulenceParams>,
    /// `vortex` operator parameters.
    vortex: Option<VortexParams>,
    /// `remapvalue` operator parameters — a `Vec` since real content
    /// (`particles/presets/rain_screen.json`) attaches more than one (one
    /// targeting `velocity`, one targeting `speed`) and both must run; a
    /// single `Option` here used to silently drop every instance after the
    /// first.
    remap: Vec<RemapParams>,
    /// `collisionsphere`/`collisionbox`/`collisionbounds`/`collisionquad`/
    /// `collisionplane` operators — every one attached, since a system can
    /// have more than one collider (e.g. a floor plane plus a wall box).
    collisions: Vec<CollisionParams>,
    /// `boids` operator parameters.
    boids: Option<BoidsParams>,
    /// Simulation clock (sum of `step` dts) — drives the time-scrolled
    /// noise fields, the reference's `m_time`/`currentTime`.
    time: f32,
    /// `starttime`: emission is suppressed until the clock reaches this.
    start_time: f32,
    /// Preset `flags & 4` (perspective particles): sphere emitters use the
    /// 3D spherical-shell spawn distribution instead of the 2D disk.
    shell_3d: bool,
    /// Monotonic source for `Particle::id`.
    next_particle_id: u64,
    /// Attached child presets (see [`ChildRef`]); each holds its own live
    /// instances, stepped and drawn with this system.
    children: Vec<ChildSystem>,
}

/// How a child preset binds to its parent (see [`ChildRef`]).
#[derive(Debug, Clone, Copy, PartialEq)]
enum ChildType {
    Static,
    Follow,
    Spawn,
    Death,
}

/// A resolved child preset plus its live instances. The child's material
/// sprite/blending are its own (a glow child can be additive over a
/// normal-blended parent); both stages draw into the same canvas.
pub struct ChildSystem {
    child_type: ChildType,
    probability: f32,
    max_instances: usize,
    config: ParticleConfig,
    sprite: Option<ParticleSprite>,
    additive: bool,
    instances: Vec<ChildInstance>,
}

struct ChildInstance {
    /// `Some` while following a live parent particle (`eventfollow`);
    /// cleared when the parent dies so the instance stops emitting and
    /// retires once its remaining particles expire.
    parent_id: Option<u64>,
    /// Where this instance's emitters currently sit (parent particle
    /// position for `eventfollow`, event position for bursts).
    last_center: [f32; 2],
    /// Burst instances (`eventspawn`/`eventdeath`) stop emitting at this
    /// system-time cutoff — the preset itself has no end marker.
    emit_until: Option<f32>,
    /// `static` children run for the parent system's whole life and are
    /// never retired, even while momentarily empty between spawns.
    persistent: bool,
    system: ParticleSystem,
}

/// `turbulentvelocityrandom` (CParticle.cpp): spawn velocity picked from a
/// curl-noise field, angle-limited around `forward`.
#[derive(Clone, Copy)]
struct TurbulentVelocityParams {
    speed_min: f32,
    speed_max: f32,
    /// Tilt angle (radians) around `right`.
    offset: f32,
    /// Direction cone: `< 2` limits deviation from `forward` to
    /// `scale/2 * pi`.
    scale: f32,
    forward: [f32; 3],
    right: [f32; 3],
    timescale: f32,
    phase_min: f32,
    phase_max: f32,
}

/// `turbulence` operator (CParticle.cpp): curl-noise acceleration. `phase`
/// and `speed` are drawn once per operator instance, not per particle.
#[derive(Clone, Copy)]
struct TurbulenceParams {
    noise_scale: f32,
    timescale: f32,
    mask: [f32; 3],
    phase: f32,
    speed: f32,
}

/// `spritetrail` renderer: velocity-aligned, velocity-stretched sprite quad.
#[derive(Clone, Copy)]
struct SpriteTrailParams {
    length: f32,
    max_length: f32,
    min_length: f32,
}

/// `mapsequencebetweencontrolpoints` initializer: spawns successive
/// particles at evenly spaced points along the segment from control point
/// 0 to control point 1 (`count` slots). `limitbehavior: "mirror"`
/// ping-pongs the sequence; anything else wraps. Neither the C++ reference
/// nor its parser implements this one — semantics derived from real
/// content (the `discharge` preset draws a rope through the chain,
/// yielding a lightning arc between the two anchors).
#[derive(Clone, Copy)]
struct MapSequenceBetweenParams {
    count: u32,
    mirror: bool,
}

/// `mapsequencearoundcontrolpoint` initializer (CParticle.cpp): spawns each
/// particle at the control point with a velocity rotated by the next angle
/// in an evenly-divided circle (`sequence / count * 2pi`).
#[derive(Clone, Copy)]
struct MapSequenceParams {
    control_point: usize,
    count: u32,
    speed_min: [f32; 3],
    speed_max: [f32; 3],
}

/// `vortex` operator (CParticle.cpp): tangential spin around a control
/// point, with optional ring shape and center attraction.
#[derive(Clone, Copy)]
struct VortexParams {
    control_point: i64,
    infinite_axis: bool,
    maintain_distance: bool,
    ring_shape: bool,
    axis: [f32; 3],
    offset: [f32; 3],
    distance_inner: f32,
    distance_outer: f32,
    speed_inner: f32,
    speed_outer: f32,
    center_force: f32,
    ring_radius: f32,
    ring_width: f32,
    ring_pull_distance: f32,
    ring_pull_force: f32,
}

/// Collider shape for a `collisionsphere`/`collisionbox`/`collisionbounds`/
/// `collisionquad`/`collisionplane` operator (see the `Operator` field docs
/// for the "no C++ reference, reasoned reconstruction" caveat). `Box` also
/// covers `collisionbounds`/`collisionquad` — this 2D engine has no distinct
/// notion of "world bounds" vs. "an oriented flat quad" vs. "a box", so all
/// three collapse to the same axis-aligned-rectangle test.
#[derive(Clone, Copy)]
enum CollisionShape {
    Sphere { radius: f32 },
    Box { half_extent: [f32; 2] },
    /// `normal` points away from the solid side (into free space).
    Plane { normal: [f32; 2] },
}

/// `collisionbehavior`: what happens to a particle that penetrates a
/// collider this frame.
#[derive(Clone, Copy, PartialEq)]
enum CollisionBehavior {
    Bounce,
    Slide,
    Delete,
}

impl CollisionBehavior {
    fn parse(s: Option<&str>) -> Self {
        match s {
            Some("slide") => Self::Slide,
            Some("delete") => Self::Delete,
            _ => Self::Bounce,
        }
    }
}

/// One resolved collider: anchored at a `controlpoint` (+ `offset`), same
/// static-at-construction simplification `vortex`/`controlpointattract`
/// already accept for control points.
#[derive(Clone, Copy)]
struct CollisionParams {
    shape: CollisionShape,
    center: [f32; 2],
    behavior: CollisionBehavior,
    bounce_factor: f32,
}

/// `boids` operator: standard separation/alignment/cohesion flocking rules.
/// Real field names recovered from the binary (see the `Operator` field
/// docs); the flocking algorithm itself is the industry-standard
/// three-rule formulation, not independently verified against WE's own
/// (unrecoverable) per-pixel/per-particle GPU dispatch.
#[derive(Clone, Copy)]
struct BoidsParams {
    separation_threshold: f32,
    neighbor_threshold: f32,
    max_speed: f32,
    separation_factor: f32,
    alignment_factor: f32,
    cohesion_factor: f32,
    /// `variablestrength != 0`: jitter each particle's flocking-force
    /// magnitude by a deterministic per-particle hash of its id, so a flock
    /// doesn't move as one mechanically uniform block.
    variable_strength: bool,
}

/// `remapvalue`'s `input`/`output` field selector. Real, named fields
/// recovered directly from the retail binary's own dispatch table (Ghidra
/// report Follow-up (x)); only the subset this engine already tracks
/// per-particle state for is implemented. The rest
/// (`distancetocontrolpoint`/`positionbetweentwocontrolpoints`/`runtime`/
/// `timeofday`/`particlesystemtime`/`layertime`/`controlpoint`/
/// `deltatocontrolpoint`/`directiontocontrolpoint`/`layerorigin`) are real
/// names too, but need control-point/scene-clock plumbing this operator
/// doesn't have access to and no cached real content (271 Workshop items
/// checked) exercises any of them — falling through to `None` (noise mode)
/// rather than silently misreading a wrong field.
#[derive(Clone, Copy, PartialEq, Debug)]
enum RemapField {
    /// No `input` configured: sample noise at the particle position/time —
    /// the original, real-content-validated behavior, unchanged.
    None,
    LifetimeFraction,
    MaxLifetime,
    Size,
    Opacity,
    Speed,
    Rotation,
    AngularSpeed,
    Color,
    Position,
    Velocity,
}

impl RemapField {
    fn parse(s: Option<&str>) -> Self {
        match s {
            Some("lifetimefraction") => Self::LifetimeFraction,
            Some("maxlifetime") => Self::MaxLifetime,
            Some("size") => Self::Size,
            Some("opacity") => Self::Opacity,
            Some("speed") => Self::Speed,
            Some("rotation") => Self::Rotation,
            Some("angularspeed") => Self::AngularSpeed,
            Some("color") => Self::Color,
            Some("position") => Self::Position,
            Some("velocity") => Self::Velocity,
            _ => Self::None,
        }
    }
}

/// `remapvalue`'s `output` field — same real dispatch table as
/// [`RemapField`], restricted to fields this operator can write back to.
#[derive(Clone, Copy, PartialEq, Debug)]
enum RemapOutput {
    Velocity,
    Speed,
    Size,
    Opacity,
    Rotation,
    AngularSpeed,
    Color,
    Position,
}

impl RemapOutput {
    fn parse(s: Option<&str>) -> Self {
        match s {
            Some("speed") => Self::Speed,
            Some("size") => Self::Size,
            Some("opacity") => Self::Opacity,
            Some("rotation") => Self::Rotation,
            Some("angularspeed") => Self::AngularSpeed,
            Some("color") => Self::Color,
            Some("position") => Self::Position,
            // "velocity" and anything unrecognized: existing default.
            _ => Self::Velocity,
        }
    }
}

/// `remapvalue`'s `operation` — how the remapped value combines with the
/// output field's current value. `Multiply` is corroborated directly: the
/// retail binary's own `alphafade`-family default-preset builders
/// (`FUN_1401bfbb0`/`FUN_1401bc4b0`, Follow-up (x)) construct exactly
/// `operation=multiply, input=lifetimefraction` to fade a field by a
/// lifetime-driven factor — the same shape as this engine's own
/// (independently, already-validated) `alphafade` — so "multiply the
/// current value" is a corroborated reading, not a guess. `Add`/`Subtract`
/// follow the same current-value-combine pattern; no real cached content
/// exercises them to cross-check against.
#[derive(Clone, Copy, PartialEq, Debug)]
enum RemapOp {
    /// Default: replace the output field outright (the original,
    /// real-content-validated `remapvalue` behavior).
    Remap,
    Multiply,
    Add,
    Subtract,
}

impl RemapOp {
    fn parse(s: Option<&str>) -> Self {
        match s {
            Some("multiply") => Self::Multiply,
            Some("add") => Self::Add,
            Some("subtract") => Self::Subtract,
            _ => Self::Remap,
        }
    }

    fn combine(self, current: f32, remapped: f32) -> f32 {
        match self {
            Self::Remap => remapped,
            Self::Multiply => current * remapped,
            Self::Add => current + remapped,
            Self::Subtract => current - remapped,
        }
    }
}

/// `remapvalue`'s `transformfunction` — shapes the normalized `[0,1]` input
/// before it's remapped into the output range. The retail binary gave up
/// only the *names* (Follow-up (x)), not formulas: `Sine`/`Square`/
/// `Triangle` are the standard one-period constructions for each name, an
/// honest best-effort reading rather than a verified port. `Saw` is a
/// linear ramp over a single non-wrapping `[0,1]` pass, which makes it
/// indistinguishable here from `None` — documented, not hidden.
/// `Simplex`/`Fbm` are unchanged from before: real, position/time-driven
/// noise sampling, validated against real content
/// (`particles/presets/rain_screen.json`).
#[derive(Clone, Copy, PartialEq, Debug)]
enum RemapTransform {
    None,
    Sine,
    Square,
    Saw,
    Triangle,
    Simplex,
    Fbm,
}

impl RemapTransform {
    /// Parses an explicit, non-empty `transformfunction` value. Whether an
    /// *absent* key defaults to `Simplex` (noise generator) or `None`
    /// (passthrough) depends on whether `input` was also given — see the
    /// call site in `ParticleSystem::new` — so that's handled there, not
    /// here.
    fn parse(s: &str) -> Self {
        match s {
            "none" => Self::None,
            "sine" => Self::Sine,
            "square" => Self::Square,
            "saw" => Self::Saw,
            "triangle" => Self::Triangle,
            "fbmnoise" => Self::Fbm,
            // "simplexnoise" and anything unrecognized.
            _ => Self::Simplex,
        }
    }

    fn is_noise(self) -> bool {
        matches!(self, Self::Simplex | Self::Fbm)
    }

    /// Shape an already-clamped `[0,1]` value. Noise variants are handled
    /// separately (via position/time sampling, not this function).
    fn shape(self, t: f32) -> f32 {
        match self {
            Self::None | Self::Saw | Self::Simplex | Self::Fbm => t,
            Self::Sine => 0.5 + 0.5 * (t * std::f32::consts::TAU).sin(),
            Self::Square => {
                if t < 0.5 {
                    0.0
                } else {
                    1.0
                }
            }
            Self::Triangle => 1.0 - (2.0 * t - 1.0).abs(),
        }
    }
}

/// `remapvalue`'s `inputcomponent`/`outputcomponent` — real, named
/// selectors recovered from the retail binary (Follow-up (x)).
#[derive(Clone, Copy, PartialEq, Debug)]
enum RemapComponent {
    All,
    X,
    Y,
    Z,
    Sum,
    Average,
    Max,
    Min,
}

impl RemapComponent {
    fn parse(s: Option<&str>) -> Self {
        match s {
            Some("x") => Self::X,
            Some("y") => Self::Y,
            Some("z") => Self::Z,
            Some("sum") => Self::Sum,
            Some("average") => Self::Average,
            Some("max") => Self::Max,
            Some("min") => Self::Min,
            _ => Self::All,
        }
    }

    /// Collapse a (2- or 3-axis) vector to one scalar, for reading a
    /// vector-valued input field. `All`/`Average` both average — a vector
    /// input must collapse to one scalar for the remap math regardless, and
    /// averaging is the least-arbitrary choice among these (stays in the
    /// field's native range, unlike `Sum`, which scales with axis count).
    fn reduce(self, v: [f32; 3], axes: usize) -> f32 {
        let axes = axes.min(3);
        match self {
            Self::X => v[0],
            Self::Y => v[1],
            Self::Z => v[2],
            Self::Sum => v[..axes].iter().sum(),
            Self::Max => v[..axes].iter().cloned().fold(f32::MIN, f32::max),
            Self::Min => v[..axes].iter().cloned().fold(f32::MAX, f32::min),
            Self::All | Self::Average => v[..axes].iter().sum::<f32>() / axes as f32,
        }
    }
}

/// `remapvalue`: reads `input` (a named particle field, or noise sampled at
/// the particle position when unset), shapes it, remaps it into
/// `[out_min, out_max]`, and combines it into `output` via `operation`. See
/// [`RemapField`]/[`RemapOutput`]/[`RemapOp`]/[`RemapTransform`] for what's
/// grounded in the retail binary vs. an honest best-effort reading.
#[derive(Clone, Copy)]
struct RemapParams {
    input: RemapField,
    input_component: RemapComponent,
    output: RemapOutput,
    output_component: RemapComponent,
    operation: RemapOp,
    transform: RemapTransform,
    in_min: f32,
    in_max: f32,
    out_min: [f32; 3],
    out_max: [f32; 3],
    input_scale: f32,
}

impl RemapParams {
    /// Read the configured input field as a single scalar (vector fields
    /// collapse via `input_component`). Only meaningful when `input !=
    /// RemapField::None` — noise mode is handled entirely in `sample01`.
    fn read_input(&self, p: &Particle, lifetime_pos: f32, max_life: f32) -> f32 {
        match self.input {
            RemapField::None => 0.0,
            RemapField::LifetimeFraction => lifetime_pos,
            RemapField::MaxLifetime => max_life,
            RemapField::Size => p.size,
            RemapField::Opacity => p.alpha,
            RemapField::Speed => (p.vx * p.vx + p.vy * p.vy).sqrt(),
            RemapField::Rotation => p.rotation,
            RemapField::AngularSpeed => p.angular_velocity,
            RemapField::Color => self.input_component.reduce(
                [
                    p.color[0] as f32 / 255.0,
                    p.color[1] as f32 / 255.0,
                    p.color[2] as f32 / 255.0,
                ],
                3,
            ),
            RemapField::Position => self.input_component.reduce([p.x, p.y, 0.0], 2),
            RemapField::Velocity => self.input_component.reduce([p.vx, p.vy, 0.0], 2),
        }
    }

    /// The shaped `[0,1]` value to lerp into `[out_min, out_max]`.
    /// `axis_seed` only matters in noise mode (a different seed per output
    /// axis, exactly the original x/y-decorrelation this operator always
    /// used); field mode ignores it, which is what gives a scalar field's
    /// single value its "broadcast to every output axis" behavior.
    fn sample01(&self, axis_seed: f64, p: &Particle, time: f32, lifetime_pos: f32, max_life: f32) -> f32 {
        if self.input == RemapField::None || self.transform.is_noise() {
            use crate::engine::noise::fbm_noise;
            let s = self.input_scale;
            let (x, y, t) = ((p.x * s) as f64, (p.y * s) as f64, time as f64);
            let octaves = if self.transform == RemapTransform::Fbm { 4 } else { 1 };
            let n = fbm_noise(x + axis_seed, y, t, octaves);
            ((n * 0.5 + 0.5) as f32).clamp(0.0, 1.0) // [-1,1] → [0,1]
        } else {
            let raw = self.read_input(p, lifetime_pos, max_life);
            let range = (self.in_max - self.in_min).abs().max(1e-6);
            let normalized = ((raw - self.in_min) / range).clamp(0.0, 1.0);
            self.transform.shape(normalized)
        }
    }

    /// Apply this operator to a 2- or 3-axis vector field (`velocity`/
    /// `position`/`color`), honoring `output_component` (writes only that
    /// one axis, leaving the rest of `current` untouched) or every in-range
    /// axis when it's `All`. Distinct per-axis noise seeds match the
    /// original x/y-decorrelated noise sampling exactly; field-mode inputs
    /// naturally broadcast the same value to every written axis instead
    /// (`sample01` ignores the seed outside noise mode).
    fn apply_vector(&self, p: &Particle, time: f32, lifetime_pos: f32, current: [f32; 3], axes: usize) -> [f32; 3] {
        let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
        let seeds = [0.0f64, 13.7, 27.4];
        let mut out = current;
        let apply_axis = |axis: usize, out: &mut [f32; 3]| {
            let t = self.sample01(seeds[axis], p, time, lifetime_pos, p.max_life);
            let remapped = lerp(self.out_min[axis], self.out_max[axis], t);
            out[axis] = self.operation.combine(current[axis], remapped);
        };
        match self.output_component {
            RemapComponent::X => apply_axis(0, &mut out),
            RemapComponent::Y if axes >= 2 => apply_axis(1, &mut out),
            RemapComponent::Z if axes >= 3 => apply_axis(2, &mut out),
            _ => {
                for axis in 0..axes {
                    apply_axis(axis, &mut out);
                }
            }
        }
        out
    }
}

struct EmitterState {
    id: u32,
    rate: f32,
    origin: [f32; 3],
    directions: [f32; 3],
    sign: [f32; 3],
    /// True for `sphererandom` (annulus/shell spawn), false for box.
    is_sphere: bool,
    /// Spawn-offset range: sphere radius in `[0]` (min/max annulus), or the
    /// box emitter's per-axis half-extents. CParticle.cpp spawns particles
    /// randomly WITHIN this volume, scaled per-axis by `directions` —
    /// distancemax is a POSITION range, never a velocity.
    distance_min: [f32; 3],
    distance_max: [f32; 3],
    /// Optional emitter-driven radial speed (`speedmin`/`speedmax` JSON
    /// fields — distinct from distance!). 0 = the reference's default:
    /// emitters set no velocity, initializers do.
    speed_min: f32,
    speed_max: f32,
    accumulator: f32,
    /// `instantaneous` one-shot burst count, emitted once when the emitter
    /// activates; `burst_done` tracks that it's fired.
    burst: u32,
    burst_done: bool,
    /// `duration`: emitter stops after `start_time + duration` (None = forever).
    duration: Option<f32>,
    /// True when the emitter has `audioprocessing*` — its rate scales with the
    /// current audio level.
    audio_reactive: bool,
}

impl ParticleSystem {
    /// Build a running particle system for one scene object.
    ///
    /// `spawn_center` is the object's own scene position, already converted to
    /// canvas pixel coordinates (see `render.rs`'s WE-origin-to-pixel
    /// conversion) — a WE emitter's own `origin` field is a *local* offset
    /// from the object's position, not an absolute canvas coordinate.
    pub fn from_config(
        config: &ParticleConfig,
        spawn_center: [f32; 2],
        overrides: Option<&InstanceOverride>,
    ) -> Self {
        let emitters = config
            .emitter
            .iter()
            .map(|e| {
                let local_origin = parse_f32_vec3(e.origin.as_deref());
                // Reference default (1, 1, 0) — a zero default would scale
                // every spawn offset to nothing for emitters that omit the
                // field, collapsing them to a point at the origin.
                let directions = e
                    .directions
                    .as_deref()
                    .map(|d| parse_f32_vec3(Some(d)))
                    .unwrap_or([1.0, 1.0, 0.0]);
                let sign = parse_f32_vec3(e.sign.as_deref());
                let is_sphere = e.name == "sphererandom";

                // Sphere radii are scalars, box extents "x y z" vectors —
                // accept either shape for both (splat scalars).
                let dist = |v: &Option<serde_json::Value>, default: f32| -> [f32; 3] {
                    v.as_ref()
                        .and_then(|v| {
                            value_as_vec3(v).or_else(|| value_as_f32(v).map(|f| [f, f, f]))
                        })
                        .unwrap_or([default, default, default])
                };
                let distance_min = dist(&e.distancemin, 0.0);
                // Reference default 256 for both emitter kinds.
                let distance_max = dist(&e.distancemax, 256.0);
                let speed_min = e.speedmin.as_ref().and_then(value_as_f32).unwrap_or(0.0);
                let speed_max = e.speedmax.as_ref().and_then(value_as_f32).unwrap_or(0.0);

                EmitterState {
                    id: e.id.unwrap_or(0),
                    rate: e.rate.unwrap_or(10.0) as f32,
                    origin: [
                        spawn_center[0] + local_origin[0],
                        // The emitter's local origin is authored y-up
                        // (CParticle.cpp negates it in both createBoxEmitter
                        // and createSphereEmitter): "0 768 0" means 768
                        // ABOVE the object — e.g. rainperspective's spawn
                        // band sits above the frame so drops fall through
                        // it, not 768 below (which pushed the whole band
                        // off-screen on 1080p scenes and rain never showed).
                        spawn_center[1] - local_origin[1],
                        local_origin[2],
                    ],
                    directions,
                    sign,
                    is_sphere,
                    distance_min,
                    distance_max,
                    speed_min,
                    speed_max,
                    accumulator: 0.0,
                    burst: e.instantaneous.unwrap_or(0.0).max(0.0) as u32,
                    burst_done: false,
                    duration: e.duration.map(|d| d as f32),
                    audio_reactive: e.audioprocessingmode.is_some()
                        || e.audioprocessingbounds.is_some(),
                }
            })
            .collect();

        // `instanceoverride.count` scales the pool before the default kicks
        // in (CParticle.cpp: `adjustedMaxCount = maxCount * countMultiplier;
        // maxParticles = adjusted > 0 ? adjusted : DEFAULT`).
        let count_mult = overrides.and_then(|o| o.count).unwrap_or(1.0);
        let max_count = match config.maxcount {
            Some(n) if n > 0 => match (n as f64 * count_mult) as usize {
                0 => 500,
                adjusted => adjusted,
            },
            _ => 500,
        };

        // Initializer-absent defaults are the reference's exact spawn values
        // (CParticle.cpp: `p.lifetime = 1.0`, `p.size = 20.0` before
        // initializers run) — size passes through the sizerandom /2, so the
        // range default is 40.
        let (life_min, life_max) =
            scalar_range_from_initializers(&config.initializer, "lifetimerandom", 1.0, 1.0);
        let (size_min, size_max) =
            scalar_range_from_initializers(&config.initializer, "sizerandom", 40.0, 40.0);
        let life_exp = initializer_exponent(&config.initializer, "lifetimerandom");
        let size_exp = initializer_exponent(&config.initializer, "sizerandom");
        let (alpha_min, alpha_max) =
            scalar_range_from_initializers(&config.initializer, "alpharandom", 1.0, 1.0);
        let velocity_range = vec3_range_from_initializers(&config.initializer, "velocityrandom");
        let (velocity_min, velocity_max) = match velocity_range {
            Some((min, max)) => (Some(min), Some(max)),
            None => (None, None),
        };
        let (color_min, color_max) =
            vec3_range_from_initializers(&config.initializer, "colorrandom")
                .unwrap_or(([255.0; 3], [255.0; 3]));

        let movement_op = config.operator.iter().find(|op| op.name == "movement");
        let gravity = movement_op
            .and_then(|op| op.gravity.as_deref())
            .map(|s| parse_f32_vec3(Some(s)))
            .map(|v| [v[0], v[1]])
            .unwrap_or([0.0, 0.0]);
        // `movement.drag`: per-second velocity decay (CParticle.cpp
        // createMovementOperator) — without it, operators that keep adding
        // velocity every frame (turbulence, vortex) grow unbounded instead
        // of settling into the drifting equilibrium real presets tune for.
        let movement_drag = movement_op.and_then(|op| op.drag).unwrap_or(0.0) as f32;

        let color_override = overrides
            .and_then(|o| o.color.as_deref())
            .map(|s| parse_f32_vec3(Some(s)))
            .map(|v| [v[0] as u8, v[1] as u8, v[2] as u8]);

        let alphafade = config
            .operator
            .iter()
            .find(|op| op.name == "alphafade")
            .map(|op| {
                (
                    op.fadeintime.unwrap_or(0.0) as f32,
                    op.fadeouttime.unwrap_or(1.0) as f32,
                )
            });
        let fade_range_op = |name: &str| -> Option<(f32, f32, f32, f32)> {
            config.operator.iter().find(|op| op.name == name).map(|op| {
                (
                    op.starttime.unwrap_or(0.0) as f32,
                    op.endtime.unwrap_or(1.0) as f32,
                    op.startvalue.as_ref().and_then(value_as_f32).unwrap_or(0.0),
                    op.endvalue.as_ref().and_then(value_as_f32).unwrap_or(1.0),
                )
            })
        };
        let sizechange = fade_range_op("sizechange");
        let alphachange = fade_range_op("alphachange");
        let colorchange = config
            .operator
            .iter()
            .find(|op| op.name == "colorchange")
            .map(|op| {
                (
                    op.starttime.unwrap_or(0.0) as f32,
                    op.endtime.unwrap_or(1.0) as f32,
                    op.startvalue
                        .as_ref()
                        .and_then(value_as_vec3)
                        .unwrap_or([1.0; 3]),
                    op.endvalue
                        .as_ref()
                        .and_then(value_as_vec3)
                        .unwrap_or([1.0; 3]),
                )
            });
        let oscillate_op = |name: &str| -> Option<OscillateParams> {
            config
                .operator
                .iter()
                .find(|op| op.name == name)
                .map(OscillateParams::from_operator)
        };
        let oscillate_alpha = oscillate_op("oscillatealpha");
        let oscillate_size = oscillate_op("oscillatesize");
        let oscillate_position = oscillate_op("oscillateposition");

        // Positional indexing (declaration order), not `id`-keyed slot
        // placement — real presets declare control points in id order, and
        // consumers (e.g. `controlpointattract`) reference them by a plain
        // array index anyway.
        let control_points: Vec<[f32; 3]> = config
            .controlpoint
            .iter()
            .map(|cp| {
                let mut offset = cp
                    .offset
                    .as_ref()
                    .and_then(value_as_vec3)
                    .unwrap_or([0.0; 3]);
                // `angles.z` rotates the offset about the object origin.
                if let Some(az) = cp
                    .angles
                    .as_ref()
                    .and_then(value_as_vec3)
                    .map(|a| a[2])
                    .filter(|z| z.abs() > 1e-6)
                {
                    let (s, c) = az.sin_cos();
                    offset = [
                        offset[0] * c - offset[1] * s,
                        offset[0] * s + offset[1] * c,
                        offset[2],
                    ];
                }
                [
                    spawn_center[0] + offset[0],
                    spawn_center[1] + offset[1],
                    offset[2],
                ]
            })
            .collect();

        let control_point_attract = config
            .operator
            .iter()
            .find(|op| op.name == "controlpointattract")
            .map(|op| {
                let idx = op.controlpoint.unwrap_or(0).max(0) as usize;
                let origin = op
                    .origin
                    .as_ref()
                    .and_then(value_as_vec3)
                    .unwrap_or([0.0; 3]);
                let scale = op.scale.unwrap_or(100.0) as f32;
                let threshold = op.threshold.unwrap_or(1000.0) as f32;
                (idx, origin, scale, threshold)
            });

        let rope_renderer = config.renderer.iter().find(|r| {
            r.get("name")
                .and_then(|n| n.as_str())
                .map(|n| n == "rope" || n == "ropetrail")
                .unwrap_or(false)
        });
        let rope_mode = rope_renderer.is_some();
        let rope_subdivision = rope_renderer
            .and_then(|r| r.get("subdivision"))
            .and_then(|v| v.as_u64())
            .unwrap_or(1)
            .max(1) as usize;
        // Rope UV controls (ObjectParser::parseParticleRenderer defaults:
        // uvscale 1, uvscrolling false, uvsmoothing true). `uvscale` divides
        // the rope's UV length so values > 1 tile the texture along it.
        let json_bool = |v: &serde_json::Value| v.as_bool().or_else(|| v.as_u64().map(|n| n != 0));
        let rope_uv_scale = rope_renderer
            .and_then(|r| r.get("uvscale"))
            .and_then(|v| v.as_f64())
            .map(|v| v as f32)
            .filter(|v| *v > 0.0)
            .unwrap_or(1.0);
        let rope_uv_scrolling = rope_renderer
            .and_then(|r| r.get("uvscrolling"))
            .and_then(json_bool)
            .unwrap_or(false);
        let rope_uv_smoothing = rope_renderer
            .and_then(|r| r.get("uvsmoothing"))
            .and_then(json_bool)
            .unwrap_or(true);

        // `spritetrail` renderer: each particle draws as a quad stretched
        // along its velocity by `clamp(|v| * length, minlength, maxlength)`
        // (genericparticle.vert's TRAILRENDERER `ComputeParticleTrailTangents`;
        // parser defaults from ObjectParser::parseParticleRenderer).
        let sprite_trail = config
            .renderer
            .iter()
            .find(|r| r.get("name").and_then(|n| n.as_str()) == Some("spritetrail"))
            .map(|r| {
                let f = |key: &str, default: f32| -> f32 {
                    r.get(key)
                        .and_then(|v| v.as_f64())
                        .map(|v| v as f32)
                        .unwrap_or(default)
                };
                SpriteTrailParams {
                    length: f("length", 0.05),
                    max_length: f("maxlength", 10.0),
                    min_length: f("minlength", 0.0),
                }
            });

        // `angularvelocityrandom`'s min/max are vec3 (reference default
        // `(0,0,-5)`/`(0,0,5)`) — we only model the z-axis (screen-space)
        // component, matching how `rotation.z` is the only axis our flat
        // 2D renderer can express.
        let (angular_velocity_min, angular_velocity_max) = config
            .initializer
            .iter()
            .find(|init| init.name == "angularvelocityrandom")
            .map(|init| {
                let min = init
                    .min
                    .as_ref()
                    .and_then(value_as_vec3)
                    .unwrap_or([0.0, 0.0, -5.0]);
                let max = init
                    .max
                    .as_ref()
                    .and_then(value_as_vec3)
                    .unwrap_or([0.0, 0.0, 5.0]);
                (min[2], max[2])
            })
            .unwrap_or((0.0, 0.0));

        let angular_op = config
            .operator
            .iter()
            .find(|op| op.name == "angularmovement");
        let angular_drag = angular_op.and_then(|op| op.drag).unwrap_or(0.0) as f32;
        let angular_force = angular_op
            .and_then(|op| op.force.as_ref())
            .and_then(value_as_vec3)
            .map(|v| v[2])
            .unwrap_or(0.0);

        // `rotationrandom`: z component of the random vec3 (our renderer is
        // flat 2D, same reduction as `angularvelocityrandom`). Reference
        // defaults: min (0,0,0), max (0,0,2pi).
        let rotation_range = config
            .initializer
            .iter()
            .find(|init| init.name == "rotationrandom")
            .map(|init| {
                let min = init
                    .min
                    .as_ref()
                    .and_then(value_as_vec3)
                    .unwrap_or([0.0; 3]);
                let max = init.max.as_ref().and_then(value_as_vec3).unwrap_or([
                    0.0,
                    0.0,
                    std::f32::consts::TAU,
                ]);
                (min[2], max[2])
            });

        let turbulent_velocity = config
            .initializer
            .iter()
            .find(|init| init.name == "turbulentvelocityrandom")
            .map(|init| TurbulentVelocityParams {
                speed_min: init
                    .speedmin
                    .as_ref()
                    .and_then(value_as_f32)
                    .unwrap_or(100.0),
                speed_max: init
                    .speedmax
                    .as_ref()
                    .and_then(value_as_f32)
                    .unwrap_or(250.0),
                offset: init.offset.unwrap_or(0.0) as f32,
                scale: init.scale.unwrap_or(1.0) as f32,
                forward: init
                    .forward
                    .as_ref()
                    .and_then(value_as_vec3)
                    .unwrap_or([0.0, 1.0, 0.0]),
                timescale: init.timescale.unwrap_or(1.0) as f32,
                phase_min: init.phasemin.unwrap_or(0.0) as f32,
                phase_max: init.phasemax.unwrap_or(0.1) as f32,
                right: init
                    .right
                    .as_ref()
                    .and_then(value_as_vec3)
                    .unwrap_or([0.0, 0.0, 1.0]),
            });

        // `turbulence`: phase and speed are randomized once per operator
        // instance (the reference draws them in the factory, not per
        // particle or per frame).
        let turbulence = config
            .operator
            .iter()
            .find(|op| op.name == "turbulence")
            .map(|op| {
                let phase_min = op.phasemin.unwrap_or(0.0) as f32;
                let phase_max = op.phasemax.unwrap_or(0.0) as f32;
                let speed_min = op.speedmin.unwrap_or(500.0) as f32;
                let speed_max = op.speedmax.unwrap_or(1000.0) as f32;
                TurbulenceParams {
                    noise_scale: op.scale.unwrap_or(0.005) as f32 * 2.0,
                    timescale: op.timescale.unwrap_or(0.01) as f32,
                    mask: op
                        .mask
                        .as_ref()
                        .and_then(value_as_vec3)
                        .unwrap_or([1.0, 1.0, 0.0]),
                    phase: phase_min + fastrand::f32() * (phase_max - phase_min).max(0.0),
                    speed: speed_min + fastrand::f32() * (speed_max - speed_min).max(0.0),
                }
            });

        let map_sequence = config
            .initializer
            .iter()
            .find(|init| init.name == "mapsequencearoundcontrolpoint")
            .map(|init| MapSequenceParams {
                control_point: init.controlpoint.unwrap_or(0).max(0) as usize,
                count: (init.count.unwrap_or(1.0) as u32).max(1),
                speed_min: init
                    .speedmin
                    .as_ref()
                    .and_then(value_as_vec3)
                    .unwrap_or([0.0; 3]),
                speed_max: init
                    .speedmax
                    .as_ref()
                    .and_then(value_as_vec3)
                    .unwrap_or([100.0; 3]),
            });

        let map_seq_between = config
            .initializer
            .iter()
            .find(|init| init.name == "mapsequencebetweencontrolpoints")
            .map(|init| MapSequenceBetweenParams {
                count: (init.count.unwrap_or(2.0) as u32).max(2),
                mirror: init.limitbehavior.as_deref() == Some("mirror"),
            });

        let vortex = config
            .operator
            .iter()
            .find(|op| op.name == "vortex" || op.name == "vortex_v2")
            .map(|op| {
                let flags = op.flags.as_ref().and_then(|v| v.as_i64()).unwrap_or(0);
                VortexParams {
                    control_point: op.controlpoint.unwrap_or(0),
                    infinite_axis: flags & 1 != 0,
                    maintain_distance: flags & 2 != 0,
                    ring_shape: flags & 4 != 0,
                    axis: op
                        .axis
                        .as_ref()
                        .and_then(value_as_vec3)
                        .unwrap_or([0.0, 0.0, 1.0]),
                    offset: op
                        .offset
                        .as_ref()
                        .and_then(value_as_vec3)
                        .unwrap_or([0.0; 3]),
                    distance_inner: op.distanceinner.unwrap_or(500.0) as f32,
                    distance_outer: op.distanceouter.unwrap_or(650.0) as f32,
                    speed_inner: op.speedinner.unwrap_or(2500.0) as f32,
                    speed_outer: op.speedouter.unwrap_or(0.0) as f32,
                    center_force: op.centerforce.unwrap_or(1.0) as f32,
                    ring_radius: op.ringradius.unwrap_or(300.0) as f32,
                    ring_width: op.ringwidth.unwrap_or(50.0) as f32,
                    ring_pull_distance: op.ringpulldistance.unwrap_or(50.0) as f32,
                    ring_pull_force: op.ringpullforce.unwrap_or(10.0) as f32,
                }
            });

        let remap: Vec<RemapParams> = config
            .operator
            .iter()
            .filter(|op| op.name == "remapvalue")
            .map(|op| {
                let range = |v: &Option<serde_json::Value>, d: f32| {
                    v.as_ref()
                        .and_then(value_as_vec3)
                        .or_else(|| v.as_ref().and_then(value_as_f32).map(|s| [s, s, s]))
                        .unwrap_or([d, d, d])
                };
                let scalar = |v: &Option<serde_json::Value>, d: f32| {
                    v.as_ref().and_then(value_as_f32).unwrap_or(d)
                };
                let input = RemapField::parse(op.input.as_deref());
                // No `transformfunction` key at all: noise-generator default
                // when there's no `input` field either (the classic,
                // real-content-validated case, `rain_screen.json`);
                // passthrough when an `input` field IS given, so setting
                // `input` alone uses the field's value directly instead of
                // silently overriding it with noise. Confirmed against the
                // retail binary's own `alphafade`-preset builder
                // (Follow-up (x)), which pairs `input=lifetimefraction` with
                // an *explicit* `transformfunction=none` — this default
                // reproduces that pairing even when a wallpaper's own config
                // omits the (redundant) explicit "none".
                let transform = match op.transformfunction.as_deref() {
                    Some(s) => RemapTransform::parse(s),
                    None if input == RemapField::None => RemapTransform::Simplex,
                    None => RemapTransform::None,
                };
                RemapParams {
                    input,
                    input_component: RemapComponent::parse(op.inputcomponent.as_deref()),
                    output: RemapOutput::parse(op.output.as_deref()),
                    output_component: RemapComponent::parse(op.outputcomponent.as_deref()),
                    operation: RemapOp::parse(op.operation.as_deref()),
                    transform,
                    in_min: scalar(&op.inputrangemin, 0.0),
                    in_max: scalar(&op.inputrangemax, 1.0),
                    out_min: range(&op.outputrangemin, 0.0),
                    out_max: range(&op.outputrangemax, 0.0),
                    input_scale: op.transforminputscale.unwrap_or(1.0) as f32,
                }
            })
            .collect();

        // `collisionsphere`/`collisionbox`/`collisionbounds`/`collisionquad`/
        // `collisionplane` — see the `Operator`/`CollisionParams` field docs
        // for the "no C++ reference" caveat. Anchored at `controlpoint` (+
        // `offset`), same static-resolve-once convention as
        // `controlpointattract`/`vortex`.
        const COLLISION_NAMES: [&str; 5] = [
            "collisionsphere",
            "collisionbox",
            "collisionbounds",
            "collisionquad",
            "collisionplane",
        ];
        let collisions: Vec<CollisionParams> = config
            .operator
            .iter()
            .filter(|op| COLLISION_NAMES.contains(&op.name.as_str()))
            .map(|op| {
                let cp_idx = op.controlpoint.unwrap_or(0).max(0) as usize;
                let cp = control_points.get(cp_idx).copied().unwrap_or([0.0; 3]);
                let offset = op
                    .offset
                    .as_ref()
                    .and_then(value_as_vec3)
                    .unwrap_or([0.0; 3]);
                let center = [cp[0] + offset[0], cp[1] + offset[1]];
                let shape = match op.name.as_str() {
                    "collisionsphere" => CollisionShape::Sphere {
                        radius: op.radius.unwrap_or(100.0) as f32,
                    },
                    "collisionplane" => {
                        // Default normal (0,1) is "up" in our y-down space
                        // (screen-up); `angles.z` rotates it, same convention
                        // `mapsequencearoundcontrolpoint`-adjacent offset
                        // rotation already uses elsewhere in this module.
                        let az = op
                            .angles
                            .as_ref()
                            .and_then(value_as_vec3)
                            .map(|a| a[2])
                            .unwrap_or(0.0);
                        let (s, c) = az.sin_cos();
                        CollisionShape::Plane {
                            normal: [-s, c],
                        }
                    }
                    // collisionbox/collisionbounds/collisionquad
                    _ => {
                        let size = op
                            .size
                            .as_ref()
                            .and_then(value_as_vec3)
                            .unwrap_or([200.0, 200.0, 0.0]);
                        CollisionShape::Box {
                            half_extent: [size[0] * 0.5, size[1] * 0.5],
                        }
                    }
                };
                CollisionParams {
                    shape,
                    center,
                    behavior: CollisionBehavior::parse(op.collisionbehavior.as_deref()),
                    bounce_factor: op.bouncefactor.unwrap_or(1.0) as f32,
                }
            })
            .collect();

        // `boids`: standard separation/alignment/cohesion flocking.
        let boids = config
            .operator
            .iter()
            .find(|op| op.name == "boids")
            .map(|op| BoidsParams {
                separation_threshold: op.separationthreshold.unwrap_or(50.0) as f32,
                neighbor_threshold: op.neighborthreshold.unwrap_or(150.0) as f32,
                max_speed: op.maxspeed.unwrap_or(500.0) as f32,
                separation_factor: op.separationfactor.unwrap_or(1.0) as f32,
                alignment_factor: op.alignmentfactor.unwrap_or(1.0) as f32,
                cohesion_factor: op.cohesionfactor.unwrap_or(1.0) as f32,
                variable_strength: op.variablestrength.unwrap_or(0.0) != 0.0,
            });

        Self {
            particles: Vec::with_capacity(max_count),
            emitters,
            max_count,
            life_min,
            life_max,
            size_min,
            size_max,
            life_exp,
            size_exp,
            alpha_min,
            alpha_max,
            velocity_min,
            velocity_max,
            color_min,
            color_max,
            gravity,
            movement_drag,
            alphafade,
            sizechange,
            alphachange,
            colorchange,
            oscillate_alpha,
            oscillate_size,
            oscillate_position,
            alpha_mult: overrides.and_then(|o| o.alpha).unwrap_or(1.0) as f32,
            lifetime_mult: overrides.and_then(|o| o.lifetime).unwrap_or(1.0) as f32,
            colorn_mult: overrides
                .and_then(|o| o.colorn.as_ref())
                .and_then(value_as_vec3_pub)
                .unwrap_or([1.0, 1.0, 1.0]),
            size_mult: overrides.and_then(|o| o.size).unwrap_or(1.0) as f32,
            speed_mult: overrides.and_then(|o| o.speed).unwrap_or(1.0) as f32,
            rate_mult: overrides.and_then(|o| o.rate).unwrap_or(1.0) as f32,
            color_override,
            control_points,
            control_point_attract,
            rope_mode,
            rope_subdivision,
            rope_uv_scale,
            rope_uv_scrolling,
            rope_uv_smoothing,
            shell_3d: config.flags.unwrap_or(0) & 4 != 0,
            next_particle_id: 0,
            children: Vec::new(),
            angular_velocity_min,
            angular_velocity_max,
            angular_force,
            angular_drag,
            sprite_frame_count: 0,
            sprite_duration: 0.0,
            animation_mode: AnimationMode::from_config(config.animationmode.as_deref()),
            sequence_multiplier: config
                .sequencemultiplier
                .map(|v| v as f32)
                .filter(|v| *v > 0.0)
                .unwrap_or(1.0),
            rotation_range,
            sprite_trail,
            map_sequence,
            map_sequence_index: 0,
            map_seq_between,
            map_seq_between_index: 0,
            turbulent_velocity,
            turbulence,
            vortex,
            remap,
            collisions,
            boids,
            time: 0.0,
            start_time: config.starttime.unwrap_or(0.0).max(0.0) as f32,
            scene_force: [0.0, 0.0],
            audio_level: 0.0,
        }
    }

    /// Apply the scene-global constant force (`general.gravity*`/`wind*`).
    pub fn set_scene_force(&mut self, force: [f32; 2]) {
        self.scene_force = force;
    }

    /// Set the current audio loudness (~[0,1]) driving audio-reactive emitters.
    pub fn set_audio_level(&mut self, level: f32) {
        self.audio_level = level.clamp(0.0, 4.0);
    }

    /// Tell the system its resolved sprite's frame count/loop duration, so
    /// `step()` can advance each particle's `frame` — called once by the
    /// caller right after resolving the sprite (this module itself never
    /// loads textures). A no-op frame count of 0/1 leaves every particle's
    /// `frame` at its spawn sentinel, matching `render_onto`'s "always frame
    /// 0" behavior for non-animated sprites.
    pub fn set_sprite_frames(&mut self, frame_count: usize, duration: f32) {
        self.sprite_frame_count = frame_count;
        self.sprite_duration = duration;
    }

    pub fn step(&mut self, dt: f32) {
        self.time += dt;
        // `boids`: needs every particle's position/velocity as of the START
        // of this frame (not yet mutated by this frame's own movement), so
        // it's computed as a separate read-only pass before the main
        // per-particle loop mutates anything.
        let boids_accel = self
            .boids
            .as_ref()
            .map(|b| compute_boids_accel(&self.particles, b));
        for (particle_index, p) in self.particles.iter_mut().enumerate() {
            // `movement` operator (CParticle.cpp createMovementOperator):
            // position integrates the CURRENT velocity first, and only then
            // do this frame's forces modify velocity (for the next frame) —
            // gravity scaled by the instance speed override, then drag decay
            // clamped so `drag*dt > 1` stops the particle instead of
            // reversing it.
            p.x += p.vx * dt;
            p.y += p.vy * dt;
            p.vx += (self.gravity[0] * self.speed_mult + self.scene_force[0]) * dt;
            p.vy += (self.gravity[1] * self.speed_mult + self.scene_force[1]) * dt;
            let drag_factor = (1.0 - self.movement_drag * dt).max(0.0);
            p.vx *= drag_factor;
            p.vy *= drag_factor;

            // `turbulence` operator (CParticle.cpp createTurbulenceOperator):
            // normalized curl-noise direction, sampled at a time-scrolled,
            // scaled particle position, masked per axis, added to velocity.
            if let Some(t) = self.turbulence {
                if t.speed > 0.0001 {
                    let noise_pos = [
                        (p.x + t.phase + t.timescale * self.time) * t.noise_scale,
                        p.y * t.noise_scale,
                        0.0,
                    ];
                    let curl = crate::engine::noise::curl_noise(noise_pos);
                    let len = vec3_length(curl);
                    if len > 0.0001 {
                        let s = t.speed / len * dt * self.speed_mult;
                        p.vx += curl[0] * t.mask[0] * s;
                        p.vy += curl[1] * t.mask[1] * s;
                    }
                }
            }

            // `vortex` operator (CParticle.cpp createVortexOperator).
            if let Some(v) = self.vortex {
                let center_base = self
                    .control_points
                    .get(v.control_point.max(0) as usize)
                    .copied()
                    .unwrap_or([0.0; 3]);
                let center = [
                    center_base[0] + v.offset[0],
                    center_base[1] + v.offset[1],
                    center_base[2] + v.offset[2],
                ];
                // The reference computes tangents in its y-up sim space and
                // projects y-up; our space is y-down (a reflection, det -1),
                // under which cross products flip sign: mirror the axis's y
                // and negate the resulting tangent to reproduce the same
                // on-screen spin direction.
                let axis = vec3_normalize_or([v.axis[0], -v.axis[1], v.axis[2]], [0.0, 0.0, 1.0]);
                let to_particle = [p.x - center[0], p.y - center[1], -center[2]];

                // Infinite axis: cylinder shape (project out the axis
                // component); else full 3D distance (sphere shape).
                let radial = if v.infinite_axis {
                    let axial = vec3_dot(to_particle, axis);
                    [
                        to_particle[0] - axis[0] * axial,
                        to_particle[1] - axis[1] * axial,
                        to_particle[2] - axis[2] * axial,
                    ]
                } else {
                    to_particle
                };
                let distance = vec3_length(radial);

                let tangent = vec3_cross(axis, radial);
                let tangent_len = vec3_length(tangent);
                if tangent_len > 0.001 {
                    // Negated: see the reflection note on `axis` above.
                    let tangent = [
                        -tangent[0] / tangent_len,
                        -tangent[1] / tangent_len,
                        -tangent[2] / tangent_len,
                    ];

                    let mut speed = 0.0;
                    let mut radial_force = [0.0f32; 3];
                    if v.ring_shape {
                        let ring_inner = v.ring_radius - v.ring_width * 0.5;
                        let ring_outer = v.ring_radius + v.ring_width * 0.5;
                        if distance < ring_inner {
                            // Hollow center: no spin.
                        } else if distance <= ring_outer {
                            let t = (distance - ring_inner) / v.ring_width;
                            speed = v.speed_inner + (v.speed_outer - v.speed_inner) * t;
                        } else if distance <= ring_outer + v.ring_pull_distance {
                            let pull_t = (distance - ring_outer) / v.ring_pull_distance;
                            speed = v.speed_outer * (1.0 - pull_t);
                            if distance > 0.001 {
                                let toward = vec3_normalize_or(radial, [0.0; 3]);
                                let f = -v.ring_pull_force * pull_t;
                                radial_force = [toward[0] * f, toward[1] * f, toward[2] * f];
                            }
                        }
                    } else {
                        let dis_mid = v.distance_outer - v.distance_inner + 0.1;
                        speed = if dis_mid < 0.0 || distance < v.distance_inner {
                            v.speed_inner
                        } else if distance > v.distance_outer {
                            v.speed_outer
                        } else {
                            let t = (distance - v.distance_inner) / dis_mid;
                            v.speed_inner + (v.speed_outer - v.speed_inner) * t
                        };
                    }

                    let k = dt * self.speed_mult;
                    p.vx += (tangent[0] * speed + radial_force[0]) * k;
                    p.vy += (tangent[1] * speed + radial_force[1]) * k;
                    if v.maintain_distance && distance > 0.001 {
                        let toward = vec3_normalize_or(radial, [0.0; 3]);
                        p.vx -= toward[0] * v.center_force * k;
                        p.vy -= toward[1] * v.center_force * k;
                    }
                }
            }

            // `boids` operator: apply this particle's precomputed
            // separation/alignment/cohesion acceleration, then cap speed.
            if let (Some(accel), Some(b)) = (&boids_accel, &self.boids) {
                let [ax, ay] = accel[particle_index];
                let strength = if b.variable_strength {
                    // Deterministic per-particle jitter in [0.5, 1.5] from a
                    // cheap integer hash of the particle's stable id — avoids
                    // adding a new per-particle RNG-state field for a value
                    // that only needs to be stable, not independently random
                    // each frame.
                    let h = p.id.wrapping_mul(2654435761);
                    0.5 + ((h >> 8) & 0xFFFF) as f32 / 65535.0
                } else {
                    1.0
                };
                let k = dt * self.speed_mult * strength;
                p.vx += ax * k;
                p.vy += ay * k;
                if b.max_speed > 0.0 {
                    let speed2 = p.vx * p.vx + p.vy * p.vy;
                    if speed2 > b.max_speed * b.max_speed {
                        let s = b.max_speed / speed2.sqrt();
                        p.vx *= s;
                        p.vy *= s;
                    }
                }
            }

            // `remapvalue` targeting `velocity`/`speed`: runs here (before
            // this frame's velocity is used for next frame's position
            // integration), matching the original, real-content-validated
            // timing exactly. Fields targeting other outputs run later,
            // after alpha/size/color are computed (see below) so
            // `remapvalue` can override them.
            let lifetime_pos_pre = if p.max_life > 0.0 {
                ((p.max_life - p.life) / p.max_life).clamp(0.0, 1.0)
            } else {
                1.0
            };
            for r in &self.remap {
                match r.output {
                    RemapOutput::Velocity => {
                        let [vx, vy, _] =
                            r.apply_vector(p, self.time, lifetime_pos_pre, [p.vx, p.vy, 0.0], 2);
                        p.vx = vx;
                        p.vy = vy;
                    }
                    RemapOutput::Speed => {
                        let t = r.sample01(0.0, p, self.time, lifetime_pos_pre, p.max_life);
                        let target = r.out_min[0] + (r.out_max[0] - r.out_min[0]) * t;
                        let mag = (p.vx * p.vx + p.vy * p.vy).sqrt();
                        let sp = r.operation.combine(mag, target);
                        if mag > 1e-4 {
                            let k = sp / mag;
                            p.vx *= k;
                            p.vy *= k;
                        }
                    }
                    _ => {}
                }
            }

            // Position-oscillate integrates a velocity (derivative of
            // scale*cos(w*t+phase)) into position each frame, matching
            // CParticle.cpp's createOscillatePositionOperator exactly.
            if let Some(osc) = &p.osc_pos {
                let age = p.max_life - p.life;
                for axis in 0..2 {
                    let w = osc.frequency[axis];
                    let d = -osc.scale[axis] * w * (w * age + osc.phase[axis]).sin() * dt;
                    if axis == 0 {
                        p.x += d;
                    } else {
                        p.y += d;
                    }
                }
            }

            if let Some((idx, origin, scale, threshold)) = self.control_point_attract {
                if let Some(cp) = self.control_points.get(idx) {
                    let center = [cp[0] + origin[0], cp[1] + origin[1]];
                    let to_center = [center[0] - p.x, center[1] - p.y];
                    let distance =
                        (to_center[0] * to_center[0] + to_center[1] * to_center[1]).sqrt();
                    let radius = threshold / 2.0;
                    if distance > 0.001 && distance < radius {
                        let force = scale * dt * self.speed_mult;
                        p.vx += (to_center[0] / distance) * force;
                        p.vy += (to_center[1] / distance) * force;
                    }
                }
            }

            // `collisionsphere`/`collisionbox`/`collisionbounds`/
            // `collisionquad`/`collisionplane`: checked last among the
            // motion-affecting operators, against this frame's final
            // position, so a collider reacts to where the particle actually
            // ended up (matching how a physics response is conventionally
            // the last word on a frame's motion).
            for collider in &self.collisions {
                resolve_collision(p, collider);
            }

            p.life -= dt;

            // Reference order (CParticle.cpp createAngularMovementOperator):
            // integrate rotation from the *current* angular velocity first,
            // then accelerate angular velocity, then decay it by drag.
            p.rotation += p.angular_velocity * dt;
            p.angular_velocity += self.angular_force * dt;
            let drag_factor = (1.0 - self.angular_drag * dt).max(0.0);
            p.angular_velocity *= drag_factor;
            p.rotation = wrap_angle(p.rotation);

            let age = p.max_life - p.life;
            let lifetime_pos = if p.max_life > 0.0 {
                (age / p.max_life).clamp(0.0, 1.0)
            } else {
                1.0
            };

            // Sprite-sheet frame advance (CParticle.cpp's "Update animation
            // frames"): only meaningful once the caller has told us the
            // resolved sprite actually has more than one frame.
            if self.sprite_frame_count > 1 {
                let frame_count = self.sprite_frame_count as f32;
                match self.animation_mode {
                    AnimationMode::RandomFrame => {
                        if p.frame < 0.0 {
                            p.frame = (fastrand::f32() * frame_count)
                                .floor()
                                .min(frame_count - 1.0);
                        }
                    }
                    AnimationMode::Once => {
                        p.frame = (lifetime_pos * frame_count * self.sequence_multiplier)
                            .min(frame_count - 1.0);
                    }
                    AnimationMode::Loop => {
                        p.frame = if self.sprite_duration > 0.0 {
                            let time_in_cycle =
                                (age * self.sequence_multiplier) % self.sprite_duration;
                            let cycle_pos = time_in_cycle / self.sprite_duration;
                            (cycle_pos * frame_count) % frame_count
                        } else {
                            (lifetime_pos * frame_count * self.sequence_multiplier) % frame_count
                        };
                    }
                }
            }

            // Alpha: alphafade takes precedence (its very common fade-in/hold/
            // fade-out shape), then a generic alphachange ramp, else fall back
            // to a plain linear death-fade so particles don't hard-cut when a
            // config has no alpha-affecting operator at all.
            let mut alpha = if let Some((fade_in, fade_out)) = self.alphafade {
                let fade = if lifetime_pos <= fade_in {
                    fade_value(lifetime_pos, 0.0, fade_in, 0.0, 1.0)
                } else if lifetime_pos > fade_out {
                    1.0 - fade_value(lifetime_pos, fade_out, 1.0, 0.0, 1.0)
                } else {
                    1.0
                };
                p.initial_alpha * fade
            } else if let Some((start, end, sv, ev)) = self.alphachange {
                p.initial_alpha * fade_value(lifetime_pos, start, end, sv, ev)
            } else {
                p.initial_alpha * (p.life / p.max_life).max(0.0)
            };

            let mut size = if let Some((start, end, sv, ev)) = self.sizechange {
                p.initial_size * fade_value(lifetime_pos, start, end, sv, ev)
            } else {
                p.size
            };

            if let Some(osc) = &p.osc_alpha {
                let cos_val = ((osc.frequency * age + osc.phase).cos() + 1.0) * 0.5;
                let mult = self
                    .oscillate_alpha
                    .map(|o| o.scale_min + (o.scale_max - o.scale_min) * cos_val)
                    .unwrap_or(1.0);
                alpha *= mult;
            }
            if let Some(osc) = &p.osc_size {
                let cos_val = ((osc.frequency * age + osc.phase).cos() + 1.0) * 0.5;
                let mult = self
                    .oscillate_size
                    .map(|o| o.scale_min + (o.scale_max - o.scale_min) * cos_val)
                    .unwrap_or(1.0);
                size *= mult;
            }

            if let Some((start, end, start_color, end_color)) = self.colorchange {
                let mult = [
                    fade_value(lifetime_pos, start, end, start_color[0], end_color[0]),
                    fade_value(lifetime_pos, start, end, start_color[1], end_color[1]),
                    fade_value(lifetime_pos, start, end, start_color[2], end_color[2]),
                ];
                p.color = [
                    (p.initial_color[0] * mult[0] * 255.0).clamp(0.0, 255.0) as u8,
                    (p.initial_color[1] * mult[1] * 255.0).clamp(0.0, 255.0) as u8,
                    (p.initial_color[2] * mult[2] * 255.0).clamp(0.0, 255.0) as u8,
                ];
            }

            // `remapvalue` targeting anything other than velocity/speed
            // (handled above, before position integration needs it): runs
            // last, after alphafade/sizechange/colorchange/oscillate, so an
            // author who explicitly attaches remapvalue to one of these
            // fields gets it as the final, overriding value.
            for r in &self.remap {
                match r.output {
                    RemapOutput::Velocity | RemapOutput::Speed => {}
                    RemapOutput::Opacity => {
                        let t = r.sample01(0.0, p, self.time, lifetime_pos, p.max_life);
                        let remapped = r.out_min[0] + (r.out_max[0] - r.out_min[0]) * t;
                        alpha = r.operation.combine(alpha, remapped);
                    }
                    RemapOutput::Size => {
                        let t = r.sample01(0.0, p, self.time, lifetime_pos, p.max_life);
                        let remapped = r.out_min[0] + (r.out_max[0] - r.out_min[0]) * t;
                        size = r.operation.combine(size, remapped);
                    }
                    RemapOutput::Rotation => {
                        let t = r.sample01(0.0, p, self.time, lifetime_pos, p.max_life);
                        let remapped = r.out_min[0] + (r.out_max[0] - r.out_min[0]) * t;
                        p.rotation = r.operation.combine(p.rotation, remapped);
                    }
                    RemapOutput::AngularSpeed => {
                        let t = r.sample01(0.0, p, self.time, lifetime_pos, p.max_life);
                        let remapped = r.out_min[0] + (r.out_max[0] - r.out_min[0]) * t;
                        p.angular_velocity = r.operation.combine(p.angular_velocity, remapped);
                    }
                    RemapOutput::Color => {
                        let current = [
                            p.color[0] as f32 / 255.0,
                            p.color[1] as f32 / 255.0,
                            p.color[2] as f32 / 255.0,
                        ];
                        let out = r.apply_vector(p, self.time, lifetime_pos, current, 3);
                        p.color = [
                            (out[0] * 255.0).clamp(0.0, 255.0) as u8,
                            (out[1] * 255.0).clamp(0.0, 255.0) as u8,
                            (out[2] * 255.0).clamp(0.0, 255.0) as u8,
                        ];
                    }
                    RemapOutput::Position => {
                        let [x, y, _] = r.apply_vector(p, self.time, lifetime_pos, [p.x, p.y, 0.0], 2);
                        p.x = x;
                        p.y = y;
                    }
                }
            }

            // `self.alpha_mult` (the instance-override multiplier) is already
            // baked into `p.initial_alpha` at spawn, matching the
            // reference's `p.alpha = random(min,max) * override` — applying
            // it again here would square it whenever an override isn't 1.0.
            p.alpha = alpha;
            p.size = size;
        }

        // Child-system event capture: deaths before the retain, spawns as
        // they're pushed below (only bothering when children are attached).
        let mut died: Vec<(u64, [f32; 2])> = Vec::new();
        let mut spawned: Vec<(u64, [f32; 2])> = Vec::new();
        if !self.children.is_empty() {
            died.extend(
                self.particles
                    .iter()
                    .filter(|p| p.life <= 0.0)
                    .map(|p| (p.id, [p.x, p.y])),
            );
        }
        self.particles.retain(|p| p.life > 0.0);

        // Hold off emission until `starttime` — existing particles still
        // simulate and die, the burst just ramps in from zero at start_time.
        let emitting = self.time >= self.start_time;
        let since_start = self.time - self.start_time;
        for emitter in &mut self.emitters {
            // Per-emitter `duration` cutoff (after start_time).
            let expired = emitter.duration.is_some_and(|d| since_start > d);
            if !emitting || expired {
                continue;
            }
            // One-shot burst: seed the accumulator with the full count once, so
            // the spawn loop below emits it in this frame (capped by max_count).
            if !emitter.burst_done {
                emitter.accumulator += emitter.burst as f32;
                emitter.burst_done = true;
            }
            // Audio-reactive emitters scale their rate with loudness (a small
            // baseline keeps a trickle when quiet).
            let audio_gain = if emitter.audio_reactive {
                0.15 + self.audio_level
            } else {
                1.0
            };
            emitter.accumulator += emitter.rate * self.rate_mult * audio_gain * dt;
            while emitter.accumulator >= 1.0 && self.particles.len() < self.max_count {
                emitter.accumulator -= 1.0;
                // `createLifetimeRandomInitializer`: `p.lifetime =
                // random(min, max) * lifetimeOverride`. `exponent` biases the
                // uniform sample toward one end before the lerp.
                let life = (self.life_min
                    + fastrand::f32().powf(self.life_exp)
                        * (self.life_max - self.life_min).max(0.0))
                    * self.lifetime_mult;
                // Reference halves the sizerandom range (CParticle.cpp: `size = (min +
                // t*(max-min)) * override / 2.0`) — the initializer's value is a
                // diameter, not a radius.
                let size = (self.size_min
                    + fastrand::f32().powf(self.size_exp)
                        * (self.size_max - self.size_min).max(0.0))
                    * self.size_mult
                    / 2.0;

                let ox = emitter.origin[0];
                let oy = emitter.origin[1];

                // Spawn offset — CParticle.cpp's create{Box,Sphere}Emitter:
                // a random point WITHIN the emitter volume, scaled per-axis
                // by `directions` (the emitter never encodes velocity in
                // its distances; rainperspective's 1024 radius is a spawn
                // band, which our old model collapsed into a ±15px column
                // at the emitter origin while misreading the radius as a
                // fallback speed).
                let mut offset = if emitter.is_sphere {
                    let (rmin, rmax) = (emitter.distance_min[0], emitter.distance_max[0]);
                    if self.shell_3d {
                        // flags & 4 (perspective): 3D spherical shell,
                        // cbrt-uniform radius; our 2D sim keeps xy.
                        let theta = fastrand::f32() * std::f32::consts::TAU;
                        let cos_t = fastrand::f32() * 2.0 - 1.0;
                        let sin_t = (1.0 - cos_t * cos_t).sqrt();
                        let r =
                            (rmin.powi(3) + fastrand::f32() * (rmax.powi(3) - rmin.powi(3))).cbrt();
                        [sin_t * theta.cos() * r, sin_t * theta.sin() * r]
                    } else {
                        // 2D disk/annulus, sqrt-uniform for even area density.
                        let angle = fastrand::f32() * std::f32::consts::TAU;
                        let r2 = rmin * rmin + fastrand::f32() * (rmax * rmax - rmin * rmin);
                        let r = r2.max(0.0).sqrt();
                        [angle.cos() * r, angle.sin() * r]
                    }
                } else {
                    // Box: per-axis distance in [min, max], random sign flip.
                    let mut o = [0.0f32; 2];
                    for (axis, out) in o.iter_mut().enumerate() {
                        let d = emitter.distance_min[axis]
                            + fastrand::f32()
                                * (emitter.distance_max[axis] - emitter.distance_min[axis]);
                        *out = if fastrand::f32() < 0.5 { -d } else { d };
                    }
                    o
                };
                offset[0] *= emitter.directions[0];
                offset[1] *= emitter.directions[1];
                // `sign`: 1 forces positive, -1 negative, 0 leaves both.
                for axis in 0..2 {
                    if emitter.sign[axis] >= 1.0 {
                        offset[axis] = offset[axis].abs();
                    } else if emitter.sign[axis] <= -1.0 {
                        offset[axis] = -offset[axis].abs();
                    }
                }
                let mut spawn_x = ox + offset[0];
                let mut spawn_y = oy + offset[1];

                // Velocity: velocityrandom initializer first; else the
                // emitter's own optional radial speed (`speedmin`/`speedmax`
                // fields, pointing outward from the offset); else zero —
                // "Emitter does not set velocity - initializers handle that".
                let (mut vx, mut vy) =
                    if let (Some(min), Some(max)) = (self.velocity_min, self.velocity_max) {
                        (
                            (min[0] + fastrand::f32() * (max[0] - min[0])) * self.speed_mult,
                            (min[1] + fastrand::f32() * (max[1] - min[1])) * self.speed_mult,
                        )
                    } else if emitter.speed_max > 0.0 || emitter.speed_min != 0.0 {
                        let len = (offset[0] * offset[0] + offset[1] * offset[1]).sqrt();
                        let dir = if len > 0.0 {
                            [offset[0] / len, offset[1] / len]
                        } else {
                            [0.0, 1.0]
                        };
                        let speed = (emitter.speed_min
                            + fastrand::f32() * (emitter.speed_max - emitter.speed_min))
                            * self.speed_mult;
                        (dir[0] * speed, dir[1] * speed)
                    } else {
                        (0.0, 0.0)
                    };

                // `mapsequencebetweencontrolpoints`: the Nth spawn sits at
                // the Nth slot along the cp0->cp1 segment (mirror =
                // ping-pong); velocityrandom still jitters it afterwards,
                // which is what shapes the discharge preset's lightning.
                if let Some(ms) = self.map_seq_between {
                    let last = ms.count - 1;
                    let idx = self.map_seq_between_index;
                    self.map_seq_between_index = self.map_seq_between_index.wrapping_add(1);
                    let slot = if ms.mirror {
                        let period = 2 * last;
                        let k = idx % period.max(1);
                        if k <= last {
                            k
                        } else {
                            period - k
                        }
                    } else {
                        idx % ms.count
                    };
                    let t = slot as f32 / last as f32;
                    if let (Some(a), Some(b)) =
                        (self.control_points.first(), self.control_points.get(1))
                    {
                        spawn_x = a[0] + (b[0] - a[0]) * t;
                        spawn_y = a[1] + (b[1] - a[1]) * t;
                    }
                }

                // `mapsequencearoundcontrolpoint` (CParticle.cpp): spawn at
                // the control point, launching successive particles at
                // evenly-spaced angles around a circle (the initializer
                // *assigns* position/velocity, unlike the additive ones).
                if let Some(ms) = self.map_sequence {
                    let angle =
                        (self.map_sequence_index as f32 / ms.count as f32) * std::f32::consts::TAU;
                    self.map_sequence_index = (self.map_sequence_index + 1) % ms.count;
                    if let Some(cp) = self.control_points.get(ms.control_point) {
                        spawn_x = cp[0];
                        spawn_y = cp[1];
                    }
                    let sx = ms.speed_min[0]
                        + fastrand::f32() * (ms.speed_max[0] - ms.speed_min[0]).max(0.0);
                    let sy = ms.speed_min[1]
                        + fastrand::f32() * (ms.speed_max[1] - ms.speed_min[1]).max(0.0);
                    let (sin, cos) = angle.sin_cos();
                    // The reference rotates in its y-up sim space
                    // (v' = (cos*x + sin*y, -sin*x + cos*y), column-major
                    // glm::mat3) and projects y-up; our space is y-down, so
                    // the net screen-space rotation is the conjugate —
                    // otherwise the spawn sequence sweeps the circle in the
                    // mirrored direction.
                    vx = (cos * sx - sin * sy) * self.speed_mult;
                    vy = (sin * sx + cos * sy) * self.speed_mult;
                }

                // `turbulentvelocityrandom` (CParticle.cpp): curl-noise
                // spawn velocity, angle-limited around `forward`, added on
                // top of any plain `velocityrandom` contribution.
                if let Some(t) = self.turbulent_velocity {
                    let forward = vec3_normalize_or(t.forward, [0.0, 1.0, 0.0]);
                    let right = vec3_normalize_or(t.right, [1.0, 0.0, 0.0]);
                    let speed =
                        t.speed_min + fastrand::f32() * (t.speed_max - t.speed_min).max(0.0);
                    let phase =
                        t.phase_min + fastrand::f32() * (t.phase_max - t.phase_min).max(0.0);
                    let time_shift = self.time * t.timescale;
                    let sample = [
                        spawn_x * 0.1 + time_shift + phase,
                        spawn_y * 0.1 + time_shift + phase * 0.7,
                        time_shift + phase * 1.3,
                    ];
                    let curl = crate::engine::noise::curl_noise(sample);
                    let mut dir = vec3_normalize_or(curl, forward);

                    // `scale` < 2 limits how far the direction may deviate
                    // from `forward` (normalized angle, max = scale/2).
                    if t.scale < 2.0 {
                        let cos_angle = vec3_dot(dir, forward).clamp(-1.0, 1.0);
                        let angle = cos_angle.acos() / std::f32::consts::PI;
                        let max_angle = t.scale / 2.0;
                        if angle > max_angle && max_angle > 0.0001 {
                            let axis = vec3_cross(dir, forward);
                            if vec3_length(axis) > 0.0001 {
                                let axis = vec3_normalize_or(axis, [0.0, 0.0, 1.0]);
                                let rot = (angle - max_angle) * std::f32::consts::PI;
                                dir = vec3_rotate_axis(dir, axis, rot);
                            }
                        }
                    }

                    // `offset` tilts the result around `right`.
                    if t.offset.abs() > 0.0001 {
                        dir = vec3_rotate_axis(dir, right, -t.offset);
                    }

                    // 2D particles: project onto the XY plane and
                    // renormalize (the reference's `flags & 4 == 0` branch —
                    // our renderer is always orthographic/2D).
                    dir[2] = 0.0;
                    let dir = vec3_normalize_or(dir, forward);

                    vx += dir[0] * speed * self.speed_mult;
                    vy += dir[1] * speed * self.speed_mult;
                }

                // `rotationrandom`: uniform in [min.z, max.z], scaled by the
                // instance speed override (the reference multiplies rotation
                // by `speedOverride` too, odd as that reads).
                let spawn_rotation = self
                    .rotation_range
                    .map(|(min, max)| {
                        (min + fastrand::f32() * (max - min).max(0.0)) * self.speed_mult
                    })
                    .unwrap_or(0.0);

                // `createColorRandomInitializer`: `p.color = randomVec3(min,
                // max) * colorOverride` — colorn is a 0-1 per-channel tint.
                let color = self.color_override.unwrap_or_else(|| {
                    [
                        (lerp_u8(self.color_min[0], self.color_max[0]) as f32 * self.colorn_mult[0])
                            .clamp(0.0, 255.0) as u8,
                        (lerp_u8(self.color_min[1], self.color_max[1]) as f32 * self.colorn_mult[1])
                            .clamp(0.0, 255.0) as u8,
                        (lerp_u8(self.color_min[2], self.color_max[2]) as f32 * self.colorn_mult[2])
                            .clamp(0.0, 255.0) as u8,
                    ]
                });

                // `alpharandom`: CParticle.cpp's `p.alpha = random(min, max) *
                // override` — a plain multiply (unlike sizerandom, not halved).
                let spawn_alpha = (self.alpha_min
                    + fastrand::f32() * (self.alpha_max - self.alpha_min).max(0.0))
                    * self.alpha_mult;

                let particle_id = self.next_particle_id;
                self.next_particle_id += 1;
                if !self.children.is_empty() {
                    spawned.push((particle_id, [spawn_x, spawn_y]));
                }
                self.particles.push(Particle {
                    id: particle_id,
                    x: spawn_x,
                    y: spawn_y,
                    vx,
                    vy,
                    life,
                    max_life: life,
                    size,
                    alpha: spawn_alpha,
                    color,
                    initial_alpha: spawn_alpha,
                    initial_size: size,
                    initial_color: [
                        color[0] as f32 / 255.0,
                        color[1] as f32 / 255.0,
                        color[2] as f32 / 255.0,
                    ],
                    osc_alpha: self.oscillate_alpha.map(|o| o.sample()),
                    osc_size: self.oscillate_size.map(|o| o.sample()),
                    osc_pos: self.oscillate_position.map(|o| {
                        let a = o.sample();
                        let b = o.sample();
                        OscillatorPos {
                            frequency: [a.frequency, b.frequency],
                            scale: [a.scale, b.scale],
                            phase: [a.phase, b.phase],
                        }
                    }),
                    rotation: spawn_rotation,
                    angular_velocity: self.angular_velocity_min
                        + fastrand::f32() * (self.angular_velocity_max - self.angular_velocity_min),
                    frame: -1.0,
                });
            }
        }

        if !self.children.is_empty() {
            self.step_children(dt, &spawned, &died);
        }
    }

    /// Attaches a resolved child preset (see [`ChildRef`]). `static`
    /// children get their single persistent instance immediately, at the
    /// same spawn center the parent was built with.
    pub fn add_child(
        &mut self,
        config: ParticleConfig,
        sprite: Option<ParticleSprite>,
        additive: bool,
        child_ref: &ChildRef,
        spawn_center: [f32; 2],
    ) {
        let child_type = match child_ref.child_type.as_deref() {
            Some("eventfollow") => ChildType::Follow,
            Some("eventspawn") => ChildType::Spawn,
            Some("eventdeath") => ChildType::Death,
            // Absent or "static": a continuously-running sub-system.
            _ => ChildType::Static,
        };
        let mut child = ChildSystem {
            child_type,
            probability: child_ref.probability.unwrap_or(1.0) as f32,
            max_instances: child_ref.maxcount.unwrap_or(16).max(1) as usize,
            config,
            sprite,
            additive,
            instances: Vec::new(),
        };
        if child_type == ChildType::Static {
            let mut system = ParticleSystem::from_config(&child.config, spawn_center, None);
            if let Some(s) = &child.sprite {
                system.set_sprite_frames(s.frames.len(), s.duration);
            }
            child.instances.push(ChildInstance {
                parent_id: None,
                last_center: spawn_center,
                emit_until: None,
                persistent: true,
                system,
            });
        }
        self.children.push(child);
    }

    /// Spawns/updates/retires child instances for this frame's parent
    /// spawn/death events. No counterpart in the C++ reference (it ignores
    /// `children` entirely); semantics derived from real preset content.
    fn step_children(&mut self, dt: f32, spawned: &[(u64, [f32; 2])], died: &[(u64, [f32; 2])]) {
        // Burst instances stop emitting after this long — the child preset
        // itself has no end marker, and a one-shot puff is the authored
        // intent of eventspawn/eventdeath (vs. a permanent stream).
        const BURST_EMIT_SECS: f32 = 0.1;

        let mut children = std::mem::take(&mut self.children);
        for child in &mut children {
            let events: &[(u64, [f32; 2])] = match child.child_type {
                ChildType::Static => &[],
                ChildType::Follow | ChildType::Spawn => spawned,
                ChildType::Death => died,
            };
            for &(pid, pos) in events {
                if child.instances.len() >= child.max_instances {
                    break;
                }
                if fastrand::f32() > child.probability {
                    continue;
                }
                let mut system = ParticleSystem::from_config(&child.config, pos, None);
                if let Some(s) = &child.sprite {
                    system.set_sprite_frames(s.frames.len(), s.duration);
                }
                let follows = child.child_type == ChildType::Follow;
                child.instances.push(ChildInstance {
                    parent_id: follows.then_some(pid),
                    last_center: pos,
                    emit_until: (!follows).then_some(BURST_EMIT_SECS),
                    persistent: false,
                    system,
                });
            }

            for inst in &mut child.instances {
                if let Some(pid) = inst.parent_id {
                    if let Some(p) = self.particles.iter().find(|p| p.id == pid) {
                        let delta = [p.x - inst.last_center[0], p.y - inst.last_center[1]];
                        if delta != [0.0, 0.0] {
                            inst.system.shift_origin(delta);
                            inst.last_center = [p.x, p.y];
                        }
                    } else {
                        // Parent died: stop emitting, let the remainder live out.
                        inst.parent_id = None;
                        inst.system.stop_emitting();
                    }
                }
                if let Some(cutoff) = inst.emit_until {
                    if inst.system.time >= cutoff {
                        inst.system.stop_emitting();
                    }
                }
                inst.system.step(dt);
            }

            child.instances.retain(|inst| {
                inst.persistent || inst.parent_id.is_some() || !inst.system.particles.is_empty()
            });
        }
        self.children = children;
    }

    /// Moves every emitter origin and resolved control point by `delta` —
    /// how an `eventfollow` child tracks its parent particle.
    fn shift_origin(&mut self, delta: [f32; 2]) {
        for emitter in &mut self.emitters {
            emitter.origin[0] += delta[0];
            emitter.origin[1] += delta[1];
        }
        for cp in &mut self.control_points {
            cp[0] += delta[0];
            cp[1] += delta[1];
        }
    }

    fn stop_emitting(&mut self) {
        for emitter in &mut self.emitters {
            emitter.rate = 0.0;
        }
    }

    /// Total pixel coverage the rasterizer would write for one frame: the
    /// sum of every particle's quad area, each clipped to the bounding-box
    /// area. Overdraw — many huge overlapping sprites — makes this far
    /// larger than the bounding box itself, and it is the number a raster
    /// budget must divide, not the buffer size.
    pub fn coverage(&self, bbox_area: f32) -> f32 {
        self.particles
            .iter()
            .map(|p| {
                let side = (p.size * 2.0).max(1.0);
                (side * side).min(bbox_area)
            })
            .sum()
    }

    /// Bounding box (`min_x, min_y, max_x, max_y`) over all alive particles'
    /// `position ± size` (with a small margin for the soft-falloff glow
    /// radius) — `None` when nothing is alive, so callers can skip a wasted
    /// raster/upload entirely. Lets `render_onto` be called with a canvas
    /// sized to just this box instead of the full scene.
    pub fn bounds(&self) -> Option<(f32, f32, f32, f32)> {
        let mut min_x = f32::MAX;
        let mut min_y = f32::MAX;
        let mut max_x = f32::MIN;
        let mut max_y = f32::MIN;
        for p in &self.particles {
            // `p.size` is a radius (see `sizerandom`'s halving comment) for
            // both the circle and textured-quad draws; pad by a couple of
            // pixels for the soft falloff's antialiasing.
            let half = p.size.max(1.0) + 2.0;
            min_x = min_x.min(p.x - half);
            min_y = min_y.min(p.y - half);
            max_x = max_x.max(p.x + half);
            max_y = max_y.max(p.y + half);
        }
        (!self.particles.is_empty()).then_some((min_x, min_y, max_x, max_y))
    }

    /// `origin` shifts every particle's canvas-space position by `-origin` —
    /// lets callers raster into a canvas sized to just a system's bounding
    /// box (see `bounds()`) instead of the full scene. Existing full-canvas
    /// callers pass `[0.0, 0.0]`.
    pub fn render_onto(
        &self,
        canvas: &mut RgbaImage,
        sprite: Option<&ParticleSprite>,
        origin: [f32; 2],
    ) {
        self.render_onto_blended(canvas, sprite, origin, false);
    }

    /// `render_onto` with the material's blending mode: `additive` layers
    /// accumulate premultiplied contributions (the reference renders each
    /// particle quad with `glBlendFunc(GL_SRC_ALPHA, GL_ONE)`, CPass.cpp),
    /// so overlapping particles brighten each other instead of the newer
    /// one occluding the older via "over" compositing.
    pub fn render_onto_blended(
        &self,
        canvas: &mut RgbaImage,
        sprite: Option<&ParticleSprite>,
        origin: [f32; 2],
        additive: bool,
    ) {
        self.render_onto_scaled(canvas, sprite, origin, additive, 1.0);
    }

    /// `render_onto_blended` with all positions/sizes multiplied by `scale`
    /// — lets a caller rasterize into a smaller-than-scene buffer that the
    /// GPU compositor then stretches back up. Soft, huge sprites (smoke
    /// clouds, rain sheets — radii in the hundreds of pixels) are the ones
    /// that need this, and they lose nothing visible to the upscale, while
    /// the CPU rasterization cost drops with scale².
    pub fn render_onto_scaled(
        &self,
        canvas: &mut RgbaImage,
        sprite: Option<&ParticleSprite>,
        origin: [f32; 2],
        additive: bool,
        scale: f32,
    ) {
        if self.rope_mode {
            self.render_rope_onto(canvas, sprite, origin, additive, scale);
            self.render_children_onto(canvas, origin, scale);
            return;
        }

        let w = canvas.width() as i32;
        let h = canvas.height() as i32;

        for p in &self.particles {
            // Authored alpha straight through — the reference draws
            // `v_Color.a = p.alpha` with no damping factor.
            let alpha = (p.alpha * 255.0).clamp(0.0, 255.0) as u8;
            if alpha == 0 {
                continue;
            }
            let sz = (p.size * scale) as i32;
            if sz <= 0 {
                continue;
            }
            let px_pos = (p.x - origin[0]) * scale;
            let py_pos = (p.y - origin[1]) * scale;

            if let Some(sprite) = sprite {
                // `p.frame` is only ever set (in `step`) when the sprite has
                // more than one frame; a single-frame sprite always draws
                // frame 0 regardless of its (unused, still -1.0) value.
                //
                // Cross-fade: the fractional part of `p.frame` blends toward
                // the NEXT frame, clamped at the last frame (the real
                // engine's ComputeSpriteFrame uses
                // `nextFrame = min(numFrames - 1, currentFrame + 1)` — no
                // wrap, in any animation mode). RandomFrame assigns integer
                // frames, so its fraction is 0 and no second sample happens.
                let (frame_idx, next_frame) = if sprite.frames.len() > 1 {
                    let f = p.frame.max(0.0);
                    let a = (f as usize).min(sprite.frames.len() - 1);
                    let b = (a + 1).min(sprite.frames.len() - 1);
                    let blend = f.fract();
                    if b != a && blend > 1.0 / 255.0 {
                        (a, Some((&sprite.frames[b], blend)))
                    } else {
                        (a, None)
                    }
                } else {
                    (0, None)
                };
                let tex = &sprite.frames[frame_idx];
                // `spritetrail`: align the quad's V axis with the particle's
                // velocity and stretch it by clamp(|v|*length, min, max) —
                // genericparticle.vert's TRAILRENDERER path, where `up` =
                // normalized velocity scaled by that clamp and texture-top
                // (v=0) points along it.
                let (half_w, half_h, rotation) = match self.sprite_trail {
                    Some(t) => {
                        let vel_len = (p.vx * p.vx + p.vy * p.vy).sqrt();
                        let trail = (vel_len * t.length)
                            .clamp(t.min_length.min(t.max_length), t.max_length)
                            .max(0.05);
                        let angle = if vel_len > 0.0001 {
                            p.vy.atan2(p.vx) + std::f32::consts::FRAC_PI_2
                        } else {
                            p.rotation
                        };
                        (p.size * scale, p.size * trail * scale, angle)
                    }
                    None => (p.size * scale, p.size * scale, p.rotation),
                };
                draw_textured_particle(
                    canvas,
                    px_pos,
                    py_pos,
                    half_w,
                    half_h,
                    rotation,
                    p.color,
                    alpha,
                    tex,
                    next_frame,
                    sprite.overbright,
                    additive,
                    scale < 0.9,
                );
                continue;
            }

            let cx = px_pos as i32;
            let cy = py_pos as i32;

            let sz2 = (sz * sz) as f32;
            for dy in -sz..=sz {
                for dx in -sz..=sz {
                    let d2 = (dx * dx + dy * dy) as f32;
                    if d2 > sz2 {
                        continue;
                    }
                    let px = cx + dx;
                    let py = cy + dy;
                    if px < 0 || py < 0 || px >= w || py >= h {
                        continue;
                    }
                    // Soft radial falloff — a flat-alpha disc reads as a harsh,
                    // hard-edged blob, especially for large ambient sprites
                    // (fog/light shafts) that are meant to be soft glows.
                    let t = 1.0 - (d2 / sz2).sqrt();
                    let falloff = t * t;
                    let src_a = (alpha as f32 / 255.0) * falloff;
                    let src_c = [
                        p.color[0] as f32 / 255.0,
                        p.color[1] as f32 / 255.0,
                        p.color[2] as f32 / 255.0,
                    ];
                    let dst = canvas.get_pixel_mut(px as u32, py as u32);
                    blend_pixel(dst, src_c, src_a, additive);
                }
            }
        }

        self.render_children_onto(canvas, origin, scale);
    }

    /// Draws every child instance over the parent's output, each with its
    /// own material sprite and blending.
    fn render_children_onto(&self, canvas: &mut RgbaImage, origin: [f32; 2], scale: f32) {
        for child in &self.children {
            for inst in &child.instances {
                inst.system.render_onto_scaled(
                    canvas,
                    child.sprite.as_ref(),
                    origin,
                    child.additive,
                    scale,
                );
            }
        }
    }

    /// Draws a connected ribbon through living particles (oldest-first, since
    /// `step()`'s retain preserves order and new spawns are appended) instead
    /// of independent circles. Position is interpolated with the reference's
    /// exact Catmull-Rom formula (CParticle.cpp renderRope); size/color/alpha
    /// are linearly interpolated per sub-segment, and the material texture is
    /// stretched across the ribbon: U spans the width (the beam profile that
    /// makes a 100px-wide rope read as a hairline filament), V runs along the
    /// rope as `(trailPosition + uvY) / (trailLength - 1)` with `uvscale`
    /// tiling, optional arc-length smoothing, and 1-cycle/sec scrolling —
    /// CParticle::renderRope's exact vertex semantics, evaluated per-pixel
    /// here instead of in its rope shader.
    fn render_rope_onto(
        &self,
        canvas: &mut RgbaImage,
        sprite: Option<&ParticleSprite>,
        origin: [f32; 2],
        additive: bool,
        scale: f32,
    ) {
        let Some(geo) = self.build_rope_geometry(origin, scale) else {
            return;
        };
        let tex = sprite.and_then(|s| s.frames.first());
        let overbright = sprite.map(|s| s.overbright).unwrap_or(1.0);
        for &(i, v_a, v_b) in &geo.quads {
            let (a, b) = (&geo.points[i], &geo.points[i + 1]);
            fill_rope_segment(
                canvas, a.pos, b.pos, a.normal, b.normal, a.size, b.size, a.color, b.color,
                a.alpha, b.alpha, additive, tex, overbright, v_a, v_b,
            );
        }
    }

    /// Builds the rope's shared geometry — Catmull-Rom subpoints, per-joint
    /// central-difference normals, and per-quad V ranges — consumed by both
    /// the CPU scanline fill and the GPU vertex emitter.
    fn build_rope_geometry(&self, origin: [f32; 2], scale: f32) -> Option<RopeGeometry> {
        let n = self.particles.len();
        if n < 2 {
            return None;
        }

        let subdiv = self.rope_subdivision;
        let mut subpoints = Vec::with_capacity((n - 1) * subdiv + 1);
        for i in 0..(n - 1) {
            let idx0 = i.saturating_sub(1);
            let p0 = &self.particles[idx0];
            let p1 = &self.particles[i];
            let p2 = &self.particles[i + 1];
            let idx3 = (i + 2).min(n - 1);
            let p3 = &self.particles[idx3];

            for step in 0..subdiv {
                let t = step as f32 / subdiv as f32;
                let pos =
                    catmull_rom_vec2([p0.x, p0.y], [p1.x, p1.y], [p2.x, p2.y], [p3.x, p3.y], t);
                subpoints.push(RopePoint {
                    pos: [(pos[0] - origin[0]) * scale, (pos[1] - origin[1]) * scale],
                    size: lerp_f32(p1.size, p2.size, t) * scale,
                    alpha: lerp_f32(p1.alpha, p2.alpha, t),
                    color: [
                        lerp_f32(p1.color[0] as f32, p2.color[0] as f32, t) as u8,
                        lerp_f32(p1.color[1] as f32, p2.color[1] as f32, t) as u8,
                        lerp_f32(p1.color[2] as f32, p2.color[2] as f32, t) as u8,
                    ],
                    normal: [0.0, 1.0],
                });
            }
        }
        let last = &self.particles[n - 1];
        subpoints.push(RopePoint {
            pos: [(last.x - origin[0]) * scale, (last.y - origin[1]) * scale],
            size: last.size * scale,
            color: last.color,
            alpha: last.alpha,
            normal: [0.0, 1.0],
        });

        // V coordinates along the rope — CParticle::renderRope's exact
        // semantics: each sub-segment quad consumes 1/(trailLength-1) of UV
        // space, `uvscale` divides the usable length (values > 1 tile the
        // texture), `uvsmoothing` redistributes V by arc length (only when
        // all lifetimes match and scrolling is off), and `uvscrolling`
        // shifts one full UV cycle per second.
        let total_sub = (subpoints.len() - 1) as f32;
        let usable_len = (total_sub / self.rope_uv_scale).max(f32::EPSILON);
        let use_smoothing =
            self.rope_uv_smoothing && self.life_min == self.life_max && !self.rope_uv_scrolling;
        let mut cumulative_arc = Vec::new();
        let mut total_arc = 0.0f32;
        if use_smoothing {
            cumulative_arc = Vec::with_capacity(subpoints.len());
            cumulative_arc.push(0.0);
            for pair in subpoints.windows(2) {
                let (dx, dy) = (
                    pair[1].pos[0] - pair[0].pos[0],
                    pair[1].pos[1] - pair[0].pos[1],
                );
                total_arc += (dx * dx + dy * dy).sqrt();
                cumulative_arc.push(total_arc);
            }
        }
        let scroll_offset = if self.rope_uv_scrolling {
            (self.time % 10000.0) * usable_len
        } else {
            0.0
        };

        // Per-joint width directions — genericropeparticle.vert's
        // `trailRightStart = cross(eye, end − prev)` / `trailRightEnd =
        // cross(eye, after − start)`: the perpendicular of the CENTRAL-
        // DIFFERENCE tangent at each joint, so consecutive quads share
        // their edge exactly and the ribbon reads as connected trapezoids
        // instead of independently-angled facets. In screen space
        // `cross((0,0,1), t)` is the 90° rotate `(−t.y, t.x)`.
        let joint_normal = |i: usize| -> [f32; 2] {
            let prev = &subpoints[i.saturating_sub(1)];
            let next = &subpoints[(i + 1).min(subpoints.len() - 1)];
            let t = [next.pos[0] - prev.pos[0], next.pos[1] - prev.pos[1]];
            let len = (t[0] * t[0] + t[1] * t[1]).sqrt();
            if len < 0.0001 {
                [0.0, 1.0]
            } else {
                [-t[1] / len, t[0] / len]
            }
        };
        let normals: Vec<[f32; 2]> = (0..subpoints.len()).map(joint_normal).collect();
        for (i, n) in normals.into_iter().enumerate() {
            subpoints[i].normal = n;
        }

        let quads = (0..subpoints.len() - 1)
            .map(|s| {
                let trail_position = if use_smoothing && total_arc > 0.0 {
                    cumulative_arc[s] / total_arc * total_sub
                } else {
                    s as f32
                } + scroll_offset;
                (
                    s,
                    trail_position / usable_len,
                    (trail_position + 1.0) / usable_len,
                )
            })
            .collect();

        Some(RopeGeometry {
            points: subpoints,
            quads,
        })
    }

    /// Appends pre-transformed triangle-list vertices for the GPU particle
    /// pipeline: 6 per sprite quad (rotation/spritetrail applied here, the
    /// same math as the CPU textured draw) or 6 per rope sub-quad. Positions
    /// are absolute scene pixels; the vertex shader converts to NDC.
    /// `frame_count` is the sprite's frame count (0/1 = static frame 0).
    pub fn emit_gpu_vertices(&self, out: &mut Vec<GpuVertex>, frame_count: usize) {
        if self.rope_mode {
            self.emit_rope_gpu_vertices(out);
            return;
        }
        for p in &self.particles {
            if p.alpha <= 0.0 || p.size <= 0.0 {
                continue;
            }
            // Frame/cross-fade selection — mirrors the CPU textured draw.
            let (frame, blend) = if frame_count > 1 {
                let f = p.frame.max(0.0);
                let a = (f as usize).min(frame_count - 1);
                let b = (a + 1).min(frame_count - 1);
                let fract = f.fract();
                if b != a && fract > 1.0 / 255.0 {
                    (a as f32, fract)
                } else {
                    (a as f32, 0.0)
                }
            } else {
                (0.0, 0.0)
            };
            let (half_w, half_h, rotation) = match self.sprite_trail {
                Some(t) => {
                    let vel_len = (p.vx * p.vx + p.vy * p.vy).sqrt();
                    let trail = (vel_len * t.length)
                        .clamp(t.min_length.min(t.max_length), t.max_length)
                        .max(0.05);
                    let angle = if vel_len > 0.0001 {
                        p.vy.atan2(p.vx) + std::f32::consts::FRAC_PI_2
                    } else {
                        p.rotation
                    };
                    (p.size, p.size * trail, angle)
                }
                None => (p.size, p.size, p.rotation),
            };
            let (c, sn) = (rotation.cos(), rotation.sin());
            let color = [
                p.color[0] as f32 / 255.0,
                p.color[1] as f32 / 255.0,
                p.color[2] as f32 / 255.0,
                p.alpha.clamp(0.0, 1.0),
            ];
            let corner = |lx: f32, ly: f32, u: f32, v: f32| GpuVertex {
                pos: [p.x + lx * c - ly * sn, p.y + lx * sn + ly * c],
                uv: [u, v],
                color,
                frame_blend: [frame, blend, 0.0, 0.0],
            };
            let tl = corner(-half_w, -half_h, 0.0, 0.0);
            let tr = corner(half_w, -half_h, 1.0, 0.0);
            let br = corner(half_w, half_h, 1.0, 1.0);
            let bl = corner(-half_w, half_h, 0.0, 1.0);
            out.extend_from_slice(&[tl, tr, br, br, bl, tl]);
        }
    }

    fn emit_rope_gpu_vertices(&self, out: &mut Vec<GpuVertex>) {
        let Some(geo) = self.build_rope_geometry([0.0, 0.0], 1.0) else {
            return;
        };
        for &(i, v_a, v_b) in &geo.quads {
            let (a, b) = (&geo.points[i], &geo.points[i + 1]);
            // Same convexity guard as fill_rope_segment: keep both joint
            // normals on the same side when the chain doubles back.
            let nb = if a.normal[0] * b.normal[0] + a.normal[1] * b.normal[1] < 0.0 {
                [-b.normal[0], -b.normal[1]]
            } else {
                b.normal
            };
            let (ha, hb) = (a.size.max(0.5), b.size.max(0.5));
            let col = |p: &RopePoint| {
                [
                    p.color[0] as f32 / 255.0,
                    p.color[1] as f32 / 255.0,
                    p.color[2] as f32 / 255.0,
                    p.alpha.clamp(0.0, 1.0),
                ]
            };
            let (ca, cb) = (col(a), col(b));
            let vert = |pos: [f32; 2], uv: [f32; 2], color: [f32; 4]| GpuVertex {
                pos,
                uv,
                color,
                frame_blend: [0.0, 0.0, 0.0, 0.0],
            };
            // U convention matches fill_rope_segment: +normal side = u 1.
            let a1 = vert(
                [a.pos[0] + a.normal[0] * ha, a.pos[1] + a.normal[1] * ha],
                [1.0, v_a],
                ca,
            );
            let b1 = vert(
                [b.pos[0] + nb[0] * hb, b.pos[1] + nb[1] * hb],
                [1.0, v_b],
                cb,
            );
            let b0 = vert(
                [b.pos[0] - nb[0] * hb, b.pos[1] - nb[1] * hb],
                [0.0, v_b],
                cb,
            );
            let a0 = vert(
                [a.pos[0] - a.normal[0] * ha, a.pos[1] - a.normal[1] * ha],
                [0.0, v_a],
                ca,
            );
            out.extend_from_slice(&[a1, b1, b0, b0, a0, a1]);
        }
    }

    /// Visits every live child instance with its child index (the GPU path
    /// keys uploaded textures by child, since instances spawn at runtime)
    /// and its material blending.
    pub fn visit_gpu_children(&self, f: &mut dyn FnMut(&ParticleSystem, usize, bool)) {
        for (child_idx, child) in self.children.iter().enumerate() {
            for inst in &child.instances {
                f(&inst.system, child_idx, child.additive);
            }
        }
    }

    /// Build-time child info for GPU texture upload: each child's sprite and
    /// whether its preset uses a rope renderer.
    pub fn child_sprite_info(&self) -> Vec<(Option<&ParticleSprite>, bool)> {
        self.children
            .iter()
            .map(|c| {
                let rope = c.config.renderer.iter().any(|r| {
                    r.get("name")
                        .and_then(|n| n.as_str())
                        .map(|n| n == "rope" || n == "ropetrail")
                        .unwrap_or(false)
                });
                (c.sprite.as_ref(), rope)
            })
            .collect()
    }

    pub fn is_rope(&self) -> bool {
        self.rope_mode
    }
}

/// One point along a rope's subdivided spine (see `build_rope_geometry`).
struct RopePoint {
    pos: [f32; 2],
    size: f32,
    color: [u8; 3],
    alpha: f32,
    /// Per-joint central-difference perpendicular (width direction).
    normal: [f32; 2],
}

/// Shared rope geometry: spine points and per-quad `(start_index, v_a, v_b)`.
struct RopeGeometry {
    points: Vec<RopePoint>,
    quads: Vec<(usize, f32, f32)>,
}

/// One pre-transformed vertex for the GPU particle pipeline (triangle list,
/// read from a storage buffer by `vs_particles`). 48 bytes, std430-compatible.
#[derive(Clone, Copy, Debug, Default)]
pub struct GpuVertex {
    /// Absolute scene pixels (y-down).
    pub pos: [f32; 2],
    pub uv: [f32; 2],
    /// rgb 0-1, a = particle alpha.
    pub color: [f32; 4],
    /// x = sprite-sheet frame, y = cross-fade weight toward frame+1.
    pub frame_blend: [f32; 4],
}

impl GpuVertex {
    /// Flattens vertices to the byte layout `vs_particles` reads.
    pub fn as_bytes(verts: &[GpuVertex]) -> Vec<u8> {
        let mut out = Vec::with_capacity(verts.len() * 48);
        for v in verts {
            for f in v
                .pos
                .iter()
                .chain(v.uv.iter())
                .chain(v.color.iter())
                .chain(v.frame_blend.iter())
            {
                out.extend_from_slice(&f.to_le_bytes());
            }
        }
        out
    }
}

/// Reference's exact Catmull-Rom formula (CParticle.cpp:2124-2130), applied
/// per-axis.
fn catmull_rom_vec2(p0: [f32; 2], p1: [f32; 2], p2: [f32; 2], p3: [f32; 2], t: f32) -> [f32; 2] {
    let t2 = t * t;
    let t3 = t2 * t;
    let mut out = [0.0; 2];
    for axis in 0..2 {
        let (p0, p1, p2, p3) = (p0[axis], p1[axis], p2[axis], p3[axis]);
        out[axis] = 0.5
            * ((2.0 * p1)
                + (-p0 + p2) * t
                + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
                + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3);
    }
    out
}

fn lerp_f32(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Rasterizes one ribbon quad between two consecutive sub-points: offsets
/// each endpoint by its half-width along its own JOINT normal (the central-
/// difference perpendicular computed by the caller), producing edge-sharing
/// trapezoids exactly like genericropeparticle.vert's
/// `position + mix(trailRightStart, trailRightEnd, uvY) * (uvX*2−1)` — where
/// `trailRight* = normalize(cross(eye, tangent)) * size`, i.e. the half-width
/// IS the particle size (spawn already halved `sizerandom`, so `p.size` is a
/// radius). With a material texture, each pixel samples it at (U = signed
/// offset across the width, V = lerp(v_a, v_b) along the rope, wrapping for
/// tiled `uvscale`) and modulates the particle color/alpha — the beam strip's
/// transparent edges are what make a wide rope render as a thin filament.
/// Without a texture (material failed to resolve), falls back to a flat fill.
#[allow(clippy::too_many_arguments)]
fn fill_rope_segment(
    canvas: &mut RgbaImage,
    a: [f32; 2],
    b: [f32; 2],
    normal_a: [f32; 2],
    normal_b: [f32; 2],
    size_a: f32,
    size_b: f32,
    color_a: [u8; 3],
    color_b: [u8; 3],
    alpha_a: f32,
    alpha_b: f32,
    additive: bool,
    tex: Option<&RgbaImage>,
    overbright: f32,
    v_a: f32,
    v_b: f32,
) {
    let dir = [b[0] - a[0], b[1] - a[1]];
    let len = (dir[0] * dir[0] + dir[1] * dir[1]).sqrt();
    if len < 0.0001 {
        return;
    }
    let dir = [dir[0] / len, dir[1] / len];
    let (half_a, half_b) = (size_a.max(0.5), size_b.max(0.5));

    // Keep both joint normals on the same side so the quad stays convex —
    // a chain doubling back >90° would otherwise produce a bowtie that the
    // point-in-quad test rejects entirely (the GPU draws such quads as two
    // crossing triangles; a same-side flip is the closest scanline analog).
    let normal_b = if normal_a[0] * normal_b[0] + normal_a[1] * normal_b[1] < 0.0 {
        [-normal_b[0], -normal_b[1]]
    } else {
        normal_b
    };

    let corners = [
        [a[0] + normal_a[0] * half_a, a[1] + normal_a[1] * half_a],
        [b[0] + normal_b[0] * half_b, b[1] + normal_b[1] * half_b],
        [b[0] - normal_b[0] * half_b, b[1] - normal_b[1] * half_b],
        [a[0] - normal_a[0] * half_a, a[1] - normal_a[1] * half_a],
    ];

    let min_x = corners
        .iter()
        .map(|c| c[0])
        .fold(f32::MAX, f32::min)
        .floor()
        .max(0.0) as i32;
    let max_x = corners
        .iter()
        .map(|c| c[0])
        .fold(f32::MIN, f32::max)
        .ceil()
        .min(canvas.width() as f32 - 1.0) as i32;
    let min_y = corners
        .iter()
        .map(|c| c[1])
        .fold(f32::MAX, f32::min)
        .floor()
        .max(0.0) as i32;
    let max_y = corners
        .iter()
        .map(|c| c[1])
        .fold(f32::MIN, f32::max)
        .ceil()
        .min(canvas.height() as f32 - 1.0) as i32;
    if min_x > max_x || min_y > max_y {
        return;
    }

    let w = canvas.width() as i32;
    let h = canvas.height() as i32;
    for py in min_y..=max_y {
        for px in min_x..=max_x {
            if px < 0 || py < 0 || px >= w || py >= h {
                continue;
            }
            let point = [px as f32 + 0.5, py as f32 + 0.5];
            if !point_in_convex_quad(point, &corners) {
                continue;
            }
            // Blend color/alpha by which endpoint the pixel is nearer along
            // the segment (a cheap stand-in for true per-pixel interpolation).
            let seg_t = ((point[0] - a[0]) * dir[0] + (point[1] - a[1]) * dir[1]) / len;
            let seg_t = seg_t.clamp(0.0, 1.0);

            if let Some(tex) = tex {
                // Signed offset from the centerline along the interpolated
                // joint normal, in [-half, +half] at this point's width.
                let center = [lerp_f32(a[0], b[0], seg_t), lerp_f32(a[1], b[1], seg_t)];
                let nl = [
                    lerp_f32(normal_a[0], normal_b[0], seg_t),
                    lerp_f32(normal_a[1], normal_b[1], seg_t),
                ];
                let nl_len = (nl[0] * nl[0] + nl[1] * nl[1]).sqrt().max(0.0001);
                let d_perp =
                    ((point[0] - center[0]) * nl[0] + (point[1] - center[1]) * nl[1]) / nl_len;
                let half_t = lerp_f32(half_a, half_b, seg_t).max(0.5);
                let u = (0.5 + 0.5 * (d_perp / half_t)).clamp(0.0, 1.0);
                let v = lerp_f32(v_a, v_b, seg_t).rem_euclid(1.0);
                let sample = sample_bilinear(tex, u, v);
                let src_a =
                    lerp_f32(alpha_a, alpha_b, seg_t).clamp(0.0, 1.0) * (sample[3] as f32 / 255.0);
                if src_a <= 0.0 {
                    continue;
                }
                let src_c = [
                    (sample[0] as f32 / 255.0)
                        * (lerp_f32(color_a[0] as f32, color_b[0] as f32, seg_t) / 255.0)
                        * overbright,
                    (sample[1] as f32 / 255.0)
                        * (lerp_f32(color_a[1] as f32, color_b[1] as f32, seg_t) / 255.0)
                        * overbright,
                    (sample[2] as f32 / 255.0)
                        * (lerp_f32(color_a[2] as f32, color_b[2] as f32, seg_t) / 255.0)
                        * overbright,
                ];
                let dst = canvas.get_pixel_mut(px as u32, py as u32);
                blend_pixel(dst, src_c, src_a, additive);
                continue;
            }

            let alpha = (lerp_f32(alpha_a, alpha_b, seg_t) * 180.0).clamp(0.0, 255.0) as u8;
            if alpha == 0 {
                continue;
            }
            let color = [
                lerp_f32(color_a[0] as f32, color_b[0] as f32, seg_t) as u8,
                lerp_f32(color_a[1] as f32, color_b[1] as f32, seg_t) as u8,
                lerp_f32(color_a[2] as f32, color_b[2] as f32, seg_t) as u8,
            ];

            let src_a = alpha as f32 / 255.0;
            let src_c = [
                color[0] as f32 / 255.0,
                color[1] as f32 / 255.0,
                color[2] as f32 / 255.0,
            ];
            let dst = canvas.get_pixel_mut(px as u32, py as u32);
            blend_pixel(dst, src_c, src_a, additive);
        }
    }
}

/// Point-in-convex-quad test via consistent cross-product sign across all
/// four edges. `corners` must be wound consistently (either all-CW or
/// all-CCW) — true by construction here (two parallel perpendicular offsets).
fn point_in_convex_quad(point: [f32; 2], corners: &[[f32; 2]; 4]) -> bool {
    let mut sign = 0.0f32;
    for i in 0..4 {
        let a = corners[i];
        let b = corners[(i + 1) % 4];
        let edge = [b[0] - a[0], b[1] - a[1]];
        let to_point = [point[0] - a[0], point[1] - a[1]];
        let cross = edge[0] * to_point[1] - edge[1] * to_point[0];
        if i == 0 {
            sign = cross;
        } else if cross * sign < 0.0 {
            return false;
        }
    }
    true
}

/// `boids` operator: O(n^2) separation/alignment/cohesion, using each
/// particle's position/velocity as of the start of the frame (the caller
/// snapshots `self.particles` before any per-particle mutation happens this
/// step). Fine for the particle counts real presets use (`maxcount` is
/// typically in the hundreds, not thousands).
fn compute_boids_accel(particles: &[Particle], b: &BoidsParams) -> Vec<[f32; 2]> {
    let n = particles.len();
    let mut accel = vec![[0.0f32; 2]; n];
    for i in 0..n {
        let pi = &particles[i];
        let mut separation = [0.0f32; 2];
        let mut sep_count: u32 = 0;
        let mut avg_vel = [0.0f32; 2];
        let mut avg_pos = [0.0f32; 2];
        let mut neighbor_count: u32 = 0;
        for (j, pj) in particles.iter().enumerate() {
            if i == j {
                continue;
            }
            let d = [pi.x - pj.x, pi.y - pj.y];
            let dist = (d[0] * d[0] + d[1] * d[1]).sqrt();
            if dist > 0.0001 && dist < b.separation_threshold {
                let f = 1.0 / dist;
                separation[0] += d[0] * f;
                separation[1] += d[1] * f;
                sep_count += 1;
            }
            if dist < b.neighbor_threshold {
                avg_vel[0] += pj.vx;
                avg_vel[1] += pj.vy;
                avg_pos[0] += pj.x;
                avg_pos[1] += pj.y;
                neighbor_count += 1;
            }
        }
        let mut a = [0.0f32; 2];
        if sep_count > 0 {
            let n = sep_count as f32;
            a[0] += separation[0] / n * b.separation_factor;
            a[1] += separation[1] / n * b.separation_factor;
        }
        if neighbor_count > 0 {
            let n = neighbor_count as f32;
            a[0] += (avg_vel[0] / n - pi.vx) * b.alignment_factor;
            a[1] += (avg_vel[1] / n - pi.vy) * b.alignment_factor;
            a[0] += (avg_pos[0] / n - pi.x) * b.cohesion_factor;
            a[1] += (avg_pos[1] / n - pi.y) * b.cohesion_factor;
        }
        accel[i] = a;
    }
    accel
}

/// Apply one collider's contact response to a particle already moved this
/// frame — see `CollisionShape`/`CollisionBehavior` docs for the "no C++
/// reference" caveat on exact semantics.
fn resolve_collision(p: &mut Particle, c: &CollisionParams) {
    // `normal` points from the collider's solid surface toward free space;
    // `penetration` is how far past the surface the particle currently is
    // (<= 0.0 means "not colliding", skipped by the caller's-eye-view check
    // below in each shape branch).
    let (normal, penetration): ([f32; 2], f32) = match c.shape {
        CollisionShape::Sphere { radius } => {
            let d = [p.x - c.center[0], p.y - c.center[1]];
            let dist = (d[0] * d[0] + d[1] * d[1]).sqrt();
            if dist >= radius || dist < 1e-5 {
                return;
            }
            ([d[0] / dist, d[1] / dist], radius - dist)
        }
        CollisionShape::Box { half_extent } => {
            let d = [p.x - c.center[0], p.y - c.center[1]];
            if d[0].abs() >= half_extent[0] || d[1].abs() >= half_extent[1] {
                return;
            }
            let px = half_extent[0] - d[0].abs();
            let py = half_extent[1] - d[1].abs();
            if px < py {
                ([d[0].signum(), 0.0], px)
            } else {
                ([0.0, d[1].signum()], py)
            }
        }
        CollisionShape::Plane { normal } => {
            let d = [p.x - c.center[0], p.y - c.center[1]];
            let signed = d[0] * normal[0] + d[1] * normal[1];
            if signed >= 0.0 {
                return;
            }
            (normal, -signed)
        }
    };

    match c.behavior {
        CollisionBehavior::Delete => {
            p.life = -1.0;
        }
        CollisionBehavior::Slide => {
            p.x += normal[0] * penetration;
            p.y += normal[1] * penetration;
            let vn = p.vx * normal[0] + p.vy * normal[1];
            if vn < 0.0 {
                p.vx -= normal[0] * vn;
                p.vy -= normal[1] * vn;
            }
        }
        CollisionBehavior::Bounce => {
            p.x += normal[0] * penetration;
            p.y += normal[1] * penetration;
            let vn = p.vx * normal[0] + p.vy * normal[1];
            if vn < 0.0 {
                let restitution = 1.0 + c.bounce_factor;
                p.vx -= normal[0] * vn * restitution;
                p.vy -= normal[1] * vn * restitution;
            }
        }
    }
}

/// Wraps an angle to `[-pi, pi]`, matching `createAngularMovementOperator`'s
/// per-axis wrap (CParticle.cpp).
fn wrap_angle(a: f32) -> f32 {
    let mut a = a % std::f32::consts::TAU;
    if a > std::f32::consts::PI {
        a -= std::f32::consts::TAU;
    } else if a < -std::f32::consts::PI {
        a += std::f32::consts::TAU;
    }
    a
}

// Minimal vec3 math for the turbulent/vortex ports (the reference uses glm;
// pulling in a linear-algebra crate for five one-liners isn't worth it).

fn vec3_dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn vec3_cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn vec3_length(v: [f32; 3]) -> f32 {
    vec3_dot(v, v).sqrt()
}

/// Normalize, or return `fallback` for a (near-)zero vector — the
/// reference's recurring `length > 0.0001 ? normalize(v) : default` guard.
fn vec3_normalize_or(v: [f32; 3], fallback: [f32; 3]) -> [f32; 3] {
    let len = vec3_length(v);
    if len > 0.0001 {
        [v[0] / len, v[1] / len, v[2] / len]
    } else {
        fallback
    }
}

/// Rotate `v` by `angle` radians around unit-length `axis` (Rodrigues'
/// formula — equivalent to the reference's `glm::rotate` mat3 use).
fn vec3_rotate_axis(v: [f32; 3], axis: [f32; 3], angle: f32) -> [f32; 3] {
    let (sin, cos) = angle.sin_cos();
    let cross = vec3_cross(axis, v);
    let dot = vec3_dot(axis, v) * (1.0 - cos);
    [
        v[0] * cos + cross[0] * sin + axis[0] * dot,
        v[1] * cos + cross[1] * sin + axis[1] * dot,
        v[2] * cos + cross[2] * sin + axis[2] * dot,
    ]
}

/// Draws one particle as a rotated, textured quad: inverse-rotates each
/// destination pixel in the quad's bounding box back into texture space,
/// bilinear-samples, and composites tinted-by-`color`/scaled-by-`alpha_byte`
/// over the canvas. The reference hands rotation+size to an external,
/// proprietary vertex shader to build the quad (not present in the
/// open-source reference repo) — this CPU inverse-mapping is our own design
/// for the same visual effect.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
fn draw_textured_particle(
    canvas: &mut RgbaImage,
    cx: f32,
    cy: f32,
    half_w: f32,
    half_h: f32,
    rotation: f32,
    color: [u8; 3],
    alpha_byte: u8,
    tex: &RgbaImage,
    next_frame: Option<(&RgbaImage, f32)>,
    overbright: f32,
    additive: bool,
    nearest: bool,
) {
    // `p.size` is already a radius, not a diameter (see `sizerandom`'s
    // comment on the halving done at spawn) — matches the flat-color circle
    // draw's `sz = p.size` convention, so a particle looks the same size
    // whether or not it has a material/texture. The two half-extents differ
    // only for `spritetrail` quads (stretched along velocity).
    let half_w = half_w.max(0.5);
    let half_h = half_h.max(0.5);
    let cos_r = rotation.cos();
    let sin_r = rotation.sin();

    // Bounding box of the rotated quad: the diagonal half-extent covers any
    // rotation angle.
    let diag = (half_w * half_w + half_h * half_h).sqrt();
    let w = canvas.width() as i32;
    let h = canvas.height() as i32;
    let min_x = ((cx - diag).floor() as i32).max(0);
    let max_x = ((cx + diag).ceil() as i32).min(w - 1);
    let min_y = ((cy - diag).floor() as i32).max(0);
    let max_y = ((cy + diag).ceil() as i32).min(h - 1);

    for py in min_y..=max_y {
        for px in min_x..=max_x {
            let dx = px as f32 + 0.5 - cx;
            let dy = py as f32 + 0.5 - cy;
            // Inverse-rotate the destination offset back into the quad's own
            // (unrotated) local space.
            let lx = dx * cos_r + dy * sin_r;
            let ly = -dx * sin_r + dy * cos_r;
            if lx < -half_w || lx > half_w || ly < -half_h || ly > half_h {
                continue;
            }
            let u = (lx + half_w) / (2.0 * half_w);
            let v = (ly + half_h) / (2.0 * half_h);
            let mut sample = if nearest {
                sample_nearest(tex, u, v)
            } else {
                sample_bilinear(tex, u, v)
            };
            // Spritesheet cross-fade: mix toward the next frame's sample
            // (genericparticle.frag's SPRITESHEETBLEND path).
            if let Some((tex_b, blend)) = next_frame {
                let b = if nearest {
                    sample_nearest(tex_b, u, v)
                } else {
                    sample_bilinear(tex_b, u, v)
                };
                for c in 0..4 {
                    sample[c] = (sample[c] as f32 + (b[c] as f32 - sample[c] as f32) * blend)
                        .clamp(0.0, 255.0) as u8;
                }
            }
            let src_a = (sample[3] as f32 / 255.0) * (alpha_byte as f32 / 255.0);
            if src_a <= 0.0 {
                continue;
            }
            // genericparticle.frag: `color.rgb *= g_Overbright` after the
            // texture × vertex-color multiply. blend_pixel clamps, so
            // overbright > 1 saturates exactly like the GPU's clamped u8 target.
            let src_c = [
                (sample[0] as f32 / 255.0) * (color[0] as f32 / 255.0) * overbright,
                (sample[1] as f32 / 255.0) * (color[1] as f32 / 255.0) * overbright,
                (sample[2] as f32 / 255.0) * (color[2] as f32 / 255.0) * overbright,
            ];
            let dst = canvas.get_pixel_mut(px as u32, py as u32);
            blend_pixel(dst, src_c, src_a, additive);
        }
    }
}

/// Composites one source sample into a canvas pixel.
///
/// Normal/translucent layers use standard "over" compositing (not a plain
/// lerp): the destination isn't always opaque — the GPU path renders
/// particles into their own transparent scene-sized buffer before
/// compositing it as a texture, so overlapping particles must accumulate
/// alpha correctly rather than assuming a fully-opaque backdrop.
///
/// Additive layers accumulate premultiplied contributions instead — the
/// reference draws every particle quad with
/// `glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE, GL_SRC_ALPHA, GL_ONE)`
/// (CPass.cpp), i.e. `dst += src * src_a` — so overlapping glows brighten
/// each other rather than the newest quad occluding the ones below. The
/// buffer then holds already-premultiplied color, which the GPU composite's
/// pure-add mode adds to the scene without re-weighting by alpha.
fn blend_pixel(dst: &mut image::Rgba<u8>, src_c: [f32; 3], src_a: f32, additive: bool) {
    if additive {
        for i in 0..3 {
            let dst_c = dst[i] as f32 / 255.0;
            dst[i] = ((dst_c + src_c[i] * src_a) * 255.0).clamp(0.0, 255.0) as u8;
        }
        let dst_a = dst[3] as f32 / 255.0;
        dst[3] = ((dst_a + src_a * src_a) * 255.0).clamp(0.0, 255.0) as u8;
        return;
    }
    let dst_a = dst[3] as f32 / 255.0;
    let out_a = src_a + dst_a * (1.0 - src_a);
    if out_a > 0.0 {
        for i in 0..3 {
            let dst_c = dst[i] as f32 / 255.0;
            let out_c = (src_c[i] * src_a + dst_c * dst_a * (1.0 - src_a)) / out_a;
            dst[i] = (out_c * 255.0).clamp(0.0, 255.0) as u8;
        }
    }
    dst[3] = (out_a * 255.0).clamp(0.0, 255.0) as u8;
}

/// Point sampling — used when the raster target is itself downscaled (the
/// GPU upscale blurs anyway), saving the 4-tap filter per pixel.
fn sample_nearest(tex: &RgbaImage, u: f32, v: f32) -> [u8; 4] {
    let (tw, th) = (tex.width(), tex.height());
    if tw == 0 || th == 0 {
        return [0; 4];
    }
    let x = ((u.clamp(0.0, 1.0) * (tw as f32 - 1.0)) + 0.5) as u32;
    let y = ((v.clamp(0.0, 1.0) * (th as f32 - 1.0)) + 0.5) as u32;
    tex.get_pixel(x.min(tw - 1), y.min(th - 1)).0
}

fn sample_bilinear(tex: &RgbaImage, u: f32, v: f32) -> [u8; 4] {
    let (tw, th) = (tex.width(), tex.height());
    let fx = (u.clamp(0.0, 1.0) * (tw as f32 - 1.0).max(0.0)).max(0.0);
    let fy = (v.clamp(0.0, 1.0) * (th as f32 - 1.0).max(0.0)).max(0.0);
    let x0 = fx.floor() as u32;
    let y0 = fy.floor() as u32;
    let x1 = (x0 + 1).min(tw - 1);
    let y1 = (y0 + 1).min(th - 1);
    let (tx, ty) = (fx - x0 as f32, fy - y0 as f32);

    let p00 = tex.get_pixel(x0, y0);
    let p10 = tex.get_pixel(x1, y0);
    let p01 = tex.get_pixel(x0, y1);
    let p11 = tex.get_pixel(x1, y1);

    let mut out = [0u8; 4];
    for i in 0..4 {
        let top = p00[i] as f32 * (1.0 - tx) + p10[i] as f32 * tx;
        let bottom = p01[i] as f32 * (1.0 - tx) + p11[i] as f32 * tx;
        out[i] = (top * (1.0 - ty) + bottom * ty).clamp(0.0, 255.0) as u8;
    }
    out
}

fn lerp_u8(min: f32, max: f32) -> u8 {
    (min + fastrand::f32() * (max - min)).clamp(0.0, 255.0) as u8
}

/// Linear ramp from `start_val` to `end_val` over `[start, end]` of `t`,
/// clamped flat outside that range. Matches `Maths::fadeValue` exactly.
fn fade_value(t: f32, start: f32, end: f32, start_val: f32, end_val: f32) -> f32 {
    if t <= start {
        start_val
    } else if t >= end {
        end_val
    } else {
        let f = (t - start) / (end - start);
        start_val + f * (end_val - start_val)
    }
}

/// Initializer names are matched exactly — they're fixed strings in the
/// reference's ObjectParser, and substring matching let
/// `turbulentvelocityrandom`/`angularvelocityrandom` shadow
/// `velocityrandom` (or `angularmovement` shadow `movement`) depending on
/// declaration order.
/// The `exponent` (distribution bias) of a named initializer, or 1.0 (uniform).
fn initializer_exponent(initializers: &[Initializer], name: &str) -> f32 {
    initializers
        .iter()
        .find(|i| i.name == name)
        .and_then(|i| i.exponent)
        .map(|e| e as f32)
        .filter(|e| *e > 0.0)
        .unwrap_or(1.0)
}

fn scalar_range_from_initializers(
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
fn vec3_range_from_initializers(
    initializers: &[Initializer],
    name: &str,
) -> Option<([f32; 3], [f32; 3])> {
    let init = initializers.iter().find(|init| init.name == name)?;
    let min = init.min.as_ref().and_then(value_as_vec3)?;
    let max = init.max.as_ref().and_then(value_as_vec3)?;
    Some((min, max))
}

fn value_as_f32(v: &serde_json::Value) -> Option<f32> {
    v.as_f64().map(|f| f as f32)
}

/// Public alias for callers outside this module (e.g. render.rs's
/// control-point override plumbing).
pub fn value_as_vec3_pub(v: &serde_json::Value) -> Option<[f32; 3]> {
    value_as_vec3(v)
}

fn value_as_vec3(v: &serde_json::Value) -> Option<[f32; 3]> {
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

fn parse_f32_vec3(s: Option<&str>) -> [f32; 3] {
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Real preset fields (velocityrandom, colorrandom, angularvelocityrandom)
    /// use `"x y z"` vector strings for min/max, not numbers — the old
    /// `Option<f64>` typing rejected these outright.
    #[test]
    fn parses_real_preset_with_vector_initializers() {
        let json = r#"{
            "maxcount": 20,
            "emitter": [{"name":"sphererandom","rate":1.5,"distancemin":0,"distancemax":512}],
            "initializer": [
                {"id":2,"name":"lifetimerandom","min":3,"max":5},
                {"id":3,"name":"sizerandom","min":1000,"max":2200},
                {"id":4,"name":"velocityrandom","min":"25 0 0","max":"80 0 0"},
                {"id":5,"name":"colorrandom","min":"255 255 255","max":"255 255 255"}
            ],
            "operator": [{"id":9,"name":"movement","gravity":"0 0 0"}],
            "renderer": [{"id":1,"name":"sprite"}]
        }"#;
        let config: ParticleConfig = serde_json::from_str(json).expect("should parse");
        let sys = ParticleSystem::from_config(&config, [100.0, 200.0], None);
        assert_eq!(sys.velocity_min, Some([25.0, 0.0, 0.0]));
        assert_eq!(sys.velocity_max, Some([80.0, 0.0, 0.0]));
        assert_eq!(sys.color_min, [255.0, 255.0, 255.0]);
    }

    #[test]
    fn scalar_initializer_without_vector_fields_still_parses() {
        let json = r#"{
            "maxcount": 16,
            "emitter": [{"name":"box","rate":1}],
            "initializer": [{"id":2,"name":"lifetimerandom","min":1,"max":1}]
        }"#;
        let config: ParticleConfig = serde_json::from_str(json).expect("should parse");
        let sys = ParticleSystem::from_config(&config, [0.0, 0.0], None);
        assert_eq!(sys.life_min, 1.0);
        assert_eq!(sys.life_max, 1.0);
        assert!(sys.velocity_min.is_none());
    }

    #[test]
    fn instance_override_sets_color_and_multipliers() {
        let json = r#"{
            "maxcount": 5,
            "emitter": [{"name":"box","rate":1}]
        }"#;
        let config: ParticleConfig = serde_json::from_str(json).expect("should parse");
        let overrides = InstanceOverride {
            alpha: Some(0.5),
            color: Some("192 192 192".to_string()),
            rate: Some(2.0),
            ..Default::default()
        };
        let sys = ParticleSystem::from_config(&config, [0.0, 0.0], Some(&overrides));
        assert_eq!(sys.color_override, Some([192, 192, 192]));
        assert_eq!(sys.alpha_mult, 0.5);
        assert_eq!(sys.rate_mult, 2.0);
    }

    /// `count` scales the pool, `lifetime` scales spawned lifetimes, and
    /// `colorn` tints the colorrandom result — the multiplier semantics of
    /// CParticle.cpp (lines 59, 755-765, 712-720).
    #[test]
    fn instance_override_count_lifetime_colorn() {
        let json = r#"{
            "maxcount": 100,
            "emitter": [{"name":"box","rate":1000}],
            "initializer": [
                {"id":1,"name":"lifetimerandom","min":2,"max":2},
                {"id":2,"name":"colorrandom","min":"200 100 50","max":"200 100 50"}
            ]
        }"#;
        let config: ParticleConfig = serde_json::from_str(json).expect("should parse");
        let overrides = InstanceOverride {
            count: Some(0.5),
            lifetime: Some(2.0),
            colorn: Some(serde_json::json!("0.5 1 1")),
            ..Default::default()
        };
        let mut sys = ParticleSystem::from_config(&config, [0.0, 0.0], Some(&overrides));
        assert_eq!(sys.max_count, 50, "count 0.5 should halve maxcount 100");
        sys.step(0.01);
        let p = &sys.particles[0];
        assert_eq!(p.life, 4.0, "lifetime 2.0 should double the 2s lifetime");
        assert_eq!(
            p.color,
            [100, 100, 50],
            "colorn should tint the red channel only"
        );
    }

    /// `enabled: false` in an instanceoverride must drop the whole system at
    /// the layer level (checked in render.rs/animated.rs, config-level here).
    #[test]
    fn instance_override_enabled_parses() {
        let o: InstanceOverride =
            serde_json::from_str(r#"{"enabled": false, "id": 3}"#).expect("should parse");
        assert_eq!(o.enabled.as_ref().and_then(|v| v.as_bool()), Some(false));
    }

    /// Emitter local origins are authored y-up (CParticle.cpp negates the y
    /// in both emitter constructors): "10 20 0" sits 20 ABOVE the object in
    /// screen space — rainperspective's "0 768 0" spawn band must end up
    /// above the frame, not below it.
    #[test]
    fn emitter_origin_is_local_offset_from_spawn_center() {
        let json = r#"{
            "maxcount": 5,
            "emitter": [{"name":"box","rate":1,"origin":"10 20 0"}]
        }"#;
        let config: ParticleConfig = serde_json::from_str(json).expect("should parse");
        let sys = ParticleSystem::from_config(&config, [100.0, 200.0], None);
        assert_eq!(sys.emitters[0].origin[0], 110.0);
        assert_eq!(sys.emitters[0].origin[1], 180.0);
    }

    #[test]
    fn movement_operator_gravity_field_parses() {
        let json = r#"{
            "maxcount": 5,
            "emitter": [{"name":"box","rate":1}],
            "operator": [{"id":1,"name":"movement","gravity":"0 -50 0"}]
        }"#;
        let config: ParticleConfig = serde_json::from_str(json).expect("should parse");
        let sys = ParticleSystem::from_config(&config, [0.0, 0.0], None);
        assert_eq!(sys.gravity, [0.0, -50.0]);
    }

    #[test]
    fn fade_value_clamps_outside_range_and_lerps_inside() {
        assert_eq!(fade_value(0.0, 0.3, 0.7, 0.0, 1.0), 0.0);
        assert_eq!(fade_value(0.3, 0.3, 0.7, 0.0, 1.0), 0.0);
        assert_eq!(fade_value(0.5, 0.3, 0.7, 0.0, 1.0), 0.5);
        assert_eq!(fade_value(0.7, 0.3, 0.7, 0.0, 1.0), 1.0);
        assert_eq!(fade_value(1.0, 0.3, 0.7, 0.0, 1.0), 1.0);
    }

    /// A real `alphafade` preset (fog1.json): fadeintime=0.5 means alpha
    /// should be 0 at spawn, ~full partway through the ramp, and full for
    /// the rest of the particle's life (no fadeouttime = holds at 1.0).
    #[test]
    fn alphafade_ramps_in_then_holds() {
        let json = r#"{
            "maxcount": 5,
            "emitter": [{"name":"box","rate":1}],
            "initializer": [{"id":1,"name":"alpharandom","min":1,"max":1}],
            "operator": [{"id":1,"name":"alphafade","fadeintime":0.5}]
        }"#;
        let config: ParticleConfig = serde_json::from_str(json).expect("should parse");
        let mut sys = ParticleSystem::from_config(&config, [0.0, 0.0], None);
        // Force-spawn one particle with a known, fixed lifetime.
        sys.life_min = 10.0;
        sys.life_max = 10.0;
        sys.emitters[0].rate = 1000.0; // spawn immediately
        sys.step(0.001); // this step only spawns; the fade math runs on existing particles first
        assert_eq!(sys.particles.len(), 1);

        // Just after spawn (age≈0), fully faded in from 0.
        sys.step(0.001);
        assert!(sys.particles[0].alpha < 0.1);

        // Halfway through the fade-in window (age=2.5s of 10s life → lifetime_pos=0.25,
        // half of fadeintime=0.5): alpha should be roughly half.
        sys.step(2.499);
        let mid = sys.particles[0].alpha;
        assert!((0.3..0.7).contains(&mid), "expected ~0.5, got {mid}");

        // Well past the fade-in window, alpha should be at full (no fadeouttime set → holds at 1.0).
        sys.step(5.0);
        assert!(sys.particles[0].alpha > 0.9);
    }

    /// A real `sizechange` preset (magic_pulse.json): startvalue=0, endvalue=2,
    /// endtime=1 (normalized) — size should start near 0 and grow past the
    /// initializer-sampled size as lifetime_pos approaches endtime.
    #[test]
    fn sizechange_scales_initial_size_over_lifetime() {
        let json = r#"{
            "maxcount": 5,
            "emitter": [{"name":"box","rate":1}],
            "initializer": [{"id":1,"name":"sizerandom","min":100,"max":100}],
            "operator": [{"id":1,"name":"sizechange","startvalue":0,"endvalue":2,"endtime":1}]
        }"#;
        let config: ParticleConfig = serde_json::from_str(json).expect("should parse");
        let mut sys = ParticleSystem::from_config(&config, [0.0, 0.0], None);
        sys.life_min = 10.0;
        sys.life_max = 10.0;
        sys.emitters[0].rate = 1000.0;
        sys.step(0.001); // this step only spawns; the fade math runs on existing particles first
        assert_eq!(sys.particles.len(), 1);
        let initial_size = sys.particles[0].initial_size;
        sys.step(0.001);
        assert!(sys.particles[0].size < initial_size * 0.1);

        // lifetime_pos=1.0 reached at age=10s (endtime=1 normalized == full life here).
        sys.step(9.99);
        assert!((sys.particles[0].size - initial_size * 2.0).abs() < initial_size * 0.1);
    }

    /// A real `colorchange` preset (torch.json): "r g b" 0..1 vector strings
    /// for startvalue/endvalue — these previously broke parsing entirely
    /// (Operator.startvalue/endvalue were typed as bare f64).
    #[test]
    fn colorchange_parses_and_multiplies_initial_color_over_lifetime() {
        let json = r#"{
            "maxcount": 5,
            "emitter": [{"name":"box","rate":1}],
            "initializer": [{"id":1,"name":"colorrandom","min":"255 255 255","max":"255 255 255"}],
            "operator": [{"id":1,"name":"colorchange","startvalue":"1 0.75 0","endvalue":"1 0 0","endtime":1}]
        }"#;
        let config: ParticleConfig = serde_json::from_str(json).expect("should parse");
        let mut sys = ParticleSystem::from_config(&config, [0.0, 0.0], None);
        sys.life_min = 10.0;
        sys.life_max = 10.0;
        sys.emitters[0].rate = 1000.0;
        sys.step(0.001); // spawns the particle; fade math runs on existing particles first
        assert_eq!(sys.particles.len(), 1);

        sys.step(0.001); // age≈0 → at startvalue (1, 0.75, 0) * white
        let c0 = sys.particles[0].color;
        assert_eq!(c0[0], 255);
        assert!((c0[1] as i32 - 191).abs() <= 2); // 0.75*255≈191
        assert_eq!(c0[2], 0);

        sys.step(9.99); // age≈10 → lifetime_pos=1.0 → at endvalue (1, 0, 0) * white
        let c1 = sys.particles[0].color;
        assert_eq!(c1, [255, 0, 0]);
    }

    /// A real `controlpointattract` preset pulls particles within `threshold/2`
    /// of a control point toward it, and leaves particles outside that radius
    /// untouched (CParticle.cpp createControlPointAttractOperator).
    #[test]
    fn controlpointattract_pulls_nearby_particles_toward_control_point() {
        let json = r#"{
            "maxcount": 5,
            "emitter": [{"name":"box","rate":1}],
            "controlpoint": [{"id":0,"offset":"200 0 0"}],
            "operator": [{"id":1,"name":"controlpointattract","controlpoint":0,"scale":500,"threshold":1000}]
        }"#;
        let config: ParticleConfig = serde_json::from_str(json).expect("should parse");
        // spawn_center [0,0] + control point offset "200 0 0" -> control point at [200, 0].
        let mut sys = ParticleSystem::from_config(&config, [0.0, 0.0], None);
        assert_eq!(sys.control_points, vec![[200.0, 0.0, 0.0]]);

        sys.life_min = 10.0;
        sys.life_max = 10.0;
        sys.emitters[0].rate = 1000.0;
        sys.step(0.001); // spawns the particle at/near [0,0]
        assert_eq!(sys.particles.len(), 1);
        let start_dist = ((sys.particles[0].x - 200.0).powi(2) + sys.particles[0].y.powi(2)).sqrt();

        for _ in 0..10 {
            sys.step(0.05);
        }
        let end_dist = ((sys.particles[0].x - 200.0).powi(2) + sys.particles[0].y.powi(2)).sqrt();
        assert!(
            end_dist < start_dist,
            "expected particle to be pulled toward control point: start={start_dist} end={end_dist}"
        );
    }

    /// A particle outside the attract radius (`threshold/2`) should be left
    /// alone entirely.
    #[test]
    fn controlpointattract_ignores_particles_outside_threshold() {
        let json = r#"{
            "maxcount": 5,
            "emitter": [{"name":"box","rate":1}],
            "controlpoint": [{"id":0,"offset":"5000 0 0"}],
            "operator": [{"id":1,"name":"controlpointattract","controlpoint":0,"scale":500,"threshold":10}]
        }"#;
        let config: ParticleConfig = serde_json::from_str(json).expect("should parse");
        let mut sys = ParticleSystem::from_config(&config, [0.0, 0.0], None);
        sys.life_min = 10.0;
        sys.life_max = 10.0;
        sys.emitters[0].rate = 1000.0;
        sys.step(0.001);
        assert_eq!(sys.particles.len(), 1);
        // Capture the emitter-assigned spawn velocity (unrelated to the
        // control point), then confirm a step leaves it untouched since the
        // particle is well outside the attract radius.
        let (vx0, vy0) = (sys.particles[0].vx, sys.particles[0].vy);
        sys.step(0.05);
        assert_eq!(sys.particles[0].vx, vx0);
        assert_eq!(sys.particles[0].vy, vy0);
    }

    /// A `rope`/`ropetrail` renderer should draw a connected ribbon through
    /// living particles rather than isolated circles — verify pixels are
    /// filled *between* particle centers, not just at them.
    #[test]
    fn rope_renderer_fills_ribbon_between_particles() {
        let json = r#"{
            "maxcount": 5,
            "emitter": [{"name":"box","rate":1}],
            "initializer": [{"id":1,"name":"sizerandom","min":40,"max":40}],
            "renderer": [{"id":1,"name":"rope","subdivision":4}]
        }"#;
        let config: ParticleConfig = serde_json::from_str(json).expect("should parse");
        let sys = ParticleSystem::from_config(&config, [0.0, 0.0], None);
        assert!(sys.rope_mode);
        assert_eq!(sys.rope_subdivision, 4);

        // Manually place three particles in a horizontal line, all alive.
        let mut sys = sys;
        for (i, x) in [0.0f32, 50.0, 100.0].into_iter().enumerate() {
            sys.particles.push(Particle {
                id: 0,
                x,
                y: 100.0,
                vx: 0.0,
                vy: 0.0,
                life: 5.0,
                max_life: 5.0,
                size: 20.0,
                alpha: 1.0,
                color: [255, 0, 0],
                initial_alpha: 1.0,
                initial_size: 20.0,
                initial_color: [1.0, 0.0, 0.0],
                osc_alpha: None,
                osc_size: None,
                osc_pos: None,
                rotation: 0.0,
                angular_velocity: 0.0,
                frame: -1.0,
            });
            let _ = i;
        }

        let mut canvas = RgbaImage::new(200, 200);
        sys.render_onto(&mut canvas, None, [0.0, 0.0]);

        // Midpoint between the first two particles (25, 100) should be filled
        // — a plain per-particle circle draw (radius 10) would leave this gap
        // empty since it's 25px from the nearest particle center.
        let mid_pixel = canvas.get_pixel(25, 100);
        assert!(
            mid_pixel[3] > 0,
            "expected ribbon fill at midpoint, got {mid_pixel:?}"
        );
    }

    /// A rope with a material texture must sample it across the ribbon width
    /// (U = perpendicular offset): a beam strip that's transparent except for
    /// its center column should render as a thin filament along the rope's
    /// spine, leaving the outer ribbon area untouched — the fix for fat
    /// solid lightning bolts (discharge presets) and giant trail wedges.
    #[test]
    fn rope_renderer_samples_texture_across_width() {
        let json = r#"{
            "maxcount": 5,
            "emitter": [{"name":"box","rate":1}],
            "renderer": [{"id":1,"name":"rope","subdivision":1}]
        }"#;
        let config: ParticleConfig = serde_json::from_str(json).expect("should parse");
        let mut sys = ParticleSystem::from_config(&config, [0.0, 0.0], None);
        for x in [0.0f32, 50.0, 100.0] {
            sys.particles.push(make_particle(x, 100.0, 20.0, 0.0));
        }

        // Beam strip: transparent except the 3 center columns (white, opaque).
        let mut beam = RgbaImage::new(33, 4);
        for y in 0..4 {
            for x in 15..=17 {
                beam.put_pixel(x, y, image::Rgba([255, 255, 255, 255]));
            }
        }
        let sprite = ParticleSprite {
            frames: vec![beam],
            duration: 0.0,
            overbright: 1.0,
        };

        let mut canvas = RgbaImage::new(200, 200);
        sys.render_onto(&mut canvas, Some(&sprite), [0.0, 0.0]);

        // The rope spine (u = 0.5 → the white center column) is filled…
        let spine = canvas.get_pixel(25, 100);
        assert!(
            spine[3] > 0,
            "expected filament on rope spine, got {spine:?}"
        );
        // …but the ribbon area off-center (still inside the 10px half-width
        // the old flat fill covered) samples the transparent texture edge.
        let edge = canvas.get_pixel(25, 92);
        assert_eq!(edge[3], 0, "expected transparent ribbon edge, got {edge:?}");
    }

    fn child_ref(child_type: Option<&str>, probability: f64) -> ChildRef {
        ChildRef {
            name: "child.json".into(),
            child_type: child_type.map(str::to_string),
            probability: Some(probability),
            maxcount: None,
        }
    }

    fn simple_child_config() -> ParticleConfig {
        serde_json::from_str(
            r#"{
                "maxcount": 10,
                "emitter": [{"name":"box","rate":1000}],
                "initializer": [{"id":1,"name":"lifetimerandom","min":5,"max":5}]
            }"#,
        )
        .expect("child config should parse")
    }

    /// An `eventfollow` child spawns one instance per parent particle and
    /// keeps its emitters glued to the parent as it moves.
    #[test]
    fn child_eventfollow_tracks_parent_particle() {
        let json = r#"{
            "maxcount": 1,
            "emitter": [{"name":"box","rate":1000}],
            "initializer": [
                {"id":1,"name":"lifetimerandom","min":10,"max":10},
                {"id":2,"name":"velocityrandom","min":"100 0 0","max":"100 0 0"}
            ]
        }"#;
        let config: ParticleConfig = serde_json::from_str(json).expect("should parse");
        let mut sys = ParticleSystem::from_config(&config, [0.0, 0.0], None);
        sys.add_child(
            simple_child_config(),
            None,
            false,
            &child_ref(Some("eventfollow"), 1.0),
            [0.0, 0.0],
        );

        sys.step(0.01); // spawns the parent + its follow instance
        assert_eq!(sys.particles.len(), 1);
        assert_eq!(sys.children[0].instances.len(), 1);

        sys.step(1.0); // parent moves +100x; instance must follow
        let parent = [sys.particles[0].x, sys.particles[0].y];
        let inst = &sys.children[0].instances[0];
        assert_eq!(inst.last_center, parent, "instance should track the parent");
        assert!(
            !inst.system.particles.is_empty(),
            "follow instance should be emitting its own particles"
        );
    }

    /// Probability 0 must never attach an instance; a dead parent stops the
    /// instance's emitters and lets it retire once its particles expire.
    #[test]
    fn child_probability_and_parent_death() {
        let json = r#"{
            "maxcount": 1,
            "emitter": [{"name":"box","rate":1000}],
            "initializer": [{"id":1,"name":"lifetimerandom","min":0.05,"max":0.05}]
        }"#;
        let config: ParticleConfig = serde_json::from_str(json).expect("should parse");

        let mut never = ParticleSystem::from_config(&config, [0.0, 0.0], None);
        never.add_child(
            simple_child_config(),
            None,
            false,
            &child_ref(Some("eventfollow"), 0.0),
            [0.0, 0.0],
        );
        never.step(0.01);
        assert!(never.children[0].instances.is_empty());

        let mut sys = ParticleSystem::from_config(&config, [0.0, 0.0], None);
        sys.add_child(
            simple_child_config(),
            None,
            false,
            &child_ref(Some("eventfollow"), 1.0),
            [0.0, 0.0],
        );
        sys.step(0.01);
        assert_eq!(sys.children[0].instances.len(), 1);
        sys.step(0.1); // parent (life 0.05) dies
        let inst = &sys.children[0].instances[0];
        assert!(
            inst.parent_id.is_none(),
            "dead parent should detach the instance"
        );
        assert!(
            inst.system.emitters.iter().all(|e| e.rate == 0.0),
            "detached instance should stop emitting"
        );
    }

    /// A type-less (static) child gets one persistent instance at the parent
    /// object's origin that emits continuously, surviving empty stretches.
    #[test]
    fn child_static_runs_continuously() {
        let json = r#"{
            "maxcount": 1,
            "emitter": [{"name":"box","rate":0}]
        }"#;
        let config: ParticleConfig = serde_json::from_str(json).expect("should parse");
        let mut sys = ParticleSystem::from_config(&config, [40.0, 40.0], None);
        sys.add_child(
            simple_child_config(),
            None,
            false,
            &child_ref(None, 1.0),
            [40.0, 40.0],
        );
        assert_eq!(
            sys.children[0].instances.len(),
            1,
            "static child instantiates immediately"
        );
        for _ in 0..10 {
            sys.step(0.05);
        }
        assert!(!sys.children[0].instances[0].system.particles.is_empty());
        assert!(sys.children[0].instances[0].persistent);
    }

    fn make_particle(x: f32, y: f32, size: f32, rotation: f32) -> Particle {
        Particle {
            id: 0,
            x,
            y,
            vx: 0.0,
            vy: 0.0,
            life: 5.0,
            max_life: 5.0,
            size,
            alpha: 1.0,
            color: [255, 255, 255],
            initial_alpha: 1.0,
            initial_size: size,
            initial_color: [1.0, 1.0, 1.0],
            osc_alpha: None,
            osc_size: None,
            osc_pos: None,
            rotation,
            angular_velocity: 0.0,
            frame: -1.0,
        }
    }

    /// A textured particle's rotation should actually affect what's drawn —
    /// sampling the same canvas location at two different rotation angles
    /// must produce different colors, proving the quad (and the texture
    /// sampled onto it) is genuinely rotated, not just passed through.
    #[test]
    fn textured_particle_rotation_changes_sampled_footprint() {
        // 8x8 sprite: left half red, right half blue, fully opaque.
        let mut tex = RgbaImage::new(8, 8);
        for y in 0..8 {
            for x in 0..8 {
                let color = if x < 4 {
                    [255, 0, 0, 255]
                } else {
                    [0, 0, 255, 255]
                };
                tex.put_pixel(x, y, image::Rgba(color));
            }
        }

        let config: ParticleConfig =
            serde_json::from_str(r#"{"maxcount":1,"emitter":[{"name":"box","rate":1}]}"#)
                .expect("should parse");

        let sprite = ParticleSprite::single(tex.clone());
        let render_at = |rotation: f32| -> image::Rgba<u8> {
            let mut sys = ParticleSystem::from_config(&config, [0.0, 0.0], None);
            sys.particles
                .push(make_particle(50.0, 50.0, 20.0, rotation));
            let mut canvas = RgbaImage::new(100, 100);
            sys.render_onto(&mut canvas, Some(&sprite), [0.0, 0.0]);
            *canvas.get_pixel(44, 50)
        };

        let unrotated = render_at(0.0);
        let rotated = render_at(std::f32::consts::FRAC_PI_2);
        assert_ne!(
            unrotated, rotated,
            "expected rotation to change the sampled footprint at a fixed canvas point"
        );
    }

    /// `rotationrandom` with a degenerate `[min.z, max.z]` range must spawn
    /// every particle at exactly that rotation (CParticle.cpp
    /// createRotationRandomInitializer, z-axis reduction); absent, spawn
    /// rotation stays 0.
    #[test]
    fn rotationrandom_sets_spawn_rotation() {
        let json = r#"{
            "maxcount": 4,
            "emitter": [{"name":"box","rate":1000}],
            "initializer": [{"id":1,"name":"rotationrandom","min":"0 0 1","max":"0 0 1"}]
        }"#;
        let config: ParticleConfig = serde_json::from_str(json).expect("should parse");
        let mut sys = ParticleSystem::from_config(&config, [0.0, 0.0], None);
        sys.step(0.001);
        assert!((sys.particles[0].rotation - 1.0).abs() < 1e-4);

        let plain: ParticleConfig =
            serde_json::from_str(r#"{"maxcount":1,"emitter":[{"name":"box","rate":1000}]}"#)
                .expect("should parse");
        let mut sys = ParticleSystem::from_config(&plain, [0.0, 0.0], None);
        sys.step(0.001);
        assert_eq!(sys.particles[0].rotation, 0.0);
    }

    /// `turbulentvelocityrandom` adds a curl-noise-directed velocity of
    /// magnitude in `[speedmin, speedmax]` on top of the plain velocity —
    /// with a zero base velocity, the spawn speed must equal the turbulent
    /// speed exactly (the direction is noise, the magnitude is not).
    #[test]
    fn turbulentvelocityrandom_sets_spawn_speed_magnitude() {
        let json = r#"{
            "maxcount": 8,
            "emitter": [{"name":"box","rate":1000}],
            "initializer": [
                {"id":1,"name":"velocityrandom","min":"0 0 0","max":"0 0 0"},
                {"id":2,"name":"turbulentvelocityrandom","speedmin":100,"speedmax":100}
            ]
        }"#;
        let config: ParticleConfig = serde_json::from_str(json).expect("should parse");
        let mut sys = ParticleSystem::from_config(&config, [0.0, 0.0], None);
        sys.step(0.001);
        assert!(!sys.particles.is_empty());
        for p in &sys.particles {
            let speed = (p.vx * p.vx + p.vy * p.vy).sqrt();
            assert!(
                (speed - 100.0).abs() < 0.5,
                "expected turbulent spawn speed 100, got {speed}"
            );
        }
    }

    /// The `turbulence` operator must push otherwise-motionless particles
    /// around (curl-noise acceleration, CParticle.cpp
    /// createTurbulenceOperator).
    #[test]
    fn turbulence_operator_accelerates_particles() {
        let json = r#"{
            "maxcount": 4,
            "emitter": [{"name":"box","rate":1000}],
            "initializer": [{"id":1,"name":"velocityrandom","min":"0 0 0","max":"0 0 0"}],
            "operator": [{"id":1,"name":"turbulence","speedmin":500,"speedmax":500,"scale":0.01}]
        }"#;
        let config: ParticleConfig = serde_json::from_str(json).expect("should parse");
        let mut sys = ParticleSystem::from_config(&config, [10.0, 20.0], None);
        for _ in 0..10 {
            sys.step(1.0 / 30.0);
        }
        let moving = sys
            .particles
            .iter()
            .any(|p| (p.vx * p.vx + p.vy * p.vy).sqrt() > 1.0);
        assert!(moving, "turbulence should have accelerated particles");
    }

    /// `vortex` spins particles tangentially around its control point: a
    /// particle at +X from the center with a +Z axis must gain -Y velocity
    /// in our y-down space (the reference's y-up cross(axis, radial),
    /// mirrored — see the reflection note in `step`), scaled by speedinner
    /// inside distanceinner.
    #[test]
    fn vortex_operator_adds_tangential_velocity() {
        let json = r#"{
            "maxcount": 4,
            "emitter": [{"name":"box","rate":0}],
            "controlpoint": [{"id":0,"offset":"0 0 0"}],
            "operator": [{"id":1,"name":"vortex","controlpoint":0,
                          "distanceinner":500,"distanceouter":650,
                          "speedinner":100,"speedouter":0}]
        }"#;
        let config: ParticleConfig = serde_json::from_str(json).expect("should parse");
        let mut sys = ParticleSystem::from_config(&config, [0.0, 0.0], None);
        sys.particles.push(make_particle(10.0, 0.0, 5.0, 0.0));
        sys.step(0.1);
        let p = &sys.particles[0];
        assert!(
            p.vy < -0.5,
            "expected tangential -Y velocity from vortex, got vy={}",
            p.vy
        );
        assert!(
            p.vx.abs() < 1e-3,
            "no radial force expected, got vx={}",
            p.vx
        );
    }

    /// `collisionsphere` with the default `bounce` behavior: a particle that
    /// has penetrated the sphere gets pushed back to the surface and its
    /// into-surface velocity component reflected (scaled by `1+bouncefactor`).
    #[test]
    fn collisionsphere_bounces_particle_off_surface() {
        let json = r#"{
            "maxcount": 1,
            "emitter": [{"name":"box","rate":0}],
            "controlpoint": [{"id":0,"offset":"0 0 0"}],
            "operator": [{"id":1,"name":"collisionsphere","controlpoint":0,"radius":50}]
        }"#;
        let config: ParticleConfig = serde_json::from_str(json).expect("should parse");
        let mut sys = ParticleSystem::from_config(&config, [0.0, 0.0], None);
        let mut p = make_particle(10.0, 0.0, 5.0, 0.0);
        p.vx = -50.0;
        sys.particles.push(p);
        sys.step(0.1);
        let p = &sys.particles[0];
        assert!((p.x - 50.0).abs() < 1e-3, "expected pushed to surface, x={}", p.x);
        assert!(p.vx > 0.0, "expected reflected outward velocity, vx={}", p.vx);
    }

    /// `collisionbox` with `slide`: the into-wall velocity component is
    /// zeroed (not reflected), and the particle is pushed back to the wall.
    #[test]
    fn collisionbox_slide_zeroes_into_wall_velocity() {
        let json = r#"{
            "maxcount": 1,
            "emitter": [{"name":"box","rate":0}],
            "controlpoint": [{"id":0,"offset":"0 0 0"}],
            "operator": [{"id":1,"name":"collisionbox","controlpoint":0,
                          "size":"100 100 0","collisionbehavior":"slide"}]
        }"#;
        let config: ParticleConfig = serde_json::from_str(json).expect("should parse");
        let mut sys = ParticleSystem::from_config(&config, [0.0, 0.0], None);
        let mut p = make_particle(60.0, 0.0, 5.0, 0.0);
        p.vx = -100.0;
        sys.particles.push(p);
        sys.step(0.15);
        let p = &sys.particles[0];
        assert!((p.x - 50.0).abs() < 1e-3, "expected pushed to wall, x={}", p.x);
        assert!(p.vx.abs() < 1e-3, "expected into-wall velocity zeroed, vx={}", p.vx);
    }

    /// `collisionplane` with `delete`: a particle on the solid side of the
    /// plane is killed outright (removed by `step`'s own retain).
    #[test]
    fn collisionplane_delete_removes_particle() {
        let json = r#"{
            "maxcount": 1,
            "emitter": [{"name":"box","rate":0}],
            "controlpoint": [{"id":0,"offset":"0 0 0"}],
            "operator": [{"id":1,"name":"collisionplane","controlpoint":0,
                          "collisionbehavior":"delete"}]
        }"#;
        let config: ParticleConfig = serde_json::from_str(json).expect("should parse");
        let mut sys = ParticleSystem::from_config(&config, [0.0, 0.0], None);
        sys.particles.push(make_particle(0.0, -10.0, 5.0, 0.0));
        sys.step(0.1);
        assert!(sys.particles.is_empty(), "particle should be deleted on contact");
    }

    /// `boids` cohesion pulls two distant particles toward each other.
    #[test]
    fn boids_cohesion_pulls_particles_together() {
        let json = r#"{
            "maxcount": 2,
            "emitter": [{"name":"box","rate":0}],
            "operator": [{"id":1,"name":"boids","neighborthreshold":500,
                          "separationthreshold":1,"cohesionfactor":1.0,"alignmentfactor":0}]
        }"#;
        let config: ParticleConfig = serde_json::from_str(json).expect("should parse");
        let mut sys = ParticleSystem::from_config(&config, [0.0, 0.0], None);
        sys.particles.push(make_particle(-100.0, 0.0, 5.0, 0.0));
        sys.particles.push(make_particle(100.0, 0.0, 5.0, 0.0));
        sys.step(0.1);
        assert!(sys.particles[0].vx > 1.0, "left particle should move right, vx={}", sys.particles[0].vx);
        assert!(sys.particles[1].vx < -1.0, "right particle should move left, vx={}", sys.particles[1].vx);
    }

    /// `boids` separation pushes two nearby particles apart.
    #[test]
    fn boids_separation_pushes_particles_apart() {
        let json = r#"{
            "maxcount": 2,
            "emitter": [{"name":"box","rate":0}],
            "operator": [{"id":1,"name":"boids","separationthreshold":50,
                          "neighborthreshold":1,"separationfactor":1.0}]
        }"#;
        let config: ParticleConfig = serde_json::from_str(json).expect("should parse");
        let mut sys = ParticleSystem::from_config(&config, [0.0, 0.0], None);
        sys.particles.push(make_particle(-2.0, 0.0, 5.0, 0.0));
        sys.particles.push(make_particle(2.0, 0.0, 5.0, 0.0));
        sys.step(0.1);
        assert!(sys.particles[0].vx < 0.0, "left particle should move further left, vx={}", sys.particles[0].vx);
        assert!(sys.particles[1].vx > 0.0, "right particle should move further right, vx={}", sys.particles[1].vx);
    }

    /// `movement.drag` decays velocity by `1 - drag*dt` per step, clamped
    /// at zero so an over-large `drag*dt` stops the particle instead of
    /// reversing it (CParticle.cpp createMovementOperator).
    #[test]
    fn movement_drag_decays_velocity_without_reversal() {
        let json = r#"{
            "maxcount": 1,
            "emitter": [{"name":"box","rate":0}],
            "operator": [{"id":1,"name":"movement","gravity":"0 0 0","drag":4.0}]
        }"#;
        let config: ParticleConfig = serde_json::from_str(json).expect("should parse");
        let mut sys = ParticleSystem::from_config(&config, [0.0, 0.0], None);
        let mut p = make_particle(50.0, 50.0, 5.0, 0.0);
        p.vx = 100.0;
        sys.particles.push(p);

        sys.step(0.1); // drag*dt = 0.4 -> vx *= 0.6
        assert!((sys.particles[0].vx - 60.0).abs() < 1e-3);

        sys.step(0.5); // drag*dt = 2.0 -> clamps to 0, must not reverse
        assert_eq!(sys.particles[0].vx, 0.0);
    }

    /// `mapsequencebetweencontrolpoints` walks spawn positions along the
    /// cp0->cp1 segment; `limitbehavior: "mirror"` ping-pongs (0, 1/3, 2/3,
    /// 1, 2/3, 1/3, 0, ...) instead of wrapping.
    #[test]
    fn mapsequence_between_walks_segment_with_mirror() {
        let json = r#"{
            "maxcount": 8,
            "emitter": [{"name":"box","rate":100000}],
            "controlpoint": [{"id":0,"offset":"0 0 0"},{"id":1,"offset":"90 0 0"}],
            "initializer": [
                {"id":1,"name":"velocityrandom","min":"0 0 0","max":"0 0 0"},
                {"id":2,"name":"mapsequencebetweencontrolpoints","count":4,"limitbehavior":"mirror"}
            ]
        }"#;
        let config: ParticleConfig = serde_json::from_str(json).expect("should parse");
        let mut sys = ParticleSystem::from_config(&config, [0.0, 0.0], None);
        sys.step(0.001);
        assert!(sys.particles.len() >= 7, "expected at least 7 spawns");
        let xs: Vec<f32> = sys.particles.iter().take(7).map(|p| p.x).collect();
        let expect = [0.0, 30.0, 60.0, 90.0, 60.0, 30.0, 0.0];
        for (i, (&x, &e)) in xs.iter().zip(expect.iter()).enumerate() {
            assert!((x - e).abs() < 1e-3, "slot {i}: expected x={e}, got {x}");
        }
    }

    /// `mapsequencearoundcontrolpoint` spawns at the control point and
    /// launches successive particles at evenly-divided angles: with
    /// `count: 4` and a +X-only speed, consecutive spawn velocities must
    /// point right/down-rotated/left/up-rotated (the reference's clockwise
    /// mat3), all with the same magnitude.
    #[test]
    fn mapsequence_spawns_at_control_point_with_rotating_velocity() {
        let json = r#"{
            "maxcount": 4,
            "emitter": [{"name":"box","rate":100000}],
            "controlpoint": [{"id":0,"offset":"50 60 0"}],
            "initializer": [{"id":1,"name":"mapsequencearoundcontrolpoint",
                             "controlpoint":0,"count":4,
                             "speedmin":"80 0 0","speedmax":"80 0 0"}]
        }"#;
        let config: ParticleConfig = serde_json::from_str(json).expect("should parse");
        let mut sys = ParticleSystem::from_config(&config, [0.0, 0.0], None);
        sys.step(0.001);
        assert!(sys.particles.len() >= 4, "expected at least 4 spawns");
        for p in &sys.particles {
            assert_eq!((p.x, p.y), (50.0, 60.0), "must spawn at the control point");
            let speed = (p.vx * p.vx + p.vy * p.vy).sqrt();
            assert!(
                (speed - 80.0).abs() < 1e-3,
                "speed must stay 80, got {speed}"
            );
        }
        // First two spawns: angle 0 (+X) then 2pi/4 = 90deg rotated.
        assert!((sys.particles[0].vx - 80.0).abs() < 1e-3);
        assert!(sys.particles[0].vy.abs() < 1e-3);
        assert!(sys.particles[1].vx.abs() < 1e-3);
        assert!(sys.particles[1].vy.abs() > 79.0);
    }

    fn two_frame_sprite() -> ParticleSprite {
        ParticleSprite {
            frames: vec![
                RgbaImage::from_pixel(4, 4, image::Rgba([255, 0, 0, 255])),
                RgbaImage::from_pixel(4, 4, image::Rgba([0, 0, 255, 255])),
            ],
            duration: 1.0,
            overbright: 1.0,
        }
    }

    fn render_center_with_frame(frame: f32) -> image::Rgba<u8> {
        let config: ParticleConfig =
            serde_json::from_str(r#"{"maxcount":1,"emitter":[{"name":"box","rate":0}]}"#)
                .expect("should parse");
        let mut sys = ParticleSystem::from_config(&config, [0.0, 0.0], None);
        let mut p = make_particle(20.0, 20.0, 8.0, 0.0);
        p.frame = frame;
        sys.particles.push(p);
        let sprite = two_frame_sprite();
        let mut canvas = RgbaImage::new(40, 40);
        sys.render_onto(&mut canvas, Some(&sprite), [0.0, 0.0]);
        *canvas.get_pixel(20, 20)
    }

    /// A fractional frame position cross-fades between the two adjacent
    /// frames (genericparticle.frag's SPRITESHEETBLEND): halfway between a
    /// red and a blue frame must paint ~50/50 purple, not hard-switch.
    #[test]
    fn fractional_frame_cross_fades_adjacent_frames() {
        let px = render_center_with_frame(0.5);
        assert!(px[0] > 100 && px[0] < 155, "red ~half, got {px:?}");
        assert!(px[2] > 100 && px[2] < 155, "blue ~half, got {px:?}");
        assert_eq!(px[1], 0);
    }

    /// The next frame is clamped at the last frame (ComputeSpriteFrame's
    /// `min(numFrames - 1, currentFrame + 1)`) — a fraction past the final
    /// frame must NOT wrap-blend back toward frame 0.
    #[test]
    fn cross_fade_clamps_at_last_frame_without_wrapping() {
        let px = render_center_with_frame(1.5);
        assert_eq!(px[0], 0, "no red from a wrap to frame 0: {px:?}");
        assert!(px[2] > 200, "pure last frame, got {px:?}");
    }

    /// Integer frame positions (RandomFrame assigns whole frames) take the
    /// single-sample path — exactly one frame's color, no blending.
    #[test]
    fn integer_frame_draws_single_frame_unblended() {
        let px = render_center_with_frame(1.0);
        assert_eq!(px[0], 0, "{px:?}");
        assert!(px[2] > 200, "{px:?}");
        let px0 = render_center_with_frame(0.0);
        assert!(px0[0] > 200, "{px0:?}");
        assert_eq!(px0[2], 0, "{px0:?}");
    }

    /// `spritetrail` stretches the quad along the particle's velocity:
    /// a fast horizontal particle must paint a footprint wider than tall
    /// (its quad's long axis lies along X), unlike the square default.
    #[test]
    fn spritetrail_stretches_quad_along_velocity() {
        let json = r#"{
            "maxcount": 1,
            "emitter": [{"name":"box","rate":0}],
            "renderer": [{"name":"spritetrail","length":1.0,"maxlength":10}]
        }"#;
        let config: ParticleConfig = serde_json::from_str(json).expect("should parse");
        let sys = ParticleSystem::from_config(&config, [0.0, 0.0], None);
        let mut p = make_particle(50.0, 50.0, 5.0, 0.0);
        p.vx = 8.0; // trail = clamp(8 * 1.0, 0, 10) = 8 -> half-height 40 along X
        let mut sys = sys;
        sys.particles.push(p);

        let tex = RgbaImage::from_pixel(4, 4, image::Rgba([255, 255, 255, 255]));
        let sprite = ParticleSprite::single(tex);
        let mut canvas = RgbaImage::new(100, 100);
        sys.render_onto(&mut canvas, Some(&sprite), [0.0, 0.0]);

        let painted_x = canvas.get_pixel(80, 50)[3] > 0; // 30px right of center
        let painted_y = canvas.get_pixel(50, 80)[3] > 0; // 30px below center
        assert!(painted_x, "trail should extend along +X (velocity)");
        assert!(!painted_y, "trail should stay narrow across velocity");
    }

    /// Name matching must be exact: `turbulentvelocityrandom` declared
    /// before `velocityrandom` (and `angularmovement` before `movement`)
    /// must not shadow them — the old substring matcher (`contains`)
    /// read the turbulent initializer's fields as the plain velocity
    /// range and the angular operator's (absent) gravity as movement's.
    #[test]
    fn similarly_named_initializers_do_not_shadow_each_other() {
        let json = r#"{
            "maxcount": 4,
            "emitter": [{"name":"box","rate":1000}],
            "initializer": [
                {"id":1,"name":"turbulentvelocityrandom","speedmin":500,"speedmax":900},
                {"id":2,"name":"velocityrandom","min":"3 3 0","max":"3 3 0"}
            ],
            "operator": [
                {"id":1,"name":"angularmovement","drag":0.5},
                {"id":2,"name":"movement","gravity":"0 -7 0"}
            ]
        }"#;
        let config: ParticleConfig = serde_json::from_str(json).expect("should parse");
        let sys = ParticleSystem::from_config(&config, [0.0, 0.0], None);
        assert_eq!(sys.velocity_min, Some([3.0, 3.0, 0.0]));
        assert_eq!(sys.velocity_max, Some([3.0, 3.0, 0.0]));
        // Preset vectors apply unflipped in our y-down space: the reference
        // flips y into its y-up sim space and flips again at projection, so
        // the net screen-space direction equals the raw preset value.
        assert_eq!(sys.gravity[1], -7.0);
    }

    /// A config with no `alpharandom` initializer at all must stay a no-op
    /// (full alpha, matching `ParticleInstance::alpha`'s `1.0` default) —
    /// the common case, and the one that must not regress.
    #[test]
    fn no_alpharandom_initializer_defaults_to_full_alpha() {
        let config: ParticleConfig =
            serde_json::from_str(r#"{"maxcount":1,"emitter":[{"name":"box","rate":1000}]}"#)
                .expect("should parse");
        let mut sys = ParticleSystem::from_config(&config, [0.0, 0.0], None);
        sys.step(0.001);
        assert_eq!(sys.particles[0].initial_alpha, 1.0);
    }

    /// `alpharandom` (e.g. fog1.json's `"min":0.15,"max":0.2`) must actually
    /// constrain spawn alpha to that range instead of always spawning at
    /// full opacity — this was silently ignored before.
    #[test]
    fn alpharandom_constrains_spawn_alpha_to_its_range() {
        let json = r#"{
            "maxcount": 20,
            "emitter": [{"name":"box","rate":1000}],
            "initializer": [{"id":1,"name":"alpharandom","min":0.15,"max":0.2}]
        }"#;
        let config: ParticleConfig = serde_json::from_str(json).expect("should parse");
        let mut sys = ParticleSystem::from_config(&config, [0.0, 0.0], None);
        sys.step(0.05); // spawn several at once (rate 1000 * 0.05s = 50, capped at maxcount)
        assert!(!sys.particles.is_empty());
        for p in &sys.particles {
            assert!(
                (0.15..=0.2).contains(&p.initial_alpha),
                "expected initial_alpha in [0.15, 0.2], got {}",
                p.initial_alpha
            );
            assert_eq!(
                p.alpha, p.initial_alpha,
                "no alphafade — alpha should equal initial_alpha before any fade operator runs"
            );
        }
    }

    /// An instance-override alpha multiplier must apply exactly once (not
    /// squared) even when combined with `alpharandom`.
    #[test]
    fn alpha_override_multiplier_applies_once_not_twice() {
        let json = r#"{
            "maxcount": 1,
            "emitter": [{"name":"box","rate":1000}],
            "initializer": [{"id":1,"name":"alpharandom","min":1.0,"max":1.0}]
        }"#;
        let config: ParticleConfig = serde_json::from_str(json).expect("should parse");
        let overrides = InstanceOverride {
            alpha: Some(0.5),
            ..Default::default()
        };
        let mut sys = ParticleSystem::from_config(&config, [0.0, 0.0], Some(&overrides));
        sys.step(0.01);
        assert_eq!(sys.particles[0].initial_alpha, 0.5);
        sys.step(0.01);
        // No alphafade/alphachange configured, so the plain lifetime-based
        // death-fade fallback (`initial_alpha * life/max_life`) applies —
        // barely distinguishable from 0.5 this soon after spawn, but not
        // bit-exact. The point of this assertion is that it's nowhere near
        // 0.5*0.5=0.25 (the double-multiply bug), not exact equality.
        assert!(
            (sys.particles[0].alpha - 0.5).abs() < 0.01,
            "override should apply once (~0.5), not 0.5*0.5=0.25; got {}",
            sys.particles[0].alpha
        );
    }

    /// Default (no `animationmode`) sprite-sheet playback loops continuously
    /// based on lifetime position: at half life with 4 frames, the particle
    /// should be sitting on frame 2.
    #[test]
    fn sprite_frame_loops_by_lifetime_position() {
        let config: ParticleConfig =
            serde_json::from_str(r#"{"maxcount":1,"emitter":[{"name":"box","rate":1000}]}"#)
                .expect("should parse");
        let mut sys = ParticleSystem::from_config(&config, [0.0, 0.0], None);
        sys.life_min = 10.0;
        sys.life_max = 10.0;
        sys.set_sprite_frames(4, 0.0);
        sys.step(0.001); // spawn
        assert_eq!(sys.particles.len(), 1);
        sys.step(5.0); // half of a 10s life
        assert!((sys.particles[0].frame - 2.0).abs() < 0.01);
    }

    /// `animationmode: "once"` plays through the sheet across the particle's
    /// lifetime and clamps at the last frame instead of wrapping.
    #[test]
    fn sprite_frame_clamps_at_last_frame_when_animation_mode_once() {
        let config: ParticleConfig = serde_json::from_str(
            r#"{"maxcount":1,"animationmode":"once","emitter":[{"name":"box","rate":1000}]}"#,
        )
        .expect("should parse");
        let mut sys = ParticleSystem::from_config(&config, [0.0, 0.0], None);
        sys.life_min = 10.0;
        sys.life_max = 10.0;
        sys.set_sprite_frames(4, 0.0);
        sys.step(0.001); // spawn
        sys.step(9.99); // nearly the whole 10s life
        assert_eq!(sys.particles[0].frame, 3.0);
    }

    /// `animationmode: "randomframe"` assigns one frame at spawn and freezes
    /// it — a second `step()` must not change it.
    #[test]
    fn sprite_random_frame_stays_fixed_after_assignment() {
        let config: ParticleConfig = serde_json::from_str(
            r#"{"maxcount":1,"animationmode":"randomframe","emitter":[{"name":"box","rate":1000}]}"#,
        )
        .expect("should parse");
        let mut sys = ParticleSystem::from_config(&config, [0.0, 0.0], None);
        sys.life_min = 10.0;
        sys.life_max = 10.0;
        sys.set_sprite_frames(5, 0.0);
        sys.step(0.001); // spawn (frame assignment lags by one step, like other operators)
        sys.step(0.001); // assigns a random frame
        let frame = sys.particles[0].frame;
        assert!((0.0..5.0).contains(&frame));
        sys.step(1.0);
        assert_eq!(sys.particles[0].frame, frame);
    }

    /// A sprite with only one frame should never advance `p.frame` at all —
    /// `render_onto` always draws frame 0 for it regardless.
    #[test]
    fn single_frame_sprite_leaves_frame_at_spawn_sentinel() {
        let config: ParticleConfig =
            serde_json::from_str(r#"{"maxcount":1,"emitter":[{"name":"box","rate":1000}]}"#)
                .expect("should parse");
        let mut sys = ParticleSystem::from_config(&config, [0.0, 0.0], None);
        sys.life_min = 10.0;
        sys.life_max = 10.0;
        sys.set_sprite_frames(1, 0.0);
        sys.step(0.001);
        sys.step(5.0);
        assert_eq!(sys.particles[0].frame, -1.0);
    }

    #[test]
    fn bounds_is_none_when_no_particles_alive() {
        let config: ParticleConfig =
            serde_json::from_str(r#"{"maxcount":5,"emitter":[{"name":"box","rate":0}]}"#)
                .expect("should parse");
        let sys = ParticleSystem::from_config(&config, [0.0, 0.0], None);
        assert!(sys.bounds().is_none());
    }

    #[test]
    fn bounds_covers_alive_particle_extents() {
        let json = r#"{
            "maxcount": 5,
            "emitter": [{"name":"box","rate":1,"distancemax":"0 0 0"}],
            "initializer": [
                {"id":1,"name":"lifetimerandom","min":10,"max":10},
                {"id":2,"name":"sizerandom","min":20,"max":20},
                {"id":3,"name":"velocityrandom","min":"0 0 0","max":"0 0 0"}
            ]
        }"#;
        let config: ParticleConfig = serde_json::from_str(json).expect("should parse");
        let mut sys = ParticleSystem::from_config(&config, [100.0, 100.0], None);
        sys.emitters[0].rate = 1000.0;
        sys.step(0.001);
        assert_eq!(sys.particles.len(), 1);

        let (min_x, min_y, max_x, max_y) = sys.bounds().expect("should have bounds");
        // Particle spawned at (100,100) (spread disabled via distancemax=0).
        // `sizerandom`'s value is halved at spawn (diameter → radius, see
        // the comment on that halving elsewhere in this file), so a
        // `sizerandom` of 20 gives an actual radius of 10; `bounds()` adds a
        // couple of pixels of margin on top of that.
        assert!(min_x < 100.0 - 9.0 && min_x > 100.0 - 16.0, "min_x={min_x}");
        assert!(max_x > 100.0 + 9.0 && max_x < 100.0 + 16.0, "max_x={max_x}");
        assert!(min_y < 100.0 - 9.0 && min_y > 100.0 - 16.0, "min_y={min_y}");
        assert!(max_y > 100.0 + 9.0 && max_y < 100.0 + 16.0, "max_y={max_y}");
    }

    /// `render_onto`'s `origin` param should shift particle canvas positions,
    /// letting a caller raster into a canvas sized to just the bbox instead
    /// of the full scene.
    #[test]
    fn render_onto_origin_shifts_particle_positions() {
        let json = r#"{
            "maxcount": 5,
            "emitter": [{"name":"box","rate":1}],
            "initializer": [
                {"id":1,"name":"lifetimerandom","min":10,"max":10},
                {"id":2,"name":"sizerandom","min":10,"max":10},
                {"id":3,"name":"velocityrandom","min":"0 0 0","max":"0 0 0"},
                {"id":4,"name":"alpharandom","min":1,"max":1}
            ]
        }"#;
        let config: ParticleConfig = serde_json::from_str(json).expect("should parse");
        let mut sys = ParticleSystem::from_config(&config, [50.0, 50.0], None);
        sys.emitters[0].rate = 1000.0;
        sys.step(0.001);
        assert_eq!(sys.particles.len(), 1);

        // A canvas the same size as the bbox, with origin shifted to the
        // bbox's top-left, should show the particle rendered near its
        // center rather than clipped at (0,0).
        let (min_x, min_y, max_x, max_y) = sys.bounds().unwrap();
        let w = (max_x - min_x).ceil() as u32;
        let h = (max_y - min_y).ceil() as u32;
        let mut canvas = RgbaImage::new(w, h);
        sys.render_onto(&mut canvas, None, [min_x, min_y]);

        let cx = (w / 2).min(w - 1);
        let cy = (h / 2).min(h - 1);
        assert!(
            canvas.get_pixel(cx, cy)[3] > 0,
            "expected particle to render near the shifted canvas center"
        );
    }

    #[test]
    fn oscillate_params_defaults_scale_min_from_scale_max_when_absent() {
        let op = Operator {
            name: "oscillatealpha".to_string(),
            frequencymin: Some(0.5),
            frequencymax: Some(1.0),
            scalemax: Some(10.0),
            ..Default::default()
        };
        let params = OscillateParams::from_operator(&op);
        assert_eq!(params.scale_min, 10.0);
        assert_eq!(params.scale_max, 10.0);
        assert_eq!(params.freq_min, 0.5);
        assert_eq!(params.freq_max, 1.0);
    }

    #[test]
    fn oscillate_sample_phase_covers_full_circle_when_unspecified() {
        let op = Operator {
            name: "oscillatealpha".to_string(),
            frequencymin: Some(1.0),
            frequencymax: Some(1.0),
            scalemin: Some(1.0),
            scalemax: Some(1.0),
            ..Default::default()
        };
        let params = OscillateParams::from_operator(&op);
        for _ in 0..50 {
            let s = params.sample();
            assert!((0.0..std::f32::consts::TAU).contains(&s.phase));
        }
    }

    #[test]
    fn remapvalue_sets_velocity_within_output_range() {
        let json = r#"{
            "maxcount": 20,
            "emitter": [{"name":"boxrandom","rate":1000,"distancemin":"0 0 0","distancemax":"0 0 0"}],
            "initializer": [
                {"id":1,"name":"lifetimerandom","min":100,"max":100},
                {"id":2,"name":"sizerandom","min":10,"max":10}
            ],
            "operator": [
                {"id":1,"name":"remapvalue","output":"velocity",
                 "outputrangemin":"-100 -200 0","outputrangemax":"100 -1200 1",
                 "transformfunction":"simplexnoise","transforminputscale":10}
            ]
        }"#;
        let config: ParticleConfig = serde_json::from_str(json).unwrap();
        let mut sys = ParticleSystem::from_config(&config, [0.0, 0.0], None);
        for _ in 0..10 {
            sys.step(1.0 / 30.0);
        }
        assert!(!sys.particles.is_empty(), "should have spawned particles");
        for p in &sys.particles {
            // Velocity is set (then position-integrated) into the remapped
            // range each frame; check it lands within bounds (+ a small margin
            // for the integration/other operators).
            assert!(
                (-101.0..=101.0).contains(&p.vx),
                "vx {} out of remapped x-range",
                p.vx
            );
            assert!(
                (-1201.0..=-199.0).contains(&p.vy),
                "vy {} out of remapped y-range",
                p.vy
            );
        }
    }

    /// Real content (`particles/presets/rain_screen.json`, workshop item
    /// `3564529588`) attaches *two* `remapvalue` operators — one targeting
    /// `velocity`, one `speed` — and both must take effect. The old
    /// `Option<RemapParams>` silently dropped every instance after the
    /// first; this is the regression test for the `Vec` fix.
    #[test]
    fn multiple_remapvalue_operators_all_apply() {
        let json = r#"{
            "maxcount": 20,
            "emitter": [{"name":"boxrandom","rate":1000,"distancemin":"0 0 0","distancemax":"0 0 0"}],
            "initializer": [
                {"id":1,"name":"lifetimerandom","min":100,"max":100}
            ],
            "operator": [
                {"id":1,"name":"remapvalue","output":"velocity",
                 "outputrangemin":"5 5 0","outputrangemax":"5 5 0",
                 "transformfunction":"simplexnoise","transforminputscale":10},
                {"id":2,"name":"remapvalue","output":"speed",
                 "outputrangemin":"50","outputrangemax":"50",
                 "transformfunction":"fbmnoise","transforminputscale":8}
            ]
        }"#;
        let config: ParticleConfig = serde_json::from_str(json).unwrap();
        let mut sys = ParticleSystem::from_config(&config, [0.0, 0.0], None);
        // Particles spawn at the *end* of `step()`, after the per-particle
        // update loop that applies `remapvalue` — a freshly-spawned particle
        // hasn't had it applied yet on the frame it's born (same reasoning
        // `remapvalue_sets_velocity_within_output_range` above steps
        // multiple times for), so step twice before checking.
        sys.step(1.0 / 30.0);
        sys.step(1.0 / 30.0);
        assert!(!sys.particles.is_empty(), "should have spawned particles");
        for p in &sys.particles {
            // A degenerate [5,5]x[5,5] range collapses velocity's own remap
            // to a fixed heading; the speed operator then rescales its
            // magnitude to 50 — only observable if *both* operators ran.
            let mag = (p.vx * p.vx + p.vy * p.vy).sqrt();
            assert!((mag - 50.0).abs() < 0.5, "speed {mag} not rescaled to ~50");
        }
    }

    /// The general field-to-field form: `input=lifetimefraction,
    /// operation=multiply, output=opacity` is exactly the shape the retail
    /// binary's own `alphafade`-preset builders construct internally
    /// (Ghidra report Follow-up (x), `FUN_1401bfbb0`/`FUN_1401bc4b0`) — a
    /// lifetime-driven fade-out multiplier. Checks opacity decreases
    /// monotonically as the particle ages toward its declared output range.
    #[test]
    fn remapvalue_multiply_by_lifetimefraction_fades_opacity() {
        let json = r#"{
            "maxcount": 5,
            "emitter": [{"name":"boxrandom","rate":1000,"distancemin":"0 0 0","distancemax":"0 0 0"}],
            "initializer": [
                {"id":1,"name":"lifetimerandom","min":30,"max":30},
                {"id":2,"name":"alpharandom","min":1,"max":1}
            ],
            "operator": [
                {"id":1,"name":"remapvalue","input":"lifetimefraction",
                 "operation":"multiply","output":"opacity",
                 "inputrangemin":0,"inputrangemax":1,
                 "outputrangemin":1,"outputrangemax":0,
                 "transformfunction":"none"}
            ]
        }"#;
        let config: ParticleConfig = serde_json::from_str(json).unwrap();
        let mut sys = ParticleSystem::from_config(&config, [0.0, 0.0], None);
        sys.step(1.0 / 30.0);
        assert!(!sys.particles.is_empty(), "should have spawned particles");
        let alpha_early = sys.particles[0].alpha;
        for _ in 0..20 {
            sys.step(1.0 / 30.0);
        }
        assert!(!sys.particles.is_empty(), "particle should still be alive");
        let alpha_late = sys.particles[0].alpha;
        assert!(
            alpha_late < alpha_early,
            "opacity should fade as lifetimefraction rises toward 1 (early={alpha_early}, late={alpha_late})"
        );
    }

    /// `transformfunction:"none"` with a real `input` field is a straight
    /// remap (default `operation`), no shaping — the simplest, most literal
    /// case of the general form, targeting `size` this time.
    #[test]
    fn remapvalue_remaps_size_from_maxlifetime() {
        let json = r#"{
            "maxcount": 5,
            "emitter": [{"name":"boxrandom","rate":1000,"distancemin":"0 0 0","distancemax":"0 0 0"}],
            "initializer": [
                {"id":1,"name":"lifetimerandom","min":40,"max":40},
                {"id":2,"name":"sizerandom","min":1,"max":1}
            ],
            "operator": [
                {"id":1,"name":"remapvalue","input":"maxlifetime",
                 "output":"size","transformfunction":"none",
                 "inputrangemin":0,"inputrangemax":40,
                 "outputrangemin":0,"outputrangemax":200}
            ]
        }"#;
        let config: ParticleConfig = serde_json::from_str(json).unwrap();
        let mut sys = ParticleSystem::from_config(&config, [0.0, 0.0], None);
        // See the comment in `multiple_remapvalue_operators_all_apply`: a
        // freshly-spawned particle hasn't had `remapvalue` applied yet on
        // its spawn frame.
        sys.step(1.0 / 30.0);
        sys.step(1.0 / 30.0);
        assert!(!sys.particles.is_empty(), "should have spawned particles");
        for p in &sys.particles {
            // maxlifetime=40 sits at the top of [0,40] -> size should land
            // at (or very near) the top of [0,200], overriding sizerandom's
            // spawn value of 1.
            assert!(
                (p.size - 200.0).abs() < 1.0,
                "size {} not remapped from maxlifetime",
                p.size
            );
        }
    }
}

#[cfg(test)]
mod colorn_probe {
    use super::*;

    #[test]
    fn colorn_override_tints_the_spawn_color() {
        let preset = r#"{
            "maxcount": 8,
            "emitter": [{"name": "sphererandom", "id": 1, "rate": 100, "distancemax": "1 1 1"}],
            "initializer": [
                {"id": 4, "max": "255 255 255", "min": "255 255 255", "name": "colorrandom"},
                {"id": 3, "max": 10, "min": 10, "name": "sizerandom"},
                {"id": 2, "max": 1, "min": 1, "name": "lifetimerandom"}
            ],
            "operator": []
        }"#;
        let cfg: ParticleConfig = serde_json::from_str(preset).expect("preset parses");
        let ov: InstanceOverride =
            serde_json::from_str(r#"{"colorn": "0.180 0.408 0.773", "id": null, "lifetime": 2.0}"#)
                .expect("override parses");
        assert_eq!(
            ov.colorn.as_ref().and_then(value_as_vec3_pub),
            Some([0.18, 0.408, 0.773])
        );
        let mut sys = ParticleSystem::from_config(&cfg, [0.0, 0.0], Some(&ov));
        for _ in 0..20 {
            sys.step(0.05);
        }
        let p = sys.particles.first().expect("something spawned");
        // `colorrandom` is pinned to white (min == max) so the tint is the only
        // thing acting on the colour, and every spawned particle is identical.
        // With a random base this assertion was only true for most samples: a
        // blue draw near the low end of [47,255] tints to ~36, which is *below*
        // the fixed red of 255*0.180 ≈ 46, so the suite failed at random.
        let expected = [255.0 * 0.180, 255.0 * 0.408, 255.0 * 0.773];
        for (channel, want) in expected.iter().enumerate() {
            let got = p.color[channel] as f32;
            assert!(
                (got - want).abs() <= 2.0,
                "channel {channel} is {got}, expected ~{want}: {:?}",
                p.color
            );
        }
    }
}
