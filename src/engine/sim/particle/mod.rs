//! Wallpaper Engine particle systems: a [`ParticleSystem`] built from a
//! [`ParticleConfig`], stepped every frame, and drawn on the CPU or handed to
//! the GPU renderer as vertices.
//!
//! This file holds the runtime state types; the rest is split by role:
//! - `config`: the JSON config types and value-parsing helpers
//! - `build`: reading a config into a system
//! - `step`: the per-frame simulation and its inputs
//! - `operators`: operator parameters and their math
//! - `draw`: CPU drawing, rope geometry, GPU vertices

use image::RgbaImage;
use serde::Deserialize;

mod config;
pub use config::*;

mod operators;
pub use operators::*;

mod draw;
pub use draw::*;

mod build;
pub use build::*;

mod step;
pub use step::*;

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
    /// Latest 16-band spectrum (L/R averaged), set per frame; audio-reactive
    /// emitters scale their emission rate by it.
    audio_bands: [f32; 16],
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
    /// `turbulentvelocityrandom` initializers; each adds its own velocity.
    turbulent_velocity: Vec<TurbulentVelocityParams>,
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
    /// Set when the emitter has `audioprocessing*` — its rate follows the
    /// audio spectrum.
    audio: Option<AudioResponse>,
}

impl ParticleSystem {}

fn lerp_f32(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

// Minimal vec3 math for the turbulent/vortex ports (the reference uses glm;
// pulling in a linear-algebra crate for five one-liners isn't worth it).

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

#[cfg(test)]
mod tests;
