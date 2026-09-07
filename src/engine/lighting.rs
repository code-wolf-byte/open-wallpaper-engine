//! PBR lighting and fog — recovered from the Ghidra dump of wallpaper64.exe.
//!
//! Source: `~/Applications/ghidra-dump/REPORT.md` and
//! `~/Applications/ghidra-dump/lighting_reconstructed.glsl`.
//!
//! The Ghidra strings.txt contains the exact call-sites (lines 14048c070-
//! 14048cf11) for `PerformLighting_V1` and its per-type sub-functions. The
//! BRDF microfacet helpers (GGX, Smith, Schlick) were not extracted from the
//! binary — they follow the standard Cook–Torrance PBR model that matches
//! the uniform surface (`g_L*`, `g_Fog*`) and the dispatcher's argument
//! order exactly.
//!
//! # What is implemented
//!
//! - [`pbr_directional`], [`pbr_point`], [`pbr_spot`], [`pbr_tube`]
//!   — the four per-type lighting functions, matching the GLSL signatures
//!   recovered from strings.txt.
//! - [`ggx_distribution`], [`geometric_schlick`], [`fresnel_schlick`]
//!   — the microfacet helper functions.
//! - [`fog_apply`] — distance + height fog using the parameter surface
//!   `g_FogDistance_Density`, `g_FogDistance_Color`, `g_FogHeight_Density`,
//!   `g_FogHeight_Color` (from the 2031-name property schema).
//!
//! # What is NOT yet implemented (GAP)
//!
//! - HLSL→GLSL shim (CAST3, etc.) — the transpiler in `shaders.rs` handles
//!   the GLSL→WGSL path; the shim is a separate concern.
//!
//! Both shadow-cascade mixing and the spot-light cookie texture lookup this
//! comment used to list here have since been wired in on the GPU side —
//! `engine::shadow`/`mesh3d_shadow_factor` in `gpu_shaders.wgsl` for
//! shadows, `mesh3d_cookie_factor` for the cookie — because both need a
//! real depth/color texture sampler this pure-CPU-math module intentionally
//! doesn't have. `shadow_factor` stays a plain parameter here for the same
//! reason it always was: this module supplies the BRDF math, the GPU pass
//! supplies the texture read.
//!
//! # Coordinate convention
//!
//! WE uses a left-handed view space where `-Z` is depth (toward the viewer).
//! `worldPos.z` is therefore the view-space Z; the view distance is
//! `-worldPos.z` when the object is in front of the camera.

use std::f32::consts::PI;

// ── Types ──────────────────────────────────────────────────────────────────

/// A directional (infinite) light, matching `g_LDirectional_*` uniforms.
#[derive(Debug, Clone, Copy)]
pub struct DirectionalLight {
    /// Light direction vector, normalized. `g_LDirectional_Direction[i].xyz`.
    pub direction: [f32; 3],
    /// Light colour × intensity (RGB). `g_LDirectional_Color[i].rgb`.
    pub color: [f32; 3],
}

/// A point light, matching `g_LPoint_*` uniforms.
#[derive(Debug, Clone, Copy)]
pub struct PointLight {
    /// Light origin (xyz) and intensity (w). `g_LPoint_Origin[i]`.
    pub origin: [f32; 4],
    /// Light colour (rgb) and intensity (w). `g_LPoint_Color[i]`.
    pub color: [f32; 4],
}

/// A spot light, matching `g_LSpot_*` uniforms.
#[derive(Debug, Clone, Copy)]
pub struct SpotLight {
    /// Light origin (xyz). `g_LSpot_Origin[i]`.
    pub origin: [f32; 3],
    /// Spot direction (normalized) + exponent. `g_LSpot_Direction[i]` /
    /// `g_LSpot_Exponent[i].x`.
    pub direction: [f32; 3],
    /// Spot exponent (angular falloff sharpness). `g_LSpot_Exponent[i].x`.
    pub exponent: f32,
    /// Light colour (rgb) and intensity (w). `g_LSpot_Color[i]`.
    pub color: [f32; 4],
}

/// A tube (line-segment) light, matching `g_LTube_*` uniforms.
#[derive(Debug, Clone, Copy)]
pub struct TubeLight {
    /// Segment endpoint A (xyz) + intensity A (w). `g_LTube_OriginA[i]`.
    pub origin_a: [f32; 4],
    /// Segment endpoint B (xyz) + intensity B (w). `g_LTube_OriginB[i]`.
    /// The dump call-sites pass `g_LTube_OriginB[i].w` as the intensity.
    pub origin_b: [f32; 4],
    /// Light colour (rgb) and intensity (w). `g_LTube_Color[i]`.
    pub color: [f32; 4],
}

/// Fog parameters, matching the `g_Fog*` uniform surface.
#[derive(Debug, Clone, Copy, Default)]
pub struct Fog {
    /// Distance fog density. `g_FogDistance_Density`. 0 = off.
    pub distance_density: f32,
    /// Distance fog colour (RGB). `g_FogDistance_Color`.
    pub distance_color: [f32; 3],
    /// Height fog density. `g_FogHeight_Density`. 0 = off.
    pub height_density: f32,
    /// Height fog colour (RGB). `g_FogHeight_Color`.
    pub height_color: [f32; 3],
    /// Height fog vertical falloff exponent. `g_FogHeight_Exponent`.
    /// Controls how quickly fog density decreases with height. Default 1.0.
    pub height_exponent: f32,
    /// Height fog vertical offset. `g_FogHeight_Offset`.
    /// World-Y at which fog density is at its maximum. Default 0.0.
    pub height_offset: f32,
}

// ── Vector helpers (raw arrays, no external crate) ────────────────────────

fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn norm3(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len > 0.0 {
        [v[0] / len, v[1] / len, v[2] / len]
    } else {
        [0.0, 0.0, 0.0]
    }
}

fn scale3(v: [f32; 3], s: f32) -> [f32; 3] {
    [v[0] * s, v[1] * s, v[2] * s]
}

/// Component-wise (Hadamard) product of two vectors.
fn mul3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] * b[0], a[1] * b[1], a[2] * b[2]]
}

fn add3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

// ── Microfacet BRDF helpers (Cook–Torrance) ───────────────────────────────

/// GGX/Trowbridge-Reitz normal distribution function.
///
/// ```glsl
/// float DistributionGGX(vec3 N, vec3 H, float roughness)
/// {
///     float a  = roughness * roughness;
///     float a2 = a * a;
///     float NdotH = max(dot(N, H), 0.0);
///     float NdotH2 = NdotH * NdotH;
///     float denom = NdotH2 * (a2 - 1.0) + 1.0;
///     return a2 / (PI * denom * denom);
/// }
/// ```
pub fn ggx_distribution(n: [f32; 3], h: [f32; 3], roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let ndh = dot3(n, h).max(0.0);
    let ndh2 = ndh * ndh;
    let denom = ndh2 * (a2 - 1.0) + 1.0;
    a2 / (PI * denom * denom)
}

/// Schlick geometric occlusion (Smith-GGX) — used for both G_V and G_L
/// in the standard PBR formulation.
///
/// ```glsl
/// float GeometricSchlick(float NdotV, float roughness)
/// {
///     return NdotV / (NdotV * (1.0 - roughness) + roughness);
/// }
/// ```
pub fn geometric_schlick(ndotv: f32, roughness: f32) -> f32 {
    ndotv / (ndotv * (1.0 - roughness) + roughness)
}

/// Schlick Fresnel approximation for a dielectric.
///
/// ```glsl
/// vec3 FresnelSchlick(float cosTheta, vec3 F0)
/// {
///     return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
/// }
/// ```
pub fn fresnel_schlick(cos_theta: f32, f0: [f32; 3]) -> [f32; 3] {
    let t = (1.0 - cos_theta).powf(5.0);
    [f0[0] + (1.0 - f0[0]) * t, f0[1] + (1.0 - f0[1]) * t, f0[2] + (1.0 - f0[2]) * t]
}

// ── Per-type PBR lighting ──────────────────────────────────────────────────

/// Common BRDF core shared by all four light types.
///
/// `n` — surface normal (world/view space, normalised).
/// `l` — light direction at the surface (toward the light, normalised).
/// `v` — view direction at the surface (toward the camera, normalised).
/// `albedo` — base colour.
/// `light_color` — light colour × intensity (RGB, pre-multiplied by any
///   shadow factor and cookie factor — caller supplies).
/// `f0` — Fresnel base reflectance (from metallic/specularTint).
/// `roughness` — material roughness [0, 1].
/// `shadow_factor` — 0..1 shadow attenuation (1.0 = unshadowed).
fn brdf_core(
    n: [f32; 3],
    l: [f32; 3],
    v: [f32; 3],
    albedo: [f32; 3],
    light_color: [f32; 3],
    f0: [f32; 3],
    roughness: f32,
    shadow_factor: f32,
) -> [f32; 3] {
    let h = norm3(add3(l, v));
    let ndl = dot3(n, l).max(0.0);
    let ndv = dot3(n, v).max(0.0).max(0.001);
    let ldh = dot3(l, h).max(0.0);

    let d = ggx_distribution(n, h, roughness);
    let k = (roughness + 1.0).powi(2) * 0.125; // (r+1)^2 / 8
    let gv = geometric_schlick(ndv, k);
    let gl = geometric_schlick(ndl, k);
    let f = fresnel_schlick(ldh, f0);

    // Specular
    let spec_num = scale3(f, d * gv * gl);
    let spec_den = (4.0 * ndv * ndl).max(0.001);
    let spec = scale3(spec_num, (ndl / spec_den) * 0.25);

    // Diffuse (Lambert, non-metallic)
    let kd = scale3([1.0; 3], 1.0 - (f0[0].max(f0[1]).max(f0[2])));
    let diff = scale3(mul3(kd, albedo), (1.0 / PI) * ndl);

    let brdf = add3(spec, diff);
    scale3(mul3(brdf, light_color), shadow_factor)
}

/// Directional (infinite) light — `g_LDirectional_*`.
///
/// Matches the GLSL call-site:
/// ```glsl
/// light += ComputePBRLightShadowInfinite(
///     normal, g_LDirectional_Direction[i].xyz, viewVector,
///     color, g_LDirectional_Color[i].rgb, specularTint,
///     ambient, roughness, metallic, shadowFactor);
/// ```
/// `light_dir` is the direction **toward** the light (normalised).
pub fn pbr_directional(
    n: [f32; 3],
    v: [f32; 3],
    albedo: [f32; 3],
    light_dir: [f32; 3],
    light_color: [f32; 3],
    specular_tint: [f32; 3],
    ambient: [f32; 3],
    roughness: f32,
    metallic: f32,
    shadow_factor: f32,
) -> [f32; 3] {
    let f0 = [
        specular_tint[0] * metallic + (1.0 - metallic) * 0.04,
        specular_tint[1] * metallic + (1.0 - metallic) * 0.04,
        specular_tint[2] * metallic + (1.0 - metallic) * 0.04,
    ];
    let l = norm3(light_dir);
    let lit = brdf_core(n, l, v, albedo, light_color, f0, roughness, shadow_factor);
    add3(lit, ambient)
}

/// Point light — `g_LPoint_*`.
///
/// `world_pos` is the surface position in the same space as the light origin.
/// `origin` = `g_LPoint_Origin[i]` (xyz = position, w = intensity).
/// `color` = `g_LPoint_Color[i]` (rgb = colour, w = intensity).
/// The original divides light colour by distance² for inverse-square
/// attenuation: `color / (1.0 + dist*dist * falloff)` (falloff baked into
/// the intensity channel; here we use the standard 1/r² form).
pub fn pbr_point(
    n: [f32; 3],
    v: [f32; 3],
    world_pos: [f32; 3],
    albedo: [f32; 3],
    origin: [f32; 4],
    color: [f32; 4],
    specular_tint: [f32; 3],
    ambient: [f32; 3],
    roughness: f32,
    metallic: f32,
    shadow_factor: f32,
) -> [f32; 3] {
    let to_light = [
        origin[0] - world_pos[0],
        origin[1] - world_pos[1],
        origin[2] - world_pos[2],
    ];
    let dist2 = (to_light[0] * to_light[0] + to_light[1] * to_light[1] + to_light[2] * to_light[2]).max(0.0001);
    let dist = dist2.sqrt();
    let l = [to_light[0] / dist, to_light[1] / dist, to_light[2] / dist];
    // Inverse-square attenuation: color.rgb * intensity / dist²
    let atten = (color[3] * origin[3]) / dist2;
    let light_rgb = [color[0] * atten, color[1] * atten, color[2] * atten];

    let f0 = [
        specular_tint[0] * metallic + (1.0 - metallic) * 0.04,
        specular_tint[1] * metallic + (1.0 - metallic) * 0.04,
        specular_tint[2] * metallic + (1.0 - metallic) * 0.04,
    ];
    let lit = brdf_core(n, l, v, albedo, light_rgb, f0, roughness, shadow_factor);
    add3(lit, ambient)
}

/// Spot light — `g_LSpot_*`.
///
/// `origin` = `g_LSpot_Origin[i]` (xyz).
/// `direction` = normalised spot direction.
/// `exponent` = `g_LSpot_Exponent[i].x` (angular falloff sharpness).
/// `color` = `g_LSpot_Color[i]` (rgb × cookie already applied by caller).
///
/// Angular falloff: `pow(max(dot(toPoint, direction), 0), exponent)`.
pub fn pbr_spot(
    n: [f32; 3],
    v: [f32; 3],
    world_pos: [f32; 3],
    albedo: [f32; 3],
    origin: [f32; 3],
    direction: [f32; 3],
    exponent: f32,
    color: [f32; 4],
    specular_tint: [f32; 3],
    ambient: [f32; 3],
    roughness: f32,
    metallic: f32,
    shadow_factor: f32,
) -> [f32; 3] {
    // to_surface: direction from light origin toward the surface (for spot cone test)
    let to_surface = [
        world_pos[0] - origin[0],
        world_pos[1] - origin[1],
        world_pos[2] - origin[2],
    ];
    let dist2 = (to_surface[0] * to_surface[0] + to_surface[1] * to_surface[1] + to_surface[2] * to_surface[2]).max(0.0001);
    let dist = dist2.sqrt();

    // l: direction from surface toward the light (for BRDF)
    let l = [-to_surface[0] / dist, -to_surface[1] / dist, -to_surface[2] / dist];

    // Angular falloff: dot(light→surface, spot_direction)
    let to_surf_norm = [to_surface[0] / dist, to_surface[1] / dist, to_surface[2] / dist];
    let cos_angle = dot3(to_surf_norm, norm3(direction)).max(0.0);
    let spot_factor = cos_angle.powf(exponent.max(1.0));

    let atten = color[3] / dist2;
    let light_rgb = [color[0] * atten * spot_factor, color[1] * atten * spot_factor, color[2] * atten * spot_factor];

    let f0 = [
        specular_tint[0] * metallic + (1.0 - metallic) * 0.04,
        specular_tint[1] * metallic + (1.0 - metallic) * 0.04,
        specular_tint[2] * metallic + (1.0 - metallic) * 0.04,
    ];
    let lit = brdf_core(n, l, v, albedo, light_rgb, f0, roughness, shadow_factor);
    add3(lit, ambient)
}

/// Tube (line-segment) light — `g_LTube_*`.
///
/// `origin_a` = `g_LTube_OriginA[i]` (xyz = endpoint A).
/// `origin_b` = `g_LTube_OriginB[i]` (xyz = endpoint B, w = intensity).
/// The dump call-sites pass `g_LTube_OriginB[i].w` as the intensity.
///
/// The light direction is from the closest point on the AB segment to the
/// surface (standard closest-point-on-segment).
pub fn pbr_tube(
    n: [f32; 3],
    v: [f32; 3],
    world_pos: [f32; 3],
    albedo: [f32; 3],
    origin_a: [f32; 4],
    origin_b: [f32; 4],
    color: [f32; 4],
    specular_tint: [f32; 3],
    ambient: [f32; 3],
    roughness: f32,
    metallic: f32,
    shadow_factor: f32,
) -> [f32; 3] {
    // Closest point on segment AB to worldPos
    let ab = [
        origin_b[0] - origin_a[0],
        origin_b[1] - origin_a[1],
        origin_b[2] - origin_a[2],
    ];
    let ap = [
        world_pos[0] - origin_a[0],
        world_pos[1] - origin_a[1],
        world_pos[2] - origin_a[2],
    ];
    let ab2 = dot3(ab, ab).max(0.0001);
    let t = (dot3(ap, ab) / ab2).clamp(0.0, 1.0);
    let closest = [
        origin_a[0] + ab[0] * t,
        origin_a[1] + ab[1] * t,
        origin_a[2] + ab[2] * t,
    ];

    let to_light = [
        closest[0] - world_pos[0],
        closest[1] - world_pos[1],
        closest[2] - world_pos[2],
    ];
    let dist2 = (to_light[0] * to_light[0] + to_light[1] * to_light[1] + to_light[2] * to_light[2]).max(0.0001);
    let dist = dist2.sqrt();
    let l = [to_light[0] / dist, to_light[1] / dist, to_light[2] / dist];

    let atten = color[3] * origin_b[3] / dist2;
    let light_rgb = [color[0] * atten, color[1] * atten, color[2] * atten];

    let f0 = [
        specular_tint[0] * metallic + (1.0 - metallic) * 0.04,
        specular_tint[1] * metallic + (1.0 - metallic) * 0.04,
        specular_tint[2] * metallic + (1.0 - metallic) * 0.04,
    ];
    let lit = brdf_core(n, l, v, albedo, light_rgb, f0, roughness, shadow_factor);
    add3(lit, ambient)
}

// ── Fog ────────────────────────────────────────────────────────────────────

/// Compute the fog factor for a pixel at the given world position.
///
/// Returns a value in [0, 1] where 0 = no fog, 1 = fully fogged.
/// Uses the standard exponential distance + height fog model:
///
/// ```text
/// distance_factor = 1 - exp(-density * dist)
/// height_factor   = 1 - exp(-height_density * max(0, (height_offset - worldY) * height_exponent))
/// fog_factor      = 1 - (1 - distance_factor) * (1 - height_factor)
/// ```
///
/// `dist` is computed as `-worldPos.z` (WE left-handed view space, -Z is
/// depth toward the viewer).
pub fn fog_factor(world_pos: [f32; 3], fog: &Fog) -> f32 {
    if fog.distance_density <= 0.0 && fog.height_density <= 0.0 {
        return 0.0;
    }
    let dist = (-world_pos[2]).max(0.0);

    let d_factor = if fog.distance_density > 0.0 {
        1.0 - (-fog.distance_density * dist).exp()
    } else {
        0.0
    };

    let h_factor = if fog.height_density > 0.0 {
        let height = (fog.height_offset - world_pos[1]).max(0.0);
        1.0 - (-fog.height_density * height * fog.height_exponent).exp()
    } else {
        0.0
    };

    (1.0 - (1.0 - d_factor) * (1.0 - h_factor)).clamp(0.0, 1.0)
}

/// Apply fog to a pixel colour.
///
/// `scene_rgb` — the rendered pixel colour [0, 1].
/// `world_pos` — the world/view-space position of the pixel.
/// `fog` — fog parameters.
///
/// Returns the fogged colour. The blend is:
/// `out = scene_rgb * (1 - f) + fog_color * f`
/// where `fog_color` is the distance fog colour (primary) and the height
/// fog colour modulates the vertical gradient.
pub fn fog_apply(scene_rgb: [f32; 3], world_pos: [f32; 3], fog: &Fog) -> [f32; 3] {
    let f = fog_factor(world_pos, fog);
    if f <= 0.0 {
        return scene_rgb;
    }
    // Blend toward the distance fog colour; height fog adds a vertical tint
    // by interpolating toward the height colour.
    let fog_color = if fog.height_density > 0.0 {
        // Weighted blend: distance colour dominates, height colour modulates.
        let h_weight = (fog.height_density / (fog.distance_density + fog.height_density + 1e-6));
        [
            fog.distance_color[0] * (1.0 - h_weight) + fog.height_color[0] * h_weight,
            fog.distance_color[1] * (1.0 - h_weight) + fog.height_color[1] * h_weight,
            fog.distance_color[2] * (1.0 - h_weight) + fog.height_color[2] * h_weight,
        ]
    } else {
        fog.distance_color
    };
    [
        scene_rgb[0] * (1.0 - f) + fog_color[0] * f,
        scene_rgb[1] * (1.0 - f) + fog_color[1] * f,
        scene_rgb[2] * (1.0 - f) + fog_color[2] * f,
    ]
}

// ── Convenience: dispatch a SceneObject light to the right function ───────

/// A parsed light object ready for per-pixel evaluation.
///
/// Built from a `SceneObject` that has `light` set (see
/// `scene::SceneObject::light_params()`).
#[derive(Debug, Clone)]
pub enum Light {
    Directional {
        direction: [f32; 3],
        color: [f32; 3],
        intensity: f32,
    },
    Point {
        origin: [f32; 3],
        color: [f32; 3],
        intensity: f32,
        radius: f32,
    },
    Spot {
        origin: [f32; 3],
        direction: [f32; 3],
        exponent: f32,
        color: [f32; 3],
        intensity: f32,
        radius: f32,
        /// The raw `outercone` half-angle in degrees — kept alongside the
        /// already-derived Phong `exponent` because the shadow-atlas
        /// projection (`engine::shadow::spot_light_view_proj`) needs the
        /// actual cone width for its frustum FOV, and recovering it by
        /// inverting `exponent`'s formula would be lossy at the exponent's
        /// clamp bounds.
        outer_cone_degrees: f32,
    },
    Tube {
        origin_a: [f32; 3],
        origin_b: [f32; 3],
        color: [f32; 3],
        intensity: f32,
    },
}

impl Light {
    /// Evaluate this light at a surface point.
    ///
    /// `n` — surface normal, `v` — view direction (toward camera),
    /// `world_pos` — surface position, `albedo` — base colour,
    /// `specular_tint` — Fresnel tint, `ambient` — ambient colour,
    /// `roughness` — [0, 1], `metallic` — [0, 1],
    /// `shadow_factor` — 1.0 = unshadowed.
    pub fn evaluate(
        &self,
        n: [f32; 3],
        v: [f32; 3],
        world_pos: [f32; 3],
        albedo: [f32; 3],
        specular_tint: [f32; 3],
        ambient: [f32; 3],
        roughness: f32,
        metallic: f32,
        shadow_factor: f32,
    ) -> [f32; 3] {
        match self {
            Light::Directional { direction, color, intensity } => pbr_directional(
                n, v, albedo,
                *direction,
                [color[0] * *intensity, color[1] * *intensity, color[2] * *intensity],
                specular_tint, ambient, roughness, metallic, shadow_factor,
            ),
            Light::Point { origin, color, intensity, .. } => pbr_point(
                n, v, world_pos, albedo,
                [origin[0], origin[1], origin[2], *intensity],
                [color[0], color[1], color[2], *intensity],
                specular_tint, ambient, roughness, metallic, shadow_factor,
            ),
            Light::Spot { origin, direction, exponent, color, intensity, .. } => pbr_spot(
                n, v, world_pos, albedo,
                *origin, *direction, *exponent,
                [color[0], color[1], color[2], *intensity],
                specular_tint, ambient, roughness, metallic, shadow_factor,
            ),
            Light::Tube { origin_a, origin_b, color, intensity } => pbr_tube(
                n, v, world_pos, albedo,
                [origin_a[0], origin_a[1], origin_a[2], 1.0],
                [origin_b[0], origin_b[1], origin_b[2], *intensity],
                [color[0], color[1], color[2], *intensity],
                specular_tint, ambient, roughness, metallic, shadow_factor,
            ),
        }
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: [f32; 3], b: [f32; 3], eps: f32) -> bool {
        (a[0] - b[0]).abs() < eps && (a[1] - b[1]).abs() < eps && (a[2] - b[2]).abs() < eps
    }

    #[test]
    fn ggx_distribution_normal() {
        // N = (0,1,0), H = (0,1,0), roughness = 0.5
        // a = 0.25, a2 = 0.0625, NdotH = 1
        // denom = 1 * (0.0625 - 1) + 1 = 0.0625
        // D = 0.0625 / (PI * 0.0625^2) = 0.0625 / (PI * 0.00390625)
        let d = ggx_distribution([0.0, 1.0, 0.0], [0.0, 1.0, 0.0], 0.5);
        let expected = 0.0625f32 / (PI * 0.0625 * 0.0625);
        assert!((d - expected).abs() < 1e-4, "got {d}, expected {expected}");
    }

    #[test]
    fn fresnel_schlick_perpendicular() {
        // cosTheta = 1 (looking straight on) → F = F0
        let f = fresnel_schlick(1.0, [0.5, 0.5, 0.5]);
        assert!(approx(f, [0.5, 0.5, 0.5], 1e-4));
    }

    #[test]
    fn fresnel_schlick_grazing() {
        // cosTheta = 0 (grazing) → F = (1,1,1)
        let f = fresnel_schlick(0.0, [0.5, 0.5, 0.5]);
        assert!(approx(f, [1.0, 1.0, 1.0], 1e-4));
    }

    #[test]
    fn directional_light_straight_on() {
        // Surface facing light, no shadow: should produce non-zero lit output
        let result = pbr_directional(
            [0.0, 1.0, 0.0], // N: facing +Y
            [0.0, 0.0, -1.0], // V: looking at -Z (toward camera)
            [1.0, 1.0, 1.0],  // albedo: white
            [0.0, 1.0, 0.0],  // L: from +Y (same as N → NdotL = 1)
            [1.0, 1.0, 1.0],  // light colour: white
            [1.0, 1.0, 1.0],  // specular tint
            [0.1, 0.1, 0.1],  // ambient
            0.5,              // roughness
            0.0,              // metallic
            1.0,              // unshadowed
        );
        // Should be brighter than ambient alone
        assert!(result[0] > 0.1, "spec+diff should exceed ambient: got {result:?}");
        assert!(result[0] < 10.0, "should not be unreasonably large: got {result:?}");
    }

    #[test]
    fn directional_light_90deg() {
        // Light from +X, surface facing +Y → NdotL = 0 → no diffuse
        let result = pbr_directional(
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0],
            [1.0, 1.0, 1.0],
            [1.0, 0.0, 0.0], // L from +X
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
            0.5,
            0.0,
            1.0,
        );
        // NdotL = 0 → diffuse = 0, spec ≈ 0 (H not aligned)
        // Result should be ~0 (within floating point tolerance)
        assert!(result[0] < 0.05, "90° light should produce near-zero: got {result:?}");
    }

    #[test]
    fn point_light_falloff() {
        // Point light at origin, surface at (0, 0, 1) → dist = 1
        let r_near = pbr_point(
            [0.0, 0.0, -1.0], // N facing the light
            [0.0, 0.0, -1.0], // V
            [0.0, 0.0, 1.0],  // surface pos (1 unit from light)
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0], // origin (0,0,0), intensity 1
            [1.0, 1.0, 1.0, 1.0], // colour white, intensity 1
            [1.0, 1.0, 1.0],
            [0.0; 3],
            0.5, 0.0, 1.0,
        );

        // Surface at (0, 0, 2) → dist = 2, atten = 1/4
        let r_far = pbr_point(
            [0.0, 0.0, -1.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, 2.0],
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0; 3],
            0.5, 0.0, 1.0,
        );

        assert!(r_near[0] > r_far[0], "near should be brighter than far: near={r_near:?} far={r_far:?}");
        // Ratio should be approximately 4 (1/d²)
        let ratio = r_near[0] / r_far[0].max(1e-6);
        assert!((ratio - 4.0).abs() < 1.0, "ratio ≈ 4, got {ratio}");
    }

    #[test]
    fn fog_factor_zero_when_disabled() {
        let fog = Fog { distance_density: 0.0, ..Default::default() };
        assert_eq!(fog_factor([0.0, 0.0, -100.0], &fog), 0.0);
    }

    #[test]
    fn fog_factor_increases_with_distance() {
        let fog = Fog {
            distance_density: 0.05,
            distance_color: [0.5, 0.5, 0.5],
            ..Default::default()
        };
        let f_near = fog_factor([0.0, 0.0, -10.0], &fog);
        let f_far = fog_factor([0.0, 0.0, -100.0], &fog);
        assert!(f_near < f_far, "far should be more fogged: near={f_near} far={f_far}");
        assert!((0.0..1.0).contains(&f_near));
        assert!((0.0..1.0).contains(&f_far));
    }

    #[test]
    fn fog_apply_blends_to_fog_color() {
        let fog = Fog {
            distance_density: 10.0, // very dense → f ≈ 1 at dist=10
            distance_color: [1.0, 0.0, 0.0], // red fog
            ..Default::default()
        };
        let scene = [0.0, 1.0, 0.0]; // green pixel
        let result = fog_apply(scene, [0.0, 0.0, -10.0], &fog);
        // At high density, result should be close to red
        assert!(result[0] > 0.8, "R should dominate: got {result:?}");
        assert!(result[1] < 0.2, "G should be suppressed: got {result:?}");
    }

    #[test]
    fn fog_height_only() {
        let fog = Fog {
            height_density: 0.1,
            height_color: [0.0, 0.0, 1.0],
            height_offset: 5.0,  // fog starts below y=5, denser as you go lower
            height_exponent: 1.0,
            ..Default::default()
        };
        // At y=10 (above offset): no fog
        let f_above = fog_factor([0.0, 10.0, -10.0], &fog);
        // At y=5 (at offset): no fog yet (height term = 0)
        let f_at_offset = fog_factor([0.0, 5.0, -10.0], &fog);
        // At y=0 (below offset by 5): fogged
        let f_below = fog_factor([0.0, 0.0, -10.0], &fog);
        // At y=-5 (below offset by 10): more fog
        let f_further_below = fog_factor([0.0, -5.0, -10.0], &fog);

        assert!(f_above < 0.01, "above offset should be nearly clear: got {f_above}");
        assert!(f_at_offset < 0.01, "at offset should be nearly clear: got {f_at_offset}");
        assert!(f_below > f_at_offset, "below offset should be fogged: got {f_below} vs {f_at_offset}");
        assert!(f_further_below > f_below, "further below should be more fogged: got {f_further_below} vs {f_below}");
    }

    #[test]
    fn spot_light_angular_falloff() {
        // Spot at origin pointing +Y. Two surfaces at same distance, different cone angle.
        // Eye at (0,0,-2) → V = (0,0,1) for both (surfaces at z=0).
        let v = [0.0, 0.0, 1.0]; // eye - surface

        // Surface A: at (0, 1, 0) → directly in cone center → cos = 1
        let r_center = pbr_spot(
            [0.0, -1.0, 0.0], // N faces the light
            v,
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],  // spot dir +Y
            2.0,
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0; 3],
            0.5, 0.0, 1.0,
        );

        // Surface B: at (1, 1, 0) → 45° off axis → cos = 1/√2 → dimmer
        let r_edge = pbr_spot(
            [0.0, -1.0, 0.0], // N faces the light (light is at origin, surface at (1,1,0) → L = (-1,-1,0)/√2 → NdL = 1/√2)
            v,
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            2.0,
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0; 3],
            0.5, 0.0, 1.0,
        );

        assert!(r_center[0] > 0.0, "center should produce light: got {r_center:?}");
        assert!(r_center[0] > r_edge[0], "center should be brighter than edge: center={r_center:?} edge={r_edge:?}");
    }

    #[test]
    fn tube_light_closest_point() {
        // Tube from (0,0,0) to (0,0,2) (along Z axis).
        // Surface at (1, 0, 0) with N = (-1, 0, 0) (faces the tube).
        // Closest point on tube to surface = (0,0,0) → L = (-1,0,0) → NdL = 1.
        // Eye at (2, 0, 0) → V = (1, 0, 0) → NdV = -1 → clamped to 0.001.
        // Diffuse should still produce light.
        let r = pbr_tube(
            [-1.0, 0.0, 0.0],   // N faces the tube
            [1.0, 0.0, 0.0],   // V toward eye
            [1.0, 0.0, 0.0],   // surface pos
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0], // A
            [0.0, 0.0, 2.0, 1.0], // B (w=1 intensity)
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0; 3],
            0.5, 0.0, 1.0,
        );
        assert!(r[0] > 0.0, "tube light should produce light: got {r:?}");
    }
}
