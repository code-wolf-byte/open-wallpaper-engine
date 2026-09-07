// ── Shared output struct ──────────────────────────────────────────────────────

struct VsOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

// ── Vertex-quad shader (layer composite) ─────────────────────────────────────
// Vertices carry pre-computed NDC positions and UVs.
// Position is already in NDC space (computed on CPU from origin/size/scale/angle).

@vertex
fn vs_quad(@location(0) pos: vec2<f32>, @location(1) uv: vec2<f32>) -> VsOutput {
    var out: VsOutput;
    out.position = vec4(pos, 0.0, 1.0);
    out.uv = uv;
    return out;
}

// ── WE dynamic-effect vertex shader ─────────────────────────────────────────
// Default vertex shader for translated WE GLSL fragment shaders that declare
// (vec4 v_TexCoord, vec2 v_Scroll) as their varyings. Fullscreen triangle.

struct WeOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) v_TexCoord: vec4<f32>,
    @location(1) v_Scroll: vec2<f32>,
}

@vertex
fn vs_we_effect(@builtin(vertex_index) vi: u32) -> WeOutput {
    var out: WeOutput;
    let x = f32(i32(vi & 1u)) * 4.0 - 1.0;
    let y = f32(i32(vi >> 1u)) * 4.0 - 1.0;
    out.position = vec4(x, y, 0.0, 1.0);
    let u = (x + 1.0) * 0.5;
    let v = (1.0 - y) * 0.5;
    out.v_TexCoord = vec4(u, v, u, v);
    out.v_Scroll = vec2(u, v);
    return out;
}

// ── Fullscreen triangle vertex shader (effects) ───────────────────────────────
// Used for effect passes that fill the entire layer render target.

@vertex
fn vs_fullscreen(@builtin(vertex_index) vi: u32) -> VsOutput {
    var out: VsOutput;
    // Oversized triangle that covers the full viewport.
    // vi=0: (-1,-1), vi=1: (3,-1), vi=2: (-1,3)
    let x = f32(i32(vi & 1u)) * 4.0 - 1.0;
    let y = f32(i32(vi >> 1u)) * 4.0 - 1.0;
    out.position = vec4(x, y, 0.0, 1.0);
    // UV: (0,0)=top-left, (1,1)=bottom-right — matches wgpu texture convention.
    // NDC y=1 (top) → v=0; NDC y=-1 (bottom) → v=1.
    out.uv = vec2((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return out;
}

// ── Shared bindings ──────────────────────────────────────────────────────────

@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var src_sampler: sampler;

// ── Passthrough (compositing / layer blit) ───────────────────────────────────

// CompositeParams places one scene object's quad and carries its per-frame
// blending state. `uv_offset` holds camera dynamics (shake + parallax);
// `rect` is (center_ndc.xy, half_size_ndc.xy) for the quad vertices.
// `angle` is the object's z-rotation in radians (WE `angles.z`); `aspect`
// (scene width/height) lets the vertex shader undo the anisotropic NDC scale
// before rotating so the quad rotates rigidly instead of shearing.
struct CompositeParams {
    opacity: f32,
    mode: i32,                      // bytes 4-7: WE colorBlendMode (0 = normal alpha blend)
    uv_offset: vec2<f32>,           // byte 8 (vec2 alignment=8)
    color: vec3<f32>,               // byte 16 (vec3 alignment=16)
    angle: f32,                     // byte 28 (fills the vec3's std140 tail)
    rect: vec4<f32>,                // byte 32: center.xy, half_extent.xy (NDC)
    aspect: f32,                    // byte 48: scene width / height
    // byte 52: 4 bytes padding (vec2 below needs 8-byte alignment).
    resolution: vec2<f32>,          // byte 56: scene size in pixels, for
                                     // converting a fragment's screen position
                                     // to full-scene UV when reading
                                     // `dest_copy_tex` (a full-scene texture) —
                                     // `uv` itself is *local* to this quad's own
                                     // rect, which only coincides with full-scene
                                     // UV for a fullscreen quad.
}
@group(0) @binding(2) var<uniform> composite: CompositeParams;
@group(0) @binding(3) var dest_copy_tex: texture_2d<f32>;
// Further extra slots (same white-1x1 dummy fallback as slot 0): hardcoded
// effect kernels read their opacity masks from whichever slot the real WE
// material assigns (waterripple/tint/opacity/spin: slot 1 -> binding 3;
// pulse: slot 2 -> binding 4; shake: slot 3 -> binding 5).
@group(0) @binding(4) var extra_tex1: texture_2d<f32>;
@group(0) @binding(5) var extra_tex2: texture_2d<f32>;

// Quad vertex shader for object composites: two triangles covering the
// object's rect, generated from vertex_index (no vertex buffer).
@vertex
fn vs_composite_quad(@builtin(vertex_index) vi: u32) -> VsOutput {
    // Corner order: (-1,-1) (1,-1) (-1,1) | (1,-1) (1,1) (-1,1)
    var corners = array<vec2<f32>, 6>(
        vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(-1.0, 1.0),
        vec2(1.0, -1.0), vec2(1.0, 1.0), vec2(-1.0, 1.0),
    );
    let corner = corners[vi];
    let he = corner * composite.rect.zw;
    // Rotate in isotropic (pixel-shaped) space: undo the NDC aspect scale on
    // one axis, rotate, then reapply it — otherwise a rotation would shear
    // non-square scenes. At angle=0 this reduces exactly to `he`.
    // NDC is Y-up while `composite.angle` is derived assuming Y-down pixel
    // space (matching the CPU compositor's convention), so negate sin here.
    let c = cos(composite.angle);
    let s = -sin(composite.angle);
    let offset = vec2(
        c * he.x - s * he.y / composite.aspect,
        s * he.x * composite.aspect + c * he.y,
    );
    var out: VsOutput;
    out.position = vec4(composite.rect.xy + offset, 0.0, 1.0);
    // NDC y=+1 (top) → v=0.
    out.uv = vec2(corner.x * 0.5 + 0.5, 0.5 - corner.y * 0.5);
    return out;
}

@fragment
fn fs_composite(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    let s = textureSample(src_tex, src_sampler, uv - composite.uv_offset);
    return vec4(s.rgb * composite.color, s.a * composite.opacity);
}

// ── Photoshop-style colorBlendMode compositing ───────────────────────────────
// Mirrors WE_COMMON_BLENDING_H / ApplyBlending in shaders/transpiler.rs and
// engine/blend.rs (the CPU compositor's copy of the same table). Reads the
// destination (already-composited scene) via `dest_copy_tex` since wgpu can't
// read+write the same attachment in one pass.

fn rgb_to_hsl(c: vec3<f32>) -> vec3<f32> {
    let fmin = min(min(c.r, c.g), c.b);
    let fmax = max(max(c.r, c.g), c.b);
    let delta = fmax - fmin;
    let l = (fmax + fmin) / 2.0;
    if (delta == 0.0) {
        return vec3(0.0, 0.0, l);
    }
    var s: f32;
    if (l < 0.5) {
        s = delta / (fmax + fmin);
    } else {
        s = delta / (2.0 - fmax - fmin);
    }
    let d_r = (((fmax - c.r) / 6.0) + (delta / 2.0)) / delta;
    let d_g = (((fmax - c.g) / 6.0) + (delta / 2.0)) / delta;
    let d_b = (((fmax - c.b) / 6.0) + (delta / 2.0)) / delta;
    var h: f32;
    if (c.r == fmax) {
        h = d_b - d_g;
    } else if (c.g == fmax) {
        h = (1.0 / 3.0) + d_r - d_b;
    } else {
        h = (2.0 / 3.0) + d_g - d_r;
    }
    if (h < 0.0) {
        h = h + 1.0;
    } else if (h > 1.0) {
        h = h - 1.0;
    }
    return vec3(h, s, l);
}

fn hue_to_rgb(f1: f32, f2: f32, hue_in: f32) -> f32 {
    var hue = hue_in;
    if (hue < 0.0) {
        hue = hue + 1.0;
    } else if (hue > 1.0) {
        hue = hue - 1.0;
    }
    if (6.0 * hue < 1.0) {
        return f1 + (f2 - f1) * 6.0 * hue;
    }
    if (2.0 * hue < 1.0) {
        return f2;
    }
    if (3.0 * hue < 2.0) {
        return f1 + (f2 - f1) * ((2.0 / 3.0) - hue) * 6.0;
    }
    return f1;
}

fn hsl_to_rgb(hsl: vec3<f32>) -> vec3<f32> {
    if (hsl.y == 0.0) {
        return vec3(hsl.z);
    }
    var f2: f32;
    if (hsl.z < 0.5) {
        f2 = hsl.z * (1.0 + hsl.y);
    } else {
        f2 = (hsl.z + hsl.y) - (hsl.y * hsl.z);
    }
    let f1 = 2.0 * hsl.z - f2;
    return vec3(
        hue_to_rgb(f1, f2, hsl.x + 1.0 / 3.0),
        hue_to_rgb(f1, f2, hsl.x),
        hue_to_rgb(f1, f2, hsl.x - 1.0 / 3.0),
    );
}

fn screen_f(base: f32, blend: f32) -> f32 { return 1.0 - ((1.0 - base) * (1.0 - blend)); }
fn overlay_f(base: f32, blend: f32) -> f32 {
    if (base < 0.5) { return 2.0 * base * blend; }
    return 1.0 - 2.0 * (1.0 - base) * (1.0 - blend);
}
fn soft_light_f(base: f32, blend: f32) -> f32 {
    if (blend < 0.5) { return 2.0 * base * blend + base * base * (1.0 - 2.0 * blend); }
    return sqrt(base) * (2.0 * blend - 1.0) + 2.0 * base * (1.0 - blend);
}
fn color_dodge_f(base: f32, blend: f32) -> f32 {
    if (blend == 1.0) { return blend; }
    return min(base / (1.0 - blend), 1.0);
}
fn color_burn_f(base: f32, blend: f32) -> f32 {
    if (blend == 0.0) { return blend; }
    return max(1.0 - ((1.0 - base) / blend), 0.0);
}

fn blend_rgb(mode: i32, base: vec3<f32>, blend: vec3<f32>) -> vec3<f32> {
    if (mode == 1) { return min(blend, base); }
    if (mode == 2) { return base * blend; }
    if (mode == 3) {
        return vec3(color_burn_f(base.r, blend.r), color_burn_f(base.g, blend.g), color_burn_f(base.b, blend.b));
    }
    if (mode == 4) { return max(base + blend - vec3(1.0), vec3(0.0)); }
    if (mode == 6) { return max(blend, base); }
    if (mode == 7) {
        return vec3(screen_f(base.r, blend.r), screen_f(base.g, blend.g), screen_f(base.b, blend.b));
    }
    if (mode == 8) {
        return vec3(color_dodge_f(base.r, blend.r), color_dodge_f(base.g, blend.g), color_dodge_f(base.b, blend.b));
    }
    if (mode == 9) { return min(base + blend, vec3(1.0)); }
    if (mode == 11) {
        return vec3(overlay_f(base.r, blend.r), overlay_f(base.g, blend.g), overlay_f(base.b, blend.b));
    }
    if (mode == 12) {
        return vec3(soft_light_f(base.r, blend.r), soft_light_f(base.g, blend.g), soft_light_f(base.b, blend.b));
    }
    if (mode == 18) { return abs(base - blend); }
    if (mode == 26) {
        let b = rgb_to_hsl(base);
        return hsl_to_rgb(vec3(rgb_to_hsl(blend).x, b.y, b.z));
    }
    if (mode == 27) {
        let b = rgb_to_hsl(base);
        return hsl_to_rgb(vec3(b.x, rgb_to_hsl(blend).y, b.z));
    }
    if (mode == 28) {
        let b = rgb_to_hsl(blend);
        return hsl_to_rgb(vec3(b.x, b.y, rgb_to_hsl(base).z));
    }
    if (mode == 29) {
        let b = rgb_to_hsl(base);
        return hsl_to_rgb(vec3(b.x, b.y, rgb_to_hsl(blend).z));
    }
    // WE extensions past the Photoshop table (common_blending.h):
    // 30 = BlendTint (max channel of base, colored), 31 = plain add,
    // 32 = glow (base + base*blend). The caller's mix-by-alpha wrapper
    // matches ApplyBlending's own mix/add-by-opacity forms.
    if (mode == 30) {
        let peak = max(base.r, max(base.g, base.b));
        return vec3(peak) * blend;
    }
    if (mode == 31) { return base + blend; }
    if (mode == 32) { return base + base * blend; }
    return blend;
}

@fragment
fn fs_composite_blend(
    @builtin(position) frag_coord: vec4<f32>,
    @location(0) uv: vec2<f32>,
) -> @location(0) vec4<f32> {
    let s = textureSample(src_tex, src_sampler, uv - composite.uv_offset);
    // `dest_copy_tex` is a full-scene snapshot, but `uv` is *local* to this
    // quad's own rect — only identical to full-scene UV when the quad
    // happens to cover the whole screen. Use the fragment's actual
    // screen-space position instead so a partial-screen quad (a particle
    // system's tight bounding box, a small rotated/scaled layer, ...) reads
    // the scene content actually behind *it*, not a stretched sample of
    // the scene's top-left corner.
    let screen_uv = frag_coord.xy / composite.resolution;
    let dest = textureSample(dest_copy_tex, src_sampler, screen_uv);
    let src_rgb = s.rgb * composite.color;
    // Mode 100 (ours, above WE's real 0-32 blend range — 30/31/32 are
    // BlendTint/add/glow in common_blending.h): pure premultiplied add for
    // additive particle layers. The CPU rasterizer already accumulated
    // `src * src_a` per particle (GL_SRC_ALPHA/GL_ONE), so the buffer's RGB
    // is ready to add as-is — weighting by its alpha here would apply each
    // particle's alpha twice.
    if (composite.mode == 100) {
        return vec4(min(dest.rgb + src_rgb, vec3(1.0)), dest.a);
    }
    let src_a = s.a * composite.opacity;
    let blended = blend_rgb(composite.mode, dest.rgb, src_rgb);
    let out_rgb = mix(dest.rgb, blended, src_a);
    return vec4(out_rgb, dest.a);
}

// ── Blit (surface presentation / FBO up- and down-sampling) ──────────────────

struct BlitParams {
    uv_scale: vec2<f32>,
    uv_offset: vec2<f32>,
}
@group(0) @binding(2) var<uniform> blit: BlitParams;

@fragment
fn fs_blit(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    return textureSample(src_tex, src_sampler, uv * blit.uv_scale + blit.uv_offset);
}

// ── Bloom (scene-level chain) ─────────────────────────────────────────────────
// Mirrors the reference's `downsample_quarter_bloom` / `downsample_eighth_blur_v`
// / `blur_h_bloom` / `combine` shaders exactly (the linux-wallpaperengine port
// synthesizes bloom from these four real WE utility shaders via a hidden
// fullscreen effect object — see CScene.cpp/WallpaperApplication.cpp). All
// texel offsets use the *scene's* texel size (WE's `g_TexelSize` is always
// scene-relative, never the destination FBO's own size — CPass.cpp).

struct BloomThresholdParams {
    // Scene texel size (1/sceneWidth, 1/sceneHeight) — the four taps sample
    // at +-texel like the reference vertex shader's v_TexCoord[0..3].
    texel: vec2<f32>,
    threshold: f32,
    strength: f32,
}
@group(1) @binding(0) var<uniform> bloom_threshold: BloomThresholdParams;

@fragment
fn fs_bloom_threshold(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    let t = bloom_threshold.texel;
    var albedo = textureSample(src_tex, src_sampler, uv - t).rgb
        + textureSample(src_tex, src_sampler, uv + t).rgb
        + textureSample(src_tex, src_sampler, uv + vec2(-t.x, t.y)).rgb
        + textureSample(src_tex, src_sampler, uv + vec2(t.x, -t.y)).rgb;
    albedo = albedo * 0.25;

    let scale = max(max(albedo.x, albedo.y), albedo.z);
    albedo = albedo * saturate(scale - bloom_threshold.threshold);

    // http://stackoverflow.com/a/34183839 (saturation boost, sat=1.0)
    let grayscale = dot(vec3(0.2989, 0.5870, 0.1140), albedo);
    albedo = -grayscale + albedo * 2.0;

    return vec4(max(vec3(0.0), albedo * bloom_threshold.strength), 1.0);
}

struct BlurParams {
    // Blur-direction step: scene texel size * 8 along x or y (WE's
    // `localTexel = g_TexelSize.{x,y} * 8.0`), zero in the other axis.
    dir_texel: vec2<f32>,
    _p1: f32, _p2: f32,
}
@group(1) @binding(0) var<uniform> blur: BlurParams;

@fragment
fn fs_blur9(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    // 13-tap kernel, weights and offsets verbatim from
    // downsample_eighth_blur_v.frag / blur_h_bloom.frag.
    let weights = array<f32, 13>(
        0.006299, 0.017298, 0.039533, 0.075189, 0.119007, 0.156756, 0.171834,
        0.156756, 0.119007, 0.075189, 0.039533, 0.017298, 0.006299,
    );
    var acc = vec3(0.0);
    for (var i = 0; i < 13; i = i + 1) {
        let o = blur.dir_texel * f32(i - 6);
        acc = acc + textureSample(src_tex, src_sampler, uv + o).rgb * weights[i];
    }
    return vec4(acc, 1.0);
}

@fragment
fn fs_bloom_combine(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    // dest_copy_tex (binding 3, "extra slot 0") carries the pre-bloom scene copy.
    let scene = textureSample(dest_copy_tex, src_sampler, uv).rgb;
    let bloom = textureSample(src_tex, src_sampler, uv).rgb;
    return vec4(scene + bloom, 1.0);
}

// ── Pulse ────────────────────────────────────────────────────────────────────

// g_AudioSpectrum*: 16 left, 16 right, then the 32- and 64-band sets. Written
// once per frame from the FFT; all zero when nothing is captured.
@group(1) @binding(1) var<storage, read> audio_spectrum: array<f32>;

/// pulse.vert's CreateAudioResponse: average the selected bands, shape by
/// bounds/exponent, scale by amount. `mode` is the AUDIOPROCESSING combo —
/// 1 = left, 2 = right, 3 = both.
fn audio_response(mode: i32, fmin: f32, fmax: f32, bounds: vec2<f32>, power: f32, amount: f32) -> f32 {
    let lo = i32(clamp(min(fmin, fmax), 0.0, 15.0));
    let hi = i32(clamp(max(fmin, fmax), 0.0, 15.0));
    var total = 0.0;
    var n = 0.0;
    for (var a = lo; a <= hi; a = a + 1) {
        if (mode != 2) { total = total + audio_spectrum[a]; n = n + 1.0; }
        if (mode != 1) { total = total + audio_spectrum[16 + a]; n = n + 1.0; }
    }
    let avg = total / max(n, 1.0);
    let shaped = smoothstep(bounds.x, bounds.y, avg);
    return clamp(pow(shaped, power), 0.0, 1.0) * amount;
}

struct PulseParams {
    time: f32,
    speed: f32,
    amount: f32,
    power: f32,
    phase: f32,
    bounds_x: f32,
    bounds_y: f32,
    // BLENDMODE combo (imageblending table, default 9 = linear dodge).
    blendmode: f32,
    tint_low: vec3<f32>,
    // PULSECOLOR combo (default 1): blend rgb·tintlow toward rgb·tinthigh.
    pulse_color: f32,
    tint_high: vec3<f32>,
    // PULSEALPHA combo (default 0): multiply alpha by the pulse.
    pulse_alpha: f32,
    noise_speed: f32,
    noise_amount: f32,
    // AUDIOPROCESSING combo: 0 = time-driven, 1 = left, 2 = right, 3 = both.
    audio_mode: f32,
    audio_fmin: f32,
    audio_fmax: f32,
    // Two scalars, not a vec2: std140 would align a vec2 to 8 bytes here and
    // silently shift every field after it out from under the packer.
    audio_bounds_lo: f32,
    audio_bounds_hi: f32,
    audio_power: f32,
    audio_amount: f32,
    _p0: f32,
    _p1: f32,
    _p2: f32,
}
@group(1) @binding(0) var<uniform> pulse: PulseParams;

// pulse.frag's non-audio path: pulse = smoothstep(bounds, sin wave)·amount
// + noise(g_Texture1, slot 1 → dest_copy_tex)·noiseamount, raised to
// `power`; PULSECOLOR routes rgb·tintlow toward rgb·tinthigh through the
// authored BLENDMODE via ApplyBlending's mix-by-pulse; PULSEALPHA scales
// alpha instead/in addition. The opacity mask (slot 2 → extra_tex1) lerps
// the whole effect against the untouched sample.
@fragment
fn fs_pulse(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    let s = textureSample(src_tex, src_sampler, uv);
    let mask = textureSample(extra_tex1, src_sampler, uv).r;
    let raw = sin(pulse.time * pulse.speed + pulse.phase - 1.5708) * 0.5 + 0.5;
    // AUDIOPROCESSING != 0 replaces the time sine entirely (pulse.vert picks
    // one or the other, never both) — without this an audio-reactive layer
    // pulses on a timer and ignores what's playing.
    let am = i32(pulse.audio_mode + 0.5);
    let noise_uv = vec2(pulse.time * 0.08333333, pulse.time * 0.02777777) * pulse.noise_speed;
    let noise = textureSample(dest_copy_tex, src_sampler, noise_uv).r * pulse.noise_amount;
    var p = pow(
        smoothstep(pulse.bounds_x, pulse.bounds_y, raw) * pulse.amount + noise,
        pulse.power,
    );
    if (am != 0) {
        p = audio_response(
            am,
            pulse.audio_fmin,
            pulse.audio_fmax,
            vec2(pulse.audio_bounds_lo, pulse.audio_bounds_hi),
            pulse.audio_power,
            pulse.audio_amount,
        );
    }
    var albedo = s;
    if pulse.pulse_color > 0.5 {
        let a = s.rgb * pulse.tint_low;
        let b = s.rgb * pulse.tint_high;
        albedo = vec4(mix(a, blend_rgb(i32(pulse.blendmode), a, b), clamp(p, 0.0, 1.0)), albedo.a);
    }
    if pulse.pulse_alpha > 0.5 {
        albedo.a = albedo.a * p;
    }
    let mixed = mix(s, albedo, mask);
    return vec4(max(mixed.rgb, vec3(0.0)), mixed.a);
}

// ── Scroll ───────────────────────────────────────────────────────────────────

struct ScrollParams {
    time: f32,
    speed_x: f32,
    speed_y: f32,
    scale_x: f32,
    scale_y: f32,
    _p1: f32,
    _p2: f32,
    _p3: f32,
}
@group(1) @binding(0) var<uniform> scroll: ScrollParams;

@fragment
fn fs_scroll(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    let tc = fract((uv + vec2(scroll.speed_x, scroll.speed_y) * scroll.time) * vec2(scroll.scale_x, scroll.scale_y));
    return textureSample(src_tex, src_sampler, tc);
}

// ── Shake ────────────────────────────────────────────────────────────────────

struct ShakeParams {
    time: f32,
    speed: f32,
    strength: f32,
    // DIRECTION combo: 0 = center (offset*2-1), 1 = left (as-is), 2 = right
    // (offset-1) — shake.frag's remaps.
    direction: f32,
    friction_x: f32,
    friction_y: f32,
    // Precomputed v_Bounds from shake.vert: (bounds.x, 1/(bounds.y-bounds.x)).
    bounds_x: f32,
    bounds_y_recip: f32,
    // NOISE combo: 1 = four layered incommensurate sines instead of one.
    noise: f32,
    _p1: f32,
    _p2: f32,
    _p3: f32,
}
@group(1) @binding(0) var<uniform> shake: ShakeParams;

const TAU: f32 = 6.28318530718;

// Faithful port of shake.frag. Displacement is per-pixel: `offset(t) *
// strength^2 * flowMask`, where flowMask decodes the author-painted flow map
// (g_Texture1, slot 1 → dest_copy_tex): rg = 0.498 gray means "this pixel
// does not move" — WITHOUT a painted flow map the effect is static, which is
// why sampling an opacity mask here (the old kernel) slid entire layers
// left-right. g_Texture2 (slot 2 → extra_tex1) is a per-pixel time-offset
// phase; g_Texture3 (slot 3 → extra_tex2) an opacity mask gating the result.
@fragment
fn fs_shake(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    // White 1x1 dummy fallback gives phase 1.0 * TAU, which is congruent to
    // the authored black default's 0.0.
    let flow_phase = textureSample(extra_tex1, src_sampler, uv).r * TAU;
    let flow_colors = textureSample(dest_copy_tex, src_sampler, uv).rg;
    let flow = (flow_colors - vec2(0.498, 0.498)) * 2.0;

    var offset = 0.0;
    if shake.noise > 0.5 {
        var sines = vec4(flow_phase)
            + fract(shake.speed * shake.time / TAU * vec4(1.0, -0.16161616, 0.0083333, -0.00019841))
            * TAU;
        let csines = cos(sines);
        var sn = sin(sines) * 0.498 + vec4(0.5);
        let base = step(vec4(0.0), csines);
        sn = mix(
            vec4(1.0) - pow(max(vec4(1.0) - sn, vec4(0.0)), vec4(shake.friction_x)),
            pow(max(sn, vec4(0.0)), vec4(shake.friction_y)),
            base,
        );
        offset = dot(vec4(0.5), sn);
    } else {
        let t = shake.speed * shake.time + flow_phase;
        var o = sin(fract(t / TAU) * TAU);
        o = o * 0.498 + 0.5;
        let base = step(0.0, cos(t));
        offset = mix(
            1.0 - pow(max(1.0 - o, 0.0), shake.friction_x),
            pow(max(o, 0.0), shake.friction_y),
            base,
        );
    }
    offset = clamp((offset - shake.bounds_x) * shake.bounds_y_recip, 0.0, 1.0);

    if shake.direction < 0.5 {
        offset = offset * 2.0 - 1.0;
    } else if shake.direction >= 1.5 {
        offset = offset - 1.0;
    }

    let disp = offset * shake.strength * shake.strength * flow;
    let displaced = textureSample(src_tex, src_sampler, uv + disp);
    let undisplaced = textureSample(src_tex, src_sampler, uv);
    let mask = textureSample(extra_tex2, src_sampler, uv).r;
    return mix(undisplaced, displaced, mask);
}

// ── Tint ─────────────────────────────────────────────────────────────────────

struct TintParams {
    r: f32,
    g: f32,
    b: f32,
    alpha: f32,
    blendmode: f32,
    _p0: f32,
    _p1: f32,
    _p2: f32,
}
@group(1) @binding(0) var<uniform> tint: TintParams;

// tint.frag: rgb = ApplyBlending(BLENDMODE, rgb, tintColor, alpha·mask).
// The combo defaults to 30 = BlendTint (max channel of the base broadcast,
// colored) but authors pick any imageblending mode; mode 0 also forces
// alpha to 1 (the shader's own `#if BLENDMODE == 0` branch).
@fragment
fn fs_tint(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    let s = textureSample(src_tex, src_sampler, uv);
    let mask = textureSample(dest_copy_tex, src_sampler, uv).r;
    let mode = i32(tint.blendmode);
    let tint_rgb = vec3(tint.r, tint.g, tint.b);
    let tinted = mix(s.rgb, blend_rgb(mode, s.rgb, tint_rgb), clamp(tint.alpha * mask, 0.0, 1.0));
    var a = s.a;
    if mode == 0 {
        a = 1.0;
    }
    return vec4(tinted, a);
}

// ── Opacity ──────────────────────────────────────────────────────────────────

struct OpacityParams {
    alpha: f32,
    _p1: f32,
    _p2: f32,
    _p3: f32,
}
@group(1) @binding(0) var<uniform> opacity: OpacityParams;

@fragment
fn fs_opacity(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    let s = textureSample(src_tex, src_sampler, uv);
    let mask = textureSample(dest_copy_tex, src_sampler, uv).r;
    return vec4(s.rgb, s.a * opacity.alpha * mask);
}

// ── Water Ripple ─────────────────────────────────────────────────────────────

struct RippleParams {
    time: f32,
    strength: f32,
    speed: f32,
    scale: f32,
}
@group(1) @binding(0) var<uniform> ripple: RippleParams;

@fragment
fn fs_waterripple(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    let s1 = sin((uv.x * ripple.scale + ripple.time * ripple.speed) * 6.28318);
    let s2 = sin((uv.y * ripple.scale * 0.7 + ripple.time * ripple.speed * 0.8) * 6.28318);
    let n = s1 * s2;
    let mask = textureSample(dest_copy_tex, src_sampler, uv).r;
    let tc = uv + vec2(n, n * 0.7) * ripple.strength * ripple.strength * mask;
    return textureSample(src_tex, src_sampler, tc);
}

// ── Water Waves ──────────────────────────────────────────────────────────────

struct WavesParams {
    time: f32,
    speed: f32,
    scale: f32,
    strength: f32,
    dir_x: f32,
    dir_y: f32,
    exponent: f32,
    _p2: f32,
}
@group(1) @binding(0) var<uniform> waves: WavesParams;

// Mirrors the real waterwaves.frag: displacement along the wave direction's
// perpendicular, shaped by pow(|sin|, exponent)·sign(sin), scaled by
// strength² and the opacity mask. The mask (g_Texture1 in the real
// material) arrives in extra slot 0 — `dest_copy_tex` is just that slot's
// binding, and its white-1×1 fallback makes an unbound mask read 1.0,
// matching the shader's `#else mask = 1.0` branch.
@fragment
fn fs_waterwaves(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    let mask = textureSample(dest_copy_tex, src_sampler, uv).r;
    let dir = vec2(waves.dir_x, waves.dir_y);
    let perp = vec2(dir.y, -dir.x);
    // TIMEOFFSET (g_Texture2, slot 2 → extra_tex1): per-pixel phase, r·2π.
    // The white 1×1 fallback adds a full period — identical to the authored
    // black default's zero.
    let time_offset = textureSample(extra_tex1, src_sampler, uv).r * 6.28318530718;
    let d = waves.time * waves.speed + dot(uv, dir) * waves.scale + time_offset;
    let s = sin(d);
    let val = pow(abs(s), waves.exponent) * sign(s);
    let tc = uv + perp * val * waves.strength * waves.strength * mask;
    return textureSample(src_tex, src_sampler, tc);
}

// ── Spin ─────────────────────────────────────────────────────────────────────

struct SpinParams {
    time: f32,
    speed: f32,
    center_x: f32,
    center_y: f32,
    size: f32,
    feather: f32,
    repeat: f32,
    _p2: f32,
}
@group(1) @binding(0) var<uniform> spin: SpinParams;

// spin.frag rotates only a feathered disc of radius `size` around `center`
// (smoothstep falloff), mixed over the untouched sample and scaled by the
// opacity mask — not a whole-layer rotation.
@fragment
fn fs_spin(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    let center = vec2(spin.center_x, spin.center_y);
    let delta = uv - center;
    let angle = spin.speed * spin.time;
    let cs = vec2(cos(angle), sin(angle));
    let rotated = vec2(delta.x * cs.x - delta.y * cs.y, delta.x * cs.y + delta.y * cs.x);
    // REPEAT combo (default 1): wrap the rotated coordinate; 0 clamps.
    var tc = rotated + center;
    if spin.repeat > 0.5 {
        tc = fract(tc);
    } else {
        tc = clamp(tc, vec2(0.0), vec2(1.0));
    }
    let original = textureSample(src_tex, src_sampler, uv);
    let spun = textureSample(src_tex, src_sampler, tc);
    var mask = smoothstep(spin.size + spin.feather + 0.00001, spin.size - spin.feather, length(delta));
    mask = mask * textureSample(dest_copy_tex, src_sampler, uv).r;
    return mix(original, spun, mask);
}

// ── GPU particles ─────────────────────────────────────────────────────────────
// The CPU simulation emits pre-transformed triangle-list vertices (6 per
// sprite quad / rope sub-quad, absolute scene pixels — see particle.rs's
// GpuVertex); this pipeline replaces the old budgeted CPU rasterizer, so
// particles draw at full output resolution with hardware filtering and
// float blending (the reference draws GPU quads too). Sprite-sheet frames
// are array layers; cross-fade samples two layers (genericparticle.frag's
// SPRITESHEETBLEND) — sampled unconditionally, WGSL forbids textureSample
// in non-uniform control flow.

struct PVertex {
    pos: vec2<f32>,
    uv: vec2<f32>,
    color: vec4<f32>,
    frame_blend: vec4<f32>,
}
@group(0) @binding(0) var<storage, read> particle_verts: array<PVertex>;
@group(0) @binding(1) var particle_tex: texture_2d_array<f32>;
@group(0) @binding(2) var particle_sampler: sampler;
struct ParticleDrawParams {
    scene_size: vec2<f32>,
    overbright: f32,
    frame_count: f32,
}
@group(0) @binding(3) var<uniform> pdraw: ParticleDrawParams;

struct ParticleVsOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) frame_blend: vec2<f32>,
}

@vertex
fn vs_particles(@builtin(vertex_index) vi: u32) -> ParticleVsOut {
    let v = particle_verts[vi];
    var out: ParticleVsOut;
    // Scene pixels (y-down) -> NDC.
    out.position = vec4(
        v.pos.x / pdraw.scene_size.x * 2.0 - 1.0,
        1.0 - v.pos.y / pdraw.scene_size.y * 2.0,
        0.0,
        1.0,
    );
    out.uv = v.uv;
    out.color = v.color;
    out.frame_blend = v.frame_blend.xy;
    return out;
}

@fragment
fn fs_particles(in: ParticleVsOut) -> @location(0) vec4<f32> {
    let frames = max(pdraw.frame_count, 1.0);
    let f0 = i32(clamp(in.frame_blend.x, 0.0, frames - 1.0));
    let f1 = min(f0 + 1, i32(frames) - 1);
    let s0 = textureSample(particle_tex, particle_sampler, in.uv, f0);
    let s1 = textureSample(particle_tex, particle_sampler, in.uv, f1);
    let s = mix(s0, s1, in.frame_blend.y);
    // genericparticle.frag: albedo * vertex color, rgb *= g_Overbright.
    return vec4(s.rgb * in.color.rgb * pdraw.overbright, s.a * in.color.a);
}

// ── Static 3D meshes ──────────────────────────────────────────────────────────
// Real geometry (spheres, skyboxes, cylinders) from a scene object's `model`
// .mdl, drawn through the scene camera with a depth buffer — unlike every other
// pipeline here, which composites flat quads. See `engine::mesh3d`.

const MESH3D_MAX_LIGHTS: u32 = 8u;
// Must match `engine::shadow::MAX_SHADOW_LIGHTS` and
// `MESH3D_MAX_SHADOW_LIGHTS` in gpu_renderer.rs exactly — no shared constant
// across the Rust/WGSL boundary, same caveat as MESH3D_MAX_LIGHTS.
const MESH3D_MAX_SHADOW_LIGHTS: u32 = 4u;
// Depth-compare bias, tuned to avoid acne without excessive peter-panning.
// WE's real bias isn't recoverable from the binary (see the Ghidra report's
// shadow-mapping follow-up) — this is a standard, documented stand-in value.
const MESH3D_SHADOW_BIAS: f32 = 0.003;

struct Mesh3dTransform {
    mvp: mat4x4<f32>,
    // World→view (no projection): gives the fragment shader a view-space
    // position, matching the space `engine::lighting`'s BRDF math (and the
    // original's own `PerformLighting_V1`) operates in.
    model_view: mat4x4<f32>,
    // view_rotation · object_rotation · diag(1/scale) — see
    // `camera3d::PerspectiveCamera::normal_view`. Only the upper-left 3x3
    // matters; normals are transformed with w=0 so the translation column
    // (borrowed from `model_view`'s own layout) never contributes.
    normal_view: mat4x4<f32>,
    // Object→world, no view/projection. Shadow-map lookups need world
    // position: each shadow-casting light's view-projection is naturally
    // built in world space (see `engine::shadow::point_light_view_proj`),
    // unlike everything else here, which stays in the main camera's view
    // space.
    model: mat4x4<f32>,
}

// Scene-wide lighting state, shared by every mesh3d draw call this frame (one
// buffer, not per-mesh) — see `GpuSceneRenderer::build_mesh3d_lighting`.
struct Mesh3dLighting {
    // x = 1.0 when the scene has any `light` objects. Real Workshop content
    // is ~100% unlit (see gpu_renderer.rs's ambientcolor note) so this stays
    // 0.0 for virtually every scene, and the fragment shader falls back to
    // the plain textured look every mesh3d scene already had — this pass is
    // additive, never a regression for existing unlit content.
    // y = the cookie-texture light's index + 1 (0.0 = no light has a
    // cookie — real content never sets `usecookie`, so this is 0.0 for
    // ~100% of the corpus; see `mesh3d_cookie_factor`).
    flags: vec4<f32>,
    ambient: vec4<f32>,       // rgb; a unused
    fog_distance: vec4<f32>,  // rgb = color, a = density (0 = off)
    fog_height: vec4<f32>,    // rgb = color, a = density (0 = off)
    fog_extra: vec4<f32>,     // x = height exponent, y = height offset
    // Fixed-size light array (scene.json's `light` objects resolve to
    // either Point or Spot — see `Scene::lights`). Unused slots have
    // light_color.a == 0 and are skipped; no separate count field needed.
    // light_pos[i].w doubles as a shadow-atlas slot selector: 0.0 = this
    // light casts no shadow, else `slot + 1` — see
    // `gpu_renderer.rs::mesh3d_lighting_bytes`.
    light_pos: array<vec4<f32>, MESH3D_MAX_LIGHTS>,
    light_color: array<vec4<f32>, MESH3D_MAX_LIGHTS>,
    // Spot-light shaping, view-space: xyz = facing direction, w = Phong-style
    // falloff exponent (see `mesh3d_spot_factor`). A zero-length xyz means
    // "this light is a Point, not a Spot" — cheaper than a separate flag,
    // and a real spot direction is never exactly zero (it's always a
    // rotated unit vector — see `Scene::lights`).
    light_spot: array<vec4<f32>, MESH3D_MAX_LIGHTS>,
    // Per-shadow-slot world-space view-projection (wgpu depth range) and
    // atlas sub-rect (u, v, scale_u, scale_v) — see `engine::shadow`. This
    // is the shared-atlas addressing scheme recovered from the original
    // binary's `g_LFeature_ShadowProjectionTransform` (a vec4 sub-rect, not
    // a second matrix — see the Ghidra report's shadow-mapping follow-up).
    shadow_view_proj: array<mat4x4<f32>, MESH3D_MAX_SHADOW_LIGHTS>,
    shadow_uv_rect: array<vec4<f32>, MESH3D_MAX_SHADOW_LIGHTS>,
    // World-space view-projection for the cookie-texture light named by
    // `flags.y` — meaningless (never read) when `flags.y == 0.0`. A single
    // slot, not one per light: real content never uses more than zero
    // cookie textures at once (see `mesh3d_cookie_factor`'s doc comment),
    // matching the same "one active thing, not a per-light array" scoping
    // `build_volumetrics` already uses for volumetric lights.
    cookie_view_proj: mat4x4<f32>,
}

@group(0) @binding(0) var<uniform> mesh3d_xform: Mesh3dTransform;
@group(0) @binding(1) var mesh3d_tex: texture_2d<f32>;
@group(0) @binding(2) var mesh3d_sampler: sampler;
@group(0) @binding(3) var<uniform> mesh3d_lighting: Mesh3dLighting;
@group(0) @binding(4) var shadow_atlas: texture_depth_2d;
@group(0) @binding(5) var shadow_sampler: sampler_comparison;
// Spot-light cookie texture (`usecookie`/`cookie` — see
// `SceneObject::cookie_texture_path`) — a harmless 1x1 white dummy, bound
// the same way `shadow_atlas` is when there's nothing real to show, for the
// ~100% of real content that never sets `usecookie`.
@group(0) @binding(6) var cookie_tex: texture_2d<f32>;
@group(0) @binding(7) var cookie_sampler: sampler;

struct Mesh3dVsOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) view_pos: vec3<f32>,
    @location(2) view_normal: vec3<f32>,
    @location(3) world_pos: vec3<f32>,
};

@vertex
fn vs_mesh3d(
    @location(0) pos: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) normal: vec3<f32>,
) -> Mesh3dVsOut {
    var out: Mesh3dVsOut;
    out.position = mesh3d_xform.mvp * vec4(pos, 1.0);
    out.uv = uv;
    out.view_pos = (mesh3d_xform.model_view * vec4(pos, 1.0)).xyz;
    out.view_normal = (mesh3d_xform.normal_view * vec4(normal, 0.0)).xyz;
    out.world_pos = (mesh3d_xform.model * vec4(pos, 1.0)).xyz;
    return out;
}

// ── Shadow-caster depth pass ──────────────────────────────────────────────
// Renders every mesh3d object's geometry, position-only, into one tile of
// the shared shadow atlas — see `engine::shadow` and
// `GpuSceneInstance::build_shadow_atlas`. No fragment stage: depth-only.

@group(0) @binding(0) var<uniform> shadow_mvp: mat4x4<f32>;

@vertex
fn vs_shadow_depth(@location(0) pos: vec3<f32>) -> @builtin(position) vec4<f32> {
    return shadow_mvp * vec4(pos, 1.0);
}

// ── Perspective-projected 2D quads ────────────────────────────────────────
// True quadrilateral silhouette for image layers in 3D perspective scenes —
// `project_quad_ndc` (gpu_renderer.rs) collapses the 4 projected corners to
// their axis-aligned bounding box, which renders an off-axis or
// steeply-angled quad as a rectangle instead of the true trapezoid (see the
// Ghidra report's quad-perspective-warp finding). This pipeline draws the
// real 4 corners instead.
//
// Scoped simplification: UV mapping across the quad is bilinear, not
// hardware perspective-correct (`corners` are already-divided NDC, not
// clip-space with a real w) — fixes the silhouette, which is what's
// visibly wrong; full perspective-correct texture mapping would need
// passing undivided clip-space corners instead, a further improvement.
// Normal blending only (mode 0) — `gpu_renderer.rs` falls back to the
// existing rect-based composite path for any other blend mode on a
// perspective quad, a narrow and rare combination not worth a second
// pipeline variant per blend mode.

struct Quad3DParams {
    corners: array<vec4<f32>, 4>, // xy = NDC (already divided); zw unused
    color: vec4<f32>,             // rgb = tint, a = opacity
}
@group(0) @binding(0) var<uniform> quad3d: Quad3DParams;
@group(0) @binding(1) var quad3d_tex: texture_2d<f32>;
@group(0) @binding(2) var quad3d_sampler: sampler;

struct Quad3DVsOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_quad3d(@builtin(vertex_index) vi: u32) -> Quad3DVsOut {
    // Two triangles from the 4 corners (bottom-left, bottom-right,
    // top-right, top-left — see `quad_world_corners` in gpu_renderer.rs).
    var idx = array<u32, 6>(0u, 1u, 2u, 0u, 2u, 3u);
    var uvs = array<vec2<f32>, 4>(
        vec2(0.0, 1.0), vec2(1.0, 1.0), vec2(1.0, 0.0), vec2(0.0, 0.0),
    );
    let corner_i = idx[vi];
    var out: Quad3DVsOut;
    out.position = vec4(quad3d.corners[corner_i].xy, 0.0, 1.0);
    out.uv = uvs[corner_i];
    return out;
}

@fragment
fn fs_quad3d(in: Quad3DVsOut) -> @location(0) vec4<f32> {
    let s = textureSample(quad3d_tex, quad3d_sampler, in.uv);
    return vec4(s.rgb * quad3d.color.rgb, s.a * quad3d.color.a);
}

const MESH3D_ROUGHNESS: f32 = 0.6;
const MESH3D_F0: vec3<f32> = vec3<f32>(0.04, 0.04, 0.04);
const MESH3D_PI: f32 = 3.14159265;

// Cook-Torrance BRDF core (GGX distribution + Smith-Schlick geometry +
// Schlick Fresnel), ported term-for-term from `engine::lighting::brdf_core`
// (that Rust version stays the CPU-side tested reference — WGSL can't call
// into it, so this is a transcription, not a shared implementation).
// Mesh3d materials don't expose per-material roughness/metallic yet, so this
// uses fixed dielectric defaults rather than a full material PBR surface.
fn mesh3d_brdf(n: vec3<f32>, l: vec3<f32>, v: vec3<f32>, albedo: vec3<f32>, light_color: vec3<f32>) -> vec3<f32> {
    let h = normalize(l + v);
    let ndl = max(dot(n, l), 0.0);
    let ndv = max(dot(n, v), 0.001);
    let ndh = max(dot(n, h), 0.0);
    let ldh = max(dot(l, h), 0.0);

    let a = MESH3D_ROUGHNESS * MESH3D_ROUGHNESS;
    let a2 = a * a;
    let denom = ndh * ndh * (a2 - 1.0) + 1.0;
    let d = a2 / (MESH3D_PI * denom * denom);

    let k = (MESH3D_ROUGHNESS + 1.0) * (MESH3D_ROUGHNESS + 1.0) * 0.125;
    let gv = ndv / (ndv * (1.0 - k) + k);
    let gl = ndl / (ndl * (1.0 - k) + k);

    let f = MESH3D_F0 + (vec3<f32>(1.0) - MESH3D_F0) * pow(1.0 - ldh, 5.0);

    let spec_den = max(4.0 * ndv * ndl, 0.001);
    let spec = f * (d * gv * gl) * (ndl / spec_den) * 0.25;

    let kd = 1.0 - max(max(MESH3D_F0.r, MESH3D_F0.g), MESH3D_F0.b);
    let diff = vec3<f32>(kd) * albedo * (ndl / MESH3D_PI);

    return (spec + diff) * light_color;
}

// `shadow_w` is `light_pos[i].w` — 0.0 means this light casts no shadow
// (the common case: real Workshop content never sets `castshadow`, per the
// Ghidra report). Otherwise `shadow_w - 1` is the slot into
// `shadow_view_proj`/`shadow_uv_rect`. Returns 1.0 = fully lit, 0.0 = fully
// shadowed, matching `engine::lighting`'s `shadow_factor` convention exactly
// — this is the first real (non-1.0-constant) caller of that parameter.
fn mesh3d_shadow_factor(shadow_w: f32, world_pos: vec3<f32>) -> f32 {
    if (shadow_w < 0.5) {
        return 1.0;
    }
    let slot = i32(shadow_w) - 1;
    let light_clip = mesh3d_lighting.shadow_view_proj[slot] * vec4(world_pos, 1.0);
    if (light_clip.w <= 0.0) {
        // Behind the light's own near plane — outside its projection
        // entirely, so there's nothing to compare against. Treat as
        // unshadowed rather than sampling garbage.
        return 1.0;
    }
    let ndc = light_clip.xy / light_clip.w;
    let depth = light_clip.z / light_clip.w;
    // NDC is Y-up in [-1,1]; texture V is Y-down in [0,1] — the standard
    // rasterizer flip, same relationship every other screen-space pass in
    // this file relies on implicitly via the rasterizer itself.
    let local_uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    let uv_rect = mesh3d_lighting.shadow_uv_rect[slot];
    let atlas_uv = uv_rect.xy + local_uv * uv_rect.zw;
    return textureSampleCompareLevel(shadow_atlas, shadow_sampler, atlas_uv, depth - MESH3D_SHADOW_BIAS);
}

// Spot angular falloff — matches `engine::lighting::pbr_spot`'s formula
// exactly: the cosine of the angle between the light's own facing direction
// and the direction from the light toward the surface, raised to a falloff
// exponent (Phong-style, not WE's real recovered `smoothstep(inner, outer,
// cosAngle)` — see the Ghidra report's light-schema follow-up for why).
// `light_spot.xyz` zero-length means "this light is a Point, not a Spot"
// (see the `Mesh3dLighting` field doc) — returns 1.0 so a point light's
// contribution is left untouched.
fn mesh3d_spot_factor(light_spot: vec4<f32>, l: vec3<f32>) -> f32 {
    let dir_len2 = dot(light_spot.xyz, light_spot.xyz);
    if (dir_len2 < 0.25) {
        return 1.0;
    }
    let spot_dir = light_spot.xyz * inverseSqrt(dir_len2);
    // `l` points from the surface toward the light; the light's own facing
    // direction points the other way, hence `-l`.
    let cos_angle = max(dot(-l, spot_dir), 0.0);
    return pow(cos_angle, max(light_spot.w, 1.0));
}

// Spot-light cookie texture — the one piece of `PerformLighting_V1`'s real
// per-pixel formula this report recovered verbatim as plaintext GLSL (see
// the report's spot-light section): `colorCookie = texSample2D(COOKIE,
// projectedCoords.xy).rgb`, multiplied into the light's color. `i` is this
// light's slot in `mesh3d_lighting.light_pos`/`light_color`; only the one
// light named by `flags.y` (real content: at most one, ever — see
// `Scene::lights`' cookie_path scoping) samples the real texture. Outside
// the light's own projected frustum (or behind it) contributes vec3(0) —
// darkness, matching a cookie texture's implicit "nothing outside the
// beam" clamp-to-border behavior — everything else passes through
// unmodified (vec3(1)).
fn mesh3d_cookie_factor(i: u32, world_pos: vec3<f32>) -> vec3<f32> {
    if (i32(mesh3d_lighting.flags.y) - 1 != i32(i)) {
        return vec3<f32>(1.0);
    }
    let clip = mesh3d_lighting.cookie_view_proj * vec4(world_pos, 1.0);
    if (clip.w <= 0.0) {
        return vec3<f32>(0.0);
    }
    let ndc = clip.xy / clip.w;
    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
        return vec3<f32>(0.0);
    }
    return textureSampleLevel(cookie_tex, cookie_sampler, uv, 0.0).rgb;
}

@fragment
fn fs_mesh3d(in: Mesh3dVsOut) -> @location(0) vec4<f32> {
    let albedo = textureSample(mesh3d_tex, mesh3d_sampler, in.uv);
    if (mesh3d_lighting.flags.x < 0.5) {
        return albedo;
    }

    let n = normalize(in.view_normal);
    // The camera sits at the view-space origin by construction, so the
    // direction back to it from any point is simply `-view_pos`.
    let v = normalize(-in.view_pos);

    var lit = mesh3d_lighting.ambient.rgb * albedo.rgb;
    for (var i = 0u; i < MESH3D_MAX_LIGHTS; i = i + 1u) {
        let lc = mesh3d_lighting.light_color[i];
        if (lc.a <= 0.0) {
            continue;
        }
        let shadow_w = mesh3d_lighting.light_pos[i].w;
        var l: vec3<f32>;
        var atten: f32;
        // Sentinels ordered most- to least-negative; real shadow_w values
        // (0, or a positive shadow-atlas slot+1) never go negative, so none
        // of these can collide with a real point/spot shadow slot.
        if (shadow_w < -1.5) {
            // Tube (line-segment) light: `light_pos[i].xyz`/`light_spot[i].xyz`
            // carry endpoints A/B (view space) — light from the closest
            // point on segment AB to this surface point, mirroring
            // `engine::lighting::pbr_tube`'s CPU-side math exactly.
            let origin_a = mesh3d_lighting.light_pos[i].xyz;
            let origin_b = mesh3d_lighting.light_spot[i].xyz;
            let ab = origin_b - origin_a;
            let ap = in.view_pos - origin_a;
            let ab2 = max(dot(ab, ab), 0.0001);
            let t = clamp(dot(ap, ab) / ab2, 0.0, 1.0);
            let closest = origin_a + ab * t;
            let to_light = closest - in.view_pos;
            let dist2 = max(dot(to_light, to_light), 0.0001);
            l = to_light * inverseSqrt(dist2);
            atten = lc.a / dist2;
        } else if (shadow_w < -0.5) {
            // Directional (infinite) light: `light_pos[i].xyz` carries a
            // pre-normalized view-space direction from the surface toward
            // the light, not a position — no distance falloff.
            l = mesh3d_lighting.light_pos[i].xyz;
            atten = lc.a;
        } else {
            let to_light = mesh3d_lighting.light_pos[i].xyz - in.view_pos;
            let dist2 = max(dot(to_light, to_light), 0.0001);
            l = to_light * inverseSqrt(dist2);
            atten = lc.a / dist2;
        }
        // `shadow_w`'s negative sentinels are already < 0.5
        // (mesh3d_shadow_factor's own "no shadow" check) — directional and
        // tube lights aren't shadow-mapped yet, same as spot. A tube's
        // `light_spot.xyz` holds endpoint B (not a direction, and not
        // reliably zero-length), so it must skip `mesh3d_spot_factor`
        // explicitly rather than relying on that function's own
        // "zero-length = not a spot" check the way Point/Directional do.
        var spot_factor = 1.0;
        var cookie = vec3<f32>(1.0);
        if (shadow_w >= -1.5) {
            spot_factor = mesh3d_spot_factor(mesh3d_lighting.light_spot[i], l);
            cookie = mesh3d_cookie_factor(i, in.world_pos);
        }
        let shadow_factor = mesh3d_shadow_factor(shadow_w, in.world_pos);
        lit = lit + mesh3d_brdf(n, l, v, albedo.rgb, lc.rgb * atten * spot_factor * cookie) * shadow_factor;
    }

    var color = lit;
    let fd = mesh3d_lighting.fog_distance;
    let fh = mesh3d_lighting.fog_height;
    if (fd.a > 0.0 || fh.a > 0.0) {
        // Matches `engine::lighting::fog_factor`/`fog_apply` exactly — see
        // that module for the derivation.
        let dist = max(-in.view_pos.z, 0.0);
        var d_factor = 0.0;
        if (fd.a > 0.0) {
            d_factor = 1.0 - exp(-fd.a * dist);
        }
        var h_factor = 0.0;
        if (fh.a > 0.0) {
            let height = max(mesh3d_lighting.fog_extra.y - in.view_pos.y, 0.0);
            h_factor = 1.0 - exp(-fh.a * height * mesh3d_lighting.fog_extra.x);
        }
        let f = clamp(1.0 - (1.0 - d_factor) * (1.0 - h_factor), 0.0, 1.0);
        var fog_color = fd.rgb;
        if (fh.a > 0.0) {
            let hw = fh.a / (fd.a + fh.a + 1e-6);
            fog_color = fd.rgb * (1.0 - hw) + fh.rgb * hw;
        }
        color = color * (1.0 - f) + fog_color * f;
    }

    return vec4<f32>(color, albedo.a);
}

// ── Volumetric light shafts ──────────────────────────────────────────────
//
// A ray-marched-against-the-shadow-map light shaft for the *first*
// `castvolumetrics` light in the scene (see `GpuSceneInstance::
// build_volumetrics`/`record_volumetrics` and the Ghidra report's
// `_rt_volumetrics*` follow-up). WE's real technique procedurally builds a
// jittered per-light cone/box mesh on the CPU and rasterizes it; the exact
// per-pixel math is unrecoverable D3D bytecode either way (same situation
// as the main PBR lighting shader — see the report's own findings), so this
// is an honest, differently-shaped approximation that reaches the same
// visual result (a light shaft respecting real shadow occlusion) through a
// simpler, GPU-standard technique: march the camera ray through world
// space, sample the shadow atlas at each step, accumulate.

struct VolumetricsParams {
    inv_view_proj: mat4x4<f32>,
    shadow_view_proj: mat4x4<f32>,
    eye: vec4<f32>,           // xyz = world camera position
    light_origin: vec4<f32>,  // xyz = world light position, w = density
    light_color: vec4<f32>,   // rgb = color * intensity, a = volumetricsexponent
    shadow_uv_rect: vec4<f32>,
    misc: vec4<f32>,          // x = 1.0 if this light also casts a shadow, y = max march range
}

@group(0) @binding(0) var<uniform> vol_params: VolumetricsParams;
@group(0) @binding(1) var vol_shadow_atlas: texture_depth_2d;
@group(0) @binding(2) var vol_shadow_sampler: sampler_comparison;

const VOLUMETRICS_STEPS: i32 = 24;
// Same self-shadowing bias concern `MESH3D_SHADOW_BIAS` addresses, kept as
// its own constant since this pass samples the same atlas from a
// completely different set of world positions (ray-march sample points,
// not surface fragments).
const VOLUMETRICS_SHADOW_BIAS: f32 = 0.003;

// Structurally the same projection→UV→compare sequence as
// `mesh3d_shadow_factor`, just against this pass's own uniform layout
// instead of `mesh3d_lighting`'s per-slot arrays.
fn volumetrics_shadow_factor(world_pos: vec3<f32>) -> f32 {
    if (vol_params.misc.x < 0.5) {
        return 1.0;
    }
    let light_clip = vol_params.shadow_view_proj * vec4(world_pos, 1.0);
    if (light_clip.w <= 0.0) {
        return 1.0;
    }
    let ndc = light_clip.xy / light_clip.w;
    let depth = light_clip.z / light_clip.w;
    let local_uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    let atlas_uv = vol_params.shadow_uv_rect.xy + local_uv * vol_params.shadow_uv_rect.zw;
    return textureSampleCompareLevel(vol_shadow_atlas, vol_shadow_sampler, atlas_uv, depth - VOLUMETRICS_SHADOW_BIAS);
}

@fragment
fn fs_volumetrics(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    // Unproject this pixel's near/far NDC points back to world space (same
    // GL-clip convention `camera3d::mat4_inverse`'s caller uses — see
    // `PerspectiveCamera::view_proj_raw`) to reconstruct the camera ray,
    // rather than needing separately-uploaded basis vectors.
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let near4 = vol_params.inv_view_proj * vec4(ndc, -1.0, 1.0);
    let far4 = vol_params.inv_view_proj * vec4(ndc, 1.0, 1.0);
    let near_world = near4.xyz / near4.w;
    let far_world = far4.xyz / far4.w;
    let ray_dir = normalize(far_world - near_world);

    let max_range = vol_params.misc.y;
    let step_size = max_range / f32(VOLUMETRICS_STEPS);
    let density = vol_params.light_origin.w;
    // `exponent` (WE's `volumetricsexponent`) reinterpreted as an
    // exponential extinction rate — how fast in-scattered light decays with
    // distance from the light — rather than a literal power on inverse-
    // square falloff: integrating inverse-square along a ray has a
    // singularity at the light itself (falloff → ∞ as a sample's distance
    // → 0), which blows out to solid white the instant any sample lands
    // near the light. Exponential (Beer-Lambert-style) extinction is the
    // standard, numerically well-behaved choice for this kind of in-
    // scattering integral, and stays bounded ([0,1] per step) everywhere.
    let decay = max(vol_params.light_color.a, 0.05);

    var accum = 0.0;
    for (var i = 0; i < VOLUMETRICS_STEPS; i = i + 1) {
        // Mid-point sampling: softens the fixed step count's banding a
        // little without needing per-pixel jitter (no blue-noise texture
        // in this pipeline to jitter with).
        let t = (f32(i) + 0.5) * step_size;
        let p = vol_params.eye.xyz + ray_dir * t;
        let shadow = volumetrics_shadow_factor(p);
        if (shadow <= 0.0) {
            continue;
        }
        let dist_to_light = length(vol_params.light_origin.xyz - p);
        let falloff = exp(-dist_to_light * decay);
        accum = accum + shadow * falloff * step_size;
    }

    // `density` (WE's own property) and a fixed brightness constant both
    // scale the whole integral, kept outside the per-step loop so they
    // can't compound step-over-step the way multiplying inside it would.
    const VOLUMETRICS_SCALE: f32 = 0.06;
    return vec4<f32>(vol_params.light_color.rgb * accum * density * VOLUMETRICS_SCALE, 1.0);
}
