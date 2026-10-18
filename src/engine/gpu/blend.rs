//! WE `colorBlendMode` Photoshop-style blend table.
//!
//! Mirrors `ApplyBlending`/`WE_COMMON_BLENDING_H` in
//! `shaders/transpiler.rs` (the GLSL/WGSL version used by the GPU compositor),
//! for the CPU-side compositor in `render.rs`. `dest` is the pixel already on
//! the canvas, `src` is the incoming layer's own color, `opacity` is the
//! layer's own alpha at that pixel.

fn rgb_to_hsl(c: [f32; 3]) -> [f32; 3] {
    let (r, g, b) = (c[0], c[1], c[2]);
    let fmin = r.min(g).min(b);
    let fmax = r.max(g).max(b);
    let delta = fmax - fmin;
    let l = (fmax + fmin) / 2.0;
    if delta == 0.0 {
        return [0.0, 0.0, l];
    }
    let s = if l < 0.5 {
        delta / (fmax + fmin)
    } else {
        delta / (2.0 - fmax - fmin)
    };
    let d_r = (((fmax - r) / 6.0) + (delta / 2.0)) / delta;
    let d_g = (((fmax - g) / 6.0) + (delta / 2.0)) / delta;
    let d_b = (((fmax - b) / 6.0) + (delta / 2.0)) / delta;
    let mut h = if r == fmax {
        d_b - d_g
    } else if g == fmax {
        (1.0 / 3.0) + d_r - d_b
    } else {
        (2.0 / 3.0) + d_g - d_r
    };
    if h < 0.0 {
        h += 1.0;
    } else if h > 1.0 {
        h -= 1.0;
    }
    [h, s, l]
}

fn hue_to_rgb(f1: f32, f2: f32, hue: f32) -> f32 {
    let mut hue = hue;
    if hue < 0.0 {
        hue += 1.0;
    } else if hue > 1.0 {
        hue -= 1.0;
    }
    if 6.0 * hue < 1.0 {
        return f1 + (f2 - f1) * 6.0 * hue;
    }
    if 2.0 * hue < 1.0 {
        return f2;
    }
    if 3.0 * hue < 2.0 {
        return f1 + (f2 - f1) * ((2.0 / 3.0) - hue) * 6.0;
    }
    f1
}

fn hsl_to_rgb(hsl: [f32; 3]) -> [f32; 3] {
    let (h, s, l) = (hsl[0], hsl[1], hsl[2]);
    if s == 0.0 {
        return [l, l, l];
    }
    let f2 = if l < 0.5 {
        l * (1.0 + s)
    } else {
        (l + s) - (s * l)
    };
    let f1 = 2.0 * l - f2;
    [
        hue_to_rgb(f1, f2, h + 1.0 / 3.0),
        hue_to_rgb(f1, f2, h),
        hue_to_rgb(f1, f2, h - 1.0 / 3.0),
    ]
}

fn screen_f(base: f32, blend: f32) -> f32 {
    1.0 - ((1.0 - base) * (1.0 - blend))
}

fn overlay_f(base: f32, blend: f32) -> f32 {
    if base < 0.5 {
        2.0 * base * blend
    } else {
        1.0 - 2.0 * (1.0 - base) * (1.0 - blend)
    }
}

fn soft_light_f(base: f32, blend: f32) -> f32 {
    if blend < 0.5 {
        2.0 * base * blend + base * base * (1.0 - 2.0 * blend)
    } else {
        base.sqrt() * (2.0 * blend - 1.0) + 2.0 * base * (1.0 - blend)
    }
}

fn color_dodge_f(base: f32, blend: f32) -> f32 {
    if blend == 1.0 {
        blend
    } else {
        (base / (1.0 - blend)).min(1.0)
    }
}

fn color_burn_f(base: f32, blend: f32) -> f32 {
    if blend == 0.0 {
        blend
    } else {
        (1.0 - ((1.0 - base) / blend)).max(0.0)
    }
}

fn map3(base: [f32; 3], blend: [f32; 3], f: fn(f32, f32) -> f32) -> [f32; 3] {
    [
        f(base[0], blend[0]),
        f(base[1], blend[1]),
        f(base[2], blend[2]),
    ]
}

fn blend_rgb(mode: u32, base: [f32; 3], blend: [f32; 3]) -> [f32; 3] {
    match mode {
        1 => [
            blend[0].min(base[0]),
            blend[1].min(base[1]),
            blend[2].min(base[2]),
        ], // Darken
        2 => [base[0] * blend[0], base[1] * blend[1], base[2] * blend[2]], // Multiply
        3 => map3(base, blend, color_burn_f),                              // ColorBurn
        4 => [
            (base[0] + blend[0] - 1.0).max(0.0),
            (base[1] + blend[1] - 1.0).max(0.0),
            (base[2] + blend[2] - 1.0).max(0.0),
        ], // Subtract
        6 => [
            blend[0].max(base[0]),
            blend[1].max(base[1]),
            blend[2].max(base[2]),
        ], // Lighten
        7 => map3(base, blend, screen_f),                                  // Screen
        8 => map3(base, blend, color_dodge_f),                             // ColorDodge
        9 => [
            (base[0] + blend[0]).min(1.0),
            (base[1] + blend[1]).min(1.0),
            (base[2] + blend[2]).min(1.0),
        ], // Add
        11 => map3(base, blend, overlay_f),                                // Overlay
        12 => map3(base, blend, soft_light_f),                             // SoftLight
        18 => [
            (base[0] - blend[0]).abs(),
            (base[1] - blend[1]).abs(),
            (base[2] - blend[2]).abs(),
        ], // Difference
        26 => {
            let b = rgb_to_hsl(base);
            hsl_to_rgb([rgb_to_hsl(blend)[0], b[1], b[2]])
        } // Hue
        27 => {
            let b = rgb_to_hsl(base);
            hsl_to_rgb([b[0], rgb_to_hsl(blend)[1], b[2]])
        } // Saturation
        28 => {
            let b = rgb_to_hsl(blend);
            hsl_to_rgb([b[0], b[1], rgb_to_hsl(base)[2]])
        } // Color
        29 => {
            let b = rgb_to_hsl(base);
            hsl_to_rgb([b[0], b[1], rgb_to_hsl(blend)[2]])
        } // Luminosity
        _ => blend,
    }
}

/// Apply WE `colorBlendMode` (Photoshop-style ID) to a single pixel.
/// `mode == 0` (or unrecognized) is a plain replace/normal blend.
/// Inputs and output are normalized `[0,1]` linear-space RGB.
pub fn apply_blending(mode: u32, dest: [f32; 3], src: [f32; 3], opacity: f32) -> [f32; 3] {
    if mode == 0 {
        return [
            dest[0] + (src[0] - dest[0]) * opacity,
            dest[1] + (src[1] - dest[1]) * opacity,
            dest[2] + (src[2] - dest[2]) * opacity,
        ];
    }
    let blended = blend_rgb(mode, dest, src);
    [
        dest[0] + (blended[0] - dest[0]) * opacity,
        dest[1] + (blended[1] - dest[1]) * opacity,
        dest[2] + (blended[2] - dest[2]) * opacity,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multiply_matches_formula() {
        let out = apply_blending(2, [0.8, 0.5, 1.0], [0.5, 0.5, 0.5], 1.0);
        assert!((out[0] - 0.4).abs() < 1e-5);
        assert!((out[1] - 0.25).abs() < 1e-5);
        assert!((out[2] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn opacity_zero_is_identity() {
        let out = apply_blending(9, [0.2, 0.3, 0.4], [0.9, 0.9, 0.9], 0.0);
        assert_eq!(out, [0.2, 0.3, 0.4]);
    }

    #[test]
    fn normal_mode_is_alpha_blend() {
        let out = apply_blending(0, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 0.5);
        assert!((out[0] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn screen_matches_formula() {
        let out = apply_blending(7, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], 1.0);
        // 1-(1-.5)(1-.5) = 0.75
        assert!((out[0] - 0.75).abs() < 1e-5);
    }
}
