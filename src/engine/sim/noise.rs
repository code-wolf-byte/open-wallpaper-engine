//! Perlin/curl noise, a line-for-line port of the reference's
//! `Render/Utils/NoiseUtils.h` — the noise field drives the
//! `turbulentvelocityrandom` initializer and `turbulence` operator, so the
//! permutation table and gradient quirks (including the reference's
//! duplicated 0xD/0xF gradient cases) must match exactly for particle
//! motion to look the same.

/// Ken Perlin's reference permutation table (256 entries; lookups wrap).
const PERLIN_PERM: [u8; 256] = [
    151, 160, 137, 91, 90, 15, 131, 13, 201, 95, 96, 53, 194, 233, 7, 225, 140, 36, 103, 30, 69,
    142, 8, 99, 37, 240, 21, 10, 23, 190, 6, 148, 247, 120, 234, 75, 0, 26, 197, 62, 94, 252, 219,
    203, 117, 35, 11, 32, 57, 177, 33, 88, 237, 149, 56, 87, 174, 20, 125, 136, 171, 168, 68, 175,
    74, 165, 71, 134, 139, 48, 27, 166, 77, 146, 158, 231, 83, 111, 229, 122, 60, 211, 133, 230,
    220, 105, 92, 41, 55, 46, 245, 40, 244, 102, 143, 54, 65, 25, 63, 161, 1, 216, 80, 73, 209, 76,
    132, 187, 208, 89, 18, 169, 200, 196, 135, 130, 116, 188, 159, 86, 164, 100, 109, 198, 173,
    186, 3, 64, 52, 217, 226, 250, 124, 123, 5, 202, 38, 147, 118, 126, 255, 82, 85, 212, 207, 206,
    59, 227, 47, 16, 58, 17, 182, 189, 28, 42, 223, 183, 170, 213, 119, 248, 152, 2, 44, 154, 163,
    70, 221, 153, 101, 155, 167, 43, 172, 9, 129, 22, 39, 253, 19, 98, 108, 110, 79, 113, 224, 232,
    178, 185, 112, 104, 218, 246, 97, 228, 251, 34, 242, 193, 238, 210, 144, 12, 191, 179, 162,
    241, 81, 51, 145, 235, 249, 14, 239, 107, 49, 192, 214, 31, 181, 199, 106, 157, 184, 84, 204,
    176, 115, 121, 50, 45, 127, 4, 150, 254, 138, 236, 205, 93, 222, 114, 67, 29, 24, 72, 243, 141,
    128, 195, 78, 66, 215, 61, 156, 180,
];

/// The C++ table is duplicated to 512 entries so `PERM[A + 1]` with `A` up
/// to 255+255 wraps naturally; masking with 255 is arithmetically identical.
#[inline]
fn perm(i: usize) -> usize {
    PERLIN_PERM[i & 255] as usize
}

/// `perlinGrad` — note cases 0xD and 0xF intentionally mirror the
/// reference's (nonstandard) duplicates of 0x9 and 0xB.
fn perlin_grad(hash: usize, x: f64, y: f64, z: f64) -> f64 {
    match hash & 0xF {
        0x0 => x + y,
        0x1 => -x + y,
        0x2 => x - y,
        0x3 => -x - y,
        0x4 => x + z,
        0x5 => -x + z,
        0x6 => x - z,
        0x7 => -x - z,
        0x8 => y + z,
        0x9 => -y + z,
        0xA => y - z,
        0xB => -y - z,
        0xC => y + x,
        0xD => -y + z,
        0xE => y - x,
        _ => -y - z, // 0xF
    }
}

/// 6t^5 - 15t^4 + 10t^3.
#[inline]
fn ease(t: f64) -> f64 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

#[inline]
fn lerp(t: f64, a: f64, b: f64) -> f64 {
    a + t * (b - a)
}

/// Classic improved Perlin noise, range roughly [-1, 1].
pub fn perlin_noise(x: f64, y: f64, z: f64) -> f64 {
    let xi = (x.floor() as i64 & 255) as usize;
    let yi = (y.floor() as i64 & 255) as usize;
    let zi = (z.floor() as i64 & 255) as usize;

    let x = x - x.floor();
    let y = y - y.floor();
    let z = z - z.floor();

    let u = ease(x);
    let v = ease(y);
    let w = ease(z);

    let a = perm(xi) + yi;
    let aa = perm(a) + zi;
    let ab = perm(a + 1) + zi;
    let b = perm(xi + 1) + yi;
    let ba = perm(b) + zi;
    let bb = perm(b + 1) + zi;

    lerp(
        w,
        lerp(
            v,
            lerp(
                u,
                perlin_grad(perm(aa), x, y, z),
                perlin_grad(perm(ba), x - 1.0, y, z),
            ),
            lerp(
                u,
                perlin_grad(perm(ab), x, y - 1.0, z),
                perlin_grad(perm(bb), x - 1.0, y - 1.0, z),
            ),
        ),
        lerp(
            v,
            lerp(
                u,
                perlin_grad(perm(aa + 1), x, y, z - 1.0),
                perlin_grad(perm(ba + 1), x - 1.0, y, z - 1.0),
            ),
            lerp(
                u,
                perlin_grad(perm(ab + 1), x, y - 1.0, z - 1.0),
                perlin_grad(perm(bb + 1), x - 1.0, y - 1.0, z - 1.0),
            ),
        ),
    )
}

/// Three decorrelated noise samples (the reference's fixed channel offsets).
fn perlin_noise_vec3(p: [f32; 3]) -> [f32; 3] {
    let (x, y, z) = (p[0] as f64, p[1] as f64, p[2] as f64);
    [
        perlin_noise(x, y, z) as f32,
        perlin_noise(x + 89.2, y + 33.1, z + 57.3) as f32,
        perlin_noise(x + 100.3, y + 120.1, z + 142.2) as f32,
    ]
}

/// Fractal Brownian motion: `octaves` layers of `perlin_noise`, each at
/// double the previous frequency and half the previous amplitude (the
/// standard lacunarity=2/persistence=0.5 construction), normalized by the
/// actual sum of amplitudes used so the result stays in [-1, 1] regardless
/// of octave count. Drives the `remapvalue` particle operator's `fbmnoise`
/// mode. Unlike `perlin_noise` itself (a verified port of the reference's
/// `NoiseUtils.h`), this octave-summation scheme isn't recoverable against
/// any ground truth in this project: the vendored linux-wallpaperengine
/// reference doesn't implement `remapvalue` at all, and no Ghidra string
/// match exists for the real formula — see the Ghidra report's fbmnoise
/// follow-up. This is the industry-standard fBm construction, not a
/// verified port.
pub fn fbm_noise(x: f64, y: f64, z: f64, octaves: u32) -> f64 {
    let mut sum = 0.0;
    let mut amplitude = 1.0;
    let mut max_amplitude = 0.0;
    let mut frequency = 1.0;
    for _ in 0..octaves.max(1) {
        sum += perlin_noise(x * frequency, y * frequency, z * frequency) * amplitude;
        max_amplitude += amplitude;
        amplitude *= 0.5;
        frequency *= 2.0;
    }
    sum / max_amplitude.max(1e-9)
}

/// Curl of the vector noise field — divergence-free, so particles driven by
/// it swirl smoothly instead of clumping. Central differences with the
/// reference's epsilon.
pub fn curl_noise(p: [f32; 3]) -> [f32; 3] {
    const E: f32 = 1e-4;

    let sample = |dx: f32, dy: f32, dz: f32| perlin_noise_vec3([p[0] + dx, p[1] + dy, p[2] + dz]);

    let x0 = sample(-E, 0.0, 0.0);
    let x1 = sample(E, 0.0, 0.0);
    let y0 = sample(0.0, -E, 0.0);
    let y1 = sample(0.0, E, 0.0);
    let z0 = sample(0.0, 0.0, -E);
    let z1 = sample(0.0, 0.0, E);

    let x = (y1[2] - y0[2]) - (z1[1] - z0[1]);
    let y = (z1[0] - z0[0]) - (x1[2] - x0[2]);
    let z = (x1[1] - x0[1]) - (y1[0] - y0[0]);

    [x / (2.0 * E), y / (2.0 * E), z / (2.0 * E)]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perlin_is_deterministic_and_bounded() {
        let a = perlin_noise(1.37, 2.41, 3.59);
        let b = perlin_noise(1.37, 2.41, 3.59);
        assert_eq!(a, b);
        assert!(a.abs() <= 1.5, "perlin out of expected range: {a}");
        // Integer lattice points evaluate to exactly zero in classic Perlin.
        assert_eq!(perlin_noise(1.0, 2.0, 3.0), 0.0);
    }

    #[test]
    fn fbm_noise_stays_in_unit_range_regardless_of_octave_count() {
        // Perlin's own bound is documented as "roughly [-1,1]" (not exactly
        // — see `perlin_is_deterministic_and_bounded`'s 1.5 margin), so fBm,
        // being a weighted average of several Perlin samples, should have
        // the same character: a generous but real bound, not unbounded
        // growth as octaves increase (the bug a wrong normalizer, e.g. a
        // hardcoded divisor sized for a specific octave count, would cause).
        for octaves in [1, 2, 3, 4, 8] {
            let v = fbm_noise(1.37, 2.41, 3.59, octaves);
            assert!(v.abs() <= 1.5, "octaves={octaves}: fbm out of range: {v}");
        }
    }

    #[test]
    fn fbm_noise_is_deterministic() {
        let a = fbm_noise(0.31, 0.72, 0.19, 4);
        let b = fbm_noise(0.31, 0.72, 0.19, 4);
        assert_eq!(a, b);
    }

    #[test]
    fn fbm_noise_adds_higher_frequency_detail_over_plain_perlin() {
        // The defining property of fBm over single-octave noise: it should
        // NOT just equal the base octave (a regression to "plain Perlin",
        // the exact bug this function replaces — see its own doc comment).
        // Sampled at a handful of points since any single point could
        // coincidentally match.
        let mut differs = false;
        for i in 0..8 {
            let t = i as f64 * 0.37;
            let base = perlin_noise(t, t * 1.3, t * 0.7);
            let fbm = fbm_noise(t, t * 1.3, t * 0.7, 4);
            if (base - fbm).abs() > 1e-6 {
                differs = true;
            }
        }
        assert!(
            differs,
            "4-octave fbm was identical to single-octave Perlin everywhere sampled"
        );
    }

    #[test]
    fn fbm_noise_one_octave_matches_plain_perlin() {
        // With a single octave, fbm's normalization (divide by that one
        // octave's own amplitude, 1.0) must reduce to exactly perlin_noise
        // — the degenerate case that pins down the normalization formula.
        let x = 0.83;
        let (y, z) = (0.44, 0.91);
        assert_eq!(fbm_noise(x, y, z, 1), perlin_noise(x, y, z));
    }

    #[test]
    fn curl_noise_varies_smoothly_and_is_finite() {
        let a = curl_noise([0.13, 0.71, 0.42]);
        let b = curl_noise([0.14, 0.71, 0.42]);
        assert!(a.iter().all(|v| v.is_finite()));
        assert_ne!(a, [0.0; 3]);
        // Nearby samples must be close (smooth field), not identical.
        assert_ne!(a, b);
        for i in 0..3 {
            assert!(
                (a[i] - b[i]).abs() < 1.0,
                "field not smooth: {a:?} vs {b:?}"
            );
        }
    }
}
