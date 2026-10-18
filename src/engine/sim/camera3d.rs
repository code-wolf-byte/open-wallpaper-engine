//! Perspective camera for genuine 3D scenes (`Scene::is_perspective`).
//!
//! The C++ reference has no perspective path — CScene.cpp unconditionally
//! calls `setOrthogonalProjection`, so 3D scenes render broken there. This
//! module implements standard lookAt + perspective math (same conventions as
//! `glm::lookAt`/`glm::perspective`, which the reference does use for
//! particle-space projections in CParticle.cpp) so image layers placed at 3D
//! origins can be projected to screen space.

use crate::engine::model::scene::Scene;

/// Column-major 4x4, `m[col][row]` — glm's layout, so it uploads to a WGSL
/// `mat4x4<f32>` as-is.
pub type Mat4 = [[f32; 4]; 4];

pub fn mat4_mul(a: &Mat4, b: &Mat4) -> Mat4 {
    let mut out = [[0.0f32; 4]; 4];
    for (col, out_col) in out.iter_mut().enumerate() {
        for (row, cell) in out_col.iter_mut().enumerate() {
            *cell = (0..4).map(|k| a[k][row] * b[col][k]).sum();
        }
    }
    out
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let len = dot(v, v).sqrt();
    if len <= 1e-12 {
        return [0.0, 0.0, 1.0];
    }
    [v[0] / len, v[1] / len, v[2] / len]
}

/// Right-handed lookAt, identical to `glm::lookAt`. `pub(crate)` so
/// `engine::gpu::shadow` can build light-space view matrices with the same math.
pub(crate) fn look_at(eye: [f32; 3], center: [f32; 3], up: [f32; 3]) -> Mat4 {
    let f = normalize(sub(center, eye));
    let s = normalize(cross(f, up));
    let u = cross(s, f);
    [
        [s[0], u[0], -f[0], 0.0],
        [s[1], u[1], -f[1], 0.0],
        [s[2], u[2], -f[2], 0.0],
        [-dot(s, eye), -dot(u, eye), dot(f, eye), 1.0],
    ]
}

/// Right-handed perspective projection, identical to `glm::perspective`
/// (vertical FOV in radians, GL clip-space z). `pub(crate)` — see `look_at`.
pub(crate) fn perspective(fovy: f32, aspect: f32, near: f32, far: f32) -> Mat4 {
    let t = 1.0 / (fovy / 2.0).tan();
    let mut m = [[0.0f32; 4]; 4];
    m[0][0] = t / aspect;
    m[1][1] = t;
    m[2][2] = (far + near) / (near - far);
    m[2][3] = -1.0;
    m[3][2] = (2.0 * far * near) / (near - far);
    m
}

/// A resolved perspective camera: projects world-space points to NDC.
pub struct PerspectiveCamera {
    view: Mat4,
    view_proj: Mat4,
    eye: [f32; 3],
}

impl PerspectiveCamera {
    /// Build from scene.json camera/general. Returns `None` unless
    /// `scene.is_perspective()`.
    pub fn from_scene(scene: &Scene, aspect: f32) -> Option<Self> {
        if !scene.is_perspective() {
            return None;
        }
        let cam = scene.camera.as_ref()?;
        let eye3 = cam.parsed_eye()?;
        let center3 = cam.parsed_center()?;
        let up3 = cam.parsed_up().unwrap_or([0.0, 1.0, 0.0]);
        let g = scene.general.as_ref();
        let fov_deg = g
            .and_then(|g| g.fov.as_ref())
            .and_then(crate::engine::model::scene::parse_value_f32)
            .unwrap_or(50.0);
        let near = g
            .and_then(|g| g.nearz.as_ref())
            .and_then(crate::engine::model::scene::parse_value_f32)
            .unwrap_or(0.1)
            .max(1e-4);
        let far = g
            .and_then(|g| g.farz.as_ref())
            .and_then(crate::engine::model::scene::parse_value_f32)
            .unwrap_or(10000.0)
            .max(near * 2.0);

        let eye = eye3.map(|v| v as f32);
        let center = center3.map(|v| v as f32);
        let up = up3.map(|v| v as f32);
        let view = look_at(eye, center, up);
        let proj = perspective(fov_deg.to_radians(), aspect, near, far);
        Some(Self {
            view,
            view_proj: mat4_mul(&proj, &view),
            eye,
        })
    }

    /// World-space camera position.
    pub fn eye(&self) -> [f32; 3] {
        self.eye
    }

    /// `view_proj` in GL clip-space z ([-1,1], not remapped to wgpu's
    /// [0,1] the way [`Self::view_proj_gpu`] is) — the convention
    /// `mat4_inverse` + NDC unprojection expects. `pub(crate)` so
    /// `engine::volumetrics` can reconstruct world-space view rays for its
    /// ray march.
    pub(crate) fn view_proj_raw(&self) -> Mat4 {
        self.view_proj
    }

    /// Project a world-space point. Returns `(ndc_x, ndc_y)`; `None` when the
    /// point is at or behind the eye plane (w <= 0).
    pub fn project(&self, p: [f32; 3]) -> Option<[f32; 2]> {
        let m = &self.view_proj;
        let x = m[0][0] * p[0] + m[1][0] * p[1] + m[2][0] * p[2] + m[3][0];
        let y = m[0][1] * p[0] + m[1][1] * p[1] + m[2][1] * p[2] + m[3][1];
        let w = m[0][3] * p[0] + m[1][3] * p[1] + m[2][3] * p[2] + m[3][3];
        if w <= 1e-6 {
            return None;
        }
        Some([x / w, y / w])
    }

    /// Distance of a world-space point along the camera's forward axis
    /// (positive in front of the camera). Used for painter's-algorithm depth
    /// sorting.
    pub fn view_depth(&self, p: [f32; 3]) -> f32 {
        let m = &self.view;
        -(m[0][2] * p[0] + m[1][2] * p[1] + m[2][2] * p[2] + m[3][2])
    }

    /// `view_proj` with GL's clip-space z ([-1,1]) remapped to wgpu's ([0,1]),
    /// column-major and ready to upload. [`project`](Self::project) reads only
    /// x/y so it never needed this; a depth buffer does.
    pub fn view_proj_gpu(&self) -> Mat4 {
        gl_to_wgpu_depth(&self.view_proj)
    }

    /// Full model-view-projection for an object, column-major and ready to
    /// upload as a `mat4x4<f32>` uniform.
    pub fn mvp(&self, origin: [f32; 3], angles: [f32; 3], scale: [f32; 3]) -> Mat4 {
        mat4_mul(&self.view_proj_gpu(), &model_matrix(origin, angles, scale))
    }

    /// Transform a world-space point into view space (camera at the origin
    /// by construction). Used to place `light` objects for the mesh3d
    /// lighting pass, which shades entirely in view space — see
    /// `model_view`.
    pub fn to_view_space(&self, p: [f32; 3]) -> [f32; 3] {
        let m = &self.view;
        [
            m[0][0] * p[0] + m[1][0] * p[1] + m[2][0] * p[2] + m[3][0],
            m[0][1] * p[0] + m[1][1] * p[1] + m[2][1] * p[2] + m[3][1],
            m[0][2] * p[0] + m[1][2] * p[1] + m[2][2] * p[2] + m[3][2],
        ]
    }

    /// Transform a world-space *direction* into view space — same rotation
    /// `to_view_space` applies to a position, without the translation
    /// (directions have no location to translate). Used to place a spot
    /// light's facing direction for the mesh3d lighting pass, which shades
    /// entirely in view space — see `to_view_space`.
    pub fn to_view_direction(&self, d: [f32; 3]) -> [f32; 3] {
        let m = &self.view;
        [
            m[0][0] * d[0] + m[1][0] * d[1] + m[2][0] * d[2],
            m[0][1] * d[0] + m[1][1] * d[1] + m[2][1] * d[2],
            m[0][2] * d[0] + m[1][2] * d[1] + m[2][2] * d[2],
        ]
    }

    /// Model-view (no projection) for an object — world→view space, ready to
    /// upload as a `mat4x4<f32>` uniform. The lighting pass needs view-space
    /// vertex positions separately from the clip-space `mvp` output.
    pub fn model_view(&self, origin: [f32; 3], angles: [f32; 3], scale: [f32; 3]) -> Mat4 {
        mat4_mul(&self.view, &model_matrix(origin, angles, scale))
    }

    /// View-space normal transform for an object: `view_rotation · object_rotation
    /// · diag(1/scale)`, the standard inverse-transpose simplification for a
    /// rotation-plus-scale model matrix (exact for any scale, uniform or not).
    /// Translation is left at whatever `mat4_mul` produces from `view`'s own
    /// translation column — callers multiply by `vec4(normal, 0.0)`, so the w=0
    /// discards it; only the upper-left 3x3 is meaningful.
    pub fn normal_view(&self, angles: [f32; 3], scale: [f32; 3]) -> Mat4 {
        let inv_scale = [
            1.0 / scale[0].max(1e-6),
            1.0 / scale[1].max(1e-6),
            1.0 / scale[2].max(1e-6),
        ];
        mat4_mul(
            &self.view,
            &model_matrix([0.0, 0.0, 0.0], angles, inv_scale),
        )
    }
}

/// Remaps GL's clip-space z range ([-1,1]) to wgpu's ([0,1]) by
/// premultiplying with the standard `diag(1,1,0.5,1)`-plus-translation fix
/// matrix. `pub(crate)` so `engine::gpu::shadow` can build shadow-map
/// view-projections in the same depth convention as the main camera.
pub(crate) fn gl_to_wgpu_depth(m: &Mat4) -> Mat4 {
    let mut fix = identity();
    fix[2][2] = 0.5;
    fix[3][2] = 0.5;
    mat4_mul(&fix, m)
}

/// General 4x4 matrix inverse (cofactor/adjugate method — no assumptions
/// about the matrix's shape, unlike `gl_to_wgpu_depth`'s special-cased
/// remap). Used to unproject screen-space rays back to world space for the
/// volumetric-lighting ray march (`engine::volumetrics`) — a raw NDC point
/// times this inverse recovers its world-space position. Returns the
/// identity matrix for a singular input (determinant ~0) rather than
/// producing NaNs; callers that can't tolerate a wrong-but-finite fallback
/// should check `is_finite()` on the result themselves.
pub fn mat4_inverse(m: &Mat4) -> Mat4 {
    // `m[col][row]`, so index as m[c][r] throughout — same convention
    // `mat4_mul` already uses.
    let a = |r: usize, c: usize| m[c][r];
    // Cofactor expansion along the first row, using the standard 2x2 minor
    // trick for a 4x4 (each 3x3 cofactor built from six precomputed 2x2
    // sub-determinants of the bottom two rows, then again for the top two).
    let s0 = a(0, 0) * a(1, 1) - a(1, 0) * a(0, 1);
    let s1 = a(0, 0) * a(1, 2) - a(1, 0) * a(0, 2);
    let s2 = a(0, 0) * a(1, 3) - a(1, 0) * a(0, 3);
    let s3 = a(0, 1) * a(1, 2) - a(1, 1) * a(0, 2);
    let s4 = a(0, 1) * a(1, 3) - a(1, 1) * a(0, 3);
    let s5 = a(0, 2) * a(1, 3) - a(1, 2) * a(0, 3);

    let c5 = a(2, 2) * a(3, 3) - a(3, 2) * a(2, 3);
    let c4 = a(2, 1) * a(3, 3) - a(3, 1) * a(2, 3);
    let c3 = a(2, 1) * a(3, 2) - a(3, 1) * a(2, 2);
    let c2 = a(2, 0) * a(3, 3) - a(3, 0) * a(2, 3);
    let c1 = a(2, 0) * a(3, 2) - a(3, 0) * a(2, 2);
    let c0 = a(2, 0) * a(3, 1) - a(3, 0) * a(2, 1);

    let det = s0 * c5 - s1 * c4 + s2 * c3 + s3 * c2 - s4 * c1 + s5 * c0;
    if det.abs() < 1e-12 {
        return identity();
    }
    let inv_det = 1.0 / det;

    // Result built row-by-row in standard (row, col) math notation, then
    // transposed into this codebase's `m[col][row]` storage at the end.
    let r = [
        [
            a(1, 1) * c5 - a(1, 2) * c4 + a(1, 3) * c3,
            -a(0, 1) * c5 + a(0, 2) * c4 - a(0, 3) * c3,
            a(3, 1) * s5 - a(3, 2) * s4 + a(3, 3) * s3,
            -a(2, 1) * s5 + a(2, 2) * s4 - a(2, 3) * s3,
        ],
        [
            -a(1, 0) * c5 + a(1, 2) * c2 - a(1, 3) * c1,
            a(0, 0) * c5 - a(0, 2) * c2 + a(0, 3) * c1,
            -a(3, 0) * s5 + a(3, 2) * s2 - a(3, 3) * s1,
            a(2, 0) * s5 - a(2, 2) * s2 + a(2, 3) * s1,
        ],
        [
            a(1, 0) * c4 - a(1, 1) * c2 + a(1, 3) * c0,
            -a(0, 0) * c4 + a(0, 1) * c2 - a(0, 3) * c0,
            a(3, 0) * s4 - a(3, 1) * s2 + a(3, 3) * s0,
            -a(2, 0) * s4 + a(2, 1) * s2 - a(2, 3) * s0,
        ],
        [
            -a(1, 0) * c3 + a(1, 1) * c1 - a(1, 2) * c0,
            a(0, 0) * c3 - a(0, 1) * c1 + a(0, 2) * c0,
            -a(3, 0) * s3 + a(3, 1) * s1 - a(3, 2) * s0,
            a(2, 0) * s3 - a(2, 1) * s1 + a(2, 2) * s0,
        ],
    ];
    let mut out = identity();
    for row in 0..4 {
        for col in 0..4 {
            out[col][row] = r[row][col] * inv_det;
        }
    }
    out
}

pub fn identity() -> Mat4 {
    let mut m = [[0.0f32; 4]; 4];
    for (i, row) in m.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    m
}

/// Model matrix for a scene object: translate(origin) · Rz·Ry·Rx · scale —
/// the same convention as [`rotate_euler`].
pub fn model_matrix(origin: [f32; 3], angles: [f32; 3], scale: [f32; 3]) -> Mat4 {
    let (sx, cx) = angles[0].sin_cos();
    let (sy, cy) = angles[1].sin_cos();
    let (sz, cz) = angles[2].sin_cos();
    // Rz·Ry·Rx, expanded, with each column pre-scaled.
    let r = [
        [cz * cy, sz * cy, -sy],
        [cz * sy * sx - sz * cx, sz * sy * sx + cz * cx, cy * sx],
        [cz * sy * cx + sz * sx, sz * sy * cx - cz * sx, cy * cx],
    ];
    let mut m = identity();
    for c in 0..3 {
        for row in 0..3 {
            m[c][row] = r[c][row] * scale[c];
        }
    }
    m[3][0] = origin[0];
    m[3][1] = origin[1];
    m[3][2] = origin[2];
    m
}

/// Rotate `v` by intrinsic Euler angles in radians (WE `angles`, stored as
/// radians in scene.json), applied as Rz·Ry·Rx like a standard model matrix.
pub fn rotate_euler(v: [f32; 3], angles: [f32; 3]) -> [f32; 3] {
    let (sx, cx) = angles[0].sin_cos();
    let (sy, cy) = angles[1].sin_cos();
    let (sz, cz) = angles[2].sin_cos();
    // Rx
    let v = [v[0], cx * v[1] - sx * v[2], sx * v[1] + cx * v[2]];
    // Ry
    let v = [cy * v[0] + sy * v[2], v[1], -sy * v[0] + cy * v[2]];
    // Rz
    [cz * v[0] - sz * v[1], sz * v[0] + cz * v[1], v[2]]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mat4_approx_eq(a: &Mat4, b: &Mat4, eps: f32) -> bool {
        (0..4).all(|c| (0..4).all(|r| (a[c][r] - b[c][r]).abs() < eps))
    }

    #[test]
    fn mat4_inverse_of_identity_is_identity() {
        assert!(mat4_approx_eq(
            &mat4_inverse(&identity()),
            &identity(),
            1e-6
        ));
    }

    #[test]
    fn mat4_inverse_roundtrips_a_real_view_proj() {
        let view = look_at([3.0, 2.0, 5.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        let proj = perspective(50f32.to_radians(), 16.0 / 9.0, 0.1, 100.0);
        let vp = mat4_mul(&proj, &view);
        let inv = mat4_inverse(&vp);
        let roundtrip = mat4_mul(&vp, &inv);
        assert!(
            mat4_approx_eq(&roundtrip, &identity(), 1e-3),
            "vp * inverse(vp) should be ~identity, got {roundtrip:?}"
        );
    }

    #[test]
    fn mat4_inverse_unprojects_a_known_world_point_back_to_itself() {
        let view = look_at([0.0, 0.0, 10.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        let proj = perspective(60f32.to_radians(), 1.0, 0.1, 100.0);
        let vp = mat4_mul(&proj, &view);
        let inv = mat4_inverse(&vp);

        let world = [1.5, -0.5, 2.0, 1.0f32];
        // Project world -> clip -> NDC.
        let clip: [f32; 4] = std::array::from_fn(|row| (0..4).map(|k| vp[k][row] * world[k]).sum());
        let ndc = [clip[0] / clip[3], clip[1] / clip[3], clip[2] / clip[3], 1.0];
        // Unproject NDC -> clip (undo the perspective divide with the same w) -> world.
        let unclip = [
            ndc[0] * clip[3],
            ndc[1] * clip[3],
            ndc[2] * clip[3],
            clip[3],
        ];
        let back: [f32; 4] =
            std::array::from_fn(|row| (0..4).map(|k| inv[k][row] * unclip[k]).sum());
        for i in 0..3 {
            assert!(
                (back[i] - world[i]).abs() < 1e-3,
                "component {i}: got {}, want {}",
                back[i],
                world[i]
            );
        }
    }

    #[test]
    fn mat4_inverse_of_singular_matrix_is_finite() {
        // All-zero rotation/scale (a genuinely singular matrix) must not
        // produce NaN/Inf.
        let mut m = identity();
        m[0][0] = 0.0;
        let inv = mat4_inverse(&m);
        assert!(inv.iter().all(|c| c.iter().all(|v| v.is_finite())));
    }

    /// `model_matrix` expands the same Rz·Ry·Rx that `rotate_euler` applies
    /// step by step — they must not drift apart.
    #[test]
    fn model_matrix_rotation_matches_rotate_euler() {
        let angles = [0.3, -1.1, 2.4];
        let scale = [2.0, 0.5, 3.0];
        let origin = [10.0, -4.0, 7.0];
        let m = model_matrix(origin, angles, scale);
        for v in [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.3, -0.7, 0.5]] {
            let want = rotate_euler([v[0] * scale[0], v[1] * scale[1], v[2] * scale[2]], angles);
            let got: Vec<f32> = (0..3)
                .map(|r| m[0][r] * v[0] + m[1][r] * v[1] + m[2][r] * v[2] + m[3][r])
                .collect();
            for i in 0..3 {
                assert!(
                    (got[i] - (want[i] + origin[i])).abs() < 1e-5,
                    "axis {i}: got {got:?} want {want:?} + {origin:?}"
                );
            }
        }
    }

    fn scene_json(json: &str) -> Scene {
        serde_json::from_str(json).unwrap()
    }

    fn test_scene() -> Scene {
        scene_json(
            r#"{
                "camera": {"eye": "0 0 10", "center": "0 0 0", "up": "0 1 0"},
                "general": {"orthogonalprojection": null, "fov": 90.0, "nearz": 0.1, "farz": 100.0}
            }"#,
        )
    }

    #[test]
    fn detects_perspective_scene() {
        assert!(test_scene().is_perspective());
        // Usable ortho projection → not perspective.
        let ortho = scene_json(
            r#"{
                "camera": {"eye": "0 0 50", "center": "0 0 0",
                           "orthogonalprojection": {"width": 1920, "height": 1080}}
            }"#,
        );
        assert!(!ortho.is_perspective());
        // No camera eye/center → cannot build a perspective view.
        let no_cam = scene_json(r#"{"general": {}}"#);
        assert!(!no_cam.is_perspective());
    }

    #[test]
    fn projects_points_in_front_of_camera() {
        let cam = PerspectiveCamera::from_scene(&test_scene(), 1.0).unwrap();
        // Point straight ahead lands at NDC center.
        let ndc = cam.project([0.0, 0.0, 0.0]).unwrap();
        assert!(ndc[0].abs() < 1e-5 && ndc[1].abs() < 1e-5);
        // fov 90°, distance 10: y=10 sits exactly on the top edge (ndc.y = 1).
        let ndc = cam.project([0.0, 10.0, 0.0]).unwrap();
        assert!((ndc[1] - 1.0).abs() < 1e-4, "got {ndc:?}");
        // Behind the camera → culled.
        assert!(cam.project([0.0, 0.0, 20.0]).is_none());
    }

    #[test]
    fn normal_view_scales_by_inverse_not_forward() {
        // test_scene's camera sits on +Z looking at the origin with up=+Y,
        // which works out to an identity view rotation, so the object-space
        // math is visible directly in the result. A non-uniform scale on X
        // must divide the X-aligned normal's component by that scale, not
        // multiply by it — that's the whole point of the inverse-transpose
        // simplification `normal_view` relies on.
        let cam = PerspectiveCamera::from_scene(&test_scene(), 1.0).unwrap();
        let m = cam.normal_view([0.0, 0.0, 0.0], [2.0, 1.0, 1.0]);
        let n = [1.0, 0.0, 0.0];
        let got = [
            m[0][0] * n[0] + m[1][0] * n[1] + m[2][0] * n[2],
            m[0][1] * n[0] + m[1][1] * n[1] + m[2][1] * n[2],
            m[0][2] * n[0] + m[1][2] * n[1] + m[2][2] * n[2],
        ];
        assert!(
            (got[0] - 0.5).abs() < 1e-5,
            "expected 1/scale.x = 0.5, got {got:?}"
        );
        assert!(got[1].abs() < 1e-5 && got[2].abs() < 1e-5, "got {got:?}");
    }

    #[test]
    fn to_view_direction_ignores_translation() {
        // `to_view_direction(d)` must equal `to_view_space(p + d) -
        // to_view_space(p)` for any `p` — the defining property of a
        // direction transform (no location to translate), and the bug
        // class this method exists to avoid (accidentally reusing
        // `to_view_space`'s translation term for a direction).
        let cam = PerspectiveCamera::from_scene(&test_scene(), 1.0).unwrap();
        let d = [0.3, -0.7, 0.5];
        let got = cam.to_view_direction(d);
        for p in [[0.0, 0.0, 0.0], [4.0, -2.0, 9.0], [-100.0, 3.0, 0.0]] {
            let want = [
                cam.to_view_space([p[0] + d[0], p[1] + d[1], p[2] + d[2]])[0]
                    - cam.to_view_space(p)[0],
                cam.to_view_space([p[0] + d[0], p[1] + d[1], p[2] + d[2]])[1]
                    - cam.to_view_space(p)[1],
                cam.to_view_space([p[0] + d[0], p[1] + d[1], p[2] + d[2]])[2]
                    - cam.to_view_space(p)[2],
            ];
            for i in 0..3 {
                assert!(
                    (got[i] - want[i]).abs() < 1e-4,
                    "axis {i}: got {got:?} want {want:?} (p={p:?})"
                );
            }
        }
    }

    #[test]
    fn view_depth_orders_near_to_far() {
        let cam = PerspectiveCamera::from_scene(&test_scene(), 1.0).unwrap();
        let near = cam.view_depth([0.0, 0.0, 5.0]);
        let far = cam.view_depth([0.0, 0.0, -20.0]);
        assert!(near > 0.0 && far > near);
        assert!(cam.view_depth([0.0, 0.0, 15.0]) < 0.0); // behind eye
    }

    #[test]
    fn euler_rotation_z_only_matches_2d() {
        let v = rotate_euler([1.0, 0.0, 0.0], [0.0, 0.0, std::f32::consts::FRAC_PI_2]);
        assert!((v[0]).abs() < 1e-6 && (v[1] - 1.0).abs() < 1e-6);
    }
}
