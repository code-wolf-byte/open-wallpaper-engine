//! Shadow-map atlas packing and light-space projections for the mesh3d
//! shadow pass — the CPU-side counterpart of the shadow-atlas architecture
//! recovered from `wallpaper64.exe` (see `~/Applications/ghidra-dump/
//! REPORT.md`, "Follow-up (h)"): one shared depth texture packed with a
//! tile per shadow-casting light, addressed by a per-light UV sub-rect,
//! rather than a separate bound texture per light.
//!
//! Deliberately not a byte-exact replica of the original: WE's atlas tiles
//! are sized per-light (a heuristic that wasn't recoverable from the
//! binary) and shelf-packed dynamically; this uses one fixed tile size for
//! every shadow-casting light. Point lights get a single perspective
//! projection each, matching the real engine's own single-projection-per-
//! point-light design (`CalculateProjectedCoordsPoint` takes exactly one
//! `ShadowPointProjection`/`ShadowPointProjectionTransform`, not six) —
//! though WE likely warps that projection (dual-paraboloid or similar) for
//! wider-than-hemisphere coverage, which this doesn't attempt to replicate.

use crate::engine::render::Mesh3dLayer;
use crate::engine::sim::camera3d::{self, Mat4};

/// Fixed per-light shadow-map tile size. See module docs: WE's real tile
/// size is a per-light heuristic not recovered from the binary.
pub const SHADOW_TILE_SIZE: u32 = 512;

/// Cap on simultaneous shadow-casting lights. Independent of
/// `MESH3D_MAX_LIGHTS` (the lit-but-possibly-unshadowed light budget) — the
/// atlas grows to fit up to this many tiles, not further. Extra
/// shadow-casting lights beyond this render unshadowed (`shadow_factor`
/// stays 1.0), the same graceful-degradation shape as `MESH3D_MAX_LIGHTS`
/// itself.
pub const MAX_SHADOW_LIGHTS: usize = 4;

/// A shadow-atlas tile: pixel offset + size within the shared atlas texture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShadowTile {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

/// Shelf-pack `count` same-size tiles into an atlas that grows only as wide
/// as needed (up to `max_width`), wrapping to a new row once a row fills —
/// the same "pack tiles, grow the atlas to fit demand" shape as WE's real
/// `_rt_shadowAtlas` allocator, traced in the Ghidra report. Returns
/// `(atlas_width, atlas_height, tiles)`; `(0, 0, [])` for `count == 0` (no
/// shadow-casting lights this scene — the caller skips allocating an atlas
/// at all rather than creating a degenerate zero-size texture).
pub fn pack_tiles(count: usize, tile: u32, max_width: u32) -> (u32, u32, Vec<ShadowTile>) {
    if count == 0 || tile == 0 {
        return (0, 0, Vec::new());
    }
    let per_row = (max_width / tile).max(1) as usize;
    let mut tiles = Vec::with_capacity(count);
    for i in 0..count {
        let col = (i % per_row) as u32;
        let row = (i / per_row) as u32;
        tiles.push(ShadowTile {
            x: col * tile,
            y: row * tile,
            w: tile,
            h: tile,
        });
    }
    let cols = per_row.min(count) as u32;
    let rows = ((count - 1) / per_row + 1) as u32;
    (cols * tile, rows * tile, tiles)
}

/// A tile's placement as a `(u, v, scale_u, scale_v)` sub-rect in [0,1]
/// atlas UV space — matches `g_LFeature_ShadowProjectionTransform`'s real
/// role (an atlas sub-rect, not a second transform matrix; see the Ghidra
/// report). Shader usage: `atlas_uv = uv_rect.xy + ndc_uv * uv_rect.zw`.
pub fn tile_uv_rect(tile: ShadowTile, atlas_w: u32, atlas_h: u32) -> [f32; 4] {
    if atlas_w == 0 || atlas_h == 0 {
        return [0.0; 4];
    }
    [
        tile.x as f32 / atlas_w as f32,
        tile.y as f32 / atlas_h as f32,
        tile.w as f32 / atlas_w as f32,
        tile.h as f32 / atlas_h as f32,
    ]
}

fn transform_point(m: &Mat4, p: [f32; 3]) -> [f32; 3] {
    [
        m[0][0] * p[0] + m[1][0] * p[1] + m[2][0] * p[2] + m[3][0],
        m[0][1] * p[0] + m[1][1] * p[1] + m[2][1] * p[2] + m[3][1],
        m[0][2] * p[0] + m[1][2] * p[1] + m[2][2] * p[2] + m[3][2],
    ]
}

fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn len3(v: [f32; 3]) -> f32 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

/// Conservative (not minimal) world-space bounding sphere of one mesh3d
/// object: centered at its own origin, radius = the farthest transformed
/// vertex from that origin.
fn mesh_bounds(layer: &Mesh3dLayer) -> ([f32; 3], f32) {
    let model = camera3d::model_matrix(layer.origin, layer.angles, layer.scale);
    let mut max_dist2 = 0.0f32;
    for p in &layer.mesh.positions {
        let w = transform_point(&model, *p);
        let d2 = {
            let d = sub3(w, layer.origin);
            d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
        };
        if d2 > max_dist2 {
            max_dist2 = d2;
        }
    }
    (layer.origin, max_dist2.sqrt().max(0.01))
}

/// Merge two bounding spheres into one that contains both (Ritter-style
/// merge — conservative, not the true minimal enclosing sphere, which isn't
/// needed here: coverage for shadow-frustum sizing, not tightness).
fn merge_spheres(a: ([f32; 3], f32), b: ([f32; 3], f32)) -> ([f32; 3], f32) {
    let d = sub3(b.0, a.0);
    let dist = len3(d);
    if dist + b.1 <= a.1 {
        return a;
    }
    if dist + a.1 <= b.1 {
        return b;
    }
    let new_r = (dist + a.1 + b.1) * 0.5;
    let t = (new_r - a.1) / dist.max(1e-6);
    (
        [a.0[0] + d[0] * t, a.0[1] + d[1] * t, a.0[2] + d[2] * t],
        new_r,
    )
}

/// World-space bounding sphere covering every mesh3d object in the scene —
/// used to size shadow-casting lights' projections so every caster fits
/// inside each light's frustum. `(origin, 1.0)` when there are no mesh3d
/// objects (shadows never render in that case, so the value is unused, but
/// callers get something sane rather than a nonsensical zero-radius sphere).
pub fn scene_bounds(mesh3d_layers: &[Mesh3dLayer]) -> ([f32; 3], f32) {
    let mut spheres = mesh3d_layers.iter().map(mesh_bounds);
    let Some(first) = spheres.next() else {
        return ([0.0, 0.0, 0.0], 1.0);
    };
    spheres.fold(first, merge_spheres)
}

/// A vector to use as `look_at`'s "up" — avoids a degenerate view matrix
/// when the light-to-target direction is itself near-vertical.
fn pick_up_vector(dir: [f32; 3]) -> [f32; 3] {
    if dir[0].abs() < 1e-3 && dir[2].abs() < 1e-3 {
        [0.0, 0.0, 1.0]
    } else {
        [0.0, 1.0, 0.0]
    }
}

/// Build a single perspective shadow view-projection for a point light,
/// wide enough to cover a bounding sphere (`target`, `radius`) from the
/// light's position — see the module docs for how this relates to WE's own
/// single-projection-per-point-light design. wgpu depth-range ([0,1]),
/// ready to upload.
pub fn point_light_view_proj(light_origin: [f32; 3], target: [f32; 3], radius: f32) -> Mat4 {
    let to_target = sub3(target, light_origin);
    let distance = len3(to_target).max(1e-4);
    let radius = radius.max(0.01);
    let half_fov = (radius / distance).clamp(-1.0, 1.0).asin();
    let fovy = (half_fov * 2.0).clamp(0.1, 170f32.to_radians());
    let near = (distance - radius).max(0.05);
    let far = distance + radius;
    let up = pick_up_vector(to_target);
    let view = camera3d::look_at(light_origin, target, up);
    // Square tiles (see `SHADOW_TILE_SIZE`), so aspect is always 1.0.
    let proj = camera3d::perspective(fovy, 1.0, near, far);
    camera3d::gl_to_wgpu_depth(&camera3d::mat4_mul(&proj, &view))
}

/// Build a single perspective shadow view-projection for a spot light,
/// aimed along its own facing `direction` with FOV set directly from its
/// `outer_cone_degrees` — unlike a point light (omnidirectional, so its
/// frustum has to be aimed at and sized to the scene's bounding sphere,
/// see `point_light_view_proj`), a spot light already has an exact,
/// physically-motivated direction and cone width, no guesswork needed.
/// `scene_center`/`scene_radius` (the same bounding sphere point lights
/// use) still size the near/far planes, projected onto the light's own
/// axis, so the depth range covers whatever the cone can actually see.
pub fn spot_light_view_proj(
    light_origin: [f32; 3],
    direction: [f32; 3],
    outer_cone_degrees: f32,
    scene_center: [f32; 3],
    scene_radius: f32,
) -> Mat4 {
    let dir_len = len3(direction).max(1e-6);
    let dir = [
        direction[0] / dir_len,
        direction[1] / dir_len,
        direction[2] / dir_len,
    ];
    let target = [
        light_origin[0] + dir[0],
        light_origin[1] + dir[1],
        light_origin[2] + dir[2],
    ];
    // Full FOV is twice the cone's half-angle.
    let fovy = (outer_cone_degrees.to_radians() * 2.0).clamp(0.1, 170f32.to_radians());
    let to_center = sub3(scene_center, light_origin);
    let axial_dist = to_center[0] * dir[0] + to_center[1] * dir[1] + to_center[2] * dir[2];
    let near = (axial_dist - scene_radius).max(0.05);
    let far = (axial_dist + scene_radius).max(near + 0.1);
    let up = pick_up_vector(dir);
    let view = camera3d::look_at(light_origin, target, up);
    let proj = camera3d::perspective(fovy, 1.0, near, far);
    camera3d::gl_to_wgpu_depth(&camera3d::mat4_mul(&proj, &view))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_tiles_empty() {
        assert_eq!(pack_tiles(0, 512, 2048), (0, 0, Vec::new()));
    }

    #[test]
    fn pack_tiles_single_row() {
        let (w, h, tiles) = pack_tiles(3, 512, 2048);
        assert_eq!((w, h), (512 * 3, 512));
        assert_eq!(
            tiles,
            vec![
                ShadowTile {
                    x: 0,
                    y: 0,
                    w: 512,
                    h: 512
                },
                ShadowTile {
                    x: 512,
                    y: 0,
                    w: 512,
                    h: 512
                },
                ShadowTile {
                    x: 1024,
                    y: 0,
                    w: 512,
                    h: 512
                },
            ]
        );
    }

    #[test]
    fn pack_tiles_wraps_to_new_row() {
        // max_width fits exactly 2 tiles per row.
        let (w, h, tiles) = pack_tiles(3, 512, 1024);
        assert_eq!((w, h), (1024, 1024));
        assert_eq!(
            tiles[2],
            ShadowTile {
                x: 0,
                y: 512,
                w: 512,
                h: 512
            }
        );
    }

    #[test]
    fn tile_uv_rect_normalizes_to_unit_range() {
        let (w, h, tiles) = pack_tiles(2, 512, 1024);
        let rect = tile_uv_rect(tiles[1], w, h);
        assert!((rect[0] - 0.5).abs() < 1e-6, "u offset: {rect:?}");
        assert!((rect[2] - 0.5).abs() < 1e-6, "u scale: {rect:?}");
    }

    #[test]
    fn merge_spheres_contains_both_centers() {
        let a = ([0.0, 0.0, 0.0], 1.0);
        let b = ([5.0, 0.0, 0.0], 1.0);
        let (center, radius) = merge_spheres(a, b);
        // Both original centers must lie within the merged sphere (allowing
        // for their own radius).
        assert!(len3(sub3(a.0, center)) + a.1 <= radius + 1e-4);
        assert!(len3(sub3(b.0, center)) + b.1 <= radius + 1e-4);
    }

    #[test]
    fn merge_spheres_one_contains_other() {
        let big = ([0.0, 0.0, 0.0], 10.0);
        let small = ([1.0, 0.0, 0.0], 1.0);
        assert_eq!(merge_spheres(big, small), big);
        assert_eq!(merge_spheres(small, big), big);
    }

    #[test]
    fn point_light_view_proj_covers_target() {
        // Light at (0,0,10) looking at the origin, bounding sphere radius 2.
        let vp = point_light_view_proj([0.0, 0.0, 10.0], [0.0, 0.0, 0.0], 2.0);
        // Project the target itself: should land at NDC center (x=y=0 after
        // divide), non-degenerate w.
        let p = [0.0, 0.0, 0.0, 1.0];
        let clip: [f32; 4] = std::array::from_fn(|row| (0..4).map(|k| vp[k][row] * p[k]).sum());
        assert!(clip[3] > 0.0, "w should be positive: {clip:?}");
        let ndc = [clip[0] / clip[3], clip[1] / clip[3]];
        assert!(ndc[0].abs() < 1e-4 && ndc[1].abs() < 1e-4, "got {ndc:?}");
    }

    #[test]
    fn point_light_view_proj_edge_of_sphere_stays_in_frustum() {
        let light = [0.0, 0.0, 10.0];
        let target = [0.0, 0.0, 0.0];
        let radius = 2.0;
        let vp = point_light_view_proj(light, target, radius);
        // A point on the sphere's edge, perpendicular to the view axis,
        // should still project inside the [-1,1] NDC range (with a little
        // slack for the FOV being an equality bound).
        let p = [radius * 0.99, 0.0, 0.0, 1.0];
        let clip: [f32; 4] = std::array::from_fn(|row| (0..4).map(|k| vp[k][row] * p[k]).sum());
        assert!(clip[3] > 0.0);
        let ndc_x = clip[0] / clip[3];
        assert!(
            ndc_x.abs() <= 1.0,
            "edge point escaped the frustum: ndc_x={ndc_x}"
        );
    }

    #[test]
    fn scene_bounds_empty_is_sane() {
        let (_, r) = scene_bounds(&[]);
        assert!(r > 0.0);
    }

    #[test]
    fn spot_light_view_proj_centers_a_target_on_its_own_axis() {
        // Light at (0,0,10) aimed straight down -Z, a wide-ish 45° cone.
        // A point directly ahead on that axis must land at NDC center,
        // regardless of where the scene's bounding sphere actually is.
        let vp = spot_light_view_proj(
            [0.0, 0.0, 10.0],
            [0.0, 0.0, -1.0],
            45.0,
            [3.0, 3.0, 3.0],
            2.0,
        );
        let p = [0.0, 0.0, 0.0, 1.0];
        let clip: [f32; 4] = std::array::from_fn(|row| (0..4).map(|k| vp[k][row] * p[k]).sum());
        assert!(clip[3] > 0.0, "w should be positive: {clip:?}");
        let ndc = [clip[0] / clip[3], clip[1] / clip[3]];
        assert!(ndc[0].abs() < 1e-4 && ndc[1].abs() < 1e-4, "got {ndc:?}");
    }

    #[test]
    fn spot_light_view_proj_off_axis_point_escapes_a_narrow_cone() {
        // A point well outside a narrow 5° cone's frustum must land outside
        // [-1,1] NDC — proves the FOV genuinely tracks outer_cone_degrees,
        // not some fixed/wide default.
        let vp = spot_light_view_proj(
            [0.0, 0.0, 10.0],
            [0.0, 0.0, -1.0],
            5.0,
            [0.0, 0.0, 0.0],
            2.0,
        );
        let p = [5.0, 0.0, 0.0, 1.0];
        let clip: [f32; 4] = std::array::from_fn(|row| (0..4).map(|k| vp[k][row] * p[k]).sum());
        assert!(clip[3] > 0.0);
        let ndc_x = clip[0] / clip[3];
        assert!(
            ndc_x.abs() > 1.0,
            "off-axis point should escape a narrow cone: ndc_x={ndc_x}"
        );
    }

    #[test]
    fn spot_light_view_proj_wide_cone_keeps_the_same_point_in_frustum() {
        // The same off-axis point from the previous test, but with a wide
        // 60° cone — must now land inside [-1,1].
        let vp = spot_light_view_proj(
            [0.0, 0.0, 10.0],
            [0.0, 0.0, -1.0],
            60.0,
            [0.0, 0.0, 0.0],
            2.0,
        );
        let p = [1.0, 0.0, 0.0, 1.0];
        let clip: [f32; 4] = std::array::from_fn(|row| (0..4).map(|k| vp[k][row] * p[k]).sum());
        assert!(clip[3] > 0.0);
        let ndc_x = clip[0] / clip[3];
        assert!(
            ndc_x.abs() <= 1.0,
            "point should be inside a wide cone: ndc_x={ndc_x}"
        );
    }
}
