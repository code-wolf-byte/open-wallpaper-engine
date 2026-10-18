//! Wallpaper Engine puppet-warp models (`.mdl`): mesh, skeleton, and bone
//! animation.
//!
//! A model JSON with a `"puppet"` field stores its texture as a packed UV
//! atlas; drawing that atlas as a plain quad shows scrambled body parts.
//! Three sections make up the file (all reverse-engineered from real
//! content — the reference only reads the mesh and renders it statically):
//!
//! - `MDLV`: triangle mesh whose positions ARE the atlas layout (identity
//!   position↔UV mapping) plus 4-bone skin indices/weights per vertex.
//! - `MDLS`: bone hierarchy; each bone record carries its local bind
//!   transform *in atlas space* (composing through parents lands on the
//!   bone's vertices in the packed layout).
//! - `MDLA`: named animations; per bone, per frame, an *absolute local
//!   TRS in assembled space*.
//!
//! Skinning with `worldAnim(t) * inverse(worldBind)` therefore both
//! assembles the scattered parts into the character AND animates it — the
//! two are the same operation. CPU-rasterizing the posed mesh keeps every
//! downstream path (effects, GPU compositing) treating the layer as a
//! plain image.

use image::RgbaImage;

/// 9-byte version marker ("MDLV0021\0").
const MARKER_SIZE: usize = 9;
/// `[u32 unknown][u32 vertexBytes]` prefix before the vertex data.
const MESH_HEADER_SIZE: usize = 8;
/// Vertex layouts by format version: position is always 3xf32 at offset 0,
/// UV is 2xf32 at the tail, skinning data (indices + weights, 32 bytes)
/// right before it. MDLV0021/0023 use 80-byte vertices (UV at 72) — the only
/// versions the reference accepts; MDLV0016 (older workshop content, e.g.
/// 2952574984) uses 52-byte vertices (UV at 44), reverse-engineered from
/// real data since the reference just rejects it. Some MDLV0013 files
/// (2349863384's "spaced out") carry no skinning at all: bare 20-byte
/// `[x y z u v]` vertices. Widest strides first — the block scan validates
/// itself, so the narrow layout only matches when nothing else does.
const LAYOUTS: [(usize, usize); 3] = [(80, 72), (52, 44), (20, 12)];

pub struct PuppetMesh {
    /// Rest-pose positions, centered: pixel = (w/2 + x, h/2 - y).
    pub positions: Vec<[f32; 2]>,
    pub uvs: Vec<[f32; 2]>,
    /// Triangle list into `positions`/`uvs`.
    pub indices: Vec<u16>,
    /// Per-vertex 4-bone linear-blend skinning (vertex bytes 12..28 =
    /// 4xu32 bone indices, 28..44 = 4xf32 weights in the MDLV0016 layout;
    /// real content observed so far is rigid — weight 1.0 on index 0).
    /// Empty when the layout has no room for skinning data.
    pub bone_indices: Vec<[u32; 4]>,
    pub weights: Vec<[f32; 4]>,
}

/// Row-major 3x4 affine transform (rows of [x y z t]).
pub type Affine = [[f32; 4]; 3];

/// One skeleton bone (MDLS section): `[u8 flag][u32][i32 parent]
/// [u32 64][16xf32 matrix][json\0]` per record, parents always earlier in
/// the list.
///
/// The record's 4x4 matrix is the bone's *local bind transform in atlas
/// space* — composing it through the parent chain lands on the per-bone
/// vertex centroids of the packed atlas layout (verified empirically on
/// real content). Animation tracks, by contrast, store the bone's local
/// TRS in *assembled* space; `worldAnim * inverse(worldBind)` is therefore
/// exactly the transform that assembles the scattered atlas parts into the
/// posed character.
pub struct PuppetBone {
    pub parent: i32,
    /// Local bind transform (atlas space), row-major 3x4.
    pub bind_local: Affine,
}

/// Local transform sample: 36 bytes per frame in an MDLA track.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BoneFrame {
    pub position: [f32; 3],
    /// Euler angles, radians (z = screen-plane rotation; x/y appear on
    /// pseudo-3D flips of flat parts like hair).
    pub angles: [f32; 3],
    pub scale: [f32; 3],
}

impl BoneFrame {
    pub const IDENTITY: BoneFrame = BoneFrame {
        position: [0.0; 3],
        angles: [0.0; 3],
        scale: [1.0; 3],
    };
}

pub struct PuppetAnimation {
    /// The id `scene.json`'s `animationlayers[].animation` references. Stored
    /// as `[u32 id][u32 0]` just before each animation's name (the first one's
    /// sits in the MDLA header), so layers can resolve by id instead of
    /// guessing an index.
    pub id: u32,
    pub name: String,
    /// `"loop"` observed; other modes treated as loop.
    pub mode: String,
    pub fps: f32,
    pub frame_count: u32,
    /// Per bone: `frame_count + 1` samples (inclusive end, so looped
    /// interpolation needs no wrap special-case), or empty when the file
    /// carries no track for that bone.
    pub tracks: Vec<Vec<BoneFrame>>,
}

pub struct PuppetModel {
    pub mesh: PuppetMesh,
    pub bones: Vec<PuppetBone>,
    pub animations: Vec<PuppetAnimation>,
}

fn f32_at(data: &[u8], offset: usize) -> Option<f32> {
    Some(f32::from_le_bytes(
        data.get(offset..offset + 4)?.try_into().ok()?,
    ))
}

fn u32_at(data: &[u8], offset: usize) -> Option<u32> {
    Some(u32::from_le_bytes(
        data.get(offset..offset + 4)?.try_into().ok()?,
    ))
}

/// Parse a puppet `.mdl`. Mirrors the reference's heuristic scan
/// (CImage.cpp `findPuppetMeshBlock`): the vertex/index block has no fixed
/// offset, so every byte offset before the `MDLS` marker is tested for a
/// `[u32][u32 vertexBytes][vertices][u32 indexBytes][u16 indices]` shape
/// whose sizes divide evenly and fit.
pub fn parse_mdl(data: &[u8]) -> Option<PuppetMesh> {
    let version = data.get(..8)?;
    if !version.starts_with(b"MDLV00") {
        return None;
    }

    let mdls_offset = (MARKER_SIZE..data.len().saturating_sub(4))
        .find(|&o| &data[o..o + 4] == b"MDLS")
        .unwrap_or(data.len());

    // Try the newer layout first; the block scan is self-validating (sizes
    // must divide evenly, fit before MDLS, and every index must be in
    // range), so a wrong stride simply finds nothing.
    LAYOUTS
        .iter()
        .find_map(|&(stride, uv_offset)| parse_mesh_block(data, mdls_offset, stride, uv_offset))
}

fn parse_mesh_block(
    data: &[u8],
    mdls_offset: usize,
    stride: usize,
    uv_offset: usize,
) -> Option<PuppetMesh> {
    for offset in MARKER_SIZE..mdls_offset.saturating_sub(MESH_HEADER_SIZE + 4) {
        let Some(vertex_bytes) = u32_at(data, offset + 4).map(|v| v as usize) else {
            continue;
        };
        let vertices_offset = offset + MESH_HEADER_SIZE;
        let index_len_offset = vertices_offset + vertex_bytes;
        if vertex_bytes < stride * 3
            || vertex_bytes % stride != 0
            || index_len_offset + 4 > mdls_offset
        {
            continue;
        }
        let Some(index_bytes) = u32_at(data, index_len_offset).map(|v| v as usize) else {
            continue;
        };
        let indices_offset = index_len_offset + 4;
        if index_bytes == 0 || index_bytes % 6 != 0 || indices_offset + index_bytes > mdls_offset {
            continue;
        }

        let vertex_count = vertex_bytes / stride;
        let mut positions = Vec::with_capacity(vertex_count);
        let mut uvs = Vec::with_capacity(vertex_count);
        // Skinning data (4 u32 bone indices + 4 f32 weights) sits right
        // before the UV tail: offset 12 in 52-byte vertices, 40 in 80-byte
        // ones, which carry a normal and tangent after the position.
        let has_skin = uv_offset >= 12 + 32;
        let skin = uv_offset.saturating_sub(32);
        let mut bone_indices = Vec::with_capacity(if has_skin { vertex_count } else { 0 });
        let mut weights = Vec::with_capacity(if has_skin { vertex_count } else { 0 });
        for i in 0..vertex_count {
            let v = vertices_offset + i * stride;
            positions.push([f32_at(data, v)?, f32_at(data, v + 4)?]);
            uvs.push([
                f32_at(data, v + uv_offset)?,
                f32_at(data, v + uv_offset + 4)?,
            ]);
            if has_skin {
                bone_indices.push([
                    u32_at(data, v + skin)?,
                    u32_at(data, v + skin + 4)?,
                    u32_at(data, v + skin + 8)?,
                    u32_at(data, v + skin + 12)?,
                ]);
                weights.push([
                    f32_at(data, v + skin + 16)?,
                    f32_at(data, v + skin + 20)?,
                    f32_at(data, v + skin + 24)?,
                    f32_at(data, v + skin + 28)?,
                ]);
            }
        }

        let index_count = index_bytes / 2;
        let mut indices = Vec::with_capacity(index_count);
        let mut valid = true;
        for i in 0..index_count {
            let o = indices_offset + i * 2;
            let idx = u16::from_le_bytes([data[o], data[o + 1]]);
            if idx as usize >= vertex_count {
                valid = false;
                break;
            }
            indices.push(idx);
        }
        if !valid {
            continue;
        }

        return Some(PuppetMesh {
            positions,
            uvs,
            indices,
            bone_indices,
            weights,
        });
    }

    None
}

/// Parse a full puppet model: mesh (MDLV), skeleton (MDLS), animations
/// (MDLA). Mesh-only files still parse (empty skeleton/animations) so the
/// static rest-pose path keeps working on content without animation data.
pub fn parse_model(data: &[u8]) -> Option<PuppetModel> {
    let mesh = parse_mdl(data)?;
    let bones = parse_mdls(data).unwrap_or_default();
    let animations = if bones.is_empty() {
        Vec::new()
    } else {
        parse_mdla(data, bones.len()).unwrap_or_default()
    };
    Some(PuppetModel {
        mesh,
        bones,
        animations,
    })
}

/// MDLS skeleton section: `"MDLS####\0"[u32 sectionEnd][u32 boneCount]`
/// then per bone `[name\0][u32 flag][i32 parent][u32 64][16xf32][json\0]`.
/// Only the parent index is needed (see `PuppetBone`).
///
/// The leading name is usually empty, which is why reading the record as if
/// it began at the flag worked at all: the name's lone NUL stood in for the
/// flag's first byte. 2321732083's samurai rigs bones 15-16 ("strap end",
/// "strap hand"), and there the offsets slid by the name's length, `matlen`
/// read as 256, and the whole skeleton was dropped — leaving 0 bones, so the
/// puppet fell back to rasterizing its raw UV atlas as a white blob.
fn parse_mdls(data: &[u8]) -> Option<Vec<PuppetBone>> {
    let s = find_marker(data, b"MDLS")?;
    let count = u32_at(data, s + 13)? as usize;
    if count == 0 || count > 4096 {
        return None;
    }
    let mut off = s + 17;
    let mut bones = Vec::with_capacity(count);
    for _ in 0..count {
        let name_end = data[off..].iter().position(|&b| b == 0)? + off;
        let rec = name_end + 1;
        let parent = i32::from_le_bytes(data.get(rec + 4..rec + 8)?.try_into().ok()?);
        let matlen = u32_at(data, rec + 8)? as usize;
        if matlen != 64 {
            return None;
        }
        // Column-major 4x4 (translation at elements 12..14): convert to our
        // row-major 3x4.
        let m = rec + 12;
        let e = |i: usize| f32_at(data, m + i * 4);
        let bind_local: Affine = [
            [e(0)?, e(4)?, e(8)?, e(12)?],
            [e(1)?, e(5)?, e(9)?, e(13)?],
            [e(2)?, e(6)?, e(10)?, e(14)?],
        ];
        let js_start = m + matlen;
        let js_end = data[js_start..].iter().position(|&b| b == 0)? + js_start;
        bones.push(PuppetBone { parent, bind_local });
        off = js_end + 1;
    }
    Some(bones)
}

/// MDLA animation section: `"MDLA####\0"[u32 fileEnd][u32 animCount]
/// [u32][u32]` then per animation `name\0 mode\0 [f32 fps][u32 frames]
/// [u32 0][u32 boneCount][u32 0]` followed by one track per bone:
/// `[u32 byteLen][byteLen bytes][u32 0]` where the payload is
/// `frames + 1` samples of 9xf32 (position, angles, scale — absolute
/// local TRS). Animations are separated by 13 bytes of flags/padding.
fn parse_mdla(data: &[u8], bone_count: usize) -> Option<Vec<PuppetAnimation>> {
    let s = find_marker(data, b"MDLA")?;
    let anim_count = u32_at(data, s + 13)? as usize;
    if anim_count == 0 || anim_count > 256 {
        return None;
    }
    let mut p = s + 25;
    // The header's third u32 is the first animation's id.
    let mut id = u32_at(data, s + 17)?;
    let mut animations = Vec::with_capacity(anim_count);
    for a in 0..anim_count {
        let name_end = data[p..].iter().position(|&b| b == 0)? + p;
        let name = String::from_utf8_lossy(&data[p..name_end]).into_owned();
        let mode_end = data[name_end + 1..].iter().position(|&b| b == 0)? + name_end + 1;
        let mode = String::from_utf8_lossy(&data[name_end + 1..mode_end]).into_owned();
        let mut q = mode_end + 1;
        let fps = f32_at(data, q)?;
        let frame_count = u32_at(data, q + 4)?;
        let track_count = u32_at(data, q + 12)? as usize;
        if track_count != bone_count || frame_count == 0 || !(fps > 0.0) {
            return None;
        }
        q += 20;

        let mut tracks = Vec::with_capacity(track_count);
        for _ in 0..track_count {
            let len = u32_at(data, q)? as usize;
            q += 4;
            let mut frames = Vec::with_capacity(len / 36);
            if len % 36 == 0 {
                for f in 0..len / 36 {
                    let o = q + f * 36;
                    frames.push(BoneFrame {
                        position: [f32_at(data, o)?, f32_at(data, o + 4)?, f32_at(data, o + 8)?],
                        angles: [
                            f32_at(data, o + 12)?,
                            f32_at(data, o + 16)?,
                            f32_at(data, o + 20)?,
                        ],
                        scale: [
                            f32_at(data, o + 24)?,
                            f32_at(data, o + 28)?,
                            f32_at(data, o + 32)?,
                        ],
                    });
                }
            }
            q += len;
            // Trailing u32 0 after each track payload.
            q += 4;
            tracks.push(frames);
        }

        animations.push(PuppetAnimation {
            id,
            name,
            mode,
            fps,
            frame_count,
            tracks,
        });
        if a + 1 < anim_count {
            // The padding between animations is not a fixed width (13 bytes in
            // MDLA0001, 39 in MDLA0006's samurai rig), so find the next record
            // by its shape instead of trusting a constant: `[u32 id][u32 0]`
            // ahead of a printable name, followed by a mode and a plausible
            // fps/frame/track header.
            let (next, next_id) = find_next_animation(data, q, bone_count)?;
            p = next;
            id = next_id;
        }
    }
    Some(animations)
}

/// Locate the animation record following `from`, returning its name offset and
/// id. Validates the whole `[id][0] name\0 mode\0 fps frames 0 boneCount 0`
/// shape so a wrong guess is rejected rather than silently mis-parsed.
fn find_next_animation(data: &[u8], from: usize, bone_count: usize) -> Option<(usize, u32)> {
    for name_at in from..(from + 256).min(data.len()) {
        if name_at < 8 {
            continue;
        }
        if u32_at(data, name_at - 4) != Some(0) {
            continue;
        }
        let Some(id) = u32_at(data, name_at - 8) else {
            continue;
        };
        let Some(rel) = data[name_at..].iter().position(|&b| b == 0) else {
            continue;
        };
        let name = &data[name_at..name_at + rel];
        if name.is_empty() || !name.iter().all(|b| (0x20..0x7f).contains(b)) {
            continue;
        }
        let mode_at = name_at + rel + 1;
        let Some(mode_rel) = data.get(mode_at..)?.iter().position(|&b| b == 0) else {
            continue;
        };
        if mode_rel == 0
            || !data[mode_at..mode_at + mode_rel]
                .iter()
                .all(|b| (0x20..0x7f).contains(b))
        {
            continue;
        }
        let q = mode_at + mode_rel + 1;
        let fps = f32_at(data, q)?;
        let frames = u32_at(data, q + 4)?;
        let tracks = u32_at(data, q + 12)? as usize;
        if fps > 0.0 && frames > 0 && tracks == bone_count {
            return Some((name_at, id));
        }
    }
    None
}

fn find_marker(data: &[u8], marker: &[u8; 4]) -> Option<usize> {
    (0..data.len().saturating_sub(4)).find(|&o| &data[o..o + 4] == marker)
}

const AFFINE_IDENTITY: Affine = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
];

fn affine_mul(a: &Affine, b: &Affine) -> Affine {
    let mut out = [[0.0f32; 4]; 3];
    for (r, row) in out.iter_mut().enumerate() {
        for c in 0..4 {
            row[c] = a[r][0] * b[0][c] + a[r][1] * b[1][c] + a[r][2] * b[2][c];
            if c == 3 {
                row[c] += a[r][3];
            }
        }
    }
    out
}

fn affine_apply(m: &Affine, p: [f32; 3]) -> [f32; 3] {
    let mut out = [0.0f32; 3];
    for (r, row) in m.iter().enumerate() {
        out[r] = row[0] * p[0] + row[1] * p[1] + row[2] * p[2] + row[3];
    }
    out
}

/// Invert an affine transform (general 3x3 inverse + translation).
fn affine_invert(m: &Affine) -> Affine {
    let a = m;
    let det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
    if det.abs() < 1e-12 {
        return AFFINE_IDENTITY;
    }
    let inv_det = 1.0 / det;
    let mut inv = [[0.0f32; 4]; 3];
    inv[0][0] = (a[1][1] * a[2][2] - a[1][2] * a[2][1]) * inv_det;
    inv[0][1] = (a[0][2] * a[2][1] - a[0][1] * a[2][2]) * inv_det;
    inv[0][2] = (a[0][1] * a[1][2] - a[0][2] * a[1][1]) * inv_det;
    inv[1][0] = (a[1][2] * a[2][0] - a[1][0] * a[2][2]) * inv_det;
    inv[1][1] = (a[0][0] * a[2][2] - a[0][2] * a[2][0]) * inv_det;
    inv[1][2] = (a[0][2] * a[1][0] - a[0][0] * a[1][2]) * inv_det;
    inv[2][0] = (a[1][0] * a[2][1] - a[1][1] * a[2][0]) * inv_det;
    inv[2][1] = (a[0][1] * a[2][0] - a[0][0] * a[2][1]) * inv_det;
    inv[2][2] = (a[0][0] * a[1][1] - a[0][1] * a[1][0]) * inv_det;
    for r in 0..3 {
        inv[r][3] = -(inv[r][0] * a[0][3] + inv[r][1] * a[1][3] + inv[r][2] * a[2][3]);
    }
    inv
}

/// Local transform T * Rz * Ry * Rx * S (WE's screen-plane rotation is z;
/// x/y Euler flips compose after it, matching scene-object angle order).
fn bone_frame_affine(f: &BoneFrame) -> Affine {
    let (sx, cx) = f.angles[0].sin_cos();
    let (sy, cy) = f.angles[1].sin_cos();
    let (sz, cz) = f.angles[2].sin_cos();
    // R = Rz * Ry * Rx, rows scaled by S, translation appended.
    let r = [
        [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
        [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
        [-sy, cy * sx, cy * cx],
    ];
    [
        [
            r[0][0] * f.scale[0],
            r[0][1] * f.scale[1],
            r[0][2] * f.scale[2],
            f.position[0],
        ],
        [
            r[1][0] * f.scale[0],
            r[1][1] * f.scale[1],
            r[1][2] * f.scale[2],
            f.position[1],
        ],
        [
            r[2][0] * f.scale[0],
            r[2][1] * f.scale[1],
            r[2][2] * f.scale[2],
            f.position[2],
        ],
    ]
}

/// Sample a track at a (fractional) frame, linearly interpolating between
/// the two neighbouring samples. Tracks carry `frame_count + 1` samples
/// (inclusive loop end), so `frame` in `[0, frame_count]` never wraps.
fn sample_track(track: &[BoneFrame], frame: f32) -> BoneFrame {
    if track.is_empty() {
        return BoneFrame::IDENTITY;
    }
    let i = (frame.floor() as usize).min(track.len() - 1);
    let j = (i + 1).min(track.len() - 1);
    let t = frame - i as f32;
    let (a, b) = (&track[i], &track[j]);
    let lerp3 = |x: [f32; 3], y: [f32; 3]| -> [f32; 3] {
        [
            x[0] + (y[0] - x[0]) * t,
            x[1] + (y[1] - x[1]) * t,
            x[2] + (y[2] - x[2]) * t,
        ]
    };
    BoneFrame {
        position: lerp3(a.position, b.position),
        angles: lerp3(a.angles, b.angles),
        scale: lerp3(a.scale, b.scale),
    }
}

/// Per-bone world transforms for `animation` at time `t` (seconds),
/// looped. Parents are guaranteed to precede children in the bone list
/// (validated in real content), so one forward pass composes the chain.
fn world_transforms(model: &PuppetModel, animation: &PuppetAnimation, t: f32) -> Vec<Affine> {
    let frame = (t * animation.fps).rem_euclid(animation.frame_count as f32);
    let mut world: Vec<Affine> = Vec::with_capacity(model.bones.len());
    for (b, bone) in model.bones.iter().enumerate() {
        let local = bone_frame_affine(&sample_track(
            animation.tracks.get(b).map(Vec::as_slice).unwrap_or(&[]),
            frame,
        ));
        let w = match bone.parent {
            p if p >= 0 && (p as usize) < world.len() => affine_mul(&world[p as usize], &local),
            _ => local,
        };
        world.push(w);
    }
    world
}

/// Skinning matrices at time `t`: `worldAnim(t) * inverse(worldBind)`.
///
/// `worldBind` composes the bone records' atlas-space bind transforms;
/// `worldAnim` composes the animation tracks' assembled-space local TRS —
/// so the skin transform is what moves each packed atlas part onto the
/// posed, assembled character.
pub struct PuppetPose {
    skin: Vec<Affine>,
}

impl PuppetModel {
    /// Whether there's anything to animate.
    pub fn has_animation(&self) -> bool {
        !self.animations.is_empty() && !self.bones.is_empty()
    }

    pub fn pose_at(&self, animation_idx: usize, t: f32) -> Option<PuppetPose> {
        let anim = self.animations.get(animation_idx)?;
        let bind = self.bind_world_transforms();
        let now = world_transforms(self, anim, t);
        let skin = bind
            .iter()
            .zip(&now)
            .map(|(b, n)| affine_mul(n, &affine_invert(b)))
            .collect();
        Some(PuppetPose { skin })
    }

    /// Pose from blended `animationlayers` (a base override layer plus additive
    /// deltas), each sampled at its own rate. Falls back to `None` when no
    /// layer resolves to a real animation.
    pub fn pose_at_layers(&self, layers: &[AnimLayer], t: f32) -> Option<PuppetPose> {
        if layers
            .iter()
            .all(|l| self.animations.get(l.anim_idx).is_none())
        {
            return None;
        }
        let bind = self.bind_world_transforms();
        let mut world: Vec<Affine> = Vec::with_capacity(self.bones.len());
        for (b, bone) in self.bones.iter().enumerate() {
            let local = bone_frame_affine(&self.blended_local(layers, b, t));
            let w = match bone.parent {
                p if p >= 0 && (p as usize) < world.len() => affine_mul(&world[p as usize], &local),
                _ => local,
            };
            world.push(w);
        }
        let skin = bind
            .iter()
            .zip(&world)
            .map(|(b, n)| affine_mul(n, &affine_invert(b)))
            .collect();
        Some(PuppetPose { skin })
    }

    /// Blend one bone's local TRS across the animation layers: the first
    /// resolvable override layer is the base (an additive first layer starts
    /// from its own frame 0), then each subsequent override lerps and each
    /// additive adds its delta-from-its-own-frame-0, weighted by `blend`.
    fn blended_local(&self, layers: &[AnimLayer], bone: usize, t: f32) -> BoneFrame {
        let mut accum: Option<BoneFrame> = None;
        for layer in layers {
            let Some(anim) = self.animations.get(layer.anim_idx) else {
                continue;
            };
            let track = anim.tracks.get(bone).map(Vec::as_slice).unwrap_or(&[]);
            let frame = (t * layer.rate * anim.fps).rem_euclid(anim.frame_count as f32);
            let cur = sample_track(track, frame);
            if accum.is_none() && !layer.additive {
                accum = Some(cur);
                continue;
            }
            let rest = sample_track(track, 0.0);
            let a = accum.get_or_insert(rest);
            let w = layer.blend.clamp(0.0, 1.0);
            if layer.additive {
                for i in 0..3 {
                    a.position[i] += (cur.position[i] - rest.position[i]) * w;
                    a.angles[i] += (cur.angles[i] - rest.angles[i]) * w;
                    a.scale[i] += (cur.scale[i] - rest.scale[i]) * w;
                }
            } else {
                for i in 0..3 {
                    a.position[i] += (cur.position[i] - a.position[i]) * w;
                    a.angles[i] += (cur.angles[i] - a.angles[i]) * w;
                    a.scale[i] += (cur.scale[i] - a.scale[i]) * w;
                }
            }
        }
        accum.unwrap_or(BoneFrame::IDENTITY)
    }

    /// Composed atlas-space bind transforms (see `PuppetBone::bind_local`).
    fn bind_world_transforms(&self) -> Vec<Affine> {
        let mut world: Vec<Affine> = Vec::with_capacity(self.bones.len());
        for bone in &self.bones {
            let w = match bone.parent {
                p if p >= 0 && (p as usize) < world.len() => {
                    affine_mul(&world[p as usize], &bone.bind_local)
                }
                _ => bone.bind_local,
            };
            world.push(w);
        }
        world
    }

    /// Skinned vertex positions (same centered y-up space as the mesh) via
    /// 4-bone linear blending; vertices without valid weights stay at rest.
    pub fn skinned_positions(&self, pose: &PuppetPose) -> Vec<[f32; 2]> {
        let mesh = &self.mesh;
        mesh.positions
            .iter()
            .enumerate()
            .map(|(i, &rest)| {
                let (Some(indices), Some(weights)) =
                    (mesh.bone_indices.get(i), mesh.weights.get(i))
                else {
                    return rest;
                };
                let total: f32 = weights.iter().sum();
                if !(total > 0.01) {
                    return rest;
                }
                let p = [rest[0], rest[1], 0.0];
                let mut out = [0.0f32; 2];
                for k in 0..4 {
                    let w = weights[k] / total;
                    if w <= 0.0 {
                        continue;
                    }
                    let m = pose
                        .skin
                        .get(indices[k] as usize)
                        .unwrap_or(&AFFINE_IDENTITY);
                    let q = affine_apply(m, p);
                    out[0] += q[0] * w;
                    out[1] += q[1] * w;
                }
                out
            })
            .collect()
    }
}

/// Everything needed to re-pose a puppet layer at runtime: the parsed
/// model plus its packed UV atlas (the layer's decoded texture).
pub struct PuppetRuntime {
    pub model: PuppetModel,
    pub atlas: RgbaImage,
    /// Scene `animationlayers` blended each frame. Empty = play animation 0.
    pub layers: Vec<AnimLayer>,
}

/// One entry of a scene object's `animationlayers`: which model animation to
/// play, how to blend it, its weight, and its own playback rate.
#[derive(Clone, Copy)]
pub struct AnimLayer {
    pub anim_idx: usize,
    /// `additive`: add this animation's delta-from-its-rest instead of
    /// overriding (blinks/expressions layered on a base idle).
    pub additive: bool,
    pub blend: f32,
    pub rate: f32,
}

impl PuppetRuntime {
    /// Rasterize the puppet at time `t` (seconds into the looped first
    /// animation) at `width` x `height`.
    pub fn render_at(&self, t: f32, width: u32, height: u32) -> RgbaImage {
        let pose = if self.layers.is_empty() {
            self.model.pose_at(0, t)
        } else {
            self.model
                .pose_at_layers(&self.layers, t)
                .or_else(|| self.model.pose_at(0, t))
        };
        match pose {
            Some(pose) => {
                let skinned = self.model.skinned_positions(&pose);
                rasterize_positions(&self.model.mesh, &skinned, &self.atlas, width, height)
            }
            None => rasterize(&self.model.mesh, &self.atlas, width, height),
        }
    }
}

/// Rasterize the rest-pose mesh: assembles the packed `atlas` into a
/// `width` x `height` image via each triangle's UV mapping. Mesh positions
/// are centered, y-up (the reference's `updatePuppetPositionBuffer` maps
/// them as `(w/2 + x, h/2 - y)`); triangles composite alpha-over in index
/// order, matching the reference's draw order for overlapping parts.
pub fn rasterize(mesh: &PuppetMesh, atlas: &RgbaImage, width: u32, height: u32) -> RgbaImage {
    rasterize_positions(mesh, &mesh.positions, atlas, width, height)
}

/// `rasterize` with externally supplied (e.g. skinned) vertex positions,
/// same centered y-up space as `PuppetMesh::positions`.
pub fn rasterize_positions(
    mesh: &PuppetMesh,
    positions: &[[f32; 2]],
    atlas: &RgbaImage,
    width: u32,
    height: u32,
) -> RgbaImage {
    let mut out = RgbaImage::new(width.max(1), height.max(1));
    let (w, h) = (width as f32, height as f32);

    let to_px = |p: [f32; 2]| -> [f32; 2] { [w / 2.0 + p[0], h / 2.0 - p[1]] };

    for tri in mesh.indices.chunks_exact(3) {
        let (i0, i1, i2) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
        let p0 = to_px(positions[i0]);
        let p1 = to_px(positions[i1]);
        let p2 = to_px(positions[i2]);
        let (uv0, uv1, uv2) = (mesh.uvs[i0], mesh.uvs[i1], mesh.uvs[i2]);

        // Signed double-area; degenerate triangles contribute nothing.
        let area = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0]);
        if area.abs() < 1e-6 {
            continue;
        }

        let min_x = p0[0].min(p1[0]).min(p2[0]).floor().max(0.0) as u32;
        let max_x = (p0[0].max(p1[0]).max(p2[0]).ceil() as i64).clamp(0, width as i64 - 1) as u32;
        let min_y = p0[1].min(p1[1]).min(p2[1]).floor().max(0.0) as u32;
        let max_y = (p0[1].max(p1[1]).max(p2[1]).ceil() as i64).clamp(0, height as i64 - 1) as u32;

        for py in min_y..=max_y {
            for px in min_x..=max_x {
                let x = px as f32 + 0.5;
                let y = py as f32 + 0.5;
                // Barycentric weights, normalized by the signed area so
                // either winding works.
                let w0 = ((p1[0] - x) * (p2[1] - y) - (p1[1] - y) * (p2[0] - x)) / area;
                let w1 = ((p2[0] - x) * (p0[1] - y) - (p2[1] - y) * (p0[0] - x)) / area;
                let w2 = 1.0 - w0 - w1;
                if w0 < 0.0 || w1 < 0.0 || w2 < 0.0 {
                    continue;
                }

                let u = w0 * uv0[0] + w1 * uv1[0] + w2 * uv2[0];
                let v = w0 * uv0[1] + w1 * uv1[1] + w2 * uv2[1];
                let src = sample_bilinear(atlas, u, v);
                if src[3] == 0 {
                    continue;
                }

                // Alpha-over in triangle order.
                let dst = out.get_pixel_mut(px, py);
                let sa = src[3] as f32 / 255.0;
                let da = dst[3] as f32 / 255.0;
                let oa = sa + da * (1.0 - sa);
                if oa > 0.0 {
                    for c in 0..3 {
                        let sc = src[c] as f32 / 255.0;
                        let dc = dst[c] as f32 / 255.0;
                        dst[c] = (((sc * sa + dc * da * (1.0 - sa)) / oa) * 255.0) as u8;
                    }
                }
                dst[3] = (oa * 255.0) as u8;
            }
        }
    }

    out
}

fn sample_bilinear(tex: &RgbaImage, u: f32, v: f32) -> [u8; 4] {
    let (tw, th) = (tex.width(), tex.height());
    if tw == 0 || th == 0 {
        return [0; 4];
    }
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

    let mut o = [0u8; 4];
    for i in 0..4 {
        let top = p00[i] as f32 * (1.0 - tx) + p10[i] as f32 * tx;
        let bot = p01[i] as f32 * (1.0 - tx) + p11[i] as f32 * tx;
        o[i] = (top * (1.0 - ty) + bot * ty).clamp(0.0, 255.0) as u8;
    }
    o
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal MDLV blob: marker, filler, one discoverable mesh
    /// block (3 vertices, 1 triangle), MDLS terminator.
    fn minimal_mdl(positions: [[f32; 2]; 3], uvs: [[f32; 2]; 3]) -> Vec<u8> {
        const STRIDE: usize = 80; // MDLV0021 layout, UV at 72
        const UV: usize = 72;
        let mut d = Vec::new();
        d.extend_from_slice(b"MDLV0021\0");
        d.extend_from_slice(&0u32.to_le_bytes()); // block's leading u32
        d.extend_from_slice(&(3 * STRIDE as u32).to_le_bytes());
        for i in 0..3 {
            let mut v = [0u8; STRIDE];
            v[0..4].copy_from_slice(&positions[i][0].to_le_bytes());
            v[4..8].copy_from_slice(&positions[i][1].to_le_bytes());
            // Normal (0,0,1) after the position, then bone index i at
            // weight 1 at offset 40/56, as real 80-byte vertices lay out.
            v[20..24].copy_from_slice(&1.0f32.to_le_bytes());
            v[40..44].copy_from_slice(&(i as u32).to_le_bytes());
            v[56..60].copy_from_slice(&1.0f32.to_le_bytes());
            v[UV..UV + 4].copy_from_slice(&uvs[i][0].to_le_bytes());
            v[UV + 4..UV + 8].copy_from_slice(&uvs[i][1].to_le_bytes());
            d.extend_from_slice(&v);
        }
        d.extend_from_slice(&6u32.to_le_bytes()); // index bytes
        for idx in [0u16, 1, 2] {
            d.extend_from_slice(&idx.to_le_bytes());
        }
        d.extend_from_slice(b"MDLS");
        d
    }

    #[test]
    fn parses_minimal_mdl() {
        let data = minimal_mdl(
            [[-10.0, -10.0], [10.0, -10.0], [0.0, 10.0]],
            [[0.0, 1.0], [1.0, 1.0], [0.5, 0.0]],
        );
        let mesh = parse_mdl(&data).expect("should parse");
        assert_eq!(mesh.positions.len(), 3);
        assert_eq!(mesh.indices, vec![0, 1, 2]);
        assert_eq!(mesh.uvs[2], [0.5, 0.0]);
        // Skinning comes from just before the UV, not the normal after the
        // position (3598808038's knight had its head skinned to the arm).
        assert_eq!(mesh.bone_indices[2], [2, 0, 0, 0]);
        assert_eq!(mesh.weights[2], [1.0, 0.0, 0.0, 0.0]);
    }

    /// MDLS bone records start with a name string. It is usually empty, so
    /// the old reader (which began at the flag byte) worked by accident until
    /// a rig named a bone — 2321732083's "strap end" slid every field and the
    /// whole skeleton was dropped.
    #[test]
    fn parses_named_bone_records() {
        fn bone(name: &str, parent: i32) -> Vec<u8> {
            let mut b = Vec::new();
            b.extend_from_slice(name.as_bytes());
            b.push(0);
            b.extend_from_slice(&1u32.to_le_bytes()); // flag
            b.extend_from_slice(&parent.to_le_bytes());
            b.extend_from_slice(&64u32.to_le_bytes()); // matrix length
            for i in 0..16 {
                // Column-major identity.
                let v: f32 = if i % 5 == 0 { 1.0 } else { 0.0 };
                b.extend_from_slice(&v.to_le_bytes());
            }
            b.push(0); // empty trailing json
            b
        }
        let mut d = Vec::new();
        d.extend_from_slice(b"MDLS0003\0");
        d.extend_from_slice(&0u32.to_le_bytes()); // section end
        d.extend_from_slice(&2u32.to_le_bytes()); // bone count
        d.extend_from_slice(&bone("", -1));
        d.extend_from_slice(&bone("strap end", 0));
        let bones = parse_mdls(&d).expect("skeleton should parse");
        assert_eq!(bones.len(), 2);
        assert_eq!(bones[0].parent, -1);
        assert_eq!(bones[1].parent, 0);
        assert_eq!(bones[1].bind_local[0][0], 1.0);
    }

    #[test]
    fn rejects_wrong_marker() {
        assert!(parse_mdl(b"NOTAMODEL blah blah blah").is_none());
    }

    /// A triangle covering the canvas center must paint atlas content
    /// there and leave the corners (outside the mesh) transparent — the
    /// exact difference between mesh rendering and the scrambled-atlas
    /// quad draw this replaces.
    #[test]
    fn rasterizes_triangle_with_uv_sampling() {
        let data = minimal_mdl(
            [[-20.0, -20.0], [20.0, -20.0], [0.0, 20.0]],
            [[0.0, 1.0], [1.0, 1.0], [0.5, 0.0]],
        );
        let mesh = parse_mdl(&data).unwrap();
        let atlas = RgbaImage::from_pixel(8, 8, image::Rgba([10, 200, 30, 255]));
        let out = rasterize(&mesh, &atlas, 50, 50);
        assert_eq!(out.get_pixel(25, 25).0, [10, 200, 30, 255]);
        assert_eq!(out.get_pixel(0, 0).0[3], 0, "corner must stay transparent");
        assert_eq!(out.get_pixel(49, 0).0[3], 0);
    }

    fn affine_close(a: &Affine, b: &Affine) -> bool {
        a.iter()
            .flatten()
            .zip(b.iter().flatten())
            .all(|(x, y)| (x - y).abs() < 1e-4)
    }

    #[test]
    fn affine_invert_roundtrips() {
        let m = bone_frame_affine(&BoneFrame {
            position: [12.0, -7.0, 3.0],
            angles: [0.2, -0.4, 1.1],
            scale: [1.5, 0.75, 1.0],
        });
        let ident = affine_mul(&m, &affine_invert(&m));
        assert!(affine_close(&ident, &AFFINE_IDENTITY), "{ident:?}");
    }

    /// Pose at t=0 must be the bind pose: every skinning matrix identity,
    /// so skinned positions equal rest positions exactly. This is the
    /// invariant that keeps animated puppets pixel-identical to the old
    /// static render at frame 0.
    #[test]
    fn pose_at_zero_is_identity_skin() {
        let track = vec![
            BoneFrame {
                position: [5.0, 6.0, 0.0],
                angles: [0.0, 0.0, 0.3],
                scale: [1.0; 3],
            };
            3
        ];
        let model = PuppetModel {
            mesh: PuppetMesh {
                positions: vec![[10.0, 20.0]],
                uvs: vec![[0.0, 0.0]],
                indices: vec![],
                bone_indices: vec![[0, 0, 0, 0]],
                weights: vec![[1.0, 0.0, 0.0, 0.0]],
            },
            bones: vec![PuppetBone {
                parent: -1,
                bind_local: bone_frame_affine(&track[0]),
            }],
            animations: vec![PuppetAnimation {
                id: 0,
                name: "t".into(),
                mode: "loop".into(),
                fps: 30.0,
                frame_count: 2,
                tracks: vec![track.clone()],
            }],
        };
        let pose = model.pose_at(0, 0.0).unwrap();
        let skinned = model.skinned_positions(&pose);
        assert!((skinned[0][0] - 10.0).abs() < 1e-3);
        assert!((skinned[0][1] - 20.0).abs() < 1e-3);
    }

    /// A bone translating over time must drag its vertices with it by the
    /// delta between the current and bind local transforms.
    #[test]
    fn animated_translation_moves_skinned_vertices() {
        let mut track = vec![BoneFrame::IDENTITY; 3];
        track[1].position = [40.0, 0.0, 0.0];
        track[2].position = [40.0, 0.0, 0.0];
        let model = PuppetModel {
            mesh: PuppetMesh {
                positions: vec![[10.0, 20.0]],
                uvs: vec![[0.0, 0.0]],
                indices: vec![],
                bone_indices: vec![[0, 0, 0, 0]],
                weights: vec![[1.0, 0.0, 0.0, 0.0]],
            },
            bones: vec![PuppetBone {
                parent: -1,
                bind_local: bone_frame_affine(&track[0]),
            }],
            animations: vec![PuppetAnimation {
                id: 0,
                name: "t".into(),
                mode: "loop".into(),
                fps: 1.0,
                frame_count: 2,
                tracks: vec![track.clone()],
            }],
        };
        // t=1s at 1fps = frame 1: bone moved +40 in x from bind.
        let pose = model.pose_at(0, 1.0).unwrap();
        let skinned = model.skinned_positions(&pose);
        assert!((skinned[0][0] - 50.0).abs() < 1e-3, "{skinned:?}");
        assert!((skinned[0][1] - 20.0).abs() < 1e-3);
    }

    /// An override base layer plus an additive layer compose: base motion +
    /// the additive animation's delta-from-its-frame-0.
    #[test]
    fn animation_layers_blend_override_plus_additive() {
        let mut base = vec![BoneFrame::IDENTITY; 3]; // anim 0: +40 x
        base[1].position = [40.0, 0.0, 0.0];
        base[2].position = [40.0, 0.0, 0.0];
        let mut add = vec![BoneFrame::IDENTITY; 3]; // anim 1: +60 y delta
        add[1].position = [0.0, 60.0, 0.0];
        add[2].position = [0.0, 60.0, 0.0];
        let anim = |name: &str, track: Vec<BoneFrame>| PuppetAnimation {
            id: 0,
            name: name.into(),
            mode: "loop".into(),
            fps: 1.0,
            frame_count: 2,
            tracks: vec![track],
        };
        let model = PuppetModel {
            mesh: PuppetMesh {
                positions: vec![[10.0, 20.0]],
                uvs: vec![[0.0, 0.0]],
                indices: vec![],
                bone_indices: vec![[0, 0, 0, 0]],
                weights: vec![[1.0, 0.0, 0.0, 0.0]],
            },
            bones: vec![PuppetBone {
                parent: -1,
                bind_local: AFFINE_IDENTITY,
            }],
            animations: vec![anim("base", base), anim("add", add)],
        };
        let layers = [
            AnimLayer {
                anim_idx: 0,
                additive: false,
                blend: 1.0,
                rate: 1.0,
            },
            AnimLayer {
                anim_idx: 1,
                additive: true,
                blend: 1.0,
                rate: 1.0,
            },
        ];
        // t=1s: base local [40,0] + additive delta [0,60] = [40,60].
        let pose = model.pose_at_layers(&layers, 1.0).unwrap();
        let skinned = model.skinned_positions(&pose);
        assert!((skinned[0][0] - 50.0).abs() < 1e-3, "{skinned:?}");
        assert!((skinned[0][1] - 80.0).abs() < 1e-3, "{skinned:?}");

        // An additive first layer still honours its blend: half the +60 y
        // delta on top of rest (3598808038's knight is all additive layers).
        let half = [AnimLayer {
            anim_idx: 1,
            additive: true,
            blend: 0.5,
            rate: 1.0,
        }];
        let pose = model.pose_at_layers(&half, 1.0).unwrap();
        let skinned = model.skinned_positions(&pose);
        assert!((skinned[0][0] - 10.0).abs() < 1e-3, "{skinned:?}");
        assert!((skinned[0][1] - 50.0).abs() < 1e-3, "{skinned:?}");
    }

    /// Child bones must inherit their parent's animated transform.
    #[test]
    fn child_bone_inherits_parent_motion() {
        let mut parent_track = vec![BoneFrame::IDENTITY; 3];
        parent_track[1].position = [0.0, 30.0, 0.0];
        parent_track[2].position = [0.0, 30.0, 0.0];
        let child_track = vec![
            BoneFrame {
                position: [7.0, 0.0, 0.0],
                ..BoneFrame::IDENTITY
            };
            3
        ];
        let model = PuppetModel {
            mesh: PuppetMesh {
                positions: vec![[0.0, 0.0]],
                uvs: vec![[0.0, 0.0]],
                indices: vec![],
                bone_indices: vec![[1, 0, 0, 0]],
                weights: vec![[1.0, 0.0, 0.0, 0.0]],
            },
            bones: vec![
                PuppetBone {
                    parent: -1,
                    bind_local: bone_frame_affine(&parent_track[0]),
                },
                PuppetBone {
                    parent: 0,
                    bind_local: bone_frame_affine(&child_track[0]),
                },
            ],
            animations: vec![PuppetAnimation {
                id: 0,
                name: "t".into(),
                mode: "loop".into(),
                fps: 1.0,
                frame_count: 2,
                tracks: vec![parent_track.clone(), child_track.clone()],
            }],
        };
        let pose = model.pose_at(0, 1.0).unwrap();
        let skinned = model.skinned_positions(&pose);
        assert!((skinned[0][0]).abs() < 1e-3, "{skinned:?}");
        assert!((skinned[0][1] - 30.0).abs() < 1e-3, "{skinned:?}");
    }
}
