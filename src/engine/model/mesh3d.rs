//! Static 3D mesh models — the `.mdl` a scene object references through its
//! `model` field (as opposed to `image`, which points at a flat quad model).
//!
//! These are plain geometry (spheres, skyboxes, cylinders) used by genuine 3D
//! perspective scenes. They are NOT puppets: puppet `.mdl`s carry `MDLS`/`MDLA`
//! (skeleton + animation) and pack their positions as a 2D UV atlas, while
//! these carry neither and store real 3D positions. Neither the C++ reference
//! nor WE's docs describe this layout — it's reverse-engineered from real
//! content (3509243656's star/earth/skybox models).
//!
//! Layout (MDLV0023):
//! ```text
//! "MDLV0023\0" [u32 15][u32 1][u32 1]
//! <material path>\0            e.g. "materials/models/1/Material__50.json"
//! [u32 flag][6 x f32 bbox]     min xyz, max xyz (≈ the unit cube)
//! [u32 15][u32 vertexBytes] <vertices> [u32 indexBytes] <indices>
//! ```
//! Vertex = 48 bytes: `position` 3×f32 @0, `normal` @12, `tangent` @24,
//! tangent handedness @36, `uv` 2×f32 @40. Indices are `u16` when the mesh has
//! ≤ 65536 vertices and `u32` above that (real content uses both: a 2143-vert
//! star is u16, a 520192-vert sphere is u32).

/// Bytes per vertex; UV sits at byte 40, normal at byte 12. Only this one
/// layout appears in real content, unlike puppet meshes which have two.
const STRIDE: usize = 48;
const NORMAL_OFFSET: usize = 12;
const UV_OFFSET: usize = 40;

pub struct Mesh3d {
    /// Real 3D positions (the mesh's own object space, ≈ a unit cube).
    pub positions: Vec<[f32; 3]>,
    /// Object-space vertex normals, parallel to `positions`. Used by the
    /// mesh3d lighting pass; not validated to be unit-length (real content is,
    /// but callers normalize defensively).
    pub normals: Vec<[f32; 3]>,
    pub uvs: Vec<[f32; 2]>,
    /// Always widened to u32 so both index widths share one draw path.
    pub indices: Vec<u32>,
    /// Material JSON path embedded in the header — how the texture resolves.
    pub material: String,
}

fn u32_at(d: &[u8], o: usize) -> Option<u32> {
    Some(u32::from_le_bytes(d.get(o..o + 4)?.try_into().ok()?))
}

fn f32_at(d: &[u8], o: usize) -> Option<f32> {
    Some(f32::from_le_bytes(d.get(o..o + 4)?.try_into().ok()?))
}

/// Parse a static 3D mesh. Returns `None` for anything that isn't one (puppet
/// meshes carry `MDLS`, so they're rejected here and handled by `puppet.rs`).
pub fn parse(data: &[u8]) -> Option<Mesh3d> {
    if !data.get(..6)?.starts_with(b"MDLV00") {
        return None;
    }
    // A skeleton means it's a puppet, not a static mesh.
    if find(data, b"MDLS").is_some() {
        return None;
    }
    let mat_start = find(data, b"materials/")?;
    let mat_end = mat_start + data[mat_start..].iter().position(|&b| b == 0)?;
    let material = String::from_utf8_lossy(&data[mat_start..mat_end]).into_owned();

    // The block follows the [u32 flag][6xf32 bbox] after the path, but scan a
    // small window rather than hardcode the offset: the header has varied
    // (the material path length shifts it, and the leading fields differ
    // between files). The shape check below is self-validating.
    for off in mat_end + 1..(mat_end + 96).min(data.len().saturating_sub(12)) {
        let Some(mesh) = try_block(data, off, &material) else {
            continue;
        };
        return Some(mesh);
    }
    None
}

/// Validate `[u32][u32 vertexBytes]<verts>[u32 indexBytes]<indices>` at `off`.
fn try_block(data: &[u8], off: usize, material: &str) -> Option<Mesh3d> {
    let vertex_bytes = u32_at(data, off + 4)? as usize;
    // At least one triangle, evenly divided, and it must fit.
    if vertex_bytes < STRIDE * 3 || vertex_bytes % STRIDE != 0 {
        return None;
    }
    let verts_off = off + 8;
    let idx_len_off = verts_off + vertex_bytes;
    let index_bytes = u32_at(data, idx_len_off)? as usize;
    let idx_off = idx_len_off + 4;
    if index_bytes == 0 || idx_off + index_bytes > data.len() {
        return None;
    }
    let vertex_count = vertex_bytes / STRIDE;
    // Index width follows the vertex count (u16 up to 65536, else u32).
    let width = if vertex_count <= u16::MAX as usize + 1 {
        2
    } else {
        4
    };
    if index_bytes % (width * 3) != 0 {
        return None;
    }
    let index_count = index_bytes / width;

    let mut indices = Vec::with_capacity(index_count);
    for k in 0..index_count {
        let o = idx_off + k * width;
        let i = if width == 2 {
            u16::from_le_bytes(data.get(o..o + 2)?.try_into().ok()?) as u32
        } else {
            u32_at(data, o)?
        };
        // Every index must address a real vertex — this is what makes the
        // scan self-validating against a wrong offset.
        if i as usize >= vertex_count {
            return None;
        }
        indices.push(i);
    }

    let mut positions = Vec::with_capacity(vertex_count);
    let mut normals = Vec::with_capacity(vertex_count);
    let mut uvs = Vec::with_capacity(vertex_count);
    for i in 0..vertex_count {
        let v = verts_off + i * STRIDE;
        positions.push([f32_at(data, v)?, f32_at(data, v + 4)?, f32_at(data, v + 8)?]);
        normals.push([
            f32_at(data, v + NORMAL_OFFSET)?,
            f32_at(data, v + NORMAL_OFFSET + 4)?,
            f32_at(data, v + NORMAL_OFFSET + 8)?,
        ]);
        uvs.push([
            f32_at(data, v + UV_OFFSET)?,
            f32_at(data, v + UV_OFFSET + 4)?,
        ]);
    }

    Some(Mesh3d {
        positions,
        normals,
        uvs,
        indices,
        material: material.to_string(),
    })
}

fn find(data: &[u8], needle: &[u8]) -> Option<usize> {
    data.windows(needle.len()).position(|w| w == needle)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal well-formed static mesh: header, material path, bbox,
    /// then one triangle.
    fn synth(vertex_count: usize) -> Vec<u8> {
        let mut d = Vec::new();
        d.extend_from_slice(b"MDLV0023\0");
        d.extend_from_slice(&15u32.to_le_bytes());
        d.extend_from_slice(&1u32.to_le_bytes());
        d.extend_from_slice(&1u32.to_le_bytes());
        d.extend_from_slice(b"materials/models/t/Material.json\0");
        d.extend_from_slice(&0u32.to_le_bytes()); // flag
        for v in [-1.0f32, -1.0, -1.0, 1.0, 1.0, 1.0] {
            d.extend_from_slice(&v.to_le_bytes());
        }
        d.extend_from_slice(&15u32.to_le_bytes());
        d.extend_from_slice(&((vertex_count * STRIDE) as u32).to_le_bytes());
        for i in 0..vertex_count {
            let mut v = [0u8; STRIDE];
            // position
            v[0..4].copy_from_slice(&(i as f32).to_le_bytes());
            v[4..8].copy_from_slice(&1.0f32.to_le_bytes());
            v[8..12].copy_from_slice(&2.0f32.to_le_bytes());
            // normal (arbitrary, distinct from position so the test can tell
            // the parser read the right offset)
            v[NORMAL_OFFSET..NORMAL_OFFSET + 4].copy_from_slice(&0.0f32.to_le_bytes());
            v[NORMAL_OFFSET + 4..NORMAL_OFFSET + 8].copy_from_slice(&1.0f32.to_le_bytes());
            v[NORMAL_OFFSET + 8..NORMAL_OFFSET + 12].copy_from_slice(&0.0f32.to_le_bytes());
            // uv
            v[UV_OFFSET..UV_OFFSET + 4].copy_from_slice(&0.25f32.to_le_bytes());
            v[UV_OFFSET + 4..UV_OFFSET + 8].copy_from_slice(&0.75f32.to_le_bytes());
            d.extend_from_slice(&v);
        }
        let idx: [u16; 3] = [0, 1, 2];
        d.extend_from_slice(&(6u32).to_le_bytes());
        for i in idx {
            d.extend_from_slice(&i.to_le_bytes());
        }
        d.extend_from_slice(&[0u8; 7]); // trailing slack, like real files
        d
    }

    #[test]
    fn parses_positions_uvs_indices_and_material() {
        let m = parse(&synth(3)).expect("should parse");
        assert_eq!(m.material, "materials/models/t/Material.json");
        assert_eq!(m.indices, vec![0, 1, 2]);
        assert_eq!(m.positions.len(), 3);
        // z is read (the whole point — puppet parsing drops it).
        assert_eq!(m.positions[2], [2.0, 1.0, 2.0]);
        assert_eq!(m.normals[0], [0.0, 1.0, 0.0]);
        assert_eq!(m.uvs[0], [0.25, 0.75]);
    }

    #[test]
    fn rejects_puppet_meshes() {
        let mut d = synth(3);
        d.extend_from_slice(b"MDLS");
        assert!(parse(&d).is_none(), "a skeleton means puppet.rs owns it");
    }

    #[test]
    fn rejects_non_mdl() {
        assert!(parse(b"not a model at all").is_none());
    }
}
