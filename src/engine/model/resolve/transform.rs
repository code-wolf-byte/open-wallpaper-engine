//! Object transforms: each object's local origin, angles and scale, and the
//! parent chain that turns them into world transforms.

use super::*;

/// An affine transform in WE scene space (Y-up), kept as decomposed
/// translation / Euler-rotation (radians, un-negated) / scale.
#[derive(Clone, Copy)]
pub struct Xform {
    pub t: [f64; 3],
    pub r: [f64; 3],
    pub s: [f64; 3],
}

impl Xform {
    pub const IDENTITY: Xform = Xform {
        t: [0.0; 3],
        r: [0.0; 3],
        s: [1.0, 1.0, 1.0],
    };

    pub fn is_identity(&self) -> bool {
        self.t == [0.0; 3] && self.r == [0.0; 3] && self.s == [1.0, 1.0, 1.0]
    }

    /// Map a point living in a child frame into the frame this Xform is
    /// expressed in: `p' = t + R · (s ⊙ p)`.
    pub fn apply_point(&self, p: [f64; 3]) -> [f64; 3] {
        let scaled = [p[0] * self.s[0], p[1] * self.s[1], p[2] * self.s[2]];
        let rot = rotate_euler_f64(scaled, self.r);
        [rot[0] + self.t[0], rot[1] + self.t[1], rot[2] + self.t[2]]
    }

    /// `world = self (parent world) ∘ local`.
    pub fn compose(&self, local: &Xform) -> Xform {
        Xform {
            t: self.apply_point(local.t),
            r: [
                self.r[0] + local.r[0],
                self.r[1] + local.r[1],
                self.r[2] + local.r[2],
            ],
            s: [
                self.s[0] * local.s[0],
                self.s[1] * local.s[1],
                self.s[2] * local.s[2],
            ],
        }
    }
}

/// f64 mirror of `camera3d::rotate_euler` (Rx then Ry then Rz) — scene coords
/// reach a few thousand pixels, so we keep the composition in f64.
pub(super) fn rotate_euler_f64(v: [f64; 3], a: [f64; 3]) -> [f64; 3] {
    let (sx, cx) = a[0].sin_cos();
    let (sy, cy) = a[1].sin_cos();
    let (sz, cz) = a[2].sin_cos();
    let v = [v[0], cx * v[1] - sx * v[2], sx * v[1] + cx * v[2]];
    let v = [cy * v[0] + sy * v[2], v[1], -sy * v[0] + cy * v[2]];
    [cz * v[0] - sz * v[1], sz * v[0] + cz * v[1], v[2]]
}

/// This object's own local transform, read from the raw scene fields (the same
/// sources `layer_from_object` uses, minus alignment which is a per-layer draw
/// concern).
pub fn obj_local_xform(obj: &SceneObject) -> Xform {
    let t = obj.parsed_origin();
    let r = obj
        .angles
        .as_ref()
        .map(crate::engine::model::json_to_animated)
        .and_then(|v| v.as_vec3())
        .map(|a| [a[0] as f64, a[1] as f64, a[2] as f64])
        .unwrap_or([0.0; 3]);
    let s = obj
        .scale
        .as_ref()
        .and_then(crate::engine::model::scene::parse_value_vec3)
        .unwrap_or([1.0, 1.0, 1.0]);
    Xform { t, r, s }
}

/// Resolve an object's `parent` reference to an index in `scene.objects`.
/// WE stores the parent's `id` (a number), occasionally a name string.
pub(super) fn parent_index(
    obj: &SceneObject,
    id_to_idx: &HashMap<i64, usize>,
    name_to_idx: &HashMap<&str, usize>,
) -> Option<usize> {
    // `disablepropagation` severs the link: the object keeps its parent for
    // grouping/visibility but stops inheriting its transform. Every occurrence
    // in the 197 installed scenes is `false`, so this changes nothing today —
    // it's here so authored content that does set it behaves correctly.
    if obj
        .disablepropagation
        .as_ref()
        .and_then(crate::engine::model::scene::parse_value_bool)
        .unwrap_or(false)
    {
        return None;
    }
    let p = obj.parent.as_ref()?;
    if let Some(id) = p.as_i64() {
        return id_to_idx.get(&id).copied();
    }
    if let Some(id) = p.as_u64() {
        return id_to_idx.get(&(id as i64)).copied();
    }
    if let Some(name) = p.as_str() {
        return name_to_idx.get(name).copied();
    }
    None
}

/// World transform of object `idx`, including its own local transform, composed
/// up the parent chain. Memoized; guards against cycles / dangling parents.
pub(super) fn world_xform(
    idx: usize,
    objects: &[SceneObject],
    id_to_idx: &HashMap<i64, usize>,
    name_to_idx: &HashMap<&str, usize>,
    memo: &mut [Option<Xform>],
    depth: u32,
) -> Xform {
    if let Some(x) = memo[idx] {
        return x;
    }
    let local = obj_local_xform(&objects[idx]);
    // depth guard: a malformed scene can loop or nest absurdly deep; fall back
    // to the local transform rather than blow the stack (tolerant-parsing).
    let world = match parent_index(&objects[idx], id_to_idx, name_to_idx) {
        Some(p) if p != idx && depth < 64 => {
            world_xform(p, objects, id_to_idx, name_to_idx, memo, depth + 1).compose(&local)
        }
        _ => local,
    };
    memo[idx] = Some(world);
    world
}

/// Per-object transform data the live path needs to recompose the parent chain
/// every frame.
///
/// `apply_parent_transforms` bakes the chain once at load, which is right for a
/// static parent but wrong for a scripted one: a parent node's transform can
/// change every frame, and those nodes are usually image-less (no `Layer` at
/// all), so their scripts never run from the layer list. This carries them.
pub struct TransformGraph {
    /// Resolved parent index per object (`None` = root).
    pub parent: Vec<Option<usize>>,
    /// Authored local transform per object — the base a script updates from.
    pub local: Vec<Xform>,
    /// Object name/id, indexed the same as every other field here — lets the
    /// live path find an object's sounds and layers by identity.
    pub name: Vec<Option<String>>,
    pub id: Vec<Option<i64>>,
}

impl TransformGraph {
    /// World transform per object, composing `locals` up each parent chain.
    /// `disabled[i]` (this frame's live `thisLayer.disablepropagation()`
    /// call — see the object-attachment API) makes object `i` stop
    /// inheriting *its own* parent's transform, exactly like the static
    /// scene.json `disablepropagation: true` field already does via
    /// `parent_index` returning `None` — see
    /// `disablepropagation_severs_transform_inheritance` below. This is
    /// that same rule evaluated live per-frame instead of once at load, for
    /// an object whose `parent` link is real (baked into `self.parent`) but
    /// wants to sever it dynamically.
    pub fn world(&self, locals: &[Xform], disabled: &[bool]) -> Vec<Xform> {
        let mut memo: Vec<Option<Xform>> = vec![None; self.parent.len()];
        (0..self.parent.len())
            .map(|i| self.world_of(i, locals, disabled, &mut memo, 0))
            .collect()
    }

    pub(super) fn world_of(
        &self,
        idx: usize,
        locals: &[Xform],
        disabled: &[bool],
        memo: &mut Vec<Option<Xform>>,
        depth: u32,
    ) -> Xform {
        if let Some(x) = memo[idx] {
            return x;
        }
        let severed = disabled.get(idx).copied().unwrap_or(false);
        // Same 64-deep guard as the load-time walker: a malformed scene must
        // not blow the stack.
        let w = match self.parent[idx] {
            Some(p) if p != idx && depth < 64 && !severed => self
                .world_of(p, locals, disabled, memo, depth + 1)
                .compose(&locals[idx]),
            _ => locals[idx],
        };
        memo[idx] = Some(w);
        w
    }
}

pub(super) fn build_transform_graph(scene: &Scene) -> TransformGraph {
    let objects = &scene.objects;
    let id_to_idx: HashMap<i64, usize> = objects
        .iter()
        .enumerate()
        .filter_map(|(i, o)| o.id.map(|id| (id, i)))
        .collect();
    let name_to_idx: HashMap<&str, usize> = objects
        .iter()
        .enumerate()
        .filter_map(|(i, o)| o.name.as_deref().map(|n| (n, i)))
        .collect();
    TransformGraph {
        parent: objects
            .iter()
            .map(|o| parent_index(o, &id_to_idx, &name_to_idx))
            .collect(),
        local: objects.iter().map(obj_local_xform).collect(),
        name: objects.iter().map(|o| o.name.clone()).collect(),
        id: objects.iter().map(|o| o.id).collect(),
    }
}

/// Parent world transform for object `idx` (identity if it has no parent). This
/// is what gets applied to a layer, whose OWN local transform is already baked
/// in by `layer_from_object`.
pub(super) fn parent_world_xform(
    idx: usize,
    objects: &[SceneObject],
    id_to_idx: &HashMap<i64, usize>,
    name_to_idx: &HashMap<&str, usize>,
    memo: &mut [Option<Xform>],
) -> Xform {
    match parent_index(&objects[idx], id_to_idx, name_to_idx) {
        Some(p) if p != idx => world_xform(p, objects, id_to_idx, name_to_idx, memo, 0),
        _ => Xform::IDENTITY,
    }
}

/// Compose each layer's baked-in local transform with its parent chain so
/// parented objects (clocks, dates, ripples, particle emitters) render at their
/// true world position/orientation/scale instead of their local offset.
pub(super) fn apply_parent_transforms(
    scene: &Scene,
    layers: &mut [Layer],
    particle_layers: &mut [ParticleLayer],
    mesh3d_layers: &mut [Mesh3dLayer],
) {
    let objects = &scene.objects;
    if objects.is_empty() {
        return;
    }
    let id_to_idx: HashMap<i64, usize> = objects
        .iter()
        .enumerate()
        .filter_map(|(i, o)| o.id.map(|id| (id, i)))
        .collect();
    let name_to_idx: HashMap<&str, usize> = objects
        .iter()
        .enumerate()
        .filter_map(|(i, o)| o.name.as_deref().map(|n| (n, i)))
        .collect();
    let mut memo: Vec<Option<Xform>> = vec![None; objects.len()];

    for layer in layers.iter_mut() {
        let idx = layer.order_index;
        if idx >= objects.len() {
            continue;
        }
        let pw = parent_world_xform(idx, objects, &id_to_idx, &name_to_idx, &mut memo);
        if pw.is_identity() {
            continue;
        }
        layer.origin = pw.apply_point(layer.origin);
        layer.scale = [
            layer.scale[0] * pw.s[0],
            layer.scale[1] * pw.s[1],
            layer.scale[2] * pw.s[2],
        ];
        // `Layer::angle` is the negated z-roll (screen convention); the raw 3D
        // `angles` are un-negated.
        layer.angle -= pw.r[2] as f32;
        layer.angles = [
            layer.angles[0] + pw.r[0] as f32,
            layer.angles[1] + pw.r[1] as f32,
            layer.angles[2] + pw.r[2] as f32,
        ];
    }

    for pl in particle_layers.iter_mut() {
        let idx = pl.order_index;
        if idx >= objects.len() {
            continue;
        }
        let pw = parent_world_xform(idx, objects, &id_to_idx, &name_to_idx, &mut memo);
        if pw.is_identity() {
            continue;
        }
        pl.origin = pw.apply_point(pl.origin);
    }

    // Meshes take the same treatment as image layers — they're ordinary scene
    // objects that happen to carry geometry. Without this a parented skybox
    // stays at the origin while the camera moves out of it.
    for m in mesh3d_layers.iter_mut() {
        let idx = m.order_index;
        if idx >= objects.len() {
            continue;
        }
        let pw = parent_world_xform(idx, objects, &id_to_idx, &name_to_idx, &mut memo);
        if pw.is_identity() {
            continue;
        }
        let o = pw.apply_point([m.origin[0] as f64, m.origin[1] as f64, m.origin[2] as f64]);
        m.origin = [o[0] as f32, o[1] as f32, o[2] as f32];
        for i in 0..3 {
            m.scale[i] *= pw.s[i] as f32;
            m.angles[i] += pw.r[i] as f32;
        }
    }
}
