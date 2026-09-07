use anyhow::{Context, Result};
use image::RgbaImage;
use std::collections::HashMap;
use std::path::Path;

use super::particle;
use super::pkg::Package;
use super::properties::SceneProperties;
use super::scene::{Scene, SceneObject};
use super::tex::TexFile;

/// Parse scene.json, first resolving `{"user": ...}` references against the
/// wallpaper's project.json properties (plus any `--set-property` overrides).
fn parse_scene_with_properties(scene_json: &str, project_dir: Option<&Path>) -> Result<Scene> {
    let mut value: serde_json::Value = serde_json::from_str(scene_json).with_context(|| {
        let preview = if scene_json.len() > 200 {
            &scene_json[..200]
        } else {
            scene_json
        };
        format!("failed to parse scene.json: {preview}...")
    })?;
    if let Some(dir) = project_dir {
        let props = SceneProperties::from_project_dir(dir);
        props.resolve_scene_json(&mut value);
    }
    Scene::from_value(value)
}

/// A fully resolved scene ready to composite into a single frame.
pub struct ResolvedScene {
    pub width: u32,
    pub height: u32,
    pub layers: Vec<Layer>,
    pub particle_layers: Vec<ParticleLayer>,
    /// Static 3D mesh objects (`model` → `.mdl`) — only genuine perspective
    /// scenes have these; the 2D compositor ignores them.
    pub mesh3d_layers: Vec<Mesh3dLayer>,
    /// Parent chain + transform scripts per object, for the live path's
    /// per-frame recompose. See [`TransformGraph`].
    pub transform_graph: TransformGraph,
    pub scene: Scene,
}

/// A scene object whose `model` field references a static 3D mesh. Rendered as
/// real geometry through the perspective camera (see `engine::mesh3d`), not as
/// a flat quad — spheres, skyboxes and cylinders can't be billboarded.
pub struct Mesh3dLayer {
    pub name: String,
    pub mesh: crate::engine::mesh3d::Mesh3d,
    /// Resolved from the material path embedded in the `.mdl` header.
    pub texture: RgbaImage,
    /// World-space transform (WE scene units; the mesh's own space is ≈ unit).
    pub origin: [f32; 3],
    pub scale: [f32; 3],
    pub angles: [f32; 3],
    /// The same three before `apply_parent_transforms` composed the parent
    /// chain in. A script-driven parent moves per frame, so the live path
    /// recomposes from these rather than trying to invert the baked world value.
    pub local_origin: [f32; 3],
    pub local_scale: [f32; 3],
    pub local_angles: [f32; 3],
    /// The material's `"cullmode": "nocull"` — draw both faces. Everything
    /// else culls back faces, which is what lets a skybox's near hemisphere
    /// drop out instead of hiding the whole scene inside it.
    pub nocull: bool,
    /// `depthtest: "disabled"` on the object (or its material pass) — draw
    /// without depth comparison. Real content always leaves it enabled; this
    /// exists so content that disables it isn't silently depth-culled.
    pub depthtest: bool,
    pub order_index: usize,
}

/// A scene object whose `particle` field references a particle preset JSON
/// (e.g. `"particles/presets/fog1.json"`). Holds enough static data for any
/// renderer to build its own `particle::ParticleSystem` (each render path
/// owns independent simulation state, so this is config, not a live system).
pub struct ParticleLayer {
    pub name: String,
    /// WE origin: absolute scene coordinates, Y-up, (0,0) at bottom-left —
    /// same convention as `Layer::origin`; callers convert to pixel space at
    /// render time (`px = origin.x`, `py = height - origin.y`).
    pub origin: [f64; 3],
    /// `origin` before the parent chain was composed in (see `Layer`).
    pub local_origin: [f64; 3],
    pub parallax_depth: [f64; 2],
    pub config: particle::ParticleConfig,
    pub overrides: Option<particle::InstanceOverride>,
    /// Resolved once from `config.material` (if present) via the same
    /// model/material→texture chain image layers use, keeping every
    /// sprite-sheet frame. `None` falls back to `render_onto`'s flat-color
    /// circle draw.
    pub sprite_texture: Option<particle::ParticleSprite>,
    /// True when the material's own pass declares `"blending":"additive"`
    /// (the overwhelming majority of real particle materials — fog, smoke,
    /// embers, rain, lightning). Composited additively onto the scene
    /// instead of the default alpha-over, which otherwise makes a sprite's
    /// near-black background visibly darken/box the scene behind it.
    pub additive_blend: bool,
    /// This object's raw index in `scene.objects` (already the scene's
    /// topological/declaration render order) — lets render loops interleave
    /// particle systems with image layers in true scene z-order, and lets
    /// effect instances (which record the same raw object index) find their
    /// owning layer regardless of how many particles/skipped/invisible
    /// objects precede it.
    pub order_index: usize,
    /// Child presets from `config.children`, resolved (JSON + material
    /// sprite) here where asset access lives; attached to the built system
    /// via `ParticleSystem::add_child` at construction time.
    pub children: Vec<ResolvedChildParticle>,
}

/// One resolved `children` entry of a particle preset (see
/// [`particle::ChildRef`]).
pub struct ResolvedChildParticle {
    pub config: particle::ParticleConfig,
    pub sprite: Option<particle::ParticleSprite>,
    pub additive: bool,
    pub child_ref: particle::ChildRef,
}

/// Everything needed to re-rasterize a script-driven text layer at runtime:
/// the script and font. The layer's projected `rect` (position + size) is
/// computed once at build through the full camera/parent transform; on a
/// content change we keep that rect's center and only rescale its extent by
/// the ratio of the new bitmap size to the old, so no projection is re-derived.
#[derive(Clone)]
pub struct TextDynamic {
    pub script: String,
    /// The scene's authored `text.scriptproperties` object (per-instance
    /// editor values, merged over the script's builder defaults).
    pub script_properties: Option<serde_json::Value>,
    pub font_data: Vec<u8>,
    pub point_size: f32,
    /// Text currently rasterized into the layer image (the `value` WE feeds
    /// back into `update` each tick).
    pub last_text: String,
}

pub struct Layer {
    pub name: String,
    pub image: RgbaImage,
    /// Additional animation frames (frame 0 is `image`). Empty = not animated.
    pub extra_frames: Vec<RgbaImage>,
    /// Duration of each frame in milliseconds. 0 = not animated / unknown.
    pub frame_duration_ms: u32,
    pub origin: [f64; 3],
    pub size: [f64; 3],
    pub scale: [f64; 3], // WE scale multiplier (default 1.0), separate from size
    /// `origin`/`scale`/`angles` before `apply_parent_transforms` composed the
    /// parent chain in — the live path recomposes from these each frame when a
    /// parent is script-driven. See `ResolvedScene::transform_graph`.
    pub local_origin: [f64; 3],
    pub local_scale: [f64; 3],
    pub local_angles: [f32; 3],
    pub parallax_depth: [f64; 2],
    /// z-rotation in radians (WE `angles.z`; the JSON value is already in radians).
    pub angle: f32,
    /// Full 3D rotation in radians, un-negated as stored in scene.json.
    /// Only used by the perspective (3D scene) path; 2D uses `angle`.
    pub angles: [f32; 3],
    pub blend_mode: u32,
    pub alpha: f32,
    /// Inline SceneScript source driving `alpha` per frame, if the property
    /// was authored as `{"value": …, "script": "…"}`. `None` = static alpha.
    pub alpha_script: Option<String>,
    pub color: [f32; 3],
    pub brightness: f32,
    /// True for WE "copybackground" layers — source is the rendered canvas so far.
    pub copybackground: bool,
    /// `.tex` `NoInterpolation` flag: sample nearest instead of linear.
    pub no_interpolation: bool,
    /// `.tex` `ClampUVs`/`ClampUVsBorder` flag: clamp-to-edge instead of repeat.
    pub clamp_uvs: bool,
    /// Animated puppet runtime (mesh + skeleton + MDLA animations) — the
    /// live GPU path re-poses and re-rasterizes `image` from this.
    pub puppet: Option<std::sync::Arc<crate::engine::puppet::PuppetRuntime>>,
    /// Embedded-video stream (mp4 inside the .tex) — the live GPU path
    /// re-uploads decoded frames from this; `image` is the first frame.
    pub video: Option<std::sync::Arc<std::sync::Mutex<VideoLayerStream>>>,
    /// Script-driven text: the GPU path re-evaluates and re-rasterizes
    /// `image` when the script's output changes (clocks, dates, countdowns).
    pub text_dynamic: Option<TextDynamic>,
    /// This object's raw index in `scene.objects` — see
    /// `ParticleLayer::order_index`.
    pub order_index: usize,
    /// The scene object's own `id`. Effects reference another object's
    /// rendered output as `_rt_imageLayerComposite_<id>_a`, so the live path
    /// needs to know which layer each id belongs to.
    pub object_id: Option<i64>,
    /// Inline SceneScripts driving `visible`/`scale`/`origin`/`angles`, if any.
    /// Evaluated per frame by the live GPU path (see `GpuSceneInstance::render`).
    pub transform_scripts: TransformScripts,
    /// The object's AUTHORED `visible` value — the starting visibility and the
    /// `value` fed to a `visible` script. Critical: scripts that only define
    /// cursor handlers (invisible click hitboxes, `"value": false`) have no
    /// `update()`, so the script returns `value` unchanged — feeding `true`
    /// here would render those hitboxes as opaque boxes.
    pub visible_base: bool,
}

/// The per-frame SceneScripts a layer's transform can carry. Each `update`
/// receives the property's base value (a `Vec3` for scale/origin/angles, a
/// bool for visible) and returns the value for this frame.
#[derive(Clone, Default)]
pub struct TransformScripts {
    pub visible: Option<String>,
    pub scale: Option<String>,
    pub origin: Option<String>,
    pub angles: Option<String>,
}

/// A loaded texture plus any additional animation frames (e.g. from a `.tex`
/// TEXS spritesheet table or a multi-image container).
struct LoadedImage {
    image: RgbaImage,
    extra_frames: Vec<RgbaImage>,
    frame_duration_ms: u32,
    no_interpolation: bool,
    clamp_uvs: bool,
    /// Present when the model is an animated puppet (mesh + skeleton +
    /// MDLA animations): `image` holds the rest pose, and the live GPU
    /// path re-poses/re-rasterizes from this over time.
    puppet: Option<std::sync::Arc<crate::engine::puppet::PuppetRuntime>>,
    /// Present for embedded-video textures: `image` holds the first frame,
    /// and the live GPU path re-uploads frames from this stream over time.
    video: Option<std::sync::Arc<std::sync::Mutex<VideoLayerStream>>>,
    /// Set when the model's material declares the `SPRITESHEET` combo and the
    /// `.tex` carries a frame table (e.g. a 2-cell AM/PM atlas): the per-frame
    /// sub-rect images. `layer_from_object` picks the single frame the object's
    /// frame script selects and uses it as the layer image. Without this the
    /// flattened atlas stretches into the one-cell quad (all cells overlaid).
    sprite_frames: Vec<RgbaImage>,
}

impl LoadedImage {
    fn single(image: RgbaImage) -> Self {
        Self {
            image,
            extra_frames: Vec::new(),
            frame_duration_ms: 0,
            no_interpolation: false,
            clamp_uvs: false,
            puppet: None,
            video: None,
            sprite_frames: Vec::new(),
        }
    }

    /// Build from a parsed `.tex`, pulling out extra animation frames if any
    /// and carrying over its `NoInterpolation`/`ClampUVs` sampler flags.
    fn from_tex(tex: &TexFile) -> Result<Self> {
        let no_interpolation = tex.no_interpolation();
        let clamp_uvs = tex.clamp_uvs();
        if tex.is_animated() {
            if let Ok(mut frames) = tex.to_rgba_frames() {
                if frames.len() > 1 {
                    let image = frames.remove(0);
                    let frame_duration_ms = tex
                        .frames()
                        .first()
                        .map(|f| ((f.frametime * 1000.0) as u32).max(1))
                        .unwrap_or(100);
                    return Ok(Self {
                        image,
                        extra_frames: frames,
                        frame_duration_ms,
                        no_interpolation,
                        clamp_uvs,
                        puppet: None,
                        video: None,
                        sprite_frames: Vec::new(),
                    });
                }
            }
        }
        if let Some(bytes) = tex.video_bytes() {
            return video_tex_to_loaded(tex, bytes);
        }
        Ok(Self {
            no_interpolation,
            clamp_uvs,
            ..Self::single(tex.to_rgba()?)
        })
    }
}

impl ResolvedScene {
    /// Load a scene wallpaper from a directory.
    ///
    /// Automatically detects whether assets are loose files or packed in a
    /// `.pkg` archive and loads accordingly. The scene file is usually
    /// `scene.json`/`scene.pkg`, but project.json's `file` field can name a
    /// different one — GIF-converted wallpapers ship
    /// `gifscene.json`/`gifscene.pkg` (e.g. workshop item 2036522973).
    #[tracing::instrument(target = "scene", level = "debug", fields(dir = %dir.display()))]
    pub fn from_directory(dir: &Path) -> Result<Self> {
        tracing::debug!(target: "scene", "resolving scene from directory");
        let scene_name = std::fs::read_to_string(dir.join("project.json"))
            .ok()
            .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
            .and_then(|p| p.get("file")?.as_str().map(str::to_string))
            .filter(|f| f.ends_with(".json"))
            .unwrap_or_else(|| "scene.json".to_string());
        let pkg_name = format!("{}.pkg", scene_name.trim_end_matches(".json"));

        let scene_json_path = dir.join(&scene_name);
        let scene_pkg_path = dir.join(&pkg_name);

        if scene_pkg_path.exists() {
            let pkg = Package::from_file(&scene_pkg_path)?;

            let scene_json = if scene_json_path.exists() {
                std::fs::read_to_string(&scene_json_path)
                    .with_context(|| format!("reading {}", scene_json_path.display()))?
            } else if let Some(data) = pkg.get(&scene_name).or_else(|| pkg.get("scene.json")) {
                String::from_utf8(data.to_vec())
                    .context("scene json inside PKG is not valid UTF-8")?
            } else {
                anyhow::bail!("no {scene_name} found in directory or PKG archive");
            };

            return Self::from_package_with_dir(&pkg, &scene_json, dir);
        }

        let scene_json = std::fs::read_to_string(&scene_json_path)
            .with_context(|| format!("reading {}", scene_json_path.display()))?;
        let scene = parse_scene_with_properties(&scene_json, Some(dir))?;

        let mut layers = Vec::new();
        let mut solid_indices = Vec::new();
        let mut particle_layers = Vec::new();
        let mut mesh3d_layers = Vec::new();
        // Effective visibility (own AND every ancestor) — see Scene::visibility_mask.
        let visible = scene.visibility_mask();
        // Raw `scene.objects` index (not a visible-only count): effect
        // instances record the same raw index, so layers and effects agree
        // on object identity no matter what gets skipped in between.
        for (obj_index, obj) in scene.objects.iter().enumerate() {
            if !visible[obj_index] {
                continue;
            }
            if obj.light.is_some() && obj.image.is_none() {
                if let Some(mut layer) = light_layer_from_object(obj) {
                    layer.order_index = obj_index;
                    layers.push(layer);
                }
                continue;
            }
            if obj.mesh3d_path().is_some() {
                if let Some(mut ml) = mesh3d_layer_from_object(obj, Some(dir), None) {
                    ml.order_index = obj_index;
                    mesh3d_layers.push(ml);
                }
                continue;
            }
            if obj.particle.is_some() {
                if let Some(mut pl) = particle_layer_from_object(obj, Some(dir), None) {
                    pl.order_index = obj_index;
                    particle_layers.push(pl);
                }
                continue;
            }
            if obj.text.is_some() {
                if let Some(mut layer) = text_layer_from_object(obj, Some(dir), None) {
                    layer.order_index = obj_index;
                    layers.push(layer);
                }
                continue;
            }
            // Note: `copybackground` objects still carry a real image/model —
            // the reference engine ignores the flag and draws them normally.
            let loaded = match obj.image.as_deref() {
                None => {
                    if obj.effects.is_empty() {
                        continue;
                    }
                    LoadedImage::single(placeholder_for(obj))
                }
                Some(path) if is_solid_layer_path(path) => {
                    // Scene-size expansion is for effect chains only — see
                    // fill_solid_layer_sizes.
                    if !obj.effects.is_empty() {
                        solid_indices.push(layers.len());
                    }
                    LoadedImage::single(white_pixel())
                }
                Some(path) => {
                    if is_special_layer(path) {
                        continue;
                    }
                    match load_texture_from_dir(dir, path) {
                        Ok(loaded) => loaded,
                        Err(e) => {
                            tracing::warn!(target: "scene", "texture load failed for '{path}': {e:#}");
                            if obj.effects.is_empty() {
                                continue;
                            }
                            LoadedImage::single(placeholder_for(obj))
                        }
                    }
                }
            };
            let mut layer = layer_from_object(obj, loaded, None);
            layer.order_index = obj_index;
            layers.push(layer);
        }

        let (width, height) = guess_scene_dimensions(&scene, &layers);
        fill_solid_layer_sizes(&mut layers, &solid_indices, width, height);
        apply_parent_transforms(
            &scene,
            &mut layers,
            &mut particle_layers,
            &mut mesh3d_layers,
        );
        Ok(Self {
            width,
            height,
            layers,
            particle_layers,
            mesh3d_layers,
            transform_graph: build_transform_graph(&scene),
            scene,
        })
    }

    /// Load from a PKG archive, falling back to the directory for loose files.
    fn from_package_with_dir(pkg: &Package, scene_json: &str, dir: &Path) -> Result<Self> {
        let scene = parse_scene_with_properties(scene_json, Some(dir))?;

        let mut layers = Vec::new();
        let mut solid_indices = Vec::new();
        let mut particle_layers = Vec::new();
        let mut mesh3d_layers = Vec::new();
        // Effective visibility (own AND every ancestor) — see Scene::visibility_mask.
        let visible = scene.visibility_mask();
        // Raw `scene.objects` index (not a visible-only count): effect
        // instances record the same raw index, so layers and effects agree
        // on object identity no matter what gets skipped in between.
        for (obj_index, obj) in scene.objects.iter().enumerate() {
            if !visible[obj_index] {
                continue;
            }
            if obj.light.is_some() && obj.image.is_none() {
                if let Some(mut layer) = light_layer_from_object(obj) {
                    layer.order_index = obj_index;
                    layers.push(layer);
                }
                continue;
            }
            if obj.mesh3d_path().is_some() {
                if let Some(mut ml) = mesh3d_layer_from_object(obj, Some(dir), Some(pkg)) {
                    ml.order_index = obj_index;
                    mesh3d_layers.push(ml);
                }
                continue;
            }
            if obj.particle.is_some() {
                if let Some(mut pl) = particle_layer_from_object(obj, Some(dir), Some(pkg)) {
                    pl.order_index = obj_index;
                    particle_layers.push(pl);
                }
                continue;
            }
            if obj.text.is_some() {
                if let Some(mut layer) = text_layer_from_object(obj, Some(dir), Some(pkg)) {
                    layer.order_index = obj_index;
                    layers.push(layer);
                }
                continue;
            }
            let loaded = match obj.image.as_deref() {
                None => {
                    if obj.effects.is_empty() {
                        continue;
                    }
                    LoadedImage::single(placeholder_for(obj))
                }
                Some(path) if is_solid_layer_path(path) => {
                    // Scene-size expansion is for effect chains only — see
                    // fill_solid_layer_sizes.
                    if !obj.effects.is_empty() {
                        solid_indices.push(layers.len());
                    }
                    LoadedImage::single(white_pixel())
                }
                Some(path) => {
                    if is_special_layer(path) {
                        continue;
                    }
                    match load_texture_from_pkg(pkg, path).or_else(|pkg_err| {
                        load_texture_from_dir(dir, path)
                            .map_err(|dir_err| anyhow::anyhow!("pkg: {pkg_err}; dir: {dir_err}"))
                    }) {
                        Ok(loaded) => loaded,
                        Err(e) => {
                            tracing::warn!(target: "scene", "texture load failed for '{path}': {e:#}");
                            if obj.effects.is_empty() {
                                continue;
                            }
                            LoadedImage::single(placeholder_for(obj))
                        }
                    }
                }
            };
            let mut layer = layer_from_object(obj, loaded, None);
            layer.order_index = obj_index;
            layers.push(layer);
        }

        let (width, height) = guess_scene_dimensions(&scene, &layers);
        fill_solid_layer_sizes(&mut layers, &solid_indices, width, height);
        apply_parent_transforms(
            &scene,
            &mut layers,
            &mut particle_layers,
            &mut mesh3d_layers,
        );
        Ok(Self {
            width,
            height,
            layers,
            particle_layers,
            mesh3d_layers,
            transform_graph: build_transform_graph(&scene),
            scene,
        })
    }

    /// Load a scene wallpaper from a PKG archive + scene.json string.
    /// No project.json is available here, so user properties keep defaults.
    pub fn from_package(pkg: &Package, scene_json: &str) -> Result<Self> {
        let scene = parse_scene_with_properties(scene_json, None)?;

        let mut layers = Vec::new();
        let mut solid_indices = Vec::new();
        let mut particle_layers = Vec::new();
        let mut mesh3d_layers = Vec::new();
        // Effective visibility (own AND every ancestor) — see Scene::visibility_mask.
        let visible = scene.visibility_mask();
        // Raw `scene.objects` index (not a visible-only count): effect
        // instances record the same raw index, so layers and effects agree
        // on object identity no matter what gets skipped in between.
        for (obj_index, obj) in scene.objects.iter().enumerate() {
            if !visible[obj_index] {
                continue;
            }
            if obj.light.is_some() && obj.image.is_none() {
                if let Some(mut layer) = light_layer_from_object(obj) {
                    layer.order_index = obj_index;
                    layers.push(layer);
                }
                continue;
            }
            if obj.mesh3d_path().is_some() {
                if let Some(mut ml) = mesh3d_layer_from_object(obj, None, Some(pkg)) {
                    ml.order_index = obj_index;
                    mesh3d_layers.push(ml);
                }
                continue;
            }
            if obj.particle.is_some() {
                if let Some(mut pl) = particle_layer_from_object(obj, None, Some(pkg)) {
                    pl.order_index = obj_index;
                    particle_layers.push(pl);
                }
                continue;
            }
            if obj.text.is_some() {
                if let Some(mut layer) = text_layer_from_object(obj, None, Some(pkg)) {
                    layer.order_index = obj_index;
                    layers.push(layer);
                }
                continue;
            }
            let loaded = match obj.image.as_deref() {
                None => {
                    if obj.effects.is_empty() {
                        continue;
                    }
                    LoadedImage::single(placeholder_for(obj))
                }
                Some(path) if is_solid_layer_path(path) => {
                    // Scene-size expansion is for effect chains only — see
                    // fill_solid_layer_sizes.
                    if !obj.effects.is_empty() {
                        solid_indices.push(layers.len());
                    }
                    LoadedImage::single(white_pixel())
                }
                Some(path) => {
                    if is_special_layer(path) {
                        continue;
                    }
                    match load_texture_from_pkg(pkg, path) {
                        Ok(loaded) => loaded,
                        Err(_) => {
                            if obj.effects.is_empty() {
                                continue;
                            }
                            LoadedImage::single(placeholder_for(obj))
                        }
                    }
                }
            };
            let mut layer = layer_from_object(obj, loaded, None);
            layer.order_index = obj_index;
            layers.push(layer);
        }

        let (width, height) = guess_scene_dimensions(&scene, &layers);
        fill_solid_layer_sizes(&mut layers, &solid_indices, width, height);
        apply_parent_transforms(
            &scene,
            &mut layers,
            &mut particle_layers,
            &mut mesh3d_layers,
        );
        Ok(Self {
            width,
            height,
            layers,
            particle_layers,
            mesh3d_layers,
            transform_graph: build_transform_graph(&scene),
            scene,
        })
    }

    /// Composite all layers into a single RGBA image, in true scene z-order
    /// (image and particle layers interleaved by `order_index`, matching the
    /// reference's single shared per-object render order — CScene.cpp's
    /// `m_objectsByRenderOrder` — instead of drawing all particles after all
    /// images).
    pub fn render(&self) -> RgbaImage {
        let mut canvas = RgbaImage::new(self.width, self.height);

        enum DrawItem {
            Image(usize),
            Particle(usize),
        }
        let mut items: Vec<(usize, DrawItem)> = self
            .layers
            .iter()
            .enumerate()
            .map(|(i, l)| (l.order_index, DrawItem::Image(i)))
            .chain(
                self.particle_layers
                    .iter()
                    .enumerate()
                    .map(|(i, pl)| (pl.order_index, DrawItem::Particle(i))),
            )
            .collect();
        items.sort_by_key(|(order, _)| *order);

        for (_, item) in items {
            match item {
                DrawItem::Image(i) => self.draw_image_layer(&self.layers[i], &mut canvas),
                // This is a single-shot preview render, so seed the
                // simulation forward a few seconds instead of drawing an
                // empty, freshly-spawned system.
                DrawItem::Particle(i) => {
                    let pl = &self.particle_layers[i];
                    let spawn_center = [
                        pl.origin[0] as f32,
                        self.height as f32 - pl.origin[1] as f32,
                    ];
                    let mut system = particle::ParticleSystem::from_config(
                        &pl.config,
                        spawn_center,
                        pl.overrides.as_ref(),
                    );
                    if let Some(sprite) = &pl.sprite_texture {
                        system.set_sprite_frames(sprite.frames.len(), sprite.duration);
                    }
                    for child in &pl.children {
                        system.add_child(
                            child.config.clone(),
                            child.sprite.clone(),
                            child.additive,
                            &child.child_ref,
                            spawn_center,
                        );
                    }
                    for _ in 0..150 {
                        system.step(1.0 / 30.0);
                    }
                    // The canvas here *is* the opaque scene, so additive
                    // accumulation directly implements the reference's
                    // GL_SRC_ALPHA/GL_ONE quad blending against it.
                    system.render_onto_blended(
                        &mut canvas,
                        pl.sprite_texture.as_ref(),
                        [0.0, 0.0],
                        pl.additive_blend,
                    );
                }
            }
        }

        canvas
    }

    fn draw_image_layer(&self, layer: &Layer, canvas: &mut RgbaImage) {
        let draw_w = if layer.size[0] != 0.0 {
            (layer.size[0] * layer.scale[0]) as u32
        } else {
            layer.image.width()
        };
        let draw_h = if layer.size[1] != 0.0 {
            (layer.size[1] * layer.scale[1]) as u32
        } else {
            layer.image.height()
        };
        if draw_w == 0 || draw_h == 0 {
            return;
        }

        let resized = if layer.image.width() != draw_w || layer.image.height() != draw_h {
            image::imageops::resize(
                &layer.image,
                draw_w,
                draw_h,
                image::imageops::FilterType::Lanczos3,
            )
        } else {
            layer.image.clone()
        };

        // WE origin is the object's center in absolute scene coordinates
        // (Y-up, (0,0) at bottom-left); convert to top-left pixel coords.
        let px = (layer.origin[0] - draw_w as f64 / 2.0) as i64;
        let py = (self.height as f64 - layer.origin[1] - draw_h as f64 / 2.0) as i64;

        let cw = canvas.width() as i64;
        let ch = canvas.height() as i64;

        let mut blend_pixel = |cx: i64, cy: i64, s: image::Rgba<u8>| {
            if cx < 0 || cx >= cw || cy < 0 || cy >= ch {
                return;
            }
            let src_rgb = [
                (s[0] as f32 / 255.0) * layer.color[0] * layer.brightness,
                (s[1] as f32 / 255.0) * layer.color[1] * layer.brightness,
                (s[2] as f32 / 255.0) * layer.color[2] * layer.brightness,
            ];
            let src_a = (s[3] as f32 / 255.0) * layer.alpha;
            if src_a <= 0.0 {
                return;
            }
            let dst = canvas.get_pixel_mut(cx as u32, cy as u32);
            let dest_rgb = [
                dst[0] as f32 / 255.0,
                dst[1] as f32 / 255.0,
                dst[2] as f32 / 255.0,
            ];
            let out =
                crate::engine::blend::apply_blending(layer.blend_mode, dest_rgb, src_rgb, src_a);
            dst[0] = (out[0].clamp(0.0, 1.0) * 255.0) as u8;
            dst[1] = (out[1].clamp(0.0, 1.0) * 255.0) as u8;
            dst[2] = (out[2].clamp(0.0, 1.0) * 255.0) as u8;
            let dst_a = dst[3] as f32 / 255.0;
            dst[3] = ((dst_a + src_a * (1.0 - dst_a)).clamp(0.0, 1.0) * 255.0) as u8;
        };

        if layer.angle == 0.0 {
            for y in 0..draw_h {
                for x in 0..draw_w {
                    blend_pixel(px + x as i64, py + y as i64, *resized.get_pixel(x, y));
                }
            }
        } else {
            // Rotated layer: walk the canvas-space bounding box of the
            // rotated rect and inverse-sample back into the unrotated
            // `resized` image (nearest-neighbor).
            let ccx = px as f64 + draw_w as f64 / 2.0;
            let ccy = py as f64 + draw_h as f64 / 2.0;
            let half_diag = ((draw_w as f64 / 2.0).powi(2) + (draw_h as f64 / 2.0).powi(2)).sqrt();
            let (sin_a, cos_a) = (layer.angle as f64).sin_cos();
            let x0 = ((ccx - half_diag).floor() as i64).max(0);
            let x1 = ((ccx + half_diag).ceil() as i64).min(cw);
            let y0 = ((ccy - half_diag).floor() as i64).max(0);
            let y1 = ((ccy + half_diag).ceil() as i64).min(ch);
            for cy in y0..y1 {
                for cx in x0..x1 {
                    let fx = cx as f64 + 0.5 - ccx;
                    let fy = cy as f64 + 0.5 - ccy;
                    // Inverse rotation (screen → unrotated local space).
                    let lx = cos_a * fx + sin_a * fy;
                    let ly = -sin_a * fx + cos_a * fy;
                    let sx = (lx + draw_w as f64 / 2.0).floor();
                    let sy = (ly + draw_h as f64 / 2.0).floor();
                    if sx < 0.0 || sy < 0.0 || sx >= draw_w as f64 || sy >= draw_h as f64 {
                        continue;
                    }
                    blend_pixel(cx, cy, *resized.get_pixel(sx as u32, sy as u32));
                }
            }
        }
    }
}

/// WE `parallaxDepth` is always a vec2 (independent x/y depth), but a bare
/// number in scene.json means "same depth on both axes".
fn parse_parallax_depth(v: &serde_json::Value) -> [f64; 2] {
    use crate::engine::model::DynamicValue;
    match crate::engine::model::json_to_animated(v).value {
        DynamicValue::Float(f) => [f as f64; 2],
        DynamicValue::Vec3([x, y, _]) => [x as f64, y as f64],
        _ => [0.0, 0.0],
    }
}

/// WE origin convention: Y-up, (0,0) at bottom-left (same as `Layer::origin`).
/// "top"/"bottom" shift the object vertically by half its scaled height so
/// the origin becomes that edge instead of the center; "left"/"right" do the
/// same horizontally. Matches CImage.cpp lines 242-256 exactly (each pair is
/// mutually exclusive — "top bottom" together, like the reference, only
/// honors "top").
pub(crate) fn alignment_offset(alignment: Option<&str>, scaled_size: [f64; 2]) -> [f64; 2] {
    let a = alignment.unwrap_or("").to_lowercase();
    let mut offset = [0.0, 0.0];
    if a.contains("top") {
        offset[1] -= scaled_size[1] / 2.0;
    } else if a.contains("bottom") {
        offset[1] += scaled_size[1] / 2.0;
    }
    if a.contains("left") {
        offset[0] += scaled_size[0] / 2.0;
    } else if a.contains("right") {
        offset[0] -= scaled_size[0] / 2.0;
    }
    offset
}

/// `alignment_override`, when given, replaces `obj.alignment` (text objects
/// combine separate `horizontalalign`/`verticalalign` fields into one string
/// before calling this, rather than reading `obj.alignment` directly).
/// Brightness multiplier for additive (colorBlendMode 9) image layers — see the
/// call site. Default 0.55; override live with `WP_ENGINE_ADDITIVE_BRIGHTNESS`.
fn additive_brightness_scale() -> f32 {
    std::env::var("WP_ENGINE_ADDITIVE_BRIGHTNESS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.55)
}

/// For a `SPRITESHEET` layer, pick the single frame the object's frame script
/// selects and make it the layer image. Frames are the `.tex` TEXS sub-rects; the
/// index comes from the object's frame-select script (WE hangs
/// `getTextureAnimation().setFrame` off the `visible` script — e.g. AM/PM by
/// `getHours()`). Falls back to frame 0 if the script sets none.
fn select_spritesheet_frame(obj: &SceneObject, loaded: &mut LoadedImage) {
    if loaded.sprite_frames.len() < 2 {
        return;
    }
    let frame = obj
        .visible
        .as_ref()
        .and_then(|v| v.get("script"))
        .and_then(|s| s.as_str())
        .and_then(|script| crate::engine::script::ScriptContext::new().eval_frame(script))
        .unwrap_or(0)
        .min(loaded.sprite_frames.len() as u32 - 1);
    loaded.image = loaded.sprite_frames[frame as usize].clone();
}

fn layer_from_object(
    obj: &SceneObject,
    mut loaded: LoadedImage,
    alignment_override: Option<&str>,
) -> Layer {
    // Spritesheet: replace the flattened atlas with the single cell the object's
    // frame script selects (e.g. AM/PM by `getHours()`), so the one-cell quad
    // shows one glyph instead of every cell squished together.
    // ponytail: frame is picked once at load from the current wall clock; a live
    // wallpaper won't flip AM↔PM mid-session without a reload (rare — re-select in
    // the render loop if that ceiling ever matters).
    select_spritesheet_frame(obj, &mut loaded);
    // Attach puppet `animationlayers` (blended each frame) to the fresh runtime.
    if let (Some(rt), Some(entries)) = (
        loaded.puppet.as_mut(),
        obj.animationlayers.as_ref().and_then(|v| v.as_array()),
    ) {
        if let Some(rt) = std::sync::Arc::get_mut(rt) {
            let n = rt.model.animations.len();
            if n > 0 {
                let layers: Vec<crate::engine::puppet::AnimLayer> = entries
                    .iter()
                    .enumerate()
                    .filter(|(_, e)| e.get("visible").and_then(|v| v.as_bool()).unwrap_or(true))
                    .map(|(i, e)| {
                        // `animation` is the MDLA record's own id (2321732083's
                        // samurai layers name 60/63/80/105). Match it; fall back
                        // to index-then-declaration-order for files whose ids we
                        // failed to read.
                        let raw = e
                            .get("animation")
                            .and_then(|v| v.as_u64())
                            .unwrap_or(i as u64);
                        let anim_idx = rt
                            .model
                            .animations
                            .iter()
                            .position(|a| a.id as u64 == raw)
                            .unwrap_or_else(|| {
                                if (raw as usize) < n {
                                    raw as usize
                                } else {
                                    i.min(n - 1)
                                }
                            });
                        crate::engine::puppet::AnimLayer {
                            anim_idx,
                            additive: e.get("additive").and_then(|v| v.as_bool()).unwrap_or(false),
                            blend: e.get("blend").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32,
                            rate: e.get("rate").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32,
                        }
                    })
                    .collect();
                rt.layers = layers;
            }
        }
    }
    let parallax = obj
        .parallax_depth
        .as_ref()
        .map(parse_parallax_depth)
        .unwrap_or([0.0, 0.0]);

    // WE stores angles already in radians; only the z component (2D roll)
    // applies to a flat layer quad. The reference negates it (rotate(-angle,
    // ...)) when building the object's screen transform.
    let angles_animated = obj
        .angles
        .as_ref()
        .map(crate::engine::model::json_to_animated);
    let angles3 = angles_animated
        .as_ref()
        .and_then(|v| v.as_vec3())
        .unwrap_or([0.0, 0.0, 0.0]);
    let angle = -angles3[2];

    let scale_animated = obj
        .scale
        .as_ref()
        .map(crate::engine::model::json_to_animated);
    let scale = scale_animated
        .as_ref()
        .and_then(|v| v.as_vec3())
        .map(|s| [s[0] as f64, s[1] as f64, s[2] as f64])
        .unwrap_or([1.0, 1.0, 1.0]);
    let origin_animated = obj
        .origin
        .as_ref()
        .map(crate::engine::model::json_to_animated);

    // Per-frame transform scripts (evaluated by the live GPU path). origin/
    // scale/angles carry their script on the animated value; visible carries
    // it on the raw `visible` object.
    let transform_scripts = TransformScripts {
        visible: obj.visible_script(),
        scale: scale_animated.as_ref().and_then(|v| v.script.clone()),
        origin: origin_animated.as_ref().and_then(|v| v.script.clone()),
        angles: angles_animated.as_ref().and_then(|v| v.script.clone()),
    };

    let alpha_animated = obj
        .alpha
        .as_ref()
        .map(crate::engine::model::json_to_animated);
    let alpha = alpha_animated
        .as_ref()
        .and_then(|v| v.as_float())
        .unwrap_or(1.0);
    // Inline SceneScript source driving `alpha`, if any — evaluated per frame
    // by the render loop's ScriptContext (see GpuSceneInstance::render).
    let alpha_script = alpha_animated.as_ref().and_then(|v| v.script.clone());
    let color = obj
        .color
        .as_ref()
        .map(crate::engine::model::json_to_animated)
        .and_then(|v| v.as_vec3())
        .unwrap_or([1.0, 1.0, 1.0]);
    let brightness = obj
        .brightness
        .as_ref()
        .map(crate::engine::model::json_to_animated)
        .and_then(|v| v.as_float())
        .unwrap_or(1.0);
    // ponytail: additive (colorBlendMode 9) image planes with a near-white base
    // (e.g. LonelyCat's water-ripple layer) blow out to a flat white disc,
    // because our water effects refract the layer's *own* bright base instead of
    // the real backdrop (backdrop capture is unimplemented — see PROGRESS.md).
    // Until that lands, scale additive layers down so the ripple structure stays
    // visible. Ceiling: this dims *every* additive image layer, not just ripples;
    // remove once screen-space backdrop capture exists. Tunable live via
    // WP_ENGINE_ADDITIVE_BRIGHTNESS (default 0.55).
    let brightness = if obj.color_blend_mode == 9 {
        brightness * additive_brightness_scale()
    } else {
        brightness
    };

    // Alignment shifts the object so its origin becomes an edge rather than
    // its center (CImage.cpp lines 242-256), expressed here as an offset
    // folded directly into `origin` — every consumer of `Layer::origin`
    // (both render paths) then gets alignment for free.
    // Text objects (the only caller passing `alignment_override`) size their
    // quad from the rasterized glyph bitmap, not the scene `size` field —
    // WE's CText uses `m_quadSize` (the raster dims) and ignores `size`
    // (which is editor bounding-box metadata). Honoring it would stretch the
    // glyphs to an unrelated box.
    let is_text = alignment_override.is_some();
    let raw_size = obj.parsed_size();
    let effective_size = if !is_text && raw_size[0] > 0.0 && raw_size[1] > 0.0 {
        [raw_size[0], raw_size[1]]
    } else {
        [loaded.image.width() as f64, loaded.image.height() as f64]
    };
    let scaled_size = [effective_size[0] * scale[0], effective_size[1] * scale[1]];
    let alignment = alignment_override.or(obj.alignment.as_deref());
    let align_offset = alignment_offset(alignment, scaled_size);
    let origin = {
        let o = obj.parsed_origin();
        [o[0] + align_offset[0], o[1] + align_offset[1], o[2]]
    };

    Layer {
        name: obj.name.clone().unwrap_or_default(),
        image: loaded.image,
        extra_frames: loaded.extra_frames,
        frame_duration_ms: loaded.frame_duration_ms,
        origin,
        // Text: leave size 0 so both render paths fall back to the raster dims.
        size: if is_text {
            [0.0, 0.0, 0.0]
        } else {
            obj.parsed_size()
        },
        scale,
        local_origin: origin,
        local_scale: scale,
        local_angles: angles3,
        parallax_depth: parallax,
        angle,
        angles: angles3,
        blend_mode: obj.color_blend_mode,
        alpha,
        alpha_script,
        color,
        brightness,
        copybackground: obj.copybackground,
        no_interpolation: loaded.no_interpolation,
        // Object-level `clampuvs` overrides the .tex ClampUVs flag when set.
        clamp_uvs: loaded.clamp_uvs
            || obj
                .clampuvs
                .as_ref()
                .and_then(|v| {
                    v.as_bool()
                        .or_else(|| v.get("value").and_then(|i| i.as_bool()))
                })
                .unwrap_or(false),
        puppet: loaded.puppet,
        video: loaded.video,
        text_dynamic: None,
        order_index: 0,
        object_id: obj.id,
        transform_scripts,
        visible_base: obj.is_visible(),
    }
}

fn is_video_path(p: &str) -> bool {
    let p = p.to_lowercase();
    p.ends_with(".mp4") || p.ends_with(".webm") || p.ends_with(".mkv") || p.ends_with(".avi")
}

/// Resolve a particle preset's JSON text, checking the wallpaper's own
/// directory, then its `scene.pkg` (if given), then the global Steam assets
/// dir — mirroring the same wallpaper-first priority used for shaders/effects
/// (`shaders::resolver::AssetResolver`).
/// Read a scene asset from wherever it lives: loose files, the pkg archive,
/// then the shared WE assets dir.
fn read_asset_bytes(dir: Option<&Path>, pkg: Option<&Package>, rel_path: &str) -> Option<Vec<u8>> {
    if let Some(dir) = dir {
        if let Ok(data) = std::fs::read(dir.join(rel_path)) {
            return Some(data);
        }
    }
    if let Some(pkg) = pkg {
        if let Some(data) = pkg.get(rel_path) {
            return Some(data.to_vec());
        }
    }
    read_from_global_assets(rel_path)
}

fn read_particle_json(dir: Option<&Path>, pkg: Option<&Package>, rel_path: &str) -> Option<String> {
    String::from_utf8(read_asset_bytes(dir, pkg, rel_path)?).ok()
}

/// Build a `ParticleLayer` from a scene object whose `particle` field is a
/// preset path, resolving and parsing the referenced JSON. Returns `None`
/// (silently skipped, like a missing texture) if the preset can't be found
/// or parsed.
fn particle_layer_from_object(
    obj: &SceneObject,
    dir: Option<&Path>,
    pkg: Option<&Package>,
) -> Option<ParticleLayer> {
    let particle_ref = match &obj.particle {
        Some(serde_json::Value::String(s)) => s.as_str(),
        _ => return None,
    };
    let json = read_particle_json(dir, pkg, particle_ref)?;
    let mut config: particle::ParticleConfig = serde_json::from_str(&json)
        .inspect_err(
            |e| tracing::warn!(target: "particle", "failed to parse '{particle_ref}': {e}"),
        )
        .ok()?;
    let overrides: Option<particle::InstanceOverride> = obj
        .instanceoverride
        .as_ref()
        .and_then(|v| serde_json::from_value(v.clone()).ok());

    // `instanceoverride.enabled: false` turns the whole system off
    // (ObjectParser defaults it to true).
    if overrides.as_ref().is_some_and(|o| {
        o.enabled
            .as_ref()
            .is_some_and(|v| v.as_bool() == Some(false) || v.as_u64() == Some(0))
    }) {
        return None;
    }

    // Control-point plumbing the simulation can't do itself (it only knows
    // spawn-relative screen offsets):
    // 1. `instanceoverride.controlpointN` repositions preset control points
    //    with absolute scene coordinates (e.g. the discharge preset's arc
    //    endpoint, "1586.7 993.2") — absolute values are world-space, so
    //    force flag 2 on.
    // 2. Any world-space control point (flags & 2) is then converted to a
    //    spawn-relative offset here, where the object's origin is known:
    //    offset = (x_we - origin_x, origin_y - y_we) in the simulation's
    //    y-down screen space.
    if let Some(over) = overrides.as_ref() {
        for (key, value) in &over.extra {
            let Some(n) = key
                .strip_prefix("controlpoint")
                .and_then(|s| s.parse::<usize>().ok())
            else {
                continue;
            };
            while config.controlpoint.len() <= n {
                config
                    .controlpoint
                    .push(particle::ControlPointConfig::default());
            }
            config.controlpoint[n].offset = Some(value.clone());
            config.controlpoint[n].flags = Some(config.controlpoint[n].flags.unwrap_or(0) | 2);
        }
    }
    let origin = obj.parsed_origin();
    for cp in &mut config.controlpoint {
        if cp.flags.unwrap_or(0) & 2 == 0 {
            continue;
        }
        if let Some(world) = cp.offset.as_ref().and_then(particle::value_as_vec3_pub) {
            let rel = [
                world[0] - origin[0] as f32,
                origin[1] as f32 - world[1],
                world[2],
            ];
            cp.offset = Some(serde_json::json!(format!(
                "{} {} {}",
                rel[0], rel[1], rel[2]
            )));
            cp.flags = Some(cp.flags.unwrap_or(0) & !2);
        }
    }
    let parallax_depth = obj
        .parallax_depth
        .as_ref()
        .map(parse_parallax_depth)
        .unwrap_or([0.0, 0.0]);

    let resolved_sprite = config.material.as_deref().and_then(|mat_path| {
        // A failure here silently degrades to the flat-color circle draw —
        // visually plausible for dust/snow but wrong for shaped sprites, so
        // it must at least be visible in the log.
        if let Some(pkg) = pkg {
            resolve_particle_sprite_pkg(pkg, mat_path)
                .inspect_err(|e| {
                    eprintln!("[particle] sprite '{mat_path}' failed (untextured fallback): {e:#}")
                })
                .ok()
        } else {
            dir.and_then(|dir| {
                resolve_particle_sprite_dir(dir, mat_path)
                    .inspect_err(|e| {
                        eprintln!(
                            "[particle] sprite '{mat_path}' failed (untextured fallback): {e:#}"
                        )
                    })
                    .ok()
            })
        }
    });
    let additive_blend = resolved_sprite
        .as_ref()
        .is_some_and(|(_, blending)| blending.as_deref() == Some("additive"));
    let sprite_texture = resolved_sprite.map(|(sprite, _)| sprite);

    // Child presets: same preset-JSON + material resolution as the parent.
    // Failures degrade to "child skipped" (logged), never sink the layer.
    let children: Vec<ResolvedChildParticle> = config
        .children
        .clone()
        .unwrap_or_default()
        .into_iter()
        .filter_map(|child_ref| {
            let json = read_particle_json(dir, pkg, &child_ref.name).or_else(|| {
                eprintln!("[particle] child preset '{}' not found", child_ref.name);
                None
            })?;
            let child_cfg: particle::ParticleConfig = serde_json::from_str(&json)
                .inspect_err(|e| {
                    eprintln!("[particle] child '{}' parse failed: {e}", child_ref.name)
                })
                .ok()?;
            let resolved = child_cfg.material.as_deref().and_then(|mat_path| {
                if let Some(pkg) = pkg {
                    resolve_particle_sprite_pkg(pkg, mat_path).ok()
                } else {
                    dir.and_then(|dir| resolve_particle_sprite_dir(dir, mat_path).ok())
                }
            });
            let additive = resolved
                .as_ref()
                .is_some_and(|(_, blending)| blending.as_deref() == Some("additive"));
            let sprite = resolved.map(|(sprite, _)| sprite);
            Some(ResolvedChildParticle {
                config: child_cfg,
                sprite,
                additive,
                child_ref,
            })
        })
        .collect();

    Some(ParticleLayer {
        name: obj.name.clone().unwrap_or_default(),
        origin: obj.parsed_origin(),
        local_origin: obj.parsed_origin(),
        parallax_depth,
        config,
        overrides,
        sprite_texture,
        additive_blend,
        order_index: 0,
        children,
    })
}

/// `text` is either a plain JSON string or a `{"user": ..., "value": "..."}`
/// wrapper (scene.json's usual user-setting shape) — deliberately NOT routed
/// through `json_to_animated`, which treats 2-3 whitespace-separated numbers
/// as a vector; that heuristic is wrong for text content that happens to be
/// purely numeric (e.g. a countdown reading "12 34").
fn extract_text_string(v: &serde_json::Value) -> Option<String> {
    match v {
        serde_json::Value::String(s) => Some(s.clone()),
        serde_json::Value::Object(map) => map.get("value").and_then(extract_text_string),
        _ => None,
    }
}

/// Build a text object's rendered bitmap as a normal `Layer` (see
/// `engine::text` module docs for why: no scripting engine means text never
/// changes after this one rasterization, so it can reuse 100% of the
/// existing image-layer pipeline in both render paths for free).
fn text_layer_from_object(
    obj: &SceneObject,
    dir: Option<&Path>,
    pkg: Option<&Package>,
) -> Option<Layer> {
    // A text object's `text` is either a plain string, a {"value": …}
    // wrapper, or a scripted {"script": "…", "value"?: …} — clocks/dates
    // (the dominant case in real content) often ship the script with NO
    // static value at all, so the script's first evaluation IS the content.
    let text_value = obj.text.as_ref()?;
    let script = text_value
        .get("script")
        .and_then(|v| v.as_str())
        .map(str::to_string);
    let script_properties = text_value.get("scriptproperties").cloned();
    let static_str = extract_text_string(text_value).unwrap_or_default();
    let text_str = match (&script, static_str.is_empty()) {
        (Some(script), _) => {
            // One-shot load-time evaluation so the layer rasterizes real
            // content (and sizes itself correctly) from frame 0.
            let mut ctx = crate::engine::script::ScriptContext::new();
            ctx.eval_update_string(script, &static_str, script_properties.as_ref())
                .filter(|t| !t.is_empty())
                .unwrap_or(static_str)
        }
        (None, false) => static_str,
        (None, true) => return None,
    };
    if text_str.is_empty() {
        return None;
    }
    let point_size = obj
        .pointsize
        .as_ref()
        .and_then(|v| {
            v.as_f64()
                .or_else(|| v.get("value").and_then(|v| v.as_f64()))
        })
        .unwrap_or(32.0) as f32;
    let scale = obj
        .scale
        .as_ref()
        .and_then(|v| crate::engine::model::json_to_animated(v).as_vec3())
        .unwrap_or([1.0, 1.0, 1.0]);
    // The on-screen text height is driven by the object's `size` BOX height,
    // not pointsize: `layer_from_object` scales the rasterized bitmap by the
    // object's `scale`, and WE authors size text via that box (a big box +
    // scale>1 = big text; a modest box + scale<1 = small text). Rasterize at
    // the box height so the bitmap ≈ its on-screen size (crisp, natural aspect,
    // no stretch). `pointsize` here is only the fallback resolution when the
    // object ships no box (then compensate for `scale` like CText does, so the
    // glyphs don't collapse to a couple of pixels).
    let box_h = obj.parsed_size()[1] as f32;
    let raster_px = if box_h > 0.0 {
        box_h
    } else {
        let avg_scale = (scale[0] + scale[1]) * 0.5;
        let compensate = if avg_scale > 0.0 && avg_scale < 1.0 {
            (1.0 / avg_scale).min(32.0)
        } else {
            1.0
        };
        point_size * compensate
    }
    .clamp(1.0, 1024.0);
    let font_data = super::text::resolve_font_data(obj.font.as_deref(), dir, pkg)?;
    // Word-wrap to `maxwidth` (scene px == bitmap px at our raster resolution)
    // and cap at `maxrows`, if authored.
    let gate = |v: &Option<serde_json::Value>| {
        v.as_ref()
            .and_then(crate::engine::scene::parse_value_bool)
            .unwrap_or(false)
    };
    let text_str = match obj
        .maxwidth
        .as_ref()
        .and_then(crate::engine::scene::parse_value_f32)
        .filter(|w| *w > 0.0)
        .filter(|_| gate(&obj.limitwidth))
    {
        Some(mw) => {
            let rows = obj
                .maxrows
                .as_ref()
                .and_then(crate::engine::scene::parse_value_f32)
                .filter(|_| gate(&obj.limitrows))
                .unwrap_or(0.0)
                .max(0.0) as usize;
            super::text::wrap_text(
                &font_data,
                &text_str,
                raster_px,
                mw,
                rows,
                gate(&obj.limituseellipsis),
            )
        }
        None => text_str,
    };
    let image = super::text::rasterize(&font_data, &text_str, raster_px)?;

    // Opaque background box: bake the text color + bg color + padding into the
    // bitmap and neutralize the layer tint (color set to white below). `pad`
    // is authored in scene px, which maps 1:1 to bitmap px (we raster at the
    // box height). Off by default — `opaquebackground` is usually false.
    let opaque_bg = obj
        .opaquebackground
        .as_ref()
        .and_then(|v| {
            v.as_bool()
                .or_else(|| v.get("value").and_then(|i| i.as_bool()))
        })
        .unwrap_or(false);
    let (image, bg_baked) = if opaque_bg {
        let text_color = obj
            .color
            .as_ref()
            .and_then(|v| crate::engine::scene::parse_value_vec3(v))
            .map(|c| [c[0] as f32, c[1] as f32, c[2] as f32])
            .unwrap_or([1.0, 1.0, 1.0]);
        let bright = obj
            .backgroundbrightness
            .as_ref()
            .and_then(crate::engine::scene::parse_value_f32)
            .unwrap_or(1.0);
        let bg = obj
            .backgroundcolor
            .as_ref()
            .and_then(|v| crate::engine::scene::parse_value_vec3(v))
            .map(|c| {
                [
                    c[0] as f32 * bright,
                    c[1] as f32 * bright,
                    c[2] as f32 * bright,
                ]
            })
            .unwrap_or([0.0, 0.0, 0.0]);
        let pad = obj
            .padding
            .as_ref()
            .and_then(crate::engine::scene::parse_value_f32)
            .unwrap_or(0.0)
            .max(0.0) as u32;
        (
            super::text::with_background(&image, text_color, bg, pad),
            true,
        )
    } else {
        (image, false)
    };

    let halign = obj
        .horizontalalign
        .as_deref()
        .or(obj.alignment.as_deref())
        .unwrap_or("center");
    let valign = obj.verticalalign.as_deref().unwrap_or("center");
    let combined = format!("{halign} {valign}");

    let mut layer = layer_from_object(obj, LoadedImage::single(image), Some(&combined));
    // WE fits the glyphs into the authored `size` box preserving their
    // aspect (letterbox), not stretch-to-fill: a tall clock box (375×161)
    // over ~99×26 glyphs would render the block ~6× too tall and collide with
    // the layer below it. Shrink the box to the bitmap aspect — the
    // constrained dimension keeps the box, the other pulls in.
    // ponytail: refit only; origin's alignment offset was folded for the
    // original box, so this is exact for center-aligned text (all clocks/dates
    // here) and would need an offset re-fold for left/top anchors.
    let bw = layer.image.width() as f64;
    let bh = layer.image.height() as f64;
    if bw > 0.0 && bh > 0.0 && layer.size[0] > 0.0 && layer.size[1] > 0.0 {
        let aspect = bw / bh;
        let (box_w, box_h) = (layer.size[0], layer.size[1]);
        let fitted = if box_w / box_h > aspect {
            [box_h * aspect, box_h]
        } else {
            [box_w, box_w / aspect]
        };
        layer.size[0] = fitted[0];
        layer.size[1] = fitted[1];
    }
    if bg_baked {
        // Text + bg colors are baked into the bitmap; don't tint it again.
        layer.color = [1.0, 1.0, 1.0];
    }
    if let Some(script) = script {
        layer.text_dynamic = Some(TextDynamic {
            script,
            script_properties,
            font_data,
            point_size: raster_px,
            last_text: text_str,
        });
    }
    Some(layer)
}

/// Build a static-3D-mesh layer for an object whose `model` names a `.mdl`.
/// `read` fetches an asset by relative path (dir- or pkg-backed), and
/// `resolve_tex` turns the mesh's embedded material path into its texture.
fn mesh3d_layer_from_object(
    obj: &SceneObject,
    dir: Option<&Path>,
    pkg: Option<&Package>,
) -> Option<Mesh3dLayer> {
    let path = obj.mesh3d_path()?;
    // The pkg stores every path lowercased, while scene.json and the `.mdl`
    // header keep the author's casing (`models/LP/LP.mdl`, and real content
    // has non-ASCII names too) — so retry lowercased on any miss.
    let bytes = read_asset_bytes(dir, pkg, path)
        .or_else(|| read_asset_bytes(dir, pkg, &path.to_lowercase()))?;
    let mesh = crate::engine::mesh3d::parse(&bytes)?;
    let resolve_tex = |p: &str| {
        dir.and_then(|d| load_texture_from_dir(d, p).ok())
            .or_else(|| pkg.and_then(|p2| load_texture_from_pkg(p2, p).ok()))
            .map(|l| l.image)
    };
    let pass = material_first_pass(dir, pkg, &mesh.material);
    let texture = resolve_tex(&mesh.material)
        .or_else(|| resolve_tex(&mesh.material.to_lowercase()))
        // Untextured materials are normal here (`"textures": [null,null,null]`
        // with a flat `color` constant): draw them as that color. An
        // unresolvable material falls through to white rather than vanishing.
        .unwrap_or_else(|| {
            let [r, g, b] = pass
                .as_ref()
                .and_then(pass_constant_color)
                .unwrap_or([255; 3]);
            RgbaImage::from_pixel(1, 1, image::Rgba([r, g, b, 255]))
        });
    let nocull = pass
        .as_ref()
        .and_then(|p| p.get("cullmode")?.as_str())
        .is_some_and(|m| m == "nocull");
    // The object's own `depthtest` wins over the material pass's.
    let depthtest = obj
        .depthtest
        .as_ref()
        .and_then(|v| v.as_str())
        .or_else(|| pass.as_ref().and_then(|p| p.get("depthtest")?.as_str()))
        .map(|m| m != "disabled")
        .unwrap_or(true);

    let scale = obj
        .scale
        .as_ref()
        .and_then(|v| crate::engine::model::json_to_animated(v).as_vec3())
        .unwrap_or([1.0, 1.0, 1.0]);
    let angles = obj
        .angles
        .as_ref()
        .and_then(|v| crate::engine::model::json_to_animated(v).as_vec3())
        .unwrap_or([0.0, 0.0, 0.0]);
    let o = obj.parsed_origin();

    tracing::debug!(
        target: "scene",
        "mesh3d '{path}': {} verts, {} tris, material '{}'",
        mesh.positions.len(),
        mesh.indices.len() / 3,
        mesh.material
    );
    Some(Mesh3dLayer {
        name: obj.name.clone().unwrap_or_default(),
        mesh,
        texture,
        origin: [o[0] as f32, o[1] as f32, o[2] as f32],
        scale,
        angles,
        local_origin: [o[0] as f32, o[1] as f32, o[2] as f32],
        local_scale: scale,
        local_angles: angles,
        nocull,
        depthtest,
        order_index: 0,
    })
}

/// A WE light object rendered as an additive radial glow — a minimal stand-in
/// for the full lighting model. Colored by `color`, sized by `radius`, with a
/// smooth quadratic falloff scaled by `intensity`. Additively blended so it
/// brightens whatever is behind it (like a real light).
fn light_layer_from_object(obj: &SceneObject) -> Option<Layer> {
    let (radius, intensity, color) = obj.light_params()?;

    let d = (radius * 2.0).min(2048.0) as u32;
    let mut img = RgbaImage::new(d, d);
    let c = d as f32 / 2.0;
    for y in 0..d {
        for x in 0..d {
            let dist = (((x as f32 - c).powi(2) + (y as f32 - c).powi(2)).sqrt() / c).min(1.0);
            let a = (1.0 - dist).powi(2) * intensity; // quadratic falloff
            img.put_pixel(
                x,
                y,
                image::Rgba([
                    (color[0] * 255.0) as u8,
                    (color[1] * 255.0) as u8,
                    (color[2] * 255.0) as u8,
                    (a * 255.0) as u8,
                ]),
            );
        }
    }

    let mut layer = layer_from_object(obj, LoadedImage::single(img), None);
    // colorBlendMode 9 = Add (linear dodge) — the glow adds light.
    layer.blend_mode = 9;
    layer.color = [1.0, 1.0, 1.0];
    Some(layer)
}

/// A live decode stream for an embedded-video texture layer: a detached
/// ffmpeg thread loops the video (PTS-paced, same `video_decode_loop` the
/// full-video wallpaper path uses) and feeds frames through a small bounded
/// channel. The consumer drains to the newest frame each render tick, so
/// slow consumers drop frames instead of lagging. The thread exits on its
/// own once this struct (the receiver) is dropped, and the temp file
/// backing the looping decoder is removed then too.
pub struct VideoLayerStream {
    rx: std::sync::mpsc::Receiver<std::sync::Arc<RgbaImage>>,
    /// Temp file the decoder re-opens on every loop — must outlive playback.
    path: std::path::PathBuf,
}

impl VideoLayerStream {
    /// Newest decoded frame since the last call, if any arrived.
    pub fn latest_frame(&self) -> Option<std::sync::Arc<RgbaImage>> {
        let mut latest = None;
        while let Ok(frame) = self.rx.try_recv() {
            latest = Some(frame);
        }
        latest
    }
}

impl Drop for VideoLayerStream {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

/// Start looping playback for an embedded video payload: writes the bytes
/// to a temp file, decodes the first frame synchronously (so the layer has
/// correct content immediately), and spawns the paced decode loop for the
/// rest.
fn start_video_stream(bytes: &[u8]) -> Result<(RgbaImage, VideoLayerStream)> {
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let tmp = std::env::temp_dir().join(format!(
        "we_embvid_{}_{}.mp4",
        std::process::id(),
        SEQ.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::write(&tmp, bytes)
        .with_context(|| format!("writing embedded video temp file: {}", tmp.display()))?;
    let first = match crate::render::ffmpeg::decode_first_frame(&tmp) {
        Ok(img) => img,
        Err(e) => {
            let _ = std::fs::remove_file(&tmp);
            return Err(e);
        }
    };
    let (tx, rx) = std::sync::mpsc::sync_channel::<std::sync::Arc<RgbaImage>>(2);
    let decode_path = tmp.clone();
    std::thread::spawn(move || {
        // Exits when the receiver (VideoLayerStream) is dropped.
        if let Err(e) = crate::render::ffmpeg::video_decode_loop(&decode_path, &tx) {
            eprintln!(
                "embedded video decoder error for '{}': {e}",
                decode_path.display()
            );
        }
    });
    Ok((first, VideoLayerStream { rx, path: tmp }))
}

/// Build a `LoadedImage` for an embedded-video texture: first frame as the
/// image, plus the live stream for the GPU path to animate.
fn video_tex_to_loaded(tex: &TexFile, bytes: &[u8]) -> Result<LoadedImage> {
    let (first, stream) = start_video_stream(bytes)?;
    let mut loaded = LoadedImage::single(first);
    loaded.no_interpolation = tex.no_interpolation();
    loaded.clamp_uvs = tex.clamp_uvs();
    loaded.video = Some(std::sync::Arc::new(std::sync::Mutex::new(stream)));
    Ok(loaded)
}

fn load_texture_from_dir(dir: &Path, image_path: &str) -> Result<LoadedImage> {
    // If it's a .json reference, resolve the model/material chain
    if image_path.ends_with(".json") {
        return resolve_model_chain_dir(dir, image_path);
    }

    let full_path = dir.join(image_path);

    // Video file — extract first frame as static thumbnail
    if is_video_path(image_path) && full_path.exists() {
        return crate::render::ffmpeg::decode_first_frame(&full_path).map(LoadedImage::single);
    }

    // Try as .tex first
    let tex_path = full_path.with_extension("tex");
    if tex_path.exists() {
        let data =
            std::fs::read(&tex_path).with_context(|| format!("reading {}", tex_path.display()))?;
        let tex =
            TexFile::parse(&data).with_context(|| format!("parsing {}", tex_path.display()))?;
        return LoadedImage::from_tex(&tex);
    }

    if full_path.exists() {
        return image::open(&full_path)
            .map(|i| LoadedImage::single(i.into_rgba8()))
            .with_context(|| format!("loading {}", full_path.display()));
    }

    let tex_path_str = format!("{}.tex", full_path.display());
    let tex_appended = Path::new(&tex_path_str);
    if tex_appended.exists() {
        let data = std::fs::read(tex_appended)
            .with_context(|| format!("reading {}", tex_appended.display()))?;
        let tex = TexFile::parse(&data)?;
        return LoadedImage::from_tex(&tex);
    }

    anyhow::bail!("texture not found: {image_path}")
}

fn load_texture_from_pkg(pkg: &Package, image_path: &str) -> Result<LoadedImage> {
    // If the image path points to a .json model/material, resolve the chain.
    if image_path.ends_with(".json") {
        return resolve_model_chain_pkg(pkg, image_path);
    }

    // Try as .tex
    let tex_name = if image_path.ends_with(".tex") {
        image_path.to_string()
    } else {
        format!("{image_path}.tex")
    };

    if let Some(data) = pkg.get(&tex_name) {
        let tex =
            TexFile::parse(data).with_context(|| format!("parsing .tex from pkg: {tex_name}"))?;
        return LoadedImage::from_tex(&tex);
    }

    // Try original path (png/jpg or video embedded in pkg)
    if let Some(data) = pkg.get(image_path) {
        if is_video_path(image_path) {
            let ext = std::path::Path::new(image_path)
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("mp4");
            let tmp = std::env::temp_dir().join(format!(
                "we_vidtex_{}.{ext}",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.subsec_nanos())
                    .unwrap_or(0)
            ));
            std::fs::write(&tmp, data)
                .with_context(|| format!("writing video temp file: {}", tmp.display()))?;
            let result = crate::render::ffmpeg::decode_first_frame(&tmp);
            let _ = std::fs::remove_file(&tmp);
            return result.map(LoadedImage::single);
        }
        return image::load_from_memory(data)
            .map(|i| LoadedImage::single(i.into_rgba8()))
            .with_context(|| format!("loading image from pkg: {image_path}"));
    }

    anyhow::bail!("texture not found in package: {image_path}")
}

/// Resolve a bare texture-reference path (same rules as any other material
/// texture reference — loose file under `dir`, `.tex` fallback, then `pkg`)
/// to a decoded image, or `None` if it doesn't resolve. Used by
/// `GpuSceneInstance::build` (gpu_renderer.rs) to load a spot light's
/// `cookie` texture without duplicating `load_texture_from_dir`/
/// `load_texture_from_pkg`'s resolution logic.
pub(crate) fn resolve_static_texture(
    dir: Option<&Path>,
    pkg: Option<&Package>,
    path: &str,
) -> Option<RgbaImage> {
    dir.and_then(|d| load_texture_from_dir(d, path).ok())
        .or_else(|| pkg.and_then(|p| load_texture_from_pkg(p, path).ok()))
        .map(|l| l.image)
}

/// Follow the model -> material -> texture reference chain in a PKG archive.
/// Shared global WE assets install, used as a fallback when a model/material/
/// texture isn't bundled with this specific wallpaper. Built-in particle
/// presets (e.g. "fog1") ship their own `particles/presets/*.json` per
/// wallpaper but rely on the shared `materials/presets/*.json` + `.tex`
/// living only in the global assets install, never copied into the
/// wallpaper's own directory/pkg — see [[wp_engine_project]] memory.
fn read_from_global_assets(rel_path: &str) -> Option<Vec<u8>> {
    let dir = crate::engine::shaders::loader::find_we_assets_dir()?;
    std::fs::read(dir.join(rel_path)).ok()
}

/// Returns the resolved texture plus the material pass's own `"blending"`
/// string (e.g. `"additive"`) and `ui_editor_properties_overbright`
/// brightness constant (default 1.0) — particles need both to composite
/// correctly (see `resolve_particle_sprite_{pkg,dir}`); plain image layers
/// just discard them.
fn find_model_chain_tex_pkg(
    pkg: &Package,
    json_path: &str,
) -> Result<(TexFile, Option<String>, f32)> {
    let data = pkg
        .get(json_path)
        .map(|d| d.to_vec())
        .or_else(|| read_from_global_assets(json_path))
        .with_context(|| {
            format!("model/material not found in pkg or global assets: {json_path}")
        })?;
    let val: serde_json::Value =
        serde_json::from_slice(&data).with_context(|| format!("parsing {json_path}"))?;

    if let Some(mat_path) = val.get("material").and_then(|v| v.as_str()) {
        return find_model_chain_tex_pkg(pkg, mat_path);
    }

    if let Some(passes) = val.get("passes").and_then(|v| v.as_array()) {
        for pass in passes {
            let blending = pass
                .get("blending")
                .and_then(|v| v.as_str())
                .map(str::to_string);
            let overbright = pass_overbright(pass);
            if let Some(textures) = pass.get("textures").and_then(|v| v.as_array()) {
                for tex_ref in textures {
                    if let Some(tex_name) = tex_ref.as_str() {
                        let tex_path = format!("materials/{tex_name}.tex");
                        if let Some(tex_data) = pkg
                            .get(&tex_path)
                            .map(|d| d.to_vec())
                            .or_else(|| read_from_global_assets(&tex_path))
                        {
                            let tex = TexFile::parse(&tex_data)
                                .with_context(|| format!("parsing {tex_path}"))?;
                            return Ok((tex, blending, overbright));
                        }

                        let alt_path = format!("{tex_name}.tex");
                        if let Some(tex_data) = pkg
                            .get(&alt_path)
                            .map(|d| d.to_vec())
                            .or_else(|| read_from_global_assets(&alt_path))
                        {
                            return Ok((TexFile::parse(&tex_data)?, blending, overbright));
                        }
                    }
                }
            }
        }
    }

    anyhow::bail!("could not resolve texture from {json_path}")
}

/// `ui_editor_properties_overbright` from a material pass's
/// `constantshadervalues` (CParticle reads it off the first pass; 1.0 when
/// absent).
fn pass_overbright(pass: &serde_json::Value) -> f32 {
    pass.get("constantshadervalues")
        .and_then(|c| c.get("ui_editor_properties_overbright"))
        .and_then(|v| v.as_f64())
        .map(|v| v as f32)
        .unwrap_or(1.0)
}

/// A mesh material's first pass. Meshes need two things out of it that the
/// texture chain doesn't carry: the flat `color` constant and `cullmode`.
fn material_first_pass(
    dir: Option<&Path>,
    pkg: Option<&Package>,
    json_path: &str,
) -> Option<serde_json::Value> {
    let data = read_asset_bytes(dir, pkg, json_path)
        .or_else(|| read_asset_bytes(dir, pkg, &json_path.to_lowercase()))?;
    let val: serde_json::Value = serde_json::from_slice(&data).ok()?;
    val.get("passes")?.as_array()?.first().cloned()
}

/// A material's flat `color` constant, as 8-bit RGB. Only meaningful for
/// materials with no texture at all — see `mesh3d_layer_from_object`.
fn pass_constant_color(pass: &serde_json::Value) -> Option<[u8; 3]> {
    let consts = pass.get("constantshadervalues")?;
    // WE keys this as `color`; `Color` is a separate (usually white) tint.
    let s = consts
        .get("color")
        .or_else(|| consts.get("Color"))?
        .as_str()?;
    let c = crate::engine::effect::parse_color(s);
    Some([
        (c[0].clamp(0.0, 1.0) * 255.0) as u8,
        (c[1].clamp(0.0, 1.0) * 255.0) as u8,
        (c[2].clamp(0.0, 1.0) * 255.0) as u8,
    ])
}

/// Follow the model -> material -> texture chain for loose files on disk.
fn find_model_chain_tex_dir(dir: &Path, json_path: &str) -> Result<(TexFile, Option<String>, f32)> {
    let data = std::fs::read(dir.join(json_path))
        .ok()
        .or_else(|| read_from_global_assets(json_path))
        .with_context(|| {
            format!(
                "model/material not found in {} or global assets: {json_path}",
                dir.display()
            )
        })?;
    let val: serde_json::Value =
        serde_json::from_slice(&data).with_context(|| format!("parsing {json_path}"))?;

    if let Some(mat_path) = val.get("material").and_then(|v| v.as_str()) {
        return find_model_chain_tex_dir(dir, mat_path);
    }

    if let Some(passes) = val.get("passes").and_then(|v| v.as_array()) {
        for pass in passes {
            let blending = pass
                .get("blending")
                .and_then(|v| v.as_str())
                .map(str::to_string);
            let overbright = pass_overbright(pass);
            if let Some(textures) = pass.get("textures").and_then(|v| v.as_array()) {
                for tex_ref in textures {
                    if let Some(tex_name) = tex_ref.as_str() {
                        let tex_path = format!("materials/{tex_name}.tex");
                        if let Some(tex_data) = std::fs::read(dir.join(&tex_path))
                            .ok()
                            .or_else(|| read_from_global_assets(&tex_path))
                        {
                            return Ok((TexFile::parse(&tex_data)?, blending, overbright));
                        }
                    }
                }
            }
        }
    }

    anyhow::bail!("could not resolve texture from {json_path}")
}

/// Walk model → material and report whether any pass declares the `SPRITESHEET`
/// combo (`image` is a multi-cell grid). `read` fetches a chain member's bytes.
fn chain_has_spritesheet(json_path: &str, read: &dyn Fn(&str) -> Option<Vec<u8>>) -> bool {
    let Some(data) = read(json_path) else {
        return false;
    };
    let Ok(val) = serde_json::from_slice::<serde_json::Value>(&data) else {
        return false;
    };
    if let Some(mat) = val.get("material").and_then(|v| v.as_str()) {
        return chain_has_spritesheet(mat, read);
    }
    val.get("passes")
        .and_then(|p| p.as_array())
        .is_some_and(|passes| {
            passes.iter().any(|pass| {
                pass.get("combos")
                    .and_then(|c| c.as_object())
                    // Materials spell it both "SPRITESHEET" and "spritesheet".
                    .and_then(|c| {
                        c.iter()
                            .find(|(k, _)| k.eq_ignore_ascii_case("spritesheet"))
                            .and_then(|(_, v)| v.as_i64())
                    })
                    .unwrap_or(0)
                    != 0
            })
        })
}

/// A model chain's texture, as a layer image. An animated `.tex` (TEXS frame
/// table + the IsGif flag) keeps its frames so the layer plays them —
/// otherwise the whole atlas gets squeezed into the one-cell quad, which is
/// how 2325500626's bonfire rendered as a 16-up contact sheet.
fn loaded_from_chain_tex(
    tex: &TexFile,
    json_path: &str,
    read: &dyn Fn(&str) -> Option<Vec<u8>>,
) -> Result<LoadedImage> {
    if tex.is_animated() {
        return LoadedImage::from_tex(tex);
    }
    Ok(apply_puppet_mesh(tex.to_rgba()?, json_path, read))
}

fn resolve_model_chain_pkg(pkg: &Package, json_path: &str) -> Result<LoadedImage> {
    let tex = find_model_chain_tex_pkg(pkg, json_path)?.0;
    if let Some(bytes) = tex.video_bytes() {
        // e.g. 2914504963's Taj Mahal backdrop: the model chain resolves to
        // an embedded-video texture — stream it instead of puppet handling.
        return video_tex_to_loaded(&tex, bytes);
    }
    let read = |rel: &str| {
        pkg.get(rel)
            .map(|d| d.to_vec())
            .or_else(|| read_from_global_assets(rel))
    };
    let mut loaded = loaded_from_chain_tex(&tex, json_path, &read)?;
    if chain_has_spritesheet(json_path, &read) {
        loaded.sprite_frames = tex.to_rgba_frames().unwrap_or_default();
    }
    Ok(loaded)
}

fn resolve_model_chain_dir(dir: &Path, json_path: &str) -> Result<LoadedImage> {
    let tex = find_model_chain_tex_dir(dir, json_path)?.0;
    if let Some(bytes) = tex.video_bytes() {
        return video_tex_to_loaded(&tex, bytes);
    }
    let read = |rel: &str| {
        std::fs::read(dir.join(rel))
            .ok()
            .or_else(|| read_from_global_assets(rel))
    };
    let mut loaded = loaded_from_chain_tex(&tex, json_path, &read)?;
    if chain_has_spritesheet(json_path, &read) {
        loaded.sprite_frames = tex.to_rgba_frames().unwrap_or_default();
    }
    Ok(loaded)
}

/// If the model JSON names a `"puppet"` mesh, its texture is a packed UV
/// atlas — reassemble it by rasterizing the rest-pose mesh (see
/// `engine::puppet`), and keep the parsed model + atlas around as a
/// `PuppetRuntime` so the live render loop can re-pose it over time when
/// the model carries MDLA animations. On any parse failure the decoded
/// texture passes through unchanged, so a malformed .mdl degrades to the
/// old scrambled-quad behavior instead of dropping the layer.
fn apply_puppet_mesh(
    atlas: RgbaImage,
    model_json_path: &str,
    read: impl Fn(&str) -> Option<Vec<u8>>,
) -> LoadedImage {
    let plain = |atlas: RgbaImage| LoadedImage::single(atlas);
    let Some(model_bytes) = read(model_json_path) else {
        return plain(atlas);
    };
    let Ok(model_json) = serde_json::from_slice::<serde_json::Value>(&model_bytes) else {
        return plain(atlas);
    };
    let Some(puppet_path) = model_json.get("puppet").and_then(|v| v.as_str()) else {
        return plain(atlas);
    };
    let Some(mdl_bytes) = read(puppet_path) else {
        tracing::warn!(target: "scene", "puppet mesh '{puppet_path}' not found — drawing raw atlas");
        return plain(atlas);
    };
    let Some(model) = crate::engine::puppet::parse_model(&mdl_bytes) else {
        tracing::warn!(target: "scene", "puppet mesh '{puppet_path}' unparsable — drawing raw atlas");
        return plain(atlas);
    };
    tracing::debug!(
        target: "scene",
        "puppet '{puppet_path}': {} vertices, {} triangles, {} bones, {} animations",
        model.mesh.positions.len(),
        model.mesh.indices.len() / 3,
        model.bones.len(),
        model.animations.len()
    );
    // With a skeleton + animation, frame 0 of the animation IS the
    // assembled pose (skin = worldAnim * inverse(atlas-space bind)); the
    // raw mesh alone only reproduces the packed atlas layout.
    let runtime = crate::engine::puppet::PuppetRuntime {
        model,
        atlas,
        layers: Vec::new(),
    };
    let assembled = if runtime.model.has_animation() {
        runtime.render_at(0.0, runtime.atlas.width(), runtime.atlas.height())
    } else {
        crate::engine::puppet::rasterize(
            &runtime.model.mesh,
            &runtime.atlas,
            runtime.atlas.width(),
            runtime.atlas.height(),
        )
    };
    let mut loaded = LoadedImage::single(assembled);
    if runtime.model.has_animation() {
        loaded.puppet = Some(std::sync::Arc::new(runtime));
    }
    loaded
}

/// Same model -> material -> texture chain as `resolve_model_chain_{pkg,dir}`,
/// but for particle sprites: keeps every sprite-sheet frame (sliced from the
/// `.tex`'s TEXS table, if any) instead of collapsing to one flat image, plus
/// the animation's total loop duration so `ParticleSystem::set_sprite_frames`
/// can drive per-particle frame advance, and the material pass's own
/// `"blending"` string so the caller can composite additive-glow presets
/// (fog/smoke/embers/rain/lightning — the overwhelming majority of real
/// particle materials) correctly instead of always alpha-blending, which
/// makes a sprite's near-black background visibly darken the scene instead
/// of contributing nothing.
fn resolve_particle_sprite_pkg(
    pkg: &Package,
    json_path: &str,
) -> Result<(particle::ParticleSprite, Option<String>)> {
    let (tex, blending, overbright) = find_model_chain_tex_pkg(pkg, json_path)?;
    let duration: f32 = tex.frames().iter().map(|f| f.frametime).sum();
    Ok((
        particle::ParticleSprite {
            frames: tex.to_particle_rgba_frames()?,
            duration,
            overbright,
        },
        blending,
    ))
}

fn resolve_particle_sprite_dir(
    dir: &Path,
    json_path: &str,
) -> Result<(particle::ParticleSprite, Option<String>)> {
    let (tex, blending, overbright) = find_model_chain_tex_dir(dir, json_path)?;
    let duration: f32 = tex.frames().iter().map(|f| f.frametime).sum();
    Ok((
        particle::ParticleSprite {
            frames: tex.to_particle_rgba_frames()?,
            duration,
            overbright,
        },
        blending,
    ))
}

// ---------------------------------------------------------------------------
// Parent-child transform composition (full TRS)
//
// In Wallpaper Engine a scene is a tree: an object's `origin`/`angles`/`scale`
// are expressed in its PARENT's frame, not screen space. `layer_from_object`
// bakes only the object's OWN local transform into each Layer, so anything with
// a parent lands at its local offset instead of its true world position (the
// clock/particles-in-the-corner bug). This pass walks each object's `parent`
// chain and pre-multiplies the parent's world transform onto the already-baked
// local layer.
//
// Position is exact TRS (parent scale → parent rotation → parent translation
// applied to the child point). Orientation and scale compose as Euler-sum and
// component-wise product respectively: exact for the flat z-rotation these 2D
// scenes use and for a single parent level, an approximation only under nested,
// non-coaxial 3D rotation (rare; the perspective path already treats layer
// angles approximately).
// ---------------------------------------------------------------------------

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
fn rotate_euler_f64(v: [f64; 3], a: [f64; 3]) -> [f64; 3] {
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
fn obj_local_xform(obj: &SceneObject) -> Xform {
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
        .and_then(crate::engine::scene::parse_value_vec3)
        .unwrap_or([1.0, 1.0, 1.0]);
    Xform { t, r, s }
}

/// Resolve an object's `parent` reference to an index in `scene.objects`.
/// WE stores the parent's `id` (a number), occasionally a name string.
fn parent_index(
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
        .and_then(crate::engine::scene::parse_value_bool)
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
fn world_xform(
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
    /// `(origin, angles, scale)` SceneScript sources per object.
    pub scripts: Vec<[Option<String>; 3]>,
    /// Object name/id, indexed the same as every other field here — feeds
    /// `ScriptContext::set_current_object`/`set_layers` so an
    /// object-attachment script (`thisLayer`, `thisScene.getLayer(...)`) can
    /// identify itself and look up other objects. See the Ghidra report's
    /// object-attachment follow-up.
    pub name: Vec<Option<String>>,
    pub id: Vec<Option<i64>>,
}

impl TransformGraph {
    /// True when at least one object's transform is script-driven AND at least
    /// one object is parented — the only case the per-frame pass is needed for.
    /// 189 of 197 real scenes answer `false` and skip the whole thing.
    pub fn needs_per_frame(&self) -> bool {
        self.parent.iter().any(Option::is_some)
            && self.scripts.iter().any(|s| s.iter().any(Option::is_some))
    }

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

    fn world_of(
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

fn build_transform_graph(scene: &Scene) -> TransformGraph {
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
    let script_of = |v: Option<&serde_json::Value>| {
        v.map(crate::engine::model::json_to_animated)
            .and_then(|a| a.script)
    };
    TransformGraph {
        parent: objects
            .iter()
            .map(|o| parent_index(o, &id_to_idx, &name_to_idx))
            .collect(),
        local: objects.iter().map(obj_local_xform).collect(),
        scripts: objects
            .iter()
            .map(|o| {
                [
                    script_of(o.origin.as_ref()),
                    script_of(o.angles.as_ref()),
                    script_of(o.scale.as_ref()),
                ]
            })
            .collect(),
        name: objects.iter().map(|o| o.name.clone()).collect(),
        id: objects.iter().map(|o| o.id).collect(),
    }
}

/// Parent world transform for object `idx` (identity if it has no parent). This
/// is what gets applied to a layer, whose OWN local transform is already baked
/// in by `layer_from_object`.
fn parent_world_xform(
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
fn apply_parent_transforms(
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

fn guess_scene_dimensions(scene: &Scene, layers: &[Layer]) -> (u32, u32) {
    // Perspective 3D scenes have no orthogonal projection and their layer
    // sizes are world-space, not pixels — inferring pixel dimensions from
    // them yields garbage (e.g. 1619x29). Render at a fixed 16:9 target; the
    // perspective camera maps world → NDC independently of this resolution.
    if scene.is_perspective() {
        return (1920, 1080);
    }

    // 1. Try orthogonal projection dimensions from camera or general settings
    if let Some(cam) = &scene.camera {
        if let Some(proj) = &cam.orthogonal_projection {
            if let (Some(w), Some(h)) = (proj.width, proj.height) {
                if w > 0 && h > 0 {
                    return (w, h);
                }
            }
        }
    }
    if let Some(gen) = &scene.general {
        if let Some(proj) = &gen.orthogonal_projection {
            if let (Some(w), Some(h)) = (proj.width, proj.height) {
                if w > 0 && h > 0 {
                    return (w, h);
                }
            }
        }
    }

    // 2. Use the largest layer dimensions
    let mut max_w = 0u32;
    let mut max_h = 0u32;
    for layer in layers {
        max_w = max_w.max(layer.image.width());
        max_h = max_h.max(layer.image.height());
    }
    if max_w > 0 && max_h > 0 {
        return (max_w, max_h);
    }

    // 3. Fallback
    (1920, 1080)
}

/// For model/copybackground layers we can't render, create a transparent
/// placeholder so any screen-space effects applied to them have a base to work
/// with. It must be transparent, not opaque grey: these are usually full-screen
/// effect-only overlays (e.g. "Light shafts"), so an opaque base paints a giant
/// grey box over the scene whenever the effect is a SKIP or an additive overlay.
/// ponytail: transparent base; revisit if an effect needs to sample the real
/// scene behind it (backdrop capture — not implemented).
fn placeholder_for(obj: &SceneObject) -> RgbaImage {
    let size = obj.parsed_size();
    let w = if size[0] > 0.0 { size[0] as u32 } else { 1920 };
    let h = if size[1] > 0.0 { size[1] as u32 } else { 1080 };
    RgbaImage::from_pixel(w, h, image::Rgba([0u8, 0, 0, 0]))
}

fn is_special_layer(image_path: &str) -> bool {
    let p = image_path.to_lowercase();
    p.contains("projectlayer") || p.contains("composelayer") || p.contains("fullscreenlayer")
}

/// WE's solid-color model bundles (`models/util/solidlayer.json`,
/// `solid_instance_model_*.json`) reference a "flat" shader that ignores its
/// texture entirely and just outputs `vec4(g_Color, g_Alpha)` — i.e. the
/// object's own color/alpha, no image. A 1×1 white texture reproduces that
/// exactly once tinted by `Layer::color`/`alpha`/`brightness`.
fn is_solid_layer_path(image_path: &str) -> bool {
    let p = image_path.to_lowercase();
    p.contains("solidlayer") || p.contains("solid_instance_model")
}

fn white_pixel() -> RgbaImage {
    RgbaImage::from_pixel(1, 1, image::Rgba([255, 255, 255, 255]))
}

/// WE sizes a solid layer to the full scene when it has no explicit size
/// (`CImage.cpp:199`: `if (solidlayer && size == 0) size = sceneSize`) — but
/// only to give an effect chain a dummy target. A size-less solid layer with
/// no effects is a script/cursor holder ("Player Options" in 2988724897);
/// stretching its opaque white over the scene hides the whole wallpaper, so
/// callers only register those that carry effects.
fn fill_solid_layer_sizes(layers: &mut [Layer], solid_indices: &[usize], width: u32, height: u32) {
    for &i in solid_indices {
        if layers[i].size[0] == 0.0 && layers[i].size[1] == 0.0 {
            layers[i].size = [width as f64, height as f64, 0.0];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Untextured mesh materials are common in real content
    /// (`"textures": [null,null,null]`), and their flat `color` is the only
    /// thing that keeps the mesh from drawing plain white.
    /// `maxwidth`/`maxrows` are only active when their `limit*` checkbox is
    /// on. WE keeps the numbers around while the box is unchecked, so applying
    /// them unconditionally wrapped text that should run free — 3091474852's
    /// day label wrapped at 500px despite `limitwidth: false`.
    /// `disablepropagation` severs transform inheritance while leaving the
    /// `parent` link intact for grouping/visibility.
    #[test]
    fn disablepropagation_severs_transform_inheritance() {
        let objs: Vec<SceneObject> = serde_json::from_str(
            r#"[{"id": 1, "origin": "10 0 0"},
                {"id": 2, "origin": "1 0 0", "parent": 1},
                {"id": 3, "origin": "1 0 0", "parent": 1, "disablepropagation": true}]"#,
        )
        .expect("valid objects");
        let id_to_idx: HashMap<i64, usize> = objs
            .iter()
            .enumerate()
            .filter_map(|(i, o)| o.id.map(|d| (d, i)))
            .collect();
        let name_to_idx: HashMap<&str, usize> = HashMap::new();
        assert_eq!(parent_index(&objs[1], &id_to_idx, &name_to_idx), Some(0));
        assert_eq!(
            parent_index(&objs[2], &id_to_idx, &name_to_idx),
            None,
            "disablepropagation must stop transform inheritance"
        );
    }

    /// The object-attachment API's `thisLayer.disablepropagation()` (see
    /// `engine::script::OBJECT_ATTACHMENT_API_JS`) is this same
    /// `disablepropagation` rule, just evaluated live per-frame instead of
    /// baked once from scene.json — `TransformGraph::world`'s `disabled`
    /// slice must sever object `i`'s *own* inheritance from its parent
    /// (matching `parent_index` above exactly), not affect what `i`'s
    /// children inherit from `i`.
    #[test]
    fn live_disablepropagation_severs_only_the_flagged_objects_own_inheritance() {
        // 0 (root, moves) -> 1 (child of 0) -> 2 (grandchild, child of 1)
        let graph = TransformGraph {
            parent: vec![None, Some(0), Some(1)],
            local: vec![Xform::IDENTITY; 3],
            scripts: vec![[None, None, None], [None, None, None], [None, None, None]],
            name: vec![None, None, None],
            id: vec![None, None, None],
        };
        let locals = vec![
            Xform { t: [10.0, 0.0, 0.0], r: [0.0; 3], s: [1.0; 3] },
            Xform { t: [1.0, 0.0, 0.0], r: [0.0; 3], s: [1.0; 3] },
            Xform { t: [1.0, 0.0, 0.0], r: [0.0; 3], s: [1.0; 3] },
        ];

        let worlds = graph.world(&locals, &[false, false, false]);
        assert_eq!(worlds[2].t, [12.0, 0.0, 0.0], "baseline: full chain composes");

        // Sever object 1's own inheritance from object 0.
        let worlds = graph.world(&locals, &[false, true, false]);
        assert_eq!(worlds[0].t, [10.0, 0.0, 0.0], "object 0 is untouched");
        assert_eq!(
            worlds[1].t,
            [1.0, 0.0, 0.0],
            "object 1 must stop inheriting object 0's transform"
        );
        assert_eq!(
            worlds[2].t,
            [2.0, 0.0, 0.0],
            "object 2 still inherits normally from (now-severed) object 1"
        );
    }

    /// `depthtest: "disabled"` on the object overrides the material pass.
    #[test]
    fn object_depthtest_overrides_material() {
        let read = |o: &SceneObject, mat: Option<&str>| -> bool {
            o.depthtest
                .as_ref()
                .and_then(|v| v.as_str())
                .or(mat)
                .map(|m| m != "disabled")
                .unwrap_or(true)
        };
        let plain: SceneObject = serde_json::from_str("{}").expect("valid");
        let off: SceneObject = serde_json::from_str(r#"{"depthtest": "disabled"}"#).expect("valid");
        assert!(read(&plain, None), "default is depth-tested");
        assert!(!read(&plain, Some("disabled")), "material pass applies");
        assert!(!read(&off, Some("enabled")), "object wins over material");
    }

    #[test]
    fn wrap_limits_respect_their_gates() {
        let obj = |lw: bool| -> SceneObject {
            serde_json::from_value(serde_json::json!({
                "text": "SUNDAY", "maxwidth": 500.0, "maxrows": 1,
                "limitwidth": lw, "limitrows": lw,
            }))
            .expect("valid object")
        };
        let active = |o: &SceneObject| {
            o.maxwidth
                .as_ref()
                .and_then(crate::engine::scene::parse_value_f32)
                .filter(|w| *w > 0.0)
                .filter(|_| {
                    o.limitwidth
                        .as_ref()
                        .and_then(crate::engine::scene::parse_value_bool)
                        .unwrap_or(false)
                })
                .is_some()
        };
        assert!(active(&obj(true)), "gate on: maxwidth applies");
        assert!(!active(&obj(false)), "gate off: maxwidth must be ignored");
    }

    #[test]
    fn material_constant_color_reads_flat_color() {
        // `color` wins over the separate (usually white) `Color` tint.
        let m: serde_json::Value = serde_json::from_str(
            r#"{"textures":[null],"constantshadervalues":
               {"Color":"1.0 1.0 1.0","color":"0.6 0.0 0.2"}}"#,
        )
        .expect("valid json");
        assert_eq!(pass_constant_color(&m), Some([153, 0, 51]));
        assert_eq!(pass_constant_color(&serde_json::json!({})), None);
    }

    fn solid_particle_config(color: &str) -> particle::ParticleConfig {
        let json = format!(
            r#"{{
                "maxcount": 2,
                "emitter": [{{"name":"box","rate":1000,"distancemin":"0 0 0","distancemax":"0 0 0"}}],
                "initializer": [
                    {{"id":1,"name":"lifetimerandom","min":100,"max":100}},
                    {{"id":2,"name":"sizerandom","min":300,"max":300}},
                    {{"id":3,"name":"velocityrandom","min":"0 0 0","max":"0 0 0"}},
                    {{"id":4,"name":"colorrandom","min":"{color}","max":"{color}"}},
                    {{"id":5,"name":"alpharandom","min":1,"max":1}}
                ]
            }}"#
        );
        serde_json::from_str(&json).expect("should parse")
    }

    /// Particles should interleave with image layers by `order_index` in
    /// true scene z-order, not always draw on top of every image — build a
    /// scene with [particle(order 0, blue), image(order 1, opaque red),
    /// particle(order 2, green)] and confirm the final canvas shows the
    /// blue particle fully hidden under the opaque red layer, with the green
    /// particle visible on top of it.
    #[test]
    fn particles_and_images_interleave_by_order_index() {
        let scene: Scene = serde_json::from_str("{}").unwrap();
        let red_image = RgbaImage::from_pixel(100, 100, image::Rgba([255, 0, 0, 255]));

        let layers = vec![Layer {
            local_origin: [0.0; 3],
            local_scale: [1.0; 3],
            local_angles: [0.0; 3],
            name: "red".to_string(),
            image: red_image,
            extra_frames: Vec::new(),
            frame_duration_ms: 0,
            origin: [50.0, 50.0, 0.0],
            size: [100.0, 100.0, 0.0],
            scale: [1.0, 1.0, 1.0],
            parallax_depth: [0.0, 0.0],
            angle: 0.0,
            angles: [0.0, 0.0, 0.0],
            blend_mode: 0,
            alpha: 1.0,
            alpha_script: None,
            color: [1.0, 1.0, 1.0],
            brightness: 1.0,
            copybackground: false,
            no_interpolation: true,
            clamp_uvs: false,
            puppet: None,
            video: None,
            text_dynamic: None,
            order_index: 1,
            object_id: None,
            transform_scripts: TransformScripts::default(),
            visible_base: true,
        }];

        let particle_layers = vec![
            ParticleLayer {
                local_origin: [0.0; 3],
                name: "blue_particles".to_string(),
                origin: [50.0, 50.0, 0.0],
                parallax_depth: [0.0, 0.0],
                config: solid_particle_config("0 0 255"),
                overrides: None,
                sprite_texture: None,
                additive_blend: false,
                order_index: 0,
                children: Vec::new(),
            },
            ParticleLayer {
                local_origin: [0.0; 3],
                name: "green_particles".to_string(),
                origin: [50.0, 50.0, 0.0],
                parallax_depth: [0.0, 0.0],
                config: solid_particle_config("0 255 0"),
                overrides: None,
                sprite_texture: None,
                additive_blend: false,
                order_index: 2,
                children: Vec::new(),
            },
        ];

        let resolved = ResolvedScene {
            transform_graph: build_transform_graph(&scene),
            width: 100,
            height: 100,
            layers,
            particle_layers,
            mesh3d_layers: Vec::new(),
            scene,
        };

        let canvas = resolved.render();
        let px = canvas.get_pixel(50, 50);
        assert!(
            px[2] < 20,
            "blue particle (order 0) should be fully hidden under the opaque red layer (order 1), got {px:?}"
        );
        assert!(
            px[1] > px[0],
            "green particle (order 2) should be visible on top of the red layer, got {px:?}"
        );
    }

    #[test]
    fn parallax_depth_number_broadcasts_to_both_axes() {
        let v = serde_json::json!(0.5);
        assert_eq!(parse_parallax_depth(&v), [0.5, 0.5]);
    }

    #[test]
    fn parallax_depth_vec2_string_keeps_axes_independent() {
        let v = serde_json::json!("0.5 0.0");
        assert_eq!(parse_parallax_depth(&v), [0.5, 0.0]);
    }

    #[test]
    fn parallax_depth_user_wrapped_value_unwraps() {
        let v = serde_json::json!({"user": "ui_editor_properties_depth", "value": "0.25 0.75"});
        assert_eq!(parse_parallax_depth(&v), [0.25, 0.75]);
    }

    #[test]
    fn alignment_none_produces_no_offset() {
        assert_eq!(alignment_offset(None, [100.0, 200.0]), [0.0, 0.0]);
    }

    #[test]
    fn alignment_top_shifts_down_by_half_height() {
        assert_eq!(alignment_offset(Some("top"), [100.0, 200.0]), [0.0, -100.0]);
    }

    #[test]
    fn alignment_bottom_shifts_up_by_half_height() {
        assert_eq!(
            alignment_offset(Some("bottom"), [100.0, 200.0]),
            [0.0, 100.0]
        );
    }

    #[test]
    fn alignment_left_shifts_right_by_half_width() {
        assert_eq!(alignment_offset(Some("left"), [100.0, 200.0]), [50.0, 0.0]);
    }

    #[test]
    fn alignment_right_shifts_left_by_half_width() {
        assert_eq!(
            alignment_offset(Some("right"), [100.0, 200.0]),
            [-50.0, 0.0]
        );
    }

    #[test]
    fn alignment_combines_vertical_and_horizontal() {
        assert_eq!(
            alignment_offset(Some("top left"), [100.0, 200.0]),
            [50.0, -100.0]
        );
    }

    #[test]
    fn alignment_top_and_bottom_together_only_honors_top() {
        // Matches the reference's if/else-if: mutually exclusive per axis.
        assert_eq!(
            alignment_offset(Some("top bottom"), [100.0, 200.0]),
            [0.0, -100.0]
        );
    }

    #[test]
    fn alignment_is_case_insensitive() {
        assert_eq!(alignment_offset(Some("TOP"), [100.0, 200.0]), [0.0, -100.0]);
    }
}
