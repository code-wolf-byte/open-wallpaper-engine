use anyhow::{Context, Result};
use image::RgbaImage;
use std::collections::HashMap;
use std::path::Path;

use crate::engine::model::scene::{Scene, SceneObject};
use crate::engine::sim::particle;
use crate::formats::pkg::Package;
use crate::formats::tex::TexFile;
use crate::workshop::properties::SceneProperties;

mod textures;
pub use textures::*;

mod transform;
pub use transform::*;

mod layers;
pub use layers::*;

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
/// real geometry through the perspective camera (see `engine::model::mesh3d`), not as
/// a flat quad — spheres, skyboxes and cylinders can't be billboarded.
pub struct Mesh3dLayer {
    pub name: String,
    pub mesh: crate::engine::model::mesh3d::Mesh3d,
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
    pub font_data: Vec<u8>,
    pub point_size: f32,
    /// Text currently rasterized into the layer image (the `value` WE feeds
    /// back into `update` each tick).
    pub last_text: String,
}

/// What a backdrop layer captures each frame (see `backdrop_layer_from_object`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backdrop {
    /// The scene beneath the layer's own quad (composition layer).
    Region,
    /// The whole scene (fullscreen / projection layer).
    FullScene,
}

pub struct Layer {
    /// `Some` for composition/fullscreen layers, whose texture is the scene
    /// beneath them rather than an image.
    pub backdrop: Option<Backdrop>,
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
    pub puppet: Option<std::sync::Arc<crate::engine::sim::puppet::PuppetRuntime>>,
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
    /// The object's AUTHORED `visible` value — the starting visibility and the
    /// `value` fed to a `visible` script. Critical: scripts that only define
    /// cursor handlers (invisible click hitboxes, `"value": false`) have no
    /// `update()`, so the script returns `value` unchanged — feeding `true`
    /// here would render those hitboxes as opaque boxes.
    pub visible_base: bool,
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
                    if let Some(mut layer) = backdrop_layer_from_object(obj, path) {
                        if layer.size[0] == 0.0 || layer.size[1] == 0.0 {
                            solid_indices.push(layers.len());
                        }
                        layer.order_index = obj_index;
                        layers.push(layer);
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
                    if let Some(mut layer) = backdrop_layer_from_object(obj, path) {
                        if layer.size[0] == 0.0 || layer.size[1] == 0.0 {
                            solid_indices.push(layers.len());
                        }
                        layer.order_index = obj_index;
                        layers.push(layer);
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
                    if let Some(mut layer) = backdrop_layer_from_object(obj, path) {
                        if layer.size[0] == 0.0 || layer.size[1] == 0.0 {
                            solid_indices.push(layers.len());
                        }
                        layer.order_index = obj_index;
                        layers.push(layer);
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
            let out = crate::engine::gpu::blend::apply_blending(
                layer.blend_mode,
                dest_rgb,
                src_rgb,
                src_a,
            );
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

impl Drop for VideoLayerStream {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
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

#[cfg(test)]
mod tests;
