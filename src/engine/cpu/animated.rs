use anyhow::{Context, Result};
use image::RgbaImage;
use std::path::Path;
use std::sync::mpsc::SyncSender;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::engine::assets::AssetStore;
use crate::engine::cpu::effect::SceneEffect;
use crate::engine::model::resolve::ResolvedScene;
use crate::engine::model::scene::SceneObject;
use crate::engine::sim::particle::{
    InstanceOverride, ParticleConfig, ParticleSprite, ParticleSystem,
};
use crate::formats::tex::TexFile;

struct SceneAnimState {
    /// Paired with each layer's `order_index` (its position in
    /// `scene.visible_objects()`) so `render_frame` can interleave with
    /// particles in true scene z-order instead of drawing all particles
    /// after all images.
    base_layers: Vec<(usize, Arc<RgbaImage>)>,
    /// Each system paired with its `order_index` and resolved sprite texture
    /// (from the preset's `material` field, if any) — `None` sprite falls
    /// back to `render_onto`'s flat-color circle draw.
    particles: Vec<(usize, ParticleSystem, Option<ParticleSprite>)>,
    effects: Vec<LayerEffect>,
    width: u32,
    height: u32,
}

struct LayerEffect {
    effect_name: String,
    values: serde_json::Value,
}

impl SceneAnimState {
    fn load(dir: &Path) -> Result<Self> {
        let resolved = ResolvedScene::from_directory(dir)?;
        let width = resolved.width;
        let height = resolved.height;

        // ponytail: layers flatten to one image here, so an embedded-video
        // layer's stream handle is dropped and this path shows its first frame
        // forever. The live GPU path streams them properly
        // (`gpu_renderer`'s per-frame `latest_frame` re-upload); this is only
        // the fallback for when GPU init fails, and streaming here would mean a
        // Lanczos3 resize of a 4K frame every tick on the CPU. Carry the
        // `video` handle through and swap per frame if that fallback ever
        // becomes a path people actually watch.
        let base_layers: Vec<(usize, Arc<RgbaImage>)> = resolved
            .layers
            .into_iter()
            .map(|l| {
                let order_index = l.order_index;
                let resized = if l.image.width() != width || l.image.height() != height {
                    image::imageops::resize(
                        &l.image,
                        width,
                        height,
                        image::imageops::FilterType::Lanczos3,
                    )
                } else {
                    l.image
                };
                (order_index, Arc::new(resized))
            })
            .collect();

        // Parse particles/effects through the same loose-file + PKG asset
        // resolver that the scene graph uses.
        let mut particles = Vec::new();
        let mut effects = Vec::new();

        if let Ok(assets) = AssetStore::from_directory(dir) {
            if let Ok(scene_json) = assets.scene_json() {
                if let Ok(scene) = crate::engine::model::scene::Scene::from_json(&scene_json) {
                    // Same iteration order `ResolvedScene` uses for `layers`
                    // (`scene.visible_objects()`), so `order_index` here is
                    // directly comparable to `Layer::order_index` above.
                    for (obj_index, obj) in scene.visible_objects().enumerate() {
                        load_particles(&assets, obj, height, obj_index, &mut particles);
                        load_effects(obj, &mut effects);
                    }
                }
            }
        }

        Ok(Self {
            base_layers,
            particles,
            effects,
            width,
            height,
        })
    }

    fn render_frame(&mut self, time: f32, dt: f32) -> RgbaImage {
        let mut canvas = RgbaImage::new(self.width, self.height);

        // Interleave images and particles by `order_index` (true scene
        // z-order) instead of drawing all particles after all images.
        enum DrawItem {
            Image(usize),
            Particle(usize),
        }
        let mut items: Vec<(usize, DrawItem)> = self
            .base_layers
            .iter()
            .enumerate()
            .map(|(i, (order, _))| (*order, DrawItem::Image(i)))
            .chain(
                self.particles
                    .iter()
                    .enumerate()
                    .map(|(i, (order, _, _))| (*order, DrawItem::Particle(i))),
            )
            .collect();
        items.sort_by_key(|(order, _)| *order);

        for (_, item) in items {
            match item {
                DrawItem::Image(i) => {
                    image::imageops::overlay(&mut canvas, self.base_layers[i].1.as_ref(), 0, 0);
                }
                DrawItem::Particle(i) => {
                    let (_, ps, sprite) = &mut self.particles[i];
                    ps.step(dt);
                    ps.render_onto(&mut canvas, sprite.as_ref(), [0.0, 0.0]);
                }
            }
        }

        for effect in &self.effects {
            if let Some(eff) =
                SceneEffect::from_effect_name(&effect.effect_name, &effect.values, time)
            {
                eff.apply(&mut canvas);
            }
        }

        canvas
    }
}

/// Run an animated scene render loop, sending frames through the channel.
/// Exits when the receiver is dropped.
pub fn scene_render_loop(
    dir: &Path,
    tx: &SyncSender<Arc<RgbaImage>>,
    target_fps: f64,
) -> Result<()> {
    let mut state = SceneAnimState::load(dir)
        .with_context(|| format!("loading animated scene from {}", dir.display()))?;

    let frame_duration = Duration::from_secs_f64(1.0 / target_fps);
    let start = Instant::now();
    let mut last_frame = start;

    loop {
        let now = Instant::now();
        let time = now.duration_since(start).as_secs_f32();
        let dt = now.duration_since(last_frame).as_secs_f32();
        last_frame = now;

        let frame = state.render_frame(time, dt);
        if tx.send(Arc::new(frame)).is_err() {
            return Ok(());
        }

        let elapsed = now.elapsed();
        if elapsed < frame_duration {
            std::thread::sleep(frame_duration - elapsed);
        }
    }
}

fn load_particles(
    assets: &AssetStore,
    obj: &SceneObject,
    height: u32,
    order_index: usize,
    particles: &mut Vec<(usize, ParticleSystem, Option<ParticleSprite>)>,
) {
    let particle_ref = match &obj.particle {
        Some(serde_json::Value::String(s)) => s.clone(),
        _ => return,
    };

    let data = match assets.read_string(&particle_ref) {
        Ok(d) => d,
        Err(_) => return,
    };

    // WE origin is the object's center in absolute scene coordinates
    // (Y-up, (0,0) at bottom-left); convert to top-left pixel coords, matching
    // render.rs's image-layer positioning.
    let origin = obj.parsed_origin();
    let spawn_center = [origin[0] as f32, height as f32 - origin[1] as f32];

    let overrides: Option<InstanceOverride> = obj
        .instanceoverride
        .as_ref()
        .and_then(|v| serde_json::from_value(v.clone()).ok());

    // `instanceoverride.enabled: false` turns the whole system off.
    if overrides.as_ref().is_some_and(|o| {
        o.enabled
            .as_ref()
            .is_some_and(|v| v.as_bool() == Some(false) || v.as_u64() == Some(0))
    }) {
        return;
    }

    if let Ok(config) = serde_json::from_str::<ParticleConfig>(&data) {
        let sprite = config
            .material
            .as_deref()
            .and_then(|mat_path| resolve_particle_sprite(assets, mat_path));
        let mut system = ParticleSystem::from_config(&config, spawn_center, overrides.as_ref());
        if let Some(sprite) = &sprite {
            system.set_sprite_frames(sprite.frames.len(), sprite.duration);
        }
        particles.push((order_index, system, sprite));
    }
}

/// Resolves a particle preset's `material` field (a material JSON path, e.g.
/// `"materials/presets/water_faucet.json"`) to its first pass's base
/// texture (every sprite-sheet frame, if animated), following one level of
/// `material`-field indirection in case the path points at a model JSON
/// instead (mirrors `render.rs`'s `resolve_particle_sprite_dir`/`_pkg`, but
/// via `AssetStore` since this loader doesn't have direct `dir`/`pkg` access).
fn resolve_particle_sprite(assets: &AssetStore, material_path: &str) -> Option<ParticleSprite> {
    let mut val: serde_json::Value = assets.read_json(material_path).ok()?;
    if let Some(mat_path) = val.get("material").and_then(|v| v.as_str()) {
        val = assets.read_json(mat_path).ok()?;
    }
    let passes = val.get("passes")?.as_array()?;
    for pass in passes {
        let Some(textures) = pass.get("textures").and_then(|v| v.as_array()) else {
            continue;
        };
        for tex_ref in textures {
            if let Some(tex_name) = tex_ref.as_str() {
                if let Some(sprite) = resolve_particle_sprite_asset(assets, tex_name) {
                    return Some(sprite);
                }
            }
        }
    }
    None
}

fn resolve_particle_sprite_asset(assets: &AssetStore, tex_name: &str) -> Option<ParticleSprite> {
    let asset = assets.read_texture(tex_name).ok()?;
    match TexFile::parse(&asset.bytes) {
        Ok(tex) => {
            let duration: f32 = tex.frames().iter().map(|f| f.frametime).sum();
            tex.to_particle_rgba_frames()
                .ok()
                .map(|frames| ParticleSprite {
                    frames,
                    duration,
                    overbright: 1.0,
                })
        }
        Err(_) => image::load_from_memory(&asset.bytes)
            .ok()
            .map(|img| ParticleSprite::single(img.into_rgba8())),
    }
}

fn load_effects(obj: &SceneObject, effects: &mut Vec<LayerEffect>) {
    for eff in &obj.effects {
        let effect_name = eff
            .name_string()
            .or_else(|| eff.file.clone())
            .unwrap_or_default();

        for pass in &eff.passes {
            if let Some(material) = &pass.material {
                let name = material
                    .trim_start_matches("materials/effects/")
                    .trim_end_matches(".json")
                    .to_string();
                effects.push(LayerEffect {
                    effect_name: name,
                    values: serde_json::Value::Object(serde_json::Map::new()),
                });
            }
        }

        if effect_name.contains('/') {
            let short = effect_name
                .rsplit('/')
                .next()
                .unwrap_or(&effect_name)
                .trim_end_matches(".json");
            effects.push(LayerEffect {
                effect_name: short.to_string(),
                values: serde_json::Value::Object(serde_json::Map::new()),
            });
        }
    }
}
