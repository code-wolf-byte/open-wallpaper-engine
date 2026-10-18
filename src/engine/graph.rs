use anyhow::Result;
use std::path::Path;

use super::assets::AssetStore;
use super::material::{EffectDefinition, Model};
use crate::engine::model::scene::{Scene, SceneObject};

pub struct SceneGraph {
    pub assets: AssetStore,
    pub scene: Scene,
    pub images: Vec<ImageNode>,
}

impl SceneGraph {
    #[tracing::instrument(target = "scene", level = "debug", fields(dir = %dir.display()))]
    pub fn from_directory(dir: &Path) -> Result<Self> {
        tracing::debug!(target: "scene", "building scene graph from directory");
        let assets = AssetStore::from_directory(dir)?;
        let scene_json = assets.scene_json()?;
        let scene = Scene::from_json(&scene_json)?;
        Self::from_scene(assets, scene)
    }

    pub fn from_scene(assets: AssetStore, scene: Scene) -> Result<Self> {
        let mut images = Vec::new();

        // Effective visibility (own AND every ancestor) — see Scene::visibility_mask.
        let visible = scene.visibility_mask();
        for object_index in scene.topological_render_order_indices() {
            let object = &scene.objects[object_index];
            if !visible[object_index] {
                continue;
            }
            let Some(image_path) = object.image.as_deref() else {
                continue;
            };

            let model = image_path
                .ends_with(".json")
                .then(|| Model::load(&assets, image_path))
                .transpose();

            let model = match model {
                Ok(model) => model,
                Err(err) => {
                    tracing::warn!(target: "scene", "could not resolve model for {:?}: {err}", object.name);
                    None
                }
            };

            let mut effects = Vec::new();
            for effect in &object.effects {
                let Some(file) = effect.file.as_deref() else {
                    continue;
                };

                match EffectDefinition::load(&assets, file) {
                    Ok(definition) => effects.push(ResolvedImageEffect {
                        id: effect.id,
                        name: effect.name_string(),
                        visible: effect.is_visible(),
                        definition,
                    }),
                    Err(err) => {
                        tracing::warn!(
                            target: "scene",
                            "could not resolve effect {file:?} for {:?}: {err}",
                            object.name
                        );
                    }
                }
            }

            images.push(ImageNode {
                object_index,
                id: object.id,
                name: object.name.clone().unwrap_or_default(),
                image_path: image_path.to_string(),
                base_texture: model
                    .as_ref()
                    .and_then(Model::first_texture)
                    .map(str::to_string)
                    .or_else(|| (!image_path.ends_with(".json")).then(|| image_path.to_string())),
                model,
                effects,
            });
        }

        tracing::debug!(target: "scene", images = images.len(), "scene graph built");
        Ok(Self {
            assets,
            scene,
            images,
        })
    }

    pub fn object(&self, node: &ImageNode) -> &SceneObject {
        &self.scene.objects[node.object_index]
    }

    pub fn render_size(&self) -> (u32, u32) {
        if let Some(cam) = &self.scene.camera {
            if let Some(proj) = &cam.orthogonal_projection {
                if let (Some(w), Some(h)) = (proj.width, proj.height) {
                    if w > 0 && h > 0 {
                        return (w, h);
                    }
                }
            }
        }

        if let Some(general) = &self.scene.general {
            if let Some(proj) = &general.orthogonal_projection {
                if let (Some(w), Some(h)) = (proj.width, proj.height) {
                    if w > 0 && h > 0 {
                        return (w, h);
                    }
                }
            }
        }

        for node in &self.images {
            if let Some(model) = &node.model {
                if let (Some(w), Some(h)) = (model.width, model.height) {
                    if w > 0 && h > 0 {
                        return (w, h);
                    }
                }
            }
        }

        (1920, 1080)
    }

    pub fn stats(&self) -> SceneGraphStats {
        let mut stats = SceneGraphStats {
            objects: self.scene.objects.len(),
            image_nodes: self.images.len(),
            ..SceneGraphStats::default()
        };

        if let Some(camera) = &self.scene.camera {
            stats.camera_fields += usize::from(camera.center.is_some());
            stats.camera_fields += usize::from(camera.eye.is_some());
            stats.camera_fields += usize::from(camera.parallax_amount.is_some());
        }
        if let Some(general) = &self.scene.general {
            stats.general_fields += usize::from(general.supports_audio_processing.is_some());
            stats.general_fields += usize::from(general.speed.is_some());
        }

        for object in &self.scene.objects {
            stats.angled_objects += usize::from(object.angles.is_some());
            for effect in &object.effects {
                for pass in &effect.passes {
                    stats.scene_override_textures += pass.textures.len();
                    for texture in pass.textures.iter().flatten() {
                        stats.scene_override_texture_names += usize::from(texture.file.is_some());
                        stats.scene_override_texture_names += usize::from(texture.name.is_some());
                    }
                }
            }
        }

        for node in &self.images {
            if node.id.is_some() {
                stats.objects_with_ids += 1;
            }
            if !node.name.is_empty() {
                stats.named_images += 1;
            }
            if node.is_render_target_layer() {
                stats.render_target_layers += 1;
            }
            if node.is_fullscreen_layer() {
                stats.fullscreen_layers += 1;
            }

            if let Some(model) = &node.model {
                stats.models += 1;
                stats.materials += 1;
                stats.model_bytes += model.filename.len();
                stats.model_bytes += model.material.filename.len();
                stats.solid_models += usize::from(model.solid_layer);
                stats.passthrough_models += usize::from(model.passthrough);
                stats.autosize_models += usize::from(model.autosize);
                stats.no_padding_models += usize::from(model.no_padding);
                stats.sized_models += usize::from(model.width.is_some() || model.height.is_some());
                stats.puppet_models += usize::from(model.puppet.is_some());
                collect_material_stats(&model.material, &mut stats);
            }

            stats.effects += node.effects.len();
            for effect in &node.effects {
                if effect.id.is_some() {
                    stats.effects_with_ids += 1;
                }
                if effect.name.as_deref().is_some_and(|name| !name.is_empty()) {
                    stats.named_effects += 1;
                }
                if effect.visible {
                    stats.visible_effects += 1;
                }

                let definition = &effect.definition;
                stats.effect_metadata_bytes += definition.filename.len()
                    + definition.name.len()
                    + definition.description.len()
                    + definition.group.len()
                    + definition.preview.len();
                stats.effect_dependencies += definition.dependencies.len();
                stats.effect_passes += definition.passes.len();
                stats.effect_fbos += definition.fbos.len();

                for fbo in &definition.fbos {
                    stats.effect_metadata_bytes += fbo.name.len() + fbo.format.len();
                    if fbo.scale != 1.0 {
                        stats.scaled_fbos += 1;
                    }
                    if fbo.unique {
                        stats.unique_fbos += 1;
                    }
                }

                for pass in &definition.passes {
                    stats.effect_binds += pass.binds.len();
                    stats.effect_commands += usize::from(pass.command.is_some());
                    stats.effect_sources += usize::from(pass.source.is_some());
                    stats.effect_targets += usize::from(pass.target.is_some());
                    if let Some(material) = &pass.material {
                        stats.materials += 1;
                        collect_material_stats(material, &mut stats);
                    }
                }
            }
        }

        stats
    }
}

#[derive(Debug, Default, Clone)]
pub struct SceneGraphStats {
    pub objects: usize,
    pub objects_with_ids: usize,
    pub camera_fields: usize,
    pub general_fields: usize,
    pub angled_objects: usize,
    pub scene_override_textures: usize,
    pub scene_override_texture_names: usize,
    pub image_nodes: usize,
    pub named_images: usize,
    pub models: usize,
    pub materials: usize,
    pub material_passes: usize,
    pub material_textures: usize,
    pub material_user_textures: usize,
    pub material_combos: usize,
    pub material_constants: usize,
    pub shader_refs: usize,
    pub model_bytes: usize,
    pub solid_models: usize,
    pub passthrough_models: usize,
    pub autosize_models: usize,
    pub no_padding_models: usize,
    pub sized_models: usize,
    pub puppet_models: usize,
    pub render_target_layers: usize,
    pub fullscreen_layers: usize,
    pub effects: usize,
    pub effects_with_ids: usize,
    pub named_effects: usize,
    pub visible_effects: usize,
    pub effect_metadata_bytes: usize,
    pub effect_dependencies: usize,
    pub effect_passes: usize,
    pub effect_binds: usize,
    pub effect_commands: usize,
    pub effect_sources: usize,
    pub effect_targets: usize,
    pub effect_fbos: usize,
    pub scaled_fbos: usize,
    pub unique_fbos: usize,
}

impl SceneGraphStats {
    pub fn summary(&self) -> String {
        format!(
            "{} objects, {} images, {} models, {} materials/{} passes, {} textures, {} effects/{} passes, {} FBOs, {} scene fields",
            self.objects,
            self.image_nodes,
            self.models,
            self.materials,
            self.material_passes,
            self.material_textures + self.material_user_textures,
            self.effects,
            self.effect_passes,
            self.effect_fbos,
            self.camera_fields
                + self.general_fields
                + self.angled_objects
                + self.scene_override_textures
                + self.scene_override_texture_names,
        )
    }
}

fn collect_material_stats(material: &super::material::Material, stats: &mut SceneGraphStats) {
    stats.material_passes += material.passes.len();

    for pass in &material.passes {
        stats.model_bytes += pass.blending.len()
            + pass.cullmode.len()
            + pass.depthtest.len()
            + pass.depthwrite.len();
        stats.shader_refs += usize::from(pass.shader.as_deref().is_some_and(|s| !s.is_empty()));
        stats.material_textures += pass.textures.len();
        stats.material_user_textures += pass.usertextures.len();
        stats.material_combos += pass.combos.len();
        stats.material_constants += pass.constants.len();
    }
}

pub struct ImageNode {
    pub object_index: usize,
    pub id: Option<i64>,
    pub name: String,
    pub image_path: String,
    pub base_texture: Option<String>,
    pub model: Option<Model>,
    pub effects: Vec<ResolvedImageEffect>,
}

impl ImageNode {
    pub fn is_render_target_layer(&self) -> bool {
        let p = self.image_path.to_lowercase();
        p.contains("projectlayer")
            || p.contains("composelayer")
            || p.contains("solid_instance_model")
            || self
                .model
                .as_ref()
                .map(|model| model.solid_layer || model.passthrough)
                .unwrap_or(false)
    }

    pub fn is_fullscreen_layer(&self) -> bool {
        let p = self.image_path.to_lowercase();
        p.contains("fullscreenlayer")
            || self
                .model
                .as_ref()
                .map(|model| model.fullscreen)
                .unwrap_or(false)
    }
}

pub struct ResolvedImageEffect {
    pub id: Option<i64>,
    pub name: Option<String>,
    pub visible: bool,
    pub definition: EffectDefinition,
}
