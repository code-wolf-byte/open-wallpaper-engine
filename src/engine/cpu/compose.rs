//! The CPU compositor: draws a resolved scene's image and particle layers
//! into one RGBA image in scene order. Used by `render-scene` and the CPU
//! fallback path; live rendering is `engine::gpu::renderer`.

use image::RgbaImage;

use crate::engine::model::resolve::{Layer, ResolvedScene};
use crate::engine::sim::particle;

impl ResolvedScene {
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

    pub(super) fn draw_image_layer(&self, layer: &Layer, canvas: &mut RgbaImage) {
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
