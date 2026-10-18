//! Named GPU render targets (mirrors CFBO from linux-wallpaperengine).
//! Used for per-object ping-pong effect chaining and named scene framebuffers.

use std::collections::HashMap;

/// A named GPU render target.
pub struct RenderTarget {
    pub name: String,
    pub texture: wgpu::Texture,
    pub width: u32,
    pub height: u32,
}

/// Shrink `w`×`h` (keeping its aspect) until both fit the device's texture
/// limit. Oversized layers are real — 2412448157 ships an 11424px strip and a
/// GPU that caps at 8192 treats the texture as a fatal validation error, not a
/// recoverable one — so every texture creation clamps through here.
pub fn fit_texture_limit(device: &wgpu::Device, w: u32, h: u32) -> (u32, u32) {
    let max = device.limits().max_texture_dimension_2d;
    if w <= max && h <= max {
        return (w.max(1), h.max(1));
    }
    let s = max as f32 / w.max(h) as f32;
    let fit = |v: u32| ((v as f32 * s) as u32).clamp(1, max);
    (fit(w), fit(h))
}

impl RenderTarget {
    pub const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

    /// Create a new render target with RENDER_ATTACHMENT | TEXTURE_BINDING | COPY_SRC | COPY_DST.
    pub fn new(device: &wgpu::Device, name: impl Into<String>, w: u32, h: u32) -> Self {
        let name = name.into();
        let (w, h) = fit_texture_limit(device, w, h);
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(&name),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: Self::FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        Self {
            name,
            texture,
            width: w,
            height: h,
        }
    }

    pub fn view(&self) -> wgpu::TextureView {
        self.texture
            .create_view(&wgpu::TextureViewDescriptor::default())
    }

    /// Clear the render target immediately (creates and submits its own encoder).
    pub fn clear(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        r: f64,
        g: f64,
        b: f64,
        a: f64,
    ) {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("fbo_clear"),
        });
        {
            let view = self.view();
            let _ = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("fbo_clear_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r, g, b, a }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
        }
        queue.submit(std::iter::once(encoder.finish()));
    }
}

/// Pool of named render targets for a scene.
#[derive(Default)]
pub struct RenderTargetPool {
    targets: HashMap<String, RenderTarget>,
}

impl RenderTargetPool {
    pub fn new() -> Self {
        Self::default()
    }

    /// Get or create a named target at the given dimensions.
    pub fn get_or_create(
        &mut self,
        device: &wgpu::Device,
        name: &str,
        w: u32,
        h: u32,
    ) -> &RenderTarget {
        self.targets
            .entry(name.to_string())
            .or_insert_with(|| RenderTarget::new(device, name, w, h))
    }

    pub fn get(&self, name: &str) -> Option<&RenderTarget> {
        self.targets.get(name)
    }
    pub fn get_mut(&mut self, name: &str) -> Option<&mut RenderTarget> {
        self.targets.get_mut(name)
    }
    pub fn iter(&self) -> impl Iterator<Item = (&String, &RenderTarget)> {
        self.targets.iter()
    }
}
