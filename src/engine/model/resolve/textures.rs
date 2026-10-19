//! Loading what layers show: textures from the wallpaper folder, its
//! scene.pkg or Wallpaper Engine's shared assets, following model, material
//! and texture chains, and embedded-video textures streamed while they play.

use super::*;

/// A loaded texture plus any additional animation frames (e.g. from a `.tex`
/// TEXS spritesheet table or a multi-image container).
pub(super) struct LoadedImage {
    pub(super) image: RgbaImage,
    pub(super) extra_frames: Vec<RgbaImage>,
    pub(super) frame_duration_ms: u32,
    pub(super) no_interpolation: bool,
    pub(super) clamp_uvs: bool,
    /// Present when the model is an animated puppet (mesh + skeleton +
    /// MDLA animations): `image` holds the rest pose, and the live GPU
    /// path re-poses/re-rasterizes from this over time.
    pub(super) puppet: Option<std::sync::Arc<crate::engine::sim::puppet::PuppetRuntime>>,
    /// Present for embedded-video textures: `image` holds the first frame,
    /// and the live GPU path re-uploads frames from this stream over time.
    pub(super) video: Option<std::sync::Arc<std::sync::Mutex<VideoLayerStream>>>,
    /// Set when the model's material declares the `SPRITESHEET` combo and the
    /// `.tex` carries a frame table (e.g. a 2-cell AM/PM atlas): the per-frame
    /// sub-rect images. `layer_from_object` picks the single frame the object's
    /// frame script selects and uses it as the layer image. Without this the
    /// flattened atlas stretches into the one-cell quad (all cells overlaid).
    pub(super) sprite_frames: Vec<RgbaImage>,
}

impl LoadedImage {
    pub(super) fn single(image: RgbaImage) -> Self {
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
    pub(super) fn from_tex(tex: &TexFile) -> Result<Self> {
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

/// Scripts copied from a Workshop asset pack name assets relative to that
/// pack (`models/bar.json`), which ships at `models/workshop/<id>/bar.json`.
/// Returns `path` itself when it exists or no such copy is found.
pub(super) fn resolve_workshop_asset(dir: &Path, pkg: Option<&Package>, path: &str) -> String {
    let exists = |p: &str| pkg.is_some_and(|k| k.contains(p)) || dir.join(p).is_file();
    if exists(path) {
        return path.to_string();
    }
    let Some((first, rest)) = path.split_once('/') else {
        return path.to_string();
    };
    let prefix = format!("{first}/workshop/");
    let suffix = format!("/{rest}");
    let matches = |p: &str| {
        p.strip_prefix(&prefix)
            .and_then(|r| r.strip_suffix(&suffix))
            .is_some_and(|id| !id.is_empty() && id.chars().all(|c| c.is_ascii_digit()))
    };
    if let Some(found) = pkg.and_then(|k| k.paths().find(|p| matches(p)).map(str::to_string)) {
        return found;
    }
    if let Ok(entries) = std::fs::read_dir(dir.join(first).join("workshop")) {
        for e in entries.flatten() {
            let candidate = format!("{prefix}{}{suffix}", e.file_name().to_string_lossy());
            if matches(&candidate) && dir.join(&candidate).is_file() {
                return candidate;
            }
        }
    }
    path.to_string()
}

pub(super) fn is_video_path(p: &str) -> bool {
    let p = p.to_lowercase();
    p.ends_with(".mp4") || p.ends_with(".webm") || p.ends_with(".mkv") || p.ends_with(".avi")
}

/// Resolve a particle preset's JSON text, checking the wallpaper's own
/// directory, then its `scene.pkg` (if given), then the global Steam assets
/// dir — mirroring the same wallpaper-first priority used for shaders/effects
/// (`shaders::resolver::AssetResolver`).
/// Read a scene asset from wherever it lives: loose files, the pkg archive,
/// then the shared WE assets dir.
pub(super) fn read_asset_bytes(
    dir: Option<&Path>,
    pkg: Option<&Package>,
    rel_path: &str,
) -> Option<Vec<u8>> {
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

/// A live decode stream for an embedded-video texture layer: a detached
/// ffmpeg thread loops the video (PTS-paced, same `video_decode_loop` the
/// full-video wallpaper path uses) and feeds frames through a small bounded
/// channel. The consumer drains to the newest frame each render tick, so
/// slow consumers drop frames instead of lagging. The thread exits on its
/// own once this struct (the receiver) is dropped, and the temp file
/// backing the looping decoder is removed then too.
pub struct VideoLayerStream {
    pub(super) rx: std::sync::mpsc::Receiver<std::sync::Arc<RgbaImage>>,
    /// Temp file the decoder re-opens on every loop — must outlive playback.
    pub(super) path: std::path::PathBuf,
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

/// Start looping playback for an embedded video payload: writes the bytes
/// to a temp file, decodes the first frame synchronously (so the layer has
/// correct content immediately), and spawns the paced decode loop for the
/// rest.
pub(super) fn start_video_stream(bytes: &[u8]) -> Result<(RgbaImage, VideoLayerStream)> {
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let tmp = std::env::temp_dir().join(format!(
        "we_embvid_{}_{}.mp4",
        std::process::id(),
        SEQ.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::write(&tmp, bytes)
        .with_context(|| format!("writing embedded video temp file: {}", tmp.display()))?;
    let first = match crate::content::video::decode_first_frame(&tmp) {
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
        if let Err(e) = crate::content::video::video_decode_loop(&decode_path, &tx) {
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
pub(super) fn video_tex_to_loaded(tex: &TexFile, bytes: &[u8]) -> Result<LoadedImage> {
    let (first, stream) = start_video_stream(bytes)?;
    let mut loaded = LoadedImage::single(first);
    loaded.no_interpolation = tex.no_interpolation();
    loaded.clamp_uvs = tex.clamp_uvs();
    loaded.video = Some(std::sync::Arc::new(std::sync::Mutex::new(stream)));
    Ok(loaded)
}

pub(super) fn load_texture_from_dir(dir: &Path, image_path: &str) -> Result<LoadedImage> {
    // If it's a .json reference, resolve the model/material chain
    if image_path.ends_with(".json") {
        return resolve_model_chain_dir(dir, image_path);
    }

    let full_path = dir.join(image_path);

    // Video file — extract first frame as static thumbnail
    if is_video_path(image_path) && full_path.exists() {
        return crate::content::video::decode_first_frame(&full_path).map(LoadedImage::single);
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

pub(super) fn load_texture_from_pkg(pkg: &Package, image_path: &str) -> Result<LoadedImage> {
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
            let result = crate::content::video::decode_first_frame(&tmp);
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
/// `GpuSceneInstance::build` (gpu::renderer) to load a spot light's
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
pub(super) fn read_from_global_assets(rel_path: &str) -> Option<Vec<u8>> {
    let dir = crate::engine::gpu::shaders::loader::find_we_assets_dir()?;
    std::fs::read(dir.join(rel_path)).ok()
}

/// Returns the resolved texture plus the material pass's own `"blending"`
/// string (e.g. `"additive"`) and `ui_editor_properties_overbright`
/// brightness constant (default 1.0) — particles need both to composite
/// correctly (see `resolve_particle_sprite_{pkg,dir}`); plain image layers
/// just discard them.
pub(super) fn find_model_chain_tex_pkg(
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
pub(super) fn pass_overbright(pass: &serde_json::Value) -> f32 {
    pass.get("constantshadervalues")
        .and_then(|c| c.get("ui_editor_properties_overbright"))
        .and_then(|v| v.as_f64())
        .map(|v| v as f32)
        .unwrap_or(1.0)
}

/// A mesh material's first pass. Meshes need two things out of it that the
/// texture chain doesn't carry: the flat `color` constant and `cullmode`.
pub(super) fn material_first_pass(
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
pub(super) fn pass_constant_color(pass: &serde_json::Value) -> Option<[u8; 3]> {
    let consts = pass.get("constantshadervalues")?;
    // WE keys this as `color`; `Color` is a separate (usually white) tint.
    let s = consts
        .get("color")
        .or_else(|| consts.get("Color"))?
        .as_str()?;
    let c = crate::engine::cpu::effect::parse_color(s);
    Some([
        (c[0].clamp(0.0, 1.0) * 255.0) as u8,
        (c[1].clamp(0.0, 1.0) * 255.0) as u8,
        (c[2].clamp(0.0, 1.0) * 255.0) as u8,
    ])
}

/// Follow the model -> material -> texture chain for loose files on disk.
pub(super) fn find_model_chain_tex_dir(
    dir: &Path,
    json_path: &str,
) -> Result<(TexFile, Option<String>, f32)> {
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
pub(super) fn chain_has_spritesheet(
    json_path: &str,
    read: &dyn Fn(&str) -> Option<Vec<u8>>,
) -> bool {
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
pub(super) fn loaded_from_chain_tex(
    tex: &TexFile,
    json_path: &str,
    read: &dyn Fn(&str) -> Option<Vec<u8>>,
) -> Result<LoadedImage> {
    if tex.is_animated() {
        return LoadedImage::from_tex(tex);
    }
    Ok(apply_puppet_mesh(tex.to_rgba()?, json_path, read))
}

pub(super) fn resolve_model_chain_pkg(pkg: &Package, json_path: &str) -> Result<LoadedImage> {
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

pub(super) fn resolve_model_chain_dir(dir: &Path, json_path: &str) -> Result<LoadedImage> {
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
