use image::RgbaImage;
use std::collections::HashMap;
use std::path::Path;

use crate::engine::tex::TexFile;

pub fn load_utility_textures(assets_dir: &Path) -> HashMap<String, RgbaImage> {
    let mut textures = HashMap::new();
    let util_dir = assets_dir.join("materials").join("util");

    let entries = match std::fs::read_dir(&util_dir) {
        Ok(e) => e,
        Err(_) => return textures,
    };

    for entry in entries.flatten() {
        let path = entry.path();
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        let name = match path.file_stem().and_then(|s| s.to_str()) {
            Some(n) => n.to_string(),
            None => continue,
        };

        let img = match ext {
            "tex" => {
                let data = match std::fs::read(&path) {
                    Ok(d) => d,
                    Err(e) => {
                        tracing::warn!(target: "shader", "reading {}: {e}", path.display());
                        continue;
                    }
                };
                match TexFile::parse(&data).and_then(|t| t.to_rgba()) {
                    Ok(img) => img,
                    Err(e) => {
                        tracing::warn!(target: "shader", "decoding {}: {e}", path.display());
                        continue;
                    }
                }
            }
            "png" | "jpg" | "jpeg" => match image::open(&path) {
                Ok(img) => img.into_rgba8(),
                Err(e) => {
                    tracing::warn!(target: "shader", "loading {}: {e}", path.display());
                    continue;
                }
            },
            _ => continue,
        };

        textures.insert(name, img);
    }

    textures
}

pub fn resolve_texture_ref<'a>(
    name: &str,
    util_textures: &'a HashMap<String, RgbaImage>,
) -> Option<&'a RgbaImage> {
    if let Some(tex) = util_textures.get(name) {
        return Some(tex);
    }
    if let Some(short) = name.rsplit('/').next() {
        if let Some(tex) = util_textures.get(short) {
            return Some(tex);
        }
    }
    if let Some(stripped) = name.strip_prefix("util/") {
        if let Some(tex) = util_textures.get(stripped) {
            return Some(tex);
        }
    }
    None
}
