//! Asset lookup priority for effects/materials/shaders/textures, mirroring
//! the reference's `Container` mount order in
//! `WallpaperApplication::setupAssetLocator`:
//!   1. the wallpaper's own directory (loose files)
//!   2. the wallpaper's `scene.pkg` / `gifscene.pkg`
//!   3. the global Wallpaper Engine `assets` directory (Steam install)
//!
//! Custom-effect wallpapers bundle their own `effects/…/effect.json`,
//! materials and shaders under (1) or (2) that should shadow anything with
//! the same relative path under (3).

use super::loader;
use crate::engine::pkg::Package;
use std::path::{Path, PathBuf};

pub struct AssetResolver {
    wallpaper_dir: Option<PathBuf>,
    wallpaper_pkgs: Vec<Package>,
    assets_dir: Option<PathBuf>,
}

impl AssetResolver {
    /// Build a resolver for one wallpaper. `wallpaper_dir` is the directory
    /// passed on the command line / from the workshop item; `assets_dir` is
    /// the global Steam `wallpaper_engine/assets` fallback.
    pub fn new(wallpaper_dir: Option<&Path>, assets_dir: Option<PathBuf>) -> Self {
        let mut wallpaper_pkgs = Vec::new();
        if let Some(dir) = wallpaper_dir {
            for name in ["scene.pkg", "gifscene.pkg"] {
                let path = dir.join(name);
                if path.exists() {
                    if let Ok(pkg) = Package::from_file(&path) {
                        wallpaper_pkgs.push(pkg);
                    }
                }
            }
        }
        Self {
            wallpaper_dir: wallpaper_dir.map(Path::to_path_buf),
            wallpaper_pkgs,
            assets_dir,
        }
    }

    /// Read a file by its path relative to an asset root (e.g.
    /// `"effects/pulse/effect.json"`, `"shaders/genericimage3.frag"`),
    /// checking wallpaper-local sources before the global assets dir.
    pub fn read(&self, rel_path: &str) -> Option<Vec<u8>> {
        if let Some(dir) = &self.wallpaper_dir {
            if let Ok(data) = std::fs::read(dir.join(rel_path)) {
                return Some(data);
            }
        }
        for pkg in &self.wallpaper_pkgs {
            if let Some(data) = pkg.get(rel_path) {
                return Some(data.to_vec());
            }
        }
        if let Some(dir) = &self.assets_dir {
            if let Ok(data) = std::fs::read(dir.join(rel_path)) {
                return Some(data);
            }
        }
        None
    }

    pub fn read_string(&self, rel_path: &str) -> Option<String> {
        self.read(rel_path).and_then(|b| String::from_utf8(b).ok())
    }

    /// GLSL `#include` search path candidates for a shader living at
    /// `dir_prefix` (e.g. `"effects/{name}/shaders/"` or `"shaders/"`),
    /// tried in the same wallpaper-first priority order.
    pub fn read_include(&self, dir_prefixes: &[&str], include_file: &str) -> Option<String> {
        for prefix in dir_prefixes {
            let rel = format!("{prefix}{include_file}");
            if let Some(s) = self.read_string(&rel) {
                return Some(s);
            }
        }
        None
    }

    pub fn load_glsl_shader_for_effect(
        &self,
        shader_name: &str,
        effect_dir: Option<&str>,
    ) -> anyhow::Result<(String, String)> {
        loader::load_glsl_shader_with_resolver(self, shader_name, effect_dir)
    }
}
