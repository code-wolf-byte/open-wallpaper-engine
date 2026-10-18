use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::{Path, PathBuf};

/// Wallpaper Engine Steam App ID
pub const WALLPAPER_ENGINE_APP_ID: &str = "431960";

/// Wallpaper type as reported in project.json.
/// Wallpaper Engine uses mixed capitalisation across versions, so we alias both.
#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum WallpaperType {
    #[serde(alias = "Video")]
    Video,
    #[serde(alias = "Scene")]
    Scene,
    #[serde(alias = "Web")]
    Web,
    #[serde(alias = "Application")]
    Application,
    /// A preset: a property set (plus its own `files/`) applied on top of
    /// another workshop item named by `dependency`. Never spelled in
    /// `project.json` — derived by [`Wallpaper::wallpaper_type`].
    #[serde(skip)]
    Preset,
    #[serde(other)]
    Unknown,
}

impl std::fmt::Display for WallpaperType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WallpaperType::Video => write!(f, "video"),
            WallpaperType::Scene => write!(f, "scene"),
            WallpaperType::Web => write!(f, "web"),
            WallpaperType::Application => write!(f, "application"),
            WallpaperType::Preset => write!(f, "preset"),
            WallpaperType::Unknown => write!(f, "unknown"),
        }
    }
}

/// Parsed project.json from a Wallpaper Engine Workshop item
#[derive(Debug, Clone, Deserialize)]
pub struct Project {
    pub title: Option<String>,
    pub file: Option<String>,
    pub preview: Option<String>,
    #[serde(rename = "type")]
    pub wallpaper_type: Option<WallpaperType>,
    pub tags: Option<Vec<String>>,
    pub description: Option<String>,
    pub contentrating: Option<String>,
    /// Workshop ID of the item this one builds on (presets, and items that
    /// reuse another item's assets).
    pub dependency: Option<String>,
    /// A preset's property values, applied over the dependency's defaults.
    pub preset: Option<serde_json::Map<String, serde_json::Value>>,
}

/// A resolved Workshop wallpaper entry (project.json + its directory)
#[derive(Debug, Clone)]
pub struct Wallpaper {
    /// Steam Workshop item ID (directory name)
    pub workshop_id: String,
    /// Root directory of this wallpaper
    pub path: PathBuf,
    pub project: Project,
}

impl Wallpaper {
    /// Absolute path to the main wallpaper file (video, image, html, etc.)
    pub fn wallpaper_file(&self) -> Option<PathBuf> {
        self.project.file.as_ref().map(|f| self.path.join(f))
    }

    /// Absolute path to the preview image
    pub fn preview_file(&self) -> Option<PathBuf> {
        self.project.preview.as_ref().map(|f| self.path.join(f))
    }

    pub fn title(&self) -> &str {
        self.project.title.as_deref().unwrap_or("<untitled>")
    }

    pub fn wallpaper_type(&self) -> &WallpaperType {
        match &self.project.wallpaper_type {
            Some(t) => t,
            None if self.project.dependency.is_some() && self.project.preset.is_some() => {
                &WallpaperType::Preset
            }
            None => &WallpaperType::Unknown,
        }
    }
}

/// Steam install roots to look for, covering native, `~/.steam` symlink,
/// Flatpak, Snap, and macOS installs.
fn steam_install_candidates() -> Vec<PathBuf> {
    let Some(home) = dirs::home_dir() else {
        return Vec::new();
    };
    vec![
        home.join(".local/share/Steam"),
        home.join(".steam/steam"),
        home.join(".var/app/com.valvesoftware.Steam/.local/share/Steam"),
        home.join(".var/app/com.valvesoftware.Steam/data/Steam"),
        home.join("snap/steam/common/.local/share/Steam"),
        home.join("Library/Application Support/Steam"),
    ]
}

/// Every `"path"` value in a `libraryfolders.vdf` — the libraries Steam has
/// on other drives. A line-level scan is enough: the file is Valve KeyValues
/// with one `"key" "value"` pair per line, and `path` only appears as a
/// library's location.
fn parse_library_folders(vdf: &str) -> Vec<PathBuf> {
    vdf.lines()
        .filter_map(|line| {
            let mut quoted = line.split('"').skip(1).step_by(2);
            match (quoted.next(), quoted.next()) {
                (Some("path"), Some(value)) => Some(PathBuf::from(value.replace("\\\\", "\\"))),
                _ => None,
            }
        })
        .collect()
}

/// Locate Steam library roots: every install found on disk, plus every
/// library its `steamapps/libraryfolders.vdf` lists.
fn find_steam_library_roots() -> Vec<PathBuf> {
    let mut roots: Vec<PathBuf> = Vec::new();
    let push = |p: PathBuf, roots: &mut Vec<PathBuf>| {
        if !p.exists() {
            return;
        }
        // Resolve symlinks so ~/.steam/steam (→ ~/.local/share/Steam) isn't scanned twice
        let real = std::fs::canonicalize(&p).unwrap_or(p);
        if !roots.contains(&real) {
            roots.push(real);
        }
    };

    for install in steam_install_candidates() {
        push(install.clone(), &mut roots);
        if let Ok(vdf) = std::fs::read_to_string(install.join("steamapps/libraryfolders.vdf")) {
            for lib in parse_library_folders(&vdf) {
                push(lib, &mut roots);
            }
        }
    }

    roots
}

/// Return all Workshop content directories for Wallpaper Engine across all Steam libraries
pub fn find_workshop_dirs() -> Vec<PathBuf> {
    find_steam_library_roots()
        .into_iter()
        .map(|root| {
            root.join("steamapps")
                .join("workshop")
                .join("content")
                .join(WALLPAPER_ENGINE_APP_ID)
        })
        .filter(|p| p.exists())
        .collect()
}

/// Find the directory of the workshop item `id`, looking beside `near`
/// first (dependencies normally live in the same workshop folder), then in
/// every known library.
pub fn find_item_dir(id: &str, near: Option<&Path>) -> Option<PathBuf> {
    if let Some(sibling) = near.and_then(|n| n.parent()).map(|p| p.join(id)) {
        if sibling.join("project.json").exists() {
            return Some(sibling);
        }
    }
    find_workshop_dirs()
        .into_iter()
        .map(|d| d.join(id))
        .find(|d| d.join("project.json").exists())
}

/// Load and parse a project.json inside a wallpaper directory
pub fn load_project(wallpaper_dir: &Path) -> Result<Project> {
    let project_path = wallpaper_dir.join("project.json");
    let data = std::fs::read_to_string(&project_path)
        .with_context(|| format!("reading {}", project_path.display()))?;
    serde_json::from_str(&data).with_context(|| format!("parsing {}", project_path.display()))
}

/// Scan all workshop directories and return every parseable wallpaper
#[tracing::instrument(target = "workshop", level = "debug")]
pub fn scan_wallpapers() -> Vec<Wallpaper> {
    let workshop_dirs = find_workshop_dirs();
    tracing::debug!(target: "workshop", dirs = workshop_dirs.len(), "scanning workshop roots");
    let mut wallpapers = Vec::new();

    for workshop_dir in workshop_dirs {
        tracing::trace!(target: "workshop", dir = %workshop_dir.display(), "reading workshop directory");
        let entries = match std::fs::read_dir(&workshop_dir) {
            Ok(e) => e,
            Err(err) => {
                tracing::warn!(target: "workshop", "cannot read {}: {}", workshop_dir.display(), err);
                continue;
            }
        };

        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }

            let workshop_id = path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string();

            match load_project(&path) {
                Ok(project) => {
                    tracing::trace!(target: "workshop", id = %workshop_id, title = project.title.as_deref().unwrap_or(""), "loaded wallpaper");
                    wallpapers.push(Wallpaper {
                        workshop_id,
                        path,
                        project,
                    });
                }
                Err(err) => {
                    tracing::warn!(target: "workshop", "skipping {}: {}", workshop_id, err);
                }
            }
        }
    }

    // Sort by title for consistent output
    wallpapers.sort_by(|a, b| a.title().cmp(b.title()));
    tracing::debug!(target: "workshop", count = wallpapers.len(), "workshop scan complete");
    wallpapers
}

/// Find a single wallpaper by its Workshop ID
pub fn find_by_id(id: &str) -> Option<Wallpaper> {
    scan_wallpapers().into_iter().find(|w| w.workshop_id == id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn library_folders_lists_every_path() {
        let vdf = r#""libraryfolders"
{
	"0"
	{
		"path"		"/home/u/.local/share/Steam"
		"label"		""
	}
	"1"
	{
		"path"		"/mnt/games/SteamLibrary"
		"apps" { "431960" "1" }
	}
}"#;
        assert_eq!(
            parse_library_folders(vdf),
            vec![
                PathBuf::from("/home/u/.local/share/Steam"),
                PathBuf::from("/mnt/games/SteamLibrary")
            ]
        );
    }

    #[test]
    fn preset_type_is_derived_from_dependency_and_preset() {
        let project: Project =
            serde_json::from_str(r#"{"title":"p","dependency":"893418273","preset":{"a":1}}"#)
                .unwrap();
        let w = Wallpaper {
            workshop_id: "1".into(),
            path: PathBuf::from("/x/1"),
            project,
        };
        assert_eq!(w.wallpaper_type(), &WallpaperType::Preset);
    }
}
