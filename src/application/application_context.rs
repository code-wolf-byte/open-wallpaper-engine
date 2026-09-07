use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use crate::render::RenderSettings;

/// Runtime configuration for a wallpaper run (the Rust counterpart of the C++
/// `ApplicationContext`). Built from the CLI by `main.rs` and consumed by
/// [`super::WallpaperApplication`].
#[derive(Debug, Clone)]
pub struct ApplicationContext {
    /// The wallpaper to show: a workshop directory, scene directory, or a
    /// plain image/video file. Shown on every output with no more specific
    /// entry in `screens` — the only thing `set`/`set-file` ever populate,
    /// which is how they keep their existing "one wallpaper, everywhere"
    /// behavior unchanged.
    pub background: PathBuf,
    /// Per-named-output overrides (`wp-engine run`'s per-screen assignment —
    /// see `settings::WpSettings::screens`), resolved to real wallpaper
    /// paths ahead of time. Empty for `set`/`set-file`.
    pub screens: HashMap<String, PathBuf>,
    /// `--set-property name=value` overrides, applied to every scene loaded
    /// in this process via the engine property system.
    pub properties: HashMap<String, String>,
    /// Live render settings for `background` (the default) — shared with the
    /// platform layer and, for the GUI, with the live quality/volume sliders.
    pub settings: Arc<Mutex<RenderSettings>>,
    /// Per-screen `RenderSettings`, independently live-adjustable — each one
    /// seeded from that screen's own saved per-wallpaper quality override
    /// (`settings::WpSettings::quality`) at launch, so a heavy scene on one
    /// monitor doesn't force every other monitor down to the same quality.
    /// Empty for `set`/`set-file`, same as `screens`.
    pub screen_settings: HashMap<String, Arc<Mutex<RenderSettings>>>,
}

impl ApplicationContext {
    pub fn new(background: PathBuf) -> Self {
        Self {
            background,
            screens: HashMap::new(),
            properties: HashMap::new(),
            settings: Arc::new(Mutex::new(RenderSettings::default())),
            screen_settings: HashMap::new(),
        }
    }

    /// Add `--set-property` style arguments (`name=value`, bare `name` = "1").
    pub fn add_property_args<I, S>(&mut self, args: I)
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        for arg in args {
            let (name, value) = crate::engine::properties::parse_property_arg(arg.as_ref());
            self.properties.insert(name, value);
        }
    }
}
