//! Showing wallpapers: `set`, `set-file` and `run`, and the saved settings they
//! apply.

use super::*;

pub(super) fn cmd_set(
    id: &str,
    properties: Vec<String>,
    quality: Option<String>,
    save: bool,
    foreground: bool,
) -> Result<()> {
    let w = workshop::find_by_id(id).ok_or_else(|| anyhow!("workshop item '{id}' not found"))?;

    println!("Loading: {}", w.path.display());
    println!("Applying \"{}\" to all outputs...", w.title());

    let mut context = ApplicationContext::new(w.path.clone());
    apply_saved_settings(&mut context, id, &properties, quality.as_deref(), save)?;
    launch(context, Some(w.title().to_string()), foreground)
}

/// Installs property overrides and a quality level on `context` in priority
/// order (saved < CLI flags — `--set-property`/`--quality` — later write
/// wins), and — when `save` is set and there's anything new to save —
/// persists the CLI-provided values under `id_or_path` (via `settings::
/// normalize_key`) for next time.
pub(super) fn apply_saved_settings(
    context: &mut ApplicationContext,
    id_or_path: &str,
    cli_properties: &[String],
    cli_quality: Option<&str>,
    save: bool,
) -> Result<()> {
    let mut settings = wp_engine::settings::WpSettings::load();
    let saved = settings.property_overrides(id_or_path);
    if !saved.is_empty() {
        println!(
            "Loaded {} saved property override(s) from {}",
            saved.len(),
            wp_engine::settings::settings_path().display()
        );
    }
    context.properties.extend(saved);
    context.add_property_args(cli_properties);

    let saved_quality = settings.quality_override(id_or_path);
    let cli_quality_parsed = cli_quality
        .map(|s| {
            wp_engine::platform::RenderQuality::parse(s).ok_or_else(|| {
                anyhow!("invalid --quality '{s}' (expected Ultra, High, Medium, or Low)")
            })
        })
        .transpose()?;
    if let Some(q) = cli_quality_parsed.or(saved_quality) {
        if cli_quality_parsed.is_none() {
            println!("Loaded saved quality: {}", q.label());
        }
        context.settings.lock().unwrap().quality = q;
    }

    if save {
        let mut saved_anything = false;
        for arg in cli_properties {
            let (name, value) = workshop::properties::parse_property_arg(arg);
            settings.set_property(id_or_path, &name, &value);
            saved_anything = true;
        }
        if !cli_properties.is_empty() {
            println!(
                "Saved {} property override(s) for next time",
                cli_properties.len()
            );
        }
        if let Some(q) = cli_quality_parsed {
            settings.set_quality(id_or_path, q);
            println!("Saved quality override for next time: {}", q.label());
            saved_anything = true;
        }
        if saved_anything {
            settings.save()?;
        }
    }
    Ok(())
}

/// Runs using persisted settings (`wp-engine config` to set them up first):
/// every `screens` entry gets its own independent wallpaper *and* its own
/// independent render quality on its own output (real per-screen rendering
/// — see `content::ScreenContent`/`content::ScreenSettings`); any output with
/// no specific entry falls back to the first configured screen's wallpaper,
/// else `default_background`, else the first playlist's first item.
///
/// Honest scope note: property overrides are still process-global (see
/// `workshop::properties`), so only the *default* background's saved
/// overrides are applied — a per-screen wallpaper's own saved overrides
/// aren't isolated from each other yet if two different screens' wallpapers
/// happen to declare a property with the same name. With an active
/// playlist (`config set-active-playlist`), the daemon plays that instead.
pub(super) fn cmd_run(foreground: bool) -> Result<()> {
    let settings = wp_engine::settings::WpSettings::load();

    #[cfg(target_os = "linux")]
    if let (Some(name), false) = (&settings.active_playlist, foreground) {
        cmd_playlist(PlaylistAction::Play { name: name.clone() })?;
        println!("Playing playlist '{name}' (background daemon).");
        return Ok(());
    }

    let default_id_or_path = settings
        .screens
        .values()
        .next()
        .or(settings.default_background.as_ref())
        .or_else(|| {
            settings
                .playlists
                .values()
                .next()
                .and_then(|p| p.items.first())
        })
        .ok_or_else(|| {
            anyhow!(
                "no saved wallpaper — use `wp-engine config set-screen`/`set-default`/\
                 `set-playlist` first, or `wp-engine set <id>` directly"
            )
        })?
        .clone();
    let default_path = resolve_id_or_path(&default_id_or_path)?;

    println!("Default background: {default_id_or_path}");
    let mut context = ApplicationContext::new(default_path);
    for (screen, id_or_path) in &settings.screens {
        match resolve_id_or_path(id_or_path) {
            Ok(path) => {
                print!("Screen '{screen}': {id_or_path}");
                context.screens.insert(screen.clone(), path);
                // Independent per-screen quality — looked up by the
                // *original* id-or-path string (not the resolved path
                // above): that's the same key `settings::normalize_key`
                // would derive from it, and using the resolved path here
                // instead would silently miss every workshop-ID-keyed
                // saved override (a real bug this comment exists to head
                // off — `normalize_key("2821407073")` and
                // `normalize_key("/path/to/2821407073")` are NOT the same
                // key).
                if let Some(q) = settings.quality_override(id_or_path) {
                    println!(" (quality: {})", q.label());
                    context.screen_settings.insert(
                        screen.clone(),
                        Arc::new(Mutex::new(wp_engine::content::RenderSettings {
                            quality: q,
                            fps: context.settings.lock().unwrap().fps,
                            ..Default::default()
                        })),
                    );
                } else {
                    println!();
                }
            }
            Err(e) => println!("Screen '{screen}': {id_or_path} — {e} (skipping)"),
        }
    }

    apply_saved_settings(&mut context, &default_id_or_path, &[], None, false)?;
    launch(context, Some(default_id_or_path), foreground)
}

/// Show `context`'s wallpaper: by default hand it to the single background
/// daemon (starting it if needed) and return, replacing whatever it was
/// showing; with `foreground`, render in this process until Ctrl-C.
///
/// macOS always renders in the foreground — its backend needs the process
/// main thread, and its GUI launches wallpapers as `set-file` children.
pub(super) fn launch(
    context: ApplicationContext,
    title: Option<String>,
    foreground: bool,
) -> Result<()> {
    #[cfg(target_os = "linux")]
    {
        use wp_engine::daemon::{self, ApplySpec, Request};
        if !foreground {
            daemon::request_or_spawn(&Request::Apply(ApplySpec::from_context(&context, title)))?;
            println!("Wallpaper active (background daemon). `wp-engine stop` removes it.");
            return Ok(());
        }
        if daemon::is_running() {
            eprintln!(
                "warning: a wp-engine daemon is also running — both will draw the wallpaper \
                 (`wp-engine stop` stops it)"
            );
        }
    }
    let _ = (&title, foreground);
    let mut app = WallpaperApplication::new(context);
    app.setup()?;
    println!("Wallpaper active. Press Ctrl-C to exit.");
    app.show()
}

pub(super) fn cmd_set_file(
    path: &PathBuf,
    properties: Vec<String>,
    quality: Option<String>,
    save: bool,
    foreground: bool,
) -> Result<()> {
    if !path.exists() {
        return Err(anyhow!("file not found: {}", path.display()));
    }
    println!("Loading: {}", path.display());
    println!("Applying to all outputs…");

    let mut context = ApplicationContext::new(path.clone());
    let key = path.display().to_string();
    apply_saved_settings(&mut context, &key, &properties, quality.as_deref(), save)?;
    let title = path.file_name().map(|n| n.to_string_lossy().into_owned());
    launch(context, title, foreground)
}
