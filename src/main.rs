use anyhow::{anyhow, Context, Result};
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use wp_engine::application::{ApplicationContext, WallpaperApplication};
use wp_engine::workshop::{self, Wallpaper};
use wp_engine::{engine, platform, ui};

#[derive(Parser)]
#[command(
    name = "wp-engine",
    about = "Wallpaper Engine client for Wayland",
    long_about = "Browse and apply Steam Workshop wallpapers on Wayland, X11, and macOS.\n\
                  Run without arguments to open the graphical interface."
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,

    /// Increase logging verbosity: -v (debug), -vv (trace), -vvv (+ deps).
    /// `RUST_LOG` overrides this if set.
    #[arg(
        short,
        long,
        action = clap::ArgAction::Count,
        global = true,
    )]
    verbose: u8,
}

#[derive(Subcommand)]
enum Command {
    /// List available Workshop wallpapers (CLI)
    List {
        #[arg(long, help = "Filter by type: video, scene, web, application")]
        r#type: Option<String>,
    },
    /// Apply a wallpaper by Steam Workshop ID (via the background daemon)
    Set {
        id: String,
        /// Override a user property: NAME=VALUE (repeatable; bare NAME = true)
        #[arg(long = "set-property", value_name = "NAME=VALUE")]
        properties: Vec<String>,
        /// Render quality for this run: Ultra, High, Medium, or Low
        #[arg(long)]
        quality: Option<String>,
        /// Persist these --set-property/--quality overrides for next time (see `config`)
        #[arg(long)]
        save: bool,
        /// Render in this process until Ctrl-C instead of handing the
        /// wallpaper to the background daemon
        #[arg(long)]
        foreground: bool,
    },
    /// Apply a scene directory, video, image, or HTML file (via the background daemon)
    SetFile {
        path: PathBuf,
        /// Override a user property: NAME=VALUE (repeatable; bare NAME = true)
        #[arg(long = "set-property", value_name = "NAME=VALUE")]
        properties: Vec<String>,
        /// Render quality for this run: Ultra, High, Medium, or Low
        #[arg(long)]
        quality: Option<String>,
        /// Persist these --set-property/--quality overrides for next time (see `config`)
        #[arg(long)]
        save: bool,
        /// Render in this process until Ctrl-C instead of handing the
        /// wallpaper to the background daemon
        #[arg(long)]
        foreground: bool,
    },
    /// List the user-configurable properties of a Workshop item
    ListProperties { id: String },
    /// Show metadata for a Workshop item (CLI)
    Info { id: String },
    /// List available GPU adapters (PAL probe)
    Probe,
    /// Inspect (and optionally extract) a Wallpaper Engine PKG archive
    PkgInfo {
        path: PathBuf,
        /// Dump all contained files to this directory
        #[arg(long)]
        dump: Option<PathBuf>,
    },
    /// Inspect a Wallpaper Engine .tex texture file
    TexInfo {
        path: PathBuf,
        /// Save decoded texture as PNG
        #[arg(long)]
        save: Option<PathBuf>,
    },
    /// Render a scene wallpaper to a PNG file (for debugging)
    RenderScene {
        /// Workshop ID or directory path
        id_or_path: String,
        /// Output PNG path
        #[arg(long, default_value = "scene_output.png")]
        output: PathBuf,
        /// Override a user property: NAME=VALUE (repeatable; bare NAME = true)
        #[arg(long = "set-property", value_name = "NAME=VALUE")]
        properties: Vec<String>,
    },
    /// Preview an animated scene wallpaper in a window (for testing animation)
    PreviewScene {
        /// Workshop ID or directory path
        id_or_path: String,
        /// Window width
        #[arg(long, default_value_t = 960)]
        width: u32,
        /// Window height
        #[arg(long, default_value_t = 540)]
        height: u32,
    },
    /// Test whether a scene wallpaper animates (headless — no window required)
    TestScene {
        /// Workshop ID or directory path
        id_or_path: String,
        /// Number of frames to collect (at 30fps, 60 = 2 seconds)
        #[arg(long, default_value_t = 60)]
        frames: usize,
    },
    /// Manage persisted settings: per-wallpaper property overrides, screen
    /// assignments, and playlists (~/.config/wp-engine/settings.json)
    Config {
        #[command(subcommand)]
        action: ConfigAction,
    },
    /// Run using persisted settings — see `config` to set them up first
    Run {
        /// Render in this process until Ctrl-C instead of handing the
        /// wallpaper to the background daemon
        #[arg(long)]
        foreground: bool,
    },
    /// Run the single-instance wallpaper daemon in the foreground (it is
    /// started automatically in the background when needed; run it directly
    /// for systemd/autostart or to watch its log)
    Daemon,
    /// Show what the background daemon is displaying
    Status,
    /// Remove the wallpaper and stop the background daemon
    Stop,
}

#[derive(Subcommand)]
enum ConfigAction {
    /// Print the full persisted settings file
    Show,
    /// Persist a property override for a wallpaper without launching it
    SetProperty {
        id: String,
        /// NAME=VALUE (repeatable; bare NAME = true)
        #[arg(required = true, value_name = "NAME=VALUE")]
        properties: Vec<String>,
    },
    /// Remove a persisted property override
    UnsetProperty { id: String, name: String },
    /// Assign a wallpaper to a named screen/output
    SetScreen { screen: String, id: String },
    /// Remove a screen's saved wallpaper assignment
    UnsetScreen { screen: String },
    /// Set the fallback wallpaper `wp-engine run` uses with no screen assigned
    SetDefault { id: String },
    /// Persist a render-quality override for a wallpaper: Ultra, High, Medium, or Low
    SetQuality { id: String, quality: String },
    /// Remove a persisted render-quality override
    UnsetQuality { id: String },
    /// Define or replace a playlist
    SetPlaylist {
        name: String,
        /// Workshop IDs or paths, in playback order
        #[arg(required = true)]
        items: Vec<String>,
        #[arg(long, default_value_t = 60)]
        delay: u32,
        #[arg(long, default_value = "timer")]
        mode: String,
        #[arg(long, default_value = "sequential")]
        order: String,
    },
    /// Remove a playlist
    RemovePlaylist { name: String },
}

fn main() {
    // CEF launches its renderer/GPU/utility processes by re-executing THIS
    // binary with `--type=...`. That must be handled before anything else —
    // before clap, which would reject the flag, and before any thread starts.
    // Returns `Some` only in a child process, which must then exit at once.
    if let Some(code) = wp_engine::render::web::subprocess_main() {
        std::process::exit(code);
    }

    let cli = Cli::parse();
    wp_engine::logging::init(cli.verbose);
    #[cfg(target_os = "linux")]
    wp_engine::daemon::set_spawn_verbosity(cli.verbose);
    tracing::debug!(target: "cli", verbosity = cli.verbose, "wp-engine starting");

    let result = match cli.command {
        // No subcommand → open GUI
        None => run_ui(),
        Some(Command::List { r#type }) => cmd_list(r#type),
        Some(Command::Set { id, properties, quality, save, foreground }) => {
            cmd_set(&id, properties, quality, save, foreground)
        }
        Some(Command::SetFile { path, properties, quality, save, foreground }) => {
            cmd_set_file(&path, properties, quality, save, foreground)
        }
        Some(Command::ListProperties { id }) => cmd_list_properties(&id),
        Some(Command::Info { id }) => cmd_info(&id),
        Some(Command::Probe) => cmd_probe(),
        Some(Command::PkgInfo { path, dump }) => cmd_pkg_info(&path, dump.as_deref()),
        Some(Command::TexInfo { path, save }) => cmd_tex_info(&path, save.as_deref()),
        Some(Command::RenderScene {
            id_or_path,
            output,
            properties,
        }) => cmd_render_scene(&id_or_path, &output, properties),
        Some(Command::PreviewScene {
            id_or_path,
            width,
            height,
        }) => cmd_preview_scene(&id_or_path, width, height),
        Some(Command::TestScene { id_or_path, frames }) => cmd_test_scene(&id_or_path, frames),
        Some(Command::Config { action }) => cmd_config(action),
        Some(Command::Run { foreground }) => cmd_run(foreground),
        Some(Command::Daemon) => cmd_daemon(),
        Some(Command::Status) => cmd_status(),
        Some(Command::Stop) => cmd_stop(),
    };

    if let Err(e) = result {
        eprintln!("error: {e:#}");
        std::process::exit(1);
    }
}

// ── GUI ───────────────────────────────────────────────────────────────────────

fn run_ui() -> Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("wp-engine")
            .with_inner_size([1100.0, 700.0])
            .with_min_inner_size([800.0, 500.0]),
        ..Default::default()
    };

    eframe::run_native(
        "wp-engine",
        options,
        Box::new(|cc| Ok(Box::new(ui::WpApp::new(cc)))),
    )
    .map_err(|e| anyhow!("GUI error: {e}"))
}

// ── CLI commands ──────────────────────────────────────────────────────────────

fn cmd_list(type_filter: Option<String>) -> Result<()> {
    let wallpapers = workshop::scan_wallpapers();

    if wallpapers.is_empty() {
        println!("No Wallpaper Engine Workshop items found.");
        println!();
        println!(
            "Expected path: ~/.local/share/Steam/steamapps/workshop/content/{}/",
            workshop::WALLPAPER_ENGINE_APP_ID
        );
        return Ok(());
    }

    let items: Vec<&Wallpaper> = wallpapers
        .iter()
        .filter(|w| {
            type_filter
                .as_deref()
                .map(|f| w.wallpaper_type().to_string() == f)
                .unwrap_or(true)
        })
        .collect();

    println!("{:<20} {:<12} {}", "ID", "Type", "Title");
    println!("{}", "─".repeat(64));
    for w in &items {
        println!(
            "{:<20} {:<12} {}",
            w.workshop_id,
            w.wallpaper_type(),
            w.title()
        );
    }
    println!("\n{} wallpaper(s)", items.len());
    Ok(())
}

fn cmd_info(id: &str) -> Result<()> {
    let w = workshop::find_by_id(id).ok_or_else(|| anyhow!("workshop item '{id}' not found"))?;

    println!("ID:          {}", w.workshop_id);
    println!("Title:       {}", w.title());
    println!("Type:        {}", w.wallpaper_type());
    println!("Path:        {}", w.path.display());
    if let Some(f) = w.wallpaper_file() {
        println!(
            "File:        {} {}",
            f.display(),
            if f.exists() { "" } else { "(MISSING)" }
        );
    }
    if let Some(tags) = &w.project.tags {
        if !tags.is_empty() {
            println!("Tags:        {}", tags.join(", "));
        }
    }
    if let Some(rating) = &w.project.contentrating {
        if !rating.is_empty() {
            println!("Rating:      {rating}");
        }
    }
    if let Some(description) = &w.project.description {
        if !description.is_empty() {
            println!("Description: {}", description.replace('\n', " "));
        }
    }
    Ok(())
}

fn cmd_set(
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
fn apply_saved_settings(
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
            wp_engine::platform::RenderQuality::parse(s)
                .ok_or_else(|| anyhow!("invalid --quality '{s}' (expected Ultra, High, Medium, or Low)"))
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
            let (name, value) = engine::properties::parse_property_arg(arg);
            settings.set_property(id_or_path, &name, &value);
            saved_anything = true;
        }
        if !cli_properties.is_empty() {
            println!("Saved {} property override(s) for next time", cli_properties.len());
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

fn cmd_list_properties(id: &str) -> Result<()> {
    let w = workshop::find_by_id(id).ok_or_else(|| anyhow!("workshop item '{id}' not found"))?;
    let all_props = engine::properties::list_properties(&w.path)?;
    if all_props.is_empty() {
        println!(
            "\"{}\" declares no user-configurable properties.",
            w.title()
        );
        return Ok(());
    }
    // Decorative declarations (HTML separators/section headers — real
    // content ships plenty of these) aren't things a user can actually
    // set; listing them as if they were is misleading, not just noisy.
    let (settable, decorative): (Vec<_>, Vec<_>) =
        all_props.into_iter().partition(|p| p.is_settable());

    if settable.is_empty() {
        println!(
            "\"{}\" declares no settable properties ({} decorative label(s) only).",
            w.title(),
            decorative.len()
        );
        return Ok(());
    }

    println!("{:<28} {:<8} {:<24} {}", "Name", "Type", "Default", "Label");
    println!("{}", "─".repeat(80));
    for p in &settable {
        println!("{:<28} {:<8} {:<24} {}", p.name, p.kind, p.value, p.text);
        if p.kind == "combo" && !p.options.is_empty() {
            let choices = p
                .options
                .iter()
                .map(|(value, label)| format!("{value}={label}"))
                .collect::<Vec<_>>()
                .join(", ");
            println!("{:<28} {:<8} choices: {choices}", "", "");
        } else if p.kind == "slider" {
            if let (Some(min), Some(max)) = (p.min, p.max) {
                let step = p.step.map(|s| format!(", step {s}")).unwrap_or_default();
                println!("{:<28} {:<8} range: {min}..={max}{step}", "", "");
            }
        }
    }
    if !decorative.is_empty() {
        println!(
            "\n({} decorative label(s) not shown — no real value to set)",
            decorative.len()
        );
    }
    println!("\nOverride with: wp-engine set {id} --set-property NAME=VALUE");
    Ok(())
}

fn cmd_config(action: ConfigAction) -> Result<()> {
    use wp_engine::settings::{Playlist, WpSettings};

    let mut settings = WpSettings::load();

    match action {
        ConfigAction::Show => {
            let path = wp_engine::settings::settings_path();
            println!("Settings file: {}", path.display());
            if !path.exists() {
                println!("(nothing saved yet)");
                return Ok(());
            }
            println!("{}", serde_json::to_string_pretty(&settings)?);
            return Ok(());
        }
        ConfigAction::SetProperty { id, properties } => {
            for arg in &properties {
                let (name, value) = engine::properties::parse_property_arg(arg);
                println!("{id}: {name} = {value}");
                settings.set_property(&id, &name, &value);
            }
        }
        ConfigAction::UnsetProperty { id, name } => {
            if settings.unset_property(&id, &name) {
                println!("Removed {id}: {name}");
            } else {
                println!("No saved override for {id}: {name}");
                return Ok(());
            }
        }
        ConfigAction::SetScreen { screen, id } => {
            settings.set_screen(&screen, &id);
            println!("{screen} -> {id}");
        }
        ConfigAction::UnsetScreen { screen } => {
            if settings.unset_screen(&screen) {
                println!("Removed screen assignment: {screen}");
            } else {
                println!("No saved assignment for screen: {screen}");
                return Ok(());
            }
        }
        ConfigAction::SetDefault { id } => {
            settings.set_default_background(&id);
            println!("Default background -> {id}");
        }
        ConfigAction::SetQuality { id, quality } => {
            let q = platform::RenderQuality::parse(&quality).ok_or_else(|| {
                anyhow!("invalid quality '{quality}' (expected Ultra, High, Medium, or Low)")
            })?;
            settings.set_quality(&id, q);
            println!("{id}: quality = {}", q.label());
        }
        ConfigAction::UnsetQuality { id } => {
            if settings.unset_quality(&id) {
                println!("Removed quality override for {id}");
            } else {
                println!("No saved quality override for {id}");
                return Ok(());
            }
        }
        ConfigAction::SetPlaylist {
            name,
            items,
            delay,
            mode,
            order,
        } => {
            println!("Playlist '{name}': {} item(s)", items.len());
            settings.playlists.insert(
                name,
                Playlist {
                    items,
                    delay_minutes: delay,
                    mode,
                    order,
                },
            );
        }
        ConfigAction::RemovePlaylist { name } => {
            if settings.playlists.remove(&name).is_some() {
                println!("Removed playlist: {name}");
            } else {
                println!("No such playlist: {name}");
                return Ok(());
            }
        }
    }

    settings.save()?;
    Ok(())
}

/// A workshop ID or a filesystem path, resolved to a real path on disk —
/// the same "ID first, then treat it as a path" rule `test-scene`/
/// `render-scene` already use.
fn resolve_id_or_path(id_or_path: &str) -> Result<PathBuf> {
    if let Some(w) = workshop::find_by_id(id_or_path) {
        return Ok(w.path);
    }
    let path = PathBuf::from(id_or_path);
    if path.exists() {
        Ok(path)
    } else {
        Err(anyhow!(
            "not a directory and workshop item '{id_or_path}' not found"
        ))
    }
}

/// Runs using persisted settings (`wp-engine config` to set them up first):
/// every `screens` entry gets its own independent wallpaper *and* its own
/// independent render quality on its own output (real per-screen rendering
/// — see `render::ScreenContent`/`render::ScreenSettings`); any output with
/// no specific entry falls back to the first configured screen's wallpaper,
/// else `default_background`, else the first playlist's first item.
///
/// Honest scope note: property overrides are still process-global (see
/// `engine::properties`), so only the *default* background's saved
/// overrides are applied — a per-screen wallpaper's own saved overrides
/// aren't isolated from each other yet if two different screens' wallpapers
/// happen to declare a property with the same name. Playlists are still
/// data-only (no rotation timer exists) — only each playlist's first item
/// is ever actually reachable through `run` today.
fn cmd_run(foreground: bool) -> Result<()> {
    let settings = wp_engine::settings::WpSettings::load();

    let default_id_or_path = settings
        .screens
        .values()
        .next()
        .or(settings.default_background.as_ref())
        .or_else(|| settings.playlists.values().next().and_then(|p| p.items.first()))
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
                        Arc::new(Mutex::new(wp_engine::render::RenderSettings {
                            quality: q,
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
fn launch(context: ApplicationContext, title: Option<String>, foreground: bool) -> Result<()> {
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

#[cfg(target_os = "linux")]
fn cmd_daemon() -> Result<()> {
    wp_engine::daemon::run_daemon()
}

#[cfg(target_os = "linux")]
fn cmd_status() -> Result<()> {
    use wp_engine::daemon::{self, Request, Response};
    if !daemon::is_running() {
        println!("Daemon: not running");
        return Ok(());
    }
    let Response::Status(status) = daemon::request(&Request::Status)? else {
        return Err(anyhow!("unexpected daemon reply to status"));
    };
    println!("Daemon: running (pid {})", status.pid);
    match status.active {
        Some(active) => {
            if let Some(title) = &active.title {
                println!("Wallpaper: {title}");
            }
            println!("Path: {}", active.background.display());
            println!("Quality: {}", active.quality);
        }
        None => println!("Wallpaper: none"),
    }
    Ok(())
}

#[cfg(target_os = "linux")]
fn cmd_stop() -> Result<()> {
    use wp_engine::daemon::{self, Request};
    if !daemon::is_running() {
        println!("Daemon: not running");
        return Ok(());
    }
    daemon::request(&Request::Shutdown)?;
    println!("Daemon stopped.");
    Ok(())
}

#[cfg(not(target_os = "linux"))]
fn cmd_daemon() -> Result<()> {
    Err(anyhow!("the background daemon is Linux-only"))
}

#[cfg(not(target_os = "linux"))]
fn cmd_status() -> Result<()> {
    cmd_daemon()
}

#[cfg(not(target_os = "linux"))]
fn cmd_stop() -> Result<()> {
    cmd_daemon()
}

fn cmd_probe() -> Result<()> {
    // ── 1. List all visible adapters ─────────────────────────────────────────
    let adapters = platform::probe_adapters();
    if adapters.is_empty() {
        println!("No GPU adapters found.");
        return Ok(());
    }
    println!(
        "{:<40} {:<10} {:<12} {}",
        "Name", "Backend", "Type", "PCI IDs"
    );
    println!("{}", "─".repeat(74));
    for a in &adapters {
        println!(
            "{:<40} {:<10} {:<12} {:04x}:{:04x}",
            a.name, a.backend, a.device_type, a.vendor_id, a.device_id
        );
    }
    println!("\n{} adapter(s) found\n", adapters.len());

    // ── 2. Open the best device to confirm driver access works ───────────────
    print!("Opening best device... ");
    match platform::open_device() {
        Ok(dev) => println!(
            "OK  →  {} ({}, {})",
            dev.info.name, dev.info.backend, dev.info.device_type
        ),
        Err(e) => println!("FAILED: {e}"),
    }

    Ok(())
}

fn cmd_set_file(
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

fn cmd_pkg_info(path: &std::path::Path, dump: Option<&std::path::Path>) -> Result<()> {
    let pkg = engine::Package::from_file(path)?;

    println!("Files   : {}", pkg.len());
    println!("Empty   : {}", pkg.is_empty());
    println!(
        "Scene   : {}",
        if pkg.contains("scene.json") {
            "yes"
        } else {
            "no"
        }
    );
    println!();

    // Collect and sort paths for a stable, readable table.
    let mut paths: Vec<&str> = pkg.paths().collect();
    paths.sort_unstable();

    for p in &paths {
        let size_kb = pkg.get(p).map(|b| b.len()).unwrap_or(0) / 1024;
        println!("{size_kb:>6} KB  {p}");
    }

    if let Some(dump_dir) = dump {
        std::fs::create_dir_all(dump_dir)
            .with_context(|| format!("failed to create dump directory: {}", dump_dir.display()))?;

        let mut count = 0usize;
        for p in &paths {
            if let Some(bytes) = pkg.get(p) {
                let dest = dump_dir.join(p);
                if let Some(parent) = dest.parent() {
                    std::fs::create_dir_all(parent).with_context(|| {
                        format!("failed to create directory: {}", parent.display())
                    })?;
                }
                std::fs::write(&dest, bytes)
                    .with_context(|| format!("failed to write: {}", dest.display()))?;
                count += 1;
            }
        }
        println!("\nExtracted {count} files to {}", dump_dir.display());
    }

    Ok(())
}

fn cmd_tex_info(path: &std::path::Path, save: Option<&std::path::Path>) -> Result<()> {
    let data = std::fs::read(path)?;
    let tex = engine::TexFile::parse(&data)?;
    println!("Format:       {:?}", tex.format());
    println!("Image size:   {}x{}", tex.image_width, tex.image_height);
    println!("Texture size: {}x{}", tex.texture_width, tex.texture_height);
    println!("Flags:        0x{:x}", tex.flags());
    println!(
        "Frames:       {} (animated: {})",
        tex.frames().len(),
        tex.is_animated()
    );
    if let Some(out) = save {
        let img = tex.to_rgba()?;
        img.save(out)?;
        println!("Saved to {}", out.display());
    }
    Ok(())
}

fn cmd_render_scene(
    id_or_path: &str,
    output: &std::path::Path,
    properties: Vec<String>,
) -> Result<()> {
    let overrides = properties
        .iter()
        .map(|p| engine::properties::parse_property_arg(p))
        .collect();
    engine::properties::set_global_overrides(overrides);

    let dir = std::path::PathBuf::from(id_or_path);
    let dir = if dir.exists() {
        dir
    } else {
        workshop::find_by_id(id_or_path)
            .map(|w| w.path)
            .ok_or_else(|| anyhow!("not a directory and workshop item '{id_or_path}' not found"))?
    };
    println!("Loading scene from {}...", dir.display());
    let graph = engine::SceneGraph::from_directory(&dir)?;
    println!("Scene graph: {}", graph.stats().summary());

    let scene = engine::ResolvedScene::from_directory(&dir)?;
    println!(
        "Rendering {}x{} with {} layers...",
        scene.width,
        scene.height,
        scene.layers.len()
    );
    for layer in scene.layers.iter().take(3) {
        println!(
            "Layer:       {} origin={:.1},{:.1},{:.1} size={:.1},{:.1},{:.1} parallax={:.3},{:.3}",
            if layer.name.is_empty() {
                "<unnamed>"
            } else {
                &layer.name
            },
            layer.origin[0],
            layer.origin[1],
            layer.origin[2],
            layer.size[0],
            layer.size[1],
            layer.size[2],
            layer.parallax_depth[0],
            layer.parallax_depth[1],
        );
    }
    let img = scene.render();
    img.save(output)?;
    println!("Saved to {}", output.display());
    Ok(())
}

fn cmd_test_scene(id_or_path: &str, num_frames: usize) -> Result<()> {
    use std::sync::{mpsc::sync_channel, Arc};

    let dir = std::path::PathBuf::from(id_or_path);
    let dir = if dir.exists() {
        dir
    } else {
        workshop::find_by_id(id_or_path)
            .map(|w| w.path)
            .ok_or_else(|| anyhow!("not a directory and workshop item '{id_or_path}' not found"))?
    };

    println!(
        "GPU scene animation test: collecting {num_frames} frames from {}",
        dir.display()
    );
    let (tx, rx) = sync_channel::<Arc<image::RgbaImage>>(2);
    let render_dir = dir.clone();
    let handle = std::thread::spawn(move || {
        if let Err(e) = engine::gpu_renderer::gpu_scene_render_loop(&render_dir, &tx, 30.0) {
            tracing::error!(target: "render", "gpu scene error: {e}");
        }
    });

    let mut collected = Vec::with_capacity(num_frames);
    for i in 0..num_frames {
        match rx.recv() {
            Ok(frame) => {
                if i == 0 || i == num_frames / 2 || i == num_frames - 1 {
                    eprintln!("  frame {i}: {}x{}", frame.width(), frame.height());
                }
                collected.push(frame);
            }
            Err(_) => {
                eprintln!("  channel closed after {} frames", collected.len());
                break;
            }
        }
    }

    // Drop the receiver so the render loop's next `send` fails and it returns,
    // then join it: letting the process exit while it's still mid-frame (still
    // holding the wgpu device/queue) can crash the driver on shutdown.
    drop(rx);
    let _ = handle.join();

    // Headless GPU-frame capture for debugging "renders weirdly" reports
    // (the only way to see a perspective/live-path frame off-screen).
    if let Ok(path) = std::env::var("WP_DEBUG_DUMP_FRAME") {
        if let Some(f) = collected.last() {
            let _ = f.save(&path);
            eprintln!("  saved last frame to {path}");
        }
    }

    if collected.len() < 2 {
        anyhow::bail!("not enough frames collected (got {})", collected.len());
    }

    // Compare frame[0] against every other frame; take the maximum change found.
    // Using max rather than first-vs-last avoids false negatives when a periodic
    // effect's endpoints happen to land near the same phase.
    let first = collected.first().unwrap();
    let first_bytes = first.as_raw();
    let total_pixels = (first.width() * first.height()) as usize;

    fn count_changed(a: &[u8], b: &[u8]) -> usize {
        a.chunks(4)
            .zip(b.chunks(4))
            .filter(|(a, b)| {
                let dr = (a[0] as i32 - b[0] as i32).abs();
                let dg = (a[1] as i32 - b[1] as i32).abs();
                let db = (a[2] as i32 - b[2] as i32).abs();
                dr + dg + db > 6
            })
            .count()
    }

    let (best_frame, changed_pixels) = collected
        .iter()
        .enumerate()
        .skip(1)
        .map(|(i, f)| (i, count_changed(first_bytes, f.as_raw())))
        .max_by_key(|(_, n)| *n)
        .unwrap_or((0, 0));

    let pct = changed_pixels as f64 / total_pixels as f64 * 100.0;
    println!(
        "\nCollected {} frames at 30fps (~{:.1}s)",
        collected.len(),
        collected.len() as f64 / 30.0
    );
    println!("Changed pixels (frame[0] vs frame[{best_frame}]): {changed_pixels}/{total_pixels} ({pct:.1}%)");

    if pct > 0.1 {
        println!("PASS: scene IS animating ({pct:.1}% of pixels changed)");
    } else {
        println!(
            "FAIL: scene appears static (only {pct:.2}% changed — likely effect/rendering bug)"
        );
    }

    // Save first and best-changed frames for visual inspection
    let last = collected.last().unwrap();
    first.save("/tmp/wp_frame_first.png")?;
    last.save("/tmp/wp_frame_last.png")?;
    println!("Saved /tmp/wp_frame_first.png and /tmp/wp_frame_last.png for visual comparison");

    Ok(())
}

fn cmd_preview_scene(id_or_path: &str, width: u32, height: u32) -> Result<()> {
    use image::imageops::FilterType;
    use minifb::{Key, Window, WindowOptions};
    use std::sync::{mpsc::sync_channel, Arc};

    let dir = std::path::PathBuf::from(id_or_path);
    let dir = if dir.exists() {
        dir
    } else {
        workshop::find_by_id(id_or_path)
            .map(|w| w.path)
            .ok_or_else(|| anyhow!("not a directory and workshop item '{id_or_path}' not found"))?
    };

    println!("Loading scene from {}...", dir.display());
    let (tx, rx) = sync_channel::<Arc<image::RgbaImage>>(2);
    let render_dir = dir.clone();
    let handle = std::thread::spawn(move || {
        if let Err(e) = engine::gpu_renderer::gpu_scene_render_loop(&render_dir, &tx, 30.0) {
            tracing::error!(target: "render", "gpu scene error: {e}");
        }
    });

    println!("Waiting for first frame...");
    let mut current = rx.recv()?;
    println!(
        "First frame received — opening {}x{} preview window. Press Esc to quit.",
        width, height
    );

    let mut window = Window::new(
        "wp-engine preview",
        width as usize,
        height as usize,
        WindowOptions::default(),
    )?;

    while window.is_open() && !window.is_key_down(Key::Escape) {
        if let Ok(frame) = rx.try_recv() {
            current = frame;
        }

        let scaled = image::imageops::resize(&*current, width, height, FilterType::Nearest);
        let buf: Vec<u32> = scaled
            .pixels()
            .map(|p| {
                let [r, g, b, _a] = p.0;
                (r as u32) << 16 | (g as u32) << 8 | b as u32
            })
            .collect();

        window
            .update_with_buffer(&buf, width as usize, height as usize)
            .unwrap_or(());
        std::thread::sleep(std::time::Duration::from_millis(16));
    }

    // See cmd_test_scene: join the render thread before exiting so it isn't
    // killed mid-frame while still holding the wgpu device/queue.
    drop(rx);
    let _ = handle.join();

    Ok(())
}
