//! The `wp-engine` command line: argument parsing and dispatch, with the
//! commands split by area into this module's files.

use anyhow::{anyhow, Context, Result};
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use wp_engine::application::{ApplicationContext, WallpaperApplication};
use wp_engine::workshop::{self, Wallpaper};
use wp_engine::{engine, formats, platform, ui};

mod library;
use library::*;

mod apply;
use apply::*;

mod config;
use config::*;

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
    /// Render a video, image or web wallpaper's frame to a PNG at a given
    /// size (for debugging)
    RenderFrame {
        /// Workshop ID, directory, or file
        id_or_path: String,
        #[arg(long, default_value = "frame_output.png")]
        output: PathBuf,
        #[arg(long, default_value_t = 1920)]
        width: u32,
        #[arg(long, default_value_t = 1080)]
        height: u32,
        /// Frame rate handed to web pages
        #[arg(long, default_value_t = 30)]
        fps: u32,
        /// Keep rendering this long before saving the latest frame
        #[arg(long, default_value_t = 2.0)]
        seconds: f32,
        /// Override a user property: NAME=VALUE (repeatable)
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
    /// Control playlist playback in the background daemon
    Playlist {
        #[command(subcommand)]
        action: PlaylistAction,
    },
    /// Start `wp-engine run` at login: on, off, or show the current state
    Autostart {
        #[arg(value_parser = ["on", "off", "status"], default_value = "status")]
        state: String,
    },
}

#[derive(Subcommand)]
enum PlaylistAction {
    /// Start playing a saved playlist (see `config set-playlist`)
    Play { name: String },
    /// Skip to the next item
    Next,
    /// Go back to the previous item
    Previous,
    /// Stop rotating, keeping the current wallpaper
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
        /// timer (every --delay minutes), daytime (each item at its --times
        /// entry), dayofweek (item N on weekday N, Monday first), or login
        /// (next item each time playback starts)
        #[arg(long, default_value = "timer")]
        mode: String,
        /// sequential or random
        #[arg(long, default_value = "sequential")]
        order: String,
        /// daytime mode: comma-separated start times, one per item ("07:00,19:30")
        #[arg(long, value_delimiter = ',')]
        times: Vec<String>,
    },
    /// Make `wp-engine run` play this playlist (omit NAME to clear)
    SetActivePlaylist { name: Option<String> },
    /// Remove a playlist
    RemovePlaylist { name: String },
    /// Set what happens while a condition holds: CONDITION is focus,
    /// maximized, fullscreen, audio, battery or lock; ACTION is run, mute,
    /// pause or stop (pause and free memory)
    SetPlayback { condition: String, action: String },
    /// Cap the frame rate of animated wallpapers (1-240)
    SetFps { fps: u32 },
    /// How video, image and web wallpapers fit the screen: fill (crop,
    /// default), fit (black bars), stretch, or center
    SetScaling { mode: String },
    /// How one wallpaper covers several monitors: per-screen (default; see
    /// set-screen), span (stretched across all of them) or clone (the same
    /// picture on each)
    SetLayout { layout: String },
}

pub fn main() {
    // CEF launches its renderer/GPU/utility processes by re-executing THIS
    // binary with `--type=...`. That must be handled before anything else —
    // before clap, which would reject the flag, and before any thread starts.
    // Returns `Some` only in a child process, which must then exit at once.
    if let Some(code) = wp_engine::content::web::subprocess_main() {
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
        Some(Command::Set {
            id,
            properties,
            quality,
            save,
            foreground,
        }) => cmd_set(&id, properties, quality, save, foreground),
        Some(Command::SetFile {
            path,
            properties,
            quality,
            save,
            foreground,
        }) => cmd_set_file(&path, properties, quality, save, foreground),
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
        Some(Command::RenderFrame {
            id_or_path,
            output,
            width,
            height,
            fps,
            seconds,
            properties,
        }) => cmd_render_frame(
            &id_or_path,
            &output,
            (width, height),
            fps,
            seconds,
            properties,
        ),
        Some(Command::Config { action }) => cmd_config(action),
        Some(Command::Run { foreground }) => cmd_run(foreground),
        Some(Command::Daemon) => cmd_daemon(),
        Some(Command::Status) => cmd_status(),
        Some(Command::Stop) => cmd_stop(),
        Some(Command::Playlist { action }) => cmd_playlist(action),
        Some(Command::Autostart { state }) => cmd_autostart(&state),
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

fn resolve_scene_dir(id_or_path: &str) -> Result<PathBuf> {
    let dir = resolve_id_or_path(id_or_path)?;
    wp_engine::content::WallpaperContent::preset_base_dir(&dir)
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
    if let Some((name, index, count)) = status.playlist {
        println!("Playlist: {name} (item {} of {count})", index + 1);
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

#[cfg(target_os = "linux")]
fn cmd_playlist(action: PlaylistAction) -> Result<()> {
    use wp_engine::daemon::{self, Request, Response};
    let req = match &action {
        PlaylistAction::Play { name } => Request::PlayPlaylist { name: name.clone() },
        PlaylistAction::Next => Request::NextItem,
        PlaylistAction::Previous => Request::PreviousItem,
        PlaylistAction::Stop => Request::StopPlaylist,
    };
    let reply = if matches!(action, PlaylistAction::Play { .. }) {
        daemon::request_or_spawn(&req)?
    } else {
        daemon::request(&req)?
    };
    if let Response::Error { message } = reply {
        return Err(anyhow!(message));
    }
    println!("OK");
    Ok(())
}

#[cfg(not(target_os = "linux"))]
fn cmd_playlist(_action: PlaylistAction) -> Result<()> {
    cmd_daemon()
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
    let outputs = platform::display::output_names();
    if !outputs.is_empty() {
        println!(
            "Screens (for `config set-screen`): {}\n",
            outputs.join(", ")
        );
    }

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

fn cmd_pkg_info(path: &std::path::Path, dump: Option<&std::path::Path>) -> Result<()> {
    let pkg = formats::pkg::Package::from_file(path)?;

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
    let tex = formats::tex::TexFile::parse(&data)?;
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
        .map(|p| workshop::properties::parse_property_arg(p))
        .collect();
    workshop::properties::set_global_overrides(overrides);

    let dir = resolve_scene_dir(id_or_path)?;
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

fn cmd_render_frame(
    id_or_path: &str,
    output: &std::path::Path,
    size: (u32, u32),
    fps: u32,
    seconds: f32,
    properties: Vec<String>,
) -> Result<()> {
    workshop::properties::set_global_overrides(
        properties
            .iter()
            .map(|p| workshop::properties::parse_property_arg(p))
            .collect(),
    );
    let path = resolve_id_or_path(id_or_path)?;
    let content = wp_engine::content::WallpaperContent::from_any_path(&path)?;
    let mut source = wp_engine::content::FrameSource::from_content_with(
        content,
        wp_engine::content::web::WebOptions { size, fps },
    )?;
    let until = std::time::Instant::now() + std::time::Duration::from_secs_f32(seconds.max(0.0));
    while std::time::Instant::now() < until {
        source.try_advance();
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    let frame = source.current_frame();
    frame.save(output)?;
    println!(
        "Saved {}x{} frame to {}",
        frame.width(),
        frame.height(),
        output.display()
    );
    Ok(())
}

fn cmd_test_scene(id_or_path: &str, num_frames: usize) -> Result<()> {
    use std::sync::{mpsc::sync_channel, Arc};

    let dir = resolve_scene_dir(id_or_path)?;

    println!(
        "GPU scene animation test: collecting {num_frames} frames from {}",
        dir.display()
    );
    let (tx, rx) = sync_channel::<Arc<image::RgbaImage>>(2);
    let render_dir = dir.clone();
    let handle = std::thread::spawn(move || {
        if let Err(e) = engine::gpu::renderer::gpu_scene_render_loop(&render_dir, &tx, 30.0) {
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

    let dir = resolve_scene_dir(id_or_path)?;

    println!("Loading scene from {}...", dir.display());
    let (tx, rx) = sync_channel::<Arc<image::RgbaImage>>(2);
    let render_dir = dir.clone();
    let handle = std::thread::spawn(move || {
        if let Err(e) = engine::gpu::renderer::gpu_scene_render_loop(&render_dir, &tx, 30.0) {
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
