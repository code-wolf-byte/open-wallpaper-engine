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

mod daemon;
use daemon::*;

mod debug;
use debug::*;

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
