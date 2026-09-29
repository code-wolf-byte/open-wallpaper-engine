//! Single-instance background daemon that owns the wallpaper renderer.
//!
//! Exactly one daemon runs per user session, enforced by an exclusive
//! `flock` on `daemon.lock` in the runtime directory. It listens on
//! `daemon.sock` next to it; every other `wp-engine` invocation that wants
//! to show a wallpaper (`set`, `set-file`, `run`, the GUI) is a client that
//! hands the wallpaper to the daemon — starting it detached first if none is
//! running — and returns. The wallpaper therefore outlives the command (or
//! GUI window) that applied it, and a second apply replaces the first
//! instead of stacking a second renderer on the same outputs.
//!
//! Linux-only: the macOS backend must own the process main thread for its
//! event loop and already launches wallpapers as child processes.

pub mod client;
pub mod protocol;
pub mod server;

use std::path::PathBuf;

pub use client::{is_running, request, request_or_spawn, set_spawn_verbosity};
pub use protocol::{ActiveWallpaper, ApplySpec, DaemonStatus, Request, Response};
pub use server::run_daemon;

/// Overrides the runtime directory (tests, or running several isolated
/// daemons side by side).
pub const RUNTIME_DIR_ENV: &str = "WP_ENGINE_RUNTIME_DIR";

/// `$WP_ENGINE_RUNTIME_DIR`, else `$XDG_RUNTIME_DIR/wp-engine`, else
/// `/tmp/wp-engine-$UID`.
pub fn runtime_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os(RUNTIME_DIR_ENV) {
        return PathBuf::from(dir);
    }
    if let Some(dir) = dirs::runtime_dir() {
        return dir.join("wp-engine");
    }
    let uid = unsafe { libc::getuid() };
    std::env::temp_dir().join(format!("wp-engine-{uid}"))
}

pub fn socket_path() -> PathBuf {
    runtime_dir().join("daemon.sock")
}

pub fn lock_path() -> PathBuf {
    runtime_dir().join("daemon.lock")
}

/// Where a daemon started in the background writes its stdout/stderr:
/// `$XDG_STATE_HOME/wp-engine/daemon.log` (falls back to the runtime dir).
pub fn log_path() -> PathBuf {
    dirs::state_dir()
        .map(|d| d.join("wp-engine"))
        .unwrap_or_else(runtime_dir)
        .join("daemon.log")
}
