//! Talking to the background daemon: `daemon`, `status`, `stop` and `playlist`
//! (Linux; elsewhere they report the daemon is unavailable).

use super::*;

#[cfg(target_os = "linux")]
pub(super) fn cmd_daemon() -> Result<()> {
    wp_engine::daemon::run_daemon()
}

#[cfg(target_os = "linux")]
pub(super) fn cmd_status() -> Result<()> {
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
pub(super) fn cmd_stop() -> Result<()> {
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
pub(super) fn cmd_daemon() -> Result<()> {
    Err(anyhow!("the background daemon is Linux-only"))
}

#[cfg(target_os = "linux")]
pub(super) fn cmd_playlist(action: PlaylistAction) -> Result<()> {
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
pub(super) fn cmd_playlist(_action: PlaylistAction) -> Result<()> {
    cmd_daemon()
}

#[cfg(not(target_os = "linux"))]
pub(super) fn cmd_status() -> Result<()> {
    cmd_daemon()
}

#[cfg(not(target_os = "linux"))]
pub(super) fn cmd_stop() -> Result<()> {
    cmd_daemon()
}
