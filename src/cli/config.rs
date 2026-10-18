//! Changing saved settings: `config` and `autostart`.

use super::*;

pub(super) fn cmd_config(action: ConfigAction) -> Result<()> {
    use wp_engine::settings::{Playlist, WpSettings};

    let mut settings = WpSettings::load();
    // Property edits to show live on a running daemon once saved.
    let mut live: Option<(String, Vec<String>)> = None;

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
            let mut names = Vec::new();
            for arg in &properties {
                let (name, value) = workshop::properties::parse_property_arg(arg);
                println!("{id}: {name} = {value}");
                settings.set_property(&id, &name, &value);
                names.push(name);
            }
            live = Some((id, names));
        }
        ConfigAction::UnsetProperty { id, name } => {
            if settings.unset_property(&id, &name) {
                println!("Removed {id}: {name}");
                live = Some((id, vec![name]));
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
            times,
        } => {
            if !["timer", "daytime", "dayofweek", "login"].contains(&mode.as_str()) {
                return Err(anyhow!(
                    "invalid mode '{mode}' (expected timer, daytime, dayofweek or login)"
                ));
            }
            if !["sequential", "random"].contains(&order.as_str()) {
                return Err(anyhow!(
                    "invalid order '{order}' (expected sequential or random)"
                ));
            }
            if mode == "daytime" {
                if times.len() != items.len() {
                    return Err(anyhow!("daytime mode needs one --times entry per item"));
                }
                if let Some(bad) = times
                    .iter()
                    .find(|t| wp_engine::schedule::parse_time_of_day(t).is_none())
                {
                    return Err(anyhow!("invalid time '{bad}' (expected HH:MM)"));
                }
            }
            println!("Playlist '{name}': {} item(s)", items.len());
            settings.playlists.insert(
                name,
                Playlist {
                    items,
                    delay_minutes: delay,
                    mode,
                    order,
                    times,
                },
            );
        }
        ConfigAction::SetActivePlaylist { name } => {
            if let Some(n) = &name {
                if !settings.playlists.contains_key(n) {
                    return Err(anyhow!("no playlist named '{n}'"));
                }
                println!("`wp-engine run` plays playlist '{n}'");
            } else {
                println!("No active playlist");
            }
            settings.active_playlist = name;
        }
        ConfigAction::SetPlayback { condition, action } => {
            let a = wp_engine::services::playback_gate::PlaybackAction::parse(&action).ok_or_else(
                || anyhow!("invalid action '{action}' (expected run, mute, pause or stop)"),
            )?;
            if !settings.playback.set(&condition, a) {
                return Err(anyhow!(
                    "invalid condition '{condition}' (expected focus, maximized, fullscreen, audio, battery or lock)"
                ));
            }
            println!("While {condition}: {}", a.label());
        }
        ConfigAction::SetScaling { mode } => {
            let m = wp_engine::content::ScaleMode::parse(&mode).ok_or_else(|| {
                anyhow!("invalid mode '{mode}' (expected fill, fit, stretch or center)")
            })?;
            settings.scaling = Some(m);
            println!("Scaling: {}", m.label());
        }
        ConfigAction::SetLayout { layout } => {
            let l = wp_engine::content::ScreenLayout::parse(&layout).ok_or_else(|| {
                anyhow!("invalid layout '{layout}' (expected per-screen, span or clone)")
            })?;
            settings.layout = Some(l);
            println!(
                "Layout: {} (applies the next time a wallpaper is set)",
                l.label()
            );
        }
        ConfigAction::SetFps { fps } => {
            if !(1..=240).contains(&fps) {
                return Err(anyhow!("fps must be between 1 and 240"));
            }
            settings.fps = Some(fps);
            println!("Frame-rate cap: {fps} fps");
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
    if let Some((wallpaper, names)) = live.filter(|_| wp_engine::daemon::is_running()) {
        let _ = wp_engine::daemon::request(&wp_engine::daemon::Request::PropertiesChanged {
            wallpaper,
            names,
        });
    }
    Ok(())
}

/// [`resolve_id_or_path`] for the scene tools: a preset item resolves to the
/// item it builds on, with the preset's values registered as overrides.
pub(super) fn cmd_autostart(state: &str) -> Result<()> {
    if state != "status" {
        wp_engine::settings::set_autostart(state == "on")?;
    }
    if wp_engine::settings::autostart_enabled() {
        let path = wp_engine::settings::autostart_path();
        println!("Autostart on ({})", path.display());
    } else {
        println!("Autostart off");
    }
    Ok(())
}
