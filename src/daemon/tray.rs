//! The daemon's tray icon (StatusNotifierItem): what's showing, playlist
//! controls, and a way back to the GUI. Menu actions go through the same
//! request handler as the socket.

use super::protocol::Request;
use super::server::{handle_request, State};
use ksni::blocking::TrayMethods;
use ksni::menu::{StandardItem, SubMenu};
use ksni::MenuItem;
use std::sync::{Arc, Mutex};

pub(super) struct Tray {
    state: Arc<Mutex<State>>,
    title: Option<String>,
    playlist: Option<String>,
    playlists: Vec<String>,
}

/// Show the tray icon. Without a tray host (no StatusNotifierWatcher, or no
/// session bus) the daemon just runs without one.
pub(super) fn spawn(state: Arc<Mutex<State>>) -> Option<ksni::blocking::Handle<Tray>> {
    let mut tray = Tray {
        state,
        title: None,
        playlist: None,
        playlists: Vec::new(),
    };
    tray.refresh();
    match tray.spawn() {
        Ok(handle) => Some(handle),
        Err(e) => {
            tracing::info!(target: "app", "no tray icon: {e}");
            None
        }
    }
}

impl Tray {
    fn refresh(&mut self) {
        // An apply in progress holds the lock; keep the last known state.
        if let Ok(s) = self.state.try_lock() {
            self.title = s.app.as_ref().map(|app| {
                s.title
                    .clone()
                    .unwrap_or_else(|| app.context().background.display().to_string())
            });
            self.playlist = s.playlist.as_ref().map(|r| r.name.clone());
        }
        self.playlists = crate::settings::WpSettings::load()
            .playlists
            .into_keys()
            .collect();
    }

    /// Run a request off the D-Bus thread: applying a wallpaper takes a while.
    fn send(&self, req: Request) {
        let state = Arc::clone(&self.state);
        std::thread::spawn(move || {
            if let super::Response::Error { message } = handle_request(req, &state) {
                tracing::warn!(target: "app", "tray: {message}");
            }
        });
    }
}

fn item(label: impl Into<String>, activate: impl Fn(&mut Tray) + Send + 'static) -> MenuItem<Tray> {
    StandardItem {
        label: label.into(),
        activate: Box::new(activate),
        ..Default::default()
    }
    .into()
}

impl ksni::Tray for Tray {
    fn id(&self) -> String {
        "wp-engine".into()
    }

    fn title(&self) -> String {
        "wp-engine".into()
    }

    fn icon_name(&self) -> String {
        "preferences-desktop-wallpaper".into()
    }

    fn tool_tip(&self) -> ksni::ToolTip {
        ksni::ToolTip {
            title: self.title.clone().unwrap_or_else(|| "No wallpaper".into()),
            ..Default::default()
        }
    }

    fn activate(&mut self, _x: i32, _y: i32) {
        open_gui();
    }

    fn menu_about_to_show(&mut self) {
        self.refresh();
    }

    fn menu(&self) -> Vec<MenuItem<Self>> {
        let mut menu = vec![
            StandardItem {
                label: self.title.clone().unwrap_or_else(|| "No wallpaper".into()),
                enabled: false,
                ..Default::default()
            }
            .into(),
            MenuItem::Separator,
            item("Open wp-engine", |_| open_gui()),
        ];
        if let Some(name) = &self.playlist {
            menu.push(MenuItem::Separator);
            menu.push(
                StandardItem {
                    label: format!("Playlist: {name}"),
                    enabled: false,
                    ..Default::default()
                }
                .into(),
            );
            menu.push(item("Next wallpaper", |t| t.send(Request::NextItem)));
            menu.push(item("Previous wallpaper", |t| {
                t.send(Request::PreviousItem)
            }));
            menu.push(item("Stop rotating", |t| t.send(Request::StopPlaylist)));
        }
        if !self.playlists.is_empty() {
            menu.push(
                SubMenu {
                    label: "Play playlist".into(),
                    submenu: self
                        .playlists
                        .iter()
                        .map(|name| {
                            let name = name.clone();
                            item(name.clone(), move |t| {
                                t.send(Request::PlayPlaylist { name: name.clone() })
                            })
                        })
                        .collect(),
                    ..Default::default()
                }
                .into(),
            );
        }
        menu.push(MenuItem::Separator);
        if self.title.is_some() {
            menu.push(item("Remove wallpaper", |t| t.send(Request::Clear)));
        }
        menu.push(item("Quit", |t| t.send(Request::Shutdown)));
        menu
    }
}

fn open_gui() {
    match std::env::current_exe() {
        Ok(exe) => {
            if let Ok(mut child) = std::process::Command::new(exe).spawn() {
                std::thread::spawn(move || child.wait());
            }
        }
        Err(e) => tracing::warn!(target: "app", "tray: can't find own executable: {e}"),
    }
}
