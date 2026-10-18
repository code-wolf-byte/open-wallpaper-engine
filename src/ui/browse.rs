//! The browse view: source tabs, search, filters and sorting over the
//! thumbnail grid.

use super::*;

impl WpApp {
    pub(super) fn rebuild_filter(&mut self) {
        let q = self.search.to_lowercase();
        let mut shown: Vec<usize> = (0..self.wallpapers.len())
            .filter(|&i| {
                let w = &self.wallpapers[i];
                let text_ok = q.is_empty()
                    || w.title().to_lowercase().contains(&q)
                    || w.project
                        .tags
                        .iter()
                        .flatten()
                        .any(|t| t.to_lowercase() == q);
                text_ok && self.filters.matches(w)
            })
            .collect();
        let title = |i: &usize| self.wallpapers[*i].title().to_lowercase();
        match self.sort {
            SortKey::Name => shown.sort_by_cached_key(title),
            SortKey::Type => shown.sort_by_cached_key(|i| {
                (type_name(self.wallpapers[*i].wallpaper_type()), title(i))
            }),
            // Newest first, like WE.
            SortKey::Installed => shown.sort_by_key(|&i| std::cmp::Reverse(self.installed[i])),
        }
        if self.descending {
            shown.reverse();
        }
        self.filtered = shown;
    }

    /// Source tabs with Displays/Settings, then search, filter and sort.
    pub(super) fn render_header(&mut self, ui: &mut egui::Ui, skin: Skin) {
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 2.0;
            for (label, active) in [
                ("Installed", true),
                ("Discover", false),
                ("Workshop", false),
            ] {
                if source_tab(ui, label, active, skin).clicked() && !active {
                    open_external(WORKSHOP_URL);
                }
            }
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                ui.spacing_mut().item_spacing.x = 6.0;
                if ui.button("⚙ Settings").clicked() {
                    self.show_settings = !self.show_settings;
                    // `wp-engine config` may have changed it meanwhile.
                    self.saved = crate::settings::WpSettings::load();
                    self.autostart = crate::settings::autostart_enabled();
                }
                let display = self.target_screen.as_deref().unwrap_or("All displays");
                if ui
                    .button(format!("🖵 Displays: {display}"))
                    .on_hover_text("Choose the display clicked wallpapers go to")
                    .clicked()
                {
                    self.show_displays = true;
                }
            });
        });
        ui.add_space(6.0);
        ui.horizontal(|ui| {
            let r = ui.add(
                egui::TextEdit::singleline(&mut self.search)
                    .desired_width(260.0)
                    .hint_text("🔍 Search"),
            );
            let mut changed = r.changed();
            let filter_btn = egui::Button::new("☰ Filter results")
                .selected(self.show_filter || self.filters.is_active());
            if ui.add(filter_btn).clicked() {
                self.show_filter = !self.show_filter;
            }
            if self.filtered.len() != self.wallpapers.len() {
                ui.label(
                    egui::RichText::new(format!(
                        "({} of {})",
                        self.filtered.len(),
                        self.wallpapers.len()
                    ))
                    .color(TEXT_MUTED),
                );
            }
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                let arrow = if self.descending { "⬇" } else { "⬆" };
                if ui.button(arrow).on_hover_text("Sort direction").clicked() {
                    self.descending = !self.descending;
                    changed = true;
                }
                egui::ComboBox::from_id_salt("sort")
                    .selected_text(self.sort.label())
                    .width(140.0)
                    .show_ui(ui, |ui| {
                        for key in SortKey::ALL {
                            changed |= ui
                                .selectable_value(&mut self.sort, key, key.label())
                                .changed();
                        }
                    });
            });
            if changed {
                self.rebuild_filter();
            }
        });
    }

    /// WE's "Filter results" panel: type, age rating and tags.
    pub(super) fn render_filters(&mut self, ui: &mut egui::Ui) {
        let mut changed = false;
        egui::ScrollArea::vertical().show(ui, |ui| {
            if ui.button("Reset Filters").clicked() {
                self.filters = Filters::default();
                changed = true;
            }
            ui.add_space(6.0);
            ui.strong("Type");
            for t in TYPES {
                changed |= filter_box(ui, &mut self.filters.hidden_types, t, t);
            }
            ui.add_space(6.0);
            ui.strong("Age rating");
            for r in AGE_RATINGS {
                changed |= filter_box(ui, &mut self.filters.hidden_ratings, r, r);
            }
            ui.add_space(6.0);
            ui.horizontal(|ui| {
                ui.strong("Tags");
                if ui.small_button("All").clicked() {
                    self.filters.hidden_tags.clear();
                    changed = true;
                }
                if ui.small_button("None").clicked() {
                    self.filters.hidden_tags = self.all_tags.iter().cloned().collect();
                    changed = true;
                }
            });
            for tag in self.all_tags.clone() {
                changed |= filter_box(ui, &mut self.filters.hidden_tags, tag.clone(), &tag);
            }
        });
        if changed {
            self.rebuild_filter();
        }
    }

    pub(super) fn render_grid(&mut self, ui: &mut egui::Ui, skin: Skin) {
        if self.wallpapers.is_empty() {
            ui.vertical_centered(|ui| {
                ui.add_space(80.0);
                ui.label(
                    egui::RichText::new("No Workshop wallpapers found")
                        .size(18.0)
                        .color(TEXT_MUTED),
                );
                ui.add_space(10.0);
                ui.label(
                    egui::RichText::new(
                        "Install Wallpaper Engine on Steam and subscribe to wallpapers,\n\
                         or use \"Open Wallpaper\" below to show any file.",
                    )
                    .color(TEXT_MUTED),
                );
            });
            return;
        }
        if self.filtered.is_empty() {
            ui.vertical_centered(|ui| {
                ui.add_space(95.0);
                ui.label(
                    egui::RichText::new(
                        "No wallpapers found for your search.\nExpand or reset the filters.",
                    )
                    .size(18.0)
                    .color(TEXT_MUTED),
                );
            });
            return;
        }

        // As many columns as fit at up to `TILE`, stretched to fill the row.
        let width = ui.available_width();
        let cols = (((width + TILE_GAP) / (TILE + TILE_GAP)).ceil() as usize).max(1);
        let tile = (width - TILE_GAP * (cols - 1) as f32) / cols as f32;
        let rows = self.filtered.len().div_ceil(cols);
        let filtered = self.filtered.clone();
        egui::ScrollArea::vertical().auto_shrink(false).show_rows(
            ui,
            tile + TILE_GAP,
            rows,
            |ui, range| {
                for row in range {
                    ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing.x = TILE_GAP;
                        for &idx in filtered.iter().skip(row * cols).take(cols) {
                            self.grid_tile(ui, idx, tile, skin);
                        }
                    });
                    ui.add_space(TILE_GAP);
                }
            },
        );
    }

    pub(super) fn grid_tile(&mut self, ui: &mut egui::Ui, idx: usize, size: f32, skin: Skin) {
        let id = self.wallpapers[idx].workshop_id.clone();
        let thumb = self.thumbnails.get(&id).cloned();
        if thumb.is_none() {
            self.enqueue_thumb(idx);
        }
        let (rect, resp) = ui.allocate_exact_size(vec2(size, size), Sense::click());
        if ui.is_rect_visible(rect) {
            draw_tile(
                ui,
                rect,
                &resp,
                &self.wallpapers[idx],
                thumb.as_ref(),
                self.selected == Some(idx),
                skin,
            );
        }
        let in_playlist = self.playlist_items().contains(&id);
        resp.context_menu(|ui| {
            let label = if in_playlist {
                "Remove from playlist"
            } else {
                "Add to playlist"
            };
            if ui.button(label).clicked() {
                self.toggle_in_playlist(&id);
                ui.close();
            }
            if id.bytes().all(|b| b.is_ascii_digit()) && ui.button("Open Workshop page").clicked() {
                open_external(&workshop_item_url(&id));
                ui.close();
            }
            if ui.button("Open folder").clicked() {
                open_external(&self.wallpapers[idx].path.to_string_lossy());
                ui.close();
            }
        });
        if resp.clicked() {
            self.selected = Some(idx);
            self.apply_wallpaper(idx);
        }
    }
}

pub(super) fn source_tab(
    ui: &mut egui::Ui,
    label: &str,
    active: bool,
    skin: Skin,
) -> egui::Response {
    let (fill, text) = if active {
        (skin.accent, Color32::WHITE)
    } else {
        (skin.panel, skin.text)
    };
    ui.add(
        egui::Button::new(egui::RichText::new(label).size(15.0).color(text))
            .fill(fill)
            .stroke(Stroke::new(1.0, skin.accent))
            .corner_radius(CornerRadius {
                nw: 3,
                ne: 3,
                sw: 0,
                se: 0,
            })
            .min_size(vec2(96.0, 28.0)),
    )
}

/// A ticked-by-default filter checkbox backed by a set of unticked keys.
pub(super) fn filter_box<K: std::hash::Hash + Eq>(
    ui: &mut egui::Ui,
    hidden: &mut HashSet<K>,
    key: K,
    text: &str,
) -> bool {
    let mut on = !hidden.contains(&key);
    if !ui.checkbox(&mut on, text).changed() {
        return false;
    }
    if on {
        hidden.remove(&key);
    } else {
        hidden.insert(key);
    }
    true
}

pub(super) fn draw_tile(
    ui: &egui::Ui,
    rect: Rect,
    resp: &egui::Response,
    w: &Wallpaper,
    thumb: Option<&egui::TextureHandle>,
    selected: bool,
    skin: Skin,
) {
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, CornerRadius::same(2), skin.tile);
    match thumb {
        Some(tex) => {
            painter.image(tex.id(), rect, full_uv(), Color32::WHITE);
        }
        None => {
            painter.text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                "⏳",
                FontId::proportional(22.0),
                TEXT_MUTED,
            );
        }
    }
    if resp.hovered() {
        painter.rect_filled(rect, CornerRadius::same(2), Color32::from_white_alpha(24));
    }

    // Title over a dark band along the bottom, as in WE's grid.
    let band = Rect::from_min_max(pos2(rect.min.x, rect.max.y - 24.0), rect.max);
    painter.rect_filled(band, CornerRadius::ZERO, Color32::from_black_alpha(170));
    let mut job = egui::text::LayoutJob::simple_singleline(
        w.title().to_string(),
        FontId::proportional(12.0),
        Color32::WHITE,
    );
    job.wrap = egui::text::TextWrapping::truncate_at_width(rect.width() - 10.0);
    painter.galley(
        pos2(band.min.x + 5.0, band.min.y + 5.0),
        painter.layout_job(job),
        Color32::WHITE,
    );

    // Type marker in the top corner for everything that isn't a scene.
    if !matches!(w.wallpaper_type(), WallpaperType::Scene) {
        let label = type_name(w.wallpaper_type()).to_uppercase();
        let galley = painter.layout_no_wrap(label, FontId::proportional(9.0), Color32::WHITE);
        let size = galley.size() + vec2(8.0, 4.0);
        let badge = Rect::from_min_size(pos2(rect.max.x - size.x - 4.0, rect.min.y + 4.0), size);
        painter.rect_filled(badge, CornerRadius::same(3), type_color(w.wallpaper_type()));
        painter.galley(badge.min + vec2(4.0, 2.0), galley, Color32::WHITE);
    }

    if selected {
        painter.rect_stroke(
            rect,
            CornerRadius::same(2),
            Stroke::new(3.0, skin.accent),
            egui::StrokeKind::Inside,
        );
    }
}
