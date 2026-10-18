//! Thumbnail loading: square, centre-cropped previews decoded a few per frame
//! so scrolling stays smooth.

use super::*;

impl WpApp {
    pub(super) fn enqueue_thumb(&mut self, idx: usize) {
        let id = &self.wallpapers[idx].workshop_id;
        if !self.thumbnails.contains_key(id) && !self.thumb_queue.contains(&idx) {
            self.thumb_queue.push(idx);
        }
    }

    pub(super) fn flush_thumb_queue(&mut self, ctx: &egui::Context) {
        // Load at most 4 thumbnails per frame to keep frame times smooth
        let batch: Vec<usize> = self
            .thumb_queue
            .drain(..self.thumb_queue.len().min(4))
            .collect();
        for idx in batch {
            self.load_thumb(idx, ctx);
        }
    }

    pub(super) fn load_thumb(&mut self, idx: usize, ctx: &egui::Context) {
        let w = &self.wallpapers[idx];
        let key = w.workshop_id.clone();
        if self.thumbnails.contains_key(&key) {
            return;
        }

        // Try preview file first, then look for any image in the directory
        let path = w.preview_file().or_else(|| {
            std::fs::read_dir(&w.path).ok()?.find_map(|e| {
                let p = e.ok()?.path();
                let ext = p.extension()?.to_str()?.to_lowercase();
                matches!(ext.as_str(), "jpg" | "jpeg" | "png").then_some(p)
            })
        });

        if let Some(p) = path {
            if let Ok(img) = image::open(&p) {
                // Square tiles, centre-cropped like WE's grid.
                let img = img.into_rgba8();
                let side = img.width().min(img.height());
                let square = image::imageops::crop_imm(
                    &img,
                    (img.width() - side) / 2,
                    (img.height() - side) / 2,
                    side,
                    side,
                )
                .to_image();
                let thumb = image::imageops::thumbnail(&square, 256, 256);
                let size = [thumb.width() as usize, thumb.height() as usize];
                let ci = egui::ColorImage::from_rgba_unmultiplied(size, &thumb.into_raw());
                let handle = ctx.load_texture(&key, ci, egui::TextureOptions::LINEAR);
                self.thumbnails.insert(key, handle);
            }
        }
    }
}
