//! Inspection and debugging tools: `probe`, `pkg-info`, `tex-info`, `render-scene`, `render-frame`, `test-scene` and `preview-scene`.

use super::*;

pub(super) fn cmd_probe() -> Result<()> {
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

pub(super) fn cmd_pkg_info(path: &std::path::Path, dump: Option<&std::path::Path>) -> Result<()> {
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

pub(super) fn cmd_tex_info(path: &std::path::Path, save: Option<&std::path::Path>) -> Result<()> {
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

pub(super) fn cmd_render_scene(
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

pub(super) fn cmd_render_frame(
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

pub(super) fn cmd_test_scene(id_or_path: &str, num_frames: usize) -> Result<()> {
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

pub(super) fn cmd_preview_scene(id_or_path: &str, width: u32, height: u32) -> Result<()> {
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
