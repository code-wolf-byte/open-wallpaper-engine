use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

#[derive(Debug, Clone, PartialEq)]
enum Status {
    Pass,
    Static,
    Fail,
}

impl std::fmt::Display for Status {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Status::Pass => write!(f, "PASS"),
            Status::Static => write!(f, "STATIC"),
            Status::Fail => write!(f, "FAIL"),
        }
    }
}

#[derive(Debug)]
struct WallpaperResult {
    id: String,
    title: String,
    path: String,
    status: Status,
    errors: Vec<String>,
    result_lines: Vec<String>,
}

fn run_with_timeout(binary: &str, wp_path: &str, timeout: Duration) -> String {
    let child = Command::new(binary)
        .arg("test-scene")
        .arg(wp_path)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn();

    let mut child = match child {
        Ok(c) => c,
        Err(e) => return format!("FAIL: failed to spawn wp-engine: {e}"),
    };

    let deadline = Instant::now() + timeout;
    loop {
        match child.try_wait() {
            Ok(Some(_)) => break,
            Ok(None) => {
                if Instant::now() >= deadline {
                    let _ = child.kill();
                    let partial = match child.wait_with_output() {
                        Ok(out) => format!(
                            "{}\n{}",
                            String::from_utf8_lossy(&out.stdout),
                            String::from_utf8_lossy(&out.stderr)
                        ),
                        Err(_) => String::new(),
                    };
                    return format!("FAIL: timeout after {}s\n{}", timeout.as_secs(), partial);
                }
                thread::sleep(Duration::from_millis(100));
            }
            Err(_) => break,
        }
    }

    match child.wait_with_output() {
        Ok(out) => format!(
            "{}\n{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        ),
        Err(e) => format!("FAIL: {e}"),
    }
}

fn classify(output: &str) -> Status {
    if output.contains("PASS:") {
        Status::Pass
    } else if output.contains("not animating") || output.contains("appears static") {
        Status::Static
    } else {
        Status::Fail
    }
}

fn bucket_errors(output: &str) -> Vec<String> {
    let mut buckets = Vec::new();
    for line in output.lines() {
        let l = line.trim();
        if l.starts_with("warning") {
            continue;
        }
        if l.contains("shaderc") && l.contains("error") {
            buckets.push("glslang_compile".to_string());
        } else if l.contains("naga SPIR-V")
            || l.contains("naga validation")
            || l.contains("naga WGSL")
        {
            buckets.push("naga".to_string());
        } else if l.contains("panicked") || l.contains("panic") {
            buckets.push("panic".to_string());
        } else if l.contains("wgpu error")
            || l.contains("Validation Error")
            || l.contains("pipeline failed")
        {
            buckets.push("wgpu_validation".to_string());
        } else if l.contains("translation failed") {
            buckets.push("translation".to_string());
        } else if l.contains("Buffer is bound") || l.contains("group[") {
            buckets.push("binding_layout".to_string());
        }
    }
    buckets.dedup();
    buckets
}

fn extract_result_lines(output: &str) -> Vec<String> {
    output
        .lines()
        .filter(|l| {
            let l = l.trim();
            l.starts_with("PASS:")
                || l.starts_with("FAIL:")
                || l.contains("not enough frames")
                || l.contains("channel closed")
                || l.contains("Changed pixels")
                || l.contains("not animating")
        })
        .take(5)
        .map(|l| l.to_string())
        .collect()
}

fn discover_wallpapers(workshop_dir: &str) -> Vec<(PathBuf, String)> {
    let mut wallpapers = Vec::new();
    let Ok(entries) = fs::read_dir(workshop_dir) else {
        return wallpapers;
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let pj = path.join("project.json");
        let Ok(content) = fs::read_to_string(&pj) else {
            continue;
        };
        let Ok(json) = serde_json::from_str::<serde_json::Value>(&content) else {
            continue;
        };
        if json.get("type").and_then(|v| v.as_str()) != Some("scene") {
            continue;
        }
        let title = json
            .get("title")
            .and_then(|v| v.as_str())
            .unwrap_or("?")
            .chars()
            .take(60)
            .collect();
        wallpapers.push((path, title));
    }

    wallpapers.sort_by(|a, b| a.0.cmp(&b.0));
    wallpapers
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Binary path: explicit arg or sibling in the same target/ dir
    let wp_engine = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            let mut p = std::env::current_exe().unwrap();
            p.pop();
            p.push("wp-engine");
            p
        });

    if !wp_engine.exists() {
        eprintln!("wp-engine binary not found at {}. Run `cargo build` first or pass the path as argv[1].", wp_engine.display());
        std::process::exit(1);
    }

    let workshop_dir = format!(
        "{}/.steam/steam/steamapps/workshop/content/431960",
        std::env::var("HOME").unwrap_or_default()
    );

    let wallpapers = discover_wallpapers(&workshop_dir);
    let total = wallpapers.len();
    println!("Found {total} scene wallpapers to test.\n");

    let parallelism = thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(8);
    println!("Running with {parallelism} parallel workers.\n");

    // Work queue: Arc<Mutex<Receiver>> shared across worker threads
    let (work_tx, work_rx) = mpsc::channel::<(PathBuf, String)>();
    let work_rx = Arc::new(Mutex::new(work_rx));

    // Results come back on a separate channel to main thread for live printing
    let (res_tx, res_rx) = mpsc::channel::<WallpaperResult>();

    let wp_engine_str = wp_engine.to_string_lossy().to_string();

    let workers: Vec<_> = (0..parallelism)
        .map(|_| {
            let work_rx = Arc::clone(&work_rx);
            let res_tx = res_tx.clone();
            let binary = wp_engine_str.clone();
            thread::spawn(move || loop {
                let task = { work_rx.lock().unwrap().recv() };
                let Ok((path, title)) = task else { break };

                let wp_path = path.to_string_lossy().to_string();
                let id = path
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_string();

                let output = run_with_timeout(&binary, &wp_path, Duration::from_secs(60));
                let status = classify(&output);
                let errors = bucket_errors(&output);
                let result_lines = extract_result_lines(&output);

                let _ = res_tx.send(WallpaperResult {
                    id,
                    title,
                    path: wp_path,
                    status,
                    errors,
                    result_lines,
                });
            })
        })
        .collect();

    // Feed work
    for (path, title) in wallpapers {
        work_tx.send((path, title)).unwrap();
    }
    drop(work_tx); // signals workers to exit when queue is empty
    drop(res_tx); // drop our copy so channel closes when all workers finish

    // Collect results and print progress live
    let mut results = Vec::new();
    for result in res_rx {
        println!(
            "[{:>12}] {:<62} … {}",
            result.id, result.title, result.status
        );
        results.push(result);
    }

    for w in workers {
        let _ = w.join();
    }

    // Sort results by ID for deterministic output
    results.sort_by(|a, b| a.id.cmp(&b.id));

    // ── Write errors.txt ──────────────────────────────────────────────────────
    let out_path = "src/tests/errors.txt";
    let mut out = fs::File::create(out_path)?;

    for r in &results {
        writeln!(
            out,
            "════════════════════════════════════════════════════════════════"
        )?;
        writeln!(out, "ID:     {}", r.id)?;
        writeln!(out, "TITLE:  {}", r.title)?;
        writeln!(out, "PATH:   {}", r.path)?;
        writeln!(out, "STATUS: {}", r.status)?;
        if !r.errors.is_empty() {
            writeln!(out, "ERRORS:")?;
            for e in &r.errors {
                writeln!(out, "  {e}")?;
            }
        }
        if !r.result_lines.is_empty() {
            writeln!(out, "RESULT:")?;
            for l in &r.result_lines {
                writeln!(out, "  {l}")?;
            }
        }
        writeln!(out)?;
    }

    // ── Summary ───────────────────────────────────────────────────────────────
    let pass = results.iter().filter(|r| r.status == Status::Pass).count();
    let static_ = results
        .iter()
        .filter(|r| r.status == Status::Static)
        .count();
    let fail = results.iter().filter(|r| r.status == Status::Fail).count();

    let mut error_counts: HashMap<String, usize> = HashMap::new();
    for r in &results {
        for e in &r.errors {
            *error_counts.entry(e.clone()).or_default() += 1;
        }
    }

    let summary = format!(
        "\n{sep}\nSUMMARY\n{sep}\nTotal scene wallpapers tested : {total}\n  PASS (animating)            : {pass}\n  PASS (static/low change)    : {static_}\n  FAIL (error/crash)          : {fail}\n\nError breakdown:\n{errors}\n",
        sep = "════════════════════════════════════════════════════════════════",
        errors = {
            let mut lines: Vec<String> = error_counts.iter()
                .map(|(k, v)| format!("  {k:<30} : {v}"))
                .collect();
            lines.sort();
            lines.join("\n")
        },
    );

    print!("{summary}");
    write!(out, "{summary}")?;
    println!("\nFull log written to: {out_path}");

    Ok(())
}
