//! Browsing what is installed: `list`, `info` and `list-properties`.

use super::*;

pub(super) fn cmd_list(type_filter: Option<String>) -> Result<()> {
    let wallpapers = workshop::scan_wallpapers();

    if wallpapers.is_empty() {
        println!("No Wallpaper Engine Workshop items found.");
        println!();
        println!(
            "Expected path: ~/.local/share/Steam/steamapps/workshop/content/{}/",
            workshop::WALLPAPER_ENGINE_APP_ID
        );
        return Ok(());
    }

    let items: Vec<&Wallpaper> = wallpapers
        .iter()
        .filter(|w| {
            type_filter
                .as_deref()
                .map(|f| w.wallpaper_type().to_string() == f)
                .unwrap_or(true)
        })
        .collect();

    println!("{:<20} {:<12} {}", "ID", "Type", "Title");
    println!("{}", "─".repeat(64));
    for w in &items {
        println!(
            "{:<20} {:<12} {}",
            w.workshop_id,
            w.wallpaper_type(),
            w.title()
        );
    }
    println!("\n{} wallpaper(s)", items.len());
    Ok(())
}

pub(super) fn cmd_info(id: &str) -> Result<()> {
    let w = workshop::find_by_id(id).ok_or_else(|| anyhow!("workshop item '{id}' not found"))?;

    println!("ID:          {}", w.workshop_id);
    println!("Title:       {}", w.title());
    println!("Type:        {}", w.wallpaper_type());
    println!("Path:        {}", w.path.display());
    if let Some(f) = w.wallpaper_file() {
        println!(
            "File:        {} {}",
            f.display(),
            if f.exists() { "" } else { "(MISSING)" }
        );
    }
    if let Some(tags) = &w.project.tags {
        if !tags.is_empty() {
            println!("Tags:        {}", tags.join(", "));
        }
    }
    if let Some(rating) = &w.project.contentrating {
        if !rating.is_empty() {
            println!("Rating:      {rating}");
        }
    }
    if let Some(description) = &w.project.description {
        if !description.is_empty() {
            println!("Description: {}", description.replace('\n', " "));
        }
    }
    Ok(())
}

pub(super) fn cmd_list_properties(id: &str) -> Result<()> {
    let w = workshop::find_by_id(id).ok_or_else(|| anyhow!("workshop item '{id}' not found"))?;
    let all_props = workshop::properties::list_properties(&w.path)?;
    if all_props.is_empty() {
        println!(
            "\"{}\" declares no user-configurable properties.",
            w.title()
        );
        return Ok(());
    }
    // Decorative declarations (HTML separators/section headers — real
    // content ships plenty of these) aren't things a user can actually
    // set; listing them as if they were is misleading, not just noisy.
    let (settable, decorative): (Vec<_>, Vec<_>) =
        all_props.into_iter().partition(|p| p.is_settable());

    if settable.is_empty() {
        println!(
            "\"{}\" declares no settable properties ({} decorative label(s) only).",
            w.title(),
            decorative.len()
        );
        return Ok(());
    }

    println!("{:<28} {:<8} {:<24} {}", "Name", "Type", "Default", "Label");
    println!("{}", "─".repeat(80));
    for p in &settable {
        println!("{:<28} {:<8} {:<24} {}", p.name, p.kind, p.value, p.text);
        if p.kind == "combo" && !p.options.is_empty() {
            let choices = p
                .options
                .iter()
                .map(|(value, label)| format!("{value}={label}"))
                .collect::<Vec<_>>()
                .join(", ");
            println!("{:<28} {:<8} choices: {choices}", "", "");
        } else if p.kind == "slider" {
            if let (Some(min), Some(max)) = (p.min, p.max) {
                let step = p.step.map(|s| format!(", step {s}")).unwrap_or_default();
                println!("{:<28} {:<8} range: {min}..={max}{step}", "", "");
            }
        }
    }
    if !decorative.is_empty() {
        println!(
            "\n({} decorative label(s) not shown — no real value to set)",
            decorative.len()
        );
    }
    println!("\nOverride with: wp-engine set {id} --set-property NAME=VALUE");
    Ok(())
}
