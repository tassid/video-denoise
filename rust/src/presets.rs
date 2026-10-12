//! Settings and presets. `presets.json` is the same file, in the same format, as the PowerShell
//! app writes, so presets saved in one show up in the other.

use std::path::Path;

use serde::{Deserialize, Serialize};

/// `m` = method index, `s` = strength 1..=10, `sh` = sharpen 0..=10, `d` = keep detail 0..=10.
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct Settings {
    pub m: usize,
    pub s: u32,
    pub sh: u32,
    pub d: u32,
    pub soft: bool,
    pub enh: bool,
}

#[derive(Serialize, Deserialize)]
struct Saved {
    name: String,
    #[serde(flatten)]
    settings: Settings,
}

pub const CUSTOM: &str = "Custom";

/// Built-in presets; the first one is the default. Without the AI engine only the last exists.
pub fn builtins(has_ai: bool) -> Vec<(String, Settings)> {
    let preset = |name: &str, m, s, sh, d, soft, enh| (name.to_string(), Settings { m, s, sh, d, soft, enh });
    let mut list = Vec::new();
    if has_ai {
        list.push(preset("Soft / zoomed video - enhance", 2, 2, 1, 8, true, true));
        list.push(preset("Natural - people and faces", 2, 3, 2, 6, false, false));
        list.push(preset("Heavy noise", 2, 6, 2, 5, false, false));
    }
    list.push(preset("Quick (no AI)", 0, 3, 3, 5, false, false));
    list
}

/// The user's own presets. A missing or unreadable file is simply no presets.
pub fn load(path: &Path, methods: usize) -> Vec<(String, Settings)> {
    let Ok(text) = std::fs::read_to_string(path) else { return Vec::new() };
    // PowerShell 5 writes a byte-order mark
    let Ok(saved) = serde_json::from_str::<Vec<Saved>>(text.trim_start_matches('\u{feff}')) else { return Vec::new() };
    saved
        .into_iter()
        .filter(|p| !p.name.trim().is_empty() && p.settings.m < methods)
        .map(|p| {
            let s = p.settings;
            (p.name, Settings { s: s.s.clamp(1, 10), sh: s.sh.min(10), d: s.d.min(10), ..s })
        })
        .collect()
}

pub fn save(path: &Path, user: &[(String, Settings)]) -> Result<(), String> {
    let rows: Vec<Saved> = user.iter().map(|(name, settings)| Saved { name: name.clone(), settings: settings.clone() }).collect();
    let json = serde_json::to_string_pretty(&rows).map_err(|e| e.to_string())?;
    std::fs::write(path, json).map_err(|e| e.to_string())
}
