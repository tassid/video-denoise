//! Video Denoise: drag-and-drop window around ffmpeg and two neural networks.
//! Rust port of `VideoDenoise.ps1`; the processing lives in `engine.rs`.
//!
//! Headless test, through the real window:
//!     video-denoise --autorun clip.mp4 [--at 0:05] [--opts "test,s=2,d=8"] [--log out.txt] [--shot ui.png]
//! runs a preview and the job, saves a picture of the window, and exits with the number of failures.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod engine;
mod presets;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
use std::sync::mpsc::{Receiver, Sender};
use std::time::{Duration, Instant};

use engine::{ENCODERS, Event, METHODS, Request};
use presets::{CUSTOM, Settings};

static EXIT_CODE: AtomicI32 = AtomicI32::new(0);

const VIDEO_EXTENSIONS: [&str; 9] = ["mp4", "mov", "mkv", "avi", "mts", "m2ts", "webm", "wmv", "mxf"];

/// The `--autorun` script: which step it is at, and where its results go.
struct Auto {
    file: PathBuf,
    at: String,
    opts: String,
    log: Option<PathBuf>,
    shot: Option<PathBuf>,
    stage: u8,
    waiting_since: Instant,
}

struct App {
    requests: Sender<Request>,
    events: Receiver<Event>,
    cancel: Arc<AtomicBool>,
    preset_file: PathBuf,
    has_ai: bool,

    files: Vec<PathBuf>,
    selected: usize,
    /// Built-in presets first (`builtin` of them), then the user's.
    presets: Vec<(String, Settings)>,
    builtin: usize,
    settings: Settings,
    encoder: usize,
    test: bool,
    preview_at: String,

    busy: bool,
    previewing: bool,
    progress: f32,
    status: String,
    log: String,
    failed: usize,
    last_output: Option<PathBuf>,
    preview: Option<(egui::TextureHandle, String)>,
    /// Text of the "Save preset" box while it is open.
    save_name: Option<String>,
    /// Index of the preset whose Delete was clicked once.
    confirm_delete: Option<usize>,
    auto: Option<Auto>,
}

impl App {
    fn new(ctx: egui::Context, auto: Option<Auto>) -> Self {
        let root = find_root();
        let models = root.join("ai").join("models");
        let has_ai = models.join(engine::DENOISER).is_file() && models.join(engine::ENHANCER).is_file();
        let preset_file = root.join("presets.json");
        let mut all = presets::builtins(has_ai);
        let builtin = all.len();
        let methods = if has_ai { METHODS.len() } else { METHODS.len() - 1 };
        for preset in presets::load(&preset_file, methods) {
            if !all.iter().any(|(name, _)| *name == preset.0) {
                all.push(preset);
            }
        }
        let cancel = Arc::new(AtomicBool::new(false));
        let (requests, events) = engine::spawn(root, ctx, cancel.clone());
        let ffmpeg = engine::tool("ffmpeg").arg("-version").output().is_ok_and(|out| out.status.success());
        App {
            requests,
            events,
            cancel,
            preset_file,
            has_ai,
            files: Vec::new(),
            selected: 0,
            settings: all[0].1.clone(), // first preset is the default
            presets: all,
            builtin,
            encoder: 0,
            test: false,
            preview_at: String::new(),
            busy: false,
            previewing: false,
            progress: 0.0,
            status: if ffmpeg { "Ready.".into() } else { "ffmpeg not found in PATH. Install with: winget install Gyan.FFmpeg".into() },
            log: String::new(),
            failed: 0,
            last_output: None,
            preview: None,
            save_name: None,
            confirm_delete: None,
            auto,
        }
    }

    /// The preset whose settings are the current ones, if any; otherwise the box shows "Custom".
    fn preset_index(&self) -> Option<usize> {
        self.presets.iter().position(|(_, settings)| *settings == self.settings)
    }

    fn add_files(&mut self, paths: impl IntoIterator<Item = PathBuf>) {
        for path in paths {
            if path.is_file() && !self.files.contains(&path) {
                self.files.push(path);
            }
        }
    }

    fn start(&mut self) {
        if self.files.is_empty() {
            self.status = "Add a video first.".into();
            return;
        }
        self.cancel.store(false, Ordering::Relaxed);
        self.busy = true;
        self.failed = 0;
        let request = Request::Run { files: self.files.clone(), settings: self.settings.clone(), encoder: self.encoder, test: self.test };
        let _ = self.requests.send(request);
    }

    fn request_preview(&mut self) {
        let Some(file) = self.files.get(self.selected).or(self.files.first()).cloned() else {
            self.status = "Add a video first.".into();
            return;
        };
        let at = if self.preview_at.trim().is_empty() {
            None
        } else {
            match parse_time(&self.preview_at) {
                Some(seconds) => Some(seconds),
                None => {
                    self.status = "Preview time not understood. Use 90, 1:30 or 0:01:30.".into();
                    return;
                }
            }
        };
        self.previewing = true;
        self.status = "Rendering preview...".into();
        let _ = self.requests.send(Request::Preview { file, at, settings: self.settings.clone() });
    }

    /// Stores the current settings under `name`. Built-in names cannot be overwritten.
    fn save_preset(&mut self, name: &str) -> Result<(), String> {
        let name = name.trim();
        if name.is_empty() {
            return Err("Give the preset a name.".into());
        }
        if name == CUSTOM || self.presets[..self.builtin].iter().any(|(n, _)| n == name) {
            return Err(format!("'{name}' is a built-in name. Pick another."));
        }
        match self.presets.iter_mut().find(|(n, _)| n == name) {
            Some(existing) => existing.1 = self.settings.clone(),
            None => self.presets.push((name.to_string(), self.settings.clone())),
        }
        presets::save(&self.preset_file, &self.presets[self.builtin..]).map_err(|e| format!("Could not write presets.json: {e}"))
    }

    fn delete_preset(&mut self, index: usize) {
        if index >= self.builtin && index < self.presets.len() {
            let (name, _) = self.presets.remove(index);
            let _ = presets::save(&self.preset_file, &self.presets[self.builtin..]);
            self.status = format!("Preset '{name}' deleted.");
        }
    }

    fn poll(&mut self, ctx: &egui::Context) {
        while let Ok(event) = self.events.try_recv() {
            match event {
                Event::Log(text) => {
                    self.log.push_str(&text);
                    self.log.push('\n');
                }
                Event::Progress { fraction, text } => {
                    self.progress = fraction;
                    self.status = text;
                }
                Event::Preview { width, height, rgb, caption } => {
                    let image = egui::ColorImage::from_rgb([width, height], &rgb);
                    self.preview = Some((ctx.load_texture("preview", image, egui::TextureOptions::LINEAR), caption));
                    self.previewing = false;
                    self.status = "Ready.".into();
                }
                Event::PreviewFailed(error) => {
                    self.previewing = false;
                    self.status = error;
                }
                Event::Finished { failed, cancelled, last_output } => {
                    self.busy = false;
                    self.progress = 0.0;
                    self.failed = failed;
                    self.status = match (cancelled, failed) {
                        (true, _) => "Cancelled.".into(),
                        (false, 0) => "All done.".into(),
                        (false, n) => format!("Finished with {n} error(s)."),
                    };
                    if last_output.is_some() {
                        self.last_output = last_output;
                    }
                }
            }
        }
    }

    fn controls(&mut self, ui: &mut egui::Ui) {
        ui.add_space(6.0);
        let drop = ui.add_sized([ui.available_width(), 72.0], egui::Button::new(egui::RichText::new("Drop videos here\n(or click to browse)").size(16.0)));
        if drop.clicked()
            && let Some(picked) = rfd::FileDialog::new().add_filter("Video files", &VIDEO_EXTENSIONS).add_filter("All files", &["*"]).pick_files()
        {
            self.add_files(picked);
        }

        egui::Frame::group(ui.style()).show(ui, |ui| {
            ui.set_width(ui.available_width());
            egui::ScrollArea::vertical().id_salt("files").max_height(84.0).min_scrolled_height(84.0).show(ui, |ui| {
                ui.set_min_height(84.0);
                let mut remove = None;
                for (index, file) in self.files.iter().enumerate() {
                    ui.horizontal(|ui| {
                        if ui.add_enabled(!self.busy, egui::Button::new("x").small()).on_hover_text("Remove from the list").clicked() {
                            remove = Some(index);
                        }
                        if ui.selectable_label(self.selected == index, engine::file_name(file)).on_hover_text(file.display().to_string()).clicked() {
                            self.selected = index;
                        }
                    });
                }
                if let Some(index) = remove {
                    self.files.remove(index);
                    self.selected = self.selected.min(self.files.len().saturating_sub(1));
                }
            });
        });
        ui.add_space(4.0);

        let ai = self.has_ai && self.settings.m == 2;
        ui.add_enabled_ui(!self.busy, |ui| {
            ui.spacing_mut().slider_width = 190.0;
            egui::Grid::new("settings").num_columns(2).spacing([14.0, 9.0]).show(ui, |ui| {
                ui.label("Preset");
                ui.horizontal(|ui| {
                    let current = self.preset_index();
                    let name = current.map_or(CUSTOM.to_string(), |i| self.presets[i].0.clone());
                    egui::ComboBox::from_id_salt("preset").width(206.0).selected_text(name.as_str()).show_ui(ui, |ui| {
                        for index in 0..self.presets.len() {
                            if ui.selectable_label(current == Some(index), self.presets[index].0.as_str()).clicked() {
                                self.settings = self.presets[index].1.clone();
                            }
                        }
                    });
                    let own = current.filter(|&i| i >= self.builtin);
                    if ui.button("Save...").on_hover_text("Save the current settings as a preset of your own").clicked() {
                        self.save_name = Some(own.map(|_| name.clone()).unwrap_or_default());
                    }
                    if self.confirm_delete != own {
                        self.confirm_delete = None;
                    }
                    let label = if self.confirm_delete.is_some() { "Really?" } else { "Delete" };
                    if ui.add_enabled(own.is_some(), egui::Button::new(label)).clicked() {
                        match self.confirm_delete.take() {
                            Some(index) => self.delete_preset(index),
                            None => self.confirm_delete = own,
                        }
                    }
                });
                ui.end_row();

                ui.label("Method");
                ui.horizontal(|ui| {
                    let methods = if self.has_ai { METHODS.len() } else { METHODS.len() - 1 };
                    egui::ComboBox::from_id_salt("method").width(206.0).selected_text(METHODS[self.settings.m.min(2)]).show_ui(ui, |ui| {
                        for (index, name) in METHODS.iter().enumerate().take(methods) {
                            ui.selectable_value(&mut self.settings.m, index, *name);
                        }
                    });
                    ui.checkbox(&mut self.test, "Test: first 10 s only");
                });
                ui.end_row();

                ui.label("Strength");
                ui.horizontal(|ui| {
                    ui.add(egui::Slider::new(&mut self.settings.s, 1..=10));
                    ui.weak(match self.settings.s {
                        0..=3 => "light",
                        4..=6 => "medium",
                        7..=8 => "strong",
                        _ => "very strong",
                    });
                });
                ui.end_row();

                ui.label("Sharpen");
                ui.horizontal(|ui| {
                    ui.add(egui::Slider::new(&mut self.settings.sh, 0..=10));
                    if self.settings.sh == 0 {
                        ui.weak("off");
                    }
                });
                ui.end_row();

                ui.label("Keep detail");
                ui.horizontal(|ui| {
                    ui.add(egui::Slider::new(&mut self.settings.d, 0..=10));
                    if self.settings.d == 0 {
                        ui.weak("off");
                    }
                });
                ui.end_row();

                ui.label("Encoder");
                ui.horizontal(|ui| {
                    egui::ComboBox::from_id_salt("encoder").width(206.0).selected_text(ENCODERS[self.encoder].0).show_ui(ui, |ui| {
                        for (index, (name, _)) in ENCODERS.iter().enumerate() {
                            ui.selectable_value(&mut self.encoder, index, *name);
                        }
                    });
                    ui.label("Preview at");
                    ui.add(egui::TextEdit::singleline(&mut self.preview_at).desired_width(52.0).hint_text("m:ss"))
                        .on_hover_text("Where to take the preview frame: 90, 1:30 or 0:01:30. Empty = middle of the video.");
                });
                ui.end_row();
            });
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                ui.add_enabled(ai, egui::Checkbox::new(&mut self.settings.soft, "Soft / zoomed source")).on_hover_text(
                    "For footage that was cropped/zoomed and looks soft with coarse grain: denoises at half size, where the grain is fine, then scales back. AI only.",
                );
                ui.add_enabled(ai, egui::Checkbox::new(&mut self.settings.enh, "Enhance - AI rebuilds detail"))
                    .on_hover_text("Makes edges crisp by reconstructing detail. It invents detail: fine on objects, can subtly change real faces. AI only.");
            });
        });

        ui.add_space(8.0);
        ui.horizontal(|ui| {
            let size = [78.0, 30.0];
            if ui.add_enabled(!self.busy && !self.previewing, egui::Button::new("Start").min_size(size.into())).clicked() {
                self.start();
            }
            if ui.add_enabled(self.busy, egui::Button::new("Cancel").min_size(size.into())).clicked() {
                self.cancel.store(true, Ordering::Relaxed);
                self.status = "Cancelling...".into();
            }
            if ui.add_enabled(!self.busy, egui::Button::new("Clear list").min_size(size.into())).clicked() {
                self.files.clear();
                self.selected = 0;
                self.status = "Ready.".into();
            }
            if ui.add_enabled(!self.busy && !self.previewing, egui::Button::new("Preview").min_size(size.into())).clicked() {
                self.request_preview();
            }
            if ui.add_enabled(self.last_output.is_some(), egui::Button::new("Show output").min_size(size.into())).clicked()
                && let Some(output) = &self.last_output
            {
                let _ = std::process::Command::new("explorer.exe").arg(format!("/select,{}", output.display())).spawn();
            }
        });
        ui.add_space(6.0);
        ui.add(egui::ProgressBar::new(self.progress).desired_height(14.0));
        ui.label(self.status.as_str());
        egui::Frame::group(ui.style()).show(ui, |ui| {
            ui.set_width(ui.available_width());
            egui::ScrollArea::vertical().id_salt("log").stick_to_bottom(true).auto_shrink([false, false]).show(ui, |ui| {
                ui.add(egui::Label::new(egui::RichText::new(self.log.as_str()).monospace().size(11.0)).wrap());
            });
        });
    }

    fn preview_panel(&mut self, ui: &mut egui::Ui) {
        let Some((texture, caption)) = &self.preview else {
            ui.centered_and_justified(|ui| {
                ui.weak(if self.previewing { "Rendering preview..." } else { "Preview shows here: original on the left, result on the right, at 100%." });
            });
            return;
        };
        ui.label(caption.as_str());
        let (available, size) = (ui.available_size(), texture.size_vec2());
        let scale = (available.x / size.x).min((available.y - 20.0) / size.y).min(1.0).max(0.05);
        if scale < 0.999 {
            ui.weak(format!("shown at {:.0}% - enlarge the window to see it at 100%", scale * 100.0));
        }
        ui.image((texture.id(), size * scale));
    }

    fn save_dialog(&mut self, ctx: &egui::Context) {
        let Some(name) = self.save_name.as_mut() else { return };
        let mut decided = None;
        egui::Window::new("Save preset").collapsible(false).resizable(false).anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0]).show(ctx, |ui| {
            ui.label("Name for the current settings (a name of yours that exists is overwritten):");
            let edit = ui.text_edit_singleline(name);
            if !edit.has_focus() && !edit.lost_focus() {
                edit.request_focus();
            }
            let entered = edit.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
            ui.horizontal(|ui| {
                if ui.button("Save").clicked() || entered {
                    decided = Some(true);
                }
                if ui.button("Cancel").clicked() {
                    decided = Some(false);
                }
            });
        });
        match decided {
            Some(true) => {
                let name = self.save_name.take().unwrap_or_default();
                self.status = match self.save_preset(&name) {
                    Ok(()) => format!("Preset '{}' saved.", name.trim()),
                    Err(error) => error,
                };
            }
            Some(false) => self.save_name = None,
            None => {}
        }
    }

    /// One step of the `--autorun` script per frame.
    fn autorun(&mut self, ctx: &egui::Context) {
        let Some(stage) = self.auto.as_ref().map(|auto| auto.stage) else { return };
        ctx.request_repaint_after(Duration::from_millis(50));
        let next = match stage {
            0 => {
                let (file, at, opts) = self.auto.as_ref().map(|a| (a.file.clone(), a.at.clone(), a.opts.clone())).unwrap_or_default();
                self.add_files([file]);
                self.preview_at = at;
                for opt in opts.split(',').map(str::trim) {
                    let number = |prefix: &str| opt.strip_prefix(prefix).and_then(|v| v.parse::<u32>().ok());
                    match opt {
                        "soft" => self.settings.soft = true,
                        "enhance" => self.settings.enh = true,
                        "test" => self.test = true,
                        _ => {
                            if let Some(v) = number("s=") {
                                self.settings.s = v.clamp(1, 10);
                            } else if let Some(v) = number("sh=") {
                                self.settings.sh = v.min(10);
                            } else if let Some(v) = number("d=") {
                                self.settings.d = v.min(10);
                            } else if let Some(v) = number("m=") {
                                self.settings.m = (v as usize).min(2);
                            } else if let Some(name) = opt.strip_prefix("p=") {
                                if let Some((_, settings)) = self.presets.iter().find(|(n, _)| n == name) {
                                    self.settings = settings.clone();
                                }
                            } else if let Some(name) = opt.strip_prefix("save=") {
                                let result = self.save_preset(name).err().unwrap_or_default();
                                self.log.push_str(&format!("save: [{result}]\n"));
                            } else if let Some(name) = opt.strip_prefix("del=")
                                && let Some(index) = self.presets.iter().position(|(n, _)| n == name)
                            {
                                self.delete_preset(index);
                            }
                        }
                    }
                }
                self.request_preview();
                1
            }
            1 if !self.previewing => {
                self.log.push_str(&format!("preview: {}\n", self.preview.as_ref().map_or(self.status.as_str(), |(_, caption)| caption.as_str())));
                self.start();
                2
            }
            2 if !self.busy => {
                ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(egui::UserData::default()));
                3
            }
            3 => {
                let shot = ctx.input(|i| {
                    i.raw.events.iter().find_map(|event| match event {
                        egui::Event::Screenshot { image, .. } => Some(image.clone()),
                        _ => None,
                    })
                });
                let waited = self.auto.as_ref().is_some_and(|a| a.waiting_since.elapsed() > Duration::from_secs(5));
                if shot.is_none() && !waited {
                    3
                } else {
                    if let Some(auto) = &self.auto {
                        if let (Some(image), Some(path)) = (&shot, &auto.shot) {
                            save_png(image, path);
                        }
                        if let Some(path) = &auto.log {
                            let _ = std::fs::write(path, format!("{}status: {}\n", self.log, self.status));
                        }
                    }
                    EXIT_CODE.store(self.failed as i32, Ordering::Relaxed);
                    ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    4
                }
            }
            same => same,
        };
        if let Some(auto) = self.auto.as_mut()
            && auto.stage != next
        {
            auto.stage = next;
            auto.waiting_since = Instant::now();
        }
    }
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        self.poll(&ctx);
        self.autorun(&ctx);
        let dropped: Vec<PathBuf> =
            ctx.input(|i| i.raw.dropped_files.iter().map(|file| file.path().to_string_lossy().to_string()).filter(|path| !path.is_empty()).map(PathBuf::from).collect());
        self.add_files(dropped);

        egui::Panel::left("controls").exact_size(470.0).resizable(false).show(ui, |ui| self.controls(ui));
        egui::CentralPanel::default().show(ui, |ui| self.preview_panel(ui));
        self.save_dialog(&ctx);
    }
}

impl Drop for App {
    /// Closing the window mid-job: stop it and give ffmpeg a moment to go, so no half-written
    /// output is left behind.
    fn drop(&mut self) {
        if self.busy {
            self.cancel.store(true, Ordering::Relaxed);
            let deadline = Instant::now() + Duration::from_secs(5);
            while Instant::now() < deadline {
                if let Ok(Event::Finished { .. }) = self.events.recv_timeout(Duration::from_millis(200)) {
                    break;
                }
            }
        }
    }
}

/// The folder that holds `ai/models` and `presets.json`: the first one found going up from the program.
fn find_root() -> PathBuf {
    let exe = std::env::current_exe().ok();
    let mut dir = exe.as_deref().and_then(Path::parent);
    while let Some(folder) = dir {
        if folder.join("ai").join("models").is_dir() {
            return folder.to_path_buf();
        }
        dir = folder.parent();
    }
    std::env::current_dir().unwrap_or_default()
}

/// "90", "1:30" or "0:01:30" (comma or dot decimals) to seconds.
fn parse_time(text: &str) -> Option<f64> {
    let mut total = 0.0;
    for part in text.trim().replace(',', ".").split(':') {
        let value: f64 = part.parse().ok()?;
        if !value.is_finite() || value < 0.0 {
            return None;
        }
        total = total * 60.0 + value;
    }
    Some(total)
}

/// Writes the window picture through ffmpeg, which is already a requirement, instead of pulling in an image crate.
fn save_png(image: &egui::ColorImage, path: &Path) {
    use std::io::Write;
    let size = format!("{}x{}", image.size[0], image.size[1]);
    let child = engine::tool("ffmpeg")
        .args(["-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", &size, "-i", "pipe:0", "-frames:v", "1"])
        .arg(path)
        .stdin(std::process::Stdio::piped())
        .spawn();
    if let Ok(mut child) = child {
        if let Some(mut stdin) = child.stdin.take() {
            let _ = stdin.write_all(image.as_raw());
        }
        let _ = child.wait();
    }
}

fn main() -> eframe::Result {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let value = |flag: &str| args.iter().position(|a| a == flag).and_then(|i| args.get(i + 1)).cloned();
    let auto = value("--autorun").map(|file| Auto {
        file: PathBuf::from(file),
        at: value("--at").unwrap_or_default(),
        opts: value("--opts").unwrap_or_default(),
        log: value("--log").map(PathBuf::from),
        shot: value("--shot").map(PathBuf::from),
        stage: 0,
        waiting_since: Instant::now(),
    });

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_title("Video Denoise").with_inner_size([1780.0, 800.0]).with_min_inner_size([900.0, 660.0]).with_drag_and_drop(true),
        ..Default::default()
    };
    let result = eframe::run_native("Video Denoise", options, Box::new(move |cc| Ok(Box::new(App::new(cc.egui_ctx.clone(), auto)))));
    let code = EXIT_CODE.load(Ordering::Relaxed);
    if code != 0 {
        std::process::exit(code);
    }
    result
}
