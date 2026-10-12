//! The work behind the window: a thread that owns the two networks and drives ffmpeg.
//!
//! ffmpeg decodes to raw RGB on a pipe -> each frame goes through the networks (ONNX Runtime,
//! DirectML) -> ffmpeg encodes from a pipe. Resizing and colour conversion are left to ffmpeg,
//! so this code only ever sees RGB frames at the size the networks should work at.
//!
//! Same processing as `ai/ai_denoise.py`; the networks stay loaded between requests, which is
//! what makes previews quick here.

use std::fs::File;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::time::Instant;

use ort::ep::DirectML;
use ort::ep::directml::PerformancePreference;
use ort::session::Session;
use ort::value::Tensor;

use crate::presets::Settings;

pub const METHODS: [&str; 3] = ["Fast", "Best quality (slow)", "AI (best, GPU)"];
pub const ENCODERS: [(&str, &str); 2] = [
    ("H.264 - CPU, most compatible", "-c:v libx264 -crf 18 -preset slow"),
    ("HEVC - NVIDIA GPU, faster", "-c:v hevc_nvenc -preset p6 -rc vbr -cq 20 -b:v 0"),
];
pub const DENOISER: &str = "drunet_color.onnx";
pub const ENHANCER: &str = "realesr_general_x4v3.onnx";

/// The enhancer returns 16x the pixels it is given; its input is kept to about 1080p.
const ENHANCE_MAX_PIXELS: f64 = 2_100_000.0;
/// Smoothed-luma gradient at which an area counts as fully textured.
const DETAIL_KNEE: f32 = 0.02;
/// Share of what the network removed that "keep detail" at 10 returns in flat / textured areas.
const DETAIL_FLAT: f32 = 0.12;
const DETAIL_TEXTURE: f32 = 0.9;
/// Preview crop, taken from the centre of the frame and shown at 100%.
const PREVIEW_W: usize = 640;
const PREVIEW_H: usize = 720;

pub enum Request {
    Run { files: Vec<PathBuf>, settings: Settings, encoder: usize, test: bool },
    Preview { file: PathBuf, at: Option<f64>, settings: Settings },
}

pub enum Event {
    Log(String),
    Progress { fraction: f32, text: String },
    /// Before (left half) and after (right half), RGB8.
    Preview { width: usize, height: usize, rgb: Vec<u8>, caption: String },
    PreviewFailed(String),
    Finished { failed: usize, cancelled: bool, last_output: Option<PathBuf> },
}

pub fn describe(s: &Settings) -> String {
    let mut text = format!("{}, strength {}, sharpen {}, detail {}", METHODS[s.m.min(2)], s.s, s.sh, s.d);
    if s.m == 2 && s.soft {
        text += ", soft source";
    }
    if s.m == 2 && s.enh {
        text += ", enhance";
    }
    text
}

pub fn spawn(root: PathBuf, ctx: egui::Context, cancel: Arc<AtomicBool>) -> (Sender<Request>, Receiver<Event>) {
    let (requests, inbox) = mpsc::channel::<Request>();
    let (events, outbox) = mpsc::channel::<Event>();
    std::thread::spawn(move || {
        let mut worker = Worker { ctx, cancel, events, networks: Networks { models: root.join("ai").join("models"), denoiser: None, enhancer: None, spent: [0.0; 3] } };
        for request in inbox {
            match request {
                Request::Run { files, settings, encoder, test } => worker.run_all(&files, &settings, encoder, test),
                Request::Preview { file, at, settings } => {
                    let started = Instant::now();
                    let event = match worker.preview(&file, at, &settings) {
                        Ok((width, height, rgb, at)) => {
                            let caption = format!(
                                "{}  @ {}  -  left: original  |  right: {}  -  {:.1} s",
                                file_name(&file),
                                clock(at),
                                describe(&settings),
                                started.elapsed().as_secs_f32()
                            );
                            Event::Preview { width, height, rgb, caption }
                        }
                        Err(error) => Event::PreviewFailed(error),
                    };
                    worker.emit(event);
                }
            }
        }
    });
    (requests, outbox)
}

struct Worker {
    ctx: egui::Context,
    cancel: Arc<AtomicBool>,
    events: Sender<Event>,
    networks: Networks,
}

/// A network loaded for one picture size `(height, width)`, from one model file.
struct Loaded {
    size: (usize, usize),
    file: &'static str,
    session: Session,
}

struct Networks {
    models: PathBuf,
    denoiser: Option<Loaded>,
    enhancer: Option<Loaded>,
    /// Seconds in the current job spent in: denoiser, detail, enhancer.
    spent: [f64; 3],
}

/// One frame, unpacked for the networks: planar RGB 0..=1 at the work size, and the denoiser's
/// input (the same, padded, plus the noise-level plane).
struct Unpacked {
    original: Vec<f32>,
    input: Vec<f32>,
}

/// What ffprobe says about the first video stream, as ffmpeg will hand it over (rotation applied).
struct Video {
    w: usize,
    h: usize,
    fps: String,
    fps_f: f64,
    duration: f64,
    /// Name of the YUV matrix for ffmpeg's scale filter, and the luma weights that go with it.
    matrix: &'static str,
    luma: (f32, f32),
    /// The same matrix as a `-colorspace` tag.
    tag: String,
    range: &'static str,
    trc: String,
    primaries: String,
    pix: &'static str,
}

impl Worker {
    fn emit(&self, event: Event) {
        let _ = self.events.send(event);
        self.ctx.request_repaint();
    }

    fn log(&self, text: impl Into<String>) {
        self.emit(Event::Log(text.into()));
    }

    fn cancelled(&self) -> bool {
        self.cancel.load(Ordering::Relaxed)
    }

    fn run_all(&mut self, files: &[PathBuf], settings: &Settings, encoder: usize, test: bool) {
        let (mut failed, mut cancelled, mut last_output) = (0, false, None);
        for (index, file) in files.iter().enumerate() {
            if self.cancelled() {
                cancelled = true;
                break;
            }
            let output = output_path(file, test);
            let label = format!("{}/{}  {}", index + 1, files.len(), file_name(file));
            self.log(format!("[{}/{}] {}  ({})", index + 1, files.len(), file_name(file), describe(settings)));
            self.emit(Event::Progress { fraction: 0.0, text: format!("{label}  -  starting...") });
            let result = if settings.m == 2 {
                self.run_ai(file, &output, settings, encoder, test, &label)
            } else {
                self.run_classic(file, &output, settings, encoder, test, &label)
            };
            match result {
                Ok(()) => {
                    self.log(format!("  OK -> {}", output.display()));
                    last_output = Some(output);
                }
                Err(_) if self.cancelled() => {
                    let _ = std::fs::remove_file(&output);
                    self.log("  cancelled.");
                    cancelled = true;
                    break;
                }
                Err(error) => {
                    failed += 1;
                    let _ = std::fs::remove_file(&output);
                    self.log(format!("  FAILED: {error}"));
                }
            }
        }
        self.emit(Event::Finished { failed, cancelled, last_output });
    }

    fn run_ai(&mut self, input: &Path, output: &Path, settings: &Settings, encoder: usize, test: bool, label: &str) -> Result<(), String> {
        let v = probe(input)?;
        let work = work_size(&v, v.w, v.h, settings);
        let (ww, wh) = work;
        let factor = if settings.enh { self.networks.enhance_plan(work, (v.w, v.h)).1 } else { 1 };
        let (ow, oh) = (ww * factor, wh * factor);
        let limit: &[&str] = if test { &["-t", "10"] } else { &[] };
        let report = report_file()?;
        self.networks.spent = [0.0; 3];

        let mut decoder = tool("ffmpeg")
            .args(["-v", "error", "-nostdin", "-i"])
            .arg(input)
            .args(limit)
            .args(["-map", "0:v:0", "-vf", &to_rgb(&v, work, "rgb48le"), "-fps_mode", "cfr", "-r", &v.fps])
            .args(["-f", "rawvideo", "-pix_fmt", "rgb48le", "pipe:1"])
            .stdout(Stdio::piped())
            .stderr(report.try_clone().map_err(|e| e.to_string())?)
            .spawn()
            .map_err(|e| format!("could not start ffmpeg: {e}"))?;

        // back to the source's own YUV flavour and size first, then sharpen luma only
        let mut filter = format!("scale={}:{}:flags=bicubic:out_color_matrix={}:out_range={},format={}", v.w, v.h, v.matrix, v.range, v.pix);
        filter += &sharpen(settings);
        let mut command = tool("ffmpeg");
        command
            .args(["-v", "error", "-n", "-f", "rawvideo", "-pix_fmt", "rgb48le", "-s", &format!("{ow}x{oh}"), "-framerate", &v.fps, "-i", "pipe:0"])
            .args(limit)
            .arg("-i")
            .arg(input)
            .args(["-map", "0:v:0", "-map", "1:a?", "-map_metadata", "1", "-vf", &filter])
            .args(encoder_args(encoder, output))
            .args(["-colorspace", &v.tag, "-color_range", v.range]);
        if !v.trc.is_empty() {
            command.args(["-color_trc", &v.trc]);
        }
        if !v.primaries.is_empty() {
            command.args(["-color_primaries", &v.primaries]);
        }
        let mut encoder = match command.args(["-c:a", "copy"]).arg(output).stdin(Stdio::piped()).stderr(report).spawn() {
            Ok(child) => child,
            Err(error) => {
                let _ = decoder.kill();
                return Err(format!("could not start ffmpeg: {error}"));
            }
        };
        let (Some(mut frames_in), Some(mut frames_out)) = (decoder.stdout.take(), encoder.stdin.take()) else {
            return Err("ffmpeg pipes are missing".into());
        };

        // Three stages on three threads, so the graphics card never waits for the processor:
        // read + unpack  ->  networks (this thread)  ->  pack + write.
        let total = ((if test { v.duration.min(10.0) } else { v.duration }) * v.fps_f).max(1.0);
        let level = noise_level(settings);
        let (started, mut reported, mut done) = (Instant::now(), Instant::now(), 0u64);
        let (unpacked_tx, unpacked_rx) = mpsc::sync_channel::<Unpacked>(2);
        let (result_tx, result_rx) = mpsc::sync_channel::<Vec<f32>>(2);
        let result = std::thread::scope(|scope| {
            scope.spawn(move || {
                let mut frame = vec![0u8; ww * wh * 6];
                while frames_in.read_exact(&mut frame).is_ok() {
                    if unpacked_tx.send(unpack(&frame, ww, wh, level)).is_err() {
                        break;
                    }
                }
            });
            let writer = scope.spawn(move || {
                for picture in result_rx {
                    if frames_out.write_all(&pack(&picture)).is_err() {
                        return false;
                    }
                }
                true // dropping `frames_out` here is what tells the encoder the video is over
            });
            let mut result = Ok(());
            for frame in &unpacked_rx {
                if self.cancelled() {
                    result = Err("cancelled".to_string());
                    break;
                }
                let picture = match self.networks.run(frame, work, (v.w, v.h), settings, v.luma, &self.events) {
                    Ok(picture) => picture,
                    Err(error) => {
                        result = Err(error);
                        break;
                    }
                };
                if result_tx.send(picture).is_err() {
                    result = Err(format!("the encoder stopped: {}", report_tail()));
                    break;
                }
                done += 1;
                if reported.elapsed().as_millis() >= 250 {
                    reported = Instant::now();
                    let fraction = (done as f64 / total).min(1.0);
                    let elapsed = started.elapsed().as_secs_f64();
                    let left = elapsed * (1.0 - fraction) / fraction.max(1e-6);
                    let text = format!("{label}  -  {:.0}%  -  {} left  ({:.1} fps)", fraction * 100.0, clock(left), done as f64 / elapsed.max(1e-6));
                    self.emit(Event::Progress { fraction: fraction as f32, text });
                }
            }
            // Unblock the other two threads: closed channels end their loops, and on a failure
            // killing ffmpeg ends whatever read or write they are waiting in.
            drop(unpacked_rx);
            drop(result_tx);
            if result.is_err() {
                let _ = decoder.kill();
                let _ = encoder.kill();
            }
            if result.is_ok() && !writer.join().unwrap_or(false) {
                result = Err(format!("the encoder stopped: {}", report_tail()));
            }
            result
        });
        if done > 0 {
            let per_frame = |seconds: f64| seconds * 1000.0 / done as f64;
            let [denoise, detail, enhance] = self.networks.spent;
            self.log(format!(
                "  {:.1} fps; ms per frame: denoise {:.0}, detail {:.0}, enhance {:.0}",
                done as f64 / started.elapsed().as_secs_f64(),
                per_frame(denoise),
                per_frame(detail),
                per_frame(enhance)
            ));
        }
        let _ = decoder.wait();
        let encoded = encoder.wait();
        result?;
        if done == 0 {
            return Err(format!("no frames decoded: {}", report_tail()));
        }
        match encoded {
            Ok(status) if status.success() => Ok(()),
            _ => Err(report_tail()),
        }
    }

    /// Fast / Best quality: one ffmpeg command, progress read from its `-progress` output.
    fn run_classic(&mut self, input: &Path, output: &Path, settings: &Settings, encoder: usize, test: bool, label: &str) -> Result<(), String> {
        let duration = probe(input).map(|v| v.duration).unwrap_or(0.0);
        let total = if test { if duration > 0.0 { duration.min(10.0) } else { 10.0 } } else { duration };
        let limit: &[&str] = if test { &["-t", "10"] } else { &[] };
        let mut child = tool("ffmpeg")
            .args(["-hide_banner", "-v", "error", "-n", "-nostats", "-progress", "pipe:1", "-i"])
            .arg(input)
            .args(limit)
            .args(["-map", "0:v:0", "-map", "0:a?", "-vf", &classic_filter(settings)])
            .args(encoder_args(encoder, output))
            .args(["-c:a", "copy"])
            .arg(output)
            .stdout(Stdio::piped())
            .stderr(report_file()?)
            .spawn()
            .map_err(|e| format!("could not start ffmpeg: {e}"))?;
        let started = Instant::now();
        if let Some(stdout) = child.stdout.take() {
            for line in BufReader::new(stdout).lines().map_while(Result::ok) {
                if self.cancelled() {
                    let _ = child.kill();
                    break;
                }
                let Some(micros) = line.strip_prefix("out_time_us=").and_then(|v| v.trim().parse::<f64>().ok()) else { continue };
                if total > 0.0 {
                    let fraction = (micros / 1e6 / total).clamp(0.0, 1.0);
                    let left = started.elapsed().as_secs_f64() * (1.0 - fraction) / fraction.max(1e-6);
                    let text = format!("{label}  -  {:.0}%  -  {} left", fraction * 100.0, clock(left));
                    self.emit(Event::Progress { fraction: fraction as f32, text });
                }
            }
        }
        match child.wait() {
            Ok(status) if status.success() && !self.cancelled() => Ok(()),
            _ => Err(report_tail()),
        }
    }

    /// One frame at `at` seconds (middle of the video when not given): a crop from the centre,
    /// original on the left, processed on the right. Returns `(width, height, rgb8, at)`.
    fn preview(&mut self, input: &Path, at: Option<f64>, settings: &Settings) -> Result<(usize, usize, Vec<u8>, f64), String> {
        let v = probe(input)?;
        let mut at = at.unwrap_or(v.duration / 2.0);
        if v.duration > 0.0 && at >= v.duration {
            return Err(format!("Preview time is past the end ({}).", clock(v.duration)));
        }
        if v.duration > 1.0 {
            at = at.min(v.duration - 1.0); // room for the frames temporal filters need
        }
        let (cw, ch) = (v.w.min(PREVIEW_W), v.h.min(PREVIEW_H));
        let seek = format!("{at:.3}");

        if settings.m != 2 {
            // skip 8 frames so temporal filters settle
            let graph = format!(
                "[0:v:0]crop={cw}:{ch},format=yuv420p,split[a][b];[b]{}[c];[a][c]hstack,select=gte(n\\,8),{}",
                classic_filter(settings),
                to_rgb(&v, (cw * 2, ch), "rgb24")
            );
            let rgb = grab(tool("ffmpeg").args(["-v", "error", "-nostdin", "-ss", &seek, "-i"]).arg(input).args([
                "-filter_complex",
                &graph,
                "-frames:v",
                "1",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "pipe:1",
            ]))?;
            return if rgb.len() == cw * 2 * ch * 3 { Ok((cw * 2, ch, rgb, at)) } else { Err("Preview failed (clip too short or unreadable).".into()) };
        }

        let frame_at = |size: (usize, usize), format: &str| {
            grab(
                tool("ffmpeg")
                    .args(["-v", "error", "-nostdin", "-ss", &seek, "-i"])
                    .arg(input)
                    .args(["-map", "0:v:0", "-vf", &format!("crop={cw}:{ch},{}", to_rgb(&v, size, format))])
                    .args(["-frames:v", "1", "-f", "rawvideo", "-pix_fmt", format, "pipe:1"]),
            )
        };
        let original = frame_at((cw, ch), "rgb24")?;
        if original.len() != cw * ch * 3 {
            return Err("Preview failed (clip too short or unreadable).".into());
        }
        let work = work_size(&v, cw, ch, settings);
        let small = frame_at(work, "rgb48le")?;
        if small.len() != work.0 * work.1 * 6 {
            return Err("Preview failed (could not read the frame).".into());
        }
        let factor = if settings.enh { self.networks.enhance_plan(work, (cw, ch)).1 } else { 1 };
        let picture = self.networks.run(unpack(&small, work.0, work.1, noise_level(settings)), work, (cw, ch), settings, v.luma, &self.events)?;
        // same last steps as the encoder: back to full size, sharpen in YUV
        let finish = format!("scale={cw}:{ch}:flags=bicubic,format=yuv444p{},format=rgb24", sharpen(settings));
        let size = format!("{}x{}", work.0 * factor, work.1 * factor);
        let after = pipe_through(
            tool("ffmpeg").args(["-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb48le", "-s", &size, "-i", "pipe:0"]).args([
                "-vf",
                &finish,
                "-frames:v",
                "1",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "pipe:1",
            ]),
            pack(&picture),
        )?;
        if after.len() != original.len() {
            return Err("Preview failed (unexpected frame size).".into());
        }
        let mut both = Vec::with_capacity(original.len() * 2);
        for (left, right) in original.chunks_exact(cw * 3).zip(after.chunks_exact(cw * 3)) {
            both.extend_from_slice(left);
            both.extend_from_slice(right);
        }
        Ok((cw * 2, ch, both, at))
    }
}

impl Networks {
    /// Which enhancer file gives a picture of `work` size closest to `target`, and how many times
    /// `work` its output is. The plain model gives 4x; two variants average that down on the
    /// graphics card to 2x or 1x, which is far less data to read back and to hand to ffmpeg.
    fn enhance_plan(&self, work: (usize, usize), target: (usize, usize)) -> (&'static str, usize) {
        let variant = |file: &'static str, factor: usize| (target == (work.0 * factor, work.1 * factor) && self.models.join(file).is_file()).then_some((file, factor));
        variant("realesr_general_x4v3_d2.onnx", 2).or(variant("realesr_general_x4v3_d4.onnx", 1)).unwrap_or((ENHANCER, 4))
    }

    /// One unpacked frame (`work` = its width x height) through the networks. Returns planar RGB
    /// at the same size, or at the enhancer's multiple of it (see [`Networks::enhance_plan`]).
    fn run(&mut self, frame: Unpacked, (w, h): (usize, usize), target: (usize, usize), settings: &Settings, luma: (f32, f32), events: &Sender<Event>) -> Result<Vec<f32>, String> {
        let n = w * h;
        let (pw, ph) = (w.div_ceil(8) * 8, h.div_ceil(8) * 8);
        let mark = Instant::now();
        let denoiser = session(&mut self.denoiser, &self.models, DENOISER, (ph, pw), events)?;
        let output = infer(denoiser, [1, 3 + 1, ph, pw], frame.input)?;
        self.spent[0] += mark.elapsed().as_secs_f64();
        if output.len() != 3 * pw * ph {
            return Err("the denoiser gave an answer of the wrong size".into());
        }

        let mark = Instant::now();
        let mut clean = vec![0f32; 3 * n];
        for c in 0..3 {
            for y in 0..h {
                let row = &output[c * pw * ph + y * pw..][..w];
                for (to, from) in clean[c * n + y * w..][..w].iter_mut().zip(row) {
                    *to = from.clamp(0.0, 1.0);
                }
            }
        }
        if settings.d > 0 {
            restore_detail(&frame.original, &mut clean, w, h, settings.d as f32, luma);
        }
        self.spent[1] += mark.elapsed().as_secs_f64();
        if !settings.enh {
            return Ok(clean);
        }

        let mark = Instant::now();
        let (file, factor) = self.enhance_plan((w, h), target);
        let enhancer = session(&mut self.enhancer, &self.models, file, (h, w), events)?;
        let enhanced = infer(enhancer, [1, 3, h, w], clean)?;
        self.spent[2] += mark.elapsed().as_secs_f64();
        if enhanced.len() != 3 * n * factor * factor {
            return Err("the enhancer gave an answer of the wrong size".into());
        }
        Ok(enhanced)
    }
}

/// The network of a model file for pictures of `size`, loaded when first needed and kept until
/// the file or the size changes. DirectML is several times faster when it knows the size up
/// front, and within a job (and between previews of one video) it never changes.
fn session<'a>(slot: &'a mut Option<Loaded>, models: &Path, file: &'static str, size: (usize, usize), events: &Sender<Event>) -> Result<&'a mut Session, String> {
    if slot.as_ref().is_none_or(|loaded| loaded.size != size || loaded.file != file) {
        let model = models.join(file);
        if !model.is_file() {
            return Err(format!("model file not found: {} (run ai\\export_onnx.py)", model.display()));
        }
        let open = |on_card: bool| -> Result<Session, String> {
            let text = |e: ort::Error| e.to_string();
            let mut builder = Session::builder().map_err(text)?;
            builder = builder.with_dimension_override("h", size.0 as i64).map_err(|e| e.to_string())?;
            builder = builder.with_dimension_override("w", size.1 as i64).map_err(|e| e.to_string())?;
            if on_card {
                // DirectML wants memory patterns off; HighPerformance picks the discrete GPU on laptops
                builder = builder.with_memory_pattern(false).map_err(|e| e.to_string())?;
                let card = DirectML::default().with_performance_preference(PerformancePreference::HighPerformance);
                builder = builder.with_execution_providers([card.build().error_on_failure()]).map_err(|e| e.to_string())?;
            }
            builder.commit_from_file(&model).map_err(text)
        };
        *slot = None; // free the old one first: two copies may not fit in video memory
        // The graphics card if it can be used, the processor if not.
        let loaded = match open(true) {
            Ok(session) => session,
            Err(reason) => {
                let _ = events.send(Event::Log(format!("  GPU not usable ({reason}); running on the CPU, much slower.")));
                open(false)?
            }
        };
        *slot = Some(Loaded { size, file, session: loaded });
    }
    slot.as_mut().map(|loaded| &mut loaded.session).ok_or_else(|| "model not loaded".to_string())
}

fn infer(session: &mut Session, shape: [usize; 4], data: Vec<f32>) -> Result<Vec<f32>, String> {
    let input = Tensor::from_array((shape, data)).map_err(|e| e.to_string())?;
    let outputs = session.run(ort::inputs!["input" => input]).map_err(|e| e.to_string())?;
    let (_, values) = outputs["output"].try_extract_tensor::<f32>().map_err(|e| e.to_string())?;
    Ok(values.to_vec())
}

/// Strength slider 1..=10 to the noise level the denoiser is told, 3..=30 out of 255.
fn noise_level(settings: &Settings) -> f32 {
    3.0 * settings.s as f32 / 255.0
}

/// RGB48 bytes (`w` x `h`) to what the networks take. The denoiser halves the picture three
/// times, so its copy has sides padded to multiples of 8 by repeating the edge.
fn unpack(frame: &[u8], w: usize, h: usize, level: f32) -> Unpacked {
    let n = w * h;
    let mut original = vec![0f32; 3 * n];
    for (i, pixel) in frame.chunks_exact(6).enumerate() {
        for c in 0..3 {
            original[c * n + i] = u16::from_le_bytes([pixel[c * 2], pixel[c * 2 + 1]]) as f32 / 65535.0;
        }
    }
    let (pw, ph) = (w.div_ceil(8) * 8, h.div_ceil(8) * 8);
    let mut input = vec![level; 4 * pw * ph];
    for c in 0..3 {
        for y in 0..ph {
            let row = &original[c * n + y.min(h - 1) * w..][..w];
            let padded = &mut input[c * pw * ph + y * pw..][..pw];
            padded[..w].copy_from_slice(row);
            padded[w..].fill(row[w - 1]);
        }
    }
    Unpacked { original, input }
}

/// Planar RGB 0..=1 to RGB48 bytes.
fn pack(picture: &[f32]) -> Vec<u8> {
    let pixels = picture.len() / 3;
    let mut bytes = vec![0u8; pixels * 6];
    for (i, pixel) in bytes.chunks_exact_mut(6).enumerate() {
        for c in 0..3 {
            let value = (picture[c * pixels + i].clamp(0.0, 1.0) * 65535.0 + 0.5) as u16;
            pixel[c * 2..c * 2 + 2].copy_from_slice(&value.to_le_bytes());
        }
    }
    bytes
}

/// Puts part of the original luma detail back on top of the denoised picture; `k` = 0..=10.
///
/// What the network removed (`original - clean`) is noise plus fine texture. It is returned almost
/// fully where the picture has structure (hair, fabric, edges: noise hides in texture there) and
/// only a little in flat areas, where it would read as noise. Luma only, so colour noise stays out.
fn restore_detail(original: &[f32], clean: &mut [f32], w: usize, h: usize, k: f32, (kr, kb): (f32, f32)) {
    let n = w * h;
    let kg = 1.0 - kr - kb;
    let luma = |p: &[f32], i: usize| kr * p[i] + kg * p[n + i] + kb * p[2 * n + i];
    let clean_luma: Vec<f32> = (0..n).map(|i| luma(clean, i)).collect();
    // structure is measured on a smoothed copy, so leftover grain does not count as texture
    let smooth = box_mean(&clean_luma, w, h, 2);
    let mut gradient = vec![0f32; n];
    for y in 0..h {
        for x in 0..w {
            let gx = if x >= 1 && x + 1 < w { smooth[y * w + x + 1] - smooth[y * w + x - 1] } else { 0.0 };
            let gy = if y >= 1 && y + 1 < h { smooth[(y + 1) * w + x] - smooth[(y - 1) * w + x] } else { 0.0 };
            gradient[y * w + x] = (gx * gx + gy * gy).sqrt();
        }
    }
    let gradient = box_mean(&gradient, w, h, 3);
    for i in 0..n {
        let structure = (gradient[i] / DETAIL_KNEE).clamp(0.0, 1.0).powi(2); // 0 = flat, 1 = textured
        let amount = (k / 10.0) * (DETAIL_FLAT + (DETAIL_TEXTURE - DETAIL_FLAT) * structure);
        let back = (luma(original, i) - clean_luma[i]) * amount;
        for c in 0..3 {
            clean[c * n + i] = (clean[c * n + i] + back).clamp(0.0, 1.0);
        }
    }
}

/// Mean over a (2r+1)-square window; at the borders, over the part of the window inside the picture.
fn box_mean(src: &[f32], w: usize, h: usize, r: usize) -> Vec<f32> {
    let mut prefix = vec![0f64; w.max(h) + 1];
    let mut rows = vec![0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            prefix[x + 1] = prefix[x] + src[y * w + x] as f64;
        }
        for x in 0..w {
            let (a, b) = (x.saturating_sub(r), (x + r + 1).min(w));
            rows[y * w + x] = ((prefix[b] - prefix[a]) / (b - a) as f64) as f32;
        }
    }
    let mut out = vec![0f32; w * h];
    for x in 0..w {
        for y in 0..h {
            prefix[y + 1] = prefix[y] + rows[y * w + x] as f64;
        }
        for y in 0..h {
            let (a, b) = (y.saturating_sub(r), (y + r + 1).min(h));
            out[y * w + x] = ((prefix[b] - prefix[a]) / (b - a) as f64) as f32;
        }
    }
    out
}

/// Size the networks work at, for a picture of `w` x `h` out of video `v`.
///
/// Footage that was zoomed holds its real detail (and its noise grain) at a smaller size; shrinking
/// first makes the grain pixel-sized, which is what denoisers expect.
fn work_size(v: &Video, w: usize, h: usize, settings: &Settings) -> (usize, usize) {
    let mut scale: f64 = if settings.soft { 0.5 } else { 1.0 };
    if settings.enh {
        scale = scale.min((ENHANCE_MAX_PIXELS / (v.w * v.h) as f64).sqrt());
    }
    if scale >= 1.0 {
        return (w, h);
    }
    (((w as f64 * scale).round() as usize).max(8), ((h as f64 * scale).round() as usize).max(8))
}

/// ffmpeg filter: the video's YUV to RGB with its own matrix, at `size`.
fn to_rgb(v: &Video, (w, h): (usize, usize), format: &str) -> String {
    format!("scale={w}:{h}:flags=bicubic:in_color_matrix={}:in_range={},format={format}", v.matrix, v.range)
}

fn sharpen(settings: &Settings) -> String {
    if settings.sh == 0 { String::new() } else { format!(",unsharp=5:5:{:.3}:5:5:0", 0.15 * settings.sh as f64) }
}

/// Slider 1..=10 to an ffmpeg filter. Fast = hqdn3d, Best quality = nlmeans; "keep detail" lays
/// part of the original back over the result (a uniform mix here; the AI path does it by texture).
fn classic_filter(settings: &Settings) -> String {
    let k = settings.s as f64;
    let mut filter = if settings.m == 0 {
        let ls = 2.0 * k;
        format!("hqdn3d={:.3}:{:.3}:{:.3}:{:.3}", ls, ls * 0.75, ls * 1.5, ls * 1.125)
    } else {
        format!("nlmeans=s={}:p=7:r=9", settings.s + 1)
    };
    if settings.d > 0 {
        filter = format!("split[vo][vd];[vd]{filter}[vn];[vo][vn]blend=all_mode=normal:all_opacity={:.3}", 0.05 * settings.d as f64);
    }
    filter + &sharpen(settings)
}

fn encoder_args(encoder: usize, output: &Path) -> Vec<String> {
    let mut args: Vec<String> = ENCODERS[encoder.min(ENCODERS.len() - 1)].1.split_whitespace().map(str::to_string).collect();
    let mkv = output.extension().is_some_and(|e| e.eq_ignore_ascii_case("mkv"));
    if args.iter().any(|a| a.contains("hevc")) && !mkv {
        args.extend(["-tag:v".to_string(), "hvc1".to_string()]);
    }
    args
}

fn probe(path: &Path) -> Result<Video, String> {
    let out = tool("ffprobe")
        .args(["-v", "error", "-select_streams", "v:0", "-show_entries"])
        .arg("stream=width,height,pix_fmt,avg_frame_rate,r_frame_rate,color_space,color_range,color_transfer,color_primaries:stream_side_data=rotation:format=duration")
        .args(["-of", "json"])
        .arg(path)
        .output()
        .map_err(|e| format!("could not start ffprobe: {e}"))?;
    let json: serde_json::Value = serde_json::from_slice(&out.stdout).map_err(|_| "ffprobe could not read the file".to_string())?;
    let stream = &json["streams"][0];
    let text = |key: &str| stream[key].as_str().unwrap_or("").to_string();
    let (Some(mut w), Some(mut h)) = (stream["width"].as_u64(), stream["height"].as_u64()) else {
        return Err("no video stream found".into());
    };
    let rotated = stream["side_data_list"].as_array().is_some_and(|list| {
        list.iter().any(|side| side["rotation"].as_f64().or_else(|| side["rotation"].as_str().and_then(|s| s.parse().ok())).is_some_and(|r| (r.abs() as i64) % 180 == 90))
    });
    if rotated {
        std::mem::swap(&mut w, &mut h); // ffmpeg auto-rotates on decode
    }
    let mut fps = text("avg_frame_rate");
    if fps.is_empty() || fps.starts_with("0/") || fps.ends_with("/0") {
        fps = text("r_frame_rate");
    }
    let (num, den) = fps.split_once('/').unwrap_or((fps.as_str(), "1"));
    let fps_f = match (num.parse::<f64>(), den.parse::<f64>()) {
        (Ok(n), Ok(d)) if n > 0.0 && d > 0.0 => n / d,
        _ => {
            fps = "30/1".into();
            30.0
        }
    };
    let space = text("color_space");
    // untagged: the same guess players make
    let (matrix, tag) = match space.as_str() {
        "bt709" => ("bt709", space.clone()),
        "smpte170m" | "bt470bg" => ("bt601", space.clone()),
        "bt2020nc" => ("bt2020", space.clone()),
        _ if h >= 720 => ("bt709", "bt709".to_string()),
        _ => ("bt601", "smpte170m".to_string()),
    };
    let luma = match matrix {
        "bt601" => (0.299, 0.114),
        "bt2020" => (0.2627, 0.0593),
        _ => (0.2126, 0.0722),
    };
    let pix = text("pix_fmt");
    let known = |value: String| if value == "unknown" { String::new() } else { value };
    Ok(Video {
        w: w as usize,
        h: h as usize,
        fps,
        fps_f,
        duration: json["format"]["duration"].as_str().and_then(|d| d.parse().ok()).unwrap_or(0.0),
        matrix,
        luma,
        tag,
        range: if text("color_range") == "pc" || pix.starts_with("yuvj") { "pc" } else { "tv" },
        trc: known(text("color_transfer")),
        primaries: known(text("color_primaries")),
        pix: if ["10", "12", "16"].iter().any(|bits| pix.contains(bits)) { "yuv420p10le" } else { "yuv420p" },
    })
}

/// `name_denoised.ext` next to the input, never an existing file. mkv takes any audio codec, so
/// copying the audio cannot fail on unusual containers.
pub fn output_path(input: &Path, test: bool) -> PathBuf {
    let dir = input.parent().unwrap_or(Path::new("."));
    let stem = input.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_else(|| "video".into());
    let ext = input.extension().map(|e| e.to_string_lossy().to_lowercase()).filter(|e| ["mp4", "mov", "mkv"].contains(&e.as_str())).unwrap_or_else(|| "mkv".into());
    let suffix = if test { "_denoised_test" } else { "_denoised" };
    let mut path = dir.join(format!("{stem}{suffix}.{ext}"));
    let mut n = 2;
    while path.exists() {
        path = dir.join(format!("{stem}{suffix}_{n}.{ext}"));
        n += 1;
    }
    path
}

pub fn file_name(path: &Path) -> String {
    path.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default()
}

/// Seconds as `h:mm:ss`.
pub fn clock(seconds: f64) -> String {
    let s = seconds.max(0.0) as u64;
    format!("{}:{:02}:{:02}", s / 3600, s / 60 % 60, s % 60)
}

/// ffmpeg / ffprobe from the PATH, without a console window popping up.
pub fn tool(name: &str) -> Command {
    let mut command = Command::new(name);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    }
    command
}

fn report_path() -> PathBuf {
    let dir = std::env::temp_dir().join("VideoDenoise");
    let _ = std::fs::create_dir_all(&dir);
    dir.join("report_rs.log")
}

/// Where ffmpeg's error output goes; emptied for every job.
fn report_file() -> Result<File, String> {
    File::create(report_path()).map_err(|e| format!("could not write the log file: {e}"))
}

fn report_tail() -> String {
    let text = std::fs::read_to_string(report_path()).unwrap_or_default();
    let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
    if lines.is_empty() { "ffmpeg failed without a message".into() } else { lines[lines.len().saturating_sub(6)..].join(" | ") }
}

/// Runs a command and returns what it wrote to its standard output.
fn grab(command: &mut Command) -> Result<Vec<u8>, String> {
    let out = command.stdin(Stdio::null()).stderr(report_file()?).output().map_err(|e| format!("could not start ffmpeg: {e}"))?;
    Ok(out.stdout)
}

/// Like [`grab`], with `input` fed to the command's standard input.
fn pipe_through(command: &mut Command, input: Vec<u8>) -> Result<Vec<u8>, String> {
    let mut child = command.stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(report_file()?).spawn().map_err(|e| format!("could not start ffmpeg: {e}"))?;
    let Some(mut stdin) = child.stdin.take() else { return Err("ffmpeg pipe is missing".into()) };
    // written from another thread: ffmpeg may start answering before it has read everything
    let writer = std::thread::spawn(move || {
        let _ = stdin.write_all(&input);
    });
    let out = child.wait_with_output().map_err(|e| e.to_string())?;
    let _ = writer.join();
    Ok(out.stdout)
}
