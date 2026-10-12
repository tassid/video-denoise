# Video Denoise

A local Windows app that removes visual noise from videos. Drop a video, pick a preset, click Start. Nothing leaves your machine.

The interface is a PowerShell (WinForms) script on top of `ffmpeg`; the AI engine is a Python script running PyTorch on the GPU.

## Requirements

- Windows 10/11
- [ffmpeg](https://ffmpeg.org/) on the PATH: `winget install Gyan.FFmpeg`
- For the **AI** method: an NVIDIA card (tested on an RTX 5060 Ti, 8 GB), Python 3.13 or [uv](https://docs.astral.sh/uv/), about 5 GB of disk

## Install

```powershell
git clone https://github.com/tassid/video-denoise.git
cd video-denoise
powershell -ExecutionPolicy Bypass -File .\setup.ps1   # only for the AI method (~3 GB)
```

Start it by double-clicking `Video Denoise.bat`.

Without running `setup.ps1` the app still works, just without the AI method.

## Usage

1. Drop one or more videos on the window (or click the drop area to browse).
2. Pick a **Preset** or adjust the controls.
3. **Preview** shows a before/after of one frame at 100%. `Preview at` picks the moment (`90`, `1:30`, `0:01:30`; empty = middle of the video).
4. Tick **Test: first 10 s only** to try settings before processing everything.
5. **Start**. The result is written next to the original as `name_denoised.mp4`. The original is never modified or overwritten.

### Controls

| Control | What it does |
|---|---|
| Method | `Fast` (hqdn3d), `Best quality` (nlmeans, slow, CPU) or `AI` (DRUNet, GPU) |
| Strength | How much noise to remove. Too high wipes out texture. |
| Sharpen | Sharpening after the denoise, luma only. |
| Keep detail | Puts part of the original detail back on top of the result. With AI it is selective: almost all of it in textured areas, almost none in flat ones. |
| Soft / zoomed source | For footage that was cropped/enlarged and has coarse grain: processes at half size and scales back. AI only. |
| Enhance | Reconstructs detail with Real-ESRGAN. **It invents detail**: fine on objects, can subtly change real faces. AI only, off in the presets meant for people. |
| Encoder | H.264 on the CPU (most compatible) or HEVC on the NVIDIA GPU (faster). |

### Presets

Built-in presets cannot be changed. **Save...** stores the current settings under a name of your own (in `presets.json`, next to the app); **Delete** removes one of yours.

## Speed (RTX 5060 Ti)

| Case | Roughly |
|---|---|
| AI, 1080p | 3 fps |
| AI + Soft source + Enhance, 1080x1920 | 6 fps |
| AI, 4K | 0.6 fps |
| Fast | close to real time |

## Known limits

- Denoising does not create sharpness. If the source is already soft (zoom, missed focus), only `Enhance` improves how it looks, and it reconstructs detail rather than recovering it.
- The `Natural` and `Heavy noise` presets are starting points; they were not tuned on much footage.
- Output is always 4:2:0 (8 or 10 bit, following the source).

## Models

Downloaded by `setup.ps1` from the official repositories, not redistributed here:

- DRUNet — [cszn/KAIR](https://github.com/cszn/KAIR) (MIT)
- Real-ESRGAN `realesr-general-x4v3` — [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) (BSD-3-Clause)

## Headless test

```powershell
powershell -ExecutionPolicy Bypass -File .\VideoDenoise.ps1 -AutoRun clip.mp4 -AutoRunOpts "test,s=2,d=8"
```

Runs preview + processing through the real window, prints the log and exits with code 0 on success. A snapshot of the window is saved to `%TEMP%\VideoDenoise\ui.png`.
