# Video Denoise — Rust port

The same app as `VideoDenoise.ps1`, as one native program: an `egui` window, ONNX Runtime with
DirectML for the networks, and `ffmpeg` on pipes for decoding and encoding.

Same models, same settings, same `presets.json` (presets saved in one app show up in the other).

## What is different from the PowerShell + Python version

| | PowerShell + Python | Rust |
|---|---|---|
| Runtime for the AI | PyTorch + CUDA (NVIDIA only, ~4 GB) | ONNX Runtime + DirectML (any DirectX 12 GPU), ~30 MB program |
| Preview | new window, ~10 s (PyTorch loads every time) | inside the window, ~2 s cold; networks stay loaded |
| Speed, AI + soft source + enhance, 1080x1920 | 5.7 fps | 8.4 fps |
| Speed, AI only, 1080x1920 | ~3 fps | 4.1 fps |

Measured on an RTX 5060 Ti with a 2-second, 240-frame cut of real footage.

Output is the same picture: Rust against Python on that clip gives PSNR 42-45 dB in luma, which
is the size of the difference two H.264 encodes of the same frames make. One deliberate
difference: when enhancing, the 4x result is averaged down to the final size on the GPU instead
of being resized by ffmpeg, which is where most of the extra speed comes from.

## Build and run

Needs the Rust toolchain, `ffmpeg` on the PATH, and the ONNX models.

```powershell
# once: export the networks to ONNX (needs the Python environment from ..\setup.ps1)
..\ai\venv\Scripts\python.exe -m pip install onnx onnxscript
..\ai\venv\Scripts\python.exe ..\ai\export_onnx.py

cargo build --release
.\target\release\video-denoise.exe
```

The program looks for `ai\models\*.onnx` and `presets.json` in the first folder above itself that
has an `ai\models` folder.

## Headless test

```powershell
.\target\release\video-denoise.exe --autorun clip.mp4 --at 0:05 --opts "test,s=2,d=8" --log out.txt --shot ui.png
```

Opens the real window, renders a preview, runs the job, saves a picture of the window, and exits
with the number of failed files. `--opts` takes `soft`, `enhance`, `test`, `s=`, `sh=`, `d=`,
`m=`, `p=<preset name>`, `save=<name>`, `del=<name>`.

## Known gaps

- Frames larger than 1440p are not split into tiles for the denoiser yet (the Python version
  does this), so 4K without "soft source" may run out of video memory.
- Not exercised by the headless test: mouse drag-and-drop, Cancel, the Save-preset box.
