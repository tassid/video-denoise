"""AI video denoise / enhance, called by VideoDenoise.ps1.

ffmpeg decodes -> every frame is processed on the GPU -> ffmpeg encodes.
Progress is written ffmpeg-style ("out_time_us=N") so the GUI reuses its progress code.

Two networks:
- DRUNet: plain denoiser (not generative). Removes noise, does not invent detail.
- Real-ESRGAN compact (--enhance): reconstructs detail. Generative: it can change fine
  features (faces!), so it is opt-in.
"""
import argparse
import json
import os
import shlex
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, 'models')
NOWIN = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
MAX_PIXELS = 4_500_000  # larger frames (4K) are denoised in overlapping halves to fit 8 GB VRAM
OVERLAP = 32
ENHANCE_MAX_PIXELS = 2_100_000  # the enhancer outputs 16x its input pixels; keep its input <= ~1080p

LUMA_WEIGHTS = {'bt709': (0.2126, 0.0722), 'bt601': (0.299, 0.114), 'bt2020': (0.2627, 0.0593)}
DETAIL_KNEE = 0.02     # smoothed-luma gradient at which an area counts as fully textured
DETAIL_FLAT = 0.12     # share of the removed signal returned in flat areas at detail=10
DETAIL_TEXTURE = 0.9   # same, in textured areas


class ResBlock(nn.Module):
    def __init__(self, nc):
        super().__init__()
        self.res = nn.Sequential(
            nn.Conv2d(nc, nc, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(nc, nc, 3, 1, 1, bias=False))

    def forward(self, x):
        return x + self.res(x)


class UNetRes(nn.Module):
    """DRUNet. Same layout and parameter names as KAIR's UNetRes, so the official weights load strictly."""

    def __init__(self, in_nc=4, out_nc=3, nc=(64, 128, 256, 512), nb=4):
        super().__init__()

        def blocks(c):
            return [ResBlock(c) for _ in range(nb)]

        self.m_head = nn.Conv2d(in_nc, nc[0], 3, 1, 1, bias=False)
        self.m_down1 = nn.Sequential(*blocks(nc[0]), nn.Conv2d(nc[0], nc[1], 2, 2, 0, bias=False))
        self.m_down2 = nn.Sequential(*blocks(nc[1]), nn.Conv2d(nc[1], nc[2], 2, 2, 0, bias=False))
        self.m_down3 = nn.Sequential(*blocks(nc[2]), nn.Conv2d(nc[2], nc[3], 2, 2, 0, bias=False))
        self.m_body = nn.Sequential(*blocks(nc[3]))
        self.m_up3 = nn.Sequential(nn.ConvTranspose2d(nc[3], nc[2], 2, 2, 0, bias=False), *blocks(nc[2]))
        self.m_up2 = nn.Sequential(nn.ConvTranspose2d(nc[2], nc[1], 2, 2, 0, bias=False), *blocks(nc[1]))
        self.m_up1 = nn.Sequential(nn.ConvTranspose2d(nc[1], nc[0], 2, 2, 0, bias=False), *blocks(nc[0]))
        self.m_tail = nn.Conv2d(nc[0], out_nc, 3, 1, 1, bias=False)

    def forward(self, x0):
        x1 = self.m_head(x0)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x + x4)
        x = self.m_up2(x + x3)
        x = self.m_up1(x + x2)
        return self.m_tail(x + x1)


class SRVGGNetCompact(nn.Module):
    """Real-ESRGAN compact x4 network (realesr-general-x4v3); names match the official weights."""

    def __init__(self, num_feat=64, num_conv=32, upscale=4):
        super().__init__()
        self.upscale = upscale
        self.body = nn.ModuleList([nn.Conv2d(3, num_feat, 3, 1, 1), nn.PReLU(num_parameters=num_feat)])
        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            self.body.append(nn.PReLU(num_parameters=num_feat))
        self.body.append(nn.Conv2d(num_feat, 3 * upscale * upscale, 3, 1, 1))
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x):
        out = x
        for layer in self.body:
            out = layer(out)
        return self.upsampler(out) + F.interpolate(x, scale_factor=self.upscale, mode='nearest')


def to_device(model, dev):
    model.eval().to(dev)
    if dev.type == 'cuda':
        model.half().to(memory_format=torch.channels_last)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_models(dev, enhance, enhance_dn):
    dn = UNetRes()
    dn.load_state_dict(torch.load(os.path.join(MODELS, 'drunet_color.pth'), map_location='cpu',
                                  weights_only=True), strict=True)
    models = {'dn': to_device(dn, dev), 'sr': None}
    if enhance:
        def params(name):
            ck = torch.load(os.path.join(MODELS, name), map_location='cpu', weights_only=True)
            return ck.get('params_ema', ck.get('params', ck))

        # official "denoise strength" control: interpolate between the strong- and weak-denoise weights
        strong, weak = params('realesr-general-x4v3.pth'), params('realesr-general-wdn-x4v3.pth')
        sr = SRVGGNetCompact()
        sr.load_state_dict({k: enhance_dn * strong[k] + (1 - enhance_dn) * weak[k] for k in strong}, strict=True)
        models['sr'] = to_device(sr, dev)
    return models


def run_tiled(model, x):
    """x: (1,4,H,W) with H, W multiples of 8."""
    h, w = x.shape[2:]
    if h * w <= MAX_PIXELS:
        return model(x)
    if w >= h:
        m = (w // 2) // 8 * 8
        left = run_tiled(model, x[..., :m + OVERLAP])
        right = run_tiled(model, x[..., m - OVERLAP:])
        return torch.cat([left[..., :m], right[..., OVERLAP:]], 3)
    m = (h // 2) // 8 * 8
    top = run_tiled(model, x[:, :, :m + OVERLAP])
    bottom = run_tiled(model, x[:, :, m - OVERLAP:])
    return torch.cat([top[:, :, :m], bottom[:, :, OVERLAP:]], 2)


def box(x, r):
    return F.avg_pool2d(x, 2 * r + 1, stride=1, padding=r, count_include_pad=False)


def resize(x, h, w):
    return F.interpolate(x, (h, w), mode='bicubic', antialias=True, align_corners=False)


def restore_detail(x, y, k, matrix='bt709'):
    """Put part of the original luma detail back on top of the denoised picture; k = 0..10.

    What the network removed (x - y) is noise plus fine texture. It is returned almost fully
    where the picture has structure (hair, fabric, edges: noise is hidden by texture there)
    and only a little in flat areas (sky, walls), where it would read as noise.
    Luma only, so colour noise does not come back.
    """
    kr, kb = LUMA_WEIGHTS[matrix]
    w = torch.tensor([kr, 1.0 - kr - kb, kb], device=y.device).view(1, 3, 1, 1)
    ly = (y * w).sum(1, keepdim=True)
    res = (x * w).sum(1, keepdim=True) - ly
    # structure is measured on a smoothed copy, so leftover grain does not count as texture
    ls = box(ly, 2)
    gx = F.pad(ls[..., :, 2:] - ls[..., :, :-2], (1, 1, 0, 0))
    gy = F.pad(ls[..., 2:, :] - ls[..., :-2, :], (0, 0, 1, 1))
    grad = box(torch.sqrt(gx * gx + gy * gy), 3)
    structure = (grad / DETAIL_KNEE).clamp_(0, 1) ** 2  # 0 = flat, 1 = textured
    amount = (k / 10.0) * (DETAIL_FLAT + (DETAIL_TEXTURE - DETAIL_FLAT) * structure)
    return (y + res * amount).clamp_(0, 1)


@torch.inference_mode()
def process(models, frame, a, dev, matrix='bt709'):
    """frame: (H,W,3) uint16 RGB -> same shape/dtype, denoised (and optionally enhanced)."""
    h, w = frame.shape[:2]
    half = dev.type == 'cuda'
    x = torch.from_numpy(frame.astype(np.float32)).to(dev).permute(2, 0, 1).unsqueeze(0) / 65535.0

    # Work size. Footage that was zoomed/enlarged holds its real detail (and its noise grain)
    # at a smaller size; shrinking first makes the grain pixel-sized, which is what denoisers expect.
    scale = a.scale
    if models['sr'] is not None:
        scale = min(scale, (ENHANCE_MAX_PIXELS / (h * w)) ** 0.5)
    wh, ww = (h, w) if scale >= 1 else (max(8, round(h * scale)), max(8, round(w * scale)))
    xs = x if (wh, ww) == (h, w) else resize(x, wh, ww).clamp_(0, 1)

    y = xs
    if a.sigma > 0:
        ph, pw = (-wh) % 8, (-ww) % 8
        t = F.pad(xs, (0, pw, 0, ph), mode='replicate') if ph or pw else xs
        t = torch.cat([t, torch.full_like(t[:, :1], a.sigma / 255.0)], 1)
        if half:
            t = t.half().contiguous(memory_format=torch.channels_last)
        y = run_tiled(models['dn'], t)[:, :, :wh, :ww].float().clamp_(0, 1)
        if a.detail > 0:
            y = restore_detail(xs, y, a.detail, matrix)

    if models['sr'] is not None:
        t = y.half().contiguous(memory_format=torch.channels_last) if half else y
        y = models['sr'](t).float().clamp_(0, 1)  # 4x the work size
    if y.shape[2:] != (h, w):
        y = resize(y, h, w).clamp_(0, 1)
    return (y[0].permute(1, 2, 0) * 65535.0).round_().cpu().numpy().astype('<u2')


def probe(path, ffprobe):
    out = subprocess.run(
        [ffprobe, '-v', 'error', '-select_streams', 'v:0', '-show_entries',
         'stream=width,height,pix_fmt,avg_frame_rate,r_frame_rate,color_space,color_range,'
         'color_transfer,color_primaries:stream_side_data=rotation:format=duration',
         '-of', 'json', path],
        capture_output=True, encoding='utf-8', errors='replace', creationflags=NOWIN)
    info = json.loads(out.stdout)
    s = info['streams'][0]
    w, h = int(s['width']), int(s['height'])
    for sd in s.get('side_data_list', []):
        if abs(int(float(sd.get('rotation', 0)))) % 180 == 90:  # ffmpeg auto-rotates on decode
            w, h = h, w
    fps = s.get('avg_frame_rate', '0/0')
    if fps.startswith('0/') or fps.endswith('/0'):
        fps = s.get('r_frame_rate', '30/1')
    num, den = (fps.split('/') + ['1'])[:2]
    cs = s.get('color_space', '')
    matrix = {'bt709': 'bt709', 'smpte170m': 'bt601', 'bt470bg': 'bt601', 'bt2020nc': 'bt2020'}.get(cs)
    if not matrix:  # untagged: same guess players make
        matrix = 'bt709' if h >= 720 else 'bt601'
        cs = 'bt709' if h >= 720 else 'smpte170m'
    pix = s.get('pix_fmt', 'yuv420p')
    return {
        'w': w, 'h': h, 'fps': fps, 'fps_f': float(num) / float(den),
        'matrix': matrix, 'cs': cs,
        'range': 'pc' if s.get('color_range') == 'pc' or pix.startswith('yuvj') else 'tv',
        'trc': s.get('color_transfer', ''), 'prim': s.get('color_primaries', ''),
        'pix': 'yuv420p10le' if ('10' in pix or '12' in pix or '16' in pix) else 'yuv420p',
    }


def read_frame(pipe, buf):
    view = memoryview(buf)
    got = 0
    while got < len(buf):
        n = pipe.readinto(view[got:])
        if not n:
            return False
        got += n
    return True


def to_rgb(v):
    return 'scale=in_color_matrix={matrix}:in_range={range},format=rgb48le'.format(**v)


def write_progress(path, seconds):
    try:
        with open(path, 'w') as f:
            f.write('out_time_us=%d\n' % int(seconds * 1e6))
    except OSError:
        pass  # GUI is reading it right now; next tick will do


def run_preview(a, models, dev, log):
    v = probe(a.input, a.ffprobe)
    cw, ch = min(v['w'], 640), min(v['h'], 720)
    dec = subprocess.run(
        [a.ffmpeg, '-v', 'error', '-nostdin', '-ss', str(a.at), '-i', a.input, '-map', '0:v:0',
         '-vf', 'crop=%d:%d,%s' % (cw, ch, to_rgb(v)), '-frames:v', '1',
         '-f', 'rawvideo', '-pix_fmt', 'rgb48le', 'pipe:1'],
        stdout=subprocess.PIPE, stderr=log, creationflags=NOWIN)
    if len(dec.stdout) != cw * ch * 6:
        raise RuntimeError('could not read a frame at %s s' % a.at)
    frame = np.frombuffer(dec.stdout, dtype='<u2').reshape(ch, cw, 3)
    both = np.concatenate([frame, process(models, frame, a, dev, v['matrix'])], axis=1)
    sharp = ',unsharp=5:5:%s:5:5:0' % a.sharpen if a.sharpen > 0 else ''
    graph = ('split[a][b];[a]crop=iw/2:ih:0:0,format=yuv444p[l];'
             '[b]crop=iw/2:ih:iw/2:0,format=yuv444p%s[r];[l][r]hstack' % sharp)
    enc = subprocess.run(
        [a.ffmpeg, '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb48le',
         '-s', '%dx%d' % (cw * 2, ch), '-i', 'pipe:0', '-vf', graph, '-frames:v', '1', a.preview],
        input=both.tobytes(), stderr=log, creationflags=NOWIN)
    return enc.returncode


def run_video(a, models, dev, log):
    v = probe(a.input, a.ffprobe)
    w, h = v['w'], v['h']
    limit = ['-t', '10'] if a.test else []

    dec = subprocess.Popen(
        [a.ffmpeg, '-v', 'error', '-nostdin', '-i', a.input, *limit, '-map', '0:v:0',
         '-vf', to_rgb(v), '-fps_mode', 'cfr', '-r', v['fps'],
         '-f', 'rawvideo', '-pix_fmt', 'rgb48le', 'pipe:1'],
        stdout=subprocess.PIPE, stderr=log, creationflags=NOWIN)

    # back to the source's YUV flavour first, then sharpen luma only
    vf = 'scale=out_color_matrix={matrix}:out_range={range},format={pix}'.format(**v)
    if a.sharpen > 0:
        vf += ',unsharp=5:5:%s:5:5:0' % a.sharpen
    tags = ['-colorspace', v['cs'], '-color_range', v['range']]
    if v['trc'] and v['trc'] != 'unknown':
        tags += ['-color_trc', v['trc']]
    if v['prim'] and v['prim'] != 'unknown':
        tags += ['-color_primaries', v['prim']]
    enc = subprocess.Popen(
        [a.ffmpeg, '-v', 'error', '-n', '-f', 'rawvideo', '-pix_fmt', 'rgb48le',
         '-s', '%dx%d' % (w, h), '-framerate', v['fps'], '-i', 'pipe:0',
         *limit, '-i', a.input, '-map', '0:v:0', '-map', '1:a?', '-map_metadata', '1',
         '-vf', vf, *shlex.split(a.encoder), *tags, '-c:a', 'copy', a.output],
        stdin=subprocess.PIPE, stderr=log, creationflags=NOWIN)

    buf = bytearray(w * h * 6)
    n, last = 0, 0.0
    try:
        while read_frame(dec.stdout, buf):
            frame = np.frombuffer(buf, dtype='<u2').reshape(h, w, 3)
            enc.stdin.write(process(models, frame, a, dev, v['matrix']).tobytes())
            n += 1
            if time.time() - last > 0.3:
                last = time.time()
                write_progress(a.progress, n / v['fps_f'])
    finally:
        try:
            enc.stdin.close()
        except OSError:
            pass
        dec.kill() if dec.poll() is None and n == 0 else dec.wait()
        enc.wait()
    if n == 0:
        raise RuntimeError('no frames decoded')
    return enc.returncode


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--output')
    p.add_argument('--preview', help='write a before/after PNG instead of a video')
    p.add_argument('--at', type=float, default=0.0)
    p.add_argument('--sigma', type=float, default=10.0, help='denoise strength (noise level 0..50)')
    p.add_argument('--sharpen', type=float, default=0.0)
    p.add_argument('--detail', type=float, default=0.0, help='original detail put back, 0..10')
    p.add_argument('--scale', type=float, default=1.0, help='work size, <1 for zoomed/soft footage')
    p.add_argument('--enhance', action='store_true', help='reconstruct detail (generative)')
    p.add_argument('--enhance-dn', type=float, default=0.5, help="enhancer's own denoise, 0..1")
    p.add_argument('--encoder', default='-c:v libx264 -crf 18 -preset slow')
    p.add_argument('--test', action='store_true')
    p.add_argument('--ffmpeg', default='ffmpeg')
    p.add_argument('--ffprobe', default='ffprobe')
    p.add_argument('--progress', default='progress.txt')
    p.add_argument('--report', default='report.log')
    a = p.parse_args()

    with open(a.report, 'a', encoding='utf-8', errors='replace') as log:
        try:
            dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if dev.type == 'cuda':
                torch.backends.cudnn.benchmark = True
            models = load_models(dev, a.enhance, a.enhance_dn)
            code = run_preview(a, models, dev, log) if a.preview else run_video(a, models, dev, log)
        except Exception:
            log.write(traceback.format_exc())
            code = 1
    sys.exit(code)


if __name__ == '__main__':
    main()
