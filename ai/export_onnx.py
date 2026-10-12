"""Export the two networks to ONNX for the Rust app (rust/).

    ai\\venv\\Scripts\\python.exe ai\\export_onnx.py

Writes ai/models/drunet_color.onnx and ai/models/realesr_general_x4v3.onnx.
Both take and return float32 NCHW in 0..1 with free height/width; inside they run in half
precision when --fp16 is given (the default), which is what makes them fast on a GPU.
The enhancer's two official weight sets are blended here at ENHANCE_DN, the value the app uses.
"""
import argparse
import os

import torch
import torch.nn as nn

from ai_denoise import MODELS, SRVGGNetCompact, UNetRes

ENHANCE_DN = 0.2


class Fp32IO(nn.Module):
    """float32 in/out around a half-precision network, so callers never handle float16."""

    def __init__(self, net, half):
        super().__init__()
        self.net = net.half() if half else net
        self.half = half

    def forward(self, x):
        return self.net(x.half()).float() if self.half else self.net(x)


def export(net, channels, path, half, dev):
    net = Fp32IO(net.eval(), half).to(dev)
    sample = torch.rand(1, channels, 64, 96, device=dev)
    torch.onnx.export(
        net, (sample,), path, dynamo=False, opset_version=17,
        input_names=['input'], output_names=['output'],
        dynamic_axes={'input': {2: 'h', 3: 'w'}, 'output': {2: 'oh', 3: 'ow'}})
    print('%s  %.1f MB' % (os.path.basename(path), os.path.getsize(path) / 1e6))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--fp32', action='store_true', help='keep full precision (slower, for comparison)')
    a = p.parse_args()
    half = not a.fp32
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if half and dev.type != 'cuda':
        raise SystemExit('half-precision export needs the GPU; use --fp32')
    suffix = '' if half else '_fp32'

    dn = UNetRes()
    dn.load_state_dict(torch.load(os.path.join(MODELS, 'drunet_color.pth'), map_location='cpu',
                                  weights_only=True), strict=True)
    export(dn, 4, os.path.join(MODELS, 'drunet_color%s.onnx' % suffix), half, dev)

    def params(name):
        ck = torch.load(os.path.join(MODELS, name), map_location='cpu', weights_only=True)
        return ck.get('params_ema', ck.get('params', ck))

    strong, weak = params('realesr-general-x4v3.pth'), params('realesr-general-wdn-x4v3.pth')
    sr = SRVGGNetCompact()
    sr.load_state_dict({k: ENHANCE_DN * strong[k] + (1 - ENHANCE_DN) * weak[k] for k in strong}, strict=True)
    export(sr, 3, os.path.join(MODELS, 'realesr_general_x4v3%s.onnx' % suffix), half, dev)
    # The app wants the result at 2x or 1x its input far more often than at 4x. Averaging down
    # on the GPU, inside the model, means 4x or 16x less data to read back and to pipe to ffmpeg.
    for factor in (2, 4):
        export(nn.Sequential(sr, nn.AvgPool2d(factor)), 3,
               os.path.join(MODELS, 'realesr_general_x4v3_d%d%s.onnx' % (factor, suffix)), half, dev)


if __name__ == '__main__':
    main()
