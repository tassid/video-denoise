# One-time setup of the AI engine: Python environment with GPU PyTorch + model weights (~3 GB download).
# The app works without it (methods "Fast" and "Best quality" only need ffmpeg).
$ErrorActionPreference = 'Stop'
$ai = Join-Path $PSScriptRoot 'ai'
$venvPython = Join-Path $ai 'venv\Scripts\python.exe'
$models = Join-Path $ai 'models'

if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Write-Warning 'ffmpeg not found in PATH. Install it with:  winget install Gyan.FFmpeg'
}

if (-not (Test-Path -LiteralPath $venvPython)) {
    if (Get-Command uv -ErrorAction SilentlyContinue) {
        uv venv (Join-Path $ai 'venv') --python 3.13
        uv pip install --python $venvPython torch numpy --index-url https://download.pytorch.org/whl/cu128
    } else {
        python -m venv (Join-Path $ai 'venv')
        & $venvPython -m pip install torch numpy --index-url https://download.pytorch.org/whl/cu128
    }
}

# Official release files; hashes pin the exact versions the app was tested with.
$weights = @(
    @{ name = 'drunet_color.pth'
       url  = 'https://github.com/cszn/KAIR/releases/download/v1.0/drunet_color.pth'
       sha  = '479ABE3C5327DFD10FF54A80EC7D4098CA80752A5C9492CDFF31CEE430BEC4B4' }
    @{ name = 'realesr-general-x4v3.pth'
       url  = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth'
       sha  = '8DC7EDB9AC80CCDC30C3A5DCA6616509367F05FBC184AD95B731F05BECE96292' }
    @{ name = 'realesr-general-wdn-x4v3.pth'
       url  = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth'
       sha  = '1641F8C4464B9F097C9FDDA5589273713F67CF59F3D909E0BD688F0CEE269DCA' }
)
[void](New-Item -ItemType Directory -Force -Path $models)
foreach ($w in $weights) {
    $path = Join-Path $models $w.name
    if (-not (Test-Path -LiteralPath $path)) {
        Write-Host "Downloading $($w.name)..."
        curl.exe -L --fail -sS -o $path $w.url
    }
    if ((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash -ne $w.sha) {
        Remove-Item -LiteralPath $path
        throw "$($w.name): checksum mismatch, file removed. Run setup again."
    }
}

& $venvPython -c "import torch; print('PyTorch', torch.__version__, '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'not available (AI will be very slow)')"
Write-Host 'Done. Start the app with "Video Denoise.bat".'
