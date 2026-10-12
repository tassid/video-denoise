# Video Denoise - small drag-and-drop GUI around ffmpeg.
# Output is written next to each input as <name>_denoised.<ext>. Originals are never touched.
param([string]$AutoRun, [string]$AutoRunAt, [int]$AutoRunMethod = -1, [string]$AutoRunOpts = '')  # opts e.g. "soft,enhance,test,s=2,d=8,sh=0"  # dev/test: preview + process this file with defaults, then exit

Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
[System.Windows.Forms.Application]::EnableVisualStyles()

$ffmpeg  = (Get-Command ffmpeg  -ErrorAction SilentlyContinue).Source
$ffprobe = (Get-Command ffprobe -ErrorAction SilentlyContinue).Source
if (-not $ffmpeg -or -not $ffprobe) {
    [void][System.Windows.Forms.MessageBox]::Show("ffmpeg / ffprobe not found in PATH.`nInstall with:  winget install Gyan.FFmpeg", 'Video Denoise')
    exit 1
}

$workDir = Join-Path $env:TEMP 'VideoDenoise'
[void](New-Item -ItemType Directory -Force -Path $workDir)
$progressFile = Join-Path $workDir 'progress.txt'
$reportFile   = Join-Path $workDir 'report.log'

$methods = @('Fast', 'Best quality (slow)')

# Optional AI engine (DRUNet on the GPU), present when the ai\ folder is set up
$aiPython = Join-Path $PSScriptRoot 'ai\venv\Scripts\python.exe'
$aiScript = Join-Path $PSScriptRoot 'ai\ai_denoise.py'
$hasAI = (Test-Path -LiteralPath $aiPython) -and (Test-Path -LiteralPath $aiScript)
if ($hasAI) { $methods += 'AI (best, GPU)' }
$encoders = [ordered]@{
    'H.264 - CPU, most compatible' = '-c:v libx264 -crf 18 -preset slow'
    'HEVC - NVIDIA GPU, faster'    = '-c:v hevc_nvenc -preset p6 -rc vbr -cq 20 -b:v 0'
}

$script:index     = 0
$script:proc      = $null
$script:duration  = 0.0
$script:current   = $null
$script:output    = $null
$script:lastOut   = $null
$script:cancelled = $false
$script:failed    = 0
$script:clock     = New-Object System.Diagnostics.Stopwatch

# ---------- UI ----------
$form = New-Object System.Windows.Forms.Form
$form.Text = 'Video Denoise'
$form.ClientSize = New-Object System.Drawing.Size(540, 632)
$form.FormBorderStyle = 'FixedSingle'
$form.MaximizeBox = $false
$form.StartPosition = 'CenterScreen'
$form.Font = New-Object System.Drawing.Font('Segoe UI', 9)

$drop = New-Object System.Windows.Forms.Label
$drop.Text = "Drop videos here`n(or click to browse)"
$drop.TextAlign = 'MiddleCenter'
$drop.Font = New-Object System.Drawing.Font('Segoe UI', 12)
$drop.BorderStyle = 'FixedSingle'
$drop.BackColor = [System.Drawing.Color]::FromArgb(240, 244, 250)
$drop.Cursor = 'Hand'
$drop.SetBounds(12, 12, 516, 90)

$list = New-Object System.Windows.Forms.ListBox
$list.HorizontalScrollbar = $true
$list.SetBounds(12, 110, 516, 95)

$lblPreset = New-Object System.Windows.Forms.Label
$lblPreset.Text = 'Method'
$lblPreset.SetBounds(12, 218, 105, 20)
$cmbPreset = New-Object System.Windows.Forms.ComboBox
$cmbPreset.DropDownStyle = 'DropDownList'
$cmbPreset.SetBounds(120, 214, 210, 24)
[void]$cmbPreset.Items.AddRange([object[]]$methods)
$cmbPreset.SelectedIndex = $methods.Count - 1
if ($AutoRunMethod -ge 0) { $cmbPreset.SelectedIndex = $AutoRunMethod }

$lblStr = New-Object System.Windows.Forms.Label
$lblStr.Text = 'Strength'
$lblStr.SetBounds(12, 250, 105, 20)
$trk = New-Object System.Windows.Forms.TrackBar
$trk.AutoSize = $false
$trk.Minimum = 1
$trk.Maximum = 10
$trk.LargeChange = 1
$trk.Value = 3
$trk.SetBounds(113, 244, 224, 32)
$lblStrVal = New-Object System.Windows.Forms.Label
$lblStrVal.SetBounds(350, 250, 180, 20)

$lblSharp = New-Object System.Windows.Forms.Label
$lblSharp.Text = 'Sharpen'
$lblSharp.SetBounds(12, 284, 105, 20)
$trkSharp = New-Object System.Windows.Forms.TrackBar
$trkSharp.AutoSize = $false
$trkSharp.Minimum = 0
$trkSharp.Maximum = 10
$trkSharp.LargeChange = 1
$trkSharp.Value = 3
$trkSharp.SetBounds(113, 278, 224, 32)
$lblSharpVal = New-Object System.Windows.Forms.Label
$lblSharpVal.SetBounds(350, 284, 180, 20)

$lblDet = New-Object System.Windows.Forms.Label
$lblDet.Text = 'Keep detail'
$lblDet.SetBounds(12, 318, 105, 20)
$trkDet = New-Object System.Windows.Forms.TrackBar
$trkDet.AutoSize = $false
$trkDet.Minimum = 0
$trkDet.Maximum = 10
$trkDet.LargeChange = 1
$trkDet.Value = 5
$trkDet.SetBounds(113, 312, 224, 32)
$lblDetVal = New-Object System.Windows.Forms.Label
$lblDetVal.SetBounds(350, 318, 180, 20)

$lblEnc = New-Object System.Windows.Forms.Label
$lblEnc.Text = 'Encoder'
$lblEnc.SetBounds(12, 354, 105, 20)
$cmbEnc = New-Object System.Windows.Forms.ComboBox
$cmbEnc.DropDownStyle = 'DropDownList'
$cmbEnc.SetBounds(120, 350, 210, 24)
[void]$cmbEnc.Items.AddRange([object[]]$encoders.Keys)
$cmbEnc.SelectedIndex = 0

$lblAt = New-Object System.Windows.Forms.Label
$lblAt.Text = 'Preview at (m:ss)'
$lblAt.SetBounds(350, 354, 108, 20)
$txtAt = New-Object System.Windows.Forms.TextBox
$txtAt.SetBounds(462, 350, 66, 24)
$tip = New-Object System.Windows.Forms.ToolTip
$tip.SetToolTip($txtAt, 'Where to take the preview frame: 90, 1:30 or 0:01:30. Empty = middle of the video.')

$chkSoft = New-Object System.Windows.Forms.CheckBox
$chkSoft.Text = 'Soft / zoomed source (AI)'
$chkSoft.SetBounds(12, 384, 250, 22)
$chkEnh = New-Object System.Windows.Forms.CheckBox
$chkEnh.Text = 'Enhance - AI rebuilds detail (AI)'
$chkEnh.SetBounds(270, 384, 258, 22)
$tip.SetToolTip($chkSoft, 'For footage that was cropped/zoomed and looks soft with coarse grain: denoises at half size, where the grain is fine, then scales back.')
$tip.SetToolTip($chkEnh, 'Makes edges crisp by reconstructing detail. It invents detail: fine on objects, can subtly change real faces.')

$chkTest = New-Object System.Windows.Forms.CheckBox
$chkTest.Text = 'Test: first 10 s only'
$chkTest.SetBounds(350, 215, 180, 22)

$btnStart = New-Object System.Windows.Forms.Button
$btnStart.Text = 'Start'
$btnStart.SetBounds(12, 416,120, 32)
$btnCancel = New-Object System.Windows.Forms.Button
$btnCancel.Text = 'Cancel'
$btnCancel.Enabled = $false
$btnCancel.SetBounds(140, 416,90, 32)
$btnClear = New-Object System.Windows.Forms.Button
$btnClear.Text = 'Clear list'
$btnClear.SetBounds(238, 416,90, 32)
$btnPreview = New-Object System.Windows.Forms.Button
$btnPreview.Text = 'Preview'
$btnPreview.SetBounds(336, 416,82, 32)
$btnOpen = New-Object System.Windows.Forms.Button
$btnOpen.Text = 'Show output'
$btnOpen.Enabled = $false
$btnOpen.SetBounds(426, 416,102, 32)

$bar = New-Object System.Windows.Forms.ProgressBar
$bar.Maximum = 1000
$bar.SetBounds(12, 458, 516, 18)

$status = New-Object System.Windows.Forms.Label
$status.Text = 'Ready.'
$status.AutoEllipsis = $true
$status.SetBounds(12, 480, 516, 18)

$log = New-Object System.Windows.Forms.TextBox
$log.Multiline = $true
$log.ReadOnly = $true
$log.ScrollBars = 'Vertical'
$log.Font = New-Object System.Drawing.Font('Consolas', 8.5)
$log.SetBounds(12, 502, 516, 118)

$form.Controls.AddRange(@($drop, $list, $lblPreset, $cmbPreset, $lblStr, $trk, $lblStrVal, $lblSharp, $trkSharp, $lblSharpVal, $lblDet, $trkDet, $lblDetVal, $chkSoft, $chkEnh, $lblAt, $txtAt, $lblEnc, $cmbEnc, $chkTest,
        $btnStart, $btnCancel, $btnClear, $btnPreview, $btnOpen, $bar, $status, $log))

# Preset row goes above the settings: push everything below the file list down to make room.
foreach ($c in $form.Controls) { if ($c.Top -ge 210) { $c.Top += 34 } }
$form.ClientSize = New-Object System.Drawing.Size(540, ($form.ClientSize.Height + 34))

# m = method index, s = strength, sh = sharpen, d = keep detail
$profiles = [ordered]@{}
if ($hasAI) {
    $profiles['Soft / zoomed video - enhance'] = @{ m = 2; s = 2; sh = 1; d = 8; soft = $true; enh = $true }
    $profiles['Natural - people and faces'] = @{ m = 2; s = 3; sh = 2; d = 6; soft = $false; enh = $false }
    $profiles['Heavy noise'] = @{ m = 2; s = 6; sh = 2; d = 5; soft = $false; enh = $false }
}
$profiles['Quick (no AI)'] = @{ m = 0; s = 3; sh = 3; d = 5; soft = $false; enh = $false }

$lblProfile = New-Object System.Windows.Forms.Label
$lblProfile.Text = 'Preset'
$lblProfile.SetBounds(12, 218, 105, 20)
$cmbProfile = New-Object System.Windows.Forms.ComboBox
$cmbProfile.DropDownStyle = 'DropDownList'
$cmbProfile.SetBounds(120, 214, 210, 24)

# User presets live next to the script; built-in names cannot be overwritten or deleted.
$builtinNames = @($profiles.Keys)
$presetFile = Join-Path $PSScriptRoot 'presets.json'
if (Test-Path -LiteralPath $presetFile) {
    try {
        $saved = Get-Content -LiteralPath $presetFile -Raw -Encoding UTF8 | ConvertFrom-Json
        foreach ($u in $saved) {
            if ($u.name -and $builtinNames -notcontains $u.name -and [int]$u.m -lt $methods.Count) {
                $profiles[[string]$u.name] = @{
                    m = [Math]::Max(0, [int]$u.m)
                    s = [Math]::Min(10, [Math]::Max(1, [int]$u.s))
                    sh = [Math]::Min(10, [Math]::Max(0, [int]$u.sh))
                    d = [Math]::Min(10, [Math]::Max(0, [int]$u.d))
                    soft = [bool]$u.soft; enh = [bool]$u.enh
                }
            }
        }
    } catch {}  # unreadable file: start with built-ins only
}

function Update-ProfileList {
    $cmbProfile.Items.Clear()
    [void]$cmbProfile.Items.AddRange([object[]]@($profiles.Keys))
    [void]$cmbProfile.Items.Add('Custom')
}
Update-ProfileList

$btnSaveProfile = New-Object System.Windows.Forms.Button
$btnSaveProfile.Text = 'Save...'
$btnSaveProfile.SetBounds(348, 213, 86, 26)
$btnDelProfile = New-Object System.Windows.Forms.Button
$btnDelProfile.Text = 'Delete'
$btnDelProfile.Enabled = $false
$btnDelProfile.SetBounds(442, 213, 86, 26)
$form.Controls.AddRange(@($lblProfile, $cmbProfile, $btnSaveProfile, $btnDelProfile))
$script:applying = $false

$timer = New-Object System.Windows.Forms.Timer
$timer.Interval = 400

# ---------- helpers ----------
function Write-Log([string]$text) {
    $log.AppendText($text + "`r`n")
}

function Add-Files($paths) {
    foreach ($p in $paths) {
        if ((Test-Path -LiteralPath $p -PathType Leaf) -and -not $list.Items.Contains($p)) {
            [void]$list.Items.Add($p)
        }
    }
}

function Set-Busy([bool]$busy) {
    $btnStart.Enabled  = -not $busy
    $btnClear.Enabled  = -not $busy
    $cmbPreset.Enabled = -not $busy
    $cmbEnc.Enabled    = -not $busy
    $chkTest.Enabled   = -not $busy
    $btnPreview.Enabled = -not $busy
    $trk.Enabled       = -not $busy
    $trkSharp.Enabled  = -not $busy
    $trkDet.Enabled    = -not $busy
    $cmbProfile.Enabled = -not $busy
    $btnSaveProfile.Enabled = -not $busy
    $btnDelProfile.Enabled = (-not $busy) -and ($builtinNames -notcontains [string]$cmbProfile.SelectedItem) -and $profiles.Contains([string]$cmbProfile.SelectedItem)
    $chkSoft.Enabled   = (-not $busy) -and (Test-AI)
    $chkEnh.Enabled    = (-not $busy) -and (Test-AI)
    $btnCancel.Enabled = $busy
}

# Slider 1-10 -> ffmpeg filter. Fast = hqdn3d (2 = old "Light", 4 = old "Medium");
# Best quality = nlmeans (4 = old "Strong", 7 = old "Very strong").
function Get-Filter {
    $k = $trk.Value
    $inv = [System.Globalization.CultureInfo]::InvariantCulture
    if ($cmbPreset.SelectedIndex -eq 0) {
        $ls = 2.0 * $k
        $v = $ls, ($ls * 0.75), ($ls * 1.5), ($ls * 1.125) | ForEach-Object { $_.ToString('0.###', $inv) }
        $f = 'hqdn3d=' + ($v -join ':')
    } else {
        $f = "nlmeans=s=$($k + 1):p=7:r=9"
    }
    # keep detail: lay part of the original back over the denoised picture (uniform mix here;
    # the AI engine does it selectively, by texture)
    if ($trkDet.Value -gt 0) {
        $mix = (0.05 * $trkDet.Value).ToString('0.###', $inv)
        $f = "split[vo][vd];[vd]$f[vn];[vo][vn]blend=all_mode=normal:all_opacity=$mix"
    }
    # sharpen after denoise, luma only (chroma amount 0) so colour noise is not brought back
    if ($trkSharp.Value -gt 0) {
        $f += ',unsharp=5:5:{0}:5:5:0' -f (0.15 * $trkSharp.Value).ToString('0.###', $inv)
    }
    return $f
}

function Set-Profile([string]$name) {
    if (-not $profiles.Contains($name)) { return }
    $p = $profiles[$name]
    $script:applying = $true
    try {
        $cmbPreset.SelectedIndex = $p.m
        $trk.Value = $p.s
        $trkSharp.Value = $p.sh
        $trkDet.Value = $p.d
        $chkSoft.Checked = $p.soft
        $chkEnh.Checked = $p.enh
    } finally { $script:applying = $false }
}

function Save-ProfileFile {
    $rows = foreach ($n in $profiles.Keys) {
        if ($builtinNames -notcontains $n) {
            $p = $profiles[$n]
            [pscustomobject]@{ name = $n; m = $p.m; s = $p.s; sh = $p.sh; d = $p.d; soft = $p.soft; enh = $p.enh }
        }
    }
    ConvertTo-Json -InputObject @($rows) | Set-Content -LiteralPath $presetFile -Encoding UTF8
}

# Stores the current settings under $name. Returns an error message, or '' on success.
function Save-UserProfile([string]$name) {
    $name = $name.Trim()
    if (-not $name) { return 'Give the preset a name.' }
    if ($name -eq 'Custom' -or $builtinNames -contains $name) { return "'$name' is a built-in name. Pick another." }
    $profiles[$name] = @{
        m = $cmbPreset.SelectedIndex; s = $trk.Value; sh = $trkSharp.Value; d = $trkDet.Value
        soft = $chkSoft.Checked; enh = $chkEnh.Checked
    }
    try { Save-ProfileFile } catch { return "Could not write presets.json: $($_.Exception.Message)" }
    Update-ProfileList
    $cmbProfile.SelectedItem = $name
    return ''
}

function Remove-UserProfile([string]$name) {
    if ($builtinNames -contains $name -or -not $profiles.Contains($name)) { return }
    $profiles.Remove($name)
    try { Save-ProfileFile } catch {}
    Update-ProfileList
    $cmbProfile.SelectedItem = 'Custom'
}

# Any manual change to a setting turns the preset box to "Custom".
function Set-CustomProfile {
    if (-not $script:applying) { $cmbProfile.SelectedItem = 'Custom' }
}

function Test-AI { return $hasAI -and $cmbPreset.SelectedIndex -eq 2 }

# Arguments shared by AI preview and AI processing. Slider 1-10 -> noise level 3-30.
function Get-AIArgs {
    $sharp = (0.15 * $trkSharp.Value).ToString('0.###', [System.Globalization.CultureInfo]::InvariantCulture)
    $extra = ''
    if ($chkSoft.Checked) { $extra += ' --scale 0.5' }
    if ($chkEnh.Checked) { $extra += ' --enhance --enhance-dn 0.2' }
    return "`"$aiScript`" --sigma $(3 * $trk.Value) --sharpen $sharp --detail $($trkDet.Value)$extra --ffmpeg `"$ffmpeg`" --ffprobe `"$ffprobe`""
}

# The AI engine runs ffmpeg as child processes, so the whole tree has to go.
function Stop-Encode {
    if ($script:proc -and -not $script:proc.HasExited) {
        try { & taskkill.exe /PID $script:proc.Id /T /F 2>$null | Out-Null; [void]$script:proc.WaitForExit(5000) } catch {}
    }
}

function Get-SettingsLabel {
    $label = "{0}, strength {1}, sharpen {2}, detail {3}" -f [string]$cmbPreset.SelectedItem, $trk.Value, $trkSharp.Value, $trkDet.Value
    if (Test-AI) {
        if ($chkSoft.Checked) { $label += ', soft source' }
        if ($chkEnh.Checked) { $label += ', enhance' }
    }
    return $label
}

function Update-DetailLabel {
    $k = $trkDet.Value
    $lblDetVal.Text = if ($k -eq 0) { 'off' } else { "$k / 10" }
}

function Update-SharpLabel {
    $k = $trkSharp.Value
    $lblSharpVal.Text = if ($k -eq 0) { 'off' } else { "$k / 10" }
}

function Update-StrengthLabel {
    $k = $trk.Value
    $word = if ($k -le 3) { 'light' } elseif ($k -le 6) { 'medium' } elseif ($k -le 8) { 'strong' } else { 'very strong' }
    $lblStrVal.Text = "$k / 10  -  $word"
}

function Get-Duration([string]$path) {
    $raw = & $ffprobe -v error -show_entries format=duration -of csv=p=0 $path 2>$null | Select-Object -First 1
    $dur = 0.0
    [void][double]::TryParse([string]$raw, [System.Globalization.NumberStyles]::Float,
        [System.Globalization.CultureInfo]::InvariantCulture, [ref]$dur)
    return $dur
}

# "90", "1:30" or "0:01:30" -> seconds. Returns -1 if not a valid time.
function ConvertTo-Seconds([string]$text) {
    $t = 0.0
    foreach ($part in $text.Trim().Replace(',', '.').Split(':')) {
        $v = 0.0
        if (-not [double]::TryParse($part, [System.Globalization.NumberStyles]::Float,
                [System.Globalization.CultureInfo]::InvariantCulture, [ref]$v) -or $v -lt 0) { return -1.0 }
        $t = $t * 60 + $v
    }
    return $t
}

# Before/after still: 100% crop from the centre of the frame, original left, denoised right.
function Show-Preview {
    if ($list.Items.Count -eq 0) { $status.Text = 'Add a video first.'; return }
    $in = if ($list.SelectedItem) { [string]$list.SelectedItem } else { [string]$list.Items[0] }
    $presetName = Get-SettingsLabel
    $filter = Get-Filter
    $dur = Get-Duration $in
    $at = $dur / 2
    if ($txtAt.Text.Trim()) {
        $at = ConvertTo-Seconds $txtAt.Text
        if ($at -lt 0) { $status.Text = 'Preview time not understood. Use 90, 1:30 or 0:01:30.'; return }
        if ($dur -gt 0 -and $at -ge $dur) {
            $status.Text = "Preview time is past the end ($([TimeSpan]::FromSeconds([Math]::Floor($dur)).ToString('g')))."
            return
        }
        # leave room for the 9 frames the preview needs
        if ($dur -gt 1) { $at = [Math]::Min($at, $dur - 1) }
    }
    $png = Join-Path $workDir 'preview.png'
    Remove-Item -LiteralPath $png -ErrorAction SilentlyContinue

    $status.Text = 'Rendering preview...'
    $form.Cursor = 'WaitCursor'
    $form.Refresh()
    try {
        $mid = $at.ToString('0.###', [System.Globalization.CultureInfo]::InvariantCulture)
        $psi = New-Object System.Diagnostics.ProcessStartInfo
        $psi.FileName = $ffmpeg
        # crop first so slow filters only work on the visible area; skip 8 frames so temporal filters settle
        $psi.Arguments = "-hide_banner -v error -y -ss $mid -i `"$in`" -filter_complex `"[0:v:0]crop=min(iw\,640):min(ih\,720),format=yuv420p,split[a][b];[b]$($filter)[c];[a][c]hstack,select=gte(n\,8)`" -frames:v 1 preview.png"
        if (Test-AI) {
            $psi.FileName = $aiPython
            $psi.Arguments = "$(Get-AIArgs) --input `"$in`" --preview preview.png --at $mid"
        }
        $psi.WorkingDirectory = $workDir
        $psi.UseShellExecute = $false
        $psi.CreateNoWindow = $true
        $p = [System.Diagnostics.Process]::Start($psi)
        if (-not $p.WaitForExit(120000)) { try { $p.Kill() } catch {} }
        $p.Dispose()
    } finally { $form.Cursor = 'Default' }

    if (-not (Test-Path -LiteralPath $png)) { $status.Text = 'Preview failed (clip too short or unreadable).'; return }
    $status.Text = 'Ready.'

    $img = [System.Drawing.Image]::FromStream((New-Object System.IO.MemoryStream(, [System.IO.File]::ReadAllBytes($png))))
    $pv = New-Object System.Windows.Forms.Form
    $stamp = [TimeSpan]::FromSeconds([Math]::Floor($at)).ToString('g')
    $pv.Text = "$presetName  -  left: original  |  right: denoised  -  $([System.IO.Path]::GetFileName($in)) @ $stamp"
    $pv.AutoScroll = $true
    $pv.StartPosition = 'CenterScreen'
    $area = [System.Windows.Forms.Screen]::PrimaryScreen.WorkingArea
    $pv.ClientSize = New-Object System.Drawing.Size([Math]::Min($img.Width, $area.Width - 80), [Math]::Min($img.Height, $area.Height - 80))
    $pb = New-Object System.Windows.Forms.PictureBox
    $pb.SizeMode = 'AutoSize'
    $pb.Image = $img
    $pv.Controls.Add($pb)
    $pv.Add_FormClosed({ param($s, $e) $s.Controls[0].Image.Dispose() })
    $pv.Show($form)
}

# Tail of a file ffmpeg may still be writing to.
function Read-Tail([string]$path, [int]$bytes) {
    if (-not (Test-Path -LiteralPath $path)) { return '' }
    try {
        $fs = [System.IO.File]::Open($path, 'Open', 'Read', 'ReadWrite')
        try {
            if ($fs.Length -gt $bytes) { [void]$fs.Seek(-$bytes, 'End') }
            $sr = New-Object System.IO.StreamReader($fs)
            return $sr.ReadToEnd()
        } finally { $fs.Dispose() }
    } catch { return '' }
}

function Get-OutputPath([string]$in, [bool]$test) {
    $dir  = [System.IO.Path]::GetDirectoryName($in)
    $name = [System.IO.Path]::GetFileNameWithoutExtension($in)
    $ext  = [System.IO.Path]::GetExtension($in).ToLower()
    # mkv takes any audio codec, so "-c:a copy" cannot fail on odd containers
    if ($ext -notin '.mp4', '.mov', '.mkv') { $ext = '.mkv' }
    $suffix = if ($test) { '_denoised_test' } else { '_denoised' }
    $out = Join-Path $dir ($name + $suffix + $ext)
    $n = 2
    while (Test-Path -LiteralPath $out) {
        $out = Join-Path $dir ("{0}{1}_{2}{3}" -f $name, $suffix, $n, $ext)
        $n++
    }
    return $out
}

function Stop-Queue([string]$message) {
    $timer.Stop()
    $script:proc = $null
    Set-Busy $false
    $status.Text = $message
    if ($AutoRun) { $form.Close() }
}

function Start-Next {
    if ($script:index -ge $list.Items.Count) {
        $bar.Value = 0
        $msg = if ($script:failed) { "Finished with $($script:failed) error(s)." } else { 'All done.' }
        Stop-Queue $msg
        return
    }

    $in = [string]$list.Items[$script:index]
    $list.SelectedIndex = $script:index
    $test = $chkTest.Checked
    $out = Get-OutputPath $in $test
    $script:current = $in
    $script:output = $out

    $dur = Get-Duration $in
    if ($test -and ($dur -le 0 -or $dur -gt 10)) { $dur = 10.0 }
    $script:duration = $dur

    Remove-Item -LiteralPath $progressFile, $reportFile -ErrorAction SilentlyContinue

    $enc = $encoders[[string]$cmbEnc.SelectedItem]
    if ($enc -match 'hevc' -and [System.IO.Path]::GetExtension($out) -ne '.mkv') { $enc += ' -tag:v hvc1' }
    $limit = if ($test) { '-t 10 ' } else { '' }
    $filter = Get-Filter

    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $ffmpeg
    $psi.Arguments = "-hide_banner -n -nostats -progress progress.txt -i `"$in`" $limit-map 0:v:0 -map 0:a? -vf $filter $enc -c:a copy `"$out`""
    if (Test-AI) {
        $psi.FileName = $aiPython
        $testFlag = if ($test) { ' --test' } else { '' }
        $psi.Arguments = "$(Get-AIArgs) --input `"$in`" --output `"$out`" --encoder `"$enc`"$testFlag"
    }
    $psi.WorkingDirectory = $workDir
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $true
    # relative file name: avoids escaping the drive colon in FFREPORT
    $psi.EnvironmentVariables['FFREPORT'] = 'file=report.log:level=24'

    Write-Log ("[{0}/{1}] {2}  ({3})" -f ($script:index + 1), $list.Items.Count, [System.IO.Path]::GetFileName($in), (Get-SettingsLabel))
    $bar.Value = 0
    $script:clock.Restart()
    $script:proc = [System.Diagnostics.Process]::Start($psi)
    $timer.Start()
}

# ---------- events ----------
$timer.Add_Tick({
        if (-not $script:proc) { return }
        $name = [System.IO.Path]::GetFileName($script:current)

        if (-not $script:proc.HasExited) {
            $tail = Read-Tail $progressFile 2048
            $m = [regex]::Matches($tail, 'out_time_us=(\d+)')
            if ($m.Count -and $script:duration -gt 0) {
                $sec = [double]$m[$m.Count - 1].Groups[1].Value / 1e6
                $frac = [Math]::Min(1.0, $sec / $script:duration)
                $bar.Value = [int]($frac * 1000)
                $eta = ''
                if ($frac -ge 0.02) {
                    $left = [TimeSpan]::FromSeconds($script:clock.Elapsed.TotalSeconds * (1 - $frac) / $frac)
                    $eta = '  -  {0}:{1:00}:{2:00} left' -f [int][Math]::Floor($left.TotalHours), $left.Minutes, $left.Seconds
                }
                $status.Text = "{0}/{1}  {2}  -  {3:P0}{4}" -f ($script:index + 1), $list.Items.Count, $name, $frac, $eta
            } else {
                $status.Text = "{0}/{1}  {2}  -  working..." -f ($script:index + 1), $list.Items.Count, $name
            }
            return
        }

        $timer.Stop()
        $code = $script:proc.ExitCode
        $script:proc.Dispose()
        $script:proc = $null

        if ($script:cancelled) {
            Remove-Item -LiteralPath $script:output -ErrorAction SilentlyContinue
            Write-Log '  cancelled.'
            $bar.Value = 0
            Stop-Queue 'Cancelled.'
            return
        }

        if ($code -eq 0 -and (Test-Path -LiteralPath $script:output)) {
            Write-Log "  OK -> $($script:output)"
            $script:lastOut = $script:output
            $btnOpen.Enabled = $true
        } else {
            $script:failed++
            Remove-Item -LiteralPath $script:output -ErrorAction SilentlyContinue
            Write-Log "  FAILED (exit code $code)"
            $err = (Read-Tail $reportFile 1500).Trim() -split "`r?`n" | Select-Object -Last 6
            foreach ($line in $err) { Write-Log "    $line" }
        }
        $script:index++
        Start-Next
    })

$btnStart.Add_Click({
        if ($list.Items.Count -eq 0) { $status.Text = 'Add a video first.'; return }
        $script:index = 0
        $script:failed = 0
        $script:cancelled = $false
        Set-Busy $true
        Start-Next
    })

$btnCancel.Add_Click({
        if ($script:proc -and -not $script:proc.HasExited) {
            $script:cancelled = $true
            Stop-Encode
        }
    })

$btnClear.Add_Click({ $list.Items.Clear(); $status.Text = 'Ready.' })

$btnPreview.Add_Click({ Show-Preview })

$trk.Add_ValueChanged({ Update-StrengthLabel })
Update-StrengthLabel
$trkSharp.Add_ValueChanged({ Update-SharpLabel })
Update-SharpLabel
$cmbPreset.Add_SelectedIndexChanged({ $chkSoft.Enabled = Test-AI; $chkEnh.Enabled = Test-AI })
$chkSoft.Enabled = Test-AI
$chkEnh.Enabled = Test-AI
$trkDet.Add_ValueChanged({ Update-DetailLabel })

$cmbProfile.Add_SelectedIndexChanged({
        $name = [string]$cmbProfile.SelectedItem
        Set-Profile $name
        $btnDelProfile.Enabled = $btnStart.Enabled -and $profiles.Contains($name) -and ($builtinNames -notcontains $name)
    })

$btnSaveProfile.Add_Click({
        Add-Type -AssemblyName Microsoft.VisualBasic
        $current = [string]$cmbProfile.SelectedItem
        $suggest = if ($profiles.Contains($current) -and $builtinNames -notcontains $current) { $current } else { '' }
        $name = [Microsoft.VisualBasic.Interaction]::InputBox('Name for the current settings (an existing name of yours is overwritten):', 'Save preset', $suggest)
        if (-not $name.Trim()) { return }  # cancelled
        $err = Save-UserProfile $name
        $status.Text = if ($err) { $err } else { "Preset '$($name.Trim())' saved." }
    })

$btnDelProfile.Add_Click({
        $name = [string]$cmbProfile.SelectedItem
        $answer = [System.Windows.Forms.MessageBox]::Show($form, "Delete preset '$name'?", 'Delete preset', 'YesNo', 'Question')
        if ($answer -eq 'Yes') { Remove-UserProfile $name; $status.Text = "Preset '$name' deleted." }
    })
$cmbPreset.Add_SelectedIndexChanged({ Set-CustomProfile })
foreach ($t in $trk, $trkSharp, $trkDet) { $t.Add_ValueChanged({ Set-CustomProfile }) }
foreach ($c in $chkSoft, $chkEnh) { $c.Add_CheckedChanged({ Set-CustomProfile }) }
$cmbProfile.SelectedIndex = 0   # first preset is the default
if ($AutoRunMethod -ge 0) { $cmbPreset.SelectedIndex = $AutoRunMethod }
Update-DetailLabel

# Delete key removes the selected file (not while a job is running: the queue is indexed)
$list.Add_KeyDown({
        param($s, $e)
        if ($e.KeyCode -eq 'Delete' -and $btnStart.Enabled -and $list.SelectedIndex -ge 0) {
            $list.Items.RemoveAt($list.SelectedIndex)
        }
    })

$btnOpen.Add_Click({
        if ($script:lastOut -and (Test-Path -LiteralPath $script:lastOut)) {
            Start-Process explorer.exe "/select,`"$($script:lastOut)`""
        }
    })

$drop.Add_Click({
        $dlg = New-Object System.Windows.Forms.OpenFileDialog
        $dlg.Multiselect = $true
        $dlg.Filter = 'Video files|*.mp4;*.mov;*.mkv;*.avi;*.mts;*.m2ts;*.webm;*.wmv;*.mxf|All files|*.*'
        if ($dlg.ShowDialog($form) -eq 'OK') { Add-Files $dlg.FileNames }
    })

$onDragEnter = {
    param($s, $e)
    if ($e.Data.GetDataPresent([System.Windows.Forms.DataFormats]::FileDrop)) { $e.Effect = 'Copy' }
}
$onDragDrop = {
    param($s, $e)
    Add-Files $e.Data.GetData([System.Windows.Forms.DataFormats]::FileDrop)
}
foreach ($c in $form.Controls + $form) {
    $c.AllowDrop = $true
    $c.Add_DragEnter($onDragEnter)
    $c.Add_DragDrop($onDragDrop)
}

$form.Add_FormClosing({
        if ($script:proc -and -not $script:proc.HasExited) {
            Stop-Encode
            Remove-Item -LiteralPath $script:output -ErrorAction SilentlyContinue
        }
    })

if ($AutoRun) {
    $form.Add_Shown({ Add-Files @($AutoRun); $txtAt.Text = $AutoRunAt
            # snapshot of the window, so layout can be checked without a screen
            $shot = New-Object System.Drawing.Bitmap($form.Width, $form.Height)
            $form.DrawToBitmap($shot, (New-Object System.Drawing.Rectangle(0, 0, $form.Width, $form.Height)))
            $shot.Save((Join-Path $workDir 'ui.png')); $shot.Dispose()
            foreach ($o in $AutoRunOpts -split ',') {
                switch -Regex ($o.Trim()) {
                    '^soft$' { $chkSoft.Checked = $true }
                    '^enhance$' { $chkEnh.Checked = $true }
                    '^test$' { $chkTest.Checked = $true }
                    '^s=(\d+)$' { $trk.Value = [int]$Matches[1] }
                    '^d=(\d+)$' { $trkDet.Value = [int]$Matches[1] }
                    '^sh=(\d+)$' { $trkSharp.Value = [int]$Matches[1] }
                    '^p=(.+)$' { $cmbProfile.SelectedItem = $Matches[1] }
                    '^save=(.+)$' { Write-Log "save: [$(Save-UserProfile $Matches[1])]" }
                    '^del=(.+)$' { Remove-UserProfile $Matches[1] }
                }
            } Show-Preview; Write-Log "preview: $($status.Text)"; $btnStart.PerformClick() })
}

[void]$form.ShowDialog()
if ($AutoRun) { $log.Text; exit $script:failed }
