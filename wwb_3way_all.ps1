# 3-way WWB (HF vs Optimum vs GenAI) on the Windows cloud VM for:
#   1. LTX-Video text-to-video
#   2. LTX-Video text-to-video + LoRA   (2-way: HF vs GenAI -- Optimum rejects adapters)
#   3. LTX-Video image-to-video
#   4. LTX-2     text-to-video          (HF leg runs in bf16 via WWB_HF_DTYPE; fp32 needs ~190 GB)
#
# Usage (PowerShell, from the repo root on branch ltx2-wwb-full):
#   .\wwb_3way_all.ps1                       # setup + all four runs
#   .\wwb_3way_all.ps1 -Clean                # wipe previous outputs in $WorkDir first
#   .\wwb_3way_all.ps1 -SkipSetup            # reuse existing build/env
#   .\wwb_3way_all.ps1 -Modes t2v,i2v        # subset
# Run it, then DISCONNECT the RDP session (do not sign out). Everything is logged to $WorkDir\run.log.

param(
    [string[]]$Modes = @("t2v", "lora", "i2v", "ltx2"),
    [switch]$SkipSetup,
    [switch]$Clean,
    [string]$Ltx2HfDtype = "bfloat16",
    [int]$NumSamples = 10
)

# ---- paths ------------------------------------------------------------------
$Repo        = "C:\Users\devcloud\testing\openvino.genai"
$Venv        = "C:\Users\devcloud\testing\.venv"
$Branch      = "ltx2-wwb-full"
$Ltx1Ov      = "C:\Users\devcloud\testing\ltx-video-fp32"
$Ltx2Ov      = "C:\Users\devcloud\testing\ltx2-fp32-ov"
$LoraDir     = "C:\Users\devcloud\testing\lora"
$LoraFile    = Join-Path $LoraDir "ltx_pixel_pytorch_lora_weights.safetensors"   # auto-downloaded if missing
$LoraAlpha   = "3.0"
$WorkDir     = "C:\Users\devcloud\testing\wwb_3way"
$Device      = "CPU"
# OV legs run in f32 so precision noise doesn't leak into the score (bf16 default on this CPU)
$OvConfig    = '{\"INFERENCE_PRECISION_HINT\":\"f32\"}'
# ------------------------------------------------------------------------------

$ErrorActionPreference = "Continue"
if ($Clean -and (Test-Path $WorkDir)) { Remove-Item -Recurse -Force $WorkDir; Write-Host "cleared $WorkDir" }
New-Item -ItemType Directory -Force $WorkDir | Out-Null
Start-Transcript -Path (Join-Path $WorkDir "run.log") -Append | Out-Null
$Wwb = Join-Path $Repo "tools\who_what_benchmark\whowhatbench\wwb.py"
$env:PYTHONPATH = Join-Path $Repo "build"

function Step($msg) { Write-Host ""; Write-Host "===== $msg =====" -ForegroundColor Cyan }

if (-not $SkipSetup) {
    Step "[setup] branch $Branch"
    Set-Location $Repo
    git fetch origin $Branch
    git checkout -B $Branch FETCH_HEAD
    git reset --hard FETCH_HEAD
    git submodule update --init --recursive
    git log --oneline -1

    Step "[setup] OpenVINO nightly (branch requires 2026.5)"
    & "$Venv\Scripts\Activate.ps1"
    pip install --pre -U openvino openvino-tokenizers --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly
    python -c "import openvino; print('OpenVINO', openvino.__version__)"

    Step "[setup] rebuild openvino_genai"
    $ov = (Resolve-Path "$Venv\Lib\site-packages\openvino\cmake").Path
    cmake -S $Repo -B "$Repo\build" -DOpenVINO_DIR="$ov" -DOpenVINO_DIR_PY="$ov" -DCMAKE_BUILD_TYPE=Release -DENABLE_SAMPLES=OFF -DENABLE_TESTS=OFF -DENABLE_JS=OFF
    cmake --build "$Repo\build" --config Release --parallel --target py_openvino_genai
    # a pip-installed wheel silently shadows the local build
    pip uninstall -y openvino-genai 2>$null
    python -c "import openvino_genai; print('openvino_genai from', openvino_genai.__file__)"

    Step "[setup] whowhatbench (no-deps: protect pinned transformers/optimum/diffusers)"
    pip install --no-deps -e "$Repo\tools\who_what_benchmark"
    pip install --no-deps sentence-transformers
    pip install scipy scikit-image imageio "imageio-ffmpeg<=0.6.0" lpips datasets
    python -c "import whowhatbench; from diffusers import LTX2Pipeline; print('wwb ok')"

    if (($Modes -contains "lora") -and -not (Test-Path $LoraFile)) {
        Step "[setup] LoRA adapter"
        python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download('svjack/ltx_video_pixel_early_lora', 'ltx_pixel_pytorch_lora_weights.safetensors', local_dir=r'$LoraDir'))"
    }
} else {
    & "$Venv\Scripts\Activate.ps1"
    Set-Location $Repo
}

Step "[preflight]"
foreach ($m in @($Ltx1Ov, $Ltx2Ov)) { if (-not (Test-Path "$m\model_index.json")) { Write-Host "!!! missing model: $m" -ForegroundColor Red } }
if (($Modes -contains "i2v") -and -not (Test-Path "$Ltx1Ov\vae_encoder")) {
    Write-Host "!!! $Ltx1Ov has no vae_encoder (export with --task image-to-video); skipping i2v" -ForegroundColor Yellow
    $Modes = $Modes | Where-Object { $_ -ne "i2v" }
}
Write-Host "modes: $($Modes -join ', ')   samples: $NumSamples   ltx2 HF dtype: $Ltx2HfDtype"

# Runs one comparison leg; output goes to the transcript.
function Leg($title, [string[]]$args) {
    Step $title
    & python $Wwb @args
    if ($LASTEXITCODE -ne 0) { Write-Host "!!! $title FAILED (exit $LASTEXITCODE)" -ForegroundColor Red }
}

# Full 3-way: HF gt, Optimum gt, then optimum_vs_hf (data only), genai_vs_hf, genai_vs_optimum (data only).
function ThreeWay($name, $modelType, $hfModel, $ovModel) {
    $d = Join-Path $WorkDir $name
    New-Item -ItemType Directory -Force "$d\gt_hf", "$d\gt_optimum" | Out-Null
    $common = @("--model-type", $modelType, "--device", $Device, "--num-samples", $NumSamples)

    Leg "[$name] HF ground truth"       ($common + @("--hf", "--base-model", $hfModel, "--gt-data", "$d\gt_hf\gt.csv"))
    Leg "[$name] Optimum ground truth"  ($common + @("--ov-config", $OvConfig, "--base-model", $ovModel, "--gt-data", "$d\gt_optimum\gt.csv"))
    Leg "[$name] Optimum vs HF"         (@("--model-type", $modelType, "--gt-data", "$d\gt_hf\gt.csv", "--target-data", "$d\gt_optimum\gt.csv", "--output", "$d\optimum_vs_hf"))
    Leg "[$name] GenAI vs HF"           ($common + @("--ov-config", $OvConfig, "--genai", "--target-model", $ovModel, "--gt-data", "$d\gt_hf\gt.csv", "--output", "$d\genai_vs_hf"))
    Leg "[$name] GenAI vs Optimum"      (@("--model-type", $modelType, "--gt-data", "$d\gt_optimum\gt.csv", "--target-data", "$d\genai_vs_hf\target.csv", "--output", "$d\genai_vs_optimum"))
}

if ($Modes -contains "t2v") { ThreeWay "ltx1_t2v" "text-to-video"  "Lightricks/LTX-Video" $Ltx1Ov }
if ($Modes -contains "i2v") { ThreeWay "ltx1_i2v" "image-to-video" "Lightricks/LTX-Video" $Ltx1Ov }

if ($Modes -contains "lora") {
    # Optimum rejects --adapters, so LoRA is HF+LoRA vs GenAI+LoRA only.
    $d = Join-Path $WorkDir "ltx1_t2v_lora"
    New-Item -ItemType Directory -Force "$d\gt_hf" | Out-Null
    $common = @("--model-type", "text-to-video", "--device", $Device, "--num-samples", $NumSamples, "--adapters", $LoraFile, "--alphas", $LoraAlpha)
    Leg "[ltx1_t2v_lora] HF+LoRA ground truth"  ($common + @("--hf", "--base-model", "Lightricks/LTX-Video", "--gt-data", "$d\gt_hf\gt.csv"))
    Leg "[ltx1_t2v_lora] GenAI+LoRA vs HF+LoRA" ($common + @("--ov-config", $OvConfig, "--genai", "--target-model", $Ltx1Ov, "--gt-data", "$d\gt_hf\gt.csv", "--output", "$d\genai_vs_hf"))
}

if ($Modes -contains "ltx2") {
    # WWB_HF_DTYPE is read by the (temp-branch) HF loader; fp32 LTX-2 would need ~190 GB resident.
    $env:WWB_HF_DTYPE = $Ltx2HfDtype
    ThreeWay "ltx2_t2v" "text-to-video" "Lightricks/LTX-2" $Ltx2Ov
    Remove-Item Env:WWB_HF_DTYPE -ErrorAction SilentlyContinue
}

Step "SUMMARY"
Get-ChildItem $WorkDir -Recurse -Filter metrics.csv | Sort-Object FullName | ForEach-Object {
    $rel = $_.FullName.Substring($WorkDir.Length + 1) -replace '\\metrics\.csv$', ''
    "{0,-36} {1}" -f $rel, (Import-Csv $_.FullName)[0].similarity
}
Stop-Transcript | Out-Null
