# §4-MeMo Sprint M.5 — push binary + run KSTE-routing smoke on Knack's S22U.
#
# Prerequisites (carried over from M.1):
#   - Memory model (qwen25-coder-0.5b-memory.sp-model + .sp-tokenizer)
#     present at /data/local/tmp/ on the device. M.1 push-and-run handles this
#     once; M.5 does NOT re-push them by default (use -PushModel to override).
#   - libsp_compute_skel.so already at /data/local/tmp/ (Sprint K.beta).
#
# Usage:
#   .\m5_push_and_run.ps1 [-Queries 100] [-DetCycles 100] [-ReportPath /data/local/tmp/m5_report.json] [-PushModel]
#
# What it does:
#   1. (optional) Pushes Memory model + tokenizer if -PushModel flag is set.
#   2. Pushes the freshly cross-compiled sp_memo_m5_routing_smoke binary.
#   3. Runs the smoke; pulls the JSON report back to host scripts/.

param(
    [int]$Queries = 100,
    [int]$DetCycles = 100,
    [string]$ReportPath = "/data/local/tmp/m5_report.json",
    [string]$BinaryPath = "..\target\aarch64-linux-android\release\sp_memo_m5_routing_smoke",
    [switch]$PushModel
)

$ErrorActionPreference = "Stop"

$MemoModel = "/data/local/tmp/qwen25-coder-0.5b-memory.sp-model"
$MemoTok   = "/data/local/tmp/qwen25-coder-0.5b-memory.sp-tokenizer"

$HostMemoModel = "D:\F\shannon-prime-repos\models\qwen25-coder-0.5b-memory.sp-model"
$HostMemoTok   = "D:\F\shannon-prime-repos\models\qwen25-coder-0.5b-memory.sp-tokenizer"

Write-Host "[M.5] adb device check..."
adb devices

if ($PushModel) {
    Write-Host "[M.5] pushing Memory model to device..."
    adb push $HostMemoModel $MemoModel
    Write-Host "[M.5] pushing Memory tokenizer to device..."
    adb push $HostMemoTok $MemoTok
} else {
    Write-Host "[M.5] skipping model push (assuming M.1 left it on device); -PushModel forces re-push"
    Write-Host "[M.5] checking pre-existing model on device..."
    adb shell "ls -la $MemoModel $MemoTok"
}

# Push the smoke binary
$BinaryFull = Resolve-Path $BinaryPath -ErrorAction SilentlyContinue
if (-not $BinaryFull) {
    Write-Host "[M.5] ERROR: binary not found at $BinaryPath"
    Write-Host "[M.5] Build first via:"
    Write-Host "[M.5]   cd tools\sp_daemon"
    Write-Host "[M.5]   `$env:SP_SYSTEM_BUILD_DIR='D:\F\shannon-prime-repos\shannon-prime-system-engine\build-android-libs'"
    Write-Host "[M.5]   `$env:SP_SYSTEM_INCLUDE='D:\F\shannon-prime-repos\engine-m5\lib\shannon-prime-system\include'"
    Write-Host "[M.5]   .\build-android.bat --bin sp_memo_m5_routing_smoke"
    exit 1
}
Write-Host "[M.5] pushing binary $BinaryFull..."
adb push $BinaryFull /data/local/tmp/sp_memo_m5_routing_smoke
adb shell chmod +x /data/local/tmp/sp_memo_m5_routing_smoke

Write-Host "[M.5] running smoke (queries=$Queries det-cycles=$DetCycles)..."
$cmd = @"
ADSP_LIBRARY_PATH="/data/local/tmp;" /data/local/tmp/sp_memo_m5_routing_smoke `
  $MemoModel $MemoTok `
  --queries $Queries --det-cycles $DetCycles --report-json $ReportPath
"@
adb shell $cmd

Write-Host "[M.5] pulling report..."
$LocalReport = Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) "m5_report.json"
adb pull $ReportPath $LocalReport
Write-Host "[M.5] report pulled to $LocalReport"
Write-Host ""
Get-Content $LocalReport
