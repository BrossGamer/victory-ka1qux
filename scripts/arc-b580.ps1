# Environment profile for Intel Arc B580 (Battlemage / Xe2-HPG, 12 GB).
# Dot-source in PowerShell before running `victory`:
#
#   . .\scripts\arc-b580.ps1                 # B580 12 GB (async relay, VICTORY_RAYS=0; avoids 2s Windows TDR)
#   . .\scripts\arc-b580.ps1 -Rays           # enable title ray tracing (pre-warm XboxBvhBuilder.dll pipelines first)
#   . .\scripts\arc-b580.ps1 -Conservative   # synchronous relay=0 / late-waits=128 (may need Windows TdrDelay >= 10)

param(
    [string]$Adapter = "B580",
    [switch]$Fast,
    [switch]$Conservative,
    [switch]$Rays
)

$fetch = Join-Path $PSScriptRoot "fetch-agility.ps1"
$root = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path (Join-Path $root "target\diag\D3D12\D3D12Core.dll"))) {
    & powershell -ExecutionPolicy Bypass -File $fetch
}

$env:VICTORY_ADAPTER = $Adapter

# Arc B580 (Xe2-HPG, 12 GB VRAM):
# - wave lanes 16..32 (compute wave ops pinned to WaveSize(32) automatically)
# - 64-bit typed + heap atomics natively supported on Xe2
# - Leave VICTORY_TEXTURE_BUDGET_MB unset so `resource::room()` tracks live DXGI
#   `QueryVideoMemoryInfo` up to 92% of the card's budget instead of starving
#   character bakes and bindless world textures under a static byte cap.
Remove-Item Env:VICTORY_TEXTURE_BUDGET_MB -ErrorAction SilentlyContinue
$env:VICTORY_INDIRECT_CLAMP       = "0"
$env:VICTORY_SKIP_MOVIES          = "1"
$env:VICTORY_RAYS                 = if ($Rays) { "1" } else { "0" }
$env:VICTORY_TEXTURE_ROOM_PERCENT = "92"

$useAsync = -not $Conservative
$env:VICTORY_CARD_BUFFERS_MIN_MB  = if ($useAsync) { "2" } else { "16" }
$env:VICTORY_CARD_BUFFERS_MB      = if ($useAsync) { "768" } else { "512" }

if ($useAsync) {
    $env:VICTORY_RELAY     = "2"
    $env:VICTORY_WIDE_BACK = "1"
    Remove-Item Env:VICTORY_LATE_WAIT_LISTS -ErrorAction SilentlyContinue
} else {
    $env:VICTORY_RELAY           = "0"
    $env:VICTORY_WIDE_BACK       = "0"
    $env:VICTORY_LATE_WAIT_LISTS = "128"
}

$mode = if ($useAsync) { "Async" } else { "Conservative" }
Write-Host "Intel Arc B580 (12 GB) profile active (Mode=$mode, Rays=$($Rays.IsPresent)):"
Write-Host "  VICTORY_ADAPTER              = $env:VICTORY_ADAPTER"
Write-Host "  VICTORY_TEXTURE_ROOM_PERCENT = $env:VICTORY_TEXTURE_ROOM_PERCENT% (live DXGI budget)"
Write-Host "  VICTORY_CARD_BUFFERS_MIN_MB  = $env:VICTORY_CARD_BUFFERS_MIN_MB"
Write-Host "  VICTORY_CARD_BUFFERS_MB      = $env:VICTORY_CARD_BUFFERS_MB"
Write-Host "  VICTORY_RELAY / WIDE_BACK    = $env:VICTORY_RELAY / $env:VICTORY_WIDE_BACK"
Write-Host "  VICTORY_RAYS                 = $env:VICTORY_RAYS"
Write-Host "  VICTORY_INDIRECT_CLAMP       = $env:VICTORY_INDIRECT_CLAMP"
