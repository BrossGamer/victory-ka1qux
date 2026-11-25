# Environment profile for Intel Arc B580 (Battlemage / Xe2-HPG, 12 GB).
# Dot-source in PowerShell before running `victory`:
#
#   . .\scripts\arc-b580.ps1                 # B580 12 GB, conservative stability defaults
#   . .\scripts\arc-b580.ps1 -Fast           # async relay + wider card buffers

param(
    [string]$Adapter = "B580",
    [switch]$Fast
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
# - 12 GB dedicated VRAM: keep texture + card buffer budgets under ~8.5 GB so D3D12
#   does not evict allocations into shared system RAM
$env:VICTORY_INDIRECT_CLAMP       = "0"
$env:VICTORY_SKIP_MOVIES          = "1"
$env:VICTORY_TEXTURE_BUDGET_MB    = "6144"
$env:VICTORY_TEXTURE_ROOM_PERCENT = "85"
$env:VICTORY_CARD_BUFFERS_MIN_MB  = if ($Fast) { "2" } else { "16" }
$env:VICTORY_CARD_BUFFERS_MB      = if ($Fast) { "768" } else { "512" }

if ($Fast) {
    $env:VICTORY_RELAY     = "2"
    $env:VICTORY_WIDE_BACK = "1"
    Remove-Item Env:VICTORY_LATE_WAIT_LISTS -ErrorAction SilentlyContinue
} else {
    $env:VICTORY_RELAY           = "0"
    $env:VICTORY_WIDE_BACK       = "0"
    $env:VICTORY_LATE_WAIT_LISTS = "128"
}

Write-Host "Intel Arc B580 (12 GB) profile active (Fast=$($Fast.IsPresent)):"
Write-Host "  VICTORY_ADAPTER              = $env:VICTORY_ADAPTER"
Write-Host "  VICTORY_TEXTURE_BUDGET_MB    = $env:VICTORY_TEXTURE_BUDGET_MB"
Write-Host "  VICTORY_CARD_BUFFERS_MIN_MB  = $env:VICTORY_CARD_BUFFERS_MIN_MB"
Write-Host "  VICTORY_CARD_BUFFERS_MB      = $env:VICTORY_CARD_BUFFERS_MB"
Write-Host "  VICTORY_RELAY / WIDE_BACK    = $env:VICTORY_RELAY / $env:VICTORY_WIDE_BACK"
Write-Host "  VICTORY_INDIRECT_CLAMP       = $env:VICTORY_INDIRECT_CLAMP"
