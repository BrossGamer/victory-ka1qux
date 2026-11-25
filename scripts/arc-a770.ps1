# Environment profile for Intel Arc A770 (Alchemist / Xe-HPG: 16 GB or 8 GB).
# Dot-source in PowerShell before running `victory`:
#
#   . .\scripts\arc-a770.ps1                 # A770 16 GB, conservative stability defaults
#   . .\scripts\arc-a770.ps1 -VramGB 8       # A770 8 GB
#   . .\scripts\arc-a770.ps1 -Fast           # async relay + wider card buffers

param(
    [ValidateSet(8, 16)]
    [int]$VramGB = 16,
    [string]$Adapter = "A770",
    [switch]$Fast
)

$fetch = Join-Path $PSScriptRoot "fetch-agility.ps1"
$root = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path (Join-Path $root "target\diag\D3D12\D3D12Core.dll"))) {
    & powershell -ExecutionPolicy Bypass -File $fetch
}

$env:VICTORY_ADAPTER = $Adapter

# Measured on A770 16 GB (driver 32.0.101.9033):
# - wave lanes 8..32 (compute wave ops pinned to WaveSize(32) automatically)
# - doubles = false (wave f64 sums lowered to integer IEEE-754 automatically)
# - view heap capped at 1,000,000 descriptors (clamped to 999,999 automatically)
$env:VICTORY_INDIRECT_CLAMP = "0"
$env:VICTORY_SKIP_MOVIES    = "1"

if ($VramGB -eq 8) {
    $env:VICTORY_TEXTURE_BUDGET_MB   = "4096"
    $env:VICTORY_TEXTURE_ROOM_PERCENT = "80"
    $env:VICTORY_CARD_BUFFERS_MIN_MB = if ($Fast) { "4" } else { "16" }
    $env:VICTORY_CARD_BUFFERS_MB     = "256"
} else {
    $env:VICTORY_TEXTURE_BUDGET_MB   = "8192"
    $env:VICTORY_TEXTURE_ROOM_PERCENT = "85"
    $env:VICTORY_CARD_BUFFERS_MIN_MB = if ($Fast) { "1" } else { "16" }
    $env:VICTORY_CARD_BUFFERS_MB     = if ($Fast) { "1024" } else { "512" }
}

if ($Fast) {
    $env:VICTORY_RELAY      = "2"
    $env:VICTORY_WIDE_BACK  = "1"
    Remove-Item Env:VICTORY_LATE_WAIT_LISTS -ErrorAction SilentlyContinue
} else {
    $env:VICTORY_RELAY           = "0"
    $env:VICTORY_WIDE_BACK       = "0"
    $env:VICTORY_LATE_WAIT_LISTS = "128"
}

Write-Host "Intel Arc A770 (${VramGB} GB) profile active (Fast=$($Fast.IsPresent)):"
Write-Host "  VICTORY_ADAPTER              = $env:VICTORY_ADAPTER"
Write-Host "  VICTORY_TEXTURE_BUDGET_MB    = $env:VICTORY_TEXTURE_BUDGET_MB"
Write-Host "  VICTORY_CARD_BUFFERS_MIN_MB  = $env:VICTORY_CARD_BUFFERS_MIN_MB"
Write-Host "  VICTORY_CARD_BUFFERS_MB      = $env:VICTORY_CARD_BUFFERS_MB"
Write-Host "  VICTORY_RELAY / WIDE_BACK    = $env:VICTORY_RELAY / $env:VICTORY_WIDE_BACK"
Write-Host "  VICTORY_INDIRECT_CLAMP       = $env:VICTORY_INDIRECT_CLAMP"
