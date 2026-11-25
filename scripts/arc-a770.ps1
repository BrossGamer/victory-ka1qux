# Environment profile for Intel Arc A770 (Alchemist / Xe-HPG: 16 GB or 8 GB).
# Dot-source in PowerShell before running `victory`:
#
#   . .\scripts\arc-a770.ps1                 # A770 16 GB, stable defaults (write marks on, compute clamp, RAYS=0)
#   . .\scripts\arc-a770.ps1 -VramGB 8       # A770 8 GB
#   . .\scripts\arc-a770.ps1 -Rays           # enable title ray tracing (pre-warm XboxBvhBuilder.dll pipelines first)
#   . .\scripts\arc-a770.ps1 -Fast           # experimental async relay=2 (faster, but lightens skin bakes & risks pagefaults)

param(
    [ValidateSet(8, 16)]
    [int]$VramGB = 16,
    [string]$Adapter = "A770",
    [switch]$Fast,
    [switch]$Rays
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
# - barycentrics = false (SV_Barycentrics / attributeAtVertex lowered automatically)
# - typed 64-bit atomics = false (lowered to RWTexture*<uint2> load/store automatically)
# - view heap capped at 1,000,000 descriptors (clamped to 999,999 automatically)
# - VICTORY_INDIRECT_CLAMP=compute bounds DispatchIndirect records (preventing
#   GEN12_DX12COMPUTE_BBHANG / PAGEFAULT from dropped Wave64 compute passes)
#   without clamping or charging mesh/vertex draws.
Remove-Item Env:VICTORY_TEXTURE_BUDGET_MB -ErrorAction SilentlyContinue
$env:VICTORY_INDIRECT_CLAMP = "1"
$env:VICTORY_SKIP_MOVIES    = "1"
$env:VICTORY_RAYS           = if ($Rays) { "1" } else { "0" }

if ($VramGB -eq 8) {
    $env:VICTORY_TEXTURE_ROOM_PERCENT = "90"
    $env:VICTORY_CARD_BUFFERS_MIN_MB  = if ($Fast) { "4" } else { "16" }
    $env:VICTORY_CARD_BUFFERS_MB      = "256"
} else {
    $env:VICTORY_TEXTURE_ROOM_PERCENT = "96"
    $env:VICTORY_CARD_BUFFERS_MIN_MB  = if ($Fast) { "1" } else { "16" }
    $env:VICTORY_CARD_BUFFERS_MB      = if ($Fast) { "1024" } else { "512" }
}

if ($Fast) {
    $env:VICTORY_RELAY     = "2"
    $env:VICTORY_WIDE_BACK = "1"
    Remove-Item Env:VICTORY_LATE_WAIT_LISTS -ErrorAction SilentlyContinue
} else {
    $env:VICTORY_RELAY           = "0"
    $env:VICTORY_WIDE_BACK       = "0"
    $env:VICTORY_LATE_WAIT_LISTS = "128"
}

$mode = if ($Fast) { "Fast-Experimental" } else { "Stable" }
Write-Host "Intel Arc A770 (${VramGB} GB) profile active (Mode=$mode, Rays=$($Rays.IsPresent)):"
Write-Host "  VICTORY_ADAPTER              = $env:VICTORY_ADAPTER"
Write-Host "  VICTORY_TEXTURE_ROOM_PERCENT = $env:VICTORY_TEXTURE_ROOM_PERCENT% (live DXGI budget)"
Write-Host "  VICTORY_CARD_BUFFERS_MIN_MB  = $env:VICTORY_CARD_BUFFERS_MIN_MB"
Write-Host "  VICTORY_CARD_BUFFERS_MB      = $env:VICTORY_CARD_BUFFERS_MB"
Write-Host "  VICTORY_RELAY / WIDE_BACK    = $env:VICTORY_RELAY / $env:VICTORY_WIDE_BACK"
Write-Host "  VICTORY_RAYS                 = $env:VICTORY_RAYS"
Write-Host "  VICTORY_INDIRECT_CLAMP       = $env:VICTORY_INDIRECT_CLAMP"
