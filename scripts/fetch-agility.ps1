# Downloads the DirectX 12 Agility SDK runtime (Microsoft's redistributable
# D3D12Core.dll and d3d12SDKLayers.dll) from the official NuGet package and
# places it in a `D3D12` folder beside every executable Cargo builds: the
# programs and the test executables (target\<profile>\ and its deps\).
#
# The DLLs are never committed (docs/asset-policy.md): they live under
# target\, which Git ignores. Run again after `cargo clean`.
#
# Why: on Windows 10 the system's D3D12 runtime stops at shader model 6.5 and
# has no mesh shaders, which the title's world needs. Intel Arc (and every
# other current card) gets SM 6.6+ and mesh shaders through this runtime.
#
# The version must match `D3D12SDKVersion` in the executables (the minor
# number: 1.616.1 -> 616), set by VICTORY_AGILITY_VERSION at build time
# (crates/victory-gpu/build.rs, default 616).
#
#   powershell -ExecutionPolicy Bypass -File scripts\fetch-agility.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\fetch-agility.ps1 -Version 1.616.1

param(
    [string]$Version = "1.616.1",
    [string[]]$Profiles = @("debug", "release", "diag")
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$target = Join-Path $root "target"
$store = Join-Path $target "agility\$Version"
$bin = Join-Path $store "build\native\bin\x64"

if (-not (Test-Path (Join-Path $bin "D3D12Core.dll"))) {
    New-Item -ItemType Directory -Force -Path $store | Out-Null
    $package = Join-Path $target "agility\Microsoft.Direct3D.D3D12.$Version.nupkg.zip"
    $url = "https://www.nuget.org/api/v2/package/Microsoft.Direct3D.D3D12/$Version"
    Write-Host "downloading $url"
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    Invoke-WebRequest -Uri $url -OutFile $package -UseBasicParsing
    Expand-Archive -Path $package -DestinationPath $store -Force
    Remove-Item $package
}

$minor = $Version.Split(".")[1]
foreach ($profile in $Profiles) {
    foreach ($dir in @((Join-Path $target $profile), (Join-Path $target "$profile\deps"))) {
        $dest = Join-Path $dir "D3D12"
        New-Item -ItemType Directory -Force -Path $dest | Out-Null
        foreach ($dll in @("D3D12Core.dll", "d3d12SDKLayers.dll")) {
            $src = Join-Path $bin $dll
            if (Test-Path $src) { Copy-Item $src $dest -Force }
        }
        Write-Host "Agility $Version -> $dest"
    }
}
Set-Content -Path (Join-Path $target "agility-version.txt") -Value $Version
Write-Host "done. Build with D3D12SDKVersion = $minor (VICTORY_AGILITY_VERSION=$minor, the default is 616)."
