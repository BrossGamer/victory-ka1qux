@echo off
setlocal
cd /d "%~dp0.."

if "%~1"=="" (
    echo Usage: scripts\play-arc.cmd "<path-to-title-exe>"
    exit /b 1
)

if not exist "target\diag\D3D12\D3D12Core.dll" (
    powershell -NoProfile -ExecutionPolicy Bypass -File "scripts\fetch-agility.ps1"
)

set "VICTORY_TEXTURE_BUDGET_MB="
set "VICTORY_INDIRECT_CLAMP=compute"
set "VICTORY_SKIP_MOVIES=1"
set "VICTORY_RELAY=0"
set "VICTORY_WIDE_BACK=0"
set "VICTORY_LATE_WAIT_LISTS=128"
set "VICTORY_RAYS=0"

echo Running Victory on Intel Arc (logging to %TEMP%\victory-arc-last.log)...
".\target\diag\victory.exe" run --execute "%~1" > "%TEMP%\victory-arc-last.log" 2>&1
type "%TEMP%\victory-arc-last.log" | powershell -NoProfile -Command "$input | Select-Object -Last 25"
