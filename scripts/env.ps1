# Activate the uv venv and wire up the system-installed Isaac Sim binary.
#
# Usage (must be DOT-SOURCED, not executed):
#     . .\scripts\env.ps1
#
# Override the Isaac Sim location with:
#     $env:ISAAC_SIM_PATH = "C:\path\to\isaac-sim-standalone-5.1.0"
#     . .\scripts\env.ps1
#
# What this does:
#   1. Refuses to run if a conda env is active (its activation wires
#      CARB_APP_PATH / ISAAC_PATH / PYTHONPATH pointing at a *different* Isaac
#      Sim install, which silently shadows this one).
#   2. Activates .venv (uv-managed Python 3.11 + torch 2.7 + IsaacLab editables).
#   3. Exports ISAAC_PATH / EXP_PATH / CARB_APP_PATH (required by IsaacLab's
#      AppLauncher and Kit) and prepends Isaac Sim's `site` directory to
#      PYTHONPATH so its sitecustomize.py runs at Python startup. That
#      sitecustomize wires up sys.path for kit / exts / extscache and registers
#      DLL search paths via os.add_dll_directory(...).
#   4. Exports RESOURCE_NAME=IsaacSim (branding expected by Kit).
#   5. Auto-accepts the Omniverse EULA (OMNI_KIT_ACCEPT_EULA=YES).
#
# Linux equivalent: scripts/env.sh.

if ($MyInvocation.InvocationName -ne '.') {
    Write-Error "[env.ps1] This file must be DOT-SOURCED, not executed:  . .\scripts\env.ps1"
    return
}

if ($env:CONDA_DEFAULT_ENV -and $env:CONDA_DEFAULT_ENV -ne 'base') {
    Write-Error "[env.ps1] A conda env ('$($env:CONDA_DEFAULT_ENV)') is active. Run 'conda deactivate' first - conda activation scripts for isaaclab/isaacsim leak ISAAC_PATH/CARB_APP_PATH/PYTHONPATH that will shadow the binary pointed to by ISAAC_SIM_PATH."
    return
}

$_envPs1Dir = Split-Path -Parent $PSCommandPath
$_projRoot  = Resolve-Path (Join-Path $_envPs1Dir '..') | Select-Object -ExpandProperty Path

if (-not $env:ISAAC_SIM_PATH) {
    $env:ISAAC_SIM_PATH = 'C:\isaac-sim-standalone-5.1.0'
}

if (-not (Test-Path (Join-Path $env:ISAAC_SIM_PATH 'setup_python_env.bat'))) {
    Write-Error "[env.ps1] ISAAC_SIM_PATH='$($env:ISAAC_SIM_PATH)' does not look like an Isaac Sim install (missing setup_python_env.bat)"
    Remove-Variable _envPs1Dir, _projRoot
    return
}

$_venvActivate = Join-Path $_projRoot '.venv\Scripts\Activate.ps1'
if (-not (Test-Path $_venvActivate)) {
    Write-Error "[env.ps1] No venv at $_projRoot\.venv. Run 'uv sync' first."
    Remove-Variable _envPs1Dir, _projRoot, _venvActivate
    return
}

. $_venvActivate

$env:ISAAC_PATH    = $env:ISAAC_SIM_PATH
$env:EXP_PATH      = Join-Path $env:ISAAC_SIM_PATH 'apps'
$env:CARB_APP_PATH = Join-Path $env:ISAAC_SIM_PATH 'kit'
$env:RESOURCE_NAME = 'IsaacSim'
if (-not $env:OMNI_KIT_ACCEPT_EULA) { $env:OMNI_KIT_ACCEPT_EULA = 'YES' }

# Prepend Isaac Sim's `site` to PYTHONPATH so its sitecustomize.py runs at
# Python startup (handles all the omni.* / extension wiring + DLL paths).
$_isaacSite = Join-Path $env:ISAAC_SIM_PATH 'site'
if ($env:PYTHONPATH) {
    if (-not ($env:PYTHONPATH -split ';' | Where-Object { $_ -eq $_isaacSite })) {
        $env:PYTHONPATH = "$_isaacSite;$($env:PYTHONPATH)"
    }
} else {
    $env:PYTHONPATH = $_isaacSite
}

$_isaacVersion = if (Test-Path (Join-Path $env:ISAAC_SIM_PATH 'VERSION')) {
    (Get-Content (Join-Path $env:ISAAC_SIM_PATH 'VERSION') -First 1)
} else { 'unknown' }
$_pyExe = (Get-Command python -ErrorAction SilentlyContinue).Source
$_pyVer = if ($_pyExe) { & python --version 2>&1 } else { '<not found>' }

Write-Host "[env.ps1] venv:         $($env:VIRTUAL_ENV)"
Write-Host "[env.ps1] ISAAC_SIM:    $($env:ISAAC_SIM_PATH)  ($_isaacVersion)"
Write-Host "[env.ps1] python:       $_pyExe  ($_pyVer)"
Write-Host "[env.ps1] EXP_PATH:     $($env:EXP_PATH)"
Write-Host "[env.ps1] ISAAC_PATH:   $($env:ISAAC_PATH)"

Remove-Variable _envPs1Dir, _projRoot, _venvActivate, _isaacSite, _isaacVersion, _pyExe, _pyVer
