#!/usr/bin/env bash
# Activate the uv venv and wire up the system-installed Isaac Sim binary.
#
# Usage (must be sourced, not executed):
#     source scripts/env.sh
#
# Override the Isaac Sim location with:
#     ISAAC_SIM_PATH=/path/to/isaac-sim-5.1 source scripts/env.sh
#
# What this does:
#   1. Refuses to run if a conda env is active (its activation script exports
#      CARB_APP_PATH / ISAAC_PATH / PYTHONPATH / LD_LIBRARY_PATH pointing at a
#      *different* Isaac Sim install, which silently shadows this one).
#   2. Activates .venv (uv-managed Python 3.11 + torch 2.7 + IsaacLab editables).
#   3. Sources Isaac Sim's setup_conda_env.sh, which:
#        - exports ISAAC_PATH, EXP_PATH, CARB_APP_PATH (required by IsaacLab's
#          AppLauncher and Kit)
#        - sources setup_python_env.sh to prepend PYTHONPATH / LD_LIBRARY_PATH
#          for kit, extensions, extscache (so `import omni.physics`, etc. work)
#        - strips Isaac Sim's bundled kit/python/lib/python3.11 from PYTHONPATH
#          so the venv's torch/numpy/... are used instead of Isaac Sim's bundles
#   4. Exports RESOURCE_NAME=IsaacSim (branding expected by Kit).
#   5. Preloads torch's bundled libgomp to avoid libgomp symbol conflicts with
#      Isaac Sim's (same trick the isaaclab conda env activation uses).
#   6. Auto-accepts the Omniverse EULA (OMNI_KIT_ACCEPT_EULA=YES).

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "[env.sh] This file must be SOURCED, not executed:  source scripts/env.sh" >&2
  exit 1
fi

if [[ -n "${CONDA_DEFAULT_ENV:-}" && "${CONDA_DEFAULT_ENV}" != "base" ]]; then
  echo "[env.sh] A conda env ('${CONDA_DEFAULT_ENV}') is active. Run 'conda deactivate' first — conda activation scripts for isaaclab/isaacsim leak ISAAC_PATH/CARB_APP_PATH/PYTHONPATH that will shadow the binary pointed to by ISAAC_SIM_PATH." >&2
  return 1
fi

_env_sh_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
_proj_root="$(cd -- "${_env_sh_dir}/.." && pwd -P)"

ISAAC_SIM_PATH="${ISAAC_SIM_PATH:-/home/liuj/isaac-sim-5.1}"

if [[ ! -f "${ISAAC_SIM_PATH}/setup_conda_env.sh" ]]; then
  echo "[env.sh] ISAAC_SIM_PATH='${ISAAC_SIM_PATH}' does not look like an Isaac Sim install (missing setup_conda_env.sh)" >&2
  unset _env_sh_dir _proj_root
  return 1
fi

if [[ ! -f "${_proj_root}/.venv/bin/activate" ]]; then
  echo "[env.sh] No venv at ${_proj_root}/.venv. Run 'uv sync' first." >&2
  unset _env_sh_dir _proj_root
  return 1
fi

# shellcheck disable=SC1091
source "${_proj_root}/.venv/bin/activate"

# setup_conda_env.sh cd's into ISAAC_SIM_PATH and back. It sources
# setup_python_env.sh internally and then prunes kit/python/lib/python3.11 from
# PYTHONPATH so the venv's packages win over Isaac Sim's bundled ones.
# shellcheck disable=SC1091
source "${ISAAC_SIM_PATH}/setup_conda_env.sh"

export RESOURCE_NAME="IsaacSim"
export ISAAC_SIM_PATH
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"

# Preload torch's bundled libgomp to avoid symbol conflicts between Isaac Sim's
# libgomp (loaded via LD_LIBRARY_PATH) and the one inside torch wheels.
_gomp="$(python - <<'PY' 2>/dev/null || true
import pathlib
try:
    import torch
    p = pathlib.Path(torch.__file__).parent / "lib" / "libgomp.so.1"
    print(p if p.exists() else "", end="")
except Exception:
    pass
PY
)"
if [[ -n "${_gomp}" && -r "${_gomp}" ]]; then
  case ":${LD_PRELOAD:-}:" in
    *":${_gomp}:"*) : ;;
    *) export LD_PRELOAD="${_gomp}${LD_PRELOAD:+:${LD_PRELOAD}}" ;;
  esac
fi
unset _gomp

echo "[env.sh] venv:         ${VIRTUAL_ENV}"
echo "[env.sh] ISAAC_SIM:    ${ISAAC_SIM_PATH}  ($(cat "${ISAAC_SIM_PATH}/VERSION" 2>/dev/null || echo unknown))"
echo "[env.sh] python:       $(command -v python)  ($(python --version 2>&1))"
echo "[env.sh] EXP_PATH:     ${EXP_PATH}"
echo "[env.sh] ISAAC_PATH:   ${ISAAC_PATH}"

unset _env_sh_dir _proj_root
