#!/usr/bin/env bash
# Activate the uv venv and wire up the system-installed Isaac Sim binary.
#
# Usage (must be sourced, not executed; works from bash and zsh):
#     source scripts/env.sh
#
# Isaac Sim location is resolved in this order (first hit wins):
#     1. $ISAAC_SIM_PATH, if already exported:
#            ISAAC_SIM_PATH=/path/to/isaac-sim-5.1 source scripts/env.sh
#     2. the IsaacLab/_isaac_sim symlink (README step 2 creates it)
#     3. $HOME/isaac-sim-5.1
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

# Locate this file and detect "sourced vs executed" in both bash and zsh.
# zsh has no BASH_SOURCE; when sourcing it sets $0 to the file (FUNCTION_ARGZERO,
# on by default — Isaac Sim's own setup_conda_env.sh relies on the same thing)
# and ZSH_EVAL_CONTEXT contains ":file".
if [[ -n "${ZSH_VERSION:-}" ]]; then
  _env_sh_file="$0"
  [[ "${ZSH_EVAL_CONTEXT:-}" == *:file* ]] || _env_sh_executed=1
else
  _env_sh_file="${BASH_SOURCE[0]:-$0}"
  [[ "${BASH_SOURCE[0]:-}" != "$0" ]] || _env_sh_executed=1
fi
if [[ -n "${_env_sh_executed:-}" ]]; then
  echo "[env.sh] This file must be SOURCED, not executed:  source scripts/env.sh" >&2
  unset _env_sh_file _env_sh_executed
  exit 1
fi
unset _env_sh_executed

if [[ -n "${CONDA_DEFAULT_ENV:-}" && "${CONDA_DEFAULT_ENV}" != "base" ]]; then
  echo "[env.sh] A conda env ('${CONDA_DEFAULT_ENV}') is active. Run 'conda deactivate' first — conda activation scripts for isaaclab/isaacsim leak ISAAC_PATH/CARB_APP_PATH/PYTHONPATH that will shadow the binary pointed to by ISAAC_SIM_PATH." >&2
  return 1
fi

_env_sh_dir="$(cd -- "$(dirname -- "${_env_sh_file}")" && pwd -P)"
_proj_root="$(cd -- "${_env_sh_dir}/.." && pwd -P)"
unset _env_sh_file

if [[ -z "${ISAAC_SIM_PATH:-}" ]]; then
  for _cand in "${_proj_root}/IsaacLab/_isaac_sim" "${HOME}/isaac-sim-5.1"; do
    if [[ -f "${_cand}/setup_conda_env.sh" ]]; then
      ISAAC_SIM_PATH="$(cd -- "${_cand}" && pwd -P)"   # resolve the symlink
      break
    fi
  done
  unset _cand
fi

if [[ -z "${ISAAC_SIM_PATH:-}" ]]; then
  echo "[env.sh] Could not locate an Isaac Sim install. Checked \$ISAAC_SIM_PATH, ${_proj_root}/IsaacLab/_isaac_sim and ${HOME}/isaac-sim-5.1. Either 'ln -s /path/to/isaac-sim-5.1 IsaacLab/_isaac_sim' or 'export ISAAC_SIM_PATH=/path/to/isaac-sim-5.1'." >&2
  unset _env_sh_dir _proj_root
  return 1
fi

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
