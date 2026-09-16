#!/usr/bin/env bash
set -Eeuo pipefail

# Generate R1 Pro gearbox demonstrations until TARGET_SUCCESS successful
# episodes have been written by GalaxeaLabExternalEnv.
#
# By default the environment only writes an HDF5 file after score==5. With
# --keep_failed [MIN_SCORE] it also writes failed episodes whose score is at
# least MIN_SCORE (a bare --keep_failed means MIN_SCORE=0). This script always
# uses successful files for the stopping condition and parses the simulator
# log for per-episode scores, successful/failed counts, and a live ETA.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"

TARGET_SUCCESS="${TARGET_SUCCESS:-150}"
STOP_GRACE_SECONDS="${STOP_GRACE_SECONDS:-120}"
DATA_DIR="${ROCO_R1PRO_DATA_DIR:-/home/pine/roco_sim/data/r1pro}"
CONDA_SH="${CONDA_SH:-/home/pine/miniconda3/etc/profile.d/conda.sh}"
TASK_NAME="Template-Galaxea-Lab-External-Direct-v0"
KEEP_FAILED=""

while (($# > 0)); do
    case "$1" in
        --keep_failed)
            if (($# > 1)) && [[ "$2" != -* ]]; then
                KEEP_FAILED="$2"
                shift 2
            else
                KEEP_FAILED=0
                shift
            fi
            ;;
        --keep_failed=*)
            KEEP_FAILED="${1#*=}"
            shift
            ;;
        -h|--help)
            printf 'Usage: %s [--keep_failed [MIN_SCORE]]\n' "${BASH_SOURCE[0]}"
            printf '  --keep_failed [MIN_SCORE]  save failed episodes with score >= MIN_SCORE; bare option means 0\n'
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            printf 'Usage: %s [--keep_failed [MIN_SCORE]]\n' "${BASH_SOURCE[0]}" >&2
            exit 2
            ;;
    esac
done

if [[ -n "${KEEP_FAILED}" ]] && ! [[ "${KEEP_FAILED}" =~ ^[0-5]$ ]]; then
    echo "--keep_failed MIN_SCORE must be an integer in the range 0..5" >&2
    exit 2
fi

if ! [[ "${TARGET_SUCCESS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "TARGET_SUCCESS must be a positive integer" >&2
    exit 2
fi
if ! [[ "${STOP_GRACE_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "STOP_GRACE_SECONDS must be a positive integer" >&2
    exit 2
fi
if [[ ! -r "${CONDA_SH}" ]]; then
    echo "Conda activation script not found: ${CONDA_SH}" >&2
    exit 2
fi

mkdir -p "${DATA_DIR}"
LOG_FILE="${DATA_DIR}/r1pro_rulegen_$(date +%Y%m%d_%H%M%S).log"

# Count valid completed HDF5 files, ignoring a file that is still being
# written or an unrelated HDF5 file in the directory.
count_success_files() {
    python - "${DATA_DIR}" <<'PY'
import pathlib
import sys

import h5py

root = pathlib.Path(sys.argv[1])
count = 0
for path in root.glob("*.hdf5"):
    try:
        with h5py.File(path, "r") as handle:
            if bool(handle.attrs.get("success", False)):
                count += 1
    except (OSError, KeyError, ValueError):
        # Partially written files are not counted until the writer closes them.
        pass
print(count)
PY
}

log_count() {
    local pattern="$1"
    grep -cF -- "${pattern}" "${LOG_FILE}" 2>/dev/null || true
}

count_failed_logs() {
    local skipped saved
    skipped="$(log_count 'Skipping unsuccessful episode')"
    saved="$(log_count 'Writing failed episode')"
    printf '%d\n' "$(( ${skipped:-0} + ${saved:-0} ))"
}

format_eta() {
    local remaining="$1"
    local succeeded="$2"
    local elapsed="$3"
    if (( succeeded <= 0 || elapsed <= 0 )); then
        printf 'estimating'
        return
    fi
    awk -v remaining="${remaining}" -v succeeded="${succeeded}" -v elapsed="${elapsed}" \
        'BEGIN {
            eta = int(remaining * elapsed / succeeded + 0.5)
            if (eta < 1) eta = 1
            days = int(eta / 86400); eta %= 86400
            hours = int(eta / 3600); eta %= 3600
            minutes = int(eta / 60); seconds = eta % 60
            if (days > 0) printf "~%dd%02dh%02dm%02ds", days, hours, minutes, seconds
            else printf "~%02dh%02dm%02ds", hours, minutes, seconds
        }'
}

emit_progress() {
    local success_files run_success run_failed run_total elapsed remaining rate eta finish_at
    success_files="$(count_success_files)"
    run_success=$((success_files - initial_success))
    (( run_success < 0 )) && run_success=0
    run_failed="$(count_failed_logs)"
    run_total=$((run_success + run_failed))
    elapsed=$(( $(date +%s) - start_epoch ))
    remaining=$((TARGET_SUCCESS - success_files))
    (( remaining < 0 )) && remaining=0
    if (( run_total > 0 )); then
        rate="$(awk -v s="${run_success}" -v t="${run_total}" 'BEGIN { printf "%.2f", 100*s/t }')"
    else
        rate="0.00"
    fi
    eta="$(format_eta "${remaining}" "${run_success}" "${elapsed}")"
    finish_at="$(estimate_finish "${remaining}" "${run_success}" "${elapsed}")"
    LAST_SUCCESS_FILES="${success_files}"
    log_line "progress=${success_files}/${TARGET_SUCCESS}; elapsed=${elapsed}s; run_total=${run_total}; run_success=${run_success}; run_failed=${run_failed}; success_rate=${rate}%; ETA=${eta}; expected_finish=${finish_at}"
}

estimate_finish() {
    local remaining="$1"
    local succeeded="$2"
    local elapsed="$3"
    if (( succeeded <= 0 || elapsed <= 0 )); then
        printf 'estimating'
        return
    fi
    local eta_seconds=$((remaining * elapsed / succeeded))
    (( eta_seconds < 1 )) && eta_seconds=1
    date -d "@$(($(date +%s) + eta_seconds))" '+%Y-%m-%d %H:%M:%S'
}

log_line() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "${LOG_FILE}"
}

declare -A SCORE_COUNTS=()
SCORE_COUNT_TOTAL=0
SCORE_SUM=0
SCORE_MIN=0
SCORE_MAX=0

record_episode_score() {
    local score="$1"
    local current_count="${SCORE_COUNTS[$score]:-0}"
    SCORE_COUNTS["$score"]=$((current_count + 1))
    SCORE_COUNT_TOTAL=$((SCORE_COUNT_TOTAL + 1))
    SCORE_SUM=$((SCORE_SUM + score))
    if (( SCORE_COUNT_TOTAL == 1 || score < SCORE_MIN )); then SCORE_MIN=$score; fi
    if (( SCORE_COUNT_TOTAL == 1 || score > SCORE_MAX )); then SCORE_MAX=$score; fi
}

format_score_summary() {
    if (( SCORE_COUNT_TOTAL == 0 )); then
        printf 'episodes=0; distribution=none'
        return
    fi
    local score count distribution='' mean
    for score in 0 1 2 3 4 5; do
        count="${SCORE_COUNTS[$score]:-0}"
        if [[ -n "${distribution}" ]]; then distribution+=','; fi
        distribution+="${score}:${count}"
    done
    mean="$(awk -v sum="${SCORE_SUM}" -v count="${SCORE_COUNT_TOTAL}" 'BEGIN { printf "%.2f", sum/count }')"
    printf 'episodes=%d; distribution=%s; mean=%s; min=%d; max=%d' \
        "${SCORE_COUNT_TOTAL}" "${distribution}" "${mean}" "${SCORE_MIN}" "${SCORE_MAX}"
}

source "${CONDA_SH}"
conda activate roco

export ROCO_DATA_DIR="${DATA_DIR}"
export ROCO_ROBOT_BUNDLE="r1_pro"
export PYTHONPATH="${REPO_ROOT}/source/Galaxea_Lab_External${PYTHONPATH:+:${PYTHONPATH}}"

initial_success="$(count_success_files)"
if (( initial_success >= TARGET_SUCCESS )); then
    log_line "Target already reached: ${initial_success}/${TARGET_SUCCESS} successful episodes."
    exit 0
fi

start_epoch="$(date +%s)"
log_line "Starting R1 Pro rule generation"
log_line "repo=${REPO_ROOT}"
log_line "data=${DATA_DIR}"
log_line "log=${LOG_FILE}"
keep_failed_log="${KEEP_FAILED:-disabled}"
log_line "embodiment=r1_pro; target=${TARGET_SUCCESS}; existing_success=${initial_success}; keep_failed=${keep_failed_log}; task=${TASK_NAME}"

agent_args=(
    --task="${TASK_NAME}"
    --enable_cameras
    --headless
    --num_envs 1
)
if [[ -n "${KEEP_FAILED}" ]]; then
    agent_args+=(--keep_failed "${KEEP_FAILED}")
fi
log_line "command: ROCO_ROBOT_BUNDLE=r1_pro python -u scripts/rule_based_agent.py ${agent_args[*]}"

(
    cd "${REPO_ROOT}"
    exec python -u scripts/rule_based_agent.py "${agent_args[@]}"
) >>"${LOG_FILE}" 2>&1 &
SIM_PID=$!

target_reached=0
stopping=0
interrupted=0
LAST_SUCCESS_FILES="${initial_success}"

stop_simulator() {
    if (( stopping == 1 )); then
        return
    fi
    stopping=1
    if kill -0 "${SIM_PID}" 2>/dev/null; then
        log_line "Stopping simulator PID ${SIM_PID} (SIGINT; allowing final HDF5 flush)"
        kill -INT "${SIM_PID}" 2>/dev/null || true
        local waited=0
        while kill -0 "${SIM_PID}" 2>/dev/null && (( waited < STOP_GRACE_SECONDS )); do
            sleep 1
            waited=$((waited + 1))
        done
        if kill -0 "${SIM_PID}" 2>/dev/null; then
            log_line "Simulator did not exit after ${STOP_GRACE_SECONDS}s; sending SIGTERM"
            kill -TERM "${SIM_PID}" 2>/dev/null || true
        fi
    fi
}

on_interrupt() {
    log_line "Interrupted by user; stopping cleanly"
    interrupted=1
    stop_simulator
}
trap on_interrupt INT TERM

# The simulator output is redirected to LOG_FILE. GNU tail follows that file
# and wakes this shell only when a completed terminal marker is appended by the
# environment, so progress is emitted once per episode rather than on a wall
# clock interval. --pid also lets tail exit when the simulator crashes/stops.
emit_progress
while IFS= read -r sim_line; do
    case "${sim_line}" in
        *"Writing successful episode"*|*"Writing failed episode"*|*"Skipping unsuccessful episode"*)
            if [[ "${sim_line}" == *"Writing successful episode"* ]]; then
                episode_outcome="success"
            elif [[ "${sim_line}" == *"Writing failed episode"* ]]; then
                episode_outcome="failed_saved"
            else
                episode_outcome="failed_discarded"
            fi
            if [[ "${sim_line}" =~ score=([0-9]+) ]]; then
                episode_score="${BASH_REMATCH[1]}"
                record_episode_score "${episode_score}"
                log_line "episode_complete; outcome=${episode_outcome}; score=${episode_score}"
            fi
            emit_progress
            if (( LAST_SUCCESS_FILES >= TARGET_SUCCESS )); then
                target_reached=1
                stop_simulator
                break
            fi
            ;;
    esac
done < <(tail --pid="${SIM_PID}" -n +1 -F "${LOG_FILE}")

set +e
wait "${SIM_PID}"
sim_exit=$?
set -e

final_success="$(count_success_files)"
final_success_new=$((final_success - initial_success))
(( final_success_new < 0 )) && final_success_new=0
final_failed="$(count_failed_logs)"
final_total=$((final_success_new + final_failed))
if (( final_total > 0 )); then
    final_rate="$(awk -v s="${final_success_new}" -v t="${final_total}" 'BEGIN { printf "%.2f", 100*s/t }')"
else
    final_rate="0.00"
fi

if (( final_success >= TARGET_SUCCESS )); then
    target_reached=1
fi

log_line "FINAL: total_runs=${final_total}; successful=${final_success_new}; failed=${final_failed}; success_rate=${final_rate}%; total_success_files=${final_success}/${TARGET_SUCCESS}; simulator_exit=${sim_exit}"
log_line "SCORE_SUMMARY: $(format_score_summary)"

if (( target_reached == 1 )); then
    exit 0
fi

log_line "Generation stopped before reaching target. Check ${LOG_FILE}."
if (( sim_exit == 0 )); then
    exit 1
fi
exit "${sim_exit}"
