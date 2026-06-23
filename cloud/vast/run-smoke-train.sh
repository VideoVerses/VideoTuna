#!/usr/bin/env bash
# Short GPU smoke training run (5–50 steps) to validate deps before a long job.
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
REPO="${WORKSPACE}/VideoTuna"
cd "${REPO}"

export PATH="${HOME}/.local/bin:${PATH}"

if [[ -f "${REPO}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${REPO}/.env"
  set +a
fi

mkdir -p "${WORKSPACE}/results"
LOG_OUT="${WORKSPACE}/results/smoke-train.log"
LOG_ERR="${WORKSPACE}/results/smoke-train.err"

log() { echo "[$(date -Iseconds)] [run-smoke-train] $*" | tee -a "${LOG_OUT}"; }

log "GPU info:"
nvidia-smi 2>&1 | tee -a "${LOG_OUT}" || log "nvidia-smi not available"

POETRY_VENV_BIN="$(poetry env info -p 2>/dev/null)/bin" || true
if [[ -n "${POETRY_VENV_BIN}" && -d "${POETRY_VENV_BIN}" ]]; then
  export PATH="${POETRY_VENV_BIN}:${PATH}"
fi

TRAIN_PROFILE="${TRAIN_PROFILE:-flux-lora}"
CONFIG_PATH="${CONFIG_PATH:-}"
DATA_CONFIG_PATH="${DATA_CONFIG_PATH:-}"

log "Smoke TRAIN_PROFILE=${TRAIN_PROFILE}"

run_cmd() {
  log "Executing: $*"
  "$@" >>"${LOG_OUT}" 2>>"${LOG_ERR}"
}

case "${TRAIN_PROFILE}" in
  flux-lora)
  CONFIG_PATH="${CONFIG_PATH:-configs/domain/flux_t2i_cloud_smoke.json}"
  DATA_CONFIG_PATH="${DATA_CONFIG_PATH:-configs/domain/flux_t2i_data.json}"
  run_cmd poetry run train-flux-lora \
    --config_path "${CONFIG_PATH}" \
    --data_config_path "${DATA_CONFIG_PATH}"
  ;;
  wan-t2v-lora)
  CONFIG_PATH="${CONFIG_PATH:-configs/domain/wan_t2v_lora_cloud_smoke.yaml}"
  run_cmd poetry run train-wan2-1-t2v-lora --base "${CONFIG_PATH}"
  ;;
  *)
  echo "Unknown TRAIN_PROFILE=${TRAIN_PROFILE}" | tee -a "${LOG_ERR}"
  echo "Valid: flux-lora, wan-t2v-lora" | tee -a "${LOG_ERR}"
  exit 1
  ;;
esac

log "Smoke training finished."
