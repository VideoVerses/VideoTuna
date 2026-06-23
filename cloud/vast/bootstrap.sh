#!/usr/bin/env bash
# VideoTuna first-boot / re-provision bootstrap for Vast.ai linux-desktop templates.
# Usable as PROVISIONING_SCRIPT or invoked from provisioning.yaml post_commands.
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
REPO="${WORKSPACE}/VideoTuna"
MARKER="${WORKSPACE}/.videotuna_provisioned"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROVISION_RETRY="${SCRIPT_DIR}/provision_retry.py"

log() { echo "[videotuna-bootstrap] $*"; }

enable_fast_hf_download() {
  if [[ "${VIDEOTUNA_FAST_HF_DOWNLOAD:-0}" == "1" ]]; then
    export HF_XET_HIGH_PERFORMANCE=1
    log "Fast HF downloads enabled (HF_XET_HIGH_PERFORMANCE=1 via VIDEOTUNA_FAST_HF_DOWNLOAD)"
  fi
}

ensure_provision_retry() {
  log "Ensuring bootstrap retry dependencies (tenacity, pyyaml)..."
  python3 "${PROVISION_RETRY}" install-bootstrap-deps
}

ensure_poetry() {
  export PATH="${HOME}/.local/bin:${PATH}"
  if command -v poetry >/dev/null 2>&1; then
    log "Poetry already installed: $(poetry --version)"
    return 0
  fi
  python3 "${PROVISION_RETRY}" install-poetry
  export PATH="${HOME}/.local/bin:${PATH}"
  poetry --version
}

setup_workspace_layout() {
  log "Creating workspace directories and symlinks..."
  mkdir -p \
    "${WORKSPACE}/data/t2i/domain" \
    "${WORKSPACE}/data/t2v/domain/videos" \
    "${WORKSPACE}/checkpoints/flux" \
    "${WORKSPACE}/checkpoints/wan" \
    "${WORKSPACE}/results" \
    "${WORKSPACE}/.cache/huggingface"

  mkdir -p "${REPO}/data"
  ln -sfn "${WORKSPACE}/data/t2i" "${REPO}/data/t2i"
  ln -sfn "${WORKSPACE}/data/t2v" "${REPO}/data/t2v"
  ln -sfn "${WORKSPACE}/checkpoints" "${REPO}/checkpoints"
  ln -sfn "${WORKSPACE}/results" "${REPO}/results"
}

write_env_file() {
  local env_file="${REPO}/.env"
  local example="${SCRIPT_DIR}/.env.cloud.example"
  if [[ -f "${env_file}" ]]; then
    log ".env already exists at ${env_file}; skipping template copy"
    return 0
  fi
  if [[ ! -f "${example}" ]]; then
    log "WARNING: ${example} not found; creating minimal .env"
    cat >"${env_file}" <<EOF
WORKSPACE=${WORKSPACE}
VIDEOTUNA_COMPUTE_BACKEND=cuda
VIDEOTUNA_ATTN_BACKEND=auto
CUDA_VISIBLE_DEVICES=0
HF_HOME=${WORKSPACE}/.cache/huggingface
TRAIN_PROFILE=flux-lora
EOF
  else
    cp "${example}" "${env_file}"
  fi
  chmod 600 "${env_file}"

  # Inject host secrets from template env when set.
  if [[ -n "${HF_TOKEN:-}" ]]; then
    if grep -q '^HF_TOKEN=' "${env_file}"; then
      sed -i "s|^HF_TOKEN=.*|HF_TOKEN=${HF_TOKEN}|" "${env_file}"
    else
      echo "HF_TOKEN=${HF_TOKEN}" >>"${env_file}"
    fi
  fi
  if [[ -n "${VIDEOTUNA_ATTN_BACKEND:-}" ]]; then
    sed -i "s|^VIDEOTUNA_ATTN_BACKEND=.*|VIDEOTUNA_ATTN_BACKEND=${VIDEOTUNA_ATTN_BACKEND}|" \
      "${env_file}" || echo "VIDEOTUNA_ATTN_BACKEND=${VIDEOTUNA_ATTN_BACKEND}" >>"${env_file}"
  fi
  if [[ "${VIDEOTUNA_FAST_HF_DOWNLOAD:-0}" == "1" ]]; then
    if grep -q '^HF_XET_HIGH_PERFORMANCE=' "${env_file}"; then
      sed -i "s|^HF_XET_HIGH_PERFORMANCE=.*|HF_XET_HIGH_PERFORMANCE=1|" "${env_file}"
    else
      echo "HF_XET_HIGH_PERFORMANCE=1" >>"${env_file}"
    fi
  fi
  log "Wrote ${env_file}"
}

install_videotuna() {
  if [[ ! -d "${REPO}" ]]; then
    log "ERROR: ${REPO} not found — git_repos phase must clone VideoTuna first"
    exit 1
  fi
  cd "${REPO}"
  export PATH="${HOME}/.local/bin:${PATH}"
  export HF_HOME="${HF_HOME:-${WORKSPACE}/.cache/huggingface}"

  log "Running poetry install -E cuda --with training..."
  python3 "${PROVISION_RETRY}" run -- poetry install -E cuda --with training --no-interaction

  poetry run python -c "import hf_xet" 2>/dev/null \
    || log "WARNING: hf-xet not importable; HF downloads use fallback path"

  log "Installing DeepSpeed (required for Wan / CogVideoX LoRA)..."
  python3 "${PROVISION_RETRY}" run -- poetry run install-deepspeed

  if [[ "${VIDEOTUNA_INSTALL_FLASH_ATTN:-0}" == "1" ]]; then
    log "Installing flash-attn (optional, datacenter GPUs)..."
    python3 "${PROVISION_RETRY}" run -- poetry run install-flash-attn \
      || log "WARNING: install-flash-attn failed; use VIDEOTUNA_ATTN_BACKEND=sdpa"
  fi
}

hf_login() {
  if [[ -z "${HF_TOKEN:-}" ]]; then
    log "HF_TOKEN not set; skipping huggingface-cli login and gated downloads"
    return 0
  fi
  export PATH="${HOME}/.local/bin:${PATH}"
  export HF_HOME="${HF_HOME:-${WORKSPACE}/.cache/huggingface}"
  cd "${REPO}"
  if command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli login --token "${HF_TOKEN}" --add-to-git-credential || true
  else
    poetry run huggingface-cli login --token "${HF_TOKEN}" --add-to-git-credential || true
  fi
}

download_weights_if_missing() {
  if [[ -z "${HF_TOKEN:-}" ]]; then
    return 0
  fi
  export HF_HOME="${HF_HOME:-${WORKSPACE}/.cache/huggingface}"

  log "Pre-downloading FLUX.1-dev (with retries)..."
  python3 "${PROVISION_RETRY}" hf-download \
    black-forest-labs/FLUX.1-dev \
    "${WORKSPACE}/checkpoints/flux/FLUX.1-dev" \
    --repo-root "${REPO}"

  log "Pre-downloading Wan2.1-T2V-14B (with retries)..."
  python3 "${PROVISION_RETRY}" hf-download \
    Wan-AI/Wan2.1-T2V-14B \
    "${WORKSPACE}/checkpoints/wan/Wan2.1-T2V-14B" \
    --repo-root "${REPO}"

  log "Pre-downloading Wan2.1-I2V-14B-480P (with retries)..."
  python3 "${PROVISION_RETRY}" hf-download \
    Wan-AI/Wan2.1-I2V-14B-480P \
    "${WORKSPACE}/checkpoints/wan/Wan2.1-I2V-14B-480P" \
    --repo-root "${REPO}"

  log "Pre-downloading Wan2.2-T2V-A14B-Diffusers into HF hub cache (with retries)..."
  python3 "${PROVISION_RETRY}" hf-download-cache \
    Wan-AI/Wan2.2-T2V-A14B-Diffusers \
    --repo-root "${REPO}"

  log "Pre-downloading Wan2.2-I2V-A14B-Diffusers into HF hub cache (with retries)..."
  python3 "${PROVISION_RETRY}" hf-download-cache \
    Wan-AI/Wan2.2-I2V-A14B-Diffusers \
    --repo-root "${REPO}"
}

run_smoke_validation() {
  cd "${REPO}"
  export PATH="${HOME}/.local/bin:${PATH}"
  # shellcheck disable=SC1091
  [[ -f .env ]] && set -a && source .env && set +a

  log "Running import smoke test..."
  poetry run test tests/test_import_smoke.py -q

  log "Describing compute environment..."
  poetry run python -c \
    "from videotuna.utils.device_utils import describe_compute_environment; print(describe_compute_environment())"
}

main() {
  log "Starting VideoTuna bootstrap (workspace=${WORKSPACE})"
  enable_fast_hf_download
  ensure_provision_retry
  ensure_poetry
  setup_workspace_layout
  write_env_file
  install_videotuna
  hf_login
  download_weights_if_missing
  run_smoke_validation
  touch "${MARKER}"
  log "Bootstrap complete. Marker: ${MARKER}"
}

main "$@"
