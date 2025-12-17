#!/usr/bin/env bash
set -euo pipefail

# Recommended environment for your current setup (adjust if you want)
export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-12.0.0}"

# Keep compile/inductor out of the picture unless explicitly testing it
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
export TORCHINDUCTOR_DISABLE="${TORCHINDUCTOR_DISABLE:-1}"
unset TORCH_LOGS || true

# MIOpen settings (keep what you already use; do NOT force re-find here)
export MIOPEN_USER_DB_PATH="${MIOPEN_USER_DB_PATH:-$HOME/.cache/miopen/userdb}"
export MIOPEN_FIND_MODE="${MIOPEN_FIND_MODE:-3}"
unset MIOPEN_FIND_ENFORCE || true

SCRIPT="${1:-train_encoder.py}"
STEPS="${2:-10}"
OUT="${3:-profile.self_cuda.txt}"
LOG="${4:-diag.required.log}"

echo "Running required diagnostics:"
echo "  target script : ${SCRIPT}"
echo "  steps         : ${STEPS}"
echo "  profiler out  : ${OUT}"
echo "  log           : ${LOG}"

stdbuf -oL -eL python3 tajeppa_required_diagnostics.py \
  --script "${SCRIPT}" \
  --steps "${STEPS}" \
  --row-limit 30 \
  --print-top 15 \
  --out "${OUT}" \
  2>&1 | tee "${LOG}"
