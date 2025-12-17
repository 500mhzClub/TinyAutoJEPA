#!/usr/bin/env bash
set -euo pipefail

# Do NOT force HSA_OVERRIDE_GFX_VERSION here.
# If you need it (e.g., gfx1200 workaround), export it before running this script.
if [[ -n "${HSA_OVERRIDE_GFX_VERSION:-}" ]]; then
  echo "Using HSA_OVERRIDE_GFX_VERSION=${HSA_OVERRIDE_GFX_VERSION}"
else
  echo "HSA_OVERRIDE_GFX_VERSION is unset (recommended for gfx1030 / 6950 XT)"
fi

export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
export TORCHINDUCTOR_DISABLE="${TORCHINDUCTOR_DISABLE:-1}"
unset TORCH_LOGS || true

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
