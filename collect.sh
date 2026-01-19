#!/usr/bin/env bash
set -euo pipefail

# ---------- Config ----------
WORKERS=8
STEPS=1000
MODEL="ppo_carracing_v3_perfected.zip"

# Target mix: 60/25/15 to total 2,000,000 transitions
TARGET_EXPERT=1200000
TARGET_RECOVER=500000
TARGET_RANDOM=300000

# Total episodes across all workers per batch
BATCH_EXPERT=160
BATCH_RECOVER=120
BATCH_RANDOM=100

# Recover-mode "safe corner bias" (lower risk of long spinouts)
SAB_PROB=0.004
SAB_LEN_MIN=2
SAB_LEN_MAX=4
SAB_COOLDOWN=220
SAB_GAS_MIN=0.20
SAB_GAS_MAX=0.45

# ---------- Helpers ----------
sum_counters () {
  # usage: sum_counters data_expert
  local d="$1"
  if [ ! -d "$d" ]; then echo 0; return; fi
  python - "$d" <<'PY'
import os, glob, sys
d=sys.argv[1]
total=0
for f in glob.glob(os.path.join(d, ".transitions_w*.txt")):
    try:
        with open(f,"r",encoding="utf-8") as fh:
            total += int((fh.read().strip() or "0"))
    except Exception:
        pass
print(total)
PY
}

count_all () {
  echo "data_expert  $(sum_counters data_expert)"
  echo "data_recover $(sum_counters data_recover)"
  echo "data_random  $(sum_counters data_random)"
}

mkdir -p data_expert data_recover data_random

if [ ! -f "$MODEL" ]; then
  echo "[ERROR] Missing model: $MODEL"
  exit 1
fi

echo "[INFO] Current transition counts (from counters):"
count_all

run_until_target () {
  # args: mode outdir target batch extra_args...
  local mode="$1"
  local outdir="$2"
  local target="$3"
  local batch="$4"
  shift 4
  local extra_args=("$@")

  while true; do
    local cur
    cur="$(sum_counters "$outdir")"
    if [ "$cur" -ge "$target" ]; then
      echo "[DONE] $mode reached target: $cur / $target transitions"
      break
    fi

    local rem=$(( target - cur ))
    echo "[INFO] $mode: current=$cur target=$target remaining=$rem"
    echo "[RUN] python collect_data.py --mode $mode --out_dir $outdir --workers $WORKERS --episodes_total $batch --steps $STEPS --no_compress ${extra_args[*]}"

    python collect_data.py \
      --mode "$mode" \
      --out_dir "$outdir" \
      --workers "$WORKERS" \
      --episodes_total "$batch" \
      --steps "$STEPS" \
      --no_compress \
      --save_metrics \
      "${extra_args[@]}"

    echo "[INFO] After batch counts:"
    count_all
  done
}

# ---------- EXPERT ----------
run_until_target \
  expert data_expert "$TARGET_EXPERT" "$BATCH_EXPERT" \
  --model "$MODEL"

# ---------- RECOVER ----------
run_until_target \
  recover data_recover "$TARGET_RECOVER" "$BATCH_RECOVER" \
  --model "$MODEL" \
  --sabotage_prob "$SAB_PROB" \
  --sabotage_len_min "$SAB_LEN_MIN" \
  --sabotage_len_max "$SAB_LEN_MAX" \
  --sabotage_cooldown "$SAB_COOLDOWN" \
  --sabotage_gas_min "$SAB_GAS_MIN" \
  --sabotage_gas_max "$SAB_GAS_MAX"

# ---------- RANDOM ----------
run_until_target \
  random data_random "$TARGET_RANDOM" "$BATCH_RANDOM"

echo "[FINAL] Transition counts:"
count_all
