#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo "Usage: $0 {mna_plr|mna_accel|frontier_plr|frontier_lp_plr|frontier_lp_accel|frontier_lp_adaptive|mna_frontier_plr|mna_frontier_accel|mna_frontier_adaptive|mna_frontier_lp_plr|mna_frontier_lp025_plr|mna_frontier_lp_accel|mna_frontier_lp_adaptive} [seed] [num_updates]" >&2
  exit 2
fi

METHOD="$1"
SEED="${2:-0}"
NUM_UPDATES="${3:-30000}"
CHECKPOINT_SAVE_INTERVAL="${CHECKPOINT_SAVE_INTERVAL:-0}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JAXUED_PYTHON="${JAXUED_PYTHON:-${PROJECT_ROOT}/.venv/bin/python}"

export WANDB_MODE="${WANDB_MODE:-disabled}"
export WANDB_SILENT="${WANDB_SILENT:-true}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/jaxued-mpl}"
export XDG_CONFIG_HOME="${XDG_CONFIG_HOME:-/tmp/jaxued-xdg}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-/tmp/jaxued-jax-cache}"

SCORE="mna"
LEARNING_PROGRESS_COEFF="1.0"
ACCEL_FLAG="--no-use_accel"
ADAPTIVE_FLAG="--no-adaptive_mutation"

case "${METHOD}" in
  mna_plr)
    SCORE="mna"
    ;;
  mna_accel)
    ACCEL_FLAG="--use_accel"
    ;;
  frontier_plr)
    SCORE="frontier"
    ;;
  frontier_lp_plr)
    SCORE="frontier_lp"
    ;;
  frontier_lp_accel)
    SCORE="frontier_lp"
    ACCEL_FLAG="--use_accel"
    ;;
  frontier_lp_adaptive)
    SCORE="frontier_lp"
    ACCEL_FLAG="--use_accel"
    ADAPTIVE_FLAG="--adaptive_mutation"
    ;;
  mna_frontier_plr)
    SCORE="mna_frontier"
    ;;
  mna_frontier_accel)
    SCORE="mna_frontier"
    ACCEL_FLAG="--use_accel"
    ;;
  mna_frontier_adaptive)
    SCORE="mna_frontier"
    ACCEL_FLAG="--use_accel"
    ADAPTIVE_FLAG="--adaptive_mutation"
    ;;
  mna_frontier_lp_plr)
    SCORE="mna_frontier_lp"
    ;;
  mna_frontier_lp025_plr)
    SCORE="mna_frontier_lp"
    LEARNING_PROGRESS_COEFF="0.25"
    ;;
  mna_frontier_lp_accel)
    SCORE="mna_frontier_lp"
    ACCEL_FLAG="--use_accel"
    ;;
  mna_frontier_lp_adaptive)
    SCORE="mna_frontier_lp"
    ACCEL_FLAG="--use_accel"
    ADAPTIVE_FLAG="--adaptive_mutation"
    ;;
  *)
    echo "Unknown method: ${METHOD}" >&2
    exit 2
    ;;
esac

cd "${PROJECT_ROOT}"
exec "${JAXUED_PYTHON}" examples/maze_plr.py \
  --project local \
  --run_name "${METHOD}_u${NUM_UPDATES}_s${SEED}" \
  --seed "${SEED}" \
  --num_updates "${NUM_UPDATES}" \
  --eval_freq 250 \
  --checkpoint_save_interval "${CHECKPOINT_SAVE_INTERVAL}" \
  --lightweight_logging \
  --score_function "${SCORE}" \
  --learning_progress_coeff "${LEARNING_PROGRESS_COEFF}" \
  "${ACCEL_FLAG}" \
  "${ADAPTIVE_FLAG}"
