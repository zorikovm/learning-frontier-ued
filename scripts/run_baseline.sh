#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo "Usage: $0 {dr|plr|accel} [seed] [num_updates]" >&2
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

cd "${PROJECT_ROOT}"

case "${METHOD}" in
  dr)
    exec "${JAXUED_PYTHON}" examples/maze_dr.py \
      --project local \
      --run_name "baseline_dr_u${NUM_UPDATES}_s${SEED}" \
      --seed "${SEED}" \
      --num_updates "${NUM_UPDATES}" \
      --eval_freq 250 \
      --checkpoint_save_interval "${CHECKPOINT_SAVE_INTERVAL}" \
      --lightweight_logging
    ;;
  plr|accel)
    ACCEL_FLAG="--no-use_accel"
    if [[ "${METHOD}" == "accel" ]]; then
      ACCEL_FLAG="--use_accel"
    fi
    exec "${JAXUED_PYTHON}" examples/maze_plr.py \
      --project local \
      --run_name "baseline_${METHOD}_maxmc_u${NUM_UPDATES}_s${SEED}" \
      --seed "${SEED}" \
      --num_updates "${NUM_UPDATES}" \
      --eval_freq 250 \
      --checkpoint_save_interval "${CHECKPOINT_SAVE_INTERVAL}" \
      --lightweight_logging \
      --score_function MaxMC \
      "${ACCEL_FLAG}"
    ;;
  *)
    echo "Unknown baseline: ${METHOD}" >&2
    exit 2
    ;;
esac
