#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 CHECKPOINT_DIRECTORY [checkpoint_step|-1]" >&2
  exit 2
fi

CHECKPOINT_DIRECTORY="$1"
CHECKPOINT_STEP="${2:--1}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JAXUED_PYTHON="${JAXUED_PYTHON:-${PROJECT_ROOT}/.venv/bin/python}"

export WANDB_MODE="disabled"
export WANDB_SILENT="true"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/jaxued-mpl}"
export XDG_CONFIG_HOME="${XDG_CONFIG_HOME:-/tmp/jaxued-xdg}"

cd "${PROJECT_ROOT}"
exec "${JAXUED_PYTHON}" examples/maze_plr.py \
  --mode eval \
  --checkpoint_directory "${CHECKPOINT_DIRECTORY}" \
  --checkpoint_to_eval "${CHECKPOINT_STEP}" \
  --lightweight_logging
