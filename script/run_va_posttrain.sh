#!/usr/bin/env bash
# Distributed launcher for LingBot-VA post-training.
# Prefer calling via ../train.sh (sets caches / wandb / HF).
# Direct use:
#   NGPU=2 CONFIG_NAME=uniarm_train bash script/run_va_posttrain.sh
set -euo pipefail

umask 007

NGPU="${NGPU:-2}"
MASTER_PORT="${MASTER_PORT:-29501}"
LOG_RANK="${LOG_RANK:-0}"
TORCHFT_LIGHTHOUSE="${TORCHFT_LIGHTHOUSE:-http://localhost:29510}"
CONFIG_NAME="${CONFIG_NAME:-uniarm_train}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export TORCHFT_LIGHTHOUSE
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "[run_va_posttrain] NGPU=${NGPU} CONFIG_NAME=${CONFIG_NAME} MASTER_PORT=${MASTER_PORT}"

# Allow either: bash run_va_posttrain.sh  OR  bash run_va_posttrain.sh --config-name ...
if [[ " $* " != *" --config-name "* ]]; then
  set -- --config-name "${CONFIG_NAME}" "$@"
fi
echo "[run_va_posttrain] args: $*"

python -m torch.distributed.run \
  --nproc_per_node="${NGPU}" \
  --local-ranks-filter="${LOG_RANK}" \
  --master_port "${MASTER_PORT}" \
  --tee 3 \
  -m wan_va.train \
  "$@"
