#!/bin/bash
# LingBot-VA post-train launcher (user entrypoint).
set -euo pipefail

# ---------------------------------------------------------------------------
# Paths (edit here)
# ---------------------------------------------------------------------------
PROJECT_ROOT="/mnt/workspace/users/wanganran_2T/lingbot-va"
VENV_PATH="${PROJECT_ROOT}/.venv/bin/activate"
CACHE_ROOT="/mnt/workspace/users/wanganran_2T/.cache/lingbot-va"
CKPT_ROOT="/mnt/workspace/users/wanganran_2T/ckpt"
OUTPUT_BASE="/mnt/workspace/users/wanganran_2T/ckpt/lingbot-va-runs"

CONFIG_NAME="${CONFIG_NAME:-uniarm_train}"
EXP_NAME="${EXP_NAME:-uniarm_pick_key}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_BASE}/${EXP_NAME}}"

# ---------------------------------------------------------------------------
# Hardware / distributed
# ---------------------------------------------------------------------------
NUM_GPUS="${NUM_GPUS:-2}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1}"
MASTER_PORT="${MASTER_PORT:-29501}"

# ---------------------------------------------------------------------------
# Train hyper-params (override cfg; leave empty to keep cfg defaults)
# ---------------------------------------------------------------------------
LEARNING_RATE="${LEARNING_RATE:-}"
NUM_STEPS="${NUM_STEPS:-}"
BATCH_SIZE="${BATCH_SIZE:-16}"
GRAD_ACCUM="${GRAD_ACCUM:-}"
SAVE_INTERVAL="${SAVE_INTERVAL:-}"
LOAD_WORKER="${LOAD_WORKER:-18}"
RESUME_FROM="${RESUME_FROM:-}"

# ---------------------------------------------------------------------------
# WandB / HuggingFace
# ---------------------------------------------------------------------------
ENABLE_WANDB="${ENABLE_WANDB:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-lingbot-va-uniarm}"
WANDB_TEAM_NAME="${WANDB_TEAM_NAME:-}"
WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.wandb.ai}"
# WANDB_API_KEY: export externally or fill below
# WANDB_API_KEY=""

HF_OFFLINE="${HF_OFFLINE:-1}"

# ---------------------------------------------------------------------------
# Env setup
# ---------------------------------------------------------------------------
cd "${PROJECT_ROOT}"
# shellcheck disable=SC1090
source "${VENV_PATH}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

mkdir -p \
  "${CACHE_ROOT}/huggingface/hub" \
  "${CACHE_ROOT}/huggingface/datasets" \
  "${CACHE_ROOT}/xdg" \
  "${CACHE_ROOT}/torch" \
  "${CACHE_ROOT}/torch_extensions" \
  "${CACHE_ROOT}/triton" \
  "${CACHE_ROOT}/cuda" \
  "${CACHE_ROOT}/tmp" \
  "${CACHE_ROOT}/wandb/cache" \
  "${CACHE_ROOT}/wandb/config" \
  "${CACHE_ROOT}/wandb/data" \
  "${OUTPUT_DIR}"

export HF_HOME="${CACHE_ROOT}/huggingface"
export HF_HUB_CACHE="${CACHE_ROOT}/huggingface/hub"
export HF_DATASETS_CACHE="${CACHE_ROOT}/huggingface/datasets"
export HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE}"
export TRANSFORMERS_CACHE="${HF_HUB_CACHE}"
export TRANSFORMERS_OFFLINE="${HF_OFFLINE}"
export HF_HUB_OFFLINE="${HF_OFFLINE}"

export XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
export TORCH_HOME="${CACHE_ROOT}/torch"
export TORCH_EXTENSIONS_DIR="${CACHE_ROOT}/torch_extensions"
export TRITON_CACHE_DIR="${CACHE_ROOT}/triton"
export CUDA_CACHE_PATH="${CACHE_ROOT}/cuda"
# Unix-domain sockets for multiprocessing must NOT live on shared/network FS
# (/mnt/workspace often returns OSError 95). Prefer local tmpfs.
LOCAL_TMP="${LOCAL_TMP:-/dev/shm/lingbot-va-tmp}"
mkdir -p "${LOCAL_TMP}"
export TMPDIR="${LOCAL_TMP}"
export TEMP="${LOCAL_TMP}"
export TMP="${LOCAL_TMP}"
# Keep a project-side scratch dir for large non-socket temp files if needed.
mkdir -p "${CACHE_ROOT}/tmp"

export WANDB_DIR="${OUTPUT_DIR}"
export WANDB_CACHE_DIR="${CACHE_ROOT}/wandb/cache"
export WANDB_CONFIG_DIR="${CACHE_ROOT}/wandb/config"
export WANDB_DATA_DIR="${CACHE_ROOT}/wandb/data"
export WANDB_PROJECT
export WANDB_BASE_URL
if [[ -n "${WANDB_TEAM_NAME}" ]]; then
  export WANDB_TEAM_NAME
fi

export NO_ALBUMENTATIONS_UPDATE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# FlexAttention on PPU:
# - Keep using flex_attention + BlockMask (same training recipe).
# - Do NOT torch.compile(..., backend=inductor): flex_attention_backward
#   fails with NoValidChoicesError on this stack.
# - Default: eager flex (LINGBOT_COMPILE_FLEX=0). Optional experiment:
#   LINGBOT_COMPILE_FLEX=1 LINGBOT_FLEX_COMPILE_BACKEND=aot_eager
export LINGBOT_COMPILE_FLEX="${LINGBOT_COMPILE_FLEX:-0}"
export LINGBOT_FLEX_DYNAMIC="${LINGBOT_FLEX_DYNAMIC:-0}"
export LINGBOT_FLEX_COMPILE_BACKEND="${LINGBOT_FLEX_COMPILE_BACKEND:-inductor}"
# Avoid reusing broken inductor artifacts from prior failed compiles.
export TORCHINDUCTOR_CACHE_DIR="${LOCAL_TMP}/torchinductor"

# This DSW often has no usable local Clash; clear proxies to avoid wandb/HF ProxyError.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY || true

if [[ ! -f "${HF_HOME}/token" && -f /root/.cache/huggingface/token ]]; then
  cp /root/.cache/huggingface/token "${HF_HOME}/token"
  echo "Synced HF token -> ${HF_HOME}/token"
fi

EXTRA_ARGS=(--config-name "${CONFIG_NAME}" --save-root "${OUTPUT_DIR}")
if [[ "${ENABLE_WANDB}" == "1" || "${ENABLE_WANDB}" == "true" ]]; then
  if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "ERROR: ENABLE_WANDB=1 时请先 export WANDB_API_KEY=..."
    exit 1
  fi
  export WANDB_API_KEY
  EXTRA_ARGS+=(--enable-wandb)
  if [[ -n "${WANDB_RUN_ID:-}" ]]; then
    export WANDB_RUN_ID
    export WANDB_RESUME="${WANDB_RESUME:-allow}"
  fi
else
  unset WANDB_MODE || true
  EXTRA_ARGS+=(--disable-wandb)
fi

[[ -n "${LEARNING_RATE}" ]] && EXTRA_ARGS+=(--learning-rate "${LEARNING_RATE}")
[[ -n "${NUM_STEPS}" ]] && EXTRA_ARGS+=(--num-steps "${NUM_STEPS}")
[[ -n "${BATCH_SIZE}" ]] && EXTRA_ARGS+=(--batch-size "${BATCH_SIZE}")
[[ -n "${GRAD_ACCUM}" ]] && EXTRA_ARGS+=(--gradient-accumulation-steps "${GRAD_ACCUM}")
[[ -n "${SAVE_INTERVAL}" ]] && EXTRA_ARGS+=(--save-interval "${SAVE_INTERVAL}")
[[ -n "${LOAD_WORKER}" ]] && EXTRA_ARGS+=(--load-worker "${LOAD_WORKER}")
[[ -n "${RESUME_FROM}" ]] && EXTRA_ARGS+=(--resume-from "${RESUME_FROM}")

echo "===== LingBot-VA train ====="
echo "PWD=$(pwd)"
echo "CONFIG_NAME=${CONFIG_NAME}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "NUM_GPUS=${NUM_GPUS} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "HF_HOME=${HF_HOME} HF_OFFLINE=${HF_OFFLINE}"
echo "ENABLE_WANDB=${ENABLE_WANDB} WANDB_PROJECT=${WANDB_PROJECT}"
echo "RESUME_FROM=${RESUME_FROM:-<none>}"
echo "EXTRA_ARGS=${EXTRA_ARGS[*]} $*"
echo "============================"

export NGPU="${NUM_GPUS}"
export MASTER_PORT
export CONFIG_NAME
bash "${PROJECT_ROOT}/script/run_va_posttrain.sh" "${EXTRA_ARGS[@]}" "$@"
