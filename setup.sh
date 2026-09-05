#!/usr/bin/env bash
# LingBot-VA environment setup for Alibaba Cloud PAI-PPU (真武 ZW810E).
#
# Why not README's plain CUDA recipe?
#   README: Python 3.10 + torch==2.9.0 (cu126) + flash-attn from PyPI
#   This DSW: Python 3.12 + PPU_SDK 2.0.0 + aiext PPU pip index
#   Official download.pytorch.org cu126 wheels will NOT run on PPU.
#   No conda: uses stdlib venv.
#
# Strategy (default):
#   Create .venv with --system-site-packages so the image's verified PPU stack
#   (torch / torchvision / torchaudio / flash-attn) is reused, then install
#   LingBot-VA pure-Python deps on top without overwriting those packages.
#
# Usage:
#   bash setup.sh                 # recommended
#   bash setup.sh --base-only     # skip lerobot / wandb post-train extras
#   bash setup.sh --force         # recreate .venv
#   bash setup.sh --upgrade-torch # also overlay torch==2.8.0+ppu2.0.0.oe
#   source .venv/bin/activate

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

VENV_DIR="${VENV_DIR:-${ROOT}/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
BASE_ONLY=0
FORCE=0
UPGRADE_TORCH=0

for arg in "$@"; do
  case "${arg}" in
    --base-only) BASE_ONLY=1 ;;
    --force) FORCE=1 ;;
    --upgrade-torch) UPGRADE_TORCH=1 ;;
    -h|--help)
      sed -n '2,22p' "$0"
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown arg: ${arg}" >&2
      exit 1
      ;;
  esac
done

log() { printf '[setup] %s\n' "$*"; }
die() { printf '[setup][ERROR] %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# 0) PPU runtime
# ---------------------------------------------------------------------------
[[ -d /usr/local/PPU_SDK ]] || die "PPU_SDK not found. Please use a PAI-PPU DSW/DLC image."
command -v "${PYTHON_BIN}" >/dev/null || die "Python not found: ${PYTHON_BIN}"

if [[ -f /usr/local/PPU_SDK/envsetup.sh ]]; then
  # shellcheck disable=SC1091
  source /usr/local/PPU_SDK/envsetup.sh || true
fi

export PPU_SDK="${PPU_SDK:-/usr/local/PPU_SDK}"
export CUDA_HOME="${CUDA_HOME:-${PPU_SDK}/CUDA_SDK}"
export CUDA_PATH="${CUDA_PATH:-${CUDA_HOME}}"
export PATH="${CUDA_HOME}/bin:${PPU_SDK}/bin:${PPU_SDK}/ppu-smi/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${PPU_SDK}/lib:${LD_LIBRARY_PATH:-}"

# Official PPU index (already set on PAI images; keep / re-export for safety).
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://aiext-pypi.mirrors.aliyuncs.com/pg1-pip/ubuntu_cu128/simple/}"
# Pure-Python deps: official PyPI only (no tuna/tsinghua/aliyun mirrors).
export PIP_EXTRA_INDEX_URL="${PIP_EXTRA_INDEX_URL:-https://pypi.org/simple/}"
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_ROOT_USER_ACTION=ignore

# Optional local proxy (Clash etc.). Default 127.0.0.1:7897 if reachable.
PROXY_URL="${PROXY_URL:-http://127.0.0.1:7897}"
if [[ -z "${http_proxy:-}${HTTP_PROXY:-}" ]]; then
  if curl -fsS -o /dev/null --connect-timeout 1 -x "${PROXY_URL}" https://pypi.org/simple/pip/ >/dev/null 2>&1; then
    export http_proxy="${PROXY_URL}"
    export https_proxy="${PROXY_URL}"
    export HTTP_PROXY="${PROXY_URL}"
    export HTTPS_PROXY="${PROXY_URL}"
    export ALL_PROXY="${PROXY_URL}"
    log "Using proxy ${PROXY_URL}"
  else
    log "Proxy ${PROXY_URL} not reachable from this host; using direct network."
  fi
else
  log "Using existing proxy env (http_proxy/HTTP_PROXY)."
fi
# Keep PAI internal / PPU indexes off the proxy.
export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,::1,.aliyuncs.com,.t-head.cn,aiext-pypi.mirrors.aliyuncs.com}"
export no_proxy="${NO_PROXY}"

# Highest torch verified installable on this image (SDK 2.0.0 / py312 / ubuntu24.04).
# Note: torch 2.9.0+ppu* is listed in the index but NOT compatible with this env.
TORCH_UPGRADE_VER="${TORCH_UPGRADE_VER:-2.8.0+ppu2.0.0.oe}"

log "Repo: ${ROOT}"
log "Python: $(${PYTHON_BIN} --version 2>&1)"
log "PPU_SDK: ${PPU_SDK} ($(cat "${PPU_SDK}/VERSION.txt" 2>/dev/null || echo unknown))"
log "PIP_INDEX_URL: ${PIP_INDEX_URL}"

if command -v ppu-smi >/dev/null 2>&1; then
  ppu-smi -L 2>/dev/null || true
fi

# Baseline image stack (must already work before we layer deps).
"${PYTHON_BIN}" - <<'PY' || die "Image PPU torch stack is broken; recreate DSW from a PAI-PPU training image."
import torch
assert torch.cuda.is_available(), "torch.cuda.is_available() == False"
print(f"image_torch={torch.__version__} device0={torch.cuda.get_device_name(0)} n={torch.cuda.device_count()}")
PY

# ---------------------------------------------------------------------------
# 1) venv without conda — inherit PPU CUDA extensions from the image
# ---------------------------------------------------------------------------
if [[ -d "${VENV_DIR}" && "${FORCE}" -eq 1 ]]; then
  log "Removing existing venv (--force): ${VENV_DIR}"
  rm -rf "${VENV_DIR}"
fi

if [[ ! -d "${VENV_DIR}" ]]; then
  log "Creating venv with --system-site-packages at ${VENV_DIR}"
  "${PYTHON_BIN}" -m venv --system-site-packages "${VENV_DIR}"
else
  log "Reusing venv: ${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
python -m pip install -U pip setuptools wheel

if [[ "${UPGRADE_TORCH}" -eq 1 ]]; then
  log "Overlaying torch==${TORCH_UPGRADE_VER} (vision/audio/flash-attn stay on image builds) ..."
  python -m pip install --upgrade "torch==${TORCH_UPGRADE_VER}"
  log "WARNING: flash-attn / torchvision may remain on image versions; if imports break, re-run without --upgrade-torch."
else
  log "Keeping image torch / flash-attn (recommended). Pass --upgrade-torch to try ${TORCH_UPGRADE_VER}."
fi

# Pin currently visible CUDA/PPU packages so later pip installs cannot replace them
# with download.pytorch.org / generic manylinux wheels.
CONSTRAINTS="${ROOT}/.ppu-constraints.txt"
python - <<'PY' > "${CONSTRAINTS}"
import importlib.metadata as md
pkgs = [
    "torch", "torchvision", "torchaudio",
    "flash-attn", "triton", "xformers",
]
print("# Auto-generated by setup.sh — keeps PPU binary stack stable.")
for name in pkgs:
    try:
        print(f"{name}=={md.version(name)}")
    except md.PackageNotFoundError:
        pass
PY
log "Wrote constraints: ${CONSTRAINTS}"

# ---------------------------------------------------------------------------
# 2) LingBot-VA Python deps (README Installation + post-training)
# ---------------------------------------------------------------------------
REQ_PPU="${ROOT}/requirements-ppu.txt"
cat > "${REQ_PPU}" <<'EOF'
# PPU setup companion for requirements.txt — no torch / flash-attn pins.
# Those come from the PAI-PPU image (or --upgrade-torch).
diffusers==0.36.0
transformers==4.55.2
accelerate
einops
easydict
peft>=0.17.0
numpy==1.26.4
tqdm
imageio[ffmpeg]
websockets
msgpack
opencv-python
matplotlib
ftfy
safetensors
Pillow
EOF

log "Installing LingBot-VA deps ..."
python -m pip install -c "${CONSTRAINTS}" -r "${REQ_PPU}"

if [[ "${BASE_ONLY}" -eq 0 ]]; then
  log "Installing post-training extras ..."
  # README: lerobot without dragging an alternate torch stack.
  python -m pip install -c "${CONSTRAINTS}" "lerobot==0.3.3" --no-deps
  python -m pip install -c "${CONSTRAINTS}" scipy wandb huggingface_hub datasets av
fi

if [[ -f "${ROOT}/pyproject.toml" ]]; then
  log "Editable install of this repo (--no-deps) ..."
  python -m pip install -e "${ROOT}" --no-deps || log "Editable install skipped (optional)."
fi

# ---------------------------------------------------------------------------
# 3) Sanity check
# ---------------------------------------------------------------------------
log "Running PPU sanity check ..."
python - <<'PY'
import sys
import torch

print(f"python: {sys.version.split()[0]}")
print(f"torch: {torch.__version__}")
print(f"cuda_available: {torch.cuda.is_available()}")
print(f"device_count: {torch.cuda.device_count()}")
assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
print(f"device0: {torch.cuda.get_device_name(0)}")

x = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
y = x @ x
torch.cuda.synchronize()
print(f"matmul_ok: shape={tuple(y.shape)}")

try:
    import flash_attn
    print(f"flash_attn: {getattr(flash_attn, '__version__', 'ok')}")
except Exception as e:
    print(f"flash_attn warning: {e}")

for pkg in ("diffusers", "transformers", "accelerate", "einops", "easydict"):
    mod = __import__(pkg)
    print(f"{pkg}: {getattr(mod, '__version__', 'ok')}")

print("SANITY_OK")
PY

cat <<EOF

============================================================
LingBot-VA PPU env ready.

Activate:
  source ${VENV_DIR}/bin/activate

Mapped from README Installation:
  Python 3.10.16     -> image Python 3.12 (PPU wheels)
  torch 2.9.0+cu126  -> image PPU torch (optional overlay ${TORCH_UPGRADE_VER})
  flash-attn         -> image PPU flash-attn (do NOT pip from PyPI source)

Critical:
  1) Never: pip install torch --index-url https://download.pytorch.org/whl/cu126
  2) Train:  set <ckpt>/transformer/config.json attn_mode="flex"
     Infer:  set attn_mode="torch" or "flashattn"
  3) RoboTwin / LIBERO sim stacks are separate (see README).

Flags:
  bash setup.sh --force
  bash setup.sh --base-only
  bash setup.sh --upgrade-torch
============================================================
EOF
