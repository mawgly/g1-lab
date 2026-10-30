#!/usr/bin/env bash
# Builds .venv-fullgpu on a rented Linux GPU server (Ubuntu with an NVIDIA driver), in ~/sim. Idempotent: a second run
# only checks. Usage: bash scripts/remote/setup_linux.sh [cu130|cu128]
# cu130 (default) is the lab's torch 2.14.1+cu130: Turing (sm_75, RTX 20xx) and newer, driver >= 580. Older cards
# (Pascal, Volta) need cu128.
set -euo pipefail
CUDA=${1:-cu130}
cd ~/sim
export PATH=$HOME/.local/bin:$PATH
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
if ! .venv-fullgpu/bin/python -c 'import torch, mujoco_warp' 2>/dev/null; then
  uv python install 3.12
  uv venv --python 3.12 .venv-fullgpu
  uv pip install --python .venv-fullgpu/bin/python "torch==2.14.1+$CUDA" --index-url "https://download.pytorch.org/whl/$CUDA"
  uv pip install --python .venv-fullgpu/bin/python -r requirements-gpu.txt
fi
.venv-fullgpu/bin/python -c "import torch, mujoco, mujoco_warp, warp; assert torch.cuda.is_available(), 'CUDA GPU unavailable'; print('ready: CUDA', torch.version.cuda, torch.cuda.get_device_name(0), round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1), 'GiB')"
