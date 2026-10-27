#!/usr/bin/env bash
# One-time setup: checks the GPU, installs uv if needed, gets sir's repo and its environment,
# and checks that this repo's tools and checkpoints are in place.   Usage: ./setup.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
set -a; source "$HERE/config.env"; set +a

echo "== 1/5 GPU"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
    echo "WARNING: nvidia-smi not found. The simulator (MuJoCo Warp) needs an NVIDIA GPU with CUDA on Linux."
fi

echo "== 2/5 uv"
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

echo "== 3/5 sir's repo: $SIR_REPO"
if [ ! -d "$SIR_REPO/.git" ]; then
    git clone https://github.com/YADlin/spirob-mjlab "$SIR_REPO"
    git -C "$SIR_REPO" checkout "$SIR_BRANCH"
else
    echo "using the existing repo (its current branch is kept)"
fi
echo "branch: $(git -C "$SIR_REPO" rev-parse --abbrev-ref HEAD) @ $(git -C "$SIR_REPO" rev-parse --short HEAD)"
(cd "$SIR_REPO" && uv sync)

echo "== 4/5 Python environment + this repo's tools"
(cd "$SIR_REPO" && uv run python -c "import torch, mjlab; print('torch', torch.__version__, '| CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU')")
(cd "$SIR_REPO" && uv run python -m py_compile "$HERE"/tools/*.py) && echo "tools compile OK"

echo "== 5/5 checkpoints"
missing=0
for f in teacher_C1_a3.pt student_c1.pt; do
    if [ -f "$HERE/checkpoints/$f" ]; then echo "found   checkpoints/$f"; else echo "MISSING checkpoints/$f"; missing=1; fi
done
[ "$missing" = 0 ] && echo -e "\nSetup done. Next:  ./run.sh check" || echo -e "\nAdd the missing checkpoints (see checkpoints/README.md), then:  ./run.sh check"
