#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "ERROR: No active virtualenv detected. Activate your venv first (e.g. 'source .venv/bin/activate')." >&2
  exit 1
fi

python -m pip install -U pip

# Install CUDA 13.0 nightly builds (useful for very new GPUs like NVIDIA GB10 / sm_121).
python -m pip install --upgrade --force-reinstall -r install/requirements_torch_cu130_nightly.txt

python - <<'PY'
import torch
print('torch:', torch.__version__)
print('torch.version.cuda:', torch.version.cuda)
print('torch.cuda.is_available():', torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit('ERROR: CUDA is not available in torch. Check NVIDIA drivers / container runtime / permissions.')

# Run a tiny kernel to ensure execution works.
x = torch.randn((8, 8), device='cuda')
y = x @ x
print('cuda matmul ok:', y.shape)
print('device:', torch.cuda.get_device_name(0))
print('capability:', torch.cuda.get_device_capability(0))
PY
