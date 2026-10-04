#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "ERROR: No active virtualenv detected. Activate your venv first (e.g. 'source .venv/bin/activate')." >&2
  exit 1
fi

python -m pip install -U pip

# Force CUDA-enabled PyTorch wheels from the PyTorch CUDA index.
python -m pip install --upgrade --force-reinstall -r install/requirements_torch_cu124.txt

python - <<'PY'
import torch
print('torch:', torch.__version__)
print('torch.version.cuda:', torch.version.cuda)
print('torch.cuda.is_available():', torch.cuda.is_available())
if not torch.cuda.is_available():
  raise SystemExit('ERROR: torch installed, but CUDA is not available. Check NVIDIA drivers and that you installed a CUDA-enabled wheel.')

print('torch.cuda.device_count():', torch.cuda.device_count())
print('torch.cuda.get_device_name(0):', torch.cuda.get_device_name(0))
PY
