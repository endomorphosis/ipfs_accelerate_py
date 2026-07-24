# Installation Troubleshooting

Use the [FAQ](faq.md) for current troubleshooting and the maintained
[installation guide](../getting-started/installation.md) for package names,
extras, CUDA wheels, MCP setup, and validation commands.

## Fast diagnosis

```bash
python -m pip show ipfs-accelerate-py
python -c "import ipfs_accelerate_py; print(ipfs_accelerate_py.__version__)"
ipfs-accelerate --help
```

For a capability report:

```bash
python - <<'PY'
from ipfs_accelerate_py import get_instance
print(get_instance().get_capabilities(detail=True))
PY
```

For CUDA, check the PyTorch build and device directly:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("torch_cuda:", torch.version.cuda)
PY
```

Do not infer support from an old extra name or a static hardware label. The
available extras are defined in `pyproject.toml`, and optional providers,
services, credentials, and native runtimes must be checked separately.

For tests, begin with the deterministic commands in the
[testing guide](../../development/testing.md). Record the Python executable,
package version, selected extra, first traceback, capability report, model,
provider, and device when reporting a failure.
