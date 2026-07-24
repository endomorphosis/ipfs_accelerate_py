# Installation

IPFS Accelerate Python has a small base install and feature-scoped optional
extras. Choose the smallest profile that supports the workload, then verify
the capabilities that the host actually exposes.

## Requirements

- Python 3.8 or newer, as declared in `pyproject.toml`;
- a supported operating system and a working compiler/runtime for any native
  optional dependency;
- network access when downloading packages or models; and
- additional drivers, credentials, or daemons for optional integrations.

Model size and device memory are workload-dependent. There is no universal RAM,
GPU, or storage requirement for every supported model.

## Published package

```bash
python -m pip install -U pip
python -m pip install ipfs-accelerate-py
```

The package metadata currently defines these extras:

| Extra | Adds |
| --- | --- |
| `minimal` | Small runtime dependency set. |
| `dev` | pytest and local development tooling. |
| `full` | Transformers, PyTorch, server, and model-manager integrations. |
| `mcp` | MCP server and GitHub integration dependencies. |
| `mcp-p2p` / `libp2p` | libp2p and TaskQueue networking dependencies. |
| `webnn` | Browser/WebNN/WebGPU support dependencies. |
| `llama_cpp` | llama.cpp server support. |
| `analysis` | Data analysis and scientific Python dependencies. |
| `monitoring` | Host and NVIDIA monitoring helpers. |
| `testing` | Broad test-suite dependencies. |
| `all` | Aggregate application dependencies, without native P2P by default. |

Install an extra with:

```bash
python -m pip install "ipfs-accelerate-py[mcp]"
python -m pip install "ipfs-accelerate-py[full]"
```

## Source checkout

```bash
git clone https://github.com/endomorphosis/ipfs_accelerate_py.git
cd ipfs_accelerate_py
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

For an editable installation with all application dependencies:

```bash
python -m pip install -e ".[all]"
```

## CUDA and PyTorch

PyPI may resolve a CPU-only PyTorch wheel. Select a CUDA wheel from the
official PyTorch index that matches the host driver. This repository includes
requirements files for CUDA 12.4 and CUDA 13 nightly builds:

```bash
python -m pip install --upgrade --force-reinstall \
  -r install/requirements_torch_cu124.txt
```

For newer NVIDIA GB10/DGX Spark-class systems, use the repository helper when a
CUDA 13 nightly build is required:

```bash
./scripts/install_torch_cuda_cu130_nightly.sh
```

Verify the runtime, not only `nvidia-smi`:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("torch_cuda:", torch.version.cuda)
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY
```

The driver, PyTorch build, model kernels, and device architecture must all be
compatible. A CUDA driver by itself does not make a model path CUDA-backed.

## Other optional backends

ROCm, OpenVINO, MPS, WebNN, WebGPU, Qualcomm, and browser runtimes are
environment-specific. Install their upstream runtime and the matching package
extra when one exists. The package should still import when an optional backend
is absent.

Use the capability report after installation:

```bash
python - <<'PY'
from ipfs_accelerate_py import get_instance

report = get_instance().get_capabilities(detail=True)
print(report.get("hardware", {}))
print(report.get("models", []))
PY
```

## MCP

Install the MCP extra and start the canonical server locally:

```bash
python -m pip install "ipfs-accelerate-py[mcp]"
ipfs-accelerate mcp start --host 127.0.0.1 --port 9000
```

Use the [MCP setup guide](../MCP_SETUP_GUIDE.md) for transport and deployment
details. Do not bind an unauthenticated development server to a public
interface.

## IPFS and P2P

IPFS and P2P are optional. Local inference and local model caches do not require
a Kubo daemon. If the workload needs a Kubo-compatible service, `ipfs_kit_py`,
or libp2p TaskQueue, install the relevant extra and configure the service
separately. See [IPFS integration](../../features/ipfs/IPFS.md) and the
[P2P guides](../p2p/README.md).

## Build a wheel

```bash
python -m pip install build
python -m build
python -m pip install dist/*.whl
```

The package version is exposed at runtime:

```bash
python -c "import ipfs_accelerate_py; print(ipfs_accelerate_py.__version__)"
```

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Package import fails | Activate the intended virtual environment and inspect `python -m pip show ipfs-accelerate-py`. |
| Optional module is unavailable | Install its feature extra and check `get_capabilities(detail=True)`. |
| CUDA reports false | Compare `torch.version.cuda`, the driver, and the selected wheel; then run a real model smoke. |
| MCP status fails | Check the port, server log, and `ipfs-accelerate mcp status`. |
| IPFS connection is refused | Start/configure the external daemon or use local storage instead. |
| Native dependency fails to build | Use a supported Python/OS toolchain and follow that dependency's installation instructions. |

## Development install check

```bash
python -m pytest \
  test/test_unified_cli_integration.py \
  test/api/test_agent_supervisor_objective_graph.py -q
```

For the complete testing policy, see [Testing](../../development/testing.md).
