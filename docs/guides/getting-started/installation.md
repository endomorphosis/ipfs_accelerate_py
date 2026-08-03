# Installation

**Status:** Current
**Audience:** Developers and operators installing the package for the first time
**Scope:** Python range, published and editable installs, optional extras from
`pyproject.toml`, CUDA/PyTorch selection, MCP and IPFS/P2P prerequisites, and
offline verification
**Non-goals:** Resolving packaging-versus-runtime version disagreement in code;
deep MCP, hardware, or supervisor operations (see the linked guides)
**Source anchors:** `pyproject.toml` (`[project]`, `[project.optional-dependencies]`,
`[project.scripts]`), `setup.py` (`version=`), `ipfs_accelerate_py/__init__.py`
(`__version__`), `install/requirements_torch_*.txt`,
`scripts/install_torch_cuda_cu130_nightly.sh`

IPFS Accelerate Python has a small base install and feature-scoped optional
extras. Choose the smallest profile that supports the workload, then verify the
capabilities that the host actually exposes. **Import success is not a
capability signal.**

This is the canonical installation path. Do not maintain a second
case-colliding filename (`INSTALLATION.md`); older uppercase links should be
updated to this page.

## Requirements

- Python 3.8 or newer (`requires-python = ">=3.8"` in `pyproject.toml`)
- A supported OS (Linux, macOS, or Windows) and a working toolchain for any
  native optional dependency you install
- Network access when downloading packages or models
- Additional drivers, credentials, or daemons only for the optional integrations
  you enable

Model size and device memory are workload-dependent. There is no universal RAM,
GPU, or storage requirement for every supported model.

## Version sources (code-owned disagreement)

Packaging metadata and the runtime export currently disagree. Documentation
must not invent a single “true” product version:

| Source | Field | Value on this tree |
| --- | --- | --- |
| `pyproject.toml` | `[project].version` | `0.0.45` |
| `setup.py` | `version=` | `0.0.45` |
| `ipfs_accelerate_py/__init__.py` | `__version__` | `0.4.0` |

When you need a version string, quote the file it came from. For example:

```bash
# Packaging metadata (installed dist, when available)
python -m pip show ipfs-accelerate-py

# Runtime export (after import)
python -c "import ipfs_accelerate_py; print(ipfs_accelerate_py.__version__)"
```

Those two commands may print different values until packaging and the package
export are reconciled in code. Do not treat either value as proof that the
other is wrong in prose.

## Published package

The project name in metadata is `ipfs_accelerate_py`. Pip normalizes that to
`ipfs-accelerate-py` on the command line:

```bash
python -m pip install -U pip
python -m pip install ipfs-accelerate-py
```

## Optional extras

Extras are defined only in `pyproject.toml` under
`[project.optional-dependencies]`. There are **no** packaging extras named
`cuda`, `openvino`, or `rocm`; those backends are environment- and
wheel-specific (see [CUDA and PyTorch](#cuda-and-pytorch) and
[Other optional backends](#other-optional-backends)).

| Extra | Adds (summary) |
| --- | --- |
| `minimal` | Small runtime set (aiohttp, duckdb, IPFS HTTP client, websockets, tqdm, numpy). |
| `dev` | pytest stack, anyio/httpx, FastAPI/uvicorn for local tests, black/flake8/mypy. |
| `textgen` | Transformers (text-generation oriented). |
| `full` | PyTorch, Transformers, FastAPI/uvicorn, sentence-transformers, model-manager and transformers IPFS integrations. |
| `mcp` | MCP server and GitHub integration dependencies (fastmcp, Flask stack, PyGithub). |
| `mcp-p2p` | libp2p and related networking deps for TaskQueue/P2P. |
| `libp2p` | Same libp2p dependency set as `mcp-p2p` under an alternate extra name. |
| `webnn` | Browser/WebNN/WebGPU support dependencies (Playwright, aiohttp, websockets). |
| `llama_cpp` | llama.cpp server support and Hugging Face Hub. |
| `distributed` | aiohttp, websockets, networkx. |
| `scraping` | HTML/article scraping stack (BeautifulSoup, Playwright, newspaper3k, readability). |
| `analysis` | pandas, numpy, scipy, scikit-learn. |
| `viz` | matplotlib, seaborn, plotly. |
| `monitoring` | Host and NVIDIA monitoring helpers (psutil, py-cpuinfo, nvidia-ml-py). |
| `testing` | Broad test-suite dependency set (includes serving TestClient stack). |
| `all` | Aggregate application dependencies **without** native P2P/libp2p by default. |

Install only what the workload needs:

```bash
python -m pip install "ipfs-accelerate-py[mcp]"
python -m pip install "ipfs-accelerate-py[full]"
python -m pip install "ipfs-accelerate-py[dev]"
```

Multiple extras can be combined:

```bash
python -m pip install "ipfs-accelerate-py[mcp,webnn]"
```

For the authoritative dependency pins, read
`[project.optional-dependencies]` in `pyproject.toml` rather than copying
stale lists from older guides.

## Console scripts

After install, `pyproject.toml` registers (among others):

| Command | Entry point |
| --- | --- |
| `ipfs-accelerate` | `ipfs_accelerate_py.cli_entry:main` (canonical product CLI) |
| `ipfs_accelerate` | `ipfs_accelerate_py.ai_inference_cli:main` (separate underscore parser) |
| `ipfs-accelerate-llama-cpp-serve` | llama.cpp helper |
| `ipfs-accelerate-agent-*` | Agent supervisor daemons and helpers |

Use each command’s own `--help`. Do not mix flags between `ipfs-accelerate` and
`ipfs_accelerate`.

Offline help smoke (no model download):

```bash
ipfs-accelerate --help
ipfs-accelerate models --help
ipfs-accelerate mcp --help
ipfs-accelerate agent --help
```

## Source checkout (editable)

```bash
git clone https://github.com/endomorphosis/ipfs_accelerate_py.git
cd ipfs_accelerate_py
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

For an editable install with the aggregate application set:

```bash
python -m pip install -e ".[all]"
```

For MCP development:

```bash
python -m pip install -e ".[mcp]"
```

Some integrations live in sibling repositories or optional git extras (for
example entries under the `full` / `all` extras). Those require network access
at install time and are not part of the base wheel.

## CUDA and PyTorch

PyPI may resolve a CPU-only PyTorch wheel. Select a CUDA wheel that matches the
host driver from the official PyTorch index when you need GPU execution. This
repository includes requirements files for CUDA 12.4 and CUDA 13 nightly
builds:

```bash
python -m pip install --upgrade --force-reinstall \
  -r install/requirements_torch_cu124.txt
```

For newer NVIDIA GB10/DGX Spark-class systems that need a CUDA 13 nightly
build, use the repository helper:

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
extra when one exists (`webnn` for browser automation paths). The package
should still import when an optional backend is absent.

Use the capability report after installation:

```bash
python - <<'PY'
from ipfs_accelerate_py import get_instance

report = get_instance().get_capabilities(detail=True)
print(report.get("hardware", {}))
print(report.get("task_types", []))
print(report.get("models", []))
PY
```

## MCP

Install the MCP extra and start the canonical server locally:

```bash
python -m pip install "ipfs-accelerate-py[mcp]"
ipfs-accelerate mcp start --host 127.0.0.1 --port 9000
```

In another terminal:

```bash
ipfs-accelerate mcp status --host 127.0.0.1 --port 9000
```

Use the [MCP setup guide](../MCP_SETUP_GUIDE.md) for transport and deployment
details. Do not bind an unauthenticated development server to a public
interface.

## IPFS and P2P

IPFS and P2P are optional. Local inference and local model caches do not require
a Kubo daemon. If the workload needs a Kubo-compatible service, `ipfs_kit_py`,
or libp2p TaskQueue, install the relevant extra (`mcp-p2p` / `libp2p` where
applicable) and configure the service separately. See
[IPFS integration](../../features/ipfs/IPFS.md) and the
[P2P guides](../p2p/README.md).

## Build a wheel

```bash
python -m pip install build
python -m build
python -m pip install dist/*.whl
```

## Offline verification recipe

These checks do not require model weights, CUDA, IPFS, or MCP:

```bash
# 1) Import and report runtime export (may differ from packaging metadata)
python - <<'PY'
import ipfs_accelerate_py
from ipfs_accelerate_py import get_instance

print("runtime __version__:", ipfs_accelerate_py.__version__)
print("capabilities:", get_instance().get_capabilities())
PY

# 2) CLI help surfaces (canonical hyphenated entry point)
ipfs-accelerate --help
```

Optional packaging metadata check (installed distribution):

```bash
python -m pip show ipfs-accelerate-py
```

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Package import fails | Activate the intended virtual environment; run `python -m pip show ipfs-accelerate-py`. |
| Optional module is unavailable | Install its feature extra; inspect `get_capabilities(detail=True)`. |
| CUDA reports false | Compare `torch.version.cuda`, the driver, and the selected wheel; then run a real model smoke. |
| MCP status fails | Check the port, server log, and `ipfs-accelerate mcp status`. |
| IPFS connection is refused | Start/configure the external daemon, or use local storage instead. |
| Native dependency fails to build | Use a supported Python/OS toolchain and follow that dependency’s own install docs. |
| Version strings disagree | Expected until code reconciles packaging (`0.0.45`) and `__version__` (`0.4.0`); cite the source of each value. |

## Development install check

```bash
python -m pytest \
  test/test_unified_cli_integration.py \
  test/api/test_agent_supervisor_objective_graph.py -q
```

Live suite paths use the `test/` tree (not `tests/`). For the complete testing
policy, see [Testing](../../development/testing.md).

## Next steps

- [Getting started](README.md) — first useful Python and CLI operations
- [Quick start](../QUICKSTART.md) — short path across CLI, MCP, and APIs
- [API overview](../../api/overview.md)
- [Hardware guide](../hardware/overview.md)
