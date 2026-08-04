# Getting Started

**Status:** Current
**Owner:** package maintainers
**Audience:** Developers taking a source checkout or installed package to a
first verified operation
**Scope:** Install profile choice, offline import and CLI help, capability
report, optional model run, MCP start, and optional agent supervisor entry
**Non-goals:** Full API or CLI reference; deep hardware tuning; packaging
version reconciliation
**Sources:** `pyproject.toml`, `requirements.txt`, `ipfs_accelerate_py/__init__.py`
(`get_instance`, `__version__`), `ipfs_accelerate_py/cli_entry.py`,
`ipfs_accelerate_py/ipfs_accelerate.py` (`run_model`, `get_capabilities`),
`docs/architecture/INFERENCE_RUNTIME.md`
**Last-verified:** 2026-08-03 @ f1d0bbefd (version pin reconciled)

This guide gets a source checkout or an installed package to a verified Python
import, a capability report, and an optional MCP server. The framework has
optional backends, so the first useful question is which capabilities are
available on the current host. **Import success is not a capability signal.**

## 1. Install

For a source checkout:

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

For a published installation:

```bash
python -m pip install ipfs-accelerate-py
```

The always-installed dependency set is loaded from `requirements.txt` and is
broad: it already includes PyTorch, Transformers, the MCP/Flask stack, and
other runtime libraries. Extras are additive profiles and sometimes repeat
base dependencies; they do not make the base install smaller. Extras are
defined in `pyproject.toml` under `[project.optional-dependencies]`. Common profiles
include `full`, `mcp`, `mcp-p2p`, `webnn`, `llama_cpp`, `analysis`, and
`testing`. There is no `cuda` / `openvino` / `rocm` packaging extra; see the
[installation guide](installation.md) for the complete table and CUDA wheel
selection.

## 2. Verify the base package (offline)

```bash
python - <<'PY'
import ipfs_accelerate_py

print("runtime __version__:", ipfs_accelerate_py.__version__)
print("package import: ok")
PY
```

This is an import/version check only. It deliberately does not call
`get_instance()`: constructing the process coordinator is side-effecting and
may initialize storage, caches, configuration, daemons, or external-resource
integrations.

`ipfs_accelerate_py.__version__` is the runtime export and matches packaging
metadata (`0.0.45` on this tree). See
[installation version sources](installation.md#version-sources).

The base import is intentionally defensive. It may expose an availability flag
or a fallback object when optional dependencies are missing.

CLI help without starting services:

```bash
ipfs-accelerate --help
ipfs-accelerate models --help
```

## 3. Inspect hardware and providers

The process-level capability report constructs the coordinator. It is a
runtime initialization step, not an offline or side-effect-free probe; run it
only when local writes, daemon discovery/startup, and configured external
integration checks are acceptable.

```bash
python - <<'PY'
from ipfs_accelerate_py import get_instance

report = get_instance().get_capabilities(detail=True)
print("hardware:", report.get("hardware", {}))
print("task_types:", report.get("task_types", []))
print("models:", report.get("models", []))
print("mcp:", report.get("mcp", {}))
PY
```

For NVIDIA systems, verify the PyTorch CUDA build rather than relying on a
static hardware label:

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

The [installation guide](installation.md) covers stable CUDA wheels and CUDA 13
nightly wheels for newer NVIDIA systems.

## 4. Run a model operation (optional)

The main compatibility API is `ipfs_accelerate_py` / `get_instance()`, not a
legacy `IPFSAccelerator` name. With the Transformers integration installed
(for example via the `full` or `textgen` extra, plus a suitable PyTorch
build), the following performs runtime initialization and may exercise the
same storage/configuration/integration setup described above:

```python
from ipfs_accelerate_py import ipfs_accelerate_py

accelerator = ipfs_accelerate_py(
    resources={"transformers": {}},
    metadata={"role": "inference"},
)
result = accelerator.run_model(
    "bert-base-uncased",
    {"input_ids": [[101, 2023, 2003, 102]]},
    model_type="text_embedding",
    device="cpu",
)
print(result)
```

The model, tokenizer, task type, provider, and device must agree. Prefer
`get_capabilities(detail=True)` before selecting a non-CPU device or a
network-backed provider. Model downloads and credentials are separate from the
base install.

Runtime discovery with the process singleton (side-effecting construction):

```python
from ipfs_accelerate_py import get_instance

accelerator = get_instance()
print(accelerator.get_capabilities(detail=True))
```

This call may initialize storage/cache/configuration, probe external
integrations, or start configured daemons. Use package import and CLI `--help`
for a cold verification instead.

## 5. Start MCP (optional)

Install the MCP extra and start the canonical server through the product CLI:

```bash
python -m pip install -e ".[mcp]"   # or: pip install "ipfs-accelerate-py[mcp]"
ipfs-accelerate mcp start --host 127.0.0.1 --port 9000
```

Verify it from another terminal:

```bash
ipfs-accelerate mcp status --host 127.0.0.1 --port 9000
```

Use `--dashboard`, `--open-browser`, or `--disable-autoscaler` only when those
behaviors are wanted. Read [MCP setup](../MCP_SETUP_GUIDE.md) before binding to
an external interface.

## 6. Start the agent supervisor (optional)

The agent supervisor is a maintainer/operator control plane, not a requirement
for inference. It turns an objective heap into evidence-backed tasks and can
launch isolated implementation lanes. Use the
[Agent Supervisor Guide](../AGENT_SUPERVISOR_GUIDE.md) for the complete workflow.

Lightweight discovery surfaces (no daemon start required for help/list):

```bash
ipfs-accelerate agent --help
python -m ipfs_accelerate_py.agent_supervisor.todo_daemon list
```

## Troubleshooting

| Symptom | First check |
| --- | --- |
| Import fails | Confirm the virtual environment and run `python -m pip show ipfs-accelerate-py`. |
| CUDA is unavailable | Check `torch.cuda.is_available()` and install a matching CUDA wheel. |
| Model provider is unavailable | Install the relevant extra and inspect `get_capabilities(detail=True)`. |
| MCP status is unhealthy | Run `ipfs-accelerate mcp status` and inspect the server log. |
| Supervisor appears idle | Check its heartbeat/status artifact; a live PID alone is not progress. |
| Version strings disagree | Cite packaging vs `__version__` sources; see the installation guide. |

## Further reading

- [Installation](installation.md)
- [Quick start](../QUICKSTART.md)
- [API overview](../../api/overview.md)
- [Architecture overview](../../architecture/overview.md)
- [System context](../../architecture/SYSTEM_CONTEXT.md)
- [Hardware guide](../hardware/overview.md)
- [Testing guide](../../development/testing.md)
- [Examples](../../../examples/README.md)
