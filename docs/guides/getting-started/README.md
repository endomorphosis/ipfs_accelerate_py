# Getting Started

This guide gets a source checkout or an installed package to a verified Python
import, a capability report, and an optional MCP server. The framework has
optional backends, so the first useful question is which capabilities are
available on the current host.

## 1. Install

For a source checkout:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

For a published installation:

```bash
python -m pip install ipfs-accelerate-py
```

Install an extra only when you need it. Common profiles include `full`, `mcp`,
`mcp-p2p`, `webnn`, `llama_cpp`, `analysis`, and `testing`.

## 2. Verify the base package

```bash
python - <<'PY'
import ipfs_accelerate_py
from ipfs_accelerate_py import get_instance

print("version:", ipfs_accelerate_py.__version__)
print("task types:", get_instance().get_capabilities()["task_types"])
PY
```

The base import is intentionally defensive. It may expose an availability flag
or a fallback object when optional dependencies are missing.

## 3. Inspect hardware and providers

```bash
python - <<'PY'
from ipfs_accelerate_py import get_instance

report = get_instance().get_capabilities(detail=True)
print("hardware:", report.get("hardware", {}))
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

## 4. Run a model operation

The main compatibility API is `ipfs_accelerate_py`, not `IPFSAccelerator`.
With the Transformers integration installed:

```python
from ipfs_accelerate_py import ipfs_accelerate_py

accelerator = ipfs_accelerate_py({}, {})
result = accelerator.run_model(
    "bert-base-uncased",
    {"input_ids": [[101, 2023, 2003, 102]]},
    model_type="text_generation",
    device="cpu",
)
print(result)
```

The model, tokenizer, task type, and device must agree. For a discovery-first
workflow, call `get_capabilities(detail=True)` before loading a model.

## 5. Start MCP

Install the MCP extra and start the canonical server:

```bash
python -m pip install -e ".[mcp]"
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
launch isolated implementation lanes. Use the [Agent Supervisor Guide](../AGENT_SUPERVISOR_GUIDE.md)
for the complete workflow.

```bash
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

## Further reading

- [Quick start](../QUICKSTART.md)
- [API overview](../../api/overview.md)
- [Architecture overview](../../architecture/overview.md)
- [Hardware guide](../hardware/overview.md)
- [Testing guide](../../development/testing.md)
- [Examples](../../../examples/README.md)
