# Quick Start

This guide uses the current `ipfs-accelerate` CLI and Python exports. Optional
model, CUDA, IPFS, MCP, and P2P integrations require their corresponding
dependencies and services.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

For a published package, replace the last command with:

```bash
python -m pip install ipfs-accelerate-py
```

Verify the package and its version:

```bash
python - <<'PY'
import ipfs_accelerate_py
print(ipfs_accelerate_py.__version__)
PY
```

## Inspect capabilities

```bash
python - <<'PY'
from ipfs_accelerate_py import get_instance

capabilities = get_instance().get_capabilities(detail=True)
print(capabilities)
PY
```

This reports the runtime capabilities discovered on the current host. It does
not claim that every optional backend is installed.

For a CUDA installation, validate the PyTorch build separately:

```bash
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("torch_cuda", torch.version.cuda)
PY
```

See [installation](getting-started/installation.md) for CUDA wheel selection.

## Python API

```python
from ipfs_accelerate_py import get_instance

accelerator = get_instance()
print(accelerator.get_capabilities())
```

When the Transformers integration is installed, a model can be loaded and run
through the main accelerator class:

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

Use a device that is actually available. The `run_model()` path is provider
and model dependent; `get_capabilities(detail=True)` is the safe first check.

## CLI

The CLI groups are discoverable from the command itself:

```bash
ipfs-accelerate --help
ipfs-accelerate models --help
ipfs-accelerate models list
ipfs-accelerate models search "embedding"
ipfs-accelerate text --ai-help
```

The current CLI uses `text`, `audio`, `vision`, `multimodal`, `specialized`,
`models`, `github`, and `mcp` groups. Older examples using
`ipfs-accelerate inference ...`, `hardware ...`, or `workflow ...` do not match
the current parser.

## MCP server

Start the canonical MCP runtime through the product CLI:

```bash
ipfs-accelerate mcp start --host 127.0.0.1 --port 9000
```

In another terminal:

```bash
ipfs-accelerate mcp status --host 127.0.0.1 --port 9000
```

The direct module paths are useful for embedding or transport-specific tests:

```bash
python -m ipfs_accelerate_py.mcp.cli --host 127.0.0.1 --port 9000
python -m ipfs_accelerate_py.mcp_server.fastapi_service
```

Read the [MCP setup guide](MCP_SETUP_GUIDE.md) before exposing a server beyond
localhost.

## Agent supervisor

The objective-driven supervisor is optional and is used to generate and run
maintainer work. Start with the [Agent Supervisor Guide](AGENT_SUPERVISOR_GUIDE.md)
for objective heaps, bundle lanes, leases, validation, and Leanstral.

The package dispatcher can show its registered daemon families without starting
anything:

```bash
python -m ipfs_accelerate_py.agent_supervisor.todo_daemon list
```

## Next steps

- [Getting started](getting-started/README.md)
- [API overview](../api/overview.md)
- [Architecture overview](../architecture/overview.md)
- [Hardware guide](hardware/overview.md)
- [Testing guide](../development/testing.md)
- [Examples](../../examples/README.md)
