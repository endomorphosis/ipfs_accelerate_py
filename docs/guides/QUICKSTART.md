# Quick Start

**Status:** Current
**Audience:** Developers who need a short path from install to first supported
Python and CLI operations
**Scope:** Editable/published install, capability inspection, Python entrypoints,
current `ipfs-accelerate` CLI groups, MCP, optional agent supervisor and Goose
**Non-goals:** Exhaustive API reference; inventing a single package version when
sources disagree
**Source anchors:** `pyproject.toml`, `ipfs_accelerate_py/cli_entry.py` →
`cli.py`, `ipfs_accelerate_py/__init__.py`, `ipfs_accelerate_py/mcp_server/`

This guide uses the current `ipfs-accelerate` CLI and Python exports. Optional
model, CUDA, IPFS, MCP, and P2P integrations require their corresponding
dependencies and services.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

For a published package, replace the last command with:

```bash
python -m pip install ipfs-accelerate-py
```

Extras live under `[project.optional-dependencies]` in `pyproject.toml`. See
[installation](getting-started/installation.md) for the full table (`mcp`,
`full`, `webnn`, `mcp-p2p`, and others). There is no packaging extra named
`cuda`, `openvino`, or `rocm`.

Verify the package offline:

```bash
python - <<'PY'
import ipfs_accelerate_py
from ipfs_accelerate_py import get_instance

print("runtime __version__:", ipfs_accelerate_py.__version__)
print(get_instance().get_capabilities())
PY
ipfs-accelerate --help
```

Packaging metadata currently declares `0.0.45` while
`ipfs_accelerate_py.__version__` is `0.4.0`. Report both sources when needed;
do not guess which declaration should win. Details:
[version sources](getting-started/installation.md#version-sources-code-owned-disagreement).

## Inspect capabilities

```bash
python - <<'PY'
from ipfs_accelerate_py import get_instance

capabilities = get_instance().get_capabilities(detail=True)
print(capabilities)
PY
```

This reports the runtime capabilities discovered on the current host. It does
not claim that every optional backend is installed, and it does not download
missing dependencies or models.

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

Process singleton (safe first call):

```python
from ipfs_accelerate_py import get_instance

accelerator = get_instance()
print(accelerator.get_capabilities())
```

When the Transformers integration is installed, a model can be loaded and run
through the main accelerator class:

```python
from ipfs_accelerate_py import ipfs_accelerate_py

accelerator = ipfs_accelerate_py(
    resources={"transformers": {}},
    metadata={"role": "inference"},
)
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

Optional LLM router helper (requires a configured provider; not offline):

```python
from ipfs_accelerate_py import generate_text

# Only when a provider is installed and credentials/env are set.
# print(generate_text("Say hello in one sentence.", max_tokens=32))
```

## CLI

The supported product entry point is the hyphenated command. Top-level groups
registered by the current parser include:

- `agent` — supervisor control contracts
- `mcp` — MCP server lifecycle
- `github`, `copilot`, `copilot-sdk` — GitHub / Copilot integrations
- `text`, `audio`, `vision`, `multimodal`, `specialized` — modality helpers
- `models` — model manager and search

Discover from the command itself:

```bash
ipfs-accelerate --help
ipfs-accelerate models --help
ipfs-accelerate models list
ipfs-accelerate models search "embedding"
ipfs-accelerate text --ai-help
ipfs-accelerate agent --help
ipfs-accelerate mcp --help
```

Older examples using `ipfs-accelerate inference ...`, `hardware ...`,
`workflow ...`, `network ...`, or `queue ...` do not match the current parser.

The underscore script is a **separate** entry point:

```bash
ipfs_accelerate --help
```

Do not mix flags between `ipfs-accelerate` and `ipfs_accelerate`.

## MCP server

Start the canonical MCP runtime through the product CLI (requires the `mcp`
extra and related deps):

```bash
ipfs-accelerate mcp start --host 127.0.0.1 --port 9000
```

In another terminal:

```bash
ipfs-accelerate mcp status --host 127.0.0.1 --port 9000
```

Direct module paths are useful for embedding or transport-specific tests:

```bash
python -m ipfs_accelerate_py.mcp.cli --host 127.0.0.1 --port 9000
python -m ipfs_accelerate_py.mcp_server.fastapi_service
```

The canonical runtime lives under `ipfs_accelerate_py.mcp_server`. The
`ipfs_accelerate_py.mcp` package remains a compatibility facade. Read the
[MCP setup guide](MCP_SETUP_GUIDE.md) before exposing a server beyond localhost.

## Agent supervisor

The objective-driven supervisor is optional and is used to generate and run
maintainer work. Start with the
[Agent Supervisor Guide](AGENT_SUPERVISOR_GUIDE.md) for objective heaps, bundle
lanes, leases, validation, and Leanstral.

Surfaces that do not require starting a full control loop:

```bash
ipfs-accelerate agent --help
python -m ipfs_accelerate_py.agent_supervisor.todo_daemon list
```

## Goose CLI (optional LLM provider)

Goose is a peer of Codex/Copilot on the LLM router. Safe chat is the default;
generic discovery does **not** install Goose and stays off until
`IPFS_ACCELERATE_GOOSE_DISCOVERY=1`. Prefer explicit provider selection.

```python
from ipfs_accelerate_py import generate_text

# Requires a goose binary (PATH or IPFS_ACCELERATE_GOOSE_PATH) and backend
# credentials in the environment — never hard-code secrets.
print(generate_text(
    "Explain content addressing in one sentence.",
    provider="goose_cli",
    model_name="muse-spark-1.1",
    goose_provider="openai",
    max_tokens=128,
))
```

Authorized agent runs, managed install paths, `GOOSE_PATH_ROOT` isolation, P2P
enable gates, offline tests, the `IPFS_ACCELERATE_GOOSE_LIVE` smoke gate, and
rollback steps are documented in the
[LLM router Goose CLI section](../LLM_ROUTER.md#goose-cli).

Offline contract tests (default; no live provider):

```bash
python -m pytest \
  test/test_llm_router_goose.py \
  test/test_goose_cli_endpoint.py \
  test/test_goose_p2p_policy.py -q
```

## Next steps

- [Getting started](getting-started/README.md)
- [Installation](getting-started/installation.md)
- [LLM router and Goose CLI](../LLM_ROUTER.md)
- [API overview](../api/overview.md)
- [Architecture overview](../architecture/overview.md)
- [Hardware guide](hardware/overview.md)
- [Testing guide](../development/testing.md)
- [Examples](../../examples/README.md)
