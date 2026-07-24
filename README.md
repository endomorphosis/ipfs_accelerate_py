# IPFS Accelerate Python

IPFS Accelerate Python is a Python framework for model inference, hardware and
provider routing, IPFS-backed storage, MCP services, P2P workflows, and an
optional objective-driven agent supervisor. Optional integrations are detected
at runtime; installing the base package does not imply that every backend or
external service is available.

## Install

For a published package:

```bash
python -m pip install ipfs-accelerate-py
```

For development from this checkout:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

Feature-scoped extras include `full`, `mcp`, `mcp-p2p`, `webnn`, `llama_cpp`,
`analysis`, and `testing`. Install only the extras required by the workload.

### CUDA verification

The PyTorch wheel, NVIDIA driver, and model must be compatible. Check the
installed build before starting a GPU workload:

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

The [installation guide](docs/guides/getting-started/installation.md) contains
the stable CUDA and CUDA 13 nightly installation paths.

## Verify the runtime

```bash
python - <<'PY'
import ipfs_accelerate_py
from ipfs_accelerate_py import get_instance

print("version:", ipfs_accelerate_py.__version__)
print(get_instance().get_capabilities(detail=True))
PY
```

`get_capabilities(detail=True)` is a JSON-friendly discovery surface. It may
report an optional feature as unavailable when its package, executable,
credential, or service is missing.

## Python API

The principal compatibility class is `ipfs_accelerate_py`; the package does
not expose an `IPFSAccelerator` class.

```python
from ipfs_accelerate_py import get_instance

accelerator = get_instance()
print(accelerator.get_capabilities())
```

When the Transformers provider is installed, the class can load and run a model
through `run_model`:

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

The package also exports optional router, model-manager, storage, browser, P2P,
and voice helpers. See the [API overview](docs/api/overview.md) for the current
exports and availability rules.

## CLI

Inspect the parser rather than relying on historical command examples:

```bash
ipfs-accelerate --help
ipfs-accelerate models --help
ipfs-accelerate models list
ipfs-accelerate models search "embedding"
ipfs-accelerate text --ai-help
```

Current top-level groups are `mcp`, `github`, `copilot`, `copilot-sdk`, `text`,
`audio`, `vision`, `multimodal`, `specialized`, and `models`.

## MCP server

The canonical MCP runtime is `ipfs_accelerate_py.mcp_server`; the
`ipfs_accelerate_py.mcp` package is a compatibility facade.

```bash
ipfs-accelerate mcp start --host 127.0.0.1 --port 9000
ipfs-accelerate mcp status --host 127.0.0.1 --port 9000
```

Direct module entry points are available for transport-specific operation:

```bash
python -m ipfs_accelerate_py.mcp.cli --host 127.0.0.1 --port 9000
python -m ipfs_accelerate_py.mcp_server.fastapi_service
```

Read the [MCP setup guide](docs/guides/MCP_SETUP_GUIDE.md) before exposing a
server beyond localhost. The [MCP++ records](mcpplusplus/README.md) contain
conformance, cutover, and migration details.

## Agent supervisor

`ipfs_accelerate_py.agent_supervisor` is an optional maintainer/operator control
plane. It turns an objective heap into evidence-backed tasks, groups related
work into bundle shards, schedules isolated implementation lanes, and records
validation/proof receipts. LLMs remain proposal sources; deterministic checks
and authoritative receipts control admission, merge, and completion.

Installed tools include:

```text
ipfs-accelerate-agent-objective-daemon
ipfs-accelerate-agent-backlog-refinery
ipfs-accelerate-agent-bundle-supervisor
ipfs-accelerate-agent-implementation-daemon
ipfs-accelerate-agent-implementation-supervisor
ipfs-accelerate-agent-artifact-query
ipfs-accelerate-agent-merge-resolver
```

Start with the [Agent Supervisor Guide](docs/guides/AGENT_SUPERVISOR_GUIDE.md),
then read the [architecture and assurance model](docs/architecture/AGENT_SUPERVISOR_ARCHITECTURE.md).

## Architecture and integrations

The maintained architecture is summarized in
[docs/architecture/overview.md](docs/architecture/overview.md). Major optional
boundaries include:

- Transformers/model providers and hardware adapters;
- IPFS and local content-addressed storage;
- MCP server transports and policy/tool registries;
- P2P TaskQueue and workflow scheduling;
- LLM and embeddings routers; and
- the objective/lease/prover-based agent supervisor.

Capability discovery and conformance probes are preferred over static feature
claims. For example, prover discovery is not a proof, and a CUDA driver is not
the same as a working CUDA PyTorch operation.

## Tests

Run focused deterministic checks first:

```bash
python -m pytest test/test_unified_cli_integration.py -q
python -m pytest test/api/test_agent_supervisor_objective_graph.py -q
python -m pytest test/api/test_agent_supervisor_todo_daemon_port.py -q
```

Run the full suite when the optional dependencies and services required by the
selected tests are installed:

```bash
python -m pytest
```

See [Testing](docs/development/testing.md) for the current test layout and
hardware/provider guidance.

## Documentation

- [Documentation index](docs/INDEX.md)
- [Getting started](docs/guides/getting-started/README.md)
- [Quick start](docs/guides/QUICKSTART.md)
- [API overview](docs/api/overview.md)
- [Architecture overview](docs/architecture/overview.md)
- [Hardware guide](docs/guides/hardware/overview.md)
- [LLM Router](docs/LLM_ROUTER.md)
- [Contributing](CONTRIBUTING.md)
- [Security policy](SECURITY.md)
- [Changelog](CHANGELOG.md)

Documentation under `docs/archive/` and `docs/development_history/` is retained
for historical context. It may describe an earlier commit, test count, score,
or planned phase and is not a current API guarantee.

## License

IPFS Accelerate Python is licensed under the GNU Affero General Public License,
version 3 or later. See [LICENSE](LICENSE).
