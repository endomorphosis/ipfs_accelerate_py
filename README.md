# IPFS Accelerate Python

IPFS Accelerate Python is a capability-driven Python framework for model
inference, hardware and provider routing, content-addressed storage, MCP
services, optional P2P workflows, and validated agent-supervisor automation.
The core package is useful on CPU; CUDA, browser runtimes, IPFS, P2P, remote
providers, and formal-assurance tools are installed and enabled separately.

[![PyPI](https://img.shields.io/pypi/v/ipfs-accelerate-py.svg)](https://pypi.org/project/ipfs-accelerate-py/)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Documentation](https://img.shields.io/badge/docs-index-brightgreen.svg)](docs/INDEX.md)

## Contents

- [What it provides](#what-it-provides)
- [Installation](#installation)
- [Quick start](#quick-start)
- [MCP server](#mcp-server)
- [Architecture](#architecture)
- [Hardware and providers](#hardware-and-providers)
- [Models and inference](#models-and-inference)
- [IPFS and P2P](#ipfs-and-p2p)
- [Performance and scaling](#performance-and-scaling)
- [Testing](#testing)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)

## What it provides

The repository brings several related but deliberately separate surfaces
together:

- **Python API and model management** for endpoint registration, model loading,
  hardware discovery, storage, and inference dispatch.
- **Unified product CLI**, installed as `ipfs-accelerate`, for MCP, GitHub,
  model-management, and provider-dependent AI commands.
- **Direct AI CLI**, installed as `ipfs_accelerate`, backed by a separate
  inference parser. Its command surface must be inspected independently.
- **Canonical MCP runtime** in `ipfs_accelerate_py.mcp_server`, with the older
  `ipfs_accelerate_py.mcp` package retained as a compatibility facade.
- **Optional routers and services** for LLMs, embeddings, HuggingFace model
  serving, WebNN/WebGPU, IPFS, and P2P TaskQueue workflows.
- **Agent supervisor control plane** for objective analysis, evidence-backed
  task generation, isolated implementation lanes, deterministic validation, and
  merge/proof receipts.

Importing the base package does not imply that every optional provider,
executable, credential, daemon, model, or hardware backend is available. The
runtime capability report is the authoritative first check.

## Installation

### Published package

```bash
python -m pip install -U pip
python -m pip install ipfs-accelerate-py
```

### Development checkout

```bash
git clone https://github.com/endomorphosis/ipfs_accelerate_py.git
cd ipfs_accelerate_py
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

### Feature profiles

The available extras are defined by `pyproject.toml`. Common profiles are:

| Extra | Intended use |
| --- | --- |
| `minimal` | Small runtime dependency set. |
| `dev` | Local development and focused tests. |
| `full` | Transformers, PyTorch, model server, and model-manager integrations. |
| `mcp` | MCP server and GitHub integration dependencies. |
| `mcp-p2p` / `libp2p` | Optional TaskQueue and libp2p networking. |
| `webnn` | Browser/WebNN/WebGPU integration. |
| `llama_cpp` | llama.cpp server support. |
| `analysis` / `monitoring` | Analysis and host/NVIDIA monitoring helpers. |
| `testing` | Broader optional test dependencies. |
| `all` | Aggregate application dependencies; native P2P remains explicit. |

Install only what the workload needs:

```bash
python -m pip install "ipfs-accelerate-py[mcp]"
python -m pip install "ipfs-accelerate-py[full]"
```

See the [installation guide](docs/guides/getting-started/installation.md) for
the complete extra list, source builds, IPFS/P2P notes, and troubleshooting.

### CUDA and PyTorch

The NVIDIA driver, PyTorch wheel, model kernels, and device architecture must
agree. A visible GPU or `nvidia-smi` result alone does not prove that the model
path is CUDA-backed.

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

For CUDA 12.4, use the repository requirements file when appropriate:

```bash
python -m pip install --upgrade --force-reinstall \
  -r install/requirements_torch_cu124.txt
```

For newer NVIDIA GB10/DGX Spark-class systems that require a CUDA 13 nightly
build:

```bash
./scripts/install_torch_cuda_cu130_nightly.sh
```

Record the driver, PyTorch version, CUDA version, model, device, and smoke-test
result in performance reports. See the [hardware guide](docs/guides/hardware/overview.md).

## Quick start

### Discover the runtime

```bash
python - <<'PY'
import ipfs_accelerate_py
from ipfs_accelerate_py import get_instance

print("version:", ipfs_accelerate_py.__version__)
print(get_instance().get_capabilities(detail=True))
PY
```

`get_capabilities(detail=True)` returns a JSON-friendly report of discovered
hardware, task types, registered models/endpoints, and optional integrations.
It reports availability; it does not download missing dependencies or models.

### Python API

The package-level compatibility API is the safest starting point:

```python
from ipfs_accelerate_py import get_instance

accelerator = get_instance()
print(accelerator.get_capabilities(detail=True))
```

With the Transformers integration installed, run a model through the current
accelerator class:

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

The model, tokenizer, task type, provider, and device must agree. Use the
capability report before selecting a non-CPU device. The [API overview](docs/api/overview.md)
documents endpoint-oriented operations and optional exports.

### Unified CLI

The supported product entry point is the hyphenated command:

```bash
ipfs-accelerate --help
ipfs-accelerate models --help
ipfs-accelerate models list
ipfs-accelerate models search "embedding"
ipfs-accelerate text --ai-help
```

The current top-level groups are `mcp`, `github`, `copilot`, `copilot-sdk`,
`text`, `audio`, `vision`, `multimodal`, `specialized`, and `models`. Older
examples using generic `inference`, `hardware`, `workflow`, `network`, or
`queue` groups are not current commands.

The underscore command is a separate parser:

```bash
ipfs_accelerate --help
```

Do not mix flags between the two scripts; use each command's own help output.

### Examples

The [examples README](examples/README.md) lists the files present in this
checkout and the extras they may require. A small deterministic starting point
is:

```bash
python examples/demonstration_example.py
python examples/llm_router_example.py
```

Model downloads, credentials, browser runtimes, Docker, IPFS, and P2P services
are separate capabilities and should be enabled deliberately.

## MCP server

The canonical MCP runtime is `ipfs_accelerate_py.mcp_server`. The
`ipfs_accelerate_py.mcp` package remains a compatibility facade for older
integrations. Inspect the runtime manifest and optional dependency state before
assuming a tool or transport is present.

### Product startup

```bash
python -m pip install "ipfs-accelerate-py[mcp]"
ipfs-accelerate mcp start --host 127.0.0.1 --port 9000
ipfs-accelerate mcp status --host 127.0.0.1 --port 9000
```

Keep development servers on localhost. Remote exposure requires authentication,
TLS, firewall policy, resource limits, and process supervision.

### Other entry points

| Entry point | Use |
| --- | --- |
| `ipfs-accelerate mcp start` | Product startup and dashboard options. |
| `python -m ipfs_accelerate_py.mcp.cli` | Direct transport/process control and optional P2P worker services. |
| `python -m ipfs_accelerate_py.mcp_server.fastapi_service` | Standalone FastAPI hosting. |
| `from ipfs_accelerate_py.mcp_server import create_server` | Programmatic construction. |

The [MCP setup guide](docs/guides/MCP_SETUP_GUIDE.md), [dashboard guide](docs/MCP_DASHBOARD_GUIDE.md),
and [canonical server README](ipfs_accelerate_py/mcp_server/README.md) are the
maintained operational references. MCP++ conformance, policy, artifact, and
cutover records live in [mcpplusplus](mcpplusplus/README.md).

### MCP++ trust boundary

MCP tools may expose inference, storage, GitHub, Docker, P2P, or operational
actions depending on the installed capabilities and policy. A registered tool
is not automatically authorized for an untrusted caller. Keep secrets out of
prompts and client configuration, validate tool arguments, and place remote
access behind an authenticated deployment boundary.

## Architecture

The runtime is layered so that local inference remains useful without the
distributed or control-plane integrations:

```text
Application and examples
        |
Python API / unified CLI / MCP server
        |
Inference, model, embedding, voice, and P2P services
        |
Hardware and provider adapters
        |
IPFS, local storage, caches, and external services
```

The optional agent supervisor is a separate maintainer/operator control plane:

```text
Objective heap (intent)
        |
AST, dependency, retrieval, GraphRAG, and proof-gap analysis
        |
Canonical todo and bundle projections
        |
Leases, resource admission, conflicts, and isolated worktrees
        |
LLM proposals -> deterministic validation -> merge/completion receipts
```

The important design rule is that provider and LLM output remains proposal
material. Deterministic scanners, type/contract checks, validators, and
authoritative prover receipts control admission, merge, and completion.

Read the [architecture overview](docs/architecture/overview.md) for runtime
layers and the [agent-supervisor architecture](docs/architecture/AGENT_SUPERVISOR_ARCHITECTURE.md)
for the control-plane trust model.

## Hardware and providers

Hardware support is adapter-driven and discovered at runtime. The following
families are supported when their upstream runtime, model path, and package
extra are available:

| Family | Typical runtime | Notes |
| --- | --- | --- |
| CPU | PyTorch/Transformers or local providers | Baseline for deterministic smoke tests. |
| NVIDIA CUDA | Matching CUDA PyTorch build | Verify with `torch.cuda.is_available()` and a model operation. |
| AMD ROCm | ROCm PyTorch distribution | CUDA wheels are not interchangeable with ROCm. |
| Apple MPS | Apple PyTorch/MPS runtime | Supported only on compatible Apple hardware. |
| Intel/OpenVINO | OpenVINO runtime | Provider and model support vary by task. |
| WebNN/WebGPU | Browser plus `webnn` extra | Separate browser runtime; validate browser flags and drivers. |
| Qualcomm and other adapters | Vendor runtime | Availability is environment-specific. |

The framework can select a provider automatically, but automatic selection is
not a guarantee that the preferred backend is healthy. For troubleshooting,
compare the package capability report with the service/worker environment and
run a small real operation on the selected device.

## Models and inference

HuggingFace-compatible models and custom providers are supported through the
installed model/inference integrations. There is no fixed model-count promise:
the usable set depends on the provider, model task, tokenizer, weights, device
memory, and optional dependencies.

The main model-management paths include:

- `ModelManager` and `get_default_model_manager()` for model registry/cache
  operations;
- `ipfs_accelerate_py(...).run_model` on the compatibility class for
  application inference;
- `generate_text` and `embed_text`/`embed_texts` for router-based provider
  selection; and
- the optional [HF model server](docs/features/hf-model-server/README.md) for
  HTTP serving with health/readiness and OpenAI-shaped request routes.

For embeddings, the router can resolve configured OpenRouter, xAI, Meta AI,
Gemini CLI, HuggingFace, backend-manager, or registered custom providers. See
the [embeddings router guide](docs/EMBEDDINGS_ROUTER.md) and [LLM router guide](docs/LLM_ROUTER.md).

## IPFS and P2P

IPFS and P2P are optional. Local inference does not require a Kubo daemon or a
peer network.

### IPFS backend selection

The IPFS backend router can select among available backends:

1. `ipfs_kit_py`, when installed and configured;
2. local HuggingFace/cache storage; and
3. a Kubo CLI backend, when the external daemon and command are available.

This is a fallback strategy, not a claim that all three are installed:

```python
from ipfs_accelerate_py import ipfs_backend_router

cid = ipfs_backend_router.add_bytes(b"hello", pin=True)
print(cid)
print(ipfs_backend_router.cat(cid))
```

See the [IPFS backend router](docs/IPFS_BACKEND_ROUTER.md) and [IPFS feature guide](docs/features/ipfs/IPFS.md).

### P2P TaskQueue and workflow services

Install and enable P2P explicitly:

```bash
python -m pip install "ipfs-accelerate-py[mcp-p2p]"
python -m ipfs_accelerate_py.mcp.cli --help
```

P2P operation also requires peer identity, queue configuration, reachable
ports, firewall/NAT policy, bounded payloads, and an explicit failure strategy.
The current product CLI does not register a generic `ipfs-accelerate p2p start`
or `p2p-workflow` command; use the [P2P guide](docs/guides/p2p/README.md) and
the live module help.

### GitHub API cache

The GitHub cache is a separate optional integration. Local cache behavior,
encryption, credentials, and P2P sharing are independently configurable; P2P
sharing is opt-in and disabled by default. See the [GitHub cache guide](docs/features/github-cache/overview.md).

## Performance and scaling

Performance depends on model, tokenizer, sequence length, batch shape,
precision, device, provider, warm-up state, cache state, concurrency, and
network services. The repository does not promise one benchmark number across
hosts.

Useful optimization steps:

1. Discover capabilities and confirm the actual device/provider.
2. Separate first-run downloads and model loading from steady-state inference.
3. Use batching and bounded concurrency appropriate to the model and device.
4. Use a local response/model cache for repeated deterministic work.
5. Measure memory, queue depth, latency, throughput, and shutdown behavior.
6. Increase process or lane parallelism only when the provider and memory budget
   can absorb duplicated model state.

For the agent supervisor, `--max-lanes` is an admission limit rather than a
promise to start that many processes. Dependencies, conflicting paths, leases,
CPU/memory/disk budgets, provider capacity, and validation gates determine
actual parallel width. See the [deployment guide](docs/guides/deployment/README.md)
and [hardware guide](docs/guides/hardware/overview.md).

## Testing

Install local development dependencies:

```bash
python -m pip install -e ".[dev]"
```

Start with deterministic focused contracts:

```bash
python -m pytest test/test_unified_cli_integration.py -q
python -m pytest test/test_hf_model_server_endpoint_contract.py -q
python -m pytest test/api/test_serving_readiness_contracts.py -q
python -m pytest test/api/test_agent_supervisor_objective_graph.py -q
python -m pytest test/api/test_agent_supervisor_todo_daemon_port.py -q
```

For Docker, router, or auto-healing changes, use the focused tests named in
their guides. Full repository coverage may require optional dependencies,
external services, credentials, browser runtimes, or a Docker daemon. A test
that imports successfully is not proof that CUDA, IPFS, P2P, an LLM provider,
or a theorem prover is healthy.

The [testing guide](docs/development/testing.md) explains the test layout,
optional boundaries, hardware checks, and supervisor smoke procedure.

## Documentation

### Start here

| Guide | Purpose |
| --- | --- |
| [Getting started](docs/guides/getting-started/README.md) | Install, discover capabilities, run a first operation. |
| [Quick start](docs/guides/QUICKSTART.md) | Short CLI, Python, MCP, and supervisor path. |
| [Installation](docs/guides/getting-started/installation.md) | Extras, CUDA, IPFS/P2P, and build details. |
| [API overview](docs/api/overview.md) | Current public Python exports. |
| [Architecture overview](docs/architecture/overview.md) | Runtime layers and integration boundaries. |
| [Hardware guide](docs/guides/hardware/overview.md) | Capability discovery and device tuning. |
| [Testing](docs/development/testing.md) | Focused tests and optional validation. |
| [FAQ](docs/guides/troubleshooting/faq.md) | Common installation and runtime questions. |

### Specialized references

- [LLM Router](docs/LLM_ROUTER.md) and [Embeddings Router](docs/EMBEDDINGS_ROUTER.md)
- [MCP setup](docs/guides/MCP_SETUP_GUIDE.md) and [MCP dashboard](docs/MCP_DASHBOARD_GUIDE.md)
- [HF model server](docs/features/hf-model-server/README.md)
- [IPFS integration](docs/features/ipfs/IPFS.md) and [P2P workflows](docs/guides/p2p/README.md)
- [GitHub integration](docs/guides/github/README.md) and [GitHub cache](docs/features/github-cache/overview.md)
- [Agent Supervisor Guide](docs/guides/AGENT_SUPERVISOR_GUIDE.md)
- [Current documentation state](docs/development/DOCUMENTATION_CURRENT_STATE.md)

The [documentation index](docs/INDEX.md) is the canonical navigation page.
Files under `docs/archive/`, `docs/development_history/`, `docs/summaries/`,
and dated phase/status directories preserve project context and are not current
API contracts.

## Contributing

Contributions are welcome. A focused contribution usually follows this shape:

1. Read [CONTRIBUTING.md](CONTRIBUTING.md) and the relevant architecture guide.
2. Confirm the live source boundary, optional dependencies, and existing tests.
3. Make a small change with deterministic tests and bounded artifacts.
4. Run the focused checks and record environment-specific failures clearly.
5. Open a pull request with the behavior change, validation command, and
   capability assumptions.

Maintainer extension points include evidence-producing scanners, prover
capability registries, objective/backlog projections, router/provider adapters,
typed lease/resource policies, and versioned artifact stores. LLM output stays
in the proposal tier until deterministic checks accept it.

## License

IPFS Accelerate Python is licensed under the GNU Affero General Public License,
version 3 or later. See [LICENSE](LICENSE) and the [AGPL FAQ](https://www.gnu.org/licenses/gpl-faq.html).

## Acknowledgments

The project builds on the work of the HuggingFace, PyTorch, FastAPI, IPFS,
libp2p, and broader open-source communities.

- [HuggingFace Transformers](https://huggingface.co/transformers/)
- [PyTorch](https://pytorch.org/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [IPFS](https://ipfs.io/)
- [Project contributors](https://github.com/endomorphosis/ipfs_accelerate_py/graphs/contributors)

For release history, security reporting, and contribution policy, see
[CHANGELOG.md](CHANGELOG.md), [SECURITY.md](SECURITY.md), and
[CONTRIBUTING.md](CONTRIBUTING.md).
