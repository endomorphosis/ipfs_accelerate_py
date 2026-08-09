# IPFS Accelerate Python

> **Capability-driven model inference, hardware and provider routing, content-addressed storage, MCP services, optional P2P workflows, and validated agent-supervisor automation**

[![PyPI](https://img.shields.io/pypi/v/ipfs-accelerate-py.svg)](https://pypi.org/project/ipfs-accelerate-py/)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Documentation](https://img.shields.io/badge/docs-index-brightgreen.svg)](docs/INDEX.md)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [MCP Server](#mcp-server)
- [Architecture](#architecture)
- [Hardware and Providers](#hardware-and-providers)
- [Models and Inference](#models-and-inference)
- [IPFS and P2P](#ipfs-and-p2p)
- [Performance and Scaling](#performance-and-scaling)
- [Testing](#testing)
- [Documentation](#documentation)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## 🚀 Overview

**IPFS Accelerate Python** is a capability-driven Python framework for model
inference, hardware and provider routing, content-addressed storage, MCP
services, optional P2P workflows, and validated agent-supervisor automation.

The core package is useful on CPU. CUDA, browser runtimes, IPFS, P2P, remote
providers, and formal-assurance tools are installed and enabled separately.
Importing the base package does **not** imply that every optional provider,
executable, credential, daemon, model, or hardware backend is available — the
runtime capability report is the authoritative first check.

### ⚡ Key highlights

- **Hardware adapters** — CPU, CUDA, ROCm, OpenVINO, Apple MPS, WebNN, WebGPU, Qualcomm, and other adapters when their upstream runtime and extras are present
- **Distributed by design** — optional IPFS content addressing, P2P TaskQueue workflows, and multi-backend storage routing
- **HuggingFace-compatible paths** — model registry, cache, and inference integrations when Transformers/PyTorch extras are installed
- **Canonical MCP++ server** — unified `ipfs_accelerate_py.mcp_server` runtime, with `ipfs_accelerate_py.mcp` retained as a compatibility facade
- **Browser-native paths** — WebNN / WebGPU via the optional `webnn` extra
- **Agent supervisor control plane** — objective analysis, evidence-backed tasks, isolated implementation lanes, deterministic validation, and merge/proof receipts
- **Capability-first** — discover what is actually available before selecting a non-CPU device or remote service

---

## 📦 Installation

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
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

### Feature profiles

Extras are defined in `pyproject.toml`. Install only what the workload needs:

| Extra | Intended use |
| --- | --- |
| `minimal` | Small runtime dependency set |
| `dev` | Local development and focused tests |
| `full` | Transformers, PyTorch, model server, and model-manager integrations |
| `mcp` | MCP server and GitHub integration dependencies |
| `mcp-p2p` / `libp2p` | Optional TaskQueue and libp2p networking |
| `webnn` | Browser / WebNN / WebGPU integration |
| `llama_cpp` | llama.cpp server support |
| `analysis` / `monitoring` | Analysis and host/NVIDIA monitoring helpers |
| `testing` | Broader optional test dependencies |
| `all` | Aggregate application dependencies; native P2P remains explicit |

```bash
python -m pip install "ipfs-accelerate-py[mcp]"
python -m pip install "ipfs-accelerate-py[full]"
```

See the [installation guide](docs/guides/getting-started/installation.md) for
the complete extra list, source builds, IPFS/P2P notes, and troubleshooting.

### NVIDIA CUDA (PyTorch)

By default, pip may install a CPU-only PyTorch wheel from PyPI because CUDA
wheels are published on PyTorch's own indexes. A visible GPU or `nvidia-smi`
result alone does not prove that the model path is CUDA-backed.

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

For NVIDIA **GB10 / DGX Spark**-class systems that need CUDA 13 nightly wheels:

```bash
./scripts/install_torch_cuda_cu130_nightly.sh
```

Record the driver, PyTorch version, CUDA version, model, device, and smoke-test
result in performance reports. See the [hardware guide](docs/guides/hardware/overview.md).

📚 **Detailed instructions**: [Installation Guide](docs/guides/getting-started/installation.md) · [Troubleshooting FAQ](docs/guides/troubleshooting/faq.md) · [Getting Started](docs/guides/getting-started/README.md)

---

## 🎯 Quick Start

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

# MCP product startup (requires mcp extra)
ipfs-accelerate mcp start --host 127.0.0.1 --port 9000
ipfs-accelerate mcp status --host 127.0.0.1 --port 9000
```

Current top-level groups include `mcp`, `github`, `copilot`, `copilot-sdk`,
`text`, `audio`, `vision`, `multimodal`, `specialized`, and `models`. Older
examples that assume generic `inference`, `hardware`, `workflow`, `network`, or
`queue` groups are **not** current product commands — use each command's own
`--help`.

The underscore command is a **separate** parser:

```bash
ipfs_accelerate --help
```

Do not mix flags between the two scripts.

### Direct MCP / P2P process control

```bash
# Canonical FastAPI MCP service
python -m ipfs_accelerate_py.mcp_server.fastapi_service

# Direct MCP CLI with optional P2P TaskQueue worker services
python -m ipfs_accelerate_py.mcp.cli --host 0.0.0.0 --port 9000

# Remote machine: MCP + worker + libp2p TaskQueue service
python -m ipfs_accelerate_py.mcp.cli \
  --host 0.0.0.0 --port 9000 \
  --p2p-task-worker --p2p-service --p2p-listen-port 9710 \
  --p2p-queue ~/.cache/ipfs_datasets_py/task_queue.duckdb

# Optional (off-host clients): public IP embedded in the announced multiaddr
export IPFS_DATASETS_PY_TASK_P2P_PUBLIC_IP="YOUR_PUBLIC_IP"
```

By default the libp2p TaskQueue service writes an announce file under your XDG
cache dir (`~/.cache/ipfs_accelerate_py/task_p2p_announce.json`). Clients that
can read that path do not need a remote multiaddr. Otherwise the process prints
`multiaddr=...` for:

```bash
export IPFS_DATASETS_PY_TASK_P2P_REMOTE_MULTIADDR="/ip4/.../tcp/9710/p2p/..."
```

Disable announce-file writes with `IPFS_ACCELERATE_PY_TASK_P2P_ANNOUNCE_FILE=0`
(or the `IPFS_DATASETS_PY_*` alias). This mode requires `ipfs_datasets_py` (and
typically `ipfs_datasets_py[p2p]`) on the remote machine.

### Real-world examples

| Example | Description | Notes |
|---------|-------------|-------|
| [demonstration_example.py](examples/demonstration_example.py) | Deterministic starting point | Low dependency surface |
| [basic_usage.py](examples/basic_usage.py) | Core package usage | Beginner |
| [llm_router_example.py](examples/llm_router_example.py) | LLM router providers | May need provider credentials |
| [embeddings_router_example.py](examples/embeddings_router_example.py) | Embeddings router | Optional providers |
| [demo_webnn_webgpu.py](examples/demo_webnn_webgpu.py) | Browser acceleration path | `webnn` extra / browser runtime |
| [mcp_integration_example.py](examples/mcp_integration_example.py) | MCP integration | `mcp` extra |

📖 **More examples**: [examples/](examples/) · [examples README](examples/README.md) · [Quick Start Guide](docs/guides/QUICKSTART.md)

---

## 🧠 MCP Server

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

### Current entry points

| Entry point | Best for | Notes |
|------------|----------|-------|
| `ipfs-accelerate mcp start` | Product startup | Dashboard options and server management |
| `python -m ipfs_accelerate_py.mcp.cli` | Direct process control | Optional TaskQueue / libp2p worker services |
| `python -m ipfs_accelerate_py.mcp_server.fastapi_service` | Standalone HTTP/FastAPI | Reads `IPFS_MCP_*` env vars; mounts MCP at `/mcp` by default |
| `from ipfs_accelerate_py.mcp_server import create_server` | Programmatic embedding | Stable import for the canonical runtime |

### MCP++ profiles and control plane

The unified runtime advertises additive MCP++ profiles such as:

- `mcp++/profile-a-idl`
- `mcp++/profile-b-cid-artifacts`
- `mcp++/profile-c-ucan`
- `mcp++/profile-d-temporal-policy`
- `mcp++/profile-e-mcp-p2p`

Operational features include meta-tools (`tools_list_*`, `tools_dispatch`,
runtime metrics), migrated categories (`ipfs`, `workflow`, `p2p`), UCAN and
policy hooks, observability bridges, and transport coverage for process helpers,
FastAPI mounting, and MCP+p2p negotiation. Treat registered tools as **not**
automatically authorized for untrusted callers.

### Cutover and rollback controls

These environment controls remain available for validation and operational
rollback:

- `IPFS_MCP_FORCE_LEGACY_ROLLBACK=1` — keep the compatibility facade on the legacy wrapper
- `IPFS_MCP_UNIFIED_CUTOVER_DRY_RUN=1` — validate unified startup while keeping legacy runtime behavior active
- `IPFS_MCP_ENABLE_UNIFIED_BRIDGE=1` — explicitly request the unified bridge on compatibility-facade paths

### Recommended documentation

- [Canonical MCP server README](ipfs_accelerate_py/mcp_server/README.md)
- [MCP setup guide](docs/guides/MCP_SETUP_GUIDE.md)
- [MCP dashboard guide](docs/MCP_DASHBOARD_GUIDE.md)
- [MCP++ package docs](mcpplusplus/README.md)
- [MCP Cutover Checklist](mcpplusplus/CUTOVER_CHECKLIST.md)

### MCP++ trust boundary

MCP tools may expose inference, storage, GitHub, Docker, P2P, or operational
actions depending on installed capabilities and policy. Keep secrets out of
prompts and client configuration, validate tool arguments, and place remote
access behind an authenticated deployment boundary.

---

## 🏗️ Architecture

The runtime is layered so local inference remains useful without distributed or
control-plane integrations:

```text
┌─────────────────────────────────────────────────────────┐
│              Application / examples / CLIs              │
│     Python API • unified CLI • MCP server • dashboards  │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────┴────────────────────────────────────┐
│     Inference, model, embedding, voice, and P2P services │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────┴────────────────────────────────────┐
│           Hardware and provider adapters                │
│     CPU • CUDA • ROCm • MPS • OpenVINO • WebNN/WebGPU   │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────┴────────────────────────────────────┐
│     IPFS, local storage, caches, and external services  │
└─────────────────────────────────────────────────────────┘
```

### Agent supervisor (optional control plane)

The optional agent supervisor is a **separate maintainer/operator** control plane:

```text
Objective heap (intent)
        │
AST, dependency, retrieval, GraphRAG, and proof-gap analysis
        │
Canonical todo and bundle projections
        │
Leases, resource admission, conflicts, and isolated worktrees
        │
LLM proposals → deterministic validation → merge/completion receipts
```

Provider and LLM output remains **proposal material**. Deterministic scanners,
type/contract checks, validators, and authoritative prover receipts control
admission, merge, and completion.

📐 **Detailed architecture**: [docs/architecture/overview.md](docs/architecture/overview.md) · [Agent-supervisor architecture](docs/architecture/AGENT_SUPERVISOR_ARCHITECTURE.md) · [Agent Supervisor Guide](docs/guides/AGENT_SUPERVISOR_GUIDE.md)

---

## 🔧 Hardware and Providers

Hardware support is adapter-driven and discovered at runtime. Families below are
supported when their upstream runtime, model path, and package extra are available:

| Family | Typical runtime | Notes |
| --- | --- | --- |
| **CPU** (x86/ARM) | PyTorch/Transformers or local providers | Baseline for deterministic smoke tests |
| **NVIDIA CUDA** | Matching CUDA PyTorch build | Verify with `torch.cuda.is_available()` **and** a model operation |
| **AMD ROCm** | ROCm PyTorch distribution | CUDA wheels are not interchangeable with ROCm |
| **Apple MPS** | Apple PyTorch/MPS | Compatible Apple silicon only |
| **Intel OpenVINO** | OpenVINO runtime | Provider and model support vary by task |
| **WebNN / WebGPU** | Browser + `webnn` extra | Separate browser runtime; validate flags and drivers |
| **Qualcomm / other** | Vendor runtime | Environment-specific |

Automatic provider selection is a convenience, not a guarantee that the preferred
backend is healthy. Compare the package capability report with the service/worker
environment and run a small real operation on the selected device.

⚙️ **Hardware guides**: [Hardware overview](docs/guides/hardware/overview.md)

---

## 🤖 Models and Inference

HuggingFace-compatible models and custom providers are supported through the
installed model/inference integrations. There is **no fixed model-count promise**:
the usable set depends on provider, task, tokenizer, weights, device memory, and
optional dependencies.

Main model-management paths include:

- `ModelManager` / `get_default_model_manager()` for registry and cache operations
- `ipfs_accelerate_py(...).run_model` on the compatibility class for application inference
- `generate_text` and `embed_text` / `embed_texts` for router-based provider selection
- Optional [HF model server](docs/features/hf-model-server/README.md) for HTTP serving with health/readiness and OpenAI-shaped routes

For embeddings, the router can resolve configured OpenRouter, xAI, Meta AI,
Gemini CLI, HuggingFace, backend-manager, or registered custom providers. See
the [embeddings router](docs/EMBEDDINGS_ROUTER.md) and [LLM router](docs/LLM_ROUTER.md).

### Goose CLI (LLM router)

**Goose CLI** (`goose_cli` / `goose`) is a peer of Codex and Copilot for text
generation. Ordinary router chat is tool-free and discovery of Goose is opt-in;
lazy install is explicit and pinned; agent execution and P2P remote use require
separate authorization gates. Operator environment variables, managed install
paths, readiness versus liveness, P2P no-replay policy, offline tests, and the
`IPFS_ACCELERATE_GOOSE_LIVE` smoke gate are documented under
[Goose CLI in the LLM router guide](docs/LLM_ROUTER.md#goose-cli).

🤖 **API and serving**: [API overview](docs/api/overview.md) · [HF model server](docs/features/hf-model-server/README.md)

---

## 🌐 IPFS and P2P

IPFS and P2P are **optional**. Local inference does not require a Kubo daemon or
a peer network.

### Why IPFS?

When enabled, IPFS integration provides content-addressed distribution and
multi-backend storage routing:

- **Content addressing** — content IDs for models and artifacts
- **Pluggable backends** — `ipfs_kit_py`, local HuggingFace/cache storage, and Kubo CLI as a fallback chain
- **Optional P2P workflows** — TaskQueue services, peer identity, and bounded payloads when explicitly installed

### IPFS backend selection

The IPFS backend router can select among available backends:

1. `ipfs_kit_py`, when installed and configured
2. Local HuggingFace/cache storage
3. Kubo CLI, when the external daemon and command are available

This is a fallback strategy, not a claim that all three are installed:

```python
from ipfs_accelerate_py import ipfs_backend_router

cid = ipfs_backend_router.add_bytes(b"hello", pin=True)
print(cid)
print(ipfs_backend_router.cat(cid))
```

Configuration examples:

```bash
# Prefer ipfs_kit_py when available
export ENABLE_IPFS_KIT=true

# Use HF cache only (good for CI)
export IPFS_BACKEND=hf_cache

# Force Kubo CLI
export IPFS_BACKEND=kubo
```

📚 **Full documentation**: [IPFS Backend Router](docs/IPFS_BACKEND_ROUTER.md) · [IPFS feature guide](docs/features/ipfs/IPFS.md)

### P2P TaskQueue and workflow services

Install and enable P2P explicitly:

```bash
python -m pip install "ipfs-accelerate-py[mcp-p2p]"
python -m ipfs_accelerate_py.mcp.cli --help
```

P2P operation also requires peer identity, queue configuration, reachable ports,
firewall/NAT policy, bounded payloads, and an explicit failure strategy. The
current product CLI does **not** register a generic `ipfs-accelerate p2p start`
command; use the [P2P guide](docs/guides/p2p/README.md) and live module help.

### GitHub API cache

The GitHub cache is a separate optional integration. Local cache behavior,
encryption, credentials, and P2P sharing are independently configurable; P2P
sharing is opt-in and disabled by default. See the [GitHub cache guide](docs/features/github-cache/overview.md)
and [GitHub integration](docs/guides/github/README.md).

---

## ⚡ Performance and Scaling

Performance depends on model, tokenizer, sequence length, batch shape,
precision, device, provider, warm-up state, cache state, concurrency, and
network services. This repository does **not** promise one benchmark number
across hosts.

Useful optimization steps:

1. Discover capabilities and confirm the actual device/provider.
2. Separate first-run downloads and model loading from steady-state inference.
3. Use batching and bounded concurrency appropriate to the model and device.
4. Use a local response/model cache for repeated deterministic work.
5. Measure memory, queue depth, latency, throughput, and shutdown behavior.
6. Increase process or lane parallelism only when the provider and memory budget
   can absorb duplicated model state.

For the agent supervisor, `--max-lanes` is an **admission limit**, not a promise
to start that many processes. Dependencies, conflicting paths, leases,
CPU/memory/disk budgets, provider capacity, and validation gates determine
actual parallel width.

📊 **Guides**: [Deployment](docs/guides/deployment/README.md) · [Hardware](docs/guides/hardware/overview.md)

---

## 🧪 Testing

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

Goose CLI contracts stay offline by default (fakes only):

```bash
python -m pytest \
  test/test_llm_router_goose.py \
  test/test_goose_cli_endpoint.py \
  test/test_goose_p2p_policy.py -q
```

Opt-in live Goose smoke requires `IPFS_ACCELERATE_GOOSE_LIVE=1` and a configured
binary/provider; see [Goose CLI](docs/LLM_ROUTER.md#goose-cli).

Full repository coverage may require optional dependencies, external services,
credentials, browser runtimes, or a Docker daemon. A test that imports
successfully is not proof that CUDA, IPFS, P2P, an LLM provider, or a theorem
prover is healthy.

🧪 **Testing guide**: [docs/development/testing.md](docs/development/testing.md)

---

## 📚 Documentation

### Start here

| Guide | Purpose |
| --- | --- |
| [Getting started](docs/guides/getting-started/README.md) | Install, discover capabilities, first operation |
| [Quick start](docs/guides/QUICKSTART.md) | Short CLI, Python, MCP, and supervisor path |
| [Installation](docs/guides/getting-started/installation.md) | Extras, CUDA, IPFS/P2P, build details |
| [API overview](docs/api/overview.md) | Current public Python exports |
| [Architecture overview](docs/architecture/overview.md) | Runtime layers and integration boundaries |
| [Hardware guide](docs/guides/hardware/overview.md) | Capability discovery and device tuning |
| [Testing](docs/development/testing.md) | Focused tests and optional validation |
| [FAQ](docs/guides/troubleshooting/faq.md) | Common installation and runtime questions |

### Specialized references

| Topic | Resources |
| --- | --- |
| **LLM / embeddings** | [LLM Router](docs/LLM_ROUTER.md) (Codex, Copilot, Grok, **Goose CLI**) · [Embeddings Router](docs/EMBEDDINGS_ROUTER.md) |
| **MCP** | [MCP setup](docs/guides/MCP_SETUP_GUIDE.md) · [Dashboard](docs/MCP_DASHBOARD_GUIDE.md) · [Server README](ipfs_accelerate_py/mcp_server/README.md) · [mcpplusplus](mcpplusplus/README.md) |
| **Serving** | [HF model server](docs/features/hf-model-server/README.md) |
| **IPFS & P2P** | [IPFS](docs/features/ipfs/IPFS.md) · [Backend router](docs/IPFS_BACKEND_ROUTER.md) · [P2P](docs/guides/p2p/README.md) |
| **GitHub** | [GitHub integration](docs/guides/github/README.md) · [GitHub cache](docs/features/github-cache/overview.md) · [Autoscaler](docs/architecture/AUTOSCALER.md) |
| **Agent supervisor** | [Operator guide](docs/guides/AGENT_SUPERVISOR_GUIDE.md) · [Architecture](docs/architecture/AGENT_SUPERVISOR_ARCHITECTURE.md) |
| **Browser** | [WebNN/WebGPU](docs/features/webnn-webgpu/WEBNN_WEBGPU_README.md) |
| **Docs state** | [Current documentation state](docs/development/DOCUMENTATION_CURRENT_STATE.md) |

The [documentation index](docs/INDEX.md) is the canonical navigation page.
Files under `docs/archive/`, `docs/development_history/`, `docs/summaries/`,
and dated phase/status directories preserve project context and are **not**
current API contracts.

📋 **Documentation Hub**: [docs/](docs/) · [Full Index](docs/INDEX.md)

---

## 🔧 Troubleshooting

### Common issues

| Issue | First checks |
| --- | --- |
| **Import / missing extra** | Install the matching profile (`[mcp]`, `[full]`, …) and re-check imports |
| **CUDA not used** | Match driver ↔ PyTorch CUDA build; verify `torch.cuda.is_available()` **and** a real model op |
| **Slow first run** | Separate download/load time from steady-state inference; warm caches deliberately |
| **Memory pressure** | Reduce batch size / concurrency; confirm device and precision |
| **MCP / remote access** | Keep localhost for dev; require auth, TLS, and firewall policy for exposure |
| **P2P / announce** | Confirm extras, queue path, ports, and announce-file or multiaddr configuration |

### Quick checks

```bash
# Version and capability report
python -c "import ipfs_accelerate_py; from ipfs_accelerate_py import get_instance; print(ipfs_accelerate_py.__version__); print(get_instance().get_capabilities(detail=True))"

# Product CLI surface
ipfs-accelerate --help
ipfs-accelerate models --help
ipfs-accelerate mcp --help
```

🆘 **Get help**: [Installation troubleshooting](docs/guides/troubleshooting/INSTALLATION_TROUBLESHOOTING_GUIDE.md) · [FAQ](docs/guides/troubleshooting/faq.md) · [GitHub Issues](https://github.com/endomorphosis/ipfs_accelerate_py/issues)

---

## 🤝 Contributing

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

- 💬 [GitHub Discussions](https://github.com/endomorphosis/ipfs_accelerate_py/discussions)
- 🐛 [Issue Tracker](https://github.com/endomorphosis/ipfs_accelerate_py/issues)
- 🔐 [Security Policy](SECURITY.md)

📖 **Full guides**: [CONTRIBUTING.md](CONTRIBUTING.md) · [SECURITY.md](SECURITY.md)

---

## 📄 License

IPFS Accelerate Python is licensed under the **GNU Affero General Public License
v3.0 or later (AGPLv3+)**.

- Free to use, modify, and distribute
- Commercial use allowed
- Network use requires source disclosure under AGPL terms

📋 **Details**: [LICENSE](LICENSE) · [AGPL FAQ](https://www.gnu.org/licenses/gpl-faq.html)

---

## 🙏 Acknowledgments

Built with the work of the HuggingFace, PyTorch, FastAPI, IPFS, libp2p, and
broader open-source communities:

- [HuggingFace Transformers](https://huggingface.co/transformers/)
- [PyTorch](https://pytorch.org/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [IPFS](https://ipfs.io/)
- [Project contributors](https://github.com/endomorphosis/ipfs_accelerate_py/graphs/contributors)

### Project information

- 📋 [Changelog](CHANGELOG.md)
- 🔐 [Security Policy](SECURITY.md)
- 🤝 [Contributing Guide](CONTRIBUTING.md)
- 📄 [License](LICENSE)

---

## 🌟 Show Your Support

If you find this project useful:

- ⭐ Star this repository on GitHub
- 📢 Share with your network
- 🐛 Report issues to help improve it
- 💡 Contribute features or fixes
- 📝 Write about your experience

---

<div align="center">

**Maintained by [Benjamin Barber](https://github.com/endomorphosis) and [contributors](https://github.com/endomorphosis/ipfs_accelerate_py/graphs/contributors)**

[Homepage](https://github.com/endomorphosis/ipfs_accelerate_py) ·
[Documentation](docs/) ·
[Issues](https://github.com/endomorphosis/ipfs_accelerate_py/issues) ·
[Discussions](https://github.com/endomorphosis/ipfs_accelerate_py/discussions)

</div>
