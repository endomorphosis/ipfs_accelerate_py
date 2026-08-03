# API Overview

This is the current high-level API reference for the Python package. Detailed
behavior remains defined by the source modules and their tests. Optional
providers, routers, and integrations may appear in the package export surface
even when they are not operational on the current host: **import success is not
runtime health**.

Prefer public root imports (`from ipfs_accelerate_py import …`) for product
code. Deeper package paths are valid for maintainers and agent-supervisor work,
but they are not a guarantee of stability or installed extras.

## Package metadata and imports

```python
import ipfs_accelerate_py

print(ipfs_accelerate_py.__version__)
```

The package version is exposed as `ipfs_accelerate_py.__version__`. Core and
optional symbols are attached at import time when their dependencies resolve.
Many optional symbols are still present as stubs or `None` when unavailable;
check the matching `*_available` flag (or call the capability report) before
relying on them.

### Core public surface

| Export | Role | Availability |
| --- | --- | --- |
| `ipfs_accelerate_py` | Main compatibility-oriented accelerator class. | Raises `NotImplementedError` when core is disabled or missing. |
| `get_instance` | Process-wide accelerator instance with optional dependency injection. | Same as core. |
| `ModelManager` | Lazily loaded model-management facade. | Resolved on first use; may raise if model-manager deps are missing. |
| `get_default_model_manager` | Obtain the default model manager. | Same lazy boundary as `ModelManager`. |
| `cli_main` | Programmatic entry to the unified CLI (`ipfs-accelerate`). | When core is enabled. |

### Optional / router surface

These names may appear on the package even when providers, credentials, or
optional extras are missing. Treat the availability flags as import-time hints
only; operational readiness still needs `get_capabilities(detail=True)` (and
provider-specific checks).

| Export | Role | Flag / note |
| --- | --- | --- |
| `generate_text`, `get_llm_provider`, `register_llm_provider` | LLM router. | `llm_router_available` |
| `embed_text`, `embed_texts`, `embed_texts_batched` | Embeddings router. | `embeddings_router_available` |
| `generate_multimodal` | Multimodal router. | `multimodal_router_available` |
| `text_to_speech`, `speech_to_text`, `process_voice_turn` | Voice / TTS-STT router. | `voice_router_available` / `tts_router_available` |
| `P2PWorkflowScheduler`, `P2PTask` | Optional distributed workflow scheduler. | May be `None` if the module does not import. |
| `get_storage`, `IPFSKitStorage` | Optional storage integration. | May be `None` without IPFS kit deps. |
| `InferenceBackendManager`, `get_backend_manager` | Inference backend manager. | `inference_backend_manager_available` |
| `accelerate_with_browser`, `get_accelerator` | Optional WebNN/WebGPU integration. | `webnn_webgpu_available` |

```python
from ipfs_accelerate_py import (
    get_instance,
    llm_router_available,
    embeddings_router_available,
    webnn_webgpu_available,
)

accelerator = get_instance()
print(accelerator.get_capabilities(detail=True))
print({
    "llm_router_available": llm_router_available,
    "embeddings_router_available": embeddings_router_available,
    "webnn_webgpu_available": webnn_webgpu_available,
})
```

An `*_available` flag of `True` means the package could import that subsystem.
It does **not** mean a provider, model, GPU, daemon, or credential is ready.

Set `IPFS_ACCEL_SKIP_CORE=1` to skip heavy core imports (useful for lightweight
tooling). Set `IPFS_ACCEL_IMPORT_EAGER=1` to force eager model-manager imports.

## Console scripts

Declared in packaging (`pyproject.toml` / `setup.py` entry points):

| Script | Entry | Notes |
| --- | --- | --- |
| `ipfs-accelerate` | `ipfs_accelerate_py.cli_entry:main` | Supported unified product CLI. |
| `ipfs_accelerate` | `ipfs_accelerate_py.ai_inference_cli:main` | Separate underscore CLI; different command surface. |
| `ipfs-accelerate-agent-*` | agent-supervisor daemons | Operator engines; see the [Agent Supervisor Guide](../guides/AGENT_SUPERVISOR_GUIDE.md). |
| `ipfs-accelerate-llama-cpp-serve` | `ipfs_accelerate_py.utils.llama_cpp:main` | Optional local serve helper. |

When the console script is not on `PATH` (editable checkouts, incomplete
installs), use the module form:

```bash
python -m ipfs_accelerate_py.cli --help
```

## Core accelerator

The compatibility class is constructed with resource and metadata mappings. It
also accepts optional injected `deps`, `ipfs_kit`, `ipfs_datasets`, and storage
objects.

```python
from ipfs_accelerate_py import ipfs_accelerate_py

accelerator = ipfs_accelerate_py(
    resources={"transformers": {}},
    metadata={"role": "inference"},
)
```

Important methods include:

```python
accelerator.get_capabilities(detail=True)  # JSON-friendly capability summary
accelerator.get_mcp_manifest(detail=True)  # MCP tools/resources/prompts
accelerator.run_model(                         # load and run a model
    model_name="bert-base-uncased",
    inputs={"input_ids": [[101, 2023, 2003, 102]]},
    model_type="text_generation",
    device="cpu",
)
```

`run_model()` requires the configured Transformers/model provider and accepts
model-specific keyword arguments. It converts list inputs to tensors and adds
an all-ones attention mask when `input_ids` is supplied without one. For
endpoint-oriented applications, use `add_endpoint()`, `rm_endpoint()`,
`get_endpoints()`, `choose_endpoint()`, and `infer()` as defined in
`ipfs_accelerate_py/ipfs_accelerate.py`.

`get_capabilities(detail=True)` is the preferred health/discovery surface. It
reports task types, registered models/endpoints, hardware information when the
detector is available, and the MCP manifest without returning callables.

## LLM router

The router is a separate provider boundary and is also used by optional agent
planning features.

```python
from ipfs_accelerate_py import generate_text, llm_router_available

if not llm_router_available:
    raise SystemExit("LLM router did not import; install router extras/deps")

answer = generate_text(
    "Summarize the role of a content identifier in one sentence.",
    provider="openrouter",       # omit to use configured provider order
    model_name="openai/gpt-4o-mini",
    max_tokens=128,
    temperature=0.1,
)
```

Provider availability depends on credentials and installed adapters. Response
caching, provider registration, and shared dependency injection are documented
in [LLM Router](../LLM_ROUTER.md).

## Unified CLI

The supported CLI is `ipfs-accelerate` (module form:
`python -m ipfs_accelerate_py.cli`). Its **registered** top-level groups are:

```bash
ipfs-accelerate --help
ipfs-accelerate agent --help
ipfs-accelerate mcp --help
ipfs-accelerate github --help
ipfs-accelerate copilot --help
ipfs-accelerate copilot-sdk --help
ipfs-accelerate text --help
ipfs-accelerate audio --help
ipfs-accelerate vision --help
ipfs-accelerate multimodal --help
ipfs-accelerate specialized --help
ipfs-accelerate models --help
```

Examples that resolve against the current parser:

```bash
ipfs-accelerate agent capabilities --help
ipfs-accelerate mcp start --host 127.0.0.1 --port 9000
ipfs-accelerate mcp status --host 127.0.0.1 --port 9000
ipfs-accelerate models list
ipfs-accelerate models search "embedding"
ipfs-accelerate text --ai-help
```

The parser epilog still mentions historical examples such as
`ipfs-accelerate inference …`, `queue …`, and `network …`. Those strings are
**not** registered command groups; the live `choices=` set rejects them. Use
the groups listed above, the MCP server, or Python APIs for those capabilities.
Full CLI reference: [CLI guide](../guides/cli/README_CLI.md).

## MCP server

The canonical runtime is `ipfs_accelerate_py.mcp_server`. The compatibility
facade is `ipfs_accelerate_py.mcp`.

```python
from ipfs_accelerate_py.mcp_server import create_server

server = create_server()
```

For command-line operation:

```bash
ipfs-accelerate mcp start --host 127.0.0.1 --port 9000
python -m ipfs_accelerate_py.mcp.cli --host 127.0.0.1 --port 9000
python -m ipfs_accelerate_py.mcp_server.fastapi_service
```

See the [MCP setup guide](../guides/MCP_SETUP_GUIDE.md) and the
[canonical server README](../../ipfs_accelerate_py/mcp_server/README.md) for
transport, policy, P2P, and deployment details.

## Agent supervisor APIs

The supervisor is a separate maintainer/operator API. Prefer the typed product
CLI for control operations:

```bash
ipfs-accelerate agent --help
ipfs-accelerate agent capabilities --help
ipfs-accelerate agent status --help
```

Python and package modules remain available for embedding and automation. The
most stable contracts are grouped by concern (module names under
`ipfs_accelerate_py.agent_supervisor` and its domain packages such as
`control`, `proof`, `objectives`, and `runtime`):

| Concern | Modules / packages |
| --- | --- |
| Objective and task identity | `objective_graph`, `objective_tracker`, `task_identity`, `taskboard_store` |
| Analysis and retrieval | `analysis_ast_index`, `analysis_cache`, `analysis_contracts`, `analysis_retrieval`, `code_evidence_graph`, `todo_vector_index` |
| Scheduling and isolation | `lease_coordination`, `resource_scheduler`, `conflict_graph`, `bundle_supervisor` |
| Proposal and execution | `todo_daemon`, `implementation_daemon_runner`, `implementation_supervisor_runner`, `task_proposal_router` |
| Formal planning and proof | `formal_plan_compiler`, `formal_plan_validator`, `formal_plan_conformance`, `multi_prover_router`, `prover_conformance`, `proof_carrying_planner` |
| Leanstral lifecycle | `leanstral_goal_development`, `leanstral_goal_lifecycle`, `leanstral_proof_provider`, `leanstral_goal_benchmark` |
| Persistence and receipts | `artifact_store`, `prover_evidence_store`, `proof_attestation`, `proof_metrics` |

The [Agent Supervisor Guide](../guides/AGENT_SUPERVISOR_GUIDE.md) documents
commands, artifact paths, and the proposal/assurance trust boundary. The
architecture documents are the reference for extension behavior.

## Optional capabilities

The following checks help distinguish importability from operational readiness:

```bash
python - <<'PY'
import torch
print({
    "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "torch_cuda": torch.version.cuda,
})
PY

# Canonical module path (under agent_supervisor.proof).
# The short alias without ".proof." is not a reliable -m entry point.
python -m ipfs_accelerate_py.agent_supervisor.proof.prover_matrix_registry \
  --output data/agent_supervisor/prover_matrix.json --no-self-tests
```

Prover discovery without a passing bounded fixture is not proof capability.
Likewise, CUDA availability should be verified with an actual model operation
before a production workload is admitted.
