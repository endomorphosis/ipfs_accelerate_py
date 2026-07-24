# API Overview

This is the current high-level API reference for the Python package. Detailed
behavior remains defined by the source modules and their tests; optional
providers may be unavailable even when the base package imports successfully.

## Package metadata and imports

```python
import ipfs_accelerate_py

print(ipfs_accelerate_py.__version__)
```

The package version is exposed as `ipfs_accelerate_py.__version__`. The public
root exports include the following core names:

| Export | Role |
| --- | --- |
| `ipfs_accelerate_py` | Main compatibility-oriented accelerator class. |
| `get_instance` | Process-wide accelerator instance with optional dependency injection. |
| `ModelManager` | Lazily loaded model-management facade. |
| `get_default_model_manager` | Obtain the default model manager. |
| `generate_text` | LLM-router text generation, when the router dependencies are available. |
| `embed_text`, `embed_texts` | Embeddings-router helpers, when configured. |
| `P2PWorkflowScheduler` | Optional distributed workflow scheduler. |
| `get_storage`, `IPFSKitStorage` | Optional storage integration. |
| `accelerate_with_browser`, `get_accelerator` | Optional WebNN/WebGPU integration. |

Use `hasattr()` or the corresponding availability flag before depending on an
optional integration. For example:

```python
from ipfs_accelerate_py import get_instance

accelerator = get_instance()
print(accelerator.get_capabilities())
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
from ipfs_accelerate_py import generate_text

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

The supported CLI is `ipfs-accelerate`. Its current top-level groups are:

```bash
ipfs-accelerate --help
ipfs-accelerate mcp --help
ipfs-accelerate text --help
ipfs-accelerate audio --help
ipfs-accelerate vision --help
ipfs-accelerate multimodal --help
ipfs-accelerate specialized --help
ipfs-accelerate models --help
ipfs-accelerate github --help
```

Examples:

```bash
ipfs-accelerate mcp start --host 0.0.0.0 --port 9000
ipfs-accelerate mcp status --host 127.0.0.1 --port 9000
ipfs-accelerate models list
ipfs-accelerate models search "embedding"
ipfs-accelerate text --ai-help
```

The CLI does not currently expose the historical `inference`, `hardware`,
`workflow`, or `network` top-level groups shown in older guides. Use the MCP
server, Python APIs, or the feature-specific tools for those capabilities.

## MCP server

The canonical runtime is `ipfs_accelerate_py.mcp_server`. The compatibility
facade is `ipfs_accelerate_py.mcp`.

```python
from ipfs_accelerate_py.mcp_server import create_server

server = create_server()
```

For command-line operation:

```bash
python -m ipfs_accelerate_py.mcp.cli --host 0.0.0.0 --port 9000
python -m ipfs_accelerate_py.mcp_server.fastapi_service
```

See the [MCP setup guide](../guides/MCP_SETUP_GUIDE.md) and the
[canonical server README](../../ipfs_accelerate_py/mcp_server/README.md) for
transport, policy, P2P, and deployment details.

## Agent supervisor APIs

The supervisor is a separate maintainer/operator API. Its most stable contracts
are grouped by concern:

| Concern | Modules |
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

python -m ipfs_accelerate_py.agent_supervisor.prover_matrix_registry \
  --output data/agent_supervisor/prover_matrix.json --no-self-tests
```

Prover discovery without a passing bounded fixture is not proof capability.
Likewise, CUDA availability should be verified with an actual model operation
before a production workload is admitted.
