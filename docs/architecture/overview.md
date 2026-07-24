# Architecture Overview

This page describes the maintained runtime boundaries in the repository. It is
intentionally capability-oriented: optional integrations are discovered at
runtime, and a component is not considered available merely because its
Python module can be imported.

## System layers

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

The package also contains a separate agent-supervisor control plane:

```text
Objective heap (intent)
        |
Objective graph + AST/GraphRAG analysis + evidence artifacts
        |
Todo and bundle projections
        |
Lease/resource admission and isolated implementation lanes
        |
LLM proposals -> deterministic validation -> merge/completion receipts
```

The agent supervisor is not on the inference hot path. It is an optional
maintainer/operator subsystem for turning a long-range objective into bounded,
validated implementation work. See the [Agent Supervisor Architecture](AGENT_SUPERVISOR_ARCHITECTURE.md)
and [Agent Supervisor Guide](../guides/AGENT_SUPERVISOR_GUIDE.md).

## Python package boundary

The package root exposes a compatibility-oriented API. The principal
constructor is `ipfs_accelerate_py.ipfs_accelerate_py`; `get_instance()` provides
a process-wide instance. Optional imports expose model management, embeddings,
LLM routing, multimodal, voice, P2P, and storage helpers when their dependencies
are present.

```python
from ipfs_accelerate_py import get_instance

accelerator = get_instance()
capabilities = accelerator.get_capabilities(detail=True)
print(capabilities["task_types"])
```

The constructor accepts resource and metadata mappings plus optional dependency
injections. It does not silently install all optional dependencies. Deployment
profiles should install the extras they need and validate the resulting
capabilities before serving traffic.

## Inference and routing

`ipfs_accelerate_py.ipfs_accelerate_py` coordinates endpoint registration,
hardware selection, model providers, and request dispatch. The LLM router is a
separate provider boundary used by applications and by optional supervisor
planning; it supports provider selection, caching, and fallback without making
provider output authoritative.

The canonical MCP runtime is `ipfs_accelerate_py.mcp_server`. The
`ipfs_accelerate_py.mcp` package remains a compatibility facade. MCP startup,
tool registration, policy, artifact, and transport details are documented in
the [MCP++ records](../../mcpplusplus/README.md) and the
[canonical MCP server README](../../ipfs_accelerate_py/mcp_server/README.md).

## Hardware and optional capabilities

Hardware support is adapter-driven. CPU execution is the baseline; CUDA,
ROCm, MPS, OpenVINO, WebNN, WebGPU, Qualcomm, and other backends depend on the
host, installed libraries, and model compatibility. Treat these as runtime
capabilities rather than static promises:

```bash
python - <<'PY'
from ipfs_accelerate_py import get_instance

print(get_instance().get_capabilities(detail=True))
PY
```

For CUDA, the PyTorch wheel and driver must agree. The installation guide
contains the CUDA 12.4 and CUDA 13 nightly paths. A driver being present is not
enough; `torch.cuda.is_available()` and a real model operation are the useful
checks.

## IPFS and P2P

Storage and distributed execution are optional integrations. The backend
router can use the local filesystem, HuggingFace caches, `ipfs_kit_py`, or a
Kubo-compatible service depending on installed dependencies and configuration.
P2P workflow and TaskQueue services are separate from local inference and must
be enabled explicitly. See [IPFS integration](../features/ipfs/IPFS.md) and
[P2P guides](../guides/p2p/README.md).

## Agent supervisor control plane

The supervisor is composed of independently testable layers:

| Layer | Current implementation boundary | Responsibility |
| --- | --- | --- |
| Intent | `objective_graph.py`, `objective_tracker.py` | Goal identity, evidence requirements, and dependencies. |
| Analysis | `analysis_ast_index.py`, `analysis_retrieval.py`, `code_evidence_graph.py`, `todo_vector_index.py` | Bounded lexical, AST, dependency, vector, and proof-gap evidence. |
| Projection | `objective_daemon.py`, `backlog_refinery.py`, `taskboard_store.py` | Convert gaps into canonical todo records and bundle shards. |
| Admission | `lease_coordination.py`, `resource_scheduler.py`, `conflict_graph.py` | Fencing, dependency readiness, resource budgets, and parallel width. |
| Execution | `todo_daemon/implementation_daemon.py`, `bundle_supervisor.py` | Isolated worktrees, implementation commands, validation, and merges. |
| Assurance | `formal_plan_*`, `prover_conformance.py`, `multi_prover_router.py`, `proof_*` | Typed plans, capability checks, proof receipts, and completion gates. |
| Recovery | `todo_daemon/implementation_supervisor.py`, `backlog_refinery.py` | Heartbeats, bounded retries, reconciliation, and repair tasks. |

Leanstral and other LLMs operate in the proposal tier. Deterministic parsers,
validators, capability probes, and authoritative prover receipts decide whether
their output can affect execution or completion. The design documents explain
the trust lattice, content-addressed artifacts, and failure semantics in more
detail:

- [Supervisor architecture](AGENT_SUPERVISOR_ARCHITECTURE.md)
- [Formal planning/prover matrix](AGENT_SUPERVISOR_FORMAL_PLANNING_PROVER_MATRIX_PLAN.md)
- [Formal verification](AGENT_SUPERVISOR_FORMAL_VERIFICATION_PLAN.md)
- [Leanstral goal development](AGENT_SUPERVISOR_LEANSTRAL_GOAL_DEVELOPMENT.md)

## Persistence and observability

Runtime status is exposed through bounded JSON/JSONL state, event logs, and
versioned JSON/DuckDB artifacts. Large source bodies and provider responses are
kept out of scheduler projections and prompts. Artifact queries should use
`ipfs-accelerate-agent-artifact-query` rather than reading raw databases into a
model context.

Supervisor lifecycle wrappers expose `check`, `ensure`, `stop`, and `spec`.
Health means both process liveness and a fresh heartbeat; a running PID alone
does not prove progress.

## Testing boundary

The repository has a large API/integration test tree rather than the small
`unit/`, `integration/`, and `performance/` layout used by older documentation.
Run focused tests first:

```bash
python -m pytest test/api/test_agent_supervisor_objective_graph.py -q
python -m pytest test/api/test_agent_supervisor_todo_daemon_port.py -q
python -m pytest test/api/test_unified_cli_integration.py -q
```

Optional hardware, MCP, P2P, browser, and external-provider tests require their
respective extras and services. The [testing guide](../development/testing.md)
describes the current test tree and how to select those suites.
