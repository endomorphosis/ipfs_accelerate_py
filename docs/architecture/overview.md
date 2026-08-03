# Architecture overview

**Status:** Current

**Audience:** Developers, operators, and agents needing a one-screen map before
subsystem guides

**Scope:** Maintained runtime boundaries, canonical vs compatibility surfaces,
and the separation between the inference/data plane and the supervisor/control
plane

**Non-goals:** Full actor tables, deep router or MCP policy detail, and ADR
records (see [SYSTEM_CONTEXT.md](SYSTEM_CONTEXT.md),
[Inference runtime](INFERENCE_RUNTIME.md),
[Model/service routing](MODEL_SERVICE_ROUTING.md),
[MCP runtime](MCP_RUNTIME.md), and
[Distributed runtime](DISTRIBUTED_RUNTIME.md))

**Last verified:** `e559ff0046c639ba1dadabe02ea0ea91d9877e20` (2026-08-03);
aligned with package layout, `pyproject.toml` scripts, and
[SYSTEM_CONTEXT.md](SYSTEM_CONTEXT.md)

This page describes the maintained runtime boundaries in the repository. It is
intentionally capability-oriented: optional integrations are discovered at
runtime, and a component is not considered available merely because its
Python module can be imported.

For actors, container maps, trust boundaries, and change rationale, read
[System context](SYSTEM_CONTEXT.md).

## Source anchors

| Concern | Primary path |
| --- | --- |
| Public package API | `ipfs_accelerate_py/__init__.py` |
| Inference coordinator | `ipfs_accelerate_py/ipfs_accelerate.py` |
| Unified CLI | `ipfs_accelerate_py/cli_entry.py`, `cli.py` |
| Canonical MCP | `ipfs_accelerate_py/mcp_server/` |
| MCP compatibility facade | `ipfs_accelerate_py/mcp/` |
| Agent supervisor | `ipfs_accelerate_py/agent_supervisor/` |
| Supervisor → LLM adapter | `agent_supervisor/todo_daemon/llm.py` |
| Scripts / extras | `pyproject.toml` |

## System layers

```text
Application and examples                    [conceptual]
        |
Python API / unified CLI                    [live: __init__, cli_entry/cli]
        |
Inference, model, embedding, voice, P2P     [live: ipfs_accelerate.py, routers, p2p_*]
        |
Hardware and provider adapters              [live: backends, api_backends, inference_backend_manager]
        |
IPFS, local storage, caches, external svcs  [live + optional: ipfs_backend_router, kit, FS/HF]
```

The MCP server is a shared protocol edge alongside these layers: it exposes
both product/data tools and supervisor/control tools, while authority remains
with the handler's owning plane. See [MCP runtime](MCP_RUNTIME.md) for current
transport-specific policy gaps and functional entrypoints.

The package also contains a **separate** agent-supervisor control plane. It is
not a layer inside the inference stack; it couples only through adapters:

```text
Objective heap (intent)                     [live: agent_supervisor/objectives]
        |
Objective graph + analysis + evidence       [live: analysis, proof, context]
        |
Todo and bundle projections                 [live: task_sources, objectives]
        |
Lease/resource admission + isolated lanes   [live: core/control, runtime, todo_daemon]
        |
LLM proposals ──adapter──► llm_router       [live: todo_daemon/llm, provider_execution]
        |
Deterministic validation → merge/receipts   [live: validation, merge, proof]
```

The agent supervisor is not on the inference hot path. It is an optional
maintainer/operator subsystem for turning a long-range objective into bounded,
validated implementation work. See the
[Agent Supervisor Architecture](AGENT_SUPERVISOR_ARCHITECTURE.md) and
[Agent Supervisor Guide](../guides/AGENT_SUPERVISOR_GUIDE.md).

### Why two planes and adapters

| Design choice | Rationale |
| --- | --- |
| Keep inference and supervisor separate | Serving models and admitting git mutations use different authority ladders; collapsing them upgrades untrusted model prose into merge power |
| Couple via adapters (`call_llm_router`, `provider_execution`) | Supervisor needs proposals and usage receipts without owning endpoint tables or bypassing leases/validation |
| Capability language for optional stacks | Hardware, IPFS, P2P, providers, and provers are environment-dependent; import ≠ capability ≠ proof |

Simpler alternatives (one god object, trust chat completion, claim features from
import success) break isolation, fail-closed admission, or operator truthfulness.
See [SYSTEM_CONTEXT.md §6–7](SYSTEM_CONTEXT.md) for full rationale and rejected
alternatives.

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

`model_catalog/` and `endpoint_usage/` own catalog resolution and usage-aware
routing identity when those paths are exercised. Deep lifecycle documentation
is maintained in [Inference runtime](INFERENCE_RUNTIME.md) and
[Model/service routing](MODEL_SERVICE_ROUTING.md).

## Current vs compatibility surfaces

| Surface | Status | Prefer for new work |
| --- | --- | --- |
| `get_instance()` / `ipfs_accelerate_py` | Current | Yes |
| `ipfs-accelerate` (`cli_entry` → `cli.py`) | Current unified CLI | Yes |
| `ipfs_accelerate` (`ai_inference_cli`) | Current **separate** script | Only when that parser is intentional |
| `ipfs_accelerate_py.mcp_server` | **Canonical** registry/runtime package; entrypoints have different transport completeness | Yes, using the concrete entrypoint guidance in `MCP_RUNTIME.md` |
| `ipfs_accelerate_py.mcp` | Compatibility facade | Migration / legacy recipes only |
| Domain imports under `agent_supervisor.<pkg>` | Current | Yes |
| Historical flat supervisor stems | Compatibility aliases | No |

The hyphenated and underscore CLI entry points are not interchangeable. Use
each command’s own `--help`.

The canonical MCP registry/runtime package is
`ipfs_accelerate_py.mcp_server`. The `ipfs_accelerate_py.mcp` package remains
a compatibility facade. MCP startup, tool registration, policy, artifact, and
transport details—including current direct-dispatch and standalone-host
limitations—are documented in [MCP runtime](MCP_RUNTIME.md); the
[MCP++ records](../../mcpplusplus/README.md) retain conformance evidence and
the [canonical MCP server README](../../ipfs_accelerate_py/mcp_server/README.md)
retains operator-oriented package notes.

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
| Intent | `agent_supervisor/objectives/` (`objective_graph`, `objective_tracker`) | Goal identity, evidence requirements, and dependencies. |
| Analysis | `agent_supervisor/analysis/` | Bounded lexical, AST, dependency, vector, and proof-gap evidence. |
| Projection | `objectives/` daemons, `task_sources/` | Convert gaps into canonical todo records and bundle shards. |
| Admission | `core/`, `control/`, resource schedulers | Fencing, dependency readiness, resource budgets, and parallel width. |
| Execution | `todo_daemon/`, `runtime/` | Isolated worktrees, implementation commands, validation, and merges. |
| Assurance | `planning/`, `proof/`, `validation/` | Typed plans, capability checks, proof receipts, and completion gates. |
| Recovery | `todo_daemon/implementation_supervisor`, `rescue/`, backlog refinery | Heartbeats, bounded retries, reconciliation, and repair tasks. |
| Provider adapter | `todo_daemon/llm.py`, `provider_execution.py` | Non-authoritative LLM calls into the inference plane. |

Leanstral and other LLMs operate in the proposal tier. Deterministic parsers,
validators, capability probes, and authoritative prover receipts decide whether
their output can affect execution or completion. The design documents explain
the trust lattice, content-addressed artifacts, and failure semantics in more
detail:

- [Supervisor architecture](AGENT_SUPERVISOR_ARCHITECTURE.md)
- [Supervisor philosophy](AGENT_SUPERVISOR_PHILOSOPHY.md)
- [Package map](agent_supervisor/PACKAGE_MAP.md)
- [Formal planning/prover matrix](AGENT_SUPERVISOR_FORMAL_PLANNING_PROVER_MATRIX_PLAN.md)
- [Formal verification](AGENT_SUPERVISOR_FORMAL_VERIFICATION_PLAN.md)
- [Leanstral goal development](AGENT_SUPERVISOR_LEANSTRAL_GOAL_DEVELOPMENT.md)

### Failure semantics (short)

- **Fail closed:** no lease, failed validation, or protected-path violation →
  no merge/completion.
- **Degrade:** missing optional providers or provers disable proposal/proof
  paths without inventing success.
- **Recover:** heartbeats, quarantine, rescue, and backlog refill — not silent
  rewrite of durable objectives.

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
python -m pytest test/test_unified_cli_integration.py -q
```

Optional hardware, MCP, P2P, browser, and external-provider tests require their
respective extras and services. The [testing guide](../development/testing.md)
describes the current test tree and how to select those suites.

## Verification

```bash
test -f docs/architecture/SYSTEM_CONTEXT.md
test -f docs/architecture/overview.md
rg -q 'Last verified' docs/architecture/overview.md
rg -qi 'rationale|why' docs/architecture/overview.md
rg -n 'mcp_server|compatibility|adapter|control plane' docs/architecture/overview.md
git diff --check
```

## Related

- [System context](SYSTEM_CONTEXT.md) — full actors, containers, flows, rationale
- [Guide conventions](GUIDE_CONVENTIONS.md) — ArchitectureGuideContract@1
- [Documentation current state](../development/DOCUMENTATION_CURRENT_STATE.md)
