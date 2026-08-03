# System context

**Status:** Current

**Audience:** New developers, integrators, maintainers, operators, security
reviewers, and implementation agents who need the product shape before reading
subsystem guides

**Scope:** Actors, containers, canonical entrypoints, the inference/data plane,
the separate supervisor/control plane, capability boundaries, trust and failure
semantics, and the rationale for coupling those planes through adapters rather
than collapsing them

**Non-goals:** Deep router lifecycle (see
[Inference runtime](INFERENCE_RUNTIME.md)), model-catalog and endpoint-usage
detail (see [Model/service routing](MODEL_SERVICE_ROUTING.md)), full MCP
transport and policy (see [MCP runtime](MCP_RUNTIME.md)), IPFS/P2P runtime
depth (see [Distributed runtime](DISTRIBUTED_RUNTIME.md)), cross-repository
ownership depth (see [Integration boundaries](INTEGRATION_BOUNDARIES.md)), and
supervisor domain deep dives under `docs/architecture/agent_supervisor/`

**Last verified:** `e559ff0046c639ba1dadabe02ea0ea91d9877e20` (2026-08-03);
package layout, `pyproject.toml` scripts, public package exports, MCP canonical
vs facade paths, and supervisor domain packages checked against the tree

## Source anchors

| Concern | Primary path / symbol | Notes |
| --- | --- | --- |
| Package boundary and public exports | `ipfs_accelerate_py/__init__.py` | Lazy/optional imports; `get_instance` |
| Inference coordinator | `ipfs_accelerate_py/ipfs_accelerate.py` (`ipfs_accelerate_py`, `get_capabilities`) | Process-local runtime coordinator |
| Unified CLI (canonical operator surface) | `ipfs_accelerate_py/cli_entry.py` → `cli.py` | Installed as `ipfs-accelerate` |
| Direct AI CLI (separate script) | `ipfs_accelerate_py/ai_inference_cli.py` | Installed as `ipfs_accelerate` (underscore) |
| Canonical MCP runtime | `ipfs_accelerate_py/mcp_server/` | Unified MCP++ server |
| MCP compatibility facade | `ipfs_accelerate_py/mcp/` | Retained alternate surface; not preferred for new work |
| Model catalog | `ipfs_accelerate_py/model_catalog/` | Resolution and identity |
| Endpoint usage | `ipfs_accelerate_py/endpoint_usage/` | Usage-aware routing ledger |
| LLM / embedding / multimodal / voice routers | `llm_router.py`, `embeddings_router.py`, `multimodal_router.py`, voice/TTS modules | Provider-optional |
| Hardware / container backends | `backends/`, `container_backends/`, `inference_backend_manager.py` | Capability-probed |
| IPFS storage routing | `ipfs_backend_router.py`, `ipfs_kit_integration.py` | Optional sibling/kit adapters |
| P2P and task execution | `p2p_tasks/`, `p2p_workflow_discovery.py`, `p2p_workflow_scheduler.py` | Explicit enablement |
| HF model server | `hf_model_server/` | Optional serving surface |
| Agent supervisor control plane | `ipfs_accelerate_py/agent_supervisor/` | Domain packages + daemons |
| Supervisor ↔ provider adapter | `agent_supervisor/todo_daemon/llm.py` (`call_llm_router`), `agent_supervisor/provider_execution.py` | Typed, receipt-bearing calls into inference routers |
| Package metadata and console scripts | `pyproject.toml` | Extras and installed entry points |

Related: [Architecture overview](overview.md) ·
[Guide conventions](GUIDE_CONVENTIONS.md) ·
[Documentation current state](../development/DOCUMENTATION_CURRENT_STATE.md) ·
[Agent supervisor philosophy](AGENT_SUPERVISOR_PHILOSOPHY.md) ·
[Supervisor package map](agent_supervisor/PACKAGE_MAP.md).

---

## 1. Context and component map

### 1.1 Product in one paragraph

`ipfs_accelerate_py` is a **capability-oriented** Python package for
hardware-accelerated inference, optional IPFS/P2P distribution, MCP tool
serving, and an optional **agent-supervisor control plane** for
objective-driven implementation work. Optional integrations are discovered and
probed at runtime. **Import success is not a capability signal**; capability is
not proof.

### 1.2 Actors

| Actor | Role | Primary surfaces |
| --- | --- | --- |
| Application developer | Calls the Python API for inference, embeddings, voice, storage | `from ipfs_accelerate_py import get_instance`, routers |
| CLI operator | Runs installable commands for inference, MCP, and agent workflows | `ipfs-accelerate`, `ipfs_accelerate` |
| MCP host / IDE client | Invokes tools over MCP transports | `mcp_server` (canonical), `mcp` (compatibility) |
| Inference consumer | Needs model results, not control-plane side effects | Routers, HF model server, worker backends |
| Maintainer / implementation agent | Turns durable objectives into validated code changes | `agent_supervisor` daemons, control service, entrypoints |
| Operator / SRE | Health, recovery, resource budgets, run state | CLI lifecycle helpers, artifact query, event logs |
| External systems | Model providers, IPFS/Kubo, HuggingFace, sibling kits | Optional adapters only |

### 1.3 Two planes (normative separation)

The product is deliberately split into two planes that share a process or host
only when wired by **adapters**:

```text
┌──────────────────────────────────────────────────────────────────────────┐
│  Clients (apps, CLI users, MCP hosts, operators)          [conceptual]   │
└───────────────┬───────────────────────────────┬──────────────────────────┘
                │                               │
                ▼                               ▼
┌───────────────────────────────┐   ┌──────────────────────────────────────┐
│  INFERENCE / DATA PLANE       │   │  SUPERVISOR / CONTROL PLANE          │
│  (serve models and content)   │   │  (admit and land software work)      │
│                               │   │                                      │
│  ipfs_accelerate.py           │   │  agent_supervisor/                   │
│  model_catalog/               │   │    objectives, planning, validation  │
│  endpoint_usage/              │   │    control, runtime, merge, rescue   │
│  llm_router / embeddings /    │   │    todo_daemon, entrypoints          │
│  multimodal / voice routers   │   │                                      │
│  backends / container_backends │   │  Authority: leases, allowlists,       │
│  ipfs_backend_router, p2p_*   │   │  deterministic validation, receipts  │
│  product capability handlers │   │  supervisor/control handlers         │
└───────────────┬───────────────┘   └──────────────────┬───────────────────┘
                │                                      │
                │     adapters (typed, non-authoritative provider calls)     │
                │◄─────────────────────────────────────┘
                │  todo_daemon/llm.call_llm_router
                │  provider_execution gateway
                │  endpoint_usage receipts (operational evidence only)
                ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  Optional capabilities: CUDA/CPU/…, HF caches, IPFS/kit, providers,     │
│  provers — present only when installed, configured, and probed          │
└──────────────────────────────────────────────────────────────────────────┘
```

`mcp_server` is a **shared protocol edge**, not a data-plane authority owner.
It exposes handlers from both boxes. Framing and registration stay at the
edge; each handler retains the authority rules of its owning plane. In
particular, exposing a supervisor operation through MCP does not bypass its
leases, validation, or merge fences.

| Plane | Owns | Must not own |
| --- | --- | --- |
| **Inference / data plane** | Endpoint registration, model/provider selection, hardware adapters, product capability handlers, IPFS/P2P content and task paths | Authoritative completion of implementation work, lease fencing of git mutations, objective-heap truth |
| **Supervisor / control plane** | Objectives, taskboards, admission, isolated implementation lanes, validation gates, merge/recovery, proof receipts | Hot-path inference serving, silent trust upgrade of model prose into merge authority |
| **Shared MCP protocol edge** | Transport framing, tool registration and routing to data- or control-plane handlers | Replacing the owning handler's authorization, lease, validation, or usage rules |

### 1.4 Container and package map

Every named box maps to a live package or module, or is labelled **conceptual**.

| Box | Kind | Live path / symbol |
| --- | --- | --- |
| Application and examples | conceptual | Repo `examples/`, external apps; not a package guarantee |
| Python public API | live | `ipfs_accelerate_py/__init__.py` |
| Inference coordinator | live | `ipfs_accelerate_py/ipfs_accelerate.py` |
| Unified CLI | live | `cli_entry.py`, `cli.py` (`ipfs-accelerate`) |
| AI inference CLI | live | `ai_inference_cli.py` (`ipfs_accelerate`) |
| Canonical MCP server | live | `mcp_server/` |
| MCP compatibility facade | live (compatibility) | `mcp/` |
| MCP++ profile modules | live | `mcplusplus_module/`, `mcp_server/mcplusplus/` |
| Model catalog | live | `model_catalog/` |
| Endpoint usage ledger | live | `endpoint_usage/` |
| LLM router | live | `llm_router.py` (+ `llm/`) |
| Embeddings router | live | `embeddings_router.py`, `embeddings/` |
| Multimodal router | live | `multimodal_router.py` |
| Voice / TTS | live | voice modules, `voice_jobs/`, TTS router exports |
| Inference backend manager | live | `inference_backend_manager.py` |
| Hardware backends | live | `backends/`, `container_backends/` |
| API provider backends | live | `api_backends/` |
| HF model server | live | `hf_model_server/` |
| IPFS backend router | live | `ipfs_backend_router.py` |
| IPFS kit integration | live (optional) | `ipfs_kit_integration.py`; sibling under `external/` when present |
| P2P tasks / workflows | live | `p2p_tasks/`, `p2p_workflow_*` |
| Agent supervisor domains | live | `agent_supervisor/{core,control,objectives,…}/` |
| Implementation daemons | live | `agent_supervisor/todo_daemon/` |
| Prompt-first entrypoints | live | `agent_supervisor/entrypoints/` |
| Supervisor provider gateway | live | `agent_supervisor/provider_execution.py` |
| Supervisor LLM adapter | live | `agent_supervisor/todo_daemon/llm.py` |
| Shared CLI runtime helpers | live | `cli_runtime/` |
| Local storage / caches / Kubo | conceptual + optional live | Filesystem, HF cache, Kubo/service; selected by router config |
| External model providers | conceptual + optional live | OpenAI/Anthropic/etc. via `api_backends` and router providers when configured |

### 1.5 Current vs compatibility surfaces

| Surface | Status | Use for new work |
| --- | --- | --- |
| `ipfs_accelerate_py.ipfs_accelerate_py` + `get_instance()` | **Current** coordinator API | Yes |
| `ipfs-accelerate` → `cli_entry:main` → `cli.py` | **Current** unified CLI | Yes |
| `ipfs_accelerate` → `ai_inference_cli:main` | **Current** but **separate** script | Only when that parser is intentional; not interchangeable with hyphenated CLI |
| `ipfs_accelerate_py.mcp_server` | **Canonical** registry/runtime package; transport completeness varies by entrypoint | Yes, with the concrete transport guidance below |
| `ipfs_accelerate_py.mcp` | **Compatibility facade** | Migration and older docs only; do not treat as preferred |
| `ipfs_accelerate_py/ipfs_accelerate_py_legacy.py` | **Legacy** module | Avoid; not the default export path |
| Flat `agent_supervisor.*` historical stems | **Compatibility aliases** (where still resolved) | Prefer domain imports (`agent_supervisor.control.…`) |
| `agent_supervisor.entrypoints` | **Current** prompt-first composition facade | Yes, for closed request/result contracts |

The hyphenated and underscore CLI entry points are **not** interchangeable
parsers. Always use each command’s own `--help`.

---

## 2. Canonical entrypoints

### 2.1 Python

```python
from ipfs_accelerate_py import get_instance

accelerator = get_instance()
capabilities = accelerator.get_capabilities(detail=True)
# capabilities["task_types"] is a runtime summary, not a static promise
```

Optional routers and helpers are re-exported from the package root when their
dependencies import successfully. Deployment profiles must install extras from
`pyproject.toml` and **probe** before serving traffic.

### 2.2 CLI and console scripts

From `pyproject.toml` `[project.scripts]` (representative):

| Script | Module | Plane |
| --- | --- | --- |
| `ipfs-accelerate` | `cli_entry:main` | Product (inference, MCP, agent subcommands) |
| `ipfs_accelerate` | `ai_inference_cli:main` | Inference-oriented direct CLI |
| `ipfs-accelerate-agent-objective-daemon` | `agent_supervisor.objectives.objective_daemon:main` | Control |
| `ipfs-accelerate-agent-backlog-refinery` | `…objectives.backlog_refinery:main` | Control |
| `ipfs-accelerate-agent-bundle-supervisor` | `…objectives.bundle_supervisor:main` | Control |
| `ipfs-accelerate-agent-implementation-daemon` | `…todo_daemon.implementation_daemon:main` | Control |
| `ipfs-accelerate-agent-implementation-supervisor` | `…todo_daemon.implementation_supervisor:main` | Control |
| `ipfs-accelerate-agent-merge-resolver` | `…merge.merge_resolver:main` | Control |
| `ipfs-accelerate-agent-artifact-query` | `…runtime.artifact_store:main` | Control (observability) |

### 2.3 MCP

| Entrypoint | Role |
| --- | --- |
| `from ipfs_accelerate_py.mcp_server import create_server` | Canonical in-process registry/lifecycle builder; not by itself a mounted MCP protocol host |
| `python -m ipfs_accelerate_py.mcp_server` | Current lifecycle shell serves `/healthz` only; do not point MCP clients at it |
| `python -m ipfs_accelerate_py.mcp_server.fastapi_service` | Current functional FastAPI/HTTP MCP host |
| `ipfs-accelerate mcp start` | Unified CLI's Flask/integrated-dashboard path; it does not call canonical `mcp_server.create_server` |
| `python -m ipfs_accelerate_py.mcp.cli …` | **Compatibility / direct** path still used in some operator recipes |

### 2.4 Supervisor control transports

The supervisor exposes **one operation vocabulary** over three transports
(Python control service, CLI, MCP). Transports differ in root and allowlist
configuration, not in operation meaning. See
[Agent Supervisor Guide](../guides/AGENT_SUPERVISOR_GUIDE.md).

---

## 3. Primary flows

### 3.1 Inference request (data plane)

```text
Client / CLI / MCP tool
        |
  transport delivery (authority depends on the selected MCP handler path)
        |
  coordinator or dedicated router
  (ipfs_accelerate_py | llm_router | embeddings_router | …)
        |
  model_catalog / endpoint_usage resolution (when used)
        |
  provider or hardware backend adapter
  (api_backends | backends | container_backends | browser)
        |
  optional storage path (local | HF cache | ipfs_backend_router)
        |
  result or structured error
```

The current FastAPI and MCP+p2p adapters can call a registered function or
`manager.dispatch` directly. Risk, UCAN, temporal-policy, and dispatch-input
gates are guaranteed only when the request explicitly invokes the
`tools_dispatch` meta-tool. Until direct transport calls are funneled through
that authority path, deployment authentication/allowlists must protect them;
an HTTP or peer connection alone is not tool authorization.

**Failure path:** missing optional dependency → feature disabled or explicit
error (not silent “success”); provider timeout/fallback stays inside the router
contract; capability absence must not be rewritten as a green probe.

### 3.2 Supervisor implementation loop (control plane)

```text
Objective heap (durable intent)
        |
  projection → taskboard / bundles
        |
  lease + resource admission
        |
  isolated worktree / implementation lane
        |
  model proposal  ──adapter──►  llm_router / provider_execution
        |                         (non-authoritative text/proposal)
        ▼
  deterministic validation / tests / scope policy
        |
  merge train or quarantine / rescue
        |
  receipts, event log, completion marks
```

**Failure path:** missing lease, failed validation, protected-path write, or
stale capability → **fail closed** (no merge/completion). Provider outage
degrades planning/proposal only; it does not invent completion evidence.

### 3.3 How the planes couple (adapters only)

The control plane needs model generations for planning and implementation
proposals. It does **not** import the inference coordinator as a shared god
object that also owns git mutation authority. Coupling points:

| Adapter | Path | What crosses | What does not cross |
| --- | --- | --- | --- |
| LLM router invocation | `agent_supervisor/todo_daemon/llm.py` (`call_llm_router`) | Bounded prompts → text proposals | Merge authority, lease truth |
| Provider execution gateway | `agent_supervisor/provider_execution.py` | Reserved/settled provider calls + redacted receipts | Usage receipts as completion proof |
| Endpoint usage identity helpers | `endpoint_usage/` used by provider_execution | Shared identity/redaction rules | Inference hot-path scheduling |
| MCP tool surface | `mcp_server` may expose both product tools and supervisor ops | Transport | Collapsed trust levels |

```text
Supervisor planning / implementation
        |
  call_llm_router / provider_execution   [adapter boundary]
        |
  llm_router / provider backends         [inference plane]
        |
  text or structured proposal only
        |
  validators / provers / tests           [control plane again]
```

---

## 4. State and identity (summary)

| Kind | Owner (typical) | Notes |
| --- | --- | --- |
| Process coordinator state | `ipfs_accelerate_py` instance | Endpoints, queues, caches in `resources` |
| Catalog / endpoint identity | `model_catalog/`, `endpoint_usage/` | Content-addressed where implemented |
| MCP dispatch / CID artifacts | `mcp_server/` | Tool intents and execution artifacts |
| IPFS CIDs / backend choice | `ipfs_backend_router`, kit adapters | Optional |
| Objectives, boards, leases | `agent_supervisor` domains | Durable intent vs schedulable projection |
| Implementation worktrees | `todo_daemon`, merge packages | Isolated lanes |
| Proof and completion receipts | `proof/`, validation, authoritative completion modules | Typed tiers; cache re-derives assurance |

Large provider bodies and secrets must stay out of scheduler projections and
documentation. Prefer `ipfs-accelerate-agent-artifact-query` for bounded
artifact inspection.

---

## 5. Trust, authorization, and failure semantics

### 5.1 Trust ladder (product-wide)

| Signal | Means | Does not mean |
| --- | --- | --- |
| Import / discovery | Module or operation vocabulary exists | Backend works |
| Capability / probe | Configured path can *attempt* work | SLA or proof |
| Validation / tests | Deterministic checks passed for a claim set | Kernel-level proof |
| Proof / attestation | Stated assurance level met | Future claims free |
| Model prose / chat log | Proposal only | Admission or completion |

### 5.2 Fail-closed conditions

- Supervisor mutations without a valid lease, allowlist, or required validation.
- Treating MCP or CLI board status alone as authorization to merge.
- Claiming optional hardware, IPFS, P2P, browser, or prover features from import
  success alone.
- Promoting usage receipts or cache hits into completion authority.

### 5.3 Degradation

| Missing piece | Expected behavior |
| --- | --- |
| Optional ML/provider extra | Router/feature unavailable; explicit error or empty capability |
| IPFS kit / Kubo | Fall back per backend router config (often local FS); no fake CID success |
| P2P services | Local inference continues; distributed paths stay off until enabled |
| Supervisor prover / LLM | Planning degrades; no silent “completed” without configured gates |
| `IPFS_ACCEL_SKIP_CORE=1` | Heavy core import skipped; related APIs raise `NotImplementedError` |

### 5.4 Recovery

- Inference: retry/fallback inside router contracts; operator re-probe after
  install/driver changes.
- Supervisor: heartbeats, bounded retries, rescue/quarantine, backlog refill;
  see operator guide. A live PID without a fresh heartbeat is not health.

---

## 6. Rationale — why this shape exists

### 6.1 Why two planes instead of one package blob

1. **Different authority models.** Inference optimizes for routing, latency,
   and optional backends. The supervisor optimizes for isolation, evidence, and
   fail-closed mutation of a repository. Collapsing them invites treating a
   fluent model answer as a merge decision.
2. **Different lifecycles.** Serving traffic should not require objective heaps,
   worktrees, or proof caches. Objective-driven maintenance should not sit on
   the inference hot path or share its process failure domain by default.
3. **Optional composition.** Many deployments need only inference + MCP. Others
   run supervisor daemons as maintainer tooling. Adapters let either plane
   improve without forcing a monokernel rewrite.
4. **Evidence hygiene.** Provider output is untrusted proposal material.
   Validation, leases, and typed receipts must remain control-plane owned even
   when the text was produced by the same machine’s `llm_router`.

### 6.2 Why adapters rather than direct ownership

- **Bounded contracts:** `call_llm_router` and `provider_execution` pass
  prompts/configs and return text/receipts — not shared mutable endpoint tables
  that both planes can race.
- **Accounting and redaction:** reservation/settlement and endpoint redaction
  live at the adapter so inference providers need not know about supervisor
  attribution.
- **Testability:** each plane’s unit and API tests can mock the adapter without
  booting the other plane’s full stack.
- **Security boundary:** allowlists and protected paths remain meaningful only
  if the control plane does not “reach through” into inference internals to
  bypass validation.

### 6.3 Why capability language everywhere

Hardware, providers, IPFS, P2P, MCP transports, browsers, and provers are
**environment-dependent**. Documenting them as universal features produces
drift and false confidence. Runtime probes (`get_capabilities`, discovery then
capability reports on the control plane) are the supported truth.

---

## 7. Alternatives considered

| Alternative | Why rejected / what breaks |
| --- | --- |
| Single process “god object” owning inference + git merge authority | Model or MCP tool compromise becomes repo compromise; no clear fail-closed edge |
| Supervisor imports and mutates `ipfs_accelerate_py` internals for “efficiency” | Couples release cadence; breaks isolation tests; blurs trust tiers |
| Trust model completion from chat transcripts | Skips validation rung; non-reproducible; unsafe under prompt injection |
| Document import success as availability | Lies about CUDA/IPFS/P2P/prover readiness; fails operators and CI |
| Delete compatibility MCP facade immediately | Breaks older scripts and tests still on `mcp/`; cutover is staged |
| One CLI binary for all historical parsers | Underscore vs hyphen scripts already diverge; forcing one silently misroutes flags |

---

## 8. Consequences

**Positive**

- Readers can place new code: data-path features near routers/backends; control
  features in supervisor domain packages.
- Optional stacks install cleanly; failure modes stay explicit.
- Security review can point at adapter and lease boundaries instead of hunting
  through a single mega-module.
- Parallel documentation and implementation lanes can own disjoint paths.

**Negative / costs**

- More packages and entry points to learn; mitigated by this map and overview.
- Dual MCP trees and dual CLIs require **canonical vs compatibility** labeling
  in every operator-facing guide.
- Adapter discipline must be enforced in review (no new reverse imports from
  inference core into supervisor foundation packages).

---

## 9. Extension and compatibility constraints

- New inference providers: register through existing router/provider patterns;
  expose capability probes; do not special-case the supervisor.
- New supervisor domains: follow the acyclic DAG in
  [PACKAGE_MAP.md](agent_supervisor/PACKAGE_MAP.md); do not reintroduce flat
  root stubs as the public API.
- New MCP tools: prefer registration on the **canonical** `mcp_server` runtime;
  keep facade shims thin.
- Sibling repositories (`ipfs_kit_py`, datasets, etc.) integrate via optional
  injection and adapters (`resources`/`metadata` injection on the coordinator,
  `agent_supervisor/integrations/`), not hard import at package import time.

---

## 10. Operational signals

| Signal | Where |
| --- | --- |
| Capability summary | `get_instance().get_capabilities(detail=True)` |
| CLI help | `ipfs-accelerate --help`, `ipfs_accelerate --help` |
| MCP health / tools | Server logs, meta-tools (`tools_list_*` on unified runtime) |
| Supervisor heartbeat / lease | Daemon status helpers, event logs (operator guide) |
| Artifacts | `ipfs-accelerate-agent-artifact-query` |
| Tests | Focused `test/api/` modules for the claim under change |

---

## 11. Verification recipe

Run from the repository root on a checkout that matches **Last verified** (or
re-run after updating that field):

```bash
# Anchors exist
test -f ipfs_accelerate_py/__init__.py
test -f ipfs_accelerate_py/ipfs_accelerate.py
test -f ipfs_accelerate_py/cli_entry.py
test -d ipfs_accelerate_py/mcp_server
test -d ipfs_accelerate_py/mcp
test -d ipfs_accelerate_py/agent_supervisor
test -f ipfs_accelerate_py/agent_supervisor/todo_daemon/llm.py
test -f ipfs_accelerate_py/agent_supervisor/provider_execution.py

# Console scripts declare both planes
rg -n 'ipfs-accelerate|agent_supervisor|cli_entry|ai_inference_cli' pyproject.toml

# Document contract
rg -q 'Last verified' docs/architecture/SYSTEM_CONTEXT.md
rg -qi 'rationale|why' docs/architecture/SYSTEM_CONTEXT.md
rg -n 'mcp_server|compatibility|call_llm_router|two planes|adapter' docs/architecture/SYSTEM_CONTEXT.md

# Whitespace / conflict markers
git diff --check
```

Optional (needs installed package and extras):

```bash
python - <<'PY'
from ipfs_accelerate_py import get_instance
print(sorted(get_instance().get_capabilities(detail=True).get("task_types", [])))
PY
```

Do not require live provider inference, GPU, or network IPFS for architecture
doc verification.

---

## 12. Related guides and deep dives

| Document | Role |
| --- | --- |
| [overview.md](overview.md) | One-screen maintained overview |
| [AGENT_SUPERVISOR_PHILOSOPHY.md](AGENT_SUPERVISOR_PHILOSOPHY.md) | Control-plane mental model |
| [AGENT_SUPERVISOR_ARCHITECTURE.md](AGENT_SUPERVISOR_ARCHITECTURE.md) | Deep supervisor architecture |
| [agent_supervisor/PACKAGE_MAP.md](agent_supervisor/PACKAGE_MAP.md) | Domain ownership DAG |
| [../guides/AGENT_SUPERVISOR_GUIDE.md](../guides/AGENT_SUPERVISOR_GUIDE.md) | Operator journeys |
| [../development/DOCUMENTATION_CURRENT_STATE.md](../development/DOCUMENTATION_CURRENT_STATE.md) | Normative vs historical docs |
| [INFERENCE_RUNTIME.md](INFERENCE_RUNTIME.md) | Current router and inference lifecycle |
| [MODEL_SERVICE_ROUTING.md](MODEL_SERVICE_ROUTING.md) | Current catalog, usage and invocation boundaries |
| [MCP_RUNTIME.md](MCP_RUNTIME.md) | Current MCP/MCP++ runtime and known transport gaps |
| [DISTRIBUTED_RUNTIME.md](DISTRIBUTED_RUNTIME.md) | Current IPFS/P2P execution boundary |
| [INTEGRATION_BOUNDARIES.md](INTEGRATION_BOUNDARIES.md) | Current sibling-repository ownership map |
