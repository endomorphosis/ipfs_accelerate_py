# Product glossary

**Status:** Reference
**Owner:** documentation-governance / architecture maintainers
**Audience:** developers, operators, integrators, implementation agents, and
readers who need shared product vocabulary before subsystem guides
**Scope:** Canonical semantic terms for the inference/data plane, MCP surfaces,
distributed identity, and agent-supervisor control plane. Distinguishes pairs
that are frequently collapsed in casual prose.
**Non-goals:** Full API reference; board-prefix inventories (see
[PROGRAMS.md](agent_supervisor/PROGRAMS.md)); ADR body text (see
[decisions/](decisions/README.md)); package install or CLI flag catalogs.
**Sources:** [DOCUMENTATION_LIFECYCLE.md](../development/DOCUMENTATION_LIFECYCLE.md);
maintained architecture guides under this directory; ADRs 0001–0006;
`ipfs_accelerate_py/model_catalog/`, `endpoint_usage/`, `mcp_server/`,
`agent_supervisor/`, `ipfs_backend_router.py`, `multiformats_identity.py`.
**Last-verified:** 2026-08-03; terms checked against landed Current guides and
Accepted ADRs on the documentation-refresh tree.
**Interface:** ProductGlossary@1

This glossary is **Reference** vocabulary. It does not alone authorize runtime
behavior. Live code, schemas, executable help, and **Current** guides override
any term interpretation here when they disagree. Plans, completion summaries,
and archives cited below are labelled and are **not** Current contracts.

Companion pages:

- [Architecture hub](README.md) — audience and concern routes
- [Documentation manifest](../development/DOCUMENTATION_MANIFEST.md) — status
  classification of maintained and non-maintained docs
- [Lifecycle policy](../development/DOCUMENTATION_LIFECYCLE.md) — status
  vocabulary and authority ladder
- [Guide conventions](GUIDE_CONVENTIONS.md) — how architecture guides are written

---

## How to use this glossary

1. Prefer the **semantic name** in product prose (what the system *is*).
2. Use board prefixes (`DOC-`, `ASI-`, `CBP-`, …) only when operating a taskboard
   or citing execution evidence ([PROGRAMS.md](agent_supervisor/PROGRAMS.md)).
3. When two terms appear similar, read the **Do not confuse with** line.
4. Optional stacks use **capability language**: import success is not
   availability; availability is not proof.

---

## Closed documentation status terms

These statuses are normative for documentation classification (lifecycle policy).
They are not runtime feature flags.

| Term | Meaning |
| --- | --- |
| **Current** | Describes landed, supported behavior for the checked-out tree. Normative narrative subject to code/test override. |
| **Reference** | Maintained orientation (glossary, package map, ADR *why*). Does not alone authorize runtime behavior. |
| **Plan** | Intended or in-progress design. Sequencing only; never a substitute for landed contracts. |
| **Historical** | Point-in-time report, completion summary, session log, or superseded guide. Context only. |
| **Generated** | Machine-produced inventory or export. Stale when generator inputs change. |
| **Vendored** | Third-party or nested-product docs; upstream contract wins. |

A filename containing `plan`, `summary`, `complete`, or `todo` does **not** make
a document Current. Plans and archives must be labelled when linked.

---

## Distinction tables (required pairs)

### Catalog · usage · router

| Term | Owns | Does **not** own |
| --- | --- | --- |
| **Catalog** (`model_catalog`, AI Service Catalog) | Service **identity**, declared readiness, content-addressed inventory revisions, side-effect-free list/get/resolve | Live quota, client construction, HTTP to providers, credential spend |
| **Usage** (`endpoint_usage`) | Capacity **accounting**, headroom, cooldowns, atomic reservation and settlement under an independent usage revision | Catalog identity rewrites, modality-specific invoke contracts |
| **Router** (modality routers: LLM, embeddings, multimodal, voice) | **Invocation**: provider construction, streaming/batching, fallback, timeouts, provider errors | Being the source of catalog truth or lease authority |

**Do not confuse:** resolving a catalog record with calling a model; a usage
reservation with a catalog publish; import of `model_catalog` with a healthy
endpoint. See [MODEL_SERVICE_ROUTING.md](MODEL_SERVICE_ROUTING.md),
[AI_SERVICE_CATALOG.md](AI_SERVICE_CATALOG.md), and
[ADR-0003](decisions/0003-capabilities-catalogs-and-routing.md).

### MCP · MCP++

| Term | Meaning |
| --- | --- |
| **MCP** | Model Context Protocol tool/runtime surface. Canonical package: `ipfs_accelerate_py.mcp_server`. Compatibility facade: `ipfs_accelerate_py.mcp` (legacy import path; not preferred for new work). |
| **MCP++** | Extended primitives and profiles layered on the MCP runtime: interface descriptors, artifacts, UCAN/policy, Event DAG, risk scheduling, framed P2P (`/mcp+p2p/…`). Implemented primarily under `mcp_server/mcplusplus/`; alternate Trio surface in `mcplusplus_module/`. Spec/evidence workspace under repo-root `mcpplusplus/` is **not** the importable product runtime. |

**Do not confuse:** presence of the `mcp` extra or a listening health port with
a full MCP++ profile; the compatibility `mcp` package with the canonical
`mcp_server` ownership target. See [MCP_RUNTIME.md](MCP_RUNTIME.md)
(**Status:** Current). Spec gap matrices under `mcpplusplus/` are **Plan** /
evidence, not Current API docs.

### Objective · task

| Term | Meaning |
| --- | --- |
| **Objective** | Durable intent: goal identity, parent/child structure, acceptance criteria, evidence expectations. Stored in objective heaps (`*.objectives.md`) and the `objectives/` package. Bound by `objective_id` / `objective_revision`. |
| **Task** | Schedulable **projection** of work: board header (`## PREFIX-###`), deps, shard filters, drainable status. Boards (`*.todo.md`) and queues under `task_sources/` may be refined or regenerated under policy. |

**Do not confuse:** board status `done` with goal completion; rewriting a board
with rewriting protected objectives. See
[ADR-0001](decisions/0001-objectives-and-task-projections.md).

### Discovery · capability · proof

| Term | Meaning |
| --- | --- |
| **Discovery** | Learning what vocabulary, descriptors, peers, or workflows **exist** without installing, reserving, or invoking. Side-effect-free list/resolve and optional probes that report absence honestly. |
| **Capability** | Environment-dependent ability to perform work: installed extras, credentials, hardware, network, published snapshots, healthy peers. **Import success is not capability.** |
| **Proof** | Typed, tiered evidence that a claim holds under policy (kernel proof, attestation, re-derived assurance). Weaker tiers (observation, model prose, cache hit) must not self-promote. |

**Do not confuse:** `import` succeeding with a feature being usable; a capability
probe with formal proof; a proof-cache hit inventing assurance it did not
re-derive. Ladder: discovery → capability → (optional) invocation → proof.

### CID · cache key

| Term | Meaning |
| --- | --- |
| **CID** | Content identifier under the multiformats profile used for coordination-facing IPLD (CIDv1 / base32 / sha2-256 / raw\|dag-json). Admitted only after fail-closed rehash verification on coordination paths. |
| **Cache key** | Local or synthetic key for API/response caches or HF-cache helpers. Often *looks* like `bafy…` plus truncated hex but is **not** a verified multiformats CID and must not grant coordination authority. |

**Do not confuse:** pin success or a synthetic cache string with a verified
epoch or lease grant. See [DISTRIBUTED_RUNTIME.md](DISTRIBUTED_RUNTIME.md) and
[ADR-0005](decisions/0005-mutable-coordination-and-immutable-replication.md).

### Merge · acceptance

| Term | Meaning |
| --- | --- |
| **Merge** | Implementation **landing**: code entered Git history (implementation commit, merge train, ancestry). Evidence that a patch was integrated—not that acceptance criteria hold. |
| **Acceptance** | Authoritative completion: deterministic gates recomputed from tree-bound evidence (`merge`, `freshness`, `semantic`, `proof`, `provider_review`, `deterministic_only` as configured). Models propose; evidence admits. |

**Do not confuse:** a green merge or fluent “done” prose with task/goal
acceptance. Merge can yield `implemented_merged_but_pending` until gates pass.
See [ADR-0002](decisions/0002-model-proposals-and-evidence-admission.md).

### Coordination · replication

| Term | Meaning |
| --- | --- |
| **Coordination** | Single-writer **mutable** active state: claims, leases, fences, heartbeats, queue ownership (DuckDB + flock CAS on the shard owner). Only the current owner authorizes mutations. |
| **Replication** | **Immutable** distribution of committed history and content (Parquet epochs, IPLD/CAR, IPFS pins). Replicas are read-only; `grants_authority` is always false. |

**Do not confuse:** fetching a replica or resolving an IPNS head with owning a
lease; multi-writer shared DuckDB files with safe coordination. See
[ADR-0005](decisions/0005-mutable-coordination-and-immutable-replication.md).

---

## Inference and product plane

| Term | Definition |
| --- | --- |
| **Inference / data plane** | Path that serves models, embeddings, voice, and related results. Separate authority ladder from the agent-supervisor control plane. |
| **Control plane (agent supervisor)** | Optional subsystem that turns durable objectives into validated implementation work. Not on the inference hot path. |
| **Modality router** | Invocation owner for one media class (`llm_router`, `embeddings_router`, `multimodal_router`, `voice_router`). |
| **ModelManager** | Planning/list/resolve facade over catalog and usage; does not replace modality routers for invoke. |
| **Endpoint** | Concrete deployable target plus credential scope for usage accounting. |
| **Provider** | Optional backend that performs inference or related work when configured and healthy. |
| **Fallback** | Router policy that selects an alternate provider/path after a failed or ineligible primary—not silent catalog mutation. |
| **Extra** | Optional install group in packaging (`pyproject.toml`). Install-time only; not runtime health. |
| **Facade / compatibility surface** | Retained alternate entry (e.g. `mcp` package) that may bridge to the canonical implementation; not the preferred ownership target for new work. |
| **Canonical surface** | Authoritative package or entry for new integrations (e.g. `mcp_server`, unified CLI `ipfs-accelerate`). |
| **Unified CLI** | Hyphenated console script `ipfs-accelerate` (`cli_entry` / `cli`). Not interchangeable with underscore `ipfs_accelerate`. |
| **Direct AI CLI** | Underscore script `ipfs_accelerate` (`ai_inference_cli`). Separate parser and capability set; use its own `--help`. |

---

## Agent supervisor control plane

| Term | Definition |
| --- | --- |
| **Objective heap** | Durable Markdown (or package-backed) goal hierarchy with acceptance and evidence expectations. |
| **Taskboard** | Drainable projection of tasks (`*.todo.md` or queue source). Regenerable under policy; not goal authority. |
| **Proposal** | Model or provider generative output (plan, patch, diagnosis). Schema-checked candidate material only. |
| **Admission** | Fail-closed policy decision that a proposal may be implemented or advanced—not that it is complete. |
| **Evidence / evidence tier** | Typed material bound to a claim. Closed tiers; no silent promotion. |
| **Assurance level** | Re-derived strength of a claim after policy evaluation; cache hits re-derive, they do not invent. |
| **Authoritative completion** | Gate evaluation that may flip task/goal completion; never self-asserted by model text or a free-form flag alone. |
| **Lease** | Time-bounded exclusive claim on a work unit or coordination shard, fenced against stale workers. |
| **Fence / fencing token** | Monotonic token that invalidates progress from expired or superseded owners. |
| **Worktree / lane** | Isolated Git working tree (and process context) for one implementation agent. |
| **Merge train** | Pipeline that lands implementation commits under policy; landing ≠ acceptance. |
| **Rescue / recovery** | Deterministic restart paths after crash or stale lease; reconstructs schedule from durable state, not chat. |
| **Entrypoint (prompt-first)** | Composition surface that binds run identity, storage, and facades without inventing control authority. |
| **Domain package** | Semantic package under `agent_supervisor/` (`objectives/`, `proof/`, `merge/`, …) with DAG ownership rules. |
| **Program / board prefix** | Scheduled work product identity (`ASI-`, `CBP-`, …). Product docs prefer semantic names. |
| **Protected path** | Operator-owned file implementation agents must not edit even if listed in task outputs. |

---

## Distributed identity and P2P

| Term | Definition |
| --- | --- |
| **Backend role** | Declared IPFS/storage role (kit, HF cache, Kubo, …) with explicit selection receipts—not silent degradation. |
| **Verified IPLD backend** | Adapter that rehashes content and refuses cache-role synthetic keys as coordination CIDs. |
| **Pin** | Capability-gated retention request; pin success is not lease authority. |
| **P2P TaskQueue** | Optional peer execution plane with protocol versions, trust tiers, and DuckDB-backed leases. |
| **Peer trust tier** | Policy class for remote peers (e.g. TRUSTED / ELEVATED / BASELINE); ads alone are non-authoritative. |
| **Workflow discovery** | Tagging/finding workflows eligible for P2P vs other runners—not coordination-manifest authority. |

---

## Trust ladder (short form)

```text
import / package present     → vocabulary only
discovery / catalog resolve  → what is declared to exist
capability probe / config    → what this host can attempt
usage reservation            → capacity claimed for a call
router invocation            → work performed (side effects)
merge / landing              → code or artifact in history
typed evidence + gates       → acceptance / proof authority
```

Each step is **fail-closed** when optional: report absence; do not invent
success from a higher-level label or a plan document.

---

## See also

| Need | Document | Status |
| --- | --- | --- |
| Architecture landing routes | [README.md](README.md) | Reference (hub) |
| Document classification | [DOCUMENTATION_MANIFEST.md](../development/DOCUMENTATION_MANIFEST.md) | Current (manifest) |
| Lifecycle and authority | [DOCUMENTATION_LIFECYCLE.md](../development/DOCUMENTATION_LIFECYCLE.md) | Current |
| System shape | [SYSTEM_CONTEXT.md](SYSTEM_CONTEXT.md) | Current |
| Supervisor mental model | [AGENT_SUPERVISOR_PHILOSOPHY.md](AGENT_SUPERVISOR_PHILOSOPHY.md) | Reference |
| ADR index | [decisions/README.md](decisions/README.md) | Reference |
| Board prefixes | [agent_supervisor/PROGRAMS.md](agent_supervisor/PROGRAMS.md) | Reference |
| Documentation refresh sequencing | [DOCUMENTATION_REFRESH_PLAN_2026_08.md](DOCUMENTATION_REFRESH_PLAN_2026_08.md) | **Plan** (not Current product behavior) |
