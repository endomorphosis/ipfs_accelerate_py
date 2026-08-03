# ADR-0003: Separate capability discovery, catalogs, usage accounting, and routing

- **Status:** Accepted
- **Date:** 2026-08-03
- **Last verified:** 2026-08-03
- **Deciders:** architecture maintainers; inference and routing package owners
- **Scope:** How the system separates (1) package import and optional capability
  discovery, (2) catalog service identity and declared readiness, (3) live
  capacity accounting and reservation, and (4) modality-router invocation and
  fallback. Applies to `model_catalog`, `endpoint_usage`, `ModelManager`, and
  the modality routers that perform provider construction and wire calls.
- **Non-goals:** MCP/MCP++ transport and UCAN policy (see `MCP_RUNTIME.md`);
  agent-supervisor admission and leases (see `AGENT_SUPERVISOR_ARCHITECTURE.md`
  and ADR-0004 when published); package install policy and hardware probe
  implementation details beyond the importability rule; sealing new catalog or
  usage delivery plans.
- **Supersedes:** none
- **Superseded-by:** none
- **Related guides:**
  [`docs/architecture/MODEL_SERVICE_ROUTING.md`](../MODEL_SERVICE_ROUTING.md),
  [`docs/architecture/AI_SERVICE_CATALOG.md`](../AI_SERVICE_CATALOG.md),
  [`docs/architecture/ENDPOINT_USAGE_AWARE_ROUTING_PLAN.md`](../ENDPOINT_USAGE_AWARE_ROUTING_PLAN.md)
  (plan status), [`docs/api/overview.md`](../../api/overview.md)
- **Source anchors:**
  `ipfs_accelerate_py/model_catalog/` (`AIServiceCatalog`, schema, snapshot,
  sources), `ipfs_accelerate_py/endpoint_usage/` (`UsageCoordinator`,
  `UsageRouteAdmission`, ledger, controls), `ipfs_accelerate_py/model_manager.py`
  (planning facade), `ipfs_accelerate_py/llm_router.py`,
  `embeddings_router.py`, `multimodal_router.py`, `voice_router.py`,
  `cli_runtime/`, `hf_model_server/`, cold-import and usage routing tests under
  `test/`

## Status meanings (do not invent new values)

| Value | Use when |
| --- | --- |
| Proposed | Decision is under review; **not** yet evidenced current design |
| Accepted | Decision matches current code/tests/ops practice and is normative for Scope |
| Deprecated | Still historical; prefer another practice for new work |
| Superseded | Replaced by the ADR in Superseded-by |
| Rejected | Considered and not adopted; retained to document the negative choice |

Only **Accepted** records are current design authority. **Proposed** records
must not be treated as implemented system law.

## Context

Operators, library consumers, MCP tools, and agent supervisors all need to
answer four different questions about model and service backends:

1. **Is the vocabulary even loadable?** Can this process import catalog and
   usage contracts without installing optional stacks or contacting the
   network?
2. **What exists?** Which providers, models, deployments, and router bindings
   are known under a content-addressed inventory revision, and what
   *declared* operational states do they claim?
3. **What capacity is usable and reserved?** What headroom, cooldowns, and
   concurrent reservations apply to an exact endpoint-plus-credential scope
   under an independent usage revision?
4. **How is work invoked?** Who constructs providers, spends credentials,
   streams or batches, applies fallback policy, and settles observed usage?

These questions have incompatible revision rules, side-effect rules, and
failure modes:

- Catalog identity must stay stable while request counters change every second.
- Listing and resolve must not install software, start processes, load weights,
  open sockets, or reserve quota.
- Multi-tenant capacity needs atomic compare-and-set reservations, fences, and
  idempotent settlement—not advisory health scalars.
- Text, embeddings, multimodal, and voice invocation differ in media policy,
  batching, streaming, and safe fallback; a generic “call endpoint” map cannot
  own those contracts.

Historical pressure toward a **single mutable registry** (model name → live
client + counters + invoke handle) makes demos easy and production unsafe:
cold list can mutate capacity or construct clients; quota churn rewrites
identity; partial refresh mixes half-published inventory with live
reservations; federation cannot pin a content-addressed snapshot; fail-open
reads spend real money.

A related confusion is treating **successful Python import** or **optional
package presence** as proof that a capability is available for production
work. Import only loads vocabulary and factories. Usable capacity and
invocation require configuration, credentials, a published catalog snapshot,
and—when enforcement is on—a successful reservation.

If this separation is ignored, agents and MCP tools cannot discover safely,
pagination and federation lose stable revisions, concurrent workers double-
charge shared accounts, and tests cannot isolate side-effect-free discovery
from network invocation.

## Decision

The system **separates four concerns into distinct planes** and forbids
collapsing them into one mutable registry.

### 1. Importability is not availability

- A successful `import ipfs_accelerate_py.model_catalog` or
  `import ipfs_accelerate_py.endpoint_usage` means only that **contracts and
  schema load**. It does **not** mean providers are installed, credentials
  exist, a catalog snapshot is published, hardware is ready, or any endpoint
  has headroom.
- Optional modality stacks, CLI providers, and local serve paths are **lazy**.
  Discovering that a binding *could* exist must not eagerly import heavy
  routers, start processes, install packages, load model weights, or probe
  the network.
- Declared operational states on catalog records (known / configured /
  authorized / reachable / healthy / routable and related tri-states) answer
  **published claims**, not live quota. Unknown does not satisfy a required
  `true` constraint.
- Non-authoritative signals that must **not** authorize invocation, raise
  quotas, or complete supervisor objectives: bare import success, soft ranking
  scores, response-cache hits, board or chat status, and federated ads without
  local trust policy.

### 2. Information / catalog plane owns service identity

- **Package authority:** `ipfs_accelerate_py/model_catalog/`.
- **Orchestration:** `AIServiceCatalog` with immutable generation swap;
  list/get/resolve never install, start, load weights, or call a provider.
- **Records:** `ProviderDescriptor`, `ModelDescriptor`, `DeploymentDescriptor`,
  `RouterBinding`, and `CatalogSnapshot` under schema `1.0`. Identities derive
  from canonical non-secret fields (`model_catalog.identity`).
- **Revision:** catalog `revision` / content ID of sorted canonical records.
  Request counters and live headroom **must not** churn this revision.
- **Sources:** static, persistent, router, deployment, and peer sources publish
  complete snapshots with provenance and explicit precedence. Failed refresh
  keeps the last generation and reports bounded diagnostics. Side-effecting
  refresh requires explicit `RefreshPolicy` / allowlist authority.
- **Capability descriptors** on records describe declared operations and
  modality claims for planning. They are catalog information, not usage
  counters and not invoke handles.

### 3. Usage / capacity plane owns live accounting and reservation

- **Package authority:** `ipfs_accelerate_py/endpoint_usage/`.
- **Scope identity:** `EndpointUsageScope` (provider, deployment/endpoint
  fingerprint, protocol, operation, optional model, tenant/project
  pseudonyms, credential-scope fingerprint). Unrelated credentials, accounts,
  tenants, or endpoints are never pooled.
- **Reads:** snapshot, headroom, and limit queries are side-effect free; they
  never reserve, probe providers, refresh the catalog, or invoke models.
- **Mutations:** `UsageCoordinator.reserve` atomically claims an estimate
  vector (compare-and-set) with TTL, owner, lease/fence, and idempotency key.
  Commit, release, and conservative settlement of unknown dimensions follow
  only after the owning router’s attempt outcome.
- **Revision:** independent `usage_revision`. Static capabilities and dynamic
  usage **never share a revision**.
- **Modes:** `off` | `observe` | `shadow` | `assist` | `enforce`. Unconfigured
  environments stay effectively off/observe until rollout gates pass. Enforce
  fails closed on missing usage service, exhausted/disabled/unroutable scopes,
  or failed atomic reservation when no alternate remains—unknown is not
  treated as unlimited.

### 4. Invocation plane owns side effects and fallback

- **Authority:** modality routers (`llm_router`, `embeddings_router`,
  `multimodal_router`, `voice_router`) plus `cli_runtime/`, `voice_jobs/`,
  `voice_providers/`, and launchable paths such as `hf_model_server/` when
  projected into catalog and usage.
- Routers own provider construction and credentials at the call site; request
  shape, batching, streaming, timeouts, and typed errors; the estimate → plan
  → **reserve** → invoke → observe → settle lifecycle; and explicit
  `FallbackClass` policy.
- Exact pins default to no admission-level fallback. Each admission candidate
  fallback is a **new** reservation and attempt linked by receipt chain; a
  spent reservation is not reused for another candidate.
- Local HF serve and CLI process spawn remain invocation ownership. They
  become routing candidates only when explicitly projected through catalog
  sources and paired with usage adapters—they are not a parallel identity
  system.

### 5. Planning facade does not close races or invoke

- `ModelManager` (`list_services`, `resolve`, `usage_snapshot`,
  `resolve_for_routing`) is the public **planning facade**. It may hard-filter
  and soft-rank over one catalog revision and one usage revision.
- Planning methods **never reserve**. Catalog-read and MCP catalog tools never
  invoke. Final atomic reservation and wire work belong to
  `UsageRouteAdmission` / the owning router (or equivalent typed adapter).
- Direct-router compatibility paths that synthesize binding/scope identities
  when callers omit catalog candidates preserve legacy APIs; they do **not**
  prove catalog-backed discovery.

### 6. One mutable registry is rejected

A single mutable map that simultaneously owns service identity, live capacity,
and invocation handles is **not** an allowed design. It cannot keep catalog
CIDs stable, discovery side-effect free, reservations atomic across modalities,
and modality-correct fallback contracts honest at the same time. Identity,
capacity, and side effects remain separate authorities with separate tests.

## Alternatives

### Alternative A: One global mutable registry

- **Summary:** One process-wide dict (or similar) maps model/service names to
  live clients, request counters, health flags, and invoke callables. List,
  resolve, reserve, and call all read and write the same structure.
- **Expected benefits:** Minimal API surface for demos; fewer packages to
  learn; one place to “register a model.”
- **Why not chosen:** Violates every plane invariant. A list or import path
  can construct clients, open network, or mutate capacity. Quota and
  concurrency churn rewrites or invalidates identity CIDs needed for
  federation, pagination, and receipts. Partial updates interleave
  half-refreshed providers with live reservations. Fail-open “best effort”
  reads spend real money and credentials. Cross-modality workers double-
  charge shared accounts because there is no single atomic reservation
  authority with typed scopes. Tests cannot prove cold import and listing are
  free of process/network/install/model-load side effects. This alternative is
  explicitly rejected for production routing.

### Alternative B: Catalog-only routing (ignore usage)

- **Summary:** Resolve solely from catalog snapshots; routers invoke the top
  candidate without ledger reservation or headroom.
- **Expected benefits:** Simpler integration; no usage service injection or
  mode rollout.
- **Why not chosen:** Concurrent callers overshoot provider and account
  limits. Multi-window headroom, cooldowns, and credential-scoped quotas
  cannot be expressed. Supervisors and multi-router processes have no shared
  atomic capacity gate.

### Alternative C: Embed usage counters inside catalog records

- **Summary:** Store live request counters, headroom, and cooldowns as fields
  on `ProviderDescriptor` / `DeploymentDescriptor` and bump catalog revision
  (or mutate in place) on every observation.
- **Expected benefits:** One snapshot type for agents; fewer revisions to
  bind in receipts.
- **Why not chosen:** Couples dynamic load to static identity. Peer
  advertisements and pagination cursors become unstable. A changing counter
  rewrites content-addressed inventory. Federation would ship live quota as
  if it were trusted identity. Corrections and superseding events in a usage
  ledger cannot be modeled cleanly.

### Alternative D: ModelManager performs invoke

- **Summary:** The planning facade also constructs providers, holds secrets,
  streams responses, and applies fallback.
- **Expected benefits:** One class for “do AI work”; thinner routers.
- **Why not chosen:** Turns a side-effect-free planning surface into a
  secret-handling, network-facing god object. Breaks modality-specific
  streaming, batching, media policy, and fallback ownership. MCP catalog-read
  tools and agent resolve paths would sit next to invoke authority with a
  blurry trust boundary.

### Alternative E: Router-local quota only

- **Summary:** Each modality router tracks its own in-process rate limits and
  never shares a coordinator.
- **Expected benefits:** No cross-package usage dependency; simpler unit tests
  per router.
- **Why not chosen:** Shared accounts across modalities and supervisor lanes
  double-charge. No single atomic reservation authority, fence, or
  idempotency key for multi-tenant capacity. Observe/shadow/enforce rollout
  cannot be centralized.

### Alternative F: Soft score bypasses hard gates

- **Summary:** Preference, latency, or rank scores may override authorization,
  pins, exhausted limits, or safety reserves when “the best model” is wanted.
- **Expected benefits:** Higher success rates under pressure; simpler UX that
  always “tries something.”
- **Why not chosen:** High preference must not override authorization, explicit
  pins, safety reserves, or capacity denials. Soft ranking applies only after
  hard filters. Enforce mode fails closed rather than inventing headroom.

### Alternative G: Treat successful import as operational readiness

- **Summary:** If `import torch` / import of a router or catalog package
  succeeds, treat the capability as available for production admission.
- **Expected benefits:** Cheap health checks; fewer environment probes.
- **Why not chosen:** Importability is not availability. Optional extras,
  CUDA, credentials, published snapshots, and live headroom are independent.
  Production admission requires configuration and, under enforce, reservation
  success—not merely that a module loaded.

## Consequences

### Positive

- Agents and MCP catalog tools can list and resolve without installing
  software, loading weights, spending quota, or constructing provider clients.
- Catalog CIDs and pagination remain reproducible under load; usage revisions
  move independently for audit and federation.
- Multi-tenant capacity has one reservation authority with typed scopes,
  fences, and receipt chains shared by routers and supervisors.
- Modality-correct invocation and fallback stay with the owners that
  understand streaming, media, and process-backed providers.
- Tests isolate planes: cold-import conformance for catalog; fake clocks and
  in-memory ledgers for usage; router contract tests under fallback.
- Operators can distinguish “package present,” “published in snapshot,”
  “has headroom,” and “reserved for this attempt.”

### Negative

- Callers must learn three packages/planes and two revision identities instead
  of one dict.
- Usage service injection and routing-mode rollout add operational surface and
  migration dual-APIs (legacy inventories vs `model_catalog_*`).
- Enforce mode fail-closed UX is stricter than silent best-effort routing and
  can surface capacity unavailable when older stacks would have guessed.
- Distributed multi-host enforcement needs a fenced reservation backend or
  explicit partitions; eventual consistency alone is insufficient.
- Direct-router compatibility paths that synthesize admission identity can be
  misread as catalog-backed discovery if documentation is careless.

### Neutral / residual risks

- Catalog records may store normalized `endpoint_uri` as part of deployment
  identity; local catalog storage and exports must be access-controlled and
  redacted for peers (fingerprint / federated URI), independent of usage
  ledger redaction rules.
- Optional usage service absent: pure catalog resolve remains; enforce modes
  must not invent unlimited headroom.
- Optional modality not installed: that binding is absent or non-routable;
  other modalities continue—absence must be reported as capability gap, not
  silent success.
- MCP transport, UCAN, and supervisor lease isolation remain out of scope;
  this ADR does not authorize those planes to collapse into the catalog.

## Evidence

| Claim in Decision | Evidence (path, test, or operational check) | Notes |
| --- | --- | --- |
| Catalog contracts are side-effect-free at import | `ipfs_accelerate_py/model_catalog/__init__.py`; `test/test_ai_service_catalog_schema.py::test_cold_import_does_not_start_provider_process_network_install_or_model_load` | Isolated interpreter blocks process/network; routers and torch not imported |
| Listing and tool registration stay side-effect free | `test/test_ai_catalog_conformance.py::test_cold_import_listing_and_tool_registration_are_side_effect_free` | Audit hooks forbid process, socket, credential, and weight loads |
| Catalog orchestration never invokes on list/get/resolve | `ipfs_accelerate_py/model_catalog/catalog.py` (`AIServiceCatalog`); `MODEL_SERVICE_ROUTING.md` plane table | Immutable generation swap; refresh policy for side-effecting sources |
| Planning facade does not reserve | `ipfs_accelerate_py/model_manager.py` (`resolve`, `resolve_for_routing`, `usage_snapshot`); `test/test_model_manager_usage_routing.py` | Usage snapshot/headroom paths asserted side-effect free |
| Atomic reserve before invoke | `endpoint_usage/coordinator.py` (`UsageCoordinator.reserve`); `endpoint_usage/routing.py` (`UsageRouteAdmission`) | CAS reservation; callback only after grant when `invoke=` supplied |
| Usage controls: reads never reserve/probe/invoke | `endpoint_usage/controls.py` module docstring and read path | Admin mutations require separate authority, fence, audit |
| Independent catalog vs usage revisions | `MODEL_SERVICE_ROUTING.md` state/identity table; usage resolution receipts | Static capabilities and dynamic usage never share a revision |
| Routers own invocation and fallback | `llm_router.py`, `embeddings_router.py`, `multimodal_router.py`, `voice_router.py`; usage routing tests | FallbackClass and receipt chain per admission attempt |
| Importability ≠ readiness (API surface) | `docs/api/overview.md` optional capability checks | Torch import vs CUDA/model operation distinction |
| Plane map and rejected registry | `docs/architecture/MODEL_SERVICE_ROUTING.md` rationale and alternatives | Current architecture guide; this ADR is the decision record |

## Verification

From the repository root, offline deterministic gates (no network in default
tests):

```bash
# Catalog plane and cold import
python -m pytest \
  test/test_ai_service_catalog_schema.py \
  test/test_ai_service_catalog_registry.py \
  test/test_ai_service_catalog_sources.py \
  test/test_model_manager_catalog.py \
  test/test_ai_catalog_conformance.py -q

# Usage and usage-aware routing plane
python -m pytest \
  test/test_endpoint_usage_schema.py \
  test/test_endpoint_usage_adapters.py \
  test/test_endpoint_usage_ledger.py \
  test/test_endpoint_usage_routing.py \
  test/test_model_manager_usage_routing.py \
  test/test_llm_router_usage_routing.py \
  test/test_embeddings_router_usage_routing.py \
  test/test_multimodal_router_usage_routing.py \
  test/test_voice_router_usage_routing.py \
  test/test_ai_router_usage_contract.py \
  test/test_endpoint_usage_controls.py \
  test/test_endpoint_usage_conformance.py \
  test/test_endpoint_usage_faults.py \
  test/test_endpoint_usage_rollout.py -q

# Structural invariants readers can re-check without full pytest
rg -n 'side-effect-free|side_effect' ipfs_accelerate_py/model_catalog ipfs_accelerate_py/endpoint_usage
rg -n 'class UsageCoordinator|def reserve' ipfs_accelerate_py/endpoint_usage/coordinator.py
rg -n 'class AIServiceCatalog' ipfs_accelerate_py/model_catalog/catalog.py
test -f docs/architecture/MODEL_SERVICE_ROUTING.md
```

Pass signals:

- Cold-import tests pass with no process/network/install/model-load events.
- Planning and usage snapshot tests do not reserve or invoke.
- Reservation tests require CAS success before invoke callback settlement.
- Catalog and usage packages remain separate modules with independent revision
  fields in receipts.

Fail / stale signals (reopen or supersede this ADR):

- List/resolve/import paths start providers, open network, or write usage.
- Catalog revision incorporates live request counters or headroom fields.
- `ModelManager` gains a production invoke path that bypasses routers and
  admission.
- A new global registry merges identity, ledger mutation, and client handles
  without a superseding ADR and evidence.

## Review triggers

- [ ] Source anchors no longer match the Decision statement
- [ ] Catalog list/get/resolve or cold import gains process, network, install,
      or model-load side effects
- [ ] Usage and catalog revisions are merged or counters rewrite catalog CIDs
- [ ] Planning facade performs reserve or provider invoke in the default path
- [ ] A single mutable registry is reintroduced as the supported integration
      surface
- [ ] Soft ranking is allowed to override hard capacity, pin, or authorization
      gates under enforce mode
- [ ] Related guide ownership (`MODEL_SERVICE_ROUTING.md`, catalog, or usage
      packages) is restructured
- [ ] Distributed reservation backend changes weaken fencing or scope isolation
- [ ] Superseding design is Accepted under a new ADR number

When superseding: create a new ADR number; set this file to **Superseded** with
`Superseded-by`; set the successor’s `Supersedes`; do not delete this file.

## Notes (optional)

- Operational depth, request-flow diagrams, MCP tool names, and extension
  recipes live in `MODEL_SERVICE_ROUTING.md` and `AI_SERVICE_CATALOG.md`. This
  ADR records the normative *why* and the rejected alternatives; it does not
  replace those guides.
- Usage window kinds, rollout gates, and supervisor envelope details remain
  partly in the sealed `ENDPOINT_USAGE_AWARE_ROUTING_PLAN.md`; prefer code and
  conformance tests when plan text and implementation diverge.
- Sibling ADRs in the documentation-refresh wave: ADR-0002 (proposal vs
  evidence admission), ADR-0004 (worktrees/leases/fencing), ADR-0005 (mutable
  coordination vs immutable replication). None of those may collapse catalog
  identity into live capacity or import success into invocation authority.
