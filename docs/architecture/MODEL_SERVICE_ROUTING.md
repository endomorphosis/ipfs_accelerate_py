# Model, service, and endpoint-usage routing

**Status:** Current

**Audience:** Developers, operators, and agents that discover models, plan
provider capacity, or invoke modality routers

**Scope:** How `model_catalog`, `endpoint_usage`, `ModelManager`, and the four
modality routers separate *what exists*, *what is currently usable*, *what
capacity is reserved*, and *how invocation/fallback occurs*

**Non-goals:** MCP/MCP++ transport and UCAN policy (see
[MCP runtime](MCP_RUNTIME.md)); agent-supervisor admission/leases (see
[Agent Supervisor Architecture](AGENT_SUPERVISOR_ARCHITECTURE.md)); package
install or hardware capability probes; rewriting the sealed delivery plans
for catalog or endpoint usage

**Last verified:** `e559ff0046c639ba1dadabe02ea0ea91d9877e20` (2026-08-03);
package layout, `ModelManager` usage facade methods, and endpoint-usage
coordinator/routing contracts checked against the tree

## Source anchors

| Concern | Primary path / symbol | Notes |
| --- | --- | --- |
| Catalog package | `ipfs_accelerate_py/model_catalog/` | Schema, registry, resolver, snapshot, sources |
| Catalog orchestration | `model_catalog.catalog.AIServiceCatalog` | Immutable generation swap; side-effect-free list/get/resolve |
| Usage package | `ipfs_accelerate_py/endpoint_usage/` | Scope identity, ledger, coordinator, routing |
| Planning facade | `ipfs_accelerate_py/model_manager.py` | `list_services`, `resolve`, `usage_snapshot`, `resolve_for_routing` |
| Atomic reserve | `endpoint_usage.coordinator.UsageCoordinator.reserve` | Closes the selection race before invoke |
| Route admission | `endpoint_usage.routing.UsageRouteAdmission` | Hard filter, rank and reserve; optionally invokes a router-owned callback and settles through the coordinator |
| Usage-aware plan | `endpoint_usage.resolution.resolve_usage_aware` | Pure planning over one catalog + usage revision |
| LLM invocation | `ipfs_accelerate_py/llm_router.py` | Provider construct, request, stream, fallback |
| Embeddings invocation | `ipfs_accelerate_py/embeddings_router.py` | Same plane; batch-aware reservation |
| Multimodal invocation | `ipfs_accelerate_py/multimodal_router.py` | Media-preserving fallback rules |
| Voice invocation | `ipfs_accelerate_py/voice_router.py`, `voice_jobs/`, `voice_providers/` | Jobs and provider adapters under router ownership |
| Local/HF serve path | `ipfs_accelerate_py/hf_model_server/` | Launchable serve path; requires explicit catalog and usage projections before it is routable |
| CLI providers | `ipfs_accelerate_py/cli_runtime/` | Process-backed providers; usage via adapters when structured |
| Catalog detail guide | [AI_SERVICE_CATALOG.md](AI_SERVICE_CATALOG.md) | Schema, sources, security, migration |
| Usage delivery plan | [ENDPOINT_USAGE_AWARE_ROUTING_PLAN.md](ENDPOINT_USAGE_AWARE_ROUTING_PLAN.md) | **Plan** status; rollout modes and windows |

## Context and component map

Routing is not one registry. Three **planes** answer four different questions
with different revision rules, side-effect rules, and failure modes.

```text
Sources (static / persistent / router / deployment / peer)
                    |
            model_catalog
       registry + resolver + snapshot
         "what exists" (immutable revision)
                    |
              ModelManager
     list / get / resolve / usage_* planning
                    |
         +----------+-----------+
         |                      |
  endpoint_usage          modality routers
  ledger + coordinator    llm | embeddings |
  "usable + reserved"     multimodal | voice
         |                      |
         +---- reserve ----> invoke / settle
                            "how invocation
                             and fallback occur"
```

| Plane | Package | Answers | Mutates on read? | Revision |
| --- | --- | --- | --- | --- |
| **Information / capability** | `model_catalog` | What providers, models, deployments, and router bindings are known; declared operations and operational-state claims | No (list/get/resolve never install, start, load weights, or call a provider) | Catalog snapshot CID/`revision` |
| **Usage / capacity** | `endpoint_usage` | What headroom, limits, cooldowns, and in-flight reservations apply to an exact endpoint+credential scope | Snapshot reads are side-effect free; reserve/commit/release mutate the ledger | Independent `usage_revision` |
| **Invocation** | modality routers (+ `cli_runtime`, `voice_*`, backend managers) | Construct providers, send requests, stream, batch, parse errors, apply fallback policy | Yes — network, process, credential use | Not a catalog revision; attempt IDs + receipts |

`ModelManager` is the public **planning facade**. It may filter and rank. It
must not close a capacity race or perform the provider call. The owning
router (or an equivalent typed execution adapter) performs the final atomic
reservation and invocation.

Importability is not availability. A successful `import model_catalog` or
`import endpoint_usage` only means the vocabulary and contracts load. Usable
capacity requires configuration, credentials, a published snapshot, and—when
enforcement is on—a successful reservation.

## Plane ownership: four questions

### 1. What exists? → `model_catalog`

Canonical records (schema `1.0`):

| Record | Meaning |
| --- | --- |
| `ProviderDescriptor` | Service/provider and aggregate capabilities |
| `ModelDescriptor` | Provider-owned model |
| `DeploymentDescriptor` | Served endpoint or local deployment (secret-free identity) |
| `RouterBinding` | Invocation route tying a model/deployment to a modality router |
| `CatalogSnapshot` | Sorted immutable collection; `revision` is content ID of canonical records |

Sources publish complete snapshots with provenance. Higher-precedence field
claims win (router discovery over static seed; active deployment over router
metadata; federated peers cannot override trusted local identities). Failed
refresh keeps the last generation and reports bounded diagnostics.

Lifecycle and operational state on records answer **declared** readiness
(known / configured / authorized / reachable / healthy / routable). Those
tri-states are independent. Unknown does not satisfy a `true` constraint.
They are **not** live quota counters.

### 2. What is currently usable? → catalog state ∩ `endpoint_usage` snapshot

Usability is the intersection of:

1. **Static eligibility** from one catalog revision (operation, modality,
   policy, device, locality, explicit pins, operational-state constraints).
2. **Dynamic availability** from one usage snapshot per
   `EndpointUsageScope`: `available`, `near_limit`, `exhausted`,
   `cooling_down`, `stale`, `unknown`, `disabled`, or `unroutable`.

`ModelManager.usage_snapshot`, `list_usage_limits`, and
`get_endpoint_headroom` read ledger material without reserving. When no usage
service is configured, `resolve` continues as pure catalog resolution;
enforcing routing modes fail closed rather than treating unknown as unlimited.

### 3. What capacity is reserved? → `endpoint_usage` ledger + coordinator

Capacity is owned by an `EndpointUsageScope` (provider, deployment or
endpoint fingerprint, protocol, operation, optional model, tenant/project
pseudonyms, credential-scope fingerprint). Limits are typed dimensions
(requests, tokens, media units, concurrency, cost, …) with window kinds—not
a single fictional universal token.

`UsageCoordinator.reserve` atomically claims the estimate vector against the
current snapshot (compare-and-set). A reservation carries TTL, owner,
lease/fence, and idempotency key. Only after a successful reserve may the
router invoke that binding. Settlement commits provider-observed usage,
releases unused capacity, or conservatively settles unknown dimensions.
Cancellation before dispatch releases fully; after dispatch, adapters decide
settlement because the provider may still charge.

### 4. How does invocation and fallback occur? → modality routers

Routers own:

- provider construction and credentials at the call site;
- request shape, batching, streaming, timeouts, and provider-specific errors;
- the estimate → plan → **reserve** → invoke → observe → settle lifecycle;
- explicit `FallbackClass` policy (`none`, `same_deployment`,
  `same_provider`, `same_model`, `equivalent_model`, `cross_provider`).

Exact pins default to `none`. Each admission-level candidate fallback is a
new attempt with a new reservation linked by receipt chain; a spent
reservation is not reused for another candidate. Provider-internal model
retries can occur inside one router callback and therefore inside one
admission reservation; the router trace records that internal route when the
provider exposes its internal model selection. Semantic
or client errors and unsafe side-effecting failures do not cross admission
fallback boundaries unless policy classifies them as safe. Wait versus
reroute shares one deadline and max-attempt bound.

Voice work uses `voice_router` plus `voice_jobs/` and `voice_providers/` under
the same plane rules (audio-second/character dimensions, stream partial
settlements). CLI-backed providers under `cli_runtime/` participate when
structured usage or reset metadata is available through adapters; process
spawn remains router/CLI ownership, not catalog ownership. Local Hugging Face
serve paths under `hf_model_server/` are launchable services, not automatic
catalog sources. They become routing candidates only when explicitly
projected through `DeploymentCatalogSource`,
`ServedEndpointDeploymentSource`, or `BackendDeploymentSource` and paired
with the appropriate usage adapter/limits; they are not a parallel identity
system.

## Primary request flow

```text
Caller (API / CLI / MCP tool / supervisor adapter)
        |
 1. Normalize request + conservative UsageEstimate
        |
 2. ModelManager.resolve / resolve_for_routing
    |   catalog revision N  +  usage revision U
    |   hard gates then soft rank
        |
 3. Owning modality router calls UsageRouteAdmission
    |   admission atomically reserves through UsageCoordinator
    |   on CAS fail / capacity deny → next candidate or wait
        |
 4. Admission invokes the router-owned callback for the reserved binding
    |   (or returns the reservation when no callback was supplied)
        |
 5. Router/provider adapter performs network/process work and parses usage,
    |   limits, Retry-After and typed errors
        |
 6. Admission settles/cancels through UsageCoordinator from callback outcome;
    |   emit UsageRoutingReceipt
        |
 7. If admission fallback is permitted and safe → new reservation/attempt
        |
 8. Return result + bounded receipt (no prompt/media/secrets)
```

Planning methods (`resolve`, `resolve_for_routing`) never reserve. Preview
and MCP `model_catalog_*` read tools never invoke. Invocation authority
(`ai.catalog/invoke` and router entrypoints) is separate from catalog read
and catalog refresh.

Catalog-first planning is the preferred integration path, but it is not the
only compatibility path in the current routers. When callers omit
`usage_candidates`, LLM and embeddings routing can construct a direct provider
candidate and synthesize the binding, scope and catalog-revision identities
needed by admission. That path preserves existing direct-router APIs; it does
not prove that the provider was published by `ModelManager` or a catalog
source.

### Catalog resolution (exists)

```python
from ipfs_accelerate_py.model_manager import get_default_model_manager

manager = get_default_model_manager()
page = manager.list_services(operation="text.chat", limit=25)
result = manager.resolve(operation="text.chat", routable=True)
if result.found:
    binding = result.candidates[0].binding
```

### Usage-aware plan (usable, not reserved)

```python
# When a usage service is injected on ModelManager:
plan = manager.resolve_for_routing(
    operation="text.chat",
    usage_request=usage_request,
    routing_policy=routing_policy,  # modes: off|observe|shadow|assist|enforce
)
# plan binds catalog_revision + usage_revision; does not reserve
```

### Reserve then invoke (capacity + side effects)

The router (not `ModelManager`) calls the shared admission protocol in
`endpoint_usage.routing`: hard-filter remaining candidates, soft-rank, then
reserve through `UsageCoordinator`. Only typed denials such as capacity,
reservation conflict, stale snapshot, circuit open, pin, or fallback boundary
advance the candidate cursor. With `invoke=` supplied, `UsageRouteAdmission`
calls the router-owned callback only after reservation and then commits or
cancels through the coordinator from its `InvokeOutcome`. Without a callback,
admission returns the granted reservation before dispatch and the caller owns
the later invocation and settlement. Provider construction, request shaping
and wire execution remain in the modality router or adapter in both forms.

## State and identity

| Identity | Plane | Stability rule |
| --- | --- | --- |
| `provider_id` / `model_id` / `deployment_id` / `binding_id` | Catalog | Derived from canonical non-secret fields (`model_catalog.identity`) |
| Catalog `revision` | Catalog | Content ID of sorted canonical records; request counters must not churn it |
| `EndpointUsageScope` / scope ID | Usage | Endpoint + credential-scope + operation (and model when the provider scopes that way) |
| `usage_revision` | Usage | Materialized snapshot of limits, headroom, reservations |
| Reservation / event / receipt IDs | Usage | Content-addressed events; corrections reference supersedees |
| Attempt / idempotency keys | Invocation | Link fallback chains; prevent double charge |

Static capabilities and dynamic usage **never share a revision**. A changing
request counter must not rewrite a catalog CID. Federated catalog ads carry
compact revision digests, not live quota.

Usage events, routing receipts and public diagnostics omit prompts, input
media, model output, credentials, bearer URLs and raw private endpoints.
Credential fingerprints are keyed local pseudonyms, not stored secrets.
Catalog records have a different trust boundary: `DeploymentDescriptor`
intentionally stores a normalized `endpoint_uri` as part of deployment
identity. Local catalog storage and catalog-read authority must therefore be
access-controlled, and exports must redact or transform private endpoints;
the peer source replaces an endpoint with a federated URI and a non-secret
fingerprint before re-export.

## Routing modes and rollout posture

Usage-aware routing modes (see `RoutingMode` and the endpoint-usage plan):

| Mode | Selection effect |
| --- | --- |
| `off` | Legacy router selection; no usage overlay |
| `observe` | Collect estimates/observations; do not change selection |
| `shadow` | Compute decisions beside legacy routing; compare, do not enforce |
| `assist` | May surface alternate candidates under policy |
| `enforce` | Hard deny on exhausted/disabled/unroutable (and policy for stale/unknown) |

Default posture for unconfigured environments is effectively off/observe
until conformance and rollout gates pass. Changing mode, fallback class, or
safety reserve changes policy identity. Rollback returns selection to prior
router behavior while preserving the ledger for diagnosis.

## Trust, authorization, and failure semantics

### Fail-closed conditions

- Ambiguous catalog alias lookup (no silent pick).
- Cursor/page from a mismatched catalog or usage revision.
- Enforcement mode with missing usage service, exhausted scope, disabled or
  unroutable endpoint, or failed atomic reservation when no alternate remains.
- Cross-scope pooling of unrelated credentials, accounts, tenants, or
  endpoints (never combined).
- Unknown treated as unlimited under enforce (policy may allow only with an
  independent conservative ceiling).
- Provider response or federated catalog record granting quota authority
  (observations settle usage; they do not raise policy ceilings alone).
- Secret-bearing keys in catalog or usage public records.

### Degradation

- Optional usage service absent: pure catalog resolve remains; enforce modes
  surface capacity unavailable rather than inventing headroom.
- Optional modality or CLI provider not installed: that binding is absent or
  non-routable; other modalities continue.
- Stale usage snapshot: enforce may deny or wait; shadow/observe record
  diagnostics and keep legacy selection.
- Source refresh failure: last immutable catalog generation stays readable.

### Recovery

- Explicit named catalog refresh under `RefreshPolicy` (side-effecting sources
  require allowlist authority).
- Wait until `next_eligible` / reset within deadline, or reroute under
  `FallbackClass`.
- Reservation TTL reclamation and crash recovery via ledger store protocols.
- Supervisor paths may return typed `usage_capacity_unavailable` and apply
  backpressure without treating usage receipts as task-completion proof.

### Non-authoritative signals

Import success, board status, chat logs, response-cache hits, and soft ranking
scores do not authorize invocation, raise quotas, or complete supervisor
objectives. Cache hits must not fabricate provider settlements.

## Rationale

The design exists so four conflicting concerns stay independently correct:

1. **Discovery without side effects** — Agents and MCP tools must list and
   resolve without installing software, loading weights, or spending quota.
2. **Stable identity under load** — Catalog CIDs must remain reproducible for
   federation, pagination, and receipts while request counters change every
   second.
3. **Atomic multi-tenant capacity** — Concurrent workers need compare-and-set
   reservations, fences, and idempotent settlement; advisory health scalars
   are not enough.
4. **Modality-correct invocation** — Text, embeddings, vision, and voice differ
   in batching, streaming, media policy, and safe fallback; one generic
   “call endpoint” registry cannot own those contracts.

Separating planes keeps each revision, authority, and test suite honest:
catalog conformance blocks process/network on cold list; usage tests use fake
clocks and in-memory ledgers; router tests assert output contracts under
fallback.

## Alternatives

| Alternative | Why rejected |
| --- | --- |
| **One global mutable registry** holding identity, live counters, and invoke handles | Violates every plane invariant: a list call could mutate capacity or construct clients; quota churn would rewrite identity CIDs; partial updates would mix half-refreshed providers with live reservations; federation could not pin a content-addressed inventory; fail-open reads would spend real money; tests could not isolate side-effect-free discovery from network. |
| **Catalog-only routing** (ignore usage) | Overshoots provider limits under concurrency; cannot express multi-window headroom or credential-scoped quotas. |
| **Usage counters embedded in catalog records** | Couples dynamic load to static revision; breaks peer advertisements and cursor stability. |
| **ModelManager performs invoke** | Turns a planning facade into a secret-handling, network-facing god object; breaks modality-specific fallback and streaming ownership. |
| **Router-local quota only** | Double-charges shared accounts across modalities and supervisor lanes; no single atomic reservation authority. |
| **Soft score bypasses hard gates** | High preference would override authorization, pins, safety, or exhausted limits. |

A simpler single map of “model name → client” is attractive for demos and
catastrophic for production: it cannot distinguish *declared*, *configured*,
*authorized*, *with headroom*, and *already reserved*, and it cannot prove
that a cold list did not start a process.

## Consequences

**Positive**

- Clear ownership tables for docs, MCP tools, and tests.
- Side-effect-free discovery and planning for agents.
- Independent catalog and usage revisions for federation and audit.
- Explicit fallback classes and receipt chains for operators.
- Supervisor and routers can share one coordinator without sharing invoke code.

**Negative / cost**

- Callers must learn three planes and two revisions instead of one dict.
- Usage service injection and mode rollout add operational surface.
- Dual APIs during migration (`list_models` vs `list_catalog_models`, legacy
  MCP names vs `model_catalog_*`).
- Enforce mode fail-closed UX is stricter than silent best-effort routing.
- Distributed multi-host enforcement needs a fenced reservation backend or
  explicit partitions; eventual consistency alone is insufficient.

## Extension and compatibility

**Add a provider**

1. Implement invocation in the owning router (and CLI adapter if process-backed).
2. Publish side-effect-free descriptors and `RouterBinding`s into catalog sources.
3. Register usage adapters for response/error shapes when limits apply.
4. Add fake-provider discovery, resolution, and invocation tests; keep cold
   list free of network and process side effects.

**Add a catalog source**

Implement bounded metadata `load()` with explicit precedence and provenance.
Mark side-effecting refresh; never put request logic in the source or catalog.

**Compatibility**

Legacy inventories and MCP discovery tools remain projections. Canonical IDs
and catalog revision are authoritative. See
[AI_SERVICE_CATALOG.md](AI_SERVICE_CATALOG.md) migration table. Usage methods
on `ModelManager` are no-ops or fail closed depending on mode when the service
is absent; they never write dynamic fields into static catalog records. The
direct-router compatibility path described above can synthesize admission
identity when no catalog candidate list is supplied, but should not be
presented as catalog-backed discovery.

## Operational signals

| Signal | Where | Use |
| --- | --- | --- |
| Catalog metrics | `catalog_*` (latency, cache, conflicts, no-match, health transitions) | Source health and resolution quality |
| Usage metrics | Reservation attempts/denials, headroom bands, fallbacks, stale scopes | Capacity and enforcement health |
| `UsageRoutingReceipt` | Attempt chain, revisions, denial codes, settlement | Incident diagnosis (redacted) |
| Catalog selection receipts | Filters, ranking, chosen binding, revision | Why a binding was preferred |
| MCP tools | `model_catalog_list_*`, `model_catalog_resolve`, usage control tools when enabled | Operator and agent surfaces |

Never label metrics with prompts, raw endpoints, credentials, or unbounded
model alias strings.

## Verification

Offline, deterministic gates (no network in default tests):

```bash
# Catalog plane
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
```

Review checks for this guide:

- Readers can name which component answers exists / usable / reserved / invoke.
- `model_catalog` and `endpoint_usage` are both present and role-separated.
- Rationale includes why one global mutable registry is rejected.
- Live paths in Source anchors exist in the tree.
- MCP transport and UCAN details remain owned by [MCP runtime](MCP_RUNTIME.md),
  not duplicated here.

## Related guides

| Document | Role |
| --- | --- |
| [AI_SERVICE_CATALOG.md](AI_SERVICE_CATALOG.md) | Deep catalog schema, sources, security, migration |
| [ENDPOINT_USAGE_AWARE_ROUTING_PLAN.md](ENDPOINT_USAGE_AWARE_ROUTING_PLAN.md) | Sealed **plan** for usage windows, rollout, supervisor envelopes |
| [overview.md](overview.md) | Repository-wide runtime boundaries |
| [GUIDE_CONVENTIONS.md](GUIDE_CONVENTIONS.md) | Architecture guide contract |
| [docs/LLM_ROUTER.md](../LLM_ROUTER.md) | LLM router operator surface |
| [docs/MCP_SERVER.md](../MCP_SERVER.md) | MCP tool names including `model_catalog_*` |
| Planned ADR DOC-017 | Capability vs catalog vs usage vs routing decision record |

---

*Plane summary for quick navigation:* **`model_catalog`** says what exists;
**`endpoint_usage`** says what is usable and what is reserved;
**modality routers** perform invocation and policy-bounded fallback; a single
global mutable registry cannot own all three without corrupting identity,
capacity accounting, or side-effect boundaries.
