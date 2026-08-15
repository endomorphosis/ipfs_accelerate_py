# LPC-110 SupervisorLogicPlatformClient@1

**Task:** LPC-110 — Implement SupervisorLogicPlatformClient  
**Goal:** LPC-G110  
**Depends on:** LPC-052 (typed provider responses), LPC-090 (supervisor maps),
LPC-100 (`LogicPlatformManifest@1` handshake)  
**Interface:** `SupervisorLogicPlatformClient@1`  
**Evidence module:** `ipfs_accelerate_py/agent_supervisor/proof/logic_platform_client.py`  
**Validation:** `python -m pytest test/api/test_supervisor_logic_platform_client.py -q`

## Summary

LPC-110 provides **one lazy supervisor-side client** for the datasets logic
platform. The client handshakes first, then exposes typed operations for
catalog access, formalization, slice/obligation/plan creation, capability
discovery, provider invocation (including reconstruction and verification),
receipt projection, counterexamples, and cache freshness.

The supervisor still owns scheduling, isolation, resources, cancellation, and
admission policy. Datasets still owns semantic identity. This client is an
adapter boundary, not a second supervisor and not a second semantic registry.

Owned deliverables for this task:

* note: `data/agent_supervisor/logic_platform_canonicalization/notes/supervisor_client.md`
* client: `ipfs_accelerate_py/agent_supervisor/proof/logic_platform_client.py`
* tests: `test/api/test_supervisor_logic_platform_client.py`

## Surface

| Symbol | Role |
| --- | --- |
| `SupervisorLogicPlatformClient` | Lazy client (`SupervisorLogicPlatformClient@1`) |
| `LogicPlatformClientRequest` | Bound request envelope with required axes |
| `LogicPlatformClientResult` | Typed operation result (lifecycle ≠ verdict) |
| `CacheFreshnessStatus` | `current` / `stale` / `unknown` / `miss` / `invalidated` |
| `check_authority_overclaim` | Fail-closed authority vs evidence-kind gate |
| `get_logic_platform_client` | Constructor helper |

Interface / schema constants:

* `SupervisorLogicPlatformClient@1`
* `ipfs_accelerate_py/agent-supervisor/logic-platform-client@1`
* `ipfs_accelerate_py/agent-supervisor/logic-platform-client-request@1`
* `ipfs_accelerate_py/agent-supervisor/logic-platform-client-result@1`
* Task / goal binding: `LPC-110` / `LPC-G110`

## Closed operation vocabulary

| Client operation | Behavior |
| --- | --- |
| `handshake` | Package-neutral `LogicPlatformManifest@1` compatibility check |
| `catalog` | Sealed catalog root/digest + adapter vocabulary inventory |
| `formalize` | Build `FormalizationArtifact@3` candidate bound to the request |
| `create_slice` | Admit `DomainLogicSlice@2` from exact source/expression identity |
| `create_obligation` | Admit `LogicObligation@2` from a slice + finite bounds |
| `create_plan` | Draft `GoalDirectedProofPlan@1` (never claims proof/completion) |
| `discover_capabilities` | Declarative capability inventory; availability ≠ proof |
| `invoke` | Typed provider ops via `SupervisorLogicProviderFacade` |
| `reconstruct` | Convenience wrapper for provider `reconstruct` |
| `verify` | Convenience wrapper for provider `verify` |
| `receipt` | Project translation/validation receipts without authority upgrade |
| `counterexample` | Canonical `FormalCounterexample` (never a positive proof) |
| `cache_freshness` | Evaluate current/stale/unknown/invalidated freshness |

Typed provider operations accepted by `invoke`:

`capability`, `translate`, `prove`, `reconstruct`, `verify`, `attest`.

## Required request bindings

Every `LogicPlatformClientRequest` binds **all** of the following axes
(LPC-G110 acceptance). Missing axes fail closed at construction.

| Axis | Field | Notes |
| --- | --- | --- |
| Task | `task_id` | Supervisor task identity |
| Tree | `tree_id` | Worktree / candidate tree identity |
| Policy | `policy_id` | Admission / proof policy identity |
| Plan | `plan_id` | Plan identity (may be draft) |
| Budget | `resource_budget` | Supervisor `ResourceBudget` |
| Network | `network_allowed` | Cannot exceed budget network flag |
| Cancellation | `cancellation` | `CancellationToken` or null |
| Deadline | `deadline_unix_ms` | Non-negative integer or null |
| Correlation | `correlation_id` | Cross-surface correlation |
| Evidence | `evidence_kind` | Closed evidence vocabulary |
| Authority | `authority_ceiling` | Cannot exceed evidence-kind support |

`binding_digest` is derived from the binding axes (not caller-supplied).

### Authority overclaim rules

1. Evidence kind `candidate` may claim at most `candidate`.
2. `kernel` / `reconstruction` cannot be claimed from advisory/candidate/model/
   trace/monitor/attack/parse evidence.
3. Unknown evidence kinds default to max ceiling `advisory`.
4. Provider-claimed authority in a success payload never upgrades the client
   result ceiling above the request binding.
5. Plans, receipts, and counterexamples are forced to candidate authority on
   the client result envelope.

## Lazy import and handshake

1. Importing `logic_platform_client.py` never imports `ipfs_datasets_py`.
2. `handshake()` is the first lazy datasets step. Default requirements pin
   `SupervisorLogicPlatformClient@1` as a required adapter version.
3. With `require_handshake=True` (default), catalog / formalize / slice /
   obligation / plan / capability / invoke / receipt / counterexample /
   cache_freshness fail closed until a compatible handshake is recorded.
4. Git, sibling repos, and checkout layout are **not** required (LPC-100).

## Operation contracts

### Handshake

* Delegates to `ipfs_datasets_py.logic.platform.manifest.handshake`.
* Compatible results store the platform manifest on the client.
* Typed incompatibilities return `HandshakeResult(compatible=False, …)` without
  raising; structural errors still raise from the manifest module.

### Catalog

* Reads `DEFAULT_CANONICAL_CATALOG_SNAPSHOT` content root/digest.
* Includes adapter vocabulary inventory and canonical family ids.
* Status is `declarative`. Catalog presence is not provider availability and
  not proof authority. Cache freshness is reported `current` for the sealed
  snapshot identity.

### Formalization / slice / obligation / plan

| Step | Interface | Client guarantee |
| --- | --- | --- |
| formalize | `FormalizationArtifact@3` | Candidate artifact; binds task/tree/policy/plan metadata |
| create_slice | `DomainLogicSlice@2` | Admitted only with exact digests and typed namespaces |
| create_obligation | `LogicObligation@2` | `from_slice` + finite bounds; authority checked |
| create_plan | `GoalDirectedProofPlan@1` | `proof_claimed=false`, `completion_claimed=false` |

### Capability discovery and typed invocation

* Discovery is declarative and may use a bound provider facade and/or
  `LogicVerificationAPI.list_providers`.
* `availability_is_not_proof=true` is always present on capability results.
* Invocation builds a supervisor `ProviderRequest` that carries the binding
  digest and request axes in the payload, then dispatches through
  `SupervisorLogicProviderFacade`.
* Success results set `authority_upgraded=false` and retain the request
  authority ceiling.

### Receipts

* Projects through `SupervisorCanonicalLogicAdapter.project_translation_validation_receipt`.
* Projected receipt authority remains `none`; `proof_success` is never set
  true by the adapter projection.

### Counterexamples

* Normalized through `formal_counterexamples.normalize_counterexample`.
* Result always carries `is_proof=false`.
* Bindings include task / plan / tree / policy ids from the request.

### Cache freshness

Closed statuses:

| Status | Meaning |
| --- | --- |
| `current` | Within TTL and not invalidated |
| `stale` | Age exceeds TTL |
| `unknown` | Missing stored_at/ttl (fail closed for admission) |
| `miss` | Repository reports no records |
| `invalidated` | Explicit invalidation |

Optional `repository.freshness(cache_key)` path projects repository reports
onto the same closed vocabulary.

## Relationship to neighboring tasks

| Task | Relationship |
| --- | --- |
| LPC-052 | Provider responses remain untrusted by default; client does not upgrade |
| LPC-090 | Catalog/vocabulary identity flows through `SupervisorCanonicalLogicAdapter@1` |
| LPC-100 | Handshake is the first client step |
| LPC-111 | Ten-point receipt admission for completion/merge (follow-on) |
| LPC-080 / LPC-081 | Cache key / proof repository freshness may be consulted |

## What this task does **not** do

* Does not implement the ten-point receipt admission gate (LPC-111).
* Does not redefine catalog, family, evidence, or authority vocabularies.
* Does not create another agent supervisor or scheduler.
* Does not claim provider availability, proof success, or production readiness
  from handshake, catalog presence, or transport success.
* Does not allow free-form family/payload routing into BackendRequest@2.
* Does not treat Git alignment as semantic compatibility.

## Implementation notes (production module)

Module path:
`ipfs_accelerate_py/agent_supervisor/proof/logic_platform_client.py`

Constructor:

```text
SupervisorLogicPlatformClient(
    adapter=None,                 # default get_canonical_logic_adapter()
    provider_facade=None,         # optional SupervisorLogicProviderFacade
    require_handshake=True,       # ops fail closed until compatible handshake
    datasets_importer=None,       # injectable import hook for tests
)
```

Helper: `get_logic_platform_client(**kwargs)`.

Result envelope (`LogicPlatformClientResult`):

* `ok` / `status` are lifecycle only (`ok`, `failed`, `unavailable`,
  `rejected`, `declarative`, `incompatible`) — never a proof verdict.
* `authority_upgraded` is hard-false at construction.
* `availability_is_not_proof` is always true.
* Plans / receipts / counterexamples force candidate authority on the envelope.

`handshake()` dual return:

* With `request=…` → `LogicPlatformClientResult`.
* Without request → raw datasets `HandshakeResult` (typed compatibility only).

## File ownership

| Path | Role |
| --- | --- |
| `ipfs_accelerate_py/agent_supervisor/proof/logic_platform_client.py` | Client implementation |
| `test/api/test_supervisor_logic_platform_client.py` | Regression suite |
| `data/agent_supervisor/logic_platform_canonicalization/notes/supervisor_client.md` | This note |

## Acceptance matrix

| Check | Fail-closed behavior | Primary APIs |
| --- | --- | --- |
| Handshake | Incompatible adapters / versions typed-fail; ops require handshake | `handshake` |
| Catalog | Sealed root/digest only; declarative status | `catalog` |
| Formalization | Artifact construction errors → failed result | `formalize` |
| Slice / obligation / plan | Missing digests/bounds/overclaim → fail | `create_slice`, `create_obligation`, `create_plan` |
| Capability discovery | Availability never equals proof | `discover_capabilities` |
| Typed invocation | Unknown ops rejected; missing facade → unavailable | `invoke`, `reconstruct`, `verify` |
| Receipts | No proof_success / authority upgrade | `receipt` |
| Counterexamples | Always `is_proof=false` | `counterexample` |
| Cache freshness | Stale/unknown/invalidated explicit | `cache_freshness` |
| Request binding | Missing axis or network overclaim rejected | `LogicPlatformClientRequest` |
| Authority | Evidence/ceiling overclaim rejected | `check_authority_overclaim` |

## Validation command

```text
python -m pytest test/api/test_supervisor_logic_platform_client.py -q
```

Green validation means the client surface above is exercised hermetically with
stub providers where needed. Live prover availability is never required for
this gate and must not be inferred from a green suite.
