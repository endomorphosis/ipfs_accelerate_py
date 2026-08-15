# LPC-110 SupervisorLogicPlatformClient@1

**Task:** LPC-110 — Implement SupervisorLogicPlatformClient  
**Goal:** LPC-G110  
**Depends on:** LPC-052 (`LogicProviderResponse@2`), LPC-090 (`SupervisorCanonicalLogicAdapter@1`), LPC-100 (`LogicPlatformManifest@1`)  
**Interface:** `SupervisorLogicPlatformClient@1`  
**Module:** `ipfs_accelerate_py.agent_supervisor.proof.logic_platform_client`  
**Path:** `ipfs_accelerate_py/agent_supervisor/proof/logic_platform_client.py`  
**Schema:** `ipfs_accelerate_py/agent-supervisor/logic-platform-client@1`  
**Version:** `1.0.0`  
**Validation:** `python -m pytest test/api/test_supervisor_logic_platform_client.py -q`

## Purpose

Provide **one lazy supervisor-side client** for the datasets logic platform.
Supervisors handshake once against `LogicPlatformManifest@1`, then call typed
operations through this boundary. Semantic identities stay datasets-owned.
The supervisor retains scheduling, isolation, resources, cancellation, leases,
and placement. Callers cannot overclaim authority: request ceilings are checked
against evidence kind before dispatch.

This note freezes `SupervisorLogicPlatformClient@1`. Executable coverage lives
in `test/api/test_supervisor_logic_platform_client.py`.

## Surface

| Symbol | Role |
| --- | --- |
| `SupervisorLogicPlatformClient` | Lazy client; import never loads datasets |
| `ClientRequestContext` | task/tree/policy/plan/budget/network/cancellation/deadline/correlation/evidence/authority bindings |
| `ClientInvocationResult` | Typed wrap of protocol-v2 response + context |
| `ClientReceiptView` | Supervisor receipt projection (untrusted by default) |
| `ClientCounterexampleView` | Public-safe counterexample projection |
| `CacheFreshnessReport` | Cache hit admission outcome |
| `check_authority_overclaim` | Fail-closed ceiling vs evidence-kind gate |
| `get_logic_platform_client` | Process-local singleton helper |

Interface constants:

* `SupervisorLogicPlatformClient@1`
* Task / goal binding: `LPC-110` / `LPC-G110`
* Compatible with manifest adapter list alongside `SupervisorCanonicalLogicAdapter@1`

## Closed operation vocabulary

| Operation | Method | Notes |
| --- | --- | --- |
| `handshake` | `handshake()` | First lazy step; default wheel/no-Git success |
| `catalog` | `catalog()` / `catalog_root()` / `catalog_digest()` | Sealed `CanonicalLogicCatalogSnapshot@1` |
| `formalize` | `formalize(...)` | Builds `FormalizationArtifact@3` |
| `create_slice` | `create_slice(...)` | Builds admitted `DomainLogicSlice@2` |
| `create_obligation` | `create_obligation(...)` | Builds `LogicObligation@2` |
| `create_plan` | `create_plan(...)` | Draft `GoalDirectedProofPlan@1` (proposal only) |
| `discover_capabilities` | `discover_capabilities(...)` | Non-executable; never mints proof authority |
| `invoke` | `invoke(operation, ...)` | Typed `LogicProviderProtocol@2` dispatch |
| `reconstruct` | `reconstruct(...)` | Executable reconstruction under finite bounds |
| `verify` | `verify(...)` | Independent verification under finite bounds |
| `receipt` | `project_receipt(...)` | Receipt view; success ≠ authority |
| `counterexample` | `project_counterexample(...)` | Strips private/raw/source material |
| `cache_freshness` | `build_cache_key` / `check_cache_freshness` / `require_cache_fresh` | Datasets-owned key semantics |

Also: `create_backend_request(...)` elevates obligations into `BackendRequest@2`.

## Request context bindings (LPC-G110)

Every operational call binds or inherits a `ClientRequestContext`:

| Field | Meaning |
| --- | --- |
| `task_id` | Supervisor task identity |
| `tree_id` | Worktree / repository tree identity |
| `policy_id` | Admission / proof policy identity |
| `plan_id` | Optional plan identity |
| `budget` | Resource budget mapping (wall time, memory, steps, …) |
| `network_allowed` | Outbound network grant (default false) |
| `cancellation` | Optional cooperative cancellation snapshot |
| `deadline_unix_ms` | Optional absolute deadline |
| `correlation_id` | Request correlation (auto-minted when empty) |
| `evidence_kind` | Evidence format/category ceiling basis |
| `authority_ceiling` | Requested trust ceiling (cannot overclaim) |

Closed constant: `REQUIRED_CONTEXT_FIELDS`.

### Authority overclaim (fail closed)

| Rule | Behavior |
| --- | --- |
| Ceiling vs evidence kind | `check_authority_overclaim` rejects ceilings above the evidence-kind max |
| Kernel with candidate/model/… | Rejected |
| Reconstruction with non-proof evidence | Rejected |
| Context construction | Overclaim raises `LogicPlatformClientAuthorityError` before dispatch |
| Obligation / backend request | Re-checked; datasets `AuthorityOverclaimError` remains the semantic gate |

## Lazy import contract

1. **Module import is side-effect free.** No `ipfs_datasets_py` import at load.
2. **Construction is side-effect free.** No network, install, probe, or Git.
3. **Handshake is the first datasets load** (unless `require_handshake=False`
   for offline probes of pure client types).
4. **Provider dispatch is injectable.** Pass a protocol-v2 provider (or test
   double) via `provider=`. Unbound clients still support handshake, catalog,
   formalize, slice/obligation/plan, cache, receipt, and counterexample paths.
5. **Vocabulary projection** goes through `SupervisorCanonicalLogicAdapter@1`
   when residual maps are needed; the client does not reintroduce hand maps.

## Handshake

```text
client = SupervisorLogicPlatformClient(provider=...)
result = client.handshake()   # default: require client + adapter @1
assert result.compatible
snapshot = client.catalog()
```

* Default requirements demand `SupervisorCanonicalLogicAdapter@1` and
  `SupervisorLogicPlatformClient@1` in `compatible_adapter_versions`.
* Typed incompatibilities return `HandshakeResult(compatible=False, …)`;
  they do not raise.
* Subsequent operations with `require_handshake=True` raise
  `LogicPlatformClientHandshakeError` until a compatible handshake succeeds.
* Git / sibling layout are never required (LPC-100).

## Typed invocation

Executable operations (`translate`, `prove`, `check`, `reconstruct`,
`verify`, `attest`) require:

1. Positive finite `RequestBounds`
2. Admitted `BackendRequest@2`
3. Context authority that does not overclaim evidence kind
4. Re-admission through `admit_provider_request_v2` before dispatch

Capability is non-executable: no proof authority, bounds optional.

Provider responses normalize to `LogicProviderResponse@2` with **untrusted
default authority** (`advisory`). Operation success never upgrades trust
(LPC-032 / LPC-052).

## Receipts

`project_receipt` builds a `ClientReceiptView` from an invocation result:

* Binds request id, operation, provider, evidence kind/authority, verdict,
  operation status, context, translation/artifact ids
* `simulated=True` marks non-admissible evidence for completion/merge
* Does **not** implement the ten-point LPC-111 admission gate (owned by
  `logic_platform_admission`); it only projects a typed view for that gate

## Counterexamples

`project_counterexample` strips private markers and retains public scalars only:

* witness: `hidden_witness`, `private_witness`, `private_inputs`
* credentials / tokens: `credential`, `access_token`, `api_key`
* dumps: `raw_output`, `prover_output`, `stdout`, `stderr`
* source: `source_excerpt`, `source_code`, `source_text`

Authority defaults to `advisory`. Redacted flag is always true for this view.

## Cache freshness

| API | Role |
| --- | --- |
| `build_cache_key(...)` | `CanonicalProofCacheKey@1` (datasets-owned semantics) |
| `check_cache_freshness(...)` | Report without raise |
| `require_cache_fresh(...)` | Raise `LogicPlatformClientFreshnessError` on failure |

Fail-closed freshness reasons:

* `environment_mismatch` / `cross_environment_hit`
* `stale_entry`
* `unknown_freshness`
* `ttl_expired`
* `simulated_evidence`

Candidate evidence claiming kernel-grade authority is rejected at key build
time (`CandidateAsKernelError` / client authority gate).

## Relationship to neighboring tasks

| Task | Relationship |
| --- | --- |
| LPC-052 | Typed responses with untrusted default authority |
| LPC-080 | Cache-key semantics owned by datasets; client consumes |
| LPC-090 | Lazy adapter for residual vocabulary maps |
| LPC-100 | Handshake / manifest compatibility |
| LPC-111 | Ten-point receipt admission (consumes client receipts) |
| LPC-120+ | Hammer adapters invoke through this client boundary |

## What this task does **not** do

* Does not implement LPC-111 ten-point admission.
* Does not redefine catalog, family, or cache-key semantics in the supervisor.
* Does not claim provider availability or proof authority from catalog presence.
* Does not create another supervisor or a second agent framework.
* Does not perform automatic provider installation or default network access.
* Does not treat operation success as proof.

## File ownership

| Path | Role |
| --- | --- |
| `ipfs_accelerate_py/agent_supervisor/proof/logic_platform_client.py` | `SupervisorLogicPlatformClient@1` |
| `test/api/test_supervisor_logic_platform_client.py` | Acceptance / regression suite |
| `data/agent_supervisor/logic_platform_canonicalization/notes/supervisor_client.md` | This note |

## Acceptance matrix

| Check | Fail-closed behavior | Primary APIs |
| --- | --- | --- |
| Handshake | Typed incompatibilities; ops blocked until compatible | `handshake` |
| Catalog | Sealed snapshot root/digest only | `catalog` |
| Formalization | `FormalizationArtifact@3` + admitted slices | `formalize` |
| Slice / obligation / plan | Typed records; plans are proposals only | `create_slice`, `create_obligation`, `create_plan` |
| Capability | Non-executable; advisory authority | `discover_capabilities` |
| Typed invocation | Bounds + BackendRequest@2 + authority check | `invoke` |
| Reconstruction / verify | Executable under finite bounds | `reconstruct`, `verify` |
| Receipts | Untrusted projection; simulated marked | `project_receipt` |
| Counterexamples | Private material stripped | `project_counterexample` |
| Cache freshness | Env/TTL/stale/simulated fail closed | `check_cache_freshness`, `require_cache_fresh` |
| Authority | Overclaim rejected before dispatch | `check_authority_overclaim`, `ClientRequestContext` |
| Lazy import | No datasets at import/construct | module load, `module_importer` |
