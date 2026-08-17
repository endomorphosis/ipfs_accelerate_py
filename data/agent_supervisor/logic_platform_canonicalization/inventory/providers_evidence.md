# LPC-005 Inventory: provider protocols, requests, receipts, and cache keys

Machine-readable companion:
`data/agent_supervisor/logic_platform_canonicalization/inventory/providers_evidence.json`

Interface: `LogicPlatformInventory@1`  
Task: `LPC-005` · Goal: `LPC-G010` · Track: inventory · Lane: `lpc-inventory-provider`

## Source revisions

| Repository | Reviewed baseline | Implementation authority |
| --- | --- | --- |
| `ipfs_datasets_py` | `ac82107e246b30e35a2bbdcf75e01370d22350c6` | `ac82107e246b30e35a2bbdcf75e01370d22350c6` (equals baseline) |
| `ipfs_accelerate_py` | `485edc0871c55b0e2ef21d83bece9fa12c2c8d84` | `ea11293bb996f052d620eae989f5377a956764b1` |

Rule: current heads are implementation authority; reviewed baselines are comparison anchors only.

### Source availability in this worktree

| Location | Available | Note |
| --- | --- | --- |
| `/home/barberb/lift_coding/external/ipfs_datasets` | no | Plan-named datasets checkout missing |
| `ipfs_datasets_py/` submodule | no | Empty directory; LPC scheduler `worktree_submodule_paths` is `[]` |

Classification method: plan contracts, LPC predicted paths, accelerate supervisor import sites, LFV completion receipts, and formal-planning matrix notes. Entries with `source_read=false` still list a generation and a classification; concrete class/schema fields must be re-verified once the datasets tree is mounted.

**Read-only production surfaces (this task):** `logic/backends`, `logic/hammers`, `logic/common/proof_cache`, `logic/ir_core`.

## Classification vocabulary

`canonical`, `canonical_component`, `compatibility_facade`, `legacy`, `experimental`, `declaration_only`, `generated`, `duplicate`, `obsolete`, `unresolved`

## Ownership and invariants

| Authority | Owner |
| --- | --- |
| Request/protocol/receipt/cache identity and translation/verdict meaning | `ipfs_datasets_py.logic` |
| Scheduling, isolation, resources, cancellation, leases, placement, single-flight, admission policy | `ipfs_accelerate_py.agent_supervisor` |

Invariants for this slice:

- Provider success is not semantic success.
- v1 generic payloads cannot bypass `BackendRequest@2`.
- Provider output stays untrusted until validation or reconstruction.
- Advisors propose; they do not prove or raise authority.
- Datasets owns cache-key semantics; supervisor owns placement and single-flight only.
- Operation status, semantic verdict, availability, evidence kind, evidence authority, boundedness, and translation preservation are orthogonal.
- Imports and declaration discovery perform no install, network, process, environment, or write side effects.

## Canonical pipeline tail

```text
LogicObligation@2
  → BackendRequest@2
  → provider execution under supervisor resources
  → typed EvidenceArtifact
  → translation validation / reconstruction / kernel checking
  → TrustedProofReceipt or typed non-proof outcome
  → supervisor admission, validation, merge policy
```

## Coverage summary

All eight required categories are classified:

| Category | Required | Classified |
| --- | --- | --- |
| BackendRequest generations | yes | yes |
| Provider protocol generations | yes | yes |
| Translation contracts | yes | yes |
| Proof-plan contracts | yes | yes |
| Receipts | yes | yes |
| Cache-key types | yes | yes |
| Installer mutation boundaries | yes | yes |
| Status / authority / boundedness enums | yes | yes |

## 1. BackendRequest generations

| ID | Name | Generation | Classification | Path / surface |
| --- | --- | --- | --- | --- |
| `br:backend-request-v2` | **BackendRequest@2** | @2 | **canonical** | `ipfs_datasets_py/logic/backends/requests_v2.py` |
| `br:logic-obligation-v2` | LogicObligation@2 | @2 | canonical_component | `logic/backends` (pipeline stage) |
| `br:backend-request-ir-core` | BackendRequest (ir_core.protocols) | @ir_core | legacy | `ipfs_datasets_py/logic/ir_core/protocols.py` |
| `br:backend-request-v1-generic` | v1 generic payloads | v1 | legacy | retained only via explicit adapters |

Notes:

- Plan preserves `backends.requests_v2` as a strong contract.
- Supervisor backend probe binds `BackendRequest` on `ir_core.protocols`; that symbol is **not** the plan-named `@2` new-write generation until source confirms identity or adapter relationship.
- LPC-051 forbids v1 generic payloads from bypassing `BackendRequest@2`.

## 2. Provider protocol generations

| ID | Name | Generation | Classification | Path / surface |
| --- | --- | --- | --- | --- |
| `pp:logic-provider-v1` | **LogicProvider@1** | @1 live wire | **canonical** | `ipfs_datasets_py/logic/backends/provider.py` |
| `pp:logic-provider-protocol-v2` | LogicProviderProtocol@2 | @2 planned | declaration_only | `.../backends/protocol_v2.py` (LPC-050) |
| `pp:protocol-v1-adapter` | protocol_v1_adapter | v1→v2 | declaration_only | `.../backends/protocol_v1_adapter.py` (LPC-051) |
| `pp:supervisor-proof-provider-protocol-v1` | Supervisor ProofProvider protocol | v1 | compatibility_facade | `formal_verification_provider.py` |
| `pp:supervisor-logic-provider-facade` | SupervisorLogicProviderFacade@1 | facade | compatibility_facade | `logic_provider_contract.py` |
| `pp:hammer-logic-provider` | IpfsDatasetsLogicProvider (Hammer) | Hammer@1 | compatibility_facade | `integrations/ipfs_datasets_logic_provider.py` |
| `pp:hammers-package` | logic.hammers | — | canonical_component | `ipfs_datasets_py/logic/hammers` |

### Operations (shared six-op surface)

`capability`, `translate`, `prove`, `reconstruct`, `verify`, `attest`

Supervisor protocol version is **1** (`PROOF_PROVIDER_PROTOCOL_VERSION`). Datasets live wire is `LogicProvider@1` via `CANONICAL_LOGIC_PROVIDER_MODULE`. LPC-050 adds operation-specific typed requests under `LogicProviderProtocol@2` unless inventory proves an equivalent successor already exists; this inventory treats `provider.py` as the live equivalent wire and `protocol_v2.py` as the planned typed successor module (not yet confirmed present).

## 3. Translation contracts

| ID | Name | Classification | Path / surface |
| --- | --- | --- | --- |
| `tc:logic-translation-receipt-v1` | **LogicTranslationReceipt@1** | **canonical** | `software_verification/receipts.py` |
| `tc:software-verification-translations` | software_verification.translations | **canonical** | `software_verification/translations.py` |
| `tc:supervisor-translation-contract` | TranslationContract (supervisor) | compatibility_facade | `logic_translation_validation.py` |
| `tc:translation-preservation-axis` | Translation preservation axis | declaration_only | `ir_core/axes.py` (LPC-030) |

Supervisor `TranslationClass` values and assurance caps:

| TranslationClass | Maximum assurance (supervisor) |
| --- | --- |
| `exact` | solver_checked |
| `equisatisfiable` | solver_checked |
| `bounded_abstraction` | solver_checked |
| `conservative_approximation` | candidate |
| `heuristic` | unverified |

Adapter maps these onto datasets `PreservationKind` / `TranslationKind` (`exact`/`lossless`, `equisatisfiable`, `bounded`, `conservative`, `heuristic`).

## 4. Proof-plan contracts

| ID | Name | Classification | Path / surface |
| --- | --- | --- | --- |
| `plan:canonical-proof-plan-v1` | CanonicalProofPlan@1 | declaration_only | `tactician/models.py` (LPC-070) |
| `plan:tactician-planner` | tactician.planner | **canonical** | `tactician/planner.py` (LPC-G070 evidence) |
| `plan:supervisor-proof-plan` | ProofPlan (supervisor) | compatibility_facade | `formal_verification_contracts.py` |
| `plan:portfolio-plan` | PortfolioPlan | compatibility_facade | `multi_prover_router.py` |

Rules:

- One plan model; do not add a second tactician.
- Models/advisors may propose; they may not mark proved, raise authority, skip reconstruction, or drop blocking obligations (LPC-071).
- Supervisor may reorder semantically valid lanes but must not rewrite meaning.

Supervisor `ProofStage` values: `translate`, `model_draft`, `solve`, `reconstruct`, `kernel_verify`, `validate`, `attest`, `persist`.

## 5. Receipts

| ID | Name | Classification | Path / surface |
| --- | --- | --- | --- |
| `rc:trusted-proof-receipt` | TrustedProofReceipt | declaration_only | plan pipeline name; module TBD without source |
| `rc:supervisor-proof-receipt` | ProofReceipt (supervisor) | compatibility_facade | `formal_verification_contracts.py` |
| `rc:proof-evidence` | ProofEvidence | canonical_component | same module |
| `rc:logic-translation-receipt-v1` | LogicTranslationReceipt@1 | **canonical** | see translation section |
| `rc:proof-receipt-attestation` | ProofReceiptAttestation@1 | canonical_component | `bridge/proof_receipt_attestation.py` |
| `rc:ten-point-receipt-admission` | Ten-point admission | canonical_component | plan §8 / epic J |

### Ten-point supervisor admission

A proof result may influence completion or merge only when all hold:

1. Receipt is structurally valid.
2. Content identity is valid.
3. Source/tree/environment/policy bindings match.
4. Translation chain is valid.
5. Evidence kind supports the claimed verdict.
6. Authority ceiling is adequate.
7. Required reconstruction or kernel checks passed.
8. It is not stale.
9. It is not simulated.
10. Supervisor policy admits that authority for the requested operation.

## 6. Cache-key types

| ID | Name | Classification | Path / surface |
| --- | --- | --- | --- |
| `ck:canonical-proof-cache-key-v1` | CanonicalProofCacheKey@1 | declaration_only | `common/canonical_cache_key.py` (LPC-080) |
| `ck:datasets-proof-cache` | logic.common.proof_cache | **canonical** | `common/proof_cache.py` |
| `ck:verification-cache-protocol` | VerificationCacheProtocol@1 | **canonical** | `backends/cache_protocol.py` |
| `ck:supervisor-proof-cache-key` | ProofCacheKey (supervisor) | compatibility_facade | `formal_verification_cache.py` |
| `ck:hammers-proof-cache` | hammers.proof_cache | **duplicate** | `hammers/proof_cache.py` |
| `ck:family-local-proof-caches` | DCEC/TDFOL local caches | legacy | family `*proof_cache*` adapters |
| `ck:proof-repository` | proof_repository | declaration_only | `common/proof_repository.py` (LPC-081) |

### Semantic fields required by LPC-G080

Source, expression, formalization, slice, obligation, assumptions, bounds, translation, provider, environment, policy, schema, checker, network policy, evidence kind, authority ceiling.

### Supervisor live key fields (`ProofCacheKey`)

`obligation`, `premises`, `translator`, `solver`, `kernel`, `toolchain`, `theorem_registry`, `policy`, `resource_budget`, `candidate_tree`

Reject: CID-looking non-CIDs, empty digests, default-string unknown objects, missing semantic fields, cross-environment hits, candidate-as-kernel entries. Cache is not a trust root; hits re-derive assurance from evidence.

## 7. Installer mutation boundaries

| ID | Name | Classification | Path / surface |
| --- | --- | --- | --- |
| `inst:toolchains-registry` | VerificationToolchainRegistry@1 | **canonical** | `backends/toolchains.py` |
| `inst:lazy-installer` | external_provers.lazy_installer | canonical_component | `external_provers/lazy_installer.py` |
| `inst:prover-installer-bridge` | prover_installer bridge | canonical_component | `integration/bridges/prover_installer.py` |
| `inst:bounded-tool-runner` | BoundedToolRunner@1 | canonical_component | `backends/process.py` |
| `inst:pure-data-import-boundary` | Pure-data import boundary | canonical_component | package-wide (LPC-061) |
| `inst:supervisor-provider-isolation` | Supervisor isolation boundary | compatibility_facade | `formal_verification_provider.py` |

### Mutation boundary (fail-closed)

Importing contracts, catalog, syntax, formalization, provider protocol, and supervisor adapter **must not**:

- import solvers
- install packages
- open the network
- start processes
- mutate files
- probe hardware
- change environment variables

Installs require **explicit** calls with pins/checksums. Do not mutate system package managers. Runtime discovery, install, and execution stay explicit operations.

## 8. Status / authority / boundedness enums

### Planned canonical axes (LPC-G030 / `ir_core/axes.py`)

| Axis | Interface / name | Classification |
| --- | --- | --- |
| Operation status | LogicOperationStatus@1 | declaration_only |
| Semantic verdict | LogicSemanticVerdict@1 | declaration_only |
| Availability | (axis) | declaration_only |
| Evidence kind | (axis) | declaration_only |
| Evidence authority | (axis) | declaration_only |
| Boundedness | Boundedness@1 | declaration_only |
| Translation preservation | (axis) | declaration_only |

### Live supervisor overlapping enums (legacy / facade)

| Enum | Classification | Role |
| --- | --- | --- |
| `AttemptStatus` | legacy | operation lifecycle |
| `ProofVerdict` | legacy | semantic result |
| `EvidenceKind` | legacy | evidence kind without trust claim |
| `EvidenceAuthority` (proof contracts) | legacy | producing/checking boundary |
| `AssuranceLevel` | legacy | ordered assurance lattice |
| `TranslationClass` | legacy | translation preservation |
| `ResourceBudget` | compatibility_facade | operational bounds carrier |
| `EvidenceAuthority` (goal_quality) | **duplicate** | name collision; non-logic objectives authority |
| Overlapping datasets `VerificationStatus` / support / runtime | **unresolved** | full census needs mounted source (LPC-031) |

### Critical non-inference rule (LPC-032)

A **succeeded** provider response can still carry **unknown/advisory** semantic fields. No code may infer proof authority from operation success alone.

### Assurance lattice (supervisor live)

`unverified` < `candidate` < `solver_checked` < `kernel_verified` < `attested`

## Generation map (quick reference)

```text
BackendRequest@2          → canonical (requests_v2)
LogicProvider@1           → canonical live wire (provider.py)
LogicProviderProtocol@2   → declaration_only successor (protocol_v2)
CanonicalProofPlan@1      → declaration_only (tactician models)
CanonicalProofCacheKey@1  → declaration_only (common/canonical_cache_key)
TrustedProofReceipt       → declaration_only plan name
LogicTranslationReceipt@1 → canonical (software_verification)
Supervisor Proof* types   → compatibility_facade / legacy until axes cutover
```

## Unresolved (not dropped)

1. **Datasets source tree readability** — external checkout and nested submodule unavailable; AST confirmation of `requests_v2`, presence/absence of `protocol_v2`, and exact `TrustedProofReceipt` type path remain open for LPC-008 composition.
2. **`ax:verification-status-overlapping`** — complete datasets `VerificationStatus` / availability / support / runtime enum list deferred to LPC-030/031 with source confirmation. Supervisor live enums above are classified.

## Downstream consumers

| Task | Uses this inventory for |
| --- | --- |
| LPC-030 / LPC-031 / LPC-032 | status/authority/boundedness axes and legacy maps |
| LPC-050 / LPC-051 / LPC-052 | provider protocol v2 and v1 adapter |
| LPC-070 / LPC-071 | proof-plan model and advisor authority |
| LPC-080 / LPC-081 | cache-key contract and proof repository |
| LPC-008 | compose inventory index |

## Acceptance check

BackendRequest generations, provider protocol generations, translation contracts, proof-plan contracts, receipts, cache-key types, installer mutation boundaries, and status/authority/boundedness enums are **all classified** in this document and in `providers_evidence.json`.
