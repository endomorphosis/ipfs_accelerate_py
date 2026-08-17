# LPC-091 Classify leftover supervisor semantic types

**Task:** LPC-091 — Classify leftover supervisor semantic types  
**Goal:** LPC-G090  
**Depends on:** LPC-090 (`supervisor_map_cutover.md`), LPC-006 (`supervisor_semantics` inventory)  
**Track:** supervisor-maps  
**Adapter boundary:** `ipfs_accelerate_py.agent_supervisor.proof.canonical_logic_adapter`  
**Interface:** `SupervisorCanonicalLogicAdapter@1`  
**Schema:** `ipfs_accelerate_py/agent-supervisor/supervisor-type-classification@1`  
**Validation:** `test -f data/agent_supervisor/logic_platform_canonicalization/notes/supervisor_type_classification.md`

## Purpose

After LPC-090 replaced hand-maintained supervisor family / property / form /
translation / provider / evidence / cache maps with catalog-projected, residual
rows, every leftover **public** supervisor semantic type still needs an explicit
classification: who owns the meaning, whether the type may remain as a wire
residual, and what migration path applies before any public removal.

This note freezes those decisions. It does **not** delete or rename public
types. Public-type removals require a later migration task that cites a row
here.

## Classification vocabulary

Every leftover type is classified as exactly one of:

| Classification | Meaning |
| --- | --- |
| **duplicate_legacy** | Supervisor-local type that parallel-owns a datasets semantic axis. Retained only as a residual public surface; meaning projects through `SupervisorCanonicalLogicAdapter@1` (or the provider facade). New records write canonical identities. |
| **compatibility_alias** | Spelling or retained name of another live supervisor type. No independent authority. |
| **operational_retain** | Supervisor-owned scheduling, isolation, routing, placement, matrix, capability probe, or control concern. Not a datasets logic-family authority. |
| **reject_merge** | Homonymous non-logic enum. Must never project onto family / property / evidence / verdict axes. |
| **canonical_projection** | Adapter or facade that already consumes datasets contracts as authority. |

### Lifecycle disposition (removal policy)

| Disposition | Allowed action |
| --- | --- |
| `active_legacy` | Keep public type. Reads project via adapter residual. Writes of new durable records use catalog ids. |
| `compatibility_alias` | Keep alias spelling; prefer the primary residual on new code. |
| `operational_retain` | Keep indefinitely under supervisor ownership. |
| `reject_merge` | Keep domain-local type; never map as logic authority. |
| `migrate_then_remove` | Public removal only after a dedicated migration task replaces residual consumers and documents the cutover. |
| `already_projected` | No further semantic ownership decision; adapter is the authority boundary. |

### Fail-closed rules

1. **No silent authority promotion.** Mapping a residual never upgrades
   evidence authority or invents a conclusive verdict.
2. **No silent identity collapse.** Distinct supervisor members that share a
   canonical id keep residual reverse maps (LPC-090).
3. **No public removal without a path.** A `migrate_then_remove` row is a
   prerequisite, not a removal itself.
4. **Reject-merge never projects** onto catalog family/property/evidence axes.
5. **Catalog presence ≠ availability / proof.** Classification never claims
   live prover usability or production admission.

## Ownership reminder

| Authority | Owner |
| --- | --- |
| Family, property, form, translation, evidence, verdict, receipt, cache **identity**, formalization meaning | `ipfs_datasets_py.logic` |
| Scheduling, isolation, resources, model routing, worktrees, leases, cancellation, workflow, matrix **placement**, cache **placement / single-flight** | `ipfs_accelerate_py.agent_supervisor` |

The supervisor decides **when and where** work executes. It must not redefine
**what the work means**.

---

## Summary matrix (acceptance surfaces)

| Type family | Primary surface | Classification | Disposition | Canonical owner / target |
| --- | --- | --- | --- | --- |
| `LogicFamily` | `analysis.analysis_operation_registry.LogicFamily` | duplicate_legacy | active_legacy → migrate_then_remove | catalog family ids via `analysis_family` map |
| `PropertyKind` | `proof.multi_prover_router.PropertyKind` | duplicate_legacy | active_legacy → migrate_then_remove | software-verification property ids via `property_kind` map |
| `LogicForm` | `proof.logic_translation_validation.LogicForm` | duplicate_legacy | active_legacy → migrate_then_remove | form / encoding labels via `logic_form` map |
| `TranslationClass` | `proof.logic_translation_validation.TranslationClass` | duplicate_legacy | active_legacy → migrate_then_remove | preservation + taxonomy via `translation_class` / `translation_preservation` maps |
| Capability types | `formal_verification_capabilities`, obligation bindings | operational_retain (+ one operation duplicate) | operational_retain / active_legacy | supervisor probes; operations project to provider protocol |
| Operation types | `ProofProviderOperation`, analysis ops, attempt status | mixed (see § Operations) | active_legacy / operational_retain | provider ops → datasets wire; analysis/control ops stay supervisor |
| Matrix types | `prover_matrix_registry.*` | operational_retain | operational_retain | supervisor evidence-bound matrix; route ids project |
| Cache types | `CacheScope`, `ProofCacheKey`, `FormalVerificationCache` | mixed | active_legacy / operational_retain | scope → datasets; placement/single-flight → supervisor |

All acceptance surfaces above are classified. Detail tables follow.

---

## 1. LogicFamily

| Field | Value |
| --- | --- |
| LPC-006 id | sem-001 |
| Path | `ipfs_accelerate_py/agent_supervisor/analysis/analysis_operation_registry.py` |
| Kind | `str, Enum` |
| Members | `tdfol`, `dcec`, `flogic`, `modal`, `deontic`, `frame`, `kg`, `event_calculus` (`KNOWLEDGE_GRAPH` / `KG` share `kg`) |
| Classification | **duplicate_legacy** |
| Disposition | **active_legacy** (public residual); removal only under **migrate_then_remove** |
| Projection | LPC-090 domain `analysis_family` → `_ANALYSIS_FAMILY_TO_CANONICAL` |
| Canonical target | datasets family ids (`tdfol`, `dcec`, `frame_logic`, `modal`, `deontic`, `event_calculus`, namespaced `supervisor.kg`) |

### Decision

`LogicFamily` redefines logic-family meaning that datasets owns. It remains a
public residual so existing analysis operation records and call sites stay
lossless. New durable supervisor records write catalog family identities.
`flogic` and `frame` both project to `frame_logic` with residual reverse map
(no silent merge). `kg` is a supervisor-namespaced extension, not a baseline
taxonomy family.

### Migration path (before public removal)

1. Consumers import canonical family ids (or adapter-projected ids) on new
   writes.
2. Residual reads continue through `SupervisorCanonicalLogicAdapter`.
3. Enumerated public export of `LogicFamily` is removed only after residual
   consumers are gone or accept catalog strings exclusively.
4. Compatibility shims, if needed, live only behind the adapter—not as a
   second registry.

### Related aliases

None on this enum beyond dual member names `KNOWLEDGE_GRAPH` / `KG`.

---

## 2. PropertyKind

| Field | Value |
| --- | --- |
| LPC-006 id | sem-002 |
| Path | `ipfs_accelerate_py/agent_supervisor/proof/multi_prover_router.py` |
| Kind | `str, Enum` |
| Members | `finite_constraint`, `state_machine`, `authorization`, `protocol`, `hyperproperty`, `runtime_trace`, `kernel_check`, `typed_planning`, `temporal_deontic`, `first_order_theorem` |
| Classification | **duplicate_legacy** |
| Disposition | **active_legacy** → **migrate_then_remove** |
| Projection | LPC-090 domain `property_kind` → `_PROPERTY_KIND_TO_CANONICAL` |
| Canonical target | datasets software-verification property ids (`satisfiability`, `reachability`, `authorization`, `trace_conformance`, `hyperproperty`, `theorem`, `invariant`, `safety`, …) |

### Decision

`PropertyKind` is a parallel property-family vocabulary used by multi-prover
routing. Residual collapses (for example `protocol` / `runtime_trace` →
`trace_conformance`, `kernel_check` / `first_order_theorem` → `theorem`) are
explicit and reversible. The type must not be treated as datasets authority.

### Compatibility aliases

| Alias | Of | Classification | Disposition |
| --- | --- | --- | --- |
| `PropertyType` | `PropertyKind` | compatibility_alias | compatibility_alias |
| `ObligationProperty` | `PropertyKind` | compatibility_alias | compatibility_alias |

### Migration path

Same residual pattern as `LogicFamily`: new records write catalog property ids;
public enum removal waits for residual consumer migration.

### Nearby operational (not PropertyKind)

| Type | Path | Classification | Disposition | Rationale |
| --- | --- | --- | --- | --- |
| `ProverRole` | `multi_prover_router.py` | operational_retain | operational_retain | Portfolio lane trust roles; not property meaning |
| `AttemptOutcome` | `multi_prover_router.py` | operational_retain | operational_retain | Per-lane routing outcome; not proof authority |
| `PortfolioVerdict` | `multi_prover_router.py` | operational_retain | operational_retain | Aggregation of lane outcomes; orchestration only |
| `RouteVerdict` | alias of `PortfolioVerdict` | compatibility_alias | compatibility_alias | Compatibility spelling |
| `TemporalPropertyKind` | `runtime/runtime_temporal_monitor.py` | reject_merge | reject_merge | Runtime lease/stop/merge safety; never maps as software-verification `PropertyKind` |

---

## 3. LogicForm

| Field | Value |
| --- | --- |
| LPC-006 id | sem-005 |
| Path | `ipfs_accelerate_py/agent_supervisor/proof/logic_translation_validation.py` |
| Kind | `str, Enum` |
| Members | `ast`, `dcec`, `tdfol`, `fol`, `tptp`, `smt-lib`, `tla+`, `protocol`, `hyperproperty` |
| Classification | **duplicate_legacy** |
| Disposition | **active_legacy** → **migrate_then_remove** |
| Projection | LPC-090 domain `logic_form` → `_LOGIC_FORM_TO_CANONICAL` |
| Canonical target | form / encoding labels (`ast`, `dcec`, `tdfol`, `first_order`, `tptp`, `smtlib`, `transition_system`, `cryptographic_protocol`, `hyperproperty`) |

### Decision

`LogicForm` labels source and target representation families for translation
validation. Datasets owns form/encoding meaning; supervisor retains the enum as
residual wire. Notable renames under projection: `fol` → `first_order`,
`smt-lib` → `smtlib`, `tla+` → `transition_system`, `protocol` →
`cryptographic_protocol`.

### Migration path

New translation artifacts record canonical form labels. Residual reads project
through the adapter. Public removal only after residual consumers migrate.

---

## 4. TranslationClass

| Field | Value |
| --- | --- |
| LPC-006 id | sem-006 |
| Path | `ipfs_accelerate_py/agent_supervisor/proof/logic_translation_validation.py` |
| Kind | `str, Enum` |
| Members | `exact`, `equisatisfiable`, `bounded_abstraction`, `conservative_approximation`, `heuristic` |
| Classification | **duplicate_legacy** |
| Disposition | **active_legacy** → **migrate_then_remove** |
| Projection | LPC-090 domains `translation_class` and `translation_preservation` |
| Canonical targets | preservation short names (`exact`, `equisatisfiable`, `bounded`, `conservative`, `heuristic`) and LPC-031 axis labels (`bounded_abstraction`, `conservative_approximation`, …) |

### Decision

`TranslationClass` is the supervisor preservation / exactness vocabulary. It
projects both to catalog taxonomy translation kinds and to the orthogonal
`LogicTranslationPreservation@1` axis (LPC-031). It must not be confused with
operation status or semantic verdict.

### Compatibility alias

| Alias | Of | Classification | Disposition |
| --- | --- | --- | --- |
| `TranslationExactness` | `TranslationClass` | compatibility_alias | compatibility_alias |

### Support enums (same module)

| Type | Classification | Disposition | Notes |
| --- | --- | --- | --- |
| `ApproximationDirection` | duplicate_legacy | active_legacy → migrate_then_remove | Parallel to datasets translation dimension contracts; residual until translation validation consolidates |
| `SemanticDimension` | duplicate_legacy | active_legacy → migrate_then_remove | Dropped-dimension inventory parallel to datasets contracts |
| `TranslationIssueCode` | duplicate_legacy | active_legacy → migrate_then_remove | Issue codes parallel datasets translation validation |

These three are **not** independent authorities. When translation validation
moves fully onto datasets contracts, they follow the same residual-then-remove
path as `TranslationClass`.

---

## 5. Supervisor capability types

Capability surfaces answer “what may be invoked / is discoverable?” They never
constitute proof evidence. `proof_attempted` / `proof_success` remain false on
capability descriptors.

| Type | Path | Classification | Disposition | Decision |
| --- | --- | --- | --- | --- |
| `CapabilityStatus` (and peer status enums in capabilities module) | `proof/formal_verification_capabilities.py` | operational_retain | operational_retain | Discoverability / version / disabled posture; supervisor ops |
| `CapabilityDimension` | same | operational_retain | operational_retain | Independent dependency axes: provider, executable, package, model, circuit, optional_dependency |
| `ProofProviderCapability` | same | operational_retain | operational_retain | Versioned operation-level routing descriptor; not proof |
| `FormalVerificationProviderCapability` | same | operational_retain | operational_retain | Provider capability row for formal-verification reports |
| `FormalVerificationCapabilityReport` | same | operational_retain | operational_retain | Aggregated capability report; package presence is not authority |
| `LogicCapabilityBinding` | `proof/change_propagation_obligations.py` | operational_retain | operational_retain | Binds change-propagation obligations to capability probes; requires explicit IR admission |
| `LogicCapabilityBinding` | `proof/contract_repair_obligations.py` | operational_retain | operational_retain | Same pattern for contract-repair obligations |
| Doctor / tactician capability probes | `integrations/*_doctor*`, `tactician_hammer_capabilities.py` | canonical_projection / operational_retain | already_projected | Load datasets surfaces for probes; do not invent second registries |

### Decision

Capability **types and reports** stay supervisor-owned. They may **cite**
datasets provider / matrix identifiers, but they do not redefine family,
property, or evidence meaning. Removal is not planned; these are permanent
operational surfaces.

---

## 6. Supervisor operation types

| Type | Path | Classification | Disposition | Projection / notes |
| --- | --- | --- | --- | --- |
| `ProofProviderOperation` | `formal_verification_capabilities.py` | duplicate_legacy | active_legacy → migrate_then_remove | Parallel to datasets `LogicProviderProtocol` operations (`capability`, `translate`, `prove`, `reconstruct`, `verify`, `attest`); converted by `SupervisorLogicProviderFacade` |
| `ProofProviderIsolation` | same | operational_retain | operational_retain | Isolation policy is supervisor ownership; LPC-090 `provider_isolation` projects `subprocess` → `native_process` for runtime labels only |
| `ProviderRequest` / `ProviderResponse` / `ResourceBudget` (provider wire) | `proof/formal_verification_provider.py` (+ contracts) | duplicate_legacy (envelope) / operational_retain (`ResourceBudget`) | active_legacy / operational_retain | Envelope converts via provider facade; `ResourceBudget` is operational_only (not `LogicBoundedness`) |
| `AnalysisOperation` | `analysis_operation_registry.py` | operational_retain | operational_retain | Supervisor analysis operation catalog; not datasets family ops |
| `AttemptStatus` | `formal_verification_contracts.py` | duplicate_legacy | active_legacy | LPC-031 / LPC-090 `operation_status` map |
| `OperationStatus` | `control/control_contracts.py` | reject_merge | reject_merge | Control-plane lifecycle; never map as logic operation status |
| `SupportStatus` | `proof/program_contracts.py` | duplicate_legacy | active_legacy | LPC-090 `availability` map |

### Decision

- **Provider operations** are residual wire vocabulary until consumers speak
  datasets provider protocol types directly.
- **Isolation and resource budgets** remain supervisor operational envelopes.
- **Control-plane `OperationStatus`** is a name collision only (`reject_merge`).
- **Analysis operations** remain supervisor-owned workflow identifiers.

### Migration path (`ProofProviderOperation` and provider envelope)

1. New cross-package calls use datasets `LogicProviderProtocol@1` wire types
   via `SupervisorLogicProviderFacade` (already the projection boundary).
2. Supervisor-public provider API may keep residual operation enums until
   external adapters migrate.
3. Public removal requires a migration task that re-exports or renames without
   dropping residual reverse mapping for in-flight receipts.

---

## 7. Supervisor matrix types

| Type | Path | Classification | Disposition | Decision |
| --- | --- | --- | --- | --- |
| `ProverState` | `proof/prover_matrix_registry.py` | operational_retain | operational_retain | Evidence-bound lifecycle above discovery (`absent` … `authoritative_for`); receipt-derived, never package-presence-as-proof |
| `SelfTestStatus` | same | operational_retain | operational_retain | Bounded fixture execution status |
| `IdentityKind` | same | operational_retain | operational_retain | Executable / package / model / translator / profile / fixture identity kinds for matrix rows |
| `ProverMatrixEntry` | same | operational_retain | operational_retain | One prover’s evidence-bound matrix row |
| `ProverMatrixSnapshot` | same | operational_retain | operational_retain | Point-in-time matrix snapshot |
| `ProverMatrixProbeConfig` | same | operational_retain | operational_retain | Probe budgets and limits |
| `ProverMatrixRegistry` | same | operational_retain | operational_retain | Durable registry + single-flight ownership |
| `ProverMatrixPaths` | same | operational_retain | operational_retain | Storage / documentation path binding |
| Provider route ids (`coq`, `e`, …) | matrix + adapter | duplicate_legacy (route labels) | active_legacy | LPC-090 `provider_route` map (`coq` → `rocq`, `e` → `eprover`); unlisted free-form routes fail closed at admission |

### Decision

The **matrix is supervisor operational ownership**: probe, quarantine,
single-flight, and receipt-bound state. Datasets Markdown / catalog provider
ids are documentation or route projection targets, not a second executable
matrix authority. Matrix **types** are retained indefinitely. Route **labels**
project through the adapter residual map.

---

## 8. Supervisor cache types

| Type | Path | Classification | Disposition | Decision |
| --- | --- | --- | --- | --- |
| `CacheScope` | `analysis_operation_registry.py` | duplicate_legacy | active_legacy → migrate_then_remove | Parallel datasets cache protocol scopes; LPC-090 `cache_scope` (`exact_tree` → `tree`, `objective_revision` → `policy`, `request`, `none`) |
| `ProofCacheKey` | `formal_verification_cache.py` | operational_retain (placement identity) with projection onto datasets semantic key | operational_retain | Supervisor binds execution identities; datasets owns `CanonicalProofCacheKey@1` semantic fields (LPC-080) |
| `DraftCacheKey` | same | operational_retain | operational_retain | Untrusted draft reuse identity; not kernel authority |
| `CacheLookupStatus` / `CacheRejectionReason` | same | operational_retain | operational_retain | Hit/miss/reject and audit reason codes |
| `FormalVerificationCache` | same | operational_retain | operational_retain | DuckDB placement, TTL, single-flight, fail-closed reconstruction |
| Contract-analysis cache adapter | `contract_analysis/cache_adapter.py` | canonical_projection | already_projected | Thin bind into datasets software-contracts cache |
| Prefix / token / doctor cache types | context, self_improvement, validation modules | operational_retain / reject_merge as domain-local | operational_retain | Not proof-receipt semantic cache keys |

### Decision

- **Semantic cache identity** (what makes two proofs the same) is datasets
  authority (`CanonicalProofCacheKey@1`).
- **Placement, lease, single-flight, TTL, and rejection reasons** are supervisor
  operational ownership and stay.
- **`CacheScope`** is residual semantic vocabulary; new records write catalog
  scopes after projection.

### Migration path (`CacheScope` only)

Same residual pattern as family/property. Do not remove `ProofCacheKey` /
`FormalVerificationCache`; they are not semantic duplicates of the datasets
key algorithm.

---

## 9. Other leftover semantic types (post LPC-090 completeness)

These are classified so LPC-091 leaves no LPC-006 residual undecided.

### Proof receipt axes (already mapped; residual public types)

| Type | Path | Classification | Disposition | Map domain |
| --- | --- | --- | --- | --- |
| `EvidenceKind` (proof) | `formal_verification_contracts.py` | duplicate_legacy | active_legacy | `evidence_kind` |
| `ProofEvidenceKind` | alias | compatibility_alias | compatibility_alias | — |
| `ZKP_ATTESTATION` | member alias | compatibility_alias | compatibility_alias | → `cryptographic_attestation` |
| `EvidenceAuthority` (proof) | same | duplicate_legacy | active_legacy | `evidence_authority` |
| `AssuranceLevel` | same | duplicate_legacy | active_legacy | `assurance_authority` |
| `AssuranceLevel.NONE` / `SOLVER_VERIFIED` | member aliases | compatibility_alias | compatibility_alias | — |
| `ProofVerdict` | same | duplicate_legacy | active_legacy | `semantic_verdict` |

### Reject-merge name collisions

| Type | Path | Classification | Disposition |
| --- | --- | --- | --- |
| `EvidenceAuthority` | `objectives/goal_quality.py` | reject_merge | reject_merge |
| `EvidenceAuthority` | `prompt/prompt_workflow.py` | reject_merge | reject_merge |
| `EvidenceAuthority` | `planning/plan_analysis_query_planner.py` | reject_merge | reject_merge |
| `EvidenceAuthorityClass` | `validation/planner_doctor_live_benchmark.py` | reject_merge | reject_merge |
| `EvidenceKind` | `analysis/repository_surface_inventory.py` | reject_merge | reject_merge |

### Reviewed plan vocabulary

| Type | Path | Classification | Disposition | Decision |
| --- | --- | --- | --- | --- |
| `DCECOperator` / `TDFOLProperty` / `FormulaOperator` / `ReviewedPredicate` / `TermSort` | `proof/formal_logic_vocabulary.py` | duplicate_legacy | active_legacy → migrate_then_remove | Parallel naming to datasets DCEC/TDFOL operators; scoped to finite-trace plan checks. Residual semantic risk; project or namespace before treating as family authority. |
| `Sort` | alias of `TermSort` | compatibility_alias | compatibility_alias | — |

### Projection adapters (already correct)

| Type | Path | Classification | Disposition |
| --- | --- | --- | --- |
| `VocabularyProjection` / `SupervisorCanonicalLogicAdapter` | `proof/canonical_logic_adapter.py` | canonical_projection | already_projected |
| `SupervisorLogicProviderFacade` | `proof/logic_provider_contract.py` | canonical_projection | already_projected |

---

## 10. Closed classification inventory

| Bucket | Count (type rows in this note) | Action posture |
| --- | --- | ---: |
| duplicate_legacy | LogicFamily, PropertyKind, LogicForm, TranslationClass, translation support enums, proof axes, provider ops/envelope, CacheScope, plan vocabulary | Retain residual; project; migrate before remove |
| compatibility_alias | PropertyType, ObligationProperty, TranslationExactness, ProofEvidenceKind, ZKP_ATTESTATION, AssuranceLevel aliases, RouteVerdict, Sort | Keep spellings; prefer primary residual |
| operational_retain | capability reports/bindings, isolation, matrix suite, cache placement, analysis ops, portfolio roles/outcomes, ResourceBudget | Keep under supervisor |
| reject_merge | goal/prompt/plan/doctor/surface Evidence\*, control OperationStatus, TemporalPropertyKind | Domain-local only |
| canonical_projection | adapter + provider facade + contract-analysis cache adapter | Authority boundary |

**Unclassified leftovers in acceptance scope: none.**

---

## Relationship to neighboring tasks

| Task | Relationship |
| --- | --- |
| LPC-006 | Inventories imports and types; this note freezes post-cutover lifecycle |
| LPC-031 | Axis maps for status/verdict/availability/evidence/authority/translation |
| LPC-080 | Datasets owns semantic cache-key fields; supervisor owns placement |
| LPC-090 | Generated residual maps; this note classifies the public types those maps serve |
| LPC-100 | Manifest pins catalog root used by residual projection |
| LPC-110 | Client consumes residual maps through the lazy adapter; does not reintroduce hand maps |

## What this task does **not** do

* Does not remove or rename any public supervisor type.
* Does not regenerate adapter map tables (LPC-090 owns those).
* Does not implement `SupervisorLogicPlatformClient` (LPC-110).
* Does not weaken fail-closed unknown handling.

## Acceptance checklist

| Required surface | Classified |
| --- | --- |
| `LogicFamily` | yes — duplicate_legacy / active_legacy → migrate_then_remove |
| `PropertyKind` | yes — duplicate_legacy / active_legacy → migrate_then_remove |
| `LogicForm` | yes — duplicate_legacy / active_legacy → migrate_then_remove |
| `TranslationClass` | yes — duplicate_legacy / active_legacy → migrate_then_remove |
| Capability types | yes — operational_retain (reports, dimensions, bindings) |
| Operation types | yes — mixed duplicate_legacy / operational_retain / reject_merge |
| Matrix types | yes — operational_retain (+ route-label projection) |
| Cache types | yes — CacheScope duplicate_legacy; placement/single-flight operational_retain |
