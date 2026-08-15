# LPC-090 Supervisor Compatibility Maps (Catalog-Projected Cutover)

**Task:** LPC-090 — Generate supervisor compatibility maps from the catalog  
**Goal:** LPC-G090  
**Depends on:** LPC-023 (`GeneratedProviderTranslationCatalog@1`), LPC-031 (`legacy_enum_mappings.md`)  
**Adapter (lazy boundary):** `ipfs_accelerate_py.agent_supervisor.proof.canonical_logic_adapter`  
**Interface:** `SupervisorCanonicalLogicAdapter@1`  
**Schema:** `ipfs_accelerate_py/agent-supervisor/supervisor-map-cutover@1`  
**Validation:** `python -m pytest test/api/test_canonical_logic_adapter.py -q`

## Purpose

Replace hand-maintained supervisor family / property / form / translation /
provider / evidence / authority / cache maps with a **generated, fail-closed
projection** onto catalog identities. New supervisor records write **canonical
identities**. Legacy enum values remain only through explicit adapters that
retain residual supervisor tokens for lossless reverse mapping.

This note is the durable cutover artifact. Machine-readable
`supervisor-map` fences are exhaustive for the closed supervisor vocabulary
surfaces below. Tests parse every fence, bind each row to the sealed catalog
root, and verify unknown values fail closed.

## Catalog root binding

All map rows bind to the sealed catalog content root from
`CanonicalLogicCatalogSnapshot@1`:

| Field | Authority |
| --- | --- |
| `catalog_root` | `DEFAULT_CANONICAL_CATALOG_SNAPSHOT.content_root` (CIDv1) |
| `catalog_digest` | `DEFAULT_CANONICAL_CATALOG_SNAPSHOT.content_digest` (`sha256:…`) |
| `catalog_interface` | `CanonicalLogicCatalogSnapshot@1` |
| `generated_projection` | `GeneratedProviderTranslationCatalog@1` (LPC-023) |

Document-level binding token used in every row:

```supervisor-map-meta
schema: ipfs_accelerate_py/agent-supervisor/supervisor-map-cutover@1
task: LPC-090
goal: LPC-G090
adapter_interface: SupervisorCanonicalLogicAdapter@1
catalog_interface: CanonicalLogicCatalogSnapshot@1
catalog_root_binding: DEFAULT_CANONICAL_CATALOG_SNAPSHOT.content_root
catalog_digest_binding: DEFAULT_CANONICAL_CATALOG_SNAPSHOT.content_digest
fail_closed: true
unknown_policy: raise CanonicalLogicAdapterError / SupervisorMapCutoverError
```

In each `supervisor-map` row, `catalog_root: $catalog_root_binding` means the
live sealed content root. Tests expand the binding against the installed
snapshot and reject any row that omits it.

## Row vocabulary

Each legacy value maps to **all five** required cutover fields:

| Field | Meaning |
| --- | --- |
| `canonical_identity` | Datasets / catalog id written on new records |
| `disposition` | How the legacy value is admitted (`map`, `residual_collapse`, `compatibility_alias`, `reject_merge`, `operational_only`) |
| `residual` | Exact supervisor token / enum residual required for reverse mapping |
| `deprecation` | Lifecycle of the legacy surface (`active_legacy`, `compatibility_alias`, `reject_merge`, `operational_only`, `namespaced_extension`) |
| `catalog_root` | Sealed catalog content root binding (`$catalog_root_binding`) |

### Disposition rules

| Disposition | Behavior |
| --- | --- |
| `map` | Direct projection; residual still stores supervisor id for identity |
| `residual_collapse` | Multiple supervisor values share one canonical id; residual is mandatory for reverse map |
| `compatibility_alias` | Alias spelling of another live wire value; projects through the primary residual |
| `reject_merge` | Homonymous non-logic enum; never projects onto logic axes |
| `operational_only` | Supervisor operational envelope; not a semantic catalog identity |

### Fail-closed policy

1. **Known label only.** A mapper accepts only labels listed for its domain.
2. **Unknown → error.** Empty, wrong-domain, or unlisted labels raise.
3. **No silent merge.** Distinct supervisor identities never collapse without an
   explicit residual (for example `flogic` and `frame` both → `frame_logic`).
4. **No authority promotion.** Mapping never upgrades evidence authority or
   invents a conclusive verdict.
5. **Catalog presence ≠ availability / proof.** Binding a catalog root never
   claims live prover availability or production admission.
6. **Reject-merge / operational-only never project** as catalog family/property
   identities.

## Domain inventory

| Domain | Supervisor surface | Catalog / canonical target | Disposition family |
| --- | --- | --- | --- |
| `analysis_family` | `LogicFamily` | family ids / `supervisor.*` | map + residual_collapse |
| `property_kind` | `PropertyKind` | software-verification property ids | map + residual_collapse |
| `logic_form` | `LogicForm` | form / family / encoding labels | map |
| `translation_class` | `TranslationClass` | preservation / taxonomy kinds | map |
| `cache_scope` | `CacheScope` | cache protocol scopes | map |
| `provider_route` | matrix prover ids | provider catalog / matrix ids | map |
| `provider_isolation` | `ProofProviderIsolation` | runtime isolation labels | map |
| `operation_status` | `AttemptStatus` | `LogicOperationStatus@1` | map (LPC-031) |
| `semantic_verdict` | `ProofVerdict` | `LogicSemanticVerdict@1` | map (LPC-031) |
| `availability` | `SupportStatus` | `LogicAvailability@1` | map (LPC-031) |
| `evidence_kind` | proof `EvidenceKind` | `LogicEvidenceKind@1` | map (LPC-031) |
| `evidence_authority` | proof `EvidenceAuthority` | `LogicEvidenceAuthority@1` | map (LPC-031) |
| `assurance_authority` | `AssuranceLevel` | `LogicEvidenceAuthority@1` | map (LPC-031) |
| `translation_preservation` | `TranslationClass` (axis) | `LogicTranslationPreservation@1` | map (LPC-031) |
| `reject_merge` | goal/prompt/plan/doctor/surface enums | — | reject_merge |
| `operational_only` | `ResourceBudget` | — | operational_only |

---

## Mapping tables

Machine-readable blocks use fenced `supervisor-map` sections. Label lines are:

```text
legacy_value: canonical_identity=<id>; disposition=<d>; residual=<token>; deprecation=<status>; catalog_root=$catalog_root_binding
```

Residual uses compact `k=v` pairs joined by `|`. Tests parse every block.

### analysis_family — supervisor.LogicFamily

Source: adapter `_ANALYSIS_FAMILY_TO_CANONICAL` projected against catalog family
ids. `flogic` / `frame` share `frame_logic` with residual reverse map.
`kg` is a supervisor-namespaced extension (not a baseline taxonomy family).

```supervisor-map
domain: analysis_family
surface: supervisor.LogicFamily
fail_closed: true
catalog_root: $catalog_root_binding
tdfol: canonical_identity=tdfol; disposition=map; residual=supervisor_id=tdfol|supervisor_enum=LogicFamily|supervisor_member=TDFOL; deprecation=active_legacy; catalog_root=$catalog_root_binding
dcec: canonical_identity=dcec; disposition=map; residual=supervisor_id=dcec|supervisor_enum=LogicFamily|supervisor_member=DCEC; deprecation=active_legacy; catalog_root=$catalog_root_binding
flogic: canonical_identity=frame_logic; disposition=residual_collapse; residual=supervisor_id=flogic|supervisor_enum=LogicFamily|supervisor_member=FLOGIC|collapse_group=frame_logic; deprecation=active_legacy; catalog_root=$catalog_root_binding
modal: canonical_identity=modal; disposition=map; residual=supervisor_id=modal|supervisor_enum=LogicFamily|supervisor_member=MODAL; deprecation=active_legacy; catalog_root=$catalog_root_binding
deontic: canonical_identity=deontic; disposition=map; residual=supervisor_id=deontic|supervisor_enum=LogicFamily|supervisor_member=DEONTIC; deprecation=active_legacy; catalog_root=$catalog_root_binding
frame: canonical_identity=frame_logic; disposition=residual_collapse; residual=supervisor_id=frame|supervisor_enum=LogicFamily|supervisor_member=FRAME|collapse_group=frame_logic; deprecation=active_legacy; catalog_root=$catalog_root_binding
kg: canonical_identity=supervisor.kg; disposition=map; residual=supervisor_id=kg|supervisor_enum=LogicFamily|supervisor_member=KNOWLEDGE_GRAPH; deprecation=namespaced_extension; catalog_root=$catalog_root_binding
event_calculus: canonical_identity=event_calculus; disposition=map; residual=supervisor_id=event_calculus|supervisor_enum=LogicFamily|supervisor_member=EVENT_CALCULUS; deprecation=active_legacy; catalog_root=$catalog_root_binding
```

### property_kind — supervisor.PropertyKind

Source: adapter `_PROPERTY_KIND_TO_CANONICAL`. `protocol` and `runtime_trace`
share `trace_conformance` with residual reverse map.

```supervisor-map
domain: property_kind
surface: supervisor.PropertyKind
fail_closed: true
catalog_root: $catalog_root_binding
finite_constraint: canonical_identity=satisfiability; disposition=map; residual=supervisor_id=finite_constraint|supervisor_enum=PropertyKind|supervisor_member=FINITE_CONSTRAINT; deprecation=active_legacy; catalog_root=$catalog_root_binding
state_machine: canonical_identity=reachability; disposition=map; residual=supervisor_id=state_machine|supervisor_enum=PropertyKind|supervisor_member=STATE_MACHINE; deprecation=active_legacy; catalog_root=$catalog_root_binding
authorization: canonical_identity=authorization; disposition=map; residual=supervisor_id=authorization|supervisor_enum=PropertyKind|supervisor_member=AUTHORIZATION; deprecation=active_legacy; catalog_root=$catalog_root_binding
protocol: canonical_identity=trace_conformance; disposition=residual_collapse; residual=supervisor_id=protocol|supervisor_enum=PropertyKind|supervisor_member=PROTOCOL|collapse_group=trace_conformance; deprecation=active_legacy; catalog_root=$catalog_root_binding
hyperproperty: canonical_identity=hyperproperty; disposition=map; residual=supervisor_id=hyperproperty|supervisor_enum=PropertyKind|supervisor_member=HYPERPROPERTY; deprecation=active_legacy; catalog_root=$catalog_root_binding
runtime_trace: canonical_identity=trace_conformance; disposition=residual_collapse; residual=supervisor_id=runtime_trace|supervisor_enum=PropertyKind|supervisor_member=RUNTIME_TRACE|collapse_group=trace_conformance; deprecation=active_legacy; catalog_root=$catalog_root_binding
kernel_check: canonical_identity=theorem; disposition=residual_collapse; residual=supervisor_id=kernel_check|supervisor_enum=PropertyKind|supervisor_member=KERNEL_CHECK|collapse_group=theorem; deprecation=active_legacy; catalog_root=$catalog_root_binding
typed_planning: canonical_identity=invariant; disposition=map; residual=supervisor_id=typed_planning|supervisor_enum=PropertyKind|supervisor_member=TYPED_PLANNING; deprecation=active_legacy; catalog_root=$catalog_root_binding
temporal_deontic: canonical_identity=safety; disposition=map; residual=supervisor_id=temporal_deontic|supervisor_enum=PropertyKind|supervisor_member=TEMPORAL_DEONTIC; deprecation=active_legacy; catalog_root=$catalog_root_binding
first_order_theorem: canonical_identity=theorem; disposition=residual_collapse; residual=supervisor_id=first_order_theorem|supervisor_enum=PropertyKind|supervisor_member=FIRST_ORDER_THEOREM|collapse_group=theorem; deprecation=active_legacy; catalog_root=$catalog_root_binding
```

### logic_form — supervisor.LogicForm

Source: adapter `_LOGIC_FORM_TO_CANONICAL`.

```supervisor-map
domain: logic_form
surface: supervisor.LogicForm
fail_closed: true
catalog_root: $catalog_root_binding
ast: canonical_identity=ast; disposition=map; residual=supervisor_id=ast|supervisor_enum=LogicForm|supervisor_member=AST; deprecation=active_legacy; catalog_root=$catalog_root_binding
dcec: canonical_identity=dcec; disposition=map; residual=supervisor_id=dcec|supervisor_enum=LogicForm|supervisor_member=DCEC; deprecation=active_legacy; catalog_root=$catalog_root_binding
tdfol: canonical_identity=tdfol; disposition=map; residual=supervisor_id=tdfol|supervisor_enum=LogicForm|supervisor_member=TDFOL; deprecation=active_legacy; catalog_root=$catalog_root_binding
fol: canonical_identity=first_order; disposition=map; residual=supervisor_id=fol|supervisor_enum=LogicForm|supervisor_member=FOL; deprecation=active_legacy; catalog_root=$catalog_root_binding
tptp: canonical_identity=tptp; disposition=map; residual=supervisor_id=tptp|supervisor_enum=LogicForm|supervisor_member=TPTP; deprecation=active_legacy; catalog_root=$catalog_root_binding
smt-lib: canonical_identity=smtlib; disposition=map; residual=supervisor_id=smt-lib|supervisor_enum=LogicForm|supervisor_member=SMT_LIB; deprecation=active_legacy; catalog_root=$catalog_root_binding
tla+: canonical_identity=transition_system; disposition=map; residual=supervisor_id=tla+|supervisor_enum=LogicForm|supervisor_member=TLA_PLUS; deprecation=active_legacy; catalog_root=$catalog_root_binding
protocol: canonical_identity=cryptographic_protocol; disposition=map; residual=supervisor_id=protocol|supervisor_enum=LogicForm|supervisor_member=PROTOCOL; deprecation=active_legacy; catalog_root=$catalog_root_binding
hyperproperty: canonical_identity=hyperproperty; disposition=map; residual=supervisor_id=hyperproperty|supervisor_enum=LogicForm|supervisor_member=HYPERPROPERTY; deprecation=active_legacy; catalog_root=$catalog_root_binding
```

### translation_class — supervisor.TranslationClass (preservation + taxonomy)

Source: adapter `_TRANSLATION_CLASS_TO_PRESERVATION` and
`_TRANSLATION_CLASS_TO_TAXONOMY_KIND`. Residual carries both.

```supervisor-map
domain: translation_class
surface: supervisor.TranslationClass
fail_closed: true
catalog_root: $catalog_root_binding
exact: canonical_identity=exact; disposition=map; residual=supervisor_id=exact|supervisor_enum=TranslationClass|supervisor_member=EXACT|taxonomy_translation_kind=lossless; deprecation=active_legacy; catalog_root=$catalog_root_binding
equisatisfiable: canonical_identity=equisatisfiable; disposition=map; residual=supervisor_id=equisatisfiable|supervisor_enum=TranslationClass|supervisor_member=EQUISATISFIABLE|taxonomy_translation_kind=equisatisfiable; deprecation=active_legacy; catalog_root=$catalog_root_binding
bounded_abstraction: canonical_identity=bounded; disposition=map; residual=supervisor_id=bounded_abstraction|supervisor_enum=TranslationClass|supervisor_member=BOUNDED_ABSTRACTION|taxonomy_translation_kind=sound_over_approximation; deprecation=active_legacy; catalog_root=$catalog_root_binding
conservative_approximation: canonical_identity=conservative; disposition=map; residual=supervisor_id=conservative_approximation|supervisor_enum=TranslationClass|supervisor_member=CONSERVATIVE_APPROXIMATION|taxonomy_translation_kind=sound_over_approximation; deprecation=active_legacy; catalog_root=$catalog_root_binding
heuristic: canonical_identity=heuristic; disposition=map; residual=supervisor_id=heuristic|supervisor_enum=TranslationClass|supervisor_member=HEURISTIC|taxonomy_translation_kind=heuristic; deprecation=active_legacy; catalog_root=$catalog_root_binding
```

### cache_scope — supervisor.CacheScope

Source: adapter `_CACHE_SCOPE_TO_CANONICAL`.

```supervisor-map
domain: cache_scope
surface: supervisor.CacheScope
fail_closed: true
catalog_root: $catalog_root_binding
exact_tree: canonical_identity=tree; disposition=map; residual=supervisor_id=exact_tree|supervisor_enum=CacheScope|supervisor_member=TREE; deprecation=active_legacy; catalog_root=$catalog_root_binding
objective_revision: canonical_identity=policy; disposition=map; residual=supervisor_id=objective_revision|supervisor_enum=CacheScope|supervisor_member=OBJECTIVE; deprecation=active_legacy; catalog_root=$catalog_root_binding
request: canonical_identity=request; disposition=map; residual=supervisor_id=request|supervisor_enum=CacheScope|supervisor_member=REQUEST; deprecation=active_legacy; catalog_root=$catalog_root_binding
none: canonical_identity=none; disposition=map; residual=supervisor_id=none|supervisor_enum=CacheScope|supervisor_member=NONE; deprecation=active_legacy; catalog_root=$catalog_root_binding
```

### provider_route — supervisor matrix prover ids

Source: adapter `_SUPERVISOR_PROVER_TO_CANONICAL_PROVIDER` joined to catalog
provider / executable matrix ids. Unlisted prover ids pass through identity
only when already a catalog provider id; free-form unknown routes fail closed
at admission (tests enforce explicit rows + unknown rejection).

```supervisor-map
domain: provider_route
surface: supervisor.prover_matrix
fail_closed: true
catalog_root: $catalog_root_binding
coq: canonical_identity=rocq; disposition=map; residual=supervisor_id=coq|domain=provider_route; deprecation=active_legacy; catalog_root=$catalog_root_binding
e: canonical_identity=eprover; disposition=map; residual=supervisor_id=e|domain=provider_route; deprecation=active_legacy; catalog_root=$catalog_root_binding
```

### provider_isolation — supervisor.ProofProviderIsolation

Source: adapter `_ISOLATION_TO_RUNTIME`.

```supervisor-map
domain: provider_isolation
surface: supervisor.ProofProviderIsolation
fail_closed: true
catalog_root: $catalog_root_binding
in_process: canonical_identity=in_process; disposition=map; residual=supervisor_id=in_process|supervisor_enum=ProofProviderIsolation|supervisor_member=IN_PROCESS; deprecation=active_legacy; catalog_root=$catalog_root_binding
subprocess: canonical_identity=native_process; disposition=map; residual=supervisor_id=subprocess|supervisor_enum=ProofProviderIsolation|supervisor_member=SUBPROCESS; deprecation=active_legacy; catalog_root=$catalog_root_binding
```

### operation_status — supervisor.AttemptStatus (LPC-031 axis)

```supervisor-map
domain: operation_status
surface: supervisor.AttemptStatus
fail_closed: true
catalog_root: $catalog_root_binding
planned: canonical_identity=planned; disposition=map; residual=supervisor_id=planned|supervisor_enum=AttemptStatus|axis=operation_status; deprecation=active_legacy; catalog_root=$catalog_root_binding
running: canonical_identity=running; disposition=map; residual=supervisor_id=running|supervisor_enum=AttemptStatus|axis=operation_status; deprecation=active_legacy; catalog_root=$catalog_root_binding
succeeded: canonical_identity=succeeded; disposition=map; residual=supervisor_id=succeeded|supervisor_enum=AttemptStatus|axis=operation_status; deprecation=active_legacy; catalog_root=$catalog_root_binding
failed: canonical_identity=failed; disposition=map; residual=supervisor_id=failed|supervisor_enum=AttemptStatus|axis=operation_status; deprecation=active_legacy; catalog_root=$catalog_root_binding
unsupported: canonical_identity=unsupported; disposition=map; residual=supervisor_id=unsupported|supervisor_enum=AttemptStatus|axis=operation_status; deprecation=active_legacy; catalog_root=$catalog_root_binding
unavailable: canonical_identity=unavailable; disposition=map; residual=supervisor_id=unavailable|supervisor_enum=AttemptStatus|axis=operation_status; deprecation=active_legacy; catalog_root=$catalog_root_binding
timed_out: canonical_identity=timed_out; disposition=map; residual=supervisor_id=timed_out|supervisor_enum=AttemptStatus|axis=operation_status; deprecation=active_legacy; catalog_root=$catalog_root_binding
cancelled: canonical_identity=cancelled; disposition=map; residual=supervisor_id=cancelled|supervisor_enum=AttemptStatus|axis=operation_status; deprecation=active_legacy; catalog_root=$catalog_root_binding
blocked: canonical_identity=blocked; disposition=map; residual=supervisor_id=blocked|supervisor_enum=AttemptStatus|axis=operation_status; deprecation=active_legacy; catalog_root=$catalog_root_binding
```

### semantic_verdict — supervisor.ProofVerdict (LPC-031 axis)

```supervisor-map
domain: semantic_verdict
surface: supervisor.ProofVerdict
fail_closed: true
catalog_root: $catalog_root_binding
proved: canonical_identity=proved; disposition=map; residual=supervisor_id=proved|supervisor_enum=ProofVerdict|axis=semantic_verdict; deprecation=active_legacy; catalog_root=$catalog_root_binding
disproved: canonical_identity=disproved; disposition=map; residual=supervisor_id=disproved|supervisor_enum=ProofVerdict|axis=semantic_verdict; deprecation=active_legacy; catalog_root=$catalog_root_binding
inconclusive: canonical_identity=inconclusive; disposition=map; residual=supervisor_id=inconclusive|supervisor_enum=ProofVerdict|axis=semantic_verdict; deprecation=active_legacy; catalog_root=$catalog_root_binding
unsupported: canonical_identity=unsupported; disposition=map; residual=supervisor_id=unsupported|supervisor_enum=ProofVerdict|axis=semantic_verdict; deprecation=active_legacy; catalog_root=$catalog_root_binding
error: canonical_identity=error; disposition=map; residual=supervisor_id=error|supervisor_enum=ProofVerdict|axis=semantic_verdict; deprecation=active_legacy; catalog_root=$catalog_root_binding
cancelled: canonical_identity=cancelled; disposition=map; residual=supervisor_id=cancelled|supervisor_enum=ProofVerdict|axis=semantic_verdict; deprecation=active_legacy; catalog_root=$catalog_root_binding
```

### availability — supervisor.SupportStatus (LPC-031 axis)

```supervisor-map
domain: availability
surface: supervisor.SupportStatus
fail_closed: true
catalog_root: $catalog_root_binding
supported: canonical_identity=available; disposition=map; residual=supervisor_id=supported|supervisor_enum=SupportStatus|axis=availability; deprecation=active_legacy; catalog_root=$catalog_root_binding
unsupported: canonical_identity=unsupported; disposition=map; residual=supervisor_id=unsupported|supervisor_enum=SupportStatus|axis=availability; deprecation=active_legacy; catalog_root=$catalog_root_binding
assumed: canonical_identity=declared; disposition=map; residual=supervisor_id=assumed|supervisor_enum=SupportStatus|axis=availability; deprecation=active_legacy; catalog_root=$catalog_root_binding
not_applicable: canonical_identity=unknown; disposition=map; residual=supervisor_id=not_applicable|supervisor_enum=SupportStatus|axis=availability; deprecation=active_legacy; catalog_root=$catalog_root_binding
unknown: canonical_identity=unknown; disposition=map; residual=supervisor_id=unknown|supervisor_enum=SupportStatus|axis=availability; deprecation=active_legacy; catalog_root=$catalog_root_binding
```

### evidence_kind — supervisor.EvidenceKind (LPC-031 axis)

```supervisor-map
domain: evidence_kind
surface: supervisor.EvidenceKind
fail_closed: true
catalog_root: $catalog_root_binding
unknown: canonical_identity=unknown; disposition=map; residual=supervisor_id=unknown|supervisor_enum=EvidenceKind|axis=evidence_kind; deprecation=active_legacy; catalog_root=$catalog_root_binding
llm_output: canonical_identity=llm_output; disposition=map; residual=supervisor_id=llm_output|supervisor_enum=EvidenceKind|axis=evidence_kind; deprecation=active_legacy; catalog_root=$catalog_root_binding
atp_candidate: canonical_identity=atp_candidate; disposition=map; residual=supervisor_id=atp_candidate|supervisor_enum=EvidenceKind|axis=evidence_kind; deprecation=active_legacy; catalog_root=$catalog_root_binding
smt_candidate: canonical_identity=smt_candidate; disposition=map; residual=supervisor_id=smt_candidate|supervisor_enum=EvidenceKind|axis=evidence_kind; deprecation=active_legacy; catalog_root=$catalog_root_binding
solver_result: canonical_identity=solver_result; disposition=map; residual=supervisor_id=solver_result|supervisor_enum=EvidenceKind|axis=evidence_kind; deprecation=active_legacy; catalog_root=$catalog_root_binding
kernel_verification: canonical_identity=kernel_checked_proof; disposition=map; residual=supervisor_id=kernel_verification|supervisor_enum=EvidenceKind|axis=evidence_kind; deprecation=active_legacy; catalog_root=$catalog_root_binding
test_result: canonical_identity=test_result; disposition=map; residual=supervisor_id=test_result|supervisor_enum=EvidenceKind|axis=evidence_kind; deprecation=active_legacy; catalog_root=$catalog_root_binding
static_analysis: canonical_identity=static_analysis; disposition=map; residual=supervisor_id=static_analysis|supervisor_enum=EvidenceKind|axis=evidence_kind; deprecation=active_legacy; catalog_root=$catalog_root_binding
cryptographic_attestation: canonical_identity=attestation; disposition=map; residual=supervisor_id=cryptographic_attestation|supervisor_enum=EvidenceKind|axis=evidence_kind; deprecation=active_legacy; catalog_root=$catalog_root_binding
zkp_attestation: canonical_identity=attestation; disposition=compatibility_alias; residual=supervisor_id=cryptographic_attestation|supervisor_enum=EvidenceKind|supervisor_member=ZKP_ATTESTATION|alias_of=cryptographic_attestation; deprecation=compatibility_alias; catalog_root=$catalog_root_binding
cache_entry: canonical_identity=cache_entry; disposition=map; residual=supervisor_id=cache_entry|supervisor_enum=EvidenceKind|axis=evidence_kind; deprecation=active_legacy; catalog_root=$catalog_root_binding
```

### evidence_authority — supervisor.EvidenceAuthority (LPC-031 axis)

Producer/checker boundary names project conservatively (no silent promotion to
`authoritative` from a boundary name alone).

```supervisor-map
domain: evidence_authority
surface: supervisor.EvidenceAuthority
fail_closed: true
catalog_root: $catalog_root_binding
unknown: canonical_identity=unknown; disposition=map; residual=supervisor_id=unknown|supervisor_enum=EvidenceAuthority|axis=evidence_authority; deprecation=active_legacy; catalog_root=$catalog_root_binding
provider: canonical_identity=advisory; disposition=map; residual=supervisor_id=provider|supervisor_enum=EvidenceAuthority|axis=evidence_authority; deprecation=active_legacy; catalog_root=$catalog_root_binding
llm: canonical_identity=advisory; disposition=map; residual=supervisor_id=llm|supervisor_enum=EvidenceAuthority|axis=evidence_authority; deprecation=active_legacy; catalog_root=$catalog_root_binding
atp: canonical_identity=advisory; disposition=map; residual=supervisor_id=atp|supervisor_enum=EvidenceAuthority|axis=evidence_authority; deprecation=active_legacy; catalog_root=$catalog_root_binding
smt: canonical_identity=advisory; disposition=map; residual=supervisor_id=smt|supervisor_enum=EvidenceAuthority|axis=evidence_authority; deprecation=active_legacy; catalog_root=$catalog_root_binding
solver: canonical_identity=bounded; disposition=map; residual=supervisor_id=solver|supervisor_enum=EvidenceAuthority|axis=evidence_authority; deprecation=active_legacy; catalog_root=$catalog_root_binding
kernel: canonical_identity=independently_checkable; disposition=map; residual=supervisor_id=kernel|supervisor_enum=EvidenceAuthority|axis=evidence_authority; deprecation=active_legacy; catalog_root=$catalog_root_binding
attestation_verifier: canonical_identity=independently_checkable; disposition=map; residual=supervisor_id=attestation_verifier|supervisor_enum=EvidenceAuthority|axis=evidence_authority; deprecation=active_legacy; catalog_root=$catalog_root_binding
validation_runner: canonical_identity=bounded; disposition=map; residual=supervisor_id=validation_runner|supervisor_enum=EvidenceAuthority|axis=evidence_authority; deprecation=active_legacy; catalog_root=$catalog_root_binding
cache: canonical_identity=none; disposition=map; residual=supervisor_id=cache|supervisor_enum=EvidenceAuthority|axis=evidence_authority; deprecation=active_legacy; catalog_root=$catalog_root_binding
```

### assurance_authority — supervisor.AssuranceLevel (LPC-031 axis)

```supervisor-map
domain: assurance_authority
surface: supervisor.AssuranceLevel
fail_closed: true
catalog_root: $catalog_root_binding
unverified: canonical_identity=none; disposition=map; residual=supervisor_id=unverified|supervisor_enum=AssuranceLevel|axis=evidence_authority; deprecation=active_legacy; catalog_root=$catalog_root_binding
none: canonical_identity=none; disposition=compatibility_alias; residual=supervisor_id=unverified|supervisor_enum=AssuranceLevel|supervisor_member=NONE|alias_of=unverified; deprecation=compatibility_alias; catalog_root=$catalog_root_binding
candidate: canonical_identity=advisory; disposition=map; residual=supervisor_id=candidate|supervisor_enum=AssuranceLevel|axis=evidence_authority; deprecation=active_legacy; catalog_root=$catalog_root_binding
solver_checked: canonical_identity=bounded; disposition=map; residual=supervisor_id=solver_checked|supervisor_enum=AssuranceLevel|axis=evidence_authority; deprecation=active_legacy; catalog_root=$catalog_root_binding
solver_verified: canonical_identity=bounded; disposition=compatibility_alias; residual=supervisor_id=solver_checked|supervisor_enum=AssuranceLevel|supervisor_member=SOLVER_VERIFIED|alias_of=solver_checked; deprecation=compatibility_alias; catalog_root=$catalog_root_binding
kernel_verified: canonical_identity=independently_checkable; disposition=map; residual=supervisor_id=kernel_verified|supervisor_enum=AssuranceLevel|axis=evidence_authority; deprecation=active_legacy; catalog_root=$catalog_root_binding
attested: canonical_identity=authoritative; disposition=map; residual=supervisor_id=attested|supervisor_enum=AssuranceLevel|axis=evidence_authority; deprecation=active_legacy; catalog_root=$catalog_root_binding
```

### translation_preservation — supervisor.TranslationClass (LPC-031 axis view)

Axis view uses LPC-031 preservation labels (distinct from adapter preservation
short names above for `bounded` / `conservative`). Residual keeps the
supervisor translation-class token.

```supervisor-map
domain: translation_preservation
surface: supervisor.TranslationClass
fail_closed: true
catalog_root: $catalog_root_binding
exact: canonical_identity=exact; disposition=map; residual=supervisor_id=exact|supervisor_enum=TranslationClass|axis=translation_preservation; deprecation=active_legacy; catalog_root=$catalog_root_binding
equisatisfiable: canonical_identity=equisatisfiable; disposition=map; residual=supervisor_id=equisatisfiable|supervisor_enum=TranslationClass|axis=translation_preservation; deprecation=active_legacy; catalog_root=$catalog_root_binding
bounded_abstraction: canonical_identity=bounded_abstraction; disposition=map; residual=supervisor_id=bounded_abstraction|supervisor_enum=TranslationClass|axis=translation_preservation; deprecation=active_legacy; catalog_root=$catalog_root_binding
conservative_approximation: canonical_identity=conservative_approximation; disposition=map; residual=supervisor_id=conservative_approximation|supervisor_enum=TranslationClass|axis=translation_preservation; deprecation=active_legacy; catalog_root=$catalog_root_binding
heuristic: canonical_identity=heuristic; disposition=map; residual=supervisor_id=heuristic|supervisor_enum=TranslationClass|axis=translation_preservation; deprecation=active_legacy; catalog_root=$catalog_root_binding
```

---

## Reject-merge surfaces (non-logic name collisions)

These enums share English names with logic evidence axes but are **not** logic
authority/kind vocabularies. Any attempt to map them as catalog semantic
identities fails closed.

```supervisor-map
domain: reject_merge
surface: goal_quality.EvidenceAuthority
disposition: reject_merge
fail_closed: true
catalog_root: $catalog_root_binding
```

```supervisor-map
domain: reject_merge
surface: prompt_workflow.EvidenceAuthority
disposition: reject_merge
fail_closed: true
catalog_root: $catalog_root_binding
```

```supervisor-map
domain: reject_merge
surface: plan_analysis.EvidenceAuthority
disposition: reject_merge
fail_closed: true
catalog_root: $catalog_root_binding
```

```supervisor-map
domain: reject_merge
surface: planner_doctor.EvidenceAuthorityClass
disposition: reject_merge
fail_closed: true
catalog_root: $catalog_root_binding
```

```supervisor-map
domain: reject_merge
surface: repository_surface.EvidenceKind
disposition: reject_merge
fail_closed: true
catalog_root: $catalog_root_binding
```

---

## Operational-only surface

```supervisor-map
domain: operational_only
surface: supervisor.ResourceBudget
disposition: operational_only
fail_closed: true
catalog_root: $catalog_root_binding
```

`ResourceBudget` is an operational execution envelope (wall time, CPU, memory,
disk, processes, premises, output, tokens, quota, network). It is **not**
`LogicBoundedness` and does not project onto a catalog semantic identity.

---

## Residual collapse examples (no silent identity loss)

| Supervisor A | Supervisor B | Shared canonical_identity | Residual restores |
| --- | --- | --- | --- |
| `flogic` | `frame` | `frame_logic` | exact `LogicFamily` member |
| `protocol` | `runtime_trace` | `trace_conformance` | exact `PropertyKind` member |
| `kernel_check` | `first_order_theorem` | `theorem` | exact `PropertyKind` member |
| `zkp_attestation` | `cryptographic_attestation` | `attestation` | primary wire residual |
| `none` (assurance) | `unverified` | `none` | primary wire residual |

## Adapter cutover contract

| Rule | Enforcement |
| --- | --- |
| Single lazy boundary | `SupervisorCanonicalLogicAdapter` never cold-imports datasets |
| New writes | Canonical identities only on new supervisor records |
| Legacy reads | Explicit map rows + residual reverse map |
| Unknown values | `CanonicalLogicAdapterError` / cutover lookup error |
| Catalog root | Bound to sealed snapshot content root, not Git layout |
| Hand lists | Domain rows here are the generated inventory; adapter maps must stay consistent with these rows |

## Relationship to neighboring tasks

| Task | Relationship |
| --- | --- |
| LPC-023 generated catalogs | Catalog / generated projection supplies provider and translation ids |
| LPC-031 legacy enum maps | Axis surfaces re-stated here with catalog root + residual + deprecation |
| LPC-091 type classification | Classifies leftover public types after this cutover inventory |
| LPC-100 manifest | Handshake pins the same catalog root |
| LPC-110 client | Consumes these maps through the lazy adapter boundary |

## What this task does **not** do

* Does not remove public supervisor types (LPC-091 / later migration).
* Does not implement `SupervisorLogicPlatformClient` (LPC-110).
* Does not claim live prover availability from catalog presence.
* Does not hand-edit baseline family / provider inventories in datasets.

## File ownership

| Path | Role |
| --- | --- |
| `data/agent_supervisor/logic_platform_canonicalization/notes/supervisor_map_cutover.md` | This generated cutover artifact |
| `test/api/test_canonical_logic_adapter.py` | Fail-closed regression suite |
| `ipfs_accelerate_py/agent_supervisor/proof/canonical_logic_adapter.py` | Lazy projection boundary (read consistency) |

## Acceptance

- Every inventoried supervisor domain above has a `supervisor-map` block.
- Every mapped legacy value carries `canonical_identity`, `disposition`,
  `residual`, `deprecation`, and `catalog_root`.
- Unknown values fail closed.
- Residual collapses restore exact supervisor identities.
- Catalog root binds to `DEFAULT_CANONICAL_CATALOG_SNAPSHOT.content_root`.
- Validation: `python -m pytest test/api/test_canonical_logic_adapter.py -q`
