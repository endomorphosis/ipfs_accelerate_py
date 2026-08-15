# LPC-031 Explicit Legacy Enum Mappings

**Task:** LPC-031 — Add explicit legacy enum mappings  
**Goal:** LPC-G030  
**Depends on:** LPC-030 (`ir_core/axes.py`, `notes/axis_migration.md`)  
**Schema:** `logic-legacy-axis-map/v1`  
**Module (contract):** `ipfs_datasets_py.logic.ir_core.legacy_axis_map`  
**Validation:** `python -m pytest ipfs_datasets_py/tests/unit/logic/ir_core/test_legacy_axis_map.py -q`

## Purpose

Every inventoried overlapping status / verdict / availability / evidence /
authority / boundedness / translation enum gains an **explicit, fail-closed**
label map onto the seven orthogonal canonical axes from LPC-030. Unknown
labels raise; non-logic name collisions are rejected rather than silently
merged.

This note is the durable mapping inventory. Executable regression coverage
lives in `test_legacy_axis_map.py` and must stay exhaustive with these tables.

## Fail-closed policy

1. **Known label only.** A mapper accepts only labels listed for its surface.
2. **Unknown → error.** Any other string, empty label, or wrong-surface label
   raises `LegacyAxisMapError` (subclass of `AxisValidationError` / `ValueError`).
3. **No silent merge.** Homonymous non-logic enums (`goal_quality`, prompt
   workflow, planner-doctor, repository surface inventory) use disposition
   `reject_merge` and never project onto logic axes.
4. **No authority promotion.** Mapping a lifecycle `succeeded` label never
   yields `LogicEvidenceAuthority.authoritative` or a conclusive verdict.
5. **Multi-axis splits stay explicit.** Overlapping composites such as
   `VerificationStatus` emit independent axis fields; missing fields stay
   `unknown` / `not_applicable` rather than inferred.

## Canonical target axes (LPC-030)

| Axis field | Type | Generation |
| --- | --- | --- |
| `operation_status` | `LogicOperationStatus` | `LogicOperationStatus@1` |
| `semantic_verdict` | `LogicSemanticVerdict` | `LogicSemanticVerdict@1` |
| `availability` | `LogicAvailability` | `LogicAvailability@1` |
| `evidence_kind` | `LogicEvidenceKind` | `LogicEvidenceKind@1` |
| `evidence_authority` | `LogicEvidenceAuthority` | `LogicEvidenceAuthority@1` |
| `boundedness` | `LogicBoundedness` | `LogicBoundedness@1` |
| `translation_preservation` | `LogicTranslationPreservation` | `LogicTranslationPreservation@1` |

## Inventoried surface catalog

| Surface id | Live type / path | Classification | Disposition | Target |
| --- | --- | --- | --- | --- |
| `supervisor.AttemptStatus` | `proof.formal_verification_contracts.AttemptStatus` | legacy | map | `operation_status` |
| `datasets.ir_core.AttemptStatus` | `logic.ir_core.protocols.AttemptStatus` | legacy | map | `operation_status` |
| `supervisor.ProofVerdict` | `proof.formal_verification_contracts.ProofVerdict` | legacy | map | `semantic_verdict` |
| `datasets.ir_core.ResultStatus` | `logic.ir_core.protocols.ResultStatus` | legacy / overlapping | map | `semantic_verdict` |
| `datasets.VerificationStatus` | `logic.verification_api.VerificationStatus` | unresolved / overlapping | multi_axis | status + availability + verdict |
| `datasets.FeatureAvailability` | `logic.verification_api.FeatureAvailability` | legacy | map | `availability` |
| `datasets.AvailabilityStatus` | `logic.conformance.matrix.AvailabilityStatus` | legacy | map | `availability` |
| `supervisor.SupportStatus` | `proof.program_contracts.SupportStatus` | legacy (support posture) | map | `availability` |
| `families.EvidenceKind` | `logic.families.models.EvidenceKind` | legacy | map | `evidence_kind` |
| `supervisor.EvidenceKind` | `proof.formal_verification_contracts.EvidenceKind` | legacy | map | `evidence_kind` |
| `datasets.ir_core.EvidenceKind` | `logic.ir_core.evidence.EvidenceKind` | legacy | map | `evidence_kind` |
| `families.EvidenceAuthority` | `logic.families.models.EvidenceAuthority` | legacy | map | `evidence_authority` |
| `supervisor.EvidenceAuthority` | `proof.formal_verification_contracts.EvidenceAuthority` | legacy (producer boundary) | map | `evidence_authority` |
| `supervisor.AssuranceLevel` | `proof.formal_verification_contracts.AssuranceLevel` | legacy (lattice) | map | `evidence_authority` |
| `families.BoundednessKind` | `logic.families.models.BoundednessKind` | legacy | map | `boundedness` |
| `parsers.BoundednessKind` | parser-local `BoundednessKind` copies | legacy / local | map | `boundedness` |
| `families.TranslationKind` | `logic.families.models.TranslationKind` | legacy | map | `translation_preservation` |
| `families.PreservationKind` | `logic.translations.family_extensions.PreservationKind` | legacy | map | `translation_preservation` |
| `supervisor.TranslationClass` | `proof.logic_translation_validation.TranslationClass` | legacy | map | `translation_preservation` |
| `goal_quality.EvidenceAuthority` | `objectives.goal_quality.EvidenceAuthority` | duplicate / non-logic | reject_merge | — |
| `prompt_workflow.EvidenceAuthority` | `prompt.prompt_workflow.EvidenceAuthority` | duplicate / non-logic | reject_merge | — |
| `plan_analysis.EvidenceAuthority` | `planning.plan_analysis_query_planner.EvidenceAuthority` | duplicate / non-logic | reject_merge | — |
| `planner_doctor.EvidenceAuthorityClass` | `validation.planner_doctor_live_benchmark.EvidenceAuthorityClass` | duplicate / non-logic | reject_merge | — |
| `repository_surface.EvidenceKind` | `analysis.repository_surface_inventory.EvidenceKind` | duplicate / non-logic | reject_merge | — |
| `supervisor.ResourceBudget` | `proof.formal_verification_contracts.ResourceBudget` | compatibility facade | operational_only | not boundedness |

Machine-readable blocks below use fenced `legacy-map` sections. Tests parse
every block. Label lines are `legacy_label: target` (single axis) or
`legacy_label: axis=value; axis=value` (multi-axis).

---

## Mapping tables

### supervisor.AttemptStatus → operation_status

```legacy-map
surface: supervisor.AttemptStatus
target_axis: operation_status
disposition: map
fail_closed: true
planned: planned
running: running
succeeded: succeeded
failed: failed
unsupported: unsupported
unavailable: unavailable
timed_out: timed_out
cancelled: cancelled
blocked: blocked
```

Notes: `succeeded` means the attempt finished cleanly. It does **not** imply
`proved`, `authoritative`, or any other semantic/trust claim.

### datasets.ir_core.AttemptStatus → operation_status

```legacy-map
surface: datasets.ir_core.AttemptStatus
target_axis: operation_status
disposition: map
fail_closed: true
succeeded: succeeded
failed: failed
timed_out: timed_out
unavailable: unavailable
cancelled: cancelled
```

### supervisor.ProofVerdict → semantic_verdict

```legacy-map
surface: supervisor.ProofVerdict
target_axis: semantic_verdict
disposition: map
fail_closed: true
proved: proved
disproved: disproved
inconclusive: inconclusive
unsupported: unsupported
error: error
cancelled: cancelled
```

### datasets.ir_core.ResultStatus → semantic_verdict

```legacy-map
surface: datasets.ir_core.ResultStatus
target_axis: semantic_verdict
disposition: map
fail_closed: true
proved: proved
disproved: disproved
satisfiable: satisfiable
unsatisfiable: unsatisfiable
monitor_satisfied: satisfied
monitor_violated: violated
ready: not_applicable
not_ready: not_applicable
approved: authorized
rejected: denied
unknown: unknown
error: error
```

### datasets.VerificationStatus → multi-axis split

Overlapping verification-API terminal status splits across lifecycle,
availability posture, and semantic verdict. Verdict is never inferred from
success alone.

```legacy-map
surface: datasets.VerificationStatus
disposition: multi_axis
fail_closed: true
succeeded: operation_status=succeeded; availability=available; semantic_verdict=unknown
partial: operation_status=partial; availability=available; semantic_verdict=unknown
unsupported: operation_status=unsupported; availability=unsupported; semantic_verdict=unsupported
unavailable: operation_status=unavailable; availability=unavailable; semantic_verdict=not_applicable
invalid: operation_status=invalid; availability=unknown; semantic_verdict=not_applicable
error: operation_status=error; availability=unknown; semantic_verdict=error
declarative: operation_status=partial; availability=declared; semantic_verdict=not_applicable
```

### datasets.FeatureAvailability → availability

```legacy-map
surface: datasets.FeatureAvailability
target_axis: availability
disposition: map
fail_closed: true
declared: declared
available: available
unavailable: unavailable
unsupported: unsupported
absent: absent
opt_in: opt_in
```

### datasets.AvailabilityStatus → availability

```legacy-map
surface: datasets.AvailabilityStatus
target_axis: availability
disposition: map
fail_closed: true
declared: declared
not_declared: absent
source_missing: source_missing
not_probed: not_probed
unknown: unknown
```

### supervisor.SupportStatus → availability

Support posture (representability in a program contract) projects onto the
availability axis only. It is not operation status and not a verdict.

```legacy-map
surface: supervisor.SupportStatus
target_axis: availability
disposition: map
fail_closed: true
supported: available
unsupported: unsupported
assumed: declared
not_applicable: unknown
unknown: unknown
```

### families.EvidenceKind → evidence_kind

```legacy-map
surface: families.EvidenceKind
target_axis: evidence_kind
disposition: map
fail_closed: true
kernel_checked_proof: kernel_checked_proof
checked_proof: checked_proof
proof_certificate: proof_certificate
unsat_core: unsat_core
model: model
counterexample: counterexample
trace: trace
monitor_verdict: monitor_verdict
policy_decision: policy_decision
attestation: attestation
candidate: candidate
declaration: declaration
```

### supervisor.EvidenceKind → evidence_kind

```legacy-map
surface: supervisor.EvidenceKind
target_axis: evidence_kind
disposition: map
fail_closed: true
unknown: unknown
llm_output: llm_output
atp_candidate: atp_candidate
smt_candidate: smt_candidate
solver_result: solver_result
kernel_verification: kernel_checked_proof
test_result: test_result
static_analysis: static_analysis
cryptographic_attestation: attestation
cache_entry: cache_entry
```

Notes: compatibility spelling `zkp_attestation` shares the wire value
`cryptographic_attestation` on the live enum and therefore maps identically
through that label. Kind never upgrades authority.

### datasets.ir_core.EvidenceKind → evidence_kind

```legacy-map
surface: datasets.ir_core.EvidenceKind
target_axis: evidence_kind
disposition: map
fail_closed: true
source: source
artifact: artifact
test_result: test_result
proof_receipt: proof_certificate
runtime_observation: runtime_observation
review: review
attestation: attestation
model_output: model_output
other: other
```

### families.EvidenceAuthority → evidence_authority

```legacy-map
surface: families.EvidenceAuthority
target_axis: evidence_authority
disposition: map
fail_closed: true
authoritative: authoritative
independently_checkable: independently_checkable
bounded: bounded
advisory: advisory
none: none
```

### supervisor.EvidenceAuthority → evidence_authority

Supervisor proof `EvidenceAuthority` labels are **producer/checker boundary**
names. They project onto trust ceilings conservatively (no silent promotion to
`authoritative` from a boundary name alone).

```legacy-map
surface: supervisor.EvidenceAuthority
target_axis: evidence_authority
disposition: map
fail_closed: true
unknown: unknown
provider: advisory
llm: advisory
atp: advisory
smt: advisory
solver: bounded
kernel: independently_checkable
attestation_verifier: independently_checkable
validation_runner: bounded
cache: none
```

### supervisor.AssuranceLevel → evidence_authority

Ordered assurance lattice projects onto the authority ceiling axis. Lattice
rank is not an inference source from operation success.

```legacy-map
surface: supervisor.AssuranceLevel
target_axis: evidence_authority
disposition: map
fail_closed: true
unverified: none
candidate: advisory
solver_checked: bounded
kernel_verified: independently_checkable
attested: authoritative
```

Notes: compatibility spellings `none`→`unverified` and
`solver_verified`→`solver_checked` share live wire values and map through
those wire labels.

### families.BoundednessKind → boundedness

```legacy-map
surface: families.BoundednessKind
target_axis: boundedness
disposition: map
fail_closed: true
unbounded: unbounded
finite_domain: finite_domain
finite_trace: finite_trace
step_bounded: step_bounded
resource_bounded: resource_bounded
approximate: approximate
not_applicable: not_applicable
```

### parsers.BoundednessKind → boundedness

Union of parser-local boundedness labels (state, fixed-point, finite-field,
normative, argumentation, hyper, resource, …). Domain-specific finite scopes
collapse to the closest canonical semantic bound; they do not become
operational `ResourceBudget` fields.

```legacy-map
surface: parsers.BoundednessKind
target_axis: boundedness
disposition: map
fail_closed: true
unbounded: unbounded
finite_state: finite_domain
step_bounded: step_bounded
finite_field: finite_domain
fixed_bit_width: finite_domain
finite_range: finite_domain
finite_circuit: finite_domain
resource_bounded: resource_bounded
finite_theory: finite_domain
finite_framework: finite_domain
bounded_unrolling: step_bounded
model_check: finite_domain
finite_trace: finite_trace
finite_heap: finite_domain
finite_schedule: finite_domain
finite_simulation: finite_domain
finite_self_composition: finite_domain
approximate: approximate
not_applicable: not_applicable
unknown: unknown
```

### families.TranslationKind → translation_preservation

```legacy-map
surface: families.TranslationKind
target_axis: translation_preservation
disposition: map
fail_closed: true
lossless: lossless
sound_over_approximation: sound_over_approximation
sound_under_approximation: sound_under_approximation
equisatisfiable: equisatisfiable
heuristic: heuristic
```

### families.PreservationKind → translation_preservation

```legacy-map
surface: families.PreservationKind
target_axis: translation_preservation
disposition: map
fail_closed: true
lossless: lossless
sound_over_approximation: sound_over_approximation
sound_under_approximation: sound_under_approximation
equisatisfiable: equisatisfiable
bounded: bounded_abstraction
heuristic: heuristic
```

### supervisor.TranslationClass → translation_preservation

```legacy-map
surface: supervisor.TranslationClass
target_axis: translation_preservation
disposition: map
fail_closed: true
exact: exact
equisatisfiable: equisatisfiable
bounded_abstraction: bounded_abstraction
conservative_approximation: conservative_approximation
heuristic: heuristic
```

---

## Reject-merge surfaces (non-logic name collisions)

These enums share English names with logic evidence axes but are **not** logic
authority/kind vocabularies. Any attempt to map them as logic axes fails
closed.

### goal_quality.EvidenceAuthority

```legacy-map
surface: goal_quality.EvidenceAuthority
disposition: reject_merge
fail_closed: true
```

Live labels (documentation only; not mapped): `diagnostic`, `proposal`,
`validation`, `proof`, `operator`, `completion_gate`.

### prompt_workflow.EvidenceAuthority

```legacy-map
surface: prompt_workflow.EvidenceAuthority
disposition: reject_merge
fail_closed: true
```

Live labels (documentation only; not mapped): `prompt`, `scan_advisory`,
`verified`, `authoritative`.

### plan_analysis.EvidenceAuthority

```legacy-map
surface: plan_analysis.EvidenceAuthority
disposition: reject_merge
fail_closed: true
```

Live labels (documentation only; not mapped): `prompt_nomination`,
`model_nomination`, `retrieval_nomination`, `current_root_fact`,
`reviewed_contract`, `reviewed_policy`, `security_analysis`,
`bounded_observation`, `proof_receipt`, `counterexample`.

### planner_doctor.EvidenceAuthorityClass

```legacy-map
surface: planner_doctor.EvidenceAuthorityClass
disposition: reject_merge
fail_closed: true
```

Live labels (documentation only; not mapped): `live-service-execution`,
`model-conformance-evidence-only`, `synthetic-fixture-observation`,
`skipped-not-promotion-eligible`.

### repository_surface.EvidenceKind

```legacy-map
surface: repository_surface.EvidenceKind
disposition: reject_merge
fail_closed: true
```

Live labels (documentation only; not mapped): `definition`, `import`,
`caller`, `registration`, `test`, `documentation`, `export`,
`classification`, `relationship`, `contradiction`.

---

## Operational-only surface

### supervisor.ResourceBudget

```legacy-map
surface: supervisor.ResourceBudget
disposition: operational_only
fail_closed: true
```

`ResourceBudget` carries wall time, CPU, memory, disk, process, premise,
output, token, quota, and network fields. It is an operational execution
envelope, **not** `LogicBoundedness`. Attempts to treat budget fields as
boundedness labels fail closed. Semantic boundedness must be set on the
`boundedness` axis explicitly.

---

## Critical non-inference examples

| Legacy observation | Mapped axes | Forbidden inference |
| --- | --- | --- |
| `AttemptStatus.succeeded` | `operation_status=succeeded` | not `semantic_verdict=proved` |
| `AttemptStatus.succeeded` | `operation_status=succeeded` | not `evidence_authority=authoritative` |
| `VerificationStatus.succeeded` | status=succeeded, availability=available, verdict=unknown | not kernel authority |
| `EvidenceKind.kernel_verification` | `evidence_kind=kernel_checked_proof` | not automatic `authoritative` |
| `AssuranceLevel.attested` | `evidence_authority=authoritative` | still independent of operation status |
| `goal_quality.EvidenceAuthority.proof` | reject_merge | never becomes logic authority |

## Acceptance

- Every inventoried legacy surface above has an explicit `legacy-map` block.
- Single-axis and multi-axis maps cover every live label on mapped surfaces.
- Unknown labels fail closed.
- Reject-merge and operational-only surfaces never project silently.
- Validation: `python -m pytest ipfs_datasets_py/tests/unit/logic/ir_core/test_legacy_axis_map.py -q`
