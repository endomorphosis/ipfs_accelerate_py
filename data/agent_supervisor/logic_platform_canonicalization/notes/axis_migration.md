# LPC-030 Orthogonal Canonical Axes

**Task:** LPC-030 — Define orthogonal canonical axes  
**Goal:** LPC-G030  
**Module:** `ipfs_datasets_py.logic.ir_core.axes`  
**Path:** `ipfs_datasets_py/ipfs_datasets_py/logic/ir_core/axes.py`  
**Schema:** `logic-axis/v1`

## Purpose

Replace overlapping status / verdict / availability / evidence / authority /
boundedness / translation enums with **seven distinct closed axes**. Each axis
answers a different question. No axis may be reused as a stand-in for another.

## Orthogonal axes

| # | Axis field | Type | Generation | Question answered |
| --- | --- | --- | --- | --- |
| 1 | `operation_status` | `LogicOperationStatus` | `LogicOperationStatus@1` | Did this attempt finish, and how? |
| 2 | `semantic_verdict` | `LogicSemanticVerdict` | `LogicSemanticVerdict@1` | What semantic conclusion (if any) was reached? |
| 3 | `availability` | `LogicAvailability` | `LogicAvailability@1` | Can the provider/feature be used at all? |
| 4 | `evidence_kind` | `LogicEvidenceKind` | `LogicEvidenceKind@1` | What format/category of evidence was emitted? |
| 5 | `evidence_authority` | `LogicEvidenceAuthority` | `LogicEvidenceAuthority@1` | What trust ceiling does that evidence carry? |
| 6 | `boundedness` | `LogicBoundedness` | `LogicBoundedness@1` | What semantic scope bounds the conclusion? |
| 7 | `translation_preservation` | `LogicTranslationPreservation` | `LogicTranslationPreservation@1` | What guarantee does a translation claim? |

These seven types are pairwise distinct Python enums. Wire values are stable
`str` members. The composite carrier is `LogicAxisCoordinate`, which stores one
independent value per axis and never derives one field from another.

## Axis inventories

### 1. Operation status (`LogicOperationStatus`)

Lifecycle of one provider or operation attempt.

| Value | Meaning |
| --- | --- |
| `planned` | Attempt scheduled, not started |
| `running` | Attempt in flight |
| `succeeded` | Attempt completed without transport/runtime failure |
| `partial` | Attempt completed with disclosed incomplete coverage |
| `failed` | Attempt failed during execution |
| `unsupported` | Operation not supported by the selected provider |
| `unavailable` | Provider/toolchain not available at attempt time |
| `timed_out` | Attempt exceeded its operational budget |
| `cancelled` | Attempt cancelled before terminal success |
| `blocked` | Attempt cannot start due to unmet dependency |
| `invalid` | Request rejected as malformed or inconsistent |
| `error` | Unclassified attempt error |

**Not a semantic verdict.** `succeeded` only means the attempt finished cleanly.

### 2. Semantic verdict (`LogicSemanticVerdict`)

Conclusion about the obligation or query.

| Value | Meaning |
| --- | --- |
| `proved` / `disproved` | Theorem-style conclusion |
| `satisfiable` / `unsatisfiable` | Satisfiability conclusion |
| `satisfied` / `violated` | Model-check or monitor conclusion |
| `authorized` / `denied` | Authorization conclusion |
| `secure` / `attack_found` | Protocol-analysis conclusion |
| `unknown` / `inconclusive` | No decisive semantic answer |
| `unsupported` / `error` / `cancelled` / `not_applicable` | Non-conclusive dispositions |

**Not operation status.** A succeeded attempt may still report `unknown`.

### 3. Availability (`LogicAvailability`)

Capability posture of a provider, feature, or toolchain.

| Value | Meaning |
| --- | --- |
| `declared` | Present in catalog/declaration only |
| `available` | Probe or install confirms usability |
| `unavailable` | Known but not usable now |
| `unsupported` | Explicitly not supported on this surface |
| `not_probed` | Declared; live probe not performed |
| `absent` | Not present |
| `opt_in` | Requires explicit enablement |
| `source_missing` | Source/pin missing |
| `unknown` | Availability undetermined |

**Not operation status.** Availability is pre-attempt posture, not attempt outcome.

### 4. Evidence kind (`LogicEvidenceKind`)

Format or category of evidence. Kind never conveys trust by itself.

Representative values: `kernel_checked_proof`, `checked_proof`,
`proof_certificate`, `unsat_core`, `model`, `counterexample`, `trace`,
`monitor_verdict`, `policy_decision`, `attestation`, `candidate`,
`declaration`, `llm_output`, `atp_candidate`, `smt_candidate`,
`solver_result`, `test_result`, `static_analysis`, `cache_entry`, `source`,
`artifact`, `runtime_observation`, `review`, `model_output`, `other`,
`unknown`.

### 5. Evidence authority (`LogicEvidenceAuthority`)

Trust ceiling of evidence, independent of kind and of operation success.

| Value | Meaning |
| --- | --- |
| `authoritative` | Highest trust ceiling admitted by policy |
| `independently_checkable` | Artifact can be rechecked without the producer |
| `bounded` | Valid only under disclosed bounds |
| `advisory` | Heuristic / non-binding |
| `none` | Explicitly carries no authority |
| `unknown` | Authority not established |

### 6. Boundedness (`LogicBoundedness`)

Semantic scope of a conclusion (domain of validity), not operational resource
budgets alone.

| Value | Meaning |
| --- | --- |
| `unbounded` | No disclosed semantic bound |
| `finite_domain` | Finite domain restriction |
| `finite_trace` | Finite trace / observation window |
| `step_bounded` | Step / depth bound |
| `resource_bounded` | Search limited by resources with semantic effect |
| `approximate` | Approximate conclusion |
| `not_applicable` | Boundedness not meaningful for this result |
| `unknown` | Bound not established |

Supervisor `ResourceBudget` remains an operational carrier; it does not replace
this axis.

### 7. Translation preservation (`LogicTranslationPreservation`)

Semantic guarantee claimed by a translation between representations.

| Value | Meaning |
| --- | --- |
| `lossless` / `exact` | Full semantic preservation claims |
| `equisatisfiable` | Satisfiability preserved both ways |
| `sound_over_approximation` | Sound over-approx |
| `sound_under_approximation` | Sound under-approx |
| `bounded_abstraction` | Abstraction under disclosed bounds |
| `conservative_approximation` | Conservative approx without tighter class |
| `heuristic` | No soundness claim |
| `unknown` / `not_applicable` | Undetermined / no translation |

A lossless translation of advisory evidence remains advisory.

## Critical non-inference rules

1. **Success ≠ proof.** `LogicOperationStatus.SUCCEEDED` does not imply
   `LogicSemanticVerdict.PROVED` (or any other conclusive verdict).
2. **Success ≠ authority.** Operation success never upgrades
   `LogicEvidenceAuthority`. Helpers
   `evidence_authority_from_operation_status` and
   `semantic_verdict_from_operation_status` always return `unknown`.
3. **Kind ≠ authority.** `LogicEvidenceKind` describes format; authority is a
   separate field and must be set explicitly.
4. **Availability ≠ attempt outcome.** A feature may be `available` while a
   particular attempt `failed` or `timed_out`.
5. **Boundedness ≠ resource budget alone.** Wall-time and memory limits are
   operational; boundedness is the semantic claim about validity scope.
6. **Translation preservation ≠ authority.** Preservation class does not raise
   evidence trust.

### Canonical representable counterexample

`succeeded_unknown_advisory_coordinate()` constructs:

| Axis | Value |
| --- | --- |
| operation_status | `succeeded` |
| semantic_verdict | `unknown` |
| availability | `available` |
| evidence_kind | `candidate` |
| evidence_authority | `advisory` |
| boundedness | `unknown` |
| translation_preservation | `not_applicable` |

This coordinate is valid and must remain representable (LPC-032 strengthens
policy checks that reject silent promotion).

## Legacy surfaces (map in LPC-031)

| Legacy surface | Classification | Target axis |
| --- | --- | --- |
| Supervisor `AttemptStatus` | legacy | `LogicOperationStatus` |
| Supervisor `ProofVerdict` | legacy | `LogicSemanticVerdict` |
| Datasets `VerificationStatus` | unresolved / overlapping | status + availability + verdict (split) |
| Datasets / IR `EvidenceKind` (multiple) | legacy / overlapping | `LogicEvidenceKind` |
| Families `EvidenceAuthority` | legacy | `LogicEvidenceAuthority` |
| Supervisor proof `EvidenceAuthority` | legacy | `LogicEvidenceAuthority` (producer boundary labels) |
| Goal-quality / planner-doctor `EvidenceAuthority*` | duplicate / non-logic | do **not** merge silently |
| Families `BoundednessKind` / parser local copies | legacy | `LogicBoundedness` |
| Families `TranslationKind` / supervisor `TranslationClass` | legacy | `LogicTranslationPreservation` |
| `FeatureAvailability` / matrix `AvailabilityStatus` | legacy | `LogicAvailability` |
| Supervisor `ResourceBudget` | compatibility facade | operational only; not boundedness |

LPC-031 owns explicit fail-closed label maps. Unknown labels fail closed.

## Migration steps

1. **LPC-030 (this note):** define the seven axes, generations, coordinate
   type, and non-inference helpers in `ir_core/axes.py`.
2. **LPC-031:** add `legacy_axis_map.py` with exhaustive mappings from
   inventoried enums; unknown labels fail closed.
3. **LPC-032:** adversarial tests that `succeeded + unknown + advisory`
   cannot pass a kernel-required policy.
4. New provider / platform responses carry all seven fields explicitly.
5. Legacy enums remain importable through adapters until cutover tasks retire
   them; do not delete live surfaces in this task.

## Acceptance

- Operation status, semantic verdict, availability, evidence kind, evidence
  authority, boundedness, and translation preservation are **distinct types**.
- A succeeded provider response can still carry unknown / advisory fields.
- No code path in the axes module infers proof authority from operation success.
- Validation: `python -m pytest ipfs_datasets_py/tests/unit/logic/ir_core/test_axes.py -q`
