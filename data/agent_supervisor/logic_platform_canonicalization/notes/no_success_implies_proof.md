# LPC-032 Forbid Inferring Proof Authority from Provider Success

**Task:** LPC-032 — Forbid inferring proof authority from provider success  
**Goal:** LPC-G030  
**Depends on:** LPC-031 (`notes/legacy_enum_mappings.md`), LPC-030 (`ir_core/axes.py`)  
**Schema:** `logic-axis/v1`  
**Validation:** `python -m pytest ipfs_datasets_py/tests/unit/logic/backends/test_success_is_not_proof.py -q`

## Purpose

Provider (or operation) **success** answers only the lifecycle question: did the
attempt finish without a transport/runtime failure? It does **not** answer:

* what semantic conclusion was reached (`semantic_verdict`);
* what trust ceiling the evidence carries (`evidence_authority`);
* whether a kernel-required policy may admit the result as proof.

This note freezes that non-inference rule. Executable adversarial coverage lives
in `ipfs_datasets_py/tests/unit/logic/backends/test_success_is_not_proof.py`.

## Canonical counterexample (must remain representable)

`succeeded_unknown_advisory_coordinate()` from
`ipfs_datasets_py.logic.ir_core.axes` constructs:

| Axis | Value | Why it is allowed |
| --- | --- | --- |
| `operation_status` | `succeeded` | Attempt completed cleanly |
| `semantic_verdict` | `unknown` | No decisive semantic answer |
| `availability` | `available` | Provider was usable |
| `evidence_kind` | `candidate` | Format only; not trust |
| `evidence_authority` | `advisory` | Non-binding trust ceiling |
| `boundedness` | `unknown` | Scope not established |
| `translation_preservation` | `not_applicable` | No translation claim |

This seven-axis coordinate is a **valid** provider outcome. Axes are orthogonal:
success, unknown verdict, and advisory authority may co-occur.

## Kernel-required policy (fail-closed)

A **kernel-required policy** admits a coordinate only when every of the
following holds. Operation success is never sufficient and never used as a
proxy for any other field.

| Gate | Required | Counterexample value | Pass? |
| --- | --- | --- | --- |
| Evidence authority | `authoritative` or `independently_checkable` | `advisory` | no |
| Semantic verdict | conclusive (`proved`, `disproved`, …) | `unknown` | no |
| Evidence kind | kernel/checked proof family | `candidate` | no |
| Operation status | *not consulted as authority* | `succeeded` | n/a |

Therefore `succeeded + unknown + advisory` **cannot pass** a kernel-required
policy.

### Rank intuition (authority only)

`LogicEvidenceAuthority` total order for ceiling comparison (not inference):

```
unknown < none < advisory < bounded < independently_checkable < authoritative
```

Kernel-required admission needs at least `independently_checkable`. Advisory
evidence remains below that floor even when `operation_status=succeeded`.

## Non-inference rules (normative)

1. **Success ≠ proof.** `LogicOperationStatus.SUCCEEDED` never implies
   `LogicSemanticVerdict.PROVED` (or any other conclusive verdict).
2. **Success ≠ authority.** Helpers
   `evidence_authority_from_operation_status` and
   `semantic_verdict_from_operation_status` always return `unknown`.
3. **Kind ≠ authority.** `candidate` / `kernel_checked_proof` describe format;
   trust is set only on `evidence_authority`.
4. **Backend surfaces must not promote.** A provider wire success
   (`ExecutionOutcome.SUCCEEDED`), ATP success, advisor role, or candidate
   result cannot satisfy kernel/theorem authority requirements.
5. **Request overclaim fails closed.** Evidence kinds such as `candidate` and
   `advisory` cannot claim `RequestAuthorityCeiling.KERNEL`.

## Backend enforcement anchors

These live surfaces already fail closed and are exercised by the adversarial
test:

| Surface | Behaviour under success + advisory/candidate |
| --- | --- |
| `ir_core.axes` non-inference helpers | Always return `unknown` authority/verdict |
| `backends.portfolio.assurance_satisfies` | `advisory` does not satisfy `authoritative` |
| `backends.toolchain_roles.role_can_satisfy_certified_authority` | Advisor + advisory cannot certify |
| `backends.requests_v2` authority overclaim | Candidate/advisory evidence cannot claim kernel |
| `backends.results.CandidateResult.require_authority` | Cannot be used as theorem authority |
| `backends.atp.execution_v2.atp_success_establishes_theorem` | Always `False` |

## Forbidden silent promotions

| Observation | Forbidden inference |
| --- | --- |
| Provider/attempt `succeeded` | `semantic_verdict=proved` |
| Provider/attempt `succeeded` | `evidence_authority=authoritative` |
| Provider/attempt `succeeded` | passes kernel-required policy |
| `evidence_kind=candidate` + success | theorem / kernel authority |
| Advisor role + success | certified authority requirement |
| ATP success / SZS theorem status alone | theorem authority |

## Acceptance

- `succeeded + unknown + advisory` is constructible as a `LogicAxisCoordinate`.
- The same coordinate fails every kernel-required gate above.
- No axis helper promotes operation success into proof authority.
- Validation:
  `python -m pytest ipfs_datasets_py/tests/unit/logic/backends/test_success_is_not_proof.py -q`
