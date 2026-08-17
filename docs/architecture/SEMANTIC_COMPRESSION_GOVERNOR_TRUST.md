# Semantic Compression Governor — Trust, Privacy, and Assurance Model

Status: operator-facing trust surface for program
`semantic-compression-governor-v1`

Interfaces: `SemanticGovernorTrustModel@1`, `scg/trust-docs@1`

Related:

- Plan: [`SEMANTIC_COMPRESSION_GOVERNOR_PLAN.md`](./SEMANTIC_COMPRESSION_GOVERNOR_PLAN.md)
- Operator guide: [`../guides/SEMANTIC_COMPRESSION_GOVERNOR.md`](../guides/SEMANTIC_COMPRESSION_GOVERNOR.md)
- Implementation packages:
  - `ipfs_accelerate_py.agent_supervisor.semantic_governor`
  - `ipfs_datasets_py.logic.software_contracts.semantic_governor`
  - kit durable policy/promotion CAS under `DurableCoordinationStore`

This document states exact authority, non-claims, and evidence classes.
Empirical calibration, content-addressed seals, and Merkle commitments are
**not** universal semantic proof. Operators who only need runbooks should start
with the guide; this file is the normative trust vocabulary.

---

## 1. System boundary

The Semantic Compression Governor (`SemanticCompressionGovernor`) is an
**auditor and proposal system**, not an autonomous production self-modifier.

| The governor may | The governor may not |
| --- | --- |
| Collect bounded audit and shadow evidence | Rewrite production safety rules without authorized CAS |
| Compare compressed vs expanded isolated runs | Lower high-risk assurance by self-authorization |
| Diagnose omissions and plan bounded expansion | Treat model agreement as proof of equivalence |
| Update empirical calibration profiles | Mark a heuristic capsule as exact |
| Propose typed declarative rules | Disable the required full-suite fallback |
| Evaluate candidates on held-out partitions | Suppress verification failures |
| Recommend promotion with receipts | Mutate production code outside isolated evaluation worktrees |
| Seal artifact identities after decisions | Claim every compressed context is semantically complete |

Optimization objective (plan §1):

```text
minimize context and inference cost
subject to accepted-patch quality, dependency coverage, verification,
risk, privacy, and authorization constraints
```

Token reduction is never an acceptance criterion by itself.

---

## 2. Roles and trust domains

| Role | Responsibility | Authority ceiling |
| --- | --- | --- |
| **Operator** | Supplies trusted configuration, promotion/rollback authorization CIDs, policy store, workspace, and operation ids | Can authorize CAS publication when all gates pass |
| **Governor runtime** | Audits, shadows, diagnoses, expands, calibrates, evaluates; emits content-addressed receipts | Proposal and measurement only; never self-promotes |
| **Policy / promotion CAS store** | Expected-generation compare-and-swap for policy and optional promotion heads | Mutates head only on matching expected version + successful gate revalidation |
| **Model provider** | Produces patches / completions under disclosure policy | Untrusted for keys, assurance, sampling rates, proof systems, and promotion |
| **Untrusted task text** | Comments, docstrings, issues, tests, logs, retrieved docs | Audit evidence only; cannot change decisions |
| **Incremental proof sealer** (external release dependency) | Optional seal of bound artifact identities | Bounded claims only; typed `unavailable` when absent |
| **IVP Merkle commitment** | Structural non-ZK commitment of verification bundles | Explicitly **not** a sealer or ZK/execution proof |

There is **no** ambient promotion path. Model output, evaluation reports,
candidates, seals, and qualifications cannot authorize themselves
(`self_promotion_forbidden` / `self_authorization_forbidden`).

---

## 3. Evidence classes: structural, empirical, heuristic, unavailable, formally proven

Operators must label every claim with one of these classes. Mixing classes
silently is a trust defect.

### 3.1 Structural

**Structural** evidence is deterministic over typed artifacts and closed
schemas. Examples:

- Content identity and CID recomputation (`software_contracts.content`)
- Schema/integrity validation of policies, candidates, and reports
- Graph/coverage inventory: included/excluded symbols and exclusion reasons
- Partition membership checks (calibration / development / held-out identities
  are disjoint and immutable)
- Expected-generation CAS identity matching (ABA-safe publication)
- Bounded seal claim set encoding (exact artifacts evaluated, thresholds
  applied, promoted policy equals evaluated candidate)
- Privacy redaction and private-field rejection on public projections

Structural evidence can prove **inventory consistency**, **identity binding**,
and **gate satisfaction**. It does **not** prove semantic sufficiency of a
compressed context for arbitrary future tasks.

### 3.2 Empirical

**Empirical** evidence is measured success/failure rates over sealed
observations with closed integer accounting (basis points, counters, micros).
Examples:

- Capsule/task/route calibration rates with Wilson intervals
- Held-out critical-omission detection and stale-rejection non-regression
- Median context reduction, expansion precision/recall
- Shadow differential outcome distributions
- Cost: gross/net savings including audit overhead

Rules:

- Empirical success may change **routing frequency and audit sampling** only.
- Empirical success **never upgrades formal exactness** and never sets
  `allow_heuristic_as_exact=true`.
- Simulated and live cohorts are **never mixed** into one quality counter.
- Empty populations yield **missing** rates, never fabricated zeros used as
  success.

Current checked-in benchmark summary
(`artifacts/agent_supervisor/semantic_compression_governor/summary.json`) is
**simulated**, `production_eligible: false`, and must not be relabeled as live
model quality.

### 3.3 Heuristic

**Heuristic** evidence is approximate or incomplete analysis:

- Capsule confidence `heuristic` or `opaque` (datasets closed taxonomy:
  `exact` | `conservative` | `heuristic` | `opaque`)
- Heuristic exclusion labels on critical dependencies (**always reject**)
- Model free-text claims, agreement between models, or “looks complete”
- Ranking features that nominate expansion hypotheses

Rules:

- Heuristic / opaque / invalid / stale capsules require raw source (or fail
  closed).
- A heuristic capsule must never be treated as exact.
- Heuristic signals may trigger shadowing or expansion; they never authorize
  promotion.

### 3.4 Unavailable

**Unavailable** is a first-class terminal status, not a soft “unknown success”.

| Surface | Typed unavailable meaning |
| --- | --- |
| Incremental proof sealer public API | Seal status `unavailable`; promotion blocked for proof-backed seal path |
| Live provider receipts / live quality cohort | Missing live quality; simulated results only |
| Real provers / formal kernels | Verification may be `unavailable` / `not-modeled` |
| Cost sensors | Unit costs null; not zero |
| Release qualification | Absent or unavailable qualification blocks promotion |
| CAS store / corruption | `cas_unavailable` / `cas_corrupt`; head unchanged |

Fail-closed: missing evidence is reported as missing. It is never inferred from
a similar-looking class name (for example, an IVP `VerificationCommitment` is
not a sealer).

### 3.5 Formally proven

**Formally proven** is reserved for independently reconstructed kernel/proof
receipts that actually encode the claimed property under a pinned toolchain and
policy.

For this program:

| Claim | Formally proven today? |
| --- | --- |
| Exact artifact CIDs were bound into a seal / commitment | Only if a released sealer (or separately authorized qualification path) actually verifies; otherwise unavailable |
| Declared evaluation thresholds were applied to a candidate | Structural + empirical report identity — not a semantic completeness proof |
| Compressed context is sufficient for all tasks | **Never** claimed as formally proven |
| Model agreement equals behavioral equivalence | **Forbidden claim** |
| IVP Merkle commitment is ZK or execution proof | **Forbidden substitution** |

The maximum justified system claim remains (plan §16):

> The system empirically and structurally audits semantic compression,
> detects and diagnoses omitted-context failures, expands context using
> counterexamples and dependency evidence, calibrates future compression
> decisions, and promotes rule changes only after held-out evaluation and
> authorized, reproducible qualification.

It never claims to prove that every compressed context is semantically complete.

---

## 4. Confidence, freshness, and verification vocabularies

These closed vocabularies are part of the trust surface.

### 4.1 Capsule confidence (datasets)

| Value | Meaning | Required follow-up |
| --- | --- | --- |
| `exact` | Proven structural exactness under admitted analyzers | Still subject to freshness and scope |
| `conservative` | Safe over-approx | Prefer when exact unavailable |
| `heuristic` | Incomplete | Raw source or reject for critical use |
| `opaque` | Unanalyzed / dynamic | Raw source or reject |

### 4.2 Freshness

`fresh` | `stale` | `unknown` — stale or unknown capsules are not production-
admissible for compressed acceptance without raw expansion or fail-closed
rejection.

### 4.3 Verification terminal status (incremental verification)

Includes: `passed`, `failed`, `proved`, `disproved`, `unknown`, `timeout`,
`unavailable`, `not-modeled`, `stale`, `invalid`, `cancelled`, `simulated`.

`simulated` outcomes are cohort-separated and never counted as live quality.

### 4.4 Governor terminal / execution mode

Governor artifacts use closed terminal statuses (`complete`, `rejected`,
`invalid`, `stale`, `inconclusive`, `evaluation_failed`,
`human_review_required`, `unavailable`, `cancelled`, `simulated`) and
execution modes (`live` | `simulated` | `replay`).

---

## 5. Untrusted inputs and instruction quarantine

**Untrusted surfaces:** comments, docstrings, issue/task text, tests, logs,
retrieved documentation, and model-produced free text.

Module: `...semantic_governor.untrusted_input`
(`detect_instruction_like_content`, `UntrustedInstructionEvidence@1`).

Invariants:

1. Detection produces **bounded audit evidence** only (digests + printable
   previews).
2. Instruction-like fragments **cannot** change routing, verification, capsule
   classification, trusted keys, proof systems, sampling, source inclusion, or
   promotion — even when they mimic configuration or authorization language.
3. Deterministic decision functions consume only **trusted configuration
   channels** (`TrustedDecisionConfig`), not quarantined text.
4. Durable payloads reject private-source field markers and model-written
   authority claims (`reject_private_and_model_authority`).

---

## 6. Privacy and disclosure

Module: `ipfs_accelerate_py.agent_supervisor.semantic_governor.privacy`
(evidence `scg/privacy-gate@1`).

| Rule | Enforcement |
| --- | --- |
| Expanded private source is **local-only** by default | Shadow disclosure policy; no ambient external send |
| Broader external disclosure requires **exact provider authorization** | `authorize_shadow_disclosure` on approved provider ids |
| Secrets redacted before provider invocation | `redact_context_for_provider`; marker `[REDACTED]` |
| Public reports store CIDs / managed references | `PublicReportProjection`; not raw private source |
| No arbitrary host filesystem paths in network/public payloads | Absolute/home/`file:` paths rejected; `<managed-path-redacted>` |
| Isolated evaluation worktrees for expanded shadow | Required; expanded run never silently replaces accepted patch |
| No default public server or GUI | CLI emits JSON only |

CLI outputs strip private-key markers and host-path-looking strings. Metrics
and dashboard projections are observability-only and never grant promotion.

---

## 7. Authorization model

Promotion and rollback require **separate trusted authorization** from any
candidate, evaluation, seal, or model artifact.

### 7.1 What counts as authorization

- An operator-supplied **authorization CID** (or mapping whose identity is a
  CID) that is **not** equal to the candidate, evaluation report, proposal,
  policy, qualification, or seal CID.
- CLI: `--authorization <cid>` or payload field `authorization` /
  `authorization_cid` / `external_authorization_cid` /
  `promotion_authorization_cid`.
- Absent authorization → reject with `absent_authorization`; head unchanged.

### 7.2 What never counts as authorization

- Model output or free-text “approve promotion”
- Evaluation `verdict=pass` alone
- Release qualification or seal alone
- Candidate proposing itself
- Ambient environment variables or default configs that imply production CAS
- Simulated benchmark success

### 7.3 High-risk assurance

Protected thresholds (for example forbidding `allow_heuristic_as_exact`,
keeping full-suite fallback enabled) cannot be weakened without explicit
authorization of that reduction. Publication re-checks
`high_risk_assurance_reduced` and protected-threshold reductions; reductions
without authority yield `high_risk_assurance_reduced` or
`protected_threshold_reduction_unauthorized`.

---

## 8. Held-out evaluation (pre-promotion)

API: `evaluate_rule_candidate` / CLI `evaluate-policy`
(evidence `scg/held-out-evaluation@1`).

| Gate | Fail-closed reason (examples) |
| --- | --- |
| Held-out cases present | `missing_held_out_data` |
| Partition is held-out | `partition_not_held_out` |
| No overlap with calibration / development / candidate-generating ids | `held_out_partition_overlap`, `candidate_generating_case_in_held_out` |
| Critical omission detection non-regression | `critical_omission_detection_regressed` |
| Stale rejection non-regression | `stale_rejection_regressed` |
| No hidden accepted regression | `hidden_accepted_regression` |
| High-risk assurance preserved | `high_risk_assurance_reduced` |
| Full-suite fallback remains enabled | `full_suite_fallback_disabled` |
| Declared cost/context thresholds | `median_context_reduction_below_threshold`, etc. |

Evaluation is **pure and emit-only**: it never mutates the candidate,
benchmark, or live policy head. A `pass` verdict is necessary for promotion but
**not sufficient**.

---

## 9. Release qualification and sealing

Module: `...semantic_governor.sealing` (evidence `scg/seal-binding@1`,
`scg/release-qualification@1`).

### 9.1 Qualification paths

| Path | Meaning |
| --- | --- |
| `incremental_seal` | Current released incremental-seal evidence present and bound |
| `authorized_release_qualification` | Independent `VerificationBundle`-backed path with separate authorization |
| `blocked` | Missing sealer, missing auth, identity mismatch, or overclaim |

IVP `VerificationCommitment` is **structural non-ZK** and **cannot** satisfy
the sealer path (`ivp_commitment_not_sealer`).

### 9.2 Bounded seal claims (allowed)

Seals may encode only:

1. `exact_artifacts_evaluated`
2. `required_evaluations_completed`
3. `declared_thresholds_applied`
4. `no_blocking_status_omitted`
5. `promoted_policy_equals_evaluated_candidate`

### 9.3 Forbidden claim kinds

Never encoded unless separately and actually proven:

- `semantic_sufficiency`
- `universal_semantic_completeness`
- `zk_proof` / `zero_knowledge_proof`
- `execution_proof` / `proof_of_test_execution`
- `ivp_commitment_is_sealer`
- `full_suite_implied`
- `model_agreement_is_equivalence`

Post-decision sealing **binds** identities; it **never upgrades** evidence
semantics.

Artifact roles that may be bound: benchmark, context_pack, verification_bundle,
differential_report, calibration_profile, candidate, promotion_decision,
evaluation_report, policy, incremental_seal.

---

## 10. Promotion and rollback CAS

APIs: `promote_compression_policy`, `rollback_compression_policy`
(evidence `scg/authorized-promotion@1`).

### 10.1 Promotion publication gates (all revalidated at publish time)

| Condition | Head mutates? | Typical reason |
| --- | --- | --- |
| Stale candidate (base ≠ live head) | No | `stale_candidate` |
| Absent release qualification | No | `absent_release_qualification` |
| Unavailable release qualification | No | `unavailable_release_qualification` |
| Absent authorization | No | `absent_authorization` |
| Self-promotion (auth CID collides with artifact) | No | `self_promotion_forbidden` |
| High-risk assurance reduced | No | `high_risk_assurance_reduced` |
| Evaluation not pass / mismatch | No | `evaluation_verdict_not_pass`, `mismatched_evaluation` |
| CAS conflict / ABA | No | `cas_conflict` |
| CAS unavailable / corrupt | No | `cas_unavailable`, `cas_corrupt` |
| All gates pass + CAS success | **Yes, once** | status `promoted` |

Statuses: `promoted` | `rejected` | `conflict` | `unavailable` | `corrupt` |
`unchanged`.

### 10.2 Rollback

Rollback is **another authorized expected-generation CAS**, not history
deletion:

- Generation advances; prior policy CIDs remain in history.
- Target is a prior immutable policy CID.
- Self-authorization and absent authorization leave the head unchanged.
- Statuses: `rolled_back` | `rejected` | `conflict` | `unavailable` |
  `corrupt` | `unchanged`.

### 10.3 Recovery

Recovery rebuilds indexes from **verified immutable blocks**. It never invents
promotion, completion, or authorization. Corruption and ambiguous promotion
fail closed. Interrupted audits recover to a consistent prior head.

---

## 11. Metrics authority ceiling

Module: `...semantic_governor.metrics` (evidence `scg/metrics@1`).

Metrics are **observability only**. They never grant acceptance, promotion, or
route authority. Net savings include audit/verification/shadow overhead.
Unavailable measurements stay unavailable.

---

## 12. Known limitations and residual risks

| Area | Limitation |
| --- | --- |
| Dynamic Python | Reflection, import hooks, metaclasses, monkey patching, plugins, generated bindings, native extensions, uncontrolled I/O |
| Capsule quality | Heuristic/opaque regions require raw source; false exact is a critical failure class |
| Formal verification | Real provers may be unavailable; simulated benchmarks are non-authoritative for live quality |
| Sealer dependency | Until released IncrementalProofSealer public API lands, seal status is typed unavailable |
| Benchmark corpus | Partitioned simulated corpus may fail production floors (e.g. median reduction) without inventing success |
| Cost | Local estimators are not live billing evidence |
| Adversarial text | Quarantine is evidence-only; operators must still use trusted config channels |
| Scope | No second semantic index, ZK system, MCP++ execution profile, public GUI, or autonomous production rewriter |

Incident posture: on suspected injection, policy poisoning, or privacy leak —
halt promotion CAS, quarantine the workspace, preserve immutable receipts,
rotate any leaked credentials, and restore via authorized rollback to a known
good policy CID.

---

## 13. Normative implementation pins

| Concern | Location |
| --- | --- |
| Untrusted input scan | `ipfs_datasets_py/.../semantic_governor/untrusted_input.py` |
| Privacy gate | `ipfs_accelerate_py/.../semantic_governor/privacy.py` |
| Held-out evaluation | `.../semantic_governor/policy_evaluation.py` |
| Promotion / rollback | `.../semantic_governor/promotion.py` |
| Sealing / qualification | `.../semantic_governor/sealing.py` |
| Metrics | `.../semantic_governor/metrics.py` |
| CLI | `.../semantic_governor/cli.py` (`semantic-governor`) |
| Public package | `.../semantic_governor` (`SemanticCompressionGovernor@1`) |
| Coverage / confidence | `.../semantic_governor/coverage.py` |
| Policy contracts | `.../semantic_governor/policy_contracts.py` |

---

## 14. Document interfaces

| Interface | Purpose |
| --- | --- |
| `SemanticGovernorTrustModel@1` | This trust/privacy/assurance vocabulary |
| `SemanticGovernorOperations@1` | Operator guide procedures (evaluation, promotion, rollback) |
| Evidence subset | `scg/trust-docs@1` |

Conflict policy for this surface: document exact authority and non-claims;
never present empirical calibration or seals as universal semantic proof.
