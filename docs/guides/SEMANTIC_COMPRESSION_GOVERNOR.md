# Semantic Compression Governor — Operator Guide

Program: `semantic-compression-governor-v1`  
Interfaces: `SemanticGovernorOperations@1`, `SemanticGovernorCLI@1`  
Console entry: `semantic-governor`

Normative trust vocabulary:
[`SEMANTIC_COMPRESSION_GOVERNOR_TRUST.md`](../architecture/SEMANTIC_COMPRESSION_GOVERNOR_TRUST.md).

Architecture plan:
[`SEMANTIC_COMPRESSION_GOVERNOR_PLAN.md`](../architecture/SEMANTIC_COMPRESSION_GOVERNOR_PLAN.md).

This guide is the human operator surface for **audit, evaluation, promotion,
rollback, metrics, recovery, and incident response**. It does not authorize
production mutation by itself.

---

## 1. What you are operating

The governor audits whether compressed coding-agent context was sufficient,
runs controlled shadow comparisons, diagnoses omissions, calibrates routes, and
evaluates declarative policy candidates on **held-out** tasks. Promotion of a
candidate is a separate, **authorized expected-generation CAS** with release
qualification.

**Models propose; operators authorize.** Evaluation `pass` is necessary for
promotion but never sufficient without explicit authorization and live CAS.

Evidence class reminder (see trust doc for full definitions):

| Class | Operator use |
| --- | --- |
| **Structural** | Identity, schema, partition, CAS, seal binding |
| **Empirical** | Rates, savings, held-out metrics (integer bp / micros) |
| **Heuristic** | Capsule confidence / ranking; never exact |
| **Unavailable** | Missing sealer, live receipts, provers, sensors — fail closed |
| **Formally proven** | Only independently reconstructed proofs of the exact claim |

---

## 2. Package layout and PYTHONPATH

Primary packages (import side-effect free):

```text
ipfs_accelerate_py.agent_supervisor.semantic_governor   # orchestration, CLI, promotion
ipfs_datasets_py.logic.software_contracts.semantic_governor  # contracts, coverage, rules
```

Typical test/runtime path:

```bash
export PYTHONPATH=ipfs_kit_py:ipfs_datasets_py:ipfs_accelerate_py:.
```

Public Python surface (lazy):

```python
from ipfs_accelerate_py.agent_supervisor.semantic_governor import (
    SemanticCompressionGovernor,
    create_semantic_compression_governor,
    evaluate_context_sufficiency,
    create_shadow_plan,
    compare_shadow_results,
    diagnose_omission,
    plan_context_expansion,
    execute_expansion_loop,
    update_calibration,
    propose_rule_change,
    evaluate_rule_candidate,
    promote_compression_policy,
)
```

Promotion/rollback and sealing are also importable as modules:

```python
from ipfs_accelerate_py.agent_supervisor.semantic_governor.promotion import (
    promote_compression_policy,
    rollback_compression_policy,
)
from ipfs_accelerate_py.agent_supervisor.semantic_governor.policy_evaluation import (
    evaluate_rule_candidate,
)
from ipfs_accelerate_py.agent_supervisor.semantic_governor.sealing import (
    qualify_policy_candidate,
    seal_governor_run,
    verify_governor_seal,
)
```

---

## 3. CLI overview

Entry point: `ipfs_accelerate_py.agent_supervisor.semantic_governor.cli:main`  
Console name: `semantic-governor`

Closed command set (exactly ten):

| Command | Primary API | Purpose |
| --- | --- | --- |
| `audit` | `evaluate_context_sufficiency` | Pre-execution sufficiency / coverage |
| `shadow` | `create_shadow_plan` | Plan/run compressed vs expanded comparison |
| `diagnose` | `diagnose_omission` | Rank omission hypotheses |
| `expand` | `execute_expansion_loop` | Bounded counterexample-guided expansion |
| `calibrate` | `update_calibration` | Empirical profile update (not formal exactness) |
| `propose-rules` | `propose_rule_change` | Typed declarative rule proposals |
| `evaluate-policy` | `evaluate_rule_candidate` | Held-out candidate evaluation (no head mutation) |
| `promote-policy` | `promote_compression_policy` | Authorized CAS publication |
| `report` | `build_governor_report` | Privacy-filtered final report projection |
| `dashboard-data` | `build_dashboard_data` | Privacy-filtered metrics summary |

Default output: **bounded deterministic JSON** on stdout. No public service,
GUI, listener, or provider is started by `--help` or import.

### 3.1 Exit codes

| Code | Meaning |
| --- | --- |
| `0` | Success |
| `1` | Error |
| `2` | Usage |
| `3` | Unavailable |
| `4` | Production gate |

### 3.2 Common invocation pattern

Most commands accept a JSON payload via stdin or `--input` (see `--help` for
the active build). Example shape:

```bash
python -m ipfs_accelerate_py.agent_supervisor.semantic_governor.cli \
  evaluate-policy \
  --input candidate_and_benchmark.json
```

Or:

```bash
semantic-governor evaluate-policy --input candidate_and_benchmark.json
```

CLI JSON **strips** private source, secrets, and host-path-looking strings.

---

## 4. Reproduce held-out evaluation

Evaluation never mutates the policy head. Use it to obtain a content-addressed
`RuleEvaluationReport` before any promotion attempt.

### 4.1 Preconditions

1. Immutable partitions: **calibration**, **development**, and **held-out**
   case identities are disjoint.
2. Candidate-generating case identities do **not** appear in held-out scoring.
3. Candidate binds to a known baseline policy version/CID.
4. Full-suite fallback remains enabled on the candidate policy.
5. `allow_heuristic_as_exact` is false.

### 4.2 Python path (hermetic)

```python
from ipfs_accelerate_py.agent_supervisor.semantic_governor.policy_evaluation import (
    evaluate_rule_candidate,
    HeldOutBenchmark,
)

report = evaluate_rule_candidate(
    candidate,          # CompressionPolicyCandidate or mapping
    held_out_benchmark, # HeldOutBenchmark or mapping
    baseline_policy=baseline_policy,  # optional; identity must match candidate
)

assert report.verdict  # EvaluationVerdict: pass | fail | ...
# report.report_cid, report.candidate_cid, report.blocking_reasons
```

### 4.3 CLI path

```bash
semantic-governor evaluate-policy --input held_out_eval.json
```

Payload must include candidate + held-out benchmark (and optional baseline).
On failure inspect `blocking_reasons` / `reason_code` fields, for example:

- `missing_held_out_data`
- `held_out_partition_overlap`
- `critical_omission_detection_regressed`
- `stale_rejection_regressed`
- `hidden_accepted_regression`
- `high_risk_assurance_reduced`
- `full_suite_fallback_disabled`
- `median_context_reduction_below_threshold`

### 4.4 Interpreting the verdict

| Verdict | Operator action |
| --- | --- |
| `pass` | Eligible for qualification + authorized promotion **only if** live gates still hold |
| non-pass | Do not promote; fix candidate or corpus; re-evaluate |
| blocking reasons present | Treat as fail-closed even if partial metrics look good |

Checked-in simulated summary
(`artifacts/agent_supervisor/semantic_compression_governor/summary.json`) may
show proposal `verdict: fail` and `production_eligible: false`. That is
**empirical documentation**, not a silent promotion path.

---

## 5. Release qualification before promotion

Promotion requires a current `ReleaseQualification` that is not blocked.

Paths (`QualificationPath`):

1. **`incremental_seal`** — released IncrementalProofSealer evidence bound to
   policy/evaluation identities.
2. **`authorized_release_qualification`** — independently authorized
   `VerificationBundle`-backed path (not an IVP commitment substituted as a
   sealer).
3. **`blocked`** — typed fail; promotion must reject.

```python
from ipfs_accelerate_py.agent_supervisor.semantic_governor.sealing import (
    qualify_policy_candidate,
)

qualification = qualify_policy_candidate(
    candidate=candidate,
    evaluation_report=report,
    # sealer capability / verification bundle / authorization as required
)
# qualification.promotes_allowed is false when path is blocked or unavailable
```

If the sealer public API is not released, status is **`unavailable`**. Do not
substitute a Merkle commitment. Content-addressed artifacts may still be stored;
proof-backed seal claims remain unavailable.

Bounded claims a seal may encode (and only these by default):

1. exact artifacts evaluated  
2. required evaluations completed  
3. declared thresholds applied  
4. no blocking status omitted  
5. promoted policy equals evaluated candidate  

---

## 6. Reproduce authorized promotion

### 6.1 Required inputs

| Input | Source |
| --- | --- |
| `candidate` | Proposed `CompressionPolicyCandidate` |
| `evaluation_report` | Held-out `RuleEvaluationReport` with `verdict=pass` |
| `authorization` | **Operator** authorization CID (not equal to candidate/eval/seal/policy CIDs) |
| `release_qualification` | Current non-blocked qualification |
| `policy_repository` | CAS-capable policy store (kit durable coordination) |
| `operation_id` | Stable idempotent operation id (1–128 chars, normalized) |
| `expected_generation` / `expected_policy_cid` | Optional; default to live head |

### 6.2 Python path

```python
from ipfs_accelerate_py.agent_supervisor.semantic_governor.promotion import (
    promote_compression_policy,
)

result = promote_compression_policy(
    candidate,
    evaluation_report,
    authorization="bafy...operator-auth...",  # CID string or mapping
    release_qualification=qualification,
    policy_repository=policy_repo,
    workspace="default",
    operation_id="promote-policy-2026-08-13-001",
    expected_generation=head.generation,
    expected_policy_cid=head.policy_cid,
    promoted_policy=promoted_policy,  # optional explicit policy body
)

# result.status in {promoted, rejected, conflict, unavailable, corrupt, unchanged}
# result.head_mutated is True only on successful publication
# result.blocking_reasons lists fail-closed codes when rejected
```

### 6.3 CLI path

```bash
semantic-governor promote-policy \
  --input promote_payload.json \
  --authorization bafy...operator-auth... \
  --store-dir /path/to/durable-coordination-store \
  --operation-id promote-policy-2026-08-13-001 \
  --workspace default \
  --expected-generation 12 \
  --expected-policy-cid bafy...current-policy...
```

Rules enforced by CLI:

- Missing `--authorization` / payload authorization → reject
  (`absent_authorization`); **no implicit promotion**.
- Missing CAS store (`--store-dir` or injected repository) → fail closed.
- Missing `--operation-id` / payload operation id → fail closed.

### 6.4 Fail-closed matrix (head must not change)

| Symptom | Reason code |
| --- | --- |
| Candidate base ≠ live policy | `stale_candidate` |
| No qualification object | `absent_release_qualification` |
| Qualification unavailable / blocked | `unavailable_release_qualification` |
| No operator auth | `absent_authorization` |
| Auth CID equals candidate/eval/seal/policy | `self_promotion_forbidden` |
| High-risk thresholds weakened | `high_risk_assurance_reduced` |
| Eval not pass / wrong candidate/partition | `evaluation_verdict_not_pass`, `mismatched_evaluation` |
| Concurrent writer / wrong generation | `cas_conflict` |
| Store down / corrupt | `cas_unavailable`, `cas_corrupt` |

Concurrent writers: at most one CAS succeeds; losers see conflict with
`head_mutated=false`.

### 6.5 Receipts to retain

After a successful promotion, archive:

- promotion result JSON (`status`, `promoted_policy_cid`, CAS transition)
- evaluation report CID
- authorization CID
- release qualification CID
- operation id and workspace
- policy generation before/after

These are **audit evidence**, not re-authorization tokens for the next promote.

---

## 7. Reproduce authorized rollback

Rollback is a **forward CAS** to a prior immutable policy CID. History is not
deleted; generation advances.

### 7.1 Python path

```python
from ipfs_accelerate_py.agent_supervisor.semantic_governor.promotion import (
    rollback_compression_policy,
)

result = rollback_compression_policy(
    authorization="bafy...operator-auth...",
    target_policy_cid="bafy...known-good-policy...",
    policy_repository=policy_repo,
    workspace="default",
    operation_id="rollback-policy-2026-08-13-001",
    expected_generation=head.generation,
    expected_policy_cid=head.policy_cid,
)

# result.status in {rolled_back, rejected, conflict, unavailable, corrupt, unchanged}
```

### 7.2 Operator checklist

1. Identify the known-good `target_policy_cid` from promotion history (never
   invent one).
2. Obtain **fresh** operator authorization (not the candidate or seal CID).
3. Read live head generation and policy CID.
4. Execute rollback CAS with a new unique `operation_id`.
5. Verify `status=rolled_back` and `head_mutated=true`.
6. Re-run smoke evaluation / dashboard-data against the restored head.
7. Document the incident and retain the rollback receipt.

Absent authorization, self-authorization, invalid target, or CAS conflict
leave the head unchanged.

---

## 8. Day-to-day audit and shadow operations

### 8.1 Recommended order

```text
audit (sufficiency)
  -> shadow (compressed vs expanded in isolated worktree)
  -> diagnose (if compressed inferior)
  -> expand (bounded steps / token growth limits)
  -> calibrate (empirical rates only)
  -> propose-rules (typed DSL)
  -> evaluate-policy (held-out)
  -> qualify + promote-policy (authorized CAS)
  -> report / dashboard-data
```

### 8.2 Shadow privacy

- Expanded private source stays **local-only** unless exact provider disclosure
  is authorized.
- Secrets redacted before any provider call.
- Expanded run is oracle/candidate only; it never silently replaces the
  accepted production patch.
- Public reports use CIDs and managed references only.

### 8.3 Calibration limits

`calibrate` updates empirical success, omission, and route frequency signals.
It **never**:

- upgrades heuristic capsules to exact
- disables full-suite fallback
- authorizes promotion
- changes trusted keys or proof systems

---

## 9. Metrics and dashboards

```bash
semantic-governor dashboard-data --input metric_sources.json
semantic-governor report --input report_bundle.json
```

Python aggregation:
`GovernorMetricsCollector` / `GovernorMetricReport` in
`ipfs_accelerate_py.agent_supervisor.semantic_governor.metrics`.

Rules operators must enforce when reading dashboards:

1. **Simulated ≠ live** — cohorts never share quality counters.
2. **Unavailable ≠ zero** — missing percentiles/costs stay missing.
3. **Net savings include audit overhead** — do not quote gross savings alone
   for production ROI.
4. Metrics never admit patches or promote policies.

Key metric families:

| Family | Examples |
| --- | --- |
| Tokens | raw / retrieval / compressed / expanded; reduction bp |
| Quality | accepted patches, regressions, proof failures, review disagreement |
| Omission | detection before/after acceptance; critical accepted count |
| Routing | route share, escalation, retries |
| Economics | model spend micros, gross/net savings, cost per accepted |
| Calibration | uses, empirical omission rate + Wilson interval, coverage |

Initial **targets** (never fabricate results to meet them): ≥95% critical
intentional-omission detection before acceptance, zero critical controlled
omissions accepted, ≥50% median final context reduction, no heuristic-as-exact,
reproducible rollback-safe promotion, audit spend below protected savings at the
configured rate.

---

## 10. Recovery and durability

Durable state lives in kit `DurableCoordinationStore` manifests: immutable
audit cases, calibration/benchmark history, policy versions, promotion state,
and receipts.

Operator recovery principles:

1. Rebuild indexes from **verified immutable blocks** only.
2. Never invent promotion, completion, or authorization during recovery.
3. Corruption or ambiguous promotion → fail closed; do not “pick the newer
   generation” without verified transition receipts.
4. Interrupted audits resume or abandon cleanly; writers never silently
   overwrite without expected-generation CAS.
5. After recovery, re-read policy head and re-run `evaluate-policy` before any
   new promote.

---

## 11. Incident response

| Incident | Immediate actions |
| --- | --- |
| Prompt injection / instruction-like content | Confirm quarantine evidence only; verify trusted config path; do not promote |
| Suspected self-promotion attempt | Inspect blocking reasons; rotate operator auth material if leaked |
| Privacy leak (raw source / secrets in report) | Halt external sends; scrub; fix projection; rotate secrets |
| Bad policy promoted | Authorized **rollback** to prior CID; retain receipts; re-evaluate |
| Sealer unavailable during release | Report typed unavailable; block proof-backed promotion claims |
| CAS conflict storm | Serialize writers; retry with refreshed expected generation |
| Live quality collapse | Raise shadow rates; stop promote; investigate omission/route metrics |

Do **not** delete history to “fix” a bad promote. Use rollback CAS.

---

## 12. Validation and tests (operator)

Documentation hygiene (board-prefix leakage check for primary agent-supervisor
docs; does not rewrite these SCG files):

```bash
python3 scripts/docs/check_agent_supervisor_docs.py
```

Representative governor unit tests (from repo root with `PYTHONPATH` set):

```bash
PYTHONPATH=ipfs_kit_py:ipfs_datasets_py:. python3 -m pytest -q \
  test/api/semantic_governor/test_policy_evaluation.py \
  test/api/semantic_governor/test_promotion.py \
  test/api/semantic_governor/test_sealing.py
```

CLI help must remain cold (no I/O side effects):

```bash
python -m ipfs_accelerate_py.agent_supervisor.semantic_governor.cli --help
python -m ipfs_accelerate_py.agent_supervisor.semantic_governor.cli promote-policy --help
```

---

## 13. Limitations operators must not re-label

1. Simulated benchmarks and harness oracle workloads are **not** live model
   quality.
2. Content-addressed storage is **not** a ZK or execution proof.
3. IVP Merkle commitments are **not** incremental sealers.
4. Empirical calibration is **not** formal exactness.
5. Seal binding after promotion is **not** semantic completeness.
6. Dashboard green status is **not** promotion authority.
7. Token reduction without quality/verification constraints is **not** success.

Maximum justified claim: the system **empirically and structurally audits**
compression, diagnoses omissions, expands with evidence, calibrates, and
promotes only after held-out evaluation plus authorized, reproducible
qualification — never that every compressed context is semantically complete.

---

## 14. Quick reference: reason codes

### Evaluation

`missing_held_out_data`, `held_out_partition_overlap`,
`candidate_generating_case_in_held_out`, `partition_not_held_out`,
`schema_or_integrity_failure`, `critical_omission_detection_regressed`,
`stale_rejection_regressed`, `hidden_accepted_regression`,
`critical_omission_accepted`, `critical_omission_detection_below_threshold`,
`accepted_regression_above_threshold`,
`median_context_reduction_below_threshold`, `high_risk_assurance_reduced`,
`full_suite_fallback_disabled`, `candidate_baseline_policy_mismatch`

### Promotion / rollback

`stale_candidate`, `absent_release_qualification`,
`unavailable_release_qualification`, `absent_authorization`,
`self_promotion_forbidden`, `high_risk_assurance_reduced`,
`mismatched_evaluation`, `evaluation_verdict_not_pass`, `cas_conflict`,
`cas_unavailable`, `cas_corrupt`, `schema_or_integrity_failure`,
`policy_head_expectation_mismatch`, `promoted_policy_cid_mismatch`,
`missing_policy_repository`, `invalid_rollback_target`,
`qualification_identity_mismatch`,
`protected_threshold_reduction_unauthorized`

### Sealing

`sealer_unavailable`, `ivp_commitment_not_sealer`,
`missing_release_qualification`,
`missing_release_qualification_authorization`,
`self_authorization_forbidden`, `evaluation_verdict_not_pass`,
`seal_overclaim_rejected`, `stale_or_tampered_seal`,
`promotion_blocked`

---

## 15. Related artifacts

| Path | Role |
| --- | --- |
| `docs/architecture/SEMANTIC_COMPRESSION_GOVERNOR_TRUST.md` | Trust/privacy/assurance model |
| `docs/architecture/SEMANTIC_COMPRESSION_GOVERNOR_PLAN.md` | Program plan |
| `artifacts/agent_supervisor/semantic_compression_governor/summary.json` | Latest simulated benchmark summary (non-production) |
| `artifacts/agent_supervisor/semantic_compression_governor/benchmark.json` | Benchmark detail |

Evidence id for this documentation pair: `scg/trust-docs@1`.
