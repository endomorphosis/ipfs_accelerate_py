# Agent Supervisor Codebase Proof Evaluation (CBP-130)

## Purpose

Closed-loop quality, coverage, token, and proof-cost gates for the
codebase-proof (CBP) stack. The benchmark preregisters a fixed baseline and a
held-out mutation/repair suite **before** outcome inspection, compares
**bulk-source** and **obligation-first** context paths on identical tasks, and
evaluates deterministic fixture gates separately from any live-model channel.

| Item | Value |
| --- | --- |
| Objective | `CBP-G130` |
| Task | `CBP-130` |
| Evidence term | `CBPEV130MET` / `CODE_PROOF_EFFICIENCY_EVIDENCE_ID` |
| Interface | `CodebaseProofBenchmark@1` |
| Corpus version | `codebase-proof-efficiency@1` |
| Validation | `python -m pytest test/api/test_agent_supervisor_code_proof_efficiency.py -q` |

## Implementation map

| Artifact | Role |
| --- | --- |
| `ipfs_accelerate_py/agent_supervisor/supervisor_code_proof_benchmark.py` | Preregistered suite, mutation seeds, claim/coverage/quality report, fixture gates |
| `ipfs_accelerate_py/agent_supervisor/supervisor_efficiency_metrics.py` | Extended with `CodeProofEfficiencyCase` / `CodeProofEfficiencyReport`, receipt projection, ≥40%/≥60% thresholds |
| `test/api/test_agent_supervisor_code_proof_efficiency.py` | Deterministic fixture gates and fail-closed regressions |

Conflict policy: **extend** efficiency metrics; store digests and counters only
— never prompt bodies, source bodies, decoded model output, or nested artifact
graphs.

## Preregistered population

### Claim families (required)

Every suite must include at least one paired task for each family:

1. `dependency_reachability`
2. `api_contract`
3. `behavioral_invariant`
4. `security_property`
5. `semantic_equivalence`
6. `supervisor_lifecycle`

### Held-out mutation / repair seeds

Seeds are fixed with the suite identity (content-addressed
`preregistration_digest`) so post-hoc substitution fails closed:

| Seed kind | What it measures |
| --- | --- |
| `false_admit` | Seeded false authoritative admissions |
| `false_refute` | Seeded false refutations |
| `stale_evidence` | Stale-evidence detection |
| `first_pass_repair` | First-pass success |
| `eventual_repair` | Eventual repair success |
| `accepted_patch_regression` | Accepted-patch regression rate |
| `warm_cache_dominated` | Warm prove-cost improvement when cache hits dominate |
| `required_coverage` | Required-coverage preservation |

### Context paths

| Path | Description |
| --- | --- |
| `bulk_source` | Baseline that ships large source context |
| `obligation_first` | Candidate that ships obligation-first capsules / deltas |

Both arms of a paired case share the same `task_reference`, claim family, and
frozen repository/policy binding.

## Metrics reported

### Claim coverage

- By **family** (required / satisfied / refuted / open / unsupported /
  not_measured / stale / unknown)
- By **evidence tier** (`query_fact` … `cryptographic_attestation`)
- By **required assurance** (`unverified` … `attested`)
- Aggregate lifecycle-state counts for every `ClaimStatus`

### Quality

- False-admit count and rate (bps)
- False-refute count and rate (bps)
- Stale-evidence detected vs expected
- First-pass success count / rate
- Eventual repair success count / rate
- Accepted-patch regression count / rate
- Mutation seed match count / total

### Efficiency (bulk vs obligation-first)

- Input tokens and **input tokens per accepted criterion**
- Retry tokens
- Provider calls
- Cache hit / reject counts and rates
- Wall time (ms)
- Proof cost (microunits)
- Warm prove-cost reduction when `warm_cache_dominated` cases are present

Ratios are exact integer basis points (`BASIS_POINTS = 10_000`).

## Fixture gates (authoritative)

Deterministic fixture channel only. Live-model observations never contribute to
pass/fail.

| Gate | Requirement |
| --- | --- |
| Zero false authoritative admissions | `false_admit_count == 0` |
| No required-coverage loss | Every required claim satisfied on bulk remains satisfied on obligation-first |
| Input token reduction | ≥ **40%** fewer input tokens per accepted criterion (`4000` bps) |
| Retry token reduction | ≥ **60%** fewer retry tokens (`6000` bps) |
| Warm prove-cost improvement | Strict improvement on warm-cache-dominated cases |
| Required family coverage | All six families present |
| Fixture channel isolation | Live-model channel is non-authoritative |

When all gates pass, the report may claim
`CODE_PROOF_EFFICIENCY_EVIDENCE_ID`.

## Live-model channel (non-authoritative)

Optional `LiveModelChannelObservation` may attach paired live observations for
reporting. The channel is labeled
`authoritative_for_fixture_gates: false`. Fixture gate evaluation ignores live
arms entirely; live summaries appear only under `live_model_summary` on the
report.

## Efficiency metrics extension

`supervisor_efficiency_metrics` adds:

- `CODE_PROOF_MIN_INPUT_TOKEN_REDUCTION_BPS = 4000`
- `CODE_PROOF_MIN_RETRY_TOKEN_REDUCTION_BPS = 6000`
- `CodeProofEfficiencyCase` — bulk vs obligation counters including retry and
  proof cost
- `CodeProofEfficiencyReport` — aggregate gates and evidence claim emission
- `build_code_proof_paired_receipts` — project benchmark arms into typed
  `EfficiencyReceipt` populations (digests only)
- `build_code_proof_efficiency_report` — from paired cases or receipt triples

The CBP-130 producer registry maps (separate from the ASI accepted-work map):

```text
CODE_PROOF_EFFICIENCY_EVIDENCE_PRODUCERS[
  CODE_PROOF_EFFICIENCY_EVIDENCE_ID
] → supervisor_efficiency_metrics.build_code_proof_efficiency_report
```

## How to run

```bash
python -m pytest test/api/test_agent_supervisor_code_proof_efficiency.py -q
```

Programmatic entry points:

```python
from ipfs_accelerate_py.agent_supervisor.supervisor_code_proof_benchmark import (
    build_preregistered_codebase_proof_suite,
    evaluate_codebase_proof_benchmark,
    run_codebase_proof_efficiency_gates,
    verify_codebase_proof_benchmark_report,
)

report = run_codebase_proof_efficiency_gates()
assert report.passed

suite = build_preregistered_codebase_proof_suite(include_live_model_channel=True)
report = evaluate_codebase_proof_benchmark(suite)
assert verify_codebase_proof_benchmark_report(report, suite)
assert report.live_model_summary["authoritative_for_fixture_gates"] is False
```

## Fail-closed behavior

- Preregistration digest mismatch rejects suite decode
- Missing required claim families reject suite construction
- Report identity is recomputed from additive counts; forged totals fail
  verification
- False-admit, coverage-loss, or insufficient token/retry reduction clear
  evidence claim references
- Forbidden payload keys (`prompt`, `source_body`, `decoded_output`, `patch`,
  proof/counterexample bodies, nested artifact graphs) are rejected on suite
  load

## Dependencies

Depends on prior CBP tracks:

- CBP-015 proof cache integration
- CBP-050 cache-aware re-proof
- CBP-060 obligation-first context capsules
- CBP-070 delta retry context
- CBP-080 CodeEditPacket materialization

This evaluation measures whether those paths improve claim reliability and
reduce context / prove cost without lowering required coverage or admitting
false authoritative results.
