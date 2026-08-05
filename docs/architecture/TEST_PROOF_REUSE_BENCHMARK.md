# Proof-Backed Test Reuse: Shadow and Warm Benchmark

Status: normative measurement contract  
Interfaces: `ProofReuseBenchmark@1`, `BenchmarkReceipt@1`, `ProofReuseMetrics@1`  
Evidence: `ptr/shadow-benchmark@1`  
Module: `ipfs_accelerate_py.agent_supervisor.self_improvement.proof_reuse_benchmark`

## Purpose

Quantify whether proof-backed pytest reuse is safe and cheap enough to
promote beyond mode `off`. The benchmark freezes a controlled eligible and
ineligible fixture population, compares decisions and deterministic cost units
across operational scenarios, and emits a compact, privacy-safe receipt.

Performance never relaxes authority. A false authoritative skip is a hard
gate failure regardless of measured savings.

## Scenarios

| Scenario | Mode | Cache state | Authoritative skip? | What is measured |
| --- | --- | --- | --- | --- |
| `off` | `off` | ignored | no | Full execution baseline |
| `shadow` | `shadow` | warm candidates | no (predict only) | Predicted hits vs forced execution |
| `cold_readwrite` | `readwrite` | empty | no | Miss overhead and write intent |
| `warm_read` | `read` | warm candidates | yes, after verify | Verified skips and saved wall time |
| `forced_rerun` | `read` | warm candidates | no (forced run) | Predicted vs actual execution |

## Controlled population

The default corpus includes:

- **eligible warm** fixtures with exact current execution keys, trusted pass
  receipts, and locally verified real certificates (ground truth:
  `should_skip`);
- **ineligible** fixtures denied before store lookup;
- **miss** fixtures with no candidate;
- **mutated** fixtures whose candidate binds a stale execution key;
- **excluded** fixtures reported in the exclusion map (effects, disabled
  reuse, unsupported surfaces) and never counted toward the warm hit rate.

Only the explicitly eligible warm population contributes to the 80% warm
skip threshold. Exclusions are recorded by reason code and must round-trip
identically in the receipt.

## Deterministic cost model

Receipt latency fields use virtual milliseconds rather than host wall-clock:

| Cost unit | Default | Role |
| --- | --- | --- |
| `execute_cost_ms` | 50 | Full setup/call/teardown stand-in |
| `verify_cost_ms` | 2 | Local certificate verification |
| `miss_lookup_cost_ms` | 1 | Empty-candidate lookup |
| `collection_cost_ms` | 1 | Per-item collection bookkeeping |

This keeps **saved wall time** and **exclusions** bit-reproducible across
hosts while still driving the real `ProofReuseLookup` / `TestProofCache`
admission path for decision correctness.

## Acceptance gates

All gates must pass for `ProofReuseBenchmarkReceipt.passed`:

1. **`false_admissions_zero`** — authoritative skips where ground truth is
   `should_run` equal exactly zero across every scenario.
2. **`warm_skip_threshold`** — at least 80% (`MIN_WARM_SKIP_BPS = 8000`) of
   the explicitly eligible unchanged warm population both verifies and skips
   under `warm_read`.
3. **`verify_cheaper_than_execute`** — aggregate warm verification cost is
   strictly less than the eligible population execution cost.
4. **`miss_overhead_bounded`** — cold/miss collection+lookup cost stays
   within `MAX_MISS_OVERHEAD_BPS` (20%) of a single execution unit per item.
5. **`receipt_reproducible`** — a second independent run yields an identical
   receipt body and content id.

## Metrics surface

Scenario summaries embed `ProofReuseMetrics@1` snapshots:

- outcome counts: predicted, verified, skipped, executed, deferred, degraded;
- reason-code histograms (safe vocabulary only);
- verify and execution latency totals;
- bytes read / written.

Telemetry never includes node ids, paths, parameter values, receipt bodies,
stdout/stderr, or private witness material.

## Receipt contract

`ProofReuseBenchmarkReceipt` (`BenchmarkReceipt@1`) is JSON-canonical and
content-addressed.  Material fields:

- `false_admissions`, `warm_eligible_count`, `warm_verified_skips`,
  `warm_skip_bps`
- `verify_latency_ms`, `execution_latency_ms`, `miss_overhead_ms`,
  `max_miss_overhead_ms`, `saved_wall_time_ms`
- `exclusions` (reason → count)
- `scenario_summaries` (one per required scenario)
- `gates` (name, passed, detail)
- `passed`

Schema:
`ipfs_accelerate_py/agent-supervisor/proof-reuse-benchmark-receipt@1`

## Operator usage

```python
from ipfs_accelerate_py.agent_supervisor.self_improvement.proof_reuse_benchmark import (
    run_proof_reuse_benchmark,
    verify_benchmark_receipt,
)

receipt = run_proof_reuse_benchmark()
assert receipt.passed
assert verify_benchmark_receipt(receipt)
```

Validation of this module itself must keep proof reuse off so the feature
never depends on its own cached certificates:

```bash
IPFS_TEST_PROOF_REUSE_MODE=off python3 -m pytest \
  external/ipfs_accelerate/test/api/test_agent_supervisor_proof_reuse_benchmark.py -q
```

## Relation to rollout

PTR-101 consumes this receipt as a promotion precondition.  Rollout may narrow
or disable reuse, sample forced reruns, and automatically return to
`shadow`/`off` on any false admission.  It must not broaden proof authority
beyond reviewed eligibility merely because the benchmark reports savings.

## Non-goals

- Live host wall-clock SLOs or CI flakiness budgets
- Granting completion, merge, or skip authority from the receipt alone
- Exporting private test output or path telemetry
- Replacing mutation, degradation, security, or cross-repository assurance
  populations (PTR-090–PTR-093 remain mandatory zero-false-skip evidence)
