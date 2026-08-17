# Incremental Proof Sealer Benchmark Analysis

Canonical schema: `incremental-proof-sealer-benchmark-summary@1`.
Raw artifact digest: `sha256:f33fbdb928f523e11e75d22cc2808594bcb1e2fb83b9fcf7837d49608afa2de6`.

This document interprets `artifacts/agent_supervisor/incremental_proof_sealer/benchmark.json`.
It does not replace the protected materialization. **Targets are not facts.**

## Provenance

Every transition row in the raw artifact is labeled `mixed` because GPU time is
`unavailable` while CPU, size, and storage fields are `estimated`. No field is
`measured`. Estimates are never sold as measurements.

- measurement_counts: {"estimated": 0, "measured": 0, "mixed": 40}
- average reuse rate: 43.6161% (mixed)
- average compute reduction: 43.0539% (mixed)

Receipt aggregation does not prove test execution. Simulated required units
cannot satisfy a production seal.

## Target assessments

| Goal | Field | Target | Actual | Met | Provenance |
|---|---|---|---|---|---|
| localized_reuse_70_percent | cache_hit_rate | 70 | 43.0159 | False | mixed |
| mixed_compute_reduction_50_percent | compute_saved_percent | 50 | 43.0539 | False | mixed |
| documentation_compute_reduction_80_percent | compute_saved_percent | 80 | 98.7124 | True | mixed |

Localized 70% reuse and mixed-history 50% compute reduction are unmet.
Documentation 80% compute reduction is met.
Unmet goals are reported honestly.

## Best / worst / fallback

- Best incremental case: transition 34 `ordinary documentation edit` saved 98.9011% (mixed).
- Worst case: transition 0 `initial repository` saved 0.0000% (mixed).
- Forced-full / fallback_transition_indices: [0, 12, 17, 20, 22, 24, 27, 29, 30, 32, 35, 39]

## Metric availability

- `proof_size_bytes`: available=40 measured=0 estimated=40 unavailable=0 min=0.0 max=36864.0 mean=18022.4
- `seal_size_bytes`: available=40 measured=0 estimated=40 unavailable=0 min=3584.0 max=4608.0 mean=3980.8
- `seal_verification_seconds`: available=40 measured=0 estimated=40 unavailable=0 min=0.0 max=0.0026999999999999997 mean=0.0013199999999999998
- `storage_growth_bytes`: available=40 measured=0 estimated=40 unavailable=0 min=1792.0 max=36864.0 mean=18886.4
- `prover_cpu_seconds`: available=40 measured=0 estimated=40 unavailable=0 min=0.1 max=9.1 mean=4.5
- `prover_gpu_seconds`: available=0 measured=0 estimated=0 unavailable=40 min=None max=None mean=None
- `peak_memory_bytes`: available=40 measured=0 estimated=40 unavailable=0 min=1792.0 max=36864.0 mean=18886.4

Named metrics required by acceptance: `proof_size_bytes`, `seal_verification_seconds`,
`storage_growth_bytes`, `prover_cpu_seconds`, `prover_gpu_seconds`, `peak_memory_bytes`.

## Limitations

- All cost and timing fields in this run are estimated planner resource costs or unavailable; none are measured wall-clock observations from a production prover.
- GPU time is unavailable because no GPU prover was present.
- Targets are not facts: unmet 70/50/80-percent goals are reported without inflation.
- Receipt aggregation does not prove test execution.
- Simulated required units cannot satisfy a production seal.
- This summary binds the raw benchmark SHA-256 and recomputes arithmetic only; it does not upgrade estimated values into measurements.
