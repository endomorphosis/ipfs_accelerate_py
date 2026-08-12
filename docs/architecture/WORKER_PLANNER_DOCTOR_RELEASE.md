# Worker Planner–Doctor Release Gate (WPD-070)

Interface: `WorkerPlannerDoctorRelease@1`  
Evidence: `wpd/release@1`  
Board: `agent-supervisor-worker-planner-doctor-v1`

## Purpose

Terminal current-tree gate for promoting **kernel-first** agent-supervisor
defaults. The release receipt is content-addressed, replayable, and never
grants mutation or process authority by itself.

## Required current-tree surfaces

| Surface | Interface |
| --- | --- |
| Pre-implementation kernel | `PreImplementationKernel@1` |
| Provider gate | `ImplementationDaemon@pre_implementation_kernel` |
| Analytical close | `AnalyticalCloseExecutor@1` |
| Residual provider | residual packet sealed path |
| Failure replan | `FailureReplanPolicy@1` |
| LLM-avoidance metrics | `LlmAvoidanceMetrics@1` |
| Paired benchmark | `WorkerPlannerDoctorBenchmark@1` |

## Fail-closed rules

1. **Safety floors must be zero** (unauthorized provider, scope escape, free re-prompt).
2. **Required modules must import** on the current tree.
3. **Benchmark** must show challenger provider-call reduction with quality non-inferiority.
4. **Synthetic-only** runs **cannot promote** (`promotion_allowed=false`).

## Operator promotion

Promotion requires a non-synthetic operator seal after the current-tree receipt
verdict is `pass`. Replay the same inputs for identity-equivalent receipts:

```bash
PYTHONPATH=external/ipfs_accelerate python3 - <<'PY'
from ipfs_accelerate_py.agent_supervisor.validation.worker_planner_doctor_release import (
    evaluate_release,
)
print(evaluate_release(synthetic_only=True).to_dict())
PY
```

## Related

- Plan: monorepo `implementation_plan/docs/47-supervisor-worker-planner-doctor-integration-plan-2026-08-06.md`
- Benchmark config: monorepo `config/supervisor_worker_planner_doctor_benchmark.json`
