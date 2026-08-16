# Agent Supervisor DuckDB + Quack Control-Plane Release

Program namespace: `agent-supervisor-duckdb-quack-control-plane-v1`  
Target branch: `agent/duckdb-quack-control-plane-20260808`  
Terminal task: **DQP-039** / Goal **DQP-G090**  
Interface: `DuckDBControlPlaneReleaseReceipt@1`

## Decision

This document records the joined release boundary for the DuckDB + Quack agent
supervisor control-plane migration. A release **pass** means:

1. Every prior board task DQP-000 … DQP-038 is terminal (`Status: completed`).
2. Independently supplied component evidence for schema, Quack, import/export,
   intent, runtime, worktree, AST/mutation, symbolic/proof, context/churn,
   control, watchdog, backup, chaos, canary, shadow, cutover, and rollback is
   present, current, non-synthetic, non-skipped, and bound to the exact tree,
   store generation, schema checksum, and Quack profile.
3. Safety floors remain absolute zero (duplicate non-idempotent effects, stale
   lease writes, unauthorized SQL, secret leakage, false completion, missing
   impact frontier admission, AST/mutation misbinding, event/projection
   divergence, accepted-state loss).
4. Quality / churn benchmarks do not regress sealed floors; warm reuse and
   duplicate-provider elimination hold.
5. Rollback evidence is present; rollback changes the authority route without
   deleting history or accepting legacy dual writes.
6. Legacy Markdown/JSON files are exports only — never decision authority in
   canary or default modes.

## Explicit non-claims

A release **pass records Quack experimental scope**. It does **not**:

- claim production high availability;
- claim multi-replica failover;
- claim DuckDB 2.0 or future protocol compatibility;
- authorize remote (non-loopback) Quack endpoints without a separate review;
- fabricate or refresh component evidence inside the release verifier.

## Beta / operational limitations

See `docs/guides/AGENT_SUPERVISOR_DUCKDB_QUACK_GUIDE.md` for operator
procedures (health, backup, restore, upgrade, rollback). Material limits:

- Quack is beta in the pinned DuckDB 1.5.x profile.
- One Quack server is one failure domain.
- Loopback bind is required unless separately reviewed.
- Protocol names and defaults may change before DuckDB 2.0.

## How to evaluate

```bash
python -m pytest -q test/api/test_agent_supervisor_duckdb_quack_release.py
```

Programmatic entry:

```python
from pathlib import Path
from ipfs_accelerate_py.agent_supervisor.validation.duckdb_quack_release import (
    run_hermetic_release,
)

receipt = run_hermetic_release(
    board_path=Path("docs/architecture/agent_supervisor_duckdb_quack_control_plane.todo.md"),
)
assert receipt.to_dict()["experimental_scope"] is True
assert receipt.to_dict()["production_ha_claimed"] is False
```

## Fail-closed reasons (non-exhaustive)

| Reason | Effect |
|---|---|
| Missing / stale / synthetic / skipped component root | blocked or fail |
| Safety floor non-zero | fail |
| Quality regression | fail |
| Legacy file decision read in canary | fail |
| Unauthorized SQL / stale lease / false completion | fail |
| Incomplete mutation lineage / projection divergence | fail |
| Absent rollback evidence | blocked/fail |
| Open prior board tasks | blocked |
| Required module missing on current tree | blocked |

## Related artifacts

| Artifact | Path |
|---|---|
| Plan | `docs/architecture/AGENT_SUPERVISOR_DUCKDB_QUACK_CONTROL_PLANE_PLAN.md` |
| Board | `docs/architecture/agent_supervisor_duckdb_quack_control_plane.todo.md` |
| Operator guide | `docs/guides/AGENT_SUPERVISOR_DUCKDB_QUACK_GUIDE.md` |
| Release module | `ipfs_accelerate_py/agent_supervisor/validation/duckdb_quack_release.py` |
| Ops CLI | `scripts/ops/agent_supervisor/duckdb_quack_control_plane.py` |
