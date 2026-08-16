# Agent Supervisor DuckDB + Quack Operator Guide

Program: `agent-supervisor-duckdb-quack-control-plane-v1`  
Task: DQP-038  
Audience: operators running local staged cutover of the DuckDB/Quack control plane.

## Scope and limitations (read first)

This control plane is **experimental / beta**.

| Limitation | Meaning for operators |
|---|---|
| **Beta** | Quack and the control-plane schema are pinned to a reviewed 1.5.x profile. Do not claim production HA. |
| **Single-failure-domain** | One state-owner process owns the database. Loss of that process is a restore/restart event, not a multi-replica failover. |
| **Loopback** | The Quack state-owner binds to loopback only in the default profile. Remote endpoints are prohibited unless an explicit, reviewed policy waiver is recorded. |
| **No dual authority** | After cutover, Markdown/JSON taskboards and exports are **non-authoritative**. Runtime decisions never read export files as input. |
| **No legacy dual writes** | Rollback switches the authority **route**; it does not re-enable dual writes into legacy files as a second authority. |

## Stages

```text
off → observe → shadow → assist → canary → default
                                    ↘ rollback
```

| Stage | Authority mode | Production effect |
|---|---|---|
| `off` | `legacy_import` | Legacy only |
| `observe` | `embedded_maintenance` | DB observed, not scheduling |
| `shadow` | `quack_shadow` | Shadow decisions, no production effect |
| `assist` | `quack_shadow` | Assisted recommendations still non-authoritative |
| `canary` | `quack_authoritative` | Isolated program uses DB authority |
| `default` | `quack_authoritative` | New local programs default to Quack |
| `rollback` | `embedded_maintenance` | Kill-switch route change; history preserved |

Default cutover is **serialized** and **evidence-gated**. The controller will deny promotion when chaos, canary, churn/quality, shadow, backup, schema, or Quack profile evidence is missing, stale, synthetic, skipped, or bound to a different tree/schema/profile.

## Health

1. Confirm the state-owner is alive and the readiness query succeeds.
2. Confirm store generation, schema checksum, and extension fingerprint match the pinned profile.
3. Run the doctor/watchdog diagnostics (DQP-032) rather than deleting locks or signalling raw PIDs by age.

```bash
python scripts/ops/agent_supervisor/duckdb_quack_control_plane.py status
python scripts/ops/agent_supervisor/duckdb_quack_control_plane.py stages
```

## Backup

- Take a consistent backup **before** canary and again **before** default.
- Record backup age in the evidence bundle. Promotion to `default` fails when backup age exceeds the policy ceiling (default 24h).
- Backups bind store UUID, generation, and schema revision.

## Restore

1. Stop workers that would write under the old generation.
2. Restore the reviewed backup into a new working path (never mutate the only copy).
3. Start the state-owner against the restored path; verify generation and readiness.
4. Resume from `rollback` or `observe` as appropriate; do **not** fabricate completions.

## Upgrade

- Upgrades are rehearsal-gated: pin the source profile (for example DuckDB/Quack 1.5.x), restore a backup into a disposable environment, apply migrations, and re-run chaos + canary + shadow parity before production default.
- Future DuckDB 2.0 compatibility is **not** claimed by this program.

## Rollback

```bash
python scripts/ops/agent_supervisor/duckdb_quack_control_plane.py rollback --from-stage canary
```

Rollback:

- changes the authority/read route to the last proved mode;
- **preserves** database history;
- **does not** accept legacy dual writes;
- records a cutover receipt with `verdict=rolled_back`.

## Promote (evidence-gated)

Hermetic / operator lab path (uses a built-in passing evidence bundle when `--json-evidence` is omitted):

```bash
python scripts/ops/agent_supervisor/duckdb_quack_control_plane.py promote \
  --from-stage off --to default --walk
```

For real cutovers, pass a JSON evidence bundle that includes current chaos, canary, churn/quality, shadow, backup, schema, and Quack profile roots bound to the exact tree and store generation.

## Legacy export

Exports (Markdown/JSON/JSONL/CSV/Parquet) remain available for humans and portable bundles. They are marked non-authoritative. Tampering with an export must not change database authority.

## Kill switch

Policy may set `kill_switch_engaged=true`. While engaged, only `rollback` transitions are accepted. Operators re-enable promotion only after reviewing live evidence.

## Related artifacts

- Plan: `docs/architecture/AGENT_SUPERVISOR_DUCKDB_QUACK_CONTROL_PLANE_PLAN.md`
- Policy module: `ipfs_accelerate_py/agent_supervisor/self_improvement/database_rollout.py`
- Ops CLI: `scripts/ops/agent_supervisor/duckdb_quack_control_plane.py`
- Board: `docs/architecture/agent_supervisor_duckdb_quack_control_plane.todo.md`
