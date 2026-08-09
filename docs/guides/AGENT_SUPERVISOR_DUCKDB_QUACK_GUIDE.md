# Agent Supervisor DuckDB / Quack Control Plane — Operator Guide

This guide is the **operations runbook** for the DuckDB/Quack control-plane
program (`DQP-` / board namespace
`agent-supervisor-duckdb-quack-control-plane-v1`).

It covers staged canary, default cutover, rollback, kill switch, and the
**exact** health / backup / restore / upgrade procedures. Normative
architecture lives in the protected plan and objective heap; this document is
the human operator surface for **DQP-038**
(`DatabaseRolloutPolicy@1` / `DatabaseCutoverReceipt@1`).

Related code:

| Surface | Path |
| --- | --- |
| Rollout policy | `ipfs_accelerate_py/agent_supervisor/self_improvement/database_rollout.py` |
| Ops facade | `scripts/ops/agent_supervisor/duckdb_quack_control_plane.py` |
| State-owner | `scripts/ops/agent_supervisor/quack_state_server.py` |
| Doctor | `scripts/ops/agent_supervisor/duckdb_quack_doctor.py` |
| Backup/restore | `ipfs_accelerate_py/agent_supervisor/runtime/control_plane_backup.py` |
| Shadow parity | `ipfs_accelerate_py/agent_supervisor/self_improvement/database_shadow_rollout.py` |
| Canary harness | `ipfs_accelerate_py/agent_supervisor/validation/duckdb_quack_canary.py` |

## Trust boundary

**Protected anchors are read-only to every automatic path.**

Operators and the launch surface may read, but never create, modify, rename,
delete, replace, or regenerate:

| Anchor class | Examples |
| --- | --- |
| Seed program | plan, objectives heap, todo board, scheduler config |
| Board validation | `scripts/validate_agent_supervisor_duckdb_quack_control_plane_board.py` |
| Board tests | `test/api/test_agent_supervisor_duckdb_quack_control_plane_board.py` |

Lifecycle state, worktrees, DuckDB files, backups, logs, and derived exports
live under **isolated** paths. Exports always carry the non-authority marker
and are never watched as input unless an explicit later import is requested.

**Discovery nominates; independent checks admit.** Canary scores, shadow
parity, and provider telemetry never authorize default cutover without a
current release gate.

## Beta, single-failure-domain, and loopback limitations

Quack is **beta** in the pinned DuckDB 1.5.x profile. The control plane
records these limitations explicitly (see
`DEFAULT_QUACK_BETA_LIMITATIONS`):

| Limitation | Operator impact |
| --- | --- |
| `quack_is_beta_in_duckdb_1_5_x` | Treat as beta; do not claim GA HA |
| `protocol_names_and_defaults_may_change_before_duckdb_2_0` | Pin builds; re-gate after upgrades |
| `server_and_clients_must_use_identical_pinned_build` | Mismatched fingerprints fail closed |
| `default_authorization_callback_permits_every_authenticated_query` | Constrain network; use secret handles |
| `no_server_push_clients_must_poll` | Clients poll; design stall detectors accordingly |
| **`one_quack_server_is_one_failure_domain`** | One supervised server + restore is **resilient, not highly available** |
| **`loopback_bind_required_unless_separately_reviewed`** | Default bind is `127.0.0.1` / `::1` only |
| `unsigned_or_community_extension_path_is_not_attested_integrity` | Refuse unattested extension installs |

**Single failure domain.** A single Quack state-owner is one failure domain.
Watchdog, backup, and restore make the system recoverable; they do **not**
provide multi-primary high availability. Document this to stakeholders before
default cutover.

**Loopback by default.** Non-loopback binds require a separately reviewed
remote bind policy (`RemoteBindPolicy`) with a non-empty review receipt. The
rollout release gate denies promotion when a remote bind is requested without
policy admission (`remote_prohibition`).

**Beta waiver.** Promotion past canary requires a recorded beta-limitation
waiver (`beta_waiver`). Operators acknowledge the limitations above rather
than suppressing them.

## Rollout ladder

Authority advances through a closed stage ladder. Promotion is **one stage at
a time**. Defaults switch only after canary.

```text
off -> observe -> shadow -> assist -> canary -> default
```

| Stage | Authority mode | Dual observation | Effect |
| --- | --- | --- | --- |
| `off` | `embedded_maintenance` | no | No Quack scheduling authority |
| `observe` | `embedded_maintenance` | no | Readiness observation only |
| `shadow` | `quack_shadow` | **temporary** | Shadow writes never control production |
| `assist` | `quack_shadow` | **temporary** | Suggestions only; files remain non-authority |
| `canary` | `quack_authoritative` | no | One isolated program is database-authoritative |
| `default` | `quack_authoritative` | no | **New local programs** default to Quack |

Dual write is **temporary evidence collection** (shadow/assist), not a
permanent two-authority architecture. Rollback never re-accepts legacy dual
writes.

### New programs default to Quack only under a valid release gate

`DatabaseRollout.new_programs_default_to_quack()` returns true **only** when
all of the following hold:

1. Current stage is `default`.
2. Policy admits `default` in `allowed_stages`.
3. Canary completed successfully on the exact tree/schema/profile.
4. Kill switch is clear.
5. Policy `new_program_default_stage` is `default`.

Otherwise new programs register at `off` (fail closed).

## Release gate

Promotion into `canary` or `default` joins independent evidence:

| Evidence kind | Source |
| --- | --- |
| `chaos_security` | DQP-034 security/concurrency/restart |
| `canary_e2e` | DQP-035 multi-daemon multi-worktree canary |
| `churn_quality` | DQP-036 quality/safety/throughput benchmark |
| `shadow_parity` | DQP-037 shadow decision parity |
| `backup_restore` | DQP-033 checkpoint/backup/restore |

Hard floors (non-compensable):

- kill switch engaged → deny
- server unavailable → deny (shadow+)
- backup age exceeds 30 days → deny (canary+)
- partial multi-program rollout → deny
- stale / synthetic / skipped / missing evidence → deny
- remote bind without reviewed policy → deny
- missing beta waiver → deny
- permanent legacy dual-write acceptance → deny
- history deletion → deny
- `default` without completed canary → deny

## Operator entry

```bash
# Machine-readable recipe (no process start)
python scripts/ops/agent_supervisor/duckdb_quack_control_plane.py recipe

# Limitations (beta / single-failure-domain / loopback)
python scripts/ops/agent_supervisor/duckdb_quack_control_plane.py limitations

# Closed stage map
python scripts/ops/agent_supervisor/duckdb_quack_control_plane.py stages

# Status
python scripts/ops/agent_supervisor/duckdb_quack_control_plane.py status --json

# Promote one stage (assist and below need no full evidence map)
python scripts/ops/agent_supervisor/duckdb_quack_control_plane.py \
  --target-stage observe promote --json

# Promote canary/default with evidence + allow-default
python scripts/ops/agent_supervisor/duckdb_quack_control_plane.py \
  --allow-default \
  --evidence-json /path/to/evidence.json \
  --target-stage canary \
  promote --json

# Rollback route only (history preserved; dual writes refused)
python scripts/ops/agent_supervisor/duckdb_quack_control_plane.py \
  --target-stage assist rollback --json

# Kill switch (forces off; preserves history)
python scripts/ops/agent_supervisor/duckdb_quack_control_plane.py kill-switch --json
python scripts/ops/agent_supervisor/duckdb_quack_control_plane.py kill-switch-clear --json
```

Interface: **`DatabaseRolloutPolicy@1`**. Receipts: **`DatabaseCutoverReceipt@1`**.

Commands: `status`, `stages`, `promote`, `rollback`, `kill-switch`,
`kill-switch-clear`, `default-program`, `release-gate`, `health`, `backup`,
`restore`, `upgrade`, `recipe`, `limitations`.

Never pass raw auth tokens on argv. Use secret handles only.

## Exact health procedure

```bash
python scripts/ops/agent_supervisor/duckdb_quack_control_plane.py health
```

Steps (in order):

1. Confirm state-owner process birth identity via
   `quack_state_server status` / `ready`.
2. Query database readiness (store generation, schema fingerprint, server id).
3. Run `duckdb_quack_doctor diagnose`; **abstain** when ownership is unknown.
4. Verify loopback bind (`127.0.0.1` / `::1`) unless a reviewed remote policy
   exists.
5. Confirm backup age is within `DEFAULT_MAX_BACKUP_AGE_SECONDS` (30 days).
6. Refuse file-age PID signalling; use fenced reclaim only.

Representative commands:

```bash
python scripts/ops/agent_supervisor/quack_state_server.py \
  --database /path/to/control.duckdb status
python scripts/ops/agent_supervisor/quack_state_server.py \
  --database /path/to/control.duckdb ready
python scripts/ops/agent_supervisor/duckdb_quack_doctor.py \
  diagnose --database /path/to/control.duckdb
```

## Exact backup procedure

```bash
python scripts/ops/agent_supervisor/duckdb_quack_control_plane.py backup
```

Steps (in order):

1. Acquire an exclusive maintenance lease **or** stop the state-owner cleanly.
2. Refuse direct-file copy while ownership is **live** or **unknown**.
3. Invoke `ControlPlaneBackup@1` to create a verified consistent snapshot.
4. Independently verify digest, schema version, and authority roots.
5. Update the retention manifest; prune only verified excess snapshots.
6. Record backup age for the release gate.

Module: `ipfs_accelerate_py.agent_supervisor.runtime.control_plane_backup`
(`ControlPlaneBackup@1`). Maximum backup age for promotion: **30 days**.

## Exact restore procedure

```bash
python scripts/ops/agent_supervisor/duckdb_quack_control_plane.py restore
```

Steps (in order):

1. Stop or fence the live state-owner; never restore under unknown ownership.
2. Select a verified `backup.manifest.json` with matching schema fingerprint.
3. Invoke `RestoreReceipt@1` **rehearsal** first when validating a new profile.
4. Apply restore; rotate store generation so pre-rotation writers fail closed.
5. Invalidate stale clients/leases; require re-attach with the new generation.
6. Run the health procedure; confirm event/task/lease roots match the snapshot.

Module: `ipfs_accelerate_py.agent_supervisor.runtime.control_plane_backup`
(`RestoreReceipt@1` / `StoreGenerationRotation@1`).

## Exact upgrade procedure

```bash
python scripts/ops/agent_supervisor/duckdb_quack_control_plane.py upgrade
```

Steps (in order):

1. Pin **identical** DuckDB and Quack extension fingerprints on server and
   clients.
2. Backup and restore-rehearse on the current pinned **1.5.x** profile first.
3. Apply schema migrations under exclusive maintenance; prove
   fresh-database ≡ upgraded-database equivalence.
4. Re-run chaos, canary, shadow, and churn gates on the **exact**
   tree / schema / profile.
5. Promote **one stage at a time** via the control-plane facade; never jump.
6. Only after canary + valid release gate, cut over `default` for new programs.

Pinned profile token: `duckdb-1.5.x-quack-pinned`. A future DuckDB/Quack
profile requires a separately tested restore rehearsal and a full release
gate; do not promote on migration dry-runs alone.

## Rollback

Rollback **switches the authority/read route only**:

- History (events, receipts, shadow digests) is **never deleted**.
- Legacy dual writes are **never accepted** as permanent authority.
- Kill switch forces `off` while preserving history.
- Attempting `--delete-history` or `--accept-legacy-dual-writes` is recorded
  as a refused reason code and does not mutate durable state.

```bash
# One stage back
python scripts/ops/agent_supervisor/duckdb_quack_control_plane.py rollback --json

# Exact prior stage
python scripts/ops/agent_supervisor/duckdb_quack_control_plane.py \
  --target-stage shadow rollback --json
```

## Validation

```bash
python -m pytest -q test/api/test_agent_supervisor_database_rollout.py
```

Acceptance (DQP-038):

- New local programs default to Quack **only** under a valid release gate.
- Rollback switches route without deleting history or accepting legacy dual
  writes.
- This guide accurately states beta / single-failure-domain / loopback
  limitations and the exact health / backup / restore / upgrade procedures.
