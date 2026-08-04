# Proof-Directed Planner and Doctor — Operator Guide

This guide is the **operations runbook** for the proof-directed Planner and
Doctor program (`PDR-` / board namespace
`agent-supervisor-proof-directed-planner-doctor-v1`).

It covers protected launch profiles, lifecycle controls, the kill switch,
capability degradation, stale state, rollback and quarantine, held-out
evaluation, and recovery **without editing protected anchors**.

Normative architecture lives in the protected plan, objective heap, seed
taskboard, and scheduler config. This document is the human operator surface
for **PDR-091** (`PlannerDoctorOperations@1`).

Related:

- [Deterministic Doctor Guide](DETERMINISTIC_DOCTOR_GUIDE.md) (LPR rollout)
- [Proof-Gated Contract Repair Guide](PROOF_GATED_CONTRACT_REPAIR_GUIDE.md)
- [Tactician-Hammer Logic Repair Guide](TACTICIAN_HAMMER_LOGIC_REPAIR_GUIDE.md)

## Trust boundary

**Protected anchors are read-only to every automatic path.**

Operators and the launch surface may read, but never create, modify, rename,
delete, replace, or regenerate:

| Anchor class | Examples |
| --- | --- |
| Seed program | plan, objectives heap, todo board, scheduler config |
| Authority | threat model, authority policy + seal, authority tests |
| Benchmark / holdout | benchmark policy + seal, architecture doc, holdout manifest, contract tests |

Lifecycle state, worktrees, merge-queue entries, logs, and derived refill task
sources live under **isolated** paths (default under
`data/agent_supervisor/proof_directed_planner_doctor/live/` or an explicit
`--state-dir`). They must not collide with protected anchors.

**Discovery nominates; independent checks admit.** Capability inventory,
telemetry, and model output never authorize mutation, promotion, or
completion.

## Defaults (fail-closed)

Seed launch profiles are **report-only / shadow**:

| Surface | Seed default | Notes |
| --- | --- | --- |
| Doctor mode | `report_only` | No write-path mutation |
| Planner mode | `shadow` | Plan without control-plane mutation |
| Rollout mode | `shadow` | Ladder: `off → observe → shadow → assist → canary → automatic` |
| `automatic` | **off** | Requires prerequisite receipt, elevated profile, and separate held-out current-tree evidence |
| Doctor mutation | **off** | Unlocks only after prerequisite task receipt (`PDR-053`) + board completion |
| Derived refill | **off** | Unlocks only after prerequisite task receipt (`PDR-081`) + board completion; writes a **separate** runtime task source |
| Seed lanes | **≤ 6** | Hard maximum for the seed program |

Privileged features remain off until their prerequisite task receipts exist.
Receipts alone do not rewrite protected anchors. Automatic mode still needs a
later independent held-out evaluation on a fresh current tree.

## Operator entry

```bash
# Machine-readable recipe (no process start)
python scripts/ops/agent_supervisor/proof_directed_planner_doctor.py recipe

# Full admission validation
python scripts/ops/agent_supervisor/proof_directed_planner_doctor.py \
  --state-dir /path/to/isolated/ops-state \
  validate

# Plan → start → status (report-only/shadow)
python scripts/ops/agent_supervisor/proof_directed_planner_doctor.py \
  --state-dir /path/to/isolated/ops-state \
  --lanes 4 \
  plan
python scripts/ops/agent_supervisor/proof_directed_planner_doctor.py \
  --state-dir /path/to/isolated/ops-state \
  start
python scripts/ops/agent_supervisor/proof_directed_planner_doctor.py \
  --state-dir /path/to/isolated/ops-state \
  status
```

Interface: **`PlannerDoctorOperations@1`**.

Commands: `validate`, `plan`, `start`, `status`, `stop`, `restart`, `pause`,
`drain`, `benchmark`, `promote`, `rollback`, `kill-switch`,
`kill-switch-clear`, `recipe`, `deposit-receipt`.

Use `--allow-dirty` only for hermetic diagnostics. Production admission
requires a **clean target**.

## Launch validation checklist

The launcher (`validate` / `plan`) checks:

1. **Clean target** — empty `git status --porcelain` (unless `--allow-dirty`).
2. **Exact gitlinks / capabilities** — configured submodule paths observed;
   required Planner/Doctor modules present; missing optional tools degrade
   rather than silently pass.
3. **Board / objective DAG** — parse seed taskboard and objective heap;
   unknown dependencies and cycles fail closed; initial ready set is reported.
4. **Protected anchors** — every listed anchor exists and remains outside the
   write set.
5. **Isolated state / worktree / merge queue** — state-dir and data paths do
   not collide with anchors; merge queue is singular and isolated.
6. **Provider / resource telemetry** — scheduler `resource_hints` declared;
   report-only does **not** require model providers.
7. **Maximum six seed lanes** — `lanes` and `max_lanes` cannot exceed 6.

Operations are **idempotent**, **fenced** (fence token + generation on every
state write), and **restartable** (stop/start or `restart` re-enters from
durable isolated state).

## Lifecycle

```text
idle → planned → running ⇄ paused
                 running → draining → stopped
                 * → killed (kill switch)
```

| Command | Effect |
| --- | --- |
| `plan` | Validate and materialize plan; create isolated layout; **no** dispatch |
| `start` | Fence start; dispatch allowed only if kill switch is clear |
| `status` | Phase, effective modes, gates, health, PID, lanes |
| `pause` | Cancel new dispatch; keep phase paused |
| `drain` | Cancel **future** dispatch; allow in-flight to finish |
| `stop` | Cancel dispatch; phase stopped |
| `restart` | Fence-stop then start; preserves kill-switch engagement |
| `promote` | Advance **one** rollout stage under gates |
| `rollback` | Demote one stage (or `--to-mode` safer only) |
| `benchmark` | Record benchmark **gate** intent; does not mutate oracles/anchors |

Re-running a command already in the desired state returns `reason_codes`
including `idempotent`.

## Kill switch

Engage:

```bash
python scripts/ops/agent_supervisor/proof_directed_planner_doctor.py \
  --state-dir /path/to/isolated/ops-state \
  kill-switch --reason operator_engage
```

Effects (always):

1. Force effective Doctor mode to **`report_only`**.
2. Demote elevated rollout modes back to the shadow family.
3. **Cancel future dispatch** safely (`dispatch_allowed=false`).
4. **Block promotion** (`promotion_blocked=true`).
5. Mark phase `killed` when a live run was active.

Clear is **operator-only** and does not auto-promote:

```bash
python scripts/ops/agent_supervisor/proof_directed_planner_doctor.py \
  --state-dir /path/to/isolated/ops-state \
  kill-switch-clear --operator-ack
```

After clear, the system remains report-only/shadow with dispatch off until an
operator explicitly `plan`/`start` again. Candidates and models cannot clear
the kill switch.

## Feature receipts

Deposit a body-free prerequisite receipt into isolated state (never into
protected anchors):

```bash
python scripts/ops/agent_supervisor/proof_directed_planner_doctor.py \
  --state-dir /path/to/isolated/ops-state \
  deposit-receipt \
  --feature refill \
  --task-id PDR-081 \
  --evidence-id evidence:refill-qualification@1
```

| Feature | Prerequisite task | Unlock rule |
| --- | --- | --- |
| `refill` | `PDR-081` | Board complete **and** receipt deposited |
| `doctor_mutation` | `PDR-053` | Board complete **and** receipt deposited |
| `automatic` | `PDR-090` | Board complete, receipt, **and** elevated profile (`automatic_enabled`); held-out current-tree evaluation still required before production automatic |

## Runbook scenarios

### Capability degradation

**Symptoms:** optional prover, embedding, or model provider missing;
`capabilities` report lists degradation; telemetry `unavailable`.

**Actions:**

1. Keep report-only/shadow — optional absence **must not** block report-only.
2. Record typed abstention / uncertainty debt; never convert unknown → pass.
3. Do not deposit promotion receipts on degraded required live checks.
4. Restore capability with an approved digest-bound deployment, then re-run
   `validate` and live qualification on the **current** tree.

### Stale state

**Symptoms:** fence generation gaps, PID reuse suspicion, worktree/lease
split-brain, state newer than HEAD, or board/objective drift vs frozen
anchors.

**Actions:**

1. `status` and inspect `operations_state.json` under the isolated state-dir.
2. `drain` or `stop`; if mutation risk, engage **kill switch**.
3. Reconcile worktrees and merge queue under the isolated roots only.
4. Re-run `validate` (clean target, DAG, anchors, gitlinks).
5. `plan` then `start` only after validation is green. Do **not** edit
   protected anchors to “fix” staleness.

### Rollback and quarantine

**Symptoms:** nonzero safety floor, authority escape, partial transaction,
failed fixed point, forged receipt, or policy violation.

**Actions:**

1. Engage kill switch (forces report-only; blocks promotion).
2. `rollback` one stage (or to `shadow`/`observe`/`off`).
3. Quarantine suspect worktrees and derived task-source rows under isolated
   paths; leave seed board/oracle/policy untouched.
4. Restore exact blob/tree/ref identities from checkpoints before any
   re-admission.
5. Independent revalidation on the current tree is mandatory.

### Held-out evaluation

Automatic mode remains disabled until:

1. Qualification and live epoch/rollout gates pass in shadow/canary.
2. A separate operator-approved **fresh current-tree** evaluation succeeds.
3. Independent **held-out** corpus evidence passes (see protected holdout
   manifest and quality oracle).
4. Safety floors are exact zero; quality non-inferiority and Pareto resource
   rules hold; anti-gaming checks pass.
5. Kill switch is clear; promotion is operator-driven one stage at a time.

Never use synthetic, skipped, or self-authored evidence for promotion.
Never mutate the holdout manifest, oracle, or benchmark policy from a
candidate lane.

### Recovery without editing anchors

1. **Detect** — `validate`, `status`, health/events under isolated state.
2. **Contain** — kill switch; `drain`/`stop`; cancel future dispatch.
3. **Diagnose** — read-only inspection of floors, receipts, gitlinks, DAG.
4. **Restore** — exact rollback of worktrees/refs; quarantine bad derived work.
5. **Re-admit** — clean target + `validate` + `plan` + optional one-stage
   `promote` only when gates hold.
6. **Never** edit plan, objectives, todo board, scheduler, authority policy,
   benchmark, or holdout anchors to recover.

## Safety floors (reminder)

Non-compensable floors include authority/policy/scope/secret/path escape,
stale or forged proof/CID admission, partial transaction, false fixed point,
rollback failure, synthetic/skipped evidence used for promotion, and
benchmark or oracle mutation. Any nonzero floor forces demotion and/or kill
switch behavior.

## Resource and lane bounds

| Bound | Seed value |
| --- | --- |
| Maximum seed lanes | 6 |
| Concurrency sweep (benchmark) | 1, 2, 4, 6 |
| Merge queue | single isolated queue |
| Implementation timeout (scheduler) | 10800 s (task-dependent) |

Lane labels are hints. Dependency, conflict, resource, provider, lease,
worktree, and merge-train constraints determine admitted width.

## Interfaces

| Interface | Role |
| --- | --- |
| `PlannerDoctorOperations@1` | Launch profile, lifecycle, kill switch (this guide) |
| `PlannerDoctorRolloutPolicy@1` | Quality-safe Pareto and anti-gaming promotion |
| `PlannerDoctorEpoch@1` | Bounded unattended improvement lifecycle |
| `PlannerDoctorRefill` | Derived successor work into a separate task source |
| `PlannerDoctorQualification@1` | Live E2E qualification evidence assembly |

## Quick reference

```bash
# Validate admission
python scripts/ops/agent_supervisor/proof_directed_planner_doctor.py \
  --state-dir "$STATE" validate

# Lifecycle
python scripts/ops/agent_supervisor/proof_directed_planner_doctor.py \
  --state-dir "$STATE" --lanes 4 plan
python scripts/ops/agent_supervisor/proof_directed_planner_doctor.py \
  --state-dir "$STATE" start
python scripts/ops/agent_supervisor/proof_directed_planner_doctor.py \
  --state-dir "$STATE" status
python scripts/ops/agent_supervisor/proof_directed_planner_doctor.py \
  --state-dir "$STATE" pause
python scripts/ops/agent_supervisor/proof_directed_planner_doctor.py \
  --state-dir "$STATE" drain
python scripts/ops/agent_supervisor/proof_directed_planner_doctor.py \
  --state-dir "$STATE" stop
python scripts/ops/agent_supervisor/proof_directed_planner_doctor.py \
  --state-dir "$STATE" restart

# Promote / rollback one stage
python scripts/ops/agent_supervisor/proof_directed_planner_doctor.py \
  --state-dir "$STATE" promote
python scripts/ops/agent_supervisor/proof_directed_planner_doctor.py \
  --state-dir "$STATE" rollback --to-mode shadow

# Kill switch
python scripts/ops/agent_supervisor/proof_directed_planner_doctor.py \
  --state-dir "$STATE" kill-switch
python scripts/ops/agent_supervisor/proof_directed_planner_doctor.py \
  --state-dir "$STATE" kill-switch-clear --operator-ack
```

Config: `config/agent_supervisor_proof_directed_planner_doctor_scheduler.json`

Ops script: `scripts/ops/agent_supervisor/proof_directed_planner_doctor.py`
