# agent_supervisor.rescue

**Layer:** Ops · **DAG role:** see [PACKAGE_MAP](../../../docs/architecture/agent_supervisor/PACKAGE_MAP.md)

## Purpose

Failure recovery: rescue planners/orchestrators, failure policy, and watchdog/recovery hooks.

## Who should import this package

| | |
| --- | --- |
| **This package may import** | `core`, `control`, merge/runtime hooks carefully |
| **Typical dependents** | todo_daemon, runtime |

## Modules

| Module | Path |
| --- | --- |
| `codex_failure_policy` | `rescue/codex_failure_policy.py` |
| `learning_recovery` | `rescue/learning_recovery.py` |
| `recovery_diagnostics` | `rescue/recovery_diagnostics.py` |
| `rescue_orchestrator` | `rescue/rescue_orchestrator.py` |
| `security_fault_matrix` | `rescue/security_fault_matrix.py` |
| `rescue_planner` | `rescue/rescue_planner.py` |
| `supervisor_recovery` | `rescue/supervisor_recovery.py` |
| `supervisor_watchdog` | `rescue/supervisor_watchdog.py` |
| `watchdog_guardian` | `rescue/watchdog_guardian.py` |
| `leased_service_watchdog` | `rescue/leased_service_watchdog.py` |
| `lane_resume` | `rescue/lane_resume.py` |
| `semantic_recovery` | `rescue/semantic_recovery.py` |
| `board_gap_refill` | `rescue/board_gap_refill.py` |
| `command_processes` | `rescue/command_processes.py` |
| `run_repo_pytest` | `rescue/run_repo_pytest.py` |
| `repair_dispatcher` | `rescue/repair_dispatcher.py` |
| `gate_recovery` | `rescue/gate_recovery.py` |

## Preferred imports

```python
from ipfs_accelerate_py.agent_supervisor.rescue.<module> import ...
```

Relative imports stay package-local (`from .<module> import ...`).

## Leased foreground services

`leased_service_watchdog` observes one explicitly registered systemd user
service. A `Config` pins its repository, board, lane inventory, exact loaded
ExecStart argv, unit fragment SHA-256, lease metadata, foreground status and
separate telemetry directory. Linux PID birth, boot, UID, namespace and cgroup
checks bind the systemd main process, protected launch trampoline and actual
foreground descendant. Unknown ownership, drop-ins and a pending daemon reload
refuse recovery.

```bash
python -m ipfs_accelerate_py.agent_supervisor.rescue.leased_service_watchdog \
  --config /path/to/leased-service-watchdog.json --once
```

Continuous observation omits `--once`. Explicit `--apply` authorizes only
`systemctl --user restart` of the pinned unit; `--once` remains observational.
The registered service must use `KillMode=control-group` and its ordinary lease
entrypoint. The watchdog never starts a raw lane or provider, signals arbitrary
PIDs, edits source or task status, reclaims foreign leases, or grants completion.

Task-declared deadlines, bounded retry waits, a missing owner and an idle ready
frontier drive diagnosis. Fresh heartbeats, log growth and native activity do
not extend task deadlines or reset the persisted restart budget. Current-task
provider failures and task/PID-bound native progress are recorded without
retaining provider bodies; historical log errors remain separate. Blocked
dependency frontiers and exhausted budgets produce actionable incidents.
Operator holds, interrupted Git operations and unowned dirty output overlap
suppress recovery. A second ownership snapshot fences every restart; subsequent
observations verify old cgroup process births are gone before recording cleanup.
Only actual board progress or configured repair-file content changes renew a
bounded recovery budget.

Temporary maintenance should use `maintenance_hold.maintenance_hold(path,
reason=..., ttl_seconds=...)` from the maintenance process, with a deadline of
at most 1800 seconds and an owned directory without group/other write access.
Keep the watchdog running while the foreground is paused.
The context manager archives its own exact marker when maintenance finishes.
If that process dies, the leased-service watchdog (`--apply`) and fleet hold
review can archive the unchanged registered inode after its deadline, retaining
the original JSON and PID birth/boot/namespace identity. An expired lease with
a live owner remains held. Unknown/manual markers, symlinks, unregistered
hardlinks and uncertain owner observations remain held. Observation-only runs
never release a marker. This does not reset recovery budgets or grant task
completion, and ordinary checkout, cgroup and effect fences still apply.

Semantic failure history is independently keyed by task and its declared
execution contract. Service/worker PID changes, backoff clocks, log growth and
native frames never erase it. Repeated provider failures, a required exhausted
dependency barrier, or sustained work without semantic validation produce
degraded, durable incidents; restarting the unit is not their repair. Legacy
attempt counters with missing in-flight proof remain explicitly unqualified
failure history instead of appearing healthy.

The deterministic lane writes its execution-progress journal before invoking
the producer and retains it across interruption and validation failure. After
six semantic failures for unchanged inputs (or the explicitly declared
`semantic_failure_limit` integer from 1 through 6), that task waits in-process for a real contract repair while other
ready tasks can advance. Declared source/configuration content changes rearm
only that contract; touching a file or waiting out a cooldown does not. Only a
completed semantic validation clears the failure history. Validator exit 76 is
an unjudged native call, not one of those six failures. A task that says
`Retry unjudged native: true` runs its producer again after the same 15-minute
backoff. Exit 75 admission waits stay validation-only. Live retries and
retry-wait heartbeats preserve their degraded state and publish the actual
retry deadline.

Heartbeats also publish `lane_execution_health` for unresolved failures on
other pending tasks. The lane stays degraded while those failures remain,
even if its current task has no failures. This aggregate is separate from
the active task's execution history and budget; completed historical tasks
are excluded, and source changes alone do not clear pending failure history.

An exhausted task can declare `semantic_recovery_command`,
`semantic_recovery_inputs`, `semantic_recovery_timeout_seconds` (at most the
task deadline and 1200 seconds), and `semantic_recovery_limit` (1 or 2). The
command must be a repository-local Python/Node script naming the exact task
ID. It executes inside the current foreground owner, never through a watchdog
spawn. The worker must verify its inherited checkout lease, operator holds,
current input hashes, edit ownership, and declared write scope before calling
the pinned model or applying any proposal. It cannot edit the taskboard or
the producer, validator, or recovery controller. These hooks remain opt-in;
an unavailable or unqualified repair stays an explicit degraded block.

`AGENT_SUPERVISOR_RECOVERY_CONTEXT_PATH` identifies an in-repository JSON
packet with schema `agent-supervisor/semantic-recovery-context@1`, exact task
and producer-contract identities, qualification fingerprint, admission index,
budget, input/source hashes, and body-free failure counts/stages. Admission
is persisted before invocation in `lane_resume_attempts.semantic-recovery.json`.
Its finite per-task budget survives process and contract changes. Repeating
the same diagnostics, JSON clocks/PIDs/formatting changes, logs, receipts,
native frames, and feedback cannot reset that budget or count as a source
repair. Failed, interrupted and no-op hooks keep the current source blocked.
Only a successful material production-source change can admit a subsequent
original producer and validator; hook success never completes a task.

Declared authored JSON under `test/e2e/functional-workflows` (or the `tests`
equivalent) is included in both the recovery source map and command-contract
hashes, alongside application code. A scenario-only repair is therefore a
material source change. Generated receipts, progress, reports, and runtime
evidence stay excluded, and this classification grants no edit permission.

A controlled canary may declare `semantic_recovery_fixture_inputs` for an
exact bounded JSON object under `fixtures/`, also listed in both Predicted
files and Execution inputs. This explicit production-input role allows a
mutable isolated fixture under runtime data without treating proof artifacts
as source. Log/progress/receipt/feedback/model artifacts are refused. The
watchdog observes the durable repair budget and its admitted deadline while
preserving independent owner-absence and task-deadline protection.

## Persistent repair dispatch

`repair_dispatcher` observes one existing watchdog manifest and starts an
explicitly configured repair worker only when evidence shows a stall. It
checks task progress as well as heartbeats. A repeatedly withheld closeout
gate becomes a diagnostic incident when it is the board's only selectable
task and its completed dependencies cannot advance it.

Observe before enabling dispatch:

```bash
python -m ipfs_accelerate_py.agent_supervisor.rescue.repair_dispatcher \
  --repo-root /path/to/workspace \
  --manifest-path data/program/watchdog/bundle_lanes.json --once
```

To run continuously, omit `--once` and supply `--apply`, `--prompt-path`, and
`--worker-argv-json`. For example, a worker argv of
`["codex", "exec", "-c", "approval_policy=\"never\"", "--sandbox", "workspace-write", "--json", "-"]`
reads the repair instructions from stdin using the
[documented non-interactive CLI](https://learn.chatgpt.com/docs/non-interactive-mode).
Choose a prompt that limits repair scope, preserves completion gates, and
requires a regression test. The dispatcher supplies the current observation
path in `REPAIR_DISPATCHER_STATUS_PATH`.

Use `--repair-lock-path` to share a fixer marker with an existing scheduler.
The default state directory is the manifest directory's `repair-dispatcher/`;
`status.json` records health and dispatch decisions, `state.json` preserves
retry budgets, and worker logs record repair attempts. A successful worker
exit awaits a new health observation. Three unsuccessful attempts for unchanged
evidence leave an explicit `attention_required` incident. Source/evidence
changes allow another bounded budget; `--evidence-path` binds additional inputs.
Standard operator holds suppress dispatch. The dispatcher does not grant
completion, alter taskboards, or restart the observed lane itself.

The deterministic lane also persists withheld-gate fingerprints in its
`*.gates.json` ledger. Repeating the same decision stops counting as a task
transition. Three unchanged refusals produce a body-free repair packet with
upstream dependencies and validation gaps; subsequent probes use a bounded
cooldown. A changed source or decision permits an immediate new probe. Other
ready tasks can advance while a gate waits. The dispatcher consumes the
lane's `gate_recovery_needed` packet directly.

After a soft refusal, an unchanged task uses its validator as a cheap probe;
it does not repeatedly rerun a browser producer. Refusal budgets bind to the
source contract, while structured diagnostic summaries ignore changing trace,
receipt and run identities. A source repair rearms the original producer,
which must still pass its final validator before the task can complete.
Transport acquisition and full-family acceptance should be separate tasks so
one transport never waits on an acceptance gate requiring another transport.

Exhausted command or validation attempts also escalate after their bounded
wait. The dispatcher checks persisted invocation evidence and the current
command contract, so fresh waiting logs cannot conceal a repeated failure.
Changed source, evidence, or actual task progress resets that observation.
Node preloads declared by a nested project resolve from that project while
task paths retain their workspace meaning; a repaired preload runtime gives
the affected command one fresh attempt budget through its contract hash.

`gate_recovery.admit_gate_repair_task` admits an explicit pending repair with
semantic validation and adds it to the gate's dependencies. It preserves
original statuses and acceptance, refuses cycles and scope changes, and can
assign a browser resource with its two-hour execution bound. The scheduler
must execute and validate the repair before closeout runs again.

## Watchdog ownership and command recovery

`watchdog_guardian` observes one manifest and its lane progress. Run it from
an environment with this checkout on `PYTHONPATH`:

```bash
python -m ipfs_accelerate_py.agent_supervisor.rescue.watchdog_guardian \
  --repo-root /path/to/workspace \
  --manifest-path /path/to/workspace/data/program/watchdog/bundle_lanes.json \
  --once
```

For persistent recovery, omit `--once`, add `--apply`, and run under a service
manager. The default poll is 30 seconds. Per-manifest locks, exact process
identity checks, and a durable restart cooldown prevent duplicate owners.
`HOLD`, `OPERATOR_STOP`, or `watchdog.disabled` in the workspace or manifest
directory suppress launches. The guardian replaces a verified absent owner;
it reports an unresponsive live owner without signaling it. Telemetry stays
in `watchdog-guardian-status.json` beside the manifest.

`lane_resume` binds produced evidence to the task command, validator, and
named scripts. Completion during retry backoff requires evidence changed by
the latest invocation, matching current content and contract, and a passing
validator. Interruptions invalidate previous invocation provenance. A changed
command or script gives an exhausted task one fresh attempt budget; unchanged
content and timestamp-only changes retain the cooldown.

`command_processes` enforces command cancellation, deadlines, and descendant
cleanup. `run_repo_pytest` accepts passing tests or independently reverified
pass reuse; ordinary skips cannot supply completion evidence. None of these
recovery helpers grants objective completion authority.

## Extending

1. Add modules here only if this package **owns** the concern ([placement table](../../../docs/architecture/agent_supervisor/PACKAGE_MAP.md)).
2. Update this README module table in the same change.
3. Prefer semantic public names; do not encode board prefixes into APIs.
4. Add focused tests under `test/api/` (or package-local tests).
5. Keep the dependency DAG acyclic.

## See also

- [Developer guide](../../../docs/architecture/agent_supervisor/DEVELOPER_GUIDE.md)
- [Package map](../../../docs/architecture/agent_supervisor/PACKAGE_MAP.md)
- [Semantic package page](../../../docs/architecture/agent_supervisor/packages/rescue.md)
- [Architecture](../../../docs/architecture/AGENT_SUPERVISOR_ARCHITECTURE.md)

Typed producer resource refusals use exit 75 plus one current-invocation JSON line
`{"task_id":"LANE-001","error":{"code":"browser_contention"}}`. The lane
persists a resource wait without incrementing semantic failures or acceptance-gate
refusals, invoking validators, or spending model recovery admissions. It retries
the original producer with 30-second exponential cooldowns capped at 300 seconds,
for at most 6 acquisitions per window and the lesser of the task deadline or
20 minutes. A closed window waits another persisted 300 seconds, then admits a
fresh bounded acquisition window automatically on the unchanged contract. Task
refusal totals and the latest 16 window summaries persist across these windows;
semantic failure history and model recovery spending remain intact. Only the
original successful producer and validator can complete the task. Untyped or
foreign-task exit 75 remains the ordinary acceptance gate; other nonzero exits
retain their original failure behavior.
