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
| `lane_resume` | `rescue/lane_resume.py` |
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

Validation-gap admission rules ship with the rescue package. An installed or
relocated supervisor can admit the bounded census and matrix repair tasks
without requiring a sibling datasets checkout or unpublished workspace files.

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
