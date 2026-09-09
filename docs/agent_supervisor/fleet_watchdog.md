# Taskboard fleet watchdog

The fleet watchdog keeps independently configured DuckDB supervisors under
persistent observation. Each board retains its existing state owner, operator,
leases, completion policy, and implementation workers. The watchdog does not
open a live DuckDB file or treat a task count as acceptance authority.

The host deployment covers SPAR, SAWM, ASEH, PCTDD, PCPR, and DOEP. A separate
inventory binds each board to its exact checkout, runtime, configuration and
native status/ensure commands. This accommodates older sealed runtimes without
copying a newer supervisor into their accepted source trees.

## Operation

Two user services run continuously:

* `ipfs-taskboard-watchdog.service` probes every 60 seconds. Each board has its
  own lock, persisted observation, recovery intent and exponential backoff.
  Missing owners get 60 seconds of grace; blocked/degraded states get five
  minutes; healthy-but-idle boards get fifteen minutes without task progress.
  A fresh heartbeat alone does not count as task progress. Recent provider
  output plus a matching live descendant prevents false idle alarms.
* `ipfs-taskboard-repair.service` processes the durable coding-repair queue,
  one board at a time. Each job runs in its own systemd cgroup for at most 40
  minutes. Unresolved work is retained and retried with a 30-minute to six-hour
  backoff. A new probe verifies recovery; a model's successful exit does not.

For a known stopped-owner condition, the watchdog invokes only the board's
configured native ensure command. Otherwise it enqueues a repair. The coding
worker first reproduces the incident, adds a regression test for the shared
supervisor, validates its change, and uses existing recovery/requalification
controls to deploy it. It can commit and normally push validated changes under
the user's authorization. Unknown faults may need several repair jobs; an
unresolved fault remains visible rather than becoming a fabricated success.

Coding jobs use the installed Codex CLI's unattended execution and the user's
existing authentication. They retain the configured model choice. See the
[official unattended execution documentation](https://learn.chatgpt.com/docs/non-interactive-mode).
No local model service is started; the existing llama-server mask is retained.

`OPERATOR_STOP`, `HOLD`, `watchdog.hold`, and `watchdog.disabled` files configured
for a board prevent both ensure and coding-repair dispatch. Existing live
workers are never killed merely because such a hold exists. To pause the whole
fleet, stop both services; a currently running repair job has its own unit:

```sh
systemctl --user stop ipfs-taskboard-watchdog.service ipfs-taskboard-repair.service
systemctl --user stop ipfs-taskboard-repair-job.service
```

## Completion and Git publication

An all-terminal observation triggers a closeout review. The worker verifies
the authoritative task and goal receipts, terminal gate, zero active claims,
settled merge queues, no blocking obligations and the accepted source heads.
It reconciles accepted worktrees through the board's merge queue. It does not
merge arbitrary abandoned implementation or rescue branches.

`fleet_completion.publish_completed_board(manifest, state_dir)` performs the
subsequent publication. Its explicit manifest contains:

```json
{
  "schema": "agent-supervisor/fleet-publication@1",
  "board_id": "example",
  "completion_gate": {
    "cwd": "/accepted/board",
    "argv": ["python3", "verify_live_completion.py"]
  },
  "repositories": [{
    "id": "accelerate",
    "root": "/accepted/board",
    "source_ref": "refs/heads/accepted-integration",
    "validation": [{"argv": ["python3", "-m", "pytest", "test/api/test_changed_behavior.py"]}],
    "dependencies": []
  }]
}
```

The gate must return `authoritative: true`, `complete: true`, the same `board_id`,
integer zero values for `active_claims`, `pending_merges`, `blocking_obligations`,
and `source_heads` mapping each repository ID to its exact accepted commit.
Each board must provide a genuine live gate; no universal positive gate is
inferred from Markdown, compatibility snapshots, or read replicas.

Repositories declare submodule dependencies as
`{"repository": "datasets", "path": "external/ipfs_datasets"}`. Dependencies
publish first. Local clone origins are followed to their actual GitHub
repository, without changing the user's remotes. Source checkouts must be clean.
Merges and validation happen in isolated worktrees against freshly fetched
GitHub `main`. The publisher checks the source and remote again before an
ordinary push. Conflicts, stale evidence, changed gitlinks without dependencies,
dirty validation output or failed tests produce a retained hold for repair.
There are no force pushes, resets of live checkouts or automatic branch deletion.

## Install and inspect

From a tested supervisor checkout, provide a host-specific inventory and a
separate development checkout for the repair worker:

```sh
python3 scripts/ops/agent_supervisor/install_fleet_watchdog.py \
  --inventory /path/to/inventory.json \
  --repair-cwd /path/to/separate/repair-checkout --enable
```

The installer creates a content-addressed standalone runtime under
`~/.local/lib/ipfs-taskboard-watchdog/releases/`, avoiding optional provider
imports and sealed-checkout module shadowing. It writes:

* Configuration: `~/.config/ipfs-taskboard-watchdog/fleet.json`
* Inventory: `~/.config/ipfs-taskboard-watchdog/inventory.json`
* Fleet health: `~/.local/state/ipfs-taskboard-watchdog/status.json`
* Per-board evidence: `~/.local/state/ipfs-taskboard-watchdog/<board>/`
* Repair queue, prompts, reports and logs:
  `~/.local/state/ipfs-taskboard-watchdog/repairs/<board>/`

User lingering must be enabled for operation without an interactive login.
Check the live units and recent observations with:

```sh
systemctl --user status ipfs-taskboard-watchdog.service ipfs-taskboard-repair.service
journalctl --user -u ipfs-taskboard-watchdog.service -n 20 --no-pager
cat ~/.local/state/ipfs-taskboard-watchdog/status.json
```

## Supervisor correction included

The multi-supervisor runner previously exempted a stalled supervisor from
restart indefinitely if its child JSON retained `active_task_id` or
`implementation_in_progress`. It now requires a fresh, in-generation child
heartbeat for that exemption. Missing, old, future or previous-generation
heartbeats cannot prevent recovery. Existing exact process-identity fencing
still governs any actual restart.

Regression coverage exercises stale-active recovery, concurrent watchdogs,
command and descendant timeouts, durable cooldowns, changing diagnoses,
operator holds, repair scheduling, non-authoritative observations and real Git
publication races. Live sealed boards must adopt the supervisor correction
through their own accepted source transition; the watchdog installation alone
does not rewrite those seals.
