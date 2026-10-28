# Taskboard fleet watchdog

The fleet watchdog keeps independently configured DuckDB supervisors under
persistent observation. Each board retains its existing state owner, operator,
leases, completion policy, and implementation workers. The watchdog does not
open a live DuckDB file or treat a task count as acceptance authority.

The host deployment covers SPAR, SAWM, ASEH, PCTDD, PCPR, and DOEP. A separate
inventory binds each board to its exact checkout, runtime, configuration and
native status/ensure commands. This accommodates older sealed runtimes without
copying a newer supervisor into their accepted source trees.

When the native status reader explicitly rejects an unavailable or invalid
live receipt, the probe can re-read it up to three times within an additional
20-second publication window. Each attempt rechecks the owner's exact process
birth and readiness. The native reader must admit a new receipt with its usual
freshness and identity checks; rejected samples are never used. Blocked, stuck,
and command-failure decisions are returned immediately. Probe details include
`native_status_attempts` so recovery from a publication gap is visible.

Process observations include the Linux state and, when readable, the kernel
wait channel. A stopped (`T`/`t`) or uninterruptible (`D`) owner, supervisor or
daemon degrades health even when supervisor heartbeats remain fresh. Its exact
live birth stays visible; the probe does not treat it as a missing process.
Persistent conditions enter the existing repair queue after the configured
grace period, while transient kernel waits can clear before repair. Holds and
native recovery admission still apply, and providers in other lanes remain
visible. These diagnostics do not authorize signalling or callback replay.

## LegalIR service fleet

The existing LegalIR deployment retains four implementation services:
`legal-ir-autoencoder-v1.service` and `legal-ir-autoencoder-v1-lane-{1,2,3}.service`.
`legal-ir-autoencoder-v1-monitor.timer` runs the bounded
`board_service_fleet_monitor.py` oneshot every 60 seconds, with five-second
accuracy. `config/legal_ir_autoencoder_fleet_monitor.json` names the four
`legal_ir_autoencoder_lane_{0,1,2,3}_monitor.json` configurations, each bound to
one exact service, state directory, board namespace and task shard. These are
host-specific deployment configurations; preserve their existing bindings.

The native monitor separates parent heartbeat, worker liveness and useful task
progress. Polling and task selection cannot renew the ready-work deadline.
Fresh active work remains healthy within its absolute deadline; a stale worker,
expired task deadline, stale parent heartbeat or stalled eligible ready backlog
can qualify for recovery. A fresh completed board with no runnable work is
healthy idle, without asserting completion of the wider legal programme.
Dependency waits and typed retry, quota, resource or native policy fences remain
visible without a service restart. Review and legal-meaning gates remain closed;
observations do not supply their missing decisions.

Each lane permits only its configured unit's restart, with a 600-second cooldown
and at most four charged attempts per hour. The attempt and restart intent are
durable before the effect; corrupt budget state suppresses recovery. Immediately
before charging, the monitor refreshes native evidence and checks the same
systemd `MainPID` and `ExecMainStartTimestampMonotonic` service birth. Changed or
unverifiable identity, newly active work, recovered idle or a native fence
suppresses the restart without charging. A replacement generation receives a
fresh ready-work window. These are sequential observations, not an atomic
snapshot or a generation-conditioned `systemctl restart`; changes after the
recheck remain possible.

The monitor writes per-lane health/recovery records and the shared
`data/legal-ir-autoencoder-v1/fleet-health.json` and `fleet-events.jsonl` beneath
the existing LegalIR worktree. Repeated structured failures, duplicate active
tasks and stale merge receipt projections remain diagnostics, without a fleet
restart or task mutation. The existing `ipfs-taskboard-legal-ir-watchdog.service`
reads that report through its installed `service_fleet_probe` with exact expected
lane/unit bindings and a 180-second freshness limit. Persistent incidents enter
its existing durable `incident.json` and repair queue; the existing
`ipfs-taskboard-legal-ir-repair.service` dispatches the configured repair worker.
No additional loop, queue or task-state owner is needed. Native child recovery,
claim ownership, holds, retry budgets and acceptance gates retain their authority;
repair success still requires a fresh configured probe.

Inspect the native timer, diagnostic services and retained evidence:

```sh
systemctl --user status legal-ir-autoencoder-v1-monitor.timer \
  ipfs-taskboard-legal-ir-watchdog.service ipfs-taskboard-legal-ir-repair.service
journalctl --user -u legal-ir-autoencoder-v1-monitor.service -n 5 --no-pager
cat /home/barberb/lift_coding/.worktrees/legal-ir-autoencoder-v1/data/legal-ir-autoencoder-v1/fleet-health.json
cat ~/.local/state/ipfs-taskboard-watchdog/legal-ir/status.json
```

The diagnostic configuration is
`~/.config/ipfs-taskboard-watchdog/legal-ir/fleet.json`; its `state_dir` contains
`legal-ir/incident.json` when an incident exists and `repairs/legal-ir/` for
retained jobs, prompts, reports and logs. For source maintenance, stop only the
native monitor timer and running oneshot before installing the verified monitor
source; preserve the four implementation lanes and diagnostic services. Restart
the same timer afterward. For protected board, plan or worker-source maintenance,
use the existing fleet quiescence procedure instead: pause both native recovery
timers and diagnostic watchdog/repair dispatch, stop the worker target, then
verify provider children and native claims have quiesced before writing. Restore
the same services only after contract validation. Never remove claims manually
or bypass review/resource gates. This prevents recovery from fighting intentional
maintenance. Enabled user units and user lingering provide persistence across
logout.

From the supervisor repository, the focused failure-injection suite is:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -o addopts= \
  -o cache_dir=/tmp/legal-ir-monitor-pytest-cache \
  test/api/test_board_service_monitor.py test/api/test_board_service_fleet_monitor.py
```

Qualification on public base `a8f18f6a3fc9a1d3f5efe8307dcff2b25a8cb95f` passed
all **160 tests**. The cases cover bounded recovery, changed/reused PID births,
fresh worker/completed-idle/resource-fence suppression and unreadable evidence.

## Isolated LegalIR diagnostic profile

`install_fleet_watchdog.py --profile legal-ir` stages a separate instance of these same watchdog/repair owners. It consumes the existing minute LegalIR fleet-monitor report and does not add another lane monitor. The six-board global configuration, units, operator holds and repair history remain unchanged. Use this profile when the global runtime/source bindings require their own separate review.

Prepare a host inventory containing exactly one `service_fleet_report` board with an absolute report path, namespace,180-second freshness limit and explicit lane-to-service bindings. Its ensure command must be empty: the existing LegalIR service monitor retains exact-service recovery. The diagnostic board cannot acquire publication or completion authority.

```sh
python3 scripts/ops/agent_supervisor/install_fleet_watchdog.py \
  --profile legal-ir --inventory /path/to/legal-ir-inventory.json \
  --repair-cwd /accepted/accelerate-development-checkout --dry-run
```

Review the returned file postimages, then repeat without `--dry-run` to stage them. Profile staging performs no systemctl commands; `--enable` records intent only. Activate explicitly through `systemctl --user daemon-reload` and enable/start only `ipfs-taskboard-legal-ir-watchdog.service` and `ipfs-taskboard-legal-ir-repair.service`. Existing installations require exact idempotence or an explicit reconciliation; they are never silently overwritten.

Configuration lives in `~/.config/ipfs-taskboard-watchdog/legal-ir/`; state and repair evidence live in `~/.local/state/ipfs-taskboard-watchdog/legal-ir/`. The dedicated coding unit is `ipfs-taskboard-legal-ir-repair-job.service`, preventing collision with the global dispatcher. The profile inherits the operator's existing repair model/argv, uses the explicitly selected development checkout, and limits this diagnostic board to three cumulative coding attempts with visible `repair_budget_exhausted` escalation.

The report probe validates regular bounded JSON, freshness, namespace, exact lane/service population, counts and native liveness. Confirmed empty engineering work stays healthy without advancing a progress timestamp or claiming task/goal completion. Newly runnable work starts its own stall interval. A fresh bound dependency wait is a noncoding wait. Stale, malformed or foreign reports cannot prove recovery; a coding exit code alone cannot do so either. A fresh bound report can verify diagnostic runtime recovery only, retaining all semantic/task/goal authority as false.

The completed135-task legacy engineering board and the separate34-goal/64-task LegalIR programme retain distinct scopes. Before corpus-worker cutover, reviewed execution packets and authenticated native task/goal scope must be admitted through the existing owners. This diagnostic profile does not rematerialize that population, rearm attempts, approve legal meanings or declare the US Code formalized.

### Observe the scientific programme before enabling recovery

An additional entry in the existing fleet can set `"observation_only": true`.
This is a strict Boolean operator policy, not a probe result. The watchdog
continues its configured status probe, health/progress assessment and local
observation writes, including while holds are present. It selects no recovery
action and preserves retained recovery counters, pending intent and cooldown.
It does not review holds, stop a repair unit, heal an owner, enqueue a coding
job or publish. Existing queued/running repair records remain unchanged and
appear as held with reason `observation_only`; direct enqueue and job execution
also enforce the configured policy. Other fleet entries retain their policy.
This does not cancel workers that were already running.

Native healing also requires exactly one explicit absolute `--inventory` in
the configured probe, with any `--board` matching the configured identity.
The selected inventory must contain exactly one matching entry with absolute
checkout/database paths, and its checkout must match the configured `cwd`.
Missing, malformed, foreign or incomplete inventories do not fall back to
`~/.config/ipfs-taskboard-watchdog/inventory.json`. Board names and observed
unit names alone cannot establish ownership. Unbound fleet heals and direct
dump/attach/collapse/restart paths refuse before their first effect. Tests of
these paths must supply a private inventory or assert that refusal.

Use an admitted read-only native status command as the probe. The policy does
not make an arbitrary operator-configured command read-only. Keep the native
owner's `bind_database_status_scope` binding to the exact board namespace,
plan root, repository tree and complete task CID population. Its status-only
grant has no command operations. Do not open the live database or substitute
saved `status`/`ready` projections for that authenticated read.

The corresponding `live_board_probe` inventory entry can additionally pin
`native_population`, an object with exactly `task_cids` and `goal_cids`.
Each is a complete list of 1–512 unique nonblank identities, at most256
characters each. Copy the accepted inventories, not aliases, counts or a sample.
The probe checks both sets against the admitted database status and completion
projection. Missing, added, substituted or duplicate identities reject the
observation. A pinned entry cannot fall back to daemon task/readiness counts;
rejected native completion flags cannot close the board. Omitting the pin
preserves the existing unpinned adapters.

For LegalIR, retain distinct entry identities for the legacy engineering fleet
and the `legal-ir-span-graph-lean-refinement-20261016` scientific programme.
The latter's retained inventory is34 goals/64 tasks; its historical blocked
snapshot is not current status. Reuse the existing G090/LIR-9004 handoff and
scientific observation controls. A fresh owner/status binding and compatible
probe must be qualified before enrollment. This policy adds no LIR launcher,
task population, credential or execution packet. Enable recovery only through
an explicit subsequent operator configuration after its native capability,
retry/resource budgets and independent verification are admitted.

## Operation

Each board may explicitly configure read-only active-storage checks:

```json
{
  "storage_checks": {
    "filesystems": [{
      "path": "/home/operator",
      "min_available_bytes": 10737418240,
      "min_available_percent": 2
    }],
    "git_worktrees": ["/home/operator/worktrees/active-board"]
  }
}
```

The watchdog reports `observation.storage_diagnostics`, including filesystem
capacity, available bytes for the service user (`f_bavail`, excluding reserved
blocks), and available percentage. Either threshold independently raises a
diagnostic when the sample is strictly below it; omitted thresholds default to
zero. Missing or inaccessible filesystems are reported as unavailable rather
than zero capacity. Configure every relevant mount explicitly; no ancestor
filesystem is substituted for a missing path.

Git checks resolve each configured checkout's `.git` marker, linked worktree
`commondir`, and registration backlink. They check that the resolved metadata
directories, object directory, and `HEAD` still exist, including when a linked
worktree survives deletion of its external Git metadata. Pointer reads are
bounded and reject special files. No Git commands run and no object integrity,
commit completeness, or unpublished-history recovery is claimed.

Storage faults write `<state_dir>/<board_id>/storage-incident.json` with
`action: diagnostic_only`; a later healthy sample records resolution there.
These observations continue under holds and appear alongside native health.
They do not change task progress, completion gates, recovery budgets, or action
selection, and do not enqueue a coding repair, delete files, or stop services.
Existing native failures still follow the normal recovery policy. Runtime
upgrades preserve configured checks. Configuration allows at most 32 filesystem
paths and 32 worktrees per board; include active submodules explicitly.

Two user services run continuously:

* `ipfs-taskboard-watchdog.service` probes every 60 seconds. Each board has its
  own lock, persisted observation, recovery intent and exponential backoff.
  Missing owners get 60 seconds of grace; blocked/degraded states get five
  minutes; healthy-but-idle boards get fifteen minutes without task progress.
  A fresh heartbeat alone does not count as task progress. Recent provider
  output plus a matching live descendant prevents false idle alarms.
  Database event-cursor changes also remain diagnostic: maintenance or lease
  events cannot reset the task-stall deadline. For native authenticated probes,
  only increasing completed-task/receipt counts or decreasing unsettled goals
  renew that deadline. Status/retry cycles, regressed counts returning to an
  earlier value, and authentication gaps cannot postpone idle-stall repair.
  Opaque-token adapters retain their explicit progress contract. Completion
  still requires the separate native gate.
* `ipfs-taskboard-repair.service` processes the durable coding-repair queue,
  one board at a time. Each job runs in its own systemd cgroup for at most 40
  minutes. Unresolved work is retained and retried with a 30-minute to six-hour
  backoff. A new probe verifies recovery; a model's successful exit does not.
  The worker status distinguishes an empty queue from held jobs and jobs
  awaiting their next retry, and records the active board before dispatch.
  Between jobs it also checks for recovered boards and actionable new evidence.
  A recent watchdog sample selects a fresh native probe; cached status alone
  cannot retire a repair. Authenticated healthy recovery retires the queued job
  without starting another coding worker. Restored native admission, new source heads, admitted task
  completions, changed blocked-task sets, or settled goals can advance a pending
  continuation while retaining its coding attempt history and a five-minute
  minimum gap after the previous job. Heartbeat, PID and event-cursor changes
  alone do not shorten retries. Each coding attempt also records a fresh
  configured probe before launch. Authenticated source or native task evidence
  changed by the final probe earns another pass five minutes after finish,
  preserving the attempt count even if the worker exits unsuccessfully. A
  successful worker report alone earns no continuation. For older finished
  jobs, a recorded native reconciliation within three minutes before launch
  can supply the baseline for up to six hours after finish; a matching fresh
  probe and an exact attempt/report marker admit this continuation only once.
  Rechecks are limited to once per two minutes per queued board, honor holds
  before and after probing, and never publish a board or change native task
  budgets. Configured source integrity must pass before recovery or publication
  can be verified, even when native health temporarily reports healthy.
  Continuation reads prioritize the previous report's pending action, blockers,
  root cause and source commits within bounded field budgets. Large deployment
  transcripts cannot push the pending action out of the next job's context;
  the original report path remains available for verification.

An operator can separately configure a board's `diagnostic_handoff`, for example
`"monitoring-handoff.json"`. It is relative to
`<state_dir>/repairs/<board_id>/`, not the live native checkout. This optional
file helps a later repair find completed root investigations and the exact
reports behind them. It supplements the previous job report; missing or rejected
handoffs do not discard that report or change the repair queue, holds or budgets.
Runtime upgrades preserve the configured path and install its reader.
Reports kept elsewhere must first be explicitly copied as diagnostic reports
into this private board directory, retaining their content hashes; the reader
does not scan global reports or follow references outside its configured scope.

The handoff uses this closed schema (example hashes and times must be replaced):

```json
{
  "schema": "fleet-repair/diagnostic-handoff@1",
  "board_id": "doep",
  "observed_at": 1789120000,
  "expires_at": 1789123600,
  "source_heads": {".": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"},
  "source_integrity_sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
  "completed_stages": ["source_qualification"],
  "reports": [{
    "path": "root-recovery/report.json",
    "sha256": "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
    "stage": "source_qualification"
  }]
}
```

`diagnostic_handoff.source_binding(observation)` returns the exact source heads
and integrity hash for this schema. The latter is SHA256 of the probe's
`details.source_integrity` serialized as UTF-8 JSON with sorted keys, ASCII
escaping and separators `(',', ':')`. Both must match the configured probe
captured immediately before the new coding job, within three minutes. The
handoff must be current and have a lifetime of at most 24 hours. These checks
bind an observation of Git heads and cleanliness, not loaded code or exact dirty
file bytes; the worker must still verify current native source and custody.

Allowed stages are `incident_observation`, `source_qualification`,
`generic_fix_qualification`, `native_fix_qualification`, `graceful_closure`,
`preservation_archive`, `source_transition`, `native_observation`,
`service_recreation`, `native_restart`, `post_start_verification`, and
`independent_review`. Every listed stage needs an explicit matching report.
At most eight reports are read, each at most 1 MiB; the handoff is limited to
32 KiB and reads check a two-second overall budget. Files must be owned regular
JSON paths beneath the board's owned `0700` repair directory. Symlinks, hardlinks,
world-writable files, traversal and private credential/vault path names are
rejected. Existing `0664` reports inside that private directory need no chmod.
Pinned directory/file identities are rechecked, and report hashes must match.

Only closed stage names, times, source observations, report paths and hashes
enter the prompt. Report bodies, free-form prose, environment and commands are
never copied. A stage is historical diagnostic data, not permission to replay
old manifests, process identities or recovery tools, waive a gate, release a
callback or claim completion. A stale, missing, malformed or oversized handoff
produces a bounded unavailable note while ordinary repair continues.

For a known stopped-owner condition, the watchdog invokes only the board's
configured native ensure command. Otherwise it enqueues a repair. The coding
worker first reproduces the incident, adds a regression test for the shared
supervisor, validates its change, and uses existing recovery/requalification
controls to deploy it. It can commit and normally push validated changes under
the user's authorization. Unknown faults may need several repair jobs; an
unresolved fault remains visible rather than becoming a fabricated success.

Coding jobs use the existing packaged Grok-to-Codex fallback owner and the user's
existing authentication. `repair_worker.argv` supplies the Codex fallback command.
An optional `repair_worker.implementation_route` uses the existing complete
six-field route validator to bind the primary model, fallback model/reasoning and
hard-quota-only boundary. For the LegalIR route, set Grok4.7, Codex `gpt-6.1-sol`
and high reasoning, and include explicit matching `--model gpt-6.1-sol` and
`-c 'model_reasoning_effort="high"'` in the fallback argv. Actual primary and
readiness-probe models then share this binding; conflicting or missing fallback
options fail before queue changes. The validator is included in the immutable
installed runtime. Other configurations without this optional binding preserve
their existing route and model behavior. This does not change task acceptance,
repair-attempt limits, resource admission or live configuration automatically.
See the
[official unattended execution documentation](https://learn.chatgpt.com/docs/non-interactive-mode).
No local model service is started; the existing llama-server mask is retained.

`OPERATOR_STOP`, `HOLD`, `watchdog.hold`, and `watchdog.disabled` files configured
for a board prevent both ensure and coding-repair dispatch by default. Existing live
workers are never killed merely because such a hold exists. A dangling symlink
at a configured hold path still counts as a stop marker. Read-only probes
continue during holds: `health` reports `operator_hold`, while `observed_health`
and the observation contain the current underlying condition.

An operator can explicitly list a custodial marker in the board configuration's
`launch_only_hold_files` when its purpose is to reserve relaunch for an existing
native operator, such as cron. The file stays in `hold_files` and on disk;
watchdog ensure/start remains forbidden, while coding repair can proceed through
native source qualification and drain gates. The repair prompt retains that
custody restriction. Full `HOLD`, `OPERATOR_STOP`, `watchdog.disabled`, and symlink
markers cannot be narrowed this way. This setting grants no task completion,
callback replay, or source admission authority. Runtime upgrades preserve it.

To pause the whole
fleet, stop both services; a currently running repair job has its own unit:

```sh
systemctl --user stop ipfs-taskboard-watchdog.service ipfs-taskboard-repair.service
systemctl --user stop ipfs-taskboard-repair-job.service
```

The typed owner reports `status_bootstrap_configured` and
`status_bootstrap_scope_bound` separately from transport availability. These
configuration flags are diagnostic; each status attach still revalidates its
live authority. An authenticated status peer whose scope is missing or no
longer admitted receives `status_scope_not_admitted`. Invalid credentials,
client IDs, stores, and peer UIDs remain opaque handshake failures. Neither
result authorizes restarting a live owner or creating federation records to
admit a legacy board. Such boards require a supported board-scoped admission
and their existing sealed requalification process before independent closeout.

Non-federated `DatabaseTaskSource` launchers can explicitly call
`bind_database_status_scope` on their ready exclusive owner with the sealed board
namespace, plan CID, original repository tree, and complete task CID population.
The owner checks every persisted task against that binding on admission and
before reads. The resulting short-lived, kernel-peer-bound status grants expose
only executor projections and the existing transactional completion snapshot;
mutation commands remain unavailable. Changed namespace, plan, tree, or task
population rejects the read. This adapter preserves the original task identity
and does not infer goal satisfaction or release qualification from task counts. The live
probe accepts the native `database-board-status@1` envelope only when its
namespace, owner birth/generation and complete task population agree with the
fresh completion snapshot. Unsettled goals remain visible and block closeout
even when every task has completed.

If a board checkout disappears, preserve its incident hold outside the missing
directory. The installer can retain that unavailable board only while one of
its explicit absolute hold paths is an existing file. It continues monitoring
the outage and honors the hold; it never reconstructs authoritative task state
from a JSON projection.

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

When upgrading from inside a repair job, add `--defer-repair-restart` to the
installation command. Monitoring adopts the new immutable release immediately;
the dispatcher finishes its current job, reloads configuration, then exits so
systemd starts it from the new release. A running job therefore does not wait
on a restart of its own dispatcher. Preserve externally owned board launchers
in the inventory; an empty `ensure_argv` leaves their native unit untouched.

Recovery reuses an existing service when its unit already runs the complete
native operator with a lifetime outside the repair job. Adding another launcher
adapter in that case would create an unnecessary sealed-control amendment.
Boards without such an admitted launcher need the full native operator delegated
before credential handoff, followed by the board's source qualification. For a
runtime update, the native maintenance gate first prevents new dispatch intents
while the current provider finishes; source and merge leases then protect the
verified shutdown and forward source transition. Current-root qualification and
native resume remain required after the transition. Historical bootstrap
completion receipts remain bound to their original sources.

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

Supervisor objective and codebase refill passes default to a 600-second timeout,
including native launches that omit the corresponding CLI options. A timed-out
pass records its existing timeout/cooldown evidence and yields to scheduling.
Explicit per-pass timeout settings remain supported; zero explicitly disables
the in-process limit. Provider execution budgets are separate from refill limits.

Legacy owner projections that can reuse a cached task snapshot are diagnostic,
even when their transport label says authenticated Quack. They may request a
completion review, but publication still requires an independent admitted live
snapshot and the board's actual acceptance gate.

Dedicated database status admission also exposes `completion.closeout.snapshot`.
`completion_closeout_snapshot(task_cids)` binds the entire sealed population to
one owner generation/birth and one transaction containing completion receipts,
tasks, goals, dependencies, unreleased claims, unsettled merges, task blocks, and
local proof obligations. Missing relations and populations over 512 rows remain
explicit unknown/truncated observations. This endpoint does not evaluate sealed
goal contracts, verify external semantic obligations, or authorize completion.

### Separate derived coordination owner

A dedicated native Quack owner can enable `--derived-coordination`. Its private
`derived-coordination.token` admits short-lived, kernel-peer-bound sessions through
`TypedStateOwnerConnection(..., derived_repository_id=repository_id)`. The token
is separate from board status admission; these sessions cannot mutate tasks or
author completion. `DerivedCoordinationClient` exposes closed operations for AST
snapshots, parse-cache lookup, and content-hash/AST-CID/state-root references.
Each request is bound to one repository; parsing admits at most eight files and
32 KiB of source, with bounded request/result sizes and owner-lock admission.

Long-lived supervisors can construct a client from the compiled fleet deployment:

```python
from pathlib import Path
from ipfs_accelerate_py.agent_supervisor.analysis.derived_coordination import DerivedCoordinationClient

client = DerivedCoordinationClient.from_fleet_deployment(
    Path.home() / ".local/state/ipfs-quack-fleet/deployment.json",
    repository_id="git:my-repository",
    client_id="supervisor:my-lane",
)
```

Each explicit operation rereads the managed owner's credential and attaches a
fresh repository-scoped session. The same client object therefore survives owner
replacement and credential rotation without retaining an expired grant. Failed
requests are returned to the caller and never replayed automatically. Missing,
invalid, or unavailable owner bindings fail closed; clients never open the DuckDB
file. Deployment and credential reads require bounded regular files.

The service uses the existing exclusive owner's DuckDB handle. Clients do not
open database files. AST evidence and content hashes deduplicate across trees;
conflicting hashes under the same snapshot identity fail. Reference records do
not verify the referenced semantic evidence. The datasets-authoritative profile
continues to prohibit an accelerator-local AST writer while permitting these
explicitly unverified derived references. Git, ipfs_kit_py, and ipfs_datasets_py
retain their existing source and semantic authority.

The same owner exposes `record_artifact`, `lookup_artifact`, `list_artifacts`, and
`capabilities` for `vector_embeddings`, `bm25_index`, `knowledge_graph`,
`proof_cache`, and `certificate` references. An immutable cache identity includes
the repository, source tree, artifact kind, SHA-256 input digest, producer ID and
revision, and SHA-256 parameters digest. Changing the source, model or producer
version, inputs, or configuration requires a different identity; publishing a
different artifact CID under an existing identity is rejected. Concurrent
supervisors use the same owner lock, and reads retain the admitted repository
scope. Listings use keyset cursors and pages of at most 64 bounded references.

For example, with a repository-scoped `DerivedCoordinationClient` named `client`:

```python
identity = dict(
    tree_id=git_tree_id,
    artifact_kind="vector_embeddings",
    input_digest=source_input_sha256,
    producer_id=embedding_model_id,
    producer_revision=pinned_model_revision,
    parameters_digest=embedding_configuration_sha256,
)
client.call("record_artifact", **identity, artifact_cid=materialized_index_cid)
cached = client.call("lookup_artifact", **identity)["result"]["artifact"]
```

Producers must materialize their artifacts and consumers must verify their
content and provenance. This registry does not generate embeddings, search an
index, validate a proof, or certify a task. Every response explicitly retains
`artifact_verified=false` and `completion_authority=false`, including proof and
certificate references. The datasets-authoritative profile permits registration
without constructing an accelerator-local AST writer. Compiled deployments
advertise these capabilities; the dedicated owner health check verifies them
through an authenticated Quack session. An incompatible capability report is
reported for repair without repeatedly restarting an otherwise reachable owner.

The aggregate owner can separately call `bind_fleet_observation_reads()` and
publish `fleet-observation-read.token` with mode 0600. An independent reader
uses `TypedStateOwnerConnection(..., fleet_observation_read=True)`; its exact
peer receives a 120-second capability for identity and fleet-observation queries
only. It cannot append observations, mutate a board, or open the derived writer
service with that credential. Admission modes and credentials remain distinct.

Repair jobs retain the last usable report path before launching a successor.
If that worker is interrupted or returns without writing a complete report,
the next attempt receives the preserved, bounded continuation context and an
explicit missing-report diagnostic. Report reads remain confined to the same
board directory and reject symlinks. This preserves recovery instructions;
it does not reset retry budgets or treat a worker's report as completion evidence.
Recovery of an idle task stall requires admitted completed-task, completion-
receipt, or goal-settlement advancement. Cursor, retry-revision, and source-head
changes do not qualify. A fresh watchdog stall also remains in force when the
original repair concerned an endpoint that has since become reachable.

A quarantined merge candidate can be retired after a newer implementation has
been accepted using `MergeQueue.supersede_quarantined`. The caller must supply
an exact, content-addressed review and an independent current-acceptance
verifier. The queue checks the candidate, canonical task, target and claim
generation, then verifies acceptance again before committing. It records a
cancellation with the original candidate and quarantine history preserved;
this does not claim that the old candidate merged or complete a task or goal.
Changed native receipts, source roots or validation evidence must reject the
review. Ordinary `cancel` continues to refuse quarantined work.


SPAR native closeout requirements use the versioned `spar-closeout-profile@1`
owner-local adapter. Its launcher verifies original bootstrap source hashes,
reconstructs exact task/goal identities and dependencies, and refuses a changed
completion policy. The admitted closeout read compares native contracts and
current revision-bound completion receipts inside the owner transaction. It also
reports a bounded fresh repository-forest observation and nomination-only report
failures. These observations neither issue accepted semantic roots nor mutate
native goals. All thirteen SPAR policy requirements remain mandatory, including
required-mode receipts, noncompensable safety floors, and the self-hosted capstone.

The remaining producer work belongs to the existing authorities: datasets must
independently verify and issue accepted semantic-root evidence, and kit must
verify the corresponding current source-forest CAS receipt. Unverified derived
AST/CID records and aggregate observations cannot substitute for either. A
subsequent SPAR-specific owner adapter must bind those real receipts to the exact
native goal revisions, task receipts, declared mode/capstone/fixed-point evidence,
and quiescent lane/merge obligations before performing native goal CAS settlement.
The current profile explicitly reports these missing producer/admission paths;
it does not use the older VRIF-specific four-producer goal contract.

Task completion CAS and admitted-event replay preserve inherited unknown-callback
reopen budgets on the task body without adding telemetry to a sealed completion
receipt. Historical mismatches remain rejected by exact receipt equality until
a separately admitted repair or event-projection recovery handles them.

The sealed SPAR launcher also asks the native owner to recover legacy completion
receipt projections before launching lanes. This narrowly repairs the old
inherited retry-counter bug: the current task and its revision must equal the
known legacy projection of the original content-verified completion event and
predecessor revision. The original completion receipt must independently verify.
The correction keeps its status, revision, completion receipt and historical
rows; it moves inherited retry telemetry to the task body and records an explicit
`intent.completion_projection_repaired` event in the same transaction. Event
replay admits either the exact preimage or the already corrected body.

This is a launcher-only owner operation, unavailable to status grants or worker
RPCs. It requires the bootstrap-bound task scope, current native owner birth and
generation, complete native populations, and settled claims/effects/runtime
records under the existing owner lock. Busy or unrecognized states defer repair;
nonmatching bodies are reported without mutation. No offline store is opened and
no semantic acceptance, task completion, or goal acceptance is issued.

A board inventory can additionally set `source_integrity_paths` to a list of
objects containing an absolute `repository` and relative `paths`. Scope these
paths to executable control-plane code; unrelated task edits remain permitted.
For example, an accelerator checkout can select
`ipfs_accelerate_py/agent_supervisor` and `scripts/ops/agent_supervisor`.

The standalone probe checks those Git paths before importing native status code
and again after the read. Dirty, missing, invalid or timed-out scopes report
`source_integrity_not_verified`, suppress automatic ensure and completion
candidacy, and cannot verify a repair as healthy or published. The check has a
five-second Git deadline, disables optional index writes and uses literal
pathspecs. Hidden index flags, symlinks, nested gitlinks and truncated file
lists are rejected; each nested repository needs its own explicit entry. It never restores files or signals existing providers.

This is a conservative cleanliness constraint. It does not prove the bytes
already loaded by running interpreters, qualify a new commit, or replace a
native sealed runtime descriptor. An external writer that keeps changing live
control-plane code must be reconciled with the recovery owner; repeated source
restoration alone cannot establish lasting qualification.

The semantic-preserving remodularization owner can persist and re-read its
[kit source-forest CAS component](kit_source_forest.md) through the existing
native authority. This component does not settle semantic acceptance or goals.

The [legacy owner merge queue port](legacy_owner_merge_queue.md) provides scoped
operations on an existing `merge_requests` queue through its admitted typed
owner. It preserves claim fences and freezes the shared connection after an
uncertain transaction. Native row migration, producer and consumer admission,
and final queue settlement still require their own qualified transitions.

A native preflight that performs many intent reads can use
`with source.intent.read_session():` to reuse a standalone client connection
for the bounded observation. Nested sessions share that scope in the calling
thread and reject repository mutations before SQL. The outer scope closes its
adapter on success or interruption. Existing pooled Quack clients retain their
per-read liveness checks and repository-owned close; injected owner connections
retain their existing lock and connection ownership. Revoked discovered Quack
sessions fail closed without silently switching transports.

The read session controls resource lifetime only. It does not begin a read
transaction, freeze task heads, grant authority, or replace native source/owner
admission. The caller must retain the final snapshot comparison and retry only
the typed observation drift that the native restart-check contract permits.
