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
workers are never killed merely because such a hold exists. A dangling symlink
at a configured hold path still counts as a stop marker. Read-only probes
continue during holds: `health` reports `operator_hold`, while `observed_health`
and the observation contain the current underlying condition. To pause the whole
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

The service uses the existing exclusive owner's DuckDB handle. Clients do not
open database files. AST evidence and content hashes deduplicate across trees;
conflicting hashes under the same snapshot identity fail. Reference records do
not verify the referenced semantic evidence. The datasets-authoritative profile
continues to prohibit an accelerator-local AST writer while permitting these
explicitly unverified derived references. Git, ipfs_kit_py, and ipfs_datasets_py
retain their existing source and semantic authority.

The aggregate owner can separately call `bind_fleet_observation_reads()` and
publish `fleet-observation-read.token` with mode 0600. An independent reader
uses `TypedStateOwnerConnection(..., fleet_observation_read=True)`; its exact
peer receives a 120-second capability for identity and fleet-observation queries
only. It cannot append observations, mutate a board, or open the derived writer
service with that credential. Admission modes and credentials remain distinct.

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
