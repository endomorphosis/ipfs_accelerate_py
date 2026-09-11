# Native Doctor interruption recovery

This opt-in route preserves an unknown provider attempt and admits a distinct
continuation when a fixed native Doctor edit operation has positively returned.
It does not apply to historical unrestricted Grok callbacks, including SAWM-016.
It does not settle a callback or authorize task completion.

## Admission before dispatch

Configure `NativeDoctorCallback` as the database daemon provider before the first
claim. Supply a real repository, private Doctor state root and exact
`DoctorExactEdit` values. The callback permits detached local edits only; it
accepts no tool command, network callback or target ref. Its default absolute
attempt ceiling is three, configurable between one and 1024. A positive daemon
`max_task_attempts` further lowers that ceiling.

The initial native attempt stores the exact recipe, source hashes, Git binary
hash, base commit and permitted paths. Provider entry refuses replacement
callbacks and retrospective profile attachment. Native Doctor disables Git
hooks, credentials, network protocols, filters and background maintenance.
The source profile binds all new producer/consumer modules and the existing
Doctor, lifecycle, daemon and coordinator implementations.

The callback creates a native lifecycle record and detached candidate, records
its start in the execution store, writes durable intent, applies the exact
edits and records complete candidate/checkpoint evidence. A closed observation
means those bounded synchronous operations returned. Missing closure is a
refusal, including interruption during worktree preparation. Original generic
provider/effect settlement remains separate. No arbitrary process-group closure
is inferred from a return code or missing process.

## Reservation and continuation

`reserve(daemon, attempt, callback)` requires the exact elapsed latest native
claim and the unchanged original context attempt. It freshly takes the actual
Doctor writer lock, validates source, full candidate/checkpoint population,
intent and native lifecycle, and preserves the candidate in strict quarantine.
A task-scope coordinator barrier precedes the control `in_progress → blocked`
CAS and its execution journal record. The original attempt, phases, callback
history and spent ordinal remain unchanged.

`admit_continuation(daemon, attempt, callback)` separately rereads that custody
and budget, performs the exact `blocked → retrying` CAS, and journals admission.
The coordinator clears its barrier only through a retained
`NativeContinuationAdmission`. This capability reads the actual native daemon,
control and execution stores plus quarantine/candidate under the real lock;
receipt dictionaries cannot substitute for it. The latest expired fence is
checked within the coordinator transaction. Every incomplete stage keeps the
task unclaimable to other supervisors, including after process death.

Only complete native admission permits the scheduler to exclude the old running
attempt from resumable work. Explicitly resuming that old attempt is denied.
A later claim has a new attempt ID, increased ordinal and increased fence, and
uses none of the previous candidate as completion evidence. Repeating an
already completed admission after a lost response returns its exact history.

The normal daemon pass automatically performs these two separate operations
only for attempts admitted with this exact callback profile. Changed source,
candidate, checkpoint, closure evidence, claim or budget leaves recovery denied.
No path deletes or restores the retained original candidate. Ordinary providers
keep their existing behavior.

## Qualification and deployment limits

Tests use real native DuckDB stores, Git worktrees, kernel file locks, competing
processes, real `os._exit` boundaries and a paused callback thread. They cover
unknown-history/opaque-row preservation, incomplete admission barriers, forged
and stale capability refusal, budget exhaustion and separate next dispatch.

The runtime candidate is source-only. The native SAWM required-branch, nested
pins, source-seal, protected DuckDB/Quack dependency and closed archival readback
qualification must be completed in a composed adoption successor before live
use. The current Grok-to-Codex Terra/high policy remains independent. No old
Grok callback gains this profile retrospectively.
