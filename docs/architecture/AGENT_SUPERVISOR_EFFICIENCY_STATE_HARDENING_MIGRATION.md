# Agent Supervisor Efficiency and State Hardening Migration

ASEH-061 integrates admitted transition and recovery behavior into the
existing **IntentRepository → typed Quack owner** path. Compatibility
adapters warn and delegate. They are not a second writer.

A plan delta cannot waive this production integration. Public APIs are
not deleted.

## Production authority

Every production mutation must traverse:

1. `IntentRepository@1` — transactional substrate (CAS, events, receipts)
2. `TypedStateOwnerCommandGateway@1` — sole writable owner
3. `QuackStateServer@1` — exclusive host of `control.duckdb`

Live callers enter through the Quack owner. The repository writes only on
the owner's bound connection or the admitted Quack transport. Direct
DuckDB SQL, `DuckDBTaskSource` independent writes, and markdown board
edits are not production authority.

## Disposition vocabulary

| Disposition | Meaning |
| --- | --- |
| **canonical** | The typed Quack owner hosted by `QuackStateServer` |
| **adapter** | Supported legacy API: warn, then one IntentRepository mutation |
| **deprecated** | Retained public surface; independent writes fail closed |
| **fail closed** | Unsupported path: warn, then refuse without writing |

## Supported legacy APIs and replacements

| Legacy caller | Replacement | Behavior |
| --- | --- | --- |
| `DuckDBTaskSource.compare_and_set_status` / `cas_status` | `IntentRepository.cas_task_status` | Warn, then CAS + event |
| `TaskTransitionService.transition` | `IntentRepository.cas_task_status` | Warn, then the admitted status CAS |
| `DatabaseTaskSource.compare_and_set_status` | `IntentRepository.cas_task_status` via typed owner commands | Already the cutover path |
| `SupervisorRecovery.rebuild` | `IntentRepository.rebuild_owner_restart_projection` | Parity check; no extra writer |
| `SupervisorRecovery.takeover` | `IntentRepository.take_over_owner_session` | Advances fence; preserves task/event truth |
| Owner start unstall / legacy drift | `IntentRepository.reconcile_legacy_stale_unstall_projection_drift` then `unstall_stale_in_progress_tasks` | Same bound repository |

Call these through `IntentRepository.route_legacy_api` or
`QuackStateServer.route_legacy_api`. Both emit
`IntentRepositoryCompatibilityWarning` and execute exactly one repository
mutation.

## Unsupported paths (fail closed)

These warn, then raise. They must not write.

- direct SQL against `control.duckdb`
- markdown board status edits
- independent `DuckDBTaskSource` writes
- dual-write (legacy path plus a second store)
- disconnected wrappers / independent writers
- silent legacy fallback
- `TaskTransitionService.transition_legacy`
- a plan delta that tries to waive production integration

## Owner-paused staged maintenance

Cutover and rollback use the same durable owner:

1. Stop new claims and drain mutating claims and leases.
2. Seal the current root and recovery evidence.
3. Pause the Quack owner (`QuackStateServer.pause_for_maintenance`).
4. Stage and validate the exact patch.
5. Merge atomically through the canonical merge/recovery path.
6. Restart with the **same** external credential handle.
7. Reconcile through `IntentRepository` (`reconcile_after_authenticated_restart`).
8. Resume claims only after event and materialized-state parity.

A running-owner mutation, active claim, or lease during cutover rejects
the migration and keeps claims closed.

## Restart and reconciliation

After process loss, rebuild from admitted events and the checkpointed
owner projection. `IntentRepository.rebuild_owner_restart_projection`
proves projections match events without appending a spurious recovery
event (that would change the watermark). Takeover advances the fencing
epoch and owner session on this same repository. Credentials are never
rematerialized; only durable grant-binding identities are checked.

`QuackStateServer.start` already unstalls stale gates through the bound
IntentRepository before listen. Authenticated restart must repeat that
reconciliation before scheduling resumes.

## Rollback

Rollback discards or reverts **only** the scoped compatibility-adapter
patch. It does not:

- restore `DuckDBTaskSource`, markdown, or direct-SQL writers
- delete public APIs
- erase observed effects or receipts
- waive the IntentRepository → typed Quack owner path

After rollback, restart with the same credential handle and reconcile
through IntentRepository before reopening claims.

## Single-write invariant

One admitted operation produces one repository transaction and at most
one domain event. Compatibility adapters call the existing
`cas_task_status`, `recover`, evidence, and queue methods. They do not
open a second connection for effect.

## Caller migration

Replace production callers as follows, then keep the public legacy name
until a later task proves deletion:

- Scheduler and daemon CAS → `DatabaseTaskSource` / typed owner command
  `compare_and_set_status`, which already binds `IntentRepository`
- Ad-hoc `DuckDBTaskSource.compare_and_set_status` →
  `IntentRepository.cas_task_status` or `route_legacy_api("compare_and_set_status", ...)`
- Candidate `TaskTransitionService.transition` → the same CAS on the
  bound owner repository
- Recovery helpers → `IntentRepository` rebuild/takeover used by
  `SupervisorRecovery`

Until replacement, caller migration, equivalence, and loss of independent
write authority are proved, the public names remain and either warn-then-route
or warn-then-fail.

## Nonclaims

- This cutover does not promote policy pointers.
- This cutover does not delete public APIs.
- Compatibility adapters, fixtures, and Markdown or DuckLake observations
  cannot independently claim authority.
- A plan delta cannot substitute a different writable owner or skip
  production integration.
