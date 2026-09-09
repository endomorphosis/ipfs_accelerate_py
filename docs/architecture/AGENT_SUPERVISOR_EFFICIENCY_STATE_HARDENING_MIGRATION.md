# Agent Supervisor Efficiency and State Hardening Migration

ASEH-061 integrates admitted transition and recovery behavior into the
existing `IntentRepository` to typed-Quack-owner path. Compatibility
adapters warn and delegate. They do not write independently, delete public
APIs, or let a plan delta waive production integration.

## Sole production authority

| Role | Interface | Path |
| --- | --- | --- |
| Writable owner | `TypedStateOwnerCommandGateway@1` | `ipfs_accelerate_py/agent_supervisor/task_sources/typed_state_owner.py` |
| Host | `QuackStateServer@1` | `ipfs_accelerate_py/agent_supervisor/runtime/quack_state_server.py` |
| Transactional adapter | `IntentRepository@1` | `ipfs_accelerate_py/agent_supervisor/task_sources/intent_repository.py` |
| Operational facade | `DatabaseTaskSource@1` | `ipfs_accelerate_py/agent_supervisor/task_sources/database_task_source.py` |

Every production mutation reaches the exclusive loopback Quack owner through
`IntentRepository`. A bound owner connection is the owner's write surface. A
live Quack transport client must submit typed owner commands; it must not
execute independent SQL against the control store.

## Owner-paused staged maintenance

1. Stop new claims.
2. Drain active mutating claims and leases.
3. Seal the current root and recovery evidence.
4. Pause the Quack owner.
5. Stage and validate the exact patch.
6. Merge atomically.
7. Restart with the same external credential handle.
8. Reconcile state and receipts through
   `IntentRepository.reconcile_after_authenticated_restart` hosted by
   `QuackStateServer.reconcile_typed_owner_authority`.
9. Reopen claims only after projection/event parity holds.

A running-owner mutation, active claim, or lease during cutover rejects the
migration and keeps claims closed.

## Supported legacy APIs

These names remain callable. Each emits `IntentCompatibilityWarning` /
`QuackCompatibilityWarning` and then performs exactly one write through
`IntentRepository`:

- `compare_and_set_status` / `cas_task_status` / `cas_status`
- `DuckDBTaskSource.compare_and_set_status`
- `TaskTransitionService.transition` / `apply_admitted_transition`
- `record_evidence`
- `record_validation_result`
- `record_queue_backoff`
- `record_queue_retry`
- `rearm_blocked_task`
- `recover`
- `reconcile_legacy_stale_unstall_projection_drift`

`apply_admitted_transition` is the production integration of the admitted
CAS, no-retry, event/revision-parity, and terminal lease/fence contract.
It never retries a stale CAS and never writes except through
`cas_task_status`.

## Caller replacements

| Legacy caller | Replacement |
| --- | --- |
| `DuckDBTaskSource` | `DatabaseTaskSource@1` |
| `DuckDBTaskSource.compare_and_set_status` | `DatabaseTaskSource.compare_and_set_status` |
| `TaskTransitionService.transition` | `IntentRepository.apply_admitted_transition` |
| `TaskTransitionService.transition_legacy` | fail closed (`reject_unsupported_legacy_path`) |
| Markdown board status | `IntentRepository@1` observation only |
| Direct SQL mutation bundles from clients | typed owner command `compare_and_set_status` |
| `IntentRepository.cas_task_status` over Quack transport | `TypedStateOwnerCommandGateway@1` |

Public APIs are not removed. Replacement, caller migration, and loss of
independent write authority must be proved before any later deletion task.

## Unsupported paths (warn then fail closed)

- Direct SQL against task/objective tables
- `transition_legacy` / silent compatibility bypass
- Independent DuckDB writes while a live Quack owner exists
- Dual write (adapter plus owner)
- Public API deletion
- Silent fallback
- A plan delta that waives ASEH-061 production integration

## Restart and reconciliation

Owner restart uses the same durable authority:

1. Authenticate with the existing external credential handle. Credentials
   are never rematerialized.
2. `reconcile_legacy_stale_unstall_projection_drift` repairs only that
   bounded legacy shape.
3. `unstall_stale_in_progress_tasks` retries leftover gates through
   repository CAS and events.
4. `assert_projection_matches_events` proves materialized state matches
   the event/receipt history.
5. `SupervisorRecovery` fences stale actors and preserves unknown-outcome
   truth. It does not become a second writer.

Scheduling resumes only after that report is admitted.

## Rollback

Rollback discards or reverts only the scoped task-worktree patch through
the canonical merge/recovery path.

- Preserve observed effects and receipts.
- Do not delete public APIs as part of rollback.
- Do not reopen claims until the previous or replacement owner restarts
  and reconciles.
- The SQL mutation inbox remains an owner-internal exclusive-writer
  protocol in the same process; rollback must not install a second
  process writer.

## Plan delta rule

ASEH-060 may prove a newer current-tree writable owner. A plan delta may
substitute that exact path only. It cannot waive production integration,
install a disconnected wrapper, or classify a public API as removed
without replacement proof.
