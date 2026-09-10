# Existing legacy queue through an admitted owner

`merge/owner_merge_queue.py` exposes the existing `MergeQueue` transitions through
`TypedStateOwnerGateway`. It operates on **`merge_requests`**, the table in the
legacy `merge_queue.duckdb`. The separate `DatabaseMergeQueue` table
`merge_queue_entries` does not represent settlement or migration of that queue.

This is an operational port and a staging prerequisite. It does not discover,
start, migrate, or admit a production queue owner. No native inventory, live
queue, task status, goal, receipt, or budget is changed by adding this module.

## Admission and operation contract

The native owner must already hold the queue store's exclusive tracked
`DuckDBConnection`, serve an authenticated Quack endpoint, and have a live typed
gateway using that same handle and transaction lock. Its reviewed migration
must have supplied the typed owner metadata, `store_generations`, and
`client_sessions` contracts. Binding requires the exact current gateway identity
and the actual existing legacy table/column/primary-key/unique-index schema.
Binding validates existing state; it creates no schema and imports no files.

The owner calls `bind_legacy_merge_queue_service` with its expected identity,
repository ID, target branch, and explicit queue capacity, processing, retry,
lease-duration, and worktree-byte policies. An existing configured worktree usage
probe can be supplied by the owner. Queue policy must match the qualified native
configuration; disabling a limit requires its own configuration authority.

Binding returns no token and issues no grant. A separately authorized existing
`OwnerClientGrant` must explicitly contain the needed operations and **exactly**
these entity scopes: `repository_id`, `target_branch`, and `consumer_id`. The
client's process birth, kernel peer, bounded grant lifetime, attached session,
store UUID, owner birth, generation, and fence must remain current. Clients
cannot supply connections, database paths, callbacks, policy changes, SQL,
bootstrap tokens, or migration commands through the port.

`OwnerMergeQueueClient` takes an already admitted `TypedStateOwnerConnection` and
those three scopes. Its `call` method accepts these closed operations:

| Operation | Arguments | Behavior |
| --- | --- | --- |
| `enqueue` | `branch_name`, `task_id`, `priority`, `lane_id`, `commit_sha`, `canonical_task_id`, `canonical_task_key`, `metadata_json` | Preserve legacy task/commit/target deduplication; return the existing row or insert one bound request. |
| `get` | `request_id` | Observe one row in the exact bound target. |
| `claim` | `request_id` | Claim that exact eligible pending row under the existing capacity checks. |
| `owns_claim` | `request_id`, `claim_token`, `claim_generation` | Observe the current consumer, token, generation and expiry fence. |
| `complete` | claim coordinates plus `metadata_json` | Apply the existing claimed-row transition; no task or goal completion authority. |
| `requeue`, `quarantine` | claim coordinates plus `reason`, `metadata_json` | Apply the existing retry or quarantine transition without deleting the row. |

The consumer comes from the admitted scope. Metadata is bounded JSON object
text. Legacy floating timestamps and metadata travel as `request_json` text to
preserve their values within the typed envelope, whose canonical representation
forbids floats. A response also binds its schema, owner identity and operation;
`completion_authority` is always `false`. Terminal idempotent replies report the
existing queue disposition and supply no semantic acceptance or closeout proof.

Unbound legacy rows and other targets stay unchanged and inaccessible through
this bound view. A stored dedupe key that conflicts with the row's target or
computed identity causes denial; it is never repaired or rebound as a side
effect. Transition metadata cannot alter the target binding.

## Transaction and recovery boundaries

Each legacy context borrows the owner handle without entering, exiting, or
closing it. It may commit or roll back only a transaction it successfully began;
it refuses an inherited transaction. Reads, writes, and transaction control use
the existing tracked handle's single-attempt path, avoiding implicit reconnect
or mutation replay. The service also refuses replacement of the admitted
underlying handle.

Before COMMIT, owner/store/session coordinates are revalidated under the
gateway's transaction lock. The final current grant, peer, scopes and expiry are
checked under the gateway's grants lock, which stays held through COMMIT. This
follows the existing lock order: transaction lock, then grants lock. Revocation
therefore occurs before the admitted commit or after it; it cannot interleave
between the final grant check and COMMIT. Admission denial rolls back while
preserving a healthy handle and poisons that borrowed operation so legacy
post-rollback deduplication cannot turn the denial into success.

BEGIN, SQL, COMMIT and rollback failures are not swallowed. An uncertain commit,
failed rollback, or natively poisoned loan freezes the shared owner binding
and retires the queue service. The gateway sets a sticky flag on the exact tracked connection:
other services' reads, writes and commits fail with `DuckDBOwnerRetiredError`,
waiting peers wake and deny, and implicit or explicit reconnect paths cannot
reopen it. Retirement itself neither closes nor reopens the handle. Only a later
explicit owner shutdown can dispose of the frozen handle; a replacement needs
a newly qualified owner and connection. Ordinary non-retired DuckDB recovery
remains available outside this uncertain queue boundary.

A healthy still-open transaction gets at most one rollback before retirement
following a failed COMMIT. A poisoned handle never enters rollback's implicit
reopen path. Failed rollback consumes only the borrowed ownership record, so
legacy cleanup cannot retry it or expose an uncertain transaction to another
service. No local file fallback, generation change or new grant is produced.

Claim expiry still denies a stale worker's mutation. The port deliberately does
not run the legacy opportunistic age-based claim reaper: expiry alone does not
prove that provider work and all Portal callbacks have stopped. Reopening,
claim recovery, cancellation, supersession, receipt import, compaction and final
queue settlement require separate native authority. JSON stage projections are
not written or pruned; the database remains the queue authority.

## SPAR migration and qualification plan

The read-only SPAR review on 2026-09-10 observed native root
`ce06f4317a92349e32662d4e1b3bd0674ba96e2c`, 51 completed task receipts, 32
unsettled goals, and intentionally drained lanes. The source validator and
configured-board preflight admitted that root, not this later staging tree.
The native transaction-failure candidate
`b32c6c83ba0b66313008083419384769018e53eb` and main fix
`9ccf71271a634f8f17b6def2da6703006ea4c33c` have the same patch identity
`ed5a55fbd932956e531877abb70788bc6ab668bc`. Preserve that fix when assembling a
native candidate: it protects `DatabaseMergeQueue`, but does not migrate the
legacy queue.

The native configured queue is
`data/agent_supervisor/semantic_preserving_autonomous_remodularization_v1/merge-queue`
under the SPAR worktree. The following are requirements for a future qualified
migration, not instructions to run a live migration from this staging change:

1. Assemble a native source candidate preserving the current native history and
   the transaction-failure fix. Port the owner integration and the actual native
   producer/merge-consumer calls; this module supplies exact-request operations,
   not a drop-in replacement for every legacy queue method. Qualify those calls
   with real callback/merge evidence and the sealed current-root validator.
   Do not rewrite the sealed planning root or weaken source checks to make a
   staging path pass.
2. Name and qualify the actual legacy queue store and its exclusive owner. A
   checksum-bound migration must preserve every `merge_requests` row and all
   target metadata, dedupe keys, statuses, attempts, consumer IDs, claim tokens,
   claim generations, retry times, and timestamps. It must supply the approved
   typed owner control metadata without introducing another task/goal truth
   store. This migration/admission artifact is not supplied by the port. An
   empty `merge_queue_entries` table is not a substitute.
3. Before any stop or file access, acquire the native maintenance, resume and
   merge leases; recheck current root/index and exact owner/master births, and
   obtain repeated fresh zero-provider observations with all Portal callbacks
   idle. Provider activity, callback activity, source drift, unknown liveness,
   or absent migration authority denies the transition. Preserve all source,
   index and provider work. Never infer quiescence from an elapsed timeout.
4. At that admitted maintenance boundary, preserve the queue database, WAL and
   legacy receipts coherently through the exclusive owner. Rehearse the approved
   migration on an isolated preserved copy and compare the complete old row
   population and coordinates, including unbound, pending, processing, completed,
   quarantined and cancelled work. The service never opens the live file to
   inspect, import or repair it. Any nullable/missing claim schema requires an
   explicit reviewed migration; binding fails closed.
5. Start only the qualified owner using DuckDB + Quack and its typed gateway,
   bind the existing queue handle, and verify the real store UUID, new owner
   birth, generation and fence through authenticated native observations.
   Issue only the separately admitted producer/consumer grants, with exact
   repository, branch and consumer scopes, expiry and process births. Do not
   generate broad grants, file fallbacks, or bootstrap credentials as recovery.
6. Verify the migrated queue population through the admitted owner before
   enabling production consumers. Use isolated qualification rows for enqueue,
   dedupe, claim, stale-token denial, retry and completion tests; do not manufacture
   task status or receipt CAS results. Verify grant revocation/expiry at commit,
   restart fencing, owner handle survival after a clean rollback, and uncertain
   transaction denial. Qualify native settlement evidence on **this** queue.
7. Keep remaining semantic gates independent: the datasets producer's exact
   accepted root, its independent admission, current forest CAS, required-mode,
   safety, capstone and fixed-point evidence, and the native goal/closeout CAS
   adapter. Queue disposition and passing infrastructure tests cannot authorize
   those gates or merging unfinished work.

Only after those conditions hold can a native watchdog adopt this port as an
automatic repair capability. Source drift and active provider/callback work
remain explicit blockers; preserving data takes precedence over repeated
unqualified migration attempts.
