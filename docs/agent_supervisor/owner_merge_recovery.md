# Owner-backed merge recovery

The recovery runtime supplies the interfaces missing from the SPAR legacy queue
port: durable scan cursors, one canonical train-consumer lease, and preserved
train receipt versions. `OwnerMergeQueueAdapter`, `OwnerMergeRecoveryRuntime`,
`MergeTrain`, and `DatabasePortalExecutionBridge` use the same admitted owner
and target. This capability does not accept tasks, goals, sources, migrations,
or callback closure.

## Source integration and its current admission prerequisite

`bind_database_portal_execution_from_args(..., owner_merge_runtime=runtime)` is
the production factory boundary. It checks the admitted checkout, target, board,
lane and attempt namespace, then injects the same queue into the Portal factory,
database bridge, post-merge recovery callback, and merge-train binding. SPAR and
configured Quack merge recovery refuse an absent pair before binding execution
callbacks or constructing a legacy queue. An already constructed owner queue
also refuses an unpaired bridge or train before creating local state.

The same call must separately supply `admitted_owner_merge_config_cid` and
`admitted_owner_merge_plan_cid` from the current native source/plan admission.
They must match the owner-provisioned runtime. Reusing a previous runtime at the
same checkout, lane and attempt root with a changed or missing current config or
plan identity is rejected. Arbitrary parsed JSON or the runtime's own stored
values are not used to manufacture these independent handoff inputs.

The native SPAR operator still needs a separately qualified queue-store
migration and bootstrap handoff before it can supply that dependency:

- `scripts/materialize_semantic_preserving_remodularization_program.py` uses
  `_build_state_owner` to retain the **task** `paths['database']` store. Its
  configured legacy `merge-queue/merge_queue.duckdb` is a different store.
- `_SparStateOwnerBootstrapBroker._grant` validates the native daemon and
  supervisor births, then issues task operations against the task owner.
  `StateOwnerBootstrapCredentials` carries that closed task-only endpoint and
  grant. Neither proves custody or migration of the legacy queue store.
- `implementation_daemon.main` consumes those task credentials and calls the
  factory without a queue runtime. It therefore reaches the explicit missing
  pair refusal for this candidate. There is no implicit migration, ambient
  credential discovery, alternate queue directory, or in-memory cursor fallback.

The native operator has no existing admission record for a coherent preserved
legacy queue/receipt/cursor population or its transfer to a new queue owner.
Adding the new grants to the task broker would not supply that missing store
custody. The next source change must connect the already admitted native
maintenance/source transition to a checksum-bound offline migration rehearsal,
then retain the qualified queue owner and issue its separate pair through the
existing exact-peer bootstrap channel. It must preserve all legacy rows, claims,
tokens, generations, lease uncertainty, receipts and cursor evidence. Existing
task receipts or a mutable status projection cannot authorize this transition.
The current source deliberately does not claim that this native invocation or
live adoption has happened.

## Owner provisioning and paired clients

Before workers, an independently qualified queue owner must already hold the
legacy queue's exact tracked DuckDB handle, canonical writer custody, owner
schema, and typed gateway. The existing `bind_legacy_merge_queue_service` checks
the exact queue schema and explicit native queue policies. It never imports or
creates a legacy queue.

The owner-only `provision_legacy_merge_recovery_schema` installs the recovery
schema and records an immutable migration identity over the supplied scope and
preserved-state payload. It is not a typed worker operation. Scope bindings are
exactly `board_namespace`, `config_cid`, `plan_cid`, `lane_id`, and absolute
`attempt_root`, with repository and target included in the scope CID. Native
source/config/plan admission must establish these values; the scope CID alone
does not approve them. Explicit receipt imports preserve contiguous immutable
versions. Explicit cursor imports preserve a nonempty initial map in a newly
provisioned scope; existing cursor state is never overwritten by migration.
No file discovery or automatic empty-on-error import is performed.

`bind_legacy_merge_recovery_service` validates the complete existing recovery
schema. Binding and every typed call use the original owner connection and
transaction lock, with no DDL, reopen, replay, or second writer. Missing or
changed tables, keys, scope content, cursor content, or receipt history deny the
operation. The borrowed transaction checks owner, session, peer, grant and TTL
at the commit boundary and freezes the shared owner on an uncertain transaction.

The owner issues two separately scoped grants to the exact native consumer:

| Client | Exact entity scopes |
| --- | --- |
| `OwnerMergeQueueClient` | repository_id, target_branch, consumer_id |
| `OwnerRecoveryRuntimeClient` | repository_id, target_branch, consumer_id, recovery_scope_cid |

They must use distinct already-attached typed connections with identical current
owner identity and consumer. The queue grant is not widened to include recovery.
`OwnerMergeRecoveryRuntime` validates the pair and the admitted namespace. It
does not issue credentials or install schema.

The generic gateway retains its one-use database-task bootstrap/session rules.
Recovery uses its own separately scoped grant; a task-only session cannot reuse
that scope or publish recovery state. The recovery dispatch revalidates the
current session after acquiring the transaction lock. Repeated recovery calls
do not consume or replace task grants, and revoked or detached recovery sessions
cannot replay a previously committed operation. A newly admitted recovery client
can read the preserved state and replay only the exact durable operation.

## Durable state and uncertainty

Generic main's eight cursor stages are priority tasks, completed requests,
false-completed requests, false-pending requests, false-processing requests,
pending requests, quarantined requests and processing requests. The earlier
native source candidate admits five stages and has no false-completion scan
cursors. Its five-stage cursor payload is incompatible with
this generic profile: provisioning, imports and reads require the complete exact
eight-stage map. They neither append an empty stage nor reset preserved progress.
Qualifying a different profile remains a separate explicit migration decision;
this source port does not perform a live migration.

Reads return the exact owner-held
revision and content identity. A save uses both as CAS preconditions and appends
immutable history. Stale CAS raises a conflict; reconstruction reads the current
head instead of resetting scan progress. Read failures propagate.

The consumer lease is canonical for the queue repository and target, across
lanes and scratch directories. It binds its acquiring owner generation, kernel
peer birth, scope, consumer, lease ID and fence. Expiry is diagnostic and cannot
authorize takeover. Normal synchronous exit explicitly releases the same lease.
An exception, including a nested callback whose exception was caught by its
caller, retains custody. Another thread cannot publish using that held lease.
A new client or owner cannot release a retained prior-generation lease. Exact
callback settlement and native lifecycle release remain separate prerequisites.

The same runtime retains the exact operation ID if an acquire reply is lost
before callback entry. It can replay that operation while the owner still
admits the exact lease, without entering the callback twice. Likewise, after a
callback returns normally, a lost release reply leaves an exact pending release
that must be replayed before another callback. An unknown callback does not
enter either recovery route. Restarted runtimes do not reconstruct these local
knowledge states from timeouts. A changed owner/peer or replaced lease denies
replay and still needs native custody reconciliation; no generic automatic
lease recovery is claimed for those cases.

Receipts are append-only versions under target-wide logical keys with a CAS head.
They preserve complete JSON values, including floats, through bounded JSON text
inside the closed typed envelope. Publication requires the exact current consumer
lease; earlier versions remain readable. Missing receipt heads differ from
unavailable or corrupt evidence. A receipt is not semantic acceptance merely
because it was durably stored.

The train creates only explicitly admitted local Git worktrees and recomputable
validation/proof caches beneath `attempt_root/merge-train-local-state`. Existing
gate validation remains in force. It creates no fallback `.merge-queue`, local
consumer lock, train receipt directory, cursor JSON, or authoritative publication
ledger. Acceptance receipt payloads and the publication ledger use owner-backed
receipt versions as well.

## Qualification limits

Disposable tests use actual DuckDB files and authenticated typed Unix sockets;
the owner fixture is not a full live Quack extension deployment. They exercise
real queue claims and Git integration, reconstructed cursor and receipt reads,
owner restart/generation fencing, wrong scope/grant/owner/session denials,
commit-time revocation, unknown callback retention and missing-pair refusal.
The canonical task fixture remains unchanged by a completed disposable queue
merge. No native SPAR store or credentials are read by these tests.

The native source validator, offline migration admission, peer-bound bootstrap
handoff and real Quack lifecycle qualification still precede deployment. SPAR's
independent datasets, current forest, required-mode, capstone, fixed-point and
goal/closeout acceptance gates remain unchanged. Fifty-one task receipts do not
settle the remaining goals or the legacy queue.

## Preserved train import coverage

`merge.legacy_train_imports.validate_train_import_coverage` checks the canonical
file-to-owner import coverage of a separately validated offline manifest. Native
migration callers still verify coherent capture, hashes, revision history and
scopes. The existing `train/distributed-publications.json` must map explicitly
to the owner's `distributed-publications` receipt key and remain its current
head. A copied but unimported ledger would otherwise make the migrated train
read an empty publication/fence history. The checker does not discover files,
infer unknown receipt keys, open a database, or authorize migration or completion.
