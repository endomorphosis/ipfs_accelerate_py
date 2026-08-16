# Agent Supervisor DuckDB + Quack Control-Plane Plan

Status: bootstrap implementation program, 2026-08-08  
Target branch: `agent/duckdb-quack-control-plane-20260808`  
Program namespace: `agent-supervisor-duckdb-quack-control-plane-v1`

## Executive decision

The target is a **versioned DuckDB control plane exposed through the Quack
remote protocol**. DuckDB owns the relational state model and transactions;
Quack funnels concurrent daemon and worktree clients through one long-lived
state-owner process. Quack is transport, not a scheduler, schema manager,
event bus, AST parser, or export format.

Markdown, JSON, JSONL, PID files, lock files, and log files cease to be
authoritative supervisor state. They remain only for four bounded roles:

1. this one-time bootstrap program, because the current configured-board
   launcher still selects the legacy Markdown task source;
2. deterministic import of existing state with provenance and reconciliation;
3. explicit, reproducible human or tool exports from a database snapshot; and
4. minimal process-bootstrap handles where the operating system requires a
   path, always treated as a non-authoritative projection of a database lease
   and process-birth identity.

Source code, Git objects, and worktree files remain files. The database stores
their identities, parsed structure, deltas, relationships, observations, and
mutation history; it does not replace Git or the filesystem.

## Why this migration is needed

The current framework has strong components but fragmented authority. It has a
transactional DuckDB task projection, several other DuckDB stores, formal plan
records, AST adapters, leases, recovery logic, and multi-lane runners. At the
same time, scheduler policy, objectives, task status, heartbeats, current
phases, events, retries, merge state, analysis products, and operator reports
are distributed across Markdown, JSON, JSONL, SQLite/DuckDB files, directories,
and process-local state. Multiple daemons therefore spend tokens and time
rediscovering, reparsing, reconciling, and narrating the same facts.

The migration makes one schema-controlled store the authority, makes every
transition queryable, gives parallel daemons transactional claims and fencing,
and lets symbolic planning consume exact AST and mutation deltas instead of
reconstructing the repository from prose on every attempt.

## Quack constraints that shape the design

The implementation must encode these constraints rather than hide them:

- Quack is beta/experimental in DuckDB 1.5.x. Protocol names and defaults may
  change before DuckDB 2.0. Server and clients therefore use an identical,
  pinned, attested DuckDB/Quack build behind an internal adapter.
- Quack exposes the full SQL surface visible to the server session. Its default
  authorization callback permits every authenticated query. LLM-generated code
  and implementation-provider subprocesses never receive a Quack token or
  arbitrary-SQL capability.
- The server binds to loopback. Any future remote deployment requires explicit
  review, TLS termination, OS isolation, credential rotation, and a real parsed
  statement authorization policy. A prefix regex is not a security boundary.
- Quack is client-driven request/response and has no server push. Workers use
  bounded polling with cursors, change sequences, and jitter; correctness never
  depends on filesystem notifications.
- DuckDB uses optimistic concurrency. Claims and state changes are short
  transactions with compare-and-swap revisions, idempotency keys, bounded
  conflict retries, and monotonic fencing epochs.
- One Quack server is one failure domain. It is supervised independently,
  checkpointed, backed up, restored in tests, and never shares a read-write
  database file with another process.

## Target topology

```text
operator CLI / MCP / supervisor control service
                 |
                 v
       typed state repository interfaces
                 |
        Quack client adapter (SQL templates,
        bounds, retries, redaction, no raw LLM SQL)
                 |
        quack:127.0.0.1:<allocated-port>
                 |
      supervised DuckDB state-owner process
                 |
      one repository-scoped control.duckdb
                 |
   migrations + normalized tables + projections

parallel master supervisors ----+
implementation daemons ---------+-- use the same typed client
worktree scanners ---------------+
planner / doctor / repair -------+
export jobs ---------------------+
```

The state-owner lives outside any implementation worktree under the shared Git
common-state root. Its identity includes repository identity, database UUID,
schema revision, server process birth identity, Quack extension fingerprint,
listen URI, credential generation, and startup epoch. Worktrees store no
private writable copy of authoritative state.

## Authority model and invariants

1. **One writer path.** Only the state-owner opens `control.duckdb` read-write.
   Every other process uses Quack. Offline recovery opens it only while a
   proved-exclusive maintenance lease is active and Quack is stopped.
2. **Database-first transitions.** A task claim, phase change, validation,
   mutation, merge, completion, retry, or cancellation is authoritative only
   after its transaction commits.
3. **Append plus projection.** Immutable domain events retain what happened;
   normalized current-state tables are transactionally updated projections.
   Rebuilding projections from admitted events is tested.
4. **Stable identity.** Repository, worktree, daemon, goal, task, attempt,
   symbol, AST, mutation, artifact, prompt, provider call, and receipt IDs are
   canonical and content- or namespace-bound. Display aliases are not keys.
5. **Fenced ownership.** A lease has owner session, scope, expiry, revision,
   and fencing epoch. A stale daemon cannot update a task, worktree, or merge
   record even if its numeric PID is reused.
6. **Schema authority.** Ordered migrations have identifiers, checksums,
   minimum/maximum application versions, preconditions, postconditions, and
   recorded receipts. Runtime modules do not independently invent tables.
7. **No unearned completion.** Task and goal completion require current-tree
   validation/evidence rows; an exported status string is never completion
   authority.
8. **Deterministic exports.** Each export binds database UUID, schema revision,
   transaction watermark, query/view revision, renderer revision, parameters,
   artifact digest, and destination. Re-exporting the same snapshot is byte
   stable.
9. **Secrets stay opaque.** Credentials are sourced from a protected secret
   handle, never written to task prompts, events, exports, process arguments,
   or implementation worktrees.
10. **Unknown fails closed.** Schema drift, server identity mismatch, expired
    leases, unsupported DuckDB/Quack versions, parse uncertainty, incomplete
    impact closure, and stale evidence block mutation rather than trigger a
    speculative LLM retry.

## Canonical schema domains

All tables carry explicit types, timestamps in UTC, revision/epoch fields where
mutable, and provenance. Flexible JSON is allowed only for bounded extension
payloads with a registered payload schema; it is not a substitute for columns
used in invariants, joins, claims, retention, or authorization.

### 1. Schema and deployment

- `control_plane_metadata`, `schema_migrations`, `schema_contracts`
- `state_servers`, `server_epochs`, `client_sessions`, `capability_snapshots`
- `credentials`, `authorization_roles`, `authorization_grants`
- `backup_snapshots`, `restore_receipts`, `maintenance_leases`

### 2. Repository and worktree forest

- `repositories`, `repository_revisions`, `submodule_edges`
- `worktrees`, `worktree_snapshots`, `worktree_paths`, `dirty_overlays`
- `branches`, `git_refs`, `merge_bases`, `merge_queue_entries`
- `resource_claims`, `path_claims`, `leases`, `lease_events`

Git remains the byte authority. These rows bind exact Git object IDs, index and
overlay digests, nested repository identities, and scanner versions.

### 3. Objectives, plans, and tasks

- `objectives`, `objective_revisions`, `goals`, `goal_edges`
- `plans`, `plan_revisions`, `planning_decisions`, `plan_candidates`
- `tasks`, `task_revisions`, `task_dependencies`, `task_outputs`
- `task_acceptance`, `task_validations`, `task_assignments`, `task_blocks`
- `refill_epochs`, `findings`, `finding_dispositions`

The existing DuckDB task source and formal-plan tables are migrated or wrapped;
parallel competing task schemas are reconciled before cutover.

### 4. Execution and lifecycle

- `supervisor_instances`, `daemon_instances`, `daemon_sessions`
- `heartbeats`, `health_samples`, `stall_detections`, `restart_decisions`
- `task_attempts`, `attempt_phases`, `task_claims`, `provider_invocations`
- `validation_runs`, `validation_results`, `merge_attempts`, `recovery_actions`
- `idempotency_records`, `effect_claims`, `completion_receipts`

Heartbeats are appended at a bounded cadence and compacted by retention policy;
the current lease/session row is updated transactionally. PID/status files may
mirror these rows for legacy tooling but do not grant authority.

### 5. Events, logs, and metrics

- `domain_events` with monotonic per-stream sequence and global event ID
- `structured_logs` with severity, component, trace/span, task/attempt/session
- `metrics`, `metric_samples`, `budget_reservations`, `budget_consumption`
- `quack_query_telemetry` for sampled transport diagnostics

Application events are explicit inserts. Quack's own in-memory/file diagnostic
log is not the authoritative audit ledger and recursive audit logging is
bounded by policy.

### 6. Code intelligence and mutations

- `source_snapshots`, `source_files`, `file_versions`, `parse_runs`
- `symbols`, `symbol_versions`, `ast_nodes`, `ast_edges`
- `imports`, `calls`, `references`, `definitions`, `type_relations`
- `mutations`, `mutation_files`, `mutation_hunks`, `ast_mutations`
- `impact_edges`, `impact_closures`, `repair_candidates`, `repair_applications`
- `proof_obligations`, `proof_attempts`, `counterexamples`, `evidence_nodes`

AST identity is scoped by repository, tree/overlay digest, path, language,
parser identity, and node path/fingerprint. Incremental scans parse changed
files and invalidate dependent closures. Every accepted code mutation binds a
before snapshot, after snapshot, textual hunks, AST edit script, symbol impact
closure, task/attempt, plan step, validation, and merge outcome. Generated or
unsupported syntax is recorded as an explicit incomplete frontier.

### 7. Context and LLM-economy records

- `context_manifests`, `context_members`, `context_deltas`
- `prompt_templates`, `prompt_instances`, `prompt_inputs`
- `provider_calls`, `provider_responses`, `failure_signatures`
- `decision_cache_entries`, `replay_suppressions`, `churn_metrics`

Raw secrets and unrestricted source dumps are excluded. Large immutable bodies
may live in the existing content-addressed artifact store; the database retains
typed metadata, digests, provenance, retention, and references.

## LLM churn reduction strategy

The database enables reduction only when callers use it deliberately:

- Build task packets from `ready_task_context_v*` views containing the task,
  unfulfilled dependencies, current worktree delta, impacted symbols, open
  proof obligations, last distinct failure, and exact validation commands.
- Key context by canonical manifest CID. If repository/schema/policy/task state
  is unchanged, reuse the prior deterministic analysis and suppress identical
  provider prompts.
- Store normalized failure signatures and mutation outcomes. Never pay for the
  same unsuccessful proposal against the same inputs after its retry policy is
  exhausted.
- Query dependency and impact deltas rather than reparsing full Markdown plans
  or resending the full repository narrative.
- Separate deterministic discovery, AST analysis, contract lookup, and proof
  cache hits from the model budget. The LLM receives only unresolved choices.
- Track input/output tokens, wall time, duplicate context fraction, unchanged
  reprompt rate, cache hit rate, context bytes per accepted mutation, and
  provider calls per completed task. Promotion requires non-inferior quality
  and safety, not token reduction alone.

## Import and export boundary

### Import

Importers read legacy Markdown/JSON/JSONL/SQLite/DuckDB artifacts under an
explicit manifest. Each source gets a byte digest, parser/schema identity,
path, timestamp observation, record counts, rejected rows, and reconciliation
decision. Import is idempotent and defaults to preview. Conflicting authorities
are never silently last-write-wins; an operator or deterministic policy must
select, merge, quarantine, or reject them.

### Export

Export is a read-only rendering job over versioned database views. Supported
profiles include a human taskboard/objective report in Markdown, bounded status
and event JSON, JSONL audit stream, CSV/Parquet analysis extracts, and a
portable release bundle. Export destinations are never watched as input unless
an explicit later import is requested. A round-trip test distinguishes lossless
portable exports from intentionally lossy human reports.

## Compatibility and rollout

The cutover is staged because Quack is beta and existing supervisors are
already active:

1. **Bootstrap:** this protected Markdown board launches the implementation.
2. **Foundation:** central schema, migrations, Quack capability adapter, server,
   typed client, transaction/CAS primitives, and security boundary.
3. **Shadow import:** ingest existing files and compare database projections
   without changing scheduler decisions.
4. **Dual observation:** write authoritative candidate events to DuckDB while
   continuing legacy projections; compare every read and lifecycle decision.
5. **Database authority canary:** one isolated program uses DuckDB/Quack for
   goals, tasks, claims, lifecycle, logs, worktrees, AST, and mutations. Files
   are exports only.
6. **Default cutover:** configured-board and daemon entrypoints accept a
   database target and Quack endpoint natively. New programs do not require
   control Markdown/JSON.
7. **Legacy retirement:** remove implicit file reads and dual writes only after
   recovery, chaos, concurrency, security, and quality/churn gates pass.

Every stage has a kill switch back to the last proved mode. Rollback does not
rewrite or discard database history; it changes the authority/read route and
records a rollback receipt.

## Testing and release gates

- migration apply/replay/checksum/drift/rollback and clean-database equivalence;
- identical pinned DuckDB and Quack extension fingerprints on server/clients;
- concurrent append, claim, renewal, expiry, fencing, conflict retry, and
  idempotency tests across real processes;
- crash at every claim/event/projection/checkpoint boundary and verified restore;
- token redaction, unauthorized query, raw-SQL escape, credential rotation,
  loopback binding, and provider-subprocess isolation tests;
- multi-daemon/multi-worktree scheduling, merge serialization, stale-owner
  recovery, and nested repository forest tests;
- incremental AST/full-rescan equivalence, rename/delete/generated syntax,
  mutation lineage, impact completeness, and proof invalidation tests;
- deterministic Markdown/JSON export, snapshot binding, round trip, and
  exported-file non-authority tests;
- old/new shadow decision equivalence, then database-only end-to-end tests;
- baseline versus candidate provider calls, tokens, context bytes, latency,
  accepted mutation quality, rollback rate, and safety violations;
- backup/restore and upgrade rehearsal from the pinned 1.5.x profile to a
  separately tested future DuckDB/Quack profile.

Release safety floors are zero for duplicate non-idempotent effects, stale
lease writes, unauthorized SQL, secret leakage, false completion, missing
impact frontier admission, AST/mutation misbinding, event/projection divergence,
and accepted-state loss in the declared crash model.

## Operational anti-stall design

- The Quack state-owner has an independent watchdog, readiness query, process
  birth identity, heartbeat, restart budget, and restore-only failure mode.
- Supervisors publish session and task progress in the database. A stall
  detector reasons over lease freshness, active worker birth identity, phase
  deadlines, log/metric progress, task eligibility, provider capacity, and
  merge/recovery state rather than file mtime alone.
- `no_shard_selectable_ready_tasks` is quiescence when other shards own all
  ready work; it is not automatically a failure. Provider-capacity backoff is
  a typed temporary state. A ready unclaimed task with no valid capacity or
  dependency explanation is actionable.
- Recovery uses fenced commands and idempotency keys. Operators never delete a
  lock or signal a numeric PID based only on age.
- Bounded diagnostic views expose live sessions, expiring leases, ready versus
  claimed tasks, stuck phases, conflict retries, failed migrations, server
  identity, backup age, and projection lag.

## Deliverables and sequencing

The companion objective heap and taskboard decompose this plan into 39
implementation tasks plus the completed control seal. Four strict shards start
with five file-disjoint foundation tasks. Dependencies serialize shared
cutovers while allowing the schema, Quack, importer, lifecycle, AST, context,
security, and verification streams to proceed in parallel.

The terminal release task may complete only after all implementation tasks are
terminal, the legacy files are demonstrably exports rather than authority, a
real multi-process Quack canary completes work across isolated worktrees, and
the quality/safety/churn report passes without suppressing failures or weakening
evidence.

## Out of scope

- replacing Git or source files with database blobs;
- exposing Quack directly to the public internet;
- treating vector similarity, an AST parser, a model response, or a passing
  ordinary test as semantic/proof authority;
- deleting legacy artifacts before import reconciliation and retention review;
- promising high availability from a single Quack process; and
- upgrading automatically to DuckDB 2.0 without a pinned compatibility and
  restore rehearsal.
