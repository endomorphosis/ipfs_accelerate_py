# Agent Supervisor DuckDB + Quack Control-Plane Objectives

This objective heap is the bootstrap intent projection for program
`agent-supervisor-duckdb-quack-control-plane-v1`. The database becomes the
authority after the canary cutover; this file then becomes a deterministic
human export.

## DQP-G000 Make DuckDB plus Quack the authoritative supervisor control plane

- Status: blocked
- Review only: true
- Parent:
- Depends on:
- Fib priority: 1
- Track: control-plane-migration
- Priority: P0
- Bundle: agent-supervisor/duckdb-quack/root
- Parallel lane: control
- Resource class: cpu-large
- Goal: Replace Markdown, JSON, JSONL, PID/status files, and independent local DuckDB files as orchestration authority with one versioned DuckDB control plane accessed by parallel daemons through Quack, while retaining deterministic imports and exports.
- Subgoals: DQP-G010, DQP-G020, DQP-G030, DQP-G040, DQP-G050, DQP-G060, DQP-G070, DQP-G080, DQP-G090
- Evidence: DQP-G010, DQP-G020, DQP-G030, DQP-G040, DQP-G050, DQP-G060, DQP-G070, DQP-G080, DQP-G090, DQP-039
- Evidence criteria: Every child goal has current-tree tests and receipts; the terminal canary completes real tasks across four isolated lanes through Quack; legacy files are proved exports rather than decision authority.
- Evidence source policy: Committed schema contracts, transactional state, current Git identities, process-birth-bound leases, reconstructed AST/mutation lineage, and current validation receipts are authoritative; prose, exports, caches, vector ranks, and model responses are not.
- Outputs: docs/architecture/AGENT_SUPERVISOR_DUCKDB_QUACK_CONTROL_PLANE_PLAN.md, docs/architecture/agent_supervisor_duckdb_quack_control_plane.objectives.md, docs/architecture/agent_supervisor_duckdb_quack_control_plane.todo.md, config/agent_supervisor_duckdb_quack_control_plane_scheduler.json
- Predicted files: ipfs_accelerate_py/agent_supervisor, scripts/ops/agent_supervisor, test/api, docs/architecture, docs/guides
- Interfaces: AgentSupervisorDuckDBControlPlane@1, QuackStateService@1, DuckDBControlPlaneReleaseReceipt@1
- Validation: python scripts/validate_agent_supervisor_duckdb_quack_control_plane_board.py --check-all && python -m pytest -q test/api/test_agent_supervisor_duckdb_quack_control_plane_board.py
- Acceptance: DQP-001 through DQP-039 are completed; schema, Quack transport, goals/tasks, lifecycle, worktrees, events, AST/mutations, context, exports, recovery, and cutover gates pass without unauthorized SQL, stale-owner writes, false completion, accepted-state loss, or quality regression.
- Gap task: Aggregate child evidence and decide release; root review does not implement subsystem behavior.
- Refinement: A single local DuckDB file with many file locks is not the target; all parallel writers must use the state-owner through Quack after cutover.
- Embedding query: duckdb quack agent supervisor control plane orchestration state migration release
- AST query: DuckDBTaskSource LeaseCoordinator MergeQueue PortalImplementationSupervisor QuackStateService
- Conflict policy: Child goals own implementations; DQP-039 alone owns the joined release receipt.

## DQP-G010 Establish one controlled relational schema and Quack boundary

- Status: active
- Review only: false
- Parent: DQP-G000
- Depends on:
- Fib priority: 1
- Track: foundation
- Priority: P0
- Bundle: agent-supervisor/duckdb-quack/foundation
- Parallel lane: dqp-foundation
- Resource class: cpu-medium
- Goal: Inventory existing authorities, define normalized identities and invariants, install checksum-bound migrations, pin/probe DuckDB and Quack, and expose safe typed transaction primitives.
- Evidence: DQP-001, DQP-002, DQP-003, DQP-004, DQP-005, DQP-006, DQP-007, DQP-008, DQP-009
- Evidence criteria: Empty-to-latest and upgrade migrations reproduce the same schema; clients cannot bypass the typed Quack adapter; conflicting transactions retry or fail with typed outcomes; server/client fingerprints match.
- Evidence source policy: Migration checksums, catalog introspection, real multi-process transactions, and capability probes are primary; installed import success is not a capability signal.
- Outputs: ipfs_accelerate_py/agent_supervisor/task_sources, ipfs_accelerate_py/agent_supervisor/runtime, test/api
- Predicted files: ipfs_accelerate_py/agent_supervisor/task_sources/duckdb_state.py, ipfs_accelerate_py/agent_supervisor/task_sources/control_plane_schema.py, ipfs_accelerate_py/agent_supervisor/runtime/quack_state_server.py
- Interfaces: ControlPlaneSchema@1, SchemaMigration@1, QuackCapabilityReport@1, StateRepository@1
- Validation: python -m pytest -q test/api -k 'duckdb and (schema or quack or transaction)'
- Acceptance: One migration catalog owns schema creation; Quack is loopback-only and version-attested; implementation providers receive no credential; CAS, idempotency, fencing, retry, and health contracts pass under independent processes.
- Gap task: Generalize existing DuckDBTaskSource, LeaseCoordinator, MergeQueue, and duckdb_state primitives without duplicating their proven identities.
- Refinement: Quack supplies remote SQL transport, not schema or application authorization.
- Embedding query: duckdb schema migration quack server client transaction fencing idempotency
- AST query: DuckDBConnection DuckDBTaskSource LeaseCoordinator MergeQueue
- Conflict policy: Contracts and migrations land before server/client and repository cutovers.

## DQP-G020 Move intent, planning, tasks, events, and exports into database authority

- Status: blocked
- Review only: false
- Parent: DQP-G000
- Depends on: DQP-G010
- Fib priority: 2
- Track: intent-state
- Priority: P0
- Bundle: agent-supervisor/duckdb-quack/intent
- Parallel lane: dqp-intent
- Resource class: cpu-large
- Goal: Import and normalize objectives, goals, plans, tasks, queues, events, logs, metrics, receipts, and artifacts; make Markdown/JSON/JSONL deterministic export adapters only.
- Evidence: DQP-010, DQP-011, DQP-012, DQP-013, DQP-025
- Evidence criteria: Source manifests reconcile all legacy records; database transactions atomically advance intent and event projections; snapshot-bound exports are reproducible and cannot mutate state.
- Evidence source policy: Versioned database rows and domain events are authoritative; imported sources retain provenance but no continuing write authority.
- Outputs: ipfs_accelerate_py/agent_supervisor/task_sources, ipfs_accelerate_py/agent_supervisor/objectives, ipfs_accelerate_py/agent_supervisor/planning, ipfs_accelerate_py/agent_supervisor/runtime
- Predicted files: ipfs_accelerate_py/agent_supervisor/task_sources/legacy_state_import.py, ipfs_accelerate_py/agent_supervisor/task_sources/state_export.py, ipfs_accelerate_py/agent_supervisor/task_sources/intent_repository.py, ipfs_accelerate_py/agent_supervisor/runtime/database_event_log.py
- Interfaces: LegacyStateImport@1, IntentRepository@1, DatabaseEventLog@1, StateExportReceipt@1
- Validation: python -m pytest -q test/api -k 'task_source or objective or plan_revision or event_log or export'
- Acceptance: Goals, tasks, plan revisions, queue state, events, audit, metrics, and artifact metadata can run without reading a control Markdown/JSON/JSONL file; lossless exports round-trip and human exports declare intentional loss.
- Gap task: Invert existing paired file/DuckDB stores rather than add a third projection.
- Refinement: JSON payload columns are allowed only for registered bounded extensions, not joinable authority hidden in blobs.
- Embedding query: objective plan task event log import export duckdb authority
- AST query: PlanRevisionStore TaskboardStore PersistentTaskQueue RotatingEventLog ArtifactStore
- Conflict policy: Import/export files are disjoint; the intent repository precedes entrypoint cutover.

## DQP-G030 Coordinate daemons, claims, worktrees, validation, merge, and recovery

- Status: blocked
- Review only: false
- Parent: DQP-G000
- Depends on: DQP-G010, DQP-G020
- Fib priority: 1
- Track: runtime-coordination
- Priority: P0
- Bundle: agent-supervisor/duckdb-quack/runtime
- Parallel lane: dqp-runtime
- Resource class: cpu-io-large
- Goal: Give every master, lane, daemon, worker, task attempt, lease, worktree, validation, merge, and recovery action a transactionally fenced database identity shared across parallel processes.
- Evidence: DQP-014, DQP-015, DQP-016, DQP-017, DQP-018, DQP-019, DQP-029, DQP-030
- Evidence criteria: Four processes claim distinct ready work; stale sessions cannot write; worktree and merge ownership survives restart; watchdog decisions are derived from database facts and exact process-birth identity.
- Evidence source policy: Database leases plus current OS/Git observations establish ownership; PID/status/lock files are compatibility projections and never grant authority.
- Outputs: ipfs_accelerate_py/agent_supervisor/runtime, ipfs_accelerate_py/agent_supervisor/merge, ipfs_accelerate_py/agent_supervisor/todo_daemon, ipfs_accelerate_py/agent_supervisor/control
- Predicted files: ipfs_accelerate_py/agent_supervisor/runtime/daemon_registry.py, ipfs_accelerate_py/agent_supervisor/merge/lease_coordination.py, ipfs_accelerate_py/agent_supervisor/merge/worktree_lifecycle.py, ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_supervisor.py
- Interfaces: DaemonSession@1, FencedTaskClaim@1, WorktreeSnapshot@1, DatabaseWatchdog@1
- Validation: python -m pytest -q test/api -k 'lease or worktree or implementation_daemon or supervisor or merge_queue'
- Acceptance: Lifecycle entrypoints propagate the DuckDB/Quack task source end to end; no local file lock arbitrates semantic ownership; status/health/log/stop are available for configured database programs.
- Gap task: Consolidate existing strong but separate lease, merge, worktree, status, and recovery algorithms around canonical IDs.
- Refinement: A fresh status-file mtime is not proof of a healthy worker, and an old one is not by itself a stall.
- Embedding query: daemon heartbeat lease fence worktree merge validation recovery quack
- AST query: PortalImplementationSupervisor ImplementationDaemon LeaseCoordinator WorktreeLifecycleStore SupervisorLoop
- Conflict policy: Registry/claims/worktrees precede runner and daemon cutovers; merge/recovery integration is serialized.

## DQP-G040 Track code snapshots, ASTs, mutations, impact, and proof lineage

- Status: blocked
- Review only: false
- Parent: DQP-G000
- Depends on: DQP-G010, DQP-G030
- Fib priority: 2
- Track: code-intelligence
- Priority: P0
- Bundle: agent-supervisor/duckdb-quack/code-intelligence
- Parallel lane: dqp-code
- Resource class: cpu-large
- Goal: Automatically index every relevant repository/worktree snapshot and bind admitted mutations to before/after blobs, AST edits, symbols, impacted consumers, proof obligations, validations, and merge outcomes.
- Evidence: DQP-020, DQP-021, DQP-022, DQP-023, DQP-024
- Evidence criteria: Incremental and clean scans are equivalent; rename/delete/generated/unsupported cases are explicit; every accepted mutation has complete lineage or an admitted abstention frontier.
- Evidence source policy: Git bytes plus parser identity and independently reconstructed proof/counterexample evidence establish facts; ASTs and graph retrieval nominate relationships but do not authorize semantics alone.
- Outputs: ipfs_accelerate_py/agent_supervisor/analysis, ipfs_accelerate_py/agent_supervisor/planning, ipfs_accelerate_py/agent_supervisor/proof
- Predicted files: ipfs_accelerate_py/agent_supervisor/analysis/duckdb_ast_index.py, ipfs_accelerate_py/agent_supervisor/analysis/mutation_ledger.py, ipfs_accelerate_py/agent_supervisor/analysis/impact_repository.py
- Interfaces: ASTIndex@1, MutationLedger@1, ImpactClosure@1, RepairLineage@1
- Validation: python -m pytest -q test/api -k 'ast or mutation or impact or symbolic or proof'
- Acceptance: Parallel worktree scanners share immutable parse results, invalidate dependent closures, and make symbolic planning/repair query exact deltas without reparsing prose or the full tree.
- Gap task: Persist and connect existing analysis_ast_index, repository_indexer, semantic event, repair, and proof structures.
- Refinement: Unsupported dynamic code blocks automatic mutation; it is never silently omitted from impact closure.
- Embedding query: git snapshot ast mutation impact closure symbolic repair proof duckdb
- AST query: AnalysisASTIndex RepositoryIndexer ProgramRepairSynthesis ProofObligation
- Conflict policy: Snapshot/AST and mutation ledgers are file-disjoint; impact and repair integration follow both.

## DQP-G050 Reduce LLM churn with query-built context and durable call memory

- Status: blocked
- Review only: false
- Parent: DQP-G000
- Depends on: DQP-G020, DQP-G040
- Fib priority: 2
- Track: context-economy
- Priority: P0
- Bundle: agent-supervisor/duckdb-quack/context
- Parallel lane: dqp-context
- Resource class: cpu-medium
- Goal: Build deterministic, delta-oriented task contexts and persist provider calls, failure signatures, replay suppression, cache dependencies, and token/quality metrics so unchanged work is not repeatedly narrated or prompted.
- Evidence: DQP-009, DQP-026, DQP-027, DQP-028, DQP-036
- Evidence criteria: Context identities are reproducible; identical unsuccessful prompts against unchanged state are suppressed; token/call reductions preserve or improve accepted-mutation quality and safety.
- Evidence source policy: Query manifests, exact cache keys, call receipts, validation outcomes, and current tree identity are authoritative; semantic similarity is nomination only.
- Outputs: ipfs_accelerate_py/agent_supervisor/context, ipfs_accelerate_py/agent_supervisor/runtime, ipfs_accelerate_py/agent_supervisor/prompt, ipfs_accelerate_py/agent_supervisor/analysis
- Predicted files: ipfs_accelerate_py/agent_supervisor/context/database_context.py, ipfs_accelerate_py/agent_supervisor/runtime/provider_call_ledger.py, ipfs_accelerate_py/agent_supervisor/prompt/delta_task_packet.py
- Interfaces: DatabaseContextManifest@1, ProviderCallLedger@1, DeltaTaskPacket@1, ChurnBenchmark@1
- Validation: python -m pytest -q test/api -k 'context or prompt or provider_usage or cache or churn'
- Acceptance: The canary uses bounded SQL views and context deltas; duplicate-context and unchanged-reprompt metrics fall from baseline without increased rollback, false completion, or missed impact rates.
- Gap task: Centralize existing analysis/proof caches and provider usage records without weakening their full key dimensions.
- Refinement: Fewer tokens are not success if quality, evidence, or safety is reduced.
- Embedding query: llm churn context delta prompt replay cache provider call tokens quality
- AST query: ContextCompiler AnalysisCache ProviderUsage PromptWorkflow
- Conflict policy: Baseline precedes context/call-ledger changes; packet cutover follows both.

## DQP-G060 Publish database-backed control APIs and retire implicit file authority

- Status: blocked
- Review only: false
- Parent: DQP-G000
- Depends on: DQP-G020, DQP-G030
- Fib priority: 3
- Track: control-api-cutover
- Priority: P0
- Bundle: agent-supervisor/duckdb-quack/control-api
- Parallel lane: dqp-control-api
- Resource class: cpu-medium
- Goal: Route Python, CLI, MCP, configured-board, and prompt-first lifecycle operations through the typed database repositories and make legacy inputs explicit import commands and outputs explicit export commands.
- Evidence: DQP-029, DQP-030, DQP-031
- Evidence criteria: Discovery remains side-effect free; database targets have Python/CLI/MCP parity; lifecycle mutations retain authorization, idempotency, lease, fence, and effect validation.
- Evidence source policy: The canonical SupervisorControlService request/result contracts remain authority; adapters cannot shell out or infer mutation permission from a path.
- Outputs: ipfs_accelerate_py/agent_supervisor/control, ipfs_accelerate_py/agent_supervisor/entrypoints, ipfs_accelerate_py/agent_supervisor/runtime, docs/guides
- Predicted files: ipfs_accelerate_py/agent_supervisor/control/control_plane.py, ipfs_accelerate_py/agent_supervisor/entrypoints/state_resolver.py, ipfs_accelerate_py/agent_supervisor/runtime/configured_board_scheduler.py
- Interfaces: DatabaseSupervisorBackend@1, DatabaseProgramTarget@1, DatabaseLifecycleCLI@1
- Validation: python -m pytest -q test/api -k 'control_plane or entrypoint or configured_board or lifecycle'
- Acceptance: A new goal/task program can be created, planned, started, observed, drained, exported, and resumed without a control Markdown/JSON file; compatibility mode is opt-in and labeled.
- Gap task: Extend, do not bypass, the existing stable control operation catalog and direct-service adapters.
- Refinement: Database access does not grant mutation authority; permits and fenced expected effects still apply.
- Embedding query: supervisor control api cli mcp database backend lifecycle cutover
- AST query: SupervisorControlService RepositorySupervisorBackend StateResolver ConfiguredBoard
- Conflict policy: Backend and lifecycle ops precede default switch and legacy deprecation.

## DQP-G070 Prove security, resilience, observability, and performance

- Status: blocked
- Review only: false
- Parent: DQP-G000
- Depends on: DQP-G010, DQP-G030
- Fib priority: 1
- Track: assurance
- Priority: P0
- Bundle: agent-supervisor/duckdb-quack/assurance
- Parallel lane: dqp-assurance
- Resource class: cpu-io-large
- Goal: Make the single state-owner operable through least privilege, watchdogs, diagnostic views, checkpoints/backups/restores, concurrency/chaos tests, and explicit beta-version upgrade gates.
- Evidence: DQP-007, DQP-032, DQP-033, DQP-034, DQP-035, DQP-036
- Evidence criteria: Unauthorized SQL and token inheritance are impossible in tests; crash/restore preserves admitted state; diagnostic views distinguish quiescence, capacity backoff, stale ownership, and actionable stalls.
- Evidence source policy: Real process/network-loopback tests, database checksums, restore comparison, and sampled measurements are primary; synthetic success or absent telemetry cannot promote.
- Outputs: ipfs_accelerate_py/agent_supervisor/runtime, ipfs_accelerate_py/agent_supervisor/rescue, scripts/ops/agent_supervisor, test/api, docs/guides
- Predicted files: ipfs_accelerate_py/agent_supervisor/runtime/database_watchdog.py, ipfs_accelerate_py/agent_supervisor/runtime/control_plane_backup.py, test/api/test_agent_supervisor_quack_chaos.py
- Interfaces: DatabaseWatchdog@1, ControlPlaneBackup@1, QuackChaosReport@1
- Validation: python -m pytest -q test/api -k 'quack or watchdog or backup or restore or concurrency'
- Acceptance: Declared crashes lose no accepted state, duplicate no non-idempotent effect, and permit no stale lease write; p95/p99 and conflict rates remain within reviewed bounds.
- Gap task: Replace file-mtime heuristics and independent logs with database-derived health while preserving exact process fencing.
- Refinement: One supervised server with restore is resilient but not highly available; documentation must say so.
- Embedding query: quack security watchdog backup restore chaos concurrency observability
- AST query: SupervisorWatchdog RecoveryDiagnostics QuackStateServer DuckDBConnection
- Conflict policy: Security, watchdog, and backup implementations are disjoint; chaos joins them.

## DQP-G080 Backfill, shadow, canary, and cut over safely

- Status: blocked
- Review only: false
- Parent: DQP-G000
- Depends on: DQP-G020, DQP-G030, DQP-G040, DQP-G050, DQP-G060, DQP-G070
- Fib priority: 1
- Track: rollout
- Priority: P0
- Bundle: agent-supervisor/duckdb-quack/rollout
- Parallel lane: dqp-rollout
- Resource class: cpu-large
- Goal: Reconcile legacy state into the new schema, compare old/new decisions, run a database-authoritative multi-worktree canary, and switch defaults only under current release evidence and an exact rollback route.
- Evidence: DQP-035, DQP-036, DQP-037, DQP-038
- Evidence criteria: Import counts/digests reconcile; shadow reads and lifecycle decisions match or have reviewed dispositions; canary reaches terminal state through Quack; rollback restores the prior route without history loss.
- Evidence source policy: Current-tree end-to-end receipts and independent state queries are primary; migration plans and dry runs alone cannot promote.
- Outputs: ipfs_accelerate_py/agent_supervisor/self_improvement, scripts/ops/agent_supervisor, test/api, docs/guides
- Predicted files: ipfs_accelerate_py/agent_supervisor/self_improvement/database_rollout.py, scripts/ops/agent_supervisor/duckdb_quack_control_plane.py, docs/guides/AGENT_SUPERVISOR_DUCKDB_QUACK_GUIDE.md
- Interfaces: DatabaseRolloutPolicy@1, ShadowParityReport@1, CanaryReceipt@1
- Validation: python -m pytest -q test/api -k 'rollout or backfill or canary or cutover'
- Acceptance: Database authority advances through off/observe/shadow/assist/canary/default stages with exact rollback; legacy files are read only by explicit importer tests.
- Gap task: Reuse existing staged-rollout and current-root evidence contracts instead of creating an ungoverned feature flag.
- Refinement: Dual write is temporary evidence collection, not a permanent two-authority architecture.
- Embedding query: duckdb quack backfill shadow parity canary cutover rollback
- AST query: SelfImprovementRollout RolloutDecision CurrentRootEvaluation
- Conflict policy: Backfill precedes shadow; canary follows all subsystem integrations; cutover is serialized.

## DQP-G090 Issue the joined release and compatibility decision

- Status: blocked
- Review only: true
- Parent: DQP-G000
- Depends on: DQP-G010, DQP-G020, DQP-G030, DQP-G040, DQP-G050, DQP-G060, DQP-G070, DQP-G080
- Fib priority: 1
- Track: release
- Priority: P0
- Bundle: agent-supervisor/duckdb-quack/release
- Parallel lane: dqp-release
- Resource class: cpu-large
- Goal: Join schema, transport, intent, runtime, code-intelligence, LLM-economy, security, recovery, export, canary, and rollback evidence into a content-bound release decision.
- Evidence: DQP-039
- Evidence criteria: The release verifier queries the authoritative database and current Git tree, rejects missing/stale/simulated/skipped evidence, and records DuckDB/Quack fingerprints and compatibility status.
- Evidence source policy: Independent joined verification is required; a completed taskboard or green narrow suite is insufficient.
- Outputs: ipfs_accelerate_py/agent_supervisor/validation/duckdb_quack_release.py, test/api/test_agent_supervisor_duckdb_quack_release.py, docs/architecture/AGENT_SUPERVISOR_DUCKDB_QUACK_RELEASE.md
- Predicted files: ipfs_accelerate_py/agent_supervisor/validation/duckdb_quack_release.py, test/api/test_agent_supervisor_duckdb_quack_release.py, docs/architecture/AGENT_SUPERVISOR_DUCKDB_QUACK_RELEASE.md
- Interfaces: DuckDBControlPlaneReleaseReceipt@1
- Validation: python -m pytest -q test/api/test_agent_supervisor_duckdb_quack_release.py
- Acceptance: DQP-039 verifies every declared gate, records any beta limitation without misrepresenting production readiness, and fails closed if the database is not sole decision authority for the canary.
- Gap task: Create one release decision, not a collection of optimistic component summaries.
- Refinement: Compatibility with future DuckDB 2.0 is unknown until separately tested; do not infer it from a 1.5.x pass.
- Embedding query: joined duckdb quack supervisor release receipt compatibility decision
- AST query: DuckDBControlPlaneReleaseVerifier ReleaseEvidence
- Conflict policy: This goal owns only terminal aggregation and documentation.
