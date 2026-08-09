# Agent Supervisor DuckDB + Quack Control-Plane Taskboard

Bootstrap task source for `agent-supervisor-duckdb-quack-control-plane-v1`
with task prefix `DQP-`.

This is intentionally the final file-authoritative program seed. The current
configured-board launcher does not yet propagate a DuckDB task source through
the implementation supervisor. DQP-017 and DQP-018 close that gap; DQP-038
runs the first database-authoritative Quack canary. Later Markdown taskboards
are explicit exports only.

Control artifacts are protected from implementation agents. The supervisor is
authorized to change only persisted `Status: todo` lines to `completed` while
this bootstrap is active.

## DQP-000 Seal the migration control program

- Status: completed
- Completion: manual
- Completion evidence: reviewed plan, objective heap, taskboard, scheduler, validator, parser test, branch isolation, and clean configured-board preflight
- Is schedulable: true
- Review only: true
- Priority: P0
- Track: control
- Depends on:
- Goal id: DQP-G000
- Outputs: .gitignore, docs/architecture/AGENT_SUPERVISOR_DUCKDB_QUACK_CONTROL_PLANE_PLAN.md, docs/architecture/agent_supervisor_duckdb_quack_control_plane.objectives.md, docs/architecture/agent_supervisor_duckdb_quack_control_plane.todo.md, config/agent_supervisor_duckdb_quack_control_plane_scheduler.json, scripts/validate_agent_supervisor_duckdb_quack_control_plane_board.py, test/api/test_agent_supervisor_duckdb_quack_control_plane_board.py
- Validation: python scripts/validate_agent_supervisor_duckdb_quack_control_plane_board.py --check-all && python -m pytest -q test/api/test_agent_supervisor_duckdb_quack_control_plane_board.py
- Board namespace: agent-supervisor-duckdb-quack-control-plane-v1
- Bundle: agent-supervisor/duckdb-quack/control
- Parallel lane: control
- Resource class: cpu-small
- Resource stage: planning
- Estimated tokens: 24000
- Implementation timeout seconds: 1800
- Predicted files: .gitignore, docs/architecture/AGENT_SUPERVISOR_DUCKDB_QUACK_CONTROL_PLANE_PLAN.md, docs/architecture/agent_supervisor_duckdb_quack_control_plane.objectives.md, docs/architecture/agent_supervisor_duckdb_quack_control_plane.todo.md, config/agent_supervisor_duckdb_quack_control_plane_scheduler.json, scripts/validate_agent_supervisor_duckdb_quack_control_plane_board.py, test/api/test_agent_supervisor_duckdb_quack_control_plane_board.py
- Interfaces: DuckDBQuackControlPlanePlan@1
- Allow concurrent with:
- Conflict policy: These control files are protected after launch; later tasks may not rewrite task identities, dependencies, or acceptance criteria.
- Preconditions: Dedicated branch and clean worktree exist; current main is an ancestor.
- Effects: A parseable, acyclic, four-shard self-improvement program is available.
- Evidence subset: architecture, goals, tasks, dependency DAG, ownership, launch policy, release gates
- Symbolic first: true
- LLM context budget bytes: 24000
- Acceptance: Production parsers consume exactly 40 unique tasks and 10 unique goals; DQP-001, DQP-002, DQP-003, DQP-004, and DQP-009 are initially ready and cover all four strict shards; control files are tracked and clean; preflight and parser test pass.
- Embedding query: seal duckdb quack supervisor migration goals tasks scheduler

## DQP-001 Inventory and classify every mutable supervisor state sink

- Status: completed
- Completion: automatic
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: state-inventory
- Depends on: DQP-000
- Goal id: DQP-G010
- Outputs: docs/architecture/AGENT_SUPERVISOR_STATE_SINK_INVENTORY.md, scripts/ops/agent_supervisor/inventory_state_sinks.py, test/api/test_agent_supervisor_state_sink_inventory.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_state_sink_inventory.py
- Board namespace: agent-supervisor-duckdb-quack-control-plane-v1
- Bundle: agent-supervisor/duckdb-quack/foundation/inventory
- Parallel lane: dqp-inventory
- Resource class: cpu-medium
- Resource stage: analysis
- Estimated tokens: 22000
- Implementation timeout seconds: 7200
- Predicted files: docs/architecture/AGENT_SUPERVISOR_STATE_SINK_INVENTORY.md, scripts/ops/agent_supervisor/inventory_state_sinks.py, test/api/test_agent_supervisor_state_sink_inventory.py
- Interfaces: SupervisorStateSinkInventory@1
- Allow concurrent with: DQP-002, DQP-003, DQP-004, DQP-009
- Conflict policy: Own only the new scanner, inventory, and test; inspect existing stores read-only.
- Preconditions: Current repository tree is bound and control artifacts are protected.
- Effects: Every MD/JSON/JSONL/PID/lock/SQLite/DuckDB/cache/artifact writer is classified as authority, static input, immutable evidence, cache, export, OS bootstrap, or emergency diagnostic with a destination domain and retirement stage.
- Evidence subset: source writers, defaults, CLI paths, daemon status, plan stores, objectives, taskboards, events, worktrees, caches, artifacts, proof stores
- Symbolic first: true
- LLM context budget bytes: 26000
- Acceptance: Scanner is deterministic and fails CI for an unclassified mutable sink; inventory includes direct DuckDB writers and cross-file atomicity gaps, records reuse candidates, and distinguishes Git/source bytes from supervisor state.
- Embedding query: agent supervisor markdown json jsonl duckdb state sink inventory authority

## DQP-002 Define canonical store, schema, identity, and authority contracts

- Status: completed
- Completion: automatic
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: state-contracts
- Depends on: DQP-000
- Goal id: DQP-G010
- Outputs: ipfs_accelerate_py/agent_supervisor/task_sources/control_plane_contracts.py, test/api/test_agent_supervisor_control_plane_contracts.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_control_plane_contracts.py
- Board namespace: agent-supervisor-duckdb-quack-control-plane-v1
- Bundle: agent-supervisor/duckdb-quack/foundation/contracts
- Parallel lane: dqp-contracts
- Resource class: cpu-medium
- Resource stage: analysis
- Estimated tokens: 22000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/agent_supervisor/task_sources/control_plane_contracts.py, test/api/test_agent_supervisor_control_plane_contracts.py
- Interfaces: ControlPlaneStoreIdentity@1, StoreGeneration@1, StateCommand@1, StateSnapshot@1, StateExportReceipt@1
- Allow concurrent with: DQP-001, DQP-003, DQP-004, DQP-009
- Conflict policy: Own a new provider-free contract module and focused test; do not open a database at import.
- Preconditions: Existing canonical content identity and control contracts are available for reuse.
- Effects: Closed records define database/store/generation/schema/session/command/revision/fence/snapshot/export identities, state authority classes, bounds, typed failures, and redaction.
- Evidence subset: canonical serialization, IDs, timestamps, revisions, fencing, idempotency, authority, compatibility, bounds
- Symbolic first: true
- LLM context budget bytes: 24000
- Acceptance: Contracts reject empty/forged/inconsistent IDs, non-finite bounds, generation/revision mismatch, secrets, mutable aliases as identity, and an export labeled authoritative; cold import performs no filesystem, database, network, provider, or process action.
- Embedding query: duckdb control plane store identity generation authority contract

## DQP-003 Build the checksum-bound migration catalog and runner

- Status: completed
- Completion: automatic
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: schema-migrations
- Depends on: DQP-000
- Goal id: DQP-G010
- Outputs: ipfs_accelerate_py/agent_supervisor/task_sources/control_plane_migrations.py, ipfs_accelerate_py/agent_supervisor/task_sources/sql/README.md, test/api/test_agent_supervisor_control_plane_migrations.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_control_plane_migrations.py
- Board namespace: agent-supervisor-duckdb-quack-control-plane-v1
- Bundle: agent-supervisor/duckdb-quack/foundation/migrations
- Parallel lane: dqp-migrations
- Resource class: cpu-medium
- Resource stage: implementation
- Estimated tokens: 24000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/agent_supervisor/task_sources/control_plane_migrations.py, ipfs_accelerate_py/agent_supervisor/task_sources/sql/README.md, test/api/test_agent_supervisor_control_plane_migrations.py
- Interfaces: ControlPlaneMigration@1, MigrationReceipt@1, MigrationCatalog@1
- Allow concurrent with: DQP-001, DQP-002, DQP-004, DQP-009
- Conflict policy: Own new migration infrastructure only; schema-domain SQL lands in later tasks through the catalog.
- Preconditions: DuckDB Python support is present for hermetic temporary-database tests.
- Effects: Ordered migrations are transactional, checksum-bound, dependency-aware, single-owner, replay-safe, inspectable, and able to prove empty-to-latest equivalence.
- Evidence subset: altered SQL, duplicate IDs, gaps, pre/postconditions, failure rollback, concurrent migration ownership, schema fingerprint
- Symbolic first: true
- LLM context budget bytes: 26000
- Acceptance: Runner records version, checksum, application/tool versions, start/end/outcome and schema fingerprint; refuses drift, gaps, downgrade, partial application, and runtime ad-hoc DDL outside an explicit compatibility path.
- Embedding query: duckdb ordered migration checksum transactional schema fingerprint

## DQP-004 Add a pinned DuckDB and Quack capability profile

- Status: completed
- Completion: automatic
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: quack-capabilities
- Depends on: DQP-000
- Goal id: DQP-G010
- Outputs: ipfs_accelerate_py/agent_supervisor/task_sources/quack_capabilities.py, docs/architecture/agent_supervisor/QUACK_COMPATIBILITY.md, test/api/test_agent_supervisor_quack_capabilities.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_quack_capabilities.py
- Board namespace: agent-supervisor-duckdb-quack-control-plane-v1
- Bundle: agent-supervisor/duckdb-quack/foundation/quack
- Parallel lane: dqp-quack
- Resource class: cpu-medium
- Resource stage: analysis
- Estimated tokens: 22000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/agent_supervisor/task_sources/quack_capabilities.py, docs/architecture/agent_supervisor/QUACK_COMPATIBILITY.md, test/api/test_agent_supervisor_quack_capabilities.py
- Interfaces: QuackCapabilityReport@1, QuackCompatibilityProfile@1
- Allow concurrent with: DQP-001, DQP-002, DQP-003, DQP-009
- Conflict policy: Own capability probing and compatibility documentation; dependency changes belong to DQP-005.
- Preconditions: Current environment may be DuckDB 1.5.2 and must produce a typed unsupported result rather than crash.
- Effects: Probe DuckDB version/platform, extension origin/version/signature, required Quack functions/settings, server/client compatibility, and upgrade/restart requirements without launching an uncontrolled service.
- Evidence subset: INSTALL/LOAD policy, quack_serve, quack_query, ATTACH, whoami, auth settings, logging, extension fingerprint
- Symbolic first: true
- LLM context budget bytes: 22000
- Acceptance: Probe distinguishes unavailable, unsupported, install-required, load-required, compatible, mismatched, and experimental; records Quack beta limitations; import success alone cannot pass; network install is never implicit in an ordinary health check.
- Embedding query: duckdb quack version extension capability compatibility beta

## DQP-005 Install the normalized control-plane schema

- Status: completed
- Completion: automatic
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: schema-domains
- Depends on: DQP-002, DQP-003, DQP-004
- Goal id: DQP-G010
- Outputs: ipfs_accelerate_py/agent_supervisor/task_sources/sql/0001_control_plane.sql, ipfs_accelerate_py/agent_supervisor/task_sources/control_plane_schema.py, test/api/test_agent_supervisor_control_plane_schema.py, pyproject.toml
- Validation: python -m pytest -q test/api/test_agent_supervisor_control_plane_schema.py test/api/test_agent_supervisor_control_plane_migrations.py
- Board namespace: agent-supervisor-duckdb-quack-control-plane-v1
- Bundle: agent-supervisor/duckdb-quack/foundation/schema
- Parallel lane: dqp-schema
- Resource class: cpu-large
- Resource stage: implementation
- Estimated tokens: 30000
- Implementation timeout seconds: 10800
- Predicted files: ipfs_accelerate_py/agent_supervisor/task_sources/sql/0001_control_plane.sql, ipfs_accelerate_py/agent_supervisor/task_sources/control_plane_schema.py, test/api/test_agent_supervisor_control_plane_schema.py, pyproject.toml
- Interfaces: ControlPlaneSchema@1
- Allow concurrent with: DQP-006, DQP-007, DQP-010
- Conflict policy: Own base schema and dependency profile; migration runner and capability API remain stable.
- Preconditions: Contracts, migration catalog, and compatible Quack dependency target are defined.
- Effects: One physical database gains normalized meta, intent, schedule, runtime, git, code, evidence, cache, control, and improve schemas plus constrained diagnostic/context views.
- Evidence subset: primary/foreign keys, unique edges, typed states, bounds, revisions, epochs, schema metadata, existing-table compatibility
- Symbolic first: true
- LLM context budget bytes: 32000
- Acceptance: Fresh and upgraded databases share an identical canonical information-schema fingerprint; schema preserves existing task CIDs and lease semantics; DuckDB/Quack profile is explicitly pinned for the optional supervisor service; no join-critical identity exists only inside opaque JSON.
- Embedding query: normalized duckdb supervisor schema intent runtime git code evidence cache

## DQP-006 Implement the loopback Quack state-owner service

- Status: completed
- Completion: automatic
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: quack-server
- Depends on: DQP-003, DQP-004, DQP-005
- Goal id: DQP-G010
- Outputs: ipfs_accelerate_py/agent_supervisor/runtime/quack_state_server.py, scripts/ops/agent_supervisor/quack_state_server.py, test/api/test_agent_supervisor_quack_state_server.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_quack_state_server.py
- Board namespace: agent-supervisor-duckdb-quack-control-plane-v1
- Bundle: agent-supervisor/duckdb-quack/foundation/server
- Parallel lane: dqp-server
- Resource class: cpu-io-medium
- Resource stage: implementation
- Estimated tokens: 28000
- Implementation timeout seconds: 10800
- Predicted files: ipfs_accelerate_py/agent_supervisor/runtime/quack_state_server.py, scripts/ops/agent_supervisor/quack_state_server.py, test/api/test_agent_supervisor_quack_state_server.py
- Interfaces: QuackStateServer@1, StateServerIdentity@1
- Allow concurrent with: DQP-007, DQP-010
- Conflict policy: Own new server/service entrypoints; lifecycle integration belongs to DQP-032.
- Preconditions: Database is migrated before serving and exact Quack capability passes.
- Effects: One long-lived process exclusively owns the database, starts Quack on an allocated loopback port, publishes database/schema/server/process-birth identity, checkpoints cleanly, and stops through a fenced control path.
- Evidence subset: exclusive owner, migration-before-ready, token handling, whoami, process birth, readiness, graceful stop, stale marker recovery
- Symbolic first: true
- LLM context budget bytes: 30000
- Acceptance: No token appears in argv, logs, status, exports, or provider environment; a second owner fails closed; ready requires live query plus matching store/generation/schema/server identities; non-loopback bind requires a separately reviewed policy unavailable by default.
- Embedding query: quack state server loopback exclusive duckdb owner readiness

## DQP-007 Implement the typed Quack client, transaction, and retry adapter

- Status: completed
- Completion: automatic
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: quack-client
- Depends on: DQP-002, DQP-004, DQP-005
- Goal id: DQP-G010
- Outputs: ipfs_accelerate_py/agent_supervisor/task_sources/quack_state_client.py, ipfs_accelerate_py/agent_supervisor/task_sources/control_plane_transactions.py, test/api/test_agent_supervisor_quack_state_client.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_quack_state_client.py
- Board namespace: agent-supervisor-duckdb-quack-control-plane-v1
- Bundle: agent-supervisor/duckdb-quack/foundation/client
- Parallel lane: dqp-client
- Resource class: cpu-medium
- Resource stage: implementation
- Estimated tokens: 30000
- Implementation timeout seconds: 10800
- Predicted files: ipfs_accelerate_py/agent_supervisor/task_sources/quack_state_client.py, ipfs_accelerate_py/agent_supervisor/task_sources/control_plane_transactions.py, test/api/test_agent_supervisor_quack_state_client.py
- Interfaces: QuackStateClient@1, StateTransaction@1
- Allow concurrent with: DQP-006, DQP-010
- Conflict policy: Own client and transaction modules; higher repositories consume them later.
- Preconditions: Canonical contracts, schema, and capability profile exist.
- Effects: Stateful ATTACH sessions verify server/store identity, use connection caching, execute only bounded typed statement templates, and implement CAS revision/fence/generation/idempotency with conflict classification and jittered retry.
- Evidence subset: parameter binding, raw identifier rejection, response loss, duplicate command, optimistic conflict, stale generation, reconnect, cursor pagination
- Symbolic first: true
- LLM context budget bytes: 32000
- Acceptance: Independent processes commit non-conflicting work concurrently; same-row conflicts return/retry predictably; replay after lost response returns the one committed result; callers cannot interpolate identifiers or run arbitrary model-supplied SQL.
- Embedding query: quack client attach transaction cas idempotency conflict retry

## DQP-008 Join schema, server, client, and existing DuckDB stores behind repositories

- Status: completed
- Completion: automatic
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: state-repositories
- Depends on: DQP-005, DQP-006, DQP-007
- Goal id: DQP-G010
- Outputs: ipfs_accelerate_py/agent_supervisor/task_sources/control_plane_repository.py, test/api/test_agent_supervisor_control_plane_repository.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_control_plane_repository.py test/api/test_agent_supervisor_duckdb_task_source.py
- Board namespace: agent-supervisor-duckdb-quack-control-plane-v1
- Bundle: agent-supervisor/duckdb-quack/foundation/repository
- Parallel lane: dqp-repository
- Resource class: cpu-large
- Resource stage: integration
- Estimated tokens: 30000
- Implementation timeout seconds: 10800
- Predicted files: ipfs_accelerate_py/agent_supervisor/task_sources/control_plane_repository.py, test/api/test_agent_supervisor_control_plane_repository.py
- Interfaces: StateRepository@1, EmbeddedStateRepository@1, QuackStateRepository@1
- Allow concurrent with: DQP-010, DQP-011
- Conflict policy: New facade first; existing DuckDBTaskSource, LeaseCoordinator, and MergeQueue are adapted incrementally, not rewritten here.
- Preconditions: Server/client live tests and canonical schema pass.
- Effects: Higher layers depend on a path-independent repository protocol with embedded test/import and Quack production adapters, identical canonical results, and explicit authority mode.
- Evidence subset: tasks, events, leases, commands, snapshots, transactions, schema verification, cold imports
- Symbolic first: true
- LLM context budget bytes: 32000
- Acceptance: Local and Quack adapters pass the same conformance population; Quack authority never silently falls back to direct file writes; imports can use embedded exclusive mode only under a maintenance lease.
- Embedding query: state repository adapter duckdb task source lease merge quack parity

## DQP-009 Establish state, latency, and LLM-churn baselines

- Status: completed
- Completion: automatic
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: baseline
- Depends on: DQP-000
- Goal id: DQP-G050
- Outputs: ipfs_accelerate_py/agent_supervisor/validation/duckdb_quack_baseline.py, test/api/test_agent_supervisor_duckdb_quack_baseline.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_duckdb_quack_baseline.py
- Board namespace: agent-supervisor-duckdb-quack-control-plane-v1
- Bundle: agent-supervisor/duckdb-quack/context/baseline
- Parallel lane: dqp-baseline
- Resource class: cpu-medium
- Resource stage: benchmark
- Estimated tokens: 22000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/agent_supervisor/validation/duckdb_quack_baseline.py, test/api/test_agent_supervisor_duckdb_quack_baseline.py
- Interfaces: SupervisorStateBaseline@1, LLMChurnBaseline@1
- Allow concurrent with: DQP-001, DQP-002, DQP-003, DQP-004
- Conflict policy: Own new measurement code/test; do not optimize or mutate production paths.
- Preconditions: Fixed hermetic workloads and redacted provider-call fixtures are available.
- Effects: Reproducibly measure file reads/writes/parses, independent DB opens, lock waits, no-op polling, task/claim latency, context bytes, provider calls/tokens, duplicate semantic inputs, cache reuse, accepted mutation quality, and rollback/failure rates.
- Evidence subset: cold/warm/restart/parallel strata, environment, workload, seed, samples, confidence, quality and safety floors
- Symbolic first: true
- LLM context budget bytes: 24000
- Acceptance: Baseline binds tree, environment, workload and metric definitions; distinguishes missing from zero; counts rejected/retry/abandoned provider usage; cannot be regenerated with weakened safety, durability, or quality criteria in the candidate change.
- Embedding query: supervisor baseline file churn lock latency llm tokens context duplicate calls

## DQP-010 Import legacy Markdown, JSON, JSONL, SQLite, and DuckDB state

- Status: completed
- Completion: automatic
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: state-import
- Depends on: DQP-001, DQP-002, DQP-003
- Goal id: DQP-G020
- Outputs: ipfs_accelerate_py/agent_supervisor/task_sources/legacy_state_import.py, test/api/test_agent_supervisor_legacy_state_import.py, test/fixtures/agent_supervisor/duckdb_quack_import/README.md
- Validation: python -m pytest -q test/api/test_agent_supervisor_legacy_state_import.py
- Board namespace: agent-supervisor-duckdb-quack-control-plane-v1
- Bundle: agent-supervisor/duckdb-quack/intent/import
- Parallel lane: dqp-import
- Resource class: cpu-io-medium
- Resource stage: migration
- Estimated tokens: 28000
- Implementation timeout seconds: 10800
- Predicted files: ipfs_accelerate_py/agent_supervisor/task_sources/legacy_state_import.py, test/api/test_agent_supervisor_legacy_state_import.py, test/fixtures/agent_supervisor/duckdb_quack_import/README.md
- Interfaces: LegacyStateImport@1, ImportManifest@1, ImportReceipt@1
- Allow concurrent with: DQP-005, DQP-006, DQP-007
- Conflict policy: Own importer and fixtures; never modify or delete an imported source.
- Preconditions: State-sink inventory and canonical contracts classify every source.
- Effects: Preview and apply import legacy objectives, taskboards, plan revisions, queues, events, statuses, worktrees, caches, artifacts, leases, and separate databases with byte/schema/parser provenance and explicit reconciliation.
- Evidence subset: duplicate sources, conflicts, corrupt/truncated input, unsupported schema, rejected rows, replay, source immutability
- Symbolic first: true
- LLM context budget bytes: 30000
- Acceptance: Exact replay is a no-op with the same receipt; conflicts are select/merge/quarantine/reject rather than last-write-wins; strict import commits atomically or not at all; every accepted row is traceable to source digest and parser version.
- Embedding query: legacy markdown json jsonl sqlite duckdb state import reconciliation

## DQP-011 Render deterministic Markdown, JSON, JSONL, CSV, and Parquet exports

- Status: completed
- Completion: automatic
- Is schedulable: true
- Review only: false
- Priority: P1
- Track: state-export
- Depends on: DQP-002, DQP-008
- Goal id: DQP-G020
- Outputs: ipfs_accelerate_py/agent_supervisor/task_sources/state_export.py, scripts/ops/agent_supervisor/export_control_plane_state.py, test/api/test_agent_supervisor_state_export.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_state_export.py
- Board namespace: agent-supervisor-duckdb-quack-control-plane-v1
- Bundle: agent-supervisor/duckdb-quack/intent/export
- Parallel lane: dqp-export
- Resource class: cpu-io-medium
- Resource stage: implementation
- Estimated tokens: 24000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/agent_supervisor/task_sources/state_export.py, scripts/ops/agent_supervisor/export_control_plane_state.py, test/api/test_agent_supervisor_state_export.py
- Interfaces: StateExporter@1, StateExportReceipt@1
- Allow concurrent with: DQP-012, DQP-013, DQP-014
- Conflict policy: Own exporter/CLI/test; legacy import is a separate explicit operation.
- Preconditions: Repository adapter provides consistent bounded snapshots.
- Effects: Versioned views render atomic redacted human and machine projections carrying store UUID, generation, schema revision, transaction watermark, view/renderer version, parameters, destination and digest.
- Evidence subset: byte determinism, pagination, redaction, atomic replacement, snapshot consistency, lossless round trip, lossy declaration
- Symbolic first: true
- LLM context budget bytes: 26000
- Acceptance: Re-export of identical snapshot/parameters is byte-identical; tampering with or deleting an export cannot affect runtime decisions; lossless portable export round-trips, while human Markdown declares non-authoritative and intentionally omitted fields.
- Embedding query: deterministic duckdb state export markdown json jsonl parquet receipt

## DQP-012 Migrate objectives, goals, plans, tasks, queues, and completion state

- Status: completed
- Completion: automatic
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: intent-repository
- Depends on: DQP-008, DQP-010
- Goal id: DQP-G020
- Outputs: ipfs_accelerate_py/agent_supervisor/task_sources/intent_repository.py, ipfs_accelerate_py/agent_supervisor/task_sources/database_task_source.py, test/api/test_agent_supervisor_intent_repository.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_intent_repository.py test/api/test_agent_supervisor_duckdb_task_source.py
- Board namespace: agent-supervisor-duckdb-quack-control-plane-v1
- Bundle: agent-supervisor/duckdb-quack/intent/repository
- Parallel lane: dqp-intent
- Resource class: cpu-large
- Resource stage: integration
- Estimated tokens: 32000
- Implementation timeout seconds: 10800
- Predicted files: ipfs_accelerate_py/agent_supervisor/task_sources/intent_repository.py, ipfs_accelerate_py/agent_supervisor/task_sources/database_task_source.py, test/api/test_agent_supervisor_intent_repository.py
- Interfaces: IntentRepository@1, DatabaseTaskSource@1, PlanRevisionRepository@1
- Allow concurrent with: DQP-013, DQP-014, DQP-016
- Conflict policy: New adapters precede targeted cutover of objective_tracker, plan_revision_store, taskboard_store, and persistent_task_queue.
- Preconditions: Imported CIDs reconcile and repository transaction semantics pass.
- Effects: Objective revisions, goal edges, plan revisions/deltas/heads, tasks/dependencies/acceptance/validations/outputs, queue backoff, attempts, blocks, and completion evidence advance in transactional state plus events.
- Evidence subset: CAS heads, supersession, continuation, recovery, dependency readiness, queue retry, goal reopen, current evidence
- Symbolic first: true
- LLM context budget bytes: 34000
- Acceptance: No cross-file saga is needed; completion cannot be selected without current required evidence; existing public task/plan/objective APIs retain canonical identities; database rebuild from admitted events matches current projections.
- Embedding query: objective goal plan revision task queue completion duckdb repository

## DQP-013 Replace JSONL event, audit, log, metric, and cursor authority

- Status: completed
- Completion: automatic
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: event-store
- Depends on: DQP-008
- Goal id: DQP-G020
- Outputs: ipfs_accelerate_py/agent_supervisor/runtime/database_event_log.py, test/api/test_agent_supervisor_database_event_log.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_database_event_log.py test/api/test_agent_supervisor_event_log.py
- Board namespace: agent-supervisor-duckdb-quack-control-plane-v1
- Bundle: agent-supervisor/duckdb-quack/intent/events
- Parallel lane: dqp-events
- Resource class: cpu-io-medium
- Resource stage: implementation
- Estimated tokens: 28000
- Implementation timeout seconds: 10800
- Predicted files: ipfs_accelerate_py/agent_supervisor/runtime/database_event_log.py, test/api/test_agent_supervisor_database_event_log.py
- Interfaces: DatabaseEventLog@1, EventCursor@1, ConsumerCheckpoint@1
- Allow concurrent with: DQP-011, DQP-012, DQP-014, DQP-016
- Conflict policy: Own new store/test; existing event API is adapted only after conformance.
- Preconditions: Repository transaction and pagination/cursor contracts pass.
- Effects: Append-only typed events, structured logs, metrics, traces, audit records, stream heads, retention, integrity checkpoints, and consumer cursors are queryable through Quack; JSONL becomes export only.
- Evidence subset: monotonic sequences, duplicate IDs, cursor expiry, bounded polling, coalescing, replay, redaction, retention, event/projection transaction
- Symbolic first: true
- LLM context budget bytes: 30000
- Acceptance: Event IDs and per-stream sequences are immutable; polling resumes without loss or duplicate effects; application audit is explicit rather than inferred from Quack diagnostics; recursive logging is bounded; exported JSONL deletion has no authority effect.
- Embedding query: duckdb event log audit metrics cursor polling jsonl export

## DQP-014 Register supervisors, daemons, sessions, process births, and heartbeats

- Status: completed
- Completion: automatic
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: daemon-registry
- Depends on: DQP-008, DQP-013
- Goal id: DQP-G030
- Outputs: ipfs_accelerate_py/agent_supervisor/runtime/daemon_registry.py, test/api/test_agent_supervisor_daemon_registry.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_daemon_registry.py
- Board namespace: agent-supervisor-duckdb-quack-control-plane-v1
- Bundle: agent-supervisor/duckdb-quack/runtime/daemons
- Parallel lane: dqp-daemons
- Resource class: cpu-medium
- Resource stage: implementation
- Estimated tokens: 26000
- Implementation timeout seconds: 7200
- Predicted files: ipfs_accelerate_py/agent_supervisor/runtime/daemon_registry.py, test/api/test_agent_supervisor_daemon_registry.py
- Interfaces: SupervisorInstance@1, DaemonInstance@1, DaemonSession@1, Heartbeat@1
- Allow concurrent with: DQP-012, DQP-016, DQP-020
- Conflict policy: Own new registry/test; PID/status compatibility adapters land with lifecycle cutover.
- Preconditions: Event store and repository transactions are available.
- Effects: Every master/lane/supervisor/daemon/worker session binds run, role, shard, process boot/start identity, server generation, Quack connection, capability, heartbeat, progress cursor, deadline, exit and restart disposition.
- Evidence subset: registration, adoption, PID reuse, session expiry, heartbeat compaction, late heartbeat, server restart, exact ancestry
- Symbolic first: true
- LLM context budget bytes: 28000
- Acceptance: Raw PID never proves identity; dead/reused/unknown process births cannot renew; duplicate active role/lane ownership is fenced; heartbeats and progress are distinct; status files can mirror but cannot create or extend a session.
- Embedding query: supervisor daemon session process birth heartbeat registry duckdb

## DQP-015 Consolidate task, resource, merge, and maintenance leases with fencing

- Status: completed
- Completion: automatic
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: coordination
- Depends on: DQP-008, DQP-014
- Goal id: DQP-G030
- Outputs: ipfs_accelerate_py/agent_supervisor/merge/database_coordination.py, test/api/test_agent_supervisor_database_coordination.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_database_coordination.py test/api/test_agent_supervisor_lease_coordination.py
- Board namespace: agent-supervisor-duckdb-quack-control-plane-v1
- Bundle: agent-supervisor/duckdb-quack/runtime/coordination
- Parallel lane: dqp-coordination
- Resource class: cpu-medium
- Resource stage: integration
- Estimated tokens: 30000
- Implementation timeout seconds: 10800
- Predicted files: ipfs_accelerate_py/agent_supervisor/merge/database_coordination.py, test/api/test_agent_supervisor_database_coordination.py
- Interfaces: FencedLease@1, TaskClaim@1, ResourceClaim@1, MaintenanceLease@1
- Allow concurrent with: DQP-016, DQP-020
- Conflict policy: Reuse LeaseCoordinator algorithms; do not delete legacy stores until canary release.
- Preconditions: Canonical session/process identity exists and transactions enforce expected generation/revision.
- Effects: Task claims, path/resource claims, provider/prover capacity, merge ownership, schema maintenance, backup, and offline recovery use one lease vocabulary and canonical task/worktree/session IDs.
- Evidence subset: acquire, renew, release, expiry, takeover, fairness, dependency readiness, epoch monotonicity, stale fence, response loss
- Symbolic first: true
- LLM context budget bytes: 32000
- Acceptance: Four processes never own the same exclusive scope; expired session cannot renew or mutate; append/fair scheduling remains concurrent; stale fencing epoch is rejected in every protected write; claim and task-attempt creation are one transaction.
- Embedding query: duckdb task resource lease claim fencing epoch multi daemon

## DQP-016 Persist repositories, branches, worktrees, snapshots, and dirty overlays

- Status: completed
- Completion: automatic
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: worktree-registry
- Depends on: DQP-008, DQP-014
- Goal id: DQP-G030
- Outputs: ipfs_accelerate_py/agent_supervisor/merge/database_worktree_registry.py, test/api/test_agent_supervisor_database_worktree_registry.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_database_worktree_registry.py test/api/test_agent_supervisor_worktree_lifecycle.py
- Board namespace: agent-supervisor-duckdb-quack-control-plane-v1
- Bundle: agent-supervisor/duckdb-quack/runtime/worktrees
- Parallel lane: dqp-worktrees
- Resource class: cpu-io-large
- Resource stage: implementation
- Estimated tokens: 30000
- Implementation timeout seconds: 10800
- Predicted files: ipfs_accelerate_py/agent_supervisor/merge/database_worktree_registry.py, test/api/test_agent_supervisor_database_worktree_registry.py
- Interfaces: RepositoryForest@1, WorktreeIdentity@1, WorktreeSnapshot@1, DirtyOverlay@1
- Allow concurrent with: DQP-015, DQP-020
- Conflict policy: Own new registry/test; retain a minimal OS mutex only around physical Git operations.
- Preconditions: Daemon session identity and canonical store generation are available.
- Effects: Register repositories, git common dirs, branches/refs, submodule edges, worktrees, leases, base/head/tree/index/dirty overlay snapshots, paths, setup cache and lifecycle transitions shared by all lanes.
- Evidence subset: canonical containment, symlinks, detached head, nested gitlinks, rename/delete/untracked policy, dead owner, stale path, reconciliation
- Symbolic first: true
- LLM context budget bytes: 32000
- Acceptance: Worktree reuse/cleanup requires matching lease and current Git observations; DB history is semantic authority while Git remains byte authority; stale/dead owner recovery uses CAS/fence; no worktree-local JSON index can override registry state.
- Embedding query: git repository worktree snapshot dirty overlay lease registry duckdb

## DQP-017 Propagate database task-source and Quack options through every runner

- Status: completed
- Completion: automatic
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: runner-integration
- Depends on: DQP-012, DQP-014, DQP-015, DQP-016
- Goal id: DQP-G030
- Outputs: ipfs_accelerate_py/agent_supervisor/runtime/configured_board_scheduler.py, ipfs_accelerate_py/agent_supervisor/runtime/multi_supervisor_runner.py, ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_supervisor.py, test/api/test_agent_supervisor_database_runner_propagation.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_database_runner_propagation.py test/api/test_agent_supervisor_configured_board_scheduler.py
- Board namespace: agent-supervisor-duckdb-quack-control-plane-v1
- Bundle: agent-supervisor/duckdb-quack/runtime/runners
- Parallel lane: dqp-runners
- Resource class: cpu-large
- Resource stage: integration
- Estimated tokens: 32000
- Implementation timeout seconds: 10800
- Predicted files: ipfs_accelerate_py/agent_supervisor/runtime/configured_board_scheduler.py, ipfs_accelerate_py/agent_supervisor/runtime/multi_supervisor_runner.py, ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_supervisor.py, test/api/test_agent_supervisor_database_runner_propagation.py
- Interfaces: DatabaseProgramConfig@1, DatabaseImplementationTrack@1
- Allow concurrent with: DQP-020, DQP-021
- Conflict policy: Own runner argument/config propagation; daemon behavioral cutover belongs to DQP-018.
- Preconditions: Database intent, daemon, claim, and worktree repositories pass conformance.
- Effects: Scheduler profiles select explicit authority mode, endpoint secret handle, store/generation/schema, task source, event store, runtime registry, worktree root, export profile and failover policy; exact values reach every lane and child.
- Evidence subset: parser round trip, argv/env redaction, defaults, explicit legacy mode, child environment, lane isolation, restart/adoption
- Symbolic first: true
- LLM context budget bytes: 34000
- Acceptance: No database selection is lost between configured-board, multi-runner, implementation supervisor and daemon; current implicit legacy-Markdown default is deprecated; Quack authority never silently becomes local DuckDB or file authority; provider subprocess lacks state credentials.
- Embedding query: configured board runner task source duckdb quack option propagation

## DQP-018 Cut the implementation daemon over to database-authoritative execution

- Status: completed
- Completion: automatic
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: daemon-integration
- Depends on: DQP-012, DQP-013, DQP-015, DQP-016, DQP-017
- Goal id: DQP-G030
- Outputs: ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py, ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon_runner.py, test/api/test_agent_supervisor_database_implementation_daemon.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_database_implementation_daemon.py test/api/test_agent_supervisor_implementation_daemon.py
- Board namespace: agent-supervisor-duckdb-quack-control-plane-v1
- Bundle: agent-supervisor/duckdb-quack/runtime/daemon
- Parallel lane: dqp-daemon-cutover
- Resource class: cpu-large
- Resource stage: integration
- Estimated tokens: 36000
- Implementation timeout seconds: 14400
- Predicted files: ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py, ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon_runner.py, test/api/test_agent_supervisor_database_implementation_daemon.py
- Interfaces: DatabaseImplementationDaemon@1, DatabaseTaskAttempt@1
- Allow concurrent with: DQP-020, DQP-021
- Conflict policy: This task owns daemon load/select/claim/phase/complete/retry paths; merge/validation cutover is DQP-019.
- Preconditions: End-to-end config propagation, task source, event store, claims, sessions, and worktree registry pass.
- Effects: Task selection, claims, attempt phases, provider calls, budgets, validation intent, completion, retry/backoff, blocking, heartbeats and status are database transitions; legacy files are optional projections.
- Evidence subset: ready selection, strict shards, lost response, provider capacity, hard quota, timeout, cancellation, crash, restart, stale worker, status parity
- Symbolic first: true
- LLM context budget bytes: 38000
- Acceptance: Four daemon processes claim distinct work; no task status is updated in Markdown under database authority; JSON queue/status/events/PID projections can be absent; crash/restart resumes from committed phase and does not duplicate provider/effect work.
- Embedding query: implementation daemon database task claim phase completion retry quack

## DQP-019 Integrate validation, merge queue, reconciliation, and rescue transactions

- Status: completed
- Completion: automatic
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: merge-recovery
- Depends on: DQP-013, DQP-015, DQP-016, DQP-018
- Goal id: DQP-G030
- Outputs: ipfs_accelerate_py/agent_supervisor/merge/database_merge_queue.py, ipfs_accelerate_py/agent_supervisor/rescue/database_recovery.py, test/api/test_agent_supervisor_database_merge_recovery.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_database_merge_recovery.py test/api/test_agent_supervisor_merge_queue.py
- Board namespace: agent-supervisor-duckdb-quack-control-plane-v1
- Bundle: agent-supervisor/duckdb-quack/runtime/merge-recovery
- Parallel lane: dqp-merge
- Resource class: cpu-io-large
- Resource stage: integration
- Estimated tokens: 34000
- Implementation timeout seconds: 14400
- Predicted files: ipfs_accelerate_py/agent_supervisor/merge/database_merge_queue.py, ipfs_accelerate_py/agent_supervisor/rescue/database_recovery.py, test/api/test_agent_supervisor_database_merge_recovery.py
- Interfaces: DatabaseMergeQueue@1, ValidationRun@1, RecoveryAction@1
- Allow concurrent with: DQP-021, DQP-022
- Conflict policy: Own merge/validation/recovery transaction adapters; preserve existing Git safety and exact process fencing.
- Preconditions: Database daemon attempts and worktree/lease identities exist.
- Effects: Validation starts/results, merge claims/attempts/outcomes, conflict resolution, publication, reconciliation replay, quarantine, retry budgets and rescue decisions share task/attempt/worktree/fence/event transactions.
- Evidence subset: serialized merge, fairness, stale result, rebase, conflict, validation failure, partial publish, crash, retry exhaustion, idempotent replay
- Symbolic first: true
- LLM context budget bytes: 36000
- Acceptance: A task completes only after accepted merge and current validation evidence commit together; stale worktree/fence results are rejected; recovery actions are idempotent and queryable; no JSON receipt or queue file alone can settle work.
- Embedding query: database validation merge queue reconciliation rescue transaction

## DQP-020 Persist repository snapshots, files, parser runs, AST nodes, and symbols

- Status: completed
- Completion: automatic
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: ast-index
- Depends on: DQP-005, DQP-008, DQP-016
- Goal id: DQP-G040
- Outputs: ipfs_accelerate_py/agent_supervisor/analysis/duckdb_ast_index.py, test/api/test_agent_supervisor_duckdb_ast_index.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_duckdb_ast_index.py test/api/test_agent_supervisor_analysis_ast_index.py
- Board namespace: agent-supervisor-duckdb-quack-control-plane-v1
- Bundle: agent-supervisor/duckdb-quack/code/ast
- Parallel lane: dqp-ast
- Resource class: cpu-large
- Resource stage: implementation
- Estimated tokens: 34000
- Implementation timeout seconds: 14400
- Predicted files: ipfs_accelerate_py/agent_supervisor/analysis/duckdb_ast_index.py, test/api/test_agent_supervisor_duckdb_ast_index.py
- Interfaces: DuckDBASTIndex@1, SourceSnapshot@1, ParseRun@1
- Allow concurrent with: DQP-015, DQP-017
- Conflict policy: Own new store/test; reuse existing AST/polyglot adapters without changing their semantics.
- Preconditions: Canonical repository/worktree snapshots and code-domain schema exist.
- Effects: Store content-addressed files, parser identity, AST units/nodes/edges, symbols, definitions, calls/imports/references/types and explicit parse frontiers keyed to exact tree/index/overlay.
- Evidence subset: Python/JS/TS fixtures, nested symbols, generated code, syntax errors, unsupported language, deduplication, parser drift, bounded bodies
- Symbolic first: true
- LLM context budget bytes: 36000
- Acceptance: Identical blobs/parser versions deduplicate across worktrees; failed/unsupported parses invalidate stale facts and remain explicit unknown; private/ignored files and secrets are excluded; AST rows are derived evidence, not source or semantic authority.
- Embedding query: duckdb repository snapshot ast nodes symbols calls imports references

## DQP-021 Add incremental worktree scanning and AST invalidation

- Status: completed
- Completion: automatic
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: ast-incremental
- Depends on: DQP-016, DQP-020
- Goal id: DQP-G040
- Outputs: ipfs_accelerate_py/agent_supervisor/analysis/database_repository_indexer.py, test/api/test_agent_supervisor_database_repository_indexer.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_database_repository_indexer.py
- Board namespace: agent-supervisor-duckdb-quack-control-plane-v1
- Bundle: agent-supervisor/duckdb-quack/code/incremental
- Parallel lane: dqp-indexer
- Resource class: cpu-large
- Resource stage: implementation
- Estimated tokens: 32000
- Implementation timeout seconds: 10800
- Predicted files: ipfs_accelerate_py/agent_supervisor/analysis/database_repository_indexer.py, test/api/test_agent_supervisor_database_repository_indexer.py
- Interfaces: DatabaseRepositoryIndexer@1, ASTInvalidation@1
- Allow concurrent with: DQP-018, DQP-019
- Conflict policy: Own new indexer/test; filesystem watchers are hints and never authority.
- Preconditions: Full snapshot AST ingest and worktree registry pass.
- Effects: Poll Git/content identities, parse only added/changed/renamed files, retire deleted bindings, invalidate dependent symbol/impact/cache/proof rows, and persist scan cursors and coverage/frontier receipts.
- Evidence subset: watcher loss/coalescing, rename, delete, untracked policy, submodule change, partial scan crash, clean rebuild equivalence
- Symbolic first: true
- LLM context budget bytes: 34000
- Acceptance: Incremental result equals a clean full scan for the same snapshot; missed notifications are recovered by reconciliation; a partial scan never advances the authoritative snapshot head; dependent facts cannot remain current after source/parser/policy drift.
- Embedding query: incremental git worktree ast scan invalidation full rebuild equivalence

## DQP-022 Record before and after code mutations with AST edit lineage

- Status: completed
- Completion: automatic
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: mutation-ledger
- Depends on: DQP-016, DQP-020, DQP-021
- Goal id: DQP-G040
- Outputs: ipfs_accelerate_py/agent_supervisor/analysis/mutation_ledger.py, test/api/test_agent_supervisor_mutation_ledger.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_mutation_ledger.py
- Board namespace: agent-supervisor-duckdb-quack-control-plane-v1
- Bundle: agent-supervisor/duckdb-quack/code/mutations
- Parallel lane: dqp-mutations
- Resource class: cpu-large
- Resource stage: implementation
- Estimated tokens: 34000
- Implementation timeout seconds: 10800
- Predicted files: ipfs_accelerate_py/agent_supervisor/analysis/mutation_ledger.py, test/api/test_agent_supervisor_mutation_ledger.py
- Interfaces: MutationSet@1, MutationFile@1, ASTMutation@1
- Allow concurrent with: DQP-019
- Conflict policy: Own new ledger/test; provider execution and Git mutation hooks are integrated by DQP-024.
- Preconditions: Pre/post worktree snapshots and AST identities are reproducible.
- Effects: Bind task/attempt/plan/operator/provider/daemon/session/worktree/lease/fence to before/after tree, index, blob, textual hunks, AST edit script, symbols, declared effects, validation, proof, merge and rollback outcome.
- Evidence subset: no-op edit, partial write, rename, multi-file SCC, formatting-only change, parse failure, rollback, stable structural identity
- Symbolic first: true
- LLM context budget bytes: 36000
- Acceptance: Every admitted byte change has one lineage or is rejected/quarantined; line-number churn alone does not forge a distinct semantic mutation; stale fence or mismatched before snapshot cannot record an accepted mutation; rollback restoration is independently verified.
- Embedding query: mutation ledger before after diff ast edit task worktree fence

## DQP-023 Materialize symbol dependency, impact, and changed-neighborhood queries

- Status: completed
- Completion: automatic
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: impact-closure
- Depends on: DQP-020, DQP-021, DQP-022
- Goal id: DQP-G040
- Outputs: ipfs_accelerate_py/agent_supervisor/analysis/database_impact_graph.py, test/api/test_agent_supervisor_database_impact_graph.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_database_impact_graph.py
- Board namespace: agent-supervisor-duckdb-quack-control-plane-v1
- Bundle: agent-supervisor/duckdb-quack/code/impact
- Parallel lane: dqp-impact
- Resource class: cpu-large
- Resource stage: analysis
- Estimated tokens: 32000
- Implementation timeout seconds: 10800
- Predicted files: ipfs_accelerate_py/agent_supervisor/analysis/database_impact_graph.py, test/api/test_agent_supervisor_database_impact_graph.py
- Interfaces: DatabaseImpactGraph@1, ImpactClosure@1, ChangedSymbolNeighborhood@1
- Allow concurrent with: DQP-025
- Conflict policy: Own query/materialization module and test; it cannot authorize mutation.
- Preconditions: AST/symbol graph and mutation lineage cover the current snapshot.
- Effects: Bounded versioned SQL queries produce callers/callees/imports/types/tests/contracts/proofs/config/docs and unresolved dynamic frontiers for a mutation or task, with dispositions and freshness.
- Evidence subset: recursion, SCC, aliases, reexports, dynamic calls, generated code, cross-language, deletion, parser uncertainty, pagination
- Symbolic first: true
- LLM context budget bytes: 34000
- Acceptance: All resolved consumers receive exactly one disposition; open or unsupported frontier blocks automatic repair; query result binds snapshot/parser/policy/schema; similarity and graph proximity remain nomination rather than semantic authority.
- Embedding query: symbol dependency impact closure changed neighborhood sql ast

## DQP-024 Integrate AST and mutation state with symbolic planning, repair, and proof

- Status: completed
- Completion: automatic
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: symbolic-integration
- Depends on: DQP-012, DQP-019, DQP-022, DQP-023
- Goal id: DQP-G040
- Outputs: ipfs_accelerate_py/agent_supervisor/planning/database_symbolic_planning.py, ipfs_accelerate_py/agent_supervisor/proof/database_repair_evidence.py, test/api/test_agent_supervisor_database_symbolic_repair.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_database_symbolic_repair.py
- Board namespace: agent-supervisor-duckdb-quack-control-plane-v1
- Bundle: agent-supervisor/duckdb-quack/code/symbolic-repair
- Parallel lane: dqp-symbolic
- Resource class: cpu-proof-large
- Resource stage: integration
- Estimated tokens: 38000
- Implementation timeout seconds: 14400
- Predicted files: ipfs_accelerate_py/agent_supervisor/planning/database_symbolic_planning.py, ipfs_accelerate_py/agent_supervisor/proof/database_repair_evidence.py, test/api/test_agent_supervisor_database_symbolic_repair.py
- Interfaces: DatabaseSymbolicPlanner@1, RepairLineage@1
- Allow concurrent with: DQP-025, DQP-026
- Conflict policy: New adapters compose existing planners/repair/proof; they do not grant new repair authority.
- Preconditions: Intent repository, accepted mutation ledger, impact closure, validation and merge receipts are current.
- Effects: Deterministic discovery and impact queries precede candidate synthesis; plans reference exact AST/symbol/mutation IDs; applied repair writes lineage/events/caches/proof obligations transactionally and revalidates actual worktree before effect.
- Evidence subset: candidate reuse, stale AST, counterexample, partial plan, unsupported operator, fixed point, proof invalidation, abstention
- Symbolic first: true
- LLM context budget bytes: 40000
- Acceptance: LLM cannot invent scope/semantics outside admitted plan; stale or incomplete impact prevents write; proof cache hits rederive applicability; all accepted repairs reach code-and-logic fixed point or roll back; unsupported classes require approval/abstain.
- Embedding query: symbolic planning repair ast mutation impact proof database

## DQP-025 Invert artifact, dataset, proof, and cache stores to database-first authority

- Status: completed
- Completion: automatic
- Is schedulable: true
- Review only: false
- Priority: P1
- Track: evidence-cache
- Depends on: DQP-008, DQP-013, DQP-023
- Goal id: DQP-G020
- Outputs: ipfs_accelerate_py/agent_supervisor/runtime/database_artifact_store.py, ipfs_accelerate_py/agent_supervisor/proof/database_evidence_store.py, test/api/test_agent_supervisor_database_evidence_stores.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_database_evidence_stores.py
- Board namespace: agent-supervisor-duckdb-quack-control-plane-v1
- Bundle: agent-supervisor/duckdb-quack/intent/evidence
- Parallel lane: dqp-evidence
- Resource class: cpu-io-large
- Resource stage: integration
- Estimated tokens: 34000
- Implementation timeout seconds: 14400
- Predicted files: ipfs_accelerate_py/agent_supervisor/runtime/database_artifact_store.py, ipfs_accelerate_py/agent_supervisor/proof/database_evidence_store.py, test/api/test_agent_supervisor_database_evidence_stores.py
- Interfaces: DatabaseArtifactStore@1, DatabaseEvidenceStore@1
- Allow concurrent with: DQP-024, DQP-026
- Conflict policy: New database-first facades precede targeted legacy-store switches; large immutable bodies may remain CAS by digest.
- Preconditions: Event store, schema, repository, and impact identity contracts pass.
- Effects: Artifact metadata/edges, datasets, validation/proof receipts, attestations, analysis/proof cache keys, invalidations, single-flight leases and use outcomes commit in database before optional exports.
- Evidence subset: content identity, provenance, redaction, size/graph quotas, corruption, stale key, single flight, cache applicability, rebuild
- Symbolic first: true
- LLM context budget bytes: 36000
- Acceptance: JSON/Parquet/file freshness no longer determines authority; every large external blob is digest-bound and verified on use; caches never promote assurance; stale or poisoned hits fail closed; database projections rebuild from admitted evidence.
- Embedding query: database artifact dataset proof evidence cache authority single flight

## DQP-026 Build bounded database context capsules and frontier views

- Status: completed
- Completion: automatic
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: context-views
- Depends on: DQP-012, DQP-013, DQP-023
- Goal id: DQP-G050
- Outputs: ipfs_accelerate_py/agent_supervisor/context/database_context.py, test/api/test_agent_supervisor_database_context.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_database_context.py
- Board namespace: agent-supervisor-duckdb-quack-control-plane-v1
- Bundle: agent-supervisor/duckdb-quack/context/views
- Parallel lane: dqp-context
- Resource class: cpu-medium
- Resource stage: implementation
- Estimated tokens: 30000
- Implementation timeout seconds: 10800
- Predicted files: ipfs_accelerate_py/agent_supervisor/context/database_context.py, test/api/test_agent_supervisor_database_context.py
- Interfaces: DatabaseContextManifest@1, ContextDelta@1, LLMContextFrontier@1
- Allow concurrent with: DQP-024, DQP-025
- Conflict policy: Own context query/manifests; existing ContextCompiler remains the semantic composition boundary.
- Preconditions: Task/plan/event and changed-symbol/impact views are current.
- Effects: Content-addressed capsules contain task, unmet dependencies, latest distinct failure, worktree delta, impacted symbols, open obligations, relevant decisions/evidence and exact validations, with delta-from-prior and hard row/byte/token budgets.
- Evidence subset: stable identity, pagination, progressive disclosure, secret/private exclusion, unchanged timestamps, stale input, overflow, exact dependency invalidation
- Symbolic first: true
- LLM context budget bytes: 32000
- Acceptance: Unchanged semantic state yields identical context CID despite heartbeat/time noise; changed evidence yields a bounded delta; omitted unresolved frontier is explicit; no secret/raw unrestricted repository dump enters a model packet.
- Embedding query: database context capsule delta task frontier impacted symbols llm

## DQP-027 Persist provider calls, usage, failure signatures, and churn decisions

- Status: completed
- Completion: automatic
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: provider-ledger
- Depends on: DQP-009, DQP-013, DQP-026
- Goal id: DQP-G050
- Outputs: ipfs_accelerate_py/agent_supervisor/runtime/provider_call_ledger.py, test/api/test_agent_supervisor_provider_call_ledger.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_provider_call_ledger.py
- Board namespace: agent-supervisor-duckdb-quack-control-plane-v1
- Bundle: agent-supervisor/duckdb-quack/context/provider-ledger
- Parallel lane: dqp-provider-ledger
- Resource class: cpu-medium
- Resource stage: implementation
- Estimated tokens: 30000
- Implementation timeout seconds: 10800
- Predicted files: ipfs_accelerate_py/agent_supervisor/runtime/provider_call_ledger.py, test/api/test_agent_supervisor_provider_call_ledger.py
- Interfaces: ProviderCallLedger@1, FailureSignature@1, ChurnDecision@1
- Allow concurrent with: DQP-025, DQP-029
- Conflict policy: Own new ledger/test; existing provider router remains authority for provider selection.
- Preconditions: Baseline and context identity metrics are defined.
- Effects: Record redacted call key, provider/model/endpoint, context/plan/task/attempt IDs, budgets, token estimates/actuals, latency, typed outcome/quota, response digest, mutation/validation result, duplicate and replay-suppression decision.
- Evidence subset: exact duplicate, semantic duplicate, hard quota, transient failure, response loss, retry budget, negative cache TTL, secret redaction
- Symbolic first: true
- LLM context budget bytes: 32000
- Acceptance: Same idempotency/call key dispatches once; unchanged failed proposal after exhausted policy is suppressed; changed evidence permits a new call; all rejected/abandoned/retry usage is charged; raw prompts/completions and secrets are not stored as ordinary rows.
- Embedding query: provider call ledger token usage failure signature replay suppression churn

## DQP-028 Generate delta task packets and deterministic-first replay suppression

- Status: completed
- Completion: automatic
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: prompt-economy
- Depends on: DQP-024, DQP-026, DQP-027
- Goal id: DQP-G050
- Outputs: ipfs_accelerate_py/agent_supervisor/prompt/delta_task_packet.py, test/api/test_agent_supervisor_delta_task_packet.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_delta_task_packet.py
- Board namespace: agent-supervisor-duckdb-quack-control-plane-v1
- Bundle: agent-supervisor/duckdb-quack/context/packets
- Parallel lane: dqp-packets
- Resource class: cpu-medium
- Resource stage: integration
- Estimated tokens: 32000
- Implementation timeout seconds: 10800
- Predicted files: ipfs_accelerate_py/agent_supervisor/prompt/delta_task_packet.py, test/api/test_agent_supervisor_delta_task_packet.py
- Interfaces: DeltaTaskPacket@1, DeterministicFirstDecision@1
- Allow concurrent with: DQP-029
- Conflict policy: Own packet/replay integration; no changes to provider semantic authority.
- Preconditions: Symbolic planner, context manifests, provider ledger and churn policies pass.
- Effects: Deterministic operators/cache/queries resolve known work before provider dispatch; model packet includes only unresolved bounded delta and exact allowed effects; repeated unchanged failures open a typed circuit until material evidence changes.
- Evidence subset: packet identity, progressive disclosure, deterministic hit, cache miss, unchanged reprompt, counterexample, scope/secret escape, context overflow
- Symbolic first: true
- LLM context budget bytes: 34000
- Acceptance: Provider never receives omitted authority or credential; packet/reply are bound to exact context and effect scope; unchanged failure cannot churn indefinitely; new counterexample/tree/plan/policy/schema produces a distinct admitted packet; deterministic resolution preserves validation/proof requirements.
- Embedding query: delta task packet deterministic first llm replay suppression context

## DQP-029 Add database-backed Python, CLI, MCP, status, health, logs, and lifecycle operations

- Status: completed
- Completion: automatic
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: control-api
- Depends on: DQP-011, DQP-012, DQP-013, DQP-017, DQP-027
- Goal id: DQP-G060
- Outputs: ipfs_accelerate_py/agent_supervisor/control/database_backend.py, ipfs_accelerate_py/agent_supervisor/control/database_operations.py, test/api/test_agent_supervisor_database_control_operations.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_database_control_operations.py
- Board namespace: agent-supervisor-duckdb-quack-control-plane-v1
- Bundle: agent-supervisor/duckdb-quack/control-api/operations
- Parallel lane: dqp-control-api
- Resource class: cpu-large
- Resource stage: integration
- Estimated tokens: 34000
- Implementation timeout seconds: 14400
- Predicted files: ipfs_accelerate_py/agent_supervisor/control/database_backend.py, ipfs_accelerate_py/agent_supervisor/control/database_operations.py, test/api/test_agent_supervisor_database_control_operations.py
- Interfaces: DatabaseSupervisorBackend@1, DatabaseControlOperations@1
- Allow concurrent with: DQP-028, DQP-032
- Conflict policy: Extend the canonical SupervisorControlService and operation catalog; adapters may not shell out.
- Preconditions: Database intent/events/runtime configuration and provider ledger are available.
- Effects: Discover/query goals/tasks/runs/lanes/daemons/events/logs/metrics/worktrees/mutations/AST/receipts/exports plus start/pause/resume/drain/stop/retry/cancel/quarantine/import-preview/export/backup through one typed service.
- Evidence subset: Python/CLI/MCP parity, discovery inertness, pagination/watch, authorization, dry run, permit, idempotency, lease/fence/effects, redaction
- Symbolic first: true
- LLM context budget bytes: 36000
- Acceptance: Read/proposal/mutation authority remains distinct; configured database program now has supported status/health/logs/stop rather than launch-only control; all transports share canonical request/result identity and direct service dispatch.
- Embedding query: database supervisor backend cli mcp status health lifecycle operations

## DQP-030 De-authoritize legacy files and add explicit compatibility modes

- Status: completed
- Completion: automatic
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: authority-cutover
- Depends on: DQP-010, DQP-011, DQP-017, DQP-018, DQP-019, DQP-029
- Goal id: DQP-G030
- Outputs: ipfs_accelerate_py/agent_supervisor/task_sources/task_source.py, ipfs_accelerate_py/agent_supervisor/entrypoints/state_resolver.py, test/api/test_agent_supervisor_state_authority_modes.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_state_authority_modes.py
- Board namespace: agent-supervisor-duckdb-quack-control-plane-v1
- Bundle: agent-supervisor/duckdb-quack/runtime/authority
- Parallel lane: dqp-authority
- Resource class: cpu-large
- Resource stage: cutover
- Estimated tokens: 34000
- Implementation timeout seconds: 14400
- Predicted files: ipfs_accelerate_py/agent_supervisor/task_sources/task_source.py, ipfs_accelerate_py/agent_supervisor/entrypoints/state_resolver.py, test/api/test_agent_supervisor_state_authority_modes.py
- Interfaces: StateAuthorityMode@1, DatabaseStateResolver@1
- Allow concurrent with: DQP-032, DQP-033
- Conflict policy: Own source selection/default/compatibility behavior; export renderer and importer remain explicit APIs.
- Preconditions: Database control/lifecycle, daemon, merge/recovery and import/export conformance pass.
- Effects: Closed modes legacy_import, embedded_maintenance, quack_shadow, quack_authoritative and export_only define every path; file watching/writes are disabled under Quack authority; compatibility is explicit and observable.
- Evidence subset: cold discovery, absent exports, tampered status/taskboard, local DB fallback, server unavailable, mode transition, rollback
- Symbolic first: true
- LLM context budget bytes: 36000
- Acceptance: Under Quack authority, changing/deleting MD/JSON/JSONL/PID/status projections cannot change scheduling/lifecycle; server failure returns unavailable/recovery-required rather than file fallback; legacy import cannot run implicitly; exports carry non-authority marker.
- Embedding query: state authority mode deauthorize markdown json quack cutover

## DQP-031 Migrate run registry, idempotency, audit, and self-improvement epochs

- Status: completed
- Completion: automatic
- Is schedulable: true
- Review only: false
- Priority: P1
- Track: lifecycle-state
- Depends on: DQP-012, DQP-013, DQP-025, DQP-029, DQP-030
- Goal id: DQP-G060
- Outputs: ipfs_accelerate_py/agent_supervisor/entrypoints/database_run_registry.py, ipfs_accelerate_py/agent_supervisor/self_improvement/database_epochs.py, test/api/test_agent_supervisor_database_run_registry.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_database_run_registry.py
- Board namespace: agent-supervisor-duckdb-quack-control-plane-v1
- Bundle: agent-supervisor/duckdb-quack/control-api/runs
- Parallel lane: dqp-runs
- Resource class: cpu-large
- Resource stage: integration
- Estimated tokens: 32000
- Implementation timeout seconds: 10800
- Predicted files: ipfs_accelerate_py/agent_supervisor/entrypoints/database_run_registry.py, ipfs_accelerate_py/agent_supervisor/self_improvement/database_epochs.py, test/api/test_agent_supervisor_database_run_registry.py
- Interfaces: DatabaseRunRegistry@1, ImprovementEpochRepository@1
- Allow concurrent with: DQP-032, DQP-033
- Conflict policy: New adapters preserve public run/control/self-improvement contracts; filesystem run trees become export/compatibility only.
- Preconditions: Database control operations, artifact/evidence and authority modes pass.
- Effects: Runs/heads/current pointers, lifecycle handles, idempotent control results, audits, improvement epochs/transitions/challengers/rollouts/token metrics and receipts become transactional rows linked to registered worktrees.
- Evidence subset: concurrent run creation, head CAS, lost response, replay, challenger isolation, epoch transition, rollback, redaction, list pagination
- Symbolic first: true
- LLM context budget bytes: 34000
- Acceptance: Directory scan cannot create a run; duplicate idempotency key with different request conflicts; exact replay returns prior result; challenger uses ordinary worktree/session/lease identities; self-improvement can be planned as goals/tasks in the same database.
- Embedding query: database run registry idempotency audit self improvement epoch

## DQP-032 Add database-derived watchdog, stall diagnostics, and safe repair

- Status: completed
- Completion: automatic
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: watchdog
- Depends on: DQP-013, DQP-014, DQP-017, DQP-018, DQP-019
- Goal id: DQP-G070
- Outputs: ipfs_accelerate_py/agent_supervisor/rescue/database_watchdog.py, scripts/ops/agent_supervisor/duckdb_quack_doctor.py, test/api/test_agent_supervisor_database_watchdog.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_database_watchdog.py
- Board namespace: agent-supervisor-duckdb-quack-control-plane-v1
- Bundle: agent-supervisor/duckdb-quack/assurance/watchdog
- Parallel lane: dqp-watchdog
- Resource class: cpu-medium
- Resource stage: operations
- Estimated tokens: 32000
- Implementation timeout seconds: 10800
- Predicted files: ipfs_accelerate_py/agent_supervisor/rescue/database_watchdog.py, scripts/ops/agent_supervisor/duckdb_quack_doctor.py, test/api/test_agent_supervisor_database_watchdog.py
- Interfaces: DatabaseWatchdog@1, StallDiagnosis@1, FencedRecoveryCommand@1
- Allow concurrent with: DQP-030, DQP-033
- Conflict policy: Own new watchdog/doctor/test; raw PID signals and lock deletion are prohibited.
- Preconditions: Sessions, heartbeats, claims, attempts, events, worktrees, merges and provider states are in the database.
- Effects: Views and policy classify healthy active, quiescent strict shard, provider capacity backoff, expiring/stale session, orphan lease/worktree, ready-unclaimable, phase/log stall, migration/server/backup fault, merge/recovery blockage and terminal drain.
- Evidence subset: delta-written state, stale mtime, live worker, PID reuse, no ready shard work, quota backoff, phase deadline, server restart, exact fence
- Symbolic first: true
- LLM context budget bytes: 34000
- Acceptance: Repair requires current expected fence/process birth/generation and is idempotent; no action follows file age alone; ready work without valid owner/capacity/dependency reason becomes actionable; doctor exposes evidence and abstains when ownership is unknown.
- Embedding query: database watchdog stall heartbeat task claim worktree safe repair

## DQP-033 Implement checkpoint, backup, restore, retention, and generation rotation

- Status: completed
- Completion: automatic
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: backup-recovery
- Depends on: DQP-005, DQP-006, DQP-007, DQP-008, DQP-013
- Goal id: DQP-G070
- Outputs: ipfs_accelerate_py/agent_supervisor/runtime/control_plane_backup.py, test/api/test_agent_supervisor_control_plane_backup.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_control_plane_backup.py
- Board namespace: agent-supervisor-duckdb-quack-control-plane-v1
- Bundle: agent-supervisor/duckdb-quack/assurance/backup
- Parallel lane: dqp-backup
- Resource class: io-large
- Resource stage: operations
- Estimated tokens: 30000
- Implementation timeout seconds: 10800
- Predicted files: ipfs_accelerate_py/agent_supervisor/runtime/control_plane_backup.py, test/api/test_agent_supervisor_control_plane_backup.py
- Interfaces: ControlPlaneBackup@1, RestoreReceipt@1, StoreGenerationRotation@1
- Allow concurrent with: DQP-030, DQP-031, DQP-032
- Conflict policy: Own backup/recovery module/test; service lifecycle integration uses a maintenance lease.
- Preconditions: State-owner can checkpoint and exclusive maintenance is fenced.
- Effects: Create verified consistent snapshots, retention manifests, corruption probes, restore rehearsals and store-generation rotation; old clients/leases fail after restore/takeover; external backup bodies are encrypted/digest-bound where configured.
- Evidence subset: crash before/after checkpoint, corrupt copy, disk full, partial restore, schema version, server stopped, stale client, backup age
- Symbolic first: true
- LLM context budget bytes: 32000
- Acceptance: Restore reproduces store/schema/event/task/lease roots and invalidates pre-rotation writers; no accepted state is lost in declared crash matrix; backup success is independently verified; direct-file maintenance cannot occur while server ownership is live/unknown.
- Embedding query: duckdb checkpoint backup restore generation rotation crash recovery

## DQP-034 Prove Quack security, concurrency, conflict, and restart behavior

- Status: completed
- Completion: automatic
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: quack-chaos
- Depends on: DQP-006, DQP-007, DQP-008, DQP-015, DQP-032, DQP-033
- Goal id: DQP-G070
- Outputs: ipfs_accelerate_py/agent_supervisor/validation/quack_chaos.py, test/api/test_agent_supervisor_quack_chaos.py, docs/architecture/agent_supervisor/DUCKDB_QUACK_THREAT_MODEL.md
- Validation: python -m pytest -q test/api/test_agent_supervisor_quack_chaos.py
- Board namespace: agent-supervisor-duckdb-quack-control-plane-v1
- Bundle: agent-supervisor/duckdb-quack/assurance/chaos
- Parallel lane: dqp-chaos
- Resource class: cpu-io-large
- Resource stage: validation
- Estimated tokens: 36000
- Implementation timeout seconds: 14400
- Predicted files: ipfs_accelerate_py/agent_supervisor/validation/quack_chaos.py, test/api/test_agent_supervisor_quack_chaos.py, docs/architecture/agent_supervisor/DUCKDB_QUACK_THREAT_MODEL.md
- Interfaces: QuackChaosReport@1, QuackSecurityPolicy@1
- Allow concurrent with: DQP-035
- Conflict policy: Own live gate, fixtures, and threat model; never expose server beyond loopback.
- Preconditions: Server/client/claims/watchdog/backup support live independent-process testing.
- Effects: Exercise four clients, append/hot-row conflicts, lost replies, latency, server kill/restart, stale connections/generations, credential rotation, token leakage, raw SQL/file/attach/install attempts, provider environment and denial logging.
- Evidence subset: authn/authz defaults, loopback, TLS boundary statement, Python-UDF limitation, retry jitter, split brain, restore, denial, secret scan
- Symbolic first: true
- LLM context budget bytes: 38000
- Acceptance: No provider/LLM process obtains token or arbitrary SQL; unauthorized/cross-root/file/extension queries fail before effect; same-row conflicts are bounded; stale clients cannot write after restart/rotation; live tests cannot silently skip when profile claims compatible.
- Embedding query: quack security concurrency chaos restart conflict authorization token

## DQP-035 Run the full multi-daemon, multi-worktree database-authoritative E2E

- Status: completed
- Completion: automatic
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: database-e2e
- Depends on: DQP-017, DQP-018, DQP-019, DQP-021, DQP-022, DQP-023, DQP-024, DQP-028, DQP-029, DQP-030, DQP-032, DQP-033, DQP-034
- Goal id: DQP-G080
- Outputs: ipfs_accelerate_py/agent_supervisor/validation/duckdb_quack_canary.py, test/api/test_agent_supervisor_duckdb_quack_canary.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_duckdb_quack_canary.py
- Board namespace: agent-supervisor-duckdb-quack-control-plane-v1
- Bundle: agent-supervisor/duckdb-quack/rollout/canary
- Parallel lane: dqp-canary
- Resource class: cpu-io-large
- Resource stage: validation
- Estimated tokens: 40000
- Implementation timeout seconds: 18000
- Predicted files: ipfs_accelerate_py/agent_supervisor/validation/duckdb_quack_canary.py, test/api/test_agent_supervisor_duckdb_quack_canary.py
- Interfaces: DuckDBQuackCanary@1
- Allow concurrent with:
- Conflict policy: Own canary harness/test; production control paths are read-only inputs.
- Preconditions: All state/runtime/code/context/control/security integrations pass focused tests.
- Effects: Bootstrap a clean DB, start state-owner, register four strict lanes, execute file-disjoint tasks in isolated worktrees, record AST/mutations/provider/validation/merge, restart server/worker, reconcile, refill, export and reach terminal state without control files.
- Evidence subset: real processes, overlap, claim/fence, worktree, mutation, validation, merge, restart, refill, export, drain, database queries
- Symbolic first: true
- LLM context budget bytes: 42000
- Acceptance: At least two lanes overlap; no duplicate claim/effect or stale write; every change has complete lineage; server/worker restart resumes; final tasks/goals/daemons/worktrees/events/proofs agree; tampered/missing exports do not affect result; all processes drain cleanly.
- Embedding query: multi daemon worktree quack authoritative canary end to end

## DQP-036 Compare quality, safety, throughput, and LLM churn with baseline

- Status: completed
- Completion: automatic
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: churn-benchmark
- Depends on: DQP-009, DQP-027, DQP-028, DQP-034, DQP-035
- Goal id: DQP-G050
- Outputs: ipfs_accelerate_py/agent_supervisor/validation/duckdb_quack_benchmark.py, test/api/test_agent_supervisor_duckdb_quack_benchmark.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_duckdb_quack_benchmark.py
- Board namespace: agent-supervisor-duckdb-quack-control-plane-v1
- Bundle: agent-supervisor/duckdb-quack/context/benchmark
- Parallel lane: dqp-benchmark
- Resource class: cpu-large
- Resource stage: benchmark
- Estimated tokens: 32000
- Implementation timeout seconds: 14400
- Predicted files: ipfs_accelerate_py/agent_supervisor/validation/duckdb_quack_benchmark.py, test/api/test_agent_supervisor_duckdb_quack_benchmark.py
- Interfaces: DuckDBQuackBenchmarkReport@1
- Allow concurrent with: DQP-037
- Conflict policy: Own benchmark/report code; cannot lower baselines or safety floors in response to failure.
- Preconditions: Original baseline and complete canary/provider ledger are available.
- Effects: Compare cold/warm/restart/parallel file I/O/parsing, lock/conflict wait, queue/task latency, throughput, context bytes, calls/tokens, duplicates, cache reuse, accepted-mutation quality, defects, rollback and safety.
- Evidence subset: environment equivalence, workload/seed, samples/confidence, missing telemetry, non-compensable quality, stratification
- Symbolic first: true
- LLM context budget bytes: 34000
- Acceptance: Warm reuse improves materially and duplicate unchanged provider work is eliminated; no quality/safety floor regresses; throughput/latency remain within reviewed bounds; result reports unavailable/missing honestly and does not infer causality beyond paired evidence.
- Embedding query: duckdb quack benchmark llm churn tokens context quality throughput

## DQP-037 Backfill legacy state and prove shadow decision parity

- Status: completed
- Completion: automatic
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: shadow-rollout
- Depends on: DQP-010, DQP-030, DQP-033, DQP-035
- Goal id: DQP-G080
- Outputs: ipfs_accelerate_py/agent_supervisor/self_improvement/database_shadow_rollout.py, test/api/test_agent_supervisor_database_shadow_rollout.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_database_shadow_rollout.py
- Board namespace: agent-supervisor-duckdb-quack-control-plane-v1
- Bundle: agent-supervisor/duckdb-quack/rollout/shadow
- Parallel lane: dqp-shadow
- Resource class: cpu-io-large
- Resource stage: rollout
- Estimated tokens: 34000
- Implementation timeout seconds: 14400
- Predicted files: ipfs_accelerate_py/agent_supervisor/self_improvement/database_shadow_rollout.py, test/api/test_agent_supervisor_database_shadow_rollout.py
- Interfaces: DatabaseShadowRollout@1, ShadowParityReport@1
- Allow concurrent with: DQP-036
- Conflict policy: Shadow writes are explicitly non-authoritative until a later canary decision; no dual authority.
- Preconditions: Import, authority modes, backup/restore and database canary pass.
- Effects: Backfill reviewed programs/state, shadow old reads and lifecycle decisions into database transactions, compare canonical tasks/readiness/events/revisions/leases/status/exports, and require reviewed dispositions for every drift.
- Evidence subset: counts/digests, duplicate/conflict, task CID, readiness, lease/fence, status, event cursor, completion, restart, export
- Symbolic first: true
- LLM context budget bytes: 36000
- Acceptance: Exact import reconciles; no unexplained authority-relevant drift; shadow never controls production effect; dual observation has bounded duration/retention; rollback and re-run preserve history and generate the same parity decision.
- Embedding query: legacy state backfill database shadow parity supervisor decisions

## DQP-038 Implement staged canary, default cutover, rollback, and operator guide

- Status: completed
- Completion: automatic
- Is schedulable: true
- Review only: false
- Priority: P0
- Track: rollout-cutover
- Depends on: DQP-034, DQP-035, DQP-036, DQP-037
- Goal id: DQP-G080
- Outputs: ipfs_accelerate_py/agent_supervisor/self_improvement/database_rollout.py, scripts/ops/agent_supervisor/duckdb_quack_control_plane.py, docs/guides/AGENT_SUPERVISOR_DUCKDB_QUACK_GUIDE.md, test/api/test_agent_supervisor_database_rollout.py
- Validation: python -m pytest -q test/api/test_agent_supervisor_database_rollout.py
- Board namespace: agent-supervisor-duckdb-quack-control-plane-v1
- Bundle: agent-supervisor/duckdb-quack/rollout/cutover
- Parallel lane: dqp-cutover
- Resource class: cpu-large
- Resource stage: rollout
- Estimated tokens: 36000
- Implementation timeout seconds: 14400
- Predicted files: ipfs_accelerate_py/agent_supervisor/self_improvement/database_rollout.py, scripts/ops/agent_supervisor/duckdb_quack_control_plane.py, docs/guides/AGENT_SUPERVISOR_DUCKDB_QUACK_GUIDE.md, test/api/test_agent_supervisor_database_rollout.py
- Interfaces: DatabaseRolloutPolicy@1, DatabaseCutoverReceipt@1
- Allow concurrent with:
- Conflict policy: Own rollout policy/ops/guide/test; default switch is serialized and evidence-gated.
- Preconditions: Chaos, canary, churn/quality and shadow gates are current for exact tree/schema/profile.
- Effects: Stages off, observe, shadow, assist, canary, default and rollback bind exact authority mode, roots, store generation, schema, Quack profile, evidence, expiry, kill switch and operator action; defaults switch only after canary.
- Evidence subset: promotion denial, stale evidence, partial rollout, server unavailable, backup age, rollback, legacy export, beta waiver, remote prohibition
- Symbolic first: true
- LLM context budget bytes: 38000
- Acceptance: New local programs default to Quack only under valid release gate; rollback switches route without deleting history or accepting legacy dual writes; guide accurately states beta/single-failure-domain/loopback limitations and exact health/backup/restore/upgrade procedures.
- Embedding query: quack database rollout canary cutover rollback operator guide

## DQP-039 Issue the joined database-control-plane release receipt

- Status: todo
- Completion: automatic
- Is schedulable: true
- Review only: true
- Priority: P0
- Track: release
- Depends on: DQP-001, DQP-002, DQP-003, DQP-004, DQP-005, DQP-006, DQP-007, DQP-008, DQP-009, DQP-010, DQP-011, DQP-012, DQP-013, DQP-014, DQP-015, DQP-016, DQP-017, DQP-018, DQP-019, DQP-020, DQP-021, DQP-022, DQP-023, DQP-024, DQP-025, DQP-026, DQP-027, DQP-028, DQP-029, DQP-030, DQP-031, DQP-032, DQP-033, DQP-034, DQP-035, DQP-036, DQP-037, DQP-038
- Goal id: DQP-G090
- Outputs: ipfs_accelerate_py/agent_supervisor/validation/duckdb_quack_release.py, test/api/test_agent_supervisor_duckdb_quack_release.py, docs/architecture/AGENT_SUPERVISOR_DUCKDB_QUACK_RELEASE.md
- Validation: python -m pytest -q test/api/test_agent_supervisor_duckdb_quack_release.py
- Board namespace: agent-supervisor-duckdb-quack-control-plane-v1
- Bundle: agent-supervisor/duckdb-quack/release
- Parallel lane: dqp-release
- Resource class: cpu-large
- Resource stage: release
- Estimated tokens: 36000
- Implementation timeout seconds: 14400
- Predicted files: ipfs_accelerate_py/agent_supervisor/validation/duckdb_quack_release.py, test/api/test_agent_supervisor_duckdb_quack_release.py, docs/architecture/AGENT_SUPERVISOR_DUCKDB_QUACK_RELEASE.md
- Interfaces: DuckDBControlPlaneReleaseReceipt@1
- Allow concurrent with:
- Conflict policy: Own terminal aggregation only; never fabricate or refresh component evidence in the verifier.
- Preconditions: Every prior task is terminal and exact current tree/store/schema/profile evidence remains fresh.
- Effects: Independently query and join schema, Quack, import/export, intent, runtime, worktree, AST/mutation, symbolic/proof, context/churn, control, watchdog, backup, chaos, canary, shadow, cutover and rollback roots into a content-bound decision.
- Evidence subset: current Git/database identity, schema checksum, extension fingerprint, task/goal completion, tests, live receipts, safety floors, beta status, rollback
- Symbolic first: true
- LLM context budget bytes: 40000
- Acceptance: Release fails on missing/stale/synthetic/skipped evidence, any legacy file decision read in canary, unauthorized SQL, stale lease write, false completion, lost accepted state, incomplete mutation lineage, projection divergence, safety/quality regression, or absent rollback; pass records Quack experimental scope without claiming production HA or future 2.0 compatibility.
- Embedding query: joined duckdb quack agent supervisor release receipt safety compatibility
