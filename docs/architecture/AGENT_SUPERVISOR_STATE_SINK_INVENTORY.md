# Agent Supervisor State Sink Inventory

Interface: `SupervisorStateSinkInventory@1`  
Schema: `ipfs_accelerate_py/agent-supervisor/state-sink-inventory@1`  
Version: `1`  
Program: `agent-supervisor-duckdb-quack-control-plane-v1` / DQP-001

This inventory classifies every mutable supervisor state sink that the
DuckDB + Quack control-plane migration must absorb, demote, or retain.
It is generated from the curated catalog in
`scripts/ops/agent_supervisor/inventory_state_sinks.py` and gated by a
deterministic scanner that **fails CI on any unclassified mutable sink**.

## Authority boundary: Git/source bytes vs supervisor state

| Kind | Authority? | Migrates into control.duckdb? | Notes |
| --- | --- | --- | --- |
| Git objects / worktree source files | **Yes for source bytes** | **No** | Git remains byte authority; DB stores identities, AST, mutations |
| Markdown/JSON/JSONL/PID/lock/SQLite/DuckDB orchestration sinks | Yes for orchestration today | **Yes** (authority rows) or export/cache only | Classified below |
| Operator-sealed scheduler/config inputs | Static input | Import once with provenance | Not daemon-rewritten authority |
| Caches and dual-write query sidecars | No | Optional projections | Never grant leases or completion |

The scanner treats Git/source bytes as `static_input` / `non_state` and
marks `is_git_source_bytes=true`. They are **not** mutable supervisor
orchestration sinks.

## Classification taxonomy

| Classification | Meaning |
| --- | --- |
| `authority` | Current write authority for orchestration decisions |
| `static_input` | Operator/source input; not rewritten as live authority |
| `immutable_evidence` | Content-addressed or append-only evidence/receipts |
| `cache` | Rebuildable accelerator; never lease/completion authority |
| `export` | Read-only rendering of authoritative state |
| `os_bootstrap` | PID/lock/status handles required by the OS or operators |
| `emergency_diagnostic` | Recovery/quarantine diagnostics; fail-closed side path |

## Catalog summary

- Sink count: **44**
- Direct DuckDB writers: **13**
- Cross-file atomicity gaps: **12**
- Reuse candidates recorded: **44**
- Discovery markers scanned: *(runtime)*
- Unclassified markers: **0** (CI fails if non-zero)

## Direct DuckDB writers

These modules open DuckDB files for read-write orchestration or
projection and must funnel through the future Quack state-owner:

- `artifact-store-json-duckdb` — `ipfs_accelerate_py/agent_supervisor/runtime/artifact_store.py` → `{artifact}.json + sidecar {artifact}.duckdb` (reuse: Bounded ArtifactStore dual JSON/DuckDB projection pattern)
- `bundle-index-duckdb-sidecar` — `ipfs_accelerate_py/agent_supervisor/task_sources/todo_vector_index.py` → `{bundle_index}.json + {bundle_index}.duckdb` (reuse: ArtifactStore dual-write + bounded query tables)
- `duckdb-state-primitives` — `ipfs_accelerate_py/agent_supervisor/task_sources/duckdb_state.py` → `{path}.duckdb + .{name}.lock (+ optional legacy .sqlite3)` (reuse: Shared exclusive_file_lock + open_duckdb_connection for all single-writer DuckDB stores)
- `duckdb-task-source` — `ipfs_accelerate_py/agent_supervisor/task_sources/duckdb_task_source.py` → `{state_root}/tasks.duckdb (caller-supplied path)` (reuse: Primary reuse target for control-plane tasks/claims schema via duckdb_state flock + short transactions)
- `formal-verification-cache-duckdb` — `ipfs_accelerate_py/agent_supervisor/proof/formal_verification_cache.py` → `{root}/formal_verification_cache.duckdb` (reuse: Content-addressed proof cache keyed by obligation digest)
- `lease-coordination-duckdb` — `ipfs_accelerate_py/agent_supervisor/merge/lease_coordination.py` → `{state_root}/coordination.duckdb` (reuse: LeaseCoordinator fencing/CAS model becomes control-plane leases)
- `legacy-landed-review-duckdb` — `ipfs_accelerate_py/agent_supervisor/todo_daemon/legacy_landed_result_cache.py` → `{root}/legacy_landed_review_results.duckdb` (reuse: Attempt result projection tables)
- `merge-queue-duckdb` — `ipfs_accelerate_py/agent_supervisor/merge/merge_queue.py` → `{queue_dir}/merge_queue.duckdb (+ legacy merge_queue.sqlite3)` (reuse: merge_queue_entries + resource_claims tables)
- `merge-resolver-duckdb` — `ipfs_accelerate_py/agent_supervisor/merge/merge_resolver.py` → `{state_dir}/merge_resolver.duckdb (+ legacy .sqlite3)` (reuse: merge_attempts domain events + MergeQueue fence)
- `proof-carrying-workflow-duckdb` — `ipfs_accelerate_py/agent_supervisor/planning/proof_carrying_planner.py` → `{store}/proof_carrying_workflow.duckdb (+ JSON twin)` (reuse: plan_revisions + ArtifactStore dual JSON/DuckDB pattern)
- `proof-scheduler-duckdb` — `ipfs_accelerate_py/agent_supervisor/proof/proof_scheduler.py` → `{root}/proof_scheduler.duckdb` (reuse: duckdb_state resolve_duckdb_path + flock pattern)
- `prompt-workflow-duckdb-materialization` — `ipfs_accelerate_py/agent_supervisor/prompt/prompt_workflow.py` → `{output_policy.duckdb_path} via DuckDBTaskSource` (reuse: DuckDBTaskSource materializer already used by prompt workflow)
- `prover-evidence-duckdb` — `ipfs_accelerate_py/agent_supervisor/proof/prover_evidence_store.py` → `{root}/prover_evidence.duckdb (+ .json projection)` (reuse: Artifact metadata tables + content-addressed receipts)

## Cross-file atomicity gaps

Multi-file sinks that cannot commit intent, events, and status as one
transaction today (migration must close these):

- `artifact-store-json-duckdb`: JSON body and DuckDB sidecar are dual-written; crash can leave one side stale until rebuild
- `bundle-index-duckdb-sidecar`: Bundle planning JSON and DuckDB sidecar may diverge after crash
- `bundle-lane-manifest`: Lane manifest, metrics JSON, coordination.duckdb, and PID files are updated without a shared transaction
- `control-audit-jsonl`: Control audit, idempotency, transaction, and lifecycle JSONL files are independent append streams
- `merge-queue-duckdb`: Queue DB, train receipts directory, and checkout lock files are independent writers
- `merge-train-state-dir`: Train receipts, consumer lock, gate cache, and merge queue DB are not one atomic unit
- `objective-heap-markdown`: Objective heap rewrites and paired taskboard/status updates are not one transaction; crash can diverge intent vs schedule
- `plan-revision-store`: Active projection, index, and event log are separate files; recovery rebuilds active from events but crash windows exist
- `proof-carrying-workflow-duckdb`: Proof-carrying planner JSON and DuckDB twin are dual-written
- `recovery-receipts-and-locks`: Incident locks, receipts, and quarantine directories update independently of lease/task authority files
- `taskboard-markdown`: Board status writes, events.jsonl, and daemon status JSON are separate files without a shared transaction
- `taskboard-store-events`: Event append and markdown board rewrite are not atomic together

## Reuse candidates

Existing primitives the control plane should generalize rather than
re-implement:

- `accepted-work-ledger-jsonl`: completion_receipts + domain_events append stream
- `analysis-cache-lock`: parse_runs + source_snapshots with content digests
- `analysis-program-cache`: source_snapshots + parse_runs keyed by tree id
- `analysis-singleflight-sqlite`: Distributed single-flight via LeaseCoordinator
- `artifact-store-json-duckdb`: Bounded ArtifactStore dual JSON/DuckDB projection pattern
- `bundle-index-duckdb-sidecar`: ArtifactStore dual-write + bounded query tables
- `bundle-lane-manifest`: daemon_instances + scheduler metrics tables
- `checkout-mutation-lock`: path_claims + maintenance_leases
- `control-audit-jsonl`: domain_events + idempotency_records streams
- `control-plane-migration-lock`: Migration ownership rows + process-birth fencing; lock is bootstrap only around the state-owner
- `daemon-pid-files`: process-birth identity on daemon_sessions
- `daemon-status-json`: heartbeats + attempt_phases projections
- `dataset-store-jsonl`: artifact metadata + content digests
- `doctor-proof-cache-sqlite`: Migrate onto formal_verification_cache.duckdb primitives
- `duckdb-state-primitives`: Shared exclusive_file_lock + open_duckdb_connection for all single-writer DuckDB stores
- `duckdb-task-source`: Primary reuse target for control-plane tasks/claims schema via duckdb_state flock + short transactions
- `formal-verification-cache-duckdb`: Content-addressed proof cache keyed by obligation digest
- `git-source-bytes`: Repository forest + worktree snapshot identities; never store source blobs as control-plane authority
- `goal-tactician-lifecycle-jsonl`: domain_events + proof_attempts audit streams
- `integration-tool-install-locks`: capability_snapshots + digest-bound toolchain installs
- `lease-coordination-duckdb`: LeaseCoordinator fencing/CAS model becomes control-plane leases
- `legacy-landed-review-duckdb`: Attempt result projection tables
- `merge-checkpoint-json`: merge_attempts checkpoint columns
- `merge-queue-duckdb`: merge_queue_entries + resource_claims tables
- `merge-resolver-duckdb`: merge_attempts domain events + MergeQueue fence
- `merge-train-state-dir`: merge_attempts + validation_runs + worktree rows
- `objective-heap-markdown`: objective_graph + goal_completion identity/revision contracts
- `persistent-task-queue-json`: task_claims + selection penalty columns
- `plan-revision-store`: plan_revisions + planning_decisions tables with CAS
- `proof-carrying-workflow-duckdb`: plan_revisions + ArtifactStore dual JSON/DuckDB pattern
- `proof-certificate-locks`: artifact retention leases + content-addressed receipts
- `proof-scheduler-duckdb`: duckdb_state resolve_duckdb_path + flock pattern
- `prompt-workflow-duckdb-materialization`: DuckDBTaskSource materializer already used by prompt workflow
- `prover-evidence-duckdb`: Artifact metadata tables + content-addressed receipts
- `recovery-receipts-and-locks`: recovery_actions + quarantine tables
- `repository-index-lock`: repository_revisions + source_files index tables
- `run-registry-lock`: client_sessions + resource_claims under Quack
- `runtime-event-log-jsonl`: domain_events append stream with monotonic sequence
- `scheduler-config-json`: launch policy rows imported once with provenance
- `taskboard-markdown`: TaskboardStore materialization journal + DuckDBTaskSource fenced projection
- `taskboard-materialization-lock`: Process-birth lease + fencing epoch in control plane
- `taskboard-store-events`: RotatingEventLog / domain_events stream
- `validation-rollout-pid-status`: daemon_sessions process-birth identity + heartbeats
- `worktree-lifecycle-records`: worktrees + leases tables with process-birth fencing

## Full sink catalog

| sink_id | class | media | domain | retirement | module | path |
| --- | --- | --- | --- | --- | --- | --- |
| `accepted-work-ledger-jsonl` | `immutable_evidence` | `jsonl` | `events_logs_metrics` | `default_cutover` | `ipfs_accelerate_py/agent_supervisor/todo_daemon/artifacts.py` | `{daemon_state}/accepted-work.jsonl (+ accepted_changes.jsonl)` |
| `analysis-cache-lock` | `cache` | `lock` | `code_intelligence` | `legacy_retirement` | `ipfs_accelerate_py/agent_supervisor/analysis/analysis_cache.py` | `{cache}/.analysis-cache.lock` |
| `analysis-program-cache` | `cache` | `cache_dir` | `code_intelligence` | `legacy_retirement` | `ipfs_accelerate_py/agent_supervisor/analysis/program_analysis_cache.py` | `{cache_root}/program-analysis/**` |
| `analysis-singleflight-sqlite` | `cache` | `sqlite` | `code_intelligence` | `legacy_retirement` | `ipfs_accelerate_py/agent_supervisor/analysis/cache_coordinator.py` | `{cache}/single-flight.sqlite3` |
| `artifact-store-json-duckdb` | `immutable_evidence` | `artifact` | `artifacts_proofs` | `dual_observation` | `ipfs_accelerate_py/agent_supervisor/runtime/artifact_store.py` | `{artifact}.json + sidecar {artifact}.duckdb` |
| `bundle-index-duckdb-sidecar` | `cache` | `duckdb` | `context_budgets` | `legacy_retirement` | `ipfs_accelerate_py/agent_supervisor/task_sources/todo_vector_index.py` | `{bundle_index}.json + {bundle_index}.duckdb` |
| `bundle-lane-manifest` | `authority` | `json` | `execution_lifecycle` | `default_cutover` | `ipfs_accelerate_py/agent_supervisor/objectives/bundle_supervisor.py` | `{state_root}/bundle_lanes.json + scheduler_metrics.json` |
| `checkout-mutation-lock` | `os_bootstrap` | `lock` | `repository_worktree` | `default_cutover` | `ipfs_accelerate_py/agent_supervisor/merge/checkout_lock.py` | `{repo}/implementation-main-merge.lock (+ related locks)` |
| `control-audit-jsonl` | `authority` | `jsonl` | `events_logs_metrics` | `default_cutover` | `ipfs_accelerate_py/agent_supervisor/control/control_plane.py` | `{state}/control-audit.jsonl + control-idempotency.jsonl + control-transactions.jsonl + supervisor-lifecycle-events.jsonl` |
| `control-plane-migration-lock` | `os_bootstrap` | `lock` | `schema_deployment` | `foundation` | `ipfs_accelerate_py/agent_supervisor/task_sources/control_plane_migrations.py` | `.{database}.migration.lock (+ control.duckdb bookkeeping)` |
| `daemon-pid-files` | `os_bootstrap` | `pid` | `execution_lifecycle` | `default_cutover` | `ipfs_accelerate_py/agent_supervisor/todo_daemon/wrapper.py` | `{state}/*.pid` |
| `daemon-status-json` | `os_bootstrap` | `json` | `execution_lifecycle` | `default_cutover` | `ipfs_accelerate_py/agent_supervisor/todo_daemon/status.py` | `{state}/*status.json / *.progress.json` |
| `dataset-store-jsonl` | `immutable_evidence` | `jsonl` | `artifacts_proofs` | `dual_observation` | `ipfs_accelerate_py/agent_supervisor/task_sources/dataset_store.py` | `{root}/{dataset_id}.jsonl (+ optional parquet)` |
| `doctor-proof-cache-sqlite` | `cache` | `sqlite` | `artifacts_proofs` | `legacy_retirement` | `ipfs_accelerate_py/agent_supervisor/proof/doctor_proof_cache.py` | `{root}/doctor_proof_cache.sqlite3 (caller path)` |
| `duckdb-state-primitives` | `authority` | `duckdb` | `schema_deployment` | `foundation` | `ipfs_accelerate_py/agent_supervisor/task_sources/duckdb_state.py` | `{path}.duckdb + .{name}.lock (+ optional legacy .sqlite3)` |
| `duckdb-task-source` | `authority` | `duckdb` | `objectives_plans_tasks` | `foundation` | `ipfs_accelerate_py/agent_supervisor/task_sources/duckdb_task_source.py` | `{state_root}/tasks.duckdb (caller-supplied path)` |
| `formal-verification-cache-duckdb` | `cache` | `duckdb` | `artifacts_proofs` | `legacy_retirement` | `ipfs_accelerate_py/agent_supervisor/proof/formal_verification_cache.py` | `{root}/formal_verification_cache.duckdb` |
| `git-source-bytes` | `static_input` | `git_source` | `non_state` | `retain_permanent` | `ipfs_accelerate_py/agent_supervisor/analysis/repository_forest.py` | `{repo_root}/.git/** and worktree source files` |
| `goal-tactician-lifecycle-jsonl` | `immutable_evidence` | `jsonl` | `artifacts_proofs` | `dual_observation` | `ipfs_accelerate_py/agent_supervisor/proof/goal_tactician_lifecycle.py` | `{root}/goal_tactician_lifecycle.journal.jsonl + leanstral-goal-lifecycle.audit.jsonl` |
| `integration-tool-install-locks` | `os_bootstrap` | `lock` | `schema_deployment` | `legacy_retirement` | `ipfs_accelerate_py/agent_supervisor/integrations/contract_repair_dependencies.py` | `{managed_root}/*dependencies*.lock / agent-llm-resolver.lock` |
| `lease-coordination-duckdb` | `authority` | `duckdb` | `execution_lifecycle` | `foundation` | `ipfs_accelerate_py/agent_supervisor/merge/lease_coordination.py` | `{state_root}/coordination.duckdb` |
| `legacy-landed-review-duckdb` | `cache` | `duckdb` | `execution_lifecycle` | `legacy_retirement` | `ipfs_accelerate_py/agent_supervisor/todo_daemon/legacy_landed_result_cache.py` | `{root}/legacy_landed_review_results.duckdb` |
| `merge-checkpoint-json` | `authority` | `json` | `repository_worktree` | `default_cutover` | `ipfs_accelerate_py/agent_supervisor/merge/merge_checkpoint.py` | `{state}/merge_checkpoint.json` |
| `merge-queue-duckdb` | `authority` | `duckdb` | `repository_worktree` | `default_cutover` | `ipfs_accelerate_py/agent_supervisor/merge/merge_queue.py` | `{queue_dir}/merge_queue.duckdb (+ legacy merge_queue.sqlite3)` |
| `merge-resolver-duckdb` | `authority` | `duckdb` | `repository_worktree` | `default_cutover` | `ipfs_accelerate_py/agent_supervisor/merge/merge_resolver.py` | `{state_dir}/merge_resolver.duckdb (+ legacy .sqlite3)` |
| `merge-train-state-dir` | `authority` | `directory` | `repository_worktree` | `default_cutover` | `ipfs_accelerate_py/agent_supervisor/merge/merge_train.py` | `{queue}/train/{receipts,worktrees,gate-cache,consumer.lock}` |
| `objective-heap-markdown` | `authority` | `markdown` | `objectives_plans_tasks` | `database_authority_canary` | `ipfs_accelerate_py/agent_supervisor/objectives/objective_tracker.py` | `docs/architecture/*.objectives.md` |
| `persistent-task-queue-json` | `authority` | `json` | `execution_lifecycle` | `default_cutover` | `ipfs_accelerate_py/agent_supervisor/task_sources/persistent_task_queue.py` | `{daemon_state}/task_queue.json` |
| `plan-revision-store` | `authority` | `json` | `objectives_plans_tasks` | `default_cutover` | `ipfs_accelerate_py/agent_supervisor/task_sources/plan_revision_store.py` | `{root}/index.json + active.json + prior_active.json + events.jsonl + supersessions.jsonl + .plan-revision-store.lock` |
| `proof-carrying-workflow-duckdb` | `authority` | `duckdb` | `objectives_plans_tasks` | `default_cutover` | `ipfs_accelerate_py/agent_supervisor/planning/proof_carrying_planner.py` | `{store}/proof_carrying_workflow.duckdb (+ JSON twin)` |
| `proof-certificate-locks` | `os_bootstrap` | `lock` | `artifacts_proofs` | `default_cutover` | `ipfs_accelerate_py/agent_supervisor/proof/test_certificate_store.py` | `{root}/locks/{cas,fence}.lock (+ token locks)` |
| `proof-scheduler-duckdb` | `authority` | `duckdb` | `artifacts_proofs` | `default_cutover` | `ipfs_accelerate_py/agent_supervisor/proof/proof_scheduler.py` | `{root}/proof_scheduler.duckdb` |
| `prompt-workflow-duckdb-materialization` | `authority` | `duckdb` | `objectives_plans_tasks` | `foundation` | `ipfs_accelerate_py/agent_supervisor/prompt/prompt_workflow.py` | `{output_policy.duckdb_path} via DuckDBTaskSource` |
| `prover-evidence-duckdb` | `immutable_evidence` | `duckdb` | `artifacts_proofs` | `dual_observation` | `ipfs_accelerate_py/agent_supervisor/proof/prover_evidence_store.py` | `{root}/prover_evidence.duckdb (+ .json projection)` |
| `recovery-receipts-and-locks` | `emergency_diagnostic` | `directory` | `execution_lifecycle` | `default_cutover` | `ipfs_accelerate_py/agent_supervisor/rescue/supervisor_recovery.py` | `{state}/{receipts,incidents,quarantine,*.lock}` |
| `repository-index-lock` | `cache` | `lock` | `code_intelligence` | `legacy_retirement` | `ipfs_accelerate_py/agent_supervisor/analysis/repository_indexer.py` | `{index_root}/.repository-index.lock` |
| `run-registry-lock` | `os_bootstrap` | `lock` | `execution_lifecycle` | `default_cutover` | `ipfs_accelerate_py/agent_supervisor/entrypoints/run_registry.py` | `{registry_root}/.run-registry.lock (+ registry JSON rows)` |
| `runtime-event-log-jsonl` | `authority` | `jsonl` | `events_logs_metrics` | `default_cutover` | `ipfs_accelerate_py/agent_supervisor/runtime/event_log.py` | `{state}/events.jsonl (+ rotated archives)`
| `scheduler-config-json` | `static_input` | `json` | `schema_deployment` | `bootstrap` | `ipfs_accelerate_py/agent_supervisor/runtime/configured_board_scheduler.py` | `config/*_scheduler.json` |
| `taskboard-markdown` | `authority` | `markdown` | `objectives_plans_tasks` | `database_authority_canary` | `ipfs_accelerate_py/agent_supervisor/task_sources/markdown_task_source.py` | `docs/architecture/*.todo.md` |
| `taskboard-materialization-lock` | `os_bootstrap` | `lock` | `execution_lifecycle` | `default_cutover` | `ipfs_accelerate_py/agent_supervisor/task_sources/taskboard_store.py` | `.{board}.materialization.lock / .{board}.store.lock` |
| `taskboard-store-events` | `authority` | `jsonl` | `events_logs_metrics` | `default_cutover` | `ipfs_accelerate_py/agent_supervisor/task_sources/taskboard_store.py` | `{board}.events.jsonl + .{board}.store.lock` |
| `validation-rollout-pid-status` | `os_bootstrap` | `pid` | `execution_lifecycle` | `default_cutover` | `ipfs_accelerate_py/agent_supervisor/validation/logic_repair_rollout.py` | `{runtime}/master.pid + *_supervisor_status.json` |
| `worktree-lifecycle-records` | `authority` | `json` | `repository_worktree` | `default_cutover` | `ipfs_accelerate_py/agent_supervisor/merge/worktree_lifecycle.py` | `{state}/worktrees/*.json`

### Sink details

#### `accepted-work-ledger-jsonl`

- **Classification:** `immutable_evidence`
- **Media:** `jsonl`
- **Destination domain:** `events_logs_metrics`
- **Retirement stage:** `default_cutover`
- **Writer module:** `ipfs_accelerate_py/agent_supervisor/todo_daemon/artifacts.py`
- **Path template:** `{daemon_state}/accepted-work.jsonl (+ accepted_changes.jsonl)`
- **Atomicity model:** `append_only`
- **Direct DuckDB writer:** `no`
- **Git/source bytes:** `no`
- **Reuse candidate:** completion_receipts + domain_events append stream

#### `analysis-cache-lock`

- **Classification:** `cache`
- **Media:** `lock`
- **Destination domain:** `code_intelligence`
- **Retirement stage:** `legacy_retirement`
- **Writer module:** `ipfs_accelerate_py/agent_supervisor/analysis/analysis_cache.py`
- **Path template:** `{cache}/.analysis-cache.lock`
- **Atomicity model:** `os_process_handle`
- **Direct DuckDB writer:** `no`
- **Git/source bytes:** `no`
- **Reuse candidate:** parse_runs + source_snapshots with content digests

#### `analysis-program-cache`

- **Classification:** `cache`
- **Media:** `cache_dir`
- **Destination domain:** `code_intelligence`
- **Retirement stage:** `legacy_retirement`
- **Writer module:** `ipfs_accelerate_py/agent_supervisor/analysis/program_analysis_cache.py`
- **Path template:** `{cache_root}/program-analysis/**`
- **Atomicity model:** `best_effort_mirror`
- **Direct DuckDB writer:** `no`
- **Git/source bytes:** `no`
- **Reuse candidate:** source_snapshots + parse_runs keyed by tree id
- **Notes:** Accelerator only; never lease or completion authority.

#### `analysis-singleflight-sqlite`

- **Classification:** `cache`
- **Media:** `sqlite`
- **Destination domain:** `code_intelligence`
- **Retirement stage:** `legacy_retirement`
- **Writer module:** `ipfs_accelerate_py/agent_supervisor/analysis/cache_coordinator.py`
- **Path template:** `{cache}/single-flight.sqlite3`
- **Atomicity model:** `single_file_transaction`
- **Direct DuckDB writer:** `no`
- **Git/source bytes:** `no`
- **Reuse candidate:** Distributed single-flight via LeaseCoordinator

#### `artifact-store-json-duckdb`

- **Classification:** `immutable_evidence`
- **Media:** `artifact`
- **Destination domain:** `artifacts_proofs`
- **Retirement stage:** `dual_observation`
- **Writer module:** `ipfs_accelerate_py/agent_supervisor/runtime/artifact_store.py`
- **Path template:** `{artifact}.json + sidecar {artifact}.duckdb`
- **Atomicity model:** `cross_file_non_atomic`
- **Direct DuckDB writer:** `yes`
- **Git/source bytes:** `no`
- **Reuse candidate:** Bounded ArtifactStore dual JSON/DuckDB projection pattern
- **Cross-file atomicity gap:** JSON body and DuckDB sidecar are dual-written; crash can leave one side stale until rebuild
- **Notes:** JSON is portable interchange; DuckDB is a query projection.

#### `bundle-index-duckdb-sidecar`

- **Classification:** `cache`
- **Media:** `duckdb`
- **Destination domain:** `context_budgets`
- **Retirement stage:** `legacy_retirement`
- **Writer module:** `ipfs_accelerate_py/agent_supervisor/task_sources/todo_vector_index.py`
- **Path template:** `{bundle_index}.json + {bundle_index}.duckdb`
- **Atomicity model:** `cross_file_non_atomic`
- **Direct DuckDB writer:** `yes`
- **Git/source bytes:** `no`
- **Reuse candidate:** ArtifactStore dual-write + bounded query tables
- **Cross-file atomicity gap:** Bundle planning JSON and DuckDB sidecar may diverge after crash

#### `bundle-lane-manifest`

- **Classification:** `authority`
- **Media:** `json`
- **Destination domain:** `execution_lifecycle`
- **Retirement stage:** `default_cutover`
- **Writer module:** `ipfs_accelerate_py/agent_supervisor/objectives/bundle_supervisor.py`
- **Path template:** `{state_root}/bundle_lanes.json + scheduler_metrics.json`
- **Atomicity model:** `cross_file_non_atomic`
- **Direct DuckDB writer:** `no`
- **Git/source bytes:** `no`
- **Reuse candidate:** daemon_instances + scheduler metrics tables
- **Cross-file atomicity gap:** Lane manifest, metrics JSON, coordination.duckdb, and PID files are updated without a shared transaction

#### `checkout-mutation-lock`

- **Classification:** `os_bootstrap`
- **Media:** `lock`
- **Destination domain:** `repository_worktree`
- **Retirement stage:** `default_cutover`
- **Writer module:** `ipfs_accelerate_py/agent_supervisor/merge/checkout_lock.py`
- **Path template:** `{repo}/implementation-main-merge.lock (+ related locks)`
- **Atomicity model:** `os_process_handle`
- **Direct DuckDB writer:** `no`
- **Git/source bytes:** `no`
- **Reuse candidate:** path_claims + maintenance_leases

#### `control-audit-jsonl`

- **Classification:** `authority`
- **Media:** `jsonl`
- **Destination domain:** `events_logs_metrics`
- **Retirement stage:** `default_cutover`
- **Writer module:** `ipfs_accelerate_py/agent_supervisor/control/control_plane.py`
- **Path template:** `{state}/control-audit.jsonl + control-idempotency.jsonl + control-transactions.jsonl + supervisor-lifecycle-events.jsonl`
- **Atomicity model:** `append_only`
- **Direct DuckDB writer:** `no`
- **Git/source bytes:** `no`
- **Reuse candidate:** domain_events + idempotency_records streams
- **Cross-file atomicity gap:** Control audit, idempotency, transaction, and lifecycle JSONL files are independent append streams

#### `control-plane-migration-lock`

- **Classification:** `os_bootstrap`
- **Media:** `lock`
- **Destination domain:** `schema_deployment`
- **Retirement stage:** `foundation`
- **Writer module:** `ipfs_accelerate_py/agent_supervisor/task_sources/control_plane_migrations.py`
- **Path template:** `.{database}.migration.lock (+ control.duckdb bookkeeping)`
- **Atomicity model:** `os_process_handle`
- **Direct DuckDB writer:** `no`
- **Git/source bytes:** `no`
- **Reuse candidate:** Migration ownership rows + process-birth fencing; lock is bootstrap only around the state-owner
- **Notes:** Serializes checksum-bound migrations; not claim authority.

#### `daemon-pid-files`

- **Classification:** `os_bootstrap`
- **Media:** `pid`
- **Destination domain:** `execution_lifecycle`
- **Retirement stage:** `default_cutover`
- **Writer module:** `ipfs_accelerate_py/agent_supervisor/todo_daemon/wrapper.py`
- **Path template:** `{state}/*.pid`
- **Atomicity model:** `os_process_handle`
- **Direct DuckDB writer:** `no`
- **Git/source bytes:** `no`
- **Reuse candidate:** process-birth identity on daemon_sessions
- **Notes:** PID files never grant lease or completion authority.

#### `daemon-status-json`

- **Classification:** `os_bootstrap`
- **Media:** `json`
- **Destination domain:** `execution_lifecycle`
- **Retirement stage:** `default_cutover`
- **Writer module:** `ipfs_accelerate_py/agent_supervisor/todo_daemon/status.py`
- **Path template:** `{state}/*status.json / *.progress.json`
- **Atomicity model:** `single_file_atomic`
- **Direct DuckDB writer:** `no`
- **Git/source bytes:** `no`
- **Reuse candidate:** heartbeats + attempt_phases projections
- **Notes:** Mirrors lease/session state for operators; not claim authority.

#### `dataset-store-jsonl`

- **Classification:** `immutable_evidence`
- **Media:** `jsonl`
- **Destination domain:** `artifacts_proofs`
- **Retirement stage:** `dual_observation`
- **Writer module:** `ipfs_accelerate_py/agent_supervisor/task_sources/dataset_store.py`
- **Path template:** `{root}/{dataset_id}.jsonl (+ optional parquet)`
- **Atomicity model:** `single_file_atomic`
- **Direct DuckDB writer:** `no`
- **Git/source bytes:** `no`
- **Reuse candidate:** artifact metadata + content digests

#### `doctor-proof-cache-sqlite`

- **Classification:** `cache`
- **Media:** `sqlite`
- **Destination domain:** `artifacts_proofs`
- **Retirement stage:** `legacy_retirement`
- **Writer module:** `ipfs_accelerate_py/agent_supervisor/proof/doctor_proof_cache.py`
- **Path template:** `{root}/doctor_proof_cache.sqlite3 (caller path)`
- **Atomicity model:** `single_file_transaction`
- **Direct DuckDB writer:** `no`
- **Git/source bytes:** `no`
- **Reuse candidate:** Migrate onto formal_verification_cache.duckdb primitives
- **Notes:** Legacy SQLite proof memo; not orchestration authority.

#### `duckdb-state-primitives`

- **Classification:** `authority`
- **Media:** `duckdb`
- **Destination domain:** `schema_deployment`
- **Retirement stage:** `foundation`
- **Writer module:** `ipfs_accelerate_py/agent_supervisor/task_sources/duckdb_state.py`
- **Path template:** `{path}.duckdb + .{name}.lock (+ optional legacy .sqlite3)`
- **Atomicity model:** `flock_plus_transaction`
- **Direct DuckDB writer:** `yes`
- **Git/source bytes:** `no`
- **Reuse candidate:** Shared exclusive_file_lock + open_duckdb_connection for all single-writer DuckDB stores
- **Notes:** Not a store itself; primitives every DuckDB writer reuses.

#### `duckdb-task-source`

- **Classification:** `authority`
- **Media:** `duckdb`
- **Destination domain:** `objectives_plans_tasks`
- **Retirement stage:** `foundation`
- **Writer module:** `ipfs_accelerate_py/agent_supervisor/task_sources/duckdb_task_source.py`
- **Path template:** `{state_root}/tasks.duckdb (caller-supplied path)`
- **Atomicity model:** `flock_plus_transaction`
- **Direct DuckDB writer:** `yes`
- **Git/source bytes:** `no`
- **Reuse candidate:** Primary reuse target for control-plane tasks/claims schema via duckdb_state flock + short transactions
- **Notes:** Versioned fenced task projection; foundational for Quack cutover.

#### `formal-verification-cache-duckdb`

- **Classification:** `cache`
- **Media:** `duckdb`
- **Destination domain:** `artifacts_proofs`
- **Retirement stage:** `legacy_retirement`
- **Writer module:** `ipfs_accelerate_py/agent_supervisor/proof/formal_verification_cache.py`
- **Path template:** `{root}/formal_verification_cache.duckdb`
- **Atomicity model:** `flock_plus_transaction`
- **Direct DuckDB writer:** `yes`
- **Git/source bytes:** `no`
- **Reuse candidate:** Content-addressed proof cache keyed by obligation digest
- **Notes:** Cache only; never completion or proof authority by itself.

#### `git-source-bytes`

- **Classification:** `static_input`
- **Media:** `git_source`
- **Destination domain:** `non_state`
- **Retirement stage:** `retain_permanent`
- **Writer module:** `ipfs_accelerate_py/agent_supervisor/analysis/repository_forest.py`
- **Path template:** `{repo_root}/.git/** and worktree source files`
- **Atomicity model:** `not_applicable`
- **Direct DuckDB writer:** `no`
- **Git/source bytes:** `yes`
- **Reuse candidate:** Repository forest + worktree snapshot identities; never store source blobs as control-plane authority
- **Notes:** Git remains the byte authority for source. The control plane stores identities, AST, and mutation history only.

#### `goal-tactician-lifecycle-jsonl`

- **Classification:** `immutable_evidence`
- **Media:** `jsonl`
- **Destination domain:** `artifacts_proofs`
- **Retirement stage:** `dual_observation`
- **Writer module:** `ipfs_accelerate_py/agent_supervisor/proof/goal_tactician_lifecycle.py`
- **Path template:** `{root}/goal_tactician_lifecycle.journal.jsonl + leanstral-goal-lifecycle.audit.jsonl`
- **Atomicity model:** `append_only`
- **Direct DuckDB writer:** `no`
- **Git/source bytes:** `no`
- **Reuse candidate:** domain_events + proof_attempts audit streams

#### `integration-tool-install-locks`

- **Classification:** `os_bootstrap`
- **Media:** `lock`
- **Destination domain:** `schema_deployment`
- **Retirement stage:** `legacy_retirement`
- **Writer module:** `ipfs_accelerate_py/agent_supervisor/integrations/contract_repair_dependencies.py`
- **Path template:** `{managed_root}/*dependencies*.lock / agent-llm-resolver.lock`
- **Atomicity model:** `os_process_handle`
- **Direct DuckDB writer:** `no`
- **Git/source bytes:** `no`
- **Reuse candidate:** capability_snapshots + digest-bound toolchain installs
- **Notes:** Tool-install single-flight only; never orchestration authority.

#### `lease-coordination-duckdb`

- **Classification:** `authority`
- **Media:** `duckdb`
- **Destination domain:** `execution_lifecycle`
- **Retirement stage:** `foundation`
- **Writer module:** `ipfs_accelerate_py/agent_supervisor/merge/lease_coordination.py`
- **Path template:** `{state_root}/coordination.duckdb`
- **Atomicity model:** `flock_plus_transaction`
- **Direct DuckDB writer:** `yes`
- **Git/source bytes:** `no`
- **Reuse candidate:** LeaseCoordinator fencing/CAS model becomes control-plane leases

#### `legacy-landed-review-duckdb`

- **Classification:** `cache`
- **Media:** `duckdb`
- **Destination domain:** `execution_lifecycle`
- **Retirement stage:** `legacy_retirement`
- **Writer module:** `ipfs_accelerate_py/agent_supervisor/todo_daemon/legacy_landed_result_cache.py`
- **Path template:** `{root}/legacy_landed_review_results.duckdb`
- **Atomicity model:** `flock_plus_transaction`
- **Direct DuckDB writer:** `yes`
- **Git/source bytes:** `no`
- **Reuse candidate:** Attempt result projection tables

#### `merge-checkpoint-json`

- **Classification:** `authority`
- **Media:** `json`
- **Destination domain:** `repository_worktree`
- **Retirement stage:** `default_cutover`
- **Writer module:** `ipfs_accelerate_py/agent_supervisor/merge/merge_checkpoint.py`
- **Path template:** `{state}/merge_checkpoint.json`
- **Atomicity model:** `single_file_atomic`
- **Direct DuckDB writer:** `no`
- **Git/source bytes:** `no`
- **Reuse candidate:** merge_attempts checkpoint columns

#### `merge-queue-duckdb`

- **Classification:** `authority`
- **Media:** `duckdb`
- **Destination domain:** `repository_worktree`
- **Retirement stage:** `default_cutover`
- **Writer module:** `ipfs_accelerate_py/agent_supervisor/merge/merge_queue.py`
- **Path template:** `{queue_dir}/merge_queue.duckdb (+ legacy merge_queue.sqlite3)`
- **Atomicity model:** `flock_plus_transaction`
- **Direct DuckDB writer:** `yes`
- **Git/source bytes:** `no`
- **Reuse candidate:** merge_queue_entries + resource_claims tables
- **Cross-file atomicity gap:** Queue DB, train receipts directory, and checkout lock files are independent writers

#### `merge-resolver-duckdb`

- **Classification:** `authority`
- **Media:** `duckdb`
- **Destination domain:** `repository_worktree`
- **Retirement stage:** `default_cutover`
- **Writer module:** `ipfs_accelerate_py/agent_supervisor/merge/merge_resolver.py`
- **Path template:** `{state_dir}/merge_resolver.duckdb (+ legacy .sqlite3)`
- **Atomicity model:** `flock_plus_transaction`
- **Direct DuckDB writer:** `yes`
- **Git/source bytes:** `no`
- **Reuse candidate:** merge_attempts domain events + MergeQueue fence

#### `merge-train-state-dir`

- **Classification:** `authority`
- **Media:** `directory`
- **Destination domain:** `repository_worktree`
- **Retirement stage:** `default_cutover`
- **Writer module:** `ipfs_accelerate_py/agent_supervisor/merge/merge_train.py`
- **Path template:** `{queue}/train/{receipts,worktrees,gate-cache,consumer.lock}`
- **Atomicity model:** `cross_file_non_atomic`
- **Direct DuckDB writer:** `no`
- **Git/source bytes:** `no`
- **Reuse candidate:** merge_attempts + validation_runs + worktree rows
- **Cross-file atomicity gap:** Train receipts, consumer lock, gate cache, and merge queue DB are not one atomic unit

#### `objective-heap-markdown`

- **Classification:** `authority`
- **Media:** `markdown`
- **Destination domain:** `objectives_plans_tasks`
- **Retirement stage:** `database_authority_canary`
- **Writer module:** `ipfs_accelerate_py/agent_supervisor/objectives/objective_tracker.py`
- **Path template:** `docs/architecture/*.objectives.md`
- **Atomicity model:** `cross_file_non_atomic`
- **Direct DuckDB writer:** `no`
- **Git/source bytes:** `no`
- **Reuse candidate:** objective_graph + goal_completion identity/revision contracts
- **Cross-file atomicity gap:** Objective heap rewrites and paired taskboard/status updates are not one transaction; crash can diverge intent vs schedule
- **Notes:** Durable intent authority today; becomes import + export only.

#### `persistent-task-queue-json`

- **Classification:** `authority`
- **Media:** `json`
- **Destination domain:** `execution_lifecycle`
- **Retirement stage:** `default_cutover`
- **Writer module:** `ipfs_accelerate_py/agent_supervisor/task_sources/persistent_task_queue.py`
- **Path template:** `{daemon_state}/task_queue.json`
- **Atomicity model:** `single_file_atomic`
- **Direct DuckDB writer:** `no`
- **Git/source bytes:** `no`
- **Reuse candidate:** task_claims + selection penalty columns

#### `plan-revision-store`

- **Classification:** `authority`
- **Media:** `json`
- **Destination domain:** `objectives_plans_tasks`
- **Retirement stage:** `default_cutover`
- **Writer module:** `ipfs_accelerate_py/agent_supervisor/task_sources/plan_revision_store.py`
- **Path template:** `{root}/index.json + active.json + prior_active.json + events.jsonl + supersessions.jsonl + .plan-revision-store.lock`
- **Atomicity model:** `cross_file_non_atomic`
- **Direct DuckDB writer:** `no`
- **Git/source bytes:** `no`
- **Reuse candidate:** plan_revisions + planning_decisions tables with CAS
- **Cross-file atomicity gap:** Active projection, index, and event log are separate files; recovery rebuilds active from events but crash windows exist

#### `proof-carrying-workflow-duckdb`

- **Classification:** `authority`
- **Media:** `duckdb`
- **Destination domain:** `objectives_plans_tasks`
- **Retirement stage:** `default_cutover`
- **Writer module:** `ipfs_accelerate_py/agent_supervisor/planning/proof_carrying_planner.py`
- **Path template:** `{store}/proof_carrying_workflow.duckdb (+ JSON twin)`
- **Atomicity model:** `cross_file_non_atomic`
- **Direct DuckDB writer:** `yes`
- **Git/source bytes:** `no`
- **Reuse candidate:** plan_revisions + ArtifactStore dual JSON/DuckDB pattern
- **Cross-file atomicity gap:** Proof-carrying planner JSON and DuckDB twin are dual-written

#### `proof-certificate-locks`

- **Classification:** `os_bootstrap`
- **Media:** `lock`
- **Destination domain:** `artifacts_proofs`
- **Retirement stage:** `default_cutover`
- **Writer module:** `ipfs_accelerate_py/agent_supervisor/proof/test_certificate_store.py`
- **Path template:** `{root}/locks/{cas,fence}.lock (+ token locks)`
- **Atomicity model:** `os_process_handle`
- **Direct DuckDB writer:** `no`
- **Git/source bytes:** `no`
- **Reuse candidate:** artifact retention leases + content-addressed receipts

#### `proof-scheduler-duckdb`

- **Classification:** `authority`
- **Media:** `duckdb`
- **Destination domain:** `artifacts_proofs`
- **Retirement stage:** `default_cutover`
- **Writer module:** `ipfs_accelerate_py/agent_supervisor/proof/proof_scheduler.py`
- **Path template:** `{root}/proof_scheduler.duckdb`
- **Atomicity model:** `flock_plus_transaction`
- **Direct DuckDB writer:** `yes`
- **Git/source bytes:** `no`
- **Reuse candidate:** duckdb_state resolve_duckdb_path + flock pattern

#### `prompt-workflow-duckdb-materialization`

- **Classification:** `authority`
- **Media:** `duckdb`
- **Destination domain:** `objectives_plans_tasks`
- **Retirement stage:** `foundation`
- **Writer module:** `ipfs_accelerate_py/agent_supervisor/prompt/prompt_workflow.py`
- **Path template:** `{output_policy.duckdb_path} via DuckDBTaskSource`
- **Atomicity model:** `flock_plus_transaction`
- **Direct DuckDB writer:** `yes`
- **Git/source bytes:** `no`
- **Reuse candidate:** DuckDBTaskSource materializer already used by prompt workflow
- **Notes:** Prompt workflow reuses DuckDBTaskSource; scanner must classify path markers.

#### `prover-evidence-duckdb`

- **Classification:** `immutable_evidence`
- **Media:** `duckdb`
- **Destination domain:** `artifacts_proofs`
- **Retirement stage:** `dual_observation`
- **Writer module:** `ipfs_accelerate_py/agent_supervisor/proof/prover_evidence_store.py`
- **Path template:** `{root}/prover_evidence.duckdb (+ .json projection)`
- **Atomicity model:** `flock_plus_transaction`
- **Direct DuckDB writer:** `yes`
- **Git/source bytes:** `no`
- **Reuse candidate:** Artifact metadata tables + content-addressed receipts

#### `recovery-receipts-and-locks`

- **Classification:** `emergency_diagnostic`
- **Media:** `directory`
- **Destination domain:** `execution_lifecycle`
- **Retirement stage:** `default_cutover`
- **Writer module:** `ipfs_accelerate_py/agent_supervisor/rescue/supervisor_recovery.py`
- **Path template:** `{state}/{receipts,incidents,quarantine,*.lock}`
- **Atomicity model:** `cross_file_non_atomic`
- **Direct DuckDB writer:** `no`
- **Git/source bytes:** `no`
- **Reuse candidate:** recovery_actions + quarantine tables
- **Cross-file atomicity gap:** Incident locks, receipts, and quarantine directories update independently of lease/task authority files
- **Notes:** Emergency path; must not become a second authority plane.

#### `repository-index-lock`

- **Classification:** `cache`
- **Media:** `lock`
- **Destination domain:** `code_intelligence`
- **Retirement stage:** `legacy_retirement`
- **Writer module:** `ipfs_accelerate_py/agent_supervisor/analysis/repository_indexer.py`
- **Path template:** `{index_root}/.repository-index.lock`
- **Atomicity model:** `os_process_handle`
- **Direct DuckDB writer:** `no`
- **Git/source bytes:** `no`
- **Reuse candidate:** repository_revisions + source_files index tables

#### `run-registry-lock`

- **Classification:** `os_bootstrap`
- **Media:** `lock`
- **Destination domain:** `execution_lifecycle`
- **Retirement stage:** `default_cutover`
- **Writer module:** `ipfs_accelerate_py/agent_supervisor/entrypoints/run_registry.py`
- **Path template:** `{registry_root}/.run-registry.lock (+ registry JSON rows)`
- **Atomicity model:** `os_process_handle`
- **Direct DuckDB writer:** `no`
- **Git/source bytes:** `no`
- **Reuse candidate:** client_sessions + resource_claims under Quack
- **Notes:** Serializes run-registry mutations; not claim authority.

#### `runtime-event-log-jsonl`

- **Classification:** `authority`
- **Media:** `jsonl`
- **Destination domain:** `events_logs_metrics`
- **Retirement stage:** `default_cutover`
- **Writer module:** `ipfs_accelerate_py/agent_supervisor/runtime/event_log.py`
- **Path template:** `{state}/events.jsonl (+ rotated archives)`
- **Atomicity model:** `append_only`
- **Direct DuckDB writer:** `no`
- **Git/source bytes:** `no`
- **Reuse candidate:** domain_events append stream with monotonic sequence

#### `scheduler-config-json`

- **Classification:** `static_input`
- **Media:** `json`
- **Destination domain:** `schema_deployment`
- **Retirement stage:** `bootstrap`
- **Writer module:** `ipfs_accelerate_py/agent_supervisor/runtime/configured_board_scheduler.py`
- **Path template:** `config/*_scheduler.json`
- **Atomicity model:** `not_applicable`
- **Direct DuckDB writer:** `no`
- **Git/source bytes:** `no`
- **Reuse candidate:** launch policy rows imported once with provenance
- **Notes:** Operator-sealed static input; not rewritten by daemons.

#### `taskboard-markdown`

- **Classification:** `authority`
- **Media:** `markdown`
- **Destination domain:** `objectives_plans_tasks`
- **Retirement stage:** `database_authority_canary`
- **Writer module:** `ipfs_accelerate_py/agent_supervisor/task_sources/markdown_task_source.py`
- **Path template:** `docs/architecture/*.todo.md`
- **Atomicity model:** `cross_file_non_atomic`
- **Direct DuckDB writer:** `no`
- **Git/source bytes:** `no`
- **Reuse candidate:** TaskboardStore materialization journal + DuckDBTaskSource fenced projection
- **Cross-file atomicity gap:** Board status writes, events.jsonl, and daemon status JSON are separate files without a shared transaction
- **Notes:** Schedulable projection; still treated as claim authority pre-cutover.

#### `taskboard-materialization-lock`

- **Classification:** `os_bootstrap`
- **Media:** `lock`
- **Destination domain:** `execution_lifecycle`
- **Retirement stage:** `default_cutover`
- **Writer module:** `ipfs_accelerate_py/agent_supervisor/task_sources/taskboard_store.py`
- **Path template:** `.{board}.materialization.lock / .{board}.store.lock`
- **Atomicity model:** `os_process_handle`
- **Direct DuckDB writer:** `no`
- **Git/source bytes:** `no`
- **Reuse candidate:** Process-birth lease + fencing epoch in control plane
- **Notes:** Lock files serialize writers; they are not claim authority.

#### `taskboard-store-events`

- **Classification:** `authority`
- **Media:** `jsonl`
- **Destination domain:** `events_logs_metrics`
- **Retirement stage:** `default_cutover`
- **Writer module:** `ipfs_accelerate_py/agent_supervisor/task_sources/taskboard_store.py`
- **Path template:** `{board}.events.jsonl + .{board}.store.lock`
- **Atomicity model:** `append_only`
- **Direct DuckDB writer:** `no`
- **Git/source bytes:** `no`
- **Reuse candidate:** RotatingEventLog / domain_events stream
- **Cross-file atomicity gap:** Event append and markdown board rewrite are not atomic together

#### `validation-rollout-pid-status`

- **Classification:** `os_bootstrap`
- **Media:** `pid`
- **Destination domain:** `execution_lifecycle`
- **Retirement stage:** `default_cutover`
- **Writer module:** `ipfs_accelerate_py/agent_supervisor/validation/logic_repair_rollout.py`
- **Path template:** `{runtime}/master.pid + *_supervisor_status.json`
- **Atomicity model:** `os_process_handle`
- **Direct DuckDB writer:** `no`
- **Git/source bytes:** `no`
- **Reuse candidate:** daemon_sessions process-birth identity + heartbeats

#### `worktree-lifecycle-records`

- **Classification:** `authority`
- **Media:** `json`
- **Destination domain:** `repository_worktree`
- **Retirement stage:** `default_cutover`
- **Writer module:** `ipfs_accelerate_py/agent_supervisor/merge/worktree_lifecycle.py`
- **Path template:** `{state}/worktrees/*.json`
- **Atomicity model:** `single_file_atomic`
- **Direct DuckDB writer:** `no`
- **Git/source bytes:** `no`
- **Reuse candidate:** worktrees + leases tables with process-birth fencing

## Scanner contract

```text
python scripts/ops/agent_supervisor/inventory_state_sinks.py --check
python -m pytest -q test/api/test_agent_supervisor_state_sink_inventory.py
```

`--check` exits non-zero when any discovered mutable path marker is
not covered by this catalog. Adding a new DuckDB/JSONL/PID/lock/status
writer without classifying it fails CI.

Source-tree scan globs such as `*.json`, `*.py`, or `*.pem` are **not**
mutable supervisor sinks; only sink-family globs (`*.duckdb`, `*.jsonl`,
`*.pid`, `*.lock`, `*.todo.md`, `*.objectives.md`, and status/progress
patterns) are admitted into discovery. Package-manager lockfiles and Git
index/ref locks (`index.lock`, `HEAD.lock`, …) are also excluded as
Git/source inputs rather than orchestration state.

## Non-goals

- This inventory does not migrate state.
- This inventory is not completion, proof, or lease authority.
- Source code and Git objects stay on the filesystem; only their
  identities and derived structure enter the control plane.
