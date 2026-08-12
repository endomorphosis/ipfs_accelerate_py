-- DQP-005: normalized agent-supervisor control-plane schema (ControlPlaneSchema@1).
-- Domain SQL for control.duckdb. Bookkeeping tables (control_plane_metadata,
-- schema_migrations, schema_migration_attempts) are installed by the migration
-- runner before this file runs and must not be redefined here.
--
-- Domains: meta, intent, schedule, runtime, git, code, evidence, cache, control, improve.
-- Join-critical identities are first-class columns; extension_json is never a join key.

-- ---------------------------------------------------------------------------
-- meta: schema contracts and store generation surface
-- ---------------------------------------------------------------------------
CREATE TABLE schema_contracts (
    contract_id VARCHAR PRIMARY KEY,
    interface_name VARCHAR NOT NULL,
    domain_name VARCHAR NOT NULL,
    schema_revision BIGINT NOT NULL,
    payload_schema VARCHAR NOT NULL,
    description VARCHAR NOT NULL,
    created_at VARCHAR NOT NULL,
    extension_schema VARCHAR NOT NULL DEFAULT '',
    extension_json VARCHAR NOT NULL DEFAULT '{}'
);
CREATE UNIQUE INDEX schema_contracts_interface_uidx
    ON schema_contracts(interface_name, domain_name, schema_revision);

CREATE TABLE store_generations (
    generation BIGINT PRIMARY KEY,
    schema_revision BIGINT NOT NULL,
    fence_epoch BIGINT NOT NULL,
    revision BIGINT NOT NULL,
    database_uuid VARCHAR NOT NULL,
    birth_id VARCHAR NOT NULL,
    created_at VARCHAR NOT NULL,
    extension_schema VARCHAR NOT NULL DEFAULT '',
    extension_json VARCHAR NOT NULL DEFAULT '{}'
);

-- ---------------------------------------------------------------------------
-- control: deployment, servers, sessions, auth, maintenance
-- ---------------------------------------------------------------------------
CREATE TABLE state_servers (
    server_id VARCHAR PRIMARY KEY,
    store_id VARCHAR NOT NULL,
    database_uuid VARCHAR NOT NULL,
    process_birth_id VARCHAR NOT NULL,
    listen_uri VARCHAR NOT NULL,
    extension_fingerprint VARCHAR NOT NULL,
    schema_revision BIGINT NOT NULL,
    generation BIGINT NOT NULL,
    started_at VARCHAR NOT NULL,
    stopped_at VARCHAR,
    status VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    extension_schema VARCHAR NOT NULL DEFAULT '',
    extension_json VARCHAR NOT NULL DEFAULT '{}'
);
CREATE UNIQUE INDEX state_servers_birth_uidx
    ON state_servers(process_birth_id);

CREATE TABLE server_epochs (
    server_id VARCHAR NOT NULL,
    epoch BIGINT NOT NULL,
    fence_epoch BIGINT NOT NULL,
    started_at VARCHAR NOT NULL,
    ended_at VARCHAR,
    PRIMARY KEY (server_id, epoch)
);

CREATE TABLE client_sessions (
    session_id VARCHAR PRIMARY KEY,
    server_id VARCHAR NOT NULL,
    owner_id VARCHAR NOT NULL,
    process_birth_id VARCHAR NOT NULL,
    attached_at VARCHAR NOT NULL,
    last_seen_at VARCHAR NOT NULL,
    fence_epoch BIGINT NOT NULL,
    generation BIGINT NOT NULL,
    status VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    extension_schema VARCHAR NOT NULL DEFAULT '',
    extension_json VARCHAR NOT NULL DEFAULT '{}'
);
CREATE INDEX client_sessions_server_idx
    ON client_sessions(server_id, status);

CREATE TABLE capability_snapshots (
    snapshot_id VARCHAR PRIMARY KEY,
    server_id VARCHAR NOT NULL,
    profile_id VARCHAR NOT NULL,
    duckdb_version VARCHAR NOT NULL,
    extension_name VARCHAR NOT NULL,
    extension_fingerprint VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    observed_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);
CREATE INDEX capability_snapshots_server_idx
    ON capability_snapshots(server_id, observed_at);

CREATE TABLE credentials (
    credential_id VARCHAR PRIMARY KEY,
    secret_handle VARCHAR NOT NULL,
    generation BIGINT NOT NULL,
    purpose VARCHAR NOT NULL,
    created_at VARCHAR NOT NULL,
    rotated_at VARCHAR,
    revoked_at VARCHAR,
    revision BIGINT NOT NULL
);
CREATE UNIQUE INDEX credentials_handle_generation_uidx
    ON credentials(secret_handle, generation);

CREATE TABLE authorization_roles (
    role_id VARCHAR PRIMARY KEY,
    role_name VARCHAR NOT NULL UNIQUE,
    description VARCHAR NOT NULL,
    created_at VARCHAR NOT NULL,
    revision BIGINT NOT NULL
);

CREATE TABLE authorization_grants (
    grant_id VARCHAR PRIMARY KEY,
    role_id VARCHAR NOT NULL,
    principal_id VARCHAR NOT NULL,
    scope VARCHAR NOT NULL,
    granted_at VARCHAR NOT NULL,
    expires_at VARCHAR,
    revoked_at VARCHAR,
    revision BIGINT NOT NULL
);
CREATE INDEX authorization_grants_principal_idx
    ON authorization_grants(principal_id, role_id);

CREATE TABLE backup_snapshots (
    backup_id VARCHAR PRIMARY KEY,
    store_id VARCHAR NOT NULL,
    database_uuid VARCHAR NOT NULL,
    schema_revision BIGINT NOT NULL,
    generation BIGINT NOT NULL,
    artifact_digest VARCHAR NOT NULL,
    created_at VARCHAR NOT NULL,
    destination_uri VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE restore_receipts (
    receipt_id VARCHAR PRIMARY KEY,
    backup_id VARCHAR NOT NULL,
    store_id VARCHAR NOT NULL,
    restored_at VARCHAR NOT NULL,
    schema_revision BIGINT NOT NULL,
    generation BIGINT NOT NULL,
    outcome VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE maintenance_leases (
    lease_id VARCHAR PRIMARY KEY,
    scope VARCHAR NOT NULL,
    owner_session_id VARCHAR NOT NULL,
    process_birth_id VARCHAR NOT NULL,
    fencing_token BIGINT NOT NULL,
    fence_epoch BIGINT NOT NULL,
    acquired_at VARCHAR NOT NULL,
    expires_at VARCHAR NOT NULL,
    released_at VARCHAR,
    state VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    extension_schema VARCHAR NOT NULL DEFAULT '',
    extension_json VARCHAR NOT NULL DEFAULT '{}'
);
CREATE UNIQUE INDEX maintenance_leases_scope_active_uidx
    ON maintenance_leases(scope, state, fencing_token);

-- ---------------------------------------------------------------------------
-- git: repositories, worktrees, refs, merge queue, path claims
-- ---------------------------------------------------------------------------
CREATE TABLE repositories (
    repository_id VARCHAR PRIMARY KEY,
    canonical_root VARCHAR NOT NULL,
    git_common_dir VARCHAR NOT NULL,
    head_commit_id VARCHAR NOT NULL,
    created_at VARCHAR NOT NULL,
    updated_at VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    extension_schema VARCHAR NOT NULL DEFAULT '',
    extension_json VARCHAR NOT NULL DEFAULT '{}'
);

CREATE TABLE repository_revisions (
    repository_id VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    commit_id VARCHAR NOT NULL,
    tree_id VARCHAR NOT NULL,
    observed_at VARCHAR NOT NULL,
    PRIMARY KEY (repository_id, revision)
);

CREATE TABLE submodule_edges (
    parent_repository_id VARCHAR NOT NULL,
    child_repository_id VARCHAR NOT NULL,
    path VARCHAR NOT NULL,
    gitlink_commit_id VARCHAR NOT NULL,
    PRIMARY KEY (parent_repository_id, path)
);

CREATE TABLE worktrees (
    worktree_id VARCHAR PRIMARY KEY,
    repository_id VARCHAR NOT NULL,
    path VARCHAR NOT NULL,
    head_commit_id VARCHAR NOT NULL,
    branch_name VARCHAR NOT NULL,
    owner_session_id VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    created_at VARCHAR NOT NULL,
    updated_at VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    fence_epoch BIGINT NOT NULL,
    extension_schema VARCHAR NOT NULL DEFAULT '',
    extension_json VARCHAR NOT NULL DEFAULT '{}'
);
CREATE UNIQUE INDEX worktrees_path_uidx ON worktrees(path);
CREATE INDEX worktrees_repository_idx ON worktrees(repository_id, status);

CREATE TABLE worktree_snapshots (
    snapshot_id VARCHAR PRIMARY KEY,
    worktree_id VARCHAR NOT NULL,
    repository_id VARCHAR NOT NULL,
    head_commit_id VARCHAR NOT NULL,
    index_digest VARCHAR NOT NULL,
    overlay_digest VARCHAR NOT NULL,
    observed_at VARCHAR NOT NULL,
    scanner_version VARCHAR NOT NULL
);
CREATE INDEX worktree_snapshots_worktree_idx
    ON worktree_snapshots(worktree_id, observed_at);

CREATE TABLE worktree_paths (
    worktree_id VARCHAR NOT NULL,
    path VARCHAR NOT NULL,
    path_digest VARCHAR NOT NULL,
    file_mode VARCHAR NOT NULL,
    blob_id VARCHAR NOT NULL,
    PRIMARY KEY (worktree_id, path)
);

CREATE TABLE dirty_overlays (
    overlay_id VARCHAR PRIMARY KEY,
    worktree_id VARCHAR NOT NULL,
    path VARCHAR NOT NULL,
    before_blob_id VARCHAR NOT NULL,
    after_blob_id VARCHAR NOT NULL,
    overlay_digest VARCHAR NOT NULL,
    observed_at VARCHAR NOT NULL
);
CREATE INDEX dirty_overlays_worktree_idx ON dirty_overlays(worktree_id, path);

CREATE TABLE branches (
    repository_id VARCHAR NOT NULL,
    branch_name VARCHAR NOT NULL,
    tip_commit_id VARCHAR NOT NULL,
    updated_at VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    PRIMARY KEY (repository_id, branch_name)
);

CREATE TABLE git_refs (
    repository_id VARCHAR NOT NULL,
    ref_name VARCHAR NOT NULL,
    object_id VARCHAR NOT NULL,
    ref_type VARCHAR NOT NULL,
    updated_at VARCHAR NOT NULL,
    PRIMARY KEY (repository_id, ref_name)
);

CREATE TABLE merge_bases (
    left_commit_id VARCHAR NOT NULL,
    right_commit_id VARCHAR NOT NULL,
    merge_base_commit_id VARCHAR NOT NULL,
    computed_at VARCHAR NOT NULL,
    PRIMARY KEY (left_commit_id, right_commit_id)
);

CREATE TABLE merge_queue_entries (
    entry_id VARCHAR PRIMARY KEY,
    repository_id VARCHAR NOT NULL,
    worktree_id VARCHAR NOT NULL,
    task_cid VARCHAR NOT NULL,
    source_branch VARCHAR NOT NULL,
    target_branch VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    ordinal BIGINT NOT NULL,
    enqueued_at VARCHAR NOT NULL,
    updated_at VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    fence_epoch BIGINT NOT NULL
);
CREATE INDEX merge_queue_entries_repo_status_idx
    ON merge_queue_entries(repository_id, status, ordinal);
CREATE INDEX merge_queue_entries_task_idx
    ON merge_queue_entries(task_cid);

CREATE TABLE resource_claims (
    claim_id VARCHAR PRIMARY KEY,
    resource_kind VARCHAR NOT NULL,
    resource_id VARCHAR NOT NULL,
    owner_session_id VARCHAR NOT NULL,
    task_cid VARCHAR NOT NULL,
    fencing_token BIGINT NOT NULL,
    fence_epoch BIGINT NOT NULL,
    acquired_at VARCHAR NOT NULL,
    expires_at VARCHAR NOT NULL,
    state VARCHAR NOT NULL,
    revision BIGINT NOT NULL
);
CREATE INDEX resource_claims_resource_idx
    ON resource_claims(resource_kind, resource_id, state);

CREATE TABLE path_claims (
    claim_id VARCHAR PRIMARY KEY,
    repository_id VARCHAR NOT NULL,
    worktree_id VARCHAR NOT NULL,
    path VARCHAR NOT NULL,
    owner_session_id VARCHAR NOT NULL,
    task_cid VARCHAR NOT NULL,
    fencing_token BIGINT NOT NULL,
    fence_epoch BIGINT NOT NULL,
    acquired_at VARCHAR NOT NULL,
    expires_at VARCHAR NOT NULL,
    state VARCHAR NOT NULL,
    revision BIGINT NOT NULL
);
CREATE INDEX path_claims_path_idx
    ON path_claims(repository_id, path, state);
CREATE INDEX path_claims_task_idx ON path_claims(task_cid);

-- leases preserves existing LeaseCoordinator semantics (task_cid PK, fencing).
CREATE TABLE leases (
    task_cid VARCHAR PRIMARY KEY,
    claim_cid VARCHAR NOT NULL,
    resolution_cid VARCHAR NOT NULL,
    claimant_did VARCHAR NOT NULL,
    logical_epoch BIGINT NOT NULL,
    fencing_token BIGINT NOT NULL,
    expires_at_ms BIGINT NOT NULL,
    attempt BIGINT NOT NULL,
    state VARCHAR NOT NULL,
    started_at_ms BIGINT NOT NULL,
    release_reason VARCHAR,
    retry_not_before_ms BIGINT NOT NULL DEFAULT 0,
    owner_session_id VARCHAR NOT NULL DEFAULT '',
    fence_epoch BIGINT NOT NULL DEFAULT 0,
    revision BIGINT NOT NULL DEFAULT 0,
    extension_schema VARCHAR NOT NULL DEFAULT '',
    extension_json VARCHAR NOT NULL DEFAULT '{}'
);
CREATE INDEX leases_scheduler_state_idx
    ON leases(state, expires_at_ms, retry_not_before_ms);
CREATE INDEX leases_claimant_idx ON leases(claimant_did, state);

CREATE TABLE lease_events (
    event_id VARCHAR PRIMARY KEY,
    task_cid VARCHAR NOT NULL,
    claim_cid VARCHAR NOT NULL,
    event_type VARCHAR NOT NULL,
    fencing_token BIGINT NOT NULL,
    observed_at_ms BIGINT NOT NULL,
    body_json VARCHAR NOT NULL
);
CREATE INDEX lease_events_task_idx ON lease_events(task_cid, observed_at_ms);

CREATE TABLE token_history (
    task_cid VARCHAR NOT NULL,
    fencing_token BIGINT NOT NULL,
    recorded_at_ms BIGINT NOT NULL DEFAULT 0,
    PRIMARY KEY (task_cid, fencing_token)
);

-- ---------------------------------------------------------------------------
-- intent: objectives, goals, plans, tasks (task_cid preserved)
-- ---------------------------------------------------------------------------
CREATE TABLE objectives (
    objective_id VARCHAR PRIMARY KEY,
    objective_alias VARCHAR NOT NULL UNIQUE,
    parent_objective_id VARCHAR NOT NULL DEFAULT '',
    title VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    priority VARCHAR NOT NULL,
    created_at VARCHAR NOT NULL,
    updated_at VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    body_json VARCHAR NOT NULL,
    extension_schema VARCHAR NOT NULL DEFAULT '',
    extension_json VARCHAR NOT NULL DEFAULT '{}'
);

CREATE TABLE objective_revisions (
    objective_id VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    status VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL,
    recorded_at VARCHAR NOT NULL,
    PRIMARY KEY (objective_id, revision)
);

CREATE TABLE goals (
    goal_cid VARCHAR PRIMARY KEY,
    goal_alias VARCHAR NOT NULL UNIQUE,
    objective_id VARCHAR NOT NULL DEFAULT '',
    parent_goal_cid VARCHAR NOT NULL DEFAULT '',
    ordinal BIGINT NOT NULL,
    title VARCHAR NOT NULL,
    status VARCHAR NOT NULL DEFAULT 'open',
    created_at VARCHAR NOT NULL DEFAULT '',
    updated_at VARCHAR NOT NULL DEFAULT '',
    revision BIGINT NOT NULL DEFAULT 0,
    body_json VARCHAR NOT NULL
);
CREATE INDEX goals_objective_idx ON goals(objective_id, ordinal);

CREATE TABLE goal_edges (
    parent_goal_cid VARCHAR NOT NULL,
    child_goal_cid VARCHAR NOT NULL,
    edge_kind VARCHAR NOT NULL,
    PRIMARY KEY (parent_goal_cid, child_goal_cid, edge_kind)
);

CREATE TABLE plans (
    plan_cid VARCHAR PRIMARY KEY,
    goal_cid VARCHAR NOT NULL,
    plan_alias VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    created_at VARCHAR NOT NULL,
    updated_at VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    body_json VARCHAR NOT NULL
);
CREATE INDEX plans_goal_idx ON plans(goal_cid, status);

CREATE TABLE plan_revisions (
    plan_cid VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    body_json VARCHAR NOT NULL,
    recorded_at VARCHAR NOT NULL,
    PRIMARY KEY (plan_cid, revision)
);

CREATE TABLE planning_decisions (
    decision_id VARCHAR PRIMARY KEY,
    plan_cid VARCHAR NOT NULL,
    goal_cid VARCHAR NOT NULL,
    decision_kind VARCHAR NOT NULL,
    decided_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);
CREATE INDEX planning_decisions_plan_idx ON planning_decisions(plan_cid, decided_at);

CREATE TABLE plan_candidates (
    candidate_id VARCHAR PRIMARY KEY,
    plan_cid VARCHAR NOT NULL,
    ordinal BIGINT NOT NULL,
    score_bps BIGINT NOT NULL,
    status VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);
CREATE UNIQUE INDEX plan_candidates_plan_ordinal_uidx
    ON plan_candidates(plan_cid, ordinal);

CREATE TABLE tasks (
    task_cid VARCHAR PRIMARY KEY,
    task_alias VARCHAR NOT NULL UNIQUE,
    goal_cid VARCHAR NOT NULL,
    plan_cid VARCHAR NOT NULL DEFAULT '',
    objective_id VARCHAR NOT NULL DEFAULT '',
    ordinal BIGINT NOT NULL,
    status VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    priority VARCHAR NOT NULL DEFAULT '',
    created_at VARCHAR NOT NULL DEFAULT '',
    updated_at VARCHAR NOT NULL DEFAULT '',
    identity_json VARCHAR NOT NULL DEFAULT '{}',
    body_json VARCHAR NOT NULL DEFAULT '{}',
    extension_schema VARCHAR NOT NULL DEFAULT '',
    extension_json VARCHAR NOT NULL DEFAULT '{}'
);
CREATE INDEX tasks_goal_idx ON tasks(goal_cid, status);
CREATE INDEX tasks_status_idx ON tasks(status, ordinal);

CREATE TABLE task_revisions (
    task_cid VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    status VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL,
    recorded_at VARCHAR NOT NULL,
    PRIMARY KEY (task_cid, revision)
);

CREATE TABLE task_dependencies (
    task_cid VARCHAR NOT NULL,
    dependency_task_cid VARCHAR NOT NULL,
    kind VARCHAR NOT NULL,
    PRIMARY KEY (task_cid, dependency_task_cid, kind)
);
CREATE INDEX task_dependencies_dependency_idx
    ON task_dependencies(dependency_task_cid);

CREATE TABLE task_outputs (
    task_cid VARCHAR NOT NULL,
    ordinal BIGINT NOT NULL,
    path VARCHAR NOT NULL,
    effect_json VARCHAR NOT NULL,
    PRIMARY KEY (task_cid, ordinal),
    UNIQUE (task_cid, path)
);

CREATE TABLE task_acceptance (
    task_cid VARCHAR NOT NULL,
    ordinal BIGINT NOT NULL,
    criterion VARCHAR NOT NULL,
    evidence_policy_json VARCHAR NOT NULL,
    PRIMARY KEY (task_cid, ordinal)
);

CREATE TABLE task_validations (
    task_cid VARCHAR NOT NULL,
    ordinal BIGINT NOT NULL,
    argv_json VARCHAR NOT NULL,
    policy_json VARCHAR NOT NULL,
    PRIMARY KEY (task_cid, ordinal)
);

-- ---------------------------------------------------------------------------
-- schedule: assignments, blocks, refill, findings
-- ---------------------------------------------------------------------------
CREATE TABLE task_assignments (
    assignment_id VARCHAR PRIMARY KEY,
    task_cid VARCHAR NOT NULL,
    owner_session_id VARCHAR NOT NULL,
    daemon_id VARCHAR NOT NULL,
    assigned_at VARCHAR NOT NULL,
    released_at VARCHAR,
    state VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    fencing_token BIGINT NOT NULL DEFAULT 0
);
CREATE INDEX task_assignments_task_idx ON task_assignments(task_cid, state);
CREATE INDEX task_assignments_owner_idx ON task_assignments(owner_session_id, state);

CREATE TABLE task_blocks (
    block_id VARCHAR PRIMARY KEY,
    task_cid VARCHAR NOT NULL,
    blocker_kind VARCHAR NOT NULL,
    blocker_id VARCHAR NOT NULL,
    reason VARCHAR NOT NULL,
    created_at VARCHAR NOT NULL,
    cleared_at VARCHAR,
    state VARCHAR NOT NULL
);
CREATE INDEX task_blocks_task_idx ON task_blocks(task_cid, state);

CREATE TABLE refill_epochs (
    epoch_id VARCHAR PRIMARY KEY,
    board_namespace VARCHAR NOT NULL,
    started_at VARCHAR NOT NULL,
    finished_at VARCHAR,
    status VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE findings (
    finding_id VARCHAR PRIMARY KEY,
    epoch_id VARCHAR NOT NULL,
    task_cid VARCHAR NOT NULL DEFAULT '',
    finding_kind VARCHAR NOT NULL,
    severity VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    created_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);
CREATE INDEX findings_epoch_idx ON findings(epoch_id, status);
CREATE INDEX findings_task_idx ON findings(task_cid);

CREATE TABLE finding_dispositions (
    disposition_id VARCHAR PRIMARY KEY,
    finding_id VARCHAR NOT NULL,
    disposition VARCHAR NOT NULL,
    decided_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);
CREATE INDEX finding_dispositions_finding_idx
    ON finding_dispositions(finding_id, decided_at);

-- ---------------------------------------------------------------------------
-- runtime: daemons, attempts, claims, validation, merge, recovery
-- ---------------------------------------------------------------------------
CREATE TABLE supervisor_instances (
    supervisor_id VARCHAR PRIMARY KEY,
    repository_id VARCHAR NOT NULL,
    process_birth_id VARCHAR NOT NULL,
    started_at VARCHAR NOT NULL,
    stopped_at VARCHAR,
    status VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    extension_schema VARCHAR NOT NULL DEFAULT '',
    extension_json VARCHAR NOT NULL DEFAULT '{}'
);

CREATE TABLE daemon_instances (
    daemon_id VARCHAR PRIMARY KEY,
    supervisor_id VARCHAR NOT NULL,
    process_birth_id VARCHAR NOT NULL,
    role VARCHAR NOT NULL,
    started_at VARCHAR NOT NULL,
    stopped_at VARCHAR,
    status VARCHAR NOT NULL,
    revision BIGINT NOT NULL
);
CREATE INDEX daemon_instances_supervisor_idx
    ON daemon_instances(supervisor_id, status);

CREATE TABLE daemon_sessions (
    session_id VARCHAR PRIMARY KEY,
    daemon_id VARCHAR NOT NULL,
    process_birth_id VARCHAR NOT NULL,
    fence_epoch BIGINT NOT NULL,
    attached_at VARCHAR NOT NULL,
    last_heartbeat_at VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    revision BIGINT NOT NULL
);
CREATE INDEX daemon_sessions_daemon_idx ON daemon_sessions(daemon_id, status);

CREATE TABLE heartbeats (
    heartbeat_cid VARCHAR PRIMARY KEY,
    task_cid VARCHAR NOT NULL,
    claimant_did VARCHAR NOT NULL,
    fencing_token BIGINT NOT NULL,
    observed_at_ms BIGINT NOT NULL,
    expires_at_ms BIGINT NOT NULL,
    capacity_millionths BIGINT NOT NULL DEFAULT 0,
    session_id VARCHAR NOT NULL DEFAULT '',
    payload_json VARCHAR NOT NULL DEFAULT '{}'
);
CREATE INDEX heartbeats_task_idx ON heartbeats(task_cid, observed_at_ms);
CREATE INDEX heartbeats_session_idx ON heartbeats(session_id, observed_at_ms);

CREATE TABLE health_samples (
    sample_id VARCHAR PRIMARY KEY,
    subject_kind VARCHAR NOT NULL,
    subject_id VARCHAR NOT NULL,
    observed_at VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);
CREATE INDEX health_samples_subject_idx
    ON health_samples(subject_kind, subject_id, observed_at);

CREATE TABLE stall_detections (
    detection_id VARCHAR PRIMARY KEY,
    subject_kind VARCHAR NOT NULL,
    subject_id VARCHAR NOT NULL,
    task_cid VARCHAR NOT NULL DEFAULT '',
    detected_at VARCHAR NOT NULL,
    reason VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE restart_decisions (
    decision_id VARCHAR PRIMARY KEY,
    subject_kind VARCHAR NOT NULL,
    subject_id VARCHAR NOT NULL,
    decided_at VARCHAR NOT NULL,
    action VARCHAR NOT NULL,
    reason VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE task_attempts (
    attempt_id VARCHAR PRIMARY KEY,
    task_cid VARCHAR NOT NULL,
    attempt_number BIGINT NOT NULL,
    owner_session_id VARCHAR NOT NULL,
    fencing_token BIGINT NOT NULL,
    fence_epoch BIGINT NOT NULL,
    started_at VARCHAR NOT NULL,
    finished_at VARCHAR,
    status VARCHAR NOT NULL,
    revision BIGINT NOT NULL
);
CREATE UNIQUE INDEX task_attempts_task_number_uidx
    ON task_attempts(task_cid, attempt_number);
CREATE INDEX task_attempts_status_idx ON task_attempts(status, started_at);

CREATE TABLE attempt_phases (
    attempt_id VARCHAR NOT NULL,
    phase_name VARCHAR NOT NULL,
    entered_at VARCHAR NOT NULL,
    exited_at VARCHAR,
    status VARCHAR NOT NULL,
    PRIMARY KEY (attempt_id, phase_name, entered_at)
);

CREATE TABLE task_claims (
    claim_id VARCHAR PRIMARY KEY,
    task_cid VARCHAR NOT NULL,
    owner_session_id VARCHAR NOT NULL,
    fencing_token BIGINT NOT NULL,
    fence_epoch BIGINT NOT NULL,
    claimed_at VARCHAR NOT NULL,
    expires_at VARCHAR NOT NULL,
    released_at VARCHAR,
    state VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    idempotency_key VARCHAR NOT NULL DEFAULT ''
);
CREATE INDEX task_claims_task_idx ON task_claims(task_cid, state);
CREATE INDEX task_claims_idempotency_idx
    ON task_claims(idempotency_key);

CREATE TABLE provider_invocations (
    invocation_id VARCHAR PRIMARY KEY,
    task_cid VARCHAR NOT NULL,
    attempt_id VARCHAR NOT NULL,
    provider_id VARCHAR NOT NULL,
    started_at VARCHAR NOT NULL,
    finished_at VARCHAR,
    status VARCHAR NOT NULL,
    input_digest VARCHAR NOT NULL,
    output_digest VARCHAR NOT NULL DEFAULT '',
    body_json VARCHAR NOT NULL
);
CREATE INDEX provider_invocations_task_idx
    ON provider_invocations(task_cid, started_at);

CREATE TABLE validation_runs (
    run_id VARCHAR PRIMARY KEY,
    task_cid VARCHAR NOT NULL,
    attempt_id VARCHAR NOT NULL,
    started_at VARCHAR NOT NULL,
    finished_at VARCHAR,
    status VARCHAR NOT NULL,
    command_digest VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);
CREATE INDEX validation_runs_task_idx ON validation_runs(task_cid, started_at);

CREATE TABLE validation_results (
    result_id VARCHAR PRIMARY KEY,
    run_id VARCHAR NOT NULL,
    task_cid VARCHAR NOT NULL,
    ordinal BIGINT NOT NULL,
    outcome VARCHAR NOT NULL,
    evidence_digest VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);
CREATE UNIQUE INDEX validation_results_run_ordinal_uidx
    ON validation_results(run_id, ordinal);

CREATE TABLE merge_attempts (
    merge_attempt_id VARCHAR PRIMARY KEY,
    entry_id VARCHAR NOT NULL,
    task_cid VARCHAR NOT NULL,
    worktree_id VARCHAR NOT NULL,
    started_at VARCHAR NOT NULL,
    finished_at VARCHAR,
    status VARCHAR NOT NULL,
    result_commit_id VARCHAR NOT NULL DEFAULT '',
    body_json VARCHAR NOT NULL
);
CREATE INDEX merge_attempts_task_idx ON merge_attempts(task_cid, started_at);

CREATE TABLE recovery_actions (
    action_id VARCHAR PRIMARY KEY,
    subject_kind VARCHAR NOT NULL,
    subject_id VARCHAR NOT NULL,
    task_cid VARCHAR NOT NULL DEFAULT '',
    action_kind VARCHAR NOT NULL,
    decided_at VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE idempotency_records (
    idempotency_key VARCHAR PRIMARY KEY,
    command_kind VARCHAR NOT NULL,
    command_id VARCHAR NOT NULL,
    store_id VARCHAR NOT NULL,
    session_id VARCHAR NOT NULL,
    result_digest VARCHAR NOT NULL,
    created_at VARCHAR NOT NULL,
    expires_at VARCHAR,
    body_json VARCHAR NOT NULL
);

CREATE TABLE effect_claims (
    effect_id VARCHAR PRIMARY KEY,
    task_cid VARCHAR NOT NULL,
    attempt_id VARCHAR NOT NULL,
    effect_kind VARCHAR NOT NULL,
    target_path VARCHAR NOT NULL,
    claimed_at VARCHAR NOT NULL,
    state VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);
CREATE INDEX effect_claims_task_idx ON effect_claims(task_cid, state);

CREATE TABLE completion_receipts (
    receipt_cid VARCHAR PRIMARY KEY,
    task_cid VARCHAR NOT NULL,
    goal_cid VARCHAR NOT NULL,
    attempt_id VARCHAR NOT NULL DEFAULT '',
    claim_cid VARCHAR NOT NULL DEFAULT '',
    fencing_token BIGINT NOT NULL DEFAULT 0,
    completed_at VARCHAR NOT NULL,
    validation_run_id VARCHAR NOT NULL DEFAULT '',
    evidence_digest VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);
CREATE INDEX completion_receipts_task_idx
    ON completion_receipts(task_cid, completed_at);

-- ---------------------------------------------------------------------------
-- control/events: domain events, logs (also under control domain)
-- ---------------------------------------------------------------------------
CREATE TABLE domain_events (
    event_id VARCHAR PRIMARY KEY,
    stream_id VARCHAR NOT NULL,
    sequence BIGINT NOT NULL,
    global_sequence BIGINT NOT NULL,
    event_type VARCHAR NOT NULL,
    task_cid VARCHAR NOT NULL DEFAULT '',
    attempt_id VARCHAR NOT NULL DEFAULT '',
    session_id VARCHAR NOT NULL DEFAULT '',
    recorded_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);
CREATE UNIQUE INDEX domain_events_stream_seq_uidx
    ON domain_events(stream_id, sequence);
CREATE UNIQUE INDEX domain_events_global_seq_uidx
    ON domain_events(global_sequence);
CREATE INDEX domain_events_task_idx ON domain_events(task_cid, sequence);

CREATE TABLE structured_logs (
    log_id VARCHAR PRIMARY KEY,
    severity VARCHAR NOT NULL,
    component VARCHAR NOT NULL,
    trace_id VARCHAR NOT NULL DEFAULT '',
    span_id VARCHAR NOT NULL DEFAULT '',
    task_cid VARCHAR NOT NULL DEFAULT '',
    attempt_id VARCHAR NOT NULL DEFAULT '',
    session_id VARCHAR NOT NULL DEFAULT '',
    recorded_at VARCHAR NOT NULL,
    message VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);
CREATE INDEX structured_logs_component_idx
    ON structured_logs(component, recorded_at);
CREATE INDEX structured_logs_task_idx ON structured_logs(task_cid, recorded_at);

-- ---------------------------------------------------------------------------
-- improve: metrics, budgets, telemetry, churn
-- ---------------------------------------------------------------------------
CREATE TABLE metrics (
    metric_id VARCHAR PRIMARY KEY,
    metric_name VARCHAR NOT NULL UNIQUE,
    unit VARCHAR NOT NULL,
    description VARCHAR NOT NULL,
    created_at VARCHAR NOT NULL
);

CREATE TABLE metric_samples (
    sample_id VARCHAR PRIMARY KEY,
    metric_id VARCHAR NOT NULL,
    observed_at VARCHAR NOT NULL,
    value_milli BIGINT NOT NULL,
    labels_json VARCHAR NOT NULL DEFAULT '{}',
    stratum VARCHAR NOT NULL DEFAULT ''
);
CREATE INDEX metric_samples_metric_idx
    ON metric_samples(metric_id, observed_at);

CREATE TABLE budget_reservations (
    reservation_id VARCHAR PRIMARY KEY,
    budget_kind VARCHAR NOT NULL,
    owner_session_id VARCHAR NOT NULL,
    task_cid VARCHAR NOT NULL DEFAULT '',
    amount BIGINT NOT NULL,
    reserved_at VARCHAR NOT NULL,
    expires_at VARCHAR,
    state VARCHAR NOT NULL
);
CREATE INDEX budget_reservations_owner_idx
    ON budget_reservations(owner_session_id, state);

CREATE TABLE budget_consumption (
    consumption_id VARCHAR PRIMARY KEY,
    reservation_id VARCHAR NOT NULL,
    amount BIGINT NOT NULL,
    consumed_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE quack_query_telemetry (
    sample_id VARCHAR PRIMARY KEY,
    session_id VARCHAR NOT NULL,
    server_id VARCHAR NOT NULL,
    observed_at VARCHAR NOT NULL,
    latency_us BIGINT NOT NULL,
    row_count BIGINT NOT NULL,
    status VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE churn_metrics (
    metric_id VARCHAR PRIMARY KEY,
    task_cid VARCHAR NOT NULL DEFAULT '',
    window_start VARCHAR NOT NULL,
    window_end VARCHAR NOT NULL,
    provider_calls BIGINT NOT NULL,
    duplicate_inputs BIGINT NOT NULL,
    cache_hits BIGINT NOT NULL,
    tokens_in BIGINT NOT NULL,
    tokens_out BIGINT NOT NULL,
    body_json VARCHAR NOT NULL
);
CREATE INDEX churn_metrics_task_idx ON churn_metrics(task_cid, window_start);

-- ---------------------------------------------------------------------------
-- code: snapshots, AST, mutations, impact
-- ---------------------------------------------------------------------------
CREATE TABLE source_snapshots (
    snapshot_id VARCHAR PRIMARY KEY,
    repository_id VARCHAR NOT NULL,
    tree_id VARCHAR NOT NULL,
    overlay_digest VARCHAR NOT NULL DEFAULT '',
    created_at VARCHAR NOT NULL,
    scanner_version VARCHAR NOT NULL
);
CREATE INDEX source_snapshots_repo_idx
    ON source_snapshots(repository_id, created_at);

CREATE TABLE source_files (
    file_id VARCHAR PRIMARY KEY,
    snapshot_id VARCHAR NOT NULL,
    path VARCHAR NOT NULL,
    language VARCHAR NOT NULL,
    blob_id VARCHAR NOT NULL,
    byte_length BIGINT NOT NULL,
    content_digest VARCHAR NOT NULL
);
CREATE UNIQUE INDEX source_files_snapshot_path_uidx
    ON source_files(snapshot_id, path);

CREATE TABLE file_versions (
    file_version_id VARCHAR PRIMARY KEY,
    repository_id VARCHAR NOT NULL,
    path VARCHAR NOT NULL,
    blob_id VARCHAR NOT NULL,
    content_digest VARCHAR NOT NULL,
    observed_at VARCHAR NOT NULL
);
CREATE INDEX file_versions_path_idx
    ON file_versions(repository_id, path, observed_at);

CREATE TABLE parse_runs (
    parse_run_id VARCHAR PRIMARY KEY,
    snapshot_id VARCHAR NOT NULL,
    parser_id VARCHAR NOT NULL,
    started_at VARCHAR NOT NULL,
    finished_at VARCHAR,
    status VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE symbols (
    symbol_id VARCHAR PRIMARY KEY,
    snapshot_id VARCHAR NOT NULL,
    file_id VARCHAR NOT NULL,
    language VARCHAR NOT NULL,
    qualified_name VARCHAR NOT NULL,
    symbol_kind VARCHAR NOT NULL,
    start_line BIGINT NOT NULL,
    end_line BIGINT NOT NULL,
    fingerprint VARCHAR NOT NULL
);
CREATE INDEX symbols_snapshot_name_idx
    ON symbols(snapshot_id, qualified_name);

CREATE TABLE symbol_versions (
    symbol_version_id VARCHAR PRIMARY KEY,
    symbol_id VARCHAR NOT NULL,
    snapshot_id VARCHAR NOT NULL,
    fingerprint VARCHAR NOT NULL,
    observed_at VARCHAR NOT NULL
);

CREATE TABLE ast_nodes (
    node_id VARCHAR PRIMARY KEY,
    snapshot_id VARCHAR NOT NULL,
    file_id VARCHAR NOT NULL,
    parent_node_id VARCHAR NOT NULL DEFAULT '',
    node_kind VARCHAR NOT NULL,
    node_path VARCHAR NOT NULL,
    fingerprint VARCHAR NOT NULL,
    start_byte BIGINT NOT NULL,
    end_byte BIGINT NOT NULL
);
CREATE INDEX ast_nodes_file_idx ON ast_nodes(file_id, node_path);

CREATE TABLE ast_edges (
    edge_id VARCHAR PRIMARY KEY,
    snapshot_id VARCHAR NOT NULL,
    source_node_id VARCHAR NOT NULL,
    target_node_id VARCHAR NOT NULL,
    edge_kind VARCHAR NOT NULL
);
CREATE INDEX ast_edges_source_idx ON ast_edges(source_node_id, edge_kind);

CREATE TABLE "imports" (
    import_id VARCHAR PRIMARY KEY,
    snapshot_id VARCHAR NOT NULL,
    file_id VARCHAR NOT NULL,
    module_name VARCHAR NOT NULL,
    alias VARCHAR NOT NULL DEFAULT '',
    start_line BIGINT NOT NULL
);

CREATE TABLE "calls" (
    call_id VARCHAR PRIMARY KEY,
    snapshot_id VARCHAR NOT NULL,
    caller_symbol_id VARCHAR NOT NULL,
    callee_symbol_id VARCHAR NOT NULL,
    file_id VARCHAR NOT NULL,
    start_line BIGINT NOT NULL
);
CREATE INDEX calls_caller_idx ON "calls"(caller_symbol_id);
CREATE INDEX calls_callee_idx ON "calls"(callee_symbol_id);

CREATE TABLE "references" (
    reference_id VARCHAR PRIMARY KEY,
    snapshot_id VARCHAR NOT NULL,
    symbol_id VARCHAR NOT NULL,
    file_id VARCHAR NOT NULL,
    start_line BIGINT NOT NULL,
    reference_kind VARCHAR NOT NULL
);
CREATE INDEX references_symbol_idx ON "references"(symbol_id);

CREATE TABLE definitions (
    definition_id VARCHAR PRIMARY KEY,
    snapshot_id VARCHAR NOT NULL,
    symbol_id VARCHAR NOT NULL,
    file_id VARCHAR NOT NULL,
    start_line BIGINT NOT NULL,
    end_line BIGINT NOT NULL
);
CREATE UNIQUE INDEX definitions_symbol_uidx ON definitions(symbol_id, snapshot_id);

CREATE TABLE type_relations (
    relation_id VARCHAR PRIMARY KEY,
    snapshot_id VARCHAR NOT NULL,
    left_symbol_id VARCHAR NOT NULL,
    right_symbol_id VARCHAR NOT NULL,
    relation_kind VARCHAR NOT NULL
);

CREATE TABLE mutations (
    mutation_id VARCHAR PRIMARY KEY,
    task_cid VARCHAR NOT NULL,
    attempt_id VARCHAR NOT NULL,
    before_snapshot_id VARCHAR NOT NULL,
    after_snapshot_id VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    created_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);
CREATE INDEX mutations_task_idx ON mutations(task_cid, created_at);

CREATE TABLE mutation_files (
    mutation_id VARCHAR NOT NULL,
    path VARCHAR NOT NULL,
    before_blob_id VARCHAR NOT NULL,
    after_blob_id VARCHAR NOT NULL,
    PRIMARY KEY (mutation_id, path)
);

CREATE TABLE mutation_hunks (
    hunk_id VARCHAR PRIMARY KEY,
    mutation_id VARCHAR NOT NULL,
    path VARCHAR NOT NULL,
    ordinal BIGINT NOT NULL,
    before_start_line BIGINT NOT NULL,
    after_start_line BIGINT NOT NULL,
    body_json VARCHAR NOT NULL
);
CREATE UNIQUE INDEX mutation_hunks_mutation_ordinal_uidx
    ON mutation_hunks(mutation_id, ordinal);

CREATE TABLE ast_mutations (
    ast_mutation_id VARCHAR PRIMARY KEY,
    mutation_id VARCHAR NOT NULL,
    before_node_id VARCHAR NOT NULL,
    after_node_id VARCHAR NOT NULL,
    edit_kind VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE impact_edges (
    edge_id VARCHAR PRIMARY KEY,
    mutation_id VARCHAR NOT NULL,
    source_symbol_id VARCHAR NOT NULL,
    target_symbol_id VARCHAR NOT NULL,
    edge_kind VARCHAR NOT NULL
);
CREATE INDEX impact_edges_mutation_idx ON impact_edges(mutation_id);

CREATE TABLE impact_closures (
    closure_id VARCHAR PRIMARY KEY,
    mutation_id VARCHAR NOT NULL,
    root_symbol_id VARCHAR NOT NULL,
    member_symbol_id VARCHAR NOT NULL,
    depth BIGINT NOT NULL
);
CREATE UNIQUE INDEX impact_closures_member_uidx
    ON impact_closures(mutation_id, root_symbol_id, member_symbol_id);

CREATE TABLE repair_candidates (
    candidate_id VARCHAR PRIMARY KEY,
    task_cid VARCHAR NOT NULL,
    mutation_id VARCHAR NOT NULL DEFAULT '',
    score_bps BIGINT NOT NULL,
    status VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);
CREATE INDEX repair_candidates_task_idx ON repair_candidates(task_cid, status);

CREATE TABLE repair_applications (
    application_id VARCHAR PRIMARY KEY,
    candidate_id VARCHAR NOT NULL,
    task_cid VARCHAR NOT NULL,
    applied_at VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);

-- ---------------------------------------------------------------------------
-- evidence: proof obligations, attempts, counterexamples, evidence nodes
-- ---------------------------------------------------------------------------
CREATE TABLE proof_obligations (
    obligation_id VARCHAR PRIMARY KEY,
    task_cid VARCHAR NOT NULL,
    snapshot_id VARCHAR NOT NULL,
    obligation_kind VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    created_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);
CREATE INDEX proof_obligations_task_idx ON proof_obligations(task_cid, status);

CREATE TABLE proof_attempts (
    attempt_id VARCHAR PRIMARY KEY,
    obligation_id VARCHAR NOT NULL,
    prover_id VARCHAR NOT NULL,
    started_at VARCHAR NOT NULL,
    finished_at VARCHAR,
    status VARCHAR NOT NULL,
    evidence_digest VARCHAR NOT NULL DEFAULT '',
    body_json VARCHAR NOT NULL
);
CREATE INDEX proof_attempts_obligation_idx
    ON proof_attempts(obligation_id, started_at);

CREATE TABLE counterexamples (
    counterexample_id VARCHAR PRIMARY KEY,
    obligation_id VARCHAR NOT NULL,
    attempt_id VARCHAR NOT NULL,
    created_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);
CREATE INDEX counterexamples_obligation_idx
    ON counterexamples(obligation_id, created_at);

CREATE TABLE evidence_nodes (
    evidence_id VARCHAR PRIMARY KEY,
    parent_evidence_id VARCHAR NOT NULL DEFAULT '',
    task_cid VARCHAR NOT NULL DEFAULT '',
    evidence_kind VARCHAR NOT NULL,
    digest VARCHAR NOT NULL,
    created_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);
CREATE INDEX evidence_nodes_task_idx ON evidence_nodes(task_cid, evidence_kind);

-- ---------------------------------------------------------------------------
-- cache: context, prompts, provider calls, decision cache
-- ---------------------------------------------------------------------------
CREATE TABLE context_manifests (
    manifest_cid VARCHAR PRIMARY KEY,
    task_cid VARCHAR NOT NULL,
    schema_revision BIGINT NOT NULL,
    repository_tree_id VARCHAR NOT NULL,
    policy_digest VARCHAR NOT NULL,
    created_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);
CREATE INDEX context_manifests_task_idx ON context_manifests(task_cid, created_at);

CREATE TABLE context_members (
    manifest_cid VARCHAR NOT NULL,
    ordinal BIGINT NOT NULL,
    member_kind VARCHAR NOT NULL,
    member_id VARCHAR NOT NULL,
    digest VARCHAR NOT NULL,
    PRIMARY KEY (manifest_cid, ordinal)
);

CREATE TABLE context_deltas (
    delta_id VARCHAR PRIMARY KEY,
    from_manifest_cid VARCHAR NOT NULL,
    to_manifest_cid VARCHAR NOT NULL,
    created_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE prompt_templates (
    template_id VARCHAR PRIMARY KEY,
    template_name VARCHAR NOT NULL UNIQUE,
    revision BIGINT NOT NULL,
    body_digest VARCHAR NOT NULL,
    created_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE prompt_instances (
    instance_id VARCHAR PRIMARY KEY,
    template_id VARCHAR NOT NULL,
    task_cid VARCHAR NOT NULL,
    manifest_cid VARCHAR NOT NULL,
    created_at VARCHAR NOT NULL,
    input_digest VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);
CREATE INDEX prompt_instances_task_idx ON prompt_instances(task_cid, created_at);

CREATE TABLE prompt_inputs (
    instance_id VARCHAR NOT NULL,
    ordinal BIGINT NOT NULL,
    input_kind VARCHAR NOT NULL,
    input_digest VARCHAR NOT NULL,
    PRIMARY KEY (instance_id, ordinal)
);

CREATE TABLE provider_calls (
    call_id VARCHAR PRIMARY KEY,
    task_cid VARCHAR NOT NULL,
    attempt_id VARCHAR NOT NULL DEFAULT '',
    provider_id VARCHAR NOT NULL,
    prompt_instance_id VARCHAR NOT NULL,
    started_at VARCHAR NOT NULL,
    finished_at VARCHAR,
    status VARCHAR NOT NULL,
    input_digest VARCHAR NOT NULL,
    output_digest VARCHAR NOT NULL DEFAULT '',
    tokens_in BIGINT NOT NULL DEFAULT 0,
    tokens_out BIGINT NOT NULL DEFAULT 0,
    body_json VARCHAR NOT NULL
);
CREATE INDEX provider_calls_task_idx ON provider_calls(task_cid, started_at);
CREATE INDEX provider_calls_input_idx ON provider_calls(input_digest);

CREATE TABLE provider_responses (
    response_id VARCHAR PRIMARY KEY,
    call_id VARCHAR NOT NULL,
    received_at VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    output_digest VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);
CREATE INDEX provider_responses_call_idx ON provider_responses(call_id);

CREATE TABLE failure_signatures (
    signature_id VARCHAR PRIMARY KEY,
    task_cid VARCHAR NOT NULL,
    signature_digest VARCHAR NOT NULL,
    failure_kind VARCHAR NOT NULL,
    first_seen_at VARCHAR NOT NULL,
    last_seen_at VARCHAR NOT NULL,
    occurrence_count BIGINT NOT NULL,
    body_json VARCHAR NOT NULL
);
CREATE UNIQUE INDEX failure_signatures_task_digest_uidx
    ON failure_signatures(task_cid, signature_digest);

CREATE TABLE decision_cache_entries (
    cache_key VARCHAR PRIMARY KEY,
    task_cid VARCHAR NOT NULL DEFAULT '',
    manifest_cid VARCHAR NOT NULL,
    decision_digest VARCHAR NOT NULL,
    created_at VARCHAR NOT NULL,
    expires_at VARCHAR,
    hit_count BIGINT NOT NULL DEFAULT 0,
    body_json VARCHAR NOT NULL
);
CREATE INDEX decision_cache_entries_manifest_idx
    ON decision_cache_entries(manifest_cid);

CREATE TABLE replay_suppressions (
    suppression_id VARCHAR PRIMARY KEY,
    task_cid VARCHAR NOT NULL,
    input_digest VARCHAR NOT NULL,
    reason VARCHAR NOT NULL,
    created_at VARCHAR NOT NULL,
    expires_at VARCHAR
);
CREATE UNIQUE INDEX replay_suppressions_task_input_uidx
    ON replay_suppressions(task_cid, input_digest);

-- ---------------------------------------------------------------------------
-- artifacts (shared content-addressed metadata; large bodies live off-db)
-- ---------------------------------------------------------------------------
CREATE TABLE artifacts (
    cid VARCHAR PRIMARY KEY,
    media_type VARCHAR NOT NULL,
    byte_length BIGINT NOT NULL,
    digest VARCHAR NOT NULL,
    storage_uri VARCHAR NOT NULL DEFAULT '',
    kind VARCHAR NOT NULL DEFAULT '',
    created_at VARCHAR NOT NULL DEFAULT '',
    provenance_json VARCHAR NOT NULL DEFAULT '{}',
    payload_json VARCHAR NOT NULL DEFAULT '{}'
);

-- ---------------------------------------------------------------------------
-- Seed schema_contracts for domain inventory (deterministic)
-- ---------------------------------------------------------------------------
INSERT INTO schema_contracts (
    contract_id, interface_name, domain_name, schema_revision,
    payload_schema, description, created_at
) VALUES
    ('contract:ControlPlaneSchema@1', 'ControlPlaneSchema@1', 'meta', 1,
     'ipfs_accelerate_py/agent-supervisor/control-plane-schema@1',
     'Normalized control-plane physical schema', '1970-01-01T00:00:00Z'),
    ('contract:meta@1', 'ControlPlaneDomain@1', 'meta', 1,
     'ipfs_accelerate_py/agent-supervisor/domain-meta@1',
     'Schema contracts and generations', '1970-01-01T00:00:00Z'),
    ('contract:intent@1', 'ControlPlaneDomain@1', 'intent', 1,
     'ipfs_accelerate_py/agent-supervisor/domain-intent@1',
     'Objectives, goals, plans, tasks', '1970-01-01T00:00:00Z'),
    ('contract:schedule@1', 'ControlPlaneDomain@1', 'schedule', 1,
     'ipfs_accelerate_py/agent-supervisor/domain-schedule@1',
     'Assignments, blocks, refill, findings', '1970-01-01T00:00:00Z'),
    ('contract:runtime@1', 'ControlPlaneDomain@1', 'runtime', 1,
     'ipfs_accelerate_py/agent-supervisor/domain-runtime@1',
     'Daemons, attempts, claims, validation, merge', '1970-01-01T00:00:00Z'),
    ('contract:git@1', 'ControlPlaneDomain@1', 'git', 1,
     'ipfs_accelerate_py/agent-supervisor/domain-git@1',
     'Repositories, worktrees, leases, path claims', '1970-01-01T00:00:00Z'),
    ('contract:code@1', 'ControlPlaneDomain@1', 'code', 1,
     'ipfs_accelerate_py/agent-supervisor/domain-code@1',
     'Source snapshots, AST, mutations, impact', '1970-01-01T00:00:00Z'),
    ('contract:evidence@1', 'ControlPlaneDomain@1', 'evidence', 1,
     'ipfs_accelerate_py/agent-supervisor/domain-evidence@1',
     'Proof obligations and evidence nodes', '1970-01-01T00:00:00Z'),
    ('contract:cache@1', 'ControlPlaneDomain@1', 'cache', 1,
     'ipfs_accelerate_py/agent-supervisor/domain-cache@1',
     'Context, prompts, decision cache', '1970-01-01T00:00:00Z'),
    ('contract:control@1', 'ControlPlaneDomain@1', 'control', 1,
     'ipfs_accelerate_py/agent-supervisor/domain-control@1',
     'Servers, sessions, auth, maintenance, events', '1970-01-01T00:00:00Z'),
    ('contract:improve@1', 'ControlPlaneDomain@1', 'improve', 1,
     'ipfs_accelerate_py/agent-supervisor/domain-improve@1',
     'Metrics, budgets, churn telemetry', '1970-01-01T00:00:00Z');

-- ---------------------------------------------------------------------------
-- Constrained diagnostic / context views (read-only projections)
-- ---------------------------------------------------------------------------
CREATE VIEW ready_task_context_v1 AS
SELECT
    t.task_cid,
    t.task_alias,
    t.goal_cid,
    t.plan_cid,
    t.status AS task_status,
    t.revision AS task_revision,
    t.ordinal,
    l.state AS lease_state,
    l.claimant_did,
    l.fencing_token,
    l.expires_at_ms,
    l.attempt AS lease_attempt,
    l.owner_session_id,
    l.fence_epoch
FROM tasks AS t
LEFT JOIN leases AS l ON l.task_cid = t.task_cid
WHERE t.status IN ('ready', 'open', 'todo', 'pending');

CREATE VIEW diagnostic_schema_inventory_v1 AS
SELECT
    domain_name,
    interface_name,
    schema_revision,
    contract_id,
    payload_schema
FROM schema_contracts
ORDER BY domain_name, interface_name;

CREATE VIEW diagnostic_lease_surface_v1 AS
SELECT
    task_cid,
    claim_cid,
    claimant_did,
    logical_epoch,
    fencing_token,
    fence_epoch,
    expires_at_ms,
    attempt,
    state,
    started_at_ms,
    release_reason,
    retry_not_before_ms,
    owner_session_id,
    revision
FROM leases;
