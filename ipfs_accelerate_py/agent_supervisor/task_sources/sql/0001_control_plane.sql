-- Control-plane base schema (DQP-005 / ControlPlaneSchema@1).
-- Bookkeeping tables (control_plane_metadata, schema_migrations,
-- schema_migration_attempts) are installed by the migration runner.
-- Domain SQL is deterministic, free of secrets, and join-critical identities
-- live in typed columns (not opaque JSON alone).

-- ---------------------------------------------------------------------------
-- meta: schema contracts, servers, sessions, credentials, maintenance
-- ---------------------------------------------------------------------------

CREATE TABLE schema_contracts (
    contract_id VARCHAR PRIMARY KEY,
    domain VARCHAR NOT NULL,
    table_name VARCHAR NOT NULL,
    interface_name VARCHAR NOT NULL,
    schema_name VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    payload_schema VARCHAR NOT NULL,
    created_at VARCHAR NOT NULL,
    UNIQUE (domain, table_name, revision)
);

CREATE TABLE state_servers (
    server_id VARCHAR PRIMARY KEY,
    store_id VARCHAR NOT NULL,
    database_uuid VARCHAR NOT NULL,
    process_birth_id VARCHAR NOT NULL,
    listen_uri VARCHAR NOT NULL,
    extension_fingerprint VARCHAR NOT NULL,
    schema_revision BIGINT NOT NULL,
    generation BIGINT NOT NULL,
    credential_generation BIGINT NOT NULL,
    started_at VARCHAR NOT NULL,
    stopped_at VARCHAR,
    status VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE server_epochs (
    server_id VARCHAR NOT NULL,
    fence_epoch BIGINT NOT NULL,
    generation BIGINT NOT NULL,
    opened_at VARCHAR NOT NULL,
    closed_at VARCHAR,
    PRIMARY KEY (server_id, fence_epoch)
);

CREATE TABLE client_sessions (
    session_id VARCHAR PRIMARY KEY,
    server_id VARCHAR NOT NULL,
    store_id VARCHAR NOT NULL,
    process_birth_id VARCHAR NOT NULL,
    fence_epoch BIGINT NOT NULL,
    opened_at VARCHAR NOT NULL,
    expires_at VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE capability_snapshots (
    snapshot_id VARCHAR PRIMARY KEY,
    profile_id VARCHAR NOT NULL,
    duckdb_version VARCHAR NOT NULL,
    extension_name VARCHAR NOT NULL,
    extension_fingerprint VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    observed_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE credentials (
    credential_id VARCHAR PRIMARY KEY,
    handle VARCHAR NOT NULL,
    generation BIGINT NOT NULL,
    purpose VARCHAR NOT NULL,
    created_at VARCHAR NOT NULL,
    rotated_at VARCHAR,
    revoked_at VARCHAR,
    UNIQUE (handle, generation)
);

CREATE TABLE authorization_roles (
    role_id VARCHAR PRIMARY KEY,
    role_name VARCHAR NOT NULL UNIQUE,
    description VARCHAR NOT NULL,
    revision BIGINT NOT NULL
);

CREATE TABLE authorization_grants (
    grant_id VARCHAR PRIMARY KEY,
    role_id VARCHAR NOT NULL,
    principal_id VARCHAR NOT NULL,
    scope VARCHAR NOT NULL,
    issued_at VARCHAR NOT NULL,
    expires_at VARCHAR,
    revision BIGINT NOT NULL,
    UNIQUE (role_id, principal_id, scope)
);

CREATE TABLE backup_snapshots (
    snapshot_id VARCHAR PRIMARY KEY,
    database_uuid VARCHAR NOT NULL,
    schema_revision BIGINT NOT NULL,
    schema_fingerprint VARCHAR NOT NULL,
    artifact_digest VARCHAR NOT NULL,
    created_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE restore_receipts (
    receipt_id VARCHAR PRIMARY KEY,
    snapshot_id VARCHAR NOT NULL,
    database_uuid VARCHAR NOT NULL,
    restored_at VARCHAR NOT NULL,
    outcome VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE maintenance_leases (
    lease_id VARCHAR PRIMARY KEY,
    store_id VARCHAR NOT NULL,
    owner_session_id VARCHAR NOT NULL,
    purpose VARCHAR NOT NULL,
    fencing_token BIGINT NOT NULL,
    fence_epoch BIGINT NOT NULL,
    expires_at_ms BIGINT NOT NULL,
    state VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    UNIQUE (store_id, purpose, fencing_token)
);

-- ---------------------------------------------------------------------------
-- git / repository forest
-- ---------------------------------------------------------------------------

CREATE TABLE repositories (
    repository_id VARCHAR PRIMARY KEY,
    canonical_root VARCHAR NOT NULL,
    default_branch VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE repository_revisions (
    repository_id VARCHAR NOT NULL,
    tree_id VARCHAR NOT NULL,
    commit_id VARCHAR NOT NULL,
    observed_at VARCHAR NOT NULL,
    scanner_version VARCHAR NOT NULL,
    PRIMARY KEY (repository_id, tree_id)
);

CREATE TABLE submodule_edges (
    parent_repository_id VARCHAR NOT NULL,
    child_repository_id VARCHAR NOT NULL,
    path VARCHAR NOT NULL,
    gitlink_oid VARCHAR NOT NULL,
    PRIMARY KEY (parent_repository_id, path)
);

CREATE TABLE worktrees (
    worktree_id VARCHAR PRIMARY KEY,
    repository_id VARCHAR NOT NULL,
    path_digest VARCHAR NOT NULL,
    head_commit_id VARCHAR NOT NULL,
    branch_name VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    fence_epoch BIGINT NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE worktree_snapshots (
    snapshot_id VARCHAR PRIMARY KEY,
    worktree_id VARCHAR NOT NULL,
    tree_id VARCHAR NOT NULL,
    index_digest VARCHAR NOT NULL,
    overlay_digest VARCHAR NOT NULL,
    observed_at VARCHAR NOT NULL
);

CREATE TABLE worktree_paths (
    worktree_id VARCHAR NOT NULL,
    path VARCHAR NOT NULL,
    path_digest VARCHAR NOT NULL,
    kind VARCHAR NOT NULL,
    PRIMARY KEY (worktree_id, path)
);

CREATE TABLE dirty_overlays (
    overlay_id VARCHAR PRIMARY KEY,
    worktree_id VARCHAR NOT NULL,
    overlay_digest VARCHAR NOT NULL,
    created_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE branches (
    repository_id VARCHAR NOT NULL,
    branch_name VARCHAR NOT NULL,
    tip_commit_id VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    PRIMARY KEY (repository_id, branch_name)
);

CREATE TABLE git_refs (
    repository_id VARCHAR NOT NULL,
    ref_name VARCHAR NOT NULL,
    object_id VARCHAR NOT NULL,
    ref_type VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    PRIMARY KEY (repository_id, ref_name)
);

CREATE TABLE merge_bases (
    left_commit_id VARCHAR NOT NULL,
    right_commit_id VARCHAR NOT NULL,
    merge_base_id VARCHAR NOT NULL,
    computed_at VARCHAR NOT NULL,
    PRIMARY KEY (left_commit_id, right_commit_id)
);

CREATE TABLE merge_queue_entries (
    entry_id VARCHAR PRIMARY KEY,
    repository_id VARCHAR NOT NULL,
    task_cid VARCHAR NOT NULL,
    worktree_id VARCHAR NOT NULL,
    source_ref VARCHAR NOT NULL,
    target_ref VARCHAR NOT NULL,
    state VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    fence_epoch BIGINT NOT NULL,
    created_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE resource_claims (
    claim_id VARCHAR PRIMARY KEY,
    resource_kind VARCHAR NOT NULL,
    resource_id VARCHAR NOT NULL,
    owner_session_id VARCHAR NOT NULL,
    fencing_token BIGINT NOT NULL,
    fence_epoch BIGINT NOT NULL,
    expires_at_ms BIGINT NOT NULL,
    state VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    UNIQUE (resource_kind, resource_id, fencing_token)
);

CREATE TABLE path_claims (
    claim_id VARCHAR PRIMARY KEY,
    repository_id VARCHAR NOT NULL,
    path VARCHAR NOT NULL,
    owner_session_id VARCHAR NOT NULL,
    task_cid VARCHAR NOT NULL,
    fencing_token BIGINT NOT NULL,
    fence_epoch BIGINT NOT NULL,
    expires_at_ms BIGINT NOT NULL,
    state VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    UNIQUE (repository_id, path, fencing_token)
);

-- leases: preserve existing task_cid / fencing / claim semantics
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
    revision BIGINT NOT NULL DEFAULT 0,
    fence_epoch BIGINT NOT NULL DEFAULT 0,
    owner_session_id VARCHAR NOT NULL DEFAULT '',
    body_json VARCHAR NOT NULL DEFAULT '{}'
);

CREATE TABLE lease_events (
    event_id VARCHAR PRIMARY KEY,
    task_cid VARCHAR NOT NULL,
    sequence BIGINT NOT NULL,
    event_type VARCHAR NOT NULL,
    fencing_token BIGINT NOT NULL,
    observed_at_ms BIGINT NOT NULL,
    body_json VARCHAR NOT NULL,
    UNIQUE (task_cid, sequence)
);

CREATE INDEX idx_leases_scheduler_state
    ON leases (state, expires_at_ms, retry_not_before_ms);

-- ---------------------------------------------------------------------------
-- intent: objectives, plans, tasks
-- ---------------------------------------------------------------------------

CREATE TABLE objectives (
    objective_id VARCHAR PRIMARY KEY,
    objective_cid VARCHAR NOT NULL UNIQUE,
    title VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE objective_revisions (
    objective_id VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    content_cid VARCHAR NOT NULL,
    created_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL,
    PRIMARY KEY (objective_id, revision)
);

CREATE TABLE goals (
    goal_cid VARCHAR PRIMARY KEY,
    goal_alias VARCHAR NOT NULL UNIQUE,
    objective_id VARCHAR NOT NULL,
    parent_goal_cid VARCHAR NOT NULL,
    ordinal BIGINT NOT NULL,
    title VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE goal_edges (
    parent_goal_cid VARCHAR NOT NULL,
    child_goal_cid VARCHAR NOT NULL,
    edge_kind VARCHAR NOT NULL,
    PRIMARY KEY (parent_goal_cid, child_goal_cid, edge_kind)
);

CREATE TABLE plans (
    plan_id VARCHAR PRIMARY KEY,
    plan_cid VARCHAR NOT NULL UNIQUE,
    objective_id VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE plan_revisions (
    plan_id VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    content_cid VARCHAR NOT NULL,
    created_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL,
    PRIMARY KEY (plan_id, revision)
);

CREATE TABLE planning_decisions (
    decision_id VARCHAR PRIMARY KEY,
    plan_id VARCHAR NOT NULL,
    decision_kind VARCHAR NOT NULL,
    content_cid VARCHAR NOT NULL,
    created_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE plan_candidates (
    candidate_id VARCHAR PRIMARY KEY,
    plan_id VARCHAR NOT NULL,
    rank_ordinal BIGINT NOT NULL,
    content_cid VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE tasks (
    task_cid VARCHAR PRIMARY KEY,
    task_alias VARCHAR NOT NULL UNIQUE,
    goal_cid VARCHAR NOT NULL,
    plan_id VARCHAR NOT NULL DEFAULT '',
    ordinal BIGINT NOT NULL,
    status VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    identity_json VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL,
    registered_at_ms BIGINT NOT NULL DEFAULT 0,
    updated_at_ms BIGINT NOT NULL DEFAULT 0
);

CREATE TABLE task_revisions (
    task_cid VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    status VARCHAR NOT NULL,
    content_cid VARCHAR NOT NULL,
    created_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL,
    PRIMARY KEY (task_cid, revision)
);

CREATE TABLE task_dependencies (
    task_cid VARCHAR NOT NULL,
    dependency_task_cid VARCHAR NOT NULL,
    kind VARCHAR NOT NULL,
    PRIMARY KEY (task_cid, dependency_task_cid, kind)
);

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

CREATE TABLE task_assignments (
    assignment_id VARCHAR PRIMARY KEY,
    task_cid VARCHAR NOT NULL,
    owner_session_id VARCHAR NOT NULL,
    assigned_at VARCHAR NOT NULL,
    state VARCHAR NOT NULL,
    revision BIGINT NOT NULL
);

CREATE TABLE task_blocks (
    block_id VARCHAR PRIMARY KEY,
    task_cid VARCHAR NOT NULL,
    reason_code VARCHAR NOT NULL,
    blocked_at VARCHAR NOT NULL,
    cleared_at VARCHAR,
    body_json VARCHAR NOT NULL
);

CREATE TABLE refill_epochs (
    epoch_id VARCHAR PRIMARY KEY,
    board_namespace VARCHAR NOT NULL,
    opened_at VARCHAR NOT NULL,
    closed_at VARCHAR,
    revision BIGINT NOT NULL
);

CREATE TABLE findings (
    finding_id VARCHAR PRIMARY KEY,
    finding_cid VARCHAR NOT NULL UNIQUE,
    source VARCHAR NOT NULL,
    severity VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE finding_dispositions (
    disposition_id VARCHAR PRIMARY KEY,
    finding_id VARCHAR NOT NULL,
    disposition VARCHAR NOT NULL,
    decided_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE INDEX task_dependencies_dependency_idx
    ON task_dependencies (dependency_task_cid);

CREATE INDEX tasks_status_revision_idx
    ON tasks (status, revision);

-- ---------------------------------------------------------------------------
-- schedule
-- ---------------------------------------------------------------------------

CREATE TABLE schedule_policies (
    policy_id VARCHAR PRIMARY KEY,
    board_namespace VARCHAR NOT NULL,
    max_lanes BIGINT NOT NULL,
    max_task_attempts BIGINT NOT NULL,
    revision BIGINT NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE schedule_lanes (
    lane_id VARCHAR PRIMARY KEY,
    board_namespace VARCHAR NOT NULL,
    lane_index BIGINT NOT NULL,
    lane_name VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    UNIQUE (board_namespace, lane_index)
);

CREATE TABLE schedule_queue_entries (
    entry_id VARCHAR PRIMARY KEY,
    task_cid VARCHAR NOT NULL,
    lane_id VARCHAR NOT NULL,
    priority BIGINT NOT NULL,
    not_before_ms BIGINT NOT NULL,
    state VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    UNIQUE (task_cid, lane_id)
);

-- ---------------------------------------------------------------------------
-- runtime: daemons, claims, attempts, validation/merge
-- ---------------------------------------------------------------------------

CREATE TABLE supervisor_instances (
    instance_id VARCHAR PRIMARY KEY,
    process_birth_id VARCHAR NOT NULL,
    board_namespace VARCHAR NOT NULL,
    started_at VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE daemon_instances (
    daemon_id VARCHAR PRIMARY KEY,
    supervisor_instance_id VARCHAR NOT NULL,
    process_birth_id VARCHAR NOT NULL,
    role VARCHAR NOT NULL,
    started_at VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE daemon_sessions (
    session_id VARCHAR PRIMARY KEY,
    daemon_id VARCHAR NOT NULL,
    fence_epoch BIGINT NOT NULL,
    opened_at VARCHAR NOT NULL,
    expires_at_ms BIGINT NOT NULL,
    status VARCHAR NOT NULL,
    revision BIGINT NOT NULL
);

CREATE TABLE heartbeats (
    heartbeat_cid VARCHAR PRIMARY KEY,
    task_cid VARCHAR NOT NULL,
    claimant_did VARCHAR NOT NULL,
    fencing_token BIGINT NOT NULL,
    observed_at_ms BIGINT NOT NULL,
    expires_at_ms BIGINT NOT NULL,
    capacity_millionths BIGINT NOT NULL,
    payload_json VARCHAR NOT NULL
);

CREATE TABLE health_samples (
    sample_id VARCHAR PRIMARY KEY,
    subject_id VARCHAR NOT NULL,
    subject_kind VARCHAR NOT NULL,
    observed_at_ms BIGINT NOT NULL,
    status VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE stall_detections (
    detection_id VARCHAR PRIMARY KEY,
    subject_id VARCHAR NOT NULL,
    detected_at_ms BIGINT NOT NULL,
    reason_code VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE restart_decisions (
    decision_id VARCHAR PRIMARY KEY,
    subject_id VARCHAR NOT NULL,
    decided_at_ms BIGINT NOT NULL,
    action VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE task_attempts (
    attempt_id VARCHAR PRIMARY KEY,
    task_cid VARCHAR NOT NULL,
    attempt_number BIGINT NOT NULL,
    worktree_id VARCHAR NOT NULL,
    owner_session_id VARCHAR NOT NULL,
    fencing_token BIGINT NOT NULL,
    fence_epoch BIGINT NOT NULL,
    status VARCHAR NOT NULL,
    started_at_ms BIGINT NOT NULL,
    finished_at_ms BIGINT,
    revision BIGINT NOT NULL,
    body_json VARCHAR NOT NULL,
    UNIQUE (task_cid, attempt_number)
);

CREATE TABLE attempt_phases (
    attempt_id VARCHAR NOT NULL,
    phase VARCHAR NOT NULL,
    entered_at_ms BIGINT NOT NULL,
    exited_at_ms BIGINT,
    status VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    PRIMARY KEY (attempt_id, phase, entered_at_ms)
);

CREATE TABLE task_claims (
    claim_id VARCHAR PRIMARY KEY,
    task_cid VARCHAR NOT NULL,
    owner_session_id VARCHAR NOT NULL,
    fencing_token BIGINT NOT NULL,
    fence_epoch BIGINT NOT NULL,
    expires_at_ms BIGINT NOT NULL,
    state VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    UNIQUE (task_cid, fencing_token)
);

CREATE TABLE provider_invocations (
    invocation_id VARCHAR PRIMARY KEY,
    attempt_id VARCHAR NOT NULL,
    provider_id VARCHAR NOT NULL,
    model_id VARCHAR NOT NULL,
    started_at_ms BIGINT NOT NULL,
    finished_at_ms BIGINT,
    outcome VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE validation_runs (
    run_id VARCHAR PRIMARY KEY,
    task_cid VARCHAR NOT NULL,
    attempt_id VARCHAR NOT NULL,
    worktree_id VARCHAR NOT NULL,
    started_at_ms BIGINT NOT NULL,
    finished_at_ms BIGINT,
    status VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE validation_results (
    result_id VARCHAR PRIMARY KEY,
    run_id VARCHAR NOT NULL,
    task_cid VARCHAR NOT NULL,
    outcome VARCHAR NOT NULL,
    evidence_cid VARCHAR NOT NULL,
    created_at_ms BIGINT NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE merge_attempts (
    merge_attempt_id VARCHAR PRIMARY KEY,
    task_cid VARCHAR NOT NULL,
    attempt_id VARCHAR NOT NULL,
    worktree_id VARCHAR NOT NULL,
    state VARCHAR NOT NULL,
    fencing_token BIGINT NOT NULL,
    fence_epoch BIGINT NOT NULL,
    started_at_ms BIGINT NOT NULL,
    finished_at_ms BIGINT,
    revision BIGINT NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE recovery_actions (
    action_id VARCHAR PRIMARY KEY,
    subject_kind VARCHAR NOT NULL,
    subject_id VARCHAR NOT NULL,
    action VARCHAR NOT NULL,
    decided_at_ms BIGINT NOT NULL,
    outcome VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE idempotency_records (
    idempotency_key VARCHAR PRIMARY KEY,
    command_kind VARCHAR NOT NULL,
    request_digest VARCHAR NOT NULL,
    response_digest VARCHAR NOT NULL,
    created_at_ms BIGINT NOT NULL,
    expires_at_ms BIGINT NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE effect_claims (
    effect_id VARCHAR PRIMARY KEY,
    task_cid VARCHAR NOT NULL,
    attempt_id VARCHAR NOT NULL,
    effect_kind VARCHAR NOT NULL,
    state VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE completion_receipts (
    receipt_cid VARCHAR PRIMARY KEY,
    task_cid VARCHAR NOT NULL,
    goal_cid VARCHAR NOT NULL,
    attempt_id VARCHAR NOT NULL,
    validation_result_id VARCHAR NOT NULL,
    merge_attempt_id VARCHAR NOT NULL,
    created_at_ms BIGINT NOT NULL,
    body_json VARCHAR NOT NULL
);

-- ---------------------------------------------------------------------------
-- evidence / events / metrics
-- ---------------------------------------------------------------------------

CREATE TABLE domain_events (
    event_id VARCHAR PRIMARY KEY,
    stream_id VARCHAR NOT NULL,
    sequence BIGINT NOT NULL,
    event_type VARCHAR NOT NULL,
    task_cid VARCHAR NOT NULL DEFAULT '',
    attempt_id VARCHAR NOT NULL DEFAULT '',
    observed_at_ms BIGINT NOT NULL,
    body_json VARCHAR NOT NULL,
    UNIQUE (stream_id, sequence)
);

CREATE TABLE structured_logs (
    log_id VARCHAR PRIMARY KEY,
    severity VARCHAR NOT NULL,
    component VARCHAR NOT NULL,
    trace_id VARCHAR NOT NULL,
    span_id VARCHAR NOT NULL,
    task_cid VARCHAR NOT NULL DEFAULT '',
    attempt_id VARCHAR NOT NULL DEFAULT '',
    session_id VARCHAR NOT NULL DEFAULT '',
    observed_at_ms BIGINT NOT NULL,
    message VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE metrics (
    metric_id VARCHAR PRIMARY KEY,
    metric_name VARCHAR NOT NULL,
    unit VARCHAR NOT NULL,
    description VARCHAR NOT NULL
);

CREATE TABLE metric_samples (
    sample_id VARCHAR PRIMARY KEY,
    metric_id VARCHAR NOT NULL,
    observed_at_ms BIGINT NOT NULL,
    value_milli BIGINT NOT NULL,
    labels_json VARCHAR NOT NULL
);

CREATE TABLE budget_reservations (
    reservation_id VARCHAR PRIMARY KEY,
    budget_kind VARCHAR NOT NULL,
    subject_id VARCHAR NOT NULL,
    amount BIGINT NOT NULL,
    reserved_at_ms BIGINT NOT NULL,
    expires_at_ms BIGINT NOT NULL,
    state VARCHAR NOT NULL
);

CREATE TABLE budget_consumption (
    consumption_id VARCHAR PRIMARY KEY,
    reservation_id VARCHAR NOT NULL,
    amount BIGINT NOT NULL,
    consumed_at_ms BIGINT NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE quack_query_telemetry (
    sample_id VARCHAR PRIMARY KEY,
    session_id VARCHAR NOT NULL,
    observed_at_ms BIGINT NOT NULL,
    latency_ms BIGINT NOT NULL,
    outcome VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE evidence_nodes (
    evidence_id VARCHAR PRIMARY KEY,
    evidence_cid VARCHAR NOT NULL UNIQUE,
    evidence_kind VARCHAR NOT NULL,
    subject_id VARCHAR NOT NULL,
    created_at_ms BIGINT NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE artifacts (
    cid VARCHAR PRIMARY KEY,
    media_type VARCHAR NOT NULL,
    byte_length BIGINT NOT NULL,
    digest VARCHAR NOT NULL,
    storage_uri VARCHAR NOT NULL,
    provenance_json VARCHAR NOT NULL
);

-- ---------------------------------------------------------------------------
-- code intelligence
-- ---------------------------------------------------------------------------

CREATE TABLE source_snapshots (
    snapshot_id VARCHAR PRIMARY KEY,
    repository_id VARCHAR NOT NULL,
    tree_id VARCHAR NOT NULL,
    overlay_digest VARCHAR NOT NULL,
    observed_at VARCHAR NOT NULL,
    UNIQUE (repository_id, tree_id, overlay_digest)
);

CREATE TABLE source_files (
    file_id VARCHAR PRIMARY KEY,
    snapshot_id VARCHAR NOT NULL,
    path VARCHAR NOT NULL,
    content_digest VARCHAR NOT NULL,
    language VARCHAR NOT NULL,
    byte_length BIGINT NOT NULL,
    UNIQUE (snapshot_id, path)
);

CREATE TABLE file_versions (
    file_id VARCHAR NOT NULL,
    version BIGINT NOT NULL,
    content_digest VARCHAR NOT NULL,
    created_at VARCHAR NOT NULL,
    PRIMARY KEY (file_id, version)
);

CREATE TABLE parse_runs (
    parse_run_id VARCHAR PRIMARY KEY,
    snapshot_id VARCHAR NOT NULL,
    parser_id VARCHAR NOT NULL,
    parser_version VARCHAR NOT NULL,
    started_at VARCHAR NOT NULL,
    finished_at VARCHAR,
    status VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE symbols (
    symbol_id VARCHAR PRIMARY KEY,
    snapshot_id VARCHAR NOT NULL,
    path VARCHAR NOT NULL,
    language VARCHAR NOT NULL,
    symbol_name VARCHAR NOT NULL,
    symbol_kind VARCHAR NOT NULL,
    fingerprint VARCHAR NOT NULL,
    UNIQUE (snapshot_id, path, symbol_name, symbol_kind, fingerprint)
);

CREATE TABLE symbol_versions (
    symbol_id VARCHAR NOT NULL,
    version BIGINT NOT NULL,
    fingerprint VARCHAR NOT NULL,
    created_at VARCHAR NOT NULL,
    PRIMARY KEY (symbol_id, version)
);

CREATE TABLE ast_nodes (
    node_id VARCHAR PRIMARY KEY,
    snapshot_id VARCHAR NOT NULL,
    path VARCHAR NOT NULL,
    language VARCHAR NOT NULL,
    parser_id VARCHAR NOT NULL,
    node_path VARCHAR NOT NULL,
    node_kind VARCHAR NOT NULL,
    fingerprint VARCHAR NOT NULL,
    UNIQUE (snapshot_id, path, parser_id, node_path)
);

CREATE TABLE ast_edges (
    parent_node_id VARCHAR NOT NULL,
    child_node_id VARCHAR NOT NULL,
    edge_kind VARCHAR NOT NULL,
    ordinal BIGINT NOT NULL,
    PRIMARY KEY (parent_node_id, child_node_id, edge_kind)
);

CREATE TABLE imports (
    import_id VARCHAR PRIMARY KEY,
    snapshot_id VARCHAR NOT NULL,
    path VARCHAR NOT NULL,
    imported_module VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE calls (
    call_id VARCHAR PRIMARY KEY,
    snapshot_id VARCHAR NOT NULL,
    caller_symbol_id VARCHAR NOT NULL,
    callee_symbol_id VARCHAR NOT NULL,
    path VARCHAR NOT NULL
);

CREATE TABLE symbol_references (
    reference_id VARCHAR PRIMARY KEY,
    snapshot_id VARCHAR NOT NULL,
    symbol_id VARCHAR NOT NULL,
    path VARCHAR NOT NULL,
    reference_kind VARCHAR NOT NULL
);

CREATE TABLE definitions (
    definition_id VARCHAR PRIMARY KEY,
    snapshot_id VARCHAR NOT NULL,
    symbol_id VARCHAR NOT NULL,
    path VARCHAR NOT NULL,
    node_id VARCHAR NOT NULL
);

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
    revision BIGINT NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE mutation_files (
    mutation_id VARCHAR NOT NULL,
    path VARCHAR NOT NULL,
    before_digest VARCHAR NOT NULL,
    after_digest VARCHAR NOT NULL,
    PRIMARY KEY (mutation_id, path)
);

CREATE TABLE mutation_hunks (
    mutation_id VARCHAR NOT NULL,
    path VARCHAR NOT NULL,
    hunk_index BIGINT NOT NULL,
    hunk_digest VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL,
    PRIMARY KEY (mutation_id, path, hunk_index)
);

CREATE TABLE ast_mutations (
    mutation_id VARCHAR NOT NULL,
    edit_index BIGINT NOT NULL,
    before_node_id VARCHAR NOT NULL,
    after_node_id VARCHAR NOT NULL,
    edit_kind VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL,
    PRIMARY KEY (mutation_id, edit_index)
);

CREATE TABLE impact_edges (
    edge_id VARCHAR PRIMARY KEY,
    from_symbol_id VARCHAR NOT NULL,
    to_symbol_id VARCHAR NOT NULL,
    edge_kind VARCHAR NOT NULL
);

CREATE TABLE impact_closures (
    closure_id VARCHAR PRIMARY KEY,
    root_symbol_id VARCHAR NOT NULL,
    snapshot_id VARCHAR NOT NULL,
    member_count BIGINT NOT NULL,
    content_cid VARCHAR NOT NULL,
    computed_at VARCHAR NOT NULL
);

CREATE TABLE repair_candidates (
    candidate_id VARCHAR PRIMARY KEY,
    task_cid VARCHAR NOT NULL,
    attempt_id VARCHAR NOT NULL,
    content_cid VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE repair_applications (
    application_id VARCHAR PRIMARY KEY,
    candidate_id VARCHAR NOT NULL,
    mutation_id VARCHAR NOT NULL,
    applied_at VARCHAR NOT NULL,
    outcome VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE proof_obligations (
    obligation_id VARCHAR PRIMARY KEY,
    task_cid VARCHAR NOT NULL,
    subject_id VARCHAR NOT NULL,
    obligation_kind VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE proof_attempts (
    proof_attempt_id VARCHAR PRIMARY KEY,
    obligation_id VARCHAR NOT NULL,
    prover_id VARCHAR NOT NULL,
    started_at VARCHAR NOT NULL,
    finished_at VARCHAR,
    outcome VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE counterexamples (
    counterexample_id VARCHAR PRIMARY KEY,
    proof_attempt_id VARCHAR NOT NULL,
    content_cid VARCHAR NOT NULL,
    created_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);

-- ---------------------------------------------------------------------------
-- cache / context / LLM economy
-- ---------------------------------------------------------------------------

CREATE TABLE context_manifests (
    manifest_id VARCHAR PRIMARY KEY,
    manifest_cid VARCHAR NOT NULL UNIQUE,
    repository_id VARCHAR NOT NULL,
    task_cid VARCHAR NOT NULL,
    schema_revision BIGINT NOT NULL,
    created_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE context_members (
    manifest_id VARCHAR NOT NULL,
    member_ordinal BIGINT NOT NULL,
    member_kind VARCHAR NOT NULL,
    content_cid VARCHAR NOT NULL,
    byte_length BIGINT NOT NULL,
    PRIMARY KEY (manifest_id, member_ordinal)
);

CREATE TABLE context_deltas (
    delta_id VARCHAR PRIMARY KEY,
    from_manifest_id VARCHAR NOT NULL,
    to_manifest_id VARCHAR NOT NULL,
    content_cid VARCHAR NOT NULL,
    created_at VARCHAR NOT NULL
);

CREATE TABLE prompt_templates (
    template_id VARCHAR PRIMARY KEY,
    template_cid VARCHAR NOT NULL UNIQUE,
    name VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE prompt_instances (
    instance_id VARCHAR PRIMARY KEY,
    template_id VARCHAR NOT NULL,
    manifest_id VARCHAR NOT NULL,
    content_cid VARCHAR NOT NULL,
    created_at VARCHAR NOT NULL
);

CREATE TABLE prompt_inputs (
    instance_id VARCHAR NOT NULL,
    input_ordinal BIGINT NOT NULL,
    input_kind VARCHAR NOT NULL,
    content_cid VARCHAR NOT NULL,
    PRIMARY KEY (instance_id, input_ordinal)
);

CREATE TABLE provider_calls (
    call_id VARCHAR PRIMARY KEY,
    attempt_id VARCHAR NOT NULL,
    provider_id VARCHAR NOT NULL,
    model_id VARCHAR NOT NULL,
    prompt_instance_id VARCHAR NOT NULL,
    started_at_ms BIGINT NOT NULL,
    finished_at_ms BIGINT,
    outcome VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE provider_responses (
    response_id VARCHAR PRIMARY KEY,
    call_id VARCHAR NOT NULL,
    content_cid VARCHAR NOT NULL,
    token_input BIGINT NOT NULL,
    token_output BIGINT NOT NULL,
    created_at_ms BIGINT NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE failure_signatures (
    signature_id VARCHAR PRIMARY KEY,
    signature_cid VARCHAR NOT NULL UNIQUE,
    task_cid VARCHAR NOT NULL,
    kind VARCHAR NOT NULL,
    created_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE decision_cache_entries (
    cache_key VARCHAR PRIMARY KEY,
    decision_cid VARCHAR NOT NULL,
    created_at_ms BIGINT NOT NULL,
    expires_at_ms BIGINT NOT NULL,
    hit_count BIGINT NOT NULL DEFAULT 0,
    body_json VARCHAR NOT NULL
);

CREATE TABLE replay_suppressions (
    suppression_id VARCHAR PRIMARY KEY,
    signature_id VARCHAR NOT NULL,
    task_cid VARCHAR NOT NULL,
    until_ms BIGINT NOT NULL,
    reason_code VARCHAR NOT NULL
);

CREATE TABLE churn_metrics (
    metric_id VARCHAR PRIMARY KEY,
    observed_at_ms BIGINT NOT NULL,
    task_cid VARCHAR NOT NULL,
    duplicate_context_fraction_milli BIGINT NOT NULL,
    cache_hit_rate_milli BIGINT NOT NULL,
    provider_calls BIGINT NOT NULL,
    body_json VARCHAR NOT NULL
);

-- ---------------------------------------------------------------------------
-- control surface projections
-- ---------------------------------------------------------------------------

CREATE TABLE control_surfaces (
    surface_id VARCHAR PRIMARY KEY,
    surface_name VARCHAR NOT NULL UNIQUE,
    authority_class VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE control_operations (
    operation_id VARCHAR PRIMARY KEY,
    surface_id VARCHAR NOT NULL,
    operation_name VARCHAR NOT NULL,
    effect_kind VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    body_json VARCHAR NOT NULL,
    UNIQUE (surface_id, operation_name)
);

CREATE TABLE control_authorization_decisions (
    decision_id VARCHAR PRIMARY KEY,
    operation_id VARCHAR NOT NULL,
    principal_id VARCHAR NOT NULL,
    verdict VARCHAR NOT NULL,
    decided_at_ms BIGINT NOT NULL,
    body_json VARCHAR NOT NULL
);

-- ---------------------------------------------------------------------------
-- improve / self-improvement rollout
-- ---------------------------------------------------------------------------

CREATE TABLE improve_experiments (
    experiment_id VARCHAR PRIMARY KEY,
    experiment_cid VARCHAR NOT NULL UNIQUE,
    status VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE improve_rollouts (
    rollout_id VARCHAR PRIMARY KEY,
    experiment_id VARCHAR NOT NULL,
    stage VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE improve_decisions (
    decision_id VARCHAR PRIMARY KEY,
    rollout_id VARCHAR NOT NULL,
    decision VARCHAR NOT NULL,
    decided_at_ms BIGINT NOT NULL,
    evidence_cid VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);

-- ---------------------------------------------------------------------------
-- constrained diagnostic / context views
-- ---------------------------------------------------------------------------

CREATE VIEW ready_task_context_v1 AS
SELECT
    t.task_cid,
    t.task_alias,
    t.goal_cid,
    t.status AS task_status,
    t.revision AS task_revision,
    t.updated_at_ms,
    l.claim_cid,
    l.claimant_did,
    l.fencing_token,
    l.logical_epoch,
    l.expires_at_ms AS lease_expires_at_ms,
    l.state AS lease_state,
    l.retry_not_before_ms,
    l.owner_session_id
FROM tasks AS t
LEFT JOIN leases AS l ON l.task_cid = t.task_cid
WHERE t.status IN ('proposed', 'admitted', 'pending', 'ready', 'retrying');

CREATE VIEW active_lease_v1 AS
SELECT
    task_cid,
    claim_cid,
    claimant_did,
    fencing_token,
    logical_epoch,
    fence_epoch,
    expires_at_ms,
    attempt,
    state,
    started_at_ms,
    retry_not_before_ms,
    owner_session_id,
    revision
FROM leases
WHERE state = 'accepted';

CREATE VIEW open_task_dependency_v1 AS
SELECT
    d.task_cid,
    d.dependency_task_cid,
    d.kind,
    dep.status AS dependency_status,
    dep.revision AS dependency_revision
FROM task_dependencies AS d
JOIN tasks AS dep ON dep.task_cid = d.dependency_task_cid
WHERE dep.status NOT IN ('completed', 'skipped', 'cancelled');

CREATE VIEW schema_domain_catalog_v1 AS
SELECT
    contract_id,
    domain,
    table_name,
    interface_name,
    schema_name,
    revision
FROM schema_contracts;

