-- DQP-005: Normalized agent-supervisor control-plane schema (version 1).
-- Applied transactionally by ControlPlaneMigrationRunner after bookkeeping
-- tables (control_plane_metadata, schema_migrations, schema_migration_attempts)
-- are installed. Join-critical identities are first-class columns; JSON is
-- allowed only for registered, bounded extension payloads.

-- ---------------------------------------------------------------------------
-- Domain: meta / control (schema, deployment, authority)
-- ---------------------------------------------------------------------------

CREATE TABLE schema_contracts (
    contract_id VARCHAR PRIMARY KEY,
    domain VARCHAR NOT NULL,
    interface_name VARCHAR NOT NULL,
    schema_version VARCHAR NOT NULL,
    checksum VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL,
    created_at VARCHAR NOT NULL,
    revision BIGINT NOT NULL
);

CREATE TABLE state_servers (
    server_id VARCHAR PRIMARY KEY,
    database_uuid VARCHAR NOT NULL,
    process_birth_id VARCHAR NOT NULL,
    listen_uri VARCHAR NOT NULL,
    extension_fingerprint VARCHAR NOT NULL,
    duckdb_version VARCHAR NOT NULL,
    quack_profile_id VARCHAR NOT NULL,
    credential_generation BIGINT NOT NULL,
    startup_epoch BIGINT NOT NULL,
    status VARCHAR NOT NULL,
    started_at VARCHAR NOT NULL,
    last_heartbeat_at VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE server_epochs (
    server_id VARCHAR NOT NULL,
    epoch BIGINT NOT NULL,
    started_at VARCHAR NOT NULL,
    ended_at VARCHAR,
    reason VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL,
    PRIMARY KEY (server_id, epoch)
);

CREATE TABLE client_sessions (
    session_id VARCHAR PRIMARY KEY,
    server_id VARCHAR NOT NULL,
    client_identity VARCHAR NOT NULL,
    process_birth_id VARCHAR NOT NULL,
    opened_at VARCHAR NOT NULL,
    expires_at VARCHAR NOT NULL,
    fencing_epoch BIGINT NOT NULL,
    status VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE capability_snapshots (
    snapshot_id VARCHAR PRIMARY KEY,
    server_id VARCHAR NOT NULL,
    profile_id VARCHAR NOT NULL,
    duckdb_version VARCHAR NOT NULL,
    extension_fingerprint VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    observed_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE credentials (
    credential_id VARCHAR PRIMARY KEY,
    secret_handle VARCHAR NOT NULL,
    purpose VARCHAR NOT NULL,
    generation BIGINT NOT NULL,
    created_at VARCHAR NOT NULL,
    rotated_at VARCHAR,
    revoked_at VARCHAR,
    body_json VARCHAR NOT NULL
);

CREATE TABLE authorization_roles (
    role_id VARCHAR PRIMARY KEY,
    role_name VARCHAR NOT NULL UNIQUE,
    description VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE authorization_grants (
    grant_id VARCHAR PRIMARY KEY,
    principal_id VARCHAR NOT NULL,
    role_id VARCHAR NOT NULL,
    scope VARCHAR NOT NULL,
    granted_at VARCHAR NOT NULL,
    expires_at VARCHAR,
    revision BIGINT NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE backup_snapshots (
    snapshot_id VARCHAR PRIMARY KEY,
    database_uuid VARCHAR NOT NULL,
    schema_version INTEGER NOT NULL,
    schema_fingerprint VARCHAR NOT NULL,
    artifact_cid VARCHAR NOT NULL,
    created_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE restore_receipts (
    receipt_id VARCHAR PRIMARY KEY,
    snapshot_id VARCHAR NOT NULL,
    restored_at VARCHAR NOT NULL,
    outcome VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE maintenance_leases (
    lease_id VARCHAR PRIMARY KEY,
    scope VARCHAR NOT NULL,
    owner_session_id VARCHAR NOT NULL,
    fencing_epoch BIGINT NOT NULL,
    expires_at VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    body_json VARCHAR NOT NULL
);

-- ---------------------------------------------------------------------------
-- Domain: git (repository and worktree forest)
-- ---------------------------------------------------------------------------

CREATE TABLE repositories (
    repository_id VARCHAR PRIMARY KEY,
    canonical_root VARCHAR NOT NULL,
    remote_url_digest VARCHAR NOT NULL,
    head_commit_id VARCHAR NOT NULL,
    scanner_version VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    observed_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE repository_revisions (
    repository_id VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    commit_id VARCHAR NOT NULL,
    tree_digest VARCHAR NOT NULL,
    observed_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL,
    PRIMARY KEY (repository_id, revision)
);

CREATE TABLE submodule_edges (
    parent_repository_id VARCHAR NOT NULL,
    child_repository_id VARCHAR NOT NULL,
    path VARCHAR NOT NULL,
    commit_id VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL,
    PRIMARY KEY (parent_repository_id, path)
);

CREATE TABLE worktrees (
    worktree_id VARCHAR PRIMARY KEY,
    repository_id VARCHAR NOT NULL,
    worktree_path_digest VARCHAR NOT NULL,
    head_commit_id VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    owner_session_id VARCHAR,
    revision BIGINT NOT NULL,
    observed_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE worktree_snapshots (
    snapshot_id VARCHAR PRIMARY KEY,
    worktree_id VARCHAR NOT NULL,
    head_commit_id VARCHAR NOT NULL,
    index_digest VARCHAR NOT NULL,
    overlay_digest VARCHAR NOT NULL,
    observed_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE worktree_paths (
    worktree_id VARCHAR NOT NULL,
    path VARCHAR NOT NULL,
    path_kind VARCHAR NOT NULL,
    blob_digest VARCHAR,
    observed_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL,
    PRIMARY KEY (worktree_id, path)
);

CREATE TABLE dirty_overlays (
    overlay_id VARCHAR PRIMARY KEY,
    worktree_id VARCHAR NOT NULL,
    overlay_digest VARCHAR NOT NULL,
    observed_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE branches (
    repository_id VARCHAR NOT NULL,
    branch_name VARCHAR NOT NULL,
    tip_commit_id VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    observed_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL,
    PRIMARY KEY (repository_id, branch_name)
);

CREATE TABLE git_refs (
    repository_id VARCHAR NOT NULL,
    ref_name VARCHAR NOT NULL,
    object_id VARCHAR NOT NULL,
    ref_type VARCHAR NOT NULL,
    observed_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL,
    PRIMARY KEY (repository_id, ref_name)
);

CREATE TABLE merge_bases (
    left_commit_id VARCHAR NOT NULL,
    right_commit_id VARCHAR NOT NULL,
    merge_base_commit_id VARCHAR NOT NULL,
    computed_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL,
    PRIMARY KEY (left_commit_id, right_commit_id)
);

CREATE TABLE merge_queue_entries (
    entry_id VARCHAR PRIMARY KEY,
    repository_id VARCHAR NOT NULL,
    task_cid VARCHAR NOT NULL,
    branch_name VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    ordinal BIGINT NOT NULL,
    revision BIGINT NOT NULL,
    created_at VARCHAR NOT NULL,
    updated_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE resource_claims (
    claim_id VARCHAR PRIMARY KEY,
    resource_kind VARCHAR NOT NULL,
    resource_key VARCHAR NOT NULL,
    owner_session_id VARCHAR NOT NULL,
    fencing_epoch BIGINT NOT NULL,
    expires_at VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    body_json VARCHAR NOT NULL,
    UNIQUE (resource_kind, resource_key)
);

CREATE TABLE path_claims (
    claim_id VARCHAR PRIMARY KEY,
    repository_id VARCHAR NOT NULL,
    path VARCHAR NOT NULL,
    owner_session_id VARCHAR NOT NULL,
    fencing_epoch BIGINT NOT NULL,
    expires_at VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    body_json VARCHAR NOT NULL,
    UNIQUE (repository_id, path)
);

-- Lease row preserves existing LeaseCoordinator semantics (task_cid PK,
-- claim/resolution CIDs, fencing token, epoch, expiry, attempt, state).
CREATE TABLE leases (
    task_cid VARCHAR PRIMARY KEY,
    claim_cid VARCHAR NOT NULL,
    resolution_cid VARCHAR NOT NULL,
    claimant_did VARCHAR NOT NULL,
    owner_session_id VARCHAR NOT NULL,
    logical_epoch BIGINT NOT NULL,
    fencing_token BIGINT NOT NULL,
    expires_at_ms BIGINT NOT NULL,
    attempt BIGINT NOT NULL,
    state VARCHAR NOT NULL,
    started_at_ms BIGINT NOT NULL,
    release_reason VARCHAR,
    retry_not_before_ms BIGINT NOT NULL,
    revision BIGINT NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE lease_events (
    event_id VARCHAR PRIMARY KEY,
    task_cid VARCHAR NOT NULL,
    claim_cid VARCHAR NOT NULL,
    event_type VARCHAR NOT NULL,
    fencing_token BIGINT NOT NULL,
    observed_at_ms BIGINT NOT NULL,
    sequence BIGINT NOT NULL,
    body_json VARCHAR NOT NULL,
    UNIQUE (task_cid, sequence)
);

CREATE TABLE token_history (
    task_cid VARCHAR NOT NULL,
    fencing_token BIGINT NOT NULL,
    recorded_at_ms BIGINT NOT NULL,
    body_json VARCHAR NOT NULL,
    PRIMARY KEY (task_cid, fencing_token)
);

-- ---------------------------------------------------------------------------
-- Domain: intent (objectives, goals, plans, tasks)
-- ---------------------------------------------------------------------------

CREATE TABLE objectives (
    objective_id VARCHAR PRIMARY KEY,
    objective_cid VARCHAR NOT NULL UNIQUE,
    title VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    priority VARCHAR NOT NULL,
    track VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    created_at VARCHAR NOT NULL,
    updated_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE objective_revisions (
    objective_id VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    objective_cid VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    recorded_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL,
    PRIMARY KEY (objective_id, revision)
);

CREATE TABLE goals (
    goal_cid VARCHAR PRIMARY KEY,
    goal_alias VARCHAR NOT NULL UNIQUE,
    parent_goal_cid VARCHAR NOT NULL,
    objective_id VARCHAR,
    ordinal BIGINT NOT NULL,
    title VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    created_at VARCHAR NOT NULL,
    updated_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL,
    UNIQUE (ordinal)
);

CREATE TABLE goal_edges (
    parent_goal_cid VARCHAR NOT NULL,
    child_goal_cid VARCHAR NOT NULL,
    edge_kind VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL,
    PRIMARY KEY (parent_goal_cid, child_goal_cid, edge_kind)
);

CREATE TABLE plans (
    plan_id VARCHAR PRIMARY KEY,
    plan_cid VARCHAR NOT NULL UNIQUE,
    goal_cid VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    created_at VARCHAR NOT NULL,
    updated_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE plan_revisions (
    plan_id VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    plan_cid VARCHAR NOT NULL,
    recorded_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL,
    PRIMARY KEY (plan_id, revision)
);

CREATE TABLE planning_decisions (
    decision_id VARCHAR PRIMARY KEY,
    plan_id VARCHAR NOT NULL,
    decision_kind VARCHAR NOT NULL,
    recorded_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE plan_candidates (
    candidate_id VARCHAR PRIMARY KEY,
    plan_id VARCHAR NOT NULL,
    candidate_cid VARCHAR NOT NULL,
    rank_ordinal BIGINT NOT NULL,
    status VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL,
    UNIQUE (plan_id, candidate_cid)
);

-- Tasks preserve canonical task_cid as the durable primary key.
CREATE TABLE tasks (
    task_cid VARCHAR PRIMARY KEY,
    task_alias VARCHAR NOT NULL UNIQUE,
    goal_cid VARCHAR NOT NULL,
    plan_id VARCHAR,
    objective_id VARCHAR,
    ordinal BIGINT NOT NULL,
    status VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    semantic_fingerprint VARCHAR NOT NULL,
    canonical_task_key VARCHAR NOT NULL,
    idempotency_key VARCHAR NOT NULL,
    created_at VARCHAR NOT NULL,
    updated_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL,
    UNIQUE (ordinal)
);

CREATE TABLE task_revisions (
    task_cid VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    status VARCHAR NOT NULL,
    recorded_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL,
    PRIMARY KEY (task_cid, revision)
);

CREATE TABLE task_dependencies (
    task_cid VARCHAR NOT NULL,
    dependency_task_cid VARCHAR NOT NULL,
    kind VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL,
    PRIMARY KEY (task_cid, dependency_task_cid, kind)
);

CREATE TABLE task_outputs (
    task_cid VARCHAR NOT NULL,
    ordinal BIGINT NOT NULL,
    path VARCHAR NOT NULL,
    effect_kind VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL,
    PRIMARY KEY (task_cid, ordinal),
    UNIQUE (task_cid, path)
);

CREATE TABLE task_acceptance (
    task_cid VARCHAR NOT NULL,
    ordinal BIGINT NOT NULL,
    criterion VARCHAR NOT NULL,
    evidence_policy VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL,
    PRIMARY KEY (task_cid, ordinal)
);

CREATE TABLE task_validations (
    task_cid VARCHAR NOT NULL,
    ordinal BIGINT NOT NULL,
    command_text VARCHAR NOT NULL,
    policy_name VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL,
    PRIMARY KEY (task_cid, ordinal)
);

CREATE TABLE task_assignments (
    assignment_id VARCHAR PRIMARY KEY,
    task_cid VARCHAR NOT NULL,
    assignee_session_id VARCHAR NOT NULL,
    assigned_at VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE task_blocks (
    block_id VARCHAR PRIMARY KEY,
    task_cid VARCHAR NOT NULL,
    block_kind VARCHAR NOT NULL,
    reason VARCHAR NOT NULL,
    created_at VARCHAR NOT NULL,
    cleared_at VARCHAR,
    body_json VARCHAR NOT NULL
);

CREATE TABLE refill_epochs (
    epoch_id VARCHAR PRIMARY KEY,
    goal_cid VARCHAR NOT NULL,
    epoch BIGINT NOT NULL,
    started_at VARCHAR NOT NULL,
    ended_at VARCHAR,
    body_json VARCHAR NOT NULL,
    UNIQUE (goal_cid, epoch)
);

CREATE TABLE findings (
    finding_id VARCHAR PRIMARY KEY,
    finding_cid VARCHAR NOT NULL UNIQUE,
    goal_cid VARCHAR NOT NULL,
    task_cid VARCHAR,
    finding_kind VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    created_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE finding_dispositions (
    disposition_id VARCHAR PRIMARY KEY,
    finding_id VARCHAR NOT NULL,
    disposition VARCHAR NOT NULL,
    recorded_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);

-- ---------------------------------------------------------------------------
-- Domain: schedule / runtime (execution and lifecycle)
-- ---------------------------------------------------------------------------

CREATE TABLE supervisor_instances (
    supervisor_id VARCHAR PRIMARY KEY,
    process_birth_id VARCHAR NOT NULL,
    repository_id VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    started_at VARCHAR NOT NULL,
    last_heartbeat_at VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE daemon_instances (
    daemon_id VARCHAR PRIMARY KEY,
    supervisor_id VARCHAR NOT NULL,
    process_birth_id VARCHAR NOT NULL,
    lane VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    started_at VARCHAR NOT NULL,
    last_heartbeat_at VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE daemon_sessions (
    session_id VARCHAR PRIMARY KEY,
    daemon_id VARCHAR NOT NULL,
    fencing_epoch BIGINT NOT NULL,
    opened_at VARCHAR NOT NULL,
    expires_at VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE heartbeats (
    heartbeat_cid VARCHAR PRIMARY KEY,
    task_cid VARCHAR NOT NULL,
    claimant_did VARCHAR NOT NULL,
    session_id VARCHAR NOT NULL,
    fencing_token BIGINT NOT NULL,
    observed_at_ms BIGINT NOT NULL,
    expires_at_ms BIGINT NOT NULL,
    capacity_millionths BIGINT NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE health_samples (
    sample_id VARCHAR PRIMARY KEY,
    subject_kind VARCHAR NOT NULL,
    subject_id VARCHAR NOT NULL,
    observed_at VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE stall_detections (
    detection_id VARCHAR PRIMARY KEY,
    subject_kind VARCHAR NOT NULL,
    subject_id VARCHAR NOT NULL,
    detected_at VARCHAR NOT NULL,
    stall_kind VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE restart_decisions (
    decision_id VARCHAR PRIMARY KEY,
    subject_kind VARCHAR NOT NULL,
    subject_id VARCHAR NOT NULL,
    decided_at VARCHAR NOT NULL,
    action VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE task_attempts (
    attempt_id VARCHAR PRIMARY KEY,
    task_cid VARCHAR NOT NULL,
    attempt_number BIGINT NOT NULL,
    claim_cid VARCHAR NOT NULL,
    session_id VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    started_at VARCHAR NOT NULL,
    finished_at VARCHAR,
    revision BIGINT NOT NULL,
    body_json VARCHAR NOT NULL,
    UNIQUE (task_cid, attempt_number)
);

CREATE TABLE attempt_phases (
    attempt_id VARCHAR NOT NULL,
    phase_name VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    started_at VARCHAR NOT NULL,
    finished_at VARCHAR,
    revision BIGINT NOT NULL,
    body_json VARCHAR NOT NULL,
    PRIMARY KEY (attempt_id, phase_name)
);

CREATE TABLE task_claims (
    claim_cid VARCHAR PRIMARY KEY,
    task_cid VARCHAR NOT NULL,
    claimant_did VARCHAR NOT NULL,
    session_id VARCHAR NOT NULL,
    fencing_token BIGINT NOT NULL,
    logical_epoch BIGINT NOT NULL,
    claimed_at_ms BIGINT NOT NULL,
    expires_at_ms BIGINT NOT NULL,
    status VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE provider_invocations (
    invocation_id VARCHAR PRIMARY KEY,
    attempt_id VARCHAR NOT NULL,
    task_cid VARCHAR NOT NULL,
    provider_name VARCHAR NOT NULL,
    model_name VARCHAR NOT NULL,
    started_at VARCHAR NOT NULL,
    finished_at VARCHAR,
    outcome VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE validation_runs (
    run_id VARCHAR PRIMARY KEY,
    task_cid VARCHAR NOT NULL,
    attempt_id VARCHAR NOT NULL,
    command_text VARCHAR NOT NULL,
    started_at VARCHAR NOT NULL,
    finished_at VARCHAR,
    outcome VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE validation_results (
    result_id VARCHAR PRIMARY KEY,
    run_id VARCHAR NOT NULL,
    task_cid VARCHAR NOT NULL,
    exit_code INTEGER NOT NULL,
    evidence_cid VARCHAR,
    recorded_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE merge_attempts (
    merge_attempt_id VARCHAR PRIMARY KEY,
    task_cid VARCHAR NOT NULL,
    repository_id VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    started_at VARCHAR NOT NULL,
    finished_at VARCHAR,
    body_json VARCHAR NOT NULL
);

CREATE TABLE recovery_actions (
    action_id VARCHAR PRIMARY KEY,
    subject_kind VARCHAR NOT NULL,
    subject_id VARCHAR NOT NULL,
    action_kind VARCHAR NOT NULL,
    recorded_at VARCHAR NOT NULL,
    outcome VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE idempotency_records (
    idempotency_key VARCHAR PRIMARY KEY,
    scope VARCHAR NOT NULL,
    request_digest VARCHAR NOT NULL,
    result_digest VARCHAR NOT NULL,
    recorded_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE effect_claims (
    effect_id VARCHAR PRIMARY KEY,
    task_cid VARCHAR NOT NULL,
    attempt_id VARCHAR NOT NULL,
    effect_kind VARCHAR NOT NULL,
    effect_digest VARCHAR NOT NULL,
    recorded_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE completion_receipts (
    receipt_cid VARCHAR PRIMARY KEY,
    task_cid VARCHAR NOT NULL,
    attempt_id VARCHAR NOT NULL,
    claim_cid VARCHAR NOT NULL,
    fencing_token BIGINT NOT NULL,
    completed_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);

-- ---------------------------------------------------------------------------
-- Domain: events / metrics
-- ---------------------------------------------------------------------------

CREATE TABLE domain_events (
    event_id VARCHAR PRIMARY KEY,
    stream_id VARCHAR NOT NULL,
    sequence BIGINT NOT NULL,
    event_type VARCHAR NOT NULL,
    task_cid VARCHAR,
    attempt_id VARCHAR,
    session_id VARCHAR,
    recorded_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL,
    UNIQUE (stream_id, sequence)
);

CREATE TABLE structured_logs (
    log_id VARCHAR PRIMARY KEY,
    severity VARCHAR NOT NULL,
    component VARCHAR NOT NULL,
    trace_id VARCHAR NOT NULL,
    span_id VARCHAR NOT NULL,
    task_cid VARCHAR,
    attempt_id VARCHAR,
    session_id VARCHAR,
    recorded_at VARCHAR NOT NULL,
    message VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE metrics (
    metric_id VARCHAR PRIMARY KEY,
    metric_name VARCHAR NOT NULL UNIQUE,
    unit VARCHAR NOT NULL,
    description VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE metric_samples (
    sample_id VARCHAR PRIMARY KEY,
    metric_id VARCHAR NOT NULL,
    observed_at VARCHAR NOT NULL,
    value_number DOUBLE NOT NULL,
    labels_json VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE budget_reservations (
    reservation_id VARCHAR PRIMARY KEY,
    budget_kind VARCHAR NOT NULL,
    subject_id VARCHAR NOT NULL,
    amount BIGINT NOT NULL,
    reserved_at VARCHAR NOT NULL,
    expires_at VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);

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
    query_digest VARCHAR NOT NULL,
    duration_ms BIGINT NOT NULL,
    outcome VARCHAR NOT NULL,
    observed_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);

-- ---------------------------------------------------------------------------
-- Domain: code / evidence (AST, mutations, proofs)
-- ---------------------------------------------------------------------------

CREATE TABLE source_snapshots (
    snapshot_id VARCHAR PRIMARY KEY,
    repository_id VARCHAR NOT NULL,
    tree_digest VARCHAR NOT NULL,
    overlay_digest VARCHAR NOT NULL,
    observed_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE source_files (
    file_id VARCHAR PRIMARY KEY,
    snapshot_id VARCHAR NOT NULL,
    path VARCHAR NOT NULL,
    language VARCHAR NOT NULL,
    blob_digest VARCHAR NOT NULL,
    byte_length BIGINT NOT NULL,
    body_json VARCHAR NOT NULL,
    UNIQUE (snapshot_id, path)
);

CREATE TABLE file_versions (
    file_id VARCHAR NOT NULL,
    version BIGINT NOT NULL,
    blob_digest VARCHAR NOT NULL,
    recorded_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL,
    PRIMARY KEY (file_id, version)
);

CREATE TABLE parse_runs (
    parse_run_id VARCHAR PRIMARY KEY,
    snapshot_id VARCHAR NOT NULL,
    parser_identity VARCHAR NOT NULL,
    started_at VARCHAR NOT NULL,
    finished_at VARCHAR,
    outcome VARCHAR NOT NULL,
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
    body_json VARCHAR NOT NULL,
    UNIQUE (snapshot_id, path, fingerprint)
);

CREATE TABLE symbol_versions (
    symbol_id VARCHAR NOT NULL,
    version BIGINT NOT NULL,
    fingerprint VARCHAR NOT NULL,
    recorded_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL,
    PRIMARY KEY (symbol_id, version)
);

CREATE TABLE ast_nodes (
    node_id VARCHAR PRIMARY KEY,
    snapshot_id VARCHAR NOT NULL,
    path VARCHAR NOT NULL,
    language VARCHAR NOT NULL,
    parser_identity VARCHAR NOT NULL,
    node_path VARCHAR NOT NULL,
    node_kind VARCHAR NOT NULL,
    fingerprint VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL,
    UNIQUE (snapshot_id, path, node_path, parser_identity)
);

CREATE TABLE ast_edges (
    edge_id VARCHAR PRIMARY KEY,
    parent_node_id VARCHAR NOT NULL,
    child_node_id VARCHAR NOT NULL,
    edge_kind VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL,
    UNIQUE (parent_node_id, child_node_id, edge_kind)
);

CREATE TABLE module_imports (
    import_id VARCHAR PRIMARY KEY,
    snapshot_id VARCHAR NOT NULL,
    path VARCHAR NOT NULL,
    imported_module VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE call_edges (
    call_id VARCHAR PRIMARY KEY,
    snapshot_id VARCHAR NOT NULL,
    caller_symbol_id VARCHAR NOT NULL,
    callee_symbol_id VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE code_references (
    reference_id VARCHAR PRIMARY KEY,
    snapshot_id VARCHAR NOT NULL,
    symbol_id VARCHAR NOT NULL,
    path VARCHAR NOT NULL,
    reference_kind VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE definitions (
    definition_id VARCHAR PRIMARY KEY,
    snapshot_id VARCHAR NOT NULL,
    symbol_id VARCHAR NOT NULL,
    path VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE type_relations (
    relation_id VARCHAR PRIMARY KEY,
    snapshot_id VARCHAR NOT NULL,
    left_symbol_id VARCHAR NOT NULL,
    right_symbol_id VARCHAR NOT NULL,
    relation_kind VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE mutations (
    mutation_id VARCHAR PRIMARY KEY,
    mutation_cid VARCHAR NOT NULL UNIQUE,
    task_cid VARCHAR NOT NULL,
    attempt_id VARCHAR NOT NULL,
    before_snapshot_id VARCHAR NOT NULL,
    after_snapshot_id VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    created_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE mutation_files (
    mutation_id VARCHAR NOT NULL,
    path VARCHAR NOT NULL,
    before_digest VARCHAR NOT NULL,
    after_digest VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL,
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
    ast_mutation_id VARCHAR PRIMARY KEY,
    mutation_id VARCHAR NOT NULL,
    before_node_id VARCHAR,
    after_node_id VARCHAR,
    edit_kind VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE impact_edges (
    edge_id VARCHAR PRIMARY KEY,
    mutation_id VARCHAR NOT NULL,
    from_symbol_id VARCHAR NOT NULL,
    to_symbol_id VARCHAR NOT NULL,
    edge_kind VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE impact_closures (
    closure_id VARCHAR PRIMARY KEY,
    mutation_id VARCHAR NOT NULL,
    root_symbol_id VARCHAR NOT NULL,
    member_count BIGINT NOT NULL,
    closure_digest VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE repair_candidates (
    candidate_id VARCHAR PRIMARY KEY,
    task_cid VARCHAR NOT NULL,
    candidate_cid VARCHAR NOT NULL,
    rank_ordinal BIGINT NOT NULL,
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
    obligation_cid VARCHAR NOT NULL UNIQUE,
    task_cid VARCHAR NOT NULL,
    mutation_id VARCHAR,
    status VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE proof_attempts (
    proof_attempt_id VARCHAR PRIMARY KEY,
    obligation_id VARCHAR NOT NULL,
    attempt_number BIGINT NOT NULL,
    outcome VARCHAR NOT NULL,
    recorded_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL,
    UNIQUE (obligation_id, attempt_number)
);

CREATE TABLE counterexamples (
    counterexample_id VARCHAR PRIMARY KEY,
    obligation_id VARCHAR NOT NULL,
    proof_attempt_id VARCHAR NOT NULL,
    counterexample_cid VARCHAR NOT NULL,
    recorded_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE evidence_nodes (
    evidence_id VARCHAR PRIMARY KEY,
    evidence_cid VARCHAR NOT NULL UNIQUE,
    evidence_kind VARCHAR NOT NULL,
    task_cid VARCHAR,
    mutation_id VARCHAR,
    artifact_cid VARCHAR,
    recorded_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE artifacts (
    cid VARCHAR PRIMARY KEY,
    media_type VARCHAR NOT NULL,
    byte_length BIGINT NOT NULL,
    digest VARCHAR NOT NULL,
    storage_uri VARCHAR NOT NULL,
    authority_class VARCHAR NOT NULL,
    created_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);

-- ---------------------------------------------------------------------------
-- Domain: cache / context (LLM economy)
-- ---------------------------------------------------------------------------

CREATE TABLE context_manifests (
    manifest_cid VARCHAR PRIMARY KEY,
    task_cid VARCHAR NOT NULL,
    repository_id VARCHAR NOT NULL,
    schema_fingerprint VARCHAR NOT NULL,
    created_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE context_members (
    manifest_cid VARCHAR NOT NULL,
    member_ordinal BIGINT NOT NULL,
    member_kind VARCHAR NOT NULL,
    member_cid VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL,
    PRIMARY KEY (manifest_cid, member_ordinal)
);

CREATE TABLE context_deltas (
    delta_id VARCHAR PRIMARY KEY,
    from_manifest_cid VARCHAR NOT NULL,
    to_manifest_cid VARCHAR NOT NULL,
    delta_digest VARCHAR NOT NULL,
    created_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
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
    task_cid VARCHAR NOT NULL,
    manifest_cid VARCHAR NOT NULL,
    prompt_digest VARCHAR NOT NULL,
    created_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE prompt_inputs (
    instance_id VARCHAR NOT NULL,
    input_ordinal BIGINT NOT NULL,
    input_kind VARCHAR NOT NULL,
    input_digest VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL,
    PRIMARY KEY (instance_id, input_ordinal)
);

CREATE TABLE provider_calls (
    call_id VARCHAR PRIMARY KEY,
    instance_id VARCHAR NOT NULL,
    task_cid VARCHAR NOT NULL,
    provider_name VARCHAR NOT NULL,
    model_name VARCHAR NOT NULL,
    started_at VARCHAR NOT NULL,
    finished_at VARCHAR,
    outcome VARCHAR NOT NULL,
    input_tokens BIGINT NOT NULL,
    output_tokens BIGINT NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE provider_responses (
    response_id VARCHAR PRIMARY KEY,
    call_id VARCHAR NOT NULL,
    response_digest VARCHAR NOT NULL,
    recorded_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE failure_signatures (
    signature_id VARCHAR PRIMARY KEY,
    signature_digest VARCHAR NOT NULL UNIQUE,
    task_cid VARCHAR NOT NULL,
    failure_kind VARCHAR NOT NULL,
    recorded_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE decision_cache_entries (
    cache_key VARCHAR PRIMARY KEY,
    decision_digest VARCHAR NOT NULL,
    manifest_cid VARCHAR NOT NULL,
    created_at VARCHAR NOT NULL,
    expires_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE replay_suppressions (
    suppression_id VARCHAR PRIMARY KEY,
    signature_digest VARCHAR NOT NULL,
    task_cid VARCHAR NOT NULL,
    reason VARCHAR NOT NULL,
    recorded_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE churn_metrics (
    metric_id VARCHAR PRIMARY KEY,
    task_cid VARCHAR NOT NULL,
    metric_name VARCHAR NOT NULL,
    value_number DOUBLE NOT NULL,
    observed_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);

-- ---------------------------------------------------------------------------
-- Domain: improve (self-improvement / analysis products)
-- ---------------------------------------------------------------------------

CREATE TABLE improvement_campaigns (
    campaign_id VARCHAR PRIMARY KEY,
    campaign_cid VARCHAR NOT NULL UNIQUE,
    title VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    created_at VARCHAR NOT NULL,
    updated_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE improvement_findings (
    finding_id VARCHAR PRIMARY KEY,
    campaign_id VARCHAR NOT NULL,
    finding_cid VARCHAR NOT NULL UNIQUE,
    status VARCHAR NOT NULL,
    created_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE improvement_actions (
    action_id VARCHAR PRIMARY KEY,
    campaign_id VARCHAR NOT NULL,
    task_cid VARCHAR,
    action_kind VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    recorded_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);

-- ---------------------------------------------------------------------------
-- Constrained diagnostic / context views (read-only projections)
-- ---------------------------------------------------------------------------

CREATE VIEW ready_task_context_v1 AS
SELECT
    t.task_cid,
    t.task_alias,
    t.goal_cid,
    t.plan_id,
    t.objective_id,
    t.status,
    t.revision,
    t.semantic_fingerprint,
    t.canonical_task_key,
    t.idempotency_key,
    t.updated_at,
    l.claim_cid AS lease_claim_cid,
    l.claimant_did AS lease_claimant_did,
    l.fencing_token AS lease_fencing_token,
    l.logical_epoch AS lease_logical_epoch,
    l.expires_at_ms AS lease_expires_at_ms,
    l.state AS lease_state
FROM tasks t
LEFT JOIN leases l ON l.task_cid = t.task_cid
WHERE t.status IN ('ready', 'open', 'todo', 'queued');

CREATE VIEW live_sessions_v1 AS
SELECT
    session_id,
    daemon_id,
    fencing_epoch,
    opened_at,
    expires_at,
    status,
    revision
FROM daemon_sessions
WHERE status = 'active';

CREATE VIEW expiring_leases_v1 AS
SELECT
    task_cid,
    claim_cid,
    claimant_did,
    owner_session_id,
    fencing_token,
    logical_epoch,
    expires_at_ms,
    state,
    revision
FROM leases
WHERE state IN ('claimed', 'active', 'held');

CREATE VIEW ready_versus_claimed_tasks_v1 AS
SELECT
    t.task_cid,
    t.status AS task_status,
    t.revision AS task_revision,
    l.state AS lease_state,
    l.claim_cid,
    l.fencing_token,
    l.expires_at_ms
FROM tasks t
LEFT JOIN leases l ON l.task_cid = t.task_cid
WHERE t.status IN ('ready', 'open', 'todo', 'queued', 'claimed', 'running');

CREATE VIEW stuck_phases_v1 AS
SELECT
    a.attempt_id,
    a.task_cid,
    a.status AS attempt_status,
    p.phase_name,
    p.status AS phase_status,
    p.started_at,
    p.revision
FROM task_attempts a
JOIN attempt_phases p ON p.attempt_id = a.attempt_id
WHERE p.status IN ('running', 'started', 'active')
  AND p.finished_at IS NULL;

CREATE VIEW failed_migrations_v1 AS
SELECT
    attempt_id,
    version,
    migration_id,
    checksum,
    started_at,
    finished_at,
    outcome,
    error_text
FROM schema_migration_attempts
WHERE outcome = 'failed';

CREATE VIEW server_identity_v1 AS
SELECT
    server_id,
    database_uuid,
    process_birth_id,
    listen_uri,
    extension_fingerprint,
    duckdb_version,
    quack_profile_id,
    credential_generation,
    startup_epoch,
    status,
    last_heartbeat_at,
    revision
FROM state_servers;

-- ---------------------------------------------------------------------------
-- Indexes for join-critical identities
-- ---------------------------------------------------------------------------

CREATE INDEX tasks_goal_status_idx ON tasks (goal_cid, status);
CREATE INDEX tasks_status_revision_idx ON tasks (status, revision);
CREATE INDEX task_dependencies_dependency_idx ON task_dependencies (dependency_task_cid);
CREATE INDEX leases_state_expires_idx ON leases (state, expires_at_ms);
CREATE INDEX leases_claimant_idx ON leases (claimant_did, fencing_token);
CREATE INDEX lease_events_task_idx ON lease_events (task_cid, sequence);
CREATE INDEX task_claims_task_idx ON task_claims (task_cid, status);
CREATE INDEX task_attempts_task_idx ON task_attempts (task_cid, attempt_number);
CREATE INDEX domain_events_stream_idx ON domain_events (stream_id, sequence);
CREATE INDEX domain_events_task_idx ON domain_events (task_cid, recorded_at);
CREATE INDEX heartbeats_task_idx ON heartbeats (task_cid, observed_at_ms);
CREATE INDEX worktrees_repository_idx ON worktrees (repository_id, status);
CREATE INDEX mutations_task_idx ON mutations (task_cid, created_at);
CREATE INDEX evidence_nodes_task_idx ON evidence_nodes (task_cid, recorded_at);
CREATE INDEX context_manifests_task_idx ON context_manifests (task_cid, created_at);
CREATE INDEX provider_calls_task_idx ON provider_calls (task_cid, started_at);
CREATE INDEX symbols_snapshot_idx ON symbols (snapshot_id, path);
CREATE INDEX ast_nodes_snapshot_idx ON ast_nodes (snapshot_id, path);

