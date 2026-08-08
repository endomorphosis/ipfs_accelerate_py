-- Control-plane domain schema (ControlPlaneSchema@1 / migration 0001).
-- Bookkeeping tables (control_plane_metadata, schema_migrations,
-- schema_migration_attempts) are installed by the migration runner before this
-- file runs. Domain SQL must not redefine them.
--
-- Logical domains: meta, intent, schedule, runtime, git, code, evidence,
-- cache, control, improve. Join-critical identities are first-class columns
-- (never only inside opaque JSON). Timestamps are UTC ISO-8601 strings.
-- Mutable rows carry revision and/or fence_epoch fields.

-- ---------------------------------------------------------------------------
-- meta: schema contracts and deployment identity
-- ---------------------------------------------------------------------------

CREATE TABLE schema_contracts (
    contract_id VARCHAR PRIMARY KEY,
    domain VARCHAR NOT NULL,
    interface_name VARCHAR NOT NULL,
    schema_name VARCHAR NOT NULL,
    schema_version INTEGER NOT NULL,
    table_name VARCHAR NOT NULL,
    description VARCHAR NOT NULL,
    created_at VARCHAR NOT NULL
);

CREATE TABLE state_servers (
    server_id VARCHAR PRIMARY KEY,
    database_uuid VARCHAR NOT NULL,
    birth_id VARCHAR NOT NULL,
    listen_uri VARCHAR NOT NULL,
    extension_fingerprint VARCHAR NOT NULL,
    tool_version VARCHAR NOT NULL,
    schema_revision INTEGER NOT NULL,
    generation INTEGER NOT NULL,
    started_at VARCHAR NOT NULL,
    stopped_at VARCHAR,
    status VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}'
);

CREATE TABLE server_epochs (
    server_id VARCHAR NOT NULL,
    fence_epoch BIGINT NOT NULL,
    started_at VARCHAR NOT NULL,
    ended_at VARCHAR,
    reason VARCHAR NOT NULL,
    PRIMARY KEY (server_id, fence_epoch)
);

CREATE TABLE client_sessions (
    session_id VARCHAR PRIMARY KEY,
    server_id VARCHAR NOT NULL,
    client_id VARCHAR NOT NULL,
    birth_id VARCHAR NOT NULL,
    opened_at VARCHAR NOT NULL,
    expires_at VARCHAR NOT NULL,
    fence_epoch BIGINT NOT NULL,
    revision BIGINT NOT NULL,
    status VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}'
);

CREATE TABLE capability_snapshots (
    snapshot_id VARCHAR PRIMARY KEY,
    server_id VARCHAR NOT NULL,
    observed_at VARCHAR NOT NULL,
    duckdb_version VARCHAR NOT NULL,
    extension_name VARCHAR NOT NULL,
    extension_fingerprint VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    report_json VARCHAR NOT NULL
);

-- ---------------------------------------------------------------------------
-- control: authorization, credentials (handles only), backup/maintenance
-- ---------------------------------------------------------------------------

CREATE TABLE authorization_roles (
    role_id VARCHAR PRIMARY KEY,
    role_name VARCHAR NOT NULL UNIQUE,
    description VARCHAR NOT NULL,
    created_at VARCHAR NOT NULL,
    revision BIGINT NOT NULL
);

CREATE TABLE authorization_grants (
    grant_id VARCHAR PRIMARY KEY,
    principal_id VARCHAR NOT NULL,
    role_id VARCHAR NOT NULL,
    scope VARCHAR NOT NULL,
    created_at VARCHAR NOT NULL,
    expires_at VARCHAR,
    revision BIGINT NOT NULL,
    status VARCHAR NOT NULL
);

CREATE TABLE credentials (
    credential_id VARCHAR PRIMARY KEY,
    secret_handle VARCHAR NOT NULL,
    purpose VARCHAR NOT NULL,
    generation INTEGER NOT NULL,
    created_at VARCHAR NOT NULL,
    rotated_at VARCHAR,
    status VARCHAR NOT NULL,
    revision BIGINT NOT NULL
);

CREATE TABLE backup_snapshots (
    backup_id VARCHAR PRIMARY KEY,
    database_uuid VARCHAR NOT NULL,
    schema_revision INTEGER NOT NULL,
    schema_fingerprint VARCHAR NOT NULL,
    created_at VARCHAR NOT NULL,
    storage_uri VARCHAR NOT NULL,
    digest VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}'
);

CREATE TABLE restore_receipts (
    receipt_id VARCHAR PRIMARY KEY,
    backup_id VARCHAR NOT NULL,
    restored_at VARCHAR NOT NULL,
    schema_revision INTEGER NOT NULL,
    schema_fingerprint VARCHAR NOT NULL,
    outcome VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}'
);

CREATE TABLE maintenance_leases (
    lease_id VARCHAR PRIMARY KEY,
    owner_session_id VARCHAR NOT NULL,
    scope VARCHAR NOT NULL,
    acquired_at VARCHAR NOT NULL,
    expires_at VARCHAR NOT NULL,
    fence_epoch BIGINT NOT NULL,
    revision BIGINT NOT NULL,
    status VARCHAR NOT NULL
);

-- ---------------------------------------------------------------------------
-- git: repository and worktree forest (Git remains byte authority)
-- ---------------------------------------------------------------------------

CREATE TABLE repositories (
    repository_id VARCHAR PRIMARY KEY,
    root_path VARCHAR NOT NULL,
    common_dir VARCHAR NOT NULL,
    head_commit VARCHAR NOT NULL,
    created_at VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}'
);

CREATE TABLE repository_revisions (
    repository_id VARCHAR NOT NULL,
    revision_id VARCHAR NOT NULL,
    commit_id VARCHAR NOT NULL,
    tree_id VARCHAR NOT NULL,
    observed_at VARCHAR NOT NULL,
    scanner_version VARCHAR NOT NULL,
    PRIMARY KEY (repository_id, revision_id)
);

CREATE TABLE submodule_edges (
    parent_repository_id VARCHAR NOT NULL,
    child_repository_id VARCHAR NOT NULL,
    path VARCHAR NOT NULL,
    gitlink_commit VARCHAR NOT NULL,
    PRIMARY KEY (parent_repository_id, path)
);

CREATE TABLE worktrees (
    worktree_id VARCHAR PRIMARY KEY,
    repository_id VARCHAR NOT NULL,
    path VARCHAR NOT NULL,
    head_commit VARCHAR NOT NULL,
    branch_name VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    owner_session_id VARCHAR,
    fence_epoch BIGINT NOT NULL DEFAULT 0,
    revision BIGINT NOT NULL,
    updated_at VARCHAR NOT NULL
);

CREATE TABLE worktree_snapshots (
    snapshot_id VARCHAR PRIMARY KEY,
    worktree_id VARCHAR NOT NULL,
    repository_id VARCHAR NOT NULL,
    head_commit VARCHAR NOT NULL,
    index_digest VARCHAR NOT NULL,
    overlay_digest VARCHAR NOT NULL,
    observed_at VARCHAR NOT NULL,
    scanner_version VARCHAR NOT NULL
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
    path VARCHAR NOT NULL,
    before_digest VARCHAR NOT NULL,
    after_digest VARCHAR NOT NULL,
    observed_at VARCHAR NOT NULL
);

CREATE TABLE branches (
    repository_id VARCHAR NOT NULL,
    branch_name VARCHAR NOT NULL,
    tip_commit VARCHAR NOT NULL,
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
    repository_id VARCHAR NOT NULL,
    left_commit VARCHAR NOT NULL,
    right_commit VARCHAR NOT NULL,
    merge_base_commit VARCHAR NOT NULL,
    computed_at VARCHAR NOT NULL,
    PRIMARY KEY (repository_id, left_commit, right_commit)
);

CREATE TABLE merge_queue_entries (
    entry_id VARCHAR PRIMARY KEY,
    repository_id VARCHAR NOT NULL,
    task_cid VARCHAR NOT NULL,
    worktree_id VARCHAR,
    source_branch VARCHAR NOT NULL,
    target_branch VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    fence_epoch BIGINT NOT NULL DEFAULT 0,
    revision BIGINT NOT NULL,
    created_at VARCHAR NOT NULL,
    updated_at VARCHAR NOT NULL
);

CREATE TABLE resource_claims (
    claim_id VARCHAR PRIMARY KEY,
    resource_kind VARCHAR NOT NULL,
    resource_key VARCHAR NOT NULL,
    owner_session_id VARCHAR NOT NULL,
    task_cid VARCHAR,
    acquired_at VARCHAR NOT NULL,
    expires_at VARCHAR NOT NULL,
    fence_epoch BIGINT NOT NULL,
    revision BIGINT NOT NULL,
    status VARCHAR NOT NULL
);

CREATE TABLE path_claims (
    claim_id VARCHAR PRIMARY KEY,
    repository_id VARCHAR NOT NULL,
    path VARCHAR NOT NULL,
    owner_session_id VARCHAR NOT NULL,
    task_cid VARCHAR,
    acquired_at VARCHAR NOT NULL,
    expires_at VARCHAR NOT NULL,
    fence_epoch BIGINT NOT NULL,
    revision BIGINT NOT NULL,
    status VARCHAR NOT NULL
);

-- ---------------------------------------------------------------------------
-- intent: objectives, plans, tasks (task_cid is durable identity)
-- ---------------------------------------------------------------------------

CREATE TABLE objectives (
    objective_id VARCHAR PRIMARY KEY,
    objective_cid VARCHAR NOT NULL UNIQUE,
    title VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    priority VARCHAR NOT NULL,
    created_at VARCHAR NOT NULL,
    updated_at VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}'
);

CREATE TABLE objective_revisions (
    objective_id VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    objective_cid VARCHAR NOT NULL,
    snapshot_json VARCHAR NOT NULL,
    created_at VARCHAR NOT NULL,
    PRIMARY KEY (objective_id, revision)
);

CREATE TABLE goals (
    goal_cid VARCHAR PRIMARY KEY,
    goal_alias VARCHAR NOT NULL UNIQUE,
    parent_goal_cid VARCHAR,
    objective_id VARCHAR,
    title VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    ordinal BIGINT NOT NULL,
    created_at VARCHAR NOT NULL,
    updated_at VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}'
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
    goal_cid VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    created_at VARCHAR NOT NULL,
    updated_at VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}'
);

CREATE TABLE plan_revisions (
    plan_id VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    plan_cid VARCHAR NOT NULL,
    snapshot_json VARCHAR NOT NULL,
    created_at VARCHAR NOT NULL,
    PRIMARY KEY (plan_id, revision)
);

CREATE TABLE planning_decisions (
    decision_id VARCHAR PRIMARY KEY,
    plan_id VARCHAR NOT NULL,
    decision_kind VARCHAR NOT NULL,
    created_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE plan_candidates (
    candidate_id VARCHAR PRIMARY KEY,
    plan_id VARCHAR NOT NULL,
    rank INTEGER NOT NULL,
    score_json VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL,
    created_at VARCHAR NOT NULL
);

CREATE TABLE tasks (
    task_cid VARCHAR PRIMARY KEY,
    task_alias VARCHAR NOT NULL UNIQUE,
    goal_cid VARCHAR NOT NULL,
    plan_id VARCHAR,
    status VARCHAR NOT NULL,
    priority VARCHAR NOT NULL,
    ordinal BIGINT NOT NULL,
    track VARCHAR NOT NULL DEFAULT '',
    created_at VARCHAR NOT NULL,
    updated_at VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    fence_epoch BIGINT NOT NULL DEFAULT 0,
    identity_json VARCHAR NOT NULL DEFAULT '{}',
    body_json VARCHAR NOT NULL DEFAULT '{}'
);

CREATE TABLE task_revisions (
    task_cid VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    status VARCHAR NOT NULL,
    snapshot_json VARCHAR NOT NULL,
    created_at VARCHAR NOT NULL,
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
    effect_json VARCHAR NOT NULL DEFAULT '{}',
    PRIMARY KEY (task_cid, ordinal),
    UNIQUE (task_cid, path)
);

CREATE TABLE task_acceptance (
    task_cid VARCHAR NOT NULL,
    ordinal BIGINT NOT NULL,
    criterion VARCHAR NOT NULL,
    evidence_policy_json VARCHAR NOT NULL DEFAULT '{}',
    PRIMARY KEY (task_cid, ordinal)
);

CREATE TABLE task_validations (
    task_cid VARCHAR NOT NULL,
    ordinal BIGINT NOT NULL,
    argv_json VARCHAR NOT NULL,
    policy_json VARCHAR NOT NULL DEFAULT '{}',
    PRIMARY KEY (task_cid, ordinal)
);

CREATE TABLE findings (
    finding_id VARCHAR PRIMARY KEY,
    finding_cid VARCHAR NOT NULL UNIQUE,
    task_cid VARCHAR,
    goal_cid VARCHAR,
    kind VARCHAR NOT NULL,
    severity VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    created_at VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}'
);

CREATE TABLE finding_dispositions (
    disposition_id VARCHAR PRIMARY KEY,
    finding_id VARCHAR NOT NULL,
    disposition VARCHAR NOT NULL,
    decided_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}'
);

-- ---------------------------------------------------------------------------
-- schedule: assignment, blocks, refill epochs
-- ---------------------------------------------------------------------------

CREATE TABLE task_assignments (
    assignment_id VARCHAR PRIMARY KEY,
    task_cid VARCHAR NOT NULL,
    session_id VARCHAR NOT NULL,
    lane_id VARCHAR NOT NULL,
    assigned_at VARCHAR NOT NULL,
    expires_at VARCHAR,
    status VARCHAR NOT NULL,
    fence_epoch BIGINT NOT NULL DEFAULT 0,
    revision BIGINT NOT NULL
);

CREATE TABLE task_blocks (
    block_id VARCHAR PRIMARY KEY,
    task_cid VARCHAR NOT NULL,
    reason VARCHAR NOT NULL,
    blocked_at VARCHAR NOT NULL,
    cleared_at VARCHAR,
    status VARCHAR NOT NULL,
    revision BIGINT NOT NULL
);

CREATE TABLE refill_epochs (
    epoch_id VARCHAR PRIMARY KEY,
    board_namespace VARCHAR NOT NULL,
    started_at VARCHAR NOT NULL,
    ended_at VARCHAR,
    generation INTEGER NOT NULL,
    status VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}'
);

-- ---------------------------------------------------------------------------
-- runtime: leases, daemons, attempts, validation, completion
-- Lease semantics preserve task_cid identity, fencing_token, fence_epoch,
-- attempt, expiry, and accepted/released/expired state vocabulary.
-- ---------------------------------------------------------------------------

CREATE TABLE supervisor_instances (
    supervisor_id VARCHAR PRIMARY KEY,
    birth_id VARCHAR NOT NULL,
    repository_id VARCHAR NOT NULL,
    started_at VARCHAR NOT NULL,
    stopped_at VARCHAR,
    status VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}'
);

CREATE TABLE daemon_instances (
    daemon_id VARCHAR PRIMARY KEY,
    supervisor_id VARCHAR NOT NULL,
    daemon_kind VARCHAR NOT NULL,
    birth_id VARCHAR NOT NULL,
    started_at VARCHAR NOT NULL,
    stopped_at VARCHAR,
    status VARCHAR NOT NULL,
    revision BIGINT NOT NULL
);

CREATE TABLE daemon_sessions (
    session_id VARCHAR PRIMARY KEY,
    daemon_id VARCHAR NOT NULL,
    birth_id VARCHAR NOT NULL,
    opened_at VARCHAR NOT NULL,
    expires_at VARCHAR NOT NULL,
    fence_epoch BIGINT NOT NULL,
    revision BIGINT NOT NULL,
    status VARCHAR NOT NULL
);

CREATE TABLE heartbeats (
    heartbeat_id VARCHAR PRIMARY KEY,
    session_id VARCHAR NOT NULL,
    task_cid VARCHAR,
    observed_at VARCHAR NOT NULL,
    expires_at VARCHAR NOT NULL,
    fencing_token BIGINT NOT NULL DEFAULT 0,
    capacity_millionths BIGINT NOT NULL DEFAULT 0,
    payload_json VARCHAR NOT NULL DEFAULT '{}'
);

CREATE TABLE stall_detections (
    detection_id VARCHAR PRIMARY KEY,
    session_id VARCHAR,
    task_cid VARCHAR,
    detected_at VARCHAR NOT NULL,
    reason VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}'
);

CREATE TABLE restart_decisions (
    decision_id VARCHAR PRIMARY KEY,
    daemon_id VARCHAR NOT NULL,
    decided_at VARCHAR NOT NULL,
    action VARCHAR NOT NULL,
    reason VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}'
);

CREATE TABLE leases (
    task_cid VARCHAR PRIMARY KEY,
    claim_cid VARCHAR NOT NULL,
    resolution_cid VARCHAR NOT NULL DEFAULT '',
    owner_session_id VARCHAR NOT NULL,
    claimant_did VARCHAR NOT NULL DEFAULT '',
    fence_epoch BIGINT NOT NULL,
    fencing_token BIGINT NOT NULL,
    attempt BIGINT NOT NULL,
    state VARCHAR NOT NULL,
    acquired_at VARCHAR NOT NULL,
    expires_at VARCHAR NOT NULL,
    started_at VARCHAR NOT NULL,
    release_reason VARCHAR NOT NULL DEFAULT '',
    retry_not_before VARCHAR,
    revision BIGINT NOT NULL
);

CREATE TABLE lease_events (
    event_id VARCHAR PRIMARY KEY,
    task_cid VARCHAR NOT NULL,
    event_type VARCHAR NOT NULL,
    fencing_token BIGINT NOT NULL,
    fence_epoch BIGINT NOT NULL,
    observed_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}'
);

CREATE TABLE task_attempts (
    attempt_id VARCHAR PRIMARY KEY,
    task_cid VARCHAR NOT NULL,
    attempt BIGINT NOT NULL,
    session_id VARCHAR NOT NULL,
    started_at VARCHAR NOT NULL,
    finished_at VARCHAR,
    status VARCHAR NOT NULL,
    fence_epoch BIGINT NOT NULL DEFAULT 0,
    revision BIGINT NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}',
    UNIQUE (task_cid, attempt)
);

CREATE TABLE attempt_phases (
    attempt_id VARCHAR NOT NULL,
    phase VARCHAR NOT NULL,
    started_at VARCHAR NOT NULL,
    finished_at VARCHAR,
    status VARCHAR NOT NULL,
    PRIMARY KEY (attempt_id, phase)
);

CREATE TABLE task_claims (
    claim_id VARCHAR PRIMARY KEY,
    task_cid VARCHAR NOT NULL,
    session_id VARCHAR NOT NULL,
    fencing_token BIGINT NOT NULL,
    fence_epoch BIGINT NOT NULL,
    claimed_at VARCHAR NOT NULL,
    expires_at VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    revision BIGINT NOT NULL
);

CREATE TABLE provider_invocations (
    invocation_id VARCHAR PRIMARY KEY,
    attempt_id VARCHAR NOT NULL,
    provider_id VARCHAR NOT NULL,
    model_id VARCHAR NOT NULL,
    started_at VARCHAR NOT NULL,
    finished_at VARCHAR,
    status VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}'
);

CREATE TABLE validation_runs (
    run_id VARCHAR PRIMARY KEY,
    task_cid VARCHAR NOT NULL,
    attempt_id VARCHAR,
    started_at VARCHAR NOT NULL,
    finished_at VARCHAR,
    status VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}'
);

CREATE TABLE validation_results (
    result_id VARCHAR PRIMARY KEY,
    run_id VARCHAR NOT NULL,
    ordinal BIGINT NOT NULL,
    command_argv_json VARCHAR NOT NULL,
    exit_code INTEGER NOT NULL,
    outcome VARCHAR NOT NULL,
    evidence_digest VARCHAR NOT NULL DEFAULT '',
    body_json VARCHAR NOT NULL DEFAULT '{}'
);

CREATE TABLE merge_attempts (
    merge_attempt_id VARCHAR PRIMARY KEY,
    task_cid VARCHAR NOT NULL,
    entry_id VARCHAR,
    started_at VARCHAR NOT NULL,
    finished_at VARCHAR,
    status VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}'
);

CREATE TABLE recovery_actions (
    action_id VARCHAR PRIMARY KEY,
    task_cid VARCHAR,
    session_id VARCHAR,
    action_kind VARCHAR NOT NULL,
    decided_at VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}'
);

CREATE TABLE idempotency_records (
    idempotency_key VARCHAR PRIMARY KEY,
    command_kind VARCHAR NOT NULL,
    request_digest VARCHAR NOT NULL,
    response_digest VARCHAR NOT NULL,
    created_at VARCHAR NOT NULL,
    expires_at VARCHAR,
    status VARCHAR NOT NULL
);

CREATE TABLE effect_claims (
    effect_id VARCHAR PRIMARY KEY,
    task_cid VARCHAR NOT NULL,
    attempt_id VARCHAR,
    effect_kind VARCHAR NOT NULL,
    claimed_at VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}'
);

CREATE TABLE completion_receipts (
    receipt_id VARCHAR PRIMARY KEY,
    task_cid VARCHAR NOT NULL,
    attempt_id VARCHAR,
    completed_at VARCHAR NOT NULL,
    outcome VARCHAR NOT NULL,
    evidence_digest VARCHAR NOT NULL,
    validation_run_id VARCHAR,
    revision BIGINT NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}'
);

CREATE TABLE health_samples (
    sample_id VARCHAR PRIMARY KEY,
    subject_kind VARCHAR NOT NULL,
    subject_id VARCHAR NOT NULL,
    observed_at VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    metrics_json VARCHAR NOT NULL DEFAULT '{}'
);

-- ---------------------------------------------------------------------------
-- events / control-plane audit stream
-- ---------------------------------------------------------------------------

CREATE TABLE domain_events (
    event_id VARCHAR PRIMARY KEY,
    stream_id VARCHAR NOT NULL,
    sequence BIGINT NOT NULL,
    event_type VARCHAR NOT NULL,
    task_cid VARCHAR,
    session_id VARCHAR,
    occurred_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL,
    UNIQUE (stream_id, sequence)
);

CREATE TABLE structured_logs (
    log_id VARCHAR PRIMARY KEY,
    severity VARCHAR NOT NULL,
    component VARCHAR NOT NULL,
    trace_id VARCHAR NOT NULL DEFAULT '',
    span_id VARCHAR NOT NULL DEFAULT '',
    task_cid VARCHAR,
    attempt_id VARCHAR,
    session_id VARCHAR,
    logged_at VARCHAR NOT NULL,
    message VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}'
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
    observed_at VARCHAR NOT NULL,
    value_integer BIGINT NOT NULL,
    labels_json VARCHAR NOT NULL DEFAULT '{}'
);

CREATE TABLE budget_reservations (
    reservation_id VARCHAR PRIMARY KEY,
    budget_kind VARCHAR NOT NULL,
    owner_session_id VARCHAR NOT NULL,
    amount BIGINT NOT NULL,
    reserved_at VARCHAR NOT NULL,
    expires_at VARCHAR,
    status VARCHAR NOT NULL
);

CREATE TABLE budget_consumption (
    consumption_id VARCHAR PRIMARY KEY,
    reservation_id VARCHAR NOT NULL,
    amount BIGINT NOT NULL,
    consumed_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}'
);

CREATE TABLE quack_query_telemetry (
    sample_id VARCHAR PRIMARY KEY,
    session_id VARCHAR,
    observed_at VARCHAR NOT NULL,
    query_kind VARCHAR NOT NULL,
    duration_ms BIGINT NOT NULL,
    outcome VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}'
);

-- ---------------------------------------------------------------------------
-- code: snapshots, symbols, mutations, impact
-- ---------------------------------------------------------------------------

CREATE TABLE source_snapshots (
    snapshot_id VARCHAR PRIMARY KEY,
    repository_id VARCHAR NOT NULL,
    tree_id VARCHAR NOT NULL,
    overlay_digest VARCHAR NOT NULL DEFAULT '',
    observed_at VARCHAR NOT NULL,
    scanner_version VARCHAR NOT NULL
);

CREATE TABLE source_files (
    file_id VARCHAR PRIMARY KEY,
    snapshot_id VARCHAR NOT NULL,
    path VARCHAR NOT NULL,
    language VARCHAR NOT NULL,
    content_digest VARCHAR NOT NULL,
    byte_length BIGINT NOT NULL,
    UNIQUE (snapshot_id, path)
);

CREATE TABLE file_versions (
    file_version_id VARCHAR PRIMARY KEY,
    repository_id VARCHAR NOT NULL,
    path VARCHAR NOT NULL,
    content_digest VARCHAR NOT NULL,
    observed_at VARCHAR NOT NULL
);

CREATE TABLE parse_runs (
    parse_run_id VARCHAR PRIMARY KEY,
    snapshot_id VARCHAR NOT NULL,
    parser_id VARCHAR NOT NULL,
    started_at VARCHAR NOT NULL,
    finished_at VARCHAR,
    status VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}'
);

CREATE TABLE symbols (
    symbol_id VARCHAR PRIMARY KEY,
    snapshot_id VARCHAR NOT NULL,
    file_id VARCHAR NOT NULL,
    language VARCHAR NOT NULL,
    qualified_name VARCHAR NOT NULL,
    kind VARCHAR NOT NULL,
    node_path VARCHAR NOT NULL,
    fingerprint VARCHAR NOT NULL
);

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
    parent_node_id VARCHAR,
    node_kind VARCHAR NOT NULL,
    node_path VARCHAR NOT NULL,
    fingerprint VARCHAR NOT NULL
);

CREATE TABLE ast_edges (
    edge_id VARCHAR PRIMARY KEY,
    snapshot_id VARCHAR NOT NULL,
    source_node_id VARCHAR NOT NULL,
    target_node_id VARCHAR NOT NULL,
    edge_kind VARCHAR NOT NULL
);

CREATE TABLE imports (
    import_id VARCHAR PRIMARY KEY,
    snapshot_id VARCHAR NOT NULL,
    file_id VARCHAR NOT NULL,
    imported_module VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}'
);

CREATE TABLE calls (
    call_id VARCHAR PRIMARY KEY,
    snapshot_id VARCHAR NOT NULL,
    caller_symbol_id VARCHAR NOT NULL,
    callee_symbol_id VARCHAR NOT NULL
);

CREATE TABLE references_graph (
    reference_id VARCHAR PRIMARY KEY,
    snapshot_id VARCHAR NOT NULL,
    from_symbol_id VARCHAR NOT NULL,
    to_symbol_id VARCHAR NOT NULL,
    reference_kind VARCHAR NOT NULL
);

CREATE TABLE definitions (
    definition_id VARCHAR PRIMARY KEY,
    snapshot_id VARCHAR NOT NULL,
    symbol_id VARCHAR NOT NULL,
    file_id VARCHAR NOT NULL,
    node_path VARCHAR NOT NULL
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
    attempt_id VARCHAR,
    before_snapshot_id VARCHAR NOT NULL,
    after_snapshot_id VARCHAR NOT NULL,
    created_at VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}'
);

CREATE TABLE mutation_files (
    mutation_id VARCHAR NOT NULL,
    path VARCHAR NOT NULL,
    before_digest VARCHAR NOT NULL,
    after_digest VARCHAR NOT NULL,
    PRIMARY KEY (mutation_id, path)
);

CREATE TABLE mutation_hunks (
    hunk_id VARCHAR PRIMARY KEY,
    mutation_id VARCHAR NOT NULL,
    path VARCHAR NOT NULL,
    ordinal BIGINT NOT NULL,
    hunk_text VARCHAR NOT NULL
);

CREATE TABLE ast_mutations (
    ast_mutation_id VARCHAR PRIMARY KEY,
    mutation_id VARCHAR NOT NULL,
    edit_script_json VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}'
);

CREATE TABLE impact_edges (
    edge_id VARCHAR PRIMARY KEY,
    mutation_id VARCHAR NOT NULL,
    from_symbol_id VARCHAR NOT NULL,
    to_symbol_id VARCHAR NOT NULL,
    edge_kind VARCHAR NOT NULL
);

CREATE TABLE impact_closures (
    closure_id VARCHAR PRIMARY KEY,
    mutation_id VARCHAR NOT NULL,
    symbol_id VARCHAR NOT NULL,
    depth INTEGER NOT NULL,
    UNIQUE (mutation_id, symbol_id)
);

CREATE TABLE repair_candidates (
    candidate_id VARCHAR PRIMARY KEY,
    task_cid VARCHAR NOT NULL,
    rank INTEGER NOT NULL,
    body_json VARCHAR NOT NULL,
    created_at VARCHAR NOT NULL
);

CREATE TABLE repair_applications (
    application_id VARCHAR PRIMARY KEY,
    candidate_id VARCHAR NOT NULL,
    mutation_id VARCHAR,
    applied_at VARCHAR NOT NULL,
    outcome VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}'
);

-- ---------------------------------------------------------------------------
-- evidence: proof obligations and counterexamples
-- ---------------------------------------------------------------------------

CREATE TABLE proof_obligations (
    obligation_id VARCHAR PRIMARY KEY,
    task_cid VARCHAR,
    symbol_id VARCHAR,
    obligation_kind VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    created_at VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}'
);

CREATE TABLE proof_attempts (
    proof_attempt_id VARCHAR PRIMARY KEY,
    obligation_id VARCHAR NOT NULL,
    started_at VARCHAR NOT NULL,
    finished_at VARCHAR,
    status VARCHAR NOT NULL,
    prover_id VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}'
);

CREATE TABLE counterexamples (
    counterexample_id VARCHAR PRIMARY KEY,
    obligation_id VARCHAR NOT NULL,
    proof_attempt_id VARCHAR,
    created_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);

CREATE TABLE evidence_nodes (
    evidence_id VARCHAR PRIMARY KEY,
    evidence_cid VARCHAR NOT NULL UNIQUE,
    evidence_kind VARCHAR NOT NULL,
    subject_kind VARCHAR NOT NULL,
    subject_id VARCHAR NOT NULL,
    digest VARCHAR NOT NULL,
    created_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}'
);

-- ---------------------------------------------------------------------------
-- cache / context: manifests, prompts, decision cache
-- ---------------------------------------------------------------------------

CREATE TABLE context_manifests (
    manifest_id VARCHAR PRIMARY KEY,
    manifest_cid VARCHAR NOT NULL UNIQUE,
    repository_id VARCHAR NOT NULL,
    schema_revision INTEGER NOT NULL,
    created_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}'
);

CREATE TABLE context_members (
    manifest_id VARCHAR NOT NULL,
    member_key VARCHAR NOT NULL,
    member_digest VARCHAR NOT NULL,
    member_kind VARCHAR NOT NULL,
    byte_length BIGINT NOT NULL,
    PRIMARY KEY (manifest_id, member_key)
);

CREATE TABLE context_deltas (
    delta_id VARCHAR PRIMARY KEY,
    from_manifest_id VARCHAR NOT NULL,
    to_manifest_id VARCHAR NOT NULL,
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
    task_cid VARCHAR,
    manifest_id VARCHAR,
    created_at VARCHAR NOT NULL,
    prompt_digest VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}'
);

CREATE TABLE prompt_inputs (
    instance_id VARCHAR NOT NULL,
    ordinal BIGINT NOT NULL,
    input_kind VARCHAR NOT NULL,
    input_digest VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}',
    PRIMARY KEY (instance_id, ordinal)
);

CREATE TABLE decision_cache_entries (
    cache_key VARCHAR PRIMARY KEY,
    manifest_cid VARCHAR NOT NULL,
    decision_digest VARCHAR NOT NULL,
    created_at VARCHAR NOT NULL,
    expires_at VARCHAR,
    body_json VARCHAR NOT NULL DEFAULT '{}'
);

CREATE TABLE replay_suppressions (
    suppression_id VARCHAR PRIMARY KEY,
    failure_signature VARCHAR NOT NULL,
    task_cid VARCHAR,
    created_at VARCHAR NOT NULL,
    expires_at VARCHAR,
    reason VARCHAR NOT NULL
);

-- ---------------------------------------------------------------------------
-- improve: provider economy and churn metrics
-- ---------------------------------------------------------------------------

CREATE TABLE provider_calls (
    call_id VARCHAR PRIMARY KEY,
    provider_id VARCHAR NOT NULL,
    model_id VARCHAR NOT NULL,
    task_cid VARCHAR,
    attempt_id VARCHAR,
    started_at VARCHAR NOT NULL,
    finished_at VARCHAR,
    input_tokens BIGINT NOT NULL DEFAULT 0,
    output_tokens BIGINT NOT NULL DEFAULT 0,
    status VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}'
);

CREATE TABLE provider_responses (
    response_id VARCHAR PRIMARY KEY,
    call_id VARCHAR NOT NULL,
    response_digest VARCHAR NOT NULL,
    created_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}'
);

CREATE TABLE failure_signatures (
    signature_id VARCHAR PRIMARY KEY,
    signature_hash VARCHAR NOT NULL UNIQUE,
    kind VARCHAR NOT NULL,
    normalized_text VARCHAR NOT NULL,
    first_seen_at VARCHAR NOT NULL,
    last_seen_at VARCHAR NOT NULL,
    occurrence_count BIGINT NOT NULL DEFAULT 1
);

CREATE TABLE churn_metrics (
    metric_sample_id VARCHAR PRIMARY KEY,
    observed_at VARCHAR NOT NULL,
    task_cid VARCHAR,
    provider_calls BIGINT NOT NULL DEFAULT 0,
    input_tokens BIGINT NOT NULL DEFAULT 0,
    output_tokens BIGINT NOT NULL DEFAULT 0,
    context_bytes BIGINT NOT NULL DEFAULT 0,
    cache_hits BIGINT NOT NULL DEFAULT 0,
    duplicate_context_fraction_millionths BIGINT NOT NULL DEFAULT 0,
    body_json VARCHAR NOT NULL DEFAULT '{}'
);

-- ---------------------------------------------------------------------------
-- Domain registration seed (schema metadata)
-- ---------------------------------------------------------------------------

INSERT INTO schema_contracts (
    contract_id, domain, interface_name, schema_name, schema_version,
    table_name, description, created_at
) VALUES
    ('contract:meta:schema_contracts', 'meta', 'ControlPlaneSchema@1',
     'ipfs_accelerate_py/agent-supervisor/control-plane-schema@1', 1,
     'schema_contracts', 'Registered schema contracts per domain table',
     '1970-01-01T00:00:00Z'),
    ('contract:intent:tasks', 'intent', 'ControlPlaneSchema@1',
     'ipfs_accelerate_py/agent-supervisor/control-plane-schema@1', 1,
     'tasks', 'Canonical task rows keyed by task_cid',
     '1970-01-01T00:00:00Z'),
    ('contract:runtime:leases', 'runtime', 'ControlPlaneSchema@1',
     'ipfs_accelerate_py/agent-supervisor/control-plane-schema@1', 1,
     'leases', 'Fenced task leases preserving task_cid and fencing semantics',
     '1970-01-01T00:00:00Z'),
    ('contract:git:repositories', 'git', 'ControlPlaneSchema@1',
     'ipfs_accelerate_py/agent-supervisor/control-plane-schema@1', 1,
     'repositories', 'Repository identities bound to Git object IDs',
     '1970-01-01T00:00:00Z'),
    ('contract:code:mutations', 'code', 'ControlPlaneSchema@1',
     'ipfs_accelerate_py/agent-supervisor/control-plane-schema@1', 1,
     'mutations', 'Code mutations bound to before/after snapshots',
     '1970-01-01T00:00:00Z'),
    ('contract:evidence:evidence_nodes', 'evidence', 'ControlPlaneSchema@1',
     'ipfs_accelerate_py/agent-supervisor/control-plane-schema@1', 1,
     'evidence_nodes', 'Content-addressed evidence nodes',
     '1970-01-01T00:00:00Z'),
    ('contract:cache:decision_cache_entries', 'cache', 'ControlPlaneSchema@1',
     'ipfs_accelerate_py/agent-supervisor/control-plane-schema@1', 1,
     'decision_cache_entries', 'Deterministic decision cache keyed by manifest',
     '1970-01-01T00:00:00Z'),
    ('contract:control:authorization_grants', 'control', 'ControlPlaneSchema@1',
     'ipfs_accelerate_py/agent-supervisor/control-plane-schema@1', 1,
     'authorization_grants', 'Authorization grants for state-owner clients',
     '1970-01-01T00:00:00Z'),
    ('contract:schedule:task_assignments', 'schedule', 'ControlPlaneSchema@1',
     'ipfs_accelerate_py/agent-supervisor/control-plane-schema@1', 1,
     'task_assignments', 'Lane and session task assignments',
     '1970-01-01T00:00:00Z'),
    ('contract:improve:churn_metrics', 'improve', 'ControlPlaneSchema@1',
     'ipfs_accelerate_py/agent-supervisor/control-plane-schema@1', 1,
     'churn_metrics', 'Provider and context churn metrics',
     '1970-01-01T00:00:00Z');

-- ---------------------------------------------------------------------------
-- Indexes for join-critical identity and lease/task lookup
-- ---------------------------------------------------------------------------

CREATE INDEX idx_tasks_status ON tasks (status);
CREATE INDEX idx_tasks_goal ON tasks (goal_cid);
CREATE INDEX idx_task_dependencies_dep ON task_dependencies (dependency_task_cid);
CREATE INDEX idx_leases_state_expiry ON leases (state, expires_at, retry_not_before);
CREATE INDEX idx_leases_session ON leases (owner_session_id);
CREATE INDEX idx_lease_events_task ON lease_events (task_cid, observed_at);
CREATE INDEX idx_task_claims_task ON task_claims (task_cid, status);
CREATE INDEX idx_domain_events_stream ON domain_events (stream_id, sequence);
CREATE INDEX idx_heartbeats_session ON heartbeats (session_id, observed_at);
CREATE INDEX idx_worktrees_repo ON worktrees (repository_id, status);
CREATE INDEX idx_mutations_task ON mutations (task_cid);
CREATE INDEX idx_evidence_subject ON evidence_nodes (subject_kind, subject_id);
CREATE INDEX idx_completion_task ON completion_receipts (task_cid);

-- ---------------------------------------------------------------------------
-- Constrained diagnostic / context views (read-only projections)
-- ---------------------------------------------------------------------------

CREATE VIEW ready_task_context_v1 AS
SELECT
    t.task_cid,
    t.task_alias,
    t.goal_cid,
    t.status,
    t.priority,
    t.revision,
    t.fence_epoch,
    t.updated_at,
    g.title AS goal_title,
    g.status AS goal_status,
    l.state AS lease_state,
    l.owner_session_id AS lease_owner_session_id,
    l.fencing_token AS lease_fencing_token,
    l.expires_at AS lease_expires_at
FROM tasks AS t
LEFT JOIN goals AS g ON g.goal_cid = t.goal_cid
LEFT JOIN leases AS l ON l.task_cid = t.task_cid
WHERE t.status IN ('proposed', 'admitted', 'pending', 'ready', 'retrying', 'todo')
  AND (l.task_cid IS NULL OR l.state NOT IN ('accepted'));

CREATE VIEW live_sessions_v1 AS
SELECT
    s.session_id,
    s.daemon_id,
    s.birth_id,
    s.opened_at,
    s.expires_at,
    s.fence_epoch,
    s.revision,
    s.status
FROM daemon_sessions AS s
WHERE s.status IN ('open', 'active', 'running');

CREATE VIEW expiring_leases_v1 AS
SELECT
    l.task_cid,
    l.claim_cid,
    l.owner_session_id,
    l.fencing_token,
    l.fence_epoch,
    l.attempt,
    l.state,
    l.expires_at,
    l.revision
FROM leases AS l
WHERE l.state = 'accepted';

CREATE VIEW stuck_phases_v1 AS
SELECT
    a.attempt_id,
    a.task_cid,
    a.attempt,
    a.session_id,
    a.status AS attempt_status,
    p.phase,
    p.started_at AS phase_started_at,
    p.status AS phase_status
FROM task_attempts AS a
JOIN attempt_phases AS p ON p.attempt_id = a.attempt_id
WHERE a.status IN ('running', 'in_progress', 'claimed')
  AND p.finished_at IS NULL;

CREATE VIEW schema_domain_inventory_v1 AS
SELECT
    domain,
    table_name,
    interface_name,
    schema_name,
    schema_version,
    description
FROM schema_contracts
ORDER BY domain, table_name;
