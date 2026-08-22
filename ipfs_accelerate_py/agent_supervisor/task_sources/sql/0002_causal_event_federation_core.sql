-- Causal event federation operational control-plane extension.
--
-- This migration is additive.  It preserves ControlPlaneSchema@1 and extends
-- the same authoritative DuckDB store with normalized federation coordination
-- records.  Semantic, proof, test, and retrieval bodies remain owned by their
-- published authorities; this database stores only typed operational metadata
-- and content-addressed references to those bodies.

-- ---------------------------------------------------------------------------
-- Existing authoritative surfaces: additive federation join columns
-- ---------------------------------------------------------------------------
-- DuckDB 1.5 does not admit NOT NULL constraints in ADD COLUMN.  Defaults keep
-- legacy rows queryable; federation writers validate non-empty values through
-- closed contracts and the state-owner command boundary.
ALTER TABLE supervisor_instances ADD COLUMN tenant_id VARCHAR DEFAULT '';
ALTER TABLE supervisor_instances ADD COLUMN federation_id VARCHAR DEFAULT '';
ALTER TABLE supervisor_instances ADD COLUMN parent_supervisor_id VARCHAR DEFAULT '';
ALTER TABLE supervisor_instances ADD COLUMN supervisor_definition_id VARCHAR DEFAULT '';
ALTER TABLE supervisor_instances ADD COLUMN role VARCHAR DEFAULT '';
ALTER TABLE supervisor_instances ADD COLUMN lifecycle_state VARCHAR DEFAULT '';
ALTER TABLE supervisor_instances ADD COLUMN assignment_revision BIGINT DEFAULT 0;
ALTER TABLE supervisor_instances ADD COLUMN lease_id VARCHAR DEFAULT '';
ALTER TABLE supervisor_instances ADD COLUMN fencing_epoch BIGINT DEFAULT 0;
ALTER TABLE supervisor_instances ADD COLUMN policy_id VARCHAR DEFAULT '';
ALTER TABLE supervisor_instances ADD COLUMN policy_revision BIGINT DEFAULT 0;
ALTER TABLE supervisor_instances ADD COLUMN admission_decision_id VARCHAR DEFAULT '';
CREATE INDEX supervisor_instances_federation_idx
    ON supervisor_instances(tenant_id, federation_id, status);

ALTER TABLE idempotency_records ADD COLUMN tenant_id VARCHAR DEFAULT '';
ALTER TABLE idempotency_records ADD COLUMN federation_id VARCHAR DEFAULT '';
ALTER TABLE idempotency_records ADD COLUMN supervisor_id VARCHAR DEFAULT '';
ALTER TABLE idempotency_records ADD COLUMN request_digest VARCHAR DEFAULT '';
CREATE INDEX idempotency_records_federation_idx
    ON idempotency_records(tenant_id, federation_id, created_at);

ALTER TABLE budget_reservations ADD COLUMN tenant_id VARCHAR DEFAULT '';
ALTER TABLE budget_reservations ADD COLUMN federation_id VARCHAR DEFAULT '';
ALTER TABLE budget_reservations ADD COLUMN supervisor_id VARCHAR DEFAULT '';
ALTER TABLE budget_reservations ADD COLUMN subagent_id VARCHAR DEFAULT '';
ALTER TABLE budget_reservations ADD COLUMN parent_reservation_id VARCHAR DEFAULT '';
ALTER TABLE budget_reservations ADD COLUMN budget_id VARCHAR DEFAULT '';
ALTER TABLE budget_reservations ADD COLUMN budget_dimension VARCHAR DEFAULT '';
ALTER TABLE budget_reservations ADD COLUMN revision BIGINT DEFAULT 0;
ALTER TABLE budget_reservations ADD COLUMN fencing_epoch BIGINT DEFAULT 0;
CREATE INDEX budget_reservations_federation_idx
    ON budget_reservations(tenant_id, federation_id, state);

-- The v1 event identity and sequences remain canonical.  Required federation
-- identities and compact content references become first-class columns.
ALTER TABLE domain_events ADD COLUMN event_cid VARCHAR DEFAULT '';
ALTER TABLE domain_events ADD COLUMN tenant_id VARCHAR DEFAULT '';
ALTER TABLE domain_events ADD COLUMN federation_id VARCHAR DEFAULT '';
ALTER TABLE domain_events ADD COLUMN supervisor_id VARCHAR DEFAULT '';
ALTER TABLE domain_events ADD COLUMN repository_id VARCHAR DEFAULT '';
ALTER TABLE domain_events ADD COLUMN tree_id VARCHAR DEFAULT '';
ALTER TABLE domain_events ADD COLUMN goal_id VARCHAR DEFAULT '';
ALTER TABLE domain_events ADD COLUMN subgoal_id VARCHAR DEFAULT '';
ALTER TABLE domain_events ADD COLUMN symbol_id VARCHAR DEFAULT '';
ALTER TABLE domain_events ADD COLUMN contract_id VARCHAR DEFAULT '';
ALTER TABLE domain_events ADD COLUMN proof_obligation_id VARCHAR DEFAULT '';
ALTER TABLE domain_events ADD COLUMN resource_class VARCHAR DEFAULT '';
ALTER TABLE domain_events ADD COLUMN causal_parent_ids_json VARCHAR DEFAULT '[]';
ALTER TABLE domain_events ADD COLUMN correlation_id VARCHAR DEFAULT '';
ALTER TABLE domain_events ADD COLUMN causation_id VARCHAR DEFAULT '';
ALTER TABLE domain_events ADD COLUMN payload_ref VARCHAR DEFAULT '';
ALTER TABLE domain_events ADD COLUMN changed_fact_refs_json VARCHAR DEFAULT '[]';
ALTER TABLE domain_events ADD COLUMN effect_class VARCHAR DEFAULT '';
ALTER TABLE domain_events ADD COLUMN deduplication_key VARCHAR DEFAULT '';
ALTER TABLE domain_events ADD COLUMN control_plane_generation BIGINT DEFAULT 0;
ALTER TABLE domain_events ADD COLUMN causal_graph_revision BIGINT DEFAULT 0;
ALTER TABLE domain_events ADD COLUMN expires_at VARCHAR;
CREATE INDEX domain_events_federation_sequence_idx
    ON domain_events(tenant_id, federation_id, global_sequence);
CREATE INDEX domain_events_correlation_idx
    ON domain_events(correlation_id, causation_id);
CREATE INDEX domain_events_deduplication_idx
    ON domain_events(tenant_id, federation_id, deduplication_key);

CREATE TABLE domain_event_causal_parents (
    event_id VARCHAR NOT NULL,
    parent_event_id VARCHAR NOT NULL,
    ordinal BIGINT NOT NULL,
    PRIMARY KEY (event_id, parent_event_id)
);
CREATE INDEX domain_event_causal_parents_parent_idx
    ON domain_event_causal_parents(parent_event_id, event_id);

CREATE TABLE domain_event_changed_facts (
    event_id VARCHAR NOT NULL,
    fact_ref VARCHAR NOT NULL,
    ordinal BIGINT NOT NULL,
    PRIMARY KEY (event_id, fact_ref)
);
CREATE INDEX domain_event_changed_facts_fact_idx
    ON domain_event_changed_facts(fact_ref, event_id);

-- ---------------------------------------------------------------------------
-- Federation identity, revision, policy, plan, and receipt authority
-- ---------------------------------------------------------------------------
CREATE TABLE federations (
    federation_id VARCHAR PRIMARY KEY,
    tenant_id VARCHAR NOT NULL,
    program_id VARCHAR NOT NULL,
    objective_ref VARCHAR NOT NULL,
    objective_revision BIGINT NOT NULL,
    policy_id VARCHAR NOT NULL,
    policy_revision BIGINT NOT NULL,
    operation_catalog_id VARCHAR NOT NULL,
    control_plane_generation BIGINT NOT NULL,
    causal_graph_revision BIGINT NOT NULL,
    semantic_state_root VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    maximum_supervisors BIGINT NOT NULL,
    maximum_subagents BIGINT NOT NULL,
    revision BIGINT NOT NULL,
    fencing_epoch BIGINT NOT NULL,
    issuer_id VARCHAR NOT NULL,
    authorization_evidence_ref VARCHAR NOT NULL,
    expires_at VARCHAR NOT NULL,
    created_at VARCHAR NOT NULL,
    updated_at VARCHAR NOT NULL,
    content_ref VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}'
);
CREATE INDEX federations_tenant_state_idx
    ON federations(tenant_id, status, program_id);

CREATE TABLE federation_revisions (
    federation_id VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    repository_population_ref VARCHAR NOT NULL,
    tree_population_ref VARCHAR NOT NULL,
    supervisor_population_ref VARCHAR NOT NULL,
    budget_hierarchy_ref VARCHAR NOT NULL,
    semantic_state_root VARCHAR NOT NULL,
    causal_graph_revision BIGINT NOT NULL,
    policy_revision BIGINT NOT NULL,
    recorded_at VARCHAR NOT NULL,
    recorded_by VARCHAR NOT NULL,
    content_ref VARCHAR NOT NULL,
    PRIMARY KEY (federation_id, revision)
);

CREATE TABLE federation_policies (
    policy_id VARCHAR PRIMARY KEY,
    tenant_id VARCHAR NOT NULL,
    federation_id VARCHAR NOT NULL DEFAULT '',
    revision BIGINT NOT NULL,
    issuer_id VARCHAR NOT NULL,
    authorization_evidence_ref VARCHAR NOT NULL DEFAULT '',
    effect_ceiling VARCHAR NOT NULL DEFAULT '',
    risk_ceiling VARCHAR NOT NULL DEFAULT '',
    maximum_supervisors BIGINT NOT NULL,
    maximum_subagents BIGINT NOT NULL,
    maximum_concurrent_subagents BIGINT NOT NULL,
    expires_at VARCHAR NOT NULL,
    content_ref VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    created_at VARCHAR NOT NULL,
    updated_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}'
);
CREATE INDEX federation_policies_federation_idx
    ON federation_policies(tenant_id, federation_id, status);

CREATE TABLE federation_plans (
    federation_plan_id VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    tenant_id VARCHAR NOT NULL,
    federation_id VARCHAR NOT NULL,
    objective_ref VARCHAR NOT NULL,
    policy_id VARCHAR NOT NULL,
    policy_revision BIGINT NOT NULL,
    causal_graph_revision BIGINT NOT NULL,
    status VARCHAR NOT NULL,
    content_ref VARCHAR NOT NULL,
    created_at VARCHAR NOT NULL,
    PRIMARY KEY (federation_plan_id, revision)
);
CREATE INDEX federation_plans_federation_idx
    ON federation_plans(tenant_id, federation_id, status);

CREATE TABLE federation_receipts (
    federation_receipt_id VARCHAR PRIMARY KEY,
    tenant_id VARCHAR NOT NULL,
    federation_id VARCHAR NOT NULL,
    receipt_kind VARCHAR NOT NULL,
    federation_revision BIGINT NOT NULL,
    control_plane_generation BIGINT NOT NULL,
    event_watermark BIGINT NOT NULL,
    issuer_id VARCHAR NOT NULL,
    content_ref VARCHAR NOT NULL,
    recorded_at VARCHAR NOT NULL
);
CREATE INDEX federation_receipts_federation_idx
    ON federation_receipts(tenant_id, federation_id, recorded_at);

-- ---------------------------------------------------------------------------
-- Supervisor and bounded logical-subagent companion records
-- ---------------------------------------------------------------------------
CREATE TABLE supervisor_definitions (
    supervisor_definition_id VARCHAR PRIMARY KEY,
    tenant_id VARCHAR NOT NULL,
    federation_id VARCHAR NOT NULL,
    specialization VARCHAR NOT NULL,
    capability_set_ref VARCHAR NOT NULL,
    allowed_operations_ref VARCHAR NOT NULL,
    effect_ceiling VARCHAR NOT NULL,
    risk_ceiling VARCHAR NOT NULL,
    resource_ceiling_ref VARCHAR NOT NULL,
    token_ceiling_ref VARCHAR NOT NULL,
    proof_requirements_ref VARCHAR NOT NULL,
    merge_policy_ref VARCHAR NOT NULL,
    policy_id VARCHAR NOT NULL,
    policy_revision BIGINT NOT NULL,
    authorization_evidence_ref VARCHAR NOT NULL,
    content_ref VARCHAR NOT NULL,
    created_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}'
);

CREATE TABLE supervisor_assignments (
    assignment_id VARCHAR PRIMARY KEY,
    tenant_id VARCHAR NOT NULL,
    federation_id VARCHAR NOT NULL,
    supervisor_id VARCHAR NOT NULL,
    parent_supervisor_id VARCHAR NOT NULL DEFAULT '',
    assignment_revision BIGINT NOT NULL,
    repository_id VARCHAR NOT NULL,
    tree_id VARCHAR NOT NULL,
    goal_ref VARCHAR NOT NULL DEFAULT '',
    subgoal_ref VARCHAR NOT NULL DEFAULT '',
    task_family VARCHAR NOT NULL DEFAULT '',
    shard_id VARCHAR NOT NULL DEFAULT '',
    lease_id VARCHAR NOT NULL DEFAULT '',
    fencing_epoch BIGINT NOT NULL,
    status VARCHAR NOT NULL,
    admission_decision_id VARCHAR NOT NULL,
    content_ref VARCHAR NOT NULL,
    created_at VARCHAR NOT NULL,
    updated_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}'
);
CREATE INDEX supervisor_assignments_owner_idx
    ON supervisor_assignments(tenant_id, federation_id, supervisor_id, status);
CREATE INDEX supervisor_assignments_repository_idx
    ON supervisor_assignments(repository_id, tree_id, status);

CREATE TABLE supervisor_capabilities (
    capability_record_id VARCHAR PRIMARY KEY,
    tenant_id VARCHAR NOT NULL,
    federation_id VARCHAR NOT NULL,
    supervisor_id VARCHAR NOT NULL,
    capability_kind VARCHAR NOT NULL,
    capability_revision BIGINT NOT NULL,
    observed_generation BIGINT NOT NULL,
    freshness_state VARCHAR NOT NULL,
    evidence_ref VARCHAR NOT NULL,
    policy_id VARCHAR NOT NULL,
    policy_revision BIGINT NOT NULL,
    admission_decision_id VARCHAR NOT NULL,
    content_ref VARCHAR NOT NULL,
    expires_at VARCHAR,
    recorded_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}'
);
CREATE INDEX supervisor_capabilities_supervisor_idx
    ON supervisor_capabilities(supervisor_id, capability_kind, freshness_state);

CREATE TABLE supervisor_checkpoints (
    checkpoint_id VARCHAR PRIMARY KEY,
    tenant_id VARCHAR NOT NULL,
    federation_id VARCHAR NOT NULL,
    supervisor_id VARCHAR NOT NULL,
    assignment_revision BIGINT NOT NULL,
    fencing_epoch BIGINT NOT NULL,
    event_cursor BIGINT NOT NULL,
    world_snapshot_ref VARCHAR NOT NULL,
    content_ref VARCHAR NOT NULL,
    recorded_at VARCHAR NOT NULL
);

CREATE TABLE supervisor_receipts (
    supervisor_receipt_id VARCHAR PRIMARY KEY,
    tenant_id VARCHAR NOT NULL,
    federation_id VARCHAR NOT NULL,
    supervisor_id VARCHAR NOT NULL,
    receipt_kind VARCHAR NOT NULL,
    assignment_revision BIGINT NOT NULL,
    fencing_epoch BIGINT NOT NULL,
    content_ref VARCHAR NOT NULL,
    recorded_at VARCHAR NOT NULL
);

CREATE TABLE subagent_definitions (
    subagent_definition_id VARCHAR PRIMARY KEY,
    tenant_id VARCHAR NOT NULL,
    federation_id VARCHAR NOT NULL,
    capability_set_ref VARCHAR NOT NULL,
    allowed_operations_ref VARCHAR NOT NULL,
    effect_scope_ref VARCHAR NOT NULL,
    resource_class VARCHAR NOT NULL,
    policy_id VARCHAR NOT NULL,
    policy_revision BIGINT NOT NULL,
    authorization_evidence_ref VARCHAR NOT NULL,
    content_ref VARCHAR NOT NULL,
    created_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}'
);

CREATE TABLE subagent_instances (
    subagent_id VARCHAR PRIMARY KEY,
    tenant_id VARCHAR NOT NULL,
    federation_id VARCHAR NOT NULL,
    supervisor_id VARCHAR NOT NULL,
    subagent_definition_id VARCHAR NOT NULL,
    task_id VARCHAR NOT NULL DEFAULT '',
    lease_id VARCHAR NOT NULL DEFAULT '',
    logical_state VARCHAR NOT NULL,
    admitted_concurrency_slot BIGINT NOT NULL DEFAULT 0,
    worker_process_birth_id VARCHAR NOT NULL DEFAULT '',
    provider_route_id VARCHAR NOT NULL DEFAULT '',
    admission_decision_id VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    fencing_epoch BIGINT NOT NULL,
    registered_at VARCHAR NOT NULL,
    updated_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}'
);
CREATE INDEX subagent_instances_supervisor_idx
    ON subagent_instances(tenant_id, federation_id, supervisor_id, logical_state);

CREATE TABLE subagent_assignments (
    subagent_assignment_id VARCHAR PRIMARY KEY,
    tenant_id VARCHAR NOT NULL,
    federation_id VARCHAR NOT NULL,
    supervisor_id VARCHAR NOT NULL,
    subagent_id VARCHAR NOT NULL,
    task_cid VARCHAR NOT NULL,
    assignment_revision BIGINT NOT NULL,
    lease_id VARCHAR NOT NULL,
    fencing_epoch BIGINT NOT NULL,
    resource_reservation_id VARCHAR NOT NULL,
    token_reservation_id VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    admission_decision_id VARCHAR NOT NULL,
    content_ref VARCHAR NOT NULL,
    assigned_at VARCHAR NOT NULL,
    updated_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}'
);
CREATE INDEX subagent_assignments_task_idx
    ON subagent_assignments(task_cid, status, fencing_epoch);

CREATE TABLE subagent_capabilities (
    subagent_capability_id VARCHAR PRIMARY KEY,
    tenant_id VARCHAR NOT NULL,
    federation_id VARCHAR NOT NULL,
    subagent_id VARCHAR NOT NULL,
    capability_kind VARCHAR NOT NULL,
    capability_revision BIGINT NOT NULL,
    evidence_ref VARCHAR NOT NULL,
    policy_id VARCHAR NOT NULL,
    policy_revision BIGINT NOT NULL,
    admission_decision_id VARCHAR NOT NULL,
    content_ref VARCHAR NOT NULL,
    freshness_state VARCHAR NOT NULL,
    expires_at VARCHAR,
    recorded_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}'
);

-- A fixed row per admitted concurrent slot makes the federation-wide ceiling
-- an authoritative compare-and-swap boundary.  Logical registration is kept
-- separate from active execution admission.
CREATE TABLE subagent_execution_slots (
    tenant_id VARCHAR NOT NULL,
    federation_id VARCHAR NOT NULL,
    slot_number BIGINT NOT NULL,
    subagent_id VARCHAR,
    supervisor_id VARCHAR NOT NULL DEFAULT '',
    worker_process_birth_id VARCHAR NOT NULL DEFAULT '',
    lease_id VARCHAR NOT NULL DEFAULT '',
    fencing_epoch BIGINT NOT NULL,
    state VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    reserved_at VARCHAR,
    released_at VARCHAR,
    PRIMARY KEY (tenant_id, federation_id, slot_number)
);
CREATE INDEX subagent_execution_slots_holder_idx
    ON subagent_execution_slots(tenant_id, federation_id, subagent_id, state);

CREATE TABLE subagent_slot_ledger (
    slot_ledger_id VARCHAR PRIMARY KEY,
    tenant_id VARCHAR NOT NULL,
    federation_id VARCHAR NOT NULL,
    slot_number BIGINT NOT NULL,
    subagent_id VARCHAR NOT NULL,
    supervisor_id VARCHAR NOT NULL,
    operation VARCHAR NOT NULL,
    prior_revision BIGINT NOT NULL,
    resulting_revision BIGINT NOT NULL,
    fencing_epoch BIGINT NOT NULL,
    event_id VARCHAR NOT NULL,
    recorded_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}'
);
CREATE INDEX subagent_slot_ledger_federation_idx
    ON subagent_slot_ledger(tenant_id, federation_id, recorded_at);

CREATE TABLE subagent_outcomes (
    outcome_id VARCHAR PRIMARY KEY,
    tenant_id VARCHAR NOT NULL,
    federation_id VARCHAR NOT NULL,
    supervisor_id VARCHAR NOT NULL,
    subagent_id VARCHAR NOT NULL,
    task_id VARCHAR NOT NULL,
    attempt_id VARCHAR NOT NULL,
    fencing_epoch BIGINT NOT NULL,
    status VARCHAR NOT NULL,
    effect_ref VARCHAR NOT NULL DEFAULT '',
    evidence_ref VARCHAR NOT NULL,
    content_ref VARCHAR NOT NULL DEFAULT '',
    recorded_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}'
);

-- ---------------------------------------------------------------------------
-- Supervisor shards and revision-fenced rebalancing
-- ---------------------------------------------------------------------------
CREATE TABLE supervisor_shards (
    shard_id VARCHAR PRIMARY KEY,
    tenant_id VARCHAR NOT NULL,
    federation_id VARCHAR NOT NULL,
    shard_kind VARCHAR NOT NULL,
    active_revision BIGINT NOT NULL,
    fencing_epoch BIGINT NOT NULL,
    state VARCHAR NOT NULL,
    created_at VARCHAR NOT NULL,
    updated_at VARCHAR NOT NULL
);
CREATE INDEX supervisor_shards_federation_idx
    ON supervisor_shards(tenant_id, federation_id, state);

CREATE TABLE shard_boundaries (
    shard_boundary_id VARCHAR PRIMARY KEY,
    tenant_id VARCHAR NOT NULL,
    federation_id VARCHAR NOT NULL,
    shard_id VARCHAR NOT NULL,
    shard_revision BIGINT NOT NULL,
    repository_id VARCHAR NOT NULL DEFAULT '',
    tree_id VARCHAR NOT NULL DEFAULT '',
    goal_ref VARCHAR NOT NULL DEFAULT '',
    subgoal_ref VARCHAR NOT NULL DEFAULT '',
    task_family VARCHAR NOT NULL DEFAULT '',
    resource_class VARCHAR NOT NULL DEFAULT '',
    boundary_kind VARCHAR NOT NULL,
    boundary_ref VARCHAR NOT NULL,
    created_at VARCHAR NOT NULL
);
CREATE INDEX shard_boundaries_shard_idx
    ON shard_boundaries(shard_id, shard_revision, boundary_kind);

CREATE TABLE shard_assignments (
    shard_assignment_id VARCHAR PRIMARY KEY,
    tenant_id VARCHAR NOT NULL,
    federation_id VARCHAR NOT NULL,
    shard_id VARCHAR NOT NULL,
    shard_revision BIGINT NOT NULL,
    supervisor_id VARCHAR NOT NULL,
    assignment_revision BIGINT NOT NULL,
    fencing_epoch BIGINT NOT NULL,
    state VARCHAR NOT NULL,
    activated_at VARCHAR,
    retired_at VARCHAR
);
CREATE INDEX shard_assignments_active_idx
    ON shard_assignments(shard_id, shard_revision, state);

CREATE TABLE shard_revisions (
    shard_id VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    tenant_id VARCHAR NOT NULL,
    federation_id VARCHAR NOT NULL,
    previous_revision BIGINT NOT NULL,
    fencing_epoch BIGINT NOT NULL,
    boundary_population_ref VARCHAR NOT NULL,
    assignment_population_ref VARCHAR NOT NULL,
    state VARCHAR NOT NULL,
    content_ref VARCHAR NOT NULL,
    recorded_at VARCHAR NOT NULL,
    PRIMARY KEY (shard_id, revision)
);

CREATE TABLE shard_rebalance_plans (
    rebalance_plan_id VARCHAR PRIMARY KEY,
    tenant_id VARCHAR NOT NULL,
    federation_id VARCHAR NOT NULL,
    shard_id VARCHAR NOT NULL,
    source_revision BIGINT NOT NULL,
    target_revision BIGINT NOT NULL,
    source_supervisor_id VARCHAR NOT NULL,
    target_supervisor_id VARCHAR NOT NULL,
    state VARCHAR NOT NULL,
    content_ref VARCHAR NOT NULL,
    created_at VARCHAR NOT NULL,
    updated_at VARCHAR NOT NULL
);

CREATE TABLE shard_rebalance_receipts (
    rebalance_receipt_id VARCHAR PRIMARY KEY,
    tenant_id VARCHAR NOT NULL,
    federation_id VARCHAR NOT NULL,
    rebalance_plan_id VARCHAR NOT NULL,
    shard_id VARCHAR NOT NULL,
    source_revision BIGINT NOT NULL,
    target_revision BIGINT NOT NULL,
    fencing_epoch BIGINT NOT NULL,
    disposition VARCHAR NOT NULL,
    content_ref VARCHAR NOT NULL,
    recorded_at VARCHAR NOT NULL
);

-- ---------------------------------------------------------------------------
-- Hierarchical budget accounts, resource/token ceilings, and ledger
-- ---------------------------------------------------------------------------
CREATE TABLE federation_budgets (
    federation_budget_id VARCHAR PRIMARY KEY,
    tenant_id VARCHAR NOT NULL,
    federation_id VARCHAR NOT NULL,
    parent_budget_id VARCHAR NOT NULL DEFAULT '',
    owner_id VARCHAR NOT NULL,
    policy_id VARCHAR NOT NULL,
    policy_revision BIGINT NOT NULL,
    revision BIGINT NOT NULL,
    status VARCHAR NOT NULL,
    content_ref VARCHAR NOT NULL,
    created_at VARCHAR NOT NULL,
    updated_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}'
);

CREATE TABLE supervisor_budgets (
    supervisor_budget_id VARCHAR PRIMARY KEY,
    tenant_id VARCHAR NOT NULL,
    federation_id VARCHAR NOT NULL,
    supervisor_id VARCHAR NOT NULL,
    federation_budget_id VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    state VARCHAR NOT NULL,
    content_ref VARCHAR NOT NULL,
    created_at VARCHAR NOT NULL,
    updated_at VARCHAR NOT NULL
);

CREATE TABLE agent_budgets (
    agent_budget_id VARCHAR PRIMARY KEY,
    tenant_id VARCHAR NOT NULL,
    federation_id VARCHAR NOT NULL,
    supervisor_id VARCHAR NOT NULL,
    subagent_id VARCHAR NOT NULL,
    supervisor_budget_id VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    state VARCHAR NOT NULL,
    content_ref VARCHAR NOT NULL,
    created_at VARCHAR NOT NULL,
    updated_at VARCHAR NOT NULL
);

CREATE TABLE token_budgets (
    token_budget_id VARCHAR PRIMARY KEY,
    tenant_id VARCHAR NOT NULL,
    federation_id VARCHAR NOT NULL,
    parent_budget_id VARCHAR NOT NULL,
    parent_budget_kind VARCHAR NOT NULL,
    maximum_input_tokens BIGINT NOT NULL,
    maximum_output_tokens BIGINT NOT NULL,
    maximum_model_calls BIGINT NOT NULL,
    maximum_provider_spend_micros BIGINT NOT NULL,
    reserved_input_tokens BIGINT NOT NULL,
    reserved_output_tokens BIGINT NOT NULL,
    consumed_input_tokens BIGINT NOT NULL,
    consumed_output_tokens BIGINT NOT NULL,
    revision BIGINT NOT NULL,
    state VARCHAR NOT NULL,
    updated_at VARCHAR NOT NULL
);

CREATE TABLE resource_budgets (
    resource_budget_id VARCHAR PRIMARY KEY,
    tenant_id VARCHAR NOT NULL,
    federation_id VARCHAR NOT NULL,
    parent_budget_id VARCHAR NOT NULL,
    parent_budget_kind VARCHAR NOT NULL,
    resource_dimension VARCHAR NOT NULL,
    limit_amount BIGINT NOT NULL,
    reserved_amount BIGINT NOT NULL,
    consumed_amount BIGINT NOT NULL,
    unit VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    state VARCHAR NOT NULL,
    updated_at VARCHAR NOT NULL
);
CREATE INDEX resource_budgets_parent_idx
    ON resource_budgets(parent_budget_kind, parent_budget_id, resource_dimension);

CREATE TABLE budget_ledger (
    budget_ledger_entry_id VARCHAR PRIMARY KEY,
    tenant_id VARCHAR NOT NULL,
    federation_id VARCHAR NOT NULL,
    budget_id VARCHAR NOT NULL,
    budget_kind VARCHAR NOT NULL,
    reservation_id VARCHAR NOT NULL DEFAULT '',
    parent_budget_id VARCHAR NOT NULL DEFAULT '',
    operation_kind VARCHAR NOT NULL,
    budget_dimension VARCHAR NOT NULL,
    amount BIGINT NOT NULL,
    expected_revision BIGINT NOT NULL,
    resulting_revision BIGINT NOT NULL,
    event_id VARCHAR NOT NULL,
    recorded_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}'
);
CREATE INDEX budget_ledger_budget_idx
    ON budget_ledger(budget_kind, budget_id, resulting_revision);

-- External federation admission uses a typed reservation group rather than an
-- opaque caller-supplied reference.  The state owner reserves all dimensions
-- transactionally and federation.create consumes the exact group under the
-- same tenant/request/policy/fence binding.
CREATE TABLE federation_admission_budget_reservations (
    reservation_id VARCHAR PRIMARY KEY,
    tenant_id VARCHAR NOT NULL,
    federation_id VARCHAR NOT NULL,
    request_cid VARCHAR NOT NULL,
    idempotency_key VARCHAR NOT NULL,
    policy_id VARCHAR NOT NULL,
    policy_revision BIGINT NOT NULL,
    resource_budget_id VARCHAR NOT NULL,
    token_budget_id VARCHAR NOT NULL,
    parent_budget_id VARCHAR NOT NULL,
    owner_id VARCHAR NOT NULL,
    authorization_evidence_ref VARCHAR NOT NULL,
    issued_at VARCHAR NOT NULL,
    expires_at VARCHAR NOT NULL,
    state VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    fencing_epoch BIGINT NOT NULL,
    content_ref VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);
CREATE UNIQUE INDEX federation_admission_budget_idempotency_uidx
    ON federation_admission_budget_reservations(tenant_id, idempotency_key);
CREATE UNIQUE INDEX federation_admission_budget_request_uidx
    ON federation_admission_budget_reservations(tenant_id, federation_id, request_cid);

CREATE TABLE federation_admission_budget_dimensions (
    reservation_id VARCHAR NOT NULL,
    tenant_id VARCHAR NOT NULL,
    federation_id VARCHAR NOT NULL,
    dimension_name VARCHAR NOT NULL,
    ceiling_amount BIGINT NOT NULL,
    reserved_amount BIGINT NOT NULL,
    consumed_amount BIGINT NOT NULL,
    ordinal BIGINT NOT NULL,
    PRIMARY KEY (reservation_id, dimension_name)
);
CREATE INDEX federation_admission_budget_dimensions_scope_idx
    ON federation_admission_budget_dimensions(
        tenant_id, federation_id, dimension_name, reservation_id
    );

-- ---------------------------------------------------------------------------
-- Transactional event outbox, bounded subscriptions, delivery, and evidence
-- ---------------------------------------------------------------------------
CREATE TABLE stream_sequence_heads (
    stream_id VARCHAR PRIMARY KEY,
    tenant_id VARCHAR NOT NULL,
    federation_id VARCHAR NOT NULL,
    current_sequence BIGINT NOT NULL,
    revision BIGINT NOT NULL,
    updated_at VARCHAR NOT NULL
);

CREATE TABLE global_sequence_head (
    head_id VARCHAR PRIMARY KEY,
    current_sequence BIGINT NOT NULL,
    revision BIGINT NOT NULL,
    updated_at VARCHAR NOT NULL
);

CREATE TABLE transactional_outbox (
    outbox_id VARCHAR PRIMARY KEY,
    event_id VARCHAR NOT NULL,
    event_cid VARCHAR NOT NULL,
    tenant_id VARCHAR NOT NULL,
    federation_id VARCHAR NOT NULL,
    stream_id VARCHAR NOT NULL,
    stream_sequence BIGINT NOT NULL,
    global_sequence BIGINT NOT NULL,
    effect_class VARCHAR NOT NULL,
    deduplication_key VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    next_attempt_at VARCHAR NOT NULL,
    claimed_by VARCHAR NOT NULL DEFAULT '',
    claim_fencing_epoch BIGINT NOT NULL DEFAULT 0,
    attempt_count BIGINT NOT NULL DEFAULT 0,
    projected_at VARCHAR,
    revision BIGINT NOT NULL,
    created_at VARCHAR NOT NULL,
    updated_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}'
);
CREATE UNIQUE INDEX transactional_outbox_event_uidx
    ON transactional_outbox(event_id);
CREATE INDEX transactional_outbox_ready_idx
    ON transactional_outbox(status, next_attempt_at, global_sequence);

-- The outbox router is a replayable derived-state producer.  A disposition
-- binds the exact source events to the durable route batch before their
-- outbox rows leave the pending population.  The normalized event members
-- keep restart/audit joins out of opaque JSON.
CREATE TABLE outbox_routing_dispositions (
    disposition_id VARCHAR PRIMARY KEY,
    tenant_id VARCHAR NOT NULL,
    federation_id VARCHAR NOT NULL,
    route_batch_id VARCHAR NOT NULL,
    first_global_sequence BIGINT NOT NULL,
    last_global_sequence BIGINT NOT NULL,
    event_count BIGINT NOT NULL,
    delivery_count BIGINT NOT NULL,
    subscription_count BIGINT NOT NULL,
    status VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    content_ref VARCHAR NOT NULL,
    created_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);
CREATE UNIQUE INDEX outbox_routing_dispositions_batch_uidx
    ON outbox_routing_dispositions(
        tenant_id, federation_id, route_batch_id, content_ref
    );
CREATE INDEX outbox_routing_dispositions_sequence_idx
    ON outbox_routing_dispositions(
        tenant_id, federation_id, last_global_sequence
    );

CREATE TABLE outbox_routing_disposition_events (
    disposition_id VARCHAR NOT NULL,
    event_id VARCHAR NOT NULL,
    global_sequence BIGINT NOT NULL,
    ordinal BIGINT NOT NULL,
    PRIMARY KEY (disposition_id, event_id)
);
CREATE UNIQUE INDEX outbox_routing_disposition_events_ordinal_uidx
    ON outbox_routing_disposition_events(disposition_id, ordinal);
CREATE INDEX outbox_routing_disposition_events_event_idx
    ON outbox_routing_disposition_events(event_id, disposition_id);

CREATE TABLE event_subscriptions (
    subscription_id VARCHAR PRIMARY KEY,
    tenant_id VARCHAR NOT NULL,
    federation_id VARCHAR NOT NULL,
    consumer_id VARCHAR NOT NULL,
    supervisor_id VARCHAR NOT NULL DEFAULT '',
    revision BIGINT NOT NULL,
    event_classes_json VARCHAR NOT NULL DEFAULT '[]',
    maximum_batch BIGINT NOT NULL,
    maximum_pending BIGINT NOT NULL,
    maximum_fanout BIGINT NOT NULL,
    retry_budget BIGINT NOT NULL,
    consecutive_failures BIGINT NOT NULL DEFAULT 0,
    expires_at VARCHAR,
    status VARCHAR NOT NULL,
    created_at VARCHAR NOT NULL,
    updated_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}'
);
CREATE INDEX event_subscriptions_consumer_idx
    ON event_subscriptions(tenant_id, federation_id, consumer_id, status);

CREATE TABLE event_subscription_selectors (
    selector_id VARCHAR PRIMARY KEY,
    subscription_id VARCHAR NOT NULL,
    subscription_revision BIGINT NOT NULL,
    selector_kind VARCHAR NOT NULL,
    selector_value VARCHAR NOT NULL,
    ordinal BIGINT NOT NULL
);
CREATE INDEX event_subscription_selectors_subscription_idx
    ON event_subscription_selectors(subscription_id, subscription_revision, ordinal);
CREATE INDEX event_subscription_selectors_match_idx
    ON event_subscription_selectors(selector_kind, selector_value);

CREATE TABLE consumer_cursors (
    consumer_id VARCHAR NOT NULL,
    subscription_id VARCHAR NOT NULL,
    subscription_revision BIGINT NOT NULL,
    tenant_id VARCHAR NOT NULL,
    federation_id VARCHAR NOT NULL,
    global_sequence BIGINT NOT NULL,
    store_generation BIGINT NOT NULL,
    last_event_id VARCHAR NOT NULL DEFAULT '',
    processing_event_id VARCHAR NOT NULL DEFAULT '',
    fencing_epoch BIGINT NOT NULL,
    revision BIGINT NOT NULL,
    updated_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}',
    PRIMARY KEY (consumer_id, subscription_id)
);

CREATE TABLE supervisor_inbox (
    inbox_entry_id VARCHAR PRIMARY KEY,
    tenant_id VARCHAR NOT NULL,
    federation_id VARCHAR NOT NULL,
    supervisor_id VARCHAR NOT NULL,
    subscription_id VARCHAR NOT NULL,
    event_id VARCHAR NOT NULL,
    event_cid VARCHAR NOT NULL,
    global_sequence BIGINT NOT NULL,
    status VARCHAR NOT NULL,
    available_at VARCHAR NOT NULL,
    expires_at VARCHAR,
    revision BIGINT NOT NULL,
    created_at VARCHAR NOT NULL,
    updated_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}'
);
CREATE UNIQUE INDEX supervisor_inbox_event_uidx
    ON supervisor_inbox(supervisor_id, subscription_id, event_id);
CREATE INDEX supervisor_inbox_ready_idx
    ON supervisor_inbox(supervisor_id, status, available_at, global_sequence);

CREATE TABLE delivery_attempts (
    attempt_id VARCHAR PRIMARY KEY,
    tenant_id VARCHAR NOT NULL,
    federation_id VARCHAR NOT NULL,
    event_id VARCHAR NOT NULL,
    outbox_id VARCHAR NOT NULL,
    delivery_id VARCHAR NOT NULL DEFAULT '',
    subscription_id VARCHAR NOT NULL,
    subscription_revision BIGINT NOT NULL,
    consumer_id VARCHAR NOT NULL,
    attempt_number BIGINT NOT NULL,
    fencing_epoch BIGINT NOT NULL,
    status VARCHAR NOT NULL,
    error_code VARCHAR NOT NULL DEFAULT '',
    recorded_at VARCHAR NOT NULL,
    finished_at VARCHAR,
    body_json VARCHAR NOT NULL DEFAULT '{}'
);
CREATE INDEX delivery_attempts_event_idx
    ON delivery_attempts(event_id, subscription_id, attempt_number);
CREATE UNIQUE INDEX delivery_attempts_owner_attempt_uidx
    ON delivery_attempts(
        event_id, subscription_id, subscription_revision,
        consumer_id, attempt_number
    );

CREATE TABLE dead_letters (
    dead_letter_id VARCHAR PRIMARY KEY,
    tenant_id VARCHAR NOT NULL,
    federation_id VARCHAR NOT NULL,
    event_id VARCHAR NOT NULL,
    outbox_id VARCHAR NOT NULL,
    subscription_id VARCHAR NOT NULL,
    subscription_revision BIGINT NOT NULL,
    consumer_id VARCHAR NOT NULL,
    retry_count BIGINT NOT NULL,
    error_code VARCHAR NOT NULL,
    evidence_ref VARCHAR NOT NULL,
    quarantined BOOLEAN NOT NULL,
    status VARCHAR NOT NULL,
    created_at VARCHAR NOT NULL,
    expires_at VARCHAR,
    resolved_at VARCHAR,
    revision BIGINT NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}'
);
CREATE INDEX dead_letters_consumer_idx
    ON dead_letters(consumer_id, status, created_at);

-- Durable delivery projections are separate from immutable domain events.
-- The queue is restart-safe, while coverage preserves every input event that
-- a coalesced wakeup represents.
CREATE TABLE event_delivery_queue (
    delivery_id VARCHAR PRIMARY KEY,
    tenant_id VARCHAR NOT NULL,
    federation_id VARCHAR NOT NULL,
    subscription_id VARCHAR NOT NULL,
    subscription_revision BIGINT NOT NULL,
    consumer_id VARCHAR NOT NULL,
    decision_id VARCHAR NOT NULL,
    representative_event_id VARCHAR NOT NULL,
    outbox_id VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    attempt_number BIGINT NOT NULL,
    fencing_epoch BIGINT NOT NULL,
    available_at VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    created_at VARCHAR NOT NULL,
    updated_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);
CREATE UNIQUE INDEX event_delivery_queue_decision_uidx
    ON event_delivery_queue(
        subscription_id, subscription_revision, decision_id
    );
CREATE INDEX event_delivery_queue_ready_idx
    ON event_delivery_queue(
        subscription_id, subscription_revision, status, available_at,
        representative_event_id
    );

CREATE TABLE event_coalescing_coverage (
    coverage_id VARCHAR PRIMARY KEY,
    decision_id VARCHAR NOT NULL,
    tenant_id VARCHAR NOT NULL,
    federation_id VARCHAR NOT NULL,
    subscription_id VARCHAR NOT NULL,
    subscription_revision BIGINT NOT NULL,
    representative_event_id VARCHAR NOT NULL,
    coalescing_mode VARCHAR NOT NULL,
    input_event_count BIGINT NOT NULL,
    content_ref VARCHAR NOT NULL,
    created_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);
CREATE UNIQUE INDEX event_coalescing_coverage_decision_uidx
    ON event_coalescing_coverage(
        subscription_id, subscription_revision, decision_id
    );

CREATE TABLE event_coalescing_inputs (
    coverage_id VARCHAR NOT NULL,
    event_id VARCHAR NOT NULL,
    ordinal BIGINT NOT NULL,
    PRIMARY KEY (coverage_id, event_id)
);
CREATE INDEX event_coalescing_inputs_event_idx
    ON event_coalescing_inputs(event_id, coverage_id);

CREATE TABLE event_coalescing (
    coalescing_id VARCHAR PRIMARY KEY,
    tenant_id VARCHAR NOT NULL,
    federation_id VARCHAR NOT NULL,
    subscription_id VARCHAR NOT NULL,
    coalescing_key VARCHAR NOT NULL,
    event_class VARCHAR NOT NULL,
    current_event_id VARCHAR NOT NULL,
    superseded_event_count BIGINT NOT NULL,
    first_global_sequence BIGINT NOT NULL,
    latest_global_sequence BIGINT NOT NULL,
    status VARCHAR NOT NULL,
    expires_at VARCHAR,
    revision BIGINT NOT NULL,
    updated_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}'
);
CREATE UNIQUE INDEX event_coalescing_key_uidx
    ON event_coalescing(subscription_id, coalescing_key);

CREATE TABLE event_acknowledgements (
    acknowledgement_id VARCHAR PRIMARY KEY,
    tenant_id VARCHAR NOT NULL,
    federation_id VARCHAR NOT NULL,
    event_id VARCHAR NOT NULL,
    subscription_id VARCHAR NOT NULL,
    consumer_id VARCHAR NOT NULL,
    subscription_revision BIGINT NOT NULL,
    global_sequence BIGINT NOT NULL,
    delivery_attempt_id VARCHAR NOT NULL,
    cursor_revision BIGINT NOT NULL,
    fencing_epoch BIGINT NOT NULL,
    disposition VARCHAR NOT NULL,
    processed_effect_ref VARCHAR NOT NULL DEFAULT '',
    recorded_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}'
);
CREATE UNIQUE INDEX event_acknowledgements_delivery_uidx
    ON event_acknowledgements(delivery_attempt_id);

-- ---------------------------------------------------------------------------
-- Multilevel causal graph and abstraction metadata
-- ---------------------------------------------------------------------------
CREATE TABLE causal_nodes (
    causal_node_id VARCHAR PRIMARY KEY,
    tenant_id VARCHAR NOT NULL,
    federation_id VARCHAR NOT NULL,
    causal_level VARCHAR NOT NULL,
    node_kind VARCHAR NOT NULL,
    subject_ref VARCHAR NOT NULL,
    repository_id VARCHAR NOT NULL DEFAULT '',
    tree_id VARCHAR NOT NULL DEFAULT '',
    owner_id VARCHAR NOT NULL,
    source_root VARCHAR NOT NULL,
    content_ref VARCHAR NOT NULL,
    graph_revision BIGINT NOT NULL,
    freshness_state VARCHAR NOT NULL,
    created_at VARCHAR NOT NULL
);
CREATE INDEX causal_nodes_subject_idx
    ON causal_nodes(federation_id, causal_level, subject_ref, graph_revision);

CREATE TABLE causal_edges (
    causal_edge_id VARCHAR PRIMARY KEY,
    tenant_id VARCHAR NOT NULL,
    federation_id VARCHAR NOT NULL,
    source_node_id VARCHAR NOT NULL,
    target_node_id VARCHAR NOT NULL,
    edge_kind VARCHAR NOT NULL,
    graph_revision BIGINT NOT NULL,
    authority_disposition VARCHAR NOT NULL,
    evidence_population_ref VARCHAR NOT NULL,
    admitted_policy_ref VARCHAR NOT NULL DEFAULT '',
    created_at VARCHAR NOT NULL,
    retired_at VARCHAR
);
CREATE INDEX causal_edges_source_idx
    ON causal_edges(source_node_id, edge_kind, graph_revision);
CREATE INDEX causal_edges_target_idx
    ON causal_edges(target_node_id, edge_kind, graph_revision);

CREATE TABLE causal_evidence (
    causal_evidence_id VARCHAR PRIMARY KEY,
    tenant_id VARCHAR NOT NULL,
    federation_id VARCHAR NOT NULL,
    causal_edge_id VARCHAR NOT NULL DEFAULT '',
    evidence_kind VARCHAR NOT NULL,
    authority_disposition VARCHAR NOT NULL,
    repository_id VARCHAR NOT NULL DEFAULT '',
    tree_id VARCHAR NOT NULL DEFAULT '',
    owner_id VARCHAR NOT NULL,
    source_root VARCHAR NOT NULL,
    content_ref VARCHAR NOT NULL,
    observed_at VARCHAR NOT NULL,
    expires_at VARCHAR
);

CREATE TABLE causal_abstraction_maps (
    abstraction_map_id VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    tenant_id VARCHAR NOT NULL,
    federation_id VARCHAR NOT NULL,
    low_level_model_ref VARCHAR NOT NULL,
    high_level_model_ref VARCHAR NOT NULL,
    abstraction_function_ref VARCHAR NOT NULL,
    intervention_mapping_ref VARCHAR NOT NULL,
    admitted_domain_ref VARCHAR NOT NULL,
    excluded_domain_ref VARCHAR NOT NULL,
    validation_evidence_ref VARCHAR NOT NULL,
    faithfulness_status VARCHAR NOT NULL,
    policy_admission_ref VARCHAR NOT NULL DEFAULT '',
    content_ref VARCHAR NOT NULL,
    recorded_at VARCHAR NOT NULL,
    PRIMARY KEY (abstraction_map_id, revision)
);

CREATE TABLE causal_abstraction_validations (
    abstraction_validation_id VARCHAR PRIMARY KEY,
    tenant_id VARCHAR NOT NULL,
    federation_id VARCHAR NOT NULL,
    abstraction_map_id VARCHAR NOT NULL,
    abstraction_map_revision BIGINT NOT NULL,
    intervention_population_ref VARCHAR NOT NULL,
    mismatch_count BIGINT NOT NULL,
    excluded_domain_ref VARCHAR NOT NULL,
    resulting_status VARCHAR NOT NULL,
    evidence_ref VARCHAR NOT NULL,
    recorded_at VARCHAR NOT NULL
);

CREATE TABLE causal_intervention_tests (
    intervention_test_id VARCHAR PRIMARY KEY,
    tenant_id VARCHAR NOT NULL,
    federation_id VARCHAR NOT NULL,
    abstraction_map_id VARCHAR NOT NULL,
    abstraction_map_revision BIGINT NOT NULL,
    low_level_variable_ref VARCHAR NOT NULL,
    low_level_outcome_ref VARCHAR NOT NULL,
    abstracted_outcome_ref VARCHAR NOT NULL,
    high_level_intervention_ref VARCHAR NOT NULL,
    high_level_outcome_ref VARCHAR NOT NULL,
    disposition VARCHAR NOT NULL,
    evidence_ref VARCHAR NOT NULL,
    recorded_at VARCHAR NOT NULL
);

CREATE TABLE causal_slices (
    causal_slice_id VARCHAR PRIMARY KEY,
    tenant_id VARCHAR NOT NULL,
    federation_id VARCHAR NOT NULL,
    graph_revision BIGINT NOT NULL,
    root_event_id VARCHAR NOT NULL,
    root_fact_ref VARCHAR NOT NULL,
    node_population_ref VARCHAR NOT NULL,
    edge_population_ref VARCHAR NOT NULL,
    content_ref VARCHAR NOT NULL,
    created_at VARCHAR NOT NULL
);

CREATE TABLE causal_frontiers (
    causal_frontier_id VARCHAR PRIMARY KEY,
    tenant_id VARCHAR NOT NULL,
    federation_id VARCHAR NOT NULL,
    event_id VARCHAR NOT NULL,
    graph_revision BIGINT NOT NULL,
    abstraction_revision_ref VARCHAR NOT NULL,
    must_wake_count BIGINT NOT NULL,
    may_wake_count BIGINT NOT NULL,
    do_not_wake_count BIGINT NOT NULL,
    content_ref VARCHAR NOT NULL,
    created_at VARCHAR NOT NULL
);

CREATE TABLE causal_frontier_members (
    causal_frontier_id VARCHAR NOT NULL,
    subject_kind VARCHAR NOT NULL,
    subject_ref VARCHAR NOT NULL,
    disposition VARCHAR NOT NULL,
    evidence_ref VARCHAR NOT NULL,
    ordinal BIGINT NOT NULL,
    PRIMARY KEY (causal_frontier_id, subject_kind, subject_ref)
);
CREATE INDEX causal_frontier_members_subject_idx
    ON causal_frontier_members(subject_kind, subject_ref, disposition);

CREATE TABLE causal_invalidations (
    causal_invalidation_id VARCHAR PRIMARY KEY,
    tenant_id VARCHAR NOT NULL,
    federation_id VARCHAR NOT NULL,
    event_id VARCHAR NOT NULL,
    graph_revision BIGINT NOT NULL,
    subject_kind VARCHAR NOT NULL,
    subject_ref VARCHAR NOT NULL,
    reason_kind VARCHAR NOT NULL,
    evidence_ref VARCHAR NOT NULL,
    state VARCHAR NOT NULL,
    created_at VARCHAR NOT NULL,
    resolved_at VARCHAR
);

-- ---------------------------------------------------------------------------
-- Opaque semantic, proof/test, and retrieval authority references
-- ---------------------------------------------------------------------------
CREATE TABLE semantic_root_references (
    semantic_root_reference_id VARCHAR PRIMARY KEY,
    tenant_id VARCHAR NOT NULL,
    federation_id VARCHAR NOT NULL,
    repository_id VARCHAR NOT NULL,
    tree_id VARCHAR NOT NULL,
    semantic_root VARCHAR NOT NULL,
    semantic_kind VARCHAR NOT NULL,
    owner_id VARCHAR NOT NULL,
    source_root VARCHAR NOT NULL,
    content_ref VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    status VARCHAR NOT NULL,
    freshness_state VARCHAR NOT NULL,
    provenance_ref VARCHAR NOT NULL,
    recorded_at VARCHAR NOT NULL
);
CREATE INDEX semantic_root_references_tree_idx
    ON semantic_root_references(repository_id, tree_id, semantic_kind, revision);

CREATE TABLE semantic_capsule_references (
    semantic_capsule_reference_id VARCHAR PRIMARY KEY,
    tenant_id VARCHAR NOT NULL,
    federation_id VARCHAR NOT NULL,
    repository_id VARCHAR NOT NULL,
    tree_id VARCHAR NOT NULL,
    subject_kind VARCHAR NOT NULL,
    subject_ref VARCHAR NOT NULL,
    dependency_root VARCHAR NOT NULL,
    owner_id VARCHAR NOT NULL,
    source_root VARCHAR NOT NULL,
    content_ref VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    status VARCHAR NOT NULL,
    freshness_state VARCHAR NOT NULL,
    provenance_ref VARCHAR NOT NULL,
    recorded_at VARCHAR NOT NULL
);
CREATE INDEX semantic_capsule_references_subject_idx
    ON semantic_capsule_references(repository_id, tree_id, subject_kind, subject_ref);

CREATE TABLE world_snapshot_references (
    world_snapshot_reference_id VARCHAR PRIMARY KEY,
    tenant_id VARCHAR NOT NULL,
    federation_id VARCHAR NOT NULL,
    repository_id VARCHAR NOT NULL DEFAULT '',
    tree_id VARCHAR NOT NULL DEFAULT '',
    control_plane_generation BIGINT NOT NULL,
    causal_graph_revision BIGINT NOT NULL,
    semantic_state_root VARCHAR NOT NULL,
    event_watermark BIGINT NOT NULL,
    owner_id VARCHAR NOT NULL,
    source_root VARCHAR NOT NULL,
    content_ref VARCHAR NOT NULL,
    freshness_state VARCHAR NOT NULL,
    recorded_at VARCHAR NOT NULL
);

CREATE TABLE proof_reference_projections (
    proof_reference_id VARCHAR PRIMARY KEY,
    tenant_id VARCHAR NOT NULL,
    federation_id VARCHAR NOT NULL,
    repository_id VARCHAR NOT NULL,
    tree_id VARCHAR NOT NULL,
    task_cid VARCHAR NOT NULL DEFAULT '',
    obligation_ref VARCHAR NOT NULL,
    proof_kind VARCHAR NOT NULL,
    proof_status VARCHAR NOT NULL,
    owner_id VARCHAR NOT NULL,
    source_root VARCHAR NOT NULL,
    content_ref VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    status VARCHAR NOT NULL,
    freshness_state VARCHAR NOT NULL,
    provenance_ref VARCHAR NOT NULL,
    recorded_at VARCHAR NOT NULL
);
CREATE INDEX proof_reference_projections_obligation_idx
    ON proof_reference_projections(repository_id, tree_id, obligation_ref, revision);

CREATE TABLE test_reference_projections (
    test_reference_id VARCHAR PRIMARY KEY,
    tenant_id VARCHAR NOT NULL,
    federation_id VARCHAR NOT NULL,
    repository_id VARCHAR NOT NULL,
    tree_id VARCHAR NOT NULL,
    task_cid VARCHAR NOT NULL DEFAULT '',
    test_ref VARCHAR NOT NULL,
    test_kind VARCHAR NOT NULL,
    test_status VARCHAR NOT NULL,
    owner_id VARCHAR NOT NULL,
    source_root VARCHAR NOT NULL,
    content_ref VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    status VARCHAR NOT NULL,
    freshness_state VARCHAR NOT NULL,
    provenance_ref VARCHAR NOT NULL,
    recorded_at VARCHAR NOT NULL
);
CREATE INDEX test_reference_projections_test_idx
    ON test_reference_projections(repository_id, tree_id, test_ref, revision);

CREATE TABLE retrieval_indexes (
    retrieval_index_id VARCHAR PRIMARY KEY,
    tenant_id VARCHAR NOT NULL,
    federation_id VARCHAR NOT NULL,
    repository_id VARCHAR NOT NULL,
    tree_id VARCHAR NOT NULL,
    retrieval_method VARCHAR NOT NULL,
    index_revision BIGINT NOT NULL,
    index_root VARCHAR NOT NULL,
    owner_id VARCHAR NOT NULL,
    source_root VARCHAR NOT NULL,
    content_ref VARCHAR NOT NULL,
    freshness_state VARCHAR NOT NULL,
    privacy_scope_ref VARCHAR NOT NULL,
    recorded_at VARCHAR NOT NULL
);
CREATE INDEX retrieval_indexes_tree_idx
    ON retrieval_indexes(repository_id, tree_id, retrieval_method, index_revision);

CREATE TABLE retrieval_index_partitions (
    retrieval_partition_id VARCHAR PRIMARY KEY,
    retrieval_index_id VARCHAR NOT NULL,
    index_revision BIGINT NOT NULL,
    partition_kind VARCHAR NOT NULL,
    partition_key VARCHAR NOT NULL,
    owner_id VARCHAR NOT NULL,
    source_root VARCHAR NOT NULL,
    content_ref VARCHAR NOT NULL,
    record_count BIGINT NOT NULL,
    checksum VARCHAR NOT NULL,
    recorded_at VARCHAR NOT NULL
);

CREATE TABLE retrieval_receipts (
    retrieval_receipt_id VARCHAR PRIMARY KEY,
    tenant_id VARCHAR NOT NULL,
    federation_id VARCHAR NOT NULL,
    retrieval_index_id VARCHAR NOT NULL,
    index_revision BIGINT NOT NULL,
    repository_id VARCHAR NOT NULL,
    tree_id VARCHAR NOT NULL,
    retrieval_method VARCHAR NOT NULL,
    partition_population_ref VARCHAR NOT NULL,
    result_population_ref VARCHAR NOT NULL,
    owner_id VARCHAR NOT NULL,
    source_root VARCHAR NOT NULL,
    content_ref VARCHAR NOT NULL,
    recorded_at VARCHAR NOT NULL
);

CREATE TABLE retrieval_nominations (
    retrieval_nomination_id VARCHAR PRIMARY KEY,
    tenant_id VARCHAR NOT NULL,
    federation_id VARCHAR NOT NULL,
    retrieval_receipt_id VARCHAR NOT NULL,
    subject_kind VARCHAR NOT NULL,
    subject_ref VARCHAR NOT NULL,
    score_micros BIGINT NOT NULL,
    authority_disposition VARCHAR NOT NULL,
    owner_id VARCHAR NOT NULL,
    source_root VARCHAR NOT NULL,
    content_ref VARCHAR NOT NULL,
    recorded_at VARCHAR NOT NULL
);
CREATE INDEX retrieval_nominations_subject_idx
    ON retrieval_nominations(subject_kind, subject_ref, authority_disposition);

-- ---------------------------------------------------------------------------
-- Typed command authorization, effect observation, and audit receipts
-- ---------------------------------------------------------------------------
CREATE TABLE federation_commands (
    federation_command_id VARCHAR PRIMARY KEY,
    tenant_id VARCHAR NOT NULL,
    federation_id VARCHAR NOT NULL,
    supervisor_id VARCHAR NOT NULL DEFAULT '',
    caller_id VARCHAR NOT NULL,
    operation VARCHAR NOT NULL,
    idempotency_key VARCHAR NOT NULL,
    expected_generation BIGINT NOT NULL,
    expected_revision BIGINT NOT NULL,
    fencing_epoch BIGINT NOT NULL,
    authorization_decision_id VARCHAR NOT NULL,
    expected_effects_ref VARCHAR NOT NULL,
    dry_run BOOLEAN NOT NULL,
    state VARCHAR NOT NULL,
    result_ref VARCHAR NOT NULL DEFAULT '',
    created_at VARCHAR NOT NULL,
    completed_at VARCHAR
);
CREATE UNIQUE INDEX federation_commands_idempotency_uidx
    ON federation_commands(tenant_id, federation_id, idempotency_key);

CREATE TABLE federation_authorization_decisions (
    authorization_decision_id VARCHAR PRIMARY KEY,
    tenant_id VARCHAR NOT NULL,
    federation_id VARCHAR NOT NULL,
    request_cid VARCHAR NOT NULL,
    caller_id VARCHAR NOT NULL,
    delegation_chain_ref VARCHAR NOT NULL,
    audience VARCHAR NOT NULL,
    operation VARCHAR NOT NULL,
    resource_scope_ref VARCHAR NOT NULL,
    policy_id VARCHAR NOT NULL,
    policy_revision BIGINT NOT NULL,
    verdict VARCHAR NOT NULL,
    reason_code VARCHAR NOT NULL,
    evidence_ref VARCHAR NOT NULL,
    expires_at VARCHAR NOT NULL,
    decided_at VARCHAR NOT NULL,
    content_ref VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}'
);
CREATE INDEX federation_authorization_decisions_caller_idx
    ON federation_authorization_decisions(tenant_id, caller_id, decided_at);
CREATE INDEX federation_authorization_decisions_request_idx
    ON federation_authorization_decisions(tenant_id, request_cid);

CREATE TABLE federation_policy_decisions (
    policy_decision_id VARCHAR PRIMARY KEY,
    tenant_id VARCHAR NOT NULL,
    federation_id VARCHAR NOT NULL,
    policy_id VARCHAR NOT NULL,
    policy_revision BIGINT NOT NULL,
    subject_kind VARCHAR NOT NULL,
    subject_ref VARCHAR NOT NULL,
    operation VARCHAR NOT NULL,
    verdict VARCHAR NOT NULL,
    reason_code VARCHAR NOT NULL,
    evidence_ref VARCHAR NOT NULL,
    decided_at VARCHAR NOT NULL
);

CREATE TABLE federation_confirmation_receipts (
    confirmation_receipt_id VARCHAR PRIMARY KEY,
    tenant_id VARCHAR NOT NULL,
    federation_id VARCHAR NOT NULL,
    federation_command_id VARCHAR NOT NULL,
    confirmation_kind VARCHAR NOT NULL,
    confirmer_id VARCHAR NOT NULL,
    target_roots_ref VARCHAR NOT NULL,
    expected_effects_ref VARCHAR NOT NULL,
    content_ref VARCHAR NOT NULL,
    recorded_at VARCHAR NOT NULL
);

CREATE TABLE federation_effect_reservations (
    effect_reservation_id VARCHAR PRIMARY KEY,
    tenant_id VARCHAR NOT NULL,
    federation_id VARCHAR NOT NULL,
    supervisor_id VARCHAR NOT NULL,
    subagent_id VARCHAR NOT NULL DEFAULT '',
    task_cid VARCHAR NOT NULL,
    attempt_id VARCHAR NOT NULL,
    effect_class VARCHAR NOT NULL,
    target_ref VARCHAR NOT NULL,
    lease_id VARCHAR NOT NULL,
    fencing_epoch BIGINT NOT NULL,
    idempotency_key VARCHAR NOT NULL,
    state VARCHAR NOT NULL,
    reserved_at VARCHAR NOT NULL,
    expires_at VARCHAR NOT NULL
);
CREATE UNIQUE INDEX federation_effect_reservations_identity_uidx
    ON federation_effect_reservations(tenant_id, federation_id, idempotency_key);

CREATE TABLE federation_effect_observations (
    effect_observation_id VARCHAR PRIMARY KEY,
    tenant_id VARCHAR NOT NULL,
    federation_id VARCHAR NOT NULL,
    effect_reservation_id VARCHAR NOT NULL,
    task_cid VARCHAR NOT NULL,
    attempt_id VARCHAR NOT NULL,
    fencing_epoch BIGINT NOT NULL,
    disposition VARCHAR NOT NULL,
    observation_ref VARCHAR NOT NULL,
    evidence_ref VARCHAR NOT NULL,
    observed_at VARCHAR NOT NULL
);

CREATE TABLE federation_audit_receipts (
    audit_receipt_id VARCHAR PRIMARY KEY,
    tenant_id VARCHAR NOT NULL,
    federation_id VARCHAR NOT NULL,
    subject_kind VARCHAR NOT NULL,
    subject_ref VARCHAR NOT NULL,
    operation VARCHAR NOT NULL,
    authorization_decision_id VARCHAR NOT NULL,
    policy_decision_id VARCHAR NOT NULL DEFAULT '',
    command_id VARCHAR NOT NULL DEFAULT '',
    event_id VARCHAR NOT NULL DEFAULT '',
    control_plane_generation BIGINT NOT NULL,
    fencing_epoch BIGINT NOT NULL,
    content_ref VARCHAR NOT NULL,
    recorded_at VARCHAR NOT NULL
);
CREATE INDEX federation_audit_receipts_subject_idx
    ON federation_audit_receipts(tenant_id, federation_id, subject_kind, subject_ref);

-- ---------------------------------------------------------------------------
-- Remaining normalized federation intent and scheduling identities
-- ---------------------------------------------------------------------------
-- These relations close the queryable CASF inventory without replacing the
-- existing task/goal/plan authorities in migration 0001.  They bind the
-- federation-specific revision, tenant, provenance, and content identities
-- that cannot safely live only in an opaque body.
CREATE TABLE programs (
    program_id VARCHAR PRIMARY KEY,
    tenant_id VARCHAR NOT NULL,
    current_revision BIGINT NOT NULL,
    objective_ref VARCHAR NOT NULL,
    policy_id VARCHAR NOT NULL,
    owner_id VARCHAR NOT NULL,
    content_ref VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    created_at VARCHAR NOT NULL,
    updated_at VARCHAR NOT NULL
);

CREATE TABLE program_revisions (
    program_revision_id VARCHAR PRIMARY KEY,
    program_id VARCHAR NOT NULL,
    tenant_id VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    objective_ref VARCHAR NOT NULL,
    policy_id VARCHAR NOT NULL,
    predecessor_revision_id VARCHAR NOT NULL DEFAULT '',
    content_ref VARCHAR NOT NULL,
    recorded_at VARCHAR NOT NULL
);
CREATE UNIQUE INDEX program_revisions_identity_uidx
    ON program_revisions(program_id, revision);

CREATE TABLE subgoals (
    subgoal_id VARCHAR PRIMARY KEY,
    tenant_id VARCHAR NOT NULL,
    federation_id VARCHAR NOT NULL,
    goal_cid VARCHAR NOT NULL,
    objective_ref VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    status VARCHAR NOT NULL,
    owner_id VARCHAR NOT NULL,
    content_ref VARCHAR NOT NULL,
    created_at VARCHAR NOT NULL,
    updated_at VARCHAR NOT NULL
);
CREATE INDEX subgoals_goal_idx
    ON subgoals(tenant_id, federation_id, goal_cid, revision);

CREATE TABLE plan_branches (
    plan_branch_id VARCHAR PRIMARY KEY,
    tenant_id VARCHAR NOT NULL,
    federation_id VARCHAR NOT NULL,
    plan_cid VARCHAR NOT NULL,
    plan_revision_id VARCHAR NOT NULL,
    parent_branch_id VARCHAR NOT NULL DEFAULT '',
    branch_kind VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    content_ref VARCHAR NOT NULL,
    created_at VARCHAR NOT NULL,
    updated_at VARCHAR NOT NULL
);

CREATE TABLE task_conflicts (
    task_conflict_id VARCHAR PRIMARY KEY,
    tenant_id VARCHAR NOT NULL,
    federation_id VARCHAR NOT NULL,
    left_task_cid VARCHAR NOT NULL,
    right_task_cid VARCHAR NOT NULL,
    conflict_kind VARCHAR NOT NULL,
    effect_class VARCHAR NOT NULL,
    evidence_ref VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    recorded_at VARCHAR NOT NULL
);
CREATE UNIQUE INDEX task_conflicts_pair_uidx
    ON task_conflicts(tenant_id, federation_id, left_task_cid, right_task_cid, revision);

CREATE TABLE task_resolutions (
    task_resolution_id VARCHAR PRIMARY KEY,
    tenant_id VARCHAR NOT NULL,
    federation_id VARCHAR NOT NULL,
    task_cid VARCHAR NOT NULL,
    conflict_id VARCHAR NOT NULL DEFAULT '',
    resolution_kind VARCHAR NOT NULL,
    predecessor_task_cid VARCHAR NOT NULL DEFAULT '',
    result_ref VARCHAR NOT NULL,
    evidence_ref VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    recorded_at VARCHAR NOT NULL
);
CREATE INDEX task_resolutions_task_idx
    ON task_resolutions(tenant_id, federation_id, task_cid, revision);

-- Base tasks remain canonical; this relation provides the tenant/federation
-- and exact repository-tree join needed before an unassigned task is visible
-- to a federation scheduler.
CREATE TABLE federation_task_bindings (
    federation_task_binding_id VARCHAR PRIMARY KEY,
    tenant_id VARCHAR NOT NULL,
    federation_id VARCHAR NOT NULL,
    task_cid VARCHAR NOT NULL,
    repository_id VARCHAR NOT NULL,
    tree_id VARCHAR NOT NULL,
    goal_cid VARCHAR NOT NULL,
    subgoal_id VARCHAR NOT NULL DEFAULT '',
    plan_revision_id VARCHAR NOT NULL,
    assignment_revision BIGINT NOT NULL,
    status VARCHAR NOT NULL,
    created_at VARCHAR NOT NULL,
    updated_at VARCHAR NOT NULL
);
CREATE UNIQUE INDEX federation_task_bindings_scope_uidx
    ON federation_task_bindings(tenant_id, federation_id, task_cid, assignment_revision);

CREATE TABLE process_births (
    process_birth_id VARCHAR PRIMARY KEY,
    tenant_id VARCHAR NOT NULL,
    federation_id VARCHAR NOT NULL DEFAULT '',
    supervisor_id VARCHAR NOT NULL DEFAULT '',
    subagent_id VARCHAR NOT NULL DEFAULT '',
    process_id BIGINT NOT NULL,
    start_marker VARCHAR NOT NULL,
    executable_ref VARCHAR NOT NULL,
    host_identity_ref VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    started_at VARCHAR NOT NULL,
    stopped_at VARCHAR
);

-- Runtime authority is separate from logical registration.  The exclusive
-- state owner records kernel-observed PID/start/boot evidence and binds it to
-- the admitted supervisor lease and fence before executable lifecycle states.
CREATE TABLE supervisor_runtime_leases (
    runtime_lease_id VARCHAR PRIMARY KEY,
    tenant_id VARCHAR NOT NULL,
    federation_id VARCHAR NOT NULL,
    supervisor_id VARCHAR NOT NULL,
    lease_id VARCHAR NOT NULL,
    process_birth_id VARCHAR NOT NULL,
    process_id BIGINT NOT NULL,
    process_start_time_ticks BIGINT NOT NULL,
    process_boot_id VARCHAR NOT NULL,
    process_parent_id BIGINT NOT NULL,
    issued_at VARCHAR NOT NULL,
    expires_at VARCHAR NOT NULL,
    revoked_at VARCHAR,
    fencing_epoch BIGINT NOT NULL,
    revision BIGINT NOT NULL,
    status VARCHAR NOT NULL,
    evidence_ref VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}'
);
CREATE UNIQUE INDEX supervisor_runtime_leases_revision_uidx
    ON supervisor_runtime_leases(
        tenant_id, federation_id, supervisor_id, fencing_epoch, revision
    );
CREATE INDEX supervisor_runtime_leases_current_idx
    ON supervisor_runtime_leases(
        tenant_id, federation_id, supervisor_id, lease_id, fencing_epoch,
        status, expires_at
    );

CREATE TABLE repository_trees (
    repository_tree_id VARCHAR PRIMARY KEY,
    repository_id VARCHAR NOT NULL,
    tree_id VARCHAR NOT NULL,
    semantic_state_root VARCHAR NOT NULL,
    semantic_owner_id VARCHAR NOT NULL,
    source_root VARCHAR NOT NULL,
    content_ref VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    status VARCHAR NOT NULL,
    freshness_state VARCHAR NOT NULL,
    recorded_at VARCHAR NOT NULL
);
CREATE UNIQUE INDEX repository_trees_identity_uidx
    ON repository_trees(repository_id, tree_id, revision);

CREATE TABLE fencing_epochs (
    fencing_epoch_id VARCHAR PRIMARY KEY,
    tenant_id VARCHAR NOT NULL,
    federation_id VARCHAR NOT NULL,
    subject_kind VARCHAR NOT NULL,
    subject_id VARCHAR NOT NULL,
    lease_id VARCHAR NOT NULL,
    epoch BIGINT NOT NULL,
    status VARCHAR NOT NULL,
    causation_event_id VARCHAR NOT NULL,
    recorded_at VARCHAR NOT NULL
);
CREATE UNIQUE INDEX fencing_epochs_subject_uidx
    ON fencing_epochs(tenant_id, federation_id, subject_kind, subject_id, epoch);

CREATE TABLE task_checkpoints (
    task_checkpoint_id VARCHAR PRIMARY KEY,
    tenant_id VARCHAR NOT NULL,
    federation_id VARCHAR NOT NULL,
    supervisor_id VARCHAR NOT NULL,
    subagent_id VARCHAR NOT NULL DEFAULT '',
    task_cid VARCHAR NOT NULL,
    attempt_id VARCHAR NOT NULL,
    lease_id VARCHAR NOT NULL,
    fencing_epoch BIGINT NOT NULL,
    checkpoint_ref VARCHAR NOT NULL,
    event_cursor BIGINT NOT NULL,
    revision BIGINT NOT NULL,
    recorded_at VARCHAR NOT NULL
);
CREATE INDEX task_checkpoints_attempt_idx
    ON task_checkpoints(task_cid, attempt_id, fencing_epoch, revision);

CREATE TABLE provider_reservations (
    provider_reservation_id VARCHAR PRIMARY KEY,
    tenant_id VARCHAR NOT NULL,
    federation_id VARCHAR NOT NULL,
    supervisor_id VARCHAR NOT NULL,
    subagent_id VARCHAR NOT NULL DEFAULT '',
    task_cid VARCHAR NOT NULL,
    provider_id VARCHAR NOT NULL,
    model_id VARCHAR NOT NULL,
    resource_class VARCHAR NOT NULL,
    capacity_units BIGINT NOT NULL,
    token_ceiling BIGINT NOT NULL,
    idempotency_key VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    reserved_at VARCHAR NOT NULL,
    expires_at VARCHAR NOT NULL
);
CREATE UNIQUE INDEX provider_reservations_idempotency_uidx
    ON provider_reservations(tenant_id, federation_id, idempotency_key);

-- ---------------------------------------------------------------------------
-- Semantic records remain opaque, source-root-bound projections
-- ---------------------------------------------------------------------------
CREATE TABLE semantic_effect_references (
    semantic_effect_id VARCHAR PRIMARY KEY,
    tenant_id VARCHAR NOT NULL,
    federation_id VARCHAR NOT NULL,
    repository_id VARCHAR NOT NULL,
    tree_id VARCHAR NOT NULL,
    symbol_ref VARCHAR NOT NULL,
    effect_kind VARCHAR NOT NULL,
    owner_id VARCHAR NOT NULL,
    source_root VARCHAR NOT NULL,
    provenance_ref VARCHAR NOT NULL,
    content_ref VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    status VARCHAR NOT NULL,
    freshness_state VARCHAR NOT NULL,
    recorded_at VARCHAR NOT NULL
);

CREATE TABLE semantic_relationship_references (
    semantic_relationship_id VARCHAR PRIMARY KEY,
    tenant_id VARCHAR NOT NULL,
    federation_id VARCHAR NOT NULL,
    repository_id VARCHAR NOT NULL,
    tree_id VARCHAR NOT NULL,
    source_ref VARCHAR NOT NULL,
    target_ref VARCHAR NOT NULL,
    relationship_kind VARCHAR NOT NULL,
    owner_id VARCHAR NOT NULL,
    source_root VARCHAR NOT NULL,
    provenance_ref VARCHAR NOT NULL,
    content_ref VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    status VARCHAR NOT NULL,
    freshness_state VARCHAR NOT NULL,
    recorded_at VARCHAR NOT NULL
);

CREATE TABLE semantic_capsule_dependencies (
    capsule_dependency_id VARCHAR PRIMARY KEY,
    tenant_id VARCHAR NOT NULL,
    federation_id VARCHAR NOT NULL,
    repository_id VARCHAR NOT NULL,
    tree_id VARCHAR NOT NULL,
    capsule_ref VARCHAR NOT NULL,
    dependency_capsule_ref VARCHAR NOT NULL,
    dependency_kind VARCHAR NOT NULL,
    owner_id VARCHAR NOT NULL,
    source_root VARCHAR NOT NULL,
    provenance_ref VARCHAR NOT NULL,
    content_ref VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    status VARCHAR NOT NULL,
    freshness_state VARCHAR NOT NULL,
    recorded_at VARCHAR NOT NULL
);

CREATE TABLE semantic_contract_references (
    semantic_contract_id VARCHAR PRIMARY KEY,
    tenant_id VARCHAR NOT NULL,
    federation_id VARCHAR NOT NULL,
    repository_id VARCHAR NOT NULL,
    tree_id VARCHAR NOT NULL,
    contract_ref VARCHAR NOT NULL,
    contract_kind VARCHAR NOT NULL,
    owner_id VARCHAR NOT NULL,
    source_root VARCHAR NOT NULL,
    provenance_ref VARCHAR NOT NULL,
    content_ref VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    status VARCHAR NOT NULL,
    freshness_state VARCHAR NOT NULL,
    recorded_at VARCHAR NOT NULL
);

CREATE TABLE environment_binding_references (
    environment_binding_id VARCHAR PRIMARY KEY,
    tenant_id VARCHAR NOT NULL,
    federation_id VARCHAR NOT NULL,
    repository_id VARCHAR NOT NULL,
    tree_id VARCHAR NOT NULL,
    environment_ref VARCHAR NOT NULL,
    binding_kind VARCHAR NOT NULL,
    owner_id VARCHAR NOT NULL,
    source_root VARCHAR NOT NULL,
    provenance_ref VARCHAR NOT NULL,
    content_ref VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    status VARCHAR NOT NULL,
    freshness_state VARCHAR NOT NULL,
    recorded_at VARCHAR NOT NULL
);

-- ---------------------------------------------------------------------------
-- Proof, test, validation, and retrieval projections
-- ---------------------------------------------------------------------------
CREATE TABLE proof_units (
    proof_unit_id VARCHAR PRIMARY KEY,
    tenant_id VARCHAR NOT NULL,
    federation_id VARCHAR NOT NULL,
    repository_id VARCHAR NOT NULL,
    tree_id VARCHAR NOT NULL,
    task_cid VARCHAR NOT NULL,
    obligation_ref VARCHAR NOT NULL,
    proof_kind VARCHAR NOT NULL,
    owner_id VARCHAR NOT NULL,
    source_root VARCHAR NOT NULL,
    provenance_ref VARCHAR NOT NULL,
    content_ref VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    status VARCHAR NOT NULL,
    freshness_state VARCHAR NOT NULL,
    recorded_at VARCHAR NOT NULL
);

CREATE TABLE proof_receipts (
    proof_receipt_id VARCHAR PRIMARY KEY,
    tenant_id VARCHAR NOT NULL,
    federation_id VARCHAR NOT NULL,
    proof_unit_id VARCHAR NOT NULL,
    attempt_id VARCHAR NOT NULL,
    repository_id VARCHAR NOT NULL,
    tree_id VARCHAR NOT NULL,
    outcome VARCHAR NOT NULL,
    evidence_ref VARCHAR NOT NULL,
    owner_id VARCHAR NOT NULL,
    source_root VARCHAR NOT NULL,
    provenance_ref VARCHAR NOT NULL,
    content_ref VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    status VARCHAR NOT NULL,
    freshness_state VARCHAR NOT NULL,
    recorded_at VARCHAR NOT NULL
);

CREATE TABLE proof_cache_entries (
    proof_cache_entry_id VARCHAR PRIMARY KEY,
    tenant_id VARCHAR NOT NULL,
    repository_id VARCHAR NOT NULL,
    tree_id VARCHAR NOT NULL,
    obligation_ref VARCHAR NOT NULL,
    dependency_root VARCHAR NOT NULL,
    policy_ref VARCHAR NOT NULL,
    provider_model_ref VARCHAR NOT NULL,
    owner_id VARCHAR NOT NULL,
    source_root VARCHAR NOT NULL,
    provenance_ref VARCHAR NOT NULL,
    content_ref VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    status VARCHAR NOT NULL,
    freshness_state VARCHAR NOT NULL,
    recorded_at VARCHAR NOT NULL,
    expires_at VARCHAR NOT NULL
);
CREATE UNIQUE INDEX proof_cache_entries_identity_uidx
    ON proof_cache_entries(tenant_id, repository_id, tree_id, obligation_ref,
                           dependency_root, policy_ref, provider_model_ref);

CREATE TABLE proof_seals (
    proof_seal_id VARCHAR PRIMARY KEY,
    tenant_id VARCHAR NOT NULL,
    federation_id VARCHAR NOT NULL,
    proof_unit_id VARCHAR NOT NULL,
    proof_receipt_id VARCHAR NOT NULL,
    repository_id VARCHAR NOT NULL,
    tree_id VARCHAR NOT NULL,
    policy_ref VARCHAR NOT NULL,
    owner_id VARCHAR NOT NULL,
    source_root VARCHAR NOT NULL,
    provenance_ref VARCHAR NOT NULL,
    content_ref VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    status VARCHAR NOT NULL,
    freshness_state VARCHAR NOT NULL,
    recorded_at VARCHAR NOT NULL
);

CREATE TABLE test_selections (
    test_selection_id VARCHAR PRIMARY KEY,
    tenant_id VARCHAR NOT NULL,
    federation_id VARCHAR NOT NULL,
    repository_id VARCHAR NOT NULL,
    tree_id VARCHAR NOT NULL,
    task_cid VARCHAR NOT NULL,
    selection_root VARCHAR NOT NULL,
    dependency_root VARCHAR NOT NULL,
    owner_id VARCHAR NOT NULL,
    source_root VARCHAR NOT NULL,
    provenance_ref VARCHAR NOT NULL,
    content_ref VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    status VARCHAR NOT NULL,
    freshness_state VARCHAR NOT NULL,
    recorded_at VARCHAR NOT NULL
);

CREATE TABLE test_attempts (
    test_attempt_id VARCHAR PRIMARY KEY,
    tenant_id VARCHAR NOT NULL,
    federation_id VARCHAR NOT NULL,
    test_selection_id VARCHAR NOT NULL,
    task_cid VARCHAR NOT NULL,
    attempt_id VARCHAR NOT NULL,
    fencing_epoch BIGINT NOT NULL,
    status VARCHAR NOT NULL,
    started_at VARCHAR NOT NULL,
    finished_at VARCHAR
);

CREATE TABLE test_receipts (
    test_receipt_id VARCHAR PRIMARY KEY,
    tenant_id VARCHAR NOT NULL,
    federation_id VARCHAR NOT NULL,
    test_attempt_id VARCHAR NOT NULL,
    repository_id VARCHAR NOT NULL,
    tree_id VARCHAR NOT NULL,
    outcome VARCHAR NOT NULL,
    evidence_ref VARCHAR NOT NULL,
    owner_id VARCHAR NOT NULL,
    source_root VARCHAR NOT NULL,
    provenance_ref VARCHAR NOT NULL,
    content_ref VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    status VARCHAR NOT NULL,
    freshness_state VARCHAR NOT NULL,
    recorded_at VARCHAR NOT NULL
);

CREATE TABLE validation_plans (
    validation_plan_id VARCHAR PRIMARY KEY,
    tenant_id VARCHAR NOT NULL,
    federation_id VARCHAR NOT NULL,
    task_cid VARCHAR NOT NULL,
    repository_id VARCHAR NOT NULL,
    tree_id VARCHAR NOT NULL,
    required_proof_root VARCHAR NOT NULL,
    required_test_root VARCHAR NOT NULL,
    policy_ref VARCHAR NOT NULL,
    owner_id VARCHAR NOT NULL,
    source_root VARCHAR NOT NULL,
    provenance_ref VARCHAR NOT NULL,
    content_ref VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    status VARCHAR NOT NULL,
    freshness_state VARCHAR NOT NULL,
    recorded_at VARCHAR NOT NULL
);

CREATE TABLE documents (
    document_id VARCHAR PRIMARY KEY,
    tenant_id VARCHAR NOT NULL,
    repository_id VARCHAR NOT NULL,
    tree_id VARCHAR NOT NULL,
    document_kind VARCHAR NOT NULL,
    source_cid VARCHAR NOT NULL,
    privacy_scope_ref VARCHAR NOT NULL,
    owner_id VARCHAR NOT NULL,
    source_root VARCHAR NOT NULL,
    provenance_ref VARCHAR NOT NULL,
    content_ref VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    status VARCHAR NOT NULL,
    freshness_state VARCHAR NOT NULL,
    recorded_at VARCHAR NOT NULL
);

CREATE TABLE document_chunks (
    document_chunk_id VARCHAR PRIMARY KEY,
    document_id VARCHAR NOT NULL,
    repository_id VARCHAR NOT NULL,
    tree_id VARCHAR NOT NULL,
    ordinal BIGINT NOT NULL,
    source_cid VARCHAR NOT NULL,
    owner_id VARCHAR NOT NULL,
    source_root VARCHAR NOT NULL,
    provenance_ref VARCHAR NOT NULL,
    content_ref VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    status VARCHAR NOT NULL,
    freshness_state VARCHAR NOT NULL,
    token_count BIGINT NOT NULL,
    recorded_at VARCHAR NOT NULL
);
CREATE UNIQUE INDEX document_chunks_ordinal_uidx
    ON document_chunks(document_id, ordinal);

CREATE TABLE bm25_terms (
    bm25_term_id VARCHAR PRIMARY KEY,
    retrieval_index_id VARCHAR NOT NULL,
    index_revision BIGINT NOT NULL,
    partition_id VARCHAR NOT NULL,
    term VARCHAR NOT NULL,
    document_frequency BIGINT NOT NULL,
    inverse_document_frequency_micros BIGINT NOT NULL,
    source_root VARCHAR NOT NULL,
    recorded_at VARCHAR NOT NULL
);
CREATE UNIQUE INDEX bm25_terms_identity_uidx
    ON bm25_terms(retrieval_index_id, index_revision, partition_id, term);

CREATE TABLE bm25_postings (
    bm25_posting_id VARCHAR PRIMARY KEY,
    bm25_term_id VARCHAR NOT NULL,
    document_id VARCHAR NOT NULL,
    document_chunk_id VARCHAR NOT NULL,
    term_frequency BIGINT NOT NULL,
    field_length BIGINT NOT NULL,
    source_root VARCHAR NOT NULL,
    recorded_at VARCHAR NOT NULL
);
CREATE INDEX bm25_postings_term_idx
    ON bm25_postings(bm25_term_id, document_id, document_chunk_id);

-- Vector bodies remain in managed content-addressed storage.  The control
-- plane retains queryable identity, dimensionality, partition, and provenance.
CREATE TABLE vectors (
    vector_id VARCHAR PRIMARY KEY,
    retrieval_index_id VARCHAR NOT NULL,
    index_revision BIGINT NOT NULL,
    partition_id VARCHAR NOT NULL,
    document_id VARCHAR NOT NULL,
    document_chunk_id VARCHAR NOT NULL,
    embedding_model_ref VARCHAR NOT NULL,
    dimensions BIGINT NOT NULL,
    vector_ref VARCHAR NOT NULL,
    source_root VARCHAR NOT NULL,
    recorded_at VARCHAR NOT NULL
);

CREATE TABLE vector_metadata (
    vector_metadata_id VARCHAR PRIMARY KEY,
    vector_id VARCHAR NOT NULL,
    metadata_kind VARCHAR NOT NULL,
    metadata_ref VARCHAR NOT NULL,
    privacy_scope_ref VARCHAR NOT NULL,
    owner_id VARCHAR NOT NULL,
    source_root VARCHAR NOT NULL,
    provenance_ref VARCHAR NOT NULL,
    content_ref VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    status VARCHAR NOT NULL,
    freshness_state VARCHAR NOT NULL,
    recorded_at VARCHAR NOT NULL
);

CREATE TABLE knowledge_graph_nodes (
    knowledge_graph_node_id VARCHAR PRIMARY KEY,
    retrieval_index_id VARCHAR NOT NULL,
    index_revision BIGINT NOT NULL,
    repository_id VARCHAR NOT NULL,
    tree_id VARCHAR NOT NULL,
    node_kind VARCHAR NOT NULL,
    subject_ref VARCHAR NOT NULL,
    authority_disposition VARCHAR NOT NULL,
    owner_id VARCHAR NOT NULL,
    source_root VARCHAR NOT NULL,
    provenance_ref VARCHAR NOT NULL,
    content_ref VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    status VARCHAR NOT NULL,
    freshness_state VARCHAR NOT NULL,
    recorded_at VARCHAR NOT NULL
);

CREATE TABLE knowledge_graph_edges (
    knowledge_graph_edge_id VARCHAR PRIMARY KEY,
    retrieval_index_id VARCHAR NOT NULL,
    index_revision BIGINT NOT NULL,
    source_node_id VARCHAR NOT NULL,
    target_node_id VARCHAR NOT NULL,
    relationship_kind VARCHAR NOT NULL,
    evidence_ref VARCHAR NOT NULL,
    authority_disposition VARCHAR NOT NULL,
    owner_id VARCHAR NOT NULL,
    source_root VARCHAR NOT NULL,
    provenance_ref VARCHAR NOT NULL,
    content_ref VARCHAR NOT NULL,
    revision BIGINT NOT NULL,
    status VARCHAR NOT NULL,
    freshness_state VARCHAR NOT NULL,
    recorded_at VARCHAR NOT NULL
);
CREATE INDEX knowledge_graph_edges_traversal_idx
    ON knowledge_graph_edges(retrieval_index_id, index_revision,
                             source_node_id, relationship_kind, target_node_id);

-- Immutable schema-extension identity.  ControlPlaneSchema@1 remains present
-- and continues to describe the historical foundation migration.
INSERT INTO schema_contracts (
    contract_id, interface_name, domain_name, schema_revision,
    payload_schema, description, created_at
) VALUES (
    'contract:CausalEventFederationSchemaExtension@1',
    'CausalEventFederationSchemaExtension@1',
    'control',
    2,
    'ipfs_accelerate_py/agent-supervisor/causal-event-federation-schema-extension@1',
    'Operational federation coordination and opaque semantic/proof/retrieval references; ipfs_datasets_py retains semantic meaning authority',
    '1970-01-01T00:00:00Z'
);
