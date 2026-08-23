"""Normalized control-plane schema inventory and install helpers (DQP-005).

Interface: ``ControlPlaneSchema@1``

Owns the physical table/view inventory for ``control.duckdb`` domain SQL
(``sql/0001_control_plane.sql``), join-critical identity columns, existing
task-CID / lease compatibility contracts, and the optional supervisor DuckDB
dependency pin. Schema DDL is applied only through the checksum-bound migration
catalog; this module never invents runtime ad-hoc tables.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, Final

from .control_plane_migrations import (
    ControlPlaneMigration,
    ControlPlaneMigrationRunner,
    MigrationCatalog,
    MigrationRunReport,
    compute_schema_fingerprint,
    duckdb_available,
    load_default_catalog,
)
from .duckdb_state import open_duckdb_connection

# ---------------------------------------------------------------------------
# Interface / version identities
# ---------------------------------------------------------------------------

CONTROL_PLANE_SCHEMA_INTERFACE: Final = "ControlPlaneSchema@1"
CONTROL_PLANE_SCHEMA_SCHEMA: Final = "ipfs_accelerate_py/agent-supervisor/control-plane-schema@1"
CONTROL_PLANE_SCHEMA_VERSION: Final[int] = 1
CONTROL_PLANE_SCHEMA_REVISION: Final[int] = 1
CONTROL_PLANE_MIGRATION_ID: Final = "0001_control_plane"
CONTROL_PLANE_MIGRATION_VERSION: Final[int] = 1
CONTROL_PLANE_SQL_FILENAME: Final = "0001_control_plane.sql"

# Additive extension identity.  The ControlPlaneSchema@1 constants above are
# intentionally immutable: the datasets-authoritative operational profile is
# checksum-derived from 0001 and must not silently acquire federation tables.
CAUSAL_EVENT_FEDERATION_SCHEMA_EXTENSION_INTERFACE: Final = "CausalEventFederationSchemaExtension@1"
CAUSAL_EVENT_FEDERATION_SCHEMA_EXTENSION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/causal-event-federation-schema-extension@1"
)
CAUSAL_EVENT_FEDERATION_SCHEMA_REVISION: Final[int] = 2
CAUSAL_EVENT_FEDERATION_MIGRATION_ID: Final = "0002_causal_event_federation_core"
CAUSAL_EVENT_FEDERATION_MIGRATION_VERSION: Final[int] = 2
CAUSAL_EVENT_FEDERATION_SQL_FILENAME: Final = "0002_causal_event_federation_core.sql"

# A constrained schema profile for deployments where ipfs_datasets_py is the
# sole semantic-truth authority.  This is deliberately a profile of the
# existing control-plane schema and migration machinery, not a second task or
# plan store.
DATASETS_AUTHORITATIVE_OPERATIONAL_PROFILE_ID: Final = (
    "datasets-authoritative-operational-control-plane@1"
)
DATASETS_AUTHORITATIVE_OPERATIONAL_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/datasets-authoritative-operational-control-plane@1"
)
DATASETS_AUTHORITATIVE_OPERATIONAL_MIGRATION_ID: Final = (
    "0001_datasets_authoritative_operational_control_plane"
)
DATASETS_AUTHORITATIVE_OPERATIONAL_MIGRATION_VERSION: Final[int] = 1

# Optional supervisor service dependency profile (pyproject extra).
SUPERVISOR_OPTIONAL_EXTRA: Final = "agent-supervisor"
PINNED_DUCKDB_VERSION_SPEC: Final = "duckdb>=1.5.0,<1.6.0"
PINNED_DUCKDB_MAJOR: Final = 1
PINNED_DUCKDB_MINOR: Final = 5
PINNED_DUCKDB_VERSION_PREFIX: Final = "1.5"
PINNED_QUACK_EXTENSION: Final = "quack"
PINNED_QUACK_EXTENSION_API: Final = "quack@1"
PINNED_PROFILE_ID: Final = "agent-supervisor-duckdb-quack-1.5"

SCHEMA_DOMAINS: Final[tuple[str, ...]] = (
    "meta",
    "intent",
    "schedule",
    "runtime",
    "git",
    "code",
    "evidence",
    "cache",
    "control",
    "improve",
)

# Bookkeeping tables installed by the migration runner (not domain SQL).
BOOKKEEPING_TABLES: Final[tuple[str, ...]] = (
    "control_plane_metadata",
    "schema_migrations",
    "schema_migration_attempts",
)

# Domain table inventory (main schema). Order is documentation-only.
_DOMAIN_TABLES: Final[dict[str, tuple[str, ...]]] = {
    "meta": (
        "schema_contracts",
        "store_generations",
    ),
    "intent": (
        "objectives",
        "objective_revisions",
        "goals",
        "goal_edges",
        "plans",
        "plan_revisions",
        "planning_decisions",
        "plan_candidates",
        "tasks",
        "task_revisions",
        "task_dependencies",
        "task_outputs",
        "task_acceptance",
        "task_validations",
        "artifacts",
    ),
    "schedule": (
        "task_assignments",
        "task_blocks",
        "refill_epochs",
        "findings",
        "finding_dispositions",
    ),
    "runtime": (
        "supervisor_instances",
        "daemon_instances",
        "daemon_sessions",
        "heartbeats",
        "health_samples",
        "stall_detections",
        "restart_decisions",
        "task_attempts",
        "attempt_phases",
        "task_claims",
        "provider_invocations",
        "validation_runs",
        "validation_results",
        "merge_attempts",
        "recovery_actions",
        "idempotency_records",
        "effect_claims",
        "completion_receipts",
    ),
    "git": (
        "repositories",
        "repository_revisions",
        "submodule_edges",
        "worktrees",
        "worktree_snapshots",
        "worktree_paths",
        "dirty_overlays",
        "branches",
        "git_refs",
        "merge_bases",
        "merge_queue_entries",
        "resource_claims",
        "path_claims",
        "leases",
        "lease_events",
        "token_history",
    ),
    "code": (
        "source_snapshots",
        "source_files",
        "file_versions",
        "parse_runs",
        "symbols",
        "symbol_versions",
        "ast_nodes",
        "ast_edges",
        "imports",
        "calls",
        "references",
        "definitions",
        "type_relations",
        "mutations",
        "mutation_files",
        "mutation_hunks",
        "ast_mutations",
        "impact_edges",
        "impact_closures",
        "repair_candidates",
        "repair_applications",
    ),
    "evidence": (
        "proof_obligations",
        "proof_attempts",
        "counterexamples",
        "evidence_nodes",
    ),
    "cache": (
        "context_manifests",
        "context_members",
        "context_deltas",
        "prompt_templates",
        "prompt_instances",
        "prompt_inputs",
        "provider_calls",
        "provider_responses",
        "failure_signatures",
        "decision_cache_entries",
        "replay_suppressions",
    ),
    "control": (
        "state_servers",
        "server_epochs",
        "client_sessions",
        "capability_snapshots",
        "credentials",
        "authorization_roles",
        "authorization_grants",
        "backup_snapshots",
        "restore_receipts",
        "maintenance_leases",
        "domain_events",
        "structured_logs",
    ),
    "improve": (
        "metrics",
        "metric_samples",
        "budget_reservations",
        "budget_consumption",
        "quack_query_telemetry",
        "churn_metrics",
    ),
}

DOMAIN_TABLES: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {key: tuple(value) for key, value in _DOMAIN_TABLES.items()}
)

# Migration-2 tables are a separately versioned extension inventory.  Keeping
# them out of DOMAIN_TABLES is deliberate: ControlPlaneSchema@1 and the
# datasets-authoritative 0001-derived profile retain their exact historical
# meaning, while full control-plane installation advances to migration 2.
CAUSAL_EVENT_FEDERATION_TABLES: Final[tuple[str, ...]] = (
    "domain_event_causal_parents",
    "domain_event_changed_facts",
    "federations",
    "federation_revisions",
    "federation_policies",
    "federation_plans",
    "federation_receipts",
    "programs",
    "program_revisions",
    "subgoals",
    "plan_branches",
    "task_conflicts",
    "task_resolutions",
    "federation_task_bindings",
    "process_births",
    "supervisor_runtime_leases",
    "repository_trees",
    "fencing_epochs",
    "task_checkpoints",
    "provider_reservations",
    "supervisor_definitions",
    "supervisor_assignments",
    "supervisor_capabilities",
    "supervisor_checkpoints",
    "supervisor_receipts",
    "subagent_definitions",
    "subagent_instances",
    "subagent_assignments",
    "subagent_capabilities",
    "subagent_execution_slots",
    "subagent_slot_ledger",
    "subagent_outcomes",
    "supervisor_shards",
    "shard_boundaries",
    "shard_assignments",
    "shard_revisions",
    "shard_rebalance_plans",
    "shard_rebalance_receipts",
    "federation_budgets",
    "supervisor_budgets",
    "agent_budgets",
    "token_budgets",
    "resource_budgets",
    "budget_ledger",
    "federation_admission_budget_reservations",
    "federation_admission_budget_dimensions",
    "stream_sequence_heads",
    "global_sequence_head",
    "transactional_outbox",
    "outbox_routing_dispositions",
    "outbox_routing_disposition_events",
    "event_subscriptions",
    "event_subscription_selectors",
    "consumer_cursors",
    "supervisor_inbox",
    "delivery_attempts",
    "dead_letters",
    "event_delivery_queue",
    "event_coalescing",
    "event_coalescing_coverage",
    "event_coalescing_inputs",
    "event_acknowledgements",
    "causal_nodes",
    "causal_edges",
    "causal_evidence",
    "causal_abstraction_maps",
    "causal_abstraction_validations",
    "causal_intervention_tests",
    "causal_slices",
    "causal_frontiers",
    "causal_frontier_members",
    "causal_invalidations",
    "semantic_root_references",
    "semantic_capsule_references",
    "semantic_effect_references",
    "semantic_relationship_references",
    "semantic_capsule_dependencies",
    "semantic_contract_references",
    "environment_binding_references",
    "world_snapshot_references",
    "proof_reference_projections",
    "proof_units",
    "proof_receipts",
    "proof_cache_entries",
    "proof_seals",
    "test_reference_projections",
    "test_selections",
    "test_attempts",
    "test_receipts",
    "validation_plans",
    "retrieval_indexes",
    "retrieval_index_partitions",
    "retrieval_receipts",
    "retrieval_nominations",
    "documents",
    "document_chunks",
    "bm25_terms",
    "bm25_postings",
    "vectors",
    "vector_metadata",
    "knowledge_graph_nodes",
    "knowledge_graph_edges",
    "federation_commands",
    "federation_authorization_decisions",
    "federation_policy_decisions",
    "federation_confirmation_receipts",
    "federation_effect_reservations",
    "federation_effect_observations",
    "federation_audit_receipts",
)

CAUSAL_EVENT_FEDERATION_REFERENCE_TABLES: Final[tuple[str, ...]] = (
    "semantic_root_references",
    "semantic_capsule_references",
    "semantic_effect_references",
    "semantic_relationship_references",
    "semantic_capsule_dependencies",
    "semantic_contract_references",
    "environment_binding_references",
    "world_snapshot_references",
    "proof_reference_projections",
    "proof_units",
    "proof_receipts",
    "proof_cache_entries",
    "proof_seals",
    "test_reference_projections",
    "test_selections",
    "test_receipts",
    "validation_plans",
    "retrieval_indexes",
    "retrieval_index_partitions",
    "retrieval_receipts",
    "retrieval_nominations",
    "documents",
    "document_chunks",
    "vector_metadata",
    "knowledge_graph_nodes",
    "knowledge_graph_edges",
)

# Closed audit map for every conceptual representation required by the CASF
# Section 9 schema contract.  An alias names the existing canonical normalized
# relation; it does not create a second authority.  Semantic entries point only
# to source-root-bound references/projections owned by ipfs_datasets_py.
CAUSAL_EVENT_FEDERATION_SECTION_9_RELATIONS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "meta.store_generations": "store_generations",
        "meta.schema_migrations": "schema_migrations",
        "meta.federations": "federations",
        "meta.supervisors": "supervisor_instances",
        "meta.subagents": "subagent_instances",
        "meta.sessions": "client_sessions",
        "meta.process_births": "process_births",
        "meta.repository_identities": "repositories",
        "meta.tree_identities": "repository_trees",
        "meta.policy_identities": "federation_policies",
        "meta.capability_snapshots": "capability_snapshots",
        "intent.programs": "programs",
        "intent.program_revisions": "program_revisions",
        "intent.objectives": "objectives",
        "intent.objective_revisions": "objective_revisions",
        "intent.goals": "goals",
        "intent.subgoals": "subgoals",
        "intent.goal_edges": "goal_edges",
        "intent.formal_plans": "plans",
        "intent.plan_revisions": "plan_revisions",
        "intent.plan_branches": "plan_branches",
        "intent.acceptance_criteria": "task_acceptance",
        "scheduling.tasks": "tasks",
        "scheduling.task_revisions": "task_revisions",
        "scheduling.task_dependencies": "task_dependencies",
        "scheduling.task_conflicts": "task_conflicts",
        "scheduling.task_claims": "task_claims",
        "scheduling.task_resolutions": "task_resolutions",
        "scheduling.leases": "leases",
        "scheduling.fencing_epochs": "fencing_epochs",
        "scheduling.attempts": "task_attempts",
        "scheduling.checkpoints": "task_checkpoints",
        "scheduling.resource_reservations": "budget_reservations",
        "scheduling.federation_admission_reservations": (
            "federation_admission_budget_reservations"
        ),
        "scheduling.federation_admission_reservation_dimensions": (
            "federation_admission_budget_dimensions"
        ),
        "scheduling.provider_reservations": "provider_reservations",
        "scheduling.merge_queue_entries": "merge_queue_entries",
        "scheduling.federation_task_scope": "federation_task_bindings",
        "semantic.repositories": "repositories",
        "semantic.trees": "repository_trees",
        "semantic.files": "source_files",
        "semantic.symbols": "symbols",
        "semantic.ast_nodes": "ast_nodes",
        "semantic.ast_edges": "ast_edges",
        "semantic.imports": "imports",
        "semantic.calls": "calls",
        "semantic.effects": "semantic_effect_references",
        "semantic.relationships": "semantic_relationship_references",
        "semantic.symbol_versions": "symbol_versions",
        "semantic.capsules": "semantic_capsule_references",
        "semantic.capsule_dependencies": "semantic_capsule_dependencies",
        "semantic.contracts": "semantic_contract_references",
        "semantic.proof_obligations": "proof_obligations",
        "semantic.environment_bindings": "environment_binding_references",
        "semantic.semantic_roots": "semantic_root_references",
        "semantic.world_snapshots": "world_snapshot_references",
        "proof.proof_units": "proof_units",
        "proof.proof_obligations": "proof_obligations",
        "proof.proof_attempts": "proof_attempts",
        "proof.proof_receipts": "proof_receipts",
        "proof.proof_cache_entries": "proof_cache_entries",
        "proof.proof_seals": "proof_seals",
        "proof.test_selections": "test_selections",
        "proof.test_attempts": "test_attempts",
        "proof.test_receipts": "test_receipts",
        "proof.validation_plans": "validation_plans",
        "proof.validation_results": "validation_results",
        "proof.counterexamples": "counterexamples",
        "proof.adversarial_findings": "findings",
        "retrieval.documents": "documents",
        "retrieval.document_chunks": "document_chunks",
        "retrieval.bm25_terms": "bm25_terms",
        "retrieval.bm25_postings": "bm25_postings",
        "retrieval.vectors": "vectors",
        "retrieval.vector_metadata": "vector_metadata",
        "retrieval.knowledge_graph_nodes": "knowledge_graph_nodes",
        "retrieval.knowledge_graph_edges": "knowledge_graph_edges",
        "retrieval.retrieval_receipts": "retrieval_receipts",
        "retrieval.index_roots": "retrieval_indexes",
        "retrieval.index_revisions": "retrieval_indexes",
        "causal.nodes": "causal_nodes",
        "causal.edges": "causal_edges",
        "causal.evidence": "causal_evidence",
        "causal.abstraction_maps": "causal_abstraction_maps",
        "causal.abstraction_validations": "causal_abstraction_validations",
        "causal.intervention_tests": "causal_intervention_tests",
        "causal.slices": "causal_slices",
        "causal.frontiers": "causal_frontiers",
        "causal.invalidations": "causal_invalidations",
        "events.domain_events": "domain_events",
        "events.transactional_outbox": "transactional_outbox",
        "events.supervisor_inbox": "supervisor_inbox",
        "events.subscriptions": "event_subscriptions",
        "events.consumer_cursors": "consumer_cursors",
        "events.dead_letters": "dead_letters",
        "events.delivery_attempts": "delivery_attempts",
        "events.coalescing": "event_coalescing",
        "events.acknowledgements": "event_acknowledgements",
        "control.commands": "federation_commands",
        "control.idempotency_records": "idempotency_records",
        "control.authorization_decisions": "federation_authorization_decisions",
        "control.policy_decisions": "federation_policy_decisions",
        "control.confirmation_receipts": "federation_confirmation_receipts",
        "control.effect_reservations": "federation_effect_reservations",
        "control.effect_observations": "federation_effect_observations",
        "control.audit_receipts": "federation_audit_receipts",
    }
)

# These relations are owned by ipfs_datasets_py semantic truth and must never
# be created in a datasets-authoritative accelerator control plane.  Keeping a
# closed, exported deny-list makes both installation and external audit fail
# closed if a full ControlPlaneSchema@1 database is supplied accidentally.
DATASETS_SEMANTIC_TRUTH_RELATIONS: Final[frozenset[str]] = frozenset(
    {
        "source_snapshots",
        "source_files",
        "file_versions",
        "parse_runs",
        "symbols",
        "symbol_versions",
        "ast_nodes",
        "ast_edges",
        "imports",
        "calls",
        "references",
        "definitions",
        "type_relations",
        "mutations",
        "mutation_files",
        "mutation_hunks",
        "ast_mutations",
        "impact_edges",
        "impact_closures",
        "repair_candidates",
        "repair_applications",
        "proof_obligations",
        "proof_attempts",
        "counterexamples",
    }
)

DATASETS_AUTHORITATIVE_OPERATIONAL_DOMAINS: Final[tuple[str, ...]] = tuple(
    domain for domain in SCHEMA_DOMAINS if domain not in {"code", "evidence"}
)

# ``evidence_nodes`` is retained solely as the existing IntentRepository's
# content-addressed operational completion/result-receipt projection.  It does
# not store or assert semantic proof authority.
DATASETS_AUTHORITATIVE_OPERATIONAL_EVIDENCE_TABLES: Final[tuple[str, ...]] = ("evidence_nodes",)

DATASETS_AUTHORITATIVE_OPERATIONAL_TABLES: Final[tuple[str, ...]] = (
    tuple(
        table
        for domain in DATASETS_AUTHORITATIVE_OPERATIONAL_DOMAINS
        for table in DOMAIN_TABLES[domain]
    )
    + DATASETS_AUTHORITATIVE_OPERATIONAL_EVIDENCE_TABLES
)

DIAGNOSTIC_VIEWS: Final[tuple[str, ...]] = (
    "ready_task_context_v1",
    "diagnostic_schema_inventory_v1",
    "diagnostic_lease_surface_v1",
)

# Existing DuckDB task-source / lease-coordinator identity contracts.
TASK_IDENTITY_COLUMNS: Final[tuple[str, ...]] = (
    "task_cid",
    "task_alias",
    "goal_cid",
    "ordinal",
    "status",
    "revision",
)

LEASE_IDENTITY_COLUMNS: Final[tuple[str, ...]] = (
    "task_cid",
    "claim_cid",
    "resolution_cid",
    "claimant_did",
    "logical_epoch",
    "fencing_token",
    "expires_at_ms",
    "attempt",
    "state",
    "started_at_ms",
    "release_reason",
    "retry_not_before_ms",
)

# Columns that must exist as real columns for joins/claims/auth — never JSON-only.
JOIN_CRITICAL_IDENTITIES: Final[tuple[tuple[str, str], ...]] = (
    ("tasks", "task_cid"),
    ("tasks", "goal_cid"),
    ("task_dependencies", "task_cid"),
    ("task_dependencies", "dependency_task_cid"),
    ("leases", "task_cid"),
    ("leases", "claim_cid"),
    ("leases", "claimant_did"),
    ("leases", "fencing_token"),
    ("lease_events", "task_cid"),
    ("token_history", "task_cid"),
    ("heartbeats", "task_cid"),
    ("task_claims", "task_cid"),
    ("task_claims", "claim_id"),
    ("task_attempts", "task_cid"),
    ("task_attempts", "attempt_id"),
    ("completion_receipts", "task_cid"),
    ("completion_receipts", "receipt_cid"),
    ("domain_events", "event_id"),
    ("domain_events", "stream_id"),
    ("domain_events", "sequence"),
    ("path_claims", "task_cid"),
    ("path_claims", "path"),
    ("resource_claims", "resource_id"),
    ("worktrees", "worktree_id"),
    ("worktrees", "repository_id"),
    ("merge_queue_entries", "task_cid"),
    ("repositories", "repository_id"),
    ("goals", "goal_cid"),
    ("objectives", "objective_id"),
    ("plans", "plan_cid"),
    ("mutations", "mutation_id"),
    ("mutations", "task_cid"),
    ("proof_obligations", "obligation_id"),
    ("evidence_nodes", "evidence_id"),
    ("context_manifests", "manifest_cid"),
    ("context_manifests", "task_cid"),
    ("idempotency_records", "idempotency_key"),
    ("state_servers", "server_id"),
    ("client_sessions", "session_id"),
    ("maintenance_leases", "lease_id"),
    ("artifacts", "cid"),
)

# Join-critical identities introduced by migration 2.  These are verified
# separately so the ControlPlaneSchema@1 profile remains stable.
CAUSAL_EVENT_FEDERATION_JOIN_CRITICAL_IDENTITIES: Final[tuple[tuple[str, str], ...]] = (
    ("supervisor_instances", "tenant_id"),
    ("supervisor_instances", "federation_id"),
    ("supervisor_instances", "parent_supervisor_id"),
    ("supervisor_instances", "fencing_epoch"),
    ("idempotency_records", "tenant_id"),
    ("idempotency_records", "federation_id"),
    ("budget_reservations", "federation_id"),
    ("budget_reservations", "budget_id"),
    ("budget_reservations", "parent_reservation_id"),
    ("domain_events", "event_cid"),
    ("domain_events", "tenant_id"),
    ("domain_events", "federation_id"),
    ("domain_events", "supervisor_id"),
    ("domain_events", "repository_id"),
    ("domain_events", "tree_id"),
    ("domain_events", "goal_id"),
    ("domain_events", "subgoal_id"),
    ("domain_events", "symbol_id"),
    ("domain_events", "contract_id"),
    ("domain_events", "proof_obligation_id"),
    ("domain_events", "correlation_id"),
    ("domain_events", "causation_id"),
    ("domain_event_causal_parents", "event_id"),
    ("domain_event_causal_parents", "parent_event_id"),
    ("domain_event_changed_facts", "event_id"),
    ("domain_event_changed_facts", "fact_ref"),
    ("federations", "federation_id"),
    ("federations", "tenant_id"),
    ("federations", "program_id"),
    ("federations", "objective_ref"),
    ("federations", "policy_id"),
    ("federation_revisions", "federation_id"),
    ("federation_policies", "policy_id"),
    ("federation_policies", "federation_id"),
    ("federation_plans", "federation_plan_id"),
    ("federation_plans", "federation_id"),
    ("federation_receipts", "federation_receipt_id"),
    ("federation_receipts", "federation_id"),
    ("supervisor_definitions", "supervisor_definition_id"),
    ("supervisor_definitions", "federation_id"),
    ("supervisor_assignments", "assignment_id"),
    ("supervisor_assignments", "federation_id"),
    ("supervisor_assignments", "supervisor_id"),
    ("supervisor_assignments", "repository_id"),
    ("supervisor_assignments", "tree_id"),
    ("supervisor_capabilities", "capability_record_id"),
    ("supervisor_capabilities", "supervisor_id"),
    ("supervisor_checkpoints", "checkpoint_id"),
    ("supervisor_checkpoints", "supervisor_id"),
    ("supervisor_receipts", "supervisor_receipt_id"),
    ("supervisor_receipts", "supervisor_id"),
    ("subagent_definitions", "subagent_definition_id"),
    ("subagent_definitions", "federation_id"),
    ("subagent_instances", "subagent_id"),
    ("subagent_instances", "supervisor_id"),
    ("subagent_assignments", "subagent_assignment_id"),
    ("subagent_assignments", "subagent_id"),
    ("subagent_assignments", "task_cid"),
    ("subagent_capabilities", "subagent_capability_id"),
    ("subagent_capabilities", "subagent_id"),
    ("subagent_execution_slots", "tenant_id"),
    ("subagent_execution_slots", "federation_id"),
    ("subagent_execution_slots", "slot_number"),
    ("subagent_execution_slots", "subagent_id"),
    ("subagent_slot_ledger", "slot_ledger_id"),
    ("subagent_slot_ledger", "federation_id"),
    ("subagent_slot_ledger", "subagent_id"),
    ("subagent_slot_ledger", "event_id"),
    ("subagent_outcomes", "outcome_id"),
    ("subagent_outcomes", "subagent_id"),
    ("subagent_outcomes", "task_id"),
    ("supervisor_shards", "shard_id"),
    ("supervisor_shards", "federation_id"),
    ("shard_boundaries", "shard_boundary_id"),
    ("shard_boundaries", "shard_id"),
    ("shard_assignments", "shard_assignment_id"),
    ("shard_assignments", "shard_id"),
    ("shard_assignments", "supervisor_id"),
    ("shard_revisions", "shard_id"),
    ("shard_rebalance_plans", "rebalance_plan_id"),
    ("shard_rebalance_plans", "shard_id"),
    ("shard_rebalance_receipts", "rebalance_receipt_id"),
    ("shard_rebalance_receipts", "rebalance_plan_id"),
    ("federation_budgets", "federation_budget_id"),
    ("federation_budgets", "federation_id"),
    ("supervisor_budgets", "supervisor_budget_id"),
    ("supervisor_budgets", "federation_budget_id"),
    ("agent_budgets", "agent_budget_id"),
    ("agent_budgets", "supervisor_budget_id"),
    ("agent_budgets", "subagent_id"),
    ("token_budgets", "token_budget_id"),
    ("token_budgets", "parent_budget_id"),
    ("resource_budgets", "resource_budget_id"),
    ("resource_budgets", "parent_budget_id"),
    ("budget_ledger", "budget_ledger_entry_id"),
    ("budget_ledger", "budget_id"),
    ("budget_ledger", "event_id"),
    ("federation_admission_budget_reservations", "reservation_id"),
    ("federation_admission_budget_reservations", "tenant_id"),
    ("federation_admission_budget_reservations", "federation_id"),
    ("federation_admission_budget_reservations", "request_cid"),
    ("federation_admission_budget_reservations", "idempotency_key"),
    ("federation_admission_budget_reservations", "policy_id"),
    ("federation_admission_budget_reservations", "resource_budget_id"),
    ("federation_admission_budget_reservations", "token_budget_id"),
    ("federation_admission_budget_dimensions", "reservation_id"),
    ("federation_admission_budget_dimensions", "tenant_id"),
    ("federation_admission_budget_dimensions", "federation_id"),
    ("federation_admission_budget_dimensions", "dimension_name"),
    ("stream_sequence_heads", "stream_id"),
    ("global_sequence_head", "head_id"),
    ("transactional_outbox", "outbox_id"),
    ("transactional_outbox", "event_id"),
    ("transactional_outbox", "federation_id"),
    ("outbox_routing_dispositions", "disposition_id"),
    ("outbox_routing_dispositions", "federation_id"),
    ("outbox_routing_dispositions", "route_batch_id"),
    ("outbox_routing_disposition_events", "disposition_id"),
    ("outbox_routing_disposition_events", "event_id"),
    ("event_subscriptions", "subscription_id"),
    ("event_subscriptions", "consumer_id"),
    ("event_subscription_selectors", "selector_id"),
    ("event_subscription_selectors", "subscription_id"),
    ("consumer_cursors", "consumer_id"),
    ("consumer_cursors", "subscription_id"),
    ("supervisor_inbox", "inbox_entry_id"),
    ("supervisor_inbox", "supervisor_id"),
    ("supervisor_inbox", "event_id"),
    ("delivery_attempts", "attempt_id"),
    ("delivery_attempts", "event_id"),
    ("delivery_attempts", "subscription_id"),
    ("delivery_attempts", "subscription_revision"),
    ("delivery_attempts", "consumer_id"),
    ("dead_letters", "dead_letter_id"),
    ("dead_letters", "event_id"),
    ("event_delivery_queue", "delivery_id"),
    ("event_delivery_queue", "subscription_id"),
    ("event_delivery_queue", "consumer_id"),
    ("event_delivery_queue", "representative_event_id"),
    ("event_delivery_queue", "outbox_id"),
    ("event_coalescing", "coalescing_id"),
    ("event_coalescing", "current_event_id"),
    ("event_coalescing_coverage", "coverage_id"),
    ("event_coalescing_coverage", "decision_id"),
    ("event_coalescing_coverage", "representative_event_id"),
    ("event_coalescing_inputs", "coverage_id"),
    ("event_coalescing_inputs", "event_id"),
    ("event_acknowledgements", "acknowledgement_id"),
    ("event_acknowledgements", "event_id"),
    ("causal_nodes", "causal_node_id"),
    ("causal_nodes", "subject_ref"),
    ("causal_edges", "causal_edge_id"),
    ("causal_edges", "source_node_id"),
    ("causal_edges", "target_node_id"),
    ("causal_evidence", "causal_evidence_id"),
    ("causal_evidence", "causal_edge_id"),
    ("causal_abstraction_maps", "abstraction_map_id"),
    ("causal_abstraction_validations", "abstraction_validation_id"),
    ("causal_abstraction_validations", "abstraction_map_id"),
    ("causal_intervention_tests", "intervention_test_id"),
    ("causal_intervention_tests", "abstraction_map_id"),
    ("causal_slices", "causal_slice_id"),
    ("causal_slices", "root_event_id"),
    ("causal_frontiers", "causal_frontier_id"),
    ("causal_frontiers", "event_id"),
    ("causal_frontier_members", "causal_frontier_id"),
    ("causal_frontier_members", "subject_ref"),
    ("causal_invalidations", "causal_invalidation_id"),
    ("causal_invalidations", "event_id"),
    ("semantic_root_references", "semantic_root_reference_id"),
    ("semantic_root_references", "repository_id"),
    ("semantic_root_references", "tree_id"),
    ("semantic_capsule_references", "semantic_capsule_reference_id"),
    ("semantic_capsule_references", "subject_ref"),
    ("world_snapshot_references", "world_snapshot_reference_id"),
    ("world_snapshot_references", "federation_id"),
    ("proof_reference_projections", "proof_reference_id"),
    ("proof_reference_projections", "obligation_ref"),
    ("test_reference_projections", "test_reference_id"),
    ("test_reference_projections", "test_ref"),
    ("retrieval_indexes", "retrieval_index_id"),
    ("retrieval_indexes", "repository_id"),
    ("retrieval_index_partitions", "retrieval_partition_id"),
    ("retrieval_index_partitions", "retrieval_index_id"),
    ("retrieval_receipts", "retrieval_receipt_id"),
    ("retrieval_receipts", "retrieval_index_id"),
    ("retrieval_nominations", "retrieval_nomination_id"),
    ("retrieval_nominations", "retrieval_receipt_id"),
    ("federation_commands", "federation_command_id"),
    ("federation_commands", "federation_id"),
    ("federation_authorization_decisions", "authorization_decision_id"),
    ("federation_authorization_decisions", "federation_id"),
    ("federation_policy_decisions", "policy_decision_id"),
    ("federation_policy_decisions", "policy_id"),
    ("federation_confirmation_receipts", "confirmation_receipt_id"),
    ("federation_confirmation_receipts", "federation_command_id"),
    ("federation_effect_reservations", "effect_reservation_id"),
    ("federation_effect_reservations", "task_cid"),
    ("federation_effect_observations", "effect_observation_id"),
    ("federation_effect_observations", "effect_reservation_id"),
    ("federation_audit_receipts", "audit_receipt_id"),
    ("federation_audit_receipts", "subject_ref"),
    ("programs", "program_id"),
    ("programs", "tenant_id"),
    ("programs", "objective_ref"),
    ("program_revisions", "program_revision_id"),
    ("program_revisions", "program_id"),
    ("subgoals", "subgoal_id"),
    ("subgoals", "federation_id"),
    ("subgoals", "goal_cid"),
    ("plan_branches", "plan_branch_id"),
    ("plan_branches", "plan_cid"),
    ("plan_branches", "plan_revision_id"),
    ("task_conflicts", "task_conflict_id"),
    ("task_conflicts", "left_task_cid"),
    ("task_conflicts", "right_task_cid"),
    ("task_resolutions", "task_resolution_id"),
    ("task_resolutions", "task_cid"),
    ("federation_task_bindings", "federation_task_binding_id"),
    ("federation_task_bindings", "tenant_id"),
    ("federation_task_bindings", "federation_id"),
    ("federation_task_bindings", "task_cid"),
    ("federation_task_bindings", "repository_id"),
    ("federation_task_bindings", "tree_id"),
    ("process_births", "process_birth_id"),
    ("process_births", "federation_id"),
    ("supervisor_runtime_leases", "runtime_lease_id"),
    ("supervisor_runtime_leases", "federation_id"),
    ("supervisor_runtime_leases", "supervisor_id"),
    ("supervisor_runtime_leases", "lease_id"),
    ("supervisor_runtime_leases", "process_birth_id"),
    ("repository_trees", "repository_tree_id"),
    ("repository_trees", "repository_id"),
    ("repository_trees", "tree_id"),
    ("fencing_epochs", "fencing_epoch_id"),
    ("fencing_epochs", "subject_id"),
    ("fencing_epochs", "lease_id"),
    ("task_checkpoints", "task_checkpoint_id"),
    ("task_checkpoints", "task_cid"),
    ("task_checkpoints", "attempt_id"),
    ("provider_reservations", "provider_reservation_id"),
    ("provider_reservations", "task_cid"),
    ("provider_reservations", "provider_id"),
    ("semantic_effect_references", "semantic_effect_id"),
    ("semantic_effect_references", "symbol_ref"),
    ("semantic_relationship_references", "semantic_relationship_id"),
    ("semantic_relationship_references", "source_ref"),
    ("semantic_relationship_references", "target_ref"),
    ("semantic_capsule_dependencies", "capsule_dependency_id"),
    ("semantic_capsule_dependencies", "capsule_ref"),
    ("semantic_capsule_dependencies", "dependency_capsule_ref"),
    ("semantic_contract_references", "semantic_contract_id"),
    ("semantic_contract_references", "contract_ref"),
    ("environment_binding_references", "environment_binding_id"),
    ("environment_binding_references", "environment_ref"),
    ("proof_units", "proof_unit_id"),
    ("proof_units", "obligation_ref"),
    ("proof_receipts", "proof_receipt_id"),
    ("proof_receipts", "proof_unit_id"),
    ("proof_cache_entries", "proof_cache_entry_id"),
    ("proof_cache_entries", "obligation_ref"),
    ("proof_seals", "proof_seal_id"),
    ("proof_seals", "proof_receipt_id"),
    ("test_selections", "test_selection_id"),
    ("test_selections", "task_cid"),
    ("test_attempts", "test_attempt_id"),
    ("test_attempts", "test_selection_id"),
    ("test_receipts", "test_receipt_id"),
    ("test_receipts", "test_attempt_id"),
    ("validation_plans", "validation_plan_id"),
    ("validation_plans", "task_cid"),
    ("documents", "document_id"),
    ("documents", "repository_id"),
    ("documents", "tree_id"),
    ("document_chunks", "document_chunk_id"),
    ("document_chunks", "document_id"),
    ("bm25_terms", "bm25_term_id"),
    ("bm25_terms", "retrieval_index_id"),
    ("bm25_postings", "bm25_posting_id"),
    ("bm25_postings", "bm25_term_id"),
    ("bm25_postings", "document_chunk_id"),
    ("vectors", "vector_id"),
    ("vectors", "retrieval_index_id"),
    ("vectors", "document_chunk_id"),
    ("vector_metadata", "vector_metadata_id"),
    ("vector_metadata", "vector_id"),
    ("knowledge_graph_nodes", "knowledge_graph_node_id"),
    ("knowledge_graph_nodes", "subject_ref"),
    ("knowledge_graph_edges", "knowledge_graph_edge_id"),
    ("knowledge_graph_edges", "source_node_id"),
    ("knowledge_graph_edges", "target_node_id"),
)

_OPAQUE_JSON_COLUMN_RE: Final = re.compile(
    r"(?:_json|payload_json|body_json|identity_json|extension_json|"
    r"effect_json|policy_json|argv_json|provenance_json)$",
    re.IGNORECASE,
)


class ControlPlaneSchemaError(RuntimeError):
    """Base class for fail-closed control-plane schema errors."""


class ControlPlaneSchemaInstallError(ControlPlaneSchemaError):
    """Schema installation or fingerprint verification failed."""


class ControlPlaneSchemaCompatibilityError(ControlPlaneSchemaError):
    """Existing task CID / lease semantics are not preserved."""


class ControlPlaneSchemaIdentityError(ControlPlaneSchemaError):
    """A join-critical identity is missing or only present in opaque JSON."""


@dataclass(frozen=True)
class DuckDBQuackDependencyProfile:
    """Pinned install profile for the optional supervisor service."""

    extra_name: str = SUPERVISOR_OPTIONAL_EXTRA
    duckdb_spec: str = PINNED_DUCKDB_VERSION_SPEC
    duckdb_major: int = PINNED_DUCKDB_MAJOR
    duckdb_minor: int = PINNED_DUCKDB_MINOR
    duckdb_version_prefix: str = PINNED_DUCKDB_VERSION_PREFIX
    extension_name: str = PINNED_QUACK_EXTENSION
    extension_api: str = PINNED_QUACK_EXTENSION_API
    profile_id: str = PINNED_PROFILE_ID

    def to_dict(self) -> dict[str, Any]:
        return {
            "extra_name": self.extra_name,
            "duckdb_spec": self.duckdb_spec,
            "duckdb_major": int(self.duckdb_major),
            "duckdb_minor": int(self.duckdb_minor),
            "duckdb_version_prefix": self.duckdb_version_prefix,
            "extension_name": self.extension_name,
            "extension_api": self.extension_api,
            "profile_id": self.profile_id,
        }


@dataclass(frozen=True)
class ControlPlaneSchema:
    """Canonical inventory for the normalized control-plane physical schema.

    Interface: ``ControlPlaneSchema@1``.
    """

    INTERFACE: ClassVar[str] = CONTROL_PLANE_SCHEMA_INTERFACE
    SCHEMA: ClassVar[str] = CONTROL_PLANE_SCHEMA_SCHEMA

    schema_revision: int = CONTROL_PLANE_SCHEMA_REVISION
    migration_id: str = CONTROL_PLANE_MIGRATION_ID
    migration_version: int = CONTROL_PLANE_MIGRATION_VERSION
    domains: tuple[str, ...] = SCHEMA_DOMAINS
    domain_tables: Mapping[str, tuple[str, ...]] = DOMAIN_TABLES
    diagnostic_views: tuple[str, ...] = DIAGNOSTIC_VIEWS
    bookkeeping_tables: tuple[str, ...] = BOOKKEEPING_TABLES
    task_identity_columns: tuple[str, ...] = TASK_IDENTITY_COLUMNS
    lease_identity_columns: tuple[str, ...] = LEASE_IDENTITY_COLUMNS
    join_critical_identities: tuple[tuple[str, str], ...] = JOIN_CRITICAL_IDENTITIES
    dependency_profile: DuckDBQuackDependencyProfile = DuckDBQuackDependencyProfile()

    def __post_init__(self) -> None:
        if int(self.schema_revision) < 1:
            raise ControlPlaneSchemaError("schema_revision must be >= 1")
        if tuple(self.domains) != SCHEMA_DOMAINS:
            raise ControlPlaneSchemaError("domains must match the closed SCHEMA_DOMAINS vocabulary")
        missing = [name for name in SCHEMA_DOMAINS if name not in self.domain_tables]
        if missing:
            raise ControlPlaneSchemaError(f"domain_tables missing domains: {missing}")

    @property
    def all_domain_tables(self) -> tuple[str, ...]:
        tables: list[str] = []
        seen: set[str] = set()
        for domain in self.domains:
            for table in self.domain_tables[domain]:
                if table not in seen:
                    tables.append(table)
                    seen.add(table)
        return tuple(tables)

    def sql_path(self) -> Path:
        return Path(__file__).resolve().parent / "sql" / CONTROL_PLANE_SQL_FILENAME

    def sql_text(self) -> str:
        path = self.sql_path()
        if not path.is_file():
            raise ControlPlaneSchemaInstallError(f"control-plane schema SQL is missing: {path}")
        return path.read_text(encoding="utf-8")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "interface": self.INTERFACE,
            "schema_revision": int(self.schema_revision),
            "migration_id": self.migration_id,
            "migration_version": int(self.migration_version),
            "domains": list(self.domains),
            "domain_tables": {
                domain: list(tables) for domain, tables in self.domain_tables.items()
            },
            "diagnostic_views": list(self.diagnostic_views),
            "bookkeeping_tables": list(self.bookkeeping_tables),
            "task_identity_columns": list(self.task_identity_columns),
            "lease_identity_columns": list(self.lease_identity_columns),
            "join_critical_identities": [
                {"table": table, "column": column}
                for table, column in self.join_critical_identities
            ],
            "dependency_profile": self.dependency_profile.to_dict(),
            "sql_filename": CONTROL_PLANE_SQL_FILENAME,
        }


@dataclass(frozen=True)
class CausalEventFederationSchemaExtension:
    """Additive migration-2 inventory layered on ControlPlaneSchema@1."""

    INTERFACE: ClassVar[str] = CAUSAL_EVENT_FEDERATION_SCHEMA_EXTENSION_INTERFACE
    SCHEMA: ClassVar[str] = CAUSAL_EVENT_FEDERATION_SCHEMA_EXTENSION_SCHEMA

    schema_revision: int = CAUSAL_EVENT_FEDERATION_SCHEMA_REVISION
    migration_id: str = CAUSAL_EVENT_FEDERATION_MIGRATION_ID
    migration_version: int = CAUSAL_EVENT_FEDERATION_MIGRATION_VERSION
    tables: tuple[str, ...] = CAUSAL_EVENT_FEDERATION_TABLES
    join_critical_identities: tuple[tuple[str, str], ...] = (
        CAUSAL_EVENT_FEDERATION_JOIN_CRITICAL_IDENTITIES
    )
    reference_tables: tuple[str, ...] = CAUSAL_EVENT_FEDERATION_REFERENCE_TABLES
    section_9_relations: Mapping[str, str] = CAUSAL_EVENT_FEDERATION_SECTION_9_RELATIONS

    def __post_init__(self) -> None:
        if self.schema_revision != CAUSAL_EVENT_FEDERATION_SCHEMA_REVISION:
            raise ControlPlaneSchemaError("unsupported causal-event federation schema revision")
        if self.migration_version != CAUSAL_EVENT_FEDERATION_MIGRATION_VERSION:
            raise ControlPlaneSchemaError("unsupported causal-event federation migration version")
        if not self.tables or len(set(self.tables)) != len(self.tables):
            raise ControlPlaneSchemaError(
                "causal-event federation table inventory is empty or duplicated"
            )
        if not self.section_9_relations or any(
            not concept or not relation for concept, relation in self.section_9_relations.items()
        ):
            raise ControlPlaneSchemaError("Section 9 relation map is empty or invalid")

    def sql_path(self) -> Path:
        return Path(__file__).resolve().parent / "sql" / CAUSAL_EVENT_FEDERATION_SQL_FILENAME

    def sql_text(self) -> str:
        path = self.sql_path()
        if not path.is_file():
            raise ControlPlaneSchemaInstallError(
                f"causal-event federation migration SQL is missing: {path}"
            )
        return path.read_text(encoding="utf-8")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "interface": self.INTERFACE,
            "schema_revision": self.schema_revision,
            "migration_id": self.migration_id,
            "migration_version": self.migration_version,
            "tables": list(self.tables),
            "join_critical_identities": [
                {"table": table, "column": column}
                for table, column in self.join_critical_identities
            ],
            "reference_tables": list(self.reference_tables),
            "section_9_relations": dict(sorted(self.section_9_relations.items())),
            "sql_filename": CAUSAL_EVENT_FEDERATION_SQL_FILENAME,
            "base_interface": CONTROL_PLANE_SCHEMA_INTERFACE,
        }


def default_control_plane_schema() -> ControlPlaneSchema:
    """Return the package ControlPlaneSchema@1 inventory."""

    return ControlPlaneSchema()


def default_causal_event_federation_schema_extension() -> CausalEventFederationSchemaExtension:
    """Return the additive federation schema inventory without changing @1."""

    return CausalEventFederationSchemaExtension()


def default_dependency_profile() -> DuckDBQuackDependencyProfile:
    """Return the pinned DuckDB/Quack optional-service profile."""

    return DuckDBQuackDependencyProfile()


def package_sql_directory() -> Path:
    return Path(__file__).resolve().parent / "sql"


def load_control_plane_catalog(
    sql_directory: Path | str | None = None,
) -> MigrationCatalog:
    """Load the contiguous package control-plane migration catalog."""

    return load_default_catalog(sql_directory or package_sql_directory())


_PROFILE_SECTION_MARKERS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "meta": "-- meta: schema contracts and store generation surface",
        "code": "-- code: snapshots, AST, mutations, impact",
        "cache": "-- cache: context, prompts, provider calls, decision cache",
        "seed": "-- Seed schema_contracts for domain inventory (deterministic)",
        "views": "-- Constrained diagnostic / context views (read-only projections)",
    }
)

_PROFILE_CONTRACT_ROWS: Final[tuple[tuple[str, ...], ...]] = (
    (
        "contract:DatasetsAuthoritativeOperationalControlPlane@1",
        "DatasetsAuthoritativeOperationalControlPlane@1",
        "meta",
        DATASETS_AUTHORITATIVE_OPERATIONAL_SCHEMA,
        "Accelerator operational coordination only; semantic and proof truth "
        "remains authoritative in ipfs_datasets_py",
    ),
    (
        "contract:meta@1",
        "ControlPlaneOperationalDomain@1",
        "meta",
        "ipfs_accelerate_py/agent-supervisor/domain-meta@1",
        "Operational schema contracts and store generations",
    ),
    (
        "contract:intent@1",
        "ControlPlaneOperationalDomain@1",
        "intent",
        "ipfs_accelerate_py/agent-supervisor/domain-intent@1",
        "Objectives, goals, plans, and tasks",
    ),
    (
        "contract:schedule@1",
        "ControlPlaneOperationalDomain@1",
        "schedule",
        "ipfs_accelerate_py/agent-supervisor/domain-schedule@1",
        "Assignments, blocks, refill, and findings",
    ),
    (
        "contract:runtime@1",
        "ControlPlaneOperationalDomain@1",
        "runtime",
        "ipfs_accelerate_py/agent-supervisor/domain-runtime@1",
        "Daemons, attempts, claims, validation, and merge",
    ),
    (
        "contract:git@1",
        "ControlPlaneOperationalDomain@1",
        "git",
        "ipfs_accelerate_py/agent-supervisor/domain-git@1",
        "Repositories, worktrees, leases, and path claims",
    ),
    (
        "contract:cache@1",
        "ControlPlaneOperationalDomain@1",
        "cache",
        "ipfs_accelerate_py/agent-supervisor/domain-cache@1",
        "Operational context references, prompts, and decision cache",
    ),
    (
        "contract:control@1",
        "ControlPlaneOperationalDomain@1",
        "control",
        "ipfs_accelerate_py/agent-supervisor/domain-control@1",
        "Servers, sessions, authorization, maintenance, and events",
    ),
    (
        "contract:improve@1",
        "ControlPlaneOperationalDomain@1",
        "improve",
        "ipfs_accelerate_py/agent-supervisor/domain-improve@1",
        "Operational metrics, budgets, and churn telemetry",
    ),
    (
        "contract:intent-completion-evidence-projection@1",
        "IntentRepository@1",
        "runtime",
        "ipfs_accelerate_py/agent-supervisor/intent-completion-evidence@1",
        "Opaque content-addressed operational completion receipts only; "
        "semantic proof authority remains in ipfs_datasets_py",
    ),
)


def _unique_marker_offset(sql_text: str, marker: str) -> int:
    count = sql_text.count(marker)
    if count != 1:
        raise ControlPlaneSchemaInstallError(
            "canonical control-plane SQL profile marker must occur exactly "
            f"once: {marker!r}; found {count}"
        )
    return sql_text.index(marker)


def _section_start(sql_text: str, marker: str) -> int:
    marker_offset = _unique_marker_offset(sql_text, marker)
    divider = "-- ---------------------------------------------------------------------------"
    start = sql_text.rfind(divider, 0, marker_offset)
    if start < 0:
        raise ControlPlaneSchemaInstallError(
            f"canonical control-plane SQL section divider missing for {marker!r}"
        )
    return start


def _profile_contract_seed_sql() -> str:
    rows: list[str] = []
    for (
        contract_id,
        interface_name,
        domain_name,
        payload_schema,
        description,
    ) in _PROFILE_CONTRACT_ROWS:
        values = (
            contract_id,
            interface_name,
            domain_name,
            payload_schema,
            description,
        )
        if any("'" in value for value in values):
            raise ControlPlaneSchemaInstallError(
                "profile schema-contract literals must not contain SQL quotes"
            )
        rows.append(
            "    ("
            f"'{contract_id}', '{interface_name}', '{domain_name}', 1, "
            f"'{payload_schema}', '{description}', "
            "'1970-01-01T00:00:00Z')"
        )
    return (
        "-- Profile-specific operational authority contracts.\n"
        "INSERT INTO schema_contracts (\n"
        "    contract_id, interface_name, domain_name, schema_revision,\n"
        "    payload_schema, description, created_at\n"
        ") VALUES\n" + ",\n".join(rows) + ";\n"
    )


def _created_relation_names(sql_text: str, relation_kind: str) -> frozenset[str]:
    pattern = re.compile(
        rf"(?im)^\s*CREATE\s+{relation_kind}\s+(?:IF\s+NOT\s+EXISTS\s+)?"
        r'"?([a-z][a-z0-9_]*)"?\s*(?:\(|AS\b)'
    )
    return frozenset(match.group(1).lower() for match in pattern.finditer(sql_text))


def datasets_authoritative_operational_schema_sql() -> str:
    """Derive the constrained profile SQL from canonical 0001 SQL.

    The extraction is marker- and inventory-checked.  A change to the canonical
    section layout, an unexpected relation, or loss of a required operational
    relation therefore refuses installation instead of silently expanding or
    narrowing accelerator authority.
    """

    canonical = default_control_plane_schema().sql_text()
    starts = {
        name: _section_start(canonical, marker) for name, marker in _PROFILE_SECTION_MARKERS.items()
    }
    ordered = [starts[name] for name in ("meta", "code", "cache", "seed", "views")]
    if ordered != sorted(ordered) or len(set(ordered)) != len(ordered):
        raise ControlPlaneSchemaInstallError(
            "canonical control-plane SQL sections are missing or out of order"
        )

    evidence_start_marker = "CREATE TABLE evidence_nodes ("
    evidence_start = _unique_marker_offset(canonical, evidence_start_marker)
    if not starts["code"] < evidence_start < starts["cache"]:
        raise ControlPlaneSchemaInstallError(
            "evidence_nodes is no longer isolated between code and cache sections"
        )

    sql_text = (
        "\n".join(
            (
                "-- Derived datasets-authoritative operational profile.\n"
                "-- ipfs_datasets_py owns semantic state, AST, proof obligations,\n"
                "-- proof attempts, and counterexamples; this migration contains\n"
                "-- operational coordination projections and opaque references only.",
                canonical[starts["meta"] : starts["code"]].strip(),
                "-- ---------------------------------------------------------------------------\n"
                "-- runtime: opaque operational completion-evidence receipt projection\n"
                "-- ---------------------------------------------------------------------------\n"
                + canonical[evidence_start : starts["cache"]].strip(),
                canonical[starts["cache"] : starts["seed"]].strip(),
                _profile_contract_seed_sql().strip(),
                canonical[starts["views"] :].strip(),
            )
        )
        + "\n"
    )

    created_tables = _created_relation_names(sql_text, "TABLE")
    expected_tables = frozenset(DATASETS_AUTHORITATIVE_OPERATIONAL_TABLES)
    if created_tables != expected_tables:
        raise ControlPlaneSchemaInstallError(
            "derived operational table inventory drifted: "
            f"missing={sorted(expected_tables - created_tables)}, "
            f"unexpected={sorted(created_tables - expected_tables)}"
        )
    created_views = _created_relation_names(sql_text, "VIEW")
    expected_views = frozenset(DIAGNOSTIC_VIEWS)
    if created_views != expected_views:
        raise ControlPlaneSchemaInstallError(
            "derived operational view inventory drifted: "
            f"missing={sorted(expected_views - created_views)}, "
            f"unexpected={sorted(created_views - expected_views)}"
        )
    forbidden = created_tables & DATASETS_SEMANTIC_TRUTH_RELATIONS
    if forbidden:
        raise ControlPlaneSchemaInstallError(
            f"derived operational profile contains datasets-owned relations: {sorted(forbidden)}"
        )
    return sql_text


def load_datasets_authoritative_operational_catalog() -> MigrationCatalog:
    """Return the checksum-bound one-migration operational profile catalog."""

    sql_text = datasets_authoritative_operational_schema_sql()
    required_relation_names = tuple(
        sorted(set(DATASETS_AUTHORITATIVE_OPERATIONAL_TABLES).union(DIAGNOSTIC_VIEWS))
    )
    relation_literals = ", ".join(f"'{name}'" for name in required_relation_names)
    forbidden_literals = ", ".join(
        f"'{name}'" for name in sorted(DATASETS_SEMANTIC_TRUTH_RELATIONS)
    )
    migration = ControlPlaneMigration.from_sql(
        version=DATASETS_AUTHORITATIVE_OPERATIONAL_MIGRATION_VERSION,
        migration_id=DATASETS_AUTHORITATIVE_OPERATIONAL_MIGRATION_ID,
        sql_text=sql_text,
        description=("datasets-authoritative accelerator operational control-plane profile"),
        preconditions=(
            "SELECT COUNT(*) = 0 FROM information_schema.tables "
            "WHERE table_schema = 'main' AND table_name IN "
            f"({forbidden_literals})",
        ),
        postconditions=(
            "SELECT COUNT(*) = "
            f"{len(required_relation_names)} FROM information_schema.tables "
            "WHERE table_schema = 'main' AND table_name IN "
            f"({relation_literals})",
            "SELECT COUNT(*) = 0 FROM information_schema.tables "
            "WHERE table_schema = 'main' AND table_name IN "
            f"({forbidden_literals})",
            "SELECT COUNT(*) = 1 FROM schema_contracts WHERE contract_id = "
            "'contract:DatasetsAuthoritativeOperationalControlPlane@1'",
            "SELECT COUNT(*) = 0 FROM schema_contracts WHERE domain_name IN ('code', 'evidence')",
        ),
        source_path=(
            f"{default_control_plane_schema().sql_path()}"
            f"#{DATASETS_AUTHORITATIVE_OPERATIONAL_PROFILE_ID}"
        ),
    )
    return MigrationCatalog.from_migrations((migration,))


def install_control_plane_schema(
    database_path: Path | str,
    *,
    catalog: MigrationCatalog | None = None,
    application_version: str | None = None,
    tool_version: str | None = None,
    owner_id: str | None = None,
) -> MigrationRunReport:
    """Apply the control-plane catalog to ``database_path`` (idempotent)."""

    if not duckdb_available():
        raise ControlPlaneSchemaInstallError(
            "DuckDB is required to install the control-plane schema"
        )
    resolved_catalog = catalog or load_control_plane_catalog()
    if resolved_catalog.latest_version < CONTROL_PLANE_MIGRATION_VERSION:
        raise ControlPlaneSchemaInstallError(
            f"control-plane catalog is missing migration {CONTROL_PLANE_MIGRATION_ID}"
        )
    migration = resolved_catalog.get(CONTROL_PLANE_MIGRATION_VERSION)
    if migration.migration_id != CONTROL_PLANE_MIGRATION_ID:
        raise ControlPlaneSchemaInstallError(
            f"expected migration_id {CONTROL_PLANE_MIGRATION_ID}, got {migration.migration_id}"
        )
    if resolved_catalog.latest_version < CAUSAL_EVENT_FEDERATION_MIGRATION_VERSION:
        raise ControlPlaneSchemaInstallError(
            f"control-plane catalog is missing migration {CAUSAL_EVENT_FEDERATION_MIGRATION_ID}"
        )
    federation_migration = resolved_catalog.get(CAUSAL_EVENT_FEDERATION_MIGRATION_VERSION)
    if federation_migration.migration_id != CAUSAL_EVENT_FEDERATION_MIGRATION_ID:
        raise ControlPlaneSchemaInstallError(
            f"expected migration_id {CAUSAL_EVENT_FEDERATION_MIGRATION_ID}, "
            f"got {federation_migration.migration_id}"
        )
    runner = ControlPlaneMigrationRunner.for_database(
        database_path,
        catalog=resolved_catalog,
        application_version=application_version,
        tool_version=tool_version,
        owner_id=owner_id,
    )
    return runner.apply()


def prove_fresh_and_upgraded_equivalence(
    left_database_path: Path | str,
    right_database_path: Path | str,
    *,
    catalog: MigrationCatalog | None = None,
    application_version: str = "0.0.45",
    tool_version: str = "1.5.2",
) -> dict[str, Any]:
    """Prove empty-to-latest fingerprints match on two independent databases."""

    resolved = catalog or load_control_plane_catalog()
    left_runner = ControlPlaneMigrationRunner.for_database(
        left_database_path,
        catalog=resolved,
        application_version=application_version,
        tool_version=tool_version,
        owner_id="schema-proof-left",
    )
    return left_runner.prove_empty_to_latest_equivalence(
        other_database_path=right_database_path,
    )


def _table_columns(connection: Any, table_name: str) -> dict[str, str]:
    rows = connection.execute(
        """
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = 'main' AND table_name = ?
        ORDER BY ordinal_position
        """,
        [table_name],
    ).fetchall()
    out: dict[str, str] = {}
    for row in rows:
        if isinstance(row, Mapping):
            out[str(row["column_name"])] = str(row["data_type"])
        else:
            out[str(row[0])] = str(row[1])
    return out


def _relation_exists(
    connection: Any,
    name: str,
    *,
    table_type: str | None = None,
) -> bool:
    sql = """
        SELECT 1
        FROM information_schema.tables
        WHERE table_schema = 'main' AND table_name = ?
    """
    params: list[Any] = [name]
    if table_type is not None:
        sql += " AND table_type = ?"
        params.append(table_type)
    sql += " LIMIT 1"
    return connection.execute(sql, params).fetchone() is not None


def _main_relation_names(connection: Any) -> frozenset[str]:
    rows = connection.execute(
        """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'main'
        ORDER BY table_name
        """
    ).fetchall()
    return frozenset(
        str(row["table_name"] if isinstance(row, Mapping) else row[0]).lower() for row in rows
    )


def _forbidden_schema_contracts(connection: Any) -> tuple[str, ...]:
    if not _relation_exists(connection, "schema_contracts"):
        return ()
    rows = connection.execute(
        """
        SELECT contract_id, interface_name, domain_name
        FROM schema_contracts
        WHERE domain_name IN ('code', 'evidence')
           OR contract_id = 'contract:ControlPlaneSchema@1'
           OR interface_name = 'ControlPlaneSchema@1'
        ORDER BY contract_id, interface_name, domain_name
        """
    ).fetchall()
    return tuple(
        "/".join(
            str(value)
            for value in (
                (row["contract_id"], row["interface_name"], row["domain_name"])
                if isinstance(row, Mapping)
                else (row[0], row[1], row[2])
            )
        )
        for row in rows
    )


def _assert_datasets_authoritative_database(
    connection: Any,
    *,
    operation: str,
) -> tuple[frozenset[str], tuple[str, ...]]:
    relations = _main_relation_names(connection)
    forbidden_relations = relations & DATASETS_SEMANTIC_TRUTH_RELATIONS
    forbidden_contracts = _forbidden_schema_contracts(connection)
    if forbidden_relations or forbidden_contracts:
        raise ControlPlaneSchemaInstallError(
            f"refusing {operation}: database is not a datasets-authoritative "
            "operational control plane; "
            f"forbidden_relations={sorted(forbidden_relations)}, "
            f"forbidden_contracts={list(forbidden_contracts)}"
        )
    return relations, forbidden_contracts


def install_datasets_authoritative_operational_schema(
    database_path: Path | str,
    *,
    application_version: str | None = None,
    tool_version: str | None = None,
    owner_id: str | None = None,
    database_uuid: str | None = None,
) -> MigrationRunReport:
    """Install the accelerator-only operational profile using the core runner.

    Existing full-control-plane databases fail closed.  The profile catalog is
    checksum-bound to deterministic sections of canonical ``0001`` SQL, so the
    same migration receipts, ownership lease, drift detection, and transaction
    behavior are reused rather than introducing another plan/task store.
    """

    if not duckdb_available():
        raise ControlPlaneSchemaInstallError(
            "DuckDB is required to install the datasets-authoritative "
            "operational control-plane schema"
        )
    path = Path(database_path)
    if path.exists():
        try:
            with open_duckdb_connection(path) as connection:
                _assert_datasets_authoritative_database(
                    connection,
                    operation="operational-profile installation",
                )
        except ControlPlaneSchemaError:
            raise
        except Exception as exc:
            raise ControlPlaneSchemaInstallError(
                f"unable to inspect existing control-plane database {path}: {exc}"
            ) from exc

    catalog = load_datasets_authoritative_operational_catalog()
    runner = ControlPlaneMigrationRunner.for_database(
        path,
        catalog=catalog,
        application_version=application_version,
        tool_version=tool_version,
        owner_id=owner_id,
        database_uuid=database_uuid,
    )
    report = runner.apply()
    verified = verify_datasets_authoritative_operational_schema(path)
    if not bool(verified.get("valid")):
        raise ControlPlaneSchemaInstallError(
            "operational-profile verification did not return valid=true"
        )
    if verified["schema_fingerprint"] != report.schema_fingerprint:
        raise ControlPlaneSchemaInstallError(
            "installed operational-profile fingerprint differs from migration receipt fingerprint"
        )
    return report


def verify_datasets_authoritative_operational_schema(
    database_path: Path | str,
) -> dict[str, Any]:
    """Verify profile identity, operational surfaces, and semantic exclusions."""

    if not duckdb_available():
        raise ControlPlaneSchemaInstallError(
            "DuckDB is required to verify the datasets-authoritative "
            "operational control-plane schema"
        )
    catalog = load_datasets_authoritative_operational_catalog()
    expected_migration = catalog.get(DATASETS_AUTHORITATIVE_OPERATIONAL_MIGRATION_VERSION)
    report: dict[str, Any] = {
        "valid": False,
        "database_path": str(database_path),
        "profile_id": DATASETS_AUTHORITATIVE_OPERATIONAL_PROFILE_ID,
        "profile_schema": DATASETS_AUTHORITATIVE_OPERATIONAL_SCHEMA,
        "migration_id": DATASETS_AUTHORITATIVE_OPERATIONAL_MIGRATION_ID,
        "migration_checksum": expected_migration.checksum,
        "catalog_fingerprint": catalog.fingerprint(),
        "required_tables_ok": [],
        "views_ok": [],
        "join_critical_ok": [],
        "task_columns_ok": [],
        "lease_columns_ok": [],
        "forbidden_relations": [],
        "forbidden_contracts": [],
        "operational_evidence": {
            "relation": "evidence_nodes",
            "purpose": "content-addressed operational completion/result receipts",
            "semantic_and_proof_authority": "ipfs_datasets_py",
        },
        "authority_contract": {},
    }
    with open_duckdb_connection(database_path) as connection:
        relations, forbidden_contracts = _assert_datasets_authoritative_database(
            connection,
            operation="operational-profile verification",
        )
        report["forbidden_relations"] = sorted(relations & DATASETS_SEMANTIC_TRUTH_RELATIONS)
        report["forbidden_contracts"] = list(forbidden_contracts)

        for table in BOOKKEEPING_TABLES:
            if table not in relations:
                raise ControlPlaneSchemaInstallError(
                    f"operational-profile bookkeeping table missing: {table}"
                )
        for table in DATASETS_AUTHORITATIVE_OPERATIONAL_TABLES:
            if table not in relations:
                raise ControlPlaneSchemaInstallError(f"operational-profile table missing: {table}")
            report["required_tables_ok"].append(table)
        for view in DIAGNOSTIC_VIEWS:
            if view not in relations:
                raise ControlPlaneSchemaInstallError(
                    f"operational-profile diagnostic view missing: {view}"
                )
            report["views_ok"].append(view)

        migration_row = connection.execute(
            """
            SELECT migration_id, checksum
            FROM schema_migrations
            WHERE version = ?
            """,
            [DATASETS_AUTHORITATIVE_OPERATIONAL_MIGRATION_VERSION],
        ).fetchone()
        if migration_row is None:
            raise ControlPlaneSchemaInstallError("operational-profile migration receipt is missing")
        applied_id = str(
            migration_row["migration_id"]
            if isinstance(migration_row, Mapping)
            else migration_row[0]
        )
        applied_checksum = str(
            migration_row["checksum"] if isinstance(migration_row, Mapping) else migration_row[1]
        )
        if (
            applied_id != expected_migration.migration_id
            or applied_checksum != expected_migration.checksum
        ):
            raise ControlPlaneSchemaInstallError(
                "operational-profile migration identity/checksum mismatch: "
                f"applied={applied_id}/{applied_checksum}, "
                f"expected={expected_migration.migration_id}/"
                f"{expected_migration.checksum}"
            )

        root_contract = connection.execute(
            """
            SELECT payload_schema, description
            FROM schema_contracts
            WHERE contract_id =
                'contract:DatasetsAuthoritativeOperationalControlPlane@1'
            """
        ).fetchone()
        if root_contract is None:
            raise ControlPlaneSchemaInstallError(
                "datasets-authoritative operational profile contract is missing"
            )
        payload_schema = str(
            root_contract["payload_schema"]
            if isinstance(root_contract, Mapping)
            else root_contract[0]
        )
        description = str(
            root_contract["description"] if isinstance(root_contract, Mapping) else root_contract[1]
        )
        if payload_schema != DATASETS_AUTHORITATIVE_OPERATIONAL_SCHEMA:
            raise ControlPlaneSchemaInstallError(
                "datasets-authoritative operational profile contract drifted"
            )
        if "operational" not in description.lower() or "ipfs_datasets_py" not in description:
            raise ControlPlaneSchemaInstallError(
                "operational profile contract does not preserve the datasets "
                "semantic/proof authority boundary"
            )
        report["authority_contract"] = {
            "contract_id": ("contract:DatasetsAuthoritativeOperationalControlPlane@1"),
            "payload_schema": payload_schema,
            "operational_authority": "ipfs_accelerate_py",
            "semantic_and_proof_authority": "ipfs_datasets_py",
        }

        profile_join_identities = tuple(
            (table, column)
            for table, column in JOIN_CRITICAL_IDENTITIES
            if table in DATASETS_AUTHORITATIVE_OPERATIONAL_TABLES
        )
        for table, column in profile_join_identities:
            columns = _table_columns(connection, table)
            if column not in columns or _OPAQUE_JSON_COLUMN_RE.search(column):
                raise ControlPlaneSchemaIdentityError(
                    f"operational join-critical identity missing or opaque: {table}.{column}"
                )
            report["join_critical_ok"].append(f"{table}.{column}")
        task_columns = _table_columns(connection, "tasks")
        for column in TASK_IDENTITY_COLUMNS:
            if column not in task_columns:
                raise ControlPlaneSchemaCompatibilityError(
                    f"operational profile is missing tasks.{column}"
                )
            report["task_columns_ok"].append(column)
        lease_columns = _table_columns(connection, "leases")
        for column in LEASE_IDENTITY_COLUMNS:
            if column not in lease_columns:
                raise ControlPlaneSchemaCompatibilityError(
                    f"operational profile is missing leases.{column}"
                )
            report["lease_columns_ok"].append(column)
        report["schema_fingerprint"] = compute_schema_fingerprint(connection)
        report["valid"] = True
    return report


def verify_installed_schema(
    database_path: Path | str,
    *,
    schema: ControlPlaneSchema | None = None,
) -> dict[str, Any]:
    """Verify domain tables, views, join-critical columns, and lease/task shapes."""

    inventory = schema or default_control_plane_schema()
    if not duckdb_available():
        raise ControlPlaneSchemaInstallError(
            "DuckDB is required to verify the control-plane schema"
        )
    report: dict[str, Any] = {
        "database_path": str(database_path),
        "schema_revision": inventory.schema_revision,
        "tables_ok": [],
        "views_ok": [],
        "join_critical_ok": [],
        "task_columns_ok": [],
        "lease_columns_ok": [],
        "opaque_json_only_identities": [],
    }
    with open_duckdb_connection(database_path) as connection:
        for table in inventory.bookkeeping_tables:
            if not _relation_exists(connection, table):
                raise ControlPlaneSchemaInstallError(f"bookkeeping table missing: {table}")
            report["tables_ok"].append(table)
        for table in inventory.all_domain_tables:
            if not _relation_exists(connection, table):
                raise ControlPlaneSchemaInstallError(f"domain table missing: {table}")
            report["tables_ok"].append(table)
        for view in inventory.diagnostic_views:
            if not _relation_exists(connection, view):
                raise ControlPlaneSchemaInstallError(f"diagnostic view missing: {view}")
            report["views_ok"].append(view)

        for table, column in inventory.join_critical_identities:
            columns = _table_columns(connection, table)
            if column not in columns:
                raise ControlPlaneSchemaIdentityError(
                    f"join-critical column {table}.{column} is missing"
                )
            if _OPAQUE_JSON_COLUMN_RE.search(column):
                raise ControlPlaneSchemaIdentityError(
                    f"join-critical identity {table}.{column} must not be an opaque JSON column"
                )
            report["join_critical_ok"].append(f"{table}.{column}")

        # Ensure no join-critical identity is represented only by a JSON twin.
        for table, column in inventory.join_critical_identities:
            columns = _table_columns(connection, table)
            json_twin = f"{column}_json"
            if json_twin in columns and column not in columns:
                report["opaque_json_only_identities"].append(f"{table}.{column}")
                raise ControlPlaneSchemaIdentityError(
                    f"join-critical identity {table}.{column} exists only as "
                    f"opaque JSON column {json_twin}"
                )

        task_columns = _table_columns(connection, "tasks")
        for column in inventory.task_identity_columns:
            if column not in task_columns:
                raise ControlPlaneSchemaCompatibilityError(
                    f"tasks.{column} required for task CID semantics is missing"
                )
            report["task_columns_ok"].append(column)

        lease_columns = _table_columns(connection, "leases")
        for column in inventory.lease_identity_columns:
            if column not in lease_columns:
                raise ControlPlaneSchemaCompatibilityError(
                    f"leases.{column} required for lease semantics is missing"
                )
            report["lease_columns_ok"].append(column)

        # task_cid must be the lease primary key surface (unique identity).
        if "task_cid" not in lease_columns:
            raise ControlPlaneSchemaCompatibilityError("leases.task_cid is required")
        report["schema_fingerprint"] = compute_schema_fingerprint(connection)
    return report


def verify_causal_event_federation_schema(
    database_path: Path | str,
    *,
    extension: CausalEventFederationSchemaExtension | None = None,
) -> dict[str, Any]:
    """Verify the additive federation inventory and authority-reference shape."""

    inventory = extension or default_causal_event_federation_schema_extension()
    if not duckdb_available():
        raise ControlPlaneSchemaInstallError(
            "DuckDB is required to verify the causal-event federation schema"
        )
    report: dict[str, Any] = {
        "database_path": str(database_path),
        "schema_revision": inventory.schema_revision,
        "migration_id": inventory.migration_id,
        "tables_ok": [],
        "join_critical_ok": [],
        "authority_reference_columns_ok": [],
        "section_9_relations_ok": [],
        "domain_event_columns_ok": [],
        "base_schema": verify_installed_schema(database_path),
    }
    with open_duckdb_connection(database_path) as connection:
        receipt = connection.execute(
            """
            SELECT migration_id, checksum
            FROM schema_migrations
            WHERE version = ?
            LIMIT 1
            """,
            [inventory.migration_version],
        ).fetchone()
        if receipt is None or str(receipt[0]) != inventory.migration_id:
            raise ControlPlaneSchemaInstallError(
                "causal-event federation migration receipt is missing or mismatched"
            )
        report["migration_checksum"] = str(receipt[1])

        contract = connection.execute(
            """
            SELECT payload_schema, schema_revision, description
            FROM schema_contracts
            WHERE contract_id =
                'contract:CausalEventFederationSchemaExtension@1'
            LIMIT 1
            """
        ).fetchone()
        if contract is None:
            raise ControlPlaneSchemaInstallError(
                "causal-event federation schema contract is missing"
            )
        if str(contract[0]) != inventory.SCHEMA or int(contract[1]) != 2:
            raise ControlPlaneSchemaInstallError("causal-event federation schema contract drifted")
        description = str(contract[2])
        if "ipfs_datasets_py" not in description or "reference" not in description:
            raise ControlPlaneSchemaInstallError(
                "federation schema contract does not preserve semantic ownership"
            )

        for table in inventory.tables:
            if not _relation_exists(connection, table):
                raise ControlPlaneSchemaInstallError(
                    f"causal-event federation table missing: {table}"
                )
            report["tables_ok"].append(table)

        for concept, relation in inventory.section_9_relations.items():
            if not _relation_exists(connection, relation):
                raise ControlPlaneSchemaInstallError(
                    f"Section 9 representation {concept} is missing relation {relation}"
                )
            report["section_9_relations_ok"].append(f"{concept}={relation}")

        for table, column in inventory.join_critical_identities:
            columns = _table_columns(connection, table)
            if column not in columns:
                raise ControlPlaneSchemaIdentityError(
                    f"federation join-critical column {table}.{column} is missing"
                )
            if _OPAQUE_JSON_COLUMN_RE.search(column):
                raise ControlPlaneSchemaIdentityError(
                    f"federation join-critical identity {table}.{column} must not "
                    "be an opaque JSON column"
                )
            report["join_critical_ok"].append(f"{table}.{column}")

        authority_columns = ("owner_id", "source_root", "content_ref")
        for table in inventory.reference_tables:
            columns = _table_columns(connection, table)
            for column in authority_columns:
                if column not in columns:
                    raise ControlPlaneSchemaIdentityError(
                        f"authority reference column {table}.{column} is missing"
                    )
                report["authority_reference_columns_ok"].append(f"{table}.{column}")

        required_event_columns = (
            "event_cid",
            "causal_parent_ids_json",
            "correlation_id",
            "causation_id",
            "tenant_id",
            "federation_id",
            "supervisor_id",
            "repository_id",
            "tree_id",
            "goal_id",
            "subgoal_id",
            "symbol_id",
            "contract_id",
            "proof_obligation_id",
            "resource_class",
            "payload_ref",
            "changed_fact_refs_json",
            "effect_class",
            "expires_at",
            "deduplication_key",
        )
        event_columns = _table_columns(connection, "domain_events")
        for column in required_event_columns:
            if column not in event_columns:
                raise ControlPlaneSchemaIdentityError(
                    f"domain_events.{column} required by federation events is missing"
                )
            report["domain_event_columns_ok"].append(column)

        report["schema_fingerprint"] = compute_schema_fingerprint(connection)
        report["valid"] = True
    return report


def assert_dependency_profile_pinned(
    pyproject_text: str,
    *,
    profile: DuckDBQuackDependencyProfile | None = None,
) -> None:
    """Fail closed unless ``pyproject.toml`` pins the supervisor DuckDB extra."""

    expected = profile or default_dependency_profile()
    text = str(pyproject_text)
    if f"{expected.extra_name}" not in text:
        raise ControlPlaneSchemaInstallError(
            f"pyproject.toml must declare optional extra {expected.extra_name!r}"
        )
    # Accept either the exact spec or an equivalent 1.5.x pin.
    has_spec = expected.duckdb_spec in text
    has_prefix_pin = (
        "duckdb" in text
        and "1.5" in text
        and ("<1.6" in text or "~=1.5" in text or "==1.5" in text)
    )
    if not (has_spec or has_prefix_pin):
        raise ControlPlaneSchemaInstallError(
            "pyproject.toml must pin DuckDB 1.5.x for the optional supervisor "
            f"service (expected {expected.duckdb_spec!r})"
        )
    if "agent-supervisor" not in text and expected.extra_name not in text:
        raise ControlPlaneSchemaInstallError(
            "pyproject.toml must name the optional supervisor service extra"
        )


def read_pyproject_text(path: Path | str | None = None) -> str:
    if path is None:
        path = Path(__file__).resolve().parents[3] / "pyproject.toml"
    return Path(path).read_text(encoding="utf-8")


__all__ = [
    "BOOKKEEPING_TABLES",
    "CAUSAL_EVENT_FEDERATION_JOIN_CRITICAL_IDENTITIES",
    "CAUSAL_EVENT_FEDERATION_MIGRATION_ID",
    "CAUSAL_EVENT_FEDERATION_MIGRATION_VERSION",
    "CAUSAL_EVENT_FEDERATION_REFERENCE_TABLES",
    "CAUSAL_EVENT_FEDERATION_SCHEMA_EXTENSION_INTERFACE",
    "CAUSAL_EVENT_FEDERATION_SCHEMA_EXTENSION_SCHEMA",
    "CAUSAL_EVENT_FEDERATION_SCHEMA_REVISION",
    "CAUSAL_EVENT_FEDERATION_SECTION_9_RELATIONS",
    "CAUSAL_EVENT_FEDERATION_SQL_FILENAME",
    "CAUSAL_EVENT_FEDERATION_TABLES",
    "CONTROL_PLANE_MIGRATION_ID",
    "CONTROL_PLANE_MIGRATION_VERSION",
    "CONTROL_PLANE_SCHEMA_INTERFACE",
    "CONTROL_PLANE_SCHEMA_REVISION",
    "CONTROL_PLANE_SCHEMA_SCHEMA",
    "CONTROL_PLANE_SCHEMA_VERSION",
    "CONTROL_PLANE_SQL_FILENAME",
    "CausalEventFederationSchemaExtension",
    "ControlPlaneSchema",
    "ControlPlaneSchemaCompatibilityError",
    "ControlPlaneSchemaError",
    "ControlPlaneSchemaIdentityError",
    "ControlPlaneSchemaInstallError",
    "DIAGNOSTIC_VIEWS",
    "DATASETS_AUTHORITATIVE_OPERATIONAL_DOMAINS",
    "DATASETS_AUTHORITATIVE_OPERATIONAL_EVIDENCE_TABLES",
    "DATASETS_AUTHORITATIVE_OPERATIONAL_MIGRATION_ID",
    "DATASETS_AUTHORITATIVE_OPERATIONAL_MIGRATION_VERSION",
    "DATASETS_AUTHORITATIVE_OPERATIONAL_PROFILE_ID",
    "DATASETS_AUTHORITATIVE_OPERATIONAL_SCHEMA",
    "DATASETS_AUTHORITATIVE_OPERATIONAL_TABLES",
    "DATASETS_SEMANTIC_TRUTH_RELATIONS",
    "DOMAIN_TABLES",
    "DuckDBQuackDependencyProfile",
    "JOIN_CRITICAL_IDENTITIES",
    "LEASE_IDENTITY_COLUMNS",
    "PINNED_DUCKDB_MAJOR",
    "PINNED_DUCKDB_MINOR",
    "PINNED_DUCKDB_VERSION_PREFIX",
    "PINNED_DUCKDB_VERSION_SPEC",
    "PINNED_PROFILE_ID",
    "PINNED_QUACK_EXTENSION",
    "PINNED_QUACK_EXTENSION_API",
    "SCHEMA_DOMAINS",
    "SUPERVISOR_OPTIONAL_EXTRA",
    "TASK_IDENTITY_COLUMNS",
    "assert_dependency_profile_pinned",
    "default_causal_event_federation_schema_extension",
    "default_control_plane_schema",
    "default_dependency_profile",
    "datasets_authoritative_operational_schema_sql",
    "install_datasets_authoritative_operational_schema",
    "install_control_plane_schema",
    "load_datasets_authoritative_operational_catalog",
    "load_control_plane_catalog",
    "package_sql_directory",
    "prove_fresh_and_upgraded_equivalence",
    "read_pyproject_text",
    "verify_datasets_authoritative_operational_schema",
    "verify_causal_event_federation_schema",
    "verify_installed_schema",
]
