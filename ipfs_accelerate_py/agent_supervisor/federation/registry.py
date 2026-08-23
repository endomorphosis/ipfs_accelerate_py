"""Typed federation registries over the canonical Quack state client.

This module never accepts a database path and never exposes arbitrary SQL.
Trusted static statement templates are installed once and the client catalog is
then sealed.  Every mutating registry method executes through
``StateTransaction.execute_command`` and appends its domain event plus outbox
row before the generation/idempotency records commit.
"""

# Python 3.8 compatibility requires ``datetime.timezone.utc`` and plain zip.
# ruff: noqa: UP017

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import MappingProxyType
from typing import Any

from ..merge.database_worktree_registry import process_birth_id, process_births_match
from ..merge.worktree_lifecycle import (
    ProcessBirthIdentity,
    current_process_birth,
    read_process_birth,
)
from ..task_sources.control_plane_contracts import (
    CommandKind,
    CommandOutcome,
    StateAuthorityClass,
    StateCommand,
    content_identity,
)
from ..task_sources.control_plane_transactions import CASResult, StateTransaction
from ..task_sources.quack_state_client import (
    QuackStateClient,
    StatementKind,
    StatementTemplate,
    TransportMode,
)
from .contracts import (
    BudgetDimensionName,
    BudgetReservation,
    FederationAuthorityError,
    FederationAuthorizationDecision,
    FederationBinding,
    FederationBoundsError,
    FederationContractError,
    FederationIdentity,
    FederationLifecycleState,
    FederationPolicy,
    FederationReceipt,
    FederationRequest,
    SubagentAssignment,
    SubagentCapability,
    SubagentDefinition,
    SubagentInstance,
    SubagentOutcome,
    SupervisorAssignment,
    SupervisorCapability,
    SupervisorDefinition,
    SupervisorInstance,
    SupervisorRole,
    _identifier,
    _integer,
    _timestamp,
    utc_now,
)
from .durable_event_router import (
    CoalescingCoverageRecord,
    DeadLetterRetryCommit,
    DurableDeliveryFailure,
    DurableFailureCommit,
    DurableQueuedDelivery,
    DurableRouteBatch,
    DurableRouteCommit,
    DurableRoutingBackpressure,
    DurableRoutingState,
    DurableSubscriptionRoutingState,
)
from .event_router import CoalescingDecision, CoalescingMode, FailureResult, QueuedDelivery
from .event_wait import StaleSubscriptionError
from .events import (
    ConsumerCursor,
    DeadLetter,
    DeliveryAttempt,
    DeliveryState,
    DomainEvent,
    EventAcknowledgement,
    EventClass,
    EventEffectClass,
    EventSelector,
    EventSubscription,
    SelectorKind,
    SubscriptionState,
)
from .lifecycle import assert_transition
from .outbox import EventDraft, OutboxRecord, materialize_event
from .outbox_worker import OutboxDisposition, OutboxScope
from .subscriptions import event_matches_subscription
from .trigger import ResolvedRepository, resolved_authorization_scope_identity


class FederationRepositoryError(RuntimeError):
    """Base typed repository failure."""


class FederationRepositoryConflict(FederationRepositoryError):
    """A population, revision, fence, or idempotency invariant conflicted."""


class FederationRepositoryNotFound(FederationRepositoryError):
    """A required federation-scoped record is absent."""


_EVENT_SCAN_PAGE_SIZE = 256
_MAX_EVENT_SCAN_CANDIDATES = 65_536


def _json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _decode(value: Any) -> Any:
    if isinstance(value, str):
        return json.loads(value)
    return value


def _expired(value: str) -> bool:
    """Compare one validated RFC3339 authority deadline with server time."""

    validated = _timestamp(value, "expires_at")
    deadline = datetime.fromisoformat(validated.replace("Z", "+00:00"))
    return deadline <= datetime.now(timezone.utc)


def _scoped_idempotency_key(
    *,
    operation: str,
    tenant_id: str,
    federation_id: str,
    caller_key: str,
) -> str:
    return "casf-idempotency:" + content_identity(
        {
            "operation": _identifier(operation, "operation"),
            "tenant_id": _identifier(tenant_id, "tenant_id"),
            "federation_id": _identifier(federation_id, "federation_id"),
            "caller_key": _identifier(caller_key, "idempotency_key"),
        }
    )


def _template(
    name: str,
    sql: str,
    parameters: Sequence[str],
    *,
    kind: StatementKind = StatementKind.MUTATION,
) -> StatementTemplate:
    return StatementTemplate(
        name=name,
        sql=sql,
        parameter_names=tuple(parameters),
        kind=kind,
        description=f"sealed CASF statement {name}",
    )


def _casf_templates() -> tuple[StatementTemplate, ...]:
    """Return the closed CASF statement catalog (no caller SQL)."""

    return (
        _template(
            "casf_insert_federation",
            """
            INSERT INTO federations (
                federation_id, tenant_id, program_id, objective_ref,
                objective_revision, policy_id, policy_revision,
                operation_catalog_id, control_plane_generation,
                causal_graph_revision, semantic_state_root, status,
                maximum_supervisors, maximum_subagents, revision,
                fencing_epoch, issuer_id, authorization_evidence_ref,
                expires_at, created_at, updated_at, content_ref, body_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "federation_id",
                "tenant_id",
                "program_id",
                "objective_ref",
                "objective_revision",
                "policy_id",
                "policy_revision",
                "operation_catalog_id",
                "control_plane_generation",
                "causal_graph_revision",
                "semantic_state_root",
                "status",
                "maximum_supervisors",
                "maximum_subagents",
                "revision",
                "fencing_epoch",
                "issuer_id",
                "authorization_evidence_ref",
                "expires_at",
                "created_at",
                "updated_at",
                "content_ref",
                "body_json",
            ),
        ),
        _template(
            "casf_insert_policy",
            """
            INSERT INTO federation_policies (
                policy_id, tenant_id, federation_id, revision, issuer_id,
                authorization_evidence_ref, expires_at,
                maximum_supervisors, maximum_subagents,
                maximum_concurrent_subagents, status, content_ref,
                created_at, updated_at, body_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (policy_id) DO NOTHING
            """,
            (
                "policy_id",
                "tenant_id",
                "federation_id",
                "revision",
                "issuer_id",
                "authorization_evidence_ref",
                "expires_at",
                "maximum_supervisors",
                "maximum_subagents",
                "maximum_concurrent_subagents",
                "status",
                "content_ref",
                "created_at",
                "updated_at",
                "body_json",
            ),
        ),
        _template(
            "casf_insert_authorization_decision",
            """
            INSERT INTO federation_authorization_decisions (
                authorization_decision_id, tenant_id, federation_id,
                request_cid, caller_id, delegation_chain_ref, audience,
                operation, resource_scope_ref, policy_id, policy_revision,
                verdict, reason_code, evidence_ref, expires_at, decided_at,
                content_ref, body_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "authorization_decision_id",
                "tenant_id",
                "federation_id",
                "request_cid",
                "caller_id",
                "delegation_chain_ref",
                "audience",
                "operation",
                "resource_scope_ref",
                "policy_id",
                "policy_revision",
                "verdict",
                "reason_code",
                "evidence_ref",
                "expires_at",
                "decided_at",
                "content_ref",
                "body_json",
            ),
        ),
        _template(
            "casf_select_authorization_decision",
            """
            SELECT authorization_decision_id, tenant_id, federation_id,
                   request_cid, caller_id, delegation_chain_ref, audience,
                   operation, resource_scope_ref, policy_id, policy_revision,
                   verdict, reason_code, evidence_ref, expires_at, decided_at,
                   content_ref, body_json
            FROM federation_authorization_decisions
            WHERE federation_id = ? AND tenant_id = ?
            ORDER BY decided_at, authorization_decision_id
            """,
            ("federation_id", "tenant_id"),
            kind=StatementKind.QUERY,
        ),
        _template(
            "casf_insert_federation_budget",
            """
            INSERT INTO federation_budgets (
                federation_budget_id, tenant_id, federation_id,
                parent_budget_id, owner_id, policy_id, policy_revision,
                revision, status, content_ref, created_at, updated_at, body_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "federation_budget_id",
                "tenant_id",
                "federation_id",
                "parent_budget_id",
                "owner_id",
                "policy_id",
                "policy_revision",
                "revision",
                "status",
                "content_ref",
                "created_at",
                "updated_at",
                "body_json",
            ),
        ),
        _template(
            "casf_select_admission_budget_by_idempotency",
            """
            SELECT reservation_id, tenant_id, federation_id, request_cid,
                   idempotency_key, policy_id, policy_revision,
                   resource_budget_id, token_budget_id, parent_budget_id,
                   owner_id, authorization_evidence_ref, issued_at, expires_at,
                   state, revision, fencing_epoch, content_ref, body_json
            FROM federation_admission_budget_reservations
            WHERE tenant_id = ? AND federation_id = ? AND idempotency_key = ?
            LIMIT 1
            """,
            ("tenant_id", "federation_id", "idempotency_key"),
            kind=StatementKind.QUERY,
        ),
        _template(
            "casf_select_admission_budget_by_id",
            """
            SELECT reservation_id, tenant_id, federation_id, request_cid,
                   idempotency_key, policy_id, policy_revision,
                   resource_budget_id, token_budget_id, parent_budget_id,
                   owner_id, authorization_evidence_ref, issued_at, expires_at,
                   state, revision, fencing_epoch, content_ref, body_json
            FROM federation_admission_budget_reservations
            WHERE reservation_id = ? AND tenant_id = ? AND federation_id = ?
            LIMIT 1
            """,
            ("reservation_id", "tenant_id", "federation_id"),
            kind=StatementKind.QUERY,
        ),
        _template(
            "casf_select_active_admission_budget_usage",
            """
            SELECT dimensions.dimension_name,
                   COALESCE(SUM(dimensions.reserved_amount), 0) AS reserved_amount
            FROM federation_admission_budget_dimensions AS dimensions
            JOIN federation_admission_budget_reservations AS reservations
              ON reservations.reservation_id = dimensions.reservation_id
            WHERE reservations.tenant_id = ?
              AND reservations.state = 'reserved'
              AND reservations.expires_at > ?
            GROUP BY dimensions.dimension_name
            """,
            ("tenant_id", "observed_at"),
            kind=StatementKind.QUERY,
        ),
        _template(
            "casf_insert_admission_budget_reservation",
            """
            INSERT INTO federation_admission_budget_reservations (
                reservation_id, tenant_id, federation_id, request_cid,
                idempotency_key, policy_id, policy_revision,
                resource_budget_id, token_budget_id, parent_budget_id,
                owner_id, authorization_evidence_ref, issued_at, expires_at,
                state, revision, fencing_epoch, content_ref, body_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "reservation_id", "tenant_id", "federation_id", "request_cid",
                "idempotency_key", "policy_id", "policy_revision",
                "resource_budget_id", "token_budget_id", "parent_budget_id",
                "owner_id", "authorization_evidence_ref", "issued_at",
                "expires_at", "state", "revision", "fencing_epoch",
                "content_ref", "body_json",
            ),
        ),
        _template(
            "casf_insert_admission_budget_dimension",
            """
            INSERT INTO federation_admission_budget_dimensions (
                reservation_id, tenant_id, federation_id, dimension_name,
                ceiling_amount, reserved_amount, consumed_amount, ordinal
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "reservation_id", "tenant_id", "federation_id",
                "dimension_name", "ceiling_amount", "reserved_amount",
                "consumed_amount", "ordinal",
            ),
        ),
        _template(
            "casf_transition_admission_budget_reservation",
            """
            UPDATE federation_admission_budget_reservations
            SET state = ?, revision = revision + 1, content_ref = ?, body_json = ?
            WHERE reservation_id = ? AND tenant_id = ? AND federation_id = ?
              AND idempotency_key = ? AND state = ? AND revision = ?
              AND fencing_epoch = ?
            RETURNING revision
            """,
            (
                "new_state", "content_ref", "body_json", "reservation_id",
                "tenant_id", "federation_id", "idempotency_key",
                "expected_state", "expected_revision", "fencing_epoch",
            ),
        ),
        _template(
            "casf_select_federation",
            """
            SELECT federation_id, tenant_id, program_id, objective_ref,
                   objective_revision, policy_id, policy_revision,
                   operation_catalog_id, control_plane_generation,
                   causal_graph_revision, status, maximum_supervisors,
                   maximum_subagents, revision, fencing_epoch, issuer_id,
                   authorization_evidence_ref, expires_at, body_json
            FROM federations
            WHERE federation_id = ? AND tenant_id = ?
            LIMIT 1
            """,
            ("federation_id", "tenant_id"),
            kind=StatementKind.QUERY,
        ),
        _template(
            "casf_seed_global_head",
            """
            INSERT INTO global_sequence_head (
                head_id, current_sequence, revision, updated_at
            )
            SELECT 'global', COALESCE(MAX(global_sequence), 0), 0, ?
            FROM domain_events
            ON CONFLICT (head_id) DO UPDATE SET
                current_sequence = GREATEST(
                    global_sequence_head.current_sequence,
                    excluded.current_sequence
                ),
                revision = CASE
                    WHEN excluded.current_sequence
                         > global_sequence_head.current_sequence
                    THEN global_sequence_head.revision + 1
                    ELSE global_sequence_head.revision
                END,
                updated_at = CASE
                    WHEN excluded.current_sequence
                         > global_sequence_head.current_sequence
                    THEN excluded.updated_at
                    ELSE global_sequence_head.updated_at
                END
            """,
            ("updated_at",),
        ),
        _template(
            "casf_advance_global_head",
            """
            UPDATE global_sequence_head
            SET current_sequence = current_sequence + 1,
                revision = revision + 1,
                updated_at = ?
            WHERE head_id = 'global'
            RETURNING current_sequence
            """,
            ("updated_at",),
        ),
        _template(
            "casf_seed_stream_head",
            """
            INSERT INTO stream_sequence_heads (
                stream_id, tenant_id, federation_id,
                current_sequence, revision, updated_at
            ) VALUES (?, ?, ?, 0, 0, ?)
            ON CONFLICT (stream_id) DO NOTHING
            """,
            ("stream_id", "tenant_id", "federation_id", "updated_at"),
        ),
        _template(
            "casf_advance_stream_head",
            """
            UPDATE stream_sequence_heads
            SET current_sequence = current_sequence + 1,
                revision = revision + 1,
                updated_at = ?
            WHERE stream_id = ? AND tenant_id = ? AND federation_id = ?
            RETURNING current_sequence
            """,
            ("updated_at", "stream_id", "tenant_id", "federation_id"),
        ),
        _template(
            "casf_insert_domain_event",
            """
            INSERT INTO domain_events (
                event_id, event_cid, stream_id, sequence, global_sequence,
                event_type, task_cid, attempt_id, session_id, recorded_at,
                body_json, causal_parent_ids_json, correlation_id,
                causation_id, tenant_id, federation_id, supervisor_id,
                repository_id, tree_id, goal_id, subgoal_id, symbol_id,
                contract_id, proof_obligation_id, resource_class,
                payload_ref, changed_fact_refs_json, effect_class, expires_at,
                deduplication_key, control_plane_generation,
                causal_graph_revision
            ) VALUES (?, ?, ?, ?, ?, ?, ?, '', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "event_id",
                "event_cid",
                "stream_id",
                "stream_sequence",
                "global_sequence",
                "event_type",
                "task_cid",
                "session_id",
                "recorded_at",
                "body_json",
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
                "control_plane_generation",
                "causal_graph_revision",
            ),
        ),
        _template(
            "casf_insert_event_parent",
            """
            INSERT INTO domain_event_causal_parents (
                event_id, parent_event_id, ordinal
            ) VALUES (?, ?, ?)
            """,
            ("event_id", "parent_event_id", "ordinal"),
        ),
        _template(
            "casf_insert_changed_fact",
            """
            INSERT INTO domain_event_changed_facts (
                event_id, fact_ref, ordinal
            ) VALUES (?, ?, ?)
            """,
            ("event_id", "fact_ref", "ordinal"),
        ),
        _template(
            "casf_insert_outbox",
            """
            INSERT INTO transactional_outbox (
                outbox_id, event_id, event_cid, tenant_id, federation_id,
                stream_id, stream_sequence, global_sequence, effect_class,
                deduplication_key, status, attempt_count, next_attempt_at,
                claimed_by, claim_fencing_epoch, projected_at, revision,
                created_at, updated_at, body_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '', 0, NULL, 1, ?, ?, ?)
            """,
            (
                "outbox_id",
                "event_id",
                "event_cid",
                "tenant_id",
                "federation_id",
                "stream_id",
                "stream_sequence",
                "global_sequence",
                "effect_class",
                "deduplication_key",
                "status",
                "attempt_count",
                "next_attempt_at",
                "created_at",
                "updated_at",
                "body_json",
            ),
        ),
        _template(
            "casf_list_pending_outbox_scopes",
            """
            SELECT tenant_id, federation_id,
                   MIN(global_sequence) AS first_global_sequence
            FROM transactional_outbox
            WHERE status IN ('pending', 'retry') AND next_attempt_at <= ?
            GROUP BY tenant_id, federation_id
            ORDER BY first_global_sequence ASC, tenant_id ASC, federation_id ASC
            LIMIT ?
            """,
            ("observed_at", "limit"),
            kind=StatementKind.QUERY,
        ),
        _template(
            "casf_list_pending_outbox_events",
            """
            SELECT events.event_id, events.event_cid, events.event_type,
                   events.stream_id, events.sequence, events.global_sequence,
                   events.causal_parent_ids_json, events.correlation_id,
                   events.causation_id, events.tenant_id, events.federation_id,
                   events.supervisor_id, events.task_cid AS task_id,
                   events.repository_id, events.tree_id, events.goal_id,
                   events.subgoal_id, events.symbol_id, events.contract_id,
                   events.proof_obligation_id, events.resource_class,
                   events.payload_ref, events.changed_fact_refs_json,
                   events.effect_class, events.recorded_at, events.expires_at,
                   events.deduplication_key
            FROM transactional_outbox AS outbox
            INNER JOIN domain_events AS events
              ON events.event_id = outbox.event_id
             AND events.event_cid = outbox.event_cid
             AND events.tenant_id = outbox.tenant_id
             AND events.federation_id = outbox.federation_id
             AND events.global_sequence = outbox.global_sequence
            WHERE outbox.tenant_id = ? AND outbox.federation_id = ?
              AND outbox.status IN ('pending', 'retry')
              AND outbox.next_attempt_at <= ?
            ORDER BY outbox.global_sequence ASC, outbox.event_id ASC
            LIMIT ?
            """,
            ("tenant_id", "federation_id", "observed_at", "limit"),
            kind=StatementKind.QUERY,
        ),
        _template(
            "casf_list_active_subscription_ids",
            """
            SELECT subscription_id
            FROM event_subscriptions
            WHERE tenant_id = ? AND federation_id = ? AND status = 'active'
              AND (expires_at IS NULL OR expires_at = '' OR expires_at > ?)
            ORDER BY subscription_id ASC
            LIMIT ?
            """,
            ("tenant_id", "federation_id", "observed_at", "limit"),
            kind=StatementKind.QUERY,
        ),
        _template(
            "casf_select_pending_outbox_event",
            """
            SELECT outbox_id, event_id, event_cid, global_sequence, status,
                   revision
            FROM transactional_outbox
            WHERE tenant_id = ? AND federation_id = ? AND event_id = ?
              AND status IN ('pending', 'retry')
            LIMIT 1
            """,
            ("tenant_id", "federation_id", "event_id"),
            kind=StatementKind.QUERY,
        ),
        _template(
            "casf_insert_outbox_routing_disposition",
            """
            INSERT INTO outbox_routing_dispositions (
                disposition_id, tenant_id, federation_id, route_batch_id,
                first_global_sequence, last_global_sequence, event_count,
                delivery_count, subscription_count, status, revision,
                content_ref, created_at, body_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'committed', 1, ?, ?, ?)
            """,
            (
                "disposition_id",
                "tenant_id",
                "federation_id",
                "route_batch_id",
                "first_global_sequence",
                "last_global_sequence",
                "event_count",
                "delivery_count",
                "subscription_count",
                "content_ref",
                "created_at",
                "body_json",
            ),
        ),
        _template(
            "casf_insert_outbox_routing_disposition_event",
            """
            INSERT INTO outbox_routing_disposition_events (
                disposition_id, event_id, global_sequence, ordinal
            ) VALUES (?, ?, ?, ?)
            """,
            ("disposition_id", "event_id", "global_sequence", "ordinal"),
        ),
        _template(
            "casf_mark_outbox_routed",
            """
            UPDATE transactional_outbox
            SET status = 'routed', revision = revision + 1, updated_at = ?
            WHERE outbox_id = ? AND event_id = ? AND event_cid = ?
              AND tenant_id = ? AND federation_id = ? AND global_sequence = ?
              AND revision = ? AND status IN ('pending', 'retry')
            RETURNING revision
            """,
            (
                "updated_at",
                "outbox_id",
                "event_id",
                "event_cid",
                "tenant_id",
                "federation_id",
                "global_sequence",
                "expected_revision",
            ),
        ),
        _template(
            "casf_count_supervisors",
            """
            SELECT COUNT(*) AS population
            FROM supervisor_instances
            WHERE tenant_id = ? AND federation_id = ?
              AND lifecycle_state NOT IN ('FAILED', 'STOPPED')
            """,
            ("tenant_id", "federation_id"),
            kind=StatementKind.QUERY,
        ),
        _template(
            "casf_count_coordinator_supervisors",
            """
            SELECT COUNT(*) AS population
            FROM supervisor_instances
            WHERE tenant_id = ? AND federation_id = ? AND role = 'coordinator'
              AND lifecycle_state NOT IN ('FAILED', 'STOPPED')
            """,
            ("tenant_id", "federation_id"),
            kind=StatementKind.QUERY,
        ),
        _template(
            "casf_select_supervisor",
            """
            SELECT supervisor_id, tenant_id, federation_id,
                   parent_supervisor_id, role, lifecycle_state,
                   lease_id, fencing_epoch, revision, extension_json,
                   supervisor_definition_id, assignment_revision,
                   policy_id, policy_revision, process_birth_id
            FROM supervisor_instances
            WHERE supervisor_id = ? AND tenant_id = ? AND federation_id = ?
            LIMIT 1
            """,
            ("supervisor_id", "tenant_id", "federation_id"),
            kind=StatementKind.QUERY,
        ),
        _template(
            "casf_select_supervisor_admission_authority",
            """
            SELECT instances.supervisor_id, instances.lifecycle_state,
                   instances.fencing_epoch, instances.lease_id,
                   instances.assignment_revision,
                   definitions.body_json AS definition_json,
                   assignments.body_json AS assignment_json,
                   assignments.status AS assignment_status,
                   assignments.fencing_epoch AS assignment_fencing_epoch,
                   assignments.assignment_revision,
                   definitions.policy_id, definitions.policy_revision
            FROM supervisor_instances AS instances
            INNER JOIN supervisor_definitions AS definitions
              ON definitions.supervisor_definition_id =
                 instances.supervisor_definition_id
             AND definitions.tenant_id = instances.tenant_id
             AND definitions.federation_id = instances.federation_id
            INNER JOIN supervisor_assignments AS assignments
              ON assignments.supervisor_id = instances.supervisor_id
             AND assignments.tenant_id = instances.tenant_id
             AND assignments.federation_id = instances.federation_id
             AND assignments.assignment_revision = instances.assignment_revision
            WHERE instances.supervisor_id = ? AND instances.tenant_id = ?
              AND instances.federation_id = ?
            LIMIT 1
            """,
            ("supervisor_id", "tenant_id", "federation_id"),
            kind=StatementKind.QUERY,
        ),
        _template(
            "casf_insert_supervisor",
            """
            INSERT INTO supervisor_instances (
                supervisor_id, repository_id, process_birth_id, started_at,
                stopped_at, status, revision, extension_schema, extension_json,
                tenant_id, federation_id, parent_supervisor_id,
                supervisor_definition_id, role, lifecycle_state,
                assignment_revision, lease_id, fencing_epoch,
                policy_id, policy_revision, admission_decision_id
            ) VALUES (?, ?, ?, ?, NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "supervisor_id",
                "repository_id",
                "process_birth_id",
                "started_at",
                "status",
                "revision",
                "extension_schema",
                "extension_json",
                "tenant_id",
                "federation_id",
                "parent_supervisor_id",
                "supervisor_definition_id",
                "role",
                "lifecycle_state",
                "assignment_revision",
                "lease_id",
                "fencing_epoch",
                "policy_id",
                "policy_revision",
                "admission_decision_id",
            ),
        ),
        _template(
            "casf_insert_supervisor_assignment",
            """
            INSERT INTO supervisor_assignments (
                assignment_id, tenant_id, federation_id, supervisor_id,
                parent_supervisor_id, assignment_revision, repository_id,
                tree_id, goal_ref, subgoal_ref, task_family, shard_id,
                lease_id, fencing_epoch, status, admission_decision_id, content_ref,
                created_at, updated_at, body_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, '', ?, '', ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "assignment_id",
                "tenant_id",
                "federation_id",
                "supervisor_id",
                "parent_supervisor_id",
                "assignment_revision",
                "repository_id",
                "tree_id",
                "goal_ref",
                "task_family",
                "lease_id",
                "fencing_epoch",
                "status",
                "admission_decision_id",
                "content_ref",
                "created_at",
                "updated_at",
                "body_json",
            ),
        ),
        _template(
            "casf_update_supervisor_lifecycle",
            """
            UPDATE supervisor_instances
            SET lifecycle_state = ?, status = ?, revision = revision + 1,
                fencing_epoch = ?
            WHERE supervisor_id = ? AND tenant_id = ? AND federation_id = ?
              AND revision = ? AND fencing_epoch = ?
            RETURNING revision
            """,
            (
                "lifecycle_state",
                "status",
                "new_fencing_epoch",
                "supervisor_id",
                "tenant_id",
                "federation_id",
                "expected_revision",
                "expected_fencing_epoch",
            ),
        ),
        _template(
            "casf_insert_process_birth_attestation",
            """
            INSERT INTO process_births (
                process_birth_id, tenant_id, federation_id, supervisor_id,
                subagent_id, process_id, start_marker, executable_ref,
                host_identity_ref, status, started_at, stopped_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, NULL)
            ON CONFLICT (process_birth_id) DO NOTHING
            """,
            (
                "process_birth_id", "tenant_id", "federation_id",
                "supervisor_id", "subagent_id", "process_id",
                "start_marker", "executable_ref", "host_identity_ref",
                "started_at",
            ),
        ),
        _template(
            "casf_insert_supervisor_runtime_lease",
            """
            INSERT INTO supervisor_runtime_leases (
                runtime_lease_id, tenant_id, federation_id, supervisor_id,
                lease_id, process_birth_id, process_id,
                process_start_time_ticks, process_boot_id, process_parent_id,
                issued_at, expires_at, revoked_at, fencing_epoch, revision,
                status, evidence_ref, body_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?,
                      'active', ?, ?)
            """,
            (
                "runtime_lease_id", "tenant_id", "federation_id",
                "supervisor_id", "lease_id", "process_birth_id",
                "process_id", "process_start_time_ticks", "process_boot_id",
                "process_parent_id", "issued_at", "expires_at",
                "fencing_epoch", "revision", "evidence_ref", "body_json",
            ),
        ),
        _template(
            "casf_select_latest_supervisor_runtime_revision",
            """
            SELECT runtime_lease_id, process_birth_id, revision, status
            FROM supervisor_runtime_leases
            WHERE tenant_id = ? AND federation_id = ?
              AND supervisor_id = ? AND lease_id = ? AND fencing_epoch = ?
            ORDER BY revision DESC
            LIMIT 1
            """,
            (
                "tenant_id", "federation_id", "supervisor_id", "lease_id",
                "fencing_epoch",
            ),
            kind=StatementKind.QUERY,
        ),
        _template(
            "casf_supersede_supervisor_runtime_lease",
            """
            UPDATE supervisor_runtime_leases
            SET status = 'superseded', revoked_at = ?
            WHERE runtime_lease_id = ? AND tenant_id = ? AND federation_id = ?
              AND supervisor_id = ? AND revision = ? AND status = 'active'
              AND revoked_at IS NULL
            RETURNING runtime_lease_id
            """,
            (
                "revoked_at", "runtime_lease_id", "tenant_id",
                "federation_id", "supervisor_id", "expected_revision",
            ),
        ),
        _template(
            "casf_update_supervisor_process_birth",
            """
            UPDATE supervisor_instances
            SET process_birth_id = ?
            WHERE supervisor_id = ? AND tenant_id = ? AND federation_id = ?
              AND revision = ? AND fencing_epoch = ?
              AND process_birth_id IN ('logical:not-started', ?)
            RETURNING process_birth_id
            """,
            (
                "process_birth_id", "supervisor_id", "tenant_id",
                "federation_id", "expected_revision", "fencing_epoch",
                "current_process_birth_id",
            ),
        ),
        _template(
            "casf_select_current_supervisor_runtime",
            """
            SELECT leases.runtime_lease_id, leases.lease_id,
                   leases.process_birth_id, leases.process_id,
                   leases.process_start_time_ticks, leases.process_boot_id,
                   leases.process_parent_id, leases.issued_at,
                   leases.expires_at, leases.fencing_epoch, leases.revision,
                   leases.status, leases.evidence_ref,
                   births.process_id AS birth_process_id,
                   births.start_marker, births.host_identity_ref,
                   births.status AS birth_status, births.stopped_at
            FROM supervisor_runtime_leases AS leases
            INNER JOIN process_births AS births
              ON births.process_birth_id = leases.process_birth_id
             AND births.tenant_id = leases.tenant_id
             AND births.federation_id = leases.federation_id
             AND births.supervisor_id = leases.supervisor_id
            WHERE leases.tenant_id = ? AND leases.federation_id = ?
              AND leases.supervisor_id = ? AND leases.lease_id = ?
              AND leases.fencing_epoch = ? AND leases.status = 'active'
              AND leases.revoked_at IS NULL AND leases.expires_at > ?
              AND births.status = 'active' AND births.stopped_at IS NULL
            ORDER BY leases.revision DESC
            LIMIT 2
            """,
            (
                "tenant_id", "federation_id", "supervisor_id", "lease_id",
                "fencing_epoch", "observed_at",
            ),
            kind=StatementKind.QUERY,
        ),
        _template(
            "casf_select_supervisor_bootstrap_health",
            """
            SELECT supervisors.lifecycle_state,
                   supervisors.fencing_epoch AS supervisor_fencing_epoch,
                   supervisors.process_birth_id,
                   leases.runtime_lease_id, leases.lease_id,
                   leases.process_id, leases.process_start_time_ticks,
                   leases.process_boot_id, leases.process_parent_id,
                   leases.issued_at AS runtime_issued_at,
                   leases.expires_at AS runtime_expires_at,
                   leases.fencing_epoch AS runtime_fencing_epoch,
                   leases.revision AS runtime_revision,
                   leases.evidence_ref AS runtime_evidence_ref,
                   subscriptions.subscription_id,
                   subscriptions.revision AS subscription_revision,
                   subscriptions.status AS subscription_status,
                   cursors.consumer_id,
                   cursors.global_sequence AS cursor_global_sequence,
                   cursors.last_event_id,
                   cursors.revision AS cursor_revision,
                   acknowledgements.acknowledgement_id,
                   acknowledgements.event_id AS acknowledged_event_id,
                   acknowledgements.delivery_attempt_id,
                   acknowledgements.global_sequence AS acknowledged_global_sequence,
                   attempts.status AS delivery_attempt_status,
                   queue.status AS delivery_queue_status,
                   events.event_type AS acknowledged_event_type,
                   events.recorded_at AS acknowledged_event_recorded_at,
                   (
                       SELECT COUNT(*)
                       FROM event_delivery_queue AS pending_queue
                       INNER JOIN domain_events AS pending_events
                         ON pending_events.event_id = pending_queue.representative_event_id
                        AND pending_events.tenant_id = pending_queue.tenant_id
                        AND pending_events.federation_id = pending_queue.federation_id
                       WHERE pending_queue.tenant_id = supervisors.tenant_id
                         AND pending_queue.federation_id = supervisors.federation_id
                         AND pending_queue.subscription_id = subscriptions.subscription_id
                         AND pending_queue.subscription_revision = subscriptions.revision
                         AND pending_queue.consumer_id = subscriptions.consumer_id
                         AND pending_queue.status IN ('pending', 'retry', 'delivered')
                   ) AS pending_required_deliveries
            FROM supervisor_instances AS supervisors
            INNER JOIN supervisor_runtime_leases AS leases
              ON leases.tenant_id = supervisors.tenant_id
             AND leases.federation_id = supervisors.federation_id
             AND leases.supervisor_id = supervisors.supervisor_id
             AND leases.lease_id = supervisors.lease_id
             AND leases.fencing_epoch = supervisors.fencing_epoch
             AND leases.process_birth_id = supervisors.process_birth_id
            INNER JOIN event_subscriptions AS subscriptions
              ON subscriptions.tenant_id = supervisors.tenant_id
             AND subscriptions.federation_id = supervisors.federation_id
             AND subscriptions.supervisor_id = supervisors.supervisor_id
            INNER JOIN consumer_cursors AS cursors
              ON cursors.tenant_id = subscriptions.tenant_id
             AND cursors.federation_id = subscriptions.federation_id
             AND cursors.subscription_id = subscriptions.subscription_id
             AND cursors.subscription_revision = subscriptions.revision
             AND cursors.consumer_id = subscriptions.consumer_id
             AND cursors.fencing_epoch = supervisors.fencing_epoch
            INNER JOIN event_acknowledgements AS acknowledgements
              ON acknowledgements.tenant_id = cursors.tenant_id
             AND acknowledgements.federation_id = cursors.federation_id
             AND acknowledgements.subscription_id = cursors.subscription_id
             AND acknowledgements.subscription_revision = cursors.subscription_revision
             AND acknowledgements.consumer_id = cursors.consumer_id
             AND acknowledgements.fencing_epoch = cursors.fencing_epoch
            INNER JOIN delivery_attempts AS attempts
              ON attempts.attempt_id = acknowledgements.delivery_attempt_id
             AND attempts.tenant_id = acknowledgements.tenant_id
             AND attempts.federation_id = acknowledgements.federation_id
             AND attempts.event_id = acknowledgements.event_id
             AND attempts.subscription_id = acknowledgements.subscription_id
             AND attempts.subscription_revision = acknowledgements.subscription_revision
             AND attempts.consumer_id = acknowledgements.consumer_id
             AND attempts.fencing_epoch = acknowledgements.fencing_epoch
            INNER JOIN event_delivery_queue AS queue
              ON queue.delivery_id = attempts.delivery_id
             AND queue.tenant_id = attempts.tenant_id
             AND queue.federation_id = attempts.federation_id
             AND queue.subscription_id = attempts.subscription_id
             AND queue.subscription_revision = attempts.subscription_revision
             AND queue.consumer_id = attempts.consumer_id
             AND queue.fencing_epoch = attempts.fencing_epoch
            INNER JOIN domain_events AS events
              ON events.event_id = acknowledgements.event_id
             AND events.tenant_id = acknowledgements.tenant_id
             AND events.federation_id = acknowledgements.federation_id
             AND events.global_sequence = acknowledgements.global_sequence
            WHERE supervisors.tenant_id = ?
              AND supervisors.federation_id = ?
              AND supervisors.supervisor_id = ?
              AND subscriptions.subscription_id = ?
              AND subscriptions.consumer_id = ?
              AND acknowledgements.event_id = ?
              AND acknowledgements.acknowledgement_id = ?
              AND acknowledgements.delivery_attempt_id = ?
              AND leases.status = 'active' AND leases.revoked_at IS NULL
              AND leases.expires_at > ?
              AND subscriptions.status = 'active'
              AND attempts.status = 'acknowledged'
              AND queue.status = 'acknowledged'
            ORDER BY leases.revision DESC
            LIMIT 2
            """,
            (
                "tenant_id", "federation_id", "supervisor_id",
                "subscription_id", "consumer_id", "event_id",
                "acknowledgement_id", "delivery_attempt_id", "observed_at",
            ),
            kind=StatementKind.QUERY,
        ),
        _template(
            "casf_count_subagents",
            """
            SELECT COUNT(*) AS population
            FROM subagent_instances
            WHERE tenant_id = ? AND federation_id = ?
            """,
            ("tenant_id", "federation_id"),
            kind=StatementKind.QUERY,
        ),
        _template(
            "casf_insert_subagent",
            """
            INSERT INTO subagent_instances (
                subagent_id, tenant_id, federation_id, supervisor_id,
                subagent_definition_id, task_id, lease_id, logical_state,
                admitted_concurrency_slot, worker_process_birth_id,
                provider_route_id, admission_decision_id, revision, fencing_epoch,
                registered_at, updated_at, body_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "subagent_id",
                "tenant_id",
                "federation_id",
                "supervisor_id",
                "subagent_definition_id",
                "task_id",
                "lease_id",
                "logical_state",
                "admitted_concurrency_slot",
                "worker_process_birth_id",
                "provider_route_id",
                "admission_decision_id",
                "revision",
                "fencing_epoch",
                "registered_at",
                "updated_at",
                "body_json",
            ),
        ),
        _template(
            "casf_select_subagent",
            """
            SELECT body_json
            FROM subagent_instances
            WHERE subagent_id = ? AND tenant_id = ?
              AND federation_id = ? AND supervisor_id = ?
            LIMIT 1
            """,
            ("subagent_id", "tenant_id", "federation_id", "supervisor_id"),
            kind=StatementKind.QUERY,
        ),
        _template(
            "casf_select_subagent_admission_authority",
            """
            SELECT instances.body_json, instances.logical_state,
                   instances.admitted_concurrency_slot,
                   instances.worker_process_birth_id, instances.revision,
                   instances.fencing_epoch, instances.lease_id,
                   definitions.body_json AS definition_json,
                   assignments.body_json AS assignment_json,
                   assignments.status AS assignment_status,
                   assignments.fencing_epoch AS assignment_fencing_epoch,
                   assignments.assignment_revision,
                   assignments.lease_id AS assignment_lease_id,
                   assignments.resource_reservation_id,
                   assignments.token_reservation_id,
                   assignments.admission_decision_id AS assignment_decision_id,
                   definitions.policy_id AS definition_policy_id,
                   definitions.policy_revision AS definition_policy_revision,
                   definitions.authorization_evidence_ref,
                   instances.admission_decision_id
            FROM subagent_instances AS instances
            INNER JOIN subagent_definitions AS definitions
              ON definitions.subagent_definition_id =
                 instances.subagent_definition_id
             AND definitions.tenant_id = instances.tenant_id
             AND definitions.federation_id = instances.federation_id
            INNER JOIN subagent_assignments AS assignments
              ON assignments.subagent_id = instances.subagent_id
             AND assignments.supervisor_id = instances.supervisor_id
             AND assignments.tenant_id = instances.tenant_id
             AND assignments.federation_id = instances.federation_id
            WHERE instances.subagent_id = ? AND instances.tenant_id = ?
              AND instances.federation_id = ? AND instances.supervisor_id = ?
            LIMIT 1
            """,
            ("subagent_id", "tenant_id", "federation_id", "supervisor_id"),
            kind=StatementKind.QUERY,
        ),
        _template(
            "casf_select_subagent_capability_authority",
            """
            SELECT body_json, policy_id, policy_revision,
                   admission_decision_id, freshness_state, expires_at
            FROM subagent_capabilities
            WHERE subagent_id = ? AND tenant_id = ? AND federation_id = ?
            ORDER BY subagent_capability_id ASC
            """,
            ("subagent_id", "tenant_id", "federation_id"),
            kind=StatementKind.QUERY,
        ),
        _template(
            "casf_select_subagent_outcome_execution",
            """
            SELECT attempts.attempt_id, attempts.attempt_number,
                   attempts.fencing_token, attempts.fence_epoch,
                   attempts.status AS attempt_status,
                   leases.claim_cid, leases.claimant_did,
                   leases.fencing_token AS lease_fencing_token,
                   leases.fence_epoch AS lease_fence_epoch,
                   leases.expires_at_ms, leases.state AS lease_state,
                   tasks.status AS task_status,
                   slots.slot_number, slots.worker_process_birth_id,
                   slots.lease_id AS slot_lease_id,
                   slots.fencing_epoch AS slot_fencing_epoch,
                   slots.state AS slot_state
            FROM subagent_instances AS instances
            INNER JOIN subagent_assignments AS assignments
              ON assignments.subagent_id = instances.subagent_id
             AND assignments.supervisor_id = instances.supervisor_id
             AND assignments.tenant_id = instances.tenant_id
             AND assignments.federation_id = instances.federation_id
             AND assignments.status = 'admitted'
             AND assignments.fencing_epoch = instances.fencing_epoch
            INNER JOIN tasks
              ON tasks.task_cid = instances.task_id
            INNER JOIN task_attempts AS attempts
              ON attempts.task_cid = instances.task_id
             AND attempts.status = 'running'
             AND attempts.fence_epoch = instances.fencing_epoch
            INNER JOIN leases
              ON leases.task_cid = instances.task_id
             AND leases.state = 'accepted'
             AND leases.claim_cid = instances.lease_id
             AND leases.claimant_did = instances.subagent_id
             AND leases.fencing_token = attempts.fencing_token
             AND leases.fence_epoch = attempts.fence_epoch
             AND leases.attempt = attempts.attempt_number
             AND leases.expires_at_ms > ?
            INNER JOIN subagent_execution_slots AS slots
              ON slots.tenant_id = instances.tenant_id
             AND slots.federation_id = instances.federation_id
             AND slots.subagent_id = instances.subagent_id
             AND slots.supervisor_id = instances.supervisor_id
             AND slots.state = 'active'
             AND slots.fencing_epoch = instances.fencing_epoch
             AND slots.lease_id = instances.lease_id
            WHERE instances.subagent_id = ? AND instances.tenant_id = ?
              AND instances.federation_id = ? AND instances.supervisor_id = ?
              AND instances.task_id = ? AND instances.logical_state = 'ACTIVE'
              AND instances.admitted_concurrency_slot = slots.slot_number
              AND tasks.status NOT IN (
                  'cancelled', 'completed', 'failed', 'rejected', 'stopped',
                  'superseded'
              )
            ORDER BY attempts.attempt_number DESC
            LIMIT 2
            """,
            (
                "now_epoch_ms", "subagent_id", "tenant_id", "federation_id",
                "supervisor_id", "task_id",
            ),
            kind=StatementKind.QUERY,
        ),
        _template(
            "casf_select_subagent_task_lease_authority",
            """
            SELECT tasks.status AS task_status, leases.claim_cid,
                   leases.claimant_did, leases.fencing_token,
                   leases.fence_epoch, leases.expires_at_ms,
                   leases.attempt, leases.state AS lease_state
            FROM subagent_instances AS instances
            INNER JOIN tasks ON tasks.task_cid = instances.task_id
            INNER JOIN leases
              ON leases.task_cid = instances.task_id
             AND leases.claim_cid = instances.lease_id
             AND leases.claimant_did = instances.subagent_id
             AND leases.fence_epoch = instances.fencing_epoch
             AND leases.state = 'accepted'
             AND leases.expires_at_ms > ?
            WHERE instances.subagent_id = ? AND instances.tenant_id = ?
              AND instances.federation_id = ? AND instances.supervisor_id = ?
              AND instances.task_id <> '' AND instances.lease_id <> ''
              AND tasks.status NOT IN (
                  'cancelled', 'completed', 'failed', 'rejected', 'stopped',
                  'superseded'
              )
            LIMIT 1
            """,
            (
                "now_epoch_ms", "subagent_id", "tenant_id", "federation_id",
                "supervisor_id",
            ),
            kind=StatementKind.QUERY,
        ),
        _template(
            "casf_insert_subagent_outcome",
            """
            INSERT INTO subagent_outcomes (
                outcome_id, tenant_id, federation_id, supervisor_id,
                subagent_id, task_id, attempt_id, status,
                evidence_ref, fencing_epoch, recorded_at, body_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "outcome_id",
                "tenant_id",
                "federation_id",
                "supervisor_id",
                "subagent_id",
                "task_id",
                "attempt_id",
                "status",
                "evidence_ref",
                "fencing_epoch",
                "recorded_at",
                "body_json",
            ),
        ),
        _template(
            "casf_insert_subscription",
            """
            INSERT INTO event_subscriptions (
                subscription_id, tenant_id, federation_id, consumer_id,
                supervisor_id, revision, event_classes_json,
                maximum_batch, maximum_pending, maximum_fanout,
                retry_budget, expires_at, status, created_at, updated_at,
                body_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "subscription_id",
                "tenant_id",
                "federation_id",
                "consumer_id",
                "supervisor_id",
                "revision",
                "event_classes_json",
                "maximum_batch",
                "maximum_pending",
                "maximum_fanout",
                "retry_budget",
                "expires_at",
                "status",
                "created_at",
                "updated_at",
                "body_json",
            ),
        ),
        _template(
            "casf_insert_subscription_selector",
            """
            INSERT INTO event_subscription_selectors (
                selector_id, subscription_id, subscription_revision,
                selector_kind, selector_value, ordinal
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                "selector_id",
                "subscription_id",
                "subscription_revision",
                "selector_kind",
                "selector_value",
                "ordinal",
            ),
        ),
        _template(
            "casf_insert_consumer_cursor",
            """
            INSERT INTO consumer_cursors (
                consumer_id, subscription_id, subscription_revision,
                tenant_id, federation_id, global_sequence, store_generation,
                last_event_id, processing_event_id, fencing_epoch, revision,
                updated_at, body_json
            ) VALUES (?, ?, ?, ?, ?, 0, ?, '', '', ?, 1, ?, ?)
            """,
            (
                "consumer_id",
                "subscription_id",
                "subscription_revision",
                "tenant_id",
                "federation_id",
                "store_generation",
                "fencing_epoch",
                "updated_at",
                "body_json",
            ),
        ),
        _template(
            "casf_select_consumer_cursor",
            """
            SELECT consumer_id, subscription_id, subscription_revision,
                   tenant_id, federation_id, global_sequence,
                   store_generation, last_event_id, processing_event_id,
                   fencing_epoch, revision, updated_at, body_json
            FROM consumer_cursors
            WHERE consumer_id = ? AND subscription_id = ?
              AND tenant_id = ? AND federation_id = ?
            LIMIT 1
            """,
            ("consumer_id", "subscription_id", "tenant_id", "federation_id"),
            kind=StatementKind.QUERY,
        ),
        _template(
            "casf_resolve_consumer_cursor_scope",
            """
            SELECT tenant_id, federation_id
            FROM consumer_cursors
            WHERE consumer_id = ? AND subscription_id = ?
            LIMIT 1
            """,
            ("consumer_id", "subscription_id"),
            kind=StatementKind.QUERY,
        ),
        _template(
            "casf_select_event_for_routing",
            """
            SELECT event_id, event_cid, event_type, stream_id, sequence,
                   global_sequence, causal_parent_ids_json, correlation_id,
                   causation_id, tenant_id, federation_id, supervisor_id,
                   task_cid AS task_id, repository_id, tree_id, goal_id,
                   subgoal_id, symbol_id, contract_id, proof_obligation_id,
                   resource_class, payload_ref, changed_fact_refs_json,
                   effect_class, recorded_at, expires_at, deduplication_key
            FROM domain_events AS events
            WHERE events.event_id = ? AND events.tenant_id = ?
              AND events.federation_id = ?
            LIMIT 1
            """,
            ("event_id", "tenant_id", "federation_id"),
            kind=StatementKind.QUERY,
        ),
        _template(
            "casf_select_outbox_for_routing",
            """
            SELECT outbox_id, event_id, event_cid, tenant_id, federation_id,
                   status
            FROM transactional_outbox AS outbox
            WHERE outbox.event_id = ? AND outbox.tenant_id = ?
              AND outbox.federation_id = ?
            LIMIT 1
            """,
            ("event_id", "tenant_id", "federation_id"),
            kind=StatementKind.QUERY,
        ),
        _template(
            "casf_select_event_for_ack",
            """
            SELECT event_id, event_cid, event_type, stream_id, sequence,
                   global_sequence, causal_parent_ids_json, correlation_id,
                   causation_id, tenant_id, federation_id, supervisor_id,
                   task_cid AS task_id, repository_id, tree_id, goal_id,
                   subgoal_id, symbol_id, contract_id, proof_obligation_id,
                   resource_class, payload_ref, changed_fact_refs_json,
                   effect_class, recorded_at, expires_at, deduplication_key
            FROM domain_events AS events
            WHERE events.event_id = ? AND events.tenant_id = ?
              AND events.federation_id = ?
              AND EXISTS (
                  SELECT 1 FROM event_delivery_queue AS queue
                  WHERE queue.representative_event_id = events.event_id
                    AND queue.tenant_id = events.tenant_id
                    AND queue.federation_id = events.federation_id
                    AND queue.subscription_id = ?
                    AND queue.subscription_revision = ?
                    AND queue.consumer_id = ?
              )
            LIMIT 1
            """,
            (
                "event_id", "tenant_id", "federation_id", "subscription_id",
                "subscription_revision", "consumer_id",
            ),
            kind=StatementKind.QUERY,
        ),
        _template(
            "casf_select_outbox_for_delivery",
            """
            SELECT outbox.outbox_id, outbox.event_id, outbox.event_cid,
                   outbox.tenant_id, outbox.federation_id, outbox.status
            FROM transactional_outbox AS outbox
            WHERE outbox.event_id = ? AND outbox.tenant_id = ?
              AND outbox.federation_id = ?
              AND EXISTS (
                  SELECT 1 FROM event_delivery_queue AS queue
                  WHERE queue.outbox_id = outbox.outbox_id
                    AND queue.representative_event_id = outbox.event_id
                    AND queue.tenant_id = outbox.tenant_id
                    AND queue.federation_id = outbox.federation_id
                    AND queue.subscription_id = ?
                    AND queue.subscription_revision = ?
                    AND queue.consumer_id = ?
              )
            LIMIT 1
            """,
            (
                "event_id", "tenant_id", "federation_id", "subscription_id",
                "subscription_revision", "consumer_id",
            ),
            kind=StatementKind.QUERY,
        ),
        _template(
            "casf_insert_delivery_attempt",
            """
            INSERT INTO delivery_attempts (
                attempt_id, tenant_id, federation_id, event_id, outbox_id,
                delivery_id, subscription_id, subscription_revision, consumer_id,
                attempt_number, fencing_epoch, status, error_code,
                recorded_at, finished_at, body_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?)
            """,
            (
                "attempt_id",
                "tenant_id",
                "federation_id",
                "event_id",
                "outbox_id",
                "delivery_id",
                "subscription_id",
                "subscription_revision",
                "consumer_id",
                "attempt_number",
                "fencing_epoch",
                "status",
                "error_code",
                "recorded_at",
                "body_json",
            ),
        ),
        _template(
            "casf_select_delivery_for_ack",
            """
            SELECT delivery_attempts.attempt_id, delivery_attempts.status,
                   delivery_attempts.attempt_number,
                   delivery_attempts.fencing_epoch,
                   delivery_attempts.delivery_id,
                   queue.revision AS queue_revision,
                   queue.status AS queue_status
            FROM delivery_attempts
            INNER JOIN event_delivery_queue AS queue
              ON queue.delivery_id = delivery_attempts.delivery_id
             AND queue.tenant_id = delivery_attempts.tenant_id
             AND queue.federation_id = delivery_attempts.federation_id
             AND queue.subscription_id = delivery_attempts.subscription_id
             AND queue.subscription_revision = delivery_attempts.subscription_revision
             AND queue.consumer_id = delivery_attempts.consumer_id
            INNER JOIN transactional_outbox
              ON transactional_outbox.outbox_id = delivery_attempts.outbox_id
             AND transactional_outbox.event_id = delivery_attempts.event_id
             AND transactional_outbox.tenant_id = delivery_attempts.tenant_id
             AND transactional_outbox.federation_id = delivery_attempts.federation_id
            WHERE delivery_attempts.attempt_id = ?
              AND delivery_attempts.tenant_id = ?
              AND delivery_attempts.federation_id = ?
              AND delivery_attempts.event_id = ?
              AND delivery_attempts.subscription_id = ?
              AND delivery_attempts.subscription_revision = ?
              AND delivery_attempts.consumer_id = ?
              AND delivery_attempts.fencing_epoch = ?
              AND NOT EXISTS (
                  SELECT 1
                  FROM delivery_attempts AS newer
                  WHERE newer.event_id = delivery_attempts.event_id
                    AND newer.subscription_id = delivery_attempts.subscription_id
                    AND newer.subscription_revision = delivery_attempts.subscription_revision
                    AND newer.consumer_id = delivery_attempts.consumer_id
                    AND newer.attempt_number > delivery_attempts.attempt_number
              )
            LIMIT 1
            """,
            (
                "attempt_id",
                "tenant_id",
                "federation_id",
                "event_id",
                "subscription_id",
                "subscription_revision",
                "consumer_id",
                "fencing_epoch",
            ),
            kind=StatementKind.QUERY,
        ),
        _template(
            "casf_mark_delivery_acknowledged",
            """
            UPDATE delivery_attempts
            SET status = 'acknowledged', finished_at = ?
            WHERE attempt_id = ? AND tenant_id = ? AND federation_id = ?
              AND event_id = ? AND subscription_id = ? AND consumer_id = ?
              AND subscription_revision = ?
              AND fencing_epoch = ? AND status = 'delivered'
              AND NOT EXISTS (
                  SELECT 1
                  FROM delivery_attempts AS newer
                  WHERE newer.event_id = delivery_attempts.event_id
                    AND newer.subscription_id = delivery_attempts.subscription_id
                    AND newer.subscription_revision = delivery_attempts.subscription_revision
                    AND newer.consumer_id = delivery_attempts.consumer_id
                    AND newer.attempt_number > delivery_attempts.attempt_number
              )
            RETURNING attempt_id
            """,
            (
                "finished_at",
                "attempt_id",
                "tenant_id",
                "federation_id",
                "event_id",
                "subscription_id",
                "consumer_id",
                "subscription_revision",
                "fencing_epoch",
            ),
        ),
        _template(
            "casf_insert_event_acknowledgement",
            """
            INSERT INTO event_acknowledgements (
                acknowledgement_id, tenant_id, federation_id, event_id,
                subscription_id, consumer_id, subscription_revision,
                global_sequence, delivery_attempt_id, cursor_revision,
                fencing_epoch, disposition, processed_effect_ref,
                recorded_at, body_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "acknowledgement_id",
                "tenant_id",
                "federation_id",
                "event_id",
                "subscription_id",
                "consumer_id",
                "subscription_revision",
                "global_sequence",
                "delivery_attempt_id",
                "cursor_revision",
                "fencing_epoch",
                "disposition",
                "processed_effect_ref",
                "recorded_at",
                "body_json",
            ),
        ),
        _template(
            "casf_advance_consumer_cursor",
            """
            UPDATE consumer_cursors
            SET global_sequence = ?, store_generation = ?, last_event_id = ?,
                processing_event_id = '', revision = revision + 1,
                updated_at = ?, body_json = ?
            WHERE consumer_id = ? AND subscription_id = ?
              AND subscription_revision = ? AND revision = ?
              AND fencing_epoch = ? AND global_sequence < ?
            RETURNING revision
            """,
            (
                "global_sequence",
                "store_generation",
                "last_event_id",
                "updated_at",
                "body_json",
                "consumer_id",
                "subscription_id",
                "subscription_revision",
                "expected_revision",
                "expected_fencing_epoch",
                "upper_global_sequence",
            ),
        ),
        _template(
            "casf_select_subscription",
            """
            SELECT subscription_id, tenant_id, federation_id, consumer_id,
                   revision, event_classes_json, maximum_batch,
                   maximum_pending, maximum_fanout, retry_budget,
                   consecutive_failures, expires_at, status
            FROM event_subscriptions
            WHERE subscription_id = ? AND tenant_id = ? AND federation_id = ?
            LIMIT 1
            """,
            ("subscription_id", "tenant_id", "federation_id"),
            kind=StatementKind.QUERY,
        ),
        _template(
            "casf_select_subscription_selectors",
            """
            SELECT selectors.selector_kind, selectors.selector_value
            FROM event_subscription_selectors AS selectors
            INNER JOIN event_subscriptions AS subscriptions
              ON subscriptions.subscription_id = selectors.subscription_id
             AND subscriptions.revision = selectors.subscription_revision
            WHERE selectors.subscription_id = ?
              AND selectors.subscription_revision = ?
              AND subscriptions.tenant_id = ?
              AND subscriptions.federation_id = ?
            ORDER BY selectors.ordinal ASC
            """,
            (
                "subscription_id", "subscription_revision", "tenant_id",
                "federation_id",
            ),
            kind=StatementKind.QUERY,
        ),
        _template(
            "casf_select_admitted_policy",
            """
            SELECT policy_id, tenant_id, federation_id, revision,
                   authorization_evidence_ref, maximum_supervisors,
                   maximum_subagents, maximum_concurrent_subagents,
                   expires_at, status, content_ref, body_json
            FROM federation_policies
            WHERE policy_id = ? AND tenant_id = ? AND federation_id = ?
              AND revision = ? AND status = 'admitted'
            LIMIT 1
            """,
            ("policy_id", "tenant_id", "federation_id", "policy_revision"),
            kind=StatementKind.QUERY,
        ),
        _template(
            "casf_lookup_idempotency",
            """
            SELECT idempotency_key, command_kind, command_id, result_digest,
                   body_json
            FROM idempotency_records
            WHERE idempotency_key = ?
            LIMIT 1
            """,
            ("idempotency_key",),
            kind=StatementKind.QUERY,
        ),
        _template(
            "casf_insert_policy_decision",
            """
            INSERT INTO federation_policy_decisions (
                policy_decision_id, tenant_id, federation_id, policy_id,
                policy_revision, subject_kind, subject_ref, operation,
                verdict, reason_code, evidence_ref, decided_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'admitted', ?, ?, ?)
            """,
            (
                "policy_decision_id", "tenant_id", "federation_id", "policy_id",
                "policy_revision", "subject_kind", "subject_ref", "operation",
                "reason_code", "evidence_ref", "decided_at",
            ),
        ),
        _template(
            "casf_insert_supervisor_definition",
            """
            INSERT INTO supervisor_definitions (
                supervisor_definition_id, tenant_id, federation_id,
                specialization, capability_set_ref, allowed_operations_ref,
                effect_ceiling, risk_ceiling, resource_ceiling_ref,
                token_ceiling_ref, proof_requirements_ref, merge_policy_ref,
                policy_id, policy_revision, authorization_evidence_ref,
                content_ref, created_at, body_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "supervisor_definition_id", "tenant_id", "federation_id",
                "specialization", "capability_set_ref", "allowed_operations_ref",
                "effect_ceiling", "risk_ceiling", "resource_ceiling_ref",
                "token_ceiling_ref", "proof_requirements_ref", "merge_policy_ref",
                "policy_id", "policy_revision", "authorization_evidence_ref",
                "content_ref", "created_at", "body_json",
            ),
        ),
        _template(
            "casf_insert_supervisor_capability",
            """
            INSERT INTO supervisor_capabilities (
                capability_record_id, tenant_id, federation_id, supervisor_id,
                capability_kind, capability_revision, observed_generation,
                freshness_state, evidence_ref, policy_id, policy_revision,
                admission_decision_id, content_ref, expires_at, recorded_at,
                body_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 'current', ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "capability_record_id", "tenant_id", "federation_id",
                "supervisor_id", "capability_kind", "capability_revision",
                "observed_generation", "evidence_ref", "policy_id",
                "policy_revision", "admission_decision_id", "content_ref",
                "expires_at", "recorded_at", "body_json",
            ),
        ),
        _template(
            "casf_insert_subagent_definition",
            """
            INSERT INTO subagent_definitions (
                subagent_definition_id, tenant_id, federation_id,
                capability_set_ref, allowed_operations_ref, effect_scope_ref,
                resource_class, policy_id, policy_revision,
                authorization_evidence_ref, content_ref, created_at, body_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "subagent_definition_id", "tenant_id", "federation_id",
                "capability_set_ref", "allowed_operations_ref", "effect_scope_ref",
                "resource_class", "policy_id", "policy_revision",
                "authorization_evidence_ref", "content_ref", "created_at", "body_json",
            ),
        ),
        _template(
            "casf_insert_subagent_assignment",
            """
            INSERT INTO subagent_assignments (
                subagent_assignment_id, tenant_id, federation_id, supervisor_id,
                subagent_id, task_cid, assignment_revision, lease_id,
                fencing_epoch, resource_reservation_id, token_reservation_id,
                status, revision, admission_decision_id, content_ref,
                assigned_at, updated_at, body_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "subagent_assignment_id", "tenant_id", "federation_id",
                "supervisor_id", "subagent_id", "task_cid",
                "assignment_revision", "lease_id", "fencing_epoch",
                "resource_reservation_id", "token_reservation_id", "status",
                "revision", "admission_decision_id", "content_ref",
                "assigned_at", "updated_at", "body_json",
            ),
        ),
        _template(
            "casf_insert_subagent_capability",
            """
            INSERT INTO subagent_capabilities (
                subagent_capability_id, tenant_id, federation_id, subagent_id,
                capability_kind, capability_revision, evidence_ref, policy_id,
                policy_revision, admission_decision_id, content_ref,
                freshness_state, expires_at, recorded_at, body_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'current', ?, ?, ?)
            """,
            (
                "subagent_capability_id", "tenant_id", "federation_id",
                "subagent_id", "capability_kind", "capability_revision",
                "evidence_ref", "policy_id", "policy_revision",
                "admission_decision_id", "content_ref", "expires_at",
                "recorded_at", "body_json",
            ),
        ),
        _template(
            "casf_count_supervisor_active_attempts",
            """
            SELECT COUNT(DISTINCT task_attempts.attempt_id) AS active_attempts
            FROM task_attempts
            INNER JOIN subagent_instances
              ON subagent_instances.task_id = task_attempts.task_cid
            WHERE subagent_instances.tenant_id = ?
              AND subagent_instances.federation_id = ?
              AND subagent_instances.supervisor_id = ?
              AND task_attempts.status NOT IN (
                  'accepted', 'cancelled', 'completed', 'failed', 'rejected', 'stopped'
              )
            """,
            ("tenant_id", "federation_id", "supervisor_id"),
            kind=StatementKind.QUERY,
        ),
        _template(
            "casf_count_supervisor_active_effects",
            """
            SELECT COUNT(*) AS active_effects
            FROM federation_effect_reservations
            WHERE tenant_id = ? AND federation_id = ? AND supervisor_id = ?
              AND state NOT IN (
                  'cancelled', 'compensated', 'completed', 'failed', 'released', 'revoked'
              )
            """,
            ("tenant_id", "federation_id", "supervisor_id"),
            kind=StatementKind.QUERY,
        ),
        _template(
            "casf_count_supervisor_active_slots",
            """
            SELECT COUNT(*) AS active_slots
            FROM subagent_execution_slots
            WHERE tenant_id = ? AND federation_id = ? AND supervisor_id = ?
              AND state = 'active' AND subagent_id IS NOT NULL
            """,
            ("tenant_id", "federation_id", "supervisor_id"),
            kind=StatementKind.QUERY,
        ),
        _template(
            "casf_seed_subagent_slot",
            """
            INSERT INTO subagent_execution_slots (
                tenant_id, federation_id, slot_number, subagent_id,
                supervisor_id, worker_process_birth_id, lease_id,
                fencing_epoch, state, revision, reserved_at, released_at
            ) VALUES (?, ?, ?, NULL, '', '', '', ?, 'available', 1, NULL, NULL)
            """,
            ("tenant_id", "federation_id", "slot_number", "fencing_epoch"),
        ),
        _template(
            "casf_select_available_subagent_slot",
            """
            SELECT slot_number, revision
            FROM subagent_execution_slots
            WHERE tenant_id = ? AND federation_id = ? AND state = 'available'
              AND fencing_epoch = ?
            ORDER BY slot_number ASC
            LIMIT 1
            """,
            ("tenant_id", "federation_id", "fencing_epoch"),
            kind=StatementKind.QUERY,
        ),
        _template(
            "casf_reserve_subagent_slot",
            """
            UPDATE subagent_execution_slots
            SET subagent_id = ?, supervisor_id = ?,
                worker_process_birth_id = ?, lease_id = ?, state = 'active',
                revision = revision + 1, reserved_at = ?, released_at = NULL
            WHERE tenant_id = ? AND federation_id = ? AND slot_number = ?
              AND revision = ? AND fencing_epoch = ? AND state = 'available'
              AND subagent_id IS NULL
              AND NOT EXISTS (
                  SELECT 1 FROM subagent_execution_slots AS occupied
                  WHERE occupied.tenant_id = ? AND occupied.federation_id = ?
                    AND occupied.subagent_id = ? AND occupied.state = 'active'
              )
            RETURNING slot_number, revision
            """,
            (
                "subagent_id", "supervisor_id", "worker_process_birth_id",
                "lease_id", "reserved_at", "tenant_id", "federation_id",
                "slot_number", "expected_slot_revision", "fencing_epoch",
                "scope_tenant_id", "scope_federation_id", "unique_subagent_id",
            ),
        ),
        _template(
            "casf_activate_subagent",
            """
            UPDATE subagent_instances
            SET admitted_concurrency_slot = ?, worker_process_birth_id = ?,
                logical_state = 'ACTIVE', revision = revision + 1,
                updated_at = ?, body_json = ?
            WHERE subagent_id = ? AND tenant_id = ? AND federation_id = ?
              AND supervisor_id = ? AND revision = ? AND fencing_epoch = ?
              AND admitted_concurrency_slot = 0 AND logical_state = 'ADMITTED'
            RETURNING revision
            """,
            (
                "slot_number", "worker_process_birth_id", "updated_at", "body_json",
                "subagent_id", "tenant_id", "federation_id", "supervisor_id",
                "expected_revision", "fencing_epoch",
            ),
        ),
        _template(
            "casf_select_active_subagent_slot",
            """
            SELECT slots.slot_number, slots.revision AS slot_revision,
                   slots.worker_process_birth_id, instances.revision AS agent_revision,
                   instances.body_json, instances.supervisor_id, instances.lease_id,
                   instances.fencing_epoch
            FROM subagent_execution_slots AS slots
            INNER JOIN subagent_instances AS instances
              ON instances.tenant_id = slots.tenant_id
             AND instances.federation_id = slots.federation_id
             AND instances.subagent_id = slots.subagent_id
            WHERE slots.tenant_id = ? AND slots.federation_id = ?
              AND slots.subagent_id = ? AND slots.state = 'active'
            LIMIT 1
            """,
            ("tenant_id", "federation_id", "subagent_id"),
            kind=StatementKind.QUERY,
        ),
        _template(
            "casf_release_subagent_slot",
            """
            UPDATE subagent_execution_slots
            SET subagent_id = NULL, supervisor_id = '',
                worker_process_birth_id = '', lease_id = '', state = 'available',
                revision = revision + 1, released_at = ?
            WHERE tenant_id = ? AND federation_id = ? AND slot_number = ?
              AND subagent_id = ? AND state = 'active'
              AND revision = ? AND fencing_epoch = ?
            RETURNING revision
            """,
            (
                "released_at", "tenant_id", "federation_id", "slot_number",
                "subagent_id", "expected_slot_revision", "fencing_epoch",
            ),
        ),
        _template(
            "casf_deactivate_subagent",
            """
            UPDATE subagent_instances
            SET admitted_concurrency_slot = 0, worker_process_birth_id = '',
                logical_state = 'ADMITTED', revision = revision + 1,
                updated_at = ?, body_json = ?
            WHERE subagent_id = ? AND tenant_id = ? AND federation_id = ?
              AND supervisor_id = ? AND revision = ? AND fencing_epoch = ?
              AND admitted_concurrency_slot = ? AND logical_state = 'ACTIVE'
            RETURNING revision
            """,
            (
                "updated_at", "body_json", "subagent_id", "tenant_id",
                "federation_id", "supervisor_id", "expected_revision",
                "fencing_epoch", "slot_number",
            ),
        ),
        _template(
            "casf_insert_subagent_slot_ledger",
            """
            INSERT INTO subagent_slot_ledger (
                slot_ledger_id, tenant_id, federation_id, slot_number,
                subagent_id, supervisor_id, operation, prior_revision,
                resulting_revision, fencing_epoch, event_id, recorded_at,
                body_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "slot_ledger_id", "tenant_id", "federation_id", "slot_number",
                "subagent_id", "supervisor_id", "operation", "prior_revision",
                "resulting_revision", "fencing_epoch", "event_id", "recorded_at",
                "body_json",
            ),
        ),
        _template(
            "casf_insert_coalescing_coverage",
            """
            INSERT INTO event_coalescing_coverage (
                coverage_id, decision_id, tenant_id, federation_id,
                subscription_id, subscription_revision,
                representative_event_id, coalescing_mode, input_event_count,
                content_ref, created_at, body_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (coverage_id) DO NOTHING
            RETURNING coverage_id
            """,
            (
                "coverage_id", "decision_id", "tenant_id", "federation_id",
                "subscription_id", "subscription_revision",
                "representative_event_id", "coalescing_mode", "input_event_count",
                "content_ref", "created_at", "body_json",
            ),
        ),
        _template(
            "casf_select_coalescing_coverage",
            """
            SELECT coverage_id, body_json
            FROM event_coalescing_coverage
            WHERE coverage_id = ?
            LIMIT 1
            """,
            ("coverage_id",),
            kind=StatementKind.QUERY,
        ),
        _template(
            "casf_insert_coalescing_input",
            """
            INSERT INTO event_coalescing_inputs (coverage_id, event_id, ordinal)
            VALUES (?, ?, ?)
            ON CONFLICT (coverage_id, event_id) DO NOTHING
            """,
            ("coverage_id", "event_id", "ordinal"),
        ),
        _template(
            "casf_insert_delivery_queue",
            """
            INSERT INTO event_delivery_queue (
                delivery_id, tenant_id, federation_id, subscription_id,
                subscription_revision, consumer_id, decision_id,
                representative_event_id, outbox_id, status, attempt_number,
                fencing_epoch, available_at, revision, created_at, updated_at,
                body_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?, 1, ?, ?, ?)
            ON CONFLICT (delivery_id) DO NOTHING
            RETURNING delivery_id
            """,
            (
                "delivery_id", "tenant_id", "federation_id", "subscription_id",
                "subscription_revision", "consumer_id", "decision_id",
                "representative_event_id", "outbox_id", "attempt_number",
                "fencing_epoch", "available_at", "created_at", "updated_at",
                "body_json",
            ),
        ),
        _template(
            "casf_select_delivery_queue",
            """
            SELECT delivery_id, tenant_id, federation_id, subscription_id,
                   subscription_revision, consumer_id, decision_id,
                   representative_event_id, outbox_id, status, attempt_number,
                   fencing_epoch, available_at, revision, body_json
            FROM event_delivery_queue
            WHERE delivery_id = ? AND tenant_id = ? AND federation_id = ?
              AND subscription_id = ?
            LIMIT 1
            """,
            ("delivery_id", "tenant_id", "federation_id", "subscription_id"),
            kind=StatementKind.QUERY,
        ),
        _template(
            "casf_select_subscription_routing_state",
            """
            SELECT subscriptions.subscription_id, subscriptions.revision,
                   subscriptions.maximum_pending,
                   subscriptions.maximum_fanout,
                   COUNT(queue.delivery_id) AS pending_deliveries
            FROM event_subscriptions AS subscriptions
            LEFT JOIN event_delivery_queue AS queue
              ON queue.tenant_id = subscriptions.tenant_id
             AND queue.federation_id = subscriptions.federation_id
             AND queue.subscription_id = subscriptions.subscription_id
             AND queue.subscription_revision = subscriptions.revision
             AND queue.status IN ('pending', 'retry', 'delivered')
            WHERE subscriptions.tenant_id = ?
              AND subscriptions.federation_id = ?
              AND subscriptions.subscription_id = ?
              AND subscriptions.status = 'active'
              AND (subscriptions.expires_at IS NULL
                   OR subscriptions.expires_at = ''
                   OR subscriptions.expires_at > ?)
            GROUP BY subscriptions.subscription_id, subscriptions.revision,
                     subscriptions.maximum_pending,
                     subscriptions.maximum_fanout
            ORDER BY subscriptions.subscription_id ASC
            LIMIT 2
            """,
            ("tenant_id", "federation_id", "subscription_id", "observed_at"),
            kind=StatementKind.QUERY,
        ),
        _template(
            "casf_list_subscription_durable_route_coverage",
            """
            SELECT DISTINCT queue.delivery_id
            FROM event_coalescing_inputs AS inputs
            INNER JOIN event_coalescing_coverage AS coverage
              ON coverage.coverage_id = inputs.coverage_id
            INNER JOIN event_delivery_queue AS queue
              ON queue.tenant_id = coverage.tenant_id
             AND queue.federation_id = coverage.federation_id
             AND queue.subscription_id = coverage.subscription_id
             AND queue.subscription_revision = coverage.subscription_revision
             AND queue.decision_id = coverage.decision_id
            INNER JOIN domain_events AS events
              ON events.event_id = inputs.event_id
             AND events.tenant_id = coverage.tenant_id
             AND events.federation_id = coverage.federation_id
            INNER JOIN transactional_outbox AS outbox
              ON outbox.event_id = inputs.event_id
             AND outbox.tenant_id = coverage.tenant_id
             AND outbox.federation_id = coverage.federation_id
            WHERE coverage.tenant_id = ? AND coverage.federation_id = ?
              AND coverage.subscription_id = ?
              AND events.global_sequence >= ? AND events.global_sequence <= ?
              AND outbox.status IN ('pending', 'retry')
            ORDER BY queue.delivery_id
            LIMIT ?
            """,
            (
                "tenant_id", "federation_id", "subscription_id",
                "first_global_sequence", "last_global_sequence", "limit",
            ),
            kind=StatementKind.QUERY,
        ),
        _template(
            "casf_list_outbox_coverage_pairs",
            """
            SELECT DISTINCT inputs.event_id, coverage.subscription_id,
                   coverage.subscription_revision, queue.delivery_id
            FROM event_coalescing_inputs AS inputs
            INNER JOIN event_coalescing_coverage AS coverage
              ON coverage.coverage_id = inputs.coverage_id
            INNER JOIN event_delivery_queue AS queue
              ON queue.tenant_id = coverage.tenant_id
             AND queue.federation_id = coverage.federation_id
             AND queue.subscription_id = coverage.subscription_id
             AND queue.subscription_revision = coverage.subscription_revision
             AND queue.decision_id = coverage.decision_id
            INNER JOIN domain_events AS events
              ON events.event_id = inputs.event_id
             AND events.tenant_id = coverage.tenant_id
             AND events.federation_id = coverage.federation_id
            WHERE coverage.tenant_id = ? AND coverage.federation_id = ?
              AND events.global_sequence >= ? AND events.global_sequence <= ?
            ORDER BY inputs.event_id, coverage.subscription_id, queue.delivery_id
            LIMIT ?
            """,
            (
                "tenant_id", "federation_id", "first_global_sequence",
                "last_global_sequence", "limit",
            ),
            kind=StatementKind.QUERY,
        ),
        _template(
            "casf_mark_queue_acknowledged",
            """
            UPDATE event_delivery_queue
            SET status = 'acknowledged', revision = revision + 1,
                updated_at = ?
            WHERE delivery_id = ? AND tenant_id = ? AND federation_id = ?
              AND subscription_id = ? AND subscription_revision = ?
              AND consumer_id = ? AND fencing_epoch = ?
              AND status = 'delivered' AND revision = ?
            RETURNING revision
            """,
            (
                "updated_at", "delivery_id", "tenant_id", "federation_id",
                "subscription_id", "subscription_revision", "consumer_id",
                "fencing_epoch", "expected_revision",
            ),
        ),
        _template(
            "casf_list_deliverable_queue",
            """
            SELECT queue.delivery_id, queue.body_json,
                   coverage.body_json AS coverage_json,
                   events.body_json AS event_json
            FROM event_delivery_queue AS queue
            INNER JOIN event_coalescing_coverage AS coverage
              ON coverage.decision_id = queue.decision_id
             AND coverage.subscription_id = queue.subscription_id
             AND coverage.subscription_revision = queue.subscription_revision
            INNER JOIN domain_events AS events
              ON events.event_id = queue.representative_event_id
             AND events.tenant_id = queue.tenant_id
             AND events.federation_id = queue.federation_id
            WHERE queue.tenant_id = ? AND queue.federation_id = ?
              AND queue.subscription_id = ? AND queue.consumer_id = ?
              AND queue.subscription_revision = ?
              AND queue.fencing_epoch = ?
              AND queue.status IN ('pending', 'retry', 'delivered')
            ORDER BY events.global_sequence ASC, queue.delivery_id ASC
            LIMIT ?
            """,
            (
                "tenant_id", "federation_id", "subscription_id", "consumer_id",
                "subscription_revision", "fencing_epoch", "limit",
            ),
            kind=StatementKind.QUERY,
        ),
        _template(
            "casf_select_queue_for_attempt",
            """
            SELECT delivery_id, outbox_id, attempt_number, revision,
                   tenant_id, federation_id, status
            FROM event_delivery_queue
            WHERE tenant_id = ? AND federation_id = ?
              AND subscription_id = ? AND subscription_revision = ?
              AND consumer_id = ? AND representative_event_id = ?
              AND attempt_number = ? AND fencing_epoch = ?
              AND status IN ('pending', 'retry')
            ORDER BY delivery_id ASC
            LIMIT 2
            """,
            (
                "tenant_id", "federation_id", "subscription_id",
                "subscription_revision", "consumer_id", "event_id",
                "prior_attempt_number", "fencing_epoch",
            ),
            kind=StatementKind.QUERY,
        ),
        _template(
            "casf_mark_queue_delivered",
            """
            UPDATE event_delivery_queue
            SET status = 'delivered', attempt_number = ?,
                revision = revision + 1, updated_at = ?, body_json = ?
            WHERE delivery_id = ? AND tenant_id = ? AND federation_id = ?
              AND subscription_id = ? AND subscription_revision = ?
              AND consumer_id = ? AND attempt_number = ? AND revision = ?
              AND fencing_epoch = ? AND status IN ('pending', 'retry')
            RETURNING revision
            """,
            (
                "attempt_number", "updated_at", "body_json", "delivery_id", "tenant_id",
                "federation_id", "subscription_id", "subscription_revision",
                "consumer_id", "prior_attempt_number", "expected_revision",
                "fencing_epoch",
            ),
        ),
        _template(
            "casf_select_attempt_for_failure",
            """
            SELECT attempts.attempt_id, attempts.status, attempts.attempt_number,
                   attempts.delivery_id, attempts.event_id, attempts.outbox_id,
                   attempts.tenant_id, attempts.federation_id,
                   queue.revision AS queue_revision, queue.status AS queue_status,
                   events.global_sequence AS event_global_sequence
            FROM delivery_attempts AS attempts
            INNER JOIN event_delivery_queue AS queue
              ON queue.delivery_id = attempts.delivery_id
             AND queue.tenant_id = attempts.tenant_id
             AND queue.federation_id = attempts.federation_id
            INNER JOIN domain_events AS events
              ON events.event_id = attempts.event_id
             AND events.tenant_id = attempts.tenant_id
             AND events.federation_id = attempts.federation_id
            WHERE attempts.attempt_id = ?
              AND attempts.tenant_id = ? AND attempts.federation_id = ?
              AND attempts.subscription_id = ?
              AND attempts.subscription_revision = ? AND attempts.consumer_id = ?
              AND attempts.fencing_epoch = ?
            LIMIT 1
            """,
            (
                "attempt_id", "tenant_id", "federation_id", "subscription_id",
                "subscription_revision", "consumer_id", "fencing_epoch",
            ),
            kind=StatementKind.QUERY,
        ),
        _template(
            "casf_mark_delivery_failed",
            """
            UPDATE delivery_attempts
            SET status = ?, error_code = ?, finished_at = ?, body_json = ?
            WHERE attempt_id = ? AND tenant_id = ? AND federation_id = ?
              AND subscription_id = ? AND subscription_revision = ?
              AND consumer_id = ? AND fencing_epoch = ? AND status = 'delivered'
            RETURNING attempt_id
            """,
            (
                "status", "error_code", "finished_at", "body_json", "attempt_id",
                "tenant_id", "federation_id", "subscription_id",
                "subscription_revision", "consumer_id", "fencing_epoch",
            ),
        ),
        _template(
            "casf_increment_subscription_failures",
            """
            UPDATE event_subscriptions
            SET consecutive_failures = consecutive_failures + 1,
                updated_at = ?
            WHERE tenant_id = ? AND federation_id = ?
              AND subscription_id = ? AND revision = ?
              AND consumer_id = ? AND status = 'active'
            RETURNING consecutive_failures
            """,
            (
                "updated_at", "tenant_id", "federation_id", "subscription_id",
                "subscription_revision", "consumer_id",
            ),
        ),
        _template(
            "casf_update_queue_after_failure",
            """
            UPDATE event_delivery_queue
            SET status = ?, revision = revision + 1, updated_at = ?
            WHERE delivery_id = ? AND tenant_id = ? AND federation_id = ?
              AND revision = ? AND fencing_epoch = ? AND status = 'delivered'
            RETURNING revision
            """,
            (
                "status", "updated_at", "delivery_id", "tenant_id",
                "federation_id", "expected_revision", "fencing_epoch",
            ),
        ),
        _template(
            "casf_quarantine_subscription",
            """
            UPDATE event_subscriptions
            SET status = 'quarantined', updated_at = ?
            WHERE tenant_id = ? AND federation_id = ?
              AND subscription_id = ? AND revision = ?
              AND consumer_id = ? AND status = 'active'
            RETURNING subscription_id
            """,
            (
                "updated_at", "tenant_id", "federation_id", "subscription_id",
                "subscription_revision", "consumer_id",
            ),
        ),
        _template(
            "casf_insert_dead_letter",
            """
            INSERT INTO dead_letters (
                dead_letter_id, tenant_id, federation_id, event_id, outbox_id,
                subscription_id, subscription_revision, consumer_id,
                retry_count, error_code, evidence_ref, quarantined, status,
                created_at, expires_at, resolved_at, revision, body_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'open', ?, ?, NULL, 1, ?)
            """,
            (
                "dead_letter_id", "tenant_id", "federation_id", "event_id",
                "outbox_id", "subscription_id", "subscription_revision",
                "consumer_id", "retry_count", "error_code", "evidence_ref",
                "quarantined", "created_at", "expires_at", "body_json",
            ),
        ),
        _template(
            "casf_is_subscription_quarantined",
            """
            SELECT status
            FROM event_subscriptions
            WHERE tenant_id = ? AND federation_id = ?
              AND subscription_id = ? AND revision = ?
            LIMIT 1
            """,
            (
                "tenant_id", "federation_id", "subscription_id",
                "subscription_revision",
            ),
            kind=StatementKind.QUERY,
        ),
        _template(
            "casf_list_dead_letters",
            """
            SELECT body_json
            FROM dead_letters
            WHERE tenant_id = ? AND federation_id = ?
              AND subscription_id = ? AND subscription_revision = ?
              AND consumer_id = ?
              AND status = 'open'
            ORDER BY created_at ASC, dead_letter_id ASC
            LIMIT ?
            """,
            (
                "tenant_id", "federation_id", "subscription_id",
                "subscription_revision", "consumer_id", "limit",
            ),
            kind=StatementKind.QUERY,
        ),
        _template(
            "casf_list_routed_wait_events",
            """
            SELECT events.event_id, events.event_cid, events.event_type,
                   events.stream_id, events.sequence, events.global_sequence,
                   events.causal_parent_ids_json, events.correlation_id,
                   events.causation_id, events.tenant_id, events.federation_id,
                   events.supervisor_id, events.task_cid AS task_id,
                   events.repository_id, events.tree_id, events.goal_id,
                   events.subgoal_id, events.symbol_id, events.contract_id,
                   events.proof_obligation_id, events.resource_class,
                   events.payload_ref, events.changed_fact_refs_json,
                   events.effect_class, events.recorded_at, events.expires_at,
                   events.deduplication_key
            FROM event_delivery_queue AS queue
            INNER JOIN domain_events AS events
              ON events.event_id = queue.representative_event_id
             AND events.tenant_id = queue.tenant_id
             AND events.federation_id = queue.federation_id
            WHERE queue.tenant_id = ? AND queue.federation_id = ?
              AND queue.subscription_id = ? AND queue.subscription_revision = ?
              AND queue.consumer_id = ?
              AND queue.status IN ('pending', 'retry', 'delivered')
              AND events.global_sequence > ?
            ORDER BY events.global_sequence ASC, queue.delivery_id ASC
            LIMIT ?
            """,
            (
                "tenant_id", "federation_id", "subscription_id",
                "subscription_revision", "consumer_id", "after_cursor", "limit",
            ),
            kind=StatementKind.QUERY,
        ),
        _template(
            "casf_reset_subscription_failures",
            """
            UPDATE event_subscriptions
            SET consecutive_failures = 0, updated_at = ?
            WHERE tenant_id = ? AND federation_id = ?
              AND subscription_id = ? AND revision = ? AND consumer_id = ?
            RETURNING subscription_id
            """,
            (
                "updated_at", "tenant_id", "federation_id", "subscription_id",
                "subscription_revision", "consumer_id",
            ),
        ),
        _template(
            "casf_select_dead_letter_for_retry",
            """
            SELECT dead.dead_letter_id, dead.event_id, dead.outbox_id,
                   dead.subscription_id, dead.subscription_revision,
                   dead.consumer_id, dead.status, dead.revision,
                   queue.delivery_id, queue.status AS queue_status,
                   queue.revision AS queue_revision,
                   queue.attempt_number, queue.fencing_epoch
            FROM dead_letters AS dead
            INNER JOIN event_delivery_queue AS queue
              ON queue.outbox_id = dead.outbox_id
             AND queue.subscription_id = dead.subscription_id
             AND queue.subscription_revision = dead.subscription_revision
             AND queue.consumer_id = dead.consumer_id
             AND queue.representative_event_id = dead.event_id
             AND queue.tenant_id = dead.tenant_id
             AND queue.federation_id = dead.federation_id
            WHERE dead.dead_letter_id = ? AND dead.tenant_id = ?
              AND dead.federation_id = ? AND dead.subscription_id = ?
              AND dead.subscription_revision = ? AND dead.consumer_id = ?
            LIMIT 1
            """,
            (
                "dead_letter_id", "tenant_id", "federation_id",
                "subscription_id", "subscription_revision", "consumer_id",
            ),
            kind=StatementKind.QUERY,
        ),
        _template(
            "casf_resolve_dead_letter_for_retry",
            """
            UPDATE dead_letters
            SET status = 'retried', resolved_at = ?, revision = revision + 1
            WHERE dead_letter_id = ? AND tenant_id = ? AND federation_id = ?
              AND subscription_id = ? AND subscription_revision = ?
              AND consumer_id = ? AND revision = ? AND status = 'open'
            RETURNING revision
            """,
            (
                "resolved_at", "dead_letter_id", "tenant_id", "federation_id",
                "subscription_id", "subscription_revision", "consumer_id",
                "expected_revision",
            ),
        ),
        _template(
            "casf_requeue_dead_letter_delivery",
            """
            UPDATE event_delivery_queue
            SET status = 'retry', available_at = ?, updated_at = ?,
                fencing_epoch = ?, revision = revision + 1
            WHERE delivery_id = ? AND tenant_id = ? AND federation_id = ?
              AND subscription_id = ? AND subscription_revision = ?
              AND consumer_id = ? AND revision = ? AND status = 'dead_lettered'
            RETURNING revision
            """,
            (
                "available_at", "updated_at", "fencing_epoch", "delivery_id",
                "tenant_id", "federation_id", "subscription_id",
                "subscription_revision", "consumer_id", "expected_revision",
            ),
        ),
        _template(
            "casf_unquarantine_subscription",
            """
            UPDATE event_subscriptions
            SET status = 'active', consecutive_failures = 0, updated_at = ?
            WHERE tenant_id = ? AND federation_id = ?
              AND subscription_id = ? AND revision = ? AND consumer_id = ?
              AND status IN ('active', 'quarantined')
            RETURNING status
            """,
            (
                "updated_at", "tenant_id", "federation_id", "subscription_id",
                "subscription_revision", "consumer_id",
            ),
        ),
        _template(
            "casf_list_matching_event_window",
            """
            SELECT event_id, event_cid, event_type, stream_id, sequence,
                   global_sequence, causal_parent_ids_json, correlation_id,
                   causation_id, tenant_id, federation_id, supervisor_id,
                   task_cid AS task_id, repository_id, tree_id, goal_id, subgoal_id,
                   symbol_id, contract_id, proof_obligation_id, resource_class,
                   payload_ref, changed_fact_refs_json, effect_class,
                   recorded_at, expires_at, deduplication_key
            FROM domain_events
            WHERE tenant_id = ? AND federation_id = ?
              AND global_sequence > ?
              AND list_contains(string_split(?, ','), event_type)
            ORDER BY global_sequence ASC
            LIMIT ?
            """,
            (
                "tenant_id",
                "federation_id",
                "after_cursor",
                "event_classes_csv",
                "limit",
            ),
            kind=StatementKind.QUERY,
        ),
    )


class FederationStateRepository:
    """Sealed federation registry over one already-attached state client."""

    INTERFACE = "FederationStateRepository@1"

    def __init__(
        self,
        client: QuackStateClient,
        *,
        event_notifier: Callable[[int], None] | None = None,
        outbox_notifier: Callable[[int], None] | None = None,
        test_failure_hook: Callable[[str], None] | None = None,
        require_quack_authority: bool = False,
        process_birth_factory: Callable[[], ProcessBirthIdentity] = current_process_birth,
        process_birth_reader: Callable[[int], ProcessBirthIdentity | None] = read_process_birth,
        supervisor_runtime_lease_seconds: int = 60,
        runtime_clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        if not isinstance(client, QuackStateClient) or not client.attached:
            raise FederationRepositoryError(
                "repository requires an already-attached typed state client"
            )
        if type(require_quack_authority) is not bool:
            raise FederationRepositoryError("require_quack_authority must be boolean")
        if not callable(process_birth_factory) or not callable(process_birth_reader):
            raise FederationRepositoryError("process-birth attestors must be callable")
        if (
            isinstance(supervisor_runtime_lease_seconds, bool)
            or not isinstance(supervisor_runtime_lease_seconds, int)
            or not 1 <= supervisor_runtime_lease_seconds <= 3_600
        ):
            raise FederationRepositoryError(
                "supervisor runtime lease must be between 1 and 3600 seconds"
            )
        if not callable(runtime_clock):
            raise FederationRepositoryError("runtime_clock must be callable")
        session = client.session
        if (
            require_quack_authority
            and session is not None
            and session.transport_mode is not TransportMode.QUACK
        ):
            raise FederationRepositoryError(
                "multi-client federation authority requires Quack; embedded mode is explicit single-owner only"
            )
        self.__client = client
        for template in _casf_templates():
            client.register_template(template)
        self._statement_catalog = client.seal_templates()
        self._event_notifier = event_notifier
        self._outbox_notifier = outbox_notifier
        self._test_failure_hook = test_failure_hook
        self._process_birth_factory = process_birth_factory
        self._process_birth_reader = process_birth_reader
        self._supervisor_runtime_lease_seconds = supervisor_runtime_lease_seconds
        self._runtime_clock = runtime_clock

    @property
    def statement_catalog(self) -> tuple[str, ...]:
        return self._statement_catalog

    def _runtime_now(self) -> datetime:
        observed = self._runtime_clock()
        if not isinstance(observed, datetime) or observed.tzinfo is None:
            raise FederationAuthorityError(
                "runtime authority clock must return a timezone-aware datetime"
            )
        return observed.astimezone(timezone.utc)

    def _attest_owner_process(
        self,
        *,
        tenant_id: str,
        federation_id: str,
        supervisor_id: str,
        subagent_id: str = "",
        recorded_at: str,
    ) -> tuple[str, ProcessBirthIdentity, str]:
        """Persist state-owner-observed PID birth evidence, never caller truth."""

        try:
            birth = self._process_birth_factory()
        except (OSError, RuntimeError, ValueError, TypeError) as exc:
            raise FederationAuthorityError(
                "kernel process-birth evidence is unavailable"
            ) from exc
        if (
            not isinstance(birth, ProcessBirthIdentity)
            or birth.pid <= 0
            or birth.start_time_ticks <= 0
            or not birth.boot_id
        ):
            raise FederationAuthorityError(
                "kernel process-birth evidence requires PID, start, and boot identity"
            )
        try:
            current = self._process_birth_reader(birth.pid)
        except (OSError, RuntimeError, ValueError, TypeError) as exc:
            raise FederationAuthorityError(
                "kernel process-birth evidence could not be revalidated"
            ) from exc
        if (
            not process_births_match(birth, current)
            or current is None
            or current.boot_id != birth.boot_id
            or current.parent_pid != birth.parent_pid
        ):
            raise FederationAuthorityError(
                "kernel process-birth evidence is stale or mismatched"
            )
        canonical_birth_id = process_birth_id(birth)
        attestation_id = "process-birth-attestation:" + content_identity(
            {
                "tenant_id": tenant_id,
                "federation_id": federation_id,
                "supervisor_id": supervisor_id,
                "subagent_id": subagent_id,
                "canonical_birth_id": canonical_birth_id,
            }
        )
        executable_ref = "process-executable:" + content_identity(
            {
                "canonical_birth_id": canonical_birth_id,
                "process_id": birth.pid,
            }
        )
        host_identity_ref = "host-boot:" + content_identity(
            {"boot_id": birth.boot_id}
        )
        self.__client.execute(
            "casf_insert_process_birth_attestation",
            {
                "process_birth_id": attestation_id,
                "tenant_id": tenant_id,
                "federation_id": federation_id,
                "supervisor_id": supervisor_id,
                "subagent_id": subagent_id,
                "process_id": birth.pid,
                "start_marker": str(birth.start_time_ticks),
                "executable_ref": executable_ref,
                "host_identity_ref": host_identity_ref,
                "started_at": recorded_at,
            },
        )
        return attestation_id, birth, canonical_birth_id

    def _assert_current_supervisor_runtime(
        self,
        *,
        row: Mapping[str, Any],
        tenant_id: str,
        federation_id: str,
        supervisor_id: str,
        observed_at: str,
    ) -> Mapping[str, Any]:
        rows = self.__client.execute(
            "casf_select_current_supervisor_runtime",
            {
                "tenant_id": tenant_id,
                "federation_id": federation_id,
                "supervisor_id": supervisor_id,
                "lease_id": row["lease_id"],
                "fencing_epoch": row["fencing_epoch"],
                "observed_at": observed_at,
            },
        )
        if len(rows) != 1:
            raise FederationAuthorityError(
                "executable supervisor state requires one current runtime lease"
            )
        runtime = rows[0]
        if (
            str(row["process_birth_id"]) != str(runtime["process_birth_id"])
            or int(runtime["process_id"]) != int(runtime["birth_process_id"])
            or str(runtime["start_marker"])
            != str(runtime["process_start_time_ticks"])
            or not str(runtime["process_boot_id"])
            or not str(runtime["host_identity_ref"])
            or not str(runtime["evidence_ref"])
        ):
            raise FederationAuthorityError(
                "supervisor runtime lease is not bound to exact process evidence"
            )
        expected = ProcessBirthIdentity(
            pid=int(runtime["process_id"]),
            start_time_ticks=int(runtime["process_start_time_ticks"]),
            boot_id=str(runtime["process_boot_id"]),
            parent_pid=int(runtime["process_parent_id"]),
        )
        try:
            current = self._process_birth_reader(expected.pid)
        except (OSError, RuntimeError, ValueError, TypeError) as exc:
            raise FederationAuthorityError(
                "supervisor runtime process cannot be revalidated"
            ) from exc
        if (
            not process_births_match(expected, current)
            or current is None
            or current.boot_id != expected.boot_id
            or current.parent_pid != expected.parent_pid
        ):
            raise FederationAuthorityError(
                "supervisor runtime process birth is no longer current"
            )
        return runtime

    def store_generation(self) -> int:
        return self.__client.load_generation().generation

    def pending_outbox_scopes(self, *, maximum: int) -> tuple[OutboxScope, ...]:
        """Return a bounded restart-safe routing frontier for the owner pump."""

        limit = _integer(maximum, "maximum", minimum=1, maximum=256)
        rows = self.__client.execute(
            "casf_list_pending_outbox_scopes",
            {"observed_at": utc_now(), "limit": limit},
        )
        if len(rows) > limit:
            raise FederationBoundsError("pending outbox scope query exceeded its bound")
        scopes = tuple(
            OutboxScope(
                tenant_id=str(row["tenant_id"]),
                federation_id=str(row["federation_id"]),
            )
            for row in rows
        )
        if len(set(scopes)) != len(scopes):
            raise FederationAuthorityError("pending outbox scope authority is duplicated")
        return scopes

    def pending_outbox_events(
        self,
        scope: OutboxScope,
        *,
        maximum: int,
    ) -> tuple[DomainEvent, ...]:
        """Load only immutable pending events inside one exact owner scope."""

        if not isinstance(scope, OutboxScope):
            raise FederationContractError("pending outbox scope must be typed")
        limit = _integer(maximum, "maximum", minimum=1, maximum=1_024)
        rows = self.__client.execute(
            "casf_list_pending_outbox_events",
            {
                "tenant_id": scope.tenant_id,
                "federation_id": scope.federation_id,
                "observed_at": utc_now(),
                "limit": limit,
            },
        )
        if len(rows) > limit:
            raise FederationBoundsError("pending outbox event query exceeded its bound")
        events = tuple(self._event_from_row(row) for row in rows)
        if any(
            event.tenant_id != scope.tenant_id
            or event.federation_id != scope.federation_id
            for event in events
        ):
            raise FederationAuthorityError("pending outbox event crossed scope")
        return events

    def active_subscription_ids(
        self,
        scope: OutboxScope,
        *,
        maximum: int,
    ) -> tuple[str, ...]:
        """Return the canonical bounded routing consumers for one scope."""

        if not isinstance(scope, OutboxScope):
            raise FederationContractError("subscription scope must be typed")
        limit = _integer(maximum, "maximum", minimum=1, maximum=4_096)
        rows = self.__client.execute(
            "casf_list_active_subscription_ids",
            {
                "tenant_id": scope.tenant_id,
                "federation_id": scope.federation_id,
                "observed_at": utc_now(),
                "limit": limit,
            },
        )
        values = tuple(
            _identifier(str(row["subscription_id"]), "subscription_id")
            for row in rows
        )
        if len(values) > limit or len(set(values)) != len(values):
            raise FederationAuthorityError(
                "active subscription authority exceeded or repeated its bound"
            )
        return values

    def mark_outbox_routed(
        self,
        scope: OutboxScope,
        events: Sequence[DomainEvent],
        *,
        route_batch_id: str,
        delivery_count: int,
        subscription_count: int,
        idempotency_key: str,
    ) -> OutboxDisposition:
        """Commit one content-addressed routing disposition under CAS.

        Routing and disposition are deliberately separate idempotent
        transactions.  A crash after durable deliveries but before this CAS
        replays the same route batch and cannot duplicate a queue row.
        """

        if not isinstance(scope, OutboxScope):
            raise FederationContractError("outbox disposition scope must be typed")
        values = tuple(events)
        if (
            not values
            or len(values) > 1_024
            or any(not isinstance(item, DomainEvent) for item in values)
        ):
            raise FederationBoundsError(
                "outbox disposition requires one bounded typed event batch"
            )
        if any(
            item.tenant_id != scope.tenant_id
            or item.federation_id != scope.federation_id
            for item in values
        ):
            raise FederationAuthorityError("outbox disposition crosses scope")
        sequences = tuple(item.global_sequence for item in values)
        event_ids = tuple(item.event_id for item in values)
        if (
            tuple(sorted(zip(sequences, event_ids)))  # noqa: B905
            != tuple(zip(sequences, event_ids))  # noqa: B905
            or len(set(event_ids)) != len(event_ids)
            or len(set(sequences)) != len(sequences)
        ):
            raise FederationContractError(
                "outbox disposition events must be uniquely ordered"
            )
        route_batch = _identifier(route_batch_id, "route_batch_id")
        deliveries = _integer(
            delivery_count,
            "delivery_count",
            minimum=0,
            maximum=len(values) * 4_096,
        )
        subscriptions = _integer(
            subscription_count,
            "subscription_count",
            minimum=0,
            maximum=4_096,
        )
        expected_key = "outbox-route:" + content_identity(
            {
                "scope": {
                    "tenant_id": scope.tenant_id,
                    "federation_id": scope.federation_id,
                },
                "event_ids": list(event_ids),
                "route_batch_id": route_batch,
            }
        )
        if idempotency_key != expected_key:
            raise FederationContractError(
                "outbox disposition idempotency identity is not canonical"
            )
        receipt_body = {
            "tenant_id": scope.tenant_id,
            "federation_id": scope.federation_id,
            "route_batch_id": route_batch,
            "event_ids": list(event_ids),
            "global_sequences": list(sequences),
            "delivery_count": deliveries,
            "subscription_count": subscriptions,
        }
        content_ref = content_identity(receipt_body)
        disposition_id = f"outbox-disposition:{content_ref.split(':', 1)[-1]}"
        command = self._command(
            command_id=f"command:outbox-disposition:{content_ref}",
            idempotency_key=idempotency_key,
            command_kind=CommandKind.APPEND,
            parameters={
                "operation": "event.outbox.disposition",
                "tenant_id": scope.tenant_id,
                "federation_id": scope.federation_id,
                "route_batch_id": route_batch,
                "disposition_id": disposition_id,
                "event_count": len(values),
            },
        )
        recorded_at = utc_now()

        def apply(
            _txn: StateTransaction,
            _command: StateCommand,
            live: Any,
        ) -> Mapping[str, Any]:
            authoritative_rows: list[Mapping[str, Any]] = []
            for event in values:
                rows = self.__client.execute(
                    "casf_select_pending_outbox_event",
                    {
                        "tenant_id": scope.tenant_id,
                        "federation_id": scope.federation_id,
                        "event_id": event.event_id,
                    },
                )
                if len(rows) != 1:
                    raise FederationRepositoryConflict(
                        "pending outbox event is absent or already disposed"
                    )
                row = rows[0]
                if (
                    str(row["event_cid"]) != event.event_cid
                    or int(row["global_sequence"]) != event.global_sequence
                ):
                    raise FederationAuthorityError(
                        "pending outbox event identity differs"
                    )
                authoritative_rows.append(row)
            active_ids = self.active_subscription_ids(scope, maximum=4_096)
            if len(active_ids) != subscriptions:
                raise DurableRoutingBackpressure(
                    "active subscription population changed before disposition"
                )
            coverage_rows = self.__client.execute(
                "casf_list_outbox_coverage_pairs",
                {
                    "tenant_id": scope.tenant_id,
                    "federation_id": scope.federation_id,
                    "first_global_sequence": min(sequences),
                    "last_global_sequence": max(sequences),
                    "limit": 65_537,
                },
            )
            if len(coverage_rows) > 65_536:
                raise FederationBoundsError(
                    "outbox disposition coverage exceeds its bound"
                )
            event_id_set = set(event_ids)
            active_id_set = set(active_ids)
            coverage_pairs = {
                (str(row["event_id"]), str(row["subscription_id"]))
                for row in coverage_rows
                if str(row["event_id"]) in event_id_set
                and str(row["subscription_id"]) in active_id_set
            }
            covered_delivery_ids = {
                str(row["delivery_id"])
                for row in coverage_rows
                if str(row["event_id"]) in event_id_set
                and str(row["subscription_id"]) in active_id_set
            }
            for subscription_id in active_ids:
                subscription = self._load_subscription(
                    subscription_id,
                    tenant_id=scope.tenant_id,
                    federation_id=scope.federation_id,
                )
                for event in values:
                    expired = bool(
                        event.expires_at
                        and datetime.fromisoformat(
                            event.expires_at.replace("Z", "+00:00")
                        )
                        <= datetime.fromisoformat(recorded_at.replace("Z", "+00:00"))
                    )
                    if (
                        not expired
                        and event_matches_subscription(event, subscription)
                        and (event.event_id, subscription_id) not in coverage_pairs
                    ):
                        raise DurableRoutingBackpressure(
                            "required subscription delivery is not durably admitted"
                        )
            if len(covered_delivery_ids) != deliveries:
                raise FederationAuthorityError(
                    "outbox delivery count differs from durable coverage"
                )
            self.__client.execute(
                "casf_insert_outbox_routing_disposition",
                {
                    "disposition_id": disposition_id,
                    "tenant_id": scope.tenant_id,
                    "federation_id": scope.federation_id,
                    "route_batch_id": route_batch,
                    "first_global_sequence": min(sequences),
                    "last_global_sequence": max(sequences),
                    "event_count": len(values),
                    "delivery_count": deliveries,
                    "subscription_count": subscriptions,
                    "content_ref": content_ref,
                    "created_at": recorded_at,
                    "body_json": _json(receipt_body),
                },
            )
            for ordinal, event in enumerate(values, start=1):
                self.__client.execute(
                    "casf_insert_outbox_routing_disposition_event",
                    {
                        "disposition_id": disposition_id,
                        "event_id": event.event_id,
                        "global_sequence": event.global_sequence,
                        "ordinal": ordinal,
                    },
                )
            if self._test_failure_hook is not None:
                self._test_failure_hook(
                    "after_outbox_disposition_before_mark"
                )
            for event, row in zip(values, authoritative_rows):  # noqa: B905
                updated = self.__client.execute(
                    "casf_mark_outbox_routed",
                    {
                        "updated_at": recorded_at,
                        "outbox_id": row["outbox_id"],
                        "event_id": event.event_id,
                        "event_cid": event.event_cid,
                        "tenant_id": scope.tenant_id,
                        "federation_id": scope.federation_id,
                        "global_sequence": event.global_sequence,
                        "expected_revision": row["revision"],
                    },
                )
                if len(updated) != 1:
                    raise FederationRepositoryConflict(
                        "outbox routing disposition CAS conflicted"
                    )
            return {
                "disposition_id": disposition_id,
                "event_ids": list(event_ids),
                "routed_global_sequence": max(sequences),
                "store_generation": live.generation,
            }

        result = self._submit(command, apply).result
        return OutboxDisposition(
            disposition_id=str(result["disposition_id"]),
            event_ids=tuple(str(item) for item in result["event_ids"]),
            routed_global_sequence=int(result["routed_global_sequence"]),
            store_generation=int(result["store_generation"]),
        )

    def resolve_event_wait_scope(
        self,
        *,
        consumer_id: str,
        subscription_id: str,
    ) -> tuple[str, str]:
        """Resolve wire-omitted scope from the durable subscription authority."""

        consumer = _identifier(consumer_id, "consumer_id")
        subscription = _identifier(subscription_id, "subscription_id")
        rows = self.__client.execute(
            "casf_resolve_consumer_cursor_scope",
            {"consumer_id": consumer, "subscription_id": subscription},
        )
        if len(rows) != 1:
            raise FederationRepositoryNotFound(
                "event wait consumer/subscription scope is absent"
            )
        tenant_id = _identifier(str(rows[0]["tenant_id"]), "tenant_id")
        federation_id = _identifier(
            str(rows[0]["federation_id"]),
            "federation_id",
        )
        admitted = self._load_subscription(
            subscription,
            tenant_id=tenant_id,
            federation_id=federation_id,
        )
        if admitted.consumer_id != consumer:
            raise FederationAuthorityError(
                "event wait consumer does not own its subscription"
            )
        return tenant_id, federation_id

    def lookup_federation_budget_reservation(
        self,
        idempotency_key: str,
        *,
        tenant_id: str,
        federation_id: str,
    ) -> BudgetReservation | None:
        """Return only the exact scoped admission reservation for a retry."""

        key = _identifier(idempotency_key, "idempotency_key")
        tenant = _identifier(tenant_id, "tenant_id")
        federation = _identifier(federation_id, "federation_id")
        rows = self.__client.execute(
            "casf_select_admission_budget_by_idempotency",
            {
                "tenant_id": tenant,
                "federation_id": federation,
                "idempotency_key": key,
            },
        )
        if not rows:
            return None
        if len(rows) != 1:
            raise FederationRepositoryConflict(
                "budget reservation idempotency authority is ambiguous"
            )
        row = rows[0]
        reservation = BudgetReservation.from_dict(_decode(row["body_json"]))
        if (
            reservation.binding.tenant_id != tenant
            or reservation.owner_id != federation
            or reservation.idempotency_key != key
            or reservation.record_id != str(row["reservation_id"])
            or reservation.cid != str(row["content_ref"])
            or reservation.status != str(row["state"])
        ):
            raise FederationAuthorityError(
                "budget reservation retry scope or content identity differs"
            )
        return reservation

    def reserve_federation_budget(
        self,
        reservation: BudgetReservation,
        *,
        capacity: Mapping[BudgetDimensionName, int],
    ) -> BudgetReservation:
        """Persist one bounded admission reservation under owner-side CAS."""

        if not isinstance(reservation, BudgetReservation):
            raise FederationContractError("reservation must be BudgetReservation")
        federation_id = f"federation:{reservation.request_cid}"
        if (
            reservation.owner_id != federation_id
            or reservation.status != "reserved"
            or reservation.revision != 1
            or reservation.policy_ref != reservation.binding.policy_ref
            or reservation.policy_revision != reservation.binding.policy_revision
            or reservation.expires_at != reservation.binding.expires_at
            or not reservation.authorization_evidence_ref.startswith(
                "budget-admission:"
            )
            or _expired(reservation.expires_at)
        ):
            raise FederationAuthorityError(
                "budget reservation is stale or differs from its authority binding"
            )
        normalized_capacity: dict[BudgetDimensionName, int] = {}
        for raw_name, amount in capacity.items():
            name = (
                raw_name
                if isinstance(raw_name, BudgetDimensionName)
                else BudgetDimensionName(raw_name)
            )
            if isinstance(amount, bool) or not isinstance(amount, int) or amount < 0:
                raise FederationContractError("budget capacity must be nonnegative integers")
            normalized_capacity[name] = amount
        for dimension in reservation.dimensions:
            if dimension.name not in normalized_capacity:
                raise FederationAuthorityError("budget capacity dimension is unavailable")
            if (
                dimension.reserved > normalized_capacity[dimension.name]
                or dimension.consumed != 0
                or dimension.reserved != dimension.ceiling
            ):
                raise FederationAuthorityError(
                    "budget reservation exceeds capacity or carries consumption"
                )

        now = utc_now()
        command = self._command(
            command_id=f"command:budget-reserve:{reservation.cid}",
            idempotency_key=reservation.idempotency_key,
            parameters={
                "operation": "budget.reserve",
                "tenant_id": reservation.binding.tenant_id,
                "federation_id": federation_id,
                "reservation_id": reservation.record_id,
            },
        )

        def apply(_txn: StateTransaction, _command: StateCommand, live: Any) -> Mapping[str, Any]:
            if reservation.binding.control_plane_generation != live.generation:
                raise FederationRepositoryConflict(
                    "budget reservation control-plane generation is stale"
                )
            existing = self.__client.execute(
                "casf_select_admission_budget_by_idempotency",
                {
                    "tenant_id": reservation.binding.tenant_id,
                    "federation_id": federation_id,
                    "idempotency_key": reservation.idempotency_key,
                },
            )
            if existing:
                admitted = BudgetReservation.from_dict(
                    _decode(existing[0]["body_json"])
                )
                if admitted != reservation or str(existing[0]["state"]) != "reserved":
                    raise FederationRepositoryConflict(
                        "budget reservation idempotency payload differs"
                    )
                return {"reservation": admitted.to_dict()}

            usage_rows = self.__client.execute(
                "casf_select_active_admission_budget_usage",
                {
                    "tenant_id": reservation.binding.tenant_id,
                    "observed_at": now,
                },
            )
            usage = {
                BudgetDimensionName(str(row["dimension_name"])): int(
                    row["reserved_amount"]
                )
                for row in usage_rows
            }
            for dimension in reservation.dimensions:
                if (
                    usage.get(dimension.name, 0) + dimension.reserved
                    > normalized_capacity[dimension.name]
                ):
                    raise FederationBoundsError(
                        f"budget capacity exhausted for {dimension.name.value}"
                    )
            self.__client.execute(
                "casf_insert_admission_budget_reservation",
                {
                    "reservation_id": reservation.record_id,
                    "tenant_id": reservation.binding.tenant_id,
                    "federation_id": federation_id,
                    "request_cid": reservation.request_cid,
                    "idempotency_key": reservation.idempotency_key,
                    "policy_id": reservation.policy_ref,
                    "policy_revision": reservation.policy_revision,
                    "resource_budget_id": reservation.resource_budget_ref,
                    "token_budget_id": reservation.token_budget_ref,
                    "parent_budget_id": reservation.parent_budget_id,
                    "owner_id": reservation.owner_id,
                    "authorization_evidence_ref": (
                        reservation.authorization_evidence_ref
                    ),
                    "issued_at": reservation.issued_at,
                    "expires_at": reservation.expires_at,
                    "state": reservation.status,
                    "revision": reservation.revision,
                    "fencing_epoch": live.fence_epoch,
                    "content_ref": reservation.cid,
                    "body_json": _json(reservation.to_dict()),
                },
            )
            for ordinal, dimension in enumerate(reservation.dimensions, start=1):
                self.__client.execute(
                    "casf_insert_admission_budget_dimension",
                    {
                        "reservation_id": reservation.record_id,
                        "tenant_id": reservation.binding.tenant_id,
                        "federation_id": federation_id,
                        "dimension_name": dimension.name.value,
                        "ceiling_amount": dimension.ceiling,
                        "reserved_amount": dimension.reserved,
                        "consumed_amount": dimension.consumed,
                        "ordinal": ordinal,
                    },
                )
            draft = EventDraft(
                event_type=EventClass.RESOURCE_PRESSURE,
                stream_id=federation_id,
                causal_parent_ids=(),
                correlation_id=f"correlation:{reservation.idempotency_key}",
                causation_id=f"causation:{reservation.request_cid}",
                tenant_id=reservation.binding.tenant_id,
                federation_id=federation_id,
                repository_id=reservation.binding.repository_ids[0],
                tree_id=reservation.binding.repository_tree_ids[0],
                payload_ref=reservation.cid,
                changed_fact_refs=(
                    reservation.record_id,
                    *(item.name.value for item in reservation.dimensions),
                ),
                effect_class=EventEffectClass.AUTHORITATIVE_STATE,
                deduplication_key=(
                    f"budget-reserve:{reservation.idempotency_key}"
                ),
            )
            event, outbox = self._allocate_event(draft, recorded_at=now)
            self._insert_event_outbox(event, outbox, binding=reservation.binding)
            return {
                "reservation": reservation.to_dict(),
                "event_id": event.event_id,
                "outbox_id": outbox.outbox_id,
                "event_global_sequence": event.global_sequence,
            }

        result = self._submit(command, apply)
        return BudgetReservation.from_dict(result.result["reservation"])  # type: ignore[return-value]

    def release_federation_budget(
        self,
        reservation_id: str,
        *,
        tenant_id: str,
        federation_id: str,
        idempotency_key: str,
        reason: str,
    ) -> None:
        """Release an unconsumed reservation with scoped idempotency."""

        reservation_ref = _identifier(reservation_id, "reservation_id")
        tenant = _identifier(tenant_id, "tenant_id")
        federation = _identifier(federation_id, "federation_id")
        caller_key = _identifier(idempotency_key, "idempotency_key")
        reason_code = _identifier(reason, "reason")
        now = utc_now()
        command = self._command(
            command_id=f"command:budget-release:{reservation_ref}:{caller_key}",
            idempotency_key=caller_key,
            command_kind=CommandKind.RELEASE,
            parameters={
                "operation": "budget.release",
                "tenant_id": tenant,
                "federation_id": federation,
                "reservation_id": reservation_ref,
            },
        )

        def apply(_txn: StateTransaction, _command: StateCommand, live: Any) -> Mapping[str, Any]:
            rows = self.__client.execute(
                "casf_select_admission_budget_by_id",
                {
                    "reservation_id": reservation_ref,
                    "tenant_id": tenant,
                    "federation_id": federation,
                },
            )
            if not rows:
                raise FederationRepositoryNotFound("budget reservation is absent")
            row = rows[0]
            reservation = BudgetReservation.from_dict(_decode(row["body_json"]))
            if (
                reservation.idempotency_key != caller_key
                or reservation.binding.tenant_id != tenant
                or reservation.owner_id != federation
            ):
                raise FederationAuthorityError("budget release scope differs")
            if str(row["state"]) == "released":
                return {"reservation_id": reservation_ref, "released": True}
            if str(row["state"]) != "reserved":
                raise FederationAuthorityError(
                    "consumed budget reservation cannot be released"
                )
            released = replace(
                reservation,
                revision=reservation.revision + 1,
                status="released",
            )
            changed = self.__client.execute(
                "casf_transition_admission_budget_reservation",
                {
                    "new_state": "released",
                    "content_ref": released.cid,
                    "body_json": _json(released.to_dict()),
                    "reservation_id": reservation_ref,
                    "tenant_id": tenant,
                    "federation_id": federation,
                    "idempotency_key": caller_key,
                    "expected_state": "reserved",
                    "expected_revision": int(row["revision"]),
                    "fencing_epoch": live.fence_epoch,
                },
            )
            if not changed:
                raise FederationRepositoryConflict("budget release CAS lost")
            draft = EventDraft(
                event_type=EventClass.RESOURCE_PRESSURE,
                stream_id=federation,
                causal_parent_ids=(),
                correlation_id=f"correlation:{caller_key}",
                causation_id=f"causation:{reservation_ref}",
                tenant_id=tenant,
                federation_id=federation,
                repository_id=reservation.binding.repository_ids[0],
                tree_id=reservation.binding.repository_tree_ids[0],
                payload_ref=released.cid,
                changed_fact_refs=(reservation_ref, reason_code),
                effect_class=EventEffectClass.AUTHORITATIVE_STATE,
                deduplication_key=f"budget-release:{caller_key}",
            )
            event, outbox = self._allocate_event(draft, recorded_at=now)
            self._insert_event_outbox(event, outbox, binding=reservation.binding)
            return {
                "reservation_id": reservation_ref,
                "released": True,
                "event_id": event.event_id,
                "outbox_id": outbox.outbox_id,
                "event_global_sequence": event.global_sequence,
            }

        self._submit(command, apply)

    def lookup_federation_creation(
        self,
        *,
        idempotency_key: str,
        tenant_id: str,
        federation_id: str,
    ) -> tuple[FederationIdentity, FederationReceipt] | None:
        """Reconcile an ambiguous create response from authoritative state.

        Only an exact ``federation.create`` idempotency result is returned.  A
        caller cannot use this read to cross a tenant/federation boundary or
        reinterpret another command's result as admission.
        """

        key = _identifier(idempotency_key, "idempotency_key")
        tenant = _identifier(tenant_id, "tenant_id")
        federation = _identifier(federation_id, "federation_id")
        scoped_key = _scoped_idempotency_key(
            operation="federation.create",
            tenant_id=tenant,
            federation_id=federation,
            caller_key=key,
        )
        rows = self.__client.execute(
            "casf_lookup_idempotency",
            {"idempotency_key": scoped_key},
        )
        if not rows:
            return None
        row = rows[0]
        if str(row["command_kind"]) != CommandKind.APPEND.value:
            raise FederationAuthorityError("idempotency record is not a create mutation")
        if not str(row["command_id"]).startswith("command:create:"):
            raise FederationAuthorityError("idempotency record is not federation.create")
        body = _decode(row["body_json"])
        if not isinstance(body, Mapping):
            raise FederationRepositoryConflict("federation create result body is corrupt")
        try:
            identity = FederationIdentity.from_dict(body["identity"])
            receipt = FederationReceipt.from_dict(body["receipt"])
        except (KeyError, TypeError, FederationContractError) as exc:
            raise FederationRepositoryConflict(
                "federation create result body is incomplete"
            ) from exc
        if identity.record_id != federation or receipt.binding.tenant_id != tenant:
            raise FederationAuthorityError("federation create result scope differs")
        if identity.binding != receipt.binding or identity.binding.tenant_id != tenant:
            raise FederationAuthorityError("federation create result binding differs")
        return identity, receipt

    def _load_admitted_policy(
        self,
        *,
        binding: FederationBinding,
        federation_id: str,
    ) -> FederationPolicy:
        rows = self.__client.execute(
            "casf_select_admitted_policy",
            {
                "policy_id": binding.policy_ref,
                "tenant_id": binding.tenant_id,
                "federation_id": federation_id,
                "policy_revision": binding.policy_revision,
            },
        )
        if not rows:
            raise FederationAuthorityError("current admitted federation policy is absent")
        row = rows[0]
        if str(row["authorization_evidence_ref"]) != binding.authorization_evidence_ref:
            raise FederationAuthorityError("policy authorization evidence differs")
        if _expired(str(row["expires_at"])):
            raise FederationAuthorityError("current admitted federation policy is expired")
        policy = FederationPolicy.from_dict(_decode(row["body_json"]))
        if policy.binding != binding:
            # The population field is a live snapshot and is verified against
            # the federation separately.  All immutable authority roots must
            # still match exactly.
            authoritative = policy.binding.to_dict()
            candidate = binding.to_dict()
            authoritative.pop("supervisor_population", None)
            candidate.pop("supervisor_population", None)
            if authoritative != candidate:
                raise FederationAuthorityError("policy authority binding differs")
        return policy

    @staticmethod
    def _validate_definition_authority(
        definition: SupervisorDefinition | SubagentDefinition,
        capabilities: Sequence[SupervisorCapability | SubagentCapability],
        *,
        binding: FederationBinding,
        policy: FederationPolicy,
        resource_budget_ref: str,
        token_budget_ref: str,
    ) -> tuple[SupervisorCapability | SubagentCapability, ...]:
        if definition.binding != binding:
            raise FederationAuthorityError("definition binding differs")
        values = tuple(capabilities)
        if not values:
            raise FederationAuthorityError("at least one admitted capability is required")
        if len(values) > 256:
            raise FederationBoundsError("capability population exceeds bound")
        if len({item.record_id for item in values}) != len(values):
            raise FederationContractError("capability identities are duplicated")
        if set(definition.capabilities) != {item.record_id for item in values}:
            raise FederationAuthorityError("definition capability set is not exact")
        if not set(definition.allowed_operations).issubset(policy.allowed_operations):
            raise FederationAuthorityError("definition operations exceed current policy")
        if definition.effect_ceiling not in policy.allowed_effects:
            raise FederationAuthorityError("definition effect ceiling exceeds current policy")
        if (
            definition.resource_budget_ref != resource_budget_ref
            or definition.token_budget_ref != token_budget_ref
        ):
            raise FederationAuthorityError("definition budget roots differ")
        for capability in values:
            if capability.binding != binding:
                raise FederationAuthorityError("capability binding differs")
            if not set(capability.allowed_operations).issubset(
                definition.allowed_operations
            ):
                raise FederationAuthorityError("capability operations exceed definition")
            if capability.effect_ceiling != definition.effect_ceiling:
                raise FederationAuthorityError("capability effect ceiling differs")
            if capability.risk_ceiling != definition.risk_ceiling:
                raise FederationAuthorityError("capability risk ceiling differs")
            if (
                capability.resource_budget_ref != resource_budget_ref
                or capability.token_budget_ref != token_budget_ref
            ):
                raise FederationAuthorityError("capability budget roots differ")
        return values

    @staticmethod
    def _assert_assignment_within_parent(
        child: SupervisorAssignment | SubagentAssignment,
        parent: SupervisorAssignment,
    ) -> None:
        """Reject every child assignment dimension not delegated by its parent."""

        comparisons = (
            (set(child.repository_ids), set(parent.repository_ids), "repositories"),
            (set(child.goal_refs), set(parent.goal_refs), "goals"),
            (set(child.task_refs), set(parent.task_refs), "tasks"),
            (
                set(child.allowed_task_families),
                set(parent.allowed_task_families),
                "task families",
            ),
        )
        for requested, admitted, name in comparisons:
            if not requested.issubset(admitted):
                raise FederationAuthorityError(
                    f"child assignment {name} exceed parent assignment"
                )

    @staticmethod
    def _assert_definition_within_parent(
        child: SupervisorDefinition | SubagentDefinition,
        parent: SupervisorDefinition,
    ) -> None:
        """Apply conservative parent ceilings where no ordered lattice exists."""

        if not set(child.allowed_operations).issubset(parent.allowed_operations):
            raise FederationAuthorityError(
                "child definition operations exceed parent definition"
            )
        if child.effect_ceiling != parent.effect_ceiling:
            raise FederationAuthorityError(
                "child effect ceiling is not conservatively bounded by parent"
            )
        if child.risk_ceiling != parent.risk_ceiling:
            raise FederationAuthorityError(
                "child risk ceiling is not conservatively bounded by parent"
            )
        if (
            child.resource_budget_ref != parent.resource_budget_ref
            or child.token_budget_ref != parent.token_budget_ref
        ):
            raise FederationAuthorityError(
                "child budget roots are not bounded by parent reservations"
            )

    def _load_supervisor_admission_authority(
        self,
        *,
        supervisor_id: str,
        tenant_id: str,
        federation_id: str,
    ) -> tuple[Mapping[str, Any], SupervisorDefinition, SupervisorAssignment]:
        rows = self.__client.execute(
            "casf_select_supervisor_admission_authority",
            {
                "supervisor_id": supervisor_id,
                "tenant_id": tenant_id,
                "federation_id": federation_id,
            },
        )
        if not rows:
            raise FederationAuthorityError(
                "supervisor has no complete authoritative admission records"
            )
        row = rows[0]
        definition = SupervisorDefinition.from_dict(_decode(row["definition_json"]))
        assignment = SupervisorAssignment.from_dict(_decode(row["assignment_json"]))
        if str(row["assignment_status"]) != "active":
            raise FederationAuthorityError("supervisor assignment is not active")
        if (
            int(row["assignment_fencing_epoch"]) != int(row["fencing_epoch"])
            or assignment.fencing_epoch != int(row["fencing_epoch"])
            or assignment.revision != int(row["assignment_revision"])
        ):
            raise FederationRepositoryConflict("supervisor assignment fence is stale")
        return row, definition, assignment

    def _load_subagent_execution_authority(
        self,
        *,
        subagent_id: str,
        tenant_id: str,
        federation_id: str,
        supervisor_id: str,
    ) -> tuple[
        Mapping[str, Any],
        SubagentInstance,
        SubagentDefinition,
        SubagentAssignment,
    ]:
        """Resolve the complete current authority used before worker admission."""

        rows = self.__client.execute(
            "casf_select_subagent_admission_authority",
            {
                "subagent_id": subagent_id,
                "tenant_id": tenant_id,
                "federation_id": federation_id,
                "supervisor_id": supervisor_id,
            },
        )
        if not rows:
            raise FederationRepositoryNotFound(
                "subagent has no complete authoritative admission records"
            )
        row = rows[0]
        instance = SubagentInstance.from_dict(_decode(row["body_json"]))
        definition = SubagentDefinition.from_dict(_decode(row["definition_json"]))
        assignment = SubagentAssignment.from_dict(_decode(row["assignment_json"]))
        if (
            instance.binding.tenant_id != tenant_id
            or instance.federation_id != federation_id
            or instance.supervisor_id != supervisor_id
            or definition.binding != instance.binding
            or assignment.binding != instance.binding
            or assignment.subject_id != instance.record_id
            or assignment.repository_ids != instance.binding.repository_ids
            or instance.task_id not in assignment.task_refs
            or assignment.fencing_epoch != instance.fencing_epoch
            or assignment.revision != int(row["assignment_revision"])
            or int(row["assignment_fencing_epoch"]) != instance.fencing_epoch
            or str(row["assignment_status"]) != "admitted"
            or str(row["assignment_lease_id"]) != instance.lease_id
            or str(row["assignment_decision_id"])
            != str(row["admission_decision_id"])
            or not str(row["admission_decision_id"])
        ):
            raise FederationAuthorityError(
                "subagent execution assignment or admission authority is stale"
            )
        policy = self._load_admitted_policy(
            binding=instance.binding,
            federation_id=federation_id,
        )
        federation_rows = self.__client.execute(
            "casf_select_federation",
            {"federation_id": federation_id, "tenant_id": tenant_id},
        )
        if not federation_rows:
            raise FederationRepositoryNotFound("parent federation is absent")
        request = FederationRequest.from_dict(_decode(federation_rows[0]["body_json"]))
        capability_rows = self.__client.execute(
            "casf_select_subagent_capability_authority",
            {
                "subagent_id": subagent_id,
                "tenant_id": tenant_id,
                "federation_id": federation_id,
            },
        )
        capabilities = tuple(
            SubagentCapability.from_dict(_decode(item["body_json"]))
            for item in capability_rows
        )
        self._validate_definition_authority(
            definition,
            capabilities,
            binding=instance.binding,
            policy=policy,
            resource_budget_ref=request.resource_budget.record_id,
            token_budget_ref=request.token_budget.record_id,
        )
        if (
            str(row["definition_policy_id"]) != instance.binding.policy_ref
            or int(row["definition_policy_revision"])
            != instance.binding.policy_revision
            or str(row["authorization_evidence_ref"])
            != instance.binding.authorization_evidence_ref
            or str(row["resource_reservation_id"])
            != definition.resource_budget_ref
            or str(row["token_reservation_id"])
            != definition.token_budget_ref
            or any(
                str(item["policy_id"]) != instance.binding.policy_ref
                or int(item["policy_revision"])
                != instance.binding.policy_revision
                or str(item["admission_decision_id"])
                != str(row["admission_decision_id"])
                or str(item["freshness_state"]) != "current"
                or not item["expires_at"]
                or _expired(str(item["expires_at"]))
                for item in capability_rows
            )
        ):
            raise FederationAuthorityError(
                "subagent execution policy, capability, or budget authority is stale"
            )
        parent, parent_definition, parent_assignment = (
            self._load_supervisor_admission_authority(
                supervisor_id=supervisor_id,
                tenant_id=tenant_id,
                federation_id=federation_id,
            )
        )
        if (
            int(parent["fencing_epoch"]) != instance.fencing_epoch
            or str(parent["lifecycle_state"])
            in {
                FederationLifecycleState.DECLARED.value,
                FederationLifecycleState.DRAINING.value,
                FederationLifecycleState.QUARANTINED.value,
                FederationLifecycleState.COMPLETED.value,
                FederationLifecycleState.FAILED.value,
                FederationLifecycleState.STOPPED.value,
            }
        ):
            raise FederationAuthorityError(
                "parent supervisor cannot admit new subagent execution"
            )
        self._assert_assignment_within_parent(assignment, parent_assignment)
        self._assert_definition_within_parent(definition, parent_definition)
        return row, instance, definition, assignment

    def _record_policy_admission(
        self,
        *,
        binding: FederationBinding,
        federation_id: str,
        subject_kind: str,
        subject_ref: str,
        operation: str,
        definition_cid: str,
        assignment_cid: str,
        capability_cids: Sequence[str],
        decided_at: str,
    ) -> str:
        decision_body = {
            "tenant_id": binding.tenant_id,
            "federation_id": federation_id,
            "policy_id": binding.policy_ref,
            "policy_revision": binding.policy_revision,
            "subject_kind": subject_kind,
            "subject_ref": subject_ref,
            "operation": operation,
            "definition_cid": definition_cid,
            "assignment_cid": assignment_cid,
            "capability_cids": list(capability_cids),
            "authorization_evidence_ref": binding.authorization_evidence_ref,
        }
        decision_id = f"policy-decision:{content_identity(decision_body)}"
        self.__client.execute(
            "casf_insert_policy_decision",
            {
                "policy_decision_id": decision_id,
                "tenant_id": binding.tenant_id,
                "federation_id": federation_id,
                "policy_id": binding.policy_ref,
                "policy_revision": binding.policy_revision,
                "subject_kind": subject_kind,
                "subject_ref": subject_ref,
                "operation": operation,
                "reason_code": "closed-policy-and-bounds-satisfied",
                "evidence_ref": f"admission-evidence:{content_identity(decision_body)}",
                "decided_at": decided_at,
            },
        )
        return decision_id

    @staticmethod
    def _assert_federation_binding(
        federation_row: Mapping[str, Any],
        binding: FederationBinding,
        *,
        supervisor_population: int,
    ) -> FederationBinding:
        """Validate an operational contract against its federation roots."""

        request = FederationRequest.from_dict(_decode(federation_row["body_json"]))
        authoritative = request.binding.to_dict()
        candidate = binding.to_dict()
        authoritative.pop("supervisor_population", None)
        candidate.pop("supervisor_population", None)
        if candidate != authoritative:
            raise FederationAuthorityError(
                "contract binding differs from the admitted federation roots"
            )
        if binding.supervisor_population != supervisor_population:
            raise FederationRepositoryConflict("contract supervisor population snapshot is stale")
        return request.binding

    def _command(
        self,
        *,
        command_id: str,
        idempotency_key: str,
        command_kind: CommandKind = CommandKind.APPEND,
        parameters: Mapping[str, Any] | None = None,
    ) -> StateCommand:
        supplied = _identifier(idempotency_key, "idempotency_key")
        scoped_parameters = dict(parameters or {})
        operation = str(scoped_parameters.get("operation") or "").strip()
        tenant_id = str(scoped_parameters.get("tenant_id") or "").strip()
        federation_id = str(scoped_parameters.get("federation_id") or "").strip()
        if operation.startswith(("federation.", "supervisor.", "subagent.", "subscription.", "event.", "budget.")):
            if not tenant_id or not federation_id:
                raise FederationAuthorityError(
                    "federation command requires exact tenant and federation scope"
                )
            # The v1 base table has a global primary key.  Namespace every
            # CASF key before it reaches that authority so equal caller keys
            # in different tenants, federations, or operations cannot collide
            # or reveal one another's result.
            stored_idempotency_key = _scoped_idempotency_key(
                operation=operation,
                tenant_id=tenant_id,
                federation_id=federation_id,
                caller_key=supplied,
            )
        else:
            stored_idempotency_key = supplied
        generation = self.__client.load_generation()
        session = self.__client.session
        if session is None:
            raise FederationRepositoryError("state client has no attached session")
        return StateCommand(
            command_id=command_id,
            command_kind=command_kind,
            store_id=generation.store_id,
            session_id=session.session_id,
            expected_generation=generation.generation,
            expected_revision=generation.revision,
            fence_epoch=generation.fence_epoch,
            idempotency_key=stored_idempotency_key,
            authority_class=StateAuthorityClass.AUTHORITATIVE,
            parameters=scoped_parameters,
        )

    def _submit(
        self,
        command: StateCommand,
        apply: Callable[[StateTransaction, StateCommand, Any], Mapping[str, Any]],
    ) -> CASResult:
        result = self.__client.submit_command(command, apply=apply)
        if result.outcome not in {
            CommandOutcome.ACCEPTED,
            CommandOutcome.IDEMPOTENT_REPLAY,
        }:
            raise FederationRepositoryConflict(
                "typed command did not commit: " + str(result.outcome.value)
            )
        # Waiters consume the persist-first delivery queue, not raw domain
        # events.  Waking them before a routing disposition commits creates a
        # check/register race with no later signal.  Only a durable route
        # commit carries this notification watermark.
        sequence = int(result.result.get("routed_global_sequence") or 0)
        if result.changed and sequence and self._event_notifier is not None:
            self._event_notifier(sequence)
        committed_sequence = int(result.result.get("event_global_sequence") or 0)
        if (
            result.changed
            and committed_sequence
            and self._outbox_notifier is not None
        ):
            self._outbox_notifier(committed_sequence)
        return result

    def _allocate_event(
        self, draft: EventDraft, *, recorded_at: str
    ) -> tuple[DomainEvent, OutboxRecord]:
        self.__client.execute("casf_seed_global_head", {"updated_at": recorded_at})
        global_rows = self.__client.execute("casf_advance_global_head", {"updated_at": recorded_at})
        self.__client.execute(
            "casf_seed_stream_head",
            {
                "stream_id": draft.stream_id,
                "tenant_id": draft.tenant_id,
                "federation_id": draft.federation_id,
                "updated_at": recorded_at,
            },
        )
        stream_rows = self.__client.execute(
            "casf_advance_stream_head",
            {
                "updated_at": recorded_at,
                "stream_id": draft.stream_id,
                "tenant_id": draft.tenant_id,
                "federation_id": draft.federation_id,
            },
        )
        if not global_rows or not stream_rows:
            raise FederationRepositoryConflict("event sequence head did not advance")
        return materialize_event(
            draft,
            stream_sequence=int(stream_rows[0]["current_sequence"]),
            global_sequence=int(global_rows[0]["current_sequence"]),
            recorded_at=recorded_at,
        )

    def _insert_event_outbox(
        self,
        event: DomainEvent,
        outbox: OutboxRecord,
        *,
        binding: FederationBinding,
    ) -> None:
        session = self.__client.session
        self.__client.execute(
            "casf_insert_domain_event",
            {
                "event_id": event.event_id,
                "event_cid": event.event_cid,
                "stream_id": event.stream_id,
                "stream_sequence": event.stream_sequence,
                "global_sequence": event.global_sequence,
                "event_type": event.event_type.value,
                "task_cid": event.task_id,
                "session_id": "" if session is None else session.session_id,
                "recorded_at": event.recorded_at,
                "body_json": _json(event.to_dict()),
                "causal_parent_ids_json": _json(list(event.causal_parent_ids)),
                "correlation_id": event.correlation_id,
                "causation_id": event.causation_id,
                "tenant_id": event.tenant_id,
                "federation_id": event.federation_id,
                "supervisor_id": event.supervisor_id,
                "repository_id": event.repository_id,
                "tree_id": event.tree_id,
                "goal_id": event.goal_id,
                "subgoal_id": event.subgoal_id,
                "symbol_id": event.symbol_id,
                "contract_id": event.contract_id,
                "proof_obligation_id": event.proof_obligation_id,
                "resource_class": event.resource_class,
                "payload_ref": event.payload_ref,
                "changed_fact_refs_json": _json(list(event.changed_fact_refs)),
                "effect_class": event.effect_class.value,
                "expires_at": event.expires_at or None,
                "deduplication_key": event.deduplication_key,
                "control_plane_generation": binding.control_plane_generation,
                "causal_graph_revision": binding.causal_graph_revision,
            },
        )
        for ordinal, parent_id in enumerate(event.causal_parent_ids, start=1):
            self.__client.execute(
                "casf_insert_event_parent",
                {"event_id": event.event_id, "parent_event_id": parent_id, "ordinal": ordinal},
            )
        for ordinal, fact_ref in enumerate(event.changed_fact_refs, start=1):
            self.__client.execute(
                "casf_insert_changed_fact",
                {"event_id": event.event_id, "fact_ref": fact_ref, "ordinal": ordinal},
            )
        if self._test_failure_hook is not None:
            self._test_failure_hook("after_event_before_outbox")
        self.__client.execute(
            "casf_insert_outbox",
            {
                "outbox_id": outbox.outbox_id,
                "event_id": outbox.event_id,
                "event_cid": outbox.event_cid,
                "tenant_id": outbox.tenant_id,
                "federation_id": outbox.federation_id,
                "stream_id": event.stream_id,
                "stream_sequence": event.stream_sequence,
                "global_sequence": outbox.global_sequence,
                "effect_class": event.effect_class.value,
                "deduplication_key": event.deduplication_key,
                "status": outbox.state.value,
                "attempt_count": outbox.attempt_count,
                "next_attempt_at": outbox.next_attempt_at,
                "created_at": outbox.created_at,
                "updated_at": outbox.created_at,
                "body_json": _json(outbox.to_dict()),
            },
        )

    @staticmethod
    def _validate_federation_creation(
        *,
        request: FederationRequest,
        policy: FederationPolicy,
        repositories: Sequence[ResolvedRepository],
        budget_reservation: BudgetReservation,
        authentication_evidence_ref: str,
        authorization_decision: FederationAuthorizationDecision,
    ) -> tuple[ResolvedRepository, ...]:
        """Fail closed when the direct admission inputs disagree.

        The trigger gateway performs the same admission checks, but the state
        owner is independently responsible for refusing a caller that bypasses
        that adapter or presents a different policy, repository resolution, or
        admitted evidence reference.
        """

        if not isinstance(request, FederationRequest):
            raise FederationContractError("request must be FederationRequest")
        if not isinstance(policy, FederationPolicy):
            raise FederationContractError("policy must be FederationPolicy")
        if policy.binding != request.binding:
            raise FederationAuthorityError("policy binding differs from request binding")
        if (
            policy.record_id != request.policy_ref
            or policy.revision != request.binding.policy_revision
        ):
            raise FederationAuthorityError("policy identity or revision differs")
        if request.caller_did not in policy.allowed_callers:
            raise FederationAuthorityError("caller is not permitted by policy")
        if request.audience not in policy.allowed_audiences:
            raise FederationAuthorityError("audience is not permitted by policy")
        if "federation.create" not in policy.allowed_operations:
            raise FederationAuthorityError("policy does not permit federation.create")
        if not set(request.effect_scope).issubset(policy.allowed_effects):
            raise FederationAuthorityError("requested effect scope exceeds policy")
        if request.maximum_supervisors > policy.maximum_supervisors:
            raise FederationAuthorityError("requested supervisor count exceeds policy")
        if request.maximum_subagents > policy.maximum_subagents:
            raise FederationAuthorityError("requested subagent count exceeds policy")
        if (
            request.resource_budget.binding != request.binding
            or request.token_budget.binding != request.binding
        ):
            raise FederationAuthorityError("request budget binding differs")

        if not isinstance(budget_reservation, BudgetReservation):
            raise FederationContractError(
                "budget_reservation must be BudgetReservation"
            )
        federation_id = f"federation:{request.cid}"
        if (
            budget_reservation.binding != request.binding
            or budget_reservation.owner_id != federation_id
            or budget_reservation.request_cid != request.cid
            or budget_reservation.idempotency_key != request.idempotency_key
            or budget_reservation.policy_ref != policy.record_id
            or budget_reservation.policy_revision != policy.revision
            or budget_reservation.resource_budget_ref
            != request.resource_budget.record_id
            or budget_reservation.token_budget_ref != request.token_budget.record_id
            or budget_reservation.status != "reserved"
            or _expired(budget_reservation.expires_at)
        ):
            raise FederationAuthorityError(
                "budget reservation differs from the admitted request"
            )
        admitted_evidence = _identifier(
            authentication_evidence_ref,
            "authentication_evidence_ref",
        )
        if admitted_evidence != request.binding.authorization_evidence_ref:
            raise FederationAuthorityError(
                "authentication evidence differs from the admitted authorization evidence"
            )
        if not isinstance(authorization_decision, FederationAuthorizationDecision):
            raise FederationContractError(
                "authorization_decision must be FederationAuthorizationDecision"
            )
        if (
            authorization_decision.request_cid != request.cid
            or authorization_decision.caller_did != request.caller_did
            or authorization_decision.audience != request.audience
            or authorization_decision.operation.value != "federation.create"
            or authorization_decision.policy_id != policy.record_id
            or authorization_decision.policy_revision != policy.revision
            or authorization_decision.expires_at != request.expiry
        ):
            raise FederationAuthorityError(
                "authorization decision differs from the admitted request"
            )

        if isinstance(repositories, (str, bytes)) or not isinstance(repositories, Sequence):
            raise FederationContractError("repositories must be a bounded sequence")
        resolved = tuple(repositories)
        if not resolved or len(resolved) != len(request.repository_roots):
            raise FederationAuthorityError("not every repository root resolved")
        if any(not isinstance(item, ResolvedRepository) for item in resolved):
            raise FederationContractError("repositories contains a non-ResolvedRepository value")
        requested_refs = tuple(
            _identifier(item.requested_ref, "repository requested_ref") for item in resolved
        )
        repository_ids = tuple(
            _identifier(item.repository_id, "repository_id") for item in resolved
        )
        tree_ids = tuple(_identifier(item.tree_id, "tree_id") for item in resolved)
        semantic_roots = tuple(
            _identifier(item.semantic_state_root, "semantic_state_root") for item in resolved
        )
        if len(set(repository_ids)) != len(repository_ids):
            raise FederationAuthorityError("resolved repository identities are duplicated")
        if requested_refs != request.repository_roots:
            raise FederationAuthorityError("resolved requested roots differ")
        if repository_ids != request.binding.repository_ids:
            raise FederationAuthorityError("resolved repository identities differ")
        if tree_ids != request.binding.repository_tree_ids:
            raise FederationAuthorityError("resolved repository trees differ")
        if semantic_roots != request.binding.semantic_state_roots:
            raise FederationAuthorityError("resolved semantic roots differ")
        expected_scope_cid = resolved_authorization_scope_identity(
            resolved,
            request.effect_scope,
        )
        if authorization_decision.resolved_scope_cid != expected_scope_cid:
            raise FederationAuthorityError(
                "authorization decision resolved scope differs"
            )
        return resolved

    def create_federation(
        self,
        *,
        request: FederationRequest,
        policy: FederationPolicy,
        repositories: Sequence[ResolvedRepository],
        budget_reservation: BudgetReservation,
        authentication_evidence_ref: str,
        authorization_decision: FederationAuthorizationDecision,
    ) -> tuple[FederationIdentity, FederationReceipt]:
        repositories = self._validate_federation_creation(
            request=request,
            policy=policy,
            repositories=repositories,
            budget_reservation=budget_reservation,
            authentication_evidence_ref=authentication_evidence_ref,
            authorization_decision=authorization_decision,
        )
        federation_id = f"federation:{request.cid}"
        now = utc_now()
        command = self._command(
            command_id=f"command:create:{request.cid}",
            idempotency_key=request.idempotency_key,
            parameters={
                "operation": "federation.create",
                "federation_id": federation_id,
                "tenant_id": request.binding.tenant_id,
                "request_cid": request.cid,
            },
        )

        def apply(_txn: StateTransaction, _command: StateCommand, live: Any) -> Mapping[str, Any]:
            if request.binding.control_plane_generation != live.generation:
                raise FederationRepositoryConflict("request control-plane generation is stale")
            if request.binding.supervisor_population != 0:
                raise FederationAuthorityError(
                    "a creation request cannot assert an existing supervisor population"
                )
            reservation_rows = self.__client.execute(
                "casf_select_admission_budget_by_id",
                {
                    "reservation_id": budget_reservation.record_id,
                    "tenant_id": request.binding.tenant_id,
                    "federation_id": federation_id,
                },
            )
            if not reservation_rows:
                raise FederationAuthorityError(
                    "authoritative budget reservation is absent"
                )
            reservation_row = reservation_rows[0]
            persisted_reservation = BudgetReservation.from_dict(
                _decode(reservation_row["body_json"])
            )
            if (
                persisted_reservation != budget_reservation
                or str(reservation_row["state"]) != "reserved"
                or int(reservation_row["revision"]) != budget_reservation.revision
                or int(reservation_row["fencing_epoch"]) != live.fence_epoch
                or _expired(str(reservation_row["expires_at"]))
            ):
                raise FederationAuthorityError(
                    "authoritative budget reservation is stale or differs"
                )
            consumed_reservation = replace(
                budget_reservation,
                revision=budget_reservation.revision + 1,
                status="consumed",
            )
            consumed = self.__client.execute(
                "casf_transition_admission_budget_reservation",
                {
                    "new_state": "consumed",
                    "content_ref": consumed_reservation.cid,
                    "body_json": _json(consumed_reservation.to_dict()),
                    "reservation_id": budget_reservation.record_id,
                    "tenant_id": request.binding.tenant_id,
                    "federation_id": federation_id,
                    "idempotency_key": budget_reservation.idempotency_key,
                    "expected_state": "reserved",
                    "expected_revision": budget_reservation.revision,
                    "fencing_epoch": live.fence_epoch,
                },
            )
            if not consumed:
                raise FederationRepositoryConflict(
                    "budget reservation consumption CAS lost"
                )
            self.__client.execute(
                "casf_insert_policy",
                {
                    "policy_id": policy.record_id,
                    "tenant_id": request.binding.tenant_id,
                    "federation_id": federation_id,
                    "revision": policy.revision,
                    "issuer_id": policy.binding.issuer,
                    "authorization_evidence_ref": authentication_evidence_ref,
                    "expires_at": policy.binding.expires_at,
                    "maximum_supervisors": policy.maximum_supervisors,
                    "maximum_subagents": policy.maximum_subagents,
                    "maximum_concurrent_subagents": policy.maximum_concurrent_subagents,
                    "status": "admitted",
                    "content_ref": policy.cid,
                    "created_at": now,
                    "updated_at": now,
                    "body_json": _json(policy.to_dict()),
                },
            )
            self.__client.execute(
                "casf_insert_authorization_decision",
                {
                    "authorization_decision_id": authorization_decision.cid,
                    "tenant_id": request.binding.tenant_id,
                    "federation_id": federation_id,
                    "request_cid": request.cid,
                    "caller_id": request.caller_did,
                    "delegation_chain_ref": (
                        authorization_decision.delegation_chain_cid
                    ),
                    "audience": request.audience,
                    "operation": authorization_decision.operation.value,
                    "resource_scope_ref": authorization_decision.resolved_scope_cid,
                    "policy_id": policy.record_id,
                    "policy_revision": policy.revision,
                    "verdict": authorization_decision.verdict.value,
                    "reason_code": authorization_decision.reason.value,
                    "evidence_ref": (
                        authorization_decision.authentication_evidence_cid
                    ),
                    "expires_at": authorization_decision.expires_at,
                    "decided_at": authorization_decision.decided_at,
                    "content_ref": authorization_decision.cid,
                    "body_json": _json(authorization_decision.to_dict()),
                },
            )
            semantic_root = content_identity([item.semantic_state_root for item in repositories])
            self.__client.execute(
                "casf_insert_federation",
                {
                    "federation_id": federation_id,
                    "tenant_id": request.binding.tenant_id,
                    "program_id": request.program_id,
                    "objective_ref": request.objective_ref,
                    "objective_revision": request.binding.objective_revision,
                    "policy_id": policy.record_id,
                    "policy_revision": policy.revision,
                    "operation_catalog_id": request.binding.operation_catalog_ref,
                    "control_plane_generation": live.generation,
                    "causal_graph_revision": request.binding.causal_graph_revision,
                    "semantic_state_root": semantic_root,
                    "status": FederationLifecycleState.ADMITTED.value,
                    "maximum_supervisors": request.maximum_supervisors,
                    "maximum_subagents": request.maximum_subagents,
                    "revision": 1,
                    "fencing_epoch": live.fence_epoch,
                    "issuer_id": request.binding.issuer,
                    "authorization_evidence_ref": authentication_evidence_ref,
                    "expires_at": request.expiry,
                    "created_at": now,
                    "updated_at": now,
                    "content_ref": request.cid,
                    "body_json": _json(request.to_dict()),
                },
            )
            if self._test_failure_hook is not None:
                self._test_failure_hook("after_state_before_event")
            self.__client.execute(
                "casf_insert_federation_budget",
                {
                    "federation_budget_id": request.binding.budget_ref,
                    "tenant_id": request.binding.tenant_id,
                    "federation_id": federation_id,
                    "parent_budget_id": "",
                    "owner_id": request.caller_did,
                    "policy_id": policy.record_id,
                    "policy_revision": policy.revision,
                    "revision": 1,
                    "status": "reserved",
                    "content_ref": budget_reservation.record_id,
                    "created_at": now,
                    "updated_at": now,
                    "body_json": _json(
                        {
                            "resource_budget": request.resource_budget.to_dict(),
                            "token_budget": request.token_budget.to_dict(),
                            "reservation_ref": budget_reservation.record_id,
                        }
                    ),
                },
            )
            concurrent_slots = min(
                request.maximum_subagents,
                policy.maximum_concurrent_subagents,
            )
            for slot_number in range(1, concurrent_slots + 1):
                self.__client.execute(
                    "casf_seed_subagent_slot",
                    {
                        "tenant_id": request.binding.tenant_id,
                        "federation_id": federation_id,
                        "slot_number": slot_number,
                        "fencing_epoch": live.fence_epoch,
                    },
                )
            draft = EventDraft(
                event_type=EventClass.GOAL_CHANGED,
                stream_id=federation_id,
                causal_parent_ids=(),
                correlation_id=f"correlation:{request.idempotency_key}",
                causation_id=f"causation:{request.cid}",
                tenant_id=request.binding.tenant_id,
                federation_id=federation_id,
                repository_id=request.binding.repository_ids[0],
                tree_id=request.binding.repository_tree_ids[0],
                goal_id=request.objective_ref,
                payload_ref=request.cid,
                changed_fact_refs=(federation_id, request.objective_ref),
                effect_class=EventEffectClass.AUTHORITATIVE_STATE,
                deduplication_key=f"federation-create:{request.idempotency_key}",
            )
            event, outbox = self._allocate_event(draft, recorded_at=now)
            self._insert_event_outbox(event, outbox, binding=request.binding)
            identity = FederationIdentity(
                record_id=federation_id,
                revision=1,
                binding=request.binding,
            )
            receipt = FederationReceipt(
                record_id=f"receipt:{event.event_cid}",
                revision=1,
                binding=request.binding,
                outcome="accepted",
                evidence_refs=tuple(
                    dict.fromkeys(
                        (
                            event.event_id,
                            outbox.outbox_id,
                            authentication_evidence_ref,
                            authorization_decision.authentication_evidence_cid,
                            authorization_decision.cid,
                            budget_reservation.record_id,
                        )
                    )
                ),
                recorded_at=now,
            )
            return {
                "identity": identity.to_dict(),
                "receipt": receipt.to_dict(),
                "event_id": event.event_id,
                "outbox_id": outbox.outbox_id,
                "event_global_sequence": event.global_sequence,
            }

        result = self._submit(command, apply)
        return (
            FederationIdentity.from_dict(result.result["identity"]),  # type: ignore[return-value]
            FederationReceipt.from_dict(result.result["receipt"]),  # type: ignore[return-value]
        )

    def register_supervisor(
        self,
        instance: SupervisorInstance,
        assignment: SupervisorAssignment,
        *,
        definition: SupervisorDefinition,
        capabilities: Sequence[SupervisorCapability],
        idempotency_key: str,
    ) -> SupervisorInstance:
        if not isinstance(definition, SupervisorDefinition):
            raise FederationContractError("definition must be SupervisorDefinition")
        capability_values = tuple(capabilities)
        if any(not isinstance(item, SupervisorCapability) for item in capability_values):
            raise FederationContractError("capabilities must be SupervisorCapability records")
        if instance.record_id != assignment.subject_id:
            raise FederationContractError("supervisor assignment subject differs")
        if instance.binding != assignment.binding:
            raise FederationAuthorityError("supervisor assignment binding differs")
        if instance.binding.repository_ids != assignment.repository_ids:
            raise FederationAuthorityError("supervisor repository boundary differs")
        if FederationLifecycleState(instance.state) not in {
            FederationLifecycleState.DECLARED,
            FederationLifecycleState.ADMITTED,
        }:
            raise FederationAuthorityError(
                "new supervisor must be declared or admitted before execution"
            )
        if instance.role is SupervisorRole.COORDINATOR and instance.parent_supervisor_id:
            raise FederationAuthorityError("coordinator supervisor cannot have a parent")
        if instance.role is not SupervisorRole.COORDINATOR and not instance.parent_supervisor_id:
            raise FederationAuthorityError(
                "non-coordinator supervisor requires a bounded parent supervisor"
            )
        command = self._command(
            command_id=f"command:supervisor-register:{instance.cid}",
            idempotency_key=idempotency_key,
            parameters={
                "operation": "supervisor.register",
                "tenant_id": instance.binding.tenant_id,
                "supervisor_id": instance.record_id,
                "federation_id": instance.federation_id,
            },
        )
        now = utc_now()

        def apply(_txn: StateTransaction, _command: StateCommand, live: Any) -> Mapping[str, Any]:
            rows = self.__client.execute(
                "casf_select_federation",
                {
                    "federation_id": instance.federation_id,
                    "tenant_id": instance.binding.tenant_id,
                },
            )
            if not rows:
                raise FederationRepositoryNotFound("parent federation is absent")
            federation = rows[0]
            counts = self.__client.execute(
                "casf_count_supervisors",
                {
                    "tenant_id": instance.binding.tenant_id,
                    "federation_id": instance.federation_id,
                },
            )
            population = int(counts[0]["population"] if counts else 0)
            if population >= int(federation["maximum_supervisors"]):
                raise FederationAuthorityError("supervisor population ceiling reached")
            self._assert_federation_binding(
                federation,
                instance.binding,
                supervisor_population=population,
            )
            request = FederationRequest.from_dict(_decode(federation["body_json"]))
            policy = self._load_admitted_policy(
                binding=instance.binding,
                federation_id=instance.federation_id,
            )
            admitted_capabilities = self._validate_definition_authority(
                definition,
                capability_values,
                binding=instance.binding,
                policy=policy,
                resource_budget_ref=request.resource_budget.record_id,
                token_budget_ref=request.token_budget.record_id,
            )
            if (
                instance.fencing_epoch != int(federation["fencing_epoch"])
                or instance.fencing_epoch != live.fence_epoch
            ):
                raise FederationRepositoryConflict("supervisor registration fence is stale")
            if instance.role is SupervisorRole.COORDINATOR:
                coordinator_counts = self.__client.execute(
                    "casf_count_coordinator_supervisors",
                    {
                        "tenant_id": instance.binding.tenant_id,
                        "federation_id": instance.federation_id,
                    },
                )
                if int(coordinator_counts[0]["population"] if coordinator_counts else 0):
                    raise FederationAuthorityError(
                        "federation already has a live coordinator supervisor"
                    )
            if instance.parent_supervisor_id:
                parent, parent_definition, parent_assignment = (
                    self._load_supervisor_admission_authority(
                        supervisor_id=instance.parent_supervisor_id,
                        tenant_id=instance.binding.tenant_id,
                        federation_id=instance.federation_id,
                    )
                )
                if str(parent["lifecycle_state"]) in {
                    FederationLifecycleState.DRAINING.value,
                    FederationLifecycleState.QUARANTINED.value,
                    FederationLifecycleState.COMPLETED.value,
                    FederationLifecycleState.FAILED.value,
                    FederationLifecycleState.STOPPED.value,
                }:
                    raise FederationAuthorityError(
                        "terminal or draining parent cannot create a child supervisor"
                    )
                if int(parent["fencing_epoch"]) != instance.fencing_epoch:
                    raise FederationRepositoryConflict("parent supervisor fence is stale")
                self._assert_assignment_within_parent(assignment, parent_assignment)
                self._assert_definition_within_parent(definition, parent_definition)
            repository_id = assignment.repository_ids[0]
            tree_id = instance.binding.repository_tree_ids[
                instance.binding.repository_ids.index(repository_id)
            ]
            admission_decision_id = self._record_policy_admission(
                binding=instance.binding,
                federation_id=instance.federation_id,
                subject_kind="supervisor",
                subject_ref=instance.record_id,
                operation="supervisor.register",
                definition_cid=definition.cid,
                assignment_cid=assignment.cid,
                capability_cids=tuple(item.cid for item in admitted_capabilities),
                decided_at=now,
            )
            self.__client.execute(
                "casf_insert_supervisor_definition",
                {
                    "supervisor_definition_id": definition.record_id,
                    "tenant_id": instance.binding.tenant_id,
                    "federation_id": instance.federation_id,
                    "specialization": definition.name,
                    "capability_set_ref": content_identity(
                        [item.record_id for item in admitted_capabilities]
                    ),
                    "allowed_operations_ref": content_identity(
                        list(definition.allowed_operations)
                    ),
                    "effect_ceiling": definition.effect_ceiling,
                    "risk_ceiling": definition.risk_ceiling,
                    "resource_ceiling_ref": definition.resource_budget_ref,
                    "token_ceiling_ref": definition.token_budget_ref,
                    "proof_requirements_ref": f"proof-requirements:{definition.cid}",
                    "merge_policy_ref": f"merge-policy:{definition.cid}",
                    "policy_id": instance.binding.policy_ref,
                    "policy_revision": instance.binding.policy_revision,
                    "authorization_evidence_ref": (
                        instance.binding.authorization_evidence_ref
                    ),
                    "content_ref": definition.cid,
                    "created_at": now,
                    "body_json": _json(definition.to_dict()),
                },
            )
            for capability in admitted_capabilities:
                self.__client.execute(
                    "casf_insert_supervisor_capability",
                    {
                        "capability_record_id": capability.record_id,
                        "tenant_id": instance.binding.tenant_id,
                        "federation_id": instance.federation_id,
                        "supervisor_id": instance.record_id,
                        "capability_kind": capability.name,
                        "capability_revision": capability.revision,
                        "observed_generation": live.generation,
                        "evidence_ref": capability.cid,
                        "policy_id": instance.binding.policy_ref,
                        "policy_revision": instance.binding.policy_revision,
                        "admission_decision_id": admission_decision_id,
                        "content_ref": capability.cid,
                        "expires_at": instance.binding.expires_at,
                        "recorded_at": now,
                        "body_json": _json(capability.to_dict()),
                    },
                )
            self.__client.execute(
                "casf_insert_supervisor",
                {
                    "supervisor_id": instance.record_id,
                    "repository_id": repository_id,
                    "process_birth_id": "logical:not-started",
                    "started_at": now,
                    "status": instance.state,
                    "revision": instance.revision,
                    "extension_schema": instance.SCHEMA,
                    "extension_json": _json(instance.to_dict()),
                    "tenant_id": instance.binding.tenant_id,
                    "federation_id": instance.federation_id,
                    "parent_supervisor_id": instance.parent_supervisor_id,
                    "supervisor_definition_id": definition.record_id,
                    "role": instance.role.value,
                    "lifecycle_state": instance.state,
                    "assignment_revision": assignment.revision,
                    "lease_id": instance.lease_id,
                    "fencing_epoch": instance.fencing_epoch,
                    "policy_id": instance.binding.policy_ref,
                    "policy_revision": instance.binding.policy_revision,
                    "admission_decision_id": admission_decision_id,
                },
            )
            self.__client.execute(
                "casf_insert_supervisor_assignment",
                {
                    "assignment_id": assignment.record_id,
                    "tenant_id": instance.binding.tenant_id,
                    "federation_id": instance.federation_id,
                    "supervisor_id": instance.record_id,
                    "parent_supervisor_id": instance.parent_supervisor_id,
                    "assignment_revision": assignment.revision,
                    "repository_id": repository_id,
                    "tree_id": tree_id,
                    "goal_ref": assignment.goal_refs[0] if assignment.goal_refs else "",
                    "task_family": assignment.allowed_task_families[0]
                    if assignment.allowed_task_families
                    else "",
                    "lease_id": instance.lease_id,
                    "fencing_epoch": instance.fencing_epoch,
                    "status": "active",
                    "admission_decision_id": admission_decision_id,
                    "content_ref": assignment.cid,
                    "created_at": now,
                    "updated_at": now,
                    "body_json": _json(assignment.to_dict()),
                },
            )
            draft = EventDraft(
                event_type=EventClass.SUPERVISOR_HEALTH_CHANGED,
                stream_id=instance.federation_id,
                causal_parent_ids=(),
                correlation_id=f"correlation:{idempotency_key}",
                causation_id=f"causation:{instance.cid}",
                tenant_id=instance.binding.tenant_id,
                federation_id=instance.federation_id,
                supervisor_id=instance.record_id,
                repository_id=repository_id,
                tree_id=tree_id,
                payload_ref=instance.cid,
                changed_fact_refs=(instance.record_id, assignment.record_id),
                effect_class=EventEffectClass.AUTHORITATIVE_STATE,
                deduplication_key=f"supervisor-register:{idempotency_key}",
            )
            event, outbox = self._allocate_event(draft, recorded_at=now)
            self._insert_event_outbox(event, outbox, binding=instance.binding)
            return {
                "instance": instance.to_dict(),
                "event_global_sequence": event.global_sequence,
            }

        result = self._submit(command, apply)
        return SupervisorInstance.from_dict(result.result["instance"])  # type: ignore[return-value]

    def attest_supervisor_runtime(
        self,
        *,
        supervisor_id: str,
        tenant_id: str,
        federation_id: str,
        expected_revision: int,
        expected_fencing_epoch: int,
        idempotency_key: str,
    ) -> Mapping[str, Any]:
        """Bind an admitted supervisor to a state-owner-observed process lease."""

        supervisor_id = _identifier(supervisor_id, "supervisor_id")
        tenant_id = _identifier(tenant_id, "tenant_id")
        federation_id = _identifier(federation_id, "federation_id")
        expected_revision = _integer(expected_revision, "expected_revision", minimum=1)
        expected_fencing_epoch = _integer(
            expected_fencing_epoch, "expected_fencing_epoch", minimum=1
        )
        key = _identifier(idempotency_key, "idempotency_key")
        command = self._command(
            command_id=(
                f"command:supervisor-runtime-attest:{supervisor_id}:"
                f"{expected_revision}:{expected_fencing_epoch}:{key}"
            ),
            idempotency_key=key,
            command_kind=CommandKind.CLAIM,
            parameters={
                "operation": "supervisor.runtime.attest",
                "tenant_id": tenant_id,
                "federation_id": federation_id,
                "supervisor_id": supervisor_id,
                "expected_revision": expected_revision,
                "expected_fencing_epoch": expected_fencing_epoch,
            },
        )

        def apply(
            _txn: StateTransaction,
            _command: StateCommand,
            live: Any,
        ) -> Mapping[str, Any]:
            if live.fence_epoch != expected_fencing_epoch:
                raise FederationRepositoryConflict("supervisor runtime fence is stale")
            rows = self.__client.execute(
                "casf_select_supervisor",
                {
                    "supervisor_id": supervisor_id,
                    "tenant_id": tenant_id,
                    "federation_id": federation_id,
                },
            )
            if not rows:
                raise FederationRepositoryNotFound("supervisor is absent")
            row = rows[0]
            if (
                int(row["revision"]) != expected_revision
                or int(row["fencing_epoch"]) != expected_fencing_epoch
            ):
                raise FederationRepositoryConflict(
                    "supervisor runtime revision or fence is stale"
                )
            if str(row["lifecycle_state"]) not in {
                FederationLifecycleState.ADMITTED.value,
                FederationLifecycleState.STARTING.value,
                FederationLifecycleState.IDLE.value,
                FederationLifecycleState.ACTIVE.value,
                FederationLifecycleState.PAUSED.value,
                FederationLifecycleState.RECOVERING.value,
            }:
                raise FederationAuthorityError(
                    "supervisor lifecycle is not eligible for runtime admission"
                )
            observed = self._runtime_now()
            issued_at = observed.isoformat().replace("+00:00", "Z")
            expires_at = (
                observed + timedelta(seconds=self._supervisor_runtime_lease_seconds)
            ).isoformat().replace("+00:00", "Z")
            attestation_id, birth, canonical_birth_id = self._attest_owner_process(
                tenant_id=tenant_id,
                federation_id=federation_id,
                supervisor_id=supervisor_id,
                recorded_at=issued_at,
            )
            latest_rows = self.__client.execute(
                "casf_select_latest_supervisor_runtime_revision",
                {
                    "tenant_id": tenant_id,
                    "federation_id": federation_id,
                    "supervisor_id": supervisor_id,
                    "lease_id": row["lease_id"],
                    "fencing_epoch": expected_fencing_epoch,
                },
            )
            if latest_rows and str(latest_rows[0]["process_birth_id"]) != attestation_id:
                raise FederationRepositoryConflict(
                    "runtime process takeover requires a new fencing epoch"
                )
            runtime_revision = (
                int(latest_rows[0]["revision"]) + 1 if latest_rows else 1
            )
            if latest_rows:
                superseded = self.__client.execute(
                    "casf_supersede_supervisor_runtime_lease",
                    {
                        "revoked_at": issued_at,
                        "runtime_lease_id": latest_rows[0]["runtime_lease_id"],
                        "tenant_id": tenant_id,
                        "federation_id": federation_id,
                        "supervisor_id": supervisor_id,
                        "expected_revision": latest_rows[0]["revision"],
                    },
                )
                if not superseded:
                    raise FederationRepositoryConflict(
                        "current supervisor runtime lease changed during renewal"
                    )
            evidence_body = {
                "tenant_id": tenant_id,
                "federation_id": federation_id,
                "supervisor_id": supervisor_id,
                "lease_id": row["lease_id"],
                "process_birth_id": attestation_id,
                "canonical_birth_id": canonical_birth_id,
                "process": birth.to_dict(),
                "issued_at": issued_at,
                "expires_at": expires_at,
                "fencing_epoch": expected_fencing_epoch,
                "revision": runtime_revision,
            }
            evidence_ref = "runtime-attestation:" + content_identity(evidence_body)
            runtime_lease_id = "supervisor-runtime-lease:" + content_identity(
                evidence_body
            )
            self.__client.execute(
                "casf_insert_supervisor_runtime_lease",
                {
                    "runtime_lease_id": runtime_lease_id,
                    "tenant_id": tenant_id,
                    "federation_id": federation_id,
                    "supervisor_id": supervisor_id,
                    "lease_id": row["lease_id"],
                    "process_birth_id": attestation_id,
                    "process_id": birth.pid,
                    "process_start_time_ticks": birth.start_time_ticks,
                    "process_boot_id": birth.boot_id,
                    "process_parent_id": birth.parent_pid,
                    "issued_at": issued_at,
                    "expires_at": expires_at,
                    "fencing_epoch": expected_fencing_epoch,
                    "revision": runtime_revision,
                    "evidence_ref": evidence_ref,
                    "body_json": _json(evidence_body),
                },
            )
            updated = self.__client.execute(
                "casf_update_supervisor_process_birth",
                {
                    "process_birth_id": attestation_id,
                    "supervisor_id": supervisor_id,
                    "tenant_id": tenant_id,
                    "federation_id": federation_id,
                    "expected_revision": expected_revision,
                    "fencing_epoch": expected_fencing_epoch,
                    "current_process_birth_id": attestation_id,
                },
            )
            if not updated:
                raise FederationRepositoryConflict(
                    "supervisor runtime process identity was already established"
                )
            binding = FederationBinding.from_dict(
                _decode(row["extension_json"])["binding"]
            )
            draft = EventDraft(
                event_type=EventClass.SUPERVISOR_HEALTH_CHANGED,
                stream_id=federation_id,
                causal_parent_ids=(),
                correlation_id=f"correlation:{key}",
                causation_id=f"causation:{runtime_lease_id}",
                tenant_id=tenant_id,
                federation_id=federation_id,
                supervisor_id=supervisor_id,
                repository_id=binding.repository_ids[0],
                tree_id=binding.repository_tree_ids[0],
                payload_ref=evidence_ref,
                changed_fact_refs=(
                    supervisor_id,
                    runtime_lease_id,
                    attestation_id,
                ),
                effect_class=EventEffectClass.AUTHORITATIVE_STATE,
                deduplication_key=f"supervisor-runtime-attest:{key}",
            )
            event, outbox = self._allocate_event(draft, recorded_at=issued_at)
            self._insert_event_outbox(event, outbox, binding=binding)
            return {
                "runtime_lease_id": runtime_lease_id,
                "process_birth_id": attestation_id,
                "evidence_ref": evidence_ref,
                "issued_at": issued_at,
                "expires_at": expires_at,
                "revision": runtime_revision,
                "event_global_sequence": event.global_sequence,
            }

        return MappingProxyType(dict(self._submit(command, apply).result))

    def transition_supervisor(
        self,
        *,
        supervisor_id: str,
        tenant_id: str,
        federation_id: str,
        requested_state: FederationLifecycleState,
        expected_revision: int,
        expected_fencing_epoch: int,
        active_effects: int,
        active_attempts: int,
        idempotency_key: str,
    ) -> Mapping[str, Any]:
        # Retained for wire compatibility only.  Lifecycle safety is derived
        # from authoritative attempts/effect reservations inside the same
        # transaction; caller-supplied zeros cannot manufacture completion.
        _integer(active_effects, "active_effects")
        _integer(active_attempts, "active_attempts")
        command = self._command(
            command_id=f"command:supervisor-transition:{supervisor_id}:{expected_revision}",
            idempotency_key=idempotency_key,
            parameters={
                "operation": "supervisor.transition",
                "tenant_id": tenant_id,
                "federation_id": federation_id,
                "supervisor_id": supervisor_id,
                "requested_state": requested_state.value,
            },
        )
        now = utc_now()

        def apply(_txn: StateTransaction, _command: StateCommand, live: Any) -> Mapping[str, Any]:
            rows = self.__client.execute(
                "casf_select_supervisor",
                {
                    "supervisor_id": supervisor_id,
                    "tenant_id": tenant_id,
                    "federation_id": federation_id,
                },
            )
            if not rows:
                raise FederationRepositoryNotFound("supervisor is absent")
            row = rows[0]
            if expected_fencing_epoch != live.fence_epoch:
                raise FederationRepositoryConflict("supervisor store fence is stale")
            attempt_rows = self.__client.execute(
                "casf_count_supervisor_active_attempts",
                {
                    "tenant_id": tenant_id,
                    "federation_id": federation_id,
                    "supervisor_id": supervisor_id,
                },
            )
            effect_rows = self.__client.execute(
                "casf_count_supervisor_active_effects",
                {
                    "tenant_id": tenant_id,
                    "federation_id": federation_id,
                    "supervisor_id": supervisor_id,
                },
            )
            slot_rows = self.__client.execute(
                "casf_count_supervisor_active_slots",
                {
                    "tenant_id": tenant_id,
                    "federation_id": federation_id,
                    "supervisor_id": supervisor_id,
                },
            )
            authoritative_attempts = int(
                attempt_rows[0]["active_attempts"] if attempt_rows else 0
            )
            # An admitted execution slot is itself authoritative evidence that
            # work is still active.  Do not let a caller complete a draining
            # supervisor merely because its attempt row is missing or lagging.
            authoritative_attempts += int(
                slot_rows[0]["active_slots"] if slot_rows else 0
            )
            authoritative_effects = int(
                effect_rows[0]["active_effects"] if effect_rows else 0
            )
            target = assert_transition(
                str(row["lifecycle_state"]),
                requested_state,
                active_effects=authoritative_effects,
                active_attempts=authoritative_attempts,
            )
            if target in {
                FederationLifecycleState.STARTING,
                FederationLifecycleState.ACTIVE,
                FederationLifecycleState.COMPLETED,
            }:
                self._assert_current_supervisor_runtime(
                    row=row,
                    tenant_id=tenant_id,
                    federation_id=federation_id,
                    supervisor_id=supervisor_id,
                    observed_at=now,
                )
            updated = self.__client.execute(
                "casf_update_supervisor_lifecycle",
                {
                    "lifecycle_state": target.value,
                    "status": target.value,
                    "new_fencing_epoch": expected_fencing_epoch,
                    "supervisor_id": supervisor_id,
                    "tenant_id": tenant_id,
                    "federation_id": federation_id,
                    "expected_revision": expected_revision,
                    "expected_fencing_epoch": expected_fencing_epoch,
                },
            )
            if not updated:
                raise FederationRepositoryConflict("supervisor revision or fence is stale")
            binding = FederationBinding.from_dict(_decode(row["extension_json"])["binding"])
            draft = EventDraft(
                event_type=EventClass.SUPERVISOR_HEALTH_CHANGED,
                stream_id=federation_id,
                causal_parent_ids=(),
                correlation_id=f"correlation:{idempotency_key}",
                causation_id=f"causation:{supervisor_id}:{expected_revision}",
                tenant_id=tenant_id,
                federation_id=federation_id,
                supervisor_id=supervisor_id,
                payload_ref=f"state:{target.value}",
                changed_fact_refs=(supervisor_id,),
                effect_class=EventEffectClass.AUTHORITATIVE_STATE,
                deduplication_key=f"supervisor-transition:{idempotency_key}",
            )
            event, outbox = self._allocate_event(draft, recorded_at=now)
            self._insert_event_outbox(event, outbox, binding=binding)
            return {
                "supervisor_id": supervisor_id,
                "state": target.value,
                "revision": int(updated[0]["revision"]),
                "event_global_sequence": event.global_sequence,
            }

        return MappingProxyType(dict(self._submit(command, apply).result))

    def register_subagent(
        self,
        instance: SubagentInstance,
        *,
        definition: SubagentDefinition,
        assignment: SubagentAssignment,
        capabilities: Sequence[SubagentCapability],
        idempotency_key: str | None = None,
    ) -> SubagentInstance:
        if not isinstance(instance, SubagentInstance):
            raise FederationContractError("logical registration requires SubagentInstance")
        if not isinstance(definition, SubagentDefinition):
            raise FederationContractError("definition must be SubagentDefinition")
        if not isinstance(assignment, SubagentAssignment):
            raise FederationContractError("assignment must be SubagentAssignment")
        capability_values = tuple(capabilities)
        if any(not isinstance(item, SubagentCapability) for item in capability_values):
            raise FederationContractError("capabilities must be SubagentCapability records")
        if assignment.subject_id != instance.record_id:
            raise FederationContractError("subagent assignment subject differs")
        if assignment.binding != instance.binding or definition.binding != instance.binding:
            raise FederationAuthorityError("subagent admission binding differs")
        if assignment.repository_ids != instance.binding.repository_ids:
            raise FederationAuthorityError("subagent repository boundary differs")
        if assignment.fencing_epoch != instance.fencing_epoch:
            raise FederationRepositoryConflict("subagent assignment fence differs")
        if instance.task_id and instance.task_id not in assignment.task_refs:
            raise FederationAuthorityError("subagent task is outside its assignment")
        if FederationLifecycleState(instance.state) not in {
            FederationLifecycleState.DECLARED,
            FederationLifecycleState.ADMITTED,
        }:
            raise FederationAuthorityError(
                "new subagent must be declared or admitted before registration"
            )
        key = idempotency_key or f"register:{instance.cid}"
        command = self._command(
            command_id=f"command:subagent-register:{instance.cid}",
            idempotency_key=key,
            parameters={
                "operation": "subagent.register",
                "tenant_id": instance.binding.tenant_id,
                "subagent_id": instance.record_id,
                "federation_id": instance.federation_id,
            },
        )
        now = utc_now()

        def apply(_txn: StateTransaction, _command: StateCommand, live: Any) -> Mapping[str, Any]:
            federation_rows = self.__client.execute(
                "casf_select_federation",
                {
                    "federation_id": instance.federation_id,
                    "tenant_id": instance.binding.tenant_id,
                },
            )
            if not federation_rows:
                raise FederationRepositoryNotFound("parent federation is absent")
            supervisor_rows = self.__client.execute(
                "casf_select_supervisor",
                {
                    "supervisor_id": instance.supervisor_id,
                    "tenant_id": instance.binding.tenant_id,
                    "federation_id": instance.federation_id,
                },
            )
            if not supervisor_rows:
                raise FederationRepositoryNotFound("parent supervisor is absent")
            supervisor_counts = self.__client.execute(
                "casf_count_supervisors",
                {
                    "tenant_id": instance.binding.tenant_id,
                    "federation_id": instance.federation_id,
                },
            )
            supervisor_population = int(
                supervisor_counts[0]["population"] if supervisor_counts else 0
            )
            self._assert_federation_binding(
                federation_rows[0],
                instance.binding,
                supervisor_population=supervisor_population,
            )
            request = FederationRequest.from_dict(
                _decode(federation_rows[0]["body_json"])
            )
            policy = self._load_admitted_policy(
                binding=instance.binding,
                federation_id=instance.federation_id,
            )
            admitted_capabilities = self._validate_definition_authority(
                definition,
                capability_values,
                binding=instance.binding,
                policy=policy,
                resource_budget_ref=request.resource_budget.record_id,
                token_budget_ref=request.token_budget.record_id,
            )
            (
                parent_authority,
                parent_definition,
                parent_assignment,
            ) = self._load_supervisor_admission_authority(
                supervisor_id=instance.supervisor_id,
                tenant_id=instance.binding.tenant_id,
                federation_id=instance.federation_id,
            )
            self._assert_assignment_within_parent(assignment, parent_assignment)
            self._assert_definition_within_parent(definition, parent_definition)
            if (
                instance.fencing_epoch != int(supervisor_rows[0]["fencing_epoch"])
                or instance.fencing_epoch != int(parent_authority["fencing_epoch"])
                or instance.fencing_epoch != int(federation_rows[0]["fencing_epoch"])
                or instance.fencing_epoch != live.fence_epoch
            ):
                raise FederationRepositoryConflict("subagent registration fence is stale")
            if str(supervisor_rows[0]["lifecycle_state"]) in {
                FederationLifecycleState.DECLARED.value,
                FederationLifecycleState.DRAINING.value,
                FederationLifecycleState.QUARANTINED.value,
                FederationLifecycleState.COMPLETED.value,
                FederationLifecycleState.FAILED.value,
                FederationLifecycleState.STOPPED.value,
            }:
                raise FederationAuthorityError(
                    "parent supervisor is not admitted for subagent registration"
                )
            counts = self.__client.execute(
                "casf_count_subagents",
                {
                    "tenant_id": instance.binding.tenant_id,
                    "federation_id": instance.federation_id,
                },
            )
            population = int(counts[0]["population"] if counts else 0)
            if population >= int(federation_rows[0]["maximum_subagents"]):
                raise FederationAuthorityError("logical subagent population ceiling reached")
            admission_decision_id = self._record_policy_admission(
                binding=instance.binding,
                federation_id=instance.federation_id,
                subject_kind="subagent",
                subject_ref=instance.record_id,
                operation="subagent.register",
                definition_cid=definition.cid,
                assignment_cid=assignment.cid,
                capability_cids=tuple(item.cid for item in admitted_capabilities),
                decided_at=now,
            )
            self.__client.execute(
                "casf_insert_subagent_definition",
                {
                    "subagent_definition_id": definition.record_id,
                    "tenant_id": instance.binding.tenant_id,
                    "federation_id": instance.federation_id,
                    "capability_set_ref": content_identity(
                        [item.record_id for item in admitted_capabilities]
                    ),
                    "allowed_operations_ref": content_identity(
                        list(definition.allowed_operations)
                    ),
                    "effect_scope_ref": definition.effect_ceiling,
                    "resource_class": definition.name,
                    "policy_id": instance.binding.policy_ref,
                    "policy_revision": instance.binding.policy_revision,
                    "authorization_evidence_ref": (
                        instance.binding.authorization_evidence_ref
                    ),
                    "content_ref": definition.cid,
                    "created_at": now,
                    "body_json": _json(definition.to_dict()),
                },
            )
            self.__client.execute(
                "casf_insert_subagent_assignment",
                {
                    "subagent_assignment_id": assignment.record_id,
                    "tenant_id": instance.binding.tenant_id,
                    "federation_id": instance.federation_id,
                    "supervisor_id": instance.supervisor_id,
                    "subagent_id": instance.record_id,
                    "task_cid": instance.task_id,
                    "assignment_revision": assignment.revision,
                    "lease_id": instance.lease_id,
                    "fencing_epoch": instance.fencing_epoch,
                    "resource_reservation_id": definition.resource_budget_ref,
                    "token_reservation_id": definition.token_budget_ref,
                    "status": "admitted",
                    "revision": assignment.revision,
                    "admission_decision_id": admission_decision_id,
                    "content_ref": assignment.cid,
                    "assigned_at": now,
                    "updated_at": now,
                    "body_json": _json(assignment.to_dict()),
                },
            )
            for capability in admitted_capabilities:
                self.__client.execute(
                    "casf_insert_subagent_capability",
                    {
                        "subagent_capability_id": capability.record_id,
                        "tenant_id": instance.binding.tenant_id,
                        "federation_id": instance.federation_id,
                        "subagent_id": instance.record_id,
                        "capability_kind": capability.name,
                        "capability_revision": capability.revision,
                        "evidence_ref": capability.cid,
                        "policy_id": instance.binding.policy_ref,
                        "policy_revision": instance.binding.policy_revision,
                        "admission_decision_id": admission_decision_id,
                        "content_ref": capability.cid,
                        "expires_at": instance.binding.expires_at,
                        "recorded_at": now,
                        "body_json": _json(capability.to_dict()),
                    },
                )
            self.__client.execute(
                "casf_insert_subagent",
                {
                    "subagent_id": instance.record_id,
                    "tenant_id": instance.binding.tenant_id,
                    "federation_id": instance.federation_id,
                    "supervisor_id": instance.supervisor_id,
                    "subagent_definition_id": definition.record_id,
                    "task_id": instance.task_id,
                    "lease_id": instance.lease_id,
                    "logical_state": instance.state,
                    "admitted_concurrency_slot": False,
                    "worker_process_birth_id": "",
                    "provider_route_id": "",
                    "admission_decision_id": admission_decision_id,
                    "revision": instance.revision,
                    "fencing_epoch": instance.fencing_epoch,
                    "registered_at": now,
                    "updated_at": now,
                    "body_json": _json(instance.to_dict()),
                },
            )
            draft = EventDraft(
                event_type=EventClass.CAPABILITY_CHANGED,
                stream_id=instance.federation_id,
                causal_parent_ids=(),
                correlation_id=f"correlation:{key}",
                causation_id=f"causation:{instance.cid}",
                tenant_id=instance.binding.tenant_id,
                federation_id=instance.federation_id,
                supervisor_id=instance.supervisor_id,
                task_id=instance.task_id,
                repository_id=instance.binding.repository_ids[0],
                tree_id=instance.binding.repository_tree_ids[0],
                payload_ref=instance.cid,
                changed_fact_refs=(instance.record_id,),
                effect_class=EventEffectClass.AUTHORITATIVE_STATE,
                deduplication_key=f"subagent-register:{key}",
            )
            event, outbox = self._allocate_event(draft, recorded_at=now)
            self._insert_event_outbox(event, outbox, binding=instance.binding)
            return {
                "instance": instance.to_dict(),
                "event_global_sequence": event.global_sequence,
            }

        result = self._submit(command, apply)
        return SubagentInstance.from_dict(result.result["instance"])  # type: ignore[return-value]

    def get_subagent(
        self,
        subagent_id: str,
        *,
        tenant_id: str,
        federation_id: str,
        supervisor_id: str,
    ) -> SubagentInstance | None:
        rows = self.__client.execute(
            "casf_select_subagent",
            {
                "subagent_id": _identifier(subagent_id, "subagent_id"),
                "tenant_id": _identifier(tenant_id, "tenant_id"),
                "federation_id": _identifier(federation_id, "federation_id"),
                "supervisor_id": _identifier(supervisor_id, "supervisor_id"),
            },
        )
        if not rows:
            return None
        return SubagentInstance.from_dict(_decode(rows[0]["body_json"]))  # type: ignore[return-value]

    def reserve_subagent_slot(
        self,
        *,
        subagent_id: str,
        tenant_id: str,
        federation_id: str,
        supervisor_id: str,
        expected_revision: int,
        expected_fencing_epoch: int,
        idempotency_key: str,
    ) -> Mapping[str, Any]:
        """Atomically admit one logical agent into a federation-wide slot."""

        subagent_id = _identifier(subagent_id, "subagent_id")
        tenant_id = _identifier(tenant_id, "tenant_id")
        federation_id = _identifier(federation_id, "federation_id")
        supervisor_id = _identifier(supervisor_id, "supervisor_id")
        expected_revision = _integer(expected_revision, "expected_revision", minimum=1)
        expected_fencing_epoch = _integer(
            expected_fencing_epoch, "expected_fencing_epoch", minimum=1
        )
        command = self._command(
            command_id=f"command:subagent-slot-reserve:{subagent_id}:{expected_revision}",
            idempotency_key=_identifier(idempotency_key, "idempotency_key"),
            command_kind=CommandKind.CLAIM,
            parameters={
                "operation": "subagent.slot.reserve",
                "tenant_id": tenant_id,
                "federation_id": federation_id,
                "supervisor_id": supervisor_id,
                "subagent_id": subagent_id,
                "expected_revision": expected_revision,
                "expected_fencing_epoch": expected_fencing_epoch,
            },
        )
        now = utc_now()

        def apply(
            _txn: StateTransaction,
            _command: StateCommand,
            live: Any,
        ) -> Mapping[str, Any]:
            if live.fence_epoch != expected_fencing_epoch:
                raise FederationRepositoryConflict("subagent slot fence is stale")
            _, instance, _, _ = self._load_subagent_execution_authority(
                subagent_id=subagent_id,
                tenant_id=tenant_id,
                federation_id=federation_id,
                supervisor_id=supervisor_id,
            )
            if instance.revision != expected_revision:
                raise FederationRepositoryConflict("subagent revision is stale")
            if instance.fencing_epoch != expected_fencing_epoch:
                raise FederationRepositoryConflict("subagent fence is stale")
            if instance.state != FederationLifecycleState.ADMITTED.value:
                raise FederationAuthorityError("subagent is not admitted for execution")
            task_lease_rows = self.__client.execute(
                "casf_select_subagent_task_lease_authority",
                {
                    "now_epoch_ms": int(time.time() * 1000),
                    "subagent_id": subagent_id,
                    "tenant_id": tenant_id,
                    "federation_id": federation_id,
                    "supervisor_id": supervisor_id,
                },
            )
            if not task_lease_rows:
                raise FederationAuthorityError(
                    "subagent execution lacks a current task lease and fence"
                )
            worker_process_birth_id, _, _ = self._attest_owner_process(
                tenant_id=tenant_id,
                federation_id=federation_id,
                supervisor_id=supervisor_id,
                subagent_id=subagent_id,
                recorded_at=now,
            )
            slots = self.__client.execute(
                "casf_select_available_subagent_slot",
                {
                    "tenant_id": tenant_id,
                    "federation_id": federation_id,
                    "fencing_epoch": expected_fencing_epoch,
                },
            )
            if not slots:
                raise FederationAuthorityError("federation concurrent subagent ceiling reached")
            slot_number = int(slots[0]["slot_number"])
            slot_revision = int(slots[0]["revision"])
            reserved = self.__client.execute(
                "casf_reserve_subagent_slot",
                {
                    "subagent_id": subagent_id,
                    "supervisor_id": supervisor_id,
                    "worker_process_birth_id": worker_process_birth_id,
                    "lease_id": instance.lease_id,
                    "reserved_at": now,
                    "tenant_id": tenant_id,
                    "federation_id": federation_id,
                    "slot_number": slot_number,
                    "expected_slot_revision": slot_revision,
                    "fencing_epoch": expected_fencing_epoch,
                    "scope_tenant_id": tenant_id,
                    "scope_federation_id": federation_id,
                    "unique_subagent_id": subagent_id,
                },
            )
            if not reserved:
                raise FederationRepositoryConflict("subagent slot CAS conflicted")
            activated = replace(
                instance,
                revision=expected_revision + 1,
                state=FederationLifecycleState.ACTIVE.value,
            )
            updated = self.__client.execute(
                "casf_activate_subagent",
                {
                    "slot_number": slot_number,
                    "worker_process_birth_id": worker_process_birth_id,
                    "updated_at": now,
                    "body_json": _json(activated.to_dict()),
                    "subagent_id": subagent_id,
                    "tenant_id": tenant_id,
                    "federation_id": federation_id,
                    "supervisor_id": supervisor_id,
                    "expected_revision": expected_revision,
                    "fencing_epoch": expected_fencing_epoch,
                },
            )
            if not updated or int(updated[0]["revision"]) != activated.revision:
                raise FederationRepositoryConflict("subagent activation CAS conflicted")
            draft = EventDraft(
                event_type=EventClass.CAPABILITY_CHANGED,
                stream_id=federation_id,
                causal_parent_ids=(),
                correlation_id=f"correlation:{idempotency_key}",
                causation_id=f"causation:{subagent_id}:slot:{slot_number}",
                tenant_id=tenant_id,
                federation_id=federation_id,
                supervisor_id=supervisor_id,
                task_id=instance.task_id,
                repository_id=instance.binding.repository_ids[0],
                tree_id=instance.binding.repository_tree_ids[0],
                payload_ref=f"subagent-slot:{slot_number}",
                changed_fact_refs=(subagent_id, f"slot:{slot_number}"),
                effect_class=EventEffectClass.AUTHORITATIVE_STATE,
                deduplication_key=f"subagent-slot-reserve:{idempotency_key}",
            )
            event, outbox = self._allocate_event(draft, recorded_at=now)
            self._insert_event_outbox(event, outbox, binding=instance.binding)
            resulting_slot_revision = int(reserved[0]["revision"])
            ledger_id = f"slot-ledger:{content_identity({'operation': 'reserve', 'event_id': event.event_id, 'subagent_id': subagent_id})}"
            self.__client.execute(
                "casf_insert_subagent_slot_ledger",
                {
                    "slot_ledger_id": ledger_id,
                    "tenant_id": tenant_id,
                    "federation_id": federation_id,
                    "slot_number": slot_number,
                    "subagent_id": subagent_id,
                    "supervisor_id": supervisor_id,
                    "operation": "reserve",
                    "prior_revision": slot_revision,
                    "resulting_revision": resulting_slot_revision,
                    "fencing_epoch": expected_fencing_epoch,
                    "event_id": event.event_id,
                    "recorded_at": now,
                    "body_json": _json(
                        {
                            "worker_process_birth_id": worker_process_birth_id,
                            "lease_id": instance.lease_id,
                        }
                    ),
                },
            )
            return {
                "subagent": activated.to_dict(),
                "slot_number": slot_number,
                "slot_revision": resulting_slot_revision,
                "event_global_sequence": event.global_sequence,
            }

        return MappingProxyType(dict(self._submit(command, apply).result))

    def release_subagent_slot(
        self,
        *,
        subagent_id: str,
        tenant_id: str,
        federation_id: str,
        supervisor_id: str,
        expected_revision: int,
        expected_fencing_epoch: int,
        idempotency_key: str,
    ) -> Mapping[str, Any]:
        """Release one active slot with agent and ledger updates atomically."""

        subagent_id = _identifier(subagent_id, "subagent_id")
        tenant_id = _identifier(tenant_id, "tenant_id")
        federation_id = _identifier(federation_id, "federation_id")
        supervisor_id = _identifier(supervisor_id, "supervisor_id")
        expected_revision = _integer(expected_revision, "expected_revision", minimum=1)
        expected_fencing_epoch = _integer(
            expected_fencing_epoch, "expected_fencing_epoch", minimum=1
        )
        command = self._command(
            command_id=f"command:subagent-slot-release:{subagent_id}:{expected_revision}",
            idempotency_key=_identifier(idempotency_key, "idempotency_key"),
            command_kind=CommandKind.RELEASE,
            parameters={
                "operation": "subagent.slot.release",
                "tenant_id": tenant_id,
                "federation_id": federation_id,
                "supervisor_id": supervisor_id,
                "subagent_id": subagent_id,
                "expected_revision": expected_revision,
                "expected_fencing_epoch": expected_fencing_epoch,
            },
        )
        now = utc_now()

        def apply(
            _txn: StateTransaction,
            _command: StateCommand,
            live: Any,
        ) -> Mapping[str, Any]:
            if live.fence_epoch != expected_fencing_epoch:
                raise FederationRepositoryConflict("subagent slot fence is stale")
            rows = self.__client.execute(
                "casf_select_active_subagent_slot",
                {
                    "tenant_id": tenant_id,
                    "federation_id": federation_id,
                    "subagent_id": subagent_id,
                },
            )
            if not rows:
                raise FederationRepositoryNotFound("active subagent slot is absent")
            row = rows[0]
            if str(row["supervisor_id"]) != supervisor_id:
                raise FederationAuthorityError("subagent slot supervisor differs")
            if int(row["agent_revision"]) != expected_revision:
                raise FederationRepositoryConflict("subagent revision is stale")
            if int(row["fencing_epoch"]) != expected_fencing_epoch:
                raise FederationRepositoryConflict("subagent fence is stale")
            instance = SubagentInstance.from_dict(_decode(row["body_json"]))
            if instance.state != FederationLifecycleState.ACTIVE.value:
                raise FederationRepositoryConflict("subagent is not active")
            slot_number = int(row["slot_number"])
            slot_revision = int(row["slot_revision"])
            released = self.__client.execute(
                "casf_release_subagent_slot",
                {
                    "released_at": now,
                    "tenant_id": tenant_id,
                    "federation_id": federation_id,
                    "slot_number": slot_number,
                    "subagent_id": subagent_id,
                    "expected_slot_revision": slot_revision,
                    "fencing_epoch": expected_fencing_epoch,
                },
            )
            if not released:
                raise FederationRepositoryConflict("subagent slot release CAS conflicted")
            admitted = replace(
                instance,
                revision=expected_revision + 1,
                state=FederationLifecycleState.ADMITTED.value,
            )
            updated = self.__client.execute(
                "casf_deactivate_subagent",
                {
                    "updated_at": now,
                    "body_json": _json(admitted.to_dict()),
                    "subagent_id": subagent_id,
                    "tenant_id": tenant_id,
                    "federation_id": federation_id,
                    "supervisor_id": supervisor_id,
                    "expected_revision": expected_revision,
                    "fencing_epoch": expected_fencing_epoch,
                    "slot_number": slot_number,
                },
            )
            if not updated or int(updated[0]["revision"]) != admitted.revision:
                raise FederationRepositoryConflict("subagent deactivation CAS conflicted")
            draft = EventDraft(
                event_type=EventClass.CAPABILITY_CHANGED,
                stream_id=federation_id,
                causal_parent_ids=(),
                correlation_id=f"correlation:{idempotency_key}",
                causation_id=f"causation:{subagent_id}:slot-release:{slot_number}",
                tenant_id=tenant_id,
                federation_id=federation_id,
                supervisor_id=supervisor_id,
                task_id=instance.task_id,
                repository_id=instance.binding.repository_ids[0],
                tree_id=instance.binding.repository_tree_ids[0],
                payload_ref=f"subagent-slot:{slot_number}:released",
                changed_fact_refs=(subagent_id, f"slot:{slot_number}"),
                effect_class=EventEffectClass.AUTHORITATIVE_STATE,
                deduplication_key=f"subagent-slot-release:{idempotency_key}",
            )
            event, outbox = self._allocate_event(draft, recorded_at=now)
            self._insert_event_outbox(event, outbox, binding=instance.binding)
            resulting_slot_revision = int(released[0]["revision"])
            ledger_id = f"slot-ledger:{content_identity({'operation': 'release', 'event_id': event.event_id, 'subagent_id': subagent_id})}"
            self.__client.execute(
                "casf_insert_subagent_slot_ledger",
                {
                    "slot_ledger_id": ledger_id,
                    "tenant_id": tenant_id,
                    "federation_id": federation_id,
                    "slot_number": slot_number,
                    "subagent_id": subagent_id,
                    "supervisor_id": supervisor_id,
                    "operation": "release",
                    "prior_revision": slot_revision,
                    "resulting_revision": resulting_slot_revision,
                    "fencing_epoch": expected_fencing_epoch,
                    "event_id": event.event_id,
                    "recorded_at": now,
                    "body_json": _json(
                        {"worker_process_birth_id": row["worker_process_birth_id"]}
                    ),
                },
            )
            return {
                "subagent": admitted.to_dict(),
                "slot_number": slot_number,
                "slot_revision": resulting_slot_revision,
                "event_global_sequence": event.global_sequence,
            }

        return MappingProxyType(dict(self._submit(command, apply).result))

    def record_subagent_outcome(self, outcome: SubagentOutcome) -> None:
        # Outcome recording is itself idempotent through its content identity.
        command = self._command(
            command_id=f"command:subagent-outcome:{outcome.cid}",
            idempotency_key=f"outcome:{outcome.cid}",
            parameters={
                "operation": "subagent.outcome",
                "tenant_id": outcome.binding.tenant_id,
                "federation_id": outcome.federation_id,
                "supervisor_id": outcome.supervisor_id,
                "subagent_id": outcome.subagent_id,
                "outcome_id": outcome.record_id,
            },
        )

        def apply(_txn: StateTransaction, _command: StateCommand, _live: Any) -> Mapping[str, Any]:
            instance_rows = self.__client.execute(
                "casf_select_subagent_admission_authority",
                {
                    "subagent_id": outcome.subagent_id,
                    "tenant_id": outcome.binding.tenant_id,
                    "federation_id": outcome.federation_id,
                    "supervisor_id": outcome.supervisor_id,
                },
            )
            if not instance_rows:
                raise FederationRepositoryNotFound("outcome subagent is absent")
            authority = instance_rows[0]
            instance = SubagentInstance.from_dict(_decode(authority["body_json"]))
            definition = SubagentDefinition.from_dict(
                _decode(authority["definition_json"])
            )
            assignment = SubagentAssignment.from_dict(
                _decode(authority["assignment_json"])
            )
            if (
                instance.binding != outcome.binding
                or instance.federation_id != outcome.federation_id
                or instance.supervisor_id != outcome.supervisor_id
                or instance.task_id != outcome.task_id
                or instance.fencing_epoch != outcome.fencing_epoch
            ):
                raise FederationAuthorityError(
                    "subagent outcome scope or fencing epoch differs from its live identity"
                )
            if (
                instance.state != FederationLifecycleState.ACTIVE.value
                or str(authority["logical_state"])
                != FederationLifecycleState.ACTIVE.value
                or int(authority["admitted_concurrency_slot"]) < 1
                or not str(authority["worker_process_birth_id"])
                or not str(authority["admission_decision_id"])
                or str(authority["assignment_status"]) != "admitted"
                or int(authority["assignment_fencing_epoch"])
                != instance.fencing_epoch
                or assignment.fencing_epoch != instance.fencing_epoch
                or assignment.revision != int(authority["assignment_revision"])
                or str(authority["assignment_lease_id"]) != instance.lease_id
                or str(authority["assignment_decision_id"])
                != str(authority["admission_decision_id"])
                or outcome.task_id not in assignment.task_refs
            ):
                raise FederationAuthorityError(
                    "subagent outcome lacks a current active admission and assignment"
                )
            policy = self._load_admitted_policy(
                binding=outcome.binding,
                federation_id=outcome.federation_id,
            )
            if (
                definition.binding != outcome.binding
                or assignment.binding != outcome.binding
                or str(authority["definition_policy_id"])
                != outcome.binding.policy_ref
                or int(authority["definition_policy_revision"])
                != outcome.binding.policy_revision
                or str(authority["authorization_evidence_ref"])
                != outcome.binding.authorization_evidence_ref
                or not set(definition.allowed_operations).issubset(
                    policy.allowed_operations
                )
                or definition.effect_ceiling not in policy.allowed_effects
                or str(authority["resource_reservation_id"])
                != definition.resource_budget_ref
                or str(authority["token_reservation_id"])
                != definition.token_budget_ref
            ):
                raise FederationAuthorityError(
                    "subagent outcome definition, policy, or budget authority is stale"
                )
            capability_rows = self.__client.execute(
                "casf_select_subagent_capability_authority",
                {
                    "subagent_id": outcome.subagent_id,
                    "tenant_id": outcome.binding.tenant_id,
                    "federation_id": outcome.federation_id,
                },
            )
            capabilities = tuple(
                SubagentCapability.from_dict(_decode(row["body_json"]))
                for row in capability_rows
            )
            if (
                not capabilities
                or len(capability_rows) != len(definition.capabilities)
                or {item.record_id for item in capabilities}
                != set(definition.capabilities)
                or any(
                    item.binding != outcome.binding
                    or not set(item.allowed_operations).issubset(
                        definition.allowed_operations
                    )
                    or item.effect_ceiling != definition.effect_ceiling
                    or item.risk_ceiling != definition.risk_ceiling
                    or item.resource_budget_ref != definition.resource_budget_ref
                    or item.token_budget_ref != definition.token_budget_ref
                    or str(row["policy_id"]) != outcome.binding.policy_ref
                    or int(row["policy_revision"])
                    != outcome.binding.policy_revision
                    or str(row["admission_decision_id"])
                    != str(authority["admission_decision_id"])
                    or str(row["freshness_state"]) != "current"
                    or not row["expires_at"]
                    or _expired(str(row["expires_at"]))
                    for item, row in zip(capabilities, capability_rows)  # noqa: B905
                )
            ):
                raise FederationAuthorityError(
                    "subagent outcome capability authority is absent or stale"
                )
            execution_rows = self.__client.execute(
                "casf_select_subagent_outcome_execution",
                {
                    "now_epoch_ms": int(time.time() * 1000),
                    "subagent_id": outcome.subagent_id,
                    "tenant_id": outcome.binding.tenant_id,
                    "federation_id": outcome.federation_id,
                    "supervisor_id": outcome.supervisor_id,
                    "task_id": outcome.task_id,
                },
            )
            if not execution_rows:
                raise FederationAuthorityError(
                    "subagent outcome lacks a current task attempt, lease, fence, and slot"
                )
            if len(execution_rows) != 1:
                raise FederationRepositoryConflict(
                    "subagent outcome execution authority is ambiguous"
                )
            execution = execution_rows[0]
            self.__client.execute(
                "casf_insert_subagent_outcome",
                {
                    "outcome_id": outcome.record_id,
                    "tenant_id": outcome.binding.tenant_id,
                    "federation_id": outcome.federation_id,
                    "supervisor_id": outcome.supervisor_id,
                    "subagent_id": outcome.subagent_id,
                    "task_id": outcome.task_id,
                    "attempt_id": execution["attempt_id"],
                    "status": outcome.outcome,
                    "evidence_ref": outcome.evidence_refs[0],
                    "fencing_epoch": outcome.fencing_epoch,
                    "recorded_at": outcome.recorded_at,
                    "body_json": _json(outcome.to_dict()),
                },
            )
            fact_refs = tuple(
                item
                for item in (
                    outcome.record_id,
                    outcome.subagent_id,
                    outcome.task_id,
                    str(execution["attempt_id"]),
                )
                if item
            )
            draft = EventDraft(
                event_type=(
                    EventClass.TASK_FAILED
                    if outcome.outcome == "failed"
                    # Success is an observed worker disposition, not the
                    # authoritative validation/completion decision.  Both
                    # success and cancellation release work for the scheduler.
                    else EventClass.TASK_RELEASED
                ),
                stream_id=outcome.federation_id,
                causal_parent_ids=(),
                correlation_id=f"correlation:{outcome.record_id}",
                causation_id=f"causation:{outcome.subagent_id}:{outcome.fencing_epoch}",
                tenant_id=outcome.binding.tenant_id,
                federation_id=outcome.federation_id,
                supervisor_id=outcome.supervisor_id,
                task_id=outcome.task_id,
                repository_id=outcome.binding.repository_ids[0],
                tree_id=outcome.binding.repository_tree_ids[0],
                payload_ref=outcome.cid,
                changed_fact_refs=fact_refs,
                effect_class=EventEffectClass.AUTHORITATIVE_STATE,
                deduplication_key=f"subagent-outcome:{outcome.cid}",
            )
            event, outbox = self._allocate_event(draft, recorded_at=outcome.recorded_at)
            self._insert_event_outbox(event, outbox, binding=outcome.binding)
            return {
                "outcome_id": outcome.record_id,
                "event_global_sequence": event.global_sequence,
            }

        self._submit(command, apply)

    def register_subscription(
        self,
        subscription: EventSubscription,
        *,
        supervisor_id: str = "",
        maximum_fanout: int = 256,
        idempotency_key: str,
    ) -> EventSubscription:
        if any(
            selector.kind
            in {SelectorKind.CAUSAL_ANCESTOR, SelectorKind.CAUSAL_DESCENDANT}
            for selector in subscription.selectors
        ):
            raise FederationAuthorityError(
                "causal subscription selectors are unavailable until an exact evaluator is bound"
            )
        command = self._command(
            command_id=f"command:subscription-register:{subscription.subscription_id}",
            idempotency_key=idempotency_key,
            parameters={
                "operation": "subscription.register",
                "subscription_id": subscription.subscription_id,
                "tenant_id": subscription.tenant_id,
                "federation_id": subscription.federation_id,
            },
        )
        now = utc_now()

        def apply(_txn: StateTransaction, _command: StateCommand, live: Any) -> Mapping[str, Any]:
            _integer(maximum_fanout, "maximum_fanout", minimum=1, maximum=4096)
            federation_rows = self.__client.execute(
                "casf_select_federation",
                {
                    "federation_id": subscription.federation_id,
                    "tenant_id": subscription.tenant_id,
                },
            )
            if not federation_rows:
                raise FederationRepositoryNotFound("subscription parent federation is absent")
            request = FederationRequest.from_dict(_decode(federation_rows[0]["body_json"]))
            if supervisor_id:
                supervisor_rows = self.__client.execute(
                    "casf_select_supervisor",
                    {
                        "supervisor_id": supervisor_id,
                        "tenant_id": subscription.tenant_id,
                        "federation_id": subscription.federation_id,
                    },
                )
                if not supervisor_rows:
                    raise FederationRepositoryNotFound("subscription supervisor is absent")
            self.__client.execute(
                "casf_insert_subscription",
                {
                    "subscription_id": subscription.subscription_id,
                    "tenant_id": subscription.tenant_id,
                    "federation_id": subscription.federation_id,
                    "consumer_id": subscription.consumer_id,
                    "supervisor_id": supervisor_id,
                    "revision": subscription.revision,
                    "event_classes_json": _json(
                        [item.value for item in subscription.event_classes]
                    ),
                    "maximum_batch": subscription.maximum_batch,
                    "maximum_pending": subscription.maximum_pending,
                    "maximum_fanout": int(maximum_fanout),
                    "retry_budget": subscription.retry_budget,
                    "expires_at": subscription.expires_at,
                    "status": subscription.state.value,
                    "created_at": now,
                    "updated_at": now,
                    "body_json": _json(subscription.to_dict()),
                },
            )
            for ordinal, selector in enumerate(subscription.selectors, start=1):
                self.__client.execute(
                    "casf_insert_subscription_selector",
                    {
                        "selector_id": content_identity(
                            {
                                "subscription_id": subscription.subscription_id,
                                "revision": subscription.revision,
                                "ordinal": ordinal,
                                "selector": selector.to_dict(),
                            }
                        ),
                        "subscription_id": subscription.subscription_id,
                        "subscription_revision": subscription.revision,
                        "selector_kind": selector.kind.value,
                        "selector_value": selector.value,
                        "ordinal": ordinal,
                    },
                )
            cursor = ConsumerCursor(
                consumer_id=subscription.consumer_id,
                subscription_id=subscription.subscription_id,
                subscription_revision=subscription.revision,
                global_sequence=0,
                store_generation=live.generation,
                revision=1,
                updated_at=now,
            )
            self.__client.execute(
                "casf_insert_consumer_cursor",
                {
                    "consumer_id": cursor.consumer_id,
                    "subscription_id": cursor.subscription_id,
                    "subscription_revision": cursor.subscription_revision,
                    "tenant_id": subscription.tenant_id,
                    "federation_id": subscription.federation_id,
                    "store_generation": cursor.store_generation,
                    "fencing_epoch": live.fence_epoch,
                    "updated_at": cursor.updated_at,
                    "body_json": _json(cursor.to_dict()),
                },
            )
            draft = EventDraft(
                event_type=EventClass.CAPABILITY_CHANGED,
                stream_id=subscription.federation_id,
                causal_parent_ids=(),
                correlation_id=f"correlation:{idempotency_key}",
                causation_id=f"causation:{subscription.cid}",
                tenant_id=subscription.tenant_id,
                federation_id=subscription.federation_id,
                supervisor_id=supervisor_id,
                repository_id=request.binding.repository_ids[0],
                tree_id=request.binding.repository_tree_ids[0],
                payload_ref=subscription.cid,
                changed_fact_refs=(
                    subscription.subscription_id,
                    subscription.consumer_id,
                ),
                effect_class=EventEffectClass.AUTHORITATIVE_STATE,
                deduplication_key=f"subscription-register:{idempotency_key}",
            )
            event, outbox = self._allocate_event(draft, recorded_at=now)
            self._insert_event_outbox(event, outbox, binding=request.binding)
            return {
                "subscription": subscription.to_dict(),
                "cursor": cursor.to_dict(),
                "event_global_sequence": event.global_sequence,
            }

        result = self._submit(command, apply)
        return EventSubscription.from_dict(result.result["subscription"])  # type: ignore[return-value]

    def load_subscription(
        self,
        *,
        tenant_id: str,
        federation_id: str,
        subscription_id: str,
    ) -> EventSubscription:
        """Return one exact persisted subscription through its full scope."""

        return self._load_subscription(
            subscription_id,
            tenant_id=tenant_id,
            federation_id=federation_id,
        )

    def _load_subscription(
        self,
        subscription_id: str,
        *,
        tenant_id: str,
        federation_id: str,
    ) -> EventSubscription:
        subscription_id = _identifier(subscription_id, "subscription_id")
        tenant_id = _identifier(tenant_id, "tenant_id")
        federation_id = _identifier(federation_id, "federation_id")
        rows = self.__client.execute(
            "casf_select_subscription",
            {
                "subscription_id": subscription_id,
                "tenant_id": tenant_id,
                "federation_id": federation_id,
            },
        )
        if not rows:
            raise FederationRepositoryNotFound("subscription is absent")
        row = rows[0]
        selector_rows = self.__client.execute(
            "casf_select_subscription_selectors",
            {
                "subscription_id": subscription_id,
                "subscription_revision": int(row["revision"]),
                "tenant_id": tenant_id,
                "federation_id": federation_id,
            },
        )
        selectors = tuple(
            EventSelector(
                kind=SelectorKind(str(item["selector_kind"])),
                value=str(item["selector_value"]),
            )
            for item in selector_rows
        )
        return EventSubscription(
            subscription_id=str(row["subscription_id"]),
            tenant_id=str(row["tenant_id"]),
            federation_id=str(row["federation_id"]),
            consumer_id=str(row["consumer_id"]),
            revision=int(row["revision"]),
            event_classes=tuple(EventClass(item) for item in _decode(row["event_classes_json"])),
            selectors=selectors,
            maximum_batch=int(row["maximum_batch"]),
            maximum_pending=int(row["maximum_pending"]),
            retry_budget=int(row["retry_budget"]),
            expires_at=str(row["expires_at"]),
            state=SubscriptionState(str(row["status"])),
        )

    @staticmethod
    def _cursor_from_row(row: Mapping[str, Any]) -> ConsumerCursor:
        return ConsumerCursor(
            consumer_id=str(row["consumer_id"]),
            subscription_id=str(row["subscription_id"]),
            subscription_revision=int(row["subscription_revision"]),
            global_sequence=int(row["global_sequence"]),
            store_generation=int(row["store_generation"]),
            revision=int(row["revision"]),
            updated_at=str(row["updated_at"]),
        )

    @staticmethod
    def _event_from_row(row: Mapping[str, Any]) -> DomainEvent:
        return DomainEvent(
            event_id=str(row["event_id"]),
            event_cid=str(row["event_cid"]),
            event_type=EventClass(str(row["event_type"])),
            stream_id=str(row["stream_id"]),
            stream_sequence=int(row["sequence"]),
            global_sequence=int(row["global_sequence"]),
            causal_parent_ids=tuple(_decode(row["causal_parent_ids_json"])),
            correlation_id=str(row["correlation_id"]),
            causation_id=str(row["causation_id"]),
            tenant_id=str(row["tenant_id"]),
            federation_id=str(row["federation_id"]),
            supervisor_id=str(row["supervisor_id"] or ""),
            task_id=str(row["task_id"] or ""),
            repository_id=str(row["repository_id"] or ""),
            tree_id=str(row["tree_id"] or ""),
            goal_id=str(row["goal_id"] or ""),
            subgoal_id=str(row["subgoal_id"] or ""),
            symbol_id=str(row["symbol_id"] or ""),
            contract_id=str(row["contract_id"] or ""),
            proof_obligation_id=str(row["proof_obligation_id"] or ""),
            resource_class=str(row["resource_class"] or ""),
            payload_ref=str(row["payload_ref"]),
            changed_fact_refs=tuple(_decode(row["changed_fact_refs_json"])),
            effect_class=EventEffectClass(str(row["effect_class"])),
            recorded_at=str(row["recorded_at"]),
            expires_at=str(row["expires_at"] or ""),
            deduplication_key=str(row["deduplication_key"]),
        )

    @staticmethod
    def _queued_delivery_body(item: DurableQueuedDelivery) -> dict[str, Any]:
        delivery = item.delivery
        return {
            "schema": (
                "ipfs_accelerate_py/agent-supervisor/"
                "causal-federation/durable-queued-delivery@1"
            ),
            "delivery_id": delivery.delivery_id,
            "subscription_id": delivery.subscription_id,
            "subscription_revision": delivery.subscription_revision,
            "consumer_id": delivery.consumer_id,
            "decision_id": delivery.decision.decision_id,
            "representative_event_id": (
                delivery.decision.representative_event.event_id
            ),
            "input_event_ids": list(delivery.decision.input_event_ids),
            "changed_fact_refs": list(delivery.decision.changed_fact_refs),
            "coalescing_mode": delivery.decision.mode.value,
            "attempt_number": delivery.attempt_number,
            "coverage_id": item.coverage.coverage_id,
        }

    @staticmethod
    def _durable_delivery_from_rows(
        *,
        queue_body: Mapping[str, Any],
        coverage_body: Mapping[str, Any],
        event_body: Mapping[str, Any],
    ) -> DurableQueuedDelivery:
        event = DomainEvent.from_dict(event_body)
        decision = CoalescingDecision(
            representative_event=event,  # type: ignore[arg-type]
            input_event_ids=tuple(queue_body["input_event_ids"]),
            changed_fact_refs=tuple(queue_body["changed_fact_refs"]),
            mode=CoalescingMode(str(queue_body["coalescing_mode"])),
        )
        delivery = QueuedDelivery(
            delivery_id=str(queue_body["delivery_id"]),
            subscription_id=str(queue_body["subscription_id"]),
            subscription_revision=int(queue_body["subscription_revision"]),
            consumer_id=str(queue_body["consumer_id"]),
            decision=decision,
            attempt_number=int(queue_body["attempt_number"]),
        )
        coverage = CoalescingCoverageRecord(
            coverage_id=str(coverage_body["coverage_id"]),
            decision_id=str(coverage_body["decision_id"]),
            subscription_id=str(coverage_body["subscription_id"]),
            subscription_revision=int(coverage_body["subscription_revision"]),
            representative_event_id=str(coverage_body["representative_event_id"]),
            input_event_ids=tuple(coverage_body["input_event_ids"]),
            changed_fact_refs=tuple(coverage_body["changed_fact_refs"]),
            mode=CoalescingMode(str(coverage_body["mode"])),
        )
        return DurableQueuedDelivery(delivery=delivery, coverage=coverage)

    def _events_for_loaded_subscription(
        self,
        subscription: EventSubscription,
        *,
        after_cursor: int,
        maximum_events: int,
    ) -> tuple[DomainEvent, ...]:
        """Page bounded candidates until matches, exhaustion, or a typed bound.

        Applying selectors after a single SQL ``LIMIT`` can falsely report an
        empty queue when a matching event follows many same-class nonmatches.
        This loop retains the closed query shape, advances a monotonic cursor,
        and raises rather than silently suppressing work if its scan budget is
        exhausted.
        """

        selected = min(maximum_events, subscription.maximum_batch)
        scan_cursor = int(after_cursor)
        scanned = 0
        matches: list[DomainEvent] = []
        event_classes_csv = ",".join(item.value for item in subscription.event_classes)
        while len(matches) < selected:
            remaining = _MAX_EVENT_SCAN_CANDIDATES - scanned
            if remaining <= 0:
                overflow = self.__client.execute(
                    "casf_list_matching_event_window",
                    {
                        "tenant_id": subscription.tenant_id,
                        "federation_id": subscription.federation_id,
                        "after_cursor": scan_cursor,
                        "event_classes_csv": event_classes_csv,
                        "limit": 1,
                    },
                )
                if overflow:
                    raise FederationBoundsError("subscription selector scan bound exhausted")
                break
            page_limit = min(_EVENT_SCAN_PAGE_SIZE, remaining)
            rows = self.__client.execute(
                "casf_list_matching_event_window",
                {
                    "tenant_id": subscription.tenant_id,
                    "federation_id": subscription.federation_id,
                    "after_cursor": scan_cursor,
                    "event_classes_csv": event_classes_csv,
                    "limit": page_limit,
                },
            )
            if not rows:
                break
            scanned += len(rows)
            for row in rows:
                event = self._event_from_row(row)
                if event_matches_subscription(event, subscription):
                    matches.append(event)
                    if len(matches) >= selected:
                        return tuple(matches)
            next_cursor = int(rows[-1]["global_sequence"])
            if next_cursor <= scan_cursor:
                raise FederationRepositoryConflict("event scan cursor did not advance")
            scan_cursor = next_cursor
            if len(rows) < page_limit:
                break
        return tuple(matches)

    def _routed_events_for_loaded_subscription(
        self,
        subscription: EventSubscription,
        *,
        after_cursor: int,
        maximum_events: int,
    ) -> tuple[DomainEvent, ...]:
        """Read only persist-first queue representatives for a waiter."""

        limit = min(
            _integer(maximum_events, "maximum_events", minimum=1, maximum=4096),
            subscription.maximum_batch,
        )
        rows = self.__client.execute(
            "casf_list_routed_wait_events",
            {
                "tenant_id": subscription.tenant_id,
                "federation_id": subscription.federation_id,
                "subscription_id": subscription.subscription_id,
                "subscription_revision": subscription.revision,
                "consumer_id": subscription.consumer_id,
                "after_cursor": _integer(after_cursor, "after_cursor"),
                "limit": limit,
            },
        )
        events = tuple(self._event_from_row(row) for row in rows)
        if len({event.event_id for event in events}) != len(events):
            raise FederationRepositoryConflict("durable wait queue returned duplicates")
        if any(not event_matches_subscription(event, subscription) for event in events):
            raise FederationAuthorityError("durable wait queue crossed subscription scope")
        return events

    def get_cursor(
        self,
        *,
        tenant_id: str,
        federation_id: str,
        consumer_id: str,
        subscription_id: str,
    ) -> ConsumerCursor:
        tenant_id = _identifier(tenant_id, "tenant_id")
        federation_id = _identifier(federation_id, "federation_id")
        consumer_id = _identifier(consumer_id, "consumer_id")
        subscription_id = _identifier(subscription_id, "subscription_id")
        rows = self.__client.execute(
            "casf_select_consumer_cursor",
            {
                "consumer_id": consumer_id,
                "subscription_id": subscription_id,
                "tenant_id": tenant_id,
                "federation_id": federation_id,
            },
        )
        if not rows:
            raise FederationRepositoryNotFound("consumer cursor is absent")
        return self._cursor_from_row(rows[0])

    def _persist_durable_delivery(
        self,
        item: DurableQueuedDelivery,
        *,
        subscription: EventSubscription,
        fencing_epoch: int,
        recorded_at: str,
    ) -> bool:
        """Persist one validated queue+coverage record in the open transaction."""

        delivery = item.delivery
        decision = delivery.decision
        if (
            delivery.subscription_id != subscription.subscription_id
            or delivery.subscription_revision != subscription.revision
            or delivery.consumer_id != subscription.consumer_id
        ):
            raise FederationAuthorityError("durable delivery owner differs")
        expected_delivery_id = (
            "delivery:"
            + content_identity(
                {
                    "subscription_id": subscription.subscription_id,
                    "subscription_revision": subscription.revision,
                    "input_event_ids": list(decision.input_event_ids),
                }
            )
        )
        if delivery.delivery_id != expected_delivery_id:
            raise FederationContractError("durable delivery identity is not canonical")
        expected_coverage_id = (
            "coalescing-coverage:"
            + content_identity(
                {
                    "decision_id": decision.decision_id,
                    "subscription_id": subscription.subscription_id,
                    "subscription_revision": subscription.revision,
                    "input_event_ids": list(decision.input_event_ids),
                }
            )
        )
        if item.coverage.coverage_id != expected_coverage_id:
            raise FederationContractError("coalescing coverage identity is not canonical")

        authoritative_events: dict[str, DomainEvent] = {}
        for event_id in decision.input_event_ids:
            rows = self.__client.execute(
                "casf_select_event_for_routing",
                {
                    "event_id": event_id,
                    "tenant_id": subscription.tenant_id,
                    "federation_id": subscription.federation_id,
                },
            )
            if not rows:
                raise FederationRepositoryNotFound("coalescing input event is absent")
            event = self._event_from_row(rows[0])
            if not event_matches_subscription(event, subscription):
                raise FederationAuthorityError(
                    "coalescing input event does not match subscription"
                )
            authoritative_events[event_id] = event
        representative = authoritative_events.get(decision.representative_event.event_id)
        if representative is None:
            raise FederationAuthorityError("representative event is outside coverage")
        if representative != decision.representative_event:
            raise FederationAuthorityError("representative event differs from authority")
        outbox_rows = self.__client.execute(
            "casf_select_outbox_for_routing",
            {
                "event_id": representative.event_id,
                "tenant_id": subscription.tenant_id,
                "federation_id": subscription.federation_id,
            },
        )
        if not outbox_rows:
            raise FederationRepositoryNotFound("representative event outbox is absent")
        if str(outbox_rows[0]["event_cid"]) != representative.event_cid:
            raise FederationAuthorityError("representative outbox identity differs")

        coverage_body = item.coverage.to_dict()
        coverage_inserted = self.__client.execute(
            "casf_insert_coalescing_coverage",
            {
                "coverage_id": item.coverage.coverage_id,
                "decision_id": item.coverage.decision_id,
                "tenant_id": subscription.tenant_id,
                "federation_id": subscription.federation_id,
                "subscription_id": subscription.subscription_id,
                "subscription_revision": subscription.revision,
                "representative_event_id": representative.event_id,
                "coalescing_mode": item.coverage.mode.value,
                "input_event_count": len(item.coverage.input_event_ids),
                "content_ref": content_identity(coverage_body),
                "created_at": recorded_at,
                "body_json": _json(coverage_body),
            },
        )
        if not coverage_inserted:
            existing_coverage = self.__client.execute(
                "casf_select_coalescing_coverage",
                {"coverage_id": item.coverage.coverage_id},
            )
            if (
                not existing_coverage
                or _decode(existing_coverage[0]["body_json"]) != coverage_body
            ):
                raise FederationRepositoryConflict(
                    "coalescing coverage identity conflicts with existing content"
                )
        for ordinal, event_id in enumerate(item.coverage.input_event_ids, start=1):
            self.__client.execute(
                "casf_insert_coalescing_input",
                {
                    "coverage_id": item.coverage.coverage_id,
                    "event_id": event_id,
                    "ordinal": ordinal,
                },
            )

        queue_body = self._queued_delivery_body(item)
        inserted = self.__client.execute(
            "casf_insert_delivery_queue",
            {
                "delivery_id": delivery.delivery_id,
                "tenant_id": subscription.tenant_id,
                "federation_id": subscription.federation_id,
                "subscription_id": subscription.subscription_id,
                "subscription_revision": subscription.revision,
                "consumer_id": subscription.consumer_id,
                "decision_id": decision.decision_id,
                "representative_event_id": representative.event_id,
                "outbox_id": str(outbox_rows[0]["outbox_id"]),
                "attempt_number": delivery.attempt_number,
                "fencing_epoch": fencing_epoch,
                "available_at": recorded_at,
                "created_at": recorded_at,
                "updated_at": recorded_at,
                "body_json": _json(queue_body),
            },
        )
        if inserted:
            return True
        existing_queue = self.__client.execute(
            "casf_select_delivery_queue",
            {
                "delivery_id": delivery.delivery_id,
                "tenant_id": subscription.tenant_id,
                "federation_id": subscription.federation_id,
                "subscription_id": subscription.subscription_id,
            },
        )
        if not existing_queue:
            raise FederationRepositoryConflict(
                "durable delivery identity conflicts with existing content"
            )
        existing_body = dict(_decode(existing_queue[0]["body_json"]))
        expected_body = dict(queue_body)
        existing_attempt = int(existing_body.pop("attempt_number", 0))
        expected_attempt = int(expected_body.pop("attempt_number", 0))
        if existing_body != expected_body or existing_attempt < expected_attempt:
            raise FederationRepositoryConflict(
                "durable delivery identity conflicts with existing content"
            )
        return False

    def load_routing_state(
        self,
        events: Sequence[DomainEvent],
        subscriptions: Sequence[EventSubscription],
        *,
        maximum_known_deliveries: int,
    ) -> DurableRoutingState:
        """Load the bounded persisted planner state through sealed reads."""

        values = tuple(events)
        consumers = tuple(subscriptions)
        limit = _integer(
            maximum_known_deliveries,
            "maximum_known_deliveries",
            minimum=1,
            maximum=65_536,
        )
        if len(values) > 4_096 or any(not isinstance(item, DomainEvent) for item in values):
            raise FederationBoundsError("routing state event batch is invalid")
        if len(consumers) > 4_096 or any(
            not isinstance(item, EventSubscription) for item in consumers
        ):
            raise FederationBoundsError("routing state subscription batch is invalid")
        if not values:
            return DurableRoutingState(
                known_delivery_ids=(),
                subscriptions=(),
                maximum_fanout_per_event=4_096,
                store_generation=self.store_generation(),
            )
        scopes = {(item.tenant_id, item.federation_id) for item in values}
        if len(scopes) != 1:
            raise FederationAuthorityError("routing state crosses event scope")
        tenant_id, federation_id = next(iter(scopes))
        if any(
            (item.tenant_id, item.federation_id) != (tenant_id, federation_id)
            for item in consumers
        ):
            raise FederationAuthorityError("routing state crosses subscription scope")
        identities = tuple(item.subscription_id for item in consumers)
        if len(set(identities)) != len(identities):
            raise FederationContractError("routing state repeats a subscription")

        first_sequence = min(item.global_sequence for item in values)
        last_sequence = max(item.global_sequence for item in values)
        known: list[str] = []
        states: list[DurableSubscriptionRoutingState] = []
        fanout_limits: list[int] = []
        observed_at = utc_now()
        for subscription in consumers:
            rows = self.__client.execute(
                "casf_select_subscription_routing_state",
                {
                    "tenant_id": tenant_id,
                    "federation_id": federation_id,
                    "subscription_id": subscription.subscription_id,
                    "observed_at": observed_at,
                },
            )
            if len(rows) != 1:
                raise FederationAuthorityError(
                    "routing requires one active canonical subscription"
                )
            row = rows[0]
            if (
                int(row["revision"]) != subscription.revision
                or int(row["maximum_pending"]) != subscription.maximum_pending
            ):
                raise FederationAuthorityError(
                    "routing subscription bounds differ from authority"
                )
            states.append(
                DurableSubscriptionRoutingState(
                    subscription_id=subscription.subscription_id,
                    subscription_revision=subscription.revision,
                    pending_deliveries=int(row["pending_deliveries"]),
                    maximum_pending=subscription.maximum_pending,
                )
            )
            fanout_limits.append(int(row["maximum_fanout"]))
            remaining = limit - len(known)
            if remaining <= 0:
                raise FederationBoundsError("durable routing coverage exceeds its bound")
            coverage_rows = self.__client.execute(
                "casf_list_subscription_durable_route_coverage",
                {
                    "tenant_id": tenant_id,
                    "federation_id": federation_id,
                    "subscription_id": subscription.subscription_id,
                    "first_global_sequence": first_sequence,
                    "last_global_sequence": last_sequence,
                    "limit": remaining + 1,
                },
            )
            if len(coverage_rows) > remaining:
                raise FederationBoundsError("durable routing coverage exceeds its bound")
            known.extend(str(item["delivery_id"]) for item in coverage_rows)
        known_ids = tuple(dict.fromkeys(known))
        return DurableRoutingState(
            known_delivery_ids=known_ids,
            subscriptions=tuple(states),
            maximum_fanout_per_event=min(fanout_limits, default=4_096),
            store_generation=self.store_generation(),
        )

    def persist_routed_batch(
        self,
        batch: DurableRouteBatch,
        *,
        idempotency_key: str,
    ) -> DurableRouteCommit:
        """Atomically persist routing dispositions and complete coverage."""

        if not isinstance(batch, DurableRouteBatch):
            raise FederationContractError("batch must be DurableRouteBatch")
        expected_batch_id = (
            "durable-route:"
            + content_identity(
                {
                    "deliveries": [item.to_dict() for item in batch.deliveries],
                    "maximum_fanout_per_event": batch.maximum_fanout_per_event,
                }
            )
        )
        if batch.batch_id != expected_batch_id:
            raise FederationContractError("durable route batch identity is not canonical")
        expected_key = f"route-idempotency:{content_identity(batch.to_dict())}"
        if idempotency_key != expected_key:
            raise FederationContractError("durable route idempotency identity is not canonical")
        if not batch.deliveries:
            return DurableRouteCommit(
                batch_id=batch.batch_id,
                inserted_delivery_ids=(),
                existing_delivery_ids=(),
                store_generation=self.store_generation(),
            )
        first_event = batch.deliveries[0].delivery.decision.representative_event
        scope = (first_event.tenant_id, first_event.federation_id)
        if any(
            (
                item.delivery.decision.representative_event.tenant_id,
                item.delivery.decision.representative_event.federation_id,
            )
            != scope
            for item in batch.deliveries
        ):
            raise FederationAuthorityError("durable route batch crosses authority scope")
        command = self._command(
            command_id=f"command:event-route:{batch.batch_id}",
            idempotency_key=idempotency_key,
            parameters={
                "operation": "event.route.persist",
                "batch_id": batch.batch_id,
                "tenant_id": scope[0],
                "federation_id": scope[1],
                "delivery_count": len(batch.deliveries),
            },
        )
        recorded_at = utc_now()

        def apply(
            _txn: StateTransaction,
            _command: StateCommand,
            live: Any,
        ) -> Mapping[str, Any]:
            inserted: list[str] = []
            existing: list[str] = []
            subscription_rows: dict[str, Mapping[str, Any]] = {}
            new_by_subscription: dict[str, int] = {}
            new_fanout_by_event: dict[str, int] = {}
            existing_delivery_ids: set[str] = set()
            for item in batch.deliveries:
                delivery = item.delivery
                subscription_id = delivery.subscription_id
                state_row = subscription_rows.get(subscription_id)
                if state_row is None:
                    state_rows = self.__client.execute(
                        "casf_select_subscription_routing_state",
                        {
                            "tenant_id": scope[0],
                            "federation_id": scope[1],
                            "subscription_id": subscription_id,
                            "observed_at": recorded_at,
                        },
                    )
                    if len(state_rows) != 1:
                        raise FederationAuthorityError(
                            "durable route requires one active subscription"
                        )
                    state_row = state_rows[0]
                    subscription_rows[subscription_id] = state_row
                queue_rows = self.__client.execute(
                    "casf_select_delivery_queue",
                    {
                        "delivery_id": delivery.delivery_id,
                        "tenant_id": scope[0],
                        "federation_id": scope[1],
                        "subscription_id": subscription_id,
                    },
                )
                if queue_rows:
                    existing_delivery_ids.add(delivery.delivery_id)
                    continue
                new_by_subscription[subscription_id] = (
                    new_by_subscription.get(subscription_id, 0) + 1
                )
                for event_id in item.coverage.input_event_ids:
                    new_fanout_by_event[event_id] = (
                        new_fanout_by_event.get(event_id, 0) + 1
                    )

            for subscription_id, additional in new_by_subscription.items():
                state_row = subscription_rows[subscription_id]
                projected = int(state_row["pending_deliveries"]) + additional
                if projected > int(state_row["maximum_pending"]):
                    raise DurableRoutingBackpressure(
                        "durable subscription pending ceiling changed before commit"
                    )
            admitted_fanout = min(
                [batch.maximum_fanout_per_event]
                + [int(row["maximum_fanout"]) for row in subscription_rows.values()]
            )
            if any(
                fanout > admitted_fanout for fanout in new_fanout_by_event.values()
            ):
                raise DurableRoutingBackpressure(
                    "durable event fanout ceiling changed before commit"
                )
            for item in batch.deliveries:
                subscription = self._load_subscription(
                    item.delivery.subscription_id,
                    tenant_id=scope[0],
                    federation_id=scope[1],
                )
                if subscription.state is not SubscriptionState.ACTIVE:
                    raise FederationAuthorityError("durable subscription is not active")
                if self._persist_durable_delivery(
                    item,
                    subscription=subscription,
                    fencing_epoch=live.fence_epoch,
                    recorded_at=recorded_at,
                ):
                    inserted.append(item.delivery.delivery_id)
                else:
                    existing.append(item.delivery.delivery_id)
            if set(existing) != existing_delivery_ids:
                raise FederationRepositoryConflict(
                    "durable route preflight changed within one owner transaction"
                )
            return {
                "batch_id": batch.batch_id,
                "inserted_delivery_ids": inserted,
                "existing_delivery_ids": existing,
                "store_generation": live.generation,
                "routed_global_sequence": max(
                    item.delivery.decision.representative_event.global_sequence
                    for item in batch.deliveries
                ),
            }

        result = self._submit(command, apply)
        return DurableRouteCommit(
            batch_id=str(result.result["batch_id"]),
            inserted_delivery_ids=tuple(result.result["inserted_delivery_ids"]),
            existing_delivery_ids=tuple(result.result["existing_delivery_ids"]),
            store_generation=int(result.result["store_generation"]),
        )

    def load_deliverable_deliveries(
        self,
        subscription_id: str,
        subscription_revision: int,
        *,
        tenant_id: str,
        federation_id: str,
        maximum: int,
        expected_fencing_epoch: int,
    ) -> tuple[DurableQueuedDelivery, ...]:
        subscription = self._load_subscription(
            _identifier(subscription_id, "subscription_id"),
            tenant_id=tenant_id,
            federation_id=federation_id,
        )
        revision = _integer(
            subscription_revision, "subscription_revision", minimum=1
        )
        if subscription.revision != revision:
            raise StaleSubscriptionError("deliverable subscription revision is stale")
        if subscription.state is not SubscriptionState.ACTIVE:
            raise FederationAuthorityError("deliverable subscription is not active")
        fence = _integer(
            expected_fencing_epoch, "expected_fencing_epoch", minimum=1
        )
        if self.__client.load_generation().fence_epoch != fence:
            raise FederationRepositoryConflict("delivery queue fence is stale")
        limit = _integer(maximum, "maximum", minimum=1, maximum=4096)
        rows = self.__client.execute(
            "casf_list_deliverable_queue",
            {
                "tenant_id": subscription.tenant_id,
                "federation_id": subscription.federation_id,
                "subscription_id": subscription.subscription_id,
                "consumer_id": subscription.consumer_id,
                "subscription_revision": revision,
                "fencing_epoch": fence,
                "limit": limit,
            },
        )
        result: list[DurableQueuedDelivery] = []
        for row in rows:
            item = self._durable_delivery_from_rows(
                queue_body=_decode(row["body_json"]),
                coverage_body=_decode(row["coverage_json"]),
                event_body=_decode(row["event_json"]),
            )
            if (
                item.delivery.subscription_id != subscription.subscription_id
                or item.delivery.subscription_revision != revision
                or item.delivery.consumer_id != subscription.consumer_id
            ):
                raise FederationAuthorityError("durable queue returned cross-scope data")
            result.append(item)
        return tuple(result)

    def record_delivery_attempt(
        self,
        attempt: DeliveryAttempt,
        *,
        tenant_id: str,
        federation_id: str,
        subscription_revision: int,
        expected_fencing_epoch: int,
        idempotency_key: str,
    ) -> DeliveryAttempt:
        """Persist a fenced delivered exposure through the state owner.

        The caller supplies only closed delivery identity.  Tenant/federation
        scope and the outbox link are resolved from the owned subscription and
        authoritative event; neither can be manufactured by a router client.
        This operation deliberately accepts only ``DELIVERED`` records.  Retry
        and dead-letter transitions use their own bounded router operations.
        """

        if not isinstance(attempt, DeliveryAttempt):
            raise FederationContractError("attempt must be DeliveryAttempt")
        if attempt.state is not DeliveryState.DELIVERED:
            raise FederationContractError("record_delivery_attempt requires delivered state")
        tenant_id = _identifier(tenant_id, "tenant_id")
        federation_id = _identifier(federation_id, "federation_id")
        subscription_revision = _integer(
            subscription_revision,
            "subscription_revision",
            minimum=1,
        )
        expected_fencing_epoch = _integer(
            expected_fencing_epoch,
            "expected_fencing_epoch",
            minimum=1,
        )
        idempotency_key = _identifier(idempotency_key, "idempotency_key")
        invocation_ref = content_identity(
            {
                "attempt_cid": attempt.cid,
                "tenant_id": tenant_id,
                "federation_id": federation_id,
                "subscription_revision": subscription_revision,
                "expected_fencing_epoch": expected_fencing_epoch,
            }
        )
        command = self._command(
            command_id=f"command:delivery-record:{invocation_ref}",
            idempotency_key=idempotency_key,
            command_kind=CommandKind.APPEND,
            parameters={
                "operation": "event.delivery.record",
                "attempt_id": attempt.attempt_id,
                "event_id": attempt.event_id,
                "tenant_id": tenant_id,
                "federation_id": federation_id,
                "subscription_id": attempt.subscription_id,
                "consumer_id": attempt.consumer_id,
                "subscription_revision": subscription_revision,
                "expected_fencing_epoch": expected_fencing_epoch,
            },
        )

        def apply(
            _txn: StateTransaction,
            _command: StateCommand,
            live: Any,
        ) -> Mapping[str, Any]:
            subscription = self._load_subscription(
                attempt.subscription_id,
                tenant_id=tenant_id,
                federation_id=federation_id,
            )
            if subscription.consumer_id != attempt.consumer_id:
                raise FederationAuthorityError("delivery consumer does not own subscription")
            if subscription.revision != subscription_revision:
                raise StaleSubscriptionError("delivery subscription revision is stale")
            if subscription.state is not SubscriptionState.ACTIVE:
                raise FederationAuthorityError("delivery subscription is not active")

            cursor_rows = self.__client.execute(
                "casf_select_consumer_cursor",
                {
                    "consumer_id": attempt.consumer_id,
                    "subscription_id": attempt.subscription_id,
                    "tenant_id": subscription.tenant_id,
                    "federation_id": subscription.federation_id,
                },
            )
            if not cursor_rows:
                raise FederationRepositoryNotFound("consumer cursor is absent")
            cursor_row = cursor_rows[0]
            if int(cursor_row["subscription_revision"]) != subscription_revision:
                raise StaleSubscriptionError("consumer cursor subscription is stale")
            if (
                int(cursor_row["fencing_epoch"]) != expected_fencing_epoch
                or live.fence_epoch != expected_fencing_epoch
            ):
                raise FederationRepositoryConflict("delivery fence is stale")

            event_rows = self.__client.execute(
                "casf_select_event_for_ack",
                {
                    "event_id": attempt.event_id,
                    "tenant_id": subscription.tenant_id,
                    "federation_id": subscription.federation_id,
                    "subscription_id": subscription.subscription_id,
                    "subscription_revision": subscription_revision,
                    "consumer_id": subscription.consumer_id,
                },
            )
            if not event_rows:
                raise FederationRepositoryNotFound("delivered event is absent")
            event = self._event_from_row(event_rows[0])
            if event.global_sequence <= int(cursor_row["global_sequence"]):
                raise FederationRepositoryConflict(
                    "delivery event is not ahead of the durable cursor"
                )
            if not event_matches_subscription(event, subscription):
                raise FederationAuthorityError(
                    "delivered event does not match the owned subscription"
                )

            outbox_rows = self.__client.execute(
                "casf_select_outbox_for_delivery",
                {
                    "event_id": event.event_id,
                    "tenant_id": subscription.tenant_id,
                    "federation_id": subscription.federation_id,
                    "subscription_id": subscription.subscription_id,
                    "subscription_revision": subscription_revision,
                    "consumer_id": subscription.consumer_id,
                },
            )
            if not outbox_rows:
                raise FederationRepositoryNotFound(
                    "delivered event has no authoritative outbox record"
                )
            outbox = outbox_rows[0]
            if str(outbox["event_cid"]) != event.event_cid:
                raise FederationAuthorityError(
                    "delivery outbox content identity differs from event"
                )

            queue_rows = self.__client.execute(
                "casf_select_queue_for_attempt",
                {
                    "tenant_id": subscription.tenant_id,
                    "federation_id": subscription.federation_id,
                    "subscription_id": subscription.subscription_id,
                    "subscription_revision": subscription_revision,
                    "consumer_id": subscription.consumer_id,
                    "event_id": event.event_id,
                    "prior_attempt_number": attempt.attempt_number - 1,
                    "fencing_epoch": expected_fencing_epoch,
                },
            )
            if len(queue_rows) != 1:
                raise FederationRepositoryConflict(
                    "delivery attempt requires one route-first durable queue record"
                )
            queue_row = queue_rows[0]
            delivery_id = str(queue_row["delivery_id"])

            self.__client.execute(
                "casf_insert_delivery_attempt",
                {
                    "attempt_id": attempt.attempt_id,
                    "tenant_id": subscription.tenant_id,
                    "federation_id": subscription.federation_id,
                    "event_id": event.event_id,
                    "outbox_id": str(outbox["outbox_id"]),
                    "delivery_id": delivery_id,
                    "subscription_id": subscription.subscription_id,
                    "subscription_revision": subscription_revision,
                    "consumer_id": subscription.consumer_id,
                    "attempt_number": attempt.attempt_number,
                    "fencing_epoch": expected_fencing_epoch,
                    "status": attempt.state.value,
                    "error_code": attempt.error_code,
                    "recorded_at": attempt.recorded_at,
                    "body_json": _json(attempt.to_dict()),
                },
            )
            queue_record = self.__client.execute(
                "casf_select_delivery_queue",
                {
                    "delivery_id": delivery_id,
                    "tenant_id": subscription.tenant_id,
                    "federation_id": subscription.federation_id,
                    "subscription_id": subscription.subscription_id,
                },
            )[0]
            queue_body = _decode(queue_record["body_json"])
            queue_body["attempt_number"] = attempt.attempt_number
            marked = self.__client.execute(
                "casf_mark_queue_delivered",
                {
                    "attempt_number": attempt.attempt_number,
                    "updated_at": attempt.recorded_at,
                    "body_json": _json(queue_body),
                    "delivery_id": delivery_id,
                    "tenant_id": subscription.tenant_id,
                    "federation_id": subscription.federation_id,
                    "subscription_id": subscription.subscription_id,
                    "subscription_revision": subscription.revision,
                    "consumer_id": subscription.consumer_id,
                    "prior_attempt_number": attempt.attempt_number - 1,
                    "expected_revision": int(queue_row["revision"]),
                    "fencing_epoch": expected_fencing_epoch,
                },
            )
            if not marked:
                raise FederationRepositoryConflict("durable delivery attempt CAS conflicted")
            return {"attempt": attempt.to_dict()}

        result = self._submit(command, apply)
        return DeliveryAttempt.from_dict(result.result["attempt"])  # type: ignore[return-value]

    def record_delivery_failure(
        self,
        failure: DurableDeliveryFailure,
        *,
        tenant_id: str,
        federation_id: str,
        subscription_revision: int,
        retry_budget: int,
        circuit_breaker_failures: int,
        expected_fencing_epoch: int,
        idempotency_key: str,
    ) -> DurableFailureCommit:
        """Atomically retry/dead-letter and optionally quarantine a consumer."""

        if not isinstance(failure, DurableDeliveryFailure):
            raise FederationContractError("failure must be DurableDeliveryFailure")
        tenant_id = _identifier(tenant_id, "tenant_id")
        federation_id = _identifier(federation_id, "federation_id")
        subscription_revision = _integer(
            subscription_revision, "subscription_revision", minimum=1
        )
        retry_budget = _integer(retry_budget, "retry_budget", maximum=1000)
        circuit_breaker_failures = _integer(
            circuit_breaker_failures,
            "circuit_breaker_failures",
            minimum=1,
            maximum=1000,
        )
        expected_fencing_epoch = _integer(
            expected_fencing_epoch, "expected_fencing_epoch", minimum=1
        )
        exposed = failure.exposed
        failure_body = {
            "attempt_id": exposed.attempt.attempt_id,
            "error_code": failure.error_code,
            "evidence_ref": failure.evidence_ref,
        }
        failure_id = f"delivery-failure:{content_identity(failure_body)}"
        command = self._command(
            command_id=f"command:delivery-failure:{failure_id}",
            idempotency_key=_identifier(idempotency_key, "idempotency_key"),
            parameters={
                "operation": "event.delivery.fail",
                "failure_id": failure_id,
                "attempt_id": exposed.attempt.attempt_id,
                "tenant_id": tenant_id,
                "federation_id": federation_id,
                "subscription_id": exposed.attempt.subscription_id,
                "subscription_revision": subscription_revision,
                "expected_fencing_epoch": expected_fencing_epoch,
            },
        )

        def apply(
            _txn: StateTransaction,
            _command: StateCommand,
            live: Any,
        ) -> Mapping[str, Any]:
            subscription = self._load_subscription(
                exposed.attempt.subscription_id,
                tenant_id=tenant_id,
                federation_id=federation_id,
            )
            if subscription.consumer_id != exposed.attempt.consumer_id:
                raise FederationAuthorityError("delivery failure consumer differs")
            if subscription.revision != subscription_revision:
                raise StaleSubscriptionError("delivery failure subscription is stale")
            if subscription.retry_budget != retry_budget:
                raise FederationAuthorityError("retry budget differs from subscription")
            if subscription.state is not SubscriptionState.ACTIVE:
                raise FederationAuthorityError("delivery failure subscription is not active")
            if live.fence_epoch != expected_fencing_epoch:
                raise FederationRepositoryConflict("delivery failure fence is stale")
            attempt_rows = self.__client.execute(
                "casf_select_attempt_for_failure",
                {
                    "attempt_id": exposed.attempt.attempt_id,
                    "tenant_id": subscription.tenant_id,
                    "federation_id": subscription.federation_id,
                    "subscription_id": subscription.subscription_id,
                    "subscription_revision": subscription.revision,
                    "consumer_id": subscription.consumer_id,
                    "fencing_epoch": expected_fencing_epoch,
                },
            )
            if not attempt_rows:
                raise FederationRepositoryNotFound(
                    "delivery failure has no durable delivered attempt"
                )
            row = attempt_rows[0]
            if (
                str(row["status"]) != DeliveryState.DELIVERED.value
                or str(row["queue_status"]) != DeliveryState.DELIVERED.value
                or str(row["delivery_id"]) != exposed.queued.delivery.delivery_id
                or int(row["attempt_number"]) != exposed.attempt.attempt_number
            ):
                raise FederationRepositoryConflict("delivery failure attempt is stale")
            subscription_rows = self.__client.execute(
                "casf_select_subscription",
                {
                    "subscription_id": subscription.subscription_id,
                    "tenant_id": subscription.tenant_id,
                    "federation_id": subscription.federation_id,
                },
            )
            if len(subscription_rows) != 1:
                raise FederationAuthorityError(
                    "delivery failure lost subscription authority"
                )
            projected_failures = (
                int(subscription_rows[0]["consecutive_failures"]) + 1
            )
            exhausted = (
                exposed.attempt.attempt_number > retry_budget
                or projected_failures >= circuit_breaker_failures
            )
            next_state = (
                DeliveryState.DEAD_LETTERED if exhausted else DeliveryState.RETRY
            )
            failed_attempt = replace(
                exposed.attempt,
                state=next_state,
                error_code=failure.error_code,
                recorded_at=failure.recorded_at,
            )
            marked = self.__client.execute(
                "casf_mark_delivery_failed",
                {
                    "status": next_state.value,
                    "error_code": failure.error_code,
                    "finished_at": failure.recorded_at,
                    "body_json": _json(failed_attempt.to_dict()),
                    "attempt_id": exposed.attempt.attempt_id,
                    "tenant_id": str(row["tenant_id"]),
                    "federation_id": str(row["federation_id"]),
                    "subscription_id": subscription.subscription_id,
                    "subscription_revision": subscription.revision,
                    "consumer_id": subscription.consumer_id,
                    "fencing_epoch": expected_fencing_epoch,
                },
            )
            if not marked:
                raise FederationRepositoryConflict("delivery failure CAS conflicted")
            failure_rows = self.__client.execute(
                "casf_increment_subscription_failures",
                {
                    "updated_at": failure.recorded_at,
                    "tenant_id": subscription.tenant_id,
                    "federation_id": subscription.federation_id,
                    "subscription_id": subscription.subscription_id,
                    "subscription_revision": subscription.revision,
                    "consumer_id": subscription.consumer_id,
                },
            )
            if not failure_rows:
                raise FederationRepositoryConflict(
                    "subscription failure counter transition conflicted"
                )
            failure_count = int(failure_rows[0]["consecutive_failures"])
            if failure_count != projected_failures:
                raise FederationRepositoryConflict(
                    "subscription breaker count changed within one owner transaction"
                )
            quarantined = failure_count >= circuit_breaker_failures
            if quarantined:
                changed = self.__client.execute(
                    "casf_quarantine_subscription",
                    {
                        "updated_at": failure.recorded_at,
                        "tenant_id": subscription.tenant_id,
                        "federation_id": subscription.federation_id,
                        "subscription_id": subscription.subscription_id,
                        "subscription_revision": subscription.revision,
                        "consumer_id": subscription.consumer_id,
                    },
                )
                if not changed:
                    raise FederationRepositoryConflict(
                        "subscription quarantine transition conflicted"
                    )
            queue_state = "dead_lettered" if exhausted else "retry"
            queue_updated = self.__client.execute(
                "casf_update_queue_after_failure",
                {
                    "status": queue_state,
                    "updated_at": failure.recorded_at,
                    "delivery_id": str(row["delivery_id"]),
                    "tenant_id": str(row["tenant_id"]),
                    "federation_id": str(row["federation_id"]),
                    "expected_revision": int(row["queue_revision"]),
                    "fencing_epoch": expected_fencing_epoch,
                },
            )
            if not queue_updated:
                raise FederationRepositoryConflict("delivery queue failure CAS conflicted")

            dead_letter: DeadLetter | None = None
            if exhausted:
                dead_letter = DeadLetter(
                    dead_letter_id=(
                        "dead-letter:"
                        + content_identity(
                            {
                                "delivery_id": str(row["delivery_id"]),
                                "attempt": exposed.attempt.attempt_number,
                            }
                        )
                    ),
                    event_id=exposed.attempt.event_id,
                    subscription_id=subscription.subscription_id,
                    consumer_id=subscription.consumer_id,
                    retry_count=exposed.attempt.attempt_number,
                    error_code=failure.error_code,
                    evidence_ref=failure.evidence_ref,
                    quarantined=quarantined,
                    created_at=failure.recorded_at,
                    expires_at=failure.expires_at,
                )
                self.__client.execute(
                    "casf_insert_dead_letter",
                    {
                        "dead_letter_id": dead_letter.dead_letter_id,
                        "tenant_id": str(row["tenant_id"]),
                        "federation_id": str(row["federation_id"]),
                        "event_id": dead_letter.event_id,
                        "outbox_id": str(row["outbox_id"]),
                        "subscription_id": dead_letter.subscription_id,
                        "subscription_revision": subscription.revision,
                        "consumer_id": dead_letter.consumer_id,
                        "retry_count": dead_letter.retry_count,
                        "error_code": dead_letter.error_code,
                        "evidence_ref": dead_letter.evidence_ref,
                        "quarantined": dead_letter.quarantined,
                        "created_at": dead_letter.created_at,
                        "expires_at": dead_letter.expires_at or None,
                        "body_json": _json(dead_letter.to_dict()),
                    },
                )
            result = FailureResult(
                attempt=failed_attempt,
                dead_letter=dead_letter,
                retry_scheduled=not exhausted,
                subscription_quarantined=quarantined,
            )
            return {
                "failure_id": failure_id,
                "attempt": result.attempt.to_dict(),
                "dead_letter": (
                    None if result.dead_letter is None else result.dead_letter.to_dict()
                ),
                "retry_scheduled": result.retry_scheduled,
                "subscription_quarantined": result.subscription_quarantined,
                "store_generation": live.generation,
                # Failure/retry releases durable queue capacity just like an
                # acknowledgement.  Re-arm the owner-local outbox pump even
                # when this event is below its routing watermark.
                "event_global_sequence": int(row["event_global_sequence"]),
            }

        committed = self._submit(command, apply).result
        dead_letter_body = committed["dead_letter"]
        result = FailureResult(
            attempt=DeliveryAttempt.from_dict(committed["attempt"]),  # type: ignore[arg-type]
            dead_letter=(
                None
                if dead_letter_body is None
                else DeadLetter.from_dict(dead_letter_body)  # type: ignore[arg-type]
            ),
            retry_scheduled=bool(committed["retry_scheduled"]),
            subscription_quarantined=bool(committed["subscription_quarantined"]),
        )
        return DurableFailureCommit(
            failure_id=str(committed["failure_id"]),
            result=result,
            store_generation=int(committed["store_generation"]),
        )

    def is_subscription_quarantined(
        self,
        subscription_id: str,
        subscription_revision: int,
        *,
        tenant_id: str,
        federation_id: str,
    ) -> bool:
        subscription_id = _identifier(subscription_id, "subscription_id")
        tenant_id = _identifier(tenant_id, "tenant_id")
        federation_id = _identifier(federation_id, "federation_id")
        revision = _integer(
            subscription_revision, "subscription_revision", minimum=1
        )
        rows = self.__client.execute(
            "casf_is_subscription_quarantined",
            {
                "tenant_id": tenant_id,
                "federation_id": federation_id,
                "subscription_id": subscription_id,
                "subscription_revision": revision,
            },
        )
        if not rows:
            raise StaleSubscriptionError("subscription revision is absent")
        return str(rows[0]["status"]) == SubscriptionState.QUARANTINED.value

    def list_dead_letters(
        self,
        subscription_id: str,
        subscription_revision: int,
        *,
        tenant_id: str,
        federation_id: str,
        maximum: int,
    ) -> tuple[DeadLetter, ...]:
        subscription = self._load_subscription(
            _identifier(subscription_id, "subscription_id"),
            tenant_id=tenant_id,
            federation_id=federation_id,
        )
        revision = _integer(
            subscription_revision, "subscription_revision", minimum=1
        )
        if subscription.revision != revision:
            raise StaleSubscriptionError("dead-letter subscription revision is stale")
        limit = _integer(maximum, "maximum", minimum=1, maximum=4096)
        rows = self.__client.execute(
            "casf_list_dead_letters",
            {
                "tenant_id": subscription.tenant_id,
                "federation_id": subscription.federation_id,
                "subscription_id": subscription.subscription_id,
                "subscription_revision": revision,
                "consumer_id": subscription.consumer_id,
                "limit": limit,
            },
        )
        return tuple(
            DeadLetter.from_dict(_decode(row["body_json"]))  # type: ignore[misc]
            for row in rows
        )

    def retry_dead_letter(
        self,
        dead_letter_id: str,
        *,
        tenant_id: str,
        federation_id: str,
        subscription_id: str,
        subscription_revision: int,
        expected_fencing_epoch: int,
        recorded_at: str,
        idempotency_key: str,
    ) -> DeadLetterRetryCommit:
        """Resolve and requeue one dead letter without erasing its evidence."""

        dead_letter_id = _identifier(dead_letter_id, "dead_letter_id")
        tenant_id = _identifier(tenant_id, "tenant_id")
        federation_id = _identifier(federation_id, "federation_id")
        subscription_id = _identifier(subscription_id, "subscription_id")
        subscription_revision = _integer(
            subscription_revision, "subscription_revision", minimum=1
        )
        expected_fencing_epoch = _integer(
            expected_fencing_epoch, "expected_fencing_epoch", minimum=1
        )
        recorded_at = _timestamp(recorded_at, "recorded_at")
        idempotency_key = _identifier(idempotency_key, "idempotency_key")
        subscription = self._load_subscription(
            subscription_id,
            tenant_id=tenant_id,
            federation_id=federation_id,
        )
        if subscription.revision != subscription_revision:
            raise StaleSubscriptionError("dead-letter retry subscription is stale")
        invocation = content_identity(
            {
                "dead_letter_id": dead_letter_id,
                "tenant_id": tenant_id,
                "federation_id": federation_id,
                "subscription_id": subscription_id,
                "subscription_revision": subscription_revision,
                "fencing_epoch": expected_fencing_epoch,
            }
        )
        command = self._command(
            command_id=f"command:dead-letter-retry:{invocation}",
            idempotency_key=idempotency_key,
            parameters={
                "operation": "event.dead-letter.retry",
                "dead_letter_id": dead_letter_id,
                "tenant_id": tenant_id,
                "federation_id": federation_id,
                "subscription_id": subscription_id,
                "subscription_revision": subscription_revision,
                "expected_fencing_epoch": expected_fencing_epoch,
            },
        )

        def apply(
            _txn: StateTransaction,
            _command: StateCommand,
            live: Any,
        ) -> Mapping[str, Any]:
            if live.fence_epoch != expected_fencing_epoch:
                raise FederationRepositoryConflict("dead-letter retry fence is stale")
            rows = self.__client.execute(
                "casf_select_dead_letter_for_retry",
                {
                    "dead_letter_id": dead_letter_id,
                    "tenant_id": tenant_id,
                    "federation_id": federation_id,
                    "subscription_id": subscription_id,
                    "subscription_revision": subscription_revision,
                    "consumer_id": subscription.consumer_id,
                },
            )
            if not rows:
                raise FederationRepositoryNotFound("owned dead letter is absent")
            row = rows[0]
            if str(row["status"]) != "open":
                raise FederationRepositoryConflict("dead letter is not open")
            if str(row["queue_status"]) != "dead_lettered":
                raise FederationRepositoryConflict("dead-letter queue state differs")
            if int(row["fencing_epoch"]) > expected_fencing_epoch:
                raise FederationRepositoryConflict("dead-letter queue fence is newer")
            resolved = self.__client.execute(
                "casf_resolve_dead_letter_for_retry",
                {
                    "resolved_at": recorded_at,
                    "dead_letter_id": dead_letter_id,
                    "tenant_id": tenant_id,
                    "federation_id": federation_id,
                    "subscription_id": subscription_id,
                    "subscription_revision": subscription_revision,
                    "consumer_id": subscription.consumer_id,
                    "expected_revision": int(row["revision"]),
                },
            )
            requeued = self.__client.execute(
                "casf_requeue_dead_letter_delivery",
                {
                    "available_at": recorded_at,
                    "updated_at": recorded_at,
                    "fencing_epoch": expected_fencing_epoch,
                    "delivery_id": str(row["delivery_id"]),
                    "tenant_id": tenant_id,
                    "federation_id": federation_id,
                    "subscription_id": subscription_id,
                    "subscription_revision": subscription_revision,
                    "consumer_id": subscription.consumer_id,
                    "expected_revision": int(row["queue_revision"]),
                },
            )
            activated = self.__client.execute(
                "casf_unquarantine_subscription",
                {
                    "updated_at": recorded_at,
                    "tenant_id": tenant_id,
                    "federation_id": federation_id,
                    "subscription_id": subscription_id,
                    "subscription_revision": subscription_revision,
                    "consumer_id": subscription.consumer_id,
                },
            )
            if not resolved or not requeued or not activated:
                raise FederationRepositoryConflict("dead-letter retry CAS conflicted")
            return {
                "dead_letter_id": dead_letter_id,
                "delivery_id": str(row["delivery_id"]),
                "subscription_id": subscription_id,
                "subscription_revision": subscription_revision,
                "requeued": True,
                "unquarantined": True,
                "store_generation": live.generation,
            }

        result = self._submit(command, apply).result
        return DeadLetterRetryCommit(
            dead_letter_id=str(result["dead_letter_id"]),
            delivery_id=str(result["delivery_id"]),
            subscription_id=str(result["subscription_id"]),
            subscription_revision=int(result["subscription_revision"]),
            requeued=bool(result["requeued"]),
            unquarantined=bool(result["unquarantined"]),
            store_generation=int(result["store_generation"]),
        )

    def acknowledge_event(
        self,
        acknowledgement: EventAcknowledgement,
        *,
        tenant_id: str,
        federation_id: str,
        delivery_attempt_id: str,
        expected_cursor_revision: int,
        expected_fencing_epoch: int,
        disposition: str = "processed",
        idempotency_key: str | None = None,
    ) -> ConsumerCursor:
        """Atomically persist an acknowledgement and advance its durable cursor.

        Network delivery remains at least once.  Repeating the same command is
        an idempotent replay, while a different acknowledgement against a stale
        revision or fence cannot advance the authoritative cursor.
        """

        if not isinstance(acknowledgement, EventAcknowledgement):
            raise FederationContractError("acknowledgement must be EventAcknowledgement")
        tenant_id = _identifier(tenant_id, "tenant_id")
        federation_id = _identifier(federation_id, "federation_id")
        delivery_attempt_id = _identifier(delivery_attempt_id, "delivery_attempt_id")
        disposition = _identifier(disposition, "disposition")
        expected_cursor_revision = _integer(
            expected_cursor_revision, "expected_cursor_revision", minimum=1
        )
        expected_fencing_epoch = _integer(
            expected_fencing_epoch, "expected_fencing_epoch", minimum=1
        )
        invocation_ref = content_identity(
            {
                "acknowledgement_cid": acknowledgement.cid,
                "tenant_id": tenant_id,
                "federation_id": federation_id,
                "delivery_attempt_id": delivery_attempt_id,
                "expected_cursor_revision": expected_cursor_revision,
                "expected_fencing_epoch": expected_fencing_epoch,
                "disposition": disposition,
            }
        )
        key = idempotency_key or f"acknowledge:{acknowledgement.cid}"
        command = self._command(
            command_id=f"command:event-acknowledge:{invocation_ref}",
            idempotency_key=key,
            command_kind=CommandKind.APPEND,
            parameters={
                "operation": "event.acknowledge",
                "acknowledgement_id": acknowledgement.acknowledgement_id,
                "tenant_id": tenant_id,
                "federation_id": federation_id,
                "consumer_id": acknowledgement.consumer_id,
                "subscription_id": acknowledgement.subscription_id,
                "event_id": acknowledgement.event_id,
                "delivery_attempt_id": delivery_attempt_id,
                "expected_cursor_revision": expected_cursor_revision,
                "expected_fencing_epoch": expected_fencing_epoch,
                "disposition": disposition,
            },
        )

        def apply(
            _txn: StateTransaction,
            _command: StateCommand,
            live: Any,
        ) -> Mapping[str, Any]:
            subscription = self._load_subscription(
                acknowledgement.subscription_id,
                tenant_id=tenant_id,
                federation_id=federation_id,
            )
            if subscription.consumer_id != acknowledgement.consumer_id:
                raise FederationAuthorityError("acknowledging consumer does not own subscription")
            if subscription.revision != acknowledgement.subscription_revision:
                raise StaleSubscriptionError("acknowledgement subscription revision is stale")
            cursor_rows = self.__client.execute(
                "casf_select_consumer_cursor",
                {
                    "consumer_id": acknowledgement.consumer_id,
                    "subscription_id": acknowledgement.subscription_id,
                    "tenant_id": subscription.tenant_id,
                    "federation_id": subscription.federation_id,
                },
            )
            if not cursor_rows:
                raise FederationRepositoryNotFound("consumer cursor is absent")
            row = cursor_rows[0]
            if int(row["revision"]) != expected_cursor_revision:
                raise FederationRepositoryConflict("consumer cursor revision is stale")
            if int(row["fencing_epoch"]) != expected_fencing_epoch:
                raise FederationRepositoryConflict("consumer cursor fence is stale")
            if int(row["store_generation"]) > live.generation:
                raise FederationRepositoryConflict("consumer cursor generation is ahead")
            if int(row["subscription_revision"]) != subscription.revision:
                raise StaleSubscriptionError("consumer cursor subscription is stale")
            event_rows = self.__client.execute(
                "casf_select_event_for_ack",
                {
                    "event_id": acknowledgement.event_id,
                    "tenant_id": subscription.tenant_id,
                    "federation_id": subscription.federation_id,
                    "subscription_id": subscription.subscription_id,
                    "subscription_revision": subscription.revision,
                    "consumer_id": subscription.consumer_id,
                },
            )
            if not event_rows:
                raise FederationRepositoryNotFound("acknowledged event is absent")
            event = self._event_from_row(event_rows[0])
            if event.global_sequence != acknowledgement.global_sequence:
                raise FederationAuthorityError(
                    "acknowledgement event sequence differs from authority"
                )
            if acknowledgement.global_sequence <= int(row["global_sequence"]):
                raise FederationRepositoryConflict(
                    "acknowledgement does not advance the current cursor"
                )
            if not event_matches_subscription(event, subscription):
                raise FederationAuthorityError(
                    "acknowledged event does not match the owned subscription"
                )
            next_eligible = self._routed_events_for_loaded_subscription(
                subscription,
                after_cursor=int(row["global_sequence"]),
                maximum_events=1,
            )
            if not next_eligible or next_eligible[0].event_id != event.event_id:
                raise FederationRepositoryConflict(
                    "acknowledgement would skip earlier eligible work"
                )
            delivery_rows = self.__client.execute(
                "casf_select_delivery_for_ack",
                {
                    "attempt_id": delivery_attempt_id,
                    "tenant_id": subscription.tenant_id,
                    "federation_id": subscription.federation_id,
                    "event_id": event.event_id,
                    "subscription_id": subscription.subscription_id,
                    "subscription_revision": subscription.revision,
                    "consumer_id": subscription.consumer_id,
                    "fencing_epoch": expected_fencing_epoch,
                },
            )
            if not delivery_rows:
                raise FederationAuthorityError("acknowledgement has no owned delivery attempt")
            if str(delivery_rows[0]["status"]) != DeliveryState.DELIVERED.value:
                raise FederationRepositoryConflict(
                    "delivery attempt is not awaiting acknowledgement"
                )
            if str(delivery_rows[0]["queue_status"]) != DeliveryState.DELIVERED.value:
                raise FederationRepositoryConflict(
                    "durable queue is not awaiting acknowledgement"
                )

            advanced = ConsumerCursor(
                consumer_id=acknowledgement.consumer_id,
                subscription_id=acknowledgement.subscription_id,
                subscription_revision=acknowledgement.subscription_revision,
                global_sequence=acknowledgement.global_sequence,
                store_generation=live.generation,
                revision=expected_cursor_revision + 1,
                updated_at=acknowledgement.recorded_at,
            )
            marked = self.__client.execute(
                "casf_mark_delivery_acknowledged",
                {
                    "finished_at": acknowledgement.recorded_at,
                    "attempt_id": delivery_attempt_id,
                    "tenant_id": subscription.tenant_id,
                    "federation_id": subscription.federation_id,
                    "event_id": event.event_id,
                    "subscription_id": subscription.subscription_id,
                    "consumer_id": subscription.consumer_id,
                    "subscription_revision": subscription.revision,
                    "fencing_epoch": expected_fencing_epoch,
                },
            )
            if not marked:
                raise FederationRepositoryConflict(
                    "delivery acknowledgement fence or state is stale"
                )
            queue_marked = self.__client.execute(
                "casf_mark_queue_acknowledged",
                {
                    "updated_at": acknowledgement.recorded_at,
                    "delivery_id": delivery_rows[0]["delivery_id"],
                    "tenant_id": subscription.tenant_id,
                    "federation_id": subscription.federation_id,
                    "subscription_id": subscription.subscription_id,
                    "subscription_revision": subscription.revision,
                    "consumer_id": subscription.consumer_id,
                    "fencing_epoch": expected_fencing_epoch,
                    "expected_revision": delivery_rows[0]["queue_revision"],
                },
            )
            if not queue_marked:
                raise FederationRepositoryConflict(
                    "durable queue acknowledgement CAS conflicted"
                )
            reset = self.__client.execute(
                "casf_reset_subscription_failures",
                {
                    "updated_at": acknowledgement.recorded_at,
                    "tenant_id": subscription.tenant_id,
                    "federation_id": subscription.federation_id,
                    "subscription_id": subscription.subscription_id,
                    "subscription_revision": subscription.revision,
                    "consumer_id": subscription.consumer_id,
                },
            )
            if not reset:
                raise FederationRepositoryConflict(
                    "subscription failure counter reset conflicted"
                )
            self.__client.execute(
                "casf_insert_event_acknowledgement",
                {
                    "acknowledgement_id": acknowledgement.acknowledgement_id,
                    "tenant_id": subscription.tenant_id,
                    "federation_id": subscription.federation_id,
                    "event_id": acknowledgement.event_id,
                    "subscription_id": acknowledgement.subscription_id,
                    "consumer_id": acknowledgement.consumer_id,
                    "subscription_revision": acknowledgement.subscription_revision,
                    "global_sequence": acknowledgement.global_sequence,
                    "delivery_attempt_id": delivery_attempt_id,
                    "cursor_revision": advanced.revision,
                    "fencing_epoch": expected_fencing_epoch,
                    "disposition": disposition,
                    "processed_effect_ref": acknowledgement.processed_effect_ref,
                    "recorded_at": acknowledgement.recorded_at,
                    "body_json": _json(acknowledgement.to_dict()),
                },
            )
            updated = self.__client.execute(
                "casf_advance_consumer_cursor",
                {
                    "global_sequence": advanced.global_sequence,
                    "store_generation": advanced.store_generation,
                    "last_event_id": acknowledgement.event_id,
                    "updated_at": advanced.updated_at,
                    "body_json": _json(advanced.to_dict()),
                    "consumer_id": advanced.consumer_id,
                    "subscription_id": advanced.subscription_id,
                    "subscription_revision": advanced.subscription_revision,
                    "expected_revision": expected_cursor_revision,
                    "expected_fencing_epoch": expected_fencing_epoch,
                    "upper_global_sequence": advanced.global_sequence,
                },
            )
            if not updated or int(updated[0]["revision"]) != advanced.revision:
                raise FederationRepositoryConflict("consumer cursor CAS did not advance")
            return {
                "cursor": advanced.to_dict(),
                # Re-arm the owner-local outbox pump even when the released
                # capacity belongs to an event below its current watermark.
                "event_global_sequence": event.global_sequence,
            }

        result = self._submit(command, apply)
        return ConsumerCursor.from_dict(result.result["cursor"])  # type: ignore[return-value]

    def events_for_subscription(
        self,
        *,
        consumer_id: str,
        subscription_id: str,
        subscription_revision: int,
        after_cursor: int,
        maximum_events: int,
    ) -> tuple[DomainEvent, ...]:
        consumer_id = _identifier(consumer_id, "consumer_id")
        subscription_id = _identifier(subscription_id, "subscription_id")
        scope_rows = self.__client.execute(
            "casf_resolve_consumer_cursor_scope",
            {"consumer_id": consumer_id, "subscription_id": subscription_id},
        )
        if not scope_rows:
            raise FederationRepositoryNotFound("consumer cursor is absent")
        tenant_id = str(scope_rows[0]["tenant_id"])
        federation_id = str(scope_rows[0]["federation_id"])
        subscription = self._load_subscription(
            subscription_id,
            tenant_id=tenant_id,
            federation_id=federation_id,
        )
        if subscription.consumer_id != consumer_id:
            raise FederationAuthorityError("consumer does not own subscription")
        if subscription.revision != subscription_revision:
            raise StaleSubscriptionError("subscription revision is stale")
        cursor = self.get_cursor(
            tenant_id=tenant_id,
            federation_id=federation_id,
            consumer_id=consumer_id,
            subscription_id=subscription_id,
        )
        after_cursor = _integer(after_cursor, "after_cursor")
        if after_cursor != cursor.global_sequence:
            raise FederationRepositoryConflict(
                "caller event cursor differs from the durable consumer cursor"
            )
        if cursor.subscription_revision != subscription_revision:
            raise StaleSubscriptionError("durable cursor subscription revision is stale")
        _integer(maximum_events, "maximum_events", minimum=1, maximum=4096)
        return self._routed_events_for_loaded_subscription(
            subscription,
            after_cursor=after_cursor,
            maximum_events=maximum_events,
        )


__all__ = [
    "FederationRepositoryConflict",
    "FederationRepositoryError",
    "FederationRepositoryNotFound",
    "FederationStateRepository",
]
