"""Real DuckDB integration tests for the sealed CASF state repository.

The direct DuckDB inspection helpers in this module are only called after the
typed embedded client has closed.  The tests therefore retain one exclusive
state-owner connection at a time and do not model multi-process authority.
"""

# The package still supports Python 3.8, where ``datetime.UTC`` is absent.
# ruff: noqa: UP017

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.federation import contracts
from ipfs_accelerate_py.agent_supervisor.federation.budgets import (
    AuthoritativeBudgetAuthority,
)
from ipfs_accelerate_py.agent_supervisor.federation.durable_event_router import (
    DurableEventRouter,
)
from ipfs_accelerate_py.agent_supervisor.federation.events import (
    ConsumerCursor,
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
from ipfs_accelerate_py.agent_supervisor.federation.outbox import (
    EventDraft,
    materialize_event,
)
from ipfs_accelerate_py.agent_supervisor.federation.registry import (
    FederationRepositoryConflict,
    FederationRepositoryNotFound,
    FederationStateRepository,
    _casf_templates,
)
from ipfs_accelerate_py.agent_supervisor.federation.trigger import (
    ResolvedRepository,
    resolved_authorization_scope_identity,
)
from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
    current_process_birth,
    read_process_birth,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_migrations import (
    duckdb_available,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_schema import (
    install_control_plane_schema,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_transactions import (
    TransactionError,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
    open_duckdb_connection,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.quack_state_client import (
    QuackClientSQLError,
    QuackStateClient,
    StatementKind,
    StatementTemplate,
    open_embedded_client,
)
from test.api.causal_federation.test_contracts import EXPIRY, NOW, sample_binding
from test.api.causal_federation.test_trigger import (
    resolved_for,
    sample_policy,
    sample_request,
)

pytestmark = pytest.mark.skipif(
    not duckdb_available(),
    reason="DuckDB is required for causal-federation repository tests",
)


FailureHook = Callable[[str], None]
_BUDGET_CAPACITY = {
    contracts.BudgetDimensionName.CPU_MILLIS: 10_000,
    contracts.BudgetDimensionName.INPUT_TOKENS: 10_000,
}


def _open_repository(
    tmp_path: Path,
    *,
    maximum_supervisors: int = 2,
    maximum_subagents: int = 2,
    maximum_concurrent_subagents: int | None = None,
    event_notifier: Callable[[int], None] | None = None,
    outbox_notifier: Callable[[int], None] | None = None,
    test_failure_hook: FailureHook | None = None,
) -> tuple[
    Path,
    QuackStateClient,
    FederationStateRepository,
    contracts.FederationBinding,
    contracts.FederationRequest,
    contracts.FederationPolicy,
]:
    database = tmp_path / "control.duckdb"
    report = install_control_plane_schema(database, owner_id="owner:registry-migration")
    assert report.to_version == 2
    client = open_embedded_client(
        database,
        owner_id="owner:registry",
        seed_generation=True,
    )
    generation = client.load_generation()
    binding = sample_binding(
        control_plane_generation=generation.generation,
        supervisor_population=0,
    )
    request = sample_request(
        binding=binding,
        maximum_supervisors=maximum_supervisors,
        maximum_subagents=maximum_subagents,
    )
    policy = sample_policy(
        binding,
        maximum_supervisors=maximum_supervisors,
        maximum_subagents=maximum_subagents,
        maximum_concurrent_subagents=(
            maximum_subagents
            if maximum_concurrent_subagents is None
            else maximum_concurrent_subagents
        ),
    )
    repository = FederationStateRepository(
        client,
        event_notifier=event_notifier,
        outbox_notifier=outbox_notifier,
        test_failure_hook=test_failure_hook,
    )
    return database, client, repository, binding, request, policy


def _create(
    repository: FederationStateRepository,
    *,
    request: contracts.FederationRequest,
    policy: contracts.FederationPolicy,
) -> tuple[contracts.FederationIdentity, contracts.FederationReceipt]:
    repositories = resolved_for(request.binding)
    reservation = AuthoritativeBudgetAuthority(
        repository,
        capacity=_BUDGET_CAPACITY,
        authority_id="authority:budget:registry-test",
        now=lambda: datetime(2030, 1, 1, tzinfo=timezone.utc),
    ).reserve(request, policy)
    return repository.create_federation(
        request=request,
        policy=policy,
        repositories=repositories,
        budget_reservation=reservation,
        authentication_evidence_ref=request.binding.authorization_evidence_ref,
        authorization_decision=_authorization_decision(
            request,
            policy,
            repositories,
        ),
    )


def _authorization_decision(
    request: contracts.FederationRequest,
    policy: contracts.FederationPolicy,
    repositories: tuple[ResolvedRepository, ...],
) -> contracts.FederationAuthorizationDecision:
    return contracts.FederationAuthorizationDecision(
        request_cid=request.cid,
        caller_did=request.caller_did,
        delegation_chain_cid="delegation-chain:validated-registry-test",
        audience=request.audience,
        operation=contracts.FederationOperation.CREATE,
        resolved_scope_cid=resolved_authorization_scope_identity(
            repositories,
            request.effect_scope,
        ),
        policy_id=policy.record_id,
        policy_revision=policy.revision,
        verdict=contracts.FederationAuthorizationVerdict.ADMITTED,
        reason=(
            contracts.FederationAuthorizationReason.AUTHENTICATED_DELEGATED_POLICY_ADMITTED
        ),
        authentication_evidence_cid="authentication-evidence-cid:registry-test",
        expires_at=request.expiry,
        decided_at=NOW,
    )


def _unpersisted_reservation(
    request: contracts.FederationRequest,
    policy: contracts.FederationPolicy,
) -> contracts.BudgetReservation:
    dimensions = tuple(
        contracts.BudgetDimension(
            name=item.name,
            ceiling=item.ceiling,
            reserved=item.ceiling,
            consumed=0,
        )
        for budget in (request.resource_budget, request.token_budget)
        for item in budget.dimensions
    )
    return contracts.BudgetReservation(
        record_id="budget-reservation:unpersisted",
        revision=1,
        binding=request.binding,
        parent_budget_id=request.binding.budget_ref,
        owner_id=f"federation:{request.cid}",
        dimensions=dimensions,
        status="reserved",
        request_cid=request.cid,
        idempotency_key=request.idempotency_key,
        policy_ref=policy.record_id,
        policy_revision=policy.revision,
        resource_budget_ref=request.resource_budget.record_id,
        token_budget_ref=request.token_budget.record_id,
        issued_at=NOW,
        expires_at=request.expiry,
        authorization_evidence_ref="budget-admission:unpersisted",
    )


def _supervisor(
    *,
    binding: contracts.FederationBinding,
    federation_id: str,
    suffix: str = "one",
    task_refs: tuple[str, ...] = (),
) -> tuple[contracts.SupervisorInstance, contracts.SupervisorAssignment]:
    supervisor_id = f"supervisor:{suffix}"
    instance = contracts.SupervisorInstance(
        record_id=supervisor_id,
        revision=1,
        binding=binding,
        state=contracts.FederationLifecycleState.DECLARED.value,
        federation_id=federation_id,
        parent_supervisor_id="",
        role=contracts.SupervisorRole.COORDINATOR,
        lease_id=f"lease:supervisor:{suffix}",
        fencing_epoch=1,
    )
    assignment = contracts.SupervisorAssignment(
        record_id=f"assignment:supervisor:{suffix}",
        revision=1,
        binding=binding,
        subject_id=supervisor_id,
        repository_ids=binding.repository_ids,
        goal_refs=(binding.objective_ref,),
        task_refs=task_refs,
        allowed_task_families=("implementation",),
        fencing_epoch=1,
    )
    return instance, assignment


def _subagent(
    *,
    binding: contracts.FederationBinding,
    federation_id: str,
    supervisor_id: str,
    suffix: str = "one",
) -> contracts.SubagentInstance:
    return contracts.SubagentInstance(
        record_id=f"subagent:{suffix}",
        revision=1,
        binding=binding,
        state=contracts.FederationLifecycleState.ADMITTED.value,
        federation_id=federation_id,
        supervisor_id=supervisor_id,
        task_id=f"task:{suffix}",
        lease_id=f"lease:subagent:{suffix}",
        fencing_epoch=1,
    )


def _definition(
    contract_type: type[contracts.SupervisorDefinition] | type[contracts.SubagentDefinition],
    capability_type: type[contracts.SupervisorCapability] | type[contracts.SubagentCapability],
    *,
    binding: contracts.FederationBinding,
    suffix: str,
) -> tuple[
    contracts.SupervisorDefinition | contracts.SubagentDefinition,
    tuple[contracts.SupervisorCapability | contracts.SubagentCapability, ...],
]:
    capability = capability_type(
        record_id=f"capability:{suffix}",
        revision=1,
        binding=binding,
        name=f"capability {suffix}",
        capabilities=(),
        allowed_operations=(),
        effect_ceiling="effect.read",
        risk_ceiling="risk:bounded",
        resource_budget_ref="budget:resource",
        token_budget_ref="budget:token",
    )
    definition = contract_type(
        record_id=f"definition:{suffix}",
        revision=1,
        binding=binding,
        name=f"definition {suffix}",
        capabilities=(capability.record_id,),
        allowed_operations=(),
        effect_ceiling="effect.read",
        risk_ceiling="risk:bounded",
        resource_budget_ref="budget:resource",
        token_budget_ref="budget:token",
    )
    return definition, (capability,)


def _register_supervisor(
    repository: FederationStateRepository,
    instance: contracts.SupervisorInstance,
    assignment: contracts.SupervisorAssignment,
    *,
    idempotency_key: str,
) -> contracts.SupervisorInstance:
    definition, capabilities = _definition(
        contracts.SupervisorDefinition,
        contracts.SupervisorCapability,
        binding=instance.binding,
        suffix=(
            instance.record_id[len("supervisor:") :]
            if instance.record_id.startswith("supervisor:")
            else instance.record_id
        ),
    )
    return repository.register_supervisor(
        instance,
        assignment,
        definition=definition,
        capabilities=capabilities,  # type: ignore[arg-type]
        idempotency_key=idempotency_key,
    )


def _register_subagent(
    repository: FederationStateRepository,
    instance: contracts.SubagentInstance,
    *,
    idempotency_key: str | None = None,
) -> contracts.SubagentInstance:
    suffix = (
        instance.record_id[len("subagent:") :]
        if instance.record_id.startswith("subagent:")
        else instance.record_id
    )
    definition, capabilities = _definition(
        contracts.SubagentDefinition,
        contracts.SubagentCapability,
        binding=instance.binding,
        suffix=suffix,
    )
    assignment = contracts.SubagentAssignment(
        record_id=f"assignment:subagent:{suffix}",
        revision=1,
        binding=instance.binding,
        subject_id=instance.record_id,
        repository_ids=instance.binding.repository_ids,
        goal_refs=(instance.binding.objective_ref,),
        task_refs=(instance.task_id,) if instance.task_id else (),
        allowed_task_families=("implementation",),
        fencing_epoch=instance.fencing_epoch,
    )
    return repository.register_subagent(
        instance,
        definition=definition,
        assignment=assignment,
        capabilities=capabilities,  # type: ignore[arg-type]
        idempotency_key=idempotency_key,
    )


def _outcome(
    instance: contracts.SubagentInstance,
    *,
    suffix: str = "accepted",
) -> contracts.SubagentOutcome:
    return contracts.SubagentOutcome(
        record_id=f"outcome:{suffix}",
        revision=1,
        binding=instance.binding,
        outcome="succeeded",
        evidence_refs=(f"evidence:{suffix}",),
        recorded_at=NOW,
        federation_id=instance.federation_id,
        supervisor_id=instance.supervisor_id,
        subagent_id=instance.record_id,
        task_id=instance.task_id,
        fencing_epoch=instance.fencing_epoch,
    )


def _snapshot(database: Path, tables: tuple[str, ...]) -> dict[str, int]:
    """Inspect committed state after the typed owner has closed."""

    with open_duckdb_connection(database) as connection:
        result = {
            table: int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in tables
        }
        generation = connection.execute(
            """
            SELECT revision
            FROM store_generations
            ORDER BY generation DESC
            LIMIT 1
            """
        ).fetchone()
        result["store_revision"] = int(generation[0])
        head = connection.execute(
            "SELECT COALESCE(MAX(current_sequence), 0) FROM global_sequence_head"
        ).fetchone()
        result["event_watermark"] = int(head[0])
        return result


def _reopen_repository(
    database: Path,
    *,
    outbox_notifier: Callable[[int], None] | None = None,
) -> tuple[QuackStateClient, FederationStateRepository]:
    client = open_embedded_client(
        database,
        owner_id="owner:registry-reopen",
        seed_generation=False,
    )
    return client, FederationStateRepository(
        client,
        outbox_notifier=outbox_notifier,
    )


def _execute_fixture_template(
    connection: object,
    template_name: str,
    values: dict[str, object],
) -> object:
    template = next(item for item in _casf_templates() if item.name == template_name)
    parameters = [values[name] for name in template.parameter_names]
    return connection.execute(template.sql, parameters)  # type: ignore[attr-defined]


def _insert_event_fixtures(
    database: Path,
    *,
    binding: contracts.FederationBinding,
    federation_id: str,
    repository_ids: tuple[str, ...],
) -> tuple[DomainEvent, ...]:
    """Insert authoritative-shape event fixtures while no owner is attached."""

    events: list[DomainEvent] = []
    with open_duckdb_connection(database) as connection:
        connection.execute("BEGIN TRANSACTION")
        try:
            for ordinal, repository_id in enumerate(repository_ids, start=1):
                global_sequence = int(
                    _execute_fixture_template(
                        connection,
                        "casf_advance_global_head",
                        {"updated_at": NOW},
                    ).fetchone()[0]  # type: ignore[attr-defined]
                )
                stream_sequence = int(
                    _execute_fixture_template(
                        connection,
                        "casf_advance_stream_head",
                        {
                            "updated_at": NOW,
                            "stream_id": federation_id,
                            "tenant_id": binding.tenant_id,
                            "federation_id": federation_id,
                        },
                    ).fetchone()[0]  # type: ignore[attr-defined]
                )
                draft = EventDraft(
                    event_type=EventClass.GOAL_CHANGED,
                    stream_id=federation_id,
                    causal_parent_ids=(),
                    correlation_id=f"correlation:fixture:{global_sequence}",
                    causation_id=f"causation:fixture:{global_sequence}",
                    tenant_id=binding.tenant_id,
                    federation_id=federation_id,
                    repository_id=repository_id,
                    tree_id=binding.repository_tree_ids[0],
                    goal_id=binding.objective_ref,
                    payload_ref=f"payload:fixture:{global_sequence}",
                    changed_fact_refs=(f"fact:fixture:{global_sequence}",),
                    effect_class=EventEffectClass.AUTHORITATIVE_STATE,
                    deduplication_key=f"fixture:{global_sequence}:{ordinal}",
                )
                event, outbox = materialize_event(
                    draft,
                    stream_sequence=stream_sequence,
                    global_sequence=global_sequence,
                    recorded_at=NOW,
                )
                _execute_fixture_template(
                    connection,
                    "casf_insert_domain_event",
                    {
                        "event_id": event.event_id,
                        "event_cid": event.event_cid,
                        "stream_id": event.stream_id,
                        "stream_sequence": event.stream_sequence,
                        "global_sequence": event.global_sequence,
                        "event_type": event.event_type.value,
                        "task_cid": event.task_id,
                        "session_id": "session:fixture",
                        "recorded_at": event.recorded_at,
                        "body_json": json.dumps(event.to_dict(), sort_keys=True),
                        "causal_parent_ids_json": "[]",
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
                        "changed_fact_refs_json": json.dumps(list(event.changed_fact_refs)),
                        "effect_class": event.effect_class.value,
                        "expires_at": None,
                        "deduplication_key": event.deduplication_key,
                        "control_plane_generation": binding.control_plane_generation,
                        "causal_graph_revision": binding.causal_graph_revision,
                    },
                )
                _execute_fixture_template(
                    connection,
                    "casf_insert_changed_fact",
                    {
                        "event_id": event.event_id,
                        "fact_ref": event.changed_fact_refs[0],
                        "ordinal": 1,
                    },
                )
                _execute_fixture_template(
                    connection,
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
                        "body_json": json.dumps(outbox.to_dict(), sort_keys=True),
                    },
                )
                events.append(event)
            connection.commit()
        except Exception:
            connection.rollback()
            raise
    return tuple(events)


def _load_event(database: Path, event_type: EventClass) -> DomainEvent:
    with open_duckdb_connection(database) as connection:
        row = connection.execute(
            """
            SELECT body_json
            FROM domain_events
            WHERE event_type = ?
            ORDER BY global_sequence DESC
            LIMIT 1
            """,
            [event_type.value],
        ).fetchone()
    assert row is not None
    return DomainEvent.from_dict(json.loads(str(row[0])))  # type: ignore[return-value]


def _insert_delivery_fixture(
    database: Path,
    *,
    event: DomainEvent,
    subscription: EventSubscription,
    attempt_id: str,
    consumer_id: str | None = None,
    attempt_number: int = 1,
    status: str = "delivered",
) -> None:
    with open_duckdb_connection(database) as connection:
        outbox = connection.execute(
            "SELECT outbox_id FROM transactional_outbox WHERE event_id = ?",
            [event.event_id],
        ).fetchone()
        assert outbox is not None
        delivery_id = f"delivery:fixture:{attempt_id}"
        routed_consumer_id = consumer_id or subscription.consumer_id
        connection.execute(
            """
            INSERT INTO event_delivery_queue (
                delivery_id, tenant_id, federation_id, subscription_id,
                subscription_revision, consumer_id, decision_id,
                representative_event_id, outbox_id, status, attempt_number,
                fencing_epoch, available_at, revision, created_at, updated_at,
                body_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, 1, ?, ?, '{}')
            """,
            [
                delivery_id,
                subscription.tenant_id,
                subscription.federation_id,
                subscription.subscription_id,
                subscription.revision,
                routed_consumer_id,
                f"decision:fixture:{attempt_id}",
                event.event_id,
                str(outbox[0]),
                status,
                attempt_number,
                NOW,
                NOW,
                NOW,
            ],
        )
        connection.execute(
            """
            INSERT INTO delivery_attempts (
                attempt_id, tenant_id, federation_id, event_id, outbox_id,
                delivery_id, subscription_id, subscription_revision, consumer_id,
                attempt_number, fencing_epoch, status, error_code,
                recorded_at, finished_at, body_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, '', ?, NULL, '{}')
            """,
            [
                attempt_id,
                subscription.tenant_id,
                subscription.federation_id,
                event.event_id,
                str(outbox[0]),
                delivery_id,
                subscription.subscription_id,
                subscription.revision,
                routed_consumer_id,
                attempt_number,
                status,
                NOW,
            ],
        )


def _insert_subagent_fixture(
    database: Path,
    instance: contracts.SubagentInstance,
) -> None:
    with open_duckdb_connection(database) as connection:
        _execute_fixture_template(
            connection,
            "casf_insert_subagent",
            {
                "subagent_id": instance.record_id,
                "tenant_id": instance.binding.tenant_id,
                "federation_id": instance.federation_id,
                "supervisor_id": instance.supervisor_id,
                "subagent_definition_id": "definition:fixture",
                "task_id": instance.task_id,
                "lease_id": instance.lease_id,
                "logical_state": instance.state,
                "admitted_concurrency_slot": False,
                "worker_process_birth_id": "",
                "provider_route_id": "",
                "admission_decision_id": "policy-decision:legacy-fixture",
                "revision": instance.revision,
                "fencing_epoch": instance.fencing_epoch,
                "registered_at": NOW,
                "updated_at": NOW,
                "body_json": json.dumps(instance.to_dict(), sort_keys=True),
            },
        )


def _insert_active_task_execution(
    database: Path,
    instance: contracts.SubagentInstance,
    *,
    include_attempt: bool = True,
) -> None:
    """Materialize scheduler-owned execution authority for an outcome test."""

    with open_duckdb_connection(database) as connection:
        connection.execute(
            """
            INSERT INTO tasks (
                task_cid, task_alias, goal_cid, ordinal, status, revision,
                created_at, updated_at
            ) VALUES (?, ?, ?, 1, 'active', 1, ?, ?)
            """,
            [
                instance.task_id,
                f"alias:{instance.task_id}",
                instance.binding.objective_ref,
                NOW,
                NOW,
            ],
        )
        connection.execute(
            """
            INSERT INTO leases (
                task_cid, claim_cid, resolution_cid, claimant_did,
                logical_epoch, fencing_token, expires_at_ms, attempt,
                state, started_at_ms, owner_session_id, fence_epoch, revision
            ) VALUES (?, ?, '', ?, 1, 1, 4102444800000, 1,
                      'accepted', 1, ?, ?, 1)
            """,
            [
                instance.task_id,
                instance.lease_id,
                instance.record_id,
                instance.record_id,
                instance.fencing_epoch,
            ],
        )
        if include_attempt:
            connection.execute(
                """
                INSERT INTO task_attempts (
                    attempt_id, task_cid, attempt_number, owner_session_id,
                    fencing_token, fence_epoch, started_at, status, revision
                ) VALUES (?, ?, 1, ?, 1, ?, ?, 'running', 1)
                """,
                [
                    f"attempt:{instance.task_id}",
                    instance.task_id,
                    instance.record_id,
                    instance.fencing_epoch,
                    NOW,
                ],
            )


def test_repository_seals_typed_catalog_and_rejects_raw_sql(tmp_path: Path) -> None:
    _, client, repository, _, _, _ = _open_repository(tmp_path)
    templates = _casf_templates()
    try:
        assert len(templates) >= 26
        assert client.templates_sealed is True
        assert {item.name for item in templates} <= set(repository.statement_catalog)
        assert not hasattr(repository, "execute_sql")

        with pytest.raises(QuackClientSQLError, match="arbitrary SQL"):
            client.execute_sql("SELECT * FROM federations")
        with pytest.raises(QuackClientSQLError, match="catalog is sealed"):
            client.register_template(
                StatementTemplate(
                    name="casf_untrusted_query",
                    sql="SELECT 1",
                    parameter_names=(),
                    kind=StatementKind.QUERY,
                )
            )
    finally:
        client.close()


def test_every_casf_template_prepares_against_migration_two(tmp_path: Path) -> None:
    database = tmp_path / "control.duckdb"
    report = install_control_plane_schema(database, owner_id="owner:prepare")
    assert report.to_version == 2
    templates = _casf_templates()

    assert len(templates) >= 26
    assert len({item.name for item in templates}) == len(templates)
    with open_duckdb_connection(database) as connection:
        for ordinal, template in enumerate(templates):
            assert template.sql.count("?") == len(template.parameter_names)
            prepared_name = f"casf_template_probe_{ordinal}"
            connection.execute(f"PREPARE {prepared_name} AS {template.sql}")
            connection.execute(f"DEALLOCATE {prepared_name}")


def test_global_event_head_reconciles_preexisting_control_plane_events(
    tmp_path: Path,
) -> None:
    database, client, _repository, _binding, request, policy = _open_repository(
        tmp_path
    )
    client.close()
    with open_duckdb_connection(database) as connection:
        connection.execute(
            """
            INSERT INTO domain_events (
                event_id, stream_id, sequence, global_sequence,
                event_type, recorded_at, body_json
            ) VALUES ('event:legacy:94', 'legacy-materializer', 94, 94,
                      'TASK_CREATED', ?, '{}')
            """,
            [NOW],
        )
        head_count = connection.execute(
            "SELECT COUNT(*) FROM global_sequence_head"
        ).fetchone()
        assert head_count is not None and int(head_count[0]) == 0

    client, repository = _reopen_repository(database)
    try:
        AuthoritativeBudgetAuthority(
            repository,
            capacity=_BUDGET_CAPACITY,
            authority_id="authority:budget:legacy-watermark-test",
            now=lambda: datetime(2030, 1, 1, tzinfo=timezone.utc),
        ).reserve(request, policy)
    finally:
        client.close()

    with open_duckdb_connection(database) as connection:
        head = connection.execute(
            "SELECT current_sequence FROM global_sequence_head WHERE head_id = 'global'"
        ).fetchone()
        maximum = connection.execute(
            "SELECT MAX(global_sequence) FROM domain_events"
        ).fetchone()
        assert head is not None and int(head[0]) == 95
        assert maximum is not None and int(maximum[0]) == 95


def test_create_is_idempotent_and_commits_one_state_event_outbox(
    tmp_path: Path,
) -> None:
    notifications: list[int] = []
    database, client, repository, _, request, policy = _open_repository(
        tmp_path,
        event_notifier=notifications.append,
    )
    before = client.load_generation()
    try:
        expected_decision = _authorization_decision(
            request,
            policy,
            resolved_for(request.binding),
        )
        first = _create(repository, request=request, policy=policy)
        after_first = client.load_generation()
        replay = _create(repository, request=request, policy=policy)
        after_replay = client.load_generation()

        assert replay == first
        assert after_first.generation == before.generation
        # Budget reservation and federation creation are distinct, durable
        # authoritative transactions; each carries its own event/outbox.
        assert after_first.revision == before.revision + 2
        assert after_replay == after_first
        assert expected_decision.cid in first[1].evidence_refs
        assert expected_decision.authentication_evidence_cid in first[1].evidence_refs
        # Raw domain events are not a waiter source.  A later persist-first
        # routing commit is the only operation allowed to signal consumers.
        assert notifications == []
    finally:
        client.close()

    snapshot = _snapshot(
        database,
        (
            "federations",
            "federation_policies",
            "federation_budgets",
            "federation_authorization_decisions",
            "domain_events",
            "transactional_outbox",
            "idempotency_records",
        ),
    )
    assert snapshot == {
        "federations": 1,
        "federation_policies": 1,
        "federation_budgets": 1,
        "federation_authorization_decisions": 1,
        "domain_events": 2,
        "transactional_outbox": 2,
        "idempotency_records": 2,
        "store_revision": 2,
        "event_watermark": 2,
    }
    with open_duckdb_connection(database) as connection:
        row = connection.execute(
            """
            SELECT authorization_decision_id, request_cid, evidence_ref,
                   content_ref, body_json
            FROM federation_authorization_decisions
            """
        ).fetchone()
    assert row is not None
    assert tuple(str(row[index]) for index in range(4)) == (
        expected_decision.cid,
        request.cid,
        expected_decision.authentication_evidence_cid,
        expected_decision.cid,
    )
    assert json.loads(str(row[4])) == expected_decision.to_dict()
    assert "signature" not in str(row[4])
    assert "key_handle" not in str(row[4])


def test_persisted_budget_capacity_release_and_creation_consumption(
    tmp_path: Path,
) -> None:
    database, client, repository, _, request, policy = _open_repository(tmp_path)
    exact_capacity = {
        contracts.BudgetDimensionName.CPU_MILLIS: 100,
        contracts.BudgetDimensionName.INPUT_TOKENS: 100,
    }
    authority = AuthoritativeBudgetAuthority(
        repository,
        capacity=exact_capacity,
        authority_id="authority:budget:persistence-test",
        now=lambda: datetime(2030, 1, 1, tzinfo=timezone.utc),
    )
    second_request = replace(
        request,
        nonce="nonce:second",
        idempotency_key="idempotency:second",
    )
    first = authority.reserve(request, policy)
    assert authority.reserve(request, policy) == first
    before_conflict = client.load_generation()

    with pytest.raises(TransactionError, match="capacity exhausted"):
        authority.reserve(second_request, policy)
    assert client.load_generation() == before_conflict

    with pytest.raises(TransactionError, match="budget reservation is absent"):
        repository.release_federation_budget(
            first.record_id,
            tenant_id=first.binding.tenant_id,
            federation_id="federation:foreign",
            idempotency_key=first.idempotency_key,
            reason="test-release",
        )
    assert client.load_generation() == before_conflict

    authority.release(
        first,
        idempotency_key=first.idempotency_key,
        reason="test-release",
    )
    second = authority.reserve(second_request, policy)
    identity, _ = repository.create_federation(
        request=second_request,
        policy=policy,
        repositories=resolved_for(second_request.binding),
        budget_reservation=second,
        authentication_evidence_ref=(
            second_request.binding.authorization_evidence_ref
        ),
        authorization_decision=_authorization_decision(
            second_request,
            policy,
            resolved_for(second_request.binding),
        ),
    )
    assert identity.record_id == second.owner_id
    with pytest.raises(TransactionError, match="consumed budget reservation"):
        authority.release(
            second,
            idempotency_key=second.idempotency_key,
            reason="test-release",
        )
    client.close()

    with open_duckdb_connection(database) as connection:
        rows = connection.execute(
            """
            SELECT reservation_id, state, revision
            FROM federation_admission_budget_reservations
            ORDER BY reservation_id
            """
        ).fetchall()
    assert sorted((str(row[1]), int(row[2])) for row in rows) == [
        ("consumed", 2),
        ("released", 2),
    ]


def test_create_rejects_inconsistent_policy_repository_budget_and_evidence(
    tmp_path: Path,
) -> None:
    database, client, repository, binding, request, policy = _open_repository(tmp_path)
    mismatched_binding = replace(binding, objective_revision=2)
    mismatched_repository = replace(
        resolved_for(binding)[0],
        tree_id="tree:unadmitted",
    )
    mismatched_budget = replace(
        request.resource_budget,
        binding=replace(binding, budget_ref="budget:other"),
    )
    reservation = _unpersisted_reservation(request, policy)
    cases = (
        (
            replace(policy, binding=mismatched_binding),
            resolved_for(binding),
            request,
            binding.authorization_evidence_ref,
            "policy binding",
        ),
        (
            replace(policy, maximum_supervisors=1),
            resolved_for(binding),
            request,
            binding.authorization_evidence_ref,
            "supervisor count exceeds",
        ),
        (
            policy,
            (),
            request,
            binding.authorization_evidence_ref,
            "not every repository root",
        ),
        (
            policy,
            (mismatched_repository,),
            request,
            binding.authorization_evidence_ref,
            "repository trees differ",
        ),
        (
            policy,
            resolved_for(binding),
            replace(request, resource_budget=mismatched_budget),
            binding.authorization_evidence_ref,
            "budget binding differs",
        ),
        (
            policy,
            resolved_for(binding),
            request,
            "authentication:unadmitted",
            "authentication evidence differs",
        ),
    )
    try:
        for candidate_policy, repositories, candidate_request, evidence, match in cases:
            with pytest.raises(contracts.FederationAuthorityError, match=match):
                repository.create_federation(
                    request=candidate_request,
                    policy=candidate_policy,
                    repositories=repositories,
                    budget_reservation=reservation,
                    authentication_evidence_ref=evidence,
                    authorization_decision=_authorization_decision(
                        request,
                        policy,
                        resolved_for(binding),
                    ),
                )
            assert client.load_generation().revision == 0
    finally:
        client.close()

    snapshot = _snapshot(
        database,
        (
            "federations",
            "federation_policies",
            "federation_authorization_decisions",
            "domain_events",
            "idempotency_records",
        ),
    )
    assert snapshot["federations"] == 0
    assert snapshot["federation_policies"] == 0
    assert snapshot["federation_authorization_decisions"] == 0
    assert snapshot["domain_events"] == 0
    assert snapshot["idempotency_records"] == 0


@pytest.mark.parametrize(
    "failure_phase",
    ["after_state_before_event", "after_event_before_outbox"],
)
def test_create_failure_rolls_back_state_event_outbox_and_generation(
    tmp_path: Path,
    failure_phase: str,
) -> None:
    def fail(phase: str) -> None:
        if phase == failure_phase:
            raise RuntimeError(f"injected failure at {phase}")

    database, client, repository, _, request, policy = _open_repository(tmp_path)
    reservation = AuthoritativeBudgetAuthority(
        repository,
        capacity=_BUDGET_CAPACITY,
        authority_id="authority:budget:rollback-test",
        now=lambda: datetime(2030, 1, 1, tzinfo=timezone.utc),
    ).reserve(request, policy)
    repository._test_failure_hook = fail
    before = client.load_generation()
    try:
        with pytest.raises(RuntimeError, match=failure_phase):
            repository.create_federation(
                request=request,
                policy=policy,
                repositories=resolved_for(request.binding),
                budget_reservation=reservation,
                authentication_evidence_ref=(
                    request.binding.authorization_evidence_ref
                ),
                authorization_decision=_authorization_decision(
                    request,
                    policy,
                    resolved_for(request.binding),
                ),
            )
        assert client.load_generation() == before
    finally:
        client.close()

    snapshot = _snapshot(
        database,
        (
            "federations",
            "federation_policies",
            "federation_budgets",
            "federation_authorization_decisions",
            "domain_events",
            "domain_event_changed_facts",
            "transactional_outbox",
            "stream_sequence_heads",
            "idempotency_records",
            "federation_admission_budget_reservations",
        ),
    )
    assert snapshot == {
        "federations": 0,
        "federation_policies": 0,
        "federation_budgets": 0,
        "federation_authorization_decisions": 0,
        "domain_events": 1,
        "domain_event_changed_facts": 3,
        "transactional_outbox": 1,
        "stream_sequence_heads": 1,
        "idempotency_records": 1,
        "federation_admission_budget_reservations": 1,
        "store_revision": 1,
        "event_watermark": 1,
    }


def test_supervisor_ceiling_and_lifecycle_revision_fence(tmp_path: Path) -> None:
    database, client, repository, binding, request, policy = _open_repository(
        tmp_path,
        maximum_supervisors=1,
    )
    try:
        identity, _ = _create(repository, request=request, policy=policy)
        first, assignment = _supervisor(
            binding=binding,
            federation_id=identity.record_id,
        )
        assert (
            _register_supervisor(
                repository,
                first,
                assignment,
                idempotency_key="supervisor-register:one",
            )
            == first
        )

        second, second_assignment = _supervisor(
            binding=binding,
            federation_id=identity.record_id,
            suffix="two",
        )
        with pytest.raises(TransactionError, match="ceiling"):
            _register_supervisor(
                repository,
                second,
                second_assignment,
                idempotency_key="supervisor-register:two",
            )

        transitioned = repository.transition_supervisor(
            supervisor_id=first.record_id,
            tenant_id=binding.tenant_id,
            federation_id=identity.record_id,
            requested_state=contracts.FederationLifecycleState.ADMITTED,
            expected_revision=1,
            expected_fencing_epoch=1,
            active_effects=0,
            active_attempts=0,
            idempotency_key="supervisor-transition:admitted",
        )
        assert transitioned["state"] == contracts.FederationLifecycleState.ADMITTED.value
        assert transitioned["revision"] == 2

        with pytest.raises(TransactionError, match="fence is stale"):
            repository.transition_supervisor(
                supervisor_id=first.record_id,
                tenant_id=binding.tenant_id,
                federation_id=identity.record_id,
                requested_state=contracts.FederationLifecycleState.STARTING,
                expected_revision=2,
                expected_fencing_epoch=2,
                active_effects=0,
                active_attempts=0,
                idempotency_key="supervisor-transition:stale-fence",
            )
    finally:
        client.close()

    snapshot = _snapshot(
        database,
        ("supervisor_instances", "supervisor_assignments", "domain_events"),
    )
    assert snapshot["supervisor_instances"] == 1
    assert snapshot["supervisor_assignments"] == 1
    assert snapshot["domain_events"] == 4
    assert snapshot["store_revision"] == 4


def test_executable_supervisor_states_require_owner_attested_runtime_lease(
    tmp_path: Path,
) -> None:
    database, client, repository, binding, request, policy = _open_repository(tmp_path)
    try:
        identity, _ = _create(repository, request=request, policy=policy)
        supervisor, assignment = _supervisor(
            binding=binding,
            federation_id=identity.record_id,
        )
        _register_supervisor(
            repository,
            supervisor,
            assignment,
            idempotency_key="supervisor-register:runtime",
        )
        admitted = repository.transition_supervisor(
            supervisor_id=supervisor.record_id,
            tenant_id=binding.tenant_id,
            federation_id=identity.record_id,
            requested_state=contracts.FederationLifecycleState.ADMITTED,
            expected_revision=1,
            expected_fencing_epoch=1,
            active_effects=0,
            active_attempts=0,
            idempotency_key="supervisor-transition:runtime-admitted",
        )
        with pytest.raises(TransactionError, match="current runtime lease"):
            repository.transition_supervisor(
                supervisor_id=supervisor.record_id,
                tenant_id=binding.tenant_id,
                federation_id=identity.record_id,
                requested_state=contracts.FederationLifecycleState.STARTING,
                expected_revision=int(admitted["revision"]),
                expected_fencing_epoch=1,
                active_effects=0,
                active_attempts=0,
                idempotency_key="supervisor-transition:runtime-unleased",
            )

        runtime = repository.attest_supervisor_runtime(
            supervisor_id=supervisor.record_id,
            tenant_id=binding.tenant_id,
            federation_id=identity.record_id,
            expected_revision=int(admitted["revision"]),
            expected_fencing_epoch=1,
            idempotency_key="supervisor-runtime:attested",
        )
        assert runtime["process_birth_id"].startswith(
            "process-birth-attestation:"
        )
        starting = repository.transition_supervisor(
            supervisor_id=supervisor.record_id,
            tenant_id=binding.tenant_id,
            federation_id=identity.record_id,
            requested_state=contracts.FederationLifecycleState.STARTING,
            expected_revision=int(admitted["revision"]),
            expected_fencing_epoch=1,
            active_effects=0,
            active_attempts=0,
            idempotency_key="supervisor-transition:runtime-starting",
        )
        active = repository.transition_supervisor(
            supervisor_id=supervisor.record_id,
            tenant_id=binding.tenant_id,
            federation_id=identity.record_id,
            requested_state=contracts.FederationLifecycleState.ACTIVE,
            expected_revision=int(starting["revision"]),
            expected_fencing_epoch=1,
            active_effects=0,
            active_attempts=0,
            idempotency_key="supervisor-transition:runtime-active",
        )
        assert active["state"] == contracts.FederationLifecycleState.ACTIVE.value
        renewed = repository.attest_supervisor_runtime(
            supervisor_id=supervisor.record_id,
            tenant_id=binding.tenant_id,
            federation_id=identity.record_id,
            expected_revision=int(active["revision"]),
            expected_fencing_epoch=1,
            idempotency_key="supervisor-runtime:renewed",
        )
        assert renewed["revision"] == 2
        draining = repository.transition_supervisor(
            supervisor_id=supervisor.record_id,
            tenant_id=binding.tenant_id,
            federation_id=identity.record_id,
            requested_state=contracts.FederationLifecycleState.DRAINING,
            expected_revision=int(active["revision"]),
            expected_fencing_epoch=1,
            active_effects=0,
            active_attempts=0,
            idempotency_key="supervisor-transition:runtime-draining",
        )
        completed = repository.transition_supervisor(
            supervisor_id=supervisor.record_id,
            tenant_id=binding.tenant_id,
            federation_id=identity.record_id,
            requested_state=contracts.FederationLifecycleState.COMPLETED,
            expected_revision=int(draining["revision"]),
            expected_fencing_epoch=1,
            active_effects=0,
            active_attempts=0,
            idempotency_key="supervisor-transition:runtime-completed",
        )
        assert completed["state"] == (
            contracts.FederationLifecycleState.COMPLETED.value
        )
    finally:
        client.close()

    with open_duckdb_connection(database) as connection:
        runtime_row = connection.execute(
            """
            SELECT process_id, process_start_time_ticks, process_boot_id,
                   expires_at, revoked_at, status
            FROM supervisor_runtime_leases
            WHERE supervisor_id = ?
            ORDER BY revision DESC
            LIMIT 1
            """,
            [supervisor.record_id],
        ).fetchone()
        assert runtime_row is not None
        assert int(runtime_row[0]) > 0
        assert int(runtime_row[1]) > 0
        assert runtime_row[2]
        assert runtime_row[3]
        assert runtime_row[4] is None
        assert runtime_row[5] == "active"
        assert connection.execute(
            "SELECT COUNT(*) FROM supervisor_runtime_leases WHERE supervisor_id = ?",
            [supervisor.record_id],
        ).fetchone()[0] == 2


def test_supervisor_runtime_rejects_kernel_mismatch_expiry_and_unleased_completion(
    tmp_path: Path,
) -> None:
    database, client, repository, binding, request, policy = _open_repository(tmp_path)
    try:
        identity, _ = _create(repository, request=request, policy=policy)
        supervisor, assignment = _supervisor(
            binding=binding,
            federation_id=identity.record_id,
        )
        _register_supervisor(
            repository,
            supervisor,
            assignment,
            idempotency_key="supervisor-register:runtime-adversarial",
        )
        repository.transition_supervisor(
            supervisor_id=supervisor.record_id,
            tenant_id=binding.tenant_id,
            federation_id=identity.record_id,
            requested_state=contracts.FederationLifecycleState.ADMITTED,
            expected_revision=1,
            expected_fencing_epoch=1,
            active_effects=0,
            active_attempts=0,
            idempotency_key="supervisor-transition:runtime-adversarial-admitted",
        )
    finally:
        client.close()

    observed = current_process_birth()
    client = open_embedded_client(
        database,
        owner_id="owner:runtime-mismatch",
        seed_generation=False,
    )
    repository = FederationStateRepository(
        client,
        process_birth_factory=lambda: replace(
            observed,
            start_time_ticks=observed.start_time_ticks + 1,
        ),
        process_birth_reader=read_process_birth,
    )
    try:
        with pytest.raises(TransactionError, match="stale or mismatched"):
            repository.attest_supervisor_runtime(
                supervisor_id=supervisor.record_id,
                tenant_id=binding.tenant_id,
                federation_id=identity.record_id,
                expected_revision=2,
                expected_fencing_epoch=1,
                idempotency_key="supervisor-runtime:kernel-mismatch",
            )
    finally:
        client.close()

    client, repository = _reopen_repository(database)
    try:
        runtime = repository.attest_supervisor_runtime(
            supervisor_id=supervisor.record_id,
            tenant_id=binding.tenant_id,
            federation_id=identity.record_id,
            expected_revision=2,
            expected_fencing_epoch=1,
            idempotency_key="supervisor-runtime:before-expiry",
        )
        starting = repository.transition_supervisor(
            supervisor_id=supervisor.record_id,
            tenant_id=binding.tenant_id,
            federation_id=identity.record_id,
            requested_state=contracts.FederationLifecycleState.STARTING,
            expected_revision=2,
            expected_fencing_epoch=1,
            active_effects=0,
            active_attempts=0,
            idempotency_key="supervisor-transition:before-expiry",
        )
        assert runtime["runtime_lease_id"]
    finally:
        client.close()

    with open_duckdb_connection(database) as connection:
        connection.execute(
            """
            UPDATE supervisor_runtime_leases
            SET expires_at = '2000-01-01T00:00:00Z'
            WHERE supervisor_id = ?
            """,
            [supervisor.record_id],
        )

    client, repository = _reopen_repository(database)
    try:
        with pytest.raises(TransactionError, match="current runtime lease"):
            repository.transition_supervisor(
                supervisor_id=supervisor.record_id,
                tenant_id=binding.tenant_id,
                federation_id=identity.record_id,
                requested_state=contracts.FederationLifecycleState.ACTIVE,
                expected_revision=int(starting["revision"]),
                expected_fencing_epoch=1,
                active_effects=0,
                active_attempts=0,
                idempotency_key="supervisor-transition:expired-runtime",
            )
    finally:
        client.close()

    completion_path = tmp_path / "completion"
    completion_path.mkdir()
    second_database, client, repository, binding, request, policy = _open_repository(
        completion_path
    )
    del second_database
    try:
        second_identity, _ = _create(repository, request=request, policy=policy)
        second, second_assignment = _supervisor(
            binding=binding,
            federation_id=second_identity.record_id,
            suffix="unleased-completion",
        )
        _register_supervisor(
            repository,
            second,
            second_assignment,
            idempotency_key="supervisor-register:unleased-completion",
        )
        repository.transition_supervisor(
            supervisor_id=second.record_id,
            tenant_id=binding.tenant_id,
            federation_id=second_identity.record_id,
            requested_state=contracts.FederationLifecycleState.ADMITTED,
            expected_revision=1,
            expected_fencing_epoch=1,
            active_effects=0,
            active_attempts=0,
            idempotency_key="supervisor-transition:unleased-admitted",
        )
        draining = repository.transition_supervisor(
            supervisor_id=second.record_id,
            tenant_id=binding.tenant_id,
            federation_id=second_identity.record_id,
            requested_state=contracts.FederationLifecycleState.DRAINING,
            expected_revision=2,
            expected_fencing_epoch=1,
            active_effects=0,
            active_attempts=0,
            idempotency_key="supervisor-transition:unleased-draining",
        )
        with pytest.raises(TransactionError, match="current runtime lease"):
            repository.transition_supervisor(
                supervisor_id=second.record_id,
                tenant_id=binding.tenant_id,
                federation_id=second_identity.record_id,
                requested_state=contracts.FederationLifecycleState.COMPLETED,
                expected_revision=int(draining["revision"]),
                expected_fencing_epoch=1,
                active_effects=0,
                active_attempts=0,
                idempotency_key="supervisor-transition:unleased-completed",
            )
    finally:
        client.close()


def test_subagent_ceiling_and_stale_or_wrong_outcomes_are_rejected(
    tmp_path: Path,
) -> None:
    database, client, repository, binding, request, policy = _open_repository(
        tmp_path,
        maximum_subagents=1,
    )
    identity: contracts.FederationIdentity
    supervisor: contracts.SupervisorInstance
    first: contracts.SubagentInstance
    outcome: contracts.SubagentOutcome
    try:
        identity, _ = _create(repository, request=request, policy=policy)
        supervisor, assignment = _supervisor(
            binding=binding,
            federation_id=identity.record_id,
            task_refs=("task:one", "task:two"),
        )
        supervisor = replace(
            supervisor,
            state=contracts.FederationLifecycleState.ADMITTED.value,
        )
        _register_supervisor(
            repository,
            supervisor,
            assignment,
            idempotency_key="supervisor-register:subagent-parent",
        )
        subagent_binding = replace(binding, supervisor_population=1)
        first = _subagent(
            binding=subagent_binding,
            federation_id=identity.record_id,
            supervisor_id=supervisor.record_id,
        )
        assert _register_subagent(repository, first) == first
        assert (
            repository.get_subagent(
                first.record_id,
                tenant_id=binding.tenant_id,
                federation_id=identity.record_id,
                supervisor_id=supervisor.record_id,
            )
            == first
        )
        assert (
            repository.get_subagent(
                first.record_id,
                tenant_id="tenant:other",
                federation_id=identity.record_id,
                supervisor_id=supervisor.record_id,
            )
            is None
        )

        second = _subagent(
            binding=subagent_binding,
            federation_id=identity.record_id,
            supervisor_id=supervisor.record_id,
            suffix="two",
        )
        terminal = replace(
            second,
            record_id="subagent:terminal",
            state=contracts.FederationLifecycleState.STOPPED.value,
        )
        with pytest.raises(
            contracts.FederationAuthorityError,
            match="declared or admitted",
        ):
            _register_subagent(repository, terminal)
        with pytest.raises(TransactionError, match="ceiling"):
            _register_subagent(repository, second)

        outcome = _outcome(first)
        stale = replace(outcome, record_id="outcome:stale", fencing_epoch=2)
        wrong = replace(
            outcome,
            record_id="outcome:wrong-scope",
            supervisor_id="supervisor:other",
        )
        with pytest.raises(TransactionError, match="scope or fencing"):
            repository.record_subagent_outcome(stale)
        with pytest.raises(TransactionError, match="outcome subagent is absent"):
            repository.record_subagent_outcome(wrong)
        with pytest.raises(TransactionError, match="active admission"):
            repository.record_subagent_outcome(outcome)

        with pytest.raises(TransactionError, match="current task lease"):
            repository.reserve_subagent_slot(
                subagent_id=first.record_id,
                tenant_id=binding.tenant_id,
                federation_id=identity.record_id,
                supervisor_id=supervisor.record_id,
                expected_revision=first.revision,
                expected_fencing_epoch=first.fencing_epoch,
                idempotency_key="slot-reserve:unleased-outcome",
            )
    finally:
        client.close()

    _insert_active_task_execution(database, first)
    client, repository = _reopen_repository(database)
    try:
        reserved = repository.reserve_subagent_slot(
            subagent_id=first.record_id,
            tenant_id=binding.tenant_id,
            federation_id=identity.record_id,
            supervisor_id=supervisor.record_id,
            expected_revision=first.revision,
            expected_fencing_epoch=first.fencing_epoch,
            idempotency_key="slot-reserve:outcome-one",
        )
        active = contracts.SubagentInstance.from_dict(reserved["subagent"])
    finally:
        client.close()

    with open_duckdb_connection(database) as connection:
        connection.execute(
            """
            UPDATE subagent_capabilities
            SET freshness_state = 'stale'
            WHERE subagent_id = ?
            """,
            [active.record_id],
        )
    client, repository = _reopen_repository(database)
    try:
        with pytest.raises(TransactionError, match="capability authority"):
            repository.record_subagent_outcome(_outcome(active))
    finally:
        client.close()

    with open_duckdb_connection(database) as connection:
        connection.execute(
            """
            UPDATE subagent_capabilities
            SET freshness_state = 'current'
            WHERE subagent_id = ?
            """,
            [active.record_id],
        )
        connection.execute(
            "UPDATE leases SET expires_at_ms = 1 WHERE task_cid = ?",
            [active.task_id],
        )
    client, repository = _reopen_repository(database)
    try:
        with pytest.raises(TransactionError, match="task attempt, lease, fence, and slot"):
            repository.record_subagent_outcome(_outcome(active))
    finally:
        client.close()

    with open_duckdb_connection(database) as connection:
        connection.execute(
            "UPDATE leases SET expires_at_ms = 4102444800000 WHERE task_cid = ?",
            [active.task_id],
        )
        connection.execute(
            """
            UPDATE federation_policies
            SET status = 'revoked'
            WHERE tenant_id = ? AND federation_id = ? AND policy_id = ?
            """,
            [binding.tenant_id, identity.record_id, binding.policy_ref],
        )
    client, repository = _reopen_repository(database)
    try:
        with pytest.raises(TransactionError, match="policy is absent"):
            repository.record_subagent_outcome(_outcome(active))
    finally:
        client.close()

    with open_duckdb_connection(database) as connection:
        connection.execute(
            """
            UPDATE federation_policies
            SET status = 'admitted'
            WHERE tenant_id = ? AND federation_id = ? AND policy_id = ?
            """,
            [binding.tenant_id, identity.record_id, binding.policy_ref],
        )
    client, repository = _reopen_repository(database)
    try:
        repository.record_subagent_outcome(_outcome(active))
    finally:
        client.close()

    snapshot = _snapshot(
        database,
        ("subagent_instances", "subagent_outcomes", "domain_events"),
    )
    assert snapshot["subagent_instances"] == 1
    assert snapshot["subagent_outcomes"] == 1
    assert snapshot["domain_events"] == 6
    assert snapshot["store_revision"] == 6


def test_terminal_legacy_identity_still_consumes_registered_subagent_ceiling(
    tmp_path: Path,
) -> None:
    database, client, repository, binding, request, policy = _open_repository(
        tmp_path,
        maximum_subagents=1,
    )
    identity: contracts.FederationIdentity
    supervisor: contracts.SupervisorInstance
    try:
        identity, _ = _create(repository, request=request, policy=policy)
        supervisor, assignment = _supervisor(
            binding=binding,
            federation_id=identity.record_id,
            task_refs=("task:legacy-terminal", "task:replacement"),
        )
        supervisor = replace(
            supervisor,
            state=contracts.FederationLifecycleState.ADMITTED.value,
        )
        _register_supervisor(
            repository,
            supervisor,
            assignment,
            idempotency_key="supervisor-register:legacy-agent-parent",
        )
    finally:
        client.close()

    subagent_binding = replace(binding, supervisor_population=1)
    terminal = replace(
        _subagent(
            binding=subagent_binding,
            federation_id=identity.record_id,
            supervisor_id=supervisor.record_id,
            suffix="legacy-terminal",
        ),
        state=contracts.FederationLifecycleState.STOPPED.value,
    )
    _insert_subagent_fixture(database, terminal)

    client, repository = _reopen_repository(database)
    try:
        candidate = _subagent(
            binding=subagent_binding,
            federation_id=identity.record_id,
            supervisor_id=supervisor.record_id,
            suffix="replacement",
        )
        with pytest.raises(TransactionError, match="ceiling"):
            _register_subagent(repository, candidate)
    finally:
        client.close()

    snapshot = _snapshot(database, ("subagent_instances",))
    assert snapshot["subagent_instances"] == 1


def test_subscription_reads_events_and_acknowledgement_advances_cursor_once(
    tmp_path: Path,
) -> None:
    database, client, repository, binding, request, policy = _open_repository(tmp_path)
    identity: contracts.FederationIdentity
    subscription: EventSubscription
    cursor: ConsumerCursor
    event: DomainEvent
    try:
        identity, _ = _create(repository, request=request, policy=policy)
        subscription = EventSubscription(
            subscription_id="subscription:goal-events",
            tenant_id=binding.tenant_id,
            federation_id=identity.record_id,
            consumer_id="consumer:goal-events",
            revision=1,
            event_classes=(EventClass.GOAL_CHANGED,),
            selectors=(
                EventSelector(kind=SelectorKind.REPOSITORY, value=binding.repository_ids[0]),
            ),
            maximum_batch=8,
            maximum_pending=16,
            retry_budget=3,
            expires_at=EXPIRY,
            state=SubscriptionState.ACTIVE,
        )
        assert (
            repository.register_subscription(
                subscription,
                idempotency_key="subscription-register:goal-events",
            )
            == subscription
        )

        cursor = repository.get_cursor(
            tenant_id=subscription.tenant_id,
            federation_id=subscription.federation_id,
            consumer_id=subscription.consumer_id,
            subscription_id=subscription.subscription_id,
        )
        assert cursor.global_sequence == 0
        assert cursor.revision == 1
        raw_events = repository._events_for_loaded_subscription(
            subscription,
            after_cursor=cursor.global_sequence,
            maximum_events=8,
        )
        router = DurableEventRouter(repository)
        router.register(subscription)
        assert router.route(raw_events, now=NOW).routing.enqueued_deliveries == 1
        events = repository.events_for_subscription(
            consumer_id=subscription.consumer_id,
            subscription_id=subscription.subscription_id,
            subscription_revision=subscription.revision,
            after_cursor=cursor.global_sequence,
            maximum_events=8,
        )
        assert len(events) == 1
        event = events[0]
        assert event.event_type is EventClass.GOAL_CHANGED
        assert event.repository_id == binding.repository_ids[0]
        with pytest.raises(FederationRepositoryNotFound, match="cursor is absent"):
            repository.events_for_subscription(
                consumer_id="consumer:other",
                subscription_id=subscription.subscription_id,
                subscription_revision=subscription.revision,
                after_cursor=0,
                maximum_events=8,
            )
        delivery = DeliveryAttempt(
            attempt_id="delivery-attempt:goal-event",
            event_id=event.event_id,
            subscription_id=subscription.subscription_id,
            consumer_id=subscription.consumer_id,
            attempt_number=1,
            state=DeliveryState.DELIVERED,
            error_code="",
            recorded_at=NOW,
        )
        fence = client.load_generation().fence_epoch
        assert (
            repository.record_delivery_attempt(
                delivery,
                tenant_id=subscription.tenant_id,
                federation_id=subscription.federation_id,
                subscription_revision=subscription.revision,
                expected_fencing_epoch=fence,
                idempotency_key="delivery:goal-event",
            )
            == delivery
        )
        revision_after_delivery = client.load_generation().revision
        assert (
            repository.record_delivery_attempt(
                delivery,
                tenant_id=subscription.tenant_id,
                federation_id=subscription.federation_id,
                subscription_revision=subscription.revision,
                expected_fencing_epoch=fence,
                idempotency_key="delivery:goal-event",
            )
            == delivery
        )
        assert client.load_generation().revision == revision_after_delivery

        acknowledgement = EventAcknowledgement(
            acknowledgement_id="acknowledgement:goal-event",
            event_id=event.event_id,
            consumer_id=subscription.consumer_id,
            subscription_id=subscription.subscription_id,
            subscription_revision=subscription.revision,
            global_sequence=event.global_sequence,
            processed_effect_ref="effect:goal-event-processed",
            recorded_at=NOW,
        )
        advanced = repository.acknowledge_event(
            acknowledgement,
            tenant_id=subscription.tenant_id,
            federation_id=subscription.federation_id,
            delivery_attempt_id="delivery-attempt:goal-event",
            expected_cursor_revision=cursor.revision,
            expected_fencing_epoch=fence,
            idempotency_key="acknowledge:goal-event",
        )
        replay = repository.acknowledge_event(
            acknowledgement,
            tenant_id=subscription.tenant_id,
            federation_id=subscription.federation_id,
            delivery_attempt_id="delivery-attempt:goal-event",
            expected_cursor_revision=cursor.revision,
            expected_fencing_epoch=fence,
            idempotency_key="acknowledge:goal-event",
        )
        assert replay == advanced
        assert advanced.global_sequence == event.global_sequence
        assert advanced.revision == 2

        with pytest.raises(TransactionError, match="different command"):
            repository.acknowledge_event(
                acknowledgement,
                tenant_id=subscription.tenant_id,
                federation_id=subscription.federation_id,
                delivery_attempt_id="delivery-attempt:goal-event",
                expected_cursor_revision=cursor.revision,
                expected_fencing_epoch=fence,
                disposition="discarded",
                idempotency_key="acknowledge:goal-event",
            )

        stale_fence_ack = replace(
            acknowledgement,
            acknowledgement_id="acknowledgement:stale-fence",
        )
        with pytest.raises(TransactionError, match="fence is stale"):
            repository.acknowledge_event(
                stale_fence_ack,
                tenant_id=subscription.tenant_id,
                federation_id=subscription.federation_id,
                delivery_attempt_id="delivery-attempt:stale-fence",
                expected_cursor_revision=advanced.revision,
                expected_fencing_epoch=fence + 1,
                idempotency_key="acknowledge:stale-fence",
            )
        assert (
            repository.events_for_subscription(
                consumer_id=subscription.consumer_id,
                subscription_id=subscription.subscription_id,
                subscription_revision=subscription.revision,
                after_cursor=advanced.global_sequence,
                maximum_events=8,
            )
            == ()
        )
    finally:
        client.close()

    snapshot = _snapshot(
        database,
        (
            "event_subscriptions",
            "event_subscription_selectors",
            "consumer_cursors",
            "event_acknowledgements",
            "domain_events",
        ),
    )
    assert snapshot["event_subscriptions"] == 1
    assert snapshot["event_subscription_selectors"] == 1
    assert snapshot["consumer_cursors"] == 1
    assert snapshot["event_acknowledgements"] == 1
    assert snapshot["domain_events"] == 3
    assert snapshot["store_revision"] == 6
    with open_duckdb_connection(database) as connection:
        status = connection.execute(
            "SELECT status FROM delivery_attempts WHERE attempt_id = ?",
            ["delivery-attempt:goal-event"],
        ).fetchone()
    assert status is not None
    assert str(status[0]) == "acknowledged"


def test_selector_scan_pages_past_nonmatching_candidates_without_truncation(
    tmp_path: Path,
) -> None:
    database, client, repository, binding, request, policy = _open_repository(tmp_path)
    subscription: EventSubscription
    baseline: int
    identity: contracts.FederationIdentity
    try:
        identity, _ = _create(repository, request=request, policy=policy)
        subscription = EventSubscription(
            subscription_id="subscription:paged-selector",
            tenant_id=binding.tenant_id,
            federation_id=identity.record_id,
            consumer_id="consumer:paged-selector",
            revision=1,
            event_classes=(EventClass.GOAL_CHANGED,),
            selectors=(
                EventSelector(
                    kind=SelectorKind.REPOSITORY,
                    value=binding.repository_ids[0],
                ),
            ),
            maximum_batch=4,
            maximum_pending=16,
            retry_budget=3,
            expires_at=EXPIRY,
            state=SubscriptionState.ACTIVE,
        )
        repository.register_subscription(
            subscription,
            idempotency_key="subscription-register:paged-selector",
        )
        initial_raw = repository._events_for_loaded_subscription(
            subscription,
            after_cursor=0,
            maximum_events=1,
        )
        router = DurableEventRouter(repository)
        router.register(subscription)
        router.route(initial_raw, now=NOW)
        initial = repository.events_for_subscription(
            consumer_id=subscription.consumer_id,
            subscription_id=subscription.subscription_id,
            subscription_revision=subscription.revision,
            after_cursor=0,
            maximum_events=1,
        )
        assert len(initial) == 1
        baseline = initial[0].global_sequence
        fence = client.load_generation().fence_epoch
        delivery = DeliveryAttempt(
            attempt_id="delivery-attempt:paged-selector-baseline",
            event_id=initial[0].event_id,
            subscription_id=subscription.subscription_id,
            consumer_id=subscription.consumer_id,
            attempt_number=1,
            state=DeliveryState.DELIVERED,
            error_code="",
            recorded_at=NOW,
        )
        repository.record_delivery_attempt(
            delivery,
            tenant_id=subscription.tenant_id,
            federation_id=subscription.federation_id,
            subscription_revision=subscription.revision,
            expected_fencing_epoch=fence,
            idempotency_key="delivery:paged-selector-baseline",
        )
        repository.acknowledge_event(
            EventAcknowledgement(
                acknowledgement_id="acknowledgement:paged-selector-baseline",
                event_id=initial[0].event_id,
                consumer_id=subscription.consumer_id,
                subscription_id=subscription.subscription_id,
                subscription_revision=subscription.revision,
                global_sequence=baseline,
                processed_effect_ref="effect:paged-selector-baseline",
                recorded_at=NOW,
            ),
            tenant_id=subscription.tenant_id,
            federation_id=subscription.federation_id,
            delivery_attempt_id=delivery.attempt_id,
            expected_cursor_revision=1,
            expected_fencing_epoch=fence,
            idempotency_key="acknowledge:paged-selector-baseline",
        )
    finally:
        client.close()

    misses = tuple(f"repository:selector-miss-{index}" for index in range(300))
    fixtures = _insert_event_fixtures(
        database,
        binding=binding,
        federation_id=identity.record_id,
        repository_ids=misses + (binding.repository_ids[0],),
    )
    client, repository = _reopen_repository(database)
    try:
        raw_selected = repository._events_for_loaded_subscription(
            subscription,
            after_cursor=baseline,
            maximum_events=1,
        )
        assert raw_selected == (fixtures[-1],)
        router = DurableEventRouter(repository)
        router.register(subscription)
        router.route(raw_selected, now=NOW)
        selected = repository.events_for_subscription(
            consumer_id=subscription.consumer_id,
            subscription_id=subscription.subscription_id,
            subscription_revision=subscription.revision,
            after_cursor=baseline,
            maximum_events=1,
        )
        assert selected == (fixtures[-1],)
        assert selected[0].global_sequence - fixtures[0].global_sequence == 300
    finally:
        client.close()


def test_delivery_and_acknowledgement_reject_forged_scope_mismatch_and_skip(
    tmp_path: Path,
) -> None:
    database, client, repository, binding, request, policy = _open_repository(tmp_path)
    identity: contracts.FederationIdentity
    subscription: EventSubscription
    first_event: DomainEvent
    cursor: ConsumerCursor
    fence: int
    try:
        identity, _ = _create(repository, request=request, policy=policy)
        subscription = EventSubscription(
            subscription_id="subscription:owned-delivery",
            tenant_id=binding.tenant_id,
            federation_id=identity.record_id,
            consumer_id="consumer:owned-delivery",
            revision=1,
            event_classes=(EventClass.GOAL_CHANGED,),
            selectors=(
                EventSelector(
                    kind=SelectorKind.REPOSITORY,
                    value=binding.repository_ids[0],
                ),
            ),
            maximum_batch=8,
            maximum_pending=16,
            retry_budget=3,
            expires_at=EXPIRY,
            state=SubscriptionState.ACTIVE,
        )
        repository.register_subscription(
            subscription,
            idempotency_key="subscription-register:owned-delivery",
        )
        raw_events = repository._events_for_loaded_subscription(
            subscription,
            after_cursor=0,
            maximum_events=1,
        )
        router = DurableEventRouter(repository)
        router.register(subscription)
        router.route(raw_events, now=NOW)
        first_event = repository.events_for_subscription(
            consumer_id=subscription.consumer_id,
            subscription_id=subscription.subscription_id,
            subscription_revision=subscription.revision,
            after_cursor=0,
            maximum_events=1,
        )[0]
        cursor = repository.get_cursor(
            tenant_id=subscription.tenant_id,
            federation_id=subscription.federation_id,
            consumer_id=subscription.consumer_id,
            subscription_id=subscription.subscription_id,
        )
        fence = client.load_generation().fence_epoch

        forged_delivery = DeliveryAttempt(
            attempt_id="delivery-attempt:forged-consumer",
            event_id=first_event.event_id,
            subscription_id=subscription.subscription_id,
            consumer_id="consumer:forged",
            attempt_number=1,
            state=DeliveryState.DELIVERED,
            error_code="",
            recorded_at=NOW,
        )
        with pytest.raises(TransactionError, match="does not own"):
            repository.record_delivery_attempt(
                forged_delivery,
                tenant_id=subscription.tenant_id,
                federation_id=subscription.federation_id,
                subscription_revision=subscription.revision,
                expected_fencing_epoch=fence,
                idempotency_key="delivery:forged-consumer",
            )

        absent_delivery_ack = EventAcknowledgement(
            acknowledgement_id="acknowledgement:absent-delivery",
            event_id=first_event.event_id,
            consumer_id=subscription.consumer_id,
            subscription_id=subscription.subscription_id,
            subscription_revision=subscription.revision,
            global_sequence=first_event.global_sequence,
            processed_effect_ref="effect:must-not-commit",
            recorded_at=NOW,
        )
        with pytest.raises(TransactionError, match="no owned delivery"):
            repository.acknowledge_event(
                absent_delivery_ack,
                tenant_id=subscription.tenant_id,
                federation_id=subscription.federation_id,
                delivery_attempt_id="delivery-attempt:absent",
                expected_cursor_revision=cursor.revision,
                expected_fencing_epoch=fence,
                idempotency_key="acknowledge:absent-delivery",
            )
        assert (
            repository.get_cursor(
                tenant_id=subscription.tenant_id,
                federation_id=subscription.federation_id,
                consumer_id=subscription.consumer_id,
                subscription_id=subscription.subscription_id,
            )
            == cursor
        )
    finally:
        client.close()

    nonmatching_event = _load_event(database, EventClass.CAPABILITY_CHANGED)
    later_event = _insert_event_fixtures(
        database,
        binding=binding,
        federation_id=identity.record_id,
        repository_ids=(binding.repository_ids[0],),
    )[0]
    _insert_delivery_fixture(
        database,
        event=first_event,
        subscription=subscription,
        attempt_id="delivery-attempt:wrong-owner",
        consumer_id="consumer:other",
    )
    _insert_delivery_fixture(
        database,
        event=nonmatching_event,
        subscription=subscription,
        attempt_id="delivery-attempt:nonmatching-event",
    )
    _insert_delivery_fixture(
        database,
        event=later_event,
        subscription=subscription,
        attempt_id="delivery-attempt:later-event",
    )

    client, repository = _reopen_repository(database)
    try:
        wrong_owner_ack = replace(
            absent_delivery_ack,
            acknowledgement_id="acknowledgement:wrong-owner",
        )
        with pytest.raises(TransactionError, match="no owned delivery"):
            repository.acknowledge_event(
                wrong_owner_ack,
                tenant_id=subscription.tenant_id,
                federation_id=subscription.federation_id,
                delivery_attempt_id="delivery-attempt:wrong-owner",
                expected_cursor_revision=cursor.revision,
                expected_fencing_epoch=fence,
                idempotency_key="acknowledge:wrong-owner",
            )

        nonmatching_ack = replace(
            absent_delivery_ack,
            acknowledgement_id="acknowledgement:nonmatching-event",
            event_id=nonmatching_event.event_id,
            global_sequence=nonmatching_event.global_sequence,
        )
        with pytest.raises(TransactionError, match="does not match"):
            repository.acknowledge_event(
                nonmatching_ack,
                tenant_id=subscription.tenant_id,
                federation_id=subscription.federation_id,
                delivery_attempt_id="delivery-attempt:nonmatching-event",
                expected_cursor_revision=cursor.revision,
                expected_fencing_epoch=fence,
                idempotency_key="acknowledge:nonmatching-event",
            )

        later_ack = replace(
            absent_delivery_ack,
            acknowledgement_id="acknowledgement:later-event",
            event_id=later_event.event_id,
            global_sequence=later_event.global_sequence,
        )
        with pytest.raises(TransactionError, match="skip earlier eligible"):
            repository.acknowledge_event(
                later_ack,
                tenant_id=subscription.tenant_id,
                federation_id=subscription.federation_id,
                delivery_attempt_id="delivery-attempt:later-event",
                expected_cursor_revision=cursor.revision,
                expected_fencing_epoch=fence,
                idempotency_key="acknowledge:later-event",
            )
        assert (
            repository.get_cursor(
                tenant_id=subscription.tenant_id,
                federation_id=subscription.federation_id,
                consumer_id=subscription.consumer_id,
                subscription_id=subscription.subscription_id,
            )
            == cursor
        )
    finally:
        client.close()

    snapshot = _snapshot(
        database,
        ("delivery_attempts", "event_acknowledgements", "idempotency_records"),
    )
    assert snapshot["delivery_attempts"] == 3
    assert snapshot["event_acknowledgements"] == 0
    assert snapshot["idempotency_records"] == 4
    assert snapshot["store_revision"] == 4


def test_events_for_subscription_rejects_caller_cursor_jump(tmp_path: Path) -> None:
    _, client, repository, binding, request, policy = _open_repository(tmp_path)
    try:
        identity, _ = _create(repository, request=request, policy=policy)
        subscription = EventSubscription(
            subscription_id="subscription:cursor-jump",
            tenant_id=binding.tenant_id,
            federation_id=identity.record_id,
            consumer_id="consumer:cursor-jump",
            revision=1,
            event_classes=(EventClass.GOAL_CHANGED,),
            selectors=(),
            maximum_batch=4,
            maximum_pending=8,
            retry_budget=1,
            expires_at=EXPIRY,
            state=SubscriptionState.ACTIVE,
        )
        repository.register_subscription(
            subscription,
            idempotency_key="subscription-register:cursor-jump",
        )
        with pytest.raises(FederationRepositoryConflict, match="durable consumer cursor"):
            repository.events_for_subscription(
                consumer_id=subscription.consumer_id,
                subscription_id=subscription.subscription_id,
                subscription_revision=subscription.revision,
                after_cursor=1,
                maximum_events=1,
            )
    finally:
        client.close()


def test_caller_zero_cannot_complete_supervisor_with_authoritative_effect(
    tmp_path: Path,
) -> None:
    database, client, repository, binding, request, policy = _open_repository(tmp_path)
    identity: contracts.FederationIdentity
    supervisor: contracts.SupervisorInstance
    try:
        identity, _ = _create(repository, request=request, policy=policy)
        supervisor, assignment = _supervisor(
            binding=binding,
            federation_id=identity.record_id,
        )
        _register_supervisor(
            repository,
            supervisor,
            assignment,
            idempotency_key="supervisor-register:completion-guard",
        )
        repository.transition_supervisor(
            supervisor_id=supervisor.record_id,
            tenant_id=binding.tenant_id,
            federation_id=identity.record_id,
            requested_state=contracts.FederationLifecycleState.ADMITTED,
            expected_revision=1,
            expected_fencing_epoch=1,
            active_effects=0,
            active_attempts=0,
            idempotency_key="supervisor-transition:completion-admitted",
        )
        repository.transition_supervisor(
            supervisor_id=supervisor.record_id,
            tenant_id=binding.tenant_id,
            federation_id=identity.record_id,
            requested_state=contracts.FederationLifecycleState.DRAINING,
            expected_revision=2,
            expected_fencing_epoch=1,
            active_effects=0,
            active_attempts=0,
            idempotency_key="supervisor-transition:completion-draining",
        )
    finally:
        client.close()

    with open_duckdb_connection(database) as connection:
        connection.execute(
            """
            INSERT INTO federation_effect_reservations (
                effect_reservation_id, tenant_id, federation_id,
                supervisor_id, subagent_id, task_cid, attempt_id,
                effect_class, target_ref, lease_id, fencing_epoch,
                idempotency_key, state, reserved_at, expires_at
            ) VALUES (?, ?, ?, ?, '', ?, ?, ?, ?, ?, 1, ?, 'reserved', ?, ?)
            """,
            [
                "effect-reservation:completion-guard",
                binding.tenant_id,
                identity.record_id,
                supervisor.record_id,
                "task:completion-guard",
                "attempt:completion-guard",
                "effect.write",
                "target:completion-guard",
                "lease:completion-guard",
                "idempotency:completion-guard",
                NOW,
                EXPIRY,
            ],
        )

    client, repository = _reopen_repository(database)
    try:
        with pytest.raises(TransactionError, match="active effects or attempts"):
            repository.transition_supervisor(
                supervisor_id=supervisor.record_id,
                tenant_id=binding.tenant_id,
                federation_id=identity.record_id,
                requested_state=contracts.FederationLifecycleState.COMPLETED,
                expected_revision=3,
                expected_fencing_epoch=1,
                active_effects=0,
                active_attempts=0,
                idempotency_key="supervisor-transition:forged-zero-complete",
            )
    finally:
        client.close()


def test_independent_clients_share_one_authoritative_subagent_slot_ceiling(
    tmp_path: Path,
) -> None:
    database, client, repository, binding, request, policy = _open_repository(
        tmp_path,
        maximum_subagents=2,
        maximum_concurrent_subagents=1,
    )
    identity: contracts.FederationIdentity
    supervisor: contracts.SupervisorInstance
    first: contracts.SubagentInstance
    second: contracts.SubagentInstance
    try:
        identity, _ = _create(repository, request=request, policy=policy)
        supervisor, supervisor_assignment = _supervisor(
            binding=binding,
            federation_id=identity.record_id,
            task_refs=("task:slot-one", "task:slot-two"),
        )
        supervisor = replace(
            supervisor,
            state=contracts.FederationLifecycleState.ADMITTED.value,
        )
        _register_supervisor(
            repository,
            supervisor,
            supervisor_assignment,
            idempotency_key="supervisor-register:slot-parent",
        )
        agent_binding = replace(binding, supervisor_population=1)
        first = _subagent(
            binding=agent_binding,
            federation_id=identity.record_id,
            supervisor_id=supervisor.record_id,
            suffix="slot-one",
        )
        second = _subagent(
            binding=agent_binding,
            federation_id=identity.record_id,
            supervisor_id=supervisor.record_id,
            suffix="slot-two",
        )
        _register_subagent(repository, first)
        _register_subagent(repository, second)
    finally:
        client.close()

    _insert_active_task_execution(database, first, include_attempt=False)
    _insert_active_task_execution(database, second, include_attempt=False)
    client, repository = _reopen_repository(database)
    try:
        reserved = repository.reserve_subagent_slot(
            subagent_id=first.record_id,
            tenant_id=binding.tenant_id,
            federation_id=identity.record_id,
            supervisor_id=supervisor.record_id,
            expected_revision=1,
            expected_fencing_epoch=1,
            idempotency_key="slot-reserve:first",
        )
        assert reserved["slot_number"] == 1
    finally:
        client.close()

    client, repository = _reopen_repository(database)
    try:
        with pytest.raises(TransactionError, match="concurrent subagent ceiling"):
            repository.reserve_subagent_slot(
                subagent_id=second.record_id,
                tenant_id=binding.tenant_id,
                federation_id=identity.record_id,
                supervisor_id=supervisor.record_id,
                expected_revision=1,
                expected_fencing_epoch=1,
                idempotency_key="slot-reserve:second-while-full",
            )
        with pytest.raises(TransactionError, match="not admitted"):
            repository.reserve_subagent_slot(
                subagent_id=first.record_id,
                tenant_id=binding.tenant_id,
                federation_id=identity.record_id,
                supervisor_id=supervisor.record_id,
                expected_revision=2,
                expected_fencing_epoch=1,
                idempotency_key="slot-reserve:first-duplicate",
            )
        released = repository.release_subagent_slot(
            subagent_id=first.record_id,
            tenant_id=binding.tenant_id,
            federation_id=identity.record_id,
            supervisor_id=supervisor.record_id,
            expected_revision=2,
            expected_fencing_epoch=1,
            idempotency_key="slot-release:first",
        )
        assert released["slot_number"] == 1
        second_reserved = repository.reserve_subagent_slot(
            subagent_id=second.record_id,
            tenant_id=binding.tenant_id,
            federation_id=identity.record_id,
            supervisor_id=supervisor.record_id,
            expected_revision=1,
            expected_fencing_epoch=1,
            idempotency_key="slot-reserve:second-after-release",
        )
        assert second_reserved["slot_number"] == 1
    finally:
        client.close()

    snapshot = _snapshot(
        database,
        ("subagent_execution_slots", "subagent_slot_ledger"),
    )
    assert snapshot["subagent_execution_slots"] == 1
    assert snapshot["subagent_slot_ledger"] == 3


def test_subagent_slot_process_birth_is_owner_attested_and_cannot_be_supplied(
    tmp_path: Path,
) -> None:
    database, client, repository, binding, request, policy = _open_repository(tmp_path)
    try:
        identity, _ = _create(repository, request=request, policy=policy)
        supervisor, supervisor_assignment = _supervisor(
            binding=binding,
            federation_id=identity.record_id,
            task_refs=("task:owner-birth",),
        )
        supervisor = replace(
            supervisor,
            state=contracts.FederationLifecycleState.ADMITTED.value,
        )
        _register_supervisor(
            repository,
            supervisor,
            supervisor_assignment,
            idempotency_key="supervisor-register:owner-birth",
        )
        subagent = _subagent(
            binding=replace(binding, supervisor_population=1),
            federation_id=identity.record_id,
            supervisor_id=supervisor.record_id,
            suffix="owner-birth",
        )
        _register_subagent(repository, subagent)
    finally:
        client.close()

    _insert_active_task_execution(database, subagent, include_attempt=False)
    observed = current_process_birth()
    client = open_embedded_client(
        database,
        owner_id="owner:subagent-birth-mismatch",
        seed_generation=False,
    )
    repository = FederationStateRepository(
        client,
        process_birth_factory=lambda: replace(
            observed,
            start_time_ticks=observed.start_time_ticks + 1,
        ),
        process_birth_reader=read_process_birth,
    )
    try:
        with pytest.raises(TransactionError, match="stale or mismatched"):
            repository.reserve_subagent_slot(
                subagent_id=subagent.record_id,
                tenant_id=binding.tenant_id,
                federation_id=identity.record_id,
                supervisor_id=supervisor.record_id,
                expected_revision=subagent.revision,
                expected_fencing_epoch=subagent.fencing_epoch,
                idempotency_key="slot-reserve:owner-birth-mismatch",
            )
    finally:
        client.close()

    client, repository = _reopen_repository(database)
    try:
        with pytest.raises(TypeError, match="worker_process_birth_id"):
            repository.reserve_subagent_slot(
                subagent_id=subagent.record_id,
                tenant_id=binding.tenant_id,
                federation_id=identity.record_id,
                supervisor_id=supervisor.record_id,
                worker_process_birth_id="process-birth:caller-forgery",  # type: ignore[call-arg]
                expected_revision=subagent.revision,
                expected_fencing_epoch=subagent.fencing_epoch,
                idempotency_key="slot-reserve:caller-birth-forgery",
            )
        reserved = repository.reserve_subagent_slot(
            subagent_id=subagent.record_id,
            tenant_id=binding.tenant_id,
            federation_id=identity.record_id,
            supervisor_id=supervisor.record_id,
            expected_revision=subagent.revision,
            expected_fencing_epoch=subagent.fencing_epoch,
            idempotency_key="slot-reserve:owner-birth-current",
        )
        assert reserved["slot_number"] == 1
    finally:
        client.close()

    with open_duckdb_connection(database) as connection:
        row = connection.execute(
            """
            SELECT slots.worker_process_birth_id, births.process_id,
                   births.start_marker, births.host_identity_ref,
                   births.supervisor_id, births.subagent_id, births.status
            FROM subagent_execution_slots AS slots
            INNER JOIN process_births AS births
              ON births.process_birth_id = slots.worker_process_birth_id
            WHERE slots.tenant_id = ? AND slots.federation_id = ?
              AND slots.subagent_id = ? AND slots.state = 'active'
            """,
            [binding.tenant_id, identity.record_id, subagent.record_id],
        ).fetchone()
        assert row is not None
        assert row[0].startswith("process-birth-attestation:")
        assert int(row[1]) > 0
        assert int(row[2]) > 0
        assert row[3].startswith("host-boot:")
        assert row[4] == supervisor.record_id
        assert row[5] == subagent.record_id
        assert row[6] == "active"


def test_child_supervisor_cannot_escalate_parent_assignment_or_definition(
    tmp_path: Path,
) -> None:
    database, client, repository, binding, request, policy = _open_repository(
        tmp_path,
        maximum_supervisors=2,
    )
    del database
    policy = replace(
        policy,
        allowed_operations=("federation.create", "federation.start"),
    )
    try:
        identity, _ = _create(repository, request=request, policy=policy)
        parent, parent_assignment = _supervisor(
            binding=binding,
            federation_id=identity.record_id,
        )
        parent = replace(
            parent,
            state=contracts.FederationLifecycleState.ADMITTED.value,
        )
        _register_supervisor(
            repository,
            parent,
            parent_assignment,
            idempotency_key="supervisor-register:bounded-parent",
        )
        child_binding = replace(binding, supervisor_population=1)

        assignment_child = contracts.SupervisorInstance(
            record_id="supervisor:assignment-escalation",
            revision=1,
            binding=child_binding,
            state=contracts.FederationLifecycleState.DECLARED.value,
            federation_id=identity.record_id,
            parent_supervisor_id=parent.record_id,
            role=contracts.SupervisorRole.REPOSITORY,
            lease_id="lease:supervisor:assignment-escalation",
            fencing_epoch=1,
        )
        assignment = contracts.SupervisorAssignment(
            record_id="assignment:supervisor:assignment-escalation",
            revision=1,
            binding=child_binding,
            subject_id=assignment_child.record_id,
            repository_ids=child_binding.repository_ids,
            goal_refs=(child_binding.objective_ref,),
            task_refs=(),
            allowed_task_families=("verification",),
            fencing_epoch=1,
        )
        definition, capabilities = _definition(
            contracts.SupervisorDefinition,
            contracts.SupervisorCapability,
            binding=child_binding,
            suffix="assignment-escalation",
        )
        with pytest.raises(TransactionError, match="task families exceed"):
            repository.register_supervisor(
                assignment_child,
                assignment,
                definition=definition,  # type: ignore[arg-type]
                capabilities=capabilities,  # type: ignore[arg-type]
                idempotency_key="supervisor-register:assignment-escalation",
            )

        definition_child = replace(
            assignment_child,
            record_id="supervisor:definition-escalation",
            lease_id="lease:supervisor:definition-escalation",
        )
        bounded_assignment = replace(
            assignment,
            record_id="assignment:supervisor:definition-escalation",
            subject_id=definition_child.record_id,
            allowed_task_families=("implementation",),
        )
        child_definition, child_capabilities = _definition(
            contracts.SupervisorDefinition,
            contracts.SupervisorCapability,
            binding=child_binding,
            suffix="definition-escalation",
        )
        child_definition = replace(
            child_definition,
            allowed_operations=("federation.start",),
        )
        child_capabilities = tuple(
            replace(item, allowed_operations=("federation.start",))
            for item in child_capabilities
        )
        with pytest.raises(TransactionError, match="operations exceed parent"):
            repository.register_supervisor(
                definition_child,
                bounded_assignment,
                definition=child_definition,  # type: ignore[arg-type]
                capabilities=child_capabilities,  # type: ignore[arg-type]
                idempotency_key="supervisor-register:definition-escalation",
            )
    finally:
        client.close()


def test_subagent_cannot_escape_parent_task_assignment(
    tmp_path: Path,
) -> None:
    _, client, repository, binding, request, policy = _open_repository(tmp_path)
    try:
        identity, _ = _create(repository, request=request, policy=policy)
        supervisor, assignment = _supervisor(
            binding=binding,
            federation_id=identity.record_id,
            task_refs=("task:delegated",),
        )
        supervisor = replace(
            supervisor,
            state=contracts.FederationLifecycleState.ADMITTED.value,
        )
        _register_supervisor(
            repository,
            supervisor,
            assignment,
            idempotency_key="supervisor-register:subagent-boundary",
        )
        escaped = _subagent(
            binding=replace(binding, supervisor_population=1),
            federation_id=identity.record_id,
            supervisor_id=supervisor.record_id,
            suffix="not-delegated",
        )
        with pytest.raises(TransactionError, match="tasks exceed parent"):
            _register_subagent(repository, escaped)
    finally:
        client.close()


def test_active_subagent_slot_prevents_forged_zero_supervisor_completion(
    tmp_path: Path,
) -> None:
    database, client, repository, binding, request, policy = _open_repository(
        tmp_path,
        maximum_subagents=1,
        maximum_concurrent_subagents=1,
    )
    identity: contracts.FederationIdentity
    supervisor: contracts.SupervisorInstance
    subagent: contracts.SubagentInstance
    try:
        identity, _ = _create(repository, request=request, policy=policy)
        supervisor, assignment = _supervisor(
            binding=binding,
            federation_id=identity.record_id,
            task_refs=("task:active-slot-completion",),
        )
        supervisor = replace(
            supervisor,
            state=contracts.FederationLifecycleState.ADMITTED.value,
        )
        _register_supervisor(
            repository,
            supervisor,
            assignment,
            idempotency_key="supervisor-register:active-slot-completion",
        )
        subagent = _subagent(
            binding=replace(binding, supervisor_population=1),
            federation_id=identity.record_id,
            supervisor_id=supervisor.record_id,
            suffix="active-slot-completion",
        )
        _register_subagent(repository, subagent)
    finally:
        client.close()

    _insert_active_task_execution(database, subagent, include_attempt=False)
    client, repository = _reopen_repository(database)
    try:
        repository.reserve_subagent_slot(
            subagent_id=subagent.record_id,
            tenant_id=binding.tenant_id,
            federation_id=identity.record_id,
            supervisor_id=supervisor.record_id,
            expected_revision=subagent.revision,
            expected_fencing_epoch=subagent.fencing_epoch,
            idempotency_key="slot-reserve:active-slot-completion",
        )
        draining = repository.transition_supervisor(
            supervisor_id=supervisor.record_id,
            tenant_id=binding.tenant_id,
            federation_id=identity.record_id,
            requested_state=contracts.FederationLifecycleState.DRAINING,
            expected_revision=supervisor.revision,
            expected_fencing_epoch=supervisor.fencing_epoch,
            active_effects=0,
            active_attempts=0,
            idempotency_key="supervisor-transition:active-slot-draining",
        )
        with pytest.raises(TransactionError, match="active effects or attempts"):
            repository.transition_supervisor(
                supervisor_id=supervisor.record_id,
                tenant_id=binding.tenant_id,
                federation_id=identity.record_id,
                requested_state=contracts.FederationLifecycleState.COMPLETED,
                expected_revision=int(draining["revision"]),
                expected_fencing_epoch=supervisor.fencing_epoch,
                active_effects=0,
                active_attempts=0,
                idempotency_key="supervisor-transition:forged-slot-completion",
            )
    finally:
        client.close()
