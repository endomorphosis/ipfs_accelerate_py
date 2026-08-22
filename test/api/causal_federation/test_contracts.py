from __future__ import annotations

import math
from dataclasses import FrozenInstanceError, fields, replace
from pathlib import Path
from typing import Any

import pytest
from ipfs_accelerate_py.agent_supervisor.federation import contracts
from ipfs_accelerate_py.agent_supervisor.federation.lifecycle import (
    admits_new_work,
    assert_transition,
    is_terminal,
    legal_transitions,
)

from ipfs_accelerate_py import agent_supervisor
from ipfs_accelerate_py.agent_supervisor import federation

NOW = "2030-01-01T00:00:00Z"
EXPIRY = "2099-01-01T00:00:00Z"


def sample_binding(**overrides: Any) -> contracts.FederationBinding:
    values: dict[str, Any] = {
        "tenant_id": "tenant:test",
        "repository_ids": ("repo:test",),
        "repository_tree_ids": ("tree:test",),
        "program_id": contracts.PROGRAM_ID,
        "objective_ref": contracts.ROOT_OBJECTIVE,
        "objective_revision": 1,
        "policy_ref": "policy:test",
        "policy_revision": 1,
        "operation_catalog_ref": "operations:test",
        "control_plane_generation": 1,
        "causal_graph_revision": 1,
        "semantic_state_roots": ("semantic:test",),
        "supervisor_population": 1,
        "budget_ref": "budget:federation",
        "expires_at": EXPIRY,
        "issuer": "did:test:issuer",
        "authorization_evidence_ref": "authz:test",
    }
    values.update(overrides)
    return contracts.FederationBinding(**values)


def sample_dimension(
    name: contracts.BudgetDimensionName = contracts.BudgetDimensionName.INPUT_TOKENS,
) -> contracts.BudgetDimension:
    return contracts.BudgetDimension(
        name=name,
        ceiling=100,
        reserved=80,
        consumed=20,
    )


def sample_budget(
    contract_type: type[contracts.BoundBudget],
    *,
    binding: contracts.FederationBinding | None = None,
    record_id: str | None = None,
) -> contracts.BoundBudget:
    dimension_name = (
        contracts.BudgetDimensionName.CPU_MILLIS
        if issubclass(contract_type, contracts.ResourceBudget)
        and not issubclass(contract_type, contracts.TokenBudget)
        else contracts.BudgetDimensionName.INPUT_TOKENS
    )
    return contract_type(
        record_id=record_id or f"{contract_type.__name__.lower()}:test",
        revision=1,
        binding=binding or sample_binding(),
        parent_budget_id="",
        owner_id="federation:test",
        dimensions=(sample_dimension(dimension_name),),
        status="reserved",
    )


def sample_contract(
    contract_type: type[contracts.ClosedContract],
) -> contracts.ClosedContract:
    binding = sample_binding()
    fixed_point = issubclass(contract_type, contracts.FederationFixedPoint)
    subagent_outcome = issubclass(contract_type, contracts.SubagentOutcome)
    receipt_outcomes: dict[type[contracts.ClosedContract], str] = {
        contracts.SupervisorHealth: "healthy",
        contracts.SupervisorCheckpoint: "checkpointed",
        contracts.SupervisorReceipt: "accepted",
        contracts.ShardRebalanceReceipt: "rebalanced",
        contracts.FederationCommandResult: "applied",
        contracts.InterventionTest: "matched",
    }
    values: dict[str, Any] = {
        "record_id": f"{contract_type.__name__.lower()}:test",
        "revision": 1,
        "binding": binding,
        "state": contracts.FederationLifecycleState.IDLE.value,
        "name": "sample definition",
        "capabilities": ("capability:test",),
        "allowed_operations": (contracts.FederationOperation.CREATE.value,),
        "effect_ceiling": "effect.read",
        "risk_ceiling": "risk.low",
        "resource_budget_ref": "budget:resource",
        "token_budget_ref": "budget:token",
        "subject_id": "subject:test",
        "repository_ids": binding.repository_ids,
        "goal_refs": ("goal:test",),
        "task_refs": ("task:test",),
        "allowed_task_families": ("task.family",),
        "fencing_epoch": 1,
        "outcome": (
            "fixed_point"
            if fixed_point
            else "succeeded"
            if subagent_outcome
            else receipt_outcomes.get(contract_type, "accepted")
        ),
        "evidence_refs": ("evidence:test",),
        "recorded_at": NOW,
        "parent_budget_id": "",
        "owner_id": "federation:test",
        "dimensions": (sample_dimension(),),
        "status": "reserved",
        "request_cid": "request:test",
        "issued_at": NOW,
        "authorization_evidence_ref": binding.authorization_evidence_ref,
        "caller_did": "did:test:caller",
        "delegation_chain": (),
        "delegation_chain_cid": "delegation-chain:test",
        "audience": "agent-supervisor:test",
        "program_id": contracts.PROGRAM_ID,
        "repository_roots": binding.repository_ids,
        "objective_ref": binding.objective_ref,
        "requested_supervisor_profile": "profile:test",
        "maximum_supervisors": 4,
        "maximum_subagents": 16,
        "resource_budget": sample_budget(
            contracts.ResourceBudget,
            binding=binding,
            record_id="budget:resource",
        ),
        "token_budget": sample_budget(
            contracts.TokenBudget,
            binding=binding,
            record_id="budget:token",
        ),
        "effect_scope": ("effect.read",),
        "policy_ref": binding.policy_ref,
        "policy_id": binding.policy_ref,
        "policy_revision": binding.policy_revision,
        "expiry": binding.expires_at,
        "expires_at": binding.expires_at,
        "nonce": "nonce:test",
        "idempotency_key": "idempotency:test",
        "allowed_callers": ("did:test:caller",),
        "allowed_audiences": ("agent-supervisor:test",),
        "allowed_effects": ("effect.read",),
        "maximum_concurrent_subagents": 8,
        "conservative_abstraction_scheduling": False,
        "supervisor_definition_refs": ("supervisor-definition:test",),
        "federation_id": "federation:test",
        "parent_supervisor_id": "",
        "role": contracts.SupervisorRole.COORDINATOR,
        "lease_id": "lease:test",
        "supervisor_id": "supervisor:test",
        "subagent_id": "subagent:test",
        "task_id": "task:test",
        "symbol_refs": ("symbol:test",),
        "effect_classes": ("read_only",),
        "operation": contracts.FederationOperation.CREATE,
        "resolved_scope_cid": "resolved-scope:test",
        "verdict": contracts.FederationAuthorizationVerdict.ADMITTED,
        "authentication_evidence_cid": "authentication-evidence:test",
        "decided_at": NOW,
        "target_id": "federation:test",
        "expected_generation": 1,
        "expected_revision": 1,
        "expected_fencing_epoch": 1,
        "dry_run": True,
        "expected_effects": ("effect.read",),
        "command_id": "command:test",
        "result_ref": "result:test",
        "event_watermark": 7,
        "task_population_ref": "tasks:test",
        "claim_population_ref": "claims:test",
        "merge_state_ref": "merge:test",
        "proof_state_ref": "proof:test",
        "semantic_roots": binding.semantic_state_roots,
        "causal_frontier_ref": "frontier:test",
        "world_snapshot_ref": "snapshot:test",
        "outstanding_required_work": 0,
        "blocker_id": "blocker:test",
        "capability": "capability:test",
        "code": contracts.CapabilityBlockerCode.MISSING,
        "authority": "authority:test",
        "reason": "required capability is unavailable",
        "independent_work_may_continue": True,
        "observed_at": NOW,
        "level": contracts.CausalLevel.L1_CODE_ARTIFACT,
        "node_type": "symbol",
        "subject_ref": "symbol:test",
        "evidence_kind": contracts.CausalEvidenceKind.EXACT_STATIC_DEPENDENCY,
        "evidence_ref": "evidence:test",
        "authoritative": True,
        "source_node_id": "node:source",
        "target_node_id": "node:target",
        "edge_kind": contracts.CausalEdgeKind.DEPENDS_ON,
        "nomination_only": False,
        "low_level_model_ref": "model:low",
        "high_level_model_ref": "model:high",
        "low_level_variables": ("variable:low",),
        "high_level_variables": ("variable:high",),
        "abstraction_function_ref": "abstraction:function",
        "intervention_mapping_ref": "intervention:mapping",
        "admitted_domain_refs": ("domain:admitted",),
        "excluded_domain_refs": (),
        "validation_evidence_refs": ("evidence:validation",),
        "faithfulness_status": contracts.AbstractionFaithfulness.EXACT,
        "policy_admitted": True,
        "abstraction_map_id": "abstraction:test",
        "low_level_intervention_ref": "intervention:low",
        "low_level_outcome_ref": "outcome:low",
        "abstracted_outcome_ref": "outcome:abstracted",
        "high_level_intervention_ref": "intervention:high",
        "high_level_outcome_ref": "outcome:high",
        "mismatch_ref": "",
        "event_id": "event:test",
        "node_id": "node:test",
        "disposition": contracts.FrontierDisposition.MUST_WAKE,
    }
    kwargs = {field.name: values[field.name] for field in fields(contract_type)}
    if contract_type is contracts.FederationAuthorizationDecision:
        kwargs.update(
            {
                "delegation_chain_cid": "delegation-chain:test",
                "resolved_scope_cid": "resolved-scope:test",
                "verdict": contracts.FederationAuthorizationVerdict.ADMITTED,
                "reason": (
                    contracts.FederationAuthorizationReason.AUTHENTICATED_DELEGATED_POLICY_ADMITTED
                ),
                "authentication_evidence_cid": "authentication-evidence:test",
                "decided_at": NOW,
            }
        )
    return contract_type(**kwargs)


CATALOG_CASES = tuple(contracts.contract_catalog().items())


@pytest.mark.parametrize(
    ("schema", "contract_type"),
    CATALOG_CASES,
    ids=[contract_type.__name__ for _, contract_type in CATALOG_CASES],
)
def test_every_named_contract_round_trips(
    schema: str,
    contract_type: type[contracts.ClosedContract],
) -> None:
    original = sample_contract(contract_type)

    encoded = original.to_dict()
    decoded = contract_type.from_dict(encoded)

    assert encoded["schema"] == schema
    assert decoded == original
    assert decoded.cid == original.cid


@pytest.mark.parametrize(
    "contract_type",
    [contract_type for _, contract_type in CATALOG_CASES],
    ids=[contract_type.__name__ for _, contract_type in CATALOG_CASES],
)
def test_every_named_contract_rejects_unknown_normative_fields(
    contract_type: type[contracts.ClosedContract],
) -> None:
    payload = sample_contract(contract_type).to_dict()
    payload["model_policy_override"] = True

    with pytest.raises(contracts.UnknownNormativeFieldError):
        contract_type.from_dict(payload)


def test_contracts_are_immutable_and_content_addressed() -> None:
    identity = sample_contract(contracts.FederationIdentity)

    with pytest.raises(FrozenInstanceError):
        identity.revision = 2  # type: ignore[misc]

    assert identity.cid == sample_contract(contracts.FederationIdentity).cid


@pytest.mark.parametrize(
    "contract_type",
    [
        contracts.SupervisorHealth,
        contracts.SupervisorCheckpoint,
        contracts.SupervisorReceipt,
        contracts.ShardRebalanceReceipt,
        contracts.FederationCommandResult,
        contracts.InterventionTest,
    ],
)
def test_generic_receipts_reject_model_authored_completion_and_empty_evidence(
    contract_type: type[contracts.BoundReceipt],
) -> None:
    receipt = sample_contract(contract_type)
    assert isinstance(receipt, contracts.BoundReceipt)

    with pytest.raises(contracts.FederationAuthorityError):
        replace(receipt, outcome="completed")
    with pytest.raises(contracts.FederationContractError):
        replace(receipt, evidence_refs=())


@pytest.mark.parametrize("number", [math.nan, math.inf, -math.inf])
def test_budget_rejects_nonfinite_values(number: float) -> None:
    with pytest.raises(contracts.FederationBoundsError):
        contracts.BudgetDimension(
            name=contracts.BudgetDimensionName.CPU_MILLIS,
            ceiling=number,
            reserved=0,
            consumed=0,
        )


def test_budget_enforces_consumed_reserved_ceiling_order() -> None:
    with pytest.raises(contracts.FederationBoundsError):
        contracts.BudgetDimension(
            name=contracts.BudgetDimensionName.CPU_MILLIS,
            ceiling=10,
            reserved=8,
            consumed=9,
        )


@pytest.mark.parametrize("fraction", [10.5, 4.5, 1.5])
def test_budget_rejects_fractional_base_units(fraction: float) -> None:
    with pytest.raises(contracts.FederationContractError):
        contracts.BudgetDimension(
            name=contracts.BudgetDimensionName.PROVIDER_SPEND_MICROS,
            ceiling=fraction,
            reserved=0,
            consumed=0,
        )


def test_integer_budget_dimension_has_a_stable_content_identity() -> None:
    dimension = sample_dimension(contracts.BudgetDimensionName.PROVIDER_SPEND_MICROS)

    assert (
        dimension.cid == sample_dimension(contracts.BudgetDimensionName.PROVIDER_SPEND_MICROS).cid
    )


def test_budget_reservation_is_typed_request_scoped_and_expiring() -> None:
    reservation = sample_contract(contracts.BudgetReservation)
    assert isinstance(reservation, contracts.BudgetReservation)
    assert reservation.request_cid
    assert reservation.idempotency_key
    assert reservation.resource_budget_ref
    assert reservation.token_budget_ref

    with pytest.raises(contracts.FederationAuthorityError):
        replace(reservation, expires_at="2098-01-01T00:00:00Z")
    with pytest.raises(contracts.FederationContractError):
        replace(reservation, issued_at=reservation.expires_at)


def test_request_population_bounds_fail_closed() -> None:
    request = sample_contract(contracts.FederationRequest)
    assert isinstance(request, contracts.FederationRequest)

    with pytest.raises(contracts.FederationBoundsError):
        replace(request, maximum_supervisors=contracts.MAX_SUPERVISORS + 1)
    with pytest.raises(contracts.FederationBoundsError):
        replace(request, maximum_subagents=contracts.MAX_SUBAGENTS + 1)


def test_authorization_decision_is_closed_redacted_and_create_only() -> None:
    decision = sample_contract(contracts.FederationAuthorizationDecision)
    assert isinstance(decision, contracts.FederationAuthorizationDecision)
    assert decision.verdict is contracts.FederationAuthorizationVerdict.ADMITTED
    assert decision.authentication_evidence_cid != "authz:test"
    assert "signature" not in decision.to_dict()
    assert "key_handle" not in decision.to_dict()

    with pytest.raises(contracts.FederationAuthorityError, match="federation.create"):
        replace(decision, operation=contracts.FederationOperation.START)


@pytest.mark.parametrize(
    "repository_id", ["/tmp/control.duckdb", "~/control.duckdb", "repo/../secret"]
)
def test_binding_rejects_arbitrary_repository_paths(repository_id: str) -> None:
    with pytest.raises(contracts.FederationContractError):
        sample_binding(repository_ids=(repository_id,))


def test_raw_secret_material_is_rejected() -> None:
    with pytest.raises(contracts.FederationSecretError):
        contracts.CapabilityBlocker(
            blocker_id="blocker:test",
            capability="capability:test",
            code=contracts.CapabilityBlockerCode.MISSING,
            authority="authority:test",
            reason="Bearer abcdefghijklmnop",
            independent_work_may_continue=True,
            observed_at=NOW,
            evidence_refs=(),
        )


def test_executable_callback_is_rejected_by_wire_decoder() -> None:
    payload = sample_contract(contracts.FederationRequest).to_dict()
    payload["caller_did"] = lambda: "did:test:forged"

    with pytest.raises(contracts.FederationContractError):
        contracts.FederationRequest.from_dict(payload)


def test_raw_sql_cannot_be_a_federation_operation() -> None:
    payload = sample_contract(contracts.FederationCommand).to_dict()
    payload["operation"] = "DROP TABLE control"

    with pytest.raises(ValueError):
        contracts.FederationCommand.from_dict(payload)


def test_raw_sql_cannot_hide_in_expected_effects_or_shard_effect_classes() -> None:
    command = sample_contract(contracts.FederationCommand)
    shard = sample_contract(contracts.ShardBoundary)
    assert isinstance(command, contracts.FederationCommand)
    assert isinstance(shard, contracts.ShardBoundary)

    with pytest.raises(contracts.FederationContractError):
        replace(command, expected_effects=("DROP TABLE control",))
    with pytest.raises(contracts.FederationContractError):
        replace(shard, effect_classes=("DROP TABLE control",))


@pytest.mark.parametrize(
    ("current", "requested"),
    [
        ("DECLARED", "ACTIVE"),
        ("DRAINING", "ACTIVE"),
        ("QUARANTINED", "ACTIVE"),
        ("COMPLETED", "ACTIVE"),
        ("STOPPED", "STARTING"),
    ],
)
def test_lifecycle_illegal_transitions_fail_closed(
    current: str,
    requested: str,
) -> None:
    with pytest.raises(contracts.FederationAuthorityError):
        assert_transition(current, requested)


@pytest.mark.parametrize(
    ("active_effects", "active_attempts"),
    [(1, 0), (0, 1), (2, 3)],
)
def test_lifecycle_cannot_complete_with_active_work(
    active_effects: int,
    active_attempts: int,
) -> None:
    with pytest.raises(contracts.FederationAuthorityError):
        assert_transition(
            "DRAINING",
            "COMPLETED",
            active_effects=active_effects,
            active_attempts=active_attempts,
        )


@pytest.mark.parametrize("current", ["IDLE", "ACTIVE", "PAUSED", "RECOVERING"])
def test_lifecycle_completion_requires_draining_even_when_work_is_quiet(
    current: str,
) -> None:
    with pytest.raises(contracts.FederationAuthorityError):
        assert_transition(current, "COMPLETED", active_effects=0, active_attempts=0)

    assert contracts.FederationLifecycleState.COMPLETED not in legal_transitions(current)


def test_drained_lifecycle_can_complete_only_after_all_work_settles() -> None:
    assert (
        assert_transition("DRAINING", "COMPLETED", active_effects=0, active_attempts=0)
        is contracts.FederationLifecycleState.COMPLETED
    )


def test_lifecycle_admission_and_terminal_semantics_are_closed() -> None:
    assert assert_transition("DECLARED", "ADMITTED") is contracts.FederationLifecycleState.ADMITTED
    assert contracts.FederationLifecycleState.ACTIVE in legal_transitions("IDLE")
    assert admits_new_work("IDLE")
    assert admits_new_work("ACTIVE")
    assert not admits_new_work("DECLARED")
    assert not admits_new_work("DRAINING")
    assert is_terminal("COMPLETED")
    assert is_terminal("FAILED")
    assert is_terminal("STOPPED")


def test_fixed_point_receipt_rejects_false_completion() -> None:
    receipt = sample_contract(contracts.FederationCompletionReceipt)
    assert isinstance(receipt, contracts.FederationCompletionReceipt)

    with pytest.raises(contracts.FederationAuthorityError):
        replace(receipt, outstanding_required_work=1)


def test_federation_receipt_cannot_impersonate_completion_or_omit_evidence() -> None:
    receipt = sample_contract(contracts.FederationReceipt)
    assert isinstance(receipt, contracts.FederationReceipt)

    with pytest.raises(contracts.FederationAuthorityError):
        replace(receipt, outcome="completed")
    with pytest.raises(contracts.FederationAuthorityError):
        replace(receipt, outcome="fixed_point")
    with pytest.raises(contracts.FederationContractError):
        replace(receipt, evidence_refs=())


def test_completion_receipt_requires_closed_outcome_zero_work_and_exact_evidence() -> None:
    receipt = sample_contract(contracts.FederationCompletionReceipt)
    assert isinstance(receipt, contracts.FederationCompletionReceipt)

    with pytest.raises(contracts.FederationAuthorityError):
        replace(receipt, outcome="accepted")
    with pytest.raises(contracts.FederationAuthorityError):
        replace(receipt, outcome="completed", outstanding_required_work=1)
    with pytest.raises(contracts.FederationContractError):
        replace(receipt, evidence_refs=())

    assert replace(receipt, outcome="completed").outcome == "completed"


@pytest.mark.parametrize("outcome", ["accepted", "completed", "fixed_point", "reported"])
def test_subagent_outcome_cannot_claim_authoritative_disposition(outcome: str) -> None:
    observed = sample_contract(contracts.SubagentOutcome)
    assert isinstance(observed, contracts.SubagentOutcome)

    with pytest.raises(contracts.FederationAuthorityError):
        replace(observed, outcome=outcome)


def test_subagent_outcome_requires_evidence_for_every_closed_disposition() -> None:
    observed = sample_contract(contracts.SubagentOutcome)
    assert isinstance(observed, contracts.SubagentOutcome)

    for outcome in ("succeeded", "failed", "cancelled"):
        assert replace(observed, outcome=outcome).outcome == outcome
    with pytest.raises(contracts.FederationContractError):
        replace(observed, evidence_refs=())


def test_federation_is_visible_in_cold_discovery_and_owns_every_module() -> None:
    discovery = agent_supervisor.agent_supervisor_cold_discovery()
    federation_surface = next(
        item for item in discovery["surfaces"] if item["id"] == "federation"
    )

    assert "federation" in agent_supervisor.AGENT_SUPERVISOR_DOMAIN_PACKAGES
    assert federation_surface == {
        "id": "federation",
        "module": "ipfs_accelerate_py.agent_supervisor.federation",
        "role": "service",
        "interface": federation.FEDERATION_INTERFACE,
    }
    package_path = Path(federation.__file__).parent
    existing_modules = tuple(
        path.stem
        for path in sorted(package_path.glob("*.py"))
        if path.name != "__init__.py"
    )
    assert federation.FEDERATION_OWNED_MODULES == existing_modules
