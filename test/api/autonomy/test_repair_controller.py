from __future__ import annotations

import inspect

import pytest
from ipfs_accelerate_py.agent_supervisor.autonomous_repair.engine import AutonomousRepairEngine
from ipfs_accelerate_py.agent_supervisor.autonomy import repair_controller as repair_controller_module
from ipfs_accelerate_py.agent_supervisor.autonomy.contracts import (
    AutonomousRepairPlan,
    AutonomyEnvelope,
    AutonomyLevel,
    AutonomyPolicy,
    CognitiveBudget,
    RepairTier,
    RiskAssessment,
    RiskClass,
    TerminalStatus,
)
from ipfs_accelerate_py.agent_supervisor.autonomy.repair_controller import (
    AUTONOMOUS_REPAIR_CONTROLLER_INTERFACE,
    LOW_RISK_MERGE_CONDITIONS,
    PROTECTED_AUTHORITY_PATHS,
    SELF_EDIT_PATH,
    AutonomousRepairController,
    RepairControllerDisposition,
    RepairControllerError,
    RepairControllerRequest,
    RepairControllerResult,
    RepairMergeDisposition,
    RepairMutationClass,
    classify_mutation,
    evaluate_merge_conjunction,
    select_repair_tier,
)


def _budget(**overrides: int) -> CognitiveBudget:
    values = {
        "max_total_model_calls": 4,
        "max_strong_model_calls": 1,
        "max_input_tokens": 8_000,
        "max_output_tokens": 2_000,
        "max_provider_spend_micros": 20_000,
        "max_proof_time_ms": 10_000,
        "max_validation_time_ms": 10_000,
        "max_human_questions": 1,
        "max_repair_rounds": 2,
        "max_plan_branches": 1,
        "max_context_expansions": 2,
        "max_wall_time_ms": 30_000,
        "validation_reserve_ms": 1_000,
    }
    values.update(overrides)
    return CognitiveBudget(**values)


def _policy(**overrides: object) -> AutonomyPolicy:
    values: dict[str, object] = {
        "policy_revision": "policy-rev-1",
        "authority_id": "operator-policy-authority",
        "human_escalation_policy_id": "human-policy-1",
        "default_level": AutonomyLevel.EXECUTE_REVERSIBLE,
        "autonomous_merge_enabled": True,
    }
    values.update(overrides)
    return AutonomyPolicy(**values)


def _risk(**overrides: object) -> RiskAssessment:
    values: dict[str, object] = {
        "risk_class": RiskClass.R2_REVERSIBLE_LOCAL,
        "reversible": True,
        "blast_radius_paths": ("ipfs_accelerate_py/agent_supervisor/autonomy",),
        "blast_radius_symbols": ("AutonomousRepairController",),
        "evidence_ids": ("evidence-risk",),
        "reason_codes": ("bounded_local_change",),
    }
    values.update(overrides)
    return RiskAssessment(**values)


def _envelope(*, policy: AutonomyPolicy, **overrides: object) -> AutonomyEnvelope:
    values: dict[str, object] = {
        "repository_id": "repo-1",
        "tree_id": "tree-1",
        "objective_id": "APMC-G000",
        "objective_revision": "objective-rev-1",
        "task_id": "APMC-013",
        "acceptance_criterion_ids": ("AC-repair-scope",),
        "risk_assessment": _risk(),
        "autonomy_level": AutonomyLevel.EXECUTE_REVERSIBLE,
        "cognitive_budget": _budget(),
        "allowed_paths": ("ipfs_accelerate_py/agent_supervisor/autonomy",),
        "allowed_symbols": ("AutonomousRepairController", "AutonomyEnvelope"),
        "required_test_ids": ("test-repair-controller",),
        "required_proof_ids": (),
        "authority_id": "operator-policy-authority",
        "policy_id": policy.policy_id,
        "provider_usage_envelope_id": "provider-envelope-1",
        "resource_budget_id": "resource-budget-1",
        "human_escalation_policy_id": "human-policy-1",
        "expiry_ms": 12_000,
        "reversible": True,
    }
    values.update(overrides)
    return AutonomyEnvelope(**values)


def _plan(*, envelope: AutonomyEnvelope, **overrides: object) -> AutonomousRepairPlan:
    values: dict[str, object] = {
        "objective_id": envelope.objective_id,
        "task_id": envelope.task_id,
        "repair_tier": RepairTier.DETERMINISTIC,
        "predicted_files": ("ipfs_accelerate_py/agent_supervisor/autonomy/runtime.py",),
        "predicted_symbols": ("AutonomousRepairController",),
        "patch_envelope_id": envelope.envelope_id,
        "context_reference_ids": ("context-ref-1",),
        "required_test_ids": envelope.required_test_ids,
        "required_proof_ids": envelope.required_proof_ids,
        "worktree_id": "worktree-isolated-1",
        "allowed_paths": envelope.allowed_paths,
        "forbidden_symbols": ("trusted_keys", "validator_policy_key"),
        "rollback_plan_id": "rollback-plan-1",
        "risk_class": envelope.risk_assessment.risk_class,
        "max_changed_files": 2,
        "max_changed_lines": 80,
    }
    values.update(overrides)
    return AutonomousRepairPlan(**values)


def _request(
    *,
    policy: AutonomyPolicy | None = None,
    envelope: AutonomyEnvelope | None = None,
    plan: AutonomousRepairPlan | None = None,
    **overrides: object,
) -> RepairControllerRequest:
    policy = policy or _policy()
    envelope = envelope or _envelope(policy=policy)
    plan = plan or _plan(envelope=envelope)
    values: dict[str, object] = {
        "plan": plan,
        "envelope": envelope,
        "policy": policy,
        "changed_paths": plan.predicted_files,
        "changed_line_count": 12,
        "validation_receipt_ids": envelope.required_test_ids or ("test-repair-controller",),
        "proof_receipt_ids": envelope.required_proof_ids,
        "adversarial_assurance_receipt_ids": ("assurance-1",),
        "diagnostic_receipt_id": "diagnostic-1",
        "isolated_worktree": True,
    }
    values.update(overrides)
    return RepairControllerRequest(**values)


def test_interface_is_versioned_facade_not_a_second_engine() -> None:
    controller = AutonomousRepairController()
    assert AUTONOMOUS_REPAIR_CONTROLLER_INTERFACE == "AutonomousRepairController@1"
    assert controller.interface == AUTONOMOUS_REPAIR_CONTROLLER_INTERFACE
    assert controller.INTERFACE == AUTONOMOUS_REPAIR_CONTROLLER_INTERFACE
    assert controller.engine is None
    assert controller.creates_repair_engine is False
    source = inspect.getsource(repair_controller_module)
    assert "class AutonomousRepairEngine" not in source
    assert "class AutonomousRepairMaterializer" not in source
    assert "write_bytes" not in source
    assert "from ..autonomous_repair.engine import AutonomousRepairEngine" in source
    with pytest.raises(RepairControllerError, match="existing AutonomousRepairEngine"):
        AutonomousRepairController(engine=object())  # type: ignore[arg-type]


def test_injected_engine_is_the_existing_authority(tmp_path) -> None:
    engine = AutonomousRepairEngine(repo_root=tmp_path)
    controller = AutonomousRepairController(engine=engine)
    assert controller.engine is engine
    assert type(controller.engine) is AutonomousRepairEngine


def test_selects_deterministic_then_template_then_model_without_raising_tier() -> None:
    assert (
        select_repair_tier(
            requested=RepairTier.MODEL_ASSISTED_BOUNDED,
            deterministic_applicable=True,
            template_applicable=True,
            model_assistance_requested=True,
        )
        is RepairTier.DETERMINISTIC
    )
    assert (
        select_repair_tier(
            requested=RepairTier.MODEL_ASSISTED_BOUNDED,
            deterministic_applicable=False,
            template_applicable=True,
            model_assistance_requested=True,
        )
        is RepairTier.TEMPLATE_CONSTRAINED
    )
    assert (
        select_repair_tier(
            requested=RepairTier.TEMPLATE_CONSTRAINED,
            deterministic_applicable=False,
            template_applicable=False,
            model_assistance_requested=True,
        )
        is RepairTier.TEMPLATE_CONSTRAINED
    )
    policy = _policy()
    envelope = _envelope(policy=policy)
    plan = _plan(envelope=envelope, repair_tier=RepairTier.MODEL_ASSISTED_BOUNDED)
    result = AutonomousRepairController().repair(
        _request(
            policy=policy,
            envelope=envelope,
            plan=plan,
            deterministic_applicable=False,
            template_applicable=False,
            model_assistance_requested=True,
            isolated_worktree=True,
        )
    )
    assert result.selected_tier is RepairTier.MODEL_ASSISTED_BOUNDED
    assert result.disposition is RepairControllerDisposition.ADMITTED
    lowered = AutonomousRepairController().repair(
        _request(
            policy=policy,
            envelope=envelope,
            plan=plan,
            deterministic_applicable=True,
            template_applicable=True,
            model_assistance_requested=True,
        )
    )
    assert lowered.selected_tier is RepairTier.DETERMINISTIC


def test_scope_escape_self_edit_and_validator_policy_key_are_rejected() -> None:
    policy = _policy()
    controller = AutonomousRepairController()
    envelope = _envelope(
        policy=policy,
        allowed_paths=("ipfs_accelerate_py/agent_supervisor/autonomy",),
        allowed_symbols=("AutonomyEnvelope",),
    )
    escaped = _plan(
        envelope=envelope,
        allowed_paths=("ipfs_accelerate_py/agent_supervisor",),
        predicted_files=("ipfs_accelerate_py/agent_supervisor/context/decision_runtime.py",),
        predicted_symbols=("DecisionRuntime",),
    )
    escape = controller.repair(
        _request(
            policy=policy,
            envelope=envelope,
            plan=escaped,
            changed_paths=escaped.predicted_files,
            validation_receipt_ids=("test-repair-controller",),
        )
    )
    assert escape.disposition is RepairControllerDisposition.REJECTED
    assert escape.mutation_class is RepairMutationClass.SCOPE_ESCAPE
    assert "scope_escape" in escape.reason_codes
    assert not escape.authorizes_merge
    assert not escape.authorizes_effect

    self_plan = _plan(
        envelope=envelope,
        predicted_files=(SELF_EDIT_PATH,),
        predicted_symbols=("AutonomousRepairController",),
    )
    self_edit = controller.repair(
        _request(
            policy=policy,
            envelope=envelope,
            plan=self_plan,
            changed_paths=self_plan.predicted_files,
        )
    )
    assert self_edit.disposition is RepairControllerDisposition.REJECTED
    assert self_edit.mutation_class is RepairMutationClass.SELF_EDIT
    assert "self_edit" in self_edit.reason_codes

    key_envelope = _envelope(
        policy=policy,
        allowed_paths=("ipfs_accelerate_py",),
        allowed_symbols=("llm_router",),
    )
    key_plan = _plan(
        envelope=key_envelope,
        allowed_paths=("ipfs_accelerate_py",),
        predicted_files=("ipfs_accelerate_py/llm_router.py",),
        predicted_symbols=("llm_router",),
        forbidden_symbols=(),
    )
    key_edit = controller.repair(
        _request(
            policy=policy,
            envelope=key_envelope,
            plan=key_plan,
            changed_paths=key_plan.predicted_files,
        )
    )
    assert key_edit.disposition is RepairControllerDisposition.REJECTED
    assert key_edit.mutation_class is RepairMutationClass.VALIDATOR_POLICY_KEY
    assert "validator_policy_key_mutation" in key_edit.reason_codes
    assert "ipfs_accelerate_py/llm_router.py" in PROTECTED_AUTHORITY_PATHS
    assert classify_mutation(
        SELF_EDIT_PATH,
        allowed_paths=("ipfs_accelerate_py/agent_supervisor/autonomy",),
    ) is RepairMutationClass.SELF_EDIT


def test_identical_failures_do_not_repeat_model_calls() -> None:
    calls: list[str] = []

    def _model(request: RepairControllerRequest) -> dict[str, object]:
        calls.append(request.failure_signature)
        return {"failed": True}

    policy = _policy()
    envelope = _envelope(policy=policy)
    plan = _plan(envelope=envelope, repair_tier=RepairTier.MODEL_ASSISTED_BOUNDED)
    request = _request(
        policy=policy,
        envelope=envelope,
        plan=plan,
        deterministic_applicable=False,
        template_applicable=False,
        model_assistance_requested=True,
        isolated_worktree=True,
        diagnostic_receipt_id="diagnostic-repeat",
        failure_extra="same-fault",
    )
    controller = AutonomousRepairController(
        model_invoker=_model,
        max_identical_failures=3,
        base_backoff_milliseconds=10,
        max_backoff_milliseconds=40,
    )
    first = controller.repair(request)
    second = controller.repair(request)
    third = controller.repair(request)
    fourth = controller.repair(request)

    assert first.disposition is RepairControllerDisposition.REJECTED
    assert first.model_call_count == 1
    assert second.disposition is RepairControllerDisposition.BACKED_OFF
    assert second.diagnostic_reused
    assert second.model_call_count == 0
    assert "no_repeated_model_call" in second.reason_codes
    assert second.backoff_milliseconds == 10
    assert third.disposition is RepairControllerDisposition.BACKED_OFF
    assert third.model_call_count == 0
    assert fourth.disposition is RepairControllerDisposition.EXHAUSTED
    assert fourth.model_call_count == 0
    assert len(calls) == 1


def test_r2_autonomous_merge_requires_every_stated_low_risk_condition() -> None:
    policy = _policy(autonomous_merge_enabled=True)
    envelope = _envelope(policy=policy)
    plan = _plan(envelope=envelope)
    request = _request(policy=policy, envelope=envelope, plan=plan)
    result = AutonomousRepairController().repair(request)
    assert result.disposition is RepairControllerDisposition.ADMITTED
    assert result.merge.disposition is RepairMergeDisposition.AUTONOMOUS_MERGE_ELIGIBLE
    assert tuple(result.merge.conditions) == LOW_RISK_MERGE_CONDITIONS
    assert all(result.merge.conditions.values())
    assert result.merge.unsatisfied == ()
    assert result.merge.authorizes_merge is False
    assert result.receipt is not None
    assert result.receipt.authorizes_merge is False
    assert result.authorizes_merge is False
    assert "r2_merge_conjunction_satisfied" in result.reason_codes

    disabled = _policy(autonomous_merge_enabled=False)
    disabled_envelope = _envelope(policy=disabled)
    incomplete = AutonomousRepairController().repair(
        _request(
            policy=disabled,
            envelope=disabled_envelope,
            plan=_plan(envelope=disabled_envelope),
        )
    )
    assert incomplete.merge.disposition is RepairMergeDisposition.PROPOSAL_ONLY
    assert "autonomous_merge_enabled" in incomplete.merge.unsatisfied
    conditions = evaluate_merge_conjunction(
        policy=disabled,
        envelope=disabled_envelope,
        plan=_plan(envelope=disabled_envelope),
        selected_tier=RepairTier.DETERMINISTIC,
        changed_paths=("ipfs_accelerate_py/agent_supervisor/autonomy/runtime.py",),
        changed_line_count=12,
        validation_receipt_ids=("test-repair-controller",),
        proof_receipt_ids=(),
        terminal_status=TerminalStatus.SUCCEEDED,
        isolated_worktree=True,
    )
    assert conditions["autonomous_merge_enabled"] is False
    assert set(conditions) == set(LOW_RISK_MERGE_CONDITIONS)


def test_r3_repair_is_proposal_only_even_when_other_conditions_hold() -> None:
    policy = _policy(
        default_level=AutonomyLevel.SELF_REPAIR_ISOLATED,
        autonomous_merge_enabled=True,
    )
    envelope = _envelope(
        policy=policy,
        risk_assessment=_risk(risk_class=RiskClass.R3_BOUNDED_REPOSITORY_MUTATION),
        autonomy_level=AutonomyLevel.SELF_REPAIR_ISOLATED,
    )
    plan = _plan(envelope=envelope, risk_class=RiskClass.R3_BOUNDED_REPOSITORY_MUTATION)
    result = AutonomousRepairController().repair(
        _request(policy=policy, envelope=envelope, plan=plan, isolated_worktree=True)
    )
    assert result.disposition is RepairControllerDisposition.ADMITTED
    assert result.merge.disposition is RepairMergeDisposition.PROPOSAL_ONLY
    assert "risk_class_r2_or_lower" in result.merge.unsatisfied
    assert result.merge.authorizes_merge is False
    assert "r3_or_incomplete_conjunction_is_proposal" in result.reason_codes


def test_model_assisted_requires_isolated_worktree_and_predetermined_checks() -> None:
    policy = _policy()
    envelope = _envelope(policy=policy, required_test_ids=())
    plan = _plan(
        envelope=envelope,
        repair_tier=RepairTier.MODEL_ASSISTED_BOUNDED,
        required_test_ids=(),
    )
    result = AutonomousRepairController().repair(
        _request(
            policy=policy,
            envelope=envelope,
            plan=plan,
            deterministic_applicable=False,
            template_applicable=False,
            model_assistance_requested=True,
            isolated_worktree=False,
            validation_receipt_ids=("validation-1",),
        )
    )
    assert result.disposition is RepairControllerDisposition.REJECTED
    assert "model_assisted_requires_isolated_worktree" in result.reason_codes
    assert "model_assisted_requires_predetermined_tests" in result.reason_codes


def test_rollback_discards_the_attempt_without_merge() -> None:
    result = AutonomousRepairController().repair(_request(roll_back=True))
    assert result.disposition is RepairControllerDisposition.ROLLED_BACK
    assert "rollback_plan_invoked" in result.reason_codes
    assert result.merge.disposition is RepairMergeDisposition.NOT_CONSIDERED
    assert result.receipt is not None
    assert result.receipt.terminal_status is TerminalStatus.CANCELLED


def test_result_is_bounded_and_does_not_claim_merge_authority() -> None:
    result = AutonomousRepairController().repair(_request())
    payload = result.to_dict()
    assert payload["schema"].endswith("repair-controller-result@1")
    assert payload["interface"] == AUTONOMOUS_REPAIR_CONTROLLER_INTERFACE
    assert payload["authorizes_merge"] is False
    assert payload["authorizes_effect"] is False
    restored_id = RepairControllerResult(
        disposition=result.disposition,
        selected_tier=result.selected_tier,
        reason_codes=result.reason_codes,
        merge=result.merge,
        plan=result.plan,
        envelope_id=result.envelope_id,
        failure_signature=result.failure_signature,
        mutation_class=result.mutation_class,
        receipt=result.receipt,
        diagnostic_reused=result.diagnostic_reused,
        backoff_milliseconds=result.backoff_milliseconds,
        model_call_count=result.model_call_count,
        source_edit_admitted=result.source_edit_admitted,
        engine_invoked=result.engine_invoked,
        materializer_invoked=result.materializer_invoked,
        suffix_receipt_id=result.suffix_receipt_id,
    ).result_id
    assert restored_id == result.result_id


def test_delegate_to_engine_does_not_mint_a_second_engine() -> None:
    result = AutonomousRepairController().repair(_request(delegate_to_engine=True))
    assert result.disposition is RepairControllerDisposition.REJECTED
    assert "existing_engine_required" in result.reason_codes
    assert "no_second_repair_engine" in result.reason_codes
    assert result.engine_invoked is False


def test_source_edit_requires_typed_operator_not_policy_flag(tmp_path) -> None:
    engine = AutonomousRepairEngine(
        repo_root=tmp_path,
        policy={"allow_code_edit_materialize": True},
    )
    controller = AutonomousRepairController(engine=engine)
    result = controller.repair(_request(apply_source_edit=True))
    assert result.disposition is RepairControllerDisposition.REJECTED
    assert "typed_admitted_source_edit_operator_required" in result.reason_codes
    assert "policy_flag_is_not_source_edit_admission" in result.reason_codes
    assert result.source_edit_admitted is False
    assert result.materializer_invoked is False
    assert result.engine_invoked is False
    assert controller.creates_repair_engine is False
    assert controller.engine is engine


def test_malformed_request_fails_closed() -> None:
    policy = _policy()
    envelope = _envelope(policy=policy)
    with pytest.raises(RepairControllerError, match="must be an AutonomousRepairPlan"):
        RepairControllerRequest(plan=object(), envelope=envelope, policy=policy)  # type: ignore[arg-type]
    other = _policy(policy_revision="policy-rev-other")
    with pytest.raises(RepairControllerError, match="policy_id does not match"):
        RepairControllerRequest(plan=_plan(envelope=envelope), envelope=envelope, policy=other)
