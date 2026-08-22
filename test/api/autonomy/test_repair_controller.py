from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.autonomous_repair.contracts import (
    AUTONOMOUS_REPAIR_INTERFACE,
    AutonomousRepairPolicy,
)
from ipfs_accelerate_py.agent_supervisor.autonomous_repair.engine import AutonomousRepairEngine
from ipfs_accelerate_py.agent_supervisor.autonomy.contracts import (
    AUTONOMOUS_META_CONTROLLER_PROGRAM_ID,
    AutonomousRepairReceipt,
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
    CANONICAL_REPAIR_ENGINE_INTERFACE,
    CANONICAL_REPAIR_ENGINE_TYPE,
    CONTROLLER_RELATIVE_PATH,
    LOW_RISK_MERGE_CONDITIONS,
    AutonomousRepairController,
    RepairBackoffPolicy,
    RepairControllerDisposition,
    RepairControllerError,
    RepairControllerResult,
    RepairMergeDisposition,
    RepairRequest,
    classify_forbidden_paths,
    is_self_edit_path,
    is_validator_policy_key_path,
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
        "max_repair_rounds": 3,
        "max_plan_branches": 1,
        "max_context_expansions": 2,
        "max_wall_time_ms": 30_000,
        "validation_reserve_ms": 1_000,
    }
    values.update(overrides)
    return CognitiveBudget(**values)


def _policy(**overrides: object) -> AutonomyPolicy:
    values: dict[str, object] = {
        "policy_revision": "policy-rev-repair-1",
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
        "blast_radius_symbols": ("AutonomyEnvelope",),
        "evidence_ids": ("evidence-risk",),
        "reason_codes": ("bounded_local_change",),
    }
    values.update(overrides)
    return RiskAssessment(**values)


def _envelope(
    *,
    policy: AutonomyPolicy,
    risk: RiskAssessment | None = None,
    **overrides: object,
) -> AutonomyEnvelope:
    assessed = risk or _risk()
    values: dict[str, object] = {
        "repository_id": "repo-1",
        "tree_id": "tree-1",
        "objective_id": "APMC-G000",
        "objective_revision": "objective-rev-1",
        "task_id": "APMC-013",
        "acceptance_criterion_ids": ("AC-repair-scope",),
        "risk_assessment": assessed,
        "autonomy_level": AutonomyLevel.EXECUTE_REVERSIBLE,
        "cognitive_budget": _budget(),
        "allowed_paths": ("ipfs_accelerate_py/agent_supervisor/autonomy",),
        "allowed_symbols": ("AutonomyEnvelope", "RepairTier"),
        "required_test_ids": ("test-repair",),
        "required_proof_ids": (),
        "authority_id": "operator-policy-authority",
        "policy_id": policy.policy_id,
        "provider_usage_envelope_id": "provider-envelope-1",
        "resource_budget_id": "resource-budget-1",
        "human_escalation_policy_id": "human-policy-1",
        "expiry_ms": 12_000,
        "reversible": assessed.reversible,
    }
    values.update(overrides)
    return AutonomyEnvelope(**values)


def _request(**overrides: object) -> RepairRequest:
    values: dict[str, object] = {
        "predicted_files": ("ipfs_accelerate_py/agent_supervisor/autonomy/contracts.py",),
        "predicted_symbols": ("AutonomyEnvelope",),
        "context_reference_ids": ("context-ref-1",),
        "worktree_id": "worktree-isolated-1",
        "rollback_plan_id": "rollback-plan-1",
        "required_test_ids": ("test-repair",),
        "completed_test_ids": ("test-repair",),
        "validation_receipt_ids": ("validation-1",),
        "changed_paths": ("ipfs_accelerate_py/agent_supervisor/autonomy/contracts.py",),
        "terminal_status": TerminalStatus.SUCCEEDED,
        "changed_line_count": 12,
    }
    values.update(overrides)
    return RepairRequest(**values)


def _controller(
    *,
    policy: AutonomyPolicy | None = None,
    envelope: AutonomyEnvelope | None = None,
    engine: AutonomousRepairEngine | None = None,
    model_invoker=None,
    backoff_policy: RepairBackoffPolicy | None = None,
) -> AutonomousRepairController:
    bound_policy = policy or _policy()
    return AutonomousRepairController(
        envelope=envelope or _envelope(policy=bound_policy),
        policy=bound_policy,
        engine=engine,
        model_invoker=model_invoker,
        backoff_policy=backoff_policy,
    )


def test_controller_interface_is_the_stated_facade() -> None:
    controller = _controller()
    assert controller.interface == AUTONOMOUS_REPAIR_CONTROLLER_INTERFACE
    assert controller.interface == "AutonomousRepairController@1"
    assert controller.engine_interface == AUTONOMOUS_REPAIR_INTERFACE
    assert controller.engine_interface == CANONICAL_REPAIR_ENGINE_INTERFACE
    assert controller.engine is None
    assert controller.authorizes_effect is False
    assert controller.authorizes_merge is False
    assert CANONICAL_REPAIR_ENGINE_TYPE is AutonomousRepairEngine


def test_controller_does_not_define_a_second_repair_engine() -> None:
    source = Path(
        "ipfs_accelerate_py/agent_supervisor/autonomy/repair_controller.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    class_names = {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}
    assert "AutonomousRepairEngine" not in class_names
    assert "AutonomousRepairController" in class_names
    assert "second repair engine" in source or "not a second repair engine" in source


def test_injected_engine_must_be_the_canonical_type(tmp_path: Path) -> None:
    policy = _policy()
    envelope = _envelope(policy=policy)
    engine = AutonomousRepairEngine(
        repo_root=tmp_path,
        policy=AutonomousRepairPolicy(apply_ir_logic=False, apply_doctor=False),
    )
    controller = AutonomousRepairController(envelope=envelope, policy=policy, engine=engine)
    assert type(controller.engine) is AutonomousRepairEngine

    class SecondRepairEngine:
        pass

    with pytest.raises(RepairControllerError, match="existing AutonomousRepairEngine"):
        AutonomousRepairController(
            envelope=envelope,
            policy=policy,
            engine=SecondRepairEngine(),  # type: ignore[arg-type]
        )

    class SubclassedEngine(AutonomousRepairEngine):
        pass

    with pytest.raises(RepairControllerError, match="existing AutonomousRepairEngine"):
        AutonomousRepairController(
            envelope=envelope,
            policy=policy,
            engine=SubclassedEngine(repo_root=tmp_path),
        )


def test_selects_deterministic_template_and_model_tiers() -> None:
    controller = _controller()
    deterministic = controller.plan(_request())
    assert deterministic.disposition is RepairControllerDisposition.ADMITTED
    assert deterministic.selected_tier is RepairTier.DETERMINISTIC
    assert deterministic.plan is not None
    assert deterministic.plan.repair_tier is RepairTier.DETERMINISTIC
    assert deterministic.plan.patch_envelope_id == controller.envelope.envelope_id
    assert deterministic.authorizes_merge is False

    template = controller.plan(_request(template_id="template:exact-rename"))
    assert template.selected_tier is RepairTier.TEMPLATE_CONSTRAINED
    assert template.plan is not None
    assert template.plan.repair_tier is RepairTier.TEMPLATE_CONSTRAINED

    model = controller.plan(
        _request(
            requested_tier=RepairTier.MODEL_ASSISTED_BOUNDED,
            requires_model=True,
        )
    )
    assert model.selected_tier is RepairTier.MODEL_ASSISTED_BOUNDED
    assert model.plan is not None
    assert model.plan.worktree_id == "worktree-isolated-1"
    assert model.plan.required_test_ids == ("test-repair",)


def test_model_assisted_repair_requires_isolated_worktree_and_checks() -> None:
    controller = _controller()
    result = controller.plan(
        _request(
            worktree_id="",
            requested_tier=RepairTier.MODEL_ASSISTED_BOUNDED,
            changed_paths=(),
            terminal_status=TerminalStatus.PENDING,
        )
    )
    assert result.disposition is RepairControllerDisposition.REJECTED_MISSING_PRECONDITIONS
    assert "isolated_worktree_required" in result.reason_codes
    assert "model_assisted_requires_isolated_worktree" in result.reason_codes
    assert result.merge_disposition is RepairMergeDisposition.REJECTED

    template = controller.plan(
        _request(
            requested_tier=RepairTier.TEMPLATE_CONSTRAINED,
            template_id="",
            changed_paths=(),
            terminal_status=TerminalStatus.PENDING,
        )
    )
    assert template.disposition is RepairControllerDisposition.REJECTED_MISSING_PRECONDITIONS
    assert "template_id_required" in template.reason_codes

    empty_policy = _policy()
    missing_checks = _controller(
        policy=empty_policy,
        envelope=_envelope(
            policy=empty_policy,
            required_test_ids=(),
            required_proof_ids=(),
        ),
    )
    result = missing_checks.plan(
        _request(required_test_ids=(), required_proof_ids=(), completed_test_ids=())
    )
    assert result.disposition is RepairControllerDisposition.REJECTED_MISSING_PRECONDITIONS
    assert "predetermined_checks_required" in result.reason_codes


def test_scope_escape_is_rejected() -> None:
    controller = _controller()
    result = controller.plan(
        _request(
            predicted_files=("docs/outside.md",),
            changed_paths=(),
            terminal_status=TerminalStatus.PENDING,
        )
    )
    assert result.disposition is RepairControllerDisposition.REJECTED_SCOPE_ESCAPE
    assert "scope_escape" in result.reason_codes
    assert result.merge_disposition is RepairMergeDisposition.REJECTED
    assert result.plan is None
    assert classify_forbidden_paths(
        ("docs/outside.md",),
        ("ipfs_accelerate_py/agent_supervisor/autonomy",),
    ) == ("scope_escape",)


def test_self_edit_of_the_controller_is_rejected() -> None:
    assert is_self_edit_path(CONTROLLER_RELATIVE_PATH)
    controller = _controller()
    result = controller.execute(
        _request(
            predicted_files=(CONTROLLER_RELATIVE_PATH,),
            predicted_symbols=("AutonomousRepairController",),
            changed_paths=(CONTROLLER_RELATIVE_PATH,),
        )
    )
    assert result.disposition is RepairControllerDisposition.REJECTED_SELF_EDIT
    assert "self_edit" in result.reason_codes
    assert result.merge_disposition is RepairMergeDisposition.REJECTED
    assert result.model_call_count == 0


def test_validator_policy_key_mutation_is_rejected() -> None:
    assert is_validator_policy_key_path(
        "config/agent_supervisor_autonomous_meta_controller_scheduler.json"
    )
    assert is_validator_policy_key_path("secrets/trusted_keys/operator.pem")
    controller = _controller()
    config_result = controller.plan(
        _request(
            predicted_files=(
                "config/agent_supervisor_autonomous_meta_controller_scheduler.json",
            ),
            predicted_symbols=("scheduler",),
            changed_paths=(),
            terminal_status=TerminalStatus.PENDING,
        )
    )
    assert (
        config_result.disposition
        is RepairControllerDisposition.REJECTED_VALIDATOR_POLICY_KEY
    )
    symbol_result = controller.plan(
        _request(
            predicted_symbols=("trusted_keys",),
            changed_paths=(),
            terminal_status=TerminalStatus.PENDING,
        )
    )
    assert (
        symbol_result.disposition
        is RepairControllerDisposition.REJECTED_VALIDATOR_POLICY_KEY
    )


def test_identical_failures_do_not_cause_repeated_model_calls() -> None:
    calls: list[RepairRequest] = []

    def invoker(request: RepairRequest) -> dict[str, str]:
        calls.append(request)
        return {"diagnostic_receipt_id": "diagnostic-repeat-1"}

    controller = _controller(
        model_invoker=invoker,
        backoff_policy=RepairBackoffPolicy(
            base_backoff_milliseconds=10,
            max_backoff_milliseconds=40,
            max_identical_failures=3,
        ),
    )
    failed = _request(
        requested_tier=RepairTier.MODEL_ASSISTED_BOUNDED,
        requires_model=True,
        failure_signature="fail:identical-scope",
        terminal_status=TerminalStatus.FAILED,
        validation_receipt_ids=(),
        completed_test_ids=(),
    )
    first = controller.execute(failed)
    assert first.disposition is RepairControllerDisposition.ROLLBACK_REQUIRED
    assert first.model_call_count == 1
    assert len(calls) == 1
    assert first.receipt is not None
    assert first.receipt.authorizes_merge is False

    second = controller.execute(failed)
    assert second.disposition is RepairControllerDisposition.IDENTICAL_FAILURE_BACKOFF
    assert second.diagnostic_reused is True
    assert second.model_call_count == 0
    assert len(calls) == 1
    assert second.backoff_milliseconds == 20
    assert "model_call_suppressed" in second.reason_codes
    assert second.merge_disposition is RepairMergeDisposition.HOLD
    assert controller.model_call_count == 1

    third = controller.execute(failed)
    fourth = controller.execute(failed)
    assert third.disposition is RepairControllerDisposition.IDENTICAL_FAILURE_BACKOFF
    assert fourth.disposition is RepairControllerDisposition.IDENTICAL_FAILURE_EXHAUSTED
    assert len(calls) == 1
    assert controller.model_call_count == 1


def test_repair_round_budget_is_not_expanded() -> None:
    policy = _policy()
    envelope = _envelope(policy=policy, cognitive_budget=_budget(max_repair_rounds=1))
    controller = _controller(policy=policy, envelope=envelope)
    first = controller.execute(_request(terminal_status=TerminalStatus.PENDING, changed_paths=()))
    assert first.disposition is RepairControllerDisposition.EXECUTED
    second = controller.execute(_request(terminal_status=TerminalStatus.PENDING, changed_paths=()))
    assert second.disposition is RepairControllerDisposition.REPAIR_BOUND_EXCEEDED
    assert second.merge_disposition is RepairMergeDisposition.HOLD


def test_r2_autonomous_merge_requires_every_low_risk_condition() -> None:
    result = _controller().execute(_request())
    assert result.disposition is RepairControllerDisposition.EXECUTED
    assert result.merge_disposition is RepairMergeDisposition.AUTONOMOUS_MERGE_ELIGIBLE
    assert set(result.merge_condition_results) == set(LOW_RISK_MERGE_CONDITIONS)
    assert all(result.merge_condition_results.values())
    assert result.authorizes_merge is False
    assert result.receipt is not None
    assert result.receipt.authorizes_merge is False
    assert result.receipt.terminal_status is TerminalStatus.SUCCEEDED


@pytest.mark.parametrize(
    ("override", "condition"),
    [
        ({"autonomous_merge_enabled": False}, "policy_autonomous_merge_enabled"),
        ({"changed_paths": ()}, "changed_paths_within_envelope"),
        ({"validation_receipt_ids": ()}, "validation_receipts_current"),
        ({"completed_test_ids": ()}, "required_tests_satisfied"),
        (
            {
                "changed_paths": (
                    "ipfs_accelerate_py/agent_supervisor/autonomy/human_escalation.py",
                )
            },
            "changed_paths_subset_of_predicted",
        ),
    ],
)
def test_r2_merge_fails_closed_if_any_low_risk_condition_is_false(
    override: dict[str, object],
    condition: str,
) -> None:
    policy_overrides = {}
    request_overrides = dict(override)
    if "autonomous_merge_enabled" in override:
        policy_overrides["autonomous_merge_enabled"] = override["autonomous_merge_enabled"]
        request_overrides.pop("autonomous_merge_enabled")
    policy = _policy(**policy_overrides)
    if "validation_receipt_ids" in override and override["validation_receipt_ids"] == ():
        request_overrides["terminal_status"] = TerminalStatus.PENDING
    result = _controller(policy=policy).execute(_request(**request_overrides))
    assert result.merge_disposition is not RepairMergeDisposition.AUTONOMOUS_MERGE_ELIGIBLE
    assert result.merge_condition_results[condition] is False
    assert result.authorizes_merge is False


def test_r3_repair_is_a_proposal_even_when_other_conditions_pass() -> None:
    policy = _policy(default_level=AutonomyLevel.SELF_REPAIR_ISOLATED)
    risk = _risk(risk_class=RiskClass.R3_BOUNDED_REPOSITORY_MUTATION)
    envelope = _envelope(
        policy=policy,
        risk=risk,
        autonomy_level=AutonomyLevel.SELF_REPAIR_ISOLATED,
    )
    result = _controller(policy=policy, envelope=envelope).execute(_request())
    assert result.disposition is RepairControllerDisposition.EXECUTED
    assert result.merge_disposition is RepairMergeDisposition.PROPOSAL_REQUIRED
    assert result.merge_condition_results["not_r3_or_higher"] is False
    assert result.merge_condition_results["risk_class_r2_or_lower"] is False
    assert result.authorizes_merge is False
    assert result.receipt is not None
    assert result.receipt.authorizes_merge is False


def test_failed_repair_binds_rollback_instead_of_merge() -> None:
    result = _controller().execute(
        _request(
            terminal_status=TerminalStatus.FAILED,
            validation_receipt_ids=(),
            completed_test_ids=(),
            failure_signature="fail:rollback",
        )
    )
    assert result.disposition is RepairControllerDisposition.ROLLBACK_REQUIRED
    assert result.merge_disposition is RepairMergeDisposition.ROLLBACK
    assert result.receipt is not None
    assert result.receipt.rollback_receipt_id == "rollback-plan-1"


def test_execute_delegates_to_the_existing_engine_without_claiming_success(
    tmp_path: Path,
) -> None:
    policy = _policy()
    engine = AutonomousRepairEngine(
        repo_root=tmp_path,
        policy=AutonomousRepairPolicy(apply_ir_logic=False, apply_doctor=False),
    )
    controller = AutonomousRepairController(
        envelope=_envelope(policy=policy),
        policy=policy,
        engine=engine,
    )
    result = controller.execute(
        _request(
            terminal_status=TerminalStatus.PENDING,
            changed_paths=(),
            work_items=(
                {
                    "work_id": "work:facade",
                    "operation": "catalog.read",
                    "path": "ipfs_accelerate_py/agent_supervisor/autonomy/contracts.py",
                },
            ),
        )
    )
    assert result.disposition is RepairControllerDisposition.EXECUTED
    assert result.engine_interface == AUTONOMOUS_REPAIR_INTERFACE
    assert result.engine_report_passed is False
    assert result.merge_disposition is not RepairMergeDisposition.AUTONOMOUS_MERGE_ELIGIBLE
    assert result.receipt is not None
    assert result.receipt.terminal_status is TerminalStatus.PENDING


def test_result_round_trip_preserves_identity_and_rejects_forged_merge() -> None:
    result = _controller().execute(_request())
    restored = RepairControllerResult.from_dict(result.to_dict())
    assert restored == result
    assert restored.result_id == result.result_id
    assert restored.to_dict()["program_id"] == AUTONOMOUS_META_CONTROLLER_PROGRAM_ID
    with pytest.raises(FrozenInstanceError):
        restored.model_call_count = 99  # type: ignore[misc]
    forged = result.to_dict()
    forged["authorizes_merge"] = True
    with pytest.raises(RepairControllerError, match="cannot authorize merge"):
        RepairControllerResult.from_dict(forged)
    tampered = result.to_dict()
    tampered["result_id"] = "forged-result"
    with pytest.raises(RepairControllerError, match="identity"):
        RepairControllerResult.from_dict(tampered)


def test_receipts_cannot_claim_independent_merge_authority() -> None:
    plan_result = _controller().plan(_request())
    assert plan_result.plan is not None
    with pytest.raises(Exception, match="authorize merge"):
        AutonomousRepairReceipt(
            plan_id=plan_result.plan.plan_id,
            envelope_id=plan_result.plan.patch_envelope_id,
            terminal_status=TerminalStatus.SUCCEEDED,
            changed_paths=plan_result.plan.predicted_files,
            validation_receipt_ids=("validation-1",),
            proof_receipt_ids=(),
            adversarial_assurance_receipt_ids=(),
            authorizes_merge=True,
        )


def test_unadmitted_source_edit_is_rejected_and_does_not_authorize_merge() -> None:
    result = _controller().plan(_request(allow_code_edit_materialize=True))
    assert result.disposition is RepairControllerDisposition.REJECTED_UNADMITTED_SOURCE_EDIT
    assert "policy_flag_is_not_source_edit_admission" in result.reason_codes
    assert result.merge_disposition is RepairMergeDisposition.REJECTED
    assert result.source_edit_admitted is False
    assert result.authorizes_merge is False
    assert result.plan is None


def test_envelope_policy_mismatch_is_rejected() -> None:
    with pytest.raises(RepairControllerError, match="policy identity"):
        AutonomousRepairController(envelope=_envelope(policy=_policy()), policy=_policy(policy_revision="other"))
