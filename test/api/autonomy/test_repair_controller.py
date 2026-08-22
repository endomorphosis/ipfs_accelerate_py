from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.autonomous_repair.contracts import (
    AUTONOMOUS_REPAIR_INTERFACE,
    AutonomousRepairReport,
)
from ipfs_accelerate_py.agent_supervisor.autonomous_repair.engine import AutonomousRepairEngine
from ipfs_accelerate_py.agent_supervisor.autonomy.contracts import (
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
    AutonomousRepairController,
    RepairControllerDisposition,
    RepairControllerError,
    RepairControllerRequest,
    RepairMergeDisposition,
    classify_repair_path,
    path_is_self_edit,
    path_is_validator_policy_key,
)

CONTROLLER_SOURCE = (
    Path(__file__).resolve().parents[3]
    / "ipfs_accelerate_py"
    / "agent_supervisor"
    / "autonomy"
    / "repair_controller.py"
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


def _envelope(
    *,
    policy: AutonomyPolicy,
    risk_class: RiskClass = RiskClass.R2_REVERSIBLE_LOCAL,
    **overrides: object,
) -> AutonomyEnvelope:
    reversible = risk_class.rank <= RiskClass.R3_BOUNDED_REPOSITORY_MUTATION.rank
    if risk_class is RiskClass.R5_IRREVERSIBLE_EXTERNAL_OR_LEGAL:
        reversible = False
    level = AutonomyLevel.EXECUTE_REVERSIBLE
    if risk_class is RiskClass.R3_BOUNDED_REPOSITORY_MUTATION:
        level = AutonomyLevel.SELF_REPAIR_ISOLATED
    if risk_class is RiskClass.R4_SECURITY_OR_PROTOCOL_SENSITIVE:
        level = AutonomyLevel.DRY_RUN
        reversible = True
    if risk_class is RiskClass.R5_IRREVERSIBLE_EXTERNAL_OR_LEGAL:
        level = AutonomyLevel.RECOMMEND
    values: dict[str, object] = {
        "repository_id": "repo-1",
        "tree_id": "tree-1",
        "objective_id": "APMC-G000",
        "objective_revision": "objective-rev-1",
        "task_id": "APMC-013",
        "acceptance_criterion_ids": ("AC-repair-scope",),
        "risk_assessment": RiskAssessment(
            risk_class=risk_class,
            reversible=reversible,
            blast_radius_paths=("ipfs_accelerate_py/agent_supervisor/autonomy",),
            blast_radius_symbols=("AutonomousRepairController",),
            evidence_ids=("evidence-risk",),
            reason_codes=("bounded_local_change",),
        ),
        "autonomy_level": level,
        "cognitive_budget": _budget(),
        "allowed_paths": ("ipfs_accelerate_py/agent_supervisor/autonomy",),
        "allowed_symbols": ("AutonomyEnvelope", "AutonomousRepairController"),
        "required_test_ids": ("test-repair-controller",),
        "required_proof_ids": (),
        "authority_id": "operator-policy-authority",
        "policy_id": policy.policy_id,
        "provider_usage_envelope_id": "provider-envelope-1",
        "resource_budget_id": "resource-budget-1",
        "human_escalation_policy_id": "human-policy-1",
        "expiry_ms": 12_000,
        "reversible": reversible,
    }
    values.update(overrides)
    return AutonomyEnvelope(**values)


def _request(**overrides: object) -> RepairControllerRequest:
    values: dict[str, object] = {
        "predicted_files": ("ipfs_accelerate_py/agent_supervisor/autonomy/contracts.py",),
        "predicted_symbols": ("AutonomyEnvelope",),
        "worktree_id": "worktree-isolated-1",
        "rollback_plan_id": "rollback-plan-1",
        "requested_tier": RepairTier.DETERMINISTIC,
        "context_reference_ids": ("context-ref-1",),
        "required_test_ids": ("test-repair-controller",),
        "required_proof_ids": (),
        "max_changed_files": 1,
        "max_changed_lines": 80,
    }
    values.update(overrides)
    return RepairControllerRequest(**values)


def _controller(
    *,
    policy: AutonomyPolicy | None = None,
    envelope: AutonomyEnvelope | None = None,
    engine: AutonomousRepairEngine | None = None,
    model_call=None,
) -> AutonomousRepairController:
    bound_policy = policy or _policy()
    return AutonomousRepairController(
        policy=bound_policy,
        envelope=envelope or _envelope(policy=bound_policy),
        engine=engine,
        model_call=model_call,
    )


def test_interface_is_the_existing_engine_facade(tmp_path: Path) -> None:
    engine = AutonomousRepairEngine(repo_root=tmp_path)
    controller = _controller(engine=engine)
    assert AUTONOMOUS_REPAIR_CONTROLLER_INTERFACE == "AutonomousRepairController@1"
    assert AutonomousRepairController.INTERFACE == AUTONOMOUS_REPAIR_CONTROLLER_INTERFACE
    assert AutonomousRepairController.ENGINE_INTERFACE == AUTONOMOUS_REPAIR_INTERFACE
    assert controller.engine_interface == AUTONOMOUS_REPAIR_INTERFACE
    assert controller.engine is engine
    assert type(controller.engine) is AutonomousRepairEngine

    tree = ast.parse(CONTROLLER_SOURCE.read_text(encoding="utf-8"))
    defined = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    assert "AutonomousRepairEngine" not in defined
    assert not any(name.endswith("RepairEngine") for name in defined)


def test_subclassed_engine_is_rejected(tmp_path: Path) -> None:
    class OtherEngine(AutonomousRepairEngine):
        pass

    policy = _policy()
    with pytest.raises(RepairControllerError, match="existing AutonomousRepairEngine"):
        AutonomousRepairController(
            policy=policy,
            envelope=_envelope(policy=policy),
            engine=OtherEngine(repo_root=tmp_path),
        )


def test_deterministic_and_template_tiers_bind_the_envelope() -> None:
    controller = _controller()
    deterministic = controller.evaluate(_request())
    assert deterministic.disposition is RepairControllerDisposition.ADMITTED
    assert deterministic.repair_tier is RepairTier.DETERMINISTIC
    assert deterministic.plan is not None
    assert deterministic.plan.patch_envelope_id == controller.envelope.envelope_id
    assert deterministic.plan.worktree_id == "worktree-isolated-1"
    assert deterministic.plan.required_test_ids == ("test-repair-controller",)
    assert not deterministic.authorizes_effect
    assert not deterministic.authorizes_merge
    assert deterministic.receipt is not None
    assert deterministic.receipt.authorizes_merge is False

    templated = controller.evaluate(
        _request(requested_tier=RepairTier.TEMPLATE_CONSTRAINED, template_available=True)
    )
    assert templated.repair_tier is RepairTier.TEMPLATE_CONSTRAINED
    assert templated.plan is not None
    assert templated.plan.predicted_symbols == ("AutonomyEnvelope",)


def test_model_assisted_requires_exact_scope_context_worktree_and_checks() -> None:
    controller = _controller()
    result = controller.evaluate(
        _request(
            requested_tier=RepairTier.MODEL_ASSISTED_BOUNDED,
            require_model=True,
            context_reference_ids=(),
        )
    )
    assert result.disposition is RepairControllerDisposition.INSUFFICIENT_CONTEXT
    assert result.plan is None
    assert result.model_call_count == 0

    admitted = controller.evaluate(
        _request(
            requested_tier=RepairTier.MODEL_ASSISTED_BOUNDED,
            require_model=True,
            context_reference_ids=("context-ref-1",),
        )
    )
    assert admitted.repair_tier is RepairTier.MODEL_ASSISTED_BOUNDED
    assert admitted.plan is not None
    assert admitted.plan.worktree_id == "worktree-isolated-1"
    assert admitted.plan.required_test_ids == ("test-repair-controller",)


def test_scope_escape_self_edit_and_validator_policy_key_are_rejected() -> None:
    policy = _policy()
    wide = _envelope(
        policy=policy,
        allowed_paths=("ipfs_accelerate_py/agent_supervisor",),
        allowed_symbols=("AutonomyEnvelope", "ProposalValidator", "AutonomousRepairController"),
    )
    controller = AutonomousRepairController(policy=policy, envelope=wide)
    escape = controller.evaluate(
        _request(predicted_files=("docs/architecture/outside.md",), predicted_symbols=("heading",))
    )
    assert escape.disposition is RepairControllerDisposition.REJECTED
    assert "scope_escape" in escape.reason_codes
    assert escape.merge.disposition is RepairMergeDisposition.REJECTED

    self_edit = controller.evaluate(
        _request(
            predicted_files=(
                "ipfs_accelerate_py/agent_supervisor/autonomy/repair_controller.py",
            ),
            predicted_symbols=("AutonomousRepairController",),
        )
    )
    assert "self_edit" in self_edit.reason_codes
    assert path_is_self_edit(
        "ipfs_accelerate_py/agent_supervisor/autonomy/repair_controller.py"
    )

    validator = controller.evaluate(
        _request(
            predicted_files=(
                "ipfs_accelerate_py/agent_supervisor/validation/proposal_validation.py",
            ),
            predicted_symbols=("ProposalValidator",),
        )
    )
    assert "validator_policy_key_mutation" in validator.reason_codes
    key = controller.evaluate(
        _request(
            predicted_files=("config/trusted_keys.json",),
            predicted_symbols=("trusted_keys",),
            forbidden_symbols=(),
        )
    )
    assert "validator_policy_key_mutation" in key.reason_codes
    assert path_is_validator_policy_key("config/trusted_keys.json")
    assert "scope_escape" in classify_repair_path(
        "docs/outside.md", ("ipfs_accelerate_py/agent_supervisor/autonomy",)
    )


def test_identical_failures_reuse_diagnosis_and_do_not_repeat_model_calls() -> None:
    calls: list[int] = []

    def model_call(**_kwargs: object) -> dict[str, str]:
        calls.append(1)
        return {
            "failure_signature": "fail:same-diagnostic",
            "diagnostic_receipt_id": "diag:repeat-1",
        }

    controller = _controller(model_call=model_call)
    first = controller.evaluate(
        _request(
            requested_tier=RepairTier.MODEL_ASSISTED_BOUNDED,
            require_model=True,
            context_reference_ids=("context-ref-1",),
        )
    )
    assert first.model_call_count == 1
    assert first.receipt is not None
    assert first.receipt.terminal_status is TerminalStatus.PENDING
    assert len(calls) == 1

    second = controller.evaluate(
        _request(
            requested_tier=RepairTier.MODEL_ASSISTED_BOUNDED,
            require_model=True,
            context_reference_ids=("context-ref-1",),
            failure_signature="fail:same-diagnostic",
        )
    )
    third = controller.evaluate(
        _request(
            requested_tier=RepairTier.MODEL_ASSISTED_BOUNDED,
            require_model=True,
            context_reference_ids=("context-ref-1",),
            failure_signature="fail:same-diagnostic",
        )
    )
    assert second.disposition is RepairControllerDisposition.IDENTICAL_FAILURE_REUSED
    assert second.diagnostic_reused
    assert second.model_call_count == 0
    assert second.backoff_milliseconds == 20
    assert third.model_call_count == 0
    assert len(calls) == 1
    assert controller.model_call_count == 1


def test_r2_autonomous_merge_requires_every_low_risk_condition() -> None:
    controller = _controller()
    result = controller.evaluate(
        _request(
            changed_paths=("ipfs_accelerate_py/agent_supervisor/autonomy/contracts.py",),
            validation_receipt_ids=("test-repair-controller",),
        )
    )
    assert result.merge.satisfied
    assert result.merge.disposition is RepairMergeDisposition.AUTONOMOUS_MERGE
    assert tuple(result.merge.conditions) == LOW_RISK_MERGE_CONDITIONS
    assert all(result.merge.conditions[name] for name in LOW_RISK_MERGE_CONDITIONS)
    assert result.receipt is not None
    assert result.receipt.authorizes_merge is False
    assert result.receipt.terminal_status is TerminalStatus.SUCCEEDED

    disabled = _controller(policy=_policy(autonomous_merge_enabled=False))
    blocked = disabled.evaluate(
        _request(
            changed_paths=("ipfs_accelerate_py/agent_supervisor/autonomy/contracts.py",),
            validation_receipt_ids=("test-repair-controller",),
        )
    )
    assert not blocked.merge.satisfied
    assert blocked.merge.conditions["autonomous_merge_enabled"] is False
    assert blocked.merge.disposition is RepairMergeDisposition.PROPOSAL


def test_r3_is_proposal_even_when_other_conditions_hold() -> None:
    policy = _policy(
        default_level=AutonomyLevel.SELF_REPAIR_ISOLATED,
        autonomous_merge_enabled=True,
    )
    envelope = _envelope(
        policy=policy,
        risk_class=RiskClass.R3_BOUNDED_REPOSITORY_MUTATION,
        autonomy_level=AutonomyLevel.SELF_REPAIR_ISOLATED,
    )
    result = AutonomousRepairController(policy=policy, envelope=envelope).evaluate(
        _request(
            changed_paths=("ipfs_accelerate_py/agent_supervisor/autonomy/contracts.py",),
            validation_receipt_ids=("test-repair-controller",),
        )
    )
    assert result.merge.risk_class is RiskClass.R3_BOUNDED_REPOSITORY_MUTATION
    assert result.merge.conditions["risk_at_most_r2"] is False
    assert not result.merge.satisfied
    assert result.merge.disposition is RepairMergeDisposition.PROPOSAL
    assert result.receipt is not None
    assert result.receipt.authorizes_merge is False


def test_run_delegates_to_the_existing_engine_only(tmp_path: Path) -> None:
    engine = AutonomousRepairEngine(repo_root=tmp_path)
    seen: list[object] = []

    def fake_run(items: object) -> AutonomousRepairReport:
        seen.append(items)
        return AutonomousRepairReport(
            policy={"domain": "agent_supervisor"},
            rows=[],
            passed=False,
            model_call_count=0,
        )

    engine.run = fake_run  # type: ignore[method-assign]
    result = _controller(engine=engine).run(_request())
    assert result.disposition is RepairControllerDisposition.ENGINE_DELEGATED
    assert result.engine_report is not None
    assert result.engine_report["interface"] == AUTONOMOUS_REPAIR_INTERFACE
    assert result.engine_report["completion_authoritative"] is False
    assert seen and result.authorizes_merge is False


def test_result_rejects_merge_and_effect_authority_claims() -> None:
    result = _controller().evaluate(_request())
    with pytest.raises(FrozenInstanceError):
        result.disposition = RepairControllerDisposition.REJECTED  # type: ignore[misc]
    payload = result.to_dict()
    assert payload["authorizes_effect"] is False
    assert payload["authorizes_merge"] is False
    assert payload["engine_interface"] == AUTONOMOUS_REPAIR_INTERFACE
