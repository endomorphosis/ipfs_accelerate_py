from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.autonomous_repair.contracts import (
    AutonomousRepairReport,
)
from ipfs_accelerate_py.agent_supervisor.autonomous_repair.engine import (
    AutonomousRepairEngine,
)
from ipfs_accelerate_py.agent_supervisor.autonomy import repair_controller as repair_module
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
from ipfs_accelerate_py.agent_supervisor.autonomy.receding_horizon import (
    PlanSuffixInvalidationReceipt,
    RecedingHorizonDisposition,
)
from ipfs_accelerate_py.agent_supervisor.autonomy.repair_controller import (
    AUTONOMOUS_REPAIR_CONTROLLER_INTERFACE,
    LOW_RISK_MERGE_CONDITIONS,
    SELF_EDIT_PATH,
    AutonomousRepairController,
    RepairControllerDisposition,
    RepairControllerError,
    RepairMergeDisposition,
    RepairRequest,
    evaluate_low_risk_merge_conjunction,
    protected_authority_reason,
)


def _budget() -> CognitiveBudget:
    return CognitiveBudget(
        max_total_model_calls=4,
        max_strong_model_calls=1,
        max_input_tokens=8_000,
        max_output_tokens=2_000,
        max_provider_spend_micros=20_000,
        max_proof_time_ms=10_000,
        max_validation_time_ms=10_000,
        max_human_questions=1,
        max_repair_rounds=2,
        max_plan_branches=1,
        max_context_expansions=2,
        max_wall_time_ms=30_000,
        validation_reserve_ms=1_000,
    )


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
    policy: AutonomyPolicy | None = None,
    *,
    risk: RiskAssessment | None = None,
    **overrides: object,
) -> AutonomyEnvelope:
    bound_policy = policy or _policy()
    values: dict[str, object] = {
        "repository_id": "repo-1",
        "tree_id": "tree-1",
        "objective_id": "APMC-G000",
        "objective_revision": "objective-rev-1",
        "task_id": "APMC-013",
        "acceptance_criterion_ids": ("AC-repair",),
        "risk_assessment": risk or _risk(),
        "autonomy_level": AutonomyLevel.EXECUTE_REVERSIBLE,
        "cognitive_budget": _budget(),
        "allowed_paths": ("ipfs_accelerate_py/agent_supervisor/autonomy",),
        "allowed_symbols": ("AutonomyEnvelope", "RepairTier"),
        "required_test_ids": ("test-repair-controller",),
        "required_proof_ids": (),
        "authority_id": "operator-policy-authority",
        "policy_id": bound_policy.policy_id,
        "provider_usage_envelope_id": "provider-envelope-1",
        "resource_budget_id": "resource-budget-1",
        "human_escalation_policy_id": "human-policy-1",
        "expiry_ms": 10_000,
        "reversible": True,
        "blast_radius": {"max_files": 2},
    }
    values.update(overrides)
    return AutonomyEnvelope(**values)


class _FakeEngine(AutonomousRepairEngine):
    def __init__(self, repo_root: str | Path) -> None:
        super().__init__(repo_root=repo_root)
        self.runs = 0
        self.last_items: list[object] | None = None
        self.forced_model_calls = 0

    def run(self, items):  # type: ignore[no-untyped-def]
        self.runs += 1
        self.last_items = list(items)
        return AutonomousRepairReport(
            policy=self.policy.to_dict(),
            rows=[{"work_id": "work:1", "disposition": "single_path_ready"}],
            passed=False,
            model_call_count=self.forced_model_calls,
            llm_used=self.forced_model_calls > 0,
            recorded_at="2026-01-01T00:00:00+00:00",
            summary={"item_count": 1},
            notes=["test-double-over-existing-engine"],
        )


def _controller(tmp_path: Path, **overrides: object) -> AutonomousRepairController:
    engine = overrides.pop("engine", None) or _FakeEngine(tmp_path)
    values: dict[str, object] = {"engine": engine}
    values.update(overrides)
    return AutonomousRepairController(**values)


def _request(
    tmp_path: Path | None = None,
    *,
    policy: AutonomyPolicy | None = None,
    envelope: AutonomyEnvelope | None = None,
    **overrides: object,
) -> RepairRequest:
    bound_policy = policy or _policy()
    values: dict[str, object] = {
        "envelope": envelope or _envelope(bound_policy),
        "policy": bound_policy,
        "predicted_files": ("ipfs_accelerate_py/agent_supervisor/autonomy/contracts.py",),
        "predicted_symbols": ("AutonomyEnvelope",),
        "preferred_tier": RepairTier.DETERMINISTIC,
        "worktree_id": "worktree-repair-1",
        "context_reference_ids": ("context-ref-1",),
        "required_test_ids": ("test-repair-controller",),
        "rollback_plan_id": "rollback-plan-1",
        "validation_receipt_ids": ("test-repair-controller",),
        "work_items": ({"work_id": "work:1", "operation": "catalog.read"},),
    }
    values.update(overrides)
    return RepairRequest(**values)


def test_interface_is_versioned_and_engine_is_not_reimplemented() -> None:
    assert AUTONOMOUS_REPAIR_CONTROLLER_INTERFACE == "AutonomousRepairController@1"
    assert AutonomousRepairController.INTERFACE == AUTONOMOUS_REPAIR_CONTROLLER_INTERFACE
    assert repair_module.AutonomousRepairEngine is AutonomousRepairEngine
    tree = ast.parse(inspect.getsource(repair_module))
    defined = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    assert "AutonomousRepairEngine" not in defined
    assert "AutonomousRepairMaterializer" not in defined
    assert not any(name.endswith("Engine") for name in defined)
    source = inspect.getsource(repair_module)
    assert "class AutonomousRepairEngine" not in source
    assert "from ..autonomous_repair.engine import AutonomousRepairEngine" in source


def test_default_construction_uses_existing_engine(tmp_path: Path) -> None:
    controller = AutonomousRepairController(repo_root=tmp_path)
    assert type(controller.engine) is AutonomousRepairEngine
    assert controller.engine.__class__.__module__.endswith("autonomous_repair.engine")
    with pytest.raises(RepairControllerError, match="existing AutonomousRepairEngine"):
        AutonomousRepairController(engine=object())  # type: ignore[arg-type]


def test_deterministic_and_template_tiers_delegate_to_existing_engine(tmp_path: Path) -> None:
    engine = _FakeEngine(tmp_path)
    controller = _controller(tmp_path, engine=engine)
    deterministic = controller.run(_request(tmp_path, preferred_tier=RepairTier.DETERMINISTIC))
    assert deterministic.selected_tier is RepairTier.DETERMINISTIC
    assert deterministic.plan is not None
    assert deterministic.plan.repair_tier is RepairTier.DETERMINISTIC
    assert deterministic.plan.predicted_files == (
        "ipfs_accelerate_py/agent_supervisor/autonomy/contracts.py",
    )
    assert deterministic.plan.predicted_symbols == ("AutonomyEnvelope",)
    assert deterministic.engine_invoked
    assert engine.runs == 1
    assert deterministic.model_call_count == 0
    assert deterministic.receipt is not None
    assert not deterministic.receipt.authorizes_merge
    assert not deterministic.authorizes_merge

    template = controller.run(
        _request(tmp_path, preferred_tier=RepairTier.TEMPLATE_CONSTRAINED)
    )
    assert template.selected_tier is RepairTier.TEMPLATE_CONSTRAINED
    assert template.plan is not None
    assert template.plan.repair_tier is RepairTier.TEMPLATE_CONSTRAINED
    assert engine.runs == 2
    assert template.model_call_count == 0


def test_model_assisted_requires_isolated_worktree_and_predetermined_checks(
    tmp_path: Path,
) -> None:
    controller = _controller(tmp_path)
    missing_worktree = controller.admit(
        _request(
            tmp_path,
            preferred_tier=RepairTier.MODEL_ASSISTED_BOUNDED,
            worktree_id="",
        )
    )
    assert missing_worktree.disposition is RepairControllerDisposition.REJECTED
    assert "isolated_worktree_required" in missing_worktree.reason_codes

    missing_tests = controller.admit(
        _request(
            tmp_path,
            preferred_tier=RepairTier.MODEL_ASSISTED_BOUNDED,
            required_test_ids=(),
            envelope=_envelope(_policy(), required_test_ids=()),
        )
    )
    assert missing_tests.disposition is RepairControllerDisposition.REJECTED
    assert "predetermined_tests_required" in missing_tests.reason_codes


def test_model_assisted_invokes_injected_route_once_then_reuses_diagnosis(
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    def _model(request: RepairRequest, plan: object) -> dict[str, str]:
        calls.append(request.failure_signature)
        return {"diagnostic_receipt_id": "diagnostic:repeat"}

    controller = _controller(tmp_path, model_call=_model)
    request = _request(
        tmp_path,
        preferred_tier=RepairTier.MODEL_ASSISTED_BOUNDED,
        failure_signature="failure:identical-scope",
        diagnostic_receipt_id="diagnostic:repeat",
    )
    first = controller.run(request)
    second = controller.run(request)
    third = controller.run(request)

    assert first.selected_tier is RepairTier.MODEL_ASSISTED_BOUNDED
    assert first.plan is not None
    assert first.plan.worktree_id == "worktree-repair-1"
    assert first.model_call_count == 1
    assert not first.diagnostic_reused
    assert second.disposition is RepairControllerDisposition.IDENTICAL_FAILURE_BACKOFF
    assert second.diagnostic_reused
    assert second.model_call_count == 0
    assert second.backoff_milliseconds == 10
    assert third.model_call_count == 0
    assert third.backoff_milliseconds == 20
    assert calls == ["failure:identical-scope"]
    assert controller.model_call_count == 1
    assert second.receipt is not None
    assert second.receipt.rollback_receipt_id == "rollback-plan-1"


def test_scope_escape_self_edit_and_validator_policy_key_are_rejected(tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    escaped = controller.admit(
        _request(
            tmp_path,
            predicted_files=("docs/outside.md",),
            predicted_symbols=("heading",),
            envelope=_envelope(_policy(), allowed_symbols=("heading",)),
        )
    )
    assert escaped.disposition is RepairControllerDisposition.REJECTED
    assert "scope_escape" in escaped.reason_codes

    self_edit = controller.admit(
        _request(
            tmp_path,
            predicted_files=(SELF_EDIT_PATH,),
            predicted_symbols=("AutonomousRepairController",),
            envelope=_envelope(
                _policy(),
                allowed_paths=("ipfs_accelerate_py/agent_supervisor/autonomy",),
                allowed_symbols=("AutonomousRepairController",),
            ),
        )
    )
    assert self_edit.disposition is RepairControllerDisposition.REJECTED
    assert "self_edit" in self_edit.reason_codes

    validator = controller.admit(
        _request(
            tmp_path,
            predicted_files=(
                "ipfs_accelerate_py/agent_supervisor/validation/proposal_validation.py",
            ),
            predicted_symbols=("validate",),
            envelope=_envelope(
                _policy(),
                allowed_paths=("ipfs_accelerate_py/agent_supervisor/validation",),
                allowed_symbols=("validate",),
            ),
        )
    )
    assert validator.disposition is RepairControllerDisposition.REJECTED
    assert "validator_policy_key" in validator.reason_codes

    key = controller.admit(
        _request(
            tmp_path,
            predicted_files=("trusted_keys/prod.pem",),
            predicted_symbols=("trusted_key",),
            envelope=_envelope(
                _policy(),
                allowed_paths=("trusted_keys",),
                allowed_symbols=("trusted_key",),
            ),
        )
    )
    assert key.disposition is RepairControllerDisposition.REJECTED
    assert "validator_policy_key" in key.reason_codes
    assert protected_authority_reason("config/policy.json") == "validator_policy_key"
    assert protected_authority_reason("validator_policy_key") == "validator_policy_key"


def test_rollback_is_bound_and_receipts_cannot_authorize_merge(tmp_path: Path) -> None:
    result = _controller(tmp_path).run(_request(tmp_path))
    assert result.plan is not None
    assert result.plan.rollback_plan_id == "rollback-plan-1"
    assert result.receipt is not None
    assert result.receipt.authorizes_merge is False
    payload = result.receipt.to_dict()
    payload["authorizes_merge"] = True
    with pytest.raises(Exception, match="authorize merge"):
        type(result.receipt).from_dict(payload)


def test_r2_autonomous_merge_requires_every_low_risk_condition(tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    result = controller.run(_request(tmp_path))
    assert set(result.low_risk_conditions) == set(LOW_RISK_MERGE_CONDITIONS)
    assert all(result.low_risk_conditions.values())
    assert result.merge_disposition is RepairMergeDisposition.AUTONOMOUS_MERGE_CANDIDATE
    assert result.receipt is not None
    assert result.receipt.terminal_status is TerminalStatus.SUCCEEDED
    assert not result.authorizes_merge

    disabled = controller.run(
        _request(tmp_path, policy=_policy(autonomous_merge_enabled=False))
    )
    assert disabled.merge_disposition is RepairMergeDisposition.PROPOSAL_ONLY
    assert disabled.low_risk_conditions["policy_autonomous_merge_enabled"] is False

    observe = _policy(default_level=AutonomyLevel.OBSERVE_ONLY)
    observe_result = controller.admit(
        _request(
            tmp_path,
            policy=observe,
            envelope=_envelope(observe, autonomy_level=AutonomyLevel.OBSERVE_ONLY),
        )
    )
    assert observe_result.disposition is RepairControllerDisposition.REJECTED
    assert "autonomy_level_denied" in observe_result.reason_codes

    pending = controller.run(_request(tmp_path, validation_receipt_ids=()))
    assert pending.merge_disposition is RepairMergeDisposition.NOT_APPLICABLE
    assert pending.receipt is not None
    assert pending.receipt.terminal_status is TerminalStatus.PENDING

    modeled = AutonomousRepairController(
        engine=_FakeEngine(tmp_path),
        model_call=lambda request, plan: {"diagnostic_receipt_id": "diagnostic:model"},
    ).run(
        _request(
            tmp_path,
            preferred_tier=RepairTier.MODEL_ASSISTED_BOUNDED,
            failure_signature="failure:model-merge",
        )
    )
    assert modeled.merge_disposition is RepairMergeDisposition.PROPOSAL_ONLY
    assert modeled.low_risk_conditions["tier_not_model_assisted"] is False


def test_r3_repair_is_proposal_only_even_when_other_conditions_hold(tmp_path: Path) -> None:
    policy = _policy(default_level=AutonomyLevel.SELF_REPAIR_ISOLATED)
    risk = _risk(risk_class=RiskClass.R3_BOUNDED_REPOSITORY_MUTATION)
    envelope = _envelope(
        policy,
        risk=risk,
        autonomy_level=AutonomyLevel.SELF_REPAIR_ISOLATED,
        reversible=True,
    )
    result = _controller(tmp_path).run(_request(tmp_path, policy=policy, envelope=envelope))
    assert result.receipt is not None
    assert result.receipt.terminal_status is TerminalStatus.SUCCEEDED
    assert result.merge_disposition is RepairMergeDisposition.PROPOSAL_ONLY
    assert result.low_risk_conditions["risk_at_most_r2"] is False
    conjunction = evaluate_low_risk_merge_conjunction(
        request=_request(tmp_path, policy=policy, envelope=envelope),
        plan=result.plan,
        receipt=result.receipt,
    )
    assert conjunction["risk_at_most_r2"] is False
    assert not all(conjunction.values())


def test_suffix_contract_is_bound_and_cannot_authorize_effects(tmp_path: Path) -> None:
    suffix = PlanSuffixInvalidationReceipt(
        objective_id="APMC-G000",
        objective_revision="objective-rev-1",
        frozen_plan_id="plan:frozen-suffix",
        disposition=RecedingHorizonDisposition.PREFIX_PRESERVED,
    )
    bound = _controller(tmp_path).admit(_request(tmp_path, suffix_receipt=suffix))
    assert bound.disposition is RepairControllerDisposition.ADMITTED
    assert "suffix_contract_bound" in bound.reason_codes

    mismatched = _controller(tmp_path).admit(
        _request(
            tmp_path,
            suffix_receipt=PlanSuffixInvalidationReceipt(
                objective_id="APMC-G999",
                objective_revision="objective-rev-1",
                frozen_plan_id="plan:frozen-suffix",
                disposition=RecedingHorizonDisposition.PREFIX_PRESERVED,
            ),
        )
    )
    assert mismatched.disposition is RepairControllerDisposition.REJECTED
    assert "suffix_objective_mismatch" in mismatched.reason_codes


def test_engine_model_calls_are_forbidden_on_deterministic_tier(tmp_path: Path) -> None:
    engine = _FakeEngine(tmp_path)
    engine.forced_model_calls = 1
    result = _controller(tmp_path, engine=engine).run(_request(tmp_path))
    assert result.disposition is RepairControllerDisposition.REJECTED
    assert "engine_model_call_forbidden" in result.reason_codes
    assert engine.runs == 1
