from __future__ import annotations

import ast
from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.autonomous_repair.engine import AutonomousRepairEngine
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
    CONTROLLER_RELATIVE_PATH,
    LOW_RISK_MERGE_CONDITIONS,
    AutonomousRepairController,
    RepairAttempt,
    RepairControllerDisposition,
    RepairControllerError,
    RepairMergeDisposition,
    RepairPathViolation,
    classify_repair_path,
    is_protected_authority_path,
    is_self_edit_path,
    is_validator_policy_key_path,
)
from ipfs_accelerate_py.agent_supervisor.autonomy import repair_controller as repair_controller_module


SAFE_FILE = "ipfs_accelerate_py/agent_supervisor/autonomy/cognitive_budget.py"
SAFE_SYMBOL = "CognitiveBudget"


class SpyEngine(AutonomousRepairEngine):
    def __init__(self, repo_root: Path) -> None:
        super().__init__(repo_root=repo_root)
        self.calls: list[tuple[object, ...]] = []

    def run(self, items):  # type: ignore[override]
        self.calls.append(tuple(items))
        return super().run(items)


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
        "max_plan_branches": 2,
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


def _envelope(*, policy: AutonomyPolicy, **overrides: object) -> AutonomyEnvelope:
    values: dict[str, object] = {
        "repository_id": "repo-1",
        "tree_id": "tree-1",
        "objective_id": "APMC-G000",
        "objective_revision": "objective-rev-1",
        "task_id": "APMC-013",
        "acceptance_criterion_ids": ("AC-repair-scope",),
        "risk_assessment": RiskAssessment(
            risk_class=RiskClass.R2_REVERSIBLE_LOCAL,
            reversible=True,
            blast_radius_paths=("ipfs_accelerate_py/agent_supervisor/autonomy",),
            blast_radius_symbols=(SAFE_SYMBOL,),
            evidence_ids=("evidence-risk",),
            reason_codes=("bounded_local_change",),
        ),
        "autonomy_level": AutonomyLevel.EXECUTE_REVERSIBLE,
        "cognitive_budget": _budget(),
        "allowed_paths": ("ipfs_accelerate_py/agent_supervisor/autonomy",),
        "allowed_symbols": (SAFE_SYMBOL,),
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
        "predicted_files": (SAFE_FILE,),
        "predicted_symbols": (SAFE_SYMBOL,),
        "patch_envelope_id": envelope.envelope_id,
        "context_reference_ids": ("context-ref-1",),
        "required_test_ids": envelope.required_test_ids,
        "required_proof_ids": envelope.required_proof_ids,
        "worktree_id": "worktree-isolated-1",
        "allowed_paths": envelope.allowed_paths,
        "forbidden_symbols": ("trusted_keys",),
        "rollback_plan_id": "rollback-plan-1",
        "risk_class": envelope.risk_assessment.risk_class,
        "max_changed_files": 2,
        "max_changed_lines": 80,
    }
    values.update(overrides)
    return AutonomousRepairPlan(**values)


def _attempt(
    tmp_path: Path,
    *,
    policy: AutonomyPolicy | None = None,
    envelope: AutonomyEnvelope | None = None,
    plan: AutonomousRepairPlan | None = None,
    **overrides: object,
) -> RepairAttempt:
    policy = policy or _policy()
    envelope = envelope or _envelope(policy=policy)
    plan = plan or _plan(envelope=envelope)
    values: dict[str, object] = {
        "envelope": envelope,
        "policy": policy,
        "plan": plan,
        "validation_receipt_ids": ("validation-repair-1",),
        "changed_paths": plan.predicted_files,
        "changed_symbols": plan.predicted_symbols,
    }
    values.update(overrides)
    return RepairAttempt(**values)


def _controller(tmp_path: Path, **overrides: object) -> tuple[AutonomousRepairController, SpyEngine]:
    engine = overrides.pop("engine", None) or SpyEngine(tmp_path)
    controller = AutonomousRepairController(engine=engine, **overrides)
    return controller, engine


def test_interface_is_versioned_facade_over_existing_engine(tmp_path: Path) -> None:
    controller, engine = _controller(tmp_path)
    assert controller.interface == AUTONOMOUS_REPAIR_CONTROLLER_INTERFACE
    assert AutonomousRepairController.INTERFACE == "AutonomousRepairController@1"
    assert isinstance(controller.engine, AutonomousRepairEngine)
    assert controller.engine is engine
    assert AutonomousRepairEngine not in AutonomousRepairController.__mro__
    source_path = Path(repair_controller_module.__file__ or "")
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    class_names = {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}
    assert not {name for name in class_names if "Engine" in name}
    assert "from ..autonomous_repair.engine import AutonomousRepairEngine" in source
    with pytest.raises(RepairControllerError, match="AutonomousRepairEngine"):
        AutonomousRepairController(engine=object())  # type: ignore[arg-type]


def test_selects_deterministic_template_and_model_tiers(tmp_path: Path) -> None:
    controller, _engine = _controller(tmp_path)
    policy = _policy()
    envelope = _envelope(policy=policy)
    deterministic = controller.select_tier(
        RepairAttempt(envelope=envelope, policy=policy, plan=_plan(envelope=envelope))
    )
    template = controller.select_tier(
        RepairAttempt(
            envelope=envelope,
            policy=policy,
            plan=_plan(envelope=envelope, repair_tier=RepairTier.TEMPLATE_CONSTRAINED),
        )
    )
    model = controller.select_tier(
        RepairAttempt(
            envelope=envelope,
            policy=policy,
            plan=_plan(envelope=envelope, repair_tier=RepairTier.MODEL_ASSISTED_BOUNDED),
        )
    )
    assert deterministic is RepairTier.DETERMINISTIC
    assert template is RepairTier.TEMPLATE_CONSTRAINED
    assert model is RepairTier.MODEL_ASSISTED_BOUNDED
    missing_tests = controller.admit(
        RepairAttempt(
            envelope=envelope,
            policy=policy,
            plan=_plan(
                envelope=envelope,
                repair_tier=RepairTier.MODEL_ASSISTED_BOUNDED,
                required_test_ids=(),
            ),
        )
    )
    assert missing_tests.admitted is False
    assert "predetermined_tests_required" in missing_tests.reason_codes


def test_scope_escape_self_edit_and_validator_policy_key_are_rejected(
    tmp_path: Path,
) -> None:
    controller, engine = _controller(tmp_path)
    policy = _policy()
    envelope = _envelope(policy=policy)

    escape = controller.run(
        RepairAttempt(
            envelope=envelope,
            policy=policy,
            plan=_plan(envelope=envelope),
            changed_paths=("docs/outside.md",),
        )
    )
    assert escape.disposition is RepairControllerDisposition.REJECTED
    assert escape.admission.violation is RepairPathViolation.SCOPE_ESCAPE
    assert "scope_escape" in escape.reason_codes
    assert escape.engine_invoked is False
    assert engine.calls == []

    self_edit = controller.run(
        RepairAttempt(
            envelope=envelope,
            policy=policy,
            plan=_plan(envelope=envelope, predicted_files=(CONTROLLER_RELATIVE_PATH,)),
        )
    )
    assert self_edit.admission.violation is RepairPathViolation.SELF_EDIT
    assert "self_edit" in self_edit.reason_codes
    assert self_edit.engine_invoked is False

    key_path = "ipfs_accelerate_py/llm_router.py"
    wide = _envelope(
        policy=policy,
        allowed_paths=("ipfs_accelerate_py",),
        allowed_symbols=("route",),
    )
    key_edit = controller.run(
        RepairAttempt(
            envelope=wide,
            policy=policy,
            plan=_plan(
                envelope=wide,
                predicted_files=(key_path,),
                predicted_symbols=("route",),
                allowed_paths=("ipfs_accelerate_py",),
            ),
        )
    )
    assert key_edit.admission.violation is RepairPathViolation.VALIDATOR_POLICY_KEY
    assert "validator_policy_key_mutation" in key_edit.reason_codes
    assert key_edit.engine_invoked is False

    protected = controller.run(
        RepairAttempt(
            envelope=wide,
            policy=policy,
            plan=_plan(
                envelope=wide,
                predicted_files=(
                    "ipfs_accelerate_py/agent_supervisor/autonomous_repair/engine.py",
                ),
                predicted_symbols=("route",),
                allowed_paths=("ipfs_accelerate_py",),
            ),
        )
    )
    assert protected.admission.violation is RepairPathViolation.PROTECTED_AUTHORITY
    assert "protected_authority_path" in protected.reason_codes
    assert engine.calls == []


def test_identical_failures_do_not_repeat_model_calls(tmp_path: Path) -> None:
    model_calls: list[str] = []

    def invoker(attempt: RepairAttempt) -> dict[str, int]:
        model_calls.append(attempt.plan.plan_id)
        return {"model_call_count": 1}

    controller, engine = _controller(tmp_path, model_invoker=invoker)
    policy = _policy()
    envelope = _envelope(policy=policy)
    plan = _plan(envelope=envelope, repair_tier=RepairTier.MODEL_ASSISTED_BOUNDED)
    first = controller.run(
        RepairAttempt(
            envelope=envelope,
            policy=policy,
            plan=plan,
            validation_receipt_ids=(),
            failure_signature="failure:identical-v1",
            diagnostic_receipt_id="diagnostic:identical-v1",
        )
    )
    second = controller.run(
        RepairAttempt(
            envelope=envelope,
            policy=policy,
            plan=plan,
            validation_receipt_ids=(),
            failure_signature="failure:identical-v1",
            diagnostic_receipt_id="diagnostic:identical-v1",
        )
    )

    assert first.engine_invoked is True
    assert first.model_call_count == 1
    assert first.receipt.terminal_status is TerminalStatus.FAILED
    assert second.disposition is RepairControllerDisposition.BACKOFF
    assert second.diagnostic_reused is True
    assert second.model_call_count == 0
    assert second.engine_invoked is False
    assert second.backoff_milliseconds > 0
    assert model_calls == [plan.plan_id]
    assert len(engine.calls) == 1
    assert "diagnosis_reused" in second.reason_codes

    third = controller.run(
        RepairAttempt(
            envelope=envelope,
            policy=policy,
            plan=plan,
            validation_receipt_ids=(),
            failure_signature="failure:identical-v1",
            diagnostic_receipt_id="diagnostic:identical-v1",
            new_evidence_since_failure=True,
        )
    )
    assert third.engine_invoked is True
    assert third.model_call_count == 1
    assert model_calls == [plan.plan_id, plan.plan_id]


def test_autonomous_merge_requires_every_stated_low_risk_condition(tmp_path: Path) -> None:
    controller, engine = _controller(tmp_path)
    policy = _policy(autonomous_merge_enabled=True)
    envelope = _envelope(policy=policy)
    result = controller.run(_attempt(tmp_path, policy=policy, envelope=envelope))

    assert result.engine_invoked is True
    assert len(engine.calls) == 1
    assert result.disposition is RepairControllerDisposition.AUTONOMOUS_MERGE
    assert result.merge.eligible is True
    assert result.merge.disposition is RepairMergeDisposition.AUTONOMOUS_MERGE
    assert result.merge.authorizes_merge is False
    assert result.authorizes_merge is False
    assert result.receipt.authorizes_merge is False
    assert result.merge.missing_conditions == ()
    assert result.merge.satisfied_conditions == LOW_RISK_MERGE_CONDITIONS
    assert set(result.merge.to_dict()["satisfied_conditions"]) == set(LOW_RISK_MERGE_CONDITIONS)

    disabled = _policy(autonomous_merge_enabled=False)
    disabled_envelope = _envelope(policy=disabled)
    missing = controller.merge_eligibility(
        _attempt(tmp_path, policy=disabled, envelope=disabled_envelope)
    )
    assert missing.eligible is False
    assert "policy_autonomous_merge_enabled" in missing.missing_conditions
    assert missing.disposition is not RepairMergeDisposition.AUTONOMOUS_MERGE


def test_r3_repair_is_proposal_even_when_other_conditions_hold(tmp_path: Path) -> None:
    controller, _engine = _controller(tmp_path)
    policy = _policy(default_level=AutonomyLevel.SELF_REPAIR_ISOLATED)
    envelope = _envelope(
        policy=policy,
        autonomy_level=AutonomyLevel.SELF_REPAIR_ISOLATED,
        risk_assessment=RiskAssessment(
            risk_class=RiskClass.R3_BOUNDED_REPOSITORY_MUTATION,
            reversible=True,
            blast_radius_paths=("ipfs_accelerate_py/agent_supervisor/autonomy",),
            blast_radius_symbols=(SAFE_SYMBOL,),
            evidence_ids=("evidence-risk",),
            reason_codes=("bounded_repository_mutation",),
        ),
    )
    result = controller.run(_attempt(tmp_path, policy=policy, envelope=envelope))
    assert result.disposition is RepairControllerDisposition.PROPOSAL
    assert result.merge.eligible is False
    assert result.merge.disposition is RepairMergeDisposition.PROPOSAL
    assert "risk_class_r2_or_lower" in result.merge.missing_conditions
    assert "r3_proposal" in result.reason_codes
    assert result.authorizes_merge is False


def test_rejected_repairs_do_not_invoke_engine_or_authorize_merge(tmp_path: Path) -> None:
    controller, engine = _controller(tmp_path)
    policy = _policy()
    envelope = _envelope(policy=policy)
    result = controller.run(
        RepairAttempt(
            envelope=envelope,
            policy=policy,
            plan=_plan(envelope=envelope, predicted_files=(CONTROLLER_RELATIVE_PATH,)),
            validation_receipt_ids=("validation-repair-1",),
        )
    )
    assert result.disposition is RepairControllerDisposition.REJECTED
    assert result.engine_invoked is False
    assert engine.calls == []
    assert result.merge.eligible is False
    assert result.receipt.authorizes_merge is False


def test_path_classifiers_cover_authority_surfaces() -> None:
    allowed = ("ipfs_accelerate_py/agent_supervisor/autonomy",)
    assert is_self_edit_path(CONTROLLER_RELATIVE_PATH)
    assert classify_repair_path(CONTROLLER_RELATIVE_PATH, allowed_paths=allowed) is (
        RepairPathViolation.SELF_EDIT
    )
    assert is_validator_policy_key_path(
        "config/agent_supervisor_autonomous_meta_controller_scheduler.json"
    )
    assert is_protected_authority_path(
        "ipfs_accelerate_py/agent_supervisor/autonomous_repair/engine.py"
    )
    assert (
        classify_repair_path(SAFE_FILE, allowed_paths=allowed) is RepairPathViolation.NONE
    )


def test_source_edit_operator_is_rejected_outside_envelope_and_does_not_mutate(
    tmp_path: Path,
) -> None:
    controller, engine = _controller(tmp_path)
    policy = _policy()
    envelope = _envelope(policy=policy)
    result = controller.run(
        RepairAttempt(
            envelope=envelope,
            policy=policy,
            plan=_plan(envelope=envelope),
            source_edit_operator={
                "operator_id": "source-edit:escape",
                "owner_root": str(tmp_path.resolve()),
                "relative_path": "docs/outside.md",
                "old_digest": "sha256:old",
                "new_digest": "sha256:new",
                "old_bytes_b64": "b2xk",
                "new_bytes_b64": "bmV3",
                "forward_diff": "--- sha256:old\n+++ sha256:new",
                "inverse_diff": "--- sha256:new\n+++ sha256:old",
                "disposition": "validation_pending",
                "admitted": True,
                "kind": "replace_exact_bytes",
            },
            apply_source_edit=True,
        )
    )
    assert result.disposition is RepairControllerDisposition.REJECTED
    assert result.admission.violation is RepairPathViolation.SCOPE_ESCAPE
    assert result.engine_invoked is False
    assert result.authorizes_merge is False
    assert engine.calls == []
    assert not (tmp_path / "docs" / "outside.md").exists()
