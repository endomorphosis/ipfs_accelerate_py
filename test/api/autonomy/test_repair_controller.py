from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.autonomous_repair.contracts import (
    AUTONOMOUS_REPAIR_INTERFACE,
    AutonomousRepairReport,
)
from ipfs_accelerate_py.agent_supervisor.autonomous_repair.engine import AutonomousRepairEngine
from ipfs_accelerate_py.agent_supervisor.autonomy.contracts import (
    AutonomousRepairPlan,
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
    LOW_RISK_MERGE_CONDITIONS,
    PROTECTED_AUTHORITY_PATHS,
    AutonomousRepairController,
    MergeDisposition,
    RepairControllerDisposition,
    RepairControllerError,
    RepairRequest,
    is_protected_authority_path,
    is_self_edit_path,
    is_validator_policy_key_path,
    merge_disposition_for,
)
from ipfs_accelerate_py.agent_supervisor.autonomy.receding_horizon import (
    PlanSuffixInvalidationReceipt,
    RecedingHorizonDisposition,
    RecedingHorizonEvidenceKind,
)


TARGET_FILE = "ipfs_accelerate_py/agent_supervisor/autonomy/example_repair_target.py"
TARGET_SYMBOL = "ExampleRepairTarget"


class RecordingEngine(AutonomousRepairEngine):
    """Existing engine subclass used only to observe facade delegation."""

    def __init__(self, repo_root: str | Path) -> None:
        super().__init__(repo_root=repo_root)
        self.runs: list[list[object]] = []

    def run(self, items):  # type: ignore[no-untyped-def]
        self.runs.append(list(items))
        return AutonomousRepairReport(
            policy={"domain": "agent_supervisor"},
            rows=[],
            passed=False,
            model_call_count=0,
            llm_used=False,
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
        "proof_reserve_ms": 1_000,
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


def _envelope(*, policy: AutonomyPolicy, **overrides: object) -> AutonomyEnvelope:
    values: dict[str, object] = {
        "repository_id": "repo-1",
        "tree_id": "tree-1",
        "objective_id": "APMC-G000",
        "objective_revision": "objective-rev-1",
        "task_id": "APMC-013",
        "acceptance_criterion_ids": ("AC-bounded-repair",),
        "risk_assessment": RiskAssessment(
            risk_class=RiskClass.R2_REVERSIBLE_LOCAL,
            reversible=True,
            blast_radius_paths=("ipfs_accelerate_py/agent_supervisor/autonomy",),
            blast_radius_symbols=(TARGET_SYMBOL,),
            evidence_ids=("evidence-risk",),
            reason_codes=("bounded_local_change",),
        ),
        "autonomy_level": AutonomyLevel.EXECUTE_REVERSIBLE,
        "cognitive_budget": _budget(),
        "allowed_paths": ("ipfs_accelerate_py/agent_supervisor/autonomy",),
        "allowed_symbols": (TARGET_SYMBOL, "AutonomousRepairController"),
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
        "predicted_files": (TARGET_FILE,),
        "predicted_symbols": (TARGET_SYMBOL,),
        "patch_envelope_id": envelope.envelope_id,
        "context_reference_ids": ("context-ref-1",),
        "required_test_ids": ("test-repair-controller",),
        "required_proof_ids": (),
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


def _receipt(*, plan: AutonomousRepairPlan, envelope: AutonomyEnvelope, **overrides: object) -> AutonomousRepairReceipt:
    values: dict[str, object] = {
        "plan_id": plan.plan_id,
        "envelope_id": envelope.envelope_id,
        "terminal_status": TerminalStatus.SUCCEEDED,
        "changed_paths": (TARGET_FILE,),
        "validation_receipt_ids": ("test-repair-controller",),
        "proof_receipt_ids": (),
        "adversarial_assurance_receipt_ids": ("assurance-1",),
        "authorizes_merge": False,
    }
    values.update(overrides)
    return AutonomousRepairReceipt(**values)


_UNSET = object()


def _request(
    tmp_path: Path,
    *,
    policy: AutonomyPolicy | None = None,
    envelope: AutonomyEnvelope | None = None,
    plan: object = _UNSET,
    **overrides: object,
) -> RepairRequest:
    bound_policy = policy or _policy()
    bound_envelope = envelope or _envelope(policy=bound_policy)
    if "plan" in overrides:
        bound_plan = overrides.pop("plan")
    elif plan is _UNSET:
        bound_plan = _plan(envelope=bound_envelope)
    else:
        bound_plan = plan
    values: dict[str, object] = {
        "envelope": bound_envelope,
        "policy": bound_policy,
        "predicted_files": (TARGET_FILE,),
        "predicted_symbols": (TARGET_SYMBOL,),
        "requested_tier": RepairTier.DETERMINISTIC,
        "plan": bound_plan,
        "context_reference_ids": ("context-ref-1",),
        "required_test_ids": ("test-repair-controller",),
        "required_proof_ids": (),
        "worktree_id": "worktree-isolated-1",
        "rollback_plan_id": "rollback-plan-1",
        "forbidden_symbols": ("trusted_keys",),
        "context_sufficient": True,
        "isolated_worktree": True,
        "validation_receipt_ids": ("test-repair-controller",),
        "adversarial_assurance_receipt_ids": ("assurance-1",),
        "changed_paths": (TARGET_FILE,),
    }
    values.update(overrides)
    return RepairRequest(**values)


def _controller(tmp_path: Path, **overrides: object) -> AutonomousRepairController:
    engine = overrides.pop("engine", RecordingEngine(tmp_path))
    values: dict[str, object] = {"engine": engine}
    values.update(overrides)
    return AutonomousRepairController(**values)  # type: ignore[arg-type]


def test_interface_is_versioned_and_engine_authority_is_reused(tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    assert AUTONOMOUS_REPAIR_CONTROLLER_INTERFACE == "AutonomousRepairController@1"
    assert controller.interface == AUTONOMOUS_REPAIR_CONTROLLER_INTERFACE
    assert controller.engine_interface == AUTONOMOUS_REPAIR_INTERFACE
    assert isinstance(controller.engine, AutonomousRepairEngine)
    assert controller.engine.run.__qualname__.startswith("AutonomousRepairEngine") or isinstance(
        controller.engine, RecordingEngine
    )


def test_controller_rejects_a_second_repair_engine(tmp_path: Path) -> None:
    class OtherEngine:
        def run(self, items):  # type: ignore[no-untyped-def]
            return items

    with pytest.raises(RepairControllerError, match="AutonomousRepairEngine"):
        AutonomousRepairController(engine=OtherEngine())  # type: ignore[arg-type]

    module_path = Path(inspect.getsourcefile(AutonomousRepairController) or "")
    source = module_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    engine_classes = [
        node.name
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name.endswith("RepairEngine")
    ]
    assert engine_classes == []
    assert "class AutonomousRepairEngine" not in source
    imported = [
        alias.name
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    ]
    assert "AutonomousRepairEngine" in imported
    assert "DecisionRuntime" in imported
    assert "AdmittedSourceEditOperator" in imported


def test_selects_deterministic_template_and_model_tiers(tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    policy = _policy()
    envelope = _envelope(policy=policy)
    deterministic = controller.repair(
        _request(tmp_path, policy=policy, envelope=envelope, requested_tier=RepairTier.DETERMINISTIC)
    )
    template = controller.repair(
        _request(
            tmp_path,
            policy=policy,
            envelope=envelope,
            requested_tier=RepairTier.TEMPLATE_CONSTRAINED,
            plan=_plan(envelope=envelope, repair_tier=RepairTier.TEMPLATE_CONSTRAINED),
        )
    )
    model = controller.repair(
        _request(
            tmp_path,
            policy=policy,
            envelope=envelope,
            requested_tier=RepairTier.MODEL_ASSISTED_BOUNDED,
            plan=_plan(envelope=envelope, repair_tier=RepairTier.MODEL_ASSISTED_BOUNDED),
        )
    )
    assert deterministic.selected_tier is RepairTier.DETERMINISTIC
    assert template.selected_tier is RepairTier.TEMPLATE_CONSTRAINED
    assert model.selected_tier is RepairTier.MODEL_ASSISTED_BOUNDED
    assert controller.select_tier(
        _request(tmp_path, policy=policy, envelope=envelope, requested_tier=RepairTier.DETERMINISTIC)
    ) is RepairTier.DETERMINISTIC


def test_model_assisted_requires_exact_scope_isolation_and_checks(tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    policy = _policy()
    envelope = _envelope(policy=policy)
    missing_worktree = controller.repair(
        _request(
            tmp_path,
            policy=policy,
            envelope=envelope,
            requested_tier=RepairTier.MODEL_ASSISTED_BOUNDED,
            plan=None,
            worktree_id="",
            isolated_worktree=False,
            context_sufficient=True,
            context_reference_ids=("context-ref-1",),
            required_test_ids=("test-repair-controller",),
            rollback_plan_id="rollback-plan-1",
        )
    )
    missing_context = controller.repair(
        _request(
            tmp_path,
            policy=policy,
            envelope=envelope,
            requested_tier=RepairTier.MODEL_ASSISTED_BOUNDED,
            plan=None,
            worktree_id="worktree-isolated-1",
            isolated_worktree=True,
            context_sufficient=False,
            context_reference_ids=(),
            required_test_ids=("test-repair-controller",),
            rollback_plan_id="rollback-plan-1",
        )
    )
    missing_tests = controller.repair(
        _request(
            tmp_path,
            policy=policy,
            envelope=envelope,
            requested_tier=RepairTier.MODEL_ASSISTED_BOUNDED,
            plan=None,
            worktree_id="worktree-isolated-1",
            isolated_worktree=True,
            context_sufficient=True,
            context_reference_ids=("context-ref-1",),
            required_test_ids=(),
            rollback_plan_id="rollback-plan-1",
        )
    )
    assert missing_worktree.disposition is RepairControllerDisposition.REJECTED_MISSING_WORKTREE
    assert missing_context.disposition is RepairControllerDisposition.REJECTED_INSUFFICIENT_CONTEXT
    assert missing_tests.disposition is RepairControllerDisposition.REJECTED_MISSING_CHECKS
    assert "isolated_worktree_required" in missing_worktree.reason_codes
    assert "sufficient_context_required" in missing_context.reason_codes
    assert "predetermined_tests_required" in missing_tests.reason_codes


def test_scope_escape_is_rejected(tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    result = controller.repair(
        _request(
            tmp_path,
            predicted_files=("docs/outside.md",),
            changed_paths=("docs/outside.md",),
            plan=None,
        )
    )
    assert result.disposition is RepairControllerDisposition.REJECTED_SCOPE_ESCAPE
    assert "scope_escape" in result.reason_codes
    assert result.authorizes_effect is False
    assert result.authorizes_merge is False


def test_self_edit_and_protected_authority_paths_are_rejected(tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    self_edit = controller.repair(
        _request(
            tmp_path,
            predicted_files=("ipfs_accelerate_py/agent_supervisor/autonomy/repair_controller.py",),
            changed_paths=("ipfs_accelerate_py/agent_supervisor/autonomy/repair_controller.py",),
            predicted_symbols=("AutonomousRepairController",),
            plan=None,
        )
    )
    engine_edit = controller.repair(
        _request(
            tmp_path,
            predicted_files=("ipfs_accelerate_py/agent_supervisor/autonomous_repair/engine.py",),
            changed_paths=("ipfs_accelerate_py/agent_supervisor/autonomous_repair/engine.py",),
            predicted_symbols=("AutonomousRepairController",),
            plan=None,
        )
    )
    policy_key = controller.repair(
        _request(
            tmp_path,
            predicted_files=("config/validator_policy_key.json",),
            changed_paths=("config/validator_policy_key.json",),
            plan=None,
        )
    )
    assert self_edit.disposition is RepairControllerDisposition.REJECTED_SELF_EDIT
    assert engine_edit.disposition is RepairControllerDisposition.REJECTED_SELF_EDIT
    assert policy_key.disposition is RepairControllerDisposition.REJECTED_VALIDATOR_POLICY_KEY
    assert is_self_edit_path("ipfs_accelerate_py/agent_supervisor/autonomy/repair_controller.py")
    assert is_protected_authority_path("ipfs_accelerate_py/agent_supervisor/autonomous_repair/engine.py")
    assert is_validator_policy_key_path("config/trusted_keys.json")
    assert "ipfs_accelerate_py/agent_supervisor/autonomy/repair_controller.py" in PROTECTED_AUTHORITY_PATHS


def test_forbidden_symbol_is_rejected(tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    policy = _policy()
    envelope = _envelope(policy=policy, allowed_symbols=("trusted_keys", TARGET_SYMBOL))
    result = controller.repair(
        _request(
            tmp_path,
            policy=policy,
            envelope=envelope,
            predicted_symbols=("trusted_keys",),
            plan=None,
        )
    )
    assert result.disposition is RepairControllerDisposition.REJECTED_SCOPE_ESCAPE
    assert "forbidden_symbol" in result.reason_codes


def test_identical_failures_do_not_repeat_model_calls(tmp_path: Path) -> None:
    calls: list[str] = []

    def model_caller(request: RepairRequest) -> dict[str, str]:
        calls.append(request.failure_signature)
        return {"diagnostic_receipt_id": "diagnostic:reuse"}

    engine = RecordingEngine(tmp_path)
    controller = AutonomousRepairController(
        engine=engine,
        model_caller=model_caller,
        max_identical_failures=3,
        base_backoff_milliseconds=10,
    )
    policy = _policy()
    envelope = _envelope(policy=policy)
    request = _request(
        tmp_path,
        policy=policy,
        envelope=envelope,
        requested_tier=RepairTier.MODEL_ASSISTED_BOUNDED,
        plan=_plan(envelope=envelope, repair_tier=RepairTier.MODEL_ASSISTED_BOUNDED),
        failure_signature="failure:syntax-repeat",
        diagnostic_receipt_id="diagnostic:reuse",
        execute=True,
        validation_receipt_ids=(),
        adversarial_assurance_receipt_ids=(),
    )
    first = controller.repair(request)
    second = controller.repair(request)
    third = controller.repair(request)
    exhausted = controller.repair(request)

    assert first.model_call_count == 1
    assert len(calls) == 1
    assert second.disposition is RepairControllerDisposition.IDENTICAL_FAILURE_BACKOFF
    assert second.diagnostic_reused is True
    assert second.model_call_count == 0
    assert third.diagnostic_reused is True
    assert len(calls) == 1
    assert exhausted.disposition is RepairControllerDisposition.IDENTICAL_FAILURE_EXHAUSTED
    assert controller.model_call_count == 1
    assert engine.runs  # first attempt delegated to the existing engine


def test_repair_delegates_to_existing_engine_and_never_authorizes_effect(tmp_path: Path) -> None:
    engine = RecordingEngine(tmp_path)
    controller = AutonomousRepairController(engine=engine)
    result = controller.repair(_request(tmp_path, execute=True))
    assert result.disposition in {
        RepairControllerDisposition.ENGINE_DELEGATED,
        RepairControllerDisposition.MERGE_ELIGIBLE,
        RepairControllerDisposition.PROPOSAL_ONLY,
        RepairControllerDisposition.ADMITTED,
    }
    assert result.engine_interface == AUTONOMOUS_REPAIR_INTERFACE
    assert "engine:" + AUTONOMOUS_REPAIR_INTERFACE in result.reason_codes
    assert len(engine.runs) == 1
    assert result.authorizes_effect is False
    assert result.authorizes_merge is False
    assert result.to_dict()["authorizes_merge"] is False


def test_r2_autonomous_merge_requires_every_low_risk_condition(tmp_path: Path) -> None:
    policy = _policy(autonomous_merge_enabled=True, default_level=AutonomyLevel.EXECUTE_REVERSIBLE)
    envelope = _envelope(policy=policy)
    plan = _plan(envelope=envelope)
    receipt = _receipt(plan=plan, envelope=envelope)
    merge, satisfied, missing = merge_disposition_for(
        policy=policy,
        envelope=envelope,
        plan=plan,
        receipt=receipt,
        isolated_worktree=True,
    )
    assert merge is MergeDisposition.AUTONOMOUS_MERGE_ELIGIBLE
    assert satisfied == LOW_RISK_MERGE_CONDITIONS
    assert missing == ()

    controller = _controller(tmp_path)
    result = controller.repair(
        _request(tmp_path, policy=policy, envelope=envelope, plan=plan, isolated_worktree=True)
    )
    assert result.merge_disposition is MergeDisposition.AUTONOMOUS_MERGE_ELIGIBLE
    assert result.disposition is RepairControllerDisposition.MERGE_ELIGIBLE
    assert result.missing_merge_conditions == ()
    assert set(result.satisfied_merge_conditions) == set(LOW_RISK_MERGE_CONDITIONS)
    assert result.authorizes_merge is False


def test_missing_any_low_risk_condition_denies_autonomous_merge(tmp_path: Path) -> None:
    policy = _policy(autonomous_merge_enabled=True)
    envelope = _envelope(policy=policy)
    plan = _plan(envelope=envelope)
    receipt = _receipt(plan=plan, envelope=envelope)
    disabled = _policy(autonomous_merge_enabled=False)
    disabled_envelope = _envelope(policy=disabled)
    disabled_plan = _plan(envelope=disabled_envelope)
    disabled_receipt = _receipt(plan=disabled_plan, envelope=disabled_envelope)
    merge, _satisfied, missing = merge_disposition_for(
        policy=disabled,
        envelope=disabled_envelope,
        plan=disabled_plan,
        receipt=disabled_receipt,
        isolated_worktree=True,
    )
    assert merge is MergeDisposition.NOT_ELIGIBLE
    assert "autonomous_merge_enabled" in missing

    no_tests = merge_disposition_for(
        policy=policy,
        envelope=envelope,
        plan=plan,
        receipt=_receipt(plan=plan, envelope=envelope, validation_receipt_ids=("other-test",)),
        isolated_worktree=True,
    )
    assert no_tests[0] is MergeDisposition.NOT_ELIGIBLE
    assert "predetermined_tests_current" in no_tests[2]

    no_worktree = merge_disposition_for(
        policy=policy,
        envelope=envelope,
        plan=plan,
        receipt=receipt,
        isolated_worktree=False,
    )
    assert no_worktree[0] is MergeDisposition.NOT_ELIGIBLE
    assert "isolated_worktree" in no_worktree[2]
    controller_denied = _controller(tmp_path).repair(
        _request(tmp_path, policy=policy, envelope=envelope, plan=plan, isolated_worktree=False)
    )
    assert controller_denied.merge_disposition is MergeDisposition.NOT_ELIGIBLE
    assert "isolated_worktree" in controller_denied.missing_merge_conditions


def test_r3_repair_is_proposal_only(tmp_path: Path) -> None:
    policy = _policy(
        autonomous_merge_enabled=True,
        default_level=AutonomyLevel.SELF_REPAIR_ISOLATED,
    )
    envelope = _envelope(
        policy=policy,
        autonomy_level=AutonomyLevel.SELF_REPAIR_ISOLATED,
        risk_assessment=RiskAssessment(
            risk_class=RiskClass.R3_BOUNDED_REPOSITORY_MUTATION,
            reversible=True,
            blast_radius_paths=("ipfs_accelerate_py/agent_supervisor/autonomy",),
            blast_radius_symbols=(TARGET_SYMBOL,),
            evidence_ids=("evidence-risk",),
            reason_codes=("bounded_repository_mutation",),
        ),
    )
    plan = _plan(envelope=envelope, risk_class=RiskClass.R3_BOUNDED_REPOSITORY_MUTATION)
    result = _controller(tmp_path).repair(
        _request(tmp_path, policy=policy, envelope=envelope, plan=plan)
    )
    assert result.merge_disposition is MergeDisposition.PROPOSAL_ONLY
    assert result.disposition is RepairControllerDisposition.PROPOSAL_ONLY
    assert result.authorizes_merge is False


def test_suffix_identical_failure_reuses_diagnosis_without_engine_run(tmp_path: Path) -> None:
    engine = RecordingEngine(tmp_path)
    controller = AutonomousRepairController(engine=engine)
    suffix = PlanSuffixInvalidationReceipt(
        objective_id="APMC-G000",
        objective_revision="objective-rev-1",
        frozen_plan_id="plan:frozen",
        disposition=RecedingHorizonDisposition.IDENTICAL_FAILURE_EXHAUSTED,
        evidence_kind=RecedingHorizonEvidenceKind.FAILED_TEST,
        evidence_id="evidence:repeat",
        diagnostic_reused=True,
        backoff_milliseconds=40,
        reason_codes=("identical_failure_exhausted",),
    )
    result = controller.repair(_request(tmp_path, suffix_receipt=suffix, execute=True))
    assert result.disposition is RepairControllerDisposition.IDENTICAL_FAILURE_EXHAUSTED
    assert result.diagnostic_reused is True
    assert engine.runs == []


def test_rollback_plan_is_bound_on_admitted_repair(tmp_path: Path) -> None:
    result = _controller(tmp_path).admit(_request(tmp_path))
    assert result.plan is not None
    assert result.plan.rollback_plan_id == "rollback-plan-1"
    assert result.plan.required_test_ids == ("test-repair-controller",)
    assert result.plan.predicted_files == (TARGET_FILE,)
    assert result.plan.predicted_symbols == (TARGET_SYMBOL,)
    assert result.plan.worktree_id == "worktree-isolated-1"


def test_result_refuses_prompts_and_merge_authority(tmp_path: Path) -> None:
    result = _controller(tmp_path).admit(_request(tmp_path))
    payload = result.to_dict()
    assert payload["authorizes_effect"] is False
    assert payload["authorizes_merge"] is False
    assert "prompt" not in payload
    assert result.engine_interface == AUTONOMOUS_REPAIR_INTERFACE


def test_admit_source_edit_requires_an_exact_admitted_operator(tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    missing = controller.admit_source_edit(_request(tmp_path))
    admitted = controller.admit_source_edit(
        _request(
            tmp_path,
            source_edit_operator={
                "operator_id": "source-edit:example",
                "owner_root": "/isolated/worktree",
                "relative_path": TARGET_FILE,
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
        )
    )
    assert missing.disposition is RepairControllerDisposition.REJECTED_SOURCE_EDIT
    assert "source_edit_operator_missing" in missing.reason_codes
    assert admitted.disposition is RepairControllerDisposition.ADMITTED
    assert admitted.source_edit_admitted is True
    assert admitted.materializer_invoked is False
    assert admitted.authorizes_effect is False
    assert admitted.authorizes_merge is False


def test_admit_source_edit_rejects_self_edit_policy_key_and_scope_escape(
    tmp_path: Path,
) -> None:
    controller = _controller(tmp_path)

    def _operator(relative_path: str) -> dict[str, object]:
        return {
            "operator_id": "source-edit:forbidden",
            "owner_root": "/isolated/worktree",
            "relative_path": relative_path,
            "old_digest": "sha256:old",
            "new_digest": "sha256:new",
            "old_bytes_b64": "b2xk",
            "new_bytes_b64": "bmV3",
            "forward_diff": "--- sha256:old\n+++ sha256:new",
            "inverse_diff": "--- sha256:new\n+++ sha256:old",
            "disposition": "validation_pending",
            "admitted": True,
            "kind": "replace_exact_bytes",
        }

    self_edit = controller.admit_source_edit(
        _request(
            tmp_path,
            source_edit_operator=_operator(
                "ipfs_accelerate_py/agent_supervisor/autonomy/repair_controller.py"
            ),
        )
    )
    policy_key = controller.admit_source_edit(
        _request(
            tmp_path,
            source_edit_operator=_operator("config/validator_policy_key.json"),
        )
    )
    escaped = controller.admit_source_edit(
        _request(tmp_path, source_edit_operator=_operator("docs/outside.md"))
    )
    assert self_edit.disposition is RepairControllerDisposition.REJECTED_SELF_EDIT
    assert policy_key.disposition is RepairControllerDisposition.REJECTED_VALIDATOR_POLICY_KEY
    assert escaped.disposition is RepairControllerDisposition.REJECTED_SCOPE_ESCAPE
    assert self_edit.authorizes_merge is False
    assert policy_key.authorizes_merge is False
    assert escaped.authorizes_merge is False
