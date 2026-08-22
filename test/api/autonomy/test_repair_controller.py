from __future__ import annotations

from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.autonomous_repair.contracts import (
    AutonomousRepairPolicy,
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
from ipfs_accelerate_py.agent_supervisor.autonomy import repair_controller as repair_controller_module
from ipfs_accelerate_py.agent_supervisor.autonomy.repair_controller import (
    AUTONOMOUS_REPAIR_CONTROLLER_INTERFACE,
    ENGINE_AUTHORITY,
    LOW_RISK_MERGE_CONDITIONS,
    AutonomousRepairController,
    MergeDisposition,
    RepairControllerDisposition,
    RepairControllerError,
    RepairControllerRequest,
    RepairControllerResult,
    classify_forbidden_paths,
    path_is_self_edit,
    path_is_validator_policy_key,
    select_repair_tier,
)


class _RecordingEngine(AutonomousRepairEngine):
    """Test double that remains the existing engine type."""

    def __init__(self, root: Path) -> None:
        super().__init__(
            repo_root=root,
            policy=AutonomousRepairPolicy(
                apply_ir_logic=False,
                apply_doctor=False,
                require_zero_model_calls=True,
                allow_code_edit_materialize=False,
            ),
        )
        self.runs: list[tuple[object, ...]] = []

    def run(self, items):  # type: ignore[override]
        self.runs.append(tuple(items))
        return AutonomousRepairReport(
            policy=self.policy.to_dict(),
            rows=[],
            passed=False,
            model_call_count=0,
            llm_used=False,
            summary={"source_edits_applied": 0},
            notes=["recording AutonomousRepairEngine"],
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
        "blast_radius_symbols": ("RepairController",),
        "evidence_ids": ("evidence-risk",),
        "reason_codes": ("bounded_local_change",),
    }
    values.update(overrides)
    return RiskAssessment(**values)


def _envelope(
    policy: AutonomyPolicy,
    *,
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
        "acceptance_criterion_ids": ("AC-repair",),
        "risk_assessment": assessed,
        "autonomy_level": AutonomyLevel.EXECUTE_REVERSIBLE,
        "cognitive_budget": _budget(),
        "allowed_paths": ("ipfs_accelerate_py/agent_supervisor/autonomy",),
        "allowed_symbols": ("AutonomousRepairController",),
        "required_test_ids": ("test-repair-controller",),
        "required_proof_ids": (),
        "authority_id": policy.authority_id,
        "policy_id": policy.policy_id,
        "provider_usage_envelope_id": "provider-envelope-1",
        "resource_budget_id": "resource-budget-1",
        "human_escalation_policy_id": policy.human_escalation_policy_id,
        "expiry_ms": 12_000,
        "reversible": assessed.reversible,
    }
    values.update(overrides)
    return AutonomyEnvelope(**values)


def _request(
    policy: AutonomyPolicy | None = None,
    envelope: AutonomyEnvelope | None = None,
    **overrides: object,
) -> RepairControllerRequest:
    bound_policy = policy or _policy()
    bound_envelope = envelope or _envelope(bound_policy)
    values: dict[str, object] = {
        "envelope": bound_envelope,
        "policy": bound_policy,
        "predicted_files": ("ipfs_accelerate_py/agent_supervisor/autonomy/semantic_memory.py",),
        "predicted_symbols": ("SemanticMemory",),
        "worktree_id": "worktree:isolated",
        "rollback_plan_id": "rollback:plan-1",
        "context_reference_ids": ("context:tree-1",),
        "changed_paths": ("ipfs_accelerate_py/agent_supervisor/autonomy/semantic_memory.py",),
        "changed_line_count": 12,
        "validation_receipt_ids": ("validation:pytest-repair",),
        "delegate_to_engine": False,
        "required_test_ids": bound_envelope.required_test_ids,
        "required_proof_ids": bound_envelope.required_proof_ids,
    }
    values.update(overrides)
    return RepairControllerRequest(**values)


def test_interface_is_versioned_and_engine_is_not_replaced(tmp_path: Path) -> None:
    engine = _RecordingEngine(tmp_path)
    controller = AutonomousRepairController(engine=engine, repo_root=tmp_path)
    source = Path(repair_controller_module.__file__).read_text(encoding="utf-8")

    assert controller.interface == AUTONOMOUS_REPAIR_CONTROLLER_INTERFACE
    assert AUTONOMOUS_REPAIR_CONTROLLER_INTERFACE == "AutonomousRepairController@1"
    assert controller.engine_type is AutonomousRepairEngine
    assert isinstance(controller.engine, AutonomousRepairEngine)
    assert controller.engine is engine
    assert "class AutonomousRepairEngine" not in source
    assert "from ..autonomous_repair.engine import" in source
    assert ENGINE_AUTHORITY in source
    with pytest.raises(RepairControllerError, match="AutonomousRepairEngine"):
        AutonomousRepairController(engine=object())  # type: ignore[arg-type]


def test_selects_deterministic_then_template_then_model() -> None:
    assert (
        select_repair_tier(
            requested=None,
            deterministic_available=True,
            template_available=True,
            context_sufficient=True,
            exact_files=True,
            exact_symbols=True,
            isolated_worktree=True,
            predetermined_checks=True,
        )
        is RepairTier.DETERMINISTIC
    )
    assert (
        select_repair_tier(
            requested=None,
            deterministic_available=False,
            template_available=True,
            context_sufficient=True,
            exact_files=True,
            exact_symbols=True,
            isolated_worktree=True,
            predetermined_checks=True,
        )
        is RepairTier.TEMPLATE_CONSTRAINED
    )
    assert (
        select_repair_tier(
            requested=None,
            deterministic_available=False,
            template_available=False,
            context_sufficient=True,
            exact_files=True,
            exact_symbols=True,
            isolated_worktree=True,
            predetermined_checks=True,
        )
        is RepairTier.MODEL_ASSISTED_BOUNDED
    )
    with pytest.raises(RepairControllerError, match="isolated worktree"):
        select_repair_tier(
            requested=RepairTier.MODEL_ASSISTED_BOUNDED,
            deterministic_available=False,
            template_available=False,
            context_sufficient=True,
            exact_files=True,
            exact_symbols=True,
            isolated_worktree=False,
            predetermined_checks=True,
        )


def test_deterministic_tier_delegates_to_existing_engine(tmp_path: Path) -> None:
    engine = _RecordingEngine(tmp_path)
    controller = AutonomousRepairController(engine=engine, repo_root=tmp_path)
    result = controller.run(_request(delegate_to_engine=True, requested_tier=RepairTier.DETERMINISTIC))

    assert result.disposition is RepairControllerDisposition.ENGINE_DELEGATED
    assert result.selected_tier is RepairTier.DETERMINISTIC
    assert result.engine_authority == ENGINE_AUTHORITY
    assert not result.creates_repair_engine
    assert not result.authorizes_effect
    assert not result.authorizes_merge
    assert result.receipt is not None
    assert result.receipt.authorizes_merge is False
    assert len(engine.runs) == 1
    assert result.engine_run_count == 1
    assert "no_second_repair_engine" in result.reason_codes
    restored = RepairControllerResult(
        disposition=result.disposition,
        selected_tier=result.selected_tier,
        merge_disposition=result.merge_disposition,
        merge_conditions=result.merge_conditions,
        reason_codes=result.reason_codes,
        plan=result.plan,
        receipt=result.receipt,
        engine_report=result.engine_report,
        model_call_count=result.model_call_count,
        engine_run_count=result.engine_run_count,
        suffix_receipt_id=result.suffix_receipt_id,
    )
    assert restored.result_id == result.result_id


def test_scope_escape_is_rejected(tmp_path: Path) -> None:
    controller = AutonomousRepairController(engine=_RecordingEngine(tmp_path), repo_root=tmp_path)
    result = controller.run(
        _request(
            predicted_files=("docs/outside.md",),
            changed_paths=("docs/outside.md",),
            predicted_symbols=("heading",),
        )
    )
    assert result.disposition is RepairControllerDisposition.REJECTED
    assert "scope_escape_rejected" in result.reason_codes
    assert result.merge_disposition is MergeDisposition.REJECTED
    assert len(controller.engine.runs) == 0  # type: ignore[attr-defined]


def test_sibling_repository_mutation_is_rejected(tmp_path: Path) -> None:
    policy = _policy()
    controller = AutonomousRepairController(engine=_RecordingEngine(tmp_path), repo_root=tmp_path)
    result = controller.run(
        _request(
            policy=policy,
            envelope=_envelope(policy, allowed_paths=("ipfs_datasets_py",)),
            predicted_files=("ipfs_datasets_py/datasets.py",),
            changed_paths=("ipfs_datasets_py/datasets.py",),
            predicted_symbols=("Dataset",),
        )
    )
    assert result.disposition is RepairControllerDisposition.REJECTED
    assert "sibling_repository_mutation_rejected" in result.reason_codes
    assert result.merge_disposition is MergeDisposition.REJECTED
    assert len(controller.engine.runs) == 0  # type: ignore[attr-defined]


def test_template_tier_delegates_to_existing_engine(tmp_path: Path) -> None:
    engine = _RecordingEngine(tmp_path)
    controller = AutonomousRepairController(engine=engine, repo_root=tmp_path)
    result = controller.run(
        _request(
            delegate_to_engine=True,
            requested_tier=RepairTier.TEMPLATE_CONSTRAINED,
            deterministic_available=False,
            template_available=True,
        )
    )
    assert result.disposition is RepairControllerDisposition.ENGINE_DELEGATED
    assert result.selected_tier is RepairTier.TEMPLATE_CONSTRAINED
    assert result.engine_authority == ENGINE_AUTHORITY
    assert not result.creates_repair_engine
    assert not result.authorizes_effect
    assert not result.authorizes_merge
    assert len(engine.runs) == 1
    assert "no_second_repair_engine" in result.reason_codes


def test_self_edit_and_validator_policy_key_mutation_are_rejected(tmp_path: Path) -> None:
    controller = AutonomousRepairController(engine=_RecordingEngine(tmp_path), repo_root=tmp_path)
    self_edit = controller.run(
        _request(
            predicted_files=(
                "ipfs_accelerate_py/agent_supervisor/autonomy/repair_controller.py",
            ),
            changed_paths=(
                "ipfs_accelerate_py/agent_supervisor/autonomy/repair_controller.py",
            ),
            predicted_symbols=("AutonomousRepairController",),
        )
    )
    validation_policy = _policy()
    key_edit = controller.run(
        _request(
            policy=validation_policy,
            envelope=_envelope(
                validation_policy,
                allowed_paths=("ipfs_accelerate_py/agent_supervisor/validation",),
            ),
            predicted_files=(
                "ipfs_accelerate_py/agent_supervisor/validation/project_dependency_preflight.py",
            ),
            changed_paths=(
                "ipfs_accelerate_py/agent_supervisor/validation/project_dependency_preflight.py",
            ),
            predicted_symbols=("preflight",),
        )
    )
    config_policy = _policy()
    policy_key = controller.run(
        _request(
            policy=config_policy,
            envelope=_envelope(config_policy, allowed_paths=("config",)),
            predicted_files=("config/policy.json",),
            changed_paths=("config/policy.json",),
            predicted_symbols=("trusted_keys",),
        )
    )

    assert path_is_self_edit("ipfs_accelerate_py/agent_supervisor/autonomous_repair/engine.py")
    assert path_is_validator_policy_key("ipfs_accelerate_py/agent_supervisor/proof/keys.pem")
    assert "self_edit_rejected" in classify_forbidden_paths(
        ("ipfs_accelerate_py/agent_supervisor/autonomy/repair_controller.py",)
    )
    assert self_edit.disposition is RepairControllerDisposition.REJECTED
    assert "self_edit_rejected" in self_edit.reason_codes
    assert key_edit.disposition is RepairControllerDisposition.REJECTED
    assert "validator_policy_key_mutation_rejected" in key_edit.reason_codes
    assert policy_key.disposition is RepairControllerDisposition.REJECTED
    assert "validator_policy_key_mutation_rejected" in policy_key.reason_codes
    assert controller.model_call_count == 0


def test_identical_failures_do_not_repeat_model_calls(tmp_path: Path) -> None:
    calls: list[str] = []

    def invoker(plan, envelope):
        calls.append(plan.plan_id)
        return {"failed": True, "diagnostic_receipt_id": "diag:repeat"}

    controller = AutonomousRepairController(
        engine=_RecordingEngine(tmp_path),
        repo_root=tmp_path,
        model_invoker=invoker,
        max_identical_failures=3,
        base_backoff_milliseconds=10,
        max_backoff_milliseconds=40,
    )
    request = _request(
        requested_tier=RepairTier.MODEL_ASSISTED_BOUNDED,
        deterministic_available=False,
        template_available=False,
        failed=True,
        failure_signature="fail:identical",
        diagnostic_receipt_id="diag:repeat",
        validation_receipt_ids=(),
        changed_line_count=0,
        changed_paths=(),
    )
    first = controller.run(request)
    second = controller.run(request)
    third = controller.run(request)
    fourth = controller.run(request)

    assert first.disposition is RepairControllerDisposition.ROLLBACK_REQUIRED
    assert first.model_call_count == 1
    assert len(calls) == 1
    assert second.disposition is RepairControllerDisposition.IDENTICAL_FAILURE_REUSED
    assert second.diagnostic_reused
    assert second.backoff_milliseconds == 20
    assert "model_call_suppressed" in second.reason_codes
    assert third.diagnostic_reused
    assert fourth.disposition is RepairControllerDisposition.IDENTICAL_FAILURE_EXHAUSTED
    assert len(calls) == 1
    assert controller.model_call_count == 1
    assert second.receipt is not None
    assert second.receipt.diagnostic_receipt_id == "diag:repeat"


def test_r2_autonomous_merge_requires_every_stated_condition(tmp_path: Path) -> None:
    controller = AutonomousRepairController(engine=_RecordingEngine(tmp_path), repo_root=tmp_path)
    eligible = controller.run(_request(requested_tier=RepairTier.DETERMINISTIC))
    assert set(eligible.merge_conditions) == set(LOW_RISK_MERGE_CONDITIONS)
    assert all(eligible.merge_conditions.values())
    assert eligible.merge_disposition is MergeDisposition.R2_AUTONOMOUS_MERGE_ELIGIBLE
    assert eligible.receipt is not None
    assert eligible.receipt.authorizes_merge is False
    assert eligible.authorizes_merge is False
    assert eligible.receipt.terminal_status is TerminalStatus.SUCCEEDED

    missing_validation = controller.run(
        _request(requested_tier=RepairTier.DETERMINISTIC, validation_receipt_ids=())
    )
    assert missing_validation.merge_disposition is MergeDisposition.NOT_ELIGIBLE
    assert missing_validation.merge_conditions["validation_receipts_present"] is False
    assert missing_validation.merge_conditions["repair_succeeded"] is False

    merge_disabled = controller.run(
        _request(
            policy=_policy(autonomous_merge_enabled=False),
            requested_tier=RepairTier.DETERMINISTIC,
        )
    )
    assert merge_disabled.merge_disposition is MergeDisposition.NOT_ELIGIBLE
    assert merge_disabled.merge_conditions["policy_autonomous_merge_enabled"] is False


def test_r3_repair_is_a_proposal_even_when_other_conditions_hold(tmp_path: Path) -> None:
    policy = _policy(default_level=AutonomyLevel.SELF_REPAIR_ISOLATED)
    risk = _risk(risk_class=RiskClass.R3_BOUNDED_REPOSITORY_MUTATION)
    envelope = _envelope(
        policy,
        risk=risk,
        autonomy_level=AutonomyLevel.SELF_REPAIR_ISOLATED,
        reversible=True,
    )
    controller = AutonomousRepairController(engine=_RecordingEngine(tmp_path), repo_root=tmp_path)
    result = controller.run(
        _request(policy=policy, envelope=envelope, requested_tier=RepairTier.DETERMINISTIC)
    )
    assert result.merge_conditions["risk_at_most_r2"] is False
    assert result.merge_disposition is MergeDisposition.R3_PROPOSAL
    assert result.receipt is not None
    assert result.receipt.authorizes_merge is False


def test_model_assisted_requires_exact_envelope_and_does_not_grant_effect(tmp_path: Path) -> None:
    calls: list[str] = []

    def invoker(plan, envelope):
        calls.append(envelope.envelope_id)
        return {"diagnostic_receipt_id": "diag:model"}

    controller = AutonomousRepairController(
        engine=_RecordingEngine(tmp_path),
        repo_root=tmp_path,
        model_invoker=invoker,
    )
    result = controller.run(
        _request(
            requested_tier=RepairTier.MODEL_ASSISTED_BOUNDED,
            deterministic_available=False,
            template_available=False,
        )
    )
    missing_context = controller.run(
        _request(
            requested_tier=RepairTier.MODEL_ASSISTED_BOUNDED,
            deterministic_available=False,
            template_available=False,
            context_reference_ids=(),
            context_sufficient=True,
        )
    )

    assert result.disposition is RepairControllerDisposition.MODEL_ASSISTED_DELEGATED
    assert result.selected_tier is RepairTier.MODEL_ASSISTED_BOUNDED
    assert result.plan is not None
    assert result.plan.worktree_id == "worktree:isolated"
    assert result.plan.required_test_ids == ("test-repair-controller",)
    assert "decision_runtime_required" in result.reason_codes
    assert result.authorizes_effect is False
    assert len(calls) == 1
    assert missing_context.disposition is RepairControllerDisposition.REJECTED
    assert "tier_or_plan_rejected" in missing_context.reason_codes
    assert len(calls) == 1
