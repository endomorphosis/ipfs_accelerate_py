from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import pytest
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
    ENGINE_INTERFACE,
    LOW_RISK_MERGE_CONDITIONS,
    PROTECTED_AUTHORITY_PATHS,
    SELF_EDIT_PATH,
    AutonomousRepairController,
    RepairControllerDisposition,
    RepairControllerError,
    RepairControllerRequest,
    RepairControllerResult,
    RepairMergeDisposition,
    classify_scope_violations,
    select_repair_tier,
)

CONTROLLER_PATH = (
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
        "default_level": AutonomyLevel.SELF_REPAIR_ISOLATED,
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
        "acceptance_criterion_ids": ("AC-repair",),
        "risk_assessment": _risk(),
        "autonomy_level": AutonomyLevel.EXECUTE_REVERSIBLE,
        "cognitive_budget": _budget(),
        "allowed_paths": ("ipfs_accelerate_py/agent_supervisor/autonomy",),
        "allowed_symbols": ("AutonomousRepairController", "select_repair_tier"),
        "required_test_ids": ("test-repair",),
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


def _request(**overrides: object) -> RepairControllerRequest:
    policy = overrides.pop("policy", None) or _policy()
    envelope = overrides.pop("envelope", None) or _envelope(policy=policy)
    values: dict[str, object] = {
        "envelope": envelope,
        "policy": policy,
        "predicted_files": ("ipfs_accelerate_py/agent_supervisor/autonomy/human_escalation.py",),
        "predicted_symbols": ("AutonomousRepairController",),
        "worktree_id": "worktree-repair-1",
        "rollback_plan_id": "rollback-repair-1",
        "context_reference_ids": ("context-ref-1",),
        "repair_tier": RepairTier.DETERMINISTIC,
        "validation_receipt_ids": ("test-repair",),
        "terminal_status": TerminalStatus.SUCCEEDED,
        "changed_paths": ("ipfs_accelerate_py/agent_supervisor/autonomy/human_escalation.py",),
        "max_changed_files": 1,
        "max_changed_lines": 80,
    }
    values.update(overrides)
    return RepairControllerRequest(**values)


class _RecordingEngine:
    def __init__(self, *, model_call_count: int = 0) -> None:
        self.calls: list[object] = []
        self.model_call_count = model_call_count

    def run(self, items: object) -> SimpleNamespace:
        self.calls.append(list(items))
        return SimpleNamespace(model_call_count=self.model_call_count, passed=False, rows=[])


class _CountingInvoker:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, plan: object) -> dict[str, str]:
        self.calls += 1
        return {
            "diagnostic_receipt_id": "diagnostic-model-1",
            "failure_signature": "fail-identical-1",
        }


def test_controller_is_a_facade_not_a_second_repair_engine() -> None:
    source = CONTROLLER_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    defined = {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}
    assert "AutonomousRepairEngine" not in defined
    assert "AutonomousRepairMaterializer" not in defined
    imported_engine = False
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and "autonomous_repair" in node.module:
            names = {alias.name for alias in node.names}
            if "AutonomousRepairEngine" in names or "AUTONOMOUS_REPAIR_INTERFACE" in names:
                imported_engine = True
    assert imported_engine
    engine = _RecordingEngine()
    controller = AutonomousRepairController(engine=engine)
    assert controller.interface == AUTONOMOUS_REPAIR_CONTROLLER_INTERFACE
    assert controller.engine_interface == ENGINE_INTERFACE
    assert controller.engine is engine
    assert not isinstance(controller, AutonomousRepairEngine)
    result = controller.evaluate(_request())
    assert engine.calls
    assert result.engine_interface == ENGINE_INTERFACE
    assert result.authorizes_effect is False
    assert result.authorizes_merge is False
    assert result.receipt is not None
    assert result.receipt.authorizes_merge is False


def test_selects_deterministic_template_and_model_tiers() -> None:
    assert select_repair_tier(requested=None) is RepairTier.DETERMINISTIC
    assert (
        select_repair_tier(requested=None, template_id="template:rename")
        is RepairTier.TEMPLATE_CONSTRAINED
    )
    assert (
        select_repair_tier(requested=None, model_assisted_requested=True)
        is RepairTier.MODEL_ASSISTED_BOUNDED
    )
    controller = AutonomousRepairController()
    deterministic = controller.evaluate(_request(repair_tier=RepairTier.DETERMINISTIC))
    assert deterministic.repair_tier is RepairTier.DETERMINISTIC
    assert deterministic.plan is not None
    template = controller.evaluate(
        _request(repair_tier=RepairTier.TEMPLATE_CONSTRAINED, template_id="template:rename")
    )
    assert template.repair_tier is RepairTier.TEMPLATE_CONSTRAINED
    model = controller.evaluate(
        _request(
            repair_tier=RepairTier.MODEL_ASSISTED_BOUNDED,
            model_assisted_requested=True,
            terminal_status=TerminalStatus.PENDING,
            validation_receipt_ids=(),
            changed_paths=(),
        )
    )
    assert model.repair_tier is RepairTier.MODEL_ASSISTED_BOUNDED
    assert model.plan is not None
    assert model.plan.worktree_id == "worktree-repair-1"
    missing_template = controller.evaluate(
        _request(repair_tier=RepairTier.TEMPLATE_CONSTRAINED, template_id="")
    )
    assert missing_template.disposition is RepairControllerDisposition.REJECTED
    assert "template_id_required" in missing_template.reason_codes


def test_model_assisted_requires_exact_envelope_bindings() -> None:
    controller = AutonomousRepairController()
    policy = _policy()
    envelope = _envelope(policy=policy, required_test_ids=())
    result = controller.evaluate(
        _request(
            policy=policy,
            envelope=envelope,
            repair_tier=RepairTier.MODEL_ASSISTED_BOUNDED,
            required_test_ids=(),
            terminal_status=TerminalStatus.PENDING,
            validation_receipt_ids=(),
            changed_paths=(),
        )
    )
    assert result.disposition is RepairControllerDisposition.REJECTED
    assert "predetermined_tests_required" in result.reason_codes


def test_scope_escape_self_edit_and_validator_policy_key_are_rejected() -> None:
    controller = AutonomousRepairController()
    escape = controller.evaluate(
        _request(
            predicted_files=("docs/outside.md",),
            changed_paths=("docs/outside.md",),
            terminal_status=TerminalStatus.PENDING,
            validation_receipt_ids=(),
        )
    )
    assert escape.disposition is RepairControllerDisposition.REJECTED
    assert "scope_escape" in escape.reason_codes
    assert escape.merge_disposition is RepairMergeDisposition.REJECTED
    assert escape.model_call_count == 0

    self_edit = controller.evaluate(
        _request(
            predicted_files=(SELF_EDIT_PATH,),
            changed_paths=(SELF_EDIT_PATH,),
            terminal_status=TerminalStatus.PENDING,
            validation_receipt_ids=(),
        )
    )
    assert self_edit.disposition is RepairControllerDisposition.REJECTED
    assert "self_edit" in self_edit.reason_codes
    assert "protected_authority_path" in self_edit.reason_codes

    key_path = controller.evaluate(
        _request(
            predicted_files=(
                "ipfs_accelerate_py/agent_supervisor/autonomy/trusted_keys.py",
            ),
            changed_paths=("ipfs_accelerate_py/agent_supervisor/autonomy/trusted_keys.py",),
            terminal_status=TerminalStatus.PENDING,
            validation_receipt_ids=(),
        )
    )
    assert key_path.disposition is RepairControllerDisposition.REJECTED
    assert "validator_policy_key_mutation" in key_path.reason_codes

    key_symbol = classify_scope_violations(
        ("ipfs_accelerate_py/agent_supervisor/autonomy/human_escalation.py",),
        ("trusted_keys",),
        allowed_paths=("ipfs_accelerate_py/agent_supervisor/autonomy",),
        allowed_symbols=("AutonomousRepairController",),
    )
    assert "validator_policy_key_mutation" in key_symbol
    engine_path = classify_scope_violations(
        ("ipfs_accelerate_py/agent_supervisor/autonomous_repair/engine.py",),
        ("AutonomousRepairController",),
        allowed_paths=("ipfs_accelerate_py/agent_supervisor",),
        allowed_symbols=("AutonomousRepairController",),
    )
    assert "protected_authority_path" in engine_path
    assert PROTECTED_AUTHORITY_PATHS[0] == SELF_EDIT_PATH


def test_identical_failures_reuse_diagnosis_and_do_not_repeat_model_calls() -> None:
    invoker = _CountingInvoker()
    controller = AutonomousRepairController(model_invoker=invoker)
    failed = _request(
        repair_tier=RepairTier.MODEL_ASSISTED_BOUNDED,
        model_assisted_requested=True,
        terminal_status=TerminalStatus.FAILED,
        failure_signature="fail-identical-1",
        validation_receipt_ids=(),
        changed_paths=(),
    )
    first = controller.evaluate(failed, now_ms=10)
    assert invoker.calls == 1
    assert first.model_call_count == 1
    assert first.diagnostic_reused is False
    assert controller.model_call_count == 1
    assert controller.failure_record("fail-identical-1") is not None

    second = controller.evaluate(failed, now_ms=20)
    assert invoker.calls == 1
    assert second.disposition is RepairControllerDisposition.BACKOFF
    assert second.diagnostic_reused is True
    assert second.model_call_count == 0
    assert second.backoff_milliseconds >= 1_000
    assert "model_call_suppressed" in second.reason_codes
    assert second.low_risk_conditions["not_identical_failure_backoff"] is False
    assert second.merge_disposition is not RepairMergeDisposition.AUTONOMOUS_MERGE

    third = controller.evaluate(failed, now_ms=30)
    fourth = controller.evaluate(failed, now_ms=40)
    assert invoker.calls == 1
    assert controller.model_call_count == 1
    assert fourth.disposition is RepairControllerDisposition.IDENTICAL_FAILURE_EXHAUSTED

    recovered = controller.evaluate(
        _request(
            repair_tier=RepairTier.MODEL_ASSISTED_BOUNDED,
            model_assisted_requested=True,
            terminal_status=TerminalStatus.FAILED,
            failure_signature="fail-identical-1",
            new_evidence_ids=("evidence-new-1",),
            validation_receipt_ids=(),
            changed_paths=(),
        ),
        now_ms=50,
    )
    assert invoker.calls == 2
    assert recovered.diagnostic_reused is False
    assert recovered.model_call_count == 1


def test_r2_autonomous_merge_requires_every_stated_low_risk_condition() -> None:
    controller = AutonomousRepairController()
    success = controller.evaluate(_request())
    assert success.disposition is RepairControllerDisposition.MERGE_ELIGIBLE
    assert success.merge_disposition is RepairMergeDisposition.AUTONOMOUS_MERGE
    assert success.merge_eligible
    assert set(success.low_risk_conditions) == set(LOW_RISK_MERGE_CONDITIONS)
    assert all(success.low_risk_conditions[name] for name in LOW_RISK_MERGE_CONDITIONS)
    assert success.receipt is not None
    assert success.receipt.authorizes_merge is False
    assert success.authorizes_merge is False
    assert "r2_merge_conjunction" in success.reason_codes

    blockers: list[tuple[str, dict[str, object]]] = [
        (
            "risk_at_most_r2",
            {
                "envelope": _envelope(
                    policy=_policy(),
                    risk_assessment=_risk(
                        risk_class=RiskClass.R3_BOUNDED_REPOSITORY_MUTATION,
                        reversible=True,
                    ),
                    autonomy_level=AutonomyLevel.SELF_REPAIR_ISOLATED,
                )
            },
        ),
        (
            "reversible",
            {
                "envelope": _envelope(
                    policy=_policy(),
                    risk_assessment=_risk(reversible=False),
                    reversible=False,
                )
            },
        ),
        (
            "autonomous_merge_enabled",
            {"policy": _policy(autonomous_merge_enabled=False)},
        ),
        (
            "isolated_worktree_bound",
            {},
        ),
        (
            "predetermined_tests_satisfied",
            {"validation_receipt_ids": ()},
        ),
        (
            "validation_receipts_current",
            {"validation_receipt_ids": (), "terminal_status": TerminalStatus.PENDING},
        ),
        (
            "terminal_succeeded",
            {"terminal_status": TerminalStatus.PENDING},
        ),
        (
            "sufficient_context",
            {"sufficient_context": False},
        ),
    ]
    for name, overrides in blockers:
        if name == "isolated_worktree_bound":
            request = _request()
            object.__setattr__(request, "worktree_id", "")
            result = controller.evaluate(request)
        elif name == "autonomous_merge_enabled":
            policy = _policy(autonomous_merge_enabled=False)
            result = controller.evaluate(_request(policy=policy, envelope=_envelope(policy=policy)))
        elif name == "sufficient_context":
            result = controller.evaluate(_request(sufficient_context=False))
        elif name == "reversible":
            policy = _policy()
            result = controller.evaluate(
                _request(
                    policy=policy,
                    envelope=_envelope(
                        policy=policy,
                        risk_assessment=_risk(reversible=False),
                        reversible=False,
                    ),
                )
            )
        elif name == "risk_at_most_r2":
            policy = _policy()
            result = controller.evaluate(
                _request(
                    policy=policy,
                    envelope=_envelope(
                        policy=policy,
                        risk_assessment=_risk(
                            risk_class=RiskClass.R3_BOUNDED_REPOSITORY_MUTATION,
                            reversible=True,
                        ),
                        autonomy_level=AutonomyLevel.SELF_REPAIR_ISOLATED,
                    ),
                )
            )
        else:
            result = controller.evaluate(_request(**overrides))
        assert result.merge_disposition is not RepairMergeDisposition.AUTONOMOUS_MERGE, name
        assert result.low_risk_conditions[name] is False, name
        assert result.authorizes_merge is False


def test_r3_successful_repair_is_proposal_only() -> None:
    policy = _policy()
    envelope = _envelope(
        policy=policy,
        risk_assessment=_risk(
            risk_class=RiskClass.R3_BOUNDED_REPOSITORY_MUTATION,
            reversible=True,
        ),
        autonomy_level=AutonomyLevel.SELF_REPAIR_ISOLATED,
    )
    result = AutonomousRepairController().evaluate(_request(policy=policy, envelope=envelope))
    assert result.disposition is RepairControllerDisposition.PROPOSAL_ONLY
    assert result.merge_disposition is RepairMergeDisposition.PROPOSAL_ONLY
    assert result.low_risk_conditions["risk_at_most_r2"] is False
    assert "r3_proposal_only" in result.reason_codes
    assert result.receipt is not None
    assert result.receipt.authorizes_merge is False


def test_r5_requires_a_human_and_cannot_merge() -> None:
    policy = _policy(default_level=AutonomyLevel.RECOMMEND)
    envelope = _envelope(
        policy=policy,
        risk_assessment=_risk(
            risk_class=RiskClass.R5_IRREVERSIBLE_EXTERNAL_OR_LEGAL,
            reversible=False,
            irreversible_external_effect=True,
            legal_or_financial_effect=True,
        ),
        autonomy_level=AutonomyLevel.RECOMMEND,
        reversible=False,
    )
    result = AutonomousRepairController().evaluate(
        _request(
            policy=policy,
            envelope=envelope,
            terminal_status=TerminalStatus.PENDING,
            validation_receipt_ids=(),
            changed_paths=(),
        )
    )
    assert result.merge_disposition is RepairMergeDisposition.HUMAN_REQUIRED
    assert result.merge_eligible is False


def test_result_is_frozen_and_cannot_claim_merge_authority() -> None:
    result = AutonomousRepairController().evaluate(_request())
    with pytest.raises(FrozenInstanceError):
        result.authorizes_merge = True  # type: ignore[misc]
    payload = result.to_dict()
    assert payload["authorizes_effect"] is False
    assert payload["authorizes_merge"] is False
    assert payload["interface"] == AUTONOMOUS_REPAIR_CONTROLLER_INTERFACE
    rebuilt_conditions = dict(result.low_risk_conditions)
    with pytest.raises(TypeError):
        result.low_risk_conditions["risk_at_most_r2"] = False  # type: ignore[index]
    assert rebuilt_conditions["risk_at_most_r2"] is True


def test_malformed_request_fails_closed() -> None:
    with pytest.raises(RepairControllerError, match="envelope"):
        RepairControllerRequest(
            envelope="not-an-envelope",  # type: ignore[arg-type]
            predicted_files=("ipfs_accelerate_py/agent_supervisor/autonomy/human_escalation.py",),
            predicted_symbols=("AutonomousRepairController",),
            worktree_id="worktree-1",
            rollback_plan_id="rollback-1",
            context_reference_ids=("context-1",),
        )
    with pytest.raises(RepairControllerError, match="source_edit_operator"):
        _request(
            source_edit_operator={"raw_prompt": "rewrite the repository"},
            terminal_status=TerminalStatus.PENDING,
            validation_receipt_ids=(),
            changed_paths=(),
        )


def test_policy_flag_cannot_admit_a_source_edit() -> None:
    result = AutonomousRepairController().evaluate(
        _request(
            allow_code_edit_materialize=True,
            terminal_status=TerminalStatus.PENDING,
            validation_receipt_ids=(),
            changed_paths=(),
        )
    )
    assert result.disposition is RepairControllerDisposition.REJECTED
    assert "policy_flag_is_not_source_edit_admission" in result.reason_codes
    assert result.mutation_applied is False
    assert result.source_edit_admitted is False


def test_result_round_trip_identity_is_stable() -> None:
    result = AutonomousRepairController().evaluate(_request())
    payload = result.to_dict()
    assert payload["result_id"] == result.result_id
    assert RepairControllerResult(
        disposition=result.disposition,
        merge_disposition=result.merge_disposition,
        repair_tier=result.repair_tier,
        reason_codes=result.reason_codes,
        low_risk_conditions=dict(result.low_risk_conditions),
        plan=result.plan,
        receipt=result.receipt,
        diagnostic_reused=result.diagnostic_reused,
        backoff_milliseconds=result.backoff_milliseconds,
        model_call_count=result.model_call_count,
        source_edit_admitted=result.source_edit_admitted,
        mutation_applied=result.mutation_applied,
    ).result_id == result.result_id
