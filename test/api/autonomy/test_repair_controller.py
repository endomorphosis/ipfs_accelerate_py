from __future__ import annotations

import ast
import base64
import hashlib
import inspect
from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.autonomous_repair.engine import (
    AutonomousRepairEngine,
)
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
    PROTECTED_AUTHORITY_PATHS,
    AutonomousRepairAttempt,
    AutonomousRepairController,
    RepairControllerDisposition,
    RepairControllerError,
    RepairMergeDisposition,
    evaluate_low_risk_merge_conditions,
    is_protected_authority_path,
    is_self_edit_path,
    is_validator_policy_key_path,
    select_repair_tier,
)


TARGET_FILE = "ipfs_accelerate_py/agent_supervisor/autonomy/semantic_memory.py"
TARGET_SYMBOL = "SemanticMemory"


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


def _policy(**overrides: object) -> AutonomyPolicy:
    values: dict[str, object] = {
        "policy_revision": "policy-rev-repair",
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
        "blast_radius_symbols": (TARGET_SYMBOL,),
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
        "allowed_symbols": (TARGET_SYMBOL,),
        "required_test_ids": ("test-repair-controller",),
        "required_proof_ids": (),
        "authority_id": "operator-policy-authority",
        "policy_id": policy.policy_id,
        "provider_usage_envelope_id": "provider-envelope-1",
        "resource_budget_id": "resource-budget-1",
        "human_escalation_policy_id": "human-policy-1",
        "expiry_ms": 10_000,
        "reversible": True,
    }
    values.update(overrides)
    return AutonomyEnvelope(**values)


def _attempt(
    *,
    policy: AutonomyPolicy | None = None,
    envelope: AutonomyEnvelope | None = None,
    **overrides: object,
) -> AutonomousRepairAttempt:
    policy = policy or _policy()
    values: dict[str, object] = {
        "envelope": envelope or _envelope(policy=policy),
        "policy": policy,
        "predicted_files": (TARGET_FILE,),
        "predicted_symbols": (TARGET_SYMBOL,),
        "worktree_id": "worktree-isolated-1",
        "rollback_plan_id": "rollback-plan-1",
        "context_reference_ids": ("context-ref-1",),
        "required_test_ids": ("test-repair-controller",),
        "max_changed_files": 2,
        "max_changed_lines": 80,
        "changed_line_count": 12,
    }
    values.update(overrides)
    return AutonomousRepairAttempt(**values)


def test_interface_is_versioned_and_no_second_engine_is_created(tmp_path: Path) -> None:
    module = inspect.getmodule(AutonomousRepairController)
    assert module is not None
    source = inspect.getsource(module)
    tree = ast.parse(source)
    class_names = {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}
    assert "AutonomousRepairEngine" not in class_names
    assert "AutonomousRepairMaterializer" not in class_names
    assert AUTONOMOUS_REPAIR_CONTROLLER_INTERFACE == "AutonomousRepairController@1"
    assert AutonomousRepairController.INTERFACE == AUTONOMOUS_REPAIR_CONTROLLER_INTERFACE
    controller = AutonomousRepairController()
    assert controller.engine is None
    assert controller.engine_authority == "AutonomousRepairEngine@1"
    engine = AutonomousRepairEngine(repo_root=tmp_path)
    composed = AutonomousRepairController(engine=engine)
    assert composed.engine is engine
    with pytest.raises(RepairControllerError, match="existing AutonomousRepairEngine"):
        AutonomousRepairController(engine=object())  # type: ignore[arg-type]


def test_selects_deterministic_template_and_model_tiers() -> None:
    deterministic = _attempt(requested_tier=RepairTier.DETERMINISTIC)
    template = _attempt(template_id="template:rename", requested_tier=None)
    model = _attempt(requested_tier=RepairTier.MODEL_ASSISTED_BOUNDED)
    assert select_repair_tier(deterministic) is RepairTier.DETERMINISTIC
    assert select_repair_tier(template) is RepairTier.TEMPLATE_CONSTRAINED
    assert select_repair_tier(model) is RepairTier.MODEL_ASSISTED_BOUNDED
    controller = AutonomousRepairController()
    admitted = controller.repair(deterministic)
    assert admitted.selected_tier is RepairTier.DETERMINISTIC
    assert admitted.plan is not None
    assert admitted.plan.repair_tier is RepairTier.DETERMINISTIC
    assert admitted.plan.predicted_files == (TARGET_FILE,)
    assert admitted.plan.predicted_symbols == (TARGET_SYMBOL,)
    templated = controller.repair(template)
    assert templated.selected_tier is RepairTier.TEMPLATE_CONSTRAINED
    modeled = controller.repair(model)
    assert modeled.selected_tier is RepairTier.MODEL_ASSISTED_BOUNDED
    assert modeled.plan is not None
    assert modeled.plan.worktree_id == "worktree-isolated-1"


def test_scope_escape_self_edit_and_validator_policy_key_are_rejected() -> None:
    controller = AutonomousRepairController()
    escape = controller.repair(
        _attempt(
            predicted_files=("docs/architecture/outside.md",),
            allowed_paths=("ipfs_accelerate_py/agent_supervisor/autonomy",),
        )
    )
    assert escape.disposition is RepairControllerDisposition.REJECTED
    assert "scope_escape_rejected" in escape.reason_codes
    assert not escape.authorizes_effect
    assert not escape.authorizes_merge

    self_edit = controller.repair(
        _attempt(
            predicted_files=(
                "ipfs_accelerate_py/agent_supervisor/autonomy/repair_controller.py",
            )
        )
    )
    assert self_edit.disposition is RepairControllerDisposition.REJECTED
    assert "self_edit_rejected" in self_edit.reason_codes
    assert is_self_edit_path(
        "ipfs_accelerate_py/agent_supervisor/autonomy/repair_controller.py"
    )

    key_path = "ipfs_accelerate_py/llm_router.py"
    policy = _policy()
    key_envelope = _envelope(
        policy=policy,
        allowed_paths=("ipfs_accelerate_py",),
        allowed_symbols=(TARGET_SYMBOL,),
    )
    key_mutation = controller.repair(
        _attempt(
            policy=policy,
            envelope=key_envelope,
            predicted_files=(key_path,),
            allowed_paths=("ipfs_accelerate_py",),
        )
    )
    assert key_mutation.disposition is RepairControllerDisposition.REJECTED
    assert "validator_policy_key_mutation_rejected" in key_mutation.reason_codes
    assert is_validator_policy_key_path(key_path)
    assert is_protected_authority_path(
        "ipfs_accelerate_py/agent_supervisor/proof/multi_prover_router.py"
    )
    assert "ipfs_accelerate_py/llm_router.py" in PROTECTED_AUTHORITY_PATHS

    forbidden_symbol = controller.repair(_attempt(predicted_symbols=("trusted_keys",)))
    assert forbidden_symbol.disposition is RepairControllerDisposition.REJECTED
    assert "forbidden_symbol_rejected" in forbidden_symbol.reason_codes


def test_model_assisted_requires_isolated_worktree_and_predetermined_checks() -> None:
    controller = AutonomousRepairController()
    policy = _policy()
    missing = controller.repair(
        _attempt(
            policy=policy,
            envelope=_envelope(
                policy=policy,
                required_test_ids=(),
                required_proof_ids=(),
            ),
            requested_tier=RepairTier.MODEL_ASSISTED_BOUNDED,
            worktree_id="",
            context_reference_ids=(),
            required_test_ids=(),
            required_proof_ids=(),
            rollback_plan_id="",
        )
    )
    assert missing.disposition is RepairControllerDisposition.REJECTED
    assert "isolated_worktree_required" in missing.reason_codes
    assert "sufficient_context_required" in missing.reason_codes
    assert "predetermined_checks_required" in missing.reason_codes
    assert "rollback_plan_required" in missing.reason_codes


def test_identical_failures_do_not_repeat_model_calls() -> None:
    controller = AutonomousRepairController(
        identical_failure_limit=3,
        base_backoff_milliseconds=10,
        max_backoff_milliseconds=80,
    )
    calls = {"count": 0}

    def _model(_plan, _attempt):
        calls["count"] += 1
        return {"advice": "reuse-diagnostic"}

    first = _attempt(
        requested_tier=RepairTier.MODEL_ASSISTED_BOUNDED,
        failure_signature="failure:same-test",
        diagnostic_receipt_id="diagnostic:v1",
    )
    second = _attempt(
        requested_tier=RepairTier.MODEL_ASSISTED_BOUNDED,
        failure_signature="failure:same-test",
        diagnostic_receipt_id="diagnostic:v2-ignored",
    )
    first_result = controller.repair(first, model_call=_model)
    second_result = controller.repair(second, model_call=_model)
    third_result = controller.repair(second, model_call=_model)
    exhausted = controller.repair(second, model_call=_model)

    assert first_result.model_call_count == 1
    assert calls["count"] == 1
    assert second_result.disposition is RepairControllerDisposition.IDENTICAL_FAILURE_BACKOFF
    assert second_result.diagnostic_reused
    assert second_result.model_call_count == 0
    assert second_result.backoff_milliseconds == 10
    assert second_result.diagnostic_receipt_id == "diagnostic:v1"
    assert "model_call_suppressed" in second_result.reason_codes
    assert third_result.model_call_count == 0
    assert calls["count"] == 1
    assert exhausted.disposition is RepairControllerDisposition.IDENTICAL_FAILURE_EXHAUSTED
    assert exhausted.model_call_count == 0
    assert calls["count"] == 1
    record = controller.failure_record("failure:same-test")
    assert record is not None
    assert record.model_call_count == 1
    assert record.exhausted


def test_autonomous_merge_requires_every_low_risk_condition() -> None:
    controller = AutonomousRepairController()
    eligible = controller.repair(
        _attempt(
            requested_tier=RepairTier.DETERMINISTIC,
            validation_receipt_ids=("validation:repair-1",),
            changed_paths=(TARGET_FILE,),
        )
    )
    assert eligible.merge_disposition is RepairMergeDisposition.AUTONOMOUS_MERGE
    assert eligible.disposition is RepairControllerDisposition.AUTONOMOUS_MERGE_ELIGIBLE
    assert eligible.satisfied_merge_conditions == LOW_RISK_MERGE_CONDITIONS
    assert eligible.missing_merge_conditions == ()
    assert eligible.receipt is not None
    assert eligible.receipt.terminal_status is TerminalStatus.SUCCEEDED
    assert eligible.receipt.authorizes_merge is False
    assert eligible.authorizes_merge is False
    assert "low_risk_merge_conjunction" in eligible.reason_codes

    disabled_policy = _policy(autonomous_merge_enabled=False)
    disabled = controller.repair(
        _attempt(
            policy=disabled_policy,
            envelope=_envelope(policy=disabled_policy),
            validation_receipt_ids=("validation:repair-1",),
            changed_paths=(TARGET_FILE,),
        )
    )
    assert disabled.merge_disposition is RepairMergeDisposition.PROPOSAL
    assert "autonomous_merge_enabled" in disabled.missing_merge_conditions

    observe_policy = _policy(default_level=AutonomyLevel.OBSERVE_ONLY)
    observe = controller.repair(
        _attempt(
            policy=observe_policy,
            envelope=_envelope(
                policy=observe_policy,
                autonomy_level=AutonomyLevel.OBSERVE_ONLY,
            ),
            validation_receipt_ids=("validation:repair-1",),
            changed_paths=(TARGET_FILE,),
        )
    )
    assert observe.merge_disposition is not RepairMergeDisposition.AUTONOMOUS_MERGE
    assert "autonomy_level_permits_execution" in observe.missing_merge_conditions

    r3_policy = _policy()
    r3 = controller.repair(
        _attempt(
            policy=r3_policy,
            envelope=_envelope(
                policy=r3_policy,
                risk_assessment=_risk(risk_class=RiskClass.R3_BOUNDED_REPOSITORY_MUTATION),
                autonomy_level=AutonomyLevel.SELF_REPAIR_ISOLATED,
            ),
            validation_receipt_ids=("validation:repair-1",),
            changed_paths=(TARGET_FILE,),
        )
    )
    assert r3.merge_disposition is RepairMergeDisposition.PROPOSAL
    assert "r3_proposal" in r3.reason_codes
    assert "risk_at_most_r2" in r3.missing_merge_conditions
    assert r3.authorizes_merge is False


def test_missing_any_named_low_risk_condition_blocks_autonomous_merge() -> None:
    policy = _policy()
    attempt = _attempt(
        policy=policy,
        validation_receipt_ids=("validation:repair-1",),
        changed_paths=(TARGET_FILE,),
    )
    satisfied, missing = evaluate_low_risk_merge_conditions(
        attempt, terminal_status=TerminalStatus.SUCCEEDED
    )
    assert missing == ()
    assert satisfied == LOW_RISK_MERGE_CONDITIONS

    no_tests = _attempt(required_test_ids=("test-repair-controller",), validation_receipt_ids=())
    _satisfied, missing_tests = evaluate_low_risk_merge_conditions(
        no_tests, terminal_status=TerminalStatus.PENDING
    )
    assert "repair_succeeded" in missing_tests
    assert "required_tests_satisfied" in missing_tests
    assert "predetermined_checks_complete" in missing_tests

    no_rollback = _attempt(
        rollback_plan_id="",
        validation_receipt_ids=("validation:repair-1",),
    )
    _satisfied, missing_rollback = evaluate_low_risk_merge_conditions(
        no_rollback, terminal_status=TerminalStatus.SUCCEEDED
    )
    assert "rollback_plan_bound" in missing_rollback


def test_source_edit_admission_is_scope_gated_and_does_not_write(tmp_path: Path) -> None:
    relative = TARGET_FILE
    target = tmp_path / relative
    target.parent.mkdir(parents=True)
    original = b"class SemanticMemory:\n    pass\n"
    patched = original.replace(b"pass", b"return None")
    target.write_bytes(original)
    old_digest = "sha256:" + hashlib.sha256(original).hexdigest()
    new_digest = "sha256:" + hashlib.sha256(patched).hexdigest()
    operator = {
        "operator_id": "source-edit:semantic-memory",
        "owner_root": str(tmp_path.resolve()),
        "relative_path": relative,
        "old_digest": old_digest,
        "new_digest": new_digest,
        "old_bytes_b64": base64.b64encode(original).decode("ascii"),
        "new_bytes_b64": base64.b64encode(patched).decode("ascii"),
        "forward_diff": f"--- {old_digest}\n+++ {new_digest}\n+ patched",
        "inverse_diff": f"--- {new_digest}\n+++ {old_digest}\n- patched",
        "disposition": "validation_pending",
        "admitted": True,
        "kind": "replace_exact_bytes",
    }
    controller = AutonomousRepairController()
    admitted = controller.admit_source_edit(_attempt(source_edit_operator=operator))
    assert admitted.relative_path == relative
    assert admitted.admitted is True
    result = controller.repair(_attempt(source_edit_operator=operator))
    assert result.source_edit_admitted is True
    assert result.authorizes_effect is False
    assert result.authorizes_merge is False
    assert "source_edit_admitted" in result.reason_codes
    assert target.read_bytes() == original

    escaped = controller.repair(
        _attempt(
            predicted_files=("docs/architecture/outside.md",),
            source_edit_operator={**operator, "relative_path": "docs/architecture/outside.md"},
        )
    )
    assert escaped.disposition is RepairControllerDisposition.REJECTED
    assert escaped.source_edit_admitted is False
    assert "scope_escape_rejected" in escaped.reason_codes
    assert target.read_bytes() == original


def test_delegates_to_existing_engine_without_claiming_completion(tmp_path: Path) -> None:
    engine = AutonomousRepairEngine(repo_root=tmp_path)
    controller = AutonomousRepairController(engine=engine)
    result = controller.repair(
        _attempt(requested_tier=RepairTier.DETERMINISTIC),
        delegate_to_engine=True,
    )
    assert result.engine_report_id
    assert "delegated_existing_repair_engine" in result.reason_codes
    assert result.receipt is None or result.receipt.terminal_status is not TerminalStatus.SUCCEEDED or result.receipt.validation_receipt_ids
    assert result.authorizes_effect is False
    snapshot = controller.snapshot()
    assert snapshot["interface"] == AUTONOMOUS_REPAIR_CONTROLLER_INTERFACE
    assert snapshot["schema"].endswith("repair-failure-memory@1")
