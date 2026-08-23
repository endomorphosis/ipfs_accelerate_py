from __future__ import annotations

import ast
from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.autonomous_repair.contracts import (
    AUTONOMOUS_REPAIR_INTERFACE,
    AutonomousRepairReport,
)
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
    ENGINE_PACKAGE_PREFIX,
    LOW_RISK_MERGE_CONDITIONS,
    REPAIR_CONTROLLER_OUTCOME_SCHEMA,
    SELF_EDIT_RELATIVE_PATH,
    AutonomousRepairController,
    RepairControllerDisposition,
    RepairControllerError,
    RepairControllerOutcome,
    RepairControllerRequest,
    RepairMergeDisposition,
    SourceEditAdmissionDisposition,
    is_protected_authority_path,
    is_self_edit_path,
    select_repair_tier,
)

REPAIR_MODULE = (
    Path(__file__).resolve().parents[3]
    / "ipfs_accelerate_py/agent_supervisor/autonomy/repair_controller.py"
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


def _risk(*, risk_class: RiskClass = RiskClass.R2_REVERSIBLE_LOCAL, **overrides: object) -> RiskAssessment:
    values: dict[str, object] = {
        "risk_class": risk_class,
        "reversible": risk_class is not RiskClass.R5_IRREVERSIBLE_EXTERNAL_OR_LEGAL,
        "blast_radius_paths": ("ipfs_accelerate_py/agent_supervisor/autonomy",),
        "blast_radius_symbols": ("AutonomousRepairController",),
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
    assessment = risk or _risk()
    values: dict[str, object] = {
        "repository_id": "repo-1",
        "tree_id": "tree-1",
        "objective_id": "APMC-G000",
        "objective_revision": "objective-rev-1",
        "task_id": "APMC-013",
        "acceptance_criterion_ids": ("AC-repair",),
        "risk_assessment": assessment,
        "autonomy_level": AutonomyLevel.EXECUTE_REVERSIBLE,
        "cognitive_budget": _budget(),
        "allowed_paths": ("ipfs_accelerate_py/agent_supervisor/autonomy",),
        "allowed_symbols": ("AutonomousRepairController", "bounded_repair_target"),
        "required_test_ids": ("test-repair-controller",),
        "required_proof_ids": (),
        "authority_id": "operator-policy-authority",
        "policy_id": policy.policy_id,
        "provider_usage_envelope_id": "provider-envelope-1",
        "resource_budget_id": "resource-budget-1",
        "human_escalation_policy_id": "human-policy-1",
        "expiry_ms": 12_000,
        "reversible": assessment.reversible,
    }
    values.update(overrides)
    return AutonomyEnvelope(**values)


def _controller(**overrides: object) -> AutonomousRepairController:
    policy = overrides.pop("policy", None) or _policy()
    envelope = overrides.pop("envelope", None) or _envelope(policy=policy)
    return AutonomousRepairController(envelope=envelope, policy=policy, **overrides)


def _request(**overrides: object) -> RepairControllerRequest:
    values: dict[str, object] = {
        "predicted_files": ("ipfs_accelerate_py/agent_supervisor/autonomy/contracts.py",),
        "predicted_symbols": ("AutonomousRepairController",),
        "requested_tier": RepairTier.DETERMINISTIC,
        "worktree_id": "worktree-isolated-1",
        "rollback_plan_id": "rollback-plan-1",
        "context_reference_ids": ("context-ref-1",),
        "required_test_ids": ("test-repair-controller",),
        "changed_paths": ("ipfs_accelerate_py/agent_supervisor/autonomy/contracts.py",),
        "validation_receipt_ids": ("test-repair-controller",),
        "execute": True,
    }
    values.update(overrides)
    return RepairControllerRequest(**values)


class RecordingEngine(AutonomousRepairEngine):
    def __init__(self, repo_root: Path) -> None:
        super().__init__(repo_root=repo_root)
        self.calls: list[tuple[object, ...]] = []

    def run(self, items):  # type: ignore[override]
        packed = tuple(items)
        self.calls.append(packed)
        return AutonomousRepairReport(
            policy={"domain": "agent_supervisor"},
            rows=[{"work_id": "work:recording", "disposition": "analysis_only"}],
            passed=False,
            model_call_count=0,
            llm_used=False,
            summary={"source_edits_applied": 0},
            notes=["recording engine delegates identity to AutonomousRepairEngine"],
        )


def test_interface_is_versioned_and_engine_authority_is_unchanged() -> None:
    controller = _controller()
    assert AUTONOMOUS_REPAIR_CONTROLLER_INTERFACE == "AutonomousRepairController@1"
    assert controller.interface == AUTONOMOUS_REPAIR_CONTROLLER_INTERFACE
    assert controller.engine_interface == AUTONOMOUS_REPAIR_INTERFACE
    assert controller.engine_class is AutonomousRepairEngine
    assert controller.authorizes_merge is False
    assert controller.authorizes_effect is False


def test_no_second_repair_engine_is_created() -> None:
    source = REPAIR_MODULE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    defined = {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}
    assert "AutonomousRepairEngine" not in defined
    assert "RepairEngine" not in defined
    imports = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module
        and node.module.endswith("autonomous_repair.engine")
    ]
    assert imports
    assert any(
        alias.name == "AutonomousRepairEngine" for node in imports for alias in node.names
    )
    policy = _policy()
    with pytest.raises(RepairControllerError, match="second repair engine"):
        AutonomousRepairController(
            envelope=_envelope(policy=policy),
            policy=policy,
            engine=object(),  # type: ignore[arg-type]
        )


def test_controller_delegates_to_injected_existing_engine(tmp_path: Path) -> None:
    engine = RecordingEngine(tmp_path)
    controller = _controller(engine=engine, repo_root=tmp_path)
    outcome = controller.run(
        _request(
            work_items=({"work_id": "work:one", "operation": "catalog.read"},),
            validation_receipt_ids=("test-repair-controller",),
        )
    )
    assert isinstance(controller.engine, AutonomousRepairEngine)
    assert engine.calls == [({"work_id": "work:one", "operation": "catalog.read"},)]
    assert outcome.engine_call_count == 1
    assert controller.engine_call_count == 1
    assert outcome.receipt is not None
    assert outcome.receipt.authorizes_merge is False


def test_software_first_tier_selection_never_implies_model_assistance() -> None:
    assert (
        select_repair_tier(
            requested=None,
            predicted_symbols=("Symbol",),
            context_reference_ids=("ctx",),
            worktree_id="worktree-1",
            required_test_ids=("t1",),
            required_proof_ids=(),
        )
        is RepairTier.TEMPLATE_CONSTRAINED
    )
    assert (
        select_repair_tier(
            requested=None,
            predicted_symbols=(),
            context_reference_ids=(),
            worktree_id="",
            required_test_ids=(),
            required_proof_ids=(),
        )
        is RepairTier.DETERMINISTIC
    )
    deterministic = _controller().run(_request(execute=False))
    assert deterministic.selected_tier is RepairTier.DETERMINISTIC
    template = _controller().run(
        _request(requested_tier=RepairTier.TEMPLATE_CONSTRAINED, execute=False)
    )
    assert template.selected_tier is RepairTier.TEMPLATE_CONSTRAINED
    assert template.disposition is RepairControllerDisposition.ADMITTED


def test_model_assisted_tier_requires_exact_envelope_and_isolated_worktree() -> None:
    controller = _controller()
    missing_worktree = controller.run(
        _request(
            requested_tier=RepairTier.MODEL_ASSISTED_BOUNDED,
            worktree_id="",
            execute=False,
        )
    )
    assert missing_worktree.disposition is RepairControllerDisposition.REJECTED_MISSING_WORKTREE
    missing_context = controller.run(
        _request(
            requested_tier=RepairTier.MODEL_ASSISTED_BOUNDED,
            context_reference_ids=(),
            execute=False,
        )
    )
    assert missing_context.disposition is RepairControllerDisposition.REJECTED_MISSING_CONTEXT
    admitted = controller.run(
        _request(
            requested_tier=RepairTier.MODEL_ASSISTED_BOUNDED,
            required_test_ids=("test-repair-controller",),
            execute=False,
        )
    )
    assert admitted.disposition is RepairControllerDisposition.ADMITTED
    assert admitted.requires_decision_runtime is True
    assert admitted.authorizes_effect is False


def test_scope_escape_is_rejected() -> None:
    outcome = _controller().run(
        _request(predicted_files=("docs/outside.md",), changed_paths=("docs/outside.md",))
    )
    assert outcome.disposition is RepairControllerDisposition.REJECTED_SCOPE_ESCAPE
    assert "scope_escape" in outcome.reason_codes
    assert outcome.merge_disposition is RepairMergeDisposition.WITHHOLD
    assert outcome.authorizes_merge is False


def test_self_edit_of_the_facade_is_rejected() -> None:
    assert is_self_edit_path(SELF_EDIT_RELATIVE_PATH)
    outcome = _controller().run(
        _request(
            predicted_files=(SELF_EDIT_RELATIVE_PATH,),
            changed_paths=(SELF_EDIT_RELATIVE_PATH,),
        )
    )
    assert outcome.disposition is RepairControllerDisposition.REJECTED_SELF_EDIT
    assert "self_edit" in outcome.reason_codes
    assert outcome.merge_disposition is RepairMergeDisposition.WITHHOLD


def test_validator_policy_key_mutation_is_rejected() -> None:
    policy = _policy()
    envelope = _envelope(
        policy=policy,
        allowed_paths=(
            "ipfs_accelerate_py/agent_supervisor/autonomy",
            "ipfs_accelerate_py/agent_supervisor/validation",
            ENGINE_PACKAGE_PREFIX,
            "config",
            "secrets",
        ),
        allowed_symbols=("AutonomousRepairController", "bounded_repair_target"),
    )
    controller = AutonomousRepairController(envelope=envelope, policy=policy)
    validator = controller.run(
        _request(
            predicted_files=(
                "ipfs_accelerate_py/agent_supervisor/validation/proposal_validation.py",
            ),
            changed_paths=(
                "ipfs_accelerate_py/agent_supervisor/validation/proposal_validation.py",
            ),
        )
    )
    policy_key = controller.run(
        _request(
            predicted_files=("config/policy.json",),
            changed_paths=("config/policy.json",),
        )
    )
    secret_key = controller.run(
        _request(
            predicted_files=("secrets/trusted_keys.pem",),
            changed_paths=("secrets/trusted_keys.pem",),
        )
    )
    engine_authority = controller.run(
        _request(
            predicted_files=(f"{ENGINE_PACKAGE_PREFIX}/engine.py",),
            changed_paths=(f"{ENGINE_PACKAGE_PREFIX}/engine.py",),
        )
    )
    assert validator.disposition is RepairControllerDisposition.REJECTED_PROTECTED_AUTHORITY
    assert policy_key.disposition is RepairControllerDisposition.REJECTED_PROTECTED_AUTHORITY
    assert secret_key.disposition is RepairControllerDisposition.REJECTED_PROTECTED_AUTHORITY
    assert engine_authority.disposition is RepairControllerDisposition.REJECTED_PROTECTED_AUTHORITY
    assert is_protected_authority_path("config/validator_policy.json")
    assert is_protected_authority_path("ipfs_accelerate_py/agent_supervisor/proof/keys/policy.key")


def test_identical_failures_do_not_repeat_model_calls() -> None:
    calls: list[object] = []

    def invoker(request: RepairControllerRequest) -> bool:
        calls.append(request.failure_signature)
        return False

    controller = _controller()
    request = _request(
        requested_tier=RepairTier.MODEL_ASSISTED_BOUNDED,
        failure_signature="fail:identical-syntax",
        diagnostic_receipt_id="diag:syntax-v1",
        validation_receipt_ids=(),
        model_invoker=invoker,
    )
    first = controller.run(request)
    second = controller.run(request)
    exhausted = controller.run(request)

    assert first.disposition is RepairControllerDisposition.FAILED
    assert first.model_call_count == 1
    assert second.disposition is RepairControllerDisposition.IDENTICAL_FAILURE_BACKOFF
    assert second.diagnostic_reused is True
    assert second.backoff_milliseconds == 200
    assert "model_call_suppressed" in second.reason_codes
    assert exhausted.disposition is RepairControllerDisposition.IDENTICAL_FAILURE_EXHAUSTED
    assert exhausted.receipt is not None
    assert exhausted.receipt.terminal_status is TerminalStatus.EXHAUSTED
    assert calls == ["fail:identical-syntax"]
    assert controller.model_call_count == 1
    retry_with_evidence = controller.run(
        _request(
            requested_tier=RepairTier.MODEL_ASSISTED_BOUNDED,
            failure_signature="fail:identical-syntax",
            diagnostic_receipt_id="diag:syntax-v2",
            validation_receipt_ids=(),
            model_invoker=invoker,
            new_evidence=True,
        )
    )
    assert retry_with_evidence.model_call_count == 2
    assert len(calls) == 2


def test_identical_failures_do_not_reinvoke_the_engine(tmp_path: Path) -> None:
    engine = RecordingEngine(tmp_path)
    controller = _controller(engine=engine, repo_root=tmp_path)
    request = _request(
        work_items=({"work_id": "work:repeat", "operation": "catalog.read"},),
        failure_signature="fail:engine",
        diagnostic_receipt_id="diag:engine",
        validation_receipt_ids=(),
    )
    first = controller.run(request)
    second = controller.run(request)
    assert first.engine_call_count == 1
    assert second.disposition is RepairControllerDisposition.IDENTICAL_FAILURE_BACKOFF
    assert second.engine_call_count == 1
    assert len(engine.calls) == 1


def test_r2_autonomous_merge_requires_every_low_risk_condition() -> None:
    outcome = _controller().run(_request())
    assert set(outcome.low_risk_merge_conditions) == set(LOW_RISK_MERGE_CONDITIONS)
    assert all(outcome.low_risk_merge_conditions[name] for name in LOW_RISK_MERGE_CONDITIONS)
    assert outcome.merge_disposition is RepairMergeDisposition.AUTONOMOUS_MERGE_ELIGIBLE
    assert outcome.disposition is RepairControllerDisposition.MERGE_ELIGIBLE
    assert outcome.authorizes_merge is False
    assert outcome.receipt is not None
    assert outcome.receipt.terminal_status is TerminalStatus.SUCCEEDED
    assert outcome.receipt.authorizes_merge is False
    assert outcome.to_dict()["authorizes_merge"] is False
    restored = RepairControllerOutcome.from_dict(outcome.to_dict())
    assert restored.outcome_id == outcome.outcome_id
    assert restored.merge_disposition is RepairMergeDisposition.AUTONOMOUS_MERGE_ELIGIBLE


def test_missing_validation_or_rollback_blocks_autonomous_merge() -> None:
    missing_tests = _controller().run(_request(validation_receipt_ids=()))
    assert missing_tests.merge_disposition is RepairMergeDisposition.WITHHOLD
    assert missing_tests.low_risk_merge_conditions["current_validation_evidence"] is False
    assert missing_tests.disposition is RepairControllerDisposition.REJECTED_MISSING_CHECKS
    policy = _policy(autonomous_merge_enabled=False)
    disabled = AutonomousRepairController(
        envelope=_envelope(policy=policy),
        policy=policy,
    ).run(_request())
    assert disabled.low_risk_merge_conditions["autonomous_merge_enabled"] is False
    assert disabled.merge_disposition is RepairMergeDisposition.WITHHOLD


def test_r3_success_is_proposal_only() -> None:
    policy = _policy(default_level=AutonomyLevel.SELF_REPAIR_ISOLATED)
    envelope = _envelope(
        policy=policy,
        risk=_risk(risk_class=RiskClass.R3_BOUNDED_REPOSITORY_MUTATION),
        autonomy_level=AutonomyLevel.SELF_REPAIR_ISOLATED,
    )
    outcome = AutonomousRepairController(envelope=envelope, policy=policy).run(_request())
    assert outcome.low_risk_merge_conditions["risk_at_most_r2"] is False
    assert outcome.merge_disposition is RepairMergeDisposition.PROPOSE
    assert outcome.disposition is RepairControllerDisposition.PROPOSAL_ONLY
    assert outcome.authorizes_merge is False
    assert outcome.receipt is not None
    assert outcome.receipt.terminal_status is TerminalStatus.SUCCEEDED


def test_r4_and_r5_cannot_autonomously_merge() -> None:
    r4_policy = _policy(default_level=AutonomyLevel.DRY_RUN)
    r4 = AutonomousRepairController(
        envelope=_envelope(
            policy=r4_policy,
            risk=_risk(
                risk_class=RiskClass.R4_SECURITY_OR_PROTOCOL_SENSITIVE,
                security_sensitive=True,
            ),
            autonomy_level=AutonomyLevel.DRY_RUN,
        ),
        policy=r4_policy,
    ).run(_request(execute=False))
    assert r4.merge_disposition is RepairMergeDisposition.WITHHOLD
    r5_policy = _policy(default_level=AutonomyLevel.RECOMMEND)
    r5 = AutonomousRepairController(
        envelope=_envelope(
            policy=r5_policy,
            risk=_risk(
                risk_class=RiskClass.R5_IRREVERSIBLE_EXTERNAL_OR_LEGAL,
                reversible=False,
                irreversible_external_effect=True,
                legal_or_financial_effect=True,
            ),
            autonomy_level=AutonomyLevel.RECOMMEND,
            reversible=False,
        ),
        policy=r5_policy,
    ).run(_request(execute=False))
    assert r5.merge_disposition is RepairMergeDisposition.WITHHOLD


def test_rollback_discards_the_isolated_worktree_without_merge() -> None:
    outcome = _controller().run(_request(rollback=True))
    assert outcome.disposition is RepairControllerDisposition.ROLLBACK
    assert outcome.receipt is not None
    assert outcome.receipt.rollback_receipt_id == "rollback-plan-1"
    assert outcome.receipt.terminal_status is TerminalStatus.CANCELLED
    assert outcome.merge_disposition is RepairMergeDisposition.WITHHOLD
    assert "rollback_plan_bound" in outcome.reason_codes


def test_predetermined_tests_and_proofs_are_required_for_success() -> None:
    policy = _policy()
    envelope = _envelope(
        policy=policy,
        required_test_ids=("test-repair-controller",),
        required_proof_ids=("proof-repair-1",),
    )
    controller = AutonomousRepairController(envelope=envelope, policy=policy)
    missing_proof = controller.run(_request())
    assert missing_proof.disposition is RepairControllerDisposition.REJECTED_MISSING_CHECKS
    complete = controller.run(
        _request(
            required_proof_ids=("proof-repair-1",),
            proof_receipt_ids=("proof-repair-1",),
        )
    )
    assert complete.receipt is not None
    assert complete.receipt.terminal_status is TerminalStatus.SUCCEEDED
    assert complete.low_risk_merge_conditions["required_proofs_satisfied"] is True


def test_forged_merge_authorization_is_rejected() -> None:
    outcome = _controller().run(_request())
    forged = dict(outcome.to_dict())
    forged["authorizes_merge"] = True
    with pytest.raises(RepairControllerError, match="cannot authorize merge"):
        RepairControllerOutcome.from_dict(forged)
    plan = outcome.plan
    assert plan is not None
    assert outcome.receipt is not None
    with pytest.raises(Exception, match="authorize merge"):
        outcome.receipt.__class__(
            plan_id=plan.plan_id,
            envelope_id=outcome.receipt.envelope_id,
            terminal_status=TerminalStatus.SUCCEEDED,
            changed_paths=plan.predicted_files,
            validation_receipt_ids=("test-repair-controller",),
            proof_receipt_ids=(),
            adversarial_assurance_receipt_ids=(),
            authorizes_merge=True,
        )


def test_policy_mismatch_and_unknown_engine_fail_closed() -> None:
    policy = _policy()
    other = _policy(policy_revision="policy-rev-other")
    envelope = _envelope(policy=policy)
    with pytest.raises(RepairControllerError, match="policy_id"):
        AutonomousRepairController(envelope=envelope, policy=other)


def test_snapshot_records_backoff_without_private_payloads() -> None:
    controller = _controller()
    request = _request(
        requested_tier=RepairTier.MODEL_ASSISTED_BOUNDED,
        failure_signature="fail:snap",
        diagnostic_receipt_id="diag:snap",
        validation_receipt_ids=(),
        model_invoker=lambda _request: False,
    )
    controller.run(request)
    snapshot = controller.snapshot()
    assert snapshot["interface"] == AUTONOMOUS_REPAIR_CONTROLLER_INTERFACE
    assert snapshot["engine_interface"] == AUTONOMOUS_REPAIR_INTERFACE
    assert snapshot["authorizes_merge"] is False
    assert snapshot["model_call_count"] == 1
    assert snapshot["failures"]
    assert REPAIR_CONTROLLER_OUTCOME_SCHEMA.endswith("@1")


def test_source_edit_policy_flag_is_not_admission() -> None:
    outcome = _controller().run(_request(allow_code_edit_materialize=True, source_edit=None))
    assert outcome.disposition is RepairControllerDisposition.REJECTED_SOURCE_EDIT
    assert outcome.source_edit_admission is not None
    assert outcome.source_edit_admission.admitted is False
    assert outcome.source_edit_admission.disposition is SourceEditAdmissionDisposition.NOT_SOURCE_EDIT
    assert "policy_flag_is_not_source_edit_admission" in outcome.source_edit_admission.reason_codes
