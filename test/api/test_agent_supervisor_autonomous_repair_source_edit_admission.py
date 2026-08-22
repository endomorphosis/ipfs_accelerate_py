"""APMC-013: source-edit admission stays fail-closed at the repair facade.

A body-free analysis plan, catalog row, or ``allow_code_edit_materialize``
policy flag cannot admit a mutation.  Only a typed exact source-edit operator
that stays inside the envelope, avoids self-edit, and avoids validator/policy/
key paths can be admitted — and the controller still does not apply bytes.
"""

from __future__ import annotations

import base64
import hashlib
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.autonomous_repair.contracts import RepairWorkItem
from ipfs_accelerate_py.agent_supervisor.autonomous_repair.edit_plan import (
    materialize_admitted_edit_plan,
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
)
from ipfs_accelerate_py.agent_supervisor.autonomy.repair_controller import (
    SELF_EDIT_RELATIVE_PATH,
    AutonomousRepairController,
    RepairControllerDisposition,
    RepairControllerRequest,
    SourceEditAdmissionDisposition,
)


def _digest(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _budget() -> CognitiveBudget:
    return CognitiveBudget(
        max_total_model_calls=2,
        max_strong_model_calls=1,
        max_input_tokens=4_000,
        max_output_tokens=1_000,
        max_provider_spend_micros=10_000,
        max_proof_time_ms=5_000,
        max_validation_time_ms=5_000,
        max_human_questions=1,
        max_repair_rounds=1,
        max_plan_branches=1,
        max_context_expansions=1,
        max_wall_time_ms=10_000,
    )


def _policy() -> AutonomyPolicy:
    return AutonomyPolicy(
        policy_revision="policy-source-edit",
        authority_id="operator-policy-authority",
        human_escalation_policy_id="human-policy-1",
        default_level=AutonomyLevel.EXECUTE_REVERSIBLE,
        autonomous_merge_enabled=False,
    )


def _envelope(policy: AutonomyPolicy, *allowed_paths: str) -> AutonomyEnvelope:
    paths = allowed_paths or ("ipfs_accelerate_py/agent_supervisor/autonomy",)
    return AutonomyEnvelope(
        repository_id="repo-1",
        tree_id="tree-1",
        objective_id="APMC-G000",
        objective_revision="objective-rev-1",
        task_id="APMC-013",
        acceptance_criterion_ids=("AC-source-edit",),
        risk_assessment=RiskAssessment(
            risk_class=RiskClass.R2_REVERSIBLE_LOCAL,
            reversible=True,
            evidence_ids=("evidence-risk",),
            reason_codes=("bounded_local_change",),
        ),
        autonomy_level=AutonomyLevel.EXECUTE_REVERSIBLE,
        cognitive_budget=_budget(),
        allowed_paths=paths,
        allowed_symbols=("AutonomousRepairController", "bounded_repair_target"),
        required_test_ids=("test-source-edit-admission",),
        required_proof_ids=(),
        authority_id="operator-policy-authority",
        policy_id=policy.policy_id,
        provider_usage_envelope_id="provider-envelope-1",
        resource_budget_id="resource-budget-1",
        human_escalation_policy_id="human-policy-1",
        expiry_ms=12_000,
        reversible=True,
    )


def _operator(
    *,
    root: Path,
    relative_path: str,
    old: bytes,
    new: bytes,
    admitted: bool = True,
) -> dict[str, object]:
    old_digest, new_digest = _digest(old), _digest(new)
    return {
        "operator_id": "source-edit:bounded",
        "owner_root": str(root.resolve()),
        "relative_path": relative_path,
        "old_digest": old_digest,
        "new_digest": new_digest,
        "old_bytes_b64": base64.b64encode(old).decode("ascii"),
        "new_bytes_b64": base64.b64encode(new).decode("ascii"),
        "forward_diff": f"--- {old_digest}\n+++ {new_digest}\n+ patched",
        "inverse_diff": f"--- {new_digest}\n+++ {old_digest}\n- patched",
        "disposition": "validation_pending",
        "admitted": admitted,
        "kind": "replace_exact_bytes",
    }


def test_body_free_engine_plan_is_not_source_edit_admission(tmp_path: Path) -> None:
    engine = AutonomousRepairEngine(repo_root=tmp_path)
    plan = materialize_admitted_edit_plan(
        work=RepairWorkItem.from_mapping(
            {"work_id": "work:catalog", "operation": "catalog.read", "path": "surface.py"}
        ),
        disposition="single_path_ready",
        surface=None,
        doctor={"operator": "analytical_transform"},
        ir_doc={"passed": True, "family_ok": {}},
        aliases=(),
        idl_methods=(),
        allow_code_edit_materialize=True,
        domain="agent_supervisor",
    )
    assert plan is not None
    assert plan.source_edit_operator is None
    assert plan.body_free is True
    assert plan.materialize_ready is False
    assert "policy_flag_is_not_source_edit_admission" in plan.materialize_preconditions
    assert "typed_admitted_source_edit_operator_required" in plan.materialize_preconditions

    policy = _policy()
    controller = AutonomousRepairController(
        envelope=_envelope(policy),
        policy=policy,
        engine=engine,
        repo_root=tmp_path,
    )
    outcome = controller.run(
        RepairControllerRequest(
            predicted_files=("ipfs_accelerate_py/agent_supervisor/autonomy/contracts.py",),
            predicted_symbols=("AutonomousRepairController",),
            requested_tier=RepairTier.DETERMINISTIC,
            worktree_id="worktree-isolated-1",
            rollback_plan_id="rollback-plan-1",
            context_reference_ids=("context-ref-1",),
            required_test_ids=("test-source-edit-admission",),
            source_edit=plan.source_edit_operator,
            allow_code_edit_materialize=True,
            execute=True,
        )
    )
    assert outcome.disposition is RepairControllerDisposition.REJECTED_SOURCE_EDIT
    assert outcome.source_edit_admission is not None
    assert outcome.source_edit_admission.admitted is False
    assert outcome.source_edit_admission.mutation_applied is False
    assert (
        outcome.source_edit_admission.disposition
        is SourceEditAdmissionDisposition.NOT_SOURCE_EDIT
    )
    assert "policy_flag_is_not_source_edit_admission" in outcome.source_edit_admission.reason_codes


def test_catalog_analysis_row_cannot_count_as_applied_source_edit() -> None:
    policy = _policy()
    controller = AutonomousRepairController(envelope=_envelope(policy), policy=policy)
    admission = controller.admit_source_edit(
        None,
        predicted_files=("ipfs_accelerate_py/agent_supervisor/autonomy/contracts.py",),
        allow_code_edit_materialize=True,
    )
    assert admission.admitted is False
    assert admission.mutation_applied is False
    assert admission.disposition is SourceEditAdmissionDisposition.NOT_SOURCE_EDIT
    assert "source_edit_operator_missing" in admission.reason_codes


def test_incomplete_or_unadmitted_operator_is_rejected() -> None:
    policy = _policy()
    controller = AutonomousRepairController(envelope=_envelope(policy), policy=policy)
    incomplete = controller.admit_source_edit(
        {"operator_id": "source-edit:incomplete"},
        predicted_files=("ipfs_accelerate_py/agent_supervisor/autonomy/contracts.py",),
    )
    assert incomplete.admitted is False
    assert incomplete.disposition is SourceEditAdmissionDisposition.REJECTED
    assert "typed_admitted_source_edit_operator_required" in incomplete.reason_codes

    unadmitted = controller.admit_source_edit(
        _operator(
            root=Path("/tmp"),
            relative_path="ipfs_accelerate_py/agent_supervisor/autonomy/contracts.py",
            old=b"old",
            new=b"new",
            admitted=False,
        ),
        predicted_files=("ipfs_accelerate_py/agent_supervisor/autonomy/contracts.py",),
    )
    assert unadmitted.admitted is False
    assert unadmitted.disposition is SourceEditAdmissionDisposition.REJECTED


def test_typed_operator_inside_envelope_is_validation_pending_and_does_not_write(
    tmp_path: Path,
) -> None:
    relative = "ipfs_accelerate_py/agent_supervisor/autonomy/contracts.py"
    target = tmp_path.joinpath(*relative.split("/"))
    target.parent.mkdir(parents=True)
    old = b"class AutonomyEnvelope:\n    pass\n"
    new = b"class AutonomyEnvelope:\n    SCHEMA = 'x'\n"
    target.write_bytes(old)
    policy = _policy()
    controller = AutonomousRepairController(
        envelope=_envelope(policy),
        policy=policy,
        repo_root=tmp_path,
    )
    operator = _operator(root=tmp_path, relative_path=relative, old=old, new=new)
    admission = controller.admit_source_edit(operator, predicted_files=(relative,))
    assert admission.admitted is True
    assert admission.mutation_applied is False
    assert admission.authorizes_effect is False
    assert admission.disposition is SourceEditAdmissionDisposition.ADMITTED_VALIDATION_PENDING
    assert admission.old_digest == _digest(old)
    assert admission.new_digest == _digest(new)
    assert target.read_bytes() == old

    outcome = controller.run(
        RepairControllerRequest(
            predicted_files=(relative,),
            predicted_symbols=("AutonomousRepairController",),
            requested_tier=RepairTier.DETERMINISTIC,
            worktree_id="worktree-isolated-1",
            rollback_plan_id="rollback-plan-1",
            context_reference_ids=("context-ref-1",),
            required_test_ids=("test-source-edit-admission",),
            source_edit=operator,
            execute=True,
        )
    )
    assert outcome.source_edit_admission is not None
    assert outcome.source_edit_admission.admitted is True
    assert outcome.source_edit_admission.mutation_applied is False
    assert target.read_bytes() == old
    assert outcome.authorizes_effect is False
    assert outcome.authorizes_merge is False


def test_source_edit_scope_escape_self_edit_and_validator_policy_key_are_rejected(
    tmp_path: Path,
) -> None:
    policy = _policy()
    envelope = _envelope(
        policy,
        "ipfs_accelerate_py/agent_supervisor/autonomy",
        "ipfs_accelerate_py/agent_supervisor/validation",
        "docs",
        "config",
    )
    controller = AutonomousRepairController(
        envelope=envelope,
        policy=policy,
        repo_root=tmp_path,
    )
    old, new = b"old-bytes\n", b"new-bytes\n"

    escaped = controller.admit_source_edit(
        _operator(root=tmp_path, relative_path="README.md", old=old, new=new),
        predicted_files=("ipfs_accelerate_py/agent_supervisor/autonomy/contracts.py",),
    )
    self_edit = controller.admit_source_edit(
        _operator(root=tmp_path, relative_path=SELF_EDIT_RELATIVE_PATH, old=old, new=new),
        predicted_files=(SELF_EDIT_RELATIVE_PATH,),
    )
    validator = controller.admit_source_edit(
        _operator(
            root=tmp_path,
            relative_path="ipfs_accelerate_py/agent_supervisor/validation/proposal_validation.py",
            old=old,
            new=new,
        ),
        predicted_files=(
            "ipfs_accelerate_py/agent_supervisor/validation/proposal_validation.py",
        ),
    )
    policy_key = controller.admit_source_edit(
        _operator(root=tmp_path, relative_path="config/policy.json", old=old, new=new),
        predicted_files=("config/policy.json",),
    )

    assert escaped.admitted is False
    assert "source_edit_path_not_in_plan" in escaped.reason_codes or "scope_escape" in escaped.reason_codes
    assert self_edit.admitted is False
    assert "self_edit" in self_edit.reason_codes
    assert validator.admitted is False
    assert "validator_policy_key" in validator.reason_codes
    assert policy_key.admitted is False
    assert "validator_policy_key" in policy_key.reason_codes
    assert escaped.mutation_applied is False
    assert self_edit.mutation_applied is False
    assert validator.mutation_applied is False
    assert policy_key.mutation_applied is False
