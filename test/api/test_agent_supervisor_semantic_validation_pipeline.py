from __future__ import annotations

from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.code_proof_obligations import (
    CandidateDiffEntry,
    DiffChangeKind,
    ImplementationObligationSet,
    compile_candidate_proof_scopes,
    derive_fresh_implementation_obligations,
)
from ipfs_accelerate_py.agent_supervisor.formal_plan_conformance import (
    evaluate_completion_admission,
)
from ipfs_accelerate_py.agent_supervisor.proposal_validation import (
    ImplementationProposal,
    ProposalValidationPolicy,
    ProposalValidationResult,
    validate_proposal,
)
from ipfs_accelerate_py.agent_supervisor.validation_scheduler import (
    ImpactDependencyGraph,
    ValidationDAGReceipt,
    ValidationScheduler,
)
from ipfs_accelerate_py.agent_supervisor.validation_commands import (
    ValidationCommand,
)


BEFORE = """\
def transform(value: int) -> int:
    return value + 1
"""
AFTER = """\
def transform(value: int) -> int:
    return value + 2
"""
VALIDATION_ID = "validation:test-service"


def _service_validation() -> ValidationCommand:
    return ValidationCommand(
        command="pytest test/api/test_service.py",
        raw_command="pytest test/api/test_service.py",
        impact_paths=("test/api/test_service.py",),
        validation_id=VALIDATION_ID,
    )


def _proposal(*, after: str = AFTER, path: str = "pkg/core.py"):
    entry = CandidateDiffEntry(
        old_path=path,
        new_path=path,
        change_kind=DiffChangeKind.MODIFY,
        before_source=BEFORE,
        after_source=after,
    )
    proposal = ImplementationProposal(
        task_id="ASI-032",
        accepted_plan_id="plan:strict",
        repository_id="repo:fixture",
        repository_tree_id="tree:candidate",
        objective_id="ASI-G101",
        baseline_id="tree:base",
        candidate_diff=(entry,),
        declared_paths=(path,),
    )
    policy = ProposalValidationPolicy(
        allowed_paths=("pkg/",),
        expected_task_id="ASI-032",
        expected_plan_id="plan:strict",
        expected_repository_id="repo:fixture",
        expected_repository_tree_id="tree:candidate",
        expected_objective_id="ASI-G101",
    )
    return proposal, policy, entry


def _runner(*, spec, **_kwargs):
    return {
        "command": spec.command,
        "returncode": 9,
        "output": "seeded transitive failure",
        "seeded_defect_id": "seed:transitive",
    }


def _passing_runner(*, spec, **_kwargs):
    return {
        "command": spec.command,
        "returncode": 0,
        "output": "validated transitive impact",
    }


def test_rejected_output_cannot_create_semantic_or_code_proof_obligations() -> None:
    proposal, policy, entry = _proposal(after=BEFORE)
    rejected = validate_proposal(proposal, policy=policy)
    scopes = compile_candidate_proof_scopes((entry,))

    assert rejected.accepted is False
    assert rejected.proof_authoritative is False
    assert rejected.completion_authoritative is False
    with pytest.raises(
        ValueError,
        match="rejected proposal cannot produce implementation proof obligations",
    ):
        derive_fresh_implementation_obligations(
            scopes,
            accepted_plan_id=proposal.accepted_plan_id,
            repository_id=proposal.repository_id,
            repository_tree_id=proposal.repository_tree_id,
            proposal_validation=rejected,
        )

    admission = evaluate_completion_admission(
        proposal_validation=rejected,
        required=True,
    )
    assert admission.admitted is False
    assert {
        "proposal_validation_rejected",
        "validation_dag_missing",
    }.issubset(admission.reason_codes)


def test_accepted_proposal_is_bound_into_fresh_code_obligations() -> None:
    proposal, policy, entry = _proposal()
    accepted = validate_proposal(proposal, policy=policy)
    scopes = compile_candidate_proof_scopes((entry,))

    obligations = derive_fresh_implementation_obligations(
        scopes,
        accepted_plan_id=proposal.accepted_plan_id,
        repository_id=proposal.repository_id,
        repository_tree_id=proposal.repository_tree_id,
        proposal_validation=accepted,
    )

    assert accepted.accepted is True
    assert obligations.binding.proposal_accepted is True
    assert (
        obligations.binding.proposal_validation_receipt_id
        == accepted.receipt.receipt_id
    )
    restored = ImplementationObligationSet.from_dict(obligations.to_dict())
    assert restored.binding.binding_id == obligations.binding.binding_id
    assert (
        restored.binding.receipt_metadata()["proposal_validation_receipt_id"]
        == accepted.receipt.receipt_id
    )
    with pytest.raises(ValueError, match="validation DAG receipt is required"):
        derive_fresh_implementation_obligations(
            scopes,
            accepted_plan_id=proposal.accepted_plan_id,
            repository_id=proposal.repository_id,
            repository_tree_id=proposal.repository_tree_id,
            proposal_validation=accepted,
            require_validation_dag=True,
        )


def test_seeded_transitive_failure_blocks_completion_despite_valid_proposal(
    tmp_path: Path,
) -> None:
    proposal, policy, entry = _proposal()
    accepted = validate_proposal(proposal, policy=policy)
    scopes = compile_candidate_proof_scopes((entry,))
    graph = ImpactDependencyGraph(
        repository_tree_id=proposal.repository_tree_id,
        dependencies={
            "pkg/service.py": ("pkg/core.py",),
            "test/api/test_service.py": ("pkg/service.py",),
        },
        validation_targets={
            VALIDATION_ID: ("test/api/test_service.py",),
        },
    )
    report = ValidationScheduler(runner=_runner).run_validated(
        accepted,
        (_service_validation(),),
        workspace_path=tmp_path,
        impact_graph=graph,
        seeded_defect_id="seed:transitive",
        seeded_defect_path="pkg/core.py",
        dependency_state="fixture",
    )
    dag = ValidationDAGReceipt.from_dict(report["validation_dag_receipt"])

    assert dag.passed is False
    assert dag.completion_authoritative is False
    assert report["merge_eligible"] is False
    with pytest.raises(
        ValueError,
        match="failed validation DAG cannot produce implementation proof obligations",
    ):
        derive_fresh_implementation_obligations(
            scopes,
            accepted_plan_id=proposal.accepted_plan_id,
            repository_id=proposal.repository_id,
            repository_tree_id=proposal.repository_tree_id,
            proposal_validation=accepted,
            validation_dag=dag,
            require_validation_dag=True,
        )
    admission = evaluate_completion_admission(
        proposal_validation=accepted,
        validation_dag=dag,
        required=True,
    )
    assert admission.admitted is False
    assert admission.reason_codes == ("validation_dag_failed",)


def test_passing_validation_dag_authority_is_bound_into_obligations(
    tmp_path: Path,
) -> None:
    proposal, policy, entry = _proposal()
    accepted = validate_proposal(proposal, policy=policy)
    scopes = compile_candidate_proof_scopes((entry,))
    graph = ImpactDependencyGraph(
        repository_tree_id=proposal.repository_tree_id,
        dependencies={
            "pkg/service.py": ("pkg/core.py",),
            "test/api/test_service.py": ("pkg/service.py",),
        },
        validation_targets={
            VALIDATION_ID: ("test/api/test_service.py",),
        },
    )
    report = ValidationScheduler(runner=_passing_runner).run_validated(
        accepted,
        (_service_validation(),),
        workspace_path=tmp_path,
        impact_graph=graph,
        validation_policy_id="policy:strict-transitive",
        dependency_state="fixture",
    )
    dag = ValidationDAGReceipt.from_dict(report["validation_dag_receipt"])

    assert dag.passed is True
    # The DAG authorizes downstream proof derivation, but never constitutes
    # completion evidence on its own.
    assert dag.completion_authoritative is False
    obligations = derive_fresh_implementation_obligations(
        scopes,
        accepted_plan_id=proposal.accepted_plan_id,
        repository_id=proposal.repository_id,
        repository_tree_id=proposal.repository_tree_id,
        proposal_validation=accepted,
        validation_dag=dag,
        require_validation_dag=True,
        expected_validation_policy_id="policy:strict-transitive",
    )

    assert obligations.binding.validation_dag_receipt_id == dag.receipt_id
    assert obligations.binding.validation_policy_id == dag.policy_id
    assert (
        obligations.binding.receipt_metadata()["validation_dag_receipt_id"]
        == dag.receipt_id
    )
    restored = ImplementationObligationSet.from_dict(obligations.to_dict())
    assert restored.binding.binding_id == obligations.binding.binding_id
    assert restored.binding.validation_policy_id == "policy:strict-transitive"
    with pytest.raises(ValueError, match="validation DAG policy"):
        derive_fresh_implementation_obligations(
            scopes,
            accepted_plan_id=proposal.accepted_plan_id,
            repository_id=proposal.repository_id,
            repository_tree_id=proposal.repository_tree_id,
            proposal_validation=accepted,
            validation_dag=dag,
            expected_validation_policy_id="policy:other",
        )

    admission = evaluate_completion_admission(
        proposal_validation=accepted,
        validation_dag=dag,
        required=True,
        expected_validation_policy_id="policy:other",
    )
    assert admission.admitted is False
    assert "validation_dag_policy_mismatch" in admission.reason_codes


def test_semantic_bindings_reject_tree_scope_and_receipt_replay() -> None:
    proposal, policy, entry = _proposal()
    accepted = validate_proposal(proposal, policy=policy)
    scopes = compile_candidate_proof_scopes((entry,))

    with pytest.raises(ValueError, match="implementation tree"):
        derive_fresh_implementation_obligations(
            scopes,
            accepted_plan_id=proposal.accepted_plan_id,
            repository_tree_id="tree:other",
            proposal_validation=accepted,
        )

    serialized = accepted.to_dict()
    serialized["receipt"]["repository_tree_id"] = "tree:other"
    with pytest.raises(ValueError):
        ProposalValidationResult.from_dict(serialized)
