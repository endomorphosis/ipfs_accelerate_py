from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.code_proof_obligations import (
    CandidateDiffEntry,
    DiffChangeKind,
    ImplementationObligationSet,
    PROOF_CANDIDATE_NON_AUTHORITY_ACCEPTANCE_CRITERIA,
    PROOF_CANDIDATE_NON_AUTHORITY_COMPLETION_ANALYZER_VERSION,
    PROOF_CANDIDATE_NON_AUTHORITY_COMPLETION_CONFIGURATION_REVISION,
    PROOF_CANDIDATE_NON_AUTHORITY_OBJECTIVE_ID,
    PROOF_CANDIDATE_NON_AUTHORITY_OBJECTIVE_REVISION,
    PROOF_CANDIDATE_NON_AUTHORITY_REQUIREMENT_ID,
    ProofCandidateNonAuthorityEvidence,
    StrictValidationProofCompletionEvidence,
    compile_candidate_proof_scopes,
    derive_fresh_implementation_obligations,
    prove_proof_candidate_non_authority,
    validate_code_proof_receipt_bindings,
)
from ipfs_accelerate_py.agent_supervisor.formal_plan_conformance import (
    CompletionAdmissionGate,
    CompletionEvidenceKind,
    CompletionPolicy,
    ConformanceBinding,
    EvidenceCheckStatus,
    FormalCompletionEvidence,
    STRICT_VALIDATION_ACCEPTANCE_CRITERIA,
    STRICT_VALIDATION_CHILD_GOAL_IDS,
    STRICT_VALIDATION_COMPLETION_ANALYZER_VERSION,
    STRICT_VALIDATION_COMPLETION_CONFIGURATION_REVISION,
    STRICT_VALIDATION_GATE_KINDS,
    STRICT_VALIDATION_OBJECTIVE_ID,
    STRICT_VALIDATION_OBJECTIVE_REVISION,
    STRICT_VALIDATION_PRODUCING_TASK_IDS,
    STRICT_VALIDATION_REQUIRED_EXHAUSTIVE_RECEIPTS,
    evaluate_completion_admission,
    evaluate_completion_evidence,
    evaluate_strict_validation_completion,
    evaluate_transitive_impact_admission_closure,
)
from ipfs_accelerate_py.agent_supervisor.formal_verification_contracts import (
    AssuranceLevel,
    EvidenceAuthority,
    EvidenceKind,
    EvidenceVerdict,
    ProofEvidence,
    ProofReceipt,
    ProofVerdict,
    ResourceBudget,
)
from ipfs_accelerate_py.agent_supervisor.proposal_validation import (
    ImplementationProposal,
    NOOP_OR_OUT_OF_SCOPE_FAIL_FAST_ACCEPTANCE_CRITERIA,
    NOOP_OR_OUT_OF_SCOPE_FAIL_FAST_COMPLETION_ANALYZER_VERSION,
    NOOP_OR_OUT_OF_SCOPE_FAIL_FAST_COMPLETION_CONFIGURATION_REVISION,
    NOOP_OR_OUT_OF_SCOPE_FAIL_FAST_OBJECTIVE_ID,
    NOOP_OR_OUT_OF_SCOPE_FAIL_FAST_OBJECTIVE_REVISION,
    NOOP_OR_OUT_OF_SCOPE_FAIL_FAST_REQUIREMENT_ID,
    ProposalValidationPolicy,
    ProposalValidationResult,
    validate_proposal,
)
from ipfs_accelerate_py.agent_supervisor.validation_scheduler import (
    ImpactDependencyGraph,
    TRANSITIVE_IMPACT_ACCEPTANCE_CRITERIA,
    TRANSITIVE_IMPACT_COMPLETION_ANALYZER_VERSION,
    TRANSITIVE_IMPACT_COMPLETION_CONFIGURATION_REVISION,
    TRANSITIVE_IMPACT_OBJECTIVE_ID,
    TRANSITIVE_IMPACT_OBJECTIVE_REVISION,
    TRANSITIVE_IMPACT_REQUIREMENT_ID,
    ValidationDAGReceipt,
    ValidationScheduler,
)
from ipfs_accelerate_py.agent_supervisor.goal_completion import (
    CompletionEvidence,
    GoalState,
)
from ipfs_accelerate_py.agent_supervisor.goal_coverage import (
    AcceptanceCoverage,
    CoverageStatus,
    GoalCoverageMap,
    ValidationReceiptCoverage,
)
from ipfs_accelerate_py.agent_supervisor.scan_receipts import (
    ExhaustionBinding,
    ExhaustionQuorumMember,
    ExhaustionQuorumResult,
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


def _proposal(
    *,
    after: str = AFTER,
    path: str = "pkg/core.py",
    task_id: str = "ASI-032",
    objective_id: str = "ASI-G101",
):
    entry = CandidateDiffEntry(
        old_path=path,
        new_path=path,
        change_kind=DiffChangeKind.MODIFY,
        before_source=BEFORE,
        after_source=after,
    )
    proposal = ImplementationProposal(
        task_id=task_id,
        accepted_plan_id="plan:strict",
        repository_id="repo:fixture",
        repository_tree_id="tree:candidate",
        objective_id=objective_id,
        baseline_id="tree:base",
        candidate_diff=(entry,),
        declared_paths=(path,),
    )
    policy = ProposalValidationPolicy(
        allowed_paths=("pkg/",),
        expected_task_id=task_id,
        expected_plan_id="plan:strict",
        expected_repository_id="repo:fixture",
        expected_repository_tree_id="tree:candidate",
        expected_objective_id=objective_id,
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


def _g100_completion_packet(tmp_path: Path) -> tuple[
    ProposalValidationResult,
    tuple[dict[str, str], ...],
    dict[str, object],
]:
    proposal, policy, _entry = _proposal(
        after=BEFORE,
        task_id="ASI-031",
        objective_id=NOOP_OR_OUT_OF_SCOPE_FAIL_FAST_OBJECTIVE_ID,
    )
    attempted_commands: list[str] = []
    report = ValidationScheduler().run_validated(
        validate_proposal(proposal, policy=policy),
        (_service_validation(),),
        workspace_path=tmp_path,
        dependency_state="g100-fixture",
        runner=lambda *, spec, **_kwargs: attempted_commands.append(
            spec.command
        ),
    )
    assert attempted_commands == []
    witness = ProposalValidationResult.from_dict(
        report["proposal_validation"]
    )
    assert not witness.accepted
    assert witness.receipt.rejection_evidence is not None
    now = datetime(2026, 7, 25, 12, 0, tzinfo=timezone.utc)
    rejection = witness.receipt.rejection_evidence
    validation_binding = {
        "status": "passed",
        "repository_id": proposal.repository_id,
        "tree_id": proposal.repository_tree_id,
        "requirement_id": NOOP_OR_OUT_OF_SCOPE_FAIL_FAST_REQUIREMENT_ID,
        "objective_id": NOOP_OR_OUT_OF_SCOPE_FAIL_FAST_OBJECTIVE_ID,
        "validation_policy_id": policy.policy_id,
        "operational_receipt_id": rejection.evidence_id,
        "command": (
            "python -m pytest "
            "test/api/test_agent_supervisor_proposal_validation.py "
            "test/api/test_agent_supervisor_validation_dag.py "
            "test/api/test_agent_supervisor_semantic_validation_pipeline.py -q"
        ),
    }
    completion_evidence = tuple(
        CompletionEvidence(
            acceptance_criterion=criterion,
            producing_task_or_scan="ASI-091",
            validation_receipt=validation_binding,
            validation_passed=True,
            repository_tree=proposal.repository_tree_id,
            freshness={"fresh": True},
            observed_at=now,
            provenance_cid=f"validation:asi-091:{index}",
            metadata={
                "evidence_source_policy": {
                    "satisfies": True,
                    "source_tier": "validation_receipt",
                }
            },
        )
        for index, criterion in enumerate(
            NOOP_OR_OUT_OF_SCOPE_FAIL_FAST_ACCEPTANCE_CRITERIA,
            start=1,
        )
    )
    coverage = GoalCoverageMap(
        criteria=[
            AcceptanceCoverage(
                criterion_id=f"g100-{index}",
                goal_id=NOOP_OR_OUT_OF_SCOPE_FAIL_FAST_OBJECTIVE_ID,
                criterion=item.acceptance_criterion,
                status=CoverageStatus.VERIFIED,
                task_ids=["ASI-031", "ASI-091"],
                changed_files=[
                    "ipfs_accelerate_py/agent_supervisor/"
                    "proposal_validation.py"
                ],
                validation_receipt_ids=[item.provenance_cid],
                provenance_cids=[item.provenance_cid],
            )
            for index, item in enumerate(completion_evidence, start=1)
        ],
        edges=[],
        receipts=[
            ValidationReceiptCoverage(
                receipt_id=item.provenance_cid,
                task_id="ASI-091",
                criterion=item.acceptance_criterion,
                command=validation_binding["command"],
                status=CoverageStatus.VERIFIED,
                passed=True,
                repository_tree=proposal.repository_tree_id,
                observed_at=now.isoformat(),
                provenance_cid=item.provenance_cid,
                explanation="Fresh ASI-G100 validation passed.",
                outcome="passed",
                reason_code="validation_passed",
                fresh=True,
            )
            for item in completion_evidence
        ],
        finding_assignments=[],
        registered_goal_ids=[
            NOOP_OR_OUT_OF_SCOPE_FAIL_FAST_OBJECTIVE_ID
        ],
        evaluated_at=now.isoformat(),
        repository_tree=proposal.repository_tree_id,
    )
    analyzer_binding = {
        "repository_id": proposal.repository_id,
        "tree_id": proposal.repository_tree_id,
        "objective_revision": (
            NOOP_OR_OUT_OF_SCOPE_FAIL_FAST_OBJECTIVE_REVISION
        ),
        "analyzer_version": (
            NOOP_OR_OUT_OF_SCOPE_FAIL_FAST_COMPLETION_ANALYZER_VERSION
        ),
        "configuration_revision": (
            NOOP_OR_OUT_OF_SCOPE_FAIL_FAST_COMPLETION_CONFIGURATION_REVISION
        ),
    }
    health = {
        "status": "healthy",
        "healthy": True,
        "safe_for_completion_reasoning": True,
        "exhaustive": True,
        "binding": analyzer_binding,
    }
    quorum = {
        "required_members": 2,
        "member_count": 2,
        "satisfied": True,
        "quorum_met": True,
        "binding": analyzer_binding,
        "members": [
            {
                "member_id": "asi-091-implementation",
                "evidence_channel": "implementation-validation",
                "receipt_cid": "scan:asi-091:implementation",
                "binding": analyzer_binding,
                "scan_mode": "exhaustive",
                "analyzer_version": (
                    NOOP_OR_OUT_OF_SCOPE_FAIL_FAST_COMPLETION_ANALYZER_VERSION
                    + "/implementation"
                ),
                "passed": True,
                "analyzer_health": {
                    "status": "healthy",
                    "healthy": True,
                },
                "exhaustive": True,
                "healthy": True,
                "safe_for_completion_reasoning": True,
                "conclusive": True,
                "contradicted": False,
                "finished_at": now.isoformat(),
            },
            {
                "member_id": "asi-091-replay",
                "evidence_channel": "independent-receipt-replay",
                "receipt_cid": "scan:asi-091:replay",
                "binding": analyzer_binding,
                "scan_mode": "exhaustive",
                "analyzer_version": (
                    NOOP_OR_OUT_OF_SCOPE_FAIL_FAST_COMPLETION_ANALYZER_VERSION
                    + "/replay"
                ),
                "passed": True,
                "analyzer_health": {
                    "status": "healthy",
                    "healthy": True,
                },
                "exhaustive": True,
                "healthy": True,
                "safe_for_completion_reasoning": True,
                "conclusive": True,
                "contradicted": False,
                "finished_at": now.isoformat(),
            },
        ],
    }
    values: dict[str, object] = {
        "evidence": completion_evidence,
        "tasks_complete": True,
        "coverage": coverage,
        "analyzer_health": health,
        "exhaustion_quorum": quorum,
        "now": now,
        "freshness_seconds": 300,
    }
    return witness, ({"task_id": "ASI-031", "status": "completed"},), values


def test_g100_completion_requires_exact_current_tree_evidence_population(
    tmp_path: Path,
) -> None:
    witness, producing_tasks, values = _g100_completion_packet(tmp_path)

    provisional = witness.evaluate_objective_completion(
        producing_tasks=producing_tasks,
        current_state=GoalState.ACTIVE,
        **values,
    )
    assert provisional.state is GoalState.PROVISIONALLY_COMPLETE
    assert provisional.gate is not None and provisional.gate.passed
    assert not provisional.verified

    verified = witness.evaluate_objective_completion(
        producing_tasks=producing_tasks,
        current_state=GoalState.PROVISIONALLY_COMPLETE,
        **values,
    )
    assert verified.state is GoalState.VERIFIED_COMPLETE
    assert verified.verified

    for invalid_producers in (
        (),
        ({"task_id": "ASI-031", "status": "active"},),
        ({"task_id": "ASI-091", "status": "completed"},),
        (
            {"task_id": "ASI-031", "status": "completed"},
            {"task_id": "ASI-031", "status": "completed"},
        ),
    ):
        rejected = witness.evaluate_objective_completion(
            producing_tasks=invalid_producers,
            current_state=GoalState.PROVISIONALLY_COMPLETE,
            **values,
        )
        assert rejected.state is GoalState.REOPENED
        assert not rejected.verified
        assert "tasks_incomplete" in rejected.reason_codes


def test_g100_completion_rejects_detached_unsafe_or_nonindependent_proof(
    tmp_path: Path,
) -> None:
    witness, producing_tasks, values = _g100_completion_packet(tmp_path)
    evidence = values["evidence"]
    assert isinstance(evidence, tuple)
    first = evidence[0]
    assert isinstance(first, CompletionEvidence)

    detached_evidence = (
        CompletionEvidence.from_dict(
            {
                **first.to_dict(),
                "validation_receipt": {
                    **first.validation_receipt,
                    "operational_receipt_id": "evidence:foreign",
                },
            }
        ),
        *evidence[1:],
    )
    cases: list[dict[str, object]] = [
        {**values, "evidence": detached_evidence},
        {**values, "evidence": evidence[:-1]},
        {
            **values,
            "analyzer_health": {
                **values["analyzer_health"],
                "safe_for_completion_reasoning": False,
            },
        },
    ]
    duplicate_quorum = deepcopy(values["exhaustion_quorum"])
    duplicate_quorum["members"][1]["receipt_cid"] = (
        duplicate_quorum["members"][0]["receipt_cid"]
    )
    cases.append({**values, "exhaustion_quorum": duplicate_quorum})
    stale_quorum = deepcopy(values["exhaustion_quorum"])
    stale_quorum["members"][1]["finished_at"] = (
        values["now"] - timedelta(hours=1)
    ).isoformat()
    cases.append({**values, "exhaustion_quorum": stale_quorum})
    non_exhaustive_quorum = deepcopy(values["exhaustion_quorum"])
    non_exhaustive_quorum["members"][1]["exhaustive"] = False
    cases.append({**values, "exhaustion_quorum": non_exhaustive_quorum})
    implicit_health_quorum = deepcopy(values["exhaustion_quorum"])
    del implicit_health_quorum["members"][1]["healthy"]
    del implicit_health_quorum["members"][1]["analyzer_health"]
    cases.append({**values, "exhaustion_quorum": implicit_health_quorum})
    typed_coverage = values["coverage"]
    assert isinstance(typed_coverage, GoalCoverageMap)
    unbound_coverage = typed_coverage.completion_gate_evidence(
        NOOP_OR_OUT_OF_SCOPE_FAIL_FAST_OBJECTIVE_ID
    )
    unbound_coverage["criteria"][0]["validation_receipt_ids"] = [
        "validation:foreign"
    ]
    cases.append({**values, "coverage": unbound_coverage})

    for invalid_values in cases:
        rejected = witness.evaluate_objective_completion(
            producing_tasks=producing_tasks,
            current_state=GoalState.PROVISIONALLY_COMPLETE,
            **invalid_values,
        )
        assert rejected.state is GoalState.PROVISIONALLY_COMPLETE
        assert not rejected.verified
        assert rejected.gate is not None and not rejected.gate.passed

    with pytest.raises(ValueError, match="configured ASI-G100 count 2"):
        witness.evaluate_objective_completion(
            producing_tasks=producing_tasks,
            current_state=GoalState.PROVISIONALLY_COMPLETE,
            required_exhaustive_receipts=1,
            **values,
        )


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
    closure = evaluate_transitive_impact_admission_closure(
        proposal_validation=accepted,
        validation_dag=dag,
    )
    assert closure == admission


def test_g101_objective_repair_requires_closed_two_phase_proof(
    tmp_path: Path,
) -> None:
    proposal, policy, _entry = _proposal()
    accepted = validate_proposal(proposal, policy=policy)
    graph = ImpactDependencyGraph(
        repository_tree_id=proposal.repository_tree_id,
        dependencies={
            "pkg/service.py": ("pkg/core.py",),
            "test/api/test_service.py": ("pkg/service.py",),
        },
        validation_targets={VALIDATION_ID: ("test/api/test_service.py",)},
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
    now = datetime(2026, 7, 24, 15, 0, tzinfo=timezone.utc)
    validation_binding = {
        "status": "passed",
        "tree_id": dag.repository_tree_id,
        "requirement_id": TRANSITIVE_IMPACT_REQUIREMENT_ID,
        "objective_id": TRANSITIVE_IMPACT_OBJECTIVE_ID,
        "operational_receipt_id": dag.receipt_id,
        "validation_policy_id": dag.policy_id,
        "command": (
            "python -m pytest "
            "test/api/test_agent_supervisor_proposal_validation.py "
            "test/api/test_agent_supervisor_validation_dag.py "
            "test/api/test_agent_supervisor_semantic_validation_pipeline.py -q"
        ),
    }
    evidence = tuple(
        CompletionEvidence(
            acceptance_criterion=criterion,
            producing_task_or_scan="ASI-075",
            validation_receipt=validation_binding,
            validation_passed=True,
            repository_tree=dag.repository_tree_id,
            freshness={"fresh": True},
            observed_at=now,
            provenance_cid=f"validation:asi-075:{index}",
            metadata={
                "evidence_source_policy": {
                    "satisfies": True,
                    "source_tier": "validation_receipt",
                }
            },
        )
        for index, criterion in enumerate(
            TRANSITIVE_IMPACT_ACCEPTANCE_CRITERIA,
            start=1,
        )
    )
    coverage = {
        "repository_tree": dag.repository_tree_id,
        "evaluated_at": now.isoformat(),
        "verified": True,
        "criteria": [
            {
                "criterion": criterion,
                "status": "verified",
                "verified": True,
                "implementation": (
                    "ipfs_accelerate_py/agent_supervisor/"
                    "validation_scheduler.py"
                ),
                "validation": (
                    "test/api/test_agent_supervisor_validation_dag.py"
                ),
            }
            for criterion in TRANSITIVE_IMPACT_ACCEPTANCE_CRITERIA
        ],
    }
    health = {
        "status": "healthy",
        "healthy": True,
        "safe_for_completion_reasoning": True,
        "exhaustive": True,
        "analyzer_version": TRANSITIVE_IMPACT_COMPLETION_ANALYZER_VERSION,
    }
    binding = {
        "tree_id": dag.repository_tree_id,
        "objective_id": TRANSITIVE_IMPACT_OBJECTIVE_ID,
        "objective_revision": TRANSITIVE_IMPACT_OBJECTIVE_REVISION,
        "validation_policy_id": dag.policy_id,
        "operational_receipt_id": dag.receipt_id,
        "analyzer_version": TRANSITIVE_IMPACT_COMPLETION_ANALYZER_VERSION,
        "configuration_revision": (
            TRANSITIVE_IMPACT_COMPLETION_CONFIGURATION_REVISION
        ),
    }
    quorum = {
        "required_members": 2,
        "member_count": 2,
        "satisfied": True,
        "quorum_met": True,
        "binding": binding,
        "members": [
            {
                "member_id": "asi-075-implementation",
                "evidence_channel": "implementation-validation",
                "receipt_cid": "scan:asi-075:implementation",
                "binding": binding,
                "scan_mode": "exhaustive",
                "analyzer_version": (
                    TRANSITIVE_IMPACT_COMPLETION_ANALYZER_VERSION
                    + "/implementation"
                ),
                "passed": True,
                "analyzer_health": {
                    "status": "healthy",
                    "healthy": True,
                },
                "exhaustive": True,
                "healthy": True,
                "safe_for_completion_reasoning": True,
                "conclusive": True,
                "contradicted": False,
                "finished_at": now.isoformat(),
            },
            {
                "member_id": "asi-075-replay",
                "evidence_channel": "receipt-replay",
                "receipt_cid": "scan:asi-075:replay",
                "binding": binding,
                "scan_mode": "exhaustive",
                "analyzer_version": (
                    TRANSITIVE_IMPACT_COMPLETION_ANALYZER_VERSION
                    + "/replay"
                ),
                "passed": True,
                "analyzer_health": {
                    "status": "healthy",
                    "healthy": True,
                },
                "exhaustive": True,
                "healthy": True,
                "safe_for_completion_reasoning": True,
                "conclusive": True,
                "contradicted": False,
                "finished_at": now.isoformat(),
            },
        ],
    }
    values = {
        "proposal_validation": accepted,
        "evidence": evidence,
        "tasks_complete": True,
        "coverage": coverage,
        "analyzer_health": health,
        "exhaustion_quorum": quorum,
        "now": now,
        "freshness_seconds": 300,
    }

    provisional = dag.evaluate_objective_completion(
        current_state=GoalState.ACTIVE,
        **values,
    )
    assert provisional.state is GoalState.PROVISIONALLY_COMPLETE
    assert provisional.gate is not None and provisional.gate.passed
    assert not provisional.verified

    verified = dag.evaluate_objective_completion(
        current_state=GoalState.PROVISIONALLY_COMPLETE,
        **values,
    )
    assert verified.state is GoalState.VERIFIED_COMPLETE
    assert verified.verified

    canonical_receipts = [
        ValidationReceiptCoverage(
            receipt_id=item.provenance_cid,
            task_id="ASI-075",
            criterion=item.acceptance_criterion,
            command=validation_binding["command"],
            status=CoverageStatus.VERIFIED,
            passed=True,
            repository_tree=dag.repository_tree_id,
            observed_at=now.isoformat(),
            provenance_cid=item.provenance_cid,
            explanation="fresh passing ASI-075 criterion validation",
            outcome="passed",
            reason_code="validation_verified",
            fresh=True,
        )
        for item in evidence
    ]
    canonical_coverage = GoalCoverageMap(
        criteria=[
            AcceptanceCoverage(
                criterion_id=f"criterion:g101:{index}",
                goal_id=TRANSITIVE_IMPACT_OBJECTIVE_ID,
                criterion=criterion,
                status=CoverageStatus.VERIFIED,
                changed_files=[
                    "ipfs_accelerate_py/agent_supervisor/"
                    "validation_scheduler.py"
                ],
                validation_receipt_ids=[
                    evidence[index - 1].provenance_cid
                ],
                explanation="implementation and validation are exact",
            )
            for index, criterion in enumerate(
                TRANSITIVE_IMPACT_ACCEPTANCE_CRITERIA,
                start=1,
            )
        ],
        edges=[],
        receipts=canonical_receipts,
        finding_assignments=[],
        registered_goal_ids=[TRANSITIVE_IMPACT_OBJECTIVE_ID],
        evaluated_at=now.isoformat(),
        repository_tree=dag.repository_tree_id,
    )
    typed_binding = ExhaustionBinding(
        repository_id="repo:fixture",
        tree_id=dag.repository_tree_id,
        analyzer_version=TRANSITIVE_IMPACT_COMPLETION_ANALYZER_VERSION,
        configuration_revision=(
            TRANSITIVE_IMPACT_COMPLETION_CONFIGURATION_REVISION
        ),
        objective_revision=TRANSITIVE_IMPACT_OBJECTIVE_REVISION,
    )
    typed_quorum = ExhaustionQuorumResult(
        binding=typed_binding,
        required_members=2,
        members=(
            ExhaustionQuorumMember(
                member_id="asi-075-typed-implementation",
                evidence_channel="implementation-validation",
                receipt_cid="scan:asi-075:typed-implementation",
                binding=typed_binding,
                scan_mode="exhaustive",
                finished_at=now.isoformat(),
            ),
            ExhaustionQuorumMember(
                member_id="asi-075-typed-replay",
                evidence_channel="receipt-replay",
                receipt_cid="scan:asi-075:typed-replay",
                binding=typed_binding,
                scan_mode="exhaustive",
                finished_at=now.isoformat(),
            ),
        ),
    )
    typed_quorum_payload = typed_quorum.to_dict()
    typed_artifact_binding = {
        "objective_id": TRANSITIVE_IMPACT_OBJECTIVE_ID,
        "validation_policy_id": dag.policy_id,
        "operational_receipt_id": dag.receipt_id,
    }
    typed_quorum_payload["binding"].update(typed_artifact_binding)
    for index, member in enumerate(typed_quorum_payload["members"]):
        member["binding"].update(typed_artifact_binding)
        member.update(
            {
                "analyzer_version": (
                    TRANSITIVE_IMPACT_COMPLETION_ANALYZER_VERSION
                    + ("/implementation" if index == 0 else "/replay")
                ),
                "passed": True,
                "analyzer_health": {
                    "status": "healthy",
                    "healthy": True,
                },
                "exhaustive": True,
                "healthy": True,
                "safe_for_completion_reasoning": True,
                "conclusive": True,
                "contradicted": False,
            }
        )
    canonical = dag.evaluate_objective_completion(
        current_state=GoalState.PROVISIONALLY_COMPLETE,
        **{
            **values,
            "coverage": canonical_coverage,
            "exhaustion_quorum": typed_quorum_payload,
        },
    )
    assert canonical.state is GoalState.VERIFIED_COMPLETE, canonical.reason_codes
    assert canonical.verified

    unbound = tuple(
        CompletionEvidence.from_dict(
            {
                **item.to_dict(),
                "validation_receipt": {
                    **validation_binding,
                    "operational_receipt_id": "receipt:foreign",
                },
            }
        )
        for item in evidence
    )
    rejected = dag.evaluate_objective_completion(
        current_state=GoalState.PROVISIONALLY_COMPLETE,
        **{**values, "evidence": unbound},
    )
    assert rejected.state is GoalState.PROVISIONALLY_COMPLETE
    assert not rejected.verified
    assert rejected.gate is not None and not rejected.gate.passed

    unsafe = dag.evaluate_objective_completion(
        current_state=GoalState.PROVISIONALLY_COMPLETE,
        **{
            **values,
            "analyzer_health": {
                **health,
                "safe_for_completion_reasoning": False,
            },
        },
    )
    assert unsafe.state is GoalState.PROVISIONALLY_COMPLETE
    assert "analyzer_unhealthy" in unsafe.reason_codes

    duplicate_quorum = deepcopy(quorum)
    duplicate_quorum["members"][1]["receipt_cid"] = (
        duplicate_quorum["members"][0]["receipt_cid"]
    )
    no_quorum = dag.evaluate_objective_completion(
        current_state=GoalState.PROVISIONALLY_COMPLETE,
        **{**values, "exhaustion_quorum": duplicate_quorum},
    )
    assert no_quorum.state is GoalState.PROVISIONALLY_COMPLETE
    assert any(
        code.startswith("exhaustion_quorum")
        for code in no_quorum.reason_codes
    )

    rejected_evidence_populations = [
        evidence[:-1],
        (
            *evidence,
            CompletionEvidence.from_dict(
                {
                    **evidence[0].to_dict(),
                    "validation_passed": False,
                    "validation_receipt": {
                        **validation_binding,
                        "status": "failed",
                    },
                    "provenance_cid": "validation:asi-075:failed-extra",
                }
            ),
        ),
        tuple(
            CompletionEvidence.from_dict(
                {
                    **item.to_dict(),
                    **(
                        {
                            "observed_at": (
                                now.replace(hour=14, minute=0).isoformat()
                            )
                        }
                        if index == 0
                        else {}
                    ),
                }
            )
            for index, item in enumerate(evidence)
        ),
        tuple(
            CompletionEvidence.from_dict(
                {
                    **item.to_dict(),
                    **(
                        {
                            "acceptance_criterion": evidence[0].acceptance_criterion,
                        }
                        if index == len(evidence) - 1
                        else {}
                    ),
                }
            )
            for index, item in enumerate(evidence)
        ),
    ]
    for rejected_evidence in rejected_evidence_populations:
        decision = dag.evaluate_objective_completion(
            current_state=GoalState.PROVISIONALLY_COMPLETE,
            **{**values, "evidence": rejected_evidence},
        )
        assert decision.state is GoalState.PROVISIONALLY_COMPLETE
        assert not decision.verified
        assert decision.gate is not None and not decision.gate.passed

    invalid_health_records = (
        {},
        {**health, "healthy": False},
        {**health, "analyzer_version": "asi-g101:foreign"},
    )
    for invalid_health in invalid_health_records:
        decision = dag.evaluate_objective_completion(
            current_state=GoalState.PROVISIONALLY_COMPLETE,
            **{**values, "analyzer_health": invalid_health},
        )
        assert decision.state is GoalState.PROVISIONALLY_COMPLETE
        assert not decision.verified

    invalid_quorums: list[dict[str, object]] = []
    insufficient = deepcopy(quorum)
    insufficient["members"] = insufficient["members"][:1]
    insufficient["member_count"] = 1
    invalid_quorums.append(insufficient)
    stale_member = deepcopy(quorum)
    stale_member["members"][1]["finished_at"] = now.replace(
        hour=14,
        minute=0,
    ).isoformat()
    invalid_quorums.append(stale_member)
    non_exhaustive = deepcopy(quorum)
    non_exhaustive["members"][1]["scan_mode"] = "partial"
    invalid_quorums.append(non_exhaustive)
    unhealthy_member = deepcopy(quorum)
    unhealthy_member["members"][1]["healthy"] = False
    invalid_quorums.append(unhealthy_member)
    foreign_tree = deepcopy(quorum)
    foreign_tree["binding"]["tree_id"] = "tree:foreign"
    invalid_quorums.append(foreign_tree)
    inconsistent_count = deepcopy(quorum)
    inconsistent_count["member_count"] = 3
    invalid_quorums.append(inconsistent_count)
    for invalid_quorum in invalid_quorums:
        decision = dag.evaluate_objective_completion(
            current_state=GoalState.PROVISIONALLY_COMPLETE,
            **{**values, "exhaustion_quorum": invalid_quorum},
        )
        assert decision.state is GoalState.PROVISIONALLY_COMPLETE
        assert not decision.verified
        assert decision.gate is not None and not decision.gate.passed


def test_formal_completion_rejects_any_submitted_invalid_receipt() -> None:
    binding = ConformanceBinding(
        plan_id="plan:asi-075",
        policy_id="policy:asi-075",
        repository_tree_id="tree:asi-075",
    )
    policy = CompletionPolicy(
        required_evidence=(CompletionEvidenceKind.TEST,),
    )
    passing = FormalCompletionEvidence(
        kind=CompletionEvidenceKind.TEST,
        goal_id=TRANSITIVE_IMPACT_OBJECTIVE_ID,
        artifact_id="receipt:passing",
        binding=binding,
        observed_at="2026-07-24T15:00:00Z",
        verdict="passed",
        freshness="current",
    )
    failed = FormalCompletionEvidence(
        kind=CompletionEvidenceKind.TEST,
        goal_id=TRANSITIVE_IMPACT_OBJECTIVE_ID,
        artifact_id="receipt:failed",
        binding=binding,
        observed_at="2026-07-24T15:00:00Z",
        verdict="failed",
        freshness="current",
    )

    result = evaluate_completion_evidence(
        TRANSITIVE_IMPACT_OBJECTIVE_ID,
        (passing, failed),
        policy=policy,
        binding=binding,
        evaluated_at="2026-07-24T15:00:01Z",
    )

    assert not result.satisfied
    assert result.checks[0].status is EvidenceCheckStatus.FAILED
    assert set(result.checks[0].evidence_ids) == {
        passing.evidence_id,
        failed.evidence_id,
    }


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
    no_proof = evaluate_completion_admission(
        proposal_validation=accepted,
        validation_dag=dag,
        required=True,
        expected_validation_policy_id=dag.policy_id,
    )
    assert no_proof.admitted is False
    assert no_proof.reason_codes == ("code_proof_missing",)
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


def _proof_receipt(
    obligations: ImplementationObligationSet,
    *,
    authoritative: bool,
    obligation_index: int = 0,
) -> ProofReceipt:
    obligation = obligations.obligations[obligation_index]
    evidence = ProofEvidence(
        kind=(
            EvidenceKind.KERNEL_VERIFICATION
            if authoritative
            else EvidenceKind.ATP_CANDIDATE
        ),
        authority=(
            EvidenceAuthority.KERNEL
            if authoritative
            else EvidenceAuthority.ATP
        ),
        verdict=(
            EvidenceVerdict.ACCEPTED
            if authoritative
            else EvidenceVerdict.CANDIDATE
        ),
        artifact_id=(
            "artifact:independent-kernel"
            if authoritative
            else "artifact:provider-candidate"
        ),
        subject_id=obligation.obligation_id,
        verifier_id=(
            "kernel:strict"
            if authoritative
            else "provider:optimistic-atp"
        ),
        independent=authoritative,
    )
    return ProofReceipt(
        obligation_id=obligation.obligation_id,
        plan_id=obligations.binding.accepted_plan_id,
        attempt_id=(
            "attempt:kernel" if authoritative else "attempt:provider-candidate"
        ),
        repository_id=obligations.binding.repository_id,
        repository_tree_id=obligations.binding.repository_tree_id,
        ast_scope_ids=obligation.ast_scope_ids,
        premise_ids=obligation.premise_ids,
        translator_id="translator:strict",
        solver_id="solver:strict",
        kernel_id="kernel:strict",
        toolchain_id="toolchain:locked",
        policy_id="policy:strict-code-proof",
        resource_budget=ResourceBudget(wall_time_ms=10_000),
        verdict=ProofVerdict.PROVED,
        evidence=(evidence,),
        # This hostile claim is intentionally stronger than the evidence.
        provider_claimed_assurance=AssuranceLevel.ATTESTED,
        metadata=obligations.binding.receipt_metadata(),
    )


def _passing_authority_chain(
    tmp_path: Path,
    *,
    task_id: str = "ASI-070",
    objective_id: str = PROOF_CANDIDATE_NON_AUTHORITY_OBJECTIVE_ID,
) -> tuple[
    ProposalValidationResult,
    ValidationDAGReceipt,
    ImplementationObligationSet,
]:
    proposal, policy, entry = _proposal(
        task_id=task_id,
        objective_id=objective_id,
    )
    accepted = validate_proposal(proposal, policy=policy)
    graph = ImpactDependencyGraph(
        repository_tree_id=proposal.repository_tree_id,
        dependencies={
            "pkg/service.py": ("pkg/core.py",),
            "test/api/test_service.py": ("pkg/service.py",),
        },
        validation_targets={VALIDATION_ID: ("test/api/test_service.py",)},
    )
    report = ValidationScheduler(runner=_passing_runner).run_validated(
        accepted,
        (_service_validation(),),
        workspace_path=tmp_path,
        impact_graph=graph,
        validation_policy_id="policy:strict-candidate-isolation",
        dependency_state="fixture",
    )
    dag = ValidationDAGReceipt.from_dict(report["validation_dag_receipt"])
    obligations = derive_fresh_implementation_obligations(
        compile_candidate_proof_scopes((entry,)),
        accepted_plan_id=proposal.accepted_plan_id,
        repository_id=proposal.repository_id,
        repository_tree_id=proposal.repository_tree_id,
        proposal_validation=accepted,
        validation_dag=dag,
        require_validation_dag=True,
        expected_validation_policy_id=dag.policy_id,
    )
    return accepted, dag, obligations


def _g040_completion_packet(tmp_path: Path) -> dict[str, object]:
    accepted, _passing_dag, obligations = _passing_authority_chain(tmp_path)
    proof_witness = prove_proof_candidate_non_authority(
        _proof_receipt(obligations, authoritative=False),
        obligations,
        objective_id=PROOF_CANDIDATE_NON_AUTHORITY_OBJECTIVE_ID,
        proposal_validation=accepted,
        validation_dag=_passing_dag,
    )
    proof_projection = proof_witness.strict_validation_completion_evidence()

    impact_proposal, impact_policy, _entry = _proposal(
        task_id="ASI-032",
        objective_id=TRANSITIVE_IMPACT_OBJECTIVE_ID,
    )
    impact_accepted = validate_proposal(
        impact_proposal,
        policy=impact_policy,
    )
    graph = ImpactDependencyGraph(
        repository_tree_id=impact_proposal.repository_tree_id,
        dependencies={
            "pkg/service.py": ("pkg/core.py",),
            "test/api/test_service.py": ("pkg/service.py",),
        },
        validation_targets={VALIDATION_ID: ("test/api/test_service.py",)},
    )
    report = ValidationScheduler(runner=_runner).run_validated(
        impact_accepted,
        (_service_validation(),),
        workspace_path=tmp_path,
        impact_graph=graph,
        seeded_defect_id="seed:transitive",
        seeded_defect_path="pkg/core.py",
        dependency_state="fixture",
    )
    impact_dag = ValidationDAGReceipt.from_dict(
        report["validation_dag_receipt"]
    )
    validation_projection = (
        impact_dag.strict_validation_completion_evidence()
    )

    now = datetime(2026, 7, 25, 10, 0, tzinfo=timezone.utc)
    repository_id = accepted.proposal.repository_id
    repository_tree = accepted.proposal.repository_tree_id
    validation_command = (
        "python -m pytest "
        "test/api/test_agent_supervisor_proposal_validation.py "
        "test/api/test_agent_supervisor_validation_dag.py "
        "test/api/test_agent_supervisor_semantic_validation_pipeline.py -q"
    )
    completion_evidence = tuple(
        CompletionEvidence(
            acceptance_criterion=criterion,
            producing_task_or_scan="ASI-089",
            validation_receipt={
                "status": "passed",
                "repository_id": repository_id,
                "tree_id": repository_tree,
                "objective_id": STRICT_VALIDATION_OBJECTIVE_ID,
                "operational_receipt_id": validation_projection.evidence_id,
                "validation_policy_id": (
                    validation_projection.validation_policy_id
                ),
                "command": validation_command,
            },
            validation_passed=True,
            repository_id=repository_id,
            repository_tree=repository_tree,
            freshness={"fresh": True},
            observed_at=now - timedelta(minutes=2),
            provenance_cid=f"validation:asi-089:{index}",
            metadata={
                "evidence_source_policy": {
                    "satisfies": True,
                    "source_tier": "validation_receipt",
                }
            },
        )
        for index, criterion in enumerate(
            STRICT_VALIDATION_ACCEPTANCE_CRITERIA,
            start=1,
        )
    )
    coverage = {
        "repository_id": repository_id,
        "repository_tree": repository_tree,
        "evaluated_at": (now - timedelta(minutes=1)).isoformat(),
        "verified": True,
        "criteria": [
            {
                "criterion": criterion,
                "status": "verified",
                "verified": True,
                "implementation": (
                    "ipfs_accelerate_py/agent_supervisor/"
                    + (
                        "proposal_validation.py"
                        if index == 1
                        else "formal_plan_conformance.py"
                    )
                ),
                "validation": validation_command,
                "validation_receipt_ids": [
                    f"validation:asi-089:{index}"
                ],
            }
            for index, criterion in enumerate(
                STRICT_VALIDATION_ACCEPTANCE_CRITERIA,
                start=1,
            )
        ],
    }
    children = [
        {
            "goal_id": goal_id,
            "state": "verified_complete",
            "verified": True,
            "completion_gate": {
                "passed": True,
                "evaluated_evidence": {
                    "repository_id": repository_id,
                    "repository_tree": repository_tree,
                    "evaluated_at": (
                        now - timedelta(minutes=3)
                    ).isoformat(),
                    "validation_evidence": [
                        {
                            "valid": True,
                            "evidence": {
                                "repository_id": repository_id,
                                "repository_tree": repository_tree,
                            },
                        }
                    ],
                },
            },
            "proof_requirements": [
                {
                    "repository_tree": repository_tree,
                    "provenance_id": f"proof:{goal_id}:current",
                    "required_assurance": "solver_checked",
                    "authoritative_assurance": "solver_checked",
                    "assurance_satisfied": True,
                    "contradicted": False,
                    "proof_verdict": "proved",
                    "freshness": "current",
                    "reason_codes": [],
                }
            ],
        }
        for goal_id in STRICT_VALIDATION_CHILD_GOAL_IDS
    ]
    binding = {
        "repository_id": repository_id,
        "tree_id": repository_tree,
        "objective_id": STRICT_VALIDATION_OBJECTIVE_ID,
        "objective_revision": STRICT_VALIDATION_OBJECTIVE_REVISION,
        "analyzer_version": (
            STRICT_VALIDATION_COMPLETION_ANALYZER_VERSION
        ),
        "configuration_revision": (
            STRICT_VALIDATION_COMPLETION_CONFIGURATION_REVISION
        ),
    }
    members = [
        {
            "member_id": "asi-089-implementation-validation",
            "evidence_channel": "implementation-validation",
            "receipt_cid": "scan:asi-089:implementation-validation",
            "binding": dict(binding),
            "scan_mode": "exhaustive",
            "analyzer_version": (
                STRICT_VALIDATION_COMPLETION_ANALYZER_VERSION
                + "/implementation"
            ),
            "passed": True,
            "analyzer_health": {
                "status": "healthy",
                "healthy": True,
            },
            "exhaustive": True,
            "healthy": True,
            "safe_for_completion_reasoning": True,
            "conclusive": True,
            "contradicted": False,
            "finished_at": (now - timedelta(minutes=4)).isoformat(),
        },
        {
            "member_id": "asi-089-independent-replay",
            "evidence_channel": "independent-replay",
            "receipt_cid": "scan:asi-089:independent-replay",
            "binding": dict(binding),
            "scan_mode": "exhaustive",
            "analyzer_version": (
                STRICT_VALIDATION_COMPLETION_ANALYZER_VERSION
                + "/replay"
            ),
            "passed": True,
            "analyzer_health": {
                "status": "healthy",
                "healthy": True,
            },
            "exhaustive": True,
            "healthy": True,
            "safe_for_completion_reasoning": True,
            "conclusive": True,
            "contradicted": False,
            "finished_at": (now - timedelta(minutes=3)).isoformat(),
        },
    ]
    return {
        "repository_id": repository_id,
        "repository_tree": repository_tree,
        "producing_tasks": [
            {"task_id": task_id, "status": "completed"}
            for task_id in STRICT_VALIDATION_PRODUCING_TASK_IDS
        ],
        "child_goals": children,
        "proposal_validation": accepted,
        "validation_projection": validation_projection,
        "proof_projection": proof_projection,
        "evidence": completion_evidence,
        "tasks_complete": True,
        "coverage": coverage,
        "analyzer_health": {
            "status": "healthy",
            "healthy": True,
            "safe_for_completion_reasoning": True,
            "exhaustive": True,
            "binding": dict(binding),
        },
        "exhaustion_quorum": {
            "required_members": (
                STRICT_VALIDATION_REQUIRED_EXHAUSTIVE_RECEIPTS
            ),
            "member_count": len(members),
            "satisfied": True,
            "quorum_met": True,
            "binding": dict(binding),
            "members": members,
        },
        "now": now,
        "freshness_seconds": 3600,
    }


def _assert_g040_rejected(packet: dict[str, object]) -> None:
    decision = evaluate_strict_validation_completion(
        current_state=GoalState.PROVISIONALLY_COMPLETE,
        **packet,
    )
    assert decision.state is not GoalState.VERIFIED_COMPLETE
    assert decision.verified is False
    assert decision.reason_codes


def test_g040_parent_completion_requires_closed_current_tree_proof_packet(
    tmp_path: Path,
) -> None:
    assert STRICT_VALIDATION_OBJECTIVE_ID == "ASI-G040"
    assert STRICT_VALIDATION_PRODUCING_TASK_IDS == (
        "ASI-010",
        "ASI-011",
        "ASI-012",
    )
    assert STRICT_VALIDATION_CHILD_GOAL_IDS == (
        "ASI-G100",
        "ASI-G101",
        "ASI-G102",
    )
    assert STRICT_VALIDATION_REQUIRED_EXHAUSTIVE_RECEIPTS == 2
    assert len(STRICT_VALIDATION_ACCEPTANCE_CRITERIA) == 2
    assert len(STRICT_VALIDATION_GATE_KINDS) == 9

    packet = _g040_completion_packet(tmp_path)
    proof_projection = packet["proof_projection"]
    assert isinstance(
        proof_projection,
        StrictValidationProofCompletionEvidence,
    )
    assert (
        StrictValidationProofCompletionEvidence.from_dict(
            proof_projection.to_dict()
        )
        == proof_projection
    )
    with pytest.raises(ValueError, match="invalid field population"):
        StrictValidationProofCompletionEvidence.from_dict(
            {**proof_projection.to_dict(), "caller_claim": "complete"}
        )

    provisional = evaluate_strict_validation_completion(
        current_state=GoalState.ACTIVE,
        **packet,
    )
    assert provisional.state is GoalState.PROVISIONALLY_COMPLETE
    assert provisional.verified is False
    assert "provisional_transition_required" in provisional.reason_codes
    assert provisional.gate is not None and provisional.gate.passed

    verified = evaluate_strict_validation_completion(
        current_state=GoalState.PROVISIONALLY_COMPLETE,
        **packet,
    )
    assert verified.state is GoalState.VERIFIED_COMPLETE
    assert verified.verified is True
    assert verified.gate is not None and verified.gate.passed

    delegated = packet["validation_projection"].evaluate_parent_completion(
        current_state=GoalState.PROVISIONALLY_COMPLETE,
        **{
            key: value
            for key, value in packet.items()
            if key != "validation_projection"
        },
    )
    assert delegated.state is GoalState.VERIFIED_COMPLETE


@pytest.mark.parametrize(
    "failure",
    (
        "missing_producer",
        "duplicate_producer",
        "active_producer",
        "missing_child",
        "duplicate_child",
        "unverified_child",
        "stale_child",
        "proofless_child",
        "foreign_child_validation",
        "foreign_child_proof",
        "bare_scheduler_projection",
        "bare_proof_projection",
        "missing_evidence",
        "failed_extra_evidence",
        "stale_extra_evidence",
        "unbound_coverage",
        "unsafe_analyzer",
        "under_quorum",
        "duplicate_quorum",
        "stale_quorum",
    ),
)
def test_g040_parent_rejects_open_or_unbound_completion_packet(
    tmp_path: Path,
    failure: str,
) -> None:
    packet = _g040_completion_packet(tmp_path)
    if failure == "missing_producer":
        packet["producing_tasks"] = packet["producing_tasks"][:-1]
    elif failure == "duplicate_producer":
        packet["producing_tasks"][-1]["task_id"] = (
            packet["producing_tasks"][0]["task_id"]
        )
    elif failure == "active_producer":
        packet["producing_tasks"][0]["status"] = "active"
    elif failure == "missing_child":
        packet["child_goals"] = packet["child_goals"][:-1]
    elif failure == "duplicate_child":
        packet["child_goals"][-1]["goal_id"] = (
            packet["child_goals"][0]["goal_id"]
        )
    elif failure == "unverified_child":
        packet["child_goals"][0]["verified"] = False
    elif failure == "stale_child":
        packet["child_goals"][0]["completion_gate"][
            "evaluated_evidence"
        ]["evaluated_at"] = (
            packet["now"] - timedelta(hours=2)
        ).isoformat()
    elif failure == "proofless_child":
        packet["child_goals"][0]["proof_requirements"] = []
    elif failure == "foreign_child_validation":
        packet["child_goals"][0]["completion_gate"][
            "evaluated_evidence"
        ]["validation_evidence"][0]["evidence"][
            "repository_tree"
        ] = "tree:foreign"
    elif failure == "foreign_child_proof":
        packet["child_goals"][0]["proof_requirements"][0][
            "repository_tree"
        ] = "tree:foreign"
    elif failure == "bare_scheduler_projection":
        packet["validation_projection"] = (
            packet["validation_projection"].to_dict()
            | {"evidence_id": "forged"}
        )
    elif failure == "bare_proof_projection":
        packet["proof_projection"] = (
            packet["proof_projection"].to_dict()
            | {"qualifies": False}
        )
    elif failure == "missing_evidence":
        packet["evidence"] = packet["evidence"][:-1]
    elif failure in {"failed_extra_evidence", "stale_extra_evidence"}:
        original = packet["evidence"][0]
        packet["evidence"] = (
            *packet["evidence"],
            replace(
                original,
                validation_passed=failure != "failed_extra_evidence",
                validation_receipt={
                    **original.validation_receipt,
                    "status": (
                        "failed"
                        if failure == "failed_extra_evidence"
                        else "passed"
                    ),
                },
                freshness={
                    "fresh": failure != "stale_extra_evidence"
                },
                observed_at=(
                    packet["now"] - timedelta(hours=2)
                    if failure == "stale_extra_evidence"
                    else original.observed_at
                ),
                provenance_cid=f"validation:asi-089:{failure}",
            ),
        )
    elif failure == "unbound_coverage":
        packet["coverage"]["criteria"][0][
            "validation_receipt_ids"
        ] = ["validation:foreign"]
    elif failure == "unsafe_analyzer":
        packet["analyzer_health"]["safe_for_completion_reasoning"] = False
    elif failure == "under_quorum":
        packet["exhaustion_quorum"]["members"].pop()
        packet["exhaustion_quorum"]["member_count"] = 1
    elif failure == "duplicate_quorum":
        packet["exhaustion_quorum"]["members"][1]["receipt_cid"] = (
            packet["exhaustion_quorum"]["members"][0]["receipt_cid"]
        )
    else:
        packet["exhaustion_quorum"]["members"][0]["finished_at"] = (
            packet["now"] - timedelta(hours=2)
        ).isoformat()

    _assert_g040_rejected(packet)


def test_g040_parent_rejects_caller_lowered_exhaustive_quorum(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match="must equal the configured ASI-G040 count",
    ):
        evaluate_strict_validation_completion(
            required_exhaustive_receipts=1,
            **_g040_completion_packet(tmp_path),
        )


def test_provider_proof_candidate_never_becomes_code_completion_evidence(
    tmp_path: Path,
) -> None:
    accepted, dag, obligations = _passing_authority_chain(tmp_path)
    candidate = _proof_receipt(obligations, authoritative=False)
    binding_result = validate_code_proof_receipt_bindings(
        candidate,
        obligations,
        required_assurance=AssuranceLevel.KERNEL_VERIFIED,
    )

    assert candidate.provider_claimed_assurance is AssuranceLevel.ATTESTED
    assert candidate.authoritative_assurance is AssuranceLevel.CANDIDATE
    assert candidate.authoritative_verdict is ProofVerdict.INCONCLUSIVE
    assert binding_result.valid is False
    assert binding_result.proof_authoritative is False
    assert {
        "code_proof_not_proved",
        "required_code_assurance_not_satisfied",
    }.issubset(binding_result.reason_codes)

    admission = evaluate_completion_admission(
        proposal_validation=accepted,
        validation_dag=dag,
        required=True,
        expected_validation_policy_id=dag.policy_id,
        code_proof_results=(binding_result,),
        require_code_proof=True,
    )
    assert admission.admitted is False
    assert {
        "code_proof_candidate_only",
        "code_proof_not_authoritative",
        "code_proof_binding_rejected",
    }.issubset(admission.reason_codes)

    evidence = prove_proof_candidate_non_authority(
        candidate,
        obligations,
        objective_id=accepted.proposal.objective_id,
        proposal_validation=accepted,
        validation_dag=dag,
    )
    assert evidence.proved_requirement_ids == (
        PROOF_CANDIDATE_NON_AUTHORITY_REQUIREMENT_ID,
    )
    assert evidence.code_proof_authoritative is False
    assert evidence.proof_authoritative is False
    assert evidence.completion_authoritative is False
    assert evidence.completion_admission == admission
    assert (
        ProofCandidateNonAuthorityEvidence.from_dict(evidence.to_dict())
        == evidence
    )
    assert CompletionAdmissionGate.from_dict(admission.to_dict()) == admission


def test_g102_candidate_witness_rejects_legacy_objective_chain(
    tmp_path: Path,
) -> None:
    accepted, dag, obligations = _passing_authority_chain(
        tmp_path,
        task_id="ASI-046",
        objective_id="ASI-G101",
    )

    with pytest.raises(
        ValueError,
        match="must bind the ASI-G102 objective",
    ):
        prove_proof_candidate_non_authority(
            _proof_receipt(obligations, authoritative=False),
            obligations,
            objective_id=accepted.proposal.objective_id,
            proposal_validation=accepted,
            validation_dag=dag,
        )


def test_g102_objective_repair_requires_bound_candidate_isolation_proof(
    tmp_path: Path,
) -> None:
    accepted, dag, obligations = _passing_authority_chain(tmp_path)
    witness = prove_proof_candidate_non_authority(
        _proof_receipt(obligations, authoritative=False),
        obligations,
        objective_id=PROOF_CANDIDATE_NON_AUTHORITY_OBJECTIVE_ID,
        proposal_validation=accepted,
        validation_dag=dag,
    )
    now = datetime(2026, 7, 24, 16, 0, tzinfo=timezone.utc)
    validation_binding = {
        "status": "passed",
        "repository_id": witness.obligation_set.binding.repository_id,
        "tree_id": witness.obligation_set.binding.repository_tree_id,
        "requirement_id": PROOF_CANDIDATE_NON_AUTHORITY_REQUIREMENT_ID,
        "objective_id": PROOF_CANDIDATE_NON_AUTHORITY_OBJECTIVE_ID,
        "operational_receipt_id": witness.evidence_id,
        "validation_policy_id": witness.validation_dag.policy_id,
        "command": (
            "python -m pytest "
            "test/api/test_agent_supervisor_proposal_validation.py "
            "test/api/test_agent_supervisor_validation_dag.py "
            "test/api/test_agent_supervisor_semantic_validation_pipeline.py -q"
        ),
    }
    completion_evidence = tuple(
        CompletionEvidence(
            acceptance_criterion=criterion,
            producing_task_or_scan="ASI-070",
            validation_receipt=validation_binding,
            validation_passed=True,
            repository_tree=witness.obligation_set.binding.repository_tree_id,
            freshness={"fresh": True},
            observed_at=now,
            provenance_cid=f"validation:asi-070:{index}",
            metadata={
                "evidence_source_policy": {
                    "satisfies": True,
                    "source_tier": "validation_receipt",
                }
            },
        )
        for index, criterion in enumerate(
            PROOF_CANDIDATE_NON_AUTHORITY_ACCEPTANCE_CRITERIA,
            start=1,
        )
    )
    coverage = {
        "repository_tree": witness.obligation_set.binding.repository_tree_id,
        "evaluated_at": now.isoformat(),
        "verified": True,
        "criteria": [
            {
                "criterion": criterion,
                "status": "verified",
                "verified": True,
                "implementation": (
                    "ipfs_accelerate_py/agent_supervisor/"
                    "code_proof_obligations.py"
                ),
                "validation": (
                    "test/api/"
                    "test_agent_supervisor_semantic_validation_pipeline.py"
                ),
            }
            for criterion in PROOF_CANDIDATE_NON_AUTHORITY_ACCEPTANCE_CRITERIA
        ],
    }
    health = {
        "status": "healthy",
        "healthy": True,
        "safe_for_completion_reasoning": True,
        "exhaustive": True,
        "analyzer_version": (
            PROOF_CANDIDATE_NON_AUTHORITY_COMPLETION_ANALYZER_VERSION
        ),
    }
    binding = {
        "repository_id": witness.obligation_set.binding.repository_id,
        "tree_id": witness.obligation_set.binding.repository_tree_id,
        "objective_id": PROOF_CANDIDATE_NON_AUTHORITY_OBJECTIVE_ID,
        "objective_revision": (
            PROOF_CANDIDATE_NON_AUTHORITY_OBJECTIVE_REVISION
        ),
        "validation_policy_id": witness.validation_dag.policy_id,
        "operational_receipt_id": witness.evidence_id,
        "analyzer_version": (
            PROOF_CANDIDATE_NON_AUTHORITY_COMPLETION_ANALYZER_VERSION
        ),
        "configuration_revision": (
            PROOF_CANDIDATE_NON_AUTHORITY_COMPLETION_CONFIGURATION_REVISION
        ),
    }
    quorum = {
        "required_members": 2,
        "member_count": 2,
        "satisfied": True,
        "quorum_met": True,
        "binding": binding,
        "members": [
            {
                "member_id": "asi-070-implementation",
                "evidence_channel": "implementation-validation",
                "receipt_cid": "scan:asi-070:implementation",
                "binding": binding,
                "scan_mode": "exhaustive",
                "analyzer_version": (
                    PROOF_CANDIDATE_NON_AUTHORITY_COMPLETION_ANALYZER_VERSION
                    + "/implementation"
                ),
                "passed": True,
                "analyzer_health": {
                    "status": "healthy",
                    "healthy": True,
                },
                "exhaustive": True,
                "healthy": True,
                "safe_for_completion_reasoning": True,
                "conclusive": True,
                "contradicted": False,
                "finished_at": now.isoformat(),
            },
            {
                "member_id": "asi-070-replay",
                "evidence_channel": "receipt-replay",
                "receipt_cid": "scan:asi-070:replay",
                "binding": binding,
                "scan_mode": "exhaustive",
                "analyzer_version": (
                    PROOF_CANDIDATE_NON_AUTHORITY_COMPLETION_ANALYZER_VERSION
                    + "/replay"
                ),
                "passed": True,
                "analyzer_health": {
                    "status": "healthy",
                    "healthy": True,
                },
                "exhaustive": True,
                "healthy": True,
                "safe_for_completion_reasoning": True,
                "conclusive": True,
                "contradicted": False,
                "finished_at": now.isoformat(),
            },
        ],
    }
    values = {
        "evidence": completion_evidence,
        "tasks_complete": True,
        "coverage": coverage,
        "analyzer_health": health,
        "exhaustion_quorum": quorum,
        "now": now,
        "freshness_seconds": 300,
    }

    provisional = witness.evaluate_objective_completion(
        current_state=GoalState.ACTIVE,
        **values,
    )
    assert provisional.state is GoalState.PROVISIONALLY_COMPLETE
    assert provisional.gate is not None and provisional.gate.passed
    assert not provisional.verified

    verified = witness.evaluate_objective_completion(
        current_state=GoalState.PROVISIONALLY_COMPLETE,
        **values,
    )
    assert verified.state is GoalState.VERIFIED_COMPLETE
    assert verified.verified

    for receipt_updates in (
        {"operational_receipt_id": "evidence:foreign-candidate"},
        {"repository_id": "repo:foreign"},
        {"tree_id": "tree:foreign"},
    ):
        unbound = tuple(
            CompletionEvidence.from_dict(
                {
                    **item.to_dict(),
                    "validation_receipt": {
                        **validation_binding,
                        **receipt_updates,
                    },
                }
            )
            for item in completion_evidence
        )
        rejected = witness.evaluate_objective_completion(
            current_state=GoalState.PROVISIONALLY_COMPLETE,
            **{**values, "evidence": unbound},
        )
        assert rejected.state is GoalState.PROVISIONALLY_COMPLETE
        assert not rejected.verified
        assert rejected.gate is not None and not rejected.gate.passed

    unsafe = witness.evaluate_objective_completion(
        current_state=GoalState.PROVISIONALLY_COMPLETE,
        **{
            **values,
            "analyzer_health": {
                **health,
                "safe_for_completion_reasoning": False,
            },
        },
    )
    assert unsafe.state is GoalState.PROVISIONALLY_COMPLETE
    assert "analyzer_unhealthy" in unsafe.reason_codes

    duplicate_quorum = deepcopy(quorum)
    duplicate_quorum["members"][1]["receipt_cid"] = (
        duplicate_quorum["members"][0]["receipt_cid"]
    )
    no_quorum = witness.evaluate_objective_completion(
        current_state=GoalState.PROVISIONALLY_COMPLETE,
        **{**values, "exhaustion_quorum": duplicate_quorum},
    )
    assert no_quorum.state is GoalState.PROVISIONALLY_COMPLETE
    assert any(
        code.startswith("exhaustion_quorum")
        for code in no_quorum.reason_codes
    )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload.__setitem__("objective_id", "ASI-G999"),
        lambda payload: payload.__setitem__("code_proof_authoritative", True),
        lambda payload: payload.__setitem__("proof_authoritative", True),
        lambda payload: payload["binding_result"].__setitem__("valid", True),
        lambda payload: payload["completion_admission"].__setitem__(
            "admitted", True
        ),
        lambda payload: payload["candidate_receipt"].__setitem__(
            "repository_tree_id", "tree:replayed"
        ),
    ],
)
def test_candidate_non_authority_evidence_rejects_tamper_and_replay(
    tmp_path: Path,
    mutate,
) -> None:
    accepted, dag, obligations = _passing_authority_chain(tmp_path)
    evidence = prove_proof_candidate_non_authority(
        _proof_receipt(obligations, authoritative=False),
        obligations,
        objective_id=accepted.proposal.objective_id,
        proposal_validation=accepted,
        validation_dag=dag,
    )
    payload = deepcopy(evidence.to_dict())
    mutate(payload)

    with pytest.raises(ValueError):
        ProofCandidateNonAuthorityEvidence.from_dict(payload)


def test_independent_kernel_receipt_is_not_candidate_rejection_evidence(
    tmp_path: Path,
) -> None:
    accepted, dag, obligations = _passing_authority_chain(tmp_path)
    receipts = tuple(
        _proof_receipt(
            obligations,
            authoritative=True,
            obligation_index=index,
        )
        for index in range(len(obligations.obligations))
    )
    receipt = receipts[0]
    result = validate_code_proof_receipt_bindings(receipt, obligations)

    assert result.valid is True
    assert result.authoritative_assurance is AssuranceLevel.KERNEL_VERIFIED
    assert result.authoritative_verdict is ProofVerdict.PROVED
    no_dag = evaluate_completion_admission(
        proposal_validation=accepted,
        required=True,
        code_proof_receipts=receipts,
        implementation_obligations=obligations,
        require_code_proof=True,
    )
    assert no_dag.admitted is False
    assert "validation_dag_missing" in no_dag.reason_codes
    detached = evaluate_completion_admission(
        proposal_validation=accepted,
        validation_dag=dag,
        required=True,
        expected_validation_policy_id=dag.policy_id,
        code_proof_results=(result,),
        require_code_proof=True,
    )
    assert detached.admitted is False
    assert {
        "code_proof_unverified_summary",
        "code_proof_missing",
    }.issubset(detached.reason_codes)
    admission = evaluate_completion_admission(
        proposal_validation=accepted,
        validation_dag=dag,
        required=True,
        expected_validation_policy_id=dag.policy_id,
        code_proof_receipts=receipts,
        implementation_obligations=obligations,
        require_code_proof=True,
    )
    assert admission.admitted is True
    with pytest.raises(ValueError, match="not a proof candidate"):
        prove_proof_candidate_non_authority(
            receipt,
            obligations,
            objective_id=accepted.proposal.objective_id,
            proposal_validation=accepted,
            validation_dag=dag,
        )


def test_accepted_proposal_rejects_same_path_with_different_changed_ast() -> None:
    proposal, policy, _entry = _proposal()
    accepted = validate_proposal(proposal, policy=policy)
    substituted = CandidateDiffEntry(
        old_path="pkg/core.py",
        new_path="pkg/core.py",
        change_kind=DiffChangeKind.MODIFY,
        before_source=BEFORE,
        after_source=AFTER.replace("+ 2", "+ 9000"),
    )

    with pytest.raises(
        ValueError,
        match="AST/interface/effect scopes do not match",
    ):
        derive_fresh_implementation_obligations(
            compile_candidate_proof_scopes((substituted,)),
            accepted_plan_id=proposal.accepted_plan_id,
            repository_id=proposal.repository_id,
            repository_tree_id=proposal.repository_tree_id,
            proposal_validation=accepted,
        )


def test_wrong_theorem_and_post_merge_receipts_are_not_reusable() -> None:
    proposal, policy, entry = _proposal()
    accepted = validate_proposal(proposal, policy=policy)
    scopes = compile_candidate_proof_scopes((entry,))
    obligations = derive_fresh_implementation_obligations(
        scopes,
        accepted_plan_id=proposal.accepted_plan_id,
        repository_id=proposal.repository_id,
        repository_tree_id=proposal.repository_tree_id,
        proposal_validation=accepted,
        goal_id=proposal.objective_id,
        code_proof_toolchain_id="toolchain:locked",
        code_proof_policy_id="policy:strict-code-proof",
    )
    receipt = _proof_receipt(obligations, authoritative=True)

    wrong_theorem = validate_code_proof_receipt_bindings(
        replace(receipt, obligation_id="obligation:foreign-theorem"),
        obligations,
    )
    assert wrong_theorem.valid is False
    assert "wrong_theorem_not_in_fresh_obligation_set" in (
        wrong_theorem.reason_codes
    )

    merged_obligations = derive_fresh_implementation_obligations(
        scopes,
        accepted_plan_id=proposal.accepted_plan_id,
        repository_id=proposal.repository_id,
        repository_tree_id="tree:post-merge",
        goal_id=proposal.objective_id,
        code_proof_toolchain_id="toolchain:locked",
        code_proof_policy_id="policy:strict-code-proof",
    )
    stale = validate_code_proof_receipt_bindings(
        receipt,
        merged_obligations,
    )
    assert stale.valid is False
    assert {
        "receipt_repository_tree_id_mismatch",
        "receipt_implementation_binding_id_mismatch",
    }.intersection(stale.reason_codes)


def test_omitted_planned_effect_fails_closed() -> None:
    proposal, policy, entry = _proposal()
    accepted = validate_proposal(proposal, policy=policy)
    obligations = derive_fresh_implementation_obligations(
        compile_candidate_proof_scopes((entry,)),
        accepted_plan_id=proposal.accepted_plan_id,
        repository_id=proposal.repository_id,
        repository_tree_id=proposal.repository_tree_id,
        proposal_validation=accepted,
        planned_effect_ids=("effect:persist-reviewed-state",),
    )

    assert obligations.complete is False
    assert (
        "planned_effect_scope_omitted"
        in obligations.incomplete_reason_codes
    )
