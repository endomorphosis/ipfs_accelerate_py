from __future__ import annotations

from copy import deepcopy

import pytest

from ipfs_accelerate_py.agent_supervisor.code_proof_obligations import (
    CandidateDiffEntry,
    DiffChangeKind,
)
from ipfs_accelerate_py.agent_supervisor.proposal_validation import (
    ImplementationProposal,
    NOOP_OR_OUT_OF_SCOPE_FAIL_FAST_REQUIREMENT_ID,
    ORDERED_PROPOSAL_GATES,
    ProposalFindingCode,
    ProposalGate,
    ProposalValidationError,
    ProposalValidationPolicy,
    ProposalValidationReceipt,
    ProposalValidationResult,
    validate_implementation_proposal,
)


TASK_ID = "ASI-031"
PLAN_ID = "plan:strict-validation"
REPOSITORY_ID = "repository:ipfs-accelerate"
TREE_ID = "tree:strict-validation"
OBJECTIVE_ID = "ASI-G100"


def _policy(**overrides: object) -> ProposalValidationPolicy:
    values: dict[str, object] = {
        "allowed_paths": (
            "ipfs_accelerate_py/agent_supervisor/",
            "test/api/",
        ),
        "expected_task_id": TASK_ID,
        "expected_plan_id": PLAN_ID,
        "expected_repository_id": REPOSITORY_ID,
        "expected_repository_tree_id": TREE_ID,
        "expected_objective_id": OBJECTIVE_ID,
    }
    values.update(overrides)
    return ProposalValidationPolicy(**values)


def _entry(
    path: str = "ipfs_accelerate_py/agent_supervisor/proposal_validation.py",
    *,
    before: str | None = "VALUE = 1\n",
    after: str | None = "VALUE = 2\n",
    **overrides: object,
) -> CandidateDiffEntry:
    values: dict[str, object] = {
        "old_path": path,
        "new_path": path,
        "change_kind": DiffChangeKind.MODIFY,
        "before_source": before,
        "after_source": after,
    }
    values.update(overrides)
    return CandidateDiffEntry(**values)


def _proposal(
    *entries: CandidateDiffEntry,
    declared_paths: tuple[str, ...] | None = None,
    **overrides: object,
) -> ImplementationProposal:
    candidate_diff = entries or (_entry(),)
    paths = (
        tuple(
            sorted(
                {
                    path
                    for entry in candidate_diff
                    for path in (entry.old_path, entry.new_path)
                    if path
                }
            )
        )
        if declared_paths is None
        else declared_paths
    )
    values: dict[str, object] = {
        "task_id": TASK_ID,
        "accepted_plan_id": PLAN_ID,
        "repository_id": REPOSITORY_ID,
        "repository_tree_id": TREE_ID,
        "objective_id": OBJECTIVE_ID,
        "baseline_id": "baseline:strict-validation",
        "candidate_diff": candidate_diff,
        "declared_paths": paths,
        "context_id": "context:strict-validation",
    }
    values.update(overrides)
    return ImplementationProposal(**values)


def test_accepts_an_exactly_bound_effectful_proposal() -> None:
    policy = _policy()
    proposal = _proposal()

    result = validate_implementation_proposal(proposal, policy=policy)

    assert result.accepted
    assert result.findings == ()
    assert result.receipt.gate_trace == ORDERED_PROPOSAL_GATES
    assert result.receipt.proposal_id == proposal.proposal_id
    assert result.receipt.policy_id == policy.policy_id
    assert result.receipt.repository_tree_id == TREE_ID
    assert result.receipt.objective_id == OBJECTIVE_ID
    assert result.receipt.changed_paths == proposal.changed_paths
    assert result.receipt.diff_digest == proposal.diff_digest
    assert result.proof_authoritative is False
    assert result.completion_authoritative is False
    assert result.receipt.proved_requirement_ids == ()
    assert ProposalValidationResult.from_dict(result.to_dict()) == result


@pytest.mark.parametrize(
    ("proposal", "expected_code", "expected_gate"),
    [
        (
            _proposal(declared_paths=(), candidate_diff=()),
            ProposalFindingCode.EMPTY_PATCH,
            ProposalGate.PATCH,
        ),
        (
            _proposal(
                _entry("docs/outside.md", before="before\n", after="after\n")
            ),
            ProposalFindingCode.PATH_OUTSIDE_SCOPE,
            ProposalGate.PATH,
        ),
    ],
)
def test_noop_and_out_of_scope_rejections_are_typed_fail_fast_evidence(
    proposal: ImplementationProposal,
    expected_code: ProposalFindingCode,
    expected_gate: ProposalGate,
) -> None:
    result = validate_implementation_proposal(proposal, policy=_policy())

    assert not result.accepted
    assert any(
        finding.code is expected_code and finding.gate is expected_gate
        for finding in result.findings
    )
    assert result.receipt.expensive_checks_started == 0
    assert result.receipt.rejection_evidence is None

    dispatched = result.with_dispatch_outcome(
        expensive_node_ids=("semantic", "proof", "targeted-tests"),
        expensive_checks_started=0,
    )
    evidence = dispatched.receipt.rejection_evidence
    assert evidence is not None
    assert evidence.requirement_id == NOOP_OR_OUT_OF_SCOPE_FAIL_FAST_REQUIREMENT_ID
    assert evidence.proved_requirement_ids == (
        NOOP_OR_OUT_OF_SCOPE_FAIL_FAST_REQUIREMENT_ID,
    )
    assert expected_code.value in evidence.rejection_codes
    assert evidence.expensive_checks_started == 0
    assert evidence.expensive_node_ids == (
        "proof",
        "semantic",
        "targeted-tests",
    )
    assert ProposalValidationResult.from_dict(dispatched.to_dict()) == dispatched


def test_findings_are_deterministic_and_bounded_by_policy() -> None:
    proposal = _proposal(
        *(
            _entry(
                f"outside/invalid_{index}.py",
                after="def broken(:\n",
                binary=True,
                generated=True,
            )
            for index in range(8)
        )
    )
    policy = _policy(max_findings=3)

    first = validate_implementation_proposal(proposal, policy=policy)
    second = validate_implementation_proposal(
        ImplementationProposal.from_dict(proposal.to_dict()),
        policy=ProposalValidationPolicy.from_dict(policy.to_dict()),
    )

    assert not first.accepted
    assert len(first.findings) == policy.max_findings
    assert first.receipt.receipt_id == second.receipt.receipt_id
    assert first.receipt.to_dict() == second.receipt.to_dict()
    assert tuple(
        (
            ORDERED_PROPOSAL_GATES.index(finding.gate),
            finding.path,
            finding.code.value,
            finding.message,
        )
        for finding in first.findings
    ) == tuple(
        sorted(
            (
                ORDERED_PROPOSAL_GATES.index(finding.gate),
                finding.path,
                finding.code.value,
                finding.message,
            )
            for finding in first.findings
        )
    )


def test_syntax_and_every_frozen_authority_dimension_fail_closed() -> None:
    proposal = _proposal(
        _entry(after="def invalid(:\n"),
        task_id="ASI-OTHER",
        accepted_plan_id="plan:other",
        repository_id="repository:other",
        repository_tree_id="tree:stale",
        objective_id="ASI-G999",
    )

    result = validate_implementation_proposal(proposal, policy=_policy())
    codes = [finding.code for finding in result.findings]

    assert not result.accepted
    assert codes.count(ProposalFindingCode.AUTHORITY_MISMATCH) == 4
    assert codes.count(ProposalFindingCode.STALE_BASELINE) == 1
    syntax = [
        finding
        for finding in result.findings
        if finding.code is ProposalFindingCode.PYTHON_SYNTAX_ERROR
    ]
    assert len(syntax) == 1
    assert syntax[0].gate is ProposalGate.AST_INTERFACE
    assert syntax[0].path.endswith("proposal_validation.py")
    assert result.receipt.gate_trace == ORDERED_PROPOSAL_GATES
    assert result.receipt.rejection_evidence is None


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload["proposal"].__setitem__(
            "diff_digest", "sha256:forged"
        ),
        lambda payload: payload["policy"].__setitem__(
            "allowed_paths", ["forged/"]
        ),
        lambda payload: payload["receipt"].__setitem__(
            "proof_authoritative", True
        ),
        lambda payload: payload["receipt"].__setitem__(
            "changed_paths", ["forged.py"]
        ),
        lambda payload: payload.__setitem__("accepted", False),
    ],
)
def test_serialized_result_rejects_tampered_identity_authority_and_verdict(
    mutate,
) -> None:
    payload = deepcopy(
        validate_implementation_proposal(_proposal(), policy=_policy()).to_dict()
    )
    mutate(payload)

    with pytest.raises(ProposalValidationError):
        ProposalValidationResult.from_dict(payload)


def test_rejection_receipt_rejects_detached_or_mutated_evidence() -> None:
    rejected = validate_implementation_proposal(
        _proposal(declared_paths=(), candidate_diff=()),
        policy=_policy(),
    ).with_dispatch_outcome(
        expensive_node_ids=("semantic", "proof"),
        expensive_checks_started=0,
    )
    payload = deepcopy(rejected.receipt.to_dict())
    assert payload["rejection_evidence"] is not None
    payload["rejection_evidence"]["expensive_checks_started"] = 1

    with pytest.raises(ProposalValidationError):
        ProposalValidationReceipt.from_dict(payload)
