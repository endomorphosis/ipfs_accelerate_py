"""Focused DCR-063 append-only replan memory tests."""

from __future__ import annotations

from dataclasses import replace

import pytest

from ipfs_accelerate_py.agent_supervisor.autonomous_repair.contracts import RepairAuthorityRoots
from ipfs_accelerate_py.agent_supervisor.planning.deterministic_candidate_portfolio import (
    CandidatePortfolio,
    CandidatePortfolioDisposition,
)
from ipfs_accelerate_py.agent_supervisor.planning.deterministic_failure_memory import (
    FailureAttempt,
    FailureClass,
    FailureMemoryReceipt,
    ReplanMemoryDisposition,
    decide_replan,
)
from ipfs_accelerate_py.agent_supervisor.planning.proof_carrying_repair_dag import (
    RepairPlanDagResult,
    RepairPlanDagDisposition,
)


def _inputs():
    roots = RepairAuthorityRoots("repo", "forest", "tree", "policy", "rpr-plan", "rpr-packet")
    portfolio = CandidatePortfolio(
        CandidatePortfolioDisposition.INTEGRATION_PENDING, (), "portfolio", ("candidate",)
    )
    plan = RepairPlanDagResult(RepairPlanDagDisposition.INTEGRATION_PENDING, (), "plan")
    return portfolio, plan, roots


def _attempt(
    roots, *, evidence=("evidence-a",), measure=(3,), previous="", failure=FailureClass.VALIDATION
):
    return FailureAttempt(
        "portfolio", "candidate", "plan", roots.content_id, failure, evidence, measure, previous
    )


def test_duplicate_replay_emits_no_work_and_restart_is_identical() -> None:
    portfolio, plan, roots = _inputs()
    first = _attempt(roots)
    receipt = FailureMemoryReceipt(first, first.content_id)
    decision = decide_replan(portfolio, plan, roots, _attempt(roots), history=(receipt,))
    reconstructed = decide_replan(portfolio, plan, roots, _attempt(roots), history=(receipt,))
    assert decision.disposition is ReplanMemoryDisposition.NO_WORK
    assert decision == reconstructed
    assert decision.execution_authorized is False


def test_retry_requires_new_evidence_and_decreasing_measure() -> None:
    portfolio, plan, roots = _inputs()
    first = _attempt(roots)
    receipt = FailureMemoryReceipt(first, first.content_id)
    retry = _attempt(
        roots, evidence=("evidence-a", "evidence-b"), measure=(2,), previous=receipt.receipt_cid
    )
    assert (
        decide_replan(portfolio, plan, roots, retry, history=(receipt,)).disposition
        is ReplanMemoryDisposition.RETRY_PENDING
    )
    nondecreasing = replace(retry, measure=(3,))
    assert (
        decide_replan(portfolio, plan, roots, nondecreasing, history=(receipt,)).disposition
        is ReplanMemoryDisposition.ABSTAINED
    )


def test_stale_roots_forged_history_and_refuted_candidate_reject_or_stop() -> None:
    portfolio, plan, roots = _inputs()
    proof = _attempt(roots, failure=FailureClass.PROOF)
    receipt = FailureMemoryReceipt(proof, proof.content_id)
    retry = _attempt(
        roots, evidence=("evidence-a", "evidence-b"), measure=(1,), previous=receipt.receipt_cid
    )
    assert decide_replan(portfolio, plan, roots, retry, history=(receipt,)).reason_codes == (
        "refuted_candidate_never_replayed",
    )
    stale = replace(retry, root_cid="other-root")
    assert (
        decide_replan(portfolio, plan, roots, stale).disposition is ReplanMemoryDisposition.REJECTED
    )
    with pytest.raises(ValueError):
        FailureMemoryReceipt(receipt.attempt, "forged")


def test_first_proof_refutation_stops_without_retry() -> None:
    portfolio, plan, roots = _inputs()
    decision = decide_replan(portfolio, plan, roots, _attempt(roots, failure=FailureClass.PROOF))
    assert decision.disposition is ReplanMemoryDisposition.NO_WORK
