"""DCR-082 pure replay recovery tests; no actual recovery action is possible."""

from __future__ import annotations

from dataclasses import replace

from ipfs_accelerate_py.agent_supervisor.autonomous_repair.contracts import (
    RepairAdmissionReceipt,
    RepairAuthorityRoots,
)
from ipfs_accelerate_py.agent_supervisor.autonomous_repair.transaction import (
    FencedWriteReceipt,
    TransactionDisposition,
    TransactionJournal,
    TransactionState,
)
from ipfs_accelerate_py.agent_supervisor.planning.deterministic_candidate_portfolio import (
    CandidatePortfolio,
    CandidatePortfolioDisposition,
)
from ipfs_accelerate_py.agent_supervisor.planning.deterministic_failure_memory import (
    FailureAttempt,
    FailureClass,
    FailureMemoryReceipt,
    decide_replan,
)
from ipfs_accelerate_py.agent_supervisor.planning.proof_carrying_repair_dag import (
    RepairPlanDagDisposition,
    RepairPlanDagResult,
)
from ipfs_accelerate_py.agent_supervisor.planning.repair_resource_scheduler import (
    RepairResourceSchedule,
    ScheduledRepairNode,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.deterministic_repair_composition import (
    DeterministicRepairCompositionDisposition,
    DeterministicRepairCompositionResult,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.deterministic_repair_recovery import (
    RecoveryDisposition,
    RecoveryOutcome,
    RecoveryRequest,
    canonical_recovery_decision_bytes,
    replay_recovery,
)


def _request() -> RecoveryRequest:
    roots = RepairAuthorityRoots(
        "repo", "cid:forest", "tree", "cid:policy", "cid:plan", "cid:packet"
    )
    admission = RepairAdmissionReceipt("task", roots, "cid:previous", "cid:derivation")
    prior_attempt = FailureAttempt(
        "cid:portfolio",
        "cid:candidate",
        "cid:plan",
        roots.content_id,
        FailureClass.RESOURCE,
        ("cid:evidence-a",),
        (2,),
    )
    new_attempt = FailureAttempt(
        "cid:portfolio",
        "cid:candidate",
        "cid:plan",
        roots.content_id,
        FailureClass.RESOURCE,
        ("cid:evidence-a", "cid:evidence-b"),
        (1,),
        prior_attempt.content_id,
    )
    prior = FailureMemoryReceipt(prior_attempt, prior_attempt.content_id)
    new = FailureMemoryReceipt(new_attempt, new_attempt.content_id)
    journal = TransactionJournal(
        "txn",
        TransactionState.ROLLED_BACK,
        TransactionDisposition.ROLLED_BACK,
        "crash rollback",
        "/tmp/isolate",
        "sha256:" + "a" * 64,
        admission.content_id,
        "cid:preview",
        "lease",
        "fence",
        (
            FencedWriteReceipt(
                "x.py", "sha256:" + "b" * 64, "sha256:" + "c" * 64, "sha256:" + "b" * 64, "fence"
            ),
        ),
        True,
    )
    portfolio = CandidatePortfolio(
        CandidatePortfolioDisposition.INTEGRATION_PENDING, (), "cid:portfolio", ("cid:candidate",)
    )
    plan_result = RepairPlanDagResult(RepairPlanDagDisposition.INTEGRATION_PENDING, (), "cid:plan")
    replan = decide_replan(portfolio, plan_result, roots, new_attempt, history=(prior,))
    prior_lease = ScheduledRepairNode("node", "lane", 0, "cid:lease-old", "cid:fence-old", ())
    reacquired_lease = ScheduledRepairNode("node", "lane", 1, "cid:lease-new", "cid:fence-new", ())
    return RecoveryRequest(
        "task",
        RecoveryOutcome.CRASH,
        prior,
        new,
        replan,
        portfolio,
        plan_result,
        roots,
        (prior,),
        RepairResourceSchedule("integration_pending", (), "cid:schedule-old", (prior_lease,)),
        RepairResourceSchedule("integration_pending", (), "cid:schedule-new", (reacquired_lease,)),
        prior_lease,
        reacquired_lease,
        admission,
        journal,
        DeterministicRepairCompositionResult(
            DeterministicRepairCompositionDisposition.DEFER_CAPABILITY,
            ("pending",),
            ("dcr050",),
            "task",
        ),
    )


def test_crash_restart_replay_is_idempotent_pending_with_zero_effects() -> None:
    request = _request()
    first, second = replay_recovery(request), replay_recovery(request)
    assert first.disposition is second.disposition is RecoveryDisposition.INTEGRATION_PENDING
    assert first.replay_cid == second.replay_cid
    assert first.to_dict()["write_call_count"] == first.to_dict()["provider_call_count"] == 0
    assert canonical_recovery_decision_bytes(first) == canonical_recovery_decision_bytes(first)


def test_fence_partial_write_no_new_evidence_and_validation_pending_abstain() -> None:
    request = _request()
    assert (
        replay_recovery(replace(request, reacquired_lease=request.prior_lease)).disposition
        is RecoveryDisposition.ABSTAINED
    )
    assert (
        replay_recovery(replace(request, new_failure=request.prior_failure)).disposition
        is RecoveryDisposition.ABSTAINED
    )
    assert (
        replay_recovery(
            replace(request, journal=replace(request.journal, rollback_verified=False))
        ).disposition
        is RecoveryDisposition.ABSTAINED
    )
    assert (
        replay_recovery(
            replace(
                request, journal=replace(request.journal, state=TransactionState.VALIDATION_PENDING)
            )
        ).disposition
        is RecoveryDisposition.ABSTAINED
    )


def test_cancel_and_forged_receipt_never_recover_successfully() -> None:
    request = _request()
    assert (
        replay_recovery(replace(request, outcome=RecoveryOutcome.CANCEL)).disposition
        is RecoveryDisposition.ABSTAINED
    )
    assert (
        replay_recovery(replace(request, prior_failure=object())).disposition
        is RecoveryDisposition.ABSTAINED
    )
    assert (
        replay_recovery(
            replace(
                request,
                dcr080=replace(request.dcr080, task_id="other-task"),
            )
        ).disposition
        is RecoveryDisposition.ABSTAINED
    )
