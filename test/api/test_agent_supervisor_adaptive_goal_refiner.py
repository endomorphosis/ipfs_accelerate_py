from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import datetime, timezone

import pytest

from ipfs_accelerate_py.agent_supervisor.adaptive_goal_refiner import (
    ADAPTIVE_GOAL_REFINER_VERSION,
    NEW_EVIDENCE_REFINEMENT_REQUIREMENT_ID,
    NEW_COUNTEREXAMPLE_REFINEMENT_ACCEPTANCE_CRITERIA,
    UNCHANGED_FAILURE_BACKOFF_REQUIREMENT_ID,
    AdaptiveGoalRefinementError,
    AdaptiveGoalRefiner,
    AdaptiveRefinementCandidate,
    AdaptiveRefinementPolicy,
    AdaptiveRefinementReceipt,
    AdaptiveRefinementRequest,
    GoalDebtKind,
    GoalQualityRecord,
    InMemoryRefinementStore,
    JsonlRefinementStore,
    RefinementDecision,
    RefinementProducerKind,
    RefinementSignal,
    RefinementSignalKind,
)
from ipfs_accelerate_py.agent_supervisor.formal_planning_contracts import (
    Actor,
    ActorKind,
    FormalWorkPlan,
    Goal,
    PlanTask,
)
from ipfs_accelerate_py.agent_supervisor.goal_refinement_verification import (
    FrozenRefinementContext,
)
from ipfs_accelerate_py.agent_supervisor.goal_completion import (
    CompletionEvidence,
    GoalState,
)


def _plan(*, with_child: bool = False, root_outcome: str = "formula:root") -> FormalWorkPlan:
    goals = [
        Goal(
            goal_id="goal:root",
            owner_actor_id="actor:supervisor",
            satisfaction_formula_id=root_outcome,
            source_ids=("objective:root",),
        )
    ]
    if with_child:
        goals.append(
            Goal(
                goal_id="goal:child",
                owner_actor_id="actor:supervisor",
                satisfaction_formula_id="formula:child",
                source_ids=("evidence:counterexample",),
            )
        )
    return FormalWorkPlan(
        vocabulary_profile_id="reviewed-test",
        vocabulary_version=1,
        actors=(Actor("actor:supervisor", ActorKind.SUPERVISOR),),
        goals=tuple(goals),
        subgoals=(),
        tasks=(PlanTask("task:root", "goal:root", actor_ids=("actor:supervisor",)),),
        events=(),
        fluents=(),
        preconditions=(),
        effects=(),
        norms=(),
        temporal_constraints=(),
        evidence_requirements=(),
        source_ids=("objective:root",),
        repository_tree_id="tree:one",
    )


def _signal(
    revision: str = "counterexample:v1",
    *,
    kind: RefinementSignalKind = RefinementSignalKind.COUNTEREXAMPLE,
    observed_at: int = 100,
) -> RefinementSignal:
    return RefinementSignal(
        kind=kind,
        subject_id="goal:root",
        evidence_revision=revision,
        observed_at=observed_at,
        failure_signature=(
            "pytest::test_contract/assertion"
            if kind is RefinementSignalKind.REPEATED_FAILURE
            else ""
        ),
        details={"check_id": "pytest:contract", "result": "failed"},
    )


def _request(
    signal: RefinementSignal | None = None,
    *,
    plan: FormalWorkPlan | None = None,
    depth: int = 0,
) -> AdaptiveRefinementRequest:
    plan = plan or _plan()
    root = next(item for item in plan.goals if item.goal_id == "goal:root")
    return AdaptiveRefinementRequest(
        plan=plan,
        root_goal_id=root.goal_id,
        root_goal_content_id=root.content_id,
        assumption_ids=("assumption:frozen",),
        signals=(signal or _signal(),),
        cycle_id="cycle:1",
        refinement_depth=depth,
    )


def _candidate(request: AdaptiveRefinementRequest) -> AdaptiveRefinementCandidate:
    return AdaptiveRefinementCandidate(
        plan=_plan(with_child=True),
        root_goal_id=request.root_goal_id,
        root_goal_content_id=request.root_goal_content_id,
        assumption_ids=request.assumption_ids,
        changed_goal_ids=("goal:child",),
        signal_kind=request.signals[0].kind,
        producer_id="leanstral:test",
        producer_kind=RefinementProducerKind.LEANSTRAL,
        rationale="isolate the counterexample into a verifiable child",
    )


@dataclass(frozen=True)
class _Verification:
    verified: bool
    frozen_context: FrozenRefinementContext
    candidate_plan_id: str
    content_id: str = "verification:independent"
    reason: str = ""


def _verification(
    request: AdaptiveRefinementRequest,
    *,
    verified: bool = True,
    candidate_plan_id: str | None = None,
) -> _Verification:
    return _Verification(
        verified=verified,
        frozen_context=request.frozen_context,
        candidate_plan_id=candidate_plan_id or _plan(with_child=True).content_id,
        reason="" if verified else "child-to-parent obligation disproved",
    )


def test_new_counterexample_triggers_exactly_one_bounded_verified_refinement() -> None:
    """Proves objective evidence 003778425160038348524906247302938706902."""

    calls = {"generator": 0, "verifier": 0}

    def generate(request: AdaptiveRefinementRequest) -> AdaptiveRefinementCandidate:
        calls["generator"] += 1
        return _candidate(request)

    def verify(
        candidate: AdaptiveRefinementCandidate, request: AdaptiveRefinementRequest
    ) -> _Verification:
        assert candidate.plan.content_id != request.plan.content_id
        calls["verifier"] += 1
        return _verification(request)

    request = _request()
    policy = AdaptiveRefinementPolicy()
    result = AdaptiveGoalRefiner(
        generate, verify, policy=policy, clock=lambda: 100
    ).refine(request)

    assert result.admitted
    assert result.admitted_plan == _plan(with_child=True)
    assert calls == {"generator": 1, "verifier": 1}
    receipt = result.receipt
    assert receipt.request_id == request.content_id
    assert receipt.cycle_id == request.cycle_id
    assert receipt.evidence_fingerprint == request.evidence_fingerprint
    assert receipt.root_goal_id == request.root_goal_id
    assert receipt.root_goal_content_id == request.root_goal_content_id
    assert receipt.assumption_ids == request.assumption_ids
    assert receipt.policy_id == policy.content_id
    assert receipt.repository_tree_id == request.repository_tree_id
    assert receipt.previous_plan_id == request.plan.content_id
    assert receipt.candidate_plan_id == result.admitted_plan.content_id
    assert receipt.refinement_index == 1
    assert receipt.requirement_ids == (
        NEW_EVIDENCE_REFINEMENT_REQUIREMENT_ID,
    )
    assert receipt.proved_requirement_ids == receipt.requirement_ids
    assert receipt.verification_receipt_id == "verification:independent"
    assert receipt.producer_kind == "leanstral"
    assert receipt.signal_kinds == (RefinementSignalKind.COUNTEREXAMPLE.value,)
    witness = receipt.new_counterexample_evidence
    assert witness is not None
    assert witness.counterexample_signal_id == request.signals[0].evidence_id
    assert witness.candidate_plan_id == result.admitted_plan.content_id
    assert receipt.evidence_ids == (witness.evidence_id,)
    assert AdaptiveRefinementReceipt.from_dict(receipt.to_dict()) == receipt
    assert receipt.receipt_id == receipt.to_dict()["receipt_id"]


def test_g098_completion_requires_fresh_complete_current_tree_proof() -> None:
    """ASI-058: runtime admission stays separate from objective completion."""

    result = AdaptiveGoalRefiner(
        _candidate,
        lambda candidate, request: _verification(request),
        clock=lambda: 100,
    ).refine(_request())
    assert result.admitted
    tree_id = result.receipt.repository_tree_id
    now = datetime(2026, 7, 24, 14, 0, tzinfo=timezone.utc)
    criteria = NEW_COUNTEREXAMPLE_REFINEMENT_ACCEPTANCE_CRITERIA
    evidence = tuple(
        CompletionEvidence(
            acceptance_criterion=criterion,
            producing_task_or_scan="ASI-058",
            producer_kind="task",
            validation_receipt={
                "status": "passed",
                "tree_id": tree_id,
                "command": (
                    "python -m pytest "
                    "test/api/test_agent_supervisor_adaptive_goal_refiner.py -q"
                ),
            },
            validation_passed=True,
            repository_tree=tree_id,
            freshness={"fresh": True},
            observed_at=now,
            provenance_cid=f"validation:asi-058:{index}",
            metadata={
                "evidence_source_policy": {
                    "satisfies": True,
                    "source_tier": "validation_receipt",
                }
            },
        )
        for index, criterion in enumerate(criteria, start=1)
    )
    coverage = {
        "repository_tree": tree_id,
        "evaluated_at": now.isoformat(),
        "verified": True,
        "criteria": [
            {
                "criterion": criterion,
                "status": "verified",
                "verified": True,
                "implementation": (
                    "ipfs_accelerate_py/agent_supervisor/"
                    "adaptive_goal_refiner.py"
                ),
                "validation": (
                    "test/api/test_agent_supervisor_"
                    "adaptive_goal_refiner.py"
                ),
            }
            for criterion in criteria
        ],
    }
    health = {
        "status": "healthy",
        "healthy": True,
        "safe_for_completion_reasoning": True,
        "analyzer_version": "asi-058-completion-analyzer@1",
    }
    binding = {
        "tree_id": tree_id,
        "analyzer_version": "asi-058-completion-analyzer@1",
        "configuration_revision": "asi-058-completion-policy@1",
        "objective_revision": "ASI-G098@asi-058",
    }
    quorum = {
        "required_members": 2,
        "member_count": 2,
        "satisfied": True,
        "quorum_met": True,
        "binding": binding,
        "members": [
            {
                "member_id": "asi-058-exhaustive-implementation",
                "evidence_channel": "implementation-validation",
                "receipt_cid": "scan:asi-058:implementation",
                "binding": binding,
                "scan_mode": "exhaustive",
                "healthy": True,
                "safe_for_completion_reasoning": True,
                "finished_at": now.isoformat(),
            },
            {
                "member_id": "asi-058-exhaustive-receipt-audit",
                "evidence_channel": "receipt-replay-audit",
                "receipt_cid": "scan:asi-058:receipt-audit",
                "binding": binding,
                "scan_mode": "exhaustive",
                "healthy": True,
                "safe_for_completion_reasoning": True,
                "finished_at": now.isoformat(),
            },
        ],
    }
    values = {
        "evidence": evidence,
        "tasks_complete": True,
        "coverage": coverage,
        "analyzer_health": health,
        "exhaustion_quorum": quorum,
        "now": now,
        "freshness_seconds": 300,
    }

    # Runtime evidence cannot self-authorize completion.
    no_proof = result.evaluate_objective_completion(
        current_state=GoalState.ACTIVE,
        tasks_complete=True,
        now=now,
        freshness_seconds=300,
    )
    assert no_proof.state is GoalState.PROVISIONALLY_COMPLETE
    assert not no_proof.verified
    assert no_proof.gate is not None and not no_proof.gate.passed

    # Even a completely passing first evaluation must take the mandatory
    # provisional transition before a later evaluation can verify the goal.
    provisional = result.evaluate_objective_completion(
        current_state=GoalState.ACTIVE,
        **values,
    )
    assert provisional.state is GoalState.PROVISIONALLY_COMPLETE
    assert not provisional.verified
    assert provisional.gate is not None and provisional.gate.passed
    assert provisional.acceptance_criteria == criteria
    assert "provisional_transition_required" in provisional.reason_codes
    assert provisional.gate.evaluated_evidence["analysis_result"] == {}

    verified = result.evaluate_objective_completion(
        current_state=GoalState.PROVISIONALLY_COMPLETE,
        **values,
    )
    assert verified.state is GoalState.VERIFIED_COMPLETE
    assert verified.verified
    assert verified.gate is not None and verified.gate.passed

    # An omitted mandatory record cannot narrow the bridge's closed criterion
    # set, and an extra failed or stale submission cannot be masked by a pass.
    missing = result.evaluate_objective_completion(
        current_state=GoalState.PROVISIONALLY_COMPLETE,
        **{**values, "evidence": evidence[:-1]},
    )
    assert missing.state is GoalState.PROVISIONALLY_COMPLETE
    assert criteria[-1] in missing.missing_criteria
    assert "validation_evidence_incomplete" in missing.gate.fail_reason_codes

    failed = replace(
        evidence[0],
        provenance_cid="validation:asi-058:failed",
        validation_passed=False,
        validation_receipt={"status": "failed", "tree_id": tree_id},
    )
    failed_submission = result.evaluate_objective_completion(
        current_state=GoalState.PROVISIONALLY_COMPLETE,
        **{**values, "evidence": (*evidence, failed)},
    )
    assert failed_submission.state is GoalState.PROVISIONALLY_COMPLETE
    assert "failed_validation" in failed_submission.reason_codes
    assert (
        "validation_evidence_incomplete"
        in failed_submission.gate.fail_reason_codes
    )

    stale = replace(
        evidence[0],
        provenance_cid="validation:asi-058:stale",
        observed_at=datetime(2026, 7, 24, 12, 0, tzinfo=timezone.utc),
    )
    stale_submission = result.evaluate_objective_completion(
        current_state=GoalState.PROVISIONALLY_COMPLETE,
        **{**values, "evidence": (stale, *evidence[1:])},
    )
    assert stale_submission.state is GoalState.PROVISIONALLY_COMPLETE
    assert "stale_evidence" in stale_submission.reason_codes

    # Health must say both healthy and completion-safe explicitly.
    implicit_health = result.evaluate_objective_completion(
        current_state=GoalState.PROVISIONALLY_COMPLETE,
        **{**values, "analyzer_health": {"status": "healthy"}},
    )
    assert implicit_health.state is GoalState.PROVISIONALLY_COMPLETE
    assert "analyzer_unhealthy" in implicit_health.reason_codes

    unsafe_health = result.evaluate_objective_completion(
        current_state=GoalState.PROVISIONALLY_COMPLETE,
        **{
            **values,
            "analyzer_health": {
                **health,
                "safe_for_completion_reasoning": False,
            },
        },
    )
    assert unsafe_health.state is GoalState.PROVISIONALLY_COMPLETE
    assert "analyzer_unhealthy" in unsafe_health.reason_codes

    # Quorum proof requires distinct channels and explicit healthy exhaustive
    # members, all fresh and bound to this exact tree.
    duplicate_channel = {
        **quorum,
        "members": [
            quorum["members"][0],
            {
                **quorum["members"][1],
                "evidence_channel": "implementation-validation",
            },
        ],
    }
    non_independent = result.evaluate_objective_completion(
        current_state=GoalState.PROVISIONALLY_COMPLETE,
        **{**values, "exhaustion_quorum": duplicate_channel},
    )
    assert non_independent.state is GoalState.PROVISIONALLY_COMPLETE
    assert "exhaustion_quorum_inconsistent" in non_independent.reason_codes

    unhealthy_quorum = {
        **quorum,
        "members": [
            quorum["members"][0],
            {**quorum["members"][1], "healthy": False},
        ],
    }
    unhealthy_receipt = result.evaluate_objective_completion(
        current_state=GoalState.PROVISIONALLY_COMPLETE,
        **{**values, "exhaustion_quorum": unhealthy_quorum},
    )
    assert unhealthy_receipt.state is GoalState.PROVISIONALLY_COMPLETE
    assert "exhaustion_quorum_unsatisfied" in unhealthy_receipt.reason_codes

    foreign = replace(
        evidence[0],
        repository_tree="tree:foreign",
        tree_id="tree:foreign",
        provenance_cid="validation:asi-058:foreign",
    )
    wrong_tree = result.evaluate_objective_completion(
        current_state=GoalState.PROVISIONALLY_COMPLETE,
        **{**values, "evidence": (foreign, *evidence[1:])},
    )
    assert wrong_tree.state is GoalState.PROVISIONALLY_COMPLETE
    assert "repository_tree_mismatch" in wrong_tree.reason_codes


def test_replayed_admitted_evidence_is_idempotent_without_more_model_calls() -> None:
    calls = 0
    store = InMemoryRefinementStore()

    def generate(request: AdaptiveRefinementRequest) -> AdaptiveRefinementCandidate:
        nonlocal calls
        calls += 1
        return _candidate(request)

    controller = AdaptiveGoalRefiner(
        generate,
        lambda candidate, request: _verification(request),
        store=store,
        clock=lambda: 100,
    )
    request = _request()

    first = controller.refine(request)
    replay = controller.refine(replace(request, cycle_id="cycle:replay"))

    assert first.decision is RefinementDecision.ADMITTED
    assert replay.decision is RefinementDecision.DUPLICATE
    assert not replay.model_called
    assert calls == 1
    assert len(store.receipts()) == 1


def test_unchanged_failure_signature_backs_off_without_another_model_call() -> None:
    """Proves objective evidence 312819945606360295782005228058369235550."""

    calls = {"generator": 0, "verifier": 0}
    now = [100]

    def generate(request: AdaptiveRefinementRequest) -> AdaptiveRefinementCandidate:
        calls["generator"] += 1
        return _candidate(request)

    def disprove(candidate: AdaptiveRefinementCandidate, request: AdaptiveRefinementRequest):
        calls["verifier"] += 1
        return _verification(request, verified=False)

    controller = AdaptiveGoalRefiner(
        generate,
        disprove,
        policy=AdaptiveRefinementPolicy(
            initial_backoff_seconds=30, max_backoff_seconds=120
        ),
        clock=lambda: now[0],
    )
    request = _request(_signal(kind=RefinementSignalKind.REPEATED_FAILURE))

    failed = controller.refine(request)
    now[0] = 110
    backed_off = controller.refine(
        replace(request, cycle_id="cycle:unchanged", signals=(
            replace(request.signals[0], observed_at=109, occurrence_count=99),
        ))
    )

    assert failed.decision is RefinementDecision.VERIFICATION_FAILED
    assert backed_off.decision is RefinementDecision.BACKED_OFF
    assert not backed_off.model_called
    assert backed_off.receipt.retry_after == 130
    assert calls == {"generator": 1, "verifier": 1}
    assert (
        UNCHANGED_FAILURE_BACKOFF_REQUIREMENT_ID
        in backed_off.receipt.requirement_ids
    )


def test_changed_evidence_bypasses_old_backoff_in_the_next_cycle() -> None:
    calls = 0

    def generate(request: AdaptiveRefinementRequest) -> AdaptiveRefinementCandidate:
        nonlocal calls
        calls += 1
        return _candidate(request)

    controller = AdaptiveGoalRefiner(
        generate,
        lambda candidate, request: _verification(request, verified=False),
        clock=lambda: 101,
    )
    first = controller.refine(_request(_signal("counterexample:v1")))
    changed = controller.refine(
        replace(
            _request(_signal("counterexample:v2")),
            cycle_id="cycle:2",
        )
    )

    assert first.decision is RefinementDecision.VERIFICATION_FAILED
    assert changed.decision is RefinementDecision.VERIFICATION_FAILED
    assert calls == 2
    assert (
        first.receipt.evidence_fingerprint
        != changed.receipt.evidence_fingerprint
    )


def test_distinct_changed_counterexamples_share_one_generation_slot_per_cycle() -> None:
    calls = 0
    controller: AdaptiveGoalRefiner

    def generate(request: AdaptiveRefinementRequest) -> AdaptiveRefinementCandidate:
        nonlocal calls
        calls += 1
        return _candidate(request)

    controller = AdaptiveGoalRefiner(
        generate,
        lambda candidate, request: _verification(request),
        clock=lambda: 101,
    )
    requests = (
        _request(_signal("counterexample:v1")),
        _request(_signal("counterexample:v2")),
    )
    with ThreadPoolExecutor(max_workers=2) as executor:
        same_cycle_results = tuple(executor.map(controller.refine, requests))
    next_cycle = controller.refine(
        replace(
            _request(_signal("counterexample:v3")),
            cycle_id="cycle:2",
        )
    )

    assert sum(item.admitted for item in same_cycle_results) == 1
    exhausted = next(
        item
        for item in same_cycle_results
        if item.decision is RefinementDecision.BUDGET_EXHAUSTED
    )
    assert not exhausted.model_called
    assert {
        item.receipt.cycle_id for item in same_cycle_results
    } == {"cycle:1"}
    assert next_cycle.admitted
    assert next_cycle.receipt.cycle_id == "cycle:2"
    assert calls == 2


@pytest.mark.parametrize(
    "kind",
    [
        RefinementSignalKind.COUNTEREXAMPLE,
        RefinementSignalKind.STALE_EVIDENCE,
        RefinementSignalKind.REPEATED_FAILURE,
        RefinementSignalKind.CAPABILITY_CHANGE,
        RefinementSignalKind.INTERFACE_CHANGE,
        RefinementSignalKind.SCOPE_CHANGE,
        RefinementSignalKind.SCOPE_CONFLICT,
        RefinementSignalKind.RESOURCE_CHANGE,
        RefinementSignalKind.RESOURCE_INFEASIBLE,
    ],
)
def test_all_reviewed_typed_changes_are_eligible(kind: RefinementSignalKind) -> None:
    request = _request(_signal(f"{kind.value}:v1", kind=kind))
    result = AdaptiveGoalRefiner(
        _candidate,
        lambda candidate, current: _verification(current),
        clock=lambda: 100,
    ).refine(request)
    assert result.admitted
    if kind is RefinementSignalKind.COUNTEREXAMPLE:
        assert result.receipt.requirement_ids == (
            NEW_EVIDENCE_REFINEMENT_REQUIREMENT_ID,
        )
        assert result.receipt.new_counterexample_evidence is not None
    else:
        assert result.receipt.requirement_ids == ()
        assert result.receipt.evidence_ids == ()
        assert result.receipt.new_counterexample_evidence is None


@pytest.mark.parametrize("mutation", ["root_id", "root_content", "assumptions", "plan_root"])
def test_frozen_root_and_assumptions_cannot_be_mutated(mutation: str) -> None:
    request = _request()
    candidate = _candidate(request)
    if mutation == "root_id":
        candidate = replace(candidate, root_goal_id="goal:replacement")
    elif mutation == "root_content":
        candidate = replace(candidate, root_goal_content_id="root:replacement")
    elif mutation == "assumptions":
        candidate = replace(candidate, assumption_ids=("assumption:invented",))
    else:
        candidate = replace(
            candidate,
            plan=_plan(with_child=True, root_outcome="formula:weakened"),
        )
    verifier_calls = 0

    def verify(candidate, request):
        nonlocal verifier_calls
        verifier_calls += 1
        return _verification(request)

    result = AdaptiveGoalRefiner(
        lambda current: candidate, verify, clock=lambda: 100
    ).refine(request)

    assert result.decision is RefinementDecision.CANDIDATE_REJECTED
    assert result.admitted_plan is None
    assert verifier_calls == 0


def test_independent_verification_must_bind_the_exact_frozen_context() -> None:
    request = _request()
    mismatched = _Verification(
        verified=True,
        frozen_context=FrozenRefinementContext(
            request.root_goal_id,
            request.root_goal_content_id,
            ("assumption:changed",),
        ),
        candidate_plan_id=_plan(with_child=True).content_id,
    )
    result = AdaptiveGoalRefiner(
        _candidate, lambda candidate, current: mismatched, clock=lambda: 100
    ).refine(request)
    assert result.decision is RefinementDecision.VERIFICATION_FAILED
    assert "frozen context" in result.receipt.reason


def test_verification_for_another_candidate_plan_cannot_be_replayed() -> None:
    request = _request()
    result = AdaptiveGoalRefiner(
        _candidate,
        lambda candidate, current: _verification(
            current, candidate_plan_id=current.plan.content_id
        ),
        clock=lambda: 100,
    ).refine(request)

    assert result.decision is RefinementDecision.VERIFICATION_FAILED
    assert result.admitted_plan is None
    assert result.receipt.requirement_ids == ()
    assert "another plan" in result.receipt.reason


def test_candidate_must_bind_request_signal_kind_and_repository_tree() -> None:
    request = _request()
    wrong_kind = replace(
        _candidate(request), signal_kind=RefinementSignalKind.CAPABILITY_CHANGE
    )
    kind_result = AdaptiveGoalRefiner(
        lambda current: wrong_kind,
        lambda candidate, current: _verification(current),
        clock=lambda: 100,
    ).refine(request)
    assert kind_result.decision is RefinementDecision.CANDIDATE_REJECTED
    assert "signal kind" in kind_result.receipt.reason
    assert kind_result.receipt.requirement_ids == ()

    wrong_tree_plan = replace(
        _plan(with_child=True), repository_tree_id="tree:other"
    )
    wrong_tree = replace(_candidate(request), plan=wrong_tree_plan)
    tree_result = AdaptiveGoalRefiner(
        lambda current: wrong_tree,
        lambda candidate, current: _verification(
            current, candidate_plan_id=wrong_tree_plan.content_id
        ),
        clock=lambda: 100,
    ).refine(request)
    assert tree_result.decision is RefinementDecision.CANDIDATE_REJECTED
    assert "repository tree" in tree_result.receipt.reason
    assert tree_result.receipt.requirement_ids == ()


def test_request_repository_tree_must_match_frozen_plan() -> None:
    with pytest.raises(AdaptiveGoalRefinementError, match="repository tree"):
        replace(_request(), repository_tree_id="tree:other")


def test_counterexample_witness_tampering_fails_closed() -> None:
    request = _request()
    result = AdaptiveGoalRefiner(
        _candidate,
        lambda candidate, current: _verification(current),
        clock=lambda: 100,
    ).refine(request)
    payload = result.receipt.to_dict()
    payload["new_counterexample_evidence"]["candidate_plan_id"] = (
        request.plan.content_id
    )

    with pytest.raises(
        AdaptiveGoalRefinementError, match="evidence identity does not match"
    ):
        AdaptiveRefinementReceipt.from_dict(payload)


def test_persisted_objective_receipts_fail_closed_on_unreviewed_shape() -> None:
    result = AdaptiveGoalRefiner(
        _candidate,
        lambda candidate, current: _verification(current),
        clock=lambda: 100,
    ).refine(_request())
    payload = result.receipt.to_dict()

    unsupported = dict(payload)
    unsupported["version"] = ADAPTIVE_GOAL_REFINER_VERSION + 1
    with pytest.raises(AdaptiveGoalRefinementError, match="receipt version"):
        AdaptiveRefinementReceipt.from_dict(unsupported)

    unknown = dict(payload)
    unknown["unreviewed_claim"] = "proved"
    with pytest.raises(AdaptiveGoalRefinementError, match="unknown refinement receipt"):
        AdaptiveRefinementReceipt.from_dict(unknown)

    missing_identity = dict(payload)
    missing_identity.pop("receipt_id")
    with pytest.raises(AdaptiveGoalRefinementError, match="identity is required"):
        AdaptiveRefinementReceipt.from_dict(missing_identity)

    unknown_witness = result.receipt.to_dict()
    unknown_witness["new_counterexample_evidence"]["unreviewed_claim"] = "proved"
    with pytest.raises(
        AdaptiveGoalRefinementError,
        match="unknown counterexample-refinement evidence",
    ):
        AdaptiveRefinementReceipt.from_dict(unknown_witness)


def test_bare_boolean_verifier_cannot_assert_proof() -> None:
    result = AdaptiveGoalRefiner(
        _candidate, lambda candidate, current: True, clock=lambda: 100
    ).refine(_request())
    assert result.decision is RefinementDecision.VERIFICATION_FAILED
    assert "boolean" in result.receipt.reason


def test_non_boolean_verification_status_cannot_assert_proof() -> None:
    request = _request()
    malformed = _Verification(
        verified="false",  # type: ignore[arg-type]
        frozen_context=request.frozen_context,
        candidate_plan_id=_plan(with_child=True).content_id,
    )
    result = AdaptiveGoalRefiner(
        _candidate, lambda candidate, current: malformed, clock=lambda: 100
    ).refine(request)

    assert result.decision is RefinementDecision.VERIFICATION_FAILED
    assert result.receipt.requirement_ids == ()
    assert "must be boolean" in result.receipt.reason


def test_depth_and_per_root_budgets_stop_before_generation() -> None:
    calls = 0

    def generate(request):
        nonlocal calls
        calls += 1
        return _candidate(request)

    policy = AdaptiveRefinementPolicy(
        max_refinements_per_root=1, max_refinement_depth=1
    )
    depth_result = AdaptiveGoalRefiner(
        generate,
        lambda candidate, request: _verification(request),
        policy=policy,
        clock=lambda: 100,
    ).refine(_request(depth=1))
    assert depth_result.decision is RefinementDecision.BUDGET_EXHAUSTED
    assert calls == 0

    store = InMemoryRefinementStore()
    controller = AdaptiveGoalRefiner(
        generate,
        lambda candidate, request: _verification(request),
        policy=policy,
        store=store,
        clock=lambda: 100,
    )
    assert controller.refine(_request(_signal("revision:1"))).admitted
    exhausted = controller.refine(_request(_signal("revision:2")))
    assert exhausted.decision is RefinementDecision.BUDGET_EXHAUSTED
    assert calls == 1


def test_changed_goal_declaration_and_change_budget_are_enforced() -> None:
    request = _request()
    omitted = replace(_candidate(request), changed_goal_ids=("goal:unrelated",))
    result = AdaptiveGoalRefiner(
        lambda current: omitted,
        lambda candidate, current: _verification(current),
        clock=lambda: 100,
    ).refine(request)
    assert result.decision is RefinementDecision.CANDIDATE_REJECTED
    assert "omitted changed goals" in result.receipt.reason

    overdeclared = replace(
        _candidate(request),
        changed_goal_ids=("goal:child", "goal:unchanged-or-unknown"),
    )
    result = AdaptiveGoalRefiner(
        lambda current: overdeclared,
        lambda candidate, current: _verification(current),
        clock=lambda: 100,
    ).refine(request)
    assert result.decision is RefinementDecision.CANDIDATE_REJECTED
    assert "unchanged or unknown goals" in result.receipt.reason

    result = AdaptiveGoalRefiner(
        _candidate,
        lambda candidate, current: _verification(current),
        policy=AdaptiveRefinementPolicy(max_changed_goals=1),
        clock=lambda: 100,
    ).refine(request)
    assert result.admitted


def test_jsonl_store_survives_restart_and_suppresses_duplicate_generation(tmp_path) -> None:
    path = tmp_path / "adaptive-refinement.jsonl"
    calls = 0

    def generate(request):
        nonlocal calls
        calls += 1
        return _candidate(request)

    request = _request()
    first = AdaptiveGoalRefiner(
        generate,
        lambda candidate, current: _verification(current),
        store=JsonlRefinementStore(path),
        clock=lambda: 100,
    ).refine(request)
    restarted = AdaptiveGoalRefiner(
        generate,
        lambda candidate, current: _verification(current),
        store=JsonlRefinementStore(path),
        clock=lambda: 101,
    ).refine(replace(request, cycle_id="cycle:restart"))

    assert first.admitted
    assert restarted.decision is RefinementDecision.DUPLICATE
    assert calls == 1
    persisted = JsonlRefinementStore(path).receipts()
    assert persisted == (first.receipt,)
    assert AdaptiveRefinementReceipt.from_dict(
        first.receipt.to_dict()
    ) == first.receipt


def test_concurrent_same_evidence_performs_one_generation_and_admission() -> None:
    calls = 0
    call_lock = __import__("threading").Lock()
    store = InMemoryRefinementStore()

    def generate(request):
        nonlocal calls
        with call_lock:
            calls += 1
        return _candidate(request)

    controller = AdaptiveGoalRefiner(
        generate,
        lambda candidate, current: _verification(current),
        store=store,
        clock=lambda: 100,
    )
    request = _request()
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda _: controller.refine(request), range(4)))

    assert calls == 1
    assert sum(result.admitted for result in results) == 1
    assert sum(
        result.decision is RefinementDecision.DUPLICATE for result in results
    ) == 3


def test_goal_quality_records_all_dimensions_and_deterministic_debt() -> None:
    quality = GoalQualityRecord(
        goal_id="goal:root",
        outcome="",
        scope_ids=(),
        assumption_ids=(),
        non_goals=(),
        acceptance_criteria=(),
        evidence_producer_ids=(),
        validation_ids=(),
        freshness_horizon_seconds=0,
        resource_envelope={},
        unsupported_semantics=("natural-language-implication",),
        breadth=9,
        max_breadth=4,
    )
    assert set(quality.debt) == set(GoalDebtKind)
    assert quality.to_dict()["debt"] == tuple(item.value for item in quality.debt)


def test_policy_and_signal_validation_fail_closed() -> None:
    with pytest.raises(AdaptiveGoalRefinementError, match="exactly one"):
        AdaptiveRefinementPolicy(max_model_calls_per_cycle=2)
    with pytest.raises(AdaptiveGoalRefinementError, match="failure_signature"):
        _signal(kind=RefinementSignalKind.REPEATED_FAILURE).__class__(
            kind=RefinementSignalKind.REPEATED_FAILURE,
            subject_id="goal:root",
            evidence_revision="failure:v1",
            observed_at=1,
        )
    with pytest.raises(AdaptiveGoalRefinementError, match="independent"):
        AdaptiveGoalRefiner(_candidate, None)  # type: ignore[arg-type]
