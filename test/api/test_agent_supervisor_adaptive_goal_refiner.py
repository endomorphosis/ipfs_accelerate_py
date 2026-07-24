from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace

import pytest

from ipfs_accelerate_py.agent_supervisor.adaptive_goal_refiner import (
    NEW_EVIDENCE_REFINEMENT_REQUIREMENT_ID,
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
    changed = controller.refine(_request(_signal("counterexample:v2")))

    assert first.decision is RefinementDecision.VERIFICATION_FAILED
    assert changed.decision is RefinementDecision.VERIFICATION_FAILED
    assert calls == 2
    assert (
        first.receipt.evidence_fingerprint
        != changed.receipt.evidence_fingerprint
    )


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
    unsupported["version"] = 2
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
