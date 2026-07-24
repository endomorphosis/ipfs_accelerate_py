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
    content_id: str = "verification:independent"
    reason: str = ""


def _verification(
    request: AdaptiveRefinementRequest, *, verified: bool = True
) -> _Verification:
    return _Verification(
        verified=verified,
        frozen_context=request.frozen_context,
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
    result = AdaptiveGoalRefiner(generate, verify, clock=lambda: 100).refine(request)

    assert result.admitted
    assert result.admitted_plan == _plan(with_child=True)
    assert calls == {"generator": 1, "verifier": 1}
    assert result.receipt.root_goal_content_id == request.root_goal_content_id
    assert result.receipt.assumption_ids == request.assumption_ids
    assert result.receipt.refinement_index == 1
    assert (
        NEW_EVIDENCE_REFINEMENT_REQUIREMENT_ID
        in result.receipt.requirement_ids
    )
    assert result.receipt.verification_receipt_id == "verification:independent"
    assert result.receipt.producer_kind == "leanstral"


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
    )
    result = AdaptiveGoalRefiner(
        _candidate, lambda candidate, current: mismatched, clock=lambda: 100
    ).refine(request)
    assert result.decision is RefinementDecision.VERIFICATION_FAILED
    assert "frozen context" in result.receipt.reason


def test_bare_boolean_verifier_cannot_assert_proof() -> None:
    result = AdaptiveGoalRefiner(
        _candidate, lambda candidate, current: True, clock=lambda: 100
    ).refine(_request())
    assert result.decision is RefinementDecision.VERIFICATION_FAILED
    assert "boolean" in result.receipt.reason


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
