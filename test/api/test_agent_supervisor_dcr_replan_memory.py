"""DCR-063: typed failure memory and non-thrashing replanning.

Acceptance:
* Replaying unchanged inputs emits no duplicate work.
* Retry/rescue cannot route to a provider/model.
* Retry/rescue cannot repeat a refuted candidate.
* Retry only on typed new evidence or a strictly decreasing measure.
* Stale/conflict/validation/proof/resource/capability failures persist across
  restart.
* Runtime model calls remain 0; write authority is never granted.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.planning.adaptive_planner import (
    AdaptivePlanner,
    FAILURE_MEMORY_INTERFACE as ADAPTIVE_FAILURE_MEMORY_INTERFACE,
    REPLAN_DECISION_INTERFACE as ADAPTIVE_REPLAN_DECISION_INTERFACE,
    decide_replan as adaptive_decide_replan,
)
from ipfs_accelerate_py.agent_supervisor.planning.deterministic_failure_memory import (
    DCR_REPLAN_EVIDENCE,
    FAILURE_MEMORY_INTERFACE,
    REPLAN_DECISION_INTERFACE,
    AttemptRouteKind,
    FailureAttempt,
    FailureClass,
    FailureMemory,
    FailureMemoryError,
    FailureMemoryPolicy,
    FailureMemoryReceipt,
    ReplanDecision,
    ReplanDisposition,
    RetryMeasure,
    decide_replan,
    is_provider_or_model_route,
    materialize_replan_fixtures,
)
from ipfs_accelerate_py.agent_supervisor.planning.formal_replanner import (
    FormalReplanner,
    FAILURE_MEMORY_INTERFACE as FORMAL_FAILURE_MEMORY_INTERFACE,
    REPLAN_DECISION_INTERFACE as FORMAL_REPLAN_DECISION_INTERFACE,
    decide_replan as formal_decide_replan,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    content_identity,
)


def _measure(**overrides: int) -> RetryMeasure:
    base = {
        "open_counterexamples": 2,
        "validation_findings": 3,
        "remaining_candidates": 4,
        "resource_debt": 100,
        "capability_gaps": 1,
    }
    base.update(overrides)
    return RetryMeasure(**base)


def _attempt(
    *,
    failure_class: FailureClass = FailureClass.VALIDATION,
    prior: str = "candidate:op-a",
    evidence: str = "evidence:v1",
    measure: RetryMeasure | None = None,
    route: AttemptRouteKind = AttemptRouteKind.DETERMINISTIC_OPERATOR,
    scope: str = "scope:dcr063",
    plan: str = "plan:dcr063",
    cex: tuple[str, ...] = ("cex:1",),
    refuted: bool = True,
    operator: str = "add_registration",
) -> FailureAttempt:
    return FailureAttempt(
        failure_class=failure_class,
        prior_candidate_cid=prior,
        evidence_cid=evidence,
        measure=measure or _measure(),
        route_kind=route,
        scope_id=scope,
        plan_id=plan,
        counterexample_ids=cex,
        refuted=refuted,
        operator_kind=operator,
    )


def test_interfaces_and_evidence_are_stable() -> None:
    assert FAILURE_MEMORY_INTERFACE == "FailureMemory@1"
    assert REPLAN_DECISION_INTERFACE == "ReplanDecision@1"
    assert DCR_REPLAN_EVIDENCE == "dcr/replan@1"
    assert FORMAL_FAILURE_MEMORY_INTERFACE == FAILURE_MEMORY_INTERFACE
    assert FORMAL_REPLAN_DECISION_INTERFACE == REPLAN_DECISION_INTERFACE
    assert ADAPTIVE_FAILURE_MEMORY_INTERFACE == FAILURE_MEMORY_INTERFACE
    assert ADAPTIVE_REPLAN_DECISION_INTERFACE == REPLAN_DECISION_INTERFACE
    assert FormalReplanner is not None
    assert AdaptivePlanner is not None


@pytest.mark.parametrize("failure_class", tuple(FailureClass))
def test_each_failure_class_is_persistable(failure_class: FailureClass) -> None:
    memory = FailureMemory()
    receipt = memory.record_attempt(
        _attempt(
            failure_class=failure_class,
            prior=f"candidate:{failure_class.value}",
            evidence=f"evidence:{failure_class.value}",
            cex=(f"cex:{failure_class.value}",),
        ),
        proposed_candidate_cid=f"candidate:next-{failure_class.value}",
    )
    assert isinstance(receipt, FailureMemoryReceipt)
    assert receipt.decision.should_replan is True
    assert receipt.decision.disposition is ReplanDisposition.RETRY_NEW_EVIDENCE
    assert receipt.decision.failure_class is failure_class
    assert receipt.decision.runtime_model_calls == 0
    assert receipt.decision.grants_write_authority is False
    assert receipt.decision.allows_provider_route is False
    subset = receipt.evidence_subset()
    assert subset["attempt_key"]
    assert subset["failure_class"] == failure_class.value
    assert subset["prior_candidate"] == f"candidate:{failure_class.value}"
    assert subset["new_evidence"] == f"evidence:{failure_class.value}"
    assert subset["measure"]["open_counterexamples"] == 2
    assert subset["disposition"] == ReplanDisposition.RETRY_NEW_EVIDENCE.value


def test_replay_unchanged_inputs_emits_no_duplicate_work() -> None:
    memory = FailureMemory()
    attempt = _attempt()
    first = memory.record_attempt(
        attempt,
        observed_at_milliseconds=10,
        proposed_candidate_cid="candidate:op-b",
    )
    second = memory.record_attempt(
        attempt,
        observed_at_milliseconds=11,
        proposed_candidate_cid="candidate:op-b",
    )
    pure = decide_replan(
        attempt,
        memory=memory,
        proposed_candidate_cid="candidate:op-b",
    )

    assert first.decision.should_replan is True
    assert first.decision.emits_work is True
    assert second.decision.should_replan is False
    assert second.decision.emits_work is False
    assert second.decision.disposition is ReplanDisposition.NO_DUPLICATE_WORK
    assert pure.disposition is ReplanDisposition.NO_DUPLICATE_WORK
    assert pure.emits_work is False
    assert "no_duplicate_work" in pure.reason_codes


def test_retry_cannot_route_to_provider_or_model() -> None:
    memory = FailureMemory()
    provider_attempt = _attempt(route=AttemptRouteKind.PROVIDER)
    model_decision = decide_replan(
        _attempt(route=AttemptRouteKind.DETERMINISTIC_OPERATOR),
        memory=memory,
        proposed_candidate_cid="candidate:op-b",
        proposed_route_kind=AttemptRouteKind.MODEL,
    )
    llm_decision = decide_replan(
        _attempt(route=AttemptRouteKind.LLM, prior="candidate:op-x"),
        memory=memory,
        proposed_route_kind=AttemptRouteKind.DETERMINISTIC_REPLAN,
    )
    recorded = memory.record_attempt(
        provider_attempt,
        proposed_candidate_cid="candidate:op-b",
        proposed_route_kind=AttemptRouteKind.PROVIDER,
    )

    assert is_provider_or_model_route(AttemptRouteKind.PROVIDER)
    assert is_provider_or_model_route("model_route")
    assert model_decision.disposition is ReplanDisposition.PROVIDER_ROUTE_FORBIDDEN
    assert model_decision.should_replan is False
    assert model_decision.allows_provider_route is False
    assert llm_decision.disposition is ReplanDisposition.PROVIDER_ROUTE_FORBIDDEN
    assert recorded.decision.disposition is ReplanDisposition.PROVIDER_ROUTE_FORBIDDEN
    assert recorded.decision.emits_work is False


def test_retry_cannot_repeat_refuted_candidate() -> None:
    memory = FailureMemory()
    first = memory.record_attempt(
        _attempt(prior="candidate:op-a", evidence="evidence:v1"),
        proposed_candidate_cid="candidate:op-b",
    )
    assert first.decision.should_replan is True
    assert memory.is_refuted("candidate:op-a")

    repeat = decide_replan(
        _attempt(prior="candidate:op-b", evidence="evidence:v2"),
        memory=memory,
        proposed_candidate_cid="candidate:op-a",
        proposed_route_kind=AttemptRouteKind.RESCUE_OPERATOR,
    )
    assert repeat.disposition is ReplanDisposition.REFUTED_CANDIDATE
    assert repeat.should_replan is False
    assert "candidate:op-a" in repeat.refuted_candidate_cids

    filtered = memory.filter_admissible_candidates(
        ("candidate:op-a", "candidate:op-b", "candidate:op-c")
    )
    assert filtered == ("candidate:op-b", "candidate:op-c")


def test_strictly_decreasing_measure_authorizes_retry_without_new_evidence() -> None:
    memory = FailureMemory()
    memory.record_attempt(
        _attempt(evidence="evidence:v1", measure=_measure()),
        proposed_candidate_cid="candidate:op-b",
    )
    improved = _attempt(
        prior="candidate:op-b",
        evidence="evidence:v1",
        measure=_measure(
            open_counterexamples=1,
            validation_findings=1,
            remaining_candidates=2,
            resource_debt=10,
            capability_gaps=0,
        ),
    )
    decision = memory.record_attempt(
        improved,
        proposed_candidate_cid="candidate:op-c",
        proposed_route_kind=AttemptRouteKind.RESCUE_OPERATOR,
    )
    assert decision.decision.disposition is (
        ReplanDisposition.RETRY_STRICTLY_DECREASING_MEASURE
    )
    assert decision.decision.should_replan is True
    assert decision.decision.selected_candidate_cid == "candidate:op-c"

    # Non-decreasing measure with same evidence must not thrash.
    stalled = decide_replan(
        _attempt(
            prior="candidate:op-c",
            evidence="evidence:v1",
            measure=_measure(
                open_counterexamples=1,
                validation_findings=1,
                remaining_candidates=2,
                resource_debt=10,
                capability_gaps=0,
            ),
        ),
        memory=memory,
        proposed_candidate_cid="candidate:op-d",
    )
    assert stalled.should_replan is False
    assert stalled.disposition in {
        ReplanDisposition.NO_DUPLICATE_WORK,
        ReplanDisposition.ABSTAIN,
    }


def test_new_typed_evidence_authorizes_retry() -> None:
    memory = FailureMemory()
    memory.record_attempt(_attempt(evidence="evidence:v1"))
    decision = decide_replan(
        _attempt(prior="candidate:op-b", evidence="evidence:v2"),
        memory=memory,
        proposed_candidate_cid="candidate:op-c",
    )
    assert decision.disposition is ReplanDisposition.RETRY_NEW_EVIDENCE
    assert decision.should_replan is True


def test_memory_persists_across_restart(tmp_path: Path) -> None:
    state = tmp_path / "failure-memory.json"
    memory = FailureMemory(
        state,
        policy=FailureMemoryPolicy(
            max_attempts=32,
            max_refuted_candidates=16,
            max_retries_per_attempt_key=4,
            max_retries_per_scope=8,
        ),
    )
    for failure_class in FailureClass:
        memory.record_attempt(
            _attempt(
                failure_class=failure_class,
                prior=f"candidate:{failure_class.value}",
                evidence=f"evidence:{failure_class.value}",
                cex=(f"cex:{failure_class.value}",),
            ),
            proposed_candidate_cid=f"candidate:next-{failure_class.value}",
        )
    state_id = memory.state_id
    counterexamples = memory.counterexample_ids
    refuted = memory.refuted_candidate_cids

    restarted = FailureMemory(state)
    assert restarted.state_id == state_id
    assert restarted.counterexample_ids == counterexamples
    assert restarted.refuted_candidate_cids == refuted
    assert set(item.failure_class for item in restarted.snapshot().attempts) == set(
        FailureClass
    )

    # Replay after restart still emits no duplicate work.
    replay = restarted.record_attempt(
        _attempt(
            failure_class=FailureClass.VALIDATION,
            prior="candidate:validation",
            evidence="evidence:validation",
            cex=("cex:validation",),
        ),
        proposed_candidate_cid="candidate:next-validation",
    )
    assert replay.decision.disposition is ReplanDisposition.NO_DUPLICATE_WORK
    assert replay.decision.emits_work is False
    # Counterexamples are never erased.
    assert "cex:validation" in restarted.counterexample_ids


def test_formal_replanner_decide_replan_is_non_thrashing() -> None:
    memory = FailureMemory()
    replanner = FormalReplanner(failure_memory=memory)
    attempt = _attempt()
    first = replanner.decide_replan(
        attempt,
        proposed_candidate_cid="candidate:op-b",
        record=True,
    )
    second = replanner.decide_replan(
        attempt,
        proposed_candidate_cid="candidate:op-b",
        record=True,
    )
    assert isinstance(first, FailureMemoryReceipt)
    assert first.decision.should_replan is True
    assert second.decision.disposition is ReplanDisposition.NO_DUPLICATE_WORK
    assert replanner.filter_refuted_candidates(
        ("candidate:op-a", "candidate:op-b")
    ) == ("candidate:op-b",)

    module_level = formal_decide_replan(
        attempt,
        failure_memory=memory,
        proposed_candidate_cid="candidate:op-a",
        record=False,
    )
    assert isinstance(module_level, ReplanDecision)
    assert module_level.disposition is ReplanDisposition.REFUTED_CANDIDATE


def test_adaptive_planner_blocks_provider_retry_and_refuted_candidates() -> None:
    memory = FailureMemory()
    planner = AdaptivePlanner(max_candidates=4)
    attempt = _attempt()
    receipt = planner.decide_replan(
        attempt,
        failure_memory=memory,
        proposed_candidate_cid="candidate:op-b",
        record=True,
    )
    assert receipt.decision.should_replan is True

    blocked = planner.plan(
        # frozen_goal is unused when replan gate rejects; still construct a
        # minimal stand-in via decide_replan path only.
        frozen_goal=_minimal_frozen_goal(),
        context={},
        failure_memory=memory,
        replan_attempt=attempt,
        proposed_candidate_cid="candidate:op-a",
        proposed_route_kind=AttemptRouteKind.MODEL,
        allow_model=True,
        model_provider=lambda _req: {"candidates": []},
    )
    assert isinstance(blocked, FailureMemoryReceipt)
    assert blocked.decision.should_replan is False
    assert blocked.decision.disposition in {
        ReplanDisposition.PROVIDER_ROUTE_FORBIDDEN,
        ReplanDisposition.REFUTED_CANDIDATE,
        ReplanDisposition.NO_DUPLICATE_WORK,
    }

    filtered = planner.filter_refuted_candidates(
        ("candidate:op-a", "candidate:op-z"),
        failure_memory=memory,
    )
    assert "candidate:op-a" not in filtered
    assert adaptive_decide_replan is decide_replan


def test_retry_measure_ordering_is_lexicographic() -> None:
    worse = _measure(open_counterexamples=2, validation_findings=5)
    better = _measure(open_counterexamples=1, validation_findings=9)
    equal = _measure(open_counterexamples=2, validation_findings=5)
    assert better.strictly_decreases(worse)
    assert not worse.strictly_decreases(better)
    assert not equal.strictly_decreases(worse)


def test_materialize_replan_fixtures(tmp_path: Path) -> None:
    destination = tmp_path / "replan-fixtures.json"
    payload = materialize_replan_fixtures(destination=destination)
    assert destination.is_file()
    assert payload["evidence_id"] == DCR_REPLAN_EVIDENCE
    assert payload["runtime_model_calls"] == 0
    assert payload["grants_write_authority"] is False
    assert payload["interfaces"]["failure_memory"] == FAILURE_MEMORY_INTERFACE
    assert (
        payload["receipts"]["replay_no_duplicate"]["decision"]["disposition"]
        == ReplanDisposition.NO_DUPLICATE_WORK.value
    )
    assert (
        payload["decisions"]["provider_route_forbidden"]["disposition"]
        == ReplanDisposition.PROVIDER_ROUTE_FORBIDDEN.value
    )
    assert (
        payload["decisions"]["refuted_candidate"]["disposition"]
        == ReplanDisposition.REFUTED_CANDIDATE.value
    )


def test_closed_schema_rejects_open_fields() -> None:
    with pytest.raises(FailureMemoryError):
        FailureAttempt.from_dict(
            {
                "failure_class": "validation",
                "prior_candidate_cid": "candidate:x",
                "evidence_cid": "evidence:x",
                "measure": {},
                "extra": "nope",
            }
        )


def _minimal_frozen_goal():
    from ipfs_accelerate_py.agent_supervisor.planning.adaptive_planner import (
        FrozenPlanningGoal,
    )
    from ipfs_accelerate_py.agent_supervisor.planning.plan_evaluator import (
        EvidenceAwarePlanPolicy,
    )

    return FrozenPlanningGoal(
        goal_id="goal:dcr063",
        goal_content_id=content_identity({"goal": "dcr063"}),
        repository_tree_id="tree:dcr063",
        policy=EvidenceAwarePlanPolicy(
            acceptance_criteria=("criterion:dcr063",),
            evidence_terms=("evidence:dcr063",),
        ),
    )
