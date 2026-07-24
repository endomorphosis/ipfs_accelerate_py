from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.formal_plan_conformance import (
    CanonicalExecutionEvent,
    CompletionEvidenceKind,
    CompletionPolicy,
    ConformanceBinding,
    ConformanceReplayPacket,
    ConformanceVerdict,
    FormalCompletionEvidence,
    InvalidationCause,
    TransitionDisposition,
    binding_for_plan,
    evaluate_formal_goal_completion,
    evaluate_plan_conformance,
    read_conformance_evidence,
    replay_conformance_evidence,
    write_conformance_evidence,
)
from ipfs_accelerate_py.agent_supervisor.formal_planning_contracts import (
    Actor,
    ActorKind,
    EventKind,
    FormalWorkPlan,
    Goal,
    PlanEvent,
    PlanTask,
)
from ipfs_accelerate_py.agent_supervisor.goal_completion import GoalState


NOW = "2026-07-23T20:00:00Z"


def _plan(*, revision: str = "one") -> FormalWorkPlan:
    return FormalWorkPlan(
        vocabulary_profile_id="supervisor-reviewed",
        vocabulary_version=1,
        actors=(
            Actor(
                actor_id="agent:one",
                kind=ActorKind.AGENT,
                capabilities=("implement",),
                authority_ids=("authority:goal",),
            ),
        ),
        goals=(
            Goal(
                goal_id="G12.S4",
                owner_actor_id="agent:one",
                satisfaction_formula_id="formula:goal",
            ),
        ),
        subgoals=(),
        tasks=(
            PlanTask(
                task_id="REF-290",
                goal_id="G12.S4",
                actor_ids=("agent:one",),
                event_ids=("event:start", "event:complete"),
            ),
        ),
        events=(
            PlanEvent(
                event_id="event:start",
                kind=EventKind.STARTED,
                actor_id="agent:one",
                task_id="REF-290",
                logical_time=1,
            ),
            PlanEvent(
                event_id="event:complete",
                kind=EventKind.COMPLETED,
                actor_id="agent:one",
                task_id="REF-290",
                logical_time=2,
            ),
        ),
        fluents=(),
        preconditions=(),
        effects=(),
        norms=(),
        temporal_constraints=(),
        evidence_requirements=(),
        source_ids=("source:objective",),
        repository_tree_id="tree:current",
        trace_bound=2,
        metadata={"revision": revision},
    )


def _events(plan: FormalWorkPlan | None = None) -> tuple[CanonicalExecutionEvent, ...]:
    plan = plan or _plan()
    return (
        CanonicalExecutionEvent(
            event_id="observed:start",
            plan_event_id="event:start",
            task_id="REF-290",
            kind=EventKind.STARTED.value,
            actor_id="agent:one",
            sequence=1,
            plan_id=plan.plan_id,
            repository_tree_id=plan.repository_tree_id,
            authorized=True,
            provenance_ids=("receipt:start",),
        ),
        CanonicalExecutionEvent(
            event_id="observed:complete",
            plan_event_id="event:complete",
            task_id="REF-290",
            kind=EventKind.COMPLETED.value,
            actor_id="agent:one",
            sequence=2,
            plan_id=plan.plan_id,
            repository_tree_id=plan.repository_tree_id,
            authorized=True,
            provenance_ids=("receipt:complete",),
        ),
    )


def _evidence(
    binding: ConformanceBinding,
    *,
    omit: CompletionEvidenceKind | None = None,
    freshness: str = "current",
) -> tuple[FormalCompletionEvidence, ...]:
    return tuple(
        FormalCompletionEvidence(
            kind=kind,
            goal_id="G12.S4",
            artifact_id=f"artifact:{kind.value}",
            binding=binding,
            observed_at="2026-07-23T19:55:00Z",
            verdict="passed",
            freshness=freshness,
            provider_id=f"provider:{kind.value}",
        )
        for kind in CompletionEvidenceKind
        if kind is not omit
    )


def test_canonical_events_conform_to_exact_accepted_plan() -> None:
    plan = _plan()
    policy = CompletionPolicy()
    binding = binding_for_plan(
        plan,
        policy,
        ast_scope_ids=("ast:module",),
        premise_ids=("premise:authority",),
        counterexample_ids=("counterexample:known",),
    )

    result = evaluate_plan_conformance(
        plan, reversed(_events(plan)), policy=policy, binding=binding
    )

    # Input collection order is immaterial; canonical sequence controls replay.
    assert result.verdict is ConformanceVerdict.CONFORMANT
    assert result.conformant
    assert result.level.value == "bounded_conformant"
    assert len(result.by_disposition(TransitionDisposition.MATCHED)) == 2
    assert not result.skipped
    assert result.binding == binding
    assert result.to_json() == result.to_json()


def test_skipped_reordered_unauthorized_failed_overridden_and_superseded_are_distinct() -> None:
    plan = _plan()
    policy = CompletionPolicy()
    binding = binding_for_plan(plan, policy)
    scenarios = {
        TransitionDisposition.SKIPPED: _events(plan)[:1],
        TransitionDisposition.REORDERED: (
            replace(_events(plan)[0], sequence=2),
            replace(_events(plan)[1], sequence=1),
        ),
        TransitionDisposition.UNAUTHORIZED: (
            replace(_events(plan)[0], actor_id="agent:intruder"),
            _events(plan)[1],
        ),
        TransitionDisposition.FAILED: (
            _events(plan)[0],
            replace(_events(plan)[1], kind="failed", status="failed"),
        ),
        TransitionDisposition.OVERRIDDEN: (
            _events(plan)[0],
            replace(
                _events(plan)[1],
                plan_event_id="",
                overrides_event_id="event:complete",
            ),
        ),
        TransitionDisposition.SUPERSEDED: (
            _events(plan)[0],
            replace(
                _events(plan)[1],
                plan_event_id="",
                supersedes_event_id="event:complete",
            ),
        ),
    }

    for disposition, events in scenarios.items():
        result = evaluate_plan_conformance(
            plan, events, policy=policy, binding=binding
        )
        assert result.by_disposition(disposition), disposition
        assert result.verdict in {
            ConformanceVerdict.INCOMPLETE,
            ConformanceVerdict.VIOLATED,
        }

    allowed = CompletionPolicy(allow_overridden=True, allow_superseded=True)
    allowed_binding = binding_for_plan(plan, allowed)
    for field in ("overrides_event_id", "supersedes_event_id"):
        event = replace(
            _events(plan)[1],
            plan_event_id="",
            **{field: "event:complete"},
        )
        result = evaluate_plan_conformance(
            plan, (_events(plan)[0], event), policy=allowed, binding=allowed_binding
        )
        assert result.verdict is ConformanceVerdict.CONFORMANT


def test_plan_consistency_never_substitutes_for_any_code_or_validation_lane() -> None:
    plan = _plan()
    policy = CompletionPolicy()
    binding = binding_for_plan(plan, policy)

    no_evidence = evaluate_formal_goal_completion(
        "G12.S4",
        plan,
        _events(plan),
        (),
        policy=policy,
        binding=binding,
        plan_consistency="kernel_verified",
        evaluated_at=NOW,
    )
    missing_code = evaluate_formal_goal_completion(
        "G12.S4",
        plan,
        _events(plan),
        _evidence(binding, omit=CompletionEvidenceKind.CODE),
        policy=policy,
        binding=binding,
        plan_consistency="kernel_verified",
        evaluated_at=NOW,
    )

    assert no_evidence.conformance.conformant
    assert no_evidence.state is GoalState.PROVISIONALLY_COMPLETE
    assert {item.kind for item in no_evidence.evidence_result.checks} == set(
        CompletionEvidenceKind
    )
    assert all(not item.satisfied for item in no_evidence.evidence_result.checks)
    assert missing_code.state is GoalState.PROVISIONALLY_COMPLETE
    assert missing_code.reason_codes == ("code_evidence_missing",)


def test_all_configured_code_test_kernel_model_protocol_and_runtime_evidence_is_required() -> None:
    plan = _plan()
    policy = CompletionPolicy(
        max_age_seconds={kind.value: 600 for kind in CompletionEvidenceKind}
    )
    binding = binding_for_plan(
        plan,
        policy,
        ast_scope_ids=("ast:module",),
        premise_ids=("premise:authority",),
    )

    decision = evaluate_formal_goal_completion(
        "G12.S4",
        plan,
        _events(plan),
        _evidence(binding),
        policy=policy,
        binding=binding,
        evaluated_at=NOW,
    )

    assert decision.verified
    assert decision.closeable
    assert decision.state is GoalState.VERIFIED_COMPLETE
    assert decision.evidence_result.satisfied
    assert {item.kind.value for item in decision.evidence_result.checks} == {
        "code",
        "test",
        "kernel",
        "model_check",
        "protocol",
        "runtime",
    }


@pytest.mark.parametrize(
    ("field", "updated", "cause"),
    [
        ("policy_id", "policy:changed", InvalidationCause.POLICY_CHANGED),
        ("repository_tree_id", "tree:changed", InvalidationCause.REPOSITORY_TREE_CHANGED),
        ("ast_scope_ids", ("ast:changed",), InvalidationCause.AST_CHANGED),
        ("premise_ids", ("premise:changed",), InvalidationCause.PREMISE_CHANGED),
        (
            "counterexample_ids",
            ("counterexample:changed",),
            InvalidationCause.COUNTEREXAMPLE_CHANGED,
        ),
    ],
)
def test_semantic_input_change_invalidates_prior_conformance_and_reopens_goal(
    field: str, updated: object, cause: InvalidationCause
) -> None:
    plan = _plan()
    policy = CompletionPolicy()
    original = binding_for_plan(
        plan,
        policy,
        ast_scope_ids=("ast:one",),
        premise_ids=("premise:one",),
        counterexample_ids=("counterexample:one",),
    )
    prior = evaluate_plan_conformance(
        plan, _events(plan), policy=policy, binding=original
    )
    changed_values = {
        "plan_id": original.plan_id,
        "policy_id": original.policy_id,
        "repository_tree_id": original.repository_tree_id,
        "ast_scope_ids": original.ast_scope_ids,
        "premise_ids": original.premise_ids,
        "counterexample_ids": original.counterexample_ids,
    }
    changed_values[field] = updated
    changed = ConformanceBinding(**changed_values)

    # When policy ID changes, the evaluator derives the configured policy ID;
    # either direction is a policy mismatch and must invalidate the receipt.
    decision = evaluate_formal_goal_completion(
        "G12.S4",
        plan,
        _events(plan),
        _evidence(changed),
        policy=policy,
        binding=changed,
        previous_state=GoalState.VERIFIED_COMPLETE,
        prior_conformance=prior,
        evaluated_at=NOW,
    )

    assert decision.state is GoalState.REOPENED
    assert decision.conformance.verdict is ConformanceVerdict.INVALIDATED
    assert cause in decision.conformance.invalidation_causes
    assert cause.value in decision.reason_codes


def test_plan_change_invalidates_receipt_and_reopens_even_when_new_trace_matches() -> None:
    first_plan = _plan(revision="one")
    policy = CompletionPolicy()
    first_binding = binding_for_plan(first_plan, policy)
    prior = evaluate_plan_conformance(
        first_plan, _events(first_plan), policy=policy, binding=first_binding
    )
    second_plan = _plan(revision="two")
    second_binding = binding_for_plan(second_plan, policy)

    decision = evaluate_formal_goal_completion(
        "G12.S4",
        second_plan,
        _events(second_plan),
        _evidence(second_binding),
        policy=policy,
        binding=second_binding,
        previous_state=GoalState.VERIFIED_COMPLETE,
        prior_conformance=prior,
        evaluated_at=NOW,
    )

    assert decision.reopened
    assert InvalidationCause.PLAN_CHANGED in decision.conformance.invalidation_causes
    assert not decision.closeable


def test_stale_failed_unbound_and_wrong_binding_evidence_each_fail_closed() -> None:
    plan = _plan()
    policy = CompletionPolicy(max_age_seconds={"code": 60})
    binding = binding_for_plan(plan, policy)
    wrong_binding = replace(binding, ast_scope_ids=("ast:old",))
    base = list(_evidence(binding))
    replacements = {
        CompletionEvidenceKind.CODE: replace(
            next(item for item in base if item.kind is CompletionEvidenceKind.CODE),
            observed_at="2026-07-23T19:00:00Z",
            evidence_id="",
        ),
        CompletionEvidenceKind.TEST: replace(
            next(item for item in base if item.kind is CompletionEvidenceKind.TEST),
            verdict="failed",
            evidence_id="",
        ),
        CompletionEvidenceKind.KERNEL: replace(
            next(item for item in base if item.kind is CompletionEvidenceKind.KERNEL),
            artifact_id="",
            evidence_id="",
        ),
        CompletionEvidenceKind.RUNTIME: replace(
            next(item for item in base if item.kind is CompletionEvidenceKind.RUNTIME),
            binding=wrong_binding,
            evidence_id="",
        ),
    }
    supplied = tuple(replacements.get(item.kind, item) for item in base)

    decision = evaluate_formal_goal_completion(
        "G12.S4",
        plan,
        _events(plan),
        supplied,
        policy=policy,
        binding=binding,
        evaluated_at=NOW,
    )
    statuses = {item.kind: item.status.value for item in decision.evidence_result.checks}

    assert statuses[CompletionEvidenceKind.CODE] == "stale"
    assert statuses[CompletionEvidenceKind.TEST] == "failed"
    assert statuses[CompletionEvidenceKind.KERNEL] == "unbound"
    assert statuses[CompletionEvidenceKind.RUNTIME] == "invalidated"
    assert not decision.verified


def test_restart_and_replay_are_identical_from_json_and_duckdb(tmp_path: Path) -> None:
    pytest.importorskip("duckdb")
    plan = _plan()
    policy = CompletionPolicy(
        max_age_seconds={kind.value: 600 for kind in CompletionEvidenceKind}
    )
    binding = binding_for_plan(
        plan,
        policy,
        ast_scope_ids=("ast:module",),
        premise_ids=("premise:one",),
        counterexample_ids=("counterexample:one",),
    )
    packet = ConformanceReplayPacket(
        goal_id="G12.S4",
        plan=plan,
        events=_events(plan),
        evidence=_evidence(binding),
        policy=policy,
        binding=binding,
        previous_state=GoalState.PROVISIONALLY_COMPLETE,
        evaluated_at=NOW,
        plan_consistency="kernel_verified",
    ).with_decision()
    json_path = tmp_path / "conformance.json"
    duckdb_path = tmp_path / "conformance.duckdb"

    write_conformance_evidence(json_path, packet)
    write_conformance_evidence(duckdb_path, packet)
    json_packet = read_conformance_evidence(json_path)
    duckdb_packet = read_conformance_evidence(duckdb_path)
    json_decision = replay_conformance_evidence(json_path)
    duckdb_decision = replay_conformance_evidence(duckdb_path)

    assert json_packet.packet_id == duckdb_packet.packet_id == packet.packet_id
    assert json_packet.to_json() == duckdb_packet.to_json() == packet.to_json()
    assert (
        json_decision.decision_id
        == duckdb_decision.decision_id
        == packet.stored_decision.decision_id
    )
    assert json_decision.conformance.verdict is ConformanceVerdict.CONFORMANT
    assert json_decision.state is GoalState.VERIFIED_COMPLETE
