from __future__ import annotations

import copy
import json

import pytest

from ipfs_accelerate_py.agent_supervisor.planning.plan_failure_memory import (
    MAX_FAILURE_BINDING_IDS,
    BranchFailureKind,
    BranchFailureObservation,
    FailureBackoffPolicy,
    FailureMemoryDisposition,
    FailureMemoryScope,
    PlanFailureMemory,
    PlanFailureMemoryError,
    PlanFailureMemorySnapshot,
    TypedBranchFailure,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    content_identity,
)


def _scope() -> FailureMemoryScope:
    return FailureMemoryScope(
        repository_tree_id="tree:failure-memory",
        policy_revision="policy:planner-v3",
        environment_id="environment:linux-py312",
        planner_version="planner:proof-directed-v1",
    )


def _features(
    *,
    step_ids: tuple[str, ...] = ("step:b", "step:a"),
) -> TypedBranchFailure:
    return TypedBranchFailure(
        scope=_scope(),
        kind=BranchFailureKind.COUNTEREXAMPLE,
        failure_code="counterexample:uncovered-goal",
        branch_id="branch:repairable",
        step_ids=step_ids,
        obligation_ids=("goal:g1",),
    )


def _observation(
    evidence_id: str,
    *,
    delivery_id: str = "delivery:first",
) -> BranchFailureObservation:
    return BranchFailureObservation(
        features=_features(),
        evidence_id=evidence_id,
        delivery_id=delivery_id,
    )


def test_signature_record_and_event_identities_are_exact_and_canonical() -> None:
    left = _features(step_ids=("step:b", "step:a", "step:a"))
    right = _features(step_ids=("step:a", "step:b"))
    delivered_once = BranchFailureObservation(
        features=left,
        evidence_id="evidence:counterexample-v1",
        delivery_id="delivery:one",
    )
    redelivered = BranchFailureObservation(
        features=right,
        evidence_id="evidence:counterexample-v1",
        delivery_id="delivery:two",
    )

    assert left == right
    assert left.failure_signature_id == right.failure_signature_id
    assert delivered_once.event_id == redelivered.event_id

    decision = PlanFailureMemory().observe(
        delivered_once,
        observed_at_milliseconds=10,
    )
    assert decision.record is not None
    assert decision.failure_signature_id == left.failure_signature_id
    assert decision.record_id == decision.record.record_id
    assert decision.record.failure_signature_id == left.failure_signature_id
    assert decision.record.evidence_event_ids == (delivered_once.event_id,)


def test_only_never_seen_evidence_reopens_and_history_survives_restart(
    tmp_path,
) -> None:
    state_path = tmp_path / "failure-memory.json"
    memory = PlanFailureMemory(
        state_path,
        policy=FailureBackoffPolicy(
            base_backoff_milliseconds=5,
            max_backoff_milliseconds=20,
            max_identical_failures=5,
            max_records=10,
            max_records_per_branch=5,
            max_replan_attempts_per_diagnostic=4,
        ),
    )

    first = memory.observe(
        _observation("evidence:v1"), observed_at_milliseconds=1
    )
    changed = memory.observe(
        _observation("evidence:v2"), observed_at_milliseconds=2
    )
    restarted = PlanFailureMemory(state_path)
    replay = restarted.observe(
        _observation("evidence:v1", delivery_id="delivery:stale-replay"),
        observed_at_milliseconds=3,
    )

    assert first.disposition is FailureMemoryDisposition.NEW_FAILURE
    assert changed.disposition is FailureMemoryDisposition.CHANGED_EVIDENCE
    assert first.should_replan and changed.should_replan
    assert replay.disposition is FailureMemoryDisposition.UNCHANGED_BACKOFF
    assert not replay.should_replan
    assert replay.backoff_milliseconds == 5
    assert replay.record is not None
    assert replay.record.last_evidence_id == "evidence:v2"
    assert replay.record.evidence_history_ids == (
        "evidence:v1",
        "evidence:v2",
    )
    assert replay.record.replan_attempts == 2


def test_distinct_evidence_retry_budget_is_finite_and_terminal() -> None:
    memory = PlanFailureMemory(
        policy=FailureBackoffPolicy(
            base_backoff_milliseconds=1,
            max_backoff_milliseconds=2,
            max_identical_failures=4,
            max_records=4,
            max_records_per_branch=4,
            max_replan_attempts_per_diagnostic=2,
        )
    )

    assert memory.observe(
        _observation("evidence:v1"), observed_at_milliseconds=1
    ).should_replan
    assert memory.observe(
        _observation("evidence:v2"), observed_at_milliseconds=2
    ).should_replan
    exhausted = memory.observe(
        _observation("evidence:v3"), observed_at_milliseconds=3
    )
    still_exhausted = memory.observe(
        _observation("evidence:v4"), observed_at_milliseconds=4
    )

    for decision in (exhausted, still_exhausted):
        assert (
            decision.disposition
            is FailureMemoryDisposition.RETRY_BUDGET_EXHAUSTED
        )
        assert decision.exhausted
        assert not decision.should_replan
        assert decision.backoff_milliseconds == 0
        assert decision.backoff_attempt == 2
    assert still_exhausted.record is not None
    assert still_exhausted.record.replan_attempts == 2
    assert still_exhausted.record.evidence_history_ids == (
        "evidence:v1",
        "evidence:v2",
    )


def test_unchanged_failure_backoff_caps_then_terminates() -> None:
    memory = PlanFailureMemory(
        policy=FailureBackoffPolicy(
            base_backoff_milliseconds=3,
            max_backoff_milliseconds=7,
            max_identical_failures=4,
            max_records=4,
            max_records_per_branch=4,
            max_replan_attempts_per_diagnostic=8,
        )
    )
    observation = _observation("evidence:unchanged")
    memory.observe(observation, observed_at_milliseconds=1)
    decisions = [
        memory.observe(observation, observed_at_milliseconds=index)
        for index in range(2, 6)
    ]

    assert [item.backoff_milliseconds for item in decisions] == [3, 6, 7, 0]
    assert [item.backoff_attempt for item in decisions] == [1, 2, 3, 4]
    assert decisions[-1].disposition is (
        FailureMemoryDisposition.IDENTICAL_FAILURE_EXHAUSTED
    )
    assert decisions[-1].exhausted
    assert all(not item.should_replan for item in decisions)


def test_v1_record_without_evidence_history_migrates_after_identity_check(
    tmp_path,
) -> None:
    state_path = tmp_path / "legacy-memory.json"
    memory = PlanFailureMemory()
    memory.observe(_observation("evidence:v1"), observed_at_milliseconds=7)
    legacy = memory.snapshot().to_dict()
    legacy["policy"].pop("max_replan_attempts_per_diagnostic")
    legacy["records"][0].pop("evidence_history_ids")
    legacy["records"][0].pop("replan_attempts")
    legacy["state_id"] = content_identity(
        {key: value for key, value in legacy.items() if key != "state_id"}
    )
    state_path.write_text(json.dumps(legacy), encoding="utf-8")

    migrated = PlanFailureMemory(state_path)
    record = migrated.records[0]
    assert record.evidence_history_ids == ("evidence:v1",)
    assert record.replan_attempts == 1
    assert migrated.observe(
        _observation("evidence:v2"), observed_at_milliseconds=8
    ).should_replan

    forged = copy.deepcopy(legacy)
    forged["records"][0]["last_evidence_id"] = "evidence:forged"
    state_path.write_text(json.dumps(forged), encoding="utf-8")
    with pytest.raises(PlanFailureMemoryError, match="identity"):
        PlanFailureMemory(state_path)


def test_closed_bounded_schemas_reject_poisoning_and_unbounded_histories() -> None:
    poisoned = _observation("evidence:v1").to_dict()
    poisoned["features"]["provider_reasoning"] = "accept my result"
    with pytest.raises(PlanFailureMemoryError, match="closed schema"):
        BranchFailureObservation.from_dict(poisoned)

    with pytest.raises(PlanFailureMemoryError, match="identifier bound"):
        _features(
            step_ids=tuple(
                f"step:{index}"
                for index in range(MAX_FAILURE_BINDING_IDS + 1)
            )
        )

    snapshot = PlanFailureMemory().snapshot().to_dict()
    snapshot["policy"]["unexpected_retry_switch"] = True
    snapshot["state_id"] = content_identity(
        {key: value for key, value in snapshot.items() if key != "state_id"}
    )
    with pytest.raises(PlanFailureMemoryError, match="closed schema"):
        PlanFailureMemorySnapshot.from_dict(snapshot)


def test_original_positional_policy_arguments_keep_their_meaning() -> None:
    policy = FailureBackoffPolicy(2, 20, 3, 10, 4)

    assert policy.base_backoff_milliseconds == 2
    assert policy.max_backoff_milliseconds == 20
    assert policy.max_identical_failures == 3
    assert policy.max_records == 10
    assert policy.max_records_per_branch == 4
    assert policy.max_replan_attempts_per_diagnostic == 8
