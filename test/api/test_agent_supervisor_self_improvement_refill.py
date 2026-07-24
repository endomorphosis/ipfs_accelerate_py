from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.backlog_refinery import (
    load_strategy,
    self_improvement_epoch_wait_active,
)
from ipfs_accelerate_py.agent_supervisor.goal_completion import (
    validate_completion_evidence,
)
from ipfs_accelerate_py.agent_supervisor.objective_graph import (
    ObjectiveWorkProposal,
    completion_evidence_source_decision,
)
from ipfs_accelerate_py.agent_supervisor.objective_tracker import (
    resolve_objective_evidence_projection,
)
from ipfs_accelerate_py.agent_supervisor.self_improvement import (
    DEFAULT_BENCHMARK_DIMENSIONS,
    EPOCH_IDEMPOTENCY_REQUIREMENT_ID,
    HEALTHY_EXHAUSTION_REQUIREMENT_ID,
    SUCCESSOR_REFILL_REQUIREMENT_ID,
    BenchmarkDisposition,
    BenchmarkObservation,
    EpochReplayEvidence,
    HealthyExhaustionEvidence,
    SelfImprovementEpochStatus,
    SelfImprovementPolicy,
    SuccessorRefillEvidence,
    run_self_improvement_epoch,
)


NOW = datetime(2026, 7, 24, 12, 0, tzinfo=timezone.utc)


def _digest(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _objective_heap() -> str:
    return f"""# Objective Heap

## ASI-G000 Root objective

- Status: active
- Evidence: root-proof

## ASI-G080 Benchmark-driven bounded self-refill

- Status: active
- Parent: ASI-G000
- Goal: Reconcile the program and record healthy exhaustion
- Evidence: {SUCCESSOR_REFILL_REQUIREMENT_ID}, {EPOCH_IDEMPOTENCY_REQUIREMENT_ID}, {HEALTHY_EXHAUSTION_REQUIREMENT_ID}

## ASI-G109 Prove bounded successor refill

- Status: active
- Parent: ASI-G080
- Goal: A drained actionable epoch creates bounded novel successors
- Evidence: {SUCCESSOR_REFILL_REQUIREMENT_ID}
- Refinement depth: 2

## ASI-G110 Prove epoch idempotency

- Status: active
- Parent: ASI-G080
- Goal: An identical self-improvement epoch is idempotent
- Evidence: {EPOCH_IDEMPOTENCY_REQUIREMENT_ID}
- Refinement depth: 2

## ASI-G111 Prove healthy exhaustion

- Status: active
- Parent: ASI-G080
- Goal: Prove a healthy epoch records exhaustion and creates no busywork
- Evidence: {HEALTHY_EXHAUSTION_REQUIREMENT_ID}
- Outputs: ipfs_accelerate_py/agent_supervisor/self_improvement.py
- Validation: python -m pytest test/api/test_agent_supervisor_self_improvement_refill.py -q
- Refinement depth: 2
"""


def _drained_board() -> str:
    return """# Tasks

## ASI-001 Completed bootstrap

- Status: completed
"""


def _paths(tmp_path: Path) -> dict[str, Path]:
    repo = tmp_path / "repo"
    repo.mkdir(parents=True)
    objective = repo / "objectives.md"
    todo = repo / "todo.md"
    objective.write_text(_objective_heap(), encoding="utf-8")
    todo.write_text(_drained_board(), encoding="utf-8")
    return {
        "repo_root": repo,
        "objective_path": objective,
        "todo_path": todo,
        "ledger_path": tmp_path / "state" / "epochs.json",
        "strategy_path": tmp_path / "state" / "strategy.json",
    }


def _observations(
    binding,
    *,
    disposition: BenchmarkDisposition = BenchmarkDisposition.HEALTHY,
    dimensions: tuple[str, ...] = DEFAULT_BENCHMARK_DIMENSIONS,
    one_channel: bool = False,
    observed_at: datetime = NOW - timedelta(minutes=1),
    fresh_until: datetime = NOW + timedelta(hours=1),
) -> tuple[BenchmarkObservation, ...]:
    result = []
    channels = (
        ("paired-benchmark-a",)
        if one_channel
        else ("paired-benchmark-a", "paired-benchmark-b")
    )
    for channel in channels:
        for dimension in dimensions:
            reasons = (
                (f"{dimension} regression exceeded its policy gate",)
                if disposition.actionable
                else ()
            )
            result.append(
                BenchmarkObservation(
                    dimension=dimension,
                    evidence_channel=channel,
                    producer_id=f"benchmark:{channel}:{dimension}",
                    repository_id=binding.repository_id,
                    repository_tree=binding.repository_tree,
                    policy_id=binding.policy_id,
                    capability_snapshot_id=binding.capability_snapshot_id,
                    command=f"python -m benchmark --dimension {dimension}",
                    toolchain="pytest+benchmark-harness/v1",
                    scope=(f"fixture:{dimension}", "paired:baseline-candidate"),
                    result={"gate": "passed", "sample_count": 3},
                    artifact_digest=_digest(
                        f"artifact:{channel}:{dimension}"
                    ),
                    disposition=disposition,
                    actionable_reasons=reasons,
                    observed_at=observed_at,
                    fresh_until=fresh_until,
                )
            )
    return tuple(result)


def _successor_proposal(
    *,
    title: str = "Repair measured cache regression",
    evidence: str = "successor-runtime-proof-unique",
    source_id: str = "benchmark-gap:cache",
) -> ObjectiveWorkProposal:
    return ObjectiveWorkProposal(
        kind="subgoal",
        title=title,
        parent_goal_id="ASI-G080",
        parent_objective_terms=("repair measured benchmark regression",),
        expected_evidence_delta=(evidence,),
        dependencies=(),
        predicted_files=("src/successor.py",),
        predicted_symbols=("repair_successor",),
        validation_commands=("python -m pytest tests/test_successor.py -q",),
        confidence=0.95,
        estimated_cost=1.0,
        novelty=0.95,
        depth=2,
        estimated_tokens=200,
        source="self-improvement-benchmark",
        source_id=source_id,
    )


def test_healthy_no_gap_epoch_proves_requirement_and_creates_no_busywork(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    objective_before = paths["objective_path"].read_bytes()
    todo_before = paths["todo_path"].read_bytes()
    provider_calls = []

    def provider(binding):
        provider_calls.append(binding.epoch_id)
        return _observations(binding)

    run = run_self_improvement_epoch(
        **paths,
        observation_provider=provider,
        capability_snapshot_id="capabilities:healthy-v1",
        observation_window="2026-07-24T12:00:00Z/PT1H",
        observed_at=NOW,
    )

    assert run.status is SelfImprovementEpochStatus.HEALTHY_EXHAUSTED
    assert run.proved_requirement_ids == (HEALTHY_EXHAUSTION_REQUIREMENT_ID,)
    assert not run.replayed
    assert len(provider_calls) == 1
    assert paths["objective_path"].read_bytes() == objective_before
    assert paths["todo_path"].read_bytes() == todo_before
    evidence = run.evidence
    assert evidence is not None
    assert evidence.goal_projection.goal_id == "ASI-G111"
    assert evidence.goal_projection.parent_goal_id == "ASI-G080"
    assert evidence.exhaustion_quorum.satisfied
    assert evidence.exhaustion_quorum.count == 2
    assert {
        member.evidence_channel
        for member in evidence.exhaustion_quorum.members
    } == {"paired-benchmark-a", "paired-benchmark-b"}
    assert evidence.classified_gap_count == 0
    assert evidence.candidate_count == 0
    assert evidence.admitted_count == 0
    assert evidence.materialized_count == 0
    assert evidence.taskboard_write_count == 0

    strategy = load_strategy(paths["strategy_path"])
    assert self_improvement_epoch_wait_active(
        strategy, epoch_id=run.epoch_id
    )
    assert strategy["last_self_improvement_exhaustion_evidence_id"] == evidence.evidence_id
    assert strategy["self_improvement_refill_state"] == "waiting_for_meaningful_trigger"

    source_decision = completion_evidence_source_decision(
        evidence.to_dict(),
        requirement=HEALTHY_EXHAUSTION_REQUIREMENT_ID,
        repository_tree=evidence.binding.repository_tree,
        policy_id=evidence.binding.policy_id,
    )
    assert source_decision.satisfies, source_decision.reason_codes
    completion = evidence.completion_evidence()
    validation = validate_completion_evidence(
        completion,
        repository_id=evidence.binding.repository_id,
        repository_tree=evidence.binding.repository_tree,
        now=NOW,
    )
    assert validation.valid, validation.reason_codes


def test_identical_epoch_replays_before_benchmark_and_repairs_wait_projection(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    calls = 0

    def provider(binding):
        nonlocal calls
        calls += 1
        return _observations(binding)

    first = run_self_improvement_epoch(
        **paths,
        observation_provider=provider,
        capability_snapshot_id="capabilities:healthy-v1",
        observation_window="window:1",
        observed_at=NOW,
    )
    corrupt_strategy = load_strategy(paths["strategy_path"])
    corrupt_strategy["last_self_improvement_exhaustion_evidence_id"] = (
        "sha256:" + "0" * 64
    )
    paths["strategy_path"].write_text(
        json.dumps(corrupt_strategy), encoding="utf-8"
    )
    second = run_self_improvement_epoch(
        **paths,
        observation_provider=provider,
        capability_snapshot_id="capabilities:healthy-v1",
        observation_window="window:1",
        observed_at=NOW + timedelta(minutes=5),
    )

    assert calls == 1
    assert second.replayed
    assert second.receipt.receipt_id == first.receipt.receipt_id
    assert second.evidence is not None
    assert (
        load_strategy(paths["strategy_path"])[
            "last_self_improvement_exhaustion_evidence_id"
        ]
        == second.evidence.evidence_id
    )
    assert self_improvement_epoch_wait_active(
        load_strategy(paths["strategy_path"]), epoch_id=first.epoch_id
    )


@pytest.mark.parametrize(
    ("mutation", "expected_status", "expected_blocker"),
    [
        ("one_channel", SelfImprovementEpochStatus.INELIGIBLE, "exhaustion_quorum_unsatisfied"),
        ("missing_dimension", SelfImprovementEpochStatus.INELIGIBLE, "benchmark_population_incomplete"),
        ("stale", SelfImprovementEpochStatus.INELIGIBLE, "benchmark_not_fresh_and_complete"),
        ("foreign_policy", SelfImprovementEpochStatus.INELIGIBLE, "benchmark_binding_mismatch"),
        ("actionable", SelfImprovementEpochStatus.ACTIONABLE, None),
    ],
)
def test_nonqualifying_epochs_never_claim_healthy_exhaustion(
    tmp_path: Path,
    mutation: str,
    expected_status: SelfImprovementEpochStatus,
    expected_blocker: str | None,
) -> None:
    paths = _paths(tmp_path)

    def provider(binding):
        if mutation == "one_channel":
            return _observations(binding, one_channel=True)
        if mutation == "missing_dimension":
            return _observations(binding, dimensions=DEFAULT_BENCHMARK_DIMENSIONS[:-1])
        if mutation == "stale":
            return _observations(
                binding,
                fresh_until=NOW - timedelta(seconds=1),
            )
        if mutation == "foreign_policy":
            records = list(_observations(binding))
            payload = records[0].to_dict()
            payload["policy_id"] = "foreign-policy"
            payload["receipt_id"] = ""
            records[0] = BenchmarkObservation(
                dimension=payload["dimension"],
                evidence_channel=payload["evidence_channel"],
                producer_id=payload["producer_id"],
                repository_id=payload["repository_id"],
                repository_tree=payload["repository_tree"],
                policy_id=payload["policy_id"],
                capability_snapshot_id=payload["capability_snapshot_id"],
                command=payload["command"],
                toolchain=payload["toolchain"],
                scope=tuple(payload["scope"]),
                result=payload["result"],
                artifact_digest=payload["artifact_digest"],
                disposition=payload["disposition"],
                actionable_reasons=tuple(payload["actionable_reasons"]),
                observed_at=payload["observed_at"],
                fresh_until=payload["fresh_until"],
                complete=payload["complete"],
            )
            return records
        return _observations(binding, disposition=BenchmarkDisposition.REGRESSION)

    run = run_self_improvement_epoch(
        **paths,
        observation_provider=provider,
        capability_snapshot_id="capabilities:healthy-v1",
        observation_window=f"window:{mutation}",
        observed_at=NOW,
    )

    assert run.status is expected_status
    assert run.evidence is None
    assert not run.proved_requirement_ids
    assert paths["objective_path"].read_text(encoding="utf-8") == _objective_heap()
    assert paths["todo_path"].read_text(encoding="utf-8") == _drained_board()
    if expected_blocker:
        assert expected_blocker in run.receipt.blocker_codes
    assert not self_improvement_epoch_wait_active(
        load_strategy(paths["strategy_path"]), epoch_id=run.epoch_id
    )


def test_provider_artifact_mutation_fails_closed_without_claim_or_followup(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)

    def provider(binding):
        records = _observations(binding)
        paths["todo_path"].write_text(
            _drained_board() + "\n<!-- unexpected write -->\n",
            encoding="utf-8",
        )
        return records

    run = run_self_improvement_epoch(
        **paths,
        observation_provider=provider,
        capability_snapshot_id="capabilities:healthy-v1",
        observation_window="window:mutation",
        observed_at=NOW,
    )

    assert run.status is SelfImprovementEpochStatus.INELIGIBLE
    assert "taskboard_mutated_during_epoch" in run.receipt.blocker_codes
    assert run.evidence is None
    assert not run.proved_requirement_ids


def test_same_byte_taskboard_rewrite_cannot_claim_zero_writes(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)

    def provider(binding):
        records = _observations(binding)
        paths["todo_path"].write_bytes(paths["todo_path"].read_bytes())
        return records

    run = run_self_improvement_epoch(
        **paths,
        observation_provider=provider,
        capability_snapshot_id="capabilities:healthy-v1",
        observation_window="window:same-byte-write",
        observed_at=NOW,
    )

    assert run.status is SelfImprovementEpochStatus.INELIGIBLE
    assert "taskboard_written_during_epoch" in run.receipt.blocker_codes
    assert run.evidence is None
    assert not run.proved_requirement_ids


def test_changed_meaningful_trigger_creates_a_new_epoch_not_busywork(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    calls = 0

    def provider(binding):
        nonlocal calls
        calls += 1
        return _observations(binding)

    first = run_self_improvement_epoch(
        **paths,
        observation_provider=provider,
        capability_snapshot_id="capabilities:v1",
        observation_window="window:1",
        observed_at=NOW,
    )
    second = run_self_improvement_epoch(
        **paths,
        observation_provider=provider,
        capability_snapshot_id="capabilities:v2",
        observation_window="window:1",
        observed_at=NOW,
    )

    assert calls == 2
    assert first.epoch_id != second.epoch_id
    assert first.status is second.status is SelfImprovementEpochStatus.HEALTHY_EXHAUSTED
    assert paths["todo_path"].read_text(encoding="utf-8") == _drained_board()


def test_restoration_rejects_tampering_unknown_fields_and_detached_identity(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)

    first = run_self_improvement_epoch(
        **paths,
        observation_provider=lambda binding: _observations(binding),
        capability_snapshot_id="capabilities:v1",
        observation_window="window:1",
        observed_at=NOW,
    )
    evidence_payload = first.evidence.to_dict()  # type: ignore[union-attr]
    restored = HealthyExhaustionEvidence.from_dict(evidence_payload)
    assert restored.evidence_id == first.evidence.evidence_id  # type: ignore[union-attr]

    unknown = copy.deepcopy(evidence_payload)
    unknown["unreviewed_authority"] = True
    with pytest.raises(ValueError, match="unknown fields"):
        HealthyExhaustionEvidence.from_dict(unknown)

    nested_unknown = copy.deepcopy(evidence_payload)
    nested_unknown["goal_projection"]["unreviewed_owner"] = "ASI-G099"
    with pytest.raises(ValueError, match="unknown fields"):
        HealthyExhaustionEvidence.from_dict(nested_unknown)

    projected_status = copy.deepcopy(evidence_payload)
    projected_status["validation_passed"] = False
    with pytest.raises(ValueError, match="projection does not match"):
        HealthyExhaustionEvidence.from_dict(projected_status)

    altered = copy.deepcopy(evidence_payload)
    altered["candidate_count"] = 1
    with pytest.raises(ValueError, match="candidate_count"):
        HealthyExhaustionEvidence.from_dict(altered)

    ledger = json.loads(paths["ledger_path"].read_text(encoding="utf-8"))
    ledger["epochs"][first.epoch_id]["evidence"]["objective_after_id"] = "tampered"
    paths["ledger_path"].write_text(
        json.dumps(ledger, sort_keys=True), encoding="utf-8"
    )
    with pytest.raises(ValueError):
        run_self_improvement_epoch(
            **paths,
            observation_provider=lambda _binding: pytest.fail(
                "tampered epoch must fail before benchmark execution"
            ),
            capability_snapshot_id="capabilities:v1",
            observation_window="window:1",
            observed_at=NOW,
        )


def test_stale_aggregate_goal_projects_to_existing_g111_without_refinement() -> None:
    projection = resolve_objective_evidence_projection(
        _objective_heap(),
        requirement_id=HEALTHY_EXHAUSTION_REQUIREMENT_ID,
        expected_goal_id="ASI-G111",
        expected_parent_goal_id="ASI-G080",
    )

    assert projection.goal_id == "ASI-G111"
    assert projection.parent_goal_id == "ASI-G080"
    with pytest.raises(ValueError, match="expected ASI-G099"):
        resolve_objective_evidence_projection(
            _objective_heap(),
            requirement_id=HEALTHY_EXHAUSTION_REQUIREMENT_ID,
            expected_goal_id="ASI-G099",
            expected_parent_goal_id="ASI-G080",
        )


def test_parent_packet_evidence_projects_to_each_unique_leaf_owner() -> None:
    expected = {
        SUCCESSOR_REFILL_REQUIREMENT_ID: "ASI-G109",
        EPOCH_IDEMPOTENCY_REQUIREMENT_ID: "ASI-G110",
        HEALTHY_EXHAUSTION_REQUIREMENT_ID: "ASI-G111",
    }
    for requirement_id, goal_id in expected.items():
        projection = resolve_objective_evidence_projection(
            _objective_heap(),
            requirement_id=requirement_id,
            expected_goal_id=goal_id,
            expected_parent_goal_id="ASI-G080",
        )
        assert projection.goal_id == goal_id

    ambiguous = _objective_heap() + f"""

## ASI-G199 Ambiguous sibling owner

- Status: active
- Parent: ASI-G080
- Evidence: {SUCCESSOR_REFILL_REQUIREMENT_ID}
"""
    with pytest.raises(ValueError, match="ambiguous owners"):
        resolve_objective_evidence_projection(
            ambiguous,
            requirement_id=SUCCESSOR_REFILL_REQUIREMENT_ID,
            expected_goal_id="ASI-G109",
            expected_parent_goal_id="ASI-G080",
        )


def test_actionable_epoch_creates_bounded_successor_and_exact_replay_is_noop(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    provider_calls = 0
    proposal_calls = 0

    def provider(binding):
        nonlocal provider_calls
        provider_calls += 1
        return _observations(binding, disposition=BenchmarkDisposition.REGRESSION)

    def proposals(_binding, _observations):
        nonlocal proposal_calls
        proposal_calls += 1
        return (_successor_proposal(),)

    kwargs = {
        **paths,
        "observation_provider": provider,
        "proposal_provider": proposals,
        "capability_snapshot_id": "capabilities:actionable-v1",
        "observation_window": "window:actionable-1",
        "observed_at": NOW,
        "materialization_journal_path": tmp_path / "state" / "materialization.json",
        "discovery_dir": tmp_path / "discovery",
        "bundle_dir": tmp_path / "bundles",
    }
    first = run_self_improvement_epoch(**kwargs)

    assert first.status is SelfImprovementEpochStatus.SUCCESSORS_CREATED
    assert first.proved_requirement_ids == (SUCCESSOR_REFILL_REQUIREMENT_ID,)
    assert first.receipt.successor_evidence is not None
    restored_successor = SuccessorRefillEvidence.from_dict(
        first.receipt.successor_evidence.to_dict()
    )
    successor_validation = validate_completion_evidence(
        restored_successor.completion_evidence(),
        repository_id=restored_successor.binding.repository_id,
        repository_tree=restored_successor.binding.repository_tree,
        now=restored_successor.observed_at,
    )
    assert successor_validation.valid, successor_validation.reason_codes
    assert len(first.receipt.created_goal_ids) == 1
    assert len(first.receipt.created_task_ids) == 1
    goal_id = first.receipt.created_goal_ids[0]
    task_id = first.receipt.created_task_ids[0]
    assert paths["objective_path"].read_text(encoding="utf-8").count(
        f"## {goal_id} "
    ) == 1
    assert paths["todo_path"].read_text(encoding="utf-8").count(
        f"## {task_id} "
    ) == 1
    objective_after = paths["objective_path"].read_bytes()
    taskboard_after = paths["todo_path"].read_bytes()

    second = run_self_improvement_epoch(**{**kwargs, "observed_at": NOW + timedelta(minutes=1)})

    assert second.replayed
    assert second.receipt.receipt_id == first.receipt.receipt_id
    assert EPOCH_IDEMPOTENCY_REQUIREMENT_ID in second.proved_requirement_ids
    assert second.replay_evidence is not None
    restored_replay = EpochReplayEvidence.from_dict(
        second.replay_evidence.to_dict()
    )
    assert restored_replay.evidence_id == second.replay_evidence.evidence_id
    assert validate_completion_evidence(
        restored_replay.completion_evidence(),
        repository_id=restored_replay.binding.repository_id,
        repository_tree=restored_replay.binding.repository_tree,
        now=restored_replay.replayed_at,
    ).valid
    assert provider_calls == 1
    assert proposal_calls == 1
    assert paths["objective_path"].read_bytes() == objective_after
    assert paths["todo_path"].read_bytes() == taskboard_after


def test_successor_policy_bounds_batch_and_foreign_actionable_fails_closed(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    proposal_calls = 0

    def bounded_proposals(_binding, _observations):
        nonlocal proposal_calls
        proposal_calls += 1
        return (
            _successor_proposal(),
            _successor_proposal(
                title="Repair measured planning regression",
                evidence="successor-planning-proof-unique",
                source_id="benchmark-gap:planning",
            ),
        )

    bounded = run_self_improvement_epoch(
        **paths,
        observation_provider=lambda binding: _observations(
            binding, disposition=BenchmarkDisposition.REGRESSION
        ),
        proposal_provider=bounded_proposals,
        capability_snapshot_id="capabilities:bounded-v1",
        observation_window="window:bounded",
        policy=SelfImprovementPolicy(max_new_successor_goals=1),
        observed_at=NOW,
        materialization_journal_path=tmp_path / "state" / "materialization.json",
        discovery_dir=tmp_path / "discovery",
        bundle_dir=tmp_path / "bundles",
    )

    assert bounded.status is SelfImprovementEpochStatus.SUCCESSORS_CREATED
    assert len(bounded.receipt.created_goal_ids) == 1
    assert (
        len(
            bounded.receipt.successor_evidence.candidate_proposal_ids  # type: ignore[union-attr]
        )
        == 2
    )
    assert proposal_calls == 1

    foreign_paths = _paths(tmp_path / "foreign")
    foreign_proposal_calls = 0

    def foreign_provider(binding):
        records = list(
            _observations(binding, disposition=BenchmarkDisposition.REGRESSION)
        )
        payload = records[0].to_dict()
        payload["policy_id"] = "foreign-policy"
        payload["receipt_id"] = ""
        records[0] = BenchmarkObservation.from_dict(payload)
        return records

    def forbidden_proposals(_binding, _observations):
        nonlocal foreign_proposal_calls
        foreign_proposal_calls += 1
        return (_successor_proposal(),)

    foreign = run_self_improvement_epoch(
        **foreign_paths,
        observation_provider=foreign_provider,
        proposal_provider=forbidden_proposals,
        capability_snapshot_id="capabilities:foreign-v1",
        observation_window="window:foreign",
        observed_at=NOW,
    )
    assert foreign.status is SelfImprovementEpochStatus.INELIGIBLE
    assert "benchmark_binding_mismatch" in foreign.receipt.blocker_codes
    assert foreign_proposal_calls == 0
    assert not foreign.receipt.created_goal_ids
