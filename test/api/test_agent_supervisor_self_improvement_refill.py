from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.objectives import backlog_refinery as backlog_refinery_module
from ipfs_accelerate_py.agent_supervisor import objective_tracker as objective_tracker_module
from ipfs_accelerate_py.agent_supervisor.objectives.backlog_refinery import (
    SELF_IMPROVEMENT_SUCCESSOR_RECORDS_KEY,
    align_completion_gate_force_goal_ids,
    filter_self_improvement_successor_candidates,
    load_strategy,
    record_configured_objective_backlog_findings,
    record_self_improvement_successor_admission,
    self_improvement_epoch_wait_active,
)
from ipfs_accelerate_py.agent_supervisor.objectives.goal_completion import (
    CompletionEvidence,
    GoalState,
    validate_completion_evidence,
)
from ipfs_accelerate_py.agent_supervisor.objectives.objective_graph import (
    EvidenceMatchKind,
    EvidenceSourcePolicy,
    ObjectiveWorkProposal,
    completion_evidence_source_decision,
)
from ipfs_accelerate_py.agent_supervisor.objectives.objective_tracker import (
    SelfImprovementGoalEvidenceReconciliation,
    reconcile_self_improvement_goal_evidence,
    resolve_objective_evidence_projection,
)
from ipfs_accelerate_py.agent_supervisor.objectives.scan_receipts import (
    RepositoryTreeIdentity,
)
from ipfs_accelerate_py.agent_supervisor.self_improvement.self_improvement import (
    DEFAULT_BENCHMARK_DIMENSIONS,
    EPOCH_IDEMPOTENCY_REQUIREMENT_ID,
    HEALTHY_EXHAUSTION_REQUIREMENT_ID,
    SELF_IMPROVEMENT_ACCEPTANCE_CRITERIA,
    SELF_IMPROVEMENT_CHILD_GOAL_IDS,
    SELF_IMPROVEMENT_COMPLETION_ANALYZER_VERSION,
    SELF_IMPROVEMENT_COMPLETION_CONFIGURATION_REVISION,
    SELF_IMPROVEMENT_OBJECTIVE_REVISION,
    SELF_IMPROVEMENT_PRODUCING_TASK_IDS,
    SELF_IMPROVEMENT_REQUIRED_EXHAUSTIVE_RECEIPTS,
    SUCCESSOR_REFILL_REQUIREMENT_ID,
    BenchmarkDisposition,
    BenchmarkObservation,
    EpochReplayEvidence,
    HealthyExhaustionEvidence,
    SelfImprovementEpochStatus,
    SelfImprovementPolicy,
    SuccessorRefillEvidence,
    evaluate_self_improvement_completion,
    run_self_improvement_epoch,
)


NOW = datetime(2026, 7, 24, 12, 0, tzinfo=timezone.utc)
COMPLETION_REPOSITORY_ID = "repository:ipfs-accelerate-py"
COMPLETION_REPOSITORY_TREE = "tree:sha256:asi-087-current"


def _completion_binding() -> dict[str, str]:
    return {
        "repository_id": COMPLETION_REPOSITORY_ID,
        "tree_id": COMPLETION_REPOSITORY_TREE,
        "objective_id": "ASI-G080",
        "objective_revision": SELF_IMPROVEMENT_OBJECTIVE_REVISION,
        "analyzer_version": SELF_IMPROVEMENT_COMPLETION_ANALYZER_VERSION,
        "configuration_revision": (
            SELF_IMPROVEMENT_COMPLETION_CONFIGURATION_REVISION
        ),
    }


def _completion_packet() -> dict[str, object]:
    validation_command = (
        "python -m pytest "
        "test/api/test_agent_supervisor_self_improvement_refill.py -q"
    )
    evidence = tuple(
        CompletionEvidence(
            acceptance_criterion=criterion,
            producing_task_or_scan="ASI-087",
            producer_kind="task",
            validation_receipt={
                "status": "passed",
                "tree_id": COMPLETION_REPOSITORY_TREE,
                "command": validation_command,
            },
            validation_passed=True,
            repository_id=COMPLETION_REPOSITORY_ID,
            repository_tree=COMPLETION_REPOSITORY_TREE,
            freshness={"fresh": True},
            observed_at=NOW - timedelta(minutes=2),
            provenance_cid=f"validation:asi-087:{index}",
        )
        for index, criterion in enumerate(
            SELF_IMPROVEMENT_ACCEPTANCE_CRITERIA,
            start=1,
        )
    )
    coverage = {
        "verified": True,
        "repository_id": COMPLETION_REPOSITORY_ID,
        "repository_tree": COMPLETION_REPOSITORY_TREE,
        "evaluated_at": (NOW - timedelta(minutes=1)).isoformat(),
        "criteria": [
            {
                "criterion": criterion,
                "status": "verified",
                "verified": True,
                "implementation": (
                    "ipfs_accelerate_py/agent_supervisor/"
                    + (
                        "backlog_refinery.py"
                        if index == 3
                        else "self_improvement.py"
                    )
                ),
                "validation": validation_command,
                "validation_receipt_ids": [
                    f"validation:asi-087:{index}"
                ],
            }
            for index, criterion in enumerate(
                SELF_IMPROVEMENT_ACCEPTANCE_CRITERIA,
                start=1,
            )
        ],
    }
    children = [
        {
            "goal_id": goal_id,
            "state": "verified_complete",
            "verified": True,
            "completion_gate": {
                "passed": True,
                "evaluated_evidence": {
                    "repository_id": COMPLETION_REPOSITORY_ID,
                    "repository_tree": COMPLETION_REPOSITORY_TREE,
                    "evaluated_at": (
                        NOW - timedelta(minutes=3)
                    ).isoformat(),
                    "validation_evidence": [
                        {
                            "valid": True,
                            "evidence": {
                                "repository_id": COMPLETION_REPOSITORY_ID,
                                "repository_tree": COMPLETION_REPOSITORY_TREE,
                            },
                        }
                    ],
                },
            },
            "proof_requirements": [
                {
                    "repository_tree": COMPLETION_REPOSITORY_TREE,
                    "provenance_id": f"proof:{goal_id}",
                    "required_assurance": "solver_checked",
                    "authoritative_assurance": "solver_checked",
                    "assurance_satisfied": True,
                    "contradicted": False,
                    "proof_verdict": "proved",
                    "freshness": "current",
                    "reason_codes": [],
                }
            ],
        }
        for goal_id in SELF_IMPROVEMENT_CHILD_GOAL_IDS
    ]
    binding = _completion_binding()
    members = [
        {
            "member_id": "asi-087-benchmark",
            "evidence_channel": "paired-benchmark",
            "receipt_cid": "scan:asi-087:benchmark",
            "binding": dict(binding),
            "scan_mode": "exhaustive",
            "producer_id": "asi-087-benchmark-producer",
            "implementation": "paired-benchmark-runner",
            "child_receipt_binding": COMPLETION_REPOSITORY_TREE,
            "child_receipt_sha256": _digest("asi-087-benchmark"),
            "aggregate_tree_binding": COMPLETION_REPOSITORY_TREE,
            "passed": True,
            "healthy": True,
            "safe_for_completion_reasoning": True,
            "exhaustive": True,
            "conclusive": True,
            "contradicted": False,
            "finished_at": (NOW - timedelta(minutes=4)).isoformat(),
        },
        {
            "member_id": "asi-087-independent-audit",
            "evidence_channel": "completion-audit",
            "receipt_cid": "scan:asi-087:audit",
            "binding": dict(binding),
            "scan_mode": "exhaustive",
            "producer_id": "asi-087-audit-producer",
            "implementation": "independent-completion-auditor",
            "child_receipt_binding": COMPLETION_REPOSITORY_TREE,
            "child_receipt_sha256": _digest("asi-087-audit"),
            "aggregate_tree_binding": COMPLETION_REPOSITORY_TREE,
            "passed": True,
            "healthy": True,
            "safe_for_completion_reasoning": True,
            "exhaustive": True,
            "conclusive": True,
            "contradicted": False,
            "finished_at": (NOW - timedelta(minutes=3)).isoformat(),
        },
    ]
    return {
        "repository_id": COMPLETION_REPOSITORY_ID,
        "repository_tree": COMPLETION_REPOSITORY_TREE,
        "producing_tasks": [
            {"task_id": task_id, "status": "completed"}
            for task_id in SELF_IMPROVEMENT_PRODUCING_TASK_IDS
        ],
        "child_goals": children,
        "evidence": evidence,
        "tasks_complete": True,
        "coverage": coverage,
        "analyzer_health": {
            "status": "healthy",
            "healthy": True,
            "safe_for_completion_reasoning": True,
            "exhaustive": True,
            "binding": dict(binding),
        },
        "exhaustion_quorum": {
            "required_members": (
                SELF_IMPROVEMENT_REQUIRED_EXHAUSTIVE_RECEIPTS
            ),
            "member_count": len(members),
            "satisfied": True,
            "quorum_met": True,
            "binding": dict(binding),
            "members": members,
        },
        "now": NOW,
        "freshness_seconds": 3600,
    }


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
                if disposition is not BenchmarkDisposition.HEALTHY
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
    confidence: float = 0.95,
    novelty: float = 0.95,
    depth: int = 2,
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
        confidence=confidence,
        estimated_cost=1.0,
        novelty=novelty,
        depth=depth,
        estimated_tokens=200,
        source="self-improvement-benchmark",
        source_id=source_id,
    )


def _typed_opaque_receipts(
    tmp_path: Path,
) -> tuple[
    tuple[HealthyExhaustionEvidence, SuccessorRefillEvidence, EpochReplayEvidence],
    str,
    str,
]:
    healthy_paths = _paths(tmp_path / "healthy")
    healthy = run_self_improvement_epoch(
        **healthy_paths,
        observation_provider=lambda binding: _observations(binding),
        capability_snapshot_id="capabilities:reconciliation-healthy-v1",
        observation_window="window:reconciliation-healthy",
        observed_at=NOW,
    )
    assert healthy.evidence is not None
    healthy_replay = run_self_improvement_epoch(
        **healthy_paths,
        observation_provider=lambda _binding: pytest.fail(
            "exact replay must not invoke the benchmark provider"
        ),
        capability_snapshot_id="capabilities:reconciliation-healthy-v1",
        observation_window="window:reconciliation-healthy",
        observed_at=NOW + timedelta(minutes=1),
    )
    replay = healthy_replay.replay_evidence
    assert replay is not None

    actionable_paths = _paths(tmp_path / "actionable")
    actionable_kwargs = {
        **actionable_paths,
        "observation_provider": lambda binding: _observations(
            binding, disposition=BenchmarkDisposition.REGRESSION
        ),
        "proposal_provider": lambda _binding, _observations: (
            _successor_proposal(source_id="reconciliation:successor"),
        ),
        "capability_snapshot_id": "capabilities:reconciliation-actionable-v1",
        "observation_window": "window:reconciliation-actionable",
        "observed_at": NOW,
        "materialization_journal_path": (
            tmp_path / "actionable" / "state" / "materialization.json"
        ),
        "discovery_dir": tmp_path / "actionable" / "discovery",
        "bundle_dir": tmp_path / "actionable" / "bundles",
    }
    successor_run = run_self_improvement_epoch(**actionable_kwargs)
    successor = successor_run.receipt.successor_evidence
    assert successor is not None
    assert (
        healthy.evidence.binding.repository_tree
        == successor.binding.repository_tree
        == replay.binding.repository_tree
    )
    assert (
        healthy.evidence.binding.policy_id
        == successor.binding.policy_id
        == replay.binding.policy_id
    )
    return (
        (healthy.evidence, successor, replay),
        healthy.evidence.binding.repository_tree,
        healthy.evidence.binding.policy_id,
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
    assert set(evidence.next_triggers) == {
        "capability_snapshot_changed",
        "operator_objective_revision",
        "policy_changed",
        "regression_observed",
        "repository_tree_changed",
        "scheduled_observation_window",
        "stale_evidence_observed",
    }

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


def test_complete_benchmark_population_emits_fresh_content_addressed_receipts(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)

    run = run_self_improvement_epoch(
        **paths,
        observation_provider=lambda binding: _observations(binding),
        capability_snapshot_id="capabilities:typed-receipts-v1",
        observation_window="window:typed-receipts",
        observed_at=NOW,
    )

    assert run.evidence is not None
    observations = run.evidence.observations
    assert len(observations) == 2 * len(DEFAULT_BENCHMARK_DIMENSIONS)
    assert {
        observation.dimension for observation in observations
    } == set(DEFAULT_BENCHMARK_DIMENSIONS)
    assert {
        observation.evidence_channel for observation in observations
    } == {"paired-benchmark-a", "paired-benchmark-b"}
    assert len({observation.receipt_id for observation in observations}) == len(
        observations
    )
    for observation in observations:
        payload = observation.to_dict()
        assert payload["producer_kind"] == "benchmark"
        assert payload["repository_tree"] == run.evidence.binding.repository_tree
        assert payload["policy_id"] == run.evidence.binding.policy_id
        assert payload["capability_snapshot_id"] == (
            run.evidence.binding.capability_snapshot_id
        )
        assert payload["command"].startswith("python -m benchmark --dimension ")
        assert payload["toolchain"] == "pytest+benchmark-harness/v1"
        assert payload["scope"]
        assert payload["result"] == {"gate": "passed", "sample_count": 3}
        assert payload["artifact_digest"].startswith("sha256:")
        assert payload["receipt_id"]
        assert observation.observed_at <= NOW <= observation.fresh_until
        assert BenchmarkObservation.from_dict(payload).receipt_id == (
            observation.receipt_id
        )

    original = observations[0]
    changed_result = original.to_dict()
    changed_result["result"] = {"gate": "passed", "sample_count": 4}
    changed_result["receipt_id"] = ""
    changed_artifact = original.to_dict()
    changed_artifact["artifact_digest"] = _digest("different-artifact")
    changed_artifact["receipt_id"] = ""
    assert BenchmarkObservation.from_dict(changed_result).receipt_id != (
        original.receipt_id
    )
    assert BenchmarkObservation.from_dict(changed_artifact).receipt_id != (
        original.receipt_id
    )


def test_all_opaque_refill_requirements_have_authoritative_typed_receipts(
    tmp_path: Path,
) -> None:
    healthy_paths = _paths(tmp_path / "healthy")
    healthy = run_self_improvement_epoch(
        **healthy_paths,
        observation_provider=lambda binding: _observations(binding),
        capability_snapshot_id="capabilities:opaque-healthy-v1",
        observation_window="window:opaque-healthy",
        observed_at=NOW,
    )
    assert healthy.evidence is not None

    actionable_paths = _paths(tmp_path / "actionable")
    actionable_kwargs = {
        **actionable_paths,
        "observation_provider": lambda binding: _observations(
            binding, disposition=BenchmarkDisposition.REGRESSION
        ),
        "proposal_provider": lambda _binding, _observations: (
            _successor_proposal(),
        ),
        "capability_snapshot_id": "capabilities:opaque-actionable-v1",
        "observation_window": "window:opaque-actionable",
        "observed_at": NOW,
        "materialization_journal_path": (
            tmp_path / "actionable" / "state" / "materialization.json"
        ),
        "discovery_dir": tmp_path / "actionable" / "discovery",
        "bundle_dir": tmp_path / "actionable" / "bundles",
    }
    successor = run_self_improvement_epoch(**actionable_kwargs)
    assert successor.receipt.successor_evidence is not None
    replay = run_self_improvement_epoch(
        **{**actionable_kwargs, "observed_at": NOW + timedelta(minutes=1)}
    )
    assert replay.replayed
    assert replay.replay_evidence is not None

    typed_receipts = (
        healthy.evidence,
        successor.receipt.successor_evidence,
        replay.replay_evidence,
    )
    assert {
        receipt.requirement_id for receipt in typed_receipts
    } == {
        HEALTHY_EXHAUSTION_REQUIREMENT_ID,
        SUCCESSOR_REFILL_REQUIREMENT_ID,
        EPOCH_IDEMPOTENCY_REQUIREMENT_ID,
    }
    for receipt in typed_receipts:
        payload = receipt.to_dict()
        assert payload["producer_kind"] in {"benchmark", "runtime"}
        assert payload["repository_tree"] == receipt.binding.repository_tree
        assert payload["policy_id"] == receipt.binding.policy_id
        assert payload["artifact_digest"] == receipt.evidence_id
        assert payload["receipt_id"] == receipt.evidence_id
        decision = completion_evidence_source_decision(
            payload,
            requirement=receipt.requirement_id,
            repository_tree=receipt.binding.repository_tree,
            policy_id=receipt.binding.policy_id,
        )
        assert decision.satisfies, decision.reason_codes
        completion = receipt.completion_evidence()
        validation = validate_completion_evidence(
            completion,
            repository_id=receipt.binding.repository_id,
            repository_tree=receipt.binding.repository_tree,
            now=(
                receipt.replayed_at
                if isinstance(receipt, EpochReplayEvidence)
                else receipt.observed_at
            ),
        )
        assert validation.valid, validation.reason_codes


def test_opaque_requirement_text_or_similarity_is_proposal_evidence_only() -> None:
    policy = EvidenceSourcePolicy()

    for requirement_id in (
        HEALTHY_EXHAUSTION_REQUIREMENT_ID,
        SUCCESSOR_REFILL_REQUIREMENT_ID,
        EPOCH_IDEMPOTENCY_REQUIREMENT_ID,
    ):
        textual = policy.evaluate(
            requirement_id,
            match_kind=EvidenceMatchKind.EXACT_TEXT,
            source_path="docs/architecture/objectives.md",
        )
        semantic = policy.evaluate(
            requirement_id,
            match_kind=EvidenceMatchKind.SEMANTIC,
            source_path="ipfs_accelerate_py/agent_supervisor/self_improvement.py",
        )
        assert not textual.satisfies
        assert "proposal_source_forbidden" in textual.reason_codes
        assert not semantic.satisfies
        assert "semantic_match_nomination_only" in semantic.reason_codes


@pytest.mark.parametrize(
    "disposition",
    [
        BenchmarkDisposition.REGRESSION,
        BenchmarkDisposition.UNCOVERED,
        BenchmarkDisposition.STALE,
        BenchmarkDisposition.BOTTLENECK,
        BenchmarkDisposition.UNSUPPORTED,
    ],
)
def test_only_supported_measured_gaps_enter_successor_generation(
    tmp_path: Path,
    disposition: BenchmarkDisposition,
) -> None:
    paths = _paths(tmp_path)
    proposal_calls = 0

    def proposals(_binding, _observations):
        nonlocal proposal_calls
        proposal_calls += 1
        return (_successor_proposal(source_id=f"gap:{disposition.value}"),)

    run = run_self_improvement_epoch(
        **paths,
        observation_provider=lambda binding: _observations(
            binding, disposition=disposition
        ),
        proposal_provider=proposals,
        capability_snapshot_id=f"capabilities:{disposition.value}",
        observation_window=f"window:{disposition.value}",
        observed_at=NOW,
        materialization_journal_path=tmp_path / "state" / "materialization.json",
        discovery_dir=tmp_path / "discovery",
        bundle_dir=tmp_path / "bundles",
    )

    assert run.status is SelfImprovementEpochStatus.SUCCESSORS_CREATED
    assert proposal_calls == 1
    assert run.receipt.actionable_dimensions == DEFAULT_BENCHMARK_DIMENSIONS
    assert len(run.receipt.created_goal_ids) == 1


@pytest.mark.parametrize(
    "disposition",
    [BenchmarkDisposition.FAILED, BenchmarkDisposition.PARTIAL],
)
def test_failed_or_partial_measurements_never_authorize_successor_writes(
    tmp_path: Path,
    disposition: BenchmarkDisposition,
) -> None:
    paths = _paths(tmp_path)
    objective_before = paths["objective_path"].read_bytes()
    todo_before = paths["todo_path"].read_bytes()
    proposal_calls = 0

    def proposals(_binding, _observations):
        nonlocal proposal_calls
        proposal_calls += 1
        return (_successor_proposal(source_id=f"invalid:{disposition.value}"),)

    run = run_self_improvement_epoch(
        **paths,
        observation_provider=lambda binding: _observations(
            binding, disposition=disposition
        ),
        proposal_provider=proposals,
        capability_snapshot_id=f"capabilities:{disposition.value}",
        observation_window=f"window:{disposition.value}",
        observed_at=NOW,
        materialization_journal_path=tmp_path / "state" / "materialization.json",
        discovery_dir=tmp_path / "discovery",
        bundle_dir=tmp_path / "bundles",
    )

    assert run.status is SelfImprovementEpochStatus.INELIGIBLE
    assert proposal_calls == 0
    assert not run.proved_requirement_ids
    assert not run.receipt.created_goal_ids
    assert paths["objective_path"].read_bytes() == objective_before
    assert paths["todo_path"].read_bytes() == todo_before


@pytest.mark.parametrize(
    ("proposal", "error"),
    [
        (
            _successor_proposal(
                source_id="quality:confidence",
                confidence=0.49,
            ),
            "quality",
        ),
        (
            _successor_proposal(
                source_id="quality:novelty",
                novelty=0.49,
            ),
            "quality",
        ),
        (
            _successor_proposal(
                source_id="refinement:depth",
                depth=4,
            ),
            "admissible",
        ),
    ],
)
def test_successor_quality_and_refinement_fail_before_transactional_writes(
    tmp_path: Path,
    proposal: ObjectiveWorkProposal,
    error: str,
) -> None:
    paths = _paths(tmp_path)
    objective_before = paths["objective_path"].read_bytes()
    todo_before = paths["todo_path"].read_bytes()

    with pytest.raises(ValueError, match=error):
        run_self_improvement_epoch(
            **paths,
            observation_provider=lambda binding: _observations(
                binding, disposition=BenchmarkDisposition.REGRESSION
            ),
            proposal_provider=lambda _binding, _observations: (proposal,),
            capability_snapshot_id=f"capabilities:{proposal.source_id}",
            observation_window=f"window:{proposal.source_id}",
            observed_at=NOW,
            materialization_journal_path=(
                tmp_path / "state" / "materialization.json"
            ),
            discovery_dir=tmp_path / "discovery",
            bundle_dir=tmp_path / "bundles",
        )

    assert paths["objective_path"].read_bytes() == objective_before
    assert paths["todo_path"].read_bytes() == todo_before
    assert not paths["ledger_path"].exists()


def test_tracker_reconciles_three_typed_receipts_to_unique_leaf_owners(
    tmp_path: Path,
) -> None:
    receipts, repository_tree, policy_id = _typed_opaque_receipts(tmp_path)
    requirement_ids = (
        SUCCESSOR_REFILL_REQUIREMENT_ID,
        EPOCH_IDEMPOTENCY_REQUIREMENT_ID,
        HEALTHY_EXHAUSTION_REQUIREMENT_ID,
    )

    result = reconcile_self_improvement_goal_evidence(
        _objective_heap(),
        typed_receipts=receipts,
        requirement_ids=requirement_ids,
        repository_tree=repository_tree,
        policy_id=policy_id,
        now=NOW + timedelta(minutes=1),
    )

    assert result.satisfied
    assert result.authoritative_requirement_ids == tuple(
        sorted(requirement_ids)
    )
    assert not result.rejected_requirement_ids
    assert not result.proposal_only_requirement_ids
    assert not result.missing_requirement_ids
    authoritative = {
        binding.requirement_id: binding
        for binding in result.bindings
        if binding.authoritative
    }
    assert {
        requirement: binding.goal_projection.goal_id
        for requirement, binding in authoritative.items()
    } == {
        SUCCESSOR_REFILL_REQUIREMENT_ID: "ASI-G109",
        EPOCH_IDEMPOTENCY_REQUIREMENT_ID: "ASI-G110",
        HEALTHY_EXHAUSTION_REQUIREMENT_ID: "ASI-G111",
    }
    assert len(
        {binding.receipt_id for binding in authoritative.values()}
    ) == 3
    assert len(
        {binding.receipt_content_id for binding in authoritative.values()}
    ) == 3
    assert all(
        binding.binding_id and binding.receipt_content_id
        for binding in authoritative.values()
    )
    restored = SelfImprovementGoalEvidenceReconciliation.from_dict(
        result.to_dict()
    )
    assert restored.reconciliation_id == result.reconciliation_id


def test_tracker_keeps_text_only_requirement_as_proposal_evidence() -> None:
    requirement_id = HEALTHY_EXHAUSTION_REQUIREMENT_ID

    result = reconcile_self_improvement_goal_evidence(
        _objective_heap(),
        requirement_ids=(requirement_id,),
        proposal_evidence={
            requirement_id: (
                "docs/architecture/objectives.md#text-match",
                "embedding:similarity/0.99",
            )
        },
        repository_tree="sha256:" + "1" * 64,
        policy_id="sha256:" + "2" * 64,
        now=NOW,
    )

    assert not result.satisfied
    assert not result.authoritative_requirement_ids
    assert result.proposal_only_requirement_ids == (requirement_id,)
    assert not result.bindings
    assert result.proposal_evidence[requirement_id] == (
        "docs/architecture/objectives.md#text-match",
        "embedding:similarity/0.99",
    )


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("command", None),
        ("toolchain", ""),
        ("scope", []),
        ("result", {}),
        ("repository_tree", "sha256:" + "3" * 64),
        ("policy_id", "sha256:" + "4" * 64),
    ],
)
def test_tracker_rejects_tampered_or_incomplete_typed_receipts(
    tmp_path: Path,
    field: str,
    replacement: object,
) -> None:
    receipts, repository_tree, policy_id = _typed_opaque_receipts(tmp_path)
    payload = receipts[0].to_dict()
    if replacement is None:
        payload.pop(field)
    else:
        payload[field] = replacement

    result = reconcile_self_improvement_goal_evidence(
        _objective_heap(),
        typed_receipts=(payload,),
        requirement_ids=(HEALTHY_EXHAUSTION_REQUIREMENT_ID,),
        repository_tree=repository_tree,
        policy_id=policy_id,
        now=NOW,
    )

    assert not result.satisfied
    assert result.rejected_requirement_ids == (
        HEALTHY_EXHAUSTION_REQUIREMENT_ID,
    )
    assert len(result.bindings) == 1
    binding = result.bindings[0]
    assert not binding.authoritative
    assert {
        "receipt_integrity_invalid",
        "receipt_canonical_projection_mismatch",
    } & set(binding.reason_codes)


def test_tracker_rejects_stale_typed_receipt(tmp_path: Path) -> None:
    receipts, repository_tree, policy_id = _typed_opaque_receipts(tmp_path)

    result = reconcile_self_improvement_goal_evidence(
        _objective_heap(),
        typed_receipts=(receipts[0],),
        requirement_ids=(HEALTHY_EXHAUSTION_REQUIREMENT_ID,),
        repository_tree=repository_tree,
        policy_id=policy_id,
        now=NOW + timedelta(days=2),
    )

    assert not result.satisfied
    assert result.rejected_requirement_ids == (
        HEALTHY_EXHAUSTION_REQUIREMENT_ID,
    )
    assert len(result.bindings) == 1
    assert "receipt_stale" in result.bindings[0].reason_codes


def test_tracker_rejects_distinct_receipts_for_one_requirement(
    tmp_path: Path,
) -> None:
    first_paths = _paths(tmp_path / "first")
    second_paths = _paths(tmp_path / "second")
    first = run_self_improvement_epoch(
        **first_paths,
        observation_provider=lambda binding: _observations(binding),
        capability_snapshot_id="capabilities:duplicate-a",
        observation_window="window:duplicate-a",
        observed_at=NOW,
    )
    second = run_self_improvement_epoch(
        **second_paths,
        observation_provider=lambda binding: _observations(binding),
        capability_snapshot_id="capabilities:duplicate-b",
        observation_window="window:duplicate-b",
        observed_at=NOW,
    )
    assert first.evidence is not None
    assert second.evidence is not None
    assert (
        first.evidence.binding.repository_tree
        == second.evidence.binding.repository_tree
    )
    assert first.evidence.binding.policy_id == second.evidence.binding.policy_id
    assert first.evidence.evidence_id != second.evidence.evidence_id

    result = reconcile_self_improvement_goal_evidence(
        _objective_heap(),
        typed_receipts=(first.evidence, second.evidence),
        requirement_ids=(HEALTHY_EXHAUSTION_REQUIREMENT_ID,),
        repository_tree=first.evidence.binding.repository_tree,
        policy_id=first.evidence.binding.policy_id,
        now=NOW,
    )

    assert not result.satisfied
    assert len(result.bindings) == 2
    assert all(not binding.authoritative for binding in result.bindings)
    assert all(
        "duplicate_requirement_receipts" in binding.reason_codes
        for binding in result.bindings
    )


@pytest.mark.parametrize("defect", ["expired", "incomplete", "future"])
def test_nonstale_actionable_measurements_must_be_fresh_and_complete(
    tmp_path: Path,
    defect: str,
) -> None:
    paths = _paths(tmp_path)
    proposal_calls = 0

    def observations(binding):
        records = []
        for item in _observations(
            binding,
            disposition=BenchmarkDisposition.REGRESSION,
        ):
            payload = item.to_dict()
            payload["receipt_id"] = ""
            if defect == "expired":
                payload["fresh_until"] = (
                    NOW - timedelta(seconds=1)
                ).isoformat()
            elif defect == "incomplete":
                payload["complete"] = False
                payload["coverage_complete"] = False
            else:
                payload["observed_at"] = (
                    NOW + timedelta(seconds=1)
                ).isoformat()
                payload["fresh_until"] = (
                    NOW + timedelta(hours=1)
                ).isoformat()
            records.append(BenchmarkObservation.from_dict(payload))
        return tuple(records)

    def proposals(_binding, _observations):
        nonlocal proposal_calls
        proposal_calls += 1
        return (_successor_proposal(),)

    run = run_self_improvement_epoch(
        **paths,
        observation_provider=observations,
        proposal_provider=proposals,
        capability_snapshot_id=f"capabilities:invalid-{defect}",
        observation_window=f"window:invalid-{defect}",
        observed_at=NOW,
    )

    assert run.status is SelfImprovementEpochStatus.INELIGIBLE
    assert "benchmark_not_fresh_and_complete" in run.receipt.blocker_codes
    assert proposal_calls == 0
    assert not run.receipt.created_goal_ids


def test_successor_filter_covers_terminal_lifecycle_and_durable_cooldown(
    tmp_path: Path,
) -> None:
    proposal = _successor_proposal()
    terminal_heap = _objective_heap() + f"""

## ASI-G199 Historical equivalent successor

- Status: verified_complete
- Parent: ASI-G080
- Goal: Historical equivalent work remains deduplication authority
- Evidence: historical-successor-proof
- Canonical proposal ID: {proposal.canonical_id}
- Semantic key: {proposal.semantic_key}
"""
    lifecycle = filter_self_improvement_successor_candidates(
        (proposal,),
        objective_text=terminal_heap,
        strategy={},
        observed_at=NOW,
    )
    assert not lifecycle.eligible
    assert [item.reason for item in lifecycle.rejected] == [
        "lifecycle_duplicate"
    ]

    strategy_path = tmp_path / "strategy.json"
    record_self_improvement_successor_admission(
        strategy_path,
        epoch_id="epoch:rejected",
        proposals=(proposal,),
        rejection_reasons={proposal.canonical_id: "quality_rejected"},
        recorded_at=NOW,
        cooldown_seconds=60,
    )
    active = filter_self_improvement_successor_candidates(
        (proposal,),
        objective_text=_objective_heap(),
        strategy=load_strategy(strategy_path),
        observed_at=NOW + timedelta(seconds=30),
    )
    assert not active.eligible
    assert [item.reason for item in active.rejected] == ["successor_cooldown"]

    expired = filter_self_improvement_successor_candidates(
        (proposal,),
        objective_text=_objective_heap(),
        strategy=load_strategy(strategy_path),
        observed_at=NOW + timedelta(seconds=61),
    )
    assert expired.eligible == (proposal,)

    record_self_improvement_successor_admission(
        strategy_path,
        epoch_id="epoch:admitted",
        proposals=(proposal,),
        admitted_proposal_ids=(proposal.canonical_id,),
        transaction_id="transaction:committed",
        recorded_at=NOW + timedelta(seconds=61),
    )
    record_self_improvement_successor_admission(
        strategy_path,
        epoch_id="epoch:later-rejection",
        proposals=(proposal,),
        rejection_reasons={proposal.canonical_id: "duplicate"},
        recorded_at=NOW + timedelta(days=1),
    )
    strategy = load_strategy(strategy_path)
    record = strategy[SELF_IMPROVEMENT_SUCCESSOR_RECORDS_KEY][
        proposal.canonical_id
    ]
    assert record["status"] == "admitted"
    assert record["transaction_id"] == "transaction:committed"
    permanent = filter_self_improvement_successor_candidates(
        (proposal,),
        objective_text=_objective_heap(),
        strategy=strategy,
        observed_at=NOW + timedelta(days=365),
    )
    assert not permanent.eligible
    assert [item.reason for item in permanent.rejected] == [
        "prior_admission_duplicate"
    ]


def _assert_parent_completion_rejected(packet: dict[str, object]) -> None:
    decision = evaluate_self_improvement_completion(
        **packet,
        current_state=GoalState.PROVISIONALLY_COMPLETE,
    )
    assert not decision.verified
    assert decision.state is not GoalState.VERIFIED_COMPLETE
    assert decision.reason_codes


def test_g080_parent_completion_requires_closed_current_tree_proof_packet() -> None:
    assert SELF_IMPROVEMENT_PRODUCING_TASK_IDS == ("ASI-022",)
    assert SELF_IMPROVEMENT_CHILD_GOAL_IDS == (
        "ASI-G109",
        "ASI-G110",
        "ASI-G111",
    )
    assert len(SELF_IMPROVEMENT_ACCEPTANCE_CRITERIA) == 5
    assert SELF_IMPROVEMENT_REQUIRED_EXHAUSTIVE_RECEIPTS == 2

    packet = _completion_packet()
    provisional = evaluate_self_improvement_completion(**packet)
    assert provisional.state is GoalState.PROVISIONALLY_COMPLETE
    assert not provisional.verified
    assert provisional.gate is not None and provisional.gate.passed
    assert "provisional_transition_required" in provisional.reason_codes
    assert provisional.gate.evaluated_evidence["coverage"][
        "producing_task_closure"
    ] == {
        "required_task_ids": ["ASI-022"],
        "submitted_task_ids": ["ASI-022"],
        "submitted_task_statuses": ["completed"],
        "population_complete": True,
        "caller_tasks_complete": True,
        "satisfied": True,
    }

    verified = evaluate_self_improvement_completion(
        **packet,
        current_state=GoalState.PROVISIONALLY_COMPLETE,
    )
    assert verified.state is GoalState.VERIFIED_COMPLETE
    assert verified.verified
    assert verified.gate is not None and verified.gate.passed
    assert align_completion_gate_force_goal_ids(
        completion_gate_decisions={"ASI-G080": provisional},
        repository_id=COMPLETION_REPOSITORY_ID,
        repository_tree=COMPLETION_REPOSITORY_TREE,
        now=NOW,
    ) == ("ASI-G080",)
    assert align_completion_gate_force_goal_ids(
        completion_gate_decisions={"ASI-G080": verified},
        repository_id=COMPLETION_REPOSITORY_ID,
        repository_tree=COMPLETION_REPOSITORY_TREE,
        now=NOW,
    ) == ()


def test_g080_verified_completion_reopens_and_requeues_on_stale_proof() -> None:
    packet = _completion_packet()
    verified = evaluate_self_improvement_completion(
        **packet,
        current_state=GoalState.PROVISIONALLY_COMPLETE,
    )
    assert verified.verified

    records = list(packet["evidence"])
    stale = records[0].to_dict()
    stale["observed_at"] = (NOW - timedelta(hours=2)).isoformat()
    records[0] = CompletionEvidence.from_dict(stale)
    packet["evidence"] = tuple(records)
    reopened = evaluate_self_improvement_completion(
        **packet,
        current_state=GoalState.VERIFIED_COMPLETE,
    )

    assert reopened.state is GoalState.REOPENED
    assert not reopened.verified
    assert "verification_invalidated" in reopened.reason_codes
    assert "stale_evidence" in reopened.reason_codes
    assert align_completion_gate_force_goal_ids(
        completion_gate_decisions={"ASI-G080": reopened.to_dict()},
        repository_id=COMPLETION_REPOSITORY_ID,
        repository_tree=COMPLETION_REPOSITORY_TREE,
        now=NOW,
    ) == ("ASI-G080",)


def test_g080_durable_backlog_projection_requires_canonical_fresh_decision() -> None:
    packet = _completion_packet()
    verified = evaluate_self_improvement_completion(
        **packet,
        current_state=GoalState.PROVISIONALLY_COMPLETE,
    ).to_dict()
    alignment = {
        "repository_id": COMPLETION_REPOSITORY_ID,
        "repository_tree": COMPLETION_REPOSITORY_TREE,
        "now": NOW,
    }
    assert align_completion_gate_force_goal_ids(
        ("ASI-G999", "ASI-G999"),
        completion_gate_decisions={"ASI-G080": verified},
        **alignment,
    ) == ("ASI-G999",)

    skeletal = {
        "state": "verified_complete",
        "verified": True,
        "completion_gate": {"passed": True},
        "actionable_reasons": [],
    }
    assert align_completion_gate_force_goal_ids(
        completion_gate_decisions={"ASI-G080": skeletal},
        **alignment,
    ) == ("ASI-G080",)

    stale = copy.deepcopy(verified)
    stale["completion_gate"]["evaluated_evidence"]["evaluated_at"] = (
        NOW - timedelta(hours=2)
    ).isoformat()
    assert align_completion_gate_force_goal_ids(
        completion_gate_decisions={"ASI-G080": stale},
        **alignment,
    ) == ("ASI-G080",)

    assert align_completion_gate_force_goal_ids(
        completion_gate_decisions={"ASI-G080": verified},
        repository_id=COMPLETION_REPOSITORY_ID,
        repository_tree="tree:sha256:new-current-tree",
        now=NOW,
    ) == ("ASI-G080",)

    incomplete = copy.deepcopy(verified)
    incomplete["completion_gate"]["checks"][0]["passed"] = False
    assert align_completion_gate_force_goal_ids(
        completion_gate_decisions={"ASI-G080": incomplete},
        **alignment,
    ) == ("ASI-G080",)

    detached = copy.deepcopy(verified)
    detached["completion_gate"]["evaluated_evidence"]["coverage"] = {}
    assert align_completion_gate_force_goal_ids(
        completion_gate_decisions={"ASI-G080": detached},
        **alignment,
    ) == ("ASI-G080",)


def test_configured_refill_receives_current_completion_alignment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packet = _completion_packet()
    provisional = evaluate_self_improvement_completion(**packet)
    verified = evaluate_self_improvement_completion(
        **packet,
        current_state=GoalState.PROVISIONALLY_COMPLETE,
    )
    captured: list[dict[str, object]] = []

    def fake_record(**kwargs):
        captured.append(kwargs)
        return kwargs

    monkeypatch.setattr(
        backlog_refinery_module,
        "record_objective_backlog_findings",
        fake_record,
    )
    monkeypatch.setattr(
        objective_tracker_module,
        "completion_tree_identity",
        lambda *_args, **_kwargs: RepositoryTreeIdentity(
            repository_id=COMPLETION_REPOSITORY_ID,
            tree_id=COMPLETION_REPOSITORY_TREE,
        ),
    )
    common = {
        "repo_root": tmp_path,
        "objective_path": tmp_path / "objectives.md",
        "todo_path": tmp_path / "todo.md",
        "discovery_dir": tmp_path / "discovery",
        "strategy_path": tmp_path / "strategy.json",
        "completion_gate_now": NOW,
        "persist_ast_dataset": False,
        "write_todo_vector_index": False,
    }

    record_configured_objective_backlog_findings(
        **common,
        completion_gate_decisions={"ASI-G080": provisional.to_dict()},
    )
    assert captured[-1]["force_goal_ids"] == ("ASI-G080",)

    record_configured_objective_backlog_findings(
        **common,
        completion_gate_decisions={"ASI-G080": verified.to_dict()},
    )
    assert captured[-1]["force_goal_ids"] == ()


@pytest.mark.parametrize(
    "mutation",
    [
        lambda tasks: tasks.clear(),
        lambda tasks: tasks.append(copy.deepcopy(tasks[0])),
        lambda tasks: tasks[0].update(status="todo"),
        lambda tasks: tasks[0].update(task_id="ASI-999"),
    ],
    ids=["missing", "duplicate", "incomplete", "foreign"],
)
def test_g080_parent_rejects_incomplete_wrong_or_duplicate_producers(
    mutation,
) -> None:
    packet = _completion_packet()
    tasks = packet["producing_tasks"]
    assert isinstance(tasks, list)
    mutation(tasks)
    _assert_parent_completion_rejected(packet)

    for closure_claim in (False, 1):
        packet = _completion_packet()
        packet["tasks_complete"] = closure_claim
        decision = evaluate_self_improvement_completion(
            **packet,
            current_state=GoalState.PROVISIONALLY_COMPLETE,
        )
        assert "tasks_incomplete" in decision.reason_codes


@pytest.mark.parametrize(
    "defect",
    ["missing", "duplicate", "failed", "stale", "foreign_tree"],
)
def test_g080_parent_rejects_each_invalid_submitted_criterion_evidence(
    defect: str,
) -> None:
    packet = _completion_packet()
    records = list(packet["evidence"])
    if defect == "missing":
        records.pop()
    elif defect == "duplicate":
        records.append(records[0])
    else:
        payload = records[0].to_dict()
        if defect == "failed":
            payload["validation_passed"] = False
            payload["validation_receipt"] = {
                **payload["validation_receipt"],
                "status": "failed",
            }
        elif defect == "stale":
            payload["observed_at"] = (NOW - timedelta(hours=2)).isoformat()
        else:
            payload["repository_tree"] = "tree:sha256:foreign"
            payload["tree_id"] = "tree:sha256:foreign"
        records[0] = CompletionEvidence.from_dict(payload)
    packet["evidence"] = tuple(records)
    _assert_parent_completion_rejected(packet)


@pytest.mark.parametrize(
    "defect",
    ["missing_row", "duplicate_row", "missing_implementation", "detached"],
)
def test_g080_parent_rejects_incomplete_or_unbound_coverage(
    defect: str,
) -> None:
    packet = _completion_packet()
    coverage = packet["coverage"]
    assert isinstance(coverage, dict)
    rows = coverage["criteria"]
    assert isinstance(rows, list)
    if defect == "missing_row":
        rows.pop()
    elif defect == "duplicate_row":
        rows.append(copy.deepcopy(rows[0]))
    elif defect == "missing_implementation":
        rows[0].pop("implementation")
    else:
        rows[0]["validation_receipt_ids"] = ["validation:detached"]
    _assert_parent_completion_rejected(packet)


@pytest.mark.parametrize(
    "defect",
    ["missing", "unhealthy", "unsafe", "foreign_binding"],
)
def test_g080_parent_requires_explicit_completion_safe_analyzer(
    defect: str,
) -> None:
    packet = _completion_packet()
    health = packet["analyzer_health"]
    assert isinstance(health, dict)
    if defect == "missing":
        packet["analyzer_health"] = None
    elif defect == "unhealthy":
        health["healthy"] = False
    elif defect == "unsafe":
        health["safe_for_completion_reasoning"] = False
    else:
        health["binding"] = {
            **health["binding"],
            "objective_revision": "ASI-G080@foreign",
        }
    _assert_parent_completion_rejected(packet)


@pytest.mark.parametrize(
    "defect",
    [
        "under_count",
        "duplicate_member",
        "duplicate_channel",
        "duplicate_receipt",
        "stale",
        "unhealthy",
        "unsafe",
        "non_exhaustive",
        "foreign_binding",
    ],
)
def test_g080_parent_requires_independent_fresh_healthy_exhaustive_quorum(
    defect: str,
) -> None:
    packet = _completion_packet()
    quorum = packet["exhaustion_quorum"]
    assert isinstance(quorum, dict)
    members = quorum["members"]
    assert isinstance(members, list)
    if defect == "under_count":
        members.pop()
        quorum["member_count"] = 1
    elif defect == "duplicate_member":
        members[1]["member_id"] = members[0]["member_id"]
    elif defect == "duplicate_channel":
        members[1]["evidence_channel"] = members[0]["evidence_channel"]
    elif defect == "duplicate_receipt":
        members[1]["receipt_cid"] = members[0]["receipt_cid"]
    elif defect == "stale":
        members[0]["finished_at"] = (NOW - timedelta(hours=2)).isoformat()
    elif defect == "unhealthy":
        members[0]["healthy"] = False
    elif defect == "unsafe":
        members[0]["safe_for_completion_reasoning"] = False
    elif defect == "non_exhaustive":
        members[0]["scan_mode"] = "incremental"
    else:
        members[0]["binding"] = {
            **members[0]["binding"],
            "tree_id": "tree:sha256:foreign",
        }
    _assert_parent_completion_rejected(packet)

    with pytest.raises(ValueError, match="configured ASI-G080 count"):
        evaluate_self_improvement_completion(
            **_completion_packet(),
            required_exhaustive_receipts=1,
        )


@pytest.mark.parametrize(
    "defect",
    ["missing", "duplicate", "unverified", "stale", "foreign_tree", "proofless"],
)
def test_g080_parent_rejects_unverified_stale_or_wrong_child_population(
    defect: str,
) -> None:
    packet = _completion_packet()
    children = packet["child_goals"]
    assert isinstance(children, list)
    if defect == "missing":
        children.pop()
    elif defect == "duplicate":
        children.append(copy.deepcopy(children[0]))
    elif defect == "unverified":
        children[0]["verified"] = False
        children[0]["state"] = "active"
    elif defect == "stale":
        children[0]["completion_gate"]["evaluated_evidence"][
            "evaluated_at"
        ] = (NOW - timedelta(hours=2)).isoformat()
    elif defect == "foreign_tree":
        children[0]["completion_gate"]["evaluated_evidence"][
            "repository_tree"
        ] = "tree:sha256:foreign"
    else:
        children[0]["proof_requirements"] = []
    _assert_parent_completion_rejected(packet)
