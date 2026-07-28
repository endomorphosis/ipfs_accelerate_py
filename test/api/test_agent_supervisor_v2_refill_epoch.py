from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.objectives.objective_graph import (
    objective_heap_content_id,
)
from ipfs_accelerate_py.agent_supervisor.self_improvement.self_improvement_v2 import (
    MAX_V2_SUCCESSOR_GOALS,
    MAX_V2_SUCCESSOR_TASKS,
    V2RefillEpochBinding,
    V2RefillEpochPreview,
    V2RefillEpochResult,
    V2RefillEpochStatus,
    V2RefillObservation,
    V2ResidualKind,
    V2ResidualSignal,
    V2SuccessorGenerationPolicy,
    generate_v2_successor_goals,
    preview_v2_refill_epoch,
    run_v2_refill_epoch,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.taskboard_store import (
    taskboard_revision,
)


NOW = datetime(2026, 7, 26, 12, 0, tzinfo=timezone.utc)


def _objective_heap() -> str:
    return """# Objective Heap

## ASI-G000 Root objective

- Status: active
- Goal: Maintain the supervisor objective graph
- Evidence: root-proof

## ASI-G290 Reward-resistant generation-2 self-improvement

- Status: active
- Parent: ASI-G000
- Goal: Improve only from typed current-tree residual evidence
- Evidence: refill-epoch-proof
- Refinement depth: 1
"""


def _drained_board() -> str:
    return """# Tasks

## ASI-120 Generate typed successors

- Status: completed
"""


def _paths(tmp_path: Path) -> dict[str, Path]:
    repo = tmp_path / "repo"
    repo.mkdir(parents=True)
    objective = repo / "objectives.md"
    taskboard = repo / "todo.md"
    objective.write_text(_objective_heap(), encoding="utf-8")
    taskboard.write_text(_drained_board(), encoding="utf-8")
    return {
        "repo_root": repo,
        "objective_path": objective,
        "taskboard_path": taskboard,
        "journal_path": tmp_path / "state" / "v2-refill-journal.json",
        "wait_state_path": tmp_path / "state" / "v2-refill-wait.json",
    }


def _residual(
    index: int = 0,
    *,
    task_count: int = 1,
    kind: V2ResidualKind = V2ResidualKind.REGRESSION,
) -> V2ResidualSignal:
    vocabularies = (
        "cache reuse hydration",
        "planner horizon decomposition",
        "analysis inference calibration",
        "validation fixture isolation",
        "throughput queue saturation",
        "persistence checkpoint recovery",
        "control lease fencing",
        "storage compaction retention",
        "context retrieval ranking",
    )
    vocabulary = vocabularies[index % len(vocabularies)]
    slug = vocabulary.replace(" ", "-")
    return V2ResidualSignal(
        residual_id=f"residual:{slug}",
        kind=kind,
        title=f"Repair {slug}",
        detail=(
            f"The current-tree benchmark receipt reports {vocabulary}. "
            f"Repair the measured {vocabulary} residual while preserving its "
            "declared safety gate."
        ),
        acceptance_criteria=(
            f"The {slug} benchmark meets its declared threshold",
            f"A focused regression test closes {slug}",
        ),
        evidence_ids=(f"receipt:{slug}",),
        predicted_files=(
            f"ipfs_accelerate_py/agent_supervisor/{slug.replace('-', '_')}.py",
            f"test/api/test_{slug.replace('-', '_')}.py",
        ),
        predicted_symbols=(f"repair_{slug.replace('-', '_')}",),
        validation_commands=(
            f"python -m pytest test/api/test_{slug.replace('-', '_')}.py -q",
        ),
        confidence=0.9,
        estimated_tokens=1_000,
        depth=2,
        task_count=task_count,
        changed=True,
        completed=False,
        source_receipt_id=f"receipt:{slug}",
    )


def _receipt(
    binding: V2RefillEpochBinding,
    suffix: str,
    *,
    producer_id: str | None = None,
    evidence_channel: str | None = None,
    receipt_id: str | None = None,
    observed_at: datetime | None = None,
    fresh_until: datetime | None = None,
    healthy: bool = True,
    exhaustive: bool = True,
) -> dict[str, Any]:
    """Return one independently produced receipt bound to the whole epoch."""

    window_end = datetime.fromisoformat(binding.observation_window_end)
    receipt_observed_at = observed_at or (window_end - timedelta(minutes=2))
    receipt_fresh_until = fresh_until or (window_end + timedelta(hours=1))
    return {
        "receipt_id": receipt_id or f"exhaustion:{suffix}",
        "producer_id": producer_id or f"producer:{suffix}",
        "evidence_channel": evidence_channel or f"channel:{suffix}",
        "implementation_id": f"implementation:{suffix}",
        "binding": binding.to_dict(),
        "healthy": healthy,
        "exhaustive": exhaustive,
        "complete": True,
        "safe_for_completion_reasoning": True,
        "observed_at": receipt_observed_at.isoformat(),
        "fresh_until": receipt_fresh_until.isoformat(),
    }


def _healthy_observation(
    binding: V2RefillEpochBinding,
) -> V2RefillObservation:
    return V2RefillObservation(
        residuals=(),
        exhaustion_receipts=(
            _receipt(binding, "paired-benchmark"),
            _receipt(binding, "independent-audit"),
        ),
    )


def _kwargs(paths: dict[str, Path], **overrides: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        **paths,
        "repository_id": "repository:ipfs-accelerate-py",
        "tree_id": "tree:sha256:v2-refill-current",
        "objective_id": "ASI-G290",
        "benchmark_policy_id": "policy:v2-benchmark",
        "benchmark_policy_revision": "sha256:benchmark-policy-v3",
        "capability_id": "capabilities:agent-supervisor",
        "capability_revision": "sha256:capabilities-v7",
        "operation_catalog_id": "catalog:supervisor-operations-v4",
        "storage_policy_id": "storage:durable-local-cas-v2",
        "observation_window_start": NOW - timedelta(hours=1),
        "observation_window_end": NOW,
        "now": NOW,
    }
    values.update(overrides)
    return values


def _binding(paths: dict[str, Path]) -> V2RefillEpochBinding:
    return V2RefillEpochBinding(
        repository_id="repository:ipfs-accelerate-py",
        tree_id="tree:sha256:v2-refill-current",
        objective_id="ASI-G290",
        objective_revision=objective_heap_content_id(
            paths["objective_path"].read_text(encoding="utf-8")
        ),
        board_revision=taskboard_revision(
            paths["taskboard_path"].read_bytes()
        ),
        benchmark_policy_id="policy:v2-benchmark",
        benchmark_policy_revision="sha256:benchmark-policy-v3",
        capability_id="capabilities:agent-supervisor",
        capability_revision="sha256:capabilities-v7",
        operation_catalog_id="catalog:supervisor-operations-v4",
        storage_policy_id="storage:durable-local-cas-v2",
        observation_window_start=NOW - timedelta(hours=1),
        observation_window_end=NOW,
    )


def _status_value(result: V2RefillEpochResult) -> str:
    status = result.status
    return status.value if isinstance(status, V2RefillEpochStatus) else str(status)


def _journal(paths: dict[str, Path]) -> dict[str, Any]:
    value = json.loads(paths["journal_path"].read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _transaction(journal: dict[str, Any], epoch_id: str) -> dict[str, Any]:
    transactions = journal["transactions"]
    assert isinstance(transactions, dict)
    matches = [
        value
        for value in transactions.values()
        if isinstance(value, dict) and value.get("epoch_id") == epoch_id
    ]
    assert len(matches) == 1
    return matches[0]


def test_preview_is_pure_and_contains_the_complete_exact_delta(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    before = {
        paths["objective_path"]: paths["objective_path"].read_bytes(),
        paths["taskboard_path"]: paths["taskboard_path"].read_bytes(),
    }
    admission = generate_v2_successor_goals(
        (_residual(task_count=2),),
        observed_at=NOW,
    )

    preview = preview_v2_refill_epoch(
        binding=_binding(paths),
        admission=admission,
        objective_text=before[paths["objective_path"]].decode("utf-8"),
        taskboard_text=before[paths["taskboard_path"]].decode("utf-8"),
    )

    assert preview.ready
    assert len(preview.goal_ids) == 1
    assert len(preview.task_ids) == 2
    assert preview.goal_task_mappings[0].task_ids == preview.task_ids
    assert preview.candidate_objective_revision != (
        preview.base_objective_revision
    )
    assert preview.candidate_board_revision != preview.base_board_revision
    assert {
        path: path.read_bytes()
        for path in before
    } == before


def test_epoch_binds_every_input_and_previews_one_exact_mapped_delta(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    provider_calls: list[str] = []
    proposal_calls: list[str] = []
    residual = _residual(task_count=2)
    admission = generate_v2_successor_goals(
        (residual,),
        observed_at=NOW,
    )

    def observe(binding: V2RefillEpochBinding) -> V2RefillObservation:
        provider_calls.append(binding.epoch_id)
        return V2RefillObservation(residuals=(residual,))

    def propose(*_args: Any, **_kwargs: Any) -> Any:
        proposal_calls.append("proposal")
        return admission

    result = run_v2_refill_epoch(
        **_kwargs(paths),
        observation_provider=observe,
        proposal_provider=propose,
    )

    assert isinstance(result, V2RefillEpochResult)
    assert _status_value(result) == "proposed"
    assert not result.replayed
    assert provider_calls == [result.epoch_id]
    assert proposal_calls == ["proposal"]

    binding = result.binding
    assert isinstance(binding, V2RefillEpochBinding)
    assert binding.repository_id == "repository:ipfs-accelerate-py"
    assert binding.tree_id == "tree:sha256:v2-refill-current"
    assert binding.objective_id == "ASI-G290"
    assert binding.objective_revision == result.preview.base_objective_revision
    assert binding.board_revision == result.preview.base_board_revision
    assert binding.benchmark_policy_id == "policy:v2-benchmark"
    assert binding.benchmark_policy_revision == "sha256:benchmark-policy-v3"
    assert binding.capability_id == "capabilities:agent-supervisor"
    assert binding.capability_revision == "sha256:capabilities-v7"
    assert binding.operation_catalog_id == "catalog:supervisor-operations-v4"
    assert binding.storage_policy_id == "storage:durable-local-cas-v2"
    assert binding.observation_window_start == (
        NOW - timedelta(hours=1)
    ).isoformat()
    assert binding.observation_window_end == NOW.isoformat()

    preview = result.preview
    assert isinstance(preview, V2RefillEpochPreview)
    assert preview.epoch_id == result.epoch_id
    assert preview.admission_id == admission.admission_id
    assert preview.goal_ids == tuple(
        item.proposal.canonical_id for item in admission.accepted
    )
    assert preview.task_ids == admission.accepted[0].task_ids
    assert len(preview.goal_task_mappings) == 1
    assert preview.goal_task_mappings[0].goal_id == preview.goal_ids[0]
    assert preview.goal_task_mappings[0].task_ids == preview.task_ids
    assert result.created_goal_ids == preview.goal_ids
    assert result.created_task_ids == preview.task_ids

    objective = paths["objective_path"].read_text(encoding="utf-8")
    taskboard = paths["taskboard_path"].read_text(encoding="utf-8")
    assert objective == preview.candidate_objective_text
    assert taskboard == preview.candidate_board_text
    assert objective.count(f"## {preview.goal_ids[0]} ") == 1
    assert all(
        taskboard.count(f"## {task_id} ") == 1
        for task_id in preview.task_ids
    )

    transaction = _transaction(_journal(paths), result.epoch_id)
    assert transaction["state"] == "committed"
    assert transaction["base_objective_revision"] == binding.objective_revision
    assert transaction["base_board_revision"] == binding.board_revision
    assert transaction["candidate_objective_revision"] == (
        preview.candidate_objective_revision
    )
    assert transaction["candidate_board_revision"] == (
        preview.candidate_board_revision
    )
    assert transaction["goal_ids"] == list(preview.goal_ids)
    assert transaction["task_ids"] == list(preview.task_ids)


@pytest.mark.parametrize(
    ("field", "changed_value"),
    [
        ("repository_id", "repository:fork"),
        ("tree_id", "tree:sha256:changed"),
        ("benchmark_policy_id", "policy:v2-benchmark-next"),
        ("benchmark_policy_revision", "sha256:benchmark-policy-v4"),
        ("capability_id", "capabilities:replacement-supervisor"),
        ("capability_revision", "sha256:capabilities-v8"),
        ("operation_catalog_id", "catalog:supervisor-operations-v5"),
        ("storage_policy_id", "storage:durable-local-cas-v3"),
        (
            "observation_window_start",
            NOW - timedelta(hours=2),
        ),
        (
            "observation_window_end",
            NOW + timedelta(minutes=1),
        ),
    ],
)
def test_each_declared_external_binding_changes_epoch_identity(
    tmp_path: Path,
    field: str,
    changed_value: Any,
) -> None:
    baseline_paths = _paths(tmp_path / "baseline")
    changed_paths = _paths(tmp_path / "changed")
    baseline = run_v2_refill_epoch(
        **_kwargs(baseline_paths),
        observation_provider=_healthy_observation,
    )
    changed = run_v2_refill_epoch(
        **_kwargs(changed_paths, **{field: changed_value}),
        observation_provider=_healthy_observation,
    )

    assert baseline.binding.to_dict() != changed.binding.to_dict()
    assert baseline.epoch_id != changed.epoch_id


@pytest.mark.parametrize("document", ["objective", "taskboard"])
def test_objective_and_board_revisions_are_epoch_identity_inputs(
    tmp_path: Path,
    document: str,
) -> None:
    baseline_paths = _paths(tmp_path / "baseline")
    changed_paths = _paths(tmp_path / "changed")
    target = changed_paths[
        "objective_path" if document == "objective" else "taskboard_path"
    ]
    target.write_text(
        target.read_text(encoding="utf-8")
        + f"\n<!-- meaningful {document} revision -->\n",
        encoding="utf-8",
    )

    baseline = run_v2_refill_epoch(
        **_kwargs(baseline_paths),
        observation_provider=_healthy_observation,
    )
    changed = run_v2_refill_epoch(
        **_kwargs(changed_paths),
        observation_provider=_healthy_observation,
    )

    revision_field = (
        "objective_revision" if document == "objective" else "board_revision"
    )
    assert getattr(baseline.binding, revision_field) != getattr(
        changed.binding, revision_field
    )
    assert baseline.epoch_id != changed.epoch_id


def test_epoch_hard_caps_at_eight_goals_and_twenty_four_tasks(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    residuals = tuple(_residual(index, task_count=3) for index in range(9))
    admission = generate_v2_successor_goals(
        residuals,
        policy=V2SuccessorGenerationPolicy(min_semantic_novelty=0.0),
        observed_at=NOW,
    )
    assert len(admission.accepted) == 8
    assert admission.generated_task_count == 24

    result = run_v2_refill_epoch(
        **_kwargs(paths),
        observation_provider=lambda _binding: V2RefillObservation(
            residuals=residuals
        ),
        proposal_provider=lambda *_args, **_kwargs: admission,
    )

    assert _status_value(result) == "proposed"
    assert len(result.created_goal_ids) == MAX_V2_SUCCESSOR_GOALS == 8
    assert len(result.created_task_ids) == MAX_V2_SUCCESSOR_TASKS == 24
    assert len(set(result.created_goal_ids)) == 8
    assert len(set(result.created_task_ids)) == 24
    assert sum(
        len(mapping.task_ids)
        for mapping in result.preview.goal_task_mappings
    ) == 24
    assert result.admission.rejected[-1].reason.value == "goal-budget"

    with pytest.raises(ValueError):
        V2SuccessorGenerationPolicy(max_goals=9)
    with pytest.raises(ValueError):
        V2SuccessorGenerationPolicy(max_tasks=25)


def test_objective_and_board_commit_share_cas_and_durable_journal(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    residual = _residual()
    admission = generate_v2_successor_goals((residual,), observed_at=NOW)
    objective_before = paths["objective_path"].read_bytes()

    def mutate_after_preview(*_args: Any, **_kwargs: Any) -> Any:
        paths["taskboard_path"].write_text(
            _drained_board() + "\n<!-- concurrent board owner -->\n",
            encoding="utf-8",
        )
        return admission

    blocked = run_v2_refill_epoch(
        **_kwargs(paths),
        observation_provider=lambda _binding: V2RefillObservation(
            residuals=(residual,)
        ),
        proposal_provider=mutate_after_preview,
    )

    assert _status_value(blocked) == "rejected"
    assert not blocked.created_goal_ids
    assert not blocked.created_task_ids
    assert paths["objective_path"].read_bytes() == objective_before
    assert paths["taskboard_path"].read_text(encoding="utf-8").endswith(
        "<!-- concurrent board owner -->\n"
    )
    transaction = _transaction(_journal(paths), blocked.epoch_id)
    assert transaction["state"] in {"blocked", "prepared"}
    assert "stale_board_revision" in transaction["reason_codes"]


def test_durable_journal_recovers_interruption_after_heap_cas(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.task_sources import taskboard_store

    paths = _paths(tmp_path)
    residual = _residual(task_count=2)
    admission = generate_v2_successor_goals((residual,), observed_at=NOW)
    objective_before = paths["objective_path"].read_bytes()
    board_before = paths["taskboard_path"].read_bytes()

    def interrupted_board_commit(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("simulated process interruption")

    with monkeypatch.context() as patch:
        patch.setattr(
            taskboard_store,
            "commit_taskboard_materialization",
            interrupted_board_commit,
        )
        with pytest.raises(RuntimeError, match="process interruption"):
            run_v2_refill_epoch(
                **_kwargs(paths),
                observation_provider=lambda _binding: V2RefillObservation(
                    residuals=(residual,)
                ),
                proposal_provider=lambda *_args, **_kwargs: admission,
            )

    assert paths["objective_path"].read_bytes() != objective_before
    assert paths["taskboard_path"].read_bytes() == board_before
    interrupted = next(
        value
        for value in _journal(paths)["transactions"].values()
        if value["state"] == "objective_committed"
    )

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        pytest.fail("durable recovery must not rerun provider or proposal work")

    recovered = run_v2_refill_epoch(
        **_kwargs(paths),
        observation_provider=forbidden,
        proposal_provider=forbidden,
    )

    assert _status_value(recovered) == "proposed"
    assert recovered.epoch_id == interrupted["epoch_id"]
    assert recovered.created_goal_ids == tuple(interrupted["goal_ids"])
    assert recovered.created_task_ids == tuple(interrupted["task_ids"])
    assert paths["taskboard_path"].read_text(encoding="utf-8") == (
        interrupted["preview"]["candidate_board_text"]
    )
    assert _transaction(
        _journal(paths), recovered.epoch_id
    )["state"] == "committed"


def test_exact_replay_does_zero_provider_proposal_write_or_task_work(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    residual = _residual()
    first = run_v2_refill_epoch(
        **_kwargs(paths),
        observation_provider=lambda _binding: V2RefillObservation(
            residuals=(residual,)
        ),
    )
    assert _status_value(first) == "proposed"
    tracked = (
        paths["objective_path"],
        paths["taskboard_path"],
        paths["journal_path"],
        paths["wait_state_path"],
    )
    before = {
        path: (
            path.read_bytes() if path.exists() else None,
            path.stat().st_mtime_ns if path.exists() else None,
        )
        for path in tracked
    }

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        pytest.fail("exact replay must return before callback or task work")

    replay = run_v2_refill_epoch(
        **_kwargs(paths, now=NOW + timedelta(minutes=5)),
        observation_provider=forbidden,
        proposal_provider=forbidden,
    )

    assert replay.replayed
    assert replay.epoch_id == first.epoch_id
    assert replay.receipt_id == first.receipt_id
    assert replay.created_goal_ids == first.created_goal_ids
    assert replay.created_task_ids == first.created_task_ids
    assert {
        path: (
            path.read_bytes() if path.exists() else None,
            path.stat().st_mtime_ns if path.exists() else None,
        )
        for path in tracked
    } == before


@pytest.mark.parametrize(
    "defect",
    [
        "under-count",
        "duplicate-producer",
        "duplicate-channel",
        "duplicate-receipt",
        "stale",
        "unhealthy",
        "non-exhaustive",
        "foreign-binding",
    ],
)
def test_no_candidate_requires_independent_fresh_healthy_exhaustive_quorum(
    tmp_path: Path,
    defect: str,
) -> None:
    paths = _paths(tmp_path)

    def observe(binding: V2RefillEpochBinding) -> V2RefillObservation:
        first = _receipt(binding, "first")
        second = _receipt(binding, "second")
        if defect == "under-count":
            receipts = (first,)
        else:
            if defect == "duplicate-producer":
                second["producer_id"] = first["producer_id"]
            elif defect == "duplicate-channel":
                second["evidence_channel"] = first["evidence_channel"]
            elif defect == "duplicate-receipt":
                second["receipt_id"] = first["receipt_id"]
            elif defect == "stale":
                second["fresh_until"] = (NOW - timedelta(seconds=1)).isoformat()
            elif defect == "unhealthy":
                second["healthy"] = False
            elif defect == "non-exhaustive":
                second["exhaustive"] = False
            else:
                second["binding"] = {
                    **second["binding"],
                    "tree_id": "tree:sha256:foreign",
                }
            receipts = (first, second)
        return V2RefillObservation(
            residuals=(),
            exhaustion_receipts=receipts,
        )

    result = run_v2_refill_epoch(
        **_kwargs(paths),
        observation_provider=observe,
    )

    assert _status_value(result) == "rejected"
    assert not result.created_goal_ids
    assert not result.created_task_ids
    assert not paths["wait_state_path"].exists()
    assert "healthy_exhaustion_quorum_unsatisfied" in result.reason_codes


def test_healthy_exhaustion_persists_six_hour_wait_and_suppresses_work(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    calls: list[str] = []

    def observe(binding: V2RefillEpochBinding) -> V2RefillObservation:
        calls.append(binding.epoch_id)
        return _healthy_observation(binding)

    first = run_v2_refill_epoch(
        **_kwargs(paths),
        observation_provider=observe,
    )

    assert _status_value(first) == "healthy_exhaustion"
    assert calls == [first.epoch_id]
    wait = json.loads(paths["wait_state_path"].read_text(encoding="utf-8"))
    assert wait["epoch_id"] == first.epoch_id
    assert wait["state"] == "waiting_for_meaningful_trigger"
    assert wait["observed_at"] == NOW.isoformat()
    assert wait["suppress_until"] == (
        NOW + timedelta(hours=6)
    ).isoformat()
    assert wait["quorum"]["satisfied"]
    assert wait["quorum"]["member_count"] == 2

    suppressed = run_v2_refill_epoch(
        **_kwargs(
            paths,
            observation_window_start=NOW,
            observation_window_end=NOW + timedelta(hours=1),
            now=NOW + timedelta(hours=1),
        ),
        observation_provider=lambda _binding: pytest.fail(
            "unchanged triggers inside the wait must not run a provider"
        ),
    )

    assert suppressed.suppressed
    assert not suppressed.replayed
    assert suppressed.previous_epoch_id == first.epoch_id
    assert calls == [first.epoch_id]

    after_wait = run_v2_refill_epoch(
        **_kwargs(
            paths,
            observation_window_start=NOW + timedelta(hours=6),
            observation_window_end=NOW + timedelta(hours=7),
            now=NOW + timedelta(hours=7),
        ),
        observation_provider=observe,
    )
    assert _status_value(after_wait) == "healthy_exhaustion"
    assert len(calls) == 2


def test_declared_meaningful_trigger_bypasses_wait_before_six_hours(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    calls: list[V2RefillEpochBinding] = []

    def observe(binding: V2RefillEpochBinding) -> V2RefillObservation:
        calls.append(binding)
        return _healthy_observation(binding)

    first = run_v2_refill_epoch(
        **_kwargs(paths),
        observation_provider=observe,
    )
    changed = run_v2_refill_epoch(
        **_kwargs(
            paths,
            capability_revision="sha256:capabilities-v8",
            observation_window_start=NOW,
            observation_window_end=NOW + timedelta(hours=1),
            now=NOW + timedelta(hours=1),
        ),
        observation_provider=observe,
    )

    assert len(calls) == 2
    assert changed.epoch_id != first.epoch_id
    assert not changed.suppressed
    assert changed.meaningful_trigger == "capability_revision_changed"
    assert _status_value(changed) == "healthy_exhaustion"
