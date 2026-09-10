"""Authority properties of the candidate repository-backed transition service."""

from __future__ import annotations

from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.control.task_transition_service import (
    PRODUCTION_CUTOVER_DEFERRED_TO,
    TaskTransitionService,
    TransitionAuthorityWarning,
    TransitionCompatibilityBypassError,
    TransitionConflictError,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import (
    CommandKind,
    StateCommand,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository


def _command(*, revision: int, status: str = "in_progress") -> StateCommand:
    return StateCommand(
        command_id=f"command:transition:{revision}:{status}",
        command_kind=CommandKind.APPEND,
        store_id="store:candidate",
        session_id="session:candidate",
        expected_generation=1,
        expected_revision=revision,
        fence_epoch=0,
        idempotency_key=f"idempotency:transition:{revision}:{status}",
        parameters={"task_cid": "task:authority", "new_status": status},
    )


@pytest.fixture()
def service(tmp_path: Path) -> TaskTransitionService:
    repository = IntentRepository(tmp_path / "intent.duckdb")
    repository.upsert_goal(
        goal_cid="goal:authority",
        goal_alias="GOAL-AUTHORITY",
        objective_id="objective:authority",
        title="authority",
    )
    repository.upsert_task(
        task_cid="task:authority",
        task_alias="TASK-AUTHORITY",
        goal_cid="goal:authority",
        status="ready",
    )
    return TaskTransitionService(repository)


def test_transition_delegates_to_repository_cas_and_keeps_event_revision_in_sync(
    service: TaskTransitionService,
) -> None:
    result = service.transition(_command(revision=1))

    assert result.changed is True
    assert result.previous_status == "ready"
    assert result.status == "in_progress"
    assert result.revision == result.receipt.revision == result.task["revision"] == 2
    assert result.receipt.event_id
    event = service.repository.list_events(
        after_global_sequence=result.receipt.global_sequence - 1, limit=1
    )[0]
    assert event["event_id"] == result.receipt.event_id
    assert event["body"]["body"]["revision"] == result.task["revision"]
    assert PRODUCTION_CUTOVER_DEFERRED_TO == ("ASEH-060", "ASEH-061")


def test_stale_command_fails_closed_without_a_retry(service: TaskTransitionService) -> None:
    service.transition(_command(revision=1))

    with pytest.raises(TransitionConflictError, match="CAS conflict"):
        service.transition(_command(revision=1, status="blocked"))

    task = service.repository.get_task("task:authority")
    assert task is not None
    assert task["status"] == "in_progress"
    assert task["revision"] == 2


def test_legacy_compatibility_bypass_warns_then_fails_closed(service: TaskTransitionService) -> None:
    with pytest.warns(TransitionAuthorityWarning, match="compatibility bypass rejected"):
        with pytest.raises(TransitionCompatibilityBypassError, match="not an admitted"):
            service.transition_legacy(caller="legacy-adapter")

    task = service.repository.get_task("task:authority")
    assert task is not None
    assert task["status"] == "ready"
    assert task["revision"] == 1
