"""Race and replay properties for fenced effectful task transitions."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from threading import Event, RLock, Thread
from typing import Callable, TypeVar

import pytest

from ipfs_accelerate_py.agent_supervisor.control.task_transition_service import (
    TaskLeaseFence,
    TaskTransitionService,
    TransitionConflictError,
    TransitionIdempotencyConflictError,
    TransitionLeaseError,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import (
    CommandKind,
    StateCommand,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository


_ResultT = TypeVar("_ResultT")


class _SerializedLeaseAuthority:
    """Small deterministic authority double with a real takeover race boundary."""

    def __init__(self, lease: TaskLeaseFence) -> None:
        self._lock = RLock()
        self.current = lease

    def execute_fenced(
        self, lease: TaskLeaseFence, callback: Callable[[], _ResultT]
    ) -> _ResultT:
        # Holding this lock during the callback is the relevant property: a
        # takeover either precedes the effect (which rejects it) or follows it
        # (which cannot retroactively authorize a stale completion).
        with self._lock:
            if lease != self.current:
                raise TransitionLeaseError("stale lease/fence cannot execute or complete")
            return callback()

    def takeover(self, owner_session_id: str) -> TaskLeaseFence:
        with self._lock:
            self.current = replace(
                self.current,
                owner_session_id=owner_session_id,
                lease_id=f"lease:{owner_session_id}",
                fencing_token=self.current.fencing_token + 1,
                fence_epoch=self.current.fence_epoch + 1,
                claim_revision=self.current.claim_revision + 1,
            )
            return self.current


def _seed_service(tmp_path: Path) -> tuple[TaskTransitionService, _SerializedLeaseAuthority]:
    repository = IntentRepository(tmp_path / "intent.duckdb")
    repository.upsert_goal(
        goal_cid="goal:fence",
        goal_alias="GOAL-FENCE",
        objective_id="objective:fence",
        title="fence",
    )
    repository.upsert_task(
        task_cid="task:fence",
        task_alias="TASK-FENCE",
        goal_cid="goal:fence",
        status="ready",
    )
    repository.record_validation_result(
        task_cid="task:fence",
        outcome="passed",
        evidence_digest="sha256:" + ("ab" * 32),
        argv=["pytest"],
    )
    authority = _SerializedLeaseAuthority(
        TaskLeaseFence(
            task_cid="task:fence",
            owner_session_id="owner:one",
            lease_id="lease:one",
            fencing_token=1,
            fence_epoch=1,
            claim_revision=1,
        )
    )
    return TaskTransitionService(repository, lease_fence_authority=authority), authority


def _command(
    lease: TaskLeaseFence,
    *,
    command_id: str,
    idempotency_key: str,
    status: str = "in_progress",
    revision: int = 1,
    evidence_digests: tuple[str, ...] = (),
) -> StateCommand:
    return StateCommand(
        command_id=command_id,
        command_kind=CommandKind.APPEND,
        store_id="store:fence",
        session_id=lease.owner_session_id,
        expected_generation=1,
        expected_revision=revision,
        fence_epoch=lease.fence_epoch,
        idempotency_key=idempotency_key,
        parameters={
            "task_cid": lease.task_cid,
            "new_status": status,
            "lease_id": lease.lease_id,
            "fencing_token": lease.fencing_token,
            "claim_revision": lease.claim_revision,
            "evidence_digests": evidence_digests,
        },
    )


def test_duplicate_idempotent_delivery_executes_the_effect_exactly_once(tmp_path: Path) -> None:
    service, authority = _seed_service(tmp_path)
    first = _command(
        authority.current,
        command_id="command:effect:first",
        idempotency_key="idempotency:effect",
    )
    duplicate = _command(
        authority.current,
        command_id="command:effect:duplicate",
        idempotency_key="idempotency:effect",
    )
    entered = Event()
    release = Event()
    effects: list[str] = []
    results: list[str] = []

    def effect() -> str:
        effects.append("executed")
        entered.set()
        assert release.wait(timeout=5)
        return "effect-receipt"

    first_thread = Thread(target=lambda: results.append(service.execute_effect(first, effect)))
    second_thread = Thread(target=lambda: results.append(service.execute_effect(duplicate, effect)))
    first_thread.start()
    assert entered.wait(timeout=5)
    second_thread.start()
    release.set()
    first_thread.join(timeout=5)
    second_thread.join(timeout=5)

    assert not first_thread.is_alive()
    assert not second_thread.is_alive()
    assert effects == ["executed"]
    assert results == ["effect-receipt", "effect-receipt"]


def test_takeover_advances_fences_and_stale_completion_cannot_mutate(tmp_path: Path) -> None:
    service, authority = _seed_service(tmp_path)
    stale = authority.current
    replacement = authority.takeover("owner:two")

    assert replacement.fencing_token == stale.fencing_token + 1
    assert replacement.fence_epoch == stale.fence_epoch + 1
    assert replacement.claim_revision == stale.claim_revision + 1

    with pytest.raises(TransitionLeaseError, match="stale lease/fence"):
        service.transition(
            _command(
                stale,
                command_id="command:stale-completion",
                idempotency_key="idempotency:stale-completion",
                status="completed",
                evidence_digests=("sha256:" + ("ab" * 32),),
            )
        )

    unchanged = service.repository.get_task("task:fence")
    assert unchanged is not None
    assert unchanged["status"] == "ready"
    assert unchanged["revision"] == 1

    completed = service.transition(
        _command(
            replacement,
            command_id="command:current-completion",
            idempotency_key="idempotency:current-completion",
            status="completed",
            evidence_digests=("sha256:" + ("ab" * 32),),
        )
    )
    assert completed.changed is True
    assert completed.status == "completed"
    assert completed.revision == 2


def test_racing_takeover_is_a_linearization_boundary_for_effects_and_completion(
    tmp_path: Path,
) -> None:
    service, authority = _seed_service(tmp_path)
    initial = authority.current
    entered_effect = Event()
    release_effect = Event()
    effect_done = Event()
    takeover_done = Event()
    effects: list[str] = []

    def effect() -> str:
        effects.append("one")
        entered_effect.set()
        assert release_effect.wait(timeout=5)
        effect_done.set()
        return "receipt:one"

    effect_thread = Thread(
        target=lambda: service.execute_effect(
            _command(
                initial,
                command_id="command:race-effect",
                idempotency_key="idempotency:race-effect",
            ),
            effect,
        )
    )
    effect_thread.start()
    assert entered_effect.wait(timeout=5)

    replacement: list[TaskLeaseFence] = []
    takeover_thread = Thread(
        target=lambda: (replacement.append(authority.takeover("owner:two")), takeover_done.set())
    )
    takeover_thread.start()
    # The old effect has the authority lock, so takeover cannot interleave
    # between validation and the effect callback.
    assert not takeover_done.wait(timeout=0.1)
    release_effect.set()
    effect_thread.join(timeout=5)
    takeover_thread.join(timeout=5)
    assert effect_done.is_set()
    assert takeover_done.is_set()
    assert effects == ["one"]

    with pytest.raises(TransitionLeaseError, match="stale lease/fence"):
        service.transition(
            _command(
                initial,
                command_id="command:race-stale-completion",
                idempotency_key="idempotency:race-stale-completion",
                status="completed",
                evidence_digests=("sha256:" + ("ab" * 32),),
            )
        )

    task = service.repository.get_task("task:fence")
    assert task is not None
    assert task["status"] == "ready"
    assert task["revision"] == 1
    assert replacement == [authority.current]


def test_idempotency_key_cannot_be_rebound_after_takeover(tmp_path: Path) -> None:
    service, authority = _seed_service(tmp_path)
    original = authority.current
    service.execute_effect(
        _command(
            original,
            command_id="command:original",
            idempotency_key="idempotency:fixed",
        ),
        lambda: "first",
    )
    replacement = authority.takeover("owner:two")

    with pytest.raises(TransitionIdempotencyConflictError, match="already bound"):
        service.execute_effect(
            _command(
                replacement,
                command_id="command:replacement",
                idempotency_key="idempotency:fixed",
            ),
            lambda: "must-not-run",
        )


def test_effectful_execution_rejects_a_stale_task_revision_before_the_callback(
    tmp_path: Path,
) -> None:
    service, authority = _seed_service(tmp_path)
    # Another admitted CAS advances the task before this effect can begin.
    service.repository.cas_task_status(
        task_cid="task:fence",
        expected_revision=1,
        new_status="in_progress",
    )
    executed: list[bool] = []

    with pytest.raises(TransitionConflictError, match="effectful execution CAS conflict"):
        service.execute_effect(
            _command(
                authority.current,
                command_id="command:stale-revision",
                idempotency_key="idempotency:stale-revision",
                revision=1,
            ),
            lambda: executed.append(True),
        )
    assert executed == []
