import os
import pytest

from ipfs_accelerate_py.p2p_tasks.worker import _copilot_session_controls_allowed


def test_copilot_session_requires_sticky_worker_id() -> None:
    payload = {"continue_session": True}
    with pytest.raises(RuntimeError, match="sticky_worker_id"):
        _copilot_session_controls_allowed(payload=payload, local_session="S1", assigned_worker_id="w1")


def test_copilot_session_requires_sticky_matches_assigned_worker() -> None:
    payload = {"continue_session": True, "sticky_worker_id": "w2", "session_id": "S1"}
    with pytest.raises(RuntimeError, match="sticky_worker_id to match assigned_worker"):
        _copilot_session_controls_allowed(payload=payload, local_session="S1", assigned_worker_id="w1")


def test_copilot_session_requires_session_id_when_local_session_set() -> None:
    payload = {"continue_session": True, "sticky_worker_id": "w1"}
    with pytest.raises(RuntimeError, match="requires a session_id"):
        _copilot_session_controls_allowed(payload=payload, local_session="S1", assigned_worker_id="w1")


def test_copilot_session_requires_matching_session_id_when_local_session_set() -> None:
    payload = {"continue_session": True, "sticky_worker_id": "w1", "session_id": "S2"}
    with pytest.raises(RuntimeError, match="requires matching session_id"):
        _copilot_session_controls_allowed(payload=payload, local_session="S1", assigned_worker_id="w1")


def test_copilot_continue_without_resume_disallowed_by_default() -> None:
    payload = {"continue_session": True, "sticky_worker_id": "w1", "session_id": "S1"}
    with pytest.raises(RuntimeError, match="disallows continue_session without resume_session_id"):
        _copilot_session_controls_allowed(payload=payload, local_session="S1", assigned_worker_id="w1")


def test_copilot_continue_without_resume_allowed_when_opted_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_PY_TASK_WORKER_ALLOW_COPILOT_CONTINUE_WITHOUT_RESUME", "1")
    payload = {"continue_session": True, "sticky_worker_id": "w1", "session_id": "S1"}
    _copilot_session_controls_allowed(payload=payload, local_session="S1", assigned_worker_id="w1")


def test_copilot_resume_allowed_with_resume_session_id(monkeypatch: pytest.MonkeyPatch) -> None:
    # Resume path should not require the allow-continue-without-resume opt-in.
    monkeypatch.delenv("IPFS_ACCELERATE_PY_TASK_WORKER_ALLOW_COPILOT_CONTINUE_WITHOUT_RESUME", raising=False)
    payload = {"resume_session_id": "R1", "sticky_worker_id": "w1", "session_id": "S1"}
    _copilot_session_controls_allowed(payload=payload, local_session="S1", assigned_worker_id="w1")
