"""Actual native owner reads remain live while the main worker is occupied.

These tests require the installed Quack transport; they never replace transport,
generation reads, credentials, process births, or task state with test doubles.
"""

from __future__ import annotations

from dataclasses import replace
import json
import threading
import time

import pytest

from benchmarks.agent_supervisor.container_coding.native_quack_qualification import (
    native_owner_session,
)
from ipfs_accelerate_py.agent_supervisor.merge.database_worktree_registry import (
    process_birth_id,
)
from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
    read_process_birth,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.native_owner_bootstrap import (
    NativeOwnerHeartbeat,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.quack_state_client import (
    QuackClientTransportError,
)


def _until(predicate, seconds=3):
    deadline = time.monotonic() + seconds
    wake = threading.Event()
    while time.monotonic() < deadline:
        if predicate():
            return
        wake.wait(0.05)
    assert predicate(), "native heartbeat condition did not occur within its bound"


def test_actual_owner_heartbeat_advances_while_main_native_client_is_busy(tmp_path):
    import os

    with native_owner_session(tmp_path / "owner") as session:
        before = session.source.get_task(session.task_cid)
        heartbeat = NativeOwnerHeartbeat(
            credentials=session.credentials,
            state_dir=tmp_path / "state",
            state_prefix="admitted",
            interval_seconds=0.1,
        ).start()
        try:
            initial = json.loads(heartbeat.path.read_text())
            # Occupy the actual main native client's request lock. The separate
            # heartbeat connection must still issue real generation reads.
            with session.client._lock:
                threading.Event().wait(0.35)
                _until(lambda: heartbeat.sequence >= initial["sequence"] + 2)
                observed = json.loads(heartbeat.path.read_text())
            assert observed["sequence"] > initial["sequence"]
            assert observed["observed_at_ms"] > initial["observed_at_ms"]
            birth = read_process_birth(os.getpid())
            assert birth is not None
            assert observed["process_birth"] == birth.to_dict()
            assert observed["process_birth_id"] == process_birth_id(birth)
            assert observed["store_id"] == session.identity.store_id
            assert observed["server_id"] == session.identity.server_id
            assert observed["owner_read_succeeded"] is True
            assert observed["dispatch_authority"] is False
            assert observed["completion_authority"] is False
            assert observed["task_progress_authority"] is False
            assert session.credentials.token not in heartbeat.path.read_text()
            after = session.source.get_task(session.task_cid)
            assert after.status == before.status == "ready"
            assert after.revision == before.revision
            assert after.body == before.body
        finally:
            heartbeat.close()
        saved = heartbeat.path.read_bytes()
        threading.Event().wait(0.25)
        assert not heartbeat.thread.is_alive()
        assert heartbeat.path.read_bytes() == saved


def test_stopped_actual_owner_cannot_refresh_last_live_observation(tmp_path):
    with native_owner_session(tmp_path / "owner") as session:
        heartbeat = NativeOwnerHeartbeat(
            credentials=session.credentials,
            state_dir=tmp_path / "state",
            state_prefix="admitted",
            interval_seconds=0.1,
        ).start()
        try:
            _until(lambda: heartbeat.sequence >= 2)
            session.server.stop()
            _until(lambda: bool(heartbeat.last_error))
            last_live = heartbeat.path.read_bytes()
            sequence = heartbeat.sequence
            threading.Event().wait(0.3)
            assert heartbeat.sequence == sequence
            assert heartbeat.path.read_bytes() == last_live
            assert json.loads(last_live)["observed_at_ms"] < time.time_ns() // 1_000_000
            assert heartbeat.last_error
        finally:
            heartbeat.close()


def test_real_native_grant_rejects_foreign_process_birth_for_heartbeat(tmp_path):
    with native_owner_session(tmp_path / "owner") as session:
        foreign = replace(session.credentials, process_birth_id="foreign-process-birth")
        heartbeat = NativeOwnerHeartbeat(
            credentials=foreign,
            state_dir=tmp_path / "state",
            state_prefix="admitted",
            interval_seconds=0.1,
        )
        try:
            with pytest.raises(QuackClientTransportError, match="failed to attach"):
                heartbeat.start()
            assert heartbeat.sequence == 0
            assert not heartbeat.path.exists()
        finally:
            heartbeat.close()
        assert session.source.get_task(session.task_cid).status == "ready"
