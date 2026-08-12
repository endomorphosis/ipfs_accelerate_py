"""SCH-012 incremental session, watch, restart, and replay tests."""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
import threading
import time
import types
from pathlib import Path
from typing import Any, Mapping

import pytest

from ipfs_accelerate_py.mcp_server.mcplusplus.kubo_cid import cid_for_bytes

from ipfs_accelerate_py.agent_supervisor.semantic_state.contracts import (
    HarnessMode,
    RootRef,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.scheduling_contracts import (
    CancellationToken,
)


def _cid(label: str) -> str:
    return cid_for_bytes(label.encode("utf-8"))


class MemoryDurablePort:
    """Hermetic durable port with generation-bearing CAS."""

    def __init__(self) -> None:
        self._objects: dict[str, dict[str, Any]] = {}
        self._roots: dict[str, RootRef] = {}
        self.cas_calls: list[tuple[str, RootRef | None, str]] = []
        self.recover_calls = 0

    def put(
        self,
        artifact: Mapping[str, Any],
        *,
        expected_cid: str,
        codec: str = "dag-json",
    ) -> Mapping[str, Any]:
        self._objects[expected_cid] = dict(artifact)
        return {"cid": expected_cid}

    def get(self, cid: str) -> Mapping[str, Any]:
        return dict(self._objects[cid])

    def get_bytes(self, cid: str) -> bytes:
        return json.dumps(self._objects[cid], sort_keys=True).encode("utf-8")

    def has(self, cid: str) -> bool:
        return cid in self._objects

    def read_root(self, repository_id: str) -> RootRef | None:
        return self._roots.get(repository_id)

    def compare_and_swap_root(
        self,
        repository_id: str,
        expected: RootRef | None,
        new_root_cid: str,
    ) -> RootRef:
        self.cas_calls.append((repository_id, expected, new_root_cid))
        current = self._roots.get(repository_id)
        if expected is None:
            if current is not None:
                from ipfs_accelerate_py.agent_supervisor.semantic_state.durable_state import (
                    RootConflict,
                )

                raise RootConflict("expected empty root")
            published = RootRef(root_cid=new_root_cid, generation=1)
        else:
            if (
                current is None
                or current.root_cid != expected.root_cid
                or current.generation != expected.generation
            ):
                from ipfs_accelerate_py.agent_supervisor.semantic_state.durable_state import (
                    RootConflict,
                )

                raise RootConflict("stale expected root")
            published = RootRef(
                root_cid=new_root_cid, generation=expected.generation + 1
            )
        self._roots[repository_id] = published
        return published

    def recover(self) -> Mapping[str, Any]:
        self.recover_calls += 1
        return {"recovered": True, "roots": len(self._roots)}


def _policy(tmp_path: Path, **overrides: object):
    session_mod = importlib.import_module(
        "ipfs_accelerate_py.agent_supervisor.semantic_state.session"
    )
    payload: dict[str, object] = {
        "repository_id": "example/repo",
        "event_log_path": tmp_path / "session-events.jsonl",
        "checkpoint_path": tmp_path / "session-cursor.json",
        "mode": HarnessMode.DEVELOPMENT.value,
        "debounce_ms": 0,
        "fence_ttl_ms": 60_000,
        "worker_enabled": False,
        "fail_closed_on_corrupt_log": True,
    }
    payload.update(overrides)
    return session_mod.SessionPolicy.from_dict(payload)


def _session(tmp_path: Path, **kwargs: Any):
    session_mod = importlib.import_module(
        "ipfs_accelerate_py.agent_supervisor.semantic_state.session"
    )
    policy = kwargs.pop("policy", None) or _policy(tmp_path)
    return session_mod.SemanticStateSession(policy, **kwargs)


# ---------------------------------------------------------------------------
# Cold import / descriptor
# ---------------------------------------------------------------------------


def test_cold_import_starts_no_resources_threads_processes_or_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = "ipfs_accelerate_py.agent_supervisor.semantic_state.session"
    sys.modules.pop(module_name, None)

    before_threads = {t.ident for t in threading.enumerate()}
    real_thread_start = threading.Thread.start
    started_threads: list[str] = []

    def guarded_start(self: threading.Thread, *args: Any, **kwargs: Any) -> None:
        started_threads.append(self.name)
        return real_thread_start(self, *args, **kwargs)

    monkeypatch.setattr(threading.Thread, "start", guarded_start)

    real_popen = subprocess.Popen
    popen_calls: list[Any] = []

    def guarded_popen(*args: Any, **kwargs: Any):
        popen_calls.append((args, kwargs))
        raise AssertionError("cold import must not spawn subprocesses")

    monkeypatch.setattr(subprocess, "Popen", guarded_popen)

    socket_mod = importlib.import_module("socket")
    real_socket = socket_mod.socket
    socket_calls: list[Any] = []

    class GuardedSocket(real_socket):  # type: ignore[misc,valid-type]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            socket_calls.append((args, kwargs))
            raise AssertionError("cold import must not open sockets")

    monkeypatch.setattr(socket_mod, "socket", GuardedSocket)

    fake_duckdb = types.ModuleType("duckdb")

    def no_connect(*args: Any, **kwargs: Any):
        raise AssertionError("cold import must not open databases")

    fake_duckdb.connect = no_connect  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "duckdb", fake_duckdb)

    mod = importlib.import_module(module_name)
    assert mod.SESSION_INTERFACE == "SemanticStateSession@1"
    assert started_threads == []
    assert popen_calls == []
    assert socket_calls == []
    after_threads = {t.ident for t in threading.enumerate()}
    assert after_threads - before_threads == set()


def test_descriptor_declares_session_invariants() -> None:
    mod = importlib.import_module(
        "ipfs_accelerate_py.agent_supervisor.semantic_state.session"
    )
    desc = mod.semantic_state_session_descriptor()
    assert desc["interface"] == "SemanticStateSession@1"
    assert desc["bundle"] == "sch/session@1"
    assert "SemanticStateSession" in desc["symbols"]
    assert "watch_session" in desc["symbols"]
    assert "replay_session" in desc["symbols"]
    assert "runtime.event_log" in desc["composes"]
    assert "concurrent_equal_snapshot_cids_coalesce" in desc["invariants"]
    assert "explicit_shutdown_cancels_and_joins_owned_work" in desc["invariants"]
    assert "publish_from_watch_event_alone" in desc["forbids"]


# ---------------------------------------------------------------------------
# Watch coalescing
# ---------------------------------------------------------------------------


def test_concurrent_equal_snapshot_cids_coalesce_without_duplicate_work(
    tmp_path: Path,
) -> None:
    mod = importlib.import_module(
        "ipfs_accelerate_py.agent_supervisor.semantic_state.session"
    )
    calls: list[str] = []
    barrier = threading.Barrier(4)
    snap = _cid("snap-equal")

    def executor(**kwargs: Any) -> Mapping[str, Any]:
        calls.append(kwargs["snapshot_cid"])
        return {
            "status": "completed",
            "output_artifact_cids": [_cid("scan-out")],
            "verified": False,
        }

    session = _session(tmp_path, scan_executor=executor)
    acks: list[Any] = []
    errors: list[BaseException] = []

    def notify() -> None:
        try:
            barrier.wait(timeout=2)
            acks.append(session.notify_watch(snap, source="watcher"))
        except BaseException as exc:  # pragma: no cover - test harness
            errors.append(exc)

    threads = [threading.Thread(target=notify) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=3)
    assert errors == []
    assert len(acks) == 4
    scheduled = [ack for ack in acks if ack.scheduled]
    coalesced = [ack for ack in acks if ack.coalesced]
    assert len(scheduled) == 1
    assert len(coalesced) == 3
    results = session.drain()
    assert len(results) == 1
    assert results[0].snapshot_cid == snap
    assert calls == [snap]
    status = session.status()
    assert status.scans_completed == 1
    assert status.scans_coalesced >= 3


def test_distinct_snapshots_serialize_and_do_not_overwrite_roots(
    tmp_path: Path,
) -> None:
    durable = MemoryDurablePort()
    session_mod = importlib.import_module(
        "ipfs_accelerate_py.agent_supervisor.semantic_state.session"
    )
    order: list[str] = []
    lock = threading.Lock()

    def executor(**kwargs: Any) -> Mapping[str, Any]:
        with lock:
            order.append(kwargs["snapshot_cid"])
        return {
            "status": "completed",
            "new_root_cid": _cid(f"root-for-{kwargs['snapshot_cid'][-8:]}"),
            "verified": True,
            "output_artifact_cids": [_cid("out")],
        }

    session = _session(tmp_path, durable_port=durable, scan_executor=executor)
    snap_a = _cid("snap-a")
    snap_b = _cid("snap-b")
    ack_a = session.notify_watch(snap_a)
    ack_b = session.notify_watch(snap_b)
    assert ack_a.scheduled and ack_b.scheduled
    assert not ack_a.coalesced and not ack_b.coalesced

    results = session.drain()
    assert len(results) == 2
    assert set(order) == {snap_a, snap_b}

    # Publish first result under its fence; second must use updated expected.
    first = results[0]
    fence = session.fence_for(first.attempt_id)
    assert fence is not None
    published = session.accept_transition(
        attempt_id=first.attempt_id,
        fencing_token=fence.fencing_token,
        new_root_cid=first.new_root_cid or _cid("root-1"),
        expected=None,
        verified=True,
    )
    assert published.generation == 1
    assert durable.read_root("example/repo") == published

    second = results[1]
    fence2 = session.fence_for(second.attempt_id)
    assert fence2 is not None
    # Stale expected token must not overwrite.
    with pytest.raises(session_mod.SessionRootConflict):
        session.accept_transition(
            attempt_id=second.attempt_id,
            fencing_token=fence2.fencing_token,
            new_root_cid=second.new_root_cid or _cid("root-2"),
            expected=None,  # empty token after bootstrap is stale
            verified=True,
        )
    assert durable.read_root("example/repo") == published
    # Correct expected succeeds without losing the prior transition.
    published2 = session.accept_transition(
        attempt_id=second.attempt_id,
        fencing_token=fence2.fencing_token,
        new_root_cid=second.new_root_cid or _cid("root-2"),
        expected=published,
        verified=True,
    )
    assert published2.generation == 2
    assert durable.read_root("example/repo") == published2


def test_watch_notification_does_not_publish_root_by_itself(tmp_path: Path) -> None:
    durable = MemoryDurablePort()
    session = _session(tmp_path, durable_port=durable)
    session.notify_watch(_cid("snap-watch-only"))
    assert durable.read_root("example/repo") is None
    assert durable.cas_calls == []
    assert session.status().current_root is None


# ---------------------------------------------------------------------------
# Fence / unverified publication gates
# ---------------------------------------------------------------------------


def test_stale_and_expired_fence_cannot_publish(tmp_path: Path) -> None:
    session_mod = importlib.import_module(
        "ipfs_accelerate_py.agent_supervisor.semantic_state.session"
    )
    clock = {"now": 1_000_000}

    def executor(**kwargs: Any) -> Mapping[str, Any]:
        return {
            "status": "completed",
            "new_root_cid": _cid("root-stale"),
            "verified": True,
        }

    session = _session(
        tmp_path,
        scan_executor=executor,
        clock_ms=lambda: clock["now"],
        policy=_policy(tmp_path, fence_ttl_ms=1_000),
    )
    session.notify_watch(_cid("snap-fence"))
    result = session.drain()[0]
    fence = session.fence_for(result.attempt_id)
    assert fence is not None

    with pytest.raises(session_mod.SessionRootPublishDenied, match="stale"):
        session.accept_transition(
            attempt_id=result.attempt_id,
            fencing_token=fence.fencing_token + 99,
            new_root_cid=_cid("root-stale"),
            verified=True,
        )

    clock["now"] = fence.expires_at_ms + 1
    with pytest.raises(session_mod.SessionRootPublishDenied, match="expired"):
        session.accept_transition(
            attempt_id=result.attempt_id,
            fencing_token=fence.fencing_token,
            new_root_cid=_cid("root-stale"),
            verified=True,
        )


def test_unverified_transition_cannot_publish_root(tmp_path: Path) -> None:
    session_mod = importlib.import_module(
        "ipfs_accelerate_py.agent_supervisor.semantic_state.session"
    )
    durable = MemoryDurablePort()

    def executor(**kwargs: Any) -> Mapping[str, Any]:
        return {
            "status": "completed",
            "new_root_cid": _cid("root-candidate"),
            "verified": False,
        }

    session = _session(tmp_path, durable_port=durable, scan_executor=executor)
    session.notify_watch(_cid("snap-candidate"))
    result = session.drain()[0]
    fence = session.fence_for(result.attempt_id)
    assert fence is not None
    with pytest.raises(session_mod.SessionRootPublishDenied, match="unverified"):
        session.accept_transition(
            attempt_id=result.attempt_id,
            fencing_token=fence.fencing_token,
            new_root_cid=_cid("root-candidate"),
            verified=False,
        )
    assert durable.read_root("example/repo") is None
    assert durable.cas_calls == []

    # Candidate event is journaled.
    events = [
        json.loads(line)
        for line in Path(session.policy.event_log_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    types = {event["type"] for event in events}
    assert "session_transition_candidate" in types


# ---------------------------------------------------------------------------
# Restart / replay
# ---------------------------------------------------------------------------


def test_restart_preserves_accepted_and_does_not_publish_unverified(
    tmp_path: Path,
) -> None:
    durable = MemoryDurablePort()
    session_mod = importlib.import_module(
        "ipfs_accelerate_py.agent_supervisor.semantic_state.session"
    )

    def executor(**kwargs: Any) -> Mapping[str, Any]:
        return {
            "status": "completed",
            "new_root_cid": _cid("root-accepted"),
            "verified": True,
        }

    session = _session(tmp_path, durable_port=durable, scan_executor=executor)
    session.notify_watch(_cid("snap-accept"))
    result = session.drain()[0]
    fence = session.fence_for(result.attempt_id)
    assert fence is not None
    published = session.accept_transition(
        attempt_id=result.attempt_id,
        fencing_token=fence.fencing_token,
        new_root_cid=_cid("root-accepted"),
        expected=None,
        verified=True,
    )
    assert published.generation == 1

    # Journal an unverified candidate after acceptance.
    session.notify_watch(_cid("snap-unverified"))
    cand = session.drain()[0]
    cand_fence = session.fence_for(cand.attempt_id)
    assert cand_fence is not None
    with pytest.raises(session_mod.SessionRootPublishDenied):
        session.accept_transition(
            attempt_id=cand.attempt_id,
            fencing_token=cand_fence.fencing_token,
            new_root_cid=_cid("root-unverified"),
            expected=published,
            verified=False,
        )

    # Simulate process restart with a fresh session over the same durable log.
    restarted = _session(
        tmp_path,
        durable_port=durable,
        scan_executor=executor,
        policy=_policy(
            tmp_path,
            event_log_path=session.policy.event_log_path,
            checkpoint_path=session.policy.checkpoint_path,
        ),
    )
    status = restarted.restart()
    assert status.current_root is not None
    assert status.current_root.root_cid == published.root_cid
    assert status.current_root.generation == published.generation
    assert durable.read_root("example/repo") == published
    # Unverified candidate must not have advanced the root.
    assert durable.read_root("example/repo").root_cid != _cid("root-unverified")
    assert durable.recover_calls >= 1
    accepted = restarted.accepted_transitions()
    assert any(item.root.root_cid == published.root_cid for item in accepted)


def test_restart_neither_loses_accepted_nor_publishes_from_journal_alone(
    tmp_path: Path,
) -> None:
    durable = MemoryDurablePort()

    def executor(**kwargs: Any) -> Mapping[str, Any]:
        return {"status": "completed", "verified": True, "new_root_cid": _cid("r1")}

    session = _session(tmp_path, durable_port=durable, scan_executor=executor)
    session.notify_watch(_cid("snap-1"))
    result = session.drain()[0]
    fence = session.fence_for(result.attempt_id)
    assert fence is not None
    published = session.accept_transition(
        attempt_id=result.attempt_id,
        fencing_token=fence.fencing_token,
        new_root_cid=_cid("r1"),
        verified=True,
    )
    cas_before = list(durable.cas_calls)

    restarted = _session(
        tmp_path,
        durable_port=durable,
        policy=_policy(
            tmp_path,
            event_log_path=session.policy.event_log_path,
            checkpoint_path=session.policy.checkpoint_path,
        ),
    )
    status = restarted.restart()
    assert status.current_root == published
    # Restart must not perform a new CAS publication.
    assert durable.cas_calls == cas_before


def test_replay_session_entrypoint_restores_cursor(tmp_path: Path) -> None:
    mod = importlib.import_module(
        "ipfs_accelerate_py.agent_supervisor.semantic_state.session"
    )
    durable = MemoryDurablePort()

    def executor(**kwargs: Any) -> Mapping[str, Any]:
        return {"status": "completed", "verified": True, "new_root_cid": _cid("r-replay")}

    session = _session(tmp_path, durable_port=durable, scan_executor=executor)
    session.notify_watch(_cid("snap-replay"))
    result = session.drain()[0]
    fence = session.fence_for(result.attempt_id)
    assert fence is not None
    session.accept_transition(
        attempt_id=result.attempt_id,
        fencing_token=fence.fencing_token,
        new_root_cid=_cid("r-replay"),
        verified=True,
    )

    owner, status = mod.replay_session(
        _policy(
            tmp_path,
            event_log_path=session.policy.event_log_path,
            checkpoint_path=session.policy.checkpoint_path,
        ),
        durable_port=durable,
        restart=True,
    )
    assert owner is not None
    assert status.current_root is not None
    assert status.current_root.root_cid == _cid("r-replay")
    assert status.cursor_position >= 1


# ---------------------------------------------------------------------------
# Corrupt / truncated event recovery
# ---------------------------------------------------------------------------


def test_truncated_event_tail_recovers(tmp_path: Path) -> None:
    durable = MemoryDurablePort()

    def executor(**kwargs: Any) -> Mapping[str, Any]:
        return {"status": "completed", "verified": True, "new_root_cid": _cid("r-trunc")}

    session = _session(tmp_path, durable_port=durable, scan_executor=executor)
    session.notify_watch(_cid("snap-trunc"))
    result = session.drain()[0]
    fence = session.fence_for(result.attempt_id)
    assert fence is not None
    published = session.accept_transition(
        attempt_id=result.attempt_id,
        fencing_token=fence.fencing_token,
        new_root_cid=_cid("r-trunc"),
        verified=True,
    )

    log_path = Path(session.policy.event_log_path)
    with log_path.open("ab") as handle:
        handle.write(b'{"type":"session_scan_started","attempt_id":"partial"')

    restarted = _session(
        tmp_path,
        durable_port=durable,
        policy=_policy(
            tmp_path,
            event_log_path=log_path,
            checkpoint_path=session.policy.checkpoint_path,
            fail_closed_on_corrupt_log=True,
        ),
    )
    status = restarted.restart()
    assert status.failed_closed is False
    assert status.current_root == published
    # Truncated suffix removed; valid events remain readable.
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            json.loads(line)


def test_corrupt_checkpoint_fails_closed(tmp_path: Path) -> None:
    session_mod = importlib.import_module(
        "ipfs_accelerate_py.agent_supervisor.semantic_state.session"
    )
    log_path = tmp_path / "events.jsonl"
    # Seed a valid event log so restart reaches checkpoint validation.
    from ipfs_accelerate_py.agent_supervisor.runtime.event_log import (
        append_jsonl_event,
    )

    append_jsonl_event(log_path, "session_watch_notification", {
        "repository_id": "example/repo",
        "snapshot_cid": _cid("snap-x"),
        "schema": session_mod.SESSION_SCHEMA,
        "interface": session_mod.SESSION_INTERFACE,
        "board_namespace": session_mod.BOARD_NAMESPACE,
    })
    ck_path = tmp_path / "bad-cursor.json"
    ck_path.write_text("{not-json", encoding="utf-8")

    session = _session(
        tmp_path,
        policy=_policy(
            tmp_path,
            event_log_path=log_path,
            checkpoint_path=ck_path,
            fail_closed_on_corrupt_log=True,
        ),
    )
    with pytest.raises(session_mod.SessionFailedClosed):
        session.restart()
    assert session.status().failed_closed is True
    assert session.status().phase == session_mod.SessionPhase.FAILED_CLOSED.value


# ---------------------------------------------------------------------------
# Shutdown
# ---------------------------------------------------------------------------


def test_explicit_shutdown_cancels_and_joins_owned_work(tmp_path: Path) -> None:
    session_mod = importlib.import_module(
        "ipfs_accelerate_py.agent_supervisor.semantic_state.session"
    )
    started = threading.Event()
    release = threading.Event()
    saw_cancel = threading.Event()

    def executor(
        *,
        repository_id: str,
        snapshot_cid: str,
        attempt_id: str,
        fencing_token: int,
        cancellation: CancellationToken,
    ) -> Mapping[str, Any]:
        started.set()
        # Wait until shutdown cancels, or time out.
        deadline = time.time() + 2.0
        while time.time() < deadline:
            if cancellation.is_cancelled():
                saw_cancel.set()
                return {
                    "status": "cancelled",
                    "diagnostic": cancellation.reason,
                    "verified": False,
                }
            if release.is_set():
                break
            time.sleep(0.01)
        return {"status": "completed", "verified": False}

    session = _session(
        tmp_path,
        scan_executor=executor,
        policy=_policy(tmp_path, worker_enabled=True, debounce_ms=0, join_timeout_s=2.0),
    )
    session.start()
    session.notify_watch(_cid("snap-shutdown"))
    assert started.wait(timeout=2.0)

    status = session.shutdown(reason="test_stop")
    assert status.shutdown is True
    assert status.phase == session_mod.SessionPhase.STOPPED.value
    assert saw_cancel.wait(timeout=2.0)
    # Worker must be joined.
    assert session._worker is None or not session._worker.is_alive()

    with pytest.raises(session_mod.SessionShutdownError):
        session.notify_watch(_cid("snap-after-shutdown"))
    release.set()


def test_shutdown_is_idempotent(tmp_path: Path) -> None:
    session = _session(tmp_path)
    first = session.shutdown(reason="once")
    second = session.shutdown(reason="twice")
    assert first.shutdown is True
    assert second.shutdown is True
    assert second.phase == "stopped"


# ---------------------------------------------------------------------------
# Module entrypoints / status
# ---------------------------------------------------------------------------


def test_watch_session_module_entrypoint(tmp_path: Path) -> None:
    mod = importlib.import_module(
        "ipfs_accelerate_py.agent_supervisor.semantic_state.session"
    )
    calls: list[str] = []

    def executor(**kwargs: Any) -> Mapping[str, Any]:
        calls.append(kwargs["snapshot_cid"])
        return {"status": "completed", "verified": False}

    owner, ack, results = mod.watch_session(
        _policy(tmp_path),
        _cid("snap-entry"),
        scan_executor=executor,
        process=True,
    )
    assert ack.scheduled is True
    assert len(results) == 1
    assert calls == [_cid("snap-entry")]
    assert owner.status().scans_completed == 1


def test_status_snapshot_is_deterministic(tmp_path: Path) -> None:
    session = _session(tmp_path)
    session.notify_watch(_cid("snap-status-a"))
    session.notify_watch(_cid("snap-status-b"))
    status = session.status()
    payload = status.to_dict()
    assert payload["interface"] == "SemanticStateSession@1"
    assert payload["repository_id"] == "example/repo"
    assert payload["pending_snapshot_cids"] == sorted(payload["pending_snapshot_cids"])
    assert set(payload["pending_snapshot_cids"]) == {
        _cid("snap-status-a"),
        _cid("snap-status-b"),
    }


def test_context_manager_shuts_down(tmp_path: Path) -> None:
    session_mod = importlib.import_module(
        "ipfs_accelerate_py.agent_supervisor.semantic_state.session"
    )
    with _session(tmp_path) as session:
        session.notify_watch(_cid("snap-cm"))
        assert session.status().shutdown is False
    assert session.status().shutdown is True
    assert session.status().phase == session_mod.SessionPhase.STOPPED.value
