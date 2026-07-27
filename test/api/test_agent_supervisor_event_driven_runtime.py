from __future__ import annotations

import argparse
import errno
import logging
import shutil
import time
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor import taskboard_store
from ipfs_accelerate_py.agent_supervisor.control_contracts import EventCursor
from ipfs_accelerate_py.agent_supervisor.event_log import (
    append_jsonl_event,
    initial_event_cursor,
    read_event_cursor_checkpoint,
    read_jsonl_event_page,
    rotate_event_log_if_needed,
    write_event_cursor_checkpoint,
)
from ipfs_accelerate_py.agent_supervisor.implementation_daemon_runner import (
    ImplementationDaemonRunContext,
    run_portal_implementation_daemon_loop,
)
from ipfs_accelerate_py.agent_supervisor.taskboard_store import (
    BlockingTimerWatcher,
    PathMetadata,
    ProjectionDeltaCheckpointStore,
    RuntimeWakeCoordinator,
    RuntimeWakeKind,
    create_directory_watcher,
    locked_taskboard,
    replace_locked_taskboard,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    PortalImplementationDaemon,
)


class LogicalClock:
    """A deterministic monotonic clock for observation-window tests."""

    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += max(0.0, float(seconds))


class LogicalWatcher:
    """A watcher which advances logical time instead of sleeping."""

    backend = "logical"
    native = False

    def __init__(self, clock: LogicalClock) -> None:
        self.clock = clock
        self.notifications = 0
        self.closed = False
        self.wait_calls = 0

    def notify(self) -> None:
        if not self.closed:
            self.notifications += 1

    def wait(self, timeout: float | None) -> bool:
        self.wait_calls += 1
        if self.notifications:
            self.notifications -= 1
            return True
        self.clock.advance(0.0 if timeout is None else timeout)
        return False

    def close(self) -> None:
        self.closed = True


def _consume_all_events(
    path: Path,
    cursor: EventCursor,
    *,
    limit: int,
) -> tuple[list[dict[str, Any]], EventCursor]:
    consumed: list[dict[str, Any]] = []
    while True:
        page = read_jsonl_event_page(path, cursor, limit=limit)
        consumed.extend(dict(item) for item in page.events)
        cursor = page.next_cursor
        if not page.has_more:
            return consumed, cursor


def _file_identity(path: Path) -> tuple[int, int, int, bytes]:
    stat_result = path.stat()
    return (
        int(stat_result.st_ino),
        int(stat_result.st_size),
        int(stat_result.st_mtime_ns),
        path.read_bytes(),
    )


def _drained_daemon(tmp_path: Path) -> PortalImplementationDaemon:
    board = tmp_path / "tasks.todo.md"
    board.write_text("# Drained task board\n", encoding="utf-8")
    return PortalImplementationDaemon(
        todo_path=board,
        state_path=tmp_path / "runtime" / "state.json",
        strategy_path=tmp_path / "runtime" / "strategy.json",
        events_path=tmp_path / "runtime" / "events.jsonl",
        repo_root=tmp_path,
        worktree_pool_enabled=False,
        validation_cache_dir=tmp_path / "runtime" / "validation-cache",
        merge_queue_dir=tmp_path / "runtime" / "merge-queue",
    )


def test_canonical_cursor_replay_is_gapless_exactly_once_across_restart_and_rotation(
    tmp_path: Path,
) -> None:
    events_path = tmp_path / "events.jsonl"
    checkpoint_path = tmp_path / "consumer.cursor.json"
    initial = initial_event_cursor(events_path)
    written = [
        append_jsonl_event(
            events_path,
            "runtime_wake",
            {"ordinal": ordinal},
        )
        for ordinal in range(1, 7)
    ]

    first_page = read_jsonl_event_page(events_path, initial, limit=3)
    assert [item["ordinal"] for item in first_page.events] == [1, 2, 3]
    assert [item["sequence"] for item in first_page.events] == [1, 2, 3]
    assert first_page.next_cursor.position == 3
    assert first_page.next_cursor.last_event_id == written[2]["event_id"]
    assert write_event_cursor_checkpoint(
        checkpoint_path, first_page.next_cursor
    )
    checkpoint_identity = _file_identity(checkpoint_path)
    assert not write_event_cursor_checkpoint(
        checkpoint_path, first_page.next_cursor
    )
    assert _file_identity(checkpoint_path) == checkpoint_identity

    rotation = rotate_event_log_if_needed(
        events_path,
        max_bytes=1,
        retain_recent=2,
        max_archives=4,
    )
    assert rotation["rotated"] is True
    for ordinal in range(7, 9):
        append_jsonl_event(
            events_path,
            "runtime_wake",
            {"ordinal": ordinal},
        )

    # Model the exact overlapping tail which can survive a crash between
    # archive installation and active-tail replacement. Cursor replay must
    # coalesce it by canonical sequence and event identity.
    duplicate_tail = events_path.with_name(
        f"{events_path.name}.rotated-recovery-duplicate"
    )
    shutil.copy2(events_path, duplicate_tail)
    events_path.with_name(f"{events_path.name}.manifest.json").unlink()

    recovered = read_event_cursor_checkpoint(checkpoint_path)
    assert recovered == first_page.next_cursor
    remaining, final_cursor = _consume_all_events(
        events_path,
        recovered,
        limit=2,
    )
    assert [item["ordinal"] for item in remaining] == [4, 5, 6, 7, 8]
    assert [item["sequence"] for item in remaining] == [4, 5, 6, 7, 8]
    assert len({item["event_id"] for item in remaining}) == len(remaining)
    assert final_cursor.position == 8

    replay_after_restart, unchanged_cursor = _consume_all_events(
        events_path,
        final_cursor.to_token(),
        limit=2,
    )
    assert replay_after_restart == []
    assert unchanged_cursor == final_cursor


def test_all_runtime_wake_kinds_use_canonical_cursors_and_two_phase_acknowledgement(
    tmp_path: Path,
) -> None:
    file_backed = (
        RuntimeWakeKind.TASK_BOARD,
        RuntimeWakeKind.OBJECTIVE,
        RuntimeWakeKind.REPOSITORY,
        RuntimeWakeKind.LEASE,
        RuntimeWakeKind.VALIDATION,
        RuntimeWakeKind.POLICY,
    )
    semantic = (
        RuntimeWakeKind.CHILD_PROCESS,
        RuntimeWakeKind.PROVIDER_CAPACITY,
    )
    targets: dict[RuntimeWakeKind, Path] = {}
    for kind in file_backed:
        target = tmp_path / f"{kind.value}.state"
        target.write_text("revision=0\n", encoding="utf-8")
        targets[kind] = target

    clock = LogicalClock()
    watcher = LogicalWatcher(clock)
    coordinator = RuntimeWakeCoordinator(
        targets,
        safety_interval_seconds=300.0,
        watcher=watcher,
        clock=clock,
        metadata_entry_limit=8,
        metadata_depth_limit=1,
    )
    observed: set[RuntimeWakeKind] = set()
    try:
        for revision, kind in enumerate(file_backed, start=1):
            targets[kind].write_text(
                f"revision={revision}:{kind.value}\n",
                encoding="utf-8",
            )
            coordinator.notify(kind, revision=f"file:{revision}")
            event = coordinator.wait()
            assert event.kinds == (kind,)
            assert len(event.metadata) == 1
            assert event.metadata[0].cursor.startswith("path-metadata:sha256:")
            assert event.semantic_cursors == {}
            coordinator.acknowledge(event)
            observed.update(event.kinds)

        for revision, kind in enumerate(semantic, start=1):
            token = f"{kind.value}:revision:{revision}"
            coordinator.notify(kind, revision=token)
            event = coordinator.wait()
            assert event.kinds == (kind,)
            assert event.semantic_cursors == {kind.value: token}

            # A wake which has not reached a durable checkpoint must replay
            # with the exact same cursor rather than being silently consumed.
            replayed = coordinator.wait()
            assert replayed.kinds == event.kinds
            assert replayed.cursor_ids == event.cursor_ids
            coordinator.acknowledge(replayed)
            observed.update(event.kinds)

            # Re-delivery of an acknowledged canonical revision is a no-op.
            coordinator.notify(kind, revision=token)

        safety_event = coordinator.wait()
        assert safety_event.kinds == (RuntimeWakeKind.OBSERVATION_WINDOW,)
        assert safety_event.safety_timer is True
        assert safety_event.reason == "safety_timer"
        assert clock() == pytest.approx(300.0)
        coordinator.acknowledge(safety_event)
        observed.update(safety_event.kinds)
    finally:
        coordinator.close()

    assert observed == set(RuntimeWakeKind)
    assert tuple(kind.value for kind in RuntimeWakeKind) == (
        "task_board",
        "objective",
        "repository",
        "child_process",
        "lease",
        "validation",
        "provider_capacity",
        "policy",
        "observation_window",
    )


def test_unsupported_native_watcher_falls_back_to_blocking_timer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class UnsupportedWatcher:
        def __init__(self, _paths: object) -> None:
            raise taskboard_store.NativeWatcherUnavailable(
                errno.ENOSYS,
                "filesystem notifications unsupported",
            )

    monkeypatch.setattr(
        taskboard_store,
        "LinuxDirectoryWatcher",
        UnsupportedWatcher,
    )
    watcher = create_directory_watcher([tmp_path], prefer_native=True)
    try:
        assert isinstance(watcher, BlockingTimerWatcher)
        assert watcher.backend == "blocking_timer"
        assert watcher.native is False
        watcher.notify()
        assert watcher.wait(0.0) is True
        assert watcher.wait(0.0) is False
    finally:
        watcher.close()


def test_daemon_event_runtime_cleanup_is_idempotent(tmp_path: Path) -> None:
    daemon = _drained_daemon(tmp_path)
    clock = LogicalClock()
    watcher = LogicalWatcher(clock)
    coordinator = RuntimeWakeCoordinator(
        daemon._runtime_source_paths(),
        watcher=watcher,
        clock=clock,
    )
    daemon._runtime_wake_coordinator = coordinator

    daemon.close_event_runtime()
    daemon.close_event_runtime()

    assert watcher.closed is True
    assert daemon._runtime_wake_coordinator is None


def test_configured_daemon_loop_waits_for_wake_and_closes_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parsed = argparse.Namespace(once=False, interval=321.0)
    context = ImplementationDaemonRunContext(
        parsed=parsed,
        state_path=Path("state.json"),
        strategy_path=Path("strategy.json"),
        events_path=Path("events.jsonl"),
    )

    class EventDrivenDaemon:
        def __init__(self) -> None:
            self.passes = 0
            self.waits: list[float] = []
            self.close_calls = 0

        def run_once(self) -> dict[str, int]:
            self.passes += 1
            return {"passes": self.passes}

        def wait_for_wake(self, *, timeout: float) -> None:
            self.waits.append(timeout)
            parsed.once = True

        def close_event_runtime(self) -> None:
            self.close_calls += 1

    daemon = EventDrivenDaemon()
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.implementation_daemon_runner.time.sleep",
        lambda _seconds: (_ for _ in ()).throw(
            AssertionError("event-driven daemon must not poll with sleep")
        ),
    )

    run_portal_implementation_daemon_loop(
        daemon,
        context,
        logger=logging.getLogger("test-event-driven-daemon-runner"),
    )

    assert daemon.passes == 2
    assert daemon.waits == [321.0]
    assert daemon.close_calls == 1


def test_metadata_scans_are_bounded_before_projection_work(
    tmp_path: Path,
) -> None:
    tree = tmp_path / "large-tree"
    tree.mkdir()
    for index in range(40):
        (tree / f"entry-{index:03d}.txt").write_text(
            f"{index}\n",
            encoding="utf-8",
        )

    metadata = PathMetadata.capture(
        tree,
        max_entries=5,
        max_depth=0,
    )

    assert metadata.kind == "directory"
    assert metadata.entries_scanned == 5
    assert metadata.entries_truncated is True
    assert metadata.entry_count <= 6
    assert metadata.entries_digest.startswith("sha256:")
    assert metadata.to_dict()["metadata_id"] == metadata.cursor


def test_projection_and_taskboard_stores_make_zero_unchanged_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_path = tmp_path / "projection-checkpoint.json"
    writes: list[Path] = []
    real_atomic_write = taskboard_store._atomic_write

    def measured_atomic_write(path: Path, payload: bytes) -> None:
        writes.append(path)
        real_atomic_write(path, payload)

    monkeypatch.setattr(taskboard_store, "_atomic_write", measured_atomic_write)
    store = ProjectionDeltaCheckpointStore(checkpoint_path)
    cursor = EventCursor.initial(
        "runtime-events",
        snapshot_id="runtime-snapshot",
    )

    first = store.materialize(
        {"ready": 2, "waiting": 1, "obsolete": True},
        cursor,
    )
    first_identity = _file_identity(checkpoint_path)
    unchanged = store.materialize(
        {"obsolete": True, "waiting": 1, "ready": 2},
        cursor.to_token(),
    )
    assert first.write_count == 1
    assert first.delta == {
        "set": {"obsolete": True, "ready": 2, "waiting": 1},
        "remove": [],
    }
    assert unchanged.changed is False
    assert unchanged.write_count == 0
    assert unchanged.delta == {"set": {}, "remove": []}
    assert writes == [checkpoint_path]
    assert _file_identity(checkpoint_path) == first_identity

    advanced = cursor.advance(position=1, event_id="event:1")
    changed = store.materialize(
        {"ready": 3, "waiting": 1},
        advanced,
    )
    assert changed.projection_changed is True
    assert changed.cursor_changed is True
    assert changed.delta == {
        "set": {"ready": 3},
        "remove": ["obsolete"],
    }
    assert len(writes) == 2
    assert store.load() == (
        {"ready": 3, "waiting": 1},
        advanced,
    )

    board = tmp_path / "taskboard.md"
    with locked_taskboard(board) as stream:
        assert replace_locked_taskboard(stream, "# Board\n") is True
    board_identity = _file_identity(board)
    with locked_taskboard(board) as stream:
        assert replace_locked_taskboard(stream, "# Board\n") is False
    assert _file_identity(board) == board_identity


def test_drained_board_ten_minute_logical_fixture_uses_under_two_percent_cpu_and_writes_nothing(
    tmp_path: Path,
) -> None:
    daemon = _drained_daemon(tmp_path)
    first = daemon.run_once()
    assert first["reason"] == "no_tasks_found"

    clock = LogicalClock()
    watcher = LogicalWatcher(clock)
    coordinator = RuntimeWakeCoordinator(
        daemon._runtime_source_paths(),
        safety_interval_seconds=300.0,
        watcher=watcher,
        clock=clock,
        metadata_entry_limit=16,
        metadata_depth_limit=1,
    )
    daemon._runtime_wake_coordinator = coordinator
    durable_paths = (
        daemon.state_path,
        daemon.events_path,
        daemon.runtime_checkpoint_path,
    )
    before = {path: _file_identity(path) for path in durable_paths}
    cpu_started = time.process_time()
    results: list[dict[str, Any]] = []
    try:
        for _window in range(2):
            event = daemon.wait_for_wake()[0]
            assert event.kinds == (RuntimeWakeKind.OBSERVATION_WINDOW,)
            results.append(daemon.run_once())
    finally:
        coordinator.close()
    cpu_seconds = time.process_time() - cpu_started

    assert clock() == pytest.approx(600.0)
    assert watcher.wait_calls == 2
    assert cpu_seconds / 600.0 < 0.02
    assert all(result["unchanged"] is True for result in results)
    assert all(result["write_count"] == 0 for result in results)
    assert all(result["projection_delta"] == {} for result in results)
    assert all(
        result["wake_kinds"] == ["observation_window"]
        for result in results
    )
    assert {path: _file_identity(path) for path in durable_paths} == before
