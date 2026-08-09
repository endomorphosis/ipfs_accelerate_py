from __future__ import annotations

import argparse
import errno
import logging
import shutil
import time
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.task_sources import taskboard_store
from ipfs_accelerate_py.agent_supervisor.control.control_contracts import EventCursor
from ipfs_accelerate_py.agent_supervisor.runtime.event_log import (
    append_jsonl_event,
    initial_event_cursor,
    read_event_cursor_checkpoint,
    read_jsonl_event_page,
    rotate_event_log_if_needed,
    write_event_cursor_checkpoint,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon_runner import (
    ImplementationDaemonRunContext,
    run_portal_implementation_daemon_loop,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.taskboard_store import (
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
    PortalTaskState,
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


def test_validation_wake_ignores_single_flight_coordination_noise(
    tmp_path: Path,
) -> None:
    daemon = _drained_daemon(tmp_path)
    cache_root = Path(daemon.validation_cache_dir)
    authoritative_entries = cache_root / "entries" / "validation"
    authoritative_entries.mkdir(parents=True, exist_ok=True)
    validation_targets = daemon._runtime_source_paths()["validation"]

    assert validation_targets == (authoritative_entries,)

    clock = LogicalClock()
    watcher = LogicalWatcher(clock)
    coordinator = RuntimeWakeCoordinator(
        {RuntimeWakeKind.VALIDATION: validation_targets},
        safety_interval_seconds=300.0,
        watcher=watcher,
        clock=clock,
    )
    try:
        # Single-flight leases and lock heartbeats coordinate producers but
        # are not successful validation evidence.
        coordination_db = cache_root / "single-flight.sqlite3"
        coordination_db.touch()
        coordinator.notify(
            RuntimeWakeKind.VALIDATION,
            revision="coordination-only-revision",
        )
        noise = coordinator.wait(timeout=0.0)
        assert noise.kinds == (RuntimeWakeKind.OBSERVATION_WINDOW,)
        coordinator.acknowledge(noise)

        entry = authoritative_entries / "aa" / "result.json"
        entry.parent.mkdir(parents=True)
        entry.write_text('{"passed":true}\n', encoding="utf-8")
        coordinator.notify(
            RuntimeWakeKind.VALIDATION,
            revision="authoritative-result-revision",
        )
        result = coordinator.wait()
        assert result.kinds == (RuntimeWakeKind.VALIDATION,)
        coordinator.acknowledge(result)
    finally:
        coordinator.close()


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


def test_daemon_wait_does_not_wake_on_its_own_post_pass_files(
    tmp_path: Path,
) -> None:
    daemon = _drained_daemon(tmp_path)
    clock = LogicalClock()
    watcher = LogicalWatcher(clock)
    coordinator = RuntimeWakeCoordinator(
        daemon._runtime_source_paths(),
        safety_interval_seconds=15.0,
        watcher=watcher,
        clock=clock,
        metadata_entry_limit=64,
        metadata_depth_limit=6,
    )
    daemon._runtime_wake_coordinator = coordinator

    # The completed pass establishes the cursor boundary for its own durable
    # writes. Model the corresponding native notification remaining queued
    # after that boundary.
    daemon.run_once()
    watcher.notify()

    try:
        event = daemon.wait_for_wake(timeout=15.0)[0]
    finally:
        coordinator.close()

    assert event.kinds == (RuntimeWakeKind.OBSERVATION_WINDOW,)
    assert event.safety_timer is True
    assert watcher.wait_calls == 2
    assert clock() == pytest.approx(15.0)


def test_parallel_lane_bookkeeping_is_not_a_repository_wake(
    tmp_path: Path,
) -> None:
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    head_path = git_dir / "HEAD"
    head_path.write_text("ref: refs/heads/main\n", encoding="utf-8")
    sibling_state = tmp_path / "runtime" / "lane-01" / "state.json"
    sibling_state.parent.mkdir(parents=True)
    sibling_state.write_text("{}\n", encoding="utf-8")
    daemon = _drained_daemon(tmp_path)
    repository_targets = daemon._runtime_source_paths()["repository"]

    assert tmp_path not in repository_targets
    assert head_path in repository_targets

    clock = LogicalClock()
    watcher = LogicalWatcher(clock)
    coordinator = RuntimeWakeCoordinator(
        daemon._runtime_source_paths(),
        safety_interval_seconds=15.0,
        watcher=watcher,
        clock=clock,
        metadata_entry_limit=64,
        metadata_depth_limit=6,
    )
    try:
        sibling_state.write_text('{"heartbeat": 1}\n', encoding="utf-8")
        watcher.notify()
        bookkeeping_event = coordinator.wait(timeout=15.0)
        assert bookkeeping_event.kinds == (
            RuntimeWakeKind.OBSERVATION_WINDOW,
        )
        coordinator.acknowledge(bookkeeping_event)

        head_path.write_text("ref: refs/heads/release\n", encoding="utf-8")
        watcher.notify()
        repository_event = coordinator.wait(timeout=15.0)
    finally:
        coordinator.close()

    assert repository_event.kinds == (RuntimeWakeKind.REPOSITORY,)
    assert repository_event.safety_timer is False
    assert clock() == pytest.approx(15.0)


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


def test_configured_daemon_loop_honors_scheduled_retry_before_interval() -> None:
    parsed = argparse.Namespace(once=False, interval=300.0)
    context = ImplementationDaemonRunContext(
        parsed=parsed,
        state_path=Path("state.json"),
        strategy_path=Path("strategy.json"),
        events_path=Path("events.jsonl"),
    )

    class RetryScheduledDaemon:
        def __init__(self) -> None:
            self.passes = 0
            self.waits: list[float] = []

        def run_once(self) -> dict[str, float]:
            self.passes += 1
            return {"next_wake_after_seconds": 17.5}

        def wait_for_wake(self, *, timeout: float) -> None:
            self.waits.append(timeout)
            parsed.once = True

    daemon = RetryScheduledDaemon()

    run_portal_implementation_daemon_loop(
        daemon,
        context,
        logger=logging.getLogger("test-provider-retry-daemon-runner"),
    )

    assert daemon.passes == 2
    assert daemon.waits == [17.5]


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


def test_projection_store_content_addresses_and_quarantines_invalid_checkpoint(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "projection-checkpoint.json"
    store = ProjectionDeltaCheckpointStore(checkpoint_path)
    cursor = EventCursor.initial(
        "runtime-events",
        snapshot_id="runtime-snapshot",
    )
    store.materialize({"ready": 2}, cursor)
    invalid = checkpoint_path.read_bytes().replace(b'"ready":2', b'"ready":3')
    checkpoint_path.write_bytes(invalid)

    with pytest.raises(
        ValueError,
        match="projection checkpoint identity does not match",
    ):
        store.load()

    repair = store.quarantine_invalid()

    assert repair["quarantined"] is True
    assert repair["reason"] == "projection checkpoint identity does not match"
    assert repair["content_sha256"].startswith("sha256:")
    assert checkpoint_path.exists() is False
    quarantine_path = Path(repair["quarantine_path"])
    assert quarantine_path.read_bytes() == invalid
    rebuilt = store.materialize({"ready": 3}, cursor)
    assert rebuilt.write_count == 1
    assert store.load() == ({"ready": 3}, cursor)


def test_daemon_restart_quarantines_invalid_runtime_checkpoint(
    tmp_path: Path,
) -> None:
    daemon = _drained_daemon(tmp_path)
    first = daemon.run_once()
    assert first["reason"] == "no_tasks_found"
    checkpoint_path = daemon.runtime_checkpoint_path
    invalid = checkpoint_path.read_bytes().replace(
        b'"task_count":0',
        b'"task_count":9',
        1,
    )
    checkpoint_path.write_bytes(invalid)

    restarted = _drained_daemon(tmp_path)

    assert restarted._runtime_checkpoint == {}
    assert restarted._runtime_checkpoint_repair["quarantined"] is True
    quarantine_path = Path(
        restarted._runtime_checkpoint_repair["quarantine_path"]
    )
    assert quarantine_path.read_bytes() == invalid
    restarted.todo_path.write_text(
        """# Task board

## PORTAL-001 Completed task

- Status: completed
""",
        encoding="utf-8",
    )
    recovered = restarted.run_once()
    loaded = restarted.runtime_checkpoint_store.load()
    assert recovered["completed_count"] == 1
    assert recovered["delta_checkpoint"]["checkpoint_repair"][
        "quarantined"
    ] is True
    assert loaded is not None
    assert loaded[0]["result"]["completed_count"] == 1


def test_daemon_restart_reconciles_board_before_reusing_cached_projection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def completed_board(task_count: int) -> str:
        tasks = ["# Completed task board"]
        for index in range(1, task_count + 1):
            tasks.extend(
                [
                    "",
                    f"## PORTAL-{index:03d} Completed task {index}",
                    "",
                    "- Status: completed",
                ]
            )
        return "\n".join(tasks) + "\n"

    daemon = _drained_daemon(tmp_path)
    daemon.todo_path.write_text(completed_board(12), encoding="utf-8")
    first = daemon.run_once()
    assert first["completed_count"] == 12

    # Model a shutdown checkpoint which observed the current board bytes but
    # still contains the prior task-state projection. A restarted watcher
    # baselines those current bytes, so no file notification will arrive.
    daemon.todo_path.write_text(completed_board(14), encoding="utf-8")
    current_source_digest, current_sources = daemon._runtime_source_head()
    loaded = daemon.runtime_checkpoint_store.load()
    assert loaded is not None
    stale_projection, cursor = loaded
    stale_projection = dict(stale_projection)
    assert stale_projection["result"]["completed_count"] == 12
    stale_projection["source_digest"] = current_source_digest
    daemon.runtime_checkpoint_store.materialize(stale_projection, cursor)

    restarted = PortalImplementationDaemon(
        todo_path=daemon.todo_path,
        state_path=daemon.state_path,
        strategy_path=daemon.strategy_path,
        events_path=daemon.events_path,
        repo_root=tmp_path,
        worktree_pool_enabled=False,
        validation_cache_dir=tmp_path / "runtime" / "validation-cache",
        merge_queue_dir=tmp_path / "runtime" / "merge-queue",
    )
    assert (
        restarted._runtime_checkpoint["source_digest"]
        == current_source_digest
    )
    assert restarted._runtime_last_result["completed_count"] == 12
    # Constructor setup may create previously absent runtime directories.
    # Model the production stall after that setup has been included in the
    # persisted source head: the checkpoint digest is current, but its cached
    # result and the task-state projection are still stale.
    real_source_head = restarted._runtime_source_head
    source_head_calls = 0

    def startup_source_head() -> tuple[str, dict[str, Any]]:
        nonlocal source_head_calls
        source_head_calls += 1
        if source_head_calls == 1:
            return current_source_digest, current_sources
        return real_source_head()

    monkeypatch.setattr(restarted, "_runtime_source_head", startup_source_head)

    reconciled = restarted.run_once()

    assert reconciled["unchanged"] is False
    assert reconciled["wake_kinds"] == []
    assert reconciled["completed_count"] == 14
    assert PortalTaskState.load(restarted.state_path).completed_count == 14
    unchanged = restarted.run_once()
    assert unchanged["unchanged"] is True
    assert unchanged["write_count"] == 0
    assert unchanged["completed_count"] == 14


def test_running_daemon_repairs_checkpoint_corrupted_between_passes(
    tmp_path: Path,
) -> None:
    daemon = _drained_daemon(tmp_path)
    daemon.run_once()
    checkpoint_path = daemon.runtime_checkpoint_path
    invalid = checkpoint_path.read_bytes().replace(
        b'"task_count":0',
        b'"task_count":9',
        1,
    )
    checkpoint_path.write_bytes(invalid)
    daemon.todo_path.write_text(
        """# Task board

## PORTAL-001 Completed task

- Status: completed
""",
        encoding="utf-8",
    )

    recovered = daemon.run_once()

    repair = recovered["delta_checkpoint"]["checkpoint_repair"]
    assert repair["quarantined"] is True
    assert Path(repair["quarantine_path"]).read_bytes() == invalid
    loaded = daemon.runtime_checkpoint_store.load()
    assert loaded is not None
    assert loaded[0]["result"]["completed_count"] == 1


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


def test_idle_populated_board_does_not_rewrite_typed_task_identities(
    tmp_path: Path,
) -> None:
    board = tmp_path / "tasks.todo.md"
    board.write_text(
        """# Stable board

## TASK-001 Completed task

- Status: completed
- Outputs: src/completed.py
- Acceptance: The completed output remains represented.

## TASK-002 Operator-blocked task

- Status: blocked
- Outputs: src/blocked.py
- Acceptance: An operator must explicitly release this task.
""",
        encoding="utf-8",
    )
    daemon = PortalImplementationDaemon(
        todo_path=board,
        state_path=tmp_path / "runtime" / "state.json",
        strategy_path=tmp_path / "runtime" / "strategy.json",
        events_path=tmp_path / "runtime" / "events.jsonl",
        repo_root=tmp_path,
        task_header_prefix="## TASK-",
        worktree_pool_enabled=False,
        validation_cache_dir=tmp_path / "runtime" / "validation-cache",
        merge_queue_dir=tmp_path / "runtime" / "merge-queue",
    )

    first = daemon.run_once()
    durable_paths = (
        daemon.state_path,
        daemon.events_path,
        daemon.runtime_checkpoint_path,
    )
    before = {path: _file_identity(path) for path in durable_paths}
    second = daemon.run_once()

    assert first["completed_count"] == 1
    assert first["blocked_count"] == 1
    assert second["unchanged"] is True
    assert second["write_count"] == 0
    assert second["projection_delta"] == {}
    assert {path: _file_identity(path) for path in durable_paths} == before


def test_expired_ready_task_cooldown_wakes_unchanged_runtime(
    tmp_path: Path,
) -> None:
    board = tmp_path / "tasks.todo.md"
    board.write_text(
        """# Retry board

## PORTAL-001 Retry after transient maintenance

- Status: todo
- Completion: manual
- Priority: P1
- Track: runtime
- Depends on:
- Outputs: src/retry.py
- Validation:
- Acceptance: The ready task is selected after its cooldown expires.
""",
        encoding="utf-8",
    )
    daemon = PortalImplementationDaemon(
        todo_path=board,
        state_path=tmp_path / "runtime" / "state.json",
        strategy_path=tmp_path / "runtime" / "strategy.json",
        events_path=tmp_path / "runtime" / "events.jsonl",
        repo_root=tmp_path,
        task_header_prefix="## PORTAL-",
        worktree_pool_enabled=False,
        validation_cache_dir=tmp_path / "runtime" / "validation-cache",
        merge_queue_dir=tmp_path / "runtime" / "merge-queue",
    )

    first = daemon.run_once()
    state = PortalTaskState.load(daemon.state_path)
    state.active_task_id = ""
    state.active_task_key = ""
    state.active_task_cid = ""
    state.implementation_in_progress = False
    state.save(daemon.state_path)
    daemon.task_queue.defer(
        "PORTAL-001",
        30,
        reason="implementation_protected_path_maintenance_active",
    )
    daemon.task_queue.save()
    daemon._runtime_last_source_digest = daemon._runtime_source_head()[0]

    cooling_down = daemon.run_once()

    assert cooling_down["unchanged"] is True
    assert cooling_down["implementation_result"] is None
    assert cooling_down["task_selection_retry_task_ids"] == ["PORTAL-001"]
    assert 0 < cooling_down["next_wake_after_seconds"] <= 30

    queue_key = daemon.task_queue.resolve_key("PORTAL-001")
    daemon.task_queue.entries[queue_key].cooldown_until = time.time() - 1
    daemon.task_queue.save()
    daemon._runtime_last_source_digest = daemon._runtime_source_head()[0]

    retried = daemon.run_once()

    assert first["active_task_id"] == "PORTAL-001"
    assert retried["unchanged"] is False
    assert retried["active_task_id"] == "PORTAL-001"
    assert "task_selection_retry_at" not in retried


def test_ephemeral_merge_consumer_lease_is_not_a_runtime_wake_source(
    tmp_path: Path,
) -> None:
    daemon = _drained_daemon(tmp_path)
    lease_paths = daemon._runtime_source_paths()["lease"]
    merge_queue_root = Path(daemon.merge_queue_dir)
    shared_claim_dir = (
        tmp_path / ".git" / "implementation-task-claims"
    )
    protected_maintenance_lock = (
        tmp_path / ".git" / "implementation-protected-path-maintenance.lock"
    )

    assert daemon.merge_queue.database_path in lease_paths
    assert daemon.merge_queue.pending_dir in lease_paths
    assert daemon.merge_queue.processing_dir in lease_paths
    assert shared_claim_dir in lease_paths
    assert protected_maintenance_lock in lease_paths
    assert merge_queue_root not in lease_paths
    assert merge_queue_root / "train" not in lease_paths

    consumer_lock = merge_queue_root / "train" / "consumer.lock"
    consumer_lock.parent.mkdir(parents=True)
    consumer_lock.write_text("initial lease\n", encoding="utf-8")
    clock = LogicalClock()
    watcher = LogicalWatcher(clock)
    coordinator = RuntimeWakeCoordinator(
        {RuntimeWakeKind.LEASE: lease_paths},
        safety_interval_seconds=30.0,
        watcher=watcher,
        clock=clock,
    )
    try:
        consumer_lock.write_text("ephemeral lease heartbeat\n", encoding="utf-8")
        coordinator.notify(RuntimeWakeKind.LEASE, revision="consumer-heartbeat")

        event = coordinator.wait()
    finally:
        coordinator.close()

    assert event.kinds == (RuntimeWakeKind.OBSERVATION_WINDOW,)
    assert event.safety_timer is True
    assert clock() == pytest.approx(30.0)


def test_shared_protected_maintenance_release_is_a_runtime_wake(
    tmp_path: Path,
) -> None:
    daemon = _drained_daemon(tmp_path)
    lease_paths = daemon._runtime_source_paths()["lease"]
    maintenance_lock = daemon._protected_path_maintenance_lock_path()
    clock = LogicalClock()
    watcher = LogicalWatcher(clock)
    coordinator = RuntimeWakeCoordinator(
        {RuntimeWakeKind.LEASE: lease_paths},
        safety_interval_seconds=30.0,
        watcher=watcher,
        clock=clock,
    )
    try:
        maintenance_lock.parent.mkdir(parents=True, exist_ok=True)
        maintenance_lock.write_text("active\n", encoding="utf-8")
        watcher.notify()
        acquired = coordinator.wait()
        coordinator.acknowledge(acquired)

        maintenance_lock.unlink()
        watcher.notify()
        released = coordinator.wait()
    finally:
        coordinator.close()

    assert acquired.kinds == (RuntimeWakeKind.LEASE,)
    assert acquired.safety_timer is False
    assert released.kinds == (RuntimeWakeKind.LEASE,)
    assert released.safety_timer is False
    assert clock() == 0.0
