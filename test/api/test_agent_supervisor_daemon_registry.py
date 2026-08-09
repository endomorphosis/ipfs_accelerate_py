"""Tests for DaemonRegistry (DQP-014).

Evidence subset: registration, adoption, PID reuse, session expiry, heartbeat
compaction, late heartbeat, server restart, exact ancestry.

Acceptance: Raw PID never proves identity; dead/reused/unknown process births
cannot renew; duplicate active role/lane ownership is fenced; heartbeats and
progress are distinct; status files can mirror but cannot create or extend a
session.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
    OwnerLiveness,
    ProcessBirthIdentity,
)
from ipfs_accelerate_py.agent_supervisor.runtime.daemon_registry import (
    DAEMON_INSTANCE_INTERFACE,
    DAEMON_REGISTRY_INTERFACE,
    DAEMON_SESSION_INTERFACE,
    HEARTBEAT_INTERFACE,
    SUPERVISOR_INSTANCE_INTERFACE,
    DaemonRegistry,
    DaemonRegistryConflictError,
    DaemonRegistryIdentityError,
    DaemonRegistrySessionError,
    ExitDisposition,
    InstanceStatus,
    RestartDisposition,
    SessionStatus,
    duckdb_available,
    open_daemon_registry,
    process_birth_id,
    process_births_match,
)

pytestmark = pytest.mark.skipif(
    not duckdb_available(),
    reason="DuckDB is required for DaemonRegistry hermetic tests",
)


class FakeClock:
    def __init__(self, start_ms: int = 1_000_000) -> None:
        self.now = int(start_ms)

    def __call__(self) -> int:
        return int(self.now)

    def advance(self, ms: int) -> None:
        self.now += int(ms)


class LivenessMap:
    """Configurable process-birth liveness probe for hermetic tests."""

    def __init__(self) -> None:
        self._by_id: dict[str, OwnerLiveness] = {}
        self._default = OwnerLiveness.ALIVE

    def set(self, birth: ProcessBirthIdentity, status: OwnerLiveness) -> None:
        self._by_id[process_birth_id(birth)] = status

    def set_default(self, status: OwnerLiveness) -> None:
        self._default = status

    def __call__(self, birth: ProcessBirthIdentity) -> OwnerLiveness:
        return self._by_id.get(process_birth_id(birth), self._default)


class BirthReaderMap:
    """Optional /proc stand-in for PID-reuse cross-checks (hermetic)."""

    def __init__(self) -> None:
        self._by_pid: dict[int, ProcessBirthIdentity | None] = {}

    def set(self, birth: ProcessBirthIdentity | None, *, pid: int | None = None) -> None:
        if birth is None:
            if pid is None:
                raise ValueError("pid is required when clearing a birth reader entry")
            self._by_pid[int(pid)] = None
            return
        self._by_pid[int(birth.pid if pid is None else pid)] = birth

    def __call__(self, pid: int) -> ProcessBirthIdentity | None:
        if int(pid) in self._by_pid:
            return self._by_pid[int(pid)]
        # Default: no host observation (do not consult real /proc).
        return None


def _birth(
    pid: int,
    *,
    start_time_ticks: int = 100,
    boot_id: str = "boot-a",
    parent_pid: int = 1,
) -> ProcessBirthIdentity:
    return ProcessBirthIdentity(
        pid=pid,
        start_time_ticks=start_time_ticks,
        boot_id=boot_id,
        parent_pid=parent_pid,
    )


def _open(
    tmp_path: Path,
    *,
    clock: FakeClock | None = None,
    liveness: LivenessMap | None = None,
    birth_reader: BirthReaderMap | None = None,
    session_ttl_ms: int = 60_000,
    heartbeat_ttl_ms: int = 15_000,
    heartbeat_retain: int = 8,
) -> tuple[DaemonRegistry, FakeClock, LivenessMap]:
    clock = clock or FakeClock()
    liveness = liveness or LivenessMap()
    birth_reader = birth_reader or BirthReaderMap()
    registry = open_daemon_registry(
        tmp_path / "daemon_registry.duckdb",
        clock_ms=clock,
        liveness=liveness,
        birth_reader=birth_reader,
        default_session_ttl_ms=session_ttl_ms,
        default_heartbeat_ttl_ms=heartbeat_ttl_ms,
        heartbeat_retain=heartbeat_retain,
    )
    return registry, clock, liveness


def _register_stack(
    registry: DaemonRegistry,
    *,
    role: str = "implementation",
    lane_id: str = "lane-1",
    supervisor_birth: ProcessBirthIdentity | None = None,
    daemon_birth: ProcessBirthIdentity | None = None,
) -> tuple[object, object]:
    supervisor = registry.register_supervisor(
        repository_id="repository:test",
        process_birth=supervisor_birth or _birth(10, start_time_ticks=1000),
    )
    daemon = registry.register_daemon(
        supervisor_id=supervisor.supervisor_id,
        role=role,
        lane_id=lane_id,
        process_birth=daemon_birth or _birth(20, start_time_ticks=2000),
    )
    return supervisor, daemon


# ---------------------------------------------------------------------------
# Interface identities
# ---------------------------------------------------------------------------


def test_interface_identities() -> None:
    assert DAEMON_REGISTRY_INTERFACE == "DaemonRegistry@1"
    assert SUPERVISOR_INSTANCE_INTERFACE == "SupervisorInstance@1"
    assert DAEMON_INSTANCE_INTERFACE == "DaemonInstance@1"
    assert DAEMON_SESSION_INTERFACE == "DaemonSession@1"
    assert HEARTBEAT_INTERFACE == "Heartbeat@1"
    assert DaemonRegistry.INTERFACE == DAEMON_REGISTRY_INTERFACE


def test_process_birth_id_differs_on_pid_reuse() -> None:
    first = _birth(42, start_time_ticks=100)
    reused = _birth(42, start_time_ticks=999)
    assert process_birth_id(first) != process_birth_id(reused)
    assert not process_births_match(first, reused)
    assert process_births_match(first, _birth(42, start_time_ticks=100))


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_registration_binds_run_role_lane_and_birth(tmp_path: Path) -> None:
    registry, _clock, _live = _open(tmp_path)
    try:
        supervisor, daemon = _register_stack(registry, role="master", lane_id="lane-a")
        session = registry.open_session(
            daemon_id=daemon.daemon_id,
            process_birth=_birth(30, start_time_ticks=3000),
            role="master",
            lane_id="lane-a",
            shard_id="shard-0",
            server_id="server:1",
            quack_connection="http://127.0.0.1:4242",
            capability={"gpu": False},
            deadline_ms=2_000_000,
        )
        assert session.role == "master"
        assert session.lane_id == "lane-a"
        assert session.shard_id == "shard-0"
        assert session.server_id == "server:1"
        assert session.quack_connection.startswith("http://127.0.0.1")
        assert session.process_birth.start_time_ticks == 3000
        assert session.progress_cursor == ""
        assert session.exit_disposition is ExitDisposition.RUNNING
        assert session.restart_disposition is RestartDisposition.NONE
        assert session.fencing_token >= 1
        assert session.fence_epoch >= 1
        loaded = registry.get_session(session.session_id)
        assert loaded is not None
        assert loaded.process_birth_id == session.process_birth_id
        assert registry.get_supervisor(supervisor.supervisor_id) is not None
        assert registry.get_daemon(daemon.daemon_id) is not None
    finally:
        registry.close()


def test_raw_pid_never_proves_identity_on_register(tmp_path: Path) -> None:
    registry, _clock, _live = _open(tmp_path)
    try:
        with pytest.raises(DaemonRegistryIdentityError, match="start_time_ticks"):
            registry.register_supervisor(
                repository_id="repository:test",
                process_birth=ProcessBirthIdentity(
                    pid=99, start_time_ticks=0, boot_id="boot"
                ),
            )
    finally:
        registry.close()


# ---------------------------------------------------------------------------
# Heartbeat identity / renew rules
# ---------------------------------------------------------------------------


def test_heartbeat_rejects_raw_pid_only(tmp_path: Path) -> None:
    registry, _clock, _live = _open(tmp_path)
    try:
        _supervisor, daemon = _register_stack(registry)
        birth = _birth(31, start_time_ticks=3100)
        session = registry.open_session(
            daemon_id=daemon.daemon_id,
            process_birth=birth,
        )
        with pytest.raises(DaemonRegistryIdentityError, match="raw PID"):
            registry.heartbeat(session.session_id, pid_only=birth.pid)
    finally:
        registry.close()


def test_dead_process_birth_cannot_renew(tmp_path: Path) -> None:
    registry, clock, liveness = _open(tmp_path)
    try:
        _supervisor, daemon = _register_stack(registry)
        birth = _birth(32, start_time_ticks=3200)
        session = registry.open_session(
            daemon_id=daemon.daemon_id,
            process_birth=birth,
        )
        liveness.set(birth, OwnerLiveness.DEAD)
        with pytest.raises(DaemonRegistryIdentityError, match="dead"):
            registry.heartbeat(session.session_id, process_birth=birth)
        loaded = registry.get_session(session.session_id)
        assert loaded is not None
        assert loaded.status is SessionStatus.EXPIRED
    finally:
        registry.close()


def test_unknown_process_birth_cannot_renew(tmp_path: Path) -> None:
    registry, _clock, liveness = _open(tmp_path)
    try:
        _supervisor, daemon = _register_stack(registry)
        birth = _birth(33, start_time_ticks=3300)
        session = registry.open_session(
            daemon_id=daemon.daemon_id,
            process_birth=birth,
        )
        liveness.set(birth, OwnerLiveness.UNKNOWN)
        with pytest.raises(DaemonRegistryIdentityError, match="unknown"):
            registry.heartbeat(session.session_id, process_birth=birth)
        # Unknown is fail-closed for renew but does not auto-expire.
        loaded = registry.get_session(session.session_id)
        assert loaded is not None
        assert loaded.status is SessionStatus.ACTIVE
    finally:
        registry.close()


def test_pid_reuse_cannot_renew(tmp_path: Path) -> None:
    registry, _clock, liveness = _open(tmp_path)
    try:
        _supervisor, daemon = _register_stack(registry)
        original = _birth(34, start_time_ticks=3400)
        session = registry.open_session(
            daemon_id=daemon.daemon_id,
            process_birth=original,
        )
        reused = _birth(34, start_time_ticks=9999)
        # Liveness of the *claimed* birth may look alive, but factors differ.
        liveness.set(reused, OwnerLiveness.ALIVE)
        with pytest.raises(DaemonRegistryIdentityError, match="does not match"):
            registry.heartbeat(session.session_id, process_birth=reused)
        # Even with matching birth_id claim forged from a different process,
        # process_birth_id differs so claim by id also fails.
        with pytest.raises(DaemonRegistryIdentityError):
            registry.heartbeat(
                session.session_id,
                process_birth_id_claim=process_birth_id(reused),
            )
    finally:
        registry.close()


def test_os_cross_check_detects_pid_reuse_under_claimed_birth(tmp_path: Path) -> None:
    """When /proc shows a different birth under the same PID, renew fails closed."""

    births = BirthReaderMap()
    registry, _clock, _live = _open(tmp_path, birth_reader=births)
    try:
        _supervisor, daemon = _register_stack(registry)
        claimed = _birth(36, start_time_ticks=3600)
        session = registry.open_session(
            daemon_id=daemon.daemon_id,
            process_birth=claimed,
        )
        # Claim still presents the original multi-factor identity, but the OS
        # reader observes a reused PID with different start ticks.
        births.set(_birth(36, start_time_ticks=99999))
        with pytest.raises(DaemonRegistryIdentityError, match="reused"):
            registry.heartbeat(session.session_id, process_birth=claimed)
        loaded = registry.get_session(session.session_id)
        assert loaded is not None
        assert loaded.status is SessionStatus.EXPIRED
    finally:
        registry.close()


def test_heartbeat_extends_session_when_birth_alive(tmp_path: Path) -> None:
    registry, clock, live = _open(tmp_path, heartbeat_ttl_ms=10_000)
    session_id: str | None = None
    expected_expiry = 0
    heartbeat_at = 0
    try:
        _supervisor, daemon = _register_stack(registry)
        birth = _birth(35, start_time_ticks=3500)
        session = registry.open_session(
            daemon_id=daemon.daemon_id,
            process_birth=birth,
            ttl_ms=30_000,
        )
        original_expiry = session.expires_at_ms
        clock.advance(5_000)
        expected_expiry = clock.now + 50_000
        beat = registry.heartbeat(
            session.session_id,
            process_birth=birth,
            ttl_ms=50_000,
        )
        assert beat.to_dict()["kind"] == "heartbeat"
        assert beat.to_dict()["is_progress"] is False
        assert beat.expires_at_ms == expected_expiry
        loaded = registry.get_session(session.session_id)
        assert loaded is not None
        assert loaded.expires_at_ms == expected_expiry
        assert loaded.expires_at_ms > original_expiry
        assert loaded.last_heartbeat_at_ms == clock.now
        assert loaded.progress_cursor == ""
        assert loaded.progress_updated_at_ms == 0
        session_id = session.session_id
        heartbeat_at = clock.now
    finally:
        registry.close()

    assert session_id is not None
    # Durable re-open must observe the extended expiry (not only in-memory).
    reopened = open_daemon_registry(
        tmp_path / "daemon_registry.duckdb",
        clock_ms=clock,
        liveness=live,
        birth_reader=BirthReaderMap(),
    )
    try:
        durable = reopened.get_session(session_id)
        assert durable is not None
        assert durable.expires_at_ms == expected_expiry
        assert durable.last_heartbeat_at_ms == heartbeat_at
    finally:
        reopened.close()


# ---------------------------------------------------------------------------
# Adoption
# ---------------------------------------------------------------------------


def test_adoption_requires_exact_matching_live_birth(tmp_path: Path) -> None:
    registry, _clock, liveness = _open(tmp_path)
    try:
        _supervisor, daemon = _register_stack(registry)
        birth = _birth(40, start_time_ticks=4000)
        session = registry.open_session(
            daemon_id=daemon.daemon_id,
            process_birth=birth,
        )
        adopted = registry.adopt_session(session.session_id, process_birth=birth)
        assert adopted.session_id == session.session_id

        with pytest.raises(DaemonRegistryIdentityError, match="does not match"):
            registry.adopt_session(
                session.session_id,
                process_birth=_birth(40, start_time_ticks=4001),
            )

        liveness.set(birth, OwnerLiveness.DEAD)
        with pytest.raises(DaemonRegistryIdentityError, match="dead"):
            registry.adopt_session(session.session_id, process_birth=birth)

        liveness.set(birth, OwnerLiveness.UNKNOWN)
        with pytest.raises(DaemonRegistryIdentityError, match="unknown"):
            registry.adopt_session(session.session_id, process_birth=birth)
    finally:
        registry.close()


# ---------------------------------------------------------------------------
# Role/lane fencing
# ---------------------------------------------------------------------------


def test_duplicate_active_role_lane_ownership_is_fenced(tmp_path: Path) -> None:
    registry, _clock, _live = _open(tmp_path)
    try:
        _supervisor, daemon = _register_stack(registry, role="worker", lane_id="lane-2")
        first_birth = _birth(50, start_time_ticks=5000)
        first = registry.open_session(
            daemon_id=daemon.daemon_id,
            process_birth=first_birth,
            role="worker",
            lane_id="lane-2",
        )
        assert first.status is SessionStatus.ACTIVE

        second_birth = _birth(51, start_time_ticks=5100)
        with pytest.raises(DaemonRegistryConflictError, match="fenced"):
            registry.open_session(
                daemon_id=daemon.daemon_id,
                process_birth=second_birth,
                role="worker",
                lane_id="lane-2",
            )
        owner = registry.active_session_for_role_lane("worker", "lane-2")
        assert owner is not None
        assert owner.session_id == first.session_id
    finally:
        registry.close()


def test_dead_owner_role_lane_can_be_reclaimed_with_ancestry(tmp_path: Path) -> None:
    registry, _clock, liveness = _open(tmp_path)
    try:
        _supervisor, daemon = _register_stack(registry, role="worker", lane_id="lane-3")
        first_birth = _birth(52, start_time_ticks=5200)
        first = registry.open_session(
            daemon_id=daemon.daemon_id,
            process_birth=first_birth,
            role="worker",
            lane_id="lane-3",
        )
        liveness.set(first_birth, OwnerLiveness.DEAD)
        second_birth = _birth(53, start_time_ticks=5300)
        second = registry.open_session(
            daemon_id=daemon.daemon_id,
            process_birth=second_birth,
            role="worker",
            lane_id="lane-3",
        )
        assert second.session_id != first.session_id
        assert first.session_id in second.ancestry
        assert registry.exact_ancestry(second.session_id) == (first.session_id,)
        prior = registry.get_session(first.session_id)
        assert prior is not None
        assert prior.status is SessionStatus.SUPERSEDED
        assert second.fencing_token > first.fencing_token
    finally:
        registry.close()


def test_unknown_owner_role_lane_remains_fenced(tmp_path: Path) -> None:
    registry, _clock, liveness = _open(tmp_path)
    try:
        _supervisor, daemon = _register_stack(registry, role="worker", lane_id="lane-4")
        first_birth = _birth(54, start_time_ticks=5400)
        registry.open_session(
            daemon_id=daemon.daemon_id,
            process_birth=first_birth,
            role="worker",
            lane_id="lane-4",
        )
        liveness.set(first_birth, OwnerLiveness.UNKNOWN)
        with pytest.raises(DaemonRegistryConflictError, match="unknown"):
            registry.open_session(
                daemon_id=daemon.daemon_id,
                process_birth=_birth(55, start_time_ticks=5500),
                role="worker",
                lane_id="lane-4",
            )
    finally:
        registry.close()


# ---------------------------------------------------------------------------
# Heartbeats vs progress
# ---------------------------------------------------------------------------


def test_heartbeats_and_progress_are_distinct(tmp_path: Path) -> None:
    registry, clock, _live = _open(tmp_path, heartbeat_ttl_ms=10_000)
    try:
        _supervisor, daemon = _register_stack(registry)
        birth = _birth(60, start_time_ticks=6000)
        session = registry.open_session(
            daemon_id=daemon.daemon_id,
            process_birth=birth,
            ttl_ms=30_000,
        )
        expiry_before = session.expires_at_ms
        heartbeat_before = session.last_heartbeat_at_ms

        clock.advance(2_000)
        progress = registry.record_progress(
            session.session_id,
            "phase:implement@2",
            process_birth=birth,
            payload={"step": 2},
        )
        assert progress.to_dict()["kind"] == "progress"
        assert progress.to_dict()["extends_session"] is False

        after_progress = registry.get_session(session.session_id)
        assert after_progress is not None
        assert after_progress.progress_cursor == "phase:implement@2"
        assert after_progress.progress_updated_at_ms == clock.now
        # Progress must not extend liveness / expiry.
        assert after_progress.expires_at_ms == expiry_before
        assert after_progress.last_heartbeat_at_ms == heartbeat_before

        clock.advance(1_000)
        beat = registry.heartbeat(session.session_id, process_birth=birth, ttl_ms=40_000)
        after_beat = registry.get_session(session.session_id)
        assert after_beat is not None
        assert after_beat.expires_at_ms == clock.now + 40_000
        assert after_beat.expires_at_ms > expiry_before
        assert after_beat.last_heartbeat_at_ms == clock.now
        # Heartbeat must not rewrite progress cursor.
        assert after_beat.progress_cursor == "phase:implement@2"
        assert after_beat.progress_updated_at_ms == progress.recorded_at_ms

        assert beat.to_dict()["is_progress"] is False
        heartbeats = registry.list_heartbeats(session.session_id)
        progresses = registry.list_progress(session.session_id)
        assert heartbeats
        assert progresses
        assert all(item.to_dict()["kind"] == "heartbeat" for item in heartbeats)
        assert all(item.to_dict()["kind"] == "progress" for item in progresses)
    finally:
        registry.close()


# ---------------------------------------------------------------------------
# Session expiry and late heartbeat
# ---------------------------------------------------------------------------


def test_session_expiry_and_late_heartbeat(tmp_path: Path) -> None:
    registry, clock, _live = _open(tmp_path, session_ttl_ms=10_000)
    try:
        _supervisor, daemon = _register_stack(registry)
        birth = _birth(70, start_time_ticks=7000)
        session = registry.open_session(
            daemon_id=daemon.daemon_id,
            process_birth=birth,
            ttl_ms=10_000,
        )
        clock.advance(10_001)
        expired_ids = registry.expire_sessions()
        assert session.session_id in expired_ids
        loaded = registry.get_session(session.session_id)
        assert loaded is not None
        assert loaded.status is SessionStatus.EXPIRED

        with pytest.raises(DaemonRegistrySessionError, match="not active|late"):
            registry.heartbeat(session.session_id, process_birth=birth)
    finally:
        registry.close()


def test_late_heartbeat_without_prior_expire_call(tmp_path: Path) -> None:
    registry, clock, _live = _open(tmp_path)
    try:
        _supervisor, daemon = _register_stack(registry)
        birth = _birth(71, start_time_ticks=7100)
        session = registry.open_session(
            daemon_id=daemon.daemon_id,
            process_birth=birth,
            ttl_ms=5_000,
        )
        clock.advance(5_001)
        with pytest.raises(DaemonRegistrySessionError, match="late heartbeat|not active"):
            registry.heartbeat(session.session_id, process_birth=birth)
        loaded = registry.get_session(session.session_id)
        assert loaded is not None
        assert loaded.status is SessionStatus.EXPIRED
    finally:
        registry.close()


# ---------------------------------------------------------------------------
# Heartbeat compaction
# ---------------------------------------------------------------------------


def test_heartbeat_compaction_retains_newest(tmp_path: Path) -> None:
    registry, clock, _live = _open(tmp_path, heartbeat_retain=3)
    try:
        _supervisor, daemon = _register_stack(registry)
        birth = _birth(80, start_time_ticks=8000)
        session = registry.open_session(
            daemon_id=daemon.daemon_id,
            process_birth=birth,
            ttl_ms=1_000_000,
        )
        for _ in range(6):
            clock.advance(100)
            registry.heartbeat(
                session.session_id,
                process_birth=birth,
                ttl_ms=1_000_000,
            )
        # open_session writes one heartbeat, plus 6 more; compaction keeps 3.
        beats = registry.list_heartbeats(session.session_id, limit=50)
        assert len(beats) == 3
        sequences = [item.sequence for item in beats]
        assert sequences == sorted(sequences, reverse=True)
        # Newest retained sequences must be the highest ones (5,6,7) or
        # equivalently the top-3 after open(1)+six renewals.
        assert max(sequences) == 7
        assert min(sequences) == 5

        report = registry.compact_heartbeats(session.session_id, retain=2)
        assert report["retain"] == 2
        assert report["deleted"] >= 1
        beats = registry.list_heartbeats(session.session_id, limit=50)
        assert len(beats) == 2
        assert [item.sequence for item in beats] == [7, 6]
    finally:
        registry.close()


# ---------------------------------------------------------------------------
# Server restart
# ---------------------------------------------------------------------------


def test_server_restart_supersedes_active_sessions(tmp_path: Path) -> None:
    registry, _clock, _live = _open(tmp_path)
    try:
        _supervisor, daemon = _register_stack(registry)
        birth = _birth(90, start_time_ticks=9000)
        session = registry.open_session(
            daemon_id=daemon.daemon_id,
            process_birth=birth,
        )
        assert registry.server_generation == 1
        generation = registry.record_server_restart()
        assert generation == 2
        loaded = registry.get_session(session.session_id)
        assert loaded is not None
        assert loaded.status is SessionStatus.SUPERSEDED
        with pytest.raises(DaemonRegistrySessionError, match="not active"):
            registry.heartbeat(session.session_id, process_birth=birth)

        # Fresh session under new generation is allowed for same role/lane.
        new_birth = _birth(91, start_time_ticks=9100)
        fresh = registry.open_session(
            daemon_id=daemon.daemon_id,
            process_birth=new_birth,
            server_generation=generation,
        )
        assert fresh.server_generation == 2
        assert fresh.status is SessionStatus.ACTIVE
    finally:
        registry.close()


# ---------------------------------------------------------------------------
# Status files: mirror only
# ---------------------------------------------------------------------------


def test_status_files_mirror_but_cannot_create_or_extend(tmp_path: Path) -> None:
    registry, clock, _live = _open(tmp_path)
    try:
        _supervisor, daemon = _register_stack(registry)
        birth = _birth(100, start_time_ticks=10000)
        session = registry.open_session(
            daemon_id=daemon.daemon_id,
            process_birth=birth,
            ttl_ms=30_000,
        )
        expiry_before = session.expires_at_ms
        heartbeat_before = session.last_heartbeat_at_ms

        status_path = tmp_path / "status" / "daemon.status.json"
        clock.advance(3_000)
        mirror = registry.mirror_status_file(status_path, session.session_id)
        assert mirror.authoritative is False
        assert mirror.to_dict()["can_create_session"] is False
        assert mirror.to_dict()["can_extend_session"] is False
        assert status_path.is_file()

        after = registry.get_session(session.session_id)
        assert after is not None
        assert after.expires_at_ms == expiry_before
        assert after.last_heartbeat_at_ms == heartbeat_before

        # Touching / rewriting the file still cannot extend the session.
        clock.advance(10_000)
        status_path.write_text(
            json.dumps(
                {
                    "session_id": session.session_id,
                    "pid": birth.pid,
                    "mirrored_at_ms": clock.now,
                    "fresh": True,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        loaded_mirror = registry.load_status_file(status_path)
        assert loaded_mirror is not None
        assert loaded_mirror.authoritative is False
        # ingest is explicitly non-creating / non-extending
        ingested = registry.ingest_status_file(status_path, now_ms=clock.now)
        assert ingested is not None
        still = registry.get_session(session.session_id)
        assert still is not None
        assert still.expires_at_ms == expiry_before
        assert still.last_heartbeat_at_ms == heartbeat_before

        # A status file alone cannot create a session.
        orphan = tmp_path / "status" / "orphan.status.json"
        orphan.write_text(
            json.dumps(
                {
                    "session_id": "session:forged",
                    "pid": 12345,
                    "role": "worker",
                    "lane_id": "lane-x",
                }
            ),
            encoding="utf-8",
        )
        forged = registry.ingest_status_file(orphan)
        assert forged is not None
        assert registry.get_session("session:forged") is None
        assert registry.active_session_for_role_lane("worker", "lane-x") is None
    finally:
        registry.close()


def test_mirror_unknown_session_fails_closed(tmp_path: Path) -> None:
    registry, _clock, _live = _open(tmp_path)
    try:
        with pytest.raises(DaemonRegistrySessionError, match="unknown"):
            registry.mirror_status_file(tmp_path / "x.status", "session:missing")
    finally:
        registry.close()


# ---------------------------------------------------------------------------
# Stop / dispositions
# ---------------------------------------------------------------------------


def test_stop_session_records_dispositions(tmp_path: Path) -> None:
    registry, _clock, _live = _open(tmp_path)
    try:
        _supervisor, daemon = _register_stack(registry)
        birth = _birth(110, start_time_ticks=11000)
        session = registry.open_session(
            daemon_id=daemon.daemon_id,
            process_birth=birth,
        )
        stopped = registry.stop_session(
            session.session_id,
            process_birth=birth,
            exit_disposition=ExitDisposition.CLEAN,
            restart_disposition=RestartDisposition.NONE,
        )
        assert stopped.status is SessionStatus.STOPPED
        assert stopped.exit_disposition is ExitDisposition.CLEAN
        with pytest.raises(DaemonRegistrySessionError, match="not active"):
            registry.heartbeat(session.session_id, process_birth=birth)
    finally:
        registry.close()


def test_idempotent_supervisor_reregistration(tmp_path: Path) -> None:
    registry, _clock, _live = _open(tmp_path)
    try:
        birth = _birth(120, start_time_ticks=12000)
        first = registry.register_supervisor(
            repository_id="repository:test",
            process_birth=birth,
            supervisor_id="supervisor:fixed",
        )
        again = registry.register_supervisor(
            repository_id="repository:test",
            process_birth=birth,
            supervisor_id="supervisor:fixed",
        )
        assert again.supervisor_id == first.supervisor_id
        assert again.process_birth_id == first.process_birth_id
        with pytest.raises(DaemonRegistryConflictError, match="different process birth"):
            registry.register_supervisor(
                repository_id="repository:test",
                process_birth=_birth(121, start_time_ticks=12100),
                supervisor_id="supervisor:fixed",
            )
    finally:
        registry.close()


def test_dead_birth_cannot_open_session(tmp_path: Path) -> None:
    registry, _clock, liveness = _open(tmp_path)
    try:
        _supervisor, daemon = _register_stack(registry)
        birth = _birth(130, start_time_ticks=13000)
        liveness.set(birth, OwnerLiveness.DEAD)
        with pytest.raises(DaemonRegistryIdentityError, match="dead"):
            registry.open_session(daemon_id=daemon.daemon_id, process_birth=birth)
    finally:
        registry.close()


def test_instance_status_vocab_and_projections(tmp_path: Path) -> None:
    registry, _clock, _live = _open(tmp_path)
    try:
        supervisor = registry.register_supervisor(
            repository_id="repository:test",
            process_birth=_birth(140, start_time_ticks=14000),
            status=InstanceStatus.STARTING,
        )
        assert supervisor.to_dict()["interface"] == SUPERVISOR_INSTANCE_INTERFACE
        assert supervisor.status is InstanceStatus.STARTING
        daemon = registry.register_daemon(
            supervisor_id=supervisor.supervisor_id,
            role="lane",
            process_birth=_birth(141, start_time_ticks=14100),
        )
        assert daemon.to_dict()["interface"] == DAEMON_INSTANCE_INTERFACE
        session = registry.open_session(
            daemon_id=daemon.daemon_id,
            process_birth=_birth(142, start_time_ticks=14200),
        )
        assert session.to_dict()["interface"] == DAEMON_SESSION_INTERFACE
    finally:
        registry.close()
