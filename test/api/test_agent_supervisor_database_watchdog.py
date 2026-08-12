"""Tests for database-derived watchdog, stall diagnostics, and safe repair (DQP-032).

Acceptance:

* Repair requires current expected fence/process birth/generation and is
  idempotent
* No action follows file age alone
* Ready work without valid owner/capacity/dependency reason becomes actionable
* Doctor exposes evidence and abstains when ownership is unknown

Evidence subset: delta-written state, stale mtime, live worker, PID reuse,
no ready shard work, quota backoff, phase deadline, server restart, exact fence.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
    OwnerLiveness,
    ProcessBirthIdentity,
)
from ipfs_accelerate_py.agent_supervisor.rescue.database_watchdog import (
    CLASSIFICATION_VALUES,
    DATABASE_WATCHDOG_INTERFACE,
    FENCED_RECOVERY_COMMAND_INTERFACE,
    STALL_DIAGNOSIS_INTERFACE,
    CommandActionKind,
    CommandStatus,
    DatabaseWatchdog,
    DatabaseWatchdogOwnershipError,
    DoctorDisposition,
    OwnershipState,
    StallClassification,
    WatchdogObservation,
    diagnose_observation,
    duckdb_available,
    open_database_watchdog,
    process_birth_id,
)

pytestmark = pytest.mark.skipif(
    not duckdb_available(),
    reason="DuckDB is required for DatabaseWatchdog hermetic tests",
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DOCTOR_SCRIPT = (
    _REPO_ROOT / "scripts" / "ops" / "agent_supervisor" / "duckdb_quack_doctor.py"
)


class FakeClock:
    def __init__(self, start_ms: int = 1_000_000) -> None:
        self.now = int(start_ms)

    def __call__(self) -> int:
        return int(self.now)

    def advance(self, ms: int) -> None:
        self.now += int(ms)


class LivenessMap:
    def __init__(self) -> None:
        self._by_id: dict[str, OwnerLiveness] = {}
        self._default = OwnerLiveness.ALIVE

    def set(self, birth: ProcessBirthIdentity, status: OwnerLiveness) -> None:
        self._by_id[process_birth_id(birth)] = status

    def set_default(self, status: OwnerLiveness) -> None:
        self._default = status

    def __call__(self, birth: ProcessBirthIdentity) -> OwnerLiveness:
        return self._by_id.get(process_birth_id(birth), self._default)


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
) -> tuple[DatabaseWatchdog, FakeClock, LivenessMap]:
    clock = clock or FakeClock()
    liveness = liveness or LivenessMap()
    watchdog = open_database_watchdog(
        tmp_path / "watchdog.duckdb",
        clock_ms=clock,
        liveness=liveness,
        heartbeat_stale_ms=30_000,
        session_expiry_ms=60_000,
    )
    return watchdog, clock, liveness


def _load_doctor_module():
    spec = importlib.util.spec_from_file_location(
        "duckdb_quack_doctor_under_test",
        _DOCTOR_SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Interface identities
# ---------------------------------------------------------------------------


def test_interface_identities() -> None:
    assert DATABASE_WATCHDOG_INTERFACE == "DatabaseWatchdog@1"
    assert STALL_DIAGNOSIS_INTERFACE == "StallDiagnosis@1"
    assert FENCED_RECOVERY_COMMAND_INTERFACE == "FencedRecoveryCommand@1"
    assert DatabaseWatchdog.INTERFACE == DATABASE_WATCHDOG_INTERFACE
    assert "ready_unclaimable" in CLASSIFICATION_VALUES
    assert "ownership_unknown" in CLASSIFICATION_VALUES
    assert "file_age_only" in CLASSIFICATION_VALUES


def test_authority_policy_rejects_file_and_pid_authority(tmp_path: Path) -> None:
    watchdog, _clock, _live = _open(tmp_path)
    try:
        policy = watchdog.authority_policy()
        assert policy["semantic_authority"] == "database"
        assert policy["file_mtime_authority"] == "none"
        assert policy["raw_pid_authority"] == "none"
        assert policy["lock_deletion"] == "prohibited"
    finally:
        watchdog.close()


# ---------------------------------------------------------------------------
# Classification: healthy, capacity, quiescence, ready-unclaimable
# ---------------------------------------------------------------------------


def test_healthy_active_with_live_session_and_fresh_heartbeat(tmp_path: Path) -> None:
    watchdog, clock, _live = _open(tmp_path)
    try:
        birth = _birth(42)
        obs = WatchdogObservation(
            now_ms=clock(),
            generation=1,
            fence_epoch=1,
            fencing_token=7,
            server_process_birth=birth,
            ownership=OwnershipState.LIVE,
            sessions=[
                {
                    "session_id": "session:1",
                    "status": "active",
                    "last_heartbeat_at_ms": clock() - 1_000,
                    "process_birth": birth.to_dict(),
                    "process_birth_id": process_birth_id(birth),
                    "fence_epoch": 1,
                    "fencing_token": 7,
                }
            ],
        )
        diagnoses = watchdog.diagnose(obs)
        classes = {d.classification for d in diagnoses}
        assert StallClassification.HEALTHY_ACTIVE in classes
        assert all(not d.actionable for d in diagnoses)
    finally:
        watchdog.close()


def test_provider_capacity_backoff_is_not_actionable(tmp_path: Path) -> None:
    watchdog, clock, _live = _open(tmp_path)
    try:
        obs = WatchdogObservation(
            now_ms=clock(),
            generation=1,
            fence_epoch=1,
            ownership=OwnershipState.LIVE,
            provider_capacity={
                "provider_id": "grok_cli",
                "backoff": True,
                "available": 0,
            },
            tasks=[
                {
                    "task_cid": "task:ready-1",
                    "status": "ready",
                }
            ],
        )
        diagnoses = watchdog.diagnose(obs)
        capacity = [
            d
            for d in diagnoses
            if d.classification is StallClassification.PROVIDER_CAPACITY_BACKOFF
        ]
        assert capacity
        assert all(not d.actionable for d in capacity)
        # Capacity is a valid reason; ready work is not ready_unclaimable.
        assert not any(
            d.classification is StallClassification.READY_UNCLAIMABLE
            for d in diagnoses
        )
    finally:
        watchdog.close()


def test_quiescent_strict_shard_when_other_shards_own_ready_work(
    tmp_path: Path,
) -> None:
    watchdog, clock, _live = _open(tmp_path)
    try:
        obs = WatchdogObservation(
            now_ms=clock(),
            generation=1,
            fence_epoch=1,
            ownership=OwnershipState.LIVE,
            shard_id="shard-a",
            tasks=[
                {
                    "task_cid": "task:other",
                    "status": "ready",
                    "shard_id": "shard-b",
                }
            ],
        )
        diagnoses = watchdog.diagnose(obs)
        assert any(
            d.classification is StallClassification.QUIESCENT_STRICT_SHARD
            and not d.actionable
            for d in diagnoses
        )
        assert not any(
            d.classification is StallClassification.READY_UNCLAIMABLE
            for d in diagnoses
        )
    finally:
        watchdog.close()


def test_ready_work_without_owner_capacity_or_dependency_is_actionable(
    tmp_path: Path,
) -> None:
    watchdog, clock, _live = _open(tmp_path)
    try:
        obs = WatchdogObservation(
            now_ms=clock(),
            generation=2,
            fence_epoch=3,
            fencing_token=9,
            ownership=OwnershipState.LIVE,
            tasks=[
                {
                    "task_cid": "task:stuck-ready",
                    "status": "ready",
                    # no owner, no dependency_reason, no capacity backoff
                }
            ],
        )
        diagnoses = watchdog.diagnose(obs)
        ready = [
            d
            for d in diagnoses
            if d.classification is StallClassification.READY_UNCLAIMABLE
        ]
        assert len(ready) == 1
        assert ready[0].actionable is True
        assert ready[0].task_cid == "task:stuck-ready"
        assert ready[0].evidence
    finally:
        watchdog.close()


def test_ready_work_with_dependency_reason_is_not_actionable(tmp_path: Path) -> None:
    watchdog, clock, _live = _open(tmp_path)
    try:
        obs = WatchdogObservation(
            now_ms=clock(),
            generation=1,
            fence_epoch=1,
            ownership=OwnershipState.LIVE,
            tasks=[
                {
                    "task_cid": "task:blocked",
                    "status": "ready",
                    "dependency_reason": "waiting_on_task:upstream",
                    "has_unmet_dependency": True,
                }
            ],
        )
        diagnoses = watchdog.diagnose(obs)
        assert not any(
            d.classification is StallClassification.READY_UNCLAIMABLE
            for d in diagnoses
        )
    finally:
        watchdog.close()


# ---------------------------------------------------------------------------
# File age alone never authorizes action
# ---------------------------------------------------------------------------


def test_file_age_alone_never_actionable_or_repairable(tmp_path: Path) -> None:
    watchdog, clock, _live = _open(tmp_path)
    try:
        obs = WatchdogObservation(
            now_ms=clock(),
            generation=1,
            fence_epoch=1,
            fencing_token=1,
            ownership=OwnershipState.ABSENT,
            file_mirrors=[
                {
                    "path": "state/status.json",
                    "mtime_ms": clock() - 3_600_000,
                    "age_ms": 3_600_000,
                    "stale": True,
                }
            ],
        )
        diagnoses = watchdog.diagnose(obs)
        file_only = [
            d
            for d in diagnoses
            if d.classification is StallClassification.FILE_AGE_ONLY
        ]
        assert file_only
        assert all(not d.actionable for d in file_only)

        command = watchdog.decide_and_apply(
            file_only[0],
            expected_fence_epoch=1,
            expected_fencing_token=1,
            expected_generation=1,
            ownership=OwnershipState.ABSENT,
        )
        assert command.action_kind is CommandActionKind.NO_OP
        assert command.status in {
            CommandStatus.REJECTED,
            CommandStatus.APPLIED,
            CommandStatus.DECIDED,
        }
        # Explicit policy text.
        if command.status is CommandStatus.REJECTED:
            assert "file age" in command.reason.lower()
    finally:
        watchdog.close()


def test_stale_mtime_with_fresh_database_heartbeat_is_healthy(tmp_path: Path) -> None:
    """Delta-written DB state wins over stale status-file mtime."""

    watchdog, clock, _live = _open(tmp_path)
    try:
        birth = _birth(99)
        obs = WatchdogObservation(
            now_ms=clock(),
            generation=1,
            fence_epoch=1,
            ownership=OwnershipState.LIVE,
            sessions=[
                {
                    "session_id": "session:live",
                    "status": "active",
                    "last_heartbeat_at_ms": clock() - 500,
                    "process_birth": birth.to_dict(),
                    "process_birth_id": process_birth_id(birth),
                }
            ],
            file_mirrors=[
                {
                    "path": "state/lane.status.json",
                    "mtime_ms": clock() - 9_999_999,
                    "stale": True,
                }
            ],
        )
        diagnoses = watchdog.diagnose(obs)
        classes = {d.classification for d in diagnoses}
        assert StallClassification.HEALTHY_ACTIVE in classes
        assert StallClassification.FILE_AGE_ONLY in classes
        assert not any(d.actionable for d in diagnoses)
    finally:
        watchdog.close()


# ---------------------------------------------------------------------------
# Stale session / orphan lease with process birth (not PID alone)
# ---------------------------------------------------------------------------


def test_stale_session_with_dead_process_birth_is_actionable(tmp_path: Path) -> None:
    watchdog, clock, liveness = _open(tmp_path)
    try:
        birth = _birth(55, start_time_ticks=1000)
        liveness.set(birth, OwnerLiveness.DEAD)
        obs = WatchdogObservation(
            now_ms=clock(),
            generation=1,
            fence_epoch=2,
            fencing_token=4,
            ownership=OwnershipState.LIVE,
            sessions=[
                {
                    "session_id": "session:dead",
                    "status": "active",
                    "last_heartbeat_at_ms": clock() - 120_000,
                    "process_birth": birth.to_dict(),
                    "process_birth_id": process_birth_id(birth),
                    "fence_epoch": 2,
                    "fencing_token": 4,
                    "server_generation": 1,
                }
            ],
        )
        diagnoses = watchdog.diagnose(obs)
        stale = [
            d
            for d in diagnoses
            if d.classification is StallClassification.STALE_SESSION
        ]
        assert stale
        assert stale[0].actionable is True
    finally:
        watchdog.close()


def test_pid_reuse_is_dead_not_alive(tmp_path: Path) -> None:
    """Same PID with different start ticks is a different process birth."""

    original = _birth(7, start_time_ticks=100)
    reused = _birth(7, start_time_ticks=999)
    assert process_birth_id(original) != process_birth_id(reused)

    diagnoses = diagnose_observation(
        WatchdogObservation(
            now_ms=1_000_000,
            generation=1,
            fence_epoch=1,
            ownership=OwnershipState.LIVE,
            sessions=[
                {
                    "session_id": "session:reused",
                    "status": "active",
                    "last_heartbeat_at_ms": 1_000_000 - 120_000,
                    "process_birth": original.to_dict(),
                    "process_birth_id": process_birth_id(original),
                    "owner_liveness": "dead",
                }
            ],
        )
    )
    assert any(
        d.classification is StallClassification.STALE_SESSION and d.actionable
        for d in diagnoses
    )


def test_orphan_lease_without_live_owner_is_actionable(tmp_path: Path) -> None:
    watchdog, clock, _live = _open(tmp_path)
    try:
        obs = WatchdogObservation(
            now_ms=clock(),
            generation=1,
            fence_epoch=1,
            fencing_token=11,
            ownership=OwnershipState.LIVE,
            leases=[
                {
                    "lease_id": "lease:1",
                    "task_cid": "task:1",
                    "state": "held",
                    "owner_session_id": "session:gone",
                    "owner_alive": False,
                    "owner_liveness": "dead",
                    "fencing_token": 11,
                    "fence_epoch": 1,
                    "process_birth_id": process_birth_id(_birth(3)),
                    "generation": 1,
                }
            ],
        )
        diagnoses = watchdog.diagnose(obs)
        orphan = [
            d
            for d in diagnoses
            if d.classification is StallClassification.ORPHAN_LEASE
        ]
        assert orphan and orphan[0].actionable
    finally:
        watchdog.close()


def test_phase_stall_past_deadline_is_actionable(tmp_path: Path) -> None:
    watchdog, clock, _live = _open(tmp_path)
    try:
        obs = WatchdogObservation(
            now_ms=clock(),
            generation=1,
            fence_epoch=1,
            fencing_token=2,
            ownership=OwnershipState.LIVE,
            attempts=[
                {
                    "attempt_id": "attempt:1",
                    "task_cid": "task:1",
                    "status": "running",
                    "phase": "implement",
                    "phase_deadline_ms": clock() - 60_000,
                    "last_progress_at_ms": clock() - 90_000,
                    "fencing_token": 2,
                    "fence_epoch": 1,
                    "process_birth_id": process_birth_id(_birth(8)),
                    "generation": 1,
                }
            ],
        )
        diagnoses = watchdog.diagnose(obs)
        assert any(
            d.classification is StallClassification.PHASE_STALL and d.actionable
            for d in diagnoses
        )
    finally:
        watchdog.close()


# ---------------------------------------------------------------------------
# Fenced repair: exact fence / process birth / generation + idempotency
# ---------------------------------------------------------------------------


def test_repair_requires_exact_fence_process_birth_and_generation(
    tmp_path: Path,
) -> None:
    watchdog, clock, _live = _open(tmp_path)
    try:
        birth = _birth(21, start_time_ticks=500)
        obs = WatchdogObservation(
            now_ms=clock(),
            generation=5,
            fence_epoch=9,
            fencing_token=13,
            server_process_birth=birth,
            ownership=OwnershipState.LIVE,
            tasks=[{"task_cid": "task:ready", "status": "ready"}],
        )
        diagnoses = watchdog.diagnose(obs)
        ready = next(
            d
            for d in diagnoses
            if d.classification is StallClassification.READY_UNCLAIMABLE
        )

        # Stale fence epoch → rejected.
        rejected = watchdog.decide_repair(
            ready,
            expected_fence_epoch=9,
            expected_fencing_token=13,
            expected_generation=5,
            expected_process_birth_id=process_birth_id(birth),
            expected_process_birth=birth,
            current_fence_epoch=8,  # stale
            current_fencing_token=13,
            current_generation=5,
            current_process_birth_id=process_birth_id(birth),
            current_process_birth=birth,
            ownership=OwnershipState.LIVE,
        )
        assert rejected.status is CommandStatus.REJECTED
        assert "fence" in rejected.reason.lower() or "mismatch" in rejected.reason.lower()

        # Exact match → decided.
        decided = watchdog.decide_repair(
            ready,
            expected_fence_epoch=9,
            expected_fencing_token=13,
            expected_generation=5,
            expected_process_birth_id=process_birth_id(birth),
            expected_process_birth=birth,
            current_fence_epoch=9,
            current_fencing_token=13,
            current_generation=5,
            current_process_birth_id=process_birth_id(birth),
            current_process_birth=birth,
            ownership=OwnershipState.LIVE,
            idempotency_key="repair:ready:exact",
        )
        assert decided.status is CommandStatus.DECIDED
        assert decided.action_kind is CommandActionKind.MARK_READY_ACTIONABLE
        assert decided.expected_fence_epoch == 9
        assert decided.expected_generation == 5
        assert decided.expected_process_birth_id == process_birth_id(birth)

        applied = watchdog.apply_repair(
            decided,
            current_fence_epoch=9,
            current_fencing_token=13,
            current_generation=5,
            current_process_birth_id=process_birth_id(birth),
            current_process_birth=birth,
            ownership=OwnershipState.LIVE,
        )
        assert applied.status is CommandStatus.APPLIED
        assert applied.result_digest
        assert applied.body.get("apply_result", {}).get("raw_pid_signal") is False
        assert applied.body.get("apply_result", {}).get("lock_deletion") is False
    finally:
        watchdog.close()


def test_repair_is_idempotent_on_idempotency_key(tmp_path: Path) -> None:
    watchdog, clock, _live = _open(tmp_path)
    try:
        birth = _birth(30)
        obs = WatchdogObservation(
            now_ms=clock(),
            generation=1,
            fence_epoch=1,
            fencing_token=1,
            server_process_birth=birth,
            ownership=OwnershipState.LIVE,
            leases=[
                {
                    "lease_id": "lease:idem",
                    "task_cid": "task:idem",
                    "state": "held",
                    "owner_alive": False,
                    "owner_liveness": "dead",
                    "fencing_token": 1,
                    "fence_epoch": 1,
                    "process_birth_id": process_birth_id(birth),
                    "generation": 1,
                }
            ],
        )
        diagnoses = watchdog.diagnose(obs)
        orphan = next(
            d
            for d in diagnoses
            if d.classification is StallClassification.ORPHAN_LEASE
        )
        key = "idempotent-repair-key-001"
        first = watchdog.decide_and_apply(
            orphan,
            expected_fence_epoch=1,
            expected_fencing_token=1,
            expected_generation=1,
            expected_process_birth_id=process_birth_id(birth),
            expected_process_birth=birth,
            idempotency_key=key,
            ownership=OwnershipState.LIVE,
        )
        assert first.status is CommandStatus.APPLIED
        second = watchdog.decide_and_apply(
            orphan,
            expected_fence_epoch=1,
            expected_fencing_token=1,
            expected_generation=1,
            expected_process_birth_id=process_birth_id(birth),
            expected_process_birth=birth,
            idempotency_key=key,
            ownership=OwnershipState.LIVE,
        )
        assert second.status is CommandStatus.REPLAYED
        assert second.command_id == first.command_id
        assert second.result_digest == first.result_digest

        third = watchdog.apply_repair(
            first.command_id,
            current_fence_epoch=1,
            current_fencing_token=1,
            current_generation=1,
            current_process_birth_id=process_birth_id(birth),
            ownership=OwnershipState.LIVE,
        )
        assert third.status is CommandStatus.REPLAYED
        assert third.command_id == first.command_id
    finally:
        watchdog.close()


def test_stale_generation_rejects_apply(tmp_path: Path) -> None:
    watchdog, clock, _live = _open(tmp_path)
    try:
        birth = _birth(40)
        obs = WatchdogObservation(
            now_ms=clock(),
            generation=3,
            fence_epoch=1,
            fencing_token=1,
            ownership=OwnershipState.LIVE,
            worktrees=[
                {
                    "worktree_id": "worktree:1",
                    "state": "active",
                    "owner_alive": False,
                    "owner_liveness": "dead",
                    "fencing_token": 1,
                    "fence_epoch": 1,
                    "process_birth_id": process_birth_id(birth),
                    "generation": 3,
                }
            ],
        )
        diagnoses = watchdog.diagnose(obs)
        orphan = next(
            d
            for d in diagnoses
            if d.classification is StallClassification.ORPHAN_WORKTREE
        )
        decided = watchdog.decide_repair(
            orphan,
            expected_fence_epoch=1,
            expected_fencing_token=1,
            expected_generation=3,
            expected_process_birth_id=process_birth_id(birth),
            current_fence_epoch=1,
            current_fencing_token=1,
            current_generation=3,
            current_process_birth_id=process_birth_id(birth),
            ownership=OwnershipState.LIVE,
            idempotency_key="gen-stale-apply",
        )
        assert decided.status is CommandStatus.DECIDED
        # Server restarted → generation advanced.
        applied = watchdog.apply_repair(
            decided,
            current_fence_epoch=1,
            current_fencing_token=1,
            current_generation=4,  # rotated
            current_process_birth_id=process_birth_id(birth),
            ownership=OwnershipState.LIVE,
        )
        assert applied.status is CommandStatus.REJECTED
    finally:
        watchdog.close()


# ---------------------------------------------------------------------------
# Doctor: evidence + abstain when ownership unknown
# ---------------------------------------------------------------------------


def test_doctor_abstains_when_ownership_unknown_and_exposes_evidence(
    tmp_path: Path,
) -> None:
    watchdog, clock, _live = _open(tmp_path)
    try:
        obs = WatchdogObservation(
            now_ms=clock(),
            generation=1,
            fence_epoch=1,
            fencing_token=1,
            ownership=OwnershipState.UNKNOWN,
            tasks=[{"task_cid": "task:ready", "status": "ready"}],
            leases=[
                {
                    "lease_id": "lease:x",
                    "state": "held",
                    "owner_alive": False,
                    "owner_liveness": "dead",
                }
            ],
        )
        report = watchdog.doctor(obs, propose_repairs=True)
        assert report.disposition is DoctorDisposition.ABSTAIN
        assert report.abstain is True
        assert report.ownership is OwnershipState.UNKNOWN
        assert report.evidence  # exposes evidence
        assert any(e.kind == "ownership" for e in report.evidence)
        assert any(
            d.classification is StallClassification.OWNERSHIP_UNKNOWN
            for d in report.diagnoses
        )
        # Nothing actionable while ownership is unknown.
        assert all(not d.actionable for d in report.diagnoses)
        assert report.commands == ()

        # Repair path also abstains / refuses.
        ownership_diag = next(
            d
            for d in report.diagnoses
            if d.classification is StallClassification.OWNERSHIP_UNKNOWN
        )
        command = watchdog.decide_repair(
            ownership_diag,
            expected_fence_epoch=1,
            expected_fencing_token=1,
            expected_generation=1,
            ownership=OwnershipState.UNKNOWN,
        )
        assert command.status is CommandStatus.ABSTAINED
        assert command.action_kind is CommandActionKind.ABSTAIN
    finally:
        watchdog.close()


def test_apply_raises_when_ownership_becomes_unknown(tmp_path: Path) -> None:
    watchdog, clock, _live = _open(tmp_path)
    try:
        obs = WatchdogObservation(
            now_ms=clock(),
            generation=1,
            fence_epoch=1,
            fencing_token=1,
            ownership=OwnershipState.LIVE,
            tasks=[{"task_cid": "task:r", "status": "ready"}],
        )
        diagnoses = watchdog.diagnose(obs)
        ready = next(
            d
            for d in diagnoses
            if d.classification is StallClassification.READY_UNCLAIMABLE
        )
        decided = watchdog.decide_repair(
            ready,
            expected_fence_epoch=1,
            expected_fencing_token=1,
            expected_generation=1,
            ownership=OwnershipState.LIVE,
            idempotency_key="ownership-flip",
        )
        assert decided.status is CommandStatus.DECIDED
        with pytest.raises(DatabaseWatchdogOwnershipError, match="unknown"):
            watchdog.apply_repair(
                decided,
                current_fence_epoch=1,
                current_fencing_token=1,
                current_generation=1,
                ownership=OwnershipState.UNKNOWN,
            )
    finally:
        watchdog.close()


def test_doctor_actionable_disposition_with_evidence(tmp_path: Path) -> None:
    watchdog, clock, _live = _open(tmp_path)
    try:
        obs = WatchdogObservation(
            now_ms=clock(),
            generation=1,
            fence_epoch=1,
            fencing_token=1,
            ownership=OwnershipState.LIVE,
            tasks=[{"task_cid": "task:a", "status": "ready"}],
        )
        report = watchdog.doctor(obs, propose_repairs=True)
        assert report.disposition is DoctorDisposition.ACTIONABLE
        assert not report.abstain
        assert report.evidence
        assert report.commands
        assert report.commands[0].action_kind is CommandActionKind.MARK_READY_ACTIONABLE
    finally:
        watchdog.close()


# ---------------------------------------------------------------------------
# Server / merge / recovery classifications
# ---------------------------------------------------------------------------


def test_server_fault_and_merge_blockage_classifications(tmp_path: Path) -> None:
    watchdog, clock, _live = _open(tmp_path)
    try:
        obs = WatchdogObservation(
            now_ms=clock(),
            generation=1,
            fence_epoch=1,
            ownership=OwnershipState.LIVE,
            server={"server_id": "server:1", "status": "failed"},
            merges=[
                {
                    "merge_id": "merge:1",
                    "status": "blocked",
                    "task_cid": "task:1",
                    "fencing_token": 1,
                    "fence_epoch": 1,
                }
            ],
            recoveries=[{"recovery_id": "recovery:1", "status": "exhausted"}],
            migrations={"migration_id": "mig:1", "status": "failed"},
            backup={"backup_id": "bak:1", "status": "corrupt", "age_ms": 99},
        )
        diagnoses = watchdog.diagnose(obs)
        classes = {d.classification for d in diagnoses}
        assert StallClassification.SERVER_FAULT in classes
        assert StallClassification.MERGE_BLOCKAGE in classes
        assert StallClassification.RECOVERY_BLOCKAGE in classes
        assert StallClassification.MIGRATION_FAULT in classes
        assert StallClassification.BACKUP_FAULT in classes
    finally:
        watchdog.close()


# ---------------------------------------------------------------------------
# Doctor CLI facade
# ---------------------------------------------------------------------------


def test_doctor_cli_classifications_and_authority(tmp_path: Path, capsys) -> None:
    module = _load_doctor_module()
    code = module.run(["classifications", "--pretty"])
    assert code == module.EXIT_SUCCESS
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert "ready_unclaimable" in payload["classifications"]
    assert payload["policy"]["file_age_alone_action"] is False
    assert payload["policy"]["ownership_unknown"] == "abstain"

    db = tmp_path / "cli-watchdog.duckdb"
    code = module.run(["authority", "--database", str(db)])
    assert code == module.EXIT_SUCCESS
    payload = json.loads(capsys.readouterr().out)
    assert payload["authority"]["file_mtime_authority"] == "none"


def test_doctor_cli_abstains_on_unknown_ownership(tmp_path: Path, capsys) -> None:
    module = _load_doctor_module()
    obs_path = tmp_path / "obs.json"
    obs_path.write_text(
        json.dumps(
            {
                "now_ms": 1_000_000,
                "generation": 1,
                "fence_epoch": 1,
                "ownership": "unknown",
                "tasks": [{"task_cid": "task:1", "status": "ready"}],
            }
        ),
        encoding="utf-8",
    )
    db = tmp_path / "cli-watchdog.duckdb"
    code = module.run(
        [
            "doctor",
            "--database",
            str(db),
            "--observation-json",
            str(obs_path),
        ]
    )
    assert code == module.EXIT_ABSTAIN
    report = json.loads(capsys.readouterr().out)
    assert report["abstain"] is True
    assert report["disposition"] == "abstain"
    assert report["evidence"]
    assert any(e.get("kind") == "ownership" for e in report["evidence"])


def test_doctor_cli_reports_actionable_ready_work(tmp_path: Path, capsys) -> None:
    module = _load_doctor_module()
    obs_path = tmp_path / "obs.json"
    obs_path.write_text(
        json.dumps(
            {
                "now_ms": 1_000_000,
                "generation": 1,
                "fence_epoch": 1,
                "fencing_token": 1,
                "ownership": "live",
                "tasks": [{"task_cid": "task:actionable", "status": "ready"}],
            }
        ),
        encoding="utf-8",
    )
    db = tmp_path / "cli-watchdog.duckdb"
    code = module.run(
        [
            "diagnose",
            "--database",
            str(db),
            "--observation-json",
            str(obs_path),
            "--propose-repairs",
        ]
    )
    assert code == module.EXIT_ACTIONABLE
    report = json.loads(capsys.readouterr().out)
    assert report["disposition"] == "actionable"
    assert any(
        d["classification"] == "ready_unclaimable" and d["actionable"]
        for d in report["diagnoses"]
    )


def test_doctor_cli_repair_is_idempotent(tmp_path: Path, capsys) -> None:
    module = _load_doctor_module()
    birth = _birth(77)
    obs = {
        "now_ms": 1_000_000,
        "generation": 1,
        "fence_epoch": 1,
        "fencing_token": 1,
        "ownership": "live",
        "server_process_birth": birth.to_dict(),
        "server_process_birth_id": process_birth_id(birth),
        "tasks": [{"task_cid": "task:cli-repair", "status": "ready"}],
    }
    obs_path = tmp_path / "obs.json"
    obs_path.write_text(json.dumps(obs), encoding="utf-8")
    db = tmp_path / "cli-watchdog.duckdb"
    args = [
        "repair",
        "--database",
        str(db),
        "--observation-json",
        str(obs_path),
        "--idempotency-key",
        "cli-repair-key",
        "--expected-fence-epoch",
        "1",
        "--expected-fencing-token",
        "1",
        "--expected-generation",
        "1",
        "--expected-process-birth-id",
        process_birth_id(birth),
    ]
    code1 = module.run(args)
    assert code1 == module.EXIT_SUCCESS
    first = json.loads(capsys.readouterr().out)
    assert first["status"] == "applied"
    code2 = module.run(args)
    assert code2 == module.EXIT_SUCCESS
    second = json.loads(capsys.readouterr().out)
    assert second["status"] == "replayed"
    assert second["command_id"] == first["command_id"]


def test_pure_diagnose_observation_without_open_store() -> None:
    """Cold diagnose path works without opening a database."""

    diagnoses = diagnose_observation(
        {
            "now_ms": 50_000,
            "generation": 1,
            "fence_epoch": 1,
            "ownership": "live",
            "tasks": [{"task_cid": "task:x", "status": "ready"}],
        }
    )
    assert any(
        d.classification is StallClassification.READY_UNCLAIMABLE and d.actionable
        for d in diagnoses
    )
