"""Deterministic multi-process tests for fenced worktree lifecycle (ASI-171)."""

from __future__ import annotations

import inspect
import json
import multiprocessing as mp
import os
from pathlib import Path

import ipfs_accelerate_py.agent_supervisor.worktree_lifecycle as lifecycle_module
import pytest
from ipfs_accelerate_py.agent_supervisor.control.control_contracts import EventCursor
from ipfs_accelerate_py.agent_supervisor.merge.campaign_leases import CampaignLeaseCoordinator
from ipfs_accelerate_py.agent_supervisor.rescue.learning_recovery import LearningCheckpointAdapter
from ipfs_accelerate_py.agent_supervisor.runtime.learning_checkpoint import (
    L3ResourceKind,
    LearningCheckpointBinding,
    StaleFenceError,
)
from ipfs_accelerate_py.agent_supervisor.worktree_lifecycle import (
    DEFAULT_LEASE_SECONDS,
    FENCED_WORKTREE_LIFECYCLE_REQUIREMENT_ID,
    CleanupDisposition,
    DuplicateAttemptError,
    FenceMismatchError,
    LifecycleFailureKind,
    OwnerLiveness,
    OwnershipError,
    ProcessBirthIdentity,
    WorkspaceLifecycleState,
    WorktreeLifecycleError,
    WorktreeLifecycleStore,
    current_process_birth,
    lifecycle_race_result,
    owner_liveness,
    proc_available,
    read_process_birth,
)

pytestmark = pytest.mark.skipif(
    os.name != "posix",
    reason="worktree lifecycle fencing tests require POSIX process birth",
)


class FakeClock:
    def __init__(self, start: float = 1_000.0) -> None:
        self.now = float(start)

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += float(seconds)


def _store(
    tmp_path: Path,
    *,
    lease_seconds: float = 60.0,
    startup_grace_seconds: float = 5.0,
    clock: FakeClock | None = None,
    proc_root: Path | None = None,
) -> WorktreeLifecycleStore:
    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    (repo / ".git").mkdir(exist_ok=True)
    return WorktreeLifecycleStore(
        repo_root=repo,
        lease_seconds=lease_seconds,
        startup_grace_seconds=startup_grace_seconds,
        clock=clock or FakeClock(),
        proc_root=proc_root or Path("/proc"),
        store_dir=tmp_path / "lifecycle",
    )


def test_requirement_id_is_stable() -> None:
    assert FENCED_WORKTREE_LIFECYCLE_REQUIREMENT_ID.startswith("asi-171:")


def test_begin_preparing_publishes_before_worktree_visibility(tmp_path: Path) -> None:
    store = _store(tmp_path)
    workspace = tmp_path / "worktrees" / "task-a-attempt-1"
    # Intentionally do not create the directory yet: the claim must be
    # cleanup-visible before git worktree add.
    record = store.begin_preparing(
        task_id="ASI-171",
        canonical_task_cid="task:asi-171",
        attempt=1,
        lane_id="lane-1",
        workspace_path=workspace,
        branch="implementation/asi-171-attempt-1",
        merge_target="main",
    )
    assert record.state is WorkspaceLifecycleState.PREPARING
    assert record.fence == 1
    assert not workspace.exists()
    loaded = store.load_workspace(workspace)
    assert loaded is not None
    assert loaded.lease_id == record.lease_id
    decision = store.evaluate_cleanup(workspace_path=workspace)
    assert not decision.allowed
    assert decision.failure_kind is LifecycleFailureKind.LIFECYCLE_RACE
    assert decision.provider_call_allowed is False
    assert decision.attempt_consumed is False


def test_owner_transitions_and_only_owner_may_advance(tmp_path: Path) -> None:
    store = _store(tmp_path)
    workspace = tmp_path / "ws"
    record = store.begin_preparing(
        task_id="T1",
        attempt=1,
        lane_id="lane",
        workspace_path=workspace,
        branch="implementation/t1",
        merge_target="main",
    )
    active = store.mark_active(
        workspace,
        lease_id=record.lease_id,
        expected_fence=record.fence,
    )
    assert active.state is WorkspaceLifecycleState.ACTIVE
    assert active.fence == record.fence + 1
    with pytest.raises(FenceMismatchError):
        store.mark_settling(
            workspace,
            lease_id=record.lease_id,
            expected_fence=record.fence,  # stale fence
        )
    settling = store.mark_settling(
        workspace,
        lease_id=active.lease_id,
        expected_fence=active.fence,
    )
    terminal = store.mark_terminal(
        workspace,
        lease_id=settling.lease_id,
        expected_fence=settling.fence,
        reason="merged",
    )
    assert terminal.state is WorkspaceLifecycleState.TERMINAL
    decision = store.evaluate_cleanup(workspace_path=workspace)
    assert decision.allowed
    assert decision.reason == "terminal_record"


def test_renew_lease_updates_workspace_and_exact_task_index(
    tmp_path: Path,
) -> None:
    clock = FakeClock(1_000.0)
    store = _store(tmp_path, clock=clock)
    workspace = tmp_path / "renew"
    record = store.begin_preparing(
        task_id="RENEW",
        canonical_task_cid="cid:renew",
        attempt=3,
        lane_id="lane",
        workspace_path=workspace,
        branch="implementation/renew",
        merge_target="main",
        state_dir=str(tmp_path / "state"),
        owner=ProcessBirthIdentity(
            pid=2**30 - 9,
            start_time_ticks=1,
            boot_id="dead-boot",
        ),
    )
    index_path = store.task_index_path_for(
        canonical_task_cid=record.canonical_task_cid,
        task_id=record.task_id,
        attempt=record.attempt,
    )
    clock.advance(5.0)

    renewed = store.renew_lease(
        workspace,
        lease_id=record.lease_id,
        expected_fence=record.fence,
    )

    assert renewed.fence == record.fence + 1
    assert renewed.updated_at == clock.now
    assert renewed.expires_at == clock.now + store.lease_seconds
    assert json.loads(
        store.workspace_path_for(workspace).read_text(encoding="utf-8")
    ) == renewed.to_dict()
    assert json.loads(index_path.read_text(encoding="utf-8")) == {
        "schema": renewed.schema,
        "workspace_path": renewed.workspace_path,
        "record_id": renewed.record_id,
        "task_id": renewed.task_id,
        "canonical_task_cid": renewed.canonical_task_cid,
        "attempt": renewed.attempt,
        "fence": renewed.fence,
        "lease_id": renewed.lease_id,
        "state": renewed.state.value,
    }
    assert store.require_exact_dead_owner(
        workspace,
        expected_record_id=renewed.record_id,
        expected_fence=renewed.fence,
        expected_lease_id=renewed.lease_id,
        expected_task_id=renewed.task_id,
        expected_canonical_task_cid=renewed.canonical_task_cid,
        expected_attempt=renewed.attempt,
        expected_branch=renewed.branch,
        expected_merge_target=renewed.merge_target,
        expected_repo_root=renewed.repo_root,
        expected_state_dir=renewed.state_dir,
    ) == renewed


def test_renew_lease_rejects_mismatched_index_without_writes(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    workspace = tmp_path / "renew-index-mismatch"
    record = store.begin_preparing(
        task_id="RENEW-INDEX",
        canonical_task_cid="cid:renew-index",
        attempt=1,
        lane_id="lane",
        workspace_path=workspace,
        branch="implementation/renew-index",
        merge_target="main",
    )
    record_path = store.workspace_path_for(workspace)
    index_path = store.task_index_path_for(
        canonical_task_cid=record.canonical_task_cid,
        task_id=record.task_id,
        attempt=record.attempt,
    )
    index_payload = json.loads(index_path.read_text(encoding="utf-8"))
    index_payload["fence"] = record.fence + 1
    index_path.write_text(json.dumps(index_payload), encoding="utf-8")
    before_record = record_path.read_bytes()
    before_index = index_path.read_bytes()

    with pytest.raises(
        WorktreeLifecycleError,
        match="task index mismatch",
    ):
        store.renew_lease(
            workspace,
            lease_id=record.lease_id,
            expected_fence=record.fence,
        )

    assert record_path.read_bytes() == before_record
    assert index_path.read_bytes() == before_index


def test_renew_lease_preserves_race_winner_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _store(tmp_path)
    workspace = tmp_path / "renew-race"
    record = store.begin_preparing(
        task_id="RENEW-RACE",
        canonical_task_cid="cid:renew-race",
        attempt=1,
        lane_id="lane",
        workspace_path=workspace,
        branch="implementation/renew-race",
        merge_target="main",
    )
    record_path = store.workspace_path_for(workspace)
    index_path = store.task_index_path_for(
        canonical_task_cid=record.canonical_task_cid,
        task_id=record.task_id,
        attempt=record.attempt,
    )
    original_load = store._load_strict_workspace_record
    raced: list[tuple[bytes, bytes]] = []

    def race_after_capture(target: str | Path):
        captured = original_load(target)
        if not raced:
            replacement_payload = captured.to_dict()
            replacement_payload["lane_id"] = "replacement-lane"
            record_path.write_text(
                json.dumps(
                    replacement_payload,
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            raced.append((record_path.read_bytes(), index_path.read_bytes()))
        return captured

    monkeypatch.setattr(
        store,
        "_load_strict_workspace_record",
        race_after_capture,
    )

    with pytest.raises(
        WorktreeLifecycleError,
        match="record changed during lease renewal",
    ):
        store.renew_lease(
            workspace,
            lease_id=record.lease_id,
            expected_fence=record.fence,
        )

    assert len(raced) == 1
    assert record_path.read_bytes() == raced[0][0]
    assert index_path.read_bytes() == raced[0][1]


def test_peer_cleanup_skips_preparing_even_when_branch_merged(tmp_path: Path) -> None:
    """Reproduce the 2026-07-28 race: branch tip == merge target, no child yet."""

    store = _store(tmp_path)
    workspace = tmp_path / "worktrees" / "race"
    workspace.mkdir(parents=True)
    record = store.begin_preparing(
        task_id="ASI-RACE",
        canonical_task_cid="cid:race",
        attempt=1,
        lane_id="owner",
        workspace_path=workspace,
        branch="implementation/race-attempt-1",
        merge_target="main",
    )
    peer = WorktreeLifecycleStore(
        repo_root=store.repo_root,
        lease_seconds=store.lease_seconds,
        startup_grace_seconds=store.startup_grace_seconds,
        clock=store.clock,
        proc_root=store.proc_root,
        store_dir=store.store_dir,
    )
    decision = peer.authorize_cleanup(
        workspace_path=workspace,
        branch=record.branch,
        caller_lease_id="peer-lease",
    )
    assert not decision.allowed
    assert "nonterminal_preparing" in decision.reason or decision.reason.endswith(
        "owner_alive"
    )
    assert decision.provider_call_allowed is False
    assert decision.attempt_consumed is False


def test_stale_reclamation_requires_expiry_and_advances_fence(tmp_path: Path) -> None:
    clock = FakeClock(1_000.0)
    store = _store(tmp_path, lease_seconds=10.0, startup_grace_seconds=0.0, clock=clock)
    workspace = tmp_path / "stale-ws"
    dead_owner = ProcessBirthIdentity(
        pid=2**30 - 7,  # almost certainly not a live PID
        start_time_ticks=1,
        boot_id="dead-boot",
    )
    record = store.begin_preparing(
        task_id="STALE",
        attempt=2,
        lane_id="lane",
        workspace_path=workspace,
        branch="implementation/stale",
        merge_target="main",
        owner=dead_owner,
    )
    early = store.evaluate_cleanup(workspace_path=workspace)
    assert not early.allowed
    assert early.reason in {
        "owner_dead_lease_unexpired",
        "preparing_startup_grace",
    }
    clock.advance(11.0)
    decision = store.authorize_cleanup(
        workspace_path=workspace,
        caller_lease_id="reclaimer",
    )
    assert decision.allowed
    assert decision.reason == "reclaimed_stale_record"
    assert decision.record is not None
    assert decision.record.fence == record.fence + 1
    assert decision.record.state is WorkspaceLifecycleState.TERMINAL


def test_controlled_restart_reclaims_only_dead_same_lane_owner(
    tmp_path: Path,
) -> None:
    clock = FakeClock(1_000.0)
    store = _store(
        tmp_path,
        lease_seconds=600.0,
        startup_grace_seconds=0.0,
        clock=clock,
    )
    lane_state = tmp_path / "state" / "lane-1"
    other_state = tmp_path / "state" / "lane-2"
    dead_owner = ProcessBirthIdentity(
        pid=2**30 - 9,
        start_time_ticks=1,
        boot_id="dead-boot",
    )
    dead_workspace = tmp_path / "dead-same-lane"
    dead_record = store.begin_preparing(
        task_id="RESTART-DEAD",
        canonical_task_cid="cid:restart-dead",
        attempt=1,
        lane_id="lane-1",
        workspace_path=dead_workspace,
        branch="implementation/restart-dead",
        merge_target="main",
        state_dir=str(lane_state),
        owner=dead_owner,
    )
    dead_record_path = store.workspace_path_for(dead_workspace)
    dead_index_path = store.task_index_path_for(
        canonical_task_cid=dead_record.canonical_task_cid,
        task_id=dead_record.task_id,
        attempt=dead_record.attempt,
    )
    other_workspace = tmp_path / "dead-other-lane"
    other_record = store.begin_preparing(
        task_id="RESTART-OTHER",
        canonical_task_cid="cid:restart-other",
        attempt=1,
        lane_id="lane-2",
        workspace_path=other_workspace,
        branch="implementation/restart-other",
        merge_target="main",
        state_dir=str(other_state),
        owner=dead_owner,
    )
    other_record_path = store.workspace_path_for(other_workspace)
    other_index_path = store.task_index_path_for(
        canonical_task_cid=other_record.canonical_task_cid,
        task_id=other_record.task_id,
        attempt=other_record.attempt,
    )
    live_workspace = tmp_path / "live-same-lane"
    live_record = store.begin_preparing(
        task_id="RESTART-LIVE",
        canonical_task_cid="cid:restart-live",
        attempt=1,
        lane_id="lane-1",
        workspace_path=live_workspace,
        branch="implementation/restart-live",
        merge_target="main",
        state_dir=str(lane_state),
    )
    live_record_path = store.workspace_path_for(live_workspace)
    live_index_path = store.task_index_path_for(
        canonical_task_cid=live_record.canonical_task_cid,
        task_id=live_record.task_id,
        attempt=live_record.attempt,
    )

    assert store.evaluate_cleanup(
        workspace_path=dead_workspace
    ).reason == "owner_dead_lease_unexpired"
    dead_before_wrong_lane = (
        dead_record_path.read_bytes(),
        dead_index_path.read_bytes(),
    )
    assert (
        store.reclaim_dead_owner_for_controlled_restart(
            dead_workspace,
            expected_state_dir="",
        )
        is None
    )
    assert dead_record_path.read_bytes() == dead_before_wrong_lane[0]
    assert dead_index_path.read_bytes() == dead_before_wrong_lane[1]
    assert (
        store.reclaim_dead_owner_for_controlled_restart(
            dead_workspace,
            expected_state_dir=other_state,
        )
        is None
    )
    assert dead_record_path.read_bytes() == dead_before_wrong_lane[0]
    assert dead_index_path.read_bytes() == dead_before_wrong_lane[1]
    other_before = (
        other_record_path.read_bytes(),
        other_index_path.read_bytes(),
    )
    live_before = (
        live_record_path.read_bytes(),
        live_index_path.read_bytes(),
    )

    recovered = store.reclaim_dead_owners_for_controlled_restart(
        expected_state_dir=lane_state,
        reclaimer_lease_id="restart-reclaimer-lease",
    )

    assert [record.task_id for record in recovered] == ["RESTART-DEAD"]
    terminal = store.load_workspace(dead_workspace)
    assert terminal is not None
    assert recovered[0] == terminal
    assert terminal.state is WorkspaceLifecycleState.TERMINAL
    assert terminal.fence == dead_record.fence + 1
    assert terminal.lease_id == "restart-reclaimer-lease"
    assert terminal.expires_at == clock.now
    assert terminal.terminal_reason == "controlled_restart_dead_owner"
    assert json.loads(dead_index_path.read_text(encoding="utf-8")) == {
        "schema": terminal.schema,
        "workspace_path": terminal.workspace_path,
        "record_id": terminal.record_id,
        "task_id": terminal.task_id,
        "canonical_task_cid": terminal.canonical_task_cid,
        "attempt": terminal.attempt,
        "fence": terminal.fence,
        "lease_id": terminal.lease_id,
        "state": terminal.state.value,
    }
    assert store.load_workspace(other_workspace).is_nonterminal
    assert store.load_workspace(live_workspace).is_nonterminal
    assert other_record_path.read_bytes() == other_before[0]
    assert other_index_path.read_bytes() == other_before[1]
    assert live_record_path.read_bytes() == live_before[0]
    assert live_index_path.read_bytes() == live_before[1]


def test_controlled_restart_reclaim_preserves_replacement_claim_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _store(tmp_path, startup_grace_seconds=0.0)
    workspace = tmp_path / "controlled-restart-replacement"
    state_dir = tmp_path / "state" / "lane"
    dead_owner = ProcessBirthIdentity(
        pid=2**30 - 9,
        start_time_ticks=1,
        boot_id="dead-boot",
    )
    original = store.begin_preparing(
        task_id="RESTART-ORIGINAL",
        canonical_task_cid="cid:restart-original",
        attempt=1,
        lane_id="lane",
        workspace_path=workspace,
        branch="implementation/restart-original",
        merge_target="main",
        state_dir=str(state_dir),
        owner=dead_owner,
    )
    original_require = store.require_exact_dead_owner
    replacement_snapshots: list[tuple[Path, bytes, Path, bytes]] = []

    def replace_after_capture(*args, **kwargs):
        terminal = store.mark_terminal(
            workspace,
            lease_id=original.lease_id,
            expected_fence=original.fence,
            reason="replacement-race",
        )
        assert store.compare_and_delete(
            workspace,
            expected_fence=terminal.fence,
            lease_id=terminal.lease_id,
        )
        replacement = store.begin_preparing(
            task_id=original.task_id,
            canonical_task_cid=original.canonical_task_cid,
            attempt=original.attempt,
            lane_id="replacement-lane",
            workspace_path=workspace,
            branch=original.branch,
            merge_target=original.merge_target,
            lease_id=f"{original.lease_id}-replacement",
            state_dir=str(state_dir),
            owner=ProcessBirthIdentity(
                pid=2**30 - 11,
                start_time_ticks=2,
                boot_id="other-dead-boot",
            ),
        )
        assert replacement.lease_id != original.lease_id
        replacement_path = store.workspace_path_for(workspace)
        replacement_index = store.task_index_path_for(
            canonical_task_cid=replacement.canonical_task_cid,
            task_id=replacement.task_id,
            attempt=replacement.attempt,
        )
        replacement_snapshots.append(
            (
                replacement_path,
                replacement_path.read_bytes(),
                replacement_index,
                replacement_index.read_bytes(),
            )
        )
        return original_require(*args, **kwargs)

    monkeypatch.setattr(
        store,
        "require_exact_dead_owner",
        replace_after_capture,
    )

    assert (
        store.reclaim_dead_owner_for_controlled_restart(
            workspace,
            expected_state_dir=state_dir,
        )
        is None
    )

    assert len(replacement_snapshots) == 1
    record_path, record_bytes, index_path, index_bytes = (
        replacement_snapshots[0]
    )
    assert record_path.read_bytes() == record_bytes
    assert index_path.read_bytes() == index_bytes


def test_controlled_restart_reclaim_pins_captured_owner_before_precheck(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _store(tmp_path, startup_grace_seconds=0.0)
    workspace = tmp_path / "controlled-restart-owner-replacement"
    state_dir = tmp_path / "state" / "lane"
    record = store.begin_preparing(
        task_id="RESTART-OWNER-REPLACEMENT",
        canonical_task_cid="cid:restart-owner-replacement",
        attempt=1,
        lane_id="lane",
        workspace_path=workspace,
        branch="implementation/restart-owner-replacement",
        merge_target="main",
        state_dir=str(state_dir),
        owner=ProcessBirthIdentity(
            pid=2**30 - 9,
            start_time_ticks=1,
            boot_id="dead-boot",
        ),
    )
    record_path = store.workspace_path_for(workspace)
    index_path = store.task_index_path_for(
        canonical_task_cid=record.canonical_task_cid,
        task_id=record.task_id,
        attempt=record.attempt,
    )
    original_require = store.require_exact_dead_owner
    replacement_snapshots: list[tuple[bytes, bytes]] = []

    def replace_owner_before_precheck(*args, **kwargs):
        replacement_payload = record.to_dict()
        replacement_payload["owner"] = ProcessBirthIdentity(
            pid=2**30 - 11,
            start_time_ticks=2,
            boot_id="replacement-dead-boot",
        ).to_dict()
        record_path.write_text(
            json.dumps(
                replacement_payload,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        replacement_snapshots.append(
            (record_path.read_bytes(), index_path.read_bytes())
        )
        return original_require(*args, **kwargs)

    monkeypatch.setattr(
        store,
        "require_exact_dead_owner",
        replace_owner_before_precheck,
    )

    assert (
        store.reclaim_dead_owner_for_controlled_restart(
            workspace,
            expected_state_dir=state_dir,
        )
        is None
    )

    assert len(replacement_snapshots) == 1
    assert record_path.read_bytes() == replacement_snapshots[0][0]
    assert index_path.read_bytes() == replacement_snapshots[0][1]


def test_controlled_restart_reclaim_preserves_post_precheck_race_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _store(tmp_path, startup_grace_seconds=0.0)
    workspace = tmp_path / "controlled-restart-race"
    state_dir = tmp_path / "state" / "lane"
    record = store.begin_preparing(
        task_id="RESTART-RACE",
        canonical_task_cid="cid:restart-race",
        attempt=1,
        lane_id="lane",
        workspace_path=workspace,
        branch="implementation/restart-race",
        merge_target="main",
        state_dir=str(state_dir),
        owner=ProcessBirthIdentity(
            pid=2**30 - 9,
            start_time_ticks=1,
            boot_id="dead-boot",
        ),
    )
    record_path = store.workspace_path_for(workspace)
    index_path = store.task_index_path_for(
        canonical_task_cid=record.canonical_task_cid,
        task_id=record.task_id,
        attempt=record.attempt,
    )
    original_require = store.require_exact_dead_owner
    raced: list[tuple[bytes, bytes]] = []

    def race_after_precheck(*args, **kwargs):
        checked = original_require(*args, **kwargs)
        replacement_payload = checked.to_dict()
        replacement_payload["owner"] = ProcessBirthIdentity(
            pid=2**30 - 11,
            start_time_ticks=2,
            boot_id="replacement-dead-boot",
        ).to_dict()
        record_path.write_text(
            json.dumps(
                replacement_payload,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        raced.append((record_path.read_bytes(), index_path.read_bytes()))
        return checked

    monkeypatch.setattr(
        store,
        "require_exact_dead_owner",
        race_after_precheck,
    )

    assert (
        store.reclaim_dead_owner_for_controlled_restart(
            workspace,
            expected_state_dir=state_dir,
        )
        is None
    )

    assert len(raced) == 1
    assert record_path.read_bytes() == raced[0][0]
    assert index_path.read_bytes() == raced[0][1]


def test_partial_finalize_repair_completes_workspace_first_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _store(tmp_path, startup_grace_seconds=0.0)
    workspace = tmp_path / "partial-finalize"
    predecessor = store.begin_preparing(
        task_id="PARTIAL-FINALIZE",
        canonical_task_cid="cid:partial-finalize",
        attempt=2,
        lane_id="dead-lane",
        workspace_path=workspace,
        branch="implementation/partial-finalize",
        merge_target="main",
        state_dir=str(tmp_path / "state"),
        owner=ProcessBirthIdentity(
            pid=2**30 - 19,
            start_time_ticks=1,
            boot_id="dead-boot",
        ),
    )
    predecessor = store.mark_active(
        workspace,
        lease_id=predecessor.lease_id,
        expected_fence=predecessor.fence,
    )
    terminal_reason = "prepared-receipt-dead-owner"
    record_path = store.workspace_path_for(workspace)
    index_path = store.task_index_path_for(
        canonical_task_cid=predecessor.canonical_task_cid,
        task_id=predecessor.task_id,
        attempt=predecessor.attempt,
    )
    predecessor_index_bytes = index_path.read_bytes()
    original_atomic_write = lifecycle_module._atomic_write_json
    crashed: list[Path] = []

    def crash_after_workspace_replace(path: Path, payload) -> None:
        original_atomic_write(path, payload)
        if path == record_path and not crashed:
            crashed.append(path)
            raise RuntimeError("injected crash after terminal workspace write")

    monkeypatch.setattr(
        lifecycle_module,
        "_atomic_write_json",
        crash_after_workspace_replace,
    )
    with pytest.raises(
        RuntimeError,
        match="injected crash after terminal workspace write",
    ):
        store.finalize_exact_dead_owner(
            workspace,
            expected_record_id=predecessor.record_id,
            expected_fence=predecessor.fence,
            expected_lease_id=predecessor.lease_id,
            expected_owner=predecessor.owner,
            expected_task_id=predecessor.task_id,
            expected_canonical_task_cid=predecessor.canonical_task_cid,
            expected_attempt=predecessor.attempt,
            expected_branch=predecessor.branch,
            expected_merge_target=predecessor.merge_target,
            expected_repo_root=predecessor.repo_root,
            expected_state_dir=predecessor.state_dir,
            reason=terminal_reason,
            now=1_234.0,
            retain_terminal=True,
        )
    monkeypatch.setattr(
        lifecycle_module,
        "_atomic_write_json",
        original_atomic_write,
    )

    assert crashed == [record_path]
    partial_record_bytes = record_path.read_bytes()
    partial = store._load_strict_workspace_record(workspace)
    assert partial.is_terminal
    assert partial.fence == predecessor.fence + 1
    assert partial.owner == predecessor.owner
    assert partial.lease_id == predecessor.lease_id
    assert partial.terminal_reason == terminal_reason
    assert index_path.read_bytes() == predecessor_index_bytes

    repaired = store.repair_partial_finalize(
        workspace,
        expected_terminal=partial,
        expected_preterminal_state=predecessor.state,
    )

    assert repaired == partial
    assert record_path.read_bytes() == partial_record_bytes
    assert json.loads(index_path.read_text(encoding="utf-8")) == (
        store._task_index_payload(partial)
    )


def test_partial_finalize_repair_preserves_replacement_index_race_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _store(tmp_path, startup_grace_seconds=0.0)
    workspace = tmp_path / "partial-finalize-index-race"
    predecessor = store.begin_preparing(
        task_id="PARTIAL-FINALIZE-RACE",
        canonical_task_cid="cid:partial-finalize-race",
        attempt=1,
        lane_id="dead-lane",
        workspace_path=workspace,
        branch="implementation/partial-finalize-race",
        merge_target="main",
        state_dir=str(tmp_path / "state"),
        owner=ProcessBirthIdentity(
            pid=2**30 - 23,
            start_time_ticks=1,
            boot_id="dead-boot",
        ),
    )
    terminal_reason = "prepared-receipt-dead-owner"
    record_path = store.workspace_path_for(workspace)
    index_path = store.task_index_path_for(
        canonical_task_cid=predecessor.canonical_task_cid,
        task_id=predecessor.task_id,
        attempt=predecessor.attempt,
    )
    original_atomic_write = lifecycle_module._atomic_write_json
    crashed: list[Path] = []

    def crash_after_workspace_replace(path: Path, payload) -> None:
        original_atomic_write(path, payload)
        if path == record_path and not crashed:
            crashed.append(path)
            raise RuntimeError("injected crash after terminal workspace write")

    monkeypatch.setattr(
        lifecycle_module,
        "_atomic_write_json",
        crash_after_workspace_replace,
    )
    with pytest.raises(RuntimeError):
        store.finalize_exact_dead_owner(
            workspace,
            expected_record_id=predecessor.record_id,
            expected_fence=predecessor.fence,
            expected_lease_id=predecessor.lease_id,
            expected_owner=predecessor.owner,
            expected_task_id=predecessor.task_id,
            expected_canonical_task_cid=predecessor.canonical_task_cid,
            expected_attempt=predecessor.attempt,
            expected_branch=predecessor.branch,
            expected_merge_target=predecessor.merge_target,
            expected_repo_root=predecessor.repo_root,
            expected_state_dir=predecessor.state_dir,
            reason=terminal_reason,
            now=1_345.0,
            retain_terminal=True,
        )
    monkeypatch.setattr(
        lifecycle_module,
        "_atomic_write_json",
        original_atomic_write,
    )
    partial_record_bytes = record_path.read_bytes()
    partial = store._load_strict_workspace_record(workspace)
    replacement_index = json.loads(index_path.read_text(encoding="utf-8"))
    replacement_index["fence"] = predecessor.fence + 7
    replacement_index["lease_id"] = "replacement-index-winner"
    original_lock = lifecycle_module.serialized_lock_update
    raced: list[tuple[bytes, bytes]] = []
    lock_paths: list[Path] = []

    def replace_index_before_lock(path: Path, **kwargs):
        lock_paths.append(path)
        if path == index_path and not raced:
            index_path.write_text(
                json.dumps(replacement_index, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            raced.append(
                (record_path.read_bytes(), index_path.read_bytes())
            )
        return original_lock(path, **kwargs)

    monkeypatch.setattr(
        lifecycle_module,
        "serialized_lock_update",
        replace_index_before_lock,
    )

    with pytest.raises(
        WorktreeLifecycleError,
        match="task index mismatch",
    ):
        store.repair_partial_finalize(
            workspace,
            expected_terminal=partial,
            expected_preterminal_state=predecessor.state,
        )

    assert len(raced) == 1
    assert lock_paths == [index_path, record_path]
    assert raced[0][0] == partial_record_bytes
    assert record_path.read_bytes() == raced[0][0]
    assert index_path.read_bytes() == raced[0][1]


def test_exact_dead_owner_adoption_does_not_wait_for_lease_expiry(
    tmp_path: Path,
) -> None:
    clock = FakeClock(1_000.0)
    store = _store(
        tmp_path,
        lease_seconds=60.0,
        startup_grace_seconds=0.0,
        clock=clock,
    )
    workspace = tmp_path / "orphan"
    state_dir = tmp_path / "state"
    dead_owner = ProcessBirthIdentity(
        pid=2**30 - 7,
        start_time_ticks=1,
        boot_id="dead-boot",
    )
    record = store.begin_preparing(
        task_id="ORPHAN",
        canonical_task_cid="cid:orphan",
        attempt=2,
        lane_id="dead-lane",
        workspace_path=workspace,
        branch="implementation/orphan-attempt-2",
        merge_target="main",
        state_dir=str(state_dir),
        owner=dead_owner,
    )
    active = store.mark_active(
        workspace,
        lease_id=record.lease_id,
        expected_fence=record.fence,
    )
    assert clock.now < active.expires_at

    adopted = store.adopt_dead_owner(
        workspace,
        expected_record_id=active.record_id,
        expected_fence=active.fence,
        expected_lease_id=active.lease_id,
        expected_task_id=active.task_id,
        expected_canonical_task_cid=active.canonical_task_cid,
        expected_attempt=active.attempt,
        expected_branch=active.branch,
        expected_merge_target=active.merge_target,
        expected_repo_root=active.repo_root,
        expected_state_dir=active.state_dir,
        lane_id="reconciliation-lane",
    )

    assert adopted.state is WorkspaceLifecycleState.ACTIVE
    assert adopted.fence == active.fence + 1
    assert adopted.lease_id != active.lease_id
    assert adopted.lease_id
    assert adopted.owner.pid == os.getpid()
    assert adopted.owner.start_time_ticks > 0
    assert adopted.owner.boot_id
    assert adopted.lane_id == "reconciliation-lane"
    assert adopted.expires_at == clock.now + store.lease_seconds
    assert store.load_workspace(workspace) == adopted
    assert (
        store.load_task_attempt(
            canonical_task_cid=adopted.canonical_task_cid,
            task_id=adopted.task_id,
            attempt=adopted.attempt,
        )
        == adopted
    )


@pytest.mark.parametrize(
    ("owner_kind", "expected_error"),
    [
        ("alive", OwnershipError),
        ("unknown", OwnershipError),
        ("malformed", OwnershipError),
        ("missing_stat", OwnershipError),
        ("mismatched", OwnershipError),
        ("index", WorktreeLifecycleError),
    ],
)
def test_dead_owner_adoption_failures_preserve_durable_records(
    tmp_path: Path,
    owner_kind: str,
    expected_error: type[Exception],
) -> None:
    proc_root = (
        tmp_path / "missing-proc"
        if owner_kind == "unknown"
        else (
            tmp_path / "malformed-proc"
            if owner_kind in {"malformed", "missing_stat"}
            else None
        )
    )
    if owner_kind in {"malformed", "missing_stat"}:
        assert proc_root is not None
        malformed_pid = 42_424_242
        (proc_root / str(malformed_pid)).mkdir(parents=True)
        if owner_kind == "malformed":
            (proc_root / str(malformed_pid) / "stat").write_text(
                "readable but malformed",
                encoding="utf-8",
            )
    store = _store(
        tmp_path,
        lease_seconds=60.0,
        startup_grace_seconds=0.0,
        proc_root=proc_root,
    )
    workspace = tmp_path / "orphan"
    owner = (
        ProcessBirthIdentity(
            pid=1,
            start_time_ticks=1,
            boot_id="unknown",
        )
        if owner_kind == "unknown"
        else (
            ProcessBirthIdentity(
                pid=malformed_pid,
                start_time_ticks=1,
                boot_id="malformed-boot",
            )
            if owner_kind in {"malformed", "missing_stat"}
            else current_process_birth()
            if owner_kind == "alive"
            else ProcessBirthIdentity(
                pid=2**30 - 7,
                start_time_ticks=1,
                boot_id="dead-boot",
            )
        )
    )
    record = store.begin_preparing(
        task_id="ORPHAN",
        canonical_task_cid="cid:orphan",
        attempt=2,
        lane_id="original-lane",
        workspace_path=workspace,
        branch="implementation/orphan-attempt-2",
        merge_target="main",
        state_dir=str(tmp_path / "state"),
        owner=owner,
    )
    active = store.mark_active(
        workspace,
        lease_id=record.lease_id,
        expected_fence=record.fence,
    )
    record_path = store.workspace_path_for(workspace)
    index_path = store.task_index_path_for(
        canonical_task_cid=active.canonical_task_cid,
        task_id=active.task_id,
        attempt=active.attempt,
    )
    if owner_kind == "index":
        index_payload = json.loads(index_path.read_text(encoding="utf-8"))
        index_payload["schema"] = "invalid"
        index_path.write_text(
            json.dumps(index_payload),
            encoding="utf-8",
        )
    before_record = record_path.read_bytes()
    before_index = index_path.read_bytes()

    with pytest.raises(expected_error):
        store.adopt_dead_owner(
            workspace,
            expected_record_id=active.record_id,
            expected_fence=active.fence,
            expected_lease_id=active.lease_id,
            expected_task_id=active.task_id,
            expected_canonical_task_cid=(
                "cid:mismatched"
                if owner_kind == "mismatched"
                else active.canonical_task_cid
            ),
            expected_attempt=active.attempt,
            expected_branch=active.branch,
            expected_merge_target=active.merge_target,
            expected_repo_root=active.repo_root,
            expected_state_dir=active.state_dir,
            lane_id="reconciliation-lane",
        )

    assert record_path.read_bytes() == before_record
    assert index_path.read_bytes() == before_index
    assert store.load_workspace(workspace) == active


def test_malformed_readable_proc_stat_is_unknown(tmp_path: Path) -> None:
    proc_root = tmp_path / "proc"
    pid = 1234
    (proc_root / str(pid)).mkdir(parents=True)
    (proc_root / str(pid) / "stat").write_text(
        "1234 malformed",
        encoding="utf-8",
    )
    owner = ProcessBirthIdentity(
        pid=pid,
        start_time_ticks=10,
        boot_id="boot",
    )

    with pytest.raises(OSError):
        read_process_birth(pid, proc_root=proc_root)
    assert (
        owner_liveness(owner, proc_root=proc_root)
        is OwnerLiveness.UNKNOWN
    )


def test_present_pid_directory_with_missing_stat_is_unknown(
    tmp_path: Path,
) -> None:
    proc_root = tmp_path / "proc"
    pid = 1234
    (proc_root / str(pid)).mkdir(parents=True)
    owner = ProcessBirthIdentity(
        pid=pid,
        start_time_ticks=10,
        boot_id="boot",
    )

    with pytest.raises(OSError):
        read_process_birth(pid, proc_root=proc_root)
    assert (
        owner_liveness(owner, proc_root=proc_root)
        is OwnerLiveness.UNKNOWN
    )


def test_dead_owner_adoption_has_no_replacement_identity_injection() -> None:
    parameters = inspect.signature(
        WorktreeLifecycleStore.adopt_dead_owner
    ).parameters
    assert "owner" not in parameters
    assert "lease_id" not in parameters


@pytest.mark.parametrize("failure", ["unprovable_owner", "reused_lease"])
def test_dead_owner_adoption_requires_live_current_owner_and_fresh_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    store = _store(tmp_path, startup_grace_seconds=0.0)
    workspace = tmp_path / "orphan"
    record = store.begin_preparing(
        task_id="ORPHAN",
        canonical_task_cid="cid:orphan",
        attempt=1,
        lane_id="dead",
        workspace_path=workspace,
        branch="implementation/orphan",
        merge_target="main",
        state_dir=str(tmp_path / "state"),
        owner=ProcessBirthIdentity(
            pid=2**30 - 7,
            start_time_ticks=1,
            boot_id="dead-boot",
        ),
    )
    active = store.mark_active(
        workspace,
        lease_id=record.lease_id,
        expected_fence=record.fence,
    )
    if failure == "unprovable_owner":
        monkeypatch.setattr(
            lifecycle_module,
            "current_process_birth",
            lambda **_kwargs: ProcessBirthIdentity(
                pid=os.getpid(),
                start_time_ticks=0,
                boot_id="",
            ),
        )
    else:
        monkeypatch.setattr(
            lifecycle_module,
            "new_lease_id",
            lambda **_kwargs: active.lease_id,
        )
    record_path = store.workspace_path_for(workspace)
    index_path = store.task_index_path_for(
        canonical_task_cid=active.canonical_task_cid,
        task_id=active.task_id,
        attempt=active.attempt,
    )
    before_record = record_path.read_bytes()
    before_index = index_path.read_bytes()

    with pytest.raises(OwnershipError):
        store.adopt_dead_owner(
            workspace,
            expected_record_id=active.record_id,
            expected_fence=active.fence,
            expected_lease_id=active.lease_id,
            expected_task_id=active.task_id,
            expected_canonical_task_cid=active.canonical_task_cid,
            expected_attempt=active.attempt,
            expected_branch=active.branch,
            expected_merge_target=active.merge_target,
            expected_repo_root=active.repo_root,
            expected_state_dir=active.state_dir,
            lane_id="reconciliation",
        )

    assert record_path.read_bytes() == before_record
    assert index_path.read_bytes() == before_index


@pytest.mark.parametrize(
    "corruption",
    ["owner", "record_id", "schema", "owner_boot_id"],
)
def test_dead_owner_adoption_rejects_noncanonical_persisted_record(
    tmp_path: Path,
    corruption: str,
) -> None:
    store = _store(tmp_path, startup_grace_seconds=0.0)
    workspace = tmp_path / "orphan"
    record = store.begin_preparing(
        task_id="ORPHAN",
        canonical_task_cid="cid:orphan",
        attempt=2,
        lane_id="dead",
        workspace_path=workspace,
        branch="implementation/orphan",
        merge_target="main",
        state_dir=str(tmp_path / "state"),
        owner=ProcessBirthIdentity(
            pid=2**30 - 7,
            start_time_ticks=1,
            boot_id="dead-boot",
        ),
    )
    active = store.mark_active(
        workspace,
        lease_id=record.lease_id,
        expected_fence=record.fence,
    )
    record_path = store.workspace_path_for(workspace)
    index_path = store.task_index_path_for(
        canonical_task_cid=active.canonical_task_cid,
        task_id=active.task_id,
        attempt=active.attempt,
    )
    payload = json.loads(record_path.read_text(encoding="utf-8"))
    if corruption == "owner_boot_id":
        payload["owner"]["boot_id"] = ""
    else:
        payload.pop(corruption)
    record_path.write_text(json.dumps(payload), encoding="utf-8")
    before_record = record_path.read_bytes()
    before_index = index_path.read_bytes()

    with pytest.raises(WorktreeLifecycleError):
        store.adopt_dead_owner(
            workspace,
            expected_record_id=active.record_id,
            expected_fence=active.fence,
            expected_lease_id=active.lease_id,
            expected_task_id=active.task_id,
            expected_canonical_task_cid=active.canonical_task_cid,
            expected_attempt=active.attempt,
            expected_branch=active.branch,
            expected_merge_target=active.merge_target,
            expected_repo_root=active.repo_root,
            expected_state_dir=active.state_dir,
            lane_id="reconciliation",
        )

    assert record_path.read_bytes() == before_record
    assert index_path.read_bytes() == before_index


@pytest.mark.parametrize("stale_field", ["record_id", "fence", "lease", "terminal"])
def test_dead_owner_adoption_rejects_stale_or_terminal_authority(
    tmp_path: Path,
    stale_field: str,
) -> None:
    store = _store(tmp_path, startup_grace_seconds=0.0)
    workspace = tmp_path / "orphan"
    record = store.begin_preparing(
        task_id="ORPHAN",
        canonical_task_cid="cid:orphan",
        attempt=2,
        lane_id="dead",
        workspace_path=workspace,
        branch="implementation/orphan",
        merge_target="main",
        state_dir=str(tmp_path / "state"),
        owner=ProcessBirthIdentity(
            pid=2**30 - 7,
            start_time_ticks=1,
            boot_id="dead-boot",
        ),
    )
    current = store.mark_active(
        workspace,
        lease_id=record.lease_id,
        expected_fence=record.fence,
    )
    if stale_field == "terminal":
        current = store.mark_terminal(
            workspace,
            lease_id=current.lease_id,
            expected_fence=current.fence,
            reason="crash-window",
        )
    record_path = store.workspace_path_for(workspace)
    index_path = store.task_index_path_for(
        canonical_task_cid=current.canonical_task_cid,
        task_id=current.task_id,
        attempt=current.attempt,
    )
    before_record = record_path.read_bytes()
    before_index = index_path.read_bytes()

    with pytest.raises(WorktreeLifecycleError):
        store.adopt_dead_owner(
            workspace,
            expected_record_id=(
                current.record_id + "-stale"
                if stale_field == "record_id"
                else current.record_id
            ),
            expected_fence=(
                current.fence + 1
                if stale_field == "fence"
                else current.fence
            ),
            expected_lease_id=(
                current.lease_id + "-stale"
                if stale_field == "lease"
                else current.lease_id
            ),
            expected_task_id=current.task_id,
            expected_canonical_task_cid=current.canonical_task_cid,
            expected_attempt=current.attempt,
            expected_branch=current.branch,
            expected_merge_target=current.merge_target,
            expected_repo_root=current.repo_root,
            expected_state_dir=current.state_dir,
            lane_id="reconciliation",
        )

    assert record_path.read_bytes() == before_record
    assert index_path.read_bytes() == before_index


def test_dead_owner_adoption_rechecks_authority_under_locked_cas(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _store(tmp_path, startup_grace_seconds=0.0)
    workspace = tmp_path / "orphan"
    record = store.begin_preparing(
        task_id="ORPHAN",
        canonical_task_cid="cid:orphan",
        attempt=2,
        lane_id="dead",
        workspace_path=workspace,
        branch="implementation/orphan",
        merge_target="main",
        state_dir=str(tmp_path / "state"),
        owner=ProcessBirthIdentity(
            pid=2**30 - 7,
            start_time_ticks=1,
            boot_id="dead-boot",
        ),
    )
    active = store.mark_active(
        workspace,
        lease_id=record.lease_id,
        expected_fence=record.fence,
    )
    original_precheck = store.require_exact_dead_owner
    raced: list = []

    def race_after_precheck(*args, **kwargs):
        checked = original_precheck(*args, **kwargs)
        raced.append(
            store.mark_settling(
                workspace,
                lease_id=checked.lease_id,
                expected_fence=checked.fence,
            )
        )
        return checked

    monkeypatch.setattr(
        store,
        "require_exact_dead_owner",
        race_after_precheck,
    )

    with pytest.raises(FenceMismatchError):
        store.adopt_dead_owner(
            workspace,
            expected_record_id=active.record_id,
            expected_fence=active.fence,
            expected_lease_id=active.lease_id,
            expected_task_id=active.task_id,
            expected_canonical_task_cid=active.canonical_task_cid,
            expected_attempt=active.attempt,
            expected_branch=active.branch,
            expected_merge_target=active.merge_target,
            expected_repo_root=active.repo_root,
            expected_state_dir=active.state_dir,
            lane_id="reconciliation",
        )

    assert len(raced) == 1
    assert store.load_workspace(workspace) == raced[0]


def test_dead_owner_adoption_pins_owner_identity_under_locked_cas(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _store(tmp_path, startup_grace_seconds=0.0)
    workspace = tmp_path / "orphan"
    record = store.begin_preparing(
        task_id="ORPHAN",
        canonical_task_cid="cid:orphan",
        attempt=2,
        lane_id="dead",
        workspace_path=workspace,
        branch="implementation/orphan",
        merge_target="main",
        state_dir=str(tmp_path / "state"),
        owner=ProcessBirthIdentity(
            pid=2**30 - 7,
            start_time_ticks=1,
            boot_id="dead-boot",
        ),
    )
    active = store.mark_active(
        workspace,
        lease_id=record.lease_id,
        expected_fence=record.fence,
    )
    record_path = store.workspace_path_for(workspace)
    original_precheck = store.require_exact_dead_owner
    raced_bytes: list[bytes] = []

    def replace_owner_after_precheck(*args, **kwargs):
        checked = original_precheck(*args, **kwargs)
        payload = json.loads(record_path.read_text(encoding="utf-8"))
        payload["owner"] = ProcessBirthIdentity(
            pid=2**30 - 9,
            start_time_ticks=2,
            boot_id="other-dead-boot",
        ).to_dict()
        record_path.write_text(json.dumps(payload), encoding="utf-8")
        raced_bytes.append(record_path.read_bytes())
        return checked

    monkeypatch.setattr(
        store,
        "require_exact_dead_owner",
        replace_owner_after_precheck,
    )

    with pytest.raises(OwnershipError):
        store.adopt_dead_owner(
            workspace,
            expected_record_id=active.record_id,
            expected_fence=active.fence,
            expected_lease_id=active.lease_id,
            expected_task_id=active.task_id,
            expected_canonical_task_cid=active.canonical_task_cid,
            expected_attempt=active.attempt,
            expected_branch=active.branch,
            expected_merge_target=active.merge_target,
            expected_repo_root=active.repo_root,
            expected_state_dir=active.state_dir,
            lane_id="reconciliation",
        )

    assert record_path.read_bytes() == raced_bytes[0]


def test_branch_fallback_reclaims_authoritative_provisional_workspace(
    tmp_path: Path,
) -> None:
    clock = FakeClock(1_000.0)
    store = _store(
        tmp_path,
        lease_seconds=10.0,
        startup_grace_seconds=0.0,
        clock=clock,
    )
    provisional = tmp_path / "provisional-attempt-path"
    pooled = tmp_path / "stable-pool-path"
    branch = "implementation/provisional-branch"
    record = store.begin_preparing(
        task_id="STALE-PROVISIONAL",
        attempt=1,
        lane_id="dead-owner",
        workspace_path=provisional,
        branch=branch,
        merge_target="main",
        owner=ProcessBirthIdentity(
            pid=2**30 - 7,
            start_time_ticks=1,
            boot_id="dead-owner",
        ),
    )
    record_bytes = store.workspace_path_for(provisional).read_bytes()
    clock.advance(11.0)

    preflight = store.evaluate_cleanup(
        workspace_path=pooled,
        branch=branch,
        caller_lease_id="reclaimer",
    )
    assert preflight.allowed
    assert preflight.disposition is CleanupDisposition.RECLAIM_THEN_ALLOW
    assert preflight.record == record
    assert store.workspace_path_for(provisional).read_bytes() == record_bytes

    decision = store.authorize_cleanup(
        workspace_path=pooled,
        branch=branch,
        caller_lease_id="reclaimer",
    )
    reclaimed = store.load_workspace(provisional)

    assert decision.allowed
    assert decision.reason == "reclaimed_stale_record"
    assert decision.record == reclaimed
    assert reclaimed is not None
    assert reclaimed.state is WorkspaceLifecycleState.TERMINAL
    assert reclaimed.fence == record.fence + 1
    assert store.load_workspace(pooled) is None


def test_unresolved_stale_reclaim_race_fails_closed(tmp_path: Path) -> None:
    clock = FakeClock(1_000.0)
    store = _store(
        tmp_path,
        lease_seconds=10.0,
        startup_grace_seconds=0.0,
        clock=clock,
    )
    workspace = tmp_path / "stale-reclaim-race"
    record = store.begin_preparing(
        task_id="STALE-RACE",
        attempt=1,
        lane_id="dead-owner",
        workspace_path=workspace,
        branch="implementation/stale-reclaim-race",
        merge_target="main",
        owner=ProcessBirthIdentity(
            pid=2**30 - 7,
            start_time_ticks=1,
            boot_id="dead-owner",
        ),
    )
    clock.advance(11.0)
    store.reclaim_stale = lambda *_args, **_kwargs: None  # type: ignore[method-assign]

    decision = store.authorize_cleanup(
        workspace_path=workspace,
        caller_lease_id="reclaimer",
    )

    assert not decision.allowed
    assert decision.disposition is CleanupDisposition.DENY
    assert decision.reason == "stale_reclaim_race_unresolved"
    assert decision.failure_kind is LifecycleFailureKind.LIFECYCLE_RACE
    assert decision.attempt_consumed is False
    assert store.load_workspace(workspace) == record


def test_missing_proc_fails_closed(tmp_path: Path) -> None:
    missing_proc = tmp_path / "no-proc"
    store = _store(tmp_path, proc_root=missing_proc)
    workspace = tmp_path / "ws-missing-proc"
    record = store.begin_preparing(
        task_id="P",
        attempt=1,
        lane_id="lane",
        workspace_path=workspace,
        branch="implementation/p",
        merge_target="main",
        owner=ProcessBirthIdentity(pid=1, start_time_ticks=1, boot_id="x"),
    )
    decision = store.evaluate_cleanup(
        workspace_path=workspace,
        caller_lease_id="other",
    )
    assert not decision.allowed
    assert decision.reason == "process_inspection_unavailable"
    assert decision.record is not None
    assert decision.record.lease_id == record.lease_id


def test_pid_reuse_treated_as_dead_owner(tmp_path: Path) -> None:
    if not proc_available():
        pytest.skip("/proc required for PID reuse observation")
    store = _store(tmp_path, lease_seconds=1.0, startup_grace_seconds=0.0)
    clock = store.clock
    assert isinstance(clock, FakeClock)
    workspace = tmp_path / "pid-reuse"
    # Claim a live PID with a wrong start-time so liveness detects reuse.
    live = current_process_birth()
    reused = ProcessBirthIdentity(
        pid=live.pid,
        start_time_ticks=max(1, live.start_time_ticks - 1),
        boot_id=live.boot_id,
    )
    record = store.begin_preparing(
        task_id="REUSE",
        attempt=1,
        lane_id="lane",
        workspace_path=workspace,
        branch="implementation/reuse",
        merge_target="main",
        owner=reused,
    )
    assert owner_liveness(reused) is OwnerLiveness.DEAD
    clock.advance(2.0)
    decision = store.authorize_cleanup(workspace_path=workspace)
    assert decision.allowed
    assert decision.record is not None
    assert decision.record.fence > record.fence


def test_duplicate_attempt_rejected_while_owner_alive(tmp_path: Path) -> None:
    store = _store(tmp_path)
    workspace = tmp_path / "dup"
    store.begin_preparing(
        task_id="DUP",
        canonical_task_cid="cid:dup",
        attempt=3,
        lane_id="lane-a",
        workspace_path=workspace,
        branch="implementation/dup",
        merge_target="main",
    )
    with pytest.raises(DuplicateAttemptError):
        store.begin_preparing(
            task_id="DUP",
            canonical_task_cid="cid:dup",
            attempt=3,
            lane_id="lane-b",
            workspace_path=workspace,
            branch="implementation/dup",
            merge_target="main",
        )


def test_duplicate_attempts_do_not_leak_candidate_workspace_guards(
    tmp_path: Path,
) -> None:
    clock = FakeClock(1_000.0)
    store = _store(
        tmp_path,
        lease_seconds=60.0,
        startup_grace_seconds=0.0,
        clock=clock,
    )
    original_workspace = tmp_path / "worktrees" / "original"
    dead_owner = ProcessBirthIdentity(
        pid=2**30 - 9,
        start_time_ticks=1,
        boot_id="dead-boot",
    )
    original = store.begin_preparing(
        task_id="DUP-GUARD",
        canonical_task_cid="cid:dup-guard",
        attempt=1,
        lane_id="lane-a",
        workspace_path=original_workspace,
        branch="implementation/dup-guard",
        merge_target="main",
        owner=dead_owner,
    )
    assert store.store_dir is not None
    initial_guards = {
        path.name
        for path in store.store_dir.iterdir()
        if path.name.endswith(".update.lock")
    }

    for index in range(20):
        candidate = tmp_path / "worktrees" / f"retry-{index}"
        with pytest.raises(
            DuplicateAttemptError,
            match="task/attempt claim lease has not expired",
        ):
            store.begin_preparing(
                task_id="DUP-GUARD",
                canonical_task_cid="cid:dup-guard",
                attempt=1,
                lane_id="lane-b",
                workspace_path=candidate,
                branch=f"implementation/dup-guard-retry-{index}",
                merge_target="main",
            )
        assert store.load_workspace(candidate) is None

    final_guards = {
        path.name
        for path in store.store_dir.iterdir()
        if path.name.endswith(".update.lock")
    }
    assert final_guards == initial_guards
    assert store.load_workspace(original_workspace) == original
    assert len(list(store.store_dir.glob(".task-*.json.update.lock"))) == 1


def test_compare_and_delete_requires_matching_fence(tmp_path: Path) -> None:
    store = _store(tmp_path)
    workspace = tmp_path / "cad"
    record = store.begin_preparing(
        task_id="CAD",
        attempt=1,
        lane_id="lane",
        workspace_path=workspace,
        branch="implementation/cad",
        merge_target="main",
    )
    assert store.compare_and_delete(workspace, expected_fence=record.fence + 99) is False
    assert store.load_workspace(workspace) is not None
    terminal = store.mark_terminal(
        workspace,
        lease_id=record.lease_id,
        expected_fence=record.fence,
        reason="done",
    )
    assert store.compare_and_delete(
        workspace,
        expected_fence=terminal.fence,
        lease_id=terminal.lease_id,
    )
    assert store.load_workspace(workspace) is None


def test_lifecycle_race_result_consumes_no_retry_or_provider() -> None:
    payload = lifecycle_race_result(
        reason="worktree_lifecycle_claim_exists",
        task_id="ASI-171",
        attempt=4,
    )
    assert payload["skipped"] is True
    assert payload["attempt_consumed"] is False
    assert payload["provider_call_allowed"] is False
    assert payload["failure_kind"] == LifecycleFailureKind.LIFECYCLE_RACE.value
    assert payload["requirement_id"] == FENCED_WORKTREE_LIFECYCLE_REQUIREMENT_ID


def test_owner_may_cleanup_nonterminal_claim(tmp_path: Path) -> None:
    store = _store(tmp_path)
    workspace = tmp_path / "owner-clean"
    record = store.begin_preparing(
        task_id="OWN",
        attempt=1,
        lane_id="lane",
        workspace_path=workspace,
        branch="implementation/own",
        merge_target="main",
    )
    decision = store.authorize_cleanup(
        workspace_path=workspace,
        caller_lease_id=record.lease_id,
    )
    assert decision.allowed
    assert decision.reason == "caller_is_record_owner"


def _barrier_worker(
    store_dir: str,
    repo_root: str,
    workspace: str,
    branch: str,
    ready: mp.synchronize.Event,
    go: mp.synchronize.Event,
    results: mp.queues.Queue,
    role: str,
) -> None:
    store = WorktreeLifecycleStore(
        repo_root=Path(repo_root),
        lease_seconds=30.0,
        startup_grace_seconds=5.0,
        store_dir=Path(store_dir),
    )
    if role == "owner":
        record = store.begin_preparing(
            task_id="BARRIER",
            canonical_task_cid="cid:barrier",
            attempt=1,
            lane_id="owner",
            workspace_path=workspace,
            branch=branch,
            merge_target="main",
        )
        ready.set()
        go.wait(timeout=5.0)
        active = store.mark_active(
            workspace,
            lease_id=record.lease_id,
            expected_fence=record.fence,
        )
        results.put(
            {
                "role": "owner",
                "state": active.state.value,
                "fence": active.fence,
                "workspace_exists": Path(workspace).exists(),
            }
        )
        return
    # Peer cleaner: wait until owner has published preparing, then try cleanup.
    ready.wait(timeout=5.0)
    decision = store.authorize_cleanup(
        workspace_path=workspace,
        branch=branch,
        caller_lease_id=f"peer-{os.getpid()}",
    )
    results.put(
        {
            "role": "peer",
            "allowed": decision.allowed,
            "reason": decision.reason,
            "attempt_consumed": decision.attempt_consumed,
            "provider_call_allowed": decision.provider_call_allowed,
        }
    )
    go.set()


def test_multiprocess_peer_cleanup_during_preparing(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    store_dir = tmp_path / "lifecycle"
    workspace = tmp_path / "worktrees" / "barrier"
    workspace.mkdir(parents=True)
    branch = "implementation/barrier-attempt-1"
    ctx = mp.get_context("spawn")
    ready = ctx.Event()
    go = ctx.Event()
    results: mp.Queue = ctx.Queue()
    owner = ctx.Process(
        target=_barrier_worker,
        args=(
            str(store_dir),
            str(repo),
            str(workspace),
            branch,
            ready,
            go,
            results,
            "owner",
        ),
    )
    peer = ctx.Process(
        target=_barrier_worker,
        args=(
            str(store_dir),
            str(repo),
            str(workspace),
            branch,
            ready,
            go,
            results,
            "peer",
        ),
    )
    owner.start()
    peer.start()
    owner.join(timeout=10.0)
    peer.join(timeout=10.0)
    assert owner.exitcode == 0
    assert peer.exitcode == 0
    payloads = [results.get(timeout=1.0), results.get(timeout=1.0)]
    by_role = {item["role"]: item for item in payloads}
    assert by_role["peer"]["allowed"] is False
    assert by_role["peer"]["attempt_consumed"] is False
    assert by_role["peer"]["provider_call_allowed"] is False
    assert by_role["owner"]["state"] == WorkspaceLifecycleState.ACTIVE.value
    assert workspace.exists()


def _simultaneous_claim_worker(
    store_dir: str,
    repo_root: str,
    workspace: str,
    ready: mp.synchronize.Event,
    go: mp.synchronize.Event,
    results: mp.queues.Queue,
    lane: str,
) -> None:
    store = WorktreeLifecycleStore(
        repo_root=Path(repo_root),
        lease_seconds=30.0,
        store_dir=Path(store_dir),
    )
    ready.set()
    go.wait(timeout=5.0)
    try:
        record = store.begin_preparing(
            task_id="SIM",
            canonical_task_cid="cid:sim",
            attempt=1,
            lane_id=lane,
            workspace_path=workspace,
            branch="implementation/sim",
            merge_target="main",
        )
        results.put({"lane": lane, "ok": True, "fence": record.fence, "lease": record.lease_id})
    except DuplicateAttemptError as exc:
        results.put({"lane": lane, "ok": False, "error": str(exc)})


def test_simultaneous_lane_startup_exactly_one_owner(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    store_dir = tmp_path / "lifecycle"
    workspace = str(tmp_path / "ws-sim")
    ctx = mp.get_context("spawn")
    ready_a = ctx.Event()
    ready_b = ctx.Event()
    go = ctx.Event()
    results: mp.Queue = ctx.Queue()
    workers = [
        ctx.Process(
            target=_simultaneous_claim_worker,
            args=(str(store_dir), str(repo), workspace, ready_a, go, results, "a"),
        ),
        ctx.Process(
            target=_simultaneous_claim_worker,
            args=(str(store_dir), str(repo), workspace, ready_b, go, results, "b"),
        ),
    ]
    for worker in workers:
        worker.start()
    ready_a.wait(timeout=5.0)
    ready_b.wait(timeout=5.0)
    go.set()
    for worker in workers:
        worker.join(timeout=10.0)
        assert worker.exitcode == 0
    payloads = [results.get(timeout=1.0), results.get(timeout=1.0)]
    winners = [item for item in payloads if item.get("ok")]
    losers = [item for item in payloads if not item.get("ok")]
    assert len(winners) == 1
    assert len(losers) == 1


def test_settling_and_active_also_block_peer_cleanup(tmp_path: Path) -> None:
    store = _store(tmp_path)
    workspace = tmp_path / "settle"
    record = store.begin_preparing(
        task_id="S",
        attempt=1,
        lane_id="lane",
        workspace_path=workspace,
        branch="implementation/s",
        merge_target="main",
    )
    active = store.mark_active(
        workspace, lease_id=record.lease_id, expected_fence=record.fence
    )
    deny_active = store.evaluate_cleanup(
        workspace_path=workspace, caller_lease_id="peer"
    )
    assert not deny_active.allowed
    assert "active" in deny_active.reason
    settling = store.mark_settling(
        workspace, lease_id=active.lease_id, expected_fence=active.fence
    )
    deny_settling = store.evaluate_cleanup(
        workspace_path=workspace, caller_lease_id="peer"
    )
    assert not deny_settling.allowed
    assert "settling" in deny_settling.reason
    assert settling.state is WorkspaceLifecycleState.SETTLING


def test_legitimate_terminal_cleanup_without_record(tmp_path: Path) -> None:
    store = _store(tmp_path)
    decision = store.authorize_cleanup(
        workspace_path=tmp_path / "orphan-merged",
        branch="implementation/orphan",
    )
    assert decision.allowed
    assert decision.reason == "no_lifecycle_record"
    assert decision.disposition is CleanupDisposition.ALLOW


def test_default_lease_is_production_scale() -> None:
    assert DEFAULT_LEASE_SECONDS >= 3600.0


def test_read_process_birth_round_trip() -> None:
    if not proc_available():
        pytest.skip("/proc unavailable")
    identity = current_process_birth()
    again = read_process_birth(identity.pid)
    assert again is not None
    assert again.pid == identity.pid
    assert again.start_time_ticks == identity.start_time_ticks
    assert owner_liveness(identity) is OwnerLiveness.ALIVE


def test_partial_worktree_creation_still_fenced(tmp_path: Path) -> None:
    store = _store(tmp_path)
    workspace = tmp_path / "partial"
    # Partial: record exists, directory half-created.
    workspace.mkdir(parents=True)
    (workspace / ".git").write_text("gitdir: /tmp/incomplete\n", encoding="utf-8")
    store.begin_preparing(
        task_id="PARTIAL",
        attempt=1,
        lane_id="lane",
        workspace_path=workspace,
        branch="implementation/partial",
        merge_target="main",
    )
    decision = store.authorize_cleanup(workspace_path=workspace, caller_lease_id="peer")
    assert not decision.allowed
    assert decision.failure_kind is LifecycleFailureKind.LIFECYCLE_RACE


def test_find_by_branch_prefers_nonterminal(tmp_path: Path) -> None:
    store = _store(tmp_path)
    ws = tmp_path / "by-branch"
    record = store.begin_preparing(
        task_id="B",
        attempt=1,
        lane_id="lane",
        workspace_path=ws,
        branch="implementation/by-branch",
        merge_target="main",
    )
    matches = store.find_by_branch("implementation/by-branch")
    assert len(matches) == 1
    assert matches[0].lease_id == record.lease_id
    decision = store.evaluate_cleanup(branch="implementation/by-branch")
    assert not decision.allowed


def test_rebind_workspace_moves_claim_without_duplicate(tmp_path: Path) -> None:
    store = _store(tmp_path)
    provisional = tmp_path / "provisional"
    pooled = tmp_path / "pooled"
    record = store.begin_preparing(
        task_id="REBIND",
        canonical_task_cid="cid:rebind",
        attempt=1,
        lane_id="lane",
        workspace_path=provisional,
        branch="implementation/rebind",
        merge_target="main",
    )
    rebound = store.rebind_workspace(
        provisional,
        pooled,
        lease_id=record.lease_id,
        expected_fence=record.fence,
    )
    assert rebound.workspace_path == str(pooled.resolve(strict=False)) or (
        rebound.workspace_path.endswith("pooled")
    )
    assert rebound.fence == record.fence + 1
    assert store.load_workspace(provisional) is None
    assert store.load_workspace(pooled) is not None
    # Peer still cannot clean the rebound preparing claim.
    decision = store.authorize_cleanup(
        workspace_path=pooled,
        caller_lease_id="peer",
    )
    assert not decision.allowed


def test_record_round_trip_json(tmp_path: Path) -> None:
    store = _store(tmp_path)
    workspace = tmp_path / "json"
    record = store.begin_preparing(
        task_id="JSON",
        canonical_task_cid="cid:json",
        attempt=9,
        lane_id="lane-9",
        workspace_path=workspace,
        branch="implementation/json",
        merge_target="main",
    )
    path = store.workspace_path_for(workspace)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema"].endswith("worktree-lifecycle-record@1")
    assert payload["state"] == "preparing"
    assert payload["attempt"] == 9
    assert payload["owner"]["pid"] == record.owner.pid

def _learning_binding(**overrides: object) -> LearningCheckpointBinding:
    payload = {
        "architecture_id": "arch:v1",
        "weights_id": "weights:0",
        "optimizer_id": "opt:adam",
        "scheduler_id": "sched:cosine",
        "tokenizer_id": "tok:v1",
        "vocab_id": "vocab:v1",
        "cursor_id": "cursor:0",
        "corpus_id": "corpus:v1",
        "split_id": "split:v1",
        "curriculum_id": "curr:v1",
        "loss_id": "loss:ce",
        "random_id": "rng:0",
        "env_id": "env:v1",
        "code_id": "code:v1",
        "compiler_id": "compiler:v1",
        "cursor_step": 0,
    }
    payload.update(overrides)
    return LearningCheckpointBinding.from_dict(payload)


def test_worktree_fence_protects_learning_checkpoint_write(tmp_path: Path) -> None:
    store = _store(tmp_path)
    workspace = tmp_path / "ws-learning"
    record = store.begin_preparing(
        task_id="PGIR-062",
        attempt=1,
        lane_id="lane",
        workspace_path=workspace,
        branch="implementation/learning",
        merge_target="main",
    )
    leases = CampaignLeaseCoordinator(tmp_path / "leases", clock=store.clock)
    adapter = LearningCheckpointAdapter(tmp_path / "recovery", leases=leases)
    checkpoint_lease = leases.acquire(L3ResourceKind.CHECKPOINT, owner_id=record.lease_id)
    cursor = EventCursor.initial("stream:learning", snapshot_id="tree:merged")
    saved = adapter.save(
        _learning_binding(),
        repository_id="repository:current",
        tree_id="tree:merged",
        generation=1,
        cursor=cursor,
        fence=checkpoint_lease.fence,
        lease=checkpoint_lease,
    )
    assert saved.fencing_epoch == checkpoint_lease.fence
    heartbeated = leases.heartbeat(checkpoint_lease, expected_fence=checkpoint_lease.fence)
    with pytest.raises(StaleFenceError):
        adapter.save(
            _learning_binding(
                weights_id="weights:1",
                cursor_id="cursor:1",
                random_id="rng:1",
                cursor_step=1,
            ),
            repository_id="repository:current",
            tree_id="tree:merged",
            generation=2,
            cursor=cursor,
            fence=checkpoint_lease.fence,
            lease=checkpoint_lease,
        )
    advanced = adapter.save(
        _learning_binding(
            weights_id="weights:1",
            cursor_id="cursor:1",
            random_id="rng:1",
            cursor_step=1,
        ),
        repository_id="repository:current",
        tree_id="tree:merged",
        generation=2,
        cursor=cursor,
        fence=heartbeated.fence,
        lease=heartbeated,
    )
    assert advanced.fencing_epoch == heartbeated.fence


def test_stale_worktree_reclaim_does_not_accept_old_checkpoint_fence(
    tmp_path: Path,
) -> None:
    clock = FakeClock(1_000.0)
    store = _store(tmp_path, lease_seconds=10.0, startup_grace_seconds=0.0, clock=clock)
    workspace = tmp_path / "stale-learning"
    record = store.begin_preparing(
        task_id="STALE-LEARN",
        attempt=1,
        lane_id="dead-owner",
        workspace_path=workspace,
        branch="implementation/stale-learn",
        merge_target="main",
        owner=ProcessBirthIdentity(
            pid=2**30 - 7,
            start_time_ticks=1,
            boot_id="dead-owner",
        ),
    )
    adapter = LearningCheckpointAdapter(tmp_path / "recovery")
    cursor = EventCursor.initial("stream:learning", snapshot_id="tree:merged")
    adapter.save(
        _learning_binding(),
        repository_id="repository:current",
        tree_id="tree:merged",
        generation=1,
        cursor=cursor,
        fence=record.fence,
    )
    clock.advance(11.0)
    decision = store.authorize_cleanup(
        workspace_path=workspace,
        caller_lease_id="reclaimer",
    )
    assert decision.allowed
    assert decision.record is not None
    assert decision.record.fence == record.fence + 1
    with pytest.raises(StaleFenceError):
        adapter.save(
            _learning_binding(
                weights_id="weights:1",
                cursor_id="cursor:1",
                random_id="rng:1",
                cursor_step=1,
            ),
            repository_id="repository:current",
            tree_id="tree:merged",
            generation=2,
            cursor=cursor,
            fence=record.fence,
        )
