"""Deterministic multi-process tests for fenced worktree lifecycle (ASI-171)."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.worktree_lifecycle import (
    DEFAULT_LEASE_SECONDS,
    CleanupDisposition,
    DuplicateAttemptError,
    FENCED_WORKTREE_LIFECYCLE_REQUIREMENT_ID,
    FenceMismatchError,
    LifecycleFailureKind,
    OwnerLiveness,
    ProcessBirthIdentity,
    WorkspaceLifecycleState,
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
