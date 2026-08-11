"""Tests for DatabaseWorktreeRegistry (DQP-016).

Evidence subset: canonical containment, symlinks, detached head, nested
gitlinks, rename/delete/untracked policy, dead owner, stale path,
reconciliation.

Acceptance: Worktree reuse/cleanup requires matching lease and current Git
observations; DB history is semantic authority while Git remains byte
authority; stale/dead owner recovery uses CAS/fence; no worktree-local JSON
index can override registry state.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.merge.database_worktree_registry import (
    DATABASE_WORKTREE_REGISTRY_INTERFACE,
    DIRTY_OVERLAY_INTERFACE,
    REPOSITORY_FOREST_INTERFACE,
    WORKTREE_IDENTITY_INTERFACE,
    WORKTREE_SNAPSHOT_INTERFACE,
    DatabaseWorktreeRegistry,
    GitObservation,
    OwnerLiveness,
    PathKind,
    PathPolicyDisposition,
    ProcessBirthIdentity,
    ReuseDisposition,
    WorktreeLifecycleState,
    WorktreeRegistryAuthorityError,
    WorktreeRegistryConflictError,
    WorktreeRegistryContainmentError,
    WorktreeRegistryError,
    WorktreeRegistryIdentityError,
    WorktreeStatus,
    digest_overlay_entries,
    duckdb_available,
    normalize_workspace_path,
    open_worktree_registry,
    path_contained_in_root,
    process_birth_id,
)

pytestmark = pytest.mark.skipif(
    not duckdb_available(),
    reason="DuckDB is required for DatabaseWorktreeRegistry hermetic tests",
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


class GitObservationMap:
    def __init__(self) -> None:
        self._by_path: dict[str, GitObservation] = {}

    def set(self, observation: GitObservation) -> None:
        self._by_path[observation.workspace_path] = observation

    def __call__(self, workspace_path: str) -> GitObservation | None:
        key = normalize_workspace_path(workspace_path)
        return self._by_path.get(key)


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


def _obs(
    workspace: str | Path,
    *,
    head_commit: str = "c0ffee",
    head_tree: str = "t0ffee",
    index_digest: str = "sha256:index1",
    dirty_overlay_digest: str = "sha256:dirty1",
    branch_name: str = "implementation/task",
    is_detached: bool = False,
    path_exists: bool = True,
    is_symlink_root: bool = False,
    git_common_dir: str = "/repo/.git",
    observed_at_ms: int = 1_000_000,
) -> GitObservation:
    return GitObservation(
        workspace_path=str(workspace),
        head_commit=head_commit,
        head_tree=head_tree,
        index_digest=index_digest,
        dirty_overlay_digest=dirty_overlay_digest,
        branch_name=branch_name,
        is_detached=is_detached,
        git_common_dir=git_common_dir,
        path_exists=path_exists,
        is_symlink_root=is_symlink_root,
        observed_at_ms=observed_at_ms,
    )


def _open(
    tmp_path: Path,
    *,
    clock: FakeClock | None = None,
    liveness: LivenessMap | None = None,
    git_observer: GitObservationMap | None = None,
    lease_ttl_ms: int = 60_000,
) -> tuple[DatabaseWorktreeRegistry, FakeClock, LivenessMap, GitObservationMap]:
    clock = clock or FakeClock()
    liveness = liveness or LivenessMap()
    git_observer = git_observer or GitObservationMap()
    registry = open_worktree_registry(
        tmp_path / "worktree_registry.duckdb",
        clock_ms=clock,
        liveness=liveness,
        git_observer=git_observer,
        default_lease_ttl_ms=lease_ttl_ms,
    )
    return registry, clock, liveness, git_observer


def _register_repo(
    registry: DatabaseWorktreeRegistry,
    *,
    common: str = "/repo/.git",
    root: str = "/repo",
    head_commit: str = "c0ffee",
    head_tree: str = "t0ffee",
):
    return registry.register_repository(
        git_common_dir=common,
        canonical_root=root,
        head_commit=head_commit,
        head_tree=head_tree,
    )


# ---------------------------------------------------------------------------
# Interface identities
# ---------------------------------------------------------------------------


def test_interface_identities() -> None:
    assert DATABASE_WORKTREE_REGISTRY_INTERFACE == "DatabaseWorktreeRegistry@1"
    assert REPOSITORY_FOREST_INTERFACE == "RepositoryForest@1"
    assert WORKTREE_IDENTITY_INTERFACE == "WorktreeIdentity@1"
    assert WORKTREE_SNAPSHOT_INTERFACE == "WorktreeSnapshot@1"
    assert DIRTY_OVERLAY_INTERFACE == "DirtyOverlay@1"
    assert DatabaseWorktreeRegistry.INTERFACE == DATABASE_WORKTREE_REGISTRY_INTERFACE


def test_authority_policy_splits_semantic_and_byte(tmp_path: Path) -> None:
    registry, _clock, _live, _git = _open(tmp_path)
    try:
        policy = registry.authority_policy()
        assert policy["semantic_authority"] == "database"
        assert policy["byte_authority"] == "git"
        assert policy["local_json_authority"] == "none"
    finally:
        registry.close()


# ---------------------------------------------------------------------------
# Repositories, branches, submodules
# ---------------------------------------------------------------------------


def test_register_repository_branch_and_submodule_edge(tmp_path: Path) -> None:
    registry, _clock, _live, _git = _open(tmp_path)
    try:
        parent = _register_repo(registry)
        child = registry.register_repository(
            git_common_dir="/repo/vendor/lib/.git",
            canonical_root="/repo/vendor/lib",
            head_commit="deadbeef",
            head_tree="treebeef",
        )
        branch = registry.register_branch(
            repository_id=parent.repository_id,
            branch_name="main",
            tip_commit="c0ffee",
        )
        assert branch.tip_commit == "c0ffee"
        edge = registry.register_submodule_edge(
            parent_repository_id=parent.repository_id,
            child_repository_id=child.repository_id,
            gitlink_path="vendor/lib",
            gitlink_commit="deadbeef",
        )
        assert edge.gitlink_path == "vendor/lib"
        assert len(registry.list_branches(parent.repository_id)) == 1
        edges = registry.list_submodule_edges(parent.repository_id)
        assert len(edges) == 1
        assert edges[0].child_repository_id == child.repository_id
        loaded = registry.get_repository(parent.repository_id)
        assert loaded is not None
        assert loaded.git_common_dir.endswith(".git")
    finally:
        registry.close()


def test_nested_gitlink_path_rejects_escape(tmp_path: Path) -> None:
    registry, _clock, _live, _git = _open(tmp_path)
    try:
        parent = _register_repo(registry)
        child = registry.register_repository(
            git_common_dir="/other/.git",
            canonical_root="/other",
        )
        with pytest.raises(WorktreeRegistryContainmentError, match="escapes"):
            registry.register_submodule_edge(
                parent_repository_id=parent.repository_id,
                child_repository_id=child.repository_id,
                gitlink_path="../escape",
                gitlink_commit="x",
            )
    finally:
        registry.close()


# ---------------------------------------------------------------------------
# Worktree registration, lease, CAS fence
# ---------------------------------------------------------------------------


def test_register_worktree_and_acquire_lease(tmp_path: Path) -> None:
    registry, clock, _live, _git = _open(tmp_path)
    try:
        repo = _register_repo(registry)
        workspace = tmp_path / "worktrees" / "task-a"
        wt = registry.register_worktree(
            repository_id=repo.repository_id,
            workspace_path=workspace,
            branch_name="implementation/task-a",
            lane_id="lane-1",
            task_id="DQP-016",
            attempt=1,
        )
        assert wt.lifecycle_state is WorktreeLifecycleState.PREPARING
        assert wt.fencing_token == 0
        birth = _birth(100, start_time_ticks=1000)
        leased = registry.acquire_lease(
            wt.worktree_id,
            process_birth=birth,
            session_id="session:1",
            ttl_ms=30_000,
        )
        assert leased.is_leased
        assert leased.fencing_token == 1
        assert leased.lifecycle_state is WorktreeLifecycleState.ACTIVE
        assert leased.lease_expires_at_ms == clock.now + 30_000
        assert leased.owner_process_birth_id == process_birth_id(birth)
        loaded = registry.get_worktree(wt.worktree_id)
        assert loaded is not None
        assert loaded.lease_id == leased.lease_id
    finally:
        registry.close()


def test_raw_pid_never_proves_identity_on_lease(tmp_path: Path) -> None:
    registry, _clock, _live, _git = _open(tmp_path)
    try:
        repo = _register_repo(registry)
        wt = registry.register_worktree(
            repository_id=repo.repository_id,
            workspace_path=tmp_path / "wt",
        )
        with pytest.raises(WorktreeRegistryIdentityError, match="start_time_ticks"):
            registry.acquire_lease(
                wt.worktree_id,
                process_birth=ProcessBirthIdentity(
                    pid=99, start_time_ticks=0, boot_id="boot"
                ),
            )
    finally:
        registry.close()


def test_live_owner_lease_is_fenced(tmp_path: Path) -> None:
    registry, _clock, _live, _git = _open(tmp_path)
    try:
        repo = _register_repo(registry)
        wt = registry.register_worktree(
            repository_id=repo.repository_id,
            workspace_path=tmp_path / "wt-fence",
        )
        first = registry.acquire_lease(
            wt.worktree_id, process_birth=_birth(10, start_time_ticks=1000)
        )
        with pytest.raises(WorktreeRegistryConflictError, match="fenced"):
            registry.acquire_lease(
                wt.worktree_id, process_birth=_birth(11, start_time_ticks=1100)
            )
        owner = registry.get_worktree(wt.worktree_id)
        assert owner is not None
        assert owner.lease_id == first.lease_id
    finally:
        registry.close()


def test_dead_owner_reclaim_uses_cas_fence(tmp_path: Path) -> None:
    registry, _clock, liveness, _git = _open(tmp_path)
    try:
        repo = _register_repo(registry)
        wt = registry.register_worktree(
            repository_id=repo.repository_id,
            workspace_path=tmp_path / "wt-reclaim",
            lane_id="lane-r",
        )
        first_birth = _birth(20, start_time_ticks=2000)
        first = registry.acquire_lease(wt.worktree_id, process_birth=first_birth)
        liveness.set(first_birth, OwnerLiveness.DEAD)
        second_birth = _birth(21, start_time_ticks=2100)
        second = registry.reclaim_dead_owner(
            wt.worktree_id,
            expected_fencing_token=first.fencing_token,
            process_birth=second_birth,
        )
        assert second.fencing_token > first.fencing_token
        assert second.lease_id != first.lease_id
        assert second.owner_process_birth_id == process_birth_id(second_birth)

        # Stale fence CAS fails.
        with pytest.raises(WorktreeRegistryConflictError, match="CAS"):
            registry.reclaim_dead_owner(
                wt.worktree_id,
                expected_fencing_token=first.fencing_token,
                process_birth=_birth(22, start_time_ticks=2200),
            )
    finally:
        registry.close()


def test_unknown_owner_cannot_be_reclaimed(tmp_path: Path) -> None:
    registry, _clock, liveness, _git = _open(tmp_path)
    try:
        repo = _register_repo(registry)
        wt = registry.register_worktree(
            repository_id=repo.repository_id,
            workspace_path=tmp_path / "wt-unknown",
        )
        birth = _birth(30, start_time_ticks=3000)
        first = registry.acquire_lease(wt.worktree_id, process_birth=birth)
        liveness.set(birth, OwnerLiveness.UNKNOWN)
        with pytest.raises(WorktreeRegistryIdentityError, match="unknown"):
            registry.reclaim_dead_owner(
                wt.worktree_id,
                expected_fencing_token=first.fencing_token,
                process_birth=_birth(31, start_time_ticks=3100),
            )
    finally:
        registry.close()


def test_release_lease_requires_matching_fence(tmp_path: Path) -> None:
    registry, _clock, _live, _git = _open(tmp_path)
    try:
        repo = _register_repo(registry)
        wt = registry.register_worktree(
            repository_id=repo.repository_id,
            workspace_path=tmp_path / "wt-release",
        )
        birth = _birth(40, start_time_ticks=4000)
        leased = registry.acquire_lease(wt.worktree_id, process_birth=birth)
        with pytest.raises(WorktreeRegistryConflictError, match="fence"):
            registry.release_lease(
                wt.worktree_id,
                lease_id=leased.lease_id,
                fencing_token=leased.fencing_token + 1,
                process_birth=birth,
            )
        released = registry.release_lease(
            wt.worktree_id,
            lease_id=leased.lease_id,
            fencing_token=leased.fencing_token,
            process_birth=birth,
        )
        assert released.lifecycle_state is WorktreeLifecycleState.TERMINAL
        assert released.status is WorktreeStatus.TERMINAL
    finally:
        registry.close()


# ---------------------------------------------------------------------------
# Snapshots, paths, dirty overlays
# ---------------------------------------------------------------------------


def test_record_snapshot_paths_and_dirty_overlay(tmp_path: Path) -> None:
    registry, _clock, _live, _git = _open(tmp_path)
    try:
        repo = _register_repo(registry)
        workspace = tmp_path / "worktrees" / "snap"
        wt = registry.register_worktree(
            repository_id=repo.repository_id,
            workspace_path=workspace,
            branch_name="implementation/snap",
        )
        birth = _birth(50, start_time_ticks=5000)
        leased = registry.acquire_lease(wt.worktree_id, process_birth=birth)
        observation = _obs(workspace)
        snapshot = registry.record_snapshot(
            wt.worktree_id,
            observation=observation,
            base_commit="base001",
            scanner_version="scanner@1",
            lease_id=leased.lease_id,
            fencing_token=leased.fencing_token,
            paths=[
                {
                    "relative_path": "src/main.py",
                    "path_kind": PathKind.FILE.value,
                    "blob_id": "blob1",
                    "policy_disposition": PathPolicyDisposition.TRACKED.value,
                },
                {
                    "relative_path": "vendor/lib",
                    "path_kind": PathKind.GITLINK.value,
                    "blob_id": "deadbeef",
                    "is_gitlink": True,
                },
                {
                    "relative_path": "link-out",
                    "path_kind": PathKind.SYMLINK.value,
                    "symlink_target": "/tmp/outside",
                    "is_symlink": True,
                },
                {
                    "relative_path": "gone.py",
                    "path_kind": PathKind.MISSING.value,
                    "policy_disposition": PathPolicyDisposition.DELETED.value,
                },
            ],
        )
        assert snapshot.head_commit == "c0ffee"
        assert snapshot.base_commit == "base001"
        paths = registry.list_paths(snapshot.snapshot_id)
        assert len(paths) == 4
        by_path = {p.relative_path: p for p in paths}
        assert by_path["vendor/lib"].is_gitlink
        assert by_path["link-out"].is_symlink
        assert by_path["link-out"].symlink_target == "/tmp/outside"
        assert by_path["gone.py"].policy_disposition is PathPolicyDisposition.DELETED

        overlay = registry.record_dirty_overlay(
            wt.worktree_id,
            snapshot_id=snapshot.snapshot_id,
            lease_id=leased.lease_id,
            fencing_token=leased.fencing_token,
            entries=[
                {"kind": "modified", "path": "src/main.py", "blob_id": "blob2"},
                {
                    "kind": "renamed",
                    "path": "src/new.py",
                    "from_path": "src/old.py",
                    "blob_id": "blob3",
                },
                {"kind": "deleted", "path": "gone.py"},
                {"kind": "untracked", "path": "tmp/scratch.bin"},
            ],
            rename_policy="track",
            delete_policy="track",
            untracked_policy="include",
        )
        assert overlay.entry_count == 4
        assert overlay.overlay_digest == digest_overlay_entries(overlay.entries)
        loaded_wt = registry.get_worktree(wt.worktree_id)
        assert loaded_wt is not None
        assert loaded_wt.dirty_overlay_digest == overlay.overlay_digest
        assert loaded_wt.current_snapshot_id == snapshot.snapshot_id
    finally:
        registry.close()


def test_detached_head_snapshot(tmp_path: Path) -> None:
    registry, _clock, _live, _git = _open(tmp_path)
    try:
        repo = _register_repo(registry)
        workspace = tmp_path / "wt-detached"
        wt = registry.register_worktree(
            repository_id=repo.repository_id,
            workspace_path=workspace,
            is_detached=True,
        )
        leased = registry.acquire_lease(
            wt.worktree_id, process_birth=_birth(60, start_time_ticks=6000)
        )
        observation = _obs(
            workspace,
            branch_name="",
            is_detached=True,
            head_commit="detached01",
        )
        snapshot = registry.record_snapshot(
            wt.worktree_id,
            observation=observation,
            lease_id=leased.lease_id,
            fencing_token=leased.fencing_token,
        )
        assert snapshot.is_detached is True
        assert snapshot.head_commit == "detached01"
        registry.register_branch(
            repository_id=repo.repository_id,
            branch_name="HEAD",
            tip_commit="detached01",
            is_detached=True,
        )
        branches = registry.list_branches(repo.repository_id)
        assert any(b.is_detached for b in branches)
    finally:
        registry.close()


def test_canonical_containment_rejects_escaped_paths(tmp_path: Path) -> None:
    with pytest.raises(WorktreeRegistryContainmentError):
        path_contained_in_root("../secrets")
    with pytest.raises(WorktreeRegistryContainmentError):
        path_contained_in_root("/absolute")
    assert path_contained_in_root("src/ok.py") == "src/ok.py"


def test_untracked_policy_ignore_drops_entries(tmp_path: Path) -> None:
    registry, _clock, _live, _git = _open(tmp_path)
    try:
        repo = _register_repo(registry)
        workspace = tmp_path / "wt-policy"
        wt = registry.register_worktree(
            repository_id=repo.repository_id,
            workspace_path=workspace,
        )
        leased = registry.acquire_lease(
            wt.worktree_id, process_birth=_birth(70, start_time_ticks=7000)
        )
        snapshot = registry.record_snapshot(
            wt.worktree_id,
            observation=_obs(workspace),
            lease_id=leased.lease_id,
            fencing_token=leased.fencing_token,
        )
        overlay = registry.record_dirty_overlay(
            wt.worktree_id,
            snapshot_id=snapshot.snapshot_id,
            lease_id=leased.lease_id,
            fencing_token=leased.fencing_token,
            entries=[
                {"kind": "modified", "path": "a.py"},
                {"kind": "untracked", "path": "noise.tmp"},
            ],
            untracked_policy="ignore",
        )
        assert overlay.entry_count == 1
        assert all(e["kind"] != "untracked" for e in overlay.entries)

        with pytest.raises(WorktreeRegistryError, match="rename_policy"):
            registry.record_dirty_overlay(
                wt.worktree_id,
                snapshot_id=snapshot.snapshot_id,
                lease_id=leased.lease_id,
                fencing_token=leased.fencing_token,
                entries=[
                    {
                        "kind": "renamed",
                        "path": "b.py",
                        "from_path": "a.py",
                    }
                ],
                rename_policy="reject",
            )
    finally:
        registry.close()


# ---------------------------------------------------------------------------
# Reuse / cleanup require lease + Git observations
# ---------------------------------------------------------------------------


def test_reuse_requires_matching_lease_and_git_observation(tmp_path: Path) -> None:
    registry, _clock, _live, git_map = _open(tmp_path)
    try:
        repo = _register_repo(registry)
        workspace = tmp_path / "wt-reuse"
        wt = registry.register_worktree(
            repository_id=repo.repository_id,
            workspace_path=workspace,
            branch_name="implementation/task",
        )
        birth = _birth(80, start_time_ticks=8000)
        leased = registry.acquire_lease(wt.worktree_id, process_birth=birth)
        observation = _obs(workspace)
        registry.record_snapshot(
            wt.worktree_id,
            observation=observation,
            lease_id=leased.lease_id,
            fencing_token=leased.fencing_token,
        )
        git_map.set(observation)

        denied_lease = registry.evaluate_reuse(
            workspace_path=workspace,
            lease_id="lease:wrong",
            fencing_token=leased.fencing_token,
            observation=observation,
        )
        assert denied_lease.disposition is ReuseDisposition.DENY
        assert denied_lease.reason == "lease_mismatch"
        assert denied_lease.allowed is False

        denied_obs = registry.evaluate_reuse(
            workspace_path=workspace,
            lease_id=leased.lease_id,
            fencing_token=leased.fencing_token,
            observation=_obs(workspace, head_commit="other"),
        )
        assert denied_obs.disposition is ReuseDisposition.DENY
        assert "commit" in denied_obs.reason or denied_obs.reason == "head_commit_mismatch"

        missing_obs = registry.evaluate_reuse(
            workspace_path=workspace,
            lease_id=leased.lease_id,
            fencing_token=leased.fencing_token,
            observation=None,
        )
        # git_observer has matching observation registered.
        assert missing_obs.allowed is True

        # Clear observer and require explicit observation.
        git_map._by_path.clear()
        require_obs = registry.evaluate_reuse(
            workspace_path=workspace,
            lease_id=leased.lease_id,
            fencing_token=leased.fencing_token,
            observation=None,
        )
        assert require_obs.disposition is ReuseDisposition.DENY
        assert require_obs.reason == "git_observation_required"

        allowed = registry.evaluate_reuse(
            workspace_path=workspace,
            lease_id=leased.lease_id,
            fencing_token=leased.fencing_token,
            observation=observation,
        )
        assert allowed.disposition is ReuseDisposition.ALLOW
        assert allowed.lease_matched is True
        assert allowed.observation_matched is True
    finally:
        registry.close()


def test_cleanup_requires_lease_or_dead_owner_plus_git(tmp_path: Path) -> None:
    registry, _clock, liveness, _git = _open(tmp_path)
    try:
        repo = _register_repo(registry)
        workspace = tmp_path / "wt-cleanup"
        wt = registry.register_worktree(
            repository_id=repo.repository_id,
            workspace_path=workspace,
        )
        birth = _birth(90, start_time_ticks=9000)
        leased = registry.acquire_lease(wt.worktree_id, process_birth=birth)
        observation = _obs(workspace)

        # Live owner + wrong lease → deny.
        denied = registry.evaluate_cleanup(
            workspace_path=workspace,
            lease_id="lease:other",
            fencing_token=1,
            observation=observation,
        )
        assert denied.disposition is ReuseDisposition.DENY
        assert denied.reason == "live_owner_lease_mismatch"

        # Matching lease but still active → deny until terminal/settling.
        active = registry.evaluate_cleanup(
            workspace_path=workspace,
            lease_id=leased.lease_id,
            fencing_token=leased.fencing_token,
            observation=observation,
        )
        assert active.disposition is ReuseDisposition.DENY
        assert active.reason == "active_lease_not_terminal"

        registry.release_lease(
            wt.worktree_id,
            lease_id=leased.lease_id,
            fencing_token=leased.fencing_token,
            process_birth=birth,
        )
        terminal = registry.evaluate_cleanup(
            workspace_path=workspace,
            lease_id=leased.lease_id,
            fencing_token=leased.fencing_token,
            observation=observation,
        )
        assert terminal.disposition is ReuseDisposition.ALLOW

        # Fresh worktree: dead owner allows reclaim-then-cleanup.
        workspace2 = tmp_path / "wt-cleanup-2"
        wt2 = registry.register_worktree(
            repository_id=repo.repository_id,
            workspace_path=workspace2,
        )
        owner = _birth(91, start_time_ticks=9100)
        registry.acquire_lease(wt2.worktree_id, process_birth=owner)
        liveness.set(owner, OwnerLiveness.DEAD)
        reclaim = registry.evaluate_cleanup(
            workspace_path=workspace2,
            observation=_obs(workspace2),
        )
        assert reclaim.disposition is ReuseDisposition.RECLAIM_THEN_ALLOW
        assert reclaim.allowed is True
    finally:
        registry.close()


def test_stale_path_reconciliation(tmp_path: Path) -> None:
    registry, _clock, _live, _git = _open(tmp_path)
    try:
        repo = _register_repo(registry)
        workspace = tmp_path / "wt-stale"
        wt = registry.register_worktree(
            repository_id=repo.repository_id,
            workspace_path=workspace,
        )
        birth = _birth(100, start_time_ticks=10000)
        leased = registry.acquire_lease(wt.worktree_id, process_birth=birth)
        present = _obs(workspace)
        registry.record_snapshot(
            wt.worktree_id,
            observation=present,
            lease_id=leased.lease_id,
            fencing_token=leased.fencing_token,
        )
        missing = _obs(workspace, path_exists=False)
        result = registry.reconcile(
            wt.worktree_id,
            observation=missing,
            lease_id=leased.lease_id,
            fencing_token=leased.fencing_token,
        )
        assert result["status"] == "stale_path"
        loaded = registry.get_worktree(wt.worktree_id)
        assert loaded is not None
        assert loaded.lifecycle_state is WorktreeLifecycleState.SETTLING
        assert loaded.body.get("stale_path") is True

        cleanup = registry.evaluate_cleanup(
            workspace_path=workspace,
            lease_id=leased.lease_id,
            fencing_token=leased.fencing_token,
            observation=missing,
        )
        assert cleanup.disposition is ReuseDisposition.ALLOW
    finally:
        registry.close()


def test_symlink_root_observation_does_not_grant_escape(tmp_path: Path) -> None:
    registry, _clock, _live, _git = _open(tmp_path)
    try:
        repo = _register_repo(registry)
        workspace = tmp_path / "wt-symlink"
        wt = registry.register_worktree(
            repository_id=repo.repository_id,
            workspace_path=workspace,
        )
        leased = registry.acquire_lease(
            wt.worktree_id, process_birth=_birth(110, start_time_ticks=11000)
        )
        observation = _obs(workspace, is_symlink_root=True)
        snapshot = registry.record_snapshot(
            wt.worktree_id,
            observation=observation,
            lease_id=leased.lease_id,
            fencing_token=leased.fencing_token,
            paths=[
                {
                    "relative_path": "alias",
                    "path_kind": "symlink",
                    "symlink_target": "../../outside",
                }
            ],
        )
        paths = registry.list_paths(snapshot.snapshot_id)
        assert paths[0].is_symlink
        # Relative path stays contained even when target points outside.
        assert paths[0].relative_path == "alias"
        with pytest.raises(WorktreeRegistryContainmentError):
            registry.record_snapshot(
                wt.worktree_id,
                observation=observation,
                lease_id=leased.lease_id,
                fencing_token=leased.fencing_token,
                paths=[{"relative_path": "../../escape", "path_kind": "file"}],
            )
    finally:
        registry.close()


# ---------------------------------------------------------------------------
# Local JSON cannot override registry
# ---------------------------------------------------------------------------


def test_local_json_index_cannot_override_registry(tmp_path: Path) -> None:
    registry, _clock, _live, _git = _open(tmp_path)
    try:
        repo = _register_repo(registry)
        workspace = tmp_path / "wt-json"
        wt = registry.register_worktree(
            repository_id=repo.repository_id,
            workspace_path=workspace,
        )
        leased = registry.acquire_lease(
            wt.worktree_id, process_birth=_birth(120, start_time_ticks=12000)
        )
        mirror_path = workspace / ".agent" / "worktree-index.json"
        receipt = registry.mirror_local_json(
            worktree_id=wt.worktree_id,
            mirror_path=mirror_path,
            body={
                "lease_id": "forged-lease",
                "fencing_token": 999,
                "lifecycle_state": "terminal",
            },
        )
        assert receipt["authoritative"] is False

        with pytest.raises(WorktreeRegistryAuthorityError, match="cannot override"):
            registry.apply_local_json_index(
                mirror_path=mirror_path,
                claimed_worktree={
                    "lease_id": "forged-lease",
                    "fencing_token": 999,
                    "lifecycle_state": "terminal",
                },
            )

        # Registry state unchanged by the mirror payload.
        loaded = registry.get_worktree(wt.worktree_id)
        assert loaded is not None
        assert loaded.lease_id == leased.lease_id
        assert loaded.fencing_token == leased.fencing_token
        assert loaded.lifecycle_state is WorktreeLifecycleState.ACTIVE
    finally:
        registry.close()


def test_setup_cache_is_shared_across_lanes(tmp_path: Path) -> None:
    registry, clock, _live, _git = _open(tmp_path)
    try:
        repo = _register_repo(registry)
        entry = registry.put_setup_cache(
            repository_id=repo.repository_id,
            cache_key="setup:venv:py312",
            cache_digest="sha256:setup1",
            payload={"python": "3.12", "packages": ["pytest"]},
            ttl_ms=10_000,
        )
        assert entry.cache_digest == "sha256:setup1"
        loaded = registry.get_setup_cache("setup:venv:py312")
        assert loaded is not None
        assert loaded.payload["python"] == "3.12"
        assert loaded.expires_at_ms == clock.now + 10_000
    finally:
        registry.close()


def test_durable_reopen_preserves_worktree_history(tmp_path: Path) -> None:
    clock = FakeClock()
    liveness = LivenessMap()
    git_map = GitObservationMap()
    db_path = tmp_path / "worktree_registry.duckdb"
    workspace = tmp_path / "wt-durable"
    worktree_id = ""
    lease_id = ""
    fencing_token = 0
    snapshot_id = ""

    registry = open_worktree_registry(
        db_path, clock_ms=clock, liveness=liveness, git_observer=git_map
    )
    try:
        repo = _register_repo(registry)
        wt = registry.register_worktree(
            repository_id=repo.repository_id,
            workspace_path=workspace,
            task_id="DQP-016",
        )
        birth = _birth(130, start_time_ticks=13000)
        leased = registry.acquire_lease(wt.worktree_id, process_birth=birth)
        observation = _obs(workspace)
        snapshot = registry.record_snapshot(
            wt.worktree_id,
            observation=observation,
            lease_id=leased.lease_id,
            fencing_token=leased.fencing_token,
        )
        worktree_id = wt.worktree_id
        lease_id = leased.lease_id
        fencing_token = leased.fencing_token
        snapshot_id = snapshot.snapshot_id
    finally:
        registry.close()

    reopened = open_worktree_registry(
        db_path, clock_ms=clock, liveness=liveness, git_observer=git_map
    )
    try:
        loaded = reopened.get_worktree(worktree_id)
        assert loaded is not None
        assert loaded.lease_id == lease_id
        assert loaded.fencing_token == fencing_token
        assert loaded.current_snapshot_id == snapshot_id
        snap = reopened.get_snapshot(snapshot_id)
        assert snap is not None
        assert snap.head_commit == "c0ffee"
        policy = reopened.authority_policy()
        assert policy["semantic_authority"] == "database"
        assert policy["byte_authority"] == "git"
    finally:
        reopened.close()


def test_same_owner_can_renew_lease(tmp_path: Path) -> None:
    registry, clock, _live, _git = _open(tmp_path)
    try:
        repo = _register_repo(registry)
        wt = registry.register_worktree(
            repository_id=repo.repository_id,
            workspace_path=tmp_path / "wt-renew",
        )
        birth = _birth(140, start_time_ticks=14000)
        first = registry.acquire_lease(
            wt.worktree_id, process_birth=birth, ttl_ms=10_000
        )
        clock.advance(5_000)
        renewed = registry.acquire_lease(
            wt.worktree_id, process_birth=birth, ttl_ms=20_000
        )
        assert renewed.lease_id == first.lease_id
        assert renewed.fencing_token == first.fencing_token
        assert renewed.lease_expires_at_ms == clock.now + 20_000
    finally:
        registry.close()
