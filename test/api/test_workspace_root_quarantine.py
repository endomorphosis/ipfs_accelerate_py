"""Ambiguous existing workspace custody is preserved without inventing a task link."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.merge import workspace_quarantine as q
from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
    WorktreeLifecycleStore,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.worktrees import WorktreePool
from ipfs_accelerate_py.agent_supervisor.merge.quarantine_validation import (
    QuarantineDenied,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    PortalImplementationDaemon, PortalTask,
)


def _resource_claim_daemon(
    repo: Path,
    *,
    lane: str,
) -> PortalImplementationDaemon:
    return PortalImplementationDaemon(
        todo_path=repo / "todo.md",
        state_path=repo / lane / "task_state.json",
        strategy_path=repo / lane / "strategy.json",
        events_path=repo / lane / "events.jsonl",
        repo_root=repo,
        task_header_prefix="## PCTDD-",
        worktree_submodule_paths=("external/ipfs_accelerate",),
    )

def _resource_claim_task(task_id: str, predicted_path: str) -> PortalTask:
    return PortalTask(
        task_id=task_id,
        title=f"Implement {task_id}",
        status="ready",
        completion="manual",
        priority="P1",
        track="runtime",
        outputs=[predicted_path],
        metadata={"predicted files": predicted_path},
    )


def _git(repo: Path, *args: str) -> str:
    import subprocess

    result = subprocess.run(
        ["git", *args], cwd=repo, text=True, capture_output=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def seed(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.name", "test")
    _git(repo, "config", "user.email", "test@example.invalid")
    (repo / "README").write_text("base\n")
    _git(repo, "add", "README")
    _git(repo, "commit", "-m", "base")
    root = tmp_path / "worktrees"
    pool = WorktreePool(repo_root=repo, worktree_root=root)
    lease = pool.acquire(cache_key="old-ambiguous", branch_name="attempt/ambiguous")
    lifecycle = WorktreeLifecycleStore(repo_root=repo)
    record = lifecycle.begin_preparing(
        task_id="unbound-historical-task",
        attempt=1,
        lane_id="old-lane",
        workspace_path=lease.path,
        branch="attempt/ambiguous",
        merge_target="main",
        state_dir=str(tmp_path / "missing-private-state"),
    )
    return repo, root, pool, lease, lifecycle, record


def test_whole_root_retained_and_independent_fresh_pool_allocates(tmp_path):
    repo, root, pool, lease, lifecycle, record = seed(tmp_path)
    before = q.census(repo, root)
    frozen = q.freeze(repo, root, expected=before)
    assert frozen["snapshot"] == before
    # There is deliberately no invented 005 workspace or task association.
    assert set(frozen) == {"schema", "root", "fresh_root", "snapshot", "cid"}
    for mutation in (
        lambda: pool.release(lease),
        lambda: pool.acquire(cache_key="independent"),
        lambda: pool.invalidate(),
        lambda: pool._discard_state(pool._read_state(lease.entry_id)),
        lambda: lifecycle.reclaim_stale(lease.path, now=time.time() + 1_000_000),
        lambda: lifecycle.reclaim_dead_owner_for_controlled_restart(
            lease.path, expected_state_dir=record.state_dir
        ),
        lambda: lifecycle.compare_and_delete(
            lease.path, expected_fence=record.fence, lease_id=record.lease_id
        ),
    ):
        with pytest.raises(QuarantineDenied, match="workspace_root_quarantined"):
            mutation()
        assert q.census(repo, root) == before
    fresh_pool = WorktreePool(repo_root=repo, worktree_root=Path(frozen["fresh_root"]))
    fresh = fresh_pool.acquire(
        cache_key="independent", branch_name="attempt/independent"
    )
    assert fresh.path != lease.path and not q.within(fresh.path, root)
    assert q.verify(repo, root) == frozen
    assert lifecycle.load_workspace(lease.path).to_dict() == record.to_dict()
    fresh_pool.release(fresh)
    assert q.census(repo, root) == before


def test_freeze_waits_for_native_mutation_scope_and_then_denies_late_writer(tmp_path):
    repo, root, _, _, _, _ = seed(tmp_path)
    entered = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    result = []

    def writer():
        with q.mutation(repo, root):
            entered.set()
            assert release.wait(timeout=5)

    worker = threading.Thread(target=writer)
    worker.start()
    assert entered.wait(timeout=5)
    before = q.census(repo, root)

    def freezer():
        result.append(q.freeze(repo, root, expected=before))
        finished.set()

    freezer_thread = threading.Thread(target=freezer)
    freezer_thread.start()
    assert not finished.wait(timeout=0.05)
    release.set()
    worker.join(timeout=5)
    freezer_thread.join(timeout=5)
    assert finished.is_set() and len(result) == 1
    with pytest.raises(QuarantineDenied):
        with q.mutation(repo, root):
            pytest.fail("late mutation entered")
    assert q.census(repo, root) == before


@pytest.mark.parametrize("operation", ["exact_delete", "partial_finalize", "quarantine"])
def test_native_lifecycle_mutations_preserve_frozen_custody(tmp_path, operation):
    repo, root, _, lease, lifecycle, record = seed(tmp_path)
    terminal = lifecycle.mark_terminal(
        lease.path, lease_id=record.lease_id, expected_fence=record.fence,
    )
    before = q.census(repo, root)
    frozen = q.freeze(repo, root, expected=before)
    with pytest.raises(QuarantineDenied, match="workspace_root_quarantined"):
        if operation == "exact_delete":
            lifecycle.compare_and_delete(
                lease.path, expected_fence=terminal.fence, lease_id=terminal.lease_id,
            )
        elif operation == "partial_finalize":
            lifecycle.repair_partial_finalize(
                lease.path, expected_terminal=terminal,
                expected_preterminal_state="preparing",
            )
        else:
            lifecycle.quarantine_exact_dead_owner(
                lease.path, expected_record_id=record.record_id,
                expected_fence=record.fence, expected_lease_id=record.lease_id,
                expected_task_id=record.task_id,
                expected_canonical_task_cid=record.canonical_task_cid,
                expected_attempt=record.attempt, expected_branch=record.branch,
                expected_merge_target=record.merge_target,
                expected_repo_root=record.repo_root,
                expected_state_dir=record.state_dir,
                fence_authority={"source": "fixture"},
            )
    assert q.census(repo, root) == before
    assert q.verify(repo, root) == frozen


def test_partial_or_malformed_native_pool_population_never_authorizes_fresh_root(
    tmp_path,
):
    repo, root, _, _, _, _ = seed(tmp_path)
    (root / ".pool-state" / "malformed.json").write_text('{"lease_token":"malformed"}')
    with pytest.raises(QuarantineDenied, match="pool_binding_invalid"):
        q.plan(repo, root)
    assert list(q.registry(repo).glob("*.json")) == []


def test_native_cross_lane_claim_guard_keeps_expired_resource_reserved(
    tmp_path, monkeypatch
):

    repo, root, _, _, _, _ = seed(tmp_path)
    holder = _resource_claim_daemon(repo, lane="retained-lane")
    other = _resource_claim_daemon(repo, lane="independent-lane")
    retained = _resource_claim_task("PCTDD-005", "external/ipfs_accelerate/retained")
    claims, _, reason, _ = holder._acquire_implementation_resource_claims(
        retained, attempt=1, started_at="2026-09-11T00:00:00Z"
    )
    assert reason == "acquired" and len(claims) == 1
    path, metadata = claims[0]
    original = path.read_bytes()
    frozen = q.freeze(repo, root, expected=q.census(repo, root))
    monkeypatch.setattr(other, "_lock_owner_is_active", lambda *args, **kwargs: False)
    overlap = _resource_claim_task(
        "PCTDD-008", "external/ipfs_accelerate/retained/child.py"
    )
    acquired, unavailable, reason, _ = other._acquire_implementation_resource_claims(
        overlap, attempt=1, started_at="2026-09-11T00:00:01Z"
    )
    assert acquired == [] and unavailable and reason == "overlapping_claim_exists"
    with pytest.raises(QuarantineDenied, match="workspace_claim_quarantined"):
        holder._release_implementation_resource_claim(path, metadata)
    independent = _resource_claim_task(
        "PCTDD-009", "external/ipfs_accelerate/independent.py"
    )
    acquired, _, reason, _ = other._acquire_implementation_resource_claims(
        independent, attempt=1, started_at="2026-09-11T00:00:02Z"
    )
    assert reason == "acquired" and len(acquired) == 1
    assert path.read_bytes() == original
    assert q.verify(repo, root) == frozen
    assert other._release_implementation_resource_claim(*acquired[0])
    assert q.verify(repo, root) == frozen


def test_all_native_global_maintenance_paths_defer_for_sibling_root(tmp_path):
    from types import SimpleNamespace
    from ipfs_accelerate_py.agent_supervisor.merge.git_gc import GitGarbageCollector
    from ipfs_accelerate_py.agent_supervisor.merge.merge_train import MergeTrain
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor import (
        PortalImplementationSupervisor,
    )
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
        PortalImplementationDaemon,
    )

    repo, root, _, _, _, _ = seed(tmp_path)
    sibling = tmp_path / "other-board"
    _git(repo, "worktree", "add", "--detach", str(sibling), "HEAD")
    frozen = q.freeze(repo, root, expected=q.census(repo, root))
    collector = GitGarbageCollector(repo_root=sibling)
    for action in (
        lambda: collector.run(aggressive=True),
        collector._prune_worktrees,
        lambda: collector._expire_reflogs(expire_all=True),
        collector._prune_objects,
    ):
        assert action()["reason"] == "retained_workspace_scope"
    assert collector.state.total_gc_runs == 0
    # Constructor-free native consumers: the custody boundary must return
    # before reading train queues/configuration or running any Git command.
    train = object.__new__(MergeTrain)
    train.repo_root = sibling
    assert train._cleanup_abandoned_worktrees() == 0
    daemon = object.__new__(PortalImplementationDaemon)
    daemon.repo_root = sibling
    daemon._database_attempt_authority = None
    # This native fork already denies all background peer cleanup because
    # it lacks canonical peer completion proof; preserve the stronger deny.
    assert (
        daemon._cleanup_already_merged_worktrees()["reason"]
        == "canonical_peer_cleanup_api_unavailable"
    )
    supervisor = object.__new__(PortalImplementationSupervisor)
    supervisor.config = SimpleNamespace(repo_root=sibling)
    assert supervisor.reconcile_backlogged_worktrees()["reason"] == "retained_workspace_scope"
    assert (
        supervisor._cleanup_backlogged_worktrees_locked()["reason"]
        == "retained_workspace_scope"
    )
    assert (
        supervisor._prune_managed_submodule_worktrees()["reason"]
        == "retained_workspace_scope"
    )
    assert q.verify(repo, root) == frozen


def test_nested_submodule_gc_observes_superproject_custody(tmp_path):
    from ipfs_accelerate_py.agent_supervisor.merge.git_gc import GitGarbageCollector

    repo, root, _, _, _, _ = seed(tmp_path)
    module = tmp_path / "module-source"
    module.mkdir()
    _git(module, "init")
    _git(module, "config", "user.name", "test")
    _git(module, "config", "user.email", "test@example.invalid")
    (module / "README").write_text("nested base\n")
    _git(module, "add", "README")
    _git(module, "commit", "-m", "nested base")
    _git(
        repo,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        str(module),
        "external/module",
    )
    frozen = q.freeze(repo, root, expected=q.census(repo, root))
    collector = GitGarbageCollector(repo_root=repo / "external/module")
    assert collector.run(aggressive=True)["reason"] == "retained_workspace_scope"
    assert q.verify(repo, root) == frozen


def nested_repository(tmp_path):
    repo, root, _, _, _, _ = seed(tmp_path)
    module = tmp_path / "nested-source"
    module.mkdir()
    _git(module, "init")
    _git(module, "config", "user.name", "test")
    _git(module, "config", "user.email", "test@example.invalid")
    (module / "README").write_text("nested base\n")
    _git(module, "add", "README")
    _git(module, "commit", "-m", "nested base")
    _git(repo, "-c", "protocol.file.allow=always", "submodule", "add",
         str(module), "external/module")
    return repo, root, repo / "external/module"


def test_nested_mutation_retains_parent_freeze_lock_until_exit(tmp_path):
    repo, root, module = nested_repository(tmp_path)
    before = q.census(repo, root)
    finished = threading.Event()
    outcomes = []

    def freeze():
        try:
            outcomes.append(q.freeze(repo, root, expected=before))
        except BaseException as error:
            outcomes.append(error)
        finally:
            finished.set()

    with q.mutation(module, root / "nested-workspace"):
        worker = threading.Thread(target=freeze)
        worker.start()
        assert not finished.wait(0.05)
    worker.join(timeout=5)
    assert finished.is_set() and len(outcomes) == 1
    assert isinstance(outcomes[0], dict), outcomes
    for target in (root, root / "nested-workspace", root.parent):
        with pytest.raises(QuarantineDenied, match="workspace_root_quarantined"):
            with q.mutation(module, target):
                pytest.fail("nested mutation admitted into parent freeze")
    # Physical overlap determines denial; unrelated module work can continue.
    with q.mutation(module, Path(outcomes[0]["fresh_root"]) / "independent"):
        assert q.verify(repo, root) == outcomes[0]


@pytest.mark.parametrize("bound_path", [True, False])
def test_nested_boundary_rejects_unreadable_or_unbound_parent_custody(
    tmp_path, bound_path
):
    from types import SimpleNamespace

    repo, root, module = nested_repository(tmp_path)
    frozen = q.freeze(repo, root, expected=q.census(repo, root))
    effects = []

    @q.mutation_boundary("workspace")
    def mutate(self, workspace=None):
        effects.append(workspace)

    owner = SimpleNamespace(repo_root=module)
    if bound_path:
        # Parent records must be validated before a disjoint path can proceed.
        next(q.registry(repo).glob("*.json")).write_text("{malformed")
    with pytest.raises((QuarantineDenied, ValueError)):
        mutate(owner, Path(frozen["fresh_root"]) if bound_path else None)
    assert effects == []



def _fifo_custody_reader(operation, path, repo, root, output, ready):
    ready.set()
    try:
        if operation == "direct":
            q.read_regular(path)
        elif operation == "registry":
            q.records(path.parent)
        elif operation == "mutation_lock":
            with q.directory_guard(path.parent):
                raise AssertionError("FIFO custody lock admitted")
        else:
            q.census(repo, root)
    except QuarantineDenied as error:
        output.put(str(error))
    except BaseException as error:
        output.put("unexpected: " + repr(error))
    else:
        output.put("unexpected admission")


@pytest.mark.parametrize(
    "operation", ["direct", "registry", "pool", "lifecycle", "claim", "mutation_lock"]
)
def test_fifo_custody_population_is_denied_without_waiting_for_writer(
    tmp_path, operation
):
    import multiprocessing
    import os

    repo, root, _, _, lifecycle, _ = seed(tmp_path)
    directory = {
        "direct": tmp_path,
        "registry": q.registry(repo),
        "pool": root / ".pool-state",
        "lifecycle": lifecycle.store_dir,
        "claim": q.registry(repo).parent / "implementation-task-claims",
        "mutation_lock": tmp_path / "fifo-lock-registry",
    }[operation]
    directory.mkdir(mode=0o700, exist_ok=True)
    filename = (
        "mutation.lock"
        if operation == "mutation_lock"
        else "fifo.lock" if operation == "claim" else "fifo.json"
    )
    path = directory / filename
    os.mkfifo(path, 0o600)
    # Isolate the actual native reader: a regression must fail in bounded time
    # even though no process ever opens the other end of this FIFO.
    context = multiprocessing.get_context("spawn")
    output = context.Queue()
    ready = context.Event()
    child = context.Process(
        target=_fifo_custody_reader, args=(operation, path, repo, root, output, ready)
    )
    child.start()
    try:
        # Match the shared-main fixture: import time is separate from the
        # unchanged ten-second bound on the actual native custody operation.
        assert ready.wait(timeout=45), "custody reader process failed to initialize"
        child.join(timeout=10)
        assert not child.is_alive(), "native custody read blocked opening a FIFO"
        assert child.exitcode == 0
        reason = output.get(timeout=1)
        expected = (
            "workspace_quarantine_lock_unowned"
            if operation == "mutation_lock"
            else "workspace_quarantine_file_invalid"
        )
        assert reason == expected
        assert path.is_fifo()
    finally:
        if child.is_alive():
            child.kill()
            child.join(timeout=3)
        output.close()
        output.join_thread()


@pytest.mark.parametrize("operation", ["registry", "census"])
def test_native_census_bounds_the_scanner_including_unmatched_entries(
    tmp_path, monkeypatch, operation
):
    from contextlib import contextmanager
    from types import SimpleNamespace

    repo, root, _, _, _, _ = seed(tmp_path)
    directory = q.registry(repo) if operation == "registry" else root / ".pool-state"
    monkeypatch.setattr(q, "MAX_RECORDS", 2)
    monkeypatch.setattr(q, "MAX_FILES", 2)
    # Registry has one extra slot for mutation.lock; census counts every name.
    bound = 3 if operation == "registry" else 2
    native_scandir = q.os.scandir
    consumed, closed = [], []

    def endless_unmatched_entries():
        for index in range(10_000):
            assert index <= bound, "scanner consumed past its first excess entry"
            consumed.append(index)
            yield SimpleNamespace(path=str(directory / f"ignored-{index}.other"))

    @contextmanager
    def controlled_scandir(path):
        if Path(path) == directory:
            try:
                yield endless_unmatched_entries()
            finally:
                closed.append(True)
        else:
            with native_scandir(path) as entries:
                yield entries

    monkeypatch.setattr(q.os, "scandir", controlled_scandir)
    reason = "registry_bound" if operation == "registry" else "population_bound"
    with pytest.raises(QuarantineDenied, match=reason):
        q.records(directory) if operation == "registry" else q.census(repo, root)
    assert consumed == list(range(bound + 1))
    assert closed == [True]


def test_disappeared_registry_does_not_become_an_empty_fence(tmp_path):
    with pytest.raises(FileNotFoundError):
        q.records(tmp_path / "missing-registry")


def test_registry_capacity_denies_new_freeze_before_overwriting_custody(
    tmp_path, monkeypatch
):
    repo, root, _, _, _, _ = seed(tmp_path)
    monkeypatch.setattr(q, "MAX_RECORDS", 1)
    frozen = q.freeze(repo, root, expected=q.census(repo, root))
    before = {path.name: path.read_bytes() for path in q.registry(repo).glob("*.json")}
    other_root = tmp_path / "other-board-root"
    proposed = q.plan(repo, other_root)
    with pytest.raises(QuarantineDenied, match="workspace_quarantine_registry_bound"):
        q.freeze(repo, other_root, expected=proposed["snapshot"])
    assert {
        path.name: path.read_bytes() for path in q.registry(repo).glob("*.json")
    } == before
    assert q.verify(repo, root) == frozen
    # Re-acknowledging existing custody still works at full capacity.
    assert q.freeze(repo, root, expected=frozen["snapshot"]) == frozen


@pytest.mark.parametrize("method", ["_run_implementation", "_run_implementation_in_ephemeral_worktree", "_create_seeded_worktree", "_cleanup_failed_setup_worktree", "_cleanup_merged_worktree", "_cleanup_worktree_submodules"])
def test_native_provider_workspace_entries_deny_before_mutation(tmp_path, method):
    import inspect
    repo, root, _, lease, _, _ = seed(tmp_path)
    frozen = q.freeze(repo, root, expected=q.census(repo, root))
    daemon = object.__new__(PortalImplementationDaemon)
    daemon.repo_root = repo
    daemon.worktree_root = root
    action = getattr(daemon, method)
    arguments = {
        name: lease.path if name == "worktree_path" else None
        for name, parameter in inspect.signature(action).parameters.items()
        if parameter.default is inspect.Parameter.empty
        and parameter.kind not in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}
    }
    if "worktree_path" in inspect.signature(action).parameters:
        arguments["worktree_path"] = lease.path
    with pytest.raises(QuarantineDenied, match="workspace_root_quarantined"):
        action(**arguments)
    assert q.verify(repo, root) == frozen


def test_native_supervisor_rescue_preserves_frozen_workspace_and_branches(tmp_path):
    from types import SimpleNamespace
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor import PortalImplementationSupervisor
    repo, root, _, lease, _, _ = seed(tmp_path)
    dirty = lease.path / "unknown-callback-output"
    dirty.write_text("preserve this callback effect")
    frozen = q.freeze(repo, root, expected=q.census(repo, root))
    before_refs = _git(repo, "show-ref")
    before_head = _git(lease.path, "rev-parse", "HEAD")
    supervisor = object.__new__(PortalImplementationSupervisor)
    supervisor.config = SimpleNamespace(repo_root=repo)
    with pytest.raises(QuarantineDenied, match="workspace_root_quarantined"):
        supervisor._rescue_dirty_worktree(
            lease.path, branch="attempt/ambiguous", head=before_head,
            target_ref="HEAD", status_lines=["?? unknown-callback-output"],
            reason="retained callback fixture",
        )
    assert _git(repo, "show-ref") == before_refs
    assert _git(lease.path, "rev-parse", "HEAD") == before_head
    assert dirty.read_text() == "preserve this callback effect"
    assert q.verify(repo, root) == frozen


def test_retained_root_denies_ancestor_cleanup_and_allows_disjoint_sibling(tmp_path):
    from types import SimpleNamespace
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
        PortalImplementationDaemon,
    )

    repo, root, _, _, _, _ = seed(tmp_path)
    frozen = q.freeze(repo, root, expected=q.census(repo, root))
    # An ancestor cleanup can recursively remove a nested retained workspace.
    # Exercise the native boundary without invoking Git or its cleanup body.
    daemon = object.__new__(PortalImplementationDaemon)
    daemon.repo_root = repo
    daemon._worktree_pool_effective_paths = {}
    effects = []
    daemon._cleanup_merged_worktree = lambda *args, **kwargs: (
        effects.append(args) or {"cleaned": False}
    )
    daemon._record_event = lambda *args, **kwargs: None
    for target in (root.parent, root, root / "nested"):
        with pytest.raises(QuarantineDenied, match="workspace_root_quarantined"):
            daemon._cleanup_failed_setup_worktree(
                target, "attempt/ancestor", task=SimpleNamespace(task_id="other"),
                attempt=1, exception_result={},
            )
        assert effects == []
    with q.mutation(repo, root.parent / "disjoint-sibling"):
        assert q.verify(repo, root) == frozen
