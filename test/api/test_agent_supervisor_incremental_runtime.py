from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import threading
import time
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.objectives import objective_graph
from ipfs_accelerate_py.agent_supervisor.objectives.backlog_refinery import (
    CodebaseScanInventory,
    scan_codebase_findings,
)
from ipfs_accelerate_py.agent_supervisor.objectives.objective_graph import scan_objective_gaps
from ipfs_accelerate_py.agent_supervisor.task_sources.dataset_store import ObjectiveDatasetStore
from ipfs_accelerate_py.agent_supervisor.todo_daemon.engine import CommandResult
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    PortalImplementationDaemon,
    PortalTask,
    PortalTaskState,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor import (
    PortalImplementationSupervisor,
    PortalSupervisorConfig,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.worktrees import (
    WorktreePool,
    python_identifier_worktree_basename,
)
from ipfs_accelerate_py.agent_supervisor.worktree_lifecycle import (
    DuplicateAttemptError,
    ProcessBirthIdentity,
    WorkspaceLifecycleState,
)


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    return result.stdout.strip()


def _python_c(source: str) -> str:
    """Build a hermetic command for the interpreter running this test."""

    return f"{shlex.quote(sys.executable)} -c {shlex.quote(source)}"


def _init_repo(path: Path) -> None:
    path.mkdir(parents=True)
    _git(path, "init")
    _git(path, "checkout", "-b", "main")
    _git(path, "config", "user.name", "Incremental Runtime Test")
    _git(path, "config", "user.email", "incremental@example.invalid")


def _seed_objective_repo(tmp_path: Path) -> tuple[Path, Path, Path]:
    repo = tmp_path / "repo"
    _init_repo(repo)
    (repo / "src").mkdir()
    (repo / "docs").mkdir()
    objective = repo / "objective.md"
    objective.write_text(
        """# Objective Heap

## INC-G001 Incremental proof

- Status: active
- Track: runtime
- Priority: P1
- Goal: Preserve equivalent objective plans across incremental scans.
- Evidence: AlphaRouter.dispatch, durable design notes, still_missing_contract
- Outputs: src, docs
- Validation: true
- Gap task: Add the remaining contract.
""",
        encoding="utf-8",
    )
    (repo / "src" / "alpha.py").write_text(
        "class AlphaRouter:\n    def dispatch(self):\n        return 'alpha'\n",
        encoding="utf-8",
    )
    (repo / "docs" / "runtime.md").write_text(
        "# Durable design notes\n\nThe durable design notes are available.\n",
        encoding="utf-8",
    )
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "seed incremental objective")
    return repo, objective, tmp_path / "datasets"


def test_objective_scan_skips_symlinks_and_never_reads_external_targets(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    objective = repo / "objective.md"
    objective.write_text("# Objective\n", encoding="utf-8")
    internal_target = repo / "internal.py"
    internal_target.write_text("INTERNAL_EVIDENCE = True\n", encoding="utf-8")
    internal_link = repo / "internal-link.py"
    internal_link.symlink_to(internal_target.name)
    external_target = tmp_path / "external.py"
    external_target.write_text(
        "EXTERNAL_EVIDENCE_MUST_NOT_BE_SCANNED = True\n",
        encoding="utf-8",
    )
    external_link = repo / "external-link.py"
    external_link.symlink_to(external_target)
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "seed symlink containment")

    candidates = objective_graph.objective_candidate_files(
        repo,
        objective_path=objective,
    )

    assert internal_target in candidates
    assert internal_link not in candidates
    assert external_link not in candidates
    records = objective_graph.collect_ast_dataset_records(
        repo,
        objective_path=objective,
    )
    assert records
    assert all(
        not Path(str(row["root_relative_path"])).is_absolute()
        for row in records
    )
    assert all(
        "EXTERNAL_EVIDENCE_MUST_NOT_BE_SCANNED" not in str(row.get("evidence_text") or "")
        for row in records
    )


def test_scan_details_are_content_addressed_durable_and_fully_recoverable(tmp_path: Path) -> None:
    store = ObjectiveDatasetStore(tmp_path / "datasets")
    details = [
        {
            "kind": "excluded_file",
            "path": "vendor/generated.bundle.js",
            "reason_code": "excluded_path_part",
            "matched_part": "vendor",
        },
        {
            "kind": "parser_failure",
            "path": "src/broken.py",
            "reason_code": "python_syntax_error",
            "error": "SyntaxError: invalid syntax at line 7",
            "line": 7,
        },
    ]

    first = store.persist_scan_details(
        scan_id="refill/tree:one",
        details=details,
        metadata={"scan_mode": "exhaustive", "repository": Path("/repo")},
    )
    assert first.detail_count == first.row_count == 2
    assert first.artifact_id == f"sha256:{first.sha256}"
    assert first.sha256 == sha256(first.jsonl_path.read_bytes()).hexdigest()
    assert first.byte_count == first.jsonl_path.stat().st_size
    assert first.reason_counts == {
        "excluded_path_part": 1,
        "python_syntax_error": 1,
    }
    assert store.load_scan_details(first) == details
    assert store.load_scan_details(first.to_dict()) == details
    assert store.load_scan_details("refill/tree:one") == details
    first_manifest = store.load_scan_details_manifest(first)
    assert first_manifest["artifact_id"] == first.artifact_id
    assert first_manifest["metadata"] == {
        "repository": "/repo",
        "scan_mode": "exhaustive",
    }

    # A subsequent incremental pass updates the logical latest pointer but
    # leaves the exhaustive pass's full diagnostic artifact addressable.
    incremental_details = [
        {
            "kind": "excluded_file",
            "path": "dist/output.js",
            "reason_code": "excluded_path_part",
            "matched_part": "dist",
        }
    ]
    second = store.persist_scan_details(
        scan_id="refill/tree:one",
        details=incremental_details,
        metadata={"scan_mode": "incremental"},
    )
    assert second.artifact_id != first.artifact_id
    assert store.load_scan_details("refill/tree:one") == incremental_details
    assert store.load_scan_details(first) == details
    assert first.jsonl_path.exists()
    assert first.manifest_path.exists()


def test_incremental_and_exhaustive_codebase_scans_report_same_coverage_dimensions(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "coverage-repo"
    _init_repo(repo)
    (repo / "first.py").write_text("# TODO: repair first path\n", encoding="utf-8")
    (repo / "second.py").write_text("# TODO: repair second path\n", encoding="utf-8")
    (repo / "asset.bin").write_bytes(b"not eligible\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "seed coverage dimensions")

    incremental = scan_codebase_findings(
        repo,
        max_findings=1,
        exhaustive=False,
        return_inventory=True,
    )
    exhaustive = scan_codebase_findings(
        repo,
        max_findings=1,
        exhaustive=True,
        return_inventory=True,
    )
    assert isinstance(incremental, CodebaseScanInventory)
    assert isinstance(exhaustive, CodebaseScanInventory)
    expected_dimensions = {
        "git_roots",
        "tracked_files",
        "eligible_files",
        "parsed_files",
        "cache_hits",
        "excluded_files",
        "parser_failures",
    }
    assert set(incremental.coverage_dict()) == expected_dimensions
    assert set(exhaustive.coverage_dict()) == expected_dimensions
    assert incremental.complete is False
    assert exhaustive.complete is True
    assert exhaustive.coverage_dict()["tracked_files"] == 3
    assert exhaustive.coverage_dict()["excluded_files"] == 1


def test_ast_and_evidence_snapshots_recompute_untrusted_source_cache(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo, objective, dataset_dir = _seed_objective_repo(tmp_path)
    parse_calls = 0
    real_parse = objective_graph.parse_python_ast_quietly

    def measured_parse(text: str):
        nonlocal parse_calls
        parse_calls += 1
        time.sleep(0.002)
        return real_parse(text)

    monkeypatch.setattr(objective_graph, "parse_python_ast_quietly", measured_parse)
    cold_stats: dict[str, object] = {}
    cold_plan = scan_objective_gaps(
        repo,
        objective_path=objective,
        max_findings=2,
        dataset_dir=dataset_dir,
        dataset_id="incremental-runtime",
        scan_stats=cold_stats,
    )
    cold_parse_calls = parse_calls
    assert cold_stats["parsed_record_count"] == 2
    assert cold_stats["reused_record_count"] == 0
    assert cold_parse_calls > 0

    warm_stats: dict[str, object] = {}
    warm_plan = scan_objective_gaps(
        repo,
        objective_path=objective,
        max_findings=2,
        dataset_dir=dataset_dir,
        dataset_id="incremental-runtime",
        scan_stats=warm_stats,
    )
    assert [asdict(item) for item in warm_plan] == [asdict(item) for item in cold_plan]
    assert parse_calls > cold_parse_calls
    assert warm_stats["parsed_record_count"] == 2
    assert warm_stats["reused_record_count"] == 0
    assert float(warm_stats["saved_parse_seconds"]) == 0

    (repo / "src" / "alpha.py").write_text(
        "class AlphaRouter:\n    def dispatch(self):\n        return 'changed'\n",
        encoding="utf-8",
    )
    changed_stats: dict[str, object] = {}
    scan_objective_gaps(
        repo,
        objective_path=objective,
        dataset_dir=dataset_dir,
        dataset_id="incremental-runtime",
        scan_stats=changed_stats,
    )
    assert changed_stats["parsed_record_count"] == 2
    assert changed_stats["reused_record_count"] == 0


def test_deleted_and_renamed_paths_remove_stale_evidence_deterministically(tmp_path: Path) -> None:
    repo, objective, dataset_dir = _seed_objective_repo(tmp_path)
    dataset_id = "rename-delete-runtime"
    scan_objective_gaps(
        repo,
        objective_path=objective,
        dataset_dir=dataset_dir,
        dataset_id=dataset_id,
    )

    _git(repo, "mv", "docs/runtime.md", "docs/renamed-runtime.md")
    rename_stats: dict[str, object] = {}
    renamed_plan = scan_objective_gaps(
        repo,
        objective_path=objective,
        dataset_dir=dataset_dir,
        dataset_id=dataset_id,
        scan_stats=rename_stats,
    )
    assert rename_stats["parsed_record_count"] == 2
    assert rename_stats["reused_record_count"] == 0
    assert rename_stats["renamed_record_count"] == 1
    assert rename_stats["deleted_record_count"] == 1
    assert renamed_plan[0].present_evidence["durable design notes"] == [
        "docs/renamed-runtime.md (exact)"
    ]
    rows = ObjectiveDatasetStore(dataset_dir).load_records(dataset_id)
    assert [row["root_relative_path"] for row in rows] == [
        "docs/renamed-runtime.md",
        "src/alpha.py",
    ]

    _git(repo, "rm", "-f", "docs/renamed-runtime.md")
    delete_stats: dict[str, object] = {}
    deleted_plan = scan_objective_gaps(
        repo,
        objective_path=objective,
        dataset_dir=dataset_dir,
        dataset_id=dataset_id,
        scan_stats=delete_stats,
    )
    assert delete_stats["deleted_record_count"] == 1
    assert delete_stats["invalidated_record_count"] == 1
    assert "durable design notes" in deleted_plan[0].missing_evidence
    remaining = ObjectiveDatasetStore(dataset_dir).load_records(dataset_id)
    assert [row["root_relative_path"] for row in remaining] == ["src/alpha.py"]
    manifest = ObjectiveDatasetStore(dataset_dir).load_manifest(dataset_id)
    assert manifest["row_count"] == 1
    assert manifest["deleted_record_count"] == 1


def _seed_repo_with_submodule(tmp_path: Path) -> tuple[Path, Path]:
    dependency = tmp_path / "dependency"
    _init_repo(dependency)
    (dependency / "dependency.py").write_text("VALUE = 7\n", encoding="utf-8")
    _git(dependency, "add", ".")
    _git(dependency, "commit", "-m", "seed dependency")

    repo = tmp_path / "implementation"
    _init_repo(repo)
    (repo / "app.py").write_text("from pathlib import Path\nVALUE = 7\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "seed implementation")
    _git(repo, "-c", "protocol.file.allow=always", "submodule", "add", str(dependency), "vendor/dependency")
    _git(repo, "commit", "-am", "add dependency")
    return repo, dependency


def _make_dead_missing_pool_lease(
    pool: WorktreePool,
    repo: Path,
    *,
    branch: str,
    delete_branch: bool = True,
):
    lease = pool.acquire(
        cache_key=f"orphan:{branch}",
        base_ref="main",
        branch_name=branch,
    )
    state_path = pool.state_root / f"{lease.entry_id}.json"
    lock_path = pool.state_root / f"{lease.entry_id}.lock"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["lease_pid"] = 2**30
    state_path.write_text(json.dumps(state), encoding="utf-8")
    lock_path.write_text(json.dumps({"pid": 2**30}), encoding="utf-8")
    _git(repo, "worktree", "remove", "--force", str(lease.path))
    if delete_branch:
        _git(repo, "branch", "-D", branch)
    return lease, state_path, lock_path


def test_generated_worktree_basename_is_a_deterministic_python_identifier() -> None:
    assert python_identifier_worktree_basename(
        "workspace",
        "ACCEL-012/child",
        "a1b2c3d4e5f6",
        "attempt",
        2,
        123,
    ) == "workspace_ACCEL_012_child_a1b2c3d4e5f6_attempt_2_123"
    for segments in (
        ("replay", "AUTO-004", "abc123", 456),
        ("main_merge", "release/v1", "implementation/task-1", 7),
        ("submodule_target", "1abc", 8),
        ("submodule_recovery", "9def", 8, 10),
    ):
        basename = python_identifier_worktree_basename(*segments)
        assert basename.isidentifier()
        assert "-" not in basename
        assert "/" not in basename


def test_non_pooled_attempt_uses_identifier_basename_and_legacy_branch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    state_dir = tmp_path / "state"
    daemon = PortalImplementationDaemon(
        todo_path=repo / "tasks.md",
        state_path=state_dir / "task-state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "events.jsonl",
        repo_root=repo,
        use_ephemeral_worktree=True,
        worktree_root=tmp_path / "worktrees",
        worktree_pool_enabled=False,
    )
    task = PortalTask(
        task_id="ACCEL-012/child",
        title="Prove a Ruff-safe checkout name",
        status="todo",
        completion="manual",
        priority="P1",
        track="runtime",
    )
    observed: dict[str, object] = {}

    def reject_duplicate_attempt(**kwargs):
        observed.update(kwargs)
        raise DuplicateAttemptError("fixture already owns the attempt")

    monkeypatch.setattr(
        daemon.worktree_lifecycle,
        "begin_preparing",
        reject_duplicate_attempt,
    )

    result = daemon._run_implementation_in_ephemeral_worktree(
        task=task,
        state=PortalTaskState(),
        attempt=2,
        started_at="2026-08-11T00:00:00+00:00",
        log_path=state_dir / "implementation.log",
        prompt="implement",
    )

    worktree_path = Path(str(result["worktree_path"]))
    assert observed["workspace_path"] == worktree_path
    assert worktree_path.name.isidentifier()
    assert worktree_path.name.startswith("workspace_accel_012_child_")
    assert "_attempt_2_" in worktree_path.name
    assert result["branch"].startswith("implementation/accel-012-child-")
    assert "-attempt-2-" in result["branch"]


def test_worktree_pool_replaces_readable_legacy_hyphenated_entry(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    (repo / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "seed")
    pool = WorktreePool(repo_root=repo, worktree_root=tmp_path / "pool")

    initial = pool.acquire(cache_key="ruff-safe", base_ref="main")
    assert initial.path.name == f"workspace_{initial.entry_id.replace('-', '_')}"
    assert initial.path.name.isidentifier()
    assert initial.entry_id.count("-") == 1
    assert initial.release()["pooled"] is True

    state_path = pool.state_root / f"{initial.entry_id}.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    legacy_path = pool.worktree_root / f"workspace-{initial.entry_id}"
    _git(repo, "worktree", "move", str(initial.path), str(legacy_path))
    state["path"] = str(legacy_path)
    state_path.write_text(json.dumps(state), encoding="utf-8")

    assert [item["lease_token"] for item in pool._states()] == [
        initial.entry_id
    ]
    replacement = pool.acquire(cache_key="ruff-safe", base_ref="main")

    assert replacement.reused is False
    assert "worktree_basename_not_python_identifier" in (
        replacement.invalidation_reasons
    )
    assert replacement.path.name == (
        f"workspace_{replacement.entry_id.replace('-', '_')}"
    )
    assert replacement.path.name.isidentifier()
    assert not legacy_path.exists()
    assert not state_path.exists()
    assert replacement.release()["pooled"] is True


def test_clean_dependency_workspaces_are_reused_without_task_mutation_leakage(tmp_path: Path) -> None:
    repo, _dependency = _seed_repo_with_submodule(tmp_path)
    pool = WorktreePool(repo_root=repo, worktree_root=tmp_path / "pool", max_entries=2)
    prepare_calls = 0

    def prepare(path: Path) -> None:
        nonlocal prepare_calls
        prepare_calls += 1
        time.sleep(0.02)
        _git(path, "-c", "protocol.file.allow=always", "submodule", "update", "--init", "--checkout")

    cold = pool.acquire(
        cache_key="linux-lock-v1",
        base_ref="main",
        branch_name="implementation/cold",
        dependency_paths=("vendor/dependency",),
        prepare=prepare,
    )
    assert cold.reused is False
    assert cold.path.name == f"workspace_{cold.entry_id.replace('-', '_')}"
    assert cold.path.name.isidentifier()
    assert (cold.path / "vendor" / "dependency" / "dependency.py").read_text(encoding="utf-8") == "VALUE = 7\n"
    cold_validation = subprocess.run(
        ["python", "-c", "from pathlib import Path; assert 'VALUE = 7' in Path('app.py').read_text()"],
        cwd=cold.path,
        capture_output=True,
        check=False,
    ).returncode
    (cold.path / "task-local.txt").write_text("first task only\n", encoding="utf-8")
    _git(cold.path, "add", "task-local.txt")
    _git(cold.path, "commit", "-m", "task-local mutation")
    cold_release = cold.release()
    assert cold_release["pooled"] is True

    warm = pool.acquire(
        cache_key="linux-lock-v1",
        base_ref="main",
        branch_name="implementation/warm",
        dependency_paths=("vendor/dependency",),
        prepare=prepare,
    )
    warm_validation = subprocess.run(
        ["python", "-c", "from pathlib import Path; assert 'VALUE = 7' in Path('app.py').read_text()"],
        cwd=warm.path,
        capture_output=True,
        check=False,
    ).returncode
    assert warm.reused is True
    assert prepare_calls == 1
    assert warm.estimated_seconds_saved > 0
    assert warm_validation == cold_validation == 0
    assert not (warm.path / "task-local.txt").exists()
    assert warm.release()["pooled"] is True
    assert pool.metrics["warm_acquisitions"] == 1


def test_pooled_admission_leaves_lifecycle_denied_entry_untouched(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "seed")
    worktree_root = tmp_path / "pool"
    daemon = PortalImplementationDaemon(
        todo_path=tmp_path / "tasks.md",
        state_path=tmp_path / "state.json",
        strategy_path=tmp_path / "strategy.json",
        events_path=tmp_path / "events.jsonl",
        repo_root=repo,
        use_ephemeral_worktree=True,
        worktree_root=worktree_root,
        worktree_pool_max_entries=1,
    )
    pool = daemon.worktree_pool
    assert pool is not None
    prior = pool.acquire(
        cache_key=daemon._implementation_worktree_cache_key(),
        base_ref="main",
        branch_name="implementation/prior-owner",
    )
    prior_path = prior.path
    prior_entry_id = prior.entry_id
    assert prior.release(reusable=True)["pooled"] is True

    lifecycle = daemon.worktree_lifecycle.begin_preparing(
        task_id="INC-POOL-OLD",
        canonical_task_cid="cid:pool-old",
        attempt=1,
        lane_id="dead-owner",
        workspace_path=prior_path,
        branch="implementation/prior-owner",
        merge_target="main",
        owner=ProcessBirthIdentity(
            pid=2**30 - 7,
            start_time_ticks=1,
            boot_id="dead-owner",
        ),
    )
    lifecycle = daemon.worktree_lifecycle.mark_active(
        prior_path,
        lease_id=lifecycle.lease_id,
        expected_fence=lifecycle.fence,
    )
    pool_state_path = (
        worktree_root / ".pool-state" / f"{prior_entry_id}.json"
    )
    lifecycle_path = daemon.worktree_lifecycle.workspace_path_for(prior_path)
    state_before = pool_state_path.read_bytes()
    lifecycle_before = lifecycle_path.read_bytes()
    head_before = _git(prior_path, "rev-parse", "HEAD")
    status_before = _git(prior_path, "status", "--porcelain")

    requested_path = worktree_root / "new-attempt"
    daemon._create_seeded_worktree(
        requested_path,
        "implementation/new-owner",
    )
    acquired_path = daemon._effective_pooled_worktree_path(requested_path)
    acquired = daemon._worktree_pool_leases[acquired_path]

    assert acquired.reused is False
    assert acquired.path != prior_path
    assert (
        "worktree_reuse_denied:owner_dead_lease_unexpired"
        in acquired.invalidation_reasons
    )
    assert pool_state_path.read_bytes() == state_before
    assert lifecycle_path.read_bytes() == lifecycle_before
    assert daemon.worktree_lifecycle.load_workspace(prior_path) == lifecycle
    assert _git(prior_path, "rev-parse", "HEAD") == head_before
    assert _git(prior_path, "status", "--porcelain") == status_before
    assert not (
        worktree_root / ".pool-state" / f"{prior_entry_id}.lock"
    ).exists()

    release = daemon._release_pooled_worktree_lease(
        acquired_path,
        reason="test_cleanup",
        reusable=True,
    )
    assert release["released"] is True
    assert release["pooled"] is True
    assert pool_state_path.read_bytes() == state_before
    assert lifecycle_path.read_bytes() == lifecycle_before
    assert _git(prior_path, "rev-parse", "HEAD") == head_before
    assert _git(prior_path, "status", "--porcelain") == status_before

    invalidation = pool.invalidate()
    denied_skip = next(
        item
        for item in invalidation["skipped"]
        if item["path"] == str(prior_path)
    )
    assert denied_skip["reason"] == (
        "worktree_reuse_denied:owner_dead_lease_unexpired"
    )
    assert pool_state_path.read_bytes() == state_before
    assert lifecycle_path.read_bytes() == lifecycle_before
    assert _git(prior_path, "rev-parse", "HEAD") == head_before
    assert _git(prior_path, "status", "--porcelain") == status_before


def test_pooled_admission_reclaims_expired_lifecycle_only_after_claim(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "seed")
    worktree_root = tmp_path / "pool"
    daemon = PortalImplementationDaemon(
        todo_path=tmp_path / "tasks.md",
        state_path=tmp_path / "state.json",
        strategy_path=tmp_path / "strategy.json",
        events_path=tmp_path / "events.jsonl",
        repo_root=repo,
        use_ephemeral_worktree=True,
        worktree_root=worktree_root,
    )
    pool = daemon.worktree_pool
    assert pool is not None
    prior = pool.acquire(
        cache_key=daemon._implementation_worktree_cache_key(),
        base_ref="main",
        branch_name="implementation/expired-owner",
    )
    prior_path = prior.path
    assert prior.release(reusable=True)["pooled"] is True

    daemon.worktree_lifecycle.clock = lambda: 1_000.0
    daemon.worktree_lifecycle.lease_seconds = 10.0
    lifecycle = daemon.worktree_lifecycle.begin_preparing(
        task_id="INC-POOL-EXPIRED",
        canonical_task_cid="cid:pool-expired",
        attempt=1,
        lane_id="dead-owner",
        workspace_path=prior_path,
        branch="implementation/expired-owner",
        merge_target="main",
        owner=ProcessBirthIdentity(
            pid=2**30 - 7,
            start_time_ticks=1,
            boot_id="dead-owner",
        ),
    )
    lifecycle = daemon.worktree_lifecycle.mark_active(
        prior_path,
        lease_id=lifecycle.lease_id,
        expected_fence=lifecycle.fence,
    )
    lifecycle_path = daemon.worktree_lifecycle.workspace_path_for(prior_path)
    lifecycle_before = lifecycle_path.read_bytes()
    daemon.worktree_lifecycle.clock = lambda: 1_011.0

    preflight_allowed, preflight_reason = (
        daemon._authorize_pooled_worktree_reuse(
            prior_path,
            lifecycle.branch,
            "preflight",
        )
    )
    assert preflight_allowed is True
    assert preflight_reason == "stale_owner_lease_expired"
    assert lifecycle_path.read_bytes() == lifecycle_before
    assert daemon.worktree_lifecycle.load_workspace(prior_path) == lifecycle

    requested_path = worktree_root / "new-attempt"
    daemon._create_seeded_worktree(
        requested_path,
        "implementation/reclaimed-owner",
    )
    acquired_path = daemon._effective_pooled_worktree_path(requested_path)
    acquired = daemon._worktree_pool_leases[acquired_path]
    reclaimed = daemon.worktree_lifecycle.load_workspace(prior_path)

    assert acquired.reused is True
    assert acquired.path == prior_path
    assert reclaimed is not None
    assert reclaimed.state is WorkspaceLifecycleState.TERMINAL
    assert reclaimed.fence == lifecycle.fence + 1

    missing_request = worktree_root / "missing-state-release"
    missing_branch = "implementation/missing-state-release"
    daemon._create_seeded_worktree(missing_request, missing_branch)
    missing_path = daemon._effective_pooled_worktree_path(missing_request)
    missing_lease = daemon._worktree_pool_leases[missing_path]
    (
        worktree_root
        / ".pool-state"
        / f"{missing_lease.entry_id}.json"
    ).unlink()

    generic_failure = daemon._cleanup_merged_worktree(
        missing_path,
        missing_branch,
        reusable=False,
    )

    assert generic_failure["cleaned"] is False
    assert generic_failure["deferred"] is False
    assert "failure_kind" not in generic_failure
    assert "attempt_consumed" not in generic_failure
    assert "provider_call_allowed" not in generic_failure
    assert generic_failure["pool_release"]["reason"] == "lease_state_missing"
    assert missing_path.exists()
    assert daemon._worktree_pool_leases[missing_path] is missing_lease
    assert missing_lease._released is False
    assert reclaimed.terminal_reason == "stale_owner_lease_expired"

    release = daemon._release_pooled_worktree_lease(
        acquired_path,
        reason="test_cleanup",
        reusable=False,
    )
    assert release["released"] is True


def test_lifecycle_denied_pool_release_stays_retryable_and_cannot_fall_through(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "seed")
    worktree_root = tmp_path / "pool"
    daemon = PortalImplementationDaemon(
        todo_path=tmp_path / "tasks.md",
        state_path=tmp_path / "state.json",
        strategy_path=tmp_path / "strategy.json",
        events_path=tmp_path / "events.jsonl",
        repo_root=repo,
        use_ephemeral_worktree=True,
        worktree_root=worktree_root,
    )
    requested_path = worktree_root / "release-attempt"
    branch_name = "implementation/release-owner"
    daemon._create_seeded_worktree(requested_path, branch_name)
    acquired_path = daemon._effective_pooled_worktree_path(requested_path)
    lease = daemon._worktree_pool_leases[acquired_path]

    daemon.worktree_lifecycle.clock = lambda: 2_000.0
    daemon.worktree_lifecycle.lease_seconds = 10.0
    lifecycle = daemon.worktree_lifecycle.begin_preparing(
        task_id="INC-POOL-RELEASE",
        canonical_task_cid="cid:pool-release",
        attempt=1,
        lane_id="dead-owner",
        workspace_path=acquired_path,
        branch=branch_name,
        merge_target="main",
        owner=ProcessBirthIdentity(
            pid=2**30 - 7,
            start_time_ticks=1,
            boot_id="dead-owner",
        ),
    )
    lifecycle = daemon.worktree_lifecycle.mark_active(
        acquired_path,
        lease_id=lifecycle.lease_id,
        expected_fence=lifecycle.fence,
    )
    pool_state_path = (
        worktree_root / ".pool-state" / f"{lease.entry_id}.json"
    )
    lock_path = worktree_root / ".pool-state" / f"{lease.entry_id}.lock"
    state_before = pool_state_path.read_bytes()
    lock_before = lock_path.read_bytes()
    head_before = _git(acquired_path, "rev-parse", "HEAD")

    # Model the narrow race where cleanup's first lifecycle check passed but a
    # conflicting claim appeared before the pool release gate.
    daemon._authorize_worktree_cleanup = lambda *_args, **_kwargs: {  # type: ignore[method-assign]
        "allowed": True,
        "reason": "precheck_passed",
    }
    denied = daemon._cleanup_merged_worktree(
        acquired_path,
        branch_name,
        reusable=False,
    )

    assert denied["cleaned"] is False
    assert denied["deferred"] is True
    assert denied["removed_worktree"] is False
    assert denied["pool_release"]["released"] is False
    assert denied["pool_release"]["retryable"] is True
    assert daemon._worktree_pool_leases[acquired_path] is lease
    assert lease._released is False
    assert pool_state_path.read_bytes() == state_before
    assert lock_path.read_bytes() == lock_before
    assert _git(acquired_path, "rev-parse", "HEAD") == head_before
    assert _git(repo, "show-ref", "--verify", f"refs/heads/{branch_name}")

    daemon.worktree_lifecycle.clock = lambda: 2_011.0
    retried = daemon._release_pooled_worktree_lease(
        acquired_path,
        reason="test_retry_after_expiry",
        reusable=False,
    )

    assert retried["released"] is True
    assert lease._released is True
    assert acquired_path not in daemon._worktree_pool_leases
    assert not acquired_path.exists()
    reclaimed = daemon.worktree_lifecycle.load_workspace(acquired_path)
    assert reclaimed is not None
    assert reclaimed.state is WorkspaceLifecycleState.TERMINAL
    assert reclaimed.fence == lifecycle.fence + 1


def test_failed_seed_cleanup_resolves_quarantined_effective_pool_path(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "seed")
    worktree_root = tmp_path / "pool"
    daemon = PortalImplementationDaemon(
        todo_path=tmp_path / "tasks.md",
        state_path=tmp_path / "state.json",
        strategy_path=tmp_path / "strategy.json",
        events_path=tmp_path / "events.jsonl",
        repo_root=repo,
        use_ephemeral_worktree=True,
        worktree_root=worktree_root,
    )
    requested_path = worktree_root / "provisional-seed-path"
    requested_key = requested_path.resolve()
    branch_name = "implementation/quarantined-seed"
    daemon._create_seeded_worktree(requested_path, branch_name)
    effective_path = daemon._worktree_pool_effective_paths[requested_key]
    lease = daemon._worktree_pool_leases[effective_path]

    lifecycle = daemon.worktree_lifecycle.begin_preparing(
        task_id="INC-POOL-QUARANTINED",
        canonical_task_cid="cid:pool-quarantined",
        attempt=1,
        lane_id="dead-owner",
        workspace_path=effective_path,
        branch=branch_name,
        merge_target="main",
        owner=ProcessBirthIdentity(
            pid=2**30 - 7,
            start_time_ticks=1,
            boot_id="dead-owner",
        ),
    )
    lifecycle = daemon.worktree_lifecycle.mark_active(
        effective_path,
        lease_id=lifecycle.lease_id,
        expected_fence=lifecycle.fence,
    )
    task = PortalTask(
        task_id="INC-POOL-QUARANTINED",
        title="Preserve quarantined pooled checkout",
        status="todo",
        completion="manual",
        priority="P1",
        track="runtime",
    )

    cleanup = daemon._cleanup_failed_setup_worktree(
        requested_path,
        branch_name,
        task=task,
        attempt=1,
        exception_result={"phase": "worktree_setup"},
    )

    assert cleanup["cleaned"] is False
    assert cleanup["failure_kind"] == "lifecycle_race"
    assert cleanup["attempt_consumed"] is False
    assert cleanup["worktree_path"] == str(effective_path)
    assert daemon._worktree_pool_effective_paths[requested_key] == effective_path
    assert daemon._worktree_pool_leases[effective_path] is lease
    assert effective_path.exists()
    assert not requested_path.exists()
    assert _git(repo, "show-ref", "--verify", f"refs/heads/{branch_name}")


def test_dirty_workspace_is_discarded_instead_of_shared(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    (repo / "app.py").write_text("VALUE = 'clean'\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "seed")
    pool = WorktreePool(repo_root=repo, worktree_root=tmp_path / "pool")

    dirty = pool.acquire(cache_key="setup-v1", base_ref="main", branch_name="implementation/dirty")
    (dirty.path / "secret.txt").write_text("must not leak\n", encoding="utf-8")
    release = dirty.release()
    assert release["pooled"] is False
    assert release["reason"] == "dirty_worktree"

    next_lease = pool.acquire(cache_key="setup-v1", base_ref="main", branch_name="implementation/next")
    assert next_lease.reused is False
    assert not (next_lease.path / "secret.txt").exists()
    assert next_lease.release()["pooled"] is True


def test_worktree_pool_reclaims_dead_leased_and_initializing_entries(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    (repo / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "seed")
    worktree_root = tmp_path / "pool"
    pool = WorktreePool(repo_root=repo, worktree_root=worktree_root)

    stale_leased = pool.acquire(
        cache_key="stale-leased",
        base_ref="main",
        branch_name="implementation/stale-leased",
    )
    stale_initializing = pool.acquire(
        cache_key="stale-initializing",
        base_ref="main",
        branch_name="implementation/stale-initializing",
    )
    stale_entries = (
        (stale_leased, "leased", 2_147_483_646),
        (stale_initializing, "initializing", 2_147_483_645),
    )
    for lease, state_name, dead_pid in stale_entries:
        state_path = worktree_root / ".pool-state" / f"{lease.entry_id}.json"
        lock_path = worktree_root / ".pool-state" / f"{lease.entry_id}.lock"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state["state"] = state_name
        state["lease_pid"] = dead_pid
        state_path.write_text(json.dumps(state), encoding="utf-8")
        lock_path.write_text(json.dumps({"pid": dead_pid}), encoding="utf-8")

    # Missing crashed workspaces must not strand their sidecars.
    for lease, _state_name, _dead_pid in stale_entries:
        _git(repo, "worktree", "remove", "--force", str(lease.path))
        assert not lease.path.exists()

    # Change the baseline and cache key so reclamation cannot depend on a
    # future acquisition matching either stale entry.
    (repo / "app.py").write_text("VALUE = 2\n", encoding="utf-8")
    _git(repo, "add", "app.py")
    _git(repo, "commit", "-m", "advance baseline")
    fresh = pool.acquire(
        cache_key="fresh",
        base_ref="main",
        branch_name="implementation/fresh",
    )

    for lease, _state_name, _dead_pid in stale_entries:
        assert not (worktree_root / ".pool-state" / f"{lease.entry_id}.json").exists()
        assert not (worktree_root / ".pool-state" / f"{lease.entry_id}.lock").exists()
        assert not lease.path.exists()
    assert fresh.invalidation_reasons == (
        "dead_lease_owner",
        "dead_lease_owner",
    )
    assert pool.metrics["reclaimed_dead_leases"] == 2
    assert fresh.release(reusable=False)["released"] is True


def test_worktree_pool_reclamation_preserves_live_and_recoverable_owners(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    (repo / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "seed")
    worktree_root = tmp_path / "pool"
    pool = WorktreePool(repo_root=repo, worktree_root=worktree_root)

    live_owner = pool.acquire(
        cache_key="live-owner",
        base_ref="main",
        branch_name="implementation/live-owner",
    )
    live_claimant = pool.acquire(
        cache_key="live-claimant",
        base_ref="main",
        branch_name="implementation/live-claimant",
    )
    recoverable_crash = pool.acquire(
        cache_key="recoverable-crash",
        base_ref="main",
        branch_name="implementation/recoverable-crash",
    )
    claimant_state_path = (
        worktree_root / ".pool-state" / f"{live_claimant.entry_id}.json"
    )
    claimant_state = json.loads(claimant_state_path.read_text(encoding="utf-8"))
    claimant_state["lease_pid"] = 2_147_483_644
    claimant_state_path.write_text(json.dumps(claimant_state), encoding="utf-8")
    # Keep the sidecar lock owned by this live process.  Reclamation must lose
    # this race even though the state record itself names a dead PID.
    recoverable_state_path = (
        worktree_root / ".pool-state" / f"{recoverable_crash.entry_id}.json"
    )
    recoverable_lock_path = (
        worktree_root / ".pool-state" / f"{recoverable_crash.entry_id}.lock"
    )
    recoverable_state = json.loads(
        recoverable_state_path.read_text(encoding="utf-8")
    )
    recoverable_state["lease_pid"] = 2_147_483_643
    recoverable_state_path.write_text(
        json.dumps(recoverable_state),
        encoding="utf-8",
    )
    recoverable_lock_path.write_text(
        json.dumps({"pid": 2_147_483_643}),
        encoding="utf-8",
    )
    # An existing dead-owner checkout may hold recoverable crash output and is
    # therefore reserved for the supervisor rescue path.

    fresh = pool.acquire(
        cache_key="fresh",
        base_ref="main",
        branch_name="implementation/fresh",
    )

    for lease in (live_owner, live_claimant, recoverable_crash):
        assert (worktree_root / ".pool-state" / f"{lease.entry_id}.json").exists()
        assert (worktree_root / ".pool-state" / f"{lease.entry_id}.lock").exists()
        assert lease.path.exists()
    assert fresh.invalidation_reasons == ()
    assert pool.metrics["reclaimed_dead_leases"] == 0
    assert fresh.release(reusable=False)["released"] is True
    assert recoverable_crash.release(reusable=False)["released"] is True
    assert live_claimant.release(reusable=False)["released"] is True
    assert live_owner.release(reusable=False)["released"] is True


def test_worktree_pool_serializes_dead_lock_replacement_between_claimants(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    (repo / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "seed")
    worktree_root = tmp_path / "pool"
    pool = WorktreePool(repo_root=repo, worktree_root=worktree_root)
    stale = pool.acquire(
        cache_key="stale",
        base_ref="main",
        branch_name="implementation/stale",
    )
    state_path = worktree_root / ".pool-state" / f"{stale.entry_id}.json"
    lock_path = worktree_root / ".pool-state" / f"{stale.entry_id}.lock"
    dead_pid = 2_147_483_646
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["lease_pid"] = dead_pid
    state_path.write_text(json.dumps(state), encoding="utf-8")
    lock_path.write_text(json.dumps({"pid": dead_pid}), encoding="utf-8")

    contenders = (
        WorktreePool(repo_root=repo, worktree_root=worktree_root),
        WorktreePool(repo_root=repo, worktree_root=worktree_root),
    )
    start = threading.Barrier(len(contenders) + 1)
    claims: list[Path | None] = []

    def claim(contender: WorktreePool) -> None:
        start.wait()
        claims.append(contender._try_claim(state))

    threads = [
        threading.Thread(target=claim, args=(contender,))
        for contender in contenders
    ]
    for thread in threads:
        thread.start()
    start.wait()
    for thread in threads:
        thread.join(timeout=10)
        assert not thread.is_alive()

    assert sum(claim is not None for claim in claims) == 1
    live_payload = lock_path.read_bytes()
    assert json.loads(live_payload)["pid"] == os.getpid()
    live_inode = lock_path.stat().st_ino

    # Once a claimant owns the durable lock, another claim attempt must neither
    # succeed nor unlink/recreate that live ownership record.
    assert pool._try_claim(state) is None
    assert lock_path.read_bytes() == live_payload
    assert lock_path.stat().st_ino == live_inode

    pool._remove_lock(lock_path)
    assert pool._discard_state(state)["removed"] is True


def test_implementation_daemon_uses_stable_pooled_path_for_populated_submodules(tmp_path: Path) -> None:
    repo, _dependency = _seed_repo_with_submodule(tmp_path)
    worktree_root = tmp_path / "daemon-pool"
    daemon = PortalImplementationDaemon(
        todo_path=tmp_path / "tasks.md",
        state_path=tmp_path / "state.json",
        strategy_path=tmp_path / "strategy.json",
        events_path=tmp_path / "events.jsonl",
        repo_root=repo,
        use_ephemeral_worktree=True,
        worktree_root=worktree_root,
        worktree_submodule_paths=("vendor/dependency",),
    )

    requested_cold = worktree_root / "task-attempt-cold"
    cold_baseline = daemon._create_seeded_worktree(
        requested_cold,
        "implementation/daemon-cold",
    )
    cold_path = daemon._effective_pooled_worktree_path(requested_cold)
    assert cold_path.exists()
    assert cold_path != requested_cold
    assert daemon._worktree_setup_result(cold_path)["cache_hit"] is False
    assert daemon._cleanup_merged_worktree(cold_path, "implementation/daemon-cold")["pooled"] is True

    requested_warm = worktree_root / "task-attempt-warm"
    warm_baseline = daemon._create_seeded_worktree(
        requested_warm,
        "implementation/daemon-warm",
    )
    warm_path = daemon._effective_pooled_worktree_path(requested_warm)
    warm_setup = daemon._worktree_setup_result(warm_path)
    assert warm_path == cold_path
    assert warm_baseline == cold_baseline
    assert warm_setup["cache_hit"] is True
    assert warm_setup["saved_duration_seconds"] >= 0
    assert _git(warm_path, "status", "--porcelain") == ""
    assert daemon._cleanup_merged_worktree(warm_path, "implementation/daemon-warm")["pooled"] is True


def test_implementation_daemon_releases_pool_lease_before_merge_queue_handoff(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "seed")
    worktree_root = tmp_path / "pool"
    todo_path = tmp_path / "tasks.md"
    todo_path.write_text(
        "## INC-001 Release pooled merge handoff\n\n"
        "- Status: todo\n"
        "- Completion: manual\n"
        "- Priority: P1\n"
        "- Track: runtime\n"
        "- Outputs: feature.py\n"
        "- Validation: python -m py_compile feature.py\n",
        encoding="utf-8",
    )
    daemon = PortalImplementationDaemon(
        todo_path=todo_path,
        state_path=tmp_path / "state.json",
        strategy_path=tmp_path / "strategy.json",
        events_path=tmp_path / "events.jsonl",
        repo_root=repo,
        task_header_prefix="## INC-",
        implement=True,
        implementation_command=_python_c(
            "from pathlib import Path; "
            "Path('feature.py').write_text('VALUE = 1\\n')"
        ),
        use_ephemeral_worktree=True,
        worktree_root=worktree_root,
    )
    daemon._consume_one_merge_candidate = lambda: None  # type: ignore[method-assign]
    monkeypatch.setattr(
        daemon,
        "_run_validation_with_candidate_binding",
        lambda *_args, **_kwargs: {
            "attempted": True,
            "passed": True,
            "returncode": 0,
            "results": [],
        },
    )
    task = daemon._load_tasks()[0]

    result = daemon._run_implementation(task, PortalTaskState())

    merge_result = result["merge_result"]
    handoff = merge_result["worktree_pool_handoff"]
    assert merge_result["queued"] is True
    assert handoff["released"] is True
    assert handoff["pooled"] is True
    assert handoff["lifecycle_finalize"]["finalized"] is True
    assert handoff["lifecycle_finalize"]["state"] == "terminal"
    assert handoff["lifecycle_finalize"]["reason"] == "pooled_merge_queue_handoff"
    assert merge_result["worktree_lifecycle_handoff"]["finalized"] is True
    assert daemon._active_worktree_lifecycle is None
    assert (
        daemon.worktree_lifecycle.load_workspace(Path(result["worktree_path"]))
        is None
    )
    assert daemon._worktree_pool_leases == {}
    assert daemon._active_worktree_lifecycle is None
    assert list(daemon.worktree_lifecycle.iter_records()) == []
    assert daemon.worktree_lifecycle.load_task_attempt(
        canonical_task_cid=daemon._canonical_ref(task),
        task_id=task.task_id,
        attempt=result["attempt"],
    ) is None
    assert list((worktree_root / ".pool-state").glob("*.lock")) == []
    assert (
        daemon.worktree_lifecycle.load_task_attempt(
            canonical_task_cid=daemon._canonical_ref(task),
            task_id=task.task_id,
            attempt=1,
        )
        is None
    )
    queued = daemon.merge_queue.dequeue(consumer_id="merge-train:test")
    assert queued is not None
    assert queued.metadata["worktree_path"] == ""
    assert queued.metadata["worktree_pool_handoff"] is True
    assert _git(repo, "rev-parse", result["branch"]) == result["implementation_commit"]


def test_failed_implementation_does_not_pin_pooled_worktree(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "seed")
    worktree_root = tmp_path / "pool"
    daemon = PortalImplementationDaemon(
        todo_path=tmp_path / "tasks.md",
        state_path=tmp_path / "state.json",
        strategy_path=tmp_path / "strategy.json",
        events_path=tmp_path / "events.jsonl",
        repo_root=repo,
        implement=True,
        implementation_command=_python_c("raise SystemExit(7)"),
        use_ephemeral_worktree=True,
        worktree_root=worktree_root,
    )
    task = PortalTask(
        task_id="INC-002",
        title="Release failed pooled implementation",
        status="todo",
        completion="manual",
        priority="P1",
        track="runtime",
    )

    result = daemon._run_implementation(task, PortalTaskState())

    assert result["returncode"] == 7
    assert result["cleanup_result"]["reason"] == "failed_implementation_pool_lease_released"
    assert result["cleanup_result"]["pool_release"]["released"] is True
    assert (
        result["cleanup_result"]["pool_release"]["lifecycle_finalize"][
            "finalized"
        ]
        is True
    )
    assert (
        result["cleanup_result"]["pool_release"]["lifecycle_finalize"]["state"]
        == "terminal"
    )
    assert daemon._active_worktree_lifecycle is None
    assert (
        daemon.worktree_lifecycle.load_workspace(Path(result["worktree_path"]))
        is None
    )
    assert daemon._worktree_pool_leases == {}
    assert daemon._active_worktree_lifecycle is None
    assert list(daemon.worktree_lifecycle.iter_records()) == []
    assert daemon.worktree_lifecycle.load_task_attempt(
        canonical_task_cid=daemon._canonical_ref(task),
        task_id=task.task_id,
        attempt=result["attempt"],
    ) is None
    assert list((worktree_root / ".pool-state").glob("*.lock")) == []


def test_pooled_provider_deferral_releases_same_attempt_lifecycle(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "seed")
    worktree_root = tmp_path / "pool"
    daemon = PortalImplementationDaemon(
        todo_path=tmp_path / "tasks.md",
        state_path=tmp_path / "state.json",
        strategy_path=tmp_path / "strategy.json",
        events_path=tmp_path / "events.jsonl",
        repo_root=repo,
        implement=True,
        implementation_command=_python_c(
            "print(\"ERROR: You've hit your usage limit.\"); "
            "raise SystemExit(1)"
        ),
        use_ephemeral_worktree=True,
        worktree_root=worktree_root,
    )
    task = PortalTask(
        task_id="INC-003",
        title="Release deferred pooled implementation lifecycle",
        status="todo",
        completion="manual",
        priority="P1",
        track="runtime",
    )

    first = daemon._run_implementation(task, PortalTaskState())

    assert first["deferred"] is True
    assert first["reason"] == "provider_capacity_exhausted"
    assert first["attempt_consumed"] is False
    assert first["cleanup_result"]["pool_release"]["released"] is True
    assert first["cleanup_result"]["lifecycle_finalize"]["finalized"] is True
    assert (
        daemon.worktree_lifecycle.load_task_attempt(
            canonical_task_cid=daemon._canonical_ref(task),
            task_id=task.task_id,
            attempt=1,
        )
        is None
    )

    daemon._active_provider_capacity_backoff = lambda: {}  # type: ignore[method-assign]
    daemon.implementation_command = _python_c("raise SystemExit(7)")
    second = daemon._run_implementation(
        task,
        PortalTaskState.load(daemon.state_path),
    )

    assert second["returncode"] == 7
    assert second.get("reason") != "worktree_lifecycle_claim_exists"


def test_nonpooled_provider_exit_finalizes_preserved_worktree_lifecycle(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "seed")
    worktree_root = tmp_path / "worktrees"
    daemon = PortalImplementationDaemon(
        todo_path=tmp_path / "tasks.md",
        state_path=tmp_path / "state.json",
        strategy_path=tmp_path / "strategy.json",
        events_path=tmp_path / "events.jsonl",
        repo_root=repo,
        implement=True,
        implementation_command=_python_c(
            "print(\"ERROR: You've hit your usage limit.\"); "
            "raise SystemExit(1)"
        ),
        use_ephemeral_worktree=True,
        worktree_root=worktree_root,
        worktree_pool_enabled=False,
    )
    task = PortalTask(
        task_id="INC-004",
        title="Release deferred non-pooled implementation lifecycle",
        status="todo",
        completion="manual",
        priority="P1",
        track="runtime",
    )

    first = daemon._run_implementation(task, PortalTaskState())

    assert first["deferred"] is True
    assert first["reason"] == "provider_capacity_exhausted"
    assert first["attempt_consumed"] is False
    assert first["cleanup_result"]["reason"] == (
        "failed_implementation_worktree_preserved"
    )
    assert first["cleanup_result"]["cleaned"] is False
    assert first["cleanup_result"]["lifecycle_finalize"]["finalized"] is True
    first_worktree = Path(first["worktree_path"])
    assert first_worktree.exists()
    assert (
        daemon.worktree_lifecycle.load_task_attempt(
            canonical_task_cid=daemon._canonical_ref(task),
            task_id=task.task_id,
            attempt=1,
        )
        is None
    )

    # Remove the intentionally preserved diagnostic checkout so an immediate
    # same-attempt retry cannot collide with the timestamp-derived branch.
    _git(repo, "worktree", "remove", "--force", str(first_worktree))
    _git(repo, "branch", "-D", first["branch"])
    daemon._active_provider_capacity_backoff = lambda: {}  # type: ignore[method-assign]
    daemon.implementation_command = _python_c("raise SystemExit(7)")

    second = daemon._run_implementation(
        task,
        PortalTaskState.load(daemon.state_path),
    )

    assert second["returncode"] == 7
    assert second.get("reason") != "worktree_lifecycle_claim_exists"
    assert second["cleanup_result"]["reason"] == (
        "failed_implementation_worktree_preserved"
    )
    assert second["cleanup_result"]["cleaned"] is False
    assert second["cleanup_result"]["lifecycle_finalize"]["finalized"] is True
    assert Path(second["worktree_path"]).exists()
    assert (
        daemon.worktree_lifecycle.load_task_attempt(
            canonical_task_cid=daemon._canonical_ref(task),
            task_id=task.task_id,
            attempt=1,
        )
        is None
    )


def test_missing_pooled_workspace_is_discarded_after_setup_race(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "seed")
    worktree_root = tmp_path / "pool"
    daemon = PortalImplementationDaemon(
        todo_path=tmp_path / "tasks.md",
        state_path=tmp_path / "state.json",
        strategy_path=tmp_path / "strategy.json",
        events_path=tmp_path / "events.jsonl",
        repo_root=repo,
        implement=True,
        implementation_command=_python_c("raise AssertionError('must not run')"),
        use_ephemeral_worktree=True,
        worktree_root=worktree_root,
    )

    def remove_workspace_before_launch(*_args, **kwargs) -> None:
        workspace = Path(kwargs["worktree_path"])
        _git(repo, "worktree", "remove", "--force", str(workspace))
        raise FileNotFoundError(f"workspace disappeared: {workspace}")

    monkeypatch.setattr(
        daemon,
        "_mark_implementation_started",
        remove_workspace_before_launch,
    )
    task = PortalTask(
        task_id="INC-003",
        title="Discard missing pooled implementation",
        status="todo",
        completion="manual",
        priority="P1",
        track="runtime",
    )

    result = daemon._run_implementation(task, PortalTaskState())

    assert result["returncode"] == 1
    assert result["exception_result"]["exception_type"] == "FileNotFoundError"
    assert result["cleanup_result"]["pool_release"]["reason"] == "reuse_disabled"
    assert daemon._worktree_pool_leases == {}
    assert list((worktree_root / ".pool-state").glob("*.json")) == []
    assert list((worktree_root / ".pool-state").glob("*.lock")) == []


def test_supervisor_does_not_reconcile_a_live_pooled_worktree(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "seed")
    worktree_root = tmp_path / "pool"
    branch = "implementation/live-pool-race"
    pool = WorktreePool(repo_root=repo, worktree_root=worktree_root)
    lease = pool.acquire(
        cache_key="live-pool-race",
        base_ref="main",
        branch_name=branch,
    )
    _git(lease.path, "commit", "--allow-empty", "-m", "candidate")
    candidate_head = _git(lease.path, "rev-parse", "HEAD")
    (lease.path / "feature.py").write_text("VALUE = 1\n", encoding="utf-8")

    state_dir = tmp_path / "state" / "lane-0"
    supervisor = PortalImplementationSupervisor(
        PortalSupervisorConfig(
            todo_path=tmp_path / "tasks.md",
            state_path=state_dir / "task_state.json",
            strategy_path=state_dir / "strategy.json",
            events_path=state_dir / "events.jsonl",
            state_dir=state_dir,
            repo_root=repo,
            worktree_root=worktree_root,
        )
    )
    supervisor._list_process_commands = lambda: []  # type: ignore[method-assign]

    result = supervisor.reconcile_backlogged_worktrees()

    live_skip = next(
        item
        for item in result["skipped"]
        if item["reason"] == "active_worktree_pool_lease"
    )
    assert live_skip["path"] == str(lease.path)
    assert live_skip["owner_source"] == "worktree_pool_lease"
    assert live_skip["owner_lease_state"] == "leased"
    assert live_skip["owner_pool_state_path"].endswith(f"{lease.entry_id}.json")
    assert result["processed_count"] == 0
    assert _git(lease.path, "branch", "--show-current") == branch
    assert _git(lease.path, "rev-parse", "HEAD") == candidate_head
    assert (lease.path / "feature.py").read_text(encoding="utf-8") == "VALUE = 1\n"
    assert _git(repo, "for-each-ref", "--format=%(refname)", "refs/heads/rescue/worktree") == ""

    release = lease.release(reusable=False)
    assert release["released"] is True


def test_supervisor_does_not_fence_a_dead_pooled_worktree_lease(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "seed")
    worktree_root = tmp_path / "pool"
    pool = WorktreePool(repo_root=repo, worktree_root=worktree_root)
    lease = pool.acquire(
        cache_key="dead-pool-owner",
        base_ref="main",
        branch_name="implementation/dead-pool-owner",
    )
    state_path = worktree_root / ".pool-state" / f"{lease.entry_id}.json"
    lock_path = worktree_root / ".pool-state" / f"{lease.entry_id}.lock"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["lease_pid"] = 0
    state_path.write_text(json.dumps(state), encoding="utf-8")
    lock_path.write_text(json.dumps({"pid": 0}), encoding="utf-8")

    state_dir = tmp_path / "state" / "lane-0"
    supervisor = PortalImplementationSupervisor(
        PortalSupervisorConfig(
            todo_path=tmp_path / "tasks.md",
            state_path=state_dir / "task_state.json",
            strategy_path=state_dir / "strategy.json",
            events_path=state_dir / "events.jsonl",
            state_dir=state_dir,
            repo_root=repo,
            worktree_root=worktree_root,
        )
    )

    owners = supervisor._shared_active_worktree_owners(worktree_root)

    assert lease.path.resolve() not in owners
    release = lease.release(reusable=False)
    assert release["released"] is True


def test_worktree_pool_reconciles_dead_missing_metadata_with_a_bounded_pass(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "seed")
    pool = WorktreePool(repo_root=repo, worktree_root=tmp_path / "pool")
    leases = [
        pool.acquire(
            cache_key=f"orphan:implementation/orphan-{ordinal}",
            base_ref="main",
            branch_name=f"implementation/orphan-{ordinal}",
        )
        for ordinal in range(2)
    ]
    orphan_entries = []
    for ordinal, lease in enumerate(leases):
        branch = f"implementation/orphan-{ordinal}"
        state_path = pool.state_root / f"{lease.entry_id}.json"
        lock_path = pool.state_root / f"{lease.entry_id}.lock"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state["lease_pid"] = 2**30
        state_path.write_text(json.dumps(state), encoding="utf-8")
        lock_path.write_text(json.dumps({"pid": 2**30}), encoding="utf-8")
        _git(repo, "worktree", "remove", "--force", str(lease.path))
        _git(repo, "branch", "-D", branch)
        orphan_entries.append((lease, state_path, lock_path))

    first = pool.reconcile_orphaned_metadata(max_entries=1)

    assert first["candidate_count"] == 2
    assert first["inspected_count"] == 1
    assert first["removed_count"] == 1
    assert first["skipped_count"] == 0
    assert first["truncated"] is True
    assert first["removed"][0]["reason"] == (
        "dead_lease_workspace_and_branch_absent"
    )
    assert sum(path.exists() for _, path, _ in orphan_entries) == 1
    assert sum(path.exists() for _, _, path in orphan_entries) == 1

    state_dir = tmp_path / "state" / "lane-0"
    supervisor = PortalImplementationSupervisor(
        PortalSupervisorConfig(
            todo_path=tmp_path / "tasks.md",
            state_path=state_dir / "task_state.json",
            strategy_path=state_dir / "strategy.json",
            events_path=state_dir / "events.jsonl",
            state_dir=state_dir,
            repo_root=repo,
            worktree_root=pool.worktree_root,
        )
    )
    second = supervisor.reconcile_orphaned_worktree_pool_metadata(
        max_entries=10
    )

    assert second["candidate_count"] == 1
    assert second["removed_count"] == 1
    assert second["truncated"] is False
    assert all(not path.exists() for _, path, _ in orphan_entries)
    assert all(not path.exists() for _, _, path in orphan_entries)
    event = json.loads(
        state_dir.joinpath("events.jsonl").read_text(
            encoding="utf-8"
        )
    )
    assert event["type"] == (
        "worktree_pool_orphan_metadata_reconciled"
    )


def test_worktree_pool_orphan_reconciliation_preserves_any_recovery_signal(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "seed")
    pool = WorktreePool(repo_root=repo, worktree_root=tmp_path / "pool")

    live = pool.acquire(
        cache_key="live",
        base_ref="main",
        branch_name="implementation/live-owner",
    )
    present = pool.acquire(
        cache_key="present",
        base_ref="main",
        branch_name="implementation/present-workspace",
    )
    present_state_path = pool.state_root / f"{present.entry_id}.json"
    present_lock_path = pool.state_root / f"{present.entry_id}.lock"
    present_state = json.loads(
        present_state_path.read_text(encoding="utf-8")
    )
    present_state["lease_pid"] = 2**30
    present_state_path.write_text(
        json.dumps(present_state),
        encoding="utf-8",
    )
    present_lock_path.write_text(
        json.dumps({"pid": 2**30}),
        encoding="utf-8",
    )
    branch_only, branch_state_path, branch_lock_path = (
        _make_dead_missing_pool_lease(
            pool,
            repo,
            branch="implementation/surviving-branch",
            delete_branch=False,
        )
    )

    result = pool.reconcile_orphaned_metadata()

    assert result["removed_count"] == 0
    assert {
        item["reason"] for item in result["skipped"]
    } == {
        "branch_present",
        "live_lease_owner",
        "workspace_present_or_unsafe",
    }
    assert present_state_path.exists()
    assert present_lock_path.exists()
    assert branch_state_path.exists()
    assert branch_lock_path.exists()
    assert live.release(reusable=False)["released"] is True
    assert present.release(reusable=False)["released"] is True
    assert branch_only.release(reusable=False)["released"] is True
    _git(repo, "branch", "-D", "implementation/surviving-branch")


def test_worktree_pool_orphan_reconciliation_preserves_replaced_state(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "seed")
    pool = WorktreePool(repo_root=repo, worktree_root=tmp_path / "pool")
    _lease, state_path, lock_path = _make_dead_missing_pool_lease(
        pool,
        repo,
        branch="implementation/replaced-orphan",
    )
    original_try_claim = pool._try_claim

    def replace_state_after_claim(state):
        claimed = original_try_claim(state)
        if claimed is not None:
            replacement = json.loads(
                state_path.read_text(encoding="utf-8")
            )
            replacement["last_used_at_epoch"] = (
                float(replacement["last_used_at_epoch"]) + 1.0
            )
            state_path.write_text(
                json.dumps(replacement),
                encoding="utf-8",
            )
        return claimed

    monkeypatch.setattr(pool, "_try_claim", replace_state_after_claim)

    result = pool.reconcile_orphaned_metadata()

    assert result["removed_count"] == 0
    assert result["skipped_count"] == 1
    assert result["skipped"][0]["reason"] == (
        "state_changed_during_cleanup"
    )
    assert state_path.exists()
    assert not lock_path.exists()


def test_worktree_pool_orphan_reconciliation_preserves_unverifiable_owners(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "seed")
    pool = WorktreePool(repo_root=repo, worktree_root=tmp_path / "pool")
    invalid_state_lease = pool.acquire(
        cache_key="invalid-state-owner",
        base_ref="main",
        branch_name="implementation/unverifiable-lease-owner",
    )
    invalid_lock_lease = pool.acquire(
        cache_key="invalid-lock-owner",
        base_ref="main",
        branch_name="implementation/unverifiable-lock-owner",
    )

    invalid_state_path = pool._state_path(invalid_state_lease.entry_id)
    invalid_state_lock = pool._lock_path(
        json.loads(invalid_state_path.read_text(encoding="utf-8"))
    )
    invalid_state = json.loads(
        invalid_state_path.read_text(encoding="utf-8")
    )
    invalid_state["lease_pid"] = "not-a-pid"
    invalid_state_path.write_text(
        json.dumps(invalid_state),
        encoding="utf-8",
    )
    invalid_state_lock.write_text(
        json.dumps({"pid": 2**30}),
        encoding="utf-8",
    )
    _git(
        repo,
        "worktree",
        "remove",
        "--force",
        str(invalid_state_lease.path),
    )
    _git(repo, "branch", "-D", "implementation/unverifiable-lease-owner")

    invalid_lock_state = pool._state_path(invalid_lock_lease.entry_id)
    invalid_lock_payload = json.loads(
        invalid_lock_state.read_text(encoding="utf-8")
    )
    invalid_lock_payload["lease_pid"] = 2**30
    invalid_lock_state.write_text(
        json.dumps(invalid_lock_payload),
        encoding="utf-8",
    )
    invalid_lock_path = pool._lock_path(invalid_lock_payload)
    invalid_lock_path.write_text(
        json.dumps({"pid": 0}),
        encoding="utf-8",
    )
    _git(
        repo,
        "worktree",
        "remove",
        "--force",
        str(invalid_lock_lease.path),
    )
    _git(repo, "branch", "-D", "implementation/unverifiable-lock-owner")

    result = pool.reconcile_orphaned_metadata()

    assert result["removed_count"] == 0
    assert {
        item["reason"] for item in result["skipped"]
    } == {
        "lease_owner_unverifiable",
        "lock_owner_unverifiable",
    }
    assert invalid_state_path.exists()
    assert invalid_state_lock.exists()
    assert invalid_lock_state.exists()
    assert invalid_lock_path.exists()


def test_worktree_pool_orphan_reconciliation_preserves_unverifiable_branch_probe(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "seed")
    pool = WorktreePool(repo_root=repo, worktree_root=tmp_path / "pool")
    _lease, state_path, lock_path = _make_dead_missing_pool_lease(
        pool,
        repo,
        branch="implementation/unverifiable-branch-probe",
    )
    original_run = pool._run

    def fail_branch_probe(command, *, cwd):
        if tuple(command[:4]) == (
            "git",
            "show-ref",
            "--verify",
            "--quiet",
        ):
            return CommandResult(
                command=tuple(command),
                returncode=128,
                stdout="",
                stderr="injected branch probe failure",
            )
        return original_run(command, cwd=cwd)

    monkeypatch.setattr(pool, "_run", fail_branch_probe)

    result = pool.reconcile_orphaned_metadata()

    assert result["removed_count"] == 0
    assert result["skipped_count"] == 1
    assert result["skipped"][0]["reason"] == (
        "branch_presence_unverifiable"
    )
    assert result["skipped"][0]["branch_probe"] == {
        "returncode": 128,
        "error": "injected branch probe failure",
    }
    assert state_path.exists()
    assert lock_path.exists()


def test_supervisor_does_not_cleanup_an_idle_pooled_worktree(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "seed")
    worktree_root = tmp_path / "pool"
    pool = WorktreePool(repo_root=repo, worktree_root=worktree_root)
    lease = pool.acquire(
        cache_key="idle-pool-entry",
        base_ref="main",
        branch_name="implementation/idle-pool-entry",
    )
    idle_path = lease.path
    entry_id = lease.entry_id
    release = lease.release(reusable=True)
    assert release["released"] is True
    assert release["pooled"] is True

    state_dir = tmp_path / "state" / "lane-0"
    supervisor = PortalImplementationSupervisor(
        PortalSupervisorConfig(
            todo_path=tmp_path / "tasks.md",
            state_path=state_dir / "task_state.json",
            strategy_path=state_dir / "strategy.json",
            events_path=state_dir / "events.jsonl",
            state_dir=state_dir,
            repo_root=repo,
            worktree_root=worktree_root,
        )
    )
    supervisor._list_process_commands = lambda: []  # type: ignore[method-assign]

    result = supervisor.cleanup_backlogged_worktrees()

    idle_skip = next(
        item
        for item in result["skipped"]
        if item["reason"] == "idle_worktree_pool_entry"
    )
    assert idle_skip["path"] == str(idle_path)
    assert idle_skip["owner_source"] == "worktree_pool_lease"
    assert idle_skip["owner_lease_state"] == "idle"
    assert idle_skip["owner_pool_state_path"].endswith(f"{entry_id}.json")
    assert result["removed_count"] == 0
    assert idle_path.exists()

    warm = pool.acquire(
        cache_key="idle-pool-entry",
        base_ref="main",
        branch_name="implementation/reused-idle-pool-entry",
    )
    assert warm.reused is True
    assert warm.path == idle_path
    assert warm.release(reusable=False)["released"] is True
