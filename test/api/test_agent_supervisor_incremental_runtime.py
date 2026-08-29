from __future__ import annotations

import json
import os
import re
import shlex
import stat
import subprocess
import sys
import threading
import time
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.objectives import objective_graph
from ipfs_accelerate_py.agent_supervisor.objectives.backlog_refinery import (
    CodebaseScanInventory,
    scan_codebase_findings,
)
from ipfs_accelerate_py.agent_supervisor.objectives.objective_graph import scan_objective_gaps
from ipfs_accelerate_py.agent_supervisor.task_sources.dataset_store import ObjectiveDatasetStore
from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    worktrees as worktree_helpers,
)
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
    WorktreeLease,
    WorktreePool,
    guarded_worktree_pool_mutation,
    inspect_worktree_pool_missing_release_terminal,
    inspect_worktree_pool_quarantine,
    python_identifier_worktree_basename,
)
from ipfs_accelerate_py.agent_supervisor.worktree_lifecycle import (
    DuplicateAttemptError,
    ProcessBirthIdentity,
    WorkspaceLifecycleState,
    current_process_birth,
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
        implementation_started=False,
        provider_dispatched=False,
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
        _git(repo, "branch", "-D", lease.branch_name)
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


def test_worktree_pool_never_warm_borrows_a_dead_leased_checkout(
    tmp_path: Path,
) -> None:
    """Dead leased state remains reserved for exact recovery or discard."""

    repo = tmp_path / "repo"
    _init_repo(repo)
    (repo / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "seed")
    worktree_root = tmp_path / "pool"
    owner = WorktreePool(repo_root=repo, worktree_root=worktree_root)
    stale = owner.acquire(
        cache_key="shared-cache",
        base_ref="main",
        branch_name="implementation/recoverable",
    )
    state_path = worktree_root / ".pool-state" / f"{stale.entry_id}.json"
    lock_path = worktree_root / ".pool-state" / f"{stale.entry_id}.lock"
    dead_pid = 2_147_483_642
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["lease_pid"] = dead_pid
    state_path.write_text(json.dumps(state), encoding="utf-8")
    state_path.chmod(0o600)
    lock_path.write_text(
        json.dumps({"pid": dead_pid, "created_at_epoch": time.time()}),
        encoding="utf-8",
    )
    lock_path.chmod(0o600)

    contender = WorktreePool(
        repo_root=repo,
        worktree_root=worktree_root,
    ).acquire(
        cache_key="shared-cache",
        base_ref="main",
        branch_name="implementation/contender",
    )

    assert contender.reused is False
    assert contender.path != stale.path
    assert "non_idle_entry_reserved" in contender.invalidation_reasons
    assert stale.path.is_dir()
    assert json.loads(state_path.read_text(encoding="utf-8")) == state
    assert json.loads(lock_path.read_text(encoding="utf-8"))["pid"] == dead_pid
    assert stat.S_IMODE(state_path.stat().st_mode) == 0o600
    assert contender.release(reusable=False)["released"] is True
    assert stale.release(reusable=False)["released"] is True


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

    def accept_current_candidate(
        workspace_path,
        task,
        _log_path,
        *,
        baseline_ref,
        **_kwargs,
    ):
        result = {
            "attempted": True,
            "passed": True,
            "returncode": 0,
            "results": [],
        }
        binding, _entries = daemon._inspect_post_validation_candidate_binding(
            workspace_path,
            task,
            baseline_ref=baseline_ref,
            proposal_validation=None,
        )
        result["candidate_binding"] = {
            **binding,
            "verified": True,
            "expected_fingerprint": binding["current_fingerprint"],
        }
        return result

    monkeypatch.setattr(
        daemon,
        "_run_validation_with_candidate_binding",
        accept_current_candidate,
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


def test_successor_generation_retires_exact_dead_predecessor_claim(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "seed")
    program_root = tmp_path / "program"
    state_dir = program_root / "run-v2" / "state" / "lane-0"
    daemon = PortalImplementationDaemon(
        todo_path=tmp_path / "tasks.md",
        state_path=state_dir / "state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "events.jsonl",
        repo_root=repo,
        implement=True,
        implementation_command=_python_c("raise SystemExit(7)"),
        use_ephemeral_worktree=True,
        worktree_root=tmp_path / "successor-generation-worktrees",
    )
    task = PortalTask(
        task_id="INC-002-DEAD-PREDECESSOR",
        title="Supersede a stopped generation lifecycle owner",
        status="todo",
        completion="manual",
        priority="P1",
        track="runtime",
    )
    canonical_task_cid = daemon._canonical_ref(task)
    predecessor_workspace = (
        tmp_path / "predecessor-generation-worktrees" / "preserved-attempt"
    )
    predecessor_workspace.mkdir(parents=True)
    marker = predecessor_workspace / "preserved.txt"
    marker.write_text("keep predecessor workspace\n", encoding="utf-8")
    predecessor = daemon.worktree_lifecycle.begin_preparing(
        task_id=task.task_id,
        canonical_task_cid=canonical_task_cid,
        attempt=1,
        lane_id="predecessor-generation:lane-0",
        workspace_path=predecessor_workspace,
        branch="implementation/inc-002-dead-predecessor-attempt-1-old",
        merge_target=daemon._main_branch_name(),
        state_dir=str(
            (program_root / "run-v1" / "state" / "lane-0").resolve()
        ),
        owner=ProcessBirthIdentity(
            pid=2**30 - 31,
            start_time_ticks=1,
            boot_id="stopped-predecessor",
        ),
    )
    predecessor_record_path = (
        daemon.worktree_lifecycle.workspace_path_for(predecessor_workspace)
    )
    predecessor_index_path = (
        daemon.worktree_lifecycle.task_index_path_for(
            canonical_task_cid=canonical_task_cid,
            task_id=task.task_id,
            attempt=1,
        )
    )
    record_before = predecessor_record_path.read_bytes()
    index_before = predecessor_index_path.read_bytes()
    successor_state_path = daemon.state_path
    daemon.state_path = (
        program_root / "run-v1" / "state" / "lane-9" / "state.json"
    )

    same_generation = (
        daemon._finalize_dead_predecessor_worktree_lifecycle_claim(
            task=task,
            attempt=1,
        )
    )

    assert same_generation["finalized"] is False
    assert same_generation["reason"] == (
        "task_attempt_claim_identity_mismatch"
    )
    assert same_generation["mismatched_fields"] == ["state_dir_custody"]
    assert predecessor_record_path.read_bytes() == record_before
    assert predecessor_index_path.read_bytes() == index_before
    daemon.state_path = successor_state_path
    actual_proc_root = daemon.worktree_lifecycle.proc_root
    fake_proc_root = tmp_path / "proc"
    (fake_proc_root / "self").mkdir(parents=True)
    (fake_proc_root / "self" / "cgroup").write_text(
        "0::/user.slice/app.slice/ipfs-accelerate-lgcvf-v2.service\n",
        encoding="utf-8",
    )
    unrelated_process = fake_proc_root / "123"
    unrelated_process.mkdir()
    (unrelated_process / "cgroup").write_text(
        "0::/user.slice/session-1.scope\n",
        encoding="utf-8",
    )
    fake_cgroup_root = tmp_path / "cgroup"
    daemon.worktree_lifecycle.proc_root = fake_proc_root
    cgroup_quiescent = daemon._predecessor_generation_cgroup_quiescence(
        predecessor,
        cgroup_root=fake_cgroup_root,
    )

    assert cgroup_quiescent["quiescent"] is True
    assert cgroup_quiescent["predecessor_cgroup"] == (
        "/user.slice/app.slice/ipfs-accelerate-lgcvf-v1.service"
    )
    predecessor_process = fake_proc_root / "456"
    predecessor_process.mkdir()
    (predecessor_process / "cgroup").write_text(
        "0::/user.slice/app.slice/"
        "ipfs-accelerate-lgcvf-v1.service/provider.scope\n",
        encoding="utf-8",
    )
    cgroup_active = daemon._predecessor_generation_cgroup_quiescence(
        predecessor,
        cgroup_root=fake_cgroup_root,
    )

    assert cgroup_active["quiescent"] is False
    assert cgroup_active["reason"] == (
        "generation_cgroup_process_still_active"
    )
    daemon.worktree_lifecycle.proc_root = actual_proc_root
    daemon.worktree_lifecycle.clock = lambda: (
        predecessor.updated_at + 181.0
    )
    survivor = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        cwd=predecessor_workspace,
        start_new_session=True,
    )
    try:
        blocked = (
            daemon._finalize_dead_predecessor_worktree_lifecycle_claim(
                task=task,
                attempt=1,
            )
        )
    finally:
        survivor.terminate()
        try:
            survivor.wait(timeout=5)
        except subprocess.TimeoutExpired:
            survivor.kill()
            survivor.wait(timeout=5)

    assert blocked["finalized"] is False
    assert blocked["reason"] == (
        "task_attempt_claim_worktree_process_still_active"
    )
    assert predecessor_record_path.read_bytes() == record_before
    assert predecessor_index_path.read_bytes() == index_before
    monkeypatch.setattr(
        daemon,
        "_predecessor_worktree_dispatch_quiescence",
        lambda _record: {
            "quiescent": True,
            "reason": "task_attempt_claim_dispatch_quiescent",
        },
    )

    result = daemon._run_implementation(task, PortalTaskState())

    assert result["returncode"] == 7
    assert result.get("reason") != "worktree_lifecycle_claim_exists"
    assert marker.read_text(encoding="utf-8") == (
        "keep predecessor workspace\n"
    )
    assert not predecessor_record_path.exists()
    assert not predecessor_index_path.exists()
    events = [
        json.loads(line)
        for line in daemon.events_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    finalized = next(
        event
        for event in events
        if event.get("type")
        == "worktree_lifecycle_dead_predecessor_claim_finalized"
    )
    assert finalized["record_id"] == predecessor.record_id
    assert finalized["predecessor_owner_liveness"] == "dead"
    assert finalized["reason"] == (
        "successor_generation_dead_owner_superseded"
    )


def test_sibling_portal_attempt_retires_only_exact_dead_same_lane_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "seed")
    state_root = tmp_path / "state"
    lane_root = state_root / "lane-1" / "spar_lane_1_database_portal_attempts"
    current_state_dir = lane_root / ("a" * 24)
    predecessor_state_dir = lane_root / ("b" * 24)
    foreign_lane_state_dir = (
        state_root
        / "lane-2"
        / "spar_lane_2_database_portal_attempts"
        / ("c" * 24)
    )
    for path in (current_state_dir, predecessor_state_dir, foreign_lane_state_dir):
        path.mkdir(parents=True)
    daemon = PortalImplementationDaemon(
        todo_path=tmp_path / "tasks.md",
        state_path=current_state_dir / "portal-task-state.json",
        strategy_path=current_state_dir / "portal-strategy.json",
        events_path=current_state_dir / "portal-events.jsonl",
        repo_root=repo,
        implement=True,
        implementation_command=_python_c("raise SystemExit(7)"),
        use_ephemeral_worktree=True,
        worktree_root=tmp_path / "worktrees",
    )
    task = PortalTask(
        task_id="INC-002-DEAD-PORTAL-SIBLING",
        title="Supersede an exact dead sibling Portal attempt",
        status="todo",
        completion="manual",
        priority="P1",
        track="runtime",
    )
    workspace = tmp_path / "preserved-worktree"
    workspace.mkdir()
    marker = workspace / "preserved.txt"
    marker.write_text("preserve\n", encoding="utf-8")
    predecessor = daemon.worktree_lifecycle.begin_preparing(
        task_id=task.task_id,
        canonical_task_cid=daemon._canonical_ref(task),
        attempt=1,
        lane_id="portal-attempt:lane-1",
        workspace_path=workspace,
        branch="implementation/inc-002-dead-portal-sibling-attempt-1",
        merge_target=daemon._main_branch_name(),
        state_dir=str(predecessor_state_dir.resolve()),
        owner=ProcessBirthIdentity(
            pid=2**30 - 41,
            start_time_ticks=1,
            boot_id="dead-portal-sibling",
        ),
    )
    live_workspace = tmp_path / "live-preserved-worktree"
    live_workspace.mkdir()
    live_predecessor = daemon.worktree_lifecycle.begin_preparing(
        task_id=task.task_id,
        canonical_task_cid=daemon._canonical_ref(task),
        attempt=2,
        lane_id="portal-attempt:lane-1",
        workspace_path=live_workspace,
        branch="implementation/inc-002-live-portal-sibling-attempt-2",
        merge_target=daemon._main_branch_name(),
        state_dir=str(predecessor_state_dir.resolve()),
    )
    monkeypatch.setattr(
        daemon,
        "_predecessor_worktree_dispatch_quiescence",
        lambda _record: {
            "quiescent": True,
            "reason": "task_attempt_claim_dispatch_quiescent",
        },
    )

    symlink_target = state_root / "symlink-target"
    (symlink_target / ("d" * 24)).mkdir(parents=True)
    (symlink_target / ("e" * 24)).mkdir(parents=True)
    symlink_attempt_root = (
        state_root
        / "lane-1"
        / "shadow_lane_1_database_portal_attempts"
    )
    symlink_attempt_root.symlink_to(symlink_target, target_is_directory=True)
    assert (
        daemon._state_dir_portal_attempt_custody_binding(
            symlink_attempt_root / ("d" * 24),
            symlink_attempt_root / ("e" * 24),
        )
        is None
    )
    regular_file_leaf = lane_root / ("f" * 24)
    regular_file_leaf.write_text("not a state directory\n", encoding="utf-8")
    assert (
        daemon._state_dir_portal_attempt_custody_binding(
            current_state_dir,
            regular_file_leaf,
        )
        is None
    )

    exact_state_path = daemon.state_path
    daemon.state_path = foreign_lane_state_dir / "portal-task-state.json"
    foreign = daemon._finalize_dead_predecessor_worktree_lifecycle_claim(
        task=task,
        attempt=1,
    )
    assert foreign["finalized"] is False
    assert foreign["reason"] == "task_attempt_claim_identity_mismatch"
    assert foreign["mismatched_fields"] == ["state_dir_custody"]
    assert daemon.worktree_lifecycle.load_workspace(workspace) == predecessor

    daemon.state_path = exact_state_path
    live = daemon._finalize_dead_predecessor_worktree_lifecycle_claim(
        task=task,
        attempt=2,
    )
    assert live["finalized"] is False
    assert live["reason"] == "task_attempt_claim_owner_alive"
    assert live["custody_kind"] == "sibling_database_portal_attempt"
    assert daemon.worktree_lifecycle.load_workspace(live_workspace) == live_predecessor

    monkeypatch.setattr(
        daemon,
        "_predecessor_worktree_dispatch_quiescence",
        lambda _record: {
            "quiescent": False,
            "reason": "task_attempt_claim_provider_descendant_active",
        },
    )
    non_quiescent = daemon._finalize_dead_predecessor_worktree_lifecycle_claim(
        task=task,
        attempt=1,
    )
    assert non_quiescent["finalized"] is False
    assert non_quiescent["reason"] == (
        "task_attempt_claim_provider_descendant_active"
    )
    assert daemon.worktree_lifecycle.load_workspace(workspace) == predecessor

    monkeypatch.setattr(
        daemon,
        "_predecessor_worktree_dispatch_quiescence",
        lambda _record: {
            "quiescent": True,
            "reason": "task_attempt_claim_dispatch_quiescent",
        },
    )
    recovered = daemon._finalize_dead_predecessor_worktree_lifecycle_claim(
        task=task,
        attempt=1,
    )

    assert recovered["finalized"] is True
    assert recovered["custody_kind"] == "sibling_database_portal_attempt"
    assert recovered["reason"] == "sibling_portal_attempt_dead_owner_superseded"
    assert recovered["predecessor_owner_liveness"] == "dead"
    assert recovered["portal_attempt_custody"]["current_lane"] == 1
    assert recovered["portal_attempt_custody"]["source_lane"] == 1
    assert marker.read_text(encoding="utf-8") == "preserve\n"
    assert daemon.worktree_lifecycle.load_workspace(workspace) is None


def test_sibling_portal_attempt_quiescence_scopes_denied_proc_cwds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Supervisor construction binds the board's default Grok model into the
    # process environment. Register the pre-test value with monkeypatch so the
    # fixture cannot leak that route fragment into later raw-command tests.
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_GROK_MODEL", "")
    repo = tmp_path / "repo"
    _init_repo(repo)
    todo_path = repo / "tasks.md"
    todo_path.write_text("# Taskboard\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "seed")
    board = "semantic-preserving-autonomous-remodularization-v1"
    target = "codex/semantic-preserving-autonomous-remodularization-v1"
    _git(repo, "branch", target)
    state_root = tmp_path / "state"
    attempt_root = state_root / "lane-1" / "spar_lane_1_database_portal_attempts"
    current_state_dir = attempt_root / ("a" * 24)
    predecessor_state_dir = attempt_root / ("b" * 24)
    current_state_dir.mkdir(parents=True)
    predecessor_state_dir.mkdir(parents=True)
    worktree_root = tmp_path / "worktrees"
    workspace = worktree_root / "workspace_aaaaaaaaaaaa_bbbbbbbbbbbb"
    worktree_root.mkdir(parents=True)
    implementation_branch = (
        "implementation/inc-002-dead-portal-proc-attempt-1"
    )
    _git(
        repo,
        "worktree",
        "add",
        "-b",
        implementation_branch,
        str(workspace),
        "HEAD",
    )
    daemon = PortalImplementationDaemon(
        todo_path=todo_path,
        state_path=current_state_dir / "portal-task-state.json",
        strategy_path=current_state_dir / "portal-strategy.json",
        events_path=current_state_dir / "portal-events.jsonl",
        repo_root=repo,
        implement=True,
        implementation_command=_python_c("raise SystemExit(7)"),
        use_ephemeral_worktree=True,
        worktree_root=worktree_root,
        board_namespace=board,
        merge_target_branch=target,
    )
    task = PortalTask(
        task_id="INC-002-DEAD-PORTAL-PROC",
        title="Quarantine an exact denied-cwd dead lease",
        status="todo",
        completion="manual",
        priority="P1",
        track="runtime",
    )
    dead_owner = ProcessBirthIdentity(
        pid=900,
        start_time_ticks=1_000,
        boot_id="dead-portal-proc-owner",
        parent_pid=600,
    )
    record = daemon.worktree_lifecycle.begin_preparing(
        task_id=task.task_id,
        canonical_task_cid=daemon._canonical_ref(task),
        attempt=1,
        lane_id="portal-attempt:lane-1",
        workspace_path=workspace,
        branch=implementation_branch,
        merge_target=daemon._main_branch_name(),
        state_dir=str(predecessor_state_dir.resolve()),
        owner=dead_owner,
    )
    daemon.worktree_lifecycle.clock = lambda: record.updated_at + 1_000.0

    implementation_lock = {
        "attempt": record.attempt,
        "board_namespace": daemon.board_namespace,
        "canonical_task_cid": record.canonical_task_cid,
        "canonical_task_key": record.canonical_task_cid,
        "kind": "implementation",
        "lease_id": "implementation-lease",
        "owner_process_birth": dead_owner.to_dict(),
        "owner_script": "implementation_daemon.py",
        "pid": dead_owner.pid,
        "repo_root": str(repo.resolve()),
        "started_at": "2026-08-29T07:31:25+00:00",
        "state_dir": str(predecessor_state_dir.resolve()),
        "task_id": record.task_id,
    }
    attempt_lock_path = predecessor_state_dir / "implementation.lock"
    attempt_lock_path.write_text(json.dumps(implementation_lock), encoding="utf-8")
    pool_root = worktree_root / ".pool-state"
    pool_root.mkdir()
    entry_id = "aaaaaaaaaaaa-bbbbbbbbbbbb"
    pool_state_path = pool_root / f"{entry_id}.json"
    pool_lock_path = pool_root / f"{entry_id}.lock"
    pool_state = {
        "base_commit": _git(repo, "rev-parse", "HEAD"),
        "branch": record.branch,
        "cache_key": "cache-key",
        "cold_setup_seconds": 1.0,
        "created_at_epoch": 1.0,
        "dependency_heads": {},
        "dependency_paths": [],
        "last_used_at_epoch": 1.0,
        "lease_pid": dead_owner.pid,
        "lease_token": entry_id,
        "path": str(workspace.resolve()),
        "repo_common_dir": str(daemon.worktree_pool.repo_common_dir),
        "repo_root": str(repo.resolve()),
        "schema": "agent-supervisor-worktree-pool-v1",
        "state": "leased",
        "use_count": 1,
    }
    pool_lock = {"pid": dead_owner.pid, "created_at_epoch": 1.0}
    pool_state_bytes = json.dumps(pool_state).encode()
    pool_lock_bytes = json.dumps(pool_lock).encode()
    pool_state_path.write_bytes(pool_state_bytes)
    pool_lock_path.write_bytes(pool_lock_bytes)

    proc_root = tmp_path / "proc"
    denied_process = proc_root / "500"
    denied_process.mkdir(parents=True)
    (denied_process / "cmdline").write_bytes(
        b"/usr/bin/tmux\0long-lived-unrelated-process\0"
    )
    (denied_process / "cwd").symlink_to(tmp_path)
    daemon.worktree_lifecycle.proc_root = proc_root
    real_readlink = os.readlink

    def yama_denied_readlink(path: str | os.PathLike[str]) -> str:
        candidate = Path(path)
        if candidate == denied_process / "cwd":
            raise PermissionError("simulated Yama ptrace_scope denial")
        return real_readlink(path)

    monkeypatch.setattr(os, "readlink", yama_denied_readlink)
    real_run = subprocess.run

    def no_containers(command: object, *args: object, **kwargs: object) -> object:
        if isinstance(command, (list, tuple)) and command and command[0] == "docker":
            return subprocess.CompletedProcess(command, 0, "", "")
        return real_run(command, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", no_containers)

    pool_state["lease_pid"] = dead_owner.pid + 1
    pool_state_path.write_text(json.dumps(pool_state), encoding="utf-8")
    missing_pool = daemon._predecessor_worktree_dispatch_quiescence(record)
    assert missing_pool["quiescent"] is False
    assert missing_pool["reason"] == "portal_attempt_pool_proof_unavailable"
    pool_state["lease_pid"] = dead_owner.pid
    pool_state_path.write_bytes(pool_state_bytes)

    real_attempt_lock = tmp_path / "implementation.lock.json"
    real_attempt_lock.write_text(json.dumps(implementation_lock), encoding="utf-8")
    attempt_lock_path.unlink()
    attempt_lock_path.symlink_to(real_attempt_lock)
    symlinked_lock = daemon._predecessor_worktree_dispatch_quiescence(record)
    assert symlinked_lock["quiescent"] is False
    assert symlinked_lock["reason"] == "portal_attempt_lock_proof_unavailable"
    attempt_lock_path.unlink()
    attempt_lock_path.write_text(json.dumps(implementation_lock), encoding="utf-8")

    real_pool_state = tmp_path / "pool-state.json"
    real_pool_state.write_bytes(pool_state_bytes)
    pool_state_path.unlink()
    pool_state_path.symlink_to(real_pool_state)
    symlinked_pool = daemon._predecessor_worktree_dispatch_quiescence(record)
    assert symlinked_pool["quiescent"] is False
    assert symlinked_pool["reason"] == "portal_attempt_pool_proof_unavailable"
    pool_state_path.unlink()
    pool_state_path.write_bytes(pool_state_bytes)

    quarantine_root = pool_root / "quarantine"
    genuinely_absent = inspect_worktree_pool_quarantine(
        worktree_root=worktree_root,
        workspace_path=workspace,
        expected_branch=record.branch,
    )
    assert genuinely_absent["status"] == "absent"
    assert genuinely_absent["cleanup_fenced"] is False

    empty_quarantine_target = tmp_path / "empty-quarantine-target"
    empty_quarantine_target.mkdir()
    quarantine_root.symlink_to(
        empty_quarantine_target,
        target_is_directory=True,
    )
    symlinked_quarantine_root = inspect_worktree_pool_quarantine(
        worktree_root=worktree_root,
        workspace_path=workspace,
        expected_branch=record.branch,
    )
    assert symlinked_quarantine_root["status"] == "invalid"
    assert symlinked_quarantine_root["cleanup_fenced"] is True
    assert symlinked_quarantine_root["reason"] == "quarantine_directory_unsafe"
    quarantine_root.unlink()

    quarantine_root.write_text("not a directory\n", encoding="utf-8")
    file_quarantine_root = inspect_worktree_pool_quarantine(
        worktree_root=worktree_root,
        workspace_path=workspace,
        expected_branch=record.branch,
    )
    assert file_quarantine_root["status"] == "invalid"
    assert file_quarantine_root["cleanup_fenced"] is True
    assert file_quarantine_root["reason"] == "quarantine_directory_unsafe"
    quarantine_root.unlink()

    real_lstat = Path.lstat

    def uninspectable_quarantine_root(path: Path) -> os.stat_result:
        if path == quarantine_root:
            raise PermissionError("simulated quarantine-root lstat denial")
        return real_lstat(path)

    monkeypatch.setattr(Path, "lstat", uninspectable_quarantine_root)
    uninspectable_root = inspect_worktree_pool_quarantine(
        worktree_root=worktree_root,
        workspace_path=workspace,
        expected_branch=record.branch,
    )
    assert uninspectable_root["status"] == "invalid"
    assert uninspectable_root["cleanup_fenced"] is True
    assert uninspectable_root["reason"] == (
        "quarantine_directory_uninspectable"
    )
    monkeypatch.setattr(Path, "lstat", real_lstat)

    (denied_process / "cmdline").write_bytes(
        b"/usr/bin/worker\0" + record.branch.encode() + b"\0"
    )
    branch_active = daemon._predecessor_worktree_dispatch_quiescence(record)
    assert branch_active["quiescent"] is False
    assert branch_active["reason"] == "task_attempt_claim_worktree_process_still_active"
    (denied_process / "cmdline").write_bytes(
        b"/usr/bin/worker\0" + str(workspace).encode() + b"\0"
    )
    workspace_active = daemon._predecessor_worktree_dispatch_quiescence(record)
    assert workspace_active["quiescent"] is False
    assert workspace_active["reason"] == "task_attempt_claim_worktree_process_still_active"
    (denied_process / "cmdline").write_bytes(
        b"/usr/bin/tmux\0long-lived-unrelated-process\0"
    )

    # Publication and mutation share the exact per-entry update guard. Pause
    # publication at its no-replace link and prove a different thread cannot
    # enter the mutation body until the marker is durable; once admitted, it
    # must observe the marker and refuse the write.
    scope_proof = daemon._predecessor_portal_attempt_scope_proof(record)
    assert scope_proof["valid"] is True
    marker_path = pool_root / "quarantine" / f"{entry_id}.json"
    real_link = os.link
    real_fsync = os.fsync
    fsynced_paths: list[str] = []

    def tracing_fsync(descriptor: int) -> None:
        try:
            fsynced_paths.append(os.readlink(f"/proc/self/fd/{descriptor}"))
        except OSError:
            fsynced_paths.append("<uninspectable>")
        real_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", tracing_fsync)

    # Fault-inject the crash seam after the exact native Git worktree lock but
    # before the immutable marker link. The pending lock must independently
    # fence every ordinary mutation, and retry must derive the same quarantine
    # identity rather than treating its own prior lock as foreign custody.
    def fail_first_marker_link(
        source: str | os.PathLike[str],
        destination: str | os.PathLike[str],
        *args: object,
        **kwargs: object,
    ) -> None:
        if Path(destination) == marker_path:
            raise OSError("simulated crash after native worktree lock")
        real_link(source, destination, *args, **kwargs)

    monkeypatch.setattr(os, "link", fail_first_marker_link)
    interrupted_publication = daemon._portal_attempt_denied_cwd_quiescence(
        record,
        denied_pids=[500],
        scope_proof=scope_proof,
    )
    assert interrupted_publication["quiescent"] is False
    assert interrupted_publication["reason"] == (
        "portal_attempt_quarantine_unproven"
    )
    assert marker_path.exists() is False
    pending_registration, pending_reason = (
        worktree_helpers._git_worktree_registration(repo, workspace)
    )
    assert pending_reason == "git_worktree_registration_exact"
    assert pending_registration is not None
    pending_lock_reason = str(pending_registration.get("locked") or "")
    assert re.fullmatch(
        r"agent-supervisor-quarantine-v1:sha256:[0-9a-f]{64}",
        pending_lock_reason,
    )
    with guarded_worktree_pool_mutation(
        repo_root=repo,
        worktree_root=worktree_root,
        workspace_path=workspace,
        expected_branch=record.branch,
        operation="test_pending_native_quarantine_lock",
    ) as pending_admission:
        assert pending_admission["allowed"] is False
        assert pending_admission["reason"] == (
            "pending_worktree_pool_quarantine"
        )

    link_entered = threading.Event()
    release_link = threading.Event()
    mutation_finished = threading.Event()
    mutation_sentinel = workspace / "mutation-must-not-run.txt"
    publisher_result: dict[str, object] = {}
    mutation_result: dict[str, object] = {}
    thread_errors: list[BaseException] = []

    def blocking_marker_link(
        source: str | os.PathLike[str],
        destination: str | os.PathLike[str],
        *args: object,
        **kwargs: object,
    ) -> None:
        if Path(destination) == marker_path:
            link_entered.set()
            assert release_link.wait(timeout=5.0)
        real_link(source, destination, *args, **kwargs)

    monkeypatch.setattr(os, "link", blocking_marker_link)

    def publish_quarantine() -> None:
        try:
            publisher_result.update(
                daemon._portal_attempt_denied_cwd_quiescence(
                    record,
                    denied_pids=[500],
                    scope_proof=scope_proof,
                )
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            thread_errors.append(exc)

    def mutate_workspace() -> None:
        try:
            with guarded_worktree_pool_mutation(
                repo_root=repo,
                worktree_root=worktree_root,
                workspace_path=workspace,
                expected_branch=record.branch,
                operation="test_publication_mutation_race",
            ) as admission:
                mutation_result.update(admission)
                if admission.get("allowed") is True:
                    mutation_sentinel.write_text("unsafe\n", encoding="utf-8")
        except BaseException as exc:  # pragma: no cover - asserted below
            thread_errors.append(exc)
        finally:
            mutation_finished.set()

    publisher = threading.Thread(target=publish_quarantine)
    publisher.start()
    assert link_entered.wait(timeout=5.0)
    mutator = threading.Thread(target=mutate_workspace)
    mutator.start()
    assert mutation_finished.wait(timeout=0.2) is False
    release_link.set()
    publisher.join(timeout=5.0)
    mutator.join(timeout=5.0)
    assert not publisher.is_alive()
    assert not mutator.is_alive()
    assert thread_errors == []
    assert publisher_result["quiescent"] is True
    assert mutation_result["allowed"] is False
    assert mutation_result["reason"] == "durable_worktree_pool_quarantine"
    assert not mutation_sentinel.exists()
    assert str(pool_root.resolve()) in fsynced_paths
    assert str((pool_root / "quarantine").resolve()) in fsynced_paths

    quiescent = daemon._predecessor_worktree_dispatch_quiescence(record)
    assert quiescent["quiescent"] is True
    quarantine_proof = quiescent["generation_cgroup_proof"]
    assert quarantine_proof["reason"] == "portal_attempt_workspace_durably_quarantined"
    assert quarantine_proof["denied_process_count"] == 1
    assert quarantine_proof["quarantine"]["published"] is True
    assert quarantine_proof["quarantine"]["valid"] is True
    assert quarantine_proof["quarantine"]["cleanup_fenced"] is True

    marker_bytes = marker_path.read_bytes()
    marker = json.loads(marker_bytes)
    assert marker["owner_process_birth"] == dead_owner.to_dict()
    assert marker["lifecycle_record_id"] == record.record_id
    assert marker["lifecycle_fence"] == record.fence
    assert marker["lifecycle_lease_id"] == record.lease_id
    assert marker["task_id"] == record.task_id
    assert marker["canonical_task_cid"] == record.canonical_task_cid
    assert marker["attempt"] == record.attempt
    assert marker["branch"] == record.branch
    assert marker["workspace_path"] == str(workspace.resolve())
    assert marker["predecessor_state_dir"] == str(predecessor_state_dir.resolve())
    assert marker["current_state_dir"] == str(current_state_dir.resolve())
    assert marker["git_worktree_lock_reason"] == pending_lock_reason

    # The native Git lock is an independent cross-authority fence. A normal
    # one-force removal fails, while repository-global prune and GC continue
    # to run successfully without discarding the quarantined registration.
    one_force_remove = subprocess.run(
        ["git", "worktree", "remove", "--force", str(workspace)],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    assert one_force_remove.returncode != 0
    assert workspace.is_dir()
    _git(repo, "worktree", "prune", "--expire", "now")
    _git(repo, "-c", "gc.worktreePruneExpire=now", "gc")
    retained_registration, retained_reason = (
        worktree_helpers._git_worktree_registration(repo, workspace)
    )
    assert retained_reason == "git_worktree_registration_exact"
    assert retained_registration is not None
    assert retained_registration.get("locked") == pending_lock_reason

    inspected = inspect_worktree_pool_quarantine(
        worktree_root=worktree_root,
        workspace_path=workspace,
        expected_branch=record.branch,
    )
    assert inspected["status"] == "valid"
    assert inspected["valid"] is True
    assert inspected["cleanup_fenced"] is True

    # Simulate a crash after marker publication but before lifecycle CAS.
    retried = daemon._predecessor_worktree_dispatch_quiescence(record)
    assert retried["quiescent"] is True
    assert retried["generation_cgroup_proof"]["quarantine"]["idempotent"] is True
    assert marker_path.read_bytes() == marker_bytes
    assert pool_state_path.read_bytes() == pool_state_bytes
    assert pool_lock_path.read_bytes() == pool_lock_bytes

    cleanup_denied = daemon._authorize_worktree_cleanup(workspace, record.branch)
    assert cleanup_denied["allowed"] is False
    assert cleanup_denied["reason"] == "durable_worktree_pool_quarantine"

    marker_path.write_text("{", encoding="utf-8")
    malformed = inspect_worktree_pool_quarantine(
        worktree_root=worktree_root,
        workspace_path=workspace,
        expected_branch=record.branch,
    )
    assert malformed["status"] == "invalid"
    assert malformed["cleanup_fenced"] is True
    malformed_cleanup = daemon._authorize_worktree_cleanup(workspace, record.branch)
    assert malformed_cleanup["allowed"] is False
    assert malformed_cleanup["reason"] == "worktree_pool_quarantine_unverifiable"
    marker_path.write_bytes(marker_bytes)

    symlink_target = tmp_path / "quarantine-marker.json"
    symlink_target.write_bytes(marker_bytes)
    marker_path.unlink()
    marker_path.symlink_to(symlink_target)
    symlinked_marker = inspect_worktree_pool_quarantine(
        worktree_root=worktree_root,
        workspace_path=workspace,
        expected_branch=record.branch,
    )
    assert symlinked_marker["status"] == "invalid"
    assert symlinked_marker["cleanup_fenced"] is True
    marker_path.unlink()
    marker_path.write_bytes(marker_bytes)

    nonfinite_state = dict(pool_state)
    nonfinite_state["cold_setup_seconds"] = float("nan")
    pool_state_path.write_text(json.dumps(nonfinite_state), encoding="utf-8")
    nonfinite = inspect_worktree_pool_quarantine(
        worktree_root=worktree_root,
        workspace_path=workspace,
        expected_branch=record.branch,
    )
    assert nonfinite["status"] == "invalid"
    assert nonfinite["cleanup_fenced"] is True
    assert nonfinite["reason"] == "quarantine_pool_binding_mismatch"
    pool_state_path.write_bytes(pool_state_bytes)

    recovered = daemon._finalize_dead_predecessor_worktree_lifecycle_claim(
        task=task,
        attempt=1,
    )
    assert recovered["finalized"] is True
    assert recovered["reason"] == "sibling_portal_attempt_dead_owner_superseded"
    assert daemon.worktree_lifecycle.load_workspace(workspace) is None
    assert workspace.is_dir()
    assert marker_path.read_bytes() == marker_bytes
    assert json.loads(pool_state_path.read_text())["state"] == "leased"

    supervisor_state = state_root / "lane-2"
    supervisor = PortalImplementationSupervisor(
        PortalSupervisorConfig(
            todo_path=todo_path,
            state_path=supervisor_state / "task_state.json",
            strategy_path=supervisor_state / "strategy.json",
            events_path=supervisor_state / "events.jsonl",
            state_dir=supervisor_state,
            repo_root=repo,
            worktree_root=worktree_root,
            merge_target_branch=target,
        )
    )
    owners = supervisor._shared_active_worktree_owners(worktree_root)
    assert owners[workspace.resolve()]["source"] == "worktree_pool_quarantine"
    assert owners[workspace.resolve()]["quarantine_status"] == "valid"

    # A valid marker blocks both stale pool-lock reclamation and the direct
    # stale-active dirty rescue path. The supervisor must retain the active
    # execution record so a later exact recovery pass can reason from it.
    assert daemon.worktree_pool._try_claim(pool_state) is None  # noqa: SLF001
    assert pool_lock_path.read_bytes() == pool_lock_bytes
    stale_state = PortalTaskState.load(supervisor.config.state_path)
    stale_state.active_task_id = task.task_id
    stale_state.active_task_title = task.title
    stale_state.active_task_track = task.track
    stale_state.active_task_started_at = "2026-08-29T07:31:25Z"
    stale_state.active_attempt = 1
    stale_state.active_phase = "implementation"
    stale_state.active_phase_started_at = "2026-08-29T07:31:25Z"
    stale_state.active_worktree_path = str(workspace)
    stale_state.active_branch = record.branch
    stale_state.implementation_in_progress = True
    stale_state.save(supervisor.config.state_path)
    monkeypatch.setattr(supervisor, "_read_managed_daemon_pid", lambda: None)
    monkeypatch.setattr(supervisor, "_list_process_commands", lambda: [])
    monkeypatch.setattr(
        supervisor,
        "_git_status_short",
        lambda _path: ["?? preserved-dirty.py"],
    )
    stale_repair = supervisor.repair_stale_active_execution_state()
    assert stale_repair["repaired"] is False
    assert stale_repair["reason"] == "durable_worktree_pool_quarantine"
    retained_state = PortalTaskState.load(supervisor.config.state_path)
    assert retained_state.implementation_in_progress is True
    assert retained_state.active_worktree_path == str(workspace)
    assert retained_state.active_branch == record.branch

    # Per-candidate cleanup does not depend on a readable pool-state record.
    pool_state_path.unlink()
    supervisor._git_worktree_records = lambda _repo: [  # type: ignore[method-assign]
        {
            "worktree": str(workspace),
            "branch": f"refs/heads/{record.branch}",
            "HEAD": pool_state["base_commit"],
        }
    ]
    supervisor._list_process_commands = lambda: []  # type: ignore[method-assign]
    monkeypatch.setattr(
        supervisor,
        "_rescue_dirty_worktree",
        lambda *args, **kwargs: pytest.fail(
            "quarantined worktree must not be rescued"
        ),
    )
    monkeypatch.setattr(
        supervisor,
        "_prune_completed_leftover_worktree",
        lambda *args, **kwargs: pytest.fail(
            "quarantined worktree must not be pruned"
        ),
    )
    monkeypatch.setattr(
        supervisor,
        "_preflight_worktree_reconciliation_merge",
        lambda *args, **kwargs: pytest.fail(
            "quarantined worktree must not enter merge preflight"
        ),
    )
    workspace_before = {
        path.relative_to(workspace).as_posix(): path.read_bytes()
        for path in workspace.rglob("*")
        if path.is_file()
    }
    refs_before = _git(repo, "show-ref")

    detected = supervisor.detect_stale_worktrees()
    detect_skip = next(
        item for item in detected["skipped"] if item["path"] == str(workspace)
    )
    assert detect_skip["reason"] == "worktree_pool_quarantine_unverifiable"
    reconciled = supervisor.reconcile_backlogged_worktrees()
    reconcile_skip = next(
        item for item in reconciled["skipped"] if item["path"] == str(workspace)
    )
    assert reconcile_skip["reason"] == (
        "worktree_pool_quarantine_unverifiable"
    )

    cleanup = supervisor._cleanup_backlogged_worktrees_locked()
    assert cleanup["prune_returncode"] == 0
    quarantine_skip = next(
        item for item in cleanup["skipped"] if item["path"] == str(workspace)
    )
    assert quarantine_skip["reason"] == "worktree_pool_quarantine_unverifiable"
    assert quarantine_skip["quarantine_status"] == "invalid"
    assert workspace.is_dir()
    assert marker_path.read_bytes() == marker_bytes
    assert refs_before == _git(repo, "show-ref")
    assert workspace_before == {
        path.relative_to(workspace).as_posix(): path.read_bytes()
        for path in workspace.rglob("*")
        if path.is_file()
    }

    pool_state_path.write_text("{", encoding="utf-8")
    corrupt_detected = supervisor.detect_stale_worktrees()
    corrupt_detect_skip = next(
        item
        for item in corrupt_detected["skipped"]
        if item["path"] == str(workspace)
    )
    assert corrupt_detect_skip["reason"] == (
        "worktree_pool_quarantine_unverifiable"
    )
    corrupt_reconciled = supervisor.reconcile_backlogged_worktrees()
    corrupt_reconcile_skip = next(
        item
        for item in corrupt_reconciled["skipped"]
        if item["path"] == str(workspace)
    )
    assert corrupt_reconcile_skip["reason"] == (
        "worktree_pool_quarantine_unverifiable"
    )
    assert marker_path.read_bytes() == marker_bytes
    assert refs_before == _git(repo, "show-ref")
    assert workspace_before == {
        path.relative_to(workspace).as_posix(): path.read_bytes()
        for path in workspace.rglob("*")
        if path.is_file()
    }

    # Marker lookup is lexical under the exact root. Replacing the workspace
    # with a symlink outside that root must expose an invalid cleanup fence,
    # never hide the surviving marker as a non-pooled path.
    pool_state_path.write_bytes(pool_state_bytes)
    outside_workspace = tmp_path / "outside-workspace"
    workspace.rename(outside_workspace)
    outside_sentinel = outside_workspace / "preserve.txt"
    outside_sentinel.write_text("preserve\n", encoding="utf-8")
    workspace.symlink_to(outside_workspace, target_is_directory=True)
    drifted = inspect_worktree_pool_quarantine(
        worktree_root=worktree_root,
        workspace_path=workspace,
        expected_branch=record.branch,
    )
    assert drifted["status"] == "invalid"
    assert drifted["cleanup_fenced"] is True
    assert drifted["reason"] == "quarantine_workspace_binding_mismatch"

    drift_detected = supervisor.detect_stale_worktrees()
    drift_detect_skip = next(
        item
        for item in drift_detected["skipped"]
        if item["path"] == str(workspace)
    )
    assert drift_detect_skip["reason"] == (
        "worktree_pool_quarantine_unverifiable"
    )
    drift_reconciled = supervisor.reconcile_backlogged_worktrees()
    drift_reconcile_skip = next(
        item
        for item in drift_reconciled["skipped"]
        if item["path"] == str(workspace)
    )
    assert drift_reconcile_skip["reason"] == (
        "worktree_pool_quarantine_unverifiable"
    )
    drift_cleanup = supervisor._cleanup_backlogged_worktrees_locked()
    drift_cleanup_skip = next(
        item
        for item in drift_cleanup["skipped"]
        if item["path"] == str(workspace)
    )
    assert drift_cleanup_skip["reason"] == (
        "worktree_pool_quarantine_unverifiable"
    )
    assert workspace.is_symlink()
    assert outside_sentinel.read_text(encoding="utf-8") == "preserve\n"
    assert marker_path.read_bytes() == marker_bytes


def test_pool_mutation_guard_is_reentrant_and_revalidates_waiting_preimage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Supervisor construction may bind the default Grok model into the process
    # environment.  Register an explicit pre-test value so this fixture cannot
    # leak a partial provider route into later sealed-route tests.
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_GROK_MODEL", "")
    repo = tmp_path / "repo"
    _init_repo(repo)
    (repo / "seed.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "seed")
    todo_path = repo / "tasks.md"
    todo_path.write_text("# Taskboard\n", encoding="utf-8")
    worktree_root = tmp_path / "worktrees"
    worktree_root.mkdir()
    workspace = worktree_root / "workspace_111111111111_222222222222"
    branch = "implementation/reentrant-quarantine-guard"
    _git(repo, "worktree", "add", "-b", branch, str(workspace), "HEAD")
    (worktree_root / ".pool-state").mkdir()

    supervisor_state = tmp_path / "supervisor-state"
    supervisor = PortalImplementationSupervisor(
        PortalSupervisorConfig(
            todo_path=todo_path,
            state_path=supervisor_state / "state.json",
            strategy_path=supervisor_state / "strategy.json",
            events_path=supervisor_state / "events.jsonl",
            state_dir=supervisor_state,
            repo_root=repo,
            worktree_root=worktree_root,
        )
    )
    daemon_state = tmp_path / "daemon-state"
    daemon = PortalImplementationDaemon(
        todo_path=todo_path,
        state_path=daemon_state / "state.json",
        strategy_path=daemon_state / "strategy.json",
        events_path=daemon_state / "events.jsonl",
        repo_root=repo,
        implement=True,
        implementation_command=_python_c("raise SystemExit(7)"),
        use_ephemeral_worktree=True,
        worktree_root=worktree_root,
    )
    monkeypatch.setattr(supervisor, "_list_process_commands", lambda: [])
    monkeypatch.setattr(daemon, "_list_process_commands", lambda: [])
    monkeypatch.setattr(
        supervisor,
        "_strict_process_commands_for_mutation",
        lambda: {
            "available": True,
            "reason": "test_process_query_current",
            "commands": [],
        },
    )
    monkeypatch.setattr(
        daemon,
        "_strict_process_commands_for_mutation",
        lambda: {
            "available": True,
            "reason": "test_process_query_current",
            "commands": [],
        },
    )
    monkeypatch.setattr(
        daemon,
        "_cleanup_merged_worktree_guarded",
        lambda *_args, **_kwargs: {
            "cleaned": True,
            "reason": "nested_cleanup_callback_completed",
        },
    )

    nested_result: dict[str, object] = {}
    nested_errors: list[BaseException] = []

    def nested_supervisor_daemon_callback() -> None:
        try:
            with supervisor._pooled_worktree_mutation_guard(  # noqa: SLF001
                workspace,
                expected_branch=branch,
                operation="test_supervisor_reconciliation_outer_guard",
            ) as outer:
                nested_result["outer"] = outer
                nested_result["cleanup"] = daemon._cleanup_merged_worktree(  # noqa: SLF001
                    workspace,
                    branch,
                )
        except BaseException as exc:  # pragma: no cover - asserted below
            nested_errors.append(exc)

    nested_thread = threading.Thread(target=nested_supervisor_daemon_callback)
    nested_thread.start()
    nested_thread.join(timeout=5.0)
    assert not nested_thread.is_alive(), "nested exact guard self-deadlocked"
    assert nested_errors == []
    assert nested_result["outer"]["allowed"] is True  # type: ignore[index]
    assert nested_result["cleanup"]["cleaned"] is True  # type: ignore[index]

    monkeypatch.setattr(
        supervisor,
        "_strict_process_commands_for_mutation",
        lambda: {"available": False, "reason": "simulated_ps_failure"},
    )
    process_unknown = supervisor._revalidate_worktree_mutation_preimage(  # noqa: SLF001
        workspace,
        expected_branch=branch,
        expected_head=_git(workspace, "rev-parse", "HEAD"),
        expected_status=(),
    )
    assert process_unknown["valid"] is False
    assert process_unknown["reason"] == "worktree_process_query_unavailable"
    monkeypatch.setattr(
        supervisor,
        "_strict_process_commands_for_mutation",
        lambda: {
            "available": True,
            "reason": "test_process_query_current",
            "commands": [],
        },
    )

    unavailable_root = tmp_path / "missing-worktree-root"
    with guarded_worktree_pool_mutation(
        repo_root=repo,
        worktree_root=unavailable_root,
        workspace_path=(
            unavailable_root / "workspace_333333333333_444444444444"
        ),
        expected_branch=branch,
        operation="test_unavailable_pool_root",
    ) as unavailable:
        assert unavailable["allowed"] is False
        assert unavailable["pooled"] is True
        assert unavailable["reason"] == "worktree_pool_mutation_guard_unavailable"

    legacy_workspace = worktree_root / "workspace-555555555555-666666666666"
    legacy_branch = "implementation/legacy-pool-entry-binding"
    _git(
        repo,
        "worktree",
        "add",
        "-b",
        legacy_branch,
        str(legacy_workspace),
        "HEAD",
    )
    with guarded_worktree_pool_mutation(
        repo_root=repo,
        worktree_root=worktree_root,
        workspace_path=legacy_workspace,
        expected_branch=legacy_branch,
        operation="test_legacy_pool_entry_binding",
    ) as legacy_guard:
        assert legacy_guard["allowed"] is True
        assert legacy_guard["pooled"] is True
        assert legacy_guard["binding"]["entry_id"] == (  # type: ignore[index]
            "555555555555-666666666666"
        )

    # A reasonless native Git lock is foreign custody, not an unlocked record.
    # It must fence the shared mutation guard even when no JSON quarantine
    # marker exists.
    _git(repo, "worktree", "lock", str(legacy_workspace))
    with guarded_worktree_pool_mutation(
        repo_root=repo,
        worktree_root=worktree_root,
        workspace_path=legacy_workspace,
        expected_branch=legacy_branch,
        operation="test_reasonless_foreign_git_lock",
    ) as reasonless_lock:
        assert reasonless_lock["allowed"] is False
        assert reasonless_lock["reason"] == "foreign_git_worktree_lock"
    assert legacy_workspace.is_dir()
    _git(repo, "worktree", "unlock", str(legacy_workspace))

    # A root/custody generation change after the kernel guard is acquired is a
    # denial even when the lexical strings remain unchanged.
    real_binding = worktree_helpers.worktree_pool_entry_guard_binding
    binding_calls = 0

    def generation_changing_binding(**kwargs: object) -> dict[str, object]:
        nonlocal binding_calls
        binding_calls += 1
        result = dict(real_binding(**kwargs))  # type: ignore[arg-type]
        if binding_calls == 2:
            identity = dict(result.get("pool_root_identity") or {})
            identity["inode"] = int(identity.get("inode") or 0) + 1
            result["pool_root_identity"] = identity
        return result

    monkeypatch.setattr(
        worktree_helpers,
        "worktree_pool_entry_guard_binding",
        generation_changing_binding,
    )
    with guarded_worktree_pool_mutation(
        repo_root=repo,
        worktree_root=worktree_root,
        workspace_path=workspace,
        expected_branch=branch,
        operation="test_pool_root_generation_change",
    ) as generation_changed:
        assert generation_changed["allowed"] is False
        assert generation_changed["reason"] == (
            "worktree_pool_mutation_guard_binding_changed"
        )
    monkeypatch.setattr(
        worktree_helpers,
        "worktree_pool_entry_guard_binding",
        real_binding,
    )

    if hasattr(os, "fork"):
        monkeypatch.setattr(
            worktree_helpers,
            "WORKTREE_POOL_MUTATION_GUARD_TIMEOUT_SECONDS",
            0.2,
        )
        read_fd, write_fd = os.pipe()
        with guarded_worktree_pool_mutation(
            repo_root=repo,
            worktree_root=worktree_root,
            workspace_path=workspace,
            expected_branch=branch,
            operation="test_parent_guard_before_fork",
        ) as parent_guard:
            assert parent_guard["allowed"] is True
            child_pid = os.fork()
            if child_pid == 0:  # pragma: no cover - asserted by parent payload
                try:
                    os.close(read_fd)
                    with guarded_worktree_pool_mutation(
                        repo_root=repo,
                        worktree_root=worktree_root,
                        workspace_path=workspace,
                        expected_branch=branch,
                        operation="test_forked_child_guard",
                    ) as child_guard:
                        os.write(
                            write_fd,
                            json.dumps(child_guard, default=str).encode("utf-8"),
                        )
                finally:
                    os.close(write_fd)
                    os._exit(0)
            os.close(write_fd)
            child_payload = os.read(read_fd, 65536)
            os.close(read_fd)
            waited_pid, wait_status = os.waitpid(child_pid, 0)
            assert waited_pid == child_pid
            assert os.waitstatus_to_exitcode(wait_status) == 0
        child_guard = json.loads(child_payload)
        assert child_guard["allowed"] is False
        assert child_guard["reason"] == (
            "worktree_pool_mutation_guard_unavailable"
        )
        monkeypatch.setattr(
            worktree_helpers,
            "WORKTREE_POOL_MUTATION_GUARD_TIMEOUT_SECONDS",
            5.0,
        )

    dirty_path = workspace / "dirty.py"
    dirty_path.write_text("DIRTY = 1\n", encoding="utf-8")
    observed_status = supervisor._git_status_short(workspace)  # noqa: SLF001
    observed_head = _git(workspace, "rev-parse", "HEAD")
    target_ref = supervisor._git_current_branch(repo) or "HEAD"  # noqa: SLF001
    worker_binding_entered = threading.Event()
    rescue_finished = threading.Event()
    rescue_result: dict[str, object] = {}
    real_binding = worktree_helpers.worktree_pool_entry_guard_binding

    def observed_binding(**kwargs: object) -> dict[str, object]:
        if threading.current_thread().name == "waiting-rescue":
            worker_binding_entered.set()
        return real_binding(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        worktree_helpers,
        "worktree_pool_entry_guard_binding",
        observed_binding,
    )

    def waiting_rescue() -> None:
        try:
            rescue_result.update(
                supervisor._rescue_dirty_worktree(  # noqa: SLF001
                    workspace,
                    branch=branch,
                    head=observed_head,
                    target_ref=target_ref,
                    status_lines=observed_status,
                    reason="deterministic_guard_wait_race",
                )
            )
        finally:
            rescue_finished.set()

    with guarded_worktree_pool_mutation(
        repo_root=repo,
        worktree_root=worktree_root,
        workspace_path=workspace,
        expected_branch=branch,
        operation="test_hold_guard_for_preimage_drift",
    ) as held:
        assert held["allowed"] is True
        rescue_thread = threading.Thread(
            target=waiting_rescue,
            name="waiting-rescue",
        )
        rescue_thread.start()
        assert worker_binding_entered.wait(timeout=5.0)
        assert rescue_finished.wait(timeout=0.2) is False
        (workspace / "drift.py").write_text("DRIFT = 1\n", encoding="utf-8")
    rescue_thread.join(timeout=5.0)
    assert not rescue_thread.is_alive()
    assert rescue_result["attempted"] is False
    assert rescue_result["preserved"] is False
    assert rescue_result["reason"] == "worktree_status_changed"
    assert _git(workspace, "branch", "--show-current") == branch


def test_successor_generation_cannot_retire_live_predecessor_claim(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "seed")
    program_root = tmp_path / "program"
    state_dir = program_root / "run-v2" / "state" / "lane-0"
    daemon = PortalImplementationDaemon(
        todo_path=tmp_path / "tasks.md",
        state_path=state_dir / "state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "events.jsonl",
        repo_root=repo,
        implement=True,
        implementation_command=_python_c(
            "raise AssertionError('provider must not run')"
        ),
        use_ephemeral_worktree=True,
        worktree_root=tmp_path / "successor-generation-worktrees",
    )
    task = PortalTask(
        task_id="INC-002-LIVE-PREDECESSOR",
        title="Preserve a live generation lifecycle owner",
        status="todo",
        completion="manual",
        priority="P1",
        track="runtime",
    )
    canonical_task_cid = daemon._canonical_ref(task)
    predecessor_workspace = (
        tmp_path / "predecessor-generation-worktrees" / "live-attempt"
    )
    predecessor = daemon.worktree_lifecycle.begin_preparing(
        task_id=task.task_id,
        canonical_task_cid=canonical_task_cid,
        attempt=1,
        lane_id="predecessor-generation:lane-0",
        workspace_path=predecessor_workspace,
        branch="implementation/inc-002-live-predecessor-attempt-1-old",
        merge_target=daemon._main_branch_name(),
        state_dir=str(
            (program_root / "run-v1" / "state" / "lane-0").resolve()
        ),
    )
    predecessor_record_path = (
        daemon.worktree_lifecycle.workspace_path_for(predecessor_workspace)
    )
    predecessor_index_path = (
        daemon.worktree_lifecycle.task_index_path_for(
            canonical_task_cid=canonical_task_cid,
            task_id=task.task_id,
            attempt=1,
        )
    )
    record_before = predecessor_record_path.read_bytes()
    index_before = predecessor_index_path.read_bytes()

    result = daemon._run_implementation(task, PortalTaskState())

    assert result["reason"] == "worktree_lifecycle_claim_exists"
    assert result["attempt_consumed"] is False
    assert result["provider_call_allowed"] is False
    assert result["predecessor_recovery"]["finalized"] is False
    assert result["predecessor_recovery"]["reason"] == (
        "task_attempt_claim_owner_alive"
    )
    assert predecessor_record_path.read_bytes() == record_before
    assert predecessor_index_path.read_bytes() == index_before
    assert (
        daemon.worktree_lifecycle.load_workspace(predecessor_workspace)
        == predecessor
    )


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


def test_rotated_dead_active_attempt_with_stale_no_dispatch_event_stays_fenced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # This regression exercises the raw-command route.  Keep it independent
    # from supervisor tests which may configure one or more sealed-route
    # fields in the process environment.
    route_fields = {
        "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_PROVIDER",
        "IPFS_ACCELERATE_AGENT_GROK_MODEL",
        "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_FALLBACK_PROVIDER",
        "IPFS_ACCELERATE_AGENT_CODEX_MODEL",
        "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_FALLBACK_TRIGGER",
        "IPFS_ACCELERATE_AGENT_CODEX_REASONING_EFFORT",
    }
    for name in tuple(os.environ):
        if name not in route_fields and not name.startswith(
            "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_ROUTE_"
        ):
            continue
        monkeypatch.delenv(name, raising=False)
    repo = tmp_path / "repo"
    _init_repo(repo)
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "seed")
    attempt_root = tmp_path / "state" / "database_portal_attempts"
    old_state_dir = attempt_root / ("1" * 24)
    new_state_dir = attempt_root / ("2" * 24)
    provider_marker = tmp_path / "provider-called"
    daemon = PortalImplementationDaemon(
        todo_path=new_state_dir / "task-projection.md",
        state_path=new_state_dir / "portal-task-state.json",
        strategy_path=new_state_dir / "strategy.json",
        events_path=new_state_dir / "events.jsonl",
        repo_root=repo,
        implement=True,
        implementation_command=_python_c(
            "from pathlib import Path; "
            f"Path({str(provider_marker)!r}).write_text('called')"
        ),
        use_ephemeral_worktree=True,
        worktree_root=tmp_path / "worktrees",
        worktree_pool_enabled=False,
    )
    clock_now = [1_000.0]
    daemon.worktree_lifecycle.clock = lambda: clock_now[0]
    task = PortalTask(
        task_id="INC-RELOAD-ROTATED",
        title="Recover a source-reloaded database Portal attempt",
        status="todo",
        completion="manual",
        priority="P1",
        track="runtime",
        canonical_task_key="task/v1/inc-reload-rotated",
        canonical_task_cid="cid:inc-reload-rotated",
    )
    old_workspace = tmp_path / "worktrees" / "old-worker"
    old = daemon.worktree_lifecycle.begin_preparing(
        task_id=task.task_id,
        canonical_task_cid=daemon._canonical_ref(task),
        attempt=1,
        lane_id="source-reloaded-worker",
        workspace_path=old_workspace,
        branch="implementation/inc-reload-rotated-old",
        merge_target=daemon._main_branch_name(),
        state_dir=str(old_state_dir),
        owner=ProcessBirthIdentity(
            pid=2**30 - 31,
            start_time_ticks=1,
            boot_id="dead-boot",
        ),
    )
    old = daemon.worktree_lifecycle.mark_active(
        old_workspace,
        lease_id=old.lease_id,
        expected_fence=old.fence,
    )
    old_state_dir.mkdir(parents=True, exist_ok=True)
    # CASF-035 exposed this exact ambiguity: the Portal result still claimed
    # no dispatch while the provider log already contained real Grok frames.
    # Neither artifact grants authority to replace the fenced ACTIVE claim.
    old_log = (
        old_state_dir
        / "implementation-logs"
        / "inc-reload-rotated-attempt-1.log"
    )
    old_log.parent.mkdir(parents=True)
    old_log.write_text(
        """{"type":"response","status":402,"provider":"grok"}
{"type":"available_commands","commands":["read_file"]}
{"type":"available_commands","commands":["apply_patch"]}
""",
        encoding="utf-8",
    )
    old_event = {
        "type": "implementation_finished",
        "task_id": task.task_id,
        "attempt": 1,
        "attempt_consumed": False,
        "provider_dispatched": False,
        "reason": "source_reload_interrupted",
    }
    old_events = old_state_dir / "events.jsonl"
    old_events.write_text(
        json.dumps(old_event, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    PortalTaskState(
        active_task_id=task.task_id,
        active_task_key=task.canonical_task_key,
        active_task_cid=task.canonical_task_cid,
        active_attempt=1,
        active_phase="implementing",
        active_phase_detail="",
        active_log_path=str(old_log),
        active_worktree_path=str(old_workspace.resolve()),
        active_branch=old.branch,
        implementation_in_progress=True,
    ).save(old_state_dir / "portal-task-state.json")
    record_path = daemon.worktree_lifecycle.workspace_path_for(old_workspace)
    index_path = daemon.worktree_lifecycle.task_index_path_for(
        canonical_task_cid=old.canonical_task_cid,
        task_id=old.task_id,
        attempt=old.attempt,
    )
    record_before = record_path.read_bytes()
    index_before = index_path.read_bytes()
    event_before = old_events.read_bytes()
    log_before = old_log.read_bytes()
    assert daemon.worktree_lifecycle.lease_seconds == 21_600.0
    clock_now[0] = old.expires_at + 1.0
    daemon._active_provider_capacity_backoff = (  # type: ignore[method-assign]
        lambda: {}
    )

    result = daemon._run_implementation(task, PortalTaskState())

    assert result.get("reason") == "worktree_lifecycle_claim_exists"
    assert result.get("provider_dispatched") is not True
    assert result.get("attempt_consumed") is False
    assert result.get("provider_call_allowed") is False
    assert not provider_marker.exists()
    assert daemon.worktree_lifecycle.load_workspace(old_workspace) == old
    assert record_path.read_bytes() == record_before
    assert index_path.read_bytes() == index_before
    assert old_events.read_bytes() == event_before
    assert old_log.read_bytes() == log_before


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
    assert result["cleanup_result"]["pool_release"]["metadata_only"] is True
    assert (
        result["cleanup_result"]["branch_disposition"]
        == "retained_exact_expected_head"
    )
    assert result["cleanup_result"]["deleted_branch"] is False
    assert _git(repo, "rev-parse", f"refs/heads/{result['branch']}") == result[
        "baseline_ref"
    ]
    assert daemon._worktree_pool_leases == {}
    assert list((worktree_root / ".pool-state").glob("*.json")) == []
    assert [
        path
        for path in (worktree_root / ".pool-state").glob("*.lock")
        if not path.name.startswith(".")
    ] == []


def test_missing_pooled_workspace_metadata_cleanup_rejects_branch_drift(
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
    advanced_head: list[str] = []

    def remove_workspace_and_advance_branch(*_args, **kwargs) -> None:
        workspace = Path(kwargs["worktree_path"])
        branch = str(kwargs["branch_name"])
        base_head = _git(repo, "rev-parse", f"refs/heads/{branch}")
        tree = _git(repo, "rev-parse", f"{base_head}^{{tree}}")
        candidate = _git(
            repo,
            "commit-tree",
            tree,
            "-p",
            base_head,
            "-m",
            "foreign branch advance",
        )
        _git(repo, "worktree", "remove", "--force", str(workspace))
        _git(
            repo,
            "update-ref",
            f"refs/heads/{branch}",
            candidate,
            base_head,
        )
        advanced_head.append(candidate)
        raise FileNotFoundError(f"workspace disappeared: {workspace}")

    monkeypatch.setattr(
        daemon,
        "_mark_implementation_started",
        remove_workspace_and_advance_branch,
    )
    task = PortalTask(
        task_id="INC-003-BRANCH-DRIFT",
        title="Preserve drifted missing pooled implementation custody",
        status="todo",
        completion="manual",
        priority="P1",
        track="runtime",
    )

    result = daemon._run_implementation(task, PortalTaskState())

    pool_release = result["cleanup_result"]["pool_release"]
    entry_id = str(result["workspace_setup"]["entry_id"])
    branch = str(result["branch"])
    assert result["returncode"] == 1
    assert pool_release["released"] is False
    assert pool_release["deferred"] is True
    assert pool_release["retryable"] is True
    assert advanced_head
    assert _git(repo, "rev-parse", f"refs/heads/{branch}") == advanced_head[0]
    assert (worktree_root / ".pool-state" / f"{entry_id}.json").is_file()
    assert (worktree_root / ".pool-state" / f"{entry_id}.lock").is_file()
    assert daemon._worktree_pool_leases


def test_missing_release_inspector_leaves_legacy_active_state_unfenced(
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
        cache_key="legacy-active",
        base_ref="main",
        branch_name="implementation/legacy-active",
    )

    inspected = inspect_worktree_pool_missing_release_terminal(
        repo_root=repo,
        worktree_root=worktree_root,
        workspace_path=lease.path,
        expected_branch=lease.branch_name,
    )

    assert inspected["status"] == "absent"
    assert inspected["cleanup_fenced"] is False
    assert inspected["reason"] == "missing_release_absent"
    assert lease.release()["released"] is True


def test_missing_workspace_after_provider_dispatch_cannot_publish_receipt(
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
    task = PortalTask(
        task_id="INC-MISSING-POST-DISPATCH",
        title="Fence missing post-dispatch workspace",
        status="todo",
        completion="manual",
        priority="P1",
        track="runtime",
    )
    requested = worktree_root / "requested"
    branch = "implementation/missing-post-dispatch"
    daemon._create_seeded_worktree(requested, branch, task=task)
    effective = daemon._worktree_pool_effective_paths[requested.resolve()]
    lease = daemon._worktree_pool_leases[effective]
    lifecycle = daemon.worktree_lifecycle.begin_preparing(
        task_id=task.task_id,
        canonical_task_cid=daemon._canonical_ref(task),
        attempt=1,
        lane_id="test-lane",
        workspace_path=effective,
        branch=branch,
        merge_target=daemon._main_branch_name(),
        state_dir=str(tmp_path / "state"),
    )
    lifecycle = daemon.worktree_lifecycle.mark_active(
        effective,
        lease_id=lifecycle.lease_id,
        expected_fence=lifecycle.fence,
    )
    daemon._active_worktree_lifecycle = lifecycle
    _git(repo, "worktree", "remove", "--force", str(effective))

    cleanup = daemon._cleanup_merged_worktree(
        effective,
        branch,
        reusable=False,
        allow_missing_pool_metadata_cleanup=True,
        implementation_started=True,
        provider_dispatched=True,
    )

    pool_release = cleanup["pool_release"]
    assert cleanup["cleaned"] is False
    assert pool_release["released"] is False
    assert pool_release["reason"] == (
        "missing_workspace_lifecycle_context_unavailable"
    )
    assert not list((worktree_root / ".pool-state").glob(".*.released-receipt"))
    assert (worktree_root / ".pool-state" / f"{lease.entry_id}.json").is_file()
    assert (worktree_root / ".pool-state" / f"{lease.entry_id}.lock").is_file()


def _missing_current_pool_lease_fixture(
    tmp_path: Path,
    *,
    label: str,
) -> tuple[
    Path,
    WorktreePool,
    WorktreeLease,
    Path,
    Path,
    bytes,
    bytes,
    dict[str, object],
]:
    repo = tmp_path / "repo"
    _init_repo(repo)
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "seed")
    worktree_root = tmp_path / "pool"
    pool = WorktreePool(repo_root=repo, worktree_root=worktree_root)
    lease = pool.acquire(
        cache_key=f"missing-{label}",
        base_ref="main",
        branch_name=f"implementation/missing-{label}",
    )
    state_path = worktree_root / ".pool-state" / f"{lease.entry_id}.json"
    lock_path = worktree_root / ".pool-state" / f"{lease.entry_id}.lock"
    state_bytes = state_path.read_bytes()
    lock_bytes = lock_path.read_bytes()
    owner = current_process_birth()
    lifecycle = {
        "schema": (
            "ipfs_accelerate_py/agent-supervisor/"
            "worktree-lifecycle-record@1"
        ),
        "record_id": f"test-missing-release-{label}",
        "task_id": f"INC-MISSING-{label}",
        "canonical_task_cid": f"cid:missing-{label}",
        "attempt": 1,
        "lane_id": "test-lane",
        "state": "active",
        "owner": owner.to_dict(),
        "lease_id": f"test-lease-{label}",
        "fence": 1,
        "workspace_path": str(lease.path),
        "branch": lease.branch_name,
        "merge_target": "main",
        "created_at": 1.0,
        "updated_at": 1.0,
        "expires_at": 2.0,
        "repo_root": str(repo.resolve()),
        "state_dir": str(tmp_path / "state"),
        "terminal_reason": "",
    }
    missing_release_context: dict[str, object] = {
        "release_phase": "failed_setup_before_provider",
        "implementation_started": False,
        "provider_dispatched": False,
        "lifecycle": lifecycle,
    }
    _git(repo, "worktree", "remove", "--force", str(lease.path))
    return (
        repo,
        pool,
        lease,
        state_path,
        lock_path,
        state_bytes,
        lock_bytes,
        missing_release_context,
    )


def test_missing_pool_terminal_state_survives_lock_unlink_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    (
        repo,
        _pool,
        lease,
        state_path,
        lock_path,
        state_bytes,
        _lock_bytes,
        release_context,
    ) = (
        _missing_current_pool_lease_fixture(tmp_path, label="lock-fault")
    )
    real_unlink = Path.unlink

    def fail_exact_lock_unlink(path: Path, *args, **kwargs) -> None:
        if path == lock_path:
            raise OSError("injected lease-lock unlink failure")
        real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_exact_lock_unlink)

    release = lease.release(
        reusable=False,
        missing_release_context=release_context,
    )

    terminal_path = Path(str(release["terminal_state_path"]))
    assert release["released"] is True
    assert release["terminal_state_durable"] is True
    assert release["lock_cleanup_durable"] is False
    assert release["cleanup_degraded"] is True
    assert release["orphan_lock_retained"] is True
    assert not state_path.exists()
    assert lock_path.is_file()
    assert terminal_path.read_bytes() == state_bytes
    assert terminal_path.suffix == ".released-state"
    assert _git(repo, "rev-parse", f"refs/heads/{lease.branch_name}") == (
        lease.base_commit
    )


def test_missing_pool_terminal_fsync_failure_rolls_back_exact_custody(
    tmp_path: Path,
    monkeypatch,
) -> None:
    (
        _repo,
        _pool,
        lease,
        state_path,
        lock_path,
        state_bytes,
        lock_bytes,
        release_context,
    ) = (
        _missing_current_pool_lease_fixture(tmp_path, label="publish-fsync")
    )
    real_fsync = os.fsync
    terminal_path = state_path.parent / f".{lease.entry_id}.released-state"
    failed = False

    def fail_first_fsync(descriptor: int) -> None:
        nonlocal failed
        descriptor_path = Path(f"/proc/self/fd/{descriptor}")
        try:
            target = descriptor_path.resolve(strict=True)
        except OSError:
            target = None
        if (
            not failed
            and target == state_path.parent
            and terminal_path.exists()
            and lock_path.exists()
        ):
            failed = True
            raise OSError("injected terminal publication fsync failure")
        real_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", fail_first_fsync)

    release = lease.release(
        reusable=False,
        missing_release_context=release_context,
    )

    assert release["released"] is False
    assert release["retryable"] is True
    assert release["rollback_exact"] is True
    assert release["reason"] == (
        "missing_workspace_terminal_publish_rolled_back"
    )
    assert state_path.read_bytes() == state_bytes
    assert lock_path.read_bytes() == lock_bytes
    assert not list(state_path.parent.glob("*.released-state"))


def test_missing_pool_post_terminal_fsync_failure_is_degraded_terminal(
    tmp_path: Path,
    monkeypatch,
) -> None:
    (
        _repo,
        _pool,
        lease,
        state_path,
        lock_path,
        state_bytes,
        _lock_bytes,
        release_context,
    ) = (
        _missing_current_pool_lease_fixture(tmp_path, label="cleanup-fsync")
    )
    real_fsync = os.fsync
    terminal_path = state_path.parent / f".{lease.entry_id}.released-state"
    failed = False

    def fail_second_fsync(descriptor: int) -> None:
        nonlocal failed
        descriptor_path = Path(f"/proc/self/fd/{descriptor}")
        try:
            target = descriptor_path.resolve(strict=True)
        except OSError:
            target = None
        if (
            not failed
            and target == state_path.parent
            and terminal_path.exists()
            and not lock_path.exists()
        ):
            failed = True
            raise OSError("injected post-terminal fsync failure")
        real_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", fail_second_fsync)

    release = lease.release(
        reusable=False,
        missing_release_context=release_context,
    )

    terminal_path = Path(str(release["terminal_state_path"]))
    assert release["released"] is True
    assert release["terminal_state_durable"] is True
    assert release["lock_cleanup_durable"] is False
    assert release["cleanup_degraded"] is True
    assert release["orphan_lock_retained"] is False
    assert not state_path.exists()
    assert not lock_path.exists()
    assert terminal_path.read_bytes() == state_bytes


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX fork")
@pytest.mark.parametrize("crash_phase", ["proposal", "terminal"])
def test_missing_release_restart_recovers_exact_crash_seams(
    tmp_path: Path,
    crash_phase: str,
) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "seed")
    program_root = tmp_path / "program"
    worktree_root = tmp_path / "pool"
    task = PortalTask(
        task_id=f"INC-MISSING-RESTART-{crash_phase.upper()}",
        title=f"Recover missing release {crash_phase} crash",
        status="todo",
        completion="manual",
        priority="P1",
        track="runtime",
    )
    branch = f"implementation/missing-restart-{crash_phase}"
    child_pid = os.fork()
    if child_pid == 0:
        predecessor_state = program_root / "run-v1" / "state" / "lane-0"
        predecessor = PortalImplementationDaemon(
            todo_path=tmp_path / "tasks.md",
            state_path=predecessor_state / "state.json",
            strategy_path=predecessor_state / "strategy.json",
            events_path=predecessor_state / "events.jsonl",
            repo_root=repo,
            use_ephemeral_worktree=True,
            worktree_root=worktree_root,
        )
        requested = worktree_root / "requested"
        predecessor._create_seeded_worktree(requested, branch, task=task)
        effective = predecessor._worktree_pool_effective_paths[
            requested.resolve()
        ]
        lease = predecessor._worktree_pool_leases[effective]
        lifecycle = predecessor.worktree_lifecycle.begin_preparing(
            task_id=task.task_id,
            canonical_task_cid=predecessor._canonical_ref(task),
            attempt=1,
            lane_id="predecessor-generation:lane-0",
            workspace_path=effective,
            branch=branch,
            merge_target=predecessor._main_branch_name(),
            state_dir=str(predecessor_state.resolve()),
        )
        lifecycle = predecessor.worktree_lifecycle.mark_active(
            effective,
            lease_id=lifecycle.lease_id,
            expected_fence=lifecycle.fence,
        )
        predecessor._active_worktree_lifecycle = lifecycle
        _git(repo, "worktree", "remove", "--force", str(effective))
        state_path = worktree_root / ".pool-state" / f"{lease.entry_id}.json"
        lock_path = worktree_root / ".pool-state" / f"{lease.entry_id}.lock"
        terminal_path = (
            worktree_root
            / ".pool-state"
            / f".{lease.entry_id}.released-state"
        )
        if crash_phase == "proposal":
            real_rename = Path.rename

            def crash_before_state_rename(path: Path, target: Path) -> Path:
                if path == state_path and Path(target) == terminal_path:
                    os._exit(72)
                return real_rename(path, target)

            Path.rename = crash_before_state_rename  # type: ignore[method-assign]
        else:
            real_unlink = Path.unlink

            def crash_before_lock_cleanup(path: Path, *args, **kwargs) -> None:
                if path == lock_path:
                    os._exit(73)
                real_unlink(path, *args, **kwargs)

            Path.unlink = crash_before_lock_cleanup  # type: ignore[method-assign]
        predecessor._cleanup_merged_worktree(
            effective,
            branch,
            reusable=False,
            allow_missing_pool_metadata_cleanup=True,
            implementation_started=False,
            provider_dispatched=False,
        )
        os._exit(99)

    waited_pid, status = os.waitpid(child_pid, 0)
    assert waited_pid == child_pid
    assert os.waitstatus_to_exitcode(status) == (
        72 if crash_phase == "proposal" else 73
    )
    successor_state = program_root / "run-v2" / "state" / "lane-0"
    successor = PortalImplementationDaemon(
        todo_path=tmp_path / "tasks.md",
        state_path=successor_state / "state.json",
        strategy_path=successor_state / "strategy.json",
        events_path=successor_state / "events.jsonl",
        repo_root=repo,
        use_ephemeral_worktree=True,
        worktree_root=worktree_root,
    )
    record = successor.worktree_lifecycle.load_task_attempt(
        canonical_task_cid=successor._canonical_ref(task),
        task_id=task.task_id,
        attempt=1,
    )
    assert record is not None
    state_root = worktree_root / ".pool-state"
    receipt_path = next(state_root.glob(".*.released-receipt"))
    entry_id = receipt_path.name.removeprefix(".").removesuffix(
        ".released-receipt"
    )
    active_path = state_root / f"{entry_id}.json"
    terminal_path = state_root / f".{entry_id}.released-state"
    assert receipt_path.is_file()
    assert (active_path.is_file(), terminal_path.is_file()) == (
        (True, False) if crash_phase == "proposal" else (False, True)
    )

    def legacy_quiescence_must_not_run(_record) -> dict[str, object]:
        raise AssertionError("exact missing-release evidence must be consumed")

    successor._predecessor_worktree_dispatch_quiescence = (  # type: ignore[method-assign]
        legacy_quiescence_must_not_run
    )
    if crash_phase == "terminal":
        receipt_bytes = receipt_path.read_bytes()
        tampered = json.loads(receipt_bytes)
        tampered["provider_dispatched"] = True
        receipt_path.write_text(
            json.dumps(tampered, sort_keys=True),
            encoding="utf-8",
        )
        rejected_tamper = (
            successor._finalize_dead_predecessor_worktree_lifecycle_claim(
                task=task,
                attempt=1,
            )
        )
        assert rejected_tamper["finalized"] is False
        assert rejected_tamper["reason"] == (
            "missing_release_terminal_identity_mismatch"
        )
        receipt_path.write_bytes(receipt_bytes)

        active_path.write_bytes(terminal_path.read_bytes())
        rejected_active = (
            successor._finalize_dead_predecessor_worktree_lifecycle_claim(
                task=task,
                attempt=1,
            )
        )
        assert rejected_active["finalized"] is False
        assert rejected_active["reason"] == (
            "missing_release_active_state_conflicts_with_terminal"
        )
        active_path.unlink()

        base_head = _git(repo, "rev-parse", f"refs/heads/{branch}")
        tree = _git(repo, "rev-parse", f"{base_head}^{{tree}}")
        advanced = _git(
            repo,
            "commit-tree",
            tree,
            "-p",
            base_head,
            "-m",
            "foreign branch advance",
        )
        _git(repo, "update-ref", f"refs/heads/{branch}", advanced, base_head)
        rejected_branch = (
            successor._finalize_dead_predecessor_worktree_lifecycle_claim(
                task=task,
                attempt=1,
            )
        )
        assert rejected_branch["finalized"] is False
        assert rejected_branch["reason"] == (
            "missing_release_retained_branch_changed"
        )
        _git(repo, "update-ref", f"refs/heads/{branch}", base_head, advanced)

    recovered = successor._finalize_dead_predecessor_worktree_lifecycle_claim(
        task=task,
        attempt=1,
    )

    assert recovered["finalized"] is True
    assert recovered["predecessor_dispatch_quiescence"]["reason"] == (
        "failed_setup_missing_release_terminal_pre_dispatch"
    )
    if crash_phase == "proposal":
        assert recovered["missing_release_proposal_recovery"]["finalized"] is True
    assert successor.worktree_lifecycle.load_workspace(record.workspace_path) is None
    assert terminal_path.is_file()
    assert not active_path.exists()


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
    stale_release = branch_only.release(reusable=False)
    assert stale_release["released"] is False
    assert stale_release["reason"] == "missing_workspace_pool_custody_changed"
    assert branch_state_path.exists()
    assert branch_lock_path.exists()
    _git(repo, "branch", "-D", "implementation/surviving-branch")
    recovered = pool.reconcile_orphaned_metadata()
    assert recovered["removed_count"] == 1
    assert recovered["removed"][0]["entry_id"] == branch_only.entry_id
    assert not branch_state_path.exists()
    assert not branch_lock_path.exists()


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

    def replace_state_after_claim(state, **kwargs):
        claimed = original_try_claim(state, **kwargs)
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
