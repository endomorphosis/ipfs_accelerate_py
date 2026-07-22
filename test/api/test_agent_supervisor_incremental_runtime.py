from __future__ import annotations

import subprocess
import time
from dataclasses import asdict
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.dataset_store import ObjectiveDatasetStore
from ipfs_accelerate_py.agent_supervisor import objective_graph
from ipfs_accelerate_py.agent_supervisor.objective_graph import scan_objective_gaps
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    PortalImplementationDaemon,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.worktrees import WorktreePool


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
- Evidence: AlphaRouter.dispatch, durable runtime notes, still_missing_contract
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
        "# Durable runtime notes\n\nThe durable runtime notes are available.\n",
        encoding="utf-8",
    )
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "seed incremental objective")
    return repo, objective, tmp_path / "datasets"


def test_ast_and_evidence_snapshots_reparse_only_changed_blobs(tmp_path: Path, monkeypatch) -> None:
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
    assert parse_calls == cold_parse_calls
    assert warm_stats["parsed_record_count"] == 0
    assert warm_stats["reused_record_count"] == 2
    assert float(warm_stats["saved_parse_seconds"]) > 0

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
    assert changed_stats["parsed_record_count"] == 1
    assert changed_stats["reused_record_count"] == 1


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
    assert rename_stats["parsed_record_count"] == 0
    assert rename_stats["renamed_record_count"] == 1
    assert rename_stats["deleted_record_count"] == 1
    assert renamed_plan[0].present_evidence["durable runtime notes"] == [
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
    assert "durable runtime notes" in deleted_plan[0].missing_evidence
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
