from __future__ import annotations

from hashlib import sha256
import json
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.backlog_refinery import (
    CodebaseScanInventory,
    scan_codebase_findings,
)
from ipfs_accelerate_py.agent_supervisor.dataset_store import ObjectiveDatasetStore
from ipfs_accelerate_py.agent_supervisor import objective_graph
from ipfs_accelerate_py.agent_supervisor.objective_graph import scan_objective_gaps
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    PortalImplementationDaemon,
    PortalTask,
    PortalTaskState,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor import (
    PortalImplementationSupervisor,
    PortalSupervisorConfig,
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


def test_implementation_daemon_releases_pool_lease_before_merge_queue_handoff(tmp_path: Path) -> None:
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
        implementation_command=(
            "python -c \"from pathlib import Path; "
            "Path('feature.py').write_text('VALUE = 1\\\\n')\""
        ),
        use_ephemeral_worktree=True,
        worktree_root=worktree_root,
    )
    daemon._consume_one_merge_candidate = lambda: None  # type: ignore[method-assign]
    task = PortalTask(
        task_id="INC-001",
        title="Release pooled merge handoff",
        status="todo",
        completion="manual",
        priority="P1",
        track="runtime",
        outputs=["feature.py"],
        validation=["python -m py_compile feature.py"],
    )

    result = daemon._run_implementation(task, PortalTaskState())

    merge_result = result["merge_result"]
    handoff = merge_result["worktree_pool_handoff"]
    assert merge_result["queued"] is True
    assert handoff["released"] is True
    assert handoff["pooled"] is True
    assert daemon._worktree_pool_leases == {}
    assert list((worktree_root / ".pool-state").glob("*.lock")) == []
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
        implementation_command="python -c \"raise SystemExit(7)\"",
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
    assert daemon._worktree_pool_leases == {}
    assert list((worktree_root / ".pool-state").glob("*.lock")) == []


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
        implementation_command="python -c \"raise AssertionError('must not run')\"",
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
