from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.merge.checkout_lock import (
    DEFAULT_CHECKOUT_MUTATION_LOCK_NAME,
    PROTECTED_PATH_MAINTENANCE_LOCK_NAME,
    board_scoped_checkout_mutation_lock_path,
    board_scoped_protected_path_maintenance_lock_path,
    checkout_mutation_lock_path,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.board_control_plane import (
    board_implementation_branch,
    board_merge_lock_name,
    board_protected_path_lock_name,
    ensure_board_implementation_branch,
    infer_board_namespace,
    isolate_board_runtime,
    is_shared_implementation_branch,
    parse_markdown_board_tasks,
    resolve_board_implementation_branch,
)


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    return result.stdout.strip()


def _seed_git_repository(path: Path) -> Path:
    path.mkdir()
    _git(path, "init")
    _git(path, "checkout", "-b", "main")
    _git(path, "config", "user.name", "Test User")
    _git(path, "config", "user.email", "test@example.invalid")
    (path / "seed.txt").write_text("seed\n", encoding="utf-8")
    _git(path, "add", "seed.txt")
    _git(path, "commit", "-m", "seed")
    return path


def test_infer_board_namespace_prefers_explicit_then_branch_then_todo() -> None:
    assert infer_board_namespace(board_namespace="federal-register-reindex") == (
        "federal-register-reindex"
    )
    assert infer_board_namespace(
        merge_target_branch="feature/state-laws-reindex"
    ) == "state-laws-reindex"
    assert infer_board_namespace(
        todo_path="docs/legal_corpora_reindex.todo.md"
    ) == "legal_corpora_reindex"
    assert infer_board_namespace(merge_target_branch="main") == "default"
    assert infer_board_namespace(state_prefix="oul") == "oul"


def test_shared_defaults_are_rewritten_to_implementation_namespace() -> None:
    assert is_shared_implementation_branch("main")
    assert is_shared_implementation_branch("")
    assert not is_shared_implementation_branch("feature/federal-register-reindex")
    assert resolve_board_implementation_branch(
        "main",
        "legal-corpora-reindex",
    ) == "implementation/legal-corpora-reindex"
    assert resolve_board_implementation_branch(
        "feature/federal-register-reindex",
        "legal-corpora-reindex",
    ) == "feature/federal-register-reindex"
    assert board_implementation_branch("open_us_law_reindex") == (
        "implementation/open_us_law_reindex"
    )


def test_board_lock_names_are_stable_and_distinct() -> None:
    lcr = board_merge_lock_name("legal-corpora-reindex")
    slr = board_merge_lock_name("state-laws-reindex")
    assert lcr != slr
    assert lcr.startswith("implementation-board-")
    assert lcr.endswith("-merge.lock")
    assert board_merge_lock_name("legal-corpora-reindex") == lcr
    assert board_protected_path_lock_name("legal-corpora-reindex") != (
        board_protected_path_lock_name("state-laws-reindex")
    )
    assert board_protected_path_lock_name("legal-corpora-reindex") != (
        PROTECTED_PATH_MAINTENANCE_LOCK_NAME
    )


def test_board_scoped_lock_paths_do_not_share_the_global_inode(
    tmp_path: Path,
) -> None:
    repo = _seed_git_repository(tmp_path / "repo")
    sibling = tmp_path / "sibling"
    _git(repo, "worktree", "add", "-b", "sibling", str(sibling))

    global_lock = checkout_mutation_lock_path(repo)
    lcr = board_scoped_checkout_mutation_lock_path(repo, "legal-corpora-reindex")
    slr = board_scoped_checkout_mutation_lock_path(sibling, "state-laws-reindex")
    oul = board_scoped_checkout_mutation_lock_path(repo, "open-us-law-reindex")

    assert global_lock.name == DEFAULT_CHECKOUT_MUTATION_LOCK_NAME
    assert lcr != slr
    assert lcr != oul
    assert lcr != global_lock
    assert lcr.parent == global_lock.parent
    assert lcr.resolve().parent == slr.resolve().parent
    assert board_scoped_protected_path_maintenance_lock_path(
        repo,
        "legal-corpora-reindex",
    ) != board_scoped_protected_path_maintenance_lock_path(
        repo,
        "state-laws-reindex",
    )


def test_ensure_board_implementation_branch_creates_missing_ref(
    tmp_path: Path,
) -> None:
    repo = _seed_git_repository(tmp_path / "repo")
    created = ensure_board_implementation_branch(
        repo,
        "implementation/legal-corpora-reindex",
    )
    assert created["created"] is True
    replay = ensure_board_implementation_branch(
        repo,
        "implementation/legal-corpora-reindex",
    )
    assert replay["created"] is False
    assert _git(repo, "rev-parse", "--abbrev-ref", "HEAD") == "main"
    assert _git(
        repo,
        "rev-parse",
        "--verify",
        "implementation/legal-corpora-reindex",
    )


def test_parse_markdown_board_tasks_extracts_status_and_deps() -> None:
    tasks = parse_markdown_board_tasks(
        """
# Board

## LCR-034 Land the remaining federal-register slice
- Status: ready
- Depends on: LCR-033
- Is schedulable: yes

## LCR-035 Follow-up
- Status: pending
- Depends on: LCR-034
"""
    )
    assert [task["task_id"] for task in tasks] == ["LCR-034", "LCR-035"]
    assert tasks[0]["status"] == "ready"
    assert tasks[0]["depends_on"] == ["LCR-033"]
    assert tasks[1]["depends_on"] == ["LCR-034"]


def test_isolate_board_runtime_registers_duckdb_catalog(tmp_path: Path) -> None:
    pytest.importorskip("duckdb")
    from ipfs_accelerate_py.agent_supervisor.task_sources.board_control_plane import (
        open_board_control_plane,
    )

    repo = _seed_git_repository(tmp_path / "repo")
    todo = repo / "legal_corpora_reindex.todo.md"
    todo.write_text(
        "## LCR-001 First task\n- Status: ready\n- Depends on: None\n",
        encoding="utf-8",
    )
    other = repo / "state_laws_reindex.todo.md"
    other.write_text(
        "## SLR-009 Gate\n- Status: ready\n",
        encoding="utf-8",
    )

    first = isolate_board_runtime(
        repo_root=repo,
        todo_path=todo,
        merge_target_branch="main",
    )
    second = isolate_board_runtime(
        repo_root=repo,
        todo_path=other,
        merge_target_branch="main",
    )

    assert first["implementation_branch"] == (
        "implementation/legal_corpora_reindex"
    )
    assert second["implementation_branch"] == (
        "implementation/state_laws_reindex"
    )
    assert first["merge_lock_name"] != second["merge_lock_name"]
    assert first["control_plane"] is not None
    assert first["registration"]["task_count"] == 1
    assert _git(
        repo,
        "rev-parse",
        "--verify",
        first["implementation_branch"],
    )

    plane = open_board_control_plane(repo)
    try:
        boards = {item["board_namespace"]: item for item in plane.list_boards()}
        assert set(boards) == {
            "legal_corpora_reindex",
            "state_laws_reindex",
        }
        plane.put_artefact(
            "ast",
            board_namespace="legal_corpora_reindex",
            artefact_id="mod:daemon",
            path="implementation_daemon.py",
            digest="sha256:abc",
            payload={"kind": "module"},
        )
        plane.put_artefact(
            "embedding",
            board_namespace="legal_corpora_reindex",
            artefact_id="task:LCR-001",
            model="test",
            vector=[0.1, 0.2, 0.3],
            text="First task",
        )
        plane.put_artefact(
            "bm25",
            board_namespace="legal_corpora_reindex",
            artefact_id="task:LCR-001",
            field="title",
            tokens="first task",
        )
        plane.put_artefact(
            "knowledge_graph",
            board_namespace="legal_corpora_reindex",
            artefact_id="edge:1",
            subject="LCR-001",
            predicate="depends_on",
            object="none",
        )
        plane.put_artefact(
            "proof_cache",
            board_namespace="legal_corpora_reindex",
            artefact_id="proof:1",
            obligation_id="obl:ready",
            status="admitted",
            payload={"ok": True},
        )
        ast_rows = plane.list_artefacts(
            "ast",
            board_namespace="legal_corpora_reindex",
        )
        assert ast_rows[0]["artefact_id"] == "mod:daemon"
        assert plane.list_artefacts(
            "embedding",
            board_namespace="legal_corpora_reindex",
        )
        assert plane.list_artefacts(
            "bm25",
            board_namespace="legal_corpora_reindex",
        )
        assert plane.list_artefacts(
            "knowledge_graph",
            board_namespace="legal_corpora_reindex",
        )
        proofs = plane.list_artefacts(
            "proof_cache",
            board_namespace="legal_corpora_reindex",
        )
        assert proofs[0]["status"] == "admitted"
        assert plane.metadata()["schema"].endswith("board-control-plane@1")
        assert Path(first["registration"]["duckdb_path"]).is_file()
    finally:
        plane.close()


def test_ingest_codebase_artefacts_stores_ast_vector_kg_and_proof(
    tmp_path: Path,
) -> None:
    pytest.importorskip("duckdb")
    from ipfs_accelerate_py.agent_supervisor.task_sources.board_control_plane import (
        open_board_control_plane,
    )

    repo = _seed_git_repository(tmp_path / "repo")
    package = repo / "pkg"
    package.mkdir()
    (package / "alpha.py").write_text(
        "import json\n\nclass Alpha:\n    def run(self):\n        return json.dumps({})\n",
        encoding="utf-8",
    )
    (package / "beta.py").write_text(
        "from pkg.alpha import Alpha\n\ndef build():\n    return Alpha()\n",
        encoding="utf-8",
    )

    plane = open_board_control_plane(repo)
    try:
        summary = plane.ingest_codebase_artefacts(
            repo,
            "legal-corpora-reindex",
            source_root=package,
        )
        assert summary["file_count"] == 2
        assert summary["ast_count"] == 2
        assert summary["embedding_count"] == 2
        assert summary["knowledge_graph_count"] >= 2
        assert summary["proof_cache_count"] >= 2
        ast_rows = plane.list_artefacts(
            "ast",
            board_namespace="legal-corpora-reindex",
        )
        assert any("alpha.py" in str(row.get("path") or row) for row in ast_rows)
        vectors = plane.list_artefacts(
            "vector_index",
            board_namespace="legal-corpora-reindex",
        )
        assert vectors
        kg = plane.list_artefacts(
            "knowledge_graph",
            board_namespace="legal-corpora-reindex",
        )
        assert any(
            str(row.get("predicate") or "") == "imports" or "imports" in str(row)
            for row in kg
        )
        proofs = plane.list_artefacts(
            "proof_cache",
            board_namespace="legal-corpora-reindex",
        )
        assert any(str(row.get("status") or "") == "verified" for row in proofs)
        if plane.ducklake_attached:
            ast_in_lake = plane._conn().execute(
                "SELECT COUNT(*) FROM lake.artefact_ast"
            ).fetchone()
            kg_in_lake = plane._conn().execute(
                "SELECT COUNT(*) FROM lake.artefact_knowledge_graph"
            ).fetchone()
            assert int(ast_in_lake[0]) >= 2
            assert int(kg_in_lake[0]) >= 2
    finally:
        plane.close()


def test_discover_board_todo_path_finds_architecture_board(tmp_path: Path) -> None:
    from ipfs_accelerate_py.agent_supervisor.task_sources.board_control_plane import (
        discover_board_todo_path,
    )

    repo = tmp_path / "repo"
    (repo / "docs" / "architecture").mkdir(parents=True)
    todo = repo / "docs" / "architecture" / "legal_corpora_reindex.todo.md"
    todo.write_text("## LCR-001 Task\n", encoding="utf-8")
    assert discover_board_todo_path(repo, "legal-corpora-reindex") == todo


def test_discover_board_todo_path_finds_versioned_fabric_boards(tmp_path: Path) -> None:
    from ipfs_accelerate_py.agent_supervisor.task_sources.board_control_plane import (
        discover_board_todo_path,
    )

    repo = tmp_path / "repo"
    lgcvf = (
        repo
        / "docs"
        / "architecture"
        / "logic_governed_compositional_verification_fabric.todo.md"
    )
    lgcvf.parent.mkdir(parents=True)
    lgcvf.write_text("## LGCVF-001 Task\n", encoding="utf-8")
    eaaef = (
        repo
        / "docs"
        / "architecture"
        / "external_agent_autonomous_execution_fabric"
        / "TASK_BOARD.md"
    )
    eaaef.parent.mkdir(parents=True)
    eaaef.write_text("## EAAEF-000 Admit\n", encoding="utf-8")
    assert discover_board_todo_path(
        repo, "logic-governed-compositional-verification-fabric-v1"
    ) == lgcvf
    assert discover_board_todo_path(
        repo, "external-agent-autonomous-execution-fabric-v1"
    ) == eaaef


def test_configured_board_common_args_isolate_shared_merge_target(
    tmp_path: Path,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.runtime.configured_board_scheduler import (
        configured_board_common_args,
        configured_board_launch_plan,
        load_configured_board,
    )

    repo = _seed_git_repository(tmp_path / "repo")
    (repo / "docs").mkdir()
    (repo / "docs/tasks.md").write_text("# Tasks\n", encoding="utf-8")
    (repo / "docs/objectives.md").write_text("# Objectives\n", encoding="utf-8")
    (repo / "docs/plan.md").write_text("plan\n", encoding="utf-8")
    (repo / "scripts/ops/agent_supervisor").mkdir(parents=True)
    (repo / "scripts/validate_board.py").write_text("print('ok')\n", encoding="utf-8")
    (
        repo / "scripts/ops/agent_supervisor/implementation_supervisor_entry.py"
    ).write_text("raise SystemExit(0)\n", encoding="utf-8")
    config = repo / "config/scheduler.json"
    config.parent.mkdir()
    config.write_text(
        """
{
  "schema": "ipfs_accelerate_py.agent_supervisor.configured_board_test.scheduler_config@1",
  "taskboard_path": "docs/tasks.md",
  "objectives_path": "docs/objectives.md",
  "plan_path": "docs/plan.md",
  "validator_path": "scripts/validate_board.py",
  "task_prefix": "TEST-",
  "goal_prefix": "TEST-G",
  "board_namespace": "configured-board-test",
  "merge_target_branch": "main",
  "source_binding": {
    "accelerator_required_ancestor": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "accelerator_required_branch": "main",
    "dependency_submodule_path": "docs",
    "dependency_planning_revision": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
  },
  "max_lanes": 1,
  "lanes": [{"index": 0, "name": "lane-0", "strict_shard_remainder": 0}],
  "strict_task_sharding": true,
  "exit_when_all_tracks_terminal": true,
  "objective_refill_enabled": false,
  "codebase_refill_enabled": false,
  "poll_interval_seconds": 5,
  "daemon_interval_seconds": 60,
  "check_interval_seconds": 30,
  "stale_seconds": 1800,
  "watchdog_startup_grace_seconds": 30,
  "implementation_timeout_seconds": 60,
  "implementation_max_timeout_seconds": 120,
  "implementation_log_stall_seconds": 30,
  "max_restarts": 1,
  "max_task_attempts": 1,
  "implementation_retry_budget": 1,
  "validation_retry_budget": 1,
  "merge_retry_budget": 1,
  "worktree_submodule_paths": [],
  "protected_paths": ["config/scheduler.json"],
  "runtime_paths": {
    "root": "workspace/agent-supervisor",
    "state": "workspace/agent-supervisor/state",
    "worktrees": "workspace/agent-supervisor/worktrees",
    "merge_queue": "workspace/agent-supervisor/merge-queue",
    "logs": "workspace/agent-supervisor/logs"
  },
  "provider": {"provider_id": "codex", "model_id": "test-model", "max_concurrency": 1}
}
""".strip()
        + "\n",
        encoding="utf-8",
    )
    board = load_configured_board(config, repo_root=repo)
    args = configured_board_common_args(board, implement=False)
    assert "--board-namespace" in args
    assert args[args.index("--board-namespace") + 1] == "configured-board-test"
    assert args[args.index("--merge-target-branch") + 1] == (
        "implementation/configured-board-test"
    )
    plan = configured_board_launch_plan(
        board,
        implement=False,
        detach=False,
        stamp="20260817T000000Z",
    )
    assert plan["implementation_branch"] == "implementation/configured-board-test"
    assert plan["merge_lock_name"] == board_merge_lock_name(
        "configured-board-test"
    )


def test_implementation_daemons_on_sibling_boards_do_not_share_locks(
    tmp_path: Path,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
        PortalImplementationDaemon,
    )

    repo = _seed_git_repository(tmp_path / "repo")
    lcr_todo = repo / "legal_corpora_reindex.todo.md"
    slr_todo = repo / "state_laws_reindex.todo.md"
    lcr_todo.write_text("## LCR-001\n- Status: ready\n", encoding="utf-8")
    slr_todo.write_text("## SLR-001\n- Status: ready\n", encoding="utf-8")
    state = tmp_path / "state"
    state.mkdir()

    lcr = PortalImplementationDaemon(
        todo_path=lcr_todo,
        state_path=state / "lcr_task_state.json",
        strategy_path=state / "lcr_strategy.json",
        events_path=state / "lcr_events.jsonl",
        repo_root=repo,
        task_header_prefix="## LCR-",
        merge_target_branch="main",
    )
    slr = PortalImplementationDaemon(
        todo_path=slr_todo,
        state_path=state / "slr_task_state.json",
        strategy_path=state / "slr_strategy.json",
        events_path=state / "slr_events.jsonl",
        repo_root=repo,
        task_header_prefix="## SLR-",
        merge_target_branch="main",
    )

    assert lcr.board_namespace == "legal_corpora_reindex"
    assert slr.board_namespace == "state_laws_reindex"
    assert lcr.resolved_merge_target_branch == (
        "implementation/legal_corpora_reindex"
    )
    assert slr.resolved_merge_target_branch == (
        "implementation/state_laws_reindex"
    )
    assert lcr._repo_merge_lock_path() != slr._repo_merge_lock_path()
    assert lcr._protected_path_maintenance_lock_path() != (
        slr._protected_path_maintenance_lock_path()
    )
    assert lcr._repo_merge_lock_path() != checkout_mutation_lock_path(repo)
