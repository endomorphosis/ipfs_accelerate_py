from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.objective_daemon import (
    build_arg_parser,
    discovery_fingerprints,
    run_objective_daemon,
)
from ipfs_accelerate_py.agent_supervisor.bundle_supervisor import (
    build_arg_parser as build_bundle_arg_parser,
    plan_bundle_lanes,
    run_bundle_supervisor,
)
from ipfs_accelerate_py.agent_supervisor.objective_graph import parse_goal_heap
from ipfs_accelerate_py.agent_supervisor.objective_tracker import fibonacci_priority
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    PortalTask,
    TodoImplementationDaemon,
    parse_args as parse_implementation_daemon_args,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor import (
    TodoImplementationSupervisor,
    TodoSupervisorConfig,
    parse_args as parse_implementation_supervisor_args,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.runner import TodoDaemonRunner


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


def _seed_repo(tmp_path: Path) -> tuple[Path, Path, Path]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "main")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.invalid")

    objective_path = repo / "objective-heap.md"
    todo_path = repo / "todo.md"
    source = repo / "src" / "control_surface.py"
    source.parent.mkdir()
    source.write_text(
        """class VoiceCommandSurface:
    def route_click(self, event):
        return event
""",
        encoding="utf-8",
    )
    objective_path.write_text(
        """# Objective Heap

## VAIOS-G010 Meta display control bridge

- Status: active
- Parent:
- Fib priority: 1
- Track: mobile
- Priority: P1
- Bundle: objective/mobile/meta-display
- Goal: Prove the glasses control bridge.
- Evidence: VoiceCommandSurface.route_click, missing_gesture_policy
- Outputs: src, tests
- Validation: test -f objective-heap.md
- Gap task: Add the missing gesture policy proof.
""",
        encoding="utf-8",
    )
    todo_path.write_text("# Agent Todos\n", encoding="utf-8")
    _git(repo, "add", "objective-heap.md", "todo.md", "src/control_surface.py")
    _git(repo, "commit", "-m", "seed objective heap")
    return repo, objective_path, todo_path


def test_todo_daemon_runtime_is_ported_to_accelerate_package():
    assert TodoDaemonRunner.__module__ == "ipfs_accelerate_py.agent_supervisor.todo_daemon.runner"
    assert TodoImplementationDaemon.__module__ == (
        "ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon"
    )
    assert TodoImplementationSupervisor.__module__ == (
        "ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor"
    )
    assert TodoSupervisorConfig.__module__ == (
        "ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor"
    )


def test_implementation_daemon_accepts_configured_submodule_paths(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    daemon = TodoImplementationDaemon(
        todo_path=repo / "todo.md",
        state_path=repo / "state.json",
        strategy_path=repo / "strategy.json",
        events_path=repo / "events.jsonl",
        repo_root=repo,
        worktree_submodule_paths=["packages/app,external/lib", "vendor/tools"],
    )

    assert daemon.worktree_submodule_paths == ("packages/app", "external/lib", "vendor/tools")

    args = parse_implementation_daemon_args(
        [
            "--todo-path",
            str(repo / "todo.md"),
            "--worktree-submodule-path",
            "packages/app",
            "--worktree-submodule-path",
            "external/lib,vendor/tools",
        ]
    )
    assert args.worktree_submodule_path == ["packages/app", "external/lib,vendor/tools"]


def test_implementation_supervisor_passes_configured_submodule_paths(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    config = TodoSupervisorConfig(
        todo_path=repo / "todo.md",
        state_path=repo / "state.json",
        strategy_path=repo / "strategy.json",
        events_path=repo / "events.jsonl",
        state_dir=repo / "state",
        implement=True,
        worktree_submodule_paths=("packages/app", "external/lib"),
    )
    supervisor = TodoImplementationSupervisor(config)

    command = supervisor._build_daemon_command()

    assert command.count("--worktree-submodule-path") == 2
    assert "packages/app" in command
    assert "external/lib" in command

    args = parse_implementation_supervisor_args(
        [
            "--implement",
            "--todo-path",
            str(repo / "todo.md"),
            "--worktree-submodule-path",
            "packages/app",
            "--worktree-submodule-path",
            "external/lib,vendor/tools",
            "--codebase-refill-scan",
            "--codebase-scan-min-open-tasks",
            "0",
            "--codebase-scan-depends-on",
            "AUTO-001,AUTO-002",
        ]
    )
    assert args.worktree_submodule_path == ["packages/app", "external/lib,vendor/tools"]
    assert args.codebase_refill_scan is True
    assert args.codebase_scan_depends_on == ["AUTO-001,AUTO-002"]


def test_implementation_supervisor_refills_drained_codebase_backlog(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "main")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.invalid")
    source = repo / "src" / "runtime.py"
    source.parent.mkdir()
    source.write_text(
        """def route_request(request):
    # TODO: inspect drained supervisor refill
    return request
""",
        encoding="utf-8",
    )
    todo_path = repo / "todo.md"
    todo_path.write_text(
        """# Agent Todos

## AUTO-001 Completed seed

- Status: completed
- Completion: manual
- Priority: P2
- Track: ops
- Depends on:
- Outputs: README.md
- Validation: test -f README.md
- Acceptance: Seed task.
""",
        encoding="utf-8",
    )
    _git(repo, "add", "todo.md", "src/runtime.py")
    _git(repo, "commit", "-m", "seed drained backlog")
    state_dir = repo / "state"
    config = TodoSupervisorConfig(
        todo_path=todo_path,
        state_path=state_dir / "task_state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "supervisor_events.jsonl",
        state_dir=state_dir,
        repo_root=repo,
        task_prefix="## AUTO-",
        codebase_refill_enabled=True,
        codebase_scan_discovery_dir=repo / "discovery",
        codebase_scan_min_open_tasks=0,
        codebase_scan_max_findings=1,
        codebase_scan_cooldown_seconds=21600,
    )

    result = TodoImplementationSupervisor(config).run_once()

    assert result["codebase_refill_count"] == 1
    assert "## AUTO-002 Resolve code annotation in src/runtime.py:2" in todo_path.read_text(encoding="utf-8")
    strategy = json.loads((state_dir / "strategy.json").read_text(encoding="utf-8"))
    assert strategy["last_codebase_scan_mode"] == "drained_exhaustive"
    assert strategy["last_drained_codebase_scan_task_count"] == 1
    assert list((repo / "discovery").glob("*-auto-002-codebase-scan-*.md"))


def test_objective_daemon_generates_todos_bundles_and_dataset(tmp_path):
    repo, objective_path, todo_path = _seed_repo(tmp_path)
    discovery_dir = repo / "data" / "agent_supervisor" / "discovery"
    bundle_dir = repo / "data" / "agent_supervisor" / "objective_bundles"
    dataset_dir = repo / "data" / "agent_supervisor" / "objective_datasets"
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--repo-root",
            str(repo),
            "--objective-path",
            str(objective_path),
            "--todo-path",
            str(todo_path),
            "--discovery-dir",
            str(discovery_dir),
            "--bundle-dir",
            str(bundle_dir),
            "--dataset-dir",
            str(dataset_dir),
            "--task-prefix",
            "ACCEL-",
            "--max-findings",
            "1",
        ]
    )

    payload = run_objective_daemon(args)

    assert payload["schema"] == "ipfs_accelerate_py.agent_supervisor.objective_daemon"
    assert payload["generated_count"] == 1
    assert payload["task_ids"] == ["ACCEL-001"]
    assert "## ACCEL-001 Close objective gap" in todo_path.read_text(encoding="utf-8")
    assert (bundle_dir / "objective-mobile-meta-display.todo.md").exists()
    assert (bundle_dir / "index.json").exists()
    manifest = dataset_dir / "accel-objective-ast.manifest.json"
    assert manifest.exists()
    assert json.loads(manifest.read_text(encoding="utf-8"))["row_count"] >= 2
    assert discovery_fingerprints(discovery_dir)


def test_objective_daemon_suppresses_existing_discovery_fingerprint(tmp_path):
    repo, objective_path, todo_path = _seed_repo(tmp_path)
    discovery_dir = repo / "data" / "agent_supervisor" / "discovery"
    bundle_dir = repo / "data" / "agent_supervisor" / "objective_bundles"
    args = argparse.Namespace(
        repo_root=repo,
        objective_path=objective_path,
        todo_path=todo_path,
        discovery_dir=discovery_dir,
        bundle_dir=bundle_dir,
        dataset_dir=repo / "data" / "agent_supervisor" / "objective_datasets",
        task_prefix="ACCEL-",
        depends_on=[],
        seen_fingerprint=[],
        repeat_existing=False,
        max_findings=1,
        no_persist_ast_dataset=True,
        submit_bundles=False,
        queue_path=None,
        queue_task_type="codex.todo_bundle",
        queue_model_name="codex",
        log_level="INFO",
    )

    first = run_objective_daemon(args)
    second = run_objective_daemon(args)

    assert first["generated_count"] == 1
    assert second["generated_count"] == 0
    assert todo_path.read_text(encoding="utf-8").count("## ACCEL-001 ") == 1


def test_objective_daemon_creates_tracking_document_and_graph(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "main")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.invalid")
    readme = repo / "README.md"
    readme.write_text("# Repo\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "seed")

    objective_path = repo / "docs" / "objective-heap.md"
    todo_path = repo / "docs" / "todo.md"
    graph_path = repo / "data" / "agent_supervisor" / "objective_graph.json"
    args = build_arg_parser().parse_args(
        [
            "--repo-root",
            str(repo),
            "--objective-path",
            str(objective_path),
            "--todo-path",
            str(todo_path),
            "--graph-path",
            str(graph_path),
            "--ensure-tracking-document",
            "--ultimate-goal",
            "Operate as a virtual AI OS for a Meta glasses remote display.",
            "--root-evidence",
            "missing_meta_display_bridge",
            "--task-prefix",
            "ACCEL-",
            "--max-findings",
            "1",
            "--no-persist-ast-dataset",
        ]
    )

    payload = run_objective_daemon(args)

    assert payload["tracking_document_created"] is True
    assert payload["ensured_goal_ids"] == ["OBJ-G000"]
    assert payload["objective_goal_count"] == 1
    assert graph_path.exists()
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    assert graph["graph"]["roots"] == ["OBJ-G000"]
    assert "missing_meta_display_bridge" in objective_path.read_text(encoding="utf-8")
    assert "## ACCEL-001 Close objective gap" in todo_path.read_text(encoding="utf-8")


def test_objective_daemon_refines_missing_evidence_into_child_goals(tmp_path):
    repo, objective_path, todo_path = _seed_repo(tmp_path)
    args = build_arg_parser().parse_args(
        [
            "--repo-root",
            str(repo),
            "--objective-path",
            str(objective_path),
            "--todo-path",
            str(todo_path),
            "--refine-objective-heap",
            "--max-refinement-children",
            "1",
            "--task-prefix",
            "ACCEL-",
            "--max-findings",
            "2",
            "--no-persist-ast-dataset",
        ]
    )

    payload = run_objective_daemon(args)
    goals = parse_goal_heap(objective_path.read_text(encoding="utf-8"))
    refined = [goal for goal in goals if goal.goal_id in payload["refined_goal_ids"]]

    assert payload["refined_goal_ids"] == ["VAIOS-G011"]
    assert payload["generated_count"] == 1
    assert len(refined) == 1
    assert refined[0].parent_goal_ids == ["VAIOS-G010"]
    assert refined[0].required_evidence == ["missing_gesture_policy"]
    assert refined[0].fields["fib_priority"] == str(fibonacci_priority(1, 0))
    todo_text = todo_path.read_text(encoding="utf-8")
    assert "Prove missing_gesture_policy for Meta display control bridge" in todo_text


def test_bundle_supervisor_plans_isolated_lanes(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    index_path = repo / "data" / "agent_supervisor" / "objective_bundles" / "index.json"
    index_path.parent.mkdir(parents=True)
    index_path.write_text(
        json.dumps(
            {
                "source_todo": "docs/main.todo.md",
                "bundles": {
                    "objective/runtime/kernel": {
                        "shard_path": "data/agent_supervisor/objective_bundles/runtime.todo.md",
                        "parallel_lane": "objective/runtime/kernel",
                        "conflict_policy": "bundle-local edits",
                        "tasks": [{"task_id": "ACCEL-001"}],
                    },
                    "objective/mobile/meta-display": {
                        "shard_path": "data/agent_supervisor/objective_bundles/mobile.todo.md",
                        "parallel_lane": "objective/mobile/meta-display",
                        "conflict_policy": "invoke resolver",
                        "tasks": [{"task_id": "ACCEL-002"}],
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    lanes = plan_bundle_lanes(
        bundle_index_path=index_path,
        repo_root=repo,
        state_root=repo / "state",
        worktree_root=repo / "worktrees",
        log_dir=repo / "logs",
        task_prefix="ACCEL-",
        implement=True,
        implementation_command="codex exec --full-auto",
        max_lanes=None,
    )

    assert [lane.bundle_key for lane in lanes] == [
        "objective/mobile/meta-display",
        "objective/runtime/kernel",
    ]
    assert lanes[0].todo_path == repo / "data/agent_supervisor/objective_bundles/mobile.todo.md"
    assert lanes[0].state_dir != lanes[1].state_dir
    assert lanes[0].worktree_root != lanes[1].worktree_root
    assert lanes[0].task_ids == ["ACCEL-002"]
    assert "--implement" in lanes[0].command
    assert "ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor" in lanes[0].command
    assert "--implementation-command" in lanes[0].command


def test_bundle_supervisor_writes_manifest_without_starting_lanes(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    index_path = repo / "objective_bundles" / "index.json"
    index_path.parent.mkdir()
    index_path.write_text(
        json.dumps(
            {
                "source_todo": "docs/main.todo.md",
                "bundles": {
                    "objective/ops/root": {
                        "shard_path": "objective_bundles/root.todo.md",
                        "parallel_lane": "objective/ops/root",
                        "conflict_policy": "use merge resolver",
                        "tasks": [{"task_id": "ACCEL-009"}],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    manifest_path = repo / "manifest.json"
    args = build_bundle_arg_parser().parse_args(
        [
            "--repo-root",
            str(repo),
            "--bundle-index-path",
            str(index_path),
            "--manifest-path",
            str(manifest_path),
            "--task-prefix",
            "ACCEL-",
            "--no-implement",
        ]
    )

    payload = run_bundle_supervisor(args)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert payload["planned_count"] == 1
    assert payload["started_count"] == 0
    assert manifest["lanes"][0]["bundle_key"] == "objective/ops/root"
    assert manifest["lanes"][0]["todo_path"] == "objective_bundles/root.todo.md"
    assert "--no-implement" in manifest["lanes"][0]["command"]


def test_implementation_daemon_invokes_configured_llm_merge_resolver(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "main")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.invalid")
    marker = repo / "README.md"
    marker.write_text("# Repo\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "seed")

    capture_path = tmp_path / "resolver-prompt.txt"
    resolver_script = tmp_path / "resolver.py"
    resolver_script.write_text(
        "import pathlib, sys\n"
        "pathlib.Path(sys.argv[1]).write_text(sys.stdin.read(), encoding='utf-8')\n",
        encoding="utf-8",
    )
    monkeypatch.setenv(
        "IPFS_ACCELERATE_AGENT_LLM_MERGE_RESOLVER_COMMAND",
        f"{shlex.quote(sys.executable)} {shlex.quote(str(resolver_script))} {shlex.quote(str(capture_path))}",
    )
    daemon = TodoImplementationDaemon(
        todo_path=repo / "todo.md",
        state_path=repo / "state" / "task_state.json",
        strategy_path=repo / "state" / "strategy.json",
        events_path=repo / "state" / "events.jsonl",
        repo_root=repo,
    )
    result = daemon._invoke_llm_merge_resolver_for_failed_merge(
        workspace=repo,
        task=PortalTask(
            task_id="ACCEL-999",
            title="Resolve semantic merge",
            status="todo",
            completion="manual",
            priority="P1",
            track="ops",
        ),
        attempt=2,
        branch_name="implementation/accel-999",
        target_branch="main",
        merge_command=["git", "merge", "implementation/accel-999"],
        merge_stdout="",
        merge_stderr="CONFLICT (content): Merge conflict",
    )

    assert result["applied"] is True
    assert result["llm_returncode"] == 0
    prompt = capture_path.read_text(encoding="utf-8")
    assert "ACCEL-999" in prompt
    assert "implementation/accel-999" in prompt
    events = [json.loads(line) for line in (repo / "state" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert events[-1]["type"] == "llm_merge_resolver_invoked"
    assert events[-1]["prompt_chars"] == len(prompt)


def test_implementation_daemon_commits_llm_resolved_merge(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "main")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.invalid")
    target = repo / "conflict.txt"
    target.write_text("base\n", encoding="utf-8")
    _git(repo, "add", "conflict.txt")
    _git(repo, "commit", "-m", "base")
    _git(repo, "checkout", "-b", "feature")
    target.write_text("feature\n", encoding="utf-8")
    _git(repo, "commit", "-am", "feature")
    _git(repo, "checkout", "main")
    target.write_text("main\n", encoding="utf-8")
    _git(repo, "commit", "-am", "main")
    merge = subprocess.run(
        ["git", "merge", "feature"],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    assert merge.returncode != 0
    target.write_text("resolved\n", encoding="utf-8")
    _git(repo, "add", "conflict.txt")
    daemon = TodoImplementationDaemon(
        todo_path=repo / "todo.md",
        state_path=repo / "state" / "task_state.json",
        strategy_path=repo / "state" / "strategy.json",
        events_path=repo / "state" / "events.jsonl",
        repo_root=repo,
    )

    result = daemon._commit_llm_resolved_merge(repo)

    assert result["completed"] is True
    assert target.read_text(encoding="utf-8") == "resolved\n"
    no_merge_head = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", "MERGE_HEAD"],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    assert no_merge_head.returncode != 0
