from __future__ import annotations

import argparse
import json
import subprocess
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
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import TodoImplementationDaemon
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor import (
    TodoImplementationSupervisor,
    TodoSupervisorConfig,
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
    assert "## ACCEL-001 Close virtual AI OS objective gap" in todo_path.read_text(encoding="utf-8")
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
    assert payload["ensured_goal_ids"] == ["VAIOS-G000"]
    assert payload["objective_goal_count"] == 1
    assert graph_path.exists()
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    assert graph["graph"]["roots"] == ["VAIOS-G000"]
    assert "missing_meta_display_bridge" in objective_path.read_text(encoding="utf-8")
    assert "## ACCEL-001 Close virtual AI OS objective gap" in todo_path.read_text(encoding="utf-8")


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
