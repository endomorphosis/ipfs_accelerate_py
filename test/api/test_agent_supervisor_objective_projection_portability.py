from __future__ import annotations

import json
import subprocess
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.objectives.objective_daemon import (
    build_arg_parser,
    persist_objective_plan_evaluations,
    run_objective_daemon,
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


def _seed_repo(parent: Path) -> tuple[Path, Path, Path]:
    repo = parent / "repo"
    repo.mkdir(parents=True)
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
- Acceptance: The glasses control bridge proof is current and validated.
- Gap task: Add the missing gesture policy proof.
""",
        encoding="utf-8",
    )
    todo_path.write_text("# Agent Todos\n", encoding="utf-8")
    _git(repo, "add", "objective-heap.md", "todo.md", "src/control_surface.py")
    _git(repo, "commit", "-m", "seed objective heap")
    return repo, objective_path, todo_path


def _project_checkout(parent: Path) -> tuple[list[str], list[str]]:
    repo, objective_path, todo_path = _seed_repo(parent)
    projection_dir = repo / "projection"
    discovery_dir = projection_dir / "discovery"
    bundle_dir = projection_dir / "bundles"
    graph_path = projection_dir / "objective_graph.json"
    plan_path = projection_dir / "plan_evaluations.json"
    args = build_arg_parser().parse_args(
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
            "--graph-path",
            str(graph_path),
            "--plan-evaluation-path",
            str(plan_path),
            "--task-prefix",
            "ACCEL-",
            "--max-findings",
            "1",
            "--no-persist-ast-dataset",
        ]
    )

    daemon_payload = run_objective_daemon(args)
    replay_payload = run_objective_daemon(args)
    assert replay_payload["generated_count"] == 0
    bundle_path = bundle_dir / "objective-mobile-meta-display.todo.md"
    bundle_index_path = bundle_dir / "index.json"
    vector_path = bundle_dir / "todo_vector_index.json"
    projected_paths = [
        todo_path,
        bundle_path,
        bundle_index_path,
        vector_path,
        graph_path,
        plan_path,
    ]
    projection_text = "\n".join(
        path.read_text(encoding="utf-8") for path in projected_paths
    )
    assert str(repo) not in projection_text

    todo_text = todo_path.read_text(encoding="utf-8")
    discovery_path = next(discovery_dir.glob("*.md"))
    discovery_relative = discovery_path.relative_to(repo).as_posix()
    assert f"- Discovery evidence: {discovery_relative}" in todo_text
    assert f"Use evidence in {discovery_relative}," in todo_text

    bundle_index = json.loads(bundle_index_path.read_text(encoding="utf-8"))
    vector_index = json.loads(vector_path.read_text(encoding="utf-8"))
    objective_graph = json.loads(graph_path.read_text(encoding="utf-8"))
    assert daemon_payload["repo_root"] == "."
    assert bundle_index["plan_evaluation_path"] == (
        "projection/plan_evaluations.json"
    )
    assert vector_index["repo_root"] == "."
    assert objective_graph["objective_path"] == "objective-heap.md"
    task_cids = sorted(
        str(task["canonical_task_cid"])
        for bundle in bundle_index["bundles"].values()
        for task in bundle["tasks"]
    )
    vector_task_cids = sorted(
        str(record["canonical_task_cid"])
        for record in vector_index["records"]
    )
    return task_cids, vector_task_cids


def test_objective_daemon_projection_paths_are_checkout_portable(tmp_path: Path) -> None:
    first = _project_checkout(tmp_path / "first-checkout")
    second = _project_checkout(tmp_path / "second-checkout")

    assert first == second


def test_external_plan_evaluation_path_remains_absolute(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    bundle_index_path = repo / "bundles" / "index.json"
    bundle_index_path.parent.mkdir()
    bundle_index_path.write_text('{"bundles": {}}\n', encoding="utf-8")
    external_plan_path = tmp_path / "external" / "plan_evaluations.json"

    persist_objective_plan_evaluations(
        external_plan_path,
        [],
        bundle_index_path=bundle_index_path,
        repo_root=repo,
    )

    bundle_index = json.loads(bundle_index_path.read_text(encoding="utf-8"))
    assert bundle_index["plan_evaluation_path"] == external_plan_path.as_posix()
