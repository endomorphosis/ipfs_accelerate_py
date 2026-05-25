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
