"""Expected-output preflight uses the Git repository that owns each path."""

from __future__ import annotations

import subprocess
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    PortalTask,
    TodoImplementationDaemon,
)


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"git {' '.join(args)} failed in {repo}:\n"
        f"stdout: {result.stdout}\n"
        f"stderr: {result.stderr}"
    )
    return result.stdout.strip()


def _init_repo(path: Path) -> Path:
    path.mkdir(parents=True)
    _git(path, "init")
    _git(path, "checkout", "-b", "main")
    _git(path, "config", "user.name", "Submodule Preflight Test")
    _git(path, "config", "user.email", "submodule-preflight@example.invalid")
    return path


def _seed_parent_with_submodule(tmp_path: Path) -> tuple[Path, Path, str]:
    child_source = _init_repo(tmp_path / "child-source")
    (child_source / "README.md").write_text("child baseline\n", encoding="utf-8")
    _git(child_source, "add", "README.md")
    _git(child_source, "commit", "-m", "child baseline")

    parent = _init_repo(tmp_path / "parent")
    (parent / ".gitignore").write_text("*.json\n", encoding="utf-8")
    _git(parent, "add", ".gitignore")
    _git(parent, "commit", "-m", "outer JSON ignore policy")
    _git(
        parent,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        str(child_source),
        "deps/child",
    )
    _git(parent, "commit", "-am", "add child submodule")
    child = parent / "deps" / "child"
    _git(child, "config", "user.name", "Submodule Preflight Test")
    _git(child, "config", "user.email", "submodule-preflight@example.invalid")
    return parent, child, _git(parent, "rev-parse", "HEAD")


def _daemon(
    parent: Path,
    tmp_path: Path,
    *,
    configured: bool = True,
    protected_paths: tuple[str, ...] = (),
) -> TodoImplementationDaemon:
    state_dir = tmp_path / "state"
    return TodoImplementationDaemon(
        todo_path=parent / "todo.md",
        state_path=state_dir / "task-state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "events.jsonl",
        repo_root=parent,
        worktree_submodule_paths=("deps/child",) if configured else (),
        implementation_protected_paths=protected_paths,
        worktree_pool_enabled=False,
    )


def _task(output: str) -> PortalTask:
    return PortalTask(
        task_id="SUBMODULE-OUTPUT-001",
        title="Produce a nested JSON report",
        status="todo",
        completion="manual",
        priority="P0",
        track="supervisor-integrity",
        outputs=[output],
        validation=["python -m pytest"],
        acceptance="The nested repository owns ignore and index authority.",
    )


def test_outer_json_ignore_does_not_reject_unignored_nested_output(
    tmp_path: Path,
) -> None:
    parent, child, baseline = _seed_parent_with_submodule(tmp_path)
    output = "deps/child/reports/result.json"
    report = child / "reports" / "result.json"
    report.parent.mkdir()
    report.write_text('{"accepted": true}\n', encoding="utf-8")
    daemon = _daemon(parent, tmp_path)
    task = _task(output)

    outer_ignore = subprocess.run(
        ["git", "check-ignore", "--no-index", "--quiet", "--", output],
        cwd=parent,
        check=False,
    )
    child_ignore = subprocess.run(
        [
            "git",
            "check-ignore",
            "--no-index",
            "--quiet",
            "--",
            "reports/result.json",
        ],
        cwd=child,
        check=False,
    )
    preflight = daemon._prepare_proposal_expected_outputs(
        parent,
        task,
        baseline_ref=baseline,
        scope_paths=(output,),
    )
    proposal = daemon._validate_implementation_patch(
        parent,
        task,
        baseline_ref=baseline,
    )

    assert outer_ignore.returncode == 0
    assert child_ignore.returncode == 1
    check = preflight["checks"][0]
    assert check["index_repository"] == "deps/child"
    assert check["index_path"] == "reports/result.json"
    assert check["ignored"] is False
    assert check["force_stage_required"] is False
    assert check["issue"] == ""
    assert proposal.accepted is True
    assert proposal.proposal.changed_paths == (output,)
    assert _git(child, "diff", "--cached", "--name-only") == ""


def test_child_ignored_output_is_force_staged_only_in_child_index(
    tmp_path: Path,
) -> None:
    parent, child, _baseline = _seed_parent_with_submodule(tmp_path)
    (child / ".gitignore").write_text("reports/*.json\n", encoding="utf-8")
    _git(child, "add", ".gitignore")
    _git(child, "commit", "-m", "ignore child reports")
    _git(parent, "add", "deps/child")
    _git(parent, "commit", "-m", "record child ignore policy")
    baseline = _git(parent, "rev-parse", "HEAD")
    output = "deps/child/reports/result.json"
    report = child / "reports" / "result.json"
    report.parent.mkdir()
    report.write_text('{"accepted": true}\n', encoding="utf-8")

    preflight = _daemon(parent, tmp_path)._prepare_proposal_expected_outputs(
        parent,
        _task(output),
        baseline_ref=baseline,
        scope_paths=(output,),
    )
    check = preflight["checks"][0]

    assert check["index_repository"] == "deps/child"
    assert check["index_path"] == "reports/result.json"
    assert check["ignored"] is True
    assert check["force_stage_required"] is True
    assert check["force_stage_attempted"] is True
    assert check["force_stage_succeeded"] is True
    assert check["staged"] is True
    assert check["issue"] == ""
    assert preflight["staged_paths"] == [output]
    assert _git(child, "diff", "--cached", "--name-only") == (
        "reports/result.json"
    )
    assert _git(parent, "ls-files", "--", output) == ""


def test_outer_json_ignore_accepts_output_tracked_by_nested_commit(
    tmp_path: Path,
) -> None:
    parent, child, baseline = _seed_parent_with_submodule(tmp_path)
    output = "deps/child/reports/committed.json"
    report = child / "reports" / "committed.json"
    report.parent.mkdir()
    report.write_text('{"committed": true}\n', encoding="utf-8")
    _git(child, "add", "reports/committed.json")
    _git(child, "commit", "-m", "track exact expected output")
    daemon = _daemon(parent, tmp_path)
    task = _task(output)

    preflight = daemon._prepare_proposal_expected_outputs(
        parent,
        task,
        baseline_ref=baseline,
        scope_paths=(output,),
    )
    proposal = daemon._validate_implementation_patch(
        parent,
        task,
        baseline_ref=baseline,
    )
    check = preflight["checks"][0]

    assert check["index_repository"] == "deps/child"
    assert check["indexed_before"] is True
    assert check["ignored"] is False
    assert check["force_stage_required"] is False
    assert check["issue"] == ""
    assert proposal.accepted is True
    assert proposal.proposal.changed_paths == (output,)


def test_unmanaged_submodule_output_remains_fail_closed(tmp_path: Path) -> None:
    parent, child, baseline = _seed_parent_with_submodule(tmp_path)
    output = "deps/child/reports/result.json"
    report = child / "reports" / "result.json"
    report.parent.mkdir()
    report.write_text('{"accepted": false}\n', encoding="utf-8")

    preflight = _daemon(
        parent,
        tmp_path,
        configured=False,
    )._prepare_proposal_expected_outputs(
        parent,
        _task(output),
        baseline_ref=baseline,
        scope_paths=(output,),
    )
    check = preflight["checks"][0]

    assert check["index_repository"] == "."
    assert check["submodule_bound"] is True
    assert check["force_stage_succeeded"] is False
    assert check["issue"] == "expected_output_force_add_forbidden"
    assert _git(child, "diff", "--cached", "--name-only") == ""


def test_protected_or_symlinked_nested_output_remains_fail_closed(
    tmp_path: Path,
) -> None:
    parent, child, _baseline = _seed_parent_with_submodule(tmp_path)
    (child / ".gitignore").write_text("reports/*.json\n", encoding="utf-8")
    _git(child, "add", ".gitignore")
    _git(child, "commit", "-m", "ignore child reports")
    _git(parent, "add", "deps/child")
    _git(parent, "commit", "-m", "record child ignore policy")
    baseline = _git(parent, "rev-parse", "HEAD")

    protected_output = "deps/child/reports/protected.json"
    protected_report = child / "reports" / "protected.json"
    protected_report.parent.mkdir()
    protected_report.write_text('{"protected": true}\n', encoding="utf-8")
    protected = _daemon(
        parent,
        tmp_path,
        protected_paths=(protected_output,),
    )._prepare_proposal_expected_outputs(
        parent,
        _task(protected_output),
        baseline_ref=baseline,
        scope_paths=(protected_output,),
    )["checks"][0]

    target = child / "target.json"
    target.write_text('{"target": true}\n', encoding="utf-8")
    symlink_output = "deps/child/reports/symlink.json"
    (child / "reports" / "symlink.json").symlink_to(target)
    symlinked = _daemon(parent, tmp_path)._prepare_proposal_expected_outputs(
        parent,
        _task(symlink_output),
        baseline_ref=baseline,
        scope_paths=(symlink_output,),
    )["checks"][0]

    assert protected["protected"] is True
    assert protected["issue"] == "expected_output_force_add_forbidden"
    assert symlinked["symlink_bound"] is True
    assert symlinked["regular_file"] is False
    assert symlinked["issue"] == "expected_output_force_add_forbidden"
    assert _git(child, "diff", "--cached", "--name-only") == ""


def test_auto_rescue_stages_only_declared_output_in_child_index(
    tmp_path: Path,
) -> None:
    parent, child, _baseline = _seed_parent_with_submodule(tmp_path)
    output = "deps/child/reports/result.json"
    report = child / "reports" / "result.json"
    report.parent.mkdir()
    report.write_text('{"accepted": true}\n', encoding="utf-8")
    child_unrelated = child / "reports" / "unrelated.txt"
    child_unrelated.write_text("leave untracked\n", encoding="utf-8")
    parent_unrelated = parent / "unrelated.txt"
    parent_unrelated.write_text("leave untracked\n", encoding="utf-8")

    staged = _daemon(parent, tmp_path)._stage_declared_candidate_outputs(
        parent,
        _task(output),
    )

    assert staged == (output,)
    assert _git(child, "diff", "--cached", "--name-only") == (
        "reports/result.json"
    )
    assert _git(parent, "diff", "--cached", "--name-only") == ""
    assert _git(child, "status", "--porcelain", "--", "reports/unrelated.txt") == (
        "?? reports/unrelated.txt"
    )
    assert _git(parent, "status", "--porcelain", "--", "unrelated.txt") == (
        "?? unrelated.txt"
    )


def test_auto_rescue_does_not_stage_unmanaged_submodule_output(
    tmp_path: Path,
) -> None:
    parent, child, _baseline = _seed_parent_with_submodule(tmp_path)
    output = "deps/child/reports/result.json"
    report = child / "reports" / "result.json"
    report.parent.mkdir()
    report.write_text('{"accepted": false}\n', encoding="utf-8")

    staged = _daemon(
        parent,
        tmp_path,
        configured=False,
    )._stage_declared_candidate_outputs(parent, _task(output))

    assert staged == ()
    assert _git(child, "diff", "--cached", "--name-only") == ""
    assert _git(child, "status", "--porcelain", "--", "reports/result.json") == (
        "?? reports/result.json"
    )
