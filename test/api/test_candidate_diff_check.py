"""Real Git vectors for the baseline-to-candidate whitespace gate."""

from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.validation.candidate_diff_check import (
    check_candidate_diff,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    PortalTask, TodoImplementationDaemon,
)


def git(root, *args):
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def repository(root):
    root.mkdir()
    git(root, "init", "-q")
    git(root, "config", "user.name", "Candidate Test")
    git(root, "config", "user.email", "candidate@example.invalid")
    (root / "a.py").write_text("x = 1\n")
    git(root, "add", ".")
    git(root, "commit", "-qm", "baseline")
    return git(root, "rev-parse", "HEAD")


@pytest.mark.parametrize("kind", ["committed", "staged", "working", "untracked", "assume-unchanged", "skip-worktree"])
def test_check_covers_candidate_bytes_without_changing_live_index(tmp_path, kind):
    root = tmp_path / "repo"
    baseline = repository(root)
    path = "new.py" if kind == "untracked" else "a.py"
    if kind in {"assume-unchanged", "skip-worktree"}:
        git(root, "update-index", f"--{kind}", path)
    (root / path).write_text("x = 2  \n")
    if kind in {"committed", "staged"}:
        git(root, "add", path)
    if kind == "committed":
        git(root, "commit", "-qm", "candidate")
    index = (root / ".git/index").read_bytes()
    result = check_candidate_diff(root, baseline)
    assert result["returncode"] != 0
    assert f"{path}:1: trailing whitespace" in result["output"]
    assert (root / ".git/index").read_bytes() == index


def test_bad_baseline_and_timeout_fail_closed(tmp_path):
    root = tmp_path / "repo"
    baseline = repository(root)
    assert check_candidate_diff(root, "missing-revision")["returncode"] != 0
    assert check_candidate_diff(root, baseline, timeout=0)["returncode"] == 124


def test_foreign_git_environment_cannot_redirect_check(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    baseline = repository(root)
    foreign = tmp_path / "foreign"
    repository(foreign)
    index = (foreign / ".git/index").read_bytes()
    (root / "new.py").write_text("bad  \n")
    monkeypatch.setenv("GIT_DIR", str(foreign / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(foreign))
    monkeypatch.setenv("GIT_INDEX_FILE", str(foreign / ".git/index"))
    result = check_candidate_diff(root, baseline)
    assert result["returncode"] != 0
    assert "new.py:1: trailing whitespace" in result["output"]
    assert (foreign / ".git/index").read_bytes() == index


def test_new_initialized_submodule_checked_against_empty_tree(tmp_path):
    child_source = tmp_path / "child"
    repository(child_source)
    (child_source / "a.py").write_text("bad  \n")
    git(child_source, "commit", "-qam", "bad child")
    root = tmp_path / "repo"
    baseline = repository(root)
    git(root, "-c", "protocol.file.allow=always", "submodule", "add", "-q", str(child_source), "deps/child")
    result = check_candidate_diff(root, baseline)
    assert result["returncode"] != 0
    assert "[deps/child] a.py:1: trailing whitespace" in result["output"]


def test_committed_child_changes_checked_against_parent_gitlink(tmp_path):
    child_source = tmp_path / "child"
    repository(child_source)
    root = tmp_path / "repo"
    repository(root)
    git(root, "-c", "protocol.file.allow=always", "submodule", "add", "-q", str(child_source), "deps/child")
    git(root, "commit", "-qam", "child")
    baseline = git(root, "rev-parse", "HEAD")
    child = root / "deps/child"
    (child / "a.py").write_text("bad  \n")
    git(child, "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qam", "changed")
    result = check_candidate_diff(root, baseline)
    assert "[deps/child] a.py:1: trailing whitespace" in result["output"]
    assert result["returncode"] != 0


@pytest.mark.parametrize("scheduler_passed", [True, False])
def test_native_validation_includes_gate_and_preserves_prior_failure(tmp_path, scheduler_passed):
    root = tmp_path / "repo"
    baseline = repository(root)
    (root / "new.py").write_text("bad  \n")
    daemon = TodoImplementationDaemon(
        todo_path=root / "todo.md", state_path=tmp_path / "state.json",
        strategy_path=tmp_path / "strategy.json", events_path=tmp_path / "events.jsonl",
        repo_root=root, merge_queue_dir=tmp_path / "queue",
        validation_scheduler=SimpleNamespace(run=lambda *a, **kw: {
            "passed": scheduler_passed, "attempted": True,
            "returncode": 0 if scheduler_passed else 19, "results": [],
        }),
    )
    task = PortalTask(task_id="DIFF-001", title="check candidate", status="todo",
                      completion="manual", priority="P1", track="test",
                      validation=["git diff --check"])
    result = daemon._run_validation_commands(root, task, tmp_path / "validation.log", baseline_ref=baseline)
    assert result["passed"] is False
    if scheduler_passed:
        assert result["reason"] == "candidate_diff_check_failed"
        assert result["candidate_diff_check"]["passed"] is False
        assert "new.py:1: trailing whitespace" in (tmp_path / "validation.log").read_text()
    else:
        assert result["returncode"] == 19
        assert "candidate_diff_check" not in result
