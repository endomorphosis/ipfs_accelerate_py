"""Durable pytest-item ledger carries greens across ephemeral worktrees."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.runtime.pytest_item_ledger import (
    LEDGER_SCHEMA,
    command_fingerprint,
    file_sha256,
    infer_board_workspace,
    ledger_context,
    load_records,
    pytest_item_ledger_dir,
    record_item,
    reusable_nodeids,
    task_id_from_workspace,
    workspace_fingerprint,
)


def _git(workspace: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=workspace, check=True, capture_output=True)


def _seed_worktree(root: Path, *, branch: str) -> Path:
    repo = root / "repo"
    repo.mkdir(parents=True)
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "test")
    (repo / "keep.txt").write_text("keep\n", encoding="utf-8")
    _git(repo, "add", "keep.txt")
    _git(repo, "commit", "-m", "seed")
    worktrees = repo / "data" / "aseh" / "worktrees"
    worktrees.mkdir(parents=True)
    workspace = worktrees / "workspace_one"
    _git(repo, "branch", branch)
    _git(repo, "worktree", "add", str(workspace), branch)
    return workspace


def test_ledger_context_none_outside_board_worktree(tmp_path: Path) -> None:
    (tmp_path / "keep.txt").write_text("x\n", encoding="utf-8")
    assert infer_board_workspace(tmp_path) is None
    assert ledger_context(tmp_path) is None


def test_record_pass_is_reusable_when_hashes_match(tmp_path: Path) -> None:
    branch = "implementation/aseh-061-deadbeef-attempt-1-1"
    workspace = _seed_worktree(tmp_path, branch=branch)
    (workspace / "test_sample.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")
    assert task_id_from_workspace(workspace) == "ASEH-061"
    context = ledger_context(workspace)
    assert context is not None
    digest = file_sha256(workspace / "test_sample.py")
    assert digest is not None
    command = command_fingerprint(["test_sample.py"])
    record_item(
        context["dest"],
        nodeid="test_sample.py::test_ok",
        outcome="passed",
        test_file="test_sample.py",
        test_file_sha256=digest,
        workspace_fingerprint_value=str(context["workspace_fingerprint"]),
        command_sha256=command,
        task_id="ASEH-061",
    )
    records = load_records(context["dest"])
    reusable = reusable_nodeids(
        records,
        command_sha256=command,
        workspace_fingerprint_value=str(context["workspace_fingerprint"]),
        test_file_hashes={"test_sample.py": digest},
    )
    assert reusable == frozenset({"test_sample.py::test_ok"})


def test_hash_or_command_mismatch_is_not_reusable(tmp_path: Path) -> None:
    dest = pytest_item_ledger_dir(tmp_path, "aseh", "ASEH-061")
    record_item(
        dest,
        nodeid="test_sample.py::test_ok",
        outcome="passed",
        test_file="test_sample.py",
        test_file_sha256="abc",
        workspace_fingerprint_value="fp1",
        command_sha256="cmd1",
        task_id="ASEH-061",
    )
    records = load_records(dest)
    assert not reusable_nodeids(
        records,
        command_sha256="cmd1",
        workspace_fingerprint_value="fp1",
        test_file_hashes={"test_sample.py": "other"},
    )
    assert not reusable_nodeids(
        records,
        command_sha256="cmd2",
        workspace_fingerprint_value="fp1",
        test_file_hashes={"test_sample.py": "abc"},
    )
    assert not reusable_nodeids(
        records,
        command_sha256="cmd1",
        workspace_fingerprint_value="fp2",
        test_file_hashes={"test_sample.py": "abc"},
    )


def test_failed_outcome_never_reusable(tmp_path: Path) -> None:
    dest = pytest_item_ledger_dir(tmp_path, "aseh", "ASEH-061")
    record_item(
        dest,
        nodeid="test_sample.py::test_bad",
        outcome="failed",
        test_file="test_sample.py",
        test_file_sha256="abc",
        workspace_fingerprint_value="fp1",
        command_sha256="cmd1",
        task_id="ASEH-061",
    )
    records = load_records(dest)
    assert reusable_nodeids(
        records,
        command_sha256="cmd1",
        workspace_fingerprint_value="fp1",
        test_file_hashes={"test_sample.py": "abc"},
    ) == frozenset()


def test_jsonl_survives_without_sessionfinish(tmp_path: Path) -> None:
    dest = pytest_item_ledger_dir(tmp_path, "aseh", "ASEH-061")
    record_item(
        dest,
        nodeid="a::test_one",
        outcome="passed",
        test_file="a.py",
        test_file_sha256="1",
        workspace_fingerprint_value="fp",
        command_sha256="cmd",
        task_id="ASEH-061",
    )
    record_item(
        dest,
        nodeid="a::test_two",
        outcome="failed",
        test_file="a.py",
        test_file_sha256="1",
        workspace_fingerprint_value="fp",
        command_sha256="cmd",
        task_id="ASEH-061",
    )
    records = load_records(dest)
    assert records["a::test_one"]["outcome"] == "passed"
    assert records["a::test_two"]["outcome"] == "failed"
    assert records["a::test_one"]["schema"] == LEDGER_SCHEMA


def test_corrupt_jsonl_fail_closed(tmp_path: Path) -> None:
    dest = pytest_item_ledger_dir(tmp_path, "aseh", "ASEH-061")
    dest.mkdir(parents=True)
    (dest / "items.jsonl").write_text("{not-json\n", encoding="utf-8")
    assert load_records(dest) == {}


def test_dirty_fingerprint_changes_with_source_edit(tmp_path: Path) -> None:
    branch = "implementation/aseh-061-deadbeef-attempt-1-1"
    workspace = _seed_worktree(tmp_path, branch=branch)
    target = workspace / "src.py"
    target.write_text("one\n", encoding="utf-8")
    first = workspace_fingerprint(workspace)
    target.write_text("two\n", encoding="utf-8")
    second = workspace_fingerprint(workspace)
    assert first != second


def test_plugin_skips_green_and_reruns_failure(tmp_path: Path) -> None:
    branch = "implementation/aseh-061-deadbeef-attempt-1-1"
    workspace = _seed_worktree(tmp_path, branch=branch)
    (workspace / "test_sample.py").write_text(
        "def test_ok():\n    assert True\n\n"
        "def test_bad():\n    assert False\n",
        encoding="utf-8",
    )
    plugin = (
        "ipfs_accelerate_py.agent_supervisor.runtime.pytest_item_ledger_plugin"
    )
    env = {
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "PYTHONPATH": str(Path(__file__).resolve().parents[2]),
    }
    first = subprocess.run(
        [
            "/usr/bin/python3.12",
            "-m",
            "pytest",
            "-p",
            "no:cacheprovider",
            "-p",
            plugin,
            "-q",
            "--tb=no",
            "test_sample.py",
        ],
        cwd=workspace,
        env={**os.environ, **env},
        capture_output=True,
        text=True,
        check=False,
    )
    assert first.returncode != 0
    context = ledger_context(workspace)
    assert context is not None
    records = load_records(context["dest"])
    assert records["test_sample.py::test_ok"]["outcome"] == "passed"
    assert records["test_sample.py::test_bad"]["outcome"] == "failed"

    second = subprocess.run(
        [
            "/usr/bin/python3.12",
            "-m",
            "pytest",
            "-p",
            "no:cacheprovider",
            "-p",
            plugin,
            "-q",
            "--tb=no",
            "test_sample.py",
        ],
        cwd=workspace,
        env={**os.environ, **env},
        capture_output=True,
        text=True,
        check=False,
    )
    combined = (second.stdout or "") + (second.stderr or "")
    assert "skipped" in combined.lower() or "SKIPPED" in combined
    assert second.returncode != 0
