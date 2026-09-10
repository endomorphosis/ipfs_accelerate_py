"""Durable pytest-item ledger carries greens across ephemeral worktrees."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime.pytest_item_ledger import (
    LEDGER_SCHEMA,
    command_fingerprint,
    file_sha256,
    git_worktree_root,
    infer_board_workspace,
    is_empty_int_failure_text,
    is_pid_marker_retry_nodeid,
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


def test_git_worktree_root_walks_up_from_test_subdir(tmp_path: Path) -> None:
    branch = "implementation/aseh-061-deadbeef-attempt-1-1"
    workspace = _seed_worktree(tmp_path, branch=branch)
    nested = workspace / "test" / "api"
    nested.mkdir(parents=True)
    assert git_worktree_root(nested) == workspace.resolve()


def test_ledger_context_none_outside_board_worktree(tmp_path: Path) -> None:
    (tmp_path / "keep.txt").write_text("x\n", encoding="utf-8")
    assert infer_board_workspace(tmp_path) is None
    assert ledger_context(tmp_path) is None


def test_ledger_context_uses_env_when_git_is_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ipfs_accelerate_py.agent_supervisor.runtime.pytest_item_ledger import (
        write_task_id_marker,
    )

    branch = "implementation/aseh-061-deadbeef-attempt-1-1"
    workspace = _seed_worktree(tmp_path, branch=branch)
    monkeypatch.setenv("ASEH_TASK_ID", "ASEH-061")

    def boom(*_args, **_kwargs):
        raise OSError("landlock denied git")

    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.runtime.pytest_item_ledger.subprocess.run",
        boom,
    )
    context = ledger_context(workspace)
    assert context is not None
    assert context["task_id"] == "ASEH-061"
    write_task_id_marker(workspace, "ASEH-061")
    monkeypatch.delenv("ASEH_TASK_ID", raising=False)
    marked = ledger_context(workspace)
    assert marked is not None
    assert marked["task_id"] == "ASEH-061"


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


def test_record_writes_workspace_local_when_durable_unwritable(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    blocked = tmp_path / "blocked-file"
    blocked.write_text("not-a-dir\n", encoding="utf-8")
    record_item(
        blocked / "ledger",
        nodeid="test_sample.py::test_ok",
        outcome="passed",
        test_file="test_sample.py",
        test_file_sha256="abc",
        workspace_fingerprint_value="fp",
        command_sha256="cmd",
        task_id="ASEH-061",
        workspace=workspace,
    )
    from ipfs_accelerate_py.agent_supervisor.runtime.pytest_item_ledger import (
        WORKSPACE_LEDGER_NAME,
        load_records,
    )

    assert (workspace / WORKSPACE_LEDGER_NAME).is_file()
    records = load_records(blocked / "ledger", workspace=workspace)
    assert records["test_sample.py::test_ok"]["outcome"] == "passed"


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


def test_ivp_selection_skips_unrelated_test_file(tmp_path: Path) -> None:
    from ipfs_accelerate_py.agent_supervisor.runtime.board_pytest_selection import (
        select_affected_test_files,
    )

    branch = "implementation/aseh-061-deadbeef-attempt-1-1"
    workspace = _seed_worktree(tmp_path, branch=branch)
    (workspace / "ipfs_accelerate_py").mkdir()
    (workspace / "ipfs_accelerate_py" / "__init__.py").write_text("", encoding="utf-8")
    (workspace / "ipfs_accelerate_py" / "mod.py").write_text("VALUE = 1\n", encoding="utf-8")
    tests = workspace / "test"
    tests.mkdir()
    (tests / "test_hit.py").write_text(
        "from ipfs_accelerate_py.mod import VALUE\n\ndef test_hit():\n    assert VALUE == 1\n",
        encoding="utf-8",
    )
    (tests / "test_miss.py").write_text(
        "def test_miss():\n    assert True\n",
        encoding="utf-8",
    )
    _git(workspace, "add", "ipfs_accelerate_py", "test")
    _git(workspace, "commit", "-m", "tests")
    (workspace / "ipfs_accelerate_py" / "mod.py").write_text("VALUE = 2\n", encoding="utf-8")
    affected = select_affected_test_files(
        workspace,
        ("test/test_hit.py", "test/test_miss.py"),
    )
    assert affected is not None
    assert "test/test_hit.py" in affected
    assert "test/test_miss.py" not in affected


def test_pid_marker_retry_helpers_match_empty_int_and_nodeids() -> None:
    node = (
        "test/api/test_agent_supervisor_configured_typed_grant_handoff.py"
        "::test_aseh_forced_owner_group_escalation_reaps_term_ignoring_tree"
    )
    assert is_pid_marker_retry_nodeid(node)
    assert is_pid_marker_retry_nodeid(
        "test/api/test_agent_supervisor_configured_typed_grant_handoff.py"
        "::test_aseh_scheduler_group_fence_survives_leader_exit"
    )
    assert not is_pid_marker_retry_nodeid(
        "test/api/test_agent_supervisor_configured_typed_grant_handoff.py"
        "::test_aseh_r45_receipt_id_reuses_memoized_validation_contracts"
    )
    assert is_empty_int_failure_text(
        "ValueError: invalid literal for int() with base 10: ''"
    )
    assert is_empty_int_failure_text(
        'FAILED test_foo - ValueError: invalid literal for int() with base 10: ""'
    )
    assert not is_empty_int_failure_text(
        "ValueError: invalid literal for int() with base 10: '12'"
    )


def test_pid_marker_plugin_retries_empty_int_then_passes() -> None:
    from types import SimpleNamespace

    from ipfs_accelerate_py.agent_supervisor.runtime import pytest_item_ledger_plugin

    calls = {"n": 0}

    def runtest() -> None:
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError("invalid literal for int() with base 10: ''")

    forced: list[object] = []
    outcome = SimpleNamespace(
        excinfo=SimpleNamespace(
            value=ValueError("invalid literal for int() with base 10: ''")
        ),
        force_result=forced.append,
    )
    item = SimpleNamespace(
        nodeid=(
            "test/api/test_agent_supervisor_configured_typed_grant_handoff.py"
            "::test_aseh_forced_owner_group_escalation_reaps_term_ignoring_tree"
        ),
        runtest=runtest,
    )
    gen = pytest_item_ledger_plugin.pytest_runtest_call(item)
    gen.send(None)
    try:
        gen.send(outcome)
    except StopIteration:
        pass
    assert calls["n"] == 2
    assert forced == [None]


def test_ivp_selection_mentions_dirty_non_python(tmp_path: Path) -> None:
    from ipfs_accelerate_py.agent_supervisor.runtime.board_pytest_selection import (
        select_affected_test_files,
    )

    branch = "implementation/aseh-061-deadbeef-attempt-1-1"
    workspace = _seed_worktree(tmp_path, branch=branch)
    docs = workspace / "docs"
    docs.mkdir()
    (docs / "note.md").write_text("migration\n", encoding="utf-8")
    tests = workspace / "test"
    tests.mkdir()
    (tests / "test_doc.py").write_text(
        "def test_doc():\n    assert 'note.md' in 'note.md'\n",
        encoding="utf-8",
    )
    (tests / "test_other.py").write_text("def test_other():\n    assert True\n", encoding="utf-8")
    _git(workspace, "add", "docs", "test")
    _git(workspace, "commit", "-m", "docs")
    (docs / "note.md").write_text("migration changed\n", encoding="utf-8")
    affected = select_affected_test_files(
        workspace,
        ("test/test_doc.py", "test/test_other.py"),
    )
    assert affected is not None
    assert "test/test_doc.py" in affected
    assert "test/test_other.py" not in affected
