"""Real Git stat-cache changes must not mutate read-only source observations."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from scripts import run_agent_supervisor_efficiency_state_hardening as operator
from scripts import validate_agent_supervisor_efficiency_state_hardening_board as validator
from ipfs_accelerate_py.agent_supervisor.runtime import configured_board_scheduler as scheduler
from test.api.test_agent_supervisor_configured_board_scheduler import _seed_configured_repo


def git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["/usr/bin/git", *args],
        cwd=root,
        env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout.strip()


def index_binding(root: Path) -> tuple:
    path = Path(git(root, "rev-parse", "--git-path", "index"))
    if not path.is_absolute():
        path = root / path
    data = path.read_bytes()
    stat = path.stat()
    return (
        stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns,
        stat.st_ctime_ns, hashlib.sha256(data).hexdigest(),
    )


def invalidate_cached_stat(path: Path) -> None:
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 2_000_000_000))


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    repo = tmp_path / "repository"
    repo.mkdir()
    git(repo, "init", "--quiet", "--initial-branch=main")
    git(repo, "config", "user.name", "Readonly Git Test")
    git(repo, "config", "user.email", "readonly@example.invalid")
    (repo / "tracked.txt").write_text("unchanged source\n")
    git(repo, "add", "tracked.txt")
    git(repo, "commit", "--quiet", "-m", "seed")
    return repo


@pytest.mark.parametrize("reader", ["operator_text", "operator_bytes", "validator", "clean_gate"])
@pytest.mark.parametrize("ambient", ["0", "1"])
def test_readonly_git_observers_preserve_exact_index_after_stat_drift(
    repository: Path, monkeypatch, reader: str, ambient: str,
) -> None:
    invalidate_cached_stat(repository / "tracked.txt")
    before = index_binding(repository)
    monkeypatch.setenv("GIT_OPTIONAL_LOCKS", ambient)
    monkeypatch.setattr(operator, "ROOT", repository)
    if reader == "operator_text":
        assert operator._git("status", "--porcelain=v1") == ""
    elif reader == "operator_bytes":
        assert operator._git_bytes("status", "--porcelain=v1") == b""
    elif reader == "validator":
        result = validator._git("status", "--porcelain=v1", cwd=repository)
        assert result.returncode == 0 and result.stdout == ""
    else:
        operator._assert_clean_tree(SimpleNamespace(merge_target_branch="main"))
    assert index_binding(repository) == before


@pytest.mark.parametrize("reader", ["operator_text", "operator_bytes", "validator"])
def test_readonly_git_still_reports_content_changes_without_refresh(
    repository: Path, monkeypatch, reader: str,
) -> None:
    (repository / "tracked.txt").write_text("changed source\n")
    before = index_binding(repository)
    monkeypatch.setattr(operator, "ROOT", repository)
    if reader == "operator_text":
        text = operator._git("status", "--porcelain=v1")
    elif reader == "operator_bytes":
        text = operator._git_bytes("status", "--porcelain=v1").decode()
    else:
        text = validator._git("status", "--porcelain=v1", cwd=repository).stdout
    assert "M tracked.txt" in text
    assert index_binding(repository) == before


def test_native_configured_preflight_preserves_parent_and_submodule_indexes(
    tmp_path: Path, monkeypatch,
) -> None:
    root, config = _seed_configured_repo(tmp_path)
    board = scheduler.load_configured_board(config, repo_root=root)
    invalidate_cached_stat(root / "README.md")
    invalidate_cached_stat(root / "dependency/dependency.txt")
    before = {path: index_binding(path) for path in (root, root / "dependency")}
    monkeypatch.setenv("GIT_OPTIONAL_LOCKS", "1")
    result = scheduler.preflight_configured_board(board)
    assert result["valid"], result["errors"]
    assert {path: index_binding(path) for path in before} == before


@pytest.mark.parametrize("runner", ["trusted", "validation"])
def test_intentional_index_and_checkout_mutations_still_work(
    repository: Path, monkeypatch, runner: str,
) -> None:
    monkeypatch.setattr(operator, "ROOT", repository)

    def mutate(*args: str) -> None:
        if runner == "trusted":
            operator._git(*args)
        else:
            result = operator._run(("/usr/bin/git", *args), cwd=repository)
            assert result.returncode == 0, result.stderr

    before = index_binding(repository)
    (repository / "new.txt").write_text("intentional staged source\n")
    mutate("add", "new.txt")
    assert index_binding(repository) != before
    assert operator._git("show", ":new.txt") == "intentional staged source"
    (repository / "tracked.txt").write_text("working mutation\n")
    mutate("checkout", "--", "tracked.txt")
    assert (repository / "tracked.txt").read_text() == "unchanged source\n"


@pytest.mark.parametrize("executable", ["git", "/usr/bin/git"])
@pytest.mark.parametrize("explicit_env", [False, True])
def test_operator_validation_diff_preserves_index(
    repository: Path, monkeypatch, executable: str, explicit_env: bool,
) -> None:
    invalidate_cached_stat(repository / "tracked.txt")
    before = index_binding(repository)
    monkeypatch.setenv("GIT_OPTIONAL_LOCKS", "1")
    result = operator._run(
        (executable, "diff", "--check"),
        cwd=repository,
        env=dict(os.environ) if explicit_env else None,
    )
    assert result.returncode == 0
    assert index_binding(repository) == before


@pytest.mark.parametrize("explicit_env", [False, True])
def test_non_git_command_environment_remains_unchanged(
    repository: Path, monkeypatch, explicit_env: bool,
) -> None:
    monkeypatch.setenv("GIT_OPTIONAL_LOCKS", "1")
    supplied = {"GIT_OPTIONAL_LOCKS": "explicit-value"}
    result = operator._run(
        ("/bin/sh", "-c", 'printf %s "$GIT_OPTIONAL_LOCKS"'),
        cwd=repository,
        env=supplied if explicit_env else None,
    )
    assert result.returncode == 0
    assert result.stdout == ("explicit-value" if explicit_env else "1")
    assert supplied == {"GIT_OPTIONAL_LOCKS": "explicit-value"}


@pytest.mark.parametrize("reader", ["operator_text", "operator_bytes", "validator", "scheduler"])
def test_readonly_diff_helpers_preserve_index(repository: Path, monkeypatch, reader: str) -> None:
    invalidate_cached_stat(repository / "tracked.txt")
    before = index_binding(repository)
    monkeypatch.setenv("GIT_OPTIONAL_LOCKS", "1")
    monkeypatch.setattr(operator, "ROOT", repository)
    if reader == "operator_text":
        assert operator._git("diff", "--check") == ""
    elif reader == "operator_bytes":
        assert operator._git_bytes("diff", "--check") == b""
    else:
        if reader == "validator":
            result = validator._git("diff", "--check", cwd=repository)
        else:
            result = scheduler._git_run(("diff", "--check"), cwd=repository)
        assert result.returncode == 0 and result.stdout == ""
    assert index_binding(repository) == before
