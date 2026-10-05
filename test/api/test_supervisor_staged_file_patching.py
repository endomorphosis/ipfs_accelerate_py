"""Real Git regressions for non-mutating supervisor candidate patch collection."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon.engine import (
    CommandResult,
    command_runner_from_legacy_function,
    run_command,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.worktrees import worktree_diff


def git(root: Path, *args: str, stdin: str | None = None) -> str:
    result = subprocess.run(
        [
            "git",
            "--no-optional-locks",
            "-c",
            "user.name=Supervisor test",
            "-c",
            "user.email=supervisor@example.invalid",
            *args,
        ],
        cwd=root,
        input=stdin,
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    root = tmp_path / "candidate"
    root.mkdir()
    git(root, "init", "--initial-branch=main")
    for name in (
        "a.txt",
        "remove.txt",
        "rename.txt",
        "unrelated.txt",
        "bracket[1].txt",
        "bracket1.txt",
        "white space.txt",
        ":literal.txt",
        "line\nbreak.txt",
        "unicodé.txt",
    ):
        (root / name).write_text(f"baseline {name}\n", encoding="utf-8")
    (root / "binary.dat").write_bytes(b"\x00baseline\xff")
    git(root, "add", "--all")
    git(root, "commit", "-m", "baseline")
    return root


def file_contents(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file() and ".git" not in path.relative_to(root).parts
    }


def collect_unchanged(root: Path, paths: list[str], **kwargs) -> str:
    index = Path(
        git(root, "rev-parse", "--path-format=absolute", "--git-path", "index").strip()
    )
    index_before = index.read_bytes()
    head_before = git(root, "rev-parse", "HEAD")
    contents_before = file_contents(root)
    patch = worktree_diff(worktree_path=root, paths=paths, **kwargs)
    assert index.read_bytes() == index_before
    assert git(root, "rev-parse", "HEAD") == head_before
    assert file_contents(root) == contents_before
    return patch


def replay(root: Path, patch: str, target: Path) -> dict[str, bytes]:
    git(root, "worktree", "add", "--detach", str(target), "HEAD")
    git(target, "apply", "--check", "-", stdin=patch)
    git(target, "apply", "-", stdin=patch)
    # The linked worktree's .git file is intentionally excluded above.
    return file_contents(target)


@pytest.mark.parametrize("staging", ("unstaged", "staged", "mixed"))
def test_final_patch_covers_staged_unstaged_rename_delete_and_binary(
    repository: Path,
    tmp_path: Path,
    staging: str,
) -> None:
    root = repository
    (root / "a.txt").write_text("first candidate\n")
    (root / "remove.txt").unlink()
    (root / "rename.txt").rename(root / "renamed.txt")
    (root / "binary.dat").write_bytes(b"\x00changed\xfe")
    (root / "new-binary.dat").write_bytes(b"\x00new\xff")
    (root / "empty.txt").touch()
    if staging in {"staged", "mixed"}:
        git(root, "add", "--all")
    if staging == "mixed":
        (root / "a.txt").write_text("final candidate\n")
        (root / "new-binary.dat").write_bytes(b"\x00new final\xfe")
    patch = collect_unchanged(root, ["."])
    assert patch
    assert collect_unchanged(root, ["."]) == patch
    assert replay(root, patch, tmp_path / "replayed") == file_contents(root)


def test_literal_names_and_selected_scope_preserve_unrelated_staging(
    repository: Path,
    tmp_path: Path,
) -> None:
    root = repository
    baseline = file_contents(root)
    selected = [
        "bracket[1].txt",
        "white space.txt",
        ":literal.txt",
        "line\nbreak.txt",
        "unicodé.txt",
    ]
    for name in selected + ["bracket1.txt", "unrelated.txt"]:
        (root / name).write_text(f"changed {name}\n", encoding="utf-8")
    git(root, "add", "--", ":(literal)bracket[1].txt", ":(literal)unrelated.txt")
    # Untracked names also require literal selection and NUL-separated discovery.
    for name in ["new[1].txt", "new1.txt", ":new.txt", "new line\nname.txt"]:
        (root / name).write_text(f"new {name}\n", encoding="utf-8")
    selected += ["new[1].txt", ":new.txt", "new line\nname.txt"]
    patch = collect_unchanged(root, selected)
    expected = {**baseline, **{name: (root / name).read_bytes() for name in selected}}
    assert replay(root, patch, tmp_path / "replayed") == expected


def test_untracked_directory_respects_ignored_files_and_no_external_diff(
    repository: Path,
    tmp_path: Path,
) -> None:
    root = repository
    baseline = file_contents(root)
    (root / "new-dir").mkdir()
    (root / "new-dir" / "included.txt").write_text("included\n")
    (root / "new-dir" / "ignored.txt").write_text("ignored\n")
    # Repository-local ignore rules do not become part of the proposed patch.
    (root / ".git" / "info" / "exclude").write_text("new-dir/ignored.txt\n")
    marker = tmp_path / "external-diff-ran"
    driver = tmp_path / "diff-driver"
    driver.write_text(f"#!/bin/sh\ntouch '{marker}'\nexit 1\n")
    driver.chmod(0o755)
    git(root, "config", "diff.external", str(driver))
    git(root, "config", "diff.test.textconv", str(driver))
    git(root, "config", "color.ui", "always")
    git(root, "config", "diff.noprefix", "true")
    (root / ".git" / "info" / "attributes").write_text("*.txt diff=test\n")
    (root / "a.txt").write_text("changed\n")
    patch = collect_unchanged(root, ["a.txt", "new-dir/"])
    assert not marker.exists()
    expected = {
        **baseline,
        "a.txt": b"changed\n",
        "new-dir/included.txt": b"included\n",
    }
    assert replay(root, patch, tmp_path / "replayed") == expected


def test_staged_edit_reverted_in_worktree_emits_no_net_patch(repository: Path) -> None:
    root = repository
    before = (root / "a.txt").read_bytes()
    (root / "a.txt").write_text("staged only\n")
    git(root, "add", "--", "a.txt")
    (root / "a.txt").write_bytes(before)
    assert collect_unchanged(root, ["a.txt"]) == ""
    assert git(root, "diff", "--cached", "--", "a.txt")


@pytest.mark.parametrize("staged", (False, True))
def test_patch_preserves_crlf_and_standalone_cr_bytes(
    repository: Path,
    tmp_path: Path,
    staged: bool,
) -> None:
    root = repository
    (root / "a.txt").write_bytes(b"candidate\r\nline\r\nstandalone\rcarriage\n")
    (root / "new.txt").write_bytes(b"new\r\nstandalone\rcarriage\n")
    if staged:
        git(root, "add", "--", "a.txt", "new.txt")
    patch = collect_unchanged(root, ["a.txt", "new.txt"])
    assert "candidate\r\n" in patch
    assert replay(root, patch, tmp_path / "replayed") == file_contents(root)


def test_literal_backslash_and_boundary_spaces_are_not_normalized(
    repository: Path,
    tmp_path: Path,
) -> None:
    root = repository
    names = ["literal\\name.txt", " leading.txt", "trailing.txt "]
    for name in names:
        (root / name).write_bytes(b"new\r\n")
    patch = collect_unchanged(root, names)
    assert replay(root, patch, tmp_path / "replayed") == file_contents(root)


def test_undecodable_text_patch_fails_without_changing_index(repository: Path) -> None:
    root = repository
    (root / "a.txt").write_bytes(b"text without NUL but invalid UTF-8 \xff\n")
    index = Path(
        git(root, "rev-parse", "--path-format=absolute", "--git-path", "index").strip()
    )
    before = index.read_bytes()
    with pytest.raises(UnicodeDecodeError):
        worktree_diff(worktree_path=root, paths=["a.txt"])
    assert index.read_bytes() == before


def test_runner_without_exact_output_capability_fails_closed(repository: Path) -> None:
    def legacy_runner(command, *, cwd, timeout_seconds):
        raise AssertionError("an incompatible runner must not execute")

    with pytest.raises(TypeError, match="preserve_output_newlines"):
        worktree_diff(
            worktree_path=repository, paths=["a.txt"], run_command_fn=legacy_runner
        )


def test_legacy_adapter_replays_large_exact_newline_patch(
    repository: Path,
    tmp_path: Path,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.legal_parser_daemon import (
        _run_command,
    )

    root = repository
    (root / "a.txt").write_bytes(b"tracked CRLF\r\nstandalone\rsegment\n" * 1600)
    git(root, "add", "--", "a.txt")
    (root / "new.txt").write_bytes(b"new CRLF\r\nstandalone\rsegment\n" * 1600)
    runner = command_runner_from_legacy_function(_run_command)
    patch = collect_unchanged(root, ["a.txt", "new.txt"], run_command_fn=runner)
    assert len(patch) > 12000
    assert replay(root, patch, tmp_path / "replayed") == file_contents(root)


def test_legacy_adapter_requires_declared_exact_output_capability(
    repository: Path,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.validation.validation_runtime import (
        ValidationRuntimeError,
    )

    def old_runner(command, *, cwd, timeout, environment=None):
        raise AssertionError(
            "legacy adaptation must not silently normalize patch bytes"
        )

    runner = command_runner_from_legacy_function(old_runner)
    with pytest.raises(ValidationRuntimeError, match="preserve exact output"):
        worktree_diff(worktree_path=repository, paths=["a.txt"], run_command_fn=runner)


def test_exact_output_option_preserves_existing_default_text_behavior(
    repository: Path,
) -> None:
    import sys

    from ipfs_accelerate_py.agent_supervisor.todo_daemon.legal_parser_daemon import (
        _run_command,
    )

    command = (
        sys.executable,
        "-c",
        "import sys; sys.stdout.buffer.write(b'x\\r\\ny\\rz' * 5000)",
    )
    ordinary = run_command(command, cwd=repository, timeout_seconds=10)
    exact = run_command(
        command, cwd=repository, timeout_seconds=10, preserve_output_newlines=True
    )
    assert ordinary.stdout == "x\ny\nz" * 5000
    assert exact.stdout == "x\r\ny\rz" * 5000
    ordinary_legacy = _run_command(command, cwd=repository)
    exact_legacy = _run_command(command, cwd=repository, preserve_output_newlines=True)
    assert ordinary_legacy["stdout"] == ordinary.stdout[-12000:]
    assert exact_legacy["stdout"] == exact.stdout


@pytest.mark.parametrize(
    "failure_command", ("status", "ls-files", "tracked-diff", "untracked-diff")
)
def test_collection_failure_is_closed_and_retry_preserves_index(
    repository: Path,
    tmp_path: Path,
    failure_command: str,
) -> None:
    root = repository
    (root / "a.txt").write_text("staged\n")
    git(root, "add", "--", "a.txt")
    (root / "new.txt").write_text("untracked\n")

    def failing_runner(command, **kwargs):
        kind = (
            "status"
            if "status" in command
            else "ls-files"
            if "ls-files" in command
            else "untracked-diff"
            if "--no-index" in command
            else "tracked-diff"
        )
        if kind == failure_command:
            return CommandResult(
                tuple(command), 124, "partial diagnostic", "test read failure"
            )
        return run_command(command, **kwargs)

    trace = {}
    assert (
        collect_unchanged(
            root, ["a.txt", "new.txt"], run_command_fn=failing_runner, raw_trace=trace
        )
        == ""
    )
    assert trace
    patch = collect_unchanged(root, ["a.txt", "new.txt"])
    assert replay(root, patch, tmp_path / "replayed") == file_contents(root)


def test_untracked_file_disappearing_mid_collection_keeps_index(
    repository: Path,
    tmp_path: Path,
) -> None:
    root = repository
    (root / "a.txt").write_text("staged\n")
    git(root, "add", "--", "a.txt")
    (root / "new.txt").write_text("untracked\n")
    index = Path(
        git(root, "rev-parse", "--path-format=absolute", "--git-path", "index").strip()
    )
    before = index.read_bytes()

    def disappearing_runner(command, **kwargs):
        if "--no-index" in command:
            (root / "new.txt").unlink()
        return run_command(command, **kwargs)

    assert (
        worktree_diff(
            worktree_path=root,
            paths=["a.txt", "new.txt"],
            run_command_fn=disappearing_runner,
        )
        == ""
    )
    assert index.read_bytes() == before
    patch = collect_unchanged(root, ["a.txt", "new.txt"])
    assert replay(root, patch, tmp_path / "replayed") == file_contents(root)


@pytest.mark.parametrize("path", ("../outside", "/outside", "a\x00b"))
def test_unsafe_collection_paths_are_rejected_before_any_git_read(
    repository: Path, path: str
) -> None:
    def forbidden_runner(*args, **kwargs):
        raise AssertionError("unsafe paths must not reach Git")

    with pytest.raises(ValueError, match="unsafe"):
        worktree_diff(
            worktree_path=repository, paths=[path], run_command_fn=forbidden_runner
        )
