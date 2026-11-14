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


def test_controlled_timeout_returns_before_withheld_eof_is_released(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """UNIT transport evidence only; no process or physical custody is modeled."""
    from dataclasses import asdict
    from io import BytesIO
    from math import isfinite
    from threading import Event, Thread

    from ipfs_accelerate_py.agent_supervisor.todo_daemon import engine

    command = ("controlled-unit-command",)
    partial_stdout = "partial café\r\nstandalone\rcarriage\n".encode("utf-8")
    partial_stderr = (
        "diagnostic café\r\nstandalone\rcarriage\n" * 400
    ).encode("utf-8")
    eof_released = Event()
    finished = Event()
    results = []
    failures = []

    class WithheldEOFProcess:
        # This is an inert interface value, never an operating-system target.
        stdin = None
        pid = 8675309
        returncode = -9

        def __init__(self):
            self.stdout = BytesIO(partial_stdout)
            self.stderr = BytesIO(partial_stderr)

        def communicate(self, input=None, timeout=None):
            if timeout is not None:
                assert isfinite(timeout) and timeout >= 0, (
                    "controlled transport requires a finite nonnegative timeout"
                )
                # Advance the controlled transport's deadline without sleeping.
                raise subprocess.TimeoutExpired(
                    command, timeout, output=partial_stdout, stderr=partial_stderr
                )
            if not eof_released.wait(10.0):
                raise AssertionError("finite controlled-transport test guard expired")
            return partial_stdout, partial_stderr

        def terminate(self):
            pass

        def kill(self):
            pass

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            return self.returncode

    process = WithheldEOFProcess()
    monkeypatch.setattr(engine.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(engine.os, "killpg", lambda *args: None, raising=False)

    def invoke():
        try:
            results.append(
                run_command(
                    command, cwd=tmp_path, timeout_seconds=1,
                    preserve_output_newlines=True,
                )
            )
        except BaseException as exc:
            failures.append(exc)
        finally:
            finished.set()

    worker = Thread(target=invoke, daemon=True)
    worker.start()
    try:
        assert finished.wait(2.0), (
            "run_command must return a timeout failure while EOF remains withheld; "
            "the test rescue has not released the controlled transport"
        )
        assert not eof_released.is_set()
        if failures:
            raise failures[0]
        result = results[0]
        assert set(asdict(result)) == {"command", "returncode", "stdout", "stderr"}
        assert result.command == command
        assert result.returncode == 124
        assert result.stdout == partial_stdout.decode("utf-8")
        expected_stderr = partial_stderr.decode("utf-8")
        diagnostic_prefix, separator, after_prefix = result.stderr.partition("\n")
        assert separator
        assert "incomplete" in diagnostic_prefix.casefold()
        assert any(
            word in diagnostic_prefix.casefold() for word in ("output", "capture")
        )
        assert expected_stderr in after_prefix
        assert result.stderr.count(expected_stderr) == 1
        assert "Command timed out after 1s" in result.stderr
        # The public compact API must keep the status ahead of large child stderr.
        for compact in (result.compact(), result.compact(limit=160)):
            compact_stderr = compact["stderr"].casefold()
            assert "incomplete" in compact_stderr
            assert any(word in compact_stderr for word in ("output", "capture"))
    finally:
        # Rescue happens only after the observable completion assertion.
        eof_released.set()
        worker.join(timeout=2.0)
        process.stdout.close()
        process.stderr.close()
        assert not worker.is_alive(), "controlled unit-test worker did not finish"


@pytest.mark.parametrize("exact", (False, True))
def test_controlled_success_preserves_output_and_four_field_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, exact: bool,
) -> None:
    """Model Popen's text translation without creating a subprocess."""
    from dataclasses import asdict
    from types import SimpleNamespace

    from ipfs_accelerate_py.agent_supervisor.todo_daemon import engine

    command = ("controlled-unit-command",)
    raw_stdout = "successful café\r\nstandalone\rcarriage\n".encode("utf-8")
    raw_stderr = "diagnostic café\r\nstandalone\rcarriage\n".encode("utf-8")

    def text_output(raw):
        return raw.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")

    def popen(*args, **kwargs):
        output = (
            (text_output(raw_stdout), text_output(raw_stderr))
            if kwargs["text"] else (raw_stdout, raw_stderr)
        )
        return SimpleNamespace(returncode=0, communicate=lambda **kwargs: output)

    monkeypatch.setattr(engine.subprocess, "Popen", popen)
    monkeypatch.setattr(engine.os, "killpg", lambda *args: None, raising=False)
    result = run_command(
        command, cwd=tmp_path, timeout_seconds=1, preserve_output_newlines=exact
    )
    assert asdict(result) == {
        "command": command,
        "returncode": 0,
        "stdout": raw_stdout.decode("utf-8") if exact else text_output(raw_stdout),
        "stderr": raw_stderr.decode("utf-8") if exact else text_output(raw_stderr),
    }


def _controlled_timeout_transport(
    monkeypatch, snapshots, *, completed_at=None, completed_output=None,
    drain_seconds=0.0, pipe_close_seconds=0.0, leader_exits=True,
):
    """UNIT interface model only; no OS process, signal, or global clock change."""
    from io import BytesIO
    from math import isfinite
    from types import SimpleNamespace

    from ipfs_accelerate_py.agent_supervisor.todo_daemon import engine

    class Clock:
        now = 0.0

        def monotonic(self):
            return self.now

    clock = Clock()

    class CapturePipe(BytesIO):
        def close(self):
            if not self.closed:
                clock.now += pipe_close_seconds
            super().close()

    class Process:
        pid = 8675309
        stdin = None
        returncode = None
        wait_budget = None
        leader_reaped = False

        def __init__(self):
            self.stdout = CapturePipe()
            self.stderr = CapturePipe()
            self.phase = 0

        def communicate(self, input=None, timeout=None):
            assert timeout is not None and isfinite(timeout) and timeout >= 0, (
                "the controlled transport requires a finite nonnegative deadline"
            )
            phase = self.phase
            self.phase += 1
            assert phase < 3, "finite controlled transport transcript exhausted"
            if (completed_at == "term" and phase == 1) or (
                completed_at == "kill" and phase == 2
            ):
                self.returncode = -9
                self.stdout.close()
                self.stderr.close()
                return completed_output
            if phase == 2:
                clock.now += min(drain_seconds, timeout)
            stdout, stderr = snapshots[phase]
            raise subprocess.TimeoutExpired(
                ("controlled-unit-command",), timeout, output=stdout, stderr=stderr
            )

        def terminate(self):
            pass

        def kill(self):
            pass

        def wait(self, timeout=None):
            assert timeout is not None and isfinite(timeout) and timeout >= 0, (
                "the direct leader wait requires a finite nonnegative deadline"
            )
            self.wait_budget = timeout
            if leader_exits:
                self.returncode = -9
                self.leader_reaped = True
                return self.returncode
            clock.now += timeout
            raise subprocess.TimeoutExpired(("controlled-unit-command",), timeout)

    process = Process()
    monkeypatch.setattr(engine.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(engine.os, "killpg", lambda *args: None, raising=False)
    monkeypatch.setattr(engine, "time", SimpleNamespace(monotonic=clock.monotonic))
    return process, clock


@pytest.mark.parametrize("exact", (False, True))
def test_controlled_incomplete_timeout_keeps_latest_partial_and_prefix(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, exact: bool,
) -> None:
    command = ("controlled-unit-command",)
    latest_stdout = "  latest café\r\nstandalone\rcarriage\n ".encode("utf-8")
    latest_stderr = ("  latest diagnostic café\r\nstandalone\rcarriage\n " * 400).encode("utf-8")
    process, _ = _controlled_timeout_transport(
        monkeypatch,
        ((b"obsolete initial", b"obsolete initial error"),
         (b"obsolete term", b"obsolete term error"),
         (latest_stdout, latest_stderr)),
    )
    result = run_command(
        command, cwd=tmp_path, timeout_seconds=1, preserve_output_newlines=exact
    )
    expected_stdout = latest_stdout.decode("utf-8")
    expected_stderr = latest_stderr.decode("utf-8")
    if not exact:
        expected_stdout = expected_stdout.replace("\r\n", "\n").replace("\r", "\n")
        expected_stderr = expected_stderr.replace("\r\n", "\n").replace("\r", "\n")
    assert result.command == command
    assert result.returncode == 124
    assert result.stdout == expected_stdout
    prefix, separator, after_prefix = result.stderr.partition("\n")
    assert separator and "incomplete" in prefix.casefold()
    assert "eof" in prefix.casefold() and "unknown" in prefix.casefold()
    assert after_prefix.startswith(expected_stderr)
    assert result.stderr.count(expected_stderr) == 1
    assert "obsolete" not in result.stdout + result.stderr
    assert "Command timed out after 1s" in result.stderr
    for compact in (result.compact(), result.compact(limit=160)):
        assert "incomplete" in compact["stderr"].casefold()
        assert "unknown" in compact["stderr"].casefold()
    assert process.stdout.closed and process.stderr.closed


@pytest.mark.parametrize("final", ((None, None), (b"", b"")), ids=("missing", "empty"))
def test_controlled_timeout_missing_snapshot_retains_prior_but_empty_replaces_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, final,
) -> None:
    _controlled_timeout_transport(
        monkeypatch,
        ((b"obsolete initial", b"obsolete initial error"),
         (b"latest term\r\n", b"latest term error\r\n"), final),
    )
    result = run_command(
        ("controlled-unit-command",), cwd=tmp_path, timeout_seconds=1,
        preserve_output_newlines=True,
    )
    expected_stdout = "latest term\r\n" if final[0] is None else ""
    expected_stderr = "latest term error\r\n" if final[1] is None else ""
    assert result.returncode == 124
    assert result.stdout == expected_stdout
    assert "obsolete" not in result.stdout + result.stderr
    assert "incomplete" in result.stderr.partition("\n")[0].casefold()
    if expected_stderr:
        assert result.stderr.partition("\n")[2].startswith(expected_stderr)
    else:
        assert "latest term" not in result.stderr


@pytest.mark.parametrize("exact", (False, True))
def test_controlled_incomplete_utf8_is_replaced_in_text_and_strict_in_exact_mode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, exact: bool,
) -> None:
    process, _ = _controlled_timeout_transport(
        monkeypatch, ((None, None), (None, None), (b"valid prefix\xc3", b"diagnostic\xff\r\n")),
    )
    if exact:
        with pytest.raises(UnicodeDecodeError):
            run_command(
                ("controlled-unit-command",), cwd=tmp_path, timeout_seconds=1,
                preserve_output_newlines=True,
            )
    else:
        result = run_command(
            ("controlled-unit-command",), cwd=tmp_path, timeout_seconds=1
        )
        assert result.returncode == 124
        assert result.stdout == "valid prefix\ufffd"
        assert result.stderr.partition("\n")[2].startswith("diagnostic\ufffd\n")
        assert "incomplete" in result.stderr.partition("\n")[0].casefold()
    assert process.stdout.closed and process.stderr.closed


def test_controlled_cleanup_wait_uses_deadline_remaining_after_drain_and_close(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    process, clock = _controlled_timeout_transport(
        monkeypatch, ((None, None), (None, None), (b"partial", b"diagnostic")),
        drain_seconds=2.0, pipe_close_seconds=1.0, leader_exits=False,
    )
    result = run_command(
        ("controlled-unit-command",), cwd=tmp_path, timeout_seconds=1
    )
    # Two drain seconds and one per capture close already spent four seconds.
    # The existing direct leader may use only the remainder of the one cutoff.
    assert process.wait_budget is not None and 0.0 <= process.wait_budget <= 1.0
    assert clock.now <= 5.0
    assert result.returncode == 124
    prefix = result.stderr.partition("\n")[0].casefold()
    assert "incomplete" in prefix and "unknown" in prefix
    assert "direct child" in prefix and not process.leader_reaped
    assert result.stdout == "partial"


@pytest.mark.parametrize("completed_at", ("term", "kill"))
def test_controlled_timeout_with_complete_capture_preserves_legacy_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, completed_at: str,
) -> None:
    from dataclasses import asdict

    command = ("controlled-unit-command",)
    _controlled_timeout_transport(
        monkeypatch, ((b"early", b"early error"), (b"later", b"later error"), (None, None)),
        completed_at=completed_at, completed_output=("captured\ntext\n", "legacy diagnostic\n"),
    )
    result = run_command(command, cwd=tmp_path, timeout_seconds=1)
    assert asdict(result) == {
        "command": command, "returncode": 124, "stdout": "captured\ntext\n",
        "stderr": "legacy diagnostic\n\nCommand timed out after 1s",
    }



def _typed_wait_observation(**overrides):
    """Construct bounded UNIT data; this helper supplies no custody authority."""
    from ipfs_accelerate_py.agent_supervisor.todo_daemon import engine

    fields = {
        "schema": "command-execution-observation/v1",
        "completion_kind": "completed",
        "process_pid": 8675309,
        "process_returncode": 0,
        "leader_waited": True,
        "output_capture": "complete",
        "work_deadline_monotonic": None,
        "cleanup_deadline_monotonic": None,
        "observed_at_monotonic": 0.0,
    }
    fields.update(overrides)
    return engine.CommandExecutionObservation(**fields)


def test_typed_wait_observation_preserves_four_field_result_contract() -> None:
    from dataclasses import FrozenInstanceError, asdict, fields, replace

    from ipfs_accelerate_py.agent_supervisor.todo_daemon import engine

    observation = _typed_wait_observation()
    old = CommandResult(("controlled-unit-command",), 0, "out", "err")
    result = CommandResult(
        old.command, old.returncode, old.stdout, old.stderr,
        execution_observation=observation,
    )
    assert [item.name for item in fields(result)] == [
        "command", "returncode", "stdout", "stderr",
    ]
    assert CommandResult.__match_args__ == ("command", "returncode", "stdout", "stderr")
    assert asdict(result) == asdict(old)
    assert result == old and hash(result) == hash(old)
    assert repr(result) == repr(old) and result.ok and old.ok
    assert result.compact() == old.compact()
    assert old.execution_observation is None
    assert result.execution_observation is observation
    assert replace(result, stdout="changed") == CommandResult(old.command, 0, "changed", "err")
    assert replace(result).execution_observation is None
    # An older unpickled instance has only the original four attributes.
    legacy = CommandResult.__new__(CommandResult)
    assert legacy.execution_observation is None
    with pytest.raises(FrozenInstanceError):
        result.execution_observation = None
    with pytest.raises(FrozenInstanceError):
        observation.output_capture = "incomplete"
    assert observation.as_dict() == asdict(observation)
    assert set(observation.as_dict()) == {
        "schema", "completion_kind", "process_pid", "process_returncode",
        "leader_waited", "output_capture", "work_deadline_monotonic",
        "cleanup_deadline_monotonic", "observed_at_monotonic",
    }
    assert engine.command_result_from_object(result).execution_observation is observation
    assert engine.command_result_from_object(
        {**asdict(old), "execution_observation": observation}
    ).execution_observation is None


@pytest.mark.parametrize("returncode", (0, 124))
def test_controlled_completed_observation_distinguishes_normal_exit_124(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, returncode: int,
) -> None:
    from types import SimpleNamespace

    from ipfs_accelerate_py.agent_supervisor.todo_daemon import engine

    process = SimpleNamespace(
        pid=8675309, returncode=returncode,
        communicate=lambda **kwargs: ("complete output", "complete diagnostic"),
    )
    monkeypatch.setattr(engine.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(engine, "time", SimpleNamespace(monotonic=lambda: 4.0))
    result = run_command(("controlled-unit-command",), cwd=tmp_path, timeout_seconds=1)
    observation = result.execution_observation
    assert type(observation) is engine.CommandExecutionObservation
    assert observation.schema == "command-execution-observation/v1"
    assert observation.completion_kind == "completed"
    assert observation.process_pid == process.pid
    assert observation.process_returncode == result.returncode == returncode
    assert observation.leader_waited is True and observation.output_capture == "complete"
    assert observation.work_deadline_monotonic is None
    assert observation.cleanup_deadline_monotonic is None
    assert observation.observed_at_monotonic == 4.0


@pytest.mark.parametrize("completed_at", ("term", "kill"))
def test_controlled_timeout_observation_keeps_policy_and_actual_codes_distinct(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, completed_at: str,
) -> None:
    process, _ = _controlled_timeout_transport(
        monkeypatch, ((b"early", b"early error"), (b"later", b"later error"), (None, None)),
        completed_at=completed_at, completed_output=("complete", "diagnostic"),
    )
    result = run_command(("controlled-unit-command",), cwd=tmp_path, timeout_seconds=1)
    assert result.returncode == 124 and process.returncode == -9
    observation = result.execution_observation
    assert result.returncode == 124 and observation.process_returncode == process.returncode == -9
    assert observation.completion_kind == "timeout"
    assert observation.leader_waited is True and observation.output_capture == "complete"


def test_controlled_reaped_leader_does_not_turn_closed_pipes_into_complete_capture(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    prefix = b'{"status":"apparently_complete"}'
    process, _ = _controlled_timeout_transport(
        monkeypatch, ((None, None), (None, None), (prefix, b"diagnostic")),
    )
    result = run_command(("controlled-unit-command",), cwd=tmp_path, timeout_seconds=1)
    observation = result.execution_observation
    assert result.stdout == prefix.decode("utf-8")
    assert process.leader_reaped and process.stdout.closed and process.stderr.closed
    assert observation.process_returncode == -9 and observation.leader_waited is True
    assert observation.completion_kind == "timeout" and observation.output_capture == "incomplete"


def test_controlled_returncode_alone_does_not_claim_genuine_leader_wait(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    process, _ = _controlled_timeout_transport(
        monkeypatch, ((None, None), (None, None), (b"prefix", b"diagnostic")),
        leader_exits=False,
    )
    # A returncode attribute is data; only communicate/wait completion proves wait.
    process.returncode = -9
    result = run_command(("controlled-unit-command",), cwd=tmp_path, timeout_seconds=1)
    assert not process.leader_reaped
    assert result.execution_observation.process_returncode == -9
    assert result.execution_observation.leader_waited is None
    assert result.execution_observation.output_capture == "incomplete"


@pytest.mark.parametrize(
    "work,cleanup", ((True, 20.0), (10.0, False), (float("nan"), 20.0),
                     (10.0, float("inf")), (20.0, 10.0), (10.0, None), (None, 20.0)),
)
def test_controlled_deadline_mode_rejects_invalid_original_deadlines_before_spawn(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, work, cleanup,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.todo_daemon import engine

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid deadline must not create even a modeled child")

    monkeypatch.setattr(engine.subprocess, "Popen", forbidden)
    with pytest.raises(ValueError, match="deadline"):
        run_command(
            ("controlled-unit-command",), cwd=tmp_path, timeout_seconds=1,
            work_deadline_monotonic=work, cleanup_deadline_monotonic=cleanup,
        )


def test_controlled_exhausted_work_deadline_refuses_before_spawn(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    from types import SimpleNamespace

    from ipfs_accelerate_py.agent_supervisor.todo_daemon import engine

    def forbidden(*args, **kwargs):
        raise AssertionError("exhausted work deadline must not create a modeled child")

    monkeypatch.setattr(engine.subprocess, "Popen", forbidden)
    monkeypatch.setattr(engine, "time", SimpleNamespace(monotonic=lambda: 10.0))
    result = run_command(
        ("controlled-unit-command",), cwd=tmp_path, timeout_seconds=1,
        work_deadline_monotonic=10.0, cleanup_deadline_monotonic=20.0,
    )
    observation = result.execution_observation
    assert result.returncode == 124 and observation.completion_kind == "not_spawned"
    assert observation.process_pid is None and observation.process_returncode is None
    assert observation.leader_waited is None and observation.output_capture == "unknown"
    assert observation.work_deadline_monotonic == 10.0
    assert observation.cleanup_deadline_monotonic == 20.0
    assert observation.observed_at_monotonic == 10.0


@pytest.mark.parametrize("spawn_seconds,expected_work", ((3.0, 7.0), (12.0, 0.0)))
def test_controlled_work_budget_is_recomputed_after_modeled_spawn_latency(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, spawn_seconds: float, expected_work: float,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.todo_daemon import engine

    process, clock = _controlled_timeout_transport(
        monkeypatch, ((None, None), (None, None), (b"prefix", b"diagnostic")),
    )
    budgets = []
    communicate = process.communicate

    def record_communicate(input=None, timeout=None):
        budgets.append(timeout)
        return communicate(input=input, timeout=timeout)

    def popen(*args, **kwargs):
        clock.now += spawn_seconds
        return process

    monkeypatch.setattr(process, "communicate", record_communicate)
    monkeypatch.setattr(engine.subprocess, "Popen", popen)
    result = run_command(
        ("controlled-unit-command",), cwd=tmp_path, timeout_seconds=10,
        work_deadline_monotonic=10.0, cleanup_deadline_monotonic=20.0,
    )
    assert budgets[0] == expected_work
    assert all(0.0 <= budget <= 5.0 for budget in budgets[1:])
    assert result.execution_observation.cleanup_deadline_monotonic == 20.0
    assert result.execution_observation.observed_at_monotonic == spawn_seconds


def test_controlled_cleanup_never_renews_original_cutoff_and_records_close_overrun(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    process, clock = _controlled_timeout_transport(
        monkeypatch, ((None, None), (None, None), (b"prefix", b"diagnostic")),
        drain_seconds=2.0, pipe_close_seconds=1.0, leader_exits=False,
    )
    communicate = process.communicate
    budgets = []

    def record_communicate(input=None, timeout=None):
        budgets.append(timeout)
        if process.phase == 1:
            clock.now += 2.0
        return communicate(input=input, timeout=timeout)

    monkeypatch.setattr(process, "communicate", record_communicate)
    result = run_command(
        ("controlled-unit-command",), cwd=tmp_path, timeout_seconds=1,
        work_deadline_monotonic=1.0, cleanup_deadline_monotonic=3.0,
    )
    assert budgets == [1.0, 3.0, 1.0]
    assert process.wait_budget == 0.0
    observation = result.execution_observation
    assert observation.cleanup_deadline_monotonic == 3.0
    assert observation.observed_at_monotonic == clock.now == 5.0
    assert observation.leader_waited is None and observation.output_capture == "incomplete"


def test_controlled_spawn_error_preserves_exception_in_explicit_deadline_mode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    from types import SimpleNamespace

    from ipfs_accelerate_py.agent_supervisor.todo_daemon import engine

    def fail_spawn(*args, **kwargs):
        raise OSError("controlled spawn failure")

    monkeypatch.setattr(engine.subprocess, "Popen", fail_spawn)
    monkeypatch.setattr(engine, "time", SimpleNamespace(monotonic=lambda: 0.0))
    with pytest.raises(OSError, match="controlled spawn failure"):
        run_command(("controlled-unit-command",), cwd=tmp_path, timeout_seconds=1)
    with pytest.raises(OSError, match="controlled spawn failure"):
        run_command(
            ("controlled-unit-command",), cwd=tmp_path, timeout_seconds=1,
            work_deadline_monotonic=10.0, cleanup_deadline_monotonic=20.0,
        )
