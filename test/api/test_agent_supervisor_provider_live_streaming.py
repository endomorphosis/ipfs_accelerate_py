"""Sparse provider output must reach supervisor logs before the child exits."""

from __future__ import annotations

import io
import os
import subprocess
import sys
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TextIO

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import grok_cli_runner as runner
from ipfs_accelerate_py.agent_supervisor.todo_daemon import supervisor_runtime


class _ObservedOutput(io.StringIO):
    def __init__(self) -> None:
        super().__init__()
        self.flushed = threading.Event()

    def flush(self) -> None:
        self.flushed.set()


@contextmanager
def _provider_pipe(
    stream: Callable[[TextIO, TextIO], None],
) -> Iterator[tuple[int, _ObservedOutput]]:
    """Close the writer and join the reader even after a failed timed wait."""

    read_fd, write_fd = os.pipe()
    destination = _ObservedOutput()
    with os.fdopen(read_fd, "r", encoding="utf-8", errors="replace") as source:
        reader = threading.Thread(target=stream, args=(source, destination), daemon=True)
        reader.start()
        try:
            yield write_fd, destination
        finally:
            os.close(write_fd)
            reader.join(timeout=2)
            assert not reader.is_alive(), "provider stream did not finish after EOF"


@pytest.mark.parametrize(
    "stream",
    (runner._stream_pipe, runner._stream_provider_pipe_without_reserved_records),
)
def test_sparse_progress_is_visible_before_provider_exit(stream) -> None:
    with _provider_pipe(stream) as (write_fd, destination):
        # Keep the provider pipe open: readline/read(size) must not hide this
        # short progress fragment until a newline, 16 KiB, or process exit.
        os.write(write_fd, b"provider is making progress")
        assert destination.flushed.wait(timeout=2), "progress stayed buffered until EOF"
        assert destination.getvalue() == "provider is making progress"


@pytest.mark.parametrize(
    "stream",
    (runner._stream_pipe, runner._stream_provider_pipe_without_reserved_records),
)
def test_live_stream_preserves_split_utf8_sequences(stream) -> None:
    with _provider_pipe(stream) as (write_fd, destination):
        os.write(write_fd, b"progress: \xe2")
        assert destination.flushed.wait(timeout=2)
        assert destination.getvalue() == "progress: "
        destination.flushed.clear()
        os.write(write_fd, b"\x82\xac done\n")
        assert destination.flushed.wait(timeout=2)
        assert destination.getvalue() == "progress: \u20ac done\n"


@pytest.mark.parametrize(
    "prefix",
    (
        runner.GROK_FAILURE_RECEIPT_PREFIX,
        runner.GROK_ROUTE_OUTCOME_PREFIX,
        runner.AGENT_IMPLEMENTATION_ROUTE_OUTCOME_PREFIX,
    ),
)
def test_live_stream_still_escapes_reserved_records_split_across_reads(prefix: str) -> None:
    with _provider_pipe(runner._stream_provider_pipe_without_reserved_records) as (
        write_fd,
        destination,
    ):
        os.write(write_fd, prefix[:7].encode())
        assert destination.flushed.wait(timeout=2)
        assert destination.getvalue() == ""
        destination.flushed.clear()
        os.write(write_fd, (prefix[7:] + '{}\nordinary\rprogress\u2028ok\n').encode())
        assert destination.flushed.wait(timeout=2)
        assert destination.getvalue() == (
            "[provider-child-output-escaped] " + prefix + '{}\n'
            "ordinary[provider-child-control-0d]progress[provider-child-control-2028]ok\n"
        )


@pytest.mark.parametrize("returncode", (0, 17))
def test_typed_capture_replays_both_pipes_before_wait_finishes(
    monkeypatch: pytest.MonkeyPatch,
    returncode: int,
) -> None:
    destinations = (_ObservedOutput(), _ObservedOutput())
    pipes = (os.pipe(), os.pipe())
    visible_before_exit = []
    with (
        os.fdopen(pipes[0][0], "r", encoding="utf-8", errors="replace") as stdout,
        os.fdopen(pipes[1][0], "r", encoding="utf-8", errors="replace") as stderr,
    ):
        class Provider:
            def wait(self) -> int:
                try:
                    visible_before_exit.extend(
                        output.flushed.wait(timeout=2) for output in destinations
                    )
                finally:
                    for _, write_fd in pipes:
                        os.close(write_fd)
                return returncode

        provider = Provider()
        provider.stdout, provider.stderr = stdout, stderr
        os.write(pipes[0][1], b"stdout progress\n")
        os.write(pipes[1][1], b"stderr progress\n")
        monkeypatch.setattr(runner.subprocess, "Popen", lambda *_args, **_kwargs: provider)
        with monkeypatch.context() as capture:
            capture.setattr(runner.sys, "stdout", destinations[0])
            capture.setattr(runner.sys, "stderr", destinations[1])
            result = runner._run_grok_with_typed_failure_capture(["fixture-grok"], env={})
        assert result == returncode
        assert visible_before_exit == [True, True]
        assert [output.getvalue() for output in destinations] == [
            "stdout progress\n", "stderr progress\n"
        ]


def test_provider_progress_keeps_the_native_hard_timeout_and_reaps_child(tmp_path: Path) -> None:
    processes = []
    with (tmp_path / "provider.log").open("w") as log:
        with pytest.raises(subprocess.TimeoutExpired):
            supervisor_runtime.run_process_group_stream(
                [sys.executable, "-u", "-c", "import time; print('progress'); time.sleep(30)"],
                cwd=tmp_path,
                stdout=log,
                timeout_seconds=0.5,
                termination_grace_seconds=0.1,
                on_started=processes.append,
            )
    assert len(processes) == 1
    assert processes[0].poll() is not None
    assert "progress" in (tmp_path / "provider.log").read_text()


@pytest.mark.parametrize("accepted_runner", (False, True))
@pytest.mark.parametrize("enable_fallback", (False, True))
def test_grok_command_builder_uses_canonical_or_explicit_accepted_runner(
    tmp_path, monkeypatch, accepted_runner, enable_fallback,
) -> None:
    monkeypatch.setattr(
        runner.subprocess, "Popen",
        lambda *_args, **_kwargs: pytest.fail("command assembly started a process"),
    )
    monkeypatch.setattr(
        runner, "resolve_codex_quota_fallback_executable",
        lambda **_kwargs: "/fixture/codex",
    )
    accepted = tmp_path / "accepted-runner.py"
    accepted.write_text("# accepted runner generation\n")
    command = runner.build_grok_quota_routed_agent_command(
        workspace=tmp_path,
        python_executable=sys.executable,
        enable_codex_fallback=enable_fallback,
        accepted_runner_path=accepted if accepted_runner else "",
    )
    expected_prefix = (
        [sys.executable, "-I", str(accepted)]
        if accepted_runner
        else [
            sys.executable, "-m",
            "ipfs_accelerate_py.agent_supervisor.runtime.grok_cli_runner",
        ]
    )
    assert command[:3] == expected_prefix
    assert command[3:5] == ["--workspace", str(tmp_path)]
    assert ("--codex-fallback-command-json" in command) is enable_fallback


def test_daemon_grok_command_reaches_canonical_runtime(tmp_path, monkeypatch) -> None:
    from ipfs_accelerate_py.agent_supervisor.todo_daemon import implementation_daemon

    monkeypatch.setattr(
        runner.subprocess, "Popen",
        lambda *_args, **_kwargs: pytest.fail("command assembly started a process"),
    )
    monkeypatch.setattr(implementation_daemon, "_grok_binary", lambda: "/fixture/grok")
    monkeypatch.setattr(implementation_daemon, "_grok_cli_available", lambda: True)
    command = implementation_daemon._grok_cli_command(
        workspace_path=tmp_path, enable_codex_fallback=False,
    )
    assert command[:3] == [
        sys.executable, "-m",
        "ipfs_accelerate_py.agent_supervisor.runtime.grok_cli_runner",
    ]


def test_merge_resolver_delegates_default_canonical_command_once(tmp_path, monkeypatch) -> None:
    from ipfs_accelerate_py.agent_supervisor.integrations import llm_merge_resolver_fallback

    captured = []
    closed = []

    class CheckoutLock:
        def close(self):
            closed.append(True)

    def acquire_lock(workspace):
        assert workspace == tmp_path
        return CheckoutLock()

    def accepted_runner_main(argv):
        captured.append((list(argv), sys.stdin.read()))
        return 23

    monkeypatch.setenv(llm_merge_resolver_fallback._INVOCATION_DEPTH_ENV, "0")
    monkeypatch.setattr(llm_merge_resolver_fallback, "_acquire_git_lock", acquire_lock)
    monkeypatch.setattr(runner, "main", accepted_runner_main)
    monkeypatch.setattr(runner, "resolve_codex_quota_fallback_executable", lambda **_kwargs: "/fixture/codex")
    monkeypatch.setattr(sys, "stdin", io.StringIO("resolve this conflict"))
    monkeypatch.setattr(
        runner.subprocess, "Popen",
        lambda *_args, **_kwargs: pytest.fail("merge delegation started a provider"),
    )

    assert llm_merge_resolver_fallback.main([str(tmp_path)]) == 23
    assert closed == [True]
    assert len(captured) == 1
    argv, prompt = captured[0]
    assert argv[:2] == ["--workspace", str(tmp_path)]
    assert prompt == "resolve this conflict"
    assert runner.CANONICAL_LEGACY_PREFLIGHT_ROUTE_FLAG in argv
