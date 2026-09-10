"""Exact process arguments preserve custody without granting task authority."""

from pathlib import Path


import pytest


from ipfs_accelerate_py.agent_supervisor.todo_daemon import supervisor as workers


GROK = ("/usr/bin/python3", "-m", "ipfs_accelerate_py.agent_supervisor.grok_cli_runner")


@pytest.mark.parametrize(
    "argv",
    [
        GROK,
        ("/usr/bin/grok",),
        ("node", "/opt/grok"),
        ("python3", "-u", "/opt/grok_cli_runner.py"),
        (*GROK, "--prompt", ""),
        ("/usr/bin/grok", "--prompt", ""),
        ("python3", "-uB", "-m", GROK[2]),
    ],
)
def test_exact_argv_recognizes_worker_when_ps_text_is_missing(monkeypatch, argv):
    item = {"pid": 41002, "cmdline": "", "argv": argv, "start_ticks": 900}
    monkeypatch.setattr(workers, "descendant_processes", lambda pid: [item])
    assert workers.active_codex_exec_workers(41001) == [item]


@pytest.mark.parametrize(
    "argv",
    [
        ("python3", "-c", "-m", GROK[2]),
        ("echo", "grok_cli_runner"),
        ("python3", "-m", "pytest", GROK[2]),
        ("python-not-an-interpreter", "-m", GROK[2]),
        "grok",
        (),
        ("grok", None),
        ("grok", "bad\0argument"),
    ],
)
def test_exact_argv_cannot_be_overridden_by_worker_text(monkeypatch, argv):
    item = {"pid": 41002, "cmdline": "/usr/bin/grok", "argv": argv}
    monkeypatch.setattr(workers, "descendant_processes", lambda pid: [item])
    assert workers.active_codex_exec_workers(41001) == []


def test_procfs_argument_reader_preserves_empty_trailing_argument(monkeypatch):
    argv = (*GROK, "--prompt", "")
    raw = b"\0".join(part.encode() for part in argv) + b"\0"

    def read(path):
        assert path == Path("/proc/41002/cmdline")
        return raw

    monkeypatch.setattr(Path, "read_bytes", read)
    assert workers._process_command_argv(41002) == argv
