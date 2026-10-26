"""Resume arguments respect native CLI protocols; no stores or models are used."""

from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon import allocation_compat


def _session_lookup(monkeypatch, provider="grok_cli", **session):
    calls = []

    def lookup(**kwargs):
        calls.append(kwargs)
        return {"provider": provider, **session}

    monkeypatch.setattr(allocation_compat, "supervisor_generate_kwargs", lookup)
    return calls


@pytest.mark.parametrize("provider", ["", "auto", "grok", "grok_cli"])
def test_custom_planning_worker_keeps_exact_python_protocol(monkeypatch, provider):
    controller = Path(__file__).resolve().parents[1]
    prefix = controller / "docs/architecture/legal_ir_grok_parallel_20261016"
    command = [
        "/usr/bin/python3", "-B", str(prefix / "planning_worker.py"),
        "--controller-prefix", str(prefix),
        "--controller-manifest-sha256", "fixture-manifest-digest",
        "--receipt-root", "/tmp/fixture-worker-receipts",
    ]
    calls = _session_lookup(monkeypatch, session_id="existing-grok-session")

    assert allocation_compat.apply_resume_argv(
        command, provider=provider, task_id="LIR-9001"
    ) == command
    assert calls == []


@pytest.mark.parametrize("command", [
    ["python3", "-m", "ipfs_accelerate_py.agent_supervisor.runtime.grok_cli_runner"],
    ["/usr/bin/env", "grok", "--prompt-file", "prompt.txt"],
    ["bash", "-c", "grok --prompt-file prompt.txt"],
    ["/opt/bin/custom-grok-wrapper", "--prompt-file", "prompt.txt"],
    [],
])
def test_custom_wrappers_skip_native_session_lookup(monkeypatch, command):
    calls = _session_lookup(monkeypatch, session_id="existing-grok-session")
    assert allocation_compat.apply_resume_argv(command, provider="grok") == command
    assert calls == []


@pytest.mark.parametrize(("executable", "provider", "router"), [
    ("grok", "grok", "grok_cli"),
    ("/opt/bin/grok", "grok-cli", "grok_cli"),
    ("grok", "xai_cli", "grok_cli"),
    ("claude", "claude", "claude_code"),
    ("/opt/bin/claude", "claude-code", "claude_code"),
    ("copilot", "copilot", "copilot_cli"),
    ("/opt/bin/copilot", "copilot_cli", "copilot_cli"),
])
def test_matching_native_cli_receives_resume(monkeypatch, executable, provider, router):
    command = [executable, "--prompt-file", "prompt.txt"]
    original = list(command)
    calls = _session_lookup(monkeypatch, router, resume_session_id="native-session")

    assert allocation_compat.apply_resume_argv(
        command, provider=provider, task_id="TASK-9", allocation_session_id="allocation-9"
    ) == [executable, "--resume", "native-session", "--prompt-file", "prompt.txt"]
    assert command == original
    assert calls == [{"task_id": "TASK-9", "provider": provider,
                      "allocation_session_id": "allocation-9"}]


@pytest.mark.parametrize("provider", ["", "auto", "efficient"])
def test_auto_provider_must_resolve_to_matching_native_cli(monkeypatch, provider):
    command = ["grok", "--prompt-file", "prompt.txt"]
    calls = _session_lookup(monkeypatch, "claude_code", session_id="claude-session")
    assert allocation_compat.apply_resume_argv(command, provider=provider) == command
    assert len(calls) == 1


def test_declared_provider_mismatch_skips_session_lookup(monkeypatch):
    command = ["grok", "--prompt-file", "prompt.txt"]
    calls = _session_lookup(monkeypatch, "claude_code", session_id="claude-session")
    assert allocation_compat.apply_resume_argv(command, provider="claude") == command
    assert calls == []


@pytest.mark.parametrize("provider", ["", "auto", "efficient"])
def test_auto_provider_can_resume_matching_native_cli(monkeypatch, provider):
    _session_lookup(monkeypatch, session_id="grok-session")
    assert allocation_compat.apply_resume_argv(["grok"], provider=provider) == [
        "grok", "--resume", "grok-session"
    ]


@pytest.mark.parametrize(("command", "provider", "router"), [
    (["grok", "--resume", "explicit"], "grok", "grok_cli"),
    (["claude", "--resume=explicit"], "claude", "claude_code"),
    (["copilot", "--resume=explicit"], "copilot", "copilot_cli"),
    (["codex", "exec", "resume", "explicit"], "codex", "codex_cli"),
    (["muse", "exec", "--session-id", "explicit"], "muse_code", "muse_code"),
    (["muse", "exec", "--session-id=explicit"], "muse_cli", "muse_code"),
])
def test_explicit_native_resume_is_preserved(monkeypatch, command, provider, router):
    _session_lookup(monkeypatch, router, session_id="different-session")
    assert allocation_compat.apply_resume_argv(command, provider=provider) == command


@pytest.mark.parametrize(("command", "provider", "router", "expected"), [
    (["/opt/bin/codex", "exec", "--json", "prompt"], "codex_cli", "codex_cli",
     ["/opt/bin/codex", "exec", "resume", "native-session", "--json", "prompt"]),
    (["/opt/bin/muse", "exec", "prompt"], "muse-code", "muse_code",
     ["/opt/bin/muse", "exec", "--session-id", "native-session", "prompt"]),
])
def test_other_supported_native_resume_protocols(monkeypatch, command, provider, router, expected):
    _session_lookup(monkeypatch, router, chat_session_id="native-session")
    assert allocation_compat.apply_resume_argv(command, provider=provider) == expected


def test_native_cli_without_resume_metadata_is_unchanged(monkeypatch):
    _session_lookup(monkeypatch)
    command = ["grok", "--prompt-file", "prompt.txt"]
    assert allocation_compat.apply_resume_argv(command, provider="grok") == command
