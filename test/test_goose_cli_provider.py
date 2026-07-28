"""Offline tests for the canonical Goose CLI adapter and structured parsers.

Every execution path uses a fake executable fixture. No live network and no
real Goose binary are required.
"""

from __future__ import annotations

import json
import os
import stat
import sys
import textwrap
from pathlib import Path
from typing import Any, Optional

import pytest

from ipfs_accelerate_py.cli_runtime.contracts import (
    CLIRequest,
    ExecutionMode,
)
from ipfs_accelerate_py.cli_runtime.errors import (
    CLIRuntimeErrorCode,
    ContractValidationError,
    InvalidStateError,
    MalformedOutputError,
    PolicyDeniedError,
)
from ipfs_accelerate_py.cli_runtime.process_runner import ProcessRunner
from ipfs_accelerate_py.cli_runtime.providers import goose as goose_mod
from ipfs_accelerate_py.cli_runtime.providers.goose import (
    DEFAULT_CHAT_MAX_TOOL_REPETITIONS,
    DEFAULT_CHAT_MAX_TURNS,
    OUTPUT_FORMAT_JSON,
    OUTPUT_FORMAT_STREAM_JSON,
    PINNED_GOOSE_VERSION,
    REQUIRED_CHAT_SAFETY_FLAGS,
    GooseAgentPolicy,
    GooseCLIProvider,
    GooseErrorKind,
    GooseProviderError,
    GooseVersionCapabilities,
    build_goose_command,
    capabilities_for_version,
    capabilities_from_help,
    classify_goose_failure,
    create_goose_provider,
    goose_error_code,
    goose_provider_spec,
    parse_goose_json,
    parse_goose_output,
    parse_goose_stream_json,
    parse_version_tuple,
)


# ---------------------------------------------------------------------------
# Fake executable fixtures
# ---------------------------------------------------------------------------


def _write_fake_goose(
    directory: Path,
    *,
    script: str,
    name: str = "goose",
) -> Path:
    """Write an executable fake goose script and return its path."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(script, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


def _json_success_script(
    text: str = "hello from goose",
    *,
    include_tool: bool = False,
    exit_code: int = 0,
    stderr: str = "",
    version: str = PINNED_GOOSE_VERSION,
) -> str:
    """Fake goose that validates chat safety flags and emits JSON."""
    tool_block = ""
    if include_tool:
        tool_block = """,
        {
          "type": "tool_use",
          "id": "t1",
          "name": "developer__shell",
          "input": {"command": "echo hi"}
        }"""
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "prompt"}],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": text},
                ]
                + (
                    [
                        {
                            "type": "tool_use",
                            "id": "t1",
                            "name": "developer__shell",
                            "input": {"command": "echo hi"},
                        }
                    ]
                    if include_tool
                    else []
                ),
            },
        ],
        "metadata": {
            "total_tokens": 12,
            "input_tokens": 4,
            "output_tokens": 8,
            "status": "completed",
        },
    }
    # Build script carefully with embedded JSON.
    body = json.dumps(payload)
    return textwrap.dedent(
        f"""\
        #!{sys.executable}
        import json, os, sys

        argv = sys.argv[1:]
        if "--version" in argv or "-V" in argv:
            print("goose {version}")
            sys.exit(0)

        if not argv or argv[0] != "run":
            print("expected run subcommand", file=sys.stderr)
            sys.exit(2)

        # Echo argv for assertions via a side channel when GOOSE_FAKE_ARGV is set.
        if os.environ.get("GOOSE_FAKE_ARGV_PATH"):
            with open(os.environ["GOOSE_FAKE_ARGV_PATH"], "w", encoding="utf-8") as fh:
                json.dump({{"argv": argv, "env_mode": os.environ.get("GOOSE_MODE"),
                           "env_provider": os.environ.get("GOOSE_PROVIDER"),
                           "env_model": os.environ.get("GOOSE_MODEL"),
                           "env_path_root": os.environ.get("GOOSE_PATH_ROOT"),
                           "cwd": os.getcwd()}}, fh)

        required = ["--no-session", "--no-profile", "--output-format",
                    "--max-turns", "--max-tool-repetitions"]
        # Chat safety: only enforce no-profile when GOOSE_MODE=chat
        mode = os.environ.get("GOOSE_MODE", "")
        if mode == "chat":
            for flag in required:
                if flag not in argv:
                    print(f"missing required flag: {{flag}}", file=sys.stderr)
                    sys.exit(3)
            if "--with-builtin" in argv or "--with-extension" in argv:
                print("chat must not enable extensions", file=sys.stderr)
                sys.exit(3)

        # instructions on stdin
        if "--instructions" not in argv and "-i" not in argv:
            print("missing --instructions", file=sys.stderr)
            sys.exit(3)
        # read stdin (prompt)
        _ = sys.stdin.read()

        fmt = "json"
        if "--output-format" in argv:
            fmt = argv[argv.index("--output-format") + 1]

        payload = json.loads({body!r})
        if fmt == "stream-json":
            for msg in payload.get("messages", []):
                print(json.dumps({{"type": "message", "message": msg}}))
            meta = payload.get("metadata") or {{}}
            print(json.dumps({{"type": "complete", **meta}}))
        else:
            print(json.dumps(payload))

        if {stderr!r}:
            print({stderr!r}, file=sys.stderr)
        sys.exit({exit_code})
        """
    )


def _error_script(
    *,
    stderr: str,
    exit_code: int = 1,
    version: str = PINNED_GOOSE_VERSION,
) -> str:
    return textwrap.dedent(
        f"""\
        #!{sys.executable}
        import sys
        argv = sys.argv[1:]
        if "--version" in argv or "-V" in argv:
            print("goose {version}")
            sys.exit(0)
        _ = sys.stdin.read() if not sys.stdin.isatty() else ""
        print({stderr!r}, file=sys.stderr)
        sys.exit({exit_code})
        """
    )


def _version_only_script(version: str) -> str:
    return textwrap.dedent(
        f"""\
        #!{sys.executable}
        import sys
        if "--version" in sys.argv or "-V" in sys.argv:
            print("goose {version}")
            sys.exit(0)
        print("run not implemented", file=sys.stderr)
        sys.exit(2)
        """
    )


@pytest.fixture()
def fake_bin(tmp_path: Path) -> Path:
    return tmp_path / "bin"


# ---------------------------------------------------------------------------
# Import / packaging safety
# ---------------------------------------------------------------------------


def test_import_providers_is_side_effect_free(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("provider import must not discover or install")

    monkeypatch.setattr(goose_mod, "discover_goose", boom)
    monkeypatch.setattr(goose_mod, "ensure_goose", boom)
    # Re-import surface
    from ipfs_accelerate_py.cli_runtime.providers import (
        GooseCLIProvider,
        create_goose_provider,
        goose_provider_spec,
    )

    spec = goose_provider_spec()
    assert spec.name == "goose_cli"
    assert "goose" in spec.aliases
    provider = create_goose_provider()
    assert isinstance(provider, GooseCLIProvider)
    assert provider.executable is None


# ---------------------------------------------------------------------------
# Version capabilities / gates
# ---------------------------------------------------------------------------


def test_parse_version_tuple():
    assert parse_version_tuple("1.44.0") == (1, 44, 0)
    assert parse_version_tuple("v1.8.0") == (1, 8, 0)
    assert parse_version_tuple("goose 1.12.3") == (1, 12, 3)
    assert parse_version_tuple("nope") == (0, 0, 0)


def test_capabilities_for_pinned_version_supports_all_safety_flags():
    caps = capabilities_for_version(PINNED_GOOSE_VERSION)
    assert caps.missing_required_chat_flags() == ()
    caps.ensure_chat_safe()  # does not raise


def test_old_version_fails_closed_on_missing_safety_flags():
    caps = capabilities_for_version("1.0.0")
    missing = caps.missing_required_chat_flags()
    assert "--no-profile" in missing
    assert "--output-format" in missing
    assert "--max-tool-repetitions" in missing
    with pytest.raises(GooseProviderError) as excinfo:
        caps.ensure_chat_safe()
    assert excinfo.value.kind is GooseErrorKind.UNSUPPORTED_VERSION
    assert goose_error_code(excinfo.value.kind) is CLIRuntimeErrorCode.UNSUPPORTED_CAPABILITY


def test_capabilities_from_help_fail_closed_when_flags_absent():
    help_text = "Usage: goose run\n  --max-turns <N>\n  -i, --instructions <FILE>\n"
    caps = capabilities_from_help(help_text, version="ancient")
    assert caps.supports_max_turns is True
    assert caps.supports_no_session is False
    assert caps.supports_no_profile is False
    with pytest.raises(GooseProviderError) as excinfo:
        build_goose_command(
            executable="/usr/bin/goose",
            mode=ExecutionMode.CHAT,
            capabilities=caps,
        )
    assert excinfo.value.kind is GooseErrorKind.UNSUPPORTED_VERSION


def test_build_chat_command_includes_all_required_safety_flags():
    plan = build_goose_command(
        executable="/opt/goose",
        mode=ExecutionMode.CHAT,
        model_name="muse-spark-1.1",
        goose_provider="openai",
        capabilities=capabilities_for_version(PINNED_GOOSE_VERSION),
    )
    argv = list(plan.argv)
    assert argv[0] == "/opt/goose"
    assert argv[1] == "run"
    for flag in REQUIRED_CHAT_SAFETY_FLAGS:
        if flag == "--instructions":
            assert "--instructions" in argv
            assert argv[argv.index("--instructions") + 1] == "-"
        else:
            assert flag in argv, f"missing {flag}"
    assert plan.env["GOOSE_MODE"] == "chat"
    assert plan.model_name == "muse-spark-1.1"
    assert plan.goose_provider == "openai"
    assert "--provider" in argv and argv[argv.index("--provider") + 1] == "openai"
    assert "--model" in argv and argv[argv.index("--model") + 1] == "muse-spark-1.1"
    assert "--with-builtin" not in argv
    assert "--with-extension" not in argv
    assert plan.max_turns == DEFAULT_CHAT_MAX_TURNS
    assert plan.max_tool_repetitions == DEFAULT_CHAT_MAX_TOOL_REPETITIONS
    assert plan.required_flags_present()


def test_chat_caps_max_turns_cannot_exceed_default_bound():
    plan = build_goose_command(
        executable="goose",
        mode=ExecutionMode.CHAT,
        max_turns=999,
        max_tool_repetitions=50,
        capabilities=capabilities_for_version(PINNED_GOOSE_VERSION),
    )
    assert plan.max_turns == DEFAULT_CHAT_MAX_TURNS
    assert plan.max_tool_repetitions == DEFAULT_CHAT_MAX_TOOL_REPETITIONS


def test_chat_rejects_builtins_and_extensions():
    with pytest.raises(InvalidStateError):
        build_goose_command(
            executable="goose",
            mode=ExecutionMode.CHAT,
            builtins=("developer",),
            capabilities=capabilities_for_version(PINNED_GOOSE_VERSION),
        )
    with pytest.raises(InvalidStateError):
        build_goose_command(
            executable="goose",
            mode=ExecutionMode.CHAT,
            extensions=("npx something",),
            capabilities=capabilities_for_version(PINNED_GOOSE_VERSION),
        )


def test_chat_rejects_text_output_format():
    with pytest.raises(InvalidStateError):
        build_goose_command(
            executable="goose",
            mode=ExecutionMode.CHAT,
            output_format="text",
            capabilities=capabilities_for_version(PINNED_GOOSE_VERSION),
        )


def test_model_name_separated_from_goose_provider():
    plan = build_goose_command(
        executable="goose",
        mode=ExecutionMode.CHAT,
        model_name="gpt-4o",
        goose_provider="openai",
        capabilities=capabilities_for_version(PINNED_GOOSE_VERSION),
    )
    assert plan.model_name == "gpt-4o"
    assert plan.goose_provider == "openai"
    assert plan.env["GOOSE_MODEL"] == "gpt-4o"
    assert plan.env["GOOSE_PROVIDER"] == "openai"
    # Router model must not be stuffed into provider or vice versa.
    assert plan.metadata["model_name"] == "gpt-4o"
    assert plan.metadata["goose_provider"] == "openai"


def test_version_gate_cannot_silently_omit_required_flags():
    """Simulate a capability set that claims support but strip would fail."""
    # Direct construction of a plan with a version that drops flags must raise.
    caps = GooseVersionCapabilities(
        version="1.7.0",
        supports_no_session=True,
        supports_no_profile=False,  # required for chat
        supports_output_format=True,
        supports_max_turns=True,
        supports_max_tool_repetitions=True,
        supports_instructions_stdin=True,
    )
    with pytest.raises(GooseProviderError) as excinfo:
        build_goose_command(
            executable="goose",
            mode=ExecutionMode.CHAT,
            capabilities=caps,
        )
    assert excinfo.value.kind is GooseErrorKind.UNSUPPORTED_VERSION
    assert "--no-profile" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Agent policy
# ---------------------------------------------------------------------------


def test_agent_requires_explicit_policy():
    with pytest.raises(PolicyDeniedError):
        build_goose_command(
            executable="goose",
            mode=ExecutionMode.AGENT,
            capabilities=capabilities_for_version(PINNED_GOOSE_VERSION),
        )


def test_agent_policy_validates_absolute_paths(tmp_path: Path):
    root = tmp_path / "root"
    root.mkdir()
    cwd = root / "work"
    cwd.mkdir()
    with pytest.raises(PolicyDeniedError):
        GooseAgentPolicy(
            allow_side_effects=True,
            cwd="relative/path",
            path_root=str(root),
        )
    with pytest.raises(PolicyDeniedError):
        GooseAgentPolicy(
            allow_side_effects=False,
            cwd=str(cwd),
            path_root=str(root),
        )
    with pytest.raises(PolicyDeniedError):
        # cwd outside path_root
        outside = tmp_path / "outside"
        outside.mkdir()
        GooseAgentPolicy(
            allow_side_effects=True,
            cwd=str(outside),
            path_root=str(root),
        )
    policy = GooseAgentPolicy(
        allow_side_effects=True,
        cwd=str(cwd),
        path_root=str(root),
        approval_mode="approve",
        builtins=("developer",),
        max_turns=10,
    )
    assert policy.cwd == str(cwd.resolve())
    assert policy.path_root == str(root.resolve())


def test_agent_policy_rejects_chat_approval_mode(tmp_path: Path):
    root = tmp_path / "root"
    root.mkdir()
    cwd = root / "w"
    cwd.mkdir()
    with pytest.raises(PolicyDeniedError):
        GooseAgentPolicy(
            allow_side_effects=True,
            cwd=str(cwd),
            path_root=str(root),
            approval_mode="chat",
        )


def test_agent_command_sets_path_root_and_builtins(tmp_path: Path):
    root = tmp_path / "root"
    root.mkdir()
    cwd = root / "work"
    cwd.mkdir()
    policy = GooseAgentPolicy(
        allow_side_effects=True,
        cwd=str(cwd),
        path_root=str(root),
        approval_mode="smart_approve",
        builtins=("developer", "computercontroller"),
        extensions=("npx -y @example/mcp",),
        max_turns=12,
        max_tool_repetitions=4,
    )
    plan = build_goose_command(
        executable="/bin/goose",
        mode=ExecutionMode.AGENT,
        agent_policy=policy,
        model_name="muse-spark-1.1",
        goose_provider="openai",
        capabilities=capabilities_for_version(PINNED_GOOSE_VERSION),
    )
    assert plan.side_effecting is True
    assert plan.cwd == str(cwd.resolve())
    assert plan.env["GOOSE_PATH_ROOT"] == str(root.resolve())
    assert plan.env["GOOSE_MODE"] == "smart_approve"
    assert "--with-builtin" in plan.argv
    assert "developer,computercontroller" in plan.argv
    assert "--with-extension" in plan.argv
    assert plan.max_turns == 12


# ---------------------------------------------------------------------------
# JSON / stream-json parsers
# ---------------------------------------------------------------------------


SAMPLE_JSON = {
    "messages": [
        {
            "role": "user",
            "content": [{"type": "text", "text": "Say hi\n"}],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "Hi there."}],
        },
    ],
    "metadata": {
        "total_tokens": 10,
        "input_tokens": 3,
        "output_tokens": 7,
        "status": "completed",
    },
}


def test_parse_goose_json_extracts_final_assistant_text():
    parsed = parse_goose_json(json.dumps(SAMPLE_JSON))
    assert parsed.text == "Hi there."
    assert parsed.side_effects_started is False
    assert parsed.tool_call_count == 0
    assert parsed.metadata["total_tokens"] == "10"
    assert parsed.status == "completed"
    assert parsed.raw_format == OUTPUT_FORMAT_JSON


def test_parse_goose_json_detects_tool_side_effects():
    payload = {
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "running tool"},
                    {
                        "type": "tool_use",
                        "id": "1",
                        "name": "developer__shell",
                        "input": {"command": "ls"},
                    },
                ],
            }
        ],
        "metadata": {"status": "completed"},
    }
    parsed = parse_goose_json(json.dumps(payload))
    assert parsed.side_effects_started is True
    assert parsed.tool_call_count == 1
    assert any(e.side_effecting for e in parsed.events)


def test_parse_goose_json_rejects_malformed():
    with pytest.raises(MalformedOutputError):
        parse_goose_json("not-json{")
    with pytest.raises(MalformedOutputError):
        parse_goose_json("")
    with pytest.raises(MalformedOutputError):
        parse_goose_json("[]")


def test_parse_goose_json_does_not_strip_internal_whitespace():
    payload = {
        "messages": [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "  spaced\n\tcontent  "}],
            }
        ],
        "metadata": {"status": "completed"},
    }
    parsed = parse_goose_json(json.dumps(payload))
    assert parsed.text == "  spaced\n\tcontent  "


def test_parse_stream_json():
    lines = [
        json.dumps(
            {
                "type": "message",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "streamed hi"}],
                },
            }
        ),
        json.dumps(
            {
                "type": "complete",
                "total_tokens": 5,
                "input_tokens": 1,
                "output_tokens": 4,
            }
        ),
    ]
    parsed = parse_goose_stream_json("\n".join(lines) + "\n")
    assert parsed.text == "streamed hi"
    assert parsed.metadata["total_tokens"] == "5"
    assert parsed.raw_format == OUTPUT_FORMAT_STREAM_JSON


def test_parse_stream_json_tool_events():
    lines = [
        json.dumps({"type": "tool_call", "name": "shell"}),
        json.dumps(
            {
                "type": "message",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "done"}],
                },
            }
        ),
        json.dumps({"type": "complete"}),
    ]
    parsed = parse_goose_stream_json("\n".join(lines))
    assert parsed.side_effects_started is True
    assert parsed.tool_call_count >= 1


def test_parse_stream_json_malformed_line():
    with pytest.raises(MalformedOutputError):
        parse_goose_stream_json("{bad\n")


def test_parse_goose_output_dispatch():
    assert parse_goose_output(
        json.dumps(SAMPLE_JSON), output_format="json"
    ).text == "Hi there."
    with pytest.raises(MalformedOutputError):
        parse_goose_output("hello", output_format="text")


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "stderr,kind",
    [
        ("Authentication failed: 401 Unauthorized api key", GooseErrorKind.AUTHENTICATION),
        ("Rate limit exceeded / quota", GooseErrorKind.QUOTA_RATE_LIMIT),
        ("Please run goose configure — provider not configured", GooseErrorKind.UNCONFIGURED_PROVIDER),
        ("Approval required before continuing", GooseErrorKind.APPROVAL_REQUIRED),
        ("Permission denied by policy", GooseErrorKind.POLICY_DENIAL),
        ("something blew up", GooseErrorKind.NONZERO_EXIT),
    ],
)
def test_classify_goose_failure_kinds(stderr: str, kind: GooseErrorKind):
    got, message, _retry = classify_goose_failure(
        stderr=stderr, exit_code=1, process_started=True
    )
    assert got is kind
    assert message


def test_classify_timeout_and_cancel():
    assert classify_goose_failure(timed_out=True)[0] is GooseErrorKind.TIMEOUT
    assert classify_goose_failure(cancelled=True)[0] is GooseErrorKind.CANCELLATION
    assert classify_goose_failure(process_started=False, spawn_error=True)[0] in {
        GooseErrorKind.NOT_INSTALLED,
        GooseErrorKind.SPAWN_FAILED,
    }


# ---------------------------------------------------------------------------
# End-to-end provider with fake executables
# ---------------------------------------------------------------------------


def test_chat_success_json_via_fake_executable(fake_bin: Path, tmp_path: Path):
    argv_path = tmp_path / "argv.json"
    script = _json_success_script("final answer")
    exe = _write_fake_goose(fake_bin, script=script)
    child_env = {**os.environ, "GOOSE_FAKE_ARGV_PATH": str(argv_path)}
    provider = GooseCLIProvider(
        executable=str(exe),
        version=PINNED_GOOSE_VERSION,
        capabilities=capabilities_for_version(PINNED_GOOSE_VERSION),
        runner=ProcessRunner(base_env=child_env),
        base_env=child_env,
    )
    request = CLIRequest(
        prompt="Say hi",
        mode=ExecutionMode.CHAT,
        model_name="muse-spark-1.1",
        provider_override="openai",
        metadata={"goose_provider": "openai"},
    )
    result = provider.generate_result(request)
    assert result.ok is True
    assert result.text == "final answer"
    assert result.side_effecting is False
    assert result.cacheable is True
    assert result.metadata.get("side_effects_started") == "false"
    assert result.model_name == "muse-spark-1.1"
    assert result.provider_name == "goose_cli"

    recorded = json.loads(argv_path.read_text(encoding="utf-8"))
    argv = recorded["argv"]
    assert "--no-session" in argv
    assert "--no-profile" in argv
    assert "--output-format" in argv
    assert argv[argv.index("--output-format") + 1] == "json"
    assert "--max-turns" in argv
    assert argv[argv.index("--max-turns") + 1] == str(DEFAULT_CHAT_MAX_TURNS)
    assert "--max-tool-repetitions" in argv
    assert "--instructions" in argv
    assert argv[argv.index("--instructions") + 1] == "-"
    assert "--with-builtin" not in argv
    assert recorded["env_mode"] == "chat"
    assert recorded["env_provider"] == "openai"
    assert recorded["env_model"] == "muse-spark-1.1"


def test_chat_stream_json_via_fake_executable(fake_bin: Path):
    exe = _write_fake_goose(fake_bin, script=_json_success_script("streamed"))
    provider = GooseCLIProvider(
        executable=str(exe),
        version=PINNED_GOOSE_VERSION,
        capabilities=capabilities_for_version(PINNED_GOOSE_VERSION),
        runner=ProcessRunner(),
    )
    request = CLIRequest(
        prompt="hi",
        mode=ExecutionMode.CHAT,
        streaming=True,
        metadata={"output_format": OUTPUT_FORMAT_STREAM_JSON},
    )
    result = provider.generate_result(request)
    assert result.ok
    assert result.text == "streamed"
    assert result.streaming is True


def test_generate_string_surface(fake_bin: Path):
    exe = _write_fake_goose(fake_bin, script=_json_success_script("string ok"))
    provider = GooseCLIProvider(
        executable=str(exe),
        version=PINNED_GOOSE_VERSION,
        capabilities=capabilities_for_version(PINNED_GOOSE_VERSION),
    )
    text = provider.generate("prompt", model_name="m", goose_provider="openai")
    assert text == "string ok"


def test_not_installed(fake_bin: Path):
    provider = GooseCLIProvider(
        executable=str(fake_bin / "missing-goose"),
        version=PINNED_GOOSE_VERSION,
        capabilities=capabilities_for_version(PINNED_GOOSE_VERSION),
        discover_kwargs={"probe_version": False},
    )
    # Point discover away from real PATH goose.
    result = provider.generate_result(
        CLIRequest(prompt="x", mode=ExecutionMode.CHAT)
    )
    # Either not installed from discover or spawn failure — both map cleanly.
    assert result.ok is False
    kind = result.metadata.get("goose_error_kind")
    assert kind in {
        GooseErrorKind.NOT_INSTALLED.value,
        GooseErrorKind.SPAWN_FAILED.value,
        GooseErrorKind.INTERNAL.value,
    }


def test_authentication_failure(fake_bin: Path):
    exe = _write_fake_goose(
        fake_bin,
        script=_error_script(
            stderr="Authentication error: 401 Unauthorized api key missing",
            exit_code=1,
        ),
    )
    provider = GooseCLIProvider(
        executable=str(exe),
        version=PINNED_GOOSE_VERSION,
        capabilities=capabilities_for_version(PINNED_GOOSE_VERSION),
    )
    result = provider.generate_result(CLIRequest(prompt="x"))
    assert result.ok is False
    assert result.metadata["goose_error_kind"] == GooseErrorKind.AUTHENTICATION.value
    assert result.error is not None
    assert result.error.code is CLIRuntimeErrorCode.AUTHENTICATION_FAILED


def test_quota_rate_limit(fake_bin: Path):
    exe = _write_fake_goose(
        fake_bin,
        script=_error_script(stderr="Error: rate limit / quota exceeded", exit_code=1),
    )
    provider = GooseCLIProvider(
        executable=str(exe),
        version=PINNED_GOOSE_VERSION,
        capabilities=capabilities_for_version(PINNED_GOOSE_VERSION),
    )
    result = provider.generate_result(CLIRequest(prompt="x"))
    assert result.metadata["goose_error_kind"] == GooseErrorKind.QUOTA_RATE_LIMIT.value


def test_unconfigured_provider(fake_bin: Path):
    exe = _write_fake_goose(
        fake_bin,
        script=_error_script(
            stderr="No provider configured. Please run goose configure.",
            exit_code=1,
        ),
    )
    provider = GooseCLIProvider(
        executable=str(exe),
        version=PINNED_GOOSE_VERSION,
        capabilities=capabilities_for_version(PINNED_GOOSE_VERSION),
    )
    result = provider.generate_result(CLIRequest(prompt="x"))
    assert (
        result.metadata["goose_error_kind"]
        == GooseErrorKind.UNCONFIGURED_PROVIDER.value
    )


def test_approval_required(fake_bin: Path):
    exe = _write_fake_goose(
        fake_bin,
        script=_error_script(stderr="Approval required before tool use", exit_code=1),
    )
    provider = GooseCLIProvider(
        executable=str(exe),
        version=PINNED_GOOSE_VERSION,
        capabilities=capabilities_for_version(PINNED_GOOSE_VERSION),
    )
    result = provider.generate_result(CLIRequest(prompt="x"))
    assert result.metadata["goose_error_kind"] == GooseErrorKind.APPROVAL_REQUIRED.value


def test_policy_denial(fake_bin: Path):
    exe = _write_fake_goose(
        fake_bin,
        script=_error_script(stderr="Permission denied by policy", exit_code=1),
    )
    provider = GooseCLIProvider(
        executable=str(exe),
        version=PINNED_GOOSE_VERSION,
        capabilities=capabilities_for_version(PINNED_GOOSE_VERSION),
    )
    result = provider.generate_result(CLIRequest(prompt="x"))
    assert result.metadata["goose_error_kind"] == GooseErrorKind.POLICY_DENIAL.value


def test_nonzero_exit(fake_bin: Path):
    exe = _write_fake_goose(
        fake_bin,
        script=_error_script(stderr="unexpected internal boom", exit_code=7),
    )
    provider = GooseCLIProvider(
        executable=str(exe),
        version=PINNED_GOOSE_VERSION,
        capabilities=capabilities_for_version(PINNED_GOOSE_VERSION),
    )
    result = provider.generate_result(CLIRequest(prompt="x"))
    assert result.ok is False
    assert result.exit_code == 7
    assert result.metadata["goose_error_kind"] == GooseErrorKind.NONZERO_EXIT.value


def test_malformed_output(fake_bin: Path):
    script = textwrap.dedent(
        f"""\
        #!{sys.executable}
        import sys
        if "--version" in sys.argv:
            print("goose {PINNED_GOOSE_VERSION}")
            sys.exit(0)
        _ = sys.stdin.read()
        print("this is not json at all")
        sys.exit(0)
        """
    )
    exe = _write_fake_goose(fake_bin, script=script)
    provider = GooseCLIProvider(
        executable=str(exe),
        version=PINNED_GOOSE_VERSION,
        capabilities=capabilities_for_version(PINNED_GOOSE_VERSION),
    )
    result = provider.generate_result(CLIRequest(prompt="x"))
    assert result.ok is False
    assert result.metadata["goose_error_kind"] == GooseErrorKind.MALFORMED_OUTPUT.value


def test_embedded_auth_error_in_json_assistant_text(fake_bin: Path):
    """Goose often exits 0 while embedding auth failures in assistant text."""
    text = (
        "Ran into this error: Authentication error: Authentication failed "
        "for https://api.openai.com. Status: 401 Unauthorized."
    )
    exe = _write_fake_goose(fake_bin, script=_json_success_script(text, exit_code=0))
    provider = GooseCLIProvider(
        executable=str(exe),
        version=PINNED_GOOSE_VERSION,
        capabilities=capabilities_for_version(PINNED_GOOSE_VERSION),
    )
    result = provider.generate_result(CLIRequest(prompt="x"))
    assert result.ok is False
    assert result.metadata["goose_error_kind"] == GooseErrorKind.AUTHENTICATION.value


def test_side_effects_started_from_tool_use(fake_bin: Path):
    exe = _write_fake_goose(
        fake_bin, script=_json_success_script("used a tool", include_tool=True)
    )
    provider = GooseCLIProvider(
        executable=str(exe),
        version=PINNED_GOOSE_VERSION,
        capabilities=capabilities_for_version(PINNED_GOOSE_VERSION),
    )
    result = provider.generate_result(CLIRequest(prompt="x"))
    # Tool use in chat is unexpected but must be surfaced.
    assert result.metadata["side_effects_started"] == "true"
    assert result.had_side_effect_event is True
    assert result.cacheable is False
    assert result.retryable is False


def test_agent_success_with_policy(fake_bin: Path, tmp_path: Path):
    root = tmp_path / "root"
    root.mkdir()
    cwd = root / "project"
    cwd.mkdir()
    argv_path = tmp_path / "argv.json"
    exe = _write_fake_goose(fake_bin, script=_json_success_script("agent done"))
    child_env = {**os.environ, "GOOSE_FAKE_ARGV_PATH": str(argv_path)}
    provider = GooseCLIProvider(
        executable=str(exe),
        version=PINNED_GOOSE_VERSION,
        capabilities=capabilities_for_version(PINNED_GOOSE_VERSION),
        runner=ProcessRunner(base_env=child_env),
        base_env=child_env,
    )
    policy = GooseAgentPolicy(
        allow_side_effects=True,
        cwd=str(cwd),
        path_root=str(root),
        approval_mode="approve",
        builtins=("developer",),
        max_turns=5,
    )
    request = CLIRequest(
        prompt="edit the file",
        mode=ExecutionMode.AGENT,
        workspace=str(cwd),
        model_name="muse-spark-1.1",
    )
    result = provider.generate_result(request, agent_policy=policy)
    assert result.ok
    assert result.text == "agent done"
    assert result.side_effecting is True
    assert result.cacheable is False
    recorded = json.loads(argv_path.read_text(encoding="utf-8"))
    assert recorded["env_mode"] == "approve"
    assert recorded["env_path_root"] == str(root.resolve())
    assert recorded["cwd"] == str(cwd.resolve())
    assert "--with-builtin" in recorded["argv"]


def test_agent_without_policy_raises(fake_bin: Path):
    exe = _write_fake_goose(fake_bin, script=_json_success_script("x"))
    provider = GooseCLIProvider(
        executable=str(exe),
        version=PINNED_GOOSE_VERSION,
        capabilities=capabilities_for_version(PINNED_GOOSE_VERSION),
    )
    with pytest.raises(PolicyDeniedError):
        provider.generate_result(
            CLIRequest(prompt="x", mode=ExecutionMode.AGENT, workspace="/tmp")
        )


def test_generate_agent_requires_policy_fields(fake_bin: Path):
    exe = _write_fake_goose(fake_bin, script=_json_success_script("x"))
    provider = GooseCLIProvider(
        executable=str(exe),
        version=PINNED_GOOSE_VERSION,
        capabilities=capabilities_for_version(PINNED_GOOSE_VERSION),
    )
    with pytest.raises(GooseProviderError) as excinfo:
        provider.generate("x", agent=True, workspace="/tmp/work")
    assert excinfo.value.kind is GooseErrorKind.POLICY_DENIAL


def test_unsupported_version_on_execute(fake_bin: Path):
    exe = _write_fake_goose(fake_bin, script=_version_only_script("1.0.0"))
    provider = GooseCLIProvider(
        executable=str(exe),
        version="1.0.0",
        capabilities=capabilities_for_version("1.0.0"),
    )
    with pytest.raises(GooseProviderError) as excinfo:
        provider.generate_result(CLIRequest(prompt="x"))
    assert excinfo.value.kind is GooseErrorKind.UNSUPPORTED_VERSION


def test_discover_does_not_install(monkeypatch, fake_bin: Path):
    calls = {"ensure": 0}

    def fake_ensure(*a, **k):
        calls["ensure"] += 1
        raise AssertionError("discover must not call ensure_goose")

    monkeypatch.setattr(goose_mod, "ensure_goose", fake_ensure)
    exe = _write_fake_goose(fake_bin, script=_version_only_script(PINNED_GOOSE_VERSION))

    # Avoid loading the packaged release manifest (may be absent in worktrees);
    # pass a minimal stub so discovery only probes the explicit executable.
    stub_manifest = {
        "pinned_version": PINNED_GOOSE_VERSION,
        "assets": [
            {
                "os": "linux",
                "arch": "x86_64",
                "libc": "gnu",
                "variant": "standard",
                "asset_name": "goose.tar.bz2",
                "size_bytes": 1,
                "sha256": "0" * 64,
            }
        ],
    }

    def fake_run(command, **kwargs):
        import subprocess

        if command and "--version" in command:
            return subprocess.CompletedProcess(
                command, 0, stdout=f"goose {PINNED_GOOSE_VERSION}\n", stderr=""
            )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    provider = GooseCLIProvider(
        executable=str(exe),
        discover_kwargs={"manifest": stub_manifest, "run": fake_run},
    )
    result = provider.discover(probe_version=True)
    assert result.available
    assert calls["ensure"] == 0


def test_create_goose_provider_factory():
    p = create_goose_provider(executable="/not/real", allow_install=False)
    assert isinstance(p, GooseCLIProvider)
    assert p.allow_install is False


def test_goose_provider_spec_is_chat_default():
    spec = goose_provider_spec()
    assert spec.capabilities.side_effecting is False
    assert spec.capabilities.agent_mode is False
    assert spec.capabilities.chat_mode is True


def test_prompt_never_appears_in_argv(fake_bin: Path, tmp_path: Path):
    argv_path = tmp_path / "argv.json"
    secret_prompt = "TOP_SECRET_PROMPT_VALUE_XYZ"
    exe = _write_fake_goose(fake_bin, script=_json_success_script("ok"))
    child_env = {**os.environ, "GOOSE_FAKE_ARGV_PATH": str(argv_path)}
    provider = GooseCLIProvider(
        executable=str(exe),
        version=PINNED_GOOSE_VERSION,
        capabilities=capabilities_for_version(PINNED_GOOSE_VERSION),
        runner=ProcessRunner(base_env=child_env),
        base_env=child_env,
    )
    provider.generate_result(CLIRequest(prompt=secret_prompt))
    recorded = json.loads(argv_path.read_text(encoding="utf-8"))
    assert secret_prompt not in json.dumps(recorded["argv"])
    assert secret_prompt not in " ".join(recorded["argv"])


def test_cancellation_before_start(fake_bin: Path):
    exe = _write_fake_goose(fake_bin, script=_json_success_script("ok"))
    provider = GooseCLIProvider(
        executable=str(exe),
        version=PINNED_GOOSE_VERSION,
        capabilities=capabilities_for_version(PINNED_GOOSE_VERSION),
    )
    result = provider.generate_result(
        CLIRequest(prompt="x", cancellation_requested=True)
    )
    assert result.ok is False
    assert result.cancelled is True
    assert result.metadata["goose_error_kind"] == GooseErrorKind.CANCELLATION.value


def test_error_kind_mapping_coverage():
    """Every classified kind maps onto a stable runtime code."""
    for kind in GooseErrorKind:
        code = goose_error_code(kind)
        assert isinstance(code, CLIRuntimeErrorCode)


def test_agent_allowed_cwd_roots_enforced(tmp_path: Path):
    root = tmp_path / "root"
    root.mkdir()
    allowed = root / "allowed"
    allowed.mkdir()
    other = root / "other"
    other.mkdir()
    with pytest.raises(PolicyDeniedError):
        GooseAgentPolicy(
            allow_side_effects=True,
            cwd=str(other),
            path_root=str(root),
            allowed_cwd_roots=(str(allowed),),
        )
    policy = GooseAgentPolicy(
        allow_side_effects=True,
        cwd=str(allowed),
        path_root=str(root),
        allowed_cwd_roots=(str(allowed),),
    )
    assert policy.cwd.endswith("allowed")


# ---------------------------------------------------------------------------
# GOOSE-011 security matrix anchors (provider surface)
# ---------------------------------------------------------------------------


def test_matrix_quota_and_prompt_not_in_error_payload(fake_bin: Path) -> None:
    """Provider quota failures classify cleanly and never echo prompts."""
    secret = "MATRIX_PROMPT_SECRET_VALUE_xyz"
    exe = _write_fake_goose(
        fake_bin,
        script=_error_script(stderr="Error: rate limit / quota exceeded", exit_code=1),
    )
    provider = GooseCLIProvider(
        executable=str(exe),
        version=PINNED_GOOSE_VERSION,
        capabilities=capabilities_for_version(PINNED_GOOSE_VERSION),
    )
    result = provider.generate_result(CLIRequest(prompt=secret))
    assert result.ok is False
    assert result.metadata["goose_error_kind"] == GooseErrorKind.QUOTA_RATE_LIMIT.value
    blob = json.dumps(result.to_dict())
    assert secret not in blob


def test_matrix_chat_plan_forbids_profile_and_extensions() -> None:
    plan = build_goose_command(
        executable="/opt/fake-goose",
        mode=ExecutionMode.CHAT,
        capabilities=capabilities_for_version(PINNED_GOOSE_VERSION),
    )
    assert "--no-profile" in plan.argv
    assert "--no-session" in plan.argv
    assert "--with-builtin" not in plan.argv
    assert "--with-extension" not in plan.argv
    assert plan.side_effecting is False
