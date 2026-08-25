from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    implementation_daemon as daemon_module,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    GROK_QUOTA_ONLY_FALLBACK_POLICY,
    IMPLEMENTATION_FALLBACK_TRIGGER_ENV,
    PortalTask,
    TodoImplementationDaemon,
)

from ipfs_accelerate_py import llm_router

ROOT = Path(__file__).resolve().parents[2]
CASF_CONFIG = ROOT / "config/agent_supervisor_causal_event_federation_scheduler.json"
FALLBACK_RUNNER = Path(daemon_module.__file__).resolve().parents[1] / "provider_fallback_runner.py"
GROK_ADAPTER = FALLBACK_RUNNER.with_name("grok_cli_runner.py")
_GROK_ADAPTER_SPEC = importlib.util.spec_from_file_location(
    "_test_agent_supervisor_grok_cli_runner",
    GROK_ADAPTER,
)
assert _GROK_ADAPTER_SPEC is not None and _GROK_ADAPTER_SPEC.loader is not None
grok_adapter_module = importlib.util.module_from_spec(_GROK_ADAPTER_SPEC)
_GROK_ADAPTER_SPEC.loader.exec_module(grok_adapter_module)


def _daemon(tmp_path: Path) -> TodoImplementationDaemon:
    todo_path = tmp_path / "todo.md"
    todo_path.write_text("# Tasks\n", encoding="utf-8")
    return TodoImplementationDaemon(
        todo_path=todo_path,
        state_path=tmp_path / "state" / "task_state.json",
        strategy_path=tmp_path / "state" / "strategy.json",
        events_path=tmp_path / "state" / "events.jsonl",
        repo_root=tmp_path,
        worktree_root=tmp_path,
    )


def _casf_route_environment() -> dict[str, str]:
    payload = json.loads(CASF_CONFIG.read_text(encoding="utf-8"))
    provider = payload["provider"]
    route = llm_router.resolve_agent_implementation_route(
        primary_provider_id=provider["primary_provider_id"],
        primary_model_id=provider["primary_model_id"],
        fallback_provider_id=provider["fallback_provider_id"],
        fallback_model_id=provider["fallback_model_id"],
        fallback_trigger=provider["fallback_trigger"],
        fallback_reasoning_effort=provider["fallback_reasoning_effort"],
    )
    assert route.authorization is None
    return route.as_environment()


def _configure_ready_casf_route(
    monkeypatch: pytest.MonkeyPatch,
    *,
    grok: str,
    codex: str,
) -> None:
    for name, value in _casf_route_environment().items():
        monkeypatch.setenv(name, value)
    for name in (
        daemon_module.PROVIDER_FALLBACK_POLICY_ENV,
        daemon_module._ROUTE_BOARD_NAMESPACE_ENV,
        daemon_module._ROUTE_AUTHORIZATION_PATH_ENV,
        daemon_module._ROUTE_AUTHORIZATION_SHA256_ENV,
        daemon_module._ROUTE_AUTHORIZATION_ID_ENV,
        daemon_module._ROUTE_AUTHORIZATION_KIND_ENV,
        daemon_module._ROUTE_SOURCE_HEAD_ENV,
        daemon_module._ROUTE_SOURCE_TREE_ENV,
        daemon_module._ROUTE_ID_ENV,
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(daemon_module, "_grok_binary", lambda: grok)
    monkeypatch.setattr(daemon_module, "_grok_cli_available", lambda: True)
    monkeypatch.setattr(
        daemon_module.shutil,
        "which",
        lambda name: codex if name == "codex" else None,
    )
    monkeypatch.setattr(
        daemon_module,
        "_grok_codex_agent_route_readiness",
        lambda *, codex: pytest.fail(
            "quota-only route must not derive authority from a models probe"
        ),
    )


def _json_argv(command: list[str], flag: str) -> list[str]:
    return json.loads(command[command.index(flag) + 1])


def test_casf_launch_route_builds_exact_ordered_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_ready_casf_route(
        monkeypatch,
        grok="/provider/grok",
        codex="/provider/codex",
    )

    command = _daemon(tmp_path)._build_implementation_command(tmp_path)

    assert command[:2] == [sys.executable, str(FALLBACK_RUNNER)]
    assert command[command.index("--fallback-policy") + 1] == (GROK_QUOTA_ONLY_FALLBACK_POLICY)
    primary = _json_argv(command, "--primary-command-json")
    assert primary[:2] == [sys.executable, str(GROK_ADAPTER)]
    assert primary[primary.index("--model") + 1] == "grok-4.6"
    assert "--require-terminal-quota-frame" in primary
    assert "--codex-fallback-command-json" not in primary
    fallback = _json_argv(command, "--fallback-command-json")
    assert fallback[fallback.index("-m") + 1] == "gpt-5.6-terra"
    assert 'model_reasoning_effort="high"' in fallback
    assert 'model_reasoning_effort="medium"' not in fallback
    assert "--primary-unavailable-kind" not in command


def test_authentication_unavailable_route_stays_reviewer_gated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _casf_route_environment()
    environment[IMPLEMENTATION_FALLBACK_TRIGGER_ENV] = "primary_quota_or_auth_unavailable"
    for name, value in environment.items():
        monkeypatch.setenv(name, value)

    with pytest.raises(RuntimeError, match="invalid sealed implementation route"):
        _daemon(tmp_path)._build_implementation_command(tmp_path)


def _write_provider(path: Path, source: str) -> None:
    path.write_text("#!/usr/bin/env python3\n" + source, encoding="utf-8")
    path.chmod(0o700)


def _observed_balance_terminal() -> str:
    return json.dumps(
        {
            "type": "error",
            "message": (
                "Internal error: {\n"
                '  "message": "API error (status 402 Payment Required): '
                'Grok Build usage balance exhausted",\n'
                '  "http_status": 402\n'
                "}"
            ),
        },
        separators=(",", ":"),
    )


def _available_commands(*, command: str = "build-with-ai") -> str:
    return json.dumps(
        {
            "type": "available_commands",
            "tools": ["read_file", "write"],
            "commands": [command, "code-review"],
        },
        separators=(",", ":"),
    )


@pytest.mark.parametrize(
    ("stdout", "stderr", "fallback_expected"),
    (
        (
            _available_commands()
            + "\n"
            + _available_commands()
            + "\n"
            + _observed_balance_terminal(),
            "",
            True,
        ),
        (
            '{"type":"error","error":{"code":"usage_limit_reached"}}',
            "",
            True,
        ),
        (
            "",
            '{"error":{"type":"insufficient_quota","message":"no capacity"}}',
            False,
        ),
        (
            '{"type":"assistant","message":"usage_limit_reached"}',
            "",
            False,
        ),
        (
            "",
            '{"error":{"message":"authentication failed"},"http_status":401}',
            False,
        ),
    ),
)
def test_actual_runner_falls_back_only_after_trusted_terminal_quota(
    stdout: str,
    stderr: str,
    fallback_expected: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    grok = tmp_path / "grok"
    codex = tmp_path / "codex"
    grok_launch_path = tmp_path / "grok-launch.json"
    fallback_launch_path = tmp_path / "fallback-launch.json"
    route_environment = _casf_route_environment()
    route_names = tuple(sorted(route_environment))
    _write_provider(
        grok,
        "import json, os, pathlib, sys\n"
        "argv = sys.argv[1:]\n"
        "prompt_path = pathlib.Path(argv[argv.index('--prompt-file') + 1])\n"
        f"pathlib.Path({str(grok_launch_path)!r}).write_text(json.dumps({{"
        "'argv': argv, 'cwd': os.getcwd(), "
        "'prompt': prompt_path.read_text(encoding='utf-8'), "
        f"'route_environment': {{name: os.environ.get(name) for name in {route_names!r}}}"
        "}, sort_keys=True), encoding='utf-8')\n"
        + (f"print({stdout!r})\n" if stdout else "")
        + (f"print({stderr!r}, file=sys.stderr)\n" if stderr else "")
        + "raise SystemExit(19)\n",
    )
    _write_provider(
        codex,
        "import json, os, pathlib, sys\n"
        f"pathlib.Path({str(fallback_launch_path)!r}).write_text(json.dumps({{"
        "'argv': sys.argv[1:], 'cwd': os.getcwd(), "
        "'prompt': sys.stdin.read(), "
        f"'route_environment': {{name: os.environ.get(name) for name in {route_names!r}}}"
        "}, sort_keys=True), encoding='utf-8')\n",
    )
    _configure_ready_casf_route(
        monkeypatch,
        grok=str(grok),
        codex=str(codex),
    )
    task = PortalTask(
        task_id="CASF-033",
        title="First model-routed CASF task",
        status="in_progress",
        completion="",
        priority="high",
        track="CASF",
        canonical_task_key="task:casf-033",
        canonical_task_cid="task:casf-033",
        board_namespace="causal-event-supervisor-federation-v1",
    )
    command = _daemon(tmp_path)._build_implementation_command(
        workspace,
        task=task,
        prompt="implement CASF model task\n",
        attempt=3,
    )
    environment = dict(os.environ)
    environment["IPFS_ACCELERATE_AGENT_PROOF_REUSE_STATE_ROOT"] = ""
    environment["IPFS_ACCELERATE_AGENT_PROVIDER_PROTECTED_STATE_ROOT"] = ""

    completed = subprocess.run(
        command,
        cwd=workspace,
        input="implement CASF model task\n",
        text=True,
        capture_output=True,
        check=False,
        env=environment,
        timeout=30,
    )

    assert grok_launch_path.is_file(), completed.stderr
    grok_launch = json.loads(grok_launch_path.read_text(encoding="utf-8"))
    grok_argv = grok_launch["argv"]
    assert grok_argv[grok_argv.index("--model") + 1] == "grok-4.6"
    assert grok_argv[grok_argv.index("--output-format") + 1] == "streaming-json"
    assert grok_argv[grok_argv.index("--cwd") + 1] == str(workspace)
    assert grok_launch["cwd"] == str(workspace)
    assert grok_launch["prompt"] == "implement CASF model task\n"
    assert grok_launch["route_environment"] == route_environment
    assert fallback_launch_path.exists() is fallback_expected
    if fallback_expected:
        assert completed.returncode == 0
        assert "grok_quota_exhausted" in completed.stderr
        fallback_launch = json.loads(fallback_launch_path.read_text(encoding="utf-8"))
        fallback_argv = fallback_launch["argv"]
        assert fallback_argv[fallback_argv.index("-m") + 1] == "gpt-5.6-terra"
        assert 'model_reasoning_effort="high"' in fallback_argv
        assert 'model_reasoning_effort="medium"' not in fallback_argv
        assert fallback_launch["cwd"] == str(workspace)
        assert fallback_launch["prompt"] == "implement CASF model task\n"
        assert fallback_launch["route_environment"] == route_environment
        receipt_flag = command.index("--route-receipt-path") + 1
        route_receipt = json.loads(Path(command[receipt_flag]).read_text(encoding="utf-8"))
        assert route_receipt["task_id"] == "CASF-033"
        assert route_receipt["attempt"] == 3
        assert route_receipt["stage"] == "implementation"
        assert route_receipt["failure_kind"] == "grok_quota_exhausted"
        assert route_receipt["fallback_policy"] == GROK_QUOTA_ONLY_FALLBACK_POLICY
    else:
        assert completed.returncode == 19
        assert "failure_not_fallback_eligible" in completed.stderr
        assert "--route-receipt-path" in command
        assert not Path(command[command.index("--route-receipt-path") + 1]).exists()


def _parse_terminal_frames(*frames: str):
    parser = grok_adapter_module._BoundedTerminalFrameParser()
    parser.feed("\n".join(frames) + "\n", final=True)
    return parser


def test_observed_grok_402_requires_exact_inert_protocol_prelude() -> None:
    prelude = _available_commands()
    parser = _parse_terminal_frames(prelude, prelude, _observed_balance_terminal())

    assert parser.exact_terminal_candidate
    assert parser.prelude_count == 2
    assert (
        grok_adapter_module._terminal_grok_quota_code(
            parser.last_event,
            allow_embedded_balance=parser.prelude_count > 0,
        )
        == "usage_pool_exhausted"
    )

    no_prelude = _parse_terminal_frames(_observed_balance_terminal())
    assert no_prelude.exact_terminal_candidate
    assert (
        grok_adapter_module._terminal_grok_quota_code(
            no_prelude.last_event,
            allow_embedded_balance=False,
        )
        == ""
    )


@pytest.mark.parametrize(
    "frames",
    (
        (
            _available_commands(),
            '{"type":"assistant","message":"usage_pool_exhausted"}',
            _observed_balance_terminal(),
        ),
        (
            _available_commands(),
            _available_commands(command="different-command"),
            _observed_balance_terminal(),
        ),
        (
            _available_commands(),
            _observed_balance_terminal(),
            _available_commands(),
        ),
        (
            _available_commands(),
            '{"type":"error","message":"Internal error: '
            '{\\"message\\":\\"arbitrary 402 text\\",'
            '\\"http_status\\":402}"}',
        ),
        (
            '{"type":"available_commands","tools":["read_file",'
            '"read_file"],"commands":[]}',
            _observed_balance_terminal(),
        ),
    ),
)
def test_observed_grok_402_rejects_mixed_or_malformed_activity(
    frames: tuple[str, ...],
) -> None:
    parser = _parse_terminal_frames(*frames)
    quota_code = (
        grok_adapter_module._terminal_grok_quota_code(
            parser.last_event,
            allow_embedded_balance=parser.prelude_count > 0,
        )
        if parser.exact_terminal_candidate
        else ""
    )
    assert quota_code == ""
