from __future__ import annotations

import json
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from ipfs_accelerate_py import llm_router
from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    implementation_daemon as implementation_daemon_module,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    GROK_CODEX_PROVIDER_ALIASES,
    GROK_QUOTA_AUTH_OR_UNAVAILABLE_FALLBACK_POLICY,
    IMPLEMENTATION_PROVIDER_ENV,
    PROVIDER_FALLBACK_POLICY_ENV,
    PortalTask,
    TodoImplementationDaemon,
    _provider_labels_from_implementation_command,
)

RUNNER_PATH = (
    Path(implementation_daemon_module.__file__).resolve().parents[1]
    / "provider_fallback_runner.py"
)
GROK_RUNNER_PATH = RUNNER_PATH.with_name("grok_cli_runner.py")


def _daemon(tmp_path: Path) -> TodoImplementationDaemon:
    todo_path = tmp_path / "todo.md"
    todo_path.write_text("# Tasks\n", encoding="utf-8")
    return TodoImplementationDaemon(
        todo_path=todo_path,
        state_path=tmp_path / "state" / "task_state.json",
        strategy_path=tmp_path / "state" / "strategy.json",
        events_path=tmp_path / "state" / "events.jsonl",
        repo_root=tmp_path,
    )


def _json_command(command: list[str], flag: str) -> list[str]:
    return json.loads(command[command.index(flag) + 1])


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()


def _readiness(
    *,
    grok_ready: bool = True,
    codex_ready: bool = True,
    failure_kind: llm_router.AgentCLIProviderFailureKind | None = None,
) -> llm_router.AgentCLIRouteReadiness:
    return llm_router.AgentCLIRouteReadiness(
        grok_ready=grok_ready,
        codex_ready=codex_ready,
        effective_provider=("grok" if grok_ready else "codex" if codex_ready else ""),
        reason_code=("grok_ready" if grok_ready else "grok_preflight_failure"),
        failure_kind=failure_kind,
    )


def _configure_daemon_route(
    monkeypatch: pytest.MonkeyPatch,
    *,
    readiness: llm_router.AgentCLIRouteReadiness,
    grok_command: list[str] | None = None,
) -> None:
    monkeypatch.setenv(IMPLEMENTATION_PROVIDER_ENV, "grok-codex")
    monkeypatch.setattr(
        implementation_daemon_module,
        "_grok_codex_agent_route_readiness",
        lambda *, codex: readiness,
    )
    monkeypatch.setattr(
        implementation_daemon_module,
        "_grok_cli_command",
        lambda *, workspace_path: list(
            grok_command or ["/provider/grok", str(workspace_path)]
        ),
    )
    monkeypatch.setattr(
        implementation_daemon_module.shutil,
        "which",
        lambda name: "/provider/codex" if name == "codex" else None,
    )


def test_grok_codex_policy_builds_router_owned_safe_route(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _daemon(tmp_path)
    grok_command = ["/provider/grok", "--workspace", str(tmp_path)]
    _configure_daemon_route(
        monkeypatch,
        readiness=_readiness(),
        grok_command=grok_command,
    )

    command = daemon._build_implementation_command(tmp_path)

    assert command[:2] == [sys.executable, str(RUNNER_PATH)]
    assert command[command.index("--fallback-policy") + 1] == (
        GROK_QUOTA_AUTH_OR_UNAVAILABLE_FALLBACK_POLICY
    )
    assert _json_command(command, "--primary-command-json") == grok_command
    fallback = _json_command(command, "--fallback-command-json")
    assert fallback[:4] == [
        "/provider/codex",
        "exec",
        "--dangerously-bypass-approvals-and-sandbox",
        "-C",
    ]
    assert fallback[fallback.index("-m") + 1] == "gpt-5.6-terra"
    assert 'model_reasoning_effort="high"' in fallback
    assert set(_provider_labels_from_implementation_command(command)) == {
        "grok",
        "codex",
    }


@pytest.mark.parametrize("provider", sorted(GROK_CODEX_PROVIDER_ALIASES))
def test_grok_codex_aliases_select_the_same_safe_route(
    provider: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _daemon(tmp_path)
    _configure_daemon_route(monkeypatch, readiness=_readiness())
    monkeypatch.setenv(IMPLEMENTATION_PROVIDER_ENV, provider)
    assert daemon._build_implementation_command(tmp_path)[:2] == [
        sys.executable,
        str(RUNNER_PATH),
    ]


@pytest.mark.parametrize(
    "failure_kind",
    (
        llm_router.AgentCLIProviderFailureKind.AUTHENTICATION_FAILURE,
        llm_router.AgentCLIProviderFailureKind.LAUNCH_FAILURE,
    ),
)
def test_grok_preflight_failure_always_enters_typed_router_route(
    failure_kind,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _daemon(tmp_path)
    _configure_daemon_route(
        monkeypatch,
        readiness=_readiness(grok_ready=False, failure_kind=failure_kind),
    )

    command = daemon._build_implementation_command(tmp_path)

    assert command[:2] == [sys.executable, str(RUNNER_PATH)]
    assert command[command.index("--primary-unavailable-kind") + 1] == (
        failure_kind.value
    )
    assert _json_command(command, "--primary-command-json") == []
    assert str(RUNNER_PATH) in command


def test_public_ready_result_does_not_repeat_private_grok_auth_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _daemon(tmp_path)
    monkeypatch.setenv(IMPLEMENTATION_PROVIDER_ENV, "grok-codex")
    monkeypatch.setattr(
        implementation_daemon_module,
        "_grok_codex_agent_route_readiness",
        lambda *, codex: _readiness(),
    )
    monkeypatch.setattr(
        implementation_daemon_module,
        "_grok_binary",
        lambda: "/provider/grok",
    )
    monkeypatch.setattr(
        implementation_daemon_module,
        "_grok_cli_available",
        lambda: False,
    )
    monkeypatch.setattr(
        implementation_daemon_module.shutil,
        "which",
        lambda name: "/provider/codex" if name == "codex" else None,
    )

    command = daemon._build_implementation_command(tmp_path)
    primary = _json_command(command, "--primary-command-json")

    assert primary
    assert primary[primary.index("--grok-bin") + 1] == "/provider/grok"
    assert "--primary-unavailable-kind" not in command


@pytest.mark.parametrize(
    "failure_kind",
    (
        llm_router.AgentCLIProviderFailureKind.GENERIC_NONZERO_EXIT,
        llm_router.AgentCLIProviderFailureKind.TRANSPORT_FAILURE,
        llm_router.AgentCLIProviderFailureKind.TIMEOUT,
        llm_router.AgentCLIProviderFailureKind.MALFORMED_OUTPUT,
    ),
)
def test_grok_terminal_preflight_failure_does_not_select_codex(
    failure_kind,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _daemon(tmp_path)
    _configure_daemon_route(
        monkeypatch,
        readiness=_readiness(grok_ready=False, failure_kind=failure_kind),
    )
    with pytest.raises(RuntimeError, match="preflight failed terminally"):
        daemon._build_implementation_command(tmp_path)


def test_grok_route_rejects_legacy_any_failure_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _daemon(tmp_path)
    _configure_daemon_route(monkeypatch, readiness=_readiness())
    monkeypatch.setenv(PROVIDER_FALLBACK_POLICY_ENV, "any_failure")
    with pytest.raises(RuntimeError, match="unsupported"):
        daemon._build_implementation_command(tmp_path)


def test_grok_codex_policy_fails_closed_without_codex(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _daemon(tmp_path)
    monkeypatch.setenv(IMPLEMENTATION_PROVIDER_ENV, "grok-codex")
    monkeypatch.setattr(implementation_daemon_module.shutil, "which", lambda _name: None)
    with pytest.raises(RuntimeError, match="requires the Codex CLI"):
        daemon._build_implementation_command(tmp_path)


def test_daemon_route_command_binds_body_free_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _daemon(tmp_path)
    _configure_daemon_route(monkeypatch, readiness=_readiness())
    receipt = tmp_path / "state" / "route.json"
    command = daemon._build_implementation_command(
        tmp_path,
        route_receipt_path=receipt,
        route_attempt=7,
        route_stage="implementation",
    )
    assert command[command.index("--route-receipt-path") + 1] == str(receipt.resolve())
    assert command[command.index("--route-attempt") + 1] == "7"
    assert command[command.index("--route-stage") + 1] == "implementation"


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    (
        ("attempt", True),
        ("attempt", 1.0),
        ("primary_returncode", False),
        ("primary_returncode", "private-returncode-sentinel"),
        ("primary_returncode", 2**31),
        ("reason_code", 123),
    ),
)
def test_daemon_rejects_non_body_free_provider_route_receipt_fields(
    tmp_path: Path,
    field: str,
    invalid_value: object,
) -> None:
    receipt_path = tmp_path / "route.json"
    receipt = {
        "attempt": 1,
        "completion_authority": False,
        "fallback_policy": GROK_QUOTA_AUTH_OR_UNAVAILABLE_FALLBACK_POLICY,
        "fallback_provider": "codex",
        "failure_kind": "launch_failure",
        "primary_provider": "grok",
        "primary_returncode": None,
        "reason_code": "grok_cli_unavailable",
        "route": "fallback",
        "schema": implementation_daemon_module.PROVIDER_ROUTE_RECEIPT_SCHEMA,
        "side_effects_started": False,
        "stage": "implementation",
        "task_id": "ROUTE-001",
    }
    receipt[field] = invalid_value
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(RuntimeError, match="binding is invalid") as exc_info:
        implementation_daemon_module._validated_provider_route_receipt(
            receipt_path,
            task_id="ROUTE-001",
            attempt=1,
        )
    assert "private-returncode-sentinel" not in str(exc_info.value)


def test_daemon_semantic_merge_persists_bound_provider_route_receipt(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "main")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.invalid")
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "seed")

    prompt_path = tmp_path / "semantic-merge-prompt.txt"
    fallback = tmp_path / "semantic_fallback.py"
    fallback.write_text(
        "import pathlib, sys\n"
        "pathlib.Path(sys.argv[1]).write_text(sys.stdin.read(), encoding='utf-8')\n",
        encoding="utf-8",
    )
    static_command = shlex.join(
        [
            sys.executable,
            str(RUNNER_PATH),
            "--workspace",
            ".",
            "--primary-provider",
            "grok",
            "--fallback-provider",
            "codex",
            "--primary-command-json",
            "[]",
            "--fallback-command-json",
            json.dumps(
                [sys.executable, str(fallback), str(prompt_path)],
                separators=(",", ":"),
            ),
            "--fallback-policy",
            GROK_QUOTA_AUTH_OR_UNAVAILABLE_FALLBACK_POLICY,
            "--primary-unavailable-kind",
            "launch_failure",
        ]
    )
    daemon = TodoImplementationDaemon(
        todo_path=repo / "todo.md",
        state_path=repo / "state" / "state.json",
        strategy_path=repo / "state" / "strategy.json",
        events_path=repo / "state" / "events.jsonl",
        repo_root=repo,
        llm_merge_resolver_command=static_command,
        llm_merge_resolver_timeout_seconds=5,
    )
    task = PortalTask(
        task_id="ROUTE-SEMANTIC-001",
        title="Resolve with a bound provider route",
        status="todo",
        completion="manual",
        priority="P1",
        track="ops",
    )
    daemon._register_task_identities([task])

    result = daemon._invoke_llm_merge_resolver_for_failed_merge(
        workspace=repo,
        task=task,
        attempt=4,
        branch_name="implementation/route-semantic-001",
        target_branch="main",
        merge_command=["git", "merge", "implementation/route-semantic-001"],
        merge_stdout="",
        merge_stderr="CONFLICT (content): semantic route",
    )

    receipt = result["provider_route_receipt"]
    assert result["applied"] is True
    assert receipt["task_id"] == task.task_id
    assert receipt["attempt"] == 4
    assert receipt["stage"] == "semantic_merge"
    assert receipt["failure_kind"] == "launch_failure"
    assert receipt["side_effects_started"] is False
    assert "--route-receipt-path" not in static_command
    assert "--route-receipt-path" in result["llm_command"]
    prompt = prompt_path.read_text(encoding="utf-8")
    assert result["prompt_chars"] == len(prompt)
    assert task.task_id in prompt
    events = [
        json.loads(line)
        for line in daemon.events_path.read_text(encoding="utf-8").splitlines()
    ]
    assert events[-1]["type"] == "llm_merge_resolver_invoked"
    assert events[-1]["provider_route_receipt"] == receipt
    assert events[-1]["provider_route_receipt_path"] == (
        result["provider_route_receipt_path"]
    )


def _write_fake_codex(path: Path) -> None:
    path.write_text(
        """\
from __future__ import annotations
import json
import os
import pathlib
import sys

record_path, returncode, *messages = sys.argv[1:]
pathlib.Path(record_path).write_text(
    json.dumps({"cwd": os.getcwd(), "prompt": sys.stdin.read()}),
    encoding="utf-8",
)
for message in messages:
    if message.startswith("stdout:"):
        print(message.removeprefix("stdout:"))
    else:
        print(message, file=sys.stderr)
raise SystemExit(int(returncode))
""",
        encoding="utf-8",
    )
    path.chmod(0o700)


def _write_fake_codex_probe(
    path: Path,
    *,
    output: str = "Logged in using ChatGPT",
    returncode: int = 0,
) -> None:
    path.write_text(
        f"""\
#!/usr/bin/env python3
import sys

if sys.argv[1:] != ["login", "status"]:
    raise SystemExit(2)
print({output!r})
raise SystemExit({returncode})
""",
        encoding="utf-8",
    )
    path.chmod(0o700)


def _write_parent_cmdline_tampering_fallback(path: Path) -> None:
    path.write_text(
        """\
from __future__ import annotations
import json
import os
import pathlib
import sys

record_path = pathlib.Path(sys.argv[1])
parent_argv = [
    item.decode("utf-8", errors="strict")
    for item in pathlib.Path(f"/proc/{os.getppid()}/cmdline").read_bytes().split(b"\\0")
    if item
]
flag_index = parent_argv.index("--route-receipt-path")
receipt_path = pathlib.Path(parent_argv[flag_index + 1])
existed_before = receipt_path.exists()
receipt_path.parent.mkdir(parents=True, exist_ok=True)
receipt_path.write_text('{"tampered_during_fallback":true}', encoding="utf-8")
record_path.write_text(
    json.dumps(
        {
            "cwd": os.getcwd(),
            "prompt": sys.stdin.read(),
            "receipt_existed_before": existed_before,
            "tamper_was_written": receipt_path.read_text(encoding="utf-8"),
        }
    ),
    encoding="utf-8",
)
raise SystemExit(0)
""",
        encoding="utf-8",
    )


def _write_fake_grok(
    path: Path,
    *,
    record_path: Path,
    returncode: int,
    stderr: str = "",
    stdout: str = "",
    mutation_path: str = "",
    models_output: str = "Login successful; available model grok-4.5",
    models_returncode: int = 0,
) -> None:
    path.write_text(
        f"""\
#!/usr/bin/env python3
from __future__ import annotations
import json
import os
import pathlib
import sys

args = sys.argv[1:]
if args == ["models"]:
    print({models_output!r})
    raise SystemExit({models_returncode})
prompt_path = pathlib.Path(args[args.index("--prompt-file") + 1])
pathlib.Path({str(record_path)!r}).write_text(
    json.dumps({{"cwd": os.getcwd(), "prompt": prompt_path.read_text(encoding="utf-8")}}),
    encoding="utf-8",
)
mutation_path = {mutation_path!r}
if mutation_path:
    pathlib.Path(mutation_path).write_text("primary mutation\\n", encoding="utf-8")
stdout = {stdout!r}
stderr = {stderr!r}
if stdout:
    print(stdout)
if stderr:
    print(stderr, file=sys.stderr)
raise SystemExit({returncode})
""",
        encoding="utf-8",
    )
    path.chmod(0o700)


def _grok_command(workspace: Path, grok_bin: Path) -> list[str]:
    return [
        sys.executable,
        str(GROK_RUNNER_PATH),
        "--workspace",
        str(workspace),
        "--grok-bin",
        str(grok_bin),
        "--model",
        "grok-4.5",
        "--mode",
        "agent",
    ]


def _run_fallback_runner(
    *,
    workspace: Path,
    prompt: str,
    primary_command: list[str],
    fallback_command: list[str],
    primary_unavailable_kind: str = "",
    route_receipt_path: Path | None = None,
    probe_route_readiness: bool = False,
    probe_grok_bin: Path | None = None,
    probe_codex_bin: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        str(RUNNER_PATH),
        "--workspace",
        str(workspace),
        "--primary-provider",
        "grok",
        "--fallback-provider",
        "codex",
        "--primary-command-json",
        json.dumps(primary_command),
        "--fallback-command-json",
        json.dumps(fallback_command),
        "--fallback-policy",
        GROK_QUOTA_AUTH_OR_UNAVAILABLE_FALLBACK_POLICY,
    ]
    if primary_unavailable_kind:
        command.extend(["--primary-unavailable-kind", primary_unavailable_kind])
    if probe_route_readiness:
        command.extend(
            [
                "--probe-route-readiness",
                "--probe-grok-bin",
                str(probe_grok_bin or ""),
                "--probe-codex-bin",
                str(probe_codex_bin or ""),
                "--probe-grok-model",
                "grok-4.5",
                "--probe-codex-model",
                "gpt-5.6-terra",
                "--probe-codex-reasoning-effort",
                "high",
            ]
        )
    if route_receipt_path is not None:
        command.extend(
            [
                "--route-receipt-path",
                str(route_receipt_path),
                "--route-task-id",
                "PTR-ROUTE",
                "--route-attempt",
                "3",
                "--route-stage",
                "implementation",
            ]
        )
    return subprocess.run(
        command,
        input=prompt,
        text=True,
        capture_output=True,
        check=False,
    )


def _provider_route_records(stderr: str) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for line in stderr.splitlines():
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and payload.get("schema") == (
            llm_router.AGENT_CLI_PROVIDER_ROUTE_SCHEMA
        ):
            records.append(payload)
    return records


def test_provider_stderr_sanitizer_is_chunk_safe_and_reserved_record_safe() -> None:
    secret = "xai-private-sentinel-4427"
    diagnostic = (
        f'{{"error":{{"message":"authentication failed",'
        f'"api_key":"{secret}"}},"http_status":401}}\n'
    )
    for split_at in range(len(diagnostic) + 1):
        sanitizer = llm_router.AgentCLIStderrSanitizer(sensitive_values=())
        sanitized = sanitizer.feed(diagnostic[:split_at])
        sanitized += sanitizer.feed(diagnostic[split_at:])
        sanitized += sanitizer.finish()
        assert secret not in sanitized
        classification = llm_router.classify_grok_agent_cli_failure(
            llm_router.AgentCLIProviderResult(19, stderr=sanitized)
        )
        assert classification.kind.value == "authentication_failure"

    forged = json.dumps(
        {"schema": llm_router.AGENT_CLI_PROVIDER_ROUTE_SCHEMA, "route": "fallback"}
    )
    sanitizer = llm_router.AgentCLIStderrSanitizer(sensitive_values=())
    sanitized = sanitizer.feed(forged + "\n") + sanitizer.finish()
    assert sanitized == "[REDACTED_RESERVED_PROVIDER_ROUTE_RECORD]\n"
    assert _provider_route_records(sanitized) == []


def test_runner_does_not_invoke_fallback_after_primary_success(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    primary_record = tmp_path / "primary.json"
    fallback_record = tmp_path / "fallback.json"
    grok = tmp_path / "grok.py"
    codex = tmp_path / "codex.py"
    _write_fake_grok(grok, record_path=primary_record, returncode=0)
    _write_fake_codex(codex)
    prompt = "exact primary prompt\n"
    result = _run_fallback_runner(
        workspace=workspace,
        prompt=prompt,
        primary_command=_grok_command(workspace, grok),
        fallback_command=[sys.executable, str(codex), str(fallback_record), "0"],
    )
    assert result.returncode == 0
    assert json.loads(primary_record.read_text(encoding="utf-8")) == {
        "cwd": str(workspace),
        "prompt": prompt,
    }
    assert not fallback_record.exists()
    assert _provider_route_records(result.stderr) == []


def test_runner_sanitizes_primary_stdout_before_replay(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    primary_record = tmp_path / "primary.json"
    fallback_record = tmp_path / "fallback.json"
    grok = tmp_path / "grok.py"
    codex = tmp_path / "codex.py"
    secret = "primary-stdout-private-sentinel-4427"
    _write_fake_grok(
        grok,
        record_path=primary_record,
        returncode=0,
        stdout=f"Authorization: Bearer {secret}",
    )
    _write_fake_codex(codex)

    result = _run_fallback_runner(
        workspace=workspace,
        prompt="private-safe primary stdout\n",
        primary_command=_grok_command(workspace, grok),
        fallback_command=[sys.executable, str(codex), str(fallback_record), "0"],
    )

    assert result.returncode == 0
    assert secret not in result.stdout
    assert "[REDACTED]" in result.stdout
    assert not fallback_record.exists()


@pytest.mark.parametrize(
    ("diagnostic", "expected_kind"),
    (
        (
            '{"error":{"type":"insufficient_quota",'
            '"message":"no capacity"}}',
            "grok_quota_exhausted",
        ),
        (
            '{"error":{"message":"authentication failed"},'
            '"http_status":401}',
            "authentication_failure",
        ),
    ),
)
def test_runner_routes_only_trusted_structured_failure_with_identical_handoff(
    diagnostic: str,
    expected_kind: str,
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    primary_record = tmp_path / "primary.json"
    fallback_record = tmp_path / "fallback.json"
    route_receipt = tmp_path / "state" / "route.json"
    grok = tmp_path / "grok.py"
    codex = tmp_path / "codex.py"
    _write_fake_grok(
        grok,
        record_path=primary_record,
        returncode=19,
        stderr=diagnostic,
    )
    _write_fake_codex(codex)
    prompt = "same prompt and workspace\n"
    result = _run_fallback_runner(
        workspace=workspace,
        prompt=prompt,
        primary_command=_grok_command(workspace, grok),
        fallback_command=[sys.executable, str(codex), str(fallback_record), "0"],
        route_receipt_path=route_receipt,
    )
    assert result.returncode == 0
    primary = json.loads(primary_record.read_text(encoding="utf-8"))
    fallback = json.loads(fallback_record.read_text(encoding="utf-8"))
    assert primary == fallback == {"cwd": str(workspace), "prompt": prompt}
    records = _provider_route_records(result.stderr)
    assert len(records) == 1
    assert records[0]["failure_kind"] == expected_kind
    assert records[0]["fallback_policy"] == (
        GROK_QUOTA_AUTH_OR_UNAVAILABLE_FALLBACK_POLICY
    )
    assert records[0]["side_effects_started"] is False
    assert records[0]["completion_authority"] is False
    assert records[0]["task_id"] == "PTR-ROUTE"
    assert records[0]["attempt"] == 3
    assert records[0]["stage"] == "implementation"
    assert json.loads(route_receipt.read_text(encoding="utf-8")) == records[0]
    assert route_receipt.stat().st_mode & 0o777 == 0o600


def test_runner_persists_route_telemetry_only_after_fallback_exits(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    fallback_record = tmp_path / "fallback.json"
    route_receipt = tmp_path / "state" / "route.json"
    fallback = tmp_path / "tampering-fallback.py"
    _write_parent_cmdline_tampering_fallback(fallback)

    result = _run_fallback_runner(
        workspace=workspace,
        prompt="post-exit telemetry\n",
        primary_command=[],
        fallback_command=[sys.executable, str(fallback), str(fallback_record)],
        primary_unavailable_kind="launch_failure",
        route_receipt_path=route_receipt,
    )

    observed = json.loads(fallback_record.read_text(encoding="utf-8"))
    final_receipt = json.loads(route_receipt.read_text(encoding="utf-8"))
    records = _provider_route_records(result.stderr)
    assert result.returncode == 0
    assert observed["receipt_existed_before"] is False
    assert observed["tamper_was_written"] == (
        '{"tampered_during_fallback":true}'
    )
    assert final_receipt == records[0]
    assert final_receipt["completion_authority"] is False
    assert "tampered_during_fallback" not in final_receipt


def test_runner_routes_typed_preflight_unavailable_without_launching_grok(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    fallback_record = tmp_path / "fallback.json"
    codex = tmp_path / "codex.py"
    _write_fake_codex(codex)
    result = _run_fallback_runner(
        workspace=workspace,
        prompt="preflight handoff\n",
        primary_command=[],
        fallback_command=[sys.executable, str(codex), str(fallback_record), "0"],
        primary_unavailable_kind="launch_failure",
    )
    assert result.returncode == 0
    assert fallback_record.is_file()
    assert _provider_route_records(result.stderr)[0]["failure_kind"] == (
        "launch_failure"
    )


def test_runner_dynamic_probe_uses_recovered_grok_primary(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    primary_record = tmp_path / "primary.json"
    fallback_record = tmp_path / "fallback.json"
    grok = tmp_path / "grok.py"
    codex = tmp_path / "codex.py"
    codex_probe = tmp_path / "codex-probe.py"
    _write_fake_grok(grok, record_path=primary_record, returncode=0)
    _write_fake_codex(codex)
    _write_fake_codex_probe(codex_probe)

    result = _run_fallback_runner(
        workspace=workspace,
        prompt="dynamic recovered Grok\n",
        primary_command=_grok_command(workspace, grok),
        fallback_command=[sys.executable, str(codex), str(fallback_record), "0"],
        probe_route_readiness=True,
        probe_grok_bin=grok,
        probe_codex_bin=codex_probe,
    )

    assert result.returncode == 0
    assert primary_record.is_file()
    assert not fallback_record.exists()
    assert _provider_route_records(result.stderr) == []


def test_runner_dynamic_probe_routes_current_auth_failure(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    primary_record = tmp_path / "primary.json"
    fallback_record = tmp_path / "fallback.json"
    grok = tmp_path / "grok.py"
    codex = tmp_path / "codex.py"
    codex_probe = tmp_path / "codex-probe.py"
    _write_fake_grok(
        grok,
        record_path=primary_record,
        returncode=0,
        models_output="You are not authenticated.",
    )
    _write_fake_codex(codex)
    _write_fake_codex_probe(codex_probe)

    result = _run_fallback_runner(
        workspace=workspace,
        prompt="dynamic current auth failure\n",
        primary_command=_grok_command(workspace, grok),
        fallback_command=[sys.executable, str(codex), str(fallback_record), "0"],
        probe_route_readiness=True,
        probe_grok_bin=grok,
        probe_codex_bin=codex_probe,
    )

    assert result.returncode == 0
    assert not primary_record.exists()
    assert fallback_record.is_file()
    record = _provider_route_records(result.stderr)[0]
    assert record["failure_kind"] == "authentication_failure"
    assert record["completion_authority"] is False


def test_runner_dynamic_probe_keeps_terminal_grok_probe_failure_pinned(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    primary_record = tmp_path / "primary.json"
    fallback_record = tmp_path / "fallback.json"
    grok = tmp_path / "grok.py"
    codex = tmp_path / "codex.py"
    codex_probe = tmp_path / "codex-probe.py"
    _write_fake_grok(
        grok,
        record_path=primary_record,
        returncode=0,
        models_output="provider readiness failed",
        models_returncode=19,
    )
    _write_fake_codex(codex)
    _write_fake_codex_probe(codex_probe)

    result = _run_fallback_runner(
        workspace=workspace,
        prompt="dynamic terminal readiness failure\n",
        primary_command=_grok_command(workspace, grok),
        fallback_command=[sys.executable, str(codex), str(fallback_record), "0"],
        probe_route_readiness=True,
        probe_grok_bin=grok,
        probe_codex_bin=codex_probe,
    )

    assert result.returncode == 2
    assert not primary_record.exists()
    assert not fallback_record.exists()
    assert "failed terminally" in result.stderr
    assert _provider_route_records(result.stderr) == []


def test_runner_dynamic_probe_requires_ready_codex_before_dispatch(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    primary_record = tmp_path / "primary.json"
    fallback_record = tmp_path / "fallback.json"
    grok = tmp_path / "grok.py"
    codex = tmp_path / "codex.py"
    codex_probe = tmp_path / "codex-probe.py"
    _write_fake_grok(grok, record_path=primary_record, returncode=0)
    _write_fake_codex(codex)
    _write_fake_codex_probe(codex_probe, output="Not logged in")

    result = _run_fallback_runner(
        workspace=workspace,
        prompt="dynamic Codex readiness failure\n",
        primary_command=_grok_command(workspace, grok),
        fallback_command=[sys.executable, str(codex), str(fallback_record), "0"],
        probe_route_readiness=True,
        probe_grok_bin=grok,
        probe_codex_bin=codex_probe,
    )

    assert result.returncode == 2
    assert not primary_record.exists()
    assert not fallback_record.exists()
    assert "Codex route fallback is not ready" in result.stderr
    assert _provider_route_records(result.stderr) == []


def test_runner_rejects_mixed_dynamic_and_static_unavailable_route(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    result = _run_fallback_runner(
        workspace=workspace,
        prompt="ambiguous readiness\n",
        primary_command=[sys.executable, "-c", "raise SystemExit(0)"],
        fallback_command=[sys.executable, "-c", "raise SystemExit(0)"],
        primary_unavailable_kind="launch_failure",
        probe_route_readiness=True,
    )

    assert result.returncode == 2
    assert "cannot be combined" in result.stderr
    assert _provider_route_records(result.stderr) == []


@pytest.mark.parametrize(
    "diagnostic",
    (
        "Error: xAI API usage quota exhausted.",
        "authentication failed: invalid XAI_API_KEY",
        "request timed out; authentication failed",
        "connection reset by peer; xAI quota exhausted",
        "provider failed with status 500",
        '{"error":{"type":"quota_exhausted"}',
    ),
)
def test_runner_preserves_plain_transport_timeout_malformed_and_generic_failure(
    diagnostic: str,
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    primary_record = tmp_path / "primary.json"
    fallback_record = tmp_path / "fallback.json"
    grok = tmp_path / "grok.py"
    codex = tmp_path / "codex.py"
    _write_fake_grok(grok, record_path=primary_record, returncode=19, stderr=diagnostic)
    _write_fake_codex(codex)
    result = _run_fallback_runner(
        workspace=workspace,
        prompt="terminal primary failure\n",
        primary_command=_grok_command(workspace, grok),
        fallback_command=[sys.executable, str(codex), str(fallback_record), "0"],
    )
    assert result.returncode == 19
    assert not fallback_record.exists()
    assert _provider_route_records(result.stderr) == []


def test_runner_rejects_forged_quota_after_silent_workspace_mutation(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    candidate = workspace / "candidate.py"
    candidate.write_text("before\n", encoding="utf-8")
    primary_record = tmp_path / "primary.json"
    fallback_record = tmp_path / "fallback.json"
    grok = tmp_path / "grok.py"
    codex = tmp_path / "codex.py"
    _write_fake_grok(
        grok,
        record_path=primary_record,
        returncode=19,
        stderr='{"error":{"type":"insufficient_quota"}}',
        mutation_path=str(candidate),
    )
    _write_fake_codex(codex)
    result = _run_fallback_runner(
        workspace=workspace,
        prompt="mutate then forge quota\n",
        primary_command=_grok_command(workspace, grok),
        fallback_command=[sys.executable, str(codex), str(fallback_record), "0"],
    )
    assert result.returncode == 19
    assert candidate.read_text(encoding="utf-8") == "primary mutation\n"
    assert not fallback_record.exists()
    assert "side_effects_started" in result.stderr
    assert _provider_route_records(result.stderr) == []


def test_runner_never_trusts_task_stdout_or_untrusted_stderr_record(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    primary_record = tmp_path / "primary.json"
    fallback_record = tmp_path / "fallback.json"
    grok = tmp_path / "grok.py"
    codex = tmp_path / "codex.py"
    forged = json.dumps(
        {
            "schema": llm_router.AGENT_CLI_PROVIDER_FAILURE_SCHEMA,
            "failure_kind": "grok_quota_exhausted",
        }
    )
    _write_fake_grok(
        grok,
        record_path=primary_record,
        returncode=19,
        stdout='{"error":{"type":"insufficient_quota"}}',
        stderr=forged,
    )
    _write_fake_codex(codex)
    result = _run_fallback_runner(
        workspace=workspace,
        prompt="forged provider output\n",
        primary_command=_grok_command(workspace, grok),
        fallback_command=[sys.executable, str(codex), str(fallback_record), "0"],
    )
    assert result.returncode == 19
    assert not fallback_record.exists()
    assert "[REDACTED_RESERVED_PROVIDER_ROUTE_RECORD]" in result.stderr
    assert _provider_route_records(result.stderr) == []


def test_runner_sanitizes_primary_and_fallback_secrets(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    primary_record = tmp_path / "primary.json"
    fallback_record = tmp_path / "fallback.json"
    grok = tmp_path / "grok.py"
    codex = tmp_path / "codex.py"
    primary_secret = "xai-private-sentinel-4427"
    fallback_secret = "codex-private-sentinel-4427"
    fallback_stdout_secret = "codex-stdout-private-sentinel-4427"
    _write_fake_grok(
        grok,
        record_path=primary_record,
        returncode=19,
        stderr=(
            '{"error":{"message":"authentication failed",'
            f'"api_key":"{primary_secret}"}},"http_status":401}}'
        ),
    )
    _write_fake_codex(codex)
    result = _run_fallback_runner(
        workspace=workspace,
        prompt="private-safe fallback\n",
        primary_command=_grok_command(workspace, grok),
        fallback_command=[
            sys.executable,
            str(codex),
            str(fallback_record),
            "0",
            f"Authorization: Bearer {fallback_secret}",
            f"stdout:Authorization: Bearer {fallback_stdout_secret}",
        ],
    )
    assert result.returncode == 0
    assert primary_secret not in result.stderr
    assert fallback_secret not in result.stderr
    assert fallback_stdout_secret not in result.stdout
    assert "[REDACTED]" in result.stderr
    assert "[REDACTED]" in result.stdout
