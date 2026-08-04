from __future__ import annotations

import io
import json
import shlex
import subprocess
import uuid
from pathlib import Path

import pytest

import ipfs_accelerate_py.llm_router as llm_router
from ipfs_accelerate_py.agent_supervisor import grok_cli_runner
from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    implementation_daemon,
    implementation_supervisor,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    PortalTask,
    TodoImplementationDaemon,
)


def _daemon(root: Path) -> TodoImplementationDaemon:
    board = root / "tasks.todo.md"
    board.write_text("# Tasks\n", encoding="utf-8")
    return TodoImplementationDaemon(
        todo_path=board,
        state_path=root / "state" / "task-state.json",
        strategy_path=root / "state" / "strategy.json",
        events_path=root / "state" / "events.jsonl",
        repo_root=root,
    )


def _clear_provider_overrides(monkeypatch) -> None:
    monkeypatch.setattr(
        grok_cli_runner,
        "_select_grok_isolation_backend",
        lambda **_kwargs: grok_cli_runner.GROK_ISOLATION_GROK_SANDBOX,
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_resolve_trusted_grok_bin",
        lambda *, configured, workspace: configured,
    )
    monkeypatch.delenv(
        implementation_daemon.IMPLEMENTATION_PROVIDER_ENV,
        raising=False,
    )
    monkeypatch.delenv("IMPLEMENTATION_DAEMON_COMMAND", raising=False)
    monkeypatch.delenv(
        implementation_daemon.PRODUCTION_PROVIDER_ROUTE_ENABLED_ENV,
        raising=False,
    )
    monkeypatch.delenv(
        implementation_daemon.PRODUCTION_PROVIDER_ALLOW_RAW_COMMAND_ENV,
        raising=False,
    )
    monkeypatch.delenv(
        implementation_daemon.LLM_MERGE_RESOLVER_COMMAND_ENV,
        raising=False,
    )


def _prompt_task(**overrides) -> PortalTask:
    payload = {
        "task_id": "ASE-001",
        "title": "Implement a prompt-only supervisor entrypoint",
        "status": "ready",
        "completion": "manual",
        "priority": "high",
        "track": "entrypoints",
        "outputs": ["ipfs_accelerate_py/agent_supervisor/entrypoints/api.py"],
        "validation": ["python -m pytest test/api/test_prompt_entrypoint.py -q"],
        "acceptance": "A prompt can launch the inferred supervisor workflow.",
    }
    payload.update(overrides)
    return PortalTask(**payload)


def test_default_implementation_provider_prefers_grok(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _clear_provider_overrides(monkeypatch)
    monkeypatch.setattr(
        implementation_daemon,
        "_grok_cli_available",
        lambda: True,
    )
    monkeypatch.setattr(
        implementation_daemon,
        "_grok_binary",
        lambda: "/opt/providers/grok",
    )
    monkeypatch.setattr(
        implementation_daemon,
        "_grok_cli_command",
        lambda *, workspace_path, model_override="": [
            "/opt/providers/grok-runner",
            str(workspace_path),
        ],
    )
    monkeypatch.setattr(
        implementation_daemon,
        "_goose_meta_spark_available",
        lambda: False,
    )
    monkeypatch.setattr(
        implementation_daemon.shutil,
        "which",
        lambda name: "/usr/local/bin/codex" if name == "codex" else None,
    )

    command = _daemon(tmp_path)._build_implementation_command(tmp_path)

    assert command[:2] == [
        "/opt/providers/grok-runner",
        str(tmp_path.resolve()),
    ]
    fallback_index = command.index("--codex-fallback-command-json")
    fallback_command = json.loads(command[fallback_index + 1])
    assert fallback_command[:2] == ["/usr/local/bin/codex", "exec"]
    assert fallback_command[fallback_command.index("-m") + 1] == "gpt-5.6-terra"
    assert 'model_reasoning_effort="medium"' in fallback_command
    assert fallback_command[-1] == "-"


def test_ordinary_prompt_task_uses_grok_with_codex_fallback(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _clear_provider_overrides(monkeypatch)
    monkeypatch.setattr(
        implementation_daemon,
        "_grok_cli_available",
        lambda: True,
    )
    monkeypatch.setattr(
        implementation_daemon,
        "_grok_binary",
        lambda: "/opt/providers/grok",
    )
    monkeypatch.setattr(
        implementation_daemon,
        "_grok_cli_command",
        lambda *, workspace_path, model_override="": [
            "/opt/providers/grok-runner",
            str(workspace_path),
        ],
    )
    monkeypatch.setattr(
        implementation_daemon,
        "_goose_meta_spark_available",
        lambda: False,
    )
    monkeypatch.setattr(
        implementation_daemon.shutil,
        "which",
        lambda name: "/usr/local/bin/codex" if name == "codex" else None,
    )
    daemon = _daemon(tmp_path)
    task = _prompt_task()

    assert daemon._production_provider_route_enabled(task) is False
    command = daemon._build_implementation_command(tmp_path, task=task)

    assert command[:2] == [
        "/opt/providers/grok-runner",
        str(tmp_path.resolve()),
    ]
    fallback_index = command.index("--codex-fallback-command-json")
    fallback_command = json.loads(command[fallback_index + 1])
    assert fallback_command[:2] == ["/usr/local/bin/codex", "exec"]
    assert fallback_command[fallback_command.index("-m") + 1] == "gpt-5.6-terra"
    assert 'model_reasoning_effort="medium"' in fallback_command


def test_systemd_minimal_path_still_selects_user_local_grok(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _clear_provider_overrides(monkeypatch)
    fake_home = tmp_path / "home"
    fake_grok = fake_home / ".local" / "bin" / "grok"
    fake_grok.parent.mkdir(parents=True)
    fake_grok.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake_grok.chmod(0o700)
    auth_path = fake_home / ".grok" / "auth.json"
    auth_path.parent.mkdir(parents=True)
    auth_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv(
        "PATH",
        "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    )
    for name in (
        "ipfs_accelerate_py_GROK_CLI_CMD",
        "IPFS_ACCELERATE_PY_GROK_CLI_CMD",
        "IPFS_DATASETS_PY_GROK_CLI_CMD",
        implementation_daemon._GROK_BIN_ENV,
        "GROK_CLI_CMD",
        "GROK_BIN",
        "GROK_HOME",
        "XAI_API_KEY",
        "ipfs_accelerate_py_XAI_API_KEY",
        "IPFS_ACCELERATE_PY_XAI_API_KEY",
        "IPFS_DATASETS_PY_XAI_API_KEY",
        "GROK_AUTH_PROVIDER_COMMAND",
    ):
        monkeypatch.delenv(name, raising=False)
    real_which = implementation_daemon.shutil.which
    monkeypatch.setattr(
        implementation_daemon.shutil,
        "which",
        lambda name: (
            "/usr/local/bin/codex"
            if name == "codex"
            else real_which(name)
        ),
    )
    daemon = _daemon(tmp_path)

    command = daemon._build_implementation_command(
        tmp_path,
        task=_prompt_task(),
    )

    assert command[0] == implementation_daemon.sys.executable
    assert command[1].endswith("grok_cli_runner.py")
    assert command[command.index("--grok-bin") + 1] == str(fake_grok)
    assert command[command.index("--model") + 1] == "grok-4.5"
    fallback_index = command.index("--codex-fallback-command-json")
    assert json.loads(command[fallback_index + 1])[:2] == [
        "/usr/local/bin/codex",
        "exec",
    ]


def test_default_provider_does_not_predispatch_codex_when_grok_is_unavailable(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _clear_provider_overrides(monkeypatch)
    monkeypatch.setattr(
        implementation_daemon,
        "_grok_cli_available",
        lambda: False,
    )
    monkeypatch.setattr(
        implementation_daemon,
        "_goose_meta_spark_available",
        lambda: False,
    )
    monkeypatch.setattr(
        implementation_daemon.shutil,
        "which",
        lambda name: "/usr/local/bin/codex" if name == "codex" else None,
    )

    with pytest.raises(RuntimeError, match="verified Grok quota-exhaustion"):
        _daemon(tmp_path)._build_implementation_command(tmp_path)


def test_quota_fallback_ignores_general_codex_model_overrides(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv(implementation_daemon._CODEX_MODEL_ENV, "gpt-5.6-sol")
    monkeypatch.setenv(
        implementation_daemon._CODEX_REASONING_EFFORT_ENV,
        "high",
    )

    command = implementation_daemon._codex_quota_fallback_command(
        codex="/usr/local/bin/codex",
        workspace_path=tmp_path,
    )

    assert command[command.index("-m") + 1] == "gpt-5.6-terra"
    assert 'model_reasoning_effort="medium"' in command
    assert 'model_reasoning_effort="high"' not in command


@pytest.mark.parametrize("provider", ["codex", "openai"])
def test_explicit_codex_alias_never_dispatches_copilot(
    tmp_path: Path,
    monkeypatch,
    provider: str,
) -> None:
    _clear_provider_overrides(monkeypatch)
    monkeypatch.setenv(implementation_daemon.IMPLEMENTATION_PROVIDER_ENV, provider)
    monkeypatch.setattr(implementation_daemon, "_grok_cli_available", lambda: True)
    monkeypatch.setattr(
        implementation_daemon,
        "_goose_meta_spark_available",
        lambda: True,
    )
    monkeypatch.setattr(implementation_daemon, "_copilot_has_auth", lambda: True)
    monkeypatch.setattr(
        implementation_daemon.shutil,
        "which",
        lambda name: f"/opt/providers/{name}" if name in {"codex", "copilot"} else None,
    )

    command = _daemon(tmp_path)._build_implementation_command(tmp_path)

    assert command[:2] == ["/opt/providers/codex", "exec"]
    assert all("copilot" not in argument for argument in command)


def test_explicit_copilot_never_dispatches_codex(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _clear_provider_overrides(monkeypatch)
    monkeypatch.setenv(
        implementation_daemon.IMPLEMENTATION_PROVIDER_ENV,
        "copilot",
    )
    monkeypatch.setattr(implementation_daemon, "_grok_cli_available", lambda: True)
    monkeypatch.setattr(
        implementation_daemon,
        "_goose_meta_spark_available",
        lambda: True,
    )
    monkeypatch.setattr(implementation_daemon, "_copilot_has_auth", lambda: True)
    monkeypatch.setattr(
        implementation_daemon.shutil,
        "which",
        lambda name: f"/opt/providers/{name}" if name in {"codex", "copilot"} else None,
    )

    command = _daemon(tmp_path)._build_implementation_command(tmp_path)

    assert command[:2] == ["bash", "-lc"]
    assert command[4] == ""
    assert command[5] == "/opt/providers/copilot"


def test_unauthenticated_grok_binary_does_not_authorize_codex(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _clear_provider_overrides(monkeypatch)
    monkeypatch.setattr(
        implementation_daemon,
        "_grok_binary",
        lambda: "/opt/providers/grok",
    )
    monkeypatch.setattr(
        llm_router,
        "_grok_cli_auth_available",
        lambda: False,
    )
    monkeypatch.setattr(
        llm_router,
        "get_llm_provider",
        lambda _provider: (_ for _ in ()).throw(
            AssertionError("provider construction must follow authentication")
        ),
    )
    monkeypatch.setattr(
        implementation_daemon,
        "_goose_meta_spark_available",
        lambda: False,
    )
    monkeypatch.setattr(
        implementation_daemon.shutil,
        "which",
        lambda name: "/usr/local/bin/codex" if name == "codex" else None,
    )

    with pytest.raises(RuntimeError, match="verified Grok quota-exhaustion"):
        _daemon(tmp_path)._build_implementation_command(tmp_path)


def test_grok_provider_construction_failure_does_not_authorize_codex(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _clear_provider_overrides(monkeypatch)
    monkeypatch.setattr(
        implementation_daemon,
        "_grok_binary",
        lambda: "/opt/providers/grok",
    )
    monkeypatch.setattr(
        llm_router,
        "_grok_cli_auth_available",
        lambda: True,
    )
    monkeypatch.setattr(
        llm_router,
        "get_llm_provider",
        lambda _provider: (_ for _ in ()).throw(
            RuntimeError("provider registry unavailable")
        ),
    )
    monkeypatch.setattr(
        implementation_daemon,
        "_goose_meta_spark_available",
        lambda: False,
    )
    monkeypatch.setattr(
        implementation_daemon.shutil,
        "which",
        lambda name: "/usr/local/bin/codex" if name == "codex" else None,
    )

    with pytest.raises(RuntimeError, match="verified Grok quota-exhaustion"):
        _daemon(tmp_path)._build_implementation_command(tmp_path)


@pytest.mark.parametrize("workspace_drifts_during_verifier", [False, True])
def test_typed_grok_quota_exhaustion_runs_terra_with_same_prompt(
    tmp_path: Path,
    monkeypatch,
    workspace_drifts_during_verifier: bool,
) -> None:
    _clear_provider_overrides(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "parent-only-openai-authority")
    codex_home = tmp_path.parent / f"codex-home-{tmp_path.name}"
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    monkeypatch.setattr(
        implementation_daemon,
        "_grok_cli_available",
        lambda: True,
    )
    monkeypatch.setattr(
        implementation_daemon,
        "_grok_binary",
        lambda: "/opt/providers/grok",
    )
    monkeypatch.setattr(
        implementation_daemon,
        "_goose_meta_spark_available",
        lambda: False,
    )
    monkeypatch.setattr(
        implementation_daemon.shutil,
        "which",
        lambda name: "/usr/local/bin/codex" if name == "codex" else None,
    )

    command = _daemon(tmp_path)._build_implementation_command(tmp_path)
    prompt = "repair the failed implementation"
    grok_calls: list[list[str]] = []
    codex_calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_grok(argv, **kwargs):
        grok_calls.append(list(argv))
        assert "OPENAI_API_KEY" not in kwargs["env"]
        assert "CODEX_HOME" not in kwargs["env"]
        prompt_path = Path(argv[argv.index("--prompt-file") + 1])
        assert prompt_path.read_text(encoding="utf-8") == prompt
        assert argv[argv.index("--output-format") + 1] == "streaming-json"
        return 23

    def fake_run(argv, **kwargs):
        codex_calls.append((list(argv), dict(kwargs)))
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(grok_cli_runner.sys, "stdin", io.StringIO(prompt))
    monkeypatch.setattr(
        grok_cli_runner,
        "_run_grok_with_typed_failure_capture",
        fake_grok,
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_terminal_grok_failure_type_from_isolated_home",
        lambda *_args, **_kwargs: "usage_pool_exhausted",
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_independently_verify_grok_quota",
        lambda **_kwargs: "usage_pool_exhausted",
    )
    real_fingerprint = grok_cli_runner._workspace_content_fingerprint
    fingerprint_calls = 0

    def tracked_fingerprint(workspace: Path) -> str:
        nonlocal fingerprint_calls
        fingerprint_calls += 1
        value = real_fingerprint(workspace)
        if workspace_drifts_during_verifier and fingerprint_calls == 3:
            return "verifier-window-drift"
        return value

    monkeypatch.setattr(
        grok_cli_runner,
        "_workspace_content_fingerprint",
        tracked_fingerprint,
    )
    codex_home.mkdir(exist_ok=True)
    (codex_home / "auth.json").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(grok_cli_runner.subprocess, "run", fake_run)

    returncode = grok_cli_runner.main(command[2:])

    assert fingerprint_calls == 3
    assert len(grok_calls) == 1
    assert grok_calls[0][0] == "/opt/providers/grok"
    if workspace_drifts_during_verifier:
        assert returncode == 23
        assert codex_calls == []
        return
    assert returncode == 0
    assert len(codex_calls) == 1
    fallback_command, fallback_kwargs = codex_calls[0]
    assert fallback_command[:2] == ["/usr/local/bin/codex", "exec"]
    assert fallback_command[fallback_command.index("-m") + 1] == "gpt-5.6-terra"
    assert 'model_reasoning_effort="medium"' in fallback_command
    assert fallback_kwargs["cwd"] == tmp_path.resolve()
    assert fallback_kwargs["input"] == prompt
    assert fallback_kwargs["text"] is True
    assert "OPENAI_API_KEY" not in fallback_kwargs["env"]
    assert fallback_kwargs["env"]["CODEX_HOME"] == str(codex_home)
    assert fallback_kwargs["env"]["PATH"] == "/usr/bin:/bin"


def test_workspace_alias_audits_reject_hardlinks_and_descendant_mounts(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "peer-auth.json"
    outside.write_text("peer authority\n", encoding="utf-8")
    alias = workspace / "innocent.json"
    alias.hardlink_to(outside)

    assert grok_cli_runner._workspace_regular_file_hardlinks(workspace) == (
        alias,
    )

    nested = workspace / "mounted peer"
    mountinfo = tmp_path / "mountinfo"
    escaped_nested = str(nested).replace(" ", r"\040")
    mountinfo.write_text(
        "24 1 0:21 / / rw - ext4 /dev/root rw\n"
        f"25 24 0:22 / {escaped_nested} rw - tmpfs tmpfs rw\n",
        encoding="utf-8",
    )
    assert grok_cli_runner._workspace_descendant_mountpoints(
        workspace,
        mountinfo_path=mountinfo,
    ) == (nested,)


def test_default_merge_resolver_nonquota_failure_cannot_reach_terra(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _clear_provider_overrides(monkeypatch)
    grok = tmp_path / "grok"
    grok.write_text("#!/bin/sh\nexit 23\n", encoding="utf-8")
    grok.chmod(0o700)
    route = shlex.split(
        implementation_daemon.default_llm_merge_resolver_command()
    )
    assert route[:3] == [
        implementation_daemon.sys.executable,
        "-m",
        "ipfs_accelerate_py.agent_supervisor.grok_cli_runner",
    ]
    route.extend(["--grok-bin", str(grok)])
    monkeypatch.setattr(grok_cli_runner.sys, "stdin", io.StringIO("repair"))
    monkeypatch.setattr(
        grok_cli_runner,
        "_run_grok_with_typed_failure_capture",
        lambda *_args, **_kwargs: 23,
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_terminal_grok_failure_type_from_isolated_home",
        lambda *_args, **_kwargs: "network_error",
    )
    monkeypatch.setattr(
        grok_cli_runner.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("Terra fallback must not run for non-quota failure")
        ),
    )

    assert grok_cli_runner.main(route[3:]) == 23


def test_shipped_ops_launchers_use_canonical_grok_quota_route(
    tmp_path: Path,
) -> None:
    from scripts.ops import (
        ai_service_catalog_supervisor,
        asref_module_refactor_supervisor,
    )
    from scripts.ops.agent_supervisor import asref_multi_lane

    routes = []
    for arguments in (
        asref_multi_lane._common_args(
            repo_root=tmp_path,
            runtime_root=tmp_path / "asref-runtime",
            merge_branch="main",
            enable_objective_refill=False,
            refill_open_task_threshold=1,
        ),
        asref_module_refactor_supervisor._common_args(
            repo_root=tmp_path,
            runtime_root=tmp_path / "module-runtime",
            merge_branch="main",
            enable_objective_refill=False,
            refill_open_task_threshold=1,
        ),
        ai_service_catalog_supervisor._common_args(
            runtime_root=tmp_path / "catalog-runtime",
            enable_objective_refill=False,
            refill_open_task_threshold=1,
        ),
    ):
        routes.append(
            shlex.split(
                arguments[arguments.index("--llm-merge-resolver-command") + 1]
            )
        )

    for route in routes:
        assert route[1:3] == [
            "-m",
            "ipfs_accelerate_py.agent_supervisor.grok_cli_runner",
        ]
        assert route[route.index("--model") + 1] == "grok-4.5"
        fallback = json.loads(
            route[route.index("--codex-fallback-command-json") + 1]
        )
        assert Path(fallback[0]).is_absolute()
        assert fallback[fallback.index("-m") + 1] == "gpt-5.6-terra"
        assert 'model_reasoning_effort="medium"' in fallback
        assert "--ignore-user-config" in fallback
        assert "--ephemeral" in fallback
        assert all("copilot" not in item.casefold() for item in route)


@pytest.mark.parametrize(
    "failure_types",
    [
        set(),
        {"unknown"},
        {"unauthorized"},
        {"rate_limited"},
        {"global_rate_limit"},
        {"concurrency_limit"},
        {"service_unavailable"},
        {"network_error"},
        {"usage_pool_exhausted", "network_error"},
    ],
)
def test_nonquota_grok_failure_never_runs_codex(
    tmp_path: Path,
    monkeypatch,
    failure_types: set[str],
) -> None:
    _clear_provider_overrides(monkeypatch)
    monkeypatch.setattr(implementation_daemon, "_grok_cli_available", lambda: True)
    monkeypatch.setattr(
        implementation_daemon, "_grok_binary", lambda: "/opt/providers/grok"
    )
    monkeypatch.setattr(
        implementation_daemon.shutil,
        "which",
        lambda name: "/usr/local/bin/codex" if name == "codex" else None,
    )
    command = _daemon(tmp_path)._build_implementation_command(tmp_path)
    monkeypatch.setattr(grok_cli_runner.sys, "stdin", io.StringIO("repair"))
    monkeypatch.setattr(
        grok_cli_runner,
        "_run_grok_with_typed_failure_capture",
        lambda *_args, **_kwargs: 23,
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_terminal_grok_failure_type_from_isolated_home",
        lambda *_args, **_kwargs: (
            next(iter(failure_types)) if len(failure_types) == 1 else ""
        ),
    )
    monkeypatch.setattr(
        grok_cli_runner.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("Codex fallback must not run")
        ),
    )

    assert grok_cli_runner.main(command[2:]) == 23


def _native_grok_failure_event(error_type: str, message: str = "") -> str:
    return json.dumps(
        {
            "method": "_x.ai/session/update",
            "params": {
                "update": {
                    "sessionUpdate": "retry_state",
                    "type": "failed",
                    "error_type": error_type,
                    "message": message,
                }
            },
        }
    )


def _write_native_grok_session(
    grok_home: Path,
    *,
    failure_message: str = grok_cli_runner._LEGACY_GROK_BALANCE_EXHAUSTED_MESSAGE,
    failure_type: str = "api",
    terminal_message: str | None = None,
    model: str = "grok-4.5",
    include_user_message: bool = True,
    injected_updates: tuple[dict[str, object], ...] = (),
) -> Path:
    session_id = str(uuid.uuid4())
    session_dir = grok_home / "sessions" / "workspace" / session_id
    session_dir.mkdir(parents=True)
    updates = [
        {
            "sessionUpdate": "retry_state",
            "type": "failed",
            "error_type": failure_type,
            "message": failure_message,
        },
        *(
            [
                {
                    "sessionUpdate": "user_message_chunk",
                    "content": {"type": "text", "text": "fixed prompt"},
                    "_meta": {"modelId": model},
                }
            ]
            if include_user_message
            else []
        ),
        *injected_updates,
        {
            "sessionUpdate": "turn_completed",
            "stop_reason": "error",
            "agent_result": (
                failure_message if terminal_message is None else terminal_message
            ),
        },
    ]
    record = session_dir / "updates.jsonl"
    record.write_text(
        "".join(
            json.dumps(
                {
                    "method": "_x.ai/session/update",
                    "params": {"sessionId": session_id, "update": update},
                }
            )
            + "\n"
            for update in updates
        ),
        encoding="utf-8",
    )
    (session_dir / "summary.json").write_text(
        json.dumps(
            {
                "info": {"id": session_id, "cwd": "/isolated/workspace"},
                "current_model_id": model,
                "grok_home": str(grok_home.resolve()),
            }
        ),
        encoding="utf-8",
    )
    return record


def test_isolated_native_terminal_record_authorizes_only_exact_quota(tmp_path: Path) -> None:
    grok_home = tmp_path / "grok-home"
    grok_home.mkdir()
    _write_native_grok_session(grok_home)

    assert (
        grok_cli_runner._terminal_grok_failure_type_from_isolated_home(grok_home)
        == "usage_pool_exhausted"
    )


def test_initial_quota_record_without_user_chunk_uses_exact_summary_model(
    tmp_path: Path,
) -> None:
    """Observed initial 402 sessions have only retry + terminal updates."""

    grok_home = tmp_path / "grok-home"
    grok_home.mkdir()
    record = _write_native_grok_session(
        grok_home,
        include_user_message=False,
    )

    assert (
        grok_cli_runner._terminal_grok_failure_type_from_isolated_home(
            grok_home,
            expected_session_id=record.parent.name,
        )
        == "usage_pool_exhausted"
    )


@pytest.mark.parametrize(
    "mutation",
    [
        {"terminal_message": "terminal transport failure: connection reset"},
        {"model": "grok-4"},
        {
            "injected_updates": (
                {"sessionUpdate": "tool_call", "toolCallId": "forged"},
            )
        },
    ],
)
def test_isolated_native_terminal_record_rejects_inexact_or_active_session(
    tmp_path: Path,
    mutation: dict[str, object],
) -> None:
    grok_home = tmp_path / "grok-home"
    grok_home.mkdir()
    _write_native_grok_session(grok_home, **mutation)

    assert not grok_cli_runner._terminal_grok_failure_type_from_isolated_home(
        grok_home
    )


def test_isolated_native_terminal_record_rejects_historical_quota(
    tmp_path: Path,
) -> None:
    grok_home = tmp_path / "grok-home"
    grok_home.mkdir()
    _write_native_grok_session(
        grok_home,
        injected_updates=(
            {
                "sessionUpdate": "retry_state",
                "type": "failed",
                "error_type": "network_error",
                "message": "connection reset",
            },
        ),
        terminal_message="connection reset",
    )

    assert not grok_cli_runner._terminal_grok_failure_type_from_isolated_home(
        grok_home
    )


@pytest.mark.parametrize(
    ("error_type", "expected"),
    [
        ("usage_pool_exhausted", "usage_pool_exhausted"),
        ("usage_limit_reached", "usage_limit_reached"),
        ("rate_limited", "rate_limited"),
        ("unauthorized", "unauthorized"),
        ("network_error", "network_error"),
    ],
)
def test_native_grok_failure_classifier_is_exact(
    error_type: str,
    expected: str,
) -> None:
    assert (
        grok_cli_runner._grok_failure_type_from_stream_event(
            _native_grok_failure_event(error_type)
        )
        == expected
    )


def test_legacy_native_402_balance_event_is_typed_quota() -> None:
    event = _native_grok_failure_event(
        "api",
        "API error (status 402 Payment Required): "
        "Grok Build usage balance exhausted",
    )
    assert (
        grok_cli_runner._grok_failure_type_from_stream_event(event)
        == "usage_pool_exhausted"
    )


def test_live_stream_capture_does_not_authorize_native_quota_event(capsys) -> None:
    event = _native_grok_failure_event("usage_pool_exhausted")
    command = [
        grok_cli_runner.sys.executable,
        "-c",
        (
            "import sys; "
            f"print({event!r}); "
            "print('bounded diagnostic', file=sys.stderr); "
            "raise SystemExit(17)"
        ),
    ]

    returncode = grok_cli_runner._run_grok_with_typed_failure_capture(
        command,
        env=grok_cli_runner.os.environ.copy(),
    )

    captured = capsys.readouterr()
    assert returncode == 17
    assert event in captured.out
    assert "bounded diagnostic" in captured.err


@pytest.mark.parametrize(
    "event",
    [
        '{"error_type":"usage_pool_exhausted"}',
        json.dumps(
            {
                "method": "_x.ai/session/update",
                "params": {
                    "update": {
                        "sessionUpdate": "agent_message_chunk",
                        "type": "failed",
                        "error_type": "usage_pool_exhausted",
                    }
                },
            }
        ),
        _native_grok_failure_event("api", "quota exhausted"),
        _native_grok_failure_event("api", "HTTP 429 rate limit"),
        json.dumps({"type": "error", "message": "quota exhausted"}),
        "not-json",
    ],
)
def test_untrusted_or_inexact_quota_text_is_not_authoritative(event: str) -> None:
    assert (
        grok_cli_runner._grok_failure_type_from_stream_event(event)
        not in grok_cli_runner.GROK_QUOTA_ERROR_TYPES
    )


@pytest.mark.parametrize(
    "suffix",
    [
        ["--model", "gpt-5.6-sol"],
        ["--config", 'model_reasoning_effort="high"'],
        ["--cd", "/other"],
        ["--oss"],
        ["--add-dir", "/"],
    ],
)
def test_runner_rejects_fallback_argv_that_widens_terra_medium_policy(
    tmp_path: Path,
    suffix: list[str],
) -> None:
    command = implementation_daemon._codex_quota_fallback_command(
        codex="/usr/local/bin/codex",
        workspace_path=tmp_path,
    )
    command[-1:-1] = suffix

    with pytest.raises(ValueError):
        grok_cli_runner._parse_codex_fallback_command(json.dumps(command))


def test_default_route_rejects_non_grok_45_primary(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _clear_provider_overrides(monkeypatch)
    monkeypatch.setattr(implementation_daemon, "_grok_cli_available", lambda: True)
    monkeypatch.setattr(
        implementation_daemon, "_grok_binary", lambda: "/opt/providers/grok"
    )
    monkeypatch.setattr(
        implementation_daemon.shutil,
        "which",
        lambda name: "/usr/local/bin/codex" if name == "codex" else None,
    )
    command = _daemon(tmp_path)._build_implementation_command(tmp_path)
    command[command.index("--model") + 1] = "grok-4"
    monkeypatch.setattr(grok_cli_runner.sys, "stdin", io.StringIO("repair"))
    monkeypatch.setattr(
        grok_cli_runner,
        "_run_grok_with_typed_failure_capture",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("wrong primary model must not launch")
        ),
    )

    assert grok_cli_runner.main(command[2:]) == 2


def test_explicit_grok_runtime_failure_does_not_fall_back(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _clear_provider_overrides(monkeypatch)
    monkeypatch.setenv(
        implementation_daemon.IMPLEMENTATION_PROVIDER_ENV,
        "grok",
    )
    monkeypatch.setattr(
        implementation_daemon,
        "_grok_cli_available",
        lambda: True,
    )
    monkeypatch.setattr(
        implementation_daemon,
        "_grok_binary",
        lambda: "/opt/providers/grok",
    )
    monkeypatch.setattr(
        implementation_daemon.shutil,
        "which",
        lambda name: "/usr/local/bin/codex" if name == "codex" else None,
    )

    command = _daemon(tmp_path)._build_implementation_command(tmp_path)
    calls: list[list[str]] = []

    def fake_run(argv, **_kwargs):
        calls.append(list(argv))
        return subprocess.CompletedProcess(argv, 29)

    monkeypatch.setattr(
        grok_cli_runner.sys,
        "stdin",
        io.StringIO("use Grok or fail"),
    )
    monkeypatch.setattr(grok_cli_runner.subprocess, "run", fake_run)

    assert "--codex-fallback-command-json" not in command
    assert grok_cli_runner.main(command[2:]) == 29
    assert len(calls) == 1
    assert calls[0][0] == "/opt/providers/grok"


def test_launch_defaults_do_not_override_grok_first_provider_inference() -> None:
    daemon_args = implementation_daemon.parse_args([])
    supervisor_args = implementation_supervisor.parse_args([])

    assert daemon_args.implementation_command == ""
    assert supervisor_args.implementation_command == ""
