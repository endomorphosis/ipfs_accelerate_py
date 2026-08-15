from __future__ import annotations

import io
import json
import shutil
import subprocess
from pathlib import Path

import ipfs_accelerate_py.llm_router as llm_router
import pytest
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


def _set_runner_grok_identity(monkeypatch, grok_bin: str) -> None:
    monkeypatch.setattr(
        implementation_daemon,
        "_grok_binary",
        lambda: grok_bin,
    )
    monkeypatch.setattr(llm_router, "find_grok_cli", lambda: grok_bin)


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
    _set_runner_grok_identity(monkeypatch, "/opt/providers/grok")
    monkeypatch.setattr(
        implementation_daemon,
        "_grok_cli_command",
        lambda *, workspace_path, grok_model=None: [
            "/opt/providers/grok-runner",
            str(workspace_path),
            "--model",
            str(grok_model),
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
        lambda name: "/opt/providers/codex" if name == "codex" else None,
    )

    command = _daemon(tmp_path)._build_implementation_command(tmp_path)

    assert command[:2] == [
        "/opt/providers/grok-runner",
        str(tmp_path.resolve()),
    ]
    assert command[command.index("--model") + 1] == "grok-4.6"
    fallback_index = command.index("--codex-fallback-command-json")
    fallback_command = json.loads(command[fallback_index + 1])
    assert fallback_command[:2] == ["/opt/providers/codex", "exec"]
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
        lambda *, workspace_path, grok_model=None: [
            "/opt/providers/grok-runner",
            str(workspace_path),
            "--model",
            str(grok_model),
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
        lambda name: "/opt/providers/codex" if name == "codex" else None,
    )
    daemon = _daemon(tmp_path)
    task = _prompt_task()

    assert daemon._production_provider_route_enabled(task) is False
    command = daemon._build_implementation_command(tmp_path, task=task)

    assert command[:2] == [
        "/opt/providers/grok-runner",
        str(tmp_path.resolve()),
    ]
    assert command[command.index("--model") + 1] == "grok-4.6"
    fallback_index = command.index("--codex-fallback-command-json")
    fallback_command = json.loads(command[fallback_index + 1])
    assert fallback_command[:2] == ["/opt/providers/codex", "exec"]


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
    fallback_index = command.index("--codex-fallback-command-json")
    assert json.loads(command[fallback_index + 1])[:2] == [
        "/usr/local/bin/codex",
        "exec",
    ]


def test_default_route_does_not_use_codex_when_grok_is_unavailable(
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
        lambda name: "/opt/providers/codex" if name == "codex" else None,
    )

    with pytest.raises(RuntimeError, match="Grok 4.5 primary is unavailable"):
        _daemon(tmp_path)._build_implementation_command(tmp_path)


def test_codex_before_copilot_uses_local_default_model(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv(implementation_daemon._CODEX_MODEL_ENV, raising=False)

    command = implementation_daemon._copilot_fallback_command(
        codex="/opt/providers/codex",
        copilot="/opt/providers/copilot",
        workspace_path=tmp_path,
    )

    # Positional arguments after the embedded shell program are stable inputs
    # consumed by that program; the Codex model is its fourth argument.
    assert command[4:8] == [
        "/opt/providers/codex",
        "/opt/providers/copilot",
        str(tmp_path),
        "gpt-5.6-sol",
    ]


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
        lambda name: "/opt/providers/codex" if name == "codex" else None,
    )

    with pytest.raises(RuntimeError, match="Grok 4.5 primary is unavailable"):
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
        lambda name: "/opt/providers/codex" if name == "codex" else None,
    )

    with pytest.raises(RuntimeError, match="Grok 4.5 primary is unavailable"):
        _daemon(tmp_path)._build_implementation_command(tmp_path)


def test_confirmed_grok_quota_exhaustion_runs_terra_medium_with_same_prompt(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _clear_provider_overrides(monkeypatch)
    monkeypatch.setattr(
        implementation_daemon,
        "_grok_cli_available",
        lambda: True,
    )
    _set_runner_grok_identity(monkeypatch, "/opt/providers/grok")
    monkeypatch.setattr(
        implementation_daemon,
        "_goose_meta_spark_available",
        lambda: False,
    )
    monkeypatch.setattr(
        implementation_daemon.shutil,
        "which",
        lambda name: "/opt/providers/codex" if name == "codex" else None,
    )

    command = _daemon(tmp_path)._build_implementation_command(tmp_path)
    fallback = json.loads(
        command[command.index("--codex-fallback-command-json") + 1]
    )
    assert fallback[fallback.index("-m") + 1] == "gpt-5.6-terra"
    assert 'model_reasoning_effort="medium"' in fallback
    prompt = "repair the failed implementation"
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_grok(argv, *, env):
        calls.append((list(argv), {"env": dict(env)}))
        prompt_path = Path(argv[argv.index("--prompt-file") + 1])
        assert prompt_path.read_text(encoding="utf-8") == prompt
        return (
            23,
            'Internal error: {"message":"API error (status 402 Payment '
            'Required): Grok Build usage balance exhausted",'
            '"http_status":402}\n',
        )

    def fake_run(argv, **kwargs):
        calls.append((list(argv), dict(kwargs)))
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(grok_cli_runner.sys, "stdin", io.StringIO(prompt))
    monkeypatch.setattr(
        grok_cli_runner,
        "_git_repository_effect_identity",
        lambda _workspace: ("a" * 40, b""),
    )
    monkeypatch.setattr(grok_cli_runner, "_run_grok_streaming", fake_grok)
    monkeypatch.setattr(grok_cli_runner.subprocess, "run", fake_run)

    returncode = grok_cli_runner.main(command[2:])

    assert returncode == 0
    assert len(calls) == 2
    assert calls[0][0][0] == "/opt/providers/grok"
    assert calls[1][0][:2] == ["/opt/providers/codex", "exec"]
    assert calls[1][1]["cwd"] == tmp_path.resolve()
    assert calls[1][1]["input"] == prompt
    assert calls[1][1]["text"] is True


@pytest.mark.parametrize(
    "diagnostic",
    [
        "ordinary implementation failure",
        "HTTP 429 Too Many Requests from Grok",
        "Grok authentication failed with HTTP 401",
        "Grok network timeout",
    ],
)
def test_non_quota_grok_failure_never_runs_codex(
    tmp_path: Path,
    monkeypatch,
    diagnostic: str,
) -> None:
    _clear_provider_overrides(monkeypatch)
    monkeypatch.setattr(implementation_daemon, "_grok_cli_available", lambda: True)
    _set_runner_grok_identity(monkeypatch, "/opt/providers/grok")
    monkeypatch.setattr(implementation_daemon, "_goose_meta_spark_available", lambda: False)
    monkeypatch.setattr(
        implementation_daemon.shutil,
        "which",
        lambda name: "/opt/providers/codex" if name == "codex" else None,
    )
    command = _daemon(tmp_path)._build_implementation_command(tmp_path)
    calls: list[list[str]] = []

    def fake_grok(argv, *, env):
        del env
        calls.append(list(argv))
        return 31, diagnostic

    monkeypatch.setattr(grok_cli_runner.sys, "stdin", io.StringIO("implement"))
    monkeypatch.setattr(
        grok_cli_runner,
        "_git_repository_effect_identity",
        lambda _workspace: ("a" * 40, b""),
    )
    monkeypatch.setattr(grok_cli_runner, "_run_grok_streaming", fake_grok)

    assert grok_cli_runner.main(command[2:]) == 31
    assert len(calls) == 1
    assert calls[0][0] == "/opt/providers/grok"


def test_grok_quota_after_workspace_effect_never_runs_codex(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _clear_provider_overrides(monkeypatch)
    monkeypatch.setattr(implementation_daemon, "_grok_cli_available", lambda: True)
    _set_runner_grok_identity(monkeypatch, "/opt/providers/grok")
    monkeypatch.setattr(implementation_daemon, "_goose_meta_spark_available", lambda: False)
    monkeypatch.setattr(
        implementation_daemon.shutil,
        "which",
        lambda name: "/opt/providers/codex" if name == "codex" else None,
    )
    command = _daemon(tmp_path)._build_implementation_command(tmp_path)
    calls: list[list[str]] = []
    repository_identities = iter(
        (("a" * 40, b""), ("a" * 40, b"? changed.py\0"))
    )

    def fake_grok(argv, *, env):
        del env
        calls.append(list(argv))
        return 32, "Grok Build usage balance exhausted; 402 Payment Required"

    monkeypatch.setattr(grok_cli_runner.sys, "stdin", io.StringIO("implement"))
    monkeypatch.setattr(
        grok_cli_runner,
        "_git_repository_effect_identity",
        lambda _workspace: next(repository_identities),
    )
    monkeypatch.setattr(grok_cli_runner, "_run_grok_streaming", fake_grok)

    assert grok_cli_runner.main(command[2:]) == 32
    assert len(calls) == 1


def test_repository_effect_identity_detects_real_ignored_effect(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".gitignore").write_text("ignored-effect.txt\n", encoding="utf-8")
    for argv in (
        ["git", "init"],
        ["git", "config", "user.name", "Fallback Fence Test"],
        ["git", "config", "user.email", "fallback-fence@example.invalid"],
        ["git", "add", ".gitignore"],
        ["git", "commit", "-m", "baseline"],
    ):
        subprocess.run(
            argv,
            cwd=repo,
            check=True,
            capture_output=True,
        )

    before = grok_cli_runner._git_repository_effect_identity(repo)
    (repo / "ignored-effect.txt").write_text(
        "changed by Grok\n",
        encoding="utf-8",
    )
    after = grok_cli_runner._git_repository_effect_identity(repo)

    assert before is not None
    assert before[1] == b""
    assert after is not None
    assert after[0] == before[0]
    assert b"ignored-effect.txt" in after[1]
    assert after != before


def test_grok_quota_after_committed_effect_never_runs_codex(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    daemon = _daemon(repo)
    for argv in (
        ["git", "init"],
        ["git", "config", "user.name", "Fallback Fence Test"],
        ["git", "config", "user.email", "fallback-fence@example.invalid"],
        ["git", "add", "tasks.todo.md"],
        ["git", "commit", "-m", "baseline"],
    ):
        subprocess.run(
            argv,
            cwd=repo,
            check=True,
            capture_output=True,
        )
    baseline_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()

    _clear_provider_overrides(monkeypatch)
    monkeypatch.setattr(implementation_daemon, "_grok_cli_available", lambda: True)
    _set_runner_grok_identity(monkeypatch, "/opt/providers/grok")
    monkeypatch.setattr(implementation_daemon, "_goose_meta_spark_available", lambda: False)
    monkeypatch.setattr(
        implementation_daemon.shutil,
        "which",
        lambda name: "/opt/providers/codex" if name == "codex" else None,
    )
    command = daemon._build_implementation_command(repo)

    def fake_grok(_argv, *, env):
        del env
        (repo / "committed-effect.txt").write_text(
            "Grok changed and committed this file.\n",
            encoding="utf-8",
        )
        subprocess.run(
            ["git", "add", "committed-effect.txt"],
            cwd=repo,
            check=True,
        )
        subprocess.run(
            ["git", "commit", "-m", "grok effect"],
            cwd=repo,
            check=True,
            capture_output=True,
        )
        return 33, "Grok Build usage balance exhausted; 402 Payment Required"

    monkeypatch.setattr(grok_cli_runner.sys, "stdin", io.StringIO("implement"))
    monkeypatch.setattr(grok_cli_runner, "_run_grok_streaming", fake_grok)

    assert grok_cli_runner.main(command[2:]) == 33
    committed_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    assert committed_head != baseline_head
    assert status == ""


def test_quota_fallback_command_rejects_model_or_effort_drift() -> None:
    valid = [
        "/usr/local/bin/codex",
        "exec",
        "--dangerously-bypass-approvals-and-sandbox",
        "-C",
        "/repo",
        "-m",
        "gpt-5.6-terra",
        "-c",
        'model_reasoning_effort="medium"',
        "-",
    ]
    assert grok_cli_runner._parse_codex_fallback_command(json.dumps(valid)) == valid
    model_drift = list(valid)
    model_drift[model_drift.index("-m") + 1] = "gpt-5.6-sol"
    effort_drift = list(valid)
    effort_drift[
        effort_drift.index('model_reasoning_effort="medium"')
    ] = 'model_reasoning_effort="high"'
    extra_model_drift = [*valid[:-1], "--model", "gpt-5.6-sol", "-"]
    for drifted in (model_drift, effort_drift, extra_model_drift):
        with pytest.raises(ValueError):
            grok_cli_runner._parse_codex_fallback_command(json.dumps(drifted))


def test_quota_fallback_rejects_non_grok_45_primary(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    discovered_codex = shutil.which("codex")
    assert discovered_codex is not None
    fallback = [
        discovered_codex,
        "exec",
        "--dangerously-bypass-approvals-and-sandbox",
        "-C",
        str(tmp_path),
        "-m",
        "gpt-5.6-terra",
        "-c",
        'model_reasoning_effort="medium"',
        "-",
    ]
    monkeypatch.setattr(
        grok_cli_runner,
        "_run_grok_streaming",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("non-4.5 Grok must fail before provider dispatch")
        ),
    )
    monkeypatch.setattr(llm_router, "find_grok_cli", lambda: "/bin/true")

    returncode = grok_cli_runner.main(
        [
            "--workspace",
            str(tmp_path),
            "--grok-bin",
            "/bin/true",
            "--model",
            "grok-not-4.5",
            "--codex-fallback-command-json",
            json.dumps(fallback),
        ]
    )

    assert returncode == 2
    assert "requires Grok primary model grok-4.6" in capsys.readouterr().err


def test_quota_fallback_rejects_unresolved_codex_executable(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    attacker_codex = tmp_path / "attacker" / "codex"
    attacker_codex.parent.mkdir()
    attacker_codex.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    attacker_codex.chmod(0o700)
    fallback = [
        str(attacker_codex),
        "exec",
        "--dangerously-bypass-approvals-and-sandbox",
        "-C",
        str(tmp_path),
        "-m",
        "gpt-5.6-terra",
        "-c",
        'model_reasoning_effort="medium"',
        "-",
    ]
    monkeypatch.setattr(llm_router, "find_grok_cli", lambda: "/bin/true")

    returncode = grok_cli_runner.main(
        [
            "--workspace",
            str(tmp_path),
            "--grok-bin",
            "/bin/true",
            "--model",
            "grok-4.6",
            "--codex-fallback-command-json",
            json.dumps(fallback),
        ]
    )

    assert returncode == 2
    assert "supervisor-discovered Codex CLI" in capsys.readouterr().err


def test_quota_fallback_rejects_grok_binary_override(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    discovered_codex = shutil.which("codex")
    assert discovered_codex is not None
    attacker_grok = tmp_path / "attacker-grok"
    attacker_grok.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    attacker_grok.chmod(0o700)
    fallback = [
        discovered_codex,
        "exec",
        "--dangerously-bypass-approvals-and-sandbox",
        "-C",
        str(tmp_path),
        "-m",
        "gpt-5.6-terra",
        "-c",
        'model_reasoning_effort="medium"',
        "-",
    ]
    monkeypatch.setattr(llm_router, "find_grok_cli", lambda: "/bin/true")

    returncode = grok_cli_runner.main(
        [
            "--workspace",
            str(tmp_path),
            "--grok-bin",
            str(attacker_grok),
            "--model",
            "grok-4.6",
            "--codex-fallback-command-json",
            json.dumps(fallback),
        ]
    )

    assert returncode == 2
    assert "supervisor-discovered Grok CLI" in capsys.readouterr().err


def test_quota_classifier_rejects_split_or_mixed_diagnostics() -> None:
    transcript = "\n".join(
        [
            "Grok implementation failed",
            "usage balance exhausted",
            "402 Payment Required",
        ]
    )

    assert grok_cli_runner._grok_quota_exhausted(transcript) is False


def test_quota_classifier_accepts_typed_xai_insufficient_quota() -> None:
    transcript = (
        '{"provider":"xAI","error":{"type":"insufficient_quota",'
        '"message":"quota exhausted"}}'
    )

    assert grok_cli_runner._grok_quota_exhausted(transcript) is True


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
        lambda name: "/opt/providers/codex" if name == "codex" else None,
    )

    command = _daemon(tmp_path)._build_implementation_command(tmp_path)
    calls: list[list[str]] = []

    def fake_grok(argv, *, env):
        del env
        calls.append(list(argv))
        return 29, "ordinary failure"

    monkeypatch.setattr(
        grok_cli_runner.sys,
        "stdin",
        io.StringIO("use Grok or fail"),
    )
    monkeypatch.setattr(grok_cli_runner, "_run_grok_streaming", fake_grok)

    assert "--codex-fallback-command-json" not in command
    assert grok_cli_runner.main(command[2:]) == 29
    assert len(calls) == 1
    assert calls[0][0] == "/opt/providers/grok"


def test_launch_defaults_do_not_override_grok_first_provider_inference() -> None:
    daemon_args = implementation_daemon.parse_args([])
    supervisor_args = implementation_supervisor.parse_args([])

    assert daemon_args.implementation_command == ""
    assert supervisor_args.implementation_command == ""
