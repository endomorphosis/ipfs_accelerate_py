from __future__ import annotations

import io
import json
import subprocess
from pathlib import Path

import ipfs_accelerate_py.llm_router as llm_router
import pytest
from ipfs_accelerate_py.agent_supervisor import grok_cli_runner, provider_failure_policy
from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    implementation_daemon,
    implementation_supervisor,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    ImplementationRetryDeferred,
    PortalTask,
    PortalTaskState,
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


def _record_valid_hard_quota_latch(
    daemon: TodoImplementationDaemon,
    *,
    task_id: str = "ROUTE-001",
    canonical_task_cid: str = "baguqeera-route-001",
    attempt: int = 1,
    returncode: int = 17,
    retry_at: str = "2999-01-01T00:00:00+00:00",
) -> dict[str, object]:
    nonce = "a" * 64
    command = [
        "/usr/bin/python3",
        "/opt/grok_cli_runner.py",
        "--model",
        "grok-4.5",
        "--grok-failure-receipt-nonce",
        nonce,
    ]
    receipt = provider_failure_policy.build_grok_failure_receipt(
        probe_stderr_text=("xAI Grok Build status 402: usage balance exhausted"),
        nonce=nonce,
        model="grok-4.5",
        probe_returncode=returncode,
        primary_dispatched=False,
    )
    daemon._record_event(
        "implementation_started",
        {
            "task_id": task_id,
            "canonical_task_cid": canonical_task_cid,
            "attempt": attempt,
            "command": command,
        },
    )
    start = daemon._iter_merge_lifecycle_events()[-1]
    daemon._record_event(
        "implementation_provider_exhausted",
        {
            "task_id": task_id,
            "canonical_task_cid": canonical_task_cid,
            "attempt": attempt,
            "returncode": returncode,
            "providers": ["grok"],
            "failure_class": "hard_quota_exhausted",
            "hard_quota_exhausted_providers": ["grok"],
            "retry_at": retry_at,
            "quota_fallback_authority": {
                "schema": (implementation_daemon.GROK_QUOTA_FALLBACK_AUTHORITY_SCHEMA),
                "primary_provider": "grok",
                "primary_model": "grok-4.5",
                "failure_class": "hard_quota_exhausted",
                "evidence_sha256": receipt["evidence_sha256"],
                "task_id": task_id,
                "canonical_task_cid": canonical_task_cid,
                "attempt": attempt,
                "primary_returncode": returncode,
                "start_event_id": start["event_id"],
                "start_sequence": start["sequence"],
                "command_sha256": daemon._implementation_command_identity(command),
                "runner_receipt_id": receipt["receipt_id"],
                "runner_receipt": receipt,
            },
        },
    )
    return receipt


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
        "_goose_meta_spark_available",
        lambda: False,
    )
    monkeypatch.setattr(
        implementation_daemon.shutil,
        "which",
        lambda name: "/opt/providers/codex" if name == "codex" else None,
    )

    command = _daemon(tmp_path)._build_implementation_command(tmp_path)

    assert command[0] == implementation_daemon.sys.executable
    assert command[1].endswith("grok_cli_runner.py")
    assert command[command.index("--model") + 1] == "grok-4.5"
    assert "--codex-fallback-command-json" not in command


def test_default_implementation_provider_fails_closed_without_grok(
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

    with pytest.raises(
        RuntimeError,
        match="Codex fallback is authorized only after a durable Grok",
    ):
        _daemon(tmp_path)._build_implementation_command(tmp_path)


def test_missing_grok_primary_uses_typed_non_consuming_deferral(
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
        "_grok_binary",
        lambda: "/opt/providers/grok",
    )
    daemon = _daemon(tmp_path)

    with pytest.raises(ImplementationRetryDeferred) as raised:
        daemon._require_primary_provider_readiness(None)

    assert raised.value.backoff_seconds == 300
    assert "Grok 4.5 primary is unavailable" in raised.value.reason
    assert not daemon.state_path.exists()


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

    with pytest.raises(
        RuntimeError,
        match="Codex fallback is authorized only after a durable Grok",
    ):
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
        lambda _provider: (_ for _ in ()).throw(RuntimeError("provider registry unavailable")),
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

    with pytest.raises(
        RuntimeError,
        match="Codex fallback is authorized only after a durable Grok",
    ):
        _daemon(tmp_path)._build_implementation_command(tmp_path)


def test_default_grok_runtime_failure_does_not_run_inline_codex(
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
        "_goose_meta_spark_available",
        lambda: False,
    )
    monkeypatch.setattr(
        implementation_daemon.shutil,
        "which",
        lambda name: "/opt/providers/codex" if name == "codex" else None,
    )

    command = _daemon(tmp_path)._build_implementation_command(tmp_path)
    prompt = "repair the failed implementation"
    calls: list[list[str]] = []

    def fake_probe(argv, *, env):
        calls.append(list(argv))
        prompt_path = Path(argv[argv.index("--prompt-file") + 1])
        assert prompt_path.read_text(encoding="utf-8") == prompt
        assert env
        return 23, "ordinary Grok execution failure"

    monkeypatch.setattr(grok_cli_runner.sys, "stdin", io.StringIO(prompt))
    monkeypatch.setattr(
        grok_cli_runner,
        "_run_isolated_grok_quota_probe",
        lambda _command, *, env: (0, ""),
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_run_grok_with_stderr_probe",
        fake_probe,
    )

    returncode = grok_cli_runner.main(command[2:])

    assert returncode == 23
    assert len(calls) == 1
    assert calls[0][0] == "/opt/providers/grok"


def test_durable_grok_hard_quota_latch_routes_exact_terra_medium(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _clear_provider_overrides(monkeypatch)
    monkeypatch.setenv(
        implementation_daemon._CODEX_MODEL_ENV,
        "hostile-model-override",
    )
    monkeypatch.setenv(
        implementation_daemon._CODEX_REASONING_EFFORT_ENV,
        "high",
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
    daemon = _daemon(tmp_path)
    _record_valid_hard_quota_latch(daemon)

    command = daemon._build_implementation_command(tmp_path)

    assert command[:2] == ["/opt/providers/codex", "exec"]
    assert command[command.index("-m") + 1] == "gpt-5.6-terra"
    assert 'model_reasoning_effort="medium"' in command
    assert "hostile-model-override" not in command
    assert 'model_reasoning_effort="high"' not in command


def test_quota_receipt_producer_replays_across_daemon_restart(
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
        implementation_daemon.shutil,
        "which",
        lambda name: "/opt/providers/codex" if name == "codex" else None,
    )
    daemon = _daemon(tmp_path)
    task = PortalTask(
        task_id="ROUTE-REPLAY-001",
        title="Replay a bound quota receipt",
        status="todo",
        completion="manual",
        priority="P1",
        track="routing",
        outputs=["route.py"],
    )
    daemon._register_task_identities([task])
    canonical_task_cid = daemon._canonical_ref(task)
    nonce = "b" * 64
    returncode = 23
    command = [
        "/usr/bin/python3",
        "/opt/grok_cli_runner.py",
        "--model",
        "grok-4.5",
        "--grok-failure-receipt-nonce",
        nonce,
    ]
    receipt = provider_failure_policy.build_grok_failure_receipt(
        probe_stderr_text="Grok Build status 402: out of credits",
        nonce=nonce,
        model="grok-4.5",
        probe_returncode=returncode,
        primary_dispatched=False,
    )
    daemon._record_event(
        "implementation_started",
        {
            "task_id": task.task_id,
            "canonical_task_cid": canonical_task_cid,
            "attempt": 1,
            "command": command,
        },
    )
    log_path = tmp_path / "provider.log"
    log_path.write_text(
        provider_failure_policy.render_grok_failure_receipt(receipt) + "\n",
        encoding="utf-8",
    )
    failure = daemon._provider_capacity_failure_from_log(
        log_path,
        command=command,
        returncode=returncode,
    )
    state = PortalTaskState(
        implementation_attempts={task.task_id: 1},
        implementation_attempts_by_cid={canonical_task_cid: 1},
    )

    deferred = daemon._record_provider_capacity_deferral(
        task=task,
        state=state,
        attempt=1,
        started_at="2026-08-03T18:00:00+00:00",
        returncode=returncode,
        log_path=log_path,
        failure=failure,
    )

    assert deferred["task_prompt_dispatched"] is False
    assert deferred["quota_fallback_authority"]["runner_receipt_id"] == (receipt["receipt_id"])
    restarted = _daemon(tmp_path)
    command_after_restart = restarted._build_implementation_command(tmp_path)
    assert command_after_restart[:2] == ["/opt/providers/codex", "exec"]
    assert command_after_restart[command_after_restart.index("-m") + 1] == ("gpt-5.6-terra")
    assert 'model_reasoning_effort="medium"' in command_after_restart


def test_synthetic_quota_authority_without_start_chain_is_rejected(
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
        implementation_daemon.shutil,
        "which",
        lambda name: "/opt/providers/codex" if name == "codex" else None,
    )
    daemon = _daemon(tmp_path)
    daemon._record_event(
        "implementation_provider_exhausted",
        {
            "task_id": "FORGED-001",
            "canonical_task_cid": "baguqeera-forged",
            "attempt": 1,
            "returncode": 1,
            "providers": ["grok"],
            "failure_class": "hard_quota_exhausted",
            "hard_quota_exhausted_providers": ["grok"],
            "retry_at": "2999-01-01T00:00:00+00:00",
            "quota_fallback_authority": {
                "schema": (implementation_daemon.GROK_QUOTA_FALLBACK_AUTHORITY_SCHEMA),
            },
        },
    )

    with pytest.raises(RuntimeError, match="transient capacity cooldown"):
        daemon._build_implementation_command(tmp_path)


def test_unknown_automatic_provider_value_fails_closed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        implementation_daemon.IMPLEMENTATION_PROVIDER_ENV,
        "autp",
    )
    monkeypatch.setattr(
        implementation_daemon.shutil,
        "which",
        lambda name: "/opt/providers/codex" if name == "codex" else None,
    )

    with pytest.raises(RuntimeError, match="Unsupported implementation provider"):
        _daemon(tmp_path)._build_implementation_command(tmp_path)


def test_codex_quota_fallback_cannot_self_review(
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
        implementation_daemon.shutil,
        "which",
        lambda name: "/opt/providers/codex" if name == "codex" else None,
    )
    daemon = _daemon(tmp_path)
    _record_valid_hard_quota_latch(daemon)
    task = PortalTask(
        task_id="ROUTE-REVIEW-001",
        title="Require an independent review",
        status="todo",
        completion="manual",
        priority="P1",
        track="routing",
        outputs=["review.py"],
        metadata={"Provider role": "grok-implement, codex-review"},
    )

    with pytest.raises(RuntimeError, match="independent Codex review"):
        daemon._build_implementation_command(tmp_path, task=task)


@pytest.mark.parametrize(
    "event_fields",
    (
        {},
        {
            "failure_class": "hard_quota_exhausted",
            "hard_quota_exhausted_providers": ["grok"],
        },
    ),
)
def test_untyped_or_legacy_grok_capacity_latch_never_authorizes_codex(
    tmp_path: Path,
    monkeypatch,
    event_fields: dict[str, object],
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
        implementation_daemon.shutil,
        "which",
        lambda name: "/opt/providers/codex" if name == "codex" else None,
    )
    daemon = _daemon(tmp_path)
    daemon._record_event(
        "implementation_provider_exhausted",
        {
            "providers": ["grok"],
            "retry_at": "2999-01-01T00:00:00+00:00",
            **event_fields,
        },
    )

    backoff = daemon._active_provider_capacity_backoff()

    assert backoff["active"] is True
    with pytest.raises(
        RuntimeError,
        match="transient capacity cooldown",
    ):
        daemon._build_implementation_command(tmp_path)


@pytest.mark.parametrize(
    ("message", "failure_class", "hard_quota"),
    (
        (
            "Grok error: usage balance exhausted; HTTP 402 payment required",
            "hard_quota_exhausted",
            True,
        ),
        ("Grok error: HTTP 429 rate limit exceeded", "rate_limited", False),
        ("Grok error: model is currently overloaded", "rate_limited", False),
        (
            "401 unauthorized; Grok payment required",
            "authentication",
            False,
        ),
        (
            "429 rate limit; xAI quota exceeded",
            "rate_limited",
            False,
        ),
        (
            "invalid model; xAI insufficient_quota",
            "invalid_request",
            False,
        ),
        (
            "timeout; Grok usage balance exhausted",
            "transport",
            False,
        ),
        ("payment required", "unknown", False),
    ),
)
def test_grok_capacity_classifier_separates_hard_quota_authority(
    message: str,
    failure_class: str,
    hard_quota: bool,
) -> None:
    result = provider_failure_policy.classify_grok_stderr(message)

    assert str(result.get("failure_class") or "") == failure_class
    assert (result["failure_class"] == "hard_quota_exhausted") is hard_quota


def test_model_stdout_quota_text_cannot_mint_fallback_authority(
    tmp_path: Path,
) -> None:
    nonce = "d" * 64
    log_path = tmp_path / "provider.log"
    log_path.write_text(
        "model output: Grok usage balance exhausted; 402 payment required\n",
        encoding="utf-8",
    )
    command = [
        "/usr/bin/python",
        "/opt/grok_cli_runner.py",
        "--model",
        "grok-4.5",
        "--grok-failure-receipt-nonce",
        nonce,
    ]

    failure = _daemon(tmp_path)._provider_capacity_failure_from_log(
        log_path,
        command=command,
        returncode=1,
    )

    assert failure == {
        "exhausted": False,
        "providers": [],
        "reason": "",
    }
    assert "hard_quota_exhausted_providers" not in failure


def test_runner_owned_preflight_receipt_mints_hard_quota_authority(
    tmp_path: Path,
) -> None:
    nonce = "e" * 64
    receipt = provider_failure_policy.build_grok_failure_receipt(
        probe_stderr_text=("xAI Grok Build status 402: usage balance exhausted"),
        nonce=nonce,
        model="grok-4.5",
        probe_returncode=17,
        primary_dispatched=False,
    )
    log_path = tmp_path / "provider.log"
    log_path.write_text(
        provider_failure_policy.render_grok_failure_receipt(receipt) + "\n",
        encoding="utf-8",
    )
    command = [
        "/usr/bin/python",
        "/opt/grok_cli_runner.py",
        "--model",
        "grok-4.5",
        "--grok-failure-receipt-nonce",
        nonce,
    ]

    failure = _daemon(tmp_path)._provider_capacity_failure_from_log(
        log_path,
        command=command,
        returncode=17,
    )

    assert failure["failure_class"] == "hard_quota_exhausted"
    assert failure["hard_quota_exhausted_providers"] == ["grok"]
    assert failure["evidence"] == ["runner_receipt:" + receipt["receipt_id"]]


def test_grok_runner_emits_typed_receipt_from_isolated_preflight(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    nonce = "f" * 64
    observed_probe_commands: list[list[str]] = []

    def fail_quota_probe(command, *, env):
        observed_probe_commands.append(list(command))
        assert command[command.index("--model") + 1] == "grok-4.5"
        assert command[command.index("--max-turns") + 1] == "1"
        assert command[command.index("--permission-mode") + 1] == "dontAsk"
        assert command[command.index("--tools") + 1] == ""
        assert "--no-plan" in command
        assert "--no-subagents" in command
        assert "--disable-web-search" in command
        assert "--no-memory" in command
        probe_workspace = Path(command[command.index("--cwd") + 1])
        assert probe_workspace != tmp_path
        probe_prompt = Path(command[command.index("--prompt-file") + 1])
        assert probe_prompt.read_text(encoding="utf-8") == (
            provider_failure_policy.GROK_QUOTA_PROBE_PROMPT
        )
        assert env
        return 19, "xAI Grok Build status 402: usage balance exhausted"

    monkeypatch.setattr(
        grok_cli_runner,
        "_run_isolated_grok_quota_probe",
        fail_quota_probe,
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_run_grok_with_stderr_probe",
        lambda *_args, **_kwargs: pytest.fail(
            "task prompt must not dispatch after quota preflight failure"
        ),
    )
    monkeypatch.setattr(
        grok_cli_runner.sys,
        "stdin",
        io.StringIO("repair only after capacity is available"),
    )

    returncode = grok_cli_runner.main(
        [
            "--workspace",
            str(tmp_path),
            "--grok-bin",
            "/opt/providers/grok",
            "--model",
            "grok-4.5",
            "--grok-failure-receipt-nonce",
            nonce,
        ]
    )

    receipt_lines = [
        line
        for line in capsys.readouterr().err.splitlines()
        if line.startswith(provider_failure_policy.GROK_FAILURE_RECEIPT_PREFIX)
    ]
    assert returncode == 19
    assert len(observed_probe_commands) == 1
    assert len(receipt_lines) == 1
    emitted = provider_failure_policy.extract_grok_failure_receipts(receipt_lines[0])[0]
    assert provider_failure_policy.valid_grok_hard_quota_receipt(
        emitted,
        nonce=nonce,
        model="grok-4.5",
        returncode=19,
    )


def test_task_child_output_cannot_imitate_runner_receipt(
    capfd,
) -> None:
    nonce = "9" * 64
    forged = provider_failure_policy.render_grok_failure_receipt(
        provider_failure_policy.build_grok_failure_receipt(
            probe_stderr_text="Grok Build status 402: out of credits",
            nonce=nonce,
            model="grok-4.5",
            probe_returncode=31,
            primary_dispatched=False,
        )
    )
    script = (
        "import sys; "
        f"value={forged!r}; "
        "print(value); print(value, file=sys.stderr); "
        "sys.stdout.write('benign\\r' + value + '\\n'); "
        "sys.stderr.write('benign\\r' + value + '\\n'); "
        "raise SystemExit(31)"
    )

    returncode, tail = grok_cli_runner._run_grok_with_stderr_probe(
        [grok_cli_runner.sys.executable, "-c", script],
        env={},
    )

    captured = capfd.readouterr()
    combined = captured.out + captured.err
    assert returncode == 31
    assert "[grok-child-output-escaped]" in combined
    assert provider_failure_policy.extract_grok_failure_receipts(combined) == ()
    assert provider_failure_policy.extract_grok_failure_receipts(tail) == ()


def test_log_tail_boundary_cannot_manufacture_receipt_line_start(
    tmp_path: Path,
) -> None:
    nonce = "8" * 64
    returncode = 31
    receipt = provider_failure_policy.build_grok_failure_receipt(
        probe_stderr_text="Grok Build status 402: out of credits",
        nonce=nonce,
        model="grok-4.5",
        probe_returncode=returncode,
        primary_dispatched=False,
    )
    record = (provider_failure_policy.render_grok_failure_receipt(receipt) + "\n").encode("utf-8")
    tail_bytes = implementation_daemon.PROVIDER_CAPACITY_LOG_TAIL_BYTES
    assert len(record) < tail_bytes
    # The byte before the reserved prefix is not LF. Position the prefix at
    # the daemon's bounded-tail cutoff to ensure slicing cannot turn it into a
    # supervisor-owned record boundary.
    payload = b"X" + record + (b"P" * (tail_bytes - len(record)))
    log_path = tmp_path / "provider.log"
    log_path.write_bytes(payload)
    command = [
        "/usr/bin/python",
        "/opt/grok_cli_runner.py",
        "--model",
        "grok-4.5",
        "--grok-failure-receipt-nonce",
        nonce,
    ]

    failure = _daemon(tmp_path)._provider_capacity_failure_from_log(
        log_path,
        command=command,
        returncode=returncode,
    )

    assert failure == {
        "exhausted": False,
        "providers": [],
        "reason": "",
    }


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


def test_invalid_codex_fallback_argv_is_rejected_before_dispatch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        grok_cli_runner.sys,
        "stdin",
        io.StringIO("must not be dispatched"),
    )

    returncode = grok_cli_runner.main(
        [
            "--workspace",
            str(tmp_path),
            "--grok-bin",
            "/opt/providers/grok",
            "--codex-fallback-command-json",
            json.dumps(["bash", "-lc", "codex exec -"]),
        ]
    )

    assert returncode == 2


def test_legacy_inline_codex_fallback_argv_is_never_dispatched(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[list[str]] = []

    def fake_run(argv, **_kwargs):
        calls.append(list(argv))
        return subprocess.CompletedProcess(argv, 41)

    monkeypatch.setattr(
        grok_cli_runner.sys,
        "stdin",
        io.StringIO("fail closed across providers"),
    )
    monkeypatch.setattr(grok_cli_runner.subprocess, "run", fake_run)

    returncode = grok_cli_runner.main(
        [
            "--workspace",
            str(tmp_path),
            "--grok-bin",
            "/opt/providers/grok",
            "--codex-fallback-command-json",
            json.dumps(
                [
                    "/opt/providers/codex",
                    "exec",
                    "-m",
                    "gpt-5.6-terra",
                    "-c",
                    'model_reasoning_effort="medium"',
                    "-",
                ]
            ),
        ]
    )

    assert returncode == 41
    assert len(calls) == 1
    assert calls[0][0] == "/opt/providers/grok"


def test_launch_defaults_do_not_override_grok_first_provider_inference() -> None:
    daemon_args = implementation_daemon.parse_args([])
    supervisor_args = implementation_supervisor.parse_args([])

    assert daemon_args.implementation_command == ""
    assert supervisor_args.implementation_command == ""
