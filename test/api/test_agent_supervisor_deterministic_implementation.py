"""Deterministic-only implementation-daemon integration tests.

These tests prove that a reviewed task validation plan crosses the typed
``TaskExecutionPolicy`` boundary without entering a prompt, implementation
command, model provider, or provider-capacity route.
"""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    implementation_daemon as implementation_daemon_module,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    PortalTask,
    TodoImplementationDaemon,
    TodoTaskState,
)


def _git(repo: Path, *arguments: str) -> None:
    subprocess.run(
        ["git", *arguments],
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
    )


def _daemon(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    protected_paths: tuple[str, ...] = (),
) -> tuple[Path, TodoImplementationDaemon]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.name", "Deterministic Test")
    _git(repo, "config", "user.email", "deterministic@example.invalid")
    todo_path = repo / "tasks.todo.md"
    todo_path.write_text("# Deterministic tasks\n", encoding="utf-8")
    (repo / ".gitignore").write_text("state/\n", encoding="utf-8")
    for relative in protected_paths:
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("protected baseline\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "baseline")

    state_dir = repo / "state"
    daemon = TodoImplementationDaemon(
        todo_path=todo_path,
        state_path=state_dir / "task_state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "events.jsonl",
        repo_root=repo,
        task_header_prefix="## SCA-",
        implement=True,
        implementation_command="model-command-must-not-run",
        implementation_protected_paths=protected_paths,
    )
    monkeypatch.setattr(
        daemon,
        "_build_implementation_prompt",
        lambda *_args, **_kwargs: pytest.fail(
            "deterministic task entered prompt construction"
        ),
    )
    monkeypatch.setattr(
        daemon,
        "_build_implementation_command",
        lambda *_args, **_kwargs: pytest.fail(
            "deterministic task entered implementation command construction"
        ),
    )
    monkeypatch.setattr(
        daemon,
        "_active_provider_capacity_backoff",
        lambda: pytest.fail("deterministic task consulted provider capacity"),
    )
    monkeypatch.setattr(
        implementation_daemon_module,
        "run_process_group_stream",
        lambda *_args, **_kwargs: pytest.fail(
            "deterministic task dispatched an implementation provider"
        ),
    )
    monkeypatch.setattr(
        daemon,
        "_decision_runtime_completion",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        daemon,
        "_mark_task_or_bundle_completed_in_todo",
        lambda _task, **_kwargs: {"updated": True, "test_double": True},
    )
    monkeypatch.setattr(
        daemon,
        "_record_task_queue_outcome",
        lambda *_args, **_kwargs: None,
    )
    return repo, daemon


def _add_validation_script(repo: Path, name: str, source: str) -> str:
    (repo / name).write_text(source, encoding="utf-8")
    _git(repo, "add", name)
    _git(repo, "commit", "-m", f"add {name}")
    return f"{shlex.quote(sys.executable)} {shlex.quote(name)}"


def _task(
    validation: str,
    *,
    context_tokens: int = 4096,
    provider_role: str = "deterministic-only",
) -> PortalTask:
    return PortalTask(
        task_id="SCA-DET-001",
        title="Run a bounded symbolic validation plan",
        status="ready",
        completion="manual",
        priority="P0",
        track="static-analysis",
        outputs=["artifact.txt"],
        validation=[validation],
        acceptance="The declared deterministic validation succeeds.",
        metadata={
            "Provider role": provider_role,
            "Context budget tokens": str(context_tokens),
        },
    )


def _events(daemon: TodoImplementationDaemon) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in daemon.events_path.read_text(encoding="utf-8").splitlines()
    ]


def test_deterministic_task_executes_declared_plan_with_zero_model_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, daemon = _daemon(tmp_path, monkeypatch)
    task = _task(
        _add_validation_script(
            repo,
            "generate_artifact.py",
            "from pathlib import Path\n"
            "Path('artifact.txt').write_text('ok\\n')\n",
        )
    )

    result = daemon._run_implementation(task, TodoTaskState())

    assert result["returncode"] == 0
    assert result["validation_result"]["passed"] is True
    assert (repo / "artifact.txt").read_text(encoding="utf-8") == "ok\n"
    receipt = json.loads(
        Path(result["task_execution_receipt_path"]).read_text(encoding="utf-8")
    )
    assert receipt["receipt_id"] == result["task_execution_receipt_id"]
    assert receipt["status"] == "succeeded"
    assert receipt["isolation_audit"] == {
        "llm_call_count": 0,
        "model_call_count": 0,
        "provider_call_count": 0,
    }
    assert receipt["attempts"][0]["executable_id"] == (
        "declared-validation-plan"
    )
    assert receipt["daemon_integration"]["raw_command_arguments_accepted"] is False
    events = _events(daemon)
    assert any(
        event["type"] == "deterministic_task_execution_authorized"
        for event in events
    )
    assert any(
        event["type"] == "deterministic_task_execution_finished"
        and event["status"] == "succeeded"
        for event in events
    )
    assert not any(
        event.get("operation") == "implementation_provider"
        for event in events
    )


def test_operator_only_zero_token_task_validates_prepared_artifact_without_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, daemon = _daemon(tmp_path, monkeypatch)
    task = _task(
        _add_validation_script(
            repo,
            "prepare_operator_receipt.py",
            "from pathlib import Path\n"
            "Path('artifact.txt').write_text('operator-reviewed\\n')\n",
        ),
        context_tokens=0,
        provider_role="operator-only",
    )

    assert daemon._task_context_token_limit(task) == 0
    assert daemon._task_uses_typed_local_execution(task) is True

    result = daemon._run_implementation(task, TodoTaskState())

    assert result["returncode"] == 0
    assert result["validation_result"]["passed"] is True
    assert (repo / "artifact.txt").read_text(encoding="utf-8") == (
        "operator-reviewed\n"
    )
    receipt = json.loads(
        Path(result["task_execution_receipt_path"]).read_text(encoding="utf-8")
    )
    assert receipt["isolation_audit"] == {
        "llm_call_count": 0,
        "model_call_count": 0,
        "provider_call_count": 0,
    }


def test_operator_only_snapshots_exact_prepared_output_into_worktree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, daemon = _daemon(tmp_path, monkeypatch)
    relative = "data/operator/recovery-receipt.json"
    prepared = repo / relative
    prepared.parent.mkdir(parents=True)
    prepared.write_text('{"decision":"clear"}\n', encoding="utf-8")
    worktree = tmp_path / "operator-workspace"
    worktree.mkdir(parents=True)
    task = PortalTask(
        task_id="SCA-OP-001",
        title="Validate an operator-prepared recovery receipt",
        status="ready",
        completion="manual",
        priority="P0",
        track="operator-recovery",
        outputs=[relative],
        validation=["python -m json.tool " + relative],
        acceptance="The reviewed receipt is stable and valid.",
        metadata={
            "Provider role": "operator-only",
            "Context budget tokens": "0",
        },
    )

    snapshots = daemon._seed_operator_prepared_outputs(worktree, task)

    assert snapshots == (
        {
            "path": relative,
            "sha256": (
                "8b41437999724e9b670a14f470f1022a2711f2915dd975f3ac8f3ab66bda00f9"
            ),
            "size": 21,
            "mode": prepared.stat().st_mode & 0o777,
        },
    )
    assert (worktree / relative).read_bytes() == prepared.read_bytes()
    prepared.write_text('{"decision":"retain"}\n', encoding="utf-8")
    assert (worktree / relative).read_text(encoding="utf-8") == (
        '{"decision":"clear"}\n'
    )
    assert any(
        event["type"] == "operator_prepared_outputs_seeded"
        and event["provider_call_allowed"] is False
        for event in _events(daemon)
    )


def test_operator_only_normalizes_stdout_suppression_before_proposal_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, daemon = _daemon(tmp_path, monkeypatch)
    relative = "state/operator-receipt.json"
    prepared = repo / relative
    prepared.parent.mkdir(parents=True, exist_ok=True)
    prepared.write_text('{"decision":"clear"}\n', encoding="utf-8")
    task = PortalTask(
        task_id="SCA-OP-002",
        title="Validate an operator-prepared JSON receipt",
        status="ready",
        completion="manual",
        priority="P0",
        track="operator-recovery",
        outputs=[relative],
        validation=[
            f"{shlex.quote(sys.executable)} -m json.tool "
            f"{shlex.quote(relative)} >/dev/null"
        ],
        acceptance="The reviewed receipt is stable and valid.",
        metadata={
            "Provider role": "operator-only",
            "Context budget tokens": "0",
        },
    )

    normalized, notes = daemon._normalize_validation_command(
        task.validation[0]
    )

    assert normalized.endswith(relative)
    assert notes == [
        "removed trailing stdout suppression from validation command"
    ]

    result = daemon._run_implementation(task, TodoTaskState())

    assert result["returncode"] == 0
    assert result["validation_result"]["passed"] is True
    assert result["validation_result"]["proposal_gate"]["accepted"] is True
    receipt = json.loads(
        Path(result["task_execution_receipt_path"]).read_text(encoding="utf-8")
    )
    assert receipt["isolation_audit"] == {
        "llm_call_count": 0,
        "model_call_count": 0,
        "provider_call_count": 0,
    }


def test_deterministic_task_reports_declared_validation_failure_without_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, daemon = _daemon(tmp_path, monkeypatch)
    task = _task(
        _add_validation_script(
            repo,
            "fail_validation.py",
            "raise SystemExit(7)\n",
        )
    )

    result = daemon._run_implementation(task, TodoTaskState())

    assert result["returncode"] != 0
    assert result["validation_result"]["attempted"] is True
    assert result["validation_result"]["passed"] is False
    receipt = json.loads(
        Path(result["task_execution_receipt_path"]).read_text(encoding="utf-8")
    )
    assert receipt["status"] == "succeeded"
    assert receipt["isolation_audit"]["model_call_count"] == 0
    assert receipt["isolation_audit"]["provider_call_count"] == 0
    assert any(
        event["type"] == "deterministic_task_execution_finished"
        and event["status"] == "failed"
        for event in _events(daemon)
    )


def test_deterministic_context_budget_rejects_before_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, daemon = _daemon(tmp_path, monkeypatch)
    task = _task("true", context_tokens=1)
    monkeypatch.setattr(
        daemon,
        "_run_validation_commands",
        lambda *_args, **_kwargs: pytest.fail(
            "over-budget deterministic context reached validation"
        ),
    )

    validation, receipt_path, receipt = (
        daemon._execute_deterministic_validation_plan(
            workspace_path=repo,
            task=task,
            attempt=1,
            log_path=repo / "state" / "budget.log",
            state=TodoTaskState(),
        )
    )

    assert validation["attempted"] is False
    assert validation["passed"] is False
    assert validation["reason"] == (
        "deterministic_execution_task_context_limit_exceeded"
    )
    assert receipt["status"] == "rejected"
    assert receipt["reason_code"] == "task_context_limit_exceeded"
    assert receipt["isolation_audit"]["model_call_count"] == 0
    assert Path(receipt_path).is_file()
    assert any(
        event["type"] == "deterministic_task_execution_rejected"
        for event in _events(daemon)
    )


def test_deterministic_validation_cannot_mutate_operator_protected_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, daemon = _daemon(
        tmp_path,
        monkeypatch,
        protected_paths=("policy/approval.json",),
    )
    task = _task(
        _add_validation_script(
            repo,
            "mutate_policy.py",
            "from pathlib import Path\n"
            "Path('policy/approval.json').write_text('mutated\\n')\n",
        )
    )

    result = daemon._run_implementation(task, TodoTaskState())

    assert result["returncode"] == 1
    assert result["reason"] == "implementation_protected_path_mutated"
    assert result["validation_result"]["passed"] is False
    assert result["protected_path_violation"]["protected_paths"] == [
        "policy/approval.json"
    ]
    receipt = json.loads(
        Path(result["task_execution_receipt_path"]).read_text(encoding="utf-8")
    )
    assert receipt["isolation_audit"]["model_call_count"] == 0
    assert any(
        event["type"] == "implementation_protected_path_mutated"
        for event in _events(daemon)
    )


def test_deterministic_materialization_rejects_undeclared_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, daemon = _daemon(tmp_path, monkeypatch)
    task = _task(
        _add_validation_script(
            repo,
            "write_undeclared.py",
            "from pathlib import Path\n"
            "Path('undeclared.py').write_text('outside scope\\n')\n",
        )
    )

    result = daemon._run_implementation(task, TodoTaskState())

    assert result["returncode"] != 0
    assert result["validation_result"]["passed"] is False
    assert result["validation_result"]["reason"] == (
        "deterministic_materialization_proposal_rejected"
    )
    assert result["validation_result"]["proposal_gate"]["accepted"] is False
    assert "undeclared.py" in result["validation_result"]["proposal_gate"][
        "changed_paths"
    ]
    receipt = json.loads(
        Path(result["task_execution_receipt_path"]).read_text(encoding="utf-8")
    )
    assert receipt["isolation_audit"]["model_call_count"] == 0
    assert receipt["isolation_audit"]["provider_call_count"] == 0
