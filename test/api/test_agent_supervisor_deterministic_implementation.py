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
from dataclasses import replace
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    implementation_daemon as implementation_daemon_module,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    DETERMINISTIC_DECLARED_OUTPUT_TARGET_REQUIRED,
    PortalTask,
    TodoImplementationDaemon,
    TodoTaskState,
    inherit_retry_budget_repair_validation,
    task_is_declared_read_only_validation,
)
from ipfs_accelerate_py.agent_supervisor.validation.validation_runtime import (
    ValidationRuntimeError,
    build_validation_environment,
    sealed_validation_launcher_argv0_environment,
    sealed_validation_python_runner,
    validation_command_launcher_argv0,
    validation_environment_for_runner,
    validation_python_executable,
    validation_python_launcher_environment,
    validation_shell_command,
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
        worktree_root=repo,
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
    # These tests exercise the deterministic execution boundary directly with
    # an in-memory task. Board-contract enforcement is covered separately, so
    # resolve that task as the current contract instead of requiring a second
    # Markdown representation that can drift from each test case.
    monkeypatch.setattr(
        daemon,
        "_completion_tasks_for_declared_output_gate",
        lambda _metadata, primary_task: ([primary_task], {}),
    )
    monkeypatch.setattr(
        daemon,
        "_declared_output_tracking_invariant",
        lambda *_args, **_kwargs: {"passed": True, "test_double": True},
    )
    monkeypatch.setattr(
        daemon,
        "_todo_completion_is_durable",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        daemon,
        "apply_post_merge_authoritative_acceptance",
        lambda *_args, **_kwargs: {
            "authoritatively_completed": True,
            "todo_update_result": {"updated": True, "test_double": True},
        },
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
    # Out-of-scope untracked files are removed before admission. The empty
    # tree is not a no-change completion, so the task still fails closed.
    assert result["validation_result"]["reason"] == (
        "deterministic_no_change_authority_unavailable"
    )
    assert not (repo / "undeclared.py").exists()
    assert "undeclared.py" not in _git_stdout(
        repo, "status", "--porcelain", "--untracked-files=all"
    )
    receipt = json.loads(
        Path(result["task_execution_receipt_path"]).read_text(encoding="utf-8")
    )
    assert receipt["isolation_audit"]["model_call_count"] == 0
    assert receipt["isolation_audit"]["provider_call_count"] == 0


def _git_stdout(repo: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
    )
    return completed.stdout.strip()


def _read_only_task(validation: str) -> PortalTask:
    return PortalTask(
        task_id="SCA-DET-RO",
        title="Run a declared check without an edit target",
        status="ready",
        completion="manual",
        priority="P0",
        track="static-analysis",
        outputs=[],
        validation=[validation],
        acceptance="The declared check passes without staging the checkout.",
        metadata={"Provider role": "deterministic-only"},
    )


def test_validation_only_task_completes_without_staging_a_clean_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, daemon = _daemon(tmp_path, monkeypatch)
    completed: list[str] = []
    monkeypatch.setattr(
        daemon,
        "_mark_task_or_bundle_completed_in_todo",
        lambda task, **_kwargs: completed.append(task.task_id)
        or {"updated": True, "test_double": True},
    )
    head = _git_stdout(repo, "rev-parse", "HEAD")

    result = daemon._run_implementation(
        _read_only_task(f"{shlex.quote(sys.executable)} -c 'assert True'"),
        TodoTaskState(),
    )

    assert result["returncode"] == 0
    assert result["validation_result"]["passed"] is True
    assert result["validation_result"]["read_only_declared_validation"] is True
    assert result["commit_result"]["reason"] == "read_only_declared_validation"
    assert result["commit_result"]["committed"] is False
    assert _git_stdout(repo, "rev-parse", "HEAD") == head
    assert _git_stdout(repo, "status", "--porcelain") == ""
    assert completed == ["SCA-DET-RO"]


def test_validation_only_task_does_not_stage_preexisting_or_command_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, daemon = _daemon(tmp_path, monkeypatch)
    (repo / "preexisting.txt").write_text("leave me\n", encoding="utf-8")
    command = _add_validation_script(
        repo,
        "read_only_check.py",
        "from pathlib import Path\n"
        "Path('check-ok.txt').write_text('ran\\n')\n",
    )
    head = _git_stdout(repo, "rev-parse", "HEAD")

    result = daemon._run_implementation(
        _read_only_task(command),
        TodoTaskState(),
    )

    assert result["returncode"] == 0
    assert result["commit_result"]["committed"] is False
    assert _git_stdout(repo, "rev-parse", "HEAD") == head
    status = _git_stdout(repo, "status", "--porcelain", "--untracked-files=all")
    assert "?? preexisting.txt" in status.splitlines()
    assert "?? check-ok.txt" in status.splitlines()
    assert all(not line.startswith(("A ", "M ")) for line in status.splitlines())


def test_validation_only_failure_does_not_complete_or_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, daemon = _daemon(tmp_path, monkeypatch)
    (repo / "preexisting.txt").write_text("leave me\n", encoding="utf-8")
    head = _git_stdout(repo, "rev-parse", "HEAD")
    completed: list[str] = []
    monkeypatch.setattr(
        daemon,
        "_mark_task_or_bundle_completed_in_todo",
        lambda task, **_kwargs: completed.append(task.task_id)
        or {"updated": True},
    )

    result = daemon._run_implementation(
        _read_only_task(
            f"{shlex.quote(sys.executable)} -c 'raise SystemExit(7)'"
        ),
        TodoTaskState(),
    )

    assert result["returncode"] != 0
    assert result["validation_result"]["passed"] is False
    assert completed == []
    assert _git_stdout(repo, "rev-parse", "HEAD") == head
    assert "?? preexisting.txt" in _git_stdout(
        repo, "status", "--porcelain"
    ).splitlines()


def test_predicted_edit_target_is_not_treated_as_read_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, daemon = _daemon(tmp_path, monkeypatch)
    task = _read_only_task(f"{shlex.quote(sys.executable)} -c 'assert True'")
    task.metadata["Predicted files"] = "artifact.txt"
    assert task_is_declared_read_only_validation(task, repo_root=repo) is False

    result = daemon._run_implementation(task, TodoTaskState())

    assert (
        result["validation_result"].get("read_only_declared_validation")
        is not True
    )
    assert result.get("commit_result", {}).get("reason") != (
        "read_only_declared_validation"
    )
    assert result["returncode"] != 0


def test_empty_completion_gap_still_requires_an_edit_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, daemon = _daemon(tmp_path, monkeypatch)
    task = _read_only_task(f"{shlex.quote(sys.executable)} -c 'assert True'")
    task.metadata["Merge role"] = "completion_gate_gap"

    with pytest.raises(ValueError, match="declared output/edit target"):
        daemon._deterministic_validation_plan_binding(repo, task)

    result = daemon._run_implementation(task, TodoTaskState())
    assert result["reason"] == "completion_gap_missing_precise_edit_targets"


def test_repaired_read_only_validation_clears_only_its_cooldown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _repo, daemon = _daemon(tmp_path, monkeypatch)
    check = _read_only_task(f"{shlex.quote(sys.executable)} -c 'assert True'")
    edit = _task(f"{shlex.quote(sys.executable)} -c 'assert True'")
    note = "ValueError: " + DETERMINISTIC_DECLARED_OUTPUT_TARGET_REQUIRED
    daemon.task_queue.defer(check.task_id, 600, reason=note)
    daemon.task_queue.defer(edit.task_id, 600, reason=note)
    daemon.task_queue.defer("SCA-OTHER", 600, reason="validation_failed")

    released = daemon._release_repaired_read_only_validation_cooldowns(
        {check.task_id: check, edit.task_id: edit}
    )

    assert released == [check.task_id]
    assert daemon.task_queue.is_cooled_down(check.task_id) is False
    assert daemon.task_queue.is_cooled_down(edit.task_id) is True
    assert daemon.task_queue.is_cooled_down("SCA-OTHER") is True


def test_absolute_approved_python_uses_the_sealed_launcher() -> None:
    argv = validation_shell_command(
        f"{shlex.quote(sys.executable)} -c 'assert True'"
    )
    script = argv[-1]
    assert "python -c 'assert True'" in script
    assert f"_IPFS_ACCELERATE_VALIDATION_ARGV0={sys.executable}" in script

    with pytest.raises(ValidationRuntimeError, match="sealed python"):
        validation_shell_command("/no/such/python -c 'assert True'")


def test_sealed_launcher_preserves_venv_argv0(tmp_path: Path) -> None:
    venv = tmp_path / "venv"
    bin_dir = venv / "bin"
    bin_dir.mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text(
        "home = /usr/bin\n"
        "include-system-site-packages = false\n"
        "version = 3.12.3\n",
        encoding="utf-8",
    )
    approved = Path(validation_python_executable()).resolve(strict=True)
    launcher = bin_dir / "python"
    launcher.symlink_to(approved)
    command = (
        f"{shlex.quote(str(launcher))} -c 'import sys; print(sys.prefix)'"
    )
    assert validation_command_launcher_argv0(command) == str(launcher)

    @sealed_validation_python_runner
    def runner() -> None:
        return None

    environment = sealed_validation_launcher_argv0_environment(
        validation_environment_for_runner(
            build_validation_environment(),
            runner,
        ),
        str(launcher),
    )
    with validation_python_launcher_environment(
        environment,
        argv0=str(launcher),
    ) as (child, receipt):
        completed = subprocess.run(
            validation_shell_command(command),
            cwd=tmp_path,
            text=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=60,
            check=False,
            env=child,
            pass_fds=tuple(receipt.inherited_fds),
        )
    assert completed.returncode == 0, completed.stdout
    prefix = Path(completed.stdout.strip().splitlines()[-1])
    assert prefix.resolve() == venv.resolve()


def test_unverified_argv0_is_rejected() -> None:
    @sealed_validation_python_runner
    def runner() -> None:
        return None

    environment = validation_environment_for_runner(
        build_validation_environment(),
        runner,
    )
    with pytest.raises(ValidationRuntimeError, match="virtualenv"):
        sealed_validation_launcher_argv0_environment(
            environment,
            "/usr/bin/python3.12",
        )


def test_discovery_repair_of_a_read_only_check_inherits_the_local_role() -> None:
    source = replace(
        _read_only_task(f"{shlex.quote(sys.executable)} -c 'assert True'"),
        task_id="SCA-010",
    )
    repair = PortalTask(
        task_id="SCA-062",
        title="Resolve implementation retry-budget failure for SCA-010",
        status="ready",
        completion="manual",
        priority="P1",
        track="ops",
        outputs=["/tmp/discovery-report"],
        validation=["test -f /tmp/discovery-report/finding.md"],
        acceptance=(
            "release SCA-010 from the strategy blocked_tasks"
        ),
        metadata={},
    )

    inherited = inherit_retry_budget_repair_validation([source, repair])
    repaired = next(task for task in inherited if task.task_id == "SCA-062")

    assert repaired.validation == source.validation
    assert repaired.metadata["Provider role"] == "deterministic-only"
    assert task_is_declared_read_only_validation(repaired) is True


def test_source_command_repair_of_a_read_only_check_inherits_the_local_role() -> None:
    validation = f"{shlex.quote(sys.executable)} -c 'assert True'"
    source = replace(_read_only_task(validation), task_id="SCA-030")
    repair = PortalTask(
        task_id="SCA-062",
        title="Resolve validation retry-budget failure for SCA-030",
        status="ready",
        completion="manual",
        priority="P1",
        track="ops",
        outputs=[],
        validation=[validation],
        acceptance="release SCA-030 from strategy blocked_tasks",
        metadata={
            "Generated by": (
                "ipfs_accelerate_py.agent_supervisor.retry-budget-repair@1"
            ),
            "Retry repair source": "SCA-030",
            "Retry failure kind": "validation",
        },
    )
    edit_source = replace(
        _task(validation),
        task_id="SCA-031",
        title="Edit a declared artifact",
    )
    edit_repair = PortalTask(
        task_id="SCA-063",
        title="Resolve validation retry-budget failure for SCA-031",
        status="ready",
        completion="manual",
        priority="P1",
        track="ops",
        outputs=["artifact.txt"],
        validation=[validation],
        acceptance="release SCA-031 from strategy blocked_tasks",
        metadata={
            "Generated by": (
                "ipfs_accelerate_py.agent_supervisor.retry-budget-repair@1"
            ),
            "Retry repair source": "SCA-031",
            "Retry failure kind": "validation",
        },
    )
    scoped_repair = replace(repair, task_id="SCA-065", outputs=["artifact.txt"])
    role_kept = replace(
        repair,
        task_id="SCA-064",
        metadata={**repair.metadata, "Provider role": "grok-implement"},
    )

    inherited = inherit_retry_budget_repair_validation(
        [source, repair, edit_source, edit_repair, scoped_repair, role_kept]
    )
    by_id = {task.task_id: task for task in inherited}
    repaired = by_id["SCA-062"]

    assert repaired.validation == [validation]
    assert repaired.metadata["Provider role"] == "deterministic-only"
    assert "Provider role" not in repair.metadata
    assert task_is_declared_read_only_validation(repaired) is True
    assert "Provider role" not in by_id["SCA-063"].metadata
    assert by_id["SCA-063"].validation == [validation]
    assert "Provider role" not in by_id["SCA-065"].metadata
    assert by_id["SCA-064"].metadata["Provider role"] == "grok-implement"
    assert by_id["SCA-064"].validation == [validation]


def test_obsolete_provider_repair_cooldown_clears_only_for_that_log(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _repo, daemon = _daemon(tmp_path, monkeypatch)
    repair = PortalTask(
        task_id="SCA-062",
        title="Resolve implementation retry-budget failure for SCA-010",
        status="ready",
        completion="manual",
        priority="P1",
        track="ops",
        outputs=[],
        validation=[f"{shlex.quote(sys.executable)} -c 'assert True'"],
        acceptance="release SCA-010 from strategy blocked_tasks",
        metadata={"Provider role": "deterministic-only"},
    )
    kept = PortalTask(
        task_id="SCA-063",
        title="Resolve implementation retry-budget failure for SCA-011",
        status="ready",
        completion="manual",
        priority="P1",
        track="ops",
        outputs=[],
        validation=[f"{shlex.quote(sys.executable)} -c 'assert True'"],
        acceptance="release SCA-011 from strategy blocked_tasks",
        metadata={"Provider role": "deterministic-only"},
    )
    note = "validation_or_implementation_failed"
    daemon.task_queue.defer(repair.task_id, 600, reason=note)
    daemon.task_queue.defer(kept.task_id, 600, reason=note)
    log_dir = daemon.implementation_log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "sca-062-attempt-1-aaa.log").write_text(
        "Command: /usr/bin/python -m ipfs_accelerate_py.agent_supervisor.grok_cli_runner\n"
        "Grok provider isolation unavailable\n",
        encoding="utf-8",
    )
    (log_dir / "sca-063-attempt-1-bbb.log").write_text(
        "deterministic declared validation failed\n",
        encoding="utf-8",
    )

    released = daemon._release_obsolete_provider_repair_cooldowns(
        {repair.task_id: repair, kept.task_id: kept}
    )

    assert released == [repair.task_id]
    assert daemon.task_queue.is_cooled_down(repair.task_id) is False
    assert daemon.task_queue.is_cooled_down(kept.task_id) is True


def test_obsolete_venv_launcher_cooldown_clears_only_for_that_log(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _repo, daemon = _daemon(tmp_path, monkeypatch)
    missing = replace(
        _read_only_task(f"{shlex.quote(sys.executable)} -c 'assert True'"),
        task_id="SCA-030",
    )
    kept = replace(
        _read_only_task("/usr/bin/python -c 'assert True'"),
        task_id="SCA-031",
    )
    note = "validation_or_implementation_failed"
    daemon.task_queue.defer(missing.task_id, 600, reason=note)
    daemon.task_queue.defer(kept.task_id, 600, reason=note)
    log_dir = daemon.implementation_log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "sca-030-attempt-1-aaa.log").write_text(
        "/usr/bin/python3.12: No module named pytest\n",
        encoding="utf-8",
    )
    (log_dir / "sca-031-attempt-1-bbb.log").write_text(
        "/usr/bin/python3.12: No module named pytest\n",
        encoding="utf-8",
    )

    released = daemon._release_obsolete_venv_launcher_cooldowns(
        {missing.task_id: missing, kept.task_id: kept}
    )

    assert released == [missing.task_id]
    assert daemon.task_queue.is_cooled_down(missing.task_id) is False
    assert daemon.task_queue.is_cooled_down(kept.task_id) is True
