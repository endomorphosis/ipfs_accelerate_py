from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    PortalTask,
    PortalTaskState,
    TodoImplementationDaemon,
    task_validation_python_modules,
)
from ipfs_accelerate_py.agent_supervisor.validation.validation_runtime import (
    VALIDATION_PYTHON_MODULE_PREFLIGHT_SCHEMA,
    VALIDATION_PYTHON_MODULES_ENV,
    ValidationRuntimeError,
    preflight_validation_python_modules,
    required_validation_python_modules,
)


def _git(repo: Path, *args: str) -> None:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout


def test_required_validation_python_modules_combines_all_scopes() -> None:
    modules = required_validation_python_modules(
        ("json", "requests"),
        environment={
            VALIDATION_PYTHON_MODULES_ENV: "packaging, pytest",
        },
    )

    assert modules == ("pytest", "packaging", "json", "requests")


def test_required_validation_python_modules_rejects_code_like_metadata() -> None:
    with pytest.raises(
        ValidationRuntimeError,
        match="must be a dotted import name",
    ):
        required_validation_python_modules(
            ("pytest; import os",),
            environment={},
        )


def test_task_validation_python_modules_uses_normalized_metadata_key() -> None:
    task = PortalTask(
        task_id="FVT-TEST",
        title="Declare validation imports",
        status="todo",
        completion="manual",
        priority="P1",
        track="runtime",
        metadata={
            "Validation-Python_Modules": "requests, json",
        },
    )

    assert task_validation_python_modules(task) == ("requests", "json")


def test_validation_python_preflight_reports_missing_module_hermetically() -> None:
    missing_module = "ipfs_accelerate_dependency_that_does_not_exist_93f07d"

    receipt = preflight_validation_python_modules(
        (missing_module,),
        environment={},
    )

    assert receipt["schema"] == VALIDATION_PYTHON_MODULE_PREFLIGHT_SCHEMA
    assert receipt["passed"] is False
    assert receipt["reason"] == "validation_python_modules_unavailable"
    assert receipt["required_modules"] == ["pytest", missing_module]
    assert receipt["missing_modules"] == [missing_module]
    assert receipt["failed_modules"] == {}
    assert receipt["python_executable"] == str(Path(sys.executable).resolve())
    assert receipt["environment"] == {
        "home_is_private": True,
        "python_no_user_site": True,
        "site_user_enabled": False,
    }
    assert receipt["validation_python_launcher"]["sealed"] is (sys.platform.startswith("linux"))
    assert "user-site packages are intentionally unavailable" in str(receipt["action"])


def test_daemon_preflight_failure_does_not_consume_attempt_or_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "main")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.invalid")
    todo_path = repo / "todo.md"
    todo_path.write_text(
        """# Tasks

## FVT-083 Certify the release

- Status: todo
- Completion: manual
- Priority: P0
- Track: release
- Outputs: release.json
- Validation: python -m pytest test/test_release.py
- Validation Python modules: pytest, requests
- Acceptance: The release is certified.
""",
        encoding="utf-8",
    )
    _git(repo, "add", "todo.md")
    _git(repo, "commit", "-m", "seed task")
    state_dir = repo / "state"
    daemon = TodoImplementationDaemon(
        todo_path=todo_path,
        state_path=state_dir / "task-state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "events.jsonl",
        repo_root=repo,
        task_header_prefix="## FVT-",
        implement=True,
        implementation_command="provider-that-must-not-run",
    )
    receipt = {
        "schema": VALIDATION_PYTHON_MODULE_PREFLIGHT_SCHEMA,
        "passed": False,
        "reason": "validation_python_modules_unavailable",
        "required_modules": ["pytest", "requests"],
        "missing_modules": ["pytest"],
        "failed_modules": {},
        "python_executable": "/sealed/validation-python/bin/python",
        "action": "install pytest into the selected validation interpreter",
    }

    def failed_preflight(task: PortalTask) -> dict[str, object]:
        assert task_validation_python_modules(task) == (
            "pytest",
            "requests",
        )
        return dict(receipt)

    monkeypatch.setattr(
        daemon,
        "_validation_python_dependency_preflight",
        failed_preflight,
    )
    monkeypatch.setattr(
        daemon,
        "_run_implementation",
        lambda *_args, **_kwargs: pytest.fail(
            "model implementation was dispatched after failed preflight"
        ),
    )

    result = daemon.run_once()

    implementation = result["implementation_result"]
    assert implementation["reason"] == ("validation_python_dependency_preflight_failed")
    assert implementation["attempt"] == 1
    assert implementation["attempt_consumed"] is False
    assert implementation["provider_dispatched"] is False
    assert implementation["missing_modules"] == ["pytest"]
    assert implementation["validation_python_preflight"] == receipt
    state = PortalTaskState.load(daemon.state_path)
    assert state.implementation_attempts == {}
    assert state.implementation_attempts_by_cid == {}
    assert state.implementation_in_progress is False
    assert state.active_task_id == ""
    events = [
        json.loads(line)
        for line in daemon.events_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    [deferred] = [event for event in events if event.get("type") == "implementation_retry_deferred"]
    assert deferred["attempt_consumed"] is False
    assert deferred["provider_dispatched"] is False
    assert deferred["missing_modules"] == ["pytest"]
