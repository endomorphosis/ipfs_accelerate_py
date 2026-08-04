"""Optional validation path filtering for failed-validation thrash recovery."""

from __future__ import annotations

from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    PortalImplementationDaemon,
)


def test_drop_missing_optional_pytest_paths(tmp_path: Path) -> None:
    (tmp_path / "test" / "api").mkdir(parents=True)
    kept = tmp_path / "test" / "api" / "test_agent_supervisor_launch_guard.py"
    kept.write_text("def test_ok():\n    assert True\n", encoding="utf-8")
    command = (
        "python -m pytest "
        "test/api/test_agent_supervisor_launch_guard.py "
        "test/api/test_agent_supervisor_runtime_factory.py "
        "test/api/test_agent_supervisor_lifecycle_orchestrator.py -q"
    )
    rebuilt, notes = PortalImplementationDaemon._drop_missing_optional_validation_paths(
        command,
        workspace_path=tmp_path,
        declared_outputs={
            "test/api/test_agent_supervisor_launch_guard.py",
            "ipfs_accelerate_py/agent_supervisor/entrypoints/launch_guard.py",
        },
    )
    assert "launch_guard" in rebuilt
    assert "runtime_factory" not in rebuilt
    assert "lifecycle_orchestrator" not in rebuilt
    assert any("runtime_factory" in note for note in notes)


def test_keep_missing_declared_output_test_path(tmp_path: Path) -> None:
    command = "python -m pytest test/api/test_missing_declared.py -q"
    rebuilt, notes = PortalImplementationDaemon._drop_missing_optional_validation_paths(
        command,
        workspace_path=tmp_path,
        declared_outputs={"test/api/test_missing_declared.py"},
    )
    assert "test_missing_declared.py" in rebuilt
    assert any("kept_missing_declared_output" in note for note in notes)
