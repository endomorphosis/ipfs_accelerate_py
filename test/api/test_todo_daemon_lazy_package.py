"""Lazy package and runpy contracts for todo implementation entry points."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


ACCELERATE_ROOT = Path(__file__).resolve().parents[2]
PACKAGE = "ipfs_accelerate_py.agent_supervisor.todo_daemon"
DAEMON_MODULE = f"{PACKAGE}.implementation_daemon"
SUPERVISOR_MODULE = f"{PACKAGE}.implementation_supervisor"


def _environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONWARNINGS"] = "default"
    environment["PYTHONPATH"] = os.pathsep.join(
        part
        for part in (
            str(ACCELERATE_ROOT),
            environment.get("PYTHONPATH", ""),
        )
        if part
    )
    return environment


def _run_script(script: str) -> None:
    completed = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        cwd=ACCELERATE_ROOT,
        env=_environment(),
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_package_import_does_not_load_implementation_entrypoints() -> None:
    _run_script(
        f"""
        import sys
        import {PACKAGE} as package

        assert {DAEMON_MODULE!r} not in sys.modules
        assert {SUPERVISOR_MODULE!r} not in sys.modules
        assert "TodoImplementationDaemon" in package.__all__
        assert "TodoImplementationSupervisor" in package.__all__
        assert "implementation_daemon" in dir(package)
        assert "implementation_supervisor" in dir(package)
        """
    )


@pytest.mark.parametrize(
    ("module_name", "other_module", "export_names"),
    [
        (
            DAEMON_MODULE,
            SUPERVISOR_MODULE,
            (
                "DEFAULT_WORKTREE_SUBMODULE_PATHS",
                "TodoImplementationDaemon",
                "TodoTask",
                "TodoTaskState",
                "WORKTREE_SUBMODULE_PATHS",
                "normalize_relative_path_list",
                "parse_task_file",
            ),
        ),
        (
            SUPERVISOR_MODULE,
            "",
            (
                "TodoImplementationSupervisor",
                "TodoSupervisorConfig",
                "supervisor_config_from_args",
            ),
        ),
    ],
)
def test_lazy_exports_cache_exact_implementation_objects(
    module_name: str,
    other_module: str,
    export_names: tuple[str, ...],
) -> None:
    _run_script(
        f"""
        import importlib
        import sys
        import {PACKAGE} as package

        module_alias = {module_name!r}.rsplit(".", 1)[-1]
        first_name = {export_names[0]!r}
        first_value = getattr(package, first_name)
        module = importlib.import_module({module_name!r})
        assert getattr(package, module_alias) is module
        assert first_value is getattr(module, first_name)
        for export_name in {export_names!r}:
            root_value = getattr(package, export_name)
            assert root_value is getattr(module, export_name)
            assert package.__dict__[export_name] is root_value
        if {other_module!r}:
            assert {other_module!r} not in sys.modules
        """
    )


@pytest.mark.parametrize("module_name", [DAEMON_MODULE, SUPERVISOR_MODULE])
def test_module_help_has_no_preloaded_module_runpy_warning(
    module_name: str,
) -> None:
    completed = subprocess.run(
        [sys.executable, "-m", module_name, "--help"],
        cwd=ACCELERATE_ROOT,
        env=_environment(),
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "usage:" in completed.stdout.lower()
    assert "found in sys.modules after import of package" not in completed.stderr
