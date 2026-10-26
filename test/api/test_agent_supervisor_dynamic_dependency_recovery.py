"""Automatic repair of unresolved dynamic project dependencies."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.validation.project_dependency_preflight import (
    PROJECT_DEPENDENCY_PROBE_SCHEMA,
    _dependency_probe_receipt_from_output,
    preflight_validation_project_dependencies,
)


def _passing_probe(payloads: list[dict[str, object]]):
    def probe(payload, **_kwargs):
        payloads.append(payload)
        return {
            "schema": PROJECT_DEPENDENCY_PROBE_SCHEMA,
            "passed": True,
            "reason": "project_dependencies_satisfied",
            "projects": [],
        }

    return probe


def _write_dynamic_project(
    root: Path,
    *,
    contract: str = "",
    requirements_file: str = "",
) -> None:
    root.mkdir()
    (root / "setup.py").write_text(
        "from setuptools import setup\n"
        "setup(name='dynamic-project', install_requires=['packaging>=23'])\n",
        encoding="utf-8",
    )
    file_table = ""
    if requirements_file:
        (root / "requirements.txt").write_text(
            requirements_file,
            encoding="utf-8",
        )
        file_table = """
[tool.setuptools.dynamic]
dependencies = {file = ["requirements.txt"]}
"""
    (root / "pyproject.toml").write_text(
        """
[project]
name = "dynamic-project"
version = "1.0.0"
requires-python = ">=3.12"
dynamic = ["dependencies"]
"""
        + file_table
        + contract,
        encoding="utf-8",
    )


def test_literal_install_requires_repairs_undeclared_dynamic_contract(
    tmp_path: Path,
) -> None:
    project = tmp_path / "dynamic_project"
    _write_dynamic_project(project)
    payloads: list[dict[str, object]] = []

    receipt = preflight_validation_project_dependencies(
        tmp_path,
        ['cd dynamic_project && python -c "import json"'],
        probe_runner=_passing_probe(payloads),
    )

    assert receipt["passed"] is True
    assert receipt["automatic_install_attempted"] is False
    assert receipt["projects"][0]["dependency_source"] == (
        "setup_py_literal_install_requires"
    )
    assert payloads[0]["projects"][0]["requirements"] == ["packaging>=23"]


def test_absolute_pytest_runner_does_not_require_supervisor_pytest(
    tmp_path: Path,
) -> None:
    project = tmp_path / "dynamic_project"
    _write_dynamic_project(project)
    runner = tmp_path / "pytest"
    runner.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    os.chmod(runner, 0o755)
    payloads: list[dict[str, object]] = []

    receipt = preflight_validation_project_dependencies(
        tmp_path,
        [f"cd dynamic_project && {runner} -q"],
        probe_runner=_passing_probe(payloads),
    )

    assert receipt["passed"] is True
    assert receipt["projects"][0]["validation_dependency_source"] == (
        "validation_command_runner"
    )
    assert payloads[0]["projects"][0]["requirements"] == []


def test_declared_unimplemented_contract_stays_fail_closed(
    tmp_path: Path,
) -> None:
    command = "cd dynamic_project && python -m pytest -q"
    command_sha256 = hashlib.sha256(command.encode("utf-8")).hexdigest()
    project = tmp_path / "dynamic_project"
    _write_dynamic_project(
        project,
        contract=f"""
[tool.ipfs-accelerate-agent-supervisor.project-dependency-preflight]
schema = "ipfs_accelerate_py/agent-supervisor/scoped-project-dependency-preflight@3"
requires-python = ">=3.12"

[[tool.ipfs-accelerate-agent-supervisor.project-dependency-preflight.targets]]
target = "tests/test_declared.py"
validation-command-sha256 = "{command_sha256}"
requirements = ["pytest"]
""",
    )

    receipt = preflight_validation_project_dependencies(
        tmp_path,
        [command],
        probe_runner=lambda *_args, **_kwargs: pytest.fail(
            "a declared contract must not be repaired by install_requires"
        ),
    )

    assert receipt["passed"] is False
    assert receipt["projects"][0]["reason"] == "dynamic_dependencies_unresolved"


def test_broken_setuptools_dependency_file_is_not_bypassed(
    tmp_path: Path,
) -> None:
    project = tmp_path / "dynamic_project"
    _write_dynamic_project(project, requirements_file="-r other.txt\n")

    receipt = preflight_validation_project_dependencies(
        tmp_path,
        ['cd dynamic_project && python -c "import json"'],
        probe_runner=lambda *_args, **_kwargs: pytest.fail(
            "a declared dependency file must not fall back to setup.py"
        ),
    )

    assert receipt["passed"] is False
    assert receipt["projects"][0]["reason"] == "dynamic_dependencies_unresolved"


def test_probe_accepts_one_trailing_receipt_after_a_warning() -> None:
    receipt = {
        "schema": PROJECT_DEPENDENCY_PROBE_SCHEMA,
        "passed": True,
        "reason": "project_dependencies_satisfied",
    }
    output = (
        "There was a problem when trying to write in your cache folder "
        "(/nonexistent/ipfs-accelerate-validation/huggingface/hub).\n"
        + json.dumps(receipt)
        + "\n"
    )

    assert _dependency_probe_receipt_from_output(output) == receipt


def test_probe_rejects_two_json_objects() -> None:
    output = (
        json.dumps({"schema": PROJECT_DEPENDENCY_PROBE_SCHEMA, "passed": False})
        + "\n"
        + json.dumps({"schema": PROJECT_DEPENDENCY_PROBE_SCHEMA, "passed": True})
        + "\n"
    )

    with pytest.raises(ValueError, match="not one JSON receipt"):
        _dependency_probe_receipt_from_output(output)
