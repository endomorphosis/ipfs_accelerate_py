from __future__ import annotations

import importlib.metadata
import sys
from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.validation.project_dependency_preflight import (
    PROJECT_DEPENDENCY_PREFLIGHT_SCHEMA,
    PROJECT_DEPENDENCY_PROBE_SCHEMA,
    _evaluate_dependency_payload,
    preflight_validation_project_dependencies,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _probe_payload(*requirements: str) -> dict[str, object]:
    return {
        "schema": PROJECT_DEPENDENCY_PROBE_SCHEMA,
        "projects": [
            {
                "root": "ipfs_kit_py",
                "project_name": "ipfs_kit_py",
                "pyproject_sha256": "a" * 64,
                "requirements": list(requirements),
                "requires_python": ">=3.10",
            }
        ],
    }


def test_dependency_probe_detects_preprovisioning_hypercorn_drift() -> None:
    def missing_distribution(name: str) -> str:
        raise importlib.metadata.PackageNotFoundError(name)

    result = _evaluate_dependency_payload(
        _probe_payload("hypercorn>=0.16.0"),
        version_getter=missing_distribution,
    )

    assert result["passed"] is False
    assert result["reason"] == "project_dependency_drift_detected"
    assert result["projects"][0]["missing"] == [
        {
            "name": "hypercorn",
            "requirement": "hypercorn>=0.16.0",
        }
    ]


def test_dependency_probe_accepts_provisioned_hypercorn_state() -> None:
    installed = {
        "hypercorn": "0.18.0",
        "priority": "2.0.0",
    }

    result = _evaluate_dependency_payload(
        _probe_payload(
            "hypercorn>=0.16.0",
            "priority>=2.0.0",
            "missing-only-on-python-one; python_version < '2'",
        ),
        version_getter=installed.__getitem__,
    )

    assert result["passed"] is True
    assert result["reason"] == "project_dependencies_satisfied"
    assert result["projects"][0]["missing"] == []
    assert result["projects"][0]["incompatible"] == []
    assert result["projects"][0]["marker_skipped"] == [
        "missing-only-on-python-one; python_version < '2'"
    ]


def test_preflight_does_not_guess_dynamic_dependencies(
    tmp_path,
) -> None:
    project = tmp_path / "dynamic_project"
    project.mkdir()
    (project / "pyproject.toml").write_text(
        """
[project]
name = "dynamic-project"
version = "1.0.0"
dynamic = ["dependencies"]
""".strip(),
        encoding="utf-8",
    )

    def forbidden_probe(*_args, **_kwargs):
        pytest.fail("dynamic dependencies must not trigger a distribution probe")

    receipt = preflight_validation_project_dependencies(
        tmp_path,
        ["cd dynamic_project && python -m pytest -q"],
        probe_runner=forbidden_probe,
    )

    assert receipt["schema"] == PROJECT_DEPENDENCY_PREFLIGHT_SCHEMA
    assert receipt["passed"] is True
    assert receipt["applicable"] is False
    assert receipt["reason"] == "no_static_pep621_dependencies"
    assert receipt["projects"][0]["reason"] == ("project_dependencies_are_dynamic")
    assert receipt["install_attempted"] is False
    assert receipt["network_accessed"] is False


def test_preflight_fails_closed_on_invalid_static_dependency_contract(
    tmp_path,
) -> None:
    project = tmp_path / "invalid_project"
    project.mkdir()
    (project / "pyproject.toml").write_text(
        """
[project]
name = "invalid-project"
version = "1.0.0"
dependencies = "hypercorn>=0.16.0"
""".strip(),
        encoding="utf-8",
    )

    receipt = preflight_validation_project_dependencies(
        tmp_path,
        ["cd invalid_project && python -m pytest -q"],
        probe_runner=lambda *_args, **_kwargs: pytest.fail(
            "invalid metadata must fail before probing"
        ),
    )

    assert receipt["passed"] is False
    assert receipt["reason"] == ("project_dependency_contract_collection_failed")
    assert receipt["projects"][0]["reason"] == ("pep621_dependencies_must_be_static_strings")


def test_preflight_fails_closed_when_approved_probe_is_unavailable(
    tmp_path,
) -> None:
    project = tmp_path / "static_project"
    project.mkdir()
    (project / "pyproject.toml").write_text(
        """
[project]
name = "static-project"
version = "1.0.0"
dependencies = ["hypercorn>=0.16.0"]
""".strip(),
        encoding="utf-8",
    )

    def unavailable_probe(*_args, **_kwargs):
        raise RuntimeError("approved interpreter unavailable")

    receipt = preflight_validation_project_dependencies(
        tmp_path,
        ["cd static_project && python -m pytest -q"],
        probe_runner=unavailable_probe,
    )

    assert receipt["passed"] is False
    assert receipt["reason"] == ("approved_validation_environment_dependency_probe_failed")
    assert receipt["probe"]["reason"] == "dependency_probe_infrastructure_error"
    assert receipt["remediation"]["kind"] == ("repair_approved_validation_dependency_probe")
    assert receipt["install_attempted"] is False
    assert receipt["network_accessed"] is False


def test_dependency_probe_runtime_is_declared_through_packaging_source_of_truth() -> None:
    requirements = {
        line.strip()
        for line in (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    setup_source = (REPO_ROOT / "setup.py").read_text(encoding="utf-8")

    assert "packaging>=23.2" in requirements
    assert (
        'install_requires = _read_requirements(this_directory / "requirements.txt")' in setup_source
    )
    assert "install_requires=install_requires" in setup_source


@pytest.mark.skipif(
    sys.version_info < (3, 10),
    reason="the fixture's static requires-python intentionally needs 3.10",
)
def test_preflight_uses_approved_interpreter_for_installed_metadata(
    tmp_path,
) -> None:
    project = tmp_path / "static_project"
    project.mkdir()
    (project / "pyproject.toml").write_text(
        """
[project]
name = "static-project"
version = "1.0.0"
requires-python = ">=3.10"
dependencies = ["packaging>=23.2"]
""".strip(),
        encoding="utf-8",
    )

    receipt = preflight_validation_project_dependencies(
        tmp_path,
        ["cd static_project && python -m pytest -q"],
    )

    assert receipt["passed"] is True
    assert receipt["applicable"] is True
    assert receipt["reason"] == ("approved_validation_environment_satisfies_project_dependencies")
    assert receipt["probe"]["python_executable"]
    assert receipt["probe"]["validation_python_launcher"]["sealed"] is (
        sys.platform.startswith("linux")
    )
    assert receipt["probe"]["projects"][0]["observed"][0]["name"] == ("packaging")
