from __future__ import annotations

import importlib.metadata
import json
import os
import sys
from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.validation import (
    project_dependency_preflight as preflight_module,
)
from ipfs_accelerate_py.agent_supervisor.validation.project_dependency_preflight import (
    PROJECT_DEPENDENCY_PREFLIGHT_SCHEMA,
    PROJECT_DEPENDENCY_PROBE_SCHEMA,
    _evaluate_dependency_payload,
    _run_bounded_probe_process,
    preflight_validation_project_dependencies,
    project_dependency_preflight_backoff_seconds,
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
    assert result["projects"][0]["missing"][0]["name"] == "hypercorn"
    assert result["projects"][0]["missing"][0]["requirement"] == ("hypercorn>=0.16.0")
    assert result["projects"][0]["missing"][0]["requirement_sha256"]


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
    assert result["projects"][0]["marker_skipped"][0]["name"] == ("missing-only-on-python-one")
    assert "marker_sha256" in result["projects"][0]["marker_skipped"][0]


def test_preflight_fails_closed_on_unknown_dynamic_dependencies(
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
    assert receipt["passed"] is False
    assert receipt["applicable"] is False
    assert receipt["reason"] == ("project_dependency_contract_collection_failed")
    assert receipt["projects"][0]["reason"] == ("dynamic_dependencies_unresolved")
    assert receipt["automatic_install_attempted"] is False


def test_preflight_resolves_setuptools_file_backed_dynamic_dependencies(
    tmp_path,
) -> None:
    project = tmp_path / "dynamic_project"
    project.mkdir()
    (project / "requirements.txt").write_text(
        "# runtime\npackaging>=23.2\n",
        encoding="utf-8",
    )
    (project / "pyproject.toml").write_text(
        """
[project]
name = "dynamic-project"
version = "1.0.0"
dynamic = ["dependencies"]

[tool.setuptools.dynamic]
dependencies = {file = ["requirements.txt"]}
""".strip(),
        encoding="utf-8",
    )
    payloads: list[dict[str, object]] = []

    def passing_probe(payload, **_kwargs):
        payloads.append(payload)
        return {
            "schema": PROJECT_DEPENDENCY_PROBE_SCHEMA,
            "passed": True,
            "reason": "project_dependencies_satisfied",
            "projects": [],
        }

    receipt = preflight_validation_project_dependencies(
        tmp_path,
        ["cd dynamic_project && python -c 'pass'"],
        probe_runner=passing_probe,
    )

    assert receipt["passed"] is True
    assert receipt["projects"][0]["dependency_source"] == ("setuptools_dynamic_file")
    assert receipt["projects"][0]["dependency_manifests"][0]["path"] == ("requirements.txt")
    assert payloads[0]["projects"][0]["requirements"] == ["packaging>=23.2"]


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
    assert receipt["automatic_install_attempted"] is False


def test_preflight_unexpected_collection_error_returns_typed_receipt(
    tmp_path,
) -> None:
    missing_workspace = tmp_path / "missing"
    receipt = preflight_validation_project_dependencies(
        missing_workspace,
        ["python -m pytest -q"],
    )

    assert receipt["passed"] is False
    assert receipt["reason"] == ("project_dependency_preflight_infrastructure_error")
    assert receipt["error_type"] == "FileNotFoundError"
    assert receipt["retry_fingerprint"] != receipt["receipt_id"]
    assert "python -m pytest" not in json.dumps(receipt)


def test_dependency_retry_fingerprint_ignores_ephemeral_workspace_path(
    tmp_path,
) -> None:
    receipts = []
    for name in ("attempt-one", "attempt-two"):
        workspace = tmp_path / name
        project = workspace / "static_project"
        project.mkdir(parents=True)
        (project / "pyproject.toml").write_text(
            """
[project]
name = "static-project"
version = "1.0.0"
dependencies = ["packaging>=23.2"]
""".strip(),
            encoding="utf-8",
        )
        receipts.append(
            preflight_validation_project_dependencies(
                workspace,
                ["cd static_project && python -c 'pass'"],
                probe_runner=lambda *_args, **_kwargs: {
                    "schema": PROJECT_DEPENDENCY_PROBE_SCHEMA,
                    "passed": False,
                    "reason": "project_dependency_drift_detected",
                    "projects": [
                        {
                            "passed": False,
                            "missing": [
                                {
                                    "name": "packaging",
                                    "requirement": "packaging>=23.2",
                                }
                            ],
                            "incompatible": [],
                            "invalid": [],
                        }
                    ],
                },
            )
        )

    assert receipts[0]["receipt_id"] != receipts[1]["receipt_id"]
    assert receipts[0]["retry_fingerprint"] == (receipts[1]["retry_fingerprint"])


def test_dependency_preflight_backoff_is_fingerprinted_and_bounded() -> None:
    fingerprint = "sha256:stable"

    assert (
        project_dependency_preflight_backoff_seconds(
            fingerprint,
            [],
        )
        == 300
    )
    assert (
        project_dependency_preflight_backoff_seconds(
            fingerprint,
            [fingerprint],
        )
        == 600
    )
    assert (
        project_dependency_preflight_backoff_seconds(
            fingerprint,
            [fingerprint, fingerprint],
        )
        == 1200
    )
    assert (
        project_dependency_preflight_backoff_seconds(
            fingerprint,
            [fingerprint] * 20,
        )
        == 1800
    )
    assert (
        project_dependency_preflight_backoff_seconds(
            fingerprint,
            ["sha256:changed", fingerprint],
        )
        == 300
    )


def test_dependency_probe_output_is_bounded_while_child_is_running(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        preflight_module,
        "MAX_PROBE_OUTPUT_BYTES",
        128,
    )

    returncode, output, error = _run_bounded_probe_process(
        [
            sys.executable,
            "-c",
            "import sys; sys.stdout.write('x' * 100000); sys.stdout.flush()",
        ],
        input_payload=b"",
        environment=os.environ,
    )

    assert returncode is not None
    assert len(output) <= 128
    assert error["reason"] == "dependency_probe_output_exceeded_bound"


def test_pytest_command_selects_declared_test_extra_deterministically(
    tmp_path,
) -> None:
    project = tmp_path / "static_project"
    project.mkdir()
    (project / "pyproject.toml").write_text(
        """
[project]
name = "static-project"
version = "1.0.0"
dependencies = ["packaging>=23.2"]

[project.optional-dependencies]
dev = ["dev-only>=1"]
testing = ["testing-only>=2"]
test = ["test-only>=3"]
""".strip(),
        encoding="utf-8",
    )
    payloads: list[dict[str, object]] = []

    def passing_probe(payload, **_kwargs):
        payloads.append(payload)
        return {
            "schema": PROJECT_DEPENDENCY_PROBE_SCHEMA,
            "passed": True,
            "reason": "project_dependencies_satisfied",
            "projects": [],
        }

    receipt = preflight_validation_project_dependencies(
        tmp_path,
        ["cd static_project && python -m pytest -q"],
        probe_runner=passing_probe,
    )

    assert receipt["passed"] is True
    assert receipt["projects"][0]["selected_validation_extras"] == ["test"]
    assert payloads[0]["projects"][0]["requirements"] == [
        "packaging>=23.2",
        "pytest",
        "test-only>=3",
    ]


def test_direct_reference_is_rejected_without_persisting_url_secrets(
    tmp_path,
) -> None:
    project = tmp_path / "static_project"
    project.mkdir()
    secret_url = "https://user:top-secret@example.invalid/pkg.whl?signature=do-not-persist"
    (project / "pyproject.toml").write_text(
        f"""
[project]
name = "static-project"
version = "1.0.0"
dependencies = ["private-pkg @ {secret_url}"]
""".strip(),
        encoding="utf-8",
    )

    receipt = preflight_validation_project_dependencies(
        tmp_path,
        ["cd static_project && python -c 'pass'"],
    )
    serialized = json.dumps(receipt, sort_keys=True)

    assert receipt["passed"] is False
    assert receipt["invalid_requirements"][0]["kind"] == ("direct_reference_unverifiable")
    assert "direct_reference_sha256" in receipt["invalid_requirements"][0]
    assert "top-secret" not in serialized
    assert "do-not-persist" not in serialized
    assert secret_url not in serialized


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
    assert receipt["probe"]["preflight_source_delivery"]["mode"] == ("compressed_argv_copy")
    assert (
        receipt["probe"]["probe_source_sha256"]
        == (receipt["probe"]["preflight_source_delivery"]["sha256"])
    )
    assert receipt["probe"]["projects"][0]["observed"][0]["name"] == ("packaging")
