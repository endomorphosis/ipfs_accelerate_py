from __future__ import annotations

import importlib.metadata
import inspect
import json
import os
import sys
from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.validation import (
    project_dependency_preflight as preflight_module,
)
from ipfs_accelerate_py.agent_supervisor.validation.project_dependency_preflight import (
    MAX_DEPENDENCY_MANIFEST_FILES,
    MAX_INSTALLED_VERSION_BYTES,
    MAX_PYPROJECT_BYTES,
    PROJECT_DEPENDENCY_PREFLIGHT_SCHEMA,
    PROJECT_DEPENDENCY_PROBE_SCHEMA,
    _evaluate_dependency_payload,
    _run_bounded_probe_process,
    preflight_validation_project_dependencies,
    project_dependency_preflight_backoff_seconds,
)
from ipfs_accelerate_py.agent_supervisor.validation.validation_commands import (
    validation_command_repository_root,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _probe_payload(*requirements: str) -> dict[str, object]:
    return {
        "schema": PROJECT_DEPENDENCY_PROBE_SCHEMA,
        "projects": [
            {
                "root": "ipfs_kit_py",
                "project_name_sha256": "b" * 64,
                "pyproject_sha256": "a" * 64,
                "requirements": list(requirements),
                "requirement_marker_extras": [""] * len(requirements),
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
    manifest = receipt["projects"][0]["dependency_manifests"][0]
    assert "path" not in manifest
    assert manifest["path_sha256"]
    assert manifest["content_sha256"]
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


def test_pyproject_symlink_cannot_escape_project_root(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    project = workspace / "project"
    project.mkdir(parents=True)
    outside_secret = "outside-pyproject-secret-must-not-persist"
    outside = tmp_path / "outside.toml"
    outside.write_text(
        f'[project]\nname = "{outside_secret}"\nversion = "1.0"\n',
        encoding="utf-8",
    )
    (project / "pyproject.toml").symlink_to(outside)

    receipt = preflight_validation_project_dependencies(
        workspace,
        ["cd project && python -c 'pass'"],
        probe_runner=lambda *_args, **_kwargs: pytest.fail(
            "an escaping pyproject must fail before probing"
        ),
    )

    assert receipt["passed"] is False
    assert receipt["projects"][0]["reason"] == ("pyproject_path_or_snapshot_invalid")
    assert outside_secret not in json.dumps(receipt)


def test_dynamic_dependency_manifest_symlink_cannot_escape_project_root(
    tmp_path,
) -> None:
    workspace = tmp_path / "workspace"
    project = workspace / "project"
    project.mkdir(parents=True)
    outside_secret = "outside-manifest-secret-must-not-persist"
    outside = tmp_path / "outside-requirements.txt"
    outside.write_text(
        f"private-package @ https://{outside_secret}.invalid/pkg.whl\n",
        encoding="utf-8",
    )
    (project / "requirements.txt").symlink_to(outside)
    (project / "pyproject.toml").write_text(
        """
[project]
name = "project"
version = "1.0"
dynamic = ["dependencies"]

[tool.setuptools.dynamic]
dependencies = {file = ["requirements.txt"]}
""".strip(),
        encoding="utf-8",
    )

    receipt = preflight_validation_project_dependencies(
        workspace,
        ["cd project && python -c 'pass'"],
        probe_runner=lambda *_args, **_kwargs: pytest.fail(
            "an escaping manifest must fail before probing"
        ),
    )

    assert receipt["passed"] is False
    assert receipt["projects"][0]["reason"] == ("dynamic_dependencies_unresolved")
    assert outside_secret not in json.dumps(receipt)


def test_oversized_sparse_pyproject_is_rejected_without_reading(
    tmp_path,
    monkeypatch,
) -> None:
    workspace = tmp_path / "workspace"
    project = workspace / "project"
    project.mkdir(parents=True)
    pyproject = project / "pyproject.toml"
    pyproject.touch()
    os.truncate(pyproject, MAX_PYPROJECT_BYTES + 1)

    def forbidden_read(*_args, **_kwargs):
        pytest.fail("an oversized stat result must be rejected before os.read")

    monkeypatch.setattr(preflight_module.os, "read", forbidden_read)
    receipt = preflight_validation_project_dependencies(
        workspace,
        ["cd project && python -c 'pass'"],
    )

    assert receipt["passed"] is False
    assert receipt["projects"][0]["reason"] == ("pyproject_exceeds_preflight_bound")
    assert receipt["projects"][0]["pyproject_bytes"] == (MAX_PYPROJECT_BYTES + 1)


def test_pyproject_mutation_during_bounded_read_fails_closed(
    tmp_path,
    monkeypatch,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    pyproject = project / "pyproject.toml"
    pyproject.write_text(
        """
[project]
name = "project"
version = "1.0"
dependencies = []
""".strip(),
        encoding="utf-8",
    )
    original_read = os.read
    mutated = False

    def racing_read(descriptor, maximum_bytes):
        nonlocal mutated
        chunk = original_read(descriptor, maximum_bytes)
        if chunk and not mutated:
            mutated = True
            with pyproject.open("ab") as stream:
                stream.write(b"\n# concurrent mutation\n")
        return chunk

    monkeypatch.setattr(preflight_module.os, "read", racing_read)
    receipt = preflight_validation_project_dependencies(
        tmp_path,
        ["cd project && python -c 'pass'"],
        probe_runner=lambda *_args, **_kwargs: pytest.fail(
            "a raced snapshot must fail before probing"
        ),
    )

    assert mutated is True
    assert receipt["passed"] is False
    assert receipt["projects"][0]["reason"] == ("pyproject_path_or_snapshot_invalid")


def test_bounded_metadata_reader_uses_python_38_compatible_stat_api() -> None:
    source = inspect.getsource(preflight_module._read_bounded_contained_regular_file)

    assert ".stat(follow_symlinks=False)" not in source
    assert source.count("os.stat(") == 2


def test_project_name_and_requires_python_are_hash_only_in_receipts(
    tmp_path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    project_name_secret = "private-project-name-secret"
    python_secret = "invalid-python-contract-secret"
    (project / "pyproject.toml").write_text(
        f"""
[project]
name = "{project_name_secret}"
version = "1.0"
requires-python = "{python_secret}"
dependencies = []
""".strip(),
        encoding="utf-8",
    )

    receipt = preflight_validation_project_dependencies(
        tmp_path,
        ["cd project && python -c 'pass'"],
    )
    serialized = json.dumps(receipt, sort_keys=True)

    assert receipt["passed"] is False
    assert receipt["projects"][0]["project_name_sha256"]
    assert receipt["projects"][0]["requires_python_sha256"]
    assert "requires_python" not in receipt["projects"][0]
    assert receipt["invalid_requirements"][0]["kind"] == ("requires-python")
    assert receipt["invalid_requirements"][0]["requirement_sha256"]
    assert project_name_secret not in serialized
    assert python_secret not in serialized


@pytest.mark.parametrize(
    ("installed_version", "error_type"),
    [
        ("invalid-installed-version-secret", "InvalidVersion"),
        (
            "x" * (MAX_INSTALLED_VERSION_BYTES + 1),
            "InstalledVersionExceedsBound",
        ),
    ],
    ids=("invalid", "oversized"),
)
def test_invalid_installed_versions_are_bounded_and_hash_only(
    installed_version,
    error_type,
) -> None:
    result = _evaluate_dependency_payload(
        _probe_payload("private-package>=1"),
        version_getter=lambda _name: installed_version,
    )
    serialized = json.dumps(result, sort_keys=True)
    invalid = result["projects"][0]["invalid"][0]

    assert result["passed"] is False
    assert invalid["kind"] == "installed_version"
    assert invalid["error_type"] == error_type
    assert invalid["installed_version_sha256"]
    assert installed_version not in serialized


def test_valid_installed_version_is_canonicalized_before_persistence() -> None:
    result = _evaluate_dependency_payload(
        _probe_payload("private-package>=1"),
        version_getter=lambda _name: "01.0",
    )

    assert result["passed"] is True
    assert result["projects"][0]["observed"][0]["installed_version"] == "1.0"
    assert result["projects"][0]["observed"][0]["installed_version_sha256"]


def test_setuptools_dynamic_test_extra_is_loaded_with_marker_provenance(
    tmp_path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "requirements-test.txt").write_text(
        "selected-only>=1; extra == 'test'\n",
        encoding="utf-8",
    )
    (project / "pyproject.toml").write_text(
        """
[project]
name = "project"
version = "1.0"
dependencies = []
dynamic = ["optional-dependencies"]

[tool.setuptools.dynamic.optional-dependencies]
test = {file = ["requirements-test.txt"]}
""".strip(),
        encoding="utf-8",
    )

    receipt = preflight_validation_project_dependencies(
        tmp_path,
        ["cd project && python -m pytest -q"],
        probe_runner=lambda payload, **_kwargs: _evaluate_dependency_payload(
            payload,
            version_getter={
                "pytest": "9.0.3",
                "selected-only": "1.2.0",
            }.__getitem__,
        ),
    )

    assert receipt["passed"] is True
    assert receipt["projects"][0]["selected_validation_extras"] == ["test"]
    assert receipt["projects"][0]["validation_dependency_source"] == ("setuptools_dynamic_file")
    observed = receipt["probe"]["projects"][0]["observed"]
    assert any(
        item["name"] == "selected-only" and item["selected_extra"] == "test" for item in observed
    )


@pytest.mark.parametrize(
    "pyproject_optional_contract",
    [
        """
[project]
name = "project"
version = "1.0"
dynamic = ["optional-dependencies"]
""",
        """
[project]
name = "project"
version = "1.0"
dynamic = ["optional-dependencies"]

[project.optional-dependencies]
test = ["private-package>=1"]

[tool.setuptools.dynamic.optional-dependencies]
test = {file = ["requirements-test.txt"]}
""",
    ],
    ids=("unresolved", "static-and-dynamic"),
)
def test_dynamic_validation_extra_fails_closed_when_contract_is_ambiguous(
    tmp_path,
    pyproject_optional_contract,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "requirements-test.txt").write_text(
        "private-package>=1\n",
        encoding="utf-8",
    )
    (project / "pyproject.toml").write_text(
        pyproject_optional_contract.strip(),
        encoding="utf-8",
    )

    receipt = preflight_validation_project_dependencies(
        tmp_path,
        ["cd project && python -m pytest -q"],
        probe_runner=lambda *_args, **_kwargs: pytest.fail(
            "an ambiguous selected extra must fail before probing"
        ),
    )

    assert receipt["passed"] is False
    assert receipt["projects"][0]["reason"] == ("validation_dependencies_unresolved")


@pytest.mark.parametrize(
    ("requires_python_contract", "expected_reason"),
    [
        (
            'dynamic = ["requires-python"]',
            "pep621_requires_python_dynamic_unresolved",
        ),
        ('requires-python = 0', "pep621_requires_python_must_be_string"),
    ],
)
def test_unresolved_or_falsey_requires_python_fails_closed(
    tmp_path,
    requires_python_contract,
    expected_reason,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text(
        f"""
[project]
name = "project"
version = "1.0"
dependencies = []
{requires_python_contract}
""".strip(),
        encoding="utf-8",
    )

    receipt = preflight_validation_project_dependencies(
        tmp_path,
        ["cd project && python -c 'pass'"],
        probe_runner=lambda *_args, **_kwargs: pytest.fail(
            "an unresolved interpreter contract must fail before probing"
        ),
    )

    assert receipt["passed"] is False
    assert receipt["projects"][0]["reason"] == expected_reason


def test_dynamic_runtime_and_validation_manifests_share_one_file_bound(
    tmp_path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    runtime_files = [
        f"requirements-{index}.txt"
        for index in range(MAX_DEPENDENCY_MANIFEST_FILES)
    ]
    for path in runtime_files:
        (project / path).write_text("packaging>=23.2\n", encoding="utf-8")
    (project / "requirements-test.txt").write_text(
        "pytest>=8\n",
        encoding="utf-8",
    )
    (project / "pyproject.toml").write_text(
        f"""
[project]
name = "project"
version = "1.0"
dynamic = ["dependencies", "optional-dependencies"]

[tool.setuptools.dynamic]
dependencies = {{file = {json.dumps(runtime_files)}}}

[tool.setuptools.dynamic.optional-dependencies]
test = {{file = ["requirements-test.txt"]}}
""".strip(),
        encoding="utf-8",
    )

    receipt = preflight_validation_project_dependencies(
        tmp_path,
        ["cd project && python -m pytest -q"],
        probe_runner=lambda *_args, **_kwargs: pytest.fail(
            "an over-bounded manifest set must fail before probing"
        ),
    )

    assert receipt["passed"] is False
    assert receipt["projects"][0]["reason"] == "validation_dependencies_unresolved"


@pytest.mark.parametrize(
    "command",
    [
        "(cd child && python -m pytest -q)",
        "bash -c 'cd child && python -m pytest -q'",
        "pushd child && python -m pytest -q",
        "popd && python -m pytest -q",
        ". ./activate && python -m pytest -q",
        "source ./activate && python -m pytest -q",
        "eval 'cd child && python -m pytest -q'",
        "if true; then cd child; fi; python -m pytest -q",
        "function enter { cd child; }; enter; python -m pytest -q",
        "{ cd child; python -m pytest -q; }",
    ],
)
def test_unrecognized_cwd_changing_validation_syntax_fails_closed(
    tmp_path,
    command,
) -> None:
    receipt = preflight_validation_project_dependencies(
        tmp_path,
        [command],
        probe_runner=lambda *_args, **_kwargs: pytest.fail(
            "an unsafe validation root must fail before probing"
        ),
    )

    assert receipt["passed"] is False
    assert receipt["reason"] == ("project_dependency_contract_collection_failed")
    assert receipt["invalid_commands"][0]["reason"] == ("validation_repository_root_is_unsafe")


@pytest.mark.parametrize(
    "command",
    [
        "python -m pytest .",
        "echo source",
        "echo bash -c",
    ],
)
def test_ordinary_shell_arguments_do_not_change_inferred_repository_root(
    command,
) -> None:
    assert validation_command_repository_root(command) == ""


def test_quoted_pytest_module_selects_validation_extra(tmp_path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text(
        """
[project]
name = "project"
version = "1.0"
dependencies = []

[project.optional-dependencies]
test = ["selected-only>=1"]
""".strip(),
        encoding="utf-8",
    )
    payloads = []

    receipt = preflight_validation_project_dependencies(
        tmp_path,
        ["cd project && python -m 'pytest' -q"],
        probe_runner=lambda payload, **_kwargs: (
            payloads.append(payload)
            or {
                "schema": PROJECT_DEPENDENCY_PROBE_SCHEMA,
                "passed": True,
                "reason": "project_dependencies_satisfied",
                "projects": [],
            }
        ),
    )

    assert receipt["passed"] is True
    assert receipt["projects"][0]["selected_validation_extras"] == ["test"]
    assert payloads[0]["projects"][0]["requirements"] == [
        "pytest",
        "selected-only>=1",
    ]
    assert payloads[0]["projects"][0]["requirement_marker_extras"] == ["", "test"]


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
