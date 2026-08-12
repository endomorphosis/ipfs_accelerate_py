"""Fail-closed tests for exact-target setup-extra dependency contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.validation.project_dependency_preflight import (
    PROJECT_DEPENDENCY_PROBE_SCHEMA,
    SCOPED_PROJECT_DEPENDENCY_CONTRACT_SCHEMA,
    preflight_validation_project_dependencies,
)

TARGET = "tests/unit/logic/gui_optimizer/test_models.py"
SETUP_TEST_EXTRA = [
    "pytest>=9.0.3,<10.0.0",
    "pytest-cov>=4.1.0",
]


def _content_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_project(
    tmp_path: Path,
    *,
    requirements: list[str] | None = None,
    requires_python: str = ">=3.12",
    setup_test_extra: list[str] | None = None,
    setup_source: str = "",
    extra_digest: str = "",
    setuptools_source: str = "",
    target: str = TARGET,
    target_source: str = "import pytest\n",
    target_digest: str = "",
    unknown_contract_field: bool = False,
) -> tuple[Path, str]:
    project = tmp_path / "project"
    project.mkdir()
    authority_requirements = (
        SETUP_TEST_EXTRA if setup_test_extra is None else setup_test_extra
    )
    if not setup_source:
        setup_source = (
            "from setuptools import setup\n"
            f"setup(extras_require={{'test': {authority_requirements!r}}})\n"
        )
    setup_payload = setup_source.encode("utf-8")
    (project / "setup.py").write_bytes(setup_payload)

    target_path = project / target
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_payload = target_source.encode("utf-8")
    target_path.write_bytes(target_payload)

    selected = (
        [authority_requirements[0]] if requirements is None else requirements
    )
    unknown = "unexpected = true\n" if unknown_contract_field else ""
    (project / "pyproject.toml").write_text(
        f"""
[project]
name = "scoped-project"
version = "1.0.0"
requires-python = ">=3.12"
dynamic = ["dependencies"]

{setuptools_source}

[tool.ipfs-accelerate-agent-supervisor.project-dependency-preflight]
schema = {json.dumps(SCOPED_PROJECT_DEPENDENCY_CONTRACT_SCHEMA)}
requires-python = {json.dumps(requires_python)}
covered-pytest-target = {json.dumps(target)}
covered-pytest-target-sha256 = {json.dumps(target_digest or hashlib.sha256(target_payload).hexdigest())}
requirements = {json.dumps(selected)}
authority = {{ file = "setup.py", sha256 = "{hashlib.sha256(setup_payload).hexdigest()}", extra = "test", extra-requirements-sha256 = "{extra_digest or _content_sha256(authority_requirements)}" }}
{unknown}
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return project, f"cd project && python3 -m pytest {target} -q"


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


def test_exact_target_contract_resolves_setup_extra_subset_without_execution(
    tmp_path: Path,
) -> None:
    _project, command = _write_project(tmp_path)
    payloads: list[dict[str, object]] = []

    receipt = preflight_validation_project_dependencies(
        tmp_path,
        [command],
        probe_runner=_passing_probe(payloads),
    )

    assert receipt["passed"] is True
    assert receipt["reason"] == (
        "approved_validation_environment_satisfies_project_dependencies"
    )
    project = receipt["projects"][0]
    assert project["dependency_source"] == (
        "agent_supervisor_scoped_setup_extra"
    )
    assert project["validation_dependency_source"] == (
        "scoped_dependency_contract"
    )
    assert project["dependency_contract_schema"] == (
        SCOPED_PROJECT_DEPENDENCY_CONTRACT_SCHEMA
    )
    assert project["selected_validation_extras"] == ["test"]
    assert project["requires_python_declared"] is True
    assert len(project["dependency_manifests"]) == 2
    assert all("path" not in manifest for manifest in project["dependency_manifests"])
    assert payloads[0]["projects"][0]["requirements"] == [
        SETUP_TEST_EXTRA[0]
    ]
    assert payloads[0]["projects"][0]["requirement_marker_extras"] == [
        "test"
    ]
    assert payloads[0]["projects"][0]["requires_python"] == ">=3.12"


@pytest.mark.parametrize(
    "command_suffix",
    [
        "::test_one -q",
        " -q tests/unit/logic/gui_optimizer/test_other.py",
        " -q -k one",
        " -q --maxfail=1",
        "",
    ],
)
def test_scoped_contract_rejects_selectors_and_cli_ambiguity(
    tmp_path: Path,
    command_suffix: str,
) -> None:
    _project, _command = _write_project(tmp_path)
    command = (
        "cd project && python3 -m pytest "
        f"{TARGET}{command_suffix}"
    )

    receipt = preflight_validation_project_dependencies(
        tmp_path,
        [command],
        probe_runner=lambda *_args, **_kwargs: pytest.fail(
            "out-of-scope commands must fail before probing"
        ),
    )

    assert receipt["passed"] is False
    assert receipt["reason"] == "project_dependency_contract_collection_failed"
    assert receipt["projects"][0]["reason"] == "dynamic_dependencies_unresolved"


def test_scoped_contract_does_not_weaken_project_wide_preflight(
    tmp_path: Path,
) -> None:
    _write_project(tmp_path)

    receipt = preflight_validation_project_dependencies(
        tmp_path,
        ["cd project && python3 -m pytest -q"],
        probe_runner=lambda *_args, **_kwargs: pytest.fail(
            "project-wide dynamic metadata must remain unresolved"
        ),
    )

    assert receipt["passed"] is False
    assert receipt["projects"][0]["reason"] == "dynamic_dependencies_unresolved"


def test_scoped_contract_rejects_marker_skipped_pytest_runner(
    tmp_path: Path,
) -> None:
    false_marker = "pytest>=999; python_version < '0'"
    _project, command = _write_project(
        tmp_path,
        requirements=[false_marker],
        setup_test_extra=[false_marker],
    )

    receipt = preflight_validation_project_dependencies(
        tmp_path,
        [command],
        probe_runner=lambda *_args, **_kwargs: pytest.fail(
            "a marker-skipped pytest runner must fail before probing"
        ),
    )

    assert receipt["passed"] is False
    assert receipt["projects"][0]["reason"] == "dynamic_dependencies_unresolved"


@pytest.mark.parametrize(
    "setuptools_source",
    [
        '[tool]\nsetuptools = "not-a-table"',
        '[tool.setuptools]\ndynamic = "not-a-table"',
    ],
)
def test_scoped_contract_rejects_malformed_setuptools_tables(
    tmp_path: Path,
    setuptools_source: str,
) -> None:
    _project, command = _write_project(
        tmp_path,
        setuptools_source=setuptools_source,
    )

    receipt = preflight_validation_project_dependencies(
        tmp_path,
        [command],
        probe_runner=lambda *_args, **_kwargs: pytest.fail(
            "malformed setuptools metadata must fail before probing"
        ),
    )

    assert receipt["passed"] is False
    assert receipt["projects"][0]["reason"] == "dynamic_dependencies_unresolved"


@pytest.mark.parametrize(
    "case",
    [
        "setup_bytes",
        "setup_extra_digest",
        "setup_computed_extra",
        "requirement_not_subset",
        "requires_python",
        "target_bytes",
        "unknown_field",
    ],
)
def test_scoped_contract_authority_and_scope_drift_fail_closed(
    tmp_path: Path,
    case: str,
) -> None:
    kwargs: dict[str, object] = {}
    if case == "setup_extra_digest":
        kwargs["extra_digest"] = "a" * 64
    elif case == "setup_computed_extra":
        kwargs["setup_source"] = (
            "from setuptools import setup\n"
            f"test_requirements = {SETUP_TEST_EXTRA!r}\n"
            "setup(extras_require={'test': test_requirements})\n"
        )
    elif case == "requirement_not_subset":
        kwargs["requirements"] = ["pytest>=8"]
    elif case == "requires_python":
        kwargs["requires_python"] = ">=3.11"
    elif case == "target_bytes":
        kwargs["target_digest"] = "b" * 64
    elif case == "unknown_field":
        kwargs["unknown_contract_field"] = True
    project, command = _write_project(tmp_path, **kwargs)
    if case == "setup_bytes":
        (project / "setup.py").write_text(
            (project / "setup.py").read_text(encoding="utf-8") + "# drift\n",
            encoding="utf-8",
        )

    receipt = preflight_validation_project_dependencies(
        tmp_path,
        [command],
        probe_runner=lambda *_args, **_kwargs: pytest.fail(
            "authority drift must fail before probing"
        ),
    )

    assert receipt["passed"] is False
    assert receipt["projects"][0]["reason"] == "dynamic_dependencies_unresolved"


def test_unknown_dynamic_project_without_scoped_contract_still_fails(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text(
        '[project]\nname="project"\nversion="1"\ndynamic=["dependencies"]\n',
        encoding="utf-8",
    )

    receipt = preflight_validation_project_dependencies(
        tmp_path,
        ["cd project && python3 -m pytest tests/test_one.py -q"],
        probe_runner=lambda *_args, **_kwargs: pytest.fail(
            "unresolved dynamic metadata must not probe"
        ),
    )

    assert receipt["passed"] is False
    assert receipt["projects"][0]["reason"] == "dynamic_dependencies_unresolved"


@pytest.mark.parametrize("authority_path", ["setup.py", TARGET])
def test_scoped_contract_authority_symlink_cannot_escape_project(
    tmp_path: Path,
    authority_path: str,
) -> None:
    project, command = _write_project(tmp_path)
    path = project / authority_path
    payload = path.read_bytes()
    outside = tmp_path / ("outside-" + Path(authority_path).name)
    outside.write_bytes(payload)
    path.unlink()
    path.symlink_to(outside)

    receipt = preflight_validation_project_dependencies(
        tmp_path,
        [command],
        probe_runner=lambda *_args, **_kwargs: pytest.fail(
            "escaping authority paths must fail before probing"
        ),
    )

    assert receipt["passed"] is False
    assert receipt["projects"][0]["reason"] == "dynamic_dependencies_unresolved"
    assert str(outside) not in json.dumps(receipt)


def test_scoped_contract_manifest_bytes_share_global_bound(tmp_path: Path) -> None:
    oversized_target = "#" * (2 * 1024 * 1024)
    _project, command = _write_project(
        tmp_path,
        target_source=oversized_target,
    )

    receipt = preflight_validation_project_dependencies(
        tmp_path,
        [command],
        probe_runner=lambda *_args, **_kwargs: pytest.fail(
            "oversized authority must fail before probing"
        ),
    )

    assert receipt["passed"] is False
    assert receipt["projects"][0]["reason"] == "dynamic_dependencies_unresolved"
