"""Focused compatibility and fail-closed tests for scoped contract v3."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.validation.project_dependency_preflight import (
    PROJECT_DEPENDENCY_PROBE_SCHEMA,
    SCOPED_PROJECT_DEPENDENCY_CONTRACT_SCHEMA_V2,
    SCOPED_PROJECT_DEPENDENCY_CONTRACT_SCHEMA_V3,
    _command_is_exact_v3_scoped_pytest_target,
    _evaluate_dependency_payload,
    preflight_validation_project_dependencies,
)

REQUIREMENTS = ["pytest>=9.0.3,<10.0.0", "pytest-cov>=4.1.0"]


def _content_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_mixed_root_project(
    workspace: Path,
    *,
    schema: str = SCOPED_PROJECT_DEPENDENCY_CONTRACT_SCHEMA_V3,
    extra: str = "lgcvf-validation",
    setup_source: str = "",
) -> tuple[Path, str, dict[str, object], list[dict[str, object]]]:
    relative_root = "ipfs_datasets_py"
    project = workspace / relative_root
    project.mkdir()
    if not setup_source:
        setup_source = (
            "from setuptools import setup\n"
            f"setup(extras_require={{{extra!r}: {REQUIREMENTS!r}}})\n"
        )
    setup_payload = setup_source.encode("utf-8")
    (project / "setup.py").write_bytes(setup_payload)

    targets = (
        (
            "tests/unit/logic/software_contracts/test_external.py",
            "external/ipfs_datasets",
        ),
        (
            "tests/unit/logic/software_verification/test_incremental.py",
            relative_root,
        ),
    )
    entries: list[dict[str, object]] = []
    encoded_entries: list[str] = []
    for index, (target, declared_root) in enumerate(targets):
        command = (
            f"cd {declared_root} && python -m pytest -q {target}"
        )
        command_sha256 = hashlib.sha256(command.encode("utf-8")).hexdigest()
        declared_output = f"{declared_root}/{target}"
        target_payload = f"# nested target {index}\n".encode()
        target_path = project / target
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes(target_payload)
        requirements = REQUIREMENTS[: index + 1]
        entry = {
            "target": target,
            "command": command,
            "command_sha256": command_sha256,
            "requirements": list(requirements),
            "board_namespace": "mixed-root-board",
            "canonical_task_cid": f"mixed-root-task-{index}",
            "declared_output": declared_output,
        }
        entries.append(entry)
        encoded_entries.append(
            "{ "
            f"target = {json.dumps(target)}, "
            f"validation-command-sha256 = {json.dumps(command_sha256)}, "
            f"requirements = {json.dumps(requirements)}, "
            "task = { "
            'board-namespace = "mixed-root-board", '
            f"canonical-task-cid = \"mixed-root-task-{index}\", "
            f"declared-output = {json.dumps(declared_output)}"
            " }, "
            'baseline = { state = "present", sha256 = '
            f'"{hashlib.sha256(target_payload).hexdigest()}" }}'
            " }"
        )

    joined_entries = ",\n".join(encoded_entries)
    (project / "pyproject.toml").write_text(
        f"""
[project]
name = "scoped-project-mixed-roots"
version = "1.0.0"
requires-python = ">=3.12"
dynamic = ["dependencies"]

[tool.ipfs-accelerate-agent-supervisor.project-dependency-preflight]
schema = {json.dumps(schema)}
requires-python = ">=3.12"
authority = {{ file = "setup.py", sha256 = "{hashlib.sha256(setup_payload).hexdigest()}", extra = {json.dumps(extra)}, extra-requirements-sha256 = "{_content_sha256(REQUIREMENTS)}" }}
targets = [
{joined_entries}
]
""".strip()
        + "\n",
        encoding="utf-8",
    )
    selected = entries[1]
    task_authority = {
        "board_namespace": selected["board_namespace"],
        "canonical_task_cid": selected["canonical_task_cid"],
        "declared_outputs": [selected["declared_output"]],
    }
    return project, str(selected["command"]), task_authority, entries


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


def test_v3_selects_exact_task_from_mixed_nested_repository_roots(
    tmp_path: Path,
) -> None:
    _project, command, task_authority, entries = _write_mixed_root_project(
        tmp_path
    )
    payloads: list[dict[str, object]] = []

    receipt = preflight_validation_project_dependencies(
        tmp_path,
        [command],
        task_authority=task_authority,
        probe_runner=_passing_probe(payloads),
    )

    assert receipt["passed"] is True
    project = receipt["projects"][0]
    assert project["root"] == "ipfs_datasets_py"
    assert project["dependency_contract_schema"] == (
        SCOPED_PROJECT_DEPENDENCY_CONTRACT_SCHEMA_V3
    )
    assert project["scoped_validation_command_sha256"] == entries[1][
        "command_sha256"
    ]
    assert project["selected_validation_extras"] == ["lgcvf-validation"]
    assert payloads[0]["projects"][0]["scoped_validation_extra"] == (
        "lgcvf-validation"
    )
    assert payloads[0]["projects"][0]["requirement_marker_extras"] == [
        "lgcvf-validation",
        "lgcvf-validation",
    ]


def test_v3_board_command_grammar_is_exact() -> None:
    target = "tests/unit/logic/software_verification/test_incremental.py"
    exact = f"cd ipfs_datasets_py && python -m pytest -q {target}"

    assert _command_is_exact_v3_scoped_pytest_target(
        exact,
        relative_root="ipfs_datasets_py",
        target=target,
    )
    for forged in (
        f"cd ipfs_datasets_py && python3 -m pytest -q {target}",
        f"cd ipfs_datasets_py && python -m pytest {target} -q",
        f"cd external/ipfs_datasets && python -m pytest -q {target}",
    ):
        assert not _command_is_exact_v3_scoped_pytest_target(
            forged,
            relative_root="ipfs_datasets_py",
            target=target,
        )


@pytest.mark.parametrize(
    ("case", "reason"),
    [
        ("suffix", "v3_target_declared_output_invalid"),
        ("traversal", "v3_target_declared_output_invalid"),
        ("selected-root", "v3_selected_declared_output_root_mismatch"),
    ],
)
def test_v3_rejects_forged_suffix_or_selected_root(
    tmp_path: Path,
    case: str,
    reason: str,
) -> None:
    project, command, task_authority, entries = _write_mixed_root_project(
        tmp_path
    )
    text = (project / "pyproject.toml").read_text(encoding="utf-8")
    if case == "selected-root":
        original = str(entries[1]["declared_output"])
        forged = f"external/ipfs_datasets/{entries[1]['target']}"
        task_authority["declared_outputs"] = [forged]
    else:
        original = str(entries[0]["declared_output"])
        forged = (
            f"external/ipfs_datasets/evil{entries[0]['target']}"
            if case == "suffix"
            else "external/ipfs_datasets/../ipfs_datasets_py/"
            f"{entries[0]['target']}"
        )
    (project / "pyproject.toml").write_text(
        text.replace(json.dumps(original), json.dumps(forged), 1),
        encoding="utf-8",
    )

    receipt = preflight_validation_project_dependencies(
        tmp_path,
        [command],
        task_authority=task_authority,
        probe_runner=lambda *_args, **_kwargs: pytest.fail(
            "forged output must fail before probing"
        ),
    )

    assert receipt["passed"] is False
    assert receipt["projects"][0]["contract_error_reason"] == reason


def test_v2_retains_legacy_root_and_extra_semantics(tmp_path: Path) -> None:
    _project, command, task_authority, _entries = _write_mixed_root_project(
        tmp_path,
        schema=SCOPED_PROJECT_DEPENDENCY_CONTRACT_SCHEMA_V2,
        extra="test",
    )

    receipt = preflight_validation_project_dependencies(
        tmp_path,
        [command],
        task_authority=task_authority,
        probe_runner=lambda *_args, **_kwargs: pytest.fail(
            "v2 mixed roots must fail before probing"
        ),
    )

    assert receipt["passed"] is False
    assert receipt["projects"][0]["contract_error_reason"] == (
        "v2_target_task_authority_invalid"
    )


def test_v2_still_rejects_non_test_setup_extra(tmp_path: Path) -> None:
    _project, command, task_authority, _entries = _write_mixed_root_project(
        tmp_path,
        schema=SCOPED_PROJECT_DEPENDENCY_CONTRACT_SCHEMA_V2,
    )

    receipt = preflight_validation_project_dependencies(
        tmp_path,
        [command],
        task_authority=task_authority,
    )

    assert receipt["passed"] is False
    assert receipt["projects"][0]["contract_error_reason"] == (
        "v2_setup_authority_unsupported"
    )


@pytest.mark.parametrize(
    "extra",
    ["", "../lgcvf-validation", "lgcvf validation", "$LGCVF", "x" * 129],
)
def test_v3_rejects_unsafe_setup_extra_names(
    tmp_path: Path,
    extra: str,
) -> None:
    _project, command, task_authority, _entries = _write_mixed_root_project(
        tmp_path,
        extra=extra,
    )

    receipt = preflight_validation_project_dependencies(
        tmp_path,
        [command],
        task_authority=task_authority,
    )

    assert receipt["passed"] is False
    assert receipt["projects"][0]["contract_error_reason"] == (
        "v3_setup_authority_extra_invalid"
    )


@pytest.mark.parametrize(
    "setup_source",
    [
        "from setuptools import setup\n"
        "name = 'lgcvf-validation'\n"
        f"setup(extras_require={{name: {REQUIREMENTS!r}}})\n",
        "from setuptools import setup\n"
        f"requirements = {REQUIREMENTS!r}\n"
        "setup(extras_require={'lgcvf-validation': requirements})\n",
        "from setuptools import setup\n"
        "setup(extras_require={'lgcvf-validation': "
        f"{tuple(REQUIREMENTS)!r}}})\n",
        "from setuptools import setup\n"
        f"setup(extras_require={{'test': {REQUIREMENTS!r}}})\n",
    ],
    ids=("dynamic-key", "dynamic-value", "nonliteral-list", "missing-extra"),
)
def test_v3_requires_one_literal_declared_setup_extra(
    tmp_path: Path,
    setup_source: str,
) -> None:
    _project, command, task_authority, _entries = _write_mixed_root_project(
        tmp_path,
        setup_source=setup_source,
    )

    receipt = preflight_validation_project_dependencies(
        tmp_path,
        [command],
        task_authority=task_authority,
    )

    assert receipt["passed"] is False
    assert receipt["projects"][0]["contract_error_reason"] == (
        "v3_setup_authority_not_static"
    )


def _probe_payload(marker_extra: str) -> dict[str, object]:
    return {
        "schema": PROJECT_DEPENDENCY_PROBE_SCHEMA,
        "projects": [
            {
                "root": "ipfs_datasets_py",
                "project_name_sha256": "b" * 64,
                "pyproject_sha256": "a" * 64,
                "requirements": ["pytest>=9.0.3,<10.0.0"],
                "requirement_marker_extras": [marker_extra],
                "requires_python": ">=3.12",
            }
        ],
    }


def test_v3_project_bound_extra_reaches_sealed_probe(tmp_path: Path) -> None:
    _project, command, task_authority, _entries = _write_mixed_root_project(
        tmp_path
    )

    receipt = preflight_validation_project_dependencies(
        tmp_path,
        [command],
        task_authority=task_authority,
    )

    assert receipt["passed"] is True
    assert receipt["probe"]["passed"] is True


@pytest.mark.parametrize("binding", ["bound", "unbound", "mismatched"])
def test_probe_extra_admission_is_exactly_project_bound(binding: str) -> None:
    payload = _probe_payload("lgcvf-validation")
    project = payload["projects"][0]
    assert isinstance(project, dict)
    if binding != "unbound":
        project["dependency_contract_schema"] = (
            SCOPED_PROJECT_DEPENDENCY_CONTRACT_SCHEMA_V3
        )
        project["scoped_validation_extra"] = (
            "lgcvf-validation" if binding == "bound" else "other-validation"
        )

    receipt = _evaluate_dependency_payload(
        payload,
        version_getter={"pytest": "9.1.1"}.__getitem__,
    )

    assert receipt["passed"] is (binding == "bound")
    if binding != "bound":
        assert receipt["projects"][0]["invalid"] == [
            {
                "kind": "requirement_marker_extras",
                "error_type": "InvalidProbePayload",
            }
        ]
