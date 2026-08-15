"""Fail-closed tests for exact-target setup-extra dependency contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.validation.project_dependency_preflight import (
    PROJECT_DEPENDENCY_PROBE_SCHEMA,
    SCOPED_PROJECT_DEPENDENCY_CONTRACT_SCHEMA,
    SCOPED_PROJECT_DEPENDENCY_CONTRACT_SCHEMA_V2,
    SCOPED_PROJECT_DEPENDENCY_PRIOR_SEED_SCHEMA,
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


def _write_v2_project(
    tmp_path: Path,
    *,
    target_states: tuple[str, ...] = ("present", "present"),
    selected_index: int = 0,
) -> tuple[Path, str, dict[str, object], list[dict[str, object]]]:
    project = tmp_path / "project"
    project.mkdir()
    setup_source = (
        "from setuptools import setup\n"
        f"setup(extras_require={{'test': {SETUP_TEST_EXTRA!r}}})\n"
    )
    setup_payload = setup_source.encode("utf-8")
    (project / "setup.py").write_bytes(setup_payload)

    entries: list[dict[str, object]] = []
    encoded_entries: list[str] = []
    for index, state in enumerate(target_states):
        target = f"tests/unit/logic/gui_optimizer/test_v2_{index}.py"
        command = f"cd project && python3 -m pytest {target} -q"
        command_sha256 = hashlib.sha256(command.encode("utf-8")).hexdigest()
        requirements = SETUP_TEST_EXTRA[: index + 1]
        board_namespace = "v2-board"
        task_cid = f"canonical-task-{index}"
        declared_output = f"project/{target}"
        target_payload = f"# target {index}\n".encode()
        baseline_fields = f"state = {json.dumps(state)}"
        if state == "present":
            target_path = project / target
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_bytes(target_payload)
            baseline_fields += (
                ", sha256 = "
                + json.dumps(hashlib.sha256(target_payload).hexdigest())
            )
        entries.append(
            {
                "target": target,
                "command": command,
                "command_sha256": command_sha256,
                "requirements": list(requirements),
                "board_namespace": board_namespace,
                "canonical_task_cid": task_cid,
                "declared_output": declared_output,
                "baseline_state": state,
            }
        )
        encoded_entries.append(
            "{ "
            f"target = {json.dumps(target)}, "
            f"validation-command-sha256 = {json.dumps(command_sha256)}, "
            f"requirements = {json.dumps(requirements)}, "
            "task = { "
            f"board-namespace = {json.dumps(board_namespace)}, "
            f"canonical-task-cid = {json.dumps(task_cid)}, "
            f"declared-output = {json.dumps(declared_output)}"
            " }, "
            f"baseline = {{ {baseline_fields} }}"
            " }"
        )

    joined_entries = ",\n".join(encoded_entries)
    (project / "pyproject.toml").write_text(
        f"""
[project]
name = "scoped-project-v2"
version = "1.0.0"
requires-python = ">=3.12"
dynamic = ["dependencies"]

[tool.ipfs-accelerate-agent-supervisor.project-dependency-preflight]
schema = {json.dumps(SCOPED_PROJECT_DEPENDENCY_CONTRACT_SCHEMA_V2)}
requires-python = ">=3.12"
authority = {{ file = "setup.py", sha256 = "{hashlib.sha256(setup_payload).hexdigest()}", extra = "test", extra-requirements-sha256 = "{_content_sha256(SETUP_TEST_EXTRA)}" }}
targets = [
{joined_entries}
]
""".strip()
        + "\n",
        encoding="utf-8",
    )
    selected = entries[selected_index]
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


def _prior_seed_authority(
    baseline_receipt: dict[str, object],
    task_authority: dict[str, object],
    declared_output: str,
    payload: bytes,
) -> dict[str, object]:
    body: dict[str, object] = {
        "schema": SCOPED_PROJECT_DEPENDENCY_PRIOR_SEED_SCHEMA,
        "board_namespace": task_authority["board_namespace"],
        "canonical_task_cid": task_authority["canonical_task_cid"],
        "baseline_receipt": baseline_receipt,
        "baseline_commit_id": "b" * 40,
        "repository_tree_id": "git-tree:" + "c" * 40,
        "proposal_repository_tree_id": "b" * 40,
        "source_proposal_id": "source-proposal-v2-seed",
        "source_proposal_receipt_id": "source-receipt-v2-seed",
        "proposal_id": "proposal-v2-seed",
        "proposal_receipt_id": "receipt-v2-seed",
        "changed_paths": [declared_output],
        "authorized_paths": [declared_output],
        "seeded_outputs": [
            {
                "path": declared_output,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "git_blob_id": "a" * 40,
            }
        ],
    }
    body["authority_sha256"] = _content_sha256(body)
    return body


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
    ("baseline_states", "selected_index", "expected_manifest_count"),
    [
        (("present", "present"), 1, 2),
        (("declared-output-absent", "present"), 0, 1),
    ],
)
def test_v2_selects_one_exact_task_bound_target(
    tmp_path: Path,
    baseline_states: tuple[str, ...],
    selected_index: int,
    expected_manifest_count: int,
) -> None:
    _project, command, task_authority, entries = _write_v2_project(
        tmp_path,
        target_states=baseline_states,
        selected_index=selected_index,
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
    selected = entries[selected_index]
    assert project["dependency_contract_schema"] == (
        SCOPED_PROJECT_DEPENDENCY_CONTRACT_SCHEMA_V2
    )
    assert project["scoped_validation_target_sha256"] == hashlib.sha256(
        str(selected["target"]).encode("utf-8")
    ).hexdigest()
    assert project["scoped_validation_command_sha256"] == selected[
        "command_sha256"
    ]
    assert project["scoped_validation_target_baseline_state"] == (
        baseline_states[selected_index]
    )
    assert len(project["dependency_manifests"]) == expected_manifest_count
    assert payloads[0]["projects"][0]["requirements"] == selected[
        "requirements"
    ]


def test_v2_authenticated_prior_seed_may_materialize_absent_target(
    tmp_path: Path,
) -> None:
    project, command, task_authority, entries = _write_v2_project(
        tmp_path,
        target_states=("declared-output-absent", "present"),
    )
    baseline_receipt = preflight_validation_project_dependencies(
        tmp_path,
        [command],
        task_authority=task_authority,
        probe_runner=_passing_probe([]),
    )
    assert baseline_receipt["passed"] is True
    assert baseline_receipt["projects"][0][
        "scoped_validation_target_materialization_state"
    ] == "absent"

    target = project / str(entries[0]["target"])
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = b"# replayed accepted prior attempt\n"
    target.write_bytes(payload)
    authority = _prior_seed_authority(
        baseline_receipt,
        task_authority,
        str(entries[0]["declared_output"]),
        payload,
    )
    receipt = preflight_validation_project_dependencies(
        tmp_path,
        [command],
        task_authority=task_authority,
        prior_seed_authority=authority,
        probe_runner=_passing_probe([]),
    )

    assert receipt["passed"] is True
    project_receipt = receipt["projects"][0]
    assert project_receipt[
        "scoped_validation_target_materialization_state"
    ] == "authenticated-prior-seed"
    assert project_receipt[
        "scoped_validation_prior_seed_authority_sha256"
    ] == authority["authority_sha256"]
    assert len(project_receipt["dependency_manifests"]) == 2


@pytest.mark.parametrize(
    "drift",
    [
        "missing_authority",
        "baseline_receipt",
        "task",
        "proposal_path",
        "content",
        "authority_digest",
    ],
)
def test_v2_present_absent_target_requires_exact_prior_seed_attestation(
    tmp_path: Path,
    drift: str,
) -> None:
    project, command, task_authority, entries = _write_v2_project(
        tmp_path,
        target_states=("declared-output-absent", "present"),
    )
    baseline_receipt = preflight_validation_project_dependencies(
        tmp_path,
        [command],
        task_authority=task_authority,
        probe_runner=_passing_probe([]),
    )
    target = project / str(entries[0]["target"])
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = b"# replayed accepted prior attempt\n"
    target.write_bytes(payload)
    authority = _prior_seed_authority(
        baseline_receipt,
        task_authority,
        str(entries[0]["declared_output"]),
        payload,
    )
    if drift == "missing_authority":
        authority = None
    elif drift == "baseline_receipt":
        authority["baseline_receipt"]["passed"] = False
        unsigned = dict(authority)
        unsigned.pop("authority_sha256")
        authority["authority_sha256"] = _content_sha256(unsigned)
    elif drift == "task":
        authority["canonical_task_cid"] = "other-task"
        unsigned = dict(authority)
        unsigned.pop("authority_sha256")
        authority["authority_sha256"] = _content_sha256(unsigned)
    elif drift == "proposal_path":
        authority["changed_paths"] = ["project/other.py"]
        unsigned = dict(authority)
        unsigned.pop("authority_sha256")
        authority["authority_sha256"] = _content_sha256(unsigned)
    elif drift == "content":
        target.write_bytes(b"# tampered after attestation\n")
    else:
        authority["authority_sha256"] = "f" * 64

    receipt = preflight_validation_project_dependencies(
        tmp_path,
        [command],
        task_authority=task_authority,
        prior_seed_authority=authority,
        probe_runner=lambda *_args, **_kwargs: pytest.fail(
            "unauthenticated seed must fail before probing"
        ),
    )

    assert receipt["passed"] is False
    assert receipt["projects"][0]["reason"] == "dynamic_dependencies_unresolved"


@pytest.mark.parametrize(
    "unsafe_target",
    [
        "tests/unit/$HOME.py",
        "tests/unit/`id`.py",
        "tests/unit/$(id).py",
        "tests/unit/foo;touch.py",
        "tests/unit/foo&&touch.py",
        "tests/unit/foo|touch.py",
        "tests/unit/foo>out.py",
        "tests/unit/foo space.py",
        "tests/unit/café.py",
    ],
)
def test_v2_rejects_targets_outside_conservative_shell_safe_charset(
    tmp_path: Path,
    unsafe_target: str,
) -> None:
    project, command, task_authority, entries = _write_v2_project(
        tmp_path,
        target_states=("declared-output-absent", "present"),
    )
    text = (project / "pyproject.toml").read_text(encoding="utf-8")
    text = text.replace(json.dumps(entries[0]["target"]), json.dumps(unsafe_target), 1)
    (project / "pyproject.toml").write_text(text, encoding="utf-8")

    receipt = preflight_validation_project_dependencies(
        tmp_path,
        [command],
        task_authority=task_authority,
        probe_runner=lambda *_args, **_kwargs: pytest.fail(
            "unsafe target must fail before probing"
        ),
    )

    assert receipt["passed"] is False
    assert receipt["projects"][0]["contract_error_reason"] == (
        "v2_target_path_invalid"
    )


@pytest.mark.parametrize(
    "case",
    [
        "missing_runtime_authority",
        "task_cid_drift",
        "output_not_declared",
        "command_digest_drift",
        "command_target_drift",
        "present_digest_drift",
        "absent_target_appeared",
        "unknown_target_field",
        "unknown_task_field",
        "unknown_baseline_field",
        "duplicate_target",
        "duplicate_command_digest",
        "coerced_target",
        "unsafe_target",
    ],
)
def test_v2_target_contract_drift_fails_closed(
    tmp_path: Path,
    case: str,
) -> None:
    project, command, task_authority, entries = _write_v2_project(
        tmp_path,
        target_states=("declared-output-absent", "present"),
    )
    if case == "missing_runtime_authority":
        task_authority = None
    elif case == "task_cid_drift":
        task_authority["canonical_task_cid"] = "other-task"
    elif case == "output_not_declared":
        task_authority["declared_outputs"] = ["project/result.py"]
    elif case == "command_digest_drift":
        text = (project / "pyproject.toml").read_text(encoding="utf-8")
        (project / "pyproject.toml").write_text(
            text.replace(str(entries[0]["command_sha256"]), "a" * 64, 1),
            encoding="utf-8",
        )
    elif case == "command_target_drift":
        command = command.replace("test_v2_0.py", "test_v2_1.py")
    elif case == "present_digest_drift":
        command = str(entries[1]["command"])
        task_authority = {
            "board_namespace": entries[1]["board_namespace"],
            "canonical_task_cid": entries[1]["canonical_task_cid"],
            "declared_outputs": [entries[1]["declared_output"]],
        }
        target = project / str(entries[1]["target"])
        target.write_text("# drift\n", encoding="utf-8")
    elif case == "absent_target_appeared":
        target = project / str(entries[0]["target"])
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("# appeared\n", encoding="utf-8")
    elif case == "unknown_target_field":
        text = (project / "pyproject.toml").read_text(encoding="utf-8")
        (project / "pyproject.toml").write_text(
            text.replace("target = ", "unknown = true, target = ", 1),
            encoding="utf-8",
        )
    elif case == "unknown_task_field":
        text = (project / "pyproject.toml").read_text(encoding="utf-8")
        (project / "pyproject.toml").write_text(
            text.replace(
                "task = { ",
                "task = { unknown = true, ",
                1,
            ),
            encoding="utf-8",
        )
    elif case == "unknown_baseline_field":
        text = (project / "pyproject.toml").read_text(encoding="utf-8")
        (project / "pyproject.toml").write_text(
            text.replace(
                'baseline = { state = "declared-output-absent" }',
                'baseline = { state = "declared-output-absent", sha256 = "'
                + "a" * 64
                + '" }',
                1,
            ),
            encoding="utf-8",
        )
    elif case in {"duplicate_target", "duplicate_command_digest"}:
        text = (project / "pyproject.toml").read_text(encoding="utf-8")
        if case == "duplicate_target":
            text = text.replace(
                str(entries[1]["target"]), str(entries[0]["target"]), 1
            )
        else:
            text = text.replace(
                str(entries[1]["command_sha256"]),
                str(entries[0]["command_sha256"]),
                1,
            )
        (project / "pyproject.toml").write_text(text, encoding="utf-8")
    elif case in {"coerced_target", "unsafe_target"}:
        text = (project / "pyproject.toml").read_text(encoding="utf-8")
        replacement = "1" if case == "coerced_target" else '"tests/../escape.py"'
        text = text.replace(json.dumps(entries[0]["target"]), replacement, 1)
        (project / "pyproject.toml").write_text(text, encoding="utf-8")

    receipt = preflight_validation_project_dependencies(
        tmp_path,
        [command],
        task_authority=task_authority,
        probe_runner=lambda *_args, **_kwargs: pytest.fail(
            "invalid v2 contracts must fail before probing"
        ),
    )

    assert receipt["passed"] is False
    assert receipt["projects"][0]["reason"] == "dynamic_dependencies_unresolved"
    expected_reasons = {
        "missing_runtime_authority": "v2_runtime_task_authority_fields_invalid",
        "task_cid_drift": "v2_validation_task_authority_mismatch",
        "output_not_declared": "v2_validation_task_authority_mismatch",
        "command_digest_drift": "v2_validation_command_not_declared",
        "command_target_drift": "v2_validation_task_authority_mismatch",
        "present_digest_drift": "v2_present_target_digest_mismatch",
        "absent_target_appeared": "v2_absent_target_is_present",
        "unknown_target_field": "v2_target_fields_invalid",
        "unknown_task_field": "v2_target_task_fields_invalid",
        "unknown_baseline_field": "v2_absent_baseline_fields_invalid",
        "duplicate_target": "v2_targets_duplicate_or_oversized",
        "duplicate_command_digest": (
            "v2_validation_command_digest_invalid_or_duplicate"
        ),
        "coerced_target": "v2_target_type_invalid",
        "unsafe_target": "v2_target_path_invalid",
    }
    assert receipt["projects"][0]["contract_error_reason"] == (
        expected_reasons[case]
    )


def test_v2_failure_retries_distinguish_command_target_and_task_drift(
    tmp_path: Path,
) -> None:
    fingerprints: set[str] = set()
    reasons: set[str] = set()
    for case in ("command", "target", "task"):
        fixture_root = tmp_path / case
        fixture_root.mkdir()
        project, command, task_authority, entries = _write_v2_project(
            fixture_root,
            target_states=("declared-output-absent", "present"),
        )
        if case == "command":
            command += " --maxfail=1"
        elif case == "target":
            target = project / str(entries[0]["target"])
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("# appeared\n", encoding="utf-8")
        else:
            task_authority["canonical_task_cid"] = "other-task"

        receipt = preflight_validation_project_dependencies(
            fixture_root,
            [command],
            task_authority=task_authority,
            probe_runner=lambda *_args, **_kwargs: pytest.fail(
                "contract drift must fail before probing"
            ),
        )

        assert receipt["passed"] is False
        fingerprints.add(str(receipt["retry_fingerprint"]))
        reasons.add(str(receipt["projects"][0]["contract_error_reason"]))

    assert len(fingerprints) == 3
    assert reasons == {
        "v2_validation_command_not_declared",
        "v2_absent_target_is_present",
        "v2_validation_task_authority_mismatch",
    }


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
