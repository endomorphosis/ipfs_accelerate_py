"""Focused fail-closed tests for append-only LGCVF successor resolution."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    content_identity,
)
from ipfs_accelerate_py.agent_supervisor.validation.lgcvf_successor_resolution import (
    AUTHORITY_VALIDATION_SCHEMA,
    DERIVED_STATES,
    EXPECTED_DISPOSITIONS,
    EXPECTED_TASK_IDS,
    LgcvfSuccessorResolutionError,
    build_successor_resolution,
    validate_successor_resolution,
)
from scripts.validate_lgcvf_successor_resolution import (
    ResolutionCliError,
    _write_once,
)


def _cid(label: str) -> str:
    return content_identity({"fixture": label})


def _seal(value: dict[str, Any], field: str) -> dict[str, Any]:
    value.pop(field, None)
    value[field] = content_identity(value)
    return value


def _authority_validator(
    *,
    task_id: str,
    disposition: str,
    receipt: dict[str, Any],
    context: dict[str, Any],
) -> dict[str, Any]:
    verdict: dict[str, Any] = {
        "schema": AUTHORITY_VALIDATION_SCHEMA,
        "valid": receipt.get("test_signature") == "valid",
        "signed": receipt.get("test_signature") == "valid",
        "task_id": task_id,
        "disposition": disposition,
        "receipt_cid": receipt["receipt_cid"],
        "context_cid": context["context_cid"],
        "release_qualified": False,
        "production_authorized": False,
    }
    return _seal(verdict, "validation_cid")


def _fixture() -> dict[str, Any]:
    plan_cid = _cid("plan")
    suite = {
        "suite_id": "fixed_datasets_semantics",
        "manifest": {
            "owner_root": "ipfs_datasets_py",
            "paths": ["tests/unit/logic/test_compositional_verification_public_api.py"],
        },
        "passed": True,
        "exit_code": 0,
        "passed_count": 17,
        "failed_count": 0,
        "skipped_count": 0,
        "xfailed_count": 0,
        "xpassed_count": 0,
        "error_count": 0,
        "nodeids_cid": _cid("datasets-nodeids"),
        "observation_cid": _cid("datasets-observation"),
    }
    qualification = _seal(
        {
            "schema": "lgcvf-independent-hermetic-qualification@1",
            "plan_cid": plan_cid,
            "passed": True,
            "test_qualification_complete": True,
            "task_implementation_complete": False,
            "objective_complete": False,
            "release_qualified": False,
            "production_authorized": False,
            "suites": [suite],
        },
        "result_cid",
    )
    benchmark = _seal(
        {
            "schema": "lgcvf-symbolic-displacement-benchmark@1",
            "overall_disposition": "partial",
            "production_authoritative": False,
            "release_qualified": False,
            "production_authorized": False,
        },
        "report_cid",
    )
    task_inputs = (
        ("LGCVF-S001", "blocked_external_authority", [], "ipfs_accelerate_py", []),
        (
            "LGCVF-S002",
            "blocked_manual",
            ["LGCVF-S001"],
            "ipfs_accelerate_py",
            [],
        ),
        (
            "LGCVF-S003",
            "todo",
            ["LGCVF-S001"],
            "ipfs_datasets_py",
            [
                (
                    "pytest ipfs_datasets_py/tests/unit/logic/"
                    "test_compositional_verification_public_api.py"
                )
            ],
        ),
    )
    tasks = []
    for task_id, status, dependencies, owner, commands in task_inputs:
        tasks.append(
            _seal(
                {
                    "task_id": task_id,
                    "status": status,
                    "depends_on": dependencies,
                    "owning_repository": owner,
                    "validation": commands,
                },
                "task_cid",
            )
        )
    predecessor = _seal(
        {
            "schema": "lgcvf-successor-tasks@1",
            "plan_cid": plan_cid,
            "qualification_cid": qualification["result_cid"],
            "benchmark_cid": benchmark["report_cid"],
            "objective_complete": False,
            "release_qualified": False,
            "production_authorized": False,
            "tasks": tasks,
        },
        "successor_tasks_cid",
    )
    roots = {
        "ipfs_accelerate_py": {"head": "1" * 40, "tree": "2" * 40},
        "ipfs_datasets_py": {
            "head": "3" * 40,
            "tree": "4" * 40,
            "gitlink": "3" * 40,
        },
    }
    receipts = {}
    for task_id in ("LGCVF-S001", "LGCVF-S002"):
        receipts[task_id] = _seal(
            {
                "schema": "lgcvf-test-signed-receipt@1",
                "task_id": task_id,
                "disposition": EXPECTED_DISPOSITIONS[task_id],
                "test_signature": "valid",
                "release_qualified": False,
                "production_authorized": False,
            },
            "receipt_cid",
        )
    return {
        "predecessor": predecessor,
        "qualification": qualification,
        "benchmark": benchmark,
        "source_roots": roots,
        "authority_receipts": receipts,
        "authority_validator": _authority_validator,
    }


def _build(inputs: dict[str, Any]) -> dict[str, Any]:
    return build_successor_resolution(**inputs)


def _revalidate(resolution: dict[str, Any], inputs: dict[str, Any]) -> dict[str, Any]:
    return validate_successor_resolution(
        resolution,
        predecessor=inputs["predecessor"],
        qualification=inputs["qualification"],
        benchmark=inputs["benchmark"],
        expected_source_roots=inputs["source_roots"],
        authority_receipts=inputs["authority_receipts"],
        authority_validator=inputs["authority_validator"],
    )


def test_builds_exact_terminal_dispositions_without_raising_authority() -> None:
    inputs = _fixture()
    resolution = _build(inputs)

    assert [task["task_id"] for task in resolution["tasks"]] == list(EXPECTED_TASK_IDS)
    assert [task["disposition"] for task in resolution["tasks"]] == [
        "self_verified_r_and_d",
        "production_declined_r_and_d",
        "completed",
    ]
    assert resolution["tasks"][1]["depends_on"] == ["LGCVF-S001"]
    assert resolution["derived_states"] == DERIVED_STATES
    assert resolution["tasks"][2]["evidence"]["datasets_commit"] == "3" * 40
    assert resolution["tasks"][2]["evidence"]["datasets_tree"] == "4" * 40
    assert resolution["tasks"][2]["evidence"]["validations"][0]["command"].startswith(
        "pytest ipfs_datasets_py/"
    )

    validation = _revalidate(resolution, inputs)
    assert validation == {
        "schema": "lgcvf-successor-resolution-validation@1",
        "valid": True,
        "resolution_cid": resolution["resolution_cid"],
        "predecessor_successor_tasks_cid": inputs["predecessor"]["successor_tasks_cid"],
        "resolved_task_ids": list(EXPECTED_TASK_IDS),
        **DERIVED_STATES,
    }


def test_rejects_resealed_dependency_and_derived_state_mutations() -> None:
    inputs = _fixture()
    resolution = _build(inputs)
    dependency_mutation = copy.deepcopy(resolution)
    dependency_mutation["tasks"][1]["depends_on"] = []
    _seal(dependency_mutation["tasks"][1], "task_resolution_cid")
    _seal(dependency_mutation, "resolution_cid")
    with pytest.raises(LgcvfSuccessorResolutionError, match="dependency differs"):
        _revalidate(dependency_mutation, inputs)

    state_mutation = copy.deepcopy(resolution)
    state_mutation["derived_states"]["objective_complete"] = True
    _seal(state_mutation, "resolution_cid")
    with pytest.raises(LgcvfSuccessorResolutionError, match="derived states differ"):
        _revalidate(state_mutation, inputs)


def test_rejects_stale_original_task_cid_and_source_roots() -> None:
    inputs = _fixture()
    resolution = _build(inputs)
    stale = copy.deepcopy(inputs)
    stale["predecessor"]["tasks"][2]["validation"] = [
        "pytest ipfs_datasets_py/tests/unit/logic/test_other.py"
    ]
    _seal(stale["predecessor"]["tasks"][2], "task_cid")
    _seal(stale["predecessor"], "successor_tasks_cid")
    with pytest.raises(LgcvfSuccessorResolutionError, match="root binding differs"):
        _revalidate(resolution, stale)

    mismatched_roots = copy.deepcopy(inputs)
    mismatched_roots["source_roots"]["ipfs_datasets_py"]["head"] = "5" * 40
    with pytest.raises(LgcvfSuccessorResolutionError, match="gitlink"):
        _revalidate(resolution, mismatched_roots)


def test_rejects_unsigned_or_context_free_authority_verdict() -> None:
    inputs = _fixture()
    inputs["authority_receipts"]["LGCVF-S001"]["test_signature"] = "invalid"
    _seal(inputs["authority_receipts"]["LGCVF-S001"], "receipt_cid")
    with pytest.raises(LgcvfSuccessorResolutionError, match="does not admit"):
        _build(inputs)

    inputs = _fixture()

    def context_free(**arguments: Any) -> dict[str, Any]:
        verdict = _authority_validator(**arguments)
        verdict["context_cid"] = _cid("foreign-context")
        return _seal(verdict, "validation_cid")

    inputs["authority_validator"] = context_free
    with pytest.raises(LgcvfSuccessorResolutionError, match="does not admit"):
        _build(inputs)

    inputs = _fixture()

    def open_schema(**arguments: Any) -> dict[str, Any]:
        verdict = _authority_validator(**arguments)
        verdict["unbound_extension"] = True
        return _seal(verdict, "validation_cid")

    inputs["authority_validator"] = open_schema
    with pytest.raises(LgcvfSuccessorResolutionError, match="verdict fields differ"):
        _build(inputs)


def test_rejects_s003_without_exact_protected_validation() -> None:
    inputs = _fixture()
    inputs["predecessor"]["tasks"][2]["validation"] = [
        "pytest ipfs_datasets_py/tests/unit/logic/unprotected.py"
    ]
    _seal(inputs["predecessor"]["tasks"][2], "task_cid")
    _seal(inputs["predecessor"], "successor_tasks_cid")
    with pytest.raises(LgcvfSuccessorResolutionError, match="uniquely present"):
        _build(inputs)

    inputs = _fixture()
    inputs["qualification"]["suites"][0]["failed_count"] = 1
    _seal(inputs["qualification"], "result_cid")
    inputs["predecessor"]["qualification_cid"] = inputs["qualification"]["result_cid"]
    _seal(inputs["predecessor"], "successor_tasks_cid")
    with pytest.raises(LgcvfSuccessorResolutionError, match="not an exact pass"):
        _build(inputs)


def test_emit_is_exclusive_and_never_replaces_existing_artifact(tmp_path: Path) -> None:
    path = tmp_path / "successor_resolution.json"
    first = {"schema": "fixture@1", "resolution_cid": _cid("first")}
    _write_once(path, first)
    original = path.read_bytes()

    with pytest.raises(ResolutionCliError, match="will not be replaced"):
        _write_once(path, {"schema": "fixture@1", "resolution_cid": _cid("second")})

    assert path.read_bytes() == original
    assert json.loads(original) == first
