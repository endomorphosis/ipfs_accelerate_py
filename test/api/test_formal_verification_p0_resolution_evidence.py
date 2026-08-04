"""Fail-closed reconciliation tests for readiness-baseline P0 findings.

Resolved is a derived state: the baseline must bind reachable implementation
commits, current implementation bytes, and current validation-test bytes.  A
status-string edit alone cannot remove hard-zero pressure.
"""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
BUILDER_PATH = REPO_ROOT / "tools" / "logic" / "build_formal_verification_tactician_receipt.py"
BASELINE_PATH = REPO_ROOT / "docs" / "architecture" / "formal_verification_readiness_baseline.json"
P0_IDS = {
    "receipt_verification_fail_open",
    "public_counterexample_raw_leak",
    "structural_repair_as_closure",
}


@pytest.fixture(scope="module")
def builder() -> ModuleType:
    spec = importlib.util.spec_from_file_location("p0_resolution_builder", BUILDER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def baseline() -> dict[str, Any]:
    return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))


def _finding(baseline: dict[str, Any], finding_id: str) -> dict[str, Any]:
    return next(finding for finding in baseline["known_findings"] if finding["id"] == finding_id)


def test_checked_in_p0_resolutions_bind_commits_artifacts_and_tests(
    builder: ModuleType,
    baseline: dict[str, Any],
) -> None:
    pressure = builder._baseline_p0_gate_pressure(
        baseline,
        repo_root=REPO_ROOT,
    )

    assert pressure["failures"] == []
    assert pressure["open_p0_findings"] == []
    assert pressure["invalid_p0_resolutions"] == []
    assert set(pressure["gate_pressure"].values()) == {0}
    resolved = {
        finding["id"]: finding["resolution_evidence"]
        for finding in pressure["resolved_p0_findings"]
    }
    assert set(resolved) == P0_IDS
    for evidence in resolved.values():
        assert evidence["valid"] is True
        assert evidence["implementation_commits"]
        assert evidence["implementation_artifacts"]
        assert evidence["validation_tests"]
        for entry in evidence["implementation_artifacts"] + evidence["validation_tests"]:
            assert entry["content_identity"].startswith("sha256:")
            assert len(entry["implementation_commit"]) == 40


def test_resolved_status_without_evidence_fails_closed(
    builder: ModuleType,
) -> None:
    pressure = builder._baseline_p0_gate_pressure(
        {
            "known_findings": [
                {
                    "id": "receipt_verification_fail_open",
                    "severity": "p0",
                    "status": "resolved",
                }
            ]
        },
        repo_root=REPO_ROOT,
    )

    assert pressure["resolved_p0_findings"] == []
    assert len(pressure["invalid_p0_resolutions"]) == 1
    failures = pressure["invalid_p0_resolutions"][0]["resolution_evidence"]["failures"]
    assert "resolution_evidence_missing_or_invalid" in failures
    assert pressure["gate_pressure"]["false_proof_count"] > 0
    assert pressure["gate_pressure"]["false_closure_count"] > 0
    assert pressure["gate_pressure"]["authority_boundary_violations"] > 0


@pytest.mark.parametrize(
    ("mutation", "expected_failure"),
    [
        ("content_identity", "content_identity_mismatch"),
        ("implementation_commit", "bound_commit_missing_or_invalid"),
        ("commit_blob", "commit_blob_identity_mismatch"),
        ("validation_test", "required_evidence_missing"),
    ],
)
def test_tampered_resolution_evidence_restores_gate_pressure(
    builder: ModuleType,
    baseline: dict[str, Any],
    mutation: str,
    expected_failure: str,
) -> None:
    finding = _finding(baseline, "public_counterexample_raw_leak")
    resolution = finding["resolution_evidence"]
    if mutation == "content_identity":
        resolution["implementation_artifacts"][0]["content_identity"] = "sha256:" + ("0" * 64)
    elif mutation == "implementation_commit":
        resolution["implementation_commits"] = [
            commit
            for commit in resolution["implementation_commits"]
            if commit["repository"] != "ipfs_datasets_py"
        ]
    elif mutation == "commit_blob":
        resolution["implementation_artifacts"][0]["commit"] = (
            "194f9cbcc64bb07cf4849485a7665e9285ee71c9"
        )
    else:
        resolution["validation_tests"] = [
            test for test in resolution["validation_tests"] if test["repository"] != "root"
        ]

    pressure = builder._baseline_p0_gate_pressure(
        baseline,
        repo_root=REPO_ROOT,
    )

    invalid = {
        item["id"]: item["resolution_evidence"] for item in pressure["invalid_p0_resolutions"]
    }
    assert set(invalid) == {"public_counterexample_raw_leak"}
    assert any(
        expected_failure in failure
        for failure in invalid["public_counterexample_raw_leak"]["failures"]
    )
    assert pressure["gate_pressure"]["secret_or_witness_leakage_count"] > 0


def test_resolved_unknown_p0_has_no_implicit_clearance_policy(
    builder: ModuleType,
    baseline: dict[str, Any],
) -> None:
    forged = copy.deepcopy(_finding(baseline, "receipt_verification_fail_open"))
    forged["id"] = "new_unclassified_p0"
    pressure = builder._baseline_p0_gate_pressure(
        {"known_findings": [forged]},
        repo_root=REPO_ROOT,
    )

    evidence = pressure["invalid_p0_resolutions"][0]["resolution_evidence"]
    assert "finding_resolution_policy_missing" in evidence["failures"]
    assert all(value > 0 for value in pressure["gate_pressure"].values())


def test_resolved_p0s_do_not_fabricate_benchmark_authority(
    builder: ModuleType,
    baseline: dict[str, Any],
) -> None:
    hard_zero = builder.derive_hard_zero_gates(
        certificate=None,
        benchmark=None,
        baseline=baseline,
        repo_root=REPO_ROOT,
    )

    derivation = hard_zero["derivation"]
    assert derivation["complete"] is False
    assert len(derivation["resolved_p0_findings"]) == 3
    assert derivation["open_p0_findings"] == []
    assert derivation["invalid_p0_resolutions"] == []
    assert "benchmark" in derivation["missing_measurements"]
    assert hard_zero["false_proof_count"] > 0
    assert hard_zero["false_closure_count"] > 0
    assert hard_zero["secret_or_witness_leakage_count"] > 0
    assert hard_zero["authority_boundary_violations"] > 0
