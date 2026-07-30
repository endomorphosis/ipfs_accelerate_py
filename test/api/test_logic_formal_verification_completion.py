"""Completion receipt contract for the logic formal verification expansion.

LogicFormalVerificationRelease@1 — program completion surface for LFV-G083 / LFV-041.

Validates that
`docs/architecture/logic_formal_verification_expansion_completion_receipt.json`
binds all 41 executable child goals, records zero authority-boundary
violations, and that release artifacts (rollout policy, prover matrix,
benchmark suite) are present and consistent with the objective heap.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
RECEIPT_PATH = (
    REPO_ROOT
    / "docs"
    / "architecture"
    / "logic_formal_verification_expansion_completion_receipt.json"
)
OBJECTIVES_PATH = (
    REPO_ROOT
    / "docs"
    / "architecture"
    / "logic_formal_verification_expansion.objectives.md"
)
ROLLOUT_PATH = (
    REPO_ROOT
    / "ipfs_datasets_py"
    / "docs"
    / "logic"
    / "software_verification_rollout.md"
)
PROVER_MATRIX_PATH = (
    REPO_ROOT
    / "ipfs_datasets_py"
    / "docs"
    / "security_verification"
    / "prover_matrix.md"
)
MATRIX_TEST_PATH = (
    REPO_ROOT
    / "ipfs_datasets_py"
    / "tests"
    / "integration"
    / "benchmarks"
    / "logic_pipeline"
    / "test_software_verification_matrix.py"
)
CAPABILITY_MATRIX_PATH = (
    REPO_ROOT
    / "ipfs_datasets_py"
    / "tests"
    / "fixtures"
    / "logic"
    / "software_verification"
    / "capability_matrix.json"
)

INTERFACE = "LogicFormalVerificationRelease@1"
SCHEMA_VERSION = "logic-formal-verification-expansion-completion-receipt/v1"
EXPECTED_CHILD_COUNT = 41
HARD_ZERO_GATES = (
    "authority_boundary_violations",
    "false_proof_count",
    "false_completion_count",
    "secret_or_witness_leakage_count",
    "unresolved_cross_provider_disagreement_count",
)


def _load_receipt() -> dict[str, Any]:
    assert RECEIPT_PATH.is_file(), f"missing completion receipt: {RECEIPT_PATH}"
    payload = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _parse_objective_child_ids() -> list[str]:
    text = OBJECTIVES_PATH.read_text(encoding="utf-8")
    ids = re.findall(r"^## (LFV-G\d+) ", text, flags=re.MULTILINE)
    assert ids and ids[0] == "LFV-G000"
    children = [goal_id for goal_id in ids if goal_id != "LFV-G000"]
    assert len(children) == EXPECTED_CHILD_COUNT
    return children


def _sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def test_completion_receipt_schema_and_interface() -> None:
    receipt = _load_receipt()
    assert receipt["schema_version"] == SCHEMA_VERSION
    assert receipt["interface"] == INTERFACE
    assert receipt["program_goal_id"] == "LFV-G000"
    assert receipt["release_goal_id"] == "LFV-G083"
    assert receipt["task_id"] == "LFV-041"
    assert "receipt_identity" in receipt
    assert str(receipt["receipt_identity"]).startswith("sha256:")
    source = receipt["source"]
    assert source["binding_mode"] == "current_tree_content_identity"
    assert re.fullmatch(r"[0-9a-f]{40}", source["parent_commit"])
    assert re.fullmatch(r"[0-9a-f]{40}", source["datasets_commit"])


def test_receipt_binds_all_forty_one_child_goals() -> None:
    receipt = _load_receipt()
    expected_ids = _parse_objective_child_ids()
    children = receipt["child_goals"]
    assert isinstance(children, list)
    assert len(children) == EXPECTED_CHILD_COUNT

    observed_ids = [child["goal_id"] for child in children]
    assert observed_ids == expected_ids
    assert "LFV-G083" in observed_ids
    assert "LFV-G005" in observed_ids
    assert "LFV-G000" not in observed_ids

    for child in children:
        assert child["bound"] is True
        assert child["goal_id"].startswith("LFV-G")
        assert isinstance(child["title"], str) and child["title"].strip()
        assert isinstance(child["evidence"], list) and child["evidence"]
        assert isinstance(child["outputs"], list) and child["outputs"]
        assert isinstance(child.get("interfaces"), list)


def test_zero_authority_boundary_violations() -> None:
    receipt = _load_receipt()
    acceptance = receipt["acceptance"]
    assert acceptance["authority_boundary_violations"] == 0
    assert acceptance["child_goal_count"] == EXPECTED_CHILD_COUNT
    assert acceptance["matrix_from_executable_evidence"] is True
    assert acceptance["benchmarks_without_timing_ratio_correctness_gates"] is True
    assert acceptance["rollout_property_specific_and_reversible"] is True

    hard_zero = receipt["hard_zero_gates"]
    for gate in HARD_ZERO_GATES:
        assert hard_zero[gate] == 0, gate


def test_release_artifacts_are_bound_and_present() -> None:
    receipt = _load_receipt()
    artifacts = receipt["artifacts"]
    required = {
        "rollout_policy": ROLLOUT_PATH,
        "prover_matrix": PROVER_MATRIX_PATH,
        "matrix_benchmark": MATRIX_TEST_PATH,
        "completion_test": Path(__file__),
        "capability_matrix": CAPABILITY_MATRIX_PATH,
        "objectives_heap": OBJECTIVES_PATH,
    }
    for key, path in required.items():
        entry = artifacts[key]
        assert entry["path"]
        assert entry["present"] is True, key
        assert path.is_file(), path
        assert entry["content_identity"] == _sha256_bytes(path.read_bytes()), key

    # Self path must match this test file relative to repo root.
    completion_rel = artifacts["completion_test"]["path"]
    assert completion_rel.endswith(
        "test/api/test_logic_formal_verification_completion.py"
    )


def test_rollout_and_benchmark_policy_surfaces() -> None:
    receipt = _load_receipt()
    rollout = receipt["rollout_policy"]
    assert rollout["stages"] == ["declared", "shadow", "canary", "enforced"]
    assert rollout["global_provider_switch"] is False
    assert rollout["reversible"] is True
    assert ROLLOUT_PATH.is_file()
    rollout_text = ROLLOUT_PATH.read_text(encoding="utf-8")
    assert INTERFACE in rollout_text
    assert "authority_boundary_violations" in rollout_text

    report = receipt["benchmark_report_summary"]
    assert report["timing_ratio_correctness_gates"] is False
    assert report["external_tools_fabricated"] is False
    assert "unavailable" in report["semantic_outcome_classes"]
    assert MATRIX_TEST_PATH.is_file()
    matrix_test = MATRIX_TEST_PATH.read_text(encoding="utf-8")
    assert "timing_ratio_correctness_gates" in matrix_test
    assert "LogicFormalVerificationRelease@1" in matrix_test

    policy = receipt["external_tool_identity_policy"]
    assert policy["fabricated_availability_forbidden"] is True
    assert policy["documentation_is_not_runtime_authority"] is True
    assert PROVER_MATRIX_PATH.is_file()


def test_g000_and_g083_objective_heap_point_at_this_receipt() -> None:
    text = OBJECTIVES_PATH.read_text(encoding="utf-8")
    assert "logic_formal_verification_expansion_completion_receipt.json" in text

    g000 = re.search(
        r"^## LFV-G000 .+?(?=^## |\Z)",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )
    g083 = re.search(
        r"^## LFV-G083 .+?(?=^## |\Z)",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )
    assert g000 is not None and g083 is not None
    assert "docs/architecture/logic_formal_verification_expansion_completion_receipt.json" in g000.group(0)
    assert "ipfs_datasets_py/docs/logic/software_verification_rollout.md" in g083.group(0)
    assert "test_software_verification_matrix.py" in g083.group(0)
    assert "test_logic_formal_verification_completion.py" in g083.group(0)


def test_receipt_identity_is_content_addressed() -> None:
    raw = RECEIPT_PATH.read_text(encoding="utf-8")
    receipt = json.loads(raw)
    stored = receipt.pop("receipt_identity")
    body = json.dumps(
        receipt,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    recomputed = _sha256_bytes(body.encode("utf-8"))
    assert stored == recomputed


def test_release_goal_child_binding_includes_release_evidence_terms() -> None:
    receipt = _load_receipt()
    release = next(
        child for child in receipt["child_goals"] if child["goal_id"] == "LFV-G083"
    )
    evidence = set(release["evidence"])
    assert "ipfs_datasets_py/docs/logic/software_verification_rollout.md" in evidence
    assert (
        "ipfs_datasets_py/tests/integration/benchmarks/logic_pipeline/"
        "test_software_verification_matrix.py"
    ) in evidence
    assert "test/api/test_logic_formal_verification_completion.py" in evidence
    for path in release["evidence"]:
        assert (REPO_ROOT / path).is_file(), path
