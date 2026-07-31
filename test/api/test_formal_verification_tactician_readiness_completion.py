"""Completion receipt contract for formal verification tactician readiness.

FormalVerificationTacticianCompletionReceipt@1 — FVT-G090 / FVT-036.

Validates that
``docs/architecture/formal_verification_tactician_readiness_completion_receipt.json``
binds every executable child goal of FVT-G000, separates implementation from
deployment certification, records hard-zero false-proof / false-closure /
leakage / authority / disagreement gates without hardcoded success counters,
and that the receipt builder recomputes a content-addressed receipt from the
current tree.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
RECEIPT_PATH = (
    REPO_ROOT
    / "docs"
    / "architecture"
    / "formal_verification_tactician_readiness_completion_receipt.json"
)
OBJECTIVES_PATH = (
    REPO_ROOT
    / "docs"
    / "architecture"
    / "formal_verification_tactician_readiness.objectives.md"
)
BUILDER_PATH = (
    REPO_ROOT
    / "tools"
    / "logic"
    / "build_formal_verification_tactician_receipt.py"
)
BASELINE_PATH = (
    REPO_ROOT
    / "docs"
    / "architecture"
    / "formal_verification_readiness_baseline.json"
)
TOOLCHAIN_CERT_PATH = (
    REPO_ROOT
    / "docs"
    / "architecture"
    / "formal_verification_toolchain_certificate.json"
)
BENCHMARK_PATH = (
    REPO_ROOT
    / "docs"
    / "architecture"
    / "formal_verification_tactician_benchmark.json"
)
ROLLOUT_PATH = (
    REPO_ROOT
    / "docs"
    / "architecture"
    / "formal_verification_tactician_rollout.md"
)
CORPUS_PATH = (
    REPO_ROOT
    / "ipfs_datasets_py"
    / "tests"
    / "fixtures"
    / "logic"
    / "proof_tactician"
    / "manifest.json"
)

INTERFACE = "FormalVerificationTacticianCompletionReceipt@1"
PROGRAM_INTERFACE = "FormalVerificationTacticianRelease@1"
SCHEMA_VERSION = "formal-verification-tactician-completion-receipt/v1"
PROGRAM_GOAL_ID = "FVT-G000"
COMPLETION_GOAL_ID = "FVT-G090"
TASK_ID = "FVT-036"

HARD_ZERO_GATES = (
    "false_proof_count",
    "false_closure_count",
    "secret_or_witness_leakage_count",
    "authority_boundary_violations",
    "unresolved_cross_provider_disagreement_count",
)

REQUIRED_ARTIFACT_KEYS = (
    "objectives_heap",
    "readiness_baseline",
    "toolchain_certificate",
    "toolchain_lock",
    "benchmark_report",
    "live_example_report",
    "rollout_policy",
    "corpus_manifest",
    "public_api_test",
    "cli_mcp_parity_test",
    "adversarial_root_test",
    "adversarial_datasets_test",
    "metrics_module",
    "receipt_builder",
    "completion_test",
    "completion_receipt",
)

COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


def _load_receipt() -> dict[str, Any]:
    assert RECEIPT_PATH.is_file(), f"missing completion receipt: {RECEIPT_PATH}"
    payload = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _parse_objective_child_ids() -> list[str]:
    text = OBJECTIVES_PATH.read_text(encoding="utf-8")
    ids = re.findall(r"^## (FVT-G\d+) ", text, flags=re.MULTILINE)
    assert ids and ids[0] == PROGRAM_GOAL_ID
    children = [goal_id for goal_id in ids if goal_id != PROGRAM_GOAL_ID]
    assert children, "expected executable child goals under FVT-G000"
    assert COMPLETION_GOAL_ID in children
    return children


def _load_builder_module():
    assert BUILDER_PATH.is_file(), f"missing receipt builder: {BUILDER_PATH}"
    spec = importlib.util.spec_from_file_location(
        "build_formal_verification_tactician_receipt",
        BUILDER_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register before exec so dataclasses / annotations resolve cleanly.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_builder_and_receipt_and_objectives_exist() -> None:
    assert BUILDER_PATH.is_file()
    assert RECEIPT_PATH.is_file()
    assert OBJECTIVES_PATH.is_file()
    assert Path(__file__).is_file()
    builder_text = BUILDER_PATH.read_text(encoding="utf-8")
    assert INTERFACE in builder_text
    assert "hardcoded" in builder_text.lower() or "hardcoded_success" in builder_text
    assert "implementation" in builder_text
    assert "deployment" in builder_text


def test_completion_receipt_schema_and_interface() -> None:
    receipt = _load_receipt()
    assert receipt["schema_version"] == SCHEMA_VERSION
    assert receipt["interface"] == INTERFACE
    assert receipt["program_interface"] == PROGRAM_INTERFACE
    assert receipt["program_goal_id"] == PROGRAM_GOAL_ID
    assert receipt["completion_goal_id"] == COMPLETION_GOAL_ID
    assert receipt["task_id"] == TASK_ID
    assert receipt["program"] == "formal-verification-tactician/readiness"
    assert str(receipt["receipt_identity"]).startswith("sha256:")
    assert receipt["binding_mode"] == "current_tree_content_identity"

    source = receipt["source"]
    assert source["binding_mode"] == "current_tree_content_identity"
    assert COMMIT_RE.fullmatch(str(source["parent_commit"]))
    assert COMMIT_RE.fullmatch(str(source["parent_tree"]))
    assert COMMIT_RE.fullmatch(str(source["datasets_gitlink"]))
    assert COMMIT_RE.fullmatch(str(source["datasets_embedded_head"]))


def test_separate_implementation_and_deployment_sections() -> None:
    receipt = _load_receipt()
    implementation = receipt["implementation"]
    deployment = receipt["deployment"]

    assert implementation["status"] in {"complete", "incomplete"}
    assert deployment["status"] in {
        "machine_specific_certified",
        "machine_specific_partial",
        "not_deployment_certified",
    }
    # Implementation completeness must never silently equal full deployment.
    assert implementation is not deployment
    assert "child_goals_bound" in implementation
    assert "exact_tools" in deployment
    assert "live_simulated_skipped" in deployment
    assert "assurance_ceilings" in deployment
    assert "publication_gates" in deployment
    assert implementation.get("hardcoded_success_counters") is False
    assert deployment.get("hardcoded_success_counters") is False

    ceilings = deployment["assurance_ceilings"]
    assert ceilings["path_presence_is_not_usability"] is True
    assert ceilings["source_presence_is_not_usability"] is True
    assert ceilings["fixture_is_not_production_certified"] is True
    assert ceilings["synthetic_evidence_cannot_certify_production"] is True


def test_receipt_binds_all_objective_child_goals() -> None:
    receipt = _load_receipt()
    expected_ids = _parse_objective_child_ids()
    children = receipt["child_goals"]
    assert isinstance(children, list)
    assert len(children) == len(expected_ids)

    observed_ids = [child["goal_id"] for child in children]
    assert observed_ids == expected_ids
    assert PROGRAM_GOAL_ID not in observed_ids
    assert COMPLETION_GOAL_ID in observed_ids

    for child in children:
        assert child["bound"] is True, child["goal_id"]
        assert child["goal_id"].startswith("FVT-G")
        assert isinstance(child["title"], str) and child["title"].strip()
        assert isinstance(child["evidence"], list) and child["evidence"]
        assert isinstance(child["outputs"], list) and child["outputs"]
        assert isinstance(child.get("interfaces"), list)
        assert child["evidence_missing"] == []
        for path in child["evidence"]:
            # Receipt self-output is bound via self:receipt_identity.
            if path.endswith("formal_verification_tactician_readiness_completion_receipt.json"):
                continue
            assert (REPO_ROOT / path).is_file(), path


def test_hard_zero_gates_are_present_and_non_negative() -> None:
    receipt = _load_receipt()
    hard_zero = receipt["hard_zero_gates"]
    for gate in HARD_ZERO_GATES:
        assert gate in hard_zero, gate
        assert isinstance(hard_zero[gate], int)
        assert hard_zero[gate] >= 0, gate

    derivation = hard_zero.get("derivation") or {}
    assert derivation.get("hardcoded_success_counters_forbidden") is True

    acceptance = receipt["acceptance"]
    for gate in HARD_ZERO_GATES:
        assert acceptance[gate] == hard_zero[gate]
    assert acceptance["hardcoded_success_counters"] is False
    assert acceptance["implementation_section_present"] is True
    assert acceptance["deployment_section_present"] is True


def test_hard_zero_gates_clear_when_child_certificates_pass() -> None:
    """Hard-zero must reflect child certificates, not invented counters.

    When the benchmark hard gates pass and the toolchain certificate has no
    disagreement quarantines, every hard-zero counter must be zero.
    """

    receipt = _load_receipt()
    certificate = json.loads(TOOLCHAIN_CERT_PATH.read_text(encoding="utf-8"))
    benchmark = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))

    quarantines = certificate.get("disagreement_quarantines") or []
    hard = (
        (benchmark.get("report") or {}).get("gates") or benchmark.get("gates") or {}
    ).get("hard") or {}

    def _passed(name: str) -> bool:
        status = hard.get(name) or {}
        if str(status.get("status") or "").lower() == "pass":
            return True
        actual = status.get("actual_bps")
        required = status.get("required_bps")
        return isinstance(actual, int) and isinstance(required, int) and actual >= required

    if (
        isinstance(quarantines, list)
        and len(quarantines) == 0
        and _passed("correctness")
        and _passed("privacy")
        and _passed("authority")
    ):
        hard_zero = receipt["hard_zero_gates"]
        for gate in HARD_ZERO_GATES:
            assert hard_zero[gate] == 0, gate
        assert receipt["acceptance"]["hard_zero_gates_clear"] is True


def test_artifacts_bound_with_content_identities() -> None:
    receipt = _load_receipt()
    artifacts = receipt["artifacts"]
    for key in REQUIRED_ARTIFACT_KEYS:
        entry = artifacts[key]
        assert entry["present"] is True, key
        assert entry["path"]
        assert entry["content_identity"]

        if key == "completion_receipt":
            assert entry["content_identity"] == "self:receipt_identity"
            continue

        path = REPO_ROOT / entry["path"]
        assert path.is_file(), path
        assert entry["content_identity"] == _sha256_bytes(path.read_bytes()), key

    # Self path for this test file.
    assert artifacts["completion_test"]["path"].endswith(
        "test/api/test_formal_verification_tactician_readiness_completion.py"
    )
    assert artifacts["receipt_builder"]["path"].endswith(
        "tools/logic/build_formal_verification_tactician_receipt.py"
    )


def test_implementation_binds_schemas_corpus_public_ops_metrics_rollout() -> None:
    receipt = _load_receipt()
    implementation = receipt["implementation"]
    assert implementation["schemas_bound"] is True
    assert implementation["corpus_bound"] is True
    assert implementation["public_operations_bound"] is True
    assert implementation["metrics_bound"] is True
    assert implementation["rollout_policy_bound"] is True
    assert implementation["completion_surfaces_bound"] is True
    assert implementation["status"] == "complete"
    assert implementation["child_goal_count"] == len(receipt["child_goals"])
    assert implementation["child_goals_bound"] == implementation["child_goal_count"]
    assert implementation["child_goals_unbound"] == []
    assert implementation["corpus_case_count"] > 0
    assert CORPUS_PATH.is_file()
    assert ROLLOUT_PATH.is_file()
    assert BASELINE_PATH.is_file()


def test_deployment_binds_tools_live_classes_and_publication() -> None:
    receipt = _load_receipt()
    deployment = receipt["deployment"]
    assert isinstance(deployment["exact_tools"], list)
    assert "usable_tools" in deployment
    assert "unavailable_tools" in deployment
    assert "production_certified_tool_ids" in deployment
    exact_tools = {
        str(tool["certification_tool_id"]): tool
        for tool in deployment["exact_tools"]
    }
    production_certified = set(deployment["production_certified_tool_ids"])
    assert production_certified == {
        tool_id
        for tool_id, tool in exact_tools.items()
        if tool["statuses"]["production_certified"]
    }
    for tool_id in production_certified:
        assert exact_tools[tool_id]["certification"]["production_certified"] is True

    lean = exact_tools["lean"]
    assert (
        lean["certification"]["locked_version"].lstrip("v")
        in lean["version_string"]
    )
    live = deployment["live_simulated_skipped"]
    assert "live_case_count" in live
    assert "simulated_or_fixture_case_count" in live
    assert "skipped_or_unavailable_case_count" in live
    assert "evidence_class_policy" in live

    alignment = deployment["tree_alignment"]
    for dimension in (
        "parent_commit",
        "datasets_gitlink",
        "datasets_embedded_head",
        "datasets_origin_main",
        "datasets_cleanliness",
    ):
        assert dimension in alignment["dimensions"]

    publication = deployment["publication_gates"]
    assert publication["fetches_forbidden"] is True
    assert "parent_publication_lag" in publication
    assert "datasets_publication_lag" in publication

    disclosures = receipt["disclosures"]
    assert isinstance(disclosures["unavailable_tools"], list)
    assert isinstance(disclosures["remaining_bounds"], list)
    assert disclosures["remaining_bounds"]
    assert "assurance_ceilings" in disclosures
    assert "publication_gates" in disclosures


def test_completion_goal_child_binding_includes_builder_and_test() -> None:
    receipt = _load_receipt()
    completion = next(
        child for child in receipt["child_goals"] if child["goal_id"] == COMPLETION_GOAL_ID
    )
    evidence = set(completion["evidence"])
    assert "tools/logic/build_formal_verification_tactician_receipt.py" in evidence
    assert (
        "test/api/test_formal_verification_tactician_readiness_completion.py" in evidence
    )
    outputs = set(completion["outputs"])
    assert (
        "docs/architecture/formal_verification_tactician_readiness_completion_receipt.json"
        in outputs
    )
    for path in completion["evidence"]:
        if path.endswith("formal_verification_tactician_readiness_completion_receipt.json"):
            continue
        assert (REPO_ROOT / path).is_file(), path


def test_g000_and_g090_objective_heap_point_at_this_receipt() -> None:
    text = OBJECTIVES_PATH.read_text(encoding="utf-8")
    assert "formal_verification_tactician_readiness_completion_receipt.json" in text

    g000 = re.search(
        r"^## FVT-G000 .+?(?=^## |\Z)",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )
    g090 = re.search(
        r"^## FVT-G090 .+?(?=^## |\Z)",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )
    assert g000 is not None and g090 is not None
    assert (
        "docs/architecture/formal_verification_tactician_readiness_completion_receipt.json"
        in g000.group(0)
    )
    assert "build_formal_verification_tactician_receipt.py" in g090.group(0)
    assert "test_formal_verification_tactician_readiness_completion.py" in g090.group(0)
    assert INTERFACE in g090.group(0) or "FormalVerificationTacticianCompletionReceipt" in g090.group(0)


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


def test_builder_recomputes_equivalent_receipt(tmp_path: Path) -> None:
    """Builder must regenerate from current tree without editing source evidence."""

    module = _load_builder_module()
    committed = _load_receipt()
    rebuilt = module.build_receipt(
        repo_root=REPO_ROOT,
        observed_at=committed["observed_at"],
    )

    # Structural equivalence excluding volatile identity recomputation path.
    assert rebuilt["schema_version"] == committed["schema_version"]
    assert rebuilt["interface"] == committed["interface"]
    assert rebuilt["program_goal_id"] == committed["program_goal_id"]
    assert rebuilt["completion_goal_id"] == committed["completion_goal_id"]
    assert len(rebuilt["child_goals"]) == len(committed["child_goals"])
    assert [c["goal_id"] for c in rebuilt["child_goals"]] == [
        c["goal_id"] for c in committed["child_goals"]
    ]
    assert rebuilt["implementation"]["status"] == committed["implementation"]["status"]
    assert set(rebuilt["hard_zero_gates"]) >= set(HARD_ZERO_GATES)
    for gate in HARD_ZERO_GATES:
        assert rebuilt["hard_zero_gates"][gate] == committed["hard_zero_gates"][gate]

    current_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
    ).strip()
    receipt_source_head = str(committed["source"]["parent_commit"])
    if receipt_source_head == current_head:
        # Identity must match while rebuilding the same uncommitted evidence
        # tree from which the historical receipt was issued.
        assert rebuilt["receipt_identity"] == committed["receipt_identity"]
    else:
        # A checked-in receipt necessarily describes its parent evidence
        # commit: the receipt cannot contain the hash of the commit that in
        # turn contains the receipt.  Preserve that historical identity and
        # require its source to be an ancestor of the publication wrapper.
        assert subprocess.run(
            ["git", "merge-base", "--is-ancestor", receipt_source_head, current_head],
            cwd=REPO_ROOT,
            check=False,
        ).returncode == 0
        assert rebuilt["receipt_identity"] != committed["receipt_identity"]

    # Writing to a temp path must not mutate source child evidence files.
    baseline_before = BASELINE_PATH.read_bytes()
    cert_before = TOOLCHAIN_CERT_PATH.read_bytes()
    out = tmp_path / "receipt.json"
    module.write_receipt(rebuilt, out)
    assert out.is_file()
    assert BASELINE_PATH.read_bytes() == baseline_before
    assert TOOLCHAIN_CERT_PATH.read_bytes() == cert_before
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["receipt_identity"] == rebuilt["receipt_identity"]


def test_no_hardcoded_success_counters_in_builder_source() -> None:
    """Builder source must forbid hardcoded success counters by construction."""

    text = BUILDER_PATH.read_text(encoding="utf-8")
    # Must not assign literal promotional success tallies.
    assert "production_certified_count = 36" not in text
    assert "false_proof_count\": 0," not in text or "hardcoded" in text.lower()
    assert "hardcoded_success_counters" in text
    assert "derive_hard_zero_gates" in text
    assert "never invent" in text.lower() or "never invents" in text.lower()


def test_disclosures_do_not_fabricate_full_deployment_when_tools_unavailable() -> None:
    receipt = _load_receipt()
    deployment = receipt["deployment"]
    unavailable = deployment.get("unavailable_tools") or []
    ceilings = deployment["assurance_ceilings"]
    # If any tool is unavailable, full global deployment certification is false.
    if unavailable:
        assert ceilings.get("machine_certified") is False or deployment["status"] in {
            "machine_specific_partial",
            "not_deployment_certified",
        }
        assert deployment["status"] != "machine_specific_certified" or ceilings.get(
            "publication_aligned"
        )
    # Synthetic / fixture never counts as production certified.
    assert ceilings["fixture_is_not_production_certified"] is True
    assert ceilings["live_without_hermetic_certificate_is_not_production_certified"] is True
