"""Fail-closed production-semantic elevation fan-in (FVT-081)."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
CERTIFIER_PATH = (
    REPO_ROOT / "tools" / "logic" / "certify_formal_verification_toolchains.py"
)
BUILDER_PATH = (
    REPO_ROOT
    / "tools"
    / "logic"
    / "build_formal_verification_tactician_receipt.py"
)
RECEIPT_PATH = (
    REPO_ROOT
    / "docs"
    / "architecture"
    / "formal_verification_production_elevation_fanin_receipt.json"
)

INTERFACE = "ProductionSemanticElevationFanIn@1"
SCHEMA = "formal-verification-production-semantic-elevation-fanin/v1"
GOAL_ID = "FVT-G213"
TASK_ID = "FVT-081"
REQUIRED_TOOLS = (
    "lean",
    "runtime-mtl",
    "datalog-authorization",
    "secpal-authorization",
    "coq",
    "isabelle",
)
REQUIRED_KINDS = {"positive", "negative", "mutation", "replay"}


def _load(path: Path, name: str):
    for candidate in (REPO_ROOT, REPO_ROOT / "ipfs_datasets_py"):
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def certifier():
    return _load(CERTIFIER_PATH, "fvt081_certifier")


@pytest.fixture(scope="module")
def builder():
    return _load(BUILDER_PATH, "fvt081_builder")


@pytest.fixture(scope="module")
def certificate_bundle(certifier) -> tuple[dict[str, Any], dict[str, Any]]:
    full_evidence: dict[str, Any] = {}
    certificate = certifier.build_certificate(
        repo_root=REPO_ROOT,
        role_aware=True,
        full_evidence_out=full_evidence,
    )
    return certificate, full_evidence


@pytest.fixture(scope="module")
def certificate(certificate_bundle) -> dict[str, Any]:
    return certificate_bundle[0]


@pytest.fixture(scope="module")
def fanin(builder, certificate) -> dict[str, Any]:
    return builder.build_production_semantic_elevation_fanin(
        repo_root=REPO_ROOT,
        observed_at="2026-08-01T00:00:00Z",
        role_aware_certificate=certificate,
    )


def _lane_for_tool(
    certifier,
    certificate: dict[str, Any],
    tool_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    spec = next(
        row
        for row in certifier.SEMANTIC_CERTIFIER_SPECS
        if tool_id in set(row.get("tool_ids") or ())
    )
    lane = next(
        row
        for row in certificate["semantic_lane_results"]
        if row.get("lane_id") == spec["lane_id"]
    )
    return dict(spec), lane


def test_interface_identity_and_derived_acceptance(
    builder,
    fanin: dict[str, Any],
) -> None:
    assert fanin["interface"] == INTERFACE
    assert fanin["schema_version"] == SCHEMA
    assert fanin["goal_id"] == GOAL_ID
    assert fanin["task_id"] == TASK_ID
    assert tuple(fanin["required_tools"]) == REQUIRED_TOOLS
    assert set(fanin["tools"]) == set(REQUIRED_TOOLS)
    assert fanin["certificate_identity"]["valid"] is True
    assert fanin["acceptance"][
        "role_aware_certificate_identity_bound"
    ] is True
    assert fanin["acceptance"]["checks_never_collapsed"] is True
    assert fanin["acceptance"]["raw_checks_not_reembedded"] is True
    assert fanin["acceptance"]["offline_only"] is True
    assert fanin["acceptance"]["structurally_valid"] == fanin["summary"][
        "structurally_valid"
    ]
    assert fanin["acceptance"]["fanin_closed"] == fanin["summary"][
        "fanin_closed"
    ]
    body = {
        key: value
        for key, value in fanin.items()
        if key != "receipt_digest_sha256"
    }
    assert fanin["receipt_digest_sha256"] == builder.content_digest(body)
    assert '"checks":[' not in json.dumps(
        fanin,
        separators=(",", ":"),
    )


def test_each_required_tool_has_exact_compact_bound_pnmr(
    certifier,
    certificate: dict[str, Any],
    fanin: dict[str, Any],
) -> None:
    for tool_id in REQUIRED_TOOLS:
        row = fanin["tools"][tool_id]
        reconstruction = row["independent_reconstruction"]
        _, lane = _lane_for_tool(certifier, certificate, tool_id)
        recomputed = certifier.recompute_semantic_tool_check_binding(
            lane,
            tool_id,
        )
        assert reconstruction["valid"] is True, tool_id
        assert reconstruction["compact_binding_valid"] is True, tool_id
        assert reconstruction["required_kinds_present"] is True, tool_id
        assert reconstruction["required_kinds_all_passed"] is True, tool_id
        assert reconstruction["required_kinds_failed"] == [], tool_id
        assert REQUIRED_KINDS <= set(
            reconstruction["check_kinds_present"]
        )
        assert reconstruction["check_set_digest_sha256"] == recomputed[
            "check_set_digest_sha256"
        ]
        assert reconstruction[
            "compact_check_set_digest_sha256"
        ] == recomputed["check_set_digest_sha256"]
        assert reconstruction["checks_total"] == recomputed["checks_total"]
        assert "checks" not in reconstruction
        if row["production_elevation_present"]:
            assert row["eligible_for_production_elevation"] is True
            assert row["semantic_authority_valid"] is True


def test_current_missing_elevations_are_disclosed_not_promoted(
    fanin: dict[str, Any],
) -> None:
    missing = set(fanin["summary"]["production_elevation_missing"])
    assert missing <= set(REQUIRED_TOOLS)
    if missing:
        assert fanin["summary"]["fanin_closed"] is False
        assert fanin["acceptance"]["fanin_closed"] is False
    for tool_id, row in fanin["tools"].items():
        if not row["production_elevation_allowed"]:
            assert row["production_elevation_present"] is False
            assert row["eligible_for_production_elevation"] is False
            assert (
                "production_elevation_not_allowed_by_evidence_class"
                in row["block_reasons"]
            ), tool_id


def test_bad_outer_certificate_identity_fails_closed(
    builder,
    certificate: dict[str, Any],
) -> None:
    corrupted = copy.deepcopy(certificate)
    corrupted["certificate_digest_sha256"] = "0" * 64
    checked = builder.build_production_semantic_elevation_fanin(
        repo_root=REPO_ROOT,
        observed_at="2026-08-01T00:00:00Z",
        role_aware_certificate=corrupted,
    )
    assert checked["certificate_identity"]["digest_valid"] is False
    assert checked["certificate_identity"]["valid"] is False
    assert checked["acceptance"][
        "role_aware_certificate_identity_bound"
    ] is False
    assert checked["summary"]["structurally_valid"] is False
    assert checked["acceptance"]["structurally_valid"] is False
    assert checked["status"] == (
        "production_semantic_elevation_fanin_blocked"
    )
    assert "role_aware_certificate_identity_invalid" in checked["summary"][
        "failures"
    ]


def test_duplicate_failed_required_kind_invalidates_reconstruction(
    builder,
    certifier,
    certificate: dict[str, Any],
) -> None:
    _, original_lane = _lane_for_tool(certifier, certificate, "lean")
    lane = copy.deepcopy(original_lane)
    receipt = lane["receipt"]
    checks = receipt["checks"]
    positive = next(
        check
        for check in checks
        if str(check.get("kind") or "") == "positive"
    )
    duplicate = copy.deepcopy(positive)
    duplicate["check_id"] = (
        str(duplicate.get("check_id") or "positive")
        + "-duplicate-failure"
    )
    duplicate["status"] = "failed"
    checks.append(duplicate)
    recomputed = certifier.recompute_semantic_tool_check_binding(
        lane,
        "lean",
    )
    compact = copy.deepcopy(lane["per_tool"]["lean"])
    compact.update(
        {
            "check_set_digest_sha256": recomputed[
                "check_set_digest_sha256"
            ],
            "checks_total": recomputed["checks_total"],
            "checks_passed": recomputed["checks_passed"],
            "check_kinds_present": recomputed["check_kinds_present"],
            "check_status_counts": recomputed["check_status_counts"],
        }
    )
    reconstruction = builder._independent_pnmr_reconstruction(
        certifier=certifier,
        semantic_result=lane,
        tool_id="lean",
        compact_tool=compact,
    )
    assert reconstruction["compact_binding_valid"] is True
    assert reconstruction["required_kinds_present"] is True
    assert reconstruction["required_kinds_all_passed"] is False
    assert reconstruction["required_kinds_failed"] == ["positive"]
    assert reconstruction["valid"] is False


def test_compact_digest_mismatch_invalidates_reconstruction(
    builder,
    certifier,
    certificate: dict[str, Any],
) -> None:
    _, lane = _lane_for_tool(certifier, certificate, "lean")
    compact = copy.deepcopy(lane["per_tool"]["lean"])
    compact["check_set_digest_sha256"] = "f" * 64
    reconstruction = builder._independent_pnmr_reconstruction(
        certifier=certifier,
        semantic_result=lane,
        tool_id="lean",
        compact_tool=compact,
    )
    assert reconstruction["recompute_valid"] is True
    assert reconstruction["compact_binding_valid"] is False
    assert reconstruction["valid"] is False


def test_offline_policy_mutation_fails_closed(
    builder,
    certifier,
    certificate: dict[str, Any],
) -> None:
    corrupted = copy.deepcopy(certificate)
    corrupted["certification_policy"]["offline_policy_satisfied"] = False
    corrupted["certificate_digest_sha256"] = certifier.content_digest(
        {
            key: value
            for key, value in corrupted.items()
            if key != "certificate_digest_sha256"
        }
    )
    checked = builder.build_production_semantic_elevation_fanin(
        repo_root=REPO_ROOT,
        observed_at="2026-08-01T00:00:00Z",
        role_aware_certificate=corrupted,
    )
    assert checked["certificate_identity"]["valid"] is True
    assert checked["offline_derivation"]["satisfied"] is False
    assert checked["acceptance"]["offline_only"] is False
    assert checked["summary"]["structurally_valid"] is False
    assert "offline_policy_not_satisfied" in checked["summary"]["failures"]


def test_checked_receipt_is_never_accepted_by_presence_alone(
    builder,
    fanin: dict[str, Any],
) -> None:
    assert RECEIPT_PATH.is_file(), RECEIPT_PATH
    checked = builder.verify_checked_production_elevation_fanin(
        repo_root=REPO_ROOT,
        live_fanin=fanin,
    )
    assert checked["present"] is True
    assert checked["stored_digest_valid"] is True
    assert checked["identity_valid"] is True
    assert checked["content_identity"]
    if checked["matches_live"]:
        assert checked["stored_digest_sha256"] == fanin[
            "receipt_digest_sha256"
        ]
