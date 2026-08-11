"""Fail-closed role-aware release candidate (FVT-066 / FVT-G213).

``RoleAwareFormalVerificationReleaseCandidate@1`` fans in the complete
supported matrix without claiming its own future merge or deployment.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CERTIFIER_PATH = (
    REPO_ROOT / "tools" / "logic" / "certify_formal_verification_toolchains.py"
)
BUILDER_PATH = (
    REPO_ROOT
    / "tools"
    / "logic"
    / "build_formal_verification_tactician_receipt.py"
)
CANDIDATE_PATH = (
    REPO_ROOT
    / "docs"
    / "architecture"
    / "formal_verification_role_aware_release_candidate.json"
)
FANIN_RECEIPT_PATH = (
    REPO_ROOT
    / "docs"
    / "architecture"
    / "formal_verification_production_elevation_fanin_receipt.json"
)
FANIN_TEST_PATH = (
    REPO_ROOT
    / "test"
    / "integration"
    / "toolchains"
    / "test_formal_verification_production_elevation_fanin.py"
)

INTERFACE = "RoleAwareFormalVerificationReleaseCandidate@1"
GOAL_ID = "FVT-G213"
TASK_ID = "FVT-066"
MAX_STAGE = "release_candidate"
FANIN_INTERFACE = "ProductionSemanticElevationFanIn@1"
FANIN_TASK_ID = "FVT-081"

REQUIRED_ELEVATIONS = {
    "lean",
    "runtime-mtl",
    "datalog-authorization",
    "secpal-authorization",
    "coq",
    "isabelle",
}
NON_AUTHORITATIVE_CLASSES = {
    "identity_plus_fixture_parser",
    "hermetic_adapter_shim",
    "hermetic_shadow_shim",
    "proposal_only_semantics",
}
REQUIRED_CHECK_KINDS = {"positive", "negative", "mutation", "replay"}


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
    return _load(CERTIFIER_PATH, "fvt_release_candidate_certifier_test")


@pytest.fixture(scope="module")
def builder():
    return _load(BUILDER_PATH, "fvt_release_candidate_builder_test")


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
def source_specialized(certificate_bundle) -> dict[str, Any]:
    return certificate_bundle[1]["specialized_receipt_aggregation"]


@pytest.fixture(scope="module")
def candidate(builder, certificate, source_specialized) -> dict[str, Any]:
    return builder.build_role_aware_release_candidate(
        repo_root=REPO_ROOT,
        observed_at="2026-08-01T00:00:00Z",
        role_aware_certificate=certificate,
        source_specialized_receipt_aggregation=source_specialized,
    )


def _tools(certificate: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["tool_id"]: row for row in certificate["tools"]}


def test_expected_outputs_exist_and_candidate_is_tracked_evidence() -> None:
    for path in (
        CERTIFIER_PATH,
        BUILDER_PATH,
        CANDIDATE_PATH,
        FANIN_RECEIPT_PATH,
        FANIN_TEST_PATH,
    ):
        assert path.is_file(), path
    # Global *.json may still match until a repository negation lands; force-
    # tracked index membership is sufficient durable evidence for FVT-G213.
    ignored = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "check-ignore", "-q", str(CANDIDATE_PATH)],
        check=False,
    )
    ls_files = subprocess.run(
        [
            "git",
            "-C",
            str(REPO_ROOT),
            "ls-files",
            "--error-unmatch",
            str(CANDIDATE_PATH.relative_to(REPO_ROOT)),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert ignored.returncode != 0 or ls_files.returncode == 0, (
        "release candidate must be non-ignored or index-tracked evidence"
    )


def test_candidate_is_compact_not_bulk_certificate_dump(
    candidate: dict[str, Any],
    certificate: dict[str, Any],
) -> None:
    """Proposal gate: never embed multi-MB formal certificate bodies."""

    encoded = json.dumps(candidate, separators=(",", ":"), ensure_ascii=False)
    assert len(encoded.encode("utf-8")) < 500_000, (
        "release candidate body must stay compact (digest-bound fan-in)"
    )
    bound = candidate["role_aware_certificate"]
    assert bound["projection_model"] == "digest_bound_compact_projection/v1"
    assert bound["raw_certificate_embedded"] is False
    lanes_by_id = {
        str(row.get("lane_id")): row
        for row in certificate.get("semantic_lane_results") or []
    }
    for lane in bound.get("semantic_lane_results") or []:
        assert "receipt" not in lane
        assert "per_tool" not in lane
        assert lane["digest_sha256"] == lanes_by_id[lane["lane_id"]].get(
            "digest_sha256"
        )
        assert set(lane["per_tool_bindings"]) == set(
            (lanes_by_id[lane["lane_id"]].get("per_tool") or {}).keys()
        )
    specialized = bound["specialized_receipt_aggregation"]
    assert "source_specialized_receipt_aggregation" not in specialized
    projection = specialized["projection"]
    assert len(projection["specialized_by_handler"]) == 21
    assert all(
        "checks" not in handler
        for handler in projection["specialized_by_handler"].values()
    )
    for tool in bound.get("tools") or []:
        assert "checks" not in tool
        assert tool.get("checks_digest_sha256")


def test_compact_projection_retains_lane_tool_and_handler_digest_bindings(
    certifier,
    candidate: dict[str, Any],
    certificate: dict[str, Any],
    source_specialized: dict[str, Any],
) -> None:
    """Every compact row remains independently tied to its full evidence."""

    bound = candidate["role_aware_certificate"]
    compact_lanes = {
        str(row["lane_id"]): row
        for row in bound["semantic_lane_results"]
    }
    for lane in certificate.get("semantic_lane_results") or []:
        lane_id = str(lane["lane_id"])
        compact = compact_lanes[lane_id]
        assert compact["digest_sha256"] == lane.get("digest_sha256")
        for tool_id, per_tool in (lane.get("per_tool") or {}).items():
            tool_binding = compact["per_tool_bindings"][tool_id]
            assert tool_binding["check_set_digest_sha256"] == per_tool.get(
                "check_set_digest_sha256"
            )
            assert tool_binding["tool_evidence_digest_sha256"] == (
                certifier.content_digest(per_tool)
            )

    compact_specialized = certificate["specialized_receipt_aggregation"]
    bound_specialized = bound["specialized_receipt_aggregation"]
    projection = bound_specialized["projection"]
    verification = bound_specialized["verification"]
    assert projection == compact_specialized
    assert verification["projection_valid"] is True
    assert verification["source_valid"] is True
    assert verification["independent_full_evidence_valid"] is True
    assert verification["handler_population_valid"] is True
    assert verification["source_handler_population_valid"] is True
    assert verification["expected_handler_count"] == 21
    assert verification["handler_count"] == 21
    assert verification["failures"] == []

    projection_body = {
        key: value
        for key, value in projection.items()
        if key != "aggregation_digest_sha256"
    }
    assert projection["aggregation_digest_sha256"] == (
        certifier.content_digest(projection_body)
    )
    assert projection["source_aggregation_digest_sha256"] == (
        source_specialized["aggregation_digest_sha256"]
    )

    compact_handlers = projection["specialized_by_handler"]
    source_handlers = source_specialized["specialized_by_handler"]
    assert set(compact_handlers) == set(source_handlers)
    assert set(verification["handlers"]) == set(compact_handlers)
    assert len(compact_handlers) == 21
    for handler_key, handler in compact_handlers.items():
        source_handler = source_handlers[handler_key]
        handler_body = {
            key: value
            for key, value in handler.items()
            if key != "tool_evidence_digest_sha256"
        }
        source_body = {
            key: value
            for key, value in source_handler.items()
            if key != "tool_evidence_digest_sha256"
        }
        assert handler["tool_evidence_digest_sha256"] == (
            certifier.content_digest(handler_body)
        )
        assert handler["source_tool_evidence_digest_sha256"] == (
            source_handler["tool_evidence_digest_sha256"]
        )
        assert source_handler["tool_evidence_digest_sha256"] == (
            certifier.content_digest(source_body)
        )
        handler_check = verification["handlers"][handler_key]
        assert handler_check[
            "projection_tool_evidence_digest_valid"
        ] is True
        assert handler_check[
            "source_tool_evidence_digest_verified"
        ] is True
        assert handler_check["mapping_valid"] is True
        assert handler_check["receipt_binding_valid"] is True

    composite_handlers = [
        handler_key
        for composite in projection["composite_lanes"].values()
        for handler_key in composite["handler_keys"]
    ]
    assert len(projection["composite_lanes"]) == 9
    assert len(composite_handlers) == 21
    assert set(composite_handlers) == set(compact_handlers)
    assert projection["enabled"] is True
    assert projection["lossless"] is True
    for handler_key, source_handler in source_handlers.items():
        assert compact_handlers[handler_key]["identity_digest_sha256"] == (
            certifier.content_digest(source_handler["identity"])
        )


def test_source_digest_provenance_is_not_treated_as_verified_without_source(
    builder,
    certificate: dict[str, Any],
) -> None:
    checked = builder.build_role_aware_release_candidate(
        repo_root=REPO_ROOT,
        observed_at="2026-08-01T00:00:00Z",
        role_aware_certificate=certificate,
    )
    verification = checked["role_aware_certificate"][
        "specialized_receipt_aggregation"
    ]["verification"]
    assert verification["projection_valid"] is True
    assert verification["source_evidence_supplied"] is False
    assert verification["source_valid"] is False
    assert all(
        row["source_tool_evidence_digest_verified"] is False
        for row in verification["handlers"].values()
    )
    assert checked["acceptance"][
        "specialized_source_evidence_independently_verified"
    ] is False
    assert "specialized_source_evidence_independently_verified" in checked[
        "blockers"
    ]


def test_specialized_projection_mutation_and_population_loss_fail_closed(
    certifier,
    builder,
    certificate: dict[str, Any],
    source_specialized: dict[str, Any],
) -> None:
    mutated = copy.deepcopy(certificate)
    specialized = mutated["specialized_receipt_aggregation"]
    handler_key = sorted(specialized["specialized_by_handler"])[0]
    specialized["specialized_by_handler"][handler_key]["tool_id"] = "forged"
    specialized["aggregation_digest_sha256"] = certifier.content_digest(
        {
            key: value
            for key, value in specialized.items()
            if key != "aggregation_digest_sha256"
        }
    )
    mutated["certificate_digest_sha256"] = certifier.content_digest(
        {
            key: value
            for key, value in mutated.items()
            if key != "certificate_digest_sha256"
        }
    )
    checked = builder.build_role_aware_release_candidate(
        repo_root=REPO_ROOT,
        observed_at="2026-08-01T00:00:00Z",
        role_aware_certificate=mutated,
        source_specialized_receipt_aggregation=source_specialized,
    )
    verification = checked["role_aware_certificate"][
        "specialized_receipt_aggregation"
    ]["verification"]
    assert verification["projection_valid"] is False
    assert any(
        "handler_mapping_mismatch" in failure
        or "projection_handler_digest_mismatch" in failure
        for failure in verification["failures"]
    )

    dropped = copy.deepcopy(certificate)
    dropped_specialized = dropped["specialized_receipt_aggregation"]
    dropped_specialized["specialized_by_handler"].pop(handler_key)
    dropped_specialized["aggregation_digest_sha256"] = (
        certifier.content_digest(
            {
                key: value
                for key, value in dropped_specialized.items()
                if key != "aggregation_digest_sha256"
            }
        )
    )
    dropped["certificate_digest_sha256"] = certifier.content_digest(
        {
            key: value
            for key, value in dropped.items()
            if key != "certificate_digest_sha256"
        }
    )
    dropped_candidate = builder.build_role_aware_release_candidate(
        repo_root=REPO_ROOT,
        observed_at="2026-08-01T00:00:00Z",
        role_aware_certificate=dropped,
        source_specialized_receipt_aggregation=source_specialized,
    )
    dropped_verification = dropped_candidate["role_aware_certificate"][
        "specialized_receipt_aggregation"
    ]["verification"]
    assert dropped_verification["handler_population_valid"] is False
    assert dropped_verification["projection_valid"] is False

    forged_identity = copy.deepcopy(certificate)
    forged_specialized = forged_identity["specialized_receipt_aggregation"]
    forged_handler = forged_specialized["specialized_by_handler"][
        handler_key
    ]
    forged_handler["identity_digest_sha256"] = "f" * 64
    forged_handler["tool_evidence_digest_sha256"] = certifier.content_digest(
        {
            key: value
            for key, value in forged_handler.items()
            if key != "tool_evidence_digest_sha256"
        }
    )
    forged_specialized["aggregation_digest_sha256"] = (
        certifier.content_digest(
            {
                key: value
                for key, value in forged_specialized.items()
                if key != "aggregation_digest_sha256"
            }
        )
    )
    forged_identity["certificate_digest_sha256"] = certifier.content_digest(
        {
            key: value
            for key, value in forged_identity.items()
            if key != "certificate_digest_sha256"
        }
    )
    forged_candidate = builder.build_role_aware_release_candidate(
        repo_root=REPO_ROOT,
        observed_at="2026-08-01T00:00:00Z",
        role_aware_certificate=forged_identity,
        source_specialized_receipt_aggregation=source_specialized,
    )
    forged_verification = forged_candidate["role_aware_certificate"][
        "specialized_receipt_aggregation"
    ]["verification"]
    assert forged_verification["projection_valid"] is True
    assert forged_verification["compact_projection_matches_source"] is False
    assert forged_verification["source_valid"] is False

    forged_composite = copy.deepcopy(certificate)
    forged_specialized = forged_composite[
        "specialized_receipt_aggregation"
    ]
    composite = next(iter(forged_specialized["composite_lanes"].values()))
    composite["handler_keys"] = list(composite["handler_keys"])[:-1]
    forged_specialized["aggregation_digest_sha256"] = (
        certifier.content_digest(
            {
                key: value
                for key, value in forged_specialized.items()
                if key != "aggregation_digest_sha256"
            }
        )
    )
    forged_composite["certificate_digest_sha256"] = (
        certifier.content_digest(
            {
                key: value
                for key, value in forged_composite.items()
                if key != "certificate_digest_sha256"
            }
        )
    )
    forged_composite_candidate = builder.build_role_aware_release_candidate(
        repo_root=REPO_ROOT,
        observed_at="2026-08-01T00:00:00Z",
        role_aware_certificate=forged_composite,
        source_specialized_receipt_aggregation=source_specialized,
    )
    composite_verification = forged_composite_candidate[
        "role_aware_certificate"
    ]["specialized_receipt_aggregation"]["verification"]
    assert composite_verification[
        "composite_handler_coverage_valid"
    ] is False
    assert composite_verification["projection_valid"] is False


def test_mutated_full_specialized_source_never_verifies(
    builder,
    certificate: dict[str, Any],
    source_specialized: dict[str, Any],
) -> None:
    mutated_source = copy.deepcopy(source_specialized)
    handler_key = sorted(mutated_source["specialized_by_handler"])[0]
    mutated_source["specialized_by_handler"][handler_key]["tool_id"] = "forged"
    checked = builder.build_role_aware_release_candidate(
        repo_root=REPO_ROOT,
        observed_at="2026-08-01T00:00:00Z",
        role_aware_certificate=certificate,
        source_specialized_receipt_aggregation=mutated_source,
    )
    verification = checked["role_aware_certificate"][
        "specialized_receipt_aggregation"
    ]["verification"]
    assert verification["projection_valid"] is True
    assert verification["source_valid"] is False
    assert verification["independent_full_evidence_valid"] is False
    assert checked["acceptance"][
        "specialized_receipt_aggregation_bound"
    ] is False


def test_self_rehashed_full_specialized_forgery_is_not_independent_evidence(
    certifier,
    builder,
    certificate: dict[str, Any],
    source_specialized: dict[str, Any],
) -> None:
    """A digest-consistent source rewrite still differs from reconstruction."""

    forged_source = copy.deepcopy(source_specialized)
    handler_key, handler = next(
        (
            key,
            row,
        )
        for key, row in forged_source["specialized_by_handler"].items()
        if row.get("cases")
    )
    handler["cases"][0]["status"] = "forged_pass"
    handler["tool_evidence_digest_sha256"] = certifier.content_digest(
        {
            key: value
            for key, value in handler.items()
            if key != "tool_evidence_digest_sha256"
        }
    )
    forged_source["aggregation_digest_sha256"] = (
        builder._specialized_source_aggregation_digest(
            certifier,
            forged_source,
        )
    )

    forged_certificate = copy.deepcopy(certificate)
    forged_certificate["specialized_receipt_aggregation"] = (
        certifier._compact_specialized_receipt_aggregation(
            forged_source
        )
    )
    forged_certificate["certificate_digest_sha256"] = (
        certifier.content_digest(
            {
                key: value
                for key, value in forged_certificate.items()
                if key != "certificate_digest_sha256"
            }
        )
    )
    checked = builder.build_role_aware_release_candidate(
        repo_root=REPO_ROOT,
        observed_at="2026-08-01T00:00:00Z",
        role_aware_certificate=forged_certificate,
        source_specialized_receipt_aggregation=forged_source,
    )
    verification = checked["role_aware_certificate"][
        "specialized_receipt_aggregation"
    ]["verification"]
    assert verification["projection_valid"] is True
    assert verification[
        "source_aggregation_digest_verified"
    ] is True
    assert verification[
        "source_matches_independent_reconstruction"
    ] is False
    assert verification["source_valid"] is False
    assert (
        "specialized:source_not_independently_reconstructed"
        in verification["failures"]
    )
    assert handler_key in verification["handlers"]


def test_self_rehashed_receipt_platform_and_global_authority_forgery_fail_closed(
    certifier,
    builder,
    certificate: dict[str, Any],
) -> None:
    forged_receipt = copy.deepcopy(certificate)
    lane = next(
        row
        for row in forged_receipt["semantic_lane_results"]
        if row.get("status") == "ran"
        and isinstance(row.get("receipt"), dict)
    )
    lane["interface"] = "ForgedSemanticReceipt@1"
    lane["receipt"]["interface"] = "ForgedSemanticReceipt@1"
    for digest_field in (
        "receipt_digest_sha256",
        "certificate_digest_sha256",
        "digest_sha256",
    ):
        if digest_field in lane["receipt"]:
            lane["receipt"][digest_field] = certifier.content_digest(
                {
                    key: value
                    for key, value in lane["receipt"].items()
                    if key != digest_field
                }
            )
    lane["digest_sha256"] = certifier.content_digest(lane["receipt"])
    forged_receipt["certificate_digest_sha256"] = certifier.content_digest(
        {
            key: value
            for key, value in forged_receipt.items()
            if key != "certificate_digest_sha256"
        }
    )
    checked_receipt = builder.build_role_aware_release_candidate(
        repo_root=REPO_ROOT,
        observed_at="2026-08-01T00:00:00Z",
        role_aware_certificate=forged_receipt,
    )
    assert checked_receipt["semantic_audit"]["valid"] is False
    assert any(
        "interface_mismatch" in failure
        for failure in checked_receipt["semantic_audit"]["failures"]
    )

    forged_platform = copy.deepcopy(certificate)
    managed = forged_platform["managed_deployment_readiness"]
    managed["ready"] = True
    managed["status"] = "all_supported_managed_capabilities_ready"
    managed["capability_blockers"] = []
    managed["dependency_blockers"] = []
    managed["all_blockers"] = []
    forged_platform["certificate_digest_sha256"] = certifier.content_digest(
        {
            key: value
            for key, value in forged_platform.items()
            if key != "certificate_digest_sha256"
        }
    )
    checked_platform = builder.build_role_aware_release_candidate(
        repo_root=REPO_ROOT,
        observed_at="2026-08-01T00:00:00Z",
        role_aware_certificate=forged_platform,
    )
    platform_audit = checked_platform["platform_support_audit"]
    assert platform_audit["valid"] is False
    assert "managed_blockers_or_ready_not_derived" in platform_audit[
        "failures"
    ]

    forged_authority = copy.deepcopy(certificate)
    tool = next(
        row
        for row in forged_authority["tools"]
        if row["tool_id"] == "java"
    )
    tool["production_certified"] = True
    production_ids = forged_authority["promotion"][
        "production_certified_tool_ids"
    ]
    production_ids.append("java")
    production_ids.sort()
    forged_authority["certificate_digest_sha256"] = (
        certifier.content_digest(
            {
                key: value
                for key, value in forged_authority.items()
                if key != "certificate_digest_sha256"
            }
        )
    )
    checked_authority = builder.build_role_aware_release_candidate(
        repo_root=REPO_ROOT,
        observed_at="2026-08-01T00:00:00Z",
        role_aware_certificate=forged_authority,
    )
    elevation_audit = checked_authority["required_elevation_audit"]
    assert elevation_audit["valid"] is False
    assert (
        "global_production_authority_not_independently_derived"
        in elevation_audit["failures"]
    )


def test_platform_audit_binds_exact_nonproduction_semantic_artifact(
    certifier,
    builder,
    certificate: dict[str, Any],
) -> None:
    checked = builder.build_role_aware_release_candidate(
        repo_root=REPO_ROOT,
        observed_at="2026-08-01T00:00:00Z",
        role_aware_certificate=certificate,
    )
    audit = checked["platform_support_audit"]
    assert audit["valid"] is True
    assert audit["live_artifact_failures"] == []
    runtime_lane = next(
        lane
        for lane in certificate["semantic_lane_results"]
        if lane.get("lane_id") == "runtime_mtl_external"
    )
    semantic_artifact = next(
        artifact
        for artifact in runtime_lane["per_tool"]["runtime-mtl-external"][
            "identity"
        ]["artifacts"]
        if artifact.get("kind") == "semantic_executable"
    )
    assert audit["non_production_artifact_omissions"] == []
    runtime_tool = _tools(certificate)["runtime-mtl-external"]
    assert runtime_tool["production_certified"] is False

    forged_semantic = copy.deepcopy(certificate)
    forged_runtime = _tools(forged_semantic)["runtime-mtl-external"]
    forged_runtime["artifact_identities"].append(
        {
            "kind": "semantic_executable",
            "path": "<host-path-redacted>",
            "sha256": f"sha256:{'f' * 64}",
            "artifact_class": "generated_hermetic_shim",
        }
    )
    forged_semantic["certificate_digest_sha256"] = certifier.content_digest(
        {
            key: value
            for key, value in forged_semantic.items()
            if key != "certificate_digest_sha256"
        }
    )
    checked_semantic = builder.build_role_aware_release_candidate(
        repo_root=REPO_ROOT,
        observed_at="2026-08-01T00:00:00Z",
        role_aware_certificate=forged_semantic,
    )
    forged_semantic_audit = checked_semantic["platform_support_audit"]
    assert forged_semantic_audit["valid"] is False
    assert (
        "runtime-mtl-external:artifact_live_identity_unavailable"
        in forged_semantic_audit["live_artifact_failures"]
    )

    forged_vendor = copy.deepcopy(certificate)
    forged_runtime = _tools(forged_vendor)["runtime-mtl-external"]
    vendor_artifact = next(
        (
            artifact
            for artifact in forged_runtime["artifact_identities"]
            if artifact.get("kind") == "executable"
        ),
        None,
    )
    if vendor_artifact is None:
        assert forged_runtime["installed"] is False
        assert forged_runtime["production_certified"] is False
    else:
        vendor_artifact["sha256"] = f"sha256:{'e' * 64}"
        forged_vendor["certificate_digest_sha256"] = (
            certifier.content_digest(
                {
                    key: value
                    for key, value in forged_vendor.items()
                    if key != "certificate_digest_sha256"
                }
            )
        )
        checked_vendor = builder.build_role_aware_release_candidate(
            repo_root=REPO_ROOT,
            observed_at="2026-08-01T00:00:00Z",
            role_aware_certificate=forged_vendor,
        )
        forged_vendor_audit = checked_vendor[
            "platform_support_audit"
        ]
        assert forged_vendor_audit["valid"] is False
        assert (
            "runtime-mtl-external:"
            "primary_executable_artifact_binding_mismatch"
            in forged_vendor_audit["live_artifact_failures"]
        )

    stale_lane = copy.deepcopy(certificate)
    runtime_lane = next(
        lane
        for lane in stale_lane["semantic_lane_results"]
        if lane.get("lane_id") == "runtime_mtl_external"
    )
    runtime_lane["certified"] = False
    stale_lane["certificate_digest_sha256"] = certifier.content_digest(
        {
            key: value
            for key, value in stale_lane.items()
            if key != "certificate_digest_sha256"
        }
    )
    checked_stale_lane = builder.build_role_aware_release_candidate(
        repo_root=REPO_ROOT,
        observed_at="2026-08-01T00:00:00Z",
        role_aware_certificate=stale_lane,
    )
    stale_lane_audit = checked_stale_lane["platform_support_audit"]
    assert stale_lane_audit["valid"] is True
    assert stale_lane_audit["live_artifact_failures"] == []

    for mutation in ("missing", "duplicate"):
        forged_population = copy.deepcopy(certificate)
        forged_runtime = _tools(forged_population)[
            "runtime-mtl-external"
        ]
        artifacts = forged_runtime["artifact_identities"]
        semantic_index = next(
            index
            for index, artifact in enumerate(artifacts)
            if artifact == semantic_artifact
        )
        if mutation == "missing":
            artifacts.pop(semantic_index)
        else:
            artifacts.append(copy.deepcopy(artifacts[semantic_index]))
        forged_population["certificate_digest_sha256"] = (
            certifier.content_digest(
                {
                    key: value
                    for key, value in forged_population.items()
                    if key != "certificate_digest_sha256"
                }
            )
        )
        checked_population = builder.build_role_aware_release_candidate(
            repo_root=REPO_ROOT,
            observed_at="2026-08-01T00:00:00Z",
            role_aware_certificate=forged_population,
        )
        population_audit = checked_population["platform_support_audit"]
        assert population_audit["valid"] is False
        assert (
            "semantic_tool_artifact_population_mismatch"
            in population_audit["failures"]
        )
        assert population_audit[
            "semantic_artifact_population_failures"
        ] == [
            "runtime-mtl-external:runtime_mtl_external:"
            "semantic_tool_artifact_population_mismatch"
        ]

    for scalar_field, forged_value in (
        ("executable_sha256", f"sha256:{'d' * 64}"),
        ("executable_artifact_class", "native_or_managed_binary"),
    ):
        forged_scalar = copy.deepcopy(certificate)
        forged_runtime = _tools(forged_scalar)[
            "runtime-mtl-external"
        ]
        forged_runtime[scalar_field] = forged_value
        forged_scalar["certificate_digest_sha256"] = (
            certifier.content_digest(
                {
                    key: value
                    for key, value in forged_scalar.items()
                    if key != "certificate_digest_sha256"
                }
            )
        )
        checked_scalar = builder.build_role_aware_release_candidate(
            repo_root=REPO_ROOT,
            observed_at="2026-08-01T00:00:00Z",
            role_aware_certificate=forged_scalar,
        )
        scalar_audit = checked_scalar["platform_support_audit"]
        assert scalar_audit["valid"] is False
        assert (
            "primary_executable_artifact_binding_mismatch"
            in scalar_audit["failures"]
        )
        assert scalar_audit["primary_executable_binding_failures"] == [
            "runtime-mtl-external:"
            "primary_executable_artifact_binding_mismatch"
        ]


def test_redacted_managed_executable_marker_resolves_only_by_exact_identity(
    certifier,
    builder,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    managed_root = tmp_path / "approved-theorem-provers"
    managed_bin = managed_root / "bin"
    managed_bin.mkdir(parents=True)
    executable = managed_bin / "vampire"
    executable.write_bytes(b"\x7fELF-focused-vampire-regression")
    executable.chmod(0o755)

    ambient_bin = tmp_path / "ambient-bin"
    ambient_bin.mkdir()
    ambient = ambient_bin / "vampire"
    ambient.write_bytes(b"\x7fELF-unapproved-ambient-vampire")
    ambient.chmod(0o755)
    monkeypatch.setenv(
        "IPFS_DATASETS_PY_THEOREM_PROVERS_ROOT",
        str(managed_root),
    )
    monkeypatch.setenv("PATH", str(ambient_bin))

    lock_entry = {
        "installer_plugin": "atp",
        "executable_candidates": ["vampire"],
    }
    artifact = {
        "kind": "semantic_executable",
        "path": "<host-path-redacted>/vampire",
        "sha256": certifier.file_digest(executable),
        "artifact_class": certifier.classify_executable_artifact(
            executable
        ),
    }
    matches = builder._matching_approved_redacted_executables(
        certifier=certifier,
        lock_entry=lock_entry,
        artifact=artifact,
    )
    assert matches == (executable.resolve(),)
    assert ambient.resolve() not in matches

    for field_name, forged_value in (
        ("sha256", "sha256:" + "f" * 64),
        ("artifact_class", "launcher_script"),
        ("path", "<host-path-redacted>/not-declared"),
    ):
        forged = {**artifact, field_name: forged_value}
        assert builder._matching_approved_redacted_executables(
            certifier=certifier,
            lock_entry=lock_entry,
            artifact=forged,
        ) == ()

    public = certifier.public_evidence_projection(
        {"path": str(matches[0])},
        repo_root=REPO_ROOT,
    )
    assert str(managed_root) not in json.dumps(public)
    assert public["path"] == "<host-path-redacted>/vampire"

    java_home = tmp_path / "managed-java"
    java_bin = java_home / "bin"
    java_bin.mkdir(parents=True)
    java = java_bin / "java"
    java.write_bytes(b"\x7fELF-focused-java-regression")
    java.chmod(0o755)
    monkeypatch.setenv("JAVA_HOME", str(java_home))
    java_artifact = {
        "kind": "executable",
        "path": "<host-path-redacted>/java",
        "sha256": certifier.file_digest(java),
        "artifact_class": certifier.classify_executable_artifact(java),
    }
    assert builder._matching_approved_redacted_executables(
        certifier=certifier,
        lock_entry={
            "tool_id": "java",
            "installer_plugin": "",
            "executable_candidates": ["java"],
        },
        artifact=java_artifact,
    ) == (java.resolve(),)


def test_candidate_interface_and_stage_ceiling(
    candidate: dict[str, Any],
) -> None:
    assert candidate["interface"] == INTERFACE
    assert candidate["goal_id"] == GOAL_ID
    assert candidate["task_id"] == TASK_ID
    assert candidate["schema_version"] == (
        "formal-verification-role-aware-release-candidate/v1"
    )
    assert candidate["binding_mode"] == "pre_merge_role_aware_release_candidate"
    assert candidate["readiness_stage"] in {"blocked", MAX_STAGE}
    assert candidate["readiness_stage"] != "deployment_ready"
    assert candidate["readiness_stage"] != "deployed"
    assert candidate["claims"]["merge"] is False
    assert candidate["claims"]["deployment"] is False
    assert candidate["claims"]["post_merge_attestation"] is False
    assert candidate["claims"]["self_referential_current_tree"] is False
    assert candidate["claims"]["max_stage"] == MAX_STAGE
    assert candidate["claims"]["merge_event_present"] is False
    assert candidate["ceilings"]["max_stage"] == MAX_STAGE
    assert candidate["ceilings"]["merge_event_present"] is False
    assert candidate["ceilings"]["deployment_claimed"] is False
    assert (
        candidate["ceilings"][
            "cannot_exceed_release_candidate_without_merge_event"
        ]
        is True
    )
    assert candidate["acceptance"]["merge_not_claimed"] is True
    assert candidate["acceptance"]["deployment_not_claimed"] is True
    assert candidate["acceptance"]["stage_at_most_release_candidate"] is True


def test_candidate_binds_exact_production_elevation_fanin(
    builder,
    certificate: dict[str, Any],
    candidate: dict[str, Any],
) -> None:
    live = builder.build_production_semantic_elevation_fanin(
        repo_root=REPO_ROOT,
        observed_at="2026-08-01T00:00:00Z",
        role_aware_certificate=certificate,
    )
    expected = builder.compact_production_elevation_fanin_binding(live)
    bound = candidate["production_semantic_elevation_fanin"]
    assert bound == expected
    assert bound["interface"] == FANIN_INTERFACE
    assert bound["task_id"] == FANIN_TASK_ID
    assert set(bound["required_tools"]) == REQUIRED_ELEVATIONS
    assert bound["receipt_digest_sha256"] == live[
        "receipt_digest_sha256"
    ]
    assert bound["tool_reconstruction_digests"] == expected[
        "tool_reconstruction_digests"
    ]
    assert bound["raw_receipt_embedded"] is False
    assert '"checks":[' not in json.dumps(
        bound,
        separators=(",", ":"),
    )
    checked = candidate[
        "checked_production_semantic_elevation_fanin"
    ]
    expected_bound = bool(
        bound["structurally_valid"] is True
        and bound["all_required_reconstructions_valid"] is True
        and bound["certificate_identity_valid"] is True
        and bound["checks_never_collapsed"] is True
        and bound["offline_only"] is True
        and checked["matches_live"] is True
    )
    assert candidate["acceptance"][
        "production_semantic_elevation_fanin_bound"
    ] is expected_bound
    assert candidate["readiness_requirements"][
        "production_semantic_elevation_fanin_bound"
    ] is expected_bound
    if FANIN_RECEIPT_PATH.is_file() and checked["matches_live"]:
        payload = json.loads(
            FANIN_RECEIPT_PATH.read_text(encoding="utf-8")
        )
        stored = payload.pop("receipt_digest_sha256")
        assert stored == builder.content_digest(payload)
        assert stored == bound["receipt_digest_sha256"]


def test_bad_outer_certificate_identity_blocks_fanin_and_candidate(
    builder,
    certificate: dict[str, Any],
) -> None:
    corrupted = copy.deepcopy(certificate)
    corrupted["certificate_digest_sha256"] = "0" * 64
    checked = builder.build_role_aware_release_candidate(
        repo_root=REPO_ROOT,
        observed_at="2026-08-01T00:00:00Z",
        role_aware_certificate=corrupted,
    )
    bound = checked["production_semantic_elevation_fanin"]
    assert bound["certificate_identity_valid"] is False
    assert bound["structurally_valid"] is False
    assert checked["acceptance"][
        "production_semantic_elevation_fanin_bound"
    ] is False
    assert checked["readiness_requirements"][
        "production_semantic_elevation_fanin_bound"
    ] is False
    assert checked["status"] == "role_aware_release_candidate_blocked"
    assert (
        "production_elevation_fanin:"
        "role_aware_certificate_identity_invalid"
        in checked["blockers"]
    )


def test_compact_pnmr_binding_mismatch_blocks_candidate(
    certifier,
    builder,
    certificate: dict[str, Any],
) -> None:
    corrupted = copy.deepcopy(certificate)
    lane = next(
        row
        for row in corrupted["semantic_lane_results"]
        if row.get("lane_id") == "kernel"
    )
    lane["per_tool"]["lean"]["check_set_digest_sha256"] = "f" * 64
    corrupted["certificate_digest_sha256"] = certifier.content_digest(
        {
            key: value
            for key, value in corrupted.items()
            if key != "certificate_digest_sha256"
        }
    )
    checked = builder.build_role_aware_release_candidate(
        repo_root=REPO_ROOT,
        observed_at="2026-08-01T00:00:00Z",
        role_aware_certificate=corrupted,
    )
    bound = checked["production_semantic_elevation_fanin"]
    assert bound["structurally_valid"] is False
    assert checked["acceptance"][
        "production_semantic_elevation_fanin_bound"
    ] is False
    assert (
        "production_elevation_fanin:"
        "required_pnmr_compact_binding_invalid"
        in checked["blockers"]
    )


def test_offline_policy_mutation_blocks_candidate_fanin(
    certifier,
    builder,
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
    checked = builder.build_role_aware_release_candidate(
        repo_root=REPO_ROOT,
        observed_at="2026-08-01T00:00:00Z",
        role_aware_certificate=corrupted,
    )
    bound = checked["production_semantic_elevation_fanin"]
    assert bound["offline_only"] is False
    assert bound["structurally_valid"] is False
    assert checked["acceptance"][
        "production_semantic_elevation_fanin_bound"
    ] is False
    assert "production_elevation_fanin:offline_policy_not_satisfied" in (
        checked["blockers"]
    )


def test_candidate_fanin_success_flags_are_consistent(
    candidate: dict[str, Any],
) -> None:
    bound = candidate["production_semantic_elevation_fanin"]
    if candidate["status"] == "role_aware_release_candidate_ready":
        assert bound["structurally_valid"] is True
        assert bound["all_required_reconstructions_valid"] is True
        assert bound["fanin_closed"] is True
        assert set(bound["production_elevation_present"]) == (
            REQUIRED_ELEVATIONS
        )
        assert candidate["acceptance"][
            "production_semantic_elevation_fanin_closed"
        ] is True
    if bound["fanin_closed"] is not True:
        assert candidate["readiness_requirements"][
            "production_semantic_elevation_fanin_closed"
        ] is False


def test_candidate_identity_binds_complete_body(
    builder, candidate: dict[str, Any]
) -> None:
    body = {
        key: value
        for key, value in candidate.items()
        if key != "candidate_identity"
    }
    assert candidate["candidate_identity"] == builder.content_digest(body)


def test_certificate_identity_and_release_candidate_hook(
    certifier, certificate: dict[str, Any]
) -> None:
    body = {
        key: value
        for key, value in certificate.items()
        if key != "certificate_digest_sha256"
    }
    assert certificate["certificate_digest_sha256"] == certifier.content_digest(
        body
    )
    release_hook = certificate["role_aware"]["release_candidate"]
    assert release_hook["interface"] == INTERFACE
    assert release_hook["goal_id"] == GOAL_ID
    assert release_hook["task_id"] == TASK_ID
    assert release_hook["max_stage"] == MAX_STAGE
    assert release_hook["claims_merge"] is False
    assert release_hook["claims_deployment"] is False
    fanin_hook = certificate["role_aware"][
        "production_semantic_elevation_fanin"
    ]
    assert fanin_hook["interface"] == FANIN_INTERFACE
    assert fanin_hook["goal_id"] == GOAL_ID
    assert fanin_hook["task_id"] == FANIN_TASK_ID
    assert fanin_hook["claims_merge"] is False
    assert fanin_hook["claims_deployment"] is False
    assert certificate["evidence"]["release_candidate_integration_test"].endswith(
        "test_formal_verification_role_aware_release_candidate.py"
    )
    assert certificate["evidence"][
        "production_elevation_fanin_integration_test"
    ].endswith("test_formal_verification_production_elevation_fanin.py")


def test_host_support_roles_ceilings_and_evidence_classes_are_derived(
    candidate: dict[str, Any],
    certificate: dict[str, Any],
) -> None:
    managed = certificate["managed_deployment_readiness"]
    assert candidate["host_support"]["host_platform"] == managed["host_platform"]
    assert candidate["host_support"]["derived_from"] == (
        "managed_deployment_readiness"
    )
    assert candidate["acceptance"]["host_support_derived"] is True

    assert candidate["roles"]["present"] is True
    assert candidate["roles"]["policy_digest_sha256"]
    assert candidate["acceptance"]["roles_bound"] is True

    assert candidate["evidence_classes"]
    assert candidate["acceptance"]["evidence_classes_derived"] is True
    assert set(candidate["evidence_classes"]) == {
        str(tool.get("evidence_class") or "unknown")
        for tool in certificate["tools"]
    }


def test_platform_exceptions_are_narrow_and_unsupported_only(
    candidate: dict[str, Any],
    certificate: dict[str, Any],
) -> None:
    managed = certificate["managed_deployment_readiness"]
    exceptions = candidate["platform_exceptions"]
    assert exceptions == managed["platform_exceptions"]
    supported = {
        row["tool_id"]
        for row in managed["platform_rows"]
        if row["managed"] and row["supported"]
    }
    exception_ids = {row["tool_id"] for row in exceptions}
    assert exception_ids.isdisjoint(supported)
    assert candidate["acceptance"]["platform_exceptions_derived_and_narrow"]
    assert candidate["acceptance"]["unsupported_only_as_narrow_exceptions"]
    for exception in exceptions:
        assert exception["narrow_scope"] is True
        assert exception["complete"] is False
        assert exception["production_certified"] is False
        assert exception["classification"] == "unsupported_here"


def test_supported_missing_tools_block_rather_than_exception(
    candidate: dict[str, Any],
    certificate: dict[str, Any],
) -> None:
    managed = certificate["managed_deployment_readiness"]
    blockers = {row["tool_id"] for row in managed["all_blockers"]}
    exceptions = {row["tool_id"] for row in managed["platform_exceptions"]}
    # Hyperproperty engines are supported managed capabilities that remain
    # installation/live-cert blockers when evidence is incomplete.
    for tool_id in ("hyperltl", "autohyper", "mchyper"):
        if tool_id in {
            row["tool_id"]
            for row in managed["platform_rows"]
            if row["managed"] and row["supported"]
        }:
            assert tool_id in blockers or managed["ready"] is True
            assert tool_id not in exceptions
    if not managed["ready"]:
        assert candidate["status"] == "role_aware_release_candidate_blocked"
        assert candidate["readiness_stage"] == "blocked"
        assert candidate["acceptance"][
            "supported_managed_capabilities_ready"
        ] is False
        assert any(
            item.startswith("supported_managed_capabilities_ready")
            or item.startswith("managed:")
            for item in candidate["blockers"]
        )


def test_offline_policy_and_quarantine_are_bound(
    candidate: dict[str, Any],
    certificate: dict[str, Any],
) -> None:
    policy = certificate["certification_policy"]
    assert candidate["offline_policy"]["satisfied"] is policy[
        "offline_policy_satisfied"
    ]
    assert candidate["offline_policy"]["forbid_install"] is policy[
        "forbid_install"
    ]
    assert candidate["offline_policy"]["forbid_download"] is policy[
        "forbid_download"
    ]
    assert candidate["offline_policy"]["forbid_network"] is policy[
        "forbid_network"
    ]
    assert candidate["acceptance"]["offline_policy_satisfied"] is True
    assert candidate["acceptance"]["no_install_during_offline_certification"]

    assert candidate["quarantine_state"]["bound"] is True
    assert candidate["quarantine_state"]["disagreement_quarantines"] == (
        certificate["disagreement_quarantines"]
    )
    assert candidate["acceptance"]["quarantine_state_bound"] is True


def test_public_surfaces_are_safe(
    candidate: dict[str, Any],
) -> None:
    text = json.dumps(candidate, sort_keys=True)
    assert "/home/" not in text
    assert "/tmp/" not in text
    assert "/private/tmp/" not in text
    assert candidate["public_evidence_policy"]["satisfied"] is True
    assert candidate["public_surfaces"]["bound"] is True
    assert candidate["acceptance"]["public_surfaces_bound"] is True


def test_digest_valid_certificate_cannot_forge_public_evidence_safety(
    certifier,
    builder,
    certificate: dict[str, Any],
) -> None:
    """The candidate must audit the full certificate, not trust its flag."""

    malicious = copy.deepcopy(certificate)
    malicious["forged_public_evidence"] = {
        "witness_path": "/home/private/secret-witness"
    }
    malicious["public_evidence_policy"] = {
        **malicious["public_evidence_policy"],
        "satisfied": True,
        "failures": [],
    }
    malicious["certificate_digest_sha256"] = certifier.content_digest(
        {
            key: value
            for key, value in malicious.items()
            if key != "certificate_digest_sha256"
        }
    )

    checked = builder.build_role_aware_release_candidate(
        repo_root=REPO_ROOT,
        observed_at="2026-08-01T00:00:00Z",
        role_aware_certificate=malicious,
    )

    assert checked["acceptance"]["role_aware_certificate_bound"] is True
    assert checked["public_surfaces"]["certificate_public_evidence_policy"][
        "declared"
    ]["satisfied"] is True
    recomputed = checked["public_surfaces"][
        "certificate_public_evidence_policy"
    ]["recomputed"]
    assert recomputed["satisfied"] is False
    assert "host_private_path" in recomputed["failures"]
    assert checked["acceptance"]["public_surfaces_bound"] is False
    assert checked["readiness_requirements"]["public_surfaces_bound"] is False
    assert checked["status"] == "role_aware_release_candidate_blocked"
    assert "public_surfaces_bound" in checked["blockers"]
    assert "/home/private/secret-witness" not in json.dumps(checked)


def test_candidate_identity_does_not_read_previous_candidate_content(
    builder,
    certificate: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regeneration is independent of whatever candidate was published before."""

    original_sha256_file = builder.sha256_file
    candidate_reads: list[Path] = []
    previous_digest = "sha256:" + ("1" * 64)

    def first_sha256_file(path: Path) -> str | None:
        resolved = Path(path).resolve()
        if resolved == CANDIDATE_PATH.resolve():
            candidate_reads.append(resolved)
            return previous_digest
        return original_sha256_file(path)

    monkeypatch.setattr(builder, "sha256_file", first_sha256_file)
    first = builder.build_role_aware_release_candidate(
        repo_root=REPO_ROOT,
        observed_at="2026-08-01T00:00:00Z",
        role_aware_certificate=certificate,
    )

    def second_sha256_file(path: Path) -> str | None:
        resolved = Path(path).resolve()
        if resolved == CANDIDATE_PATH.resolve():
            candidate_reads.append(resolved)
            return "sha256:" + ("2" * 64)
        return original_sha256_file(path)

    monkeypatch.setattr(builder, "sha256_file", second_sha256_file)
    second = builder.build_role_aware_release_candidate(
        repo_root=REPO_ROOT,
        observed_at="2026-08-01T00:00:00Z",
        role_aware_certificate=certificate,
    )

    assert candidate_reads == []
    assert first["candidate_identity"] == second["candidate_identity"]
    artifact = first["artifacts"]["release_candidate"]
    assert "content_identity_before_generation" not in artifact
    assert "present_before_generation" not in artifact
    assert artifact["previous_candidate_content_not_read"] is True


def test_synthetic_fixture_hermetic_advisor_shadow_cannot_promote(
    certificate: dict[str, Any],
    candidate: dict[str, Any],
) -> None:
    for tool in certificate["tools"]:
        if (
            tool["evidence_class"] in NON_AUTHORITATIVE_CLASSES
            or tool.get("executable_artifact_class")
            == "generated_hermetic_shim"
        ):
            assert tool["production_certified"] is False, tool["tool_id"]
    assert candidate["acceptance"][
        "synthetic_evidence_cannot_certify_production"
    ] is True
    assert candidate["ceilings"]["non_authoritative_promotions"] == []


def test_non_certifying_authority_roles_never_promote(
    certificate: dict[str, Any],
    candidate: dict[str, Any],
) -> None:
    tools = _tools(certificate)
    roles = certificate["authority_roles"]["tools"]
    for tool_id, role in roles.items():
        if not role["can_satisfy_certified_authority"] and tool_id in tools:
            assert tools[tool_id]["production_certified"] is False, tool_id
    assert candidate["acceptance"]["authority_ceiling_respected"] is True


def test_raw_check_and_artifact_digests_affect_identity(
    certifier,
    builder,
    certificate: dict[str, Any],
    candidate: dict[str, Any],
) -> None:
    # Certificate digest must change when a check is omitted.
    mutated_cert = copy.deepcopy(certificate)
    target = next(
        tool
        for tool in mutated_cert["tools"]
        if tool.get("checks")
    )
    target["checks"] = list(target["checks"])[:-1]
    original_body = {
        key: value
        for key, value in certificate.items()
        if key != "certificate_digest_sha256"
    }
    mutated_body = {
        key: value
        for key, value in mutated_cert.items()
        if key != "certificate_digest_sha256"
    }
    assert certifier.content_digest(mutated_body) != certifier.content_digest(
        original_body
    )

    # Candidate digest material records per-tool check digests.
    material = candidate["digest_material"]
    assert material["certificate_digest_sha256"] == certificate[
        "certificate_digest_sha256"
    ]
    assert material["tool_check_digests"]
    assert material["quarantine_digest"]
    assert material["lock_digest"]

    # Mutating a bound tool check digest changes candidate identity.
    mutated_candidate = copy.deepcopy(candidate)
    tool_id = next(iter(mutated_candidate["digest_material"]["tool_check_digests"]))
    mutated_candidate["digest_material"]["tool_check_digests"][tool_id] = "deadbeef"
    original_id = candidate["candidate_identity"]
    mutated_body = {
        key: value
        for key, value in mutated_candidate.items()
        if key != "candidate_identity"
    }
    assert builder.content_digest(mutated_body) != original_id


def test_omitted_semantic_check_changes_certificate_and_candidate(
    certifier,
    builder,
    certificate: dict[str, Any],
) -> None:
    results = certificate.get("semantic_lane_results") or []
    if not results:
        pytest.skip("no semantic lane results in this environment")
    ran = [
        row
        for row in results
        if row.get("status") == "ran"
        and isinstance(row.get("receipt"), dict)
        and row["receipt"].get("checks")
    ]
    if not ran:
        pytest.skip("no ran semantic lanes with checks")
    mutated = copy.deepcopy(certificate)
    lane = next(
        row
        for row in mutated["semantic_lane_results"]
        if row.get("lane_id") == ran[0]["lane_id"]
    )
    lane["receipt"]["checks"] = list(lane["receipt"]["checks"])[:-1]
    mutated["certificate_digest_sha256"] = certifier.content_digest(
        {
            key: value
            for key, value in mutated.items()
            if key != "certificate_digest_sha256"
        }
    )
    original_candidate = builder.build_role_aware_release_candidate(
        repo_root=REPO_ROOT,
        observed_at="2026-08-01T00:00:00Z",
        role_aware_certificate=certificate,
    )
    mutated_candidate = builder.build_role_aware_release_candidate(
        repo_root=REPO_ROOT,
        observed_at="2026-08-01T00:00:00Z",
        role_aware_certificate=mutated,
    )
    assert (
        mutated_candidate["candidate_identity"]
        != original_candidate["candidate_identity"]
    )
    mutated_fanin = mutated_candidate[
        "production_semantic_elevation_fanin"
    ]
    assert mutated_fanin["structurally_valid"] is False
    assert mutated_candidate["acceptance"][
        "production_semantic_elevation_fanin_bound"
    ] is False
    assert mutated_candidate["readiness_requirements"][
        "production_semantic_elevation_fanin_bound"
    ] is False
    assert mutated_candidate["status"] == (
        "role_aware_release_candidate_blocked"
    )


def test_forged_deployment_claim_is_rejected_by_stage_ceiling(
    builder,
    certificate: dict[str, Any],
) -> None:
    candidate = builder.build_role_aware_release_candidate(
        repo_root=REPO_ROOT,
        observed_at="2026-08-01T00:00:00Z",
        role_aware_certificate=certificate,
    )
    # Even if every readiness requirement were true, the published claims
    # section must never advertise merge/deployment.
    assert candidate["claims"]["max_stage"] == MAX_STAGE
    assert candidate["readiness_stage"] in {"blocked", MAX_STAGE}
    # Forging the claims after the fact breaks content identity.
    forged = copy.deepcopy(candidate)
    forged["claims"]["deployment"] = True
    forged["readiness_stage"] = "deployment_ready"
    forged.pop("candidate_identity", None)
    forged["candidate_identity"] = builder.content_digest(forged)
    # Rebuild from evidence cannot produce deployment_ready.
    rebuilt = builder.build_role_aware_release_candidate(
        repo_root=REPO_ROOT,
        observed_at="2026-08-01T00:00:00Z",
        role_aware_certificate=certificate,
    )
    assert rebuilt["readiness_stage"] != "deployment_ready"
    assert rebuilt["claims"]["deployment"] is False
    assert rebuilt["candidate_identity"] != forged["candidate_identity"]


def test_certified_source_commit_and_tree_are_bound(
    candidate: dict[str, Any],
) -> None:
    source = candidate["source"]
    assert source["model"] == "pre_merge_release_candidate_source/v1"
    candidate_in_source = subprocess.run(
        [
            "git",
            "-C",
            str(REPO_ROOT),
            "cat-file",
            "-e",
            (
                f"{source['certified_source_commit']}:"
                f"{CANDIDATE_PATH.relative_to(REPO_ROOT).as_posix()}"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    ).returncode == 0
    assert source["candidate_path_present_in_source_tree"] is (
        candidate_in_source
    )
    assert source["candidate_excluded_from_source_tree"] is (
        not candidate_in_source
    )
    assert (
        source["generated_candidate_identity_excluded_from_source_identity"]
        is True
    )
    assert (
        source["source_binding_uses_committed_tree_not_candidate_identity"]
        is True
    )
    assert source["self_referential_current_tree_claim_forbidden"] is True
    assert source["merge_event_required_to_exceed_release_candidate"] is True
    assert source["merge_event_present"] is False
    assert source["claims_own_future_merge"] is False
    assert source["claims_own_future_deployment"] is False
    if source["source_commit_bound"]:
        assert re.fullmatch(r"[0-9a-f]{40}", source["certified_source_commit"])
        assert re.fullmatch(r"[0-9a-f]{40}", source["certified_source_tree"])
        assert candidate["acceptance"]["certified_source_bound"] is True
    assert (
        "docs/architecture/formal_verification_role_aware_release_candidate.json"
        in source["attestation_paths"]
    )


def test_source_validity_allows_only_declared_generated_artifact_dirtiness(
    builder,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = {
        "certified_source_commit": "a" * 40,
        "certified_source_tree": "b" * 40,
        "datasets_gitlink": "c" * 40,
        "datasets_embedded_head": "c" * 40,
        "source_commit_bound": True,
        "tree_alignment": {},
    }
    candidate_relative = CANDIDATE_PATH.relative_to(REPO_ROOT).as_posix()
    completion_relative = (
        builder.DEFAULT_RECEIPT_RELATIVE.as_posix()
    )
    fanin_relative = (
        builder.DEFAULT_PRODUCTION_ELEVATION_FANIN_RECEIPT_RELATIVE.as_posix()
    )

    monkeypatch.setattr(
        builder,
        "build_source_attestation",
        lambda _repo_root: {
            **base,
            # This is the builder CLI's own generation order: completion,
            # production fan-in, then candidate.
            "dirty_paths_at_certification": [
                completion_relative,
                fanin_relative,
                candidate_relative,
            ],
        },
    )
    generated_only = builder.build_release_candidate_source_attestation(
        REPO_ROOT
    )
    assert generated_only["non_candidate_dirty_paths"] == []
    assert generated_only["valid_for_release_candidate"] is True

    monkeypatch.setattr(
        builder,
        "build_source_attestation",
        lambda _repo_root: {
            **base,
            "dirty_paths_at_certification": [
                completion_relative,
                fanin_relative,
                candidate_relative,
                BUILDER_PATH.relative_to(REPO_ROOT).as_posix(),
            ],
        },
    )
    source_dirty = builder.build_release_candidate_source_attestation(
        REPO_ROOT
    )
    assert source_dirty["non_candidate_dirty_paths"] == [
        BUILDER_PATH.relative_to(REPO_ROOT).as_posix()
    ]
    assert source_dirty["valid_for_release_candidate"] is False


def test_checked_in_candidate_is_content_addressed_and_not_false_ready(
    builder,
) -> None:
    checked = json.loads(CANDIDATE_PATH.read_text(encoding="utf-8"))
    assert checked["interface"] == INTERFACE
    assert checked["goal_id"] == GOAL_ID
    assert checked["task_id"] == TASK_ID
    assert CANDIDATE_PATH.stat().st_size < 1_000_000, (
        "checked-in candidate must stay under single-file admission budget"
    )
    stored = checked.pop("candidate_identity")
    assert stored == builder.content_digest(checked)
    assert checked["readiness_stage"] in {"blocked", MAX_STAGE}
    assert checked["readiness_stage"] != "deployment_ready"
    assert checked["claims"]["merge"] is False
    assert checked["claims"]["deployment"] is False
    checked_fanin = checked["checked_production_semantic_elevation_fanin"]
    assert checked_fanin["present"] is True
    assert checked_fanin["stored_digest_valid"] is True
    assert checked_fanin["identity_valid"] is True
    assert checked_fanin["matches_live"] is True
    if checked["status"] == "role_aware_release_candidate_ready":
        assert checked["readiness_stage"] == MAX_STAGE
        assert all(checked["readiness_requirements"].values())
    else:
        assert checked["status"] == "role_aware_release_candidate_blocked"
        assert checked["blockers"]


def test_required_elevations_are_disclosed(
    candidate: dict[str, Any],
) -> None:
    assert candidate["required_elevation_audit"]["valid"] is True
    assert candidate["platform_support_audit"]["valid"] is True
    assert candidate["required_elevation_audit"][
        "expected_global_production_certified_tool_ids"
    ] == candidate["elevations"]["production_certified_tool_ids"]
    assert set(candidate["elevations"]["required"]) == REQUIRED_ELEVATIONS
    missing = set(candidate["elevations"]["missing_required"])
    assert missing <= REQUIRED_ELEVATIONS
    # Until specialized elevation goals complete, elevations remain open.
    if missing:
        assert candidate["acceptance"][
            "required_semantic_elevations_present"
        ] is False


def test_semantic_lanes_bind_canonical_receipts_and_compact_check_sets(
    certifier,
    certificate: dict[str, Any],
    candidate: dict[str, Any],
) -> None:
    candidate_lanes = {
        row["lane_id"]: row
        for row in candidate["role_aware_certificate"][
            "semantic_lane_results"
        ]
    }
    for result in certificate.get("semantic_lane_results") or []:
        if result.get("status") != "ran":
            continue
        receipt = result.get("receipt")
        assert isinstance(receipt, dict)
        assert result["digest_sha256"] == certifier.content_digest(receipt)
        for tool_id, per_tool in (result.get("per_tool") or {}).items():
            assert "checks" not in per_tool
            assert REQUIRED_CHECK_KINDS <= set(
                per_tool["check_kinds_present"]
            )
            assert re.fullmatch(
                r"[0-9a-f]{64}",
                per_tool["check_set_digest_sha256"],
            )
            compact_tool = candidate_lanes[result["lane_id"]][
                "per_tool_bindings"
            ][tool_id]
            assert compact_tool["check_set_digest_sha256"] == (
                per_tool["check_set_digest_sha256"]
            )
            assert compact_tool["tool_evidence_digest_sha256"] == (
                certifier.content_digest(per_tool)
            )
    assert candidate["acceptance"]["semantic_receipts_full_and_bound"] is False
    # Hyperproperty lane now runs; residual vendor/managed installation still
    # blocks readiness (sealed root auth and managed pins remain incomplete).
    assert any(
        "hyperltl" in blocker or "autohyper" in blocker or "mchyper" in blocker
        for blocker in candidate["blockers"]
    )


def test_builder_constants_align_with_goal_packet(builder) -> None:
    assert builder.RELEASE_CANDIDATE_INTERFACE == INTERFACE
    assert builder.RELEASE_CANDIDATE_GOAL_ID == GOAL_ID
    assert builder.RELEASE_CANDIDATE_TASK_ID == TASK_ID
    assert builder.RELEASE_CANDIDATE_MAX_STAGE == MAX_STAGE
    assert builder.DEFAULT_RELEASE_CANDIDATE_RELATIVE.as_posix() == (
        "docs/architecture/formal_verification_role_aware_release_candidate.json"
    )
    assert builder.DEFAULT_RELEASE_CANDIDATE_TEST_RELATIVE.as_posix() == (
        "test/integration/test_formal_verification_role_aware_release_candidate.py"
    )
    assert builder.PRODUCTION_ELEVATION_FANIN_INTERFACE == FANIN_INTERFACE
    assert builder.PRODUCTION_ELEVATION_FANIN_GOAL_ID == GOAL_ID
    assert builder.PRODUCTION_ELEVATION_FANIN_TASK_ID == FANIN_TASK_ID
    assert (
        builder.DEFAULT_PRODUCTION_ELEVATION_FANIN_RECEIPT_RELATIVE.as_posix()
        == "docs/architecture/"
        "formal_verification_production_elevation_fanin_receipt.json"
    )
    assert (
        builder.DEFAULT_PRODUCTION_ELEVATION_FANIN_TEST_RELATIVE.as_posix()
        == "test/integration/toolchains/"
        "test_formal_verification_production_elevation_fanin.py"
    )
