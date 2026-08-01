"""Production-semantic elevation fan-in (FVT-081 / FVT-G213).

``ProductionSemanticElevationFanIn@1``

For each required baseline elevation tool (lean, runtime-mtl,
datalog-authorization, secpal-authorization, coq, isabelle):

* independently reconstruct positive, negative, mutation, and replay evidence
  from the bound semantic lane receipt;
* allow production elevation only when that reconstruction is complete and the
  lane's ``production_elevation_allowed`` gate is true;
* fail closed when elevation surfaces claim production without reconstruction,
  when evidence classes cannot satisfy production authority, or when checks are
  collapsed / hardcoded.

The fan-in may be structurally valid while elevations remain missing; it never
claims merge or deployment.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
CERTIFIER_PATH = (
    REPO_ROOT / "tools" / "logic" / "certify_formal_verification_toolchains.py"
)
BUILDER_PATH = (
    REPO_ROOT / "tools" / "logic" / "build_formal_verification_tactician_receipt.py"
)
RECEIPT_PATH = (
    REPO_ROOT
    / "docs"
    / "architecture"
    / "formal_verification_production_elevation_fanin_receipt.json"
)
CANDIDATE_PATH = (
    REPO_ROOT
    / "docs"
    / "architecture"
    / "formal_verification_role_aware_release_candidate.json"
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
REQUIRED_CHECK_KINDS = {"positive", "negative", "mutation", "replay"}
VALIDATION_COMMAND = (
    "PYTHONPATH=ipfs_datasets_py python -m pytest "
    "test/integration/toolchains/test_formal_verification_production_elevation_fanin.py "
    "test/integration/test_formal_verification_role_aware_release_candidate.py "
    "test/integration/test_formal_verification_real_tool_matrix.py -q"
)


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
    return _load(CERTIFIER_PATH, "fvt_production_elevation_fanin_certifier")


@pytest.fixture(scope="module")
def builder():
    return _load(BUILDER_PATH, "fvt_production_elevation_fanin_builder")


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


@pytest.fixture(scope="module")
def candidate(builder, certificate, certificate_bundle) -> dict[str, Any]:
    return builder.build_role_aware_release_candidate(
        repo_root=REPO_ROOT,
        observed_at="2026-08-01T00:00:00Z",
        role_aware_certificate=certificate,
        source_specialized_receipt_aggregation=certificate_bundle[1].get(
            "specialized_receipt_aggregation"
        ),
    )


# ---------------------------------------------------------------------------
# Expected outputs / constants
# ---------------------------------------------------------------------------


def test_expected_outputs_exist() -> None:
    for path in (
        CERTIFIER_PATH,
        BUILDER_PATH,
        RECEIPT_PATH,
        Path(__file__),
        CANDIDATE_PATH,
    ):
        assert path.is_file(), path


def test_builder_and_certifier_constants(builder, certifier) -> None:
    assert builder.PRODUCTION_ELEVATION_FANIN_INTERFACE == INTERFACE
    assert builder.PRODUCTION_ELEVATION_FANIN_SCHEMA_VERSION == SCHEMA
    assert builder.PRODUCTION_ELEVATION_FANIN_GOAL_ID == GOAL_ID
    assert builder.PRODUCTION_ELEVATION_FANIN_TASK_ID == TASK_ID
    assert set(builder.PRODUCTION_ELEVATION_REQUIRED_CHECK_KINDS) == (
        REQUIRED_CHECK_KINDS
    )
    assert builder.DEFAULT_PRODUCTION_ELEVATION_FANIN_RECEIPT_RELATIVE.as_posix() == (
        "docs/architecture/formal_verification_production_elevation_fanin_receipt.json"
    )
    assert builder.DEFAULT_PRODUCTION_ELEVATION_FANIN_TEST_RELATIVE.as_posix() == (
        "test/integration/toolchains/test_formal_verification_production_elevation_fanin.py"
    )
    assert builder.PRODUCTION_ELEVATION_FANIN_VALIDATION_COMMAND == (
        VALIDATION_COMMAND
    )
    assert certifier.PRODUCTION_ELEVATION_FANIN_INTERFACE == INTERFACE
    assert certifier.PRODUCTION_ELEVATION_FANIN_GOAL_ID == GOAL_ID
    assert certifier.PRODUCTION_ELEVATION_FANIN_TASK_ID == TASK_ID


def test_checked_in_receipt_schema_and_identity(builder) -> None:
    payload = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    assert payload["schema_version"] == SCHEMA
    assert payload["interface"] == INTERFACE
    assert payload["goal_id"] == GOAL_ID
    assert payload["task_id"] == TASK_ID
    assert payload["program"] == (
        "formal-verification-tactician/toolchain-release-candidate"
    )
    assert list(payload["required_tools"]) == list(REQUIRED_TOOLS)
    assert set(payload["policy"]["required_check_kinds"]) == REQUIRED_CHECK_KINDS
    assert payload["policy"][
        "independent_reconstruction_required_before_production_elevation"
    ] is True
    assert payload["policy"]["no_install"] is True
    assert payload["policy"]["hardcoded_success_forbidden"] is True
    assert payload["claims"]["merge"] is False
    assert payload["claims"]["deployment"] is False
    assert payload["evidence"]["integration_test"].endswith(
        "test_formal_verification_production_elevation_fanin.py"
    )
    assert payload["evidence"]["validation_command"] == VALIDATION_COMMAND
    body = {
        key: value
        for key, value in payload.items()
        if key != "receipt_digest_sha256"
    }
    assert payload["receipt_digest_sha256"] == builder.content_digest(body)
    assert RECEIPT_PATH.stat().st_size < 1_000_000


# ---------------------------------------------------------------------------
# Live fan-in reconstruction gates
# ---------------------------------------------------------------------------


def test_fanin_interface_and_structural_validity(fanin: dict[str, Any]) -> None:
    assert fanin["interface"] == INTERFACE
    assert fanin["schema_version"] == SCHEMA
    assert fanin["goal_id"] == GOAL_ID
    assert fanin["task_id"] == TASK_ID
    assert list(fanin["required_tools"]) == list(REQUIRED_TOOLS)
    assert fanin["acceptance"]["required_tools_population_exact"] is True
    assert fanin["acceptance"]["checks_never_collapsed"] is True
    assert fanin["acceptance"]["raw_checks_not_reembedded"] is True
    assert fanin["acceptance"]["merge_not_claimed"] is True
    assert fanin["acceptance"]["deployment_not_claimed"] is True
    assert fanin["acceptance"]["no_elevation_without_reconstruction"] is True
    assert fanin["acceptance"]["production_elevation_allowed_respected"] is True
    assert fanin["summary"]["structurally_valid"] is True
    assert fanin["summary"]["failures"] == []
    assert fanin["status"] in {
        "production_semantic_elevation_fanin_structurally_valid",
        "production_semantic_elevation_fanin_closed",
    }
    # Until specialized production elevation goals complete, elevations remain
    # open and the fan-in stays structurally valid without being closed.
    if fanin["summary"]["production_elevation_missing"]:
        assert fanin["summary"]["fanin_closed"] is False
        assert fanin["status"] == (
            "production_semantic_elevation_fanin_structurally_valid"
        )


def test_each_required_tool_has_independent_pnmr_surface(
    fanin: dict[str, Any],
    certificate: dict[str, Any],
    certifier,
) -> None:
    lanes = {
        str(row.get("lane_id")): row
        for row in certificate.get("semantic_lane_results") or []
    }
    tools = fanin["tools"]
    assert set(tools) == set(REQUIRED_TOOLS)
    for tool_id in REQUIRED_TOOLS:
        row = tools[tool_id]
        assert row["tool_id"] == tool_id
        assert row["lane_id"]
        reconstruction = row["independent_reconstruction"]
        assert set(reconstruction["required_check_kinds"]) == REQUIRED_CHECK_KINDS
        assert reconstruction["raw_checks_embedded"] is False
        assert "checks" not in reconstruction

        lane = lanes.get(row["lane_id"]) or {}
        if lane.get("status") == "ran":
            recomputed = certifier.recompute_semantic_tool_check_binding(
                lane,
                tool_id,
            )
            assert reconstruction["recompute_valid"] is True
            assert reconstruction["check_set_digest_sha256"] == (
                recomputed["check_set_digest_sha256"]
            )
            assert set(reconstruction["check_kinds_present"]) == set(
                recomputed["check_kinds_present"]
            )
            assert reconstruction["required_kinds_present"] is (
                REQUIRED_CHECK_KINDS
                <= set(recomputed["check_kinds_present"])
            )
            if reconstruction["required_kinds_all_passed"]:
                for kind in REQUIRED_CHECK_KINDS:
                    assert any(
                        str(check.get("kind")) == kind
                        and str(check.get("status")) == "passed"
                        for check in recomputed.get("checks") or []
                    )
            assert reconstruction["valid"] is (
                reconstruction["required_kinds_present"]
                and reconstruction["required_kinds_all_passed"]
                and bool(reconstruction["check_set_digest_sha256"])
            )
        else:
            assert reconstruction["valid"] is False
            assert "semantic_lane_not_run" in (
                reconstruction.get("recompute_failure") or ""
            ) or "semantic_lane_not_run" in row["block_reasons"]

        # Production elevation may only be present with full reconstruction
        # under an allowed production class.
        if row["production_elevation_present"]:
            assert reconstruction["valid"] is True
            assert row["production_elevation_allowed"] is True
            assert row["eligible_for_production_elevation"] is True
        if row["production_elevation_present"] and not reconstruction["valid"]:
            pytest.fail(
                f"{tool_id}: production elevation without independent PNMR"
            )


def test_production_elevation_gate_respects_evidence_class(
    fanin: dict[str, Any],
) -> None:
    for tool_id, row in fanin["tools"].items():
        if not row["production_elevation_allowed"]:
            assert row["production_elevation_present"] is False
            assert "production_elevation_not_allowed_by_evidence_class" in (
                row["block_reasons"]
            ) or row["lane_status"] != "ran"
        # Surface consistency: no partial production claims.
        assert row["surfaces_consistent"] is True or not any(
            row["surfaces"].values()
        )
        if any(row["surfaces"].values()) and not all(row["surfaces"].values()):
            pytest.fail(f"{tool_id}: inconsistent elevation surfaces")


def test_digest_identity_and_compactness(
    builder,
    fanin: dict[str, Any],
) -> None:
    body = {
        key: value
        for key, value in fanin.items()
        if key != "receipt_digest_sha256"
    }
    assert fanin["receipt_digest_sha256"] == builder.content_digest(body)
    encoded = json.dumps(fanin, separators=(",", ":"), ensure_ascii=False)
    assert len(encoded.encode("utf-8")) < 250_000
    # Never re-embed bulk check lists on the fan-in surface.
    text = encoded
    assert '"checks":[' not in text
    assert fanin["role_aware_certificate"]["raw_certificate_embedded"] is False


def test_mutating_reconstruction_digest_changes_identity(
    builder,
    certifier,
    certificate: dict[str, Any],
    fanin: dict[str, Any],
) -> None:
    mutated = copy.deepcopy(certificate)
    target_lane = None
    target_tool = None
    for tool_id in REQUIRED_TOOLS:
        row = fanin["tools"][tool_id]
        if row["independent_reconstruction"]["valid"]:
            target_tool = tool_id
            target_lane = row["lane_id"]
            break
    if target_lane is None or target_tool is None:
        pytest.skip("no reconstructed required tool available in this environment")

    lane = next(
        item
        for item in mutated["semantic_lane_results"]
        if item.get("lane_id") == target_lane
    )
    receipt = lane.get("receipt")
    assert isinstance(receipt, dict)
    # Drop one required-kind check from the raw receipt when present.
    checks = receipt.get("checks")
    engines = receipt.get("engines")
    mutated_any = False
    if isinstance(checks, list) and checks:
        for index, check in enumerate(list(checks)):
            if str(check.get("kind") or "") in REQUIRED_CHECK_KINDS:
                checks.pop(index)
                mutated_any = True
                break
        if not mutated_any:
            checks.pop()
            mutated_any = True
    elif isinstance(engines, list):
        for engine in engines:
            if str(engine.get("engine_id") or "") != target_tool:
                continue
            engine_checks = engine.get("checks")
            if isinstance(engine_checks, list) and engine_checks:
                for index, check in enumerate(list(engine_checks)):
                    if str(check.get("kind") or "") in REQUIRED_CHECK_KINDS:
                        engine_checks.pop(index)
                        mutated_any = True
                        break
                if not mutated_any:
                    engine_checks.pop()
                    mutated_any = True
            break
    if not mutated_any:
        pytest.skip("unable to mutate required-kind checks for target tool")

    for field_name in (
        "receipt_digest_sha256",
        "certificate_digest_sha256",
        "digest_sha256",
    ):
        if field_name in receipt:
            receipt[field_name] = certifier.content_digest(
                {
                    key: value
                    for key, value in receipt.items()
                    if key != field_name
                }
            )
    lane["digest_sha256"] = certifier.content_digest(receipt)
    mutated["certificate_digest_sha256"] = certifier.content_digest(
        {
            key: value
            for key, value in mutated.items()
            if key != "certificate_digest_sha256"
        }
    )
    mutated_fanin = builder.build_production_semantic_elevation_fanin(
        repo_root=REPO_ROOT,
        observed_at="2026-08-01T00:00:00Z",
        role_aware_certificate=mutated,
    )
    assert (
        mutated_fanin["receipt_digest_sha256"]
        != fanin["receipt_digest_sha256"]
    )
    mutated_row = mutated_fanin["tools"][target_tool]
    original_row = fanin["tools"][target_tool]
    assert (
        mutated_row["independent_reconstruction"]["check_set_digest_sha256"]
        != original_row["independent_reconstruction"]["check_set_digest_sha256"]
    )


def test_forged_production_elevation_without_reconstruction_fails_closed(
    builder,
    certifier,
    certificate: dict[str, Any],
) -> None:
    forged = copy.deepcopy(certificate)
    tool = next(
        row
        for row in forged["tools"]
        if row.get("tool_id") == "lean"
    )
    tool["production_certified"] = True
    promotion = forged.setdefault("promotion", {})
    production_ids = list(promotion.get("production_certified_tool_ids") or [])
    if "lean" not in production_ids:
        production_ids.append("lean")
        production_ids.sort()
    promotion["production_certified_tool_ids"] = production_ids
    role_aware = forged.setdefault("role_aware", {})
    elevated = list(role_aware.get("elevated_tool_ids") or [])
    if "lean" not in elevated:
        elevated.append("lean")
        elevated.sort()
    role_aware["elevated_tool_ids"] = elevated
    elevations = list(role_aware.get("elevations") or [])
    elevations.append(
        {
            "tool_id": "lean",
            "lane_id": "kernel",
            "elevated": True,
            "evidence_class": "forged",
        }
    )
    role_aware["elevations"] = elevations
    lane = next(
        row
        for row in forged["semantic_lane_results"]
        if row.get("lane_id") == "kernel"
    )
    lane["elevated_tool_ids"] = ["lean"]
    # Force reconstruction invalid by stripping the receipt entirely after
    # claiming elevation surfaces.
    lane["status"] = "not_run"
    lane["receipt"] = None
    lane["digest_sha256"] = None
    lane["per_tool"] = {}
    lane["certified"] = False
    forged["certificate_digest_sha256"] = certifier.content_digest(
        {
            key: value
            for key, value in forged.items()
            if key != "certificate_digest_sha256"
        }
    )
    checked = builder.build_production_semantic_elevation_fanin(
        repo_root=REPO_ROOT,
        observed_at="2026-08-01T00:00:00Z",
        role_aware_certificate=forged,
    )
    assert checked["summary"]["structurally_valid"] is False
    assert checked["acceptance"]["no_elevation_without_reconstruction"] is False
    lean_row = checked["tools"]["lean"]
    assert "elevation_without_independent_reconstruction" in (
        lean_row["block_reasons"]
    ) or any(
        "elevation_without_independent_reconstruction" in failure
        for failure in checked["summary"]["failures"]
    )
    assert checked["status"] == "production_semantic_elevation_fanin_blocked"


def test_release_candidate_binds_compact_fanin(
    candidate: dict[str, Any],
    fanin: dict[str, Any],
    builder,
) -> None:
    bound = candidate["production_semantic_elevation_fanin"]
    expected = builder.compact_production_elevation_fanin_binding(fanin)
    assert bound["interface"] == INTERFACE
    assert bound["goal_id"] == GOAL_ID
    assert bound["task_id"] == TASK_ID
    assert bound["receipt_digest_sha256"] == fanin["receipt_digest_sha256"]
    assert bound["structurally_valid"] is True
    assert bound["raw_receipt_embedded"] is False
    assert bound["path"].endswith(
        "formal_verification_production_elevation_fanin_receipt.json"
    )
    assert set(bound["required_tools"]) == set(REQUIRED_TOOLS)
    assert bound["tool_reconstruction_digests"] == (
        expected["tool_reconstruction_digests"]
    )
    assert candidate["acceptance"][
        "production_semantic_elevation_fanin_bound"
    ] is True
    assert candidate["acceptance"][
        "production_elevation_requires_independent_pnmr"
    ] is True
    assert candidate["readiness_requirements"][
        "production_semantic_elevation_fanin_bound"
    ] is True


def test_public_surfaces_have_no_host_private_paths(fanin: dict[str, Any]) -> None:
    text = json.dumps(fanin, sort_keys=True)
    assert "/home/" not in text
    assert "/tmp/" not in text
    assert "/private/tmp/" not in text
    assert fanin["public_evidence_policy"]["satisfied"] is True
    digest = str(fanin["receipt_digest_sha256"] or "")
    if digest.startswith("sha256:"):
        digest = digest[len("sha256:") :]
    assert re.fullmatch(r"[0-9a-f]{64}", digest)
