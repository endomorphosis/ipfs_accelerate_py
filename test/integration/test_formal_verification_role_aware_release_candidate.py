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

INTERFACE = "RoleAwareFormalVerificationReleaseCandidate@1"
GOAL_ID = "FVT-G213"
TASK_ID = "FVT-066"
MAX_STAGE = "release_candidate"

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
def certificate(certifier) -> dict[str, Any]:
    return certifier.build_certificate(repo_root=REPO_ROOT, role_aware=True)


@pytest.fixture(scope="module")
def candidate(builder, certificate) -> dict[str, Any]:
    return builder.build_role_aware_release_candidate(
        repo_root=REPO_ROOT,
        observed_at="2026-08-01T00:00:00Z",
        role_aware_certificate=certificate,
    )


def _tools(certificate: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["tool_id"]: row for row in certificate["tools"]}


def test_expected_outputs_exist_and_candidate_is_tracked_evidence() -> None:
    for path in (CERTIFIER_PATH, BUILDER_PATH, CANDIDATE_PATH):
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
    assert "specialized_by_handler" not in bound.get(
        "specialized_receipt_aggregation", {}
    )
    for tool in bound.get("tools") or []:
        assert "checks" not in tool
        assert tool.get("checks_digest_sha256")


def test_compact_projection_retains_lane_tool_and_handler_digest_bindings(
    certifier,
    candidate: dict[str, Any],
    certificate: dict[str, Any],
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

    full_specialized = certificate["specialized_receipt_aggregation"]
    compact_specialized = bound["specialized_receipt_aggregation"]
    assert compact_specialized["aggregation_digest_sha256"] == (
        full_specialized["aggregation_digest_sha256"]
    )
    assert compact_specialized["aggregation_digest_valid"] is True
    full_handlers = full_specialized["specialized_by_handler"]
    assert set(compact_specialized["handlers"]) == set(full_handlers)
    for handler_key, handler in full_handlers.items():
        compact_handler = compact_specialized["handlers"][handler_key]
        assert compact_handler["tool_evidence_digest_sha256"] == handler[
            "tool_evidence_digest_sha256"
        ]
        assert compact_handler["tool_evidence_digest_valid"] is True
        assert compact_handler["check_set_digest_sha256"] == handler[
            "check_set_digest_sha256"
        ]
        assert compact_handler["raw_receipt_digest"] == handler[
            "raw_receipt_digest"
        ]


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
    assert certificate["evidence"]["release_candidate_integration_test"].endswith(
        "test_formal_verification_role_aware_release_candidate.py"
    )


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

    monkeypatch.setattr(
        builder,
        "build_source_attestation",
        lambda _repo_root: {
            **base,
            "dirty_paths_at_certification": [candidate_relative],
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
    if checked["status"] == "role_aware_release_candidate_ready":
        assert checked["readiness_stage"] == MAX_STAGE
        assert all(checked["readiness_requirements"].values())
    else:
        assert checked["status"] == "role_aware_release_candidate_blocked"
        assert checked["blockers"]


def test_required_elevations_are_disclosed(
    candidate: dict[str, Any],
) -> None:
    assert set(candidate["elevations"]["required"]) == REQUIRED_ELEVATIONS
    missing = set(candidate["elevations"]["missing_required"])
    assert missing <= REQUIRED_ELEVATIONS
    # Until specialized elevation goals complete, elevations remain open.
    if missing:
        assert candidate["acceptance"][
            "required_semantic_elevations_present"
        ] is False


def test_semantic_lanes_retain_full_check_sets_when_ran(
    certificate: dict[str, Any],
    candidate: dict[str, Any],
) -> None:
    for result in certificate.get("semantic_lane_results") or []:
        if result.get("status") != "ran":
            continue
        for tool_id, per_tool in (result.get("per_tool") or {}).items():
            checks = per_tool.get("checks") or []
            assert REQUIRED_CHECK_KINDS <= {check["kind"] for check in checks}, (
                tool_id
            )
    assert candidate["acceptance"]["semantic_receipts_full_and_bound"] is (
        bool(certificate.get("semantic_lane_results"))
        or not certificate["role_aware"]["enabled"]
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
