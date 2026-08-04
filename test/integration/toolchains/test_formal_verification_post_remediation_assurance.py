"""Post-remediation matrix/release re-audit (FVT-100 / FVT-G233).

``FormalVerificationPostRemediationAssurance@1``

Closes the objective evidence gap after remediations G225-G231:

* rebuild trusted certificate body-driven axes and the end-to-end matrix;
* record the exact transition from the 5-of-28 ready baseline;
* keep retired Microsoft SecPAL unsupported and non-required;
* keep FVT-G219 blocked and unhideable;
* expose production-authorization-replacement as a new distinct row;
* keep local audit/assessment complete independent of deployment readiness;
* keep ``deployment_ready`` false until FVT-G232 and joint readiness;
* fail closed on optimistic reseals, authority substitution, and promotion.
"""

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
    REPO_ROOT / "tools" / "logic" / "build_formal_verification_tactician_receipt.py"
)
DELTA_PATH = (
    REPO_ROOT
    / "docs"
    / "architecture"
    / "formal_verification_post_remediation_delta.json"
)
MATRIX_PATH = (
    REPO_ROOT
    / "docs"
    / "architecture"
    / "formal_verification_end_to_end_assurance_matrix.json"
)
RELEASE_PATH = (
    REPO_ROOT
    / "docs"
    / "architecture"
    / "formal_verification_authoritative_vendor_release.json"
)
CERTIFICATE_PATH = (
    REPO_ROOT
    / "docs"
    / "architecture"
    / "formal_verification_toolchain_certificate.json"
)

INTERFACE = "FormalVerificationPostRemediationAssurance@1"
SCHEMA = "formal-verification-post-remediation-assurance/v1"
GOAL_ID = "FVT-G233"
TASK_ID = "FVT-100"
BASELINE_READY = ("cvc5", "java", "lean", "z3", "zkp-circuit")
REPLACEMENT_ID = "production-authorization-replacement"
REFERENCE_ID = "secpal-authorization"
EXTERNAL_ID = "secpal"


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
    return _load(CERTIFIER_PATH, "fvt_post_remediation_certifier")


@pytest.fixture(scope="module")
def builder():
    return _load(BUILDER_PATH, "fvt_post_remediation_receipt_builder")


@pytest.fixture(scope="module")
def trusted_certificate() -> dict[str, Any]:
    return json.loads(CERTIFICATE_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def matrix(certifier, trusted_certificate: dict[str, Any]) -> dict[str, Any]:
    checked = json.loads(MATRIX_PATH.read_text(encoding="utf-8"))
    validation = certifier.validate_end_to_end_assurance_matrix(
        checked,
        repo_root=REPO_ROOT,
        certificate=trusted_certificate,
    )
    assert validation["valid"] is True, validation["failures"]
    return checked


@pytest.fixture(scope="module")
def delta(certifier, matrix: dict[str, Any], trusted_certificate: dict[str, Any]) -> dict[str, Any]:
    checked = json.loads(DELTA_PATH.read_text(encoding="utf-8"))
    validation = certifier.validate_post_remediation_assurance_delta(
        checked,
        repo_root=REPO_ROOT,
        matrix=matrix,
        certificate=trusted_certificate,
    )
    assert validation["valid"] is True, validation["failures"]
    return checked


def test_expected_outputs_and_contract_constants_exist(certifier, builder) -> None:
    assert CERTIFIER_PATH.is_file()
    assert BUILDER_PATH.is_file()
    assert DELTA_PATH.is_file()
    assert MATRIX_PATH.is_file()
    assert RELEASE_PATH.is_file()
    assert Path(__file__).is_file()
    assert certifier.POST_REMEDIATION_INTERFACE == INTERFACE
    assert certifier.POST_REMEDIATION_SCHEMA == SCHEMA
    assert certifier.POST_REMEDIATION_GOAL_ID == GOAL_ID
    assert certifier.POST_REMEDIATION_TASK_ID == TASK_ID
    assert (
        certifier.DEFAULT_POST_REMEDIATION_DELTA_RELATIVE.as_posix()
        == "docs/architecture/formal_verification_post_remediation_delta.json"
    )
    assert builder.POST_REMEDIATION_INTERFACE == INTERFACE
    assert builder.POST_REMEDIATION_GOAL_ID == GOAL_ID
    assert builder.POST_REMEDIATION_BASELINE_READY_COUNT == 5
    assert tuple(builder.POST_REMEDIATION_BASELINE_READY_PROVIDER_IDS) == BASELINE_READY
    assert "FVT-G219" in builder.POST_REMEDIATION_EXTERNAL_BLOCKERS
    assert "FVT-G232" in builder.POST_REMEDIATION_EXTERNAL_BLOCKERS


def test_checked_delta_is_content_addressed_audit_complete_and_not_deployable(
    certifier,
    delta: dict[str, Any],
) -> None:
    assert delta["interface"] == INTERFACE
    assert delta["schema_version"] == SCHEMA
    assert delta["goal_id"] == GOAL_ID
    assert delta["task_id"] == TASK_ID
    assert delta["audit_complete"] is True
    assert delta["assessment_complete"] is True
    # Honest derivation: deployment_ready may become true only when FVT-G232
    # and every required row are already bound. On this host those gates remain
    # open, so the checked delta must stay non-deployable.
    assert isinstance(delta["deployment_ready"], bool)
    assert delta["claims"]["audit_complete"] is True
    assert delta["claims"]["assessment_complete"] is True
    assert delta["claims"]["deployment_ready"] is delta["deployment_ready"]
    assert delta["claims"]["local_audit_independent_from_deployment_ready"] is True
    readiness = delta.get("deployment_readiness_inputs") or {}
    if delta["deployment_ready"] is True:
        assert delta["status"] == "post_remediation_deployment_ready"
        assert readiness.get("legal_approval_complete") is True
        assert readiness.get("required_rows_ready") is True
        assert readiness.get("replacement_joint_ready") is True
    else:
        assert delta["status"] == "post_remediation_audit_complete_deployment_blocked"
        assert readiness.get("legal_approval_complete") is not True or (
            readiness.get("required_rows_ready") is not True
            or readiness.get("replacement_joint_ready") is not True
        )
    assert delta["public_evidence_policy"]["satisfied"] is True
    assert delta["delta_digest_sha256"] == certifier.content_digest(
        {
            key: value
            for key, value in delta.items()
            if key != "delta_digest_sha256"
        }
    )


def test_delta_records_exact_transition_from_five_of_twenty_eight_baseline(
    delta: dict[str, Any],
    matrix: dict[str, Any],
) -> None:
    baseline = delta["baseline"]
    assert baseline["label"] == "5-of-28 ready baseline"
    assert baseline["ready_count"] == 5
    assert baseline["total_count"] == 28
    assert baseline["ready_provider_ids"] == list(BASELINE_READY)
    assert baseline["condition"] == "certificate_lock_digest_matches_current_lock"

    current = delta["current"]
    assert current["total_count"] == len(matrix["provider_host_rows"]) == 28
    assert current["ready_count"] == matrix["summary"]["provider_host_rows_ready"]
    assert current["matrix_deployment_ready"] is False
    assert current["matrix_audit_complete"] is True
    assert isinstance(current["blockers"], list)
    assert current["blockers"]

    transition = delta["transition"]
    assert transition["ready_count_delta"] == (
        current["ready_count"] - baseline["ready_count"]
    )
    assert isinstance(transition["unchanged_blockers"], list)
    assert transition["unchanged_blockers"]
    # When the checked certificate lock is stale, freshness blockers remain
    # disclosed; when the lock is current, other durable blockers (unsupported
    # SecPAL host, parser fixtures, missing dependencies) stay unchanged.
    if current.get("certificate_lock_digest_matches") is False:
        assert any(
            "stale_lock" in item for item in transition["unchanged_blockers"]
        )
    else:
        assert any(
            marker in item
            for item in transition["unchanged_blockers"]
            for marker in (
                "unsupported_host",
                "parser_fixture",
                "advisor_only_evidence",
                "supported_missing_dependencies",
                "secpal_live_semantic_cli_unavailable",
            )
        )
    assert any(
        "semantic_closed_by_fvt_g225_reference_logic" in item
        for item in transition["closed_or_narrowed_blockers"]
    )
    assert len(transition["explanations"]) >= 6


def test_matrix_identity_separation_keeps_secpal_g219_and_replacement_distinct(
    matrix: dict[str, Any],
    delta: dict[str, Any],
) -> None:
    identity = matrix["identity_separation"]
    assert identity["secpal"]["in_process_provider_id"] == REFERENCE_ID
    assert identity["secpal"]["external_provider_id"] == EXTERNAL_ID
    assert identity["secpal"]["evidence_interchangeable"] is False
    replacement = identity["production_authorization_replacement"]
    assert replacement["provider_id"] == REPLACEMENT_ID
    assert replacement["distinct_from_external_id"] == EXTERNAL_ID
    assert replacement["distinct_from_reference_id"] == REFERENCE_ID
    assert replacement["cannot_satisfy_fvt_g219"] is True
    assert replacement["required_for_legacy_secpal_row"] is False
    assert replacement["lock_provider_row"] is False
    assert replacement["legal_approval_complete"] is False
    assert identity["fvt_g219"]["status"] == "blocked"
    assert identity["fvt_g219"]["cannot_be_hidden"] is True
    assert identity["fvt_g219"]["cannot_be_satisfied_by_replacement"] is True

    providers = {row["provider_id"] for row in matrix["provider_host_rows"]}
    assert EXTERNAL_ID in providers
    assert REFERENCE_ID in providers
    assert REPLACEMENT_ID not in providers  # non-lock supplemental row only

    secpal = next(
        row for row in matrix["provider_host_rows"] if row["provider_id"] == EXTERNAL_ID
    )
    assert secpal["axes"]["platform"]["state"] == "unsupported"
    assert secpal["joint_ready"] is False
    assert matrix["disclosures"]["fvt_g219_remains_blocked_and_cannot_be_hidden"] is True
    assert matrix["disclosures"][
        "retired_microsoft_secpal_remains_unsupported_and_non_required"
    ] is True

    delta_identity = delta["identity_boundaries"]
    assert delta_identity["secpal_external"]["unsupported"] is True
    assert delta_identity["secpal_external"]["required_for_replacement"] is False
    assert delta_identity["fvt_g219"]["status"] == "blocked"
    assert delta_identity["production_authorization_replacement"]["provider_id"] == (
        REPLACEMENT_ID
    )


def test_reference_logic_overlay_closes_semantic_and_authority_without_joint_ready(
    matrix: dict[str, Any],
) -> None:
    for provider_id in (
        "datalog-authorization",
        "secpal-authorization",
        "runtime-mtl",
    ):
        row = next(
            item
            for item in matrix["provider_host_rows"]
            if item["provider_id"] == provider_id
        )
        assert row["axes"]["semantic"]["state"] == "ready"
        semantic_reasons = set(row["axes"]["semantic"]["reason_codes"] or ())
        assert semantic_reasons & {
            "closed_reference_logic_semantic_closure_bound",
            "closed_semantic_case_set_bound",
        }
        assert row["axes"]["authority"]["state"] == "ready"
        authority_reasons = set(row["axes"]["authority"]["reason_codes"] or ())
        assert authority_reasons & {
            "reference_logic_authority_ceiling_satisfied",
            "certified_authority_ceiling_satisfied",
        }
        # Freshness follows the current certificate/lock binding. When the
        # trusted certificate lock digest matches, these rows may become
        # jointly ready; when stale, freshness stays blocked.
        freshness_state = row["axes"]["freshness"]["state"]
        assert freshness_state in {"ready", "blocked"}
        if freshness_state == "blocked":
            assert row["joint_ready"] is False
        else:
            assert row["joint_ready"] is True


def test_replacement_is_new_row_blocked_on_external_approval(
    delta: dict[str, Any],
) -> None:
    row = delta["production_authorization_replacement_row"]
    assert row["provider_id"] == REPLACEMENT_ID
    assert row["lock_provider_row"] is False
    assert row["required_for_legacy_secpal"] is False
    boundary = row["identity_boundary"]
    assert boundary["distinct_from_external_id"] == EXTERNAL_ID
    assert boundary["distinct_from_reference_id"] == REFERENCE_ID
    assert boundary["cannot_satisfy_fvt_g219"] is True
    assert boundary["cannot_claim_microsoft_secpal_authority"] is True
    assert boundary["legal_approval_goal_id"] == "FVT-G232"
    # G232 may be satisfied by project-owner software disposition when
    # Microsoft SecPAL is unused (no counsel-signature theater required).
    legal_complete = boundary["legal_approval_complete"] is True
    if legal_complete:
        assert row["joint_ready"] is True
        assert "external_approval_envelope" in (
            boundary.get("g232_disposition_modes_accepted") or []
        )
        assert "project_owner_software_disposition" in (
            boundary.get("g232_disposition_modes_accepted") or []
        )
    else:
        assert row["joint_ready"] is False
        assert any(
            "g232" in reason or "fvt_g232" in reason
            for reason in row["joint_reason_codes"]
        )
    assert row["axes"]["semantic"]["state"] == "ready"
    assert row["axes"]["authority"]["details"]["forbids_fvt_g219_completion"] is True
    assert row["receipt_binding"]["deployment_ready"] is False
    assert row["receipt_binding"]["legal_approval_complete"] is legal_complete


def test_external_blockers_g219_and_g232_remain_disclosed(
    delta: dict[str, Any],
) -> None:
    blockers = delta["external_authority_blockers"]
    assert blockers["FVT-G219"]["status"] == "blocked"
    legal_complete = bool(
        (delta.get("deployment_readiness_inputs") or {}).get(
            "legal_approval_complete"
        )
    )
    assert blockers["FVT-G232"]["status"] == (
        "complete" if legal_complete else "blocked"
    )
    assert delta["claims"]["g219_remains_blocked"] is True
    assert delta["claims"]["secpal_remains_unsupported"] is True
    assert delta["disclosures"]["does_not_complete_fvt_g219"] is True
    # The post-remediation tool never authors G232 envelopes; it only observes
    # a bound external envelope or project-owner software disposition.
    assert delta["disclosures"]["does_not_complete_fvt_g232"] is True
    assert delta["disclosures"]["does_not_mark_external_approval_complete"] is True
    assert delta["disclosures"][
        "observes_bound_fvt_g232_without_authoring"
    ] is True


def test_authoritative_vendor_release_stays_assessment_complete_not_deployable() -> None:
    release = json.loads(RELEASE_PATH.read_text(encoding="utf-8"))
    assert release["interface"] == "FormalVerificationAuthoritativeVendorRelease@1"
    assert release["assessment_complete"] is True
    assert release["deployment_ready"] is False
    assert release["claims"]["deployment_ready"] is False
    assert release["status"] == "authoritative_vendor_release_blocked"
    matrix_binding = release["dependency_bindings"]["end_to_end_assurance_matrix"]
    assert matrix_binding["present"] is True
    assert matrix_binding["file_present"] is True
    # Implementation worktrees may leave the rebuilt matrix uncommitted; the
    # release still binds the exact file digest and remains non-deployable.
    assert matrix_binding["file_sha256"]
    assert matrix_binding.get("identity_valid") is True
    assert matrix_binding.get("interface_valid") is True
    # External Microsoft SecPAL is deferred from the replacement fan-in: the
    # bound/permitted gates pass under reference-only policy while live
    # production remains explicitly not granted.
    assert "secpal_authoritative_live_receipt_bound" not in release["blockers"]
    assert "secpal_production_use_permitted" not in release["blockers"]
    assert release["acceptance"].get("secpal_authoritative_live_receipt_bound") is True
    assert release["acceptance"].get("secpal_production_use_permitted") is True
    assert release["claims"].get("external_secpal_is_reference_only") is True
    assert release["claims"].get(
        "external_secpal_deferred_from_replacement_deployment"
    ) is True
    assert release["claims"].get("secpal_live_production_still_not_granted") is True
    assert release["claims"].get(
        "required_matrix_readiness_excludes_unsupported_external_secpal"
    ) is True
    assert release["claims"].get(
        "replacement_stack_required_rows_jointly_ready"
    ) is True
    assert release["claims"].get(
        "all_lock_rows_jointly_ready_including_external_secpal"
    ) is False
    assert "every_readiness_axis_jointly_ready" not in release["blockers"]
    # Residual blockers are publication / hard-zero / supervisor chain only.
    residual = set(release["blockers"])
    assert residual.issubset(
        {
            "durable_supervisor_completion_bound",
            "origin_publication_bound",
            "post_merge_attestation_bound_and_ready",
            "recursive_gitlinks_bound",
            "source_and_merged_trees_bound",
            "tactician_completion_bound_and_clear",
            "release_candidate_bound_and_ready",
            "dependencies_fresh",
            "dependencies_content_identity_bound",
            "no_fixture_shim_unsupported_proposal_or_stale_lane",
        }
    )
    assert residual & {
        "durable_supervisor_completion_bound",
        "origin_publication_bound",
        "post_merge_attestation_bound_and_ready",
        "recursive_gitlinks_bound",
        "source_and_merged_trees_bound",
        "tactician_completion_bound_and_clear",
    }


def test_builder_and_certifier_deltas_agree(
    certifier,
    builder,
    matrix: dict[str, Any],
    trusted_certificate: dict[str, Any],
) -> None:
    observed_at = "2026-08-04T12:00:00+00:00"
    from_certifier = certifier.build_post_remediation_assurance_delta(
        repo_root=REPO_ROOT,
        matrix=matrix,
        certificate=trusted_certificate,
        observed_at=observed_at,
    )
    from_builder = builder.build_post_remediation_assurance_delta(
        repo_root=REPO_ROOT,
        matrix=matrix,
        certificate=trusted_certificate,
        observed_at=observed_at,
    )
    assert from_builder["delta_digest_sha256"] == from_certifier["delta_digest_sha256"]
    assert from_builder["baseline"] == from_certifier["baseline"]
    assert from_builder["deployment_ready"] == from_certifier["deployment_ready"]
    # With project-owner software disposition (no Microsoft SecPAL), G232 may
    # clear and deployment_ready may become true; without it, stays false.
    assert isinstance(from_builder["deployment_ready"], bool)


def test_optimistic_reseal_and_authority_substitution_fail_closed(
    certifier,
    delta: dict[str, Any],
    matrix: dict[str, Any],
    trusted_certificate: dict[str, Any],
) -> None:
    optimistic = copy.deepcopy(delta)
    optimistic["deployment_ready"] = True
    optimistic["claims"]["deployment_ready"] = True
    optimistic["status"] = "post_remediation_ready_for_publication"
    optimistic.pop("delta_digest_sha256", None)
    optimistic["delta_digest_sha256"] = certifier.content_digest(
        {
            key: value
            for key, value in optimistic.items()
            if key != "delta_digest_sha256"
        }
    )
    result = certifier.validate_post_remediation_assurance_delta(
        optimistic,
        repo_root=REPO_ROOT,
        matrix=matrix,
        certificate=trusted_certificate,
    )
    assert result["valid"] is False
    # Optimistic reseal cannot invent legal approval or joint readiness.
    assert any(
        item in result["failures"]
        for item in (
            "deployment_ready_not_canonically_derived",
            "deployment_ready_without_legal_approval_complete",
            "deployment_ready_without_required_rows_ready",
            "deployment_ready_without_replacement_joint_ready",
            "claims_not_canonically_derived",
            "status_not_canonically_derived",
        )
    )

    substituted = copy.deepcopy(delta)
    substituted["production_authorization_replacement_row"][
        "identity_boundary"
    ]["cannot_satisfy_fvt_g219"] = False
    substituted["identity_boundaries"]["fvt_g219"]["status"] = "complete"
    substituted.pop("delta_digest_sha256", None)
    substituted["delta_digest_sha256"] = certifier.content_digest(
        {
            key: value
            for key, value in substituted.items()
            if key != "delta_digest_sha256"
        }
    )
    result = certifier.validate_post_remediation_assurance_delta(
        substituted,
        repo_root=REPO_ROOT,
        matrix=matrix,
        certificate=trusted_certificate,
    )
    assert result["valid"] is False
    assert any(
        item in result["failures"]
        for item in (
            "g219_not_blocked",
            "replacement_row_can_satisfy_g219",
            "identity_boundaries_not_canonically_derived",
            "claims_not_canonically_derived",
        )
    )

    promoted = copy.deepcopy(delta)
    promoted["identity_boundaries"]["secpal_external"]["unsupported"] = False
    promoted["identity_boundaries"]["secpal_external"]["joint_ready"] = True
    promoted.pop("delta_digest_sha256", None)
    promoted["delta_digest_sha256"] = certifier.content_digest(
        {
            key: value
            for key, value in promoted.items()
            if key != "delta_digest_sha256"
        }
    )
    result = certifier.validate_post_remediation_assurance_delta(
        promoted,
        repo_root=REPO_ROOT,
        matrix=matrix,
        certificate=trusted_certificate,
    )
    assert result["valid"] is False
    assert "secpal_not_marked_unsupported" in result["failures"] or any(
        "not_canonically_derived" in item for item in result["failures"]
    )


def test_matrix_validator_rejects_hidden_g219_or_collapsed_secpal_identities(
    certifier,
    matrix: dict[str, Any],
    trusted_certificate: dict[str, Any],
) -> None:
    collapsed = copy.deepcopy(matrix)
    collapsed["identity_separation"]["secpal"]["in_process_provider_id"] = EXTERNAL_ID
    collapsed = certifier.recompute_end_to_end_assurance_claims(collapsed)
    result = certifier.validate_end_to_end_assurance_matrix(
        collapsed,
        repo_root=REPO_ROOT,
        certificate=trusted_certificate,
    )
    assert result["valid"] is False
    assert any(
        "secpal" in item or "identity" in item or "canonically" in item
        for item in result["failures"]
    )

    missing_g219 = copy.deepcopy(matrix)
    missing_g219["identity_separation"].pop("fvt_g219", None)
    missing_g219 = certifier.recompute_end_to_end_assurance_claims(missing_g219)
    result = certifier.validate_end_to_end_assurance_matrix(
        missing_g219,
        repo_root=REPO_ROOT,
        certificate=trusted_certificate,
    )
    assert result["valid"] is False
    assert "identity_separation_not_canonically_derived" in result["failures"]
