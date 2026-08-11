"""Post-merge deployment attestation (FVT-067 / FVT-G214).

``RoleAwareFormalVerificationRelease@1`` finalizer after FVT-G213 merge.
Fail-closed: absent terminal evidence remains partial and is never
deployment-ready; mutations invalidate the receipt identity.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
FINALIZER_PATH = (
    REPO_ROOT / "tools" / "logic" / "finalize_formal_verification_deployment.py"
)
BUILDER_PATH = (
    REPO_ROOT / "tools" / "logic" / "build_formal_verification_tactician_receipt.py"
)
CERTIFIER_PATH = (
    REPO_ROOT / "tools" / "logic" / "certify_formal_verification_toolchains.py"
)
DEPLOYMENT_RECEIPT_PATH = (
    REPO_ROOT
    / "docs"
    / "architecture"
    / "formal_verification_role_aware_deployment_receipt.json"
)
COMPLETION_RECEIPT_PATH = (
    REPO_ROOT
    / "docs"
    / "architecture"
    / "formal_verification_tactician_readiness_completion_receipt.json"
)
RELEASE_CANDIDATE_PATH = (
    REPO_ROOT
    / "docs"
    / "architecture"
    / "formal_verification_role_aware_release_candidate.json"
)

INTERFACE = "RoleAwareFormalVerificationRelease@1"
GOAL_ID = "FVT-G214"
TASK_ID = "FVT-067"
RELEASE_CANDIDATE_TASK_ID = "FVT-066"
RELEASE_CANDIDATE_GOAL_ID = "FVT-G213"
SUPERVISOR_COMPLETION_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor.member_completion_receipt@1"
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
def finalizer():
    return _load(FINALIZER_PATH, "fvt_post_merge_finalizer_test")


@pytest.fixture(scope="module")
def builder():
    return _load(BUILDER_PATH, "fvt_post_merge_builder_test")


@pytest.fixture(scope="module")
def certifier():
    return _load(CERTIFIER_PATH, "fvt_post_merge_certifier_test")


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
def release_candidate(
    builder,
    certificate_bundle,
) -> dict[str, Any]:
    certificate, full_evidence = certificate_bundle
    return builder.build_role_aware_release_candidate(
        repo_root=REPO_ROOT,
        observed_at="2026-08-01T00:00:00Z",
        role_aware_certificate=certificate,
        source_specialized_receipt_aggregation=full_evidence[
            "specialized_receipt_aggregation"
        ],
    )


@pytest.fixture(scope="module")
def completion(builder) -> dict[str, Any]:
    return builder.build_receipt(
        repo_root=REPO_ROOT,
        observed_at="2026-08-01T00:00:00Z",
    )


@pytest.fixture(scope="module")
def attestation(finalizer, certificate, completion) -> dict[str, Any]:
    return finalizer.build_post_merge_attestation(
        repo_root=REPO_ROOT,
        observed_at="2026-08-01T00:00:00Z",
        publication_mode=finalizer.PUBLICATION_MODE_EXTERNAL,
        role_aware_certificate=certificate,
        completion_receipt=completion,
        g213_terminal_evidence=None,
    )


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _coherent_g213_terminal(finalizer) -> dict[str, Any]:
    """Build a coherent FVT-066 terminal snapshot against real published commits."""

    implementation_commit = _git("rev-parse", "origin/main^")
    merge_commit = _git("rev-parse", "origin/main")
    implementation_tree = _git("rev-parse", f"{implementation_commit}^{{tree}}")
    merge_tree = _git("rev-parse", f"{merge_commit}^{{tree}}")
    cid = (
        "baguqeeraghmkwno643c75mfl6wkop527fctnlvr2vcp75hqgjezjbtwykfba"
    )
    key = "task/v1/g213-terminal-test-key"
    completion_receipt = {
        "schema": SUPERVISOR_COMPLETION_SCHEMA,
        "status": "succeeded",
        "task_id": RELEASE_CANDIDATE_TASK_ID,
        "canonical_task_cid": cid,
        "canonical_task_key": key,
        "implementation_commit": implementation_commit,
        "merge_commit": merge_commit,
    }
    event_body = {
        "type": "implementation_finished",
        "timestamp": "2026-08-01T00:00:00Z",
        "task_id": RELEASE_CANDIDATE_TASK_ID,
        "canonical_task_cid": cid,
        "canonical_task_key": key,
        "implementation_commit": implementation_commit,
        "validation": {
            "attempted": True,
            "passed": True,
            "returncode": 0,
            "target_commit": implementation_commit,
        },
        "merge": {
            "merged": True,
            "implementation_commit": implementation_commit,
            "merge_commit": merge_commit,
            "target_branch": "origin/main",
            "integration_commit_proof": {
                "passed": True,
                "implementation_tree": implementation_tree,
                "merge_tree": merge_tree,
            },
        },
        "completion_receipts": [completion_receipt],
        "stream_id": "event-log:sha256:" + "a" * 64,
        "snapshot_id": "event-log-snapshot:sha256:" + "b" * 64,
        "sequence": 1,
        "previous_event_id": "",
    }
    event = dict(event_body)
    event["event_id"] = finalizer.content_digest(event_body)
    return {
        "task_id": RELEASE_CANDIDATE_TASK_ID,
        "canonical_task_cid": cid,
        "canonical_task_key": key,
        "task_state": {
            "canonical_identity": {
                "canonical_task_cid": cid,
                "canonical_task_key": key,
            }
        },
        "event_chain": {
            "valid": True,
            "event_count": 1,
            "last_sequence": 1,
            "last_event_id": event["event_id"],
            "errors": [],
        },
        "events": [event],
    }


def test_expected_outputs_exist() -> None:
    for path in (
        FINALIZER_PATH,
        DEPLOYMENT_RECEIPT_PATH,
        COMPLETION_RECEIPT_PATH,
        Path(__file__),
    ):
        assert path.is_file(), path
    text = FINALIZER_PATH.read_text(encoding="utf-8")
    assert INTERFACE in text
    assert GOAL_ID in text
    assert TASK_ID in text
    assert "deployment-ready" in text or "deployment_ready" in text
    assert "member_completion" in text
    assert "circular" in text.lower() or "self_referential" in text


def test_attestation_interface_goal_and_fail_closed_without_terminal(
    attestation: dict[str, Any],
) -> None:
    assert attestation["interface"] == INTERFACE
    assert attestation["goal_id"] == GOAL_ID
    assert attestation["task_id"] == TASK_ID
    assert attestation["schema_version"] == (
        "formal-verification-role-aware-deployment-receipt/v1"
    )
    assert attestation["binding_mode"] == (
        "post_merge_external_content_addressed_attestation"
    )
    assert attestation["publication_mode"] == "external_content_addressed"
    assert attestation["status"] == "role_aware_deployment_blocked"
    assert attestation["readiness_stage"] == "blocked"
    assert attestation["status"] != "role_aware_deployment_ready"
    assert attestation["readiness_stage"] != "deployment_ready"
    assert attestation["claims"]["deployment"] is False
    assert attestation["claims"]["post_merge_attestation"] is False
    assert attestation["claims"]["current_task_future_event"] is False
    assert attestation["claims"]["self_referential_current_tree"] is False
    assert attestation["acceptance"]["g213_terminal_receipt_bound"] is False
    assert "g213_terminal_receipt_bound" in attestation["deployment_blockers"]
    assert attestation["acceptance"]["never_claims_current_task_future_event"] is True
    assert attestation["acceptance"]["circular_tree_identity_forbidden"] is True


def test_attestation_identity_is_content_addressed(
    finalizer, attestation: dict[str, Any]
) -> None:
    body = {
        key: value
        for key, value in attestation.items()
        if key != "receipt_identity"
    }
    assert attestation["receipt_identity"] == finalizer.content_digest(body)
    publication = attestation["post_merge"]["publication"]
    assert (
        publication["receipt_identity"]
        == finalizer.RECEIPT_IDENTITY_SELF_REFERENCE
    )
    assert publication["receipt_identity_is_self_reference"] is True
    assert publication["receipt_identity_resolution"] == (
        "top_level.receipt_identity"
    )
    assert publication["bound"] is True
    assert publication["output_observation"] == (
        "deferred_until_after_atomic_write"
    )


def test_release_candidate_digest_is_bound(
    finalizer, attestation: dict[str, Any]
) -> None:
    assert RELEASE_CANDIDATE_PATH.is_file()
    checked = json.loads(RELEASE_CANDIDATE_PATH.read_text(encoding="utf-8"))
    stored = checked.pop("candidate_identity")
    assert stored == finalizer.content_digest(checked)

    bound = attestation["release_candidate"]
    assert bound["goal_id"] == RELEASE_CANDIDATE_GOAL_ID
    assert bound["task_id"] == RELEASE_CANDIDATE_TASK_ID
    assert bound["checked_identity_valid"] is True
    assert bound["checked_candidate_identity"] == stored
    digest_binding = bound["digest_material_verification"]
    assert str(digest_binding["digest_material_identity"]).startswith("sha256:")
    assert attestation["acceptance"]["candidate_digest_bound"] is True

    # Honest dual path: when certificate + RC are reissued in lockstep under an
    # authenticated sealed vendor root, digest material binds; otherwise the
    # finalizer must publish external-anchor drift as a partial result and must
    # not weaken certificate/source/lock gates.
    if bound["bound"] is True:
        assert digest_binding["valid"] is True
        assert digest_binding["failures"] == []
        assert bound["block_reasons"] == []
        assert attestation["acceptance"]["release_candidate_bound"] is True
        assert (
            attestation["acceptance"]["candidate_digest_material_bound"]
            is True
        )
        assert "release_candidate_digest_material_invalid" not in (
            attestation["deployment_blockers"]
        )
    else:
        assert digest_binding["valid"] is False
        assert set(digest_binding["failures"]) == {
            "certificate_digest_matches_bound_certificate",
            "lock_digest_matches_live_certificate",
            "specialized_projection_matches_live_certificate",
            "specialized_source_binding_matches_live_certificate",
        }
        assert bound["block_reasons"] == [
            "release_candidate_digest_material_invalid"
        ]
        assert attestation["acceptance"]["release_candidate_bound"] is False
        assert (
            attestation["acceptance"]["candidate_digest_material_bound"]
            is False
        )
        assert "release_candidate_digest_material_invalid" in (
            attestation["deployment_blockers"]
        )


def test_forged_candidate_identity_cannot_conceal_digest_material_drift(
    finalizer, certificate, completion
) -> None:
    forged = json.loads(RELEASE_CANDIDATE_PATH.read_text(encoding="utf-8"))
    forged["digest_material"]["certificate_digest_sha256"] = "0" * 64
    forged_body = {
        key: value for key, value in forged.items() if key != "candidate_identity"
    }
    forged["candidate_identity"] = finalizer.content_digest(forged_body)

    receipt = finalizer.build_post_merge_attestation(
        repo_root=REPO_ROOT,
        observed_at="2026-08-01T00:00:00Z",
        role_aware_certificate=certificate,
        completion_receipt=completion,
        release_candidate=forged,
        g213_terminal_evidence=None,
    )
    binding = receipt["release_candidate"]
    assert binding["checked_identity_valid"] is True
    assert binding["matches_live_recompute"] is False
    assert binding["digest_material_verification"]["valid"] is False
    assert (
        "certificate_digest_matches_projection"
        in binding["digest_material_verification"]["failures"]
    )
    assert binding["bound"] is False
    assert receipt["acceptance"]["release_candidate_bound"] is False
    assert receipt["acceptance"]["candidate_digest_material_bound"] is False
    assert "release_candidate_digest_material_invalid" in (
        receipt["deployment_blockers"]
    )
    assert receipt["claims"]["deployment"] is False


def test_rehashed_specialized_handler_composite_and_source_maps_fail_closed(
    finalizer,
    certifier,
    certificate,
    release_candidate,
) -> None:
    baseline = finalizer.verify_release_candidate_digest_material(
        release_candidate,
        certifier=certifier,
        role_aware_certificate=certificate,
    )
    # Independent FVT-066 audit can fail closed on host/vendor-seal state
    # (e.g. hyper checked-vendor root) without invalidating the compact
    # projection maps this mutation test covers. Require those maps to be
    # coherent; do not demand full digest-material validity.
    mutation_preconditions = (
        "specialized_handler_self_digests_recomputed",
        "specialized_projection_handler_digests_match",
        "specialized_projection_matches_live_certificate",
        "specialized_composite_coverage_exact",
        "specialized_source_handler_digests_match",
    )
    for key in mutation_preconditions:
        assert baseline["checks"][key] is True, key
    assert set(baseline["failures"]) <= {
        "specialized_fvt066_independent_audit_bound",
    }

    forged_handler_candidate = copy.deepcopy(release_candidate)
    specialized = forged_handler_candidate["role_aware_certificate"][
        "specialized_receipt_aggregation"
    ]["projection"]
    handler_key = sorted(specialized["specialized_by_handler"])[0]
    handler = specialized["specialized_by_handler"][handler_key]
    handler["authority_ceiling"] = "forged_authority"
    handler["tool_evidence_digest_sha256"] = certifier.content_digest(
        {
            key: value
            for key, value in handler.items()
            if key != "tool_evidence_digest_sha256"
        }
    )
    specialized["aggregation_digest_sha256"] = certifier.content_digest(
        {
            key: value
            for key, value in specialized.items()
            if key != "aggregation_digest_sha256"
        }
    )
    material = forged_handler_candidate["digest_material"]
    material["specialized_projection_aggregation_digest"] = specialized[
        "aggregation_digest_sha256"
    ]
    material["specialized_projection_handler_digests"][handler_key] = (
        handler["tool_evidence_digest_sha256"]
    )
    # A coherent forgery also updates the candidate-embedded audit digest.
    # This embedded value is not independent authority; the separately loaded
    # certificate/source projection must still reject the mutation.
    forged_handler_candidate["role_aware_certificate"][
        "specialized_receipt_aggregation"
    ]["verification"]["projection_aggregation_digest_sha256"] = (
        specialized["aggregation_digest_sha256"]
    )
    handler_verification = (
        finalizer.verify_release_candidate_digest_material(
            forged_handler_candidate,
            certifier=certifier,
            role_aware_certificate=certificate,
        )
    )
    assert handler_verification["valid"] is False
    assert "specialized_projection_matches_live_certificate" in (
        handler_verification["failures"]
    )

    forged_composite_candidate = copy.deepcopy(release_candidate)
    specialized = forged_composite_candidate["role_aware_certificate"][
        "specialized_receipt_aggregation"
    ]["projection"]
    composite = next(iter(specialized["composite_lanes"].values()))
    composite["handler_keys"] = composite["handler_keys"][:-1]
    specialized["aggregation_digest_sha256"] = certifier.content_digest(
        {
            key: value
            for key, value in specialized.items()
            if key != "aggregation_digest_sha256"
        }
    )
    forged_composite_candidate["digest_material"][
        "specialized_projection_aggregation_digest"
    ] = specialized["aggregation_digest_sha256"]
    composite_verification = (
        finalizer.verify_release_candidate_digest_material(
            forged_composite_candidate,
            certifier=certifier,
            role_aware_certificate=certificate,
        )
    )
    assert composite_verification["valid"] is False
    assert "specialized_composite_coverage_exact" in (
        composite_verification["failures"]
    )

    forged_source_map_candidate = copy.deepcopy(release_candidate)
    forged_source_map_candidate["digest_material"][
        "specialized_source_handler_digests"
    ][handler_key] = "0" * 64
    source_map_verification = (
        finalizer.verify_release_candidate_digest_material(
            forged_source_map_candidate,
            certifier=certifier,
            role_aware_certificate=certificate,
        )
    )
    assert source_map_verification["valid"] is False
    assert "specialized_source_handler_digests_match" in (
        source_map_verification["failures"]
    )


def test_checked_in_deployment_receipt_is_content_addressed_and_not_false_ready(
    finalizer,
) -> None:
    checked = json.loads(DEPLOYMENT_RECEIPT_PATH.read_text(encoding="utf-8"))
    stored = checked.pop("receipt_identity")
    assert stored == finalizer.content_digest(checked)
    assert checked["status"] != "role_aware_deployment_ready"
    assert checked["status"] != "deployment_ready"
    assert checked.get("deployment_blockers")
    # Post-merge finalizer product may own the path; G200 reissue also uses
    # this interface. Either goal is acceptable as long as the gate stays
    # fail-closed.
    assert checked["interface"] == INTERFACE
    assert checked["status"] in {
        "role_aware_deployment_blocked",
        "post_merge_attestation_partial",
    }


def test_mutating_publication_or_event_invalidates_identity(
    finalizer, certificate, completion
) -> None:
    base = finalizer.build_post_merge_attestation(
        repo_root=REPO_ROOT,
        observed_at="2026-08-01T00:00:00Z",
        role_aware_certificate=certificate,
        completion_receipt=completion,
        g213_terminal_evidence=None,
    )
    mutated = copy.deepcopy(base)
    mutated["post_merge"]["publication"]["mode"] = "forged"
    body = {
        key: value for key, value in mutated.items() if key != "receipt_identity"
    }
    assert finalizer.content_digest(body) != base["receipt_identity"]

    mutated_event = copy.deepcopy(base)
    terminal = mutated_event["post_merge"]["terminal"]
    terminal["event_chain"] = dict(terminal.get("event_chain") or {})
    terminal["event_chain"]["event_count"] = 999
    body2 = {
        key: value
        for key, value in mutated_event.items()
        if key != "receipt_identity"
    }
    assert finalizer.content_digest(body2) != base["receipt_identity"]


def test_cannot_attest_current_task_future_event(
    finalizer, certificate, completion
) -> None:
    forged = {
        "task_id": TASK_ID,
        "canonical_task_cid": "baguqeera-forged",
        "canonical_task_key": "task/v1/forged",
        "events": [],
    }
    receipt = finalizer.build_post_merge_attestation(
        repo_root=REPO_ROOT,
        observed_at="2026-08-01T00:00:00Z",
        role_aware_certificate=certificate,
        completion_receipt=completion,
        g213_terminal_evidence=forged,
    )
    assert receipt["status"] == "role_aware_deployment_blocked"
    assert receipt["acceptance"]["g213_terminal_receipt_bound"] is False
    assert receipt["post_merge"]["terminal"]["claims_current_task_future_event"] is True
    assert "cannot_attest_current_task_future_event" in receipt["deployment_blockers"]
    assert receipt["claims"]["deployment"] is False


def test_coherent_g213_terminal_binds_merge_gates_without_false_ready(
    finalizer, certificate, completion
) -> None:
    terminal = _coherent_g213_terminal(finalizer)
    verified = finalizer.verify_g213_terminal_evidence(
        repo_root=REPO_ROOT,
        evidence=terminal,
    )
    assert verified["bound"] is True
    assert verified["event_chain"]["valid"] is True
    assert verified["validation"]["bound"] is True
    assert verified["merge"]["bound"] is True
    assert verified["expected_outputs"]["bound"] is True
    assert verified["commit_binding"]["valid"] is True
    assert verified["commit_binding"]["published_to_origin_main"] is True

    receipt = finalizer.build_post_merge_attestation(
        repo_root=REPO_ROOT,
        observed_at="2026-08-01T00:00:00Z",
        role_aware_certificate=certificate,
        completion_receipt=completion,
        g213_terminal_evidence=terminal,
    )
    assert receipt["acceptance"]["g213_terminal_receipt_bound"] is True
    assert receipt["acceptance"]["event_chain_continuous"] is True
    assert receipt["acceptance"]["validation_result_bound"] is True
    assert receipt["acceptance"]["merged_commit_bound"] is True
    assert receipt["acceptance"]["origin_publication_bound"] is True
    assert receipt["acceptance"]["g213_expected_outputs_bound"] is True
    assert receipt["acceptance"]["publication_bound"] is True
    merge_binding = receipt["release_candidate"]["terminal_merge_blob_binding"]
    if merge_binding["current_blob"] == merge_binding["merged_blob"]:
        assert receipt["acceptance"][
            "release_candidate_merge_blob_bound"
        ] is True
        assert merge_binding["bound"] is True
        assert receipt["claims"]["merge"] is True
        assert "release_candidate_merge_blob_bound" not in receipt[
            "deployment_blockers"
        ]
    else:
        # A newly generated candidate that is not in the asserted terminal
        # merge must remain explicitly pre-merge and fail closed.
        assert receipt["acceptance"][
            "release_candidate_merge_blob_bound"
        ] is False
        assert merge_binding["bound"] is False
        assert receipt["claims"]["merge"] is False
        assert "release_candidate_terminal_merge_blob_mismatch" in (
            merge_binding["failures"]
        )
        assert "release_candidate_merge_blob_bound" in receipt[
            "deployment_blockers"
        ]
    # Other gates (hard-zero, elevations, managed capabilities) still block.
    assert receipt["status"] == "role_aware_deployment_blocked"
    assert receipt["claims"]["deployment"] is False
    assert "g213_terminal_receipt_bound" not in receipt["deployment_blockers"]


def test_g213_terminal_rejects_target_assumption_and_legacy_pseudo_cid(
    finalizer,
) -> None:
    assumed = _coherent_g213_terminal(finalizer)
    assumed["task_state"]["assumed_completed_task_ids"] = [
        "FVT-054",
        RELEASE_CANDIDATE_TASK_ID,
    ]
    assumed["task_state"]["assumed_completed_count"] = 2
    verified_assumed = finalizer.verify_g213_terminal_evidence(
        repo_root=REPO_ROOT,
        evidence=assumed,
    )
    assert verified_assumed["bound"] is False
    assert verified_assumed["assumed_completion_rejected"] is True
    assert RELEASE_CANDIDATE_TASK_ID in verified_assumed[
        "target_assumed_completion_references"
    ]
    assert "g213_target_assumed_completion_forbidden" in (
        verified_assumed["block_reasons"]
    )

    unrelated_dependencies = _coherent_g213_terminal(finalizer)
    unrelated_dependencies["task_state"][
        "assumed_completed_task_ids"
    ] = ["FVT-054", "FVT-055"]
    unrelated_dependencies["task_state"]["assumed_completed_count"] = 2
    verified_dependencies = finalizer.verify_g213_terminal_evidence(
        repo_root=REPO_ROOT,
        evidence=unrelated_dependencies,
    )
    assert verified_dependencies["assumed_completion_rejected"] is False
    assert "g213_target_assumed_completion_forbidden" not in (
        verified_dependencies["block_reasons"]
    )

    legacy = _coherent_g213_terminal(finalizer)
    legacy["canonical_task_cid"] = "task:legacy-pseudo-cid"
    legacy["task_state"]["canonical_identity"][
        "canonical_task_cid"
    ] = "task:legacy-pseudo-cid"
    verified_legacy = finalizer.verify_g213_terminal_evidence(
        repo_root=REPO_ROOT,
        evidence=legacy,
    )
    assert verified_legacy["bound"] is False
    assert "canonical_task_cid_not_strict_cidv1" in (
        verified_legacy["block_reasons"]
    )


def test_stale_or_broken_event_chain_never_binds(
    finalizer, certificate, completion
) -> None:
    terminal = _coherent_g213_terminal(finalizer)
    terminal["events"][0]["event_id"] = "sha256:" + "0" * 64
    receipt = finalizer.build_post_merge_attestation(
        repo_root=REPO_ROOT,
        observed_at="2026-08-01T00:00:00Z",
        role_aware_certificate=certificate,
        completion_receipt=completion,
        g213_terminal_evidence=terminal,
    )
    assert receipt["acceptance"]["g213_terminal_receipt_bound"] is False
    assert receipt["acceptance"]["event_chain_continuous"] is False
    assert receipt["status"] != "role_aware_deployment_ready"


def test_receipt_commit_mode_requires_parent_and_limited_diff(
    finalizer, certificate, completion
) -> None:
    source_commit = _git("rev-parse", "HEAD")
    # Missing receipt commit → blocked.
    missing = finalizer.build_post_merge_attestation(
        repo_root=REPO_ROOT,
        observed_at="2026-08-01T00:00:00Z",
        publication_mode=finalizer.PUBLICATION_MODE_RECEIPT_COMMIT,
        role_aware_certificate=certificate,
        completion_receipt=completion,
        receipt_commit=None,
    )
    assert missing["binding_mode"] == "post_merge_receipt_commit_publication"
    assert missing["acceptance"]["publication_bound"] is False
    assert "receipt_commit_missing" in missing["deployment_blockers"]
    assert missing["status"] == "role_aware_deployment_blocked"

    # A real HEAD commit is almost certainly not a limited generated-artifact
    # receipt commit parented on certified source, so publication stays closed.
    with_commit = finalizer.build_post_merge_attestation(
        repo_root=REPO_ROOT,
        observed_at="2026-08-01T00:00:00Z",
        publication_mode=finalizer.PUBLICATION_MODE_RECEIPT_COMMIT,
        role_aware_certificate=certificate,
        completion_receipt=completion,
        receipt_commit=source_commit,
    )
    assert with_commit["status"] == "role_aware_deployment_blocked"
    publication = with_commit["post_merge"]["publication"]
    assert publication["mode"] == "receipt_commit"
    assert publication["allowed_paths"]
    assert set(publication["allowed_paths"]) <= {
        "docs/architecture/formal_verification_role_aware_deployment_receipt.json",
        "docs/architecture/formal_verification_tactician_readiness_completion_receipt.json",
        "docs/architecture/formal_verification_toolchain_certificate.json",
    }


def test_public_surfaces_and_compact_body(
    attestation: dict[str, Any],
) -> None:
    text = json.dumps(attestation, sort_keys=True)
    assert "/home/" not in text
    assert "/tmp/" not in text
    assert "/private/tmp/" not in text
    assert attestation["public_evidence_policy"]["satisfied"] is True
    assert attestation["acceptance"]["public_surfaces_bound"] is True
    # Compact: no bulk semantic receipt dumps.
    encoded = json.dumps(attestation, separators=(",", ":"), ensure_ascii=False)
    assert len(encoded.encode("utf-8")) < 1_000_000
    for lane in attestation["role_aware_certificate"].get("semantic_lane_results") or []:
        assert "receipt" not in lane
        assert "per_tool" not in lane


def test_hard_zero_authority_quarantine_and_capability_gates_recorded(
    attestation: dict[str, Any],
) -> None:
    for key in (
        "false_proof_count",
        "false_closure_count",
        "secret_or_witness_leakage_count",
        "authority_boundary_violations",
        "unresolved_cross_provider_disagreement_count",
    ):
        assert key in attestation["hard_zero_gates"]
        assert isinstance(attestation["hard_zero_gates"][key], int)
        assert attestation["hard_zero_gates"][key] >= 0
    assert "authority_ceiling_respected" in attestation["acceptance"]
    assert "quarantines_bound" in attestation["acceptance"]
    assert "supported_capability_closure" in attestation["acceptance"]
    # Elevation gate tracks whatever the current certificate elevates; lean may
    # already be production-certified while other required lanes remain open.
    assert "required_elevations_complete" in attestation["acceptance"]
    missing = set(attestation["elevations"]["missing_required"])
    assert missing <= {
        "lean",
        "runtime-mtl",
        "datalog-authorization",
        "secpal-authorization",
        "coq",
        "isabelle",
    }
    if attestation["acceptance"]["required_elevations_complete"] is False:
        assert missing
    else:
        assert not missing


def test_finalize_writes_atomic_external_attestation(
    finalizer, certificate, completion, tmp_path: Path
) -> None:
    output = tmp_path / "post_merge_attestation.json"
    receipt = finalizer.finalize_deployment(
        repo_root=REPO_ROOT,
        output=output,
        observed_at="2026-08-01T00:00:00Z",
        publication_mode=finalizer.PUBLICATION_MODE_EXTERNAL,
        g213_terminal_evidence=None,
        write=True,
    )
    assert output.is_file()
    on_disk = json.loads(output.read_text(encoding="utf-8"))
    assert on_disk == receipt
    assert "publication_write" not in receipt
    assert on_disk["receipt_identity"] == receipt["receipt_identity"]
    assert on_disk["status"] == "role_aware_deployment_blocked"
    assert on_disk["goal_id"] == GOAL_ID
    publication = on_disk["post_merge"]["publication"]
    assert (
        publication["receipt_identity"]
        == finalizer.RECEIPT_IDENTITY_SELF_REFERENCE
    )
    assert publication["output_present"] is None
    verified = finalizer.load_verified_receipt(output, expected=receipt)
    assert verified == receipt
    body = {
        key: value for key, value in on_disk.items() if key != "receipt_identity"
    }
    assert on_disk["receipt_identity"] == finalizer.content_digest(body)


def test_existing_output_cannot_influence_embedded_publication(
    finalizer, tmp_path: Path
) -> None:
    output = tmp_path / "post_merge_attestation.json"
    output.write_text('{"prior":"one"}\n', encoding="utf-8")
    first = finalizer.verify_external_publication(
        receipt_identity=finalizer.RECEIPT_IDENTITY_SELF_REFERENCE,
        output_path=output,
        repo_root=REPO_ROOT,
    )
    output.write_text('{"prior":"two"}\n', encoding="utf-8")
    second = finalizer.verify_external_publication(
        receipt_identity=finalizer.RECEIPT_IDENTITY_SELF_REFERENCE,
        output_path=output,
        repo_root=REPO_ROOT,
    )
    assert first == second
    assert first["output_present"] is None
    assert first["output_file_sha256"] is None


def test_on_disk_identity_verification_rejects_post_write_tampering(
    finalizer, certificate, completion, tmp_path: Path
) -> None:
    output = tmp_path / "post_merge_attestation.json"
    receipt = finalizer.finalize_deployment(
        repo_root=REPO_ROOT,
        output=output,
        observed_at="2026-08-01T00:00:00Z",
        publication_mode=finalizer.PUBLICATION_MODE_EXTERNAL,
        g213_terminal_evidence=None,
        write=True,
    )
    tampered = json.loads(output.read_text(encoding="utf-8"))
    tampered["status"] = "role_aware_deployment_ready"
    output.write_text(json.dumps(tampered, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="on-disk verification"):
        finalizer.load_verified_receipt(output)
    with pytest.raises(RuntimeError, match="round trip"):
        finalizer.load_verified_receipt(output, expected=receipt)


def test_source_and_datasets_gitlink_recorded(
    attestation: dict[str, Any],
) -> None:
    source = attestation["source"]
    assert source.get("certified_source_commit")
    assert source.get("certified_source_tree")
    assert "datasets_gitlink" in source
    assert attestation["acceptance"]["circular_tree_identity_forbidden"] is True
    assert source.get("attestation_excluded_from_source_tree") is True
