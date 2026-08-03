"""Authoritative, fail-closed vendor release fan-in (FVT-089 / FVT-G221).

The checked artifact is expected to be blocked on this repository snapshot.
Authentic SecPAL provenance and compatibility execution are valuable evidence,
but the reviewed EULA, unsupported vendor platform, and missing production
authority cannot be converted into deployment readiness.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
BUILDER_PATH = (
    REPO_ROOT / "tools" / "logic" / "build_formal_verification_tactician_receipt.py"
)
RELEASE_PATH = (
    REPO_ROOT
    / "docs"
    / "architecture"
    / "formal_verification_authoritative_vendor_release.json"
)
VENDOR_TEST_PATH = (
    REPO_ROOT
    / "test"
    / "integration"
    / "toolchains"
    / "test_secpal_ergoai_authoritative_live_evidence.py"
)

INTERFACE = "FormalVerificationAuthoritativeVendorRelease@1"
SCHEMA = "formal-verification-authoritative-vendor-release/v1"
GOAL_ID = "FVT-G221"
TASK_ID = "FVT-089"
VALIDATION_COMMAND = (
    "PYTHONPATH=ipfs_datasets_py python -m pytest "
    "test/integration/test_formal_verification_authoritative_vendor_release.py "
    "test/integration/toolchains/"
    "test_formal_verification_end_to_end_assurance_matrix.py "
    "test/integration/toolchains/"
    "test_secpal_ergoai_authoritative_live_evidence.py -q"
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
def builder():
    return _load(BUILDER_PATH, "fvt_authoritative_vendor_release_builder")


def _seal(builder, payload: dict[str, Any], field: str) -> dict[str, Any]:
    sealed = copy.deepcopy(payload)
    sealed.pop(field, None)
    digest = builder.content_digest(sealed)
    if field in {"receipt_digest_sha256", "matrix_digest_sha256"}:
        digest = digest.removeprefix("sha256:")
    sealed[field] = digest
    return sealed


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _axis(state: str = "ready") -> dict[str, Any]:
    return {
        "state": state,
        "reason_codes": [],
        "evidence_refs": ["sha256:" + "1" * 64],
    }


def _matrix(builder, observed_at: str) -> dict[str, Any]:
    """Synthetic counterfactual; never repository or deployment authority."""

    rows = []
    for row_id, tool_id in (
        ("secpal-external@windows-x86_64", "secpal-external"),
        ("ergoai@linux-aarch64", "ergoai"),
    ):
        rows.append(
            {
                "row_id": row_id,
                "tool_id": tool_id,
                "host": row_id.rsplit("@", 1)[-1],
                "axes": {
                    axis: _axis() for axis in builder.AUTHORITATIVE_VENDOR_REQUIRED_AXES
                },
            }
        )
    return _seal(
        builder,
        {
            "interface": "FormalVerificationEndToEndAssuranceMatrix@1",
            "schema_version": "formal-verification-end-to-end-assurance-matrix/v1",
            "goal_id": "FVT-G220",
            "task_id": "FVT-088",
            "observed_at": observed_at,
            "test_fixture_class": "synthetic_counterfactual_never_authoritative",
            "rows": rows,
        },
        "matrix_digest_sha256",
    )


def _secpal(builder, observed_at: str) -> dict[str, Any]:
    """Synthetic EULA counterfactual; never external SecPAL authority."""

    return _seal(
        builder,
        {
            "interface": "SecPALAuthoritativeLiveEvidence@1",
            "schema_version": "secpal-authoritative-live-evidence/v1",
            "goal_id": "FVT-G219",
            "task_id": "FVT-086",
            "observed_at": observed_at,
            "test_fixture_class": "synthetic_counterfactual_never_authoritative",
            "artifact": {
                "sha256": builder.SECPAL_RESEARCH_RELEASE_MSI_SHA256,
                "size_bytes": builder.SECPAL_RESEARCH_RELEASE_MSI_SIZE_BYTES,
                "product_code": builder.SECPAL_RESEARCH_RELEASE_PRODUCT_CODE,
                "product_version": builder.SECPAL_RESEARCH_RELEASE_PRODUCT_VERSION,
                "target_clr": builder.SECPAL_RESEARCH_RELEASE_CLR_VERSION,
            },
            "authenticode": {"verified": True},
            "license": {
                "terms_reviewed": True,
                # A claimed override must not defeat the reviewed EULA.
                "production_use_allowed": True,
                "eula_sha256": builder.SECPAL_RESEARCH_RELEASE_EULA_SHA256,
            },
            "platform": {"vendor_supported": True},
            "live_vendor_execution": True,
            "authoritative_live_evidence": True,
            "arbitrary_policy_query_execution": True,
            "restricted_bytes_published": False,
            "cases": [
                {"kind": kind, "status": "passed"}
                for kind in builder.SECPAL_AUTHORITATIVE_REQUIRED_CASE_KINDS
            ],
            "external_authority_blockers": [],
            "block_reasons": [],
        },
        "receipt_digest_sha256",
    )


def _ergoai(builder, observed_at: str) -> dict[str, Any]:
    """Synthetic advisor counterfactual; never managed-runtime evidence."""

    return _seal(
        builder,
        {
            "interface": "LiveErgoAIAdvisorCertification@1",
            "schema_version": "live-ergoai-advisor-certification/v1",
            "goal_id": "FVT-G218",
            "task_id": "FVT-085",
            "observed_at": observed_at,
            "test_fixture_class": "synthetic_counterfactual_never_authoritative",
            "selected_platform": "linux-aarch64",
            "managed_vendor_live_evidence": True,
            "vendor_certified": True,
            "production_certified": True,
            "managed_identity": {
                "release_artifact_sha256": "2" * 64,
                "selected_platform": "linux-aarch64",
            },
            "authority_ceiling": "advisory",
            "grants_proof_authority": False,
            "grants_theorem_authority": False,
            "install_attempted": False,
            "download_attempted": False,
            "network_used": False,
            "cases": [
                {"kind": kind, "status": "passed"}
                for kind in builder.ERGOAI_AUTHORITATIVE_REQUIRED_CASE_KINDS
            ],
            "block_reasons": [],
        },
        "receipt_digest_sha256",
    )


def _candidate(builder, observed_at: str) -> dict[str, Any]:
    return _seal(
        builder,
        {
            "interface": builder.RELEASE_CANDIDATE_INTERFACE,
            "schema_version": builder.RELEASE_CANDIDATE_SCHEMA_VERSION,
            "goal_id": builder.RELEASE_CANDIDATE_GOAL_ID,
            "task_id": builder.RELEASE_CANDIDATE_TASK_ID,
            "observed_at": observed_at,
            "test_fixture_class": "synthetic_counterfactual_never_authoritative",
            "status": "role_aware_release_candidate_ready",
            "acceptance": {
                "authority_ceiling_respected": True,
                "semantic_receipts_full_and_bound": True,
            },
            "quarantine_state": {"bound": True, "count": 0},
            "claims": {"merge": False, "deployment": False},
            "blockers": [],
            "platform_exceptions": [],
        },
        "candidate_identity",
    )


def _post_merge(builder, observed_at: str) -> dict[str, Any]:
    commit = "a" * 40
    tree = "b" * 40
    return _seal(
        builder,
        {
            "interface": builder.ROLE_AWARE_INTERFACE,
            "schema_version": builder.ROLE_AWARE_SCHEMA_VERSION,
            "goal_id": builder.ROLE_AWARE_GOAL_ID,
            "task_id": builder.ROLE_AWARE_TASK_ID,
            "observed_at": observed_at,
            "test_fixture_class": "synthetic_counterfactual_never_authoritative",
            "status": "role_aware_deployment_ready_for_attestation_publication",
            "acceptance": {"public_evidence_safe": True},
            "readiness_requirements": {"all_prior_gates": True},
            "deployment_blockers": [],
            "source": {
                "source_commit_bound": True,
                "valid_for_attestation": True,
                "certified_source_commit": commit,
                "certified_source_tree": tree,
                "datasets_gitlink": commit,
                "datasets_embedded_head": commit,
            },
            "supervisor_evidence": {
                "bound": True,
                "provisional_bound": True,
                "publication_bound": True,
                "publication_phase": "published_final",
                "member_completion_receipt_bound": True,
                "validation_bound": True,
                "state_terminal_bound": True,
                "merge_commit_tree_bound": True,
                "commit_bindings": [
                    {
                        "source_trees_bound": True,
                        "merge_tree_matches_implementation": True,
                        "published_to_origin_main": True,
                    }
                ],
            },
        },
        "receipt_identity",
    )


def _completion(builder, observed_at: str) -> dict[str, Any]:
    return _seal(
        builder,
        {
            "interface": builder.INTERFACE,
            "schema_version": builder.SCHEMA_VERSION,
            "completion_goal_id": builder.COMPLETION_GOAL_ID,
            "task_id": builder.TASK_ID,
            "observed_at": observed_at,
            "test_fixture_class": "synthetic_counterfactual_never_authoritative",
            "acceptance": {
                "implementation_complete": True,
                "hard_zero_gates_clear": True,
            },
            "implementation": {"status": "complete"},
            "hard_zero_gates": {key: 0 for key in builder.HARD_ZERO_GATE_KEYS},
        },
        "receipt_identity",
    )


def _inputs(builder, observed_at: str) -> dict[str, dict[str, Any]]:
    return {
        "end_to_end_assurance_matrix": _matrix(builder, observed_at),
        "secpal_authoritative_live_receipt": _secpal(builder, observed_at),
        "ergoai_authoritative_live_receipt": _ergoai(builder, observed_at),
        "role_aware_release_candidate": _candidate(builder, observed_at),
        "post_merge_deployment_receipt": _post_merge(builder, observed_at),
        "completion_receipt": _completion(builder, observed_at),
    }


def test_expected_outputs_and_builder_constants(builder) -> None:
    for path in (BUILDER_PATH, RELEASE_PATH, VENDOR_TEST_PATH, Path(__file__)):
        assert path.is_file(), path
    assert builder.AUTHORITATIVE_VENDOR_RELEASE_INTERFACE == INTERFACE
    assert builder.AUTHORITATIVE_VENDOR_RELEASE_SCHEMA_VERSION == SCHEMA
    assert builder.AUTHORITATIVE_VENDOR_RELEASE_GOAL_ID == GOAL_ID
    assert builder.AUTHORITATIVE_VENDOR_RELEASE_TASK_ID == TASK_ID
    assert builder.AUTHORITATIVE_VENDOR_RELEASE_VALIDATION_COMMAND == (
        VALIDATION_COMMAND
    )
    assert builder.DEFAULT_AUTHORITATIVE_VENDOR_RELEASE_RELATIVE.as_posix() == (
        "docs/architecture/formal_verification_authoritative_vendor_release.json"
    )


def test_checked_release_is_content_addressed_public_and_blocked(builder) -> None:
    release = json.loads(RELEASE_PATH.read_text(encoding="utf-8"))
    assert release["interface"] == INTERFACE
    assert release["schema_version"] == SCHEMA
    assert release["goal_id"] == GOAL_ID
    assert release["task_id"] == TASK_ID
    assert release["status"] == "authoritative_vendor_release_blocked"
    assert release["deployment_ready"] is False
    assert release["claims"]["deployment_ready"] is False
    assert release["claims"]["current_task_merge"] is False
    assert release["claims"]["current_release_artifact_published"] is False
    assert release["public_evidence_policy"]["satisfied"] is True
    assert release["release_identity"] == builder.content_digest(
        {key: value for key, value in release.items() if key != "release_identity"}
    )
    known = release["known_secpal_release"]
    assert known["artifact_bytes_embedded"] is False
    assert known["eula_text_embedded"] is False
    assert known["msi_sha256"] == builder.SECPAL_RESEARCH_RELEASE_MSI_SHA256
    assert known["eula_sha256"] == builder.SECPAL_RESEARCH_RELEASE_EULA_SHA256
    assert known["production_purpose_disposition"] == (
        "not_intended_for_live_environment"
    )
    assert release["evidence"]["validation_command"] == VALIDATION_COMMAND
    assert RELEASE_PATH.stat().st_size < 500_000


def test_exact_recovered_secpal_eula_is_an_unoverrideable_hard_gate(builder) -> None:
    receipt = _secpal(builder, "2026-08-03T12:00:00Z")
    audit = builder._audit_secpal_authoritative_live(receipt)
    assert audit["official_release_identity_bound"] is True
    assert audit["microsoft_authenticode_verified"] is True
    assert audit["semantic_cases"]["valid"] is True
    assert audit["claimed_production_use_allowed"] is True
    assert audit["known_eula_live_purpose_block"] is True
    assert audit["production_use_allowed"] is False
    assert audit["valid"] is False
    assert "production_use_allowed" in audit["blockers"]


def test_all_other_counterfactual_inputs_still_cannot_override_eula(
    builder,
    monkeypatch,
) -> None:
    observed_at = _now()
    monkeypatch.setattr(
        builder,
        "_observe_recursive_gitlinks",
        lambda _root: {
            "valid": True,
            "gitlink_count": 1,
            "rows": [{"path": "ipfs_datasets_py", "bound": True}],
            "network_used": False,
            "fetch_attempted": False,
        },
    )
    release = builder.build_authoritative_vendor_release(
        repo_root=REPO_ROOT,
        observed_at=observed_at,
        **_inputs(builder, observed_at),
    )
    # The maps are internally self-consistent, but self-resealing cannot make
    # them exact committed repository receipts or canonical matrix evidence.
    assert release["acceptance"]["dependencies_reachable"] is True
    assert release["acceptance"]["dependencies_content_identity_bound"] is False
    assert release["acceptance"]["dependencies_fresh"] is True
    assert release["end_to_end_assurance"]["local_row_claims_ready"] is True
    assert release["end_to_end_assurance"]["hardened_validation_valid"] is False
    assert release["acceptance"]["every_readiness_axis_jointly_ready"] is False
    assert release["acceptance"]["ergoai_managed_vendor_live_receipt_bound"] is False
    assert release["acceptance"]["post_merge_attestation_bound_and_ready"] is False
    assert release["acceptance"]["recursive_gitlinks_bound"] is True
    assert release["acceptance"]["secpal_production_use_permitted"] is False
    assert release["acceptance"]["secpal_authoritative_live_receipt_bound"] is False
    assert release["deployment_ready"] is False
    assert "secpal_production_use_permitted" in release["blockers"]


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("fixture", "disallowed_boolean:is_fixture"),
        ("shim", "disallowed_boolean:is_hermetic_advisor_shim"),
        ("unsupported", "disallowed_state:unsupported"),
        ("proposal", "disallowed_evidence_class:proposal_only_semantics"),
        ("stale", "evidence_age_exceeds_policy"),
        ("missing_authority", "authoritative_live_evidence"),
    ],
)
def test_fixture_shim_unsupported_proposal_stale_and_missing_authority_fail_closed(
    builder,
    mutation: str,
    expected: str,
) -> None:
    observed_at = _now()
    inputs = _inputs(builder, observed_at)
    if mutation == "fixture":
        inputs["ergoai_authoritative_live_receipt"]["is_fixture"] = True
        inputs["ergoai_authoritative_live_receipt"] = _seal(
            builder,
            inputs["ergoai_authoritative_live_receipt"],
            "receipt_digest_sha256",
        )
    elif mutation == "shim":
        inputs["ergoai_authoritative_live_receipt"]["is_hermetic_advisor_shim"] = True
        inputs["ergoai_authoritative_live_receipt"] = _seal(
            builder,
            inputs["ergoai_authoritative_live_receipt"],
            "receipt_digest_sha256",
        )
    elif mutation == "unsupported":
        inputs["end_to_end_assurance_matrix"]["rows"][0]["axes"]["platform"] = _axis(
            "unsupported"
        )
        inputs["end_to_end_assurance_matrix"] = _seal(
            builder,
            inputs["end_to_end_assurance_matrix"],
            "matrix_digest_sha256",
        )
    elif mutation == "proposal":
        inputs["role_aware_release_candidate"][
            "evidence_class"
        ] = "proposal_only_semantics"
        inputs["role_aware_release_candidate"] = _seal(
            builder,
            inputs["role_aware_release_candidate"],
            "candidate_identity",
        )
    elif mutation == "stale":
        inputs["ergoai_authoritative_live_receipt"]["observed_at"] = (
            datetime.now(UTC) - timedelta(days=2)
        ).strftime("%Y-%m-%dT%H:%M:%SZ")
        inputs["ergoai_authoritative_live_receipt"] = _seal(
            builder,
            inputs["ergoai_authoritative_live_receipt"],
            "receipt_digest_sha256",
        )
    else:
        inputs["secpal_authoritative_live_receipt"][
            "authoritative_live_evidence"
        ] = False
        inputs["secpal_authoritative_live_receipt"] = _seal(
            builder,
            inputs["secpal_authoritative_live_receipt"],
            "receipt_digest_sha256",
        )

    release = builder.build_authoritative_vendor_release(
        repo_root=REPO_ROOT,
        observed_at=observed_at,
        **inputs,
    )
    encoded_evidence = json.dumps(release["disallowed_evidence"])
    encoded_details = json.dumps(release["blocker_details"])
    encoded_freshness = json.dumps(release["dependency_bindings"])
    assert release["deployment_ready"] is False
    assert expected in encoded_evidence + encoded_details + encoded_freshness
    if mutation in {"fixture", "shim", "unsupported", "proposal"}:
        assert (
            release["acceptance"]["no_fixture_shim_unsupported_proposal_or_stale_lane"]
            is False
        )
    if mutation == "stale":
        assert release["acceptance"]["dependencies_fresh"] is False


def test_resealing_a_modified_checked_receipt_cannot_restore_authority(
    builder,
) -> None:
    observed_at = _now()
    inputs = _inputs(builder, observed_at)
    checked = json.loads(
        (REPO_ROOT / builder.DEFAULT_RELEASE_CANDIDATE_RELATIVE).read_text(
            encoding="utf-8"
        )
    )
    checked["status"] = "role_aware_release_candidate_ready"
    checked["blockers"] = []
    checked["observed_at"] = observed_at
    inputs["role_aware_release_candidate"] = _seal(
        builder,
        checked,
        "candidate_identity",
    )
    release = builder.build_authoritative_vendor_release(
        repo_root=REPO_ROOT,
        observed_at=observed_at,
        **inputs,
    )
    binding = release["dependency_bindings"]["role_aware_release_candidate"]
    assert binding["identity_valid"] is True
    assert binding["repository_body_matches"] is False
    assert binding["repository_content_bound"] is False
    assert binding["binding_valid"] is False
    assert (
        "payload_does_not_match_selected_repository_file" in binding["binding_failures"]
    )


def test_backdating_both_release_and_evidence_fails_current_wall_clock(
    builder,
) -> None:
    backdated = (datetime.now(UTC) - timedelta(days=3)).strftime("%Y-%m-%dT%H:%M:%SZ")
    release = builder.build_authoritative_vendor_release(
        repo_root=REPO_ROOT,
        observed_at=backdated,
        **_inputs(builder, backdated),
    )
    freshness = release["dependency_bindings"]["ergoai_managed_vendor_live"][
        "freshness"
    ]
    assert freshness["valid"] is False
    assert freshness["wall_clock_age_seconds"] > (
        builder.AUTHORITATIVE_VENDOR_RELEASE_MAX_EVIDENCE_AGE_SECONDS
    )
    assert "evidence_wall_clock_age_exceeds_policy" in freshness["failures"]
    assert "release_observed_at_age_exceeds_policy" in freshness["failures"]


def test_cli_standalone_mode_does_not_rewrite_completion(builder, tmp_path) -> None:
    completion = REPO_ROOT / builder.DEFAULT_RECEIPT_RELATIVE
    before = completion.read_bytes()
    output = tmp_path / "authoritative-vendor-release.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(BUILDER_PATH),
            "--repo-root",
            str(REPO_ROOT),
            "--authoritative-vendor-release-output",
            str(output),
            "--observed-at",
            _now(),
            "--quiet",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    assert output.is_file()
    assert completion.read_bytes() == before
    generated = json.loads(output.read_text(encoding="utf-8"))
    assert generated["interface"] == INTERFACE
    assert generated["deployment_ready"] is False
