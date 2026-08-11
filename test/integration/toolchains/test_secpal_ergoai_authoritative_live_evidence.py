"""Cross-vendor authority boundary for SecPAL and ErgoAI (FVT-G221).

This suite intentionally does not install or execute either tool.  It checks
that durable live receipts retain their distinct authority ceilings and that
authentic compatibility/fixture evidence cannot become deployment authority.
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
BUILDER_PATH = (
    REPO_ROOT / "tools" / "logic" / "build_formal_verification_tactician_receipt.py"
)
RELEASE_PATH = (
    REPO_ROOT
    / "docs"
    / "architecture"
    / "formal_verification_authoritative_vendor_release.json"
)
SECPAL_PATH = (
    REPO_ROOT / "docs" / "architecture" / "formal_verification_secpal_live_receipt.json"
)
ERGOAI_JAVA_PATH = (
    REPO_ROOT
    / "docs"
    / "architecture"
    / "formal_verification_ergoai_java_api_live_receipt.json"
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
    return _load(BUILDER_PATH, "fvt_secpal_ergoai_authority_builder")


def _seal(builder, payload: dict[str, Any]) -> dict[str, Any]:
    sealed = copy.deepcopy(payload)
    sealed.pop("receipt_digest_sha256", None)
    sealed["receipt_digest_sha256"] = builder.content_digest(sealed).removeprefix(
        "sha256:"
    )
    return sealed


def _secpal_counterfactual(builder) -> dict[str, Any]:
    """All technical cases pass; exact reviewed EULA still blocks release."""

    return _seal(
        builder,
        {
            "interface": "SecPALAuthoritativeLiveEvidence@1",
            "schema_version": "secpal-authoritative-live-evidence/v1",
            "goal_id": "FVT-G219",
            "task_id": "FVT-086",
            "observed_at": "2026-08-03T12:00:00Z",
            "artifact": {
                "sha256": builder.SECPAL_RESEARCH_RELEASE_MSI_SHA256,
                "size_bytes": builder.SECPAL_RESEARCH_RELEASE_MSI_SIZE_BYTES,
                "product_code": builder.SECPAL_RESEARCH_RELEASE_PRODUCT_CODE,
                "product_version": builder.SECPAL_RESEARCH_RELEASE_PRODUCT_VERSION,
                "target_clr": builder.SECPAL_RESEARCH_RELEASE_CLR_VERSION,
            },
            "authenticode": {
                "verified": True,
                "timestamp": "2007-06-09T00:00:00Z",
            },
            "license": {
                "terms_reviewed": True,
                "production_use_allowed": True,
                "redistribution_allowed": False,
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
    )


def _ergoai_synthetic_counterfactual(builder) -> dict[str, Any]:
    """Self-consistent claims that are deliberately not live authority."""

    return _seal(
        builder,
        {
            "interface": "LiveErgoAIAdvisorCertification@1",
            "schema_version": "live-ergoai-advisor-certification/v1",
            "goal_id": "FVT-G160",
            "task_id": "FVT-050",
            "observed_at": "2026-08-03T12:00:00Z",
            "test_fixture_class": "synthetic_counterfactual_never_authoritative",
            "selected_platform": "linux-aarch64",
            "managed_vendor_live_evidence": True,
            "vendor_certified": True,
            "production_certified": True,
            "managed_identity": {
                "version": "3.0",
                "release_artifact_sha256": (
                    "46f9747db118567a7da50f70b439e35e"
                    "e36ea02c3dfde971a57c77a8ce94aa01"
                ),
                "vendor_executable_sha256": "a" * 64,
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
    )


def test_release_binds_distinct_vendor_surfaces_and_stays_blocked(builder) -> None:
    release = json.loads(RELEASE_PATH.read_text(encoding="utf-8"))
    dependencies = release["dependency_bindings"]
    assert "secpal_authoritative_live" in dependencies
    assert "ergoai_managed_vendor_live" in dependencies
    assert dependencies["secpal_authoritative_live"]["path"].endswith(
        "formal_verification_secpal_live_receipt.json"
    )
    # The optional Java/JDK receipt may be disclosed as fallback current
    # evidence, but its interface can never impersonate the core ErgoAI receipt.
    ergo = dependencies["ergoai_managed_vendor_live"]
    if ergo["fallback_used"]:
        assert ergo["path"].endswith(
            "formal_verification_ergoai_java_api_live_receipt.json"
        )
        assert ergo["interface_valid"] is False
    assert release["deployment_ready"] is False
    assert (
        release["policy"]["operator_compatibility_is_not_vendor_platform_support"]
        is True
    )
    assert release["policy"]["advisory_ergoai_never_grants_proof_authority"] is True


def test_authentic_secpal_technical_success_does_not_override_vendor_terms(
    builder,
) -> None:
    receipt = _secpal_counterfactual(builder)
    audit = builder._audit_secpal_authoritative_live(receipt)
    assert audit["official_release_identity_bound"] is True
    assert audit["microsoft_authenticode_verified"] is True
    assert audit["vendor_supported_platform"] is True
    assert audit["live_vendor_execution"] is True
    assert audit["semantic_cases"]["valid"] is True
    assert audit["known_eula_live_purpose_block"] is True
    assert audit["production_use_allowed"] is False
    assert audit["valid"] is False


def test_mono_sample_compatibility_is_preserved_but_not_platform_authority(
    builder,
) -> None:
    receipt = _secpal_counterfactual(builder)
    receipt["platform"]["vendor_supported"] = False
    receipt["operator_compatibility_only"] = True
    receipt["compatibility_execution"] = {
        "runtime": "Ubuntu Mono 6.8.0.105",
        "host": "linux-aarch64",
        "authentic_microsoft_sample_scenarios_passed": 18,
        "authentic_microsoft_sample_scenario_count": 18,
    }
    receipt = _seal(builder, receipt)
    audit = builder._audit_secpal_authoritative_live(receipt)
    assert audit["live_vendor_execution"] is True
    assert audit["operator_compatibility_only"] is True
    assert audit["vendor_supported_platform"] is False
    assert "vendor_supported_platform" in audit["blockers"]
    assert "not_operator_compatibility_only" in audit["blockers"]
    assert audit["valid"] is False


def test_synthetic_ergoai_claims_cannot_grant_advisory_live_authority(
    builder,
) -> None:
    receipt = _ergoai_synthetic_counterfactual(builder)
    audit = builder._audit_ergoai_authoritative_live(
        receipt,
        repo_root=REPO_ROOT,
    )
    assert audit["valid"] is False
    assert audit["repository_receipt_bound"] is False
    assert audit["independent_rederivation_succeeded"] is False
    assert audit["claimed_semantic_cases"]["valid"] is True
    assert audit["semantic_cases"]["valid"] is False
    assert audit["raw_checks_digest_bound"] is False
    assert audit["authority_ceiling"] == "advisory"
    assert "exact_committed_repository_receipt_bound" in audit["blockers"]

    forged = copy.deepcopy(receipt)
    forged["grants_proof_authority"] = True
    forged["authority_ceiling"] = "certified"
    forged = _seal(builder, forged)
    forged_audit = builder._audit_ergoai_authoritative_live(
        forged,
        repo_root=REPO_ROOT,
    )
    assert forged_audit["valid"] is False
    assert "advisory_authority_ceiling_respected" in forged_audit["blockers"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("is_fixture", True),
        ("is_hermetic_advisor_shim", True),
        ("evidence_class", "proposal_only_semantics"),
        ("status", "unsupported"),
    ],
)
def test_ergoai_fixture_shim_proposal_and_unsupported_labels_are_disclosed(
    builder,
    field: str,
    value: Any,
) -> None:
    receipt = _ergoai_synthetic_counterfactual(builder)
    receipt[field] = value
    findings = builder._authoritative_disallowed_findings(receipt)
    assert findings
    assert any(field in finding["path"] for finding in findings)


def test_optional_java_receipt_cannot_substitute_for_core_ergoai(builder) -> None:
    if not ERGOAI_JAVA_PATH.is_file():
        pytest.skip("optional Java/JDK receipt is not present in this checkout")
    java_receipt = json.loads(ERGOAI_JAVA_PATH.read_text(encoding="utf-8"))
    assert java_receipt["interface"] == "ErgoAIJavaAPILiveCertification@1"
    audit = builder._audit_ergoai_authoritative_live(
        java_receipt,
        repo_root=REPO_ROOT,
    )
    assert audit["valid"] is False
    assert audit["managed_vendor_live_evidence"] is False
    assert any(
        finding["reason"] == "disallowed_boolean:is_hermetic_advisor_shim"
        for finding in builder._authoritative_disallowed_findings(java_receipt)
    )


@pytest.mark.parametrize(
    ("mutation", "value"),
    [
        ("artifact", "b" * 64),
        ("platform", "synthetic-os-quantum"),
        (
            "cases",
            [
                {"kind": kind, "status": "passed"}
                for kind in (
                    "entailment",
                    "non_entailment",
                    "contradiction",
                    "mutation",
                    "replay",
                    "malformed",
                    "timeout",
                    "resource_bound",
                )
            ],
        ),
    ],
)
def test_arbitrary_ergoai_identity_platform_or_cases_fail_closed(
    builder,
    mutation: str,
    value: Any,
) -> None:
    receipt = _ergoai_synthetic_counterfactual(builder)
    if mutation == "artifact":
        receipt["managed_identity"]["release_artifact_sha256"] = value
    elif mutation == "platform":
        receipt["selected_platform"] = value
        receipt["managed_identity"]["selected_platform"] = value
    else:
        receipt["cases"] = value
    receipt = _seal(builder, receipt)
    audit = builder._audit_ergoai_authoritative_live(
        receipt,
        repo_root=REPO_ROOT,
    )
    assert audit["valid"] is False
    assert audit["repository_receipt_bound"] is False
    assert audit["independent_rederivation_succeeded"] is False
    assert audit["raw_checks_digest_bound"] is False
    if mutation in {"artifact", "platform"}:
        assert audit["lock_identity_bound"] is False


def test_checked_secpal_receipt_if_present_remains_public_and_non_deploying(
    builder,
) -> None:
    if not SECPAL_PATH.is_file():
        pytest.skip("external SecPAL receipt has not been published yet")
    receipt = json.loads(SECPAL_PATH.read_text(encoding="utf-8"))
    assert receipt["interface"] == "SecPALAuthoritativeLiveEvidence@1"
    encoded = SECPAL_PATH.read_text(encoding="utf-8")
    assert "SecPal_Research_Release.msi" not in encoded or (
        "artifact_bytes" not in receipt
    )
    audit = builder._audit_secpal_authoritative_live(receipt)
    assert audit["production_use_allowed"] is False
    assert audit["valid"] is False
