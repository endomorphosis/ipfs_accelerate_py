"""Fail-closed end-to-end formal-verification assurance matrix.

FVT-088 / FVT-G220 — ``FormalVerificationEndToEndAssuranceMatrix@1``.

The semantic certificate is only one input.  Each provider/host tuple must
independently bind dependency, packaging, installer, capability, semantic,
platform, authority, freshness, and public-surface evidence before its joint
readiness claim can become true.
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

REPO_ROOT = Path(__file__).resolve().parents[3]
CERTIFIER_PATH = (
    REPO_ROOT / "tools" / "logic" / "certify_formal_verification_toolchains.py"
)
MATRIX_PATH = (
    REPO_ROOT
    / "docs"
    / "architecture"
    / "formal_verification_end_to_end_assurance_matrix.json"
)
CERTIFICATE_PATH = (
    REPO_ROOT
    / "docs"
    / "architecture"
    / "formal_verification_toolchain_certificate.json"
)
LOCK_PATH = REPO_ROOT / "config" / "formal_verification_toolchains.lock.json"

INTERFACE = "FormalVerificationEndToEndAssuranceMatrix@1"
SCHEMA = "formal-verification-end-to-end-assurance-matrix/v1"
GOAL_ID = "FVT-G220"
TASK_ID = "FVT-088"
AXES = (
    "dependency",
    "packaging",
    "installer",
    "capability",
    "semantic",
    "platform",
    "authority",
    "freshness",
    "public_surface",
)
REQUIRED_FAILURE_CLASSES = {
    "supported_missing_dependencies",
    "missing_wheel_files",
    "placeholder_dispatch",
    "stale_lock",
    "wrong_architecture_artifact",
    "parser_fixture",
    "advisor_only_evidence",
    "unsupported_host",
}
SHA256 = re.compile(r"sha256:[0-9a-f]{64}")


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
    return _load(CERTIFIER_PATH, "fvt_end_to_end_assurance_certifier")


@pytest.fixture(scope="module")
def matrix() -> dict[str, Any]:
    return json.loads(MATRIX_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def trusted_certificate() -> dict[str, Any]:
    return json.loads(CERTIFICATE_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def lock() -> dict[str, Any]:
    return json.loads(LOCK_PATH.read_text(encoding="utf-8"))


def _providers(matrix: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(row["provider_id"]): row for row in matrix["provider_host_rows"]
    }


def _all_ready_axes(certifier) -> dict[str, dict[str, Any]]:
    evidence = certifier._assurance_inline_evidence_ref(
        "synthetic_axis_evidence", {"scope": "adversarial_test"}
    )
    return {
        axis_name: certifier._assurance_axis(
            "ready",
            required=True,
            reason_codes=[f"{axis_name}_synthetically_ready"],
            evidence_refs=[evidence],
        )
        for axis_name in AXES
    }


def test_expected_outputs_and_contract_constants_exist(certifier) -> None:
    assert CERTIFIER_PATH.is_file()
    assert MATRIX_PATH.is_file()
    assert Path(__file__).is_file()
    assert certifier.END_TO_END_ASSURANCE_INTERFACE == INTERFACE
    assert certifier.END_TO_END_ASSURANCE_SCHEMA == SCHEMA
    assert certifier.END_TO_END_ASSURANCE_GOAL_ID == GOAL_ID
    assert certifier.END_TO_END_ASSURANCE_TASK_ID == TASK_ID
    assert tuple(certifier.END_TO_END_ASSURANCE_AXES) == AXES
    assert (
        certifier.DEFAULT_END_TO_END_ASSURANCE_RELATIVE.as_posix()
        == "docs/architecture/formal_verification_end_to_end_assurance_matrix.json"
    )


def test_checked_matrix_is_independently_valid_and_fail_closed(
    certifier,
    matrix: dict[str, Any],
    trusted_certificate: dict[str, Any],
) -> None:
    validation = certifier.validate_end_to_end_assurance_matrix(
        matrix,
        repo_root=REPO_ROOT,
        certificate=trusted_certificate,
    )
    assert validation == {
        "valid": True,
        "failures": [],
        "recomputed_deployment_ready": False,
        "rows_validated": len(matrix["provider_host_rows"]),
        "axes_validated": len(matrix["provider_host_rows"]) * len(AXES),
    }
    assert matrix["interface"] == INTERFACE
    assert matrix["schema_version"] == SCHEMA
    assert matrix["goal_id"] == GOAL_ID
    assert matrix["task_id"] == TASK_ID
    assert matrix["summary"]["audit_complete"] is True
    assert matrix["summary"]["deployment_ready"] is False
    assert matrix["summary"]["status"] == "deployment_blocked"
    assert matrix["claims"]["one_axis_cannot_inherit_another_axis_success"] is True
    assert matrix["claims"]["unsupported_is_not_ready"] is True


def test_every_lock_provider_has_exact_independent_axes_and_evidence(
    certifier,
    matrix: dict[str, Any],
    lock: dict[str, Any],
) -> None:
    expected = {str(row["tool_id"]) for row in lock["tools"]}
    providers = _providers(matrix)
    assert set(providers) == expected
    assert matrix["coverage"]["all_lock_providers_present"] is True
    assert matrix["coverage"]["all_rows_have_exact_axes"] is True

    for provider_id, row in providers.items():
        assert row["row_id"] == (
            f"{provider_id}@{row['host']['platform_id']}"
        )
        assert tuple(row["axes"]) == AXES
        assert row["identity_boundary"]["inherits_evidence_from"] == []
        for axis_name, axis in row["axes"].items():
            assert axis["state"] in certifier.END_TO_END_ASSURANCE_STATES
            assert axis["ready"] is (axis["state"] == "ready")
            assert type(axis["required"]) is bool
            assert axis["reason_codes"]
            assert axis["evidence_refs"]
            for ref in axis["evidence_refs"]:
                assert ref["kind"] in {"repository_file", "inline_digest"}
                if ref["present"]:
                    assert SHA256.fullmatch(str(ref["sha256"]))
                if ref["kind"] == "repository_file":
                    path = Path(str(ref["path"]))
                    assert not path.is_absolute()
                    assert ".." not in path.parts
        assert row["joint_ready"] is certifier.provider_host_joint_ready(
            row["axes"]
        )


def test_failure_classes_are_explicitly_distinguishable(
    certifier,
    matrix: dict[str, Any],
) -> None:
    assert set(matrix["validation"]["required_failure_classes"]) == (
        REQUIRED_FAILURE_CLASSES
    )
    state_for_failure = {
        "supported_missing_dependencies": ("dependency", "blocked"),
        "missing_wheel_files": ("packaging", "blocked"),
        "placeholder_dispatch": ("installer", "blocked"),
        "stale_lock": ("freshness", "blocked"),
        "wrong_architecture_artifact": ("platform", "blocked"),
        "parser_fixture": ("semantic", "blocked"),
        "advisor_only_evidence": ("semantic", "blocked"),
        "unsupported_host": ("platform", "unsupported"),
    }
    for failure_class, (axis_name, state) in state_for_failure.items():
        axes = _all_ready_axes(certifier)
        axes[axis_name] = certifier._assurance_axis(
            state,
            required=True,
            reason_codes=[failure_class],
            evidence_refs=[
                certifier._assurance_inline_evidence_ref(
                    failure_class, {"failure_class": failure_class}
                )
            ],
        )
        assert certifier.provider_host_joint_ready(axes) is False
        assert axes[axis_name]["reason_codes"] == [failure_class]


@pytest.mark.parametrize("mutated_axis", AXES)
def test_adversarial_mutation_of_every_axis_fails_joint_readiness(
    certifier,
    mutated_axis: str,
) -> None:
    axes = _all_ready_axes(certifier)
    assert certifier.provider_host_joint_ready(axes) is True
    axes[mutated_axis] = certifier._assurance_axis(
        "blocked",
        required=True,
        reason_codes=[f"adversarial_{mutated_axis}_mutation"],
        evidence_refs=[
            certifier._assurance_inline_evidence_ref(
                f"adversarial_{mutated_axis}", {"mutated": mutated_axis}
            )
        ],
    )
    assert certifier.provider_host_joint_ready(axes) is False


def test_secpal_reference_and_external_vendor_identity_never_collapse(
    matrix: dict[str, Any],
    lock: dict[str, Any],
) -> None:
    providers = _providers(matrix)
    reference = providers["secpal-authorization"]
    external = providers["secpal"]
    assert matrix["coverage"]["secpal_identities_separate"] is True
    assert reference["identity_boundary"]["identity_class"] == (
        "in_process_secpal_reference"
    )
    assert external["identity_boundary"]["identity_class"] == (
        "external_secpal_vendor"
    )
    assert reference["identity_boundary"]["identity_class"] != (
        external["identity_boundary"]["identity_class"]
    )
    assert reference["identity_boundary"]["inherits_evidence_from"] == []
    assert external["identity_boundary"]["inherits_evidence_from"] == []

    secpal_lock = next(row for row in lock["tools"] if row["tool_id"] == "secpal")
    artifact = secpal_lock["deployment_contract"]["vendor_install"][
        "operator_artifact"
    ]
    assert artifact["artifact_sha256"] == (
        "c1988b9f1f6a2fb602bac4fc777a1765e59e74126285a095684a4743ea683159"
    )
    assert artifact["artifact_size_bytes"] == 2458624
    assert artifact["authenticode_evidence"]["verified"] is True
    assert artifact["license_evidence"]["sha256"] == (
        "de075e7848fb737b9da3cfec5ce7c906742f4767fa04ed2bc38e69e2dd5e4fad"
    )
    assert artifact["redistribution_permitted"] is False
    assert artifact["production_use_permitted"] is False

    # Provenance and the non-certifying shadow boundary are valid.  They do
    # not leak into packaging, installation, capability, semantics, platform,
    # or joint deployability.
    assert external["axes"]["dependency"]["state"] == "ready"
    assert "official_restricted_artifact_provenance_bound" in (
        external["axes"]["dependency"]["reason_codes"]
    )
    assert external["axes"]["authority"]["state"] == "ready"
    assert external["axes"]["packaging"]["state"] == "blocked"
    assert external["axes"]["installer"]["state"] == "blocked"
    assert external["axes"]["capability"]["state"] == "blocked"
    assert external["axes"]["semantic"]["state"] == "blocked"
    assert external["axes"]["platform"]["state"] == "unsupported"
    assert external["joint_ready"] is False


def test_secpal_operator_compatibility_cannot_be_promoted(
    matrix: dict[str, Any],
    lock: dict[str, Any],
) -> None:
    secpal_lock = next(row for row in lock["tools"] if row["tool_id"] == "secpal")
    probe = secpal_lock["deployment_contract"]["vendor_install"][
        "operator_artifact"
    ]["operator_compatibility_probe"]
    assert probe["status"] == "observed_unbound"
    assert probe["sample_scenarios_executed"] == 18
    assert probe["sample_scenarios_exit_zero"] is True
    assert probe["runtime_package_identity_bound"] is False
    assert probe["arbitrary_policy_interface_verified"] is False
    assert probe["vendor_supported_platform"] is False
    assert probe["live_certification_eligible"] is False

    external = _providers(matrix)["secpal"]
    semantic = external["axes"]["semantic"]
    capability = external["axes"]["capability"]
    platform_axis = external["axes"]["platform"]
    assert semantic["state"] == "blocked"
    assert semantic["details"]["arbitrary_policy_semantics_certified"] is False
    assert semantic["details"]["operator_compatibility_observed"] is True
    assert semantic["details"]["operator_compatibility_scope"] == (
        "shipped_samples_only"
    )
    assert capability["details"]["operator_compatibility_is_vendor_support"] is False
    assert capability["details"]["operator_compatibility_is_live_capability"] is False
    assert (
        platform_axis["details"][
            "operator_compatibility_is_not_vendor_platform_support"
        ]
        is True
    )
    assert "secpal_live_semantic_cli_unavailable" in semantic["reason_codes"]


def test_ergoai_advice_remains_separate_from_independent_proof_authority(
    matrix: dict[str, Any],
) -> None:
    ergoai = _providers(matrix)["ergoai"]
    boundary = ergoai["identity_boundary"]
    assert matrix["coverage"]["ergoai_advisor_separate_from_proof_authority"] is True
    assert boundary["identity_class"] == "ergoai_advisor"
    assert boundary["role"] == "advisor"
    assert boundary["authority_ceiling"] == "advisory"
    assert boundary["can_satisfy_certified_authority"] is False
    assert boundary["independent_proof_authority"] is False
    assert boundary["independent_reconstruction_required"] is True
    assert ergoai["axes"]["semantic"]["state"] == "blocked"
    assert "advisor_only_evidence" in ergoai["axes"]["semantic"]["reason_codes"]
    assert ergoai["joint_ready"] is False
    assert any(
        row["identity_boundary"]["independent_proof_authority"] is True
        for row in matrix["provider_host_rows"]
        if row["provider_id"] != "ergoai"
    )


def test_stale_lock_digest_blocks_only_freshness_before_joint_recompute(
    certifier,
) -> None:
    certificate = json.loads(
        (REPO_ROOT / certifier.DEFAULT_CERTIFICATE_RELATIVE).read_text(
            encoding="utf-8"
        )
    )
    certificate.setdefault("lock", {})["digest_sha256"] = "0" * 64
    rebuilt = certifier.build_end_to_end_assurance_matrix(
        repo_root=REPO_ROOT,
        certificate=certificate,
        observed_at="2026-08-03T00:00:00+00:00",
    )
    for row in rebuilt["provider_host_rows"]:
        assert row["axes"]["freshness"]["state"] == "blocked"
        assert row["axes"]["freshness"]["reason_codes"] == [
            "stale_lock",
            "stale_lock_digest",
        ]
        assert row["joint_ready"] is False


def test_checked_matrix_surfaces_required_failure_class_tokens(
    matrix: dict[str, Any],
) -> None:
    """Live rows must use the catalog tokens, not only synthetic adversarial cases."""

    observed: set[str] = set()
    for row in matrix["provider_host_rows"]:
        for axis in row["axes"].values():
            observed.update(axis.get("reason_codes") or ())
    # Always-present fail-closed classes on the current blocked deployment.
    for required in (
        "supported_missing_dependencies",
        "placeholder_dispatch",
        "stale_lock",
        "parser_fixture",
        "advisor_only_evidence",
        "unsupported_host",
    ):
        assert required in observed, f"missing live failure class token: {required}"
    # Wheel-contract absence is encoded when packaging evidence is incomplete;
    # architecture mismatch is encoded only when a wrong-arch artifact is bound.
    for optional in ("missing_wheel_files", "wrong_architecture_artifact"):
        assert optional in matrix["validation"]["required_failure_classes"]


def test_validator_rejects_optimistic_claim_and_stale_repository_evidence(
    certifier,
    matrix: dict[str, Any],
    trusted_certificate: dict[str, Any],
) -> None:
    optimistic = copy.deepcopy(matrix)
    secpal = _providers(optimistic)["secpal"]
    secpal["joint_ready"] = True
    result = certifier.validate_end_to_end_assurance_matrix(
        optimistic,
        repo_root=REPO_ROOT,
        certificate=trusted_certificate,
    )
    assert result["valid"] is False
    assert any("joint_ready_not_derived" in item for item in result["failures"])
    assert "matrix_digest_mismatch" in result["failures"]

    stale = copy.deepcopy(matrix)
    first_ref = stale["provider_host_rows"][0]["axes"]["packaging"][
        "evidence_refs"
    ][0]
    assert first_ref["kind"] == "repository_file"
    first_ref["sha256"] = "sha256:" + "0" * 64
    stale = certifier.recompute_end_to_end_assurance_claims(stale)
    result = certifier.validate_end_to_end_assurance_matrix(
        stale,
        repo_root=REPO_ROOT,
        certificate=trusted_certificate,
    )
    assert result["valid"] is False
    assert any("digest_stale" in item for item in result["failures"])


def test_validator_rejects_fully_resealed_all_ready_forgery(
    certifier,
    trusted_certificate: dict[str, Any],
) -> None:
    canonical = certifier.build_end_to_end_assurance_matrix(
        repo_root=REPO_ROOT,
        certificate=trusted_certificate,
        observed_at="2026-08-03T00:00:00+00:00",
    )
    assert certifier.validate_end_to_end_assurance_matrix(
        canonical,
        repo_root=REPO_ROOT,
        certificate=trusted_certificate,
    )["valid"] is True

    forged = copy.deepcopy(canonical)
    for row in forged["provider_host_rows"]:
        for axis_name, axis in row["axes"].items():
            axis["state"] = "ready"
            axis["ready"] = True
            axis["required"] = True
            axis["reason_codes"] = [f"forged_{axis_name}_ready"]
    forged = certifier.recompute_end_to_end_assurance_claims(forged)
    assert forged["summary"]["deployment_ready"] is True

    result = certifier.validate_end_to_end_assurance_matrix(
        forged,
        repo_root=REPO_ROOT,
        certificate=trusted_certificate,
    )
    assert result["valid"] is False
    assert result["recomputed_deployment_ready"] is True
    assert any(
        failure.endswith(":axes_not_canonically_derived")
        for failure in result["failures"]
    )


def test_validator_binds_the_exact_trusted_certificate_body(
    certifier,
    trusted_certificate: dict[str, Any],
) -> None:
    canonical = certifier.build_end_to_end_assurance_matrix(
        repo_root=REPO_ROOT,
        certificate=trusted_certificate,
        observed_at="2026-08-03T00:00:00+00:00",
    )
    assert certifier.validate_end_to_end_assurance_matrix(
        canonical,
        repo_root=REPO_ROOT,
        certificate_path=CERTIFICATE_PATH,
    )["valid"] is True
    different_certificate = copy.deepcopy(trusted_certificate)
    different_certificate["program"] = "adversarial-certificate-substitution"
    different_certificate["certificate_digest_sha256"] = certifier.content_digest(
        {
            key: value
            for key, value in different_certificate.items()
            if key != "certificate_digest_sha256"
        }
    )

    result = certifier.validate_end_to_end_assurance_matrix(
        canonical,
        repo_root=REPO_ROOT,
        certificate=different_certificate,
    )
    assert result["valid"] is False
    assert "trusted_certificate_digest_mismatch" in result["failures"]


def test_public_matrix_never_publishes_restricted_vendor_bytes_or_host_paths(
    matrix: dict[str, Any],
) -> None:
    encoded = json.dumps(matrix, sort_keys=True)
    assert matrix["public_evidence_policy"]["satisfied"] is True
    assert "/home/" not in encoded
    assert "Binary/" not in encoded
    assert "SecPal_Research_Release.msi" not in encoded
    assert "%USERPROFILE%" not in encoded
    assert '"raw_process_output":' not in encoded


def test_cli_exposes_explicit_assurance_generation_surface() -> None:
    completed = subprocess.run(
        [sys.executable, str(CERTIFIER_PATH), "--help"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert "--end-to-end-assurance" in completed.stdout
    assert "--end-to-end-assurance-output" in completed.stdout
