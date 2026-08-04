"""Production-candidate SecPAL-style authorization provider (FVT-099 / FVT-G231).

``ProductionAuthorizationReplacement@1``

Closes the objective evidence gap for a separately named, project-owned
authorization prover with SecPAL-style typed delegation semantics.

Acceptance covered:

* new provider id distinct from ``secpal`` and ``secpal-authorization``;
* clean-room design record with no Microsoft MSI/sample/trademark claims;
* typed language covers principal identity, delegation depth/scope, can-say,
  can-act-as, roles, exclusions, revocation/time validity, conflict,
  unknown/no-proof, constraints, and deterministic proof/counterexample
  witnesses;
* positive, negative, mutation, replay, malformed, cycle/resource-bound,
  differential, fuzz/property, and denial-safety cases pass;
* public verification API binds the identity and authorization ceiling;
* provider cannot satisfy FVT-G219 or claim Microsoft SecPAL authority;
* deployment_ready remains false (FVT-G232 owns legal/deployment approval).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
PROVIDER_PATH = (
    REPO_ROOT
    / "ipfs_datasets_py"
    / "ipfs_datasets_py"
    / "logic"
    / "backends"
    / "secpal_style_authorization.py"
)
API_PATH = (
    REPO_ROOT
    / "ipfs_datasets_py"
    / "ipfs_datasets_py"
    / "logic"
    / "verification_api.py"
)
RECEIPT_PATH = (
    REPO_ROOT
    / "docs"
    / "architecture"
    / "formal_verification_production_authorization_replacement_receipt.json"
)

INTERFACE = "ProductionAuthorizationReplacement@1"
SCHEMA = "production-authorization-replacement/v1"
GOAL_ID = "FVT-G231"
TASK_ID = "FVT-099"
PROVIDER_ID = "production-authorization-replacement"
REFERENCE_ID = "secpal-authorization"
EXTERNAL_ID = "secpal"

REQUIRED_CASE_KINDS = {
    "positive",
    "negative",
    "mutation",
    "replay",
    "malformed",
    "cycle_resource_bound",
    "differential",
    "fuzz_property",
    "denial_safety",
}
REQUIRED_LANGUAGE_FEATURES = {
    "principal_identity",
    "delegation_depth",
    "delegation_scope",
    "can_say",
    "can_act_as",
    "roles",
    "exclusions",
    "revocation",
    "time_validity",
    "conflict",
    "unknown_no_proof",
    "constraints",
    "deterministic_proof_witness",
    "counterexample_witness",
}


def _ensure_import_paths() -> None:
    datasets_root = REPO_ROOT / "ipfs_datasets_py"
    for candidate in (str(REPO_ROOT), str(datasets_root)):
        if candidate not in sys.path:
            sys.path.insert(0, candidate)


@pytest.fixture(scope="module")
def provider_module():
    _ensure_import_paths()
    from ipfs_datasets_py.logic.backends import secpal_style_authorization as mod

    return mod


@pytest.fixture(scope="module")
def backend(provider_module):
    return provider_module.SecPALStyleAuthorizationBackend()


@pytest.fixture(scope="module")
def live_results(provider_module, backend):
    return provider_module.run_all_production_cases(backend=backend)


@pytest.fixture(scope="module")
def live_receipt(provider_module):
    return provider_module.build_production_authorization_replacement_receipt(
        repo_root=REPO_ROOT
    )


@pytest.fixture(scope="module")
def static_receipt() -> dict[str, Any]:
    assert RECEIPT_PATH.is_file(), f"missing receipt: {RECEIPT_PATH}"
    return json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def verification_api():
    _ensure_import_paths()
    from ipfs_datasets_py.logic import verification_api as api

    return api


# ---------------------------------------------------------------------------
# Artifact presence
# ---------------------------------------------------------------------------


def test_expected_outputs_exist() -> None:
    assert PROVIDER_PATH.is_file()
    assert API_PATH.is_file()
    assert RECEIPT_PATH.is_file()
    assert Path(__file__).is_file()


def test_provider_identity_constants(provider_module) -> None:
    assert (
        provider_module.PRODUCTION_AUTHORIZATION_REPLACEMENT_INTERFACE == INTERFACE
    )
    assert provider_module.PRODUCTION_AUTHORIZATION_REPLACEMENT_SCHEMA == SCHEMA
    assert provider_module.PRODUCTION_AUTHORIZATION_PROVIDER_ID == PROVIDER_ID
    assert provider_module.PRODUCTION_AUTHORIZATION_GOAL_ID == GOAL_ID
    assert provider_module.PRODUCTION_AUTHORIZATION_TASK_ID == TASK_ID
    assert provider_module.PRODUCTION_AUTHORIZATION_AUTHORITY_CEILING == "authorization"
    assert set(provider_module.REQUIRED_CASE_KINDS) == REQUIRED_CASE_KINDS
    assert set(provider_module.TYPED_LANGUAGE_FEATURES) == REQUIRED_LANGUAGE_FEATURES


def test_provider_id_is_separately_named(backend, provider_module) -> None:
    identity = backend.identity()
    assert identity["provider_id"] == PROVIDER_ID
    assert identity["provider_id"] != REFERENCE_ID
    assert identity["provider_id"] != EXTERNAL_ID
    assert EXTERNAL_ID not in identity["aliases"]
    assert REFERENCE_ID not in identity["aliases"]
    assert "secpal" not in {item.casefold() for item in identity["aliases"]}
    assert identity["forbids_fvt_g219_completion"] is True
    assert identity["forbids_microsoft_secpal_authority"] is True
    assert identity["forbids_deployment_authority"] is True
    assert identity["deployment_ready"] is False
    assert identity["legal_approval_complete"] is False
    assert identity["authority_ceiling"] == "authorization"
    provider_module.assert_identity_boundary(PROVIDER_ID)
    with pytest.raises(provider_module.ProductionAuthorizationError):
        provider_module.assert_identity_boundary(EXTERNAL_ID)
    with pytest.raises(provider_module.ProductionAuthorizationError):
        provider_module.assert_identity_boundary(REFERENCE_ID)


def test_clean_room_design_record_forbids_microsoft_bytes(provider_module) -> None:
    record = provider_module.CLEAN_ROOM_DESIGN_RECORD
    assert record["restricted_bytes_used"] is False
    assert record["microsoft_msi_used"] is False
    assert record["decompiled_code_used"] is False
    assert record["sample_source_used"] is False
    assert record["trademark_implication"] is False
    assert record["microsoft_vendor_compatibility_claim"] is False
    assert record["implementation_origin"] == "project-owned clean-room"
    assert provider_module.forbids_fvt_g219_completion() is True
    assert provider_module.forbids_microsoft_secpal_authority() is True


# ---------------------------------------------------------------------------
# Executable semantics
# ---------------------------------------------------------------------------


def test_all_required_case_kinds_pass(live_results) -> None:
    assert live_results
    by_kind = {kind: [] for kind in REQUIRED_CASE_KINDS}
    for result in live_results:
        assert result.passed, (
            f"{result.case_id} failed: got={result.outcome} "
            f"expected={result.expected_outcome} diag={result.diagnostics}"
        )
        if result.kind in by_kind:
            by_kind[result.kind].append(result)
    for kind in REQUIRED_CASE_KINDS:
        assert by_kind[kind], f"missing required case kind: {kind}"
        assert all(item.passed for item in by_kind[kind])


def test_positive_negative_conflict_unknown_outcomes(live_results) -> None:
    outcomes = {item.case_id: item.outcome for item in live_results}
    assert outcomes["case:positive-allow"] == "allow"
    assert outcomes["case:positive-can-say"] == "allow"
    assert outcomes["case:positive-can-act-as-delegation"] == "allow"
    assert outcomes["case:negative-deny"] == "deny"
    assert outcomes["case:negative-exclusion"] == "deny"
    assert outcomes["case:negative-revocation"] == "deny"
    assert outcomes["case:unknown-no-proof"] == "unknown"
    assert outcomes["case:conflict-explicit"] == "conflict"
    assert outcomes["case:denial-safety"] == "deny"


def test_mutation_changes_verdict(live_results) -> None:
    by_id = {item.case_id: item for item in live_results}
    base = by_id["case:positive-allow"]
    mutated = by_id["case:mutation-role-membership"]
    assert base.passed and base.outcome == "allow"
    assert mutated.passed and mutated.outcome == "deny"
    assert mutated.outcome != base.outcome


def test_replay_is_deterministic(live_results) -> None:
    replay = next(item for item in live_results if item.kind == "replay")
    assert replay.passed
    assert replay.witness_digest
    assert len(replay.witness_digest) == 64


def test_malformed_fails_closed(live_results) -> None:
    malformed = next(item for item in live_results if item.kind == "malformed")
    assert malformed.passed
    assert malformed.outcome == "malformed_rejected"


def test_differential_agrees_on_outcome_not_identity(
    live_results, provider_module
) -> None:
    differential = next(
        item for item in live_results if item.kind == "differential"
    )
    assert differential.passed
    meta = differential.metadata.to_dict()
    assert meta["production_provider_id"] == PROVIDER_ID
    assert meta["reference_provider_id"] == REFERENCE_ID
    assert meta["external_provider_id"] == EXTERNAL_ID
    assert meta["production_provider_id"] != meta["reference_provider_id"]
    assert meta["production_provider_id"] != meta["external_provider_id"]


def test_fuzz_property_and_denial_safety(live_results) -> None:
    fuzz = next(item for item in live_results if item.kind == "fuzz_property")
    denial = next(item for item in live_results if item.kind == "denial_safety")
    assert fuzz.passed
    assert denial.passed
    assert denial.outcome == "deny"
    assert fuzz.metadata.to_dict().get("violations", 1) == 0


def test_typed_language_features_covered(live_receipt) -> None:
    coverage = live_receipt["typed_language"]["coverage"]
    assert set(live_receipt["typed_language"]["features"]) == REQUIRED_LANGUAGE_FEATURES
    for feature in REQUIRED_LANGUAGE_FEATURES:
        assert coverage.get(feature) is True, f"language feature uncovered: {feature}"
    assert live_receipt["typed_language"]["all_required_features_covered"] is True


# ---------------------------------------------------------------------------
# Receipt evidence
# ---------------------------------------------------------------------------


def test_static_receipt_schema_and_identity(static_receipt) -> None:
    assert static_receipt["schema_version"] == SCHEMA
    assert static_receipt["interface"] == INTERFACE
    assert static_receipt["goal_id"] == GOAL_ID
    assert static_receipt["task_id"] == TASK_ID
    assert static_receipt["certified"] is True
    assert static_receipt["deployment_ready"] is False
    assert static_receipt["legal_approval_complete"] is False
    provider = static_receipt["provider"]
    assert provider["provider_id"] == PROVIDER_ID
    assert provider["authority_ceiling"] == "authorization"
    assert provider["forbids_fvt_g219_completion"] is True
    assert provider["forbids_microsoft_secpal_authority"] is True
    boundary = static_receipt["identity_boundary"]
    assert boundary["cannot_satisfy_fvt_g219"] is True
    assert boundary["cannot_claim_microsoft_secpal_authority"] is True
    assert boundary["ids_are_pairwise_distinct"] is True
    assert boundary["provider_id"] != boundary["reference_provider_id"]
    assert boundary["provider_id"] != boundary["external_provider_id"]
    assert set(static_receipt["cases"]["required_kinds"]) == REQUIRED_CASE_KINDS
    assert static_receipt["cases"]["all_required_kinds_passed"] is True
    assert static_receipt["cases"]["all_passed"] is True
    assert static_receipt["acceptance"]["cannot_satisfy_fvt_g219"] is True
    assert static_receipt["acceptance"]["deployment_ready"] is False
    assert static_receipt["authority"]["ceiling"] == "authorization"
    assert static_receipt["authority"]["forbids_theorem_authority"] is True
    assert static_receipt["clean_room"]["microsoft_msi_used"] is False


def test_live_receipt_matches_static_acceptance(
    live_receipt, static_receipt
) -> None:
    assert live_receipt["certified"] is True
    assert live_receipt["provider"]["provider_id"] == static_receipt["provider"][
        "provider_id"
    ]
    assert live_receipt["cases"]["all_passed"] is True
    assert set(live_receipt["cases"]["required_kinds"]) == set(
        static_receipt["cases"]["required_kinds"]
    )
    # Live rebuild must still pass every required kind.
    assert live_receipt["cases"]["all_required_kinds_passed"] is True
    assert live_receipt["deployment_ready"] is False


def test_receipt_binds_provider_module_bytes(static_receipt) -> None:
    bindings = static_receipt["bindings"]
    assert "secpal_style_authorization.py" in bindings["provider_module"]
    assert bindings["provider_module_sha256"]
    assert len(bindings["provider_module_sha256"]) == 64
    assert "verification_api.py" in bindings["verification_api"]
    assert "test_production_authorization_replacement.py" in bindings[
        "integration_test"
    ]


# ---------------------------------------------------------------------------
# Public verification API surface
# ---------------------------------------------------------------------------


def test_verification_api_lists_production_provider(verification_api) -> None:
    response = verification_api.list_providers()
    assert response.status.value in {"declarative", "succeeded"}
    providers = response.result["providers"]
    match = [
        item
        for item in providers
        if item.get("provider_id") == PROVIDER_ID
    ]
    assert match, "production-authorization-replacement missing from list_providers"
    entry = match[0]
    assert entry.get("authority_ceiling") == "authorization"
    assert entry.get("forbids_fvt_g219_completion") is True
    assert entry.get("deployment_ready") is False
    # Must not collapse into the external secpal id.
    assert entry.get("provider_id") != EXTERNAL_ID


def test_verification_api_probe_production_provider(verification_api) -> None:
    response = verification_api.probe_provider(PROVIDER_ID)
    assert response.status.value == "succeeded"
    assert response.result["available"] is True
    assert response.result["provider_id"] == PROVIDER_ID
    assert response.result["forbids_fvt_g219_completion"] is True
    assert response.result["deployment_ready"] is False
    identity = response.result["identity"]
    assert identity["provider_id"] == PROVIDER_ID
    assert identity["distinct_from_external_id"] == EXTERNAL_ID
    assert identity["distinct_from_reference_id"] == REFERENCE_ID


def test_verification_api_identity_and_receipt_operations(verification_api) -> None:
    identity = verification_api.production_authorization_identity()
    assert identity.provider_id == PROVIDER_ID
    assert identity.result["identity"]["provider_id"] == PROVIDER_ID
    assert (
        identity.result["interface"] == INTERFACE
        or identity.result["identity"]["interface"] == INTERFACE
    )
    receipt_response = verification_api.production_authorization_receipt(
        repo_root=str(REPO_ROOT)
    )
    assert receipt_response.result["certified"] is True
    assert receipt_response.result["deployment_ready"] is False
    assert receipt_response.result["forbids_fvt_g219_completion"] is True
    receipt = receipt_response.result["receipt"]
    assert receipt["provider"]["provider_id"] == PROVIDER_ID


def test_verification_api_check_routes_to_production_provider(
    verification_api, provider_module
) -> None:
    cases = provider_module.build_production_cases()
    allow = next(item for item in cases if item.case_id == "case:positive-allow")
    response = verification_api.check(
        {
            "query_kind": "policy_approval",
            "logic_family": "authorization",
            "requested_backend_id": PROVIDER_ID,
            "payload": {
                "encoding": "authorization-ir",
                "authorization_ir": allow.document.to_dict(),
                "query_id": allow.query.query_id,
            },
            "bounds": {"timeout_ms": 500, "max_steps": 256},
        }
    )
    assert response.provider_id == PROVIDER_ID
    assert response.authority.value == "authorization"
    assert response.result["outcome"] == "allow"
    assert response.result["forbids_fvt_g219_completion"] is True
    assert response.result["forbids_microsoft_secpal_authority"] is True
    assert response.result["deployment_ready"] is False
    assert response.result["is_theorem_authority"] is False


def test_cannot_satisfy_fvt_g219_or_claim_microsoft_authority(
    static_receipt, backend
) -> None:
    """The production replacement is not Microsoft SecPAL live evidence."""

    assert static_receipt["acceptance"]["cannot_satisfy_fvt_g219"] is True
    assert static_receipt["acceptance"]["cannot_claim_microsoft_secpal_authority"] is True
    assert static_receipt["identity_boundary"]["cannot_satisfy_fvt_g219"] is True
    identity = backend.identity()
    assert identity["forbids_fvt_g219_completion"] is True
    # No claim markers that would re-open G219.
    blob = json.dumps(static_receipt).casefold()
    assert "fvt-g219 complete" not in blob
    assert "microsoft secpal authority" not in blob
    assert identity["provider_id"] != EXTERNAL_ID


def test_public_surface_packaging_and_lazy_dependencies(static_receipt) -> None:
    surface = static_receipt["public_surface"]
    assert surface["verification_api_bound"] is True
    assert surface["provider_id"] == PROVIDER_ID
    assert surface["requires_install"] is False
    assert surface["lazy_dependencies"] == []
    assert surface["hammer_advisor_authority"] is False
    assert surface["cache_safe"] is True
    assert "in-process" in surface["packaging"].casefold()
