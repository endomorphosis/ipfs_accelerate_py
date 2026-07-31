"""Production ZKP circuit deployment certification (FVT-047 / FVT-G190).

Exercises ``config/formal_verification_zkp_deployment.lock.json`` and
``tools/logic/certification/zkp.py`` (``ZKPDeploymentCertification@1``).

Acceptance covered:

* circuit, ceremony, proving-key and verification-key digests, public-input
  schema, backend, expiry, freshness, and revocation are exact and reviewable;
* live positive verification and corrupted proof/key/public-input, circuit
  mismatch, mutation, replay, stale, and revoked cases pass;
* private witnesses and secrets never enter Git, logs, caches, public
  receipts, or model context;
* ZKP authority attests an underlying receipt and never replaces semantic
  theorem authority.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
LOCK_PATH = REPO_ROOT / "config" / "formal_verification_zkp_deployment.lock.json"
CERT_PATH = REPO_ROOT / "tools" / "logic" / "certification" / "zkp.py"
SHARED_LOCK_PATH = REPO_ROOT / "config" / "formal_verification_toolchains.lock.json"

INTERFACE = "ZKPDeploymentCertification@1"
LOCK_INTERFACE = "ZKPDeploymentLock@1"
SCHEMA_VERSION = "zkp-deployment-certification/v1"
LOCK_SCHEMA_VERSION = "zkp-deployment-lock/v1"
GOAL_ID = "FVT-G190"
TASK_ID = "FVT-047"
LANE_ID = "attestation"
TOOL_ID = "zkp-circuit"
HANDLER_ID = "zkp_deployment_certifier"
AUTHORITY_CEILING = "attestation"
AUTHORITY_SCOPE = "receipt_attestation_only"

_SHA256_CID = re.compile(r"^sha256:[0-9a-f]{64}$")

REQUIRED_CASE_KINDS = {
    "positive",
    "corrupted_proof",
    "corrupted_key",
    "corrupted_public_input",
    "circuit_mismatch",
    "mutation",
    "replay",
    "stale",
    "revoked",
    "secret_safety",
    "authority",
}

REQUIRED_PUBLIC_INPUT_KEYS = {
    "theorem_id",
    "property_id",
    "translation_receipt_id",
    "receipt_id",
    "tree_id",
    "policy_id",
    "circuit_id",
    "circuit_version",
    "ceremony_id",
    "crs_id",
    "proving_key_id",
    "verification_key_id",
    "backend_id",
    "backend_version",
    "backend_mode",
    "revocation_policy_id",
    "issued_at",
    "expires_at",
    "underlying_authority",
    "underlying_status",
    "source_result_digest",
}

FORBIDDEN_SECRET_NEEDLES = (
    "private-witness-FVT047-SECRET-AXIOM-NEVER-LEAK",
    "BEGIN PRIVATE",
    "BEGIN RSA PRIVATE",
    "coordinator_seed=",
    "toxic_waste=",
    "trapdoor=",
    "witness_bytes=",
)


def _load_module(path: Path, name: str):
    assert path.is_file(), f"missing expected output: {path}"
    datasets_root = REPO_ROOT / "ipfs_datasets_py"
    for candidate in (str(REPO_ROOT), str(datasets_root)):
        if candidate not in sys.path:
            sys.path.insert(0, candidate)
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def zkp_cert():
    return _load_module(CERT_PATH, "tools_logic_certification_zkp")


@pytest.fixture(scope="module")
def lock() -> dict[str, Any]:
    assert LOCK_PATH.is_file(), f"missing deployment lock: {LOCK_PATH}"
    payload = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


@pytest.fixture(scope="module")
def receipt(zkp_cert, lock) -> dict[str, Any]:
    return zkp_cert.build_certification_receipt(
        repo_root=REPO_ROOT,
        lock_path=LOCK_PATH,
        lock=lock,
        env=zkp_cert.offline_env(),
    )


# ---------------------------------------------------------------------------
# Expected outputs / lock contract
# ---------------------------------------------------------------------------


def test_expected_outputs_exist() -> None:
    assert LOCK_PATH.is_file()
    assert CERT_PATH.is_file()
    assert Path(__file__).is_file()


def test_module_constants(zkp_cert) -> None:
    assert zkp_cert.INTERFACE == INTERFACE
    assert zkp_cert.LOCK_INTERFACE == LOCK_INTERFACE
    assert zkp_cert.SCHEMA_VERSION == SCHEMA_VERSION
    assert zkp_cert.LOCK_SCHEMA_VERSION == LOCK_SCHEMA_VERSION
    assert zkp_cert.GOAL_ID == GOAL_ID
    assert zkp_cert.TASK_ID == TASK_ID
    assert zkp_cert.LANE_ID == LANE_ID
    assert zkp_cert.TOOL_ID == TOOL_ID
    assert zkp_cert.HANDLER_ID == HANDLER_ID
    assert zkp_cert.AUTHORITY_CEILING == AUTHORITY_CEILING
    assert zkp_cert.AUTHORITY_SCOPE == AUTHORITY_SCOPE
    assert zkp_cert.CERTIFICATION_SURFACE == "tools.logic.certification.zkp"
    assert REQUIRED_CASE_KINDS <= set(zkp_cert.REQUIRED_CASE_KINDS)


def test_lock_schema_and_reviewable_bindings(lock: dict[str, Any], zkp_cert) -> None:
    assert lock["schema_version"] == LOCK_SCHEMA_VERSION
    assert lock["interface"] == LOCK_INTERFACE
    assert lock["goal_id"] == GOAL_ID
    assert lock["task_id"] == TASK_ID
    assert lock["tool_id"] == TOOL_ID
    assert lock["lane_id"] == LANE_ID
    assert lock["binding_mode"] == "reviewed_secret_safe_deployment"
    assert lock["replaces_gap_id"] == "circuit_witness"

    reasons = zkp_cert.validate_deployment_lock(lock)
    assert reasons == [], reasons

    bindings = zkp_cert.lock_public_bindings(lock)
    for field in (
        "circuit_id",
        "circuit_version",
        "circuit_public_digest",
        "ceremony_id",
        "ceremony_digest",
        "crs_id",
        "crs_digest",
        "proving_key_id",
        "proving_key_digest",
        "verification_key_id",
        "verification_key_digest",
        "backend_id",
        "backend_version",
        "backend_mode",
        "public_input_schema_digest",
        "revocation_policy_id",
        "verification_key_expires_at",
        "max_attestation_ttl_seconds",
    ):
        assert bindings.get(field), f"missing reviewable binding: {field}"

    for digest_field in (
        "circuit_public_digest",
        "ceremony_digest",
        "crs_digest",
        "proving_key_digest",
        "verification_key_digest",
        "public_input_schema_digest",
    ):
        assert _SHA256_CID.fullmatch(str(bindings[digest_field])), digest_field

    # Digest bases must recompute exactly.
    assert lock["circuit"]["circuit_public_digest"] == zkp_cert.identity_digest(
        lock["circuit"]["digest_basis"]
    )
    assert lock["ceremony"]["ceremony_digest"] == zkp_cert.identity_digest(
        lock["ceremony"]["digest_basis"]
    )
    assert lock["crs"]["crs_digest"] == zkp_cert.identity_digest(
        lock["crs"]["digest_basis"]
    )
    assert lock["keys"]["proving_key"]["proving_key_digest"] == zkp_cert.identity_digest(
        lock["keys"]["proving_key"]["digest_basis"]
    )
    assert lock["keys"]["verification_key"][
        "verification_key_digest"
    ] == zkp_cert.identity_digest(lock["keys"]["verification_key"]["digest_basis"])

    assert lock["backend"]["backend_mode"] == "cryptographic"
    assert lock["backend"]["simulated_backends_fail_closed"] is True
    assert lock["authority"]["ceiling"] == AUTHORITY_CEILING
    assert lock["authority"]["never_replaces_theorem_authority"] is True
    assert lock["freshness"]["stale_attestation_fails_closed"] is True
    assert lock["revocation"]["revoked_material_fails_closed"] is True
    assert lock["keys"]["proving_key"]["bytes_in_repository"] is False
    assert lock["keys"]["verification_key"]["bytes_in_repository"] is False
    assert "${ZKP_DEPLOYMENT_SECRET_ROOT}" in lock["keys"]["proving_key"][
        "secret_path_template"
    ]
    assert "${ZKP_DEPLOYMENT_SECRET_ROOT}" in lock["keys"]["verification_key"][
        "secret_path_template"
    ]

    required_keys = set(lock["public_input_schema"]["required_keys"])
    assert REQUIRED_PUBLIC_INPUT_KEYS <= required_keys


def test_lock_contains_no_private_secret_material(lock: dict[str, Any]) -> None:
    # Exclude secret_safety policy list (it names forbidden identifiers).
    payload = {key: value for key, value in lock.items() if key != "secret_safety"}
    encoded = json.dumps(payload, sort_keys=True, default=str)
    for needle in FORBIDDEN_SECRET_NEEDLES:
        assert needle not in encoded, needle
    # Policy may list forbidden field names; values must not carry them as data.
    assert "proving_key_bytes" not in encoded
    assert "witness_bytes" not in encoded
    assert lock["secret_safety"]["forbid_private_witness_in_lock"] is True
    assert lock["secret_safety"]["reference_private_artifacts_by_digest_only"] is True
    assert "toxic_waste" in lock["secret_safety"]["forbidden_lock_fields"]


def test_shared_toolchains_lock_points_at_deployment_binding() -> None:
    assert SHARED_LOCK_PATH.is_file()
    shared = json.loads(SHARED_LOCK_PATH.read_text(encoding="utf-8"))
    tools = shared.get("tools") or shared.get("toolchains") or []
    # Accept either list form or nested structure.
    entries: list[dict[str, Any]] = []
    if isinstance(tools, list):
        entries = [item for item in tools if isinstance(item, dict)]
    elif isinstance(tools, dict):
        entries = [item for item in tools.values() if isinstance(item, dict)]
    # Fallback: scan entire JSON for the zkp-circuit tool_id.
    if not any(item.get("tool_id") == TOOL_ID for item in entries):
        raw = SHARED_LOCK_PATH.read_text(encoding="utf-8")
        assert "config/formal_verification_zkp_deployment.lock.json" in raw
        assert "zkp-circuit" in raw
        return
    zkp_entries = [item for item in entries if item.get("tool_id") == TOOL_ID]
    assert zkp_entries, "shared lock must declare zkp-circuit"
    source = zkp_entries[0].get("source") or ""
    assert "formal_verification_zkp_deployment.lock.json" in source


# ---------------------------------------------------------------------------
# Full certification receipt
# ---------------------------------------------------------------------------


def test_production_certified_receipt(receipt: dict[str, Any]) -> None:
    assert receipt["schema_version"] == SCHEMA_VERSION
    assert receipt["interface"] == INTERFACE
    assert receipt["goal_id"] == GOAL_ID
    assert receipt["task_id"] == TASK_ID
    assert receipt["lane_id"] == LANE_ID
    assert receipt["tool_id"] == TOOL_ID
    assert receipt["authority_ceiling"] == AUTHORITY_CEILING
    assert receipt["authority_scope"] == AUTHORITY_SCOPE
    assert receipt["usable"] is True
    assert receipt["production_certified"] is True
    assert receipt["certified"] is True
    assert receipt["promotion_blocked"] is False
    assert receipt["block_reasons"] == []
    assert receipt["network_used"] is False
    assert receipt["install_attempted"] is False
    assert receipt["download_attempted"] is False
    assert receipt["policy"]["no_install"] is True
    assert receipt["policy"]["no_download"] is True
    assert receipt["policy"]["no_network"] is True
    assert receipt["policy"]["secret_safe"] is True
    assert receipt["policy"]["never_replaces_theorem_authority"] is True
    assert receipt["receipt_digest_sha256"]


def test_all_required_case_kinds_pass(receipt: dict[str, Any]) -> None:
    cases = receipt["cases"]
    assert cases
    kinds = {case["kind"] for case in cases}
    assert REQUIRED_CASE_KINDS <= kinds, sorted(REQUIRED_CASE_KINDS - kinds)

    by_id = {check["check_id"]: check for check in receipt["checks"]}
    required_check_ids = {
        "zkp.lock_reviewable_binding",
        "zkp.positive_verification",
        "zkp.corrupted_proof",
        "zkp.corrupted_key",
        "zkp.corrupted_public_input",
        "zkp.circuit_mismatch",
        "zkp.mutation",
        "zkp.deterministic_replay",
        "zkp.stale",
        "zkp.revoked",
        "zkp.secret_safety",
        "zkp.authority_boundary",
        "zkp.corpus_coverage",
        "zkp.public_input_schema",
        "zkp.offline_policy",
    }
    assert required_check_ids <= set(by_id), sorted(required_check_ids - set(by_id))
    for check_id in required_check_ids:
        assert by_id[check_id]["status"] == "passed", (
            check_id,
            by_id[check_id],
        )

    for case in cases:
        if case["kind"] in REQUIRED_CASE_KINDS:
            assert case["passed"] is True, case


def test_positive_verification_bindings(receipt: dict[str, Any]) -> None:
    by_id = {check["check_id"]: check for check in receipt["checks"]}
    positive = by_id["zkp.positive_verification"]
    assert positive["expected"] == "verified"
    assert positive["observed"] == "verified"
    bindings = positive["bindings"]
    assert bindings["proof_digest"].startswith("sha256:")
    assert bindings["public_input_digest"]
    assert bindings["circuit_id"] == receipt["bindings"]["circuit_id"]
    assert bindings["verification_key_id"] == receipt["bindings"]["verification_key_id"]


def test_fail_closed_negative_cases(receipt: dict[str, Any]) -> None:
    by_id = {check["check_id"]: check for check in receipt["checks"]}
    assert by_id["zkp.corrupted_proof"]["observed"] == "rejected"
    assert by_id["zkp.corrupted_key"]["observed"] == "rejected"
    assert by_id["zkp.corrupted_public_input"]["observed"] == "rejected"
    assert by_id["zkp.circuit_mismatch"]["status"] == "passed"
    assert by_id["zkp.stale"]["status"] == "passed"
    assert by_id["zkp.revoked"]["status"] == "passed"
    assert by_id["zkp.mutation"]["observed"] == "distinct"
    assert by_id["zkp.deterministic_replay"]["observed"] == "matched"


def test_secret_safety_case(receipt: dict[str, Any], zkp_cert) -> None:
    by_id = {check["check_id"]: check for check in receipt["checks"]}
    assert by_id["zkp.secret_safety"]["status"] == "passed"
    encoded = json.dumps(receipt, sort_keys=True, default=str)
    assert zkp_cert.PRIVATE_SECRET_MARKER not in encoded
    # Bindings and public receipt surfaces must not contain the private marker.
    assert zkp_cert.PRIVATE_SECRET_MARKER not in json.dumps(
        receipt.get("bindings") or {}, default=str
    )


def test_authority_never_replaces_theorem(receipt: dict[str, Any]) -> None:
    by_id = {check["check_id"]: check for check in receipt["checks"]}
    authority = by_id["zkp.authority_boundary"]
    assert authority["status"] == "passed"
    assert authority["observed"] == "theorem"
    assert authority["bindings"]["authority_ceiling"] == AUTHORITY_CEILING
    assert authority["bindings"]["never_replaces_theorem_authority"] is True
    assert receipt["authority_ceiling"] == AUTHORITY_CEILING
    assert receipt["authority_scope"] == AUTHORITY_SCOPE


def test_lane_handler_entry_point(zkp_cert, lock) -> None:
    result = zkp_cert.lane_handler(repo_root=REPO_ROOT, lock=lock)
    assert result["handler_id"] == HANDLER_ID
    assert result["lane_id"] == LANE_ID
    assert result["owner_module"] == "tools.logic.certification.zkp"
    assert result["certified"] is True
    assert result["status"] == "certified"
    assert result["production_certified"] is True


def test_backend_policy_from_lock_aligns_with_attestation_bridge(
    zkp_cert, lock
) -> None:
    policy = zkp_cert.backend_policy_from_lock(lock)
    assert policy.backend_id == lock["backend"]["backend_id"]
    assert policy.circuit_id == lock["circuit"]["circuit_id"]
    assert policy.circuit_version == lock["circuit"]["circuit_version"]
    assert policy.ceremony_id == lock["ceremony"]["ceremony_id"]
    assert policy.crs_id == lock["crs"]["crs_id"]
    assert policy.proving_key_id == lock["keys"]["proving_key"]["proving_key_id"]
    assert (
        policy.verification_key_id
        == lock["keys"]["verification_key"]["verification_key_id"]
    )
    assert policy.revocation_policy_id == lock["revocation"]["policy_id"]
    assert policy.backend_mode.value == "cryptographic"
    assert policy.simulated is False


def test_tampered_lock_fails_closed(zkp_cert, lock) -> None:
    bad = json.loads(json.dumps(lock))
    bad["circuit"]["circuit_public_digest"] = "sha256:" + ("0" * 64)
    reasons = zkp_cert.validate_deployment_lock(bad)
    assert "circuit_digest_basis_mismatch" in reasons

    receipt = zkp_cert.build_certification_receipt(
        repo_root=REPO_ROOT,
        lock=bad,
    )
    assert receipt["production_certified"] is False
    assert receipt["promotion_blocked"] is True
    assert receipt["certified"] is False
