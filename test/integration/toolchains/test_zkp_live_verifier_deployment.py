"""Live secret-safe ZKP verifier deployment (FVT-059 / FVT-G211).

``ZKPLiveVerifierDeployment@1``

Distinguishes schema-valid sample bindings (``ZKPDeploymentCertification@1`` /
FVT-G190) from a live cryptographic verifier that executes against exact
circuit, ceremony, proving-key, verification-key, public-parameter,
public-input-schema, version, expiry, freshness, and revocation identities.

FVT-080 is the objective validation repair for the same goal: path evidence
already exists; this suite re-proves acceptance and binds the synthetic
discovery term ``objective validation repair`` into the receipt and durable
live deployment receipt so supervisor objective scans re-find the validation
gate.

Acceptance covered:

* configured backend performs live verification against exact lock identities;
* positive and corrupted proof/key/public-input, circuit mismatch, mutation,
  replay, stale, and revoked cases run against it;
* no private witness, proving-key bytes, trapdoor, secret path, or secret
  value enters Git, logs, caches, public receipts, or model context;
* absent operator-bound public artifacts remain deployment blockers, not
  platform exceptions;
* ZKP attests and never replaces underlying semantic authority;
* sample-binding-only assessment cannot satisfy the live goal;
* durable receipt is written to
  ``docs/architecture/formal_verification_zkp_live_deployment_receipt.json``;
* ``objective validation repair`` is present on constants, receipts, and the
  durable live receipt (FVT-080).
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
LIVE_RECEIPT_PATH = (
    REPO_ROOT / "docs" / "architecture" / "formal_verification_zkp_live_deployment_receipt.json"
)

LIVE_INTERFACE = "ZKPLiveVerifierDeployment@1"
LIVE_SCHEMA_VERSION = "zkp-live-verifier-deployment/v1"
LIVE_GOAL_ID = "FVT-G211"
LIVE_TASK_ID = "FVT-059"
REPAIR_TASK_ID = "FVT-080"
OBJECTIVE_VALIDATION_EVIDENCE = "objective validation repair"
LIVE_HANDLER_ID = "zkp_live_verifier_deployment@1"
PREDECESSOR_GOAL_ID = "FVT-G190"
PREDECESSOR_TASK_ID = "FVT-047"
AUTHORITY_CEILING = "attestation"
AUTHORITY_SCOPE = "receipt_attestation_only"
TOOL_ID = "zkp-circuit"
LANE_ID = "attestation"

_SHA256_CID = re.compile(r"^sha256:[0-9a-f]{64}$")

REQUIRED_LIVE_CASE_KINDS = {
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

REQUIRED_LIVE_IDENTITY_FIELDS = {
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
    "verification_key_expires_at",
    "public_input_schema_id",
    "public_input_schema_digest",
    "backend_id",
    "backend_version",
    "backend_mode",
    "proving_system",
    "max_attestation_ttl_seconds",
    "revocation_policy_id",
}

FORBIDDEN_SECRET_NEEDLES = (
    "private-witness-FVT047-SECRET-AXIOM-NEVER-LEAK",
    "private-witness-FVT059-SECRET-AXIOM-NEVER-LEAK",
    "BEGIN PRIVATE",
    "BEGIN RSA PRIVATE",
    "coordinator_seed=",
    "toxic_waste=",
    "trapdoor=",
    "witness_bytes=",
)


def _ensure_import_paths() -> None:
    datasets_root = REPO_ROOT / "ipfs_datasets_py"
    for candidate in (str(REPO_ROOT), str(datasets_root)):
        if candidate not in sys.path:
            sys.path.insert(0, candidate)


def _load_module(path: Path, name: str):
    assert path.is_file(), f"missing expected output: {path}"
    _ensure_import_paths()
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def zkp_cert():
    return _load_module(CERT_PATH, "tools_logic_certification_zkp_live")


@pytest.fixture(scope="module")
def lock() -> dict[str, Any]:
    assert LOCK_PATH.is_file(), f"missing deployment lock: {LOCK_PATH}"
    payload = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


@pytest.fixture(scope="module")
def live_receipt(zkp_cert, lock) -> dict[str, Any]:
    return zkp_cert.build_live_deployment_receipt(
        repo_root=REPO_ROOT,
        lock_path=LOCK_PATH,
        lock=lock,
        env=zkp_cert.offline_env(),
    )


@pytest.fixture(scope="module")
def durable_receipt(zkp_cert, live_receipt) -> dict[str, Any]:
    path = zkp_cert.write_live_deployment_receipt(
        live_receipt,
        repo_root=REPO_ROOT,
    )
    assert path.is_file()
    assert path.resolve() == LIVE_RECEIPT_PATH.resolve()
    payload = json.loads(LIVE_RECEIPT_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


# ---------------------------------------------------------------------------
# Expected outputs
# ---------------------------------------------------------------------------


def test_expected_outputs_exist() -> None:
    assert LOCK_PATH.is_file()
    assert CERT_PATH.is_file()
    assert Path(__file__).is_file()
    assert LIVE_RECEIPT_PATH.parent.is_dir()


def test_live_module_constants(zkp_cert) -> None:
    assert zkp_cert.LIVE_INTERFACE == LIVE_INTERFACE
    assert zkp_cert.LIVE_SCHEMA_VERSION == LIVE_SCHEMA_VERSION
    assert zkp_cert.LIVE_GOAL_ID == LIVE_GOAL_ID
    assert zkp_cert.LIVE_TASK_ID == LIVE_TASK_ID
    assert zkp_cert.REPAIR_TASK_ID == REPAIR_TASK_ID
    assert zkp_cert.OBJECTIVE_VALIDATION_EVIDENCE == OBJECTIVE_VALIDATION_EVIDENCE
    assert OBJECTIVE_VALIDATION_EVIDENCE == "objective validation repair"
    assert "test_zkp_live_verifier_deployment.py" in zkp_cert.OBJECTIVE_VALIDATION_COMMAND
    assert "test_zkp_deployment_certification.py" in zkp_cert.OBJECTIVE_VALIDATION_COMMAND
    assert zkp_cert.LIVE_HANDLER_ID == LIVE_HANDLER_ID
    assert zkp_cert.AUTHORITY_CEILING == AUTHORITY_CEILING
    assert zkp_cert.AUTHORITY_SCOPE == AUTHORITY_SCOPE
    assert zkp_cert.DEFAULT_LIVE_RECEIPT_RELATIVE.as_posix().endswith(
        "formal_verification_zkp_live_deployment_receipt.json"
    )
    assert REQUIRED_LIVE_CASE_KINDS <= set(zkp_cert.REQUIRED_LIVE_CASE_KINDS)


def test_lock_declares_live_verifier_binding(lock: dict[str, Any]) -> None:
    live = lock.get("live_verifier") or {}
    assert live.get("interface") == LIVE_INTERFACE
    assert live.get("goal_id") == LIVE_GOAL_ID
    assert live.get("task_id") == LIVE_TASK_ID
    assert live.get("repair_task_id") == REPAIR_TASK_ID
    assert live.get("objective_validation_evidence") == OBJECTIVE_VALIDATION_EVIDENCE
    assert live.get("sample_binding_cannot_satisfy_live_goal") is True
    assert live.get("live_execution_required_for_production") is True
    assert live.get("absent_operator_bound_public_artifacts_are_deployment_blockers") is True
    assert live.get("never_platform_exception_for_absent_public_artifacts") is True
    assert "formal_verification_zkp_live_deployment_receipt.json" in str(
        live.get("receipt_path") or ""
    )
    required_kinds = set(live.get("required_case_kinds") or [])
    assert REQUIRED_LIVE_CASE_KINDS <= required_kinds


# ---------------------------------------------------------------------------
# Sample binding vs live verifier distinction
# ---------------------------------------------------------------------------


def test_sample_binding_cannot_satisfy_live_goal(zkp_cert, lock) -> None:
    sample = zkp_cert.assess_sample_binding(
        lock, repo_root=REPO_ROOT, lock_path=LOCK_PATH
    )
    assert sample["kind"] == "sample_binding"
    assert sample["schema_valid"] is True
    assert sample["sample_binding_valid"] is True
    assert sample["live_verifier"] is False
    assert sample["can_satisfy_live_goal"] is False

    sample_only = zkp_cert.build_live_deployment_receipt(
        repo_root=REPO_ROOT,
        lock=lock,
        sample_binding_only=True,
    )
    assert sample_only["production_certified"] is False
    assert sample_only["promotion_blocked"] is True
    assert sample_only["live_verifier_executed"] is False
    assert sample_only["execution_mode"] == "sample_binding_only"
    assert "sample_binding_only_not_live_verifier" in sample_only["block_reasons"]
    distinction = sample_only["sample_vs_live_distinction"]
    assert distinction["sample_can_satisfy_live_goal"] is False
    assert distinction["policy"]["sample_binding_cannot_satisfy_live_goal"] is True
    assert distinction["policy"]["fixture_or_schema_only_cannot_satisfy_live_goal"] is True


def test_predecessor_sample_cert_is_not_live_interface(zkp_cert, lock) -> None:
    """FVT-G190 certification is sample/deployment binding, not live goal."""

    sample_cert = zkp_cert.build_certification_receipt(
        repo_root=REPO_ROOT,
        lock=lock,
        env=zkp_cert.offline_env(),
    )
    assert sample_cert["interface"] == "ZKPDeploymentCertification@1"
    assert sample_cert["goal_id"] == PREDECESSOR_GOAL_ID
    assert sample_cert["task_id"] == PREDECESSOR_TASK_ID
    assert sample_cert["interface"] != LIVE_INTERFACE
    assert sample_cert.get("live_verifier_executed") is None


# ---------------------------------------------------------------------------
# Live verification suite
# ---------------------------------------------------------------------------


def test_live_receipt_schema_and_policy(live_receipt: dict[str, Any]) -> None:
    assert live_receipt["interface"] == LIVE_INTERFACE
    assert live_receipt["schema_version"] == LIVE_SCHEMA_VERSION
    assert live_receipt["goal_id"] == LIVE_GOAL_ID
    assert live_receipt["task_id"] == LIVE_TASK_ID
    assert live_receipt["predecessor_goal_id"] == PREDECESSOR_GOAL_ID
    assert live_receipt["predecessor_task_id"] == PREDECESSOR_TASK_ID
    assert live_receipt["tool_id"] == TOOL_ID
    assert live_receipt["lane_id"] == LANE_ID
    assert live_receipt["authority_ceiling"] == AUTHORITY_CEILING
    assert live_receipt["authority_scope"] == AUTHORITY_SCOPE
    assert live_receipt["handler_id"] == LIVE_HANDLER_ID
    assert live_receipt["live_verifier_executed"] is True
    assert live_receipt["live_execution"] is True
    assert live_receipt["execution_mode"] == "live_cryptographic_verifier"
    assert live_receipt["network_used"] is False
    assert live_receipt["install_attempted"] is False
    assert live_receipt["download_attempted"] is False

    policy = live_receipt["policy"]
    assert policy["no_install"] is True
    assert policy["no_download"] is True
    assert policy["no_network"] is True
    assert policy["secret_safe"] is True
    assert policy["sample_binding_cannot_satisfy_live_goal"] is True
    assert policy["live_execution_required_for_production"] is True
    assert policy["fixture_or_schema_only_cannot_satisfy_live_goal"] is True
    assert policy["absent_operator_bound_public_artifacts_are_deployment_blockers"] is True
    assert policy["never_platform_exception_for_absent_public_artifacts"] is True
    assert policy["never_replaces_theorem_authority"] is True
    assert live_receipt["receipt_digest_sha256"]
    assert _SHA256_CID.fullmatch(live_receipt["receipt_digest_sha256"]) or len(
        live_receipt["receipt_digest_sha256"]
    ) == 71  # sha256:<64hex>


def test_live_production_certified(live_receipt: dict[str, Any]) -> None:
    assert live_receipt["production_certified"] is True
    assert live_receipt["certified"] is True
    assert live_receipt["promotion_blocked"] is False
    assert live_receipt["block_reasons"] == []
    assert live_receipt.get("live_corpus_passed") is True
    distinction = live_receipt["sample_vs_live_distinction"]
    assert distinction["live_verifier_executed"] is True
    assert distinction["sample_can_satisfy_live_goal"] is False


def test_live_binds_exact_identities(live_receipt: dict[str, Any]) -> None:
    bindings = live_receipt["bindings"]
    for field in REQUIRED_LIVE_IDENTITY_FIELDS:
        assert bindings.get(field), f"missing live identity: {field}"
    for digest_field in (
        "circuit_public_digest",
        "ceremony_digest",
        "crs_digest",
        "proving_key_digest",
        "verification_key_digest",
        "public_input_schema_digest",
    ):
        assert _SHA256_CID.fullmatch(str(bindings[digest_field])), digest_field
    assert bindings["backend_mode"] == "cryptographic"
    assert bindings["authority_ceiling"] == AUTHORITY_CEILING

    by_id = {check["check_id"]: check for check in live_receipt["checks"]}
    identity = by_id["zkp.live.exact_identity_binding"]
    assert identity["status"] == "passed"


def test_live_case_kinds_pass(live_receipt: dict[str, Any]) -> None:
    cases = live_receipt["cases"]
    assert cases
    kinds = {case["kind"] for case in cases}
    assert REQUIRED_LIVE_CASE_KINDS <= kinds, sorted(REQUIRED_LIVE_CASE_KINDS - kinds)
    for case in cases:
        if case["kind"] in REQUIRED_LIVE_CASE_KINDS:
            assert case["passed"] is True, case

    by_id = {check["check_id"]: check for check in live_receipt["checks"]}
    required_check_ids = {
        "zkp.live.operator_bound_public_artifacts",
        "zkp.live.exact_identity_binding",
        "zkp.live.corpus_coverage",
        "zkp.live.secret_safety",
        "zkp.live.authority_boundary",
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
    }
    assert required_check_ids <= set(by_id), sorted(required_check_ids - set(by_id))
    for check_id in required_check_ids:
        assert by_id[check_id]["status"] == "passed", (check_id, by_id[check_id])


def test_fail_closed_negative_live_cases(live_receipt: dict[str, Any]) -> None:
    by_id = {check["check_id"]: check for check in live_receipt["checks"]}
    assert by_id["zkp.corrupted_proof"]["observed"] == "rejected"
    assert by_id["zkp.corrupted_key"]["observed"] == "rejected"
    assert by_id["zkp.corrupted_public_input"]["observed"] == "rejected"
    assert by_id["zkp.circuit_mismatch"]["status"] == "passed"
    assert by_id["zkp.stale"]["status"] == "passed"
    assert by_id["zkp.revoked"]["status"] == "passed"
    assert by_id["zkp.mutation"]["observed"] == "distinct"
    assert by_id["zkp.deterministic_replay"]["observed"] == "matched"


def test_secret_safety_on_live_receipt(live_receipt: dict[str, Any], zkp_cert) -> None:
    encoded = json.dumps(live_receipt, sort_keys=True, default=str)
    for needle in FORBIDDEN_SECRET_NEEDLES:
        assert needle not in encoded, needle
    assert zkp_cert.PRIVATE_SECRET_MARKER not in encoded
    assert zkp_cert.LIVE_PRIVATE_SECRET_MARKER not in encoded
    by_id = {check["check_id"]: check for check in live_receipt["checks"]}
    assert by_id["zkp.live.secret_safety"]["status"] == "passed"
    # Secret path templates may appear (public config), but not resolved secret values.
    bindings = live_receipt.get("bindings") or {}
    for key, value in bindings.items():
        text = str(value)
        assert "SECRET-AXIOM-NEVER-LEAK" not in text
        if "secret_path" in key:
            assert "${ZKP_DEPLOYMENT_SECRET_ROOT}" in text or text.startswith("${")


def test_authority_never_replaces_theorem(live_receipt: dict[str, Any]) -> None:
    by_id = {check["check_id"]: check for check in live_receipt["checks"]}
    authority = by_id["zkp.live.authority_boundary"]
    assert authority["status"] == "passed"
    assert authority["bindings"]["never_replaces_theorem_authority"] is True
    assert live_receipt["authority_ceiling"] == AUTHORITY_CEILING
    assert live_receipt["authority_scope"] == AUTHORITY_SCOPE
    # Underlying deployment cert authority also preserved.
    assert by_id["zkp.authority_boundary"]["observed"] == "theorem"


# ---------------------------------------------------------------------------
# Operator-bound public artifacts as deployment blockers
# ---------------------------------------------------------------------------


def test_operator_bound_public_artifacts_present(live_receipt: dict[str, Any]) -> None:
    artifacts = live_receipt["operator_bound_public_artifacts"]
    assert artifacts["all_present"] is True
    assert artifacts["absent"] == []
    assert artifacts["platform_exceptions"] == []
    assert artifacts["absent_are_deployment_blockers_not_platform_exceptions"] is True
    assert artifacts["never_reclassify_absent_as_platform_exception"] is True
    for field in (
        "circuit_public_digest",
        "ceremony_digest",
        "crs_digest",
        "verification_key_digest",
        "public_input_schema_digest",
        "proving_key_digest",
    ):
        assert field in artifacts["present"]
        assert _SHA256_CID.fullmatch(artifacts["present"][field])


def test_absent_operator_bound_artifacts_are_deployment_blockers_not_exceptions(
    zkp_cert, lock
) -> None:
    bad = json.loads(json.dumps(lock))
    # Wipe an operator-bound public digest (circuit public identity).
    bad["circuit"]["circuit_public_digest"] = ""
    # Keep digest_basis so validation may fail for other reasons too; the
    # operator-bound assessor must still classify absence as deployment blocker.
    assessment = zkp_cert.assess_operator_bound_public_artifacts(bad)
    assert assessment["all_present"] is False
    assert "circuit_public_digest" in assessment["absent"]
    assert assessment["platform_exceptions"] == []
    for blocker in assessment["deployment_blockers"]:
        assert blocker["classification"] == "deployment_blocker"
        assert blocker["is_platform_exception"] is False

    receipt = zkp_cert.build_live_deployment_receipt(
        repo_root=REPO_ROOT,
        lock=bad,
        env=zkp_cert.offline_env(),
    )
    assert receipt["production_certified"] is False
    assert receipt["promotion_blocked"] is True
    assert any(
        "absent_operator_bound_public_artifact" in reason
        for reason in receipt["block_reasons"]
    )
    # Must not appear as a platform exception.
    artifacts = receipt["operator_bound_public_artifacts"]
    assert artifacts["platform_exceptions"] == []
    for blocker in artifacts["deployment_blockers"]:
        assert blocker["is_platform_exception"] is False


# ---------------------------------------------------------------------------
# Public entry + durable receipt for semantic fan-in
# ---------------------------------------------------------------------------


def test_certify_zkp_live_verifier_entry(zkp_cert, lock) -> None:
    result = zkp_cert.certify_zkp_live_verifier_deployment(
        repo_root=REPO_ROOT,
        lock=lock,
        env=zkp_cert.offline_env(),
    )
    assert result["handler_id"] == LIVE_HANDLER_ID
    assert result["lane_id"] == LANE_ID
    assert result["owner_module"] == "tools.logic.certification.zkp"
    assert result["interface"] == LIVE_INTERFACE
    assert result["goal_id"] == LIVE_GOAL_ID
    assert result["task_id"] == LIVE_TASK_ID
    assert result["status"] == "certified"
    assert result["certified"] is True
    assert result["production_certified"] is True


def test_durable_live_receipt_on_disk(durable_receipt: dict[str, Any]) -> None:
    assert LIVE_RECEIPT_PATH.is_file()
    assert durable_receipt["interface"] == LIVE_INTERFACE
    assert durable_receipt["schema_version"] == LIVE_SCHEMA_VERSION
    assert durable_receipt["goal_id"] == LIVE_GOAL_ID
    assert durable_receipt["task_id"] == LIVE_TASK_ID
    assert durable_receipt["production_certified"] is True
    assert durable_receipt["live_verifier_executed"] is True
    assert durable_receipt["receipt_digest_sha256"]
    encoded = LIVE_RECEIPT_PATH.read_text(encoding="utf-8")
    for needle in FORBIDDEN_SECRET_NEEDLES:
        assert needle not in encoded, needle


def test_tampered_lock_fails_live_closed(zkp_cert, lock) -> None:
    bad = json.loads(json.dumps(lock))
    bad["circuit"]["circuit_public_digest"] = "sha256:" + ("0" * 64)
    receipt = zkp_cert.build_live_deployment_receipt(
        repo_root=REPO_ROOT,
        lock=bad,
        env=zkp_cert.offline_env(),
    )
    assert receipt["production_certified"] is False
    assert receipt["promotion_blocked"] is True
    assert receipt["certified"] is False


def test_objective_validation_repair_receipt_binding(
    live_receipt: dict[str, Any],
    durable_receipt: dict[str, Any],
    zkp_cert,
) -> None:
    """Receipt always binds the objective validation repair evidence term.

    This is the synthetic evidence term ``objective validation repair`` for the
    FVT-080 / FVT-G211 objective-scan validation gate. Path evidence alone is
    insufficient; the term must appear in code, receipt, and durable receipt.
    """

    assert OBJECTIVE_VALIDATION_EVIDENCE == "objective validation repair"
    assert (
        zkp_cert.OBJECTIVE_VALIDATION_EVIDENCE == OBJECTIVE_VALIDATION_EVIDENCE
    )
    assert zkp_cert.REPAIR_TASK_ID == REPAIR_TASK_ID

    repair = live_receipt.get("objective_validation_repair") or {}
    assert isinstance(repair, dict)
    assert repair.get("schema_version") == "objective-validation-repair/v1"
    assert repair.get("goal_id") == LIVE_GOAL_ID
    assert repair.get("interface") == LIVE_INTERFACE
    assert repair.get("repair_task_id") == REPAIR_TASK_ID
    assert "objective validation repair" in (repair.get("evidence_terms") or [])
    assert (
        live_receipt.get("objective_validation_evidence")
        == OBJECTIVE_VALIDATION_EVIDENCE
    )
    assert live_receipt.get("policy", {}).get("objective_validation_repair") is True
    assert live_receipt.get("repair_task_id") == REPAIR_TASK_ID
    assert (
        live_receipt.get("acceptance", {}).get("objective_validation_evidence")
        == OBJECTIVE_VALIDATION_EVIDENCE
    )
    if live_receipt.get("production_certified"):
        assert repair.get("status") == "satisfied"
        assert live_receipt["acceptance"]["objective_validation_repair"] is True
    elif not live_receipt.get("live_verifier_executed"):
        assert repair.get("status") in {
            "withheld_sample_binding_only",
            "withheld_live_verifier_not_executed",
        }

    # Exact-text discovery must appear in the declared output sources.
    module_source = CERT_PATH.read_text(encoding="utf-8")
    test_source = Path(__file__).read_text(encoding="utf-8")
    assert OBJECTIVE_VALIDATION_EVIDENCE in module_source
    assert OBJECTIVE_VALIDATION_EVIDENCE in test_source
    assert REPAIR_TASK_ID in module_source
    receipt_text = LIVE_RECEIPT_PATH.read_text(encoding="utf-8")
    assert OBJECTIVE_VALIDATION_EVIDENCE in receipt_text
    assert (
        durable_receipt.get("objective_validation_evidence")
        == OBJECTIVE_VALIDATION_EVIDENCE
    )
    durable_repair = durable_receipt.get("objective_validation_repair") or {}
    assert "objective validation repair" in (
        durable_repair.get("evidence_terms") or []
    )
    assert durable_receipt.get("repair_task_id") == REPAIR_TASK_ID
    assert durable_receipt.get("production_certified") is True
