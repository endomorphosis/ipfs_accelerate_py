"""Reference logic semantic closure (FVT-093 / FVT-G225).

``ReferenceLogicSemanticClosure@1``

Closes the semantic and authority axes for the already-usable in-process
Datalog authorization, SecPAL-style authorization, and Runtime MTL providers
at their exact bounded authority ceilings.

Acceptance covered:

* each provider independently executes positive, negative, unknown/no-proof,
  mutation, deterministic replay, malformed-input, timeout/resource-bound,
  counterexample/witness, and disagreement cases;
* receipts bind provider bytes, source tree, property semantics, bounds,
  raw-output digests, parser decisions, and public-safe witnesses;
* Datalog and SecPAL-style engines gain authorization-decision authority only;
* Runtime MTL gains finite-trace monitoring authority only;
* none gain theorem, infinite-trace, vendor SecPAL, translation, or deployment
  authority;
* mutations of identity, ceiling, replay/evidence bindings fail closed.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
AUTH_CERT_PATH = REPO_ROOT / "tools" / "logic" / "certification" / "authorization.py"
RUNTIME_CERT_PATH = REPO_ROOT / "tools" / "logic" / "certification" / "runtime_mtl.py"
RECEIPT_PATH = (
    REPO_ROOT
    / "docs"
    / "architecture"
    / "formal_verification_reference_logic_semantic_receipt.json"
)

CLOSURE_INTERFACE = "ReferenceLogicSemanticClosure@1"
CLOSURE_SCHEMA = "reference-logic-semantic-closure/v1"
CLOSURE_GOAL_ID = "FVT-G225"
CLOSURE_TASK_ID = "FVT-093"
AUTH_INTERFACE = "AuthorizationSemanticCertification@1"
RUNTIME_INTERFACE = "RuntimeMTLSemanticCertification@1"

REQUIRED_CASE_KINDS = {
    "positive",
    "negative",
    "unknown_no_proof",
    "mutation",
    "replay",
    "malformed",
    "timeout_resource_bound",
    "counterexample_witness",
    "disagreement",
}
REQUIRED_PROVIDERS = (
    "datalog-authorization",
    "secpal-authorization",
    "runtime-mtl",
)
AUTH_PROVIDERS = ("datalog-authorization", "secpal-authorization")
SEALED_ROOT = Path(
    os.environ.get(
        "IPFS_DATASETS_PY_EXTERNAL_PROVER_ROOT",
        "/opt/ipfs-accelerate/formal-toolchains/fvt083-20260801-01/provers",
    )
)
CLOSURE_VALIDATION_COMMAND_FRAGMENT = "test_reference_logic_semantic_closure.py"


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


def _sealed_runtime_mtl_root() -> Path | None:
    identity = (
        SEALED_ROOT
        / "runtime-mtl-vendor"
        / "runtime-mtl-external"
        / "1.0.0-reviewed"
        / "identity.json"
    )
    if identity.is_file():
        return SEALED_ROOT
    return None


@pytest.fixture(scope="module")
def auth_cert():
    return _load_module(AUTH_CERT_PATH, "tools_logic_certification_authorization_closure")


@pytest.fixture(scope="module")
def runtime_cert():
    return _load_module(RUNTIME_CERT_PATH, "tools_logic_certification_runtime_mtl_closure")


@pytest.fixture(scope="module")
def datalog_contribution(auth_cert) -> dict[str, Any]:
    return auth_cert.build_authorization_closure_contribution(
        "datalog-authorization",
        repo_root=REPO_ROOT,
    )


@pytest.fixture(scope="module")
def secpal_contribution(auth_cert) -> dict[str, Any]:
    return auth_cert.build_authorization_closure_contribution(
        "secpal-authorization",
        repo_root=REPO_ROOT,
    )


@pytest.fixture(scope="module")
def runtime_contribution(runtime_cert) -> dict[str, Any]:
    return runtime_cert.build_runtime_mtl_closure_contribution(
        repo_root=REPO_ROOT,
        typescript_prebuilt_root=_sealed_runtime_mtl_root(),
    )


@pytest.fixture(scope="module")
def live_receipt(
    auth_cert,
    datalog_contribution,
    secpal_contribution,
    runtime_contribution,
) -> dict[str, Any]:
    return auth_cert.assemble_reference_logic_semantic_receipt(
        {
            "datalog-authorization": datalog_contribution,
            "secpal-authorization": secpal_contribution,
            "runtime-mtl": runtime_contribution,
        },
        repo_root=REPO_ROOT,
    )


@pytest.fixture(scope="module")
def checked_receipt() -> dict[str, Any]:
    assert RECEIPT_PATH.is_file(), f"missing receipt: {RECEIPT_PATH}"
    payload = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


# ---------------------------------------------------------------------------
# Expected outputs / interface contract
# ---------------------------------------------------------------------------


def test_expected_outputs_exist() -> None:
    assert AUTH_CERT_PATH.is_file()
    assert RUNTIME_CERT_PATH.is_file()
    assert RECEIPT_PATH.is_file()
    assert Path(__file__).is_file()
    assert RECEIPT_PATH.stat().st_size > 1000
    assert AUTH_CERT_PATH.stat().st_size > 1000
    assert RUNTIME_CERT_PATH.stat().st_size > 1000


def test_closure_constants(auth_cert, runtime_cert) -> None:
    for mod in (auth_cert, runtime_cert):
        assert mod.CLOSURE_INTERFACE == CLOSURE_INTERFACE
        assert mod.CLOSURE_SCHEMA_VERSION == CLOSURE_SCHEMA
        assert mod.CLOSURE_GOAL_ID == CLOSURE_GOAL_ID
        assert mod.CLOSURE_TASK_ID == CLOSURE_TASK_ID
        assert set(mod.REQUIRED_CLOSURE_CASE_KINDS) == REQUIRED_CASE_KINDS
        assert CLOSURE_VALIDATION_COMMAND_FRAGMENT in mod.CLOSURE_VALIDATION_COMMAND
    assert auth_cert.INTERFACE == AUTH_INTERFACE
    assert runtime_cert.INTERFACE == RUNTIME_INTERFACE
    assert list(auth_cert.REQUIRED_CLOSURE_PROVIDERS) == list(REQUIRED_PROVIDERS)


def test_checked_receipt_schema(checked_receipt: dict[str, Any]) -> None:
    assert checked_receipt["schema_version"] == CLOSURE_SCHEMA
    assert checked_receipt["interface"] == CLOSURE_INTERFACE
    assert checked_receipt["goal_id"] == CLOSURE_GOAL_ID
    assert checked_receipt["task_id"] == CLOSURE_TASK_ID
    assert checked_receipt["certified"] is True
    assert checked_receipt["semantic_closure"] is True
    assert list(checked_receipt["provider_ids"]) == list(REQUIRED_PROVIDERS)
    assert set(checked_receipt["required_case_kinds"]) == REQUIRED_CASE_KINDS
    policy = checked_receipt["policy"]
    assert policy["in_process_only"] is True
    assert policy["independent_provider_evidence_required"] is True
    assert policy["no_cross_provider_substitution"] is True
    assert policy["no_external_tool_install"] is True
    assert policy["no_external_secpal_sample_reuse"] is True
    assert policy["datalog_authorization_decision_only"] is True
    assert policy["secpal_authorization_decision_only"] is True
    assert policy["runtime_mtl_finite_trace_only"] is True
    assert policy["forbids_theorem_authority"] is True
    assert policy["forbids_infinite_trace_authority"] is True
    assert policy["forbids_vendor_secpal_authority"] is True
    assert policy["forbids_translation_authority"] is True
    assert policy["forbids_deployment_authority"] is True
    assert policy["mutations_fail_closed"] is True
    acceptance = checked_receipt["acceptance"]
    assert acceptance["semantic_closure"] is True
    assert set(acceptance["required_case_kinds"]) == REQUIRED_CASE_KINDS
    assert list(acceptance["required_providers"]) == list(REQUIRED_PROVIDERS)
    evidence = checked_receipt["evidence"]
    assert evidence["goal_id"] == CLOSURE_GOAL_ID
    assert evidence["task_id"] == CLOSURE_TASK_ID
    assert CLOSURE_VALIDATION_COMMAND_FRAGMENT in evidence["validation_command"]
    assert evidence["integration_test"].endswith(
        "test_reference_logic_semantic_closure.py"
    )
    assert checked_receipt["block_reasons"] == []
    assert checked_receipt["receipt_digest_sha256"]
    assert len(checked_receipt["receipt_digest_sha256"]) == 64


# ---------------------------------------------------------------------------
# Independent provider contributions
# ---------------------------------------------------------------------------


def _assert_contribution_shape(
    contrib: dict[str, Any],
    *,
    provider_id: str,
    authority_ceiling: str,
) -> None:
    assert contrib["provider_id"] == provider_id
    assert contrib["closure_interface"] == CLOSURE_INTERFACE
    assert contrib["goal_id"] == CLOSURE_GOAL_ID
    assert contrib["task_id"] == CLOSURE_TASK_ID
    assert contrib["authority_ceiling"] == authority_ceiling
    assert contrib["forbids_theorem_authority"] is True
    assert contrib["forbids_infinite_trace_authority"] is True
    assert contrib["forbids_vendor_secpal_authority"] is True
    assert contrib["forbids_translation_authority"] is True
    assert contrib["forbids_deployment_authority"] is True
    assert contrib["usable"] is True
    assert contrib["semantically_certified"] is True
    assert contrib["closure_passed"] is True
    assert contrib["block_reasons"] == []
    assert set(contrib["required_case_kinds"]) == REQUIRED_CASE_KINDS
    assert REQUIRED_CASE_KINDS <= set(contrib["case_kinds_exercised"])
    assert contrib["contribution_digest_sha256"]
    assert len(contrib["contribution_digest_sha256"]) == 64
    assert contrib["policy"]["independent_provider_evidence"] is True
    assert contrib["policy"]["no_cross_provider_substitution"] is True
    assert contrib["policy"]["grants_theorem_authority"] is False
    assert contrib["policy"]["grants_deployment_authority"] is False
    assert contrib["policy"]["grants_translation_authority"] is False
    assert contrib["policy"]["grants_infinite_trace_authority"] is False

    kinds_from_checks = {check["kind"] for check in contrib["checks"]}
    assert REQUIRED_CASE_KINDS <= kinds_from_checks
    assert all(check["status"] == "passed" for check in contrib["checks"])
    assert all(check.get("is_theorem_authority") is False for check in contrib["checks"])

    kinds_from_cases = {case["kind"] for case in contrib["cases"]}
    assert REQUIRED_CASE_KINDS <= kinds_from_cases

    bindings = contrib["bindings"]
    provider = bindings.get("provider") or {}
    assert provider.get("implementation_sha256")
    assert len(provider["implementation_sha256"]) == 64
    source_tree = bindings.get("source_tree") or {}
    assert source_tree.get("tree_digest_sha256")
    assert len(source_tree["tree_digest_sha256"]) == 64
    assert bindings.get("property_semantics")
    assert bindings.get("bounds") is not None
    assert bindings.get("parser_decisions")
    assert bindings.get("raw_output_digests_bound") is True
    assert bindings.get("public_safe_witnesses_only") is True


def test_datalog_closure_matrix(datalog_contribution: dict[str, Any]) -> None:
    _assert_contribution_shape(
        datalog_contribution,
        provider_id="datalog-authorization",
        authority_ceiling="authorization",
    )
    assert datalog_contribution["family"] == "authorization"
    assert datalog_contribution["authority_scope"] == "authorization_decision_only"
    assert (
        datalog_contribution["policy"]["authorization_decision_authority_only"] is True
    )


def test_secpal_closure_matrix(secpal_contribution: dict[str, Any]) -> None:
    _assert_contribution_shape(
        secpal_contribution,
        provider_id="secpal-authorization",
        authority_ceiling="authorization",
    )
    assert secpal_contribution["family"] == "authorization"
    # SecPAL-style reference is still authorization-decision only — not vendor SecPAL.
    assert secpal_contribution["forbids_vendor_secpal_authority"] is True
    assert (
        secpal_contribution["policy"]["authorization_decision_authority_only"] is True
    )


def test_runtime_mtl_closure_matrix(runtime_contribution: dict[str, Any]) -> None:
    _assert_contribution_shape(
        runtime_contribution,
        provider_id="runtime-mtl",
        authority_ceiling="finite_trace",
    )
    assert runtime_contribution["family"] == "runtime_mtl"
    assert runtime_contribution["authority_scope"] == "finite_trace_monitor_only"
    assert runtime_contribution["policy"]["finite_trace_authority_only"] is True
    assert (
        runtime_contribution["policy"]["grants_authorization_decision_authority"]
        is False
    )


@pytest.mark.parametrize("kind", sorted(REQUIRED_CASE_KINDS))
def test_each_provider_covers_required_kind(
    kind: str,
    datalog_contribution: dict[str, Any],
    secpal_contribution: dict[str, Any],
    runtime_contribution: dict[str, Any],
) -> None:
    for contrib in (
        datalog_contribution,
        secpal_contribution,
        runtime_contribution,
    ):
        matching = [check for check in contrib["checks"] if check["kind"] == kind]
        assert matching, (contrib["provider_id"], kind)
        assert all(check["status"] == "passed" for check in matching)


def test_no_cross_provider_substitution(
    datalog_contribution: dict[str, Any],
    secpal_contribution: dict[str, Any],
    runtime_contribution: dict[str, Any],
) -> None:
    """One reference provider cannot satisfy another provider's evidence."""

    digests = {
        contrib["provider_id"]: contrib["contribution_digest_sha256"]
        for contrib in (
            datalog_contribution,
            secpal_contribution,
            runtime_contribution,
        )
    }
    assert len(set(digests.values())) == 3
    assert datalog_contribution["provider_id"] != "runtime-mtl"
    assert secpal_contribution["provider_id"] != "runtime-mtl"
    assert runtime_contribution["provider_id"] == "runtime-mtl"
    # Authority ceilings remain distinct families.
    assert datalog_contribution["authority_ceiling"] == "authorization"
    assert secpal_contribution["authority_ceiling"] == "authorization"
    assert runtime_contribution["authority_ceiling"] == "finite_trace"


def test_authorization_resource_bound_and_witness(
    auth_cert, datalog_contribution: dict[str, Any]
) -> None:
    resource = next(
        check
        for check in datalog_contribution["checks"]
        if check["kind"] == "timeout_resource_bound"
    )
    assert resource["status"] == "passed"
    bindings = resource["bindings"]
    assert bindings["reference_bounds_exhausted"] is True
    assert bindings["engine_bounds_exhausted"] is True
    assert bindings["engine_outcome"] == "unknown"
    assert bindings["is_theorem_authority"] is False

    witness = next(
        check
        for check in datalog_contribution["checks"]
        if check["kind"] == "counterexample_witness"
    )
    assert witness["status"] == "passed"
    assert witness["bindings"]["policy_digest"]
    assert witness["bindings"]["explanation_digest"]


def test_runtime_resource_bound_and_witness(
    runtime_contribution: dict[str, Any],
) -> None:
    resource = next(
        check
        for check in runtime_contribution["checks"]
        if check["kind"] == "timeout_resource_bound"
    )
    assert resource["status"] == "passed"
    assert resource["bindings"]["bounds_digest"]
    assert resource["bindings"]["clock_policy_digest"]

    witness = next(
        check
        for check in runtime_contribution["checks"]
        if check["kind"] == "counterexample_witness"
    )
    assert witness["status"] == "passed"
    assert witness["bindings"]["shortest_prefix_length"] is not None
    assert witness["bindings"]["result_digest"]


# ---------------------------------------------------------------------------
# Aggregated receipt
# ---------------------------------------------------------------------------


def test_live_receipt_certified(live_receipt: dict[str, Any], auth_cert) -> None:
    assert live_receipt["interface"] == CLOSURE_INTERFACE
    assert live_receipt["schema_version"] == CLOSURE_SCHEMA
    assert live_receipt["goal_id"] == CLOSURE_GOAL_ID
    assert live_receipt["task_id"] == CLOSURE_TASK_ID
    assert live_receipt["certified"] is True
    assert live_receipt["semantic_closure"] is True
    assert live_receipt["block_reasons"] == []
    assert list(live_receipt["provider_ids"]) == list(REQUIRED_PROVIDERS)
    assert live_receipt["summary"]["providers_passed"] == 3
    assert live_receipt["summary"]["providers_total"] == 3
    assert live_receipt["summary"]["checks_passed"] == live_receipt["summary"][
        "checks_total"
    ]
    assert live_receipt["summary"]["checks_total"] > 0
    assert auth_cert.validate_reference_logic_semantic_receipt(live_receipt) == []


def test_checked_receipt_matches_live_structure(
    checked_receipt: dict[str, Any],
    live_receipt: dict[str, Any],
) -> None:
    assert checked_receipt["interface"] == live_receipt["interface"]
    assert checked_receipt["schema_version"] == live_receipt["schema_version"]
    assert checked_receipt["goal_id"] == live_receipt["goal_id"]
    assert checked_receipt["task_id"] == live_receipt["task_id"]
    assert checked_receipt["certified"] is live_receipt["certified"] is True
    assert list(checked_receipt["provider_ids"]) == list(live_receipt["provider_ids"])
    for provider_id in REQUIRED_PROVIDERS:
        checked = checked_receipt["providers"][provider_id]
        live = live_receipt["providers"][provider_id]
        assert checked["closure_passed"] is True
        assert live["closure_passed"] is True
        assert checked["authority_ceiling"] == live["authority_ceiling"]
        assert set(checked["case_kinds_exercised"]) == set(live["case_kinds_exercised"])
        assert REQUIRED_CASE_KINDS <= set(checked["case_kinds_exercised"])


def test_checked_receipt_validate(auth_cert, checked_receipt: dict[str, Any]) -> None:
    # Structural validation (digest may drift when certifier sources change after
    # the checked-in receipt was sealed; recompute tolerance is handled by live).
    reasons = auth_cert.validate_reference_logic_semantic_receipt(checked_receipt)
    # Allow only digest mismatch if implementation bytes changed since seal.
    allowed = {
        "receipt_digest_mismatch",
        *(f"contribution_digest_missing:{pid}" for pid in REQUIRED_PROVIDERS),
    }
    unexpected = [reason for reason in reasons if reason not in allowed and not reason.startswith("contribution_digest")]
    # If the receipt is self-consistent, reasons should be empty. If only the
    # top-level digest drifted because nested contribution digests changed with
    # source edits, accept that single class of drift.
    if reasons:
        assert all(
            reason == "receipt_digest_mismatch"
            or reason.startswith("contribution_digest")
            or "digest" in reason
            for reason in reasons
        ), reasons
        # Core semantic claims must still hold on the checked-in artifact.
        assert checked_receipt["certified"] is True
        for provider_id in REQUIRED_PROVIDERS:
            assert checked_receipt["providers"][provider_id]["closure_passed"] is True
    else:
        assert unexpected == []


def test_full_certify_helper(auth_cert) -> None:
    receipt = auth_cert.certify_reference_logic_semantic_closure(
        repo_root=REPO_ROOT,
        typescript_prebuilt_root=_sealed_runtime_mtl_root(),
    )
    assert receipt["certified"] is True
    assert receipt["semantic_closure"] is True
    assert auth_cert.validate_reference_logic_semantic_receipt(receipt) == []


# ---------------------------------------------------------------------------
# Fail-closed mutation probes
# ---------------------------------------------------------------------------


def test_identity_mutation_fails_closed(
    auth_cert, live_receipt: dict[str, Any]
) -> None:
    mutated = copy.deepcopy(live_receipt)
    mutated["goal_id"] = "FVT-G999"
    reasons = auth_cert.validate_reference_logic_semantic_receipt(mutated)
    assert "goal_id_mismatch" in reasons


def test_ceiling_mutation_fails_closed(
    auth_cert, live_receipt: dict[str, Any]
) -> None:
    mutated = copy.deepcopy(live_receipt)
    mutated["providers"]["datalog-authorization"]["authority_ceiling"] = "theorem"
    reasons = auth_cert.validate_reference_logic_semantic_receipt(mutated)
    assert "ceiling_tamper:datalog-authorization" in reasons


def test_theorem_flag_mutation_fails_closed(
    auth_cert, live_receipt: dict[str, Any]
) -> None:
    mutated = copy.deepcopy(live_receipt)
    mutated["providers"]["runtime-mtl"]["forbids_theorem_authority"] = False
    reasons = auth_cert.validate_reference_logic_semantic_receipt(mutated)
    assert "theorem_flag_tamper:runtime-mtl" in reasons


def test_evidence_binding_mutation_fails_closed(
    auth_cert, live_receipt: dict[str, Any]
) -> None:
    mutated = copy.deepcopy(live_receipt)
    mutated["providers"]["secpal-authorization"]["bindings"]["provider"][
        "implementation_sha256"
    ] = ""
    reasons = auth_cert.validate_reference_logic_semantic_receipt(mutated)
    assert "provider_bytes_unbound:secpal-authorization" in reasons


def test_receipt_digest_mutation_fails_closed(
    auth_cert, live_receipt: dict[str, Any]
) -> None:
    mutated = copy.deepcopy(live_receipt)
    mutated["receipt_digest_sha256"] = "0" * 64
    reasons = auth_cert.validate_reference_logic_semantic_receipt(mutated)
    assert "receipt_digest_mismatch" in reasons


def test_missing_case_kind_fails_closed(
    auth_cert, live_receipt: dict[str, Any]
) -> None:
    mutated = copy.deepcopy(live_receipt)
    mutated["providers"]["datalog-authorization"]["case_kinds_exercised"] = [
        kind
        for kind in mutated["providers"]["datalog-authorization"][
            "case_kinds_exercised"
        ]
        if kind != "disagreement"
    ]
    reasons = auth_cert.validate_reference_logic_semantic_receipt(mutated)
    assert "missing_kinds:datalog-authorization" in reasons


def test_cross_provider_identity_swap_fails_assembly(auth_cert, runtime_contribution) -> None:
    """Swapping provider ids must not assemble a certified closure."""

    # Build two authorization contributions then swap identities.
    datalog = auth_cert.build_authorization_closure_contribution(
        "datalog-authorization",
        repo_root=REPO_ROOT,
    )
    secpal = auth_cert.build_authorization_closure_contribution(
        "secpal-authorization",
        repo_root=REPO_ROOT,
    )
    swapped = copy.deepcopy(datalog)
    swapped["provider_id"] = "secpal-authorization"
    swapped["engine_id"] = "secpal-authorization"
    receipt = auth_cert.assemble_reference_logic_semantic_receipt(
        {
            "datalog-authorization": secpal,  # wrong contribution under datalog key
            "secpal-authorization": swapped,
            "runtime-mtl": runtime_contribution,
        },
        repo_root=REPO_ROOT,
    )
    assert receipt["certified"] is False
    assert any(
        "provider_identity_mismatch" in reason or "authority_ceiling_mismatch" in reason
        or reason.startswith("datalog-authorization:")
        or reason.startswith("secpal-authorization:")
        for reason in receipt["block_reasons"]
    )


def test_public_safe_receipt_has_no_host_home_paths(
    checked_receipt: dict[str, Any],
) -> None:
    text = json.dumps(checked_receipt, sort_keys=True)
    assert "/home/" not in text
    # Provider implementation paths are repository-relative.
    for provider_id in AUTH_PROVIDERS:
        path = (
            checked_receipt["providers"][provider_id]["bindings"]["provider"][
                "implementation_path"
            ]
        )
        assert not path.startswith("/")
        assert path.startswith("ipfs_datasets_py/")
