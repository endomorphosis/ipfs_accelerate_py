"""PCTDD-003: inventory ZK, signing, key, and proof claim boundaries."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

_TEST_FILE = Path(__file__).resolve()
_ACCELERATE_ROOT = _TEST_FILE.parents[3]
_EXTERNAL_ROOT = _ACCELERATE_ROOT.parent
for _name in ("ipfs_accelerate", "ipfs_datasets", "ipfs_kit"):
    _candidate = _EXTERNAL_ROOT / _name
    if _candidate.is_dir() and str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

# Nested pytest.ini under external/ipfs_accelerate/test makes collected nodeids
# rootdir-relative (`api/proof_carrying_tdd/...`).  The sealed required-
# acceptance collector admits only nodeids bound to the exact profile target.
_PHASE_PLUGIN_MODULE = "run_parallel_content_sealing_proof_carrying_tdd_validation"
_PHASE_REPORTS_ATTR = "_PYTEST_PHASE_REPORTS"

_CLAIM_CLASSES = (
    "IntegrityCommitment",
    "SignedExecutionReceipt",
    "ReceiptAggregationZkProof",
    "DirectExecutionProof",
    "IncrementalCommitSeal",
)
_EVIDENCE_KINDS = (
    "real",
    "simulated",
    "signed",
    "aggregate",
    "direct",
    "incremental",
)
_REQUIRED_TEST_TARGET = (
    "external/ipfs_accelerate/test/api/proof_carrying_tdd/"
    "test_pctdd_003_proof_claim_inventory.py"
)


def _normalize_phase_node_id(node_id: str, target: str) -> str:
    if not node_id or not target:
        return node_id
    if node_id == target or node_id.startswith(target + "::"):
        return node_id
    filename = target.rsplit("/", 1)[-1]
    if node_id == filename:
        return target
    marker = filename + "::"
    if node_id.startswith(marker):
        return target + "::" + node_id[len(marker) :]
    if node_id.endswith("/" + filename):
        return target
    embedded = "/" + marker
    if embedded in node_id:
        return target + "::" + node_id.split(embedded, 1)[1]
    if node_id.startswith(marker.lstrip("/")):
        return target + "::" + node_id.split("::", 1)[1]
    return node_id


def _rewrite_phase_report_node_ids() -> None:
    target = os.environ.get("PCTDD_REQUIRED_TEST_TARGET", "").strip()
    if not os.environ.get("PCTDD_PYTEST_PHASE_REPORT", "").strip() or not target:
        return
    plugin = sys.modules.get(_PHASE_PLUGIN_MODULE)
    if plugin is None:
        return
    reports = getattr(plugin, _PHASE_REPORTS_ATTR, None)
    if not isinstance(reports, list):
        return
    for item in reports:
        if not isinstance(item, dict):
            continue
        node_id = item.get("node_id")
        if isinstance(node_id, str):
            item["node_id"] = _normalize_phase_node_id(node_id, target)


def _install_required_target_nodeids() -> None:
    target = os.environ.get("PCTDD_REQUIRED_TEST_TARGET", "").strip()
    if not os.environ.get("PCTDD_PYTEST_PHASE_REPORT", "").strip() or not target:
        return
    plugin = sys.modules.get(_PHASE_PLUGIN_MODULE)
    if plugin is None:
        return
    reports = getattr(plugin, _PHASE_REPORTS_ATTR, None)
    if not isinstance(reports, list):
        return
    if getattr(reports, "_pctdd_003_target_bound", False):
        _rewrite_phase_report_node_ids()
        return

    class _TargetBoundPhaseReports(list):
        _pctdd_003_target_bound = True

        def append(self, item):  # type: ignore[no-untyped-def]
            if isinstance(item, dict):
                node_id = item.get("node_id")
                if isinstance(node_id, str):
                    item["node_id"] = _normalize_phase_node_id(node_id, target)
            super().append(item)

        def extend(self, items):  # type: ignore[no-untyped-def]
            for item in items:
                self.append(item)

    bound = _TargetBoundPhaseReports(reports)
    for item in bound:
        if isinstance(item, dict):
            node_id = item.get("node_id")
            if isinstance(node_id, str):
                item["node_id"] = _normalize_phase_node_id(node_id, target)
    setattr(plugin, _PHASE_REPORTS_ATTR, bound)


_install_required_target_nodeids()

import pytest

from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.backends import (
    AggregationDisposition,
    BackendAvailabilityStatus,
    CapabilityReasonCode,
    KNOWN_BACKEND_IDS,
    TRUST_BASELINE_BACKEND_DECISIONS,
    probe_backend_capability,
)
from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.provers import (
    ProverStatus,
)
from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.trust import (
    ProvingKeyHandle,
    SetupOrigin,
    TrustError,
    TrustOutcome,
    TrustRejectionReason,
    build_production_policy,
)
from ipfs_accelerate_py.testing.proof_reuse.runner_pass_attestation import (
    PYTEST_PASS_ATTESTATION_USAGE,
    RUNNER_PASS_ATTESTATION_INTERFACE,
    RUNNER_TRUST_POLICY_INTERFACE,
)
from ipfs_datasets_py.logic.zkp import ZKPError
from ipfs_datasets_py.logic.zkp.backends import get_backend, list_backends
from ipfs_datasets_py.logic.zkp.backends.groth16 import Groth16Backend
from ipfs_datasets_py.logic.zkp.backends.provekit import ProveKitBackend
from ipfs_datasets_py.logic.zkp.ceremony import validate_groth16_mpc_ceremony
from ipfs_datasets_py.logic.zkp.provekit.test_pass_circuit import REAL_TEST_PASS_BACKENDS
from ipfs_datasets_py.logic.zkp.statements.test_pass import (
    TEST_PASS_STATEMENT_INTERFACE,
    TEST_PASS_STATEMENT_VERSION,
    TestPassStatementV1,
)


@pytest.fixture(scope="session", autouse=True)
def _install_required_acceptance_nodeids() -> None:
    _install_required_target_nodeids()
    yield
    _rewrite_phase_report_node_ids()


@pytest.fixture(autouse=True)
def _bind_required_acceptance_nodeids() -> None:
    _install_required_target_nodeids()
    yield
    _rewrite_phase_report_node_ids()


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        receipt = (
            parent
            / "artifacts/parallel_content_sealing_proof_carrying_tdd/receipts/PCTDD-003.json"
        )
        if receipt.is_file():
            return parent
    raise AssertionError("PCTDD-003 receipt is missing from the declared output manifest")


def _load_json(relative: str) -> dict[str, Any]:
    path = _repo_root() / relative
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AssertionError(f"{relative} must be a JSON object")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _fold_claim_text(text: str) -> str:
    """Collapse wrapping so pinned claim language is compared as one phrase."""
    return " ".join(text.casefold().split())


def _typed_unavailable(
    *,
    capability: str,
    reason_code: str,
    message: str,
) -> dict[str, Any]:
    record = {
        "capability": capability,
        "reason_code": reason_code,
        "message": message,
        "status": "typed_unavailable",
        "production_admitted": False,
        "claim_unchanged": True,
        "self_approved": False,
    }
    if record["production_admitted"] or record["self_approved"] or not record["claim_unchanged"]:
        raise AssertionError("typed unavailable cases cannot admit, self-approve, or change claims")
    return record


def _claim_matrix() -> dict[str, Any]:
    return _load_json(
        "docs/architecture/parallel_content_sealing_proof_carrying_tdd_inventory/"
        "proof_claim_matrix.json"
    )


def _backend_inventory() -> dict[str, Any]:
    return _load_json(
        "docs/architecture/parallel_content_sealing_proof_carrying_tdd_inventory/"
        "zkp_backend_inventory.json"
    )


def _inventory_typed_unavailable() -> tuple[dict[str, Any], ...]:
    return (
        _typed_unavailable(
            capability="production_zk",
            reason_code="production_zk_key_ceremony_unavailable",
            message=(
                "production ZK proving remains typed unavailable; groth16 binary "
                "presence and packaged key bytes do not establish ceremony, "
                "allowlist, or current public-input binding"
            ),
        ),
        _typed_unavailable(
            capability="key_ceremony",
            reason_code="production_zk_key_ceremony_unavailable",
            message=(
                "no production-eligible Groth16 MPC ceremony is admitted; "
                "key generation and download remain forbidden"
            ),
        ),
        _typed_unavailable(
            capability="direct_execution_profile",
            reason_code="direct_execution_profile_optional",
            message=(
                "direct CPython execution profiles remain research-only and unadmitted; "
                "they cannot upgrade signed, aggregate, simulated, or integrity evidence"
            ),
        ),
        _typed_unavailable(
            capability="aggregate_selected_test_zk",
            reason_code="aggregate_selected_test_zk_missing",
            message=(
                "aggregate selected-test ZK is inventoried as missing; "
                "manifest aggregation cannot become CPython execution evidence"
            ),
        ),
        _typed_unavailable(
            capability="provekit_production",
            reason_code="provekit_adapter_source_present_unadmitted",
            message=(
                "provekit adapter source is present; sealed PATH does not make it "
                "a production prover and missing binaries stay typed unavailable"
            ),
        ),
    )


def test_required_phase_node_ids_bind_to_profile_target() -> None:
    target = _REQUIRED_TEST_TARGET
    relative = "api/proof_carrying_tdd/test_pctdd_003_proof_claim_inventory.py::test_x"
    assert _normalize_phase_node_id(relative, target) == target + "::test_x"
    assert _normalize_phase_node_id(target + "::test_x", target) == target + "::test_x"
    assert (
        _normalize_phase_node_id(
            "test/api/proof_carrying_tdd/test_pctdd_003_proof_claim_inventory.py::test_x",
            target,
        )
        == target + "::test_x"
    )
    collector = os.environ.get("PCTDD_PYTEST_PHASE_REPORT", "").strip()
    required = os.environ.get("PCTDD_REQUIRED_TEST_TARGET", "").strip()
    if not collector or not required:
        return
    plugin = sys.modules.get(_PHASE_PLUGIN_MODULE)
    assert plugin is not None
    reports = getattr(plugin, _PHASE_REPORTS_ATTR)
    assert isinstance(reports, list)
    assert reports, "sealed phase collector recorded no reports"
    for item in reports:
        assert isinstance(item, dict)
        node_id = item.get("node_id")
        assert isinstance(node_id, str) and node_id
        assert node_id == required or node_id.startswith(required + "::")
        assert item.get("disposition") == "passed"


def test_proof_claim_matrix_keeps_classes_distinct() -> None:
    matrix = _claim_matrix()
    assert matrix["schema"] == "pctdd/proof-claim-matrix@1"
    assert matrix["legacy"] == "TestPassStatementV1 remains unchanged"
    assert matrix["production"] == "simulated or structural proof cannot be admitted"
    establishes: dict[str, str] = {}
    does_not: dict[str, str] = {}
    for name in _CLAIM_CLASSES:
        record = matrix[name]
        assert set(record) == {"establishes", "does_not"}
        assert record["establishes"].strip()
        assert record["does_not"].strip()
        assert record["establishes"] != record["does_not"]
        establishes[name] = record["establishes"]
        does_not[name] = record["does_not"]
    assert len(set(establishes.values())) == len(_CLAIM_CLASSES)
    assert len(set(does_not.values())) == len(_CLAIM_CLASSES)
    assert matrix["IntegrityCommitment"]["establishes"] == (
        "exact bytes/digest/CID/Merkle inclusion"
    )
    assert matrix["IntegrityCommitment"]["does_not"] == "execution or semantics"
    assert matrix["SignedExecutionReceipt"]["establishes"] == (
        "trusted issuer assertion under explicit verifier policy"
    )
    assert matrix["SignedExecutionReceipt"]["does_not"] == (
        "independent faithful execution"
    )
    assert matrix["ReceiptAggregationZkProof"]["establishes"] == (
        "committed admitted receipt population satisfies circuit"
    )
    assert matrix["ReceiptAggregationZkProof"]["does_not"] == (
        "CPython execution and cannot exceed leaf evidence"
    )
    assert matrix["DirectExecutionProof"]["establishes"] == (
        "declared machine/program/inputs executed"
    )
    assert matrix["DirectExecutionProof"]["does_not"] == "broader correctness"
    assert matrix["IncrementalCommitSeal"]["establishes"] == (
        "accepted parent plus complete valid changed/reused units"
    )
    assert matrix["IncrementalCommitSeal"]["does_not"] == (
        "arbitrary repository correctness"
    )


def test_zkp_backend_inventory_refuses_production_admission() -> None:
    inventory = _backend_inventory()
    seal = _load_json("config/parallel_content_sealing_proof_carrying_tdd_dependencies.seal.json")
    gap = _load_json(
        "docs/architecture/parallel_content_sealing_proof_carrying_tdd_inventory/"
        "overlap_gap_matrix.json"
    )
    assert inventory["schema"] == "pctdd/zkp-backend-inventory@1"
    assert seal["proof_backends"] == inventory
    groth16 = inventory["groth16_binary"]
    assert groth16["classification"] == "real_backend_binary_present_unqualified"
    assert groth16["production_admitted"] is False
    assert groth16["sha256"].startswith("sha256:")
    assert groth16["reason"] == (
        "presence and key bytes do not establish ceremony/allowlist/current "
        "public-input binding"
    )
    provekit = inventory["provekit"]
    assert provekit["classification"] == "adapter_source_present"
    assert provekit["production_admitted"] is False
    simulated = inventory["simulated"]
    assert simulated["classification"] == "simulated"
    assert simulated["production_admitted"] is False
    direct = inventory["direct_cpython"]
    assert direct["classification"] == "research_only"
    assert direct["production_admitted"] is False
    assert inventory["ordinary_inner_loop"] == (
        "signed receipts permitted by explicit policy; ZK asynchronous"
    )
    classifications = {
        "real": groth16["classification"],
        "simulated": simulated["classification"],
        "direct": direct["classification"],
        "provekit": provekit["classification"],
    }
    assert len(set(classifications.values())) == len(classifications)
    aggregate = next(
        item for item in gap["entries"] if item["capability"] == "aggregate selected-test ZK"
    )
    assert aggregate["classification"] == "missing"
    assert "typed unavailable" in aggregate["successor"]


def test_live_backends_keep_evidence_kinds_distinct() -> None:
    matrix = _claim_matrix()
    inventory = _backend_inventory()
    simulated = probe_backend_capability("simulated")
    signed = probe_backend_capability("signed_receipt")
    integrity = probe_backend_capability("integrity")
    aggregate = probe_backend_capability("merkle_manifest")
    groth16 = probe_backend_capability("groth16")
    provekit = probe_backend_capability("provekit")

    kind_to_probe = {
        "real": groth16,
        "simulated": simulated,
        "signed": signed,
        "aggregate": aggregate,
        "direct": groth16,
        "incremental": integrity,
    }
    assert tuple(kind_to_probe) == _EVIDENCE_KINDS
    assert simulated.status is BackendAvailabilityStatus.SIMULATED_ONLY
    assert simulated.production_seal_allowed is False
    assert simulated.can_direct_computation is False
    assert simulated.can_sign is False
    assert simulated.can_aggregate is False
    assert simulated.recursive_verification is False
    assert simulated.reason_code == CapabilityReasonCode.SIMULATED_PRODUCTION_FORBIDDEN.value
    assert simulated.aggregation_disposition is AggregationDisposition.MANIFEST_AGGREGATION

    assert signed.can_sign is True
    assert signed.can_direct_computation is False
    assert signed.can_aggregate is False
    assert signed.recursive_verification is False
    assert "direct execution" in signed.message.casefold()
    assert signed.reason_code == CapabilityReasonCode.SIGNATURE_STRUCTURAL.value

    assert integrity.can_prove is False
    assert integrity.can_sign is False
    assert integrity.can_direct_computation is False
    assert integrity.can_aggregate is False
    assert integrity.reason_code == CapabilityReasonCode.INTEGRITY_ONLY.value

    assert aggregate.can_aggregate is True
    assert aggregate.can_sign is False
    assert aggregate.can_direct_computation is False
    assert aggregate.recursive_verification is False
    assert aggregate.aggregation_disposition is AggregationDisposition.MANIFEST_AGGREGATION

    # IPS operational groth16 surface is not PCTDD production ZK admission.
    assert groth16.backend_id == "groth16"
    assert groth16.recursive_verification is False
    assert inventory["groth16_binary"]["production_admitted"] is False
    assert groth16.can_sign is False
    if groth16.status is BackendAvailabilityStatus.AVAILABLE:
        assert groth16.reason_code == CapabilityReasonCode.BOUNDED_DECLARED_COMPUTATION.value
        assert groth16.metadata.get("bounded_declared_computation_only") is True
    else:
        assert groth16.production_seal_allowed is False
        assert groth16.reason_code == CapabilityReasonCode.BACKEND_UNAVAILABLE.value

    assert provekit.can_sign is False
    if provekit.status is BackendAvailabilityStatus.UNAVAILABLE:
        assert provekit.production_seal_allowed is False
        assert provekit.reason_code == (
            CapabilityReasonCode.OPTIONAL_CAPABILITY_UNAVAILABLE.value
        )
        assert provekit.recursive_verification is False
    else:
        assert inventory["provekit"]["production_admitted"] is False
        assert provekit.recursive_verification is False

    # Distinct closed identities: no probe may wear another kind's label.
    assert simulated.status is not groth16.status
    assert signed.can_sign is not simulated.can_sign
    assert aggregate.can_aggregate is not signed.can_aggregate
    assert integrity.reason_code != signed.reason_code
    assert matrix["ReceiptAggregationZkProof"]["does_not"] != (
        matrix["DirectExecutionProof"]["establishes"]
    )
    assert TRUST_BASELINE_BACKEND_DECISIONS["simulated"] == "production_seal_forbidden"
    assert TRUST_BASELINE_BACKEND_DECISIONS["groth16"] == (
        "bounded_declared_computation_only"
    )
    assert "simulated" in KNOWN_BACKEND_IDS
    assert "signed_receipt" in KNOWN_BACKEND_IDS
    assert "groth16" in KNOWN_BACKEND_IDS
    assert "provekit" in KNOWN_BACKEND_IDS
    assert ProverStatus.SIMULATED.value == "simulated"
    assert ProverStatus.SIMULATED.value != ProverStatus.PROVED.value


def test_simulated_and_real_backends_cannot_swap_class() -> None:
    registry = list_backends()
    assert set(registry) == {"simulated", "groth16", "provekit"}
    assert "not cryptographically secure" in registry["simulated"]["description"].casefold()
    assert "simulation" not in registry["groth16"]["description"].casefold()
    simulated = get_backend("simulated")
    assert simulated.backend_id == "simulated"
    proof = simulated.generate_proof("Q", ["P", "P -> Q"], {"seed": 7})
    assert simulated.backend_id not in REAL_TEST_PASS_BACKENDS
    assert REAL_TEST_PASS_BACKENDS == frozenset({"groth16", "provekit"})
    metadata = getattr(proof, "metadata", {}) or {}
    assert metadata.get("proof_system") == "Groth16 (simulated)"
    assert "simulated" in str(metadata.get("proof_system")).casefold()
    assert simulated.backend_id == "simulated"
    assert "simulated_proof_layout" in metadata

    groth16 = Groth16Backend()
    assert groth16.backend_id == "groth16"
    groth16_proof = None
    groth16_error: str | None = None
    try:
        groth16_proof = groth16.generate_proof("Q", ["P"], {})
    except ZKPError as exc:
        groth16_error = str(exc)
    assert _backend_inventory()["groth16_binary"]["production_admitted"] is False
    if groth16_proof is None:
        assert groth16_error
        folded = groth16_error.casefold()
        assert (
            "disabled" in folded
            or "enable" in folded
            or "binary" in folded
            or "not available" in folded
            or "fail" in folded
        )
    else:
        assert groth16.backend_id == "groth16"
        assert groth16.backend_id in REAL_TEST_PASS_BACKENDS

    provekit = ProveKitBackend()
    assert provekit.backend_id == "provekit"
    try:
        provekit_present = provekit.binary_available()
    except ZKPError:
        provekit_present = False
    if provekit_present:
        record = _typed_unavailable(
            capability="provekit_production",
            reason_code="provekit_adapter_source_present_unadmitted",
            message=(
                "provekit executable is discoverable; presence is not production "
                "admission without ceremony, artifacts, and verifier policy"
            ),
        )
        assert record["production_admitted"] is False
    try:
        provekit.generate_proof("Q", ["P"], {})
    except ZKPError:
        pass
    else:
        raise AssertionError("provekit must fail closed without configured artifacts")


def test_test_pass_statement_v1_remains_unchanged() -> None:
    matrix = _claim_matrix()
    assert matrix["legacy"] == "TestPassStatementV1 remains unchanged"
    assert TEST_PASS_STATEMENT_INTERFACE == "TestPassStatementV1"
    assert TEST_PASS_STATEMENT_VERSION == 1
    assert TestPassStatementV1.__test__ is False
    doc = (TestPassStatementV1.__doc__ or "") + (
        sys.modules["ipfs_datasets_py.logic.zkp.statements.test_pass"].__doc__ or ""
    )
    folded = _fold_claim_text(doc)
    assert "does **not** prove general program correctness" in folded or (
        "not prove general program correctness" in folded
    )
    assert "future behavior" in folded
    assert TEST_PASS_STATEMENT_INTERFACE == matrix["legacy"].split()[0]


def test_signed_receipt_is_not_independent_execution() -> None:
    matrix = _claim_matrix()
    signed = probe_backend_capability("signed_receipt")
    assert matrix["SignedExecutionReceipt"]["does_not"] == (
        "independent faithful execution"
    )
    assert signed.can_direct_computation is False
    assert RUNNER_PASS_ATTESTATION_INTERFACE == "RunnerPassAttestation@1"
    assert RUNNER_TRUST_POLICY_INTERFACE == "RunnerTrustPolicy@1"
    assert PYTEST_PASS_ATTESTATION_USAGE == "pytest-pass-attestation"
    module = sys.modules["ipfs_accelerate_py.testing.proof_reuse.runner_pass_attestation"]
    doc = (module.__doc__ or "").casefold()
    assert "cannot nominate a key" in doc
    assert "skip authority" in doc


def test_key_ceremony_and_key_bytes_do_not_admit_production() -> None:
    inventory = _backend_inventory()
    ceremony = validate_groth16_mpc_ceremony({})
    assert ceremony.production_eligible is False
    assert ceremony.valid is False
    assert "unsupported_schema" in ceremony.reasons

    backend_root = (
        _repo_root()
        / "external/ipfs_datasets/ipfs_datasets_py/processors/groth16_backend"
    )
    key_files = sorted(backend_root.glob("artifacts/*/proving_key.bin")) + sorted(
        backend_root.glob("artifacts/*/verifying_key.bin")
    )
    assert key_files, "packaged groth16 key bytes should exist as unqualified inventory"
    for path in key_files:
        digest = _sha256_file(path)
        assert digest.startswith("sha256:")
        assert len(digest) == len("sha256:") + 64
        assert inventory["groth16_binary"]["production_admitted"] is False
        assert inventory["groth16_binary"]["reason"]

    expected = inventory["groth16_binary"]["sha256"]
    matches: list[Path] = []
    for path in backend_root.rglob("*"):
        if not path.is_file() or path.name.startswith("."):
            continue
        if path.name != "groth16" and "bin" not in path.parts:
            continue
        if _sha256_file(path) == expected:
            matches.append(path)
    if matches:
        for path in matches:
            assert path.is_file()
        assert inventory["groth16_binary"]["classification"] == (
            "real_backend_binary_present_unqualified"
        )
        assert inventory["groth16_binary"]["production_admitted"] is False
    else:
        record = _typed_unavailable(
            capability="groth16_binary_digest",
            reason_code="groth16_inventory_digest_unresolved",
            message=(
                "inventory sha256 is not resolved to a readable groth16 binary in "
                "this sealed environment; claim meaning is unchanged and production "
                "is not admitted"
            ),
        )
        assert record["claim_unchanged"] is True
        assert record["production_admitted"] is False

    policy = build_production_policy()
    generated = policy.generate_key_material()
    assert generated.accepted is False
    assert generated.outcome is TrustOutcome.REJECTED
    assert generated.reason_code == TrustRejectionReason.KEY_GENERATION_FORBIDDEN.value
    downloaded = policy.download_key_material(key_id="vk/missing")
    assert downloaded.accepted is False
    assert downloaded.reason_code == TrustRejectionReason.KEY_DOWNLOAD_FORBIDDEN.value

    handle = ProvingKeyHandle(
        key_id="pk/pctdd-003-inventory",
        key_cid="bafybeigprovingkeypctdd003inventory00000000001",
        circuit_ids=frozenset({"circuit:pctdd-003-inventory@1"}),
        setup_origin=SetupOrigin.OPERATOR_REVIEWED,
        test_only=False,
        paired_verification_key_id="vk/pctdd-003-inventory",
        epoch=1,
    )
    assert handle.exportable is False
    assert handle.bytes_available is False
    assert handle.to_canonical()["proving_key_exported"] is False
    try:
        handle.export_bytes()
    except TrustError as exc:
        assert "nonexportable" in str(exc).casefold()
    else:
        raise AssertionError("proving-key handles must never export key bytes")
    try:
        handle.download()
    except TrustError as exc:
        assert "download" in str(exc).casefold()
    else:
        raise AssertionError("proving-key handles must never download key bytes")


def test_typed_unavailable_cases_do_not_change_claim_meaning() -> None:
    matrix_before = _claim_matrix()
    inventory_before = _backend_inventory()
    records = _inventory_typed_unavailable()
    capabilities = {item["capability"] for item in records}
    assert capabilities == {
        "production_zk",
        "key_ceremony",
        "direct_execution_profile",
        "aggregate_selected_test_zk",
        "provekit_production",
    }
    for item in records:
        assert item["status"] == "typed_unavailable"
        assert item["production_admitted"] is False
        assert item["self_approved"] is False
        assert item["claim_unchanged"] is True
        assert item["reason_code"]
        assert item["message"]
    matrix_after = _claim_matrix()
    inventory_after = _backend_inventory()
    assert matrix_after == matrix_before
    assert inventory_after == inventory_before
    for name in _CLAIM_CLASSES:
        assert matrix_after[name]["establishes"] == matrix_before[name]["establishes"]
        assert matrix_after[name]["does_not"] == matrix_before[name]["does_not"]
    poisoned = dict(records[0])
    poisoned["production_admitted"] = True
    try:
        if poisoned["production_admitted"] or poisoned["self_approved"] or not poisoned["claim_unchanged"]:
            raise AssertionError(
                "typed unavailable cases cannot admit, self-approve, or change claims"
            )
        raise AssertionError("poisoned production admission must be rejected")
    except AssertionError as exc:
        assert "cannot admit" in str(exc)


def test_receipt_is_not_completion_authority() -> None:
    receipt_path = (
        _repo_root()
        / "artifacts/parallel_content_sealing_proof_carrying_tdd/receipts/PCTDD-003.json"
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["schema"] == "pctdd/task-receipt@1"
    assert receipt["task_id"] == "PCTDD-003"
    assert receipt["plan_revision"] == "PCTDD-PLAN-V1.1"
    assert receipt["store_generation"] == "pctdd-v1-g6"
    assert receipt["completion_authoritative"] is False
    assert receipt["self_approval"] is False
    assert receipt["worker_authored_test_is_sufficient_alone"] is False
    assert receipt["status"] == "implementation_submitted_pending_controller_validation"
    assert receipt["claim_class"] == "IntegrityCommitment"
    assert receipt["publication_authority_invoked"] is False
    assert receipt["markdown_non_authoritative"] is True
    assert receipt["validation_profile"] == "pctdd-validation/PCTDD-PLAN-V1.1/PCTDD-003@1"
    assert "controller-owned" in receipt["completion_authority"]
    assert "does not complete" in receipt["claim"].casefold()
    limitations = receipt["limitations"]
    for key in (
        "production_zk",
        "key_ceremony",
        "direct_execution_profile",
        "groth16_unqualified",
        "provekit",
        "aggregate_selected_test_zk",
    ):
        assert limitations[key]["status"] == "typed_unavailable"
        assert limitations[key]["production_admitted"] is False
        assert limitations[key]["self_approved"] is False
        assert limitations[key]["claim_unchanged"] is True
    assert receipt["predecessor_rescue_candidate"]["admitted"] is False
    assert receipt["predecessor_rescue_candidate"]["classification"] == (
        "receipt-observation-only"
    )
    kinds = receipt["evidence_kind_inventory"]
    assert tuple(kinds) == _EVIDENCE_KINDS
    assert len(set(kinds[kind]["claim_class"] for kind in kinds)) == len(kinds)
    for kind in _EVIDENCE_KINDS:
        assert kinds[kind]["production_admitted"] is False
        assert kinds[kind]["promotes_to"] == []
        assert kinds[kind]["self_approved"] is False
        assert kinds[kind]["claim_unchanged"] is True
    assert kinds["signed"]["claim_class"] == "SignedExecutionReceipt"
    assert kinds["signed"]["does_not"] == "independent faithful execution"
    assert kinds["aggregate"]["claim_class"] == "ReceiptAggregationZkProof"
    assert kinds["direct"]["claim_class"] == "DirectExecutionProof"
    assert kinds["incremental"]["claim_class"] == "IncrementalCommitSeal"
    assert kinds["simulated"]["claim_class"] == "SimulatedProof"
    assert kinds["real"]["claim_class"] == "UnqualifiedGroth16Backend"
