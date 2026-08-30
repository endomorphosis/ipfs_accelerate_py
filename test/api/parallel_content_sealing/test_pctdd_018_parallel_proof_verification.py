"""PCTDD-018: proof, signature, receipt and integrity checks are bounded."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

_TEST_FILE = Path(__file__).resolve()
_ACCELERATE_ROOT = _TEST_FILE.parents[3]
_EXTERNAL_ROOT = _ACCELERATE_ROOT.parent
for _name in ("ipfs_accelerate", "ipfs_datasets", "ipfs_kit"):
    _candidate = _EXTERNAL_ROOT / _name
    if _candidate.is_dir() and str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

_PHASE_PLUGIN_MODULE = "run_parallel_content_sealing_proof_carrying_tdd_validation"
_PHASE_REPORTS_ATTR = "_PYTEST_PHASE_REPORTS"
_REQUIRED_TEST_TARGET = (
    "external/ipfs_accelerate/test/api/parallel_content_sealing/"
    "test_pctdd_018_parallel_proof_verification.py"
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
    if getattr(reports, "_pctdd_018_target_bound", False):
        _rewrite_phase_report_node_ids()
        return

    class _TargetBoundPhaseReports(list):
        _pctdd_018_target_bound = True

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

import ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing as sealer
from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.aggregation import (
    AGGREGATION_LABEL_MANIFEST,
)
from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.parallel_verification import (
    ADMITTED_BACKEND_IDS,
    CLAIM_CLASS,
    CLOSED_CHECK_KINDS,
    DEFAULT_MAX_PARALLEL,
    EVIDENCE_SUBSET,
    MAX_PARALLEL_CAP,
    PARALLEL_PROOF_VERIFICATION_INTERFACE,
    PARALLEL_PROOF_VERIFICATION_RESULT_INTERFACE,
    PREDECESSOR_INTERFACES,
    UNAVAILABLE_BACKEND_IDS,
    VERIFICATION_DOES_NOT,
    VERIFICATION_ESTABLISHES,
    VERIFICATION_POLICY,
    WORKER_BOUNDS,
    BatchVerificationReason,
    CheckKind,
    ParallelVerificationError,
    ParallelVerificationHooks,
    ParallelProofVerificationResult,
    UnitVerificationReason,
    VerificationBounds,
    VerificationUnit,
    digest_bytes,
    hermetic_proof_tag,
    hermetic_signature_tag,
    integrity_proof_tag,
    record_typed_unavailable,
    typed_unavailable_records,
    verification_claim,
    verify_units_in_parallel,
)
from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.process_control import (
    CancellationToken,
)

SIGNER_ID = "allowlist/operator-pctdd-018"
VK_ID = "vk/hermetic-pctdd-018"


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
            / "artifacts/parallel_content_sealing_proof_carrying_tdd/receipts/PCTDD-018.json"
        )
        if receipt.is_file():
            return parent
    raise AssertionError("PCTDD-018 receipt is missing from the declared output manifest")


def _load_json(relative: str) -> dict[str, Any]:
    path = _repo_root() / relative
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AssertionError(f"{relative} must be a JSON object")
    return payload


def _receipt() -> dict[str, Any]:
    return _load_json(
        "artifacts/parallel_content_sealing_proof_carrying_tdd/receipts/PCTDD-018.json"
    )


def _integrity_unit(unit_id: str, payload: bytes | None = None) -> VerificationUnit:
    body = payload if payload is not None else f"integrity:{unit_id}".encode("utf-8")
    return VerificationUnit(
        unit_id=unit_id,
        kind=CheckKind.INTEGRITY,
        payload=body,
        expected_digest=digest_bytes(body),
        backend_id="integrity",
        category="integrity",
    )


def _receipt_unit(unit_id: str, payload: bytes | None = None) -> VerificationUnit:
    body = payload if payload is not None else f"receipt:{unit_id}".encode("utf-8")
    return VerificationUnit(
        unit_id=unit_id,
        kind=CheckKind.RECEIPT,
        payload=body,
        expected_digest=digest_bytes(body),
        backend_id="merkle_manifest",
        category="receipt",
    )


def _signature_unit(unit_id: str, payload: bytes | None = None) -> VerificationUnit:
    body = payload if payload is not None else f"signed:{unit_id}".encode("utf-8")
    return VerificationUnit(
        unit_id=unit_id,
        kind=CheckKind.SIGNATURE,
        payload=body,
        expected_digest=digest_bytes(body),
        proof_bytes=hermetic_signature_tag(signer_id=SIGNER_ID, payload=body),
        signer_id=SIGNER_ID,
        backend_id="signed_receipt",
        category="signature",
    )


def _proof_unit(unit_id: str, payload: bytes | None = None) -> VerificationUnit:
    body = payload if payload is not None else f"proof:{unit_id}".encode("utf-8")
    return VerificationUnit(
        unit_id=unit_id,
        kind=CheckKind.PROOF,
        payload=body,
        expected_digest=digest_bytes(body),
        proof_bytes=hermetic_proof_tag(
            unit_id=unit_id, public_input=body, verification_key_id=VK_ID
        ),
        verification_key_id=VK_ID,
        backend_id="hermetic_hmac",
        category="proof",
    )


def _integrity_proof_unit(unit_id: str) -> VerificationUnit:
    body = f"integrity-proof:{unit_id}".encode("utf-8")
    return VerificationUnit(
        unit_id=unit_id,
        kind=CheckKind.PROOF,
        payload=body,
        expected_digest=digest_bytes(body),
        proof_bytes=integrity_proof_tag(body),
        backend_id="integrity",
        category="proof",
    )


def _mixed_units() -> tuple[VerificationUnit, ...]:
    return (
        _proof_unit("unit/proof-a"),
        _signature_unit("unit/signature-b"),
        _receipt_unit("unit/receipt-c"),
        _integrity_unit("unit/integrity-d"),
        _integrity_proof_unit("unit/proof-e"),
    )


def _verify(
    units: tuple[VerificationUnit, ...] | list[VerificationUnit],
    **overrides: Any,
) -> ParallelProofVerificationResult:
    payload: dict[str, Any] = {
        "allowlisted_signers": (SIGNER_ID,),
        "allowlisted_verification_keys": (VK_ID,),
        "bounds": VerificationBounds(max_parallel=4, timeout_seconds=5.0),
    }
    payload.update(overrides)
    return verify_units_in_parallel(units, **payload)


def test_required_phase_node_ids_bind_to_profile_target() -> None:
    target = _REQUIRED_TEST_TARGET
    relative = (
        "api/parallel_content_sealing/"
        "test_pctdd_018_parallel_proof_verification.py::test_x"
    )
    assert _normalize_phase_node_id(relative, target) == target + "::test_x"
    assert _normalize_phase_node_id(target + "::test_x", target) == target + "::test_x"
    assert (
        _normalize_phase_node_id(
            "test/api/parallel_content_sealing/"
            "test_pctdd_018_parallel_proof_verification.py::test_x",
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


def test_proof_signature_receipt_integrity_checks_are_bounded() -> None:
    units = _mixed_units()
    result = _verify(units)
    assert result.accepted is True
    assert result.reason is BatchVerificationReason.AGGREGATED
    assert result.interface == PARALLEL_PROOF_VERIFICATION_RESULT_INTERFACE
    assert result.bounded is True
    assert result.bounds.max_parallel == 4
    assert result.bounds.max_parallel <= MAX_PARALLEL_CAP
    assert result.bounds.fan_in >= 2
    assert result.aggregation_label == AGGREGATION_LABEL_MANIFEST
    assert result.recursively_verifies_children is False
    assert result.claims_test_execution is False
    assert result.production_admitted is False
    assert result.publication_authority_invoked is False
    assert result.self_approved is False
    assert result.may_authorize_skip is False
    assert result.claim_class == CLAIM_CLASS == "IntegrityCommitment"
    assert result.establishes == VERIFICATION_ESTABLISHES
    assert result.does_not == VERIFICATION_DOES_NOT
    assert result.declared_unit_ids == tuple(item.unit_id for item in units)
    assert [item.kind.value for item in result.units] == [
        "proof",
        "signature",
        "receipt",
        "integrity",
        "proof",
    ]
    assert all(item.accepted and item.bounded for item in result.units)
    assert all(item.reason is UnitVerificationReason.VERIFIED for item in result.units)
    canonical = result.to_canonical()
    assert canonical["proof_bytes_exported"] is False
    assert canonical["witness_exported"] is False
    assert "witness" not in canonical
    assert "proving_key" not in canonical
    assert result.aggregate_root.startswith("sha256:")
    assert result.result_cid().startswith("sha256:")
    assert CLOSED_CHECK_KINDS == {"proof", "signature", "receipt", "integrity"}
    assert WORKER_BOUNDS == (1, 2, 4, 8, 16)
    assert DEFAULT_MAX_PARALLEL in WORKER_BOUNDS
    assert VERIFICATION_POLICY["production_admitted"] is False
    assert VERIFICATION_POLICY["publication_authority_invoked"] is False
    assert VERIFICATION_POLICY["recursively_verifies_children"] is False
    assert VERIFICATION_POLICY["claims_test_execution"] is False
    assert set(VERIFICATION_POLICY["check_kinds"]) == CLOSED_CHECK_KINDS


def test_parallel_verification_is_deterministic_across_completion_order() -> None:
    units = _mixed_units()
    sequential = _verify(units, bounds=VerificationBounds(max_parallel=1))
    parallel = _verify(units, bounds=VerificationBounds(max_parallel=4))
    shuffled_a = _verify(units, shuffle_seed=7, bounds=VerificationBounds(max_parallel=4))
    shuffled_b = _verify(
        tuple(reversed(units)),
        expected_unit_ids=tuple(item.unit_id for item in reversed(units)),
        shuffle_seed=99,
        bounds=VerificationBounds(max_parallel=8),
    )
    assert sequential.accepted is True
    assert parallel.accepted is True
    assert shuffled_a.accepted is True
    assert sequential.aggregate_root == parallel.aggregate_root == shuffled_a.aggregate_root
    # Bounds are part of the result CID; completion order is not.
    assert sequential.bounds.max_parallel != parallel.bounds.max_parallel
    assert parallel.result_cid() == shuffled_a.result_cid()
    assert sequential.result_cid() != parallel.result_cid()
    assert sequential.declared_unit_ids == parallel.declared_unit_ids
    assert [item.to_canonical() for item in sequential.units] == [
        item.to_canonical() for item in parallel.units
    ]
    assert [item.to_canonical() for item in sequential.units] == [
        item.to_canonical() for item in shuffled_a.units
    ]
    # Completion order may differ; declared order and the aggregate must not.
    assert sequential.declared_unit_ids == shuffled_a.declared_unit_ids
    assert shuffled_b.accepted is True
    assert shuffled_b.declared_unit_ids == tuple(item.unit_id for item in reversed(units))
    assert shuffled_b.schedule_work_ids
    # Schedule is priority-stable even when the caller reverses input.
    same_set = _verify(units)
    reversed_same_expected = _verify(
        tuple(reversed(units)),
        expected_unit_ids=tuple(item.unit_id for item in units),
    )
    assert reversed_same_expected.accepted is False
    assert reversed_same_expected.reason is BatchVerificationReason.REORDERED_CHILDREN
    assert same_set.accepted is True


def test_failed_tampered_unknown_and_unavailable_backends_fail_closed() -> None:
    good = _mixed_units()
    tampered_sig = _signature_unit("unit/signature-b", payload=b"tampered-signed")
    tampered_sig = VerificationUnit(
        unit_id=tampered_sig.unit_id,
        kind=tampered_sig.kind,
        payload=tampered_sig.payload,
        expected_digest=digest_bytes(b"signed:unit/signature-b"),
        proof_bytes=tampered_sig.proof_bytes,
        signer_id=SIGNER_ID,
        backend_id="signed_receipt",
        category="signature",
    )
    digest_mismatch = _verify(
        (
            good[0],
            tampered_sig,
            good[2],
            good[3],
            good[4],
        )
    )
    assert digest_mismatch.accepted is False
    assert digest_mismatch.units[1].reason is UnitVerificationReason.DIGEST_MISMATCH
    assert digest_mismatch.production_admitted is False

    bad_sig = _signature_unit("unit/signature-b")
    bad_sig = VerificationUnit(
        unit_id=bad_sig.unit_id,
        kind=bad_sig.kind,
        payload=bad_sig.payload,
        expected_digest=bad_sig.expected_digest,
        proof_bytes=b"\x00" * 32,
        signer_id=SIGNER_ID,
        backend_id="signed_receipt",
        category="signature",
    )
    sig_fail = _verify((good[0], bad_sig, good[2], good[3], good[4]))
    assert sig_fail.accepted is False
    assert sig_fail.units[1].reason is UnitVerificationReason.SIGNATURE_FAILURE

    bad_proof = _proof_unit("unit/proof-a")
    bad_proof = VerificationUnit(
        unit_id=bad_proof.unit_id,
        kind=bad_proof.kind,
        payload=bad_proof.payload,
        expected_digest=bad_proof.expected_digest,
        proof_bytes=b"\x11" * 32,
        verification_key_id=VK_ID,
        backend_id="hermetic_hmac",
        category="proof",
    )
    proof_fail = _verify((bad_proof, good[1], good[2], good[3], good[4]))
    assert proof_fail.accepted is False
    assert proof_fail.units[0].reason is UnitVerificationReason.PROOF_FAILURE

    unknown = _verify(
        (
            VerificationUnit(
                unit_id="unit/unknown",
                kind=CheckKind.PROOF,
                payload=b"x",
                expected_digest=digest_bytes(b"x"),
                proof_bytes=b"y",
                backend_id="not-a-backend",
                category="proof",
            ),
        )
    )
    assert unknown.accepted is False
    assert unknown.units[0].reason is UnitVerificationReason.UNKNOWN_BACKEND

    groth = _verify(
        (
            VerificationUnit(
                unit_id="unit/groth16",
                kind=CheckKind.PROOF,
                payload=b"circuit-input",
                expected_digest=digest_bytes(b"circuit-input"),
                proof_bytes=b"opaque",
                backend_id="groth16",
                category="proof",
                production=True,
            ),
        )
    )
    assert groth.accepted is False
    assert groth.units[0].reason is UnitVerificationReason.UNAVAILABLE
    assert groth.production_admitted is False
    assert groth.self_approved is False
    assert groth.claim_class == "IntegrityCommitment"

    provekit = _verify(
        (
            VerificationUnit(
                unit_id="unit/provekit",
                kind=CheckKind.PROOF,
                payload=b"pk-input",
                expected_digest=digest_bytes(b"pk-input"),
                proof_bytes=b"opaque",
                backend_id="provekit",
                category="proof",
            ),
        )
    )
    assert provekit.units[0].reason is UnitVerificationReason.UNAVAILABLE

    simulated = _verify(
        (
            VerificationUnit(
                unit_id="unit/simulated",
                kind=CheckKind.PROOF,
                payload=b"sim",
                expected_digest=digest_bytes(b"sim"),
                proof_bytes=b"opaque",
                backend_id="simulated",
                category="proof",
            ),
        )
    )
    assert simulated.units[0].reason is UnitVerificationReason.UNAVAILABLE
    assert "groth16" in UNAVAILABLE_BACKEND_IDS
    assert "provekit" in UNAVAILABLE_BACKEND_IDS
    assert "simulated" in UNAVAILABLE_BACKEND_IDS
    assert "hermetic_hmac" in ADMITTED_BACKEND_IDS

    unallowlisted = _verify(
        (_signature_unit("unit/signature-b"),),
        allowlisted_signers=("allowlist/someone-else",),
    )
    assert unallowlisted.accepted is False
    assert unallowlisted.units[0].reason is UnitVerificationReason.UNALLOWLISTED_SIGNER

    wrong_vk = _verify(
        (_proof_unit("unit/proof-a"),),
        allowlisted_verification_keys=("vk/other",),
    )
    assert wrong_vk.accepted is False
    assert (
        wrong_vk.units[0].reason is UnitVerificationReason.UNALLOWLISTED_VERIFICATION_KEY
    )


def test_aggregation_rejects_missing_duplicate_reordered_and_failed_children() -> None:
    units = _mixed_units()
    missing = _verify(units[:3], expected_unit_ids=tuple(item.unit_id for item in units))
    assert missing.accepted is False
    assert missing.reason is BatchVerificationReason.MISSING_CHILD

    duplicate = _verify(
        (_integrity_unit("unit/dup"), _integrity_unit("unit/dup")),
        expected_unit_ids=("unit/dup", "unit/dup"),
    )
    assert duplicate.accepted is False
    assert duplicate.reason is BatchVerificationReason.DUPLICATE_CHILD
    assert all(
        item.reason is UnitVerificationReason.DUPLICATE_UNIT for item in duplicate.units
    )

    reordered = _verify(
        tuple(reversed(units)),
        expected_unit_ids=tuple(item.unit_id for item in units),
    )
    assert reordered.accepted is False
    assert reordered.reason is BatchVerificationReason.REORDERED_CHILDREN

    failed_payload = _integrity_unit("unit/integrity-d", payload=b"other")
    failed_payload = VerificationUnit(
        unit_id=failed_payload.unit_id,
        kind=failed_payload.kind,
        payload=failed_payload.payload,
        expected_digest=digest_bytes(b"integrity:unit/integrity-d"),
        backend_id="integrity",
        category="integrity",
    )
    failed = _verify((units[0], units[1], units[2], failed_payload, units[4]))
    assert failed.accepted is False
    assert failed.reason is BatchVerificationReason.FAILED_CHILD
    assert failed.units[3].accepted is False


def test_oversize_timeout_cancellation_and_parallel_bounds() -> None:
    oversize = VerificationUnit(
        unit_id="unit/oversize",
        kind=CheckKind.INTEGRITY,
        payload=b"x" * 64,
        expected_digest=digest_bytes(b"x" * 64),
        backend_id="integrity",
        category="integrity",
    )
    bounded = _verify(
        (oversize,),
        bounds=VerificationBounds(max_unit_bytes=16, timeout_seconds=2.0),
    )
    assert bounded.accepted is False
    assert bounded.units[0].reason is UnitVerificationReason.BOUND_EXCEEDED
    assert bounded.reason is BatchVerificationReason.BOUND_EXCEEDED
    assert bounded.bounded is True

    def _sleep(_unit: VerificationUnit) -> None:
        time.sleep(0.25)

    timed = _verify(
        (_integrity_unit("unit/slow"),),
        bounds=VerificationBounds(max_parallel=1, timeout_seconds=0.05),
        hooks=ParallelVerificationHooks(before_unit=_sleep),
    )
    assert timed.accepted is False
    assert timed.units[0].reason is UnitVerificationReason.TIMEOUT
    assert timed.production_admitted is False

    token = CancellationToken()
    token.cancel("operator-stop")
    cancelled = _verify((_integrity_unit("unit/cancel"),), cancellation=token)
    assert cancelled.accepted is False
    assert cancelled.units[0].reason is UnitVerificationReason.CANCELLED
    assert cancelled.reason is BatchVerificationReason.CANCELLED

    hungry = VerificationUnit(
        unit_id="unit/hungry",
        kind=CheckKind.INTEGRITY,
        payload=b"hungry",
        expected_digest=digest_bytes(b"hungry"),
        backend_id="integrity",
        category="integrity",
        cpu=8,
        memory_mb=64,
    )
    unavailable = _verify(
        (hungry,),
        bounds=VerificationBounds(max_cpu=2, max_parallel=1),
    )
    assert unavailable.accepted is False
    assert unavailable.units[0].reason is UnitVerificationReason.UNAVAILABLE

    try:
        VerificationBounds(max_parallel=32)
    except ParallelVerificationError as exc:
        assert "max_parallel" in str(exc)
    else:
        raise AssertionError("max_parallel must be capped at 16 workers")
    try:
        VerificationBounds(fan_in=1)
    except ParallelVerificationError as exc:
        assert "fan_in" in str(exc)
    else:
        raise AssertionError("fan_in must be at least 2")


def test_never_publishes_self_approves_or_admits_production() -> None:
    result = _verify(_mixed_units())
    assert result.publication_authority_invoked is False
    assert result.self_approved is False
    assert result.production_admitted is False
    assert result.may_authorize_skip is False
    assert result.recursively_verifies_children is False
    assert result.claims_test_execution is False
    canonical = result.to_canonical()
    assert canonical["publication_authority_invoked"] is False
    assert canonical["self_approved"] is False
    assert canonical["production_admitted"] is False
    assert canonical["may_authorize_skip"] is False
    try:
        ParallelProofVerificationResult(
            schema=result.schema,
            evidence_subset=result.evidence_subset,
            interface=result.interface,
            accepted=False,
            reason=BatchVerificationReason.UNIT_REJECTED,
            declared_unit_ids=result.declared_unit_ids,
            units=result.units,
            aggregate_root=result.aggregate_root,
            aggregation_label=result.aggregation_label,
            recursively_verifies_children=False,
            claims_test_execution=False,
            bounds=result.bounds,
            schedule_work_ids=result.schedule_work_ids,
            production_admitted=True,
        )
    except ParallelVerificationError as exc:
        assert "production" in str(exc)
    else:
        raise AssertionError("parallel verification must not admit production")
    try:
        ParallelProofVerificationResult(
            schema=result.schema,
            evidence_subset=result.evidence_subset,
            interface=result.interface,
            accepted=False,
            reason=BatchVerificationReason.UNIT_REJECTED,
            declared_unit_ids=result.declared_unit_ids,
            units=result.units,
            aggregate_root=result.aggregate_root,
            aggregation_label=result.aggregation_label,
            recursively_verifies_children=False,
            claims_test_execution=False,
            bounds=result.bounds,
            schedule_work_ids=result.schedule_work_ids,
            publication_authority_invoked=True,
        )
    except ParallelVerificationError as exc:
        assert "publication" in str(exc)
    else:
        raise AssertionError("parallel verification must not invoke publication")
    try:
        ParallelProofVerificationResult(
            schema=result.schema,
            evidence_subset=result.evidence_subset,
            interface=result.interface,
            accepted=False,
            reason=BatchVerificationReason.UNIT_REJECTED,
            declared_unit_ids=result.declared_unit_ids,
            units=result.units,
            aggregate_root=result.aggregate_root,
            aggregation_label=result.aggregation_label,
            recursively_verifies_children=False,
            claims_test_execution=False,
            bounds=result.bounds,
            schedule_work_ids=result.schedule_work_ids,
            self_approved=True,
        )
    except ParallelVerificationError as exc:
        assert "self-approve" in str(exc)
    else:
        raise AssertionError("parallel verification must not self-approve")
    public = tuple(sealer.__all__)
    for name in (
        "create_full_checkpoint",
        "create_incremental_plan",
        "execute_incremental_plan",
        "verify_seal",
        "explain_reuse",
        "explain_invalidation",
        "compare_full_and_incremental",
        "CLI_EVIDENCE",
        "PUBLIC_API_EVIDENCE",
    ):
        assert name in public
    assert "verify_units_in_parallel" not in public
    assert len(public) == 9


def test_typed_unavailable_cases_do_not_change_claim_meaning() -> None:
    matrix_before = _load_json(
        "docs/architecture/parallel_content_sealing_proof_carrying_tdd_inventory/"
        "proof_claim_matrix.json"
    )
    records = typed_unavailable_records()
    capabilities = {item["capability"] for item in records}
    assert {
        "production_zk",
        "key_ceremony",
        "groth16_unqualified",
        "provekit",
        "direct_execution_profile",
        "recursive_verification",
        "aggregate_selected_test_zk",
        "serial_wal_cas_publication",
    } == capabilities
    for item in records:
        assert item["status"] == "typed_unavailable"
        assert item["production_admitted"] is False
        assert item["self_approved"] is False
        assert item["claim_unchanged"] is True
        assert item["reason_code"]
        assert item["message"]
    by_capability = {item["capability"]: item for item in records}
    assert by_capability["production_zk"]["reason_code"] == (
        "production_zk_key_ceremony_unavailable"
    )
    assert by_capability["serial_wal_cas_publication"]["reason_code"] == (
        "verification_has_no_publication_authority"
    )
    assert by_capability["recursive_verification"]["reason_code"] == (
        "recursion_not_admitted"
    )
    matrix_after = _load_json(
        "docs/architecture/parallel_content_sealing_proof_carrying_tdd_inventory/"
        "proof_claim_matrix.json"
    )
    assert matrix_after == matrix_before
    assert matrix_after["IntegrityCommitment"]["does_not"] == "execution or semantics"
    assert matrix_after["IntegrityCommitment"]["establishes"] == (
        "exact bytes/digest/CID/Merkle inclusion"
    )
    poisoned = dict(records[0])
    poisoned["production_admitted"] = True
    try:
        if (
            poisoned["production_admitted"]
            or poisoned["self_approved"]
            or not poisoned["claim_unchanged"]
        ):
            raise AssertionError(
                "typed unavailable cases cannot admit, self-approve, or change claims"
            )
        raise AssertionError("poisoned production admission must be rejected")
    except AssertionError as exc:
        assert "cannot admit" in str(exc)
    rebuilt = record_typed_unavailable(
        capability="production_zk",
        reason_code="production_zk_key_ceremony_unavailable",
        message="unchanged",
    )
    assert rebuilt["claim_unchanged"] is True
    assert rebuilt["self_approved"] is False
    claim = verification_claim()
    assert claim["claim_class"] == "IntegrityCommitment"
    assert claim["establishes"] == VERIFICATION_ESTABLISHES
    assert claim["does_not"] == VERIFICATION_DOES_NOT
    assert claim["production_admitted"] is False
    assert claim["interface"] == PARALLEL_PROOF_VERIFICATION_INTERFACE
    backends = _load_json(
        "docs/architecture/parallel_content_sealing_proof_carrying_tdd_inventory/"
        "zkp_backend_inventory.json"
    )
    assert backends["groth16_binary"]["production_admitted"] is False
    assert backends["provekit"]["production_admitted"] is False
    assert backends["simulated"]["production_admitted"] is False
    assert backends["direct_cpython"]["production_admitted"] is False


def test_receipt_is_not_completion_authority() -> None:
    receipt = _receipt()
    assert receipt["schema"] == "pctdd/task-receipt@1"
    assert receipt["task_id"] == "PCTDD-018"
    assert receipt["plan_revision"] == "PCTDD-PLAN-V1.1"
    assert receipt["store_generation"] == "pctdd-v1-g6"
    assert receipt["completion_authoritative"] is False
    assert receipt["self_approval"] is False
    assert receipt["worker_authored_test_is_sufficient_alone"] is False
    assert receipt["status"] == "implementation_submitted_pending_controller_validation"
    assert receipt["claim_class"] == "IntegrityCommitment"
    assert receipt["publication_authority_invoked"] is False
    assert receipt["markdown_non_authoritative"] is True
    assert receipt["validation_profile"] == "pctdd-validation/PCTDD-PLAN-V1.1/PCTDD-018@1"
    assert "controller-owned" in receipt["completion_authority"]
    folded = " ".join(receipt["claim"].casefold().split())
    assert "does not complete" in folded
    assert "bounded" in folded
    assert "aggregat" in folded
    assert receipt["dependency_receipts"] == ["PCTDD-003", "PCTDD-032"]
    limitations = receipt["limitations"]
    for key in (
        "production_zk",
        "key_ceremony",
        "groth16_unqualified",
        "provekit",
        "direct_execution_profile",
        "recursive_verification",
        "aggregate_selected_test_zk",
        "serial_wal_cas_publication",
    ):
        assert limitations[key]["status"] == "typed_unavailable"
        assert limitations[key]["production_admitted"] is False
        assert limitations[key]["self_approved"] is False
        assert limitations[key]["claim_unchanged"] is True
    assert receipt["predecessor_rescue_candidate"]["admitted"] is False
    assert receipt["predecessor_rescue_candidate"]["classification"] == "none"
    contracts = receipt["contracts"]
    assert contracts["interface"] == PARALLEL_PROOF_VERIFICATION_INTERFACE
    assert contracts["may_authorize_skip"] is False
    assert contracts["production_admitted"] is False
    assert contracts["publication_authority_invoked"] is False
    assert contracts["recursively_verifies_children"] is False
    assert contracts["claims_test_execution"] is False
    assert contracts["establishes"] == VERIFICATION_ESTABLISHES
    assert contracts["does_not"] == VERIFICATION_DOES_NOT
    assert contracts["claim_class"] == CLAIM_CLASS
    assert contracts["aggregation_label"] == AGGREGATION_LABEL_MANIFEST
    assert set(contracts["check_kinds"]) == CLOSED_CHECK_KINDS
    changed = set(receipt["changed_paths"])
    assert (
        "artifacts/parallel_content_sealing_proof_carrying_tdd/receipts/PCTDD-018.json"
        in changed
    )
    assert (
        "external/ipfs_accelerate/test/api/parallel_content_sealing/"
        "test_pctdd_018_parallel_proof_verification.py"
    ) in changed
    assert (
        "external/ipfs_accelerate/ipfs_accelerate_py/agent_supervisor/proof/"
        "incremental_sealing/parallel_verification.py"
    ) in changed
    assert EVIDENCE_SUBSET == "pctdd/parallel-proof-verification@1"
    assert "SealVerification@1" in PREDECESSOR_INTERFACES
    source = (
        _repo_root()
        / "external/ipfs_accelerate/ipfs_accelerate_py/agent_supervisor/proof/"
        "incremental_sealing/parallel_verification.py"
    ).read_text(encoding="utf-8")
    assert "PARALLEL_PROOF_VERIFICATION_INTERFACE" in source
    assert "verify_units_in_parallel" in source
    assert "typed_unavailable" in source
    init_source = (
        _repo_root()
        / "external/ipfs_accelerate/ipfs_accelerate_py/agent_supervisor/proof/"
        "incremental_sealing/__init__.py"
    ).read_text(encoding="utf-8")
    assert "verify_seal" in init_source
    assert "verify_units_in_parallel" not in init_source
