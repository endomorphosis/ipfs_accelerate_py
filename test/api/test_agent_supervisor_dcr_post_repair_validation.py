"""DCR-073 receipt-only validation tests; no commands or repositories run."""

from __future__ import annotations

import hashlib

from ipfs_accelerate_py.agent_supervisor.autonomous_repair.contracts import (
    RepairAdmissionReceipt,
    RepairAuthorityRoots,
)
from ipfs_accelerate_py.agent_supervisor.autonomous_repair.transaction import (
    FencedWriteReceipt,
    TransactionDisposition,
    TransactionJournal,
    TransactionState,
)
from ipfs_accelerate_py.agent_supervisor.autonomous_repair.validation import (
    AfterEpochDetectorReceipt,
    AfterSourceReceipt,
    PostRepairDisposition,
    PostRepairValidationRequest,
    RepairValidationRoots,
    canonical_repair_proof_transition_bytes,
    evaluate_post_repair_validation,
)


def _digest(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _request(*, cancelled: bool = False) -> PostRepairValidationRequest:
    roots = RepairAuthorityRoots(
        repository_id="repo",
        repository_forest_cid="cid:before-forest",
        git_tree_id="tree-before",
        policy_root="cid:policy",
        rpr_plan_cid="cid:plan",
        rpr_packet_cid="cid:packet",
    )
    admission = RepairAdmissionReceipt(
        repair_id="repair-073",
        authority_roots=roots,
        predecessor_evidence_cid="cid:predecessor",
        derivation_cid="cid:derivation",
    )
    before, after = b"before\n", b"after\n"
    write = FencedWriteReceipt(
        relative_path="security.py",
        before_digest=_digest(before),
        after_digest=_digest(after),
        inverse_digest=_digest(before),
        fence_id="fence",
    )
    transaction = TransactionJournal(
        transaction_id="txn",
        state=TransactionState.VALIDATION_PENDING,
        disposition=TransactionDisposition.VALIDATION_PENDING,
        reason="isolated validation pending",
        root_realpath="/tmp/isolated",
        baseline_digest="sha256:" + "a" * 64,
        admission_cid=admission.content_id,
        preview_cid="cid:preview",
        lease_id="lease",
        fence_id="fence",
        writes=(write,),
        rollback_verified=True,
    )
    before_roots = RepairValidationRoots(
        forest_cid="cid:before-forest",
        graph_cid="cid:before-graph",
        epoch_cid="cid:before-epoch",
        finding_cid="cid:before-finding",
        proof_cid="cid:before-proof",
    )
    after_roots = RepairValidationRoots(
        forest_cid="cid:after-forest",
        graph_cid="cid:after-graph",
        epoch_cid="cid:after-epoch",
        finding_cid="cid:after-finding",
        proof_cid="cid:after-proof",
    )
    expected = {
        "static_validation": "dcr073.static-validator@1",
        "analyzer_index": "dcr012.analyzer-index@1",
        "graph": "dcr021.contract-graph@1",
        "mismatch_closure": "dcr024.mismatch-closure@1",
        "reconstruction": "dcr033.kernel-reconstruction@1",
        "cache_equivalence": "dcr034.cache-cold-equivalence@1",
        "logic_stages": "dcr035.required-logic-stages@1",
        "live_observation": "dcr023.live-observation@1",
    }
    receipts = tuple(
        AfterEpochDetectorReceipt(
            kind=kind,
            detector_id=detector,
            roots=after_roots,
            changed_paths=("security.py",),
            status="passed",
            provenance="detector",
            argv=("pytest", "security.py") if kind == "static_validation" else (),
            output_digest="sha256:" + "b" * 64 if kind == "static_validation" else "",
        )
        for kind, detector in expected.items()
    )
    return PostRepairValidationRequest(
        repair_id="repair-073",
        authority_roots=roots,
        admission=admission,
        transaction=transaction,
        before_roots=before_roots,
        after_roots=after_roots,
        after_sources=(
            AfterSourceReceipt(
                relative_path="security.py",
                before_digest=_digest(before),
                after_bytes=after,
                after_digest=_digest(after),
            ),
        ),
        detector_receipts=receipts,
        live_observation_required=True,
        cancelled=cancelled,
    )


def test_typed_current_closed_fixture_is_pending_and_never_publishes() -> None:
    result = evaluate_post_repair_validation(_request())

    assert result.disposition is PostRepairDisposition.INTEGRATION_PENDING
    assert result.validation_receipt is result.reproof_receipt is None
    payload = result.to_dict()
    assert payload["execution_authorized"] is payload["completion_authorized"] is False
    assert payload["publication_authorized"] is False
    assert canonical_repair_proof_transition_bytes(result) == canonical_repair_proof_transition_bytes(result)


def test_stale_epoch_or_missing_changed_path_receipt_refutes() -> None:
    request = _request()
    request = PostRepairValidationRequest(
        **{**request.__dict__, "after_roots": request.before_roots}
    )
    assert evaluate_post_repair_validation(request).disposition is PostRepairDisposition.REFUTED

    request = _request()
    request = PostRepairValidationRequest(
        **{**request.__dict__, "detector_receipts": request.detector_receipts[:-1]}
    )
    assert evaluate_post_repair_validation(request).disposition is PostRepairDisposition.REFUTED


def test_cancellation_cannot_become_a_success_or_reproof() -> None:
    result = evaluate_post_repair_validation(_request(cancelled=True))
    assert result.disposition is PostRepairDisposition.CANCELLED
    assert result.validation_receipt is result.reproof_receipt is None
