"""DCR-073 receipt-only post-repair validation and reproof foundation.

No validator, analyzer, prover, command, network client, or repository is
opened here.  This module only checks externally produced typed receipts.  The
current DCR-070/DCR-072 routes cannot produce an admitted packet and a real
validation-pending transaction together, so successful structural checks stay
integration-pending and never publish, merge, or complete a repair.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum
from pathlib import PurePosixPath
from typing import Final

from ..proof.formal_verification_contracts import canonical_json_bytes, content_identity
from .contracts import (
    PostEditValidationReceipt,
    RepairAdmissionReceipt,
    RepairAuthorityRoots,
    ReproofReceipt,
)
from .transaction import TransactionDisposition, TransactionJournal, TransactionState

DCR073_VALIDATION_SCHEMA: Final = "ipfs_accelerate_py/agent-supervisor/dcr-073-post-repair@1"
DCR073_ACTIVATION: Final = "integration_pending_dcr070_dcr072"
_REQUIRED_DETECTORS: Final[dict[str, str]] = {
    "static_validation": "dcr073.static-validator@1",
    "analyzer_index": "dcr012.analyzer-index@1",
    "graph": "dcr021.contract-graph@1",
    "mismatch_closure": "dcr024.mismatch-closure@1",
    "reconstruction": "dcr033.kernel-reconstruction@1",
    "cache_equivalence": "dcr034.cache-cold-equivalence@1",
    "logic_stages": "dcr035.required-logic-stages@1",
}
_LIVE_DETECTOR: Final[tuple[str, str]] = ("live_observation", "dcr023.live-observation@1")


class PostRepairDisposition(str, Enum):  # noqa: UP042 - package supports Python 3.8
    INTEGRATION_PENDING = "integration_pending"
    REFUTED = "refuted"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


class PostRepairValidationError(ValueError):
    """A receipt cannot prove the requested post-repair transition."""


def _digest(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _path(value: str) -> str:
    candidate = PurePosixPath(value)
    if not value or candidate.is_absolute() or ".." in candidate.parts or "\x00" in value:
        raise PostRepairValidationError("changed path is unsafe")
    return candidate.as_posix()


@dataclass(frozen=True)
class RepairValidationRoots:
    """Exact before/after forest, graph, observation, finding, and proof roots."""

    forest_cid: str
    graph_cid: str
    epoch_cid: str
    finding_cid: str
    proof_cid: str

    def __post_init__(self) -> None:
        if any(not isinstance(item, str) or not item.strip() for item in self.to_dict().values()):
            raise PostRepairValidationError("validation roots must be non-empty identifiers")

    def to_dict(self) -> dict[str, str]:
        return {
            "forest_cid": self.forest_cid,
            "graph_cid": self.graph_cid,
            "epoch_cid": self.epoch_cid,
            "finding_cid": self.finding_cid,
            "proof_cid": self.proof_cid,
        }


@dataclass(frozen=True)
class AfterSourceReceipt:
    """Exact changed source bytes; source content is not semantic authority."""

    relative_path: str
    before_digest: str
    after_bytes: bytes
    after_digest: str

    def __post_init__(self) -> None:
        _path(self.relative_path)
        if not isinstance(self.after_bytes, bytes) or not self.after_bytes:
            raise PostRepairValidationError("after source bytes are required")
        if self.after_digest != _digest(self.after_bytes):
            raise PostRepairValidationError("after source digest does not match exact bytes")
        if self.before_digest == self.after_digest:
            raise PostRepairValidationError("changed source must not retain stale bytes")


@dataclass(frozen=True)
class AfterEpochDetectorReceipt:
    """One current detector-produced, content-addressed post-repair receipt."""

    kind: str
    detector_id: str
    roots: RepairValidationRoots
    changed_paths: tuple[str, ...]
    status: str
    provenance: str
    argv: tuple[str, ...] = ()
    exit_code: int = 0
    output_digest: str = ""
    model_call_count: int = 0
    provider_call_count: int = 0
    network_call_count: int = 0

    def __post_init__(self) -> None:
        if not self.kind or not self.detector_id or not isinstance(self.roots, RepairValidationRoots):
            raise PostRepairValidationError("detector receipt has no typed identity")
        paths = tuple(_path(item) for item in self.changed_paths)
        if not paths or len(paths) != len(set(paths)):
            raise PostRepairValidationError("detector changed-path accounting is not exact")
        object.__setattr__(self, "changed_paths", paths)
        if self.status != "passed" or self.provenance != "detector":
            raise PostRepairValidationError("expected/copied/synthetic/skipped detector status is denied")
        if any(
            type(item) is not int or item != 0
            for item in (self.model_call_count, self.provider_call_count, self.network_call_count)
        ):
            raise PostRepairValidationError("detector receipt must record zero model/provider/network calls")
        if self.kind == "static_validation":
            if not self.argv or any(not item or any(char.isspace() for char in item) for item in self.argv):
                raise PostRepairValidationError("static validation must use structured argv")
            if self.exit_code != 0 or not self.output_digest.startswith("sha256:"):
                raise PostRepairValidationError("static validation did not pass with digest")
        elif self.argv or self.exit_code != 0 or self.output_digest:
            raise PostRepairValidationError("only static validation may carry command result fields")

    @property
    def receipt_cid(self) -> str:
        return content_identity(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": DCR073_VALIDATION_SCHEMA,
            "kind": self.kind,
            "detector_id": self.detector_id,
            "roots": self.roots.to_dict(),
            "changed_paths": list(self.changed_paths),
            "status": self.status,
            "provenance": self.provenance,
            "argv": list(self.argv),
            "exit_code": self.exit_code,
            "output_digest": self.output_digest,
            "model_call_count": self.model_call_count,
            "provider_call_count": self.provider_call_count,
            "network_call_count": self.network_call_count,
        }


@dataclass(frozen=True)
class RepairProofTransition:
    """Non-publishing validation/reproof result; no transition grants completion."""

    disposition: PostRepairDisposition
    reason_codes: tuple[str, ...]
    before_roots: RepairValidationRoots
    after_roots: RepairValidationRoots
    detector_receipt_cids: tuple[str, ...]
    validation_receipt: PostEditValidationReceipt | None = None
    reproof_receipt: ReproofReceipt | None = None

    @property
    def transition_cid(self) -> str:
        return content_identity(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": DCR073_VALIDATION_SCHEMA,
            "authoritative": False,
            "activation_status": DCR073_ACTIVATION,
            "disposition": self.disposition.value,
            "reason_codes": list(self.reason_codes),
            "before_roots": self.before_roots.to_dict(),
            "after_roots": self.after_roots.to_dict(),
            "detector_receipt_cids": list(self.detector_receipt_cids),
            "validation_receipt_cid": (
                self.validation_receipt.content_id if self.validation_receipt else ""
            ),
            "reproof_receipt_cid": self.reproof_receipt.content_id if self.reproof_receipt else "",
            "execution_authorized": False,
            "completion_authorized": False,
            "publication_authorized": False,
            "model_call_count": 0,
            "provider_call_count": 0,
            "network_call_count": 0,
        }


@dataclass(frozen=True)
class PostRepairValidationRequest:
    repair_id: str
    authority_roots: RepairAuthorityRoots
    admission: RepairAdmissionReceipt
    transaction: TransactionJournal
    before_roots: RepairValidationRoots
    after_roots: RepairValidationRoots
    after_sources: tuple[AfterSourceReceipt, ...]
    detector_receipts: tuple[AfterEpochDetectorReceipt, ...]
    live_observation_required: bool
    cancelled: bool = False


def _failure(
    request: PostRepairValidationRequest, disposition: PostRepairDisposition, *reasons: str
) -> RepairProofTransition:
    return RepairProofTransition(
        disposition=disposition,
        reason_codes=tuple(sorted(set(reasons))),
        before_roots=request.before_roots,
        after_roots=request.after_roots,
        detector_receipt_cids=(),
    )


def evaluate_post_repair_validation(request: PostRepairValidationRequest) -> RepairProofTransition:
    """Validate externally-produced epochs only; never execute their argv."""

    if not isinstance(request, PostRepairValidationRequest):
        raise PostRepairValidationError("request must be a typed post-repair validation request")
    if request.cancelled:
        return _failure(request, PostRepairDisposition.CANCELLED, "cancelled_before_validation")
    try:
        if not isinstance(request.admission, RepairAdmissionReceipt):
            raise PostRepairValidationError("exact admitted DCR-070 packet is absent")
        if request.admission.repair_id != request.repair_id or request.admission.authority_roots != request.authority_roots:
            raise PostRepairValidationError("admitted packet roots are stale or mismatched")
        if request.before_roots.forest_cid != request.authority_roots.repository_forest_cid:
            raise PostRepairValidationError("before forest does not bind exact repair authority roots")
        transaction = request.transaction
        if (
            not isinstance(transaction, TransactionJournal)
            or transaction.state is not TransactionState.VALIDATION_PENDING
            or transaction.disposition is not TransactionDisposition.VALIDATION_PENDING
            or transaction.admission_cid != request.admission.content_id
            or not transaction.writes
        ):
            raise PostRepairValidationError("DCR-072 validation-pending transaction is absent or pending")
        if request.before_roots.epoch_cid == request.after_roots.epoch_cid:
            raise PostRepairValidationError("after epoch is unchanged and stale")
        paths = tuple(item.relative_path for item in request.after_sources)
        write_paths = tuple(item.relative_path for item in transaction.writes)
        if not paths or len(paths) != len(set(paths)) or set(paths) != set(write_paths):
            raise PostRepairValidationError("changed-path accounting does not match DCR-072 write set")
        writes = {item.relative_path: item for item in transaction.writes}
        for source in request.after_sources:
            write = writes[source.relative_path]
            if source.before_digest != write.before_digest or source.after_digest != write.after_digest:
                raise PostRepairValidationError("after source does not bind DCR-072 before/after bytes")
        expected = dict(_REQUIRED_DETECTORS)
        if request.live_observation_required:
            expected[_LIVE_DETECTOR[0]] = _LIVE_DETECTOR[1]
        by_kind: dict[str, AfterEpochDetectorReceipt] = {}
        for receipt in request.detector_receipts:
            if not isinstance(receipt, AfterEpochDetectorReceipt):
                raise PostRepairValidationError("after-epoch receipt must be typed")
            if receipt.kind in by_kind:
                raise PostRepairValidationError("duplicate after-epoch receipt kind")
            by_kind[receipt.kind] = receipt
        if set(by_kind) != set(expected):
            raise PostRepairValidationError("after-epoch receipt set is incomplete or unsupported")
        for kind, detector_id in expected.items():
            receipt = by_kind[kind]
            if receipt.detector_id != detector_id or receipt.roots != request.after_roots:
                raise PostRepairValidationError("detector producer or roots drifted")
            if set(receipt.changed_paths) != set(paths):
                raise PostRepairValidationError("detector omitted or invented a changed path")
            if receipt.receipt_cid != content_identity(receipt.to_dict()):
                raise PostRepairValidationError("detector receipt identity does not recompute")
        # Current DCR-070 and DCR-072 runtime integrations remain unavailable.
        # Thus even structurally passing typed fixtures cannot mint DCR-002
        # validation/reproof envelopes, publish, merge, or complete anything.
        return RepairProofTransition(
            disposition=PostRepairDisposition.INTEGRATION_PENDING,
            reason_codes=("dcr070_dcr072_runtime_integration_pending",),
            before_roots=request.before_roots,
            after_roots=request.after_roots,
            detector_receipt_cids=tuple(sorted(item.receipt_cid for item in by_kind.values())),
        )
    except PostRepairValidationError as exc:
        return _failure(request, PostRepairDisposition.REFUTED, str(exc))


def canonical_repair_proof_transition_bytes(value: RepairProofTransition) -> bytes:
    if not isinstance(value, RepairProofTransition):
        raise PostRepairValidationError("transition must be typed")
    return canonical_json_bytes(value.to_dict())


__all__ = [
    "AfterEpochDetectorReceipt",
    "AfterSourceReceipt",
    "DCR073_ACTIVATION",
    "DCR073_VALIDATION_SCHEMA",
    "PostRepairDisposition",
    "PostRepairValidationError",
    "PostRepairValidationRequest",
    "RepairProofTransition",
    "RepairValidationRoots",
    "canonical_repair_proof_transition_bytes",
    "evaluate_post_repair_validation",
]
