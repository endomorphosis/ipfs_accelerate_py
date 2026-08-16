"""DCR-082 pure recovery replay; no locks, writes, processes, or providers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Final

from ..autonomous_repair.contracts import RepairAdmissionReceipt, RepairAuthorityRoots
from ..autonomous_repair.transaction import TransactionJournal, TransactionState
from ..autonomous_repair.validation import RepairProofTransition
from ..planning.deterministic_candidate_portfolio import CandidatePortfolio
from ..planning.deterministic_failure_memory import (
    FailureMemoryReceipt,
    ReplanMemoryDecision,
    ReplanMemoryDisposition,
    decide_replan,
)
from ..planning.proof_carrying_repair_dag import RepairPlanDagResult
from ..planning.repair_resource_scheduler import RepairResourceSchedule, ScheduledRepairNode
from ..proof.formal_verification_contracts import canonical_json_bytes, content_identity
from .deterministic_repair_composition import (
    DCR080_COMPOSITION_SCHEMA,
    DeterministicRepairCompositionDisposition,
    DeterministicRepairCompositionResult,
)

DCR082_RECOVERY_SCHEMA: Final = "ipfs_accelerate_py/agent-supervisor/dcr-082-recovery@1"
DCR082_ACTIVATION: Final = "integration_pending_live_dcr080"


class RecoveryOutcome(str, Enum):  # noqa: UP042 - Python 3.8
    TRANSIENT = "transient"
    STALE = "stale"
    CONFLICT = "conflict"
    CANCEL = "cancel"
    CRASH = "crash"


class RecoveryDisposition(str, Enum):  # noqa: UP042 - Python 3.8
    INTEGRATION_PENDING = "integration_pending"
    ABSTAINED = "abstained"
    REJECTED = "rejected"


class RecoveryError(ValueError):
    pass


@dataclass(frozen=True)
class RecoveryRequest:
    task_id: str
    outcome: RecoveryOutcome
    prior_failure: FailureMemoryReceipt
    new_failure: FailureMemoryReceipt
    replan: ReplanMemoryDecision
    portfolio: CandidatePortfolio
    plan_result: RepairPlanDagResult
    current_roots: RepairAuthorityRoots
    history: tuple[FailureMemoryReceipt, ...]
    prior_schedule: RepairResourceSchedule
    reacquired_schedule: RepairResourceSchedule
    prior_lease: ScheduledRepairNode
    reacquired_lease: ScheduledRepairNode
    admission: RepairAdmissionReceipt
    journal: TransactionJournal
    dcr080: DeterministicRepairCompositionResult
    validation: RepairProofTransition | None = None


@dataclass(frozen=True)
class RecoveryDecision:
    disposition: RecoveryDisposition
    reason_codes: tuple[str, ...]
    replay_cid: str = ""
    task_id: str = ""
    roots_cid: str = ""
    dcr080_receipt_cid: str = ""

    @property
    def decision_cid(self) -> str:
        return content_identity(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": DCR082_RECOVERY_SCHEMA,
            "authoritative": False,
            "activation_status": DCR082_ACTIVATION,
            "disposition": self.disposition.value,
            "reason_codes": list(self.reason_codes),
            "replay_cid": self.replay_cid,
            "task_id": self.task_id,
            "roots_cid": self.roots_cid,
            "dcr080_receipt_cid": self.dcr080_receipt_cid,
            "execution_authorized": False,
            "completion_authorized": False,
            "write_call_count": 0,
            "lock_call_count": 0,
            "process_call_count": 0,
            "provider_call_count": 0,
            "model_call_count": 0,
            "network_call_count": 0,
        }


def _result(disposition: RecoveryDisposition, *reasons: str) -> RecoveryDecision:
    return RecoveryDecision(disposition, tuple(sorted(set(reasons))))


def replay_recovery(request: RecoveryRequest) -> RecoveryDecision:
    """Classify an exact journal replay.  It cannot start, retry, or mutate work."""

    if not isinstance(request, RecoveryRequest):
        raise RecoveryError("recovery request must be typed")
    try:
        if not isinstance(request.outcome, RecoveryOutcome):
            raise RecoveryError("recovery outcome is not closed")
        for receipt in (request.prior_failure, request.new_failure):
            if (
                not isinstance(receipt, FailureMemoryReceipt)
                or receipt.receipt_cid != receipt.attempt.content_id
            ):
                raise RecoveryError("DCR-063 failure receipt is forged")
        if request.new_failure.attempt.previous_receipt_cid != request.prior_failure.receipt_cid:
            raise RecoveryError("new failure receipt is not append-only from prior failure")
        if request.new_failure.receipt_cid == request.prior_failure.receipt_cid:
            raise RecoveryError("duplicate recovery replay has no new evidence")
        if not set(request.new_failure.attempt.evidence_cids).difference(
            request.prior_failure.attempt.evidence_cids
        ):
            raise RecoveryError("retry requires strictly new typed evidence")
        if not request.new_failure.attempt.measure < request.prior_failure.attempt.measure:
            raise RecoveryError("retry measure did not decrease")
        expected_decision = decide_replan(
            request.portfolio,
            request.plan_result,
            request.current_roots,
            request.new_failure.attempt,
            history=request.history,
        )
        if (
            not isinstance(request.replan, ReplanMemoryDecision)
            or request.replan != expected_decision
            or request.replan.disposition is not ReplanMemoryDisposition.RETRY_PENDING
        ):
            raise RecoveryError("DCR-063 retry decision is not pending")
        if not isinstance(request.prior_lease, ScheduledRepairNode) or not isinstance(
            request.reacquired_lease, ScheduledRepairNode
        ):
            raise RecoveryError("DCR-064 typed leases are required")
        if (
            request.prior_lease.node_id != request.reacquired_lease.node_id
            or request.prior_lease.fence_cid == request.reacquired_lease.fence_cid
        ):
            raise RecoveryError("lease must be reacquired with a distinct fence")
        for schedule, node in (
            (request.prior_schedule, request.prior_lease),
            (request.reacquired_schedule, request.reacquired_lease),
        ):
            if (
                not isinstance(schedule, RepairResourceSchedule)
                or not schedule.schedule_cid
                or node not in schedule.nodes
            ):
                raise RecoveryError("DCR-064 schedule/node/fence binding is absent or foreign")
        if (
            not isinstance(request.admission, RepairAdmissionReceipt)
            or request.admission.repair_id != request.task_id
        ):
            raise RecoveryError("exact DCR-070 packet is required")
        if (
            not isinstance(request.journal, TransactionJournal)
            or request.journal.admission_cid != request.admission.content_id
        ):
            raise RecoveryError("DCR-072 journal is stale or unbound")
        if request.journal.state is TransactionState.VALIDATION_PENDING:
            raise RecoveryError("prior mutation cannot be retried before validation")
        if (
            request.journal.state is TransactionState.ROLLED_BACK
            and not request.journal.rollback_verified
        ):
            raise RecoveryError("partial write rollback is not byte-verified")
        if request.journal.state not in {
            TransactionState.ROLLED_BACK,
            TransactionState.CANCELLED,
            TransactionState.REJECTED,
            TransactionState.INTEGRATION_PENDING,
        }:
            raise RecoveryError("journal state is not idempotently replayable")
        if request.validation is not None and not isinstance(
            request.validation, RepairProofTransition
        ):
            raise RecoveryError("DCR-073 state must be typed when present")
        if (
            request.validation is not None
            and request.validation.before_roots.forest_cid
            != request.current_roots.repository_forest_cid
        ):
            raise RecoveryError("DCR-073 validation roots are stale for recovery")
        dcr080_body = (
            request.dcr080.to_dict()
            if isinstance(request.dcr080, DeterministicRepairCompositionResult)
            else {}
        )
        if (
            not isinstance(request.dcr080, DeterministicRepairCompositionResult)
            or request.dcr080.task_id != request.task_id
            or request.dcr080.disposition
            not in {
                DeterministicRepairCompositionDisposition.DEFER_CAPABILITY,
                DeterministicRepairCompositionDisposition.REJECTED,
            }
            or request.dcr080.receipt_cid != content_identity(dcr080_body)
            or dcr080_body.get("schema") != DCR080_COMPOSITION_SCHEMA
            or any(
                dcr080_body.get(name) != 0
                for name in ("model_call_count", "provider_call_count", "network_call_count")
            )
        ):
            raise RecoveryError("DCR-080 daemon receipt is absent, forged, or wrong task")
        expected_states = {
            RecoveryOutcome.TRANSIENT: {TransactionState.INTEGRATION_PENDING},
            RecoveryOutcome.STALE: {TransactionState.REJECTED},
            RecoveryOutcome.CONFLICT: {TransactionState.REJECTED},
            RecoveryOutcome.CANCEL: {TransactionState.CANCELLED},
            RecoveryOutcome.CRASH: {TransactionState.ROLLED_BACK},
        }
        if request.journal.state not in expected_states[request.outcome]:
            raise RecoveryError("outcome does not match replayable journal state")
        replay_cid = content_identity(
            {
                "prior_failure": request.prior_failure.receipt_cid,
                "new_failure": request.new_failure.receipt_cid,
                "prior_fence": request.prior_lease.fence_cid,
                "new_fence": request.reacquired_lease.fence_cid,
                "journal": request.journal.journal_cid,
                "dcr080": request.dcr080.receipt_cid,
            }
        )
        return RecoveryDecision(
            RecoveryDisposition.INTEGRATION_PENDING,
            ("live_dcr080_recovery_executor_required",),
            replay_cid,
            request.task_id,
            request.current_roots.content_id,
            request.dcr080.receipt_cid,
        )
    except RecoveryError as exc:
        return _result(RecoveryDisposition.ABSTAINED, str(exc))


def canonical_recovery_decision_bytes(value: RecoveryDecision) -> bytes:
    if not isinstance(value, RecoveryDecision):
        raise RecoveryError("recovery decision must be typed")
    return canonical_json_bytes(value.to_dict())


__all__ = [
    "DCR082_ACTIVATION",
    "DCR082_RECOVERY_SCHEMA",
    "RecoveryDecision",
    "RecoveryDisposition",
    "RecoveryError",
    "RecoveryOutcome",
    "RecoveryRequest",
    "canonical_recovery_decision_bytes",
    "replay_recovery",
]
