"""DCR-063 append-only deterministic failure memory and retry decisions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Final, Sequence

from ..autonomous_repair.contracts import RepairAuthorityRoots, repair_evidence_cid
from .deterministic_candidate_portfolio import CandidatePortfolio
from .proof_carrying_repair_dag import RepairPlanDagResult


DCR_FAILURE_MEMORY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-repair-failure-memory@1"
)


class FailureClass(str, Enum):
    STALE = "stale"
    CONFLICT = "conflict"
    VALIDATION = "validation"
    PROOF = "proof"
    RESOURCE = "resource"
    CAPABILITY = "capability"


class ReplanMemoryDisposition(str, Enum):
    RETRY_PENDING = "retry_pending"
    NO_WORK = "no_work"
    ABSTAINED = "abstained"
    REJECTED = "rejected"


def _cid(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be non-empty exact text")
    if "synthetic" in value.lower() or "stub" in value.lower():
        raise ValueError(f"{name} may not be synthetic or stub")
    return value


@dataclass(frozen=True)
class FailureAttempt:
    portfolio_cid: str
    candidate_cid: str
    plan_cid: str
    root_cid: str
    failure_class: FailureClass
    evidence_cids: tuple[str, ...]
    measure: tuple[int, ...]
    previous_receipt_cid: str = ""

    def __post_init__(self) -> None:
        for name in ("portfolio_cid", "candidate_cid", "plan_cid", "root_cid"):
            object.__setattr__(self, name, _cid(getattr(self, name), name))
        if not isinstance(self.failure_class, FailureClass):
            raise ValueError("failure_class must be closed")
        evidence = tuple(sorted({_cid(value, "evidence_cid") for value in self.evidence_cids}))
        if not evidence:
            raise ValueError("evidence_cids must be non-empty")
        object.__setattr__(self, "evidence_cids", evidence)
        measure = tuple(self.measure)
        if not measure or any(type(value) is not int or value < 0 for value in measure):
            raise ValueError("measure must be a non-empty tuple of non-negative integers")
        object.__setattr__(self, "measure", measure)
        if self.previous_receipt_cid:
            object.__setattr__(
                self,
                "previous_receipt_cid",
                _cid(self.previous_receipt_cid, "previous_receipt_cid"),
            )

    @property
    def attempt_key(self) -> str:
        return repair_evidence_cid(
            {
                "portfolio_cid": self.portfolio_cid,
                "candidate_cid": self.candidate_cid,
                "plan_cid": self.plan_cid,
                "root_cid": self.root_cid,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": DCR_FAILURE_MEMORY_SCHEMA,
            "attempt_key": self.attempt_key,
            "portfolio_cid": self.portfolio_cid,
            "candidate_cid": self.candidate_cid,
            "plan_cid": self.plan_cid,
            "root_cid": self.root_cid,
            "failure_class": self.failure_class.value,
            "evidence_cids": list(self.evidence_cids),
            "measure": list(self.measure),
            "previous_receipt_cid": self.previous_receipt_cid,
        }

    @property
    def content_id(self) -> str:
        return repair_evidence_cid(self.to_dict())


@dataclass(frozen=True)
class FailureMemoryReceipt:
    attempt: FailureAttempt
    receipt_cid: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.attempt, FailureAttempt)
            or self.receipt_cid != self.attempt.content_id
        ):
            raise ValueError("failure receipt must exactly bind its typed attempt")


@dataclass(frozen=True)
class ReplanMemoryDecision:
    disposition: ReplanMemoryDisposition
    reason_codes: tuple[str, ...]
    receipt_cid: str = ""
    execution_authorized: bool = False
    completion_authorized: bool = False
    model_call_count: int = 0
    provider_call_count: int = 0
    network_call_count: int = 0


def _history_error(history: Sequence[FailureMemoryReceipt]) -> str | None:
    previous = ""
    for receipt in history:
        if not isinstance(receipt, FailureMemoryReceipt):
            return "typed_failure_receipt_required"
        attempt = receipt.attempt
        if attempt.previous_receipt_cid != previous:
            return "forged_or_non_append_only_history"
        previous = receipt.receipt_cid
    return None


def decide_replan(
    portfolio: Any,
    plan_result: Any,
    current_roots: Any,
    attempt: Any,
    *,
    history: Sequence[FailureMemoryReceipt] = (),
) -> ReplanMemoryDecision:
    """Make one byte-reproducible retry/no-work decision without scheduling work."""
    if not isinstance(portfolio, CandidatePortfolio) or not isinstance(
        plan_result, RepairPlanDagResult
    ):
        return ReplanMemoryDecision(
            ReplanMemoryDisposition.REJECTED, ("typed_dcr062_dcr061_inputs_required",)
        )
    if not isinstance(current_roots, RepairAuthorityRoots) or not isinstance(
        attempt, FailureAttempt
    ):
        return ReplanMemoryDecision(
            ReplanMemoryDisposition.REJECTED, ("typed_roots_and_attempt_required",)
        )
    error = _history_error(tuple(history))
    if error:
        return ReplanMemoryDecision(ReplanMemoryDisposition.REJECTED, (error,))
    if (
        attempt.portfolio_cid != portfolio.portfolio_cid
        or attempt.plan_cid != plan_result.plan_cid
        or attempt.root_cid != current_roots.content_id
        or attempt.candidate_cid not in portfolio.candidate_cids
    ):
        return ReplanMemoryDecision(
            ReplanMemoryDisposition.REJECTED, ("stale_root_plan_or_portfolio_binding",)
        )
    if attempt.failure_class is FailureClass.PROOF:
        return ReplanMemoryDecision(
            ReplanMemoryDisposition.NO_WORK,
            ("refuted_candidate_never_replayed",),
        )
    previous_attempts = [
        item.attempt for item in history if item.attempt.attempt_key == attempt.attempt_key
    ]
    if previous_attempts:
        prior = previous_attempts[-1]
        if attempt.failure_class is FailureClass.PROOF or prior.failure_class is FailureClass.PROOF:
            return ReplanMemoryDecision(
                ReplanMemoryDisposition.NO_WORK, ("refuted_candidate_never_replayed",)
            )
        if set(attempt.evidence_cids) == set(prior.evidence_cids):
            return ReplanMemoryDecision(
                ReplanMemoryDisposition.NO_WORK, ("unchanged_evidence_replay",)
            )
        if not set(attempt.evidence_cids).difference(prior.evidence_cids):
            return ReplanMemoryDecision(
                ReplanMemoryDisposition.ABSTAINED, ("strictly_new_evidence_required",)
            )
        if not attempt.measure < prior.measure:
            return ReplanMemoryDecision(
                ReplanMemoryDisposition.ABSTAINED, ("well_founded_measure_not_decreased",)
            )
    if attempt.previous_receipt_cid != (history[-1].receipt_cid if history else ""):
        return ReplanMemoryDecision(
            ReplanMemoryDisposition.REJECTED, ("append_receipt_predecessor_mismatch",)
        )
    return ReplanMemoryDecision(
        ReplanMemoryDisposition.RETRY_PENDING,
        ("integration_pending_live_dcr062_evidence",),
        attempt.content_id,
    )


__all__ = [
    "DCR_FAILURE_MEMORY_SCHEMA",
    "FailureAttempt",
    "FailureClass",
    "FailureMemoryReceipt",
    "ReplanMemoryDecision",
    "ReplanMemoryDisposition",
    "decide_replan",
]
