# ruff: noqa: UP042 - the package retains Python 3.8 compatibility
"""Externally authorized policy-pointer promotion and exact rollback.

``AutonomyPromotionController@1`` is the sole APMC policy-pointer CAS owner.
It evaluates non-compensable gates, refuses self-authorization, and either
performs an expected-old compare-and-swap or returns a non-promotion receipt
that names every missed gate.  It cannot lower thresholds.

Cold import performs no filesystem, network, or provider action.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass, field
from threading import Lock
from types import MappingProxyType
from typing import Any, Final

from ..proof.formal_verification_contracts import (
    CONTRACT_VERSION,
    canonical_json_bytes,
    content_identity,
)
from .contracts import (
    MAX_IDENTIFIER_BYTES,
    AutonomyContractError,
    AutonomyPromotionReceipt,
    PromotionStatus,
)

AUTONOMY_PROMOTION_CONTROLLER_INTERFACE: Final[str] = (
    "AutonomyPromotionController@1"
)
AUTONOMY_PROMOTION_RECEIPT_INTERFACE: Final[str] = "AutonomyPromotionReceipt@1"
AUTONOMY_PROMOTION_CONTROLLER_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/autonomy/promotion-controller@1"
)
POLICY_POINTER_CAS_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/autonomy/policy-pointer-cas@1"
)
REQUIRED_SAFETY_GATES: Final[tuple[str, ...]] = (
    "false_completions",
    "unauthorized_mutations",
    "simulated_as_live",
    "stale_authoritative_cache_hits",
    "confirmation_replays",
    "path_or_scope_escapes",
    "hidden_validation_reductions",
    "escaped_critical_seeded_defects",
    "self_authorized_policy_promotions",
)
THRESHOLD_GATES: Final[tuple[str, ...]] = (
    "token_input_reduction_bps",
    "model_call_reduction_bps",
    "retry_input_reduction_bps",
    "distilled_class_coverage_bps",
    "low_risk_without_large_model_bps",
    "human_intervention_reduction_bps",
    "deterministic_question_resolution_bps",
    "held_out_decision_accuracy_bps",
)
REQUIRED_THRESHOLD_BPS: Final[Mapping[str, int]] = MappingProxyType(
    {
        "token_input_reduction_bps": 3_000,
        "model_call_reduction_bps": 2_500,
        "retry_input_reduction_bps": 4_000,
        "distilled_class_coverage_bps": 2_000,
        "low_risk_without_large_model_bps": 5_000,
        "human_intervention_reduction_bps": 3_000,
        "deterministic_question_resolution_bps": 8_000,
        "held_out_decision_accuracy_bps": 9_000,
    }
)
MAX_INTEGER: Final[int] = (1 << 63) - 1


class PromotionControllerError(AutonomyContractError):
    """Raised when promotion inputs themselves are malformed."""


@dataclass
class PolicyPointerStore:
    """Process-local expected-old policy pointer with generation fencing."""

    current_policy_id: str
    generation: int = 0
    _lock: Lock = field(default_factory=Lock, repr=False)

    def compare_and_swap(
        self,
        *,
        expected_old: str,
        candidate: str,
        observed_generation: int,
    ) -> Mapping[str, Any]:
        expected = _identifier(expected_old, "expected_old")
        next_policy = _identifier(candidate, "candidate")
        with self._lock:
            if observed_generation != self.generation:
                return MappingProxyType(
                    {
                        "schema": POLICY_POINTER_CAS_SCHEMA,
                        "applied": False,
                        "reason": "cas_generation_mismatch",
                        "current_policy_id": self.current_policy_id,
                        "generation": self.generation,
                    }
                )
            if self.current_policy_id != expected:
                return MappingProxyType(
                    {
                        "schema": POLICY_POINTER_CAS_SCHEMA,
                        "applied": False,
                        "reason": "cas_expected_old_mismatch",
                        "current_policy_id": self.current_policy_id,
                        "generation": self.generation,
                    }
                )
            previous = self.current_policy_id
            self.current_policy_id = next_policy
            self.generation += 1
            body = {
                "schema": POLICY_POINTER_CAS_SCHEMA,
                "applied": True,
                "expected_old_policy_id": previous,
                "resulting_policy_id": self.current_policy_id,
                "generation": self.generation,
            }
            body["cas_receipt_id"] = content_identity(body)
            return MappingProxyType(body)

    def rollback(self, *, rollback_policy_id: str, observed_generation: int) -> Mapping[str, Any]:
        return self.compare_and_swap(
            expected_old=self.current_policy_id,
            candidate=rollback_policy_id,
            observed_generation=observed_generation,
        )


def _identifier(value: Any, name: str, *, required: bool = True) -> str:
    text = str(value or "").strip()
    if required and not text:
        raise PromotionControllerError(f"{name} is required")
    if text and (
        len(text.encode("utf-8")) > MAX_IDENTIFIER_BYTES
        or any(char.isspace() for char in text)
        or "\x00" in text
    ):
        raise PromotionControllerError(f"{name} must be a compact identifier")
    return text


def _int(value: Any, name: str) -> int:
    if type(value) is not int or isinstance(value, bool):
        raise PromotionControllerError(f"{name} must be an integer")
    if value < 0 or value > MAX_INTEGER:
        raise PromotionControllerError(f"{name} is out of bounds")
    return value


def _bool_map(value: Any, name: str, required_keys: tuple[str, ...]) -> Mapping[str, bool]:
    if not isinstance(value, Mapping):
        raise PromotionControllerError(f"{name} must be a mapping")
    if set(value) != set(required_keys):
        raise PromotionControllerError(f"{name} must report every required gate")
    return MappingProxyType(
        {key: bool(value[key]) for key in required_keys}
    )


@dataclass(frozen=True)
class PromotionRequest:
    """Exact promotion attempt.  The candidate cannot authorize itself."""

    candidate_policy_id: str
    expected_old_policy_id: str
    authorization_id: str
    safety_gate_results: Mapping[str, bool]
    safety_gate_receipt_ids: tuple[str, ...]
    held_out_evaluation_ids: tuple[str, ...]
    threshold_bps: Mapping[str, int]
    tree_id: str
    candidate_version: str
    authorization_subject: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "candidate_policy_id", _identifier(self.candidate_policy_id, "candidate_policy_id")
        )
        object.__setattr__(
            self,
            "expected_old_policy_id",
            _identifier(self.expected_old_policy_id, "expected_old_policy_id"),
        )
        object.__setattr__(
            self, "authorization_id", _identifier(self.authorization_id, "authorization_id")
        )
        object.__setattr__(
            self,
            "safety_gate_results",
            _bool_map(self.safety_gate_results, "safety_gate_results", REQUIRED_SAFETY_GATES),
        )
        object.__setattr__(
            self,
            "safety_gate_receipt_ids",
            tuple(_identifier(item, "safety_gate_receipt_ids") for item in self.safety_gate_receipt_ids),
        )
        object.__setattr__(
            self,
            "held_out_evaluation_ids",
            tuple(_identifier(item, "held_out_evaluation_ids") for item in self.held_out_evaluation_ids),
        )
        if not self.safety_gate_receipt_ids or not self.held_out_evaluation_ids:
            raise PromotionControllerError("promotion requires safety and held-out evidence")
        raw = dict(self.threshold_bps)
        if set(raw) != set(THRESHOLD_GATES):
            raise PromotionControllerError("threshold_bps must report every efficiency gate")
        object.__setattr__(
            self,
            "threshold_bps",
            MappingProxyType({key: _int(raw[key], key) for key in THRESHOLD_GATES}),
        )
        object.__setattr__(self, "tree_id", _identifier(self.tree_id, "tree_id"))
        object.__setattr__(
            self, "candidate_version", _identifier(self.candidate_version, "candidate_version")
        )
        subject = _identifier(
            self.authorization_subject or self.authorization_id,
            "authorization_subject",
        )
        object.__setattr__(self, "authorization_subject", subject)


class AutonomyPromotionController:
    """Evaluate gates and CAS the policy pointer, or name exact blockers."""

    INTERFACE: Final[str] = AUTONOMY_PROMOTION_CONTROLLER_INTERFACE

    def __init__(self, store: PolicyPointerStore) -> None:
        if not isinstance(store, PolicyPointerStore):
            raise TypeError("store must be a PolicyPointerStore")
        self._store = store

    @property
    def interface(self) -> str:
        return self.INTERFACE

    @property
    def store(self) -> PolicyPointerStore:
        return self._store

    def evaluate(self, request: PromotionRequest) -> tuple[tuple[str, ...], bool]:
        blockers: list[str] = []
        if (
            request.authorization_id == request.candidate_policy_id
            or request.authorization_subject == request.candidate_policy_id
        ):
            blockers.append("self_authorized_policy_promotions")
        for gate, passed in request.safety_gate_results.items():
            if not passed:
                blockers.append(gate)
        for gate, required in REQUIRED_THRESHOLD_BPS.items():
            if int(request.threshold_bps[gate]) < required:
                blockers.append(gate)
        unique = tuple(dict.fromkeys(blockers))
        return unique, not unique

    def apply(self, request: PromotionRequest) -> AutonomyPromotionReceipt:
        blockers, eligible = self.evaluate(request)
        if not eligible:
            return AutonomyPromotionReceipt(
                candidate_policy_id=request.candidate_policy_id,
                expected_old_policy_id=request.expected_old_policy_id,
                resulting_policy_id=self._store.current_policy_id,
                status=PromotionStatus.NON_PROMOTED,
                safety_gate_results=request.safety_gate_results,
                held_out_evaluation_ids=request.held_out_evaluation_ids,
                safety_gate_receipt_ids=request.safety_gate_receipt_ids,
                authorization_id=request.authorization_id,
                compare_and_swap_receipt_id="",
                rollback_policy_id=request.expected_old_policy_id,
                blocker_codes=blockers,
                self_authorized=False,
            )
        cas = self._store.compare_and_swap(
            expected_old=request.expected_old_policy_id,
            candidate=request.candidate_policy_id,
            observed_generation=self._store.generation,
        )
        if cas.get("applied") is not True:
            return AutonomyPromotionReceipt(
                candidate_policy_id=request.candidate_policy_id,
                expected_old_policy_id=request.expected_old_policy_id,
                resulting_policy_id=str(cas.get("current_policy_id") or self._store.current_policy_id),
                status=PromotionStatus.NON_PROMOTED,
                safety_gate_results=request.safety_gate_results,
                held_out_evaluation_ids=request.held_out_evaluation_ids,
                safety_gate_receipt_ids=request.safety_gate_receipt_ids,
                authorization_id=request.authorization_id,
                compare_and_swap_receipt_id="",
                rollback_policy_id=request.expected_old_policy_id,
                blocker_codes=(str(cas.get("reason") or "cas_failed"),),
                self_authorized=False,
            )
        return AutonomyPromotionReceipt(
            candidate_policy_id=request.candidate_policy_id,
            expected_old_policy_id=request.expected_old_policy_id,
            resulting_policy_id=str(cas["resulting_policy_id"]),
            status=PromotionStatus.PROMOTED,
            safety_gate_results=request.safety_gate_results,
            held_out_evaluation_ids=request.held_out_evaluation_ids,
            safety_gate_receipt_ids=request.safety_gate_receipt_ids,
            authorization_id=request.authorization_id,
            compare_and_swap_receipt_id=str(cas["cas_receipt_id"]),
            rollback_policy_id=request.expected_old_policy_id,
            blocker_codes=(),
            self_authorized=False,
        )

    def rollback(
        self,
        receipt: AutonomyPromotionReceipt,
        *,
        authorization_id: str,
    ) -> AutonomyPromotionReceipt:
        auth = _identifier(authorization_id, "authorization_id")
        if receipt.status is not PromotionStatus.PROMOTED:
            raise PromotionControllerError("only a promoted pointer can be rolled back")
        if auth == receipt.candidate_policy_id:
            raise PromotionControllerError("candidate cannot authorize rollback")
        cas = self._store.rollback(
            rollback_policy_id=receipt.rollback_policy_id,
            observed_generation=self._store.generation,
        )
        if cas.get("applied") is not True:
            return AutonomyPromotionReceipt(
                candidate_policy_id=receipt.candidate_policy_id,
                expected_old_policy_id=receipt.expected_old_policy_id,
                resulting_policy_id=str(cas.get("current_policy_id") or self._store.current_policy_id),
                status=PromotionStatus.PROMOTED,
                safety_gate_results=receipt.safety_gate_results,
                held_out_evaluation_ids=receipt.held_out_evaluation_ids,
                safety_gate_receipt_ids=receipt.safety_gate_receipt_ids,
                authorization_id=auth,
                compare_and_swap_receipt_id=receipt.compare_and_swap_receipt_id,
                rollback_policy_id=receipt.rollback_policy_id,
                blocker_codes=(str(cas.get("reason") or "rollback_cas_failed"),),
                self_authorized=False,
            )
        return AutonomyPromotionReceipt(
            candidate_policy_id=receipt.candidate_policy_id,
            expected_old_policy_id=receipt.expected_old_policy_id,
            resulting_policy_id=str(cas["resulting_policy_id"]),
            status=PromotionStatus.ROLLED_BACK,
            safety_gate_results=receipt.safety_gate_results,
            held_out_evaluation_ids=receipt.held_out_evaluation_ids,
            safety_gate_receipt_ids=receipt.safety_gate_receipt_ids,
            authorization_id=auth,
            compare_and_swap_receipt_id=str(cas["cas_receipt_id"]),
            rollback_policy_id=receipt.rollback_policy_id,
            blocker_codes=(),
            self_authorized=False,
        )


__all__ = (
    "AUTONOMY_PROMOTION_CONTROLLER_INTERFACE",
    "AUTONOMY_PROMOTION_CONTROLLER_SCHEMA",
    "AUTONOMY_PROMOTION_RECEIPT_INTERFACE",
    "POLICY_POINTER_CAS_SCHEMA",
    "REQUIRED_SAFETY_GATES",
    "REQUIRED_THRESHOLD_BPS",
    "THRESHOLD_GATES",
    "AutonomyPromotionController",
    "PolicyPointerStore",
    "PromotionControllerError",
    "PromotionRequest",
)
