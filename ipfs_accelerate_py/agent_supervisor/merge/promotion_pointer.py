"""Exclusive compare-and-swap promotion pointer.

The promoted checkpoint pointer is a distinct L3 authority.  Holding a
checkpoint, evaluation, or run lease does not imply this key.  Compare-and-
swap admits one new pointer only when:

* policy admission authorized CAS;
* the caller holds the current ``promotion-pointer`` lease fence;
* the expected current pointer still matches.

A stale CAS loses.  Restoring a prior pointer requires a new non-promote
decision; it is never a silent overwrite.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from ..control.promotion_admission import (
    PromotionAdmissionReceipt,
    admit_promotion,
)
from ..proof.formal_verification_contracts import content_identity
from ..runtime.learning_checkpoint import L3ResourceKind, StaleFenceError
from ..validation.promotion_comparison import PromotionDecision
from .campaign_leases import (
    CampaignLease,
    CampaignLeaseCoordinator,
    CampaignLeaseError,
    LeaseExpiredError,
)
from .checkout_lock import serialized_lock_update


PROMOTION_POINTER_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/promotion-pointer@1"
)
PROMOTION_POINTER_RECEIPT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/promotion-pointer-receipt@1"
)
PROMOTION_POINTER_DIRNAME: Final = "agent-promotion-pointers"
PROMOTION_POINTER_FILENAME: Final = "CURRENT.json"
PROMOTION_LEASE_RESOURCE: Final = L3ResourceKind.PROMOTION_POINTER


class PromotionPointerError(RuntimeError):
    """Malformed or unauthorized pointer mutation."""


class StalePromotionPointerError(PromotionPointerError):
    """Compare-and-swap lost because the expected pointer was stale."""


class PromotionPointerLeaseError(PromotionPointerError):
    """Pointer mutation attempted without the exclusive promotion lease."""


def _text(value: Any, name: str, *, required: bool = True) -> str:
    if value is None:
        text = ""
    elif not isinstance(value, str):
        raise PromotionPointerError(f"{name} must be a string")
    else:
        text = value.strip()
    if "\x00" in text:
        raise PromotionPointerError(f"{name} must not contain NUL")
    if required and not text:
        raise PromotionPointerError(f"{name} must be a non-empty string")
    return text


def _int(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PromotionPointerError(f"{name} must be an integer")
    if value < minimum:
        raise PromotionPointerError(f"{name} must be at least {minimum}")
    return value


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise PromotionPointerError(f"{name} must be a boolean")
    return value


def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(
                json.dumps(
                    payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
                ).encode("utf-8")
                + b"\n"
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


@dataclass(frozen=True)
class PromotionPointer:
    """Exactly-one current promoted checkpoint pointer."""

    checkpoint_id: str
    decision_receipt_id: str
    fence: int
    previous_checkpoint_id: str = ""
    schema: str = PROMOTION_POINTER_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "checkpoint_id", _text(self.checkpoint_id, "checkpoint_id")
        )
        object.__setattr__(
            self,
            "decision_receipt_id",
            _text(self.decision_receipt_id, "decision_receipt_id"),
        )
        object.__setattr__(self, "fence", _int(self.fence, "fence", minimum=0))
        object.__setattr__(
            self,
            "previous_checkpoint_id",
            _text(self.previous_checkpoint_id, "previous_checkpoint_id", required=False),
        )
        object.__setattr__(self, "schema", _text(self.schema, "schema"))
        if self.schema != PROMOTION_POINTER_SCHEMA:
            raise PromotionPointerError("unsupported promotion pointer schema")

    @property
    def pointer_id(self) -> str:
        return content_identity(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "decision_receipt_id": self.decision_receipt_id,
            "fence": self.fence,
            "previous_checkpoint_id": self.previous_checkpoint_id,
            "schema": self.schema,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PromotionPointer":
        if not isinstance(payload, Mapping):
            raise PromotionPointerError("promotion pointer must be an object")
        return cls(
            checkpoint_id=payload.get("checkpoint_id", ""),
            decision_receipt_id=payload.get("decision_receipt_id", ""),
            fence=payload.get("fence", 0),
            previous_checkpoint_id=payload.get("previous_checkpoint_id", ""),
            schema=payload.get("schema", PROMOTION_POINTER_SCHEMA),
        )


@dataclass(frozen=True)
class PromotionPointerReceipt:
    """Durable audit of one CAS attempt, including stale losses."""

    accepted: bool
    stale: bool
    decision: PromotionDecision
    admission_receipt_id: str
    expected_checkpoint_id: str
    observed_checkpoint_id: str
    pointer: PromotionPointer | None
    lease_id: str
    fence: int
    reason: str
    schema: str = PROMOTION_POINTER_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "accepted", _bool(self.accepted, "accepted"))
        object.__setattr__(self, "stale", _bool(self.stale, "stale"))
        decision = (
            self.decision
            if isinstance(self.decision, PromotionDecision)
            else PromotionDecision(str(self.decision))
        )
        object.__setattr__(self, "decision", decision)
        object.__setattr__(
            self,
            "admission_receipt_id",
            _text(self.admission_receipt_id, "admission_receipt_id"),
        )
        object.__setattr__(
            self,
            "expected_checkpoint_id",
            _text(self.expected_checkpoint_id, "expected_checkpoint_id", required=False),
        )
        object.__setattr__(
            self,
            "observed_checkpoint_id",
            _text(self.observed_checkpoint_id, "observed_checkpoint_id", required=False),
        )
        if self.pointer is not None and not isinstance(self.pointer, PromotionPointer):
            object.__setattr__(self, "pointer", PromotionPointer.from_dict(self.pointer))
        object.__setattr__(self, "lease_id", _text(self.lease_id, "lease_id"))
        object.__setattr__(self, "fence", _int(self.fence, "fence"))
        object.__setattr__(self, "reason", _text(self.reason, "reason"))
        object.__setattr__(self, "schema", _text(self.schema, "schema"))
        if self.schema != PROMOTION_POINTER_RECEIPT_SCHEMA:
            raise PromotionPointerError("unsupported promotion pointer receipt schema")
        if self.accepted and (self.stale or self.pointer is None):
            raise PromotionPointerError("accepted CAS must publish a non-stale pointer")
        if self.stale and self.accepted:
            raise PromotionPointerError("stale CAS cannot be accepted")

    @property
    def receipt_id(self) -> str:
        return content_identity(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "admission_receipt_id": self.admission_receipt_id,
            "decision": self.decision.value,
            "expected_checkpoint_id": self.expected_checkpoint_id,
            "fence": self.fence,
            "lease_id": self.lease_id,
            "observed_checkpoint_id": self.observed_checkpoint_id,
            "pointer": None if self.pointer is None else self.pointer.to_dict(),
            "reason": self.reason,
            "schema": self.schema,
            "stale": self.stale,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PromotionPointerReceipt":
        if not isinstance(payload, Mapping):
            raise PromotionPointerError("promotion pointer receipt must be an object")
        claimed = payload.get("receipt_id")
        result = cls(
            accepted=payload.get("accepted", False),
            stale=payload.get("stale", False),
            decision=payload.get("decision", PromotionDecision.REJECT),
            admission_receipt_id=payload.get("admission_receipt_id", ""),
            expected_checkpoint_id=payload.get("expected_checkpoint_id", ""),
            observed_checkpoint_id=payload.get("observed_checkpoint_id", ""),
            pointer=payload.get("pointer"),
            lease_id=payload.get("lease_id", ""),
            fence=payload.get("fence", 0),
            reason=payload.get("reason", ""),
            schema=payload.get("schema", PROMOTION_POINTER_RECEIPT_SCHEMA),
        )
        if claimed is not None and claimed != result.receipt_id:
            raise PromotionPointerError("forged promotion pointer receipt_id")
        return result


class PromotionPointerStore:
    """Serialized exclusive-key store for the current promoted pointer."""

    def __init__(
        self,
        root: Path | str,
        *,
        coordinator: CampaignLeaseCoordinator | None = None,
        resource_id: str = "",
    ) -> None:
        self.root = Path(root)
        self.resource_id = str(resource_id or "").strip()
        self.coordinator = coordinator or CampaignLeaseCoordinator(self.root)
        self.pointer_path = (
            self.root / PROMOTION_POINTER_DIRNAME / PROMOTION_POINTER_FILENAME
        )
        self.pointer_path.parent.mkdir(parents=True, exist_ok=True)

    def acquire_lease(self, owner_id: str) -> CampaignLease:
        return self.coordinator.acquire(
            PROMOTION_LEASE_RESOURCE,
            owner_id=owner_id,
            resource_id=self.resource_id,
        )

    def current(self) -> PromotionPointer | None:
        try:
            payload = json.loads(self.pointer_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise PromotionPointerError("promotion pointer record is malformed") from exc
        if not isinstance(payload, Mapping):
            raise PromotionPointerError("promotion pointer record must be an object")
        return PromotionPointer.from_dict(payload)

    def _require_lease(self, lease: CampaignLease, expected_fence: int) -> CampaignLease:
        if lease.resource is not PROMOTION_LEASE_RESOURCE:
            raise PromotionPointerLeaseError(
                "promotion pointer requires the exclusive promotion-pointer lease"
            )
        if lease.resource_id != self.resource_id:
            raise PromotionPointerLeaseError("promotion lease resource_id mismatch")
        try:
            return self.coordinator.assert_write_fence(lease, expected_fence)
        except (StaleFenceError, CampaignLeaseError, LeaseExpiredError) as exc:
            raise PromotionPointerLeaseError(str(exc)) from exc

    def _receipt(
        self,
        *,
        accepted: bool,
        stale: bool,
        admission: PromotionAdmissionReceipt,
        expected: str,
        observed: str,
        pointer: PromotionPointer | None,
        lease: CampaignLease,
        reason: str,
        decision: PromotionDecision | None = None,
    ) -> PromotionPointerReceipt:
        return PromotionPointerReceipt(
            accepted=accepted,
            stale=stale,
            decision=decision or admission.decision,
            admission_receipt_id=admission.receipt_id,
            expected_checkpoint_id=expected,
            observed_checkpoint_id=observed,
            pointer=pointer,
            lease_id=lease.lease_id,
            fence=lease.fence,
            reason=reason,
        )

    def compare_and_swap(
        self,
        *,
        admission: PromotionAdmissionReceipt,
        lease: CampaignLease,
        expected_fence: int,
        raise_on_stale: bool = False,
    ) -> PromotionPointerReceipt:
        """Publish the admitted candidate or lose if the expected pointer is stale."""

        live = self._require_lease(lease, expected_fence)
        if not admission.cas_authorized or not admission.admitted:
            return self._receipt(
                accepted=False,
                stale=False,
                admission=admission,
                expected=admission.expected_current_pointer,
                observed=self.current().checkpoint_id if self.current() is not None else "",
                pointer=self.current(),
                lease=live,
                reason="cas_not_authorized",
            )
        if admission.decision is not PromotionDecision.PROMOTE:
            return self._receipt(
                accepted=False,
                stale=False,
                admission=admission,
                expected=admission.expected_current_pointer,
                observed=self.current().checkpoint_id if self.current() is not None else "",
                pointer=self.current(),
                lease=live,
                reason="decision_is_not_promote",
            )
        with serialized_lock_update(self.pointer_path):
            current = self.current()
            observed = current.checkpoint_id if current is not None else ""
            expected = admission.expected_current_pointer
            expected_fence_value = current.fence if current is not None else -1
            if observed != expected:
                receipt = self._receipt(
                    accepted=False,
                    stale=True,
                    admission=admission,
                    expected=expected,
                    observed=observed,
                    pointer=current,
                    lease=live,
                    reason=(
                        "stale_compare_and_swap "
                        f"(expected {expected!r}, observed {observed!r}/{expected_fence_value})"
                    ),
                )
                if raise_on_stale:
                    raise StalePromotionPointerError(receipt.reason)
                return receipt
            nxt = PromotionPointer(
                checkpoint_id=admission.candidate_checkpoint_id,
                decision_receipt_id=admission.receipt_id,
                fence=(current.fence + 1) if current is not None else 0,
                previous_checkpoint_id=observed,
            )
            _atomic_write(self.pointer_path, nxt.to_dict())
            return self._receipt(
                accepted=True,
                stale=False,
                admission=admission,
                expected=expected,
                observed=observed,
                pointer=nxt,
                lease=live,
                reason="promoted_pointer_swapped",
            )

    def restore_prior(
        self,
        *,
        admission: PromotionAdmissionReceipt,
        lease: CampaignLease,
        expected_fence: int,
        prior_checkpoint_id: str,
        raise_on_stale: bool = False,
    ) -> PromotionPointerReceipt:
        """CAS-restore the prior pointer only with a new non-promote decision."""

        live = self._require_lease(lease, expected_fence)
        prior = _text(prior_checkpoint_id, "prior_checkpoint_id")
        if admission.decision is PromotionDecision.PROMOTE or admission.admitted:
            raise PromotionPointerError(
                "rollback requires a new non-promote decision"
            )
        if admission.candidate_checkpoint_id != prior:
            raise PromotionPointerError(
                "rollback decision must name the prior pointer as its candidate"
            )
        with serialized_lock_update(self.pointer_path):
            current = self.current()
            observed = current.checkpoint_id if current is not None else ""
            expected = admission.expected_current_pointer
            if current is None or observed != expected:
                receipt = self._receipt(
                    accepted=False,
                    stale=True,
                    admission=admission,
                    expected=expected,
                    observed=observed,
                    pointer=current,
                    lease=live,
                    reason=(
                        "stale_rollback_compare_and_swap "
                        f"(expected {expected!r}, observed {observed!r})"
                    ),
                )
                if raise_on_stale:
                    raise StalePromotionPointerError(receipt.reason)
                return receipt
            nxt = PromotionPointer(
                checkpoint_id=prior,
                decision_receipt_id=admission.receipt_id,
                fence=current.fence + 1,
                previous_checkpoint_id=observed,
            )
            _atomic_write(self.pointer_path, nxt.to_dict())
            return self._receipt(
                accepted=True,
                stale=False,
                admission=admission,
                expected=expected,
                observed=observed,
                pointer=nxt,
                lease=live,
                reason="prior_pointer_restored",
                decision=admission.decision,
            )


def execute_admitted_promotion(
    *,
    store: PromotionPointerStore,
    admission: PromotionAdmissionReceipt | Mapping[str, Any],
    lease: CampaignLease,
    expected_fence: int,
    raise_on_stale: bool = False,
) -> PromotionPointerReceipt:
    """Apply an already-computed admission receipt to the exclusive pointer."""

    if isinstance(admission, PromotionAdmissionReceipt):
        receipt = admission
    elif isinstance(admission, Mapping) and "m3_results" in admission:
        receipt = PromotionAdmissionReceipt.from_dict(admission)
    else:
        receipt = admit_promotion(admission)
    return store.compare_and_swap(
        admission=receipt,
        lease=lease,
        expected_fence=expected_fence,
        raise_on_stale=raise_on_stale,
    )


__all__ = (
    "PROMOTION_LEASE_RESOURCE",
    "PROMOTION_POINTER_DIRNAME",
    "PROMOTION_POINTER_FILENAME",
    "PROMOTION_POINTER_RECEIPT_SCHEMA",
    "PROMOTION_POINTER_SCHEMA",
    "PromotionPointer",
    "PromotionPointerError",
    "PromotionPointerLeaseError",
    "PromotionPointerReceipt",
    "PromotionPointerStore",
    "StalePromotionPointerError",
    "execute_admitted_promotion",
)
