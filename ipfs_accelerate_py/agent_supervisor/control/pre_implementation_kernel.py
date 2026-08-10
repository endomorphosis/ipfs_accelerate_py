"""Fail-closed admission kernel for deterministic repair execution.

This control-plane kernel deliberately has no model-provider disposition.  It
only records whether the composed deterministic services produced receipts
that are usable for a repair attempt.  A missing or malformed receipt is an
abstention, never an invitation to retry through a model provider.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Final, Mapping


PRE_IMPLEMENTATION_KERNEL_INTERFACE: Final[str] = "PreImplementationKernel@1"
PRE_IMPLEMENTATION_KERNEL_VERSION: Final[int] = 1


def _content_id(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(dict(payload), sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class PreImplementationKernelReceipt:
    """Body-free observation of deterministic-service admission."""

    task_id: str
    disposition: str
    reason_codes: tuple[str, ...]
    service_receipt_ids: tuple[str, ...] = ()

    @property
    def content_id(self) -> str:
        return _content_id(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "interface": PRE_IMPLEMENTATION_KERNEL_INTERFACE,
            "version": PRE_IMPLEMENTATION_KERNEL_VERSION,
            "task_id": self.task_id,
            "disposition": self.disposition,
            "reason_codes": list(self.reason_codes),
            "service_receipt_ids": list(self.service_receipt_ids),
        }


@dataclass(frozen=True)
class PreImplementationKernelDecision:
    """Admission result.  ``provider_authorized`` is permanently false."""

    receipt: PreImplementationKernelReceipt
    provider_authorized: bool = False
    skip_provider: bool = True
    provider_hook_count: int = 0

    @property
    def receipt_cid(self) -> str:
        return self.receipt.content_id

    @property
    def disposition(self) -> str:
        return self.receipt.disposition


def evaluate_pre_implementation_gate(
    *, task_id: str, service_receipt_ids: tuple[str, ...] = (), reason_codes: tuple[str, ...] = ()
) -> PreImplementationKernelDecision:
    """Seal deterministic admission without any residual/provider escape hatch."""

    receipts = tuple(str(item).strip() for item in service_receipt_ids if str(item).strip())
    reasons = tuple(str(item).strip() for item in reason_codes if str(item).strip())
    if not receipts:
        reasons = reasons or ("missing_service_receipt",)
        disposition = "abstain"
    else:
        disposition = "deterministic_admitted"
    return PreImplementationKernelDecision(
        receipt=PreImplementationKernelReceipt(
            task_id=str(task_id),
            disposition=disposition,
            reason_codes=reasons,
            service_receipt_ids=receipts,
        )
    )


__all__ = [
    "PRE_IMPLEMENTATION_KERNEL_INTERFACE",
    "PreImplementationKernelDecision",
    "PreImplementationKernelReceipt",
    "evaluate_pre_implementation_gate",
]
