"""Provider gate for deterministic repair.

The DCR-080 daemon composition is intentionally not a provider router.  This
module exists at the control boundary so callers have one auditable answer:
model-provider dispatch is never authorized by deterministic repair.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from .pre_implementation_kernel import (
    PRE_IMPLEMENTATION_KERNEL_INTERFACE,
    PreImplementationKernelDecision,
    evaluate_pre_implementation_gate,
)


PRE_IMPLEMENTATION_PROVIDER_GATE_INTERFACE: Final[str] = (
    "DeterministicRepairCompositionRoot@pre_implementation_kernel"
)
EVENT_PRE_IMPLEMENTATION_KERNEL: Final[str] = "pre_implementation_kernel_evaluated"


@dataclass(frozen=True)
class ProviderGateDecision:
    """An explicit, non-bypassable denial of provider dispatch."""

    kernel: PreImplementationKernelDecision
    provider_authorized: bool = False
    skip_provider: bool = True
    provider_hook_count: int = 0

    @property
    def receipt_cid(self) -> str:
        return self.kernel.receipt_cid

    @property
    def disposition(self) -> str:
        return self.kernel.disposition

    def to_event_payload(self, *, task_id: str, attempt: int) -> dict[str, object]:
        return {
            "event": EVENT_PRE_IMPLEMENTATION_KERNEL,
            "interface": PRE_IMPLEMENTATION_PROVIDER_GATE_INTERFACE,
            "kernel_interface": PRE_IMPLEMENTATION_KERNEL_INTERFACE,
            "task_id": str(task_id),
            "attempt": int(attempt),
            "disposition": self.disposition,
            "provider_authorized": False,
            "skip_provider": True,
            "provider_hook_count": 0,
            "receipt_cid": self.receipt_cid,
            "kernel_receipt": self.kernel.receipt.to_dict(),
        }


def evaluate_provider_gate(
    *, task_id: str, service_receipt_ids: tuple[str, ...] = (), reason_codes: tuple[str, ...] = ()
) -> ProviderGateDecision:
    """Return a sealed denial; no parameter can enable provider dispatch."""

    return ProviderGateDecision(
        kernel=evaluate_pre_implementation_gate(
            task_id=task_id,
            service_receipt_ids=service_receipt_ids,
            reason_codes=reason_codes,
        )
    )


def assert_provider_dispatch_allowed(decision: ProviderGateDecision) -> None:
    """Always fail: a deterministic repair receipt is not provider authority."""

    del decision
    raise PermissionError("provider dispatch is forbidden for deterministic repair")


__all__ = [
    "EVENT_PRE_IMPLEMENTATION_KERNEL",
    "PRE_IMPLEMENTATION_PROVIDER_GATE_INTERFACE",
    "ProviderGateDecision",
    "assert_provider_dispatch_allowed",
    "evaluate_provider_gate",
]
