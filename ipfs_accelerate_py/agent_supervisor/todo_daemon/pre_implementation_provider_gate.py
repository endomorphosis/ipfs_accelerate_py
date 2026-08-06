"""Pre-implementation provider gate for ImplementationDaemon (WPD-021).

Interface: ``ImplementationDaemon@pre_implementation_kernel``

Workers must not dispatch a model provider unless the pre-implementation
kernel sealed ``residual_llm_authorized`` with a residual packet CID.
``closed_deterministic``, ``abstain_review``, and ``defer_capability`` keep
the provider path unreachable.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Final

from .implementation_disposition import (
    ImplementationDisposition,
    ImplementationForestRoots,
    PreImplementationKernelReceipt,
    implementation_disposition_cid,
    provider_invocation_authorized,
)
from .pre_implementation_kernel import (
    AnalyticalRepairCandidate,
    KernelEvaluationRequest,
    KernelEvaluationResult,
    PreImplementationKernel,
    build_pre_implementation_kernel,
)


PRE_IMPLEMENTATION_PROVIDER_GATE_INTERFACE: Final[str] = (
    "ImplementationDaemon@pre_implementation_kernel"
)
PRE_IMPLEMENTATION_PROVIDER_GATE_VERSION: Final[int] = 1
PRE_IMPLEMENTATION_PROVIDER_GATE_EVIDENCE: Final[str] = (
    "wpd/pre-implementation-provider-gate@1"
)

EVENT_PRE_IMPLEMENTATION_KERNEL: Final[str] = "pre_implementation_kernel_evaluated"


@dataclass(frozen=True)
class ProviderGateDecision:
    """Decision binding a kernel receipt to provider authorization."""

    disposition: ImplementationDisposition
    provider_authorized: bool
    provider_hook_count: int
    skip_provider: bool
    reason_code: str
    receipt: PreImplementationKernelReceipt
    residual_packet_cid: str = ""
    analytical_candidate_count: int = 0

    @property
    def receipt_cid(self) -> str:
        return self.receipt.content_id

    @property
    def closed_deterministic(self) -> bool:
        return self.disposition is ImplementationDisposition.CLOSED_DETERMINISTIC

    def to_event_payload(self, *, task_id: str, attempt: int) -> dict[str, Any]:
        return {
            "event": EVENT_PRE_IMPLEMENTATION_KERNEL,
            "task_id": task_id,
            "attempt": int(attempt),
            "disposition": self.disposition.value,
            "provider_authorized": self.provider_authorized,
            "provider_hook_count": self.provider_hook_count,
            "skip_provider": self.skip_provider,
            "reason_code": self.reason_code,
            "receipt_cid": self.receipt_cid,
            "residual_packet_cid": self.residual_packet_cid,
            "analytical_candidate_count": self.analytical_candidate_count,
            "kernel_receipt": self.receipt.to_dict(),
            "interface": PRE_IMPLEMENTATION_PROVIDER_GATE_INTERFACE,
        }


def build_forest_roots_from_identity(
    *,
    repository_id: str,
    repository_forest_cid: str,
    git_tree_id: str,
    policy_root: str = "",
    dirty_overlay_cid: str = "",
) -> ImplementationForestRoots:
    """Construct forest roots from opaque identity strings."""

    return ImplementationForestRoots(
        repository_id=repository_id,
        repository_forest_cid=repository_forest_cid,
        git_tree_id=git_tree_id,
        policy_root=policy_root or repository_forest_cid,
        dirty_overlay_cid=dirty_overlay_cid,
    )


def evaluate_provider_gate(
    *,
    task_cid: str,
    forest_roots: ImplementationForestRoots,
    attempt: int = 1,
    residual_packet_cid: str = "",
    analytical_candidates: tuple[AnalyticalRepairCandidate, ...] = (),
    kernel: PreImplementationKernel | None = None,
    planner_available: bool = True,
    doctor_available: bool = True,
    allow_legacy_residual: bool = True,
    policy_revision: str = "1",
) -> ProviderGateDecision:
    """Evaluate the pre-implementation kernel and return a provider gate.

    When no analytical candidates exist and no residual packet is supplied,
    ``allow_legacy_residual`` seals a synthetic residual packet identity so
    existing model-assisted workers remain reachable until WPD-023 tightens
    residual packet construction.  The gate still requires
    ``residual_llm_authorized`` for any provider call.
    """

    packet = str(residual_packet_cid or "").strip()
    candidates = tuple(analytical_candidates or ())
    if (
        allow_legacy_residual
        and not packet
        and not candidates
        and planner_available
        and doctor_available
    ):
        packet = implementation_disposition_cid(
            {
                "kind": "legacy_worker_prompt_residual",
                "task_cid": task_cid,
                "attempt": int(attempt),
            }
        )

    active_kernel = kernel or build_pre_implementation_kernel(
        planner_available=planner_available,
        doctor_available=doctor_available,
    )
    # When a kernel instance is injected, leave request capability flags unset
    # so the kernel's own planner/doctor availability controls deferral.
    request = KernelEvaluationRequest(
        task_cid=task_cid,
        forest_roots=forest_roots,
        attempt=int(attempt),
        residual_packet_cid=packet,
        analytical_candidates=candidates,
        policy_revision=policy_revision,
        planner_available=None if kernel is not None else planner_available,
        doctor_available=None if kernel is not None else doctor_available,
    )
    result: KernelEvaluationResult = active_kernel.evaluate(request)
    disposition = result.disposition
    authorized = provider_invocation_authorized(disposition)
    # Residual path requires a sealed packet on the receipt.
    if authorized and not result.receipt.residual_packet_cid:
        authorized = False
        skip = True
        reason = "residual_packet_required"
    else:
        skip = not authorized
        reason = result.reason_code

    return ProviderGateDecision(
        disposition=disposition,
        provider_authorized=authorized,
        provider_hook_count=int(result.provider_hook_count),
        skip_provider=skip,
        reason_code=reason,
        receipt=result.receipt,
        residual_packet_cid=result.receipt.residual_packet_cid,
        analytical_candidate_count=int(result.analytical_candidate_count),
    )


def assert_provider_dispatch_allowed(decision: ProviderGateDecision) -> None:
    """Fail closed when a caller attempts provider dispatch illegally."""

    if not decision.provider_authorized or decision.skip_provider:
        raise PermissionError(
            "provider dispatch blocked by pre-implementation kernel: "
            f"disposition={decision.disposition.value} "
            f"reason={decision.reason_code}"
        )
    if not decision.residual_packet_cid:
        raise PermissionError(
            "provider dispatch requires residual_packet_cid on residual_llm_authorized"
        )


__all__ = [
    "EVENT_PRE_IMPLEMENTATION_KERNEL",
    "PRE_IMPLEMENTATION_PROVIDER_GATE_EVIDENCE",
    "PRE_IMPLEMENTATION_PROVIDER_GATE_INTERFACE",
    "PRE_IMPLEMENTATION_PROVIDER_GATE_VERSION",
    "ProviderGateDecision",
    "assert_provider_dispatch_allowed",
    "build_forest_roots_from_identity",
    "evaluate_provider_gate",
]
