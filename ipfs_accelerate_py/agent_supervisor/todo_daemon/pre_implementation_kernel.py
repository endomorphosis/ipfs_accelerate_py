"""Pre-implementation kernel evaluation for workers (WPD-020).

Interface: ``PreImplementationKernel@1``

Workers call :meth:`PreImplementationKernel.evaluate` **before** any model
provider invocation.  The kernel returns a sealed
:class:`~ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_disposition.PreImplementationKernelReceipt`
with a closed :class:`ImplementationDisposition` that gates residual LLM use.

Order of evaluation (deterministic-first):

1. Capability probe — missing required planner/doctor backends →
   ``defer_capability`` (never silent success, never free-form LLM fallback).
2. Analytical unique-mapping probe — a single hermetic repair candidate that
   closes the claim → ``closed_deterministic`` (zero provider hooks).
3. Ambiguity probe — multiple competing repair candidates or an explicit
   ambiguity signal → ``abstain_review`` (typed residual for operator; no
   provider call).
4. Residual authorization — only when an explicit residual packet CID is
   supplied and capabilities are present → ``residual_llm_authorized``.

Fail-closed rules:

* Cold import never loads LLM / remote model-provider surfaces.
* Optional backends are explicit; absence yields typed deferral.
* Unknown task / forest bindings fail closed.
* Source bodies and secrets never enter kernel receipts.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, Protocol

from .implementation_disposition import (
    ImplementationDisposition,
    ImplementationDispositionAuthorityError,
    ImplementationDispositionError,
    ImplementationForestRoots,
    PreImplementationKernelReceipt,
    implementation_disposition_cid,
    seal_pre_implementation_kernel_receipt,
)


# ---------------------------------------------------------------------------
# Interface identity
# ---------------------------------------------------------------------------

PRE_IMPLEMENTATION_KERNEL_INTERFACE: Final[str] = "PreImplementationKernel@1"
PRE_IMPLEMENTATION_KERNEL_VERSION: Final[int] = 1
PRE_IMPLEMENTATION_KERNEL_EVIDENCE: Final[str] = "wpd/pre-implementation-kernel@1"
PRE_IMPLEMENTATION_KERNEL_PRODUCER: Final[str] = "pre-implementation-kernel@1"

# Reason codes used by hermetic fixtures and production probes.
REASON_ANALYTICAL_UNIQUE_MAPPING: Final[str] = "analytical_unique_mapping"
REASON_AMBIGUOUS_CANDIDATES: Final[str] = "ambiguous_repair_candidates"
REASON_MISSING_BACKEND: Final[str] = "missing_required_backend"
REASON_RESIDUAL_AUTHORIZED: Final[str] = "residual_packet_authorized"
REASON_NO_ANALYTICAL_CLOSE: Final[str] = "no_analytical_close"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class PreImplementationKernelError(RuntimeError):
    """Fail-closed rejection for an unsafe or incomplete kernel evaluation."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str = "pre_implementation_kernel_error",
    ) -> None:
        super().__init__(message)
        self.reason_code = str(reason_code or "pre_implementation_kernel_error")


class PreImplementationKernelInputError(PreImplementationKernelError, ValueError):
    """Caller supplied an incomplete or malformed evaluation request."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str = "invalid_kernel_input",
    ) -> None:
        super().__init__(message, reason_code=reason_code)


# ---------------------------------------------------------------------------
# Evaluation request / analytical probe contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnalyticalRepairCandidate:
    """Hermetic description of one deterministic repair candidate.

    Bodies are forbidden — only opaque identifiers and reason codes.
    """

    candidate_id: str
    reason_code: str = REASON_ANALYTICAL_UNIQUE_MAPPING
    closes_claim: bool = True
    evidence_cids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not str(self.candidate_id or "").strip():
            raise PreImplementationKernelInputError(
                "candidate_id is required",
                reason_code="missing_candidate_id",
            )
        object.__setattr__(self, "candidate_id", str(self.candidate_id).strip())
        object.__setattr__(
            self,
            "reason_code",
            str(self.reason_code or REASON_ANALYTICAL_UNIQUE_MAPPING).strip(),
        )
        object.__setattr__(
            self,
            "evidence_cids",
            tuple(str(item).strip() for item in self.evidence_cids if str(item).strip()),
        )


class AnalyticalProbe(Protocol):
    """Optional probe that returns ordered analytical repair candidates."""

    def __call__(self, request: "KernelEvaluationRequest") -> Sequence[AnalyticalRepairCandidate]:
        ...


@dataclass(frozen=True)
class KernelEvaluationRequest:
    """Inputs for a single pre-implementation kernel evaluation.

    Callers must supply the claimed task identity and exact forest roots.
    Optional residual packet CID is only admitted when the kernel selects
    ``residual_llm_authorized``.
    """

    task_cid: str
    forest_roots: ImplementationForestRoots
    attempt: int = 1
    policy_revision: str = "1"
    residual_packet_cid: str = ""
    analytical_candidates: tuple[AnalyticalRepairCandidate, ...] = ()
    require_planner: bool = True
    require_doctor: bool = True
    planner_available: bool | None = None
    doctor_available: bool | None = None
    evidence_cids: tuple[str, ...] = ()
    obligation_graph_cid: str = ""
    plan_cid: str = ""
    doctor_cid: str = ""
    shared_validation_command_cids: tuple[str, ...] = ()
    shared_edit_packet_cids: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        task = str(self.task_cid or "").strip()
        if not task or any(ch.isspace() for ch in task):
            raise PreImplementationKernelInputError(
                "task_cid is required and must be compact",
                reason_code="missing_task_cid",
            )
        object.__setattr__(self, "task_cid", task)
        if not isinstance(self.forest_roots, ImplementationForestRoots):
            raise PreImplementationKernelInputError(
                "forest_roots must be ImplementationForestRoots",
                reason_code="invalid_forest_roots",
            )
        if int(self.attempt) < 1:
            raise PreImplementationKernelInputError(
                "attempt must be >= 1",
                reason_code="invalid_attempt",
            )
        object.__setattr__(self, "attempt", int(self.attempt))
        object.__setattr__(
            self,
            "residual_packet_cid",
            str(self.residual_packet_cid or "").strip(),
        )
        object.__setattr__(
            self,
            "policy_revision",
            str(self.policy_revision or "").strip(),
        )
        object.__setattr__(
            self,
            "analytical_candidates",
            tuple(self.analytical_candidates or ()),
        )
        object.__setattr__(
            self,
            "evidence_cids",
            tuple(str(item).strip() for item in self.evidence_cids if str(item).strip()),
        )
        object.__setattr__(
            self,
            "shared_validation_command_cids",
            tuple(
                str(item).strip()
                for item in self.shared_validation_command_cids
                if str(item).strip()
            ),
        )
        object.__setattr__(
            self,
            "shared_edit_packet_cids",
            tuple(
                str(item).strip()
                for item in self.shared_edit_packet_cids
                if str(item).strip()
            ),
        )


@dataclass(frozen=True)
class KernelEvaluationResult:
    """Outcome of :meth:`PreImplementationKernel.evaluate`."""

    receipt: PreImplementationKernelReceipt
    provider_hook_count: int
    analytical_candidate_count: int
    reason_code: str

    @property
    def disposition(self) -> ImplementationDisposition:
        return self.receipt.disposition

    @property
    def authorizes_provider(self) -> bool:
        return self.receipt.authorizes_provider


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------


@dataclass
class PreImplementationKernel:
    """Deterministic-first pre-implementation evaluator for workers.

    Optional analytical probes and capability flags are injected so hermetic
    tests and production composition share one surface.  Provider hooks are
    never invoked by this class — callers record ``provider_hook_count`` which
    is always zero from kernel evaluation itself.
    """

    planner_available: bool = True
    doctor_available: bool = True
    analytical_probe: AnalyticalProbe | None = None
    allow_residual_without_explicit_packet: bool = False

    def evaluate(
        self,
        request: KernelEvaluationRequest | Mapping[str, Any],
    ) -> KernelEvaluationResult:
        """Evaluate a claimed task and return a sealed kernel receipt.

        Never calls a model provider.  ``provider_hook_count`` is always 0.
        """

        normalized = self._normalize_request(request)
        capability = self._capability_disposition(normalized)
        if capability is not None:
            receipt = self._seal(
                normalized,
                disposition=ImplementationDisposition.DEFER_CAPABILITY,
                reason_code=capability,
            )
            return KernelEvaluationResult(
                receipt=receipt,
                provider_hook_count=0,
                analytical_candidate_count=0,
                reason_code=capability,
            )

        candidates = self._collect_candidates(normalized)
        closing = [item for item in candidates if item.closes_claim]
        if len(closing) == 1:
            winner = closing[0]
            evidence = tuple(
                dict.fromkeys(
                    list(normalized.evidence_cids) + list(winner.evidence_cids)
                )
            )
            receipt = self._seal(
                normalized,
                disposition=ImplementationDisposition.CLOSED_DETERMINISTIC,
                reason_code=winner.reason_code or REASON_ANALYTICAL_UNIQUE_MAPPING,
                evidence_cids=evidence,
            )
            return KernelEvaluationResult(
                receipt=receipt,
                provider_hook_count=0,
                analytical_candidate_count=len(candidates),
                reason_code=receipt.reason_code,
            )

        if len(closing) > 1 or (
            len(candidates) > 1 and not closing and not normalized.residual_packet_cid
        ):
            receipt = self._seal(
                normalized,
                disposition=ImplementationDisposition.ABSTAIN_REVIEW,
                reason_code=REASON_AMBIGUOUS_CANDIDATES,
                evidence_cids=normalized.evidence_cids,
            )
            return KernelEvaluationResult(
                receipt=receipt,
                provider_hook_count=0,
                analytical_candidate_count=len(candidates),
                reason_code=REASON_AMBIGUOUS_CANDIDATES,
            )

        if normalized.residual_packet_cid:
            receipt = self._seal(
                normalized,
                disposition=ImplementationDisposition.RESIDUAL_LLM_AUTHORIZED,
                reason_code=REASON_RESIDUAL_AUTHORIZED,
                residual_packet_cid=normalized.residual_packet_cid,
                evidence_cids=normalized.evidence_cids,
            )
            return KernelEvaluationResult(
                receipt=receipt,
                provider_hook_count=0,
                analytical_candidate_count=len(candidates),
                reason_code=REASON_RESIDUAL_AUTHORIZED,
            )

        # No unique analytical close and no residual packet → abstain, never
        # free-form re-prompt.
        receipt = self._seal(
            normalized,
            disposition=ImplementationDisposition.ABSTAIN_REVIEW,
            reason_code=REASON_NO_ANALYTICAL_CLOSE,
            evidence_cids=normalized.evidence_cids,
        )
        return KernelEvaluationResult(
            receipt=receipt,
            provider_hook_count=0,
            analytical_candidate_count=len(candidates),
            reason_code=REASON_NO_ANALYTICAL_CLOSE,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _normalize_request(
        self,
        request: KernelEvaluationRequest | Mapping[str, Any],
    ) -> KernelEvaluationRequest:
        if isinstance(request, KernelEvaluationRequest):
            return request
        if not isinstance(request, Mapping):
            raise PreImplementationKernelInputError(
                "request must be KernelEvaluationRequest or mapping",
                reason_code="invalid_request_type",
            )
        payload = dict(request)
        forest = payload.get("forest_roots")
        if isinstance(forest, Mapping):
            if "schema" in forest:
                payload["forest_roots"] = ImplementationForestRoots.from_dict(forest)
            else:
                payload["forest_roots"] = ImplementationForestRoots(**forest)
        candidates = payload.get("analytical_candidates") or ()
        normalized_candidates: list[AnalyticalRepairCandidate] = []
        for item in candidates:
            if isinstance(item, AnalyticalRepairCandidate):
                normalized_candidates.append(item)
            elif isinstance(item, Mapping):
                normalized_candidates.append(
                    AnalyticalRepairCandidate(
                        candidate_id=str(item.get("candidate_id") or ""),
                        reason_code=str(
                            item.get("reason_code") or REASON_ANALYTICAL_UNIQUE_MAPPING
                        ),
                        closes_claim=bool(item.get("closes_claim", True)),
                        evidence_cids=tuple(item.get("evidence_cids") or ()),
                    )
                )
            else:
                raise PreImplementationKernelInputError(
                    "analytical_candidates entries must be mappings or "
                    "AnalyticalRepairCandidate",
                    reason_code="invalid_candidate",
                )
        payload["analytical_candidates"] = tuple(normalized_candidates)
        return KernelEvaluationRequest(**payload)

    def _capability_disposition(
        self,
        request: KernelEvaluationRequest,
    ) -> str | None:
        planner_ok = (
            self.planner_available
            if request.planner_available is None
            else bool(request.planner_available)
        )
        doctor_ok = (
            self.doctor_available
            if request.doctor_available is None
            else bool(request.doctor_available)
        )
        if request.require_planner and not planner_ok:
            return REASON_MISSING_BACKEND
        if request.require_doctor and not doctor_ok:
            return REASON_MISSING_BACKEND
        return None

    def _collect_candidates(
        self,
        request: KernelEvaluationRequest,
    ) -> list[AnalyticalRepairCandidate]:
        candidates = list(request.analytical_candidates)
        if self.analytical_probe is not None:
            probed = self.analytical_probe(request)
            for item in probed:
                if isinstance(item, AnalyticalRepairCandidate):
                    candidates.append(item)
                else:
                    raise PreImplementationKernelError(
                        "analytical_probe must yield AnalyticalRepairCandidate",
                        reason_code="invalid_probe_output",
                    )
        return candidates

    def _view_cids(
        self,
        request: KernelEvaluationRequest,
        *,
        disposition: ImplementationDisposition,
        reason_code: str,
    ) -> tuple[str, str, str]:
        """Derive stable dual-view CIDs from the evaluation identity."""

        base = {
            "task_cid": request.task_cid,
            "forest": request.forest_roots.content_id,
            "disposition": disposition.value,
            "reason_code": reason_code,
            "attempt": request.attempt,
            "policy_revision": request.policy_revision,
        }
        obligation = str(request.obligation_graph_cid or "").strip() or (
            implementation_disposition_cid({**base, "view": "obligation"})
        )
        plan = str(request.plan_cid or "").strip() or (
            implementation_disposition_cid({**base, "view": "planner"})
        )
        doctor = str(request.doctor_cid or "").strip() or (
            implementation_disposition_cid({**base, "view": "doctor"})
        )
        if plan == doctor:
            # Dual-view contract requires distinct plan/doctor identities.
            doctor = implementation_disposition_cid(
                {**base, "view": "doctor", "salt": "dual"}
            )
        return obligation, plan, doctor

    def _seal(
        self,
        request: KernelEvaluationRequest,
        *,
        disposition: ImplementationDisposition,
        reason_code: str,
        residual_packet_cid: str = "",
        evidence_cids: Sequence[str] = (),
    ) -> PreImplementationKernelReceipt:
        obligation, plan, doctor = self._view_cids(
            request,
            disposition=disposition,
            reason_code=reason_code,
        )
        try:
            return seal_pre_implementation_kernel_receipt(
                task_cid=request.task_cid,
                disposition=disposition,
                forest_roots=request.forest_roots,
                plan_cid=plan,
                doctor_cid=doctor,
                obligation_graph_cid=obligation,
                attempt=request.attempt,
                residual_packet_cid=residual_packet_cid,
                reason_code=reason_code,
                evidence_cids=tuple(evidence_cids),
                policy_revision=request.policy_revision,
                shared_validation_command_cids=request.shared_validation_command_cids,
                shared_edit_packet_cids=request.shared_edit_packet_cids,
                producer_id=PRE_IMPLEMENTATION_KERNEL_PRODUCER,
            )
        except (
            ImplementationDispositionError,
            ImplementationDispositionAuthorityError,
        ) as exc:
            raise PreImplementationKernelError(
                f"failed to seal kernel receipt: {exc}",
                reason_code="receipt_seal_failed",
            ) from exc


def build_pre_implementation_kernel(
    *,
    planner_available: bool = True,
    doctor_available: bool = True,
    analytical_probe: AnalyticalProbe | None = None,
) -> PreImplementationKernel:
    """Construct a production-default pre-implementation kernel."""

    return PreImplementationKernel(
        planner_available=planner_available,
        doctor_available=doctor_available,
        analytical_probe=analytical_probe,
    )


def evaluate_pre_implementation(
    request: KernelEvaluationRequest | Mapping[str, Any],
    *,
    planner_available: bool = True,
    doctor_available: bool = True,
    analytical_probe: AnalyticalProbe | None = None,
) -> KernelEvaluationResult:
    """Module-level convenience wrapper around :meth:`PreImplementationKernel.evaluate`."""

    return build_pre_implementation_kernel(
        planner_available=planner_available,
        doctor_available=doctor_available,
        analytical_probe=analytical_probe,
    ).evaluate(request)


__all__ = [
    "PRE_IMPLEMENTATION_KERNEL_EVIDENCE",
    "PRE_IMPLEMENTATION_KERNEL_INTERFACE",
    "PRE_IMPLEMENTATION_KERNEL_PRODUCER",
    "PRE_IMPLEMENTATION_KERNEL_VERSION",
    "REASON_AMBIGUOUS_CANDIDATES",
    "REASON_ANALYTICAL_UNIQUE_MAPPING",
    "REASON_MISSING_BACKEND",
    "REASON_NO_ANALYTICAL_CLOSE",
    "REASON_RESIDUAL_AUTHORIZED",
    "AnalyticalProbe",
    "AnalyticalRepairCandidate",
    "KernelEvaluationRequest",
    "KernelEvaluationResult",
    "PreImplementationKernel",
    "PreImplementationKernelError",
    "PreImplementationKernelInputError",
    "build_pre_implementation_kernel",
    "evaluate_pre_implementation",
]
