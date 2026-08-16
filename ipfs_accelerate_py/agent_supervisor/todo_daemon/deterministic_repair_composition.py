"""DCR-080 fail-closed daemon composition for deterministic repair.

This module is deliberately a *read-only* composition root.  It re-inspects
the strict DCR-050 Doctor projection before considering any later repair
stage.  Current DCR-050/DCR-060 integrations are pending, so this route has
no execution, mutation, publication, or completion branch.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from ..proof.formal_verification_contracts import content_identity

DCR080_COMPOSITION_SCHEMA = "ipfs_accelerate_py/agent-supervisor/dcr-080-daemon-composition@1"


class DeterministicRepairCompositionDisposition(StrEnum):
    """Closed non-authoritative results for the daemon route."""

    DEFER_CAPABILITY = "defer_capability"
    ABSTAIN_REVIEW = "abstain_review"
    REJECTED = "rejected"


@dataclass(frozen=True)
class DeterministicRepairCompositionResult:
    """Auditable stop result; it is never a repair authorization."""

    disposition: DeterministicRepairCompositionDisposition
    reason_codes: tuple[str, ...]
    transitions: tuple[str, ...]
    task_id: str
    doctor_composition_cid: str = ""

    @property
    def receipt_cid(self) -> str:
        return content_identity(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": DCR080_COMPOSITION_SCHEMA,
            "authoritative": False,
            "task_id": self.task_id,
            "disposition": self.disposition.value,
            "reason_codes": list(self.reason_codes),
            "transitions": list(self.transitions),
            "doctor_composition_cid": self.doctor_composition_cid,
            "execution_authorized": False,
            "mutation_authorized": False,
            "publication_authorized": False,
            "completion_authorized": False,
            "model_call_count": 0,
            "provider_call_count": 0,
            "network_call_count": 0,
        }


def _contains_callable(value: Any) -> bool:
    if callable(value):
        return True
    if isinstance(value, Mapping):
        return any(_contains_callable(item) for item in value.values())
    if isinstance(value, (tuple, list, set, frozenset)):
        return any(_contains_callable(item) for item in value)
    return False


class DeterministicRepairCompositionRoot:
    """Compose DCR stages in order without invoking a repair implementation.

    Later DCR-070/072/073/074 stages are intentionally not accepted as loose
    dictionaries or success booleans.  The first stage is DCR-050; because it
    remains pending in this checkout, no untyped later-stage object is ever
    inspected or invoked.
    """

    def run(
        self,
        *,
        task_id: str,
        doctor_binding: Mapping[str, Any] | None = None,
        planner_handles: Any = None,
        planner_evidence: Any = None,
        logic_gate: Any = None,
        admission: Any = None,
        transaction: Any = None,
        validation: Any = None,
        publication: Any = None,
    ) -> DeterministicRepairCompositionResult:
        """Reinspect DCR-050 then fail closed before all pending stages.

        Parameters for following stages are deliberately retained only to
        reject callable/provider-shaped routes.  They cannot bypass the
        DCR-050 pending projection or cause a subprocess, network action, or
        source mutation.
        """

        if not isinstance(task_id, str) or not task_id.strip():
            raise ValueError("DCR-080 task_id must be exact non-empty text")
        supplied = (
            doctor_binding,
            planner_handles,
            planner_evidence,
            logic_gate,
            admission,
            transaction,
            validation,
            publication,
        )
        if _contains_callable(supplied):
            return DeterministicRepairCompositionResult(
                DeterministicRepairCompositionDisposition.REJECTED,
                ("callable_or_dynamic_route_rejected",),
                ("dcr050_doctor_reinspection",),
                task_id.strip(),
            )

        # Import only this existing strict, local adapter at the explicit
        # composition boundary.  It does not build Doctor stages or load a
        # provider/model surface.
        from ..control.default_doctor_factory import inspect_dcr_doctor_composition

        binding = doctor_binding if isinstance(doctor_binding, Mapping) else {}
        doctor = inspect_dcr_doctor_composition(binding)
        doctor_body = doctor.to_dict()
        doctor_cid = str(doctor_body.get("composition_cid") or "")
        if (
            doctor.disposition != "integration_pending"
            or doctor.binding_complete is not True
            or not doctor_cid
            or any(
                doctor_body.get(name) != 0
                for name in ("model_call_count", "provider_call_count", "network_call_count")
            )
        ):
            return DeterministicRepairCompositionResult(
                DeterministicRepairCompositionDisposition.DEFER_CAPABILITY,
                tuple(sorted(set(doctor.reason_codes) | {"dcr050_not_current_live"})),
                ("dcr050_doctor_reinspection",),
                task_id.strip(),
                doctor_cid,
            )

        # A complete DCR-050 projection is still explicitly transitional.
        # Do not allow caller-provided planner/admission/transaction objects
        # to skip that gate.  The names record the intended future sequence.
        return DeterministicRepairCompositionResult(
            DeterministicRepairCompositionDisposition.DEFER_CAPABILITY,
            (
                "dcr050_transitional_composition_not_live",
                "dcr060_dcr035_dcr070_dcr072_dcr073_dcr074_not_reached",
            ),
            ("dcr050_doctor_reinspection",),
            task_id.strip(),
            doctor_cid,
        )


def run_deterministic_repair(
    *, task_id: str, doctor_binding: Mapping[str, Any] | None = None
) -> DeterministicRepairCompositionResult:
    """Run the daemon's isolated zero-LLM DCR-080 composition route."""

    return DeterministicRepairCompositionRoot().run(
        task_id=task_id,
        doctor_binding=doctor_binding,
    )


__all__ = [
    "DCR080_COMPOSITION_SCHEMA",
    "DeterministicRepairCompositionDisposition",
    "DeterministicRepairCompositionResult",
    "DeterministicRepairCompositionRoot",
    "run_deterministic_repair",
]
