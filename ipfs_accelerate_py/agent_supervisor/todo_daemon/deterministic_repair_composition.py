"""Live, receipt-bound deterministic repair composition for DCR-080.

The composition invokes concrete Planner, Doctor, logic, repair, validation,
and publication services.  It closes only with a repair/publication receipt
pair or a proved-valid validation observation.  Every other outcome is an
explicit abstention and provider dispatch is structurally absent.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from ..control.pre_implementation_provider_gate import evaluate_provider_gate


DETERMINISTIC_REPAIR_COMPOSITION_INTERFACE = "DeterministicRepairCompositionRoot@1"


def _cid(value: Mapping[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(
        json.dumps(dict(value), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        converted = to_dict()
        if isinstance(converted, Mapping):
            return converted
    return {}


def _receipt_id(value: Any) -> str:
    record = _as_mapping(value)
    for key in ("receipt_cid", "receipt_id", "content_id", "id"):
        candidate = record.get(key)
        if candidate:
            return str(candidate)
    for key in ("receipt", "observation", "run_receipt"):
        nested = record.get(key)
        if nested is not None:
            candidate = _receipt_id(nested)
            if candidate:
                return candidate
    for attr in ("receipt_cid", "receipt_id", "content_id"):
        candidate = getattr(value, attr, "")
        if candidate:
            return str(candidate)
    return ""


def _proved_valid(value: Any) -> bool:
    record = _as_mapping(value)
    return bool(
        record.get("proved_valid") is True
        or record.get("valid") is True and record.get("proof_receipt")
        or record.get("disposition") in {"proved_valid", "validated"}
    ) and bool(_receipt_id(value))


@dataclass(frozen=True)
class DeterministicRepairResult:
    task_id: str
    disposition: str
    reason_codes: tuple[str, ...]
    service_receipts: Mapping[str, str] = field(default_factory=dict)
    proved_valid: bool = False

    @property
    def receipt_cid(self) -> str:
        return _cid(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "interface": DETERMINISTIC_REPAIR_COMPOSITION_INTERFACE,
            "task_id": self.task_id,
            "disposition": self.disposition,
            "reason_codes": list(self.reason_codes),
            "service_receipts": dict(sorted(self.service_receipts.items())),
            "proved_valid": self.proved_valid,
            "provider_authorized": False,
            "provider_hook_count": 0,
        }


class DeterministicRepairCompositionRoot:
    """Receipt-checked service composition; it owns no synthetic success path."""

    INTERFACE = DETERMINISTIC_REPAIR_COMPOSITION_INTERFACE

    def __init__(
        self,
        *,
        planner: Any,
        doctor: Any,
        logic: Any,
        repair: Any | None = None,
        validator: Any | None = None,
        publisher: Any | None = None,
    ) -> None:
        self.planner = planner
        self.doctor = doctor
        self.logic = logic
        self.repair = repair if repair is not None else doctor
        self.validator = validator if validator is not None else planner
        self.publisher = publisher

    @staticmethod
    def discovery() -> dict[str, Any]:
        return {
            "interface": DETERMINISTIC_REPAIR_COMPOSITION_INTERFACE,
            "services": ["planner", "doctor", "logic", "repair", "validator", "publisher"],
            "provider_authorized": False,
            "model_calls": 0,
            "automatic_fallback": False,
        }

    def _call(self, service: Any, method: str, context: Mapping[str, Any]) -> tuple[Any, str]:
        if service is None:
            return None, "service_unavailable"
        candidate = getattr(service, method, None)
        if not callable(candidate):
            candidate = getattr(service, "execute", None)
            if callable(candidate):
                try:
                    return candidate({"operation": method, **dict(context)}), ""
                except Exception as exc:  # fail closed with a typed local reason
                    return None, f"{method}_failed:{type(exc).__name__}"
            return None, f"{method}_unavailable"
        try:
            return candidate(**dict(context)), ""
        except Exception as exc:  # services may abstain via exceptions; never fabricate a receipt
            return None, f"{method}_failed:{type(exc).__name__}"

    def run(self, *, task_id: str) -> DeterministicRepairResult:
        receipts: dict[str, str] = {}
        reasons: list[str] = []
        context: dict[str, Any] = {"task_id": str(task_id)}
        stages = (
            ("doctor", self.doctor, "inspect"),
            ("planner", self.planner, "plan"),
            ("logic", self.logic, "prove"),
            ("repair", self.repair, "repair"),
            ("validator", self.validator, "validate"),
        )
        validation = None
        for name, service, method in stages:
            result, failure = self._call(service, method, context)
            if failure:
                return self._abstain(task_id, receipts, reasons + [failure])
            receipt = _receipt_id(result)
            if not receipt:
                return self._abstain(task_id, receipts, reasons + [f"missing_{name}_service_receipt"])
            receipts[name] = receipt
            context[f"{name}_receipt"] = receipt
            if name == "validator":
                validation = result

        if _proved_valid(validation):
            return self._close(task_id, receipts, proved_valid=True)

        publication, failure = self._call(self.publisher, "publish", context)
        if failure:
            return self._abstain(task_id, receipts, reasons + [failure])
        publication_receipt = _receipt_id(publication)
        if not publication_receipt:
            return self._abstain(task_id, receipts, reasons + ["missing_publication_service_receipt"])
        receipts["publication"] = publication_receipt
        return self._close(task_id, receipts, proved_valid=False)

    def _close(self, task_id: str, receipts: Mapping[str, str], *, proved_valid: bool) -> DeterministicRepairResult:
        if not proved_valid and not (receipts.get("repair") and receipts.get("publication")):
            return self._abstain(task_id, receipts, ["close_receipt_incomplete"])
        return DeterministicRepairResult(
            task_id=str(task_id), disposition="closed_deterministic", reason_codes=(),
            service_receipts=dict(receipts), proved_valid=proved_valid,
        )

    def _abstain(self, task_id: str, receipts: Mapping[str, str], reasons: list[str]) -> DeterministicRepairResult:
        gate = evaluate_provider_gate(
            task_id=str(task_id), service_receipt_ids=tuple(receipts.values()), reason_codes=tuple(reasons)
        )
        return DeterministicRepairResult(
            task_id=str(task_id), disposition="abstain", reason_codes=tuple(reasons) or (gate.disposition,),
            service_receipts=dict(receipts), proved_valid=False,
        )


def build_repair_composition_root(
    *, checkout_root: str | Path | None = None, planner: Any | None = None,
    doctor: Any | None = None, logic: Any | None = None, repair: Any | None = None,
    validator: Any | None = None, publisher: Any | None = None,
) -> DeterministicRepairCompositionRoot:
    """Build the production defaults, retaining injected services for tests/embedding."""

    root = Path(checkout_root or Path.cwd()).resolve()
    if doctor is None:
        from ..control.default_doctor_factory import build_default_doctor_service
        doctor = build_default_doctor_service(root)
    if logic is None:
        from ..proof.ir_integration import DatasetsLogicFacade
        logic = DatasetsLogicFacade(repo_root=root)
    if planner is None:
        from ..planning.default_planner_factory import build_default_planner_handles
        planner = build_default_planner_handles(doctor_service=doctor, datasets_logic=logic)
    return DeterministicRepairCompositionRoot(
        planner=planner, doctor=doctor, logic=logic, repair=repair,
        validator=validator, publisher=publisher,
    )


def run_deterministic_repair(
    *, task_id: str, checkout_root: str | Path | None = None,
    composition_root: DeterministicRepairCompositionRoot | None = None,
) -> DeterministicRepairResult:
    """Run the only DCR-080 repair route; it never invokes a model provider."""

    try:
        root = composition_root or build_repair_composition_root(checkout_root=checkout_root)
    except Exception as exc:  # construction is a capability boundary, not success
        return DeterministicRepairResult(
            task_id=str(task_id),
            disposition="abstain",
            reason_codes=(f"composition_unavailable:{type(exc).__name__}",),
        )
    return root.run(task_id=task_id)


__all__ = [
    "DETERMINISTIC_REPAIR_COMPOSITION_INTERFACE",
    "DeterministicRepairCompositionRoot",
    "DeterministicRepairResult",
    "build_repair_composition_root",
    "run_deterministic_repair",
]
