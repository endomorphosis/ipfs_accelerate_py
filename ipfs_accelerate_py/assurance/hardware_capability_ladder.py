"""Canonical hardware capability ladder (PCPR-031 origin, PCPR-034 consolidation).

Replace fabricated hardware availability with an ordered ladder. Device
visibility, adapter presence, package import, configuration, or testing
defaults never imply ``production_authorized``. Simulated results stay
``Simulated``. Missing environments stay typed unavailable (null), never a
fabricated False that could be read as a live probe.

Production execution is admitted only when ``production_authorized`` is
True. This evaluator never grants that rung: live CUDA, CPU execution,
model/provider qualification, and a closed PCPR release remain later
tasks.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Final, Literal, Optional

TASK_ID: Final[str] = "PCPR-031"
CONSOLIDATION_TASK_ID: Final[str] = "PCPR-034"
GOAL_ID: Final[str] = "PCPR-G410"
INTERFACE: Final[str] = "HardwareCapabilityLadder@1"
SCHEMA: Final[str] = "ipfs_accelerate_py/assurance/hardware-capability-ladder@1"
CONSOLIDATION_INTERFACE: Final[str] = "CapabilityLadderConsolidation@1"
CONSOLIDATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/assurance/capability-ladder-consolidation@1"
)
PRODUCTION_EXECUTION_CODE: Final[str] = (
    "production_execution_requires_production_authorized"
)

LadderRung = Literal[
    "declared",
    "installed",
    "detected",
    "canary_passed",
    "model_compatible",
    "resource_sufficient",
    "qualified",
    "production_authorized",
]

LADDER_RUNGS: Final[tuple[str, ...]] = (
    "declared",
    "installed",
    "detected",
    "canary_passed",
    "model_compatible",
    "resource_sufficient",
    "qualified",
    "production_authorized",
)

WEAK_ORIGINS: Final[frozenset[str]] = frozenset(
    {"absent", "declared", "fixture", "simulated", "unavailable"}
)
OBSERVABLE_ORIGINS: Final[frozenset[str]] = frozenset(
    {"hermetic_observed", "live_observed"}
)

_UNAVAILABLE_BACKENDS: Final[tuple[str, ...]] = (
    "cuda",
    "rocm",
    "mps",
    "openvino",
    "qualcomm",
    "qnn",
    "webnn",
    "webgpu",
    "metal",
    "xpu",
    "habana",
    "tpu",
    "apple",
)


class HardwareCapabilityLadderError(ValueError):
    """Malformed hardware-capability ladder construction."""


def _ternary(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    raise HardwareCapabilityLadderError("ladder rungs must be bool or None")


def empty_ladder(
    *,
    backend: str,
    declared: bool = False,
    origin: str = "absent",
    evidence_kind: str = "unavailable",
    live: bool = False,
    outcome: str = "Unavailable",
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a ladder payload with every evidence rung typed unavailable."""

    name = str(backend or "").strip()
    if not name:
        raise HardwareCapabilityLadderError("backend must be a non-empty string")
    payload: dict[str, Any] = {
        "backend": name,
        "available": None,
        "declared": bool(declared),
        "installed": None,
        "detected": None,
        "canary_passed": None,
        "model_compatible": None,
        "resource_sufficient": None,
        "qualified": False,
        "production_authorized": False,
        "origin": origin,
        "live": False,
        "simulated": origin == "simulated",
        "outcome": outcome,
        "evidence_kind": evidence_kind,
        "package_import_is_not_qualification": True,
        "device_visibility_is_not_qualification": True,
        "adapter_presence_is_not_qualification": True,
        "configuration_is_not_qualification": True,
        "testing_defaults_are_not_qualification": True,
        "schema": SCHEMA,
        "interface": INTERFACE,
        "task_id": TASK_ID,
    }
    if extra:
        payload.update(dict(extra))
    payload["live"] = False
    payload["production_authorized"] = False
    payload["qualified"] = False
    if live:
        raise HardwareCapabilityLadderError(
            "this evaluator cannot mint live hardware qualification"
        )
    return payload


def unavailable_backend(backend: str, **extra: Any) -> dict[str, Any]:
    """Typed unavailable backend. Absence is not recorded as measured False."""

    return empty_ladder(
        backend=backend,
        declared=False,
        origin="absent",
        evidence_kind="unavailable",
        outcome="Unavailable",
        extra=extra,
    )


def declared_cpu_baseline(**extra: Any) -> dict[str, Any]:
    """CPU as a declared process host. Not production_authorized. Not live CUDA."""

    payload = empty_ladder(
        backend="cpu",
        declared=True,
        origin="declared",
        evidence_kind="measured",
        outcome="Unavailable",
        extra=extra,
    )
    payload["available"] = True
    payload["installed"] = True
    payload["detected"] = None
    payload["live"] = False
    payload["production_authorized"] = False
    payload["qualified"] = False
    return payload


def simulated_backend(backend: str, *, available: bool | None = None, **extra: Any) -> dict[str, Any]:
    """Explicit simulation payload. Never live. Never production_authorized."""

    payload = empty_ladder(
        backend=backend,
        declared=True,
        origin="simulated",
        evidence_kind="simulated",
        outcome="Simulated",
        extra=extra,
    )
    payload["available"] = available
    payload["simulated"] = True
    payload["live"] = False
    payload["production_authorized"] = False
    payload["qualified"] = False
    return payload


def from_package_import(
    backend: str,
    *,
    package: str,
    version: str | None = None,
) -> dict[str, Any]:
    """Package import is the installed rung only. It is not detection or qualification."""

    payload = empty_ladder(
        backend=backend,
        declared=True,
        origin="declared",
        evidence_kind="measured",
        outcome="Unavailable",
        extra={"package": package, "version": version},
    )
    payload["installed"] = True
    payload["available"] = False
    payload["detected"] = None
    payload["live"] = False
    payload["production_authorized"] = False
    return payload


def from_device_visibility(
    backend: str,
    *,
    devices: list[Any] | None = None,
    origin: str = "hermetic_observed",
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """nvidia-smi / torch.cuda.is_available is detection, not qualification."""

    if origin in WEAK_ORIGINS:
        raise HardwareCapabilityLadderError(
            "device visibility cannot be recorded under a weak origin as live qualification"
        )
    payload = empty_ladder(
        backend=backend,
        declared=True,
        origin=origin,
        evidence_kind="measured",
        outcome="Unavailable",
        extra=extra,
    )
    payload["installed"] = True
    payload["detected"] = True
    payload["available"] = True
    payload["devices"] = list(devices or [])
    payload["canary_passed"] = None
    payload["live"] = False
    payload["production_authorized"] = False
    payload["qualified"] = False
    return payload


def from_measured_absence(
    backend: str,
    *,
    origin: str = "hermetic_observed",
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """A real probe ran and reported the backend is not present."""

    payload = empty_ladder(
        backend=backend,
        declared=True,
        origin=origin,
        evidence_kind="measured",
        outcome="Unavailable",
        extra=extra,
    )
    payload["installed"] = True
    payload["detected"] = False
    payload["available"] = False
    payload["live"] = False
    payload["production_authorized"] = False
    return payload


def from_platform_presence(
    backend: str,
    *,
    platform_name: str,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """OS/arch presence is declared only. Darwin is not Metal qualification."""

    payload = empty_ladder(
        backend=backend,
        declared=True,
        origin="declared",
        evidence_kind="measured",
        outcome="Unavailable",
        extra={"platform": platform_name, **dict(extra or {})},
    )
    payload["available"] = None
    payload["installed"] = None
    payload["detected"] = None
    payload["live"] = False
    payload["production_authorized"] = False
    return payload


def unavailable_hardware_map() -> dict[str, dict[str, Any]]:
    """Ordinary-runtime map when no live hardware probe is admitted."""

    payload = {"cpu": declared_cpu_baseline(cores=None)}
    for name in _UNAVAILABLE_BACKENDS:
        payload[name] = unavailable_backend(name)
    return payload


def simulated_hardware_map() -> dict[str, dict[str, Any]]:
    """Explicit-simulation map. CUDA is never reported live."""

    payload = {
        "cpu": simulated_backend("cpu", available=True, cores=1),
    }
    for name in _UNAVAILABLE_BACKENDS:
        payload[name] = simulated_backend(name, available=False)
    return payload


def assess_ladder(report: Mapping[str, Any]) -> dict[str, Any]:
    """Compute production_authorized. This task never authorizes production."""

    if not isinstance(report, Mapping):
        raise HardwareCapabilityLadderError("ladder report must be a mapping")
    origin = str(report.get("origin") or "absent")
    if origin in {"simulated", "fixture"} or report.get("simulated") is True:
        return MappingProxyType(
            {
                "production_authorized": False,
                "qualified": False,
                "live": False,
                "outcome": "Simulated",
                "reason": "simulated_or_fixture_origin_cannot_authorize_production",
                "blocking_rung": "origin",
            }
        )
    for rung in LADDER_RUNGS[:-1]:
        value = _ternary(report.get(rung)) if rung in report else None
        if value is not True:
            return {
                "production_authorized": False,
                "qualified": False,
                "live": False,
                "outcome": "Unavailable",
                "reason": f"ladder_rung_not_proven:{rung}",
                "blocking_rung": rung,
            }
    # All evidence rungs True still cannot authorize production here: live CUDA
    # and CPU qualification are PCPR-037/PCPR-038 and this evaluator has no
    # live campaign.
    return {
        "production_authorized": False,
        "qualified": False,
        "live": False,
        "outcome": "Unavailable",
        "reason": "this_evaluator_does_not_authorize_production",
        "blocking_rung": "production_authorized",
    }


def attach_ladder(report: dict[str, Any]) -> dict[str, Any]:
    """Fill missing rungs and force production_authorized False."""

    if "declared" not in report:
        report["declared"] = bool(report.get("available") is True)
    for rung in LADDER_RUNGS:
        report.setdefault(rung, False if rung in {"qualified", "production_authorized"} else None)
    report["production_authorized"] = False
    report["qualified"] = False
    report["live"] = False
    report.setdefault("package_import_is_not_qualification", True)
    report.setdefault("device_visibility_is_not_qualification", True)
    report.setdefault("adapter_presence_is_not_qualification", True)
    report.setdefault("schema", SCHEMA)
    report.setdefault("interface", INTERFACE)
    assessed = assess_ladder(report)
    report["ladder_assessment"] = dict(assessed)
    return report


def production_authorized(report: Mapping[str, Any] | None) -> bool:
    if not isinstance(report, Mapping):
        return False
    return assess_ladder(report).get("production_authorized") is True


def from_canary(
    backend: str,
    *,
    passed: bool | None,
    origin: str = "hermetic_observed",
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Record a canary probe. A passing canary is not qualification."""

    if origin in WEAK_ORIGINS:
        raise HardwareCapabilityLadderError(
            "canary evidence cannot be recorded under a weak origin as live qualification"
        )
    payload = empty_ladder(
        backend=backend,
        declared=True,
        origin=origin,
        evidence_kind="measured",
        outcome="Unavailable",
        extra=extra,
    )
    payload["installed"] = True
    payload["detected"] = True
    payload["available"] = True
    payload["canary_passed"] = None if passed is None else bool(passed)
    payload["live"] = False
    payload["qualified"] = False
    payload["production_authorized"] = False
    return attach_ladder(payload)


def stamp_consolidation(report: dict[str, Any]) -> dict[str, Any]:
    """Mark a ladder report as PCPR-034-consolidated. Never authorizes production."""

    attach_ladder(report)
    report["consolidation_schema"] = CONSOLIDATION_SCHEMA
    report["consolidation_interface"] = CONSOLIDATION_INTERFACE
    report["consolidation_task_id"] = CONSOLIDATION_TASK_ID
    report["production_authorized"] = False
    report["qualified"] = False
    report["live"] = False
    return report


def admit_production_execution(
    report: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Admit production work only when production_authorized is proven.

    This evaluator never admits production. Detection, canary, package import,
    and advisory recommendation remain non-authoritative.
    """

    if not isinstance(report, Mapping):
        assessed = {
            "production_authorized": False,
            "qualified": False,
            "live": False,
            "outcome": "Unavailable",
            "reason": "ladder_report_absent",
            "blocking_rung": "declared",
        }
        backend = ""
    else:
        assessed = assess_ladder(report)
        backend = str(report.get("backend") or report.get("name") or "")
    return {
        "admitted": False,
        "backend": backend,
        "production_authorized": False,
        "qualified": False,
        "live": False,
        "outcome": "Unavailable",
        "status": "unavailable",
        "code": PRODUCTION_EXECUTION_CODE,
        "reason": assessed.get("reason"),
        "blocking_rung": assessed.get("blocking_rung"),
        "schema": CONSOLIDATION_SCHEMA,
        "interface": CONSOLIDATION_INTERFACE,
        "task_id": CONSOLIDATION_TASK_ID,
        "ladder_assessment": dict(assessed),
    }


def refused_production_execution(
    backend: str,
    *,
    reason: str = "detection_is_not_production_authorized",
) -> dict[str, Any]:
    """Typed refusal for production work selected from detection alone."""

    name = str(backend or "").strip() or "unspecified"
    return {
        "admitted": False,
        "backend": name,
        "hardware": None,
        "production_authorized": False,
        "qualified": False,
        "live": False,
        "outcome": "Unavailable",
        "status": "unavailable",
        "code": PRODUCTION_EXECUTION_CODE,
        "reason": reason,
        "schema": CONSOLIDATION_SCHEMA,
        "interface": CONSOLIDATION_INTERFACE,
        "task_id": CONSOLIDATION_TASK_ID,
    }


__all__ = (
    "CONSOLIDATION_INTERFACE",
    "CONSOLIDATION_SCHEMA",
    "CONSOLIDATION_TASK_ID",
    "GOAL_ID",
    "INTERFACE",
    "LADDER_RUNGS",
    "PRODUCTION_EXECUTION_CODE",
    "SCHEMA",
    "TASK_ID",
    "HardwareCapabilityLadderError",
    "admit_production_execution",
    "assess_ladder",
    "attach_ladder",
    "declared_cpu_baseline",
    "empty_ladder",
    "from_canary",
    "from_device_visibility",
    "from_measured_absence",
    "from_package_import",
    "from_platform_presence",
    "production_authorized",
    "refused_production_execution",
    "simulated_backend",
    "simulated_hardware_map",
    "stamp_consolidation",
    "unavailable_backend",
    "unavailable_hardware_map",
)
