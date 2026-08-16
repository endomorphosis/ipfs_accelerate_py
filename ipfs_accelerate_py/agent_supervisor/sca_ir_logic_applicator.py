"""SCA-facing re-export of general IR logic application.

Canonical implementation (domain-agnostic):
:mod:`ipfs_accelerate_py.agent_supervisor.proof.ir_logic_application`

SCA scripts may keep importing this module. New planner/doctor/repair code
should import from ``proof.ir_logic_application`` and set ``domain`` explicitly
(e.g. ``domain="sca"`` only when the work is SCA residual).
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .proof.ir_logic_application import *  # noqa: F403
from .proof.ir_logic_application import (
    DEFAULT_APPLY_FAMILIES,
    IR_LOGIC_APPLICATION_INTERFACE,
    IR_LOGIC_APPLY_SCHEMA,
    IR_LOGIC_BATCH_SCHEMA,
    SHARED_IR_FAMILIES,
    STRUCTURAL_IR_FAMILIES,
    SCA_IR_LOGIC_APPLICATOR_INTERFACE,
    SCA_IR_LOGIC_APPLY_SCHEMA,
    SCA_IR_LOGIC_BATCH_SCHEMA,
    IrLogicApplicationError,
    IrLogicApplyPolicy,
    IrWorkSurface,
    ScaIrLogicApplicatorError,
    apply_intent_logic,
    apply_legal_logic,
    apply_logic_to_findings,
    apply_logic_to_ir as _apply_logic_to_ir,
    apply_logic_to_surfaces as _apply_logic_to_surfaces,
    apply_plan_admission,
    apply_security_logic,
    apply_ui_logic,
    load_apply_policy_from_supervisor_profile,
    project_candidate_plan,
)


def apply_logic_to_ir(
    surface: IrWorkSurface | Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Apply IR logic; default domain is ``sca`` only for this SCA wrapper."""
    if "domain" not in kwargs and surface is None:
        kwargs.setdefault("domain", "sca")
    elif "domain" not in kwargs and isinstance(surface, Mapping):
        if not surface.get("domain"):
            kwargs.setdefault("domain", "sca")
    elif (
        "domain" not in kwargs
        and isinstance(surface, IrWorkSurface)
        and surface.domain == "agent_supervisor"
    ):
        kwargs.setdefault("domain", "sca")
    kwargs.setdefault("consumer", kwargs.get("consumer") or "symbolic_repair")
    return _apply_logic_to_ir(surface, **kwargs)


def apply_logic_to_surfaces(
    surfaces: Sequence[Mapping[str, Any] | IrWorkSurface],
    **kwargs: Any,
) -> dict[str, Any]:
    kwargs.setdefault("domain", "sca")
    kwargs.setdefault("consumer", "symbolic_repair")
    return _apply_logic_to_surfaces(surfaces, **kwargs)


apply_logic_to_findings = apply_logic_to_surfaces


__all__ = [
    "DEFAULT_APPLY_FAMILIES",
    "SHARED_IR_FAMILIES",
    "STRUCTURAL_IR_FAMILIES",
    "IR_LOGIC_APPLICATION_INTERFACE",
    "IR_LOGIC_APPLY_SCHEMA",
    "IR_LOGIC_BATCH_SCHEMA",
    "SCA_IR_LOGIC_APPLICATOR_INTERFACE",
    "SCA_IR_LOGIC_APPLY_SCHEMA",
    "SCA_IR_LOGIC_BATCH_SCHEMA",
    "IrLogicApplyPolicy",
    "IrLogicApplicationError",
    "IrWorkSurface",
    "ScaIrLogicApplicatorError",
    "apply_intent_logic",
    "apply_legal_logic",
    "apply_logic_to_findings",
    "apply_logic_to_ir",
    "apply_logic_to_surfaces",
    "apply_plan_admission",
    "apply_security_logic",
    "apply_ui_logic",
    "load_apply_policy_from_supervisor_profile",
    "project_candidate_plan",
]
