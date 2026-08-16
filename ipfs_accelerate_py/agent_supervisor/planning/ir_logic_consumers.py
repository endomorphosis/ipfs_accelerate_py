"""Planner / doctor consumers for domain-agnostic IR logic application.

Wires :mod:`..proof.ir_logic_application` into general planner and doctor
flows without requiring the SCA taskboard. SCA is only one optional domain.
"""

from __future__ import annotations

from typing import Any, Final, Mapping, Sequence

from ..proof.ir_logic_application import (
    DEFAULT_APPLY_FAMILIES,
    IrLogicApplyPolicy,
    IrWorkSurface,
    apply_logic_to_ir,
    apply_logic_to_surfaces,
)


IR_LOGIC_PLANNING_CONTEXT_KEY: Final = "ir_logic"
IR_LOGIC_DOCTOR_CONTEXT_KEY: Final = "ir_logic"
IR_LOGIC_CONSUMERS_INTERFACE: Final = "IrLogicConsumers@1"


def surfaces_from_planning_context(
    context: Mapping[str, Any] | None,
    *,
    domain: str = "planner",
) -> list[IrWorkSurface]:
    """Extract work surfaces from a general planning context mapping."""
    ctx = dict(context or {})
    surfaces: list[IrWorkSurface] = []

    # Explicit list
    for key in ("ir_work_surfaces", "work_surfaces", "operations", "findings"):
        items = ctx.get(key)
        if not isinstance(items, (list, tuple)):
            continue
        for item in items:
            if isinstance(item, Mapping):
                raw = dict(item)
                raw.setdefault("domain", domain)
                raw.setdefault("consumer", "planner")
                surfaces.append(IrWorkSurface.from_mapping(raw))
            elif isinstance(item, str) and item.strip():
                surfaces.append(
                    IrWorkSurface(
                        operation=item.strip(),
                        domain=domain,
                        consumer="planner",
                    )
                )

    # Single operation / goal fields
    if not surfaces:
        op = str(
            ctx.get("operation")
            or ctx.get("primary_operation")
            or ctx.get("tool")
            or ""
        )
        goal = ctx.get("goal") or ctx.get("frozen_goal") or {}
        if isinstance(goal, Mapping) and not op:
            op = str(
                goal.get("operation")
                or goal.get("name")
                or goal.get("goal_id")
                or ""
            )
        if op:
            surfaces.append(
                IrWorkSurface(
                    operation=op,
                    kind=str(ctx.get("kind") or ctx.get("work_kind") or "work_item"),
                    contract_id=str(ctx.get("contract_id") or ""),
                    path=str(ctx.get("path") or ""),
                    symbol=str(ctx.get("symbol") or ""),
                    goal_id=str(
                        ctx.get("goal_id")
                        or (goal.get("goal_id") if isinstance(goal, Mapping) else "")
                        or ""
                    ),
                    program_id=str(ctx.get("program_id") or ctx.get("repository_id") or ""),
                    domain=str(ctx.get("domain") or domain),
                    consumer="planner",
                    metadata={"source": "planning_context"},
                )
            )
    return surfaces


def enrich_planning_context_with_ir_logic(
    context: Mapping[str, Any] | None = None,
    *,
    surfaces: Sequence[Mapping[str, Any] | IrWorkSurface] | None = None,
    policy: IrLogicApplyPolicy | Mapping[str, Any] | None = None,
    domain: str = "planner",
    max_surfaces: int = 8,
) -> dict[str, Any]:
    """Apply IR logic and return an enriched planning context fragment.

    Returns a dict suitable for merging into planner context under
    :data:`IR_LOGIC_PLANNING_CONTEXT_KEY`. Does not mutate the input.
    """
    ctx = dict(context or {})
    if surfaces is None:
        work = surfaces_from_planning_context(ctx, domain=domain)
    else:
        work = [
            item
            if isinstance(item, IrWorkSurface)
            else IrWorkSurface.from_mapping(
                {**dict(item), "domain": dict(item).get("domain") or domain}
            )
            for item in surfaces
        ]
    if isinstance(policy, Mapping) or policy is None:
        pol = IrLogicApplyPolicy.from_mapping(policy)
    else:
        pol = policy
    pol.max_surfaces = min(int(pol.max_surfaces), int(max_surfaces))
    if not work:
        return {
            "schema": "ipfs_accelerate_py/agent-supervisor/planning-ir-logic@1",
            "interface": IR_LOGIC_CONSUMERS_INTERFACE,
            "passed": True,
            "skipped": True,
            "reason": "no_work_surfaces",
            "domain": domain,
            "consumer": "planner",
            "rows": [],
        }
    report = apply_logic_to_surfaces(
        work[: pol.max_surfaces],
        policy=pol,
        domain=domain,
        consumer="planner",
    )
    return {
        "schema": "ipfs_accelerate_py/agent-supervisor/planning-ir-logic@1",
        "interface": IR_LOGIC_CONSUMERS_INTERFACE,
        "passed": bool(report.get("passed")),
        "skipped": False,
        "domain": domain,
        "consumer": "planner",
        "selected_count": report.get("selected_count"),
        "summary": report.get("summary"),
        "rows": report.get("rows"),
        "gates": {
            "logic_applied_to_ir": bool(report.get("passed")),
            "no_false_execution_grants": all(
                bool((row.get("gates") or {}).get("no_false_execution_grants", True))
                for row in (report.get("rows") or [])
            ),
        },
        "families": list(DEFAULT_APPLY_FAMILIES),
    }


def attach_ir_logic_to_symbolic_plan(
    plan_result: Any,
    context: Mapping[str, Any] | None = None,
    *,
    domain: str = "planner",
) -> Any:
    """Attach IR logic receipts onto a symbolic planning result when possible.

    Supports dict results and objects with a mutable ``metadata`` / ``context``
    mapping. Non-dict results without metadata are returned unchanged with no
    forged fields.
    """
    ir_doc = enrich_planning_context_with_ir_logic(context, domain=domain)
    if isinstance(plan_result, dict):
        out = dict(plan_result)
        out[IR_LOGIC_PLANNING_CONTEXT_KEY] = ir_doc
        return out
    for attr in ("metadata", "context", "extras"):
        bag = getattr(plan_result, attr, None)
        if isinstance(bag, dict):
            bag[IR_LOGIC_PLANNING_CONTEXT_KEY] = ir_doc
            return plan_result
        if bag is not None and hasattr(bag, "__dict__"):
            # immutable mappingproxy etc. — skip
            continue
    # Best-effort: setattr private cache without claiming contract membership
    try:
        object.__setattr__(plan_result, "_ir_logic", ir_doc)
    except Exception:  # noqa: BLE001
        pass
    return plan_result


def surfaces_from_doctor_finding(
    finding: Mapping[str, Any] | Any,
    *,
    domain: str = "doctor",
) -> IrWorkSurface:
    """Project a general doctor finding into an IR work surface."""
    if hasattr(finding, "to_dict") and callable(finding.to_dict):
        raw = dict(finding.to_dict())
    elif isinstance(finding, Mapping):
        raw = dict(finding)
    else:
        raw = {
            "finding_id": str(getattr(finding, "finding_id", "") or getattr(finding, "id", "")),
            "kind": str(getattr(finding, "kind", "") or getattr(finding, "reason_code", "")),
            "contract_id": str(getattr(finding, "contract_id", "")),
            "path": str(getattr(finding, "path", "")),
            "symbol": str(getattr(finding, "symbol", "")),
            "operation": str(getattr(finding, "operation", "")),
        }
    raw.setdefault("domain", domain)
    raw.setdefault("consumer", "doctor")
    return IrWorkSurface.from_mapping(raw)


def diagnose_with_ir_logic(
    finding: Mapping[str, Any] | Any,
    *,
    disposition: Mapping[str, Any] | None = None,
    policy: IrLogicApplyPolicy | Mapping[str, Any] | None = None,
    domain: str = "doctor",
) -> dict[str, Any]:
    """General doctor diagnosis: optional disposition + IR logic application.

    Not SCA-specific. SCA bridge may wrap this with its transform vocabulary.
    """
    surface = surfaces_from_doctor_finding(finding, domain=domain)
    apply_doc = apply_logic_to_ir(
        surface,
        policy=policy,
        consumer="doctor",
        domain=domain,
    )
    return {
        "schema": "ipfs_accelerate_py/agent-supervisor/doctor-ir-diagnosis@1",
        "interface": IR_LOGIC_CONSUMERS_INTERFACE,
        "domain": domain,
        "consumer": "doctor",
        "disposition": dict(disposition or {}),
        "model_call_count": 0,
        "surface": surface.to_dict(),
        IR_LOGIC_DOCTOR_CONTEXT_KEY: {
            "passed": apply_doc.get("passed"),
            "family_ok": apply_doc.get("family_ok"),
            "gates": apply_doc.get("gates"),
            "candidate_plan": apply_doc.get("candidate_plan"),
            "families": {
                name: {
                    "ok": (doc or {}).get("ok"),
                    "status": (doc or {}).get("status"),
                    "logic_applied": (doc or {}).get("logic_applied"),
                }
                for name, doc in (apply_doc.get("families") or {}).items()
            },
        },
        "notes": [
            "Doctor IR application is domain-agnostic intermediate context.",
            "No execution authority is granted from IR apply alone.",
        ],
    }


def probe_ir_logic_consumer_capability() -> dict[str, Any]:
    """Capability probe for default planner / doctor inventory."""
    try:
        sample = apply_logic_to_ir(
            IrWorkSurface(
                operation="capability.probe",
                kind="work_item",
                domain="agent_supervisor",
                consumer="generic",
            ),
            policy=IrLogicApplyPolicy(
                families=DEFAULT_APPLY_FAMILIES,
                evaluate_security=True,
                include_plan_admission=False,
            ),
        )
        return {
            "available": bool(sample.get("passed")),
            "interface": IR_LOGIC_CONSUMERS_INTERFACE,
            "families": list(DEFAULT_APPLY_FAMILIES),
            "family_ok": sample.get("family_ok"),
            "grants_execution_authority": False,
            "domains": [
                "planner",
                "doctor",
                "symbolic_repair",
                "contract_repair",
                "sca",
                "generic",
            ],
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "available": False,
            "interface": IR_LOGIC_CONSUMERS_INTERFACE,
            "error": f"{type(exc).__name__}: {exc}",
            "grants_execution_authority": False,
        }


__all__ = [
    "IR_LOGIC_CONSUMERS_INTERFACE",
    "IR_LOGIC_DOCTOR_CONTEXT_KEY",
    "IR_LOGIC_PLANNING_CONTEXT_KEY",
    "attach_ir_logic_to_symbolic_plan",
    "diagnose_with_ir_logic",
    "enrich_planning_context_with_ir_logic",
    "probe_ir_logic_consumer_capability",
    "surfaces_from_doctor_finding",
    "surfaces_from_planning_context",
]
