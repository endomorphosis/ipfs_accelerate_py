"""Deep hooks: IR logic → planner / doctor / formal-plan intermediate paths.

These hooks do more than side-channel metadata.  They:

* **Prepare** planning context so frozen symbolic requests bind IR receipts
* **Project** IR apply results into formal-plan compiler channels
  (AST / policy / evidence / task records)
* **Compose** adaptive hard-gates that fail closed when IR is required
* **Enrich** doctor synthesis requests with IR intermediate context
* Remain **domain-agnostic** (planner, doctor, repair, SCA, generic)

Authority: IR hooks never grant execution; security evaluation and empty
grant sources remain fail-closed.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Callable, Final, Mapping, Sequence

from ..proof.ir_logic_application import (
    DEFAULT_APPLY_FAMILIES,
    IrLogicApplyPolicy,
    IrLogicRequiredGateResult,
    apply_logic_to_surfaces,
    evaluate_required_ir_logic_gate,
)
from .ir_logic_consumers import (
    IR_LOGIC_DOCTOR_CONTEXT_KEY,
    IR_LOGIC_PLANNING_CONTEXT_KEY,
    diagnose_with_ir_logic,
    enrich_planning_context_with_ir_logic,
)

IR_LOGIC_HOOKS_INTERFACE: Final = "IrLogicHooks@1"
IR_LOGIC_COMPILER_CHANNEL_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/ir-logic-compiler-channels@1"
)
IR_LOGIC_HARD_GATE_ID: Final = "ir_logic_application"


def _should_apply(ctx: Mapping[str, Any], *, default_when_surfaces: bool = True) -> bool:
    flag = ctx.get("apply_ir_logic")
    if flag is False:
        return False
    if flag is True:
        return True
    if not default_when_surfaces:
        return False
    return any(
        ctx.get(key)
        for key in (
            "ir_work_surfaces",
            "work_surfaces",
            "operations",
            "operation",
            "primary_operation",
            "findings",
            "path",
            "symbol",
            "contract_id",
            "goal_id",
        )
    )


def prepare_planning_context(
    context: Mapping[str, Any] | None,
    *,
    domain: str = "planner",
    policy: IrLogicApplyPolicy | Mapping[str, Any] | None = None,
    max_surfaces: int = 8,
    force: bool = False,
) -> dict[str, Any]:
    """Return a new planning context with IR logic applied and bound.

    The frozen symbolic candidate request stores this context, so IR receipts
    become part of the planning intermediate representation identity.
    """
    ctx = dict(context or {})
    if not force and not _should_apply(ctx):
        ctx.setdefault(IR_LOGIC_PLANNING_CONTEXT_KEY, {
            "skipped": True,
            "reason": "apply_ir_logic_not_requested",
            "interface": IR_LOGIC_HOOKS_INTERFACE,
        })
        return ctx

    domain = str(ctx.get("domain") or domain)
    ir_doc = enrich_planning_context_with_ir_logic(
        ctx,
        policy=policy,
        domain=domain,
        max_surfaces=max_surfaces,
    )
    ctx[IR_LOGIC_PLANNING_CONTEXT_KEY] = ir_doc
    ctx["ir_logic_bound"] = True
    ctx["ir_logic_passed"] = bool(ir_doc.get("passed"))
    # Promote compact family status for gates / templates
    family_ok: dict[str, bool] = {}
    for row in ir_doc.get("rows") or []:
        for name, ok in (row.get("family_ok") or {}).items():
            family_ok[name] = bool(family_ok.get(name, True)) and bool(ok)
    if not family_ok and isinstance(ir_doc.get("summary"), Mapping):
        # batch summary path
        summary = ir_doc.get("summary") or {}
        mapping = {
            "intent_ir": "intent_ok",
            "legal_ir": "legal_ok",
            "security_ir": "security_ok",
            "ui_ir": "ui_ok",
            "ast": "ast_ok",
            "knowledge_graph": "knowledge_graph_ok",
            "vector_index": "vector_index_ok",
        }
        for fam, sk in mapping.items():
            if sk in summary:
                family_ok[fam] = int(summary.get(sk) or 0) > 0
    ctx["ir_logic_family_ok"] = family_ok
    return ctx


def project_compiler_channels_from_ir(
    ir_doc: Mapping[str, Any] | None,
    *,
    domain: str = "planner",
) -> dict[str, list[dict[str, Any]]]:
    """Project IR apply receipts into FormalPlanCompiler record channels.

    Channels:
    * ``ast_records`` — path/symbol intermediate IR
    * ``policy_records`` — security/legal constraint summaries
    * ``evidence_records`` — IR family apply evidence (non-authoritative)
    * ``task_records`` — intent-derived repair/index/verify actions
    * ``objective_records`` — goal/operation objectives
    """
    ir_doc = dict(ir_doc or {})
    rows = list(ir_doc.get("rows") or [])
    if not rows and ir_doc.get("families"):
        # single-surface apply shape
        rows = [ir_doc]

    ast_records: list[dict[str, Any]] = []
    policy_records: list[dict[str, Any]] = []
    evidence_records: list[dict[str, Any]] = []
    task_records: list[dict[str, Any]] = []
    objective_records: list[dict[str, Any]] = []

    for row in rows:
        if not isinstance(row, Mapping):
            continue
        op = str(row.get("operation") or "unknown")
        contract_id = str(row.get("contract_id") or f"surface:{op}")
        families = row.get("families") or {}
        plan = row.get("candidate_plan") or {}

        objective_records.append(
            {
                "objective_id": f"obj:ir:{op}",
                "kind": "ir_bound_work",
                "operation": op,
                "contract_id": contract_id,
                "domain": domain,
                "source": "ir_logic_hooks",
            }
        )

        for action_id in plan.get("action_ids") or []:
            task_records.append(
                {
                    "task_id": str(action_id),
                    "operation": op,
                    "contract_id": contract_id,
                    "kind": "ir_projected_action",
                    "domain": domain,
                    "source": "ir_logic_hooks",
                }
            )

        ast_fam = families.get("ast") or {}
        if isinstance(ast_fam, Mapping) and ast_fam.get("ok"):
            rec = (
                ast_fam.get("record") or {}
                if isinstance(ast_fam.get("record"), Mapping)
                else {}
            )
            ast_records.append(
                {
                    "path": rec.get("path") or f"agent_supervisor/work_surfaces/{domain}/{op}.py",
                    "symbols": list(rec.get("symbols") or []),
                    "calls": list(rec.get("calls") or []),
                    "blob_identity": rec.get("blob_identity"),
                    "index_id": ast_fam.get("index_id"),
                    "operation": op,
                    "source": "ir_logic_ast",
                    "domain": domain,
                }
            )

        for fam_name in ("security_ir", "legal_ir", "intent_ir", "ui_ir"):
            fam = families.get(fam_name) or {}
            if not isinstance(fam, Mapping) or not fam.get("ok"):
                continue
            policy_records.append(
                {
                    "policy_id": f"policy:{fam_name}:{op}",
                    "family": fam_name,
                    "status": fam.get("status"),
                    "role": fam.get("role"),
                    "constraint_count": fam.get("constraint_count"),
                    "grants_execution_authority": bool(
                        fam.get("grants_execution_authority")
                    ),
                    "operation": op,
                    "domain": domain,
                    "source": "ir_logic_hooks",
                    "authoritative": False,
                }
            )

        for fam_name, fam in families.items():
            if not isinstance(fam, Mapping):
                continue
            evidence_records.append(
                {
                    "evidence_id": f"ev:ir:{fam_name}:{op}",
                    "kind": "ir_logic_application",
                    "family": fam_name,
                    "status": fam.get("status"),
                    "ok": bool(fam.get("ok")),
                    "logic_applied": list(fam.get("logic_applied") or []),
                    "authoritative": False,
                    "grants_execution_authority": bool(
                        fam.get("grants_execution_authority")
                    ),
                    "operation": op,
                    "domain": domain,
                    "source": "ir_logic_hooks",
                }
            )

        # structural graph / vector as evidence
        kg = families.get("knowledge_graph") or {}
        if isinstance(kg, Mapping) and kg.get("ok"):
            evidence_records.append(
                {
                    "evidence_id": f"ev:ir:kg:{op}",
                    "kind": "knowledge_graph_closure",
                    "node_count": kg.get("node_count"),
                    "edge_count": kg.get("edge_count"),
                    "closure_node_count": (kg.get("closure") or {}).get("node_count")
                    if isinstance(kg.get("closure"), Mapping)
                    else None,
                    "authoritative": False,
                    "operation": op,
                    "domain": domain,
                    "source": "ir_logic_hooks",
                }
            )
        vec = families.get("vector_index") or {}
        if isinstance(vec, Mapping) and vec.get("ok"):
            evidence_records.append(
                {
                    "evidence_id": f"ev:ir:vec:{op}",
                    "kind": "vector_index_hits",
                    "row_count": vec.get("row_count"),
                    "hit_count": len(vec.get("hits") or []),
                    "model_id": vec.get("model_id"),
                    "authoritative": False,
                    "semantic_authority": False,
                    "operation": op,
                    "domain": domain,
                    "source": "ir_logic_hooks",
                }
            )

    return {
        "schema": IR_LOGIC_COMPILER_CHANNEL_SCHEMA,
        "interface": IR_LOGIC_HOOKS_INTERFACE,
        "domain": domain,
        "objective_records": objective_records,
        "task_records": task_records,
        "ast_records": ast_records,
        "policy_records": policy_records,
        "evidence_records": evidence_records,
    }


def inject_ir_into_formal_plan_source(
    source: Mapping[str, Any] | None,
    *,
    domain: str = "planner",
    policy: IrLogicApplyPolicy | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Deep hook for FormalPlanCompiler: merge IR channels into compile source."""
    payload = dict(source or {})
    if payload.get("apply_ir_logic") is False:
        return payload

    # Build surfaces from source operations / objectives / tasks
    surfaces: list[Any] = []
    for key in ("ir_work_surfaces", "work_surfaces", "operations", "findings"):
        items = payload.get(key)
        if isinstance(items, (list, tuple)):
            surfaces.extend(items)
    if not surfaces:
        op = str(
            payload.get("operation")
            or payload.get("primary_operation")
            or ""
        )
        if not op:
            for rec in payload.get("objective_records") or payload.get("objectives") or ():
                if isinstance(rec, Mapping):
                    op = str(
                        rec.get("operation")
                        or rec.get("name")
                        or rec.get("objective_id")
                        or ""
                    )
                    if op:
                        break
        if op:
            surfaces = [
                {
                    "operation": op,
                    "kind": str(payload.get("kind") or "work_item"),
                    "path": str(payload.get("path") or ""),
                    "symbol": str(payload.get("symbol") or ""),
                    "domain": payload.get("domain") or domain,
                    "contract_id": str(payload.get("contract_id") or ""),
                }
            ]

    if not surfaces and not payload.get("apply_ir_logic"):
        return payload

    domain = str(payload.get("domain") or domain)
    ir_doc = apply_logic_to_surfaces(
        surfaces,
        policy=policy,
        domain=domain,
        consumer="planner",
    )
    channels = project_compiler_channels_from_ir(ir_doc, domain=domain)

    def _merge(key: str, values: list[dict[str, Any]]) -> None:
        if not values:
            return
        existing = list(payload.get(key) or [])
        # de-dupe by simple id fields
        seen = set()
        merged = []
        for item in existing + values:
            if not isinstance(item, Mapping):
                continue
            token = str(
                item.get("evidence_id")
                or item.get("policy_id")
                or item.get("task_id")
                or item.get("objective_id")
                or item.get("path")
                or id(item)
            )
            if token in seen:
                continue
            seen.add(token)
            merged.append(dict(item))
        payload[key] = merged

    _merge("objective_records", channels["objective_records"])
    _merge("task_records", channels["task_records"])
    _merge("ast_records", channels["ast_records"])
    _merge("policy_records", channels["policy_records"])
    _merge("evidence_records", channels["evidence_records"])
    payload[IR_LOGIC_PLANNING_CONTEXT_KEY] = {
        "passed": ir_doc.get("passed"),
        "summary": ir_doc.get("summary"),
        "selected_count": ir_doc.get("selected_count"),
        "domain": domain,
        "hook": "inject_ir_into_formal_plan_source",
        "interface": IR_LOGIC_HOOKS_INTERFACE,
    }
    payload["ir_logic_bound"] = True
    return payload


def compose_hard_gate_with_ir(
    base_gate: Callable[..., Any] | None = None,
    *,
    context: Mapping[str, Any] | None = None,
    require_ir: bool | None = None,
) -> Callable[..., Any]:
    """Compose adaptive hard-gate evaluator with IR application gate.

    When IR is required (context.require_ir_logic or require_ir=True), candidates
    fail closed if planning context did not bind a passing IR apply receipt.
    When IR is optional, the gate records advisory evidence only.
    """
    from .adaptive_planner import (
        GateProducerKind,
        HardConstraintReceipt,
        HardPlanConstraint,
        adaptive_plan_candidate_snapshot_id,
        deterministic_hard_gate_receipts,
    )

    base = base_gate or deterministic_hard_gate_receipts
    ctx = dict(context or {})
    if require_ir is None:
        require_ir = bool(ctx.get("require_ir_logic"))

    def _evaluator(plan, frozen_goal, request):
        receipts = list(base(plan, frozen_goal, request) or ())
        ir = ctx.get(IR_LOGIC_PLANNING_CONTEXT_KEY) or {}
        bound = bool(ctx.get("ir_logic_bound")) or bool(ir)
        ir_ok = bool(ir.get("passed")) if isinstance(ir, Mapping) else False

        # When IR was never requested and not bound, leave baseline gates alone.
        if not require_ir and not bound:
            return tuple(receipts)

        snapshot_id = adaptive_plan_candidate_snapshot_id(
            plan,
            goal_content_id=frozen_goal.goal_content_id,
            repository_tree_id=frozen_goal.repository_tree_id,
            policy_digest=frozen_goal.policy_digest,
        )
        if require_ir and not bound:
            ir_passed = False
            reasons = ("ir_logic_not_bound",)
        elif require_ir and bound and not ir_ok:
            ir_passed = False
            reasons = ("ir_logic_application_failed",)
        else:
            # Required+passed, or optional+bound (advisory always passes gate)
            ir_passed = True
            reasons = (
                ("ir_logic_bound",)
                if bound and ir_ok
                else (("ir_logic_bound_partial",) if bound else ("ir_logic_optional_skip",))
            )

        branch = getattr(plan, "branch", None)
        candidate_id = str(
            getattr(plan, "candidate_id", None)
            or getattr(branch, "branch_id", None)
            or getattr(plan, "branch_id", None)
            or "candidate"
        )
        evidence_ids = tuple(
            f"ir:{name}"
            for name, ok in (ctx.get("ir_logic_family_ok") or {}).items()
            if ok
        )
        receipts.append(
            HardConstraintReceipt(
                constraint=HardPlanConstraint.SAFETY,
                candidate_id=candidate_id,
                candidate_snapshot_id=snapshot_id,
                goal_content_id=frozen_goal.goal_content_id,
                repository_tree_id=frozen_goal.repository_tree_id,
                policy_digest=frozen_goal.policy_digest,
                passed=ir_passed if require_ir else True,
                producer_kind=GateProducerKind.FORMAL_VALIDATOR,
                producer_id=IR_LOGIC_HARD_GATE_ID,
                evidence_ids=evidence_ids,
                reason_codes=reasons,
            )
        )
        return tuple(receipts)

    return _evaluator


def attach_ir_logic_to_doctor_request(
    request: Any,
    *,
    domain: str = "doctor",
    policy: IrLogicApplyPolicy | Mapping[str, Any] | None = None,
) -> Any:
    """Deep hook: bind IR intermediate context onto a DoctorSynthesisRequest."""
    if request is None:
        return request
    meta = dict(getattr(request, "metadata", None) or {})
    if meta.get("apply_ir_logic") is False:
        return request
    if meta.get(IR_LOGIC_DOCTOR_CONTEXT_KEY) and meta.get("ir_logic_bound"):
        return request

    finding = {
        "finding_id": str(getattr(request, "finding_id", "") or meta.get("finding_id") or ""),
        "kind": str(meta.get("kind") or meta.get("finding_kind") or "work_item"),
        "path": str(getattr(request, "path", None) or meta.get("path") or ""),
        "symbol": str(meta.get("symbol") or ""),
        "operation": str(meta.get("operation") or meta.get("operator") or ""),
        "contract_id": str(meta.get("contract_id") or ""),
    }
    # Prefer proposal path if present
    proposal = getattr(request, "proposal", None)
    if proposal is not None and not finding["path"]:
        finding["path"] = str(getattr(proposal, "path", "") or "")
    if proposal is not None and not finding["operation"]:
        finding["operation"] = str(
            getattr(proposal, "operator_id", None)
            or getattr(proposal, "operator_kind", None)
            or meta.get("operator_id")
            or "doctor.repair"
        )
    if not finding["operation"]:
        finding["operation"] = (
            finding["path"].rsplit("/", 1)[-1]
            if finding["path"]
            else "doctor.work"
        )

    try:
        diag = diagnose_with_ir_logic(
            finding,
            policy=policy,
            domain=str(meta.get("domain") or domain),
        )
        meta[IR_LOGIC_DOCTOR_CONTEXT_KEY] = diag.get(IR_LOGIC_DOCTOR_CONTEXT_KEY) or {}
        meta["ir_logic_bound"] = True
        meta["ir_logic_passed"] = bool(
            (diag.get(IR_LOGIC_DOCTOR_CONTEXT_KEY) or {}).get("passed")
        )
        meta["ir_logic_domain"] = str(meta.get("domain") or domain)
        object.__setattr__(request, "metadata", MappingProxyType(meta))
    except Exception as exc:  # noqa: BLE001
        meta[IR_LOGIC_DOCTOR_CONTEXT_KEY] = {
            "passed": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
        meta["ir_logic_bound"] = False
        try:
            object.__setattr__(request, "metadata", MappingProxyType(meta))
        except Exception:  # noqa: BLE001
            pass
    return request


def ir_validation_findings(
    plan: Any,
    *,
    require_ir: bool = False,
) -> list[dict[str, Any]]:
    """Deep hook for FormalPlanValidator: findings about IR binding on plan.metadata."""
    findings: list[dict[str, Any]] = []
    metadata = getattr(plan, "metadata", None)
    if isinstance(plan, Mapping):
        metadata = plan.get("metadata")
    meta = dict(metadata or {}) if isinstance(metadata, Mapping) else {}
    ir = meta.get(IR_LOGIC_PLANNING_CONTEXT_KEY) or meta.get("ir_logic") or {}
    bound = bool(meta.get("ir_logic_bound") or ir)
    passed = bool(ir.get("passed")) if isinstance(ir, Mapping) else False

    if require_ir or meta.get("require_ir_logic"):
        if not bound:
            findings.append(
                {
                    "code": "ir_logic_not_bound",
                    "severity": "error",
                    "message": "plan requires IR logic application but none was bound",
                    "authoritative": False,
                }
            )
        elif not passed:
            findings.append(
                {
                    "code": "ir_logic_application_failed",
                    "severity": "error",
                    "message": "bound IR logic application did not pass",
                    "authoritative": False,
                }
            )
        else:
            findings.append(
                {
                    "code": "ir_logic_bound",
                    "severity": "info",
                    "message": "IR logic families applied to intermediate representation",
                    "authoritative": False,
                    "families": list(DEFAULT_APPLY_FAMILIES),
                }
            )
    elif bound:
        findings.append(
            {
                "code": "ir_logic_advisory",
                "severity": "info",
                "message": "IR logic present on plan metadata (advisory)",
                "passed": passed,
                "authoritative": False,
            }
        )
    return findings


def symbolic_repair_ir_portfolio_bind(
    planning_report: Mapping[str, Any] | None,
    ir_apply_report: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Deep bind between SCA/general symbolic repair planning and IR apply stage."""
    plan = dict(planning_report or {})
    ir = dict(ir_apply_report or {})
    portfolios = list(plan.get("portfolios") or [])
    ir_rows = {
        str(row.get("operation") or ""): row
        for row in (ir.get("rows") or [])
        if isinstance(row, Mapping)
    }
    bound = []
    for pf in portfolios:
        if not isinstance(pf, Mapping):
            continue
        item = dict(pf)
        op = str(item.get("operation") or "")
        if op and op in ir_rows and not item.get("ir_logic_apply"):
            row = ir_rows[op]
            item["ir_logic_apply"] = {
                "passed": row.get("passed"),
                "family_ok": row.get("family_ok"),
                "gates": row.get("gates"),
                "candidate_plan": row.get("candidate_plan"),
                "source": "symbolic_repair_ir_portfolio_bind",
            }
            item["ir_logic_bound"] = True
        bound.append(item)
    plan["portfolios"] = bound
    plan["ir_apply_summary"] = ir.get("summary")
    plan["ir_logic_deep_bound"] = True
    plan["ir_logic_interface"] = IR_LOGIC_HOOKS_INTERFACE
    return plan


def evaluate_required_ir_logic_hook_gate(
    context: Mapping[str, Any] | None,
) -> IrLogicRequiredGateResult:
    """Evaluate the opt-in DCR-035 gate without changing legacy hook policy.

    Absent receipt/identity fields intentionally flow to the typed
    ``integration_pending`` result; no legacy default flag can turn them into
    approval.
    """

    payload = dict(context or {})
    raw_receipts = payload.get("dcr035_stage_receipts") or ()
    if not isinstance(raw_receipts, Sequence) or isinstance(raw_receipts, (str, bytes)):
        raw_receipts = ()
    identities = payload.get("dcr035_identity_cids")
    if not isinstance(identities, Mapping):
        identities = {}
    return evaluate_required_ir_logic_gate(
        tuple(raw_receipts),
        required_identity_cids=identities,
    )


__all__ = [
    "IR_LOGIC_COMPILER_CHANNEL_SCHEMA",
    "IR_LOGIC_HARD_GATE_ID",
    "IR_LOGIC_HOOKS_INTERFACE",
    "attach_ir_logic_to_doctor_request",
    "compose_hard_gate_with_ir",
    "evaluate_required_ir_logic_hook_gate",
    "inject_ir_into_formal_plan_source",
    "ir_validation_findings",
    "prepare_planning_context",
    "project_compiler_channels_from_ir",
    "symbolic_repair_ir_portfolio_bind",
]
