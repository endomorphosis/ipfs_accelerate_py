"""Autonomous repair engine — deterministic, domain-agnostic, no LLM.

Pipeline per work item:

1. Expand interface/ORB/IDL aliases
2. Resolve MCP package surfaces
3. Apply IR logic (intent/legal/security/ui + AST/KG/vector)
4. Doctor transform receipt (model_call_count must stay 0)
5. Emit a repair plan with disposition + ordered deterministic steps

Never marks completion authoritative. Never invents KERNEL_VERIFIED.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .contracts import (
    REPAIR_PLAN_SCHEMA,
    AutonomousRepairPolicy,
    AutonomousRepairReport,
    RepairDisposition,
    RepairWorkItem,
)
from .edit_plan import (
    AdmittedEditPlan,
    materialize_admitted_edit_plan,
    write_edit_plans,
)
from .interface_alias_registry import (
    InterfaceAliasRegistry,
    default_mcp_idl_alias_registry,
)
from .mcp_surface_resolution import (
    SurfaceHit,
    resolve_mcp_surfaces,
)


class AutonomousRepairEngine:
    """Reusable no-LLM repair orchestrator for any supervisor domain."""

    def __init__(
        self,
        *,
        repo_root: str | Path,
        policy: AutonomousRepairPolicy | Mapping[str, Any] | None = None,
        alias_registry: InterfaceAliasRegistry | None = None,
        surface_files: Sequence[tuple[str, str | Path]] | None = None,
        edit_plan_dir: str | Path | None = None,
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.policy = (
            AutonomousRepairPolicy.from_mapping(policy)
            if isinstance(policy, Mapping) or policy is None
            else policy
        )
        self.alias_registry = alias_registry or default_mcp_idl_alias_registry()
        self.surface_files = surface_files
        self.edit_plan_dir = (
            Path(edit_plan_dir)
            if edit_plan_dir
            else self.repo_root / "data" / "agent_supervisor" / "autonomous_repair" / "edit_plans"
        )

    def run(
        self,
        items: Sequence[RepairWorkItem | Mapping[str, Any]],
    ) -> AutonomousRepairReport:
        policy = self.policy
        work_items = [
            item if isinstance(item, RepairWorkItem) else RepairWorkItem.from_mapping(item)
            for item in items
        ][: policy.max_items]

        ops = [w.operation for w in work_items]
        surfaces = resolve_mcp_surfaces(
            ops,
            repo_root=self.repo_root,
            surface_files=self.surface_files,
            alias_registry=self.alias_registry,
            prefer_mcp_server=policy.prefer_mcp_server,
        )
        by_op = surfaces.by_operation()

        model_calls = 0
        rows: list[dict[str, Any]] = []
        edit_plans: list[AdmittedEditPlan] = []
        for work in work_items:
            row = self._process_one(work, by_op.get(work.operation))
            model_calls += int(row.get("model_call_count") or 0)
            if row.get("edit_plan"):
                # re-hydrate for write if present as dict from process
                pass
            rows.append(row)

        # Second stage: body-free admitted edit plans for single/multi path
        for work in work_items:
            surface = by_op.get(work.operation)
            # find matching row
            row = next((r for r in rows if r.get("work_id") == work.work_id), None)
            if not row:
                continue
            plan = materialize_admitted_edit_plan(
                work=work,
                disposition=str(row.get("disposition") or ""),
                surface=surface,
                doctor=row.get("doctor") or {},
                ir_doc={
                    "passed": (row.get("ir_logic") or {}).get("passed"),
                    "family_ok": (row.get("ir_logic") or {}).get("family_ok") or {},
                },
                aliases=row.get("alias_closure") or [],
                idl_methods=row.get("idl_matched_methods") or [],
                allow_code_edit_materialize=policy.allow_code_edit_materialize,
                domain=work.domain or policy.domain,
            )
            if plan is None:
                continue
            edit_plans.append(plan)
            row["edit_plan"] = plan.to_dict()
            row["code_edit_admitted"] = bool(plan.materialize_ready)
            row["edit_plan_id"] = plan.plan_id

        written_paths: list[str] = []
        if edit_plans:
            written_paths = write_edit_plans(edit_plans, output_dir=self.edit_plan_dir)

        analysis_passed = (
            bool(rows)
            and all(
                r.get("disposition") not in {RepairDisposition.BLOCKED.value, None, ""}
                for r in rows
            )
            and model_calls == 0
        )
        # This engine emits analysis and body-free plans only.  It cannot
        # observe an admitted byte-changing source mutation or its validation,
        # so no row may be counted as repair success/completion here.
        passed = False

        disposition_counts: dict[str, int] = {}
        for r in rows:
            d = str(r.get("disposition") or "unknown")
            disposition_counts[d] = disposition_counts.get(d, 0) + 1

        return AutonomousRepairReport(
            policy=policy.to_dict(),
            rows=rows,
            passed=passed,
            model_call_count=model_calls,
            llm_used=model_calls > 0,
            recorded_at=datetime.now(UTC).isoformat(),
            summary={
                "item_count": len(rows),
                "disposition_counts": disposition_counts,
                "surface_resolution": surfaces.to_dict(),
                "single_path_ready": disposition_counts.get(
                    RepairDisposition.SINGLE_PATH_READY.value, 0
                ),
                "multi_path_collapse": disposition_counts.get(
                    RepairDisposition.MULTI_PATH_COLLAPSE.value, 0
                ),
                "missing_surface": disposition_counts.get(
                    RepairDisposition.MISSING_SURFACE.value, 0
                ),
                "idl_gap": disposition_counts.get(RepairDisposition.IDL_GAP.value, 0),
                "edit_plans_count": len(edit_plans),
                "materialize_ready_count": sum(1 for p in edit_plans if p.materialize_ready),
                "edit_plan_dir": str(self.edit_plan_dir),
                "edit_plan_files": written_paths,
                "analysis_passed": analysis_passed,
                "source_edits_applied": 0,
                "validation_pending": 0,
            },
            notes=[
                "Autonomous repair engine is domain-agnostic; consumers set domain.",
                "LLM implement remains forbidden when require_zero_model_calls.",
                "Completion is never authoritative from this engine alone.",
                "Single-path ready items emit body-free admitted edit plans.",
                "Body-free plans never count as applied/success/completed.",
                "Typed source edits remain validation-pending until external re-proof.",
            ],
        )

    def _process_one(
        self,
        work: RepairWorkItem,
        surface: SurfaceHit | None,
    ) -> dict[str, Any]:
        policy = self.policy
        domain = work.domain or policy.domain
        idl_hits = self.alias_registry.match_idl(work.operation)
        alias_closure = sorted(self.alias_registry.expand(work.operation))

        ir_doc: dict[str, Any] = {}
        model_calls = 0
        if policy.apply_ir_logic:
            try:
                from ..proof.ir_logic_application import (
                    IrLogicApplyPolicy,
                    IrWorkSurface,
                    apply_logic_to_ir,
                )

                ir_doc = apply_logic_to_ir(
                    IrWorkSurface(
                        operation=work.operation,
                        kind=work.kind,
                        contract_id=work.contract_id,
                        path=work.path,
                        symbol=work.symbol or work.operation,
                        finding_id=work.work_id,
                        domain=domain,
                        consumer=policy.consumer,
                        metadata=dict(work.metadata),
                    ),
                    policy=IrLogicApplyPolicy(
                        families=policy.ir_families,
                        evaluate_security=True,
                        include_plan_admission=False,
                    ),
                )
            except Exception as exc:  # noqa: BLE001
                ir_doc = {
                    "passed": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }

        doctor: dict[str, Any] = {}
        if policy.apply_doctor:
            # Prefer domain-agnostic doctor IR; SCA bridge only when domain is sca.
            try:
                from ..planning.ir_logic_consumers import diagnose_with_ir_logic

                general_diag = diagnose_with_ir_logic(
                    {
                        "finding_id": work.work_id,
                        "kind": work.kind,
                        "contract_id": work.contract_id,
                        "path": work.path,
                        "operation": work.operation,
                        "symbol": work.symbol or work.operation,
                    },
                    domain=domain,
                )
                doctor = {
                    "disposition": "analysis",
                    "operator": "general_ir_diagnosis",
                    "model_call_count": int(general_diag.get("model_call_count") or 0),
                    "ir_passed": (general_diag.get("ir_logic") or {}).get("passed"),
                    "source": "ir_logic_consumers",
                }
                model_calls += int(doctor.get("model_call_count") or 0)

                if domain in ("sca", "swissknife_contract_assurance"):
                    try:
                        from ..sca_doctor_bridge import diagnose_finding_with_ir

                        sca_diag = diagnose_finding_with_ir(
                            {
                                "finding_id": work.work_id,
                                "kind": work.kind,
                                "contract_id": work.contract_id,
                                "path": work.path,
                                "symbol": work.symbol or work.operation,
                                "snapshot_id": f"auto-repair:{domain}",
                            },
                            domain="sca",
                        )
                        doctor["sca_bridge"] = {
                            "disposition": (sca_diag.get("disposition") or {}).get("disposition"),
                            "operator": (sca_diag.get("disposition") or {}).get("operator"),
                            "ir_passed": (sca_diag.get("ir_logic_apply") or {}).get("passed"),
                        }
                        doctor["model_call_count"] = (
                            int(doctor.get("model_call_count") or 0)
                            + int((sca_diag.get("disposition") or {}).get("model_call_count") or 0)
                            + int(sca_diag.get("model_call_count") or 0)
                        )
                        model_calls = int(doctor["model_call_count"])
                    except Exception as sca_exc:  # noqa: BLE001
                        doctor["sca_bridge"] = {
                            "error": f"{type(sca_exc).__name__}: {sca_exc}",
                        }
            except Exception as exc:  # noqa: BLE001
                doctor = {
                    "disposition": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "model_call_count": 0,
                    "source": "ir_logic_consumers",
                }

        disposition = self._disposition(
            surface=surface,
            idl_hits=idl_hits,
            ir_passed=bool(ir_doc.get("passed")),
            doctor=doctor,
        )
        plan = self._build_plan(
            work=work,
            surface=surface,
            idl_hits=idl_hits,
            disposition=disposition,
            doctor=doctor,
            ir_doc=ir_doc,
        )

        return {
            "work_id": work.work_id,
            "operation": work.operation,
            "contract_id": work.contract_id,
            "package": work.package,
            "kind": work.kind,
            "domain": domain,
            "alias_closure": alias_closure,
            "idl_matched_methods": idl_hits,
            "idl_coverage": bool(idl_hits),
            "surface": surface.to_dict() if surface else None,
            "doctor": doctor,
            "ir_logic": {
                "passed": ir_doc.get("passed"),
                "family_ok": ir_doc.get("family_ok"),
                "error": ir_doc.get("error"),
            },
            "disposition": disposition.value,
            "plan": plan,
            "model_call_count": model_calls,
            "code_edit_admitted": False,  # set after edit_plan stage
            "repair_success": False,
            "completion_disposition": "analysis_only",
            "completion_authoritative": False,
        }

    def _disposition(
        self,
        *,
        surface: SurfaceHit | None,
        idl_hits: list[str],
        ir_passed: bool,
        doctor: Mapping[str, Any],
    ) -> RepairDisposition:
        doc_disp = str(doctor.get("disposition") or "")
        if doc_disp in {"analytical_abstention", "error"} or (
            doctor.get("model_call_count") and self.policy.require_zero_model_calls
        ):
            return RepairDisposition.BLOCKED
        if not ir_passed and self.policy.apply_ir_logic:
            return RepairDisposition.BLOCKED

        status = surface.status if surface else "missing"
        # Prefer effective_match_count after multi-path collapse rules.
        effective = getattr(surface, "effective_match_count", None) if surface else None
        if effective is None:
            effective = surface.match_count if surface else 0
        collapsed = bool(getattr(surface, "collapsed", False)) if surface else False

        if status == "resolved" and (effective == 1 or collapsed):
            return RepairDisposition.SINGLE_PATH_READY
        if status == "resolved" and effective > 1:
            return RepairDisposition.MULTI_PATH_COLLAPSE
        if status == "ambiguous":
            return RepairDisposition.MULTI_PATH_COLLAPSE
        if status == "missing":
            return RepairDisposition.MISSING_SURFACE
        # Surface present-ish but IDL not aligned for GUI/ORB discovery
        if not idl_hits:
            return RepairDisposition.IDL_GAP
        return RepairDisposition.ANALYSIS_ONLY

    def _build_plan(
        self,
        *,
        work: RepairWorkItem,
        surface: SurfaceHit | None,
        idl_hits: list[str],
        disposition: RepairDisposition,
        doctor: Mapping[str, Any],
        ir_doc: Mapping[str, Any],
    ) -> dict[str, Any]:
        steps: list[str] = []
        operator = str(doctor.get("operator") or "analytical_transform")
        steps.append(f"Apply doctor operator `{operator}` (deterministic, zero model calls)")
        if surface and surface.preferred_path:
            steps.append(
                f"Prefer MCP surface anchor: {surface.preferred_path}"
                + (f" handler={surface.handler}" if surface.handler else "")
            )
        if self.policy.prefer_mediation:
            steps.append(
                "Route GUI/ORB effects through package_mcp_interop / tools/call "
                "(or tools_dispatch); never direct cross-package imports"
            )
        if idl_hits:
            steps.append(f"Bind interface/ORB IDL methods: {', '.join(idl_hits)}")
        else:
            steps.append(
                "Extend interface alias registry / IDL descriptor so GUI names "
                f"resolve for operation `{work.operation}`"
            )
        if disposition is RepairDisposition.MULTI_PATH_COLLAPSE:
            steps.append(
                "Collapse multi-match anchors to a single mcp_server register_tool "
                "identity; re-index runtime contract components"
            )
        if disposition is RepairDisposition.MISSING_SURFACE:
            steps.append(
                f"Register missing tool surface for `{work.operation}` via "
                "manager.register_tool / server.register_tool"
            )
        if disposition is RepairDisposition.SINGLE_PATH_READY:
            steps.append(
                "Single-path anchor admitted for deterministic edit packet "
                "(materialize only when policy.allow_code_edit_materialize)"
            )
        steps.append("Re-run surface resolution + IR apply + ready checks")
        steps.append("Bind observation-bound claim receipts; do not forge KERNEL_VERIFIED")
        steps.append("Board completion remains non-authoritative until external re-proof")

        return {
            "schema": REPAIR_PLAN_SCHEMA,
            "work_id": work.work_id,
            "operation": work.operation,
            "disposition": disposition.value,
            "doctor_operator": operator,
            "preferred_path": surface.preferred_path if surface else None,
            "handler": surface.handler if surface else None,
            "idl_methods": idl_hits,
            "ordered_steps": steps,
            "ir_families_ok": ir_doc.get("family_ok") or {},
            "allow_model": False,
            "deterministic_first": True,
            "grants_execution_authority": False,
        }


def run_autonomous_repair(
    items: Sequence[RepairWorkItem | Mapping[str, Any]],
    *,
    repo_root: str | Path,
    policy: AutonomousRepairPolicy | Mapping[str, Any] | None = None,
    alias_registry: InterfaceAliasRegistry | None = None,
    surface_files: Sequence[tuple[str, str | Path]] | None = None,
    edit_plan_dir: str | Path | None = None,
) -> AutonomousRepairReport:
    """Functional entry point for :class:`AutonomousRepairEngine`."""
    return AutonomousRepairEngine(
        repo_root=repo_root,
        policy=policy,
        alias_registry=alias_registry,
        surface_files=surface_files,
        edit_plan_dir=edit_plan_dir,
    ).run(items)


__all__ = [
    "AutonomousRepairEngine",
    "run_autonomous_repair",
]
