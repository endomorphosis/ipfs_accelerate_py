"""DCR-071: autonomous repair engine gated on structural source edits.

Interface: ``AutonomousRepairEngine@1``

The engine plans and materializes admitted structural source edits.  Catalog
identity rows, analysis-only / missing / IDL dispositions, and receipt-write
failures are always nonpassing.  A successful run requires changed source
bytes plus a reversible diff for every applied site.

Runtime model calls remain 0.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar, Final

from .contracts import (
    AUTONOMOUS_REPAIR_INTERFACE,
    AutonomousRepairPolicy,
    AutonomousRepairReport,
    RepairDisposition,
    RepairWorkItem,
)
from .edit_plan import (
    SourceEditPlan,
    SourceEditPlanDisposition,
    SourceEditSite,
    build_catalog_evidence_plan,
    build_source_edit_plan,
    disposition_for_row,
    is_non_mutating_disposition,
    make_source_edit_site,
)
from .materialize import (
    DCR_MATERIALIZATION_EVIDENCE,
    STRUCTURAL_REPAIR_MATERIALIZER_INTERFACE,
    MaterializationResult,
    MaterializeDisposition,
    StructuralRepairMaterializer,
    materialize_source_edit_plan,
)
from .transaction import (
    FencedWrite,
    MultiRootRepairTransaction,
    PathLeaseBinding,
    RollbackJournal,
    materialize_transaction_receipts,
)


ENGINE_INTERFACE: Final[str] = AUTONOMOUS_REPAIR_INTERFACE
ENGINE_VERSION: Final[int] = 1


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _row_mapping(item: RepairWorkItem | Mapping[str, Any] | Any) -> dict[str, Any]:
    if isinstance(item, RepairWorkItem):
        return item.to_dict()
    if isinstance(item, Mapping):
        return dict(item)
    if hasattr(item, "to_dict") and callable(item.to_dict):
        payload = item.to_dict()
        if isinstance(payload, Mapping):
            return dict(payload)
    raise TypeError("work item must be a mapping or RepairWorkItem")


def _site_from_row(row: Mapping[str, Any]) -> SourceEditSite | None:
    """Extract a structural site when the row carries rendered replacement text."""

    path = str(row.get("path") or (row.get("write_paths") or [""])[0] or "")
    old_span = str(
        row.get("old_span_text")
        or row.get("span_text")
        or row.get("before_text")
        or ""
    )
    replacement = str(
        row.get("replacement_text")
        or row.get("replacement")
        or row.get("after_text")
        or ""
    )
    anchor = str(row.get("ast_anchor") or row.get("anchor") or row.get("symbol") or "")
    if not path or not anchor:
        return None
    if not old_span and not replacement:
        return None
    operator_args = dict(row.get("operator_args") or {})
    if row.get("catalog_only") is True:
        operator_args["catalog_only"] = True
    try:
        return make_source_edit_site(
            path=path,
            old_span_text=old_span,
            replacement_text=replacement,
            ast_anchor=anchor,
            start_offset=int(row.get("start_offset") or row.get("start") or 0),
            operator_id=str(row.get("operator_id") or ""),
            operator_args=operator_args,
            unique_anchor=bool(row.get("unique_anchor", True)),
        )
    except Exception:
        return None


def plan_for_work_item(
    item: RepairWorkItem | Mapping[str, Any],
    *,
    worktree_root: str = "",
) -> SourceEditPlan:
    """Build a fail-closed plan for one work item / engine row."""

    row = _row_mapping(item)
    disposition = disposition_for_row(row)

    if is_non_mutating_disposition(row.get("disposition") or row.get("kind") or disposition.value):
        return build_catalog_evidence_plan(
            work_id=str(row.get("work_id") or row.get("id") or "work:unknown"),
            catalog_evidence={
                key: row[key]
                for key in (
                    "catalog_binding",
                    "catalog_id",
                    "identity",
                    "surface_id",
                    "contract_id",
                    "operation",
                )
                if key in row
            },
            disposition=disposition
            if disposition
            is not SourceEditPlanDisposition.IMPLEMENTABLE
            else SourceEditPlanDisposition.CATALOG_EVIDENCE_ONLY,
            reason_codes=(disposition.value,),
        )

    if row.get("catalog_only") is True or row.get("identity_catalog") is True:
        return build_catalog_evidence_plan(
            work_id=str(row.get("work_id") or "work:catalog"),
            catalog_evidence=dict(row.get("catalog_evidence") or row),
            disposition=SourceEditPlanDisposition.CATALOG_EVIDENCE_ONLY,
            reason_codes=(SourceEditPlanDisposition.CATALOG_EVIDENCE_ONLY.value,),
        )

    site = _site_from_row(row)
    sites: tuple[SourceEditSite, ...] = (site,) if site is not None else ()
    return build_source_edit_plan(
        sites=sites,
        disposition=disposition if sites else SourceEditPlanDisposition.NON_IMPLEMENTABLE,
        work_id=str(row.get("work_id") or row.get("id") or ""),
        packet_cid=str(row.get("packet_cid") or ""),
        operator_cid=str(row.get("operator_cid") or ""),
        owner_root=str(row.get("owner_root") or ""),
        worktree_root=worktree_root or str(row.get("worktree_root") or ""),
        admission_cid=str(row.get("admission_cid") or ""),
        catalog_evidence=dict(row.get("catalog_evidence") or {}),
        row=row,
    )


@dataclass
class AutonomousRepairEngine:
    """Plan + materialize structural source edits; gate success fail-closed."""

    INTERFACE: ClassVar[str] = ENGINE_INTERFACE

    worktree_root: str | Path
    policy: AutonomousRepairPolicy = field(default_factory=AutonomousRepairPolicy)
    receipt_dir: str | Path | None = None

    def __post_init__(self) -> None:
        root = Path(self.worktree_root)
        if not root.is_dir() or root.is_symlink():
            raise ValueError("worktree_root must be an existing non-symlink directory")
        self.worktree_root = root
        if not isinstance(self.policy, AutonomousRepairPolicy):
            self.policy = AutonomousRepairPolicy.from_mapping(
                self.policy if isinstance(self.policy, Mapping) else {}
            )
        if self.receipt_dir is not None:
            self.receipt_dir = Path(self.receipt_dir)

    def plan_items(
        self,
        items: Sequence[RepairWorkItem | Mapping[str, Any]],
    ) -> tuple[SourceEditPlan, ...]:
        limited = tuple(items)[: max(1, int(self.policy.max_items))]
        return tuple(
            plan_for_work_item(item, worktree_root=str(self.worktree_root))
            for item in limited
        )

    def materialize_plan(
        self,
        plan: SourceEditPlan,
        *,
        write: bool = True,
        force_receipt_failure: bool = False,
    ) -> MaterializationResult:
        if not self.policy.allow_code_edit_materialize and write:
            # Fail closed: without the materialize flag, only evidence/preview.
            return StructuralRepairMaterializer(
                worktree_root=self.worktree_root,
                receipt_dir=self.receipt_dir,
            ).preview(plan)
        return materialize_source_edit_plan(
            plan,
            worktree_root=self.worktree_root,
            write=write,
            receipt_dir=self.receipt_dir,
            force_receipt_failure=force_receipt_failure,
        )

    def run(
        self,
        items: Sequence[RepairWorkItem | Mapping[str, Any]],
        *,
        write: bool = True,
        force_receipt_failure: bool = False,
    ) -> AutonomousRepairReport:
        """Run plan+materialize for each item and gate overall success."""

        plans = self.plan_items(items)
        rows: list[dict[str, Any]] = []
        any_pass = False
        any_fail = False
        notes: list[str] = []

        for plan in plans:
            result = self.materialize_plan(
                plan,
                write=write and self.policy.allow_code_edit_materialize,
                force_receipt_failure=force_receipt_failure,
            )
            row_passed = bool(result.passed)
            if row_passed:
                any_pass = True
            else:
                any_fail = True
            # Explicit nonpassing classifications for the source-edit gate.
            if plan.disposition is SourceEditPlanDisposition.ANALYSIS_ONLY:
                row_passed = False
                notes.append("analysis_only_nonpassing")
            elif plan.disposition is SourceEditPlanDisposition.MISSING_SURFACE:
                row_passed = False
                notes.append("missing_surface_nonpassing")
            elif plan.disposition is SourceEditPlanDisposition.IDL_GAP:
                row_passed = False
                notes.append("idl_gap_nonpassing")
            elif plan.disposition is SourceEditPlanDisposition.CATALOG_EVIDENCE_ONLY:
                row_passed = False
                notes.append("catalog_evidence_never_mutation_success")
            elif result.disposition is MaterializeDisposition.RECEIPT_WRITE_FAILED:
                row_passed = False
                notes.append("receipt_write_failure_nonpassing")

            rows.append(
                {
                    "work_id": plan.work_id or plan.plan_id,
                    "plan_id": plan.plan_id,
                    "disposition": plan.disposition.value,
                    "materialize_disposition": result.disposition.value,
                    "implementable": plan.implementable,
                    "passed": row_passed,
                    "claims_source_mutation": plan.claims_source_mutation,
                    "changed_source_bytes": any(
                        receipt.changed_source_bytes for receipt in result.receipts
                    ),
                    "reversible": any(receipt.reversible for receipt in result.receipts),
                    "receipt_written": any(
                        receipt.receipt_written for receipt in result.receipts
                    ),
                    "evidence_subset": result.evidence_subset(),
                    "reason_codes": list(plan.reason_codes),
                    "paths": [site.path for site in plan.sites],
                    "ast_anchors": [site.ast_anchor for site in plan.sites],
                    "catalog_evidence": dict(plan.catalog_evidence),
                    "materializer_interface": STRUCTURAL_REPAIR_MATERIALIZER_INTERFACE,
                    "evidence_id": DCR_MATERIALIZATION_EVIDENCE,
                    "runtime_model_calls": 0,
                }
            )

        # Overall pass requires at least one structural success and zero failures.
        # Catalog/analysis-only-only runs never pass.
        overall_passed = any_pass and not any_fail and bool(rows)
        if not any_pass:
            overall_passed = False

        summary = {
            "item_count": len(rows),
            "passed_count": sum(1 for row in rows if row.get("passed")),
            "failed_count": sum(1 for row in rows if not row.get("passed")),
            "structural_success_count": sum(
                1
                for row in rows
                if row.get("passed")
                and row.get("changed_source_bytes")
                and row.get("reversible")
            ),
            "evidence_id": DCR_MATERIALIZATION_EVIDENCE,
            "materializer_interface": STRUCTURAL_REPAIR_MATERIALIZER_INTERFACE,
            "allow_code_edit_materialize": self.policy.allow_code_edit_materialize,
            "runtime_model_calls": 0,
        }

        return AutonomousRepairReport(
            policy=self.policy.to_dict(),
            rows=rows,
            passed=overall_passed,
            model_call_count=0,
            llm_used=False,
            recorded_at=_utc_now(),
            summary=summary,
            notes=sorted(set(notes)),
        )

    def run_single(
        self,
        item: RepairWorkItem | Mapping[str, Any],
        *,
        write: bool = True,
        force_receipt_failure: bool = False,
    ) -> dict[str, Any]:
        report = self.run(
            (item,),
            write=write,
            force_receipt_failure=force_receipt_failure,
        )
        return report.rows[0] if report.rows else {"passed": False}


def source_edit_gate_allows_pass(row: Mapping[str, Any]) -> bool:
    """Return True only when a row satisfies the DCR-071 source-edit gate."""

    if not isinstance(row, Mapping):
        return False
    if row.get("passed") is not True:
        return False
    disposition = str(row.get("disposition") or "").strip().lower()
    if disposition in {
        RepairDisposition.ANALYSIS_ONLY.value,
        RepairDisposition.MISSING_SURFACE.value,
        RepairDisposition.IDL_GAP.value,
        "catalog_only",
        "catalog_evidence_only",
        "identity_catalog",
        "analysis_only",
        "missing_surface",
        "idl_gap",
    }:
        return False
    if row.get("changed_source_bytes") is not True:
        return False
    if row.get("reversible") is not True:
        return False
    if row.get("receipt_written") is not True:
        return False
    return True


__all__ = [
    "ENGINE_INTERFACE",
    "ENGINE_VERSION",
    "AutonomousRepairEngine",
    "plan_for_work_item",
    "source_edit_gate_allows_pass",
]
