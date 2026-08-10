"""DCR-071: source-edit gate — catalog/analysis rows never pass as mutations.

Acceptance:
* A successful result contains changed source bytes and reversible diff.
* Analysis-only / missing / IDL rows are nonpassing.
* Catalog identity bindings are evidence only, never mutation success.
* Receipt-write failures are nonpassing.
* Runtime model calls remain 0.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.autonomous_repair.contracts import (
    AUTONOMOUS_REPAIR_INTERFACE,
    AutonomousRepairPolicy,
    RepairDisposition,
)
from ipfs_accelerate_py.agent_supervisor.autonomous_repair.edit_plan import (
    SourceEditPlanDisposition,
    build_catalog_evidence_plan,
    build_source_edit_plan,
    is_non_mutating_disposition,
    make_source_edit_site,
)
from ipfs_accelerate_py.agent_supervisor.autonomous_repair.engine import (
    ENGINE_INTERFACE,
    AutonomousRepairEngine,
    plan_for_work_item,
    source_edit_gate_allows_pass,
)
from ipfs_accelerate_py.agent_supervisor.autonomous_repair.materialize import (
    STRUCTURAL_REPAIR_MATERIALIZER_INTERFACE,
)


SOURCE_PATH = "pkg/demo_tool.py"
OLD = 'TOOL_NAME = "legacy_tool"\n'
NEW = 'TOOL_NAME = "canonical_tool"\n'


def _workspace(tmp_path: Path, body: str = OLD) -> Path:
    root = tmp_path / "worktree"
    root.mkdir()
    target = root / SOURCE_PATH
    target.parent.mkdir(parents=True)
    target.write_text(body, encoding="utf-8")
    return root


def _structural_item(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "work_id": "work:structural-demo",
        "disposition": RepairDisposition.SINGLE_PATH_READY.value,
        "path": SOURCE_PATH,
        "old_span_text": OLD,
        "replacement_text": NEW,
        "ast_anchor": "pkg.demo_tool:TOOL_NAME",
        "operator_id": "dcr-operator:rename_alias@1",
        "start_offset": 0,
        "unique_anchor": True,
        "admission_cid": "sha256:" + "ab" * 32,
        "implementable": True,
    }
    base.update(overrides)
    return base


def test_engine_interface_is_canonical() -> None:
    assert ENGINE_INTERFACE == AUTONOMOUS_REPAIR_INTERFACE == "AutonomousRepairEngine@1"
    assert STRUCTURAL_REPAIR_MATERIALIZER_INTERFACE == "StructuralRepairMaterializer@1"


def test_structural_source_edit_passes_with_changed_bytes_and_reversible_diff(
    tmp_path: Path,
) -> None:
    root = _workspace(tmp_path)
    engine = AutonomousRepairEngine(
        worktree_root=root,
        policy=AutonomousRepairPolicy(allow_code_edit_materialize=True),
    )
    report = engine.run([_structural_item()])
    assert report.model_call_count == 0
    assert report.llm_used is False
    assert report.passed is True
    assert len(report.rows) == 1
    row = report.rows[0]
    assert row["passed"] is True
    assert row["changed_source_bytes"] is True
    assert row["reversible"] is True
    assert row["receipt_written"] is True
    assert source_edit_gate_allows_pass(row) is True
    assert (root / SOURCE_PATH).read_text(encoding="utf-8") == NEW


@pytest.mark.parametrize(
    "disposition",
    [
        RepairDisposition.ANALYSIS_ONLY.value,
        RepairDisposition.MISSING_SURFACE.value,
        RepairDisposition.IDL_GAP.value,
    ],
)
def test_analysis_missing_idl_rows_are_nonpassing(
    tmp_path: Path,
    disposition: str,
) -> None:
    root = _workspace(tmp_path)
    engine = AutonomousRepairEngine(
        worktree_root=root,
        policy=AutonomousRepairPolicy(allow_code_edit_materialize=True),
    )
    # Even if a site payload is present, non-mutating dispositions cannot pass.
    report = engine.run(
        [
            _structural_item(
                disposition=disposition,
                work_id=f"work:{disposition}",
            )
        ]
    )
    assert report.passed is False
    row = report.rows[0]
    assert row["passed"] is False
    assert source_edit_gate_allows_pass(row) is False
    assert is_non_mutating_disposition(disposition) is True
    # Source must remain untouched.
    assert (root / SOURCE_PATH).read_text(encoding="utf-8") == OLD


def test_catalog_identity_row_never_counts_as_mutation_success(tmp_path: Path) -> None:
    root = _workspace(tmp_path)
    engine = AutonomousRepairEngine(
        worktree_root=root,
        policy=AutonomousRepairPolicy(allow_code_edit_materialize=True),
    )
    report = engine.run(
        [
            {
                "work_id": "work:catalog",
                "disposition": "catalog_only",
                "catalog_only": True,
                "catalog_id": "catalog:surface-identity-1",
                "identity": "surface:demo",
                "path": SOURCE_PATH,
                # Catalog rows historically claimed success without edits.
                "implementable": True,
            }
        ]
    )
    assert report.passed is False
    row = report.rows[0]
    assert row["passed"] is False
    assert row["changed_source_bytes"] is False
    assert source_edit_gate_allows_pass(row) is False
    assert (root / SOURCE_PATH).read_text(encoding="utf-8") == OLD


def test_receipt_write_failure_is_nonpassing(tmp_path: Path) -> None:
    root = _workspace(tmp_path)
    engine = AutonomousRepairEngine(
        worktree_root=root,
        policy=AutonomousRepairPolicy(allow_code_edit_materialize=True),
    )
    report = engine.run([_structural_item()], force_receipt_failure=True)
    assert report.passed is False
    row = report.rows[0]
    assert row["passed"] is False
    assert source_edit_gate_allows_pass(row) is False
    # Source may have been written, but gate still fails without receipt.
    assert "receipt_write_failure_nonpassing" in report.notes


def test_materialize_disabled_cannot_claim_source_mutation_success(
    tmp_path: Path,
) -> None:
    root = _workspace(tmp_path)
    engine = AutonomousRepairEngine(
        worktree_root=root,
        policy=AutonomousRepairPolicy(allow_code_edit_materialize=False),
    )
    report = engine.run([_structural_item()], write=True)
    assert report.passed is False
    assert (root / SOURCE_PATH).read_text(encoding="utf-8") == OLD


def test_plan_for_non_mutating_dispositions_is_not_implementable() -> None:
    for disposition in (
        RepairDisposition.ANALYSIS_ONLY.value,
        RepairDisposition.MISSING_SURFACE.value,
        RepairDisposition.IDL_GAP.value,
    ):
        plan = plan_for_work_item(
            {
                "work_id": f"work:{disposition}",
                "disposition": disposition,
                "path": SOURCE_PATH,
                "old_span_text": OLD,
                "replacement_text": NEW,
                "ast_anchor": "pkg.demo_tool:TOOL_NAME",
            }
        )
        assert plan.implementable is False
        assert plan.claims_source_mutation is False


def test_catalog_evidence_plan_never_claims_mutation() -> None:
    plan = build_catalog_evidence_plan(
        work_id="work:catalog",
        catalog_evidence={"catalog_id": "catalog:1"},
    )
    assert plan.disposition is SourceEditPlanDisposition.CATALOG_EVIDENCE_ONLY
    assert plan.implementable is False
    assert plan.claims_source_mutation is False


def test_mixed_batch_fails_if_any_nonpassing_row(tmp_path: Path) -> None:
    root = _workspace(tmp_path)
    engine = AutonomousRepairEngine(
        worktree_root=root,
        policy=AutonomousRepairPolicy(allow_code_edit_materialize=True),
    )
    report = engine.run(
        [
            _structural_item(work_id="work:ok"),
            {
                "work_id": "work:analysis",
                "disposition": RepairDisposition.ANALYSIS_ONLY.value,
            },
        ]
    )
    assert report.passed is False
    by_id = {row["work_id"]: row for row in report.rows}
    assert by_id["work:ok"]["passed"] is True
    assert by_id["work:analysis"]["passed"] is False


def test_no_byte_change_plan_is_not_implementable() -> None:
    site = make_source_edit_site(
        path=SOURCE_PATH,
        old_span_text=OLD,
        replacement_text=OLD,
        ast_anchor="pkg.demo_tool:TOOL_NAME",
    )
    plan = build_source_edit_plan(
        sites=(site,),
        disposition=SourceEditPlanDisposition.IMPLEMENTABLE,
        work_id="work:noop",
    )
    assert plan.implementable is False
    assert SourceEditPlanDisposition.NO_BYTE_CHANGE.value in plan.reason_codes
