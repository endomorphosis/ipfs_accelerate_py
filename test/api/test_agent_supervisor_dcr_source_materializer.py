"""DCR-071: structural source materializer replaces catalog materialization.

Acceptance:
* Successful result contains changed source bytes and reversible diff.
* Writes only operator-rendered edits with exact old-span hash and unique AST
  anchor under the admitted owner worktree.
* Catalog bindings remain evidence, never mutation success.
* Analysis-only / missing / IDL rows and receipt-write failures are nonpassing.
* Runtime model calls remain 0.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.autonomous_repair.edit_plan import (
    DCR_MATERIALIZATION_EVIDENCE,
    SOURCE_EDIT_PLAN_INTERFACE,
    SourceEditPlanDisposition,
    SourceEditPlanError,
    build_source_edit_plan,
    make_source_edit_site,
)
from ipfs_accelerate_py.agent_supervisor.autonomous_repair.materialize import (
    ADMITTED_SOURCE_EDIT_OPERATOR_INTERFACE,
    CODE_EDIT_PACKET_INTERFACE,
    STRUCTURAL_REPAIR_MATERIALIZER_INTERFACE,
    AdmittedSourceEditOperator,
    MaterializeDisposition,
    StructuralMaterializeError,
    StructuralRepairMaterializer,
    apply_operator,
    invert_operator,
    materialize_materialization_vectors,
    materialize_source_edit_plan,
)
from ipfs_accelerate_py.agent_supervisor.proof.code_edit_packet import CodeEditPacket


SOURCE_PATH = "src/handler.py"
OLD_SPAN = "def handler(event):\n    return event\n"
NEW_SPAN = "def handler(event):\n    return normalize(event)\n"
FILE_PREFIX = "# owned surface\n"
FILE_SUFFIX = "\n# trailer\n"


def _sha(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _workspace(tmp_path: Path, body: str | None = None) -> Path:
    root = tmp_path / "owner-worktree"
    root.mkdir()
    target = root / SOURCE_PATH
    target.parent.mkdir(parents=True)
    if body is None:
        body = FILE_PREFIX + OLD_SPAN + FILE_SUFFIX
    target.write_text(body, encoding="utf-8")
    return root


def _site(**overrides: object):
    values: dict[str, object] = {
        "path": SOURCE_PATH,
        "old_span_text": OLD_SPAN,
        "replacement_text": NEW_SPAN,
        "ast_anchor": "src.handler:handler",
        "start_offset": len(FILE_PREFIX),
        "operator_id": "dcr-operator:repair_dispatch_binding@1",
        "operator_args": {"binding": "handler", "mode": "normalize"},
        "unique_anchor": True,
    }
    values.update(overrides)
    return make_source_edit_site(**values)  # type: ignore[arg-type]


def _plan(root: Path, **overrides: object):
    site = overrides.pop("site", None) or _site()
    values: dict[str, object] = {
        "sites": (site,),
        "disposition": SourceEditPlanDisposition.IMPLEMENTABLE,
        "work_id": "work:dcr071",
        "admission_cid": "sha256:" + "cd" * 32,
        "packet_cid": "sha256:" + "ef" * 32,
        "owner_root": "ipfs-accelerate",
        "worktree_root": str(root),
        "implementable": True,
    }
    values.update(overrides)
    return build_source_edit_plan(**values)  # type: ignore[arg-type]


def test_interfaces_and_evidence_are_canonical() -> None:
    assert STRUCTURAL_REPAIR_MATERIALIZER_INTERFACE == "StructuralRepairMaterializer@1"
    assert ADMITTED_SOURCE_EDIT_OPERATOR_INTERFACE == "AdmittedSourceEditOperator@1"
    assert CODE_EDIT_PACKET_INTERFACE == "CodeEditPacket@1"
    assert SOURCE_EDIT_PLAN_INTERFACE == "SourceEditPlan@1"
    assert DCR_MATERIALIZATION_EVIDENCE == "dcr/materialization@1"


def test_site_requires_exact_old_span_hash() -> None:
    with pytest.raises(SourceEditPlanError, match="before_hash"):
        from ipfs_accelerate_py.agent_supervisor.autonomous_repair.edit_plan import (
            SourceEditSite,
        )

        SourceEditSite(
            path=SOURCE_PATH,
            start_offset=0,
            end_offset=len(OLD_SPAN),
            before_hash=_sha("not-the-span"),
            old_span_text=OLD_SPAN,
            replacement_text=NEW_SPAN,
            ast_anchor="src.handler:handler",
        )


def test_site_requires_unique_ast_anchor_shape() -> None:
    with pytest.raises(SourceEditPlanError, match="anchor"):
        make_source_edit_site(
            path=SOURCE_PATH,
            old_span_text=OLD_SPAN,
            replacement_text=NEW_SPAN,
            ast_anchor="not a valid anchor",
        )


def test_apply_operator_writes_source_bytes_and_reversible_diff(tmp_path: Path) -> None:
    root = _workspace(tmp_path)
    site = _site()
    admitted = AdmittedSourceEditOperator(
        operator_id=site.operator_id,
        site=site,
        admission_cid="sha256:" + "11" * 32,
        grants_write_authority=True,
    )
    receipt = apply_operator(admitted, worktree_root=root, plan_id="plan:1", write=True)
    assert receipt.passed is True
    assert receipt.changed_source_bytes is True
    assert receipt.reversible is True
    assert receipt.written is True
    assert receipt.receipt_written is True
    assert receipt.ast_anchor == "src.handler:handler"
    assert receipt.operator_args["binding"] == "handler"
    assert receipt.patch
    assert receipt.inverse_patch
    assert receipt.before_hash != receipt.after_hash
    assert receipt.runtime_model_calls == 0
    body = (root / SOURCE_PATH).read_text(encoding="utf-8")
    assert NEW_SPAN in body
    assert OLD_SPAN not in body


def test_stale_span_is_rejected(tmp_path: Path) -> None:
    root = _workspace(tmp_path, body=FILE_PREFIX + "def other():\n    pass\n" + FILE_SUFFIX)
    site = _site()
    admitted = AdmittedSourceEditOperator(
        operator_id=site.operator_id,
        site=site,
        admission_cid="sha256:" + "22" * 32,
    )
    with pytest.raises(StructuralMaterializeError, match="stale_span"):
        apply_operator(admitted, worktree_root=root, write=True)


def test_path_escape_is_rejected(tmp_path: Path) -> None:
    root = _workspace(tmp_path)
    with pytest.raises(SourceEditPlanError, match="path"):
        make_source_edit_site(
            path="../escape.py",
            old_span_text="x",
            replacement_text="y",
            ast_anchor="escape:x",
        )
    # Missing worktree root is rejected before any write.
    site = _site()
    with pytest.raises(StructuralMaterializeError, match="worktree_root"):
        apply_operator(
            AdmittedSourceEditOperator(
                operator_id=site.operator_id,
                site=site,
                admission_cid="sha256:" + "33" * 32,
            ),
            worktree_root=root / "missing-not-a-dir",
            write=False,
        )


def test_structural_materializer_apply_and_preview(tmp_path: Path) -> None:
    root = _workspace(tmp_path)
    plan = _plan(root)
    materializer = StructuralRepairMaterializer(worktree_root=root)

    preview = materializer.preview(plan)
    assert preview.passed is False
    assert preview.disposition is MaterializeDisposition.PREVIEWED
    assert (root / SOURCE_PATH).read_text(encoding="utf-8").find(OLD_SPAN) >= 0

    applied = materializer.apply(plan)
    assert applied.passed is True
    assert applied.disposition is MaterializeDisposition.APPLIED
    assert applied.runtime_model_calls == 0
    assert applied.plan.implementable is True
    assert len(applied.receipts) == 1
    assert applied.receipts[0].passed is True
    assert isinstance(applied.code_edit_packet, CodeEditPacket)
    assert applied.code_edit_packet.interface == CODE_EDIT_PACKET_INTERFACE
    assert SOURCE_PATH in applied.code_edit_packet.predicted_files
    body = (root / SOURCE_PATH).read_text(encoding="utf-8")
    assert NEW_SPAN in body


def test_inverse_restores_prior_bytes(tmp_path: Path) -> None:
    root = _workspace(tmp_path)
    plan = _plan(root)
    result = materialize_source_edit_plan(plan, worktree_root=root, write=True)
    assert result.passed is True
    receipt = result.receipts[0]
    inverse = invert_operator(receipt, worktree_root=root)
    assert inverse.passed is True
    restored = (root / SOURCE_PATH).read_text(encoding="utf-8")
    assert restored == FILE_PREFIX + OLD_SPAN + FILE_SUFFIX


def test_analysis_only_plan_is_nonpassing(tmp_path: Path) -> None:
    root = _workspace(tmp_path)
    plan = build_source_edit_plan(
        sites=(),
        disposition=SourceEditPlanDisposition.ANALYSIS_ONLY,
        work_id="work:analysis",
        implementable=False,
    )
    result = materialize_source_edit_plan(plan, worktree_root=root, write=True)
    assert result.passed is False
    assert result.disposition is MaterializeDisposition.ANALYSIS_ONLY


def test_missing_and_idl_plans_are_nonpassing(tmp_path: Path) -> None:
    root = _workspace(tmp_path)
    for disposition, expected in (
        (SourceEditPlanDisposition.MISSING_SURFACE, MaterializeDisposition.MISSING_SURFACE),
        (SourceEditPlanDisposition.IDL_GAP, MaterializeDisposition.IDL_GAP),
    ):
        plan = build_source_edit_plan(
            sites=(),
            disposition=disposition,
            work_id=f"work:{disposition.value}",
            implementable=False,
        )
        result = materialize_source_edit_plan(plan, worktree_root=root, write=True)
        assert result.passed is False
        assert result.disposition is expected


def test_catalog_evidence_plan_never_passes(tmp_path: Path) -> None:
    root = _workspace(tmp_path)
    plan = build_source_edit_plan(
        sites=(),
        disposition=SourceEditPlanDisposition.CATALOG_EVIDENCE_ONLY,
        work_id="work:catalog",
        catalog_evidence={"catalog_id": "catalog:surface-1"},
        implementable=False,
    )
    result = materialize_source_edit_plan(plan, worktree_root=root, write=True)
    assert result.passed is False
    assert result.disposition is MaterializeDisposition.CATALOG_EVIDENCE_ONLY
    assert (root / SOURCE_PATH).read_text(encoding="utf-8").find(OLD_SPAN) >= 0


def test_receipt_write_failure_is_nonpassing(tmp_path: Path) -> None:
    root = _workspace(tmp_path)
    plan = _plan(root)
    result = materialize_source_edit_plan(
        plan,
        worktree_root=root,
        write=True,
        force_receipt_failure=True,
    )
    assert result.passed is False
    assert result.disposition is MaterializeDisposition.RECEIPT_WRITE_FAILED
    assert any(
        item.disposition is MaterializeDisposition.RECEIPT_WRITE_FAILED
        for item in result.receipts
    )


def test_ambiguous_anchor_rejected_at_plan() -> None:
    site_a = _site(ast_anchor="src.handler:handler")
    site_b = _site(
        path="src/other.py",
        old_span_text="x = 1\n",
        replacement_text="x = 2\n",
        ast_anchor="src.handler:handler",  # duplicate
        start_offset=0,
    )
    plan = build_source_edit_plan(
        sites=(site_a, site_b),
        disposition=SourceEditPlanDisposition.IMPLEMENTABLE,
        work_id="work:ambiguous",
    )
    assert plan.implementable is False
    assert SourceEditPlanDisposition.AMBIGUOUS_ANCHOR.value in plan.reason_codes


def test_materialize_vectors_catalog_is_evidence_only() -> None:
    catalog = materialize_materialization_vectors(
        [
            {
                "case_id": "happy",
                "passed": True,
                "disposition": MaterializeDisposition.APPLIED.value,
            },
            {
                "case_id": "analysis",
                "passed": False,
                "disposition": MaterializeDisposition.ANALYSIS_ONLY.value,
                "reason_codes": ["analysis_only"],
            },
        ]
    )
    assert catalog["evidence_id"] == DCR_MATERIALIZATION_EVIDENCE
    assert catalog["interface"] == STRUCTURAL_REPAIR_MATERIALIZER_INTERFACE
    assert catalog["runtime_model_calls"] == 0
    assert catalog["acceptance"]["success_requires_changed_source_bytes"] is True
    assert catalog["acceptance"]["catalog_bindings_never_mutation_success"] is True
    assert catalog["acceptance"]["receipt_write_failure_nonpassing"] is True
    by_id = {item["case_id"]: item for item in catalog["vectors"]}
    assert by_id["happy"]["passed"] is True
    assert by_id["analysis"]["passed"] is False


def test_result_roundtrip_preserves_identity(tmp_path: Path) -> None:
    root = _workspace(tmp_path)
    plan = _plan(root)
    result = materialize_source_edit_plan(plan, worktree_root=root, write=True)
    rebuilt = type(result).from_dict(result.to_dict())
    assert rebuilt.passed is True
    assert rebuilt.disposition is MaterializeDisposition.APPLIED
    assert rebuilt.plan.plan_id == result.plan.plan_id
    assert rebuilt.receipts[0].after_hash == result.receipts[0].after_hash


def test_admitted_operator_requires_byte_change() -> None:
    site = _site(replacement_text=OLD_SPAN)
    with pytest.raises(StructuralMaterializeError, match="byte-changing"):
        AdmittedSourceEditOperator(
            operator_id="dcr-operator:rename_alias@1",
            site=site,
            admission_cid="sha256:" + "44" * 32,
        )
