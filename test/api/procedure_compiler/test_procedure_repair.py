"""Procedure-guided repair integrates only under explicit independent ceilings."""

from __future__ import annotations

import importlib.util
from dataclasses import replace
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.autonomous_repair.contracts import (
    PostEditValidationReceipt,
    PublicationReceipt,
    RepairAdmissionReceipt,
    RepairAuthorityRoots,
    ReproofReceipt,
)
from ipfs_accelerate_py.agent_supervisor.autonomous_repair.procedure_guided import (
    AutonomousMergeCeiling,
    ProcedureGuidedRepairAdapter,
    ProcedureGuidedRepairRequest,
    ProcedureRepairDisposition,
)
from ipfs_accelerate_py.agent_supervisor.procedure_compiler.contracts import RiskClass
from ipfs_accelerate_py.agent_supervisor.procedure_compiler.registry import (
    InMemoryProcedureRegistryStore,
)


def _helpers():
    path = Path(__file__).with_name("test_registry.py")
    spec = importlib.util.spec_from_file_location("_pcpc027_registry_helpers", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _promoted_registry():
    helpers = _helpers()
    procedure, _candidate, certificate, context = helpers.issue_for()
    registry = helpers.make_registry(context, InMemoryProcedureRegistryStore())
    promoted = helpers.promote_head(
        registry, helpers.register_spec(registry, procedure, certificate), procedure
    )
    return registry, procedure, promoted.revision


def _request(
    revision_id: str, *, procedure_id: str, post_tree: str = "tree-after"
) -> ProcedureGuidedRepairRequest:
    roots = RepairAuthorityRoots(
        repository_id="repo-1",
        repository_forest_cid="forest-1",
        git_tree_id="tree-before",
        policy_root="policy-1",
        rpr_plan_cid="plan-1",
        rpr_packet_cid="packet-1",
    )
    admission = RepairAdmissionReceipt("repair-1", roots, "evidence-1", "derived-1")
    validation = PostEditValidationReceipt(
        "repair-1", roots, "evidence-2", admission.content_id, "mutation-1", True
    )
    reproof = ReproofReceipt(
        "repair-1", roots, "evidence-3", admission.content_id, validation.content_id,
        "mutation-1", True,
    )
    publication = PublicationReceipt(
        "repair-1", roots, "evidence-4", admission.content_id, validation.content_id,
        reproof.content_id, "mutation-1", True,
    )
    return ProcedureGuidedRepairRequest(
        repair_id="repair-1",
        procedure_id=procedure_id,
        expected_revision_id=revision_id,
        patch_digest="sha256:patch-1",
        patch_bytes=80,
        changed_paths=("src/focused.py",),
        lease_id="lease-1",
        merge_authorization_cid="merge-auth-1",
        admission=admission,
        validation=validation,
        reproof=reproof,
        publication=publication,
        isolated_worktree=True,
        symlink_free=True,
        submodule_free=True,
        patch_changes_bytes=True,
        tests_preserved=True,
        tests_passed=True,
        proofs_passed=True,
        post_merge_tree_id=post_tree,
    )


def test_current_promoted_low_risk_procedure_merges_only_after_every_ceiling() -> None:
    registry, procedure, revision = _promoted_registry()
    request = _request(revision.revision_id, procedure_id=procedure.name)
    adapter = ProcedureGuidedRepairAdapter(
        registry,
        ceiling=AutonomousMergeCeiling(
            max_risk=RiskClass.REPOSITORY_WRITE, allowed_paths=("src",)
        ),
    )

    ready = adapter.evaluate(request)
    assert ready.disposition is ProcedureRepairDisposition.MERGE_READY
    assert ready.completion_authoritative is False
    result = adapter.merge(request, merge_executor=lambda _: "tree-after")
    assert result.disposition is ProcedureRepairDisposition.MERGED
    assert result.merged is result.merge_invoked is True
    assert result.completion_authoritative is False
    assert registry.get(procedure.name, demote_stale=False).revision_id == revision.revision_id


def test_stale_high_risk_or_unsafe_boundary_never_invokes_merger() -> None:
    registry, _procedure, revision = _promoted_registry()
    adapter = ProcedureGuidedRepairAdapter(registry)
    calls: list[str] = []
    stale = adapter.merge(
        _request("old-revision", procedure_id=_procedure.name),
        merge_executor=lambda _: calls.append("merged") or "tree-after"
    )
    assert stale.disposition is ProcedureRepairDisposition.REVIEW_REQUIRED
    assert not calls

    high_revision = replace(revision, risk_ceiling=RiskClass.AUTHORITY_OR_SECURITY)
    registry.get = lambda *_args, **_kwargs: high_revision  # type: ignore[method-assign]
    high_risk = ProcedureGuidedRepairAdapter(
        registry, ceiling=AutonomousMergeCeiling(max_risk=RiskClass.REVERSIBLE_LOCAL)
    ).evaluate(_request(high_revision.revision_id, procedure_id=_procedure.name))
    assert high_risk.disposition is ProcedureRepairDisposition.ESCALATED

    unsafe = adapter.evaluate(
        _request(revision.revision_id, procedure_id=_procedure.name, post_tree="")
    )
    assert unsafe.disposition is ProcedureRepairDisposition.REVIEW_REQUIRED
    assert unsafe.completion_authoritative is False
