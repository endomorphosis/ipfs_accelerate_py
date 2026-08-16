"""DCR-074 publication proposals are pure evidence checks, never Git effects."""

from __future__ import annotations

from dataclasses import replace

from ipfs_accelerate_py.agent_supervisor.autonomous_repair.contracts import (
    AuthorityStage,
    DeterministicRepairDisposition,
    PostEditValidationReceipt,
    RepairAdmissionReceipt,
    RepairAuthorityRoots,
    RepairEvidenceEnvelope,
    ReproofReceipt,
)
from ipfs_accelerate_py.agent_supervisor.autonomous_repair.operators.registry import (
    OperatorDescriptor,
    OperatorRegistry,
)
from ipfs_accelerate_py.agent_supervisor.autonomous_repair.publish import (
    ObservedCommitRecord,
    ObservedGitlinkRecord,
    PublicationProposalDisposition,
    propose_publication,
)
from ipfs_accelerate_py.agent_supervisor.autonomous_repair.root_ownership import RootBinding
from ipfs_accelerate_py.agent_supervisor.autonomous_repair.validation import (
    PostRepairDisposition,
    RepairProofTransition,
    RepairValidationRoots,
)
from ipfs_accelerate_py.agent_supervisor.planning.proof_carrying_repair_dag import (
    DoctorTransformBinding,
    ProofCarryingRepairPlan,
    RepairPlanDagDisposition,
    RepairPlanDagResult,
    RepairPlanNode,
    RepairPlanNodeKind,
)


def _registry() -> OperatorRegistry:
    descriptor = OperatorDescriptor.from_mapping(
        {
            "operator_id": "publication.operator",
            "kind": "replace_exact_bytes",
            "input_schema": {"type": "object", "required": ["source_digest"], "properties": {"source_digest": "sha256"}, "additional_properties": False},
            "owner_root": "mcp-plus-plus",
            "write_scope": ["provider.py"],
            "before_predicates": ["before"], "after_predicates": ["after"],
            "applicability_proofs": ["proof"],
            "preview": {"kind": "metadata_only", "fields": ["source_digest"]},
            "inverse": {"kind": "restore_exact_before_bytes", "binding": "source_digest"},
            "validation_commands": [["pytest", "provider.py"]],
        }
    )
    return OperatorRegistry((descriptor,), reviewed_manifest={descriptor.operator_id: descriptor.descriptor_id})


def _roots() -> RepairAuthorityRoots:
    return RepairAuthorityRoots("repo", "cid:forest", "tree", "cid:policy", "cid:plan", "cid:packet")


def _reproof_chain(roots: RepairAuthorityRoots) -> tuple[RepairEvidenceEnvelope, RepairEvidenceEnvelope]:
    observed = RepairEvidenceEnvelope("repair", DeterministicRepairDisposition.REFUTED_REPAIRABLE, AuthorityStage.OBSERVED, roots, "cid:observe")
    derived = RepairEvidenceEnvelope("repair", DeterministicRepairDisposition.REFUTED_REPAIRABLE, AuthorityStage.DERIVED, roots, "cid:observe", AuthorityStage.OBSERVED, observed.content_id, "cid:derive")
    admission = RepairAdmissionReceipt("repair", roots, derived.content_id, "cid:derive")
    admitted = RepairEvidenceEnvelope("repair", DeterministicRepairDisposition.REPAIRED_PENDING_VALIDATION, AuthorityStage.ADMITTED, roots, "cid:observe", AuthorityStage.DERIVED, derived.content_id, "cid:derive", admission.content_id, admission_receipt=admission)
    mutated = RepairEvidenceEnvelope("repair", DeterministicRepairDisposition.REPAIRED_PENDING_VALIDATION, AuthorityStage.MUTATED, roots, "cid:observe", AuthorityStage.ADMITTED, admitted.content_id, "cid:derive", admission.content_id, "cid:mutation", admission_receipt=admission)
    validation = PostEditValidationReceipt("repair", roots, mutated.content_id, admission.content_id, "cid:mutation", True)
    post = RepairEvidenceEnvelope("repair", DeterministicRepairDisposition.REPAIRED_PENDING_VALIDATION, AuthorityStage.POST_EDIT_VALIDATED, roots, "cid:observe", AuthorityStage.MUTATED, mutated.content_id, "cid:derive", admission.content_id, "cid:mutation", validation.content_id, admission_receipt=admission, post_edit_validation_receipt=validation)
    reproof = ReproofReceipt("repair", roots, post.content_id, admission.content_id, validation.content_id, "cid:mutation", True)
    reproved = RepairEvidenceEnvelope("repair", DeterministicRepairDisposition.PROVED_VALID, AuthorityStage.REPROVED, roots, "cid:observe", AuthorityStage.POST_EDIT_VALIDATED, post.content_id, "cid:derive", admission.content_id, "cid:mutation", validation.content_id, reproof.content_id, admission_receipt=admission, post_edit_validation_receipt=validation, reproof_receipt=reproof)
    return reproved, post


def _plan(roots: RepairAuthorityRoots) -> tuple[ProofCarryingRepairPlan, RepairPlanDagResult]:
    registry = _registry()
    descriptor = registry.enumerate()[0]
    registry_cid = registry.report()["registry_cid"]
    def node(node_id: str, kind: RepairPlanNodeKind, dependencies: tuple[str, ...] = ()) -> RepairPlanNode:
        return RepairPlanNode(node_id, kind, "mcp-plus-plus", "provider.py", "span", "sha256:before", "after", descriptor, registry_cid, "cid:proof", "cid:gate", "cid:impact", "cid:noninterference", (("pytest", "provider.py"),), "cid:inverse", "cid:rollback", dependencies)
    plan = ProofCarryingRepairPlan(DoctorTransformBinding("cid:dcr051", "cid:dcr052", "cid:doctor"), roots, registry, registry_cid, (node("provider", RepairPlanNodeKind.PROVIDER_COMMIT), node("consumer", RepairPlanNodeKind.CONSUMER_VALIDATION, ("provider",)), node("pin", RepairPlanNodeKind.OUTER_GITLINK_PIN, ("consumer",))))
    return plan, RepairPlanDagResult(RepairPlanDagDisposition.INTEGRATION_PENDING, ("pending",), plan.content_id)


def _inputs():
    roots = _roots()
    reproved, post = _reproof_chain(roots)
    plan, dag = _plan(roots)
    before = RepairValidationRoots("cid:forest", "cid:g0", "cid:e0", "cid:f0", "cid:p0")
    after = RepairValidationRoots("cid:forest2", "cid:g1", "cid:e1", "cid:f1", "cid:p1")
    validation = RepairProofTransition(PostRepairDisposition.INTEGRATION_PENDING, ("pending",), before, after, ("cid:detector",))
    provider_before, provider_after = "a" * 40, "b" * 40
    consumer_before, consumer_after = "c" * 40, "d" * 40
    bindings = (RootBinding("mcp-plus-plus", "/tmp/provider", provider_before, "e" * 40, "sha256:clean", False), RootBinding("swissknife", "/tmp/consumer", consumer_before, "f" * 40, "sha256:clean", False))
    commits = (ObservedCommitRecord("mcp-plus-plus", provider_before, provider_after, "1" * 40, (provider_before,), "sha256:diff", "cid:operator", validation.transition_cid, reproved.reproof_cid), ObservedCommitRecord("swissknife", consumer_before, consumer_after, "2" * 40, (consumer_before,), "sha256:diff2", "cid:operator", validation.transition_cid, reproved.reproof_cid))
    gitlink = ObservedGitlinkRecord("swissknife", "mcp-plus-plus", "Mcp-Plus-Plus", provider_before, provider_after)
    return validation, reproved, post, bindings, plan, dag, commits, gitlink


def test_ordered_observed_proposal_is_pending_and_has_zero_effects() -> None:
    value = _inputs()
    proposal = propose_publication(validation=value[0], reproved=value[1], immediate_predecessor=value[2], root_bindings=value[3], plan=value[4], dag_result=value[5], commits=value[6], gitlink=value[7])
    assert proposal.disposition is PublicationProposalDisposition.INTEGRATION_PENDING
    assert proposal.to_dict()["git_call_count"] == proposal.to_dict()["network_call_count"] == 0
    assert proposal.to_dict()["publication_authorized"] is False


def test_head_drift_forged_chain_premature_pin_and_dirty_overlay_do_not_propose() -> None:
    value = _inputs()
    drift = replace(value[6][0], predecessor_head="9" * 40, parent_heads=("9" * 40,))
    proposal = propose_publication(validation=value[0], reproved=value[1], immediate_predecessor=value[2], root_bindings=value[3], plan=value[4], dag_result=value[5], commits=(drift, value[6][1]), gitlink=value[7])
    assert proposal.disposition is PublicationProposalDisposition.STALE
    forged = replace(value[1], previous_envelope_cid="cid:forged")
    assert propose_publication(validation=value[0], reproved=forged, immediate_predecessor=value[2], root_bindings=value[3], plan=value[4], dag_result=value[5], commits=value[6], gitlink=value[7]).disposition is PublicationProposalDisposition.REJECTED
    dirty = (replace(value[3][0], dirty=True), value[3][1])
    assert propose_publication(validation=value[0], reproved=value[1], immediate_predecessor=value[2], root_bindings=dirty, plan=value[4], dag_result=value[5], commits=value[6], gitlink=value[7]).disposition is PublicationProposalDisposition.REJECTED
    plan = replace(
        value[4],
        nodes=tuple(
            replace(node, dependencies=())
            if node.kind is RepairPlanNodeKind.OUTER_GITLINK_PIN
            else node
            for node in value[4].nodes
        ),
    )
    assert propose_publication(validation=value[0], reproved=value[1], immediate_predecessor=value[2], root_bindings=value[3], plan=plan, dag_result=RepairPlanDagResult(RepairPlanDagDisposition.INTEGRATION_PENDING, ("pending",), plan.content_id), commits=value[6], gitlink=value[7]).disposition is PublicationProposalDisposition.REPLAN
