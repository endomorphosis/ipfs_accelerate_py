"""Focused fail-closed coverage for implementation-site admission."""

from __future__ import annotations

from dataclasses import replace

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.contract_repair_contracts import (
    AuthorityRoots,
    EvidenceReference,
    RepairCandidate,
    RepairStrategy,
    SourceSpan,
    candidate_set_identity,
)
from ipfs_accelerate_py.agent_supervisor.planning.implementation_site_admissibility import (
    ImplementationSiteAdmissibility,
    PlacementDisposition,
    PlacementProposal,
    RepositoryAuthority,
)
from ipfs_accelerate_py.agent_supervisor.proof.contract_repair_obligations import (
    IRClaim,
    ObligationKind,
    PlacementObligation,
    ProofObligation,
)
from ipfs_accelerate_py.agent_supervisor.proof.contract_repair_prover import (
    CandidateProofBundle,
    CandidateProofResult,
    ContractRepairProofDisposition,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    AssuranceLevel,
    CodeProofObligation,
    EvidenceAuthority,
    EvidenceKind,
    EvidenceVerdict,
    ProofEvidence,
    ProofReceipt,
    ProofVerdict,
    ResourceBudget,
)


ROOTS = AuthorityRoots(
    repository_id="repository:test",
    forest_id="forest:test",
    tree_id="tree:test",
    graph_id="graph:test",
    index_id="index:test",
    model_id="model:test",
    config_id="config:test",
    translator_id="translator:test",
    toolchain_id="toolchain:test",
    policy_id="policy:test",
)

PLACEMENT_KINDS = (
    ObligationKind.PLACEMENT_OWNERSHIP,
    ObligationKind.PLACEMENT_NO_OMITTED_COMPATIBLE_IMPLEMENTATION,
    ObligationKind.PLACEMENT_DEPENDENCY_DAG,
    ObligationKind.PLACEMENT_VISIBILITY_REGISTRATION,
    ObligationKind.PLACEMENT_EXACT_STUB_CONTRACT,
)
SUPPORT_KINDS = (
    ObligationKind.EFFECT_COMPATIBILITY,
    ObligationKind.CAPABILITY_COMPATIBILITY,
    ObligationKind.MEMORY_COMPATIBILITY,
)


def ref(kind: str, artifact: str) -> EvidenceReference:
    return EvidenceReference(kind, artifact, producer_id="test")


def candidate(name: str) -> RepairCandidate:
    return RepairCandidate(
        ROOTS,
        f"trace:{name}",
        RepairStrategy.NEW_IMPLEMENTATION,
        SourceSpan(f"pkg/{name}.py", 0, 10, f"blob:{name}"),
        (ref("candidate", f"candidate:{name}"),),
    )


def obligation(item: RepairCandidate, kind: ObligationKind) -> ProofObligation:
    source = ref("reviewed_source", f"source:{item.target_span.path}:{kind.value}")
    claim = IRClaim(
        predicate=kind.value,
        subject_id=item.content_id,
        premise_ids=("premise:one",),
        source_ids=(source.content_id,),
        assumption_ids=("assumption:one",),
        repository_id=ROOTS.repository_id,
        tree_id=ROOTS.tree_id,
        translator_id=ROOTS.translator_id,
        toolchain_id=ROOTS.toolchain_id,
        policy_id=ROOTS.policy_id,
        capability_id="datasets.logic_ir",
        capability_revision="logic:one",
    )
    code = CodeProofObligation(
        repository_id=ROOTS.repository_id,
        repository_tree_id=ROOTS.tree_id,
        ast_scope_ids=(item.target_span.artifact_id,),
        statement=kind.value,
        premise_ids=("premise:one",),
        template_id=f"contract-repair/{kind.value}",
        template_version="1",
        template_semantic_hash=claim.content_id,
        invariant_class="contract_repair",
        task_id="RPR-011",
        required_assurance=AssuranceLevel.KERNEL_VERIFIED,
    )
    return ProofObligation(kind, item.content_id, claim, code, (source,))


def proved_result(item: ProofObligation) -> CandidateProofResult:
    code = item.code_obligation
    evidence = ProofEvidence(
        EvidenceKind.KERNEL_VERIFICATION,
        EvidenceAuthority.KERNEL,
        EvidenceVerdict.ACCEPTED,
        artifact_id=f"kernel-artifact:{item.obligation_id}",
        subject_id=code.obligation_id,
        verifier_id="kernel:test",
        independent=True,
    )
    receipt = ProofReceipt(
        obligation_id=code.obligation_id,
        plan_id="plan:test",
        attempt_id=f"attempt:{item.obligation_id}",
        repository_id=ROOTS.repository_id,
        repository_tree_id=ROOTS.tree_id,
        ast_scope_ids=code.ast_scope_ids,
        premise_ids=code.premise_ids,
        translator_id=ROOTS.translator_id,
        solver_id="hammer",
        kernel_id="kernel:test",
        toolchain_id=ROOTS.toolchain_id,
        policy_id=ROOTS.policy_id,
        resource_budget=ResourceBudget(),
        verdict=ProofVerdict.PROVED,
        evidence=(evidence,),
    )
    return CandidateProofResult(
        item.obligation_id,
        receipt,
        ContractRepairProofDisposition.PROVED,
        ("independent_reconstruction",),
        f"cache:{item.obligation_id}",
    )


def proposal(item: RepairCandidate, candidates: tuple[RepairCandidate, ...]) -> PlacementProposal:
    obligations = tuple(obligation(item, kind) for kind in (*PLACEMENT_KINDS, *SUPPORT_KINDS))
    bundle = CandidateProofBundle(
        item.content_id,
        ROOTS.repository_id,
        ROOTS.tree_id,
        tuple(proved_result(value) for value in obligations),
        "hammer",
        "test",
    )
    authority = RepositoryAuthority(
        ROOTS,
        candidate_set_identity(candidates),
        item.target_span.path,
        "interface:pkg.reader",
        "owner:pkg",
        "anchor:pkg.reader",
        "sender:requirement",
        "sender:requirement",
        (ref("architecture", f"authority:{item.target_span.path}"),),
        ownership_exact=True,
        owner_unambiguous=True,
        write_authorized=True,
        visibility_route_satisfiable=True,
        export_route_satisfiable=True,
        registration_route_satisfiable=True,
        required_effects_supported=True,
        required_capabilities_supported=True,
        memory_policy_supported=True,
    )
    return PlacementProposal(
        item,
        PlacementObligation(obligations[: len(PLACEMENT_KINDS)]),
        obligations[len(PLACEMENT_KINDS) :],
        bundle,
        authority,
    )


def test_admits_one_exact_fully_reconstructed_site_without_granting_writes() -> None:
    item = candidate("receiver")
    decision = ImplementationSiteAdmissibility().decide(
        (proposal(item, (item,)),), candidates=(item,)
    )

    assert decision.disposition is PlacementDisposition.ADMITTED
    assert decision.selected_candidate_id == item.content_id
    assert decision.target_path == "pkg/receiver.py"
    assert decision.write_paths == ()
    assert decision.permitted_write_paths == ()
    assert len(decision.proof_receipt_ids) == len(PLACEMENT_KINDS) + len(SUPPORT_KINDS)


@pytest.mark.parametrize(
    ("change", "reason"),
    (
        ({"external_read_only": True}, "external_read_only_target"),
        ({"generated": True}, "generated_vendor_archive_target"),
        ({"forbidden_layer": True}, "forbidden_dependency_layer"),
        ({"dependency_cycle": True}, "dependency_cycle"),
        ({"owner_unambiguous": False}, "ambiguous_owner"),
        ({"visibility_route_satisfiable": False}, "visibility_route_unsatisfied"),
        ({"export_route_satisfiable": False}, "export_route_unsatisfied"),
        ({"required_capabilities_supported": False}, "required_capabilities_unsupported"),
    ),
)
def test_excludes_read_only_generated_and_unsupportable_sites(
    change: dict[str, bool], reason: str
) -> None:
    item = candidate("receiver")
    value = proposal(item, (item,))
    value = replace(value, repository_authority=replace(value.repository_authority, **change))

    decision = ImplementationSiteAdmissibility().assess((value,), candidates=(item,))

    assert decision.disposition is PlacementDisposition.ABSTAINED
    assert decision.write_paths == ()
    assert reason in decision.reason_codes


def test_requires_exact_stub_support_obligations_and_current_proof_bindings() -> None:
    item = candidate("receiver")
    value = proposal(item, (item,))
    mismatched_stub = replace(
        value,
        repository_authority=replace(
            value.repository_authority, generated_stub_contract_id="sender:other"
        ),
    )
    decision = ImplementationSiteAdmissibility().evaluate((mismatched_stub,), candidates=(item,))
    assert decision.disposition is PlacementDisposition.ABSTAINED
    assert "stub_contract_mismatch" in decision.reason_codes

    missing_memory = replace(value, supporting_obligations=value.supporting_obligations[:-1])
    decision = ImplementationSiteAdmissibility().evaluate((missing_memory,), candidates=(item,))
    assert decision.disposition is PlacementDisposition.ABSTAINED
    assert "missing_required_placement_obligation" in decision.reason_codes

    stale_result = replace(value.proof_bundle.results[0], receipt=replace(
        value.proof_bundle.results[0].receipt,
        freshness="stale",
    ), disposition=ContractRepairProofDisposition.NON_CONCLUSIVE)
    stale_bundle = replace(value.proof_bundle, results=(stale_result, *value.proof_bundle.results[1:]))
    decision = ImplementationSiteAdmissibility().evaluate(
        (replace(value, proof_bundle=stale_bundle),), candidates=(item,)
    )
    assert decision.disposition is PlacementDisposition.ABSTAINED
    assert "placement_proof_not_authoritative" in decision.reason_codes


def test_multiple_equally_admissible_sites_and_candidate_set_drift_abstain() -> None:
    first, second = candidate("first"), candidate("second")
    candidates = (first, second)
    decision = ImplementationSiteAdmissibility().decide(
        (proposal(first, candidates), proposal(second, candidates)), candidates=candidates
    )
    assert decision.disposition is PlacementDisposition.AMBIGUOUS
    assert decision.reason_codes == ("multiple_equal_admissible_sites",)
    assert decision.write_paths == ()

    duplicate = proposal(first, candidates)
    decision = ImplementationSiteAdmissibility().decide((duplicate, duplicate), candidates=candidates)
    assert decision.disposition is PlacementDisposition.ABSTAINED
    assert decision.reason_codes == ("duplicate_placement_proposal",)

    value = proposal(first, candidates)
    drifted = replace(
        value,
        repository_authority=replace(value.repository_authority, candidate_set_id="candidate-set:stale"),
    )
    decision = ImplementationSiteAdmissibility().decide((drifted,), candidates=candidates)
    assert decision.disposition is PlacementDisposition.ABSTAINED
    assert "candidate_set_binding_mismatch" in decision.reason_codes
