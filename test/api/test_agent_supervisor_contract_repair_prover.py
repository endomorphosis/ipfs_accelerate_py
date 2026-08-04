"""Fail-closed smoke coverage for contract-repair proof orchestration."""

from __future__ import annotations

from ipfs_accelerate_py.agent_supervisor.proof.contract_repair_prover import (
    ContractRepairProofDisposition,
    ContractRepairProver,
)
from ipfs_accelerate_py.agent_supervisor.analysis.contract_repair_contracts import (
    EvidenceReference,
)
from ipfs_accelerate_py.agent_supervisor.proof.contract_repair_obligations import (
    IRClaim,
    ObligationKind,
    ProofObligation,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    AssuranceLevel,
    CodeProofObligation,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_capabilities import (
    ProofProviderCapability,
    ProofProviderIsolation,
    ProofProviderOperation,
)


class _CandidateOnlyBackend:
    provider_id = "hammer"
    provider_version = "test"

    def capabilities(self):
        return ProofProviderCapability(
            provider_id=self.provider_id, provider_version=self.provider_version,
            protocol_versions=(1,), operations=(ProofProviderOperation.CAPABILITY, ProofProviderOperation.PROVE),
            isolation=(ProofProviderIsolation.IN_PROCESS,), network_access_required=False,
            resource_limits_supported=True,
        )

    def prove(self, request):
        return {
            "status": "candidate",
            "proof_candidate": {
                "candidate_id": "candidate:one",
                "request_id": request.request_id,
            },
        }


class _CounterexampleBackend(_CandidateOnlyBackend):
    def prove(self, request):
        return {"status": "counterexample", "counterexample": {"model": {"bad": True}}}


def _obligation() -> ProofObligation:
    source = EvidenceReference("reviewed_source", "source:one", producer_id="test")
    claim = IRClaim(
        predicate="error_compatibility", subject_id="candidate:one",
        premise_ids=("premise:one",), source_ids=(source.content_id,),
        assumption_ids=("assumption:one",), repository_id="repo:one", tree_id="tree:one",
        translator_id="translator:one", toolchain_id="toolchain:one", policy_id="policy:one",
        capability_id="datasets.logic_ir", capability_revision="logic:one",
    )
    code = CodeProofObligation(
        repository_id="repo:one", repository_tree_id="tree:one", ast_scope_ids=("scope:one",),
        statement="error compatibility", premise_ids=("premise:one",),
        template_id="contract-repair/error", template_version="1", template_semantic_hash=claim.content_id,
        invariant_class="contract_repair", task_id="RPR-010",
        required_assurance=AssuranceLevel.KERNEL_VERIFIED,
    )
    return ProofObligation(ObligationKind.ERROR_COMPATIBILITY, "candidate:one", claim, code, (source,))


def test_candidate_without_independent_reconstruction_is_not_authoritative() -> None:
    # The backend lacks reconstruction.  We exercise its capability admission
    # directly: it must not become proof authority merely by being present.
    prover = ContractRepairProver(_CandidateOnlyBackend())
    assert prover._backend_supports(ProofProviderOperation.PROVE) is True
    assert prover._backend_supports(ProofProviderOperation.RECONSTRUCT) is False
    assert ContractRepairProofDisposition.NON_CONCLUSIVE.value == "non_conclusive"

    result = prover.prove_obligation(
        _obligation(), premises={"premise:one": {"statement": "reviewed premise"}}
    )
    assert result.disposition is ContractRepairProofDisposition.UNSUPPORTED
    assert result.receipt.authoritative_assurance is AssuranceLevel.UNVERIFIED
    assert result.reason_codes == ("independent_reconstruction_unavailable",)


def test_unverified_model_is_retained_as_a_minimal_reference_but_not_a_refutation() -> None:
    result = ContractRepairProver(_CounterexampleBackend()).prove_obligation(
        _obligation(), premises={"premise:one": {"statement": "reviewed premise"}}
    )
    assert result.disposition is ContractRepairProofDisposition.NON_CONCLUSIVE
    assert result.counterexample is not None
    assert result.receipt.authoritative_verdict.value == "inconclusive"
