"""Fail-closed coverage for missing-input prove/refute/reconstruct (RPR-036)."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.change_propagation_contracts import (
    BehaviorEvidencePrecedence,
    BehaviorKind,
    ConsumerDisposition,
    ConsumerMigrationObligation,
    ContractClauseDelta,
    DeltaDisposition,
    DeltaKind,
    GraphNodeRef,
    GraphProvenance,
    ImpactClosureReceipt,
    ImpactCompleteness,
    ImpactConsumer,
    MissingInputRequirement,
    ProgramContractDelta,
    PropagationAuthorityRoots,
    RequiredBehaviorContract,
    ValueCandidate,
    ValueCandidateDisposition,
    ValueCandidateKind,
)
from ipfs_accelerate_py.agent_supervisor.integrations.change_propagation_capabilities import (
    ChangePropagationCapability,
    ChangePropagationCapabilityReport,
    ChangePropagationCapabilityStatus,
)
from ipfs_accelerate_py.agent_supervisor.proof.change_propagation_obligations import (
    AssumptionBinding,
    ChangePropagationObligationCompiler,
    LogicCapabilityBinding,
    ObligationContext,
    ObligationKind,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_counterexamples import (
    CounterexampleBindings,
    CounterexampleKind,
    normalize_counterexample,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_capabilities import (
    ProofProviderCapability,
    ProofProviderIsolation,
    ProofProviderOperation,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    AssuranceLevel,
    CodeProofObligation,
    EvidenceAuthority,
    EvidenceFreshness,
    EvidenceKind,
    EvidenceVerdict,
    ProofEvidence,
    ProofVerdict,
    content_identity,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_provider import (
    ProviderFailureCode,
    ProviderResponse,
)
from ipfs_accelerate_py.agent_supervisor.proof.kernel_verification import (
    KernelFailureCode,
    KernelTarget,
    KernelVerificationResult,
    KernelVerificationStatus,
)
from ipfs_accelerate_py.agent_supervisor.proof.missing_input_synthesis import (
    FacetDisposition,
    MissingInputSynthesizer,
    SynthesisDisposition,
    ValueMappingProof,
    reconstruct_missing_input_proof,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def roots() -> PropagationAuthorityRoots:
    return PropagationAuthorityRoots(
        repository_id="repository:one",
        base_forest_id="forest:base",
        base_tree_id="tree:base",
        base_overlay_id="overlay:base",
        candidate_forest_id="forest:candidate",
        candidate_tree_id="tree:candidate",
        candidate_overlay_id="overlay:candidate",
        graph_id="graph:one",
        index_id="index:one",
        model_id="model:one",
        config_id="config:one",
        translator_id="translator:one",
        toolchain_id="toolchain:one",
        policy_id="policy:one",
    )


def node(path: str = "pkg/caller.py", symbol: str = "symbol:caller") -> GraphNodeRef:
    return GraphNodeRef(
        node_id=f"node:{symbol}",
        kind="function",
        path=path,
        symbol_id=symbol,
        artifact_id=f"blob:{symbol}",
        provenance=GraphProvenance.TRUSTED,
        extractor_id="extractor:ast",
    )


def clause() -> ContractClauseDelta:
    return ContractClauseDelta(
        clause_id="clause:param-add",
        kind=DeltaKind.PARAMETER_ADD,
        disposition=DeltaDisposition.BREAKING,
        subject_symbol_id="symbol:process",
        consumer_domain="domain:python-callers",
        before_contract_ref="contract:before",
        after_contract_ref="contract:after",
        reason="third-argument-required",
    )


def delta(auth: PropagationAuthorityRoots | None = None) -> ProgramContractDelta:
    auth = auth or roots()
    return ProgramContractDelta(
        roots=auth,
        change_set_id="changeset:one",
        subject_symbol_id="symbol:process",
        before_contract_ref="contract:before",
        after_contract_ref="contract:after",
        clauses=(clause(),),
        evidence_refs=("evidence:delta",),
        proof_refs=("proof:delta",),
    )


def closure(auth: PropagationAuthorityRoots | None = None) -> ImpactClosureReceipt:
    auth = auth or roots()
    return ImpactClosureReceipt(
        roots=auth,
        delta_id="delta:one",
        completeness=ImpactCompleteness.COMPLETE,
        consumers=(
            ImpactConsumer(
                consumer_id="consumer:one",
                node=node(),
                depth=1,
                mandatory=True,
                edge_refs=("edge:call",),
                path_condition_ref="path:always",
            ),
        ),
        evidence_refs=("evidence:closure",),
    )


def consumer(
    auth: PropagationAuthorityRoots | None = None,
    *,
    missing_input_ids: tuple[str, ...] = ("missing:context",),
    behavior_contract_ids: tuple[str, ...] = (),
) -> ConsumerMigrationObligation:
    auth = auth or roots()
    return ConsumerMigrationObligation(
        roots=auth,
        obligation_id="obligation:consumer:one",
        consumer_id="consumer:one",
        delta_id="delta:one",
        disposition=ConsumerDisposition.MIGRATE,
        clause_ids=("clause:param-add",),
        node=node(),
        proof_refs=("proof:obligation",),
        missing_input_ids=missing_input_ids,
        behavior_contract_ids=behavior_contract_ids,
        invalidation_refs=("tree:candidate",),
    )


def missing_input(
    auth: PropagationAuthorityRoots | None = None,
    *,
    propagation_depth_bound: int = 0,
) -> MissingInputRequirement:
    auth = auth or roots()
    return MissingInputRequirement(
        roots=auth,
        requirement_id="missing:context",
        obligation_id="obligation:consumer:one",
        clause_id="clause:param-add",
        parameter_name="context",
        type_ref="type:Context",
        nullability="nonnull",
        information_content_ref="info:request-context",
        construction_precondition_refs=("pre:context-ready",),
        result_postcondition_refs=("post:context-valid",),
        allowed_error_refs=("error:ContextMissing",),
        effect_refs=("effect:none",),
        capability_refs=("capability:request.read",),
        authorization_refs=("auth:caller",),
        resource_refs=("resource:stack",),
        ownership_refs=("ownership:borrowed",),
        propagation_depth_bound=propagation_depth_bound,
        proof_refs=("proof:missing",),
    )


def value_candidate(
    auth: PropagationAuthorityRoots | None = None,
    *,
    candidate_id: str = "candidate:ctx-param",
    kind: ValueCandidateKind = ValueCandidateKind.PARAMETER,
    disposition: ValueCandidateDisposition = ValueCandidateDisposition.NOMINATED,
    expression_ref: str = "expr:ctx",
) -> ValueCandidate:
    auth = auth or roots()
    return ValueCandidate(
        roots=auth,
        candidate_id=candidate_id,
        requirement_id="missing:context",
        kind=kind,
        disposition=disposition,
        source_node=node(path="pkg/caller.py", symbol=f"symbol:{candidate_id}"),
        expression_ref=expression_ref,
        type_ref="type:Context",
        semantic_authority=False,
        proof_refs=(),
    )


def behavior(auth: PropagationAuthorityRoots | None = None) -> RequiredBehaviorContract:
    auth = auth or roots()
    return RequiredBehaviorContract(
        roots=auth,
        behavior_id="behavior:context-type",
        kind=BehaviorKind.CLASS,
        subject_symbol_id="symbol:Context",
        evidence_precedence=BehaviorEvidencePrecedence.REVIEWED_IDL,
        field_refs=("field:request_id",),
        constructor_refs=("ctor:Context",),
        method_refs=("method:validate",),
        invariant_refs=("inv:nonempty-id",),
        state_transition_refs=("state:init->ready",),
        effect_refs=("effect:none",),
        capability_refs=("capability:request.read",),
        authorization_refs=("auth:caller",),
        resource_refs=("resource:heap",),
        proof_refs=("proof:behavior",),
        placement_decision_ref="placement:pkg.types",
    )


def capability() -> LogicCapabilityBinding:
    report = ChangePropagationCapabilityReport(
        capabilities=(
            ChangePropagationCapability(
                "datasets.logic_ir",
                ChangePropagationCapabilityStatus.AVAILABLE,
                module_paths=("/tmp/logic_ir.py",),
                interface_version="logic-ir@1",
                supported_semantics=("ir", "tdfol", "cec", "smt", "hammer"),
                reconstruction_compatible=True,
                details={"capability_revision": "logic:one"},
            ),
        ),
        accelerator_module_paths=(),
        datasets_module_paths=("/tmp/logic_ir.py",),
        datasets_gitlink_revision="gitlink:one",
    )
    return LogicCapabilityBinding.from_report(report)


def context() -> ObligationContext:
    return ObligationContext(
        assumptions=(
            AssumptionBinding(
                assumption_id="assumption:reviewed-one",
                kind="reviewed_assumption",
                evidence_ref="evidence:assumption:one",
            ),
        ),
        capability=capability(),
    )


def compile_migration(
    *,
    candidates: tuple[ValueCandidate, ...] = (),
    with_behavior: bool = False,
    missing: MissingInputRequirement | None = None,
):
    auth = roots()
    req = missing or missing_input(auth)
    behavior_ids = ("behavior:context-type",) if with_behavior else ()
    behaviors = (behavior(auth),) if with_behavior else ()
    cand = candidates or (value_candidate(auth),)
    return ChangePropagationObligationCompiler().compile(
        delta(auth),
        closure(auth),
        consumer(auth, behavior_contract_ids=behavior_ids),
        context(),
        missing_inputs=(req,),
        value_candidates=cand,
        behavior_contracts=behaviors,
    )


def premises_for(compilation) -> dict[str, dict[str, str]]:
    premise_ids: set[str] = set()
    for item in compilation.obligations:
        premise_ids.update(item.claim.premise_ids)
    return {
        premise_id: {"premise_id": premise_id, "statement": f"reviewed:{premise_id}"}
        for premise_id in premise_ids
    }


# ---------------------------------------------------------------------------
# Mock backends
# ---------------------------------------------------------------------------


class _BaseBackend:
    provider_id = "hammer"
    provider_version = "test-1"

    def capabilities(self):
        return ProofProviderCapability(
            provider_id=self.provider_id,
            provider_version=self.provider_version,
            protocol_versions=(1,),
            operations=(
                ProofProviderOperation.CAPABILITY,
                ProofProviderOperation.PROVE,
            ),
            isolation=(ProofProviderIsolation.IN_PROCESS,),
            network_access_required=False,
            resource_limits_supported=True,
        )


class _CandidateOnlyBackend(_BaseBackend):
    """Solver emits a candidate without reconstruction support."""

    def prove(self, request):
        return {
            "status": "candidate",
            "proof_candidate": {
                "candidate_id": "solver:cand",
                "request_id": request.request_id,
            },
        }


class _TimeoutBackend(_BaseBackend):
    def prove(self, request):
        return ProviderResponse.failure(
            request,
            ProviderFailureCode.TIMED_OUT,
            "solver timed out",
            provider_id=self.provider_id,
            provider_version=self.provider_version,
        )


class _CounterexampleBackend(_BaseBackend):
    def prove(self, request):
        return {
            "status": "counterexample",
            "counterexample": {"model": {"bad": True}},
            "unsatisfied_clauses": ("clause:type-mismatch",),
        }


class _ReconstructingBackend(_BaseBackend):
    """Proves and reconstructs every obligation for any candidate_id."""

    def capabilities(self):
        return ProofProviderCapability(
            provider_id=self.provider_id,
            provider_version=self.provider_version,
            protocol_versions=(1,),
            operations=(
                ProofProviderOperation.CAPABILITY,
                ProofProviderOperation.PROVE,
                ProofProviderOperation.RECONSTRUCT,
            ),
            isolation=(ProofProviderIsolation.IN_PROCESS,),
            network_access_required=False,
            resource_limits_supported=True,
        )

    def prove(self, request):
        return {
            "status": "proved",
            "proof_candidate": {
                "candidate_id": "solver:ok",
                "request_id": request.request_id,
                "value_candidate_id": request.payload.get("candidate_id", ""),
            },
        }

    def reconstruct(self, request):
        obligation_payload = request.payload["obligation"]
        code = CodeProofObligation.from_dict(obligation_payload)
        obligation_id = code.obligation_id
        candidate_id = str(request.payload.get("value_candidate_id") or "")
        proof_candidate = request.payload.get("proof_candidate") or {}
        if not candidate_id:
            candidate_id = str(proof_candidate.get("candidate_id", "solver:ok"))
        kernel_id = "kernel:test"
        evidence = ProofEvidence(
            kind=EvidenceKind.KERNEL_VERIFICATION,
            authority=EvidenceAuthority.KERNEL,
            verdict=EvidenceVerdict.ACCEPTED,
            artifact_id=content_identity(
                {
                    "request": request.request_id,
                    "obligation": obligation_id,
                    "candidate": candidate_id,
                }
            ),
            subject_id=obligation_id,
            verifier_id=kernel_id,
            freshness=EvidenceFreshness.CURRENT,
            independent=True,
            simulated=False,
            metadata={
                "bindings_verified": True,
                "candidate_id": candidate_id,
                "reconstruction": True,
                "toolchain_id": "toolchain:one",
            },
        )
        verification = KernelVerificationResult(
            target=KernelTarget.LEAN,
            status=KernelVerificationStatus.ACCEPTED,
            failure_code=KernelFailureCode.NONE,
            reason_codes=("independent_kernel_acceptance",),
            obligation_id=obligation_id,
            request_id=request.request_id,
            candidate_id=candidate_id,
            reconstruction_id=content_identity({"req": request.request_id}),
            kernel_id=kernel_id,
            toolchain_id="toolchain:one",
            environment_lock_id="env:test",
            checked_source_digest=content_identity({"src": obligation_id}),
            kernel_output_digest=content_identity({"out": request.request_id}),
            evidence=evidence,
        )
        return {
            "status": "reconstructed",
            "kernel_verification": verification.to_dict(),
        }


class _SelectiveReconstructingBackend(_ReconstructingBackend):
    """Reconstructs only for listed value candidate ids; refutes others."""

    def __init__(self, prove_ids: set[str], refute_ids: set[str] | None = None) -> None:
        self.prove_ids = set(prove_ids)
        self.refute_ids = set(refute_ids or ())

    def prove(self, request):
        value_id = str(request.payload.get("candidate_id", ""))
        if value_id in self.refute_ids:
            return {
                "status": "counterexample",
                "counterexample": {"model": {"bad": value_id}},
                "unsatisfied_clauses": (f"clause:reject:{value_id}",),
            }
        if value_id in self.prove_ids or value_id.startswith("behavior:"):
            return super().prove(request)
        return {"status": "unknown"}


def _verified_counterexample(obligation, raw):
    return normalize_counterexample(
        raw,
        kind=CounterexampleKind.SMT_MODEL,
        bindings=CounterexampleBindings(
            tree_ids=(obligation.claim.tree_id,),
            obligation_ids=(obligation.code_obligation.obligation_id,),
            provider_ids=("hammer",),
            policy_ids=(obligation.claim.policy_id,),
        ),
        violated_property=obligation.claim.predicate,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_candidate_without_reconstruction_has_no_code_authority() -> None:
    compilation = compile_migration()
    synthesizer = MissingInputSynthesizer(_CandidateOnlyBackend())
    assert synthesizer._backend_supports(ProofProviderOperation.PROVE) is True
    assert synthesizer._backend_supports(ProofProviderOperation.RECONSTRUCT) is False

    receipt = synthesizer.synthesize(
        compilation,
        premises=premises_for(compilation),
        value_candidates=(value_candidate(),),
        missing_inputs=(missing_input(),),
    )
    assert receipt.code_authority is False
    assert len(receipt.value_mapping_proofs) == 1
    proof = receipt.value_mapping_proofs[0]
    assert proof.disposition is SynthesisDisposition.UNSUPPORTED
    assert proof.code_authority is False
    assert "independent_reconstruction_unavailable" in proof.reason_codes or any(
        "independent_reconstruction_unavailable" in item.reason_codes
        for item in proof.facet_results
    )


def test_unverified_counterexample_preserved_but_not_refutation() -> None:
    compilation = compile_migration()
    result = MissingInputSynthesizer(_CounterexampleBackend()).prove_obligation(
        next(
            item
            for item in compilation.obligations
            if item.kind is ObligationKind.TYPE_SCHEMA_RANGE_NULLABILITY
        ),
        candidate_id="candidate:ctx-param",
        premises=premises_for(compilation),
    )
    assert result.disposition is FacetDisposition.UNKNOWN
    assert result.counterexample is not None
    assert result.receipt.authoritative_verdict is not ProofVerdict.DISPROVED
    assert "unverified_counterexample" in result.reason_codes
    assert "clause:type-mismatch" in result.unsatisfied_clauses


def test_verified_counterexample_is_authoritative_refutation() -> None:
    compilation = compile_migration()
    synthesizer = MissingInputSynthesizer(
        _CounterexampleBackend(),
        counterexample_verifier=_verified_counterexample,
    )
    obligation = next(
        item
        for item in compilation.obligations
        if item.kind is ObligationKind.INFORMATION_SUFFICIENCY
    )
    result = synthesizer.prove_obligation(
        obligation,
        candidate_id="candidate:ctx-param",
        premises=premises_for(compilation),
    )
    assert result.disposition is FacetDisposition.REFUTED
    assert result.counterexample is not None
    assert result.receipt.authoritative_verdict is ProofVerdict.DISPROVED
    assert result.code_authority is False


def test_incomplete_premise_slice_is_non_conclusive() -> None:
    compilation = compile_migration()
    obligation = next(
        item
        for item in compilation.obligations
        if item.kind in {
            ObligationKind.SOURCE_SCOPE_PATH_AVAILABILITY,
            ObligationKind.TYPE_SCHEMA_RANGE_NULLABILITY,
        }
    )
    result = MissingInputSynthesizer(_ReconstructingBackend()).prove_obligation(
        obligation,
        candidate_id="candidate:ctx-param",
        premises={},  # empty slice
    )
    assert result.disposition is FacetDisposition.UNKNOWN
    assert "incomplete_premise_slice" in result.reason_codes
    assert result.receipt.satisfies_completion(AssuranceLevel.KERNEL_VERIFIED) is False


def test_timeout_disposition_has_no_code_authority() -> None:
    compilation = compile_migration()
    receipt = MissingInputSynthesizer(_TimeoutBackend()).synthesize(
        compilation,
        premises=premises_for(compilation),
        value_candidates=(value_candidate(),),
        missing_inputs=(missing_input(),),
    )
    proof = receipt.value_mapping_proofs[0]
    assert proof.disposition is SynthesisDisposition.TIMEOUT
    assert proof.code_authority is False
    assert receipt.code_authority is False


def test_unique_reconstructed_candidate_grants_code_authority() -> None:
    compilation = compile_migration()
    cand = value_candidate()
    receipt = MissingInputSynthesizer(_ReconstructingBackend()).synthesize(
        compilation,
        premises=premises_for(compilation),
        value_candidates=(cand,),
        missing_inputs=(missing_input(),),
    )
    proof = receipt.value_mapping_proofs[0]
    assert proof.disposition is SynthesisDisposition.UNIQUE_PROVED
    assert proof.proved_candidate_ids == ("candidate:ctx-param",)
    assert proof.unique_candidate_id == "candidate:ctx-param"
    assert proof.expression_ref == "expr:ctx"
    assert proof.code_authority is True
    assert receipt.code_authority is True
    assert all(
        item.disposition is FacetDisposition.PROVED
        for item in proof.facet_results
        if item.obligation_kind
        not in {ObligationKind.CLOSURE_COVERAGE, ObligationKind.CONSUMER_COMPATIBILITY}
    )


def test_multiple_independently_proved_candidates_are_ambiguous_not_ranked() -> None:
    auth = roots()
    first = value_candidate(auth, candidate_id="candidate:zzz-late", expression_ref="expr:late")
    second = value_candidate(auth, candidate_id="candidate:aaa-early", expression_ref="expr:early")
    # Compile with the first listed candidate so a mapping claim exists; synthesis
    # still evaluates both independently.
    compilation = compile_migration(candidates=(first, second))
    receipt = MissingInputSynthesizer(_ReconstructingBackend()).synthesize(
        compilation,
        premises=premises_for(compilation),
        # Intentionally reverse search/nomination order relative to id order.
        value_candidates=(first, second),
        missing_inputs=(missing_input(auth),),
    )
    proof = receipt.value_mapping_proofs[0]
    assert proof.disposition is SynthesisDisposition.AMBIGUOUS
    # Content order, never nomination order.
    assert proof.proved_candidate_ids == ("candidate:aaa-early", "candidate:zzz-late")
    assert proof.code_authority is False
    assert receipt.code_authority is False
    assert "multiple_independently_proved_candidates" in proof.reason_codes


def test_no_candidate_is_refuted_and_may_thread_upstream_with_origin() -> None:
    auth = roots()
    req = missing_input(auth, propagation_depth_bound=2)
    compilation = compile_migration(candidates=(), missing=req)
    # Compiler with no non-nomination candidate still lowers facets.
    receipt = MissingInputSynthesizer(_ReconstructingBackend()).synthesize(
        compilation,
        premises=premises_for(compilation),
        value_candidates=(),
        missing_inputs=(req,),
        allow_upstream_threading=True,
    )
    proof = receipt.value_mapping_proofs[0]
    assert proof.disposition is SynthesisDisposition.REFUTED
    assert proof.proved_candidate_ids == ()
    assert proof.code_authority is False
    assert proof.upstream_thread is not None
    assert proof.upstream_thread.origin_requirement_id == "missing:context"
    assert proof.upstream_thread.origin_consumer_id == "consumer:one"
    assert proof.upstream_thread.parameter_name == "context"
    assert "thread_upstream_with_origin" in proof.upstream_thread.reason_codes


def test_upstream_thread_requires_origin_and_depth_bound() -> None:
    auth = roots()
    req = missing_input(auth, propagation_depth_bound=0)
    compilation = compile_migration(candidates=(), missing=req)
    receipt = MissingInputSynthesizer(_ReconstructingBackend()).synthesize(
        compilation,
        premises=premises_for(compilation),
        value_candidates=(),
        missing_inputs=(req,),
        allow_upstream_threading=True,
    )
    proof = receipt.value_mapping_proofs[0]
    assert proof.disposition is SynthesisDisposition.REFUTED
    assert proof.upstream_thread is None


def test_all_candidates_refuted_preserves_minimal_counterexamples() -> None:
    auth = roots()
    cand = value_candidate(auth)
    compilation = compile_migration(candidates=(cand,))
    synthesizer = MissingInputSynthesizer(
        _SelectiveReconstructingBackend(prove_ids=set(), refute_ids={cand.candidate_id}),
        counterexample_verifier=_verified_counterexample,
    )
    receipt = synthesizer.synthesize(
        compilation,
        premises=premises_for(compilation),
        value_candidates=(cand,),
        missing_inputs=(missing_input(auth),),
    )
    proof = receipt.value_mapping_proofs[0]
    assert proof.disposition is SynthesisDisposition.REFUTED
    assert proof.refuted_candidate_ids == (cand.candidate_id,)
    assert any(item.counterexample is not None for item in proof.facet_results)
    assert proof.unsatisfied_clauses
    assert proof.code_authority is False


def test_nomination_only_candidate_never_gains_code_authority() -> None:
    auth = roots()
    cand = value_candidate(
        auth,
        candidate_id="candidate:vector",
        kind=ValueCandidateKind.VECTOR_NOMINATION,
    )
    compilation = compile_migration(candidates=(cand,))
    receipt = MissingInputSynthesizer(_ReconstructingBackend()).synthesize(
        compilation,
        premises=premises_for(compilation),
        value_candidates=(cand,),
        missing_inputs=(missing_input(auth),),
    )
    proof = receipt.value_mapping_proofs[0]
    assert proof.disposition is SynthesisDisposition.UNSUPPORTED
    assert proof.proved_candidate_ids == ()
    assert proof.code_authority is False
    assert any(
        "nomination_only_not_authoritative" in item.reason_codes
        for item in proof.facet_results
    )


def test_behavior_proof_set_unique_proved() -> None:
    compilation = compile_migration(with_behavior=True)
    assert compilation.behavior_refinement_claims
    receipt = MissingInputSynthesizer(_ReconstructingBackend()).synthesize(
        compilation,
        premises=premises_for(compilation),
        value_candidates=(value_candidate(),),
        missing_inputs=(missing_input(),),
    )
    assert receipt.behavior_proof_sets
    behavior_proof = receipt.behavior_proof_sets[0]
    assert behavior_proof.behavior_id == "behavior:context-type"
    assert behavior_proof.disposition is SynthesisDisposition.UNIQUE_PROVED
    assert behavior_proof.code_authority is True
    assert receipt.code_authority is True


def test_behavior_timeout_blocks_code_authority() -> None:
    compilation = compile_migration(with_behavior=True)
    receipt = MissingInputSynthesizer(_TimeoutBackend()).synthesize(
        compilation,
        premises=premises_for(compilation),
        value_candidates=(value_candidate(),),
        missing_inputs=(missing_input(),),
    )
    assert receipt.behavior_proof_sets[0].disposition is SynthesisDisposition.TIMEOUT
    assert receipt.behavior_proof_sets[0].code_authority is False
    assert receipt.code_authority is False


def test_mixed_proved_and_inconclusive_is_unknown_not_unique() -> None:
    auth = roots()
    good = value_candidate(auth, candidate_id="candidate:good", expression_ref="expr:good")
    bad = value_candidate(auth, candidate_id="candidate:bad", expression_ref="expr:bad")
    compilation = compile_migration(candidates=(good, bad))
    backend = _SelectiveReconstructingBackend(prove_ids={"candidate:good"})
    receipt = MissingInputSynthesizer(backend).synthesize(
        compilation,
        premises=premises_for(compilation),
        value_candidates=(good, bad),
        missing_inputs=(missing_input(auth),),
    )
    proof = receipt.value_mapping_proofs[0]
    assert proof.disposition is SynthesisDisposition.UNKNOWN
    assert "candidate:good" in proof.proved_candidate_ids
    assert "candidate:bad" in proof.inconclusive_candidate_ids
    assert proof.code_authority is False


def test_value_mapping_proof_rejects_search_order_uniqueness() -> None:
    with pytest.raises(Exception):
        ValueMappingProof(
            requirement_id="missing:context",
            consumer_id="consumer:one",
            disposition=SynthesisDisposition.UNIQUE_PROVED,
            facet_results=(),
            proved_candidate_ids=("a", "b"),  # not unique
        )


def test_reconstruct_missing_input_proof_convenience() -> None:
    compilation = compile_migration()
    obligation = next(
        item
        for item in compilation.obligations
        if item.kind is ObligationKind.PARAMETER_THREADING
    )
    result = reconstruct_missing_input_proof(
        obligation,
        candidate_id="candidate:ctx-param",
        backend=_ReconstructingBackend(),
        premises=premises_for(compilation),
    )
    assert result.disposition is FacetDisposition.PROVED
    assert result.authoritative is True
    assert result.code_authority is False  # single facet never authorizes code


def test_synthesis_dispositions_are_closed_vocabulary() -> None:
    values = {item.value for item in SynthesisDisposition}
    assert values == {
        "unique_proved",
        "refuted",
        "ambiguous",
        "unknown",
        "timeout",
        "unsupported",
    }


def test_receipt_serializes_without_code_authority_on_failure() -> None:
    compilation = compile_migration()
    receipt = MissingInputSynthesizer(_CandidateOnlyBackend()).synthesize(
        compilation,
        premises=premises_for(compilation),
        value_candidates=(value_candidate(),),
        missing_inputs=(missing_input(),),
    )
    payload = receipt.to_dict()
    assert payload["code_authority"] is False
    assert payload["interface"] == "MissingInputSynthesizer@1"
    assert payload["invalidators"]
    assert "tree:candidate" in payload["invalidators"]
    assert "toolchain:one" in payload["invalidators"]
