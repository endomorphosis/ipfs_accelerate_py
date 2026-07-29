"""Contract tests for the proof-gated repair-record boundary."""

from __future__ import annotations

import math

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.contract_repair_contracts import (
    AuthorityRoots,
    BrokenContractTrace,
    ContractRepairAuthorityError,
    ContractRepairError,
    DecisionDisposition,
    EvidenceReference,
    ForgedContractRepairIdentityError,
    MemorySafetyDisposition,
    MemorySafetyFacet,
    RepairCandidate,
    RepairStrategy,
    RepairTargetDecision,
    SourceSpan,
    TraceDisposition,
    candidate_set_identity,
)


@pytest.fixture
def roots() -> AuthorityRoots:
    return AuthorityRoots(
        repository_id="repository:one", forest_id="forest:one", tree_id="tree:one",
        graph_id="graph:one", index_id="index:one", model_id="model:one",
        config_id="config:one", translator_id="translator:one",
        toolchain_id="toolchain:one", policy_id="policy:one",
    )


@pytest.fixture
def span() -> SourceSpan:
    return SourceSpan("pkg/caller.py", 3, 11, "blob:caller")


@pytest.fixture
def evidence() -> EvidenceReference:
    return EvidenceReference("reviewed_test", "evidence:one", "case:rename")


def test_trace_is_content_addressed_root_and_span_bound(
    roots: AuthorityRoots, span: SourceSpan, evidence: EvidenceReference
) -> None:
    trace = BrokenContractTrace(
        roots, span, "symbol:caller", "old_receiver", TraceDisposition.LIKELY_REFACTOR,
        evidence_refs=(evidence,), graph_frontier_refs=("frontier:one",),
    )

    assert trace.content_id.startswith("b")
    assert trace.to_dict()["roots"]["tree_id"] == "tree:one"
    assert trace.to_dict()["caller_span"]["path"] == "pkg/caller.py"


@pytest.mark.parametrize("bad", ["../escape.py", "/absolute.py", "."])
def test_spans_and_authority_paths_are_repository_relative(bad: str) -> None:
    with pytest.raises(ContractRepairAuthorityError):
        SourceSpan(bad, 0, 1, "blob:one")


def test_record_rejects_source_bodies_and_nonfinite_values(
    roots: AuthorityRoots, span: SourceSpan, evidence: EvidenceReference
) -> None:
    with pytest.raises(ContractRepairError, match="source bodies"):
        EvidenceReference("source_body", "evidence:one")

    # Floats have no place in compact, canonical authority records.
    with pytest.raises(ContractRepairError):
        SourceSpan("pkg/a.py", math.inf, 1, "blob:one")  # type: ignore[arg-type]

    with pytest.raises(ContractRepairError, match="unresolvable"):
        BrokenContractTrace(
            roots, span, "symbol:caller", "dynamic_receiver", TraceDisposition.DYNAMIC,
            target_span=span, evidence_refs=(evidence,),
        )


def test_from_dict_rejects_forged_derived_identity(roots: AuthorityRoots) -> None:
    payload = roots.to_record()
    payload["content_id"] = "baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"

    with pytest.raises(ForgedContractRepairIdentityError):
        AuthorityRoots.from_dict(payload)


def test_memory_safety_does_not_promote_resource_bounds(
    roots: AuthorityRoots, span: SourceSpan, evidence: EvidenceReference
) -> None:
    with pytest.raises(ContractRepairError, match="proof references"):
        MemorySafetyFacet(roots, span, "python", MemorySafetyDisposition.PROVED)

    unsupported = MemorySafetyFacet(
        roots, span, "python", MemorySafetyDisposition.UNSUPPORTED,
        unsupported_refs=("reflection:unmodeled",),
    )
    empirical = MemorySafetyFacet(
        roots, span, "rust", MemorySafetyDisposition.EMPIRICAL,
        evidence_refs=(evidence,),
    )
    proved = MemorySafetyFacet(
        roots, span, "rust", MemorySafetyDisposition.PROVED,
        proof_refs=(evidence,),
    )
    assert {unsupported.disposition, empirical.disposition, proved.disposition} == {
        MemorySafetyDisposition.UNSUPPORTED,
        MemorySafetyDisposition.EMPIRICAL,
        MemorySafetyDisposition.PROVED,
    }
    assert "max_memory_bytes" not in proved.to_json()


def test_decision_requires_complete_candidate_identity_and_derived_write_paths(
    roots: AuthorityRoots, span: SourceSpan, evidence: EvidenceReference
) -> None:
    trace = BrokenContractTrace(
        roots, span, "symbol:caller", "missing", TraceDisposition.MISSING_LOCAL,
        evidence_refs=(evidence,),
    )
    candidate = RepairCandidate(
        roots, trace.content_id, RepairStrategy.NEW_IMPLEMENTATION, span, (evidence,),
        proof_refs=(evidence,), permitted_read_paths=("pkg/caller.py",),
        candidate_write_paths=("pkg/receiver.py",),
    )
    set_id = candidate_set_identity((candidate,))

    decision = RepairTargetDecision(
        roots, (candidate,), set_id, DecisionDisposition.ADMITTED,
        RepairStrategy.NEW_IMPLEMENTATION, candidate.content_id,
        permitted_read_paths=("pkg/caller.py",), permitted_write_paths=("pkg/receiver.py",),
        evidence_refs=(evidence,), proof_refs=(evidence,), invalidation_refs=("tree:one",),
    )
    assert decision.permitted_write_paths == ("pkg/receiver.py",)
    assert RepairTargetDecision.from_dict(decision.to_record()) == decision

    with pytest.raises(ForgedContractRepairIdentityError):
        RepairTargetDecision(
            roots, (candidate,), "candidate-set:forged", DecisionDisposition.ADMITTED,
            RepairStrategy.NEW_IMPLEMENTATION, candidate.content_id,
            permitted_write_paths=("pkg/receiver.py",), evidence_refs=(evidence,),
            proof_refs=(evidence,), invalidation_refs=("tree:one",),
        )

    with pytest.raises(ContractRepairAuthorityError, match="derived"):
        RepairTargetDecision(
            roots, (candidate,), set_id, DecisionDisposition.ADMITTED,
            RepairStrategy.NEW_IMPLEMENTATION, candidate.content_id,
            permitted_write_paths=("pkg/not-authorized.py",), evidence_refs=(evidence,),
            proof_refs=(evidence,), invalidation_refs=("tree:one",),
        )


def test_nonadmitted_decisions_cannot_select_or_grant_writes(
    roots: AuthorityRoots, span: SourceSpan, evidence: EvidenceReference
) -> None:
    candidate = RepairCandidate(
        roots, "trace:one", RepairStrategy.AMBIGUOUS, span, (evidence,)
    )
    with pytest.raises(ContractRepairAuthorityError):
        RepairTargetDecision(
            roots, (candidate,), candidate_set_identity((candidate,)),
            DecisionDisposition.ABSTAINED, RepairStrategy.AMBIGUOUS,
            candidate.content_id, evidence_refs=(evidence,), invalidation_refs=("tree:one",),
        )
