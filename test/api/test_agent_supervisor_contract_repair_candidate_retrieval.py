"""Adversarial conformance tests for repair-candidate nomination."""

from __future__ import annotations

from ipfs_accelerate_py.agent_supervisor.analysis.contract_repair_candidate_retrieval import (
    CandidateDisposition,
    CandidateRetrievalBounds,
    CandidateRetrievalBoundsError,
    ContractRepairCandidateRetriever,
    REJECTION_FORBIDDEN_LAYER,
    REJECTION_FORGED_HISTORY,
    REJECTION_GENERATED_VENDOR_ARCHIVE_TARGET,
    REJECTION_PARTIAL_CANDIDATE,
    REJECTION_POISONED_VECTOR,
    REJECTION_READ_ONLY_TARGET,
    REJECTION_SAME_NAME_INCOMPATIBLE,
    REJECTION_STALE_OR_CROSS_TREE,
)
from ipfs_accelerate_py.agent_supervisor.analysis.contract_repair_contracts import (
    AuthorityRoots,
    BrokenContractTrace,
    CallRequirementContract,
    EvidenceReference,
    MemorySafetyDisposition,
    MemorySafetyFacet,
    RepairStrategy,
    SourceSpan,
    TraceDisposition,
)


ROOTS = AuthorityRoots(
    repository_id="repository:fixture", forest_id="forest:fixture", tree_id="tree:fixture",
    graph_id="graph:fixture", index_id="index:fixture", model_id="model:fixture",
    config_id="config:fixture", translator_id="translator:fixture",
    toolchain_id="toolchain:fixture", policy_id="policy:fixture",
)
EVIDENCE = EvidenceReference("fixture", "evidence:fixture", "case:retrieval")


def _inputs() -> tuple[BrokenContractTrace, CallRequirementContract, MemorySafetyFacet]:
    caller = SourceSpan("pkg/caller.py", 4, 16, "blob:caller")
    trace = BrokenContractTrace(
        ROOTS, caller, "symbol:caller", "old_receiver", TraceDisposition.LIKELY_REFACTOR,
        evidence_refs=(EVIDENCE,),
    )
    requirement = CallRequirementContract(
        ROOTS, trace.content_id, caller, (EVIDENCE,), evidence_refs=(EVIDENCE,),
    )
    facet = MemorySafetyFacet(
        ROOTS, caller, "python", MemorySafetyDisposition.SUPPORTED,
        evidence_refs=(EVIDENCE,),
    )
    return trace, requirement, facet


def _candidate(path: str = "pkg/moved.py", **extra: object) -> dict[str, object]:
    value: dict[str, object] = {
        "target_span": SourceSpan(path, 20, 42, "blob:" + path.replace("/", ":")),
        "evidence_refs": (EVIDENCE,),
        "history_reviewed": True,
    }
    value.update(extra)
    return value


def test_union_is_deterministic_deduplicated_and_non_authoritative() -> None:
    trace, requirement, facet = _inputs()
    retriever = ContractRepairCandidateRetriever(ROOTS)
    signals = {
        "vector": (_candidate(score=0.9, semantic_authority=False),),
        "history": (_candidate(),),
        "ast": (_candidate(),),
    }

    forward = retriever.retrieve(trace, requirement, facet, candidates_by_signal=signals)
    reverse = retriever.retrieve(trace, requirement, facet, candidates_by_signal=dict(reversed(tuple(signals.items()))))

    assert forward.content_id == reverse.content_id
    assert len(forward.candidates) == 1
    candidate = forward.candidates[0]
    assert candidate.disposition is CandidateDisposition.NOMINATED
    assert candidate.strategy is RepairStrategy.RENAME_SUBSTITUTION
    assert tuple(signal for signal, _ in candidate.signal_evidence) == ("ast", "exact_history", "vector")
    assert candidate.candidate.candidate_write_paths == ()
    assert candidate.candidate.permitted_read_paths == ()
    assert candidate.write_paths == forward.write_paths == ()
    assert forward.semantic_authority is False
    assert forward.admitted_candidate_id == ""
    assert forward.candidate_set_id
    assert type(forward).from_dict(forward.to_record()).content_id == forward.content_id


def test_adversarial_targets_are_retained_with_stable_diagnostics() -> None:
    trace, requirement, facet = _inputs()
    receipt = ContractRepairCandidateRetriever(ROOTS).retrieve(
        trace, requirement, facet,
        candidates_by_signal={
            "ast": (
                _candidate("pkg/same_name.py", same_name=True, signature_compatible=False),
                _candidate("pkg/read_only.py", read_only=True),
                _candidate("vendor/generated.py"),
                _candidate("pkg/forbidden.py", forbidden_layer=True),
                _candidate("pkg/forged.py", forged_history=True),
                _candidate("pkg/stale.py", tree_id="tree:other"),
                {"partial": True, "evidence_refs": (EVIDENCE,)},
            ),
            "vector": (_candidate("pkg/poison.py", semantic_authority=True, score=0.1),),
        },
    )

    by_path = {item.target_span.path: item for item in receipt.candidates}
    assert by_path["pkg/same_name.py"].diagnostics == (REJECTION_SAME_NAME_INCOMPATIBLE,)
    assert by_path["pkg/read_only.py"].diagnostics == (REJECTION_READ_ONLY_TARGET,)
    assert by_path["vendor/generated.py"].diagnostics == (REJECTION_GENERATED_VENDOR_ARCHIVE_TARGET,)
    assert by_path["pkg/forbidden.py"].diagnostics == (REJECTION_FORBIDDEN_LAYER,)
    assert by_path["pkg/forged.py"].diagnostics == (REJECTION_FORGED_HISTORY,)
    assert by_path["pkg/stale.py"].diagnostics == (REJECTION_STALE_OR_CROSS_TREE,)
    assert by_path["pkg/poison.py"].diagnostics == (REJECTION_POISONED_VECTOR,)
    assert any(item.diagnostics == (REJECTION_PARTIAL_CANDIDATE,) for item in receipt.candidates)
    assert all(item.disposition is CandidateDisposition.REJECTED for item in receipt.candidates)
    assert by_path["pkg/poison.py"].strategy is RepairStrategy.REJECT
    assert any(item.strategy is RepairStrategy.REJECT for item in receipt.candidates)


def test_bounds_refuse_an_incomplete_union_instead_of_silently_dropping_candidates() -> None:
    trace, requirement, facet = _inputs()
    retriever = ContractRepairCandidateRetriever(ROOTS, bounds=CandidateRetrievalBounds(max_candidates=1, max_candidates_per_signal=2))

    try:
        retriever.retrieve(trace, requirement, facet, candidates_by_signal={"ast": (_candidate("pkg/a.py"), _candidate("pkg/b.py"))})
    except CandidateRetrievalBoundsError:
        pass
    else:  # pragma: no cover - assertion produces a clearer failure than an empty receipt
        raise AssertionError("over-budget union must fail closed")
