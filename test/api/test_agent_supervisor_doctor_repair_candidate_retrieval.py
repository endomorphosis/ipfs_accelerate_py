"""Adversarial conformance tests for doctor repair candidate retrieval (LPR-031)."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.doctor_repair_candidate_retrieval import (
    APPROXIMATE_SIGNALS,
    EXACT_SIGNALS,
    SIGNAL_PRECEDENCE,
    DoctorCandidateDisposition,
    DoctorCandidateEvidence,
    DoctorCandidateKind,
    DoctorCandidateQuery,
    DoctorCandidateRetrievalBindingError,
    DoctorCandidateRetrievalBounds,
    DoctorCandidateRetrievalBoundsError,
    DoctorCandidateSet,
    DoctorCandidateSignal,
    DoctorEligibilityStatus,
    DoctorRepairCandidate,
    DoctorRepairCandidateRetriever,
    DoctorRetrievalAuthorityRoots,
    DoctorSourceAuthority,
    REJECTION_BODY_OR_SECRET,
    REJECTION_COMPATIBILITY_CLAIM,
    REJECTION_FORGED,
    REJECTION_GENERATED,
    REJECTION_PARTIAL,
    REJECTION_PLACEMENT_CLAIM,
    REJECTION_POISONED,
    REJECTION_READ_ONLY,
    REJECTION_SEMANTIC_AUTHORITY_CLAIM,
    REJECTION_STALE_OR_CROSS_TREE,
    REJECTION_TARGET_CLAIM,
    REJECTION_VALUE_AUTHORITY_CLAIM,
    REJECTION_VECTOR_LANE_DISABLED,
    REJECTION_WRITE_SCOPE_CLAIM,
    candidate_set_identity,
    retrieve_doctor_repair_candidates,
)
from ipfs_accelerate_py.agent_supervisor.integrations.ipfs_datasets_embedding_provider import (
    DeterministicLocalEmbeddingBackend,
    EmbeddingLaneStatus,
    IpfsDatasetsEmbeddingProvider,
    create_pinned_embedding_policy,
)


ROOTS = DoctorRetrievalAuthorityRoots(
    repository_id="repository:fixture",
    forest_id="forest:fixture",
    tree_id="tree:fixture",
    overlay_id="overlay:fixture",
    graph_id="graph:fixture",
    index_id="index:fixture",
    model_id="model:fixture",
    config_id="config:fixture",
    corpus_id="corpus:fixture",
    policy_id="policy:fixture",
    toolchain_id="toolchain:fixture",
)


def _candidate(
    ref: str = "symbol:moved_fn",
    *,
    path: str = "pkg/moved.py",
    **extra: object,
) -> dict[str, object]:
    value: dict[str, object] = {
        "candidate_ref": ref,
        "path": path,
        "symbol_id": ref,
        "evidence_refs": ("evidence:fixture",),
        "history_reviewed": True,
    }
    value.update(extra)
    return value


def test_exact_routes_precede_lexical_kg_and_vector() -> None:
    assert SIGNAL_PRECEDENCE[:5] == (
        "exact_symbol",
        "exact_contract",
        "exact_value",
        "exact_lineage",
        "exact_graph",
    )
    assert SIGNAL_PRECEDENCE[5:] == ("lexical", "knowledge_graph", "vector")
    assert EXACT_SIGNALS.isdisjoint(APPROXIMATE_SIGNALS)
    assert set(SIGNAL_PRECEDENCE) == set(DoctorCandidateSignal.__members__[m].value for m in DoctorCandidateSignal.__members__)


def test_union_is_deterministic_deduplicated_and_non_authoritative() -> None:
    class EnabledLane:
        vector_lane = EmbeddingLaneStatus.ENABLED
        vector_lane_enabled = True

    retriever = DoctorRepairCandidateRetriever(ROOTS, embedding_provider=EnabledLane())
    signals = {
        "vector": (_candidate(score=0.9, semantic_authority=False),),
        "exact_symbol": (_candidate(),),
        "exact_graph": (_candidate(),),
        "lexical": (_candidate(score=0.1),),
    }
    forward = retriever.retrieve(
        "finding:rename",
        subject_path="pkg/caller.py",
        subject_symbol="old_name",
        candidates_by_signal=signals,
    )
    reverse = retriever.retrieve(
        "finding:rename",
        subject_path="pkg/caller.py",
        subject_symbol="old_name",
        candidates_by_signal=dict(reversed(tuple(signals.items()))),
    )
    assert forward.content_id == reverse.content_id
    assert len(forward.candidates) == 1
    nomination = forward.candidates[0]
    assert nomination.disposition is DoctorCandidateDisposition.NOMINATED
    assert nomination.semantic_authority is False
    signals_seen = [signal for signal, _ in nomination.signal_evidence]
    # Exact-first ordering of signal evidence.
    assert signals_seen == sorted(signals_seen, key=lambda s: SIGNAL_PRECEDENCE.index(s))
    assert "exact_symbol" in signals_seen
    assert nomination.candidate.evidence is not None
    assert nomination.candidate.evidence.candidate_cid
    assert nomination.candidate.evidence.source_authority in {
        DoctorSourceAuthority.REVIEWED,
        DoctorSourceAuthority.AUTHORITATIVE,
        DoctorSourceAuthority.NOMINATED,
    }
    # Score is retained as a separate fact on evidence, not as authority.
    assert nomination.candidate.write_paths == ()
    assert forward.write_paths == ()
    assert forward.admitted_candidate_id == ""
    assert forward.semantic_authority is False
    assert forward.candidate_set_id == candidate_set_identity(forward.repair_candidates)
    assert DoctorCandidateSet.from_dict(forward.to_record()).content_id == forward.content_id


def test_exact_signal_outranks_vector_for_eligibility() -> None:
    class EnabledLane:
        vector_lane = EmbeddingLaneStatus.ENABLED
        vector_lane_enabled = True

    receipt = DoctorRepairCandidateRetriever(
        ROOTS, embedding_provider=EnabledLane()
    ).retrieve(
        "finding:rank",
        subject_path="pkg/caller.py",
        candidates_by_signal={
            "exact_symbol": (_candidate("symbol:exact", path="pkg/exact.py"),),
            "vector": (
                _candidate(
                    "symbol:vector",
                    path="pkg/vector.py",
                    score=0.99,
                    semantic_authority=False,
                ),
            ),
        },
    )
    nominated = [
        item
        for item in receipt.candidates
        if item.disposition is DoctorCandidateDisposition.NOMINATED
    ]
    assert len(nominated) == 2
    by_ref = {item.candidate.candidate_ref: item for item in nominated}
    assert by_ref["symbol:exact"].eligibility_rank < by_ref["symbol:vector"].eligibility_rank
    assert receipt.eligibility_status is DoctorEligibilityStatus.NOMINATED_SET


def test_adversarial_targets_rejected_before_scoring() -> None:
    receipt = DoctorRepairCandidateRetriever(ROOTS).retrieve(
        "finding:adversarial",
        subject_path="pkg/caller.py",
        candidates_by_signal={
            "exact_symbol": (
                _candidate("symbol:stale", tree_id="tree:other", score=0.99),
                _candidate("symbol:readonly", read_only=True, score=0.98),
                _candidate("symbol:vendor", path="vendor/generated.py", score=0.97),
                _candidate("symbol:forged", forged_history=True, score=0.96),
                _candidate("symbol:poison", poisoned=True, score=0.95),
                {
                    "partial": True,
                    "evidence_refs": ("evidence:partial",),
                    "score": 0.94,
                },
                _candidate(
                    "symbol:authority",
                    semantic_authority=True,
                    score=0.93,
                ),
                _candidate(
                    "symbol:compat_claim",
                    compatible=True,
                    score=0.92,
                ),
                _candidate(
                    "symbol:write_claim",
                    write_paths=("pkg/caller.py",),
                    score=0.91,
                ),
                _candidate(
                    "symbol:placement",
                    placement="pkg/new_site.py",
                    score=0.90,
                ),
                _candidate(
                    "symbol:target",
                    selected_target=True,
                    score=0.89,
                ),
                _candidate(
                    "symbol:value_auth",
                    value_authority=True,
                    score=0.88,
                ),
                _candidate(
                    "symbol:body",
                    source_body="def leak():\n    return x\n",
                    api_key="should_never_appear",
                    score=0.87,
                ),
            ),
            "vector": (
                _candidate(
                    "symbol:vector_poison",
                    semantic_authority=True,
                    score=float("nan"),
                ),
            ),
        },
    )
    by_ref = {item.candidate.candidate_ref: item for item in receipt.candidates}
    assert REJECTION_STALE_OR_CROSS_TREE in by_ref["symbol:stale"].diagnostics
    assert REJECTION_READ_ONLY in by_ref["symbol:readonly"].diagnostics
    assert REJECTION_GENERATED in by_ref["symbol:vendor"].diagnostics
    assert REJECTION_FORGED in by_ref["symbol:forged"].diagnostics
    assert REJECTION_POISONED in by_ref["symbol:poison"].diagnostics
    assert any(REJECTION_PARTIAL in item.diagnostics for item in receipt.candidates)
    assert REJECTION_SEMANTIC_AUTHORITY_CLAIM in by_ref["symbol:authority"].diagnostics
    assert REJECTION_COMPATIBILITY_CLAIM in by_ref["symbol:compat_claim"].diagnostics
    assert REJECTION_WRITE_SCOPE_CLAIM in by_ref["symbol:write_claim"].diagnostics
    assert REJECTION_PLACEMENT_CLAIM in by_ref["symbol:placement"].diagnostics
    assert REJECTION_TARGET_CLAIM in by_ref["symbol:target"].diagnostics
    assert REJECTION_VALUE_AUTHORITY_CLAIM in by_ref["symbol:value_auth"].diagnostics
    assert REJECTION_BODY_OR_SECRET in by_ref["symbol:body"].diagnostics
    assert REJECTION_POISONED in by_ref["symbol:vector_poison"].diagnostics
    assert all(
        item.disposition is DoctorCandidateDisposition.REJECTED
        for item in receipt.candidates
    )
    # Rejected candidates must not retain scores that could authorize selection.
    for item in receipt.candidates:
        if item.candidate.evidence is not None:
            assert item.candidate.evidence.score_millionths is None
    blob = str(receipt.to_record())
    assert "should_never_appear" not in blob
    assert "def leak" not in blob


def test_vector_lane_disabled_rejects_vector_but_keeps_exact() -> None:
    class DisabledLane:
        vector_lane = EmbeddingLaneStatus.CANARY_FAILED
        vector_lane_enabled = False

    receipt = DoctorRepairCandidateRetriever(
        ROOTS, embedding_provider=DisabledLane()
    ).retrieve(
        "finding:lane",
        subject_path="pkg/caller.py",
        candidates_by_signal={
            "exact_symbol": (_candidate("symbol:exact"),),
            "vector": (_candidate("symbol:vec", score=0.5, semantic_authority=False),),
        },
    )
    by_ref = {item.candidate.candidate_ref: item for item in receipt.candidates}
    assert by_ref["symbol:exact"].disposition is DoctorCandidateDisposition.NOMINATED
    assert by_ref["symbol:vec"].disposition is DoctorCandidateDisposition.REJECTED
    assert REJECTION_VECTOR_LANE_DISABLED in by_ref["symbol:vec"].diagnostics
    assert receipt.vector_lane_status == EmbeddingLaneStatus.CANARY_FAILED.value


def test_pinned_embedding_provider_enables_vector_lane() -> None:
    policy = create_pinned_embedding_policy(
        dimensions=8,
        corpus_root_id="corpus:fixture",
        index_root_id="index:fixture",
        tree_id="tree:fixture",
        forest_id="forest:fixture",
        config_id="config:fixture",
        model_artifact_id="model:fixture",
    )
    provider = IpfsDatasetsEmbeddingProvider(
        policy,
        backend=DeterministicLocalEmbeddingBackend(policy),
    )
    roots = DoctorRetrievalAuthorityRoots(
        repository_id="repository:fixture",
        forest_id="forest:fixture",
        tree_id="tree:fixture",
        overlay_id="overlay:fixture",
        graph_id="graph:fixture",
        index_id="index:fixture",
        model_id="model:fixture",
        config_id="config:fixture",
        embedding_policy_id=policy.policy_id,
    )
    receipt = DoctorRepairCandidateRetriever(
        roots,
        embedding_provider=provider,
        embedding_policy=policy,
    ).retrieve(
        "finding:embed",
        subject_path="pkg/caller.py",
        candidates_by_signal={
            "vector": (
                _candidate(
                    "symbol:vec",
                    score=0.4,
                    semantic_authority=False,
                    embedding_policy_id=policy.policy_id,
                ),
            ),
        },
    )
    assert receipt.vector_lane_status == EmbeddingLaneStatus.ENABLED.value
    assert receipt.embedding_policy_id == policy.policy_id
    assert receipt.candidates[0].disposition is DoctorCandidateDisposition.NOMINATED
    assert receipt.candidates[0].candidate.evidence is not None
    assert receipt.candidates[0].candidate.evidence.source_authority is DoctorSourceAuthority.NOMINATED


def test_hard_compatibility_and_information_content_are_separate_from_scores() -> None:
    receipt = DoctorRepairCandidateRetriever(ROOTS).retrieve(
        "finding:facts",
        subject_path="pkg/caller.py",
        candidates_by_signal={
            "exact_value": (
                _candidate(
                    "expr:ctx",
                    hard_compatible=True,
                    information_content_ref="info:request-context",
                    score=0.12,
                    kind="reaching_value",
                ),
            ),
        },
    )
    nomination = receipt.candidates[0]
    evidence = nomination.candidate.evidence
    assert evidence is not None
    assert evidence.hard_compatible is True
    assert evidence.information_content_ref == "info:request-context"
    assert evidence.score_millionths == 120_000
    assert evidence.candidate_cid
    # hard_compatible fact does not grant write or semantic authority.
    assert nomination.candidate.semantic_authority is False
    assert nomination.candidate.write_paths == ()
    assert nomination.candidate.kind is DoctorCandidateKind.REACHING_VALUE


def test_no_candidate_and_multiple_equally_eligible_remain_explicit() -> None:
    empty = retrieve_doctor_repair_candidates(
        ROOTS,
        "finding:empty",
        subject_path="pkg/caller.py",
        candidates_by_signal={},
    )
    assert empty.eligibility_status is DoctorEligibilityStatus.NO_CANDIDATE
    assert empty.admitted_candidate_id == ""
    assert empty.equally_eligible_ids == ()
    assert empty.candidates[0].disposition is DoctorCandidateDisposition.REJECTED
    assert REJECTION_PARTIAL in empty.candidates[0].diagnostics

    tied = DoctorRepairCandidateRetriever(ROOTS).retrieve(
        "finding:tie",
        subject_path="pkg/caller.py",
        candidates_by_signal={
            "exact_symbol": (
                _candidate("symbol:a", path="pkg/a.py"),
                _candidate("symbol:b", path="pkg/b.py"),
            ),
        },
    )
    assert tied.eligibility_status is DoctorEligibilityStatus.MULTIPLE_EQUALLY_ELIGIBLE
    assert len(tied.equally_eligible_ids) == 2
    assert tied.admitted_candidate_id == ""
    assert tied.write_paths == ()
    # Ties cannot authorize target/value/placement/write.
    for item in tied.candidates:
        assert item.candidate.target_claim is False
        assert item.candidate.placement_claim is False
        assert item.candidate.value_authority is False
        assert item.candidate.write_paths == ()


def test_unique_eligible_still_does_not_admit() -> None:
    receipt = DoctorRepairCandidateRetriever(ROOTS).retrieve(
        "finding:unique",
        subject_path="pkg/caller.py",
        candidates_by_signal={
            "exact_contract": (_candidate("contract:receiver", kind="adapter"),),
        },
    )
    assert receipt.eligibility_status is DoctorEligibilityStatus.UNIQUE_ELIGIBLE
    assert len(receipt.equally_eligible_ids) == 1
    assert receipt.admitted_candidate_id == ""
    assert receipt.semantic_authority is False


def test_kinds_for_rename_move_constructor_factory_and_analogous() -> None:
    class EnabledLane:
        vector_lane = EmbeddingLaneStatus.ENABLED
        vector_lane_enabled = True

    receipt = DoctorRepairCandidateRetriever(
        ROOTS, embedding_provider=EnabledLane()
    ).retrieve(
        "finding:kinds",
        subject_path="pkg/caller.py",
        candidates_by_signal={
            "exact_symbol": (_candidate("symbol:renamed", rename=True),),
            "exact_lineage": (
                _candidate("symbol:moved", path="pkg/new_mod.py", move=True),
            ),
            "exact_graph": (
                _candidate(
                    "symbol:Factory.create",
                    path="pkg/factory.py",
                    factory=True,
                ),
            ),
            "exact_contract": (
                _candidate("symbol:Ctor", constructor=True, path="pkg/types.py"),
            ),
            "vector": (
                _candidate(
                    "symbol:analog",
                    score=0.3,
                    semantic_authority=False,
                ),
            ),
        },
    )
    by_ref = {item.candidate.candidate_ref: item for item in receipt.candidates}
    assert by_ref["symbol:renamed"].candidate.kind is DoctorCandidateKind.RENAME
    assert by_ref["symbol:moved"].candidate.kind is DoctorCandidateKind.MOVE
    assert by_ref["symbol:Factory.create"].candidate.kind is DoctorCandidateKind.FACTORY
    assert by_ref["symbol:Ctor"].candidate.kind is DoctorCandidateKind.CONSTRUCTOR
    assert by_ref["symbol:analog"].candidate.kind is DoctorCandidateKind.ANALOGOUS_REPAIR


def test_bounds_refuse_over_budget_per_signal_and_union() -> None:
    retriever = DoctorRepairCandidateRetriever(
        ROOTS,
        bounds=DoctorCandidateRetrievalBounds(max_candidates=2, max_candidates_per_signal=1),
    )
    with pytest.raises(DoctorCandidateRetrievalBoundsError):
        retriever.retrieve(
            "finding:budget",
            subject_path="pkg/caller.py",
            candidates_by_signal={
                "exact_symbol": (
                    _candidate("symbol:a"),
                    _candidate("symbol:b"),
                ),
            },
        )
    with pytest.raises(DoctorCandidateRetrievalBoundsError):
        DoctorRepairCandidateRetriever(
            ROOTS,
            bounds=DoctorCandidateRetrievalBounds(max_candidates=1, max_candidates_per_signal=8),
        ).retrieve(
            "finding:budget2",
            subject_path="pkg/caller.py",
            candidates_by_signal={
                "exact_symbol": (_candidate("symbol:a"),),
                "exact_graph": (_candidate("symbol:b"),),
            },
        )


def test_cross_root_query_fails_closed() -> None:
    other = DoctorRetrievalAuthorityRoots(
        repository_id="repository:other",
        forest_id="forest:other",
        tree_id="tree:other",
        overlay_id="overlay:other",
        graph_id="graph:other",
        index_id="index:other",
        model_id="model:other",
        config_id="config:other",
    )
    retriever = DoctorRepairCandidateRetriever(ROOTS)
    with pytest.raises(DoctorCandidateRetrievalBindingError):
        retriever.retrieve(
            "finding:x",
            query=DoctorCandidateQuery(roots=other, finding_id="finding:x"),
        )


def test_query_and_evidence_records_round_trip() -> None:
    query = DoctorCandidateQuery(
        roots=ROOTS,
        finding_id="finding:rt",
        subject_path="pkg/caller.py",
        subject_symbol="old",
        obligation_refs=("obl:1",),
        expected_behavior_refs=("beh:1",),
    )
    assert DoctorCandidateQuery.from_dict(query.to_record()).content_id == query.content_id
    evidence = DoctorCandidateEvidence(
        candidate_cid="cid:candidate",
        source_authority=DoctorSourceAuthority.REVIEWED,
        hard_compatible=False,
        information_content_ref="info:x",
        primary_signal="exact_symbol",
        score_millionths=42,
    )
    assert DoctorCandidateEvidence.from_dict(evidence.to_record()).content_id == evidence.content_id
    candidate = DoctorRepairCandidate(
        roots=ROOTS,
        finding_id="finding:rt",
        candidate_ref="symbol:x",
        kind=DoctorCandidateKind.RENAME,
        path="pkg/x.py",
        symbol_id="symbol:x",
        evidence=evidence,
    )
    assert DoctorRepairCandidate.from_dict(candidate.to_record()).content_id == candidate.content_id
    with pytest.raises(DoctorCandidateRetrievalBindingError):
        DoctorRepairCandidate(
            roots=ROOTS,
            finding_id="finding:rt",
            candidate_ref="symbol:bad",
            kind=DoctorCandidateKind.RENAME,
            semantic_authority=True,
        )


def test_signal_aliases_map_to_canonical_families() -> None:
    class EnabledLane:
        vector_lane = EmbeddingLaneStatus.ENABLED
        vector_lane_enabled = True

    receipt = DoctorRepairCandidateRetriever(
        ROOTS, embedding_provider=EnabledLane()
    ).retrieve(
        "finding:alias",
        subject_path="pkg/caller.py",
        candidates_by_signal={
            "ast": (_candidate("symbol:ast"),),
            "history": (_candidate("symbol:hist", path="pkg/hist.py"),),
            "bm25": (_candidate("symbol:bm25", path="pkg/bm25.py"),),
            "kg": (_candidate("symbol:kg", path="pkg/kg.py"),),
            "embedding": (
                _candidate("symbol:emb", path="pkg/emb.py", score=0.2, semantic_authority=False),
            ),
        },
    )
    signals = {
        signal
        for nomination in receipt.candidates
        for signal, _ in nomination.signal_evidence
    }
    assert DoctorCandidateSignal.EXACT_SYMBOL.value in signals
    assert DoctorCandidateSignal.EXACT_LINEAGE.value in signals
    assert DoctorCandidateSignal.LEXICAL.value in signals
    assert DoctorCandidateSignal.KNOWLEDGE_GRAPH.value in signals
    assert DoctorCandidateSignal.VECTOR.value in signals
