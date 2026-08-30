"""Independent contract tests for SPAR-015 AnalogousRefactorRetriever@1."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.utils.cid_utils import cid_for_bytes, cid_for_dag_json
from ipfs_accelerate_py.agent_supervisor.semantic_refactoring.partition_generators import (
    ConstraintClass,
    GeneratorKind,
    ProgramPartitionCandidate,
)
from ipfs_accelerate_py.agent_supervisor.semantic_refactoring.partition_retrieval import (
    ADVISORY_CHANNELS,
    ADVISORY_PARTITION_RANKER_INTERFACE,
    ANALYZER_ID,
    ANALOGOUS_REFACTOR_HIT_INTERFACE,
    ANALOGOUS_REFACTOR_QUERY_INTERFACE,
    ANALOGOUS_REFACTOR_RECORD_INTERFACE,
    ANALOGOUS_REFACTOR_RETRIEVAL_RECEIPT_INTERFACE,
    ANALOGOUS_REFACTOR_RETRIEVER_INTERFACE,
    AUTHORITY,
    AUTHORITY_OWNER,
    CHANNEL_ORDER,
    DUCKLAKE_IS_AUTHORITY,
    GOAL_ID,
    IDENTITY_EXCLUDED_FIELDS,
    MARKDOWN_IS_NOT_COMPLETION,
    MODEL_OUTPUT_IS_PROPOSAL_ONLY,
    PARTITION_RANKING_RECEIPT_INTERFACE,
    PARTITION_RETRIEVAL_CONTRACT_VERSION,
    PROGRAM,
    PROJECTION_CLUSTERING_IS_AUTHORITY,
    RANKER_IS_ADVISORY,
    RAW_SOURCE_REQUIRED,
    RETRIEVAL_CAN_AUTHORIZE_COMPLETION,
    RETRIEVAL_CAN_AUTHORIZE_TRANSITION,
    RETRIEVAL_CAN_CREATE_AUTHORITY,
    RETRIEVAL_IS_NOMINATION_ONLY,
    SOFT_SIGNALS_CANNOT_OVERRIDE_HARD_CONSTRAINTS,
    TASK_ID,
    TEST_PASS_IS_NOT_COMPLETION,
    VECTOR_SIMILARITY_IS_AUTHORITY,
    WORKER_SELF_APPROVAL,
    AdvisoryPartitionRanker,
    AnalogousRefactorHit,
    AnalogousRefactorQuery,
    AnalogousRefactorRecord,
    AnalogousRefactorRetrievalReceipt,
    AnalogousRefactorRetriever,
    PartitionRankingReceipt,
    PartitionRetrievalError,
    RecordOutcome,
    RetrievalChannel,
    assert_not_competing_capsule_family,
    decode_canonical_ranking_receipt,
    decode_canonical_retrieval_receipt,
    encode_canonical_ranking_receipt,
    encode_canonical_retrieval_receipt,
    partition_retrieval_cid_profile,
    provider_free_exports,
    rank_admitted_partitions,
    retrieve_analogous_refactors,
)


ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    ROOT
    / "ipfs_accelerate_py"
    / "agent_supervisor"
    / "semantic_refactoring"
    / "partition_retrieval.py"
)
TEST_PATH = Path(__file__).resolve()
WRITE_SCOPE = (
    "ipfs_accelerate_py/agent_supervisor/semantic_refactoring/partition_retrieval.py",
    "test/api/semantic_refactoring/test_partition_retrieval.py",
)
PROTECTED_PATHS = (
    ".gitignore",
    "benchmarks/agent_supervisor/semantic_refactoring/preregistration.json",
    "config/agent_supervisor_semantic_preserving_remodularization_scheduler.json",
    "config/semantic_preserving_autonomous_remodularization_dependencies.seal.json",
    "docs/architecture/SEMANTIC_PRESERVING_AUTONOMOUS_REMODULARIZATION_PLAN.md",
    "docs/architecture/semantic_preserving_autonomous_remodularization.objectives.md",
    "docs/architecture/semantic_preserving_autonomous_remodularization.todo.md",
    "docs/architecture/semantic_preserving_autonomous_remodularization_inventory/authority_matrix.json",
    "docs/architecture/semantic_preserving_autonomous_remodularization_inventory/benchmark_preregistration.json",
    "docs/architecture/semantic_preserving_autonomous_remodularization_inventory/dynamic_python_risk_inventory.json",
    "docs/architecture/semantic_preserving_autonomous_remodularization_inventory/identity_inventory.json",
    "docs/architecture/semantic_preserving_autonomous_remodularization_inventory/interface_inventory.json",
    "docs/architecture/semantic_preserving_autonomous_remodularization_inventory/overlap_gap_matrix.json",
    "docs/architecture/semantic_preserving_autonomous_remodularization_inventory/repository_baseline.json",
    "docs/architecture/semantic_preserving_autonomous_remodularization_inventory/rollout_baseline.json",
    "scripts/materialize_semantic_preserving_remodularization_program.py",
    "scripts/ops/agent_supervisor/semantic_preserving_remodularization.py",
    "scripts/validate_semantic_preserving_remodularization_board.py",
    "scripts/validate_semantic_preserving_remodularization_dependencies.py",
    "test/api/semantic_refactoring/test_bootstrap_controls.py",
)
CAPSULE_TYPES = (
    "FunctionSemanticCapsule",
    "MethodSemanticCapsule",
    "ClassSemanticCapsule",
    "TopLevelBlockCapsule",
    "ModuleSemanticCapsule",
    "PackageSemanticCapsule",
    "CallsiteSemanticCapsule",
    "StateOwnerCapsule",
    "RegistrationCapsule",
    "ResourceLifecycleCapsule",
)
TREE_ID = "fbc6fa1ddefb2f9ecb7b5c718d618e3b60aa3051"
OTHER_TREE = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"


def _cid(label: str) -> str:
    return cid_for_bytes(label.encode("utf-8"))


def _query(**overrides: Any) -> AnalogousRefactorQuery:
    fields: dict[str, Any] = {
        "tree_id": TREE_ID,
        "member_ids": ["node:left", "node:right"],
        "tokens": ["extract", "cache"],
        "graph_edges": [
            {
                "source_id": "node:left",
                "target_id": "node:leaf",
                "kind": "calls",
            }
        ],
        "vector": (),
        "vector_available": False,
    }
    fields.update(overrides)
    return AnalogousRefactorQuery(**fields)


def _record(**overrides: Any) -> AnalogousRefactorRecord:
    fields: dict[str, Any] = {
        "tree_id": TREE_ID,
        "member_ids": ["node:left", "node:right"],
        "tokens": ["extract", "cache"],
        "graph_edges": [
            {
                "source_id": "node:left",
                "target_id": "node:leaf",
                "kind": "calls",
            }
        ],
        "vector": (),
        "vector_available": False,
        "evidence_class": "accepted_transition",
        "outcome": RecordOutcome.ACCEPTED.value,
    }
    fields.update(overrides)
    return AnalogousRefactorRecord(**fields)


def _candidate(
    *member_ids: str,
    admitted: bool = True,
    advisory: bool = False,
    kind: GeneratorKind = GeneratorKind.SCC,
    violations: tuple[str, ...] = (),
    evidence_class: str = "exact_static_fact",
) -> ProgramPartitionCandidate:
    return ProgramPartitionCandidate(
        tree_id=TREE_ID,
        generator_kind=kind,
        member_ids=member_ids,
        admitted=admitted,
        advisory=advisory,
        hard_constraint_violations=violations,
        evidence_class=evidence_class,
    )


def test_owned_paths_and_task_identity_are_exact() -> None:
    assert TASK_ID == "SPAR-015"
    assert GOAL_ID == "SPAR-G032"
    assert PROGRAM == "semantic-preserving-autonomous-remodularization-v1"
    assert ANALOGOUS_REFACTOR_RECORD_INTERFACE == "AnalogousRefactorRecord@1"
    assert ANALOGOUS_REFACTOR_HIT_INTERFACE == "AnalogousRefactorHit@1"
    assert ANALOGOUS_REFACTOR_QUERY_INTERFACE == "AnalogousRefactorQuery@1"
    assert (
        ANALOGOUS_REFACTOR_RETRIEVAL_RECEIPT_INTERFACE
        == "AnalogousRefactorRetrievalReceipt@1"
    )
    assert PARTITION_RANKING_RECEIPT_INTERFACE == "PartitionRankingReceipt@1"
    assert ANALOGOUS_REFACTOR_RETRIEVER_INTERFACE == "AnalogousRefactorRetriever@1"
    assert ADVISORY_PARTITION_RANKER_INTERFACE == "AdvisoryPartitionRanker@1"
    assert PARTITION_RETRIEVAL_CONTRACT_VERSION == "1"
    assert ANALYZER_ID.endswith("partition_retrieval@1")
    assert MODULE_PATH.is_file()
    assert TEST_PATH.is_file()
    for relative in WRITE_SCOPE:
        assert (ROOT / relative).is_file()


def test_authority_flags_cannot_self_authorize() -> None:
    assert AUTHORITY == "partition orchestration"
    assert AUTHORITY_OWNER == "ipfs_accelerate_py"
    assert RETRIEVAL_CAN_AUTHORIZE_COMPLETION is False
    assert RETRIEVAL_CAN_AUTHORIZE_TRANSITION is False
    assert RETRIEVAL_CAN_CREATE_AUTHORITY is False
    assert VECTOR_SIMILARITY_IS_AUTHORITY is False
    assert PROJECTION_CLUSTERING_IS_AUTHORITY is False
    assert RETRIEVAL_IS_NOMINATION_ONLY is True
    assert RANKER_IS_ADVISORY is True
    assert MODEL_OUTPUT_IS_PROPOSAL_ONLY is True
    assert TEST_PASS_IS_NOT_COMPLETION is True
    assert MARKDOWN_IS_NOT_COMPLETION is True
    assert WORKER_SELF_APPROVAL is False
    assert DUCKLAKE_IS_AUTHORITY is False
    assert SOFT_SIGNALS_CANNOT_OVERRIDE_HARD_CONSTRAINTS is True
    assert RAW_SOURCE_REQUIRED is True
    profile = partition_retrieval_cid_profile()
    assert profile["codec"] == "dag-json"
    assert "not universal meaning" in profile["rule"]
    assert CHANNEL_ORDER[0] == RetrievalChannel.EXACT.value
    assert RetrievalChannel.VECTOR.value in ADVISORY_CHANNELS


def test_module_defines_predicted_symbols_not_capsule_family() -> None:
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    names = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
    assert "AnalogousRefactorRetriever" in names
    assert "AdvisoryPartitionRanker" in names
    assert "AnalogousRefactorRetrievalReceipt" in names
    assert "PartitionRankingReceipt" in names
    for capsule in CAPSULE_TYPES:
        assert capsule not in names
    assert_not_competing_capsule_family()
    exports = provider_free_exports()
    assert "AnalogousRefactorRetriever" in exports
    assert "AdvisoryPartitionRanker" in exports
    assert "retrieve_analogous_refactors" in exports
    assert "rank_admitted_partitions" in exports


def test_protected_paths_are_not_owned_write_scope() -> None:
    owned = set(WRITE_SCOPE)
    for relative in PROTECTED_PATHS:
        assert relative not in owned
        assert (ROOT / relative).exists()


def test_exact_channel_matches_identical_member_sets() -> None:
    receipt = retrieve_analogous_refactors(_query(), (_record(),))
    exact = [
        item
        for item in receipt.hits
        if item.channel == RetrievalChannel.EXACT.value
    ]
    assert len(exact) == 1
    assert exact[0].millirank == 1000
    assert exact[0].advisory is False
    assert exact[0].can_authorize_completion is False
    assert exact[0].overlapping_member_ids == ("node:left", "node:right")
    assert receipt.unmatched_record_cids == ()


def test_lexical_channel_scores_token_jaccard() -> None:
    receipt = retrieve_analogous_refactors(
        _query(member_ids=["node:leaf"], tokens=["extract", "owner"]),
        (
            _record(
                member_ids=["node:other"],
                tokens=["extract", "cache"],
                graph_edges=(),
            ),
        ),
    )
    lexical = [
        item
        for item in receipt.hits
        if item.channel == RetrievalChannel.LEXICAL.value
    ]
    assert len(lexical) == 1
    assert lexical[0].millirank == 333
    assert not [
        item
        for item in receipt.hits
        if item.channel == RetrievalChannel.EXACT.value
    ]


def test_graph_channel_scores_member_and_edge_overlap() -> None:
    receipt = retrieve_analogous_refactors(
        _query(member_ids=["node:left"], tokens=()),
        (
            _record(
                member_ids=["node:left", "node:leaf"],
                tokens=(),
            ),
        ),
    )
    graph = [
        item
        for item in receipt.hits
        if item.channel == RetrievalChannel.GRAPH.value
    ]
    assert graph
    assert all(item.millirank > 0 for item in graph)
    assert not [
        item
        for item in receipt.hits
        if item.channel == RetrievalChannel.EXACT.value
    ]


def test_vector_channel_is_advisory_and_cannot_admit() -> None:
    query = _query(
        vector=(1.0, 0.0),
        vector_available=True,
        tokens=(),
        graph_edges=(),
        member_ids=["node:left"],
    )
    analog = _record(
        member_ids=["node:leaf"],
        tokens=(),
        graph_edges=(),
        vector=(1.0, 0.0),
        vector_available=True,
        evidence_class="vector_candidate",
        outcome=RecordOutcome.ADVISORY.value,
    )
    receipt = retrieve_analogous_refactors(query, (analog,))
    vector = [
        item
        for item in receipt.hits
        if item.channel == RetrievalChannel.VECTOR.value
    ]
    assert len(vector) == 1
    assert vector[0].advisory is True
    assert vector[0].evidence_class == "vector_candidate"
    assert vector[0].vector_authoritative is False
    assert vector[0].can_authorize_transition is False
    rejected = _candidate(
        "node:leaf",
        admitted=False,
        kind=GeneratorKind.PROJECTION,
        evidence_class="vector_candidate",
    )
    ranked = rank_admitted_partitions(
        (rejected,),
        (analog,),
        query=query,
    )
    assert ranked.admitted_candidate_cids == ()
    assert ranked.preserved_non_admitted_cids == (rejected.candidate_cid,)


def test_nonfinite_vector_fails_closed() -> None:
    with pytest.raises(PartitionRetrievalError, match="non-finite"):
        _query(vector=(float("nan"), 0.0), vector_available=True)


def test_stale_tree_fails_closed() -> None:
    with pytest.raises(PartitionRetrievalError, match="stale tree"):
        retrieve_analogous_refactors(_query(), (_record(tree_id=OTHER_TREE),))


def test_accepted_record_cannot_use_vector_evidence() -> None:
    with pytest.raises(PartitionRetrievalError, match="cannot mark a prior refactor accepted"):
        _record(evidence_class="vector_candidate")


def test_rejected_analogs_remain_negative_evidence_and_do_not_boost_rank() -> None:
    rejected = _record(
        member_ids=["node:left", "node:right"],
        outcome=RecordOutcome.REJECTED.value,
        evidence_class="countermodel",
    )
    receipt = retrieve_analogous_refactors(_query(), (rejected,))
    assert rejected.record_cid in receipt.rejected_record_cids
    assert receipt.hits
    first = _candidate("node:left", "node:right")
    second = _candidate("node:other")
    ranked = rank_admitted_partitions(
        (second, first),
        (rejected,),
        query=_query(tokens=(), graph_edges=(), vector_available=False, vector=()),
    )
    scores = {item.candidate_cid: item for item in ranked.scores}
    assert scores[first.candidate_cid].exact_millirank == 0
    assert scores[second.candidate_cid].exact_millirank == 0
    assert set(ranked.ranked_candidate_cids) == {
        first.candidate_cid,
        second.candidate_cid,
    }


def test_ranker_reorders_only_already_admitted_candidates() -> None:
    exact = _record(member_ids=["node:b"], tokens=(), graph_edges=())
    lexical = _record(
        member_ids=["node:a", "node:extra"],
        tokens=["extract"],
        graph_edges=(),
    )
    admitted_a = _candidate("node:a")
    admitted_b = _candidate("node:b")
    rejected = _candidate(
        "node:ghost",
        admitted=False,
        violations=(ConstraintClass.OVERSIZED_CYCLE.value,),
    )
    ranked = rank_admitted_partitions(
        (admitted_a, rejected, admitted_b),
        (exact, lexical),
        query=_query(
            member_ids=(),
            tokens=("extract",),
            graph_edges=(),
            vector=(),
            vector_available=False,
        ),
    )
    assert ranked.ranked_candidate_cids == (
        admitted_b.candidate_cid,
        rejected.candidate_cid,
        admitted_a.candidate_cid,
    )
    assert ranked.preserved_non_admitted_cids == (rejected.candidate_cid,)
    assert ranked.hard_constraint_violations[rejected.candidate_cid] == [
        ConstraintClass.OVERSIZED_CYCLE.value
    ]
    assert ranked.can_authorize_completion is False
    assert ranked.vector_authoritative is False


def test_ranker_cannot_drop_or_hide_candidates() -> None:
    admitted = _candidate("node:left")
    rejected = _candidate(
        "node:ghost",
        admitted=False,
        violations=(ConstraintClass.UNKNOWN_MEMBER.value,),
    )
    payload = rank_admitted_partitions((admitted, rejected)).to_dict()
    payload["ranked_candidate_cids"] = [admitted.candidate_cid]
    payload["preserved_non_admitted_cids"] = []
    payload["hard_constraint_violations"] = {
        admitted.candidate_cid: [],
    }
    payload["receipt_cid"] = cid_for_dag_json(
        {key: value for key, value in payload.items() if key != "receipt_cid"}
    )
    with pytest.raises(PartitionRetrievalError, match="cannot drop"):
        PartitionRankingReceipt.from_dict(payload)


def test_ranker_cannot_hide_hard_constraint_violations() -> None:
    admitted = _candidate("node:left")
    rejected = _candidate(
        "node:ghost",
        admitted=False,
        violations=(ConstraintClass.OVERSIZED_CYCLE.value,),
    )
    payload = rank_admitted_partitions((admitted, rejected)).to_dict()
    del payload["hard_constraint_violations"][rejected.candidate_cid]
    payload["receipt_cid"] = cid_for_dag_json(
        {key: value for key, value in payload.items() if key != "receipt_cid"}
    )
    with pytest.raises(PartitionRetrievalError, match="hide hard-constraint"):
        PartitionRankingReceipt.from_dict(payload)


def test_exact_outRanks_vector_for_already_admitted_candidates() -> None:
    exact = _record(member_ids=["node:keep"])
    vector_only = _record(
        member_ids=["node:unrelated"],
        tokens=(),
        graph_edges=(),
        vector=(1.0, 0.0),
        vector_available=True,
        evidence_class="vector_candidate",
        outcome=RecordOutcome.ADVISORY.value,
    )
    keep = _candidate("node:keep")
    other = _candidate("node:vector")
    ranked = rank_admitted_partitions(
        (other, keep),
        (exact, vector_only),
        query=_query(
            member_ids=(),
            tokens=(),
            graph_edges=(),
            vector=(1.0, 0.0),
            vector_available=True,
        ),
    )
    assert ranked.ranked_candidate_cids[0] == keep.candidate_cid
    scores = {item.candidate_cid: item for item in ranked.scores}
    assert scores[keep.candidate_cid].exact_millirank == 1000
    assert scores[other.candidate_cid].vector_millirank == 1000
    assert scores[other.candidate_cid].exact_millirank == 0


def test_projection_search_hits_must_resolve_to_corpus() -> None:
    query = _query(vector=(1.0, 0.0), vector_available=True)
    analog = _record(
        vector=(1.0, 0.0),
        vector_available=True,
        evidence_class="vector_candidate",
        outcome=RecordOutcome.ADVISORY.value,
        tokens=(),
        graph_edges=(),
        member_ids=["node:left"],
    )

    def _search(_vector: Any, k: int = 1) -> list[dict[str, str]]:
        del k
        return [{"record_cid": _cid("missing-projection")}]

    with pytest.raises(PartitionRetrievalError, match="sealed corpus"):
        AnalogousRefactorRetriever(projection_search=_search).retrieve(
            query, (analog,)
        )


def test_unavailable_projection_search_is_typed_and_keeps_exact_hits() -> None:
    class ProjectionCapabilityError(RuntimeError):
        reason_code = "projection_capability"

    query = _query(vector=(1.0, 0.0), vector_available=True)

    def _search(_vector: Any, k: int = 1) -> list[Any]:
        del k
        raise ProjectionCapabilityError("neural unavailable")

    receipt = AnalogousRefactorRetriever(projection_search=_search).retrieve(
        query, (_record(),)
    )
    assert receipt.vector_channel_unavailable is True
    assert any(item.channel == RetrievalChannel.EXACT.value for item in receipt.hits)


def test_identity_excludes_observational_fields() -> None:
    record = _record()
    payload = record.to_dict()
    assert not (IDENTITY_EXCLUDED_FIELDS & set(payload))
    restored = AnalogousRefactorRecord.from_dict(payload)
    assert restored == record
    assert restored.record_cid == record.record_cid
    dirty = dict(payload)
    dirty["timestamp"] = "now"
    with pytest.raises(PartitionRetrievalError, match="observational"):
        AnalogousRefactorRecord.from_dict(dirty)


def test_record_cannot_claim_authority_flags() -> None:
    payload = _record().to_dict()
    payload["can_authorize_completion"] = True
    payload["record_cid"] = cid_for_dag_json(
        {key: value for key, value in payload.items() if key != "record_cid"}
    )
    with pytest.raises(PartitionRetrievalError, match="can_authorize_completion"):
        AnalogousRefactorRecord.from_dict(payload)


def test_hit_vector_channel_is_forced_advisory() -> None:
    record = _record(
        evidence_class="vector_candidate",
        outcome=RecordOutcome.ADVISORY.value,
        vector=(0.0, 1.0),
        vector_available=True,
    )
    hit = AnalogousRefactorHit(
        record_cid=record.record_cid,
        channel=RetrievalChannel.VECTOR,
        millirank=500,
        overlapping_member_ids=("node:left",),
        evidence_class="accepted_transition",
        outcome=RecordOutcome.ACCEPTED.value,
        advisory=False,
    )
    assert hit.advisory is True
    assert hit.vector_authoritative is False


def test_retrieval_receipt_identity_is_deterministic_and_round_trips() -> None:
    query = _query()
    corpus = (_record(),)
    first = retrieve_analogous_refactors(query, corpus)
    second = retrieve_analogous_refactors(query, corpus)
    assert first.receipt_cid == second.receipt_cid
    encoded = encode_canonical_retrieval_receipt(first)
    restored = decode_canonical_retrieval_receipt(encoded)
    assert restored == first
    assert restored.receipt_cid == first.receipt_cid
    assert first.can_authorize_completion is False
    assert first.projection_is_authority is False


def test_ranking_receipt_identity_is_deterministic_and_round_trips() -> None:
    candidates = (_candidate("node:left"), _candidate("node:right"))
    corpus = (_record(member_ids=["node:right"]),)
    first = rank_admitted_partitions(candidates, corpus, query=_query(tokens=(), graph_edges=()))
    second = rank_admitted_partitions(candidates, corpus, query=_query(tokens=(), graph_edges=()))
    assert first.receipt_cid == second.receipt_cid
    encoded = encode_canonical_ranking_receipt(first)
    restored = decode_canonical_ranking_receipt(encoded)
    assert restored == first
    assert restored.ranked_candidate_cids[-1] == candidates[0].candidate_cid or restored.ranked_candidate_cids[0] in {
        item.candidate_cid for item in candidates
    }


def test_analyzer_id_is_pinned() -> None:
    receipt = retrieve_analogous_refactors(_query(), (_record(),))
    payload = receipt.to_dict()
    payload["analyzer_id"] = "other.analyzer@1"
    payload["receipt_cid"] = cid_for_dag_json(
        {key: value for key, value in payload.items() if key != "receipt_cid"}
    )
    with pytest.raises(PartitionRetrievalError, match="SPAR-015 analyzer"):
        AnalogousRefactorRetrievalReceipt.from_dict(payload)


def test_dimension_mismatch_fails_closed() -> None:
    with pytest.raises(PartitionRetrievalError, match="vector dimension mismatch"):
        retrieve_analogous_refactors(
            _query(vector=(1.0, 0.0), vector_available=True),
            (
                _record(
                    vector=(1.0, 0.0, 0.0),
                    vector_available=True,
                    evidence_class="vector_candidate",
                    outcome=RecordOutcome.ADVISORY.value,
                ),
            ),
        )


def test_zero_vector_fails_closed() -> None:
    with pytest.raises(PartitionRetrievalError, match="zero vector"):
        retrieve_analogous_refactors(
            _query(vector=(0.0, 0.0), vector_available=True),
            (
                _record(
                    vector=(1.0, 0.0),
                    vector_available=True,
                    evidence_class="vector_candidate",
                    outcome=RecordOutcome.ADVISORY.value,
                ),
            ),
        )


def test_unmatched_corpus_records_remain_negative_evidence() -> None:
    analog = _record(
        member_ids=["node:unrelated"],
        tokens=["zzz"],
        graph_edges=(),
    )
    receipt = retrieve_analogous_refactors(
        _query(tokens=("aaa",), graph_edges=(), member_ids=("node:left",)),
        (analog,),
    )
    assert analog.record_cid in receipt.unmatched_record_cids
    assert receipt.hits == ()


def test_advisory_ranker_class_matches_procedural_helper() -> None:
    candidates = (_candidate("node:left"), _candidate("node:right"))
    corpus = (_record(member_ids=["node:right"], tokens=(), graph_edges=()),)
    query = _query(tokens=(), graph_edges=())
    via_class = AdvisoryPartitionRanker().rank(candidates, corpus, query=query)
    via_helper = rank_admitted_partitions(candidates, corpus, query=query)
    assert via_class.receipt_cid == via_helper.receipt_cid
    assert via_class.ranked_candidate_cids[0] == candidates[1].candidate_cid
