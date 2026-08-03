"""Adversarial conformance tests for program-logic hypothesis nomination (LPR-009)."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.program_logic_hypothesis_retrieval import (
    HypothesisCandidateSet,
    HypothesisHardGateFacts,
    HypothesisNominationDisposition,
    HypothesisQuery,
    HypothesisRetrievalBindingError,
    HypothesisRetrievalBounds,
    HypothesisRetrievalBoundsError,
    HypothesisSignal,
    LogicHypothesisNomination,
    ProgramLogicHypothesisRetriever,
    REJECTION_BODY_OR_SECRET,
    REJECTION_COMPATIBILITY_AS_ADMISSION,
    REJECTION_CROSS_GAP_OR_GOAL,
    REJECTION_EXCLUDED_PREMISE,
    REJECTION_FORGED,
    REJECTION_PARTIAL,
    REJECTION_POISONED,
    REJECTION_SEMANTIC_AUTHORITY_CLAIM,
    REJECTION_STALE_OR_CROSS_ROOT,
    REJECTION_SUFFICIENCY_CLAIM,
    candidate_set_identity,
    retrieve_program_logic_hypotheses,
)
from ipfs_accelerate_py.agent_supervisor.analysis.program_logic_prediction_contracts import (
    GapDisposition,
    GapMissingClass,
    HypothesisDisposition,
    LogicGap,
    LogicSubgoal,
    ProgramLogicAuthorityRoots,
    SourceAuthorityClass,
    SourceRouteKind,
    SubgoalDisposition,
    TacticianSearchPlan,
)


@pytest.fixture
def roots() -> ProgramLogicAuthorityRoots:
    return ProgramLogicAuthorityRoots(
        repository_id="repository:fixture",
        objective_id="objective:fixture",
        trace_id="trace:fixture",
        change_id="change:fixture",
        consumer_id="consumer:fixture",
        forest_id="forest:fixture",
        tree_id="tree:fixture",
        overlay_id="overlay:fixture",
        graph_id="graph:fixture",
        index_id="index:fixture",
        corpus_id="corpus:fixture",
        model_id="model:fixture",
        translator_id="translator:fixture",
        toolchain_id="toolchain:fixture",
        policy_id="policy:fixture",
        environment_id="environment:fixture",
    )


def _gap(roots: ProgramLogicAuthorityRoots, **extra: object) -> LogicGap:
    values: dict[str, object] = {
        "roots": roots,
        "gap_id": "gap:missing-context",
        "goal_id": "goal:repair-caller",
        "missing_class": GapMissingClass.VALUE,
        "disposition": GapDisposition.REQUIRED,
        "observed_fact_ref": "fact:observed-missing",
        "required_fact_ref": "fact:required-context",
        "discrepancy_ref": "fact:discrepancy",
        "dependency_slice_refs": ("slice:caller",),
        "candidate_source_routes": (
            SourceRouteKind.LOCAL_STATIC,
            SourceRouteKind.DATAFLOW,
            SourceRouteKind.GRAPH,
            SourceRouteKind.VECTOR,
        ),
        "severity": "mandatory",
        "automation_eligible": True,
        "invalidation_refs": ("tree:fixture",),
    }
    values.update(extra)
    return LogicGap(**values)


def _plan(roots: ProgramLogicAuthorityRoots, **extra: object) -> TacticianSearchPlan:
    subgoals = extra.pop(
        "subgoals",
        (
            LogicSubgoal(
                subgoal_id="subgoal:prove-context",
                goal_id="goal:repair-caller",
                disposition=SubgoalDisposition.PLANNED,
                claim_ref="claim:context-available",
                source_route=SourceRouteKind.DATAFLOW,
                source_authority=SourceAuthorityClass.AUTHORITATIVE,
                score_millipercent=25_000,
            ),
        ),
    )
    values: dict[str, object] = {
        "roots": roots,
        "plan_id": "plan:fixture",
        "goal_ids": ("goal:repair-caller",),
        "ordered_source_routes": (
            SourceRouteKind.LOCAL_STATIC,
            SourceRouteKind.DATAFLOW,
            SourceRouteKind.GRAPH,
            SourceRouteKind.VECTOR,
        ),
        "query_refs": ("query:fixture",),
        "selected_premise_ids": ("premise:type-context",),
        "excluded_premise_ids": ("premise:poisoned",),
        "subgoals": subgoals,
        "planner_id": "planner:fixture",
        "model_id": "model:fixture",
        "config_id": "config:fixture",
        "invalidation_refs": ("tree:fixture",),
    }
    values.update(extra)
    return TacticianSearchPlan(**values)


def _hit(
    consequence: str = "consequence:reuse-local-ctx",
    *,
    construction: str = "construction:local_ctx",
    **extra: object,
) -> dict[str, object]:
    value: dict[str, object] = {
        "claimed_consequence_ref": consequence,
        "construction_ref": construction,
        "value_ref": "value:local_ctx",
        "evidence_refs": ("evidence:fixture",),
        "history_reviewed": True,
        "counterexample_target_ref": "counterexample:missing-context",
        "information_content_ref": "info:request-context",
        "type_compatible": True,
        "same_name": False,
        "same_type": True,
        "nomination_score_millipercent": 12_500,
    }
    value.update(extra)
    return value


def test_union_is_deterministic_deduplicated_and_non_authoritative(
    roots: ProgramLogicAuthorityRoots,
) -> None:
    gap = _gap(roots)
    # Empty subgoals so auto-projection does not add a second nomination.
    plan = _plan(roots, subgoals=())
    retriever = ProgramLogicHypothesisRetriever(roots)
    signals = {
        "vector": (_hit(score=0.9, semantic_authority=False),),
        "analytical_construction": (_hit(),),
        "existing_value": (_hit(),),
        "graph": (_hit(),),
    }

    forward = retriever.retrieve(gap, plan, candidates_by_signal=signals)
    reverse = retriever.retrieve(
        gap,
        plan,
        candidates_by_signal=dict(reversed(tuple(signals.items()))),
    )

    assert forward.content_id == reverse.content_id
    assert forward.query_id == forward.query.content_id
    # Same consequence/construction collapses to one nomination with multi-signal evidence.
    assert len(forward.nominations) == 1
    nomination = forward.nominations[0]
    assert nomination.disposition is HypothesisNominationDisposition.NOMINATED
    assert set(signal for signal, _ in nomination.signal_evidence) == {
        HypothesisSignal.VECTOR.value,
        HypothesisSignal.ANALYTICAL_CONSTRUCTION.value,
        HypothesisSignal.EXISTING_VALUE.value,
        HypothesisSignal.GRAPH_NEIGHBORHOOD.value,
    }
    assert nomination.hypothesis.semantic_authority is False
    assert nomination.semantic_authority is False
    assert nomination.hard_gate_facts.information_sufficiency is False
    assert nomination.hard_gate_facts.type_compatible is True
    assert nomination.write_paths == forward.write_paths == ()
    assert forward.semantic_authority is False
    assert forward.admitted_hypothesis_id == ""
    assert forward.candidate_set_id == candidate_set_identity(forward.nominations)
    assert forward.no_candidate is False
    assert type(forward).from_dict(forward.to_record()).content_id == forward.content_id
    assert HypothesisQuery.from_dict(forward.query.to_record()).content_id == forward.query.content_id


def test_unions_all_declared_signal_families(roots: ProgramLogicAuthorityRoots) -> None:
    gap = _gap(roots)
    plan = _plan(roots, subgoals=())  # avoid auto-projecting subgoals into one family
    families = {
        "analytical_construction": _hit("consequence:analytical", construction="c:a"),
        "existing_value": _hit("consequence:value", construction="c:v"),
        "constructor_adapter": _hit("consequence:ctor", construction="c:ctor"),
        "theorem_premise": _hit(
            "consequence:premise",
            construction="c:premise",
            selected_premise_ids=("premise:type-context",),
        ),
        "dataflow": _hit("consequence:df", construction="c:df"),
        "graph_neighborhood": _hit("consequence:graph", construction="c:g"),
        "schema": _hit("consequence:schema", construction="c:s"),
        "lineage": _hit("consequence:lineage", construction="c:l"),
        "test_spec_analogue": _hit("consequence:test", construction="c:t"),
        "lexical": _hit("consequence:lex", construction="c:lex"),
        "vector": _hit("consequence:vec", construction="c:vec", score=0.42),
        "tactician_subgoal": _hit("consequence:sub", construction="c:sub"),
        "learned_model": _hit("consequence:model", construction="c:model"),
    }
    receipt = ProgramLogicHypothesisRetriever(roots).retrieve(
        gap,
        plan,
        candidates_by_signal={name: (payload,) for name, payload in families.items()},
    )
    assert len(receipt.nominations) == len(families)
    signal_root_names = {signal for signal, _ in receipt.signal_roots}
    assert signal_root_names == {item.value for item in HypothesisSignal}
    assert all(item.hypothesis.semantic_authority is False for item in receipt.nominations)
    assert all(
        item.disposition is HypothesisNominationDisposition.NOMINATED
        for item in receipt.nominations
    )

    # Alias coverage for BM25 / history / model / graph.
    aliased = ProgramLogicHypothesisRetriever(roots).retrieve(
        gap,
        plan,
        candidates_by_signal={
            "bm25": (_hit("consequence:bm25", construction="c:bm25"),),
            "history": (_hit("consequence:hist", construction="c:hist"),),
            "llm": (_hit("consequence:llm", construction="c:llm"),),
            "kg": (_hit("consequence:kg", construction="c:kg"),),
            "adapter": (_hit("consequence:ad", construction="c:ad"),),
        },
    )
    signals = {
        signal
        for nomination in aliased.nominations
        for signal, _ in nomination.signal_evidence
    }
    assert HypothesisSignal.LEXICAL.value in signals
    assert HypothesisSignal.LINEAGE.value in signals
    assert HypothesisSignal.LEARNED_MODEL.value in signals
    assert HypothesisSignal.GRAPH_NEIGHBORHOOD.value in signals
    assert HypothesisSignal.CONSTRUCTOR_ADAPTER.value in signals


def test_hard_gate_facts_separate_from_ranking_scores(
    roots: ProgramLogicAuthorityRoots,
) -> None:
    gap = _gap(roots)
    plan = _plan(roots, subgoals=())
    receipt = ProgramLogicHypothesisRetriever(roots).retrieve(
        gap,
        plan,
        candidates_by_signal={
            "vector": (
                _hit(
                    nomination_score_millipercent=99_000,
                    type_compatible=False,
                    effect_compatible=True,
                    same_name=True,
                    same_type=True,
                    similarity=0.99,
                ),
            ),
        },
    )
    nomination = receipt.nominations[0]
    assert nomination.nomination_score_millipercent == 99_000
    assert nomination.hypothesis.nomination_score_millipercent == 99_000
    # High score does not flip hard facts or establish sufficiency.
    assert nomination.hard_gate_facts.type_compatible is False
    assert nomination.hard_gate_facts.effect_compatible is True
    assert nomination.hard_gate_facts.same_name is True
    assert nomination.hard_gate_facts.same_type is True
    assert nomination.hard_gate_facts.information_sufficiency is False
    assert nomination.hard_gate_facts.similarity_millipercent == 99_000
    assert nomination.disposition is HypothesisNominationDisposition.NOMINATED


def test_same_name_type_similarity_cannot_establish_sufficiency(
    roots: ProgramLogicAuthorityRoots,
) -> None:
    gap = _gap(roots)
    plan = _plan(roots, subgoals=())
    receipt = ProgramLogicHypothesisRetriever(roots).retrieve(
        gap,
        plan,
        candidates_by_signal={
            "lexical": (
                _hit(
                    "consequence:name-only",
                    construction="c:name",
                    same_name=True,
                    same_type=True,
                    similarity=1.0,
                    information_sufficiency=True,
                ),
            ),
            "vector": (
                _hit(
                    "consequence:sim-only",
                    construction="c:sim",
                    name_match_sufficient=True,
                ),
            ),
            "existing_value": (
                _hit(
                    "consequence:type-only",
                    construction="c:type",
                    type_match_sufficient=True,
                ),
            ),
        },
    )
    by_c = {
        item.hypothesis.claimed_consequence_ref: item for item in receipt.nominations
    }
    assert REJECTION_SUFFICIENCY_CLAIM in by_c["consequence:name-only"].diagnostics
    assert REJECTION_SUFFICIENCY_CLAIM in by_c["consequence:sim-only"].diagnostics
    assert REJECTION_SUFFICIENCY_CLAIM in by_c["consequence:type-only"].diagnostics
    assert all(
        item.disposition is HypothesisNominationDisposition.REJECTED
        for item in receipt.nominations
    )
    assert all(
        item.hard_gate_facts.information_sufficiency is False
        for item in receipt.nominations
    )


def test_adversarial_targets_retained_with_stable_diagnostics(
    roots: ProgramLogicAuthorityRoots,
) -> None:
    gap = _gap(roots)
    plan = _plan(roots, subgoals=())
    receipt = ProgramLogicHypothesisRetriever(roots).retrieve(
        gap,
        plan,
        candidates_by_signal={
            "analytical_construction": (
                _hit("consequence:stale", construction="c:stale", tree_id="tree:other"),
                _hit("consequence:forged", construction="c:forged", forged_history=True),
                _hit(
                    "consequence:authority",
                    construction="c:auth",
                    semantic_authority=True,
                ),
                {
                    "partial": True,
                    "claimed_consequence_ref": "consequence:partial",
                    "evidence_refs": ("evidence:partial",),
                },
                _hit(
                    "consequence:body",
                    construction="c:body",
                    source_body="def leak():\n    return hypothesis\n",
                    api_key="should_never_appear",
                ),
                _hit(
                    "consequence:excluded",
                    construction="c:ex",
                    selected_premise_ids=("premise:poisoned",),
                    inherit_plan_premises=False,
                ),
                _hit(
                    "consequence:cross-goal",
                    construction="c:cg",
                    goal_id="goal:other",
                ),
                _hit(
                    "consequence:compat-admit",
                    construction="c:ca",
                    admits_compatibility=True,
                ),
            ),
            "vector": (
                _hit(
                    "consequence:poison",
                    construction="c:poison",
                    semantic_authority=True,
                    score=float("nan"),
                ),
            ),
        },
    )

    by_c = {
        item.hypothesis.claimed_consequence_ref: item for item in receipt.nominations
    }
    assert REJECTION_STALE_OR_CROSS_ROOT in by_c["consequence:stale"].diagnostics
    assert REJECTION_FORGED in by_c["consequence:forged"].diagnostics
    assert REJECTION_SEMANTIC_AUTHORITY_CLAIM in by_c["consequence:authority"].diagnostics
    assert REJECTION_PARTIAL in by_c["consequence:partial"].diagnostics
    assert REJECTION_BODY_OR_SECRET in by_c["consequence:body"].diagnostics
    assert REJECTION_EXCLUDED_PREMISE in by_c["consequence:excluded"].diagnostics
    assert REJECTION_CROSS_GAP_OR_GOAL in by_c["consequence:cross-goal"].diagnostics
    assert (
        REJECTION_COMPATIBILITY_AS_ADMISSION
        in by_c["consequence:compat-admit"].diagnostics
    )
    assert REJECTION_POISONED in by_c["consequence:poison"].diagnostics
    assert all(
        item.disposition is HypothesisNominationDisposition.REJECTED
        for item in receipt.nominations
    )
    assert all(item.hypothesis.semantic_authority is False for item in receipt.nominations)
    serialized = receipt.to_record()
    blob = str(serialized)
    assert "should_never_appear" not in blob
    assert "def leak" not in blob


def test_bounds_refuse_over_budget_per_signal_and_union(
    roots: ProgramLogicAuthorityRoots,
) -> None:
    gap = _gap(roots)
    plan = _plan(roots, subgoals=())
    retriever = ProgramLogicHypothesisRetriever(
        roots,
        bounds=HypothesisRetrievalBounds(max_candidates=2, max_candidates_per_signal=1),
    )
    with pytest.raises(HypothesisRetrievalBoundsError):
        retriever.retrieve(
            gap,
            plan,
            candidates_by_signal={
                "analytical_construction": (
                    _hit("consequence:a", construction="c:a"),
                    _hit("consequence:b", construction="c:b"),
                ),
            },
        )
    with pytest.raises(HypothesisRetrievalBoundsError):
        ProgramLogicHypothesisRetriever(
            roots,
            bounds=HypothesisRetrievalBounds(
                max_candidates=1, max_candidates_per_signal=8
            ),
        ).retrieve(
            gap,
            plan,
            candidates_by_signal={
                "analytical_construction": (
                    _hit("consequence:a", construction="c:a"),
                ),
                "graph": (_hit("consequence:b", construction="c:b"),),
            },
        )


def test_cross_root_gap_plan_and_forged_bindings_fail_closed(
    roots: ProgramLogicAuthorityRoots,
) -> None:
    other = ProgramLogicAuthorityRoots(
        repository_id="repository:other",
        objective_id="objective:other",
        trace_id="trace:other",
        change_id="change:other",
        consumer_id="consumer:other",
        forest_id="forest:other",
        tree_id="tree:other",
        overlay_id="overlay:other",
        graph_id="graph:other",
        index_id="index:other",
        corpus_id="corpus:other",
        model_id="model:other",
        translator_id="translator:other",
        toolchain_id="toolchain:other",
        policy_id="policy:other",
        environment_id="environment:other",
    )
    gap = _gap(roots)
    plan = _plan(roots)
    foreign_gap = _gap(other, gap_id="gap:foreign")
    foreign_plan = _plan(other, plan_id="plan:foreign")
    retriever = ProgramLogicHypothesisRetriever(roots)
    with pytest.raises(HypothesisRetrievalBindingError):
        retriever.retrieve(foreign_gap, plan)
    with pytest.raises(HypothesisRetrievalBindingError):
        retriever.retrieve(gap, foreign_plan)
    with pytest.raises(HypothesisRetrievalBindingError):
        retriever.retrieve(gap, plan, graph_id="graph:forged")
    with pytest.raises(HypothesisRetrievalBindingError):
        HypothesisQuery.from_gap_and_plan(foreign_gap, plan)


def test_empty_signal_set_emits_explicit_no_candidate(
    roots: ProgramLogicAuthorityRoots,
) -> None:
    gap = _gap(roots)
    plan = _plan(roots, subgoals=())
    receipt = retrieve_program_logic_hypotheses(
        roots,
        gap,
        plan,
        candidates_by_signal={},
    )
    assert len(receipt.nominations) == 1
    nomination = receipt.nominations[0]
    assert nomination.disposition is HypothesisNominationDisposition.NO_CANDIDATE
    assert nomination.diagnostics == (REJECTION_PARTIAL,)
    assert nomination.hypothesis.semantic_authority is False
    assert receipt.no_candidate is True
    assert receipt.ambiguous is False
    assert receipt.admitted_hypothesis_id == ""
    assert receipt.write_paths == ()


def test_ambiguity_is_explicit_for_conflicting_constructions(
    roots: ProgramLogicAuthorityRoots,
) -> None:
    gap = _gap(roots)
    plan = _plan(roots, subgoals=())
    receipt = ProgramLogicHypothesisRetriever(roots).retrieve(
        gap,
        plan,
        candidates_by_signal={
            "analytical_construction": (
                _hit(
                    "consequence:shared",
                    construction="construction:path-a",
                    value_ref="value:a",
                ),
            ),
            "dataflow": (
                _hit(
                    "consequence:shared",
                    construction="construction:path-b",
                    value_ref="value:b",
                ),
            ),
        },
    )
    assert len(receipt.nominations) == 2
    assert receipt.ambiguous is True
    assert receipt.no_candidate is False
    assert all(
        item.disposition is HypothesisNominationDisposition.AMBIGUOUS
        for item in receipt.nominations
    )
    assert all(
        item.hypothesis.disposition is HypothesisDisposition.AMBIGUOUS
        for item in receipt.nominations
    )


def test_counterexample_targets_and_tactician_subgoals(
    roots: ProgramLogicAuthorityRoots,
) -> None:
    gap = _gap(roots)
    plan = _plan(roots)
    receipt = ProgramLogicHypothesisRetriever(roots).retrieve(
        gap,
        plan,
        counterexample_target_ref="counterexample:from-query",
        candidates_by_signal={
            "dataflow": (
                _hit(
                    "consequence:df",
                    construction="c:df",
                    counterexample_target_ref="counterexample:from-hit",
                ),
            ),
        },
    )
    # Manual dataflow hit + auto-projected tactician subgoal.
    assert len(receipt.nominations) >= 2
    signals = {
        signal
        for nomination in receipt.nominations
        for signal, _ in nomination.signal_evidence
    }
    assert HypothesisSignal.TACTICIAN_SUBGOAL.value in signals
    assert HypothesisSignal.DATAFLOW.value in signals
    by_c = {
        item.hypothesis.claimed_consequence_ref: item for item in receipt.nominations
    }
    assert (
        by_c["consequence:df"].hypothesis.counterexample_target_ref
        == "counterexample:from-hit"
    )
    # Auto subgoal inherits query counterexample when not set on the hit.
    sub = by_c.get("claim:context-available")
    assert sub is not None
    assert sub.hypothesis.counterexample_target_ref == "counterexample:from-query"


def test_hard_gate_facts_and_nomination_round_trip(
    roots: ProgramLogicAuthorityRoots,
) -> None:
    facts = HypothesisHardGateFacts(
        type_compatible=True,
        effect_compatible=False,
        information_content_ref="info:request-context",
        same_name=True,
        same_type=False,
        similarity_millipercent=42_000,
        notes=("observed-only",),
    )
    assert facts.information_sufficiency is False
    assert HypothesisHardGateFacts.from_dict(facts.to_record()) == facts

    with pytest.raises(HypothesisRetrievalBindingError):
        HypothesisHardGateFacts(information_sufficiency=True)

    gap = _gap(roots)
    plan = _plan(roots, subgoals=())
    receipt = ProgramLogicHypothesisRetriever(roots).retrieve(
        gap,
        plan,
        candidates_by_signal={
            "existing_value": (_hit(),),
        },
    )
    nomination = receipt.nominations[0]
    assert LogicHypothesisNomination.from_dict(nomination.to_record()) == nomination
    assert HypothesisCandidateSet.from_dict(receipt.to_record()) == receipt


def test_query_binds_gap_plan_and_rejects_authority(
    roots: ProgramLogicAuthorityRoots,
) -> None:
    gap = _gap(roots)
    plan = _plan(roots)
    query = HypothesisQuery.from_gap_and_plan(
        gap,
        plan,
        corpus_id=roots.corpus_id,
        counterexample_target_ref="counterexample:x",
    )
    assert query.gap_id == gap.gap_id
    assert query.plan_id == plan.plan_id
    assert query.selected_premise_ids == plan.selected_premise_ids
    assert query.semantic_authority is False
    with pytest.raises(HypothesisRetrievalBindingError):
        HypothesisQuery(
            roots=roots,
            gap_id=gap.gap_id,
            goal_id=gap.goal_id,
            plan_id=plan.plan_id,
            semantic_authority=True,
        )


def test_stateless_entry_point_matches_retriever(
    roots: ProgramLogicAuthorityRoots,
) -> None:
    gap = _gap(roots)
    plan = _plan(roots, subgoals=())
    signals = {
        "constructor_adapter": (
            _hit("consequence:ctor", construction="c:ctor", factory_ref="factory:make"),
        ),
        "schema": (_hit("consequence:schema", construction="c:schema"),),
    }
    via_class = ProgramLogicHypothesisRetriever(roots).retrieve(
        gap, plan, candidates_by_signal=signals
    )
    via_fn = retrieve_program_logic_hypotheses(
        roots, gap, plan, candidates_by_signal=signals
    )
    assert via_class.content_id == via_fn.content_id
    assert via_class.semantic_authority is False
    assert via_fn.admitted_hypothesis_id == ""


def test_signal_roots_cover_all_families_for_replay(
    roots: ProgramLogicAuthorityRoots,
) -> None:
    gap = _gap(roots)
    plan = _plan(roots, subgoals=())
    receipt = ProgramLogicHypothesisRetriever(roots).retrieve(
        gap,
        plan,
        candidates_by_signal={
            "graph": (
                _hit(
                    "consequence:g",
                    construction="c:g",
                    evidence_refs=("graph:edge:1", "graph:node:1"),
                ),
            ),
            "lexical": (
                _hit(
                    "consequence:g",
                    construction="c:g",
                    evidence_refs=("lexical:doc:1",),
                ),
            ),
        },
    )
    assert len(receipt.nominations) == 1
    nomination = receipt.nominations[0]
    evidence_signals = dict(nomination.signal_evidence)
    assert HypothesisSignal.GRAPH_NEIGHBORHOOD.value in evidence_signals
    assert HypothesisSignal.LEXICAL.value in evidence_signals
    assert len(receipt.signal_roots) == len(HypothesisSignal)
    assert receipt.candidate_set_id == candidate_set_identity(receipt.nominations)
