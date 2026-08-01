"""Tests for the program-repair Code Tactician adapter (LPR-008)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.program_logic_prediction_contracts import (
    GoalDisposition,
    GoalFamily,
    ProgramLogicAuthorityRoots,
    ProgramLogicGoal,
    SourceRouteKind,
    TacticianSearchPlan,
)
from ipfs_accelerate_py.agent_supervisor.analysis.program_logic_premise_corpus import (
    PremiseFeatureSet,
    PremiseLicensePolicy,
    PremiseSourceClass,
    PremiseSpanDigest,
    ProgramLogicPremise,
    ProgramLogicPremiseCorpus,
)
from ipfs_accelerate_py.agent_supervisor.integrations.ipfs_datasets_tactician_provider import (
    CODE_SOURCE_PRECEDENCE,
    CODE_TACTICIAN_PLANNER_ID,
    GENERIC_TACTICIAN_INTERFACE,
    CodeSourceType,
    CodeTacticianError,
    CodeTacticianExclusion,
    CodeTacticianPolicy,
    CodeTacticianQueryResult,
    CodeTacticianQuerySpec,
    CodeTacticianReasonCode,
    CodeTacticianRequest,
    CodeTacticianStatus,
    IpfsDatasetsTacticianProvider,
    code_source_rank,
    inspect_code_tactician_capability,
    is_approximate_or_model,
    is_local_authoritative,
    map_code_source_to_route,
    map_premise_source,
    parse_code_source_type,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def roots() -> ProgramLogicAuthorityRoots:
    return ProgramLogicAuthorityRoots(
        repository_id="repository:one",
        objective_id="objective:one",
        trace_id="trace:one",
        change_id="change:one",
        consumer_id="consumer:one",
        forest_id="forest:one",
        tree_id="tree:one",
        overlay_id="overlay:one",
        graph_id="graph:one",
        index_id="index:one",
        corpus_id="corpus:one",
        model_id="model:one",
        translator_id="translator:one",
        toolchain_id="toolchain:one",
        policy_id="policy:one",
        environment_id="environment:one",
    )


def _features() -> PremiseFeatureSet:
    return PremiseFeatureSet(
        symbol_feature_refs=("symbol:process",),
        type_feature_refs=("type:Context",),
        effect_feature_refs=("effect:io",),
        import_feature_refs=("import:service.context",),
    )


def _span() -> PremiseSpanDigest:
    return PremiseSpanDigest(
        path="src/service.py",
        start_offset=10,
        end_offset=40,
        content_digest="sha256:" + "ab" * 32,
    )


def _license() -> PremiseLicensePolicy:
    return PremiseLicensePolicy(
        license_id="license:spdx:Apache-2.0",
        redaction_policy="span_only",
        export_policy="exportable",
    )


def _premise(
    roots: ProgramLogicAuthorityRoots,
    premise_id: str,
    source_class: PremiseSourceClass,
    *,
    digest_byte: str = "11",
) -> ProgramLogicPremise:
    return ProgramLogicPremise(
        roots=roots,
        premise_id=premise_id,
        source_class=source_class,
        statement_ref=f"stmt:{premise_id}",
        statement_digest="sha256:" + (digest_byte * 32),
        lowering_ref=f"lower:{premise_id}",
        source_precedence=code_source_rank(map_premise_source(source_class)),
        features=_features(),
        span=_span(),
        license_policy=_license(),
        graph_identity=roots.graph_id,
        tree_identity=roots.tree_id,
        invalidator_refs=("invalidate:tree-drift",),
    )


def _goal(
    roots: ProgramLogicAuthorityRoots,
    goal_id: str = "goal:accept-input",
) -> ProgramLogicGoal:
    return ProgramLogicGoal(
        roots=roots,
        goal_id=goal_id,
        family=GoalFamily.BEHAVIOR,
        disposition=GoalDisposition.OPEN,
        positive_statement_ref="stmt:accept-input",
        affected_symbol_ids=("symbol:process",),
        source_refs=("src/service.py::process",),
        invalidation_refs=("inv:tree:one",),
        bound_refs=("gap:type-facts", "gap:contract"),
    )


def _corpus(
    roots: ProgramLogicAuthorityRoots,
    premises: Sequence[ProgramLogicPremise] | None = None,
) -> ProgramLogicPremiseCorpus:
    if premises is None:
        premises = (
            _premise(
                roots,
                "p:contract",
                PremiseSourceClass.REVIEWED_CONTRACT,
                digest_byte="11",
            ),
            _premise(
                roots,
                "p:types",
                PremiseSourceClass.TYPE_AND_EFFECT_FACTS,
                digest_byte="12",
            ),
            _premise(
                roots,
                "p:graph",
                PremiseSourceClass.PROGRAM_GRAPH,
                digest_byte="13",
            ),
            _premise(
                roots,
                "p:vector",
                PremiseSourceClass.VECTOR_ANALOGUE,
                digest_byte="14",
            ),
            _premise(
                roots,
                "p:model",
                PremiseSourceClass.MODEL_HYPOTHESIS,
                digest_byte="15",
            ),
        )
    return ProgramLogicPremiseCorpus(
        roots=roots,
        premises=tuple(premises),
        producer_ref="test-premise-corpus@1",
    )


# ---------------------------------------------------------------------------
# Fake generic Logic Tactician
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _FakeRoute:
    route_id: str
    source_id: str
    source_class: str
    stage_index: int
    disposition: str
    rationale: str
    addresses_gaps: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class _FakeSubgoal:
    subgoal_id: str
    parent_goal_id: str
    statement_ref: str
    depends_on: list[str] = field(default_factory=list)
    addresses_gaps: list[str] = field(default_factory=list)
    rationale: str = ""


@dataclass(frozen=True)
class _FakePlan:
    plan_id: str
    goal_id: str
    selected_routes: list[_FakeRoute]
    excluded_routes: list[_FakeRoute]
    subgoals: list[_FakeSubgoal]
    stop_disposition: str = "continue"
    planner_id: str = "logic.tactician.deterministic@1"
    semantic_authority: bool = False


class _FakeLogicTactician:
    """Minimal duck-typed generic planner used by adapter tests."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def plan(
        self,
        goal: Any,
        sources: Sequence[Any],
        policy: Any = None,
        **kwargs: Any,
    ) -> _FakePlan:
        goal_id = (
            goal.get("goal_id")
            if isinstance(goal, dict)
            else getattr(goal, "goal_id", "goal:unknown")
        )
        if hasattr(goal, "get"):
            goal_id = goal.get("goal_id", goal_id)

        # Accept MappingProxyType / dict-like and objects with attributes.
        def _src_id(source: Any) -> str:
            if isinstance(source, dict) or hasattr(source, "keys"):
                try:
                    return str(source["source_id"])  # type: ignore[index]
                except Exception:
                    pass
            return str(getattr(source, "source_id", "source:unknown"))

        def _src_class(source: Any) -> str:
            if isinstance(source, dict) or hasattr(source, "keys"):
                try:
                    return str(source["source_class"])  # type: ignore[index]
                except Exception:
                    pass
            return str(getattr(source, "source_class", "unknown"))

        def _src_rationale(source: Any) -> str:
            if isinstance(source, dict) or hasattr(source, "keys"):
                try:
                    return str(source["rationale"])  # type: ignore[index]
                except Exception:
                    pass
            return str(getattr(source, "rationale", "selected"))

        ordered = list(sources)
        # Stable sort by precedence then source_id (mirrors generic planner).
        def _prec(source: Any) -> int:
            if isinstance(source, dict) or hasattr(source, "keys"):
                try:
                    return int(source["precedence"])  # type: ignore[index]
                except Exception:
                    pass
            return int(getattr(source, "precedence", 0))

        ordered.sort(key=lambda s: (_prec(s), _src_id(s)))
        max_routes = 32
        if policy is not None:
            max_routes = int(
                policy.get("max_routes", max_routes)
                if hasattr(policy, "get")
                else getattr(policy, "max_routes", max_routes)
            )

        selected: list[_FakeRoute] = []
        excluded: list[_FakeRoute] = []
        for index, source in enumerate(ordered):
            source_id = _src_id(source)
            source_class = _src_class(source)
            if len(selected) < max_routes:
                selected.append(
                    _FakeRoute(
                        route_id=f"route:selected:{source_id}",
                        source_id=source_id,
                        source_class=source_class,
                        stage_index=len(selected),
                        disposition="selected",
                        rationale=_src_rationale(source),
                    )
                )
            else:
                excluded.append(
                    _FakeRoute(
                        route_id=f"route:excluded:{source_id}",
                        source_id=source_id,
                        source_class=source_class,
                        stage_index=index,
                        disposition="excluded",
                        rationale="Excluded after selection budget",
                    )
                )

        gaps = []
        if isinstance(goal, dict) or hasattr(goal, "keys"):
            try:
                gaps = list(goal["proof_gaps"])  # type: ignore[index]
            except Exception:
                gaps = list(getattr(goal, "proof_gaps", ()) or ())
        else:
            gaps = list(getattr(goal, "proof_gaps", ()) or ())

        subgoals: list[_FakeSubgoal] = []
        for index, gap in enumerate(gaps[:16]):
            depends = [f"subgoal:{gaps[index - 1]}"] if index > 0 else []
            subgoals.append(
                _FakeSubgoal(
                    subgoal_id=f"subgoal:{gap}",
                    parent_goal_id=str(goal_id),
                    statement_ref=f"stmt#{gap}",
                    depends_on=depends,
                    addresses_gaps=[str(gap)],
                    rationale=f"Cover {gap}",
                )
            )

        plan = _FakePlan(
            plan_id=f"generic-plan:{goal_id}",
            goal_id=str(goal_id),
            selected_routes=selected,
            excluded_routes=excluded,
            subgoals=subgoals,
        )
        self.calls.append(
            {
                "goal_id": goal_id,
                "source_ids": [_src_id(s) for s in ordered],
                "selected": [r.source_id for r in selected],
                "excluded": [r.source_id for r in excluded],
            }
        )
        return plan


def _provider_with_fake(
    *,
    query_adapters: dict[str, Any] | None = None,
    policy: CodeTacticianPolicy | None = None,
) -> tuple[IpfsDatasetsTacticianProvider, _FakeLogicTactician]:
    fake = _FakeLogicTactician()

    def importer(name: str) -> Any:
        raise AssertionError(f"optional import should not run when planner injected: {name}")

    provider = IpfsDatasetsTacticianProvider(
        policy=policy,
        importer=importer,
        planner_factory=lambda: fake,
        query_adapters=query_adapters or {},
    )
    return provider, fake


def _request(
    roots: ProgramLogicAuthorityRoots,
    **overrides: Any,
) -> CodeTacticianRequest:
    values: dict[str, Any] = {
        "roots": roots,
        "goals": (_goal(roots),),
        "corpus": _corpus(roots),
        "policy": CodeTacticianPolicy(allow_model_hypothesis=False),
        "information_demands": (
            CodeSourceType.AUTHORITATIVE_CONTRACT,
            CodeSourceType.TYPE_AND_EFFECT_FACTS,
            CodeSourceType.PROGRAM_GRAPH,
            CodeSourceType.VECTOR_ANALOGUE,
        ),
        "admitted_tree_id": roots.tree_id,
        "admitted_corpus_id": roots.corpus_id,
        "expected_roots": roots,
        "logic_family_refs": ("logic:fol",),
        "translation_refs": ("translation:smtlib2",),
    }
    values.update(overrides)
    return CodeTacticianRequest(**values)


# ---------------------------------------------------------------------------
# Closed taxonomy / policy
# ---------------------------------------------------------------------------


def test_source_precedence_is_local_first() -> None:
    assert CODE_SOURCE_PRECEDENCE[0] is CodeSourceType.AUTHORITATIVE_CONTRACT
    assert CODE_SOURCE_PRECEDENCE[-1] is CodeSourceType.MODEL_HYPOTHESIS
    local_ranks = [
        code_source_rank(item)
        for item in CODE_SOURCE_PRECEDENCE
        if is_local_authoritative(item)
    ]
    approx_ranks = [
        code_source_rank(item)
        for item in CODE_SOURCE_PRECEDENCE
        if is_approximate_or_model(item)
    ]
    assert local_ranks
    assert approx_ranks
    assert max(local_ranks) < min(approx_ranks)


def test_policy_rejects_approximate_before_local() -> None:
    bad_order = (
        CodeSourceType.VECTOR_ANALOGUE,
        CodeSourceType.AUTHORITATIVE_CONTRACT,
    )
    with pytest.raises(CodeTacticianError, match="local authoritative"):
        CodeTacticianPolicy(source_class_order=bad_order)


def test_policy_forbids_network_write_proof_and_semantic_authority() -> None:
    with pytest.raises(CodeTacticianError):
        CodeTacticianPolicy(network_allowed=True)
    with pytest.raises(CodeTacticianError):
        CodeTacticianPolicy(write_allowed=True)
    with pytest.raises(CodeTacticianError):
        CodeTacticianPolicy(proof_execution_allowed=True)
    with pytest.raises(CodeTacticianError):
        CodeTacticianPolicy(semantic_authority=True)


def test_policy_rejects_unbounded_budgets() -> None:
    with pytest.raises(CodeTacticianError, match="hard maximum"):
        CodeTacticianPolicy(max_routes=10_000)


def test_parse_and_map_helpers() -> None:
    assert parse_code_source_type("program_graph") is CodeSourceType.PROGRAM_GRAPH
    with pytest.raises(CodeTacticianError, match="unsupported source type"):
        parse_code_source_type("free_form_web_search")
    assert map_premise_source(PremiseSourceClass.REVIEWED_CONTRACT) is (
        CodeSourceType.AUTHORITATIVE_CONTRACT
    )
    assert map_code_source_to_route(CodeSourceType.VECTOR_ANALOGUE) is (
        SourceRouteKind.VECTOR
    )


# ---------------------------------------------------------------------------
# Lazy capability / unavailable abstention
# ---------------------------------------------------------------------------


def test_construction_and_capability_do_not_import() -> None:
    calls: list[str] = []

    def importer(name: str) -> Any:
        calls.append(name)
        raise AssertionError("capability must not import optional code")

    provider = IpfsDatasetsTacticianProvider(importer=importer)
    cap = provider.capabilities()
    pure = inspect_code_tactician_capability()

    assert calls == []
    assert cap.imported is False
    assert cap.available is False
    assert cap.health == "lazy"
    assert cap.semantic_authority is False
    assert cap.completion_authority is False
    assert pure.to_dict()["schema"].endswith("tactician-capability@1")
    assert pure.interface_version == GENERIC_TACTICIAN_INTERFACE


def test_unavailable_provider_returns_typed_abstention(roots: ProgramLogicAuthorityRoots) -> None:
    def importer(name: str) -> Any:
        raise ModuleNotFoundError(name)

    provider = IpfsDatasetsTacticianProvider(importer=importer)
    response = provider.plan(_request(roots))

    assert response.status is CodeTacticianStatus.UNAVAILABLE
    assert response.reason_code is CodeTacticianReasonCode.OPTIONAL_MODULE_UNAVAILABLE
    assert response.plan is None
    assert response.semantic_authority is False
    assert response.interface_version == GENERIC_TACTICIAN_INTERFACE
    assert "unavailable" in response.message.lower() or response.message


# ---------------------------------------------------------------------------
# Happy-path planning
# ---------------------------------------------------------------------------


def test_plan_is_deterministic_and_local_first(roots: ProgramLogicAuthorityRoots) -> None:
    provider, fake = _provider_with_fake()
    request = _request(roots)

    first = provider.plan(request)
    second = provider.plan(request)

    assert first.status is CodeTacticianStatus.PLANNED
    assert first.reason_code is CodeTacticianReasonCode.OK
    assert first.plan is not None
    assert first.semantic_authority is False
    assert first.plan.semantic_authority is False
    assert first.to_dict() == second.to_dict()
    assert first.plan.plan_id == second.plan.plan_id
    assert first.plan.planner_id
    assert first.generic_plan_ref

    routes = list(first.plan.ordered_source_routes)
    assert routes
    # First routes must be local/authoritative kinds, not vector/llm.
    assert routes[0] in {
        SourceRouteKind.REVIEWED_CONTRACT,
        SourceRouteKind.LOCAL_STATIC,
        SourceRouteKind.DATAFLOW,
        SourceRouteKind.GRAPH,
        SourceRouteKind.REVIEWED_TEST,
        SourceRouteKind.NORMATIVE_SPEC,
    }
    if SourceRouteKind.VECTOR in routes:
        assert routes.index(SourceRouteKind.REVIEWED_CONTRACT) < routes.index(
            SourceRouteKind.VECTOR
        ) or SourceRouteKind.LOCAL_STATIC in routes

    # Model hypothesis is denied by default policy even if present in corpus.
    assert CodeSourceType.MODEL_HYPOTHESIS not in first.selected_source_types
    assert CodeSourceType.AUTHORITATIVE_CONTRACT in first.selected_source_types
    assert fake.calls  # planner was invoked


def test_plan_records_query_result_and_exclusion_ids(
    roots: ProgramLogicAuthorityRoots,
) -> None:
    def vector_adapter(query: CodeTacticianQuerySpec) -> CodeTacticianQueryResult:
        return CodeTacticianQueryResult(
            query_id=query.query_id,
            result_id="query-result:vector:1",
            source_type=query.source_type,
            adapter_ref=query.adapter_ref,
            status="completed",
            hit_refs=("hit:analogue:1", "hit:analogue:2"),
        )

    provider, _ = _provider_with_fake(
        query_adapters={"adapter:vector@1": vector_adapter}
    )
    query = CodeTacticianQuerySpec(
        query_id="query:vector:goal",
        source_type=CodeSourceType.VECTOR_ANALOGUE,
        adapter_ref="adapter:vector@1",
        target_ref="index:one",
        root_bindings={
            "tree_id": roots.tree_id,
            "corpus_id": roots.corpus_id,
            "index_id": roots.index_id,
        },
        parameters={"top_k": 3, "signal": "analogue"},
    )
    request = _request(roots, query_specs=(query,))
    response = provider.plan(request)

    assert response.status is CodeTacticianStatus.PLANNED
    assert response.plan is not None
    assert "query:vector:goal" in response.plan.query_refs
    assert "query-result:vector:1" in response.plan.query_refs
    assert len(response.query_results) == 1
    assert response.query_results[0].result_id == "query-result:vector:1"
    assert response.query_results[0].semantic_authority is False
    # Missing adapter for a second query records exclusion.
    provider2, _ = _provider_with_fake()
    response2 = provider2.plan(_request(roots, query_specs=(query,)))
    assert response2.query_results[0].status == "adapter_missing"
    assert any(
        item.subject_ref == "query:vector:goal" for item in response2.exclusions
    )


def test_queries_only_through_bounded_referenced_adapters(
    roots: ProgramLogicAuthorityRoots,
) -> None:
    with pytest.raises(CodeTacticianError, match="not queryable"):
        CodeTacticianQuerySpec(
            query_id="query:bad",
            source_type=CodeSourceType.AUTHORITATIVE_CONTRACT,
            adapter_ref="adapter:x",
            target_ref="ref:x",
        )
    with pytest.raises(CodeTacticianError, match="body"):
        CodeTacticianQuerySpec(
            query_id="query:body",
            source_type=CodeSourceType.VECTOR_ANALOGUE,
            adapter_ref="adapter:vector@1",
            target_ref="index:one",
            parameters={"source_body": "def evil(): pass"},
        )


# ---------------------------------------------------------------------------
# Rejection gates
# ---------------------------------------------------------------------------


def test_reject_cross_root_goal_and_corpus(roots: ProgramLogicAuthorityRoots) -> None:
    other = ProgramLogicAuthorityRoots(
        repository_id="repository:two",
        objective_id="objective:two",
        trace_id="trace:two",
        change_id="change:two",
        consumer_id="consumer:two",
        forest_id="forest:two",
        tree_id="tree:two",
        overlay_id="overlay:two",
        graph_id="graph:two",
        index_id="index:two",
        corpus_id="corpus:two",
        model_id="model:two",
        translator_id="translator:two",
        toolchain_id="toolchain:two",
        policy_id="policy:two",
        environment_id="environment:two",
    )
    provider, _ = _provider_with_fake()
    with pytest.raises(CodeTacticianError, match="roots"):
        CodeTacticianRequest(
            roots=roots,
            goals=(_goal(other),),
            corpus=_corpus(roots),
        )


def test_reject_stale_admitted_tree(roots: ProgramLogicAuthorityRoots) -> None:
    provider, _ = _provider_with_fake()
    response = provider.plan(
        _request(roots, admitted_tree_id="tree:stale-other")
    )
    assert response.status is CodeTacticianStatus.REJECTED
    assert response.reason_code is CodeTacticianReasonCode.STALE_ROOTS
    assert response.semantic_authority is False
    assert response.plan is None


def test_reject_cross_root_expected_roots(roots: ProgramLogicAuthorityRoots) -> None:
    other = ProgramLogicAuthorityRoots(
        repository_id="repository:two",
        objective_id="objective:two",
        trace_id="trace:two",
        change_id="change:two",
        consumer_id="consumer:two",
        forest_id="forest:two",
        tree_id="tree:two",
        overlay_id="overlay:two",
        graph_id="graph:two",
        index_id="index:two",
        corpus_id="corpus:two",
        model_id="model:two",
        translator_id="translator:two",
        toolchain_id="toolchain:two",
        policy_id="policy:two",
        environment_id="environment:two",
    )
    provider, _ = _provider_with_fake()
    response = provider.plan(_request(roots, expected_roots=other))
    assert response.status is CodeTacticianStatus.REJECTED
    assert response.reason_code is CodeTacticianReasonCode.CROSS_ROOT


def test_reject_free_form_authority_in_metadata(
    roots: ProgramLogicAuthorityRoots,
) -> None:
    provider, _ = _provider_with_fake()
    with pytest.raises(CodeTacticianError, match="authority|body"):
        CodeTacticianRequest(
            roots=roots,
            goals=(_goal(roots),),
            corpus=_corpus(roots),
            metadata={"semantic_authority": True},
        )


def test_reject_unsupported_information_demand(
    roots: ProgramLogicAuthorityRoots,
) -> None:
    provider, _ = _provider_with_fake()
    with pytest.raises(CodeTacticianError, match="unsupported source type"):
        _request(roots, information_demands=("not_a_real_source",))


def test_malformed_mapping_request_returns_typed_malformed(
    roots: ProgramLogicAuthorityRoots,
) -> None:
    provider, _ = _provider_with_fake()
    response = provider.plan({"roots": roots, "goals": [], "corpus": _corpus(roots)})
    assert response.status is CodeTacticianStatus.MALFORMED
    assert response.reason_code is CodeTacticianReasonCode.MALFORMED_REQUEST
    assert response.semantic_authority is False


def test_model_hypothesis_requires_explicit_admission(
    roots: ProgramLogicAuthorityRoots,
) -> None:
    provider, _ = _provider_with_fake(
        policy=CodeTacticianPolicy(allow_model_hypothesis=False)
    )
    response = provider.plan(
        _request(
            roots,
            information_demands=(
                CodeSourceType.AUTHORITATIVE_CONTRACT,
                CodeSourceType.MODEL_HYPOTHESIS,
            ),
        )
    )
    assert response.status is CodeTacticianStatus.PLANNED
    assert CodeSourceType.MODEL_HYPOTHESIS not in response.selected_source_types
    assert any(
        "model_hypothesis" in item.source_type or "model_hypothesis" in item.subject_ref
        for item in response.exclusions
    ) or CodeSourceType.MODEL_HYPOTHESIS in response.excluded_source_types


def test_allow_model_hypothesis_when_policy_admits(
    roots: ProgramLogicAuthorityRoots,
) -> None:
    provider, _ = _provider_with_fake(
        policy=CodeTacticianPolicy(allow_model_hypothesis=True)
    )
    response = provider.plan(
        _request(
            roots,
            policy=CodeTacticianPolicy(allow_model_hypothesis=True),
            information_demands=(
                CodeSourceType.AUTHORITATIVE_CONTRACT,
                CodeSourceType.MODEL_HYPOTHESIS,
            ),
        )
    )
    assert response.status is CodeTacticianStatus.PLANNED
    assert response.plan is not None
    assert CodeSourceType.MODEL_HYPOTHESIS in response.selected_source_types
    # Still never semantic authority.
    assert response.plan.semantic_authority is False
    assert SourceRouteKind.LLM in response.plan.ordered_source_routes
    llm_index = response.plan.ordered_source_routes.index(SourceRouteKind.LLM)
    # At least one local route precedes the model route.
    assert any(
        response.plan.ordered_source_routes.index(route) < llm_index
        for route in response.plan.ordered_source_routes
        if route
        in {
            SourceRouteKind.REVIEWED_CONTRACT,
            SourceRouteKind.LOCAL_STATIC,
            SourceRouteKind.GRAPH,
        }
    )


# ---------------------------------------------------------------------------
# Plan envelope / contracts
# ---------------------------------------------------------------------------


def test_search_plan_binds_goals_corpus_roots_and_policies(
    roots: ProgramLogicAuthorityRoots,
) -> None:
    provider, _ = _provider_with_fake()
    response = provider.plan(_request(roots))
    plan = response.plan
    assert plan is not None
    assert isinstance(plan, TacticianSearchPlan)
    assert plan.roots.content_id == roots.content_id
    assert "goal:accept-input" in plan.goal_ids
    assert plan.stop_policy_ref
    assert plan.escalation_policy_ref
    assert plan.abstention_policy_ref
    assert plan.resource_policy_ref
    assert plan.invalidation_refs
    assert "p:contract" in plan.selected_premise_ids
    assert plan.planned_logic_family_refs == ("logic:fol",)
    assert plan.translation_refs == ("translation:smtlib2",)
    assert plan.subgoals
    assert all(sg.proof_status.value == "unproved" for sg in plan.subgoals)
    # Round-trip the plan contract.
    assert TacticianSearchPlan.from_dict(plan.to_dict()).plan_id == plan.plan_id


def test_response_to_dict_is_body_free_and_deterministic(
    roots: ProgramLogicAuthorityRoots,
) -> None:
    provider, _ = _provider_with_fake()
    response = provider.plan(_request(roots))
    payload = response.to_dict()
    assert payload["semantic_authority"] is False
    assert payload["status"] == "planned"
    assert "source_body" not in str(payload)
    assert "embedding" not in str(payload)
    again = provider.plan(_request(roots)).to_dict()
    assert payload == again


def test_selected_and_excluded_premises_are_disjoint(
    roots: ProgramLogicAuthorityRoots,
) -> None:
    provider, _ = _provider_with_fake(
        policy=CodeTacticianPolicy(max_routes=2, max_sources=2)
    )
    response = provider.plan(
        _request(
            roots,
            policy=CodeTacticianPolicy(max_routes=2, max_sources=2),
            information_demands=tuple(CODE_SOURCE_PRECEDENCE),
        )
    )
    assert response.status is CodeTacticianStatus.PLANNED
    plan = response.plan
    assert plan is not None
    selected = set(plan.selected_premise_ids)
    excluded = set(plan.excluded_premise_ids)
    assert selected.isdisjoint(excluded)


def test_interface_and_planner_identity_constants() -> None:
    assert GENERIC_TACTICIAN_INTERFACE == "ipfs_datasets_py.logic.tactician@1"
    assert CODE_TACTICIAN_PLANNER_ID.endswith("code-tactician@1")


def test_exclusion_record_shape() -> None:
    item = CodeTacticianExclusion(
        exclusion_id="exclusion:1",
        subject_ref="source:vector_analogue",
        source_type="vector_analogue",
        rationale="budget",
        stage="policy",
    )
    assert item.to_dict()["stage"] == "policy"


def test_multi_goal_plan_lists_all_goal_ids(roots: ProgramLogicAuthorityRoots) -> None:
    provider, _ = _provider_with_fake()
    goals = (
        _goal(roots, "goal:one"),
        _goal(roots, "goal:two"),
    )
    response = provider.plan(_request(roots, goals=goals))
    assert response.status is CodeTacticianStatus.PLANNED
    assert response.plan is not None
    assert set(response.plan.goal_ids) == {"goal:one", "goal:two"}


def test_query_root_binding_mismatch_rejects(roots: ProgramLogicAuthorityRoots) -> None:
    provider, _ = _provider_with_fake()
    query = CodeTacticianQuerySpec(
        query_id="query:stale",
        source_type=CodeSourceType.VECTOR_ANALOGUE,
        adapter_ref="adapter:vector@1",
        target_ref="index:one",
        root_bindings={"tree_id": "tree:other"},
    )
    response = provider.plan(_request(roots, query_specs=(query,)))
    assert response.status is CodeTacticianStatus.REJECTED
    assert response.reason_code is CodeTacticianReasonCode.CROSS_ROOT


def test_build_request_accepts_mapping(roots: ProgramLogicAuthorityRoots) -> None:
    provider, _ = _provider_with_fake()
    request = provider.build_request(
        {
            "roots": roots.to_dict(),
            "goals": [_goal(roots).to_dict()],
            "corpus": _corpus(roots).to_dict(),
            "admitted_tree_id": roots.tree_id,
            "admitted_corpus_id": roots.corpus_id,
        }
    )
    assert isinstance(request, CodeTacticianRequest)
    assert request.roots.content_id == roots.content_id
    response = provider.plan(request)
    assert response.status is CodeTacticianStatus.PLANNED
