"""Tests for dynamic/reflection/registry/generated/FFI impact frontiers (RPR-031)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.change_propagation_contracts import (
    GraphNodeRef,
    GraphProvenance,
    ImpactClosureReceipt,
    ImpactCompleteness,
    ImpactConsumer,
    PropagationAuthorityRoots,
)
from ipfs_accelerate_py.agent_supervisor.analysis.dynamic_impact_frontier import (
    ClosureAttempt,
    ClosureMechanism,
    DynamicImpactFrontier,
    DynamicImpactFrontierAnalyzer,
    DynamicImpactFrontierAuthorityError,
    DynamicImpactFrontierError,
    FrontierDisposition,
    FrontierKind,
    FrontierObservation,
    ImpactFrontierEntry,
    all_frontier_kinds,
    required_kind_coverage,
)


@pytest.fixture
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


def _observation(
    kind: str | FrontierKind,
    route: str,
    *,
    contract: str = "contract:dispatch",
    evidence: Sequence[str] = ("evidence:site",),
    claim_kind: str = "",
    timed_out: bool = False,
    absent_evidence: bool = False,
    required: bool = True,
    graph_node_id: str = "",
    graph_edge_id: str = "",
    reason: str = "",
) -> FrontierObservation:
    return FrontierObservation(
        kind=kind,
        route=route,
        affected_contract_ref=contract,
        reason=reason,
        evidence_refs=tuple(evidence),
        required=required,
        claim_kind=claim_kind,
        graph_node_id=graph_node_id,
        graph_edge_id=graph_edge_id,
        timed_out=timed_out,
        absent_evidence=absent_evidence,
    )


@dataclass(frozen=True)
class _Graph:
    frontier_refs: tuple[str, ...] = ()
    exclusion_refs: tuple[str, ...] = ()
    complete: bool = False


@dataclass(frozen=True)
class _Capability:
    status: str


@dataclass(frozen=True)
class _CapabilityReport:
    capabilities: tuple[_Capability, ...] = ()


class _Resolver:
    def __init__(self, observations: Sequence[FrontierObservation]) -> None:
        self._observations = tuple(observations)

    def frontier_observations(
        self, roots: PropagationAuthorityRoots, delta_id: str
    ) -> Sequence[FrontierObservation]:
        assert roots.graph_id
        assert delta_id
        return self._observations


# ---------------------------------------------------------------------------
# Kind vocabulary and entry invariants
# ---------------------------------------------------------------------------


def test_all_required_frontier_kinds_are_enumerated() -> None:
    kinds = {item.value for item in all_frontier_kinds()}
    assert kinds == {
        "reflection",
        "introspection",
        "string_dispatch",
        "monkey_patch",
        "plugin_entry_point",
        "runtime_di_registry",
        "callback",
        "generated_code",
        "native_ffi",
        "remote_service",
        "excluded_root",
        "unbounded_resource",
    }


@pytest.mark.parametrize(
    ("alias", "expected"),
    [
        ("reflection", FrontierKind.REFLECTION),
        ("introspection", FrontierKind.INTROSPECTION),
        ("getattr", FrontierKind.STRING_DISPATCH),
        ("eval", FrontierKind.STRING_DISPATCH),
        ("import_string", FrontierKind.STRING_DISPATCH),
        ("monkey_patch", FrontierKind.MONKEY_PATCH),
        ("plugin_registry", FrontierKind.PLUGIN_ENTRY_POINT),
        ("entry_points", FrontierKind.PLUGIN_ENTRY_POINT),
        ("registry", FrontierKind.RUNTIME_DI_REGISTRY),
        ("di", FrontierKind.RUNTIME_DI_REGISTRY),
        ("callback", FrontierKind.CALLBACK),
        ("generated", FrontierKind.GENERATED_CODE),
        ("ffi", FrontierKind.NATIVE_FFI),
        ("native_extension", FrontierKind.NATIVE_FFI),
        ("remote_service", FrontierKind.REMOTE_SERVICE),
        ("vendored", FrontierKind.EXCLUDED_ROOT),
        ("read_only", FrontierKind.EXCLUDED_ROOT),
        ("unbounded_resource", FrontierKind.UNBOUNDED_RESOURCE),
    ],
)
def test_kind_aliases_map_to_closed_vocabulary(
    alias: str, expected: FrontierKind
) -> None:
    assert FrontierKind.coerce(alias) is expected


def test_closed_entry_requires_supported_mechanism_evidence_and_route_scope() -> None:
    entry = ImpactFrontierEntry(
        entry_id="frontier:reflection:src.reflect.invoke:0",
        kind=FrontierKind.REFLECTION,
        disposition=FrontierDisposition.CLOSED_OBSERVED_ROUTE,
        route="src/reflect.py:invoke",
        affected_contract_ref="contract:dispatch",
        reason="manifest closed observed route",
        evidence_refs=("manifest:plugin-map@1",),
        supported_closure_mechanisms=(ClosureMechanism.REVIEWED_MANIFEST,),
        closed_by=ClosureMechanism.REVIEWED_MANIFEST,
        closed_route_only=True,
    )
    assert entry.closed_route_only is True
    assert not entry.is_open_required

    with pytest.raises(DynamicImpactFrontierAuthorityError):
        ImpactFrontierEntry(
            entry_id="frontier:bad",
            kind=FrontierKind.REFLECTION,
            disposition=FrontierDisposition.CLOSED_OBSERVED_ROUTE,
            route="src/reflect.py:invoke",
            affected_contract_ref="contract:dispatch",
            reason="missing closed_by",
            evidence_refs=("manifest:x",),
            supported_closure_mechanisms=(ClosureMechanism.REVIEWED_MANIFEST,),
            closed_route_only=True,
        )

    with pytest.raises(DynamicImpactFrontierAuthorityError):
        ImpactFrontierEntry(
            entry_id="frontier:bad2",
            kind=FrontierKind.REFLECTION,
            disposition=FrontierDisposition.CLOSED_OBSERVED_ROUTE,
            route="src/reflect.py:invoke",
            affected_contract_ref="contract:dispatch",
            reason="vector claim",
            evidence_refs=("vector:hit",),
            supported_closure_mechanisms=(ClosureMechanism.REVIEWED_MANIFEST,),
            closed_by=ClosureMechanism.REVIEWED_MANIFEST,
            closed_route_only=True,
            claim_kind="vector",
        )


# ---------------------------------------------------------------------------
# Analyzer: emit all kinds, open required blocks complete
# ---------------------------------------------------------------------------


def test_analyzer_emits_bounded_entries_for_every_required_kind(
    roots: PropagationAuthorityRoots,
) -> None:
    observations = [
        _observation("reflection", "src/reflect.py:invoke"),
        _observation("introspection", "src/inspect.py:probe"),
        _observation("getattr", "src/dyn.py:getattr_dispatch"),
        _observation("monkey_patch", "src/patch.py:apply"),
        _observation("plugin_entry_point", "src/plugins.py:load"),
        _observation("runtime_di_registry", "src/di.py:resolve"),
        _observation("callback", "src/hooks.py:on_event"),
        _observation("generated_code", "generated/client.py:Client.call"),
        _observation("native_ffi", "native/bridge.c:call"),
        _observation("remote_service", "services/remote.py:rpc"),
        _observation("excluded_root", "vendor/lib/mod.py:use"),
        _observation("unbounded_resource", "src/worker.py:run"),
    ]
    frontier = DynamicImpactFrontierAnalyzer().analyze(
        roots, "delta:dynamic", observations, affected_contract_ref="contract:dispatch"
    )

    covered = required_kind_coverage(frontier.entries)
    assert covered == set(all_frontier_kinds())
    assert frontier.completeness is ImpactCompleteness.PARTIAL_WITH_FRONTIER
    assert not frontier.impact_completeness_possible
    assert frontier.open_required_entry_ids
    for entry in frontier.entries:
        assert entry.route
        assert entry.affected_contract_ref == "contract:dispatch"
        assert entry.supported_closure_mechanisms
        assert entry.reason
        assert entry.disposition is FrontierDisposition.OPEN
        assert entry.evidence_refs


def test_complete_impact_impossible_while_required_entry_open(
    roots: PropagationAuthorityRoots,
) -> None:
    frontier = DynamicImpactFrontierAnalyzer().analyze(
        roots,
        "delta:open",
        [_observation("ffi", "native/bridge.c:call")],
    )
    assert frontier.impact_completeness_possible is False
    with pytest.raises(DynamicImpactFrontierAuthorityError, match="complete impact"):
        DynamicImpactFrontier(
            roots=roots,
            delta_id="delta:open",
            entries=frontier.entries,
            completeness=ImpactCompleteness.COMPLETE,
        )


def test_fixture_style_reflection_plugin_ffi_remain_explicit(
    roots: PropagationAuthorityRoots,
) -> None:
    """Mirror the reflection-plugin-registry-ffi-frontier change-propagation fixture."""
    observations = [
        {"kind": "reflection", "site": "src/reflect.py:invoke", "evidence_refs": ["ev:r"]},
        {
            "kind": "plugin_registry",
            "site": "src/plugins.py:load",
            "evidence_refs": ["ev:p"],
        },
        {"kind": "ffi", "site": "native/bridge.c:call", "evidence_refs": ["ev:f"]},
    ]
    frontier = DynamicImpactFrontierAnalyzer().analyze(
        roots, "delta:dynamic_frontier", observations
    )
    kinds = {entry.kind for entry in frontier.entries}
    assert kinds == {
        FrontierKind.REFLECTION,
        FrontierKind.PLUGIN_ENTRY_POINT,
        FrontierKind.NATIVE_FFI,
    }
    assert frontier.completeness is ImpactCompleteness.PARTIAL_WITH_FRONTIER
    assert all(entry.disposition is FrontierDisposition.OPEN for entry in frontier.entries)


# ---------------------------------------------------------------------------
# Closure policy: manifests / witnesses close route only; vector/LLM cannot
# ---------------------------------------------------------------------------


def test_reviewed_manifest_closes_only_observed_route(
    roots: PropagationAuthorityRoots,
) -> None:
    route = "src/plugins.py:load"
    other = "src/plugins.py:other_entry"
    frontier = DynamicImpactFrontierAnalyzer().analyze(
        roots,
        "delta:manifest",
        [
            _observation("plugin", route, evidence=("ev:plugin",)),
            _observation("plugin", other, evidence=("ev:other",)),
        ],
        closure_attempts=[
            ClosureAttempt(
                entry_route=route,
                mechanism=ClosureMechanism.REVIEWED_MANIFEST,
                evidence_refs=("manifest:entry-points@1",),
                roots_graph_id=roots.graph_id,
                observed_route_only=True,
            )
        ],
    )
    by_route = {entry.route: entry for entry in frontier.entries}
    closed = by_route[route]
    open_entry = by_route[other]
    assert closed.disposition is FrontierDisposition.CLOSED_OBSERVED_ROUTE
    assert closed.closed_by is ClosureMechanism.REVIEWED_MANIFEST
    assert closed.closed_route_only is True
    assert "manifest:entry-points@1" in closed.evidence_refs
    assert open_entry.disposition is FrontierDisposition.OPEN
    assert open_entry.is_open_required
    assert frontier.impact_completeness_possible is False


def test_root_bound_runtime_witness_closes_observed_route(
    roots: PropagationAuthorityRoots,
) -> None:
    route = "src/reflect.py:invoke"
    frontier = DynamicImpactFrontierAnalyzer().analyze(
        roots,
        "delta:witness",
        [_observation("reflection", route)],
        closure_attempts=[
            {
                "entry_route": route,
                "mechanism": "root_bound_runtime_witness",
                "evidence_refs": ["runtime:trace@tree:candidate"],
                "roots_graph_id": roots.graph_id,
                "observed_route_only": True,
            }
        ],
    )
    entry = frontier.entries[0]
    assert entry.disposition is FrontierDisposition.CLOSED_OBSERVED_ROUTE
    assert entry.closed_by is ClosureMechanism.ROOT_BOUND_RUNTIME_WITNESS
    assert frontier.impact_completeness_possible is True
    assert frontier.completeness is ImpactCompleteness.COMPLETE


def test_vector_kg_llm_claims_cannot_close_frontier(
    roots: PropagationAuthorityRoots,
) -> None:
    route = "src/dyn.py:dispatch"
    for claim in ("vector", "kg", "knowledge_graph", "graphrag", "llm", "model"):
        frontier = DynamicImpactFrontierAnalyzer().analyze(
            roots,
            f"delta:{claim}",
            [_observation("string_dispatch", route, claim_kind=claim)],
            closure_attempts=[
                ClosureAttempt(
                    entry_route=route,
                    mechanism=ClosureMechanism.REVIEWED_MANIFEST,
                    evidence_refs=("manifest:ignored",),
                    claim_kind=claim,
                )
            ],
        )
        entry = frontier.entries[0]
        # Observation claim_kind alone forces NOMINATED_ONLY before attempts.
        assert entry.disposition is FrontierDisposition.NOMINATED_ONLY
        assert entry.closed_by is None
        assert entry.is_open_required

    # Attempt-only non-closing claim with clean observation still cannot close
    # when mechanism is unsupported for a mismatched path — use vector attempt.
    frontier = DynamicImpactFrontierAnalyzer().analyze(
        roots,
        "delta:vector-attempt",
        [_observation("string_dispatch", route)],
        closure_attempts=[
            ClosureAttempt(
                entry_route=route,
                mechanism=ClosureMechanism.REVIEWED_MANIFEST,
                evidence_refs=("vector:neighbor",),
                claim_kind="vector",
            )
        ],
    )
    assert frontier.entries[0].disposition is FrontierDisposition.OPEN


def test_stale_graph_id_on_closure_attempt_is_ignored(
    roots: PropagationAuthorityRoots,
) -> None:
    route = "src/registry.py:get"
    frontier = DynamicImpactFrontierAnalyzer().analyze(
        roots,
        "delta:stale",
        [_observation("registry", route)],
        closure_attempts=[
            ClosureAttempt(
                entry_route=route,
                mechanism=ClosureMechanism.ADMITTED_EXTRACTOR,
                evidence_refs=("extractor:registry@1",),
                roots_graph_id="graph:stale",
            )
        ],
    )
    assert frontier.entries[0].disposition is FrontierDisposition.OPEN


def test_global_closure_beyond_observed_route_is_rejected(
    roots: PropagationAuthorityRoots,
) -> None:
    route = "src/cb.py:handler"
    frontier = DynamicImpactFrontierAnalyzer().analyze(
        roots,
        "delta:global",
        [_observation("callback", route)],
        closure_attempts=[
            ClosureAttempt(
                entry_route=route,
                mechanism=ClosureMechanism.CONSERVATIVE_RESOLVER,
                evidence_refs=("resolver:cb",),
                observed_route_only=False,
            )
        ],
    )
    assert frontier.entries[0].disposition is FrontierDisposition.OPEN


# ---------------------------------------------------------------------------
# Absent evidence / timeout remain unknown
# ---------------------------------------------------------------------------


def test_absent_evidence_and_timeout_remain_unknown(
    roots: PropagationAuthorityRoots,
) -> None:
    absent = DynamicImpactFrontierAnalyzer().analyze(
        roots,
        "delta:absent",
        [
            _observation(
                "remote_service",
                "services/x.py:call",
                evidence=(),
                absent_evidence=True,
            )
        ],
    )
    assert absent.entries[0].disposition is FrontierDisposition.UNKNOWN
    assert absent.entries[0].is_open_required

    timed = DynamicImpactFrontierAnalyzer().analyze(
        roots,
        "delta:timeout",
        [_observation("ffi", "native/a.c:call", timed_out=True)],
    )
    assert timed.entries[0].disposition is FrontierDisposition.UNKNOWN
    assert timed.timeout is True

    via_report = DynamicImpactFrontierAnalyzer().analyze(
        roots,
        "delta:cap-timeout",
        [_observation("generated", "generated/x.py:run")],
        capability_report=_CapabilityReport(
            capabilities=(_Capability(status="timed_out"),)
        ),
    )
    assert via_report.timeout is True
    assert via_report.entries[0].disposition is FrontierDisposition.UNKNOWN


# ---------------------------------------------------------------------------
# Graph / resolver harvest and closure receipt projection
# ---------------------------------------------------------------------------


def test_graph_frontier_and_exclusion_refs_become_entries(
    roots: PropagationAuthorityRoots,
) -> None:
    graph = _Graph(
        frontier_refs=("reflection:src/reflect.py:invoke", "plugin:src/plugins.py:load"),
        exclusion_refs=("vendor/third_party",),
        complete=False,
    )
    frontier = DynamicImpactFrontierAnalyzer().analyze(
        roots,
        "delta:graph",
        graph=graph,
        affected_contract_ref="contract:dispatch",
    )
    kinds = {entry.kind for entry in frontier.entries}
    assert FrontierKind.REFLECTION in kinds
    assert FrontierKind.PLUGIN_ENTRY_POINT in kinds
    assert FrontierKind.EXCLUDED_ROOT in kinds
    assert frontier.graph_id == roots.graph_id


def test_resolver_frontier_observations_are_merged(
    roots: PropagationAuthorityRoots,
) -> None:
    resolver = _Resolver(
        [
            _observation("monkey_patch", "src/patch.py:swap"),
            _observation("callback", "src/hooks.py:fire"),
        ]
    )
    frontier = DynamicImpactFrontierAnalyzer().analyze(
        roots,
        "delta:resolver",
        [_observation("getattr", "src/dyn.py:get")],
        resolver=resolver,
    )
    routes = {entry.route for entry in frontier.entries}
    assert routes == {
        "src/dyn.py:get",
        "src/patch.py:swap",
        "src/hooks.py:fire",
    }


def test_apply_to_closure_receipt_blocks_complete_when_frontier_open(
    roots: PropagationAuthorityRoots,
) -> None:
    node = GraphNodeRef(
        node_id="node:dispatch",
        kind="function",
        path="src/dispatch.py",
        symbol_id="symbol:dispatch",
        artifact_id="blob:dispatch",
        provenance=GraphProvenance.TRUSTED,
        extractor_id="extractor:ast",
    )
    consumer = ImpactConsumer(
        consumer_id="consumer:one",
        node=node,
        depth=1,
        mandatory=True,
        edge_refs=("edge:call",),
        path_condition_ref="path:always",
    )
    complete = ImpactClosureReceipt(
        roots=roots,
        delta_id="delta:one",
        completeness=ImpactCompleteness.COMPLETE,
        consumers=(consumer,),
    )
    frontier = DynamicImpactFrontierAnalyzer().analyze(
        roots,
        "delta:one",
        [
            _observation(
                "reflection",
                "src/reflect.py:invoke",
                graph_node_id="node:reflect",
                graph_edge_id="edge:dynamic",
            )
        ],
    )
    projected = frontier.apply_to_closure_receipt(complete)
    assert projected.completeness is ImpactCompleteness.PARTIAL_WITH_FRONTIER
    assert "node:reflect" in projected.frontier_node_ids
    assert "edge:dynamic" in projected.frontier_edge_ids

    # Closing the only required entry allows complete to remain complete when
    # the receipt had no residual frontier of its own.
    closed = DynamicImpactFrontierAnalyzer().analyze(
        roots,
        "delta:one",
        [_observation("reflection", "src/reflect.py:invoke")],
        closure_attempts=[
            ClosureAttempt(
                entry_route="src/reflect.py:invoke",
                mechanism=ClosureMechanism.ROOT_BOUND_RUNTIME_WITNESS,
                evidence_refs=("runtime:one",),
                roots_graph_id=roots.graph_id,
            )
        ],
    )
    assert closed.impact_completeness_possible is True
    restored = closed.apply_to_closure_receipt(complete)
    assert restored.completeness is ImpactCompleteness.COMPLETE
    assert restored.frontier_node_ids == ()


def test_root_mismatch_on_receipt_projection_fails_closed(
    roots: PropagationAuthorityRoots,
) -> None:
    other = PropagationAuthorityRoots(
        repository_id="repository:one",
        base_forest_id="forest:base",
        base_tree_id="tree:base",
        base_overlay_id="overlay:base",
        candidate_forest_id="forest:candidate",
        candidate_tree_id="tree:other",
        candidate_overlay_id="overlay:other",
        graph_id="graph:other",
        index_id="index:one",
        model_id="model:one",
        config_id="config:one",
        translator_id="translator:one",
        toolchain_id="toolchain:one",
        policy_id="policy:one",
    )
    receipt = ImpactClosureReceipt(
        roots=other,
        delta_id="delta:one",
        completeness=ImpactCompleteness.COMPLETE,
        consumers=(),
    )
    frontier = DynamicImpactFrontierAnalyzer().analyze(
        roots, "delta:one", [_observation("ffi", "native/x.c:f")]
    )
    with pytest.raises(DynamicImpactFrontierAuthorityError, match="roots"):
        frontier.apply_to_closure_receipt(receipt)


# ---------------------------------------------------------------------------
# Serialization and empty / deterministic behaviour
# ---------------------------------------------------------------------------


def test_frontier_round_trip_and_deterministic_ordering(
    roots: PropagationAuthorityRoots,
) -> None:
    observations = [
        _observation("ffi", "native/b.c:call"),
        _observation("reflection", "src/a.py:invoke"),
        _observation("plugin", "src/c.py:load"),
    ]
    first = DynamicImpactFrontierAnalyzer().analyze(roots, "delta:order", observations)
    second = DynamicImpactFrontierAnalyzer().analyze(
        roots, "delta:order", list(reversed(observations))
    )
    assert [entry.entry_id for entry in first.entries] == [
        entry.entry_id for entry in second.entries
    ]
    restored = DynamicImpactFrontier.from_dict(first.to_dict())
    assert restored.delta_id == first.delta_id
    assert restored.completeness is first.completeness
    assert [entry.kind for entry in restored.entries] == [
        entry.kind for entry in first.entries
    ]
    assert restored.impact_completeness_possible is first.impact_completeness_possible


def test_empty_observations_yield_complete_adapter_scope(
    roots: PropagationAuthorityRoots,
) -> None:
    frontier = DynamicImpactFrontierAnalyzer().analyze(roots, "delta:empty")
    assert frontier.entries == ()
    assert frontier.completeness is ImpactCompleteness.COMPLETE
    assert frontier.impact_completeness_possible is True


def test_entry_records_route_contract_evidence_mechanisms_and_reason(
    roots: PropagationAuthorityRoots,
) -> None:
    frontier = DynamicImpactFrontierAnalyzer().analyze(
        roots,
        "delta:fields",
        [
            FrontierObservation(
                kind=FrontierKind.STRING_DISPATCH,
                route="src/dyn.py:getattr",
                affected_contract_ref="contract:process",
                reason="getattr target not statically resolvable",
                evidence_refs=("ast:getattr-call", "graph:edge-dynamic"),
            )
        ],
    )
    entry = frontier.entries[0]
    payload = entry.to_dict()
    assert payload["route"] == "src/dyn.py:getattr"
    assert payload["affected_contract_ref"] == "contract:process"
    assert payload["evidence_refs"] == ["ast:getattr-call", "graph:edge-dynamic"]
    assert "reviewed_manifest" in payload["supported_closure_mechanisms"]
    assert payload["reason"]
    assert payload["disposition"] == "open"


def test_unsupported_kind_and_malformed_observation_fail_closed() -> None:
    with pytest.raises(DynamicImpactFrontierError):
        FrontierKind.coerce("not_a_real_kind")
    with pytest.raises(DynamicImpactFrontierError):
        FrontierObservation(
            kind="reflection",
            route="",
            affected_contract_ref="contract:x",
        )


def test_all_kinds_list_supported_closure_mechanisms(
    roots: PropagationAuthorityRoots,
) -> None:
    """Every kind documents at least one admitted closure mechanism."""
    observations = [
        _observation(kind, f"route/{kind.value}")
        for kind in all_frontier_kinds()
    ]
    frontier = DynamicImpactFrontierAnalyzer().analyze(
        roots, "delta:mechanisms", observations
    )
    for entry in frontier.entries:
        assert entry.supported_closure_mechanisms
        for mechanism in entry.supported_closure_mechanisms:
            assert mechanism in {
                ClosureMechanism.REVIEWED_MANIFEST,
                ClosureMechanism.ROOT_BOUND_RUNTIME_WITNESS,
                ClosureMechanism.ADMITTED_EXTRACTOR,
                ClosureMechanism.CONSERVATIVE_RESOLVER,
            }
            # Never vector/kg/llm.
            assert mechanism.value not in {
                "vector",
                "kg",
                "llm",
                "graphrag",
            }
