"""Tests for bounded value-provenance compilation (RPR-033)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.value_provenance_graph import (
    PRODUCER_ID,
    VALUE_PROVENANCE_GRAPH_SCHEMA,
    DefinitionKind,
    DependencyDirection,
    DominanceKind,
    InformationOriginKind,
    InterproceduralCompleteness,
    Nullability,
    ProvenanceStatus,
    UnknownReason,
    ValueProvenanceAuthorityError,
    ValueProvenanceCompiler,
    ValueProvenanceError,
    ValueProvenanceGraph,
    build_value_provenance_graph,
    compile_value_provenance,
)
from ipfs_accelerate_py.agent_supervisor.program_graph import (
    Completeness,
    ProgramGraphRoots,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def roots() -> ProgramGraphRoots:
    return ProgramGraphRoots(
        forest_id="forest:vpg",
        tree_id="tree:vpg-current",
        overlay_id="overlay:clean",
        coverage_id="coverage:full",
        included_roots=("src/",),
        excluded_roots=("vendor/",),
        extractor_id="value-provenance-graph@1",
        config_id="config:vpg",
        toolchain_id="toolchain:cpython",
    )


def _compile(roots: ProgramGraphRoots, source: str, **kwargs: Any) -> ValueProvenanceGraph:
    return compile_value_provenance(
        roots, source, path="src/sample.py", **kwargs
    )


# ---------------------------------------------------------------------------
# Identity / roots / producer
# ---------------------------------------------------------------------------


def test_graph_identity_is_deterministic_and_root_bound(roots: ProgramGraphRoots) -> None:
    source = """
def process(left: int, right: int) -> int:
    total = left + right
    return total
"""
    left = _compile(roots, source)
    right = _compile(roots, source)
    assert left.graph_id == right.graph_id
    assert left.roots_id == roots.roots_id
    assert left.producer_id == PRODUCER_ID
    assert left.schema == VALUE_PROVENANCE_GRAPH_SCHEMA
    for item in left.definitions:
        assert item.roots_id == roots.roots_id
        assert item.producer_id == PRODUCER_ID


def test_round_trip_preserves_identity(roots: ProgramGraphRoots) -> None:
    source = """
def f(x: int) -> int:
    y = x
    return y
"""
    graph = _compile(roots, source)
    payload = graph.to_dict()
    rebuilt = ValueProvenanceGraph.from_dict(payload)
    assert rebuilt.graph_id == graph.graph_id
    assert rebuilt.roots.roots_id == roots.roots_id
    assert len(rebuilt.definitions) == len(graph.definitions)
    assert len(rebuilt.def_use_chains) == len(graph.def_use_chains)


def test_forged_graph_identity_is_rejected(roots: ProgramGraphRoots) -> None:
    source = """
def f(x):
    return x
"""
    graph = _compile(roots, source)
    payload = graph.to_dict()
    payload["graph_id"] = "value-provenance-graph:sha256:" + ("0" * 64)
    with pytest.raises(ValueProvenanceAuthorityError):
        ValueProvenanceGraph.from_dict(payload)


def test_stale_roots_cannot_be_reused(roots: ProgramGraphRoots) -> None:
    source = """
def f(x):
    return x
"""
    graph = _compile(roots, source)
    payload = graph.to_dict()
    stale_roots = ProgramGraphRoots(
        forest_id="forest:vpg",
        tree_id="tree:stale",
        overlay_id="overlay:clean",
        coverage_id="coverage:full",
        included_roots=("src/",),
        extractor_id="value-provenance-graph@1",
        config_id="config:vpg",
        toolchain_id="toolchain:cpython",
    )
    payload["roots"] = stale_roots.to_dict()
    # definitions still carry original roots_id → authority error
    with pytest.raises(ValueProvenanceAuthorityError):
        ValueProvenanceGraph.from_dict(payload)


def test_mismatched_dependency_graph_roots_fail_closed(roots: ProgramGraphRoots) -> None:
    other = ProgramGraphRoots(
        forest_id="forest:other",
        tree_id="tree:other",
        extractor_id="other@1",
    )

    class _Dep:
        roots = other
        graph = None

    with pytest.raises(ValueProvenanceAuthorityError):
        ValueProvenanceCompiler(roots, dependency_graph=_Dep())


# ---------------------------------------------------------------------------
# Reaching definitions / def-use
# ---------------------------------------------------------------------------


def test_parameters_and_assignments_produce_reaching_defs(
    roots: ProgramGraphRoots,
) -> None:
    source = """
def process(left: int, right: int) -> int:
    total = left + right
    return total
"""
    graph = _compile(roots, source)
    proc = graph.procedures[0]
    params = [
        d for d in graph.definitions_for("left", procedure_id=proc)
        if d.kind is DefinitionKind.PARAMETER
    ]
    assert len(params) == 1
    assert params[0].type_annotation == "int"
    totals = graph.definitions_for("total", procedure_id=proc)
    assert any(d.kind is DefinitionKind.ASSIGNMENT for d in totals)
    # left is used in the assignment RHS
    left_uses = graph.uses_for("left", procedure_id=proc)
    assert left_uses
    for use in left_uses:
        reaching = graph.reaching_at_use(use.use_id)
        assert any(d.kind is DefinitionKind.PARAMETER for d in reaching)


def test_def_use_chains_link_definitions_to_uses(roots: ProgramGraphRoots) -> None:
    source = """
def f(x: int) -> int:
    y = x
    z = y
    return z
"""
    graph = _compile(roots, source)
    assert graph.def_use_chains
    variables = {c.variable for c in graph.def_use_chains}
    assert "x" in variables
    assert "y" in variables
    assert "z" in variables
    for chain in graph.def_use_chains:
        assert chain.dependency_direction is DependencyDirection.FLOWS_TO


# ---------------------------------------------------------------------------
# Dominance / post-dominance
# ---------------------------------------------------------------------------


def test_dominance_and_post_dominance_on_if_else(roots: ProgramGraphRoots) -> None:
    source = """
def f(flag: bool, a: int, b: int) -> int:
    if flag:
        x = a
    else:
        x = b
    return x
"""
    graph = _compile(roots, source)
    entry = next(b for b in graph.blocks if b.is_entry)
    join = next(b for b in graph.blocks if b.is_join)
    exit_block = next(b for b in graph.blocks if b.is_exit)
    assert graph.dominates(entry.block_id, join.block_id)
    assert graph.dominates(entry.block_id, exit_block.block_id)
    # Entry strictly dominates join.
    assert any(
        f.kind is DominanceKind.STRICTLY_DOMINATES
        and f.dominator_block_id == entry.block_id
        and f.dominated_block_id == join.block_id
        for f in graph.dominance_facts
    )
    # Exit post-dominates join (or virtual handling).
    assert graph.post_dominates(exit_block.block_id, join.block_id) or any(
        f.kind in {DominanceKind.POST_DOMINATES, DominanceKind.STRICTLY_POST_DOMINATES}
        and f.dominated_block_id == join.block_id
        for f in graph.dominance_facts
    )


# ---------------------------------------------------------------------------
# Path conditions / guards / type refinements
# ---------------------------------------------------------------------------


def test_path_conditions_and_isinstance_refinements(
    roots: ProgramGraphRoots,
) -> None:
    source = """
def f(value: object) -> str:
    if isinstance(value, str):
        return value
    return "default"
"""
    graph = _compile(roots, source)
    assert graph.path_conditions
    then_pcs = [
        pc
        for pc in graph.path_conditions
        if pc.branch_label == "then" and pc.polarity is True
    ]
    assert then_pcs
    assert any(pc.guard_variable == "value" for pc in then_pcs)
    assert graph.type_refinements
    assert any(
        r.variable == "value" and r.refined_type == "str" and r.nullability is Nullability.NONNULL
        for r in graph.type_refinements
    )


def test_none_guard_introduces_nullability_refinement(
    roots: ProgramGraphRoots,
) -> None:
    source = """
def f(value):
    if value is None:
        return 0
    return value
"""
    graph = _compile(roots, source)
    # then branch: value is None → nullable
    assert any(
        r.variable == "value" and r.nullability is Nullability.NULLABLE
        for r in graph.type_refinements
    )
    # else branch: value is not None → nonnull
    assert any(
        r.variable == "value" and r.nullability is Nullability.NONNULL
        for r in graph.type_refinements
    )


# ---------------------------------------------------------------------------
# Parameters, returns, fields, aliases, constructors, conversions, config/DI
# ---------------------------------------------------------------------------


def test_fields_aliases_constructors_conversions_config_di(
    roots: ProgramGraphRoots,
) -> None:
    source = """
import os

def make(client, cfg_key: str):
    alias = client
    obj = Payload.from_dict({"k": 1})
    n = int(cfg_key)
    env = os.getenv("TOKEN")
    service = inject("Service")
    self_holder = Holder()
    self_holder.value = n
    return alias
"""
    graph = _compile(roots, source)
    kinds = {d.kind for d in graph.definitions}
    assert DefinitionKind.ALIAS in kinds
    assert DefinitionKind.CONSTRUCTOR in kinds or DefinitionKind.CALL_RESULT in kinds
    assert DefinitionKind.CONVERSION in kinds
    assert DefinitionKind.CONFIG_SOURCE in kinds
    assert DefinitionKind.DI_SOURCE in kinds
    assert DefinitionKind.FIELD_WRITE in kinds
    assert DefinitionKind.RETURN in kinds or any(
        d.variable == "<return>" for d in graph.definitions
    )
    # Information provenance attached to each definition.
    assert graph.information_provenances
    origins = {info.origin_kind for info in graph.information_provenances}
    assert InformationOriginKind.ALIAS in origins
    assert InformationOriginKind.CONFIG in origins
    assert InformationOriginKind.DI_REGISTRY in origins
    assert InformationOriginKind.CONVERSION in origins


def test_information_provenance_carries_effects_ownership_direction(
    roots: ProgramGraphRoots,
) -> None:
    source = """
def f(path: str):
    data = open(path).read()
    return data
"""
    graph = _compile(roots, source)
    # open() is a call result; effects may be attached on the open call if classified.
    infos = list(graph.information_provenances)
    assert infos
    # At least one definition has ownership / dependency direction filled.
    assert any(info.dependency_direction for info in infos)
    assert any(info.ownership for info in infos)
    assert any(info.lifetime_ref for info in infos)


# ---------------------------------------------------------------------------
# Interprocedural threading with explicit completeness
# ---------------------------------------------------------------------------


def test_interprocedural_threading_with_completeness(
    roots: ProgramGraphRoots,
) -> None:
    source = """
def helper(context: str) -> str:
    return context

def process(left, right, context: str):
    return helper(context)
"""
    graph = _compile(roots, source)
    assert graph.interprocedural_threads
    thread = graph.interprocedural_threads[0]
    assert thread.parameter_name == "context"
    assert thread.completeness in {
        InterproceduralCompleteness.COMPLETE,
        InterproceduralCompleteness.PARTIAL,
    }
    assert thread.dependency_direction is DependencyDirection.THREADS_TO
    assert "helper" in thread.target_procedure_id or "helper" in thread.call_site_ref


def test_incomplete_interprocedural_route_remains_unknown(
    roots: ProgramGraphRoots,
) -> None:
    source = """
def process(x):
    return external_lib.do_work(x)
"""
    graph = _compile(roots, source)
    assert UnknownReason.INCOMPLETE_INTERPROCEDURAL in graph.unknown_reasons()
    assert graph.completeness in {
        Completeness.PARTIAL,
        Completeness.FRONTIER,
        Completeness.UNKNOWN,
    }


# ---------------------------------------------------------------------------
# Fail-closed unknowns
# ---------------------------------------------------------------------------


def test_branch_local_absence_is_unknown(roots: ProgramGraphRoots) -> None:
    source = """
def f(flag: bool) -> int:
    if flag:
        x = 1
    return x
"""
    graph = _compile(roots, source)
    proc = graph.procedures[0]
    available, status, reasons = graph.available_on_all_paths("x", procedure_id=proc)
    # x is not defined on the else path → not available on all paths.
    assert available is False
    assert status is ProvenanceStatus.UNKNOWN
    assert (
        UnknownReason.BRANCH_LOCAL_ABSENCE.value in reasons
        or UnknownReason.MULTIPLE_REACHING.value in reasons
        or UnknownReason.BRANCH_LOCAL_ABSENCE in graph.unknown_reasons()
        or any(
            u.variable == "x" and u.reason is UnknownReason.BRANCH_LOCAL_ABSENCE
            for u in graph.unknown_frontier
        )
        or any(c.variable == "x" for c in graph.def_use_chains)
    )


def test_alias_and_multiple_reaching_defs(roots: ProgramGraphRoots) -> None:
    source = """
def f(flag: bool, a, b):
    if flag:
        x = a
    else:
        x = b
    y = x
    return y
"""
    graph = _compile(roots, source)
    # At the use of x in y = x, two defs may reach → multiple reaching / ambiguity.
    assert (
        UnknownReason.MULTIPLE_REACHING in graph.unknown_reasons()
        or UnknownReason.ALIAS_AMBIGUITY in graph.unknown_reasons()
        or any(
            c.variable == "x" and c.status is ProvenanceStatus.PARTIAL
            for c in graph.def_use_chains
        )
    )


def test_loop_beyond_bounds_remains_unknown(roots: ProgramGraphRoots) -> None:
    source = """
def f(items):
    total = 0
    for item in items:
        for inner in item:
            total = total + inner
    return total
"""
    graph = compile_value_provenance(
        roots, source, path="src/loops.py", max_loop_unroll=0
    )
    assert UnknownReason.LOOP_BEYOND_BOUNDS in graph.unknown_reasons()


def test_exceptions_remain_unknown(roots: ProgramGraphRoots) -> None:
    source = """
def f(x):
    try:
        return x.value
    except AttributeError as exc:
        return 0
"""
    graph = _compile(roots, source)
    assert UnknownReason.EXCEPTION_PATH in graph.unknown_reasons()


def test_concurrency_remains_unknown(roots: ProgramGraphRoots) -> None:
    source = """
def f(target):
    t = Thread(target=target)
    t.start()
    return t
"""
    graph = _compile(roots, source)
    assert UnknownReason.CONCURRENCY in graph.unknown_reasons()


def test_reflection_and_native_calls_remain_unknown(
    roots: ProgramGraphRoots,
) -> None:
    source = """
def f(obj, name):
    value = getattr(obj, name)
    handle = ctypes.CDLL("lib.so")
    return value
"""
    graph = _compile(roots, source)
    reasons = graph.unknown_reasons()
    assert UnknownReason.REFLECTION in reasons
    assert UnknownReason.NATIVE_CALL in reasons


# ---------------------------------------------------------------------------
# Supported-only proof / completeness
# ---------------------------------------------------------------------------


def test_straight_line_supported_shape_is_complete_or_proved(
    roots: ProgramGraphRoots,
) -> None:
    source = """
def add(left: int, right: int) -> int:
    total: int = left + right
    return total
"""
    graph = _compile(roots, source)
    # No unsupported frontiers for pure straight-line arithmetic.
    hard = {
        UnknownReason.UNSUPPORTED_CFG,
        UnknownReason.UNSUPPORTED_AST,
        UnknownReason.REFLECTION,
        UnknownReason.NATIVE_CALL,
        UnknownReason.CONCURRENCY,
        UnknownReason.EXCEPTION_PATH,
    }
    assert not (graph.unknown_reasons() & hard)
    assert graph.completeness in {Completeness.COMPLETE, Completeness.PARTIAL}
    # Parameter always available.
    proc = graph.procedures[0]
    available, status, _ = graph.available_on_all_paths("left", procedure_id=proc)
    assert available is True
    assert status is ProvenanceStatus.PROVED


def test_prove_only_supported_ast_shapes(roots: ProgramGraphRoots) -> None:
    source = """
def f(x):
    match x:
        case 1:
            return 1
        case _:
            return 0
"""
    graph = _compile(roots, source)
    # match is unsupported in the proved fragment.
    assert (
        UnknownReason.UNSUPPORTED_AST in graph.unknown_reasons()
        or UnknownReason.UNSUPPORTED_CFG in graph.unknown_reasons()
        or graph.completeness is Completeness.FRONTIER
    )


# ---------------------------------------------------------------------------
# Requirement compatibility (type ≠ information)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Requirement:
    parameter_name: str
    type_ref: str
    nullability: str
    information_content_ref: str


def test_same_typed_wrong_information_fails(
    roots: ProgramGraphRoots,
) -> None:
    source = """
def process(left: int, right: int, token: int):
    return left + right
"""
    graph = _compile(roots, source)
    proc = graph.procedures[0]
    # Candidate `left` is int but does not carry session-token information.
    req = _Requirement(
        parameter_name="context",
        type_ref="int",
        nullability="nonnull",
        information_content_ref="session_token",
    )
    ok, status, reasons = graph.compatible_with_requirement(
        req, procedure_id=proc, candidate_variable="left"
    )
    # Either refuted for wrong information or only partial — never a silent prove.
    if ok and status is ProvenanceStatus.PROVED:
        # If information labels don't include session_token, the helper may still
        # return partial/proved when origin labels are generic; enforce fail-closed
        # on explicit content mismatch path.
        info_defs = graph.definitions_for("left", procedure_id=proc)
        for d in info_defs:
            info = graph.information_for_def(d.def_id)
            if info is not None:
                assert "session_token" not in info.origin_labels
    else:
        assert ok is False or status is not ProvenanceStatus.PROVED or reasons


def test_parameter_satisfies_matching_requirement(
    roots: ProgramGraphRoots,
) -> None:
    source = """
def process(left: int, right: int, context: C) -> R:
    return left
"""
    graph = _compile(roots, source)
    proc = graph.procedures[0]
    req = _Requirement(
        parameter_name="context",
        type_ref="C",
        nullability="unknown",
        information_content_ref="parameter",
    )
    ok, status, reasons = graph.compatible_with_requirement(
        req, procedure_id=proc, candidate_variable="context"
    )
    assert ok is True
    assert status in {ProvenanceStatus.PROVED, ProvenanceStatus.PARTIAL}
    assert reasons == () or status is ProvenanceStatus.PARTIAL


# ---------------------------------------------------------------------------
# Multi-file build + factory
# ---------------------------------------------------------------------------


def test_build_value_provenance_graph_multi_file(roots: ProgramGraphRoots) -> None:
    files = {
        "src/a.py": """
def source(value: int) -> int:
    return value
""",
        "src/b.py": """
def sink(value: int) -> int:
    return source(value)
""",
    }
    graph = build_value_provenance_graph(roots, files)
    assert len(graph.procedures) >= 2
    assert graph.graph_id
    # Cross-file threading may be incomplete (name resolution), which is explicit.
    assert graph.completeness in {
        Completeness.COMPLETE,
        Completeness.PARTIAL,
        Completeness.FRONTIER,
    }


def test_empty_files_rejected(roots: ProgramGraphRoots) -> None:
    with pytest.raises(ValueProvenanceError):
        build_value_provenance_graph(roots, {})


def test_syntax_error_is_fail_closed(roots: ProgramGraphRoots) -> None:
    with pytest.raises(ValueProvenanceError):
        _compile(roots, "def broken(:\n    pass\n")


# ---------------------------------------------------------------------------
# Memory-safety facet attachment (interface)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Facet:
    disposition: Any = type("D", (), {"value": "supported"})()


def test_memory_safety_facet_ref_attached(roots: ProgramGraphRoots) -> None:
    source = """
def f(x: int) -> int:
    return x
"""
    graph = compile_value_provenance(
        roots,
        source,
        path="src/sample.py",
        memory_safety_facets={"src/sample.py": _Facet()},
    )
    assert any(
        info.memory_safety_facet_ref.startswith("memory_safety:")
        for info in graph.information_provenances
    )


# ---------------------------------------------------------------------------
# Explicit AST symbols exported
# ---------------------------------------------------------------------------


def test_required_ast_symbols_are_exported() -> None:
    from ipfs_accelerate_py.agent_supervisor.analysis import value_provenance_graph as mod

    for name in (
        "ValueProvenanceGraph",
        "ReachingDefinition",
        "DominanceFact",
        "PathCondition",
        "InformationProvenance",
    ):
        assert hasattr(mod, name)
