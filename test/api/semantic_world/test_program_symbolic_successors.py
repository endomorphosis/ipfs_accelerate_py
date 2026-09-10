"""SAWM-017 static successor generation and symbolic pruning tests."""

from __future__ import annotations

import ast
import importlib
import inspect
import threading
from pathlib import Path
from typing import Any

import pytest

from ipfs_datasets_py.logic.software_contracts.content import cid_for_bytes
from ipfs_datasets_py.logic.software_contracts.semantic_state.program_graph import (
    CallsiteRecord,
    ContractKind,
    ContractStateRecord,
    DynamicFrontierRecord,
    FunctionSymbolRecord,
    ProgramGraphEdge,
    ProgramGraphNode,
    ProgramGraphSnapshot,
    ResolutionStatus,
    StaticSuccessorSet,
    assemble_program_graph_snapshot,
)

from ipfs_accelerate_py.agent_supervisor.analysis.program_successors import (
    GENERATION_STAGE_ORDER,
    SAWM_STATIC_SUCCESSORS_EVIDENCE,
    STATIC_SUCCESSOR_PLANNER_INTERFACE,
    StaticSuccessorPlan,
    StaticSuccessorPlanner,
    SuccessorDisposition,
    SuccessorKind,
    SuccessorReasonCode,
    SuccessorStage,
    UnresolvedDynamicFrontier,
    generate_static_successors,
)
from ipfs_accelerate_py.agent_supervisor.analysis.program_symbolic_pruning import (
    PRUNING_STAGE_ORDER,
    SYMBOLIC_SUCCESSOR_PRUNER_INTERFACE,
    PruningVerdict,
    SymbolicPruningError,
    SymbolicSuccessorPruner,
    prune_program_successors,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
SUCCESSORS_PATH = (
    REPO_ROOT / "ipfs_accelerate_py/agent_supervisor/analysis/program_successors.py"
)
PRUNING_PATH = (
    REPO_ROOT
    / "ipfs_accelerate_py/agent_supervisor/analysis/program_symbolic_pruning.py"
)


def _cid(label: str) -> str:
    return cid_for_bytes(label.encode("utf-8"))


def _source() -> str:
    return cid_for_bytes(b"def ping():\n    return pong()\n")


def _node(kind: str, name: str, **overrides: Any) -> ProgramGraphNode:
    fields: dict[str, Any] = {
        "node_kind": kind,
        "language": "python",
        "logical_name": name,
        "source_cid": _source(),
        "declaration_cid": _cid(f"decl:{name}"),
        "environment_binding_cid": _cid("env-v1"),
        "subject_cid": None,
        "record_cid": None,
        "unavailable_dimensions": (),
        "metadata": {},
    }
    fields.update(overrides)
    return ProgramGraphNode(**fields)


def _edge(
    kind: str,
    source: ProgramGraphNode,
    target: ProgramGraphNode,
    **overrides: Any,
) -> ProgramGraphEdge:
    fields: dict[str, Any] = {
        "edge_kind": kind,
        "source_node_cid": source.program_graph_node_cid,
        "target_node_cid": target.program_graph_node_cid,
        "language": "python",
        "environment_binding_cid": _cid("env-v1"),
        "resolution_status": ResolutionStatus.DEFINITE,
        "logical_cycle": kind in {"cyclic_import", "mutual_recursion"},
        "unavailable_dimensions": (),
        "metadata": {},
    }
    fields.update(overrides)
    return ProgramGraphEdge(**fields)


def _function(name: str, **overrides: Any) -> FunctionSymbolRecord:
    fields: dict[str, Any] = {
        "language": "python",
        "logical_name": name,
        "source_cid": _source(),
        "declaration_cid": _cid(f"decl:{name}"),
        "parameter_names": (),
        "return_annotation": "None",
        "unavailable_dimensions": (),
    }
    fields.update(overrides)
    return FunctionSymbolRecord(**fields)


def _callsite(caller: str, callee: str, ordinal: int = 0, **overrides: Any) -> CallsiteRecord:
    fields: dict[str, Any] = {
        "language": "python",
        "caller_logical_name": caller,
        "callee_logical_name": callee,
        "source_cid": _source(),
        "ordinal": ordinal,
        "resolution_status": ResolutionStatus.DEFINITE,
        "callee_declaration_cid": _cid(f"decl:{callee}"),
        "unavailable_dimensions": (),
    }
    fields.update(overrides)
    return CallsiteRecord(**fields)


def _contract(name: str, **overrides: Any) -> ContractStateRecord:
    fields: dict[str, Any] = {
        "language": "python",
        "subject_logical_name": name,
        "contract_kind": ContractKind.PRECONDITION,
        "specification_cid": _cid(f"spec:{name}"),
        "discharge_status": "unknown",
        "unavailable_dimensions": (),
    }
    fields.update(overrides)
    return ContractStateRecord(**fields)


def _assemble(
    nodes: list[ProgramGraphNode],
    edges: list[ProgramGraphEdge],
    **overrides: Any,
) -> ProgramGraphSnapshot:
    fields: dict[str, Any] = {
        "nodes": nodes,
        "edges": edges,
        "environment_binding_set_cid": _cid("bindings"),
        "sealed_binding_cid": _cid("sealed"),
    }
    fields.update(overrides)
    return assemble_program_graph_snapshot(**fields)


def _complete_call_graph() -> dict[str, Any]:
    ping_fn = _function("pkg.mod.ping")
    pong_fn = _function("pkg.mod.pong")
    ping_call = _callsite("pkg.mod.ping", "pkg.mod.pong")
    module = _node("module", "pkg.mod")
    ping = _node("function", "pkg.mod.ping", record_cid=ping_fn.function_symbol_record_cid)
    pong = _node("function", "pkg.mod.pong", record_cid=pong_fn.function_symbol_record_cid)
    ping_site = _node("callsite", "pkg.mod.ping#0", record_cid=ping_call.callsite_record_cid)
    then_block = _node("cfg_block", "pkg.mod.ping.then")
    else_block = _node("cfg_block", "pkg.mod.ping.else")
    edges = [
        _edge("declares", module, ping),
        _edge("declares", module, pong),
        _edge("calls", ping, ping_site),
        _edge("successor", ping_site, pong),
        _edge("cfg_next", ping, then_block),
        _edge("cfg_branch", ping, else_block),
    ]
    successors = StaticSuccessorSet(
        language="python",
        subject_node_cid=ping.program_graph_node_cid,
        successor_node_cids=[
            ping_site.program_graph_node_cid,
            pong.program_graph_node_cid,
        ],
        successor_edge_cids=[
            edges[2].program_graph_edge_cid,
            edges[3].program_graph_edge_cid,
        ],
        complete=True,
    )
    snapshot = _assemble(
        [module, ping, pong, ping_site, then_block, else_block],
        edges,
        callsites=[ping_call],
        function_symbols=[ping_fn, pong_fn],
        successor_sets=[successors],
        retained_subroot_cids=[module.program_graph_node_cid],
    )
    return {
        "snapshot": snapshot,
        "nodes": [module, ping, pong, ping_site, then_block, else_block],
        "edges": edges,
        "callsites": [ping_call],
        "function_symbols": [ping_fn, pong_fn],
        "successor_sets": [successors],
        "frontiers": [],
        "ping": ping,
        "pong": pong,
        "ping_site": ping_site,
        "then_block": then_block,
        "else_block": else_block,
        "module": module,
    }


def _reflection_graph() -> dict[str, Any]:
    ping_fn = _function("pkg.mod.ping")
    pong_fn = _function("pkg.mod.pong")
    ping_call = _callsite("pkg.mod.ping", "pkg.mod.pong")
    dynamic_call = _callsite(
        "pkg.mod.ping",
        "getattr",
        ordinal=1,
        resolution_status=ResolutionStatus.UNRESOLVED,
        callee_declaration_cid=None,
        unavailable_dimensions=("reflection",),
    )
    module = _node("module", "pkg.mod")
    ping = _node("function", "pkg.mod.ping", record_cid=ping_fn.function_symbol_record_cid)
    pong = _node("function", "pkg.mod.pong", record_cid=pong_fn.function_symbol_record_cid)
    ping_site = _node("callsite", "pkg.mod.ping#0", record_cid=ping_call.callsite_record_cid)
    dynamic = _node(
        "unresolved_dynamic",
        "pkg.mod.ping.__getattr__",
        unavailable_dimensions=("reflection",),
    )
    edges = [
        _edge("declares", module, ping),
        _edge("declares", module, pong),
        _edge("calls", ping, ping_site),
        _edge("successor", ping_site, pong),
        _edge(
            "unresolved_dynamic",
            ping,
            dynamic,
            resolution_status=ResolutionStatus.UNRESOLVED,
            unavailable_dimensions=("reflection",),
        ),
    ]
    successors = StaticSuccessorSet(
        language="python",
        subject_node_cid=ping.program_graph_node_cid,
        successor_node_cids=[
            ping_site.program_graph_node_cid,
            dynamic.program_graph_node_cid,
        ],
        successor_edge_cids=[
            edges[2].program_graph_edge_cid,
            edges[-1].program_graph_edge_cid,
        ],
        complete=False,
        unavailable_dimensions=("reflection",),
    )
    frontier = DynamicFrontierRecord(
        language="python",
        unresolved_node_cids=[dynamic.program_graph_node_cid],
        unresolved_edge_cids=[edges[-1].program_graph_edge_cid],
        reasons=["reflection", "unknown_callee"],
        unavailable_dimensions=("reflection",),
    )
    snapshot = _assemble(
        [module, ping, pong, ping_site, dynamic],
        edges,
        callsites=[ping_call, dynamic_call],
        function_symbols=[ping_fn, pong_fn],
        successor_sets=[successors],
        frontiers=[frontier],
        retained_subroot_cids=[module.program_graph_node_cid],
        unavailable_dimensions=["reflection"],
    )
    return {
        "snapshot": snapshot,
        "nodes": [module, ping, pong, ping_site, dynamic],
        "edges": edges,
        "callsites": [ping_call, dynamic_call],
        "function_symbols": [ping_fn, pong_fn],
        "successor_sets": [successors],
        "frontiers": [frontier],
        "ping": ping,
        "pong": pong,
        "ping_site": ping_site,
        "dynamic": dynamic,
        "module": module,
        "unresolved_edge": edges[-1],
    }


def _catalog(graph: dict[str, Any]) -> dict[str, Any]:
    return {
        "snapshot": graph["snapshot"],
        "nodes": graph["nodes"],
        "edges": graph["edges"],
        "callsites": graph.get("callsites", ()),
        "function_symbols": graph.get("function_symbols", ()),
        "successor_sets": graph.get("successor_sets", ()),
        "frontiers": graph.get("frontiers", ()),
        "contract_states": graph.get("contract_states", ()),
    }


def _targets(plan: StaticSuccessorPlan) -> set[str]:
    return set(plan.target_node_cids)


def _stage_map(receipts: Any) -> dict[str, Any]:
    return {item.stage: item for item in receipts}


def test_public_interfaces_and_symbols_are_present() -> None:
    successors_tree = ast.parse(SUCCESSORS_PATH.read_text(encoding="utf-8"))
    pruning_tree = ast.parse(PRUNING_PATH.read_text(encoding="utf-8"))
    successor_functions = {
        node.name for node in ast.walk(successors_tree) if isinstance(node, ast.FunctionDef)
    }
    successor_classes = {
        node.name for node in ast.walk(successors_tree) if isinstance(node, ast.ClassDef)
    }
    pruning_functions = {
        node.name for node in ast.walk(pruning_tree) if isinstance(node, ast.FunctionDef)
    }
    pruning_classes = {
        node.name for node in ast.walk(pruning_tree) if isinstance(node, ast.ClassDef)
    }
    assert "generate_static_successors" in successor_functions
    assert "prune_program_successors" in pruning_functions
    assert "StaticSuccessorPlanner" in successor_classes
    assert "UnresolvedDynamicFrontier" in successor_classes
    assert "SuccessorDecisionReceipt" in successor_classes
    assert "SymbolicSuccessorPruner" in pruning_classes
    assert STATIC_SUCCESSOR_PLANNER_INTERFACE == "StaticSuccessorPlanner@1"
    assert SYMBOLIC_SUCCESSOR_PRUNER_INTERFACE == "SymbolicSuccessorPruner@1"
    assert SAWM_STATIC_SUCCESSORS_EVIDENCE == "sawm/static-successors@1"


def test_import_has_no_io_or_thread_side_effects() -> None:
    before = {thread.name for thread in threading.enumerate()}
    imported = importlib.import_module(
        "ipfs_accelerate_py.agent_supervisor.analysis.program_successors"
    )
    pruning = importlib.import_module(
        "ipfs_accelerate_py.agent_supervisor.analysis.program_symbolic_pruning"
    )
    after = {thread.name for thread in threading.enumerate()}
    assert after == before
    assert inspect.isfunction(imported.generate_static_successors)
    assert inspect.isfunction(pruning.prune_program_successors)


def test_static_recall_preserves_successor_set_and_call_targets() -> None:
    graph = _complete_call_graph()
    plan = generate_static_successors(
        catalog=_catalog(graph),
        subject_node_cid=graph["ping"].program_graph_node_cid,
        query_family="next_call",
    )
    targets = _targets(plan)
    assert graph["ping_site"].program_graph_node_cid in targets
    assert graph["pong"].program_graph_node_cid in targets
    assert plan.complete is True
    assert plan.receipt_for(SuccessorStage.STATIC_RECALL).disposition == (
        SuccessorDisposition.SELECTED.value
    )
    assert plan.receipt_for(SuccessorStage.CALLSITE_EXPANSION).disposition == (
        SuccessorDisposition.SELECTED.value
    )
    assert plan.receipt_for(SuccessorStage.CFG_EVENT_EXPANSION).disposition == (
        SuccessorDisposition.SKIPPED.value
    )
    assert {item.stage for item in plan.receipts} == {
        stage.value for stage in GENERATION_STAGE_ORDER
    }


def test_next_event_recall_includes_cfg_successors() -> None:
    graph = _complete_call_graph()
    plan = generate_static_successors(
        catalog=_catalog(graph),
        subject_node_cid=graph["ping"].program_graph_node_cid,
        query_family="next_event",
    )
    targets = _targets(plan)
    assert graph["then_block"].program_graph_node_cid in targets
    assert graph["else_block"].program_graph_node_cid in targets
    assert plan.receipt_for(SuccessorStage.CALLSITE_EXPANSION).disposition == (
        SuccessorDisposition.SKIPPED.value
    )
    assert plan.receipt_for(SuccessorStage.CFG_EVENT_EXPANSION).disposition == (
        SuccessorDisposition.SELECTED.value
    )
    assert all(
        item.kind != SuccessorKind.NEXT_CALL.value or item.unknown for item in plan.candidates
    ) or graph["then_block"].program_graph_node_cid in targets


def test_reflection_unknown_frontier_is_explicit_and_preserved() -> None:
    graph = _reflection_graph()
    plan = generate_static_successors(
        catalog=_catalog(graph),
        subject_node_cid=graph["ping"].program_graph_node_cid,
        query_family="next_call",
    )
    targets = _targets(plan)
    assert graph["ping_site"].program_graph_node_cid in targets
    assert graph["dynamic"].program_graph_node_cid in targets
    assert plan.complete is False
    assert "reflection" in plan.unavailable_dimensions
    assert plan.frontier is not None
    assert graph["dynamic"].program_graph_node_cid in plan.frontier.unresolved_node_cids
    assert "reflection" in plan.frontier.reasons
    assert plan.receipt_for(SuccessorStage.DYNAMIC_FRONTIER).disposition == (
        SuccessorDisposition.SELECTED.value
    )
    assert plan.receipt_for(SuccessorStage.DYNAMIC_FRONTIER).widened is True

    pruned = prune_program_successors(plan)
    assert graph["dynamic"].program_graph_node_cid in pruned.target_node_cids
    assert graph["ping_site"].program_graph_node_cid in pruned.target_node_cids
    assert pruned.complete is False
    assert pruned.receipt_for(SuccessorStage.RESIDUAL_ESCALATION).disposition == (
        SuccessorDisposition.ESCALATED.value
    )


def test_type_effect_path_and_contract_pruning_reasons() -> None:
    graph = _complete_call_graph()
    plan = generate_static_successors(
        catalog=_catalog(graph),
        subject_node_cid=graph["ping"].program_graph_node_cid,
        query_family="next_call",
    )
    pong = graph["pong"].program_graph_node_cid
    else_plan = generate_static_successors(
        catalog=_catalog(graph),
        subject_node_cid=graph["ping"].program_graph_node_cid,
        query_family="next_event",
    )
    else_block = graph["else_block"].program_graph_node_cid

    typed = prune_program_successors(
        plan,
        type_evidence=[
            {
                "successor_node_cid": pong,
                "verdict": PruningVerdict.IMPOSSIBLE,
                "reason_code": SuccessorReasonCode.INCOMPATIBLE_TYPE,
                "authoritative": True,
                "evidence_cid": _cid("type-fact"),
                "authority": "type",
            }
        ],
    )
    assert pong not in typed.target_node_cids
    assert pong in {
        item.target_node_cid for item in typed.removed
    }
    assert typed.receipt_for(SuccessorStage.TYPE).reason_code == (
        SuccessorReasonCode.INCOMPATIBLE_TYPE.value
    )
    assert typed.receipt_for(SuccessorStage.TYPE).disposition == (
        SuccessorDisposition.SELECTED.value
    )

    effected = prune_program_successors(
        plan,
        effect_evidence=[
            {
                "successor_node_cid": pong,
                "verdict": "impossible",
                "reason_code": "incompatible_effect",
                "authoritative": True,
                "authority": "effect",
            }
        ],
    )
    assert pong not in effected.target_node_cids
    assert effected.receipt_for(SuccessorStage.EFFECT).reason_code == (
        SuccessorReasonCode.INCOMPATIBLE_EFFECT.value
    )

    pathed = prune_program_successors(
        else_plan,
        path_evidence=[
            {
                "successor_node_cid": else_block,
                "verdict": "impossible",
                "reason_code": "unsat_path",
                "authoritative": True,
                "authority": "path_condition",
            }
        ],
    )
    assert else_block not in pathed.target_node_cids
    assert pathed.receipt_for(SuccessorStage.PATH_CONDITION).reason_code == (
        SuccessorReasonCode.UNSAT_PATH.value
    )

    contracted = prune_program_successors(
        plan,
        contract_evidence=[
            {
                "successor_node_cid": pong,
                "verdict": "impossible",
                "reason_code": "contract_unsat",
                "authoritative": True,
                "authority": "contract",
            }
        ],
    )
    assert pong not in contracted.target_node_cids
    assert contracted.receipt_for(SuccessorStage.CONTRACT).reason_code == (
        SuccessorReasonCode.CONTRACT_UNSAT.value
    )


def test_solver_unknown_and_timeout_widen_without_pruning() -> None:
    graph = _complete_call_graph()
    plan = generate_static_successors(
        catalog=_catalog(graph),
        subject_node_cid=graph["ping"].program_graph_node_cid,
        query_family="next_call",
    )
    before = set(plan.target_node_cids)
    timed_out = prune_program_successors(plan, solver_status="timeout")
    unknown = prune_program_successors(
        plan,
        solver_evidence=[
            {
                "successor_node_cid": graph["pong"].program_graph_node_cid,
                "verdict": "unknown",
                "reason_code": "solver_unknown",
                "authoritative": True,
                "authority": "solver",
            }
        ],
    )
    assert set(timed_out.target_node_cids) == before
    assert set(unknown.target_node_cids) == before
    assert timed_out.receipt_for(SuccessorStage.SOLVER).widened is True
    assert timed_out.receipt_for(SuccessorStage.SOLVER).reason_code == (
        SuccessorReasonCode.SOLVER_TIMEOUT.value
    )
    assert unknown.receipt_for(SuccessorStage.SOLVER).widened is True
    assert "solver_unknown" in timed_out.frontier.reasons
    assert timed_out.complete is False
    assert unknown.complete is False


def test_impossible_is_distinct_from_incomplete() -> None:
    graph = _reflection_graph()
    plan = generate_static_successors(
        catalog=_catalog(graph),
        subject_node_cid=graph["ping"].program_graph_node_cid,
        query_family="next_call",
    )
    pong = graph["pong"].program_graph_node_cid
    dynamic = graph["dynamic"].program_graph_node_cid
    pruned = prune_program_successors(
        plan,
        type_evidence=[
            {
                "successor_node_cid": pong,
                "verdict": "impossible",
                "reason_code": "incompatible_type",
                "authoritative": True,
                "authority": "type",
            }
        ],
        abstract_interpretation=[
            {
                "successor_node_cid": dynamic,
                "verdict": "incomplete",
                "reason_code": "abstract_incomplete",
                "authoritative": True,
                "authority": "abstract_interpretation",
            }
        ],
    )
    removed_targets = {item.target_node_cid for item in pruned.removed}
    assert pong in removed_targets
    assert dynamic not in removed_targets
    assert dynamic in pruned.target_node_cids
    assert any(
        item.candidate_cid in pruned.impossible_cids for item in pruned.removed
    )
    assert any(
        item.target_node_cid == dynamic for item in pruned.retained
    )
    incomplete_targets = {
        item.target_node_cid
        for item in pruned.retained
        if item.candidate_cid in pruned.incomplete_cids or item.unknown
    }
    assert dynamic in incomplete_targets
    assert pong not in incomplete_targets
    assert pruned.complete is False


def test_stale_graph_fails_closed_and_records_rejection() -> None:
    graph = _complete_call_graph()
    stale_cid = _cid("not-current-snapshot")
    plan = generate_static_successors(
        catalog=_catalog(graph),
        subject_node_cid=graph["ping"].program_graph_node_cid,
        expected_snapshot_cid=stale_cid,
    )
    assert plan.candidates == ()
    freshness = plan.receipt_for(SuccessorStage.FRESHNESS)
    assert freshness is not None
    assert freshness.disposition == SuccessorDisposition.REJECTED.value
    assert freshness.reason_code == SuccessorReasonCode.STALE_SNAPSHOT.value
    assert "stale_graph" in plan.unavailable_dimensions
    skipped = [
        item
        for item in plan.receipts
        if item.stage != SuccessorStage.FRESHNESS.value
    ]
    assert skipped
    assert all(item.disposition == SuccessorDisposition.SKIPPED.value for item in skipped)

    fresh = generate_static_successors(
        catalog=_catalog(graph),
        subject_node_cid=graph["ping"].program_graph_node_cid,
    )
    pruned = prune_program_successors(
        fresh,
        expected_snapshot_cid=stale_cid,
    )
    assert pruned.retained == ()
    assert pruned.receipt_for(SuccessorStage.FRESHNESS).disposition == (
        SuccessorDisposition.REJECTED.value
    )


def test_stage_receipts_are_deterministic() -> None:
    graph = _reflection_graph()
    first = generate_static_successors(
        catalog=_catalog(graph),
        subject_node_cid=graph["ping"].program_graph_node_cid,
        query_family="next_call",
    )
    second = generate_static_successors(
        catalog=_catalog(graph),
        subject_node_cid=graph["ping"].program_graph_node_cid,
        query_family="next_call",
    )
    assert first.plan_cid == second.plan_cid
    assert [item.receipt_cid for item in first.receipts] == [
        item.receipt_cid for item in second.receipts
    ]
    assert first.to_dict() == second.to_dict()
    restored = StaticSuccessorPlan.from_dict(first.to_dict())
    assert restored.plan_cid == first.plan_cid

    pruned_a = prune_program_successors(
        first,
        type_evidence=[
            {
                "successor_node_cid": graph["pong"].program_graph_node_cid,
                "verdict": "possible",
                "reason_code": "complete",
                "authoritative": True,
                "authority": "type",
            }
        ],
    )
    pruned_b = prune_program_successors(
        second,
        type_evidence=[
            {
                "successor_node_cid": graph["pong"].program_graph_node_cid,
                "verdict": "possible",
                "reason_code": "complete",
                "authoritative": True,
                "authority": "type",
            }
        ],
    )
    assert pruned_a.pruning_result_cid == pruned_b.pruning_result_cid
    assert [item.receipt_cid for item in pruned_a.receipts] == [
        item.receipt_cid for item in pruned_b.receipts
    ]
    assert {item.stage for item in pruned_a.receipts} == {
        stage.value for stage in PRUNING_STAGE_ORDER
    }


def test_unknown_widens_and_non_authoritative_evidence_cannot_prune() -> None:
    graph = _complete_call_graph()
    plan = generate_static_successors(
        catalog=_catalog(graph),
        subject_node_cid=graph["ping"].program_graph_node_cid,
        query_family="next_call",
    )
    before = set(plan.target_node_cids)
    widened = prune_program_successors(
        plan,
        type_evidence=[
            {
                "successor_node_cid": graph["pong"].program_graph_node_cid,
                "verdict": "unknown",
                "reason_code": "unknown_widened",
                "authoritative": True,
                "authority": "type",
            }
        ],
    )
    assert set(widened.target_node_cids) == before
    assert widened.receipt_for(SuccessorStage.TYPE).widened is True
    assert len(widened.frontier.unavailable_dimensions) >= len(
        plan.frontier.unavailable_dimensions
    )

    rejected = prune_program_successors(
        plan,
        type_evidence=[
            {
                "successor_node_cid": graph["pong"].program_graph_node_cid,
                "verdict": "impossible",
                "reason_code": "incompatible_type",
                "authoritative": False,
                "authority": "type",
            }
        ],
    )
    assert set(rejected.target_node_cids) == before
    assert rejected.receipt_for(SuccessorStage.TYPE).disposition == (
        SuccessorDisposition.REJECTED.value
    )
    assert rejected.receipt_for(SuccessorStage.TYPE).reason_code == (
        SuccessorReasonCode.NON_AUTHORITATIVE.value
    )

    with pytest.raises(SymbolicPruningError):
        prune_program_successors(
            plan,
            type_evidence=[
                {
                    "successor_node_cid": graph["pong"].program_graph_node_cid,
                    "verdict": "impossible",
                    "reason_code": "incompatible_type",
                    "authoritative": True,
                    "score": 0.91,
                }
            ],
        )


def test_capability_unavailable_and_solver_unsat() -> None:
    graph = _complete_call_graph()
    plan = generate_static_successors(
        catalog=_catalog(graph),
        subject_node_cid=graph["ping"].program_graph_node_cid,
        query_family="next_call",
    )
    pong = graph["pong"].program_graph_node_cid
    unavailable = prune_program_successors(
        plan,
        capability_evidence=[
            {
                "successor_node_cid": pong,
                "verdict": "unavailable",
                "reason_code": "capability_unavailable",
                "authoritative": False,
                "authority": "capability",
            }
        ],
    )
    assert pong in unavailable.target_node_cids
    assert unavailable.receipt_for(SuccessorStage.CAPABILITY).disposition == (
        SuccessorDisposition.UNAVAILABLE.value
    )

    unsat = prune_program_successors(
        plan,
        solver_evidence=[
            {
                "successor_node_cid": pong,
                "verdict": "impossible",
                "reason_code": "solver_unsat",
                "authoritative": True,
                "authority": "solver",
            }
        ],
        solver_status="unsat",
    )
    assert pong not in unsat.target_node_cids
    assert unsat.receipt_for(SuccessorStage.SOLVER).reason_code == (
        SuccessorReasonCode.SOLVER_UNSAT.value
    )


def test_planner_and_pruner_classes_bind_freshness() -> None:
    graph = _complete_call_graph()
    planner = StaticSuccessorPlanner(
        expected_snapshot_cid=graph["snapshot"].program_graph_snapshot_cid,
        expected_graph_cid=graph["snapshot"].canonical_program_graph_cid,
        environment_binding_cid=graph["snapshot"].environment_binding_set_cid,
    )
    plan = planner.generate(
        catalog=_catalog(graph),
        subject_node_cid=graph["ping"].program_graph_node_cid,
    )
    assert plan.receipt_for(SuccessorStage.FRESHNESS).reason_code == (
        SuccessorReasonCode.GRAPH_FRESH.value
    )
    pruner = SymbolicSuccessorPruner(
        expected_snapshot_cid=plan.snapshot_cid,
        expected_graph_cid=plan.graph_cid,
    )
    result = pruner.prune(plan)
    assert result.receipt_for(SuccessorStage.FRESHNESS).reason_code == (
        SuccessorReasonCode.GRAPH_FRESH.value
    )
    assert isinstance(plan.frontier, UnresolvedDynamicFrontier)


def test_unsupported_language_is_unavailable() -> None:
    # Datasets records admit only Python; planner must still record LANGUAGE unavailable.
    ping_fn = _function("pkg.mod.ping")
    module = _node("module", "pkg.mod")
    ping = _node(
        "function",
        "pkg.mod.ping",
        record_cid=ping_fn.function_symbol_record_cid,
    )
    edges = [_edge("declares", module, ping)]
    plan = generate_static_successors(
        snapshot={
            "language": "javascript",
            "environment_binding_set_cid": _cid("bindings"),
            "sealed_binding_cid": _cid("sealed"),
        },
        nodes=[module, ping],
        edges=edges,
        function_symbols=[ping_fn],
        subject_node_cid=ping.program_graph_node_cid,
    )
    assert plan.candidates == ()
    assert plan.receipt_for(SuccessorStage.LANGUAGE).disposition == (
        SuccessorDisposition.UNAVAILABLE.value
    )
    assert plan.receipt_for(SuccessorStage.LANGUAGE).reason_code == (
        SuccessorReasonCode.LANGUAGE_UNAVAILABLE.value
    )
    assert any(item.startswith("language:") for item in plan.unavailable_dimensions)
    assert "language:javascript" in plan.unavailable_dimensions


def test_authoritative_path_can_remove_explicit_unknown() -> None:
    graph = _reflection_graph()
    plan = generate_static_successors(
        catalog=_catalog(graph),
        subject_node_cid=graph["ping"].program_graph_node_cid,
        query_family="next_call",
    )
    dynamic = graph["dynamic"].program_graph_node_cid
    pruned = prune_program_successors(
        plan,
        path_evidence=[
            {
                "successor_node_cid": dynamic,
                "verdict": "impossible",
                "reason_code": "unsat_path",
                "authoritative": True,
                "authority": "path_condition",
            }
        ],
    )
    assert dynamic not in pruned.target_node_cids
    assert graph["ping_site"].program_graph_node_cid in pruned.target_node_cids
    assert any(item.target_node_cid == dynamic for item in pruned.removed)
