"""SAWM-017 static successor generation and symbolic pruning tests."""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from ipfs_datasets_py.logic.software_contracts.content import cid_for_bytes
from ipfs_datasets_py.logic.software_contracts.semantic_state.program_graph import (
    DynamicFrontierReason,
)
from ipfs_datasets_py.logic.software_verification.program_graph_builder import (
    build_program_graph,
)

from ipfs_accelerate_py.agent_supervisor.analysis.program_successors import (
    GENERATION_STAGE_ORDER,
    STATIC_SUCCESSOR_PLANNER_INTERFACE,
    STATIC_SUCCESSORS_EVIDENCE,
    SUCCESSOR_DECISION_RECEIPT_INTERFACE,
    UNRESOLVED_DYNAMIC_FRONTIER_INTERFACE,
    CompletenessVerdict,
    FreshnessState,
    StageStatus,
    StaticSuccessorError,
    StaticSuccessorPlanner,
    SuccessorDecisionReceipt,
    SuccessorKind,
    UnresolvedDynamicFrontier,
    generate_static_successors,
)
from ipfs_accelerate_py.agent_supervisor.analysis.program_symbolic_pruning import (
    PRUNE_STAGE_ORDER,
    SYMBOLIC_PRUNING_EVIDENCE,
    SYMBOLIC_SUCCESSOR_PRUNER_INTERFACE,
    AbstractConstraint,
    CapabilityConstraint,
    ContractConstraint,
    EffectConstraint,
    NeuralNomination,
    PathConstraint,
    PruneReason,
    SolverObligation,
    SymbolicPruningEvidence,
    SymbolicSuccessorPruner,
    TypeConstraint,
    prune_program_successors,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
SUCCESSORS_PATH = (
    REPO_ROOT / "ipfs_accelerate_py/agent_supervisor/analysis/program_successors.py"
)
PRUNING_PATH = (
    REPO_ROOT / "ipfs_accelerate_py/agent_supervisor/analysis/program_symbolic_pruning.py"
)
_OPT_OUTS = {
    "IPFS_DATASETS_AUTO_INSTALL": "0",
    "IPFS_DATASETS_AUTO_INSTALL_TEST_DEPS": "0",
    "IPFS_DATASETS_PY_MINIMAL_IMPORTS": "1",
    "IPFS_KIT_AUTO_INSTALL_DEPS": "0",
    "PYTHONDONTWRITEBYTECODE": "1",
}

PING_PONG = """
def ping():
    return pong()

def pong():
    return ping()
"""
WRAP = "from pkg.a import ping\n\ndef wrap():\n    return ping()\n"
DYNAMIC = """
def hidden(name):
    return getattr(hidden, name)

def boom(code):
    return eval(code)
"""
TYPED = """
def add(x: int, y: int) -> int:
    assert x >= 0
    if y:
        z = x + y
    else:
        z = x
    return z
"""
EXC = """
class Error(Exception):
    pass

def run():
    try:
        raise Error()
    except Error:
        return 1
"""
NATIVE = """
import ctypes

def load():
    return ctypes.CDLL("libc.so.6")
"""
IO = """
def spill(path):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("x")
"""


def _cid(label: str) -> str:
    return cid_for_bytes(label.encode("utf-8"))


def _tree(**overrides: str) -> dict[str, str]:
    sources = {
        "pkg/__init__.py": "",
        "pkg/a.py": PING_PONG,
        "pkg/b.py": WRAP,
        "pkg/dyn.py": DYNAMIC,
        "pkg/typed.py": TYPED,
        "pkg/exc.py": EXC,
        "pkg/native.py": NATIVE,
        "pkg/io.py": IO,
    }
    sources.update(overrides)
    return sources


def _build(**overrides: str):
    return build_program_graph(_tree(**overrides))


def _names(plan, catalog) -> set[str]:
    found = set()
    for candidate in plan.candidates:
        try:
            found.add(catalog.node(candidate.successor_node_cid).logical_name)
        except StaticSuccessorError:
            continue
    return found


def _known(plan):
    return tuple(item for item in plan.candidates if not item.unknown)


# ---------------------------------------------------------------------------
# Interfaces / symbols / authority boundary
# ---------------------------------------------------------------------------


def test_public_interfaces_and_predicted_symbols() -> None:
    import ipfs_accelerate_py.agent_supervisor.analysis.program_successors as successors
    import ipfs_accelerate_py.agent_supervisor.analysis.program_symbolic_pruning as pruning

    assert STATIC_SUCCESSOR_PLANNER_INTERFACE == "StaticSuccessorPlanner@1"
    assert SUCCESSOR_DECISION_RECEIPT_INTERFACE == "SuccessorDecisionReceipt@1"
    assert UNRESOLVED_DYNAMIC_FRONTIER_INTERFACE == "UnresolvedDynamicFrontier@1"
    assert SYMBOLIC_SUCCESSOR_PRUNER_INTERFACE == "SymbolicSuccessorPruner@1"
    assert STATIC_SUCCESSORS_EVIDENCE == "sawm/static-successors@1"
    assert SYMBOLIC_PRUNING_EVIDENCE == "sawm/symbolic-pruning@1"
    predicted = {
        "StaticSuccessorPlanner",
        "SymbolicSuccessorPruner",
        "UnresolvedDynamicFrontier",
        "SuccessorDecisionReceipt",
        "generate_static_successors",
        "prune_program_successors",
    }
    names = set(successors.__all__) | set(pruning.__all__)
    assert predicted <= names
    assert callable(generate_static_successors)
    assert callable(prune_program_successors)


def test_modules_do_not_introduce_a_second_scanner_solver_or_router() -> None:
    banned = {
        "SMTSolver",
        "Z3Solver",
        "ProgramGraphBuilder",
        "ContextCompiler",
        "AdaptivePlanner",
        "MultiProverRouter",
    }
    for path in (SUCCESSORS_PATH, PRUNING_PATH):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        defined = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        }
        assert not (defined & banned)
        parse_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "parse"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "ast"
        ]
        assert parse_calls == []
        source = path.read_text(encoding="utf-8")
        assert "build_program_graph(" not in source
        assert "subprocess" not in source
        assert "socket" not in source


def test_cold_import_does_not_scan_or_mutate() -> None:
    script = r"""
import json
import os
import sys
import threading

effects = []

def forbidden(name):
    def call(*args, **kwargs):
        effects.append(name)
        raise AssertionError(f"forbidden import side effect: {name}")
    return call

os.system = forbidden("os.system")
_orig_start = threading.Thread.start

def _thread_start(self, *args, **kwargs):
    effects.append("threading.Thread.start")
    raise AssertionError("forbidden import side effect: threading.Thread.start")

threading.Thread.start = _thread_start
import ipfs_accelerate_py.agent_supervisor.analysis.program_successors as successors
import ipfs_accelerate_py.agent_supervisor.analysis.program_symbolic_pruning as pruning
print(json.dumps({
    "effects": effects,
    "scan": getattr(successors, "IMPORT_SCAN_PERFORMED", False),
    "planner": successors.STATIC_SUCCESSOR_PLANNER_INTERFACE,
    "pruner": pruning.SYMBOLIC_SUCCESSOR_PRUNER_INTERFACE,
}))
"""
    env = dict(os.environ)
    env.update(_OPT_OUTS)
    env["PYTHONPATH"] = "ipfs_datasets_py:ipfs_kit_py:."
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(REPO_ROOT),
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["effects"] == []
    assert payload["scan"] is False
    assert payload["planner"] == "StaticSuccessorPlanner@1"
    assert payload["pruner"] == "SymbolicSuccessorPruner@1"


# ---------------------------------------------------------------------------
# Static recall
# ---------------------------------------------------------------------------


def test_static_recall_preserves_direct_and_mutual_call_successors() -> None:
    receipt = _build()
    plan = generate_static_successors(
        receipt,
        subject_logical_name="pkg.a.ping",
        query_family="next_call",
        expected_snapshot_cid=receipt.snapshot.program_graph_snapshot_cid,
        expected_environment_binding_cid=receipt.environment_binding_cid,
        expected_source_cids=dict(receipt.source_cids),
    )
    names = _names(plan, plan.catalog)
    assert any("pong" in name or "ping#" in name for name in names)
    assert not any("boom" in name or "hidden" in name for name in names)
    assert any(item.kind == SuccessorKind.CALL_TARGET.value for item in plan.candidates)
    wrap = generate_static_successors(
        receipt, subject_logical_name="pkg.b.wrap", query_family="next_call"
    )
    wrap_names = _names(wrap, wrap.catalog)
    assert any("ping" in name for name in wrap_names)
    assert plan.receipt.freshness == FreshnessState.FRESH.value
    assert plan.completeness in {item.value for item in CompletenessVerdict}


def test_event_successors_include_raises_and_effects() -> None:
    receipt = _build()
    plan = generate_static_successors(
        receipt, subject_logical_name="pkg.exc.run", query_family="next_event"
    )
    kinds = {item.kind for item in plan.candidates}
    names = _names(plan, plan.catalog)
    assert SuccessorKind.NEXT_EVENT.value in kinds or any(
        "raise" in name or "except" in name or "effect" in name for name in names
    )
    assert plan.receipt.stage_map()["call_successors"].status == StageStatus.SKIPPED.value
    assert plan.receipt.stage_map()["event_successors"].status in {
        StageStatus.SELECTED.value,
        StageStatus.SKIPPED.value,
    }
    io_plan = generate_static_successors(
        receipt, subject_logical_name="pkg.io.spill", query_family="next_event"
    )
    io_names = _names(io_plan, io_plan.catalog)
    assert io_plan.candidates
    assert any("write" in name or "effect" in name or "open" in name for name in io_names) or any(
        item.kind == SuccessorKind.NEXT_EVENT.value for item in io_plan.candidates
    )


def test_cfg_successors_are_recalled_for_next_call() -> None:
    receipt = _build()
    plan = generate_static_successors(
        receipt, subject_logical_name="pkg.typed.add", query_family="next_call"
    )
    assert plan.receipt.stage_map()["cfg_successors"].status in {
        StageStatus.SELECTED.value,
        StageStatus.SKIPPED.value,
    }
    assert any(
        item.kind in {SuccessorKind.CFG_SUCCESSOR.value, SuccessorKind.CALL_TARGET.value}
        for item in plan.candidates
    ) or plan.frontier.open


def test_generation_is_deterministic() -> None:
    receipt = _build()
    first = generate_static_successors(receipt, subject_logical_name="pkg.a.ping")
    second = generate_static_successors(receipt, subject_logical_name="pkg.a.ping")
    assert first.plan_cid == second.plan_cid
    assert first.receipt.receipt_cid == second.receipt.receipt_cid
    assert first.frontier.frontier_cid == second.frontier.frontier_cid
    assert [item.candidate_cid for item in first.candidates] == [
        item.candidate_cid for item in second.candidates
    ]


def test_every_generation_stage_is_recorded() -> None:
    receipt = _build()
    plan = generate_static_successors(receipt, subject_logical_name="pkg.a.ping")
    recorded = [item.stage for item in plan.receipt.stages]
    assert recorded == list(GENERATION_STAGE_ORDER)
    allowed = {item.value for item in StageStatus}
    assert {item.status for item in plan.receipt.stages} <= allowed


def test_unsupported_query_family_is_rejected_not_invented() -> None:
    receipt = _build()
    plan = generate_static_successors(
        receipt, subject_logical_name="pkg.a.ping", query_family="repair"
    )
    assert plan.candidates == ()
    assert plan.receipt.stage_map()["subject_resolution"].status == StageStatus.REJECTED.value
    assert plan.completeness == CompletenessVerdict.INCOMPLETE.value


# ---------------------------------------------------------------------------
# Dynamic / reflection unknown frontier
# ---------------------------------------------------------------------------


def test_dynamic_and_reflection_unknown_are_explicit_and_preserved() -> None:
    receipt = _build()
    boom = generate_static_successors(receipt, subject_logical_name="pkg.dyn.boom")
    hidden = generate_static_successors(receipt, subject_logical_name="pkg.dyn.hidden")
    assert boom.frontier.open
    assert hidden.frontier.open
    reasons = set(boom.frontier.reasons) | set(hidden.frontier.reasons)
    dims = set(boom.frontier.unavailable_dimensions) | set(hidden.frontier.unavailable_dimensions)
    assert reasons & {
        DynamicFrontierReason.EVAL.value,
        DynamicFrontierReason.REFLECTION.value,
        DynamicFrontierReason.UNKNOWN_CALLEE.value,
        DynamicFrontierReason.INCOMPLETE_ANALYSIS.value,
    } or dims
    assert boom.unknown_candidates or boom.frontier.unresolved_node_cids
    pruned = prune_program_successors(boom)
    assert pruned.frontier.open
    unknown_ids = {item.candidate_cid for item in boom.unknown_candidates}
    retained_unknown = {item.candidate_cid for item in pruned.retained if item.unknown}
    assert unknown_ids <= retained_unknown | set(pruned.frontier.unresolved_node_cids)
    assert pruned.completeness != CompletenessVerdict.COMPLETE.value


def test_native_calls_remain_on_the_frontier() -> None:
    receipt = _build()
    plan = generate_static_successors(receipt, subject_logical_name="pkg.native.load")
    assert plan.frontier.open or plan.completeness == CompletenessVerdict.INCOMPLETE.value
    joined = " ".join(plan.frontier.unavailable_dimensions + plan.frontier.reasons)
    assert "native" in joined or "effect" in joined or plan.frontier.open


def test_unresolved_dynamic_frontier_widens_and_rejects_illegal_reasons() -> None:
    base = UnresolvedDynamicFrontier(
        unresolved_node_cids=(_cid("n1"),),
        reasons=(DynamicFrontierReason.REFLECTION.value,),
        unavailable_dimensions=("reflection",),
    )
    widened = base.union(
        reasons=(DynamicFrontierReason.PLUGIN.value,),
        unavailable_dimensions=("plugin",),
    )
    assert widened.widened is True
    assert DynamicFrontierReason.PLUGIN.value in widened.reasons
    assert DynamicFrontierReason.REFLECTION.value in widened.reasons
    with pytest.raises(StaticSuccessorError, match="DynamicFrontierReason"):
        UnresolvedDynamicFrontier(reasons=("made_up_reason",), unavailable_dimensions=("x",))


# ---------------------------------------------------------------------------
# Freshness / stale graph
# ---------------------------------------------------------------------------


def test_stale_snapshot_rejects_generation_without_inventing_successors() -> None:
    receipt = _build()
    plan = generate_static_successors(
        receipt,
        subject_logical_name="pkg.a.ping",
        expected_snapshot_cid=_cid("other-snapshot"),
    )
    assert plan.candidates == ()
    assert plan.freshness == FreshnessState.STALE.value
    assert plan.receipt.stage_map()["snapshot_freshness"].status == StageStatus.REJECTED.value
    assert plan.completeness == CompletenessVerdict.INCOMPLETE.value


def test_stale_source_binding_rejects_generation() -> None:
    receipt = _build()
    drifted = dict(receipt.source_cids)
    first_path = next(iter(drifted))
    drifted[first_path] = _cid("drifted-bytes")
    plan = generate_static_successors(
        receipt,
        subject_logical_name="pkg.a.ping",
        expected_source_cids=drifted,
    )
    assert plan.candidates == ()
    assert plan.receipt.stage_map()["snapshot_freshness"].reason_code == "stale_source"


def test_stale_graph_refuses_pruning_and_preserves_candidates() -> None:
    receipt = _build()
    plan = generate_static_successors(receipt, subject_logical_name="pkg.a.ping")
    result = prune_program_successors(
        plan, expected_snapshot_cid=_cid("not-the-snapshot")
    )
    assert result.retained == plan.candidates
    assert result.pruned == ()
    assert result.receipt.stage_map()["snapshot_freshness"].status == StageStatus.REJECTED.value
    assert result.completeness == CompletenessVerdict.INCOMPLETE.value
    assert "stale_graph" in result.frontier.unavailable_dimensions


# ---------------------------------------------------------------------------
# Symbolic pruning
# ---------------------------------------------------------------------------


def test_absent_authoritative_evidence_preserves_static_possibilities() -> None:
    receipt = _build()
    plan = generate_static_successors(receipt, subject_logical_name="pkg.a.ping")
    result = prune_program_successors(plan)
    retained = {item.candidate_cid for item in result.retained}
    original = {item.candidate_cid for item in plan.candidates}
    assert original <= retained
    assert result.pruned == ()
    recorded = [item.stage for item in result.receipt.stages]
    assert recorded == list(PRUNE_STAGE_ORDER)


def test_type_contradiction_prunes_only_with_authoritative_evidence() -> None:
    receipt = _build()
    plan = generate_static_successors(receipt, subject_logical_name="pkg.typed.add")
    known = _known(plan)
    assert known
    target = known[0]
    ignored = prune_program_successors(
        plan,
        evidence=SymbolicPruningEvidence(
            type_facts=(
                TypeConstraint(
                    successor_node_cid=target.successor_node_cid,
                    observed_annotation="int",
                    required_annotation="str",
                    contradiction=True,
                    authoritative=False,
                ),
            )
        ),
    )
    assert target.candidate_cid in {item.candidate_cid for item in ignored.retained}
    assert ignored.receipt.stage_map()["type"].status == StageStatus.REJECTED.value
    applied = prune_program_successors(
        plan,
        evidence=SymbolicPruningEvidence(
            type_facts=(
                TypeConstraint(
                    successor_node_cid=target.successor_node_cid,
                    observed_annotation="int",
                    required_annotation="str",
                    contradiction=True,
                    authoritative=True,
                    evidence_cid=_cid("type-proof"),
                ),
            )
        ),
    )
    assert target.candidate_cid in {item.candidate_cid for item in applied.pruned}
    assert applied.pruned[0].reason == PruneReason.TYPE_CONTRADICTION.value
    assert applied.receipt.stage_map()["type"].status == StageStatus.SELECTED.value


def test_effect_and_capability_pruning_and_unavailable_capability_widens() -> None:
    receipt = _build()
    plan = generate_static_successors(
        receipt, subject_logical_name="pkg.io.spill", query_family="next_event"
    )
    known = _known(plan) or plan.candidates
    assert known
    target = known[0]
    denied = prune_program_successors(
        plan,
        evidence=SymbolicPruningEvidence(
            effect_facts=(
                EffectConstraint(
                    successor_node_cid=target.successor_node_cid,
                    effect="io",
                    forbidden=True,
                    definite=True,
                    authoritative=True,
                    evidence_cid=_cid("effect-proof"),
                ),
            ),
            capability_facts=(
                CapabilityConstraint(
                    effect="io",
                    allowed=False,
                    available=True,
                    authoritative=True,
                    evidence_cid=_cid("cap-proof"),
                ),
            ),
        ),
    )
    assert denied.pruned
    assert {item.reason for item in denied.pruned} <= {
        PruneReason.FORBIDDEN_EFFECT.value,
        PruneReason.CAPABILITY_DENIED.value,
    }
    widened = prune_program_successors(
        plan,
        evidence=SymbolicPruningEvidence(
            capability_facts=(
                CapabilityConstraint(
                    effect="network",
                    allowed=False,
                    available=False,
                    authoritative=True,
                    evidence_cid=_cid("cap-missing"),
                ),
            )
        ),
    )
    assert widened.receipt.stage_map()["capability"].status == StageStatus.UNAVAILABLE.value
    assert "capability" in widened.frontier.unavailable_dimensions
    assert widened.frontier.widened is True


def test_path_and_contract_pruning_reasons() -> None:
    receipt = _build()
    plan = generate_static_successors(receipt, subject_logical_name="pkg.typed.add")
    known = _known(plan)
    assert known
    target = known[0]
    path_result = prune_program_successors(
        plan,
        evidence=SymbolicPruningEvidence(
            path_facts=(
                PathConstraint(
                    successor_edge_cid=target.successor_edge_cid,
                    satisfiable=False,
                    authoritative=True,
                    evidence_cid=_cid("path-unsat"),
                    condition="y == 0 is false on this edge",
                ),
            )
        ),
    )
    assert target.candidate_cid in {item.candidate_cid for item in path_result.pruned}
    assert path_result.pruned[0].reason == PruneReason.UNSATISFIABLE_PATH.value
    unknown_path = prune_program_successors(
        plan,
        evidence=SymbolicPruningEvidence(
            path_facts=(
                PathConstraint(
                    successor_edge_cid=target.successor_edge_cid,
                    satisfiable=None,
                    authoritative=False,
                ),
            )
        ),
    )
    assert unknown_path.receipt.stage_map()["path_condition"].status == StageStatus.ESCALATED.value
    assert DynamicFrontierReason.SOLVER_UNKNOWN.value in unknown_path.frontier.reasons
    contract_result = prune_program_successors(
        plan,
        evidence=SymbolicPruningEvidence(
            contract_facts=(
                ContractConstraint(
                    successor_node_cid=target.subject_node_cid,
                    discharge_status="violated",
                    contract_kind="precondition",
                    authoritative=True,
                    evidence_cid=_cid("contract-violated"),
                    specification_cid=_cid("spec"),
                ),
            )
        ),
    )
    assert contract_result.pruned
    assert all(item.reason == PruneReason.VIOLATED_CONTRACT.value for item in contract_result.pruned)
    unknown_contract = prune_program_successors(
        plan,
        evidence=SymbolicPruningEvidence(
            contract_facts=(
                ContractConstraint(
                    successor_node_cid=target.subject_node_cid,
                    discharge_status="unknown",
                    contract_kind="precondition",
                    authoritative=False,
                ),
            )
        ),
    )
    assert unknown_contract.pruned == ()
    assert unknown_contract.receipt.stage_map()["contract"].status == StageStatus.UNAVAILABLE.value


def test_solver_unknown_and_timeout_widen_and_do_not_prune() -> None:
    receipt = _build()
    plan = generate_static_successors(receipt, subject_logical_name="pkg.a.ping")
    original = {item.candidate_cid for item in plan.candidates}
    unknown = prune_program_successors(
        plan,
        evidence=SymbolicPruningEvidence(
            solver_facts=(
                SolverObligation(
                    target_cid=plan.subject_node_cid or plan.snapshot_cid,
                    verdict="unknown",
                    authoritative=False,
                ),
            )
        ),
    )
    assert {item.candidate_cid for item in unknown.retained} == original
    assert unknown.pruned == ()
    assert unknown.receipt.stage_map()["solver"].status == StageStatus.ESCALATED.value
    assert DynamicFrontierReason.SOLVER_UNKNOWN.value in unknown.frontier.reasons
    timeout = prune_program_successors(
        plan,
        evidence=SymbolicPruningEvidence(
            solver_facts=(
                SolverObligation(
                    target_cid=plan.subject_node_cid or plan.snapshot_cid,
                    verdict="timeout",
                    timeout=True,
                    authoritative=False,
                ),
            )
        ),
    )
    assert timeout.pruned == ()
    assert timeout.completeness == CompletenessVerdict.INCOMPLETE.value
    assert timeout.frontier.widened is True


def test_solver_unsat_can_prove_impossibility_and_remove_unknown() -> None:
    receipt = _build()
    plan = generate_static_successors(receipt, subject_logical_name="pkg.a.ping")
    obligation = _cid("unsat-obligation")
    result = prune_program_successors(
        plan,
        evidence=SymbolicPruningEvidence(
            solver_facts=(
                SolverObligation(
                    target_cid=plan.subject_node_cid or plan.snapshot_cid,
                    verdict="unsat",
                    authoritative=True,
                    obligation_cid=obligation,
                    covers_frontier=True,
                ),
            )
        ),
    )
    assert result.retained == ()
    assert result.pruned
    assert all(item.reason == PruneReason.SOLVER_UNSAT.value for item in result.pruned)
    assert result.frontier.open is False
    assert result.completeness == CompletenessVerdict.IMPOSSIBLE.value


def test_impossible_versus_incomplete_are_distinct() -> None:
    receipt = _build()
    plan = generate_static_successors(receipt, subject_logical_name="pkg.dyn.boom")
    incomplete = prune_program_successors(plan)
    assert incomplete.completeness == CompletenessVerdict.INCOMPLETE.value
    known = generate_static_successors(receipt, subject_logical_name="pkg.a.ping")
    target = _known(known)[0]
    partial = prune_program_successors(
        known,
        evidence=SymbolicPruningEvidence(
            type_facts=(
                TypeConstraint(
                    successor_node_cid=target.successor_node_cid,
                    observed_annotation="int",
                    required_annotation="str",
                    contradiction=True,
                    authoritative=True,
                    evidence_cid=_cid("one-type"),
                ),
            )
        ),
    )
    if partial.retained or partial.frontier.open:
        assert partial.completeness != CompletenessVerdict.IMPOSSIBLE.value


def test_abstract_interpretation_contradiction_versus_unknown() -> None:
    receipt = _build()
    plan = generate_static_successors(receipt, subject_logical_name="pkg.typed.add")
    known = _known(plan)
    assert known
    target = known[0]
    contradiction = prune_program_successors(
        plan,
        evidence=SymbolicPruningEvidence(
            abstract_facts=(
                AbstractConstraint(
                    successor_node_cid=target.successor_node_cid,
                    domain="nullness",
                    contradiction=True,
                    authoritative=True,
                    evidence_cid=_cid("abs-unsat"),
                    value="bottom",
                ),
            )
        ),
    )
    assert target.candidate_cid in {item.candidate_cid for item in contradiction.pruned}
    unknown = prune_program_successors(
        plan,
        evidence=SymbolicPruningEvidence(
            abstract_facts=(
                AbstractConstraint(
                    successor_node_cid=target.successor_node_cid,
                    domain="interval",
                    unknown=True,
                    authoritative=False,
                    value="top",
                ),
            )
        ),
    )
    assert unknown.pruned == ()
    assert unknown.receipt.stage_map()["abstract_interpretation"].status == StageStatus.ESCALATED.value
    assert "abstract_interpretation" in unknown.frontier.unavailable_dimensions


def test_neural_nominations_cannot_prune() -> None:
    receipt = _build()
    plan = generate_static_successors(receipt, subject_logical_name="pkg.a.ping")
    known = _known(plan)
    assert known
    result = prune_program_successors(
        plan,
        evidence=SymbolicPruningEvidence(
            neural_nominations=(
                NeuralNomination(candidate_cid=known[0].candidate_cid, millipercent=99000),
            )
        ),
    )
    assert result.pruned == ()
    assert {item.candidate_cid for item in result.retained} == {
        item.candidate_cid for item in plan.candidates
    }
    assert result.receipt.stage_map()["neural_nomination"].status == StageStatus.REJECTED.value


def test_unknown_widens_across_pruning_stages() -> None:
    receipt = _build()
    plan = generate_static_successors(receipt, subject_logical_name="pkg.dyn.hidden")
    before = set(plan.frontier.reasons)
    result = prune_program_successors(
        plan,
        evidence=SymbolicPruningEvidence(
            widen_reasons=(DynamicFrontierReason.PLUGIN.value,),
            widen_dimensions=("plugin",),
            solver_facts=(
                SolverObligation(
                    target_cid=plan.snapshot_cid,
                    verdict="timeout",
                    timeout=True,
                ),
            ),
        ),
    )
    after = set(result.frontier.reasons)
    assert before <= after
    assert DynamicFrontierReason.PLUGIN.value in after
    assert DynamicFrontierReason.SOLVER_UNKNOWN.value in after
    assert result.frontier.widened is True


def test_pruning_receipts_are_deterministic() -> None:
    receipt = _build()
    plan = generate_static_successors(receipt, subject_logical_name="pkg.typed.add")
    known = _known(plan)[0]
    evidence = SymbolicPruningEvidence(
        type_facts=(
            TypeConstraint(
                successor_node_cid=known.successor_node_cid,
                observed_annotation="int",
                required_annotation="str",
                contradiction=True,
                authoritative=True,
                evidence_cid=_cid("det-type"),
            ),
        )
    )
    first = prune_program_successors(plan, evidence=evidence)
    second = prune_program_successors(plan, evidence=evidence)
    assert first.result_cid == second.result_cid
    assert first.receipt.receipt_cid == second.receipt.receipt_cid
    assert first.receipt.parent_receipt_cid == plan.receipt.receipt_cid


def test_every_prune_stage_is_recorded() -> None:
    receipt = _build()
    plan = generate_static_successors(receipt, subject_logical_name="pkg.a.ping")
    result = prune_program_successors(plan)
    assert [item.stage for item in result.receipt.stages] == list(PRUNE_STAGE_ORDER)
    allowed = {item.value for item in StageStatus}
    assert {item.status for item in result.receipt.stages} <= allowed


def test_planner_and_pruner_classes_are_usable_directly() -> None:
    receipt = _build()
    planner = StaticSuccessorPlanner()
    pruner = SymbolicSuccessorPruner()
    plan = planner.generate(receipt, subject_logical_name="pkg.a.pong")
    result = pruner.prune(plan)
    assert plan.subject_node_cid is not None
    assert result.receipt.evidence_term == SYMBOLIC_PRUNING_EVIDENCE
    assert plan.receipt.evidence_term == STATIC_SUCCESSORS_EVIDENCE
    assert isinstance(plan.receipt, SuccessorDecisionReceipt)
    assert isinstance(plan.frontier, UnresolvedDynamicFrontier)


def test_missing_subject_is_rejected() -> None:
    receipt = _build()
    plan = generate_static_successors(receipt, subject_logical_name="pkg.missing.fn")
    assert plan.candidates == ()
    assert plan.receipt.stage_map()["subject_resolution"].status == StageStatus.REJECTED.value


def test_catalog_snapshot_alone_is_insufficient() -> None:
    receipt = _build()
    with pytest.raises(StaticSuccessorError, match="insufficient"):
        generate_static_successors(receipt.snapshot, subject_logical_name="pkg.a.ping")
