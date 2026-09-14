"""SAWM-018 program-world Tactician and production Hammer tests."""

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
    FunctionSymbolRecord,
    ProgramGraphEdge,
    ProgramGraphNode,
    ProgramGraphSnapshot,
    ResolutionStatus,
    StaticSuccessorSet,
    assemble_program_graph_snapshot,
    DynamicFrontierRecord,
)

from ipfs_accelerate_py.agent_supervisor.analysis.program_logic_prediction_contracts import (
    CountermodelDisposition,
    ProofStatus,
    ProgramLogicAuthorityRoots,
)
from ipfs_accelerate_py.agent_supervisor.analysis.program_successors import (
    generate_static_successors,
)
from ipfs_accelerate_py.agent_supervisor.proof.program_world_hammer import (
    ITP_EXECUTABLES,
    PROGRAM_WORLD_HAMMER_INTERFACE,
    PROGRAM_WORLD_PROOF_ADMISSION_INTERFACE,
    VALIDATION_PATH,
    ProgramWorldAdmissionDisposition,
    ProgramWorldBackendStatus,
    ProgramWorldHammer,
    ProgramWorldHammerAuthorityError,
    ProgramWorldProofAdmission,
    ProgramWorldSearchOutcome,
    probe_program_world_backends,
    replay_program_world_countermodel,
    search_program_world_proof,
)
from ipfs_accelerate_py.agent_supervisor.proof.program_world_tactician import (
    PROGRAM_WORLD_PREMISE_CORPUS_INTERFACE,
    PROGRAM_WORLD_TACTICIAN_INTERFACE,
    SAWM_TACTICIAN_HAMMER_EVIDENCE,
    ProgramWorldCompilationDisposition,
    ProgramWorldPremiseCorpus,
    ProgramWorldTactician,
    ProgramWorldTacticianAuthorityError,
    compile_program_world_goals,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
TACTICIAN_PATH = (
    REPO_ROOT / "ipfs_accelerate_py/agent_supervisor/proof/program_world_tactician.py"
)
HAMMER_PATH = (
    REPO_ROOT / "ipfs_accelerate_py/agent_supervisor/proof/program_world_hammer.py"
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
        "module": module,
    }


def _reflection_graph() -> dict[str, Any]:
    ping_fn = _function("pkg.mod.ping")
    pong_fn = _function("pkg.mod.pong")
    ping_call = _callsite("pkg.mod.ping", "pkg.mod.pong")
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
        callsites=[ping_call],
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
        "callsites": [ping_call],
        "function_symbols": [ping_fn, pong_fn],
        "successor_sets": [successors],
        "frontiers": [frontier],
        "ping": ping,
        "dynamic": dynamic,
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


def _roots(plan: Any | None = None, **overrides: Any) -> ProgramLogicAuthorityRoots:
    fields: dict[str, Any] = {
        "repository_id": "repository:sawm-018",
        "objective_id": "objective:sawm-018",
        "trace_id": "trace:sawm-018",
        "change_id": "change:sawm-018",
        "consumer_id": "consumer:sawm-018",
        "forest_id": "forest:sawm-018",
        "tree_id": "tree:current",
        "overlay_id": "overlay:sawm-018",
        "graph_id": "graph:sawm-018",
        "index_id": "index:sawm-018",
        "corpus_id": "corpus:sawm-018",
        "model_id": "model:none",
        "translator_id": "translator:sawm-018",
        "toolchain_id": "toolchain:sawm-018",
        "policy_id": "policy:sawm-018",
        "environment_id": "environment:sawm-018",
    }
    if plan is not None:
        fields["graph_id"] = plan.graph_cid
    fields.update(overrides)
    return ProgramLogicAuthorityRoots(**fields)


def _plan(graph: dict[str, Any] | None = None):
    bound = graph or _complete_call_graph()
    return generate_static_successors(
        catalog=_catalog(bound),
        subject_node_cid=bound["ping"].program_graph_node_cid,
        query_family="next_call",
    )


def _compile(**overrides: Any):
    plan = overrides.pop("successor_plan", None)
    if plan is None and "goals" not in overrides and "repair_residuals" not in overrides:
        plan = _plan()
    roots = overrides.pop("roots", None) or _roots(plan)
    return compile_program_world_goals(
        roots,
        successor_plan=plan,
        expected_tree_id=overrides.pop("expected_tree_id", roots.tree_id),
        **overrides,
    )


def _names(path: Path) -> tuple[set[str], set[str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    functions = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    classes = {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}
    return functions, classes


def test_public_interfaces_and_symbols_are_present() -> None:
    tactician_fns, tactician_cls = _names(TACTICIAN_PATH)
    hammer_fns, hammer_cls = _names(HAMMER_PATH)
    assert "compile_program_world_goals" in tactician_fns
    assert "search_program_world_proof" in hammer_fns
    assert "replay_program_world_countermodel" in hammer_fns
    assert "ProgramWorldTactician" in tactician_cls
    assert "ProgramWorldPremiseCorpus" in tactician_cls
    assert "ProgramWorldHammer" in hammer_cls
    assert "ProgramWorldProofAdmission" in hammer_cls
    assert PROGRAM_WORLD_TACTICIAN_INTERFACE == "ProgramWorldTactician@1"
    assert PROGRAM_WORLD_HAMMER_INTERFACE == "ProgramWorldHammer@1"
    assert PROGRAM_WORLD_PROOF_ADMISSION_INTERFACE == "ProgramWorldProofAdmission@1"
    assert PROGRAM_WORLD_PREMISE_CORPUS_INTERFACE == "ProgramWorldPremiseCorpus@1"
    assert SAWM_TACTICIAN_HAMMER_EVIDENCE == "sawm/tactician-hammer@1"


def test_import_has_no_io_or_thread_side_effects() -> None:
    before = {thread.name for thread in threading.enumerate()}
    tactician = importlib.import_module(
        "ipfs_accelerate_py.agent_supervisor.proof.program_world_tactician"
    )
    hammer = importlib.import_module(
        "ipfs_accelerate_py.agent_supervisor.proof.program_world_hammer"
    )
    after = {thread.name for thread in threading.enumerate()}
    assert after == before
    assert inspect.isfunction(tactician.compile_program_world_goals)
    assert inspect.isfunction(hammer.search_program_world_proof)
    assert inspect.isfunction(hammer.replay_program_world_countermodel)


def test_compile_emits_finite_obligation_inventory_and_premise_corpus_cid() -> None:
    compilation = _compile()
    assert compilation.finite
    assert compilation.obligations
    assert compilation.goals
    assert compilation.premise_corpus_cid
    assert compilation.corpus.corpus_cid == compilation.premise_corpus_cid
    assert compilation.tactic_plan.plan_id
    assert compilation.tactic_plan.semantic_authority is False
    assert compilation.disposition is ProgramWorldCompilationDisposition.COMPLETE
    assert compilation.roots.tree_id == "tree:current"
    identities = {item.content_id for item in compilation.goals}
    assert len(identities) == len(compilation.goals)


def test_unknown_frontier_is_residual_or_unsupported_not_closed() -> None:
    graph = _reflection_graph()
    plan = generate_static_successors(
        catalog=_catalog(graph),
        subject_node_cid=graph["ping"].program_graph_node_cid,
        query_family="next_call",
    )
    compilation = compile_program_world_goals(_roots(plan), successor_plan=plan)
    assert compilation.disposition is ProgramWorldCompilationDisposition.PARTIAL
    assert compilation.residual_goal_ids or compilation.unsupported_goal_ids
    assert "unsupported_logic" in compilation.reason_codes or "residual_frontier" in compilation.reason_codes
    assert any(not item.complete or item.unknown for item in compilation.obligations)


def test_model_nominations_cannot_become_axioms_or_proof() -> None:
    compilation = _compile(
        nominations=[
            {
                "kind": "model",
                "premise_id": "nom:model-one",
                "statement_ref": "stmt:model-nominated",
            }
        ]
    )
    assert "nom:model-one" in compilation.corpus.nominated_premise_ids
    assert "nom:model-one" not in compilation.corpus.axiomatic_premise_ids
    assert "nom:model-one" in compilation.tactic_plan.excluded_premise_ids
    assert compilation.tactic_plan.semantic_authority is False
    with pytest.raises(ProgramWorldTacticianAuthorityError):
        compile_program_world_goals(
            _roots(),
            repair_residuals=[
                {
                    "subject_cid": "sym:ping",
                    "evidence_cid": "ev:repair",
                    "statement_ref": "stmt:repair",
                }
            ],
            nominations=[{"premise_id": "nom:bad", "statement_ref": "stmt:x", "semantic_authority": True}],
            expected_tree_id="tree:current",
        )


def test_contradictory_reviewed_premises_abstain() -> None:
    compilation = compile_program_world_goals(
        _roots(),
        repair_residuals=[
            {
                "subject_cid": "sym:ping",
                "evidence_cid": "ev:repair",
                "statement_ref": "stmt:repair",
            }
        ],
        reviewed_premises=[
            {
                "premise_id": "exp:left",
                "statement_ref": "stmt:left",
                "conflicts_with": ["exp:right"],
            },
            {
                "premise_id": "exp:right",
                "statement_ref": "stmt:right",
                "conflicts_with": ["exp:left"],
            },
        ],
        expected_tree_id="tree:current",
    )
    assert compilation.disposition is ProgramWorldCompilationDisposition.CONFLICT
    assert "contradiction" in compilation.reason_codes
    result = search_program_world_proof(
        compilation,
        reconstructor=lambda payload: {
            "kernel_accepted": True,
            "native_source_id": "native:forged",
            "kernel_id": "kernel:injected",
        },
        probe_solvers=False,
        which=lambda _name: None,
    )
    assert result.outcome is ProgramWorldSearchOutcome.ABSTAINED
    assert result.admission.disposition is ProgramWorldAdmissionDisposition.ABSTAINED
    assert not result.admission.admitted


def test_stale_tree_fails_closed() -> None:
    with pytest.raises(ProgramWorldTacticianAuthorityError):
        compile_program_world_goals(
            _roots(),
            repair_residuals=[{"subject_cid": "sym:ping", "evidence_cid": "ev:x"}],
            expected_tree_id="tree:other",
        )
    compilation = _compile()
    result = search_program_world_proof(
        compilation,
        expected_tree_id="tree:other",
        probe_solvers=False,
        which=lambda _name: None,
    )
    assert result.outcome is ProgramWorldSearchOutcome.STALE
    assert result.admission.disposition is ProgramWorldAdmissionDisposition.STALE


def test_timeout_and_unsupported_are_typed_and_not_admitted() -> None:
    compilation = _compile()
    timeout = search_program_world_proof(
        compilation,
        candidate={"outcome": "timeout"},
        probe_solvers=False,
        which=lambda _name: None,
    )
    assert timeout.outcome is ProgramWorldSearchOutcome.TIMEOUT
    assert not timeout.admission.admitted
    unsupported = search_program_world_proof(
        compilation,
        candidate={"outcome": "unsupported"},
        probe_solvers=False,
        which=lambda _name: None,
    )
    assert unsupported.outcome is ProgramWorldSearchOutcome.UNSUPPORTED
    assert unsupported.admission.proof_status is ProofStatus.UNSUPPORTED
    assert not unsupported.admission.admitted


def test_unavailability_is_typed_against_sealed_validation_path() -> None:
    probe = probe_program_world_backends(
        path=VALIDATION_PATH,
        which=lambda _name: None,
        find_spec=lambda _name: None,
        probe_solvers=False,
    )
    assert probe.kernel_status is ProgramWorldBackendStatus.UNAVAILABLE
    assert probe.reconstruction_ready is False
    assert "kernel_unavailable" in probe.reason_codes
    assert probe.to_dict()["proof_success"] is False
    compilation = _compile()
    result = search_program_world_proof(
        compilation,
        path=VALIDATION_PATH,
        which=lambda _name: None,
        find_spec=lambda _name: None,
        probe_solvers=False,
    )
    assert result.outcome is ProgramWorldSearchOutcome.UNAVAILABLE
    assert result.admission.disposition is ProgramWorldAdmissionDisposition.UNAVAILABLE
    assert not result.admission.admitted
    for executable in ITP_EXECUTABLES:
        assert f"executable:{executable}" in probe.missing


def test_model_verified_claim_is_not_proof_without_reconstruction() -> None:
    compilation = _compile()
    result = search_program_world_proof(
        compilation,
        candidate={"outcome": "verified", "model_authored": True, "solver_sat": True},
        probe_solvers=False,
        which=lambda _name: None,
    )
    assert result.outcome is not ProgramWorldSearchOutcome.VERIFIED
    assert not result.admission.admitted
    assert "model_authored_not_authority" in result.admission.reason_codes
    assert "solver_sat_not_proof" in result.admission.reason_codes
    assert result.reconstruction is not None
    assert result.reconstruction.kernel_accepted is False


def test_reconstructed_kernel_proof_is_admitted() -> None:
    compilation = _compile()

    def reconstructor(payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "kernel_accepted": True,
            "native_source_id": "native:theorem:ping-pong",
            "kernel_id": "kernel:injected-lean",
        }

    result = search_program_world_proof(
        compilation,
        candidate={"outcome": "candidate", "obligation_id": compilation.obligations[0].obligation_id},
        reconstructor=reconstructor,
        probe_solvers=False,
        which=lambda _name: None,
    )
    assert result.outcome is ProgramWorldSearchOutcome.VERIFIED
    assert result.admission.disposition is ProgramWorldAdmissionDisposition.ADMITTED_PROOF
    assert result.admission.proof_status is ProofStatus.KERNEL_VERIFIED
    assert result.admission.admitted
    assert result.reconstruction is not None
    assert result.reconstruction.kernel_accepted
    assert result.admission.reconstruction_id == result.reconstruction.reconstruction_id
    assert result.premise_corpus_cid == compilation.premise_corpus_cid
    assert result.tactic_plan_id == compilation.tactic_plan.plan_id


def test_model_nominated_candidate_is_admitted_only_after_kernel_reconstruction() -> None:
    compilation = _compile(
        nominations=[{"kind": "model", "premise_id": "nom:draft", "statement_ref": "stmt:draft"}]
    )
    rejected = search_program_world_proof(
        compilation,
        candidate={"outcome": "verified", "model_authored": True},
        probe_solvers=False,
        which=lambda _name: None,
    )
    assert not rejected.admission.admitted
    admitted = search_program_world_proof(
        compilation,
        candidate={"outcome": "candidate", "model_authored": True},
        reconstructor=lambda payload: {
            "kernel_accepted": True,
            "native_source_id": "native:reconstructed-from-nomination",
            "kernel_id": "kernel:injected",
        },
        probe_solvers=False,
        which=lambda _name: None,
    )
    assert admitted.admission.admitted
    assert admitted.admission.model_authored is False
    assert admitted.outcome is ProgramWorldSearchOutcome.VERIFIED


def test_raw_countermodel_is_diagnostic_until_replayed() -> None:
    compilation = _compile()
    diagnostic = replay_program_world_countermodel(
        compilation,
        {
            "solver_countermodel_id": "cm:raw",
            "translation_map_id": "translation-map:exact",
            "originating_logic_ir_id": "logic-ir:goal",
            "raw_diagnostic_refs": ["diag:smt-model"],
        },
    )
    assert diagnostic.validation.disposition is CountermodelDisposition.DIAGNOSTIC_ONLY
    assert not diagnostic.validated
    replayed = replay_program_world_countermodel(
        compilation,
        {
            "solver_countermodel_id": "cm:raw",
            "translation_map_id": "translation-map:exact",
            "originating_logic_ir_id": "logic-ir:goal",
            "raw_diagnostic_refs": ["diag:smt-model"],
        },
        replay=lambda payload: {
            "status": "validated",
            "replay_method": "deterministic_logic_ir_replay",
            "evidence_id": "replay:logic-ir-1",
        },
    )
    assert replayed.validated
    assert replayed.validation.disposition is CountermodelDisposition.VALIDATED
    search = search_program_world_proof(
        compilation,
        candidate={
            "outcome": "counterexample",
            "solver_countermodel_id": "cm:raw",
            "translation_map_id": "translation-map:exact",
            "originating_logic_ir_id": "logic-ir:goal",
            "raw_diagnostic_refs": ["diag:smt-model"],
        },
        replay=lambda payload: {
            "status": "validated",
            "replay_method": "deterministic_logic_ir_replay",
            "evidence_id": "replay:logic-ir-1",
        },
        probe_solvers=False,
        which=lambda _name: None,
    )
    assert search.outcome is ProgramWorldSearchOutcome.REFUTED
    assert search.admission.disposition is ProgramWorldAdmissionDisposition.ADMITTED_REFUTATION
    assert search.admission.proof_status is ProofStatus.VALIDATED_REFUTED
    assert search.admission.admitted


def test_model_countermodel_claim_is_not_truth_without_replay() -> None:
    compilation = _compile()
    result = search_program_world_proof(
        compilation,
        candidate={
            "outcome": "counterexample",
            "model_authored": True,
            "solver_countermodel_id": "cm:model",
            "translation_map_id": "translation-map:exact",
            "originating_logic_ir_id": "logic-ir:goal",
            "raw_diagnostic_refs": ["diag:model-said-so"],
        },
        probe_solvers=False,
        which=lambda _name: None,
    )
    assert result.outcome is not ProgramWorldSearchOutcome.REFUTED
    assert not result.admission.admitted
    assert result.replay is not None
    assert result.replay.validation.disposition is CountermodelDisposition.DIAGNOSTIC_ONLY


def test_proof_and_refutation_together_abstain() -> None:
    compilation = _compile()
    result = search_program_world_proof(
        compilation,
        candidate={
            "outcome": "candidate",
            "solver_countermodel_id": "cm:raw",
            "translation_map_id": "translation-map:exact",
            "originating_logic_ir_id": "logic-ir:goal",
            "raw_diagnostic_refs": ["diag:smt-model"],
            "proof_candidate": {"obligation_id": compilation.obligations[0].obligation_id},
        },
        reconstructor=lambda payload: {
            "kernel_accepted": True,
            "native_source_id": "native:ok",
            "kernel_id": "kernel:injected",
        },
        replay=lambda payload: {
            "status": "validated",
            "replay_method": "deterministic_logic_ir_replay",
            "evidence_id": "replay:both",
        },
        probe_solvers=False,
        which=lambda _name: None,
    )
    assert result.outcome is ProgramWorldSearchOutcome.ABSTAINED
    assert result.admission.disposition is ProgramWorldAdmissionDisposition.ABSTAINED
    assert "contradiction" in result.admission.reason_codes


def test_program_world_hammer_facade_and_admission_contract() -> None:
    compilation = _compile()
    hammer = ProgramWorldHammer(
        reconstructor=lambda payload: {
            "kernel_accepted": True,
            "native_source_id": "native:facade",
            "kernel_id": "kernel:injected",
        },
        which=lambda _name: None,
        probe_solvers=False,
    )
    result = hammer.search(compilation, candidate={"outcome": "candidate"})
    assert isinstance(result.admission, ProgramWorldProofAdmission)
    assert result.admission.admitted
    probe = hammer.probe_backends()
    assert probe.kernel_status is ProgramWorldBackendStatus.UNAVAILABLE
    with pytest.raises(ProgramWorldHammerAuthorityError):
        ProgramWorldProofAdmission(
            admission_id="admission:forged",
            disposition=ProgramWorldAdmissionDisposition.ADMITTED_PROOF,
            tree_id="tree:current",
            compilation_id=compilation.compilation_id,
            proof_status=ProofStatus.CANDIDATE,
            reconstruction_id="recon:missing-kernel",
        )


def test_tactician_class_compile_matches_function() -> None:
    plan = _plan()
    roots = _roots(plan)
    via_class = ProgramWorldTactician(roots).compile(successor_plan=plan)
    via_fn = compile_program_world_goals(roots, successor_plan=plan)
    assert via_class.disposition is via_fn.disposition
    assert via_class.premise_corpus_cid == via_fn.premise_corpus_cid
    assert isinstance(via_class.corpus, ProgramWorldPremiseCorpus)
    assert via_class.tactic_plan.semantic_authority is False


def test_repair_residual_and_empty_inventory() -> None:
    empty = compile_program_world_goals(_roots(), expected_tree_id="tree:current")
    assert empty.disposition is ProgramWorldCompilationDisposition.ABSTAINED
    assert empty.goals
    repair = compile_program_world_goals(
        _roots(),
        repair_residuals=[
            {
                "subject_cid": "sym:ping",
                "evidence_cid": "ev:repair",
                "unsupported": True,
                "unavailable_dimensions": ["native"],
            }
        ],
        expected_tree_id="tree:current",
    )
    assert repair.disposition is ProgramWorldCompilationDisposition.PARTIAL
    assert repair.unsupported_goal_ids or repair.residual_goal_ids
