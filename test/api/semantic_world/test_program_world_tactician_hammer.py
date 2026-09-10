"""SAWM-018 Tactician and production Hammer integration tests."""

from __future__ import annotations

import ast
import importlib
import inspect
import threading
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.program_logic_prediction_contracts import (
    CountermodelDisposition,
    ProgramLogicAuthorityRoots,
    ProofStatus,
)
from ipfs_accelerate_py.agent_supervisor.analysis.program_logic_premise_corpus import (
    PremiseSourceClass,
)
from ipfs_accelerate_py.agent_supervisor.proof.program_world_hammer import (
    PROGRAM_WORLD_HAMMER_INTERFACE,
    PROGRAM_WORLD_PROOF_ADMISSION_INTERFACE,
    ProgramWorldCandidateKind,
    ProgramWorldHammer,
    ProgramWorldHammerCandidate,
    ProgramWorldProofAdmission,
    ProgramWorldProofDisposition,
    ProgramWorldReconstructionReceipt,
    ProgramWorldReconstructionStatus,
    replay_program_world_countermodel,
    search_program_world_proof,
)
from ipfs_accelerate_py.agent_supervisor.proof.program_world_tactician import (
    PROGRAM_WORLD_PREMISE_CORPUS_INTERFACE,
    PROGRAM_WORLD_PROOF_SEARCH_INTERFACE,
    PROGRAM_WORLD_TACTICIAN_INTERFACE,
    SAWM_TACTICIAN_HAMMER_EVIDENCE,
    ProgramWorldCapabilityKind,
    ProgramWorldCompilationDisposition,
    ProgramWorldObligation,
    ProgramWorldObligationKind,
    ProgramWorldPolarity,
    ProgramWorldPremiseCorpus,
    ProgramWorldPremiseNomination,
    ProgramWorldReasonCode,
    ProgramWorldStaleBindingError,
    ProgramWorldTactician,
    compile_program_world_goals,
    probe_program_world_proof_capabilities,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
TACTICIAN_PATH = (
    REPO_ROOT / "ipfs_accelerate_py/agent_supervisor/proof/program_world_tactician.py"
)
HAMMER_PATH = (
    REPO_ROOT / "ipfs_accelerate_py/agent_supervisor/proof/program_world_hammer.py"
)

_BANNED_CLASS_DEFS = {
    "LogicTactician",
    "HammerBackend",
    "GoalDirectedProofTactician",
    "TacticianHammerCoordinator",
    "MultiProverRouter",
    "IsolatedHammerLoader",
    "FormalVerificationCache",
    "CorpusManifest",
    "ProofTactician",
}


def _roots(**overrides: Any) -> ProgramLogicAuthorityRoots:
    fields: dict[str, Any] = {
        "repository_id": "repo:sawm-018",
        "objective_id": "SAWM-018",
        "trace_id": "trace:sawm-018",
        "change_id": "change:sawm-018",
        "consumer_id": "consumer:program-world",
        "forest_id": "forest:sawm-018",
        "tree_id": "tree:current-018",
        "overlay_id": "overlay:none",
        "graph_id": "graph:current-018",
        "index_id": "index:current-018",
        "corpus_id": "corpus:current-018",
        "model_id": "model:none",
        "translator_id": "translator:fol-v1",
        "toolchain_id": "toolchain:pinned-018",
        "policy_id": "policy:sawm-018",
        "environment_id": "env:current-018",
    }
    fields.update(overrides)
    return ProgramLogicAuthorityRoots(**fields)


def _obligation(**overrides: Any) -> ProgramWorldObligation:
    fields: dict[str, Any] = {
        "obligation_id": "obligation:transition-ping",
        "kind": ProgramWorldObligationKind.TRANSITION,
        "statement_ref": "stmt:ping-postcondition",
        "subject_cid": "symbol:pkg.mod.ping",
        "polarity": ProgramWorldPolarity.POSITIVE,
        "language": "python",
        "logic_family": "fol",
        "environment_binding_cid": "env:current-018",
        "tree_id": "tree:current-018",
        "policy_cid": "policy:sawm-018",
        "evidence_cids": ("evidence:graph-edge-1",),
    }
    fields.update(overrides)
    return ProgramWorldObligation(**fields)


def _premise(**overrides: Any) -> ProgramWorldPremiseNomination:
    fields: dict[str, Any] = {
        "premise_id": "premise:contract-ping",
        "statement_ref": "stmt:ping-contract",
        "source_class": PremiseSourceClass.REVIEWED_CONTRACT,
        "origin": "current_tree",
        "evidence_cid": "evidence:contract-1",
        "tree_id": "tree:current-018",
    }
    fields.update(overrides)
    return ProgramWorldPremiseNomination(**fields)


def _compile(**overrides: Any):
    kwargs: dict[str, Any] = {
        "roots": _roots(),
        "obligations": [_obligation()],
        "premises": [
            _premise(),
            _premise(
                premise_id="premise:graph-ping",
                statement_ref="stmt:ping-calls-pong",
                source_class=PremiseSourceClass.PROGRAM_GRAPH,
                evidence_cid="evidence:graph-1",
            ),
        ],
        "expected_tree_id": "tree:current-018",
        "expected_environment_id": "env:current-018",
        "expected_policy_id": "policy:sawm-018",
    }
    kwargs.update(overrides)
    return compile_program_world_goals(**kwargs)


class _ProofBackend:
    def __init__(self, **overrides: Any) -> None:
        self.overrides = overrides
        self.calls = 0

    def search(self, request):
        self.calls += 1
        goal = request.goal
        fields: dict[str, Any] = {
            "candidate_id": "candidate:proof-1",
            "kind": ProgramWorldCandidateKind.PROOF,
            "origin": "hammer",
            "status": "candidate",
            "theorem_id": goal.positive_statement_ref,
            "tree_id": request.compilation.roots.tree_id,
            "environment_id": request.compilation.roots.environment_id,
            "toolchain_id": request.compilation.roots.toolchain_id,
            "translation_map_id": "translation:map-1",
            "originating_logic_ir_id": goal.positive_statement_ref,
            "kernel_script_ref": "script:native-1",
        }
        fields.update(self.overrides)
        return ProgramWorldHammerCandidate(**fields)


class _AcceptingReconstructor:
    def reconstruct(self, candidate, *, compilation, goal):
        return ProgramWorldReconstructionReceipt(
            reconstruction_id="reconstruction:accepted",
            theorem_id=goal.positive_statement_ref,
            kernel_id="kernel:lean4",
            tree_id=compilation.roots.tree_id,
            environment_id=compilation.roots.environment_id,
            toolchain_id=compilation.roots.toolchain_id,
            status=ProgramWorldReconstructionStatus.KERNEL_ACCEPTED,
            kernel_accepted=True,
            matching_theorem=True,
            sorry_free=True,
            native_source_digest="sha256:" + ("ab" * 32),
            environment_lock_id="envlock:pinned-018",
        )


class _RejectingReconstructor:
    def reconstruct(self, candidate, *, compilation, goal):
        return ProgramWorldReconstructionReceipt(
            reconstruction_id="reconstruction:rejected",
            theorem_id=goal.positive_statement_ref,
            kernel_id="kernel:lean4",
            tree_id=compilation.roots.tree_id,
            environment_id=compilation.roots.environment_id,
            toolchain_id=compilation.roots.toolchain_id,
            status=ProgramWorldReconstructionStatus.KERNEL_REJECTED,
            kernel_accepted=False,
            matching_theorem=True,
            sorry_free=True,
            diagnostic="kernel rejected reconstructed tactic",
        )


class _ReplayOk:
    def replay(self, candidate, *, compilation, goal):
        return {
            "status": "validated",
            "replay_method": "deterministic_logic_ir_replay",
            "evidence_id": "replay:cm-1",
            "tree_id": compilation.roots.tree_id,
            "originating_logic_ir_id": goal.positive_statement_ref,
        }


class _ReplayFail:
    def replay(self, candidate, *, compilation, goal):
        return {
            "status": "replay_failed",
            "replay_method": "deterministic_logic_ir_replay",
            "evidence_id": "replay:cm-fail",
            "tree_id": compilation.roots.tree_id,
            "originating_logic_ir_id": goal.positive_statement_ref,
        }


def _class_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}


def _function_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}


def test_public_interfaces_and_predicted_symbols() -> None:
    tactician_classes = _class_names(TACTICIAN_PATH)
    hammer_classes = _class_names(HAMMER_PATH)
    tactician_functions = _function_names(TACTICIAN_PATH)
    hammer_functions = _function_names(HAMMER_PATH)
    assert "ProgramWorldTactician" in tactician_classes
    assert "ProgramWorldPremiseCorpus" in tactician_classes
    assert "ProgramWorldHammer" in hammer_classes
    assert "ProgramWorldProofAdmission" in hammer_classes
    assert "compile_program_world_goals" in tactician_functions
    assert "search_program_world_proof" in hammer_functions
    assert "replay_program_world_countermodel" in hammer_functions
    assert PROGRAM_WORLD_PROOF_SEARCH_INTERFACE == "ProgramWorldProofSearch@1"
    assert PROGRAM_WORLD_TACTICIAN_INTERFACE == "ProgramWorldTactician@1"
    assert PROGRAM_WORLD_HAMMER_INTERFACE == "ProgramWorldHammer@1"
    assert PROGRAM_WORLD_PREMISE_CORPUS_INTERFACE == "ProgramWorldPremiseCorpus@1"
    assert PROGRAM_WORLD_PROOF_ADMISSION_INTERFACE == "ProgramWorldProofAdmission@1"
    assert SAWM_TACTICIAN_HAMMER_EVIDENCE == "sawm/tactician-hammer@1"
    assert inspect.isfunction(compile_program_world_goals)
    assert inspect.isfunction(search_program_world_proof)
    assert inspect.isfunction(replay_program_world_countermodel)


def test_adapter_does_not_define_a_second_prover_or_tactic_engine() -> None:
    defined = _class_names(TACTICIAN_PATH) | _class_names(HAMMER_PATH)
    assert defined & _BANNED_CLASS_DEFS == set()
    source = TACTICIAN_PATH.read_text(encoding="utf-8") + HAMMER_PATH.read_text(
        encoding="utf-8"
    )
    for banned in (
        "class LogicTactician",
        "class HammerBackend",
        "class MultiProverRouter",
        "class FormalVerificationCache",
    ):
        assert banned not in source


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
    assert compilation.disposition is ProgramWorldCompilationDisposition.PLANNED
    assert compilation.reason_code is ProgramWorldReasonCode.TACTICIAN_PLANNED
    assert compilation.semantic_authority is False
    assert compilation.inventory.finite is True
    assert compilation.inventory.to_dict()["finite"] is True
    assert compilation.inventory.obligation_cids
    assert compilation.inventory.goal_ids == ("obligation:transition-ping",)
    assert compilation.inventory.complete is True
    assert isinstance(compilation.corpus, ProgramWorldPremiseCorpus)
    assert compilation.corpus.corpus_cid
    assert compilation.corpus.semantic_authority is False
    assert "premise:contract-ping" in compilation.corpus.axiom_premise_ids
    replay = _compile()
    assert compilation.compilation_cid == replay.compilation_cid
    assert compilation.corpus.corpus_cid == replay.corpus.corpus_cid


def test_tactic_plan_is_advisory_and_excludes_model_nominations() -> None:
    compilation = _compile(
        premises=[
            _premise(),
            _premise(
                premise_id="premise:model-guess",
                statement_ref="stmt:model-guess",
                source_class=PremiseSourceClass.MODEL_HYPOTHESIS,
                origin="model",
                nominated_by_model=True,
            ),
        ]
    )
    assert compilation.tactic_plan is not None
    assert compilation.tactic_plan.semantic_authority is False
    assert "premise:model-guess" in compilation.corpus.nominated_premise_ids
    assert "premise:model-guess" not in compilation.corpus.axiom_premise_ids
    assert "premise:model-guess" not in compilation.tactic_plan.selected_premise_ids
    for subgoal in compilation.tactic_plan.subgoals:
        assert subgoal.proof_status is ProofStatus.UNPROVED


def test_contradictory_obligations_abstain() -> None:
    compilation = _compile(
        obligations=[
            _obligation(),
            _obligation(
                obligation_id="obligation:negation-ping",
                kind=ProgramWorldObligationKind.NEGATIVE,
                polarity=ProgramWorldPolarity.NEGATIVE,
            ),
        ]
    )
    assert compilation.disposition is ProgramWorldCompilationDisposition.CONFLICT
    assert compilation.reason_code is ProgramWorldReasonCode.CONFLICTING_OBLIGATIONS
    assert compilation.tactic_plan is None
    assert "stmt:ping-postcondition" in compilation.inventory.conflict_statement_refs
    result = search_program_world_proof(
        compilation,
        backend=_ProofBackend(),
        reconstructor=_AcceptingReconstructor(),
    )
    assert result.admission.disposition is ProgramWorldProofDisposition.CONFLICT
    assert result.admission.admitted is False
    assert result.admission.reason_code is ProgramWorldReasonCode.CONTRADICTION


def test_unsupported_logic_family_is_typed() -> None:
    compilation = _compile(
        obligations=[_obligation(logic_family="modal-mu-calculus")]
    )
    assert compilation.disposition is ProgramWorldCompilationDisposition.UNSUPPORTED
    assert compilation.reason_code is ProgramWorldReasonCode.LOGIC_FAMILY_UNSUPPORTED
    result = search_program_world_proof(compilation, backend=_ProofBackend())
    assert result.admission.disposition is ProgramWorldProofDisposition.UNSUPPORTED
    assert result.admission.admitted is False


def test_stale_tree_bindings_fail_closed() -> None:
    with pytest.raises(ProgramWorldStaleBindingError):
        _compile(expected_tree_id="tree:other")
    with pytest.raises(ProgramWorldStaleBindingError):
        _compile(
            obligations=[_obligation(tree_id="tree:stale")],
        )


def test_reconstructed_kernel_proof_is_admitted() -> None:
    compilation = _compile()
    result = search_program_world_proof(
        compilation,
        backend=_ProofBackend(),
        reconstructor=_AcceptingReconstructor(),
    )
    assert result.admission.admitted is True
    assert result.admission.disposition is ProgramWorldProofDisposition.ADMITTED_PROOF
    assert result.admission.reason_code is ProgramWorldReasonCode.RECONSTRUCTED
    assert result.admission.reconstruction is not None
    assert result.admission.reconstruction.conclusive_proof is True
    assert result.admission.semantic_authority is False
    assert "premise:contract-ping" in result.selected_axiom_ids


def test_unreconstructed_solver_candidate_is_not_admitted() -> None:
    compilation = _compile()
    result = search_program_world_proof(
        compilation,
        backend=_ProofBackend(proof_success_claimed=True, kernel_accepted_claimed=True),
        reconstructor=_RejectingReconstructor(),
    )
    assert result.admission.admitted is False
    assert result.admission.disposition is ProgramWorldProofDisposition.CANDIDATE
    assert result.admission.reason_code is ProgramWorldReasonCode.CANDIDATE_NOT_ADMITTED


def test_model_cannot_create_proof_authority() -> None:
    compilation = _compile()
    result = search_program_world_proof(
        compilation,
        backend=_ProofBackend(
            origin="model",
            proof_success_claimed=True,
            kernel_accepted_claimed=True,
        ),
        reconstructor=_RejectingReconstructor(),
    )
    assert result.admission.admitted is False
    assert result.admission.disposition is ProgramWorldProofDisposition.REJECTED
    assert (
        result.admission.reason_code is ProgramWorldReasonCode.MODEL_CANNOT_CREATE_PROOF
    )


def test_replayed_countermodel_is_admitted() -> None:
    compilation = _compile()
    candidate = ProgramWorldHammerCandidate(
        candidate_id="candidate:cm-1",
        kind=ProgramWorldCandidateKind.COUNTERMODEL,
        origin="hammer",
        status="counterexample",
        theorem_id="stmt:ping-postcondition",
        tree_id="tree:current-018",
        translation_map_id="translation:map-1",
        originating_logic_ir_id="stmt:ping-postcondition",
        countermodel_id="countermodel:raw-1",
        raw_diagnostic_refs=("diag:raw-1",),
    )
    admission = replay_program_world_countermodel(
        compilation,
        candidate,
        replayer=_ReplayOk(),
    )
    assert admission.admitted is True
    assert admission.disposition is ProgramWorldProofDisposition.ADMITTED_COUNTERMODEL
    assert admission.reason_code is ProgramWorldReasonCode.REPLAYED
    assert admission.countermodel is not None
    assert admission.countermodel.disposition is CountermodelDisposition.VALIDATED


def test_raw_solver_countermodel_stays_diagnostic_until_replayed() -> None:
    compilation = _compile()
    candidate = ProgramWorldHammerCandidate(
        candidate_id="candidate:cm-raw",
        kind=ProgramWorldCandidateKind.COUNTERMODEL,
        origin="solver",
        status="counterexample",
        theorem_id="stmt:ping-postcondition",
        tree_id="tree:current-018",
        translation_map_id="translation:map-1",
        originating_logic_ir_id="stmt:ping-postcondition",
        countermodel_id="countermodel:raw-2",
        raw_diagnostic_refs=("diag:raw-2",),
        proof_success_claimed=True,
    )
    admission = replay_program_world_countermodel(compilation, candidate)
    assert admission.admitted is False
    assert admission.disposition is ProgramWorldProofDisposition.DIAGNOSTIC
    assert admission.reason_code is ProgramWorldReasonCode.DIAGNOSTIC_ONLY
    assert admission.countermodel is not None
    assert admission.countermodel.disposition is CountermodelDisposition.DIAGNOSTIC_ONLY


def test_model_cannot_create_countermodel_truth() -> None:
    compilation = _compile()
    candidate = ProgramWorldHammerCandidate(
        candidate_id="candidate:cm-model",
        kind=ProgramWorldCandidateKind.COUNTERMODEL,
        origin="llm",
        status="counterexample",
        theorem_id="stmt:ping-postcondition",
        tree_id="tree:current-018",
        translation_map_id="translation:map-1",
        originating_logic_ir_id="stmt:ping-postcondition",
        countermodel_id="countermodel:model-1",
        raw_diagnostic_refs=("diag:model-1",),
        proof_success_claimed=True,
    )
    admission = replay_program_world_countermodel(compilation, candidate)
    assert admission.admitted is False
    assert admission.countermodel is not None
    assert admission.countermodel.disposition is not CountermodelDisposition.VALIDATED
    failed = replay_program_world_countermodel(
        compilation,
        candidate,
        replayer=_ReplayFail(),
    )
    assert failed.admitted is False
    assert failed.disposition is ProgramWorldProofDisposition.DIAGNOSTIC


def test_timeout_is_typed_and_non_admitted() -> None:
    compilation = _compile()
    result = search_program_world_proof(
        compilation,
        backend=_ProofBackend(status="timeout", timeout=True),
        reconstructor=_AcceptingReconstructor(),
    )
    assert result.admission.admitted is False
    assert result.admission.disposition is ProgramWorldProofDisposition.TIMEOUT
    assert result.admission.reason_code is ProgramWorldReasonCode.TIMEOUT


def test_unavailability_is_typed_against_validation_path() -> None:
    receipts = probe_program_world_proof_capabilities()
    kinds = {item.kind for item in receipts}
    assert ProgramWorldCapabilityKind.TACTICIAN in kinds
    assert ProgramWorldCapabilityKind.HAMMER in kinds
    assert ProgramWorldCapabilityKind.NATIVE_RECONSTRUCTION in kinds
    assert ProgramWorldCapabilityKind.BACKEND in kinds
    kernel = next(
        item
        for item in receipts
        if item.kind is ProgramWorldCapabilityKind.NATIVE_RECONSTRUCTION
    )
    compilation = _compile(capabilities=receipts)
    result = search_program_world_proof(compilation, capabilities=receipts)
    assert result.admission.admitted is False
    assert result.admission.disposition is ProgramWorldProofDisposition.UNAVAILABLE
    if not kernel.available:
        assert result.admission.reason_code in {
            ProgramWorldReasonCode.KERNEL_UNAVAILABLE,
            ProgramWorldReasonCode.BACKEND_UNAVAILABLE,
            ProgramWorldReasonCode.HAMMER_UNAVAILABLE,
        }
        assert result.admission.reconstruction is None or (
            result.admission.reconstruction.kernel_accepted is False
        )


def test_empty_obligations_are_rejected() -> None:
    compilation = compile_program_world_goals(roots=_roots(), obligations=())
    assert compilation.disposition is ProgramWorldCompilationDisposition.REJECTED
    assert compilation.reason_code is ProgramWorldReasonCode.EMPTY_OBLIGATIONS
    assert compilation.goals == ()


def test_search_rejects_stale_hammer_candidate_tree() -> None:
    compilation = _compile()
    result = search_program_world_proof(
        compilation,
        backend=_ProofBackend(tree_id="tree:other"),
        reconstructor=_AcceptingReconstructor(),
    )
    assert result.admission.admitted is False
    assert result.admission.disposition is ProgramWorldProofDisposition.STALE
    assert result.admission.reason_code is ProgramWorldReasonCode.STALE_TREE


def test_program_world_tactician_class_matches_module_function() -> None:
    tactician = ProgramWorldTactician()
    compilation = tactician.compile(
        roots=_roots(),
        obligations=[_obligation()],
        premises=[_premise()],
    )
    assert compilation.disposition is ProgramWorldCompilationDisposition.PLANNED
    hammer = ProgramWorldHammer(
        backend=_ProofBackend(),
        reconstructor=_AcceptingReconstructor(),
    )
    result = hammer.search(compilation)
    assert isinstance(result.admission, ProgramWorldProofAdmission)
    assert result.admission.admitted is True
