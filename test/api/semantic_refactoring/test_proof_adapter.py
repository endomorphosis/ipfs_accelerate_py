"""Independent contract tests for SPAR-030 Tactician/Hammer remodularization adapter."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.utils.cid_utils import cid_for_bytes, cid_for_dag_json
from ipfs_accelerate_py.agent_supervisor.semantic_refactoring.proof_adapter import (
    ADAPTER_IS_NOMINATION_ONLY,
    ANALYZER_ID,
    AUTHORITY,
    AUTHORITY_OWNER,
    BOUNDED_PROOF_SEARCH_RECEIPT_INTERFACE,
    BOUNDARY_OBLIGATION_INTERFACE,
    COUNTERMODEL_REPLAY_INTERFACE,
    DECLARED_ARTIFACT_KINDS,
    DECLARED_SEARCH_OUTCOMES,
    DECLARED_SEARCH_STATUSES,
    DRY_RUN_IS_DETERMINISTIC,
    DRY_RUN_MUTATES,
    DUCKLAKE_IS_AUTHORITY,
    FORBIDDEN_PROOF_NAMES,
    GENERAL_PYTHON_EQUIVALENCE_CLAIMED,
    GOAL_ID,
    GUESSED_AXIOMS_REJECTED,
    HAMMER_OWNS_PRODUCTION_PROOF,
    IDENTITY_EXCLUDED_FIELDS,
    IMPLICIT_INSTALL_FORBIDDEN,
    IMPLICIT_NETWORK_FORBIDDEN,
    INCOMPLETE_CONTRACTS_ARE_TYPED_TERMINALS,
    MARKDOWN_IS_NOT_COMPLETION,
    MODEL_OUTPUT_IS_PROPOSAL_ONLY,
    NETWORK_DENIED,
    NETWORK_DENY,
    OBLIGATION_DECOMPOSITION_INTERFACE,
    PREMISE_CORPUS_INTERFACE,
    PREMISE_SELECTOR_DETERMINISTIC,
    PROGRAM,
    PROOF_ADAPTER_CAN_AUTHORIZE_COMPLETION,
    PROOF_ADAPTER_CAN_AUTHORIZE_TRANSITION,
    PROOF_ADAPTER_CAN_CREATE_AUTHORITY,
    PROOF_ADAPTER_CAN_CREATE_PROOF_AUTHORITY,
    PROOF_ADAPTER_CONTRACT_VERSION,
    PROOF_CANDIDATE_CANNOT_ADMIT_PROOFS,
    PROOF_RECONSTRUCTION_INTERFACE,
    PROJECTION_CLUSTERING_IS_AUTHORITY,
    RAW_COUNTERMODEL_CANNOT_REFUTE,
    RAW_SOURCE_REQUIRED,
    SOFT_SIGNALS_CANNOT_OVERRIDE_HARD_CONSTRAINTS,
    TACTICIAN_HAMMER_ADAPTER_INTERFACE,
    TACTICIAN_OWNS_SEARCH,
    TASK_ID,
    TEST_PASS_IS_NOT_COMPLETION,
    TEST_PASS_IS_NOT_PROOF,
    UNKNOWN_REMAINS_UNKNOWN,
    VECTOR_SIMILARITY_IS_AUTHORITY,
    WORKER_SELF_APPROVAL,
    ArtifactKind,
    BoundedProofSearchReceipt,
    ObligationStatus,
    PremiseCorpus,
    ProofAdapterError,
    ProofReconstruction,
    SearchStatus,
    TacticianHammerAdapter,
    assert_not_competing_capsule_family,
    compile_premise_corpus,
    decode_canonical_corpus,
    decode_canonical_decomposition,
    decode_canonical_receipt,
    dry_run_bounded_proof_search,
    encode_canonical_corpus,
    encode_canonical_decomposition,
    encode_canonical_receipt,
    lower_boundary_obligations,
    proof_adapter_cid_profile,
    proof_adapter_descriptor,
    provider_free_exports,
    reconstruct_proof,
    replay_countermodel,
    run_bounded_proof_search,
)


ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    ROOT
    / "ipfs_accelerate_py"
    / "agent_supervisor"
    / "semantic_refactoring"
    / "proof_adapter.py"
)
TEST_PATH = Path(__file__).resolve()
WRITE_SCOPE = (
    "ipfs_accelerate_py/agent_supervisor/semantic_refactoring/proof_adapter.py",
    "test/api/semantic_refactoring/test_proof_adapter.py",
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
WRITE_PATHS = ("pkg/mod.py", "pkg/extracted.py")
VALIDATION = ("python3 -m pytest -q tests/test_mod.py",)
CORPUS_REVISION = "corpus:r1"
ENVIRONMENT_ID = "env:local"
TOOLCHAIN_ID = "toolchain:hammer@1"
OBLIGATION_ID = "obl:inputs"


def _cid(label: str) -> str:
    return cid_for_bytes(label.encode("utf-8"))


def _wave(**overrides: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "tree_id": TREE_ID,
        "receipt_cid": _cid("wave"),
        "packet_cids": [_cid("packet")],
        "write_paths": list(WRITE_PATHS),
        "worktree_id": _cid("worktree"),
        "status": "applied",
        "writes_repository": False,
        "executor_is_nomination_only": True,
    }
    fields.update(overrides)
    return fields


def _selection(**overrides: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "tree_id": TREE_ID,
        "validation_selection_cid": _cid("selection"),
        "packet_cid": _cid("packet"),
        "raw_source_cids": [_cid("source")],
        "write_paths": list(WRITE_PATHS),
        "validation_commands": list(VALIDATION),
        "selected_proof_ids": [OBLIGATION_ID],
        "raw_source_required": True,
        "adapter_is_nomination_only": True,
        "datasets_owns_selection": True,
    }
    fields.update(overrides)
    return fields


def _clause(**overrides: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "clause_id": "clause:inputs",
        "kind": "input",
        "polarity": "assume",
        "finite": True,
        "required": True,
        "obligation_id": OBLIGATION_ID,
    }
    fields.update(overrides)
    return fields


def _contract(**overrides: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "tree_id": TREE_ID,
        "contract_cid": _cid("contract"),
        "disposition": "admitted",
        "guessed_axiom": False,
        "clauses": [_clause()],
    }
    fields.update(overrides)
    return fields


def _premise(**overrides: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "premise_id": "premise:bound",
        "premise_cid": _cid("premise"),
        "source_authority": "specification",
        "tree_id": TREE_ID,
        "corpus_revision": CORPUS_REVISION,
        "environment_id": ENVIRONMENT_ID,
        "toolchain_id": TOOLCHAIN_ID,
    }
    fields.update(overrides)
    return fields


def _corpus(**overrides: Any) -> PremiseCorpus:
    fields: dict[str, Any] = {
        "tree_id": TREE_ID,
        "corpus_revision": CORPUS_REVISION,
        "premises": [_premise()],
        "environment_id": ENVIRONMENT_ID,
        "toolchain_id": TOOLCHAIN_ID,
    }
    fields.update(overrides)
    return compile_premise_corpus(**fields)


def _artifact(**overrides: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "obligation_id": OBLIGATION_ID,
        "kind": ArtifactKind.RECONSTRUCTED_PROOF.value,
        "artifact_cid": _cid("reconstruction"),
        "reconstruction_cid": _cid("reconstruction"),
        "kernel_checked": True,
        "replayed": False,
        "tree_id": TREE_ID,
        "corpus_revision": CORPUS_REVISION,
        "environment_id": ENVIRONMENT_ID,
        "toolchain_id": TOOLCHAIN_ID,
    }
    fields.update(overrides)
    return fields


def _lower(**overrides: Any) -> Any:
    fields: dict[str, Any] = {
        "contracts": [_contract()],
        "corpus": _corpus(),
        "wave": _wave(),
        "selection": _selection(),
    }
    fields.update(overrides)
    return lower_boundary_obligations(**fields)


def _search(**overrides: Any) -> BoundedProofSearchReceipt:
    decomposition = overrides.pop("decomposition", None)
    corpus = overrides.pop("corpus", None)
    if corpus is None:
        corpus = _corpus()
    if decomposition is None:
        decomposition = _lower(corpus=corpus)
    fields: dict[str, Any] = {
        "decomposition": decomposition,
        "corpus": corpus,
        "artifacts": {OBLIGATION_ID: _artifact()},
    }
    fields.update(overrides)
    return run_bounded_proof_search(**fields)


def test_owned_paths_and_task_identity_are_exact() -> None:
    assert TASK_ID == "SPAR-030"
    assert GOAL_ID == "SPAR-G053"
    assert PROGRAM == "semantic-preserving-autonomous-remodularization-v1"
    assert TACTICIAN_HAMMER_ADAPTER_INTERFACE == "TacticianHammerRemodularizationAdapter@1"
    assert PREMISE_CORPUS_INTERFACE == "PremiseCorpus@1"
    assert BOUNDARY_OBLIGATION_INTERFACE == "BoundaryObligation@1"
    assert OBLIGATION_DECOMPOSITION_INTERFACE == "ObligationDecomposition@1"
    assert PROOF_RECONSTRUCTION_INTERFACE == "ProofReconstruction@1"
    assert COUNTERMODEL_REPLAY_INTERFACE == "CountermodelReplay@1"
    assert BOUNDED_PROOF_SEARCH_RECEIPT_INTERFACE == "BoundedProofSearchReceipt@1"
    assert PROOF_ADAPTER_CONTRACT_VERSION == "1"
    assert ANALYZER_ID.endswith("proof_adapter@1")
    assert MODULE_PATH.is_file()
    assert TEST_PATH.is_file()
    for relative in WRITE_SCOPE:
        assert (ROOT / relative).is_file()


def test_authority_flags_cannot_self_authorize() -> None:
    assert AUTHORITY == "proof search"
    assert AUTHORITY_OWNER == "ipfs_accelerate_py"
    assert PROOF_ADAPTER_CAN_AUTHORIZE_COMPLETION is False
    assert PROOF_ADAPTER_CAN_AUTHORIZE_TRANSITION is False
    assert PROOF_ADAPTER_CAN_CREATE_AUTHORITY is False
    assert PROOF_ADAPTER_CAN_CREATE_PROOF_AUTHORITY is False
    assert VECTOR_SIMILARITY_IS_AUTHORITY is False
    assert PROJECTION_CLUSTERING_IS_AUTHORITY is False
    assert MODEL_OUTPUT_IS_PROPOSAL_ONLY is True
    assert TEST_PASS_IS_NOT_COMPLETION is True
    assert TEST_PASS_IS_NOT_PROOF is True
    assert PROOF_CANDIDATE_CANNOT_ADMIT_PROOFS is True
    assert RAW_COUNTERMODEL_CANNOT_REFUTE is True
    assert MARKDOWN_IS_NOT_COMPLETION is True
    assert WORKER_SELF_APPROVAL is False
    assert DUCKLAKE_IS_AUTHORITY is False
    assert SOFT_SIGNALS_CANNOT_OVERRIDE_HARD_CONSTRAINTS is True
    assert ADAPTER_IS_NOMINATION_ONLY is True
    assert RAW_SOURCE_REQUIRED is True
    assert NETWORK_DENIED is True
    assert DRY_RUN_IS_DETERMINISTIC is True
    assert DRY_RUN_MUTATES is False
    assert UNKNOWN_REMAINS_UNKNOWN is True
    assert GUESSED_AXIOMS_REJECTED is True
    assert INCOMPLETE_CONTRACTS_ARE_TYPED_TERMINALS is True
    assert GENERAL_PYTHON_EQUIVALENCE_CLAIMED is False
    assert IMPLICIT_INSTALL_FORBIDDEN is True
    assert IMPLICIT_NETWORK_FORBIDDEN is True
    assert TACTICIAN_OWNS_SEARCH is True
    assert HAMMER_OWNS_PRODUCTION_PROOF is True
    assert DECLARED_SEARCH_OUTCOMES == {
        "verified",
        "candidate",
        "counterexample",
        "timeout",
        "unsupported",
        "unavailable",
        "unknown",
        "stale",
        "error",
    }
    assert DECLARED_SEARCH_STATUSES == {
        "verified",
        "refuted",
        "blocked",
        "incomplete",
        "rejected",
        "timeout",
        "unknown",
    }
    assert ArtifactKind.RECONSTRUCTED_PROOF.value in DECLARED_ARTIFACT_KINDS
    profile = proof_adapter_cid_profile()
    assert profile["codec"] == "dag-json"
    assert "not universal meaning" in profile["rule"]


def test_module_defines_predicted_symbols_not_capsule_family() -> None:
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    names = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
    assert "TacticianHammerAdapter" in names
    assert "PremiseCorpus" in names
    assert "BoundaryObligation" in names
    assert "ObligationDecomposition" in names
    assert "BoundedProofSearchReceipt" in names
    assert "ProofReconstruction" in names
    assert "CountermodelReplay" in names
    for capsule in CAPSULE_TYPES:
        assert capsule not in names
    assert_not_competing_capsule_family()
    exports = provider_free_exports()
    assert "TacticianHammerAdapter" in exports
    assert "compile_premise_corpus" in exports
    assert "lower_boundary_obligations" in exports
    assert "run_bounded_proof_search" in exports
    assert "reconstruct_proof" in exports
    assert "replay_countermodel" in exports


def test_protected_paths_are_not_owned_write_scope() -> None:
    owned = set(WRITE_SCOPE)
    for relative in PROTECTED_PATHS:
        assert relative not in owned
        assert (ROOT / relative).exists()


def test_module_does_not_claim_proof_authority_or_open_network() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    assert not (names & FORBIDDEN_PROOF_NAMES)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    assert not any("tactician_hammer_coordinator" in name for name in imported)
    assert not any("procedure_compiler" in name for name in imported)
    descriptor = proof_adapter_descriptor()
    assert descriptor["interface"] == TACTICIAN_HAMMER_ADAPTER_INTERFACE
    assert descriptor["network"] == NETWORK_DENY
    assert descriptor["nomination_only"] is True
    assert descriptor["claims_general_equivalence"] is False
    assert descriptor["proof_candidate_cannot_admit_proofs"] is True
    assert descriptor["raw_countermodel_cannot_refute"] is True
    assert descriptor["unknown_remains_unknown"] is True
    assert descriptor["guessed_axioms_rejected"] is True
    assert descriptor["tactician_owns_search"] is True
    assert descriptor["hammer_owns_production_proof"] is True
    forbids = set(descriptor["forbids"])
    assert "claim_general_equivalence" in forbids
    assert "admit_vector_proof" in forbids
    assert "promote_candidate_to_proof" in forbids
    assert "guess_axiom" in forbids
    assert "open_network" in forbids


def test_compile_premise_corpus_is_content_addressed_and_body_free() -> None:
    corpus = _corpus()
    assert corpus.tree_id == TREE_ID
    assert corpus.selector_mode == PREMISE_SELECTOR_DETERMINISTIC
    assert corpus.premise_ids == ("premise:bound",)
    assert corpus.premises[0].source_authority == "specification"
    again = _corpus()
    assert corpus.corpus_cid == again.corpus_cid
    restored = decode_canonical_corpus(encode_canonical_corpus(corpus))
    assert restored == corpus


def test_guessed_and_vector_premises_cannot_become_axioms() -> None:
    with pytest.raises(ProofAdapterError, match="guessed"):
        _corpus(premises=[_premise(source_authority="guessed")])
    with pytest.raises(ProofAdapterError, match="cannot become axioms"):
        _corpus(premises=[_premise(source_authority="vector_candidate")])
    with pytest.raises(ProofAdapterError, match="cannot become axioms"):
        _corpus(premises=[_premise(source_authority="model_hypothesis")])
    with pytest.raises(ProofAdapterError, match="cannot admit"):
        _corpus(vector_evidence={"evidence_class": "heuristic", "admit_proof": True})


def test_lower_binds_spar016_025_026_without_competing_authority() -> None:
    decomposition = _lower()
    assert decomposition.tree_id == TREE_ID
    assert decomposition.wave_cid == _cid("wave")
    assert decomposition.selection_cid == _cid("selection")
    assert decomposition.packet_cid == _cid("packet")
    assert decomposition.write_paths == WRITE_PATHS
    assert decomposition.raw_source_cids == (_cid("source"),)
    assert decomposition.validation_commands == VALIDATION
    assert decomposition.selected_proof_ids == (OBLIGATION_ID,)
    assert decomposition.lowered_obligation_ids == (OBLIGATION_ID,)
    obligation = decomposition.obligations[0]
    assert obligation.kind == "input"
    assert obligation.polarity == "assume"
    assert obligation.finite is True
    assert obligation.status == ObligationStatus.LOWERED.value
    assert obligation.required is True


def test_incomplete_and_guessed_contracts_are_typed_terminals() -> None:
    with pytest.raises(ProofAdapterError, match="incomplete SPAR-016"):
        _lower(contracts=[_contract(disposition="incomplete")])
    with pytest.raises(ProofAdapterError, match="incomplete SPAR-016"):
        _lower(contracts=[_contract(clauses=[])])
    with pytest.raises(ProofAdapterError, match="guessed axioms"):
        _lower(contracts=[_contract(guessed_axiom=True)])
    with pytest.raises(ProofAdapterError, match="unsupported required"):
        _lower(contracts=[_contract(unsupported_required=True)])
    with pytest.raises(ProofAdapterError, match="unsupported required"):
        _lower(
            contracts=[
                _contract(clauses=[_clause(finite=False)]),
            ]
        )
    with pytest.raises(ProofAdapterError, match="unsupported required"):
        _lower(
            contracts=[
                _contract(clauses=[_clause(semantics="higher_order")]),
            ]
        )


def test_reconstructed_proof_verifies_required_obligation() -> None:
    receipt = _search()
    assert receipt.status == SearchStatus.VERIFIED.value
    assert receipt.adapter_is_nomination_only is True
    assert receipt.can_authorize_completion is False
    assert receipt.can_authorize_transition is False
    assert receipt.can_create_proof_authority is False
    assert receipt.network == NETWORK_DENY
    assert receipt.mutated is False
    assert receipt.deterministic is True
    assert receipt.claims_general_equivalence is False
    assert len(receipt.reconstructions) == 1
    assert receipt.reconstructions[0].kernel_checked is True
    assert receipt.steps[0].kind == ArtifactKind.RECONSTRUCTED_PROOF.value
    assert receipt.typed_terminal is False
    adapter = TacticianHammerAdapter()
    again = adapter.search(
        decomposition=_lower(),
        corpus=_corpus(),
        artifacts={OBLIGATION_ID: _artifact()},
    )
    assert again.receipt_cid == receipt.receipt_cid


def test_proof_candidate_cannot_admit_proofs() -> None:
    receipt = _search(
        artifacts={
            OBLIGATION_ID: _artifact(
                kind=ArtifactKind.PROOF_CANDIDATE.value,
                kernel_checked=False,
                reconstruction_cid=_cid("candidate"),
                artifact_cid=_cid("candidate"),
            )
        }
    )
    assert receipt.status == SearchStatus.INCOMPLETE.value
    assert receipt.steps[0].kind == ArtifactKind.PROOF_CANDIDATE.value
    assert receipt.reconstructions == ()
    with pytest.raises(ProofAdapterError, match="proof_candidate cannot admit proofs"):
        reconstruct_proof(
            _artifact(kind=ArtifactKind.PROOF_CANDIDATE.value, kernel_checked=False),
            tree_id=TREE_ID,
            corpus_revision=CORPUS_REVISION,
            environment_id=ENVIRONMENT_ID,
            toolchain_id=TOOLCHAIN_ID,
        )


def test_raw_countermodel_cannot_refute_until_replay() -> None:
    receipt = _search(
        artifacts={
            OBLIGATION_ID: _artifact(
                kind=ArtifactKind.COUNTERMODEL.value,
                kernel_checked=False,
                replayed=False,
                artifact_cid=_cid("raw-cm"),
            )
        }
    )
    assert receipt.status == SearchStatus.INCOMPLETE.value
    assert receipt.steps[0].kind == ArtifactKind.COUNTERMODEL.value
    assert receipt.steps[0].conclusiveness == "diagnostic"
    assert receipt.replays == ()
    with pytest.raises(ProofAdapterError, match="raw countermodels cannot refute"):
        replay_countermodel(
            _artifact(
                kind=ArtifactKind.COUNTERMODEL.value,
                replayed=False,
                artifact_cid=_cid("raw-cm"),
            ),
            tree_id=TREE_ID,
            corpus_revision=CORPUS_REVISION,
            environment_id=ENVIRONMENT_ID,
            toolchain_id=TOOLCHAIN_ID,
        )


def test_replayed_countermodel_is_conclusive_refutation() -> None:
    receipt = _search(
        artifacts={
            OBLIGATION_ID: _artifact(
                kind=ArtifactKind.REPLAYED_COUNTEREXAMPLE.value,
                kernel_checked=False,
                replayed=True,
                artifact_cid=_cid("replay"),
                replay_cid=_cid("replay"),
            )
        }
    )
    assert receipt.status == SearchStatus.REFUTED.value
    assert receipt.steps[0].kind == ArtifactKind.REPLAYED_COUNTEREXAMPLE.value
    assert receipt.steps[0].conclusiveness == "conclusive_refutation"
    assert len(receipt.replays) == 1
    assert receipt.replays[0].replayed is True


def test_unknown_remains_unknown_without_artifacts() -> None:
    receipt = _search(artifacts={})
    assert receipt.status == SearchStatus.UNKNOWN.value
    assert receipt.unknown_remains_unknown is True
    assert receipt.steps[0].kind == ArtifactKind.UNKNOWN.value
    assert receipt.typed_terminal is True


def test_stale_tree_or_corpus_is_rejected() -> None:
    receipt = _search(
        artifacts={
            OBLIGATION_ID: _artifact(tree_id=OTHER_TREE),
        }
    )
    assert receipt.status == SearchStatus.REJECTED.value
    assert receipt.steps[0].kind == ArtifactKind.STALE.value
    with pytest.raises(ProofAdapterError, match="stale reconstruction"):
        reconstruct_proof(
            _artifact(corpus_revision="corpus:other"),
            tree_id=TREE_ID,
            corpus_revision=CORPUS_REVISION,
            environment_id=ENVIRONMENT_ID,
            toolchain_id=TOOLCHAIN_ID,
        )


def test_timeout_is_a_typed_non_conclusive_terminal() -> None:
    receipt = _search(max_steps=1, artifacts={})
    # One obligation consumes the only step; unknown still wins if searched.
    assert receipt.steps_used == 1
    extra = _clause(clause_id="clause:outputs", obligation_id="obl:outputs", kind="output")
    decomposition = _lower(
        contracts=[_contract(clauses=[_clause(), extra])],
        selection=_selection(selected_proof_ids=[OBLIGATION_ID, "obl:outputs"]),
    )
    timed = run_bounded_proof_search(
        decomposition=decomposition,
        corpus=_corpus(),
        artifacts={OBLIGATION_ID: _artifact()},
        max_steps=1,
    )
    assert timed.status == SearchStatus.TIMEOUT.value
    assert any(item.kind == ArtifactKind.TIMEOUT.value for item in timed.steps)


def test_missing_predecessors_fail_closed() -> None:
    corpus = _corpus()
    with pytest.raises(ProofAdapterError, match="SPAR-016"):
        lower_boundary_obligations(
            contracts=[],
            corpus=corpus,
            wave=_wave(),
            selection=_selection(),
        )
    with pytest.raises(ProofAdapterError, match="SPAR-025"):
        lower_boundary_obligations(
            contracts=[_contract()],
            corpus=corpus,
            wave=None,
            selection=_selection(),
        )
    with pytest.raises(ProofAdapterError, match="SPAR-026"):
        lower_boundary_obligations(
            contracts=[_contract()],
            corpus=corpus,
            wave=_wave(),
            selection=None,
        )


def test_tree_mismatch_fails_closed() -> None:
    with pytest.raises(ProofAdapterError, match="SPAR-026 tree_id"):
        _lower(selection=_selection(tree_id=OTHER_TREE))
    with pytest.raises(ProofAdapterError, match="SPAR-016"):
        _lower(contracts=[_contract(tree_id=OTHER_TREE)])


def test_write_paths_must_match_and_stay_exact() -> None:
    with pytest.raises(ProofAdapterError, match="write_paths"):
        _lower(wave=_wave(write_paths=["pkg/mod.py"]))
    with pytest.raises(ProofAdapterError, match="unrestricted scope"):
        _lower(selection=_selection(write_paths=[]))
    with pytest.raises(ProofAdapterError, match="unrestricted scope"):
        _lower(selection=_selection(write_paths=["pkg/*.py"]))
    with pytest.raises(ProofAdapterError, match="unrestricted scope"):
        _lower(selection=_selection(write_paths=["/tmp/pkg/mod.py"]))
    with pytest.raises(ProofAdapterError, match="unrestricted scope"):
        _lower(selection=_selection(write_paths=["pkg/../secret.py"]))


def test_missing_raw_source_is_typed_terminal() -> None:
    with pytest.raises(ProofAdapterError, match="raw source"):
        _lower(selection=_selection(raw_source_cids=[]))


def test_vectors_cannot_admit_proofs_or_suppress_raw_source() -> None:
    with pytest.raises(ProofAdapterError, match="raw-source"):
        _lower(
            vector_evidence={
                "evidence_class": "vector_candidate",
                "suppress_raw_source": True,
            }
        )
    with pytest.raises(ProofAdapterError, match="cannot admit"):
        _search(
            vector_evidence={
                "evidence_class": "model_hypothesis",
                "admit_proof": True,
            }
        )


def test_body_free_artifacts_reject_proof_dumps() -> None:
    with pytest.raises(ProofAdapterError, match="body-free"):
        _search(
            artifacts={
                OBLIGATION_ID: _artifact(proof_body="theorem T : True := by trivial"),
            }
        )
    with pytest.raises(ProofAdapterError, match="body-free"):
        _corpus(premises=[_premise(source="axiom true")])


def test_network_is_denied() -> None:
    with pytest.raises(ProofAdapterError, match="network is denied"):
        _search(network="allow")


def test_hammer_and_tactician_injectables_remain_nomination_only() -> None:
    seen: dict[str, str] = {}

    def hammer(**request: Any) -> dict[str, Any]:
        seen["network"] = request["network"]
        seen["selector_mode"] = request["selector_mode"]
        return _artifact()

    receipt = _search(artifacts={}, hammer=hammer)
    assert seen["network"] == NETWORK_DENY
    assert seen["selector_mode"] == PREMISE_SELECTOR_DETERMINISTIC
    assert receipt.status == SearchStatus.VERIFIED.value
    assert receipt.can_create_proof_authority is False


def test_selected_proof_ids_must_bind_lowered_obligations() -> None:
    with pytest.raises(ProofAdapterError, match="selected_proof_ids"):
        _lower(selection=_selection(selected_proof_ids=["obl:missing"]))


def test_optional_non_finite_residual_does_not_block_when_unselected() -> None:
    residual_clause = _clause(
        clause_id="clause:dynamic",
        kind="condition",
        finite=False,
        required=False,
        obligation_id="obl:dynamic",
    )
    decomposition = _lower(
        contracts=[_contract(clauses=[_clause(), residual_clause])],
        selection=_selection(selected_proof_ids=[OBLIGATION_ID]),
    )
    assert any(item["reason"] == "non_finite_clause" for item in decomposition.residuals)
    receipt = _search(decomposition=decomposition)
    assert receipt.status == SearchStatus.VERIFIED.value


def test_unavailable_required_obligation_is_blocked() -> None:
    receipt = _search(
        artifacts={
            OBLIGATION_ID: _artifact(
                kind=ArtifactKind.UNAVAILABLE.value,
                kernel_checked=False,
                artifact_cid="",
            )
        }
    )
    assert receipt.status == SearchStatus.BLOCKED.value
    assert receipt.steps[0].kind == ArtifactKind.UNAVAILABLE.value


def test_dry_run_is_deterministic_and_does_not_mutate() -> None:
    corpus = _corpus()
    decomposition = _lower(corpus=corpus)
    first = dry_run_bounded_proof_search(
        decomposition=decomposition,
        corpus=corpus,
        artifacts={OBLIGATION_ID: _artifact()},
    )
    second = TacticianHammerAdapter().dry_run(
        decomposition=decomposition,
        corpus=corpus,
        artifacts={OBLIGATION_ID: _artifact()},
    )
    assert first.receipt_cid == second.receipt_cid
    assert first.mutated is False
    assert first.deterministic is True
    with pytest.raises(ProofAdapterError, match="cannot mutate"):
        run_bounded_proof_search(
            decomposition=decomposition,
            corpus=corpus,
            artifacts={OBLIGATION_ID: _artifact()},
            mutate=True,
        )


def test_round_trip_and_receipt_are_deterministic() -> None:
    first = _search()
    second = _search()
    assert first.receipt_cid == second.receipt_cid
    restored = decode_canonical_receipt(encode_canonical_receipt(first))
    assert restored == first
    corpus = _corpus()
    assert decode_canonical_corpus(encode_canonical_corpus(corpus)) == corpus
    decomposition = _lower(corpus=corpus)
    assert (
        decode_canonical_decomposition(encode_canonical_decomposition(decomposition))
        == decomposition
    )


def test_receipt_cannot_claim_authority_flags() -> None:
    receipt = _search()
    payload = receipt.to_dict()
    payload["can_authorize_completion"] = True
    payload["receipt_cid"] = cid_for_dag_json(
        {key: value for key, value in payload.items() if key != "receipt_cid"}
    )
    with pytest.raises(ProofAdapterError, match="can_authorize_completion"):
        BoundedProofSearchReceipt.from_dict(payload)
    payload = receipt.to_dict()
    payload["adapter_is_nomination_only"] = False
    payload["receipt_cid"] = cid_for_dag_json(
        {key: value for key, value in payload.items() if key != "receipt_cid"}
    )
    with pytest.raises(ProofAdapterError, match="nomination_only"):
        BoundedProofSearchReceipt.from_dict(payload)
    payload = receipt.to_dict()
    payload["claims_general_equivalence"] = True
    payload["receipt_cid"] = cid_for_dag_json(
        {key: value for key, value in payload.items() if key != "receipt_cid"}
    )
    with pytest.raises(ProofAdapterError, match="general Python equivalence"):
        BoundedProofSearchReceipt.from_dict(payload)


def test_identity_excludes_observational_fields() -> None:
    receipt = _search()
    payload = receipt.to_dict()
    assert not (IDENTITY_EXCLUDED_FIELDS & set(payload))
    dirty = dict(payload)
    dirty["timestamp"] = "now"
    with pytest.raises(ProofAdapterError, match="observational"):
        BoundedProofSearchReceipt.from_dict(dirty)


def test_empty_validation_commands_fail_closed() -> None:
    with pytest.raises(ProofAdapterError, match="validation_commands"):
        _lower(selection=_selection(validation_commands=[]))


def test_unapplied_wave_fails_closed() -> None:
    with pytest.raises(ProofAdapterError, match="applied"):
        _lower(wave=_wave(status="rolled_back"))


def test_packet_cid_must_bind_wave_and_selection() -> None:
    with pytest.raises(ProofAdapterError, match="packet_cid"):
        _lower(selection=_selection(packet_cid=_cid("other-packet")))


def test_kernel_checked_reconstruction_round_trip() -> None:
    reconstruction = reconstruct_proof(
        _artifact(),
        tree_id=TREE_ID,
        corpus_revision=CORPUS_REVISION,
        environment_id=ENVIRONMENT_ID,
        toolchain_id=TOOLCHAIN_ID,
    )
    assert reconstruction.kernel_checked is True
    restored = ProofReconstruction.from_dict(reconstruction.to_dict())
    assert restored == reconstruction
    with pytest.raises(ProofAdapterError, match="kernel_checked"):
        reconstruct_proof(
            _artifact(kernel_checked=False),
            tree_id=TREE_ID,
            corpus_revision=CORPUS_REVISION,
            environment_id=ENVIRONMENT_ID,
            toolchain_id=TOOLCHAIN_ID,
        )
