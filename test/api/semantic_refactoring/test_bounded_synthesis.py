"""Independent contract tests for SPAR-032 bounded CEGIS/CEGAR synthesis."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.utils.cid_utils import cid_for_bytes
from ipfs_accelerate_py.agent_supervisor.semantic_refactoring.bounded_synthesis import (
    ADAPTER_CANDIDATE_INTERFACE,
    ANALYZER_ID,
    AUTHORITY,
    AUTHORITY_OWNER,
    BOUNDED_SYNTHESIS_RECEIPT_INTERFACE,
    BOUNDARY_ADAPTER_SYNTHESIZER_INTERFACE,
    CANDIDATES_REMAIN_UNPROMOTED,
    CANDIDATE_CANNOT_ADMIT_PROOFS,
    DECLARED_ADAPTER_ARTIFACT_KINDS,
    DECLARED_ROUND_MODES,
    DECLARED_ROUND_OUTCOMES,
    DECLARED_SYNTHESIS_STATUSES,
    DECLARED_TEMPLATE_KINDS,
    DRY_RUN_IS_DETERMINISTIC,
    DRY_RUN_MUTATES,
    DUCKLAKE_IS_AUTHORITY,
    FORBIDDEN_SYNTHESIS_NAMES,
    GENERAL_PYTHON_EQUIVALENCE_CLAIMED,
    GOAL_ID,
    GUESSED_AXIOMS_REJECTED,
    IDENTITY_EXCLUDED_FIELDS,
    IMPLICIT_INSTALL_FORBIDDEN,
    IMPLICIT_NETWORK_FORBIDDEN,
    INCOMPLETE_CONTRACTS_ARE_TYPED_TERMINALS,
    MARKDOWN_IS_NOT_COMPLETION,
    MODEL_OUTPUT_IS_PROPOSAL_ONLY,
    NETWORK_DENIED,
    NETWORK_DENY,
    PROGRAM,
    PROJECTION_CLUSTERING_IS_AUTHORITY,
    RAW_COUNTERMODEL_CANNOT_REFUTE,
    RAW_SOURCE_REQUIRED,
    SOFT_SIGNALS_CANNOT_OVERRIDE_HARD_CONSTRAINTS,
    SYNTHESIS_CAN_AUTHORIZE_COMPLETION,
    SYNTHESIS_CAN_AUTHORIZE_TRANSITION,
    SYNTHESIS_CAN_CREATE_AUTHORITY,
    SYNTHESIS_CAN_CREATE_PROOF_AUTHORITY,
    SYNTHESIS_CONTRACT_VERSION,
    SYNTHESIS_COUNTEREXAMPLE_INTERFACE,
    SYNTHESIS_EXAMPLE_INTERFACE,
    SYNTHESIS_OBLIGATION_INTERFACE,
    SYNTHESIS_ROUND_INTERFACE,
    SYNTHESIZED_CANDIDATES_REENTER_VALIDATION,
    SYNTHESIZER_IS_NOMINATION_ONLY,
    TASK_ID,
    TEMPLATE_SELECTOR_DETERMINISTIC,
    TEST_PASS_IS_NOT_COMPLETION,
    TEST_PASS_IS_NOT_PROOF,
    UNKNOWN_REMAINS_UNKNOWN,
    VALIDATION_REENTRY_INTERFACE,
    VECTOR_SIMILARITY_IS_AUTHORITY,
    WORKER_SELF_APPROVAL,
    AdapterArtifactKind,
    BoundedSynthesisError,
    BoundedSynthesisReceipt,
    BoundaryAdapterSynthesizer,
    RoundMode,
    RoundOutcome,
    SynthesisStatus,
    TemplateKind,
    ValidationReentryStatus,
    assert_not_competing_capsule_family,
    bounded_synthesis_cid_profile,
    bounded_synthesis_descriptor,
    compile_synthesis_counterexample,
    compile_synthesis_example,
    compile_synthesis_obligation,
    decode_canonical_receipt,
    dry_run_bounded_synthesis,
    encode_canonical_receipt,
    enumerate_templates,
    provider_free_exports,
    run_bounded_synthesis,
)


ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    ROOT
    / "ipfs_accelerate_py"
    / "agent_supervisor"
    / "semantic_refactoring"
    / "bounded_synthesis.py"
)
TEST_PATH = Path(__file__).resolve()
WRITE_SCOPE = (
    "ipfs_accelerate_py/agent_supervisor/semantic_refactoring/bounded_synthesis.py",
    "test/api/semantic_refactoring/test_bounded_synthesis.py",
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
OBLIGATION_ID = "obl:inputs"
SOURCE_CID_LABEL = "source"


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
        "raw_source_cids": [_cid(SOURCE_CID_LABEL)],
        "write_paths": list(WRITE_PATHS),
        "validation_commands": list(VALIDATION),
        "raw_source_required": True,
        "adapter_is_nomination_only": True,
        "datasets_owns_selection": True,
    }
    fields.update(overrides)
    return fields


def _example(**overrides: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "example_id": "ex:boundary",
        "kind": "boundary",
        "obligation_id": OBLIGATION_ID,
        "tree_id": TREE_ID,
        "source_cid": _cid(SOURCE_CID_LABEL),
        "polarity": "satisfy",
        "template_hint": "",
        "source_authority": "specification",
    }
    fields.update(overrides)
    return fields


def _counterexample(**overrides: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "counterexample_id": "ce:identity",
        "kind": "boundary",
        "obligation_id": OBLIGATION_ID,
        "tree_id": TREE_ID,
        "source_cid": _cid("replay"),
        "source_authority": "replayed_counterexample",
        "replayed": True,
        "conflicts_with": "identity",
        "abstraction_gap": False,
        "predicate_id": "",
    }
    fields.update(overrides)
    return fields


def _obligation(**overrides: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "obligation_id": OBLIGATION_ID,
        "kind": "input",
        "tree_id": TREE_ID,
        "contract_cid": _cid("contract"),
        "finite": True,
        "required": True,
        "status": "lowered",
    }
    fields.update(overrides)
    return fields


def _validate_ok(**request: Any) -> dict[str, Any]:
    return {
        "status": "validated",
        "result_cid": _cid("validation"),
        "tree_id": request["tree_id"],
        "network": request["network"],
    }


def _validate_reject(**request: Any) -> dict[str, Any]:
    return {
        "status": "rejected",
        "result_cid": _cid("validation-reject"),
        "tree_id": request["tree_id"],
        "reason_code": "translation_rejected",
        "network": request["network"],
    }


def _run(**overrides: Any) -> BoundedSynthesisReceipt:
    fields: dict[str, Any] = {
        "wave": _wave(),
        "selection": _selection(),
        "examples": [_example()],
        "obligations": [_obligation()],
        "validate": _validate_ok,
    }
    fields.update(overrides)
    return run_bounded_synthesis(**fields)


def test_owned_paths_and_task_identity_are_exact() -> None:
    assert TASK_ID == "SPAR-032"
    assert GOAL_ID == "SPAR-G053"
    assert PROGRAM == "semantic-preserving-autonomous-remodularization-v1"
    assert BOUNDARY_ADAPTER_SYNTHESIZER_INTERFACE == "BoundaryAdapterSynthesizer@1"
    assert SYNTHESIS_EXAMPLE_INTERFACE == "SynthesisExample@1"
    assert SYNTHESIS_COUNTEREXAMPLE_INTERFACE == "SynthesisCounterexample@1"
    assert SYNTHESIS_OBLIGATION_INTERFACE == "SynthesisObligation@1"
    assert ADAPTER_CANDIDATE_INTERFACE == "AdapterCandidate@1"
    assert SYNTHESIS_ROUND_INTERFACE == "SynthesisRound@1"
    assert VALIDATION_REENTRY_INTERFACE == "ValidationReentry@1"
    assert BOUNDED_SYNTHESIS_RECEIPT_INTERFACE == "BoundedSynthesisReceipt@1"
    assert SYNTHESIS_CONTRACT_VERSION == "1"
    assert ANALYZER_ID.endswith("bounded_synthesis@1")
    assert MODULE_PATH.is_file()
    assert TEST_PATH.is_file()
    for relative in WRITE_SCOPE:
        assert (ROOT / relative).is_file()


def test_authority_flags_cannot_self_authorize() -> None:
    assert AUTHORITY == "bounded synthesis"
    assert AUTHORITY_OWNER == "ipfs_accelerate_py"
    assert SYNTHESIS_CAN_AUTHORIZE_COMPLETION is False
    assert SYNTHESIS_CAN_AUTHORIZE_TRANSITION is False
    assert SYNTHESIS_CAN_CREATE_AUTHORITY is False
    assert SYNTHESIS_CAN_CREATE_PROOF_AUTHORITY is False
    assert VECTOR_SIMILARITY_IS_AUTHORITY is False
    assert PROJECTION_CLUSTERING_IS_AUTHORITY is False
    assert MODEL_OUTPUT_IS_PROPOSAL_ONLY is True
    assert TEST_PASS_IS_NOT_COMPLETION is True
    assert TEST_PASS_IS_NOT_PROOF is True
    assert CANDIDATE_CANNOT_ADMIT_PROOFS is True
    assert RAW_COUNTERMODEL_CANNOT_REFUTE is True
    assert MARKDOWN_IS_NOT_COMPLETION is True
    assert WORKER_SELF_APPROVAL is False
    assert DUCKLAKE_IS_AUTHORITY is False
    assert SOFT_SIGNALS_CANNOT_OVERRIDE_HARD_CONSTRAINTS is True
    assert SYNTHESIZER_IS_NOMINATION_ONLY is True
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
    assert SYNTHESIZED_CANDIDATES_REENTER_VALIDATION is True
    assert CANDIDATES_REMAIN_UNPROMOTED is True
    assert DECLARED_SYNTHESIS_STATUSES == {
        "nominated",
        "incomplete",
        "refuted",
        "blocked",
        "unknown",
        "rejected",
        "timeout",
    }
    assert DECLARED_ROUND_MODES == {"cegis", "cegar", "validation"}
    assert DECLARED_ROUND_OUTCOMES == {
        "covered",
        "refuted",
        "refined",
        "uncovered",
        "validated",
        "validation_rejected",
        "timeout",
        "unknown",
    }
    assert TemplateKind.IDENTITY.value in DECLARED_TEMPLATE_KINDS
    assert AdapterArtifactKind.BOUNDARY_ADAPTER.value in DECLARED_ADAPTER_ARTIFACT_KINDS
    profile = bounded_synthesis_cid_profile()
    assert profile["codec"] == "dag-json"
    assert "not universal meaning" in profile["rule"]


def test_module_defines_predicted_symbols_not_capsule_family() -> None:
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    names = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
    assert "BoundaryAdapterSynthesizer" in names
    assert "AdapterCandidate" in names
    assert "BoundedSynthesisReceipt" in names
    assert "SynthesisExample" in names
    assert "SynthesisCounterexample" in names
    assert "ValidationReentry" in names
    for capsule in CAPSULE_TYPES:
        assert capsule not in names
    assert_not_competing_capsule_family()
    exports = provider_free_exports()
    assert "BoundaryAdapterSynthesizer" in exports
    assert "run_bounded_synthesis" in exports
    assert "dry_run_bounded_synthesis" in exports
    assert "reenter_translation_validation" in exports
    assert "enumerate_templates" in exports


def test_protected_paths_are_not_owned_write_scope() -> None:
    owned = set(WRITE_SCOPE)
    for relative in PROTECTED_PATHS:
        assert relative not in owned
        assert (ROOT / relative).exists()


def test_module_does_not_claim_proof_authority_or_open_network() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    assert not (names & FORBIDDEN_SYNTHESIS_NAMES)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    assert not any("tactician_hammer_coordinator" in name for name in imported)
    assert not any("procedure_compiler" in name for name in imported)
    descriptor = bounded_synthesis_descriptor()
    assert descriptor["interface"] == BOUNDARY_ADAPTER_SYNTHESIZER_INTERFACE
    assert descriptor["network"] == NETWORK_DENY
    assert descriptor["nomination_only"] is True
    assert descriptor["claims_general_equivalence"] is False
    assert descriptor["candidate_cannot_admit_proofs"] is True
    assert descriptor["raw_countermodel_cannot_refute"] is True
    assert descriptor["unknown_remains_unknown"] is True
    assert descriptor["guessed_axioms_rejected"] is True
    assert descriptor["synthesized_candidates_reenter_validation"] is True
    assert descriptor["candidates_remain_unpromoted"] is True
    assert descriptor["template_selector"] == TEMPLATE_SELECTOR_DETERMINISTIC
    forbids = set(descriptor["forbids"])
    assert "claim_general_equivalence" in forbids
    assert "promote_candidate" in forbids
    assert "open_network" in forbids


def test_identity_template_nominates_after_validation_reentry() -> None:
    receipt = _run()
    assert receipt.status == SynthesisStatus.NOMINATED.value
    assert receipt.synthesizer_is_nomination_only is True
    assert receipt.can_authorize_completion is False
    assert receipt.can_authorize_transition is False
    assert receipt.can_create_proof_authority is False
    assert receipt.network == NETWORK_DENY
    assert receipt.mutated is False
    assert receipt.deterministic is True
    assert receipt.claims_general_equivalence is False
    assert receipt.candidates_remain_unpromoted is True
    assert len(receipt.candidates) == 1
    assert receipt.candidates[0].template_id == TemplateKind.IDENTITY.value
    assert receipt.candidates[0].kind == AdapterArtifactKind.BOUNDARY_ADAPTER.value
    assert receipt.candidates[0].unpromoted is True
    assert receipt.reentry is not None
    assert receipt.reentry.status == ValidationReentryStatus.VALIDATED.value
    assert receipt.reentry.can_authorize_completion is False
    assert receipt.typed_terminal is False
    restored = decode_canonical_receipt(encode_canonical_receipt(receipt))
    assert restored == receipt
    adapter = BoundaryAdapterSynthesizer()
    again = adapter.synthesize(
        wave=_wave(),
        selection=_selection(),
        examples=[_example()],
        obligations=[_obligation()],
        validate=_validate_ok,
    )
    assert again.receipt_cid == receipt.receipt_cid


def test_replayed_counterexample_refutes_identity_and_selects_wrapper() -> None:
    receipt = _run(counterexamples=[_counterexample()])
    assert receipt.status == SynthesisStatus.NOMINATED.value
    assert receipt.candidates[-1].template_id == TemplateKind.WRAPPER.value
    assert any(
        item.outcome == RoundOutcome.REFUTED.value
        and item.template_id == TemplateKind.IDENTITY.value
        for item in receipt.rounds
    )
    assert any(item.mode == RoundMode.CEGIS.value for item in receipt.rounds)


def test_raw_countermodel_cannot_refute_until_replay() -> None:
    receipt = _run(
        counterexamples=[
            _counterexample(
                counterexample_id="ce:raw",
                source_authority="countermodel",
                replayed=False,
                conflicts_with="identity",
                source_cid=_cid("raw-cm"),
            )
        ]
    )
    assert receipt.candidates[0].template_id == TemplateKind.IDENTITY.value
    assert receipt.status == SynthesisStatus.NOMINATED.value
    with pytest.raises(BoundedSynthesisError, match="raw countermodels cannot refute"):
        compile_synthesis_counterexample(
            **_counterexample(source_authority="countermodel", replayed=True)
        )


def test_cegar_refines_abstraction_then_nominates() -> None:
    receipt = _run(
        counterexamples=[
            _counterexample(
                counterexample_id="ce:gap",
                conflicts_with="",
                abstraction_gap=True,
                predicate_id="pred:type",
            )
        ]
    )
    assert any(
        item.mode == RoundMode.CEGAR.value and item.outcome == RoundOutcome.REFINED.value
        for item in receipt.rounds
    )
    assert any(item.distinguishing is True for item in receipt.abstractions)
    assert receipt.status == SynthesisStatus.NOMINATED.value
    assert receipt.candidates[-1].template_id == TemplateKind.IDENTITY.value


def test_missing_validation_reentry_is_incomplete_not_completion() -> None:
    receipt = _run(validate=None)
    assert receipt.status == SynthesisStatus.INCOMPLETE.value
    assert receipt.reentry is not None
    assert receipt.reentry.status == ValidationReentryStatus.MISSING.value
    assert receipt.reentry.reason_code == "validation_reentry_required"
    assert receipt.candidates[0].unpromoted is True
    assert receipt.can_authorize_completion is False
    assert receipt.typed_terminal is True


def test_validation_rejection_becomes_immutable_counterexample() -> None:
    seen: list[str] = []

    def validate(**request: Any) -> dict[str, Any]:
        seen.append(request["candidate"]["template_id"])
        if request["candidate"]["template_id"] == TemplateKind.IDENTITY.value:
            return _validate_reject(**request)
        return _validate_ok(**request)

    receipt = _run(validate=validate)
    assert TemplateKind.IDENTITY.value in seen
    assert TemplateKind.WRAPPER.value in seen
    assert receipt.candidates[-1].template_id == TemplateKind.WRAPPER.value
    assert receipt.status == SynthesisStatus.NOMINATED.value
    assert any(
        item.source_authority == "validation_failure" for item in receipt.counterexamples
    )
    assert any(
        item.outcome == RoundOutcome.VALIDATION_REJECTED.value for item in receipt.rounds
    )


def test_unknown_remains_unknown_without_examples_or_obligations() -> None:
    receipt = run_bounded_synthesis(
        wave=_wave(),
        selection=_selection(),
        examples=(),
        obligations=(),
        validate=_validate_ok,
    )
    assert receipt.status == SynthesisStatus.UNKNOWN.value
    assert receipt.unknown_remains_unknown is True
    assert receipt.candidates == ()
    assert receipt.typed_terminal is True


def test_guard_protocol_state_and_initialization_templates() -> None:
    cases = (
        ("guard", TemplateKind.GUARD.value, AdapterArtifactKind.GUARD.value, "condition"),
        (
            "protocol",
            TemplateKind.PROTOCOL.value,
            AdapterArtifactKind.PROTOCOL.value,
            "effect",
        ),
        (
            "state",
            TemplateKind.PROJECTION.value,
            AdapterArtifactKind.STATE_MAPPING.value,
            "state",
        ),
        (
            "initialization",
            TemplateKind.INITIALIZATION.value,
            AdapterArtifactKind.INITIALIZATION_REPAIR.value,
            "resource",
        ),
    )
    for example_kind, template_id, artifact_kind, obligation_kind in cases:
        receipt = _run(
            examples=[_example(kind=example_kind, example_id=f"ex:{example_kind}")],
            obligations=[_obligation(kind=obligation_kind)],
        )
        assert receipt.status == SynthesisStatus.NOMINATED.value
        assert receipt.candidates[0].template_id == template_id
        assert receipt.candidates[0].kind == artifact_kind


def test_incomplete_and_unsupported_obligations_are_typed_terminals() -> None:
    with pytest.raises(BoundedSynthesisError, match="incomplete SPAR-030"):
        _run(obligations=[_obligation(status="incomplete")])
    with pytest.raises(BoundedSynthesisError, match="unsupported required"):
        _run(obligations=[_obligation(status="unsupported")])
    with pytest.raises(BoundedSynthesisError, match="non-finite"):
        _run(obligations=[_obligation(finite=False)])


def test_guessed_and_vector_examples_cannot_admit() -> None:
    with pytest.raises(BoundedSynthesisError, match="cannot admit"):
        compile_synthesis_example(**_example(source_authority="guessed"))
    with pytest.raises(BoundedSynthesisError, match="cannot admit"):
        compile_synthesis_example(**_example(source_authority="vector_candidate"))
    with pytest.raises(BoundedSynthesisError, match="cannot admit"):
        compile_synthesis_example(**_example(source_authority="model_hypothesis"))
    with pytest.raises(BoundedSynthesisError, match="cannot admit"):
        _run(vector_evidence={"evidence_class": "heuristic", "admit_proof": True})


def test_vectors_cannot_admit_or_suppress_raw_source() -> None:
    with pytest.raises(BoundedSynthesisError, match="raw-source"):
        _run(
            vector_evidence={
                "evidence_class": "vector_candidate",
                "suppress_raw_source": True,
            }
        )
    with pytest.raises(BoundedSynthesisError, match="cannot admit"):
        _run(
            vector_evidence={
                "evidence_class": "model_hypothesis",
                "admit_equivalence": True,
            }
        )


def test_missing_predecessors_fail_closed() -> None:
    with pytest.raises(BoundedSynthesisError, match="SPAR-025"):
        run_bounded_synthesis(
            wave=None,
            selection=_selection(),
            examples=[_example()],
        )
    with pytest.raises(BoundedSynthesisError, match="SPAR-026"):
        run_bounded_synthesis(
            wave=_wave(),
            selection=None,
            examples=[_example()],
        )


def test_tree_mismatch_fails_closed() -> None:
    with pytest.raises(BoundedSynthesisError, match="SPAR-026 tree_id"):
        _run(selection=_selection(tree_id=OTHER_TREE))
    with pytest.raises(BoundedSynthesisError, match="example tree_id"):
        _run(examples=[_example(tree_id=OTHER_TREE)])


def test_write_paths_must_match_and_stay_exact() -> None:
    with pytest.raises(BoundedSynthesisError, match="write_paths"):
        _run(wave=_wave(write_paths=["pkg/mod.py"]))
    with pytest.raises(BoundedSynthesisError, match="unrestricted scope"):
        _run(selection=_selection(write_paths=[]))
    with pytest.raises(BoundedSynthesisError, match="unrestricted scope"):
        _run(selection=_selection(write_paths=["pkg/*.py"]))
    with pytest.raises(BoundedSynthesisError, match="unrestricted scope"):
        _run(selection=_selection(write_paths=["/tmp/pkg/mod.py"]))
    with pytest.raises(BoundedSynthesisError, match="unrestricted scope"):
        _run(selection=_selection(write_paths=["pkg/../secret.py"]))


def test_missing_raw_source_is_typed_terminal() -> None:
    with pytest.raises(BoundedSynthesisError, match="raw source"):
        _run(selection=_selection(raw_source_cids=[]))
    with pytest.raises(BoundedSynthesisError, match="declared raw source"):
        _run(examples=[_example(source_cid=_cid("other-source"))])


def test_body_free_artifacts_reject_source_dumps() -> None:
    with pytest.raises(BoundedSynthesisError, match="body-free"):
        _run(examples=[_example(source="def adapter():\n    return 1\n")])
    with pytest.raises(BoundedSynthesisError, match="body-free"):
        compile_synthesis_obligation(**_obligation(proof_body="theorem T : True"))


def test_network_is_denied() -> None:
    with pytest.raises(BoundedSynthesisError, match="network is denied"):
        _run(network="allow")


def test_validate_injectable_remains_nomination_only() -> None:
    seen: dict[str, str] = {}

    def validate(**request: Any) -> dict[str, Any]:
        seen["network"] = request["network"]
        seen["tree_id"] = request["tree_id"]
        return _validate_ok(**request)

    receipt = _run(validate=validate)
    assert seen["network"] == NETWORK_DENY
    assert seen["tree_id"] == TREE_ID
    assert receipt.status == SynthesisStatus.NOMINATED.value
    assert receipt.can_create_authority is False
    assert receipt.projection_is_authority is False


def test_spar030_replay_is_ingested_and_raw_step_cannot_refute() -> None:
    receipt = _run(
        counterexamples=(),
        proof_receipt={
            "tree_id": TREE_ID,
            "replays": [
                {
                    "obligation_id": OBLIGATION_ID,
                    "replay_cid": _cid("replay"),
                    "replayed": True,
                    "conflicts_with": "identity",
                }
            ],
            "steps": [
                {
                    "obligation_id": OBLIGATION_ID,
                    "kind": "countermodel",
                    "replayed": False,
                    "artifact_cid": _cid("raw-cm"),
                }
            ],
        },
    )
    assert any(item.conflicts_with == "identity" for item in receipt.counterexamples)
    assert any(
        item.source_authority == "countermodel" and item.replayed is False
        for item in receipt.counterexamples
    )
    assert receipt.candidates[-1].template_id == TemplateKind.WRAPPER.value


def test_spar028_mismatch_requires_independent_observation() -> None:
    with pytest.raises(BoundedSynthesisError, match="independently observed"):
        _run(
            differential_receipt={
                "tree_id": TREE_ID,
                "comparisons": [
                    {
                        "dimension": "state",
                        "status": "mismatch",
                        "comparison_cid": _cid("diff"),
                    }
                ],
            }
        )
    receipt = _run(
        examples=[_example(kind="state", example_id="ex:state")],
        obligations=[_obligation(kind="state")],
        differential_receipt={
            "tree_id": TREE_ID,
            "comparisons": [
                {
                    "dimension": "state",
                    "status": "mismatch",
                    "comparison_cid": _cid("diff"),
                    "independently_observed": True,
                    "obligation_id": OBLIGATION_ID,
                    "conflicts_with": "projection",
                }
            ],
        },
    )
    assert receipt.status in {
        SynthesisStatus.REFUTED.value,
        SynthesisStatus.INCOMPLETE.value,
        SynthesisStatus.TIMEOUT.value,
    }
    assert any(
        item.source_authority == "differential_mismatch"
        for item in receipt.counterexamples
    )


def test_timeout_is_a_typed_non_completing_terminal() -> None:
    receipt = _run(
        counterexamples=[_counterexample()],
        validate=_validate_reject,
        max_rounds=1,
    )
    assert receipt.status in {
        SynthesisStatus.TIMEOUT.value,
        SynthesisStatus.INCOMPLETE.value,
        SynthesisStatus.REFUTED.value,
    }
    assert receipt.can_authorize_completion is False
    assert receipt.typed_terminal is True


def test_dry_run_is_deterministic_and_does_not_mutate() -> None:
    first = dry_run_bounded_synthesis(
        wave=_wave(),
        selection=_selection(),
        examples=[_example()],
        obligations=[_obligation()],
        validate=_validate_ok,
    )
    second = BoundaryAdapterSynthesizer().dry_run(
        wave=_wave(),
        selection=_selection(),
        examples=[_example()],
        obligations=[_obligation()],
        validate=_validate_ok,
    )
    assert first.receipt_cid == second.receipt_cid
    assert first.mutated is False
    assert first.deterministic is True
    with pytest.raises(BoundedSynthesisError, match="cannot mutate"):
        run_bounded_synthesis(
            wave=_wave(),
            selection=_selection(),
            examples=[_example()],
            mutate=True,
        )


def test_template_catalog_is_finite_and_deterministic() -> None:
    templates = enumerate_templates()
    assert tuple(item["template_id"] for item in templates) == (
        "identity",
        "wrapper",
        "projection",
        "guard",
        "protocol",
        "initialization",
    )
    assert templates == enumerate_templates()


def test_compile_helpers_are_content_addressed_and_body_free() -> None:
    example = compile_synthesis_example(**_example())
    again = compile_synthesis_example(**_example())
    assert example.example_cid == again.example_cid
    obligation = compile_synthesis_obligation(**_obligation())
    assert obligation.status == "lowered"
    assert obligation.finite is True
    assert "timestamp" in IDENTITY_EXCLUDED_FIELDS
    with pytest.raises(BoundedSynthesisError, match="observational"):
        _run(wave=_wave(timestamp="now"))
