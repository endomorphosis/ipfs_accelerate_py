"""Independent contract tests for SPAR-027 translation-validation orchestrator."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.utils.cid_utils import cid_for_bytes, cid_for_dag_json
from ipfs_accelerate_py.agent_supervisor.semantic_refactoring.translation_validation import (
    ANALYZER_ID,
    AUTHORITY,
    AUTHORITY_OWNER,
    CLAIM_IS_NOMINATION_ONLY,
    DECLARED_DIMENSIONS,
    DEFAULT_REQUIRED_DIMENSIONS,
    DIMENSION_ADMITTED_EVIDENCE,
    DRY_RUN_IS_DETERMINISTIC,
    DRY_RUN_MUTATES,
    DUCKLAKE_IS_AUTHORITY,
    FORBIDDEN_EQUIVALENCE_NAMES,
    GOAL_ID,
    IDENTITY_EXCLUDED_FIELDS,
    MARKDOWN_IS_NOT_COMPLETION,
    MODEL_OUTPUT_IS_PROPOSAL_ONLY,
    ORCHESTRATOR_IS_NOMINATION_ONLY,
    PROCEDURE_COMPILER_IS_NOT_EXTRACTION_RECEIPT,
    PROGRAM,
    PROOF_CANDIDATE_CANNOT_ADMIT_PROOFS,
    PROJECTION_CLUSTERING_IS_AUTHORITY,
    RAW_SOURCE_REQUIRED,
    REFACTOR_EQUIVALENCE_CLAIM_INTERFACE,
    RUNTIME_OBSERVATION_IS_NOT_STATIC_FACT,
    SOFT_SIGNALS_CANNOT_OVERRIDE_HARD_CONSTRAINTS,
    TASK_ID,
    TEST_PASS_IS_NOT_COMPLETION,
    TEST_PASS_IS_NOT_PROOF,
    TRANSLATION_VALIDATION_ORCHESTRATOR_INTERFACE,
    TRANSLATION_VALIDATION_REQUEST_INTERFACE,
    TRANSLATION_VALIDATION_RESULT_INTERFACE,
    VALIDATION_CAN_AUTHORIZE_COMPLETION,
    VALIDATION_CAN_AUTHORIZE_TRANSITION,
    VALIDATION_CAN_CREATE_AUTHORITY,
    VALIDATION_CLAIMS_GENERAL_PYTHON_EQUIVALENCE,
    VALIDATION_COLLAPSES_EVIDENCE_CLASSES,
    VALIDATION_CONTRACT_VERSION,
    VECTOR_SIMILARITY_IS_AUTHORITY,
    WORKER_SELF_APPROVAL,
    DimensionStatus,
    DimensionVerdict,
    EvidenceClass,
    ObservationProfile,
    RefactorEquivalenceClaim,
    TranslationValidationError,
    TranslationValidationOrchestrator,
    TranslationValidationRequest,
    TranslationValidationResult,
    ValidationDimension,
    ValidationStatus,
    admitted_evidence_classes,
    assert_evidence_class_admitted,
    assert_not_competing_capsule_family,
    compile_equivalence_claim,
    compile_translation_validation_request,
    decode_canonical_claim,
    decode_canonical_request,
    decode_canonical_result,
    dry_run_translation_validation,
    encode_canonical_claim,
    encode_canonical_request,
    encode_canonical_result,
    provider_free_exports,
    translation_validation_cid_profile,
    translation_validation_descriptor,
    validate_translation,
)


ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    ROOT
    / "ipfs_accelerate_py"
    / "agent_supervisor"
    / "semantic_refactoring"
    / "translation_validation.py"
)
TEST_PATH = Path(__file__).resolve()
WRITE_SCOPE = (
    "ipfs_accelerate_py/agent_supervisor/semantic_refactoring/translation_validation.py",
    "test/api/semantic_refactoring/test_translation_validation.py",
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
ORIGINAL_SOURCE = "def leaf():\n    return 1\n"
CANDIDATE_SOURCE = "def leaf():\n    return 1\n"
INVALID_SOURCE = "def leaf(\n"


def _cid(label: str) -> str:
    return cid_for_bytes(label.encode("utf-8"))


def _packet(**overrides: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "tree_id": TREE_ID,
        "packet_cid": _cid("packet"),
        "preimage": {
            "source_cids": [_cid("source")],
            "environment_cid": _cid("env"),
            "graph_cid": _cid("graph"),
        },
        "write_paths": list(WRITE_PATHS),
        "validation_commands": list(VALIDATION),
    }
    fields.update(overrides)
    return fields


def _wave(**overrides: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "tree_id": TREE_ID,
        "receipt_cid": _cid("wave"),
        "packet_cids": [_cid("packet")],
        "write_paths": list(WRITE_PATHS),
        "after_source_cids": [_cid("candidate")],
        "status": "applied",
        "mutated": False,
        "writes_repository": False,
        "advance_accepted_roots": False,
        "executor_is_nomination_only": True,
    }
    fields.update(overrides)
    return fields


def _selection(**overrides: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "tree_id": TREE_ID,
        "validation_selection_cid": _cid("selection"),
        "packet_cid": _cid("packet"),
        "effective_fallback": "none",
        "full_suite_required": False,
        "raw_source_cids": [_cid("source")],
        "write_paths": list(WRITE_PATHS),
        "validation_commands": list(VALIDATION),
        "selected_pytest_node_ids": ["tests/test_mod.py::test_a"],
        "selected_proof_ids": ["proof:mod"],
    }
    fields.update(overrides)
    return fields


def _evidence(
    dimension: str,
    evidence_class: str,
    *,
    status: str = DimensionStatus.PASS.value,
    label: str | None = None,
    full_suite_executed: bool = False,
) -> dict[str, Any]:
    return {
        "dimension": dimension,
        "evidence_class": evidence_class,
        "evidence_cid": _cid(label or f"{dimension}:{evidence_class}"),
        "status": status,
        "full_suite_executed": full_suite_executed,
    }


def _all_passing_evidence() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for dimension in DEFAULT_REQUIRED_DIMENSIONS:
        admitted = sorted(admitted_evidence_classes(dimension))
        evidence_class = admitted[0]
        items.append(
            _evidence(
                dimension,
                evidence_class,
                full_suite_executed=dimension == ValidationDimension.TESTS.value,
            )
        )
    return items


def _profile(*required: str, optional: tuple[str, ...] = ()) -> ObservationProfile:
    return ObservationProfile(
        required_dimensions=required or DEFAULT_REQUIRED_DIMENSIONS,
        optional_dimensions=optional,
    )


def _compile(**overrides: Any) -> TranslationValidationRequest:
    fields: dict[str, Any] = {
        "packet": _packet(),
        "wave": _wave(),
        "selection": _selection(),
        "dimension_evidence": _all_passing_evidence(),
        "observation_profile": _profile(),
        "candidate_source_cids": [_cid("candidate")],
    }
    fields.update(overrides)
    return compile_translation_validation_request(**fields)


def test_owned_paths_and_task_identity_are_exact() -> None:
    assert TASK_ID == "SPAR-027"
    assert GOAL_ID == "SPAR-G051"
    assert PROGRAM == "semantic-preserving-autonomous-remodularization-v1"
    assert TRANSLATION_VALIDATION_REQUEST_INTERFACE == "TranslationValidationRequest@1"
    assert TRANSLATION_VALIDATION_RESULT_INTERFACE == "TranslationValidationResult@1"
    assert REFACTOR_EQUIVALENCE_CLAIM_INTERFACE == "RefactorEquivalenceClaim@1"
    assert (
        TRANSLATION_VALIDATION_ORCHESTRATOR_INTERFACE
        == "TranslationValidationOrchestrator@1"
    )
    assert VALIDATION_CONTRACT_VERSION == "1"
    assert ANALYZER_ID.endswith("translation_validation@1")
    assert MODULE_PATH.is_file()
    assert TEST_PATH.is_file()
    for relative in WRITE_SCOPE:
        assert (ROOT / relative).is_file()


def test_authority_flags_cannot_self_authorize() -> None:
    assert AUTHORITY == "validation orchestration"
    assert AUTHORITY_OWNER == "ipfs_accelerate_py"
    assert VALIDATION_CAN_AUTHORIZE_COMPLETION is False
    assert VALIDATION_CAN_AUTHORIZE_TRANSITION is False
    assert VALIDATION_CAN_CREATE_AUTHORITY is False
    assert VALIDATION_CLAIMS_GENERAL_PYTHON_EQUIVALENCE is False
    assert VALIDATION_COLLAPSES_EVIDENCE_CLASSES is False
    assert PROCEDURE_COMPILER_IS_NOT_EXTRACTION_RECEIPT is True
    assert VECTOR_SIMILARITY_IS_AUTHORITY is False
    assert PROJECTION_CLUSTERING_IS_AUTHORITY is False
    assert MODEL_OUTPUT_IS_PROPOSAL_ONLY is True
    assert TEST_PASS_IS_NOT_COMPLETION is True
    assert TEST_PASS_IS_NOT_PROOF is True
    assert RUNTIME_OBSERVATION_IS_NOT_STATIC_FACT is True
    assert PROOF_CANDIDATE_CANNOT_ADMIT_PROOFS is True
    assert MARKDOWN_IS_NOT_COMPLETION is True
    assert WORKER_SELF_APPROVAL is False
    assert DUCKLAKE_IS_AUTHORITY is False
    assert SOFT_SIGNALS_CANNOT_OVERRIDE_HARD_CONSTRAINTS is True
    assert ORCHESTRATOR_IS_NOMINATION_ONLY is True
    assert CLAIM_IS_NOMINATION_ONLY is True
    assert RAW_SOURCE_REQUIRED is True
    assert DRY_RUN_IS_DETERMINISTIC is True
    assert DRY_RUN_MUTATES is False
    profile = translation_validation_cid_profile()
    assert profile["codec"] == "dag-json"
    assert "not universal meaning" in profile["rule"]


def test_module_defines_predicted_symbols_not_capsule_family() -> None:
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    names = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
    assert "TranslationValidationRequest" in names
    assert "TranslationValidationResult" in names
    assert "RefactorEquivalenceClaim" in names
    assert "TranslationValidationOrchestrator" in names
    for capsule in CAPSULE_TYPES:
        assert capsule not in names
    assert_not_competing_capsule_family()
    exports = provider_free_exports()
    assert "TranslationValidationRequest" in exports
    assert "TranslationValidationResult" in exports
    assert "RefactorEquivalenceClaim" in exports
    assert "validate_translation" in exports
    assert "compile_equivalence_claim" in exports


def test_protected_paths_are_not_owned_write_scope() -> None:
    owned = set(WRITE_SCOPE)
    for relative in PROTECTED_PATHS:
        assert relative not in owned
        assert (ROOT / relative).exists()


def test_module_does_not_claim_general_python_equivalence() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    assert not (names & FORBIDDEN_EQUIVALENCE_NAMES)
    descriptor = translation_validation_descriptor()
    assert descriptor["interface"] == TRANSLATION_VALIDATION_ORCHESTRATOR_INTERFACE
    assert descriptor["claims_general_python_equivalence"] is False
    assert descriptor["collapses_evidence_classes"] is False
    assert descriptor["procedure_compiler_is_not_extraction_receipt"] is True
    forbids = set(descriptor["forbids"])
    assert "prove_python_equivalence" in forbids
    assert "collapse_evidence" in forbids
    assert "admit_from_vectors" in forbids
    assert "promote_test_to_proof" in forbids


def test_does_not_reuse_procedure_compiler_translation_types() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    assert not any("procedure_compiler" in name for name in imported)
    assert not any("logic_translation_validation" in name for name in imported)
    assert "GeneratedToolCertificate" not in source
    names = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
    assert "TranslationValidation" not in names
    assert "GeneratedToolCertificate" not in names


def test_all_declared_dimensions_have_admitted_evidence_classes() -> None:
    assert set(DEFAULT_REQUIRED_DIMENSIONS) == DECLARED_DIMENSIONS
    assert set(DIMENSION_ADMITTED_EVIDENCE) == DECLARED_DIMENSIONS
    assert EvidenceClass.TEST.value not in admitted_evidence_classes(
        ValidationDimension.PROOFS.value
    )
    assert EvidenceClass.RECONSTRUCTED_PROOF.value not in admitted_evidence_classes(
        ValidationDimension.TESTS.value
    )
    assert EvidenceClass.RUNTIME_OBSERVATION.value not in admitted_evidence_classes(
        ValidationDimension.SYNTAX.value
    )
    with pytest.raises(TranslationValidationError, match="cannot admit"):
        assert_evidence_class_admitted(
            ValidationDimension.PROOFS.value, EvidenceClass.TEST.value
        )
    with pytest.raises(TranslationValidationError, match="proof_candidate"):
        assert_evidence_class_admitted(
            ValidationDimension.PROOFS.value, EvidenceClass.PROOF_CANDIDATE.value
        )


def test_compile_binds_spar025_and_spar026_without_collapsing() -> None:
    request = _compile()
    assert request.tree_id == TREE_ID
    assert request.packet_cid == _cid("packet")
    assert request.wave_receipt_cid == _cid("wave")
    assert request.selection_cid == _cid("selection")
    assert request.original_source_cids == (_cid("source"),)
    assert request.candidate_source_cids == (_cid("candidate"),)
    assert request.write_paths == WRITE_PATHS
    assert request.validation_commands == VALIDATION
    assert request.raw_source_required is True
    assert request.orchestrator_is_nomination_only is True
    assert request.claims_general_python_equivalence is False
    assert request.collapse_evidence_classes is False
    assert request.can_authorize_completion is False
    dimensions = tuple(item.dimension for item in request.dimension_evidence)
    assert dimensions == DEFAULT_REQUIRED_DIMENSIONS
    classes = {item.evidence_class for item in request.dimension_evidence}
    assert EvidenceClass.TEST.value in classes
    assert EvidenceClass.RECONSTRUCTED_PROOF.value in classes
    assert EvidenceClass.EXACT_STATIC_FACT.value in classes


def test_conjunction_of_separate_evidence_classes_validates() -> None:
    result = validate_translation(_compile())
    assert result.status == ValidationStatus.VALIDATED.value
    assert result.conjunction_passed is True
    assert result.typed_terminal is False
    assert result.can_authorize_completion is False
    assert result.can_authorize_transition is False
    assert result.mutated is False
    assert result.claims_general_python_equivalence is False
    by_dimension = {item.dimension: item for item in result.verdicts}
    assert by_dimension["tests"].evidence_class == EvidenceClass.TEST.value
    assert by_dimension["proofs"].evidence_class == EvidenceClass.RECONSTRUCTED_PROOF.value
    assert by_dimension["traces"].evidence_class == EvidenceClass.RUNTIME_OBSERVATION.value
    assert by_dimension["syntax"].evidence_class == EvidenceClass.EXACT_STATIC_FACT.value
    claim = compile_equivalence_claim(result)
    assert claim.equivalent_under_profile is True
    assert claim.general_python_equivalence is False
    assert claim.can_authorize_completion is False
    claimed = {item["dimension"]: item["evidence_class"] for item in claim.per_dimension_evidence}
    assert claimed["tests"] == EvidenceClass.TEST.value
    assert claimed["proofs"] == EvidenceClass.RECONSTRUCTED_PROOF.value
    assert claimed["tests"] != claimed["proofs"]


def test_test_evidence_cannot_admit_proofs() -> None:
    evidence = _all_passing_evidence()
    for item in evidence:
        if item["dimension"] == ValidationDimension.PROOFS.value:
            item["evidence_class"] = EvidenceClass.TEST.value
            item["evidence_cid"] = _cid("collapsed-test")
    with pytest.raises(TranslationValidationError, match="cannot admit proofs"):
        _compile(dimension_evidence=evidence)


def test_reconstructed_proof_cannot_admit_tests() -> None:
    evidence = _all_passing_evidence()
    for item in evidence:
        if item["dimension"] == ValidationDimension.TESTS.value:
            item["evidence_class"] = EvidenceClass.RECONSTRUCTED_PROOF.value
            item["full_suite_executed"] = False
    with pytest.raises(TranslationValidationError, match="must not collapse"):
        _compile(dimension_evidence=evidence)


def test_runtime_observation_cannot_admit_syntax() -> None:
    evidence = _all_passing_evidence()
    for item in evidence:
        if item["dimension"] == ValidationDimension.SYNTAX.value:
            item["evidence_class"] = EvidenceClass.RUNTIME_OBSERVATION.value
    with pytest.raises(TranslationValidationError, match="must not collapse"):
        _compile(dimension_evidence=evidence)


def test_shared_evidence_cid_across_disjoint_classes_is_collapse() -> None:
    shared = _cid("collapsed")
    evidence = _all_passing_evidence()
    for item in evidence:
        if item["dimension"] in {
            ValidationDimension.TESTS.value,
            ValidationDimension.PROOFS.value,
        }:
            item["evidence_cid"] = shared
    with pytest.raises(TranslationValidationError, match="must not collapse"):
        _compile(dimension_evidence=evidence)


def test_covers_all_dimensions_is_rejected() -> None:
    with pytest.raises(TranslationValidationError, match="must not collapse"):
        _compile(
            dimension_evidence=[
                {
                    "dimension": ValidationDimension.TESTS.value,
                    "evidence_class": EvidenceClass.TEST.value,
                    "evidence_cid": _cid("one"),
                    "status": DimensionStatus.PASS.value,
                    "covers_all_dimensions": True,
                }
            ],
            observation_profile=_profile(ValidationDimension.TESTS.value),
        )


def test_vector_evidence_cannot_admit_equivalence() -> None:
    with pytest.raises(TranslationValidationError, match="cannot admit"):
        _compile(
            vector_evidence={
                "evidence_class": EvidenceClass.VECTOR_CANDIDATE.value,
                "admit_equivalence": True,
            }
        )
    with pytest.raises(TranslationValidationError, match="cannot admit"):
        _compile(packet=_packet(evidence_class="model_hypothesis"))
    with pytest.raises(TranslationValidationError, match="raw-source"):
        _compile(
            vector_evidence={
                "evidence_class": EvidenceClass.VECTOR_CANDIDATE.value,
                "suppress_raw_source": True,
            }
        )


def test_missing_required_dimension_is_incomplete_terminal() -> None:
    evidence = [
        item
        for item in _all_passing_evidence()
        if item["dimension"] != ValidationDimension.PROOFS.value
    ]
    result = validate_translation(_compile(dimension_evidence=evidence))
    assert result.status == ValidationStatus.INCOMPLETE.value
    assert result.conjunction_passed is False
    assert result.typed_terminal is True
    assert "proofs" in result.terminal_reason
    claim = compile_equivalence_claim(result)
    assert claim.equivalent_under_profile is False
    assert claim.general_python_equivalence is False


def test_unsupported_required_dimension_is_typed_terminal() -> None:
    evidence = _all_passing_evidence()
    for item in evidence:
        if item["dimension"] == ValidationDimension.TRACES.value:
            item["status"] = DimensionStatus.UNSUPPORTED.value
            item["evidence_cid"] = ""
    result = validate_translation(_compile(dimension_evidence=evidence))
    assert result.status == ValidationStatus.UNSUPPORTED.value
    assert result.typed_terminal is True
    assert result.conjunction_passed is False
    assert "traces" in result.terminal_reason


def test_failed_required_dimension_rejects() -> None:
    evidence = _all_passing_evidence()
    for item in evidence:
        if item["dimension"] == ValidationDimension.IMPORTS.value:
            item["status"] = DimensionStatus.FAIL.value
    result = validate_translation(_compile(dimension_evidence=evidence))
    assert result.status == ValidationStatus.REJECTED.value
    assert result.typed_terminal is True
    assert "imports" in result.terminal_reason


def test_countermodel_fails_proofs_without_collapsing() -> None:
    evidence = _all_passing_evidence()
    for item in evidence:
        if item["dimension"] == ValidationDimension.PROOFS.value:
            item["evidence_class"] = EvidenceClass.COUNTERMODEL.value
            item["status"] = DimensionStatus.FAIL.value
            item["evidence_cid"] = _cid("countermodel")
    result = validate_translation(_compile(dimension_evidence=evidence))
    assert result.status == ValidationStatus.REJECTED.value
    proofs = next(item for item in result.verdicts if item.dimension == "proofs")
    assert proofs.evidence_class == EvidenceClass.COUNTERMODEL.value
    assert proofs.status == DimensionStatus.FAIL.value


def test_proof_candidate_cannot_admit_proofs_dimension() -> None:
    evidence = _all_passing_evidence()
    for item in evidence:
        if item["dimension"] == ValidationDimension.PROOFS.value:
            item["evidence_class"] = EvidenceClass.PROOF_CANDIDATE.value
    with pytest.raises(TranslationValidationError, match="proof_candidate"):
        _compile(dimension_evidence=evidence)


def test_full_suite_required_without_execution_is_incomplete() -> None:
    result = validate_translation(
        _compile(
            selection=_selection(effective_fallback="both", full_suite_required=True),
            dimension_evidence=[
                _evidence(
                    item["dimension"],
                    item["evidence_class"],
                    full_suite_executed=False,
                    label=item["dimension"],
                )
                for item in _all_passing_evidence()
            ],
        )
    )
    assert result.status == ValidationStatus.INCOMPLETE.value
    tests = next(item for item in result.verdicts if item.dimension == "tests")
    assert tests.status == DimensionStatus.MISSING.value
    assert tests.full_suite_executed is False


def test_optional_dimension_missing_does_not_block() -> None:
    request = _compile(
        observation_profile=_profile(
            ValidationDimension.SYNTAX.value,
            ValidationDimension.TESTS.value,
            optional=(ValidationDimension.TRACES.value,),
        ),
        dimension_evidence=[
            _evidence(ValidationDimension.SYNTAX.value, EvidenceClass.EXACT_STATIC_FACT.value),
            _evidence(
                ValidationDimension.TESTS.value,
                EvidenceClass.TEST.value,
                full_suite_executed=True,
            ),
        ],
    )
    result = validate_translation(request)
    assert result.status == ValidationStatus.VALIDATED.value
    traces = next(item for item in result.verdicts if item.dimension == "traces")
    assert traces.status == DimensionStatus.NOT_IN_PROFILE.value
    assert traces.required is False


def test_syntax_is_derived_from_raw_sources() -> None:
    request = _compile(
        observation_profile=_profile(ValidationDimension.SYNTAX.value),
        dimension_evidence=[],
        original_sources={"pkg/mod.py": ORIGINAL_SOURCE},
        candidate_sources={"pkg/extracted.py": CANDIDATE_SOURCE},
    )
    syntax = request.dimension_evidence[0]
    assert syntax.dimension == ValidationDimension.SYNTAX.value
    assert syntax.evidence_class == EvidenceClass.EXACT_STATIC_FACT.value
    assert syntax.status == DimensionStatus.PASS.value
    result = validate_translation(request)
    assert result.conjunction_passed is True


def test_unparseable_candidate_fails_syntax() -> None:
    request = _compile(
        observation_profile=_profile(ValidationDimension.SYNTAX.value),
        dimension_evidence=[],
        original_sources={"pkg/mod.py": ORIGINAL_SOURCE},
        candidate_sources={"pkg/extracted.py": INVALID_SOURCE},
    )
    result = validate_translation(request)
    assert result.status == ValidationStatus.REJECTED.value
    assert request.dimension_evidence[0].status == DimensionStatus.FAIL.value


def test_missing_raw_source_is_typed_terminal() -> None:
    with pytest.raises(TranslationValidationError, match="raw source"):
        _compile(packet=_packet(preimage={"source_cids": []}))
    with pytest.raises(TranslationValidationError, match="raw source"):
        _compile(packet=_packet(preimage={"environment_cid": _cid("env")}))
    with pytest.raises(TranslationValidationError, match="candidate source"):
        _compile(wave=_wave(after_source_cids=[]), candidate_source_cids=None)


def test_empty_write_paths_are_unrestricted_scope() -> None:
    with pytest.raises(TranslationValidationError, match="unrestricted scope"):
        _compile(packet=_packet(write_paths=[]))
    with pytest.raises(TranslationValidationError, match="unrestricted scope"):
        _compile(packet=_packet(write_paths=["pkg/*.py"]))
    with pytest.raises(TranslationValidationError, match="unrestricted scope"):
        _compile(packet=_packet(write_paths=["/tmp/pkg/mod.py"]))
    with pytest.raises(TranslationValidationError, match="unrestricted scope"):
        _compile(packet=_packet(write_paths=["pkg/../secret.py"]))


def test_tree_mismatch_fails_closed() -> None:
    with pytest.raises(TranslationValidationError, match="SPAR-025"):
        _compile(wave=_wave(tree_id=OTHER_TREE))
    with pytest.raises(TranslationValidationError, match="SPAR-026"):
        _compile(selection=_selection(tree_id=OTHER_TREE))


def test_missing_predecessors_fail_closed() -> None:
    with pytest.raises(TranslationValidationError, match="SPAR-019"):
        compile_translation_validation_request(
            packet=None,
            wave=_wave(),
            selection=_selection(),
            dimension_evidence=_all_passing_evidence(),
        )
    with pytest.raises(TranslationValidationError, match="SPAR-025"):
        compile_translation_validation_request(
            packet=_packet(),
            wave=None,
            selection=_selection(),
            dimension_evidence=_all_passing_evidence(),
        )
    with pytest.raises(TranslationValidationError, match="SPAR-026"):
        compile_translation_validation_request(
            packet=_packet(),
            wave=_wave(),
            selection=None,
            dimension_evidence=_all_passing_evidence(),
        )
    with pytest.raises(TranslationValidationError, match="packet_cids"):
        _compile(wave=_wave(packet_cids=[_cid("other-packet")]))


def test_general_python_equivalence_is_rejected() -> None:
    request = _compile()
    payload = request.to_dict()
    payload["claims_general_python_equivalence"] = True
    payload["request_cid"] = cid_for_dag_json(
        {key: value for key, value in payload.items() if key != "request_cid"}
    )
    with pytest.raises(TranslationValidationError, match="general Python equivalence"):
        TranslationValidationRequest.from_dict(payload)
    result = validate_translation(request)
    claim_payload = compile_equivalence_claim(result).to_dict()
    claim_payload["general_python_equivalence"] = True
    claim_payload["claim_cid"] = cid_for_dag_json(
        {key: value for key, value in claim_payload.items() if key != "claim_cid"}
    )
    with pytest.raises(TranslationValidationError, match="general Python equivalence"):
        RefactorEquivalenceClaim.from_dict(claim_payload)


def test_round_trip_and_receipts_are_deterministic() -> None:
    first = _compile()
    second = _compile()
    assert first.request_cid == second.request_cid
    restored = decode_canonical_request(encode_canonical_request(first))
    assert restored == first
    result = validate_translation(first)
    again = validate_translation(second)
    assert result.result_cid == again.result_cid
    assert decode_canonical_result(encode_canonical_result(result)) == result
    claim = compile_equivalence_claim(result)
    assert decode_canonical_claim(encode_canonical_claim(claim)) == claim
    orchestrator = TranslationValidationOrchestrator()
    compiled = orchestrator.compile_request(
        packet=_packet(),
        wave=_wave(),
        selection=_selection(),
        dimension_evidence=_all_passing_evidence(),
        observation_profile=_profile(),
        candidate_source_cids=[_cid("candidate")],
    )
    assert compiled.request_cid == first.request_cid
    validated = orchestrator.validate(compiled)
    assert validated.result_cid == result.result_cid
    assert orchestrator.claim(validated).claim_cid == claim.claim_cid


def test_dry_run_is_deterministic_and_does_not_mutate() -> None:
    request = _compile()
    first = dry_run_translation_validation(request)
    second = TranslationValidationOrchestrator().dry_run(request)
    assert first.result_cid == second.result_cid
    assert first.mutated is False
    assert first.deterministic is True
    with pytest.raises(TranslationValidationError, match="cannot mutate"):
        validate_translation(request, mutate=True)


def test_identity_excludes_observational_fields() -> None:
    request = _compile()
    payload = request.to_dict()
    assert not (IDENTITY_EXCLUDED_FIELDS & set(payload))
    dirty = dict(payload)
    dirty["timestamp"] = "now"
    with pytest.raises(TranslationValidationError, match="observational"):
        TranslationValidationRequest.from_dict(dirty)


def test_request_cannot_claim_authority_flags() -> None:
    request = _compile()
    payload = request.to_dict()
    payload["can_authorize_completion"] = True
    payload["request_cid"] = cid_for_dag_json(
        {key: value for key, value in payload.items() if key != "request_cid"}
    )
    with pytest.raises(TranslationValidationError, match="can_authorize_completion"):
        TranslationValidationRequest.from_dict(payload)
    payload = request.to_dict()
    payload["orchestrator_is_nomination_only"] = False
    payload["request_cid"] = cid_for_dag_json(
        {key: value for key, value in payload.items() if key != "request_cid"}
    )
    with pytest.raises(TranslationValidationError, match="nomination_only"):
        TranslationValidationRequest.from_dict(payload)
    payload = request.to_dict()
    payload["collapse_evidence_classes"] = True
    payload["request_cid"] = cid_for_dag_json(
        {key: value for key, value in payload.items() if key != "request_cid"}
    )
    with pytest.raises(TranslationValidationError, match="must not collapse"):
        TranslationValidationRequest.from_dict(payload)


def test_empty_validation_commands_fail_closed() -> None:
    with pytest.raises(TranslationValidationError, match="validation_commands"):
        _compile(packet=_packet(validation_commands=[]))


def test_may_fact_is_not_collapsed_to_static_fact() -> None:
    request = _compile(
        observation_profile=_profile(ValidationDimension.TYPES_EFFECTS.value),
        dimension_evidence=[
            _evidence(
                ValidationDimension.TYPES_EFFECTS.value,
                EvidenceClass.MAY_FACT.value,
            )
        ],
    )
    result = validate_translation(request)
    assert result.conjunction_passed is True
    verdict = result.verdicts[0]
    assert verdict.evidence_class == EvidenceClass.MAY_FACT.value
    claim = compile_equivalence_claim(result)
    assert claim.per_dimension_evidence[0]["evidence_class"] == EvidenceClass.MAY_FACT.value


def test_accepted_transition_cannot_admit_a_dimension() -> None:
    with pytest.raises(TranslationValidationError, match="accepted_transition"):
        _compile(
            observation_profile=_profile(ValidationDimension.SYNTAX.value),
            dimension_evidence=[
                _evidence(
                    ValidationDimension.SYNTAX.value,
                    EvidenceClass.ACCEPTED_TRANSITION.value,
                )
            ],
        )
