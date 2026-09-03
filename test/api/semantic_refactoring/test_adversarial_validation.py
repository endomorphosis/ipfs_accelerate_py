"""Independent contract tests for SPAR-029 mutation and adversarial validation."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.utils.cid_utils import cid_for_bytes, cid_for_dag_json
from ipfs_accelerate_py.agent_supervisor.semantic_refactoring.adversarial_validation import (
    ANALYZER_ID,
    AUTHORITY,
    AUTHORITY_OWNER,
    CRITICAL_SURVIVOR_BLOCKS_ACCEPTANCE,
    DECLARED_FAMILIES,
    DEFAULT_CRITICAL_MUTANT_KINDS,
    DEFAULT_REQUIRED_FAMILIES,
    DRY_RUN_IS_DETERMINISTIC,
    DRY_RUN_MUTATES,
    DUCKLAKE_IS_AUTHORITY,
    FAMILY_ADMITTED_EVIDENCE,
    FORBIDDEN_EQUIVALENCE_NAMES,
    GOAL_ID,
    IDENTITY_EXCLUDED_FIELDS,
    MARKDOWN_IS_NOT_COMPLETION,
    MODEL_OUTPUT_IS_PROPOSAL_ONLY,
    MUTANT_KILL_IS_NOT_COMPLETION,
    MUTATION_CAMPAIGN_RECEIPT_INTERFACE,
    MUTATION_CAMPAIGN_REQUEST_INTERFACE,
    PROGRAM,
    PROOF_CANDIDATE_CANNOT_ADMIT_PROOFS,
    PROJECTION_CLUSTERING_IS_AUTHORITY,
    RAW_SOURCE_REQUIRED,
    REFACTOR_MUTATION_AND_ADVERSARIAL_VALIDATOR_INTERFACE,
    RUNTIME_OBSERVATION_IS_NOT_STATIC_FACT,
    SOFT_SIGNALS_CANNOT_OVERRIDE_HARD_CONSTRAINTS,
    TASK_ID,
    TEST_PASS_IS_NOT_COMPLETION,
    TEST_PASS_IS_NOT_PROOF,
    UNKNOWN_REQUIRED_DYNAMICS_BLOCK_ACCEPTANCE,
    VALIDATION_CAN_AUTHORIZE_COMPLETION,
    VALIDATION_CAN_AUTHORIZE_TRANSITION,
    VALIDATION_CAN_CREATE_AUTHORITY,
    VALIDATION_CLAIMS_GENERAL_PYTHON_EQUIVALENCE,
    VALIDATION_COLLAPSES_EVIDENCE_CLASSES,
    VALIDATION_CONTRACT_VERSION,
    VALIDATOR_CAN_MUTATE_SOURCE,
    VALIDATOR_IS_NOMINATION_ONLY,
    VECTOR_SIMILARITY_IS_AUTHORITY,
    WORKER_SELF_APPROVAL,
    AdversarialKind,
    AdversarialValidationError,
    CampaignFamily,
    CampaignProfile,
    CampaignStatus,
    EvidenceClass,
    MetamorphicKind,
    MutantKind,
    MutantStatus,
    MutationCampaignReceipt,
    MutationCampaignRequest,
    PropertyKind,
    RefactorMutationAndAdversarialValidator,
    RelationStatus,
    admitted_evidence_classes,
    adversarial_validation_cid_profile,
    adversarial_validation_descriptor,
    assert_evidence_class_admitted,
    assert_not_competing_capsule_family,
    compile_mutation_campaign_request,
    decode_canonical_receipt,
    decode_canonical_request,
    dry_run_mutation_campaign,
    encode_canonical_receipt,
    encode_canonical_request,
    generate_bounded_adversarial_cases,
    generate_bounded_mutants,
    generate_bounded_relations,
    provider_free_exports,
    validate_mutation_campaign,
)


ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    ROOT
    / "ipfs_accelerate_py"
    / "agent_supervisor"
    / "semantic_refactoring"
    / "adversarial_validation.py"
)
TEST_PATH = Path(__file__).resolve()
WRITE_SCOPE = (
    "ipfs_accelerate_py/agent_supervisor/semantic_refactoring/adversarial_validation.py",
    "test/api/semantic_refactoring/test_adversarial_validation.py",
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
CANDIDATE_SOURCE = "from pkg.extracted import leaf\n"
EXTRACTED_SOURCE = "def leaf():\n    return 1\n"
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
        "adapter_kinds": ["facade", "reexport"],
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


def _graph(**overrides: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "tree_id": TREE_ID,
        "graph_view_cid": _cid("graph-view"),
        "unresolved_required_dynamics": False,
        "unresolved_frontier": {"items": [], "unresolved_count": 0},
    }
    fields.update(overrides)
    return fields


def _relation_verdict(
    item_id: str,
    kind: str,
    family: str,
    *,
    status: str = RelationStatus.PASS.value,
    evidence_class: str = EvidenceClass.TEST.value,
    full_suite_executed: bool = False,
) -> dict[str, Any]:
    return {
        "family": family,
        "item_id": item_id,
        "kind": kind,
        "evidence_class": evidence_class,
        "status": status,
        "evidence_cid": _cid(f"{family}:{item_id}:{status}"),
        "required": True,
        "critical": True,
        "full_suite_executed": full_suite_executed,
    }


def _mutant_verdict(
    mutant_id: str,
    kind: str,
    *,
    status: str = MutantStatus.KILLED.value,
    evidence_class: str = EvidenceClass.TEST.value,
    critical: bool = True,
    full_suite_executed: bool = False,
) -> dict[str, Any]:
    return {
        "family": CampaignFamily.MUTATION.value,
        "mutant_id": mutant_id,
        "kind": kind,
        "evidence_class": evidence_class,
        "status": status,
        "evidence_cid": _cid(f"mutant:{mutant_id}:{status}"),
        "required": True,
        "critical": critical,
        "full_suite_executed": full_suite_executed,
    }


def _adversarial_verdict(
    case_id: str,
    kind: str,
    *,
    status: str = RelationStatus.PASS.value,
    evidence_class: str = EvidenceClass.EXACT_STATIC_FACT.value,
) -> dict[str, Any]:
    return {
        "family": CampaignFamily.ADVERSARIAL.value,
        "case_id": case_id,
        "kind": kind,
        "evidence_class": evidence_class,
        "status": status,
        "evidence_cid": _cid(f"adversarial:{case_id}:{status}"),
        "required": True,
        "critical": True,
    }


def _passing_verdicts() -> dict[str, Any]:
    properties, metamorphics = generate_bounded_relations(write_paths=WRITE_PATHS)
    mutants = generate_bounded_mutants(write_paths=WRITE_PATHS)
    cases = generate_bounded_adversarial_cases(write_paths=WRITE_PATHS)
    return {
        "relation_verdicts": [
            _relation_verdict(
                item.relation_id,
                item.kind,
                CampaignFamily.PROPERTY.value,
                evidence_class=EvidenceClass.EXACT_STATIC_FACT.value,
            )
            for item in properties
        ]
        + [
            _relation_verdict(
                item.relation_id,
                item.kind,
                CampaignFamily.METAMORPHIC.value,
            )
            for item in metamorphics
        ],
        "mutant_verdicts": [
            _mutant_verdict(item.mutant_id, item.kind) for item in mutants
        ],
        "adversarial_verdicts": [
            _adversarial_verdict(item.case_id, item.kind) for item in cases
        ],
    }


def _compile(**overrides: Any) -> MutationCampaignRequest:
    fields: dict[str, Any] = {
        "packet": _packet(),
        "wave": _wave(),
        "selection": _selection(),
        "graph": _graph(),
        "campaign_profile": CampaignProfile(),
        "generate": True,
    }
    fields.update(_passing_verdicts())
    fields.update(overrides)
    return compile_mutation_campaign_request(**fields)


def test_owned_paths_and_task_identity_are_exact() -> None:
    assert TASK_ID == "SPAR-029"
    assert GOAL_ID == "SPAR-G052"
    assert PROGRAM == "semantic-preserving-autonomous-remodularization-v1"
    assert MUTATION_CAMPAIGN_REQUEST_INTERFACE == "MutationCampaignRequest@1"
    assert MUTATION_CAMPAIGN_RECEIPT_INTERFACE == "MutationCampaignReceipt@1"
    assert (
        REFACTOR_MUTATION_AND_ADVERSARIAL_VALIDATOR_INTERFACE
        == "RefactorMutationAndAdversarialValidator@1"
    )
    assert VALIDATION_CONTRACT_VERSION == "1"
    assert ANALYZER_ID.endswith("adversarial_validation@1")
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
    assert CRITICAL_SURVIVOR_BLOCKS_ACCEPTANCE is True
    assert UNKNOWN_REQUIRED_DYNAMICS_BLOCK_ACCEPTANCE is True
    assert VECTOR_SIMILARITY_IS_AUTHORITY is False
    assert PROJECTION_CLUSTERING_IS_AUTHORITY is False
    assert MODEL_OUTPUT_IS_PROPOSAL_ONLY is True
    assert TEST_PASS_IS_NOT_COMPLETION is True
    assert TEST_PASS_IS_NOT_PROOF is True
    assert MUTANT_KILL_IS_NOT_COMPLETION is True
    assert RUNTIME_OBSERVATION_IS_NOT_STATIC_FACT is True
    assert PROOF_CANDIDATE_CANNOT_ADMIT_PROOFS is True
    assert MARKDOWN_IS_NOT_COMPLETION is True
    assert WORKER_SELF_APPROVAL is False
    assert DUCKLAKE_IS_AUTHORITY is False
    assert SOFT_SIGNALS_CANNOT_OVERRIDE_HARD_CONSTRAINTS is True
    assert VALIDATOR_IS_NOMINATION_ONLY is True
    assert VALIDATOR_CAN_MUTATE_SOURCE is False
    assert RAW_SOURCE_REQUIRED is True
    assert DRY_RUN_IS_DETERMINISTIC is True
    assert DRY_RUN_MUTATES is False
    profile = adversarial_validation_cid_profile()
    assert profile["codec"] == "dag-json"
    assert "not universal meaning" in profile["rule"]


def test_module_defines_predicted_symbols_not_capsule_family() -> None:
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    names = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
    assert "RefactorMutationAndAdversarialValidator" in names
    assert "MutationCampaignRequest" in names
    assert "MutationCampaignReceipt" in names
    for capsule in CAPSULE_TYPES:
        assert capsule not in names
    assert_not_competing_capsule_family()
    exports = provider_free_exports()
    assert "RefactorMutationAndAdversarialValidator" in exports
    assert "validate_mutation_campaign" in exports
    assert "generate_bounded_mutants" in exports


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
    descriptor = adversarial_validation_descriptor()
    assert descriptor["interface"] == REFACTOR_MUTATION_AND_ADVERSARIAL_VALIDATOR_INTERFACE
    assert descriptor["claims_general_python_equivalence"] is False
    assert descriptor["collapses_evidence_classes"] is False
    assert descriptor["critical_survivor_blocks_acceptance"] is True
    assert descriptor["unknown_required_dynamics_block_acceptance"] is True
    assert descriptor["can_mutate_source"] is False
    forbids = set(descriptor["forbids"])
    assert "prove_python_equivalence" in forbids
    assert "collapse_evidence" in forbids
    assert "admit_from_vectors" in forbids
    assert "kill_all_mutants_without_evidence" in forbids


def test_does_not_reuse_procedure_compiler_or_spar027_types() -> None:
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
    assert "translation_validation" not in imported
    names = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
    assert "TranslationValidation" not in names
    assert "GeneratedToolCertificate" not in names


def test_all_declared_families_have_admitted_evidence_classes() -> None:
    assert set(DEFAULT_REQUIRED_FAMILIES) == DECLARED_FAMILIES
    assert set(FAMILY_ADMITTED_EVIDENCE) == DECLARED_FAMILIES
    assert EvidenceClass.RECONSTRUCTED_PROOF.value not in admitted_evidence_classes(
        CampaignFamily.MUTATION.value
    )
    assert EvidenceClass.TEST.value in admitted_evidence_classes(
        CampaignFamily.MUTATION.value
    )
    with pytest.raises(AdversarialValidationError, match="cannot admit"):
        assert_evidence_class_admitted(
            CampaignFamily.MUTATION.value, EvidenceClass.EXACT_STATIC_FACT.value
        )
    with pytest.raises(AdversarialValidationError, match="proof_candidate"):
        assert_evidence_class_admitted(
            CampaignFamily.PROPERTY.value, EvidenceClass.PROOF_CANDIDATE.value
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
    assert request.validator_is_nomination_only is True
    assert request.claims_general_python_equivalence is False
    assert request.collapse_evidence_classes is False
    assert request.can_authorize_completion is False
    families = {item.family for item in request.relation_verdicts}
    assert CampaignFamily.PROPERTY.value in families
    assert CampaignFamily.METAMORPHIC.value in families
    assert {item.kind for item in request.mutants} >= set(DEFAULT_CRITICAL_MUTANT_KINDS)


def test_passing_campaign_validates_without_authorizing_completion() -> None:
    result = validate_mutation_campaign(_compile())
    assert result.status == CampaignStatus.VALIDATED.value
    assert result.conjunction_passed is True
    assert result.typed_terminal is False
    assert result.can_authorize_completion is False
    assert result.can_authorize_transition is False
    assert result.mutated is False
    assert result.critical_survivors == ()
    assert result.unknown_required_dynamics == ()
    assert result.claims_general_python_equivalence is False
    killed = {item.kind for item in result.mutant_verdicts}
    assert set(DEFAULT_CRITICAL_MUTANT_KINDS) <= killed


def test_critical_survivor_rejects() -> None:
    verdicts = _passing_verdicts()
    for item in verdicts["mutant_verdicts"]:
        if item["kind"] == MutantKind.DROP_REEXPORT.value:
            item["status"] = MutantStatus.SURVIVED.value
    result = validate_mutation_campaign(_compile(**verdicts))
    assert result.status == CampaignStatus.REJECTED.value
    assert result.typed_terminal is True
    assert result.conjunction_passed is False
    assert result.critical_survivors
    assert "critical survivor" in result.terminal_reason


def test_unknown_required_dynamics_are_typed_terminal() -> None:
    result = validate_mutation_campaign(
        _compile(graph=_graph(unresolved_required_dynamics=True))
    )
    assert result.status == CampaignStatus.UNSUPPORTED.value
    assert result.typed_terminal is True
    assert result.conjunction_passed is False
    assert result.unknown_required_dynamics
    assert "unknown required dynamics" in result.terminal_reason


def test_unknown_critical_mutant_is_typed_terminal() -> None:
    verdicts = _passing_verdicts()
    for item in verdicts["mutant_verdicts"]:
        if item["kind"] == MutantKind.INVERT_BOUNDARY_GUARD.value:
            item["status"] = MutantStatus.UNKNOWN.value
            item["evidence_cid"] = ""
    result = validate_mutation_campaign(_compile(**verdicts))
    assert result.status == CampaignStatus.UNSUPPORTED.value
    assert result.typed_terminal is True
    assert "unknown required dynamics" in result.terminal_reason


def test_missing_required_mutant_evidence_is_incomplete() -> None:
    verdicts = _passing_verdicts()
    verdicts["mutant_verdicts"] = [
        item
        for item in verdicts["mutant_verdicts"]
        if item["kind"] != MutantKind.RENAME_FACADE_SYMBOL.value
    ]
    result = validate_mutation_campaign(_compile(**verdicts))
    assert result.status == CampaignStatus.INCOMPLETE.value
    assert result.typed_terminal is True
    assert "missing required" in result.terminal_reason


def test_failed_required_property_rejects() -> None:
    verdicts = _passing_verdicts()
    for item in verdicts["relation_verdicts"]:
        if item["kind"] == PropertyKind.FACADE_REEXPORT_IDENTITY.value:
            item["status"] = RelationStatus.FAIL.value
    result = validate_mutation_campaign(_compile(**verdicts))
    assert result.status == CampaignStatus.REJECTED.value
    assert "failed required" in result.terminal_reason


def test_vector_evidence_cannot_admit_campaign() -> None:
    with pytest.raises(AdversarialValidationError, match="cannot admit"):
        _compile(
            vector_evidence={
                "evidence_class": EvidenceClass.VECTOR_CANDIDATE.value,
                "admit_equivalence": True,
            }
        )
    with pytest.raises(AdversarialValidationError, match="cannot admit"):
        _compile(packet=_packet(evidence_class="model_hypothesis"))
    with pytest.raises(AdversarialValidationError, match="raw-source"):
        _compile(
            vector_evidence={
                "evidence_class": EvidenceClass.VECTOR_CANDIDATE.value,
                "suppress_raw_source": True,
            }
        )


def test_shared_evidence_cid_across_disjoint_families_is_collapse() -> None:
    shared = _cid("collapsed")
    verdicts = _passing_verdicts()
    for item in verdicts["relation_verdicts"]:
        if item["family"] == CampaignFamily.PROPERTY.value:
            item["evidence_cid"] = shared
            item["evidence_class"] = EvidenceClass.EXACT_STATIC_FACT.value
    for item in verdicts["mutant_verdicts"]:
        item["evidence_cid"] = shared
    with pytest.raises(AdversarialValidationError, match="must not collapse"):
        _compile(**verdicts)


def test_covers_all_families_is_rejected() -> None:
    with pytest.raises(AdversarialValidationError, match="must not collapse"):
        _compile(
            generate=False,
            properties=[
                {
                    "relation_id": "property:facade_reexport_identity",
                    "kind": PropertyKind.FACADE_REEXPORT_IDENTITY.value,
                    "subject_id": WRITE_PATHS[0],
                }
            ],
            relation_verdicts=[
                {
                    "family": CampaignFamily.PROPERTY.value,
                    "item_id": "property:facade_reexport_identity",
                    "kind": PropertyKind.FACADE_REEXPORT_IDENTITY.value,
                    "evidence_class": EvidenceClass.EXACT_STATIC_FACT.value,
                    "status": RelationStatus.PASS.value,
                    "evidence_cid": _cid("one"),
                    "covers_all_families": True,
                }
            ],
            campaign_profile=CampaignProfile(
                required_families=(CampaignFamily.PROPERTY.value,),
                optional_families=(),
            ),
        )


def test_facade_reexport_is_derived_from_raw_sources() -> None:
    request = _compile(
        generate=False,
        campaign_profile=CampaignProfile(
            required_families=(CampaignFamily.PROPERTY.value,),
            optional_families=(),
        ),
        properties=[
            {
                "relation_id": "property:facade_reexport_identity",
                "kind": PropertyKind.FACADE_REEXPORT_IDENTITY.value,
                "subject_id": WRITE_PATHS[0],
            }
        ],
        relation_verdicts=[],
        mutant_verdicts=[],
        adversarial_verdicts=[],
        original_sources={"pkg/mod.py": ORIGINAL_SOURCE},
        candidate_sources={"pkg/mod.py": CANDIDATE_SOURCE, "pkg/extracted.py": EXTRACTED_SOURCE},
        graph=None,
    )
    derived = [item for item in request.relation_verdicts if item.kind == "facade_reexport_identity"]
    assert derived
    assert derived[0].evidence_class == EvidenceClass.EXACT_STATIC_FACT.value
    assert derived[0].status == RelationStatus.PASS.value
    result = validate_mutation_campaign(request)
    assert result.conjunction_passed is True


def test_unparseable_candidate_fails_derived_property() -> None:
    request = _compile(
        generate=False,
        campaign_profile=CampaignProfile(
            required_families=(CampaignFamily.PROPERTY.value,),
            optional_families=(),
        ),
        properties=[
            {
                "relation_id": "property:facade_reexport_identity",
                "kind": PropertyKind.FACADE_REEXPORT_IDENTITY.value,
                "subject_id": WRITE_PATHS[0],
            }
        ],
        relation_verdicts=[],
        mutant_verdicts=[],
        adversarial_verdicts=[],
        original_sources={"pkg/mod.py": ORIGINAL_SOURCE},
        candidate_sources={"pkg/extracted.py": INVALID_SOURCE},
        graph=None,
    )
    result = validate_mutation_campaign(request)
    assert result.status == CampaignStatus.REJECTED.value
    assert request.relation_verdicts[0].status == RelationStatus.FAIL.value


def test_missing_raw_source_is_typed_terminal() -> None:
    with pytest.raises(AdversarialValidationError, match="raw source"):
        _compile(packet=_packet(preimage={"source_cids": []}))
    with pytest.raises(AdversarialValidationError, match="raw source"):
        _compile(packet=_packet(preimage={"environment_cid": _cid("env")}))
    with pytest.raises(AdversarialValidationError, match="candidate source"):
        _compile(wave=_wave(after_source_cids=[]), candidate_source_cids=None)


def test_empty_write_paths_are_unrestricted_scope() -> None:
    with pytest.raises(AdversarialValidationError, match="unrestricted scope"):
        _compile(packet=_packet(write_paths=[]))
    with pytest.raises(AdversarialValidationError, match="unrestricted scope"):
        _compile(packet=_packet(write_paths=["pkg/*.py"]))
    with pytest.raises(AdversarialValidationError, match="unrestricted scope"):
        _compile(packet=_packet(write_paths=["/tmp/pkg/mod.py"]))
    with pytest.raises(AdversarialValidationError, match="unrestricted scope"):
        _compile(packet=_packet(write_paths=["pkg/../secret.py"]))


def test_tree_mismatch_fails_closed() -> None:
    with pytest.raises(AdversarialValidationError, match="SPAR-025"):
        _compile(wave=_wave(tree_id=OTHER_TREE))
    with pytest.raises(AdversarialValidationError, match="SPAR-026"):
        _compile(selection=_selection(tree_id=OTHER_TREE))


def test_missing_predecessors_fail_closed() -> None:
    with pytest.raises(AdversarialValidationError, match="SPAR-019"):
        compile_mutation_campaign_request(
            packet=None,
            wave=_wave(),
            selection=_selection(),
        )
    with pytest.raises(AdversarialValidationError, match="SPAR-025"):
        compile_mutation_campaign_request(
            packet=_packet(),
            wave=None,
            selection=_selection(),
        )
    with pytest.raises(AdversarialValidationError, match="SPAR-026"):
        compile_mutation_campaign_request(
            packet=_packet(),
            wave=_wave(),
            selection=None,
        )
    with pytest.raises(AdversarialValidationError, match="packet_cids"):
        _compile(wave=_wave(packet_cids=[_cid("other-packet")]))


def test_generated_campaign_is_bounded_and_deterministic() -> None:
    first = generate_bounded_mutants(write_paths=WRITE_PATHS)
    second = generate_bounded_mutants(write_paths=WRITE_PATHS)
    assert first == second
    assert len(first) <= 64
    assert {item.kind for item in first} == set(DEFAULT_CRITICAL_MUTANT_KINDS)
    properties, metamorphics = generate_bounded_relations(write_paths=WRITE_PATHS)
    assert any(item.kind == PropertyKind.FACADE_REEXPORT_IDENTITY.value for item in properties)
    assert any(
        item.kind == MetamorphicKind.FACADE_VS_EXTRACTED_CALL.value for item in metamorphics
    )
    cases = generate_bounded_adversarial_cases(
        write_paths=WRITE_PATHS, facade_kinds=("plugin", "registry")
    )
    kinds = {item.kind for item in cases}
    assert AdversarialKind.REQUIRED_UNRESOLVED_DYNAMIC.value in kinds
    assert AdversarialKind.PLUGIN_ENTRY_POINT_DRIFT.value in kinds
    assert AdversarialKind.REGISTRY_ALIAS_COLLISION.value in kinds


def test_declared_relations_are_reused_instead_of_duplicated() -> None:
    request = _compile(
        properties=[
            {
                "relation_id": "property:facade_reexport_identity",
                "kind": PropertyKind.FACADE_REEXPORT_IDENTITY.value,
                "subject_id": WRITE_PATHS[0],
            }
        ]
    )
    matching = [
        item
        for item in request.properties
        if item.kind == PropertyKind.FACADE_REEXPORT_IDENTITY.value
    ]
    assert len(matching) == 1
    assert matching[0].reused is True


def test_non_critical_survivor_does_not_block() -> None:
    request = _compile(
        generate=False,
        campaign_profile=CampaignProfile(
            required_families=(CampaignFamily.MUTATION.value,),
            optional_families=(),
        ),
        mutants=[
            {
                "mutant_id": "mutant:suppress_deprecation",
                "kind": MutantKind.SUPPRESS_DEPRECATION.value,
                "subject_id": WRITE_PATHS[0],
                "required": True,
                "critical": False,
            }
        ],
        relation_verdicts=[],
        adversarial_verdicts=[],
        mutant_verdicts=[
            _mutant_verdict(
                "mutant:suppress_deprecation",
                MutantKind.SUPPRESS_DEPRECATION.value,
                status=MutantStatus.SURVIVED.value,
                critical=False,
            )
        ],
        graph=None,
    )
    result = validate_mutation_campaign(request)
    assert result.status == CampaignStatus.VALIDATED.value
    assert result.critical_survivors == ()


def test_full_suite_required_without_execution_is_incomplete() -> None:
    verdicts = _passing_verdicts()
    for item in verdicts["mutant_verdicts"]:
        item["full_suite_executed"] = False
    result = validate_mutation_campaign(
        _compile(
            selection=_selection(effective_fallback="both", full_suite_required=True),
            **verdicts,
        )
    )
    assert result.status == CampaignStatus.INCOMPLETE.value
    assert all(item.status == MutantStatus.MISSING.value for item in result.mutant_verdicts)


def test_optional_family_missing_does_not_block() -> None:
    request = _compile(
        generate=False,
        campaign_profile=CampaignProfile(
            required_families=(CampaignFamily.PROPERTY.value,),
            optional_families=(CampaignFamily.ADVERSARIAL.value,),
        ),
        properties=[
            {
                "relation_id": "property:facade_reexport_identity",
                "kind": PropertyKind.FACADE_REEXPORT_IDENTITY.value,
                "subject_id": WRITE_PATHS[0],
            }
        ],
        relation_verdicts=[
            _relation_verdict(
                "property:facade_reexport_identity",
                PropertyKind.FACADE_REEXPORT_IDENTITY.value,
                CampaignFamily.PROPERTY.value,
                evidence_class=EvidenceClass.EXACT_STATIC_FACT.value,
            )
        ],
        mutant_verdicts=[],
        adversarial_verdicts=[],
        graph=None,
    )
    result = validate_mutation_campaign(request)
    assert result.status == CampaignStatus.VALIDATED.value


def test_general_python_equivalence_is_rejected() -> None:
    request = _compile()
    payload = request.to_dict()
    payload["claims_general_python_equivalence"] = True
    payload["request_cid"] = cid_for_dag_json(
        {key: value for key, value in payload.items() if key != "request_cid"}
    )
    with pytest.raises(AdversarialValidationError, match="general Python equivalence"):
        MutationCampaignRequest.from_dict(payload)


def test_round_trip_and_receipts_are_deterministic() -> None:
    first = _compile()
    second = _compile()
    assert first.request_cid == second.request_cid
    restored = decode_canonical_request(encode_canonical_request(first))
    assert restored == first
    result = validate_mutation_campaign(first)
    again = validate_mutation_campaign(second)
    assert result.receipt_cid == again.receipt_cid
    assert decode_canonical_receipt(encode_canonical_receipt(result)) == result
    validator = RefactorMutationAndAdversarialValidator()
    compiled = validator.compile_request(
        packet=_packet(),
        wave=_wave(),
        selection=_selection(),
        graph=_graph(),
        **_passing_verdicts(),
    )
    assert compiled.request_cid == first.request_cid
    validated = validator.validate(compiled)
    assert validated.receipt_cid == result.receipt_cid


def test_dry_run_is_deterministic_and_does_not_mutate() -> None:
    request = _compile()
    first = dry_run_mutation_campaign(request)
    second = RefactorMutationAndAdversarialValidator().dry_run(request)
    assert first.receipt_cid == second.receipt_cid
    assert first.mutated is False
    assert first.deterministic is True
    with pytest.raises(AdversarialValidationError, match="cannot mutate"):
        validate_mutation_campaign(request, mutate=True)


def test_identity_excludes_observational_fields() -> None:
    request = _compile()
    payload = request.to_dict()
    assert not (IDENTITY_EXCLUDED_FIELDS & set(payload))
    dirty = dict(payload)
    dirty["timestamp"] = "now"
    with pytest.raises(AdversarialValidationError, match="observational"):
        MutationCampaignRequest.from_dict(dirty)


def test_request_cannot_claim_authority_flags() -> None:
    request = _compile()
    payload = request.to_dict()
    payload["can_authorize_completion"] = True
    payload["request_cid"] = cid_for_dag_json(
        {key: value for key, value in payload.items() if key != "request_cid"}
    )
    with pytest.raises(AdversarialValidationError, match="can_authorize_completion"):
        MutationCampaignRequest.from_dict(payload)
    payload = request.to_dict()
    payload["validator_is_nomination_only"] = False
    payload["request_cid"] = cid_for_dag_json(
        {key: value for key, value in payload.items() if key != "request_cid"}
    )
    with pytest.raises(AdversarialValidationError, match="nomination_only"):
        MutationCampaignRequest.from_dict(payload)
    payload = request.to_dict()
    payload["collapse_evidence_classes"] = True
    payload["request_cid"] = cid_for_dag_json(
        {key: value for key, value in payload.items() if key != "request_cid"}
    )
    with pytest.raises(AdversarialValidationError, match="must not collapse"):
        MutationCampaignRequest.from_dict(payload)


def test_empty_validation_commands_fail_closed() -> None:
    with pytest.raises(AdversarialValidationError, match="validation_commands"):
        _compile(packet=_packet(validation_commands=[]))


def test_accepted_transition_cannot_admit_a_family() -> None:
    verdicts = _passing_verdicts()
    for item in verdicts["relation_verdicts"]:
        if item["family"] == CampaignFamily.PROPERTY.value:
            item["evidence_class"] = EvidenceClass.ACCEPTED_TRANSITION.value
    with pytest.raises(AdversarialValidationError, match="accepted_transition"):
        _compile(**verdicts)


def test_unknown_dynamic_import_frontier_is_unsupported() -> None:
    result = validate_mutation_campaign(
        _compile(
            graph=_graph(),
            frontier={
                "tree_id": TREE_ID,
                "findings": [
                    {
                        "kind": "dynamic_import",
                        "unresolved": True,
                        "required": True,
                    }
                ],
            },
        )
    )
    assert result.status == CampaignStatus.UNSUPPORTED.value
    assert any(
        item.kind == AdversarialKind.UNKNOWN_DYNAMIC_IMPORT.value
        and item.status == RelationStatus.UNSUPPORTED.value
        for item in result.adversarial_verdicts
    )
