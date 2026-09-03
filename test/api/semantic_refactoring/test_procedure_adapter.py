"""Independent contract tests for SPAR-034 ProofCarryingProcedure refactor adapter."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.utils.cid_utils import cid_for_bytes, cid_for_dag_json
from ipfs_accelerate_py.agent_supervisor.semantic_refactoring.procedure_adapter import (
    ADAPTER_IS_NOMINATION_ONLY,
    ADVERSARIAL_QUALIFICATION_INTERFACE,
    ADVERSARIAL_QUALIFICATION_ROUTE_EXISTS,
    ALLOWED_HOLE_TYPES,
    ANALYZER_ID,
    ANTI_UNIFIED_REFACTOR_PLAN_INTERFACE,
    AUTHORITY,
    AUTHORITY_OWNER,
    CANDIDATES_REMAIN_UNPROMOTED,
    DECLARED_COMPILATION_STATUSES,
    DECLARED_OUTCOMES,
    DECLARED_QUALIFICATION_ROUTES,
    DECLARED_QUALIFICATION_STATUSES,
    DRY_RUN_IS_DETERMINISTIC,
    DRY_RUN_MUTATES,
    DUCKLAKE_IS_AUTHORITY,
    FORBIDDEN_HOLE_TYPES,
    FORBIDDEN_PROCEDURE_NAMES,
    GENERAL_PYTHON_EQUIVALENCE_CLAIMED,
    GOAL_ID,
    HELD_OUT_QUALIFICATION_INTERFACE,
    HELD_OUT_QUALIFICATION_REQUIRED_FOR_PROMOTION,
    HOLES_PRESERVED,
    IDENTITY_EXCLUDED_FIELDS,
    IMPLICIT_INSTALL_FORBIDDEN,
    IMPLICIT_NETWORK_FORBIDDEN,
    INCOMPLETE_CONTRACTS_ARE_TYPED_TERMINALS,
    INFERRED_PROCEDURE_CONTRACT_INTERFACE,
    MARKDOWN_IS_NOT_COMPLETION,
    MODEL_OUTPUT_IS_PROPOSAL_ONLY,
    NETWORK_DENIED,
    NETWORK_DENY,
    NORMALIZED_REFACTOR_TRAJECTORY_INTERFACE,
    PATHS_NEVER_BECOME_PARAMETERS,
    PRESERVED_HOLE_INTERFACE,
    PROCEDURE_ADAPTER_CONTRACT_VERSION,
    PROCEDURE_AUTHORITY,
    PROCEDURE_AUTHORITY_INTERFACE,
    PROCEDURE_AUTHORITY_OWNER,
    PROCEDURE_CANNOT_SELF_CERTIFY,
    PROCEDURE_CANNOT_SELF_PROMOTE,
    PROCEDURE_CAN_AUTHORIZE_COMPLETION,
    PROCEDURE_CAN_AUTHORIZE_TRANSITION,
    PROCEDURE_CAN_CREATE_AUTHORITY,
    PROCEDURE_CAN_CREATE_PROCEDURE_AUTHORITY,
    PROCEDURE_PROMOTION_NOMINATION_INTERFACE,
    PROGRAM,
    PROJECTION_CLUSTERING_IS_AUTHORITY,
    PROOF_CARRYING_PROCEDURE_REFACTOR_ADAPTER_INTERFACE,
    RAW_SOURCE_REQUIRED,
    REFACTOR_PROCEDURE_COMPILATION_RECEIPT_INTERFACE,
    SOFT_SIGNALS_CANNOT_OVERRIDE_HARD_CONSTRAINTS,
    TASK_ID,
    TEST_PASS_IS_NOT_COMPLETION,
    TEST_PASS_IS_NOT_PROOF,
    UNKNOWN_REMAINS_UNKNOWN,
    VALIDATION_NEVER_DROPPED,
    VECTOR_SIMILARITY_IS_AUTHORITY,
    WORKER_SELF_APPROVAL,
    AdversarialPolarity,
    AdversarialQualification,
    CompilationStatus,
    HeldOutQualification,
    ProcedureAdapterError,
    ProcedurePromotionNomination,
    ProofCarryingProcedureRefactorAdapter,
    QualificationRoute,
    QualificationStatus,
    RefactorProcedureCompilationReceipt,
    TrajectoryOutcome,
    anti_unify_refactor_plans,
    assert_not_competing_capsule_family,
    compile_normalized_trajectory,
    compile_preserved_hole,
    compile_refactor_procedure,
    decode_canonical_receipt,
    dry_run_refactor_procedure,
    encode_canonical_receipt,
    infer_procedure_contract,
    nominate_procedure_promotion,
    preserve_holes,
    procedure_adapter_cid_profile,
    procedure_adapter_descriptor,
    provider_free_exports,
)


ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    ROOT
    / "ipfs_accelerate_py"
    / "agent_supervisor"
    / "semantic_refactoring"
    / "procedure_adapter.py"
)
TEST_PATH = Path(__file__).resolve()
WRITE_SCOPE = (
    "ipfs_accelerate_py/agent_supervisor/semantic_refactoring/procedure_adapter.py",
    "test/api/semantic_refactoring/test_procedure_adapter.py",
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


def _cid(label: str) -> str:
    return cid_for_bytes(label.encode("utf-8"))


def _wave(label: str = "wave", **overrides: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "tree_id": TREE_ID,
        "receipt_cid": _cid(label),
        "packet_cids": [_cid(f"packet-{label}")],
        "write_paths": list(WRITE_PATHS),
        "status": "applied",
        "writes_repository": False,
        "executor_is_nomination_only": True,
    }
    fields.update(overrides)
    return fields


def _transition(label: str = "t1", **overrides: Any) -> dict[str, Any]:
    wave_label = overrides.pop("wave_label", "wave-a") if "wave_label" in overrides else None
    fields: dict[str, Any] = {
        "tree_id": TREE_ID,
        "outcome": TrajectoryOutcome.ACCEPTED.value,
        "wave_receipt_cid": _cid("wave-a"),
        "packet_cid": _cid("packet-wave-a"),
        "raw_source_cids": [_cid("source")],
        "write_paths": list(WRITE_PATHS),
        "validation_commands": list(VALIDATION),
        "evidence_class": "transition",
        "transition_cid": _cid(label),
    }
    if wave_label is not None:
        fields["wave_receipt_cid"] = _cid(wave_label)
        fields["packet_cid"] = _cid(f"packet-{wave_label}")
    fields.update(overrides)
    return fields


def _held_out(**overrides: Any) -> dict[str, Any]:
    fields = _transition(
        "held-out",
        wave_label="wave-held",
        packet_cid=_cid("packet-wave-held"),
        wave_receipt_cid=_cid("wave-held"),
        transition_cid=_cid("held-out"),
    )
    fields.update(overrides)
    return fields


def _adversarial(**overrides: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "case_id": "adv:boundary",
        "kind": "boundary",
        "polarity": AdversarialPolarity.MUST_SURVIVE.value,
        "tree_id": TREE_ID,
        "source_authority": "specification",
    }
    fields.update(overrides)
    return fields


def _compile(**overrides: Any) -> RefactorProcedureCompilationReceipt:
    fields: dict[str, Any] = {
        "waves": [_wave("wave-a"), _wave("wave-b"), _wave("wave-held")],
        "transitions": [
            _transition("t1", wave_label="wave-a"),
            _transition(
                "t2",
                wave_label="wave-b",
                packet_cid=_cid("packet-wave-b"),
                wave_receipt_cid=_cid("wave-b"),
            ),
        ],
        "held_out": [_held_out()],
        "adversarial": [_adversarial()],
    }
    fields.update(overrides)
    return compile_refactor_procedure(**fields)


def test_owned_paths_and_task_identity_are_exact() -> None:
    assert TASK_ID == "SPAR-034"
    assert GOAL_ID == "SPAR-G062"
    assert PROGRAM == "semantic-preserving-autonomous-remodularization-v1"
    assert (
        PROOF_CARRYING_PROCEDURE_REFACTOR_ADAPTER_INTERFACE
        == "ProofCarryingProcedureRefactorAdapter@1"
    )
    assert NORMALIZED_REFACTOR_TRAJECTORY_INTERFACE == "NormalizedRefactorTrajectory@1"
    assert ANTI_UNIFIED_REFACTOR_PLAN_INTERFACE == "AntiUnifiedRefactorPlan@1"
    assert INFERRED_PROCEDURE_CONTRACT_INTERFACE == "InferredProcedureContract@1"
    assert PRESERVED_HOLE_INTERFACE == "PreservedHole@1"
    assert HELD_OUT_QUALIFICATION_INTERFACE == "HeldOutQualification@1"
    assert ADVERSARIAL_QUALIFICATION_INTERFACE == "AdversarialQualification@1"
    assert PROCEDURE_PROMOTION_NOMINATION_INTERFACE == "ProcedurePromotionNomination@1"
    assert (
        REFACTOR_PROCEDURE_COMPILATION_RECEIPT_INTERFACE
        == "RefactorProcedureCompilationReceipt@1"
    )
    assert PROCEDURE_ADAPTER_CONTRACT_VERSION == "1"
    assert ANALYZER_ID.endswith("procedure_adapter@1")
    assert MODULE_PATH.is_file()
    assert TEST_PATH.is_file()
    for relative in WRITE_SCOPE:
        assert (ROOT / relative).is_file()


def test_authority_flags_cannot_self_authorize() -> None:
    assert AUTHORITY == "procedure compilation"
    assert AUTHORITY_OWNER == "ipfs_accelerate_py"
    assert PROCEDURE_AUTHORITY == "ProofCarryingProcedureCompiler"
    assert PROCEDURE_AUTHORITY_INTERFACE == "ProofCarryingProcedureCompiler@1"
    assert PROCEDURE_AUTHORITY_OWNER == "ipfs_accelerate_py"
    assert PROCEDURE_CAN_AUTHORIZE_COMPLETION is False
    assert PROCEDURE_CAN_AUTHORIZE_TRANSITION is False
    assert PROCEDURE_CAN_CREATE_AUTHORITY is False
    assert PROCEDURE_CAN_CREATE_PROCEDURE_AUTHORITY is False
    assert PROCEDURE_CANNOT_SELF_CERTIFY is True
    assert PROCEDURE_CANNOT_SELF_PROMOTE is True
    assert HELD_OUT_QUALIFICATION_REQUIRED_FOR_PROMOTION is True
    assert ADVERSARIAL_QUALIFICATION_ROUTE_EXISTS is True
    assert CANDIDATES_REMAIN_UNPROMOTED is True
    assert HOLES_PRESERVED is True
    assert VALIDATION_NEVER_DROPPED is True
    assert PATHS_NEVER_BECOME_PARAMETERS is True
    assert VECTOR_SIMILARITY_IS_AUTHORITY is False
    assert PROJECTION_CLUSTERING_IS_AUTHORITY is False
    assert MODEL_OUTPUT_IS_PROPOSAL_ONLY is True
    assert TEST_PASS_IS_NOT_COMPLETION is True
    assert TEST_PASS_IS_NOT_PROOF is True
    assert MARKDOWN_IS_NOT_COMPLETION is True
    assert WORKER_SELF_APPROVAL is False
    assert DUCKLAKE_IS_AUTHORITY is False
    assert SOFT_SIGNALS_CANNOT_OVERRIDE_HARD_CONSTRAINTS is True
    assert ADAPTER_IS_NOMINATION_ONLY is True
    assert RAW_SOURCE_REQUIRED is True
    assert NETWORK_DENIED is True
    assert NETWORK_DENY == "deny"
    assert DRY_RUN_IS_DETERMINISTIC is True
    assert DRY_RUN_MUTATES is False
    assert UNKNOWN_REMAINS_UNKNOWN is True
    assert INCOMPLETE_CONTRACTS_ARE_TYPED_TERMINALS is True
    assert GENERAL_PYTHON_EQUIVALENCE_CLAIMED is False
    assert IMPLICIT_INSTALL_FORBIDDEN is True
    assert IMPLICIT_NETWORK_FORBIDDEN is True
    assert DECLARED_COMPILATION_STATUSES == {
        "nominated",
        "incomplete",
        "rejected",
        "blocked",
        "unknown",
        "refused",
    }
    assert DECLARED_OUTCOMES == {"accepted", "rejected"}
    assert DECLARED_QUALIFICATION_ROUTES == {"held_out", "adversarial"}
    assert DECLARED_QUALIFICATION_STATUSES == {"passed", "failed", "missing"}
    profile = procedure_adapter_cid_profile()
    assert profile["codec"] == "dag-json"
    assert "not universal meaning" in profile["rule"]


def test_module_defines_predicted_symbols_not_capsule_family() -> None:
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    names = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
    assert "ProofCarryingProcedureRefactorAdapter" in names
    assert "NormalizedRefactorTrajectory" in names
    assert "AntiUnifiedRefactorPlan" in names
    assert "InferredProcedureContract" in names
    assert "PreservedHole" in names
    assert "HeldOutQualification" in names
    assert "AdversarialQualification" in names
    assert "ProcedurePromotionNomination" in names
    assert "RefactorProcedureCompilationReceipt" in names
    for capsule in CAPSULE_TYPES:
        assert capsule not in names
    assert_not_competing_capsule_family()
    exports = provider_free_exports()
    assert "ProofCarryingProcedureRefactorAdapter" in exports
    assert "compile_refactor_procedure" in exports
    assert "dry_run_refactor_procedure" in exports
    assert "anti_unify_refactor_plans" in exports
    assert "infer_procedure_contract" in exports
    assert "preserve_holes" in exports
    assert "nominate_procedure_promotion" in exports


def test_protected_paths_are_not_owned_write_scope() -> None:
    owned = set(WRITE_SCOPE)
    for relative in PROTECTED_PATHS:
        assert relative not in owned
        assert (ROOT / relative).exists()


def test_module_does_not_implement_forbidden_procedure_shortcuts() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    assert not (names & FORBIDDEN_PROCEDURE_NAMES)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    assert not any("procedure_compiler" in name for name in imported)
    descriptor = procedure_adapter_descriptor()
    assert descriptor["interface"] == PROOF_CARRYING_PROCEDURE_REFACTOR_ADAPTER_INTERFACE
    assert descriptor["procedure_authority"] == PROCEDURE_AUTHORITY
    assert descriptor["network"] == NETWORK_DENY
    assert descriptor["nomination_only"] is True
    assert descriptor["procedure_cannot_self_certify"] is True
    assert descriptor["procedure_cannot_self_promote"] is True
    assert descriptor["held_out_or_adversarial_qualification_required"] is True
    assert descriptor["holes_preserved"] is True
    assert descriptor["validation_never_dropped"] is True
    assert descriptor["paths_never_become_parameters"] is True
    assert descriptor["candidates_remain_unpromoted"] is True
    assert descriptor["claims_general_equivalence"] is False
    forbids = set(descriptor["forbids"])
    assert "self_certify" in forbids
    assert "self_promote" in forbids
    assert "drop_holes" in forbids
    assert "omit_validation" in forbids
    assert "path_parameter" in forbids
    assert "promote_without_qualification" in forbids
    assert "open_network" in forbids


def test_two_accepted_waves_anti_unify_and_nominate_after_qualification() -> None:
    receipt = _compile()
    assert receipt.status == CompilationStatus.NOMINATED.value
    assert receipt.adapter_is_nomination_only is True
    assert receipt.can_authorize_completion is False
    assert receipt.can_authorize_transition is False
    assert receipt.can_create_procedure_authority is False
    assert receipt.promoted is False
    assert receipt.self_certified is False
    assert receipt.candidates_remain_unpromoted is True
    assert receipt.holes_preserved is True
    assert receipt.network == NETWORK_DENY
    assert receipt.mutated is False
    assert receipt.deterministic is True
    assert receipt.typed_terminal is False
    assert QualificationRoute.HELD_OUT.value in receipt.qualification_routes
    assert QualificationRoute.ADVERSARIAL.value in receipt.qualification_routes
    assert receipt.held_out_status == QualificationStatus.PASSED.value
    assert receipt.adversarial_status == QualificationStatus.PASSED.value
    assert receipt.plan_cid
    assert receipt.contract_cid
    assert receipt.nomination_cid
    restored = decode_canonical_receipt(encode_canonical_receipt(receipt))
    assert restored == receipt
    adapter = ProofCarryingProcedureRefactorAdapter()
    again = adapter.compile(
        waves=[_wave("wave-a"), _wave("wave-b"), _wave("wave-held")],
        transitions=[
            _transition("t1", wave_label="wave-a"),
            _transition(
                "t2",
                wave_label="wave-b",
                packet_cid=_cid("packet-wave-b"),
                wave_receipt_cid=_cid("wave-b"),
            ),
        ],
        held_out=[_held_out()],
        adversarial=[_adversarial()],
    )
    assert again.receipt_cid == receipt.receipt_cid


def test_validation_union_is_never_dropped() -> None:
    extra = "python3 -m pytest -q tests/test_extra.py"
    receipt = _compile(
        transitions=[
            _transition("t1", wave_label="wave-a"),
            _transition(
                "t2",
                wave_label="wave-b",
                packet_cid=_cid("packet-wave-b"),
                wave_receipt_cid=_cid("wave-b"),
                validation_commands=list(VALIDATION) + [extra],
            ),
        ],
        held_out=[_held_out()],
    )
    assert receipt.status == CompilationStatus.NOMINATED.value
    plan = anti_unify_refactor_plans(
        [
            compile_normalized_trajectory(
                tree_id=TREE_ID,
                wave_receipt_cid=_cid("wave-a"),
                packet_cid=_cid("packet-wave-a"),
                transition_cid=_cid("t1"),
                write_paths=list(WRITE_PATHS),
                validation_commands=list(VALIDATION),
                raw_source_cids=[_cid("source")],
                preconditions=["accepted_predecessor", "exact_preimage", "raw_source"],
                effects=["bounded_internal_implementation"],
                rollback="restore_exact_preimages",
                outcome="accepted",
            ),
            compile_normalized_trajectory(
                tree_id=TREE_ID,
                wave_receipt_cid=_cid("wave-b"),
                packet_cid=_cid("packet-wave-b"),
                transition_cid=_cid("t2"),
                write_paths=list(WRITE_PATHS),
                validation_commands=list(VALIDATION) + [extra],
                raw_source_cids=[_cid("source")],
                preconditions=["accepted_predecessor", "exact_preimage", "raw_source"],
                effects=["bounded_internal_implementation"],
                rollback="restore_exact_preimages",
                outcome="accepted",
            ),
        ]
    )
    assert extra in plan.shared_validation_commands
    assert VALIDATION[0] in plan.shared_validation_commands
    contract = infer_procedure_contract(plan)
    assert extra in contract.postconditions
    assert contract.rollback == "restore_exact_preimages"


def test_differing_packet_cids_become_preserved_allowed_holes() -> None:
    receipt = _compile()
    assert receipt.status == CompilationStatus.NOMINATED.value
    assert receipt.hole_cids
    assert receipt.holes_preserved is True
    hole = compile_preserved_hole(
        hole_id="hole:packet",
        hole_type="SELECT_ONE_OF_ALLOWED_SYMBOLS",
        origin="packet_cid",
    )
    assert hole.hole_type in ALLOWED_HOLE_TYPES
    with pytest.raises(ProcedureAdapterError, match="forbidden hole type"):
        compile_preserved_hole(
            hole_id="hole:bad",
            hole_type="RELEASE_PROMOTION",
            origin="promotion",
        )
    for forbidden in FORBIDDEN_HOLE_TYPES:
        with pytest.raises(ProcedureAdapterError, match="forbidden hole type"):
            compile_preserved_hole(
                hole_id="hole:forbidden",
                hole_type=forbidden,
                origin="forbidden",
            )


def test_paths_never_become_parameters_and_disjoint_paths_counterexemplify() -> None:
    first = compile_normalized_trajectory(
        tree_id=TREE_ID,
        wave_receipt_cid=_cid("wave-a"),
        packet_cid=_cid("packet-wave-a"),
        transition_cid=_cid("t1"),
        write_paths=["pkg/mod.py"],
        validation_commands=list(VALIDATION),
        raw_source_cids=[_cid("source")],
        preconditions=["accepted_predecessor", "exact_preimage", "raw_source"],
        effects=["bounded_internal_implementation"],
        rollback="restore_exact_preimages",
        outcome="accepted",
    )
    second = compile_normalized_trajectory(
        tree_id=TREE_ID,
        wave_receipt_cid=_cid("wave-b"),
        packet_cid=_cid("packet-wave-b"),
        transition_cid=_cid("t2"),
        write_paths=["pkg/other.py"],
        validation_commands=list(VALIDATION),
        raw_source_cids=[_cid("source")],
        preconditions=["accepted_predecessor", "exact_preimage", "raw_source"],
        effects=["bounded_internal_implementation"],
        rollback="restore_exact_preimages",
        outcome="accepted",
    )
    plan = anti_unify_refactor_plans([first, second])
    assert plan.anti_unified is False
    assert plan.counterexample_cids
    assert PATHS_NEVER_BECOME_PARAMETERS is True
    receipt = _compile(
        waves=[
            _wave("wave-a", write_paths=["pkg/mod.py"], packet_cids=[_cid("packet-wave-a")]),
            _wave("wave-b", write_paths=["pkg/other.py"], packet_cids=[_cid("packet-wave-b")]),
        ],
        transitions=[
            _transition(
                "t1",
                wave_label="wave-a",
                write_paths=["pkg/mod.py"],
                packet_cid=_cid("packet-wave-a"),
                wave_receipt_cid=_cid("wave-a"),
            ),
            _transition(
                "t2",
                wave_label="wave-b",
                write_paths=["pkg/other.py"],
                packet_cid=_cid("packet-wave-b"),
                wave_receipt_cid=_cid("wave-b"),
            ),
        ],
        held_out=(),
        adversarial=[_adversarial()],
    )
    assert receipt.status == CompilationStatus.REJECTED.value
    assert receipt.typed_terminal is True
    assert receipt.can_authorize_completion is False


def test_missing_qualification_route_is_incomplete_typed_terminal() -> None:
    receipt = _compile(held_out=(), adversarial=())
    assert receipt.status == CompilationStatus.INCOMPLETE.value
    assert receipt.typed_terminal is True
    assert receipt.promoted is False
    assert receipt.qualification_routes == ()
    assert receipt.held_out_status == QualificationStatus.MISSING.value
    assert receipt.adversarial_status == QualificationStatus.MISSING.value
    assert receipt.can_authorize_completion is False


def test_held_out_only_or_adversarial_only_can_qualify() -> None:
    held_only = _compile(adversarial=())
    assert held_only.status == CompilationStatus.NOMINATED.value
    assert held_only.qualification_routes == (QualificationRoute.HELD_OUT.value,)
    adv_only = _compile(held_out=())
    assert adv_only.status == CompilationStatus.NOMINATED.value
    assert adv_only.qualification_routes == (QualificationRoute.ADVERSARIAL.value,)


def test_held_out_must_be_disjoint_from_training_trajectories() -> None:
    receipt = _compile(
        held_out=[
            _transition(
                "t1",
                wave_label="wave-a",
                packet_cid=_cid("packet-wave-a"),
                wave_receipt_cid=_cid("wave-a"),
                transition_cid=_cid("t1"),
            )
        ],
        adversarial=(),
    )
    assert receipt.status == CompilationStatus.REJECTED.value
    assert receipt.held_out_status == QualificationStatus.FAILED.value
    assert receipt.typed_terminal is True


def test_adversarial_must_fail_closed_claims_are_refused_by_adapter() -> None:
    receipt = _compile(
        held_out=(),
        adversarial=[
            _adversarial(
                case_id="adv:self",
                polarity=AdversarialPolarity.MUST_FAIL_CLOSED.value,
                claim="self_certify",
            )
        ],
    )
    assert receipt.status == CompilationStatus.NOMINATED.value
    assert receipt.adversarial_status == QualificationStatus.PASSED.value


def test_raw_countermodel_cannot_qualify_until_replay() -> None:
    receipt = _compile(
        held_out=(),
        adversarial=[
            _adversarial(
                case_id="adv:raw",
                source_authority="replayed_counterexample",
                replayed=False,
            )
        ],
    )
    assert receipt.status == CompilationStatus.REJECTED.value
    assert receipt.adversarial_status == QualificationStatus.FAILED.value


def test_single_accepted_trajectory_remains_unknown() -> None:
    receipt = compile_refactor_procedure(
        waves=[_wave("wave-a")],
        transitions=[_transition("t1", wave_label="wave-a")],
        adversarial=[_adversarial()],
    )
    assert receipt.status == CompilationStatus.UNKNOWN.value
    assert receipt.typed_terminal is True
    assert receipt.can_authorize_completion is False
    assert UNKNOWN_REMAINS_UNKNOWN is True


def test_negative_episodes_are_retained_and_excluded_from_anti_unification() -> None:
    receipt = _compile(
        transitions=[
            _transition("t1", wave_label="wave-a"),
            _transition(
                "t2",
                wave_label="wave-b",
                packet_cid=_cid("packet-wave-b"),
                wave_receipt_cid=_cid("wave-b"),
            ),
            _transition(
                "neg",
                wave_label="wave-a",
                outcome=TrajectoryOutcome.REJECTED.value,
                evidence_class="countermodel",
                packet_cid=_cid("packet-wave-a"),
                wave_receipt_cid=_cid("wave-a"),
                transition_cid=_cid("neg"),
            ),
        ]
    )
    assert receipt.status == CompilationStatus.NOMINATED.value
    assert _cid("neg") in receipt.negative_transition_cids
    assert len(receipt.trajectory_cids) == 2


def test_vector_evidence_cannot_admit_a_procedure() -> None:
    with pytest.raises(ProcedureAdapterError, match="cannot admit"):
        _compile(vector_evidence={"evidence_class": "vector_candidate"})
    with pytest.raises(ProcedureAdapterError, match="cannot admit"):
        _compile(vector_evidence={"evidence_class": "model_hypothesis", "admit_procedure": True})
    with pytest.raises(ProcedureAdapterError, match="raw-source"):
        _compile(
            vector_evidence={
                "evidence_class": "heuristic",
                "suppress_raw_source": True,
            }
        )
    with pytest.raises(ProcedureAdapterError, match="cannot admit"):
        _transition(evidence_class="vector_candidate")
        compile_refactor_procedure(
            waves=[_wave("wave-a"), _wave("wave-b")],
            transitions=[
                _transition("t1", wave_label="wave-a", evidence_class="vector_candidate"),
                _transition(
                    "t2",
                    wave_label="wave-b",
                    packet_cid=_cid("packet-wave-b"),
                    wave_receipt_cid=_cid("wave-b"),
                ),
            ],
            adversarial=[_adversarial()],
        )


def test_raw_source_and_write_scope_are_required() -> None:
    with pytest.raises(ProcedureAdapterError, match="raw source"):
        _compile(
            transitions=[
                _transition("t1", wave_label="wave-a", raw_source_cids=[]),
                _transition(
                    "t2",
                    wave_label="wave-b",
                    packet_cid=_cid("packet-wave-b"),
                    wave_receipt_cid=_cid("wave-b"),
                ),
            ]
        )
    with pytest.raises(ProcedureAdapterError, match="unrestricted scope"):
        _compile(
            waves=[_wave("wave-a", write_paths=[]), _wave("wave-b"), _wave("wave-held")],
        )
    with pytest.raises(ProcedureAdapterError, match="unrestricted scope"):
        _compile(
            waves=[
                _wave("wave-a", write_paths=["pkg/*.py"]),
                _wave("wave-b"),
                _wave("wave-held"),
            ]
        )
    with pytest.raises(ProcedureAdapterError, match="unrestricted scope"):
        _compile(
            waves=[
                _wave("wave-a", write_paths=["/tmp/pkg/mod.py"]),
                _wave("wave-b"),
                _wave("wave-held"),
            ]
        )
    with pytest.raises(ProcedureAdapterError, match="unrestricted scope"):
        _compile(
            waves=[
                _wave("wave-a", write_paths=["pkg/../secret.py"]),
                _wave("wave-b"),
                _wave("wave-held"),
            ]
        )


def test_missing_predecessors_fail_closed() -> None:
    with pytest.raises(ProcedureAdapterError, match="SPAR-025"):
        compile_refactor_procedure(
            waves=None,
            transitions=[_transition("t1")],
        )
    with pytest.raises(ProcedureAdapterError, match="SPAR-033"):
        compile_refactor_procedure(
            waves=[_wave("wave-a")],
            transitions=None,
        )


def test_tree_mismatch_fails_closed() -> None:
    with pytest.raises(ProcedureAdapterError, match="tree_id"):
        _compile(
            transitions=[
                _transition("t1", wave_label="wave-a", tree_id=OTHER_TREE),
                _transition(
                    "t2",
                    wave_label="wave-b",
                    packet_cid=_cid("packet-wave-b"),
                    wave_receipt_cid=_cid("wave-b"),
                ),
            ]
        )


def test_body_free_artifacts_reject_source_dumps() -> None:
    with pytest.raises(ProcedureAdapterError, match="body-free"):
        _compile(
            transitions=[
                _transition("t1", wave_label="wave-a", source="def f():\n    return 1\n"),
                _transition(
                    "t2",
                    wave_label="wave-b",
                    packet_cid=_cid("packet-wave-b"),
                    wave_receipt_cid=_cid("wave-b"),
                ),
            ]
        )


def test_network_is_denied() -> None:
    with pytest.raises(ProcedureAdapterError, match="network is denied"):
        _compile(network="allow")


def test_existing_procedure_authority_port_cannot_self_certify() -> None:
    seen: dict[str, Any] = {}

    def promote(**request: Any) -> dict[str, Any]:
        seen.update(request)
        return {
            "producer": "ProofCarryingProcedureCompiler",
            "authority_response_cid": _cid("authority"),
            "self_certified": False,
            "promoted": False,
        }

    receipt = _compile(promote=promote)
    assert receipt.status == CompilationStatus.NOMINATED.value
    assert receipt.promoted is False
    assert seen["network"] == NETWORK_DENY
    assert seen["self_certified"] is False
    assert seen["can_authorize_completion"] is False

    def self_certify(**request: Any) -> dict[str, Any]:
        del request
        return {"producer": ANALYZER_ID, "self_certified": True}

    with pytest.raises(ProcedureAdapterError, match="self-certify"):
        _compile(promote=self_certify)

    adapter = ProofCarryingProcedureRefactorAdapter()
    with pytest.raises(ProcedureAdapterError, match="self-certify"):
        compile_refactor_procedure(
            waves=[_wave("wave-a"), _wave("wave-b"), _wave("wave-held")],
            transitions=[
                _transition("t1", wave_label="wave-a"),
                _transition(
                    "t2",
                    wave_label="wave-b",
                    packet_cid=_cid("packet-wave-b"),
                    wave_receipt_cid=_cid("wave-b"),
                ),
            ],
            held_out=[_held_out()],
            promote=adapter.compile,
        )


def test_spar032_promoted_candidates_fail_closed() -> None:
    with pytest.raises(ProcedureAdapterError, match="unpromoted"):
        _compile(
            synthesis_receipt={
                "tree_id": TREE_ID,
                "candidates_remain_unpromoted": False,
                "status": "nominated",
            }
        )
    with pytest.raises(ProcedureAdapterError, match="equivalence"):
        _compile(
            normalization_receipt={
                "tree_id": TREE_ID,
                "claims_general_equivalence": True,
            }
        )


def test_identity_excludes_observational_fields() -> None:
    trajectory = compile_normalized_trajectory(
        tree_id=TREE_ID,
        wave_receipt_cid=_cid("wave-a"),
        packet_cid=_cid("packet-wave-a"),
        transition_cid=_cid("t1"),
        write_paths=list(WRITE_PATHS),
        validation_commands=list(VALIDATION),
        raw_source_cids=[_cid("source")],
        preconditions=["accepted_predecessor", "exact_preimage", "raw_source"],
        effects=["bounded_internal_implementation"],
        rollback="restore_exact_preimages",
        outcome="accepted",
    )
    payload = trajectory.to_dict()
    assert not (IDENTITY_EXCLUDED_FIELDS & set(payload))
    dirty = dict(payload)
    dirty["timestamp"] = "now"
    with pytest.raises(ProcedureAdapterError, match="observational"):
        compile_normalized_trajectory(
            tree_id=TREE_ID,
            wave_receipt_cid=_cid("wave-a"),
            packet_cid=_cid("packet-wave-a"),
            transition_cid=_cid("t1"),
            write_paths=list(WRITE_PATHS),
            validation_commands=list(VALIDATION),
            raw_source_cids=[_cid("source")],
            preconditions=["accepted_predecessor", "exact_preimage", "raw_source"],
            effects=["bounded_internal_implementation"],
            rollback="restore_exact_preimages",
            outcome="accepted",
            timestamp="now",
        )


def test_receipt_cannot_claim_authority_flags() -> None:
    receipt = _compile()
    payload = receipt.to_dict()
    payload["can_authorize_completion"] = True
    payload["receipt_cid"] = cid_for_dag_json(
        {key: value for key, value in payload.items() if key != "receipt_cid"}
    )
    with pytest.raises(ProcedureAdapterError, match="can_authorize_completion"):
        RefactorProcedureCompilationReceipt.from_dict(payload)
    payload = receipt.to_dict()
    payload["promoted"] = True
    payload["receipt_cid"] = cid_for_dag_json(
        {key: value for key, value in payload.items() if key != "receipt_cid"}
    )
    with pytest.raises(ProcedureAdapterError, match="cannot promote"):
        RefactorProcedureCompilationReceipt.from_dict(payload)
    payload = receipt.to_dict()
    payload["self_certified"] = True
    payload["receipt_cid"] = cid_for_dag_json(
        {key: value for key, value in payload.items() if key != "receipt_cid"}
    )
    with pytest.raises(ProcedureAdapterError, match="self-certify"):
        RefactorProcedureCompilationReceipt.from_dict(payload)
    payload = receipt.to_dict()
    payload["adapter_is_nomination_only"] = False
    payload["receipt_cid"] = cid_for_dag_json(
        {key: value for key, value in payload.items() if key != "receipt_cid"}
    )
    with pytest.raises(ProcedureAdapterError, match="nomination_only"):
        RefactorProcedureCompilationReceipt.from_dict(payload)


def test_nomination_requires_qualification_routes() -> None:
    plan_trajectories = [
        compile_normalized_trajectory(
            tree_id=TREE_ID,
            wave_receipt_cid=_cid("wave-a"),
            packet_cid=_cid("packet-wave-a"),
            transition_cid=_cid("t1"),
            write_paths=list(WRITE_PATHS),
            validation_commands=list(VALIDATION),
            raw_source_cids=[_cid("source")],
            preconditions=["accepted_predecessor", "exact_preimage", "raw_source"],
            effects=["bounded_internal_implementation"],
            rollback="restore_exact_preimages",
            outcome="accepted",
        ),
        compile_normalized_trajectory(
            tree_id=TREE_ID,
            wave_receipt_cid=_cid("wave-b"),
            packet_cid=_cid("packet-wave-b"),
            transition_cid=_cid("t2"),
            write_paths=list(WRITE_PATHS),
            validation_commands=list(VALIDATION),
            raw_source_cids=[_cid("source")],
            preconditions=["accepted_predecessor", "exact_preimage", "raw_source"],
            effects=["bounded_internal_implementation"],
            rollback="restore_exact_preimages",
            outcome="accepted",
        ),
    ]
    plan = anti_unify_refactor_plans(plan_trajectories)
    contract = infer_procedure_contract(plan)
    holes = preserve_holes(plan)
    assert all(hole.hole_type in ALLOWED_HOLE_TYPES for hole in holes)
    with pytest.raises(ProcedureAdapterError, match="qualification"):
        nominate_procedure_promotion(plan, contract, ())
    nomination = nominate_procedure_promotion(
        plan, contract, (QualificationRoute.ADVERSARIAL.value,)
    )
    assert nomination.nominated is True
    assert nomination.promoted is False
    assert nomination.self_certified is False
    restored = ProcedurePromotionNomination.from_dict(nomination.to_dict())
    assert restored == nomination


def test_dry_run_is_deterministic_and_does_not_mutate() -> None:
    first = dry_run_refactor_procedure(
        waves=[_wave("wave-a"), _wave("wave-b"), _wave("wave-held")],
        transitions=[
            _transition("t1", wave_label="wave-a"),
            _transition(
                "t2",
                wave_label="wave-b",
                packet_cid=_cid("packet-wave-b"),
                wave_receipt_cid=_cid("wave-b"),
            ),
        ],
        held_out=[_held_out()],
        adversarial=[_adversarial()],
    )
    second = ProofCarryingProcedureRefactorAdapter().dry_run(
        waves=[_wave("wave-a"), _wave("wave-b"), _wave("wave-held")],
        transitions=[
            _transition("t1", wave_label="wave-a"),
            _transition(
                "t2",
                wave_label="wave-b",
                packet_cid=_cid("packet-wave-b"),
                wave_receipt_cid=_cid("wave-b"),
            ),
        ],
        held_out=[_held_out()],
        adversarial=[_adversarial()],
    )
    assert first.receipt_cid == second.receipt_cid
    assert first.mutated is False
    assert first.deterministic is True
    with pytest.raises(ProcedureAdapterError, match="cannot mutate"):
        compile_refactor_procedure(
            waves=[_wave("wave-a"), _wave("wave-b")],
            transitions=[
                _transition("t1", wave_label="wave-a"),
                _transition(
                    "t2",
                    wave_label="wave-b",
                    packet_cid=_cid("packet-wave-b"),
                    wave_receipt_cid=_cid("wave-b"),
                ),
            ],
            adversarial=[_adversarial()],
            mutate=True,
        )


def test_qualification_artifacts_cannot_self_certify() -> None:
    held = HeldOutQualification(
        status=QualificationStatus.PASSED.value,
        held_out_trajectory_cids=[_cid("held-out")],
    )
    assert held.self_certified is False
    payload = held.to_dict()
    payload["self_certified"] = True
    payload["qualification_cid"] = cid_for_dag_json(
        {key: value for key, value in payload.items() if key != "qualification_cid"}
    )
    with pytest.raises(ProcedureAdapterError, match="self-certify"):
        HeldOutQualification.from_dict(payload)
    adv = AdversarialQualification(
        status=QualificationStatus.PASSED.value,
        case_ids=["adv:boundary"],
    )
    assert adv.self_certified is False
    payload = adv.to_dict()
    payload["self_certified"] = True
    payload["qualification_cid"] = cid_for_dag_json(
        {key: value for key, value in payload.items() if key != "qualification_cid"}
    )
    with pytest.raises(ProcedureAdapterError, match="self-certify"):
        AdversarialQualification.from_dict(payload)
