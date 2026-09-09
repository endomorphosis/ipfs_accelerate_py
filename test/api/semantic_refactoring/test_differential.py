"""Independent contract tests for SPAR-028 DifferentialExecutionReceipt@1."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.utils.cid_utils import cid_for_bytes, cid_for_dag_json
from ipfs_accelerate_py.agent_supervisor.semantic_refactoring.differential import (
    ANALYZER_ID,
    AUTHORITY,
    AUTHORITY_OWNER,
    DECLARED_COMPARISON_OUTCOMES,
    DECLARED_DIFFERENTIAL_STATUSES,
    DECLARED_DIMENSIONS,
    DECLARED_WORKFLOW_ROLES,
    DIFFERENTIAL_CAN_AUTHORIZE_COMPLETION,
    DIFFERENTIAL_CAN_AUTHORIZE_TRANSITION,
    DIFFERENTIAL_CAN_CREATE_AUTHORITY,
    DIFFERENTIAL_CONTRACT_VERSION,
    DIFFERENTIAL_EXECUTION_INTERFACE,
    DIFFERENTIAL_EXECUTION_RECEIPT_INTERFACE,
    DIMENSION_COMPARISON_INTERFACE,
    DUCKLAKE_IS_AUTHORITY,
    EXECUTOR_IS_NOMINATION_ONLY,
    FORBIDDEN_EXECUTION_NAMES,
    GENERAL_PYTHON_EQUIVALENCE_CLAIMED,
    GOAL_ID,
    HERMETIC_PAIRED_EXECUTION,
    IDENTITY_EXCLUDED_FIELDS,
    IMPLICIT_INSTALL_FORBIDDEN,
    IMPLICIT_NETWORK_FORBIDDEN,
    MARKDOWN_IS_NOT_COMPLETION,
    MODEL_OUTPUT_IS_PROPOSAL_ONLY,
    NETWORK_DENIED,
    NETWORK_DENY,
    OBSERVATION_PROFILE_INTERFACE,
    PROGRAM,
    PROJECTION_CLUSTERING_IS_AUTHORITY,
    RAW_SOURCE_REQUIRED,
    REQUIRED_TRACE_DIMENSIONS,
    SOFT_SIGNALS_CANNOT_OVERRIDE_HARD_CONSTRAINTS,
    TASK_ID,
    TEST_PASS_IS_NOT_COMPLETION,
    TRACES_PROVE_ONLY_OBSERVATIONS,
    UNRESOLVED_DYNAMICS_LOWER_AUTONOMY,
    VECTOR_SIMILARITY_IS_AUTHORITY,
    WORKER_SELF_APPROVAL,
    WORKFLOW_OBSERVATION_INTERFACE,
    ComparisonOutcome,
    DifferentialExecutionError,
    DifferentialExecutionReceipt,
    DifferentialExecutor,
    DifferentialStatus,
    DimensionComparison,
    ObservationProfile,
    TraceDimension,
    WorkflowObservation,
    WorkflowRole,
    assert_not_competing_capsule_family,
    compare_declared_dimensions,
    compile_observation_profile,
    decode_canonical_observation,
    decode_canonical_profile,
    decode_canonical_receipt,
    differential_cid_profile,
    differential_descriptor,
    encode_canonical_observation,
    encode_canonical_profile,
    encode_canonical_receipt,
    execute_differential_pair,
    provider_free_exports,
    run_hermetic_workflow,
)


ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    ROOT
    / "ipfs_accelerate_py"
    / "agent_supervisor"
    / "semantic_refactoring"
    / "differential.py"
)
TEST_PATH = Path(__file__).resolve()
WRITE_SCOPE = (
    "ipfs_accelerate_py/agent_supervisor/semantic_refactoring/differential.py",
    "test/api/semantic_refactoring/test_differential.py",
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


def _obs(label: str) -> dict[str, str]:
    return {"kind": label, "projection_cid": _cid(label)}


def _dimension_cids(label: str = "same") -> dict[str, dict[str, str]]:
    return {dimension: _obs(f"{label}:{dimension}") for dimension in DECLARED_DIMENSIONS}


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
        "raw_source_required": True,
        "adapter_is_nomination_only": True,
        "datasets_owns_selection": True,
    }
    fields.update(overrides)
    return fields


def _workflow(*, role: str, label: str = "same", **overrides: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "role": role,
        "workflow_id": f"{role}-workflow",
        "network": NETWORK_DENY,
        "hermetic": True,
        "raw_source_cids": [_cid("source")],
        "observations": _dimension_cids(label),
        "unsupported_dimensions": [],
        "unobserved_dimensions": [],
    }
    fields.update(overrides)
    return fields


def _execute(**overrides: Any) -> DifferentialExecutionReceipt:
    fields: dict[str, Any] = {
        "old_workflow": _workflow(role="old"),
        "new_workflow": _workflow(role="new"),
        "wave": _wave(),
        "selection": _selection(),
    }
    fields.update(overrides)
    return execute_differential_pair(**fields)


def test_owned_paths_and_task_identity_are_exact() -> None:
    assert TASK_ID == "SPAR-028"
    assert GOAL_ID == "SPAR-G052"
    assert PROGRAM == "semantic-preserving-autonomous-remodularization-v1"
    assert DIFFERENTIAL_EXECUTION_INTERFACE == "DifferentialExecution@1"
    assert DIFFERENTIAL_EXECUTION_RECEIPT_INTERFACE == "DifferentialExecutionReceipt@1"
    assert OBSERVATION_PROFILE_INTERFACE == "ObservationProfile@1"
    assert WORKFLOW_OBSERVATION_INTERFACE == "WorkflowObservation@1"
    assert DIMENSION_COMPARISON_INTERFACE == "DimensionComparison@1"
    assert DIFFERENTIAL_CONTRACT_VERSION == "1"
    assert ANALYZER_ID.endswith("differential@1")
    assert MODULE_PATH.is_file()
    assert TEST_PATH.is_file()
    for relative in WRITE_SCOPE:
        assert (ROOT / relative).is_file()


def test_authority_flags_cannot_self_authorize() -> None:
    assert AUTHORITY == "validation orchestration"
    assert AUTHORITY_OWNER == "ipfs_accelerate_py"
    assert DIFFERENTIAL_CAN_AUTHORIZE_COMPLETION is False
    assert DIFFERENTIAL_CAN_AUTHORIZE_TRANSITION is False
    assert DIFFERENTIAL_CAN_CREATE_AUTHORITY is False
    assert VECTOR_SIMILARITY_IS_AUTHORITY is False
    assert PROJECTION_CLUSTERING_IS_AUTHORITY is False
    assert MODEL_OUTPUT_IS_PROPOSAL_ONLY is True
    assert TEST_PASS_IS_NOT_COMPLETION is True
    assert MARKDOWN_IS_NOT_COMPLETION is True
    assert WORKER_SELF_APPROVAL is False
    assert DUCKLAKE_IS_AUTHORITY is False
    assert SOFT_SIGNALS_CANNOT_OVERRIDE_HARD_CONSTRAINTS is True
    assert EXECUTOR_IS_NOMINATION_ONLY is True
    assert RAW_SOURCE_REQUIRED is True
    assert NETWORK_DENIED is True
    assert HERMETIC_PAIRED_EXECUTION is True
    assert GENERAL_PYTHON_EQUIVALENCE_CLAIMED is False
    assert TRACES_PROVE_ONLY_OBSERVATIONS is True
    assert UNRESOLVED_DYNAMICS_LOWER_AUTONOMY is True
    assert IMPLICIT_INSTALL_FORBIDDEN is True
    assert IMPLICIT_NETWORK_FORBIDDEN is True
    assert REQUIRED_TRACE_DIMENSIONS == DECLARED_DIMENSIONS
    assert DECLARED_DIMENSIONS == (
        "outputs",
        "exceptions",
        "effects",
        "import_events",
        "registrations",
        "state_transitions",
        "resources",
        "declared_trace_projections",
    )
    assert DECLARED_COMPARISON_OUTCOMES == {
        "agree",
        "disagree",
        "unobserved",
        "unsupported",
    }
    assert DECLARED_WORKFLOW_ROLES == {"old", "new"}
    assert DECLARED_DIFFERENTIAL_STATUSES == {
        "equivalent",
        "divergent",
        "blocked",
        "rejected",
    }
    profile = differential_cid_profile()
    assert profile["codec"] == "dag-json"
    assert "not universal meaning" in profile["rule"]


def test_module_defines_predicted_symbols_not_capsule_family() -> None:
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    names = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
    assert "DifferentialExecutionReceipt" in names
    assert "DifferentialExecutor" in names
    assert "WorkflowObservation" in names
    assert "ObservationProfile" in names
    assert "DimensionComparison" in names
    for capsule in CAPSULE_TYPES:
        assert capsule not in names
    assert_not_competing_capsule_family()
    exports = provider_free_exports()
    assert "DifferentialExecutionReceipt" in exports
    assert "execute_differential_pair" in exports
    assert "compare_declared_dimensions" in exports
    assert "run_hermetic_workflow" in exports


def test_protected_paths_are_not_owned_write_scope() -> None:
    owned = set(WRITE_SCOPE)
    for relative in PROTECTED_PATHS:
        assert relative not in owned
        assert (ROOT / relative).exists()


def test_module_does_not_claim_general_equivalence_or_open_network() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    assert not (names & FORBIDDEN_EXECUTION_NAMES)
    descriptor = differential_descriptor()
    assert descriptor["interface"] == DIFFERENTIAL_EXECUTION_INTERFACE
    assert descriptor["receipt_interface"] == DIFFERENTIAL_EXECUTION_RECEIPT_INTERFACE
    assert descriptor["network"] == NETWORK_DENY
    assert descriptor["claims_general_equivalence"] is False
    assert descriptor["nomination_only"] is True
    forbids = set(descriptor["forbids"])
    assert "claim_general_equivalence" in forbids
    assert "open_network" in forbids
    assert "implicit_install" in forbids
    assert "admit_vector_agreement" in forbids
    assert "suppress_raw_source" in forbids


def test_matching_paired_observations_emit_equivalent_receipt() -> None:
    receipt = _execute()
    assert receipt.tree_id == TREE_ID
    assert receipt.wave_cid == _cid("wave")
    assert receipt.selection_cid == _cid("selection")
    assert receipt.packet_cid == _cid("packet")
    assert receipt.status == DifferentialStatus.EQUIVALENT.value
    assert receipt.required_dimensions_agree is True
    assert receipt.agreeing_required_dimensions == DECLARED_DIMENSIONS
    assert receipt.network == NETWORK_DENY
    assert receipt.executor_is_nomination_only is True
    assert receipt.claims_general_equivalence is False
    assert receipt.can_authorize_completion is False
    assert receipt.can_authorize_transition is False
    assert receipt.can_create_authority is False
    assert receipt.projection_is_authority is False
    assert receipt.write_paths == WRITE_PATHS
    assert receipt.raw_source_cids == (_cid("source"),)
    assert receipt.validation_commands == VALIDATION
    assert len(receipt.comparisons) == len(DECLARED_DIMENSIONS)
    assert {item.outcome for item in receipt.comparisons} == {ComparisonOutcome.AGREE.value}
    executor = DifferentialExecutor()
    again = executor.execute(
        old_workflow=_workflow(role="old"),
        new_workflow=_workflow(role="new"),
        wave=_wave(),
        selection=_selection(),
    )
    assert again.receipt_cid == receipt.receipt_cid


def test_output_mismatch_is_typed_terminal() -> None:
    with pytest.raises(DifferentialExecutionError, match="required dimensions disagree"):
        _execute(new_workflow=_workflow(role="new", label="other"))


def test_unsupported_required_dimension_is_blocker() -> None:
    workflow = _workflow(role="new")
    observations = dict(workflow["observations"])
    observations.pop("import_events")
    workflow["observations"] = observations
    workflow["unsupported_dimensions"] = ["import_events"]
    with pytest.raises(DifferentialExecutionError, match="unsupported required"):
        _execute(new_workflow=workflow)


def test_unobserved_required_dimension_is_blocker() -> None:
    workflow = _workflow(role="old")
    observations = dict(workflow["observations"])
    observations.pop("registrations")
    workflow["observations"] = observations
    with pytest.raises(DifferentialExecutionError, match="unobserved required"):
        _execute(old_workflow=workflow)


def test_optional_unobserved_dimension_does_not_block() -> None:
    profile = ObservationProfile(
        required_dimensions=("outputs", "exceptions"),
        optional_dimensions=("effects",),
    )
    old = _workflow(
        role="old",
        observations={
            "outputs": _obs("same:outputs"),
            "exceptions": _obs("same:exceptions"),
        },
        unobserved_dimensions=["effects"],
    )
    new = _workflow(
        role="new",
        observations={
            "outputs": _obs("same:outputs"),
            "exceptions": _obs("same:exceptions"),
        },
        unobserved_dimensions=["effects"],
    )
    receipt = _execute(
        old_workflow=old,
        new_workflow=new,
        observation_profile=profile,
    )
    assert receipt.required_dimensions_agree is True
    optional = [item for item in receipt.comparisons if item.dimension == "effects"]
    assert optional[0].outcome == ComparisonOutcome.UNOBSERVED.value
    assert optional[0].required is False


def test_network_is_denied() -> None:
    with pytest.raises(DifferentialExecutionError, match="network is denied"):
        _execute(old_workflow=_workflow(role="old", network="allow"))
    with pytest.raises(DifferentialExecutionError, match="implicit install/network"):
        _execute(old_workflow=_workflow(role="old", install=True))
    with pytest.raises(DifferentialExecutionError, match="implicit subprocess"):
        _execute(
            old_workflow=_workflow(role="old", command="python3 pkg/mod.py", observations={})
        )


def test_missing_predecessors_fail_closed() -> None:
    with pytest.raises(DifferentialExecutionError, match="SPAR-025"):
        execute_differential_pair(
            old_workflow=_workflow(role="old"),
            new_workflow=_workflow(role="new"),
            wave=None,
            selection=_selection(),
        )
    with pytest.raises(DifferentialExecutionError, match="SPAR-026"):
        execute_differential_pair(
            old_workflow=_workflow(role="old"),
            new_workflow=_workflow(role="new"),
            wave=_wave(),
            selection=None,
        )


def test_tree_mismatch_fails_closed() -> None:
    with pytest.raises(DifferentialExecutionError, match="SPAR-025 tree_id does not match SPAR-026"):
        _execute(wave=_wave(tree_id=OTHER_TREE))
    with pytest.raises(DifferentialExecutionError, match="SPAR-025 tree_id does not match SPAR-026"):
        _execute(selection=_selection(tree_id=OTHER_TREE))


def test_unapplied_wave_fails_closed() -> None:
    with pytest.raises(DifferentialExecutionError, match="applied"):
        _execute(wave=_wave(status="rolled_back"))


def test_packet_cid_must_bind_wave_and_selection() -> None:
    with pytest.raises(DifferentialExecutionError, match="packet_cid"):
        _execute(selection=_selection(packet_cid=_cid("other-packet")))


def test_write_paths_must_match_and_stay_exact() -> None:
    with pytest.raises(DifferentialExecutionError, match="write_paths"):
        _execute(wave=_wave(write_paths=["pkg/mod.py"]))
    with pytest.raises(DifferentialExecutionError, match="unrestricted scope"):
        _execute(selection=_selection(write_paths=[]))
    with pytest.raises(DifferentialExecutionError, match="unrestricted scope"):
        _execute(selection=_selection(write_paths=["pkg/*.py"]))
    with pytest.raises(DifferentialExecutionError, match="unrestricted scope"):
        _execute(selection=_selection(write_paths=["/tmp/pkg/mod.py"]))
    with pytest.raises(DifferentialExecutionError, match="unrestricted scope"):
        _execute(selection=_selection(write_paths=["pkg/../secret.py"]))


def test_missing_raw_source_is_typed_terminal() -> None:
    with pytest.raises(DifferentialExecutionError, match="raw source"):
        _execute(selection=_selection(raw_source_cids=[]))


def test_vectors_cannot_admit_agreement_or_suppress_raw_source() -> None:
    with pytest.raises(DifferentialExecutionError, match="raw-source"):
        _execute(
            vector_evidence={
                "evidence_class": "vector_candidate",
                "suppress_raw_source": True,
            }
        )
    with pytest.raises(DifferentialExecutionError, match="cannot admit"):
        _execute(
            vector_evidence={
                "evidence_class": "model_hypothesis",
                "admits_agreement": True,
            }
        )
    with pytest.raises(DifferentialExecutionError, match="cannot admit"):
        _execute(
            vector_evidence={
                "evidence_class": "heuristic",
                "admits_equivalence": True,
            }
        )


def test_body_free_observations_reject_source_dumps() -> None:
    with pytest.raises(DifferentialExecutionError, match="body-free"):
        _execute(
            old_workflow=_workflow(
                role="old",
                observations={"outputs": {"source": "def leaf(): return 1"}},
            )
        )


def test_hermetic_runner_is_invoked_with_network_deny() -> None:
    seen: dict[str, str] = {}

    def runner(*, role: str, network: str) -> dict[str, Any]:
        seen["role"] = role
        seen["network"] = network
        return _workflow(role=role)

    receipt = _execute(
        old_workflow={"runner": runner, "role": "old"},
        new_workflow={"runner": runner, "role": "new"},
    )
    assert seen["network"] == NETWORK_DENY
    assert receipt.required_dimensions_agree is True
    observation = run_hermetic_workflow(_workflow(role="old"), role="old")
    assert observation.role == WorkflowRole.OLD.value
    assert observation.network == NETWORK_DENY
    assert observation.hermetic is True


def test_general_equivalence_cannot_be_claimed() -> None:
    with pytest.raises(DifferentialExecutionError, match="general Python equivalence"):
        ObservationProfile(claims_general_equivalence=True)
    receipt = _execute()
    payload = receipt.to_dict()
    payload["claims_general_equivalence"] = True
    payload["receipt_cid"] = cid_for_dag_json(
        {key: value for key, value in payload.items() if key != "receipt_cid"}
    )
    with pytest.raises(DifferentialExecutionError, match="general Python equivalence"):
        DifferentialExecutionReceipt.from_dict(payload)


def test_receipt_cannot_claim_authority_flags() -> None:
    receipt = _execute()
    payload = receipt.to_dict()
    payload["can_authorize_completion"] = True
    payload["receipt_cid"] = cid_for_dag_json(
        {key: value for key, value in payload.items() if key != "receipt_cid"}
    )
    with pytest.raises(DifferentialExecutionError, match="can_authorize_completion"):
        DifferentialExecutionReceipt.from_dict(payload)
    payload = receipt.to_dict()
    payload["executor_is_nomination_only"] = False
    payload["receipt_cid"] = cid_for_dag_json(
        {key: value for key, value in payload.items() if key != "receipt_cid"}
    )
    with pytest.raises(DifferentialExecutionError, match="nomination_only"):
        DifferentialExecutionReceipt.from_dict(payload)


def test_round_trip_and_receipt_are_deterministic() -> None:
    first = _execute()
    second = _execute()
    assert first.receipt_cid == second.receipt_cid
    restored = decode_canonical_receipt(encode_canonical_receipt(first))
    assert restored == first
    assert restored.receipt_cid == first.receipt_cid
    profile = compile_observation_profile(None)
    assert decode_canonical_profile(encode_canonical_profile(profile)) == profile
    observation = run_hermetic_workflow(_workflow(role="old"), role="old")
    assert decode_canonical_observation(encode_canonical_observation(observation)) == observation
    comparisons = compare_declared_dimensions(
        observation,
        run_hermetic_workflow(_workflow(role="new"), role="new"),
        profile=profile,
    )
    assert all(item.outcome == ComparisonOutcome.AGREE.value for item in comparisons)


def test_identity_excludes_observational_fields() -> None:
    receipt = _execute()
    payload = receipt.to_dict()
    assert not (IDENTITY_EXCLUDED_FIELDS & set(payload))
    dirty = dict(payload)
    dirty["timestamp"] = "now"
    with pytest.raises(DifferentialExecutionError, match="observational"):
        DifferentialExecutionReceipt.from_dict(dirty)


def test_empty_validation_commands_fail_closed() -> None:
    with pytest.raises(DifferentialExecutionError, match="validation_commands"):
        _execute(selection=_selection(validation_commands=[]))


def test_import_events_and_trace_projections_are_compared() -> None:
    receipt = _execute()
    dimensions = {item.dimension: item for item in receipt.comparisons}
    assert TraceDimension.IMPORT_EVENTS.value in dimensions
    assert TraceDimension.REGISTRATIONS.value in dimensions
    assert TraceDimension.STATE_TRANSITIONS.value in dimensions
    assert TraceDimension.RESOURCES.value in dimensions
    assert TraceDimension.DECLARED_TRACE_PROJECTIONS.value in dimensions
    assert dimensions["import_events"].outcome == ComparisonOutcome.AGREE.value
    assert dimensions["declared_trace_projections"].outcome == ComparisonOutcome.AGREE.value


def test_dimension_comparison_round_trip() -> None:
    comparison = DimensionComparison(
        dimension="outputs",
        old_observation_cid=_cid("old-out"),
        new_observation_cid=_cid("old-out"),
        outcome=ComparisonOutcome.AGREE,
        required=True,
    )
    restored = DimensionComparison.from_dict(comparison.to_dict())
    assert restored == comparison
    assert restored.comparison_cid == comparison.comparison_cid
