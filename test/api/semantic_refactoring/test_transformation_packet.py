"""Independent contract tests for SPAR-019 RefactorTransformationPacket@1."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.utils.cid_utils import cid_for_bytes, cid_for_dag_json
from ipfs_accelerate_py.agent_supervisor.semantic_refactoring.partition_generators import (
    ConstraintClass,
    GeneratorKind,
    ProgramPartitionCandidate,
)
from ipfs_accelerate_py.agent_supervisor.semantic_refactoring.partition_policy import (
    ANALYZER_ID as SPAR014_ANALYZER_ID,
    compare_partition_candidates,
)
from ipfs_accelerate_py.agent_supervisor.semantic_refactoring.transformation_packet import (
    ANALYZER_ID,
    AUTHORITY,
    AUTHORITY_OWNER,
    BOUNDED_EDIT_INTERFACE,
    DECLARED_ALLOWED_EFFECTS,
    DECLARED_EDIT_KINDS,
    DECLARED_FORBIDDEN_EFFECTS,
    DRY_RUN_IS_DETERMINISTIC,
    DRY_RUN_MUTATES,
    DRY_RUN_RECEIPT_INTERFACE,
    DUCKLAKE_IS_AUTHORITY,
    EFFECT_SCOPE_INTERFACE,
    EXPECTED_DELTA_INTERFACE,
    GOAL_ID,
    IDENTITY_EXCLUDED_FIELDS,
    LEASE_FENCE_BINDING_INTERFACE,
    MARKDOWN_IS_NOT_COMPLETION,
    MODEL_OUTPUT_IS_PROPOSAL_ONLY,
    PACKET_CAN_AUTHORIZE_COMPLETION,
    PACKET_CAN_AUTHORIZE_TRANSITION,
    PACKET_CAN_CREATE_AUTHORITY,
    PACKET_CAN_RETIRE_FACADE,
    PACKET_CONTRACT_VERSION,
    PACKET_IS_NOMINATION_ONLY,
    PROGRAM,
    PROJECTION_CLUSTERING_IS_AUTHORITY,
    RAW_SOURCE_REQUIRED,
    REFACTOR_TRANSFORMATION_PACKET_INTERFACE,
    ROLLBACK_MODE,
    ROLLBACK_PLAN_INTERFACE,
    SOFT_SIGNALS_CANNOT_OVERRIDE_HARD_CONSTRAINTS,
    TASK_ID,
    TEST_PASS_IS_NOT_COMPLETION,
    TRANSFORMATION_PACKET_RECEIPT_INTERFACE,
    TRANSFORMATION_PREIMAGE_INTERFACE,
    VECTOR_SIMILARITY_IS_AUTHORITY,
    WORKER_SELF_APPROVAL,
    AdapterKind,
    BoundedEdit,
    DryRunReceipt,
    EditKind,
    EffectScope,
    ExpectedDelta,
    LeaseFenceBinding,
    RefactorTransformationPacket,
    RollbackPlan,
    RewriteKind,
    SyntaxSupport,
    TransformationPacketError,
    TransformationPacketReceipt,
    TransformationPreimage,
    assert_not_competing_capsule_family,
    compile_refactor_transformation_packet,
    compile_transformation_packet_receipt,
    decode_canonical_packet,
    decode_canonical_receipt,
    dry_run_transformation_packet,
    encode_canonical_packet,
    encode_canonical_receipt,
    provider_free_exports,
    transformation_packet_cid_profile,
)


ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    ROOT
    / "ipfs_accelerate_py"
    / "agent_supervisor"
    / "semantic_refactoring"
    / "transformation_packet.py"
)
TEST_PATH = Path(__file__).resolve()
WRITE_SCOPE = (
    "ipfs_accelerate_py/agent_supervisor/semantic_refactoring/transformation_packet.py",
    "test/api/semantic_refactoring/test_transformation_packet.py",
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


def _candidate(**overrides: Any) -> ProgramPartitionCandidate:
    fields: dict[str, Any] = {
        "tree_id": TREE_ID,
        "generator_kind": GeneratorKind.SCC,
        "member_ids": ("node:leaf",),
        "admitted": True,
        "consumer_ids": ("pkg.cli",),
    }
    fields.update(overrides)
    return ProgramPartitionCandidate(**fields)


def _contracts(**overrides: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "tree_id": TREE_ID,
        "source_cid": _cid("source"),
        "partition_cid": "",
        "contract_set_cid": _cid("contracts"),
        "contracts": [
            {
                "edge_id": "edge:import",
                "source_id": "pkg.cli",
                "target_id": "node:leaf",
                "kind": "import",
                "disposition": "admitted",
                "complete": True,
                "required": True,
                "allowed_effects": ["bounded_source_edit", "isolated_validation"],
                "forbidden_effects": ["network"],
            }
        ],
    }
    fields.update(overrides)
    return fields


def _target_api(candidate: ProgramPartitionCandidate, **overrides: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "tree_id": TREE_ID,
        "plan_cid": _cid("target-api"),
        "modules": [
            {
                "module_id": candidate.candidate_cid,
                "member_ids": list(candidate.member_ids),
                "public_exports": [
                    {
                        "member_id": candidate.member_ids[0],
                        "consumer_ids": ["pkg.cli"],
                    }
                ],
            }
        ],
    }
    fields.update(overrides)
    return fields


def _facade(
    candidate: ProgramPartitionCandidate,
    *,
    migration_kind: str = "reexport",
    disposition: str = "preserve",
    required: bool = True,
    facade_required: bool = False,
    **overrides: Any,
) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "tree_id": TREE_ID,
        "plan_cid": _cid("facade"),
        "consumer_plans": [
            {
                "obligation_id": "obl:import",
                "consumer_id": "pkg.cli",
                "subject_id": "symbol:pkg.mod.Record",
                "subject_module": "pkg.mod",
                "kind": "import_path",
                "disposition": disposition,
                "migration_kind": migration_kind,
                "required": required,
                "target_module_id": candidate.candidate_cid,
            }
        ],
        "subject_facades": [
            {
                "subject_id": "symbol:pkg.mod.Record",
                "subject_module": "pkg.mod",
                "facade_required": facade_required,
                "consumer_ids": ["pkg.cli"],
                "undispositioned_consumer_ids": [],
                "migration_kinds": [migration_kind],
                "target_module_id": candidate.candidate_cid,
                "can_retire_facade": False,
            }
        ],
    }
    fields.update(overrides)
    return fields


def _preimage(**overrides: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "repository_id": PROGRAM,
        "environment_cid": _cid("env"),
        "graph_cid": _cid("graph"),
        "source_cids": [_cid("source")],
    }
    fields.update(overrides)
    return fields


def _compile(
    candidate: ProgramPartitionCandidate | None = None,
    **overrides: Any,
) -> RefactorTransformationPacket:
    resolved = candidate or _candidate()
    fields: dict[str, Any] = {
        "boundary_contracts": _contracts(partition_cid=resolved.candidate_cid),
        "target_api_plan": _target_api(resolved),
        "facade_plan": _facade(resolved),
        "candidates": (resolved,),
        "preimage": _preimage(),
        "write_paths": WRITE_PATHS,
        "lease_id": _cid("lease"),
        "fence_id": _cid("fence"),
        "epoch_id": _cid("epoch"),
        "validation_commands": VALIDATION,
        "repository_id": PROGRAM,
    }
    fields.update(overrides)
    return compile_refactor_transformation_packet(**fields)


def test_owned_paths_and_task_identity_are_exact() -> None:
    assert TASK_ID == "SPAR-019"
    assert GOAL_ID == "SPAR-G041"
    assert PROGRAM == "semantic-preserving-autonomous-remodularization-v1"
    assert REFACTOR_TRANSFORMATION_PACKET_INTERFACE == "RefactorTransformationPacket@1"
    assert TRANSFORMATION_PREIMAGE_INTERFACE == "TransformationPreimage@1"
    assert BOUNDED_EDIT_INTERFACE == "BoundedEdit@1"
    assert EXPECTED_DELTA_INTERFACE == "ExpectedDelta@1"
    assert EFFECT_SCOPE_INTERFACE == "EffectScope@1"
    assert LEASE_FENCE_BINDING_INTERFACE == "LeaseFenceBinding@1"
    assert ROLLBACK_PLAN_INTERFACE == "RollbackPlan@1"
    assert TRANSFORMATION_PACKET_RECEIPT_INTERFACE == "TransformationPacketReceipt@1"
    assert DRY_RUN_RECEIPT_INTERFACE == "DryRunReceipt@1"
    assert PACKET_CONTRACT_VERSION == "1"
    assert ANALYZER_ID.endswith("transformation_packet@1")
    assert MODULE_PATH.is_file()
    assert TEST_PATH.is_file()
    for relative in WRITE_SCOPE:
        assert (ROOT / relative).is_file()


def test_authority_flags_cannot_self_authorize() -> None:
    assert AUTHORITY == "partition orchestration"
    assert AUTHORITY_OWNER == "ipfs_accelerate_py"
    assert PACKET_CAN_AUTHORIZE_COMPLETION is False
    assert PACKET_CAN_AUTHORIZE_TRANSITION is False
    assert PACKET_CAN_CREATE_AUTHORITY is False
    assert PACKET_CAN_RETIRE_FACADE is False
    assert VECTOR_SIMILARITY_IS_AUTHORITY is False
    assert PROJECTION_CLUSTERING_IS_AUTHORITY is False
    assert MODEL_OUTPUT_IS_PROPOSAL_ONLY is True
    assert TEST_PASS_IS_NOT_COMPLETION is True
    assert MARKDOWN_IS_NOT_COMPLETION is True
    assert WORKER_SELF_APPROVAL is False
    assert DUCKLAKE_IS_AUTHORITY is False
    assert SOFT_SIGNALS_CANNOT_OVERRIDE_HARD_CONSTRAINTS is True
    assert PACKET_IS_NOMINATION_ONLY is True
    assert RAW_SOURCE_REQUIRED is True
    assert DRY_RUN_IS_DETERMINISTIC is True
    assert DRY_RUN_MUTATES is False
    profile = transformation_packet_cid_profile()
    assert profile["codec"] == "dag-json"
    assert "not universal meaning" in profile["rule"]
    assert DECLARED_EDIT_KINDS == {"move", "rewrite", "adapter", "facade"}
    assert "bounded_source_edit" in DECLARED_ALLOWED_EFFECTS
    assert "network" in DECLARED_FORBIDDEN_EFFECTS
    assert ROLLBACK_MODE == "restore_preimages_or_discard_worktree"


def test_module_defines_predicted_symbols_not_capsule_family() -> None:
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    names = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
    assert "RefactorTransformationPacket" in names
    assert "TransformationPreimage" in names
    assert "BoundedEdit" in names
    assert "ExpectedDelta" in names
    assert "EffectScope" in names
    assert "LeaseFenceBinding" in names
    assert "RollbackPlan" in names
    assert "TransformationPacketReceipt" in names
    assert "DryRunReceipt" in names
    for capsule in CAPSULE_TYPES:
        assert capsule not in names
    assert_not_competing_capsule_family()
    exports = provider_free_exports()
    assert "RefactorTransformationPacket" in exports
    assert "compile_refactor_transformation_packet" in exports
    assert "compile_transformation_packet_receipt" in exports
    assert "dry_run_transformation_packet" in exports


def test_protected_paths_are_not_owned_write_scope() -> None:
    owned = set(WRITE_SCOPE)
    for relative in PROTECTED_PATHS:
        assert relative not in owned
        assert (ROOT / relative).exists()


def test_compile_binds_preimages_moves_rewrites_scope_lease_validation_rollback() -> None:
    candidate = _candidate()
    packet = _compile(candidate)
    assert packet.tree_id == TREE_ID
    assert packet.preimage.source_cids == (_cid("source"),)
    assert packet.preimage.environment_cid == _cid("env")
    assert packet.preimage.graph_cid == _cid("graph")
    assert packet.preimage.partition_cid == candidate.candidate_cid
    assert packet.target_api_plan_cid == _cid("target-api")
    assert packet.facade_plan_cid == _cid("facade")
    kinds = {item.kind for item in packet.edits}
    assert EditKind.MOVE.value in kinds
    assert EditKind.REWRITE.value in kinds
    move = next(item for item in packet.edits if item.kind == EditKind.MOVE.value)
    assert move.source_id == "pkg.mod"
    assert move.destination_id == candidate.candidate_cid
    assert move.member_ids == ("node:leaf",)
    rewrite = next(item for item in packet.edits if item.kind == EditKind.REWRITE.value)
    assert rewrite.rewrite_kind == RewriteKind.REEXPORT.value
    assert packet.effect_scope.write_paths == WRITE_PATHS
    assert "bounded_source_edit" in packet.effect_scope.allowed_effects
    assert "network" in packet.effect_scope.forbidden_effects
    assert packet.lease_fence.lease_id == _cid("lease")
    assert packet.lease_fence.fence_id == _cid("fence")
    assert packet.validation_commands == VALIDATION
    assert packet.rollback.mode == ROLLBACK_MODE
    assert packet.rollback.restore_source_cids == packet.preimage.source_cids
    assert packet.rollback.advance_accepted_roots is False
    assert packet.packet_is_nomination_only is True
    assert packet.can_authorize_completion is False
    assert packet.can_retire_facade is False
    assert packet.unrestricted_scope is False
    assert packet.expected_delta.moved_member_ids == ("node:leaf",)
    assert packet.expected_delta.preserved_identity_ids == ("node:leaf",)


def test_wrapper_migration_nominates_adapter() -> None:
    candidate = _candidate()
    packet = _compile(candidate, facade_plan=_facade(candidate, migration_kind="wrapper"))
    adapters = [item for item in packet.edits if item.kind == EditKind.ADAPTER.value]
    assert adapters
    assert adapters[0].adapter_kind == AdapterKind.WRAPPER.value


def test_facade_disposition_keeps_original_module_and_cannot_retire() -> None:
    candidate = _candidate()
    packet = _compile(
        candidate,
        facade_plan=_facade(
            candidate,
            migration_kind="facade",
            disposition="facade",
            facade_required=True,
        ),
    )
    facades = [item for item in packet.edits if item.kind == EditKind.FACADE.value]
    assert facades
    assert "symbol:pkg.mod.Record" in packet.expected_delta.facade_subject_ids
    assert packet.can_retire_facade is False


def test_empty_write_paths_are_unrestricted_scope() -> None:
    with pytest.raises(TransformationPacketError, match="unrestricted scope"):
        _compile(write_paths=())


def test_wildcard_and_absolute_paths_fail_closed() -> None:
    with pytest.raises(TransformationPacketError, match="unrestricted scope"):
        _compile(write_paths=("pkg/*.py",))
    with pytest.raises(TransformationPacketError, match="unrestricted scope"):
        _compile(write_paths=("/tmp/pkg/mod.py",))
    with pytest.raises(TransformationPacketError, match="unrestricted scope"):
        _compile(write_paths=("pkg/../secret.py",))


def test_unsupported_syntax_is_typed_terminal() -> None:
    candidate = _candidate()
    with pytest.raises(TransformationPacketError, match="typed terminal"):
        _compile(
            candidate,
            facade_plan=_facade(candidate, migration_kind="codegen"),
        )
    with pytest.raises(TransformationPacketError, match="typed terminal"):
        BoundedEdit(
            kind=EditKind.MOVE,
            source_id="pkg.mod",
            destination_id="pkg.extracted",
            member_ids=("node:leaf",),
            syntax_support=SyntaxSupport.UNSUPPORTED,
            write_paths=WRITE_PATHS,
        )


def test_incomplete_spar016_contract_fails_closed() -> None:
    with pytest.raises(TransformationPacketError, match="typed terminal"):
        _compile(
            boundary_contracts=_contracts(
                contracts=[
                    {
                        "edge_id": "edge:import",
                        "disposition": "retrieval",
                        "complete": False,
                        "required": True,
                    }
                ]
            )
        )


def test_required_undispositioned_spar018_fails_closed() -> None:
    candidate = _candidate()
    with pytest.raises(TransformationPacketError, match="typed terminal"):
        _compile(
            candidate,
            facade_plan=_facade(
                candidate,
                disposition="undispositioned",
                migration_kind="facade",
            ),
        )


def test_required_unsupported_spar018_fails_closed() -> None:
    candidate = _candidate()
    with pytest.raises(TransformationPacketError, match="typed terminal"):
        _compile(
            candidate,
            facade_plan=_facade(
                candidate,
                disposition="unsupported",
                migration_kind="deprecation",
            ),
        )


def test_rejected_candidates_remain_negative_evidence() -> None:
    admitted = _candidate()
    rejected = _candidate(
        member_ids=("node:ghost",),
        generator_kind=GeneratorKind.GRAPH,
        admitted=False,
        consumer_ids=(),
        hard_constraint_violations=(ConstraintClass.OVERSIZED_CYCLE.value,),
    )
    comparison = compare_partition_candidates((admitted, rejected))
    packet = _compile(
        admitted,
        candidates=(admitted, rejected),
        comparison=comparison,
        boundary_contracts=_contracts(partition_cid=admitted.candidate_cid),
        target_api_plan=_target_api(admitted),
        facade_plan=_facade(admitted),
    )
    assert packet.selected_candidate_cids == (admitted.candidate_cid,)
    assert rejected.candidate_cid in packet.rejected_candidate_cids
    assert packet.negative_evidence_cids == packet.rejected_candidate_cids
    with pytest.raises(TransformationPacketError, match="not ranked by SPAR-014"):
        _compile(
            admitted,
            candidates=(admitted, rejected),
            comparison=comparison,
            selected_candidate_cids=(rejected.candidate_cid,),
            boundary_contracts=_contracts(partition_cid=admitted.candidate_cid),
            target_api_plan=_target_api(admitted),
            facade_plan=_facade(admitted),
        )


def test_advisory_and_vector_evidence_cannot_admit_a_packet() -> None:
    projection = _candidate(
        member_ids=("node:vector",),
        generator_kind=GeneratorKind.PROJECTION,
        admitted=False,
        advisory=True,
        consumer_ids=(),
        evidence_class="vector_candidate",
    )
    admitted = _candidate()
    packet = _compile(
        admitted,
        candidates=(admitted, projection),
        boundary_contracts=_contracts(partition_cid=admitted.candidate_cid),
        target_api_plan=_target_api(admitted),
        facade_plan=_facade(admitted),
    )
    assert projection.candidate_cid in packet.advisory_candidate_cids
    assert projection.candidate_cid not in packet.selected_candidate_cids
    with pytest.raises(TransformationPacketError, match="not ranked by SPAR-014"):
        _compile(
            admitted,
            candidates=(admitted, projection),
            selected_candidate_cids=(projection.candidate_cid,),
            boundary_contracts=_contracts(partition_cid=admitted.candidate_cid),
            target_api_plan=_target_api(admitted),
            facade_plan=_facade(admitted),
        )


def test_overlapping_ranked_candidates_require_explicit_selection() -> None:
    scc = _candidate(
        member_ids=("node:leaf", "node:helper"),
        generator_kind=GeneratorKind.SCC,
    )
    contract = _candidate(
        member_ids=("node:leaf",),
        generator_kind=GeneratorKind.CONTRACT,
    )
    comparison = compare_partition_candidates((scc, contract))
    with pytest.raises(TransformationPacketError, match="selected_candidate_cids"):
        _compile(
            contract,
            candidates=(scc, contract),
            comparison=comparison,
            boundary_contracts=_contracts(partition_cid=contract.candidate_cid),
            target_api_plan=_target_api(contract),
            facade_plan=_facade(contract),
        )
    packet = _compile(
        contract,
        candidates=(scc, contract),
        comparison=comparison,
        selected_candidate_cids=(contract.candidate_cid,),
        boundary_contracts=_contracts(partition_cid=contract.candidate_cid),
        target_api_plan=_target_api(contract),
        facade_plan=_facade(contract),
    )
    assert packet.selected_candidate_cids == (contract.candidate_cid,)


def test_dry_run_is_deterministic_and_does_not_mutate() -> None:
    packet = _compile()
    first = dry_run_transformation_packet(packet)
    second = dry_run_transformation_packet(packet)
    assert first.dry_run_cid == second.dry_run_cid
    assert first.mutated is False
    assert first.deterministic is True
    assert first.can_authorize_transition is False
    assert first.packet_cid == packet.packet_cid
    assert first.write_paths == WRITE_PATHS
    restored = DryRunReceipt.from_dict(first.to_dict())
    assert restored == first


def test_packet_round_trip_and_receipt_are_deterministic() -> None:
    first = _compile()
    second = _compile()
    assert first.packet_cid == second.packet_cid
    restored = decode_canonical_packet(encode_canonical_packet(first))
    assert restored == first
    assert restored.packet_cid == first.packet_cid
    receipt = compile_transformation_packet_receipt(first)
    assert receipt.packet_cid == first.packet_cid
    assert receipt.can_authorize_completion is False
    assert receipt.can_retire_facade is False
    assert decode_canonical_receipt(encode_canonical_receipt(receipt)) == receipt


def test_identity_excludes_observational_fields() -> None:
    packet = _compile()
    payload = packet.to_dict()
    assert not (IDENTITY_EXCLUDED_FIELDS & set(payload))
    dirty = dict(payload)
    dirty["timestamp"] = "now"
    with pytest.raises(TransformationPacketError, match="observational"):
        RefactorTransformationPacket.from_dict(dirty)


def test_packet_cannot_claim_authority_flags() -> None:
    packet = _compile()
    payload = packet.to_dict()
    payload["can_authorize_completion"] = True
    payload["packet_cid"] = cid_for_dag_json(
        {key: value for key, value in payload.items() if key != "packet_cid"}
    )
    with pytest.raises(TransformationPacketError, match="can_authorize_completion"):
        RefactorTransformationPacket.from_dict(payload)
    payload = packet.to_dict()
    payload["packet_is_nomination_only"] = False
    payload["packet_cid"] = cid_for_dag_json(
        {key: value for key, value in payload.items() if key != "packet_cid"}
    )
    with pytest.raises(TransformationPacketError, match="nomination_only"):
        RefactorTransformationPacket.from_dict(payload)
    payload = packet.to_dict()
    payload["can_retire_facade"] = True
    payload["packet_cid"] = cid_for_dag_json(
        {key: value for key, value in payload.items() if key != "packet_cid"}
    )
    with pytest.raises(TransformationPacketError, match="can_retire_facade"):
        RefactorTransformationPacket.from_dict(payload)


def test_tree_mismatch_fails_closed() -> None:
    candidate = _candidate()
    with pytest.raises(TransformationPacketError, match="tree_id"):
        _compile(
            candidate,
            boundary_contracts=_contracts(tree_id=OTHER_TREE),
        )
    with pytest.raises(TransformationPacketError, match="tree_id"):
        _compile(
            candidate,
            target_api_plan=_target_api(candidate, tree_id=OTHER_TREE),
        )


def test_comparison_must_remain_spar014_analyzer() -> None:
    candidate = _candidate()
    comparison = compare_partition_candidates((candidate,))
    payload = comparison.to_dict()
    payload["analyzer_id"] = ANALYZER_ID
    identity = {key: value for key, value in payload.items() if key != "receipt_cid"}
    payload["receipt_cid"] = cid_for_dag_json(identity)
    with pytest.raises(Exception, match="SPAR-014 analyzer"):
        _compile(candidate, comparison=payload)


def test_missing_predecessors_fail_closed() -> None:
    with pytest.raises(TransformationPacketError, match="SPAR-016"):
        _compile(boundary_contracts=_contracts(contracts=[]))
    candidate = _candidate()
    with pytest.raises(TransformationPacketError, match="SPAR-017"):
        _compile(candidate, target_api_plan={"tree_id": TREE_ID, "plan_cid": _cid("x"), "modules": []})
    with pytest.raises(TransformationPacketError, match="SPAR-018"):
        _compile(
            candidate,
            facade_plan={
                "tree_id": TREE_ID,
                "plan_cid": _cid("y"),
                "consumer_plans": [],
                "subject_facades": [],
            },
        )


def test_rollback_cannot_advance_roots_or_drop_preimages() -> None:
    with pytest.raises(TransformationPacketError, match="accepted roots"):
        RollbackPlan(
            restore_source_cids=(_cid("source"),),
            advance_accepted_roots=True,
        )
    with pytest.raises(TransformationPacketError, match="restore_source_cids"):
        RollbackPlan(restore_source_cids=())


def test_network_cannot_be_an_allowed_effect() -> None:
    with pytest.raises(TransformationPacketError, match="allowed_effects"):
        EffectScope(
            write_paths=WRITE_PATHS,
            allowed_effects=("network",),
        )


def test_preserved_identities_must_be_moved_members() -> None:
    with pytest.raises(TransformationPacketError, match="subset"):
        ExpectedDelta(
            moved_member_ids=("node:leaf",),
            preserved_identity_ids=("node:other",),
        )


def test_empty_validation_commands_fail_closed() -> None:
    with pytest.raises(TransformationPacketError, match="validation_commands"):
        _compile(validation_commands=())


def test_lease_fence_requires_cids() -> None:
    with pytest.raises(TransformationPacketError, match="lease_id"):
        LeaseFenceBinding(lease_id="not-a-cid", fence_id=_cid("fence"))


def test_edit_paths_must_stay_inside_scope() -> None:
    packet = _compile()
    payload = packet.to_dict()
    payload["edits"][0]["write_paths"] = ["pkg/outside.py"]
    payload["edits"][0]["edit_cid"] = cid_for_dag_json(
        {key: value for key, value in payload["edits"][0].items() if key != "edit_cid"}
    )
    payload["packet_cid"] = cid_for_dag_json(
        {key: value for key, value in payload.items() if key != "packet_cid"}
    )
    with pytest.raises(TransformationPacketError, match="effect_scope"):
        RefactorTransformationPacket.from_dict(payload)
