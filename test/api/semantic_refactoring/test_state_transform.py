"""Independent contract tests for SPAR-022 ExplicitStateObjectPlan."""

from __future__ import annotations

import ast
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.utils.cid_utils import cid_for_bytes, cid_for_dag_json
from ipfs_accelerate_py.agent_supervisor.semantic_refactoring.partition_generators import (
    GeneratorKind,
    ProgramPartitionCandidate,
)
from ipfs_accelerate_py.agent_supervisor.semantic_refactoring.transformation_packet import (
    ANALYZER_ID as SPAR019_ANALYZER_ID,
    AdapterKind,
    BoundedEdit,
    EditKind,
    RefactorTransformationPacket,
    compile_refactor_transformation_packet,
)
from ipfs_accelerate_py.agent_supervisor.semantic_refactoring.state_transform import (
    ANALYZER_ID,
    AUTHORITY,
    AUTHORITY_OWNER,
    DECLARED_EXPLICIT_STATE_OBJECT_KINDS,
    DECLARED_LIFECYCLE_KINDS,
    DECLARED_SYNCHRONIZATION_KINDS,
    DRY_RUN_IS_DETERMINISTIC,
    DRY_RUN_MUTATES,
    DUCKLAKE_IS_AUTHORITY,
    DUPLICATED_MUTABLE_STATE_REJECTED,
    EXECUTOR_IS_NOMINATION_ONLY,
    EXPLICIT_STATE_OBJECT_INTERFACE,
    EXPLICIT_STATE_OBJECT_PLAN_INTERFACE,
    EXPLICIT_STATE_OBJECT_RECEIPT_INTERFACE,
    GOAL_ID,
    HANDLED_ADAPTER_KINDS,
    IDENTITY_EXCLUDED_FIELDS,
    INCOMPLETE_OBLIGATIONS_ARE_TYPED_TERMINAL,
    MARKDOWN_IS_NOT_COMPLETION,
    MISSING_RELEASE_IS_NOT_GUESSED,
    MODEL_OUTPUT_IS_PROPOSAL_ONLY,
    PLAN_IS_NOMINATION_ONLY,
    PROGRAM,
    PROJECTION_CLUSTERING_IS_AUTHORITY,
    RAW_SOURCE_REQUIRED,
    RUNTIME_OBSERVATION_IS_NOT_STATIC_FACT,
    SOFT_SIGNALS_CANNOT_OVERRIDE_HARD_CONSTRAINTS,
    TASK_ID,
    TEST_PASS_IS_NOT_COMPLETION,
    TRANSFORM_CAN_AUTHORIZE_COMPLETION,
    TRANSFORM_CAN_AUTHORIZE_TRANSITION,
    TRANSFORM_CAN_CREATE_AUTHORITY,
    TRANSFORM_CONTRACT_VERSION,
    UNKNOWN_UNIQUENESS_WIDENS_FRONTIER,
    VECTOR_SIMILARITY_IS_AUTHORITY,
    WORKER_SELF_APPROVAL,
    ExplicitStateObject,
    ExplicitStateObjectKind,
    ExplicitStateObjectPlan,
    ExplicitStateObjectReceipt,
    StateTransformError,
    assert_not_competing_capsule_family,
    compile_explicit_state_object_plan,
    compile_explicit_state_object_receipt,
    compile_explicit_state_objects,
    decode_canonical_plan,
    decode_canonical_receipt,
    decode_canonical_transform,
    dry_run_explicit_state_objects,
    encode_canonical_plan,
    encode_canonical_receipt,
    encode_canonical_transform,
    execute_explicit_state_object_plan,
    execute_explicit_state_objects,
    provider_free_exports,
    state_transform_cid_profile,
    verify_complete_obligations,
    verify_preimages,
)


ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    ROOT
    / "ipfs_accelerate_py"
    / "agent_supervisor"
    / "semantic_refactoring"
    / "state_transform.py"
)
TEST_PATH = Path(__file__).resolve()
WRITE_SCOPE = (
    "ipfs_accelerate_py/agent_supervisor/semantic_refactoring/state_transform.py",
    "test/api/semantic_refactoring/test_state_transform.py",
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
WRITE_PATHS = ("pkg/mod.py", "pkg/extracted.py")
VALIDATION = ("python3 -m pytest -q tests/test_mod.py",)
OWNER_ID = "owner:module_global:CACHE"
ALIAS_ID = "alias:CACHE"
SYMBOL_ID = OWNER_ID


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
                "edge_id": "edge:state",
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
    migration_kind: str = "wrapper",
    disposition: str = "migrate",
    required: bool = True,
    facade_required: bool = False,
    kind: str = "state",
    obligation_id: str = "obl:state",
    subject_id: str = SYMBOL_ID,
    **overrides: Any,
) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "tree_id": TREE_ID,
        "plan_cid": _cid("facade"),
        "consumer_plans": [
            {
                "obligation_id": obligation_id,
                "consumer_id": "pkg.cli",
                "subject_id": subject_id,
                "subject_module": "pkg.mod",
                "kind": kind,
                "disposition": disposition,
                "migration_kind": migration_kind,
                "required": required,
                "target_module_id": candidate.candidate_cid,
            }
        ],
        "subject_facades": [
            {
                "subject_id": subject_id,
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


def _with_adapter(
    packet: RefactorTransformationPacket,
    *,
    adapter_kind: str,
    obligation_id: str,
    member_ids: Sequence[str] = (OWNER_ID,),
) -> RefactorTransformationPacket:
    extra = BoundedEdit(
        kind=EditKind.ADAPTER,
        source_id="pkg.mod",
        destination_id=packet.expected_delta.destination_module_ids[0],
        member_ids=member_ids,
        adapter_kind=adapter_kind,
        write_paths=WRITE_PATHS,
        obligation_id=obligation_id,
    )
    return RefactorTransformationPacket(
        tree_id=packet.tree_id,
        preimage=packet.preimage,
        edits=list(packet.edits) + [extra],
        expected_delta=packet.expected_delta,
        effect_scope=packet.effect_scope,
        lease_fence=packet.lease_fence,
        rollback=packet.rollback,
        validation_commands=packet.validation_commands,
        selected_candidate_cids=packet.selected_candidate_cids,
        rejected_candidate_cids=packet.rejected_candidate_cids,
        advisory_candidate_cids=packet.advisory_candidate_cids,
        evidence_cids=packet.evidence_cids,
        boundary_contract_set_cid=packet.boundary_contract_set_cid,
        target_api_plan_cid=packet.target_api_plan_cid,
        facade_plan_cid=packet.facade_plan_cid,
        comparison_receipt_cid=packet.comparison_receipt_cid,
        analyzer_id=packet.analyzer_id,
    )


def _ownership(
    *,
    uniqueness: str = "unique",
    mutable: bool = True,
    admitted: bool = True,
    complete: bool = True,
    missing_release: bool = False,
    include_release: bool = True,
    include_initialize: bool = True,
    include_lock: bool = True,
    include_candidate: bool = True,
    unresolved: bool = False,
    evidence_class: str = "exact_static_fact",
    extra_owner: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    lifecycle: list[dict[str, Any]] = []
    if include_initialize:
        lifecycle.append(
            {
                "relation_id": "life:init:CACHE",
                "kind": "initialize",
                "owner_id": OWNER_ID,
                "missing_counterpart": False,
            }
        )
    if include_release:
        lifecycle.append(
            {
                "relation_id": "life:release:CACHE",
                "kind": "release",
                "owner_id": OWNER_ID,
                "missing_counterpart": missing_release,
            }
        )
    elif missing_release:
        lifecycle.append(
            {
                "relation_id": "life:acquire:CACHE",
                "kind": "acquire",
                "owner_id": OWNER_ID,
                "missing_counterpart": True,
            }
        )
    synchronization: list[dict[str, Any]] = []
    if include_lock:
        synchronization.append(
            {
                "relation_id": "sync:lock:CACHE",
                "kind": "lock",
                "owner_id": OWNER_ID,
                "guarded_subject_ids": ["CACHE"],
            }
        )
    owners = [
        {
            "owner_id": OWNER_ID,
            "uniqueness": uniqueness,
            "alias_set_id": ALIAS_ID,
            "owning_symbol_id": "CACHE",
            "mutable": mutable,
            "evidence_class": evidence_class,
        }
    ]
    if extra_owner is not None:
        owners.append(dict(extra_owner))
    candidates: list[dict[str, Any]] = []
    if include_candidate:
        candidates.append(
            {
                "owner_id": OWNER_ID,
                "uniqueness": uniqueness,
                "alias_set_id": ALIAS_ID,
                "lifecycle_ids": [item["relation_id"] for item in lifecycle],
                "synchronization_ids": [
                    item["relation_id"] for item in synchronization
                ],
                "complete_obligations": complete,
                "admitted": admitted,
                "evidence_class": evidence_class,
            }
        )
    unresolved_items: list[dict[str, Any]] = []
    if unresolved:
        unresolved_items.append(
            {
                "subject_id": OWNER_ID,
                "reason": "unknown_uniqueness",
            }
        )
    return {
        "owners": owners,
        "alias_sets": [
            {
                "alias_set_id": ALIAS_ID,
                "representative_id": "CACHE",
                "member_ids": ["CACHE"],
            }
        ],
        "lifecycle_relations": lifecycle,
        "synchronization_relations": synchronization,
        "extraction_candidates": candidates,
        "unresolved": unresolved_items,
    }


def test_owned_paths_and_task_identity_are_exact() -> None:
    assert TASK_ID == "SPAR-022"
    assert GOAL_ID == "SPAR-G042"
    assert PROGRAM == "semantic-preserving-autonomous-remodularization-v1"
    assert EXPLICIT_STATE_OBJECT_INTERFACE == "ExplicitStateObject@1"
    assert EXPLICIT_STATE_OBJECT_PLAN_INTERFACE == "ExplicitStateObjectPlan@1"
    assert EXPLICIT_STATE_OBJECT_RECEIPT_INTERFACE == "ExplicitStateObjectReceipt@1"
    assert TRANSFORM_CONTRACT_VERSION == "1"
    assert ANALYZER_ID.endswith("state_transform@1")
    assert ANALYZER_ID != SPAR019_ANALYZER_ID
    assert MODULE_PATH.is_file()
    assert TEST_PATH.is_file()
    for relative in WRITE_SCOPE:
        assert (ROOT / relative).is_file()


def test_authority_flags_cannot_self_authorize() -> None:
    assert AUTHORITY == "partition orchestration"
    assert AUTHORITY_OWNER == "ipfs_accelerate_py"
    assert TRANSFORM_CAN_AUTHORIZE_COMPLETION is False
    assert TRANSFORM_CAN_AUTHORIZE_TRANSITION is False
    assert TRANSFORM_CAN_CREATE_AUTHORITY is False
    assert VECTOR_SIMILARITY_IS_AUTHORITY is False
    assert PROJECTION_CLUSTERING_IS_AUTHORITY is False
    assert MODEL_OUTPUT_IS_PROPOSAL_ONLY is True
    assert TEST_PASS_IS_NOT_COMPLETION is True
    assert MARKDOWN_IS_NOT_COMPLETION is True
    assert WORKER_SELF_APPROVAL is False
    assert DUCKLAKE_IS_AUTHORITY is False
    assert SOFT_SIGNALS_CANNOT_OVERRIDE_HARD_CONSTRAINTS is True
    assert EXECUTOR_IS_NOMINATION_ONLY is True
    assert PLAN_IS_NOMINATION_ONLY is True
    assert RAW_SOURCE_REQUIRED is True
    assert DRY_RUN_IS_DETERMINISTIC is True
    assert DRY_RUN_MUTATES is False
    assert RUNTIME_OBSERVATION_IS_NOT_STATIC_FACT is True
    assert DUPLICATED_MUTABLE_STATE_REJECTED is True
    assert MISSING_RELEASE_IS_NOT_GUESSED is True
    assert UNKNOWN_UNIQUENESS_WIDENS_FRONTIER is True
    assert INCOMPLETE_OBLIGATIONS_ARE_TYPED_TERMINAL is True
    profile = state_transform_cid_profile()
    assert profile["codec"] == "dag-json"
    assert "not universal meaning" in profile["rule"]
    assert DECLARED_EXPLICIT_STATE_OBJECT_KINDS == {
        "state_object",
        "protocol",
        "boundary_adapter",
        "injection",
    }
    assert HANDLED_ADAPTER_KINDS == {
        AdapterKind.STATE.value,
        AdapterKind.BOUNDARY.value,
        AdapterKind.PROTOCOL.value,
        AdapterKind.WRAPPER.value,
    }
    assert DECLARED_LIFECYCLE_KINDS == {
        "initialize",
        "acquire",
        "release",
        "finalize",
    }
    assert DECLARED_SYNCHRONIZATION_KINDS == {"lock", "transaction", "condition"}


def test_module_defines_predicted_symbols_not_capsule_family() -> None:
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    names = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
    assert "ExplicitStateObjectPlan" in names
    assert "ExplicitStateObject" in names
    assert "ExplicitStateObjectReceipt" in names
    for capsule in CAPSULE_TYPES:
        assert capsule not in names
    assert_not_competing_capsule_family()
    exports = provider_free_exports()
    assert "ExplicitStateObjectPlan" in exports
    assert "compile_explicit_state_object_plan" in exports
    assert "execute_explicit_state_object_plan" in exports
    assert "dry_run_explicit_state_objects" in exports


def test_protected_paths_are_not_owned_write_scope() -> None:
    owned = set(WRITE_SCOPE)
    for relative in PROTECTED_PATHS:
        assert relative not in owned
        assert (ROOT / relative).exists()


def test_wrapper_injection_is_nominated_from_packet() -> None:
    packet = _compile()
    graph = _ownership()
    transforms = compile_explicit_state_objects(packet, ownership_graph=graph)
    kinds = {item.transform_kind for item in transforms}
    assert ExplicitStateObjectKind.INJECTION.value in kinds
    injection = next(
        item
        for item in transforms
        if item.transform_kind == ExplicitStateObjectKind.INJECTION.value
    )
    assert injection.owner_id == OWNER_ID
    assert injection.source_module == "pkg.mod"
    assert injection.destination_module == packet.expected_delta.destination_module_ids[0]
    assert injection.adapter_kind == AdapterKind.WRAPPER.value
    assert injection.transform_is_nomination_only is True
    assert injection.complete_obligations is True
    assert injection.write_paths == WRITE_PATHS


def test_state_protocol_and_boundary_adapters_are_nominated() -> None:
    packet = _compile()
    graph = _ownership()
    state = _with_adapter(
        packet, adapter_kind=AdapterKind.STATE.value, obligation_id="obl:state-object"
    )
    state_kinds = {
        item.transform_kind
        for item in compile_explicit_state_objects(state, ownership_graph=graph)
    }
    assert ExplicitStateObjectKind.STATE_OBJECT.value in state_kinds
    protocol = _with_adapter(
        packet, adapter_kind=AdapterKind.PROTOCOL.value, obligation_id="obl:protocol"
    )
    protocol_kinds = {
        item.transform_kind
        for item in compile_explicit_state_objects(protocol, ownership_graph=graph)
    }
    assert ExplicitStateObjectKind.PROTOCOL.value in protocol_kinds
    boundary = _with_adapter(
        packet, adapter_kind=AdapterKind.BOUNDARY.value, obligation_id="obl:boundary"
    )
    boundary_kinds = {
        item.transform_kind
        for item in compile_explicit_state_objects(boundary, ownership_graph=graph)
    }
    assert ExplicitStateObjectKind.BOUNDARY_ADAPTER.value in boundary_kinds


def test_admitted_extraction_candidate_nominates_state_object() -> None:
    candidate = _candidate()
    packet = _compile(
        candidate,
        facade_plan=_facade(
            candidate,
            migration_kind="reexport",
            kind="import_path",
            disposition="preserve",
            obligation_id="obl:import",
            subject_id="symbol:pkg.mod.Record",
        ),
    )
    graph = _ownership()
    transforms = compile_explicit_state_objects(packet, ownership_graph=graph)
    kinds = {item.transform_kind for item in transforms}
    assert ExplicitStateObjectKind.STATE_OBJECT.value in kinds
    state = next(
        item
        for item in transforms
        if item.transform_kind == ExplicitStateObjectKind.STATE_OBJECT.value
    )
    assert state.owner_id == OWNER_ID
    assert state.adapter_kind == AdapterKind.STATE.value
    assert state.lifecycle_ids == ("life:init:CACHE", "life:release:CACHE")
    assert state.synchronization_ids == ("sync:lock:CACHE",)
    plan = compile_explicit_state_object_plan(packet, ownership_graph=graph)
    assert plan.complete_obligations is True
    assert plan.unique_owners is True
    assert plan.no_duplicated_mutable_state is True
    assert plan.missing_release_guessed is False
    assert plan.plan_is_nomination_only is True
    assert OWNER_ID in plan.owner_ids


def test_preimages_are_verified_against_packet() -> None:
    packet = _compile()
    graph = _ownership()
    assert verify_preimages(packet) == packet.preimage.preimage_cid
    assert (
        verify_preimages(packet, claimed_preimage_cid=packet.preimage.preimage_cid)
        == packet.preimage.preimage_cid
    )
    with pytest.raises(StateTransformError, match="preimage does not verify"):
        verify_preimages(packet, claimed_preimage_cid=_cid("other-preimage"))
    with pytest.raises(StateTransformError, match="preimage does not verify"):
        compile_explicit_state_objects(
            packet,
            ownership_graph=graph,
            claimed_preimage_cid=_cid("stale"),
        )


def test_complete_obligations_are_verified() -> None:
    graph = _ownership()
    obligations = verify_complete_obligations(graph, OWNER_ID)
    assert obligations["complete_obligations"] is True
    assert obligations["admitted"] is True
    assert obligations["uniqueness"] == "unique"
    assert "life:release:CACHE" in obligations["lifecycle_ids"]


def test_incomplete_obligations_are_typed_terminals() -> None:
    packet = _compile()
    with pytest.raises(StateTransformError, match="incomplete"):
        compile_explicit_state_objects(packet, ownership_graph=_ownership(complete=False))
    with pytest.raises(StateTransformError, match="incomplete"):
        compile_explicit_state_objects(packet, ownership_graph=_ownership(admitted=False))
    with pytest.raises(StateTransformError, match="incomplete"):
        compile_explicit_state_objects(
            packet,
            ownership_graph=_ownership(include_initialize=False, include_candidate=False),
        )


def test_missing_release_is_not_guessed() -> None:
    packet = _compile()
    with pytest.raises(StateTransformError, match="missing release"):
        compile_explicit_state_objects(
            packet,
            ownership_graph=_ownership(include_release=False, include_candidate=False),
        )
    with pytest.raises(StateTransformError, match="missing release"):
        compile_explicit_state_objects(
            packet,
            ownership_graph=_ownership(missing_release=True, include_candidate=False),
        )


def test_duplicated_unique_owners_are_rejected() -> None:
    packet = _compile()
    extra = {
        "owner_id": "owner:module_global:CACHE2",
        "uniqueness": "unique",
        "alias_set_id": ALIAS_ID,
        "owning_symbol_id": "CACHE",
        "mutable": True,
        "evidence_class": "exact_static_fact",
    }
    with pytest.raises(StateTransformError, match="duplicated mutable state"):
        compile_explicit_state_objects(
            packet, ownership_graph=_ownership(extra_owner=extra)
        )


def test_unknown_uniqueness_cannot_admit_a_transform() -> None:
    packet = _compile()
    with pytest.raises(StateTransformError, match="unknown uniqueness"):
        compile_explicit_state_objects(
            packet, ownership_graph=_ownership(uniqueness="unknown")
        )


def test_shared_ownership_cannot_admit_a_transform() -> None:
    packet = _compile()
    with pytest.raises(StateTransformError, match="unique ownership"):
        compile_explicit_state_objects(
            packet, ownership_graph=_ownership(uniqueness="shared")
        )


def test_vector_or_model_evidence_cannot_admit_a_transform() -> None:
    packet = _compile()
    with pytest.raises(StateTransformError, match="vector, model, or heuristic"):
        compile_explicit_state_objects(
            packet,
            ownership_graph=_ownership(evidence_class="vector_candidate"),
        )
    with pytest.raises(StateTransformError, match="vector, model, or heuristic"):
        compile_explicit_state_objects(
            packet,
            ownership_graph=_ownership(evidence_class="model_hypothesis"),
        )


def test_unresolved_owner_cannot_admit_a_transform() -> None:
    packet = _compile()
    with pytest.raises(StateTransformError, match="unresolved"):
        compile_explicit_state_objects(
            packet, ownership_graph=_ownership(unresolved=True)
        )


def test_dry_run_is_deterministic_and_does_not_mutate() -> None:
    packet = _compile()
    graph = _ownership()
    first = dry_run_explicit_state_objects(packet, ownership_graph=graph)
    second = execute_explicit_state_objects(packet, ownership_graph=graph)
    assert first.receipt_cid == second.receipt_cid
    assert first.mutated is False
    assert first.deterministic is True
    assert first.can_authorize_transition is False
    assert first.can_authorize_completion is False
    assert first.executor_is_nomination_only is True
    assert first.packet_cid == packet.packet_cid
    assert first.preimage_cid == packet.preimage.preimage_cid
    assert first.preimage_verified is True
    assert first.complete_obligations is True
    assert first.unique_owners is True
    assert first.missing_release_guessed is False
    assert first.write_paths == WRITE_PATHS
    restored = ExplicitStateObjectReceipt.from_dict(first.to_dict())
    assert restored == first
    with pytest.raises(StateTransformError, match="cannot mutate"):
        execute_explicit_state_objects(packet, ownership_graph=graph, mutate=True)


def test_plan_round_trip_is_deterministic() -> None:
    packet = _compile()
    graph = _ownership()
    first = compile_explicit_state_objects(packet, ownership_graph=graph)
    second = compile_explicit_state_objects(packet, ownership_graph=graph)
    assert [item.transform_cid for item in first] == [
        item.transform_cid for item in second
    ]
    restored = decode_canonical_transform(encode_canonical_transform(first[0]))
    assert restored == first[0]
    plan = compile_explicit_state_object_plan(packet, ownership_graph=graph)
    assert decode_canonical_plan(encode_canonical_plan(plan)) == plan
    assert execute_explicit_state_object_plan(packet, ownership_graph=graph) == plan
    receipt = compile_explicit_state_object_receipt(packet, ownership_graph=graph)
    assert decode_canonical_receipt(encode_canonical_receipt(receipt)) == receipt
    assert receipt.plan_cid == plan.plan_cid


def test_identity_excludes_observational_fields() -> None:
    packet = _compile()
    graph = _ownership()
    plan = compile_explicit_state_object_plan(packet, ownership_graph=graph)
    payload = plan.to_dict()
    assert not (IDENTITY_EXCLUDED_FIELDS & set(payload))
    dirty = dict(payload)
    dirty["timestamp"] = "now"
    with pytest.raises(StateTransformError, match="observational"):
        ExplicitStateObjectPlan.from_dict(dirty)


def test_plan_cannot_claim_authority_flags() -> None:
    packet = _compile()
    graph = _ownership()
    plan = compile_explicit_state_object_plan(packet, ownership_graph=graph)
    payload = plan.to_dict()
    payload["can_authorize_completion"] = True
    payload["plan_cid"] = cid_for_dag_json(
        {key: value for key, value in payload.items() if key != "plan_cid"}
    )
    with pytest.raises(StateTransformError, match="can_authorize_completion"):
        ExplicitStateObjectPlan.from_dict(payload)
    payload = plan.to_dict()
    payload["plan_is_nomination_only"] = False
    payload["plan_cid"] = cid_for_dag_json(
        {key: value for key, value in payload.items() if key != "plan_cid"}
    )
    with pytest.raises(StateTransformError, match="nomination_only"):
        ExplicitStateObjectPlan.from_dict(payload)


def test_empty_write_paths_are_unrestricted_scope() -> None:
    packet = _compile()
    with pytest.raises(StateTransformError, match="unrestricted scope"):
        ExplicitStateObject(
            transform_kind=ExplicitStateObjectKind.INJECTION,
            owner_id=OWNER_ID,
            uniqueness="unique",
            source_module="pkg.mod",
            destination_module="pkg.extracted",
            write_paths=(),
            preimage_cid=packet.preimage.preimage_cid,
            packet_cid=packet.packet_cid,
            tree_id=packet.tree_id,
        )


def test_incomplete_transform_cannot_be_constructed() -> None:
    packet = _compile()
    with pytest.raises(StateTransformError, match="incomplete"):
        ExplicitStateObject(
            transform_kind=ExplicitStateObjectKind.STATE_OBJECT,
            owner_id=OWNER_ID,
            uniqueness="unique",
            source_module="pkg.mod",
            destination_module="pkg.extracted",
            write_paths=WRITE_PATHS,
            preimage_cid=packet.preimage.preimage_cid,
            packet_cid=packet.packet_cid,
            tree_id=packet.tree_id,
            complete_obligations=False,
        )


def test_packet_must_remain_spar019_analyzer() -> None:
    packet = _compile()
    payload = packet.to_dict()
    payload["analyzer_id"] = ANALYZER_ID
    identity = {key: value for key, value in payload.items() if key != "packet_cid"}
    payload["packet_cid"] = cid_for_dag_json(identity)
    with pytest.raises(Exception, match="SPAR-019 analyzer"):
        compile_explicit_state_objects(payload, ownership_graph=_ownership())


def test_reexport_only_packet_without_state_owners_is_not_a_state_transform() -> None:
    candidate = _candidate()
    packet = _compile(
        candidate,
        facade_plan=_facade(
            candidate,
            migration_kind="reexport",
            kind="import_path",
            disposition="preserve",
            obligation_id="obl:import",
            subject_id="symbol:pkg.mod.Record",
        ),
    )
    with pytest.raises(StateTransformError, match="requires explicit state-object"):
        compile_explicit_state_objects(
            packet,
            ownership_graph=_ownership(include_candidate=False, admitted=False),
        )


def test_ownership_graph_is_required_for_adapter_edits() -> None:
    packet = _compile()
    with pytest.raises(StateTransformError, match="incomplete"):
        compile_explicit_state_objects(packet)
