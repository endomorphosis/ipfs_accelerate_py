"""Independent contract tests for SPAR-024 BindingCompatibilityAdapter."""

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
    RefactorTransformationPacket,
    RewriteKind,
    compile_refactor_transformation_packet,
)
from ipfs_accelerate_py.agent_supervisor.semantic_refactoring.binding_compatibility import (
    ADAPTER_CAN_AUTHORIZE_COMPLETION,
    ADAPTER_CAN_AUTHORIZE_TRANSITION,
    ADAPTER_CAN_CREATE_AUTHORITY,
    ADAPTER_CONTRACT_VERSION,
    ADAPTER_IS_NOMINATION_ONLY,
    ANALYZER_ID,
    AUTHORITY,
    AUTHORITY_OWNER,
    BINDING_COMPATIBILITY_ADAPTER_INTERFACE,
    BINDING_COMPATIBILITY_PLAN_INTERFACE,
    BINDING_COMPATIBILITY_RECEIPT_INTERFACE,
    BINDING_IDENTITY_KINDS,
    DECLARED_BINDING_COMPATIBILITY_KINDS,
    DECLARED_INCOMPATIBILITY_CLASSIFICATIONS,
    DRY_RUN_IS_DETERMINISTIC,
    DRY_RUN_MUTATES,
    DUCKLAKE_IS_AUTHORITY,
    EXECUTOR_IS_NOMINATION_ONLY,
    GOAL_ID,
    HANDLED_ADAPTER_KINDS,
    HANDLED_MIGRATION_KINDS,
    HANDLED_REWRITE_KINDS,
    IDENTITY_EXCLUDED_FIELDS,
    INTENTIONAL_INCOMPATIBILITY_MUST_BE_CLASSIFIED,
    MARKDOWN_IS_NOT_COMPLETION,
    MODEL_OUTPUT_IS_PROPOSAL_ONLY,
    PLAN_IS_NOMINATION_ONLY,
    PROGRAM,
    PROJECTION_CLUSTERING_IS_AUTHORITY,
    RAW_SOURCE_REQUIRED,
    RUNTIME_OBSERVATION_IS_NOT_STATIC_FACT,
    SILENT_INCOMPATIBILITY_REJECTED,
    SOFT_SIGNALS_CANNOT_OVERRIDE_HARD_CONSTRAINTS,
    TASK_ID,
    TEST_PASS_IS_NOT_COMPLETION,
    VECTOR_SIMILARITY_IS_AUTHORITY,
    WORKER_SELF_APPROVAL,
    BindingCompatibilityAdapter,
    BindingCompatibilityError,
    BindingCompatibilityKind,
    BindingCompatibilityPlan,
    BindingCompatibilityReceipt,
    IncompatibilityClassification,
    assert_not_competing_capsule_family,
    binding_compatibility_cid_profile,
    classify_intentional_incompatibility,
    compile_binding_compatibility_adapters,
    compile_binding_compatibility_plan,
    compile_binding_compatibility_receipt,
    decode_canonical_adapter,
    decode_canonical_plan,
    decode_canonical_receipt,
    dry_run_binding_compatibility_adapters,
    encode_canonical_adapter,
    encode_canonical_plan,
    encode_canonical_receipt,
    execute_binding_compatibility_adapters,
    execute_binding_compatibility_plan,
    provider_free_exports,
    verify_preimages,
)


ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    ROOT
    / "ipfs_accelerate_py"
    / "agent_supervisor"
    / "semantic_refactoring"
    / "binding_compatibility.py"
)
TEST_PATH = Path(__file__).resolve()
WRITE_SCOPE = (
    "ipfs_accelerate_py/agent_supervisor/semantic_refactoring/binding_compatibility.py",
    "test/api/semantic_refactoring/test_binding_compatibility.py",
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
SYMBOL_ID = "symbol:pkg.mod.Record"
CONSUMER_PLANS = (
    {
        "obligation_id": "obl:signature",
        "consumer_id": "pkg.cli",
        "subject_id": SYMBOL_ID,
        "subject_module": "pkg.mod",
        "kind": "signature",
        "disposition": "migrate",
        "migration_kind": "wrapper",
        "required": True,
        "target_module_id": "",
    },
)


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
                "edge_id": "edge:binding",
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
    kind: str = "signature",
    obligation_id: str = "obl:signature",
    **overrides: Any,
) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "tree_id": TREE_ID,
        "plan_cid": _cid("facade"),
        "consumer_plans": [
            {
                "obligation_id": obligation_id,
                "consumer_id": "pkg.cli",
                "subject_id": SYMBOL_ID,
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
                "subject_id": SYMBOL_ID,
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


def _inventory(
    *,
    kind: str = "signature",
    disposition: str = "migrate",
    obligation_id: str = "obl:signature",
    required: bool = True,
) -> dict[str, Any]:
    return {
        "tree_id": TREE_ID,
        "inventory_cid": _cid("inventory"),
        "obligations": [
            {
                "obligation_id": obligation_id,
                "consumer_id": "pkg.cli",
                "subject_id": SYMBOL_ID,
                "subject_module": "pkg.mod",
                "kind": kind,
                "disposition": disposition,
                "required": required,
            }
        ],
        "consumers": [
            {
                "consumer_id": "pkg.cli",
                "module_name": "pkg.cli",
                "role": "module",
                "required": True,
                "evidence_class": "exact_static_fact",
            }
        ],
    }


def test_owned_paths_and_task_identity_are_exact() -> None:
    assert TASK_ID == "SPAR-024"
    assert GOAL_ID == "SPAR-G042"
    assert PROGRAM == "semantic-preserving-autonomous-remodularization-v1"
    assert BINDING_COMPATIBILITY_ADAPTER_INTERFACE == "BindingCompatibilityAdapter@1"
    assert BINDING_COMPATIBILITY_PLAN_INTERFACE == "BindingCompatibilityPlan@1"
    assert BINDING_COMPATIBILITY_RECEIPT_INTERFACE == "BindingCompatibilityReceipt@1"
    assert ADAPTER_CONTRACT_VERSION == "1"
    assert ANALYZER_ID.endswith("binding_compatibility@1")
    assert ANALYZER_ID != SPAR019_ANALYZER_ID
    assert MODULE_PATH.is_file()
    assert TEST_PATH.is_file()
    for relative in WRITE_SCOPE:
        assert (ROOT / relative).is_file()


def test_authority_flags_cannot_self_authorize() -> None:
    assert AUTHORITY == "partition orchestration"
    assert AUTHORITY_OWNER == "ipfs_accelerate_py"
    assert ADAPTER_CAN_AUTHORIZE_COMPLETION is False
    assert ADAPTER_CAN_AUTHORIZE_TRANSITION is False
    assert ADAPTER_CAN_CREATE_AUTHORITY is False
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
    assert ADAPTER_IS_NOMINATION_ONLY is True
    assert RAW_SOURCE_REQUIRED is True
    assert DRY_RUN_IS_DETERMINISTIC is True
    assert DRY_RUN_MUTATES is False
    assert RUNTIME_OBSERVATION_IS_NOT_STATIC_FACT is True
    assert SILENT_INCOMPATIBILITY_REJECTED is True
    assert INTENTIONAL_INCOMPATIBILITY_MUST_BE_CLASSIFIED is True
    profile = binding_compatibility_cid_profile()
    assert profile["codec"] == "dag-json"
    assert "not universal meaning" in profile["rule"]
    assert DECLARED_BINDING_COMPATIBILITY_KINDS == {
        "signature",
        "annotation",
        "module_name",
        "qualname",
        "pickle",
        "serialization",
        "introspection",
        "traceback",
        "documentation",
        "patch_target",
    }
    assert DECLARED_INCOMPATIBILITY_CLASSIFICATIONS == {
        "preserve",
        "wrapper",
        "migration",
        "explicit_incompatibility",
    }
    assert HANDLED_ADAPTER_KINDS == {AdapterKind.WRAPPER.value}
    assert HANDLED_REWRITE_KINDS == {
        RewriteKind.SERIALIZATION.value,
        RewriteKind.INTROSPECTION.value,
        RewriteKind.TRACEBACK.value,
        RewriteKind.PATCH_TARGET.value,
        RewriteKind.DEPRECATION.value,
    }
    assert HANDLED_MIGRATION_KINDS == {
        "wrapper",
        "serialization",
        "introspection",
        "traceback",
        "deprecation",
        "patch_target",
    }
    assert BINDING_IDENTITY_KINDS == {
        "module_name",
        "qualname",
        "pickle",
        "serialization",
    }


def test_module_defines_predicted_symbols_not_capsule_family() -> None:
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    names = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
    assert "BindingCompatibilityAdapter" in names
    assert "BindingCompatibilityPlan" in names
    assert "BindingCompatibilityReceipt" in names
    for capsule in CAPSULE_TYPES:
        assert capsule not in names
    assert_not_competing_capsule_family()
    exports = provider_free_exports()
    assert "BindingCompatibilityAdapter" in exports
    assert "compile_binding_compatibility_plan" in exports
    assert "execute_binding_compatibility_plan" in exports
    assert "dry_run_binding_compatibility_adapters" in exports


def test_protected_paths_are_not_owned_write_scope() -> None:
    owned = set(WRITE_SCOPE)
    for relative in PROTECTED_PATHS:
        assert relative not in owned
        assert (ROOT / relative).exists()


def test_signature_wrapper_is_nominated_from_packet() -> None:
    packet = _compile()
    adapters = compile_binding_compatibility_adapters(
        packet, consumer_plans=CONSUMER_PLANS
    )
    kinds = {item.adapter_kind for item in adapters}
    assert BindingCompatibilityKind.SIGNATURE.value in kinds
    wrapper = next(
        item
        for item in adapters
        if item.adapter_kind == BindingCompatibilityKind.SIGNATURE.value
    )
    assert wrapper.subject_id == SYMBOL_ID
    assert wrapper.source_module == "pkg.mod"
    assert wrapper.destination_module == packet.expected_delta.destination_module_ids[0]
    assert wrapper.packet_adapter_kind == AdapterKind.WRAPPER.value
    assert wrapper.classification == IncompatibilityClassification.WRAPPER.value
    assert wrapper.adapter_is_nomination_only is True
    assert wrapper.write_paths == WRITE_PATHS


def test_annotation_qualname_and_module_wrappers_are_nominated() -> None:
    candidate = _candidate()
    annotation = _compile(
        candidate,
        facade_plan=_facade(
            candidate,
            migration_kind="wrapper",
            kind="annotation",
            obligation_id="obl:annotation",
        ),
    )
    annotation_kinds = {
        item.adapter_kind
        for item in compile_binding_compatibility_adapters(
            annotation,
            consumer_plans=(
                {
                    **CONSUMER_PLANS[0],
                    "kind": "annotation",
                    "obligation_id": "obl:annotation",
                },
            ),
        )
    }
    assert BindingCompatibilityKind.ANNOTATION.value in annotation_kinds
    qualname = _compile(
        candidate,
        facade_plan=_facade(
            candidate,
            migration_kind="wrapper",
            kind="qualname",
            obligation_id="obl:qualname",
        ),
    )
    qualname_kinds = {
        item.adapter_kind
        for item in compile_binding_compatibility_adapters(
            qualname,
            consumer_plans=(
                {
                    **CONSUMER_PLANS[0],
                    "kind": "qualname",
                    "obligation_id": "obl:qualname",
                    "migration_kind": "wrapper",
                },
            ),
        )
    }
    assert BindingCompatibilityKind.QUALNAME.value in qualname_kinds


def test_pickle_introspection_traceback_docs_and_patch_are_nominated() -> None:
    candidate = _candidate()
    cases = (
        ("serialization", "pickle", BindingCompatibilityKind.PICKLE.value),
        ("introspection", "introspection", BindingCompatibilityKind.INTROSPECTION.value),
        ("traceback", "traceback", BindingCompatibilityKind.TRACEBACK.value),
        ("deprecation", "documentation", BindingCompatibilityKind.DOCUMENTATION.value),
        ("patch_target", "patch_target", BindingCompatibilityKind.PATCH_TARGET.value),
    )
    for migration_kind, kind, expected in cases:
        packet = _compile(
            candidate,
            facade_plan=_facade(
                candidate,
                migration_kind=migration_kind,
                kind=kind,
                obligation_id=f"obl:{kind}",
            ),
        )
        adapters = compile_binding_compatibility_adapters(
            packet,
            consumer_plans=(
                {
                    **CONSUMER_PLANS[0],
                    "kind": kind,
                    "obligation_id": f"obl:{kind}",
                    "migration_kind": migration_kind,
                },
            ),
        )
        kinds = {item.adapter_kind for item in adapters}
        assert expected in kinds
        nominated = next(item for item in adapters if item.adapter_kind == expected)
        assert nominated.classification == IncompatibilityClassification.MIGRATION.value
        assert nominated.migration_kind == migration_kind


def test_preimages_are_verified_against_packet() -> None:
    packet = _compile()
    assert verify_preimages(packet) == packet.preimage.preimage_cid
    assert (
        verify_preimages(packet, claimed_preimage_cid=packet.preimage.preimage_cid)
        == packet.preimage.preimage_cid
    )
    with pytest.raises(BindingCompatibilityError, match="preimage does not verify"):
        verify_preimages(packet, claimed_preimage_cid=_cid("other-preimage"))
    with pytest.raises(BindingCompatibilityError, match="preimage does not verify"):
        compile_binding_compatibility_adapters(
            packet, claimed_preimage_cid=_cid("stale")
        )


def test_undispositioned_consumers_are_typed_terminals() -> None:
    packet = _compile()
    with pytest.raises(BindingCompatibilityError, match="undispositioned"):
        compile_binding_compatibility_adapters(
            packet,
            undispositioned_consumer_ids=("pkg.ghost",),
        )
    with pytest.raises(BindingCompatibilityError, match="undispositioned"):
        compile_binding_compatibility_adapters(
            packet,
            consumer_plans=(
                {
                    **CONSUMER_PLANS[0],
                    "disposition": "undispositioned",
                    "required": True,
                },
            ),
        )


def test_required_unsupported_obligation_is_typed_terminal() -> None:
    packet = _compile()
    with pytest.raises(BindingCompatibilityError, match="undispositioned"):
        compile_binding_compatibility_adapters(
            packet,
            inventory=_inventory(disposition="unsupported"),
        )


def test_explicit_incompatibility_is_classified_not_silent() -> None:
    packet = _compile()
    adapters = compile_binding_compatibility_adapters(
        packet,
        consumer_plans=(
            {
                **CONSUMER_PLANS[0],
                "disposition": "explicit_incompatibility",
            },
        ),
        inventory=_inventory(disposition="explicit_incompatibility"),
    )
    nominated = adapters[0]
    assert (
        nominated.classification
        == IncompatibilityClassification.EXPLICIT_INCOMPATIBILITY.value
    )
    assert nominated.preserve_binding is False
    plan = compile_binding_compatibility_plan(
        packet,
        consumer_plans=(
            {
                **CONSUMER_PLANS[0],
                "disposition": "explicit_incompatibility",
            },
        ),
        inventory=_inventory(disposition="explicit_incompatibility"),
    )
    assert plan.classified_incompatibility_ids == ("obl:signature",)
    assert plan.preserve_binding is False
    assert plan.silent_incompatibility is False


def test_silent_binding_incompatibility_is_rejected() -> None:
    with pytest.raises(BindingCompatibilityError, match="silent binding"):
        classify_intentional_incompatibility(
            disposition="preserve",
            adapter_kind="qualname",
            source_module="pkg.mod",
            destination_module="pkg.extracted",
        )
    packet = _compile()
    with pytest.raises(BindingCompatibilityError, match="silent binding"):
        BindingCompatibilityAdapter(
            adapter_kind=BindingCompatibilityKind.QUALNAME,
            subject_id=SYMBOL_ID,
            source_module="pkg.mod",
            destination_module="pkg.extracted",
            write_paths=WRITE_PATHS,
            preimage_cid=packet.preimage.preimage_cid,
            packet_cid=packet.packet_cid,
            tree_id=packet.tree_id,
            classification=IncompatibilityClassification.PRESERVE,
            preserve_binding=True,
        )


def test_inventory_tree_must_match_packet() -> None:
    packet = _compile()
    dirty = _inventory()
    dirty["tree_id"] = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    with pytest.raises(BindingCompatibilityError, match="SPAR-011 tree_id"):
        compile_binding_compatibility_adapters(packet, inventory=dirty)


def test_required_inventory_obligation_must_be_classified() -> None:
    packet = _compile()
    with pytest.raises(BindingCompatibilityError, match="missing a classified adapter"):
        compile_binding_compatibility_adapters(
            packet,
            inventory=_inventory(
                kind="pickle",
                obligation_id="obl:pickle",
            ),
        )


def test_vector_evidence_cannot_admit_inventory() -> None:
    packet = _compile()
    dirty = _inventory()
    dirty["evidence_class"] = "vector_candidate"
    with pytest.raises(BindingCompatibilityError, match="vector or model"):
        compile_binding_compatibility_adapters(packet, inventory=dirty)


def test_dry_run_is_deterministic_and_does_not_mutate() -> None:
    packet = _compile()
    first = dry_run_binding_compatibility_adapters(packet)
    second = execute_binding_compatibility_adapters(packet)
    assert first.receipt_cid == second.receipt_cid
    assert first.mutated is False
    assert first.deterministic is True
    assert first.can_authorize_transition is False
    assert first.can_authorize_completion is False
    assert first.executor_is_nomination_only is True
    assert first.packet_cid == packet.packet_cid
    assert first.preimage_cid == packet.preimage.preimage_cid
    assert first.preimage_verified is True
    assert first.write_paths == WRITE_PATHS
    restored = BindingCompatibilityReceipt.from_dict(first.to_dict())
    assert restored == first
    with pytest.raises(BindingCompatibilityError, match="cannot mutate"):
        execute_binding_compatibility_adapters(packet, mutate=True)


def test_plan_round_trip_is_deterministic() -> None:
    packet = _compile()
    first = compile_binding_compatibility_adapters(packet)
    second = compile_binding_compatibility_adapters(packet)
    assert [item.adapter_cid for item in first] == [
        item.adapter_cid for item in second
    ]
    restored = decode_canonical_adapter(encode_canonical_adapter(first[0]))
    assert restored == first[0]
    plan = compile_binding_compatibility_plan(packet)
    assert decode_canonical_plan(encode_canonical_plan(plan)) == plan
    assert execute_binding_compatibility_plan(packet) == plan
    receipt = compile_binding_compatibility_receipt(packet)
    assert decode_canonical_receipt(encode_canonical_receipt(receipt)) == receipt
    assert receipt.plan_cid == plan.plan_cid


def test_identity_excludes_observational_fields() -> None:
    packet = _compile()
    plan = compile_binding_compatibility_plan(packet)
    payload = plan.to_dict()
    assert not (IDENTITY_EXCLUDED_FIELDS & set(payload))
    dirty = dict(payload)
    dirty["timestamp"] = "now"
    with pytest.raises(BindingCompatibilityError, match="observational"):
        BindingCompatibilityPlan.from_dict(dirty)


def test_plan_cannot_claim_authority_flags() -> None:
    packet = _compile()
    plan = compile_binding_compatibility_plan(packet)
    payload = plan.to_dict()
    payload["can_authorize_completion"] = True
    payload["plan_cid"] = cid_for_dag_json(
        {key: value for key, value in payload.items() if key != "plan_cid"}
    )
    with pytest.raises(BindingCompatibilityError, match="can_authorize_completion"):
        BindingCompatibilityPlan.from_dict(payload)
    payload = plan.to_dict()
    payload["plan_is_nomination_only"] = False
    payload["plan_cid"] = cid_for_dag_json(
        {key: value for key, value in payload.items() if key != "plan_cid"}
    )
    with pytest.raises(BindingCompatibilityError, match="nomination_only"):
        BindingCompatibilityPlan.from_dict(payload)


def test_same_source_and_destination_require_preserve_or_explicit() -> None:
    packet = _compile()
    with pytest.raises(BindingCompatibilityError, match="preserve_binding"):
        BindingCompatibilityAdapter(
            adapter_kind=BindingCompatibilityKind.SIGNATURE,
            subject_id=SYMBOL_ID,
            source_module="pkg.mod",
            destination_module="pkg.mod",
            write_paths=WRITE_PATHS,
            preimage_cid=packet.preimage.preimage_cid,
            packet_cid=packet.packet_cid,
            tree_id=packet.tree_id,
            classification=IncompatibilityClassification.WRAPPER,
            preserve_binding=False,
        )
    preserved = BindingCompatibilityAdapter(
        adapter_kind=BindingCompatibilityKind.SIGNATURE,
        subject_id=SYMBOL_ID,
        source_module="pkg.mod",
        destination_module="pkg.mod",
        write_paths=WRITE_PATHS,
        preimage_cid=packet.preimage.preimage_cid,
        packet_cid=packet.packet_cid,
        tree_id=packet.tree_id,
        classification=IncompatibilityClassification.PRESERVE,
        preserve_binding=True,
    )
    assert preserved.preserve_binding is True
    explicit = BindingCompatibilityAdapter(
        adapter_kind=BindingCompatibilityKind.QUALNAME,
        subject_id=SYMBOL_ID,
        source_module="pkg.mod",
        destination_module="pkg.mod",
        write_paths=WRITE_PATHS,
        preimage_cid=packet.preimage.preimage_cid,
        packet_cid=packet.packet_cid,
        tree_id=packet.tree_id,
        classification=IncompatibilityClassification.EXPLICIT_INCOMPATIBILITY,
        preserve_binding=False,
    )
    assert explicit.classification == (
        IncompatibilityClassification.EXPLICIT_INCOMPATIBILITY.value
    )


def test_packet_must_remain_spar019_analyzer() -> None:
    packet = _compile()
    payload = packet.to_dict()
    payload["analyzer_id"] = ANALYZER_ID
    identity = {key: value for key, value in payload.items() if key != "packet_cid"}
    payload["packet_cid"] = cid_for_dag_json(identity)
    with pytest.raises(Exception, match="SPAR-019 analyzer"):
        compile_binding_compatibility_adapters(payload)


def test_empty_write_paths_are_unrestricted_scope() -> None:
    packet = _compile()
    with pytest.raises(BindingCompatibilityError, match="unrestricted scope"):
        BindingCompatibilityAdapter(
            adapter_kind=BindingCompatibilityKind.SIGNATURE,
            subject_id=SYMBOL_ID,
            source_module="pkg.mod",
            destination_module="pkg.extracted",
            write_paths=(),
            preimage_cid=packet.preimage.preimage_cid,
            packet_cid=packet.packet_cid,
            tree_id=packet.tree_id,
            classification=IncompatibilityClassification.WRAPPER,
            preserve_binding=False,
        )


def test_cli_only_packet_is_not_a_binding_compatibility_adapter() -> None:
    candidate = _candidate()
    packet = _compile(
        candidate,
        facade_plan=_facade(
            candidate,
            migration_kind="cli",
            kind="cli",
            disposition="migrate",
            obligation_id="obl:cli",
        ),
    )
    with pytest.raises(
        BindingCompatibilityError, match="binding, introspection, or serialization"
    ):
        compile_binding_compatibility_adapters(packet)


def test_reexport_only_packet_is_not_a_binding_compatibility_adapter() -> None:
    candidate = _candidate()
    packet = _compile(
        candidate,
        facade_plan=_facade(
            candidate,
            migration_kind="reexport",
            kind="import_path",
            disposition="preserve",
            obligation_id="obl:import",
        ),
    )
    with pytest.raises(
        BindingCompatibilityError, match="binding, introspection, or serialization"
    ):
        compile_binding_compatibility_adapters(packet)
