"""Independent contract tests for SPAR-021 ImportRewrite and ReexportPlan."""

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
    BoundedEdit,
    EditKind,
    RefactorTransformationPacket,
    RewriteKind,
    compile_refactor_transformation_packet,
)
from ipfs_accelerate_py.agent_supervisor.semantic_refactoring.import_rewriter import (
    ANALYZER_ID,
    AUTHORITY,
    AUTHORITY_OWNER,
    DECLARED_IMPORT_REWRITE_KINDS,
    DRY_RUN_IS_DETERMINISTIC,
    DRY_RUN_MUTATES,
    DUCKLAKE_IS_AUTHORITY,
    EXECUTOR_IS_NOMINATION_ONLY,
    GOAL_ID,
    HANDLED_REWRITE_KINDS,
    IDENTITY_EXCLUDED_FIELDS,
    IMPORT_REWRITE_INTERFACE,
    IMPORT_REWRITE_RECEIPT_INTERFACE,
    MARKDOWN_IS_NOT_COMPLETION,
    MODEL_OUTPUT_IS_PROPOSAL_ONLY,
    PROGRAM,
    PROJECTION_CLUSTERING_IS_AUTHORITY,
    RAW_SOURCE_REQUIRED,
    REEXPORT_PLAN_INTERFACE,
    REWRITE_CAN_AUTHORIZE_COMPLETION,
    REWRITE_CAN_AUTHORIZE_TRANSITION,
    REWRITE_CAN_CREATE_AUTHORITY,
    REWRITE_CONTRACT_VERSION,
    SOFT_SIGNALS_CANNOT_OVERRIDE_HARD_CONSTRAINTS,
    TASK_ID,
    TEST_PASS_IS_NOT_COMPLETION,
    VECTOR_SIMILARITY_IS_AUTHORITY,
    WORKER_SELF_APPROVAL,
    ImportRewrite,
    ImportRewriteKind,
    ImportRewriteReceipt,
    ImportRewriterError,
    ReexportPlan,
    assert_not_competing_capsule_family,
    compile_import_rewrite_receipt,
    compile_import_rewrites,
    compile_reexport_plan,
    decode_canonical_plan,
    decode_canonical_receipt,
    decode_canonical_rewrite,
    dry_run_import_rewrites,
    encode_canonical_plan,
    encode_canonical_receipt,
    encode_canonical_rewrite,
    execute_import_rewrites,
    execute_reexport_plan,
    import_rewriter_cid_profile,
    provider_free_exports,
    verify_preimages,
)


ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    ROOT
    / "ipfs_accelerate_py"
    / "agent_supervisor"
    / "semantic_refactoring"
    / "import_rewriter.py"
)
TEST_PATH = Path(__file__).resolve()
WRITE_SCOPE = (
    "ipfs_accelerate_py/agent_supervisor/semantic_refactoring/import_rewriter.py",
    "test/api/semantic_refactoring/test_import_rewriter.py",
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
        "obligation_id": "obl:import",
        "consumer_id": "pkg.cli",
        "subject_id": SYMBOL_ID,
        "subject_module": "pkg.mod",
        "kind": "import_path",
        "disposition": "preserve",
        "migration_kind": "reexport",
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
                "subject_id": SYMBOL_ID,
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


def _with_rewrite(
    packet: RefactorTransformationPacket,
    *,
    rewrite_kind: str,
    obligation_id: str = "obl:callsite",
    member_ids: Sequence[str] = (SYMBOL_ID,),
) -> RefactorTransformationPacket:
    extra = BoundedEdit(
        kind=EditKind.REWRITE,
        source_id="pkg.mod",
        destination_id=packet.expected_delta.destination_module_ids[0],
        member_ids=member_ids,
        rewrite_kind=rewrite_kind,
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


def test_owned_paths_and_task_identity_are_exact() -> None:
    assert TASK_ID == "SPAR-021"
    assert GOAL_ID == "SPAR-G041"
    assert PROGRAM == "semantic-preserving-autonomous-remodularization-v1"
    assert IMPORT_REWRITE_INTERFACE == "ImportRewrite@1"
    assert REEXPORT_PLAN_INTERFACE == "ReexportPlan@1"
    assert IMPORT_REWRITE_RECEIPT_INTERFACE == "ImportRewriteReceipt@1"
    assert REWRITE_CONTRACT_VERSION == "1"
    assert ANALYZER_ID.endswith("import_rewriter@1")
    assert ANALYZER_ID != SPAR019_ANALYZER_ID
    assert MODULE_PATH.is_file()
    assert TEST_PATH.is_file()
    for relative in WRITE_SCOPE:
        assert (ROOT / relative).is_file()


def test_authority_flags_cannot_self_authorize() -> None:
    assert AUTHORITY == "partition orchestration"
    assert AUTHORITY_OWNER == "ipfs_accelerate_py"
    assert REWRITE_CAN_AUTHORIZE_COMPLETION is False
    assert REWRITE_CAN_AUTHORIZE_TRANSITION is False
    assert REWRITE_CAN_CREATE_AUTHORITY is False
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
    assert DRY_RUN_IS_DETERMINISTIC is True
    assert DRY_RUN_MUTATES is False
    profile = import_rewriter_cid_profile()
    assert profile["codec"] == "dag-json"
    assert "not universal meaning" in profile["rule"]
    assert DECLARED_IMPORT_REWRITE_KINDS == {"import", "callsite"}
    assert HANDLED_REWRITE_KINDS == {"import", "callsite", "reexport"}


def test_module_defines_predicted_symbols_not_capsule_family() -> None:
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    names = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
    assert "ImportRewrite" in names
    assert "ReexportPlan" in names
    assert "ImportRewriteReceipt" in names
    for capsule in CAPSULE_TYPES:
        assert capsule not in names
    assert_not_competing_capsule_family()
    exports = provider_free_exports()
    assert "ImportRewrite" in exports
    assert "ReexportPlan" in exports
    assert "compile_import_rewrites" in exports
    assert "compile_reexport_plan" in exports
    assert "execute_import_rewrites" in exports
    assert "dry_run_import_rewrites" in exports


def test_protected_paths_are_not_owned_write_scope() -> None:
    owned = set(WRITE_SCOPE)
    for relative in PROTECTED_PATHS:
        assert relative not in owned
        assert (ROOT / relative).exists()


def test_preserve_reexport_creates_authorized_export_not_consumer_rewrite() -> None:
    packet = _compile()
    plan = compile_reexport_plan(packet, consumer_plans=CONSUMER_PLANS)
    assert plan.authorized is True
    assert plan.rewrite_kind == RewriteKind.REEXPORT.value
    assert plan.source_module == "pkg.mod"
    assert plan.destination_module == packet.expected_delta.destination_module_ids[0]
    assert SYMBOL_ID in plan.symbol_ids
    assert "pkg.cli" in plan.consumer_ids
    assert plan.preimage_cid == packet.preimage.preimage_cid
    assert plan.packet_cid == packet.packet_cid
    assert plan.plan_is_nomination_only is True
    assert plan.can_authorize_completion is False
    rewrites = compile_import_rewrites(packet, consumer_plans=CONSUMER_PLANS)
    assert rewrites == ()


def test_migrate_nominates_exact_consumer_import_rewrite() -> None:
    candidate = _candidate()
    packet = _compile(
        candidate,
        facade_plan=_facade(candidate, disposition="migrate"),
    )
    consumer_plans = (
        {
            **CONSUMER_PLANS[0],
            "disposition": "migrate",
            "target_module_id": candidate.candidate_cid,
        },
    )
    rewrites = compile_import_rewrites(packet, consumer_plans=consumer_plans)
    assert len(rewrites) == 1
    rewrite = rewrites[0]
    assert rewrite.rewrite_kind == ImportRewriteKind.IMPORT.value
    assert rewrite.consumer_id == "pkg.cli"
    assert rewrite.symbol_id == SYMBOL_ID
    assert rewrite.source_module == "pkg.mod"
    assert rewrite.destination_module == candidate.candidate_cid
    assert rewrite.preimage_cid == packet.preimage.preimage_cid
    assert rewrite.rewrite_is_nomination_only is True
    plan = compile_reexport_plan(packet, consumer_plans=consumer_plans)
    assert plan.authorized is True
    assert SYMBOL_ID in plan.symbol_ids


def test_callsite_rewrite_is_nominated_from_packet_edit() -> None:
    packet = _with_rewrite(_compile(), rewrite_kind=RewriteKind.CALLSITE.value)
    rewrites = compile_import_rewrites(packet)
    kinds = {item.rewrite_kind for item in rewrites}
    assert ImportRewriteKind.CALLSITE.value in kinds
    callsite = next(
        item for item in rewrites if item.rewrite_kind == ImportRewriteKind.CALLSITE.value
    )
    assert callsite.symbol_id == SYMBOL_ID
    assert callsite.source_module == "pkg.mod"
    assert callsite.write_paths == WRITE_PATHS


def test_preimages_are_verified_against_packet() -> None:
    packet = _compile()
    assert verify_preimages(packet) == packet.preimage.preimage_cid
    assert (
        verify_preimages(packet, claimed_preimage_cid=packet.preimage.preimage_cid)
        == packet.preimage.preimage_cid
    )
    with pytest.raises(ImportRewriterError, match="preimage does not verify"):
        verify_preimages(packet, claimed_preimage_cid=_cid("other-preimage"))
    with pytest.raises(ImportRewriterError, match="preimage does not verify"):
        compile_import_rewrites(packet, claimed_preimage_cid=_cid("stale"))


def test_new_cycles_are_rejected() -> None:
    packet = _compile()
    destination = packet.expected_delta.destination_module_ids[0]
    with pytest.raises(ImportRewriterError, match="cycles"):
        compile_reexport_plan(
            packet,
            import_graph={destination: ["pkg.mod"]},
        )


def test_cycle_free_preserve_reexport_is_admitted() -> None:
    packet = _compile()
    destination = packet.expected_delta.destination_module_ids[0]
    plan = compile_reexport_plan(
        packet,
        consumer_plans=CONSUMER_PLANS,
        import_graph={"pkg.cli": ["pkg.mod"]},
    )
    assert plan.source_module == "pkg.mod"
    assert plan.destination_module == destination
    receipt = dry_run_import_rewrites(
        packet,
        consumer_plans=CONSUMER_PLANS,
        import_graph={"pkg.cli": ["pkg.mod"]},
    )
    assert receipt.cycle_free is True


def test_undispositioned_consumers_are_typed_terminals() -> None:
    packet = _compile()
    with pytest.raises(ImportRewriterError, match="undispositioned"):
        compile_import_rewrites(
            packet,
            undispositioned_consumer_ids=("pkg.ghost",),
        )
    with pytest.raises(ImportRewriterError, match="undispositioned"):
        compile_reexport_plan(
            packet,
            consumer_plans=(
                {
                    **CONSUMER_PLANS[0],
                    "disposition": "undispositioned",
                    "required": True,
                },
            ),
        )


def test_dry_run_is_deterministic_and_does_not_mutate() -> None:
    packet = _compile()
    first = dry_run_import_rewrites(packet, consumer_plans=CONSUMER_PLANS)
    second = execute_import_rewrites(packet, consumer_plans=CONSUMER_PLANS)
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
    assert first.reexport_cids
    restored = ImportRewriteReceipt.from_dict(first.to_dict())
    assert restored == first
    with pytest.raises(ImportRewriterError, match="cannot mutate"):
        execute_import_rewrites(packet, mutate=True)


def test_rewrite_and_plan_round_trip_are_deterministic() -> None:
    candidate = _candidate()
    packet = _compile(
        candidate,
        facade_plan=_facade(candidate, disposition="migrate"),
    )
    consumer_plans = (
        {
            **CONSUMER_PLANS[0],
            "disposition": "migrate",
            "target_module_id": candidate.candidate_cid,
        },
    )
    first = compile_import_rewrites(packet, consumer_plans=consumer_plans)
    second = compile_import_rewrites(packet, consumer_plans=consumer_plans)
    assert [item.rewrite_cid for item in first] == [item.rewrite_cid for item in second]
    restored = decode_canonical_rewrite(encode_canonical_rewrite(first[0]))
    assert restored == first[0]
    plan = compile_reexport_plan(packet, consumer_plans=consumer_plans)
    assert decode_canonical_plan(encode_canonical_plan(plan)) == plan
    receipt = compile_import_rewrite_receipt(packet, consumer_plans=consumer_plans)
    assert decode_canonical_receipt(encode_canonical_receipt(receipt)) == receipt
    assert execute_reexport_plan(packet, consumer_plans=consumer_plans) == plan


def test_identity_excludes_observational_fields() -> None:
    packet = _compile()
    plan = compile_reexport_plan(packet, consumer_plans=CONSUMER_PLANS)
    payload = plan.to_dict()
    assert not (IDENTITY_EXCLUDED_FIELDS & set(payload))
    dirty = dict(payload)
    dirty["timestamp"] = "now"
    with pytest.raises(ImportRewriterError, match="observational"):
        ReexportPlan.from_dict(dirty)


def test_rewrite_cannot_claim_authority_flags() -> None:
    packet = _compile()
    plan = compile_reexport_plan(packet, consumer_plans=CONSUMER_PLANS)
    payload = plan.to_dict()
    payload["can_authorize_completion"] = True
    payload["plan_cid"] = cid_for_dag_json(
        {key: value for key, value in payload.items() if key != "plan_cid"}
    )
    with pytest.raises(ImportRewriterError, match="can_authorize_completion"):
        ReexportPlan.from_dict(payload)
    payload = plan.to_dict()
    payload["plan_is_nomination_only"] = False
    payload["plan_cid"] = cid_for_dag_json(
        {key: value for key, value in payload.items() if key != "plan_cid"}
    )
    with pytest.raises(ImportRewriterError, match="nomination_only"):
        ReexportPlan.from_dict(payload)


def test_unauthorized_reexport_fails_closed() -> None:
    packet = _compile()
    plan = compile_reexport_plan(packet, consumer_plans=CONSUMER_PLANS)
    with pytest.raises(ImportRewriterError, match="authorized"):
        ReexportPlan(
            source_module=plan.source_module,
            destination_module=plan.destination_module,
            symbol_ids=plan.symbol_ids,
            write_paths=plan.write_paths,
            preimage_cid=plan.preimage_cid,
            packet_cid=plan.packet_cid,
            tree_id=plan.tree_id,
            authorized=False,
        )


def test_wrapper_only_packet_has_no_authorized_reexport() -> None:
    candidate = _candidate()
    packet = _compile(
        candidate,
        facade_plan=_facade(candidate, migration_kind="wrapper"),
    )
    with pytest.raises(ImportRewriterError, match="authorized reexport"):
        compile_reexport_plan(packet)


def test_same_source_and_destination_fail_closed() -> None:
    packet = _compile()
    with pytest.raises(ImportRewriterError, match="must differ"):
        ImportRewrite(
            rewrite_kind=ImportRewriteKind.IMPORT,
            consumer_id="pkg.cli",
            symbol_id=SYMBOL_ID,
            source_module="pkg.mod",
            destination_module="pkg.mod",
            write_paths=WRITE_PATHS,
            preimage_cid=packet.preimage.preimage_cid,
            packet_cid=packet.packet_cid,
            tree_id=packet.tree_id,
        )


def test_packet_must_remain_spar019_analyzer() -> None:
    packet = _compile()
    payload = packet.to_dict()
    payload["analyzer_id"] = ANALYZER_ID
    identity = {key: value for key, value in payload.items() if key != "packet_cid"}
    payload["packet_cid"] = cid_for_dag_json(identity)
    with pytest.raises(Exception, match="SPAR-019 analyzer"):
        compile_import_rewrites(payload)


def test_empty_write_paths_are_unrestricted_scope() -> None:
    packet = _compile()
    with pytest.raises(ImportRewriterError, match="unrestricted scope"):
        ImportRewrite(
            rewrite_kind=ImportRewriteKind.IMPORT,
            consumer_id="pkg.cli",
            symbol_id=SYMBOL_ID,
            source_module="pkg.mod",
            destination_module="pkg.extracted",
            write_paths=(),
            preimage_cid=packet.preimage.preimage_cid,
            packet_cid=packet.packet_cid,
            tree_id=packet.tree_id,
        )
