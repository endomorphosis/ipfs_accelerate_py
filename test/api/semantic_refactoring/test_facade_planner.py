"""Independent contract tests for SPAR-018 CompatibilityFacadePlan@1."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.utils.cid_utils import cid_for_bytes, cid_for_dag_json
from ipfs_accelerate_py.agent_supervisor.semantic_refactoring.partition_generators import (
    ConstraintClass,
    GeneratorKind,
    PartitionCutEdge,
    ProgramPartitionCandidate,
)
from ipfs_accelerate_py.agent_supervisor.semantic_refactoring.partition_policy import (
    ANALYZER_ID as SPAR014_ANALYZER_ID,
    compare_partition_candidates,
)
from ipfs_accelerate_py.agent_supervisor.semantic_refactoring.facade_planner import (
    ANALYZER_ID,
    AUTHORITY,
    AUTHORITY_OWNER,
    COMPATIBILITY_FACADE_PLAN_INTERFACE,
    CONSUMER_MIGRATION_PLAN_INTERFACE,
    DECLARED_COMPATIBILITY_KINDS,
    DECLARED_CONSUMER_SURFACES,
    DECLARED_MIGRATION_KINDS,
    DISPOSITIONED,
    DUCKLAKE_IS_AUTHORITY,
    FACADE_CAN_AUTHORIZE_COMPLETION,
    FACADE_CAN_AUTHORIZE_TRANSITION,
    FACADE_CAN_CREATE_AUTHORITY,
    FACADE_CAN_RETIRE_FACADE,
    FACADE_PLANNING_RECEIPT_INTERFACE,
    FACADE_PLAN_CONTRACT_VERSION,
    GOAL_ID,
    IDENTITY_EXCLUDED_FIELDS,
    KIND_MIGRATION,
    KIND_SURFACE,
    MARKDOWN_IS_NOT_COMPLETION,
    MODEL_OUTPUT_IS_PROPOSAL_ONLY,
    PLAN_IS_NOMINATION_ONLY,
    PROGRAM,
    PROJECTION_CLUSTERING_IS_AUTHORITY,
    RAW_SOURCE_REQUIRED,
    SOFT_SIGNALS_CANNOT_OVERRIDE_HARD_CONSTRAINTS,
    SUBJECT_FACADE_PLAN_INTERFACE,
    TASK_ID,
    TEST_PASS_IS_NOT_COMPLETION,
    VECTOR_SIMILARITY_IS_AUTHORITY,
    WORKER_SELF_APPROVAL,
    CompatibilityDisposition,
    CompatibilityFacadePlan,
    CompatibilityKind,
    ConsumerMigrationPlan,
    ConsumerRole,
    ConsumerSurface,
    FacadePlannerError,
    FacadePlanningReceipt,
    MigrationKind,
    SubjectFacadePlan,
    assert_not_competing_capsule_family,
    compile_facade_plan_receipt,
    decode_canonical_plan,
    decode_canonical_receipt,
    encode_canonical_plan,
    encode_canonical_receipt,
    facade_planner_cid_profile,
    migration_kind_for,
    plan_compatibility_facades,
    provider_free_exports,
    surface_for_kind,
)


ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    ROOT
    / "ipfs_accelerate_py"
    / "agent_supervisor"
    / "semantic_refactoring"
    / "facade_planner.py"
)
TEST_PATH = Path(__file__).resolve()
WRITE_SCOPE = (
    "ipfs_accelerate_py/agent_supervisor/semantic_refactoring/facade_planner.py",
    "test/api/semantic_refactoring/test_facade_planner.py",
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
TASK_SURFACES = {
    "import",
    "api",
    "binding",
    "serialization",
    "introspection",
    "cli",
    "plugin",
    "registration",
    "documentation",
    "patch",
}
TASK_MIGRATIONS = {
    "facade",
    "reexport",
    "wrapper",
    "deprecation",
    "cli",
    "plugin",
    "registry",
    "serialization",
    "introspection",
    "traceback",
    "patch_target",
}


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


def _obligation(
    kind: CompatibilityKind = CompatibilityKind.IMPORT_PATH,
    **overrides: Any,
) -> dict[str, Any]:
    fields = {
        "obligation_id": f"obl:{kind.value}:record",
        "consumer_id": "pkg.cli",
        "subject_id": "symbol:pkg.mod.Record",
        "subject_module": "pkg.mod",
        "kind": kind.value,
        "disposition": CompatibilityDisposition.PRESERVE.value,
        "required": True,
    }
    fields.update(overrides)
    return fields


def _inventory(
    obligations: tuple[dict[str, Any], ...] | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    if obligations is None:
        obligations = (_obligation(),)
    fields: dict[str, Any] = {
        "tree_id": TREE_ID,
        "obligations": list(obligations),
    }
    fields.update(overrides)
    return fields


def test_owned_paths_and_task_identity_are_exact() -> None:
    assert TASK_ID == "SPAR-018"
    assert GOAL_ID == "SPAR-G033"
    assert PROGRAM == "semantic-preserving-autonomous-remodularization-v1"
    assert COMPATIBILITY_FACADE_PLAN_INTERFACE == "CompatibilityFacadePlan@1"
    assert CONSUMER_MIGRATION_PLAN_INTERFACE == "ConsumerMigrationPlan@1"
    assert SUBJECT_FACADE_PLAN_INTERFACE == "SubjectFacadePlan@1"
    assert FACADE_PLANNING_RECEIPT_INTERFACE == "FacadePlanningReceipt@1"
    assert FACADE_PLAN_CONTRACT_VERSION == "1"
    assert ANALYZER_ID.endswith("facade_planner@1")
    assert MODULE_PATH.is_file()
    assert TEST_PATH.is_file()
    for relative in WRITE_SCOPE:
        assert (ROOT / relative).is_file()


def test_authority_flags_cannot_self_authorize() -> None:
    assert AUTHORITY == "partition orchestration"
    assert AUTHORITY_OWNER == "ipfs_accelerate_py"
    assert FACADE_CAN_AUTHORIZE_COMPLETION is False
    assert FACADE_CAN_AUTHORIZE_TRANSITION is False
    assert FACADE_CAN_CREATE_AUTHORITY is False
    assert FACADE_CAN_RETIRE_FACADE is False
    assert VECTOR_SIMILARITY_IS_AUTHORITY is False
    assert PROJECTION_CLUSTERING_IS_AUTHORITY is False
    assert MODEL_OUTPUT_IS_PROPOSAL_ONLY is True
    assert TEST_PASS_IS_NOT_COMPLETION is True
    assert MARKDOWN_IS_NOT_COMPLETION is True
    assert WORKER_SELF_APPROVAL is False
    assert DUCKLAKE_IS_AUTHORITY is False
    assert SOFT_SIGNALS_CANNOT_OVERRIDE_HARD_CONSTRAINTS is True
    assert PLAN_IS_NOMINATION_ONLY is True
    assert RAW_SOURCE_REQUIRED is True
    profile = facade_planner_cid_profile()
    assert profile["codec"] == "dag-json"
    assert "not universal meaning" in profile["rule"]


def test_closed_vocabularies_cover_plan_surfaces() -> None:
    assert DECLARED_CONSUMER_SURFACES == TASK_SURFACES
    assert DECLARED_MIGRATION_KINDS == TASK_MIGRATIONS
    assert DECLARED_COMPATIBILITY_KINDS == {kind.value for kind in CompatibilityKind}
    assert set(KIND_SURFACE) == set(CompatibilityKind)
    assert set(KIND_MIGRATION) == set(CompatibilityKind)
    assert {surface.value for surface in KIND_SURFACE.values()} == TASK_SURFACES
    assert DISPOSITIONED == {
        "preserve",
        "migrate",
        "facade",
        "explicit_incompatibility",
        "unsupported",
    }


def test_module_defines_predicted_symbols_not_capsule_family() -> None:
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    names = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
    assert "CompatibilityFacadePlan" in names
    assert "ConsumerMigrationPlan" in names
    assert "SubjectFacadePlan" in names
    assert "FacadePlanningReceipt" in names
    for capsule in CAPSULE_TYPES:
        assert capsule not in names
    assert_not_competing_capsule_family()
    exports = provider_free_exports()
    assert "CompatibilityFacadePlan" in exports
    assert "plan_compatibility_facades" in exports
    assert "compile_facade_plan_receipt" in exports


def test_protected_paths_are_not_owned_write_scope() -> None:
    owned = set(WRITE_SCOPE)
    for relative in PROTECTED_PATHS:
        assert relative not in owned
        assert (ROOT / relative).exists()


@pytest.mark.parametrize("kind", list(CompatibilityKind))
def test_every_declared_kind_receives_a_consumer_plan(kind: CompatibilityKind) -> None:
    plan = plan_compatibility_facades(
        _inventory((_obligation(kind),)),
        candidates=(_candidate(),),
    )
    assert len(plan.consumer_plans) == 1
    item = plan.consumer_plans[0]
    assert item.kind == kind.value
    assert item.surface == surface_for_kind(kind).value
    assert item.migration_kind == migration_kind_for(
        kind, CompatibilityDisposition.PRESERVE
    ).value
    assert item.consumer_id == "pkg.cli"
    assert plan.plan_is_nomination_only is True
    assert plan.can_authorize_completion is False
    assert plan.can_retire_facade is False


def test_preserve_import_nominates_reexport() -> None:
    plan = plan_compatibility_facades(
        _inventory((_obligation(CompatibilityKind.IMPORT_PATH),)),
        candidates=(_candidate(),),
    )
    item = plan.consumer_plans[0]
    assert item.migration_kind == MigrationKind.REEXPORT.value
    assert item.disposition == CompatibilityDisposition.PRESERVE.value
    assert item.surface == ConsumerSurface.IMPORT.value
    facade = plan.subject_facades[0]
    assert facade.facade_required is False
    assert facade.subject_module == "pkg.mod"


def test_migrate_signature_nominates_wrapper() -> None:
    plan = plan_compatibility_facades(
        _inventory(
            (
                _obligation(
                    CompatibilityKind.SIGNATURE,
                    disposition=CompatibilityDisposition.MIGRATE.value,
                ),
            )
        ),
        candidates=(_candidate(),),
    )
    assert plan.consumer_plans[0].migration_kind == MigrationKind.WRAPPER.value
    assert plan.consumer_plans[0].surface == ConsumerSurface.API.value


def test_facade_disposition_nominates_facade_and_keeps_original_module() -> None:
    plan = plan_compatibility_facades(
        _inventory(
            (
                _obligation(
                    CompatibilityKind.MODULE_ATTRIBUTE,
                    disposition=CompatibilityDisposition.FACADE.value,
                ),
            )
        ),
        candidates=(_candidate(),),
    )
    assert plan.consumer_plans[0].migration_kind == MigrationKind.FACADE.value
    assert plan.subject_facades[0].facade_required is True
    assert plan.subject_facades[0].can_retire_facade is False


def test_explicit_incompatibility_nominates_deprecation() -> None:
    plan = plan_compatibility_facades(
        _inventory(
            (
                _obligation(
                    CompatibilityKind.DOCUMENTATION,
                    disposition=CompatibilityDisposition.EXPLICIT_INCOMPATIBILITY.value,
                ),
            )
        ),
        candidates=(_candidate(),),
    )
    assert plan.consumer_plans[0].migration_kind == MigrationKind.DEPRECATION.value


def test_cli_plugin_registry_serialization_introspection_traceback_patch() -> None:
    kinds = (
        (CompatibilityKind.CLI, MigrationKind.CLI, ConsumerSurface.CLI),
        (CompatibilityKind.PLUGIN, MigrationKind.PLUGIN, ConsumerSurface.PLUGIN),
        (CompatibilityKind.REGISTRY, MigrationKind.REGISTRY, ConsumerSurface.REGISTRATION),
        (
            CompatibilityKind.SERIALIZATION,
            MigrationKind.SERIALIZATION,
            ConsumerSurface.SERIALIZATION,
        ),
        (
            CompatibilityKind.INTROSPECTION,
            MigrationKind.INTROSPECTION,
            ConsumerSurface.INTROSPECTION,
        ),
        (
            CompatibilityKind.TRACEBACK,
            MigrationKind.TRACEBACK,
            ConsumerSurface.INTROSPECTION,
        ),
        (
            CompatibilityKind.PATCH_TARGET,
            MigrationKind.PATCH_TARGET,
            ConsumerSurface.PATCH,
        ),
    )
    obligations = tuple(
        _obligation(
            kind,
            obligation_id=f"obl:{kind.value}",
            disposition=CompatibilityDisposition.MIGRATE.value,
        )
        for kind, _migration, _surface in kinds
    )
    plan = plan_compatibility_facades(
        _inventory(obligations),
        candidates=(_candidate(),),
    )
    by_kind = {item.kind: item for item in plan.consumer_plans}
    for kind, migration, surface in kinds:
        assert by_kind[kind.value].migration_kind == migration.value
        assert by_kind[kind.value].surface == surface.value
    assert set(plan.covered_migration_kinds) == {item[1].value for item in kinds}


def test_every_consumer_is_covered_exactly_once_per_obligation() -> None:
    obligations = (
        _obligation(
            CompatibilityKind.IMPORT_PATH,
            obligation_id="obl:import",
            consumer_id="pkg.cli",
        ),
        _obligation(
            CompatibilityKind.PATCH_TARGET,
            obligation_id="obl:patch",
            consumer_id="tests.unit.test_record",
        ),
    )
    plan = plan_compatibility_facades(
        _inventory(obligations),
        candidates=(_candidate(consumer_ids=("pkg.cli", "tests.unit.test_record")),),
    )
    consumers = {item.consumer_id for item in plan.consumer_plans}
    assert consumers == {"pkg.cli", "tests.unit.test_record"}
    roles = {item.consumer_id: item.role for item in plan.consumer_plans}
    assert roles["tests.unit.test_record"] == ConsumerRole.TEST.value
    assert plan.subject_facades[0].consumer_ids == (
        "pkg.cli",
        "tests.unit.test_record",
    )


def test_required_undispositioned_consumer_fails_closed() -> None:
    with pytest.raises(FacadePlannerError, match="typed terminal"):
        plan_compatibility_facades(
            _inventory(
                (
                    _obligation(
                        disposition=CompatibilityDisposition.UNDISPOSITIONED.value
                    ),
                )
            ),
            candidates=(_candidate(),),
        )


def test_unsupported_required_behavior_fails_closed() -> None:
    with pytest.raises(FacadePlannerError, match="typed terminal"):
        plan_compatibility_facades(
            _inventory(
                (
                    _obligation(
                        disposition=CompatibilityDisposition.UNSUPPORTED.value
                    ),
                )
            ),
            candidates=(_candidate(),),
        )


def test_optional_unsupported_nominates_deprecation() -> None:
    plan = plan_compatibility_facades(
        _inventory(
            (
                _obligation(
                    CompatibilityKind.DOCUMENTATION,
                    disposition=CompatibilityDisposition.UNSUPPORTED.value,
                    required=False,
                ),
            )
        ),
        candidates=(_candidate(),),
    )
    assert plan.consumer_plans[0].migration_kind == MigrationKind.DEPRECATION.value
    assert plan.consumer_plans[0].required is False


def test_optional_undispositioned_keeps_original_module_as_facade() -> None:
    plan = plan_compatibility_facades(
        _inventory(
            (
                _obligation(
                    required=False,
                    disposition=CompatibilityDisposition.UNDISPOSITIONED.value,
                ),
            )
        ),
        candidates=(_candidate(),),
    )
    assert plan.consumer_plans[0].migration_kind == MigrationKind.FACADE.value
    facade = plan.subject_facades[0]
    assert facade.facade_required is True
    assert facade.undispositioned_consumer_ids == ("pkg.cli",)
    assert facade.can_retire_facade is False


def test_spar014_candidate_binds_target_module() -> None:
    candidate = _candidate()
    plan = plan_compatibility_facades(
        _inventory((_obligation(),)),
        candidates=(candidate,),
    )
    assert plan.consumer_plans[0].target_module_id == candidate.candidate_cid
    assert plan.subject_facades[0].target_module_id == candidate.candidate_cid
    assert plan.selected_candidate_cids == (candidate.candidate_cid,)
    assert plan.comparison_receipt_cid


def test_target_api_plan_mapping_binds_public_export_consumers() -> None:
    candidate = _candidate()
    module_id = candidate.candidate_cid
    target_api_plan = {
        "plan_cid": _cid("target-api-plan"),
        "modules": [
            {
                "module_id": module_id,
                "member_ids": ["node:leaf"],
                "public_exports": [
                    {
                        "member_id": "node:leaf",
                        "consumer_ids": ["pkg.cli"],
                    }
                ],
            }
        ],
    }
    plan = plan_compatibility_facades(
        _inventory((_obligation(),)),
        candidates=(candidate,),
        target_api_plan=target_api_plan,
    )
    assert plan.consumer_plans[0].target_module_id == module_id
    assert plan.target_api_plan_cid == _cid("target-api-plan")


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
    assert rejected.candidate_cid in comparison.rejected_candidate_cids
    plan = plan_compatibility_facades(
        _inventory((_obligation(),)),
        candidates=(admitted, rejected),
        comparison=comparison,
    )
    assert plan.selected_candidate_cids == (admitted.candidate_cid,)
    assert rejected.candidate_cid in plan.rejected_candidate_cids
    assert plan.negative_evidence_cids == plan.rejected_candidate_cids
    with pytest.raises(FacadePlannerError, match="not ranked by SPAR-014"):
        plan_compatibility_facades(
            _inventory((_obligation(),)),
            candidates=(admitted, rejected),
            comparison=comparison,
            selected_candidate_cids=(rejected.candidate_cid,),
        )


def test_advisory_and_vector_evidence_cannot_admit_a_facade() -> None:
    projection = _candidate(
        member_ids=("node:vector",),
        generator_kind=GeneratorKind.PROJECTION,
        admitted=False,
        advisory=True,
        consumer_ids=(),
        evidence_class="vector_candidate",
    )
    admitted = _candidate()
    plan = plan_compatibility_facades(
        _inventory((_obligation(),)),
        candidates=(admitted, projection),
    )
    assert projection.candidate_cid in plan.advisory_candidate_cids
    assert projection.candidate_cid not in plan.selected_candidate_cids
    with pytest.raises(FacadePlannerError, match="not ranked by SPAR-014"):
        plan_compatibility_facades(
            _inventory((_obligation(),)),
            candidates=(admitted, projection),
            selected_candidate_cids=(projection.candidate_cid,),
        )


def test_partition_consumers_must_appear_in_inventory() -> None:
    with pytest.raises(FacadePlannerError, match="missing from SPAR-011"):
        plan_compatibility_facades(
            _inventory((_obligation(consumer_id="pkg.cli"),)),
            candidates=(_candidate(consumer_ids=("pkg.cli", "pkg.missing")),),
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
    with pytest.raises(FacadePlannerError, match="selected_candidate_cids"):
        plan_compatibility_facades(
            _inventory((_obligation(),)),
            candidates=(scc, contract),
            comparison=comparison,
        )
    plan = plan_compatibility_facades(
        _inventory((_obligation(),)),
        candidates=(scc, contract),
        comparison=comparison,
        selected_candidate_cids=(contract.candidate_cid,),
    )
    assert plan.selected_candidate_cids == (contract.candidate_cid,)


def test_plan_round_trip_and_receipt_are_deterministic() -> None:
    first = plan_compatibility_facades(
        _inventory((_obligation(),)),
        candidates=(_candidate(),),
    )
    second = plan_compatibility_facades(
        _inventory((_obligation(),)),
        candidates=(_candidate(),),
    )
    assert first.plan_cid == second.plan_cid
    restored = decode_canonical_plan(encode_canonical_plan(first))
    assert restored == first
    assert restored.plan_cid == first.plan_cid
    receipt = compile_facade_plan_receipt(first)
    assert receipt.plan_cid == first.plan_cid
    assert receipt.can_authorize_completion is False
    assert receipt.can_retire_facade is False
    assert decode_canonical_receipt(encode_canonical_receipt(receipt)) == receipt


def test_identity_excludes_observational_fields() -> None:
    plan = plan_compatibility_facades(
        _inventory((_obligation(),)),
        candidates=(_candidate(),),
    )
    payload = plan.to_dict()
    assert not (IDENTITY_EXCLUDED_FIELDS & set(payload))
    dirty = dict(payload)
    dirty["timestamp"] = "now"
    with pytest.raises(FacadePlannerError, match="observational"):
        CompatibilityFacadePlan.from_dict(dirty)


def test_plan_cannot_claim_authority_flags() -> None:
    plan = plan_compatibility_facades(
        _inventory((_obligation(),)),
        candidates=(_candidate(),),
    )
    payload = plan.to_dict()
    payload["can_authorize_completion"] = True
    payload["plan_cid"] = cid_for_dag_json(
        {key: value for key, value in payload.items() if key != "plan_cid"}
    )
    with pytest.raises(FacadePlannerError, match="can_authorize_completion"):
        CompatibilityFacadePlan.from_dict(payload)
    payload = plan.to_dict()
    payload["plan_is_nomination_only"] = False
    payload["plan_cid"] = cid_for_dag_json(
        {key: value for key, value in payload.items() if key != "plan_cid"}
    )
    with pytest.raises(FacadePlannerError, match="nomination_only"):
        CompatibilityFacadePlan.from_dict(payload)
    payload = plan.to_dict()
    payload["can_retire_facade"] = True
    payload["plan_cid"] = cid_for_dag_json(
        {key: value for key, value in payload.items() if key != "plan_cid"}
    )
    with pytest.raises(FacadePlannerError, match="can_retire_facade"):
        CompatibilityFacadePlan.from_dict(payload)


def test_migration_kind_must_remain_canonical() -> None:
    with pytest.raises(FacadePlannerError, match="canonical"):
        ConsumerMigrationPlan(
            consumer_id="pkg.cli",
            obligation_id="obl:import",
            subject_id="symbol:pkg.mod.Record",
            subject_module="pkg.mod",
            kind=CompatibilityKind.IMPORT_PATH,
            disposition=CompatibilityDisposition.PRESERVE,
            migration_kind=MigrationKind.WRAPPER,
        )


def test_subject_facade_cannot_claim_retirement() -> None:
    facade = SubjectFacadePlan(
        subject_id="symbol:pkg.mod.Record",
        subject_module="pkg.mod",
        facade_required=False,
        consumer_ids=("pkg.cli",),
        migration_kinds=(MigrationKind.REEXPORT.value,),
    )
    payload = facade.to_dict()
    payload["can_retire_facade"] = True
    payload["facade_cid"] = cid_for_dag_json(
        {key: value for key, value in payload.items() if key != "facade_cid"}
    )
    with pytest.raises(FacadePlannerError, match="can_retire_facade"):
        SubjectFacadePlan.from_dict(payload)


def test_empty_inventory_fails_closed() -> None:
    with pytest.raises(FacadePlannerError, match="requires obligations"):
        plan_compatibility_facades(
            {"tree_id": TREE_ID, "obligations": []},
            candidates=(_candidate(),),
        )


def test_tree_mismatch_fails_closed() -> None:
    with pytest.raises(FacadePlannerError, match="tree_id"):
        plan_compatibility_facades(
            _inventory(tree_id=OTHER_TREE),
            candidates=(_candidate(),),
        )


def test_comparison_must_remain_spar014_analyzer() -> None:
    candidate = _candidate()
    comparison = compare_partition_candidates((candidate,))
    payload = comparison.to_dict()
    payload["analyzer_id"] = ANALYZER_ID
    identity = {key: value for key, value in payload.items() if key != "receipt_cid"}
    payload["receipt_cid"] = cid_for_dag_json(identity)
    with pytest.raises(Exception, match="SPAR-014 analyzer"):
        plan_compatibility_facades(
            _inventory((_obligation(),)),
            candidates=(candidate,),
            comparison=payload,
        )


def test_cut_edge_consumer_still_requires_inventory_coverage() -> None:
    candidate = _candidate(
        consumer_ids=("pkg.cli",),
        cut_edges=(
            PartitionCutEdge(
                source_id="pkg.cli",
                target_id="node:leaf",
                kind="calls",
                constraint_class="soft",
            ),
        ),
    )
    plan = plan_compatibility_facades(
        _inventory((_obligation(),)),
        candidates=(candidate,),
    )
    assert plan.consumer_plans[0].target_module_id == candidate.candidate_cid
