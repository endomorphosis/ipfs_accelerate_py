"""Independent contract tests for SPAR-017 TargetModuleAPIPlan@1."""

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
from ipfs_accelerate_py.agent_supervisor.semantic_refactoring.target_api import (
    ADAPTER_KIND,
    ANALYZER_ID,
    AUTHORITY,
    AUTHORITY_OWNER,
    AdapterAuthority,
    DUCKLAKE_IS_AUTHORITY,
    EXISTING_ADAPTER_AUTHORITIES,
    ExportVisibility,
    GOAL_ID,
    IDENTITY_EXCLUDED_FIELDS,
    MARKDOWN_IS_NOT_COMPLETION,
    MODEL_OUTPUT_IS_PROPOSAL_ONLY,
    MODULE_DEPENDENCY_EDGE_INTERFACE,
    PLAN_IS_NOMINATION_ONLY,
    PROGRAM,
    PROJECTION_CLUSTERING_IS_AUTHORITY,
    PUBLIC_SURFACE_PROTOCOL,
    RAW_SOURCE_REQUIRED,
    RESPONSIBILITY_STATEMENT_INTERFACE,
    SOFT_SIGNALS_CANNOT_OVERRIDE_HARD_CONSTRAINTS,
    STATE_OWNER_INTERFACE_INTERFACE,
    STATE_OWNER_KIND,
    SYMBOL_KIND,
    TASK_ID,
    TARGET_ADAPTER_INTERFACE,
    TARGET_API_CAN_AUTHORIZE_COMPLETION,
    TARGET_API_CAN_AUTHORIZE_TRANSITION,
    TARGET_API_CAN_CREATE_AUTHORITY,
    TARGET_API_CONTRACT_VERSION,
    TARGET_API_SYNTHESIS_RECEIPT_INTERFACE,
    TARGET_EXPORT_INTERFACE,
    TARGET_MODULE_API_INTERFACE,
    TARGET_MODULE_API_PLAN_INTERFACE,
    TARGET_PROTOCOL_INTERFACE,
    TEST_PASS_IS_NOT_COMPLETION,
    VECTOR_SIMILARITY_IS_AUTHORITY,
    WORKER_SELF_APPROVAL,
    ModuleDependencyEdge,
    ResponsibilityStatement,
    StateOwnerInterface,
    TargetAPIError,
    TargetAPISynthesisReceipt,
    TargetAdapter,
    TargetExport,
    TargetModuleAPI,
    TargetModuleAPIPlan,
    TargetProtocol,
    assert_not_competing_capsule_family,
    compile_target_api_receipt,
    decode_canonical_plan,
    decode_canonical_receipt,
    encode_canonical_plan,
    encode_canonical_receipt,
    provider_free_exports,
    synthesize_target_module_apis,
    target_api_cid_profile,
)


ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    ROOT
    / "ipfs_accelerate_py"
    / "agent_supervisor"
    / "semantic_refactoring"
    / "target_api.py"
)
TEST_PATH = Path(__file__).resolve()
WRITE_SCOPE = (
    "ipfs_accelerate_py/agent_supervisor/semantic_refactoring/target_api.py",
    "test/api/semantic_refactoring/test_target_api.py",
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


def _cid(label: str) -> str:
    return cid_for_bytes(label.encode("utf-8"))


def _candidate(**overrides: Any) -> ProgramPartitionCandidate:
    fields: dict[str, Any] = {
        "tree_id": TREE_ID,
        "generator_kind": GeneratorKind.SCC,
        "member_ids": ("node:leaf",),
        "admitted": True,
    }
    fields.update(overrides)
    return ProgramPartitionCandidate(**fields)


def test_owned_paths_and_task_identity_are_exact() -> None:
    assert TASK_ID == "SPAR-017"
    assert GOAL_ID == "SPAR-G033"
    assert PROGRAM == "semantic-preserving-autonomous-remodularization-v1"
    assert TARGET_MODULE_API_PLAN_INTERFACE == "TargetModuleAPIPlan@1"
    assert TARGET_MODULE_API_INTERFACE == "TargetModuleAPI@1"
    assert TARGET_EXPORT_INTERFACE == "TargetExport@1"
    assert TARGET_PROTOCOL_INTERFACE == "TargetProtocol@1"
    assert TARGET_ADAPTER_INTERFACE == "TargetAdapter@1"
    assert STATE_OWNER_INTERFACE_INTERFACE == "StateOwnerInterface@1"
    assert RESPONSIBILITY_STATEMENT_INTERFACE == "ResponsibilityStatement@1"
    assert MODULE_DEPENDENCY_EDGE_INTERFACE == "ModuleDependencyEdge@1"
    assert TARGET_API_SYNTHESIS_RECEIPT_INTERFACE == "TargetAPISynthesisReceipt@1"
    assert TARGET_API_CONTRACT_VERSION == "1"
    assert ANALYZER_ID.endswith("target_api@1")
    assert MODULE_PATH.is_file()
    assert TEST_PATH.is_file()
    for relative in WRITE_SCOPE:
        assert (ROOT / relative).is_file()


def test_authority_flags_cannot_self_authorize() -> None:
    assert AUTHORITY == "partition orchestration"
    assert AUTHORITY_OWNER == "ipfs_accelerate_py"
    assert TARGET_API_CAN_AUTHORIZE_COMPLETION is False
    assert TARGET_API_CAN_AUTHORIZE_TRANSITION is False
    assert TARGET_API_CAN_CREATE_AUTHORITY is False
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
    profile = target_api_cid_profile()
    assert profile["codec"] == "dag-json"
    assert "not universal meaning" in profile["rule"]
    assert AdapterAuthority.SPAR_NARROW.value in EXISTING_ADAPTER_AUTHORITIES
    assert PUBLIC_SURFACE_PROTOCOL == "public_surface"
    assert ADAPTER_KIND == "adapter"
    assert STATE_OWNER_KIND == "state_owner_interface"
    assert SYMBOL_KIND == "symbol"


def test_module_defines_predicted_symbols_not_capsule_family() -> None:
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    names = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
    assert "TargetModuleAPIPlan" in names
    assert "TargetModuleAPI" in names
    assert "TargetAPISynthesisReceipt" in names
    for capsule in CAPSULE_TYPES:
        assert capsule not in names
    assert_not_competing_capsule_family()
    exports = provider_free_exports()
    assert "TargetModuleAPIPlan" in exports
    assert "synthesize_target_module_apis" in exports
    assert "compile_target_api_receipt" in exports


def test_protected_paths_are_not_owned_write_scope() -> None:
    owned = set(WRITE_SCOPE)
    for relative in PROTECTED_PATHS:
        assert relative not in owned
        assert (ROOT / relative).exists()


def test_internal_module_is_private_without_consumers_or_incoming_cuts() -> None:
    candidate = _candidate(member_ids=("node:leaf", "node:helper"))
    plan = synthesize_target_module_apis((candidate,))
    module = plan.modules[0]
    assert module.module_id == candidate.candidate_cid
    assert {item.member_id for item in module.private_exports} == {
        "node:helper",
        "node:leaf",
    }
    assert module.public_exports == ()
    assert module.protocols == ()
    assert module.can_authorize_completion is False
    assert plan.plan_is_nomination_only is True
    assert plan.comparison_receipt_cid
    assert plan.analyzer_id == ANALYZER_ID


def test_consumers_without_cuts_make_the_whole_surface_public() -> None:
    candidate = _candidate(
        member_ids=("node:leaf",),
        consumer_ids=("consumer:cli",),
    )
    plan = synthesize_target_module_apis((candidate,))
    module = plan.modules[0]
    assert len(module.public_exports) == 1
    assert module.public_exports[0].visibility == ExportVisibility.PUBLIC.value
    assert module.public_exports[0].consumer_ids == ("consumer:cli",)
    assert module.private_exports == ()
    assert module.protocols[0].kind == PUBLIC_SURFACE_PROTOCOL
    assert module.protocols[0].member_ids == ("node:leaf",)


def test_incoming_cut_targets_are_public_and_helpers_stay_private() -> None:
    public = _candidate(
        member_ids=("node:leaf", "node:helper"),
        cut_edges=(
            PartitionCutEdge(
                source_id="consumer:cli",
                target_id="node:leaf",
                kind="calls",
                constraint_class="soft",
            ),
        ),
        consumer_ids=("consumer:cli",),
    )
    plan = synthesize_target_module_apis((public,))
    module = plan.modules[0]
    assert [item.member_id for item in module.public_exports] == ["node:leaf"]
    assert [item.member_id for item in module.private_exports] == ["node:helper"]
    assert module.protocols[0].member_ids == ("node:leaf",)


def test_cycle_free_dependency_direction_between_disjoint_modules() -> None:
    leaf = _candidate(member_ids=("node:leaf",), generator_kind=GeneratorKind.SCC)
    owner = _candidate(
        member_ids=("node:owner",),
        generator_kind=GeneratorKind.STATE,
        cut_edges=(
            PartitionCutEdge(
                source_id="node:owner",
                target_id="node:leaf",
                kind="calls",
                constraint_class="soft",
            ),
        ),
    )
    plan = synthesize_target_module_apis((leaf, owner))
    assert set(plan.selected_candidate_cids) == {
        leaf.candidate_cid,
        owner.candidate_cid,
    }
    by_id = {item.module_id: item for item in plan.modules}
    assert by_id[owner.candidate_cid].depends_on_module_ids == (leaf.candidate_cid,)
    assert by_id[leaf.candidate_cid].depends_on_module_ids == ()
    assert len(plan.dependency_edges) == 1
    edge = plan.dependency_edges[0]
    assert edge.source_module_id == owner.candidate_cid
    assert edge.target_module_id == leaf.candidate_cid
    assert edge.kind == "calls"
    assert "node:owner" in edge.witness_member_ids
    assert plan.can_authorize_transition is False


def test_cyclic_dependency_fails_closed() -> None:
    left = _candidate(
        member_ids=("node:left",),
        generator_kind=GeneratorKind.SCC,
        cut_edges=(
            PartitionCutEdge(
                source_id="node:left",
                target_id="node:right",
                kind="calls",
                constraint_class="soft",
            ),
        ),
    )
    right = _candidate(
        member_ids=("node:right",),
        generator_kind=GeneratorKind.STATE,
        cut_edges=(
            PartitionCutEdge(
                source_id="node:right",
                target_id="node:left",
                kind="calls",
                constraint_class="soft",
            ),
        ),
    )
    with pytest.raises(TargetAPIError, match="cycle-free"):
        synthesize_target_module_apis((left, right))


def test_overlapping_ranked_candidates_require_explicit_selection() -> None:
    scc = _candidate(member_ids=("node:leaf", "node:helper"), generator_kind=GeneratorKind.SCC)
    contract = _candidate(
        member_ids=("node:leaf",),
        generator_kind=GeneratorKind.CONTRACT,
        consumer_ids=("consumer:cli",),
    )
    comparison = compare_partition_candidates((scc, contract))
    assert scc.candidate_cid in comparison.ranked_candidate_cids
    assert contract.candidate_cid in comparison.ranked_candidate_cids
    with pytest.raises(TargetAPIError, match="selected_candidate_cids"):
        synthesize_target_module_apis((scc, contract), comparison=comparison)
    plan = synthesize_target_module_apis(
        (scc, contract),
        comparison=comparison,
        selected_candidate_cids=(contract.candidate_cid,),
    )
    assert plan.selected_candidate_cids == (contract.candidate_cid,)
    assert plan.modules[0].member_ids == ("node:leaf",)


def test_rejected_candidates_remain_negative_evidence() -> None:
    admitted = _candidate(member_ids=("node:leaf",))
    rejected = _candidate(
        member_ids=("node:ghost",),
        generator_kind=GeneratorKind.GRAPH,
        admitted=False,
        hard_constraint_violations=(ConstraintClass.OVERSIZED_CYCLE.value,),
    )
    comparison = compare_partition_candidates((admitted, rejected))
    assert rejected.candidate_cid in comparison.rejected_candidate_cids
    plan = synthesize_target_module_apis(
        (admitted, rejected),
        comparison=comparison,
    )
    assert plan.selected_candidate_cids == (admitted.candidate_cid,)
    assert rejected.candidate_cid in plan.rejected_candidate_cids
    assert plan.negative_evidence_cids == plan.rejected_candidate_cids
    with pytest.raises(TargetAPIError, match="not ranked by SPAR-014"):
        synthesize_target_module_apis(
            (admitted, rejected),
            comparison=comparison,
            selected_candidate_cids=(rejected.candidate_cid,),
        )


def test_advisory_and_vector_evidence_cannot_admit_a_module() -> None:
    projection = _candidate(
        member_ids=("node:vector",),
        generator_kind=GeneratorKind.PROJECTION,
        admitted=False,
        advisory=True,
        evidence_class="vector_candidate",
    )
    admitted = _candidate(member_ids=("node:leaf",))
    plan = synthesize_target_module_apis((admitted, projection))
    assert projection.candidate_cid in plan.advisory_candidate_cids
    assert projection.candidate_cid not in plan.selected_candidate_cids
    with pytest.raises(TargetAPIError, match="not ranked by SPAR-014"):
        synthesize_target_module_apis(
            (admitted, projection),
            selected_candidate_cids=(projection.candidate_cid,),
        )


def test_state_owner_interface_is_emitted_and_split_fails_closed() -> None:
    owner = _candidate(
        member_ids=("node:owner",),
        generator_kind=GeneratorKind.STATE,
        state_owner_ids=("state:cache",),
    )
    plan = synthesize_target_module_apis((owner,))
    iface = plan.modules[0].state_owner_interfaces[0]
    assert iface.owner_id == "state:cache"
    assert iface.kind == STATE_OWNER_KIND
    assert iface.member_ids == ("node:owner",)
    assert plan.modules[0].responsibility.owned_state_owner_ids == ("state:cache",)
    other = _candidate(
        member_ids=("node:other",),
        generator_kind=GeneratorKind.SCC,
        state_owner_ids=("state:cache",),
    )
    with pytest.raises(TargetAPIError, match="state owner split"):
        synthesize_target_module_apis((owner, other))


def test_external_cut_nominates_existing_authority_adapter() -> None:
    candidate = _candidate(
        member_ids=("node:leaf",),
        cut_edges=(
            PartitionCutEdge(
                source_id="node:leaf",
                target_id="datasets:capsule",
                kind="uses",
                constraint_class="soft",
            ),
            PartitionCutEdge(
                source_id="node:leaf",
                target_id="vfs:root",
                kind="uses",
                constraint_class="soft",
            ),
        ),
    )
    plan = synthesize_target_module_apis((candidate,))
    module = plan.modules[0]
    authorities = {item.authority: item for item in module.adapters}
    assert set(authorities) == {
        AdapterAuthority.DATASETS_SEMANTIC.value,
        AdapterAuthority.KIT_VFS.value,
    }
    assert authorities[AdapterAuthority.DATASETS_SEMANTIC.value].can_create_authority is False
    assert module.external_dependency_ids == ("datasets:capsule", "vfs:root")
    assert all(item.kind == ADAPTER_KIND for item in module.adapters)


def test_responsibility_statement_is_canonical() -> None:
    candidate = _candidate(
        member_ids=("node:leaf",),
        state_owner_ids=("state:cache",),
    )
    plan = synthesize_target_module_apis((candidate,))
    statement = plan.modules[0].responsibility
    assert statement.module_id == candidate.candidate_cid
    assert "owns members node:leaf" in statement.statement
    assert "state owners state:cache" in statement.statement
    with pytest.raises(TargetAPIError, match="canonical"):
        ResponsibilityStatement(
            module_id=candidate.candidate_cid,
            owned_member_ids=("node:leaf",),
            owned_state_owner_ids=("state:cache",),
            statement="invented prose",
        )


def test_private_export_cannot_list_consumers() -> None:
    with pytest.raises(TargetAPIError, match="private export cannot list consumers"):
        TargetExport(
            member_id="node:leaf",
            visibility=ExportVisibility.PRIVATE,
            consumer_ids=("consumer:cli",),
        )


def test_reflexive_dependency_fails_closed() -> None:
    with pytest.raises(TargetAPIError, match="reflexive"):
        ModuleDependencyEdge(
            source_module_id=_cid("mod-a"),
            target_module_id=_cid("mod-a"),
            kind="calls",
        )


def test_plan_round_trip_and_receipt_are_deterministic() -> None:
    leaf = _candidate(member_ids=("node:leaf",))
    owner = _candidate(
        member_ids=("node:owner",),
        generator_kind=GeneratorKind.STATE,
        cut_edges=(
            PartitionCutEdge(
                source_id="node:owner",
                target_id="node:leaf",
                kind="imports",
                constraint_class="soft",
            ),
        ),
    )
    first = synthesize_target_module_apis((owner, leaf))
    second = synthesize_target_module_apis((leaf, owner))
    assert first.plan_cid == second.plan_cid
    restored = decode_canonical_plan(encode_canonical_plan(first))
    assert restored == first
    assert restored.plan_cid == first.plan_cid
    receipt = compile_target_api_receipt(first)
    assert receipt.plan_cid == first.plan_cid
    assert receipt.can_authorize_completion is False
    assert decode_canonical_receipt(encode_canonical_receipt(receipt)) == receipt


def test_identity_excludes_observational_fields() -> None:
    plan = synthesize_target_module_apis((_candidate(),))
    payload = plan.to_dict()
    assert not (IDENTITY_EXCLUDED_FIELDS & set(payload))
    dirty = dict(payload)
    dirty["timestamp"] = "now"
    with pytest.raises(TargetAPIError, match="observational"):
        TargetModuleAPIPlan.from_dict(dirty)


def test_plan_cannot_claim_authority_flags() -> None:
    plan = synthesize_target_module_apis((_candidate(),))
    payload = plan.to_dict()
    payload["can_authorize_completion"] = True
    payload["plan_cid"] = cid_for_dag_json(
        {key: value for key, value in payload.items() if key != "plan_cid"}
    )
    with pytest.raises(TargetAPIError, match="can_authorize_completion"):
        TargetModuleAPIPlan.from_dict(payload)
    payload = plan.to_dict()
    payload["plan_is_nomination_only"] = False
    payload["plan_cid"] = cid_for_dag_json(
        {key: value for key, value in payload.items() if key != "plan_cid"}
    )
    with pytest.raises(TargetAPIError, match="nomination_only"):
        TargetModuleAPIPlan.from_dict(payload)


def test_adapter_cannot_claim_authority() -> None:
    adapter = TargetAdapter(
        module_id=_cid("mod"),
        authority=AdapterAuthority.SPAR_NARROW,
        external_ids=("spar:boundary",),
    )
    payload = adapter.to_dict()
    payload["can_create_authority"] = True
    payload["adapter_cid"] = cid_for_dag_json(
        {key: value for key, value in payload.items() if key != "adapter_cid"}
    )
    with pytest.raises(TargetAPIError, match="can_create_authority"):
        TargetAdapter.from_dict(payload)


def test_empty_candidates_fail_closed() -> None:
    with pytest.raises(TargetAPIError, match="requires candidates"):
        synthesize_target_module_apis(())


def test_tree_mismatch_fails_closed() -> None:
    local = _candidate()
    stale = _candidate(
        member_ids=("node:other",),
        generator_kind=GeneratorKind.STATE,
        tree_id=OTHER_TREE,
    )
    with pytest.raises(TargetAPIError, match="tree_id"):
        synthesize_target_module_apis((local, stale))


def test_comparison_must_remain_spar014_analyzer() -> None:
    candidate = _candidate()
    comparison = compare_partition_candidates((candidate,))
    payload = comparison.to_dict()
    payload["analyzer_id"] = ANALYZER_ID
    identity = {key: value for key, value in payload.items() if key != "receipt_cid"}
    payload["receipt_cid"] = cid_for_dag_json(identity)
    with pytest.raises(Exception, match="SPAR-014 analyzer"):
        synthesize_target_module_apis((candidate,), comparison=payload)


def test_exports_must_cover_members_exactly_once() -> None:
    candidate = _candidate(member_ids=("node:leaf",))
    plan = synthesize_target_module_apis((candidate,))
    payload = plan.modules[0].to_dict()
    payload["private_exports"] = []
    payload.pop("api_cid")
    with pytest.raises(TargetAPIError, match="exactly once"):
        TargetModuleAPI(
            **{
                key: value
                for key, value in payload.items()
                if key
                not in {
                    "schema",
                    "interface",
                    "can_authorize_transition",
                    "can_authorize_completion",
                    "can_create_authority",
                    "projection_is_authority",
                }
            }
        )


def test_spar014_comparison_receipt_is_bound() -> None:
    candidate = _candidate()
    comparison = compare_partition_candidates((candidate,))
    assert comparison.analyzer_id == SPAR014_ANALYZER_ID
    plan = synthesize_target_module_apis((candidate,), comparison=comparison)
    assert plan.comparison_receipt_cid == comparison.receipt_cid
    receipt = compile_target_api_receipt(plan)
    assert receipt.selected_candidate_cids == (candidate.candidate_cid,)
    restored = TargetAPISynthesisReceipt.from_dict(receipt.to_dict())
    assert restored == receipt


def test_protocol_and_export_round_trip() -> None:
    export = TargetExport(
        member_id="node:leaf",
        visibility=ExportVisibility.PUBLIC,
        consumer_ids=("consumer:cli",),
    )
    assert TargetExport.from_dict(export.to_dict()) == export
    protocol = TargetProtocol(
        module_id=_cid("mod"),
        member_ids=("node:leaf",),
    )
    assert TargetProtocol.from_dict(protocol.to_dict()) == protocol
    owner = StateOwnerInterface(
        owner_id="state:cache",
        module_id=_cid("mod"),
        member_ids=("node:leaf",),
    )
    assert StateOwnerInterface.from_dict(owner.to_dict()) == owner
