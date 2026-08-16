"""Focused DCR-061 planning-only repair DAG tests."""

from __future__ import annotations

from dataclasses import replace

from ipfs_accelerate_py.agent_supervisor.autonomous_repair.contracts import RepairAuthorityRoots
from ipfs_accelerate_py.agent_supervisor.autonomous_repair.operators.registry import (
    OperatorDescriptor,
    OperatorRegistry,
)
from ipfs_accelerate_py.agent_supervisor.planning.formal_plan_compiler import FormalPlanCompiler
from ipfs_accelerate_py.agent_supervisor.planning.formal_plan_validator import FormalPlanValidator
from ipfs_accelerate_py.agent_supervisor.planning.proof_carrying_repair_dag import (
    DoctorTransformBinding,
    ProofCarryingRepairPlan,
    RepairPlanDagDisposition,
    RepairPlanNode,
    RepairPlanNodeKind,
    compile_proof_carrying_repair_plan,
)


def _plan() -> ProofCarryingRepairPlan:
    descriptor = OperatorDescriptor.from_mapping(
        {
            "operator_id": "repair.exact",
            "kind": "replace_exact_bytes",
            "owner_root": "external/ipfs_accelerate",
            "write_scope": ["module.py", "gitlink"],
            "before_predicates": ["before-valid"],
            "after_predicates": ["after-valid"],
            "applicability_proofs": ["proof-required"],
            "input_schema": {
                "type": "object",
                "required": ["source_digest"],
                "properties": {"source_digest": "sha256"},
                "additional_properties": False,
            },
            "preview": {"kind": "metadata_only", "fields": ["source_digest"]},
            "inverse": {"kind": "restore_exact_before_bytes", "binding": "source_digest"},
            "validation_commands": [["python", "-m", "py_compile", "module.py"]],
        }
    )
    registry = OperatorRegistry(
        (descriptor,), reviewed_manifest={descriptor.operator_id: descriptor.descriptor_id}
    )
    registry_cid = registry.report()["registry_cid"]
    node = RepairPlanNode(
        node_id="repair-node",
        kind=RepairPlanNodeKind.REPAIR,
        owner_root="external/ipfs_accelerate",
        write_path="module.py",
        source_span="line-1-2",
        before_digest="sha256-before",
        after_predicate="after-valid",
        descriptor=descriptor,
        registry_cid=registry_cid,
        proof_cid="proof-cid",
        logic_gate_cid="logic-gate-cid",
        impact_cid="impact-cid",
        noninterference_cid="noninterference-cid",
        validation_argv=(("python", "-m", "py_compile", "module.py"),),
        inverse_cid="inverse-cid",
        rollback_cid="rollback-cid",
        resource_bounds=(("cpu", 1),),
    )
    return ProofCarryingRepairPlan(
        DoctorTransformBinding("dcr051-cid", "dcr052-cid", "doctor-cid"),
        RepairAuthorityRoots(
            "repo", "forest-cid", "tree-cid", "policy-cid", "plan-cid", "packet-cid"
        ),
        registry,
        registry_cid,
        (node,),
    )


def test_dcr061_canonical_plan_is_pending_and_never_authoritative() -> None:
    plan = _plan()
    result = compile_proof_carrying_repair_plan(
        plan, compiler=FormalPlanCompiler(), validator=FormalPlanValidator()
    )

    assert result.disposition is RepairPlanDagDisposition.INTEGRATION_PENDING
    assert result.reason_codes == ("integration_pending_dcr052_dcr060_dcr064_dcr070",)
    assert result.plan_cid == plan.content_id
    assert result.node_cids == (plan.nodes[0].content_id,)
    assert result.execution_authorized is False
    assert result.completion_authorized is False


def test_dcr061_rejects_unordered_duplicate_write_and_stale_registry() -> None:
    plan = _plan()
    duplicate = replace(plan.nodes[0], node_id="duplicate", resource_bounds=(("cpu", 2),))
    invalid = replace(plan, nodes=(plan.nodes[0], duplicate), pinned_registry_cid="stale-registry")
    result = compile_proof_carrying_repair_plan(
        invalid, compiler=FormalPlanCompiler(), validator=FormalPlanValidator()
    )

    assert result.disposition is RepairPlanDagDisposition.REJECTED
    assert "pinned_registry_cid_invalid_or_stale" in result.reason_codes
    assert "duplicate_or_unordered_overlapping_write" in result.reason_codes


def test_dcr061_rejects_premature_outer_pin() -> None:
    plan = _plan()
    pin = replace(
        plan.nodes[0],
        node_id="pin-node",
        kind=RepairPlanNodeKind.OUTER_GITLINK_PIN,
        write_path="gitlink",
    )
    result = compile_proof_carrying_repair_plan(
        replace(plan, nodes=(plan.nodes[0], pin)),
        compiler=FormalPlanCompiler(),
        validator=FormalPlanValidator(),
    )

    assert result.disposition is RepairPlanDagDisposition.REJECTED
    assert "premature_pin_requires_provider_commit_and_consumer_validation" in result.reason_codes
