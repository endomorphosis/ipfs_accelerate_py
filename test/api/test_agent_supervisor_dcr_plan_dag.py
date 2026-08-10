"""DCR-061: compile Doctor transforms into ownership-safe task DAGs.

Acceptance:
* Missing bindings, cycles, cross-root writes, premature pin updates, prose
  nodes, or provider/model nodes are structurally unrepresentable.
* Every executable node binds evidence, operator, owner, write set, validation,
  rollback, proof transition, and dependencies.
* Submodule pin updates retain explicit order after owned validation.
* Runtime model calls remain 0; write authority is never granted by the plan.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.planning.formal_plan_compiler import (
    FormalPlanCompiler,
    compile_proof_carrying_repair_plan as compiler_entry,
)
from ipfs_accelerate_py.agent_supervisor.planning.formal_plan_validator import (
    FormalPlanValidator,
    validate_proof_carrying_repair_plan as validator_entry,
)
from ipfs_accelerate_py.agent_supervisor.planning.proof_carrying_repair_dag import (
    DCR_PLAN_DAG_EVIDENCE,
    PROOF_CARRYING_REPAIR_PLAN_INTERFACE,
    REPAIR_PLAN_NODE_INTERFACE,
    ProofCarryingRepairPlan,
    RepairPlanDagCompilation,
    RepairPlanDagDisposition,
    RepairPlanDagError,
    RepairPlanDagRejectionReason,
    RepairPlanNode,
    RepairPlanNodeKind,
    compile_proof_carrying_repair_plan,
    is_structurally_representable,
    materialize_plan_dag_fixtures,
    validate_proof_carrying_repair_plan,
)


ACCEL_PATH = (
    "external/ipfs_accelerate/ipfs_accelerate_py/agent_supervisor/planning/op.py"
)
DATASETS_PATH = "external/ipfs_datasets/ipfs_datasets_py/logic/op.py"
SWISS_PATH = "swissknife/src/tools/echo.ts"


def _transform(
    *,
    proposal_id: str = "t-accel-1",
    write_paths: tuple[str, ...] | None = None,
    include_pin: bool = False,
    operator_id: str = "doctor-operator:add_registration@1",
    before_hashes: dict[str, str] | None = None,
    **extra: object,
) -> dict[str, object]:
    paths = write_paths or (ACCEL_PATH,)
    hashes = before_hashes or {path: f"sha256:before-{i}" for i, path in enumerate(paths)}
    payload: dict[str, object] = {
        "proposal_id": proposal_id,
        "operator": {"operator_id": operator_id, "kind": "add_registration"},
        "write_paths": list(paths),
        "before_hashes": hashes,
        "applicability_proof_cid": f"proof:applicability:{proposal_id}",
        "rollback_ref": f"rollback:{proposal_id}",
        "expected_proof_transition": f"proof:{proposal_id}->admitted",
        "resource_class": "cpu-proof-solver",
        "include_pin_update": include_pin,
    }
    payload.update(extra)
    return payload


def _base_node(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "node_id": "node:n1",
        "kind": RepairPlanNodeKind.OPERATOR_APPLY.value,
        "operator_ref": "doctor-operator:add_registration@1",
        "evidence_cid": "evidence:e1",
        "owner_root": "ipfs-accelerate",
        "write_set": [ACCEL_PATH],
        "before_hashes": {ACCEL_PATH: "sha256:before-0"},
        "validation_ref": "validation:v1",
        "rollback_ref": "rollback:r1",
        "proof_transition": "proof:e1->admitted",
        "depends_on": [],
        "resource_class": "cpu-proof-solver",
    }
    payload.update(overrides)
    return payload


def test_interfaces_and_evidence_are_stable() -> None:
    assert REPAIR_PLAN_NODE_INTERFACE == "RepairPlanNode@1"
    assert PROOF_CARRYING_REPAIR_PLAN_INTERFACE == "ProofCarryingRepairPlan@1"
    assert DCR_PLAN_DAG_EVIDENCE == "dcr/plan-dag@1"
    assert set(RepairPlanNodeKind) == {
        RepairPlanNodeKind.EVIDENCE_BIND,
        RepairPlanNodeKind.RESOURCE_RESERVE,
        RepairPlanNodeKind.DEPENDENCY_GATE,
        RepairPlanNodeKind.OPERATOR_APPLY,
        RepairPlanNodeKind.PROOF_TRANSITION,
        RepairPlanNodeKind.VALIDATION,
        RepairPlanNodeKind.ROLLBACK,
        RepairPlanNodeKind.PIN_UPDATE,
    }


def test_compile_doctor_transform_into_ownership_safe_dag() -> None:
    compilation = compile_proof_carrying_repair_plan(
        _transform(include_pin=True),
        plan_id="plan:dcr061-happy",
    )
    assert isinstance(compilation, RepairPlanDagCompilation)
    assert compilation.disposition is RepairPlanDagDisposition.COMPILED
    assert compilation.ok is True
    assert compilation.runtime_model_calls == 0
    assert compilation.grants_write_authority is False

    plan = compilation.plan
    assert isinstance(plan, ProofCarryingRepairPlan)
    assert plan.plan_id == "plan:dcr061-happy"
    assert plan.runtime_model_calls == 0
    assert plan.grants_write_authority is False
    assert len(plan.nodes) == 8  # evidence..rollback + pin

    kinds = [node.kind for node in plan.nodes]
    assert RepairPlanNodeKind.OPERATOR_APPLY in kinds
    assert RepairPlanNodeKind.VALIDATION in kinds
    assert RepairPlanNodeKind.PIN_UPDATE in kinds
    assert RepairPlanNodeKind.ROLLBACK in kinds

    apply = next(n for n in plan.nodes if n.kind is RepairPlanNodeKind.OPERATOR_APPLY)
    assert apply.owner_root == "ipfs-accelerate"
    assert apply.write_set == (ACCEL_PATH,)
    assert apply.before_hashes[ACCEL_PATH]
    assert apply.validation_ref
    assert apply.rollback_ref
    assert apply.proof_transition
    assert apply.evidence_cid
    assert apply.operator_ref
    assert apply.resource_class == "cpu-proof-solver"
    assert apply.node_cid.startswith("b")

    pin = next(n for n in plan.nodes if n.kind is RepairPlanNodeKind.PIN_UPDATE)
    assert pin.owner_root == "orchestration"
    assert pin.target_root == "ipfs-accelerate"
    assert pin.pin_path == "external/ipfs_accelerate"
    # Pin appears after validation in topological order.
    order = list(plan.topological_order)
    validation_id = next(
        n.node_id for n in plan.nodes if n.kind is RepairPlanNodeKind.VALIDATION
    )
    assert order.index(validation_id) < order.index(pin.node_id)
    assert validation_id in pin.depends_on

    evidence = plan.evidence_subset()
    assert evidence["evidence_id"] == DCR_PLAN_DAG_EVIDENCE
    assert evidence["runtime_model_calls"] == 0
    for node_ev in evidence["nodes"]:
        assert node_ev["node_cid"]
        assert "depends_on" in node_ev
        assert node_ev["owner"]
        assert "write_set" in node_ev
        assert "before_hashes" in node_ev
        assert node_ev["validation"]
        assert node_ev["rollback"]
        assert node_ev["proof_transition"]


def test_formal_plan_compiler_and_validator_integrate() -> None:
    compiler = FormalPlanCompiler()
    compilation = compiler.compile_proof_carrying_repair_plan(
        [_transform(proposal_id="via-compiler")],
        plan_id="plan:via-compiler",
    )
    assert compilation.ok is True
    assert compilation.plan is not None

    # Module-level re-exports share the same implementation.
    via_entry = compiler_entry(_transform(proposal_id="via-entry"), plan_id="plan:via-entry")
    assert via_entry.ok is True

    receipt = FormalPlanValidator().validate_proof_carrying_repair_plan(compilation.plan)
    assert receipt["ok"] is True
    assert receipt["evidence_id"] == DCR_PLAN_DAG_EVIDENCE
    assert receipt["runtime_model_calls"] == 0
    assert len(receipt["node_cids"]) == len(compilation.plan.nodes)

    via_validator_entry = validator_entry(compilation.plan)
    assert via_validator_entry["ok"] is True

    # Round-trip through dict remains valid.
    rebuilt = validate_proof_carrying_repair_plan(compilation.plan.to_dict())
    assert rebuilt["ok"] is True


def test_missing_bindings_are_structurally_unrepresentable() -> None:
    required = (
        "node_id",
        "kind",
        "operator_ref",
        "evidence_cid",
        "owner_root",
        "validation_ref",
        "rollback_ref",
        "proof_transition",
        "resource_class",
    )
    for field in required:
        payload = _base_node()
        payload[field] = ""
        with pytest.raises(RepairPlanDagError, match="missing_binding"):
            RepairPlanNode.from_dict(payload)

    # Operator apply without write set.
    payload = _base_node(write_set=[], before_hashes={})
    with pytest.raises(RepairPlanDagError, match="missing_binding"):
        RepairPlanNode.from_dict(payload)

    assert is_structurally_representable(missing_binding="operator_ref") is False


def test_cycles_are_structurally_unrepresentable() -> None:
    a = RepairPlanNode.from_dict(
        _base_node(
            node_id="node:a",
            depends_on=["node:b"],
        )
    )
    b = RepairPlanNode.from_dict(
        _base_node(
            node_id="node:b",
            depends_on=["node:a"],
        )
    )
    with pytest.raises(RepairPlanDagError, match="dependency_cycle"):
        ProofCarryingRepairPlan(
            plan_id="plan:cycle",
            nodes=(a, b),
            topological_order=(),
        )

    # Self-cycle on a single node.
    with pytest.raises(RepairPlanDagError, match="dependency_cycle"):
        RepairPlanNode.from_dict(_base_node(depends_on=["node:n1"]))

    assert is_structurally_representable(depends_on_cycle=True) is False


def test_cross_root_writes_are_structurally_unrepresentable() -> None:
    with pytest.raises(RepairPlanDagError, match="cross_root_write"):
        RepairPlanNode.from_dict(
            _base_node(
                owner_root="ipfs-accelerate",
                write_set=[ACCEL_PATH, DATASETS_PATH],
                before_hashes={
                    ACCEL_PATH: "sha256:a",
                    DATASETS_PATH: "sha256:b",
                },
            )
        )

    with pytest.raises(RepairPlanDagError, match="cross_root_write"):
        compile_proof_carrying_repair_plan(
            _transform(
                write_paths=(ACCEL_PATH, SWISS_PATH),
                before_hashes={ACCEL_PATH: "sha256:a", SWISS_PATH: "sha256:b"},
            )
        )

    # Owner/path mismatch.
    with pytest.raises(RepairPlanDagError, match="cross_root_write"):
        RepairPlanNode.from_dict(
            _base_node(
                owner_root="swissknife",
                write_set=[ACCEL_PATH],
                before_hashes={ACCEL_PATH: "sha256:a"},
            )
        )


def test_premature_pin_updates_are_structurally_unrepresentable() -> None:
    # Pin with no owned validation/operator predecessor.
    pin_only = RepairPlanNode(
        node_id="node:pin-only",
        kind=RepairPlanNodeKind.PIN_UPDATE,
        operator_ref="doctor-operator:add_registration@1",
        evidence_cid="evidence:e1",
        owner_root="orchestration",
        write_set=(),
        before_hashes={},
        validation_ref="validation:v1",
        rollback_ref="rollback:r1",
        proof_transition="proof:e1->admitted",
        depends_on=(),
        resource_class="cpu-proof-solver",
        target_root="ipfs-accelerate",
    )
    with pytest.raises(RepairPlanDagError, match="premature_pin_update"):
        ProofCarryingRepairPlan(
            plan_id="plan:premature-pin",
            nodes=(pin_only,),
            topological_order=(),
        )

    # Pin that does not depend on the owned validation node.
    apply = RepairPlanNode.from_dict(_base_node(node_id="node:apply"))
    validation = RepairPlanNode.from_dict(
        _base_node(
            node_id="node:validation",
            kind=RepairPlanNodeKind.VALIDATION.value,
            depends_on=["node:apply"],
        )
    )
    pin = RepairPlanNode(
        node_id="node:pin",
        kind=RepairPlanNodeKind.PIN_UPDATE,
        operator_ref="doctor-operator:add_registration@1",
        evidence_cid="evidence:e1",
        owner_root="orchestration",
        write_set=(),
        before_hashes={},
        validation_ref="validation:v1",
        rollback_ref="rollback:r1",
        proof_transition="proof:e1->admitted",
        depends_on=(),  # missing validation dependency
        resource_class="cpu-proof-solver",
        target_root="ipfs-accelerate",
    )
    with pytest.raises(RepairPlanDagError, match="premature_pin_update"):
        ProofCarryingRepairPlan(
            plan_id="plan:pin-no-dep",
            nodes=(apply, validation, pin),
            topological_order=(),
        )

    assert is_structurally_representable(premature_pin=True) is False


def test_prose_and_provider_model_nodes_are_structurally_unrepresentable() -> None:
    for kind in (
        "prose",
        "prose_node",
        "freeform",
        "source_body",
        "natural_language",
    ):
        with pytest.raises(RepairPlanDagError, match="prose_node|forbidden"):
            RepairPlanNode.from_dict(_base_node(kind=kind))

    for kind in (
        "provider",
        "provider_model",
        "model",
        "model_call",
        "llm",
        "llm_call",
        "chat",
        "completion",
        "prompt",
    ):
        with pytest.raises(RepairPlanDagError, match="provider_model_node"):
            RepairPlanNode.from_dict(_base_node(kind=kind))

    # Prose body smuggled through a binding string.
    with pytest.raises(RepairPlanDagError, match="prose_node"):
        RepairPlanNode.from_dict(
            _base_node(
                proof_transition=(
                    "def repair():\n"
                    "    password=secret\n"
                    "    return open('x').read()\n"
                    + ("x" * 64)
                )
            )
        )

    # Transform compiler rejects prose text input.
    with pytest.raises(RepairPlanDagError):
        compile_proof_carrying_repair_plan("please call the model and fix it")


def test_pin_update_order_after_owned_validation() -> None:
    compilation = compile_proof_carrying_repair_plan(
        [
            _transform(proposal_id="owned-a", include_pin=True),
            _transform(
                proposal_id="owned-b",
                write_paths=(
                    "external/ipfs_accelerate/ipfs_accelerate_py/agent_supervisor/planning/other.py",
                ),
                before_hashes={
                    "external/ipfs_accelerate/ipfs_accelerate_py/agent_supervisor/planning/other.py":
                        "sha256:before-other"
                },
                include_pin=True,
            ),
        ],
        plan_id="plan:multi-pin",
    )
    plan = compilation.plan
    assert plan is not None
    order = list(plan.topological_order)
    for node in plan.nodes:
        if node.kind is RepairPlanNodeKind.PIN_UPDATE:
            for dep in node.depends_on:
                assert order.index(dep) < order.index(node.node_id)


def test_rejected_compilation_receipt_when_not_raising() -> None:
    result = compile_proof_carrying_repair_plan(
        _transform(write_paths=(ACCEL_PATH, DATASETS_PATH), before_hashes={
            ACCEL_PATH: "sha256:a",
            DATASETS_PATH: "sha256:b",
        }),
        raise_on_reject=False,
    )
    assert isinstance(result, RepairPlanDagCompilation)
    assert result.disposition is RepairPlanDagDisposition.REJECTED
    assert result.plan is None
    assert result.ok is False
    assert any("cross_root" in code for code in result.reason_codes)


def test_materialize_plan_dag_fixtures(tmp_path: Path) -> None:
    dest = tmp_path / "plan-dag-fixtures.json"
    payload = materialize_plan_dag_fixtures(destination=dest)
    assert dest.is_file()
    assert payload["evidence_id"] == DCR_PLAN_DAG_EVIDENCE
    assert payload["runtime_model_calls"] == 0
    assert payload["grants_write_authority"] is False
    assert payload["validation"]["ok"] is True
    assert payload["compilation"]["disposition"] == "compiled"


def test_rejection_reason_vocabulary_is_closed() -> None:
    codes = {item.value for item in RepairPlanDagRejectionReason}
    assert "missing_binding" in codes
    assert "dependency_cycle" in codes
    assert "cross_root_write" in codes
    assert "premature_pin_update" in codes
    assert "prose_node" in codes
    assert "provider_model_node" in codes
