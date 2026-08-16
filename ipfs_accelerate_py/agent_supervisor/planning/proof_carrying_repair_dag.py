"""DCR-061 canonical repair DAGs over the existing formal planner stack.

This module does not compile source, invoke providers, or replace
``FormalPlanCompiler`` / ``FormalPlanValidator``.  It validates a closed repair
projection which those existing components can consume only after the separate
live DCR gates are available.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final

from ..autonomous_repair.contracts import RepairAuthorityRoots, repair_evidence_cid
from ..autonomous_repair.operators.registry import OperatorDescriptor, OperatorRegistry
from .formal_plan_compiler import FormalPlanCompiler
from .formal_plan_validator import FormalPlanValidator


DCR_REPAIR_PLAN_DAG_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/proof-carrying-repair-plan-dag@1"
)
DCR_REPAIR_PLAN_NODE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/proof-carrying-repair-plan-node@1"
)


class RepairPlanNodeKind(str, Enum):
    REPAIR = "repair"
    PROVIDER_COMMIT = "provider_commit"
    CONSUMER_VALIDATION = "consumer_validation"
    OUTER_GITLINK_PIN = "outer_gitlink_pin"


class RepairPlanDagDisposition(str, Enum):
    INTEGRATION_PENDING = "integration_pending"
    REJECTED = "rejected"


def _text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field_name} must be non-empty exact text")
    if "synthetic" in value.lower() or "stub" in value.lower():
        raise ValueError(f"{field_name} may not be synthetic or stub")
    return value


@dataclass(frozen=True)
class DoctorTransformBinding:
    """Closed DCR-051/052 transform identities, never a callable transform."""

    dcr051_transform_cid: str
    dcr052_receipt_cid: str
    doctor_receipt_cid: str

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            object.__setattr__(self, name, _text(getattr(self, name), name))

    def to_dict(self) -> dict[str, str]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class RepairPlanNode:
    """One source-free, owner-scoped deterministic repair operation."""

    node_id: str
    kind: RepairPlanNodeKind
    owner_root: str
    write_path: str
    source_span: str
    before_digest: str
    after_predicate: str
    descriptor: OperatorDescriptor
    registry_cid: str
    proof_cid: str
    logic_gate_cid: str
    impact_cid: str
    noninterference_cid: str
    validation_argv: tuple[tuple[str, ...], ...]
    inverse_cid: str
    rollback_cid: str
    dependencies: tuple[str, ...] = ()
    resource_bounds: tuple[tuple[str, int], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.kind, RepairPlanNodeKind):
            raise ValueError("node kind must be closed")
        for name in (
            "node_id", "owner_root", "write_path", "source_span", "before_digest",
            "after_predicate", "registry_cid", "proof_cid", "logic_gate_cid", "impact_cid",
            "noninterference_cid", "inverse_cid", "rollback_cid",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        if self.write_path.startswith("/") or ".." in self.write_path.split("/"):
            raise ValueError("write_path must be relative and non-escaping")
        if not isinstance(self.descriptor, OperatorDescriptor):
            raise ValueError("node requires typed DCR-040 operator descriptor")
        commands = tuple(tuple(command) for command in self.validation_argv)
        if not commands or any(
            not command
            or any(not isinstance(arg, str) or not arg for arg in command)
            for command in commands
        ):
            raise ValueError("validation_argv must be non-empty structured argv")
        object.__setattr__(self, "validation_argv", commands)
        deps = tuple(_text(value, "dependency") for value in self.dependencies)
        if len(set(deps)) != len(deps) or self.node_id in deps:
            raise ValueError("dependencies must be unique and cannot self-reference")
        object.__setattr__(self, "dependencies", tuple(sorted(deps)))
        bounds = tuple(sorted(self.resource_bounds))
        if any(
            not isinstance(key, str) or type(value) is not int or value <= 0
            for key, value in bounds
        ):
            raise ValueError("resource_bounds must be positive integer pairs")
        object.__setattr__(self, "resource_bounds", bounds)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": DCR_REPAIR_PLAN_NODE_SCHEMA,
            "node_id": self.node_id,
            "kind": self.kind.value,
            "owner_root": self.owner_root,
            "write_path": self.write_path,
            "source_span": self.source_span,
            "before_digest": self.before_digest,
            "after_predicate": self.after_predicate,
            "descriptor_id": self.descriptor.descriptor_id,
            "registry_cid": self.registry_cid,
            "proof_cid": self.proof_cid,
            "logic_gate_cid": self.logic_gate_cid,
            "impact_cid": self.impact_cid,
            "noninterference_cid": self.noninterference_cid,
            "validation_argv": [list(command) for command in self.validation_argv],
            "inverse_cid": self.inverse_cid,
            "rollback_cid": self.rollback_cid,
            "dependencies": list(self.dependencies),
            "resource_bounds": [[key, value] for key, value in self.resource_bounds],
        }

    @property
    def content_id(self) -> str:
        return repair_evidence_cid(self.to_dict())


@dataclass(frozen=True)
class ProofCarryingRepairPlan:
    transform: DoctorTransformBinding
    authority_roots: RepairAuthorityRoots
    registry: OperatorRegistry
    pinned_registry_cid: str
    nodes: tuple[RepairPlanNode, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.transform, DoctorTransformBinding):
            raise ValueError("typed DCR-051/052 transform binding is required")
        if not isinstance(self.authority_roots, RepairAuthorityRoots):
            raise ValueError("typed DCR-003 authority roots are required")
        if not isinstance(self.registry, OperatorRegistry):
            raise ValueError("typed DCR-040 operator registry is required")
        object.__setattr__(
            self,
            "pinned_registry_cid",
            _text(self.pinned_registry_cid, "pinned_registry_cid"),
        )
        nodes = tuple(self.nodes)
        if not nodes or any(not isinstance(node, RepairPlanNode) for node in nodes):
            raise ValueError("plan must contain typed repair nodes")
        if len({node.node_id for node in nodes}) != len(nodes):
            raise ValueError("plan node ids must be unique")
        object.__setattr__(self, "nodes", tuple(sorted(nodes, key=lambda node: node.node_id)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": DCR_REPAIR_PLAN_DAG_SCHEMA,
            "transform": self.transform.to_dict(),
            "authority_roots": self.authority_roots.to_dict(),
            "pinned_registry_cid": self.pinned_registry_cid,
            "nodes": [node.to_dict() for node in self.nodes],
        }

    @property
    def content_id(self) -> str:
        return repair_evidence_cid(self.to_dict())


@dataclass(frozen=True)
class RepairPlanDagResult:
    disposition: RepairPlanDagDisposition
    reason_codes: tuple[str, ...]
    plan_cid: str = ""
    node_cids: tuple[str, ...] = ()
    execution_authorized: bool = False
    completion_authorized: bool = False
    model_call_count: int = 0
    provider_call_count: int = 0
    network_call_count: int = 0


def _ordered(before: str, after: str, nodes: Mapping[str, RepairPlanNode]) -> bool:
    """Whether an explicit dependency path orders ``before`` before ``after``."""
    pending = list(nodes[after].dependencies)
    seen: set[str] = set()
    while pending:
        current = pending.pop()
        if current == before:
            return True
        if current not in seen:
            seen.add(current)
            pending.extend(nodes[current].dependencies)
    return False


def compile_proof_carrying_repair_plan(
    plan: Any,
    *,
    compiler: FormalPlanCompiler,
    validator: FormalPlanValidator,
) -> RepairPlanDagResult:
    """Validate a deterministic DAG while retaining the existing formal stack.

    The compiler and validator must be actual reviewed instances; DCR-061 does
    not call them on source-free node metadata and does not replace either.
    """
    reasons: list[str] = []
    if not isinstance(compiler, FormalPlanCompiler):
        reasons.append("typed_formal_plan_compiler_required")
    if not isinstance(validator, FormalPlanValidator):
        reasons.append("typed_formal_plan_validator_required")
    if not isinstance(plan, ProofCarryingRepairPlan):
        reasons.append("typed_proof_carrying_repair_plan_required")
    if reasons:
        return RepairPlanDagResult(RepairPlanDagDisposition.REJECTED, tuple(sorted(reasons)))
    report = plan.registry.report()
    body = dict(report)
    actual_registry_cid = body.pop("registry_cid", "")
    if (
        actual_registry_cid != repair_evidence_cid(body)
        or actual_registry_cid != plan.pinned_registry_cid
    ):
        reasons.append("pinned_registry_cid_invalid_or_stale")
    descriptors = {item.descriptor_id: item for item in plan.registry.enumerate()}
    nodes = {node.node_id: node for node in plan.nodes}
    for node in plan.nodes:
        if (
            node.descriptor.descriptor_id not in descriptors
            or node.registry_cid != plan.pinned_registry_cid
        ):
            reasons.append("operator_descriptor_or_registry_binding_invalid")
        if node.descriptor.owner_root != node.owner_root:
            reasons.append("exactly_one_owner_root_required")
        if node.write_path not in node.descriptor.write_scope:
            reasons.append("write_path_outside_descriptor_scope")
        for dep in node.dependencies:
            if dep not in nodes:
                reasons.append("unknown_dependency")
    for node in plan.nodes:
        for other in plan.nodes:
            if node.node_id >= other.node_id:
                continue
            same_path = node.write_path == other.write_path
            overlap = (
                node.write_path.startswith(other.write_path + "/")
                or other.write_path.startswith(node.write_path + "/")
            )
            ordered = _ordered(node.node_id, other.node_id, nodes) or _ordered(
                other.node_id, node.node_id, nodes
            )
            if (same_path or overlap) and not ordered:
                reasons.append("duplicate_or_unordered_overlapping_write")
    # cycle detection is independent of source bodies and deterministic.
    for node_id in nodes:
        if _ordered(node_id, node_id, nodes):
            reasons.append("cycle_detected")
    pin_nodes = [node for node in plan.nodes if node.kind is RepairPlanNodeKind.OUTER_GITLINK_PIN]
    for pin in pin_nodes:
        ancestors = {node_id for node_id in nodes if _ordered(node_id, pin.node_id, nodes)}
        has_provider = any(
            nodes[node_id].kind is RepairPlanNodeKind.PROVIDER_COMMIT
            for node_id in ancestors
        )
        has_consumer_validation = any(
            nodes[node_id].kind is RepairPlanNodeKind.CONSUMER_VALIDATION
            for node_id in ancestors
        )
        if not has_provider or not has_consumer_validation:
            reasons.append("premature_pin_requires_provider_commit_and_consumer_validation")
    if reasons:
        return RepairPlanDagResult(
            RepairPlanDagDisposition.REJECTED,
            tuple(sorted(set(reasons))),
            plan_cid=plan.content_id,
        )
    return RepairPlanDagResult(
        RepairPlanDagDisposition.INTEGRATION_PENDING,
        ("integration_pending_dcr052_dcr060_dcr064_dcr070",),
        plan_cid=plan.content_id,
        node_cids=tuple(node.content_id for node in plan.nodes),
    )


__all__ = [
    "DCR_REPAIR_PLAN_DAG_SCHEMA",
    "DCR_REPAIR_PLAN_NODE_SCHEMA",
    "DoctorTransformBinding",
    "ProofCarryingRepairPlan",
    "RepairPlanDagDisposition",
    "RepairPlanDagResult",
    "RepairPlanNode",
    "RepairPlanNodeKind",
    "compile_proof_carrying_repair_plan",
]
