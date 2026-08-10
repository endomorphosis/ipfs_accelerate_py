"""DCR-061: compile Doctor transforms into ownership-safe task DAGs.

Interfaces
----------
* ``RepairPlanNode@1`` — one executable plan node with full bindings.
* ``ProofCarryingRepairPlan@1`` — acyclic ownership-safe task DAG.

Normative rules (fail-closed / structurally unrepresentable)
------------------------------------------------------------
Every executable node must bind:

* evidence, operator, owner root, exact write set, validation, rollback,
  proof transition, resource class, and dependencies.

The following are **structurally unrepresentable** as admitted plans:

* missing required bindings
* dependency cycles
* cross-root write sets
* premature submodule pin updates (pin before owned validation)
* prose / freeform source-body nodes
* provider / model / LLM nodes

Construction raises :class:`RepairPlanDagError` (a
:class:`~..proof.formal_verification_contracts.ContractValidationError`)
rather than emitting a weakened plan.  Runtime model calls remain 0 and
authority is never granted by this compiler.
"""

from __future__ import annotations

import json
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, ClassVar, Final

from ..proof.formal_verification_contracts import (
    CanonicalContract,
    ContractValidationError,
    content_identity,
)


# ---------------------------------------------------------------------------
# Interfaces / evidence / schemas
# ---------------------------------------------------------------------------

REPAIR_PLAN_NODE_INTERFACE: Final[str] = "RepairPlanNode@1"
PROOF_CARRYING_REPAIR_PLAN_INTERFACE: Final[str] = "ProofCarryingRepairPlan@1"
DCR_PLAN_DAG_EVIDENCE: Final[str] = "dcr/plan-dag@1"
PROOF_CARRYING_REPAIR_DAG_VERSION: Final[int] = 1

REPAIR_PLAN_NODE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/repair-plan-node@1"
)
PROOF_CARRYING_REPAIR_PLAN_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/proof-carrying-repair-plan@1"
)
REPAIR_PLAN_DAG_COMPILATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/repair-plan-dag-compilation@1"
)
DEFAULT_PLAN_DAG_FIXTURES_REL: Final[str] = (
    "data/agent_supervisor/deterministic_contract_repair/plan-dag-fixtures.json"
)

# Closed executable node kinds.  Anything outside this set cannot construct.
class RepairPlanNodeKind(str, Enum):
    """Finite vocabulary of ownership-safe executable plan nodes."""

    EVIDENCE_BIND = "evidence_bind"
    RESOURCE_RESERVE = "resource_reserve"
    DEPENDENCY_GATE = "dependency_gate"
    OPERATOR_APPLY = "operator_apply"
    PROOF_TRANSITION = "proof_transition"
    VALIDATION = "validation"
    ROLLBACK = "rollback"
    PIN_UPDATE = "pin_update"


# Explicitly forbidden surface names — structural rejection vocabulary.
_FORBIDDEN_NODE_KINDS: Final[frozenset[str]] = frozenset(
    {
        "prose",
        "prose_node",
        "freeform",
        "freeform_edit",
        "source_body",
        "llm",
        "llm_call",
        "model",
        "model_call",
        "provider",
        "provider_call",
        "provider_model",
        "chat",
        "completion",
        "generate",
        "natural_language",
        "prompt",
    }
)

# DCR-003 root set with path-prefix ownership (structural; no live git required).
_ROOT_SPECS: Final[Mapping[str, Mapping[str, Any]]] = MappingProxyType(
    {
        "orchestration": {
            "role": "orchestration_only",
            "relative_path": ".",
            "pin_path": "",
            "writable": False,
        },
        "swissknife": {
            "role": "consumer",
            "relative_path": "swissknife",
            "pin_path": "swissknife",
            "writable": True,
        },
        "mcp-plus-plus": {
            "role": "consumer",
            "relative_path": "Mcp-Plus-Plus",
            "pin_path": "Mcp-Plus-Plus",
            "writable": True,
        },
        "ipfs-accelerate": {
            "role": "provider",
            "relative_path": "external/ipfs_accelerate",
            "pin_path": "external/ipfs_accelerate",
            "writable": True,
        },
        "ipfs-datasets": {
            "role": "provider",
            "relative_path": "external/ipfs_datasets",
            "pin_path": "external/ipfs_datasets",
            "writable": True,
        },
        "ipfs-kit": {
            "role": "provider",
            "relative_path": "external/ipfs_kit",
            "pin_path": "external/ipfs_kit",
            "writable": True,
        },
    }
)

_REQUIRED_NODE_BINDINGS: Final[tuple[str, ...]] = (
    "node_id",
    "kind",
    "operator_ref",
    "evidence_cid",
    "owner_root",
    "write_set",
    "before_hashes",
    "validation_ref",
    "rollback_ref",
    "proof_transition",
    "depends_on",
    "resource_class",
)

_PROSE_MARKERS: Final[tuple[str, ...]] = (
    "def ",
    "class ",
    "import ",
    "#!/",
    "function ",
    "private_key",
    "BEGIN ",
    "password=",
)

_MAX_NODES: Final[int] = 512
_MAX_PATH_BYTES: Final[int] = 1_024
_MAX_TEXT_BYTES: Final[int] = 4_096


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class RepairPlanDagError(ContractValidationError):
    """Malformed repair DAG input or closed-boundary violation."""


class RepairPlanDagRejectionReason(str, Enum):
    """Closed, audit-stable rejection codes for plan-DAG compilation."""

    MISSING_BINDING = "missing_binding"
    FORBIDDEN_NODE_KIND = "forbidden_node_kind"
    UNKNOWN_NODE_KIND = "unknown_node_kind"
    PROSE_NODE = "prose_node"
    PROVIDER_MODEL_NODE = "provider_model_node"
    CROSS_ROOT_WRITE = "cross_root_write"
    UNKNOWN_OWNER_ROOT = "unknown_owner_root"
    ORCHESTRATION_WRITE = "orchestration_write"
    DEPENDENCY_CYCLE = "dependency_cycle"
    UNKNOWN_DEPENDENCY = "unknown_dependency"
    DUPLICATE_NODE_ID = "duplicate_node_id"
    PREMATURE_PIN_UPDATE = "premature_pin_update"
    EMPTY_PLAN = "empty_plan"
    INVALID_PATH = "invalid_path"
    MISSING_TRANSFORM = "missing_transform"
    INVALID_RESOURCE = "invalid_resource"
    WRITE_SET_HASH_MISMATCH = "write_set_hash_mismatch"
    PIN_ORDER_VIOLATION = "pin_order_violation"


# ---------------------------------------------------------------------------
# Path / text helpers
# ---------------------------------------------------------------------------


def _text(value: Any, name: str, *, required: bool = True, limit: int = _MAX_TEXT_BYTES) -> str:
    if value is None:
        text = ""
    elif isinstance(value, str):
        text = value.strip()
    else:
        raise RepairPlanDagError(f"{name} must be a string")
    if required and not text:
        raise RepairPlanDagError(f"{RepairPlanDagRejectionReason.MISSING_BINDING.value}:{name}")
    if len(text.encode("utf-8")) > limit:
        raise RepairPlanDagError(f"{name} exceeds byte bound")
    return text


def _safe_relative_path(value: Any, *, field: str) -> str:
    text = _text(value, field, required=True, limit=_MAX_PATH_BYTES)
    normalized = text.replace("\\", "/")
    path = PurePosixPath(normalized)
    if (
        path.is_absolute()
        or ".." in path.parts
        or "\x00" in normalized
        or (path.parts and path.parts[0].endswith(":"))
    ):
        raise RepairPlanDagError(
            f"{RepairPlanDagRejectionReason.INVALID_PATH.value}:{field}"
        )
    return path.as_posix()


def _assert_body_free(*texts: str) -> None:
    for text in texts:
        if not text:
            continue
        lowered = text.lower()
        for marker in _PROSE_MARKERS:
            if marker.lower() in lowered and ("\n" in text or len(text) > 256):
                raise RepairPlanDagError(
                    f"{RepairPlanDagRejectionReason.PROSE_NODE.value}:body"
                )


def _ids(values: Any, *, field: str, required: bool = False) -> tuple[str, ...]:
    if values is None:
        items: Iterable[Any] = ()
    elif isinstance(values, str):
        items = (values,)
    elif isinstance(values, Sequence) and not isinstance(values, (bytes, bytearray)):
        items = values
    else:
        raise RepairPlanDagError(f"{field} must be a sequence of strings")
    out: list[str] = []
    for item in items:
        text = _text(item, field, required=True)
        if text not in out:
            out.append(text)
    if required and not out:
        raise RepairPlanDagError(
            f"{RepairPlanDagRejectionReason.MISSING_BINDING.value}:{field}"
        )
    return tuple(out)


def _hash_map(value: Any, *, field: str, write_set: Sequence[str]) -> Mapping[str, str]:
    if value is None:
        raw: Mapping[str, Any] = {}
    elif isinstance(value, Mapping):
        raw = value
    else:
        raise RepairPlanDagError(f"{field} must be a mapping")
    result: dict[str, str] = {}
    for key, digest in raw.items():
        path = _safe_relative_path(key, field=f"{field}.key")
        text = _text(digest, f"{field}[{path}]", required=True)
        _assert_body_free(text)
        result[path] = text
    # Every write-set path must have a before hash; hashes may not invent paths.
    missing = [path for path in write_set if path not in result]
    extras = [path for path in result if path not in set(write_set)]
    if missing or extras:
        raise RepairPlanDagError(
            f"{RepairPlanDagRejectionReason.WRITE_SET_HASH_MISMATCH.value}:{field}"
        )
    return MappingProxyType(dict(sorted(result.items())))


def _resolve_owner_for_path(path: str) -> str:
    """Return the unique DCR root that owns ``path`` (workspace-relative)."""

    normalized = _safe_relative_path(path, field="write_path")
    matches: list[str] = []
    for root_id, spec in _ROOT_SPECS.items():
        if root_id == "orchestration":
            continue
        prefix = str(spec["relative_path"]).rstrip("/")
        if normalized == prefix or normalized.startswith(prefix + "/"):
            matches.append(root_id)
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise RepairPlanDagError(
            f"{RepairPlanDagRejectionReason.CROSS_ROOT_WRITE.value}:ambiguous"
        )
    # Pin paths live at the declared pin_path under the parent/orchestration tree.
    for root_id, spec in _ROOT_SPECS.items():
        pin = str(spec.get("pin_path") or "")
        if pin and (normalized == pin or normalized.startswith(pin + "/")):
            return "orchestration"
    raise RepairPlanDagError(
        f"{RepairPlanDagRejectionReason.CROSS_ROOT_WRITE.value}:unowned:{normalized}"
    )


def _normalize_kind(value: Any) -> RepairPlanNodeKind:
    raw = _text(getattr(value, "value", value), "kind", required=True)
    lowered = raw.lower().replace("-", "_").replace(" ", "_")
    if lowered in _FORBIDDEN_NODE_KINDS:
        if lowered in {
            "provider",
            "provider_call",
            "provider_model",
            "model",
            "model_call",
            "llm",
            "llm_call",
            "chat",
            "completion",
            "generate",
            "prompt",
        }:
            raise RepairPlanDagError(
                f"{RepairPlanDagRejectionReason.PROVIDER_MODEL_NODE.value}:{lowered}"
            )
        raise RepairPlanDagError(
            f"{RepairPlanDagRejectionReason.PROSE_NODE.value}:{lowered}"
        )
    try:
        return RepairPlanNodeKind(lowered)
    except ValueError as exc:
        raise RepairPlanDagError(
            f"{RepairPlanDagRejectionReason.UNKNOWN_NODE_KIND.value}:{lowered}"
        ) from exc


def _operator_ref(value: Any) -> str:
    if isinstance(value, Mapping):
        for key in ("operator_id", "operator_ref", "id", "kind"):
            if value.get(key):
                return _text(value[key], "operator_ref", required=True)
        raise RepairPlanDagError(
            f"{RepairPlanDagRejectionReason.MISSING_BINDING.value}:operator_ref"
        )
    if hasattr(value, "operator_id"):
        return _text(getattr(value, "operator_id"), "operator_ref", required=True)
    return _text(value, "operator_ref", required=True)


def _known_owner(root_id: str) -> Mapping[str, Any]:
    if root_id not in _ROOT_SPECS:
        raise RepairPlanDagError(
            f"{RepairPlanDagRejectionReason.UNKNOWN_OWNER_ROOT.value}:{root_id}"
        )
    return _ROOT_SPECS[root_id]


# ---------------------------------------------------------------------------
# RepairPlanNode@1
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RepairPlanNode(CanonicalContract):
    """One fully-bound, ownership-safe executable plan node.

    Missing bindings, prose bodies, provider/model kinds, and multi-root write
    sets cannot construct a node — they raise :class:`RepairPlanDagError`.
    """

    SCHEMA: ClassVar[str] = REPAIR_PLAN_NODE_SCHEMA
    INTERFACE: ClassVar[str] = REPAIR_PLAN_NODE_INTERFACE

    node_id: str
    kind: RepairPlanNodeKind
    operator_ref: str
    evidence_cid: str
    owner_root: str
    write_set: tuple[str, ...]
    before_hashes: Mapping[str, str]
    validation_ref: str
    rollback_ref: str
    proof_transition: str
    depends_on: tuple[str, ...]
    resource_class: str
    target_root: str = ""
    pin_path: str = ""
    grants_write_authority: bool = False
    runtime_model_calls: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "node_id", _text(self.node_id, "node_id"))
        kind = _normalize_kind(self.kind)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "operator_ref", _operator_ref(self.operator_ref))
        object.__setattr__(
            self, "evidence_cid", _text(self.evidence_cid, "evidence_cid")
        )

        target = _text(self.target_root, "target_root", required=False)
        pin = _text(self.pin_path, "pin_path", required=False)

        # Pin updates are parent/orchestration gitlink advances.  Resolve target
        # and pin path before ownership checks so empty write sets can be filled.
        if kind is RepairPlanNodeKind.PIN_UPDATE:
            if not target:
                raise RepairPlanDagError(
                    f"{RepairPlanDagRejectionReason.MISSING_BINDING.value}:target_root"
                )
            target_spec = _known_owner(target)
            if target == "orchestration":
                raise RepairPlanDagError(
                    f"{RepairPlanDagRejectionReason.PREMATURE_PIN_UPDATE.value}:orchestration"
                )
            expected_pin = str(target_spec.get("pin_path") or "")
            if not pin:
                pin = expected_pin
            if not pin or pin != expected_pin:
                raise RepairPlanDagError(
                    f"{RepairPlanDagRejectionReason.PIN_ORDER_VIOLATION.value}:pin_path"
                )
            owner = "orchestration"
        else:
            owner = _text(self.owner_root, "owner_root")
        spec = _known_owner(owner)
        object.__setattr__(self, "owner_root", owner)
        object.__setattr__(self, "target_root", target)
        object.__setattr__(self, "pin_path", pin)

        paths = tuple(
            _safe_relative_path(path, field="write_set")
            for path in (self.write_set or ())
            if str(path).strip()
        )
        if kind is RepairPlanNodeKind.PIN_UPDATE:
            if paths and paths != (pin,):
                raise RepairPlanDagError(
                    f"{RepairPlanDagRejectionReason.CROSS_ROOT_WRITE.value}:pin_write_set"
                )
            paths = (pin,)
        elif not paths and kind is RepairPlanNodeKind.OPERATOR_APPLY:
            raise RepairPlanDagError(
                f"{RepairPlanDagRejectionReason.MISSING_BINDING.value}:write_set"
            )
        # Deduplicate while preserving sorted order for identity stability.
        paths = tuple(sorted(dict.fromkeys(paths)))
        object.__setattr__(self, "write_set", paths)

        # Ownership: every write path must resolve to exactly the declared owner.
        for path in paths:
            path_owner = _resolve_owner_for_path(path)
            if kind is RepairPlanNodeKind.PIN_UPDATE:
                # Pin path is the submodule gitlink owned by orchestration.
                if path != pin:
                    raise RepairPlanDagError(
                        f"{RepairPlanDagRejectionReason.CROSS_ROOT_WRITE.value}:{path}"
                    )
            elif path_owner != owner:
                raise RepairPlanDagError(
                    f"{RepairPlanDagRejectionReason.CROSS_ROOT_WRITE.value}:{path}"
                )
        if paths and owner == "orchestration" and kind is not RepairPlanNodeKind.PIN_UPDATE:
            raise RepairPlanDagError(
                f"{RepairPlanDagRejectionReason.ORCHESTRATION_WRITE.value}"
            )
        if (
            paths
            and not spec.get("writable")
            and kind is not RepairPlanNodeKind.PIN_UPDATE
        ):
            raise RepairPlanDagError(
                f"{RepairPlanDagRejectionReason.ORCHESTRATION_WRITE.value}:{owner}"
            )

        raw_hashes = self.before_hashes
        if kind is RepairPlanNodeKind.PIN_UPDATE and (
            not raw_hashes or pin not in dict(raw_hashes or {})
        ):
            pin_hash = content_identity(
                {"pin_path": pin, "target_root": target, "kind": "pin_update"}
            )
            raw_hashes = {pin: pin_hash}
        hashes = _hash_map(raw_hashes, field="before_hashes", write_set=paths)
        object.__setattr__(self, "before_hashes", hashes)

        object.__setattr__(
            self, "validation_ref", _text(self.validation_ref, "validation_ref")
        )
        object.__setattr__(
            self, "rollback_ref", _text(self.rollback_ref, "rollback_ref")
        )
        object.__setattr__(
            self, "proof_transition", _text(self.proof_transition, "proof_transition")
        )
        deps = _ids(self.depends_on, field="depends_on", required=False)
        if self.node_id in deps:
            raise RepairPlanDagError(
                f"{RepairPlanDagRejectionReason.DEPENDENCY_CYCLE.value}:self"
            )
        object.__setattr__(self, "depends_on", deps)
        object.__setattr__(
            self, "resource_class", _text(self.resource_class, "resource_class")
        )

        # Authority and model counters hard-fail closed.
        object.__setattr__(self, "grants_write_authority", False)
        object.__setattr__(self, "runtime_model_calls", 0)

        # Reject prose smuggled through binding strings.
        _assert_body_free(
            self.node_id,
            self.operator_ref,
            self.evidence_cid,
            self.validation_ref,
            self.rollback_ref,
            self.proof_transition,
            self.resource_class,
            *self.write_set,
            *self.before_hashes.values(),
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "interface": self.INTERFACE,
            "node_id": self.node_id,
            "kind": self.kind.value,
            "operator_ref": self.operator_ref,
            "evidence_cid": self.evidence_cid,
            "owner_root": self.owner_root,
            "write_set": list(self.write_set),
            "before_hashes": dict(self.before_hashes),
            "validation_ref": self.validation_ref,
            "rollback_ref": self.rollback_ref,
            "proof_transition": self.proof_transition,
            "depends_on": list(self.depends_on),
            "resource_class": self.resource_class,
            "target_root": self.target_root,
            "pin_path": self.pin_path,
            "grants_write_authority": False,
            "runtime_model_calls": 0,
            "evidence_id": DCR_PLAN_DAG_EVIDENCE,
            "version": PROOF_CARRYING_REPAIR_DAG_VERSION,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RepairPlanNode":
        if not isinstance(payload, Mapping):
            raise RepairPlanDagError("repair plan node must be an object")
        return cls(
            node_id=str(payload.get("node_id") or ""),
            kind=payload.get("kind") or "",
            operator_ref=payload.get("operator_ref")
            or payload.get("operator")
            or "",
            evidence_cid=str(payload.get("evidence_cid") or ""),
            owner_root=str(payload.get("owner_root") or ""),
            write_set=tuple(payload.get("write_set") or ()),
            before_hashes=payload.get("before_hashes") or {},
            validation_ref=str(payload.get("validation_ref") or ""),
            rollback_ref=str(payload.get("rollback_ref") or ""),
            proof_transition=str(payload.get("proof_transition") or ""),
            depends_on=tuple(payload.get("depends_on") or ()),
            resource_class=str(payload.get("resource_class") or ""),
            target_root=str(payload.get("target_root") or ""),
            pin_path=str(payload.get("pin_path") or ""),
        )

    @property
    def node_cid(self) -> str:
        return self.content_id


# ---------------------------------------------------------------------------
# ProofCarryingRepairPlan@1
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProofCarryingRepairPlan(CanonicalContract):
    """Acyclic ownership-safe task DAG with explicit pin ordering."""

    SCHEMA: ClassVar[str] = PROOF_CARRYING_REPAIR_PLAN_SCHEMA
    INTERFACE: ClassVar[str] = PROOF_CARRYING_REPAIR_PLAN_INTERFACE

    plan_id: str
    nodes: tuple[RepairPlanNode, ...]
    topological_order: tuple[str, ...]
    source_transform_cids: tuple[str, ...] = ()
    grants_write_authority: bool = False
    runtime_model_calls: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "plan_id", _text(self.plan_id, "plan_id"))
        nodes = tuple(self.nodes or ())
        if not nodes:
            raise RepairPlanDagError(
                f"{RepairPlanDagRejectionReason.EMPTY_PLAN.value}"
            )
        if len(nodes) > _MAX_NODES:
            raise RepairPlanDagError("repair plan exceeds node bound")
        normalized: list[RepairPlanNode] = []
        by_id: dict[str, RepairPlanNode] = {}
        for raw in nodes:
            node = (
                raw
                if isinstance(raw, RepairPlanNode)
                else RepairPlanNode.from_dict(raw)  # type: ignore[arg-type]
            )
            if node.node_id in by_id:
                raise RepairPlanDagError(
                    f"{RepairPlanDagRejectionReason.DUPLICATE_NODE_ID.value}:{node.node_id}"
                )
            by_id[node.node_id] = node
            normalized.append(node)
        object.__setattr__(self, "nodes", tuple(normalized))

        # Unknown dependencies and cycles are unrepresentable.
        for node in normalized:
            for dep in node.depends_on:
                if dep not in by_id:
                    raise RepairPlanDagError(
                        f"{RepairPlanDagRejectionReason.UNKNOWN_DEPENDENCY.value}:{dep}"
                    )
        order = _topological_order(by_id)
        object.__setattr__(self, "topological_order", order)

        # Premature pin updates: every pin_update must depend (transitively)
        # on a validation node for the same target root's operator_apply.
        index = {node_id: idx for idx, node_id in enumerate(order)}
        validation_by_root: dict[str, list[str]] = defaultdict(list)
        operator_by_root: dict[str, list[str]] = defaultdict(list)
        for node in normalized:
            if node.kind is RepairPlanNodeKind.VALIDATION:
                validation_by_root[node.owner_root].append(node.node_id)
            if node.kind is RepairPlanNodeKind.OPERATOR_APPLY:
                operator_by_root[node.owner_root].append(node.node_id)

        for node in normalized:
            if node.kind is not RepairPlanNodeKind.PIN_UPDATE:
                continue
            target = node.target_root
            owned_preds = set(validation_by_root.get(target, ())) | set(
                operator_by_root.get(target, ())
            )
            if not owned_preds:
                raise RepairPlanDagError(
                    f"{RepairPlanDagRejectionReason.PREMATURE_PIN_UPDATE.value}:"
                    f"no_owned_commit:{target}"
                )
            # Pin must list at least one owned validation/operator as a direct
            # dependency and appear strictly after those chosen predecessors.
            # Independent transforms on the same root may pin after their own
            # validation without waiting for sibling chains.
            direct = [dep for dep in node.depends_on if dep in owned_preds]
            if not direct:
                raise RepairPlanDagError(
                    f"{RepairPlanDagRejectionReason.PREMATURE_PIN_UPDATE.value}:"
                    f"missing_dep:{target}"
                )
            pin_idx = index[node.node_id]
            for pred in direct:
                if index[pred] >= pin_idx:
                    raise RepairPlanDagError(
                        f"{RepairPlanDagRejectionReason.PREMATURE_PIN_UPDATE.value}:"
                        f"order:{pred}"
                    )

        sources = _ids(self.source_transform_cids, field="source_transform_cids")
        object.__setattr__(self, "source_transform_cids", sources)
        object.__setattr__(self, "grants_write_authority", False)
        object.__setattr__(self, "runtime_model_calls", 0)

    def _payload(self) -> dict[str, Any]:
        return {
            "interface": self.INTERFACE,
            "plan_id": self.plan_id,
            "nodes": [node.to_dict() for node in self.nodes],
            "topological_order": list(self.topological_order),
            "source_transform_cids": list(self.source_transform_cids),
            "grants_write_authority": False,
            "runtime_model_calls": 0,
            "evidence_id": DCR_PLAN_DAG_EVIDENCE,
            "version": PROOF_CARRYING_REPAIR_DAG_VERSION,
            "node_cids": [node.node_cid for node in self.nodes],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProofCarryingRepairPlan":
        if not isinstance(payload, Mapping):
            raise RepairPlanDagError("repair plan must be an object")
        return cls(
            plan_id=str(payload.get("plan_id") or ""),
            nodes=tuple(payload.get("nodes") or ()),
            topological_order=tuple(payload.get("topological_order") or ()),
            source_transform_cids=tuple(payload.get("source_transform_cids") or ()),
        )

    def node_map(self) -> Mapping[str, RepairPlanNode]:
        return MappingProxyType({node.node_id: node for node in self.nodes})

    def evidence_subset(self) -> dict[str, Any]:
        """Project the DCR-061 evidence subset for each node."""

        return {
            "evidence_id": DCR_PLAN_DAG_EVIDENCE,
            "plan_id": self.plan_id,
            "plan_cid": self.content_id,
            "nodes": [
                {
                    "node_cid": node.node_cid,
                    "node_id": node.node_id,
                    "depends_on": list(node.depends_on),
                    "owner": node.owner_root,
                    "write_set": list(node.write_set),
                    "before_hashes": dict(node.before_hashes),
                    "validation": node.validation_ref,
                    "rollback": node.rollback_ref,
                    "proof_transition": node.proof_transition,
                    "kind": node.kind.value,
                    "operator_ref": node.operator_ref,
                    "resource_class": node.resource_class,
                }
                for node in self.nodes
            ],
            "topological_order": list(self.topological_order),
            "runtime_model_calls": 0,
            "grants_write_authority": False,
        }


def _topological_order(by_id: Mapping[str, RepairPlanNode]) -> tuple[str, ...]:
    indegree: dict[str, int] = {node_id: 0 for node_id in by_id}
    children: dict[str, list[str]] = defaultdict(list)
    for node_id, node in by_id.items():
        for dep in node.depends_on:
            children[dep].append(node_id)
            indegree[node_id] += 1
    # Deterministic Kahn: ready set sorted by node_id.
    ready = deque(sorted(node_id for node_id, deg in indegree.items() if deg == 0))
    order: list[str] = []
    while ready:
        current = ready.popleft()
        order.append(current)
        for child in sorted(children[current]):
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
        # Keep ready queue sorted for stability after appends.
        if len(ready) > 1:
            ready = deque(sorted(ready))
    if len(order) != len(by_id):
        raise RepairPlanDagError(
            f"{RepairPlanDagRejectionReason.DEPENDENCY_CYCLE.value}"
        )
    return tuple(order)


# ---------------------------------------------------------------------------
# Compilation result
# ---------------------------------------------------------------------------


class RepairPlanDagDisposition(str, Enum):
    COMPILED = "compiled"
    REJECTED = "rejected"


@dataclass(frozen=True)
class RepairPlanDagCompilation(CanonicalContract):
    """Result of compiling Doctor transforms into a proof-carrying repair DAG."""

    SCHEMA: ClassVar[str] = REPAIR_PLAN_DAG_COMPILATION_SCHEMA

    disposition: RepairPlanDagDisposition
    plan: ProofCarryingRepairPlan | None
    reason_codes: tuple[str, ...] = ()
    grants_write_authority: bool = False
    runtime_model_calls: int = 0

    def __post_init__(self) -> None:
        try:
            disposition = RepairPlanDagDisposition(
                str(getattr(self.disposition, "value", self.disposition))
            )
        except ValueError as exc:
            raise RepairPlanDagError(f"unsupported disposition: {self.disposition!r}") from exc
        object.__setattr__(self, "disposition", disposition)
        if disposition is RepairPlanDagDisposition.COMPILED:
            if not isinstance(self.plan, ProofCarryingRepairPlan):
                raise RepairPlanDagError("compiled result requires ProofCarryingRepairPlan")
        else:
            object.__setattr__(self, "plan", None)
        codes = _ids(self.reason_codes, field="reason_codes")
        if not codes:
            codes = (disposition.value,)
        object.__setattr__(self, "reason_codes", codes)
        object.__setattr__(self, "grants_write_authority", False)
        object.__setattr__(self, "runtime_model_calls", 0)

    def _payload(self) -> dict[str, Any]:
        return {
            "disposition": self.disposition.value,
            "plan": self.plan.to_dict() if self.plan is not None else None,
            "reason_codes": list(self.reason_codes),
            "grants_write_authority": False,
            "runtime_model_calls": 0,
            "evidence_id": DCR_PLAN_DAG_EVIDENCE,
            "version": PROOF_CARRYING_REPAIR_DAG_VERSION,
        }

    @property
    def ok(self) -> bool:
        return self.disposition is RepairPlanDagDisposition.COMPILED and self.plan is not None


# ---------------------------------------------------------------------------
# Transform → node expansion
# ---------------------------------------------------------------------------


def _mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        payload = to_dict()
        if isinstance(payload, Mapping):
            return dict(payload)
    raise RepairPlanDagError("transform record must be a mapping or expose to_dict()")


def _transform_fields(transform: Any) -> dict[str, Any]:
    """Normalize DoctorTransformProposal-like records into compile inputs."""

    raw = _mapping(transform)
    # Nested proposal under a receipt.
    if "proposal" in raw and isinstance(raw["proposal"], Mapping):
        proposal = dict(raw["proposal"])
    else:
        proposal = raw

    operator = proposal.get("operator") or proposal.get("operator_ref") or ""
    operator_ref = _operator_ref(operator)
    write_set = tuple(
        _safe_relative_path(path, field="write_paths")
        for path in (
            proposal.get("write_set")
            or proposal.get("write_paths")
            or ()
        )
        if str(path).strip()
    )
    if not write_set:
        raise RepairPlanDagError(
            f"{RepairPlanDagRejectionReason.MISSING_TRANSFORM.value}:write_set"
        )

    # Infer owner from write set — multi-root is unrepresentable.
    owners = {_resolve_owner_for_path(path) for path in write_set}
    if len(owners) != 1:
        raise RepairPlanDagError(
            f"{RepairPlanDagRejectionReason.CROSS_ROOT_WRITE.value}:"
            f"{','.join(sorted(owners))}"
        )
    owner_root = next(iter(owners))
    if owner_root == "orchestration":
        raise RepairPlanDagError(
            f"{RepairPlanDagRejectionReason.ORCHESTRATION_WRITE.value}"
        )

    before = proposal.get("before_hashes") or {}
    if not before:
        before = {
            path: content_identity({"path": path, "role": "before", "operator": operator_ref})
            for path in write_set
        }

    proposal_id = _text(
        proposal.get("proposal_id")
        or proposal.get("transform_id")
        or proposal.get("id")
        or content_identity(proposal),
        "proposal_id",
    )
    evidence_cid = _text(
        proposal.get("applicability_proof_cid")
        or proposal.get("evidence_cid")
        or proposal.get("finding_cid")
        or content_identity({"proposal_id": proposal_id, "role": "evidence"}),
        "evidence_cid",
    )
    validation_ref = _text(
        proposal.get("validation_ref")
        or f"validation:{proposal_id}",
        "validation_ref",
    )
    rollback_ref = _text(
        proposal.get("rollback_ref") or f"rollback:{proposal_id}",
        "rollback_ref",
    )
    proof_transition = _text(
        proposal.get("expected_proof_transition")
        or proposal.get("proof_transition")
        or f"proof:{proposal_id}->admitted",
        "proof_transition",
    )
    resource_class = _text(
        proposal.get("resource_class") or "cpu-proof-solver",
        "resource_class",
    )
    include_pin = bool(proposal.get("include_pin_update") or proposal.get("pin_update"))
    external_deps = _ids(
        proposal.get("depends_on") or proposal.get("dependencies") or (),
        field="depends_on",
    )
    return {
        "proposal_id": proposal_id,
        "operator_ref": operator_ref,
        "write_set": write_set,
        "owner_root": owner_root,
        "before_hashes": before,
        "evidence_cid": evidence_cid,
        "validation_ref": validation_ref,
        "rollback_ref": rollback_ref,
        "proof_transition": proof_transition,
        "resource_class": resource_class,
        "include_pin": include_pin,
        "external_deps": external_deps,
        "transform_cid": content_identity(proposal),
    }


def _expand_transform(fields: Mapping[str, Any]) -> list[RepairPlanNode]:
    """Expand one admitted transform into the closed per-transform node chain."""

    pid = fields["proposal_id"]
    owner = fields["owner_root"]
    operator_ref = fields["operator_ref"]
    evidence_cid = fields["evidence_cid"]
    write_set = fields["write_set"]
    before_hashes = fields["before_hashes"]
    validation_ref = fields["validation_ref"]
    rollback_ref = fields["rollback_ref"]
    proof_transition = fields["proof_transition"]
    resource_class = fields["resource_class"]
    external_deps = list(fields["external_deps"])

    def nid(suffix: str) -> str:
        return f"node:{pid}:{suffix}"

    evidence = RepairPlanNode(
        node_id=nid("evidence"),
        kind=RepairPlanNodeKind.EVIDENCE_BIND,
        operator_ref=operator_ref,
        evidence_cid=evidence_cid,
        owner_root=owner,
        write_set=(),
        before_hashes={},
        validation_ref=validation_ref,
        rollback_ref=rollback_ref,
        proof_transition=proof_transition,
        depends_on=tuple(external_deps),
        resource_class=resource_class,
    )
    resource = RepairPlanNode(
        node_id=nid("resource"),
        kind=RepairPlanNodeKind.RESOURCE_RESERVE,
        operator_ref=operator_ref,
        evidence_cid=evidence_cid,
        owner_root=owner,
        write_set=(),
        before_hashes={},
        validation_ref=validation_ref,
        rollback_ref=rollback_ref,
        proof_transition=proof_transition,
        depends_on=(evidence.node_id,),
        resource_class=resource_class,
    )
    gate = RepairPlanNode(
        node_id=nid("gate"),
        kind=RepairPlanNodeKind.DEPENDENCY_GATE,
        operator_ref=operator_ref,
        evidence_cid=evidence_cid,
        owner_root=owner,
        write_set=(),
        before_hashes={},
        validation_ref=validation_ref,
        rollback_ref=rollback_ref,
        proof_transition=proof_transition,
        depends_on=(resource.node_id,),
        resource_class=resource_class,
    )
    apply = RepairPlanNode(
        node_id=nid("apply"),
        kind=RepairPlanNodeKind.OPERATOR_APPLY,
        operator_ref=operator_ref,
        evidence_cid=evidence_cid,
        owner_root=owner,
        write_set=write_set,
        before_hashes=before_hashes,
        validation_ref=validation_ref,
        rollback_ref=rollback_ref,
        proof_transition=proof_transition,
        depends_on=(gate.node_id,),
        resource_class=resource_class,
    )
    proof = RepairPlanNode(
        node_id=nid("proof"),
        kind=RepairPlanNodeKind.PROOF_TRANSITION,
        operator_ref=operator_ref,
        evidence_cid=evidence_cid,
        owner_root=owner,
        write_set=(),
        before_hashes={},
        validation_ref=validation_ref,
        rollback_ref=rollback_ref,
        proof_transition=proof_transition,
        depends_on=(apply.node_id,),
        resource_class=resource_class,
    )
    validation = RepairPlanNode(
        node_id=nid("validation"),
        kind=RepairPlanNodeKind.VALIDATION,
        operator_ref=operator_ref,
        evidence_cid=evidence_cid,
        owner_root=owner,
        write_set=write_set,
        before_hashes=before_hashes,
        validation_ref=validation_ref,
        rollback_ref=rollback_ref,
        proof_transition=proof_transition,
        depends_on=(proof.node_id,),
        resource_class=resource_class,
    )
    rollback = RepairPlanNode(
        node_id=nid("rollback"),
        kind=RepairPlanNodeKind.ROLLBACK,
        operator_ref=operator_ref,
        evidence_cid=evidence_cid,
        owner_root=owner,
        write_set=write_set,
        before_hashes=before_hashes,
        validation_ref=validation_ref,
        rollback_ref=rollback_ref,
        proof_transition=proof_transition,
        depends_on=(validation.node_id,),
        resource_class=resource_class,
    )
    nodes = [evidence, resource, gate, apply, proof, validation, rollback]
    if fields["include_pin"]:
        pin = RepairPlanNode(
            node_id=nid("pin"),
            kind=RepairPlanNodeKind.PIN_UPDATE,
            operator_ref=operator_ref,
            evidence_cid=evidence_cid,
            owner_root="orchestration",
            write_set=(),
            before_hashes={},
            validation_ref=validation_ref,
            rollback_ref=rollback_ref,
            proof_transition=proof_transition,
            depends_on=(validation.node_id,),
            resource_class=resource_class,
            target_root=owner,
        )
        nodes.append(pin)
    return nodes


# ---------------------------------------------------------------------------
# Public compile / validate API
# ---------------------------------------------------------------------------


def compile_proof_carrying_repair_plan(
    transforms: Iterable[Any] | Mapping[str, Any] | None = None,
    *,
    plan_id: str = "",
    nodes: Iterable[Any] | None = None,
    raise_on_reject: bool = True,
) -> RepairPlanDagCompilation | ProofCarryingRepairPlan:
    """Compile Doctor transforms (or explicit nodes) into a repair DAG.

    Parameters
    ----------
    transforms:
        One or more DoctorTransformProposal-like records (or a mapping with a
        ``transforms`` / ``proposals`` list).
    nodes:
        Optional pre-built :class:`RepairPlanNode` records.  When supplied they
        are validated as a plan without transform expansion.
    raise_on_reject:
        When true (default), structural violations raise
        :class:`RepairPlanDagError` so they remain unrepresentable as admitted
        plans.  When false, a rejected compilation receipt is returned.
    """

    try:
        plan = _compile_plan(transforms=transforms, plan_id=plan_id, nodes=nodes)
        result = RepairPlanDagCompilation(
            disposition=RepairPlanDagDisposition.COMPILED,
            plan=plan,
            reason_codes=(
                "compiled",
                "ownership_safe",
                "acyclic",
                "bindings_complete",
                "runtime_model_calls_0",
            ),
        )
        return result
    except RepairPlanDagError as exc:
        if raise_on_reject:
            raise
        return RepairPlanDagCompilation(
            disposition=RepairPlanDagDisposition.REJECTED,
            plan=None,
            reason_codes=(str(exc),),
        )


def _compile_plan(
    *,
    transforms: Iterable[Any] | Mapping[str, Any] | None,
    plan_id: str,
    nodes: Iterable[Any] | None,
) -> ProofCarryingRepairPlan:
    explicit_nodes: list[RepairPlanNode] = []
    source_cids: list[str] = []

    if nodes is not None:
        for raw in nodes:
            if isinstance(raw, RepairPlanNode):
                explicit_nodes.append(raw)
            else:
                explicit_nodes.append(RepairPlanNode.from_dict(_mapping(raw)))

    transform_list: list[Any] = []
    if transforms is not None:
        if isinstance(transforms, Mapping) and not hasattr(transforms, "to_dict"):
            if any(
                key in transforms
                for key in ("transforms", "proposals", "nodes", "plan_id")
            ):
                plan_id = plan_id or str(transforms.get("plan_id") or "")
                if transforms.get("nodes") and not nodes:
                    for raw in transforms.get("nodes") or ():
                        if isinstance(raw, RepairPlanNode):
                            explicit_nodes.append(raw)
                        else:
                            explicit_nodes.append(RepairPlanNode.from_dict(_mapping(raw)))
                transform_list = list(
                    transforms.get("transforms")
                    or transforms.get("proposals")
                    or ()
                )
            else:
                # Single transform mapping.
                transform_list = [transforms]
        elif isinstance(transforms, (str, bytes)):
            raise RepairPlanDagError("transforms must be objects, not prose text")
        elif isinstance(transforms, Sequence) and not isinstance(
            transforms, (str, bytes, bytearray)
        ):
            transform_list = list(transforms)
        else:
            transform_list = [transforms]

    for transform in transform_list:
        fields = _transform_fields(transform)
        source_cids.append(fields["transform_cid"])
        explicit_nodes.extend(_expand_transform(fields))

    if not explicit_nodes:
        raise RepairPlanDagError(f"{RepairPlanDagRejectionReason.EMPTY_PLAN.value}")

    if not plan_id:
        plan_id = content_identity(
            {
                "nodes": [node.node_id for node in explicit_nodes],
                "sources": source_cids,
                "evidence": DCR_PLAN_DAG_EVIDENCE,
            }
        )

    return ProofCarryingRepairPlan(
        plan_id=plan_id if plan_id.startswith("plan:") else f"plan:{plan_id}",
        nodes=tuple(explicit_nodes),
        topological_order=(),  # recomputed in __post_init__
        source_transform_cids=tuple(source_cids),
    )


def validate_proof_carrying_repair_plan(
    plan: ProofCarryingRepairPlan | Mapping[str, Any] | RepairPlanDagCompilation,
) -> dict[str, Any]:
    """Validate a repair DAG; structural defects raise (unrepresentable).

    Returns a small receipt when the plan is admitted.  Re-runs the same
    constructor invariants so round-tripped dicts cannot smuggle forbidden
    structure past serialization.
    """

    if isinstance(plan, RepairPlanDagCompilation):
        if plan.plan is None:
            raise RepairPlanDagError(
                f"{RepairPlanDagRejectionReason.EMPTY_PLAN.value}"
            )
        plan = plan.plan
    if not isinstance(plan, ProofCarryingRepairPlan):
        plan = ProofCarryingRepairPlan.from_dict(_mapping(plan))

    # Re-canonicalize every node to catch payload drift.
    rebuilt = ProofCarryingRepairPlan(
        plan_id=plan.plan_id,
        nodes=tuple(
            RepairPlanNode.from_dict(node.to_dict()) for node in plan.nodes
        ),
        topological_order=(),
        source_transform_cids=plan.source_transform_cids,
    )
    if rebuilt.topological_order != plan.topological_order:
        # Topological order must be the unique deterministic order.
        if set(rebuilt.topological_order) != set(plan.topological_order):
            raise RepairPlanDagError(
                f"{RepairPlanDagRejectionReason.DEPENDENCY_CYCLE.value}:order"
            )

    # Forbidden-kind tripwire on serialized kind strings.
    for node in rebuilt.nodes:
        if node.kind.value in _FORBIDDEN_NODE_KINDS:
            raise RepairPlanDagError(
                f"{RepairPlanDagRejectionReason.FORBIDDEN_NODE_KIND.value}:"
                f"{node.kind.value}"
            )
        for field_name in _REQUIRED_NODE_BINDINGS:
            value = getattr(node, field_name)
            if value is None or value == "" or value == ():
                if field_name == "depends_on":
                    continue
                if field_name == "write_set" and node.kind in {
                    RepairPlanNodeKind.EVIDENCE_BIND,
                    RepairPlanNodeKind.RESOURCE_RESERVE,
                    RepairPlanNodeKind.DEPENDENCY_GATE,
                    RepairPlanNodeKind.PROOF_TRANSITION,
                }:
                    continue
                if field_name == "before_hashes" and not node.write_set:
                    continue
                raise RepairPlanDagError(
                    f"{RepairPlanDagRejectionReason.MISSING_BINDING.value}:{field_name}"
                )

    return {
        "ok": True,
        "interface": PROOF_CARRYING_REPAIR_PLAN_INTERFACE,
        "evidence_id": DCR_PLAN_DAG_EVIDENCE,
        "plan_id": rebuilt.plan_id,
        "plan_cid": rebuilt.content_id,
        "node_count": len(rebuilt.nodes),
        "node_cids": [node.node_cid for node in rebuilt.nodes],
        "topological_order": list(rebuilt.topological_order),
        "runtime_model_calls": 0,
        "grants_write_authority": False,
        "evidence_subset": rebuilt.evidence_subset(),
    }


def materialize_plan_dag_fixtures(
    *,
    destination: str | Path | None = None,
    transforms: Iterable[Any] | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Materialize plan-dag-fixtures.json evidence for DCR-061."""

    if transforms is None:
        transforms = (
            {
                "proposal_id": "fixture-accelerate-registration",
                "operator": {
                    "operator_id": "doctor-operator:add_registration@1",
                    "kind": "add_registration",
                },
                "write_paths": [
                    "external/ipfs_accelerate/ipfs_accelerate_py/agent_supervisor/planning/fixture_op.py"
                ],
                "before_hashes": {
                    "external/ipfs_accelerate/ipfs_accelerate_py/agent_supervisor/planning/fixture_op.py":
                        "sha256:fixture-before-1"
                },
                "applicability_proof_cid": "proof:fixture-applicability",
                "rollback_ref": "rollback:fixture-1",
                "expected_proof_transition": "proof:fixture->admitted",
                "resource_class": "cpu-proof-solver",
                "include_pin_update": True,
            },
        )
    compilation = compile_proof_carrying_repair_plan(
        transforms, plan_id="plan:dcr061-fixture"
    )
    assert isinstance(compilation, RepairPlanDagCompilation)
    assert compilation.plan is not None
    payload = {
        "artifact_schema": REPAIR_PLAN_DAG_COMPILATION_SCHEMA,
        "evidence_id": DCR_PLAN_DAG_EVIDENCE,
        "interface": PROOF_CARRYING_REPAIR_PLAN_INTERFACE,
        "node_interface": REPAIR_PLAN_NODE_INTERFACE,
        "version": PROOF_CARRYING_REPAIR_DAG_VERSION,
        "runtime_model_calls": 0,
        "grants_write_authority": False,
        "compilation": compilation.to_dict(),
        "evidence_subset": compilation.plan.evidence_subset(),
        "validation": validate_proof_carrying_repair_plan(compilation.plan),
    }
    if destination is None:
        root = Path(repo_root) if repo_root else Path.cwd()
        destination = root / DEFAULT_PLAN_DAG_FIXTURES_REL
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return payload


def is_structurally_representable(
    *,
    kind: str | None = None,
    missing_binding: str | None = None,
    write_set: Sequence[str] | None = None,
    owner_root: str | None = None,
    depends_on_cycle: bool = False,
    premature_pin: bool = False,
) -> bool:
    """Return False for every acceptance-forbidden structure (test helper)."""

    try:
        if kind is not None:
            _normalize_kind(kind)
        if missing_binding:
            return False
        if depends_on_cycle:
            return False
        if premature_pin:
            return False
        if write_set is not None and owner_root is not None:
            for path in write_set:
                if _resolve_owner_for_path(path) != owner_root:
                    return False
        return True
    except RepairPlanDagError:
        return False


__all__ = [
    "DCR_PLAN_DAG_EVIDENCE",
    "DEFAULT_PLAN_DAG_FIXTURES_REL",
    "PROOF_CARRYING_REPAIR_DAG_VERSION",
    "PROOF_CARRYING_REPAIR_PLAN_INTERFACE",
    "PROOF_CARRYING_REPAIR_PLAN_SCHEMA",
    "REPAIR_PLAN_DAG_COMPILATION_SCHEMA",
    "REPAIR_PLAN_NODE_INTERFACE",
    "REPAIR_PLAN_NODE_SCHEMA",
    "ProofCarryingRepairPlan",
    "RepairPlanDagCompilation",
    "RepairPlanDagDisposition",
    "RepairPlanDagError",
    "RepairPlanDagRejectionReason",
    "RepairPlanNode",
    "RepairPlanNodeKind",
    "compile_proof_carrying_repair_plan",
    "is_structurally_representable",
    "materialize_plan_dag_fixtures",
    "validate_proof_carrying_repair_plan",
]
