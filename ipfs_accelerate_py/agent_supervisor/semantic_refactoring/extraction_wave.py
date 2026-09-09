"""SPAR-025 checkpointed transactional extraction waves.

This module extends current supervisor partition orchestration with
``ExtractionWave@1``.  It consumes SPAR-019 ``RefactorTransformationPacket@1``
mappings plus SPAR-020/021/022/023/024 nomination-only executors, then applies
one bounded packet at a time in an isolated fenced worktree with exact before
hashes, dependency order, rollback, VFS mutation receipts, and effect auditing.

The executor is nomination-only.  It does not write the repository, does not
replace kit VFS/CAS authority, does not apply undeclared CST transforms, and
cannot authorize a transition, completion, merge, or competing authority.
Failed waves restore exact preimages or discard the isolated worktree, retain
negative evidence, and release lease/fence identifiers.  Vector, model, and
heuristic evidence cannot admit a wave.  Observational metadata is excluded
from identity.  Dry-run is deterministic and never mutates.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import PurePosixPath
from typing import Any, ClassVar, Final, Mapping, Sequence
import unicodedata

from ipfs_accelerate_py.utils.cid_utils import cid_for_dag_json, validate_cid

from .binding_compatibility import (
    BindingCompatibilityError,
    execute_binding_compatibility_adapters,
)
from .codemod import (
    CodemodError,
    MemberLocator,
    apply_cst_extraction,
)
from .import_rewriter import (
    HANDLED_REWRITE_KINDS as SPAR021_REWRITE_KINDS,
    ImportRewriterError,
    execute_import_rewrites,
)
from .initialization_transform import (
    HANDLED_ADAPTER_KINDS as SPAR023_ADAPTER_KINDS,
    InitializationTransformError,
    execute_initialization_rewrites,
)
from .partition_generators import (
    IDENTITY_EXCLUDED_FIELDS as SPAR013_IDENTITY_EXCLUDED_FIELDS,
)
from .state_transform import (
    StateTransformError,
    execute_explicit_state_objects,
)
from .transformation_packet import (
    AdapterKind,
    DECLARED_ALLOWED_EFFECTS,
    DECLARED_FORBIDDEN_EFFECTS,
    EditKind,
    ROLLBACK_MODE as SPAR019_ROLLBACK_MODE,
    RefactorTransformationPacket,
    RewriteKind,
    TransformationPacketError,
)


TASK_ID: Final[str] = "SPAR-025"
GOAL_ID: Final[str] = "SPAR-G043"
PROGRAM: Final[str] = "semantic-preserving-autonomous-remodularization-v1"
AUTHORITY: Final[str] = "partition orchestration"
AUTHORITY_OWNER: Final[str] = "ipfs_accelerate_py"
ANALYZER_ID: Final[str] = (
    "ipfs_accelerate_py.agent_supervisor.semantic_refactoring.extraction_wave@1"
)
VFS_AUTHORITY_OWNER: Final[str] = "ipfs_kit_py"

EXTRACTION_WAVE_INTERFACE: Final[str] = "ExtractionWave@1"
EXTRACTION_WAVE_PLAN_INTERFACE: Final[str] = "ExtractionWavePlan@1"
EXTRACTION_WAVE_CHECKPOINT_INTERFACE: Final[str] = "ExtractionWaveCheckpoint@1"
VFS_MUTATION_RECEIPT_INTERFACE: Final[str] = "VfsMutationReceipt@1"
EFFECT_AUDIT_INTERFACE: Final[str] = "EffectAudit@1"
ROLLBACK_PLAN_INTERFACE: Final[str] = "RollbackPlan@1"
EXTRACTION_WAVE_RECEIPT_INTERFACE: Final[str] = "ExtractionWaveReceipt@1"

EXTRACTION_WAVE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/extraction-wave@1"
)
EXTRACTION_WAVE_PLAN_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/extraction-wave-plan@1"
)
EXTRACTION_WAVE_CHECKPOINT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/extraction-wave-checkpoint@1"
)
VFS_MUTATION_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/vfs-mutation-receipt@1"
)
EFFECT_AUDIT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/effect-audit@1"
)
ROLLBACK_PLAN_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/extraction-wave-rollback-plan@1"
)
EXTRACTION_WAVE_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/extraction-wave-receipt@1"
)

WAVE_CONTRACT_VERSION: Final[str] = "1"

WAVE_CAN_AUTHORIZE_TRANSITION: Final[bool] = False
WAVE_CAN_AUTHORIZE_COMPLETION: Final[bool] = False
WAVE_CAN_CREATE_AUTHORITY: Final[bool] = False
WAVE_CAN_RETIRE_FACADE: Final[bool] = False
WAVE_WRITES_REPOSITORY: Final[bool] = False
WAVE_OWNS_VFS: Final[bool] = False
VECTOR_SIMILARITY_IS_AUTHORITY: Final[bool] = False
PROJECTION_CLUSTERING_IS_AUTHORITY: Final[bool] = False
MODEL_OUTPUT_IS_PROPOSAL_ONLY: Final[bool] = True
TEST_PASS_IS_NOT_COMPLETION: Final[bool] = True
MARKDOWN_IS_NOT_COMPLETION: Final[bool] = True
WORKER_SELF_APPROVAL: Final[bool] = False
DUCKLAKE_IS_AUTHORITY: Final[bool] = False
SOFT_SIGNALS_CANNOT_OVERRIDE_HARD_CONSTRAINTS: Final[bool] = True
EXECUTOR_IS_NOMINATION_ONLY: Final[bool] = True
PLAN_IS_NOMINATION_ONLY: Final[bool] = True
RAW_SOURCE_REQUIRED: Final[bool] = True
DRY_RUN_IS_DETERMINISTIC: Final[bool] = True
DRY_RUN_MUTATES: Final[bool] = False
ONE_PACKET_AT_A_TIME: Final[bool] = True
FAILED_WAVES_RESTORE_PREIMAGES: Final[bool] = True
NEGATIVE_EVIDENCE_RETAINED: Final[bool] = True
KIT_OWNS_VFS: Final[bool] = True
USES_CURRENT_LEASE_FENCE: Final[bool] = True
USES_CURRENT_WORKTREE: Final[bool] = True

MAX_TEXT_CHARS: Final[int] = 16_384
MAX_MEMBERS: Final[int] = 16_384
MAX_PACKETS: Final[int] = 64
MAX_WRITE_PATHS: Final[int] = 64
MAX_PATH_CHARS: Final[int] = 1_024
MAX_RECEIPTS: Final[int] = 1_024
MAX_STEPS: Final[int] = 64

IDENTITY_EXCLUDED_FIELDS: Final[frozenset[str]] = SPAR013_IDENTITY_EXCLUDED_FIELDS
ROLLBACK_MODE: Final[str] = SPAR019_ROLLBACK_MODE

_FORBIDDEN_CAPSULE_TYPE_NAMES: Final[frozenset[str]] = frozenset(
    {
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
    }
)

_NON_ADMITTING_EVIDENCE: Final[frozenset[str]] = frozenset(
    {
        "vector_candidate",
        "model_hypothesis",
        "heuristic",
    }
)

_AUTHORITY_FLAG_NAMES: Final[tuple[str, ...]] = (
    "can_authorize_transition",
    "can_authorize_completion",
    "can_create_authority",
    "projection_is_authority",
)

_SPAR022_STATE_KINDS: Final[frozenset[str]] = frozenset(
    {
        AdapterKind.STATE.value,
        AdapterKind.BOUNDARY.value,
        AdapterKind.PROTOCOL.value,
    }
)
_SPAR024_REWRITE_KINDS: Final[frozenset[str]] = frozenset(
    {
        RewriteKind.SERIALIZATION.value,
        RewriteKind.INTROSPECTION.value,
        RewriteKind.TRACEBACK.value,
        RewriteKind.PATCH_TARGET.value,
        RewriteKind.DEPRECATION.value,
    }
)

_PREDECESSOR_ERRORS: Final[tuple[type[BaseException], ...]] = (
    CodemodError,
    ImportRewriterError,
    StateTransformError,
    InitializationTransformError,
    BindingCompatibilityError,
    TransformationPacketError,
)


class ExtractionWaveError(ValueError):
    """Fail-closed violation of a SPAR-025 extraction-wave contract."""

    def __init__(
        self,
        message: str,
        *,
        rollback: "RollbackPlan | None" = None,
        negative_evidence_cids: Sequence[str] = (),
    ) -> None:
        super().__init__(message)
        self.rollback = rollback
        self.negative_evidence_cids = tuple(negative_evidence_cids)


class WaveStatus(str, Enum):
    APPLIED = "applied"
    ROLLED_BACK = "rolled_back"
    REJECTED = "rejected"


DECLARED_WAVE_STATUSES: Final[frozenset[str]] = frozenset(
    status.value for status in WaveStatus
)


def _text(value: Any, name: str, *, empty: bool = False) -> str:
    if type(value) is not str:
        raise ExtractionWaveError(f"{name} must be a string")
    if value != value.strip() or unicodedata.normalize("NFC", value) != value:
        raise ExtractionWaveError(f"{name} must be trimmed NFC text")
    if not empty and not value:
        raise ExtractionWaveError(f"{name} must be a nonempty string")
    if any(not char.isprintable() for char in value):
        raise ExtractionWaveError(f"{name} contains invalid text")
    if len(value) > MAX_TEXT_CHARS:
        raise ExtractionWaveError(f"{name} exceeds text bound")
    return value


def _cid(value: Any, name: str) -> str:
    text = _text(value, name)
    try:
        return validate_cid(text)
    except Exception as exc:
        raise ExtractionWaveError(f"{name} must be a valid CID") from exc


def _optional_cid(value: Any, name: str) -> str:
    if value in (None, ""):
        return ""
    return _cid(value, name)


def _bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise ExtractionWaveError(f"{name} must be a boolean")
    return value


def _int(value: Any, name: str, *, minimum: int = 0, maximum: int | None = None) -> int:
    if type(value) is bool or type(value) is not int:
        raise ExtractionWaveError(f"{name} must be an integer")
    if value < minimum:
        raise ExtractionWaveError(f"{name} is out of range")
    if maximum is not None and value > maximum:
        raise ExtractionWaveError(f"{name} is out of range")
    return value


def _tree_id(value: Any) -> str:
    text = _text(value, "tree_id")
    if len(text) not in {40, 64} or any(
        char not in "0123456789abcdef" for char in text
    ):
        raise ExtractionWaveError("tree_id must be a lowercase hex Git tree identity")
    return text


def _closed(data: Mapping[str, Any], fields: frozenset[str], name: str) -> dict[str, Any]:
    if not isinstance(data, Mapping) or isinstance(data, (str, bytes, bytearray)):
        raise ExtractionWaveError(f"{name} must be an object")
    extra = set(data) - fields
    missing = fields - set(data)
    if extra & IDENTITY_EXCLUDED_FIELDS:
        raise ExtractionWaveError(
            f"{name} identity excludes observational fields: "
            f"{sorted(extra & IDENTITY_EXCLUDED_FIELDS)}"
        )
    if extra:
        raise ExtractionWaveError(f"unknown {name} field: {sorted(extra)}")
    if missing:
        raise ExtractionWaveError(f"missing {name} field: {sorted(missing)}")
    return dict(data)


def _reject_excluded(payload: Mapping[str, Any], name: str) -> None:
    present = IDENTITY_EXCLUDED_FIELDS & set(payload)
    if present:
        raise ExtractionWaveError(
            f"{name} identity excludes observational fields: {sorted(present)}"
        )


def _require_dag_json(value: Any, name: str) -> None:
    try:
        cid_for_dag_json(value)
    except Exception as exc:
        raise ExtractionWaveError(f"{name} must be strict DAG-JSON") from exc


def _verify_cid(claimed: Any, computed: str, name: str) -> None:
    cid = _cid(claimed, name)
    if cid != computed:
        raise ExtractionWaveError(f"{name} does not verify")


def _unique_sorted_text(values: Any, name: str, *, limit: int) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        raise ExtractionWaveError(f"{name} must be a list")
    ordered = tuple(sorted(_text(item, name) for item in values))
    if len(ordered) > limit:
        raise ExtractionWaveError(f"{name} exceeds maximum length")
    if len(ordered) != len(set(ordered)):
        raise ExtractionWaveError(f"{name} must not contain duplicates")
    return ordered


def _unique_ordered_cids(values: Any, name: str, *, limit: int) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        raise ExtractionWaveError(f"{name} must be a list")
    ordered: list[str] = []
    seen: set[str] = set()
    for item in values:
        cid = _cid(item, name)
        if cid in seen:
            raise ExtractionWaveError(f"{name} must not contain duplicates")
        seen.add(cid)
        ordered.append(cid)
    if len(ordered) > limit:
        raise ExtractionWaveError(f"{name} exceeds maximum length")
    return tuple(ordered)


def _enum(value: Any, enum_type: type[Enum], name: str) -> str:
    if isinstance(value, enum_type):
        return value.value
    text = _text(value, name)
    try:
        return enum_type(text).value
    except ValueError as exc:
        raise ExtractionWaveError(f"unknown {name}: {text}") from exc


def _project(value: Any) -> Any:
    if value is None or type(value) in {str, bool, int}:
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _project(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_project(item) for item in value]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _project(to_dict())
    raise ExtractionWaveError(f"unsupported projected type {type(value).__name__}")


def _mapping(value: Any, name: str) -> dict[str, Any]:
    projected = _project(value)
    if not isinstance(projected, dict):
        raise ExtractionWaveError(f"{name} must be an object")
    _reject_excluded(projected, name)
    return projected


def extraction_wave_cid_profile() -> dict[str, str]:
    return {
        "profile_id": "ipfs_accelerate_py.cid-utils@1",
        "codec": "dag-json",
        "rule": (
            "CID identifies exact canonical bytes under declared codec/profile, "
            "not universal meaning"
        ),
    }


def _pop_authority_flags(payload: dict[str, Any], name: str) -> None:
    for flag in _AUTHORITY_FLAG_NAMES:
        if payload.pop(flag) is not False:
            raise ExtractionWaveError(f"{name} cannot claim {flag}")


def _exact_path(value: Any, name: str = "write_paths") -> str:
    raw = _text(value, name, empty=False)
    if len(raw) > MAX_PATH_CHARS:
        raise ExtractionWaveError(f"{name} exceeds path bound")
    normalized = raw.replace("\\", "/")
    candidate = PurePosixPath(normalized)
    if (
        candidate.is_absolute()
        or ".." in candidate.parts
        or normalized in {".", ""}
        or normalized.startswith("./")
        or any(char in normalized for char in "*?[]{}")
        or "//" in normalized
        or normalized.endswith("/")
    ):
        raise ExtractionWaveError(
            f"{name} must be an exact repository-relative path; unrestricted scope is rejected"
        )
    if normalized != candidate.as_posix():
        raise ExtractionWaveError(
            f"{name} must be a normalized repository-relative path"
        )
    return normalized


def _exact_paths(values: Any, name: str, *, required: bool = True) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise ExtractionWaveError(f"{name} must be a list of exact paths")
    ordered: list[str] = []
    seen: set[str] = set()
    for item in values:
        path = _exact_path(item, name)
        if path not in seen:
            seen.add(path)
            ordered.append(path)
    if required and not ordered:
        raise ExtractionWaveError(
            f"{name} must not be empty; unrestricted scope is rejected"
        )
    if len(ordered) > MAX_WRITE_PATHS:
        raise ExtractionWaveError(f"{name} exceeds path bound")
    return tuple(ordered)


def _coerce_packet(
    packet: RefactorTransformationPacket | Mapping[str, Any],
) -> RefactorTransformationPacket:
    if isinstance(packet, RefactorTransformationPacket):
        resolved = packet
    elif isinstance(packet, Mapping):
        try:
            resolved = RefactorTransformationPacket.from_dict(packet)
        except TransformationPacketError as exc:
            raise ExtractionWaveError(str(exc)) from exc
    else:
        raise ExtractionWaveError("packet must be a RefactorTransformationPacket")
    if resolved.can_authorize_completion is not False:
        raise ExtractionWaveError("wave cannot authorize completion")
    if resolved.can_authorize_transition is not False:
        raise ExtractionWaveError("wave cannot authorize a transition")
    if resolved.packet_is_nomination_only is not True:
        raise ExtractionWaveError("wave must remain nomination_only")
    if resolved.unrestricted_scope is not False:
        raise ExtractionWaveError("unrestricted scope is rejected")
    if resolved.unsupported_syntax is not False:
        raise ExtractionWaveError("unsupported syntax is a typed terminal")
    unknown_effects = [
        item
        for item in resolved.effect_scope.allowed_effects
        if item not in DECLARED_ALLOWED_EFFECTS
    ]
    if unknown_effects:
        raise ExtractionWaveError(f"unknown allowed_effects: {unknown_effects}")
    forbidden = set(resolved.effect_scope.forbidden_effects)
    if not DECLARED_FORBIDDEN_EFFECTS <= forbidden and "network" not in forbidden:
        raise ExtractionWaveError("packet must retain declared forbidden effects")
    return resolved


def _coerce_packets(
    packets: Sequence[RefactorTransformationPacket | Mapping[str, Any]]
    | RefactorTransformationPacket
    | Mapping[str, Any],
) -> tuple[RefactorTransformationPacket, ...]:
    if isinstance(packets, (RefactorTransformationPacket, Mapping)):
        items: Sequence[Any] = (packets,)
    elif isinstance(packets, (str, bytes, bytearray)) or not isinstance(
        packets, Sequence
    ):
        raise ExtractionWaveError("packets must be a list")
    else:
        items = packets
    if not items:
        raise ExtractionWaveError("wave requires at least one packet")
    if len(items) > MAX_PACKETS:
        raise ExtractionWaveError("packets exceed maximum length")
    resolved = tuple(_coerce_packet(item) for item in items)
    ids = [item.packet_cid for item in resolved]
    if len(ids) != len(set(ids)):
        raise ExtractionWaveError("packet_cids must not contain duplicates")
    tree_ids = {item.tree_id for item in resolved}
    if len(tree_ids) != 1:
        raise ExtractionWaveError("wave packets must share one tree_id")
    leases = {item.lease_fence.lease_id for item in resolved}
    fences = {item.lease_fence.fence_id for item in resolved}
    if len(leases) != 1 or len(fences) != 1:
        raise ExtractionWaveError("wave packets must share one lease/fence")
    return resolved


def _worktree_id(
    value: Any,
    *,
    tree_id: str,
    lease_id: str,
    fence_id: str,
) -> str:
    text = _text(value, "worktree_id", empty=True)
    if text:
        return _cid(text, "worktree_id")
    return cid_for_dag_json(
        {
            "tree_id": tree_id,
            "lease_id": lease_id,
            "fence_id": fence_id,
            "isolated": True,
        }
    )


def _edit_kinds(packet: RefactorTransformationPacket) -> frozenset[str]:
    return frozenset(item.kind for item in packet.edits)


def _adapter_kinds(packet: RefactorTransformationPacket) -> frozenset[str]:
    return frozenset(item.adapter_kind for item in packet.edits if item.adapter_kind)


def _rewrite_kinds(packet: RefactorTransformationPacket) -> frozenset[str]:
    return frozenset(item.rewrite_kind for item in packet.edits if item.rewrite_kind)


def verify_before_hashes(
    packet: RefactorTransformationPacket,
    *,
    claimed_before_hashes: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """Return exact preimage source CIDs. Stale claimed hashes fail closed."""

    before = tuple(packet.preimage.source_cids)
    if not before:
        raise ExtractionWaveError("wave requires exact before hashes")
    if claimed_before_hashes is None:
        return before
    claimed = _unique_sorted_text(
        list(claimed_before_hashes), "claimed_before_hashes", limit=MAX_MEMBERS
    )
    if set(claimed) != set(before):
        raise ExtractionWaveError("before hashes do not verify")
    return before


def audit_wave_effects(packet: RefactorTransformationPacket) -> "EffectAudit":
    """Audit one packet's effect scope. Undeclared effects fail closed."""

    allowed = tuple(packet.effect_scope.allowed_effects)
    forbidden = tuple(packet.effect_scope.forbidden_effects)
    undeclared = [
        item for item in allowed if item not in DECLARED_ALLOWED_EFFECTS
    ]
    if undeclared:
        raise ExtractionWaveError(f"undeclared effects are rejected: {undeclared}")
    overlap = set(allowed) & set(forbidden)
    if overlap:
        raise ExtractionWaveError(
            f"allowed effects cannot also be forbidden: {sorted(overlap)}"
        )
    return EffectAudit(
        packet_cid=packet.packet_cid,
        tree_id=packet.tree_id,
        write_paths=packet.effect_scope.write_paths,
        allowed_effects=allowed,
        forbidden_effects=forbidden,
        audited_effects=allowed,
    )


def order_wave_packets(
    packets: Sequence[RefactorTransformationPacket],
    *,
    packet_dependencies: Mapping[str, Sequence[str]] | None = None,
) -> tuple[RefactorTransformationPacket, ...]:
    """Return packets in dependency order. Cycles fail closed."""

    if ONE_PACKET_AT_A_TIME is not True:
        raise ExtractionWaveError("wave must apply one packet at a time")
    indexed = {item.packet_cid: item for item in packets}
    if packet_dependencies is None:
        return tuple(packets)
    if not isinstance(packet_dependencies, Mapping) or isinstance(
        packet_dependencies, (str, bytes, bytearray)
    ):
        raise ExtractionWaveError("packet_dependencies must be an object")
    incoming: dict[str, set[str]] = {cid: set() for cid in indexed}
    outgoing: dict[str, set[str]] = {cid: set() for cid in indexed}
    for source, deps in packet_dependencies.items():
        src = _cid(source, "packet_dependencies")
        if src not in indexed:
            raise ExtractionWaveError("packet_dependencies reference unknown packet")
        if isinstance(deps, (str, bytes, bytearray)) or not isinstance(deps, Sequence):
            raise ExtractionWaveError("packet_dependencies values must be a list")
        for dep in deps:
            predecessor = _cid(dep, "packet_dependencies")
            if predecessor not in indexed:
                raise ExtractionWaveError(
                    "packet_dependencies reference unknown packet"
                )
            if predecessor == src:
                raise ExtractionWaveError("packet cannot depend on itself")
            incoming[src].add(predecessor)
            outgoing[predecessor].add(src)
    ready = [item.packet_cid for item in packets if not incoming[item.packet_cid]]
    ordered: list[str] = []
    while ready:
        current = ready.pop(0)
        ordered.append(current)
        for nxt in sorted(outgoing[current]):
            incoming[nxt].discard(current)
            if not incoming[nxt] and nxt not in ordered and nxt not in ready:
                ready.append(nxt)
    if len(ordered) != len(indexed):
        raise ExtractionWaveError("packet dependency cycle is rejected")
    return tuple(indexed[cid] for cid in ordered)


@dataclass(frozen=True, slots=True)
class EffectAudit:
    """Body-free effect-scope audit for one wave step."""

    packet_cid: str
    tree_id: str
    write_paths: Sequence[str]
    allowed_effects: Sequence[str]
    forbidden_effects: Sequence[str]
    audited_effects: Sequence[str]
    undeclared_effects: Sequence[str] = ()

    interface: ClassVar[str] = EFFECT_AUDIT_INTERFACE
    schema: ClassVar[str] = EFFECT_AUDIT_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "packet_cid",
            "tree_id",
            "write_paths",
            "allowed_effects",
            "forbidden_effects",
            "audited_effects",
            "undeclared_effects",
            "audit_cid",
        }
    )

    def __post_init__(self) -> None:
        if tuple(self.undeclared_effects):
            raise ExtractionWaveError("undeclared effects are a typed terminal")
        object.__setattr__(self, "packet_cid", _cid(self.packet_cid, "packet_cid"))
        object.__setattr__(self, "tree_id", _tree_id(self.tree_id))
        object.__setattr__(
            self,
            "write_paths",
            _exact_paths(list(self.write_paths), "write_paths", required=True),
        )
        object.__setattr__(
            self,
            "allowed_effects",
            _unique_sorted_text(
                list(self.allowed_effects), "allowed_effects", limit=MAX_MEMBERS
            ),
        )
        object.__setattr__(
            self,
            "forbidden_effects",
            _unique_sorted_text(
                list(self.forbidden_effects), "forbidden_effects", limit=MAX_MEMBERS
            ),
        )
        object.__setattr__(
            self,
            "audited_effects",
            _unique_sorted_text(
                list(self.audited_effects), "audited_effects", limit=MAX_MEMBERS
            ),
        )
        object.__setattr__(self, "undeclared_effects", ())

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": EFFECT_AUDIT_SCHEMA,
            "interface": EFFECT_AUDIT_INTERFACE,
            "packet_cid": self.packet_cid,
            "tree_id": self.tree_id,
            "write_paths": list(self.write_paths),
            "allowed_effects": list(self.allowed_effects),
            "forbidden_effects": list(self.forbidden_effects),
            "audited_effects": list(self.audited_effects),
            "undeclared_effects": [],
        }
        _require_dag_json(payload, self.__class__.__name__)
        return payload

    @property
    def audit_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["audit_cid"] = self.audit_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EffectAudit":
        _reject_excluded(data, cls.__name__)
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("audit_cid")
        if payload.pop("schema") != EFFECT_AUDIT_SCHEMA:
            raise ExtractionWaveError("unsupported EffectAudit schema")
        if payload.pop("interface") != EFFECT_AUDIT_INTERFACE:
            raise ExtractionWaveError("unsupported EffectAudit interface")
        result = cls(**payload)
        _verify_cid(claimed, result.audit_cid, "EffectAudit audit_cid")
        return result


@dataclass(frozen=True, slots=True)
class VfsMutationReceipt:
    """Nominated kit-owned VFS mutation. SPAR-025 never writes VFS."""

    packet_cid: str
    tree_id: str
    write_paths: Sequence[str]
    before_source_cids: Sequence[str]
    after_source_cids: Sequence[str]
    worktree_id: str
    kit_owns_vfs: bool = True
    mutated: bool = False
    writes_repository: bool = False

    interface: ClassVar[str] = VFS_MUTATION_RECEIPT_INTERFACE
    schema: ClassVar[str] = VFS_MUTATION_RECEIPT_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "packet_cid",
            "tree_id",
            "write_paths",
            "before_source_cids",
            "after_source_cids",
            "worktree_id",
            "kit_owns_vfs",
            "mutated",
            "writes_repository",
            "can_authorize_transition",
            "can_create_authority",
            "receipt_cid",
        }
    )

    def __post_init__(self) -> None:
        if _bool(self.mutated, "mutated"):
            raise ExtractionWaveError("wave cannot mutate VFS")
        if _bool(self.writes_repository, "writes_repository"):
            raise ExtractionWaveError("wave cannot write the repository")
        if not _bool(self.kit_owns_vfs, "kit_owns_vfs"):
            raise ExtractionWaveError("kit remains the VFS authority")
        object.__setattr__(self, "packet_cid", _cid(self.packet_cid, "packet_cid"))
        object.__setattr__(self, "tree_id", _tree_id(self.tree_id))
        object.__setattr__(
            self,
            "write_paths",
            _exact_paths(list(self.write_paths), "write_paths", required=True),
        )
        object.__setattr__(
            self,
            "before_source_cids",
            _unique_sorted_text(
                list(self.before_source_cids), "before_source_cids", limit=MAX_MEMBERS
            ),
        )
        object.__setattr__(
            self,
            "after_source_cids",
            _unique_sorted_text(
                list(self.after_source_cids), "after_source_cids", limit=MAX_MEMBERS
            ),
        )
        if not self.before_source_cids:
            raise ExtractionWaveError("VFS receipt requires before_source_cids")
        object.__setattr__(self, "worktree_id", _cid(self.worktree_id, "worktree_id"))
        object.__setattr__(self, "kit_owns_vfs", True)
        object.__setattr__(self, "mutated", False)
        object.__setattr__(self, "writes_repository", False)

    @property
    def can_authorize_transition(self) -> bool:
        return False

    @property
    def can_create_authority(self) -> bool:
        return False

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": VFS_MUTATION_RECEIPT_SCHEMA,
            "interface": VFS_MUTATION_RECEIPT_INTERFACE,
            "packet_cid": self.packet_cid,
            "tree_id": self.tree_id,
            "write_paths": list(self.write_paths),
            "before_source_cids": list(self.before_source_cids),
            "after_source_cids": list(self.after_source_cids),
            "worktree_id": self.worktree_id,
            "kit_owns_vfs": True,
            "mutated": False,
            "writes_repository": False,
            "can_authorize_transition": False,
            "can_create_authority": False,
        }
        _require_dag_json(payload, self.__class__.__name__)
        return payload

    @property
    def receipt_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["receipt_cid"] = self.receipt_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "VfsMutationReceipt":
        _reject_excluded(data, cls.__name__)
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("receipt_cid")
        if payload.pop("schema") != VFS_MUTATION_RECEIPT_SCHEMA:
            raise ExtractionWaveError("unsupported VfsMutationReceipt schema")
        if payload.pop("interface") != VFS_MUTATION_RECEIPT_INTERFACE:
            raise ExtractionWaveError("unsupported VfsMutationReceipt interface")
        if payload.pop("can_authorize_transition") is not False:
            raise ExtractionWaveError("VFS receipt cannot authorize a transition")
        if payload.pop("can_create_authority") is not False:
            raise ExtractionWaveError("VFS receipt cannot create authority")
        result = cls(**payload)
        _verify_cid(claimed, result.receipt_cid, "VfsMutationReceipt receipt_cid")
        return result


@dataclass(frozen=True, slots=True)
class RollbackPlan:
    """Restore exact preimages or discard the isolated worktree."""

    restore_source_cids: Sequence[str]
    packet_rollback_cids: Sequence[str] = ()
    negative_evidence_cids: Sequence[str] = ()
    worktree_id: str = ""
    mode: str = ROLLBACK_MODE
    retain_negative_evidence: bool = True
    release_lease_fence: bool = True
    advance_accepted_roots: bool = False
    discard_worktree: bool = True

    interface: ClassVar[str] = ROLLBACK_PLAN_INTERFACE
    schema: ClassVar[str] = ROLLBACK_PLAN_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "restore_source_cids",
            "packet_rollback_cids",
            "negative_evidence_cids",
            "worktree_id",
            "mode",
            "retain_negative_evidence",
            "release_lease_fence",
            "advance_accepted_roots",
            "discard_worktree",
            "rollback_cid",
        }
    )

    def __post_init__(self) -> None:
        mode = _text(self.mode, "mode")
        if mode != ROLLBACK_MODE:
            raise ExtractionWaveError("rollback mode must remain canonical")
        sources = tuple(
            sorted(_cid(item, "restore_source_cids") for item in self.restore_source_cids)
        )
        if not sources:
            raise ExtractionWaveError("rollback requires restore_source_cids")
        if len(sources) != len(set(sources)):
            raise ExtractionWaveError("restore_source_cids must not contain duplicates")
        if _bool(self.advance_accepted_roots, "advance_accepted_roots"):
            raise ExtractionWaveError("rollback cannot advance accepted roots")
        if not _bool(self.retain_negative_evidence, "retain_negative_evidence"):
            raise ExtractionWaveError("rollback must retain negative evidence")
        if not _bool(self.release_lease_fence, "release_lease_fence"):
            raise ExtractionWaveError("rollback must release lease/fence")
        if not _bool(self.discard_worktree, "discard_worktree"):
            raise ExtractionWaveError("failed wave must discard the isolated worktree")
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "restore_source_cids", sources)
        object.__setattr__(
            self,
            "packet_rollback_cids",
            _unique_sorted_text(
                list(self.packet_rollback_cids),
                "packet_rollback_cids",
                limit=MAX_PACKETS,
            ),
        )
        object.__setattr__(
            self,
            "negative_evidence_cids",
            _unique_sorted_text(
                list(self.negative_evidence_cids),
                "negative_evidence_cids",
                limit=MAX_RECEIPTS,
            ),
        )
        object.__setattr__(
            self, "worktree_id", _optional_cid(self.worktree_id, "worktree_id")
        )
        object.__setattr__(self, "retain_negative_evidence", True)
        object.__setattr__(self, "release_lease_fence", True)
        object.__setattr__(self, "advance_accepted_roots", False)
        object.__setattr__(self, "discard_worktree", True)

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": ROLLBACK_PLAN_SCHEMA,
            "interface": ROLLBACK_PLAN_INTERFACE,
            "restore_source_cids": list(self.restore_source_cids),
            "packet_rollback_cids": list(self.packet_rollback_cids),
            "negative_evidence_cids": list(self.negative_evidence_cids),
            "worktree_id": self.worktree_id,
            "mode": self.mode,
            "retain_negative_evidence": True,
            "release_lease_fence": True,
            "advance_accepted_roots": False,
            "discard_worktree": True,
        }
        _require_dag_json(payload, self.__class__.__name__)
        return payload

    @property
    def rollback_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["rollback_cid"] = self.rollback_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RollbackPlan":
        _reject_excluded(data, cls.__name__)
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("rollback_cid")
        if payload.pop("schema") != ROLLBACK_PLAN_SCHEMA:
            raise ExtractionWaveError("unsupported RollbackPlan schema")
        if payload.pop("interface") != ROLLBACK_PLAN_INTERFACE:
            raise ExtractionWaveError("unsupported RollbackPlan interface")
        result = cls(**payload)
        _verify_cid(claimed, result.rollback_cid, "RollbackPlan rollback_cid")
        return result


def compile_wave_rollback(
    packets: Sequence[RefactorTransformationPacket],
    *,
    worktree_id: str,
    negative_evidence_cids: Sequence[str] = (),
) -> RollbackPlan:
    """Compile the wave rollback over exact packet preimages."""

    sources: list[str] = []
    packet_rollbacks: list[str] = []
    negative: list[str] = list(negative_evidence_cids)
    for packet in packets:
        sources.extend(packet.preimage.source_cids)
        packet_rollbacks.append(packet.rollback.rollback_cid)
        negative.extend(packet.negative_evidence_cids)
    return RollbackPlan(
        restore_source_cids=tuple(sorted(set(sources))),
        packet_rollback_cids=tuple(sorted(set(packet_rollbacks))),
        negative_evidence_cids=tuple(sorted(set(negative))),
        worktree_id=worktree_id,
    )


@dataclass(frozen=True, slots=True)
class ExtractionWaveCheckpoint:
    """Exact before/after checkpoint for one applied packet."""

    step_index: int
    packet_cid: str
    tree_id: str
    before_source_cids: Sequence[str]
    after_source_cids: Sequence[str]
    vfs_mutation_cid: str
    effect_audit_cid: str
    executor_receipt_cids: Sequence[str]
    deferred_edit_cids: Sequence[str] = ()
    worktree_id: str = ""

    interface: ClassVar[str] = EXTRACTION_WAVE_CHECKPOINT_INTERFACE
    schema: ClassVar[str] = EXTRACTION_WAVE_CHECKPOINT_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "step_index",
            "packet_cid",
            "tree_id",
            "before_source_cids",
            "after_source_cids",
            "vfs_mutation_cid",
            "effect_audit_cid",
            "executor_receipt_cids",
            "deferred_edit_cids",
            "worktree_id",
            "checkpoint_cid",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "step_index",
            _int(self.step_index, "step_index", maximum=MAX_STEPS - 1),
        )
        object.__setattr__(self, "packet_cid", _cid(self.packet_cid, "packet_cid"))
        object.__setattr__(self, "tree_id", _tree_id(self.tree_id))
        object.__setattr__(
            self,
            "before_source_cids",
            _unique_sorted_text(
                list(self.before_source_cids), "before_source_cids", limit=MAX_MEMBERS
            ),
        )
        object.__setattr__(
            self,
            "after_source_cids",
            _unique_sorted_text(
                list(self.after_source_cids), "after_source_cids", limit=MAX_MEMBERS
            ),
        )
        if not self.before_source_cids:
            raise ExtractionWaveError("checkpoint requires before_source_cids")
        object.__setattr__(
            self, "vfs_mutation_cid", _cid(self.vfs_mutation_cid, "vfs_mutation_cid")
        )
        object.__setattr__(
            self, "effect_audit_cid", _cid(self.effect_audit_cid, "effect_audit_cid")
        )
        object.__setattr__(
            self,
            "executor_receipt_cids",
            _unique_sorted_text(
                list(self.executor_receipt_cids),
                "executor_receipt_cids",
                limit=MAX_RECEIPTS,
            ),
        )
        object.__setattr__(
            self,
            "deferred_edit_cids",
            _unique_sorted_text(
                list(self.deferred_edit_cids), "deferred_edit_cids", limit=MAX_MEMBERS
            ),
        )
        object.__setattr__(
            self, "worktree_id", _optional_cid(self.worktree_id, "worktree_id")
        )

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": EXTRACTION_WAVE_CHECKPOINT_SCHEMA,
            "interface": EXTRACTION_WAVE_CHECKPOINT_INTERFACE,
            "step_index": self.step_index,
            "packet_cid": self.packet_cid,
            "tree_id": self.tree_id,
            "before_source_cids": list(self.before_source_cids),
            "after_source_cids": list(self.after_source_cids),
            "vfs_mutation_cid": self.vfs_mutation_cid,
            "effect_audit_cid": self.effect_audit_cid,
            "executor_receipt_cids": list(self.executor_receipt_cids),
            "deferred_edit_cids": list(self.deferred_edit_cids),
            "worktree_id": self.worktree_id,
        }
        _require_dag_json(payload, self.__class__.__name__)
        return payload

    @property
    def checkpoint_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["checkpoint_cid"] = self.checkpoint_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExtractionWaveCheckpoint":
        _reject_excluded(data, cls.__name__)
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("checkpoint_cid")
        if payload.pop("schema") != EXTRACTION_WAVE_CHECKPOINT_SCHEMA:
            raise ExtractionWaveError("unsupported ExtractionWaveCheckpoint schema")
        if payload.pop("interface") != EXTRACTION_WAVE_CHECKPOINT_INTERFACE:
            raise ExtractionWaveError("unsupported ExtractionWaveCheckpoint interface")
        result = cls(**payload)
        _verify_cid(
            claimed, result.checkpoint_cid, "ExtractionWaveCheckpoint checkpoint_cid"
        )
        return result


def _coerce_checkpoint(
    value: ExtractionWaveCheckpoint | Mapping[str, Any],
) -> ExtractionWaveCheckpoint:
    if isinstance(value, ExtractionWaveCheckpoint):
        return value
    if isinstance(value, Mapping):
        if "checkpoint_cid" in value:
            return ExtractionWaveCheckpoint.from_dict(value)
        payload = {
            key: item
            for key, item in _project(value).items()
            if key not in {"schema", "interface", "checkpoint_cid"}
        }
        return ExtractionWaveCheckpoint(**payload)
    raise ExtractionWaveError("checkpoint must be an ExtractionWaveCheckpoint")


@dataclass(frozen=True, slots=True)
class ExtractionWavePlan:
    """Ordered one-packet-at-a-time wave plan. Nomination only."""

    tree_id: str
    packet_cids: Sequence[str]
    checkpoint_cids: Sequence[str]
    rollback_cid: str
    write_paths: Sequence[str]
    worktree_id: str
    lease_id: str
    fence_id: str
    analyzer_id: str = ANALYZER_ID

    interface: ClassVar[str] = EXTRACTION_WAVE_PLAN_INTERFACE
    schema: ClassVar[str] = EXTRACTION_WAVE_PLAN_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "tree_id",
            "packet_cids",
            "checkpoint_cids",
            "rollback_cid",
            "write_paths",
            "worktree_id",
            "lease_id",
            "fence_id",
            "analyzer_id",
            "can_authorize_transition",
            "can_authorize_completion",
            "can_create_authority",
            "projection_is_authority",
            "plan_is_nomination_only",
            "one_packet_at_a_time",
            "plan_cid",
        }
    )

    def __post_init__(self) -> None:
        analyzer = _text(self.analyzer_id, "analyzer_id")
        if analyzer != ANALYZER_ID:
            raise ExtractionWaveError("analyzer_id must remain the SPAR-025 analyzer")
        packets = _unique_ordered_cids(
            list(self.packet_cids), "packet_cids", limit=MAX_PACKETS
        )
        if not packets:
            raise ExtractionWaveError("plan requires packet_cids")
        object.__setattr__(self, "tree_id", _tree_id(self.tree_id))
        object.__setattr__(self, "packet_cids", packets)
        object.__setattr__(
            self,
            "checkpoint_cids",
            _unique_ordered_cids(
                list(self.checkpoint_cids), "checkpoint_cids", limit=MAX_STEPS
            ),
        )
        object.__setattr__(self, "rollback_cid", _cid(self.rollback_cid, "rollback_cid"))
        object.__setattr__(
            self,
            "write_paths",
            _exact_paths(list(self.write_paths), "write_paths", required=True),
        )
        object.__setattr__(self, "worktree_id", _cid(self.worktree_id, "worktree_id"))
        object.__setattr__(self, "lease_id", _cid(self.lease_id, "lease_id"))
        object.__setattr__(self, "fence_id", _cid(self.fence_id, "fence_id"))
        object.__setattr__(self, "analyzer_id", analyzer)

    @property
    def can_authorize_transition(self) -> bool:
        return False

    @property
    def can_authorize_completion(self) -> bool:
        return False

    @property
    def can_create_authority(self) -> bool:
        return False

    @property
    def projection_is_authority(self) -> bool:
        return False

    @property
    def plan_is_nomination_only(self) -> bool:
        return True

    @property
    def one_packet_at_a_time(self) -> bool:
        return True

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": EXTRACTION_WAVE_PLAN_SCHEMA,
            "interface": EXTRACTION_WAVE_PLAN_INTERFACE,
            "tree_id": self.tree_id,
            "packet_cids": list(self.packet_cids),
            "checkpoint_cids": list(self.checkpoint_cids),
            "rollback_cid": self.rollback_cid,
            "write_paths": list(self.write_paths),
            "worktree_id": self.worktree_id,
            "lease_id": self.lease_id,
            "fence_id": self.fence_id,
            "analyzer_id": self.analyzer_id,
            "can_authorize_transition": False,
            "can_authorize_completion": False,
            "can_create_authority": False,
            "projection_is_authority": False,
            "plan_is_nomination_only": True,
            "one_packet_at_a_time": True,
        }
        _require_dag_json(payload, self.__class__.__name__)
        return payload

    @property
    def plan_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["plan_cid"] = self.plan_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExtractionWavePlan":
        _reject_excluded(data, cls.__name__)
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("plan_cid")
        if payload.pop("schema") != EXTRACTION_WAVE_PLAN_SCHEMA:
            raise ExtractionWaveError("unsupported ExtractionWavePlan schema")
        if payload.pop("interface") != EXTRACTION_WAVE_PLAN_INTERFACE:
            raise ExtractionWaveError("unsupported ExtractionWavePlan interface")
        _pop_authority_flags(payload, "ExtractionWavePlan")
        if payload.pop("plan_is_nomination_only") is not True:
            raise ExtractionWaveError("plan must remain nomination_only")
        if payload.pop("one_packet_at_a_time") is not True:
            raise ExtractionWaveError("wave must apply one packet at a time")
        result = cls(**payload)
        _verify_cid(claimed, result.plan_cid, "ExtractionWavePlan plan_cid")
        return result


@dataclass(frozen=True, slots=True)
class ExtractionWaveReceipt:
    """Body-free SPAR-025 execution receipt. Independent validation remains separate."""

    tree_id: str
    packet_cids: Sequence[str]
    checkpoint_cids: Sequence[str]
    plan_cid: str
    rollback_cid: str
    write_paths: Sequence[str]
    worktree_id: str
    vfs_mutation_cids: Sequence[str]
    effect_audit_cids: Sequence[str]
    negative_evidence_cids: Sequence[str] = ()
    status: WaveStatus | str = WaveStatus.APPLIED
    analyzer_id: str = ANALYZER_ID
    mutated: bool = False
    deterministic: bool = True
    writes_repository: bool = False
    advance_accepted_roots: bool = False

    interface: ClassVar[str] = EXTRACTION_WAVE_RECEIPT_INTERFACE
    schema: ClassVar[str] = EXTRACTION_WAVE_RECEIPT_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "tree_id",
            "packet_cids",
            "checkpoint_cids",
            "plan_cid",
            "rollback_cid",
            "write_paths",
            "worktree_id",
            "vfs_mutation_cids",
            "effect_audit_cids",
            "negative_evidence_cids",
            "status",
            "analyzer_id",
            "mutated",
            "deterministic",
            "writes_repository",
            "advance_accepted_roots",
            "can_authorize_transition",
            "can_authorize_completion",
            "can_create_authority",
            "projection_is_authority",
            "executor_is_nomination_only",
            "receipt_cid",
        }
    )

    def __post_init__(self) -> None:
        analyzer = _text(self.analyzer_id, "analyzer_id")
        if analyzer != ANALYZER_ID:
            raise ExtractionWaveError("analyzer_id must remain the SPAR-025 analyzer")
        status = _enum(self.status, WaveStatus, "status")
        if _bool(self.mutated, "mutated"):
            raise ExtractionWaveError("executor cannot mutate")
        if not _bool(self.deterministic, "deterministic"):
            raise ExtractionWaveError("executor must remain deterministic")
        if _bool(self.writes_repository, "writes_repository"):
            raise ExtractionWaveError("wave cannot write the repository")
        if _bool(self.advance_accepted_roots, "advance_accepted_roots"):
            raise ExtractionWaveError("wave cannot advance accepted roots")
        packets = _unique_ordered_cids(
            list(self.packet_cids), "packet_cids", limit=MAX_PACKETS
        )
        if not packets:
            raise ExtractionWaveError("receipt requires packet_cids")
        object.__setattr__(self, "tree_id", _tree_id(self.tree_id))
        object.__setattr__(self, "packet_cids", packets)
        object.__setattr__(
            self,
            "checkpoint_cids",
            _unique_ordered_cids(
                list(self.checkpoint_cids), "checkpoint_cids", limit=MAX_STEPS
            ),
        )
        if status == WaveStatus.APPLIED.value and len(self.checkpoint_cids) != len(
            packets
        ):
            raise ExtractionWaveError("applied wave requires one checkpoint per packet")
        object.__setattr__(self, "plan_cid", _cid(self.plan_cid, "plan_cid"))
        object.__setattr__(self, "rollback_cid", _cid(self.rollback_cid, "rollback_cid"))
        object.__setattr__(
            self,
            "write_paths",
            _exact_paths(list(self.write_paths), "write_paths", required=True),
        )
        object.__setattr__(self, "worktree_id", _cid(self.worktree_id, "worktree_id"))
        object.__setattr__(
            self,
            "vfs_mutation_cids",
            _unique_ordered_cids(
                list(self.vfs_mutation_cids), "vfs_mutation_cids", limit=MAX_STEPS
            ),
        )
        object.__setattr__(
            self,
            "effect_audit_cids",
            _unique_ordered_cids(
                list(self.effect_audit_cids), "effect_audit_cids", limit=MAX_STEPS
            ),
        )
        object.__setattr__(
            self,
            "negative_evidence_cids",
            _unique_sorted_text(
                list(self.negative_evidence_cids),
                "negative_evidence_cids",
                limit=MAX_RECEIPTS,
            ),
        )
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "analyzer_id", analyzer)
        object.__setattr__(self, "mutated", False)
        object.__setattr__(self, "deterministic", True)
        object.__setattr__(self, "writes_repository", False)
        object.__setattr__(self, "advance_accepted_roots", False)

    @property
    def can_authorize_transition(self) -> bool:
        return False

    @property
    def can_authorize_completion(self) -> bool:
        return False

    @property
    def can_create_authority(self) -> bool:
        return False

    @property
    def projection_is_authority(self) -> bool:
        return False

    @property
    def executor_is_nomination_only(self) -> bool:
        return True

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": EXTRACTION_WAVE_RECEIPT_SCHEMA,
            "interface": EXTRACTION_WAVE_RECEIPT_INTERFACE,
            "tree_id": self.tree_id,
            "packet_cids": list(self.packet_cids),
            "checkpoint_cids": list(self.checkpoint_cids),
            "plan_cid": self.plan_cid,
            "rollback_cid": self.rollback_cid,
            "write_paths": list(self.write_paths),
            "worktree_id": self.worktree_id,
            "vfs_mutation_cids": list(self.vfs_mutation_cids),
            "effect_audit_cids": list(self.effect_audit_cids),
            "negative_evidence_cids": list(self.negative_evidence_cids),
            "status": self.status,
            "analyzer_id": self.analyzer_id,
            "mutated": False,
            "deterministic": True,
            "writes_repository": False,
            "advance_accepted_roots": False,
            "can_authorize_transition": False,
            "can_authorize_completion": False,
            "can_create_authority": False,
            "projection_is_authority": False,
            "executor_is_nomination_only": True,
        }
        _require_dag_json(payload, self.__class__.__name__)
        return payload

    @property
    def receipt_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["receipt_cid"] = self.receipt_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExtractionWaveReceipt":
        _reject_excluded(data, cls.__name__)
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("receipt_cid")
        if payload.pop("schema") != EXTRACTION_WAVE_RECEIPT_SCHEMA:
            raise ExtractionWaveError("unsupported ExtractionWaveReceipt schema")
        if payload.pop("interface") != EXTRACTION_WAVE_RECEIPT_INTERFACE:
            raise ExtractionWaveError("unsupported ExtractionWaveReceipt interface")
        _pop_authority_flags(payload, "ExtractionWaveReceipt")
        if payload.pop("executor_is_nomination_only") is not True:
            raise ExtractionWaveError("executor must remain nomination_only")
        result = cls(**payload)
        _verify_cid(claimed, result.receipt_cid, "ExtractionWaveReceipt receipt_cid")
        return result


def _after_source_cids(
    packet: RefactorTransformationPacket,
    *,
    before: Sequence[str],
    move_receipt: Any | None,
) -> tuple[str, ...]:
    if move_receipt is None:
        return tuple(sorted(before))
    collected: list[str] = []
    for item in getattr(move_receipt, "source_cids", ()):
        if isinstance(item, Mapping):
            cid = item.get("cid")
            if cid:
                collected.append(_cid(cid, "after_source_cids"))
        else:
            collected.append(_cid(item, "after_source_cids"))
    if not collected:
        return tuple(sorted(before))
    return tuple(sorted(set(collected)))


def _deferred_edit_cids(packet: RefactorTransformationPacket) -> tuple[str, ...]:
    return tuple(
        sorted(item.edit_cid for item in packet.edits if item.kind == EditKind.FACADE.value)
    )


def _require_move_inputs(
    packet: RefactorTransformationPacket,
    *,
    raw_sources: Mapping[str, str] | None,
    locators: Sequence[Any] | None,
) -> None:
    if EditKind.MOVE.value not in _edit_kinds(packet):
        return
    if not isinstance(raw_sources, Mapping) or isinstance(
        raw_sources, (str, bytes, bytearray)
    ):
        raise ExtractionWaveError("raw source is required")
    if not raw_sources:
        raise ExtractionWaveError("raw source is required")
    if locators is None or (
        isinstance(locators, (str, bytes, bytearray))
        or not isinstance(locators, Sequence)
        or not tuple(locators)
    ):
        raise ExtractionWaveError("MOVE locators are required")


def _scoped_destination_paths(
    packet: RefactorTransformationPacket,
    destination_paths: Mapping[str, str] | None,
) -> Mapping[str, str] | None:
    if destination_paths is None:
        return None
    allowed = set(packet.effect_scope.write_paths)
    return {
        key: path
        for key, path in destination_paths.items()
        if path in allowed
    }


def _scoped_locators(
    packet: RefactorTransformationPacket,
    locators: Sequence[Any] | None,
) -> Sequence[Any] | None:
    if locators is None:
        return None
    allowed = set(packet.effect_scope.write_paths)
    scoped: list[Any] = []
    for item in locators:
        if isinstance(item, MemberLocator):
            locator = item
        elif isinstance(item, Mapping):
            locator = (
                MemberLocator.from_dict(item)
                if "locator_cid" in item
                else MemberLocator(
                    **{
                        key: value
                        for key, value in _project(item).items()
                        if key not in {"schema", "interface", "locator_cid"}
                    }
                )
            )
        else:
            raise ExtractionWaveError("locator must be a MemberLocator")
        if locator.path in allowed:
            scoped.append(locator)
    return tuple(scoped)


def _scoped_raw_sources(
    packet: RefactorTransformationPacket,
    raw_sources: Mapping[str, str] | None,
) -> Mapping[str, str] | None:
    if raw_sources is None:
        return None
    allowed = set(packet.effect_scope.write_paths)
    return {
        path: text
        for path, text in raw_sources.items()
        if path in allowed
    }


def _dispatch_packet(
    packet: RefactorTransformationPacket,
    *,
    raw_sources: Mapping[str, str] | None,
    locators: Sequence[Any] | None,
    destination_paths: Mapping[str, str] | None,
    predecessor_kwargs: Mapping[str, Any],
) -> tuple[tuple[str, ...], Any | None]:
    receipts: list[str] = []
    move_receipt = None
    kinds = _edit_kinds(packet)
    adapters = _adapter_kinds(packet)
    rewrites = _rewrite_kinds(packet)
    kwargs = dict(predecessor_kwargs)
    scoped_locators = _scoped_locators(packet, locators)
    scoped_destinations = _scoped_destination_paths(packet, destination_paths)
    scoped_sources = _scoped_raw_sources(packet, raw_sources)
    if EditKind.MOVE.value in kinds:
        _require_move_inputs(
            packet, raw_sources=scoped_sources, locators=scoped_locators
        )
        result = apply_cst_extraction(
            packet,
            raw_sources=scoped_sources or {},
            locators=scoped_locators or (),
            destination_paths=scoped_destinations,
        )
        move_receipt = result.receipt()
        receipts.append(move_receipt.receipt_cid)
    if rewrites & SPAR021_REWRITE_KINDS:
        receipt = execute_import_rewrites(packet, **kwargs)
        receipts.append(receipt.receipt_cid)
    if adapters & _SPAR022_STATE_KINDS:
        receipt = execute_explicit_state_objects(packet, **kwargs)
        receipts.append(receipt.receipt_cid)
    if adapters & SPAR023_ADAPTER_KINDS:
        receipt = execute_initialization_rewrites(packet, **kwargs)
        receipts.append(receipt.receipt_cid)
    if AdapterKind.WRAPPER.value in adapters or rewrites & _SPAR024_REWRITE_KINDS:
        receipt = execute_binding_compatibility_adapters(packet, **kwargs)
        receipts.append(receipt.receipt_cid)
    if not receipts and EditKind.FACADE.value not in kinds:
        raise ExtractionWaveError("packet has no dispatchable extraction-wave edits")
    return tuple(sorted(set(receipts))), move_receipt


def execute_extraction_wave(
    packets: Sequence[RefactorTransformationPacket | Mapping[str, Any]]
    | RefactorTransformationPacket
    | Mapping[str, Any],
    *,
    raw_sources: Mapping[str, str] | None = None,
    locators: Sequence[MemberLocator | Mapping[str, Any]] | None = None,
    destination_paths: Mapping[str, str] | None = None,
    packet_dependencies: Mapping[str, Sequence[str]] | None = None,
    claimed_before_hashes: Mapping[str, Sequence[str]] | None = None,
    worktree_id: str = "",
    mutate: bool = False,
    **predecessor_kwargs: Any,
) -> ExtractionWaveReceipt:
    """Apply one bounded packet at a time. Nomination only; never mutates."""

    if mutate is not False:
        raise ExtractionWaveError("executor cannot mutate")
    if WAVE_WRITES_REPOSITORY is not False:
        raise ExtractionWaveError("wave cannot write the repository")
    resolved = _coerce_packets(packets)
    ordered = order_wave_packets(resolved, packet_dependencies=packet_dependencies)
    tree_id = ordered[0].tree_id
    lease_id = ordered[0].lease_fence.lease_id
    fence_id = ordered[0].lease_fence.fence_id
    isolated = _worktree_id(
        worktree_id, tree_id=tree_id, lease_id=lease_id, fence_id=fence_id
    )
    rollback = compile_wave_rollback(ordered, worktree_id=isolated)
    checkpoints: list[ExtractionWaveCheckpoint] = []
    vfs_cids: list[str] = []
    audit_cids: list[str] = []
    write_paths: list[str] = []
    seen_paths: set[str] = set()
    try:
        for index, packet in enumerate(ordered):
            claimed = None
            if claimed_before_hashes is not None:
                claimed = claimed_before_hashes.get(packet.packet_cid)
            before = verify_before_hashes(packet, claimed_before_hashes=claimed)
            audit = audit_wave_effects(packet)
            receipts, move_receipt = _dispatch_packet(
                packet,
                raw_sources=raw_sources,
                locators=locators,
                destination_paths=destination_paths,
                predecessor_kwargs=predecessor_kwargs,
            )
            after = _after_source_cids(
                packet, before=before, move_receipt=move_receipt
            )
            vfs = VfsMutationReceipt(
                packet_cid=packet.packet_cid,
                tree_id=packet.tree_id,
                write_paths=packet.effect_scope.write_paths,
                before_source_cids=before,
                after_source_cids=after,
                worktree_id=isolated,
            )
            checkpoint = ExtractionWaveCheckpoint(
                step_index=index,
                packet_cid=packet.packet_cid,
                tree_id=packet.tree_id,
                before_source_cids=before,
                after_source_cids=after,
                vfs_mutation_cid=vfs.receipt_cid,
                effect_audit_cid=audit.audit_cid,
                executor_receipt_cids=receipts,
                deferred_edit_cids=_deferred_edit_cids(packet),
                worktree_id=isolated,
            )
            checkpoints.append(checkpoint)
            vfs_cids.append(vfs.receipt_cid)
            audit_cids.append(audit.audit_cid)
            for path in packet.effect_scope.write_paths:
                if path not in seen_paths:
                    seen_paths.add(path)
                    write_paths.append(path)
    except _PREDECESSOR_ERRORS as exc:
        raise ExtractionWaveError(
            f"wave step failed: {exc}",
            rollback=rollback,
            negative_evidence_cids=rollback.negative_evidence_cids,
        ) from exc
    plan = ExtractionWavePlan(
        tree_id=tree_id,
        packet_cids=tuple(item.packet_cid for item in ordered),
        checkpoint_cids=tuple(item.checkpoint_cid for item in checkpoints),
        rollback_cid=rollback.rollback_cid,
        write_paths=write_paths,
        worktree_id=isolated,
        lease_id=lease_id,
        fence_id=fence_id,
    )
    return ExtractionWaveReceipt(
        tree_id=tree_id,
        packet_cids=plan.packet_cids,
        checkpoint_cids=plan.checkpoint_cids,
        plan_cid=plan.plan_cid,
        rollback_cid=rollback.rollback_cid,
        write_paths=write_paths,
        worktree_id=isolated,
        vfs_mutation_cids=tuple(vfs_cids),
        effect_audit_cids=tuple(audit_cids),
        negative_evidence_cids=rollback.negative_evidence_cids,
        status=WaveStatus.APPLIED,
    )


def compile_extraction_wave_plan(
    packets: Sequence[RefactorTransformationPacket | Mapping[str, Any]]
    | RefactorTransformationPacket
    | Mapping[str, Any],
    **kwargs: Any,
) -> ExtractionWavePlan:
    """Compile the unique SPAR-025 ExtractionWavePlan for one wave."""

    receipt = execute_extraction_wave(packets, **kwargs)
    resolved = _coerce_packets(packets)
    ordered = order_wave_packets(
        resolved, packet_dependencies=kwargs.get("packet_dependencies")
    )
    return ExtractionWavePlan(
        tree_id=receipt.tree_id,
        packet_cids=receipt.packet_cids,
        checkpoint_cids=receipt.checkpoint_cids,
        rollback_cid=receipt.rollback_cid,
        write_paths=receipt.write_paths,
        worktree_id=receipt.worktree_id,
        lease_id=ordered[0].lease_fence.lease_id,
        fence_id=ordered[0].lease_fence.fence_id,
    )


def compile_extraction_wave_receipt(
    packets: Sequence[RefactorTransformationPacket | Mapping[str, Any]]
    | RefactorTransformationPacket
    | Mapping[str, Any],
    **kwargs: Any,
) -> ExtractionWaveReceipt:
    """Compile a SPAR-025 receipt over a checkpointed wave."""

    return execute_extraction_wave(packets, **kwargs)


def dry_run_extraction_wave(
    packets: Sequence[RefactorTransformationPacket | Mapping[str, Any]]
    | RefactorTransformationPacket
    | Mapping[str, Any],
    **kwargs: Any,
) -> ExtractionWaveReceipt:
    """Return a deterministic no-mutation dry-run of one extraction wave."""

    if kwargs.pop("mutate", False):
        raise ExtractionWaveError("dry-run cannot mutate")
    receipt = execute_extraction_wave(packets, **kwargs)
    if receipt.mutated:
        raise ExtractionWaveError("dry-run cannot mutate")
    return receipt


def rollback_extraction_wave(
    packets: Sequence[RefactorTransformationPacket | Mapping[str, Any]]
    | RefactorTransformationPacket
    | Mapping[str, Any],
    *,
    worktree_id: str = "",
    negative_evidence_cids: Sequence[str] = (),
    packet_dependencies: Mapping[str, Sequence[str]] | None = None,
) -> ExtractionWaveReceipt:
    """Restore exact preimages and discard the isolated worktree."""

    resolved = _coerce_packets(packets)
    ordered = order_wave_packets(resolved, packet_dependencies=packet_dependencies)
    tree_id = ordered[0].tree_id
    lease_id = ordered[0].lease_fence.lease_id
    fence_id = ordered[0].lease_fence.fence_id
    isolated = _worktree_id(
        worktree_id, tree_id=tree_id, lease_id=lease_id, fence_id=fence_id
    )
    rollback = compile_wave_rollback(
        ordered, worktree_id=isolated, negative_evidence_cids=negative_evidence_cids
    )
    write_paths: list[str] = []
    seen: set[str] = set()
    for packet in ordered:
        for path in packet.effect_scope.write_paths:
            if path not in seen:
                seen.add(path)
                write_paths.append(path)
    plan = ExtractionWavePlan(
        tree_id=tree_id,
        packet_cids=tuple(item.packet_cid for item in ordered),
        checkpoint_cids=(),
        rollback_cid=rollback.rollback_cid,
        write_paths=write_paths,
        worktree_id=isolated,
        lease_id=lease_id,
        fence_id=fence_id,
    )
    return ExtractionWaveReceipt(
        tree_id=tree_id,
        packet_cids=plan.packet_cids,
        checkpoint_cids=(),
        plan_cid=plan.plan_cid,
        rollback_cid=rollback.rollback_cid,
        write_paths=write_paths,
        worktree_id=isolated,
        vfs_mutation_cids=(),
        effect_audit_cids=(),
        negative_evidence_cids=rollback.negative_evidence_cids,
        status=WaveStatus.ROLLED_BACK,
    )


def apply_extraction_wave(
    packets: Sequence[RefactorTransformationPacket | Mapping[str, Any]]
    | RefactorTransformationPacket
    | Mapping[str, Any],
    **kwargs: Any,
) -> ExtractionWaveReceipt:
    """Transactionally apply a wave; failed steps restore preimages."""

    try:
        return execute_extraction_wave(packets, **kwargs)
    except ExtractionWaveError as exc:
        if exc.rollback is None:
            raise
        rolled = rollback_extraction_wave(
            packets,
            worktree_id=kwargs.get("worktree_id") or "",
            negative_evidence_cids=exc.negative_evidence_cids,
            packet_dependencies=kwargs.get("packet_dependencies"),
        )
        if rolled.status != WaveStatus.ROLLED_BACK.value:
            raise ExtractionWaveError("failed wave must roll back") from exc
        if rolled.advance_accepted_roots:
            raise ExtractionWaveError("rollback cannot advance accepted roots") from exc
        return rolled


@dataclass(frozen=True, slots=True)
class ExtractionWave:
    """Checkpointed transactional extraction-wave executor. Nomination only."""

    interface: ClassVar[str] = EXTRACTION_WAVE_INTERFACE
    schema: ClassVar[str] = EXTRACTION_WAVE_SCHEMA

    def execute(
        self,
        packets: Sequence[RefactorTransformationPacket | Mapping[str, Any]]
        | RefactorTransformationPacket
        | Mapping[str, Any],
        **kwargs: Any,
    ) -> ExtractionWaveReceipt:
        return execute_extraction_wave(packets, **kwargs)

    def dry_run(
        self,
        packets: Sequence[RefactorTransformationPacket | Mapping[str, Any]]
        | RefactorTransformationPacket
        | Mapping[str, Any],
        **kwargs: Any,
    ) -> ExtractionWaveReceipt:
        return dry_run_extraction_wave(packets, **kwargs)

    def rollback(
        self,
        packets: Sequence[RefactorTransformationPacket | Mapping[str, Any]]
        | RefactorTransformationPacket
        | Mapping[str, Any],
        **kwargs: Any,
    ) -> ExtractionWaveReceipt:
        return rollback_extraction_wave(packets, **kwargs)

    def apply(
        self,
        packets: Sequence[RefactorTransformationPacket | Mapping[str, Any]]
        | RefactorTransformationPacket
        | Mapping[str, Any],
        **kwargs: Any,
    ) -> ExtractionWaveReceipt:
        return apply_extraction_wave(packets, **kwargs)


def encode_canonical_checkpoint(
    checkpoint: ExtractionWaveCheckpoint,
) -> dict[str, Any]:
    return checkpoint.to_dict()


def decode_canonical_checkpoint(
    payload: Mapping[str, Any],
) -> ExtractionWaveCheckpoint:
    return ExtractionWaveCheckpoint.from_dict(payload)


def encode_canonical_plan(plan: ExtractionWavePlan) -> dict[str, Any]:
    return plan.to_dict()


def decode_canonical_plan(payload: Mapping[str, Any]) -> ExtractionWavePlan:
    return ExtractionWavePlan.from_dict(payload)


def encode_canonical_receipt(receipt: ExtractionWaveReceipt) -> dict[str, Any]:
    return receipt.to_dict()


def decode_canonical_receipt(payload: Mapping[str, Any]) -> ExtractionWaveReceipt:
    return ExtractionWaveReceipt.from_dict(payload)


def encode_canonical_rollback(plan: RollbackPlan) -> dict[str, Any]:
    return plan.to_dict()


def decode_canonical_rollback(payload: Mapping[str, Any]) -> RollbackPlan:
    return RollbackPlan.from_dict(payload)


def provider_free_exports() -> tuple[str, ...]:
    return tuple(sorted(__all__))


def assert_not_competing_capsule_family() -> None:
    names = set(globals())
    overlap = names & _FORBIDDEN_CAPSULE_TYPE_NAMES
    if overlap:
        raise ExtractionWaveError(
            f"extraction wave must not define capsule types: {sorted(overlap)}"
        )


__all__ = [
    "ANALYZER_ID",
    "AUTHORITY",
    "AUTHORITY_OWNER",
    "DECLARED_WAVE_STATUSES",
    "DRY_RUN_IS_DETERMINISTIC",
    "DRY_RUN_MUTATES",
    "DUCKLAKE_IS_AUTHORITY",
    "EFFECT_AUDIT_INTERFACE",
    "EXECUTOR_IS_NOMINATION_ONLY",
    "EXTRACTION_WAVE_CHECKPOINT_INTERFACE",
    "EXTRACTION_WAVE_INTERFACE",
    "EXTRACTION_WAVE_PLAN_INTERFACE",
    "EXTRACTION_WAVE_RECEIPT_INTERFACE",
    "FAILED_WAVES_RESTORE_PREIMAGES",
    "GOAL_ID",
    "IDENTITY_EXCLUDED_FIELDS",
    "KIT_OWNS_VFS",
    "MARKDOWN_IS_NOT_COMPLETION",
    "MODEL_OUTPUT_IS_PROPOSAL_ONLY",
    "NEGATIVE_EVIDENCE_RETAINED",
    "ONE_PACKET_AT_A_TIME",
    "PLAN_IS_NOMINATION_ONLY",
    "PROGRAM",
    "PROJECTION_CLUSTERING_IS_AUTHORITY",
    "RAW_SOURCE_REQUIRED",
    "ROLLBACK_MODE",
    "ROLLBACK_PLAN_INTERFACE",
    "SOFT_SIGNALS_CANNOT_OVERRIDE_HARD_CONSTRAINTS",
    "TASK_ID",
    "TEST_PASS_IS_NOT_COMPLETION",
    "USES_CURRENT_LEASE_FENCE",
    "USES_CURRENT_WORKTREE",
    "VECTOR_SIMILARITY_IS_AUTHORITY",
    "VFS_AUTHORITY_OWNER",
    "VFS_MUTATION_RECEIPT_INTERFACE",
    "WAVE_CAN_AUTHORIZE_COMPLETION",
    "WAVE_CAN_AUTHORIZE_TRANSITION",
    "WAVE_CAN_CREATE_AUTHORITY",
    "WAVE_CAN_RETIRE_FACADE",
    "WAVE_CONTRACT_VERSION",
    "WAVE_OWNS_VFS",
    "WAVE_WRITES_REPOSITORY",
    "WORKER_SELF_APPROVAL",
    "EffectAudit",
    "ExtractionWave",
    "ExtractionWaveCheckpoint",
    "ExtractionWaveError",
    "ExtractionWavePlan",
    "ExtractionWaveReceipt",
    "RollbackPlan",
    "VfsMutationReceipt",
    "WaveStatus",
    "apply_extraction_wave",
    "assert_not_competing_capsule_family",
    "audit_wave_effects",
    "compile_extraction_wave_plan",
    "compile_extraction_wave_receipt",
    "compile_wave_rollback",
    "decode_canonical_checkpoint",
    "decode_canonical_plan",
    "decode_canonical_receipt",
    "decode_canonical_rollback",
    "dry_run_extraction_wave",
    "encode_canonical_checkpoint",
    "encode_canonical_plan",
    "encode_canonical_receipt",
    "encode_canonical_rollback",
    "execute_extraction_wave",
    "extraction_wave_cid_profile",
    "order_wave_packets",
    "provider_free_exports",
    "rollback_extraction_wave",
    "verify_before_hashes",
]
