"""SPAR-038 fixed-point remodularization controller.

After every accepted or rejected wave, this controller rebuilds semantic
state, graph, SCCs, contracts, frontier, partitions, and selections, then
composes those live slice identities with current roots and merge receipts.

Federation ``FixedPointStore`` is not this contract.  A boolean or prebuilt
convergence flag is not live rebuild identity.  Workers cannot self-approve
a fixed point.  The controller is nomination-only: it cannot authorize a
transition, completion, merge, or competing authority.  Unsupported required
behavior is a typed terminal, never success.  Observational metadata is
excluded from identity.  Dry-run is deterministic and never mutates.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Final, Mapping, Sequence
import unicodedata

from ipfs_accelerate_py.utils.cid_utils import cid_for_dag_json, validate_cid

from .partition_generators import (
    IDENTITY_EXCLUDED_FIELDS as SPAR013_IDENTITY_EXCLUDED_FIELDS,
)


TASK_ID: Final[str] = "SPAR-038"
GOAL_ID: Final[str] = "SPAR-G071"
PROGRAM: Final[str] = "semantic-preserving-autonomous-remodularization-v1"
AUTHORITY: Final[str] = "operational refactoring authority"
AUTHORITY_OWNER: Final[str] = "ipfs_accelerate_py"
ANALYZER_ID: Final[str] = (
    "ipfs_accelerate_py.agent_supervisor.semantic_refactoring.fixed_point@1"
)
PREDECESSOR_TASK_IDS: Final[tuple[str, ...]] = ("SPAR-027", "SPAR-037")

FIXED_POINT_REMODULARIZATION_CONTROLLER_INTERFACE: Final[str] = (
    "FixedPointRemodularizationController@1"
)
ACCEPTED_WAVE_FIXED_POINT_INTERFACE: Final[str] = "AcceptedWaveFixedPoint@1"
COMPOSED_ROOT_INTERFACE: Final[str] = "ComposedRoot@1"
REBUILD_SLICE_SET_INTERFACE: Final[str] = "RebuildSliceSet@1"
MERGE_RECEIPT_INTERFACE: Final[str] = "SparMergeReceipt@1"
WAVE_DISPOSITION_INTERFACE: Final[str] = "WaveDisposition@1"
TYPED_TERMINAL_INTERFACE: Final[str] = "TypedTerminal@1"
FIXED_POINT_ITERATION_INTERFACE: Final[str] = "FixedPointIteration@1"
FIXED_POINT_RECEIPT_INTERFACE: Final[str] = "FixedPointReceipt@1"

FIXED_POINT_REMODULARIZATION_CONTROLLER_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/fixed-point-remodularization-controller@1"
)
ACCEPTED_WAVE_FIXED_POINT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/accepted-wave-fixed-point@1"
)
COMPOSED_ROOT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/composed-root@1"
)
REBUILD_SLICE_SET_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/rebuild-slice-set@1"
)
MERGE_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/spar-merge-receipt@1"
)
WAVE_DISPOSITION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/wave-disposition@1"
)
TYPED_TERMINAL_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/typed-terminal@1"
)
FIXED_POINT_ITERATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/fixed-point-iteration@1"
)
FIXED_POINT_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/fixed-point-receipt@1"
)

FIXED_POINT_CONTRACT_VERSION: Final[str] = "1"

CONTROLLER_CAN_AUTHORIZE_TRANSITION: Final[bool] = False
CONTROLLER_CAN_AUTHORIZE_COMPLETION: Final[bool] = False
CONTROLLER_CAN_CREATE_AUTHORITY: Final[bool] = False
CONTROLLER_CAN_ACCEPT_FIXED_POINT: Final[bool] = False
CONTROLLER_WRITES_REPOSITORY: Final[bool] = False
FEDERATION_FIXED_POINT_STORE_IS_AUTHORITY: Final[bool] = False
BOOLEAN_FIXED_POINT_IS_AUTHORITY: Final[bool] = False
PREBUILT_FIXED_POINT_IS_AUTHORITY: Final[bool] = False
VECTOR_SIMILARITY_IS_AUTHORITY: Final[bool] = False
PROJECTION_CLUSTERING_IS_AUTHORITY: Final[bool] = False
MODEL_OUTPUT_IS_PROPOSAL_ONLY: Final[bool] = True
TEST_PASS_IS_NOT_COMPLETION: Final[bool] = True
MARKDOWN_IS_NOT_COMPLETION: Final[bool] = True
WORKER_SELF_APPROVAL: Final[bool] = False
DUCKLAKE_IS_AUTHORITY: Final[bool] = False
SOFT_SIGNALS_CANNOT_OVERRIDE_HARD_CONSTRAINTS: Final[bool] = True
CONTROLLER_IS_NOMINATION_ONLY: Final[bool] = True
RAW_SOURCE_REQUIRED: Final[bool] = True
DRY_RUN_IS_DETERMINISTIC: Final[bool] = True
DRY_RUN_MUTATES: Final[bool] = False
NEGATIVE_EVIDENCE_RETAINED: Final[bool] = True
REBUILD_AFTER_ACCEPTED_WAVE: Final[bool] = True
REBUILD_AFTER_REJECTED_WAVE: Final[bool] = True
LIVE_REBUILD_REQUIRED: Final[bool] = True
TWO_EPOCH_UNCHANGED_REQUIRED: Final[bool] = True

MAX_TEXT_CHARS: Final[int] = 16_384
MAX_MEMBERS: Final[int] = 16_384
MAX_RECEIPTS: Final[int] = 1_024
MAX_EVIDENCE_CIDS: Final[int] = 1_024
MAX_ITERATIONS: Final[int] = 8
DEFAULT_MAX_ITERATIONS: Final[int] = 8

IDENTITY_EXCLUDED_FIELDS: Final[frozenset[str]] = SPAR013_IDENTITY_EXCLUDED_FIELDS

REBUILD_SLICE_ORDER: Final[tuple[str, ...]] = (
    "semantic_state",
    "graph",
    "sccs",
    "contracts",
    "frontier",
    "partitions",
    "selections",
)
DECLARED_REBUILD_SLICES: Final[frozenset[str]] = frozenset(REBUILD_SLICE_ORDER)

EXISTING_ADAPTER_AUTHORITIES: Final[tuple[str, ...]] = (
    "datasets_semantic",
    "kit_storage",
    "kit_vfs",
    "accelerator_supervisor",
    "accelerator_runtime",
    "spar_narrow",
)

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
        "FixedPointStore",
    }
)

_NON_ADMITTING_EVIDENCE: Final[frozenset[str]] = frozenset(
    {
        "vector_candidate",
        "model_hypothesis",
        "heuristic",
    }
)

_FORBIDDEN_BOOLEAN_KEYS: Final[frozenset[str]] = frozenset(
    {
        "fixed_point",
        "boolean_fixed_point",
        "prebuilt_fixed_point",
        "federation_fixed_point",
        "self_approved",
        "accepted_fixed_point",
        "worker_approved",
    }
)

_FORBIDDEN_STORE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "federation_store",
        "fixed_point_store",
        "FixedPointStore",
        "federation_fixed_point_store",
    }
)

_AUTHORITY_FLAG_NAMES: Final[tuple[str, ...]] = (
    "can_authorize_transition",
    "can_authorize_completion",
    "can_create_authority",
    "can_accept_fixed_point",
    "projection_is_authority",
    "federation_store_is_authority",
    "boolean_fixed_point_is_authority",
    "prebuilt_fixed_point_is_authority",
    "worker_self_approval",
)

FORBIDDEN_FIXED_POINT_NAMES: Final[frozenset[str]] = frozenset(
    {
        "FixedPointStore",
        "admit_boolean_fixed_point",
        "admit_prebuilt_fixed_point",
        "self_approve_fixed_point",
        "federation_fixed_point",
    }
)

ALLOWED_WAVE_STATUSES: Final[frozenset[str]] = frozenset(
    {"applied", "rolled_back", "rejected"}
)
ALLOWED_VALIDATION_STATUSES: Final[frozenset[str]] = frozenset(
    {"validated", "rejected", "unsupported", "incomplete"}
)
REJECTED_WAVE_STATUSES: Final[frozenset[str]] = frozenset(
    {"rolled_back", "rejected"}
)
REJECTED_VALIDATION_STATUSES: Final[frozenset[str]] = frozenset({"rejected"})
TERMINAL_VALIDATION_STATUSES: Final[frozenset[str]] = frozenset({"unsupported"})


class FixedPointError(ValueError):
    """Fail-closed violation of a SPAR-038 fixed-point contract."""


class WaveOutcome(str, Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class ControllerStatus(str, Enum):
    CONTINUE = "continue"
    NOMINATED_FIXED_POINT = "nominated_fixed_point"
    TYPED_TERMINAL = "typed_terminal"


class TerminalKind(str, Enum):
    UNSUPPORTED = "unsupported"
    HUMAN_REVIEW = "human_review"
    CAPABILITY_UNAVAILABLE = "capability_unavailable"
    MAX_ITERATIONS = "max_iterations"


class MergeDisposition(str, Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"


DECLARED_WAVE_OUTCOMES: Final[frozenset[str]] = frozenset(
    item.value for item in WaveOutcome
)
DECLARED_CONTROLLER_STATUSES: Final[frozenset[str]] = frozenset(
    item.value for item in ControllerStatus
)
DECLARED_TERMINAL_KINDS: Final[frozenset[str]] = frozenset(
    item.value for item in TerminalKind
)
DECLARED_MERGE_DISPOSITIONS: Final[frozenset[str]] = frozenset(
    item.value for item in MergeDisposition
)


def _text(value: Any, name: str, *, empty: bool = False) -> str:
    if type(value) is not str:
        raise FixedPointError(f"{name} must be a string")
    if value != value.strip() or unicodedata.normalize("NFC", value) != value:
        raise FixedPointError(f"{name} must be trimmed NFC text")
    if not empty and not value:
        raise FixedPointError(f"{name} must be a nonempty string")
    if any(not char.isprintable() for char in value):
        raise FixedPointError(f"{name} contains invalid text")
    if len(value) > MAX_TEXT_CHARS:
        raise FixedPointError(f"{name} exceeds text bound")
    return value


def _cid(value: Any, name: str) -> str:
    text = _text(value, name)
    try:
        return validate_cid(text)
    except Exception as exc:
        raise FixedPointError(f"{name} must be a valid CID") from exc


def _optional_cid(value: Any, name: str) -> str:
    if value in (None, ""):
        return ""
    return _cid(value, name)


def _bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise FixedPointError(f"{name} must be a boolean")
    return value


def _int(
    value: Any,
    name: str,
    *,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if type(value) is bool or type(value) is not int:
        raise FixedPointError(f"{name} must be an integer")
    if value < minimum:
        raise FixedPointError(f"{name} is out of range")
    if maximum is not None and value > maximum:
        raise FixedPointError(f"{name} is out of range")
    return value


def _tree_id(value: Any) -> str:
    text = _text(value, "tree_id")
    if len(text) not in {40, 64} or any(
        char not in "0123456789abcdef" for char in text
    ):
        raise FixedPointError("tree_id must be a lowercase hex Git tree identity")
    return text


def _closed(data: Mapping[str, Any], fields: frozenset[str], name: str) -> dict[str, Any]:
    if not isinstance(data, Mapping) or isinstance(data, (str, bytes, bytearray)):
        raise FixedPointError(f"{name} must be an object")
    extra = set(data) - fields
    missing = fields - set(data)
    if extra & IDENTITY_EXCLUDED_FIELDS:
        raise FixedPointError(
            f"{name} identity excludes observational fields: "
            f"{sorted(extra & IDENTITY_EXCLUDED_FIELDS)}"
        )
    if extra:
        raise FixedPointError(f"unknown {name} field: {sorted(extra)}")
    if missing:
        raise FixedPointError(f"missing {name} field: {sorted(missing)}")
    return dict(data)


def _reject_excluded(payload: Mapping[str, Any], name: str) -> None:
    present = IDENTITY_EXCLUDED_FIELDS & set(payload)
    if present:
        raise FixedPointError(
            f"{name} identity excludes observational fields: {sorted(present)}"
        )


def _require_dag_json(value: Any, name: str) -> None:
    try:
        cid_for_dag_json(value)
    except Exception as exc:
        raise FixedPointError(f"{name} must be strict DAG-JSON") from exc


def _verify_cid(claimed: Any, computed: str, name: str) -> None:
    cid = _cid(claimed, name)
    if cid != computed:
        raise FixedPointError(f"{name} does not verify")


def _enum(value: Any, enum_type: type[Enum], name: str) -> str:
    if isinstance(value, enum_type):
        return value.value
    text = _text(value, name)
    try:
        return enum_type(text).value
    except ValueError as exc:
        raise FixedPointError(f"unknown {name}: {text}") from exc


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
    raise FixedPointError(f"unsupported projected type {type(value).__name__}")


def _mapping(value: Any, name: str) -> dict[str, Any]:
    projected = _project(value)
    if not isinstance(projected, dict):
        raise FixedPointError(f"{name} must be an object")
    _reject_excluded(projected, name)
    return projected


def _unique_sorted_text(values: Any, name: str, *, limit: int) -> tuple[str, ...]:
    if values in (None, (), []):
        ordered: tuple[str, ...] = ()
    elif not isinstance(values, (list, tuple)):
        raise FixedPointError(f"{name} must be a list")
    else:
        ordered = tuple(sorted(_text(item, name) for item in values))
    if len(ordered) > limit:
        raise FixedPointError(f"{name} exceeds maximum length")
    if len(ordered) != len(set(ordered)):
        raise FixedPointError(f"{name} must not contain duplicates")
    return ordered


def _cids(values: Any, name: str, *, required: bool = False) -> tuple[str, ...]:
    if values in (None, (), []):
        ordered: tuple[str, ...] = ()
    elif not isinstance(values, (list, tuple)):
        raise FixedPointError(f"{name} must be a list")
    else:
        ordered = tuple(sorted(_cid(item, name) for item in values))
    if len(ordered) != len(set(ordered)):
        raise FixedPointError(f"{name} must not contain duplicates")
    if required and not ordered:
        raise FixedPointError(f"{name} must not be empty")
    if len(ordered) > MAX_EVIDENCE_CIDS:
        raise FixedPointError(f"{name} exceeds maximum length")
    return ordered


def _pop_authority_flags(payload: dict[str, Any], name: str) -> None:
    for flag in _AUTHORITY_FLAG_NAMES:
        if flag not in payload:
            continue
        if payload.pop(flag) is not False:
            raise FixedPointError(f"{name} cannot claim {flag}")


def _reject_non_admitting(payload: Mapping[str, Any], name: str) -> None:
    present = _NON_ADMITTING_EVIDENCE & set(payload)
    if present:
        raise FixedPointError(
            f"{name} cannot admit a fixed point from {sorted(present)}"
        )
    booleans = _FORBIDDEN_BOOLEAN_KEYS & set(payload)
    for key in booleans:
        value = payload[key]
        if value is True:
            raise FixedPointError(
                f"{name} cannot admit boolean or prebuilt fixed point via {key}"
            )
        if value not in (False, None, ""):
            raise FixedPointError(
                f"{name} cannot admit boolean or prebuilt fixed point via {key}"
            )
    stores = _FORBIDDEN_STORE_KEYS & set(payload)
    if stores:
        raise FixedPointError(
            f"{name} cannot use federation FixedPointStore: {sorted(stores)}"
        )


def fixed_point_cid_profile() -> dict[str, str]:
    return {
        "profile_id": "ipfs_accelerate_py.cid-utils@1",
        "codec": "dag-json",
        "rule": (
            "CID identifies exact canonical bytes under declared codec/profile, "
            "not universal meaning"
        ),
    }


def _slice_cids(values: Any, name: str) -> dict[str, str]:
    payload = _mapping(values, name)
    extra = set(payload) - DECLARED_REBUILD_SLICES
    missing = DECLARED_REBUILD_SLICES - set(payload)
    if extra:
        raise FixedPointError(f"{name} has unknown rebuild slices: {sorted(extra)}")
    if missing:
        raise FixedPointError(
            f"{name} requires live rebuild of every slice: {sorted(missing)}"
        )
    return {key: _cid(payload[key], f"{name}.{key}") for key in REBUILD_SLICE_ORDER}


@dataclass(frozen=True, slots=True)
class RebuildSliceSet:
    """Live identities of the seven slices rebuilt after a wave."""

    semantic_state_cid: str
    graph_cid: str
    sccs_cid: str
    contracts_cid: str
    frontier_cid: str
    partitions_cid: str
    selections_cid: str

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "semantic_state_cid",
            "graph_cid",
            "sccs_cid",
            "contracts_cid",
            "frontier_cid",
            "partitions_cid",
            "selections_cid",
            "slice_set_cid",
            "live_rebuild",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "semantic_state_cid", _cid(self.semantic_state_cid, "semantic_state_cid")
        )
        object.__setattr__(self, "graph_cid", _cid(self.graph_cid, "graph_cid"))
        object.__setattr__(self, "sccs_cid", _cid(self.sccs_cid, "sccs_cid"))
        object.__setattr__(
            self, "contracts_cid", _cid(self.contracts_cid, "contracts_cid")
        )
        object.__setattr__(self, "frontier_cid", _cid(self.frontier_cid, "frontier_cid"))
        object.__setattr__(
            self, "partitions_cid", _cid(self.partitions_cid, "partitions_cid")
        )
        object.__setattr__(
            self, "selections_cid", _cid(self.selections_cid, "selections_cid")
        )

    @property
    def live_rebuild(self) -> bool:
        return True

    def as_mapping(self) -> dict[str, str]:
        return {
            "semantic_state": self.semantic_state_cid,
            "graph": self.graph_cid,
            "sccs": self.sccs_cid,
            "contracts": self.contracts_cid,
            "frontier": self.frontier_cid,
            "partitions": self.partitions_cid,
            "selections": self.selections_cid,
        }

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": REBUILD_SLICE_SET_SCHEMA,
            "interface": REBUILD_SLICE_SET_INTERFACE,
            "semantic_state_cid": self.semantic_state_cid,
            "graph_cid": self.graph_cid,
            "sccs_cid": self.sccs_cid,
            "contracts_cid": self.contracts_cid,
            "frontier_cid": self.frontier_cid,
            "partitions_cid": self.partitions_cid,
            "selections_cid": self.selections_cid,
            "live_rebuild": True,
        }
        _require_dag_json(payload, self.__class__.__name__)
        return payload

    @property
    def slice_set_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["slice_set_cid"] = self.slice_set_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RebuildSliceSet":
        _reject_excluded(data, cls.__name__)
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("slice_set_cid")
        if payload.pop("schema") != REBUILD_SLICE_SET_SCHEMA:
            raise FixedPointError("unsupported RebuildSliceSet schema")
        if payload.pop("interface") != REBUILD_SLICE_SET_INTERFACE:
            raise FixedPointError("unsupported RebuildSliceSet interface")
        if payload.pop("live_rebuild") is not True:
            raise FixedPointError("rebuild slices must be a live rebuild")
        result = cls(**payload)
        _verify_cid(claimed, result.slice_set_cid, "RebuildSliceSet slice_set_cid")
        return result

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | "RebuildSliceSet") -> "RebuildSliceSet":
        if isinstance(values, RebuildSliceSet):
            return values
        if "slice_set_cid" in values:
            return cls.from_dict(values)
        slices = _slice_cids(values, "rebuild")
        return cls(
            semantic_state_cid=slices["semantic_state"],
            graph_cid=slices["graph"],
            sccs_cid=slices["sccs"],
            contracts_cid=slices["contracts"],
            frontier_cid=slices["frontier"],
            partitions_cid=slices["partitions"],
            selections_cid=slices["selections"],
        )


@dataclass(frozen=True, slots=True)
class MergeReceipt:
    """Current-authority merge receipt bound to one wave. Not worker approval."""

    merge_cid: str
    wave_cid: str
    tree_id: str
    disposition: str
    validation_cid: str = ""

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "merge_cid",
            "wave_cid",
            "tree_id",
            "disposition",
            "validation_cid",
            "can_authorize_transition",
            "can_authorize_completion",
            "worker_self_approval",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "merge_cid", _cid(self.merge_cid, "merge_cid"))
        object.__setattr__(self, "wave_cid", _cid(self.wave_cid, "wave_cid"))
        object.__setattr__(self, "tree_id", _tree_id(self.tree_id))
        object.__setattr__(
            self,
            "disposition",
            _enum(self.disposition, MergeDisposition, "disposition"),
        )
        object.__setattr__(
            self, "validation_cid", _optional_cid(self.validation_cid, "validation_cid")
        )

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": MERGE_RECEIPT_SCHEMA,
            "interface": MERGE_RECEIPT_INTERFACE,
            "merge_cid": self.merge_cid,
            "wave_cid": self.wave_cid,
            "tree_id": self.tree_id,
            "disposition": self.disposition,
            "validation_cid": self.validation_cid,
            "can_authorize_transition": False,
            "can_authorize_completion": False,
            "worker_self_approval": False,
        }
        _require_dag_json(payload, self.__class__.__name__)
        return payload

    def to_dict(self) -> dict[str, Any]:
        return self.identity_payload()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MergeReceipt":
        _reject_excluded(data, cls.__name__)
        payload = _closed(data, cls._FIELDS, cls.__name__)
        if payload.pop("schema") != MERGE_RECEIPT_SCHEMA:
            raise FixedPointError("unsupported SparMergeReceipt schema")
        if payload.pop("interface") != MERGE_RECEIPT_INTERFACE:
            raise FixedPointError("unsupported SparMergeReceipt interface")
        _pop_authority_flags(payload, "MergeReceipt")
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class WaveDisposition:
    """Accepted or rejected wave bound to SPAR-025/027 evidence."""

    wave_cid: str
    outcome: str
    wave_status: str
    validation_status: str
    validation_cid: str = ""
    packet_cids: Sequence[str] = ()

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "wave_cid",
            "outcome",
            "wave_status",
            "validation_status",
            "validation_cid",
            "packet_cids",
            "disposition_cid",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "wave_cid", _cid(self.wave_cid, "wave_cid"))
        object.__setattr__(self, "outcome", _enum(self.outcome, WaveOutcome, "outcome"))
        object.__setattr__(self, "wave_status", _text(self.wave_status, "wave_status"))
        if self.wave_status not in ALLOWED_WAVE_STATUSES:
            raise FixedPointError(f"unknown wave_status: {self.wave_status}")
        object.__setattr__(
            self, "validation_status", _text(self.validation_status, "validation_status")
        )
        if self.validation_status not in ALLOWED_VALIDATION_STATUSES:
            raise FixedPointError(
                f"unknown validation_status: {self.validation_status}"
            )
        object.__setattr__(
            self, "validation_cid", _optional_cid(self.validation_cid, "validation_cid")
        )
        object.__setattr__(
            self,
            "packet_cids",
            _cids(list(self.packet_cids), "packet_cids"),
        )
        if self.outcome == WaveOutcome.ACCEPTED.value:
            if self.wave_status != "applied":
                raise FixedPointError("accepted wave requires applied wave status")
            if self.validation_status != "validated":
                raise FixedPointError(
                    "accepted wave requires validated translation-validation status"
                )
        if (
            self.outcome == WaveOutcome.REJECTED.value
            and self.wave_status not in REJECTED_WAVE_STATUSES
            and self.validation_status not in REJECTED_VALIDATION_STATUSES
            and self.validation_status not in TERMINAL_VALIDATION_STATUSES
        ):
            raise FixedPointError(
                "rejected wave requires rejected/rolled_back wave or rejected validation"
            )

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": WAVE_DISPOSITION_SCHEMA,
            "interface": WAVE_DISPOSITION_INTERFACE,
            "wave_cid": self.wave_cid,
            "outcome": self.outcome,
            "wave_status": self.wave_status,
            "validation_status": self.validation_status,
            "validation_cid": self.validation_cid,
            "packet_cids": list(self.packet_cids),
        }
        _require_dag_json(payload, self.__class__.__name__)
        return payload

    @property
    def disposition_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["disposition_cid"] = self.disposition_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "WaveDisposition":
        _reject_excluded(data, cls.__name__)
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("disposition_cid")
        if payload.pop("schema") != WAVE_DISPOSITION_SCHEMA:
            raise FixedPointError("unsupported WaveDisposition schema")
        if payload.pop("interface") != WAVE_DISPOSITION_INTERFACE:
            raise FixedPointError("unsupported WaveDisposition interface")
        result = cls(**payload)
        _verify_cid(claimed, result.disposition_cid, "WaveDisposition disposition_cid")
        return result


@dataclass(frozen=True, slots=True)
class TypedTerminal:
    """Typed stop that is never success or completion."""

    kind: str
    reason: str

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "kind",
            "reason",
            "terminal_cid",
            "can_authorize_completion",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _enum(self.kind, TerminalKind, "kind"))
        object.__setattr__(self, "reason", _text(self.reason, "reason"))

    @property
    def can_authorize_completion(self) -> bool:
        return False

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": TYPED_TERMINAL_SCHEMA,
            "interface": TYPED_TERMINAL_INTERFACE,
            "kind": self.kind,
            "reason": self.reason,
            "can_authorize_completion": False,
        }
        _require_dag_json(payload, self.__class__.__name__)
        return payload

    @property
    def terminal_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["terminal_cid"] = self.terminal_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TypedTerminal":
        _reject_excluded(data, cls.__name__)
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("terminal_cid")
        if payload.pop("schema") != TYPED_TERMINAL_SCHEMA:
            raise FixedPointError("unsupported TypedTerminal schema")
        if payload.pop("interface") != TYPED_TERMINAL_INTERFACE:
            raise FixedPointError("unsupported TypedTerminal interface")
        if payload.pop("can_authorize_completion") is not False:
            raise FixedPointError("typed terminal cannot authorize completion")
        result = cls(**payload)
        _verify_cid(claimed, result.terminal_cid, "TypedTerminal terminal_cid")
        return result


@dataclass(frozen=True, slots=True)
class ComposedRoot:
    """Content-addressed composition of live rebuild slices and merge receipts."""

    tree_id: str
    slices: RebuildSliceSet
    merge_receipt_cids: Sequence[str]
    wave_cid: str
    wave_outcome: str

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "tree_id",
            "slices",
            "merge_receipt_cids",
            "wave_cid",
            "wave_outcome",
            "composed_root_cid",
            "live_rebuild",
            "federation_store_is_authority",
            "boolean_fixed_point_is_authority",
            "prebuilt_fixed_point_is_authority",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "tree_id", _tree_id(self.tree_id))
        slices = self.slices
        if not isinstance(slices, RebuildSliceSet):
            slices = RebuildSliceSet.from_mapping(slices)
        object.__setattr__(self, "slices", slices)
        object.__setattr__(
            self,
            "merge_receipt_cids",
            _cids(list(self.merge_receipt_cids), "merge_receipt_cids"),
        )
        object.__setattr__(self, "wave_cid", _cid(self.wave_cid, "wave_cid"))
        object.__setattr__(
            self, "wave_outcome", _enum(self.wave_outcome, WaveOutcome, "wave_outcome")
        )

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": COMPOSED_ROOT_SCHEMA,
            "interface": COMPOSED_ROOT_INTERFACE,
            "tree_id": self.tree_id,
            "slices": self.slices.to_dict(),
            "merge_receipt_cids": list(self.merge_receipt_cids),
            "wave_cid": self.wave_cid,
            "wave_outcome": self.wave_outcome,
            "live_rebuild": True,
            "federation_store_is_authority": False,
            "boolean_fixed_point_is_authority": False,
            "prebuilt_fixed_point_is_authority": False,
        }
        _require_dag_json(payload, self.__class__.__name__)
        return payload

    @property
    def composed_root_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["composed_root_cid"] = self.composed_root_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ComposedRoot":
        _reject_excluded(data, cls.__name__)
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("composed_root_cid")
        if payload.pop("schema") != COMPOSED_ROOT_SCHEMA:
            raise FixedPointError("unsupported ComposedRoot schema")
        if payload.pop("interface") != COMPOSED_ROOT_INTERFACE:
            raise FixedPointError("unsupported ComposedRoot interface")
        if payload.pop("live_rebuild") is not True:
            raise FixedPointError("composed root requires a live rebuild")
        _pop_authority_flags(payload, "ComposedRoot")
        payload["slices"] = RebuildSliceSet.from_dict(payload["slices"])
        result = cls(**payload)
        _verify_cid(claimed, result.composed_root_cid, "ComposedRoot composed_root_cid")
        return result


def _merge_cid_of(item: Mapping[str, Any] | MergeReceipt | str) -> str:
    if isinstance(item, MergeReceipt):
        return item.merge_cid
    if isinstance(item, Mapping):
        if "schema" in item:
            return MergeReceipt.from_dict(item).merge_cid
        return _cid(item.get("merge_cid"), "merge_cid")
    return _cid(item, "merge_cid")


def compose_current_roots(
    *,
    tree_id: str,
    rebuild: Mapping[str, Any] | RebuildSliceSet,
    merge_receipts: Sequence[Mapping[str, Any] | MergeReceipt | str] = (),
    wave_cid: str,
    wave_outcome: str,
) -> ComposedRoot:
    """Compose live rebuild slices with current merge receipts. Nomination-only."""

    return ComposedRoot(
        tree_id=tree_id,
        slices=RebuildSliceSet.from_mapping(rebuild),
        merge_receipt_cids=tuple(_merge_cid_of(item) for item in merge_receipts),
        wave_cid=wave_cid,
        wave_outcome=wave_outcome,
    )


def _parse_merge_receipts(
    values: Any,
    *,
    tree_id: str,
    wave_cid: str,
) -> tuple[MergeReceipt, ...]:
    if values in (None, (), []):
        return ()
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise FixedPointError("merge_receipts must be a list")
    receipts: list[MergeReceipt] = []
    seen: set[str] = set()
    for item in values:
        if isinstance(item, MergeReceipt):
            receipt = item
        elif isinstance(item, Mapping):
            payload = dict(item)
            if "schema" in payload:
                receipt = MergeReceipt.from_dict(payload)
            else:
                receipt = MergeReceipt(
                    merge_cid=payload.get("merge_cid"),
                    wave_cid=payload.get("wave_cid") or wave_cid,
                    tree_id=payload.get("tree_id") or tree_id,
                    disposition=payload.get("disposition"),
                    validation_cid=payload.get("validation_cid") or "",
                )
        else:
            raise FixedPointError("merge_receipts items must be objects")
        if receipt.tree_id != tree_id:
            raise FixedPointError("merge receipt tree_id does not match")
        if receipt.wave_cid != wave_cid:
            raise FixedPointError("merge receipt wave_cid does not match")
        if receipt.merge_cid in seen:
            raise FixedPointError("merge_receipts must not contain duplicates")
        seen.add(receipt.merge_cid)
        receipts.append(receipt)
    if len(receipts) > MAX_RECEIPTS:
        raise FixedPointError("merge_receipts exceeds maximum length")
    return tuple(sorted(receipts, key=lambda item: item.merge_cid))


def _derive_outcome(
    *,
    wave_status: str,
    validation_status: str,
    merge_receipts: Sequence[MergeReceipt],
) -> str:
    accepted_merges = [
        item for item in merge_receipts if item.disposition == MergeDisposition.ACCEPTED.value
    ]
    rejected_merges = [
        item for item in merge_receipts if item.disposition == MergeDisposition.REJECTED.value
    ]
    if accepted_merges and rejected_merges:
        raise FixedPointError("merge receipts cannot mix accepted and rejected dispositions")
    if accepted_merges:
        if wave_status != "applied":
            raise FixedPointError("accepted merge requires applied wave status")
        if validation_status != "validated":
            raise FixedPointError(
                "accepted merge requires validated translation-validation status"
            )
        return WaveOutcome.ACCEPTED.value
    if (
        rejected_merges
        or wave_status in REJECTED_WAVE_STATUSES
        or validation_status in REJECTED_VALIDATION_STATUSES
        or validation_status in TERMINAL_VALIDATION_STATUSES
    ):
        return WaveOutcome.REJECTED.value
    raise FixedPointError(
        "wave outcome requires an accepted merge receipt or a rejected wave/validation"
    )


def _parse_terminal(value: Any) -> TypedTerminal | None:
    if value in (None, "", {}):
        return None
    if isinstance(value, TypedTerminal):
        return value
    if isinstance(value, Mapping):
        payload = dict(value)
        if "schema" in payload:
            return TypedTerminal.from_dict(payload)
        return TypedTerminal(kind=payload.get("kind"), reason=payload.get("reason"))
    raise FixedPointError("terminal must be an object")


@dataclass(frozen=True, slots=True)
class FixedPointIteration:
    """One live rebuild epoch after an accepted or rejected wave."""

    tree_id: str
    iteration: int
    wave_disposition: WaveDisposition
    composed_root: ComposedRoot
    prior_composed_root_cid: str = ""
    remaining_mandatory_task_cids: Sequence[str] = ()
    pending_merge_cids: Sequence[str] = ()
    negative_evidence_cids: Sequence[str] = ()
    terminal: TypedTerminal | None = None

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "tree_id",
            "iteration",
            "wave_disposition",
            "composed_root",
            "prior_composed_root_cid",
            "remaining_mandatory_task_cids",
            "pending_merge_cids",
            "negative_evidence_cids",
            "terminal",
            "iteration_cid",
            "live_rebuild",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "tree_id", _tree_id(self.tree_id))
        object.__setattr__(
            self,
            "iteration",
            _int(self.iteration, "iteration", minimum=1, maximum=MAX_ITERATIONS + 1),
        )
        disposition = self.wave_disposition
        if not isinstance(disposition, WaveDisposition):
            disposition = WaveDisposition.from_dict(disposition)
        object.__setattr__(self, "wave_disposition", disposition)
        composed = self.composed_root
        if not isinstance(composed, ComposedRoot):
            composed = ComposedRoot.from_dict(composed)
        object.__setattr__(self, "composed_root", composed)
        if composed.tree_id != self.tree_id:
            raise FixedPointError("composed root tree_id does not match")
        if composed.wave_cid != disposition.wave_cid:
            raise FixedPointError("composed root wave_cid does not match")
        if composed.wave_outcome != disposition.outcome:
            raise FixedPointError("composed root wave_outcome does not match")
        object.__setattr__(
            self,
            "prior_composed_root_cid",
            _optional_cid(self.prior_composed_root_cid, "prior_composed_root_cid"),
        )
        object.__setattr__(
            self,
            "remaining_mandatory_task_cids",
            _cids(
                list(self.remaining_mandatory_task_cids),
                "remaining_mandatory_task_cids",
            ),
        )
        object.__setattr__(
            self,
            "pending_merge_cids",
            _cids(list(self.pending_merge_cids), "pending_merge_cids"),
        )
        object.__setattr__(
            self,
            "negative_evidence_cids",
            _cids(list(self.negative_evidence_cids), "negative_evidence_cids"),
        )
        terminal = self.terminal
        if terminal not in (None,):
            if not isinstance(terminal, TypedTerminal):
                terminal = _parse_terminal(terminal)
            object.__setattr__(self, "terminal", terminal)

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": FIXED_POINT_ITERATION_SCHEMA,
            "interface": FIXED_POINT_ITERATION_INTERFACE,
            "tree_id": self.tree_id,
            "iteration": self.iteration,
            "wave_disposition": self.wave_disposition.to_dict(),
            "composed_root": self.composed_root.to_dict(),
            "prior_composed_root_cid": self.prior_composed_root_cid,
            "remaining_mandatory_task_cids": list(self.remaining_mandatory_task_cids),
            "pending_merge_cids": list(self.pending_merge_cids),
            "negative_evidence_cids": list(self.negative_evidence_cids),
            "terminal": None if self.terminal is None else self.terminal.to_dict(),
            "live_rebuild": True,
        }
        _require_dag_json(payload, self.__class__.__name__)
        return payload

    @property
    def iteration_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["iteration_cid"] = self.iteration_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FixedPointIteration":
        _reject_excluded(data, cls.__name__)
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("iteration_cid")
        if payload.pop("schema") != FIXED_POINT_ITERATION_SCHEMA:
            raise FixedPointError("unsupported FixedPointIteration schema")
        if payload.pop("interface") != FIXED_POINT_ITERATION_INTERFACE:
            raise FixedPointError("unsupported FixedPointIteration interface")
        if payload.pop("live_rebuild") is not True:
            raise FixedPointError("iteration requires a live rebuild")
        payload["wave_disposition"] = WaveDisposition.from_dict(
            payload["wave_disposition"]
        )
        payload["composed_root"] = ComposedRoot.from_dict(payload["composed_root"])
        if payload["terminal"] is not None:
            payload["terminal"] = TypedTerminal.from_dict(payload["terminal"])
        result = cls(**payload)
        _verify_cid(claimed, result.iteration_cid, "FixedPointIteration iteration_cid")
        return result


def _decide_status(iteration: FixedPointIteration, *, max_iterations: int) -> str:
    if iteration.terminal is not None:
        return ControllerStatus.TYPED_TERMINAL.value
    if iteration.wave_disposition.validation_status == "unsupported":
        return ControllerStatus.TYPED_TERMINAL.value
    if iteration.iteration > max_iterations:
        return ControllerStatus.TYPED_TERMINAL.value
    unchanged = (
        bool(iteration.prior_composed_root_cid)
        and iteration.prior_composed_root_cid == iteration.composed_root.composed_root_cid
    )
    if (
        unchanged
        and not iteration.remaining_mandatory_task_cids
        and not iteration.pending_merge_cids
    ):
        return ControllerStatus.NOMINATED_FIXED_POINT.value
    return ControllerStatus.CONTINUE.value


def _coerce_terminal(
    iteration: FixedPointIteration,
    *,
    status: str,
    max_iterations: int,
) -> TypedTerminal | None:
    if iteration.terminal is not None:
        return iteration.terminal
    if status != ControllerStatus.TYPED_TERMINAL.value:
        return None
    if iteration.wave_disposition.validation_status == "unsupported":
        return TypedTerminal(
            kind=TerminalKind.UNSUPPORTED.value,
            reason="translation validation reported unsupported required behavior",
        )
    if iteration.iteration > max_iterations:
        return TypedTerminal(
            kind=TerminalKind.MAX_ITERATIONS.value,
            reason="live rebuild exceeded max_iterations without unchanged composed roots",
        )
    raise FixedPointError("typed terminal is missing a declared kind")


@dataclass(frozen=True, slots=True)
class FixedPointReceipt:
    """Nomination-only SPAR accepted-wave/fixed-point receipt."""

    tree_id: str
    iteration: FixedPointIteration
    status: str
    analyzer_id: str = ANALYZER_ID
    worktree_id: str = ""
    lease_id: str = ""
    fence_id: str = ""

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "tree_id",
            "iteration",
            "status",
            "analyzer_id",
            "worktree_id",
            "lease_id",
            "fence_id",
            "receipt_cid",
            "nominated",
            "accepted",
            "live_rebuild",
            "controller_is_nomination_only",
            "can_authorize_transition",
            "can_authorize_completion",
            "can_create_authority",
            "can_accept_fixed_point",
            "projection_is_authority",
            "federation_store_is_authority",
            "boolean_fixed_point_is_authority",
            "prebuilt_fixed_point_is_authority",
            "worker_self_approval",
            "writes_repository",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "tree_id", _tree_id(self.tree_id))
        iteration = self.iteration
        if not isinstance(iteration, FixedPointIteration):
            iteration = FixedPointIteration.from_dict(iteration)
        object.__setattr__(self, "iteration", iteration)
        if iteration.tree_id != self.tree_id:
            raise FixedPointError("receipt tree_id does not match iteration")
        object.__setattr__(
            self, "status", _enum(self.status, ControllerStatus, "status")
        )
        object.__setattr__(self, "analyzer_id", _text(self.analyzer_id, "analyzer_id"))
        if self.analyzer_id != ANALYZER_ID:
            raise FixedPointError("analyzer_id must remain SPAR-038")
        object.__setattr__(
            self, "worktree_id", _text(self.worktree_id, "worktree_id", empty=True)
        )
        object.__setattr__(self, "lease_id", _text(self.lease_id, "lease_id", empty=True))
        object.__setattr__(self, "fence_id", _text(self.fence_id, "fence_id", empty=True))
        if self.status == ControllerStatus.NOMINATED_FIXED_POINT.value:
            if iteration.terminal is not None:
                raise FixedPointError("nominated fixed point cannot carry a typed terminal")
            if iteration.remaining_mandatory_task_cids:
                raise FixedPointError(
                    "nominated fixed point requires zero remaining mandatory tasks"
                )
            if iteration.pending_merge_cids:
                raise FixedPointError("nominated fixed point requires zero pending merges")
            if (
                not iteration.prior_composed_root_cid
                or iteration.prior_composed_root_cid
                != iteration.composed_root.composed_root_cid
            ):
                raise FixedPointError(
                    "nominated fixed point requires two unchanged live composed roots"
                )
        if self.status == ControllerStatus.TYPED_TERMINAL.value:
            if iteration.terminal is None:
                raise FixedPointError("typed terminal status requires a typed terminal")
        if self.status == ControllerStatus.CONTINUE.value and iteration.terminal is not None:
            raise FixedPointError("continue status cannot carry a typed terminal")

    @property
    def nominated(self) -> bool:
        return self.status == ControllerStatus.NOMINATED_FIXED_POINT.value

    @property
    def accepted(self) -> bool:
        return False

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
    def can_accept_fixed_point(self) -> bool:
        return False

    @property
    def projection_is_authority(self) -> bool:
        return False

    @property
    def federation_store_is_authority(self) -> bool:
        return False

    @property
    def boolean_fixed_point_is_authority(self) -> bool:
        return False

    @property
    def prebuilt_fixed_point_is_authority(self) -> bool:
        return False

    @property
    def worker_self_approval(self) -> bool:
        return False

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": FIXED_POINT_RECEIPT_SCHEMA,
            "interface": FIXED_POINT_RECEIPT_INTERFACE,
            "tree_id": self.tree_id,
            "iteration": self.iteration.to_dict(),
            "status": self.status,
            "analyzer_id": ANALYZER_ID,
            "worktree_id": self.worktree_id,
            "lease_id": self.lease_id,
            "fence_id": self.fence_id,
            "nominated": self.nominated,
            "accepted": False,
            "live_rebuild": True,
            "controller_is_nomination_only": True,
            "can_authorize_transition": False,
            "can_authorize_completion": False,
            "can_create_authority": False,
            "can_accept_fixed_point": False,
            "projection_is_authority": False,
            "federation_store_is_authority": False,
            "boolean_fixed_point_is_authority": False,
            "prebuilt_fixed_point_is_authority": False,
            "worker_self_approval": False,
            "writes_repository": False,
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
    def from_dict(cls, data: Mapping[str, Any]) -> "FixedPointReceipt":
        _reject_excluded(data, cls.__name__)
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("receipt_cid")
        if payload.pop("schema") != FIXED_POINT_RECEIPT_SCHEMA:
            raise FixedPointError("unsupported FixedPointReceipt schema")
        if payload.pop("interface") != FIXED_POINT_RECEIPT_INTERFACE:
            raise FixedPointError("unsupported FixedPointReceipt interface")
        if payload.pop("accepted") is not False:
            raise FixedPointError("workers cannot self-approve a fixed point")
        if payload.pop("live_rebuild") is not True:
            raise FixedPointError("fixed-point receipt requires a live rebuild")
        if payload.pop("controller_is_nomination_only") is not True:
            raise FixedPointError("controller must remain nomination_only")
        if payload.pop("writes_repository") is not False:
            raise FixedPointError("controller cannot write the repository")
        nominated = payload.pop("nominated")
        _pop_authority_flags(payload, "FixedPointReceipt")
        payload["iteration"] = FixedPointIteration.from_dict(payload["iteration"])
        result = cls(**payload)
        if nominated is not result.nominated:
            raise FixedPointError("nominated flag does not match status")
        _verify_cid(claimed, result.receipt_cid, "FixedPointReceipt receipt_cid")
        return result


def rebuild_after_wave(evidence: Mapping[str, Any]) -> FixedPointIteration:
    """Rebuild all seven slices after an accepted or rejected wave."""

    payload = _mapping(evidence, "fixed-point evidence")
    _reject_non_admitting(payload, "fixed-point evidence")
    if payload.get("mutate") is True:
        raise FixedPointError("controller cannot mutate; dry-run never mutates")
    tree_id = _tree_id(payload.get("tree_id"))
    rebuild = payload.get("rebuild")
    if rebuild in (None, "", {}):
        raise FixedPointError(
            "live rebuild of semantic state, graph, SCCs, contracts, frontier, "
            "partitions, and selections is required after every accepted or rejected wave"
        )
    current_roots = payload.get("current_roots")
    if current_roots not in (None, "", {}):
        _slice_cids(current_roots, "current_roots")
    slices = RebuildSliceSet.from_mapping(rebuild)
    wave_payload = _mapping(payload.get("wave"), "wave")
    wave_cid = _cid(wave_payload.get("wave_cid") or wave_payload.get("receipt_cid"), "wave_cid")
    wave_status = _text(wave_payload.get("status"), "wave status")
    if wave_status not in ALLOWED_WAVE_STATUSES:
        raise FixedPointError(f"unknown wave_status: {wave_status}")
    packet_cids = _cids(list(wave_payload.get("packet_cids") or ()), "packet_cids")
    validation_payload = payload.get("validation")
    if validation_payload in (None, "", {}):
        validation_status = "incomplete"
        validation_cid = ""
    else:
        validation_map = _mapping(validation_payload, "validation")
        validation_status = _text(validation_map.get("status"), "validation status")
        validation_cid = _optional_cid(
            validation_map.get("validation_cid") or validation_map.get("result_cid") or "",
            "validation_cid",
        )
    if validation_status not in ALLOWED_VALIDATION_STATUSES:
        raise FixedPointError(f"unknown validation_status: {validation_status}")
    merge_receipts = _parse_merge_receipts(
        payload.get("merge_receipts"),
        tree_id=tree_id,
        wave_cid=wave_cid,
    )
    outcome = _derive_outcome(
        wave_status=wave_status,
        validation_status=validation_status,
        merge_receipts=merge_receipts,
    )
    if outcome == WaveOutcome.ACCEPTED.value and not merge_receipts:
        raise FixedPointError("accepted wave must compose a current merge receipt")
    if outcome == WaveOutcome.ACCEPTED.value and not REBUILD_AFTER_ACCEPTED_WAVE:
        raise FixedPointError("accepted wave must rebuild")
    if outcome == WaveOutcome.REJECTED.value and not REBUILD_AFTER_REJECTED_WAVE:
        raise FixedPointError("rejected wave must rebuild")
    disposition = WaveDisposition(
        wave_cid=wave_cid,
        outcome=outcome,
        wave_status=wave_status,
        validation_status=validation_status,
        validation_cid=validation_cid,
        packet_cids=packet_cids,
    )
    composed = compose_current_roots(
        tree_id=tree_id,
        rebuild=slices,
        merge_receipts=merge_receipts,
        wave_cid=wave_cid,
        wave_outcome=outcome,
    )
    negative = list(payload.get("negative_evidence_cids") or ())
    if outcome == WaveOutcome.REJECTED.value:
        if wave_cid not in negative:
            negative.append(wave_cid)
        if validation_cid and validation_cid not in negative:
            negative.append(validation_cid)
    terminal = _parse_terminal(payload.get("terminal"))
    return FixedPointIteration(
        tree_id=tree_id,
        iteration=_int(
            payload.get("iteration", 1),
            "iteration",
            minimum=1,
            maximum=MAX_ITERATIONS + 1,
        ),
        wave_disposition=disposition,
        composed_root=composed,
        prior_composed_root_cid=payload.get("prior_composed_root_cid") or "",
        remaining_mandatory_task_cids=list(
            payload.get("remaining_mandatory_task_cids") or ()
        ),
        pending_merge_cids=list(payload.get("pending_merge_cids") or ()),
        negative_evidence_cids=negative,
        terminal=terminal,
    )


def run_fixed_point_controller(
    evidence: Mapping[str, Any],
    *,
    mutate: bool = False,
) -> FixedPointReceipt:
    """Run one live rebuild epoch. Nomination-only; never accepts a fixed point."""

    if mutate is True:
        raise FixedPointError("controller cannot mutate; dry-run never mutates")
    payload = _mapping(evidence, "fixed-point evidence")
    if payload.get("mutate") is True:
        raise FixedPointError("controller cannot mutate; dry-run never mutates")
    iteration = rebuild_after_wave(payload)
    max_iterations = _int(
        payload.get("max_iterations", DEFAULT_MAX_ITERATIONS),
        "max_iterations",
        minimum=1,
        maximum=MAX_ITERATIONS,
    )
    status = _decide_status(iteration, max_iterations=max_iterations)
    terminal = _coerce_terminal(
        iteration, status=status, max_iterations=max_iterations
    )
    if terminal is not iteration.terminal:
        iteration = FixedPointIteration(
            tree_id=iteration.tree_id,
            iteration=iteration.iteration,
            wave_disposition=iteration.wave_disposition,
            composed_root=iteration.composed_root,
            prior_composed_root_cid=iteration.prior_composed_root_cid,
            remaining_mandatory_task_cids=iteration.remaining_mandatory_task_cids,
            pending_merge_cids=iteration.pending_merge_cids,
            negative_evidence_cids=iteration.negative_evidence_cids,
            terminal=terminal,
        )
        status = _decide_status(iteration, max_iterations=max_iterations)
    return FixedPointReceipt(
        tree_id=iteration.tree_id,
        iteration=iteration,
        status=status,
        worktree_id=str(payload.get("worktree_id") or ""),
        lease_id=str(payload.get("lease_id") or ""),
        fence_id=str(payload.get("fence_id") or ""),
    )


def dry_run_fixed_point(evidence: Mapping[str, Any]) -> FixedPointReceipt:
    """Deterministic dry-run. Never mutates and never accepts a fixed point."""

    return run_fixed_point_controller(evidence, mutate=False)


class FixedPointRemodularizationController:
    """SPAR-038 accepted-wave/fixed-point controller. Nomination-only."""

    interface: ClassVar[str] = FIXED_POINT_REMODULARIZATION_CONTROLLER_INTERFACE
    schema: ClassVar[str] = FIXED_POINT_REMODULARIZATION_CONTROLLER_SCHEMA
    analyzer_id: ClassVar[str] = ANALYZER_ID

    def compose(
        self,
        *,
        tree_id: str,
        rebuild: Mapping[str, Any] | RebuildSliceSet,
        merge_receipts: Sequence[Mapping[str, Any] | MergeReceipt] = (),
        wave_cid: str,
        wave_outcome: str,
    ) -> ComposedRoot:
        return compose_current_roots(
            tree_id=tree_id,
            rebuild=rebuild,
            merge_receipts=merge_receipts,
            wave_cid=wave_cid,
            wave_outcome=wave_outcome,
        )

    def rebuild(self, evidence: Mapping[str, Any]) -> FixedPointIteration:
        return rebuild_after_wave(evidence)

    def run(
        self,
        evidence: Mapping[str, Any],
        *,
        mutate: bool = False,
    ) -> FixedPointReceipt:
        return run_fixed_point_controller(evidence, mutate=mutate)

    def dry_run(self, evidence: Mapping[str, Any]) -> FixedPointReceipt:
        return dry_run_fixed_point(evidence)


def encode_canonical_receipt(receipt: FixedPointReceipt) -> dict[str, Any]:
    return receipt.to_dict()


def decode_canonical_receipt(payload: Mapping[str, Any]) -> FixedPointReceipt:
    return FixedPointReceipt.from_dict(payload)


def encode_canonical_iteration(iteration: FixedPointIteration) -> dict[str, Any]:
    return iteration.to_dict()


def decode_canonical_iteration(payload: Mapping[str, Any]) -> FixedPointIteration:
    return FixedPointIteration.from_dict(payload)


def encode_canonical_composed_root(root: ComposedRoot) -> dict[str, Any]:
    return root.to_dict()


def decode_canonical_composed_root(payload: Mapping[str, Any]) -> ComposedRoot:
    return ComposedRoot.from_dict(payload)


def provider_free_exports() -> tuple[str, ...]:
    return tuple(sorted(__all__))


def assert_not_competing_capsule_family() -> None:
    names = set(globals())
    overlap = names & _FORBIDDEN_CAPSULE_TYPE_NAMES
    if overlap:
        raise FixedPointError(
            f"fixed-point controller must not define competing types: {sorted(overlap)}"
        )
    if "FixedPointStore" in names:
        raise FixedPointError("Federation FixedPointStore is not a SPAR accepted-wave contract")


__all__ = [
    "ACCEPTED_WAVE_FIXED_POINT_INTERFACE",
    "ALLOWED_VALIDATION_STATUSES",
    "ALLOWED_WAVE_STATUSES",
    "ANALYZER_ID",
    "AUTHORITY",
    "AUTHORITY_OWNER",
    "BOOLEAN_FIXED_POINT_IS_AUTHORITY",
    "COMPOSED_ROOT_INTERFACE",
    "CONTROLLER_CAN_ACCEPT_FIXED_POINT",
    "CONTROLLER_CAN_AUTHORIZE_COMPLETION",
    "CONTROLLER_CAN_AUTHORIZE_TRANSITION",
    "CONTROLLER_CAN_CREATE_AUTHORITY",
    "CONTROLLER_IS_NOMINATION_ONLY",
    "CONTROLLER_WRITES_REPOSITORY",
    "ControllerStatus",
    "ComposedRoot",
    "DECLARED_CONTROLLER_STATUSES",
    "DECLARED_MERGE_DISPOSITIONS",
    "DECLARED_REBUILD_SLICES",
    "DECLARED_TERMINAL_KINDS",
    "DECLARED_WAVE_OUTCOMES",
    "DEFAULT_MAX_ITERATIONS",
    "DRY_RUN_IS_DETERMINISTIC",
    "DRY_RUN_MUTATES",
    "DUCKLAKE_IS_AUTHORITY",
    "EXISTING_ADAPTER_AUTHORITIES",
    "FEDERATION_FIXED_POINT_STORE_IS_AUTHORITY",
    "FIXED_POINT_CONTRACT_VERSION",
    "FIXED_POINT_ITERATION_INTERFACE",
    "FIXED_POINT_RECEIPT_INTERFACE",
    "FIXED_POINT_REMODULARIZATION_CONTROLLER_INTERFACE",
    "FORBIDDEN_FIXED_POINT_NAMES",
    "FixedPointError",
    "FixedPointIteration",
    "FixedPointReceipt",
    "FixedPointRemodularizationController",
    "GOAL_ID",
    "IDENTITY_EXCLUDED_FIELDS",
    "LIVE_REBUILD_REQUIRED",
    "MARKDOWN_IS_NOT_COMPLETION",
    "MAX_ITERATIONS",
    "MERGE_RECEIPT_INTERFACE",
    "MODEL_OUTPUT_IS_PROPOSAL_ONLY",
    "MergeDisposition",
    "MergeReceipt",
    "NEGATIVE_EVIDENCE_RETAINED",
    "PREBUILT_FIXED_POINT_IS_AUTHORITY",
    "PREDECESSOR_TASK_IDS",
    "PROGRAM",
    "PROJECTION_CLUSTERING_IS_AUTHORITY",
    "RAW_SOURCE_REQUIRED",
    "REBUILD_AFTER_ACCEPTED_WAVE",
    "REBUILD_AFTER_REJECTED_WAVE",
    "REBUILD_SLICE_ORDER",
    "REBUILD_SLICE_SET_INTERFACE",
    "RebuildSliceSet",
    "SOFT_SIGNALS_CANNOT_OVERRIDE_HARD_CONSTRAINTS",
    "TASK_ID",
    "TEST_PASS_IS_NOT_COMPLETION",
    "TWO_EPOCH_UNCHANGED_REQUIRED",
    "TYPED_TERMINAL_INTERFACE",
    "TerminalKind",
    "TypedTerminal",
    "VECTOR_SIMILARITY_IS_AUTHORITY",
    "WAVE_DISPOSITION_INTERFACE",
    "WORKER_SELF_APPROVAL",
    "WaveDisposition",
    "WaveOutcome",
    "assert_not_competing_capsule_family",
    "compose_current_roots",
    "decode_canonical_composed_root",
    "decode_canonical_iteration",
    "decode_canonical_receipt",
    "dry_run_fixed_point",
    "encode_canonical_composed_root",
    "encode_canonical_iteration",
    "encode_canonical_receipt",
    "fixed_point_cid_profile",
    "provider_free_exports",
    "rebuild_after_wave",
    "run_fixed_point_controller",
]
