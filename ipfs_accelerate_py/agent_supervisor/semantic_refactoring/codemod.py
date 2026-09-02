"""SPAR-020 CST-preserving move and extraction codemods.

This module extends current supervisor partition orchestration with
``CSTExtractionCodemod@1``.  It consumes SPAR-019
``RefactorTransformationPacket@1`` mappings plus required raw source and
applies deterministic bounded MOVE/extraction edits.

The current best available CST capability is ``parso`` as a parse helper;
``asttokens`` is a source-map parse helper only.  ``libcst`` is typed
unavailable under the sealed validation interpreter and must not be claimed
usable.  ``AnalyticalChangeTransformer`` is not an extraction executor.

Comments and source maps are preserved for supported module-level function
and class targets.  Unsupported constructs, missing raw source, rewrite /
adapter / façade execution, and unrestricted scope are typed terminals.
The adapter cannot authorize a transition, completion, repository write, or
competing authority.  Dry-run is deterministic and never mutates.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import PurePosixPath
from typing import Any, ClassVar, Final, Mapping, Sequence
import importlib.util
import unicodedata

from ipfs_accelerate_py.utils.cid_utils import cid_for_bytes, cid_for_dag_json, validate_cid

from .partition_generators import (
    IDENTITY_EXCLUDED_FIELDS as SPAR013_IDENTITY_EXCLUDED_FIELDS,
)
from .transformation_packet import (
    DECLARED_ALLOWED_EFFECTS,
    DECLARED_FORBIDDEN_EFFECTS,
    EditKind,
    RefactorTransformationPacket,
    TransformationPacketError,
)


TASK_ID: Final[str] = "SPAR-020"
GOAL_ID: Final[str] = "SPAR-G041"
PROGRAM: Final[str] = "semantic-preserving-autonomous-remodularization-v1"
AUTHORITY: Final[str] = "partition orchestration"
AUTHORITY_OWNER: Final[str] = "ipfs_accelerate_py"
ANALYZER_ID: Final[str] = (
    "ipfs_accelerate_py.agent_supervisor.semantic_refactoring.codemod@1"
)

CST_EXTRACTION_CODEMOD_INTERFACE: Final[str] = "CSTExtractionCodemod@1"
MEMBER_LOCATOR_INTERFACE: Final[str] = "MemberLocator@1"
SOURCE_MAP_ENTRY_INTERFACE: Final[str] = "SourceMapEntry@1"
CST_CAPABILITY_PROBE_INTERFACE: Final[str] = "CSTCapabilityProbe@1"
EXTRACTION_RESULT_INTERFACE: Final[str] = "ExtractionResult@1"
CODEMOD_RECEIPT_INTERFACE: Final[str] = "CodemodReceipt@1"

CST_EXTRACTION_CODEMOD_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/cst-extraction-codemod@1"
)
MEMBER_LOCATOR_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/member-locator@1"
)
SOURCE_MAP_ENTRY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/source-map-entry@1"
)
CST_CAPABILITY_PROBE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/cst-capability-probe@1"
)
EXTRACTION_RESULT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/extraction-result@1"
)
CODEMOD_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/codemod-receipt@1"
)

CODEMOD_CONTRACT_VERSION: Final[str] = "1"

CODEMOD_CAN_AUTHORIZE_TRANSITION: Final[bool] = False
CODEMOD_CAN_AUTHORIZE_COMPLETION: Final[bool] = False
CODEMOD_CAN_CREATE_AUTHORITY: Final[bool] = False
CODEMOD_CAN_RETIRE_FACADE: Final[bool] = False
CODEMOD_WRITES_REPOSITORY: Final[bool] = False
VECTOR_SIMILARITY_IS_AUTHORITY: Final[bool] = False
PROJECTION_CLUSTERING_IS_AUTHORITY: Final[bool] = False
MODEL_OUTPUT_IS_PROPOSAL_ONLY: Final[bool] = True
TEST_PASS_IS_NOT_COMPLETION: Final[bool] = True
MARKDOWN_IS_NOT_COMPLETION: Final[bool] = True
WORKER_SELF_APPROVAL: Final[bool] = False
DUCKLAKE_IS_AUTHORITY: Final[bool] = False
SOFT_SIGNALS_CANNOT_OVERRIDE_HARD_CONSTRAINTS: Final[bool] = True
CODEMOD_IS_NOMINATION_ONLY: Final[bool] = True
RAW_SOURCE_REQUIRED: Final[bool] = True
DRY_RUN_IS_DETERMINISTIC: Final[bool] = True
DRY_RUN_MUTATES: Final[bool] = False
LIBCST_IS_USABLE: Final[bool] = False
LIBCST_MUST_NOT_BE_CLAIMED_USABLE: Final[bool] = True
PARSO_IS_PARSE_HELPER: Final[bool] = True
ASTTOKENS_IS_PARSE_HELPER: Final[bool] = True
ANALYTICAL_CHANGE_TRANSFORMER_IS_EXECUTOR: Final[bool] = False
BEST_AVAILABLE_CST_BACKEND: Final[str] = "parso"

LIBCST_STATUS_TYPED_UNAVAILABLE: Final[str] = "typed_unavailable"
LIBCST_STATUS_PRESENT_NOT_ADMITTED: Final[str] = "present_not_admitted"
PARSE_HELPER_PRESENT: Final[str] = "present_parse_helper"
PARSE_HELPER_UNAVAILABLE: Final[str] = "typed_unavailable"

MAX_TEXT_CHARS: Final[int] = 1_048_576
MAX_MEMBERS: Final[int] = 16_384
MAX_PATH_CHARS: Final[int] = 1_024
MAX_WRITE_PATHS: Final[int] = 64
MAX_SOURCE_MAPS: Final[int] = 16_384

IDENTITY_EXCLUDED_FIELDS: Final[frozenset[str]] = SPAR013_IDENTITY_EXCLUDED_FIELDS

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

_AUTHORITY_FLAG_NAMES: Final[tuple[str, ...]] = (
    "can_authorize_transition",
    "can_authorize_completion",
    "can_create_authority",
    "projection_is_authority",
)

_SUPPORTED_BACKENDS: Final[frozenset[str]] = frozenset({"parso"})
_DEFERRED_EDIT_KINDS: Final[frozenset[str]] = frozenset(
    {EditKind.REWRITE.value, EditKind.ADAPTER.value, EditKind.FACADE.value}
)
_ERROR_NODE_TYPES: Final[frozenset[str]] = frozenset({"error_node", "error_leaf"})


class CodemodError(ValueError):
    """Fail-closed violation of a SPAR-020 CST extraction contract."""


class TargetKind(str, Enum):
    FUNCTION = "function"
    CLASS = "class"


DECLARED_TARGET_KINDS: Final[frozenset[str]] = frozenset(
    kind.value for kind in TargetKind
)


def _text(value: Any, name: str, *, empty: bool = False) -> str:
    if type(value) is not str:
        raise CodemodError(f"{name} must be a string")
    if value != value.strip() or unicodedata.normalize("NFC", value) != value:
        raise CodemodError(f"{name} must be trimmed NFC text")
    if not empty and not value:
        raise CodemodError(f"{name} must be a nonempty string")
    if any(not char.isprintable() and char not in "\n\t\r" for char in value):
        raise CodemodError(f"{name} contains invalid text")
    if len(value) > MAX_TEXT_CHARS:
        raise CodemodError(f"{name} exceeds text bound")
    return value


def _cid(value: Any, name: str) -> str:
    text = _text(value, name)
    try:
        return validate_cid(text)
    except Exception as exc:
        raise CodemodError(f"{name} must be a valid CID") from exc


def _bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise CodemodError(f"{name} must be a boolean")
    return value


def _int(value: Any, name: str, *, minimum: int = 0) -> int:
    if type(value) is bool or type(value) is not int:
        raise CodemodError(f"{name} must be an integer")
    if value < minimum:
        raise CodemodError(f"{name} is out of range")
    return value


def _tree_id(value: Any) -> str:
    text = _text(value, "tree_id")
    if len(text) not in {40, 64} or any(
        char not in "0123456789abcdef" for char in text
    ):
        raise CodemodError("tree_id must be a lowercase hex Git tree identity")
    return text


def _closed(data: Mapping[str, Any], fields: frozenset[str], name: str) -> dict[str, Any]:
    if not isinstance(data, Mapping) or isinstance(data, (str, bytes, bytearray)):
        raise CodemodError(f"{name} must be an object")
    extra = set(data) - fields
    missing = fields - set(data)
    if extra & IDENTITY_EXCLUDED_FIELDS:
        raise CodemodError(
            f"{name} identity excludes observational fields: "
            f"{sorted(extra & IDENTITY_EXCLUDED_FIELDS)}"
        )
    if extra:
        raise CodemodError(f"unknown {name} field: {sorted(extra)}")
    if missing:
        raise CodemodError(f"missing {name} field: {sorted(missing)}")
    return dict(data)


def _reject_excluded(payload: Mapping[str, Any], name: str) -> None:
    present = IDENTITY_EXCLUDED_FIELDS & set(payload)
    if present:
        raise CodemodError(
            f"{name} identity excludes observational fields: {sorted(present)}"
        )


def _require_dag_json(value: Any, name: str) -> None:
    try:
        cid_for_dag_json(value)
    except Exception as exc:
        raise CodemodError(f"{name} must be strict DAG-JSON") from exc


def _verify_cid(claimed: Any, computed: str, name: str) -> None:
    cid = _cid(claimed, name)
    if cid != computed:
        raise CodemodError(f"{name} does not verify")


def _unique_sorted_text(values: Any, name: str, *, limit: int) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        raise CodemodError(f"{name} must be a list")
    ordered = tuple(sorted(_text(item, name) for item in values))
    if len(ordered) > limit:
        raise CodemodError(f"{name} exceeds maximum length")
    if len(ordered) != len(set(ordered)):
        raise CodemodError(f"{name} must not contain duplicates")
    return ordered


def _enum(value: Any, enum_type: type[Enum], name: str) -> str:
    if isinstance(value, enum_type):
        return value.value
    text = _text(value, name)
    try:
        return enum_type(text).value
    except ValueError as exc:
        raise CodemodError(f"unknown {name}: {text}") from exc


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
    raise CodemodError(f"unsupported projected type {type(value).__name__}")


def _pop_authority_flags(payload: dict[str, Any], name: str) -> None:
    for flag in _AUTHORITY_FLAG_NAMES:
        if payload.pop(flag, False) is not False:
            raise CodemodError(f"{name} cannot claim {flag}")


def _exact_path(value: Any, name: str = "write_paths") -> str:
    raw = _text(value, name, empty=False)
    if len(raw) > MAX_PATH_CHARS:
        raise CodemodError(f"{name} exceeds path bound")
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
        raise CodemodError(
            f"{name} must be an exact repository-relative path; unrestricted scope is rejected"
        )
    if normalized != candidate.as_posix():
        raise CodemodError(f"{name} must be a normalized repository-relative path")
    return normalized


def _exact_paths(values: Any, name: str, *, required: bool = True) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise CodemodError(f"{name} must be a list of exact paths")
    ordered: list[str] = []
    seen: set[str] = set()
    for item in values:
        path = _exact_path(item, name)
        if path not in seen:
            seen.add(path)
            ordered.append(path)
    if required and not ordered:
        raise CodemodError(
            f"{name} must not be empty; unrestricted scope is rejected"
        )
    if len(ordered) > MAX_WRITE_PATHS:
        raise CodemodError(f"{name} exceeds path bound")
    return tuple(ordered)


def _source_text(value: Any, name: str) -> str:
    if type(value) is not str:
        raise CodemodError(f"{name} must be a string")
    if "\x00" in value:
        raise CodemodError(f"{name} contains invalid text")
    if len(value) > MAX_TEXT_CHARS:
        raise CodemodError(f"{name} exceeds text bound")
    return value


def _source_cid_for(text: str) -> str:
    return cid_for_bytes(text.encode("utf-8"))


def codemod_cid_profile() -> dict[str, str]:
    return {
        "profile_id": "ipfs_accelerate_py.cid-utils@1",
        "codec": "dag-json",
        "rule": (
            "CID identifies exact canonical bytes under declared codec/profile, "
            "not universal meaning"
        ),
    }


def _spec_status(module_name: str, *, helper: bool) -> str:
    present = importlib.util.find_spec(module_name) is not None
    if module_name == "libcst":
        return (
            LIBCST_STATUS_PRESENT_NOT_ADMITTED
            if present
            else LIBCST_STATUS_TYPED_UNAVAILABLE
        )
    if helper:
        return PARSE_HELPER_PRESENT if present else PARSE_HELPER_UNAVAILABLE
    return "present" if present else PARSE_HELPER_UNAVAILABLE


def probe_cst_capability() -> "CSTCapabilityProbe":
    """Probe CST tools without importing libcst or claiming it usable."""

    parso_status = _spec_status("parso", helper=True)
    return CSTCapabilityProbe(
        libcst=_spec_status("libcst", helper=False),
        parso=parso_status,
        asttokens=_spec_status("asttokens", helper=True),
        libcst_usable=False,
        parso_is_parse_helper=parso_status == PARSE_HELPER_PRESENT,
        asttokens_is_parse_helper=_spec_status("asttokens", helper=True)
        == PARSE_HELPER_PRESENT,
        analytical_change_transformer_is_executor=False,
        best_available=BEST_AVAILABLE_CST_BACKEND
        if parso_status == PARSE_HELPER_PRESENT
        else "none",
    )


@dataclass(frozen=True, slots=True)
class CSTCapabilityProbe:
    """Runtime CST capability observation. Not completion authority."""

    libcst: str
    parso: str
    asttokens: str
    libcst_usable: bool = False
    parso_is_parse_helper: bool = True
    asttokens_is_parse_helper: bool = True
    analytical_change_transformer_is_executor: bool = False
    best_available: str = BEST_AVAILABLE_CST_BACKEND

    interface: ClassVar[str] = CST_CAPABILITY_PROBE_INTERFACE
    schema: ClassVar[str] = CST_CAPABILITY_PROBE_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "libcst",
            "parso",
            "asttokens",
            "libcst_usable",
            "parso_is_parse_helper",
            "asttokens_is_parse_helper",
            "analytical_change_transformer_is_executor",
            "best_available",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "libcst", _text(self.libcst, "libcst"))
        object.__setattr__(self, "parso", _text(self.parso, "parso"))
        object.__setattr__(self, "asttokens", _text(self.asttokens, "asttokens"))
        object.__setattr__(
            self, "libcst_usable", _bool(self.libcst_usable, "libcst_usable")
        )
        object.__setattr__(
            self,
            "parso_is_parse_helper",
            _bool(self.parso_is_parse_helper, "parso_is_parse_helper"),
        )
        object.__setattr__(
            self,
            "asttokens_is_parse_helper",
            _bool(self.asttokens_is_parse_helper, "asttokens_is_parse_helper"),
        )
        object.__setattr__(
            self,
            "analytical_change_transformer_is_executor",
            _bool(
                self.analytical_change_transformer_is_executor,
                "analytical_change_transformer_is_executor",
            ),
        )
        object.__setattr__(
            self, "best_available", _text(self.best_available, "best_available")
        )
        if self.libcst_usable is not False:
            raise CodemodError("libcst must not be claimed usable")
        if self.libcst not in {
            LIBCST_STATUS_TYPED_UNAVAILABLE,
            LIBCST_STATUS_PRESENT_NOT_ADMITTED,
        }:
            raise CodemodError("libcst status must remain typed and non-usable")
        if self.analytical_change_transformer_is_executor is not False:
            raise CodemodError(
                "AnalyticalChangeTransformer is not an extraction executor"
            )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema": CST_CAPABILITY_PROBE_SCHEMA,
            "interface": CST_CAPABILITY_PROBE_INTERFACE,
            "libcst": self.libcst,
            "parso": self.parso,
            "asttokens": self.asttokens,
            "libcst_usable": False,
            "parso_is_parse_helper": self.parso_is_parse_helper,
            "asttokens_is_parse_helper": self.asttokens_is_parse_helper,
            "analytical_change_transformer_is_executor": False,
            "best_available": self.best_available,
        }
        _require_dag_json(payload, self.__class__.__name__)
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CSTCapabilityProbe":
        _reject_excluded(data, cls.__name__)
        payload = _closed(data, cls._FIELDS, cls.__name__)
        if payload.pop("schema") != CST_CAPABILITY_PROBE_SCHEMA:
            raise CodemodError("unsupported CSTCapabilityProbe schema")
        if payload.pop("interface") != CST_CAPABILITY_PROBE_INTERFACE:
            raise CodemodError("unsupported CSTCapabilityProbe interface")
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class MemberLocator:
    """Exact member-to-source binding required for CST extraction."""

    member_id: str
    path: str
    symbol: str
    kind: TargetKind | str = TargetKind.FUNCTION

    interface: ClassVar[str] = MEMBER_LOCATOR_INTERFACE
    schema: ClassVar[str] = MEMBER_LOCATOR_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "member_id",
            "path",
            "symbol",
            "kind",
            "locator_cid",
        }
    )

    def __post_init__(self) -> None:
        kind = _enum(self.kind, TargetKind, "kind")
        object.__setattr__(self, "member_id", _text(self.member_id, "member_id"))
        object.__setattr__(self, "path", _exact_path(self.path, "path"))
        object.__setattr__(self, "symbol", _text(self.symbol, "symbol"))
        object.__setattr__(self, "kind", kind)
        if not self.symbol.isidentifier():
            raise CodemodError("symbol must be a Python identifier")

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": MEMBER_LOCATOR_SCHEMA,
            "interface": MEMBER_LOCATOR_INTERFACE,
            "member_id": self.member_id,
            "path": self.path,
            "symbol": self.symbol,
            "kind": self.kind,
        }
        _require_dag_json(payload, self.__class__.__name__)
        return payload

    @property
    def locator_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["locator_cid"] = self.locator_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MemberLocator":
        _reject_excluded(data, cls.__name__)
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("locator_cid")
        if payload.pop("schema") != MEMBER_LOCATOR_SCHEMA:
            raise CodemodError("unsupported MemberLocator schema")
        if payload.pop("interface") != MEMBER_LOCATOR_INTERFACE:
            raise CodemodError("unsupported MemberLocator interface")
        result = cls(**payload)
        _verify_cid(claimed, result.locator_cid, "MemberLocator locator_cid")
        return result


def _coerce_locator(value: MemberLocator | Mapping[str, Any]) -> MemberLocator:
    if isinstance(value, MemberLocator):
        return value
    if isinstance(value, Mapping):
        if "locator_cid" in value:
            return MemberLocator.from_dict(value)
        payload = {
            key: item
            for key, item in _project(value).items()
            if key not in {"schema", "interface", "locator_cid"}
        }
        return MemberLocator(**payload)
    raise CodemodError("locator must be a MemberLocator")


@dataclass(frozen=True, slots=True)
class SourceMapEntry:
    """Exact origin-to-destination span binding for one extracted member."""

    member_id: str
    origin_path: str
    destination_path: str
    origin_start_line: int
    origin_start_col: int
    origin_end_line: int
    origin_end_col: int
    destination_start_line: int
    destination_start_col: int
    destination_end_line: int
    destination_end_col: int
    comment_prefix_preserved: bool

    interface: ClassVar[str] = SOURCE_MAP_ENTRY_INTERFACE
    schema: ClassVar[str] = SOURCE_MAP_ENTRY_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "member_id",
            "origin_path",
            "destination_path",
            "origin_start_line",
            "origin_start_col",
            "origin_end_line",
            "origin_end_col",
            "destination_start_line",
            "destination_start_col",
            "destination_end_line",
            "destination_end_col",
            "comment_prefix_preserved",
            "map_cid",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "member_id", _text(self.member_id, "member_id"))
        object.__setattr__(
            self, "origin_path", _exact_path(self.origin_path, "origin_path")
        )
        object.__setattr__(
            self,
            "destination_path",
            _exact_path(self.destination_path, "destination_path"),
        )
        object.__setattr__(
            self,
            "origin_start_line",
            _int(self.origin_start_line, "origin_start_line", minimum=1),
        )
        object.__setattr__(
            self,
            "origin_start_col",
            _int(self.origin_start_col, "origin_start_col", minimum=0),
        )
        object.__setattr__(
            self,
            "origin_end_line",
            _int(self.origin_end_line, "origin_end_line", minimum=1),
        )
        object.__setattr__(
            self,
            "origin_end_col",
            _int(self.origin_end_col, "origin_end_col", minimum=0),
        )
        object.__setattr__(
            self,
            "destination_start_line",
            _int(self.destination_start_line, "destination_start_line", minimum=1),
        )
        object.__setattr__(
            self,
            "destination_start_col",
            _int(self.destination_start_col, "destination_start_col", minimum=0),
        )
        object.__setattr__(
            self,
            "destination_end_line",
            _int(self.destination_end_line, "destination_end_line", minimum=1),
        )
        object.__setattr__(
            self,
            "destination_end_col",
            _int(self.destination_end_col, "destination_end_col", minimum=0),
        )
        object.__setattr__(
            self,
            "comment_prefix_preserved",
            _bool(self.comment_prefix_preserved, "comment_prefix_preserved"),
        )

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": SOURCE_MAP_ENTRY_SCHEMA,
            "interface": SOURCE_MAP_ENTRY_INTERFACE,
            "member_id": self.member_id,
            "origin_path": self.origin_path,
            "destination_path": self.destination_path,
            "origin_start_line": self.origin_start_line,
            "origin_start_col": self.origin_start_col,
            "origin_end_line": self.origin_end_line,
            "origin_end_col": self.origin_end_col,
            "destination_start_line": self.destination_start_line,
            "destination_start_col": self.destination_start_col,
            "destination_end_line": self.destination_end_line,
            "destination_end_col": self.destination_end_col,
            "comment_prefix_preserved": self.comment_prefix_preserved,
        }
        _require_dag_json(payload, self.__class__.__name__)
        return payload

    @property
    def map_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["map_cid"] = self.map_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SourceMapEntry":
        _reject_excluded(data, cls.__name__)
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("map_cid")
        if payload.pop("schema") != SOURCE_MAP_ENTRY_SCHEMA:
            raise CodemodError("unsupported SourceMapEntry schema")
        if payload.pop("interface") != SOURCE_MAP_ENTRY_INTERFACE:
            raise CodemodError("unsupported SourceMapEntry interface")
        result = cls(**payload)
        _verify_cid(claimed, result.map_cid, "SourceMapEntry map_cid")
        return result


def _coerce_source_map(value: SourceMapEntry | Mapping[str, Any]) -> SourceMapEntry:
    if isinstance(value, SourceMapEntry):
        return value
    if isinstance(value, Mapping):
        if "map_cid" in value:
            return SourceMapEntry.from_dict(value)
        payload = {
            key: item
            for key, item in _project(value).items()
            if key not in {"schema", "interface", "map_cid"}
        }
        return SourceMapEntry(**payload)
    raise CodemodError("source_maps must contain SourceMapEntry values")


@dataclass(frozen=True, slots=True)
class CodemodReceipt:
    """Body-free SPAR-020 receipt. Nomination only; never completion."""

    tree_id: str
    packet_cid: str
    write_paths: Sequence[str]
    moved_member_ids: Sequence[str]
    deferred_edit_cids: Sequence[str]
    source_cids: Sequence[Mapping[str, str]]
    source_map_cids: Sequence[str]
    mutated: bool = False
    deterministic: bool = True
    libcst_usable: bool = False
    can_authorize_transition: bool = False
    can_authorize_completion: bool = False
    can_create_authority: bool = False
    codemod_is_nomination_only: bool = True
    writes_repository: bool = False

    interface: ClassVar[str] = CODEMOD_RECEIPT_INTERFACE
    schema: ClassVar[str] = CODEMOD_RECEIPT_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "tree_id",
            "packet_cid",
            "write_paths",
            "moved_member_ids",
            "deferred_edit_cids",
            "source_cids",
            "source_map_cids",
            "mutated",
            "deterministic",
            "libcst_usable",
            "can_authorize_transition",
            "can_authorize_completion",
            "can_create_authority",
            "codemod_is_nomination_only",
            "writes_repository",
            "receipt_cid",
        }
    )

    def __post_init__(self) -> None:
        paths = _exact_paths(list(self.write_paths), "write_paths", required=True)
        moved = _unique_sorted_text(
            list(self.moved_member_ids), "moved_member_ids", limit=MAX_MEMBERS
        )
        deferred = _unique_sorted_text(
            list(self.deferred_edit_cids), "deferred_edit_cids", limit=MAX_MEMBERS
        )
        maps = _unique_sorted_text(
            list(self.source_map_cids), "source_map_cids", limit=MAX_SOURCE_MAPS
        )
        object.__setattr__(self, "tree_id", _tree_id(self.tree_id))
        object.__setattr__(self, "packet_cid", _cid(self.packet_cid, "packet_cid"))
        object.__setattr__(self, "write_paths", paths)
        object.__setattr__(self, "moved_member_ids", moved)
        object.__setattr__(self, "deferred_edit_cids", deferred)
        object.__setattr__(self, "source_map_cids", maps)
        object.__setattr__(
            self, "source_cids", _normalize_source_cids(self.source_cids)
        )
        object.__setattr__(self, "mutated", _bool(self.mutated, "mutated"))
        object.__setattr__(
            self, "deterministic", _bool(self.deterministic, "deterministic")
        )
        object.__setattr__(
            self, "libcst_usable", _bool(self.libcst_usable, "libcst_usable")
        )
        object.__setattr__(
            self,
            "can_authorize_transition",
            _bool(self.can_authorize_transition, "can_authorize_transition"),
        )
        object.__setattr__(
            self,
            "can_authorize_completion",
            _bool(self.can_authorize_completion, "can_authorize_completion"),
        )
        object.__setattr__(
            self,
            "can_create_authority",
            _bool(self.can_create_authority, "can_create_authority"),
        )
        object.__setattr__(
            self,
            "codemod_is_nomination_only",
            _bool(self.codemod_is_nomination_only, "codemod_is_nomination_only"),
        )
        object.__setattr__(
            self,
            "writes_repository",
            _bool(self.writes_repository, "writes_repository"),
        )
        if self.mutated is not False:
            raise CodemodError("CST extraction dry-run/apply must not mutate the repository")
        if self.deterministic is not True:
            raise CodemodError("CST extraction must remain deterministic")
        if self.libcst_usable is not False:
            raise CodemodError("libcst must not be claimed usable")
        if self.can_authorize_transition is not False:
            raise CodemodError("codemod cannot authorize a transition")
        if self.can_authorize_completion is not False:
            raise CodemodError("codemod cannot authorize completion")
        if self.can_create_authority is not False:
            raise CodemodError("codemod cannot create authority")
        if self.codemod_is_nomination_only is not True:
            raise CodemodError("codemod must remain nomination_only")
        if self.writes_repository is not False:
            raise CodemodError("codemod cannot write the repository")

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": CODEMOD_RECEIPT_SCHEMA,
            "interface": CODEMOD_RECEIPT_INTERFACE,
            "tree_id": self.tree_id,
            "packet_cid": self.packet_cid,
            "write_paths": list(self.write_paths),
            "moved_member_ids": list(self.moved_member_ids),
            "deferred_edit_cids": list(self.deferred_edit_cids),
            "source_cids": [dict(item) for item in self.source_cids],
            "source_map_cids": list(self.source_map_cids),
            "mutated": False,
            "deterministic": True,
            "libcst_usable": False,
            "can_authorize_transition": False,
            "can_authorize_completion": False,
            "can_create_authority": False,
            "codemod_is_nomination_only": True,
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
    def from_dict(cls, data: Mapping[str, Any]) -> "CodemodReceipt":
        _reject_excluded(data, cls.__name__)
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("receipt_cid")
        if payload.pop("schema") != CODEMOD_RECEIPT_SCHEMA:
            raise CodemodError("unsupported CodemodReceipt schema")
        if payload.pop("interface") != CODEMOD_RECEIPT_INTERFACE:
            raise CodemodError("unsupported CodemodReceipt interface")
        _pop_authority_flags(payload, "CodemodReceipt")
        if payload.pop("libcst_usable") is not False:
            raise CodemodError("libcst must not be claimed usable")
        if payload.pop("codemod_is_nomination_only") is not True:
            raise CodemodError("codemod must remain nomination_only")
        if payload.pop("writes_repository") is not False:
            raise CodemodError("codemod cannot write the repository")
        if payload.pop("mutated") is not False:
            raise CodemodError("CST extraction dry-run/apply must not mutate the repository")
        if payload.pop("deterministic") is not True:
            raise CodemodError("CST extraction must remain deterministic")
        result = cls(**payload)
        _verify_cid(claimed, result.receipt_cid, "CodemodReceipt receipt_cid")
        return result


def _normalize_source_cids(values: Any) -> tuple[dict[str, str], ...]:
    if not isinstance(values, (list, tuple)):
        raise CodemodError("source_cids must be a list")
    ordered: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in values:
        if not isinstance(item, Mapping):
            raise CodemodError("source_cids entries must be objects")
        path = _exact_path(item.get("path"), "source_cids.path")
        cid = _cid(item.get("cid"), "source_cids.cid")
        if path in seen:
            raise CodemodError("source_cids must not contain duplicates")
        seen.add(path)
        ordered.append({"path": path, "cid": cid})
    ordered.sort(key=lambda item: item["path"])
    if len(ordered) > MAX_WRITE_PATHS:
        raise CodemodError("source_cids exceeds path bound")
    return tuple(ordered)


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    """Nominated in-memory CST extraction. Not an accepted transition."""

    tree_id: str
    packet_cid: str
    sources: Mapping[str, str]
    source_maps: Sequence[SourceMapEntry]
    deferred_edit_cids: Sequence[str]
    moved_member_ids: Sequence[str]
    write_paths: Sequence[str]
    mutated: bool = False
    deterministic: bool = True
    libcst_usable: bool = False
    can_authorize_transition: bool = False
    can_authorize_completion: bool = False
    can_create_authority: bool = False
    writes_repository: bool = False

    interface: ClassVar[str] = EXTRACTION_RESULT_INTERFACE
    schema: ClassVar[str] = EXTRACTION_RESULT_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "tree_id",
            "packet_cid",
            "sources",
            "source_maps",
            "deferred_edit_cids",
            "moved_member_ids",
            "write_paths",
            "mutated",
            "deterministic",
            "libcst_usable",
            "can_authorize_transition",
            "can_authorize_completion",
            "can_create_authority",
            "writes_repository",
            "result_cid",
        }
    )

    def __post_init__(self) -> None:
        if not isinstance(self.sources, Mapping) or isinstance(
            self.sources, (str, bytes, bytearray)
        ):
            raise CodemodError("sources must be an object")
        sources = {
            _exact_path(path, "sources"): _source_text(text, "sources")
            for path, text in self.sources.items()
        }
        maps = tuple(_coerce_source_map(item) for item in self.source_maps)
        if len(maps) > MAX_SOURCE_MAPS:
            raise CodemodError("source_maps exceeds maximum length")
        paths = _exact_paths(list(self.write_paths), "write_paths", required=True)
        moved = _unique_sorted_text(
            list(self.moved_member_ids), "moved_member_ids", limit=MAX_MEMBERS
        )
        deferred = _unique_sorted_text(
            list(self.deferred_edit_cids), "deferred_edit_cids", limit=MAX_MEMBERS
        )
        object.__setattr__(self, "tree_id", _tree_id(self.tree_id))
        object.__setattr__(self, "packet_cid", _cid(self.packet_cid, "packet_cid"))
        object.__setattr__(self, "sources", dict(sorted(sources.items())))
        object.__setattr__(self, "source_maps", maps)
        object.__setattr__(self, "deferred_edit_cids", deferred)
        object.__setattr__(self, "moved_member_ids", moved)
        object.__setattr__(self, "write_paths", paths)
        object.__setattr__(self, "mutated", _bool(self.mutated, "mutated"))
        object.__setattr__(
            self, "deterministic", _bool(self.deterministic, "deterministic")
        )
        object.__setattr__(
            self, "libcst_usable", _bool(self.libcst_usable, "libcst_usable")
        )
        object.__setattr__(
            self,
            "can_authorize_transition",
            _bool(self.can_authorize_transition, "can_authorize_transition"),
        )
        object.__setattr__(
            self,
            "can_authorize_completion",
            _bool(self.can_authorize_completion, "can_authorize_completion"),
        )
        object.__setattr__(
            self,
            "can_create_authority",
            _bool(self.can_create_authority, "can_create_authority"),
        )
        object.__setattr__(
            self,
            "writes_repository",
            _bool(self.writes_repository, "writes_repository"),
        )
        if self.mutated is not False:
            raise CodemodError("CST extraction must not mutate the repository")
        if self.deterministic is not True:
            raise CodemodError("CST extraction must remain deterministic")
        if self.libcst_usable is not False:
            raise CodemodError("libcst must not be claimed usable")
        if self.can_authorize_transition is not False:
            raise CodemodError("codemod cannot authorize a transition")
        if self.can_authorize_completion is not False:
            raise CodemodError("codemod cannot authorize completion")
        if self.can_create_authority is not False:
            raise CodemodError("codemod cannot create authority")
        if self.writes_repository is not False:
            raise CodemodError("codemod cannot write the repository")
        for item in maps:
            if item.origin_path not in self.sources and item.origin_path not in paths:
                raise CodemodError("source map origin_path is outside write_paths")
            if item.destination_path not in self.sources:
                raise CodemodError("source map destination_path missing from sources")
            if item.destination_path not in paths:
                raise CodemodError("source map destination_path is outside write_paths")

    def source_cid_records(self) -> tuple[dict[str, str], ...]:
        return tuple(
            {"path": path, "cid": _source_cid_for(text)}
            for path, text in self.sources.items()
        )

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": EXTRACTION_RESULT_SCHEMA,
            "interface": EXTRACTION_RESULT_INTERFACE,
            "tree_id": self.tree_id,
            "packet_cid": self.packet_cid,
            "source_cids": [dict(item) for item in self.source_cid_records()],
            "source_maps": [item.to_dict() for item in self.source_maps],
            "deferred_edit_cids": list(self.deferred_edit_cids),
            "moved_member_ids": list(self.moved_member_ids),
            "write_paths": list(self.write_paths),
            "mutated": False,
            "deterministic": True,
            "libcst_usable": False,
            "can_authorize_transition": False,
            "can_authorize_completion": False,
            "can_create_authority": False,
            "writes_repository": False,
        }
        _require_dag_json(payload, self.__class__.__name__)
        return payload

    @property
    def result_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema": EXTRACTION_RESULT_SCHEMA,
            "interface": EXTRACTION_RESULT_INTERFACE,
            "tree_id": self.tree_id,
            "packet_cid": self.packet_cid,
            "sources": dict(self.sources),
            "source_maps": [item.to_dict() for item in self.source_maps],
            "deferred_edit_cids": list(self.deferred_edit_cids),
            "moved_member_ids": list(self.moved_member_ids),
            "write_paths": list(self.write_paths),
            "mutated": False,
            "deterministic": True,
            "libcst_usable": False,
            "can_authorize_transition": False,
            "can_authorize_completion": False,
            "can_create_authority": False,
            "writes_repository": False,
            "result_cid": self.result_cid,
        }
        return payload

    def receipt(self) -> CodemodReceipt:
        return CodemodReceipt(
            tree_id=self.tree_id,
            packet_cid=self.packet_cid,
            write_paths=self.write_paths,
            moved_member_ids=self.moved_member_ids,
            deferred_edit_cids=self.deferred_edit_cids,
            source_cids=self.source_cid_records(),
            source_map_cids=tuple(item.map_cid for item in self.source_maps),
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExtractionResult":
        _reject_excluded(data, cls.__name__)
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("result_cid")
        if payload.pop("schema") != EXTRACTION_RESULT_SCHEMA:
            raise CodemodError("unsupported ExtractionResult schema")
        if payload.pop("interface") != EXTRACTION_RESULT_INTERFACE:
            raise CodemodError("unsupported ExtractionResult interface")
        _pop_authority_flags(payload, "ExtractionResult")
        if payload.pop("libcst_usable") is not False:
            raise CodemodError("libcst must not be claimed usable")
        if payload.pop("writes_repository") is not False:
            raise CodemodError("codemod cannot write the repository")
        if payload.pop("mutated") is not False:
            raise CodemodError("CST extraction must not mutate the repository")
        if payload.pop("deterministic") is not True:
            raise CodemodError("CST extraction must remain deterministic")
        result = cls(**payload)
        _verify_cid(claimed, result.result_cid, "ExtractionResult result_cid")
        return result


def _load_parso() -> Any:
    if importlib.util.find_spec("parso") is None:
        raise CodemodError("parso parse helper is typed unavailable")
    import parso

    return parso


def _has_error_node(node: Any) -> bool:
    if getattr(node, "type", "") in _ERROR_NODE_TYPES:
        return True
    for child in getattr(node, "children", None) or ():
        if _has_error_node(child):
            return True
    return False


def _parse_module(source: str) -> Any:
    parso = _load_parso()
    module = parso.parse(source)
    errors = list(module.iter_errors()) if hasattr(module, "iter_errors") else []
    if errors or _has_error_node(module):
        raise CodemodError("unsupported syntax is a typed terminal")
    return module


def _offset(source: str, line: int, col: int) -> int:
    if line < 1 or col < 0:
        raise CodemodError("source map out of range")
    current = 1
    index = 0
    length = len(source)
    while current < line:
        if index >= length:
            raise CodemodError("source map out of range")
        if source[index] == "\n":
            current += 1
        index += 1
    offset = index + col
    if offset > length:
        raise CodemodError("source map out of range")
    return offset


def _span_from_offsets(source: str, start: int, end: int) -> tuple[int, int, int, int]:
    start_line = source.count("\n", 0, start) + 1
    last_nl = source.rfind("\n", 0, start)
    start_col = start if last_nl < 0 else start - last_nl - 1
    end_line = source.count("\n", 0, end) + 1
    last_nl = source.rfind("\n", 0, end)
    end_col = end if last_nl < 0 else end - last_nl - 1
    return start_line, start_col, end_line, end_col


@dataclass(frozen=True, slots=True)
class _LocatedDef:
    name: str
    kind: str
    node: Any
    unsupported: bool


def _inner_definition(node: Any) -> tuple[Any, bool]:
    if node.type in {"funcdef", "classdef"}:
        return node, False
    if node.type == "async_stmt":
        return node, True
    if node.type == "decorated":
        inners = [
            child
            for child in node.children
            if child.type in {"funcdef", "classdef", "async_stmt"}
        ]
        if len(inners) != 1:
            return node, True
        inner, async_flag = _inner_definition(inners[0])
        return inner, async_flag
    return node, True


def _kind_for_node(node: Any) -> str:
    if node.type == "classdef":
        return TargetKind.CLASS.value
    return TargetKind.FUNCTION.value


def _top_level_definitions(source: str) -> dict[str, _LocatedDef]:
    module = _parse_module(source)
    found: dict[str, _LocatedDef] = {}
    for child in getattr(module, "children", None) or ():
        if child.type not in {"funcdef", "classdef", "decorated", "async_stmt"}:
            continue
        inner, unsupported = _inner_definition(child)
        name_node = getattr(inner, "name", None)
        name = getattr(name_node, "value", None)
        if not name:
            if inner.type == "async_stmt":
                for grandchild in getattr(inner, "children", None) or ():
                    nested = getattr(grandchild, "name", None)
                    nested_name = getattr(nested, "value", None)
                    if nested_name:
                        name = nested_name
                        unsupported = True
                        inner = grandchild
                        break
            if not name:
                continue
        if name in found:
            raise CodemodError("unsupported syntax is a typed terminal")
        found[name] = _LocatedDef(
            name=name,
            kind=_kind_for_node(inner),
            node=child,
            unsupported=unsupported or child.type == "async_stmt",
        )
    return found


def _confirm_with_asttokens(source: str, symbol: str, kind: str) -> None:
    if importlib.util.find_spec("asttokens") is None:
        return
    import ast
    import asttokens

    try:
        tokens = asttokens.ASTTokens(source, parse=True)
    except SyntaxError as exc:
        raise CodemodError("unsupported syntax is a typed terminal") from exc
    tree = tokens.tree
    if tree is None:
        return
    for node in getattr(tree, "body", ()):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == symbol:
            raise CodemodError("unsupported syntax is a typed terminal")
        if kind == TargetKind.FUNCTION.value and isinstance(node, ast.FunctionDef):
            continue
        if kind == TargetKind.CLASS.value and isinstance(node, ast.ClassDef):
            continue


def _cut_node(source: str, node: Any) -> tuple[str, str, tuple[int, int, int, int], bool]:
    leaf = node.get_first_leaf()
    prefix = getattr(leaf, "prefix", "") or ""
    start = _offset(source, *node.start_pos) - len(prefix)
    if start < 0:
        start = 0
    end = _offset(source, *node.end_pos)
    extracted = source[start:end]
    remaining = source[:start] + source[end:]
    comment_preserved = "#" in prefix or "#" in extracted
    return extracted, remaining, _span_from_offsets(source, start, end), comment_preserved


def _normalize_remaining(text: str) -> str:
    if not text.strip():
        return ""
    if text.endswith("\n"):
        return text
    return text + "\n"


def _paste(destination: str, extracted: str) -> tuple[str, tuple[int, int, int, int]]:
    chunk = extracted[1:] if extracted.startswith("\n") else extracted
    if not chunk.endswith("\n"):
        chunk += "\n"
    if not destination:
        span = _span_from_offsets(chunk, 0, len(chunk))
        return chunk, span
    prefix = destination
    if not prefix.endswith("\n"):
        prefix += "\n"
    if not prefix.endswith("\n\n"):
        prefix += "\n"
    start = len(prefix)
    combined = prefix + chunk
    return combined, _span_from_offsets(combined, start, len(combined))


def _coerce_packet(
    packet: RefactorTransformationPacket | Mapping[str, Any],
) -> RefactorTransformationPacket:
    if isinstance(packet, RefactorTransformationPacket):
        resolved = packet
    elif isinstance(packet, Mapping):
        try:
            resolved = RefactorTransformationPacket.from_dict(packet)
        except TransformationPacketError as exc:
            raise CodemodError(str(exc)) from exc
    else:
        raise CodemodError("packet must be a RefactorTransformationPacket")
    if resolved.can_authorize_completion is not False:
        raise CodemodError("codemod cannot authorize completion")
    if resolved.can_authorize_transition is not False:
        raise CodemodError("codemod cannot authorize a transition")
    if resolved.packet_is_nomination_only is not True:
        raise CodemodError("codemod must remain nomination_only")
    if resolved.unrestricted_scope is not False:
        raise CodemodError("unrestricted scope is rejected")
    unknown_effects = [
        item
        for item in resolved.effect_scope.allowed_effects
        if item not in DECLARED_ALLOWED_EFFECTS
    ]
    if unknown_effects:
        raise CodemodError(f"unknown allowed_effects: {unknown_effects}")
    forbidden = set(resolved.effect_scope.forbidden_effects)
    if not DECLARED_FORBIDDEN_EFFECTS <= forbidden and "network" not in forbidden:
        raise CodemodError("packet must retain declared forbidden effects")
    return resolved


def _locator_index(
    locators: Sequence[MemberLocator | Mapping[str, Any]],
) -> dict[str, MemberLocator]:
    if isinstance(locators, (str, bytes, bytearray)) or not isinstance(
        locators, Sequence
    ):
        raise CodemodError("locators must be a list")
    indexed: dict[str, MemberLocator] = {}
    for item in locators:
        locator = _coerce_locator(item)
        if locator.member_id in indexed:
            raise CodemodError("locators must not contain duplicates")
        indexed[locator.member_id] = locator
    return indexed


def _destination_path_for(
    edit: Any,
    *,
    write_paths: Sequence[str],
    destination_paths: Mapping[str, str] | None,
) -> str:
    if destination_paths:
        raw = destination_paths.get(edit.destination_id)
        if raw:
            return _exact_path(raw, "destination_paths")
    if len(write_paths) >= 2:
        return write_paths[1]
    raise CodemodError("MOVE destination_id has no bound destination path")


def extract_with_cst_codemod(
    packet: RefactorTransformationPacket | Mapping[str, Any],
    *,
    raw_sources: Mapping[str, str],
    locators: Sequence[MemberLocator | Mapping[str, Any]],
    destination_paths: Mapping[str, str] | None = None,
    backend: str = BEST_AVAILABLE_CST_BACKEND,
    dry_run: bool = True,
) -> ExtractionResult:
    """Nominate CST-preserving MOVE/extraction results in memory.

    ``dry_run`` does not change behavior: the adapter never writes the
    repository.  Rewrite, adapter, and façade edits are deferred to SPAR-021+.
    """

    del dry_run
    if backend == "libcst":
        raise CodemodError("libcst must not be claimed usable")
    if backend not in _SUPPORTED_BACKENDS:
        raise CodemodError("unsupported CST backend is a typed terminal")
    if ANALYTICAL_CHANGE_TRANSFORMER_IS_EXECUTOR is not False:
        raise CodemodError("AnalyticalChangeTransformer is not an extraction executor")
    probe = probe_cst_capability()
    if probe.libcst_usable is not False:
        raise CodemodError("libcst must not be claimed usable")
    if probe.parso != PARSE_HELPER_PRESENT:
        raise CodemodError("parso parse helper is typed unavailable")
    if probe.analytical_change_transformer_is_executor is not False:
        raise CodemodError("AnalyticalChangeTransformer is not an extraction executor")
    if not isinstance(raw_sources, Mapping) or isinstance(
        raw_sources, (str, bytes, bytearray)
    ):
        raise CodemodError("raw source is required")
    resolved = _coerce_packet(packet)
    write_paths = resolved.effect_scope.write_paths
    locator_by_id = _locator_index(locators)
    dest_bindings = {
        _text(key, "destination_paths"): _exact_path(value, "destination_paths")
        for key, value in dict(destination_paths or {}).items()
    }
    for path in dest_bindings.values():
        if path not in write_paths:
            raise CodemodError("destination path is outside effect_scope")
    working = {
        _exact_path(path, "raw_sources"): _source_text(text, "raw_sources")
        for path, text in raw_sources.items()
    }
    for path in working:
        if path not in write_paths:
            raise CodemodError("raw source path is outside effect_scope")
    moves = [item for item in resolved.edits if item.kind == EditKind.MOVE.value]
    if not moves:
        raise CodemodError("CST extraction requires MOVE edits")
    deferred = tuple(
        sorted(
            item.edit_cid
            for item in resolved.edits
            if item.kind in _DEFERRED_EDIT_KINDS
        )
    )
    maps: list[SourceMapEntry] = []
    moved: list[str] = []
    for edit in moves:
        dest_path = _destination_path_for(
            edit, write_paths=write_paths, destination_paths=dest_bindings
        )
        if dest_path not in write_paths:
            raise CodemodError("destination path is outside effect_scope")
        if dest_path not in working:
            working[dest_path] = ""
        for member_id in edit.member_ids:
            locator = locator_by_id.get(member_id)
            if locator is None:
                raise CodemodError("raw source locator is required")
            if locator.path not in write_paths:
                raise CodemodError("locator path is outside effect_scope")
            if locator.path not in working:
                raise CodemodError("raw source is required")
            origin = working[locator.path]
            destination = working[dest_path]
            origin_defs = _top_level_definitions(origin)
            dest_defs = _top_level_definitions(destination) if destination.strip() else {}
            located = origin_defs.get(locator.symbol)
            already = dest_defs.get(locator.symbol)
            if located is not None and already is not None:
                raise CodemodError("unsupported syntax is a typed terminal")
            if located is None and already is None:
                raise CodemodError("unsupported syntax is a typed terminal")
            if located is None and already is not None:
                if already.unsupported or already.kind != locator.kind:
                    raise CodemodError("unsupported syntax is a typed terminal")
                moved.append(member_id)
                continue
            if located.unsupported or located.kind != locator.kind:
                raise CodemodError("unsupported syntax is a typed terminal")
            _confirm_with_asttokens(origin, locator.symbol, locator.kind)
            extracted, remaining, origin_span, comments = _cut_node(origin, located.node)
            new_dest, dest_span = _paste(destination, extracted)
            working[locator.path] = _normalize_remaining(remaining)
            working[dest_path] = new_dest
            maps.append(
                SourceMapEntry(
                    member_id=member_id,
                    origin_path=locator.path,
                    destination_path=dest_path,
                    origin_start_line=origin_span[0],
                    origin_start_col=origin_span[1],
                    origin_end_line=origin_span[2],
                    origin_end_col=origin_span[3],
                    destination_start_line=dest_span[0],
                    destination_start_col=dest_span[1],
                    destination_end_line=dest_span[2],
                    destination_end_col=dest_span[3],
                    comment_prefix_preserved=comments,
                )
            )
            moved.append(member_id)
    return ExtractionResult(
        tree_id=resolved.tree_id,
        packet_cid=resolved.packet_cid,
        sources=working,
        source_maps=maps,
        deferred_edit_cids=deferred,
        moved_member_ids=moved,
        write_paths=write_paths,
    )


def dry_run_cst_extraction(
    packet: RefactorTransformationPacket | Mapping[str, Any],
    *,
    raw_sources: Mapping[str, str],
    locators: Sequence[MemberLocator | Mapping[str, Any]],
    destination_paths: Mapping[str, str] | None = None,
    backend: str = BEST_AVAILABLE_CST_BACKEND,
) -> CodemodReceipt:
    """Return a deterministic no-mutation dry-run of one CST extraction."""

    result = extract_with_cst_codemod(
        packet,
        raw_sources=raw_sources,
        locators=locators,
        destination_paths=destination_paths,
        backend=backend,
        dry_run=True,
    )
    return result.receipt()


def apply_cst_extraction(
    packet: RefactorTransformationPacket | Mapping[str, Any],
    *,
    raw_sources: Mapping[str, str],
    locators: Sequence[MemberLocator | Mapping[str, Any]],
    destination_paths: Mapping[str, str] | None = None,
    backend: str = BEST_AVAILABLE_CST_BACKEND,
) -> ExtractionResult:
    """Return nominated extracted sources. Does not write the repository."""

    if CODEMOD_WRITES_REPOSITORY is not False:
        raise CodemodError("codemod cannot write the repository")
    return extract_with_cst_codemod(
        packet,
        raw_sources=raw_sources,
        locators=locators,
        destination_paths=destination_paths,
        backend=backend,
        dry_run=False,
    )


@dataclass(frozen=True, slots=True)
class CSTExtractionCodemod:
    """Adapter over the current best available CST parse helper (parso)."""

    backend: str = BEST_AVAILABLE_CST_BACKEND

    interface: ClassVar[str] = CST_EXTRACTION_CODEMOD_INTERFACE
    schema: ClassVar[str] = CST_EXTRACTION_CODEMOD_SCHEMA

    def __post_init__(self) -> None:
        backend = _text(self.backend, "backend")
        if backend == "libcst":
            raise CodemodError("libcst must not be claimed usable")
        if backend not in _SUPPORTED_BACKENDS:
            raise CodemodError("unsupported CST backend is a typed terminal")
        object.__setattr__(self, "backend", backend)
        probe = probe_cst_capability()
        if probe.libcst_usable is not False:
            raise CodemodError("libcst must not be claimed usable")
        if probe.parso != PARSE_HELPER_PRESENT:
            raise CodemodError("parso parse helper is typed unavailable")

    def extract(
        self,
        packet: RefactorTransformationPacket | Mapping[str, Any],
        *,
        raw_sources: Mapping[str, str],
        locators: Sequence[MemberLocator | Mapping[str, Any]],
        destination_paths: Mapping[str, str] | None = None,
    ) -> ExtractionResult:
        return extract_with_cst_codemod(
            packet,
            raw_sources=raw_sources,
            locators=locators,
            destination_paths=destination_paths,
            backend=self.backend,
        )

    def dry_run(
        self,
        packet: RefactorTransformationPacket | Mapping[str, Any],
        *,
        raw_sources: Mapping[str, str],
        locators: Sequence[MemberLocator | Mapping[str, Any]],
        destination_paths: Mapping[str, str] | None = None,
    ) -> CodemodReceipt:
        return dry_run_cst_extraction(
            packet,
            raw_sources=raw_sources,
            locators=locators,
            destination_paths=destination_paths,
            backend=self.backend,
        )

    def apply(
        self,
        packet: RefactorTransformationPacket | Mapping[str, Any],
        *,
        raw_sources: Mapping[str, str],
        locators: Sequence[MemberLocator | Mapping[str, Any]],
        destination_paths: Mapping[str, str] | None = None,
    ) -> ExtractionResult:
        return apply_cst_extraction(
            packet,
            raw_sources=raw_sources,
            locators=locators,
            destination_paths=destination_paths,
            backend=self.backend,
        )


def compile_codemod_receipt(result: ExtractionResult | Mapping[str, Any]) -> CodemodReceipt:
    resolved = (
        result if isinstance(result, ExtractionResult) else ExtractionResult.from_dict(result)
    )
    return resolved.receipt()


def encode_canonical_result(result: ExtractionResult) -> dict[str, Any]:
    return result.to_dict()


def decode_canonical_result(payload: Mapping[str, Any]) -> ExtractionResult:
    return ExtractionResult.from_dict(payload)


def encode_canonical_receipt(receipt: CodemodReceipt) -> dict[str, Any]:
    return receipt.to_dict()


def decode_canonical_receipt(payload: Mapping[str, Any]) -> CodemodReceipt:
    return CodemodReceipt.from_dict(payload)


def provider_free_exports() -> tuple[str, ...]:
    return tuple(sorted(__all__))


def assert_not_competing_capsule_family() -> None:
    names = set(globals())
    overlap = names & _FORBIDDEN_CAPSULE_TYPE_NAMES
    if overlap:
        raise CodemodError(
            f"CST extraction codemod must not define capsule types: {sorted(overlap)}"
        )


__all__ = [
    "ANALYTICAL_CHANGE_TRANSFORMER_IS_EXECUTOR",
    "ANALYZER_ID",
    "ASTTOKENS_IS_PARSE_HELPER",
    "AUTHORITY",
    "AUTHORITY_OWNER",
    "BEST_AVAILABLE_CST_BACKEND",
    "CODEMOD_CAN_AUTHORIZE_COMPLETION",
    "CODEMOD_CAN_AUTHORIZE_TRANSITION",
    "CODEMOD_CAN_CREATE_AUTHORITY",
    "CODEMOD_CAN_RETIRE_FACADE",
    "CODEMOD_CONTRACT_VERSION",
    "CODEMOD_IS_NOMINATION_ONLY",
    "CODEMOD_RECEIPT_INTERFACE",
    "CODEMOD_WRITES_REPOSITORY",
    "CSTExtractionCodemod",
    "CST_CAPABILITY_PROBE_INTERFACE",
    "CST_EXTRACTION_CODEMOD_INTERFACE",
    "DECLARED_TARGET_KINDS",
    "DRY_RUN_IS_DETERMINISTIC",
    "DRY_RUN_MUTATES",
    "DUCKLAKE_IS_AUTHORITY",
    "EXTRACTION_RESULT_INTERFACE",
    "GOAL_ID",
    "IDENTITY_EXCLUDED_FIELDS",
    "LIBCST_IS_USABLE",
    "LIBCST_MUST_NOT_BE_CLAIMED_USABLE",
    "MARKDOWN_IS_NOT_COMPLETION",
    "MEMBER_LOCATOR_INTERFACE",
    "MODEL_OUTPUT_IS_PROPOSAL_ONLY",
    "PARSO_IS_PARSE_HELPER",
    "PROGRAM",
    "PROJECTION_CLUSTERING_IS_AUTHORITY",
    "RAW_SOURCE_REQUIRED",
    "SOFT_SIGNALS_CANNOT_OVERRIDE_HARD_CONSTRAINTS",
    "SOURCE_MAP_ENTRY_INTERFACE",
    "TASK_ID",
    "TEST_PASS_IS_NOT_COMPLETION",
    "VECTOR_SIMILARITY_IS_AUTHORITY",
    "WORKER_SELF_APPROVAL",
    "CSTCapabilityProbe",
    "CodemodError",
    "CodemodReceipt",
    "ExtractionResult",
    "MemberLocator",
    "SourceMapEntry",
    "TargetKind",
    "apply_cst_extraction",
    "assert_not_competing_capsule_family",
    "codemod_cid_profile",
    "compile_codemod_receipt",
    "decode_canonical_receipt",
    "decode_canonical_result",
    "dry_run_cst_extraction",
    "encode_canonical_receipt",
    "encode_canonical_result",
    "extract_with_cst_codemod",
    "probe_cst_capability",
    "provider_free_exports",
]


assert TASK_ID == "SPAR-020"
assert CST_EXTRACTION_CODEMOD_INTERFACE == "CSTExtractionCodemod@1"
assert CODEMOD_IS_NOMINATION_ONLY is True
assert CODEMOD_CAN_AUTHORIZE_COMPLETION is False
assert CODEMOD_WRITES_REPOSITORY is False
assert LIBCST_IS_USABLE is False
assert ANALYTICAL_CHANGE_TRANSFORMER_IS_EXECUTOR is False
assert DRY_RUN_MUTATES is False
assert "libcst" not in {item.lower() for item in __all__ if item == "libcst"}
