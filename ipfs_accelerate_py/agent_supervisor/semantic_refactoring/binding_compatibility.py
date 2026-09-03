"""SPAR-024 binding, introspection, and serialization compatibility adapter.

This module extends current supervisor partition orchestration with
``BindingCompatibilityAdapter@1``.  It consumes SPAR-019
``RefactorTransformationPacket@1`` wrapper/serialization/introspection/
traceback/documentation/patch-target edits, SPAR-018 façade-plan mappings,
and SPAR-011 public-compatibility inventory mappings, then nominates exact
wrappers or migrations for signatures, annotations, module/qualname, pickle,
reflection, tracebacks, docs, and patch targets, and classifies intentional
incompatibility.

SPAR-011 payloads are ingested as mappings only.  This module does not
replace datasets semantic authority, does not apply CST transforms, and
cannot authorize a transition, completion, or competing authority.
Vector, model, and heuristic evidence cannot admit an adapter.
Observational metadata is excluded from identity.  Dry-run is
deterministic and never mutates.  Silent public incompatibility is a
typed terminal.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import PurePosixPath
from typing import Any, ClassVar, Final, Mapping, Sequence
import unicodedata

from ipfs_accelerate_py.utils.cid_utils import cid_for_dag_json, validate_cid

from .partition_generators import (
    IDENTITY_EXCLUDED_FIELDS as SPAR013_IDENTITY_EXCLUDED_FIELDS,
)
from .transformation_packet import (
    ANALYZER_ID as SPAR019_ANALYZER_ID,
    AdapterKind,
    EditKind,
    RefactorTransformationPacket,
    RewriteKind,
)


TASK_ID: Final[str] = "SPAR-024"
GOAL_ID: Final[str] = "SPAR-G042"
PROGRAM: Final[str] = "semantic-preserving-autonomous-remodularization-v1"
AUTHORITY: Final[str] = "partition orchestration"
AUTHORITY_OWNER: Final[str] = "ipfs_accelerate_py"
ANALYZER_ID: Final[str] = (
    "ipfs_accelerate_py.agent_supervisor.semantic_refactoring.binding_compatibility@1"
)

BINDING_COMPATIBILITY_ADAPTER_INTERFACE: Final[str] = (
    "BindingCompatibilityAdapter@1"
)
BINDING_COMPATIBILITY_PLAN_INTERFACE: Final[str] = "BindingCompatibilityPlan@1"
BINDING_COMPATIBILITY_RECEIPT_INTERFACE: Final[str] = (
    "BindingCompatibilityReceipt@1"
)

BINDING_COMPATIBILITY_ADAPTER_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/binding-compatibility-adapter@1"
)
BINDING_COMPATIBILITY_PLAN_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/binding-compatibility-plan@1"
)
BINDING_COMPATIBILITY_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/binding-compatibility-receipt@1"
)

ADAPTER_CONTRACT_VERSION: Final[str] = "1"

ADAPTER_CAN_AUTHORIZE_TRANSITION: Final[bool] = False
ADAPTER_CAN_AUTHORIZE_COMPLETION: Final[bool] = False
ADAPTER_CAN_CREATE_AUTHORITY: Final[bool] = False
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
ADAPTER_IS_NOMINATION_ONLY: Final[bool] = True
RAW_SOURCE_REQUIRED: Final[bool] = True
DRY_RUN_IS_DETERMINISTIC: Final[bool] = True
DRY_RUN_MUTATES: Final[bool] = False
RUNTIME_OBSERVATION_IS_NOT_STATIC_FACT: Final[bool] = True
SILENT_INCOMPATIBILITY_REJECTED: Final[bool] = True
INTENTIONAL_INCOMPATIBILITY_MUST_BE_CLASSIFIED: Final[bool] = True

MAX_TEXT_CHARS: Final[int] = 16_384
MAX_MEMBERS: Final[int] = 16_384
MAX_ADAPTERS: Final[int] = 16_384
MAX_WRITE_PATHS: Final[int] = 64
MAX_PATH_CHARS: Final[int] = 1_024
MAX_OBLIGATIONS: Final[int] = 16_384

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

_NON_ADMITTING_EVIDENCE: Final[frozenset[str]] = frozenset(
    {
        "vector_candidate",
        "model_hypothesis",
        "heuristic",
    }
)

HANDLED_ADAPTER_KINDS: Final[frozenset[str]] = frozenset(
    {
        AdapterKind.WRAPPER.value,
    }
)

HANDLED_REWRITE_KINDS: Final[frozenset[str]] = frozenset(
    {
        RewriteKind.SERIALIZATION.value,
        RewriteKind.INTROSPECTION.value,
        RewriteKind.TRACEBACK.value,
        RewriteKind.PATCH_TARGET.value,
        RewriteKind.DEPRECATION.value,
    }
)

HANDLED_MIGRATION_KINDS: Final[frozenset[str]] = frozenset(
    {
        "wrapper",
        "serialization",
        "introspection",
        "traceback",
        "deprecation",
        "patch_target",
    }
)

REQUIRED_UNSUPPORTED_DISPOSITIONS: Final[frozenset[str]] = frozenset(
    {
        "undispositioned",
        "unsupported",
    }
)

BINDING_IDENTITY_KINDS: Final[frozenset[str]] = frozenset(
    {
        "module_name",
        "qualname",
        "pickle",
        "serialization",
    }
)


class BindingCompatibilityError(ValueError):
    """Fail-closed violation of a SPAR-024 binding-compatibility contract."""


class BindingCompatibilityKind(str, Enum):
    SIGNATURE = "signature"
    ANNOTATION = "annotation"
    MODULE_NAME = "module_name"
    QUALNAME = "qualname"
    PICKLE = "pickle"
    SERIALIZATION = "serialization"
    INTROSPECTION = "introspection"
    TRACEBACK = "traceback"
    DOCUMENTATION = "documentation"
    PATCH_TARGET = "patch_target"


class IncompatibilityClassification(str, Enum):
    PRESERVE = "preserve"
    WRAPPER = "wrapper"
    MIGRATION = "migration"
    EXPLICIT_INCOMPATIBILITY = "explicit_incompatibility"


DECLARED_BINDING_COMPATIBILITY_KINDS: Final[frozenset[str]] = frozenset(
    kind.value for kind in BindingCompatibilityKind
)
DECLARED_INCOMPATIBILITY_CLASSIFICATIONS: Final[frozenset[str]] = frozenset(
    kind.value for kind in IncompatibilityClassification
)

_REWRITE_TO_KIND: Final[Mapping[str, str]] = {
    RewriteKind.SERIALIZATION.value: BindingCompatibilityKind.PICKLE.value,
    RewriteKind.INTROSPECTION.value: BindingCompatibilityKind.INTROSPECTION.value,
    RewriteKind.TRACEBACK.value: BindingCompatibilityKind.TRACEBACK.value,
    RewriteKind.PATCH_TARGET.value: BindingCompatibilityKind.PATCH_TARGET.value,
    RewriteKind.DEPRECATION.value: BindingCompatibilityKind.DOCUMENTATION.value,
}

_ADAPTER_TO_KIND: Final[Mapping[str, str]] = {
    AdapterKind.WRAPPER.value: BindingCompatibilityKind.SIGNATURE.value,
}

_MIGRATION_TO_KIND: Final[Mapping[str, str]] = {
    "wrapper": BindingCompatibilityKind.SIGNATURE.value,
    "serialization": BindingCompatibilityKind.PICKLE.value,
    "introspection": BindingCompatibilityKind.INTROSPECTION.value,
    "traceback": BindingCompatibilityKind.TRACEBACK.value,
    "deprecation": BindingCompatibilityKind.DOCUMENTATION.value,
    "patch_target": BindingCompatibilityKind.PATCH_TARGET.value,
}

_KIND_TO_MIGRATION: Final[Mapping[str, str]] = {
    BindingCompatibilityKind.SIGNATURE.value: "wrapper",
    BindingCompatibilityKind.ANNOTATION.value: "wrapper",
    BindingCompatibilityKind.MODULE_NAME.value: "wrapper",
    BindingCompatibilityKind.QUALNAME.value: "wrapper",
    BindingCompatibilityKind.PICKLE.value: "serialization",
    BindingCompatibilityKind.SERIALIZATION.value: "serialization",
    BindingCompatibilityKind.INTROSPECTION.value: "introspection",
    BindingCompatibilityKind.TRACEBACK.value: "traceback",
    BindingCompatibilityKind.DOCUMENTATION.value: "deprecation",
    BindingCompatibilityKind.PATCH_TARGET.value: "patch_target",
}


def _text(value: Any, name: str, *, empty: bool = False) -> str:
    if type(value) is not str:
        raise BindingCompatibilityError(f"{name} must be a string")
    if value != value.strip() or unicodedata.normalize("NFC", value) != value:
        raise BindingCompatibilityError(f"{name} must be trimmed NFC text")
    if not empty and not value:
        raise BindingCompatibilityError(f"{name} must be a nonempty string")
    if any(not char.isprintable() for char in value):
        raise BindingCompatibilityError(f"{name} contains invalid text")
    if len(value) > MAX_TEXT_CHARS:
        raise BindingCompatibilityError(f"{name} exceeds text bound")
    return value


def _cid(value: Any, name: str) -> str:
    text = _text(value, name)
    try:
        return validate_cid(text)
    except Exception as exc:
        raise BindingCompatibilityError(f"{name} must be a valid CID") from exc


def _bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise BindingCompatibilityError(f"{name} must be a boolean")
    return value


def _tree_id(value: Any) -> str:
    text = _text(value, "tree_id")
    if len(text) not in {40, 64} or any(
        char not in "0123456789abcdef" for char in text
    ):
        raise BindingCompatibilityError(
            "tree_id must be a lowercase hex Git tree identity"
        )
    return text


def _closed(data: Mapping[str, Any], fields: frozenset[str], name: str) -> dict[str, Any]:
    if not isinstance(data, Mapping) or isinstance(data, (str, bytes, bytearray)):
        raise BindingCompatibilityError(f"{name} must be an object")
    extra = set(data) - fields
    missing = fields - set(data)
    if extra & IDENTITY_EXCLUDED_FIELDS:
        raise BindingCompatibilityError(
            f"{name} identity excludes observational fields: "
            f"{sorted(extra & IDENTITY_EXCLUDED_FIELDS)}"
        )
    if extra:
        raise BindingCompatibilityError(f"unknown {name} field: {sorted(extra)}")
    if missing:
        raise BindingCompatibilityError(f"missing {name} field: {sorted(missing)}")
    return dict(data)


def _reject_excluded(payload: Mapping[str, Any], name: str) -> None:
    present = IDENTITY_EXCLUDED_FIELDS & set(payload)
    if present:
        raise BindingCompatibilityError(
            f"{name} identity excludes observational fields: {sorted(present)}"
        )


def _require_dag_json(value: Any, name: str) -> None:
    try:
        cid_for_dag_json(value)
    except Exception as exc:
        raise BindingCompatibilityError(f"{name} must be strict DAG-JSON") from exc


def _verify_cid(claimed: Any, computed: str, name: str) -> None:
    cid = _cid(claimed, name)
    if cid != computed:
        raise BindingCompatibilityError(f"{name} does not verify")


def _unique_sorted_text(values: Any, name: str, *, limit: int) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        raise BindingCompatibilityError(f"{name} must be a list")
    ordered = tuple(sorted(_text(item, name) for item in values))
    if len(ordered) > limit:
        raise BindingCompatibilityError(f"{name} exceeds maximum length")
    if len(ordered) != len(set(ordered)):
        raise BindingCompatibilityError(f"{name} must not contain duplicates")
    return ordered


def _enum(value: Any, enum_type: type[Enum], name: str) -> str:
    if isinstance(value, enum_type):
        return value.value
    text = _text(value, name)
    try:
        return enum_type(text).value
    except ValueError as exc:
        raise BindingCompatibilityError(f"unknown {name}: {text}") from exc


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
    raise BindingCompatibilityError(
        f"unsupported projected type {type(value).__name__}"
    )


def _mapping(value: Any, name: str) -> dict[str, Any]:
    projected = _project(value)
    if not isinstance(projected, dict):
        raise BindingCompatibilityError(f"{name} must be an object")
    _reject_excluded(projected, name)
    return projected


def binding_compatibility_cid_profile() -> dict[str, str]:
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
            raise BindingCompatibilityError(f"{name} cannot claim {flag}")


def _exact_path(value: Any, name: str = "write_paths") -> str:
    raw = _text(value, name, empty=False)
    if len(raw) > MAX_PATH_CHARS:
        raise BindingCompatibilityError(f"{name} exceeds path bound")
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
        raise BindingCompatibilityError(
            f"{name} must be an exact repository-relative path; unrestricted scope is rejected"
        )
    if normalized != candidate.as_posix():
        raise BindingCompatibilityError(
            f"{name} must be a normalized repository-relative path"
        )
    return normalized


def _exact_paths(values: Any, name: str, *, required: bool = True) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise BindingCompatibilityError(f"{name} must be a list of exact paths")
    ordered: list[str] = []
    seen: set[str] = set()
    for item in values:
        path = _exact_path(item, name)
        if path not in seen:
            seen.add(path)
            ordered.append(path)
    if required and not ordered:
        raise BindingCompatibilityError(
            f"{name} must not be empty; unrestricted scope is rejected"
        )
    if len(ordered) > MAX_WRITE_PATHS:
        raise BindingCompatibilityError(f"{name} exceeds path bound")
    return tuple(ordered)


def _required_bool(value: Any, name: str = "required") -> bool:
    if value is None:
        return True
    return _bool(value, name)


@dataclass(frozen=True, slots=True)
class BindingCompatibilityAdapter:
    """One exact signature, binding, pickle, reflection, docs, or patch wrapper.

    Nomination only.
    """

    adapter_kind: BindingCompatibilityKind | str
    subject_id: str
    source_module: str
    destination_module: str
    write_paths: Sequence[str]
    preimage_cid: str
    packet_cid: str
    tree_id: str
    classification: IncompatibilityClassification | str
    obligation_id: str = ""
    consumer_id: str = ""
    migration_kind: str = ""
    packet_adapter_kind: str = ""
    packet_rewrite_kind: str = ""
    preserve_binding: bool = True

    interface: ClassVar[str] = BINDING_COMPATIBILITY_ADAPTER_INTERFACE
    schema: ClassVar[str] = BINDING_COMPATIBILITY_ADAPTER_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "adapter_kind",
            "subject_id",
            "source_module",
            "destination_module",
            "write_paths",
            "preimage_cid",
            "packet_cid",
            "tree_id",
            "classification",
            "obligation_id",
            "consumer_id",
            "migration_kind",
            "packet_adapter_kind",
            "packet_rewrite_kind",
            "preserve_binding",
            "can_authorize_transition",
            "can_authorize_completion",
            "can_create_authority",
            "projection_is_authority",
            "adapter_is_nomination_only",
            "adapter_cid",
        }
    )

    def __post_init__(self) -> None:
        kind = _enum(self.adapter_kind, BindingCompatibilityKind, "adapter_kind")
        classification = _enum(
            self.classification, IncompatibilityClassification, "classification"
        )
        source = _text(self.source_module, "source_module")
        destination = _text(self.destination_module, "destination_module")
        preserve = _bool(self.preserve_binding, "preserve_binding")
        if (
            source == destination
            and not preserve
            and classification
            != IncompatibilityClassification.EXPLICIT_INCOMPATIBILITY.value
        ):
            raise BindingCompatibilityError(
                "adapter source and destination must differ unless preserve_binding "
                "or explicit_incompatibility"
            )
        if (
            classification == IncompatibilityClassification.PRESERVE.value
            and not preserve
        ):
            raise BindingCompatibilityError(
                "preserve classification requires preserve_binding"
            )
        if (
            kind in BINDING_IDENTITY_KINDS
            and source != destination
            and classification == IncompatibilityClassification.PRESERVE.value
        ):
            raise BindingCompatibilityError(
                "silent binding incompatibility is a typed terminal"
            )
        if (
            classification
            == IncompatibilityClassification.EXPLICIT_INCOMPATIBILITY.value
            and preserve
        ):
            raise BindingCompatibilityError(
                "explicit_incompatibility cannot claim preserve_binding"
            )
        migration = _text(self.migration_kind, "migration_kind", empty=True)
        if not migration:
            migration = _KIND_TO_MIGRATION[kind]
        if migration not in HANDLED_MIGRATION_KINDS:
            raise BindingCompatibilityError("unknown migration_kind")
        expected_migration = _KIND_TO_MIGRATION[kind]
        if (
            classification != IncompatibilityClassification.EXPLICIT_INCOMPATIBILITY.value
            and migration != expected_migration
        ):
            raise BindingCompatibilityError(
                "migration_kind does not match adapter_kind"
            )
        packet_adapter = _text(
            self.packet_adapter_kind, "packet_adapter_kind", empty=True
        )
        if packet_adapter and packet_adapter not in HANDLED_ADAPTER_KINDS:
            raise BindingCompatibilityError("unknown packet_adapter_kind")
        packet_rewrite = _text(
            self.packet_rewrite_kind, "packet_rewrite_kind", empty=True
        )
        if packet_rewrite and packet_rewrite not in HANDLED_REWRITE_KINDS:
            raise BindingCompatibilityError("unknown packet_rewrite_kind")
        if packet_adapter and packet_rewrite:
            raise BindingCompatibilityError(
                "adapter cannot declare both packet_adapter_kind and packet_rewrite_kind"
            )
        object.__setattr__(self, "adapter_kind", kind)
        object.__setattr__(self, "subject_id", _text(self.subject_id, "subject_id"))
        object.__setattr__(self, "source_module", source)
        object.__setattr__(self, "destination_module", destination)
        object.__setattr__(
            self,
            "write_paths",
            _exact_paths(list(self.write_paths), "write_paths", required=True),
        )
        object.__setattr__(self, "preimage_cid", _cid(self.preimage_cid, "preimage_cid"))
        object.__setattr__(self, "packet_cid", _cid(self.packet_cid, "packet_cid"))
        object.__setattr__(self, "tree_id", _tree_id(self.tree_id))
        object.__setattr__(self, "classification", classification)
        object.__setattr__(
            self, "obligation_id", _text(self.obligation_id, "obligation_id", empty=True)
        )
        object.__setattr__(
            self, "consumer_id", _text(self.consumer_id, "consumer_id", empty=True)
        )
        object.__setattr__(self, "migration_kind", migration)
        object.__setattr__(self, "packet_adapter_kind", packet_adapter)
        object.__setattr__(self, "packet_rewrite_kind", packet_rewrite)
        object.__setattr__(self, "preserve_binding", preserve)

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
    def adapter_is_nomination_only(self) -> bool:
        return True

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": BINDING_COMPATIBILITY_ADAPTER_SCHEMA,
            "interface": BINDING_COMPATIBILITY_ADAPTER_INTERFACE,
            "adapter_kind": self.adapter_kind,
            "subject_id": self.subject_id,
            "source_module": self.source_module,
            "destination_module": self.destination_module,
            "write_paths": list(self.write_paths),
            "preimage_cid": self.preimage_cid,
            "packet_cid": self.packet_cid,
            "tree_id": self.tree_id,
            "classification": self.classification,
            "obligation_id": self.obligation_id,
            "consumer_id": self.consumer_id,
            "migration_kind": self.migration_kind,
            "packet_adapter_kind": self.packet_adapter_kind,
            "packet_rewrite_kind": self.packet_rewrite_kind,
            "preserve_binding": self.preserve_binding,
            "can_authorize_transition": False,
            "can_authorize_completion": False,
            "can_create_authority": False,
            "projection_is_authority": False,
            "adapter_is_nomination_only": True,
        }
        _require_dag_json(payload, self.__class__.__name__)
        return payload

    @property
    def adapter_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["adapter_cid"] = self.adapter_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BindingCompatibilityAdapter":
        _reject_excluded(data, cls.__name__)
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("adapter_cid")
        if payload.pop("schema") != BINDING_COMPATIBILITY_ADAPTER_SCHEMA:
            raise BindingCompatibilityError(
                "unsupported BindingCompatibilityAdapter schema"
            )
        if payload.pop("interface") != BINDING_COMPATIBILITY_ADAPTER_INTERFACE:
            raise BindingCompatibilityError(
                "unsupported BindingCompatibilityAdapter interface"
            )
        _pop_authority_flags(payload, "BindingCompatibilityAdapter")
        if payload.pop("adapter_is_nomination_only") is not True:
            raise BindingCompatibilityError("adapter must remain nomination_only")
        result = cls(**payload)
        _verify_cid(
            claimed, result.adapter_cid, "BindingCompatibilityAdapter adapter_cid"
        )
        return result


@dataclass(frozen=True, slots=True)
class BindingCompatibilityPlan:
    """Deterministic SPAR-024 adapter plan. Nomination only."""

    adapter_cids: Sequence[str]
    write_paths: Sequence[str]
    preimage_cid: str
    packet_cid: str
    tree_id: str
    obligation_ids: Sequence[str] = ()
    classified_incompatibility_ids: Sequence[str] = ()
    inventory_cid: str = ""
    preserve_binding: bool = True
    silent_incompatibility: bool = False

    interface: ClassVar[str] = BINDING_COMPATIBILITY_PLAN_INTERFACE
    schema: ClassVar[str] = BINDING_COMPATIBILITY_PLAN_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "adapter_cids",
            "write_paths",
            "preimage_cid",
            "packet_cid",
            "tree_id",
            "obligation_ids",
            "classified_incompatibility_ids",
            "inventory_cid",
            "preserve_binding",
            "silent_incompatibility",
            "can_authorize_transition",
            "can_authorize_completion",
            "can_create_authority",
            "projection_is_authority",
            "plan_is_nomination_only",
            "plan_cid",
        }
    )

    def __post_init__(self) -> None:
        if _bool(self.silent_incompatibility, "silent_incompatibility"):
            raise BindingCompatibilityError(
                "silent binding incompatibility is a typed terminal"
            )
        adapters = tuple(sorted(_cid(item, "adapter_cids") for item in self.adapter_cids))
        if not adapters:
            raise BindingCompatibilityError("plan requires adapter_cids")
        if len(adapters) != len(set(adapters)):
            raise BindingCompatibilityError("adapter_cids must not contain duplicates")
        if len(adapters) > MAX_ADAPTERS:
            raise BindingCompatibilityError("adapter_cids exceed maximum length")
        object.__setattr__(self, "adapter_cids", adapters)
        object.__setattr__(
            self,
            "write_paths",
            _exact_paths(list(self.write_paths), "write_paths", required=True),
        )
        object.__setattr__(self, "preimage_cid", _cid(self.preimage_cid, "preimage_cid"))
        object.__setattr__(self, "packet_cid", _cid(self.packet_cid, "packet_cid"))
        object.__setattr__(self, "tree_id", _tree_id(self.tree_id))
        object.__setattr__(
            self,
            "obligation_ids",
            _unique_sorted_text(
                list(self.obligation_ids), "obligation_ids", limit=MAX_OBLIGATIONS
            ),
        )
        object.__setattr__(
            self,
            "classified_incompatibility_ids",
            _unique_sorted_text(
                list(self.classified_incompatibility_ids),
                "classified_incompatibility_ids",
                limit=MAX_OBLIGATIONS,
            ),
        )
        inventory = _text(self.inventory_cid, "inventory_cid", empty=True)
        if inventory:
            inventory = _cid(inventory, "inventory_cid")
        object.__setattr__(self, "inventory_cid", inventory)
        object.__setattr__(
            self, "preserve_binding", _bool(self.preserve_binding, "preserve_binding")
        )
        object.__setattr__(self, "silent_incompatibility", False)

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

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": BINDING_COMPATIBILITY_PLAN_SCHEMA,
            "interface": BINDING_COMPATIBILITY_PLAN_INTERFACE,
            "adapter_cids": list(self.adapter_cids),
            "write_paths": list(self.write_paths),
            "preimage_cid": self.preimage_cid,
            "packet_cid": self.packet_cid,
            "tree_id": self.tree_id,
            "obligation_ids": list(self.obligation_ids),
            "classified_incompatibility_ids": list(self.classified_incompatibility_ids),
            "inventory_cid": self.inventory_cid,
            "preserve_binding": self.preserve_binding,
            "silent_incompatibility": False,
            "can_authorize_transition": False,
            "can_authorize_completion": False,
            "can_create_authority": False,
            "projection_is_authority": False,
            "plan_is_nomination_only": True,
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
    def from_dict(cls, data: Mapping[str, Any]) -> "BindingCompatibilityPlan":
        _reject_excluded(data, cls.__name__)
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("plan_cid")
        if payload.pop("schema") != BINDING_COMPATIBILITY_PLAN_SCHEMA:
            raise BindingCompatibilityError(
                "unsupported BindingCompatibilityPlan schema"
            )
        if payload.pop("interface") != BINDING_COMPATIBILITY_PLAN_INTERFACE:
            raise BindingCompatibilityError(
                "unsupported BindingCompatibilityPlan interface"
            )
        _pop_authority_flags(payload, "BindingCompatibilityPlan")
        if payload.pop("plan_is_nomination_only") is not True:
            raise BindingCompatibilityError("plan must remain nomination_only")
        result = cls(**payload)
        _verify_cid(claimed, result.plan_cid, "BindingCompatibilityPlan plan_cid")
        return result


@dataclass(frozen=True, slots=True)
class BindingCompatibilityReceipt:
    """Body-free SPAR-024 execution receipt. Independent validation remains separate."""

    tree_id: str
    packet_cid: str
    preimage_cid: str
    adapter_cids: Sequence[str]
    plan_cid: str
    write_paths: Sequence[str]
    classified_incompatibility_ids: Sequence[str] = ()
    inventory_cid: str = ""
    analyzer_id: str = ANALYZER_ID
    preserve_binding: bool = True
    silent_incompatibility: bool = False
    preimage_verified: bool = True
    mutated: bool = False
    deterministic: bool = True

    interface: ClassVar[str] = BINDING_COMPATIBILITY_RECEIPT_INTERFACE
    schema: ClassVar[str] = BINDING_COMPATIBILITY_RECEIPT_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "tree_id",
            "packet_cid",
            "preimage_cid",
            "adapter_cids",
            "plan_cid",
            "write_paths",
            "classified_incompatibility_ids",
            "inventory_cid",
            "analyzer_id",
            "preserve_binding",
            "silent_incompatibility",
            "preimage_verified",
            "mutated",
            "deterministic",
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
            raise BindingCompatibilityError(
                "analyzer_id must remain the SPAR-024 analyzer"
            )
        if _bool(self.silent_incompatibility, "silent_incompatibility"):
            raise BindingCompatibilityError(
                "silent binding incompatibility is a typed terminal"
            )
        if not _bool(self.preimage_verified, "preimage_verified"):
            raise BindingCompatibilityError(
                "receipt cannot skip preimage verification"
            )
        if _bool(self.mutated, "mutated"):
            raise BindingCompatibilityError("executor cannot mutate")
        if not _bool(self.deterministic, "deterministic"):
            raise BindingCompatibilityError("executor must remain deterministic")
        adapters = tuple(sorted(_cid(item, "adapter_cids") for item in self.adapter_cids))
        if not adapters:
            raise BindingCompatibilityError("receipt requires adapter_cids")
        if len(adapters) != len(set(adapters)):
            raise BindingCompatibilityError("adapter_cids must not contain duplicates")
        if len(adapters) > MAX_ADAPTERS:
            raise BindingCompatibilityError("adapter_cids exceed maximum length")
        inventory = _text(self.inventory_cid, "inventory_cid", empty=True)
        if inventory:
            inventory = _cid(inventory, "inventory_cid")
        object.__setattr__(self, "tree_id", _tree_id(self.tree_id))
        object.__setattr__(self, "packet_cid", _cid(self.packet_cid, "packet_cid"))
        object.__setattr__(self, "preimage_cid", _cid(self.preimage_cid, "preimage_cid"))
        object.__setattr__(self, "adapter_cids", adapters)
        object.__setattr__(self, "plan_cid", _cid(self.plan_cid, "plan_cid"))
        object.__setattr__(
            self,
            "write_paths",
            _exact_paths(list(self.write_paths), "write_paths", required=True),
        )
        object.__setattr__(
            self,
            "classified_incompatibility_ids",
            _unique_sorted_text(
                list(self.classified_incompatibility_ids),
                "classified_incompatibility_ids",
                limit=MAX_OBLIGATIONS,
            ),
        )
        object.__setattr__(self, "inventory_cid", inventory)
        object.__setattr__(self, "analyzer_id", analyzer)
        object.__setattr__(
            self, "preserve_binding", _bool(self.preserve_binding, "preserve_binding")
        )
        object.__setattr__(self, "silent_incompatibility", False)
        object.__setattr__(self, "preimage_verified", True)
        object.__setattr__(self, "mutated", False)
        object.__setattr__(self, "deterministic", True)

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
            "schema": BINDING_COMPATIBILITY_RECEIPT_SCHEMA,
            "interface": BINDING_COMPATIBILITY_RECEIPT_INTERFACE,
            "tree_id": self.tree_id,
            "packet_cid": self.packet_cid,
            "preimage_cid": self.preimage_cid,
            "adapter_cids": list(self.adapter_cids),
            "plan_cid": self.plan_cid,
            "write_paths": list(self.write_paths),
            "classified_incompatibility_ids": list(self.classified_incompatibility_ids),
            "inventory_cid": self.inventory_cid,
            "analyzer_id": self.analyzer_id,
            "preserve_binding": self.preserve_binding,
            "silent_incompatibility": False,
            "preimage_verified": True,
            "mutated": False,
            "deterministic": True,
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
    def from_dict(cls, data: Mapping[str, Any]) -> "BindingCompatibilityReceipt":
        _reject_excluded(data, cls.__name__)
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("receipt_cid")
        if payload.pop("schema") != BINDING_COMPATIBILITY_RECEIPT_SCHEMA:
            raise BindingCompatibilityError(
                "unsupported BindingCompatibilityReceipt schema"
            )
        if payload.pop("interface") != BINDING_COMPATIBILITY_RECEIPT_INTERFACE:
            raise BindingCompatibilityError(
                "unsupported BindingCompatibilityReceipt interface"
            )
        _pop_authority_flags(payload, "BindingCompatibilityReceipt")
        if payload.pop("executor_is_nomination_only") is not True:
            raise BindingCompatibilityError("executor must remain nomination_only")
        result = cls(**payload)
        _verify_cid(
            claimed, result.receipt_cid, "BindingCompatibilityReceipt receipt_cid"
        )
        return result


def _coerce_packet(
    value: RefactorTransformationPacket | Mapping[str, Any],
) -> RefactorTransformationPacket:
    if isinstance(value, RefactorTransformationPacket):
        packet = value
    elif isinstance(value, Mapping):
        packet = RefactorTransformationPacket.from_dict(value)
    else:
        raise BindingCompatibilityError(
            "packet must be a SPAR-019 RefactorTransformationPacket"
        )
    if packet.analyzer_id != SPAR019_ANALYZER_ID:
        raise BindingCompatibilityError("packet must remain the SPAR-019 analyzer")
    return packet


def _as_mapping_list(value: Any, name: str) -> tuple[dict[str, Any], ...]:
    if value in (None, ()):
        return ()
    if not isinstance(value, (list, tuple)):
        raise BindingCompatibilityError(f"{name} must be a list")
    items: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, Mapping):
            items.append(_mapping(item, name))
            continue
        to_dict = getattr(item, "to_dict", None)
        if callable(to_dict):
            items.append(_mapping(to_dict(), name))
            continue
        raise BindingCompatibilityError(f"{name} entries must be objects")
    return tuple(items)


def _reject_non_admitting(payload: Mapping[str, Any], name: str) -> None:
    evidence = _text(
        payload.get("evidence_class") or payload.get("evidence") or "",
        "evidence_class",
        empty=True,
    )
    if evidence in _NON_ADMITTING_EVIDENCE:
        raise BindingCompatibilityError(
            f"vector or model evidence cannot admit {name}"
        )


def _reject_undispositioned(
    *,
    undispositioned_consumer_ids: Sequence[str],
    consumer_plans: Sequence[Mapping[str, Any]],
    obligations: Sequence[Mapping[str, Any]],
) -> None:
    leftover = _unique_sorted_text(
        list(undispositioned_consumer_ids),
        "undispositioned_consumer_ids",
        limit=MAX_MEMBERS,
    )
    if leftover:
        raise BindingCompatibilityError(
            "undispositioned consumers are a typed terminal"
        )
    records = list(consumer_plans) + list(obligations)
    for plan in records:
        required = _required_bool(plan.get("required", True), "required")
        disposition = _text(plan.get("disposition") or "", "disposition", empty=True)
        if required and disposition in REQUIRED_UNSUPPORTED_DISPOSITIONS:
            raise BindingCompatibilityError(
                "undispositioned consumers are a typed terminal"
            )


def verify_preimages(
    packet: RefactorTransformationPacket | Mapping[str, Any],
    *,
    claimed_preimage_cid: str = "",
    source_cids: Sequence[str] | None = None,
) -> str:
    """Verify SPAR-019 packet preimages. Does not authorize a transition."""

    resolved = _coerce_packet(packet)
    preimage_cid = resolved.preimage.preimage_cid
    if claimed_preimage_cid:
        claimed = _cid(claimed_preimage_cid, "claimed_preimage_cid")
        if claimed != preimage_cid:
            raise BindingCompatibilityError("preimage does not verify")
    if source_cids is not None:
        claimed_sources = _unique_sorted_text(
            list(source_cids), "source_cids", limit=MAX_MEMBERS
        )
        if set(claimed_sources) != set(resolved.preimage.source_cids):
            raise BindingCompatibilityError("preimage does not verify")
    return preimage_cid


def classify_intentional_incompatibility(
    *,
    disposition: str,
    adapter_kind: str,
    source_module: str,
    destination_module: str,
) -> str:
    """Classify preserve/wrapper/migration/explicit incompatibility. Nomination only."""

    kind = _enum(adapter_kind, BindingCompatibilityKind, "adapter_kind")
    source = _text(source_module, "source_module")
    destination = _text(destination_module, "destination_module")
    disp = _text(disposition, "disposition", empty=True)
    if disp in REQUIRED_UNSUPPORTED_DISPOSITIONS:
        raise BindingCompatibilityError(
            "unsupported required compatibility obligation is a typed terminal"
        )
    if disp == IncompatibilityClassification.EXPLICIT_INCOMPATIBILITY.value:
        return IncompatibilityClassification.EXPLICIT_INCOMPATIBILITY.value
    if disp == IncompatibilityClassification.PRESERVE.value:
        if kind in BINDING_IDENTITY_KINDS and source != destination:
            raise BindingCompatibilityError(
                "silent binding incompatibility is a typed terminal"
            )
        return IncompatibilityClassification.PRESERVE.value
    if disp in {"", "migrate", "facade"}:
        migration = _KIND_TO_MIGRATION[kind]
        if migration == "wrapper":
            return IncompatibilityClassification.WRAPPER.value
        return IncompatibilityClassification.MIGRATION.value
    raise BindingCompatibilityError(f"unknown disposition: {disp}")


def _coerce_inventory(
    value: Any,
    tree_id: str,
) -> dict[str, Any]:
    if value in (None, (), {}):
        return {
            "tree_id": tree_id,
            "inventory_cid": "",
            "obligations": (),
        }
    payload = _mapping(value, "SPAR-011 inventory")
    _reject_non_admitting(payload, "SPAR-011 inventory")
    inventory_tree = payload.get("tree_id")
    if inventory_tree not in {None, ""}:
        if _tree_id(inventory_tree) != tree_id:
            raise BindingCompatibilityError("SPAR-011 tree_id does not match packet")
    inventory_cid = _text(payload.get("inventory_cid") or "", "inventory_cid", empty=True)
    if inventory_cid:
        inventory_cid = _cid(inventory_cid, "inventory_cid")
    obligations = _as_mapping_list(payload.get("obligations") or (), "obligations")
    if not obligations:
        raise BindingCompatibilityError("SPAR-011 inventory requires obligations")
    for item in obligations:
        _reject_non_admitting(item, "SPAR-011 obligation")
        kind = _text(item.get("kind") or "", "kind", empty=True)
        if kind and kind not in DECLARED_BINDING_COMPATIBILITY_KINDS:
            continue
        if not _text(item.get("obligation_id") or "", "obligation_id", empty=True):
            raise BindingCompatibilityError("SPAR-011 obligation_id is required")
    return {
        "tree_id": tree_id,
        "inventory_cid": inventory_cid,
        "obligations": obligations,
    }


def _overlay_for(
    *,
    obligation_id: str,
    subject_id: str,
    derived_kind: str,
    consumer_plans: Sequence[Mapping[str, Any]],
    obligations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    for item in (*obligations, *consumer_plans):
        item_obligation = _text(
            item.get("obligation_id") or "", "obligation_id", empty=True
        )
        item_subject = _text(item.get("subject_id") or "", "subject_id", empty=True)
        item_kind = _text(item.get("kind") or "", "kind", empty=True)
        if obligation_id and item_obligation == obligation_id:
            return item
        if subject_id and item_subject == subject_id:
            if not item_kind or item_kind == derived_kind:
                return item
    return {}


def _kind_for(
    *,
    overlay: Mapping[str, Any],
    packet_adapter_kind: str,
    packet_rewrite_kind: str,
    migration_kind: str,
) -> str:
    declared = _text(overlay.get("kind") or "", "kind", empty=True)
    if declared:
        if declared not in DECLARED_BINDING_COMPATIBILITY_KINDS:
            raise BindingCompatibilityError(
                "packet requires binding, introspection, or serialization edits"
            )
        return declared
    if packet_rewrite_kind:
        return _REWRITE_TO_KIND[packet_rewrite_kind]
    if packet_adapter_kind:
        return _ADAPTER_TO_KIND[packet_adapter_kind]
    if migration_kind:
        return _MIGRATION_TO_KIND[migration_kind]
    raise BindingCompatibilityError(
        "packet requires binding, introspection, or serialization edits"
    )


def _handled_edits(packet: RefactorTransformationPacket) -> tuple[Any, ...]:
    nominated = []
    for item in packet.edits:
        if (
            item.kind == EditKind.ADAPTER.value
            and item.adapter_kind in HANDLED_ADAPTER_KINDS
        ):
            nominated.append(item)
            continue
        if (
            item.kind == EditKind.REWRITE.value
            and item.rewrite_kind in HANDLED_REWRITE_KINDS
        ):
            nominated.append(item)
    return tuple(nominated)


def _sort_adapters(
    items: Sequence[BindingCompatibilityAdapter],
) -> tuple[BindingCompatibilityAdapter, ...]:
    unique: dict[str, BindingCompatibilityAdapter] = {}
    for item in items:
        unique[item.adapter_cid] = item
    ordered = tuple(
        sorted(
            unique.values(),
            key=lambda item: (
                item.adapter_kind,
                item.subject_id,
                item.source_module,
                item.destination_module,
                item.obligation_id,
                item.consumer_id,
                item.classification,
                item.adapter_cid,
            ),
        )
    )
    if len(ordered) > MAX_ADAPTERS:
        raise BindingCompatibilityError(
            "binding compatibility adapters exceed maximum length"
        )
    return ordered


def _collect_packet_adapters(
    packet: RefactorTransformationPacket,
    *,
    consumer_plans: Sequence[Mapping[str, Any]],
    obligations: Sequence[Mapping[str, Any]],
) -> tuple[BindingCompatibilityAdapter, ...]:
    nominated: list[BindingCompatibilityAdapter] = []
    for edit in _handled_edits(packet):
        members = edit.member_ids or (edit.obligation_id or edit.source_id,)
        for subject_id in members:
            derived_kind = _kind_for(
                overlay={},
                packet_adapter_kind=edit.adapter_kind,
                packet_rewrite_kind=edit.rewrite_kind,
                migration_kind="",
            )
            overlay = _overlay_for(
                obligation_id=edit.obligation_id,
                subject_id=subject_id,
                derived_kind=derived_kind,
                consumer_plans=consumer_plans,
                obligations=obligations,
            )
            kind = _kind_for(
                overlay=overlay,
                packet_adapter_kind=edit.adapter_kind,
                packet_rewrite_kind=edit.rewrite_kind,
                migration_kind=_text(
                    overlay.get("migration_kind") or "", "migration_kind", empty=True
                ),
            )
            classification = classify_intentional_incompatibility(
                disposition=_text(
                    overlay.get("disposition") or "", "disposition", empty=True
                ),
                adapter_kind=kind,
                source_module=edit.source_id,
                destination_module=edit.destination_id,
            )
            preserve = (
                classification == IncompatibilityClassification.PRESERVE.value
                or (
                    classification != IncompatibilityClassification.EXPLICIT_INCOMPATIBILITY.value
                    and edit.source_id == edit.destination_id
                )
            )
            if classification == IncompatibilityClassification.EXPLICIT_INCOMPATIBILITY.value:
                preserve = False
            nominated.append(
                BindingCompatibilityAdapter(
                    adapter_kind=kind,
                    subject_id=subject_id,
                    source_module=edit.source_id,
                    destination_module=edit.destination_id,
                    write_paths=edit.write_paths,
                    preimage_cid=packet.preimage.preimage_cid,
                    packet_cid=packet.packet_cid,
                    tree_id=packet.tree_id,
                    classification=classification,
                    obligation_id=edit.obligation_id
                    or _text(
                        overlay.get("obligation_id") or "", "obligation_id", empty=True
                    ),
                    consumer_id=_text(
                        overlay.get("consumer_id") or "", "consumer_id", empty=True
                    ),
                    migration_kind=_KIND_TO_MIGRATION[kind],
                    packet_adapter_kind=edit.adapter_kind,
                    packet_rewrite_kind=edit.rewrite_kind,
                    preserve_binding=preserve,
                )
            )
    return _sort_adapters(nominated)


def _require_handled(
    packet: RefactorTransformationPacket,
    adapters: Sequence[BindingCompatibilityAdapter],
) -> None:
    if adapters:
        return
    raise BindingCompatibilityError(
        "packet requires binding, introspection, or serialization edits"
    )


def _require_classified_inventory(
    *,
    adapters: Sequence[BindingCompatibilityAdapter],
    obligations: Sequence[Mapping[str, Any]],
) -> None:
    nominated = {
        (item.obligation_id, item.adapter_kind)
        for item in adapters
        if item.obligation_id
    }
    nominated_subjects = {
        (item.subject_id, item.adapter_kind) for item in adapters
    }
    for item in obligations:
        kind = _text(item.get("kind") or "", "kind", empty=True)
        if kind not in DECLARED_BINDING_COMPATIBILITY_KINDS:
            continue
        required = _required_bool(item.get("required", True), "required")
        if not required:
            continue
        obligation_id = _text(item.get("obligation_id") or "", "obligation_id")
        subject_id = _text(item.get("subject_id") or "", "subject_id", empty=True)
        if (obligation_id, kind) in nominated:
            continue
        if subject_id and (subject_id, kind) in nominated_subjects:
            continue
        raise BindingCompatibilityError(
            "required SPAR-011 binding obligation is missing a classified adapter"
        )
    explicit = [
        item
        for item in adapters
        if item.classification
        == IncompatibilityClassification.EXPLICIT_INCOMPATIBILITY.value
    ]
    if explicit and not all(item.obligation_id or item.subject_id for item in explicit):
        raise BindingCompatibilityError(
            "intentional incompatibility must be classified"
        )


def compile_binding_compatibility_adapters(
    packet: RefactorTransformationPacket | Mapping[str, Any],
    *,
    inventory: Mapping[str, Any] | None = None,
    consumer_plans: Sequence[Mapping[str, Any]] | None = None,
    undispositioned_consumer_ids: Sequence[str] = (),
    claimed_preimage_cid: str = "",
) -> tuple[BindingCompatibilityAdapter, ...]:
    """Nominate exact binding/introspection/serialization wrappers from a SPAR-019 packet."""

    resolved = _coerce_packet(packet)
    verify_preimages(resolved, claimed_preimage_cid=claimed_preimage_cid)
    plans = _as_mapping_list(consumer_plans or (), "consumer_plans")
    inventory_map = _coerce_inventory(inventory, resolved.tree_id)
    _reject_undispositioned(
        undispositioned_consumer_ids=undispositioned_consumer_ids,
        consumer_plans=plans,
        obligations=inventory_map["obligations"],
    )
    adapters = _collect_packet_adapters(
        resolved,
        consumer_plans=plans,
        obligations=inventory_map["obligations"],
    )
    _require_handled(resolved, adapters)
    _require_classified_inventory(
        adapters=adapters, obligations=inventory_map["obligations"]
    )
    return adapters


def compile_binding_compatibility_plan(
    packet: RefactorTransformationPacket | Mapping[str, Any],
    **kwargs: Any,
) -> BindingCompatibilityPlan:
    """Compile the unique SPAR-024 BindingCompatibilityPlan for one packet."""

    resolved = _coerce_packet(packet)
    inventory_map = _coerce_inventory(kwargs.get("inventory"), resolved.tree_id)
    adapters = compile_binding_compatibility_adapters(resolved, **kwargs)
    obligations = tuple(
        sorted({item.obligation_id for item in adapters if item.obligation_id})
    )
    classified = tuple(
        sorted(
            {
                item.obligation_id or item.subject_id
                for item in adapters
                if item.classification
                == IncompatibilityClassification.EXPLICIT_INCOMPATIBILITY.value
            }
        )
    )
    preserve = all(item.preserve_binding for item in adapters) or not any(
        item.classification
        == IncompatibilityClassification.EXPLICIT_INCOMPATIBILITY.value
        for item in adapters
    )
    if any(
        item.classification
        == IncompatibilityClassification.EXPLICIT_INCOMPATIBILITY.value
        for item in adapters
    ):
        preserve = False
    return BindingCompatibilityPlan(
        adapter_cids=tuple(item.adapter_cid for item in adapters),
        write_paths=resolved.effect_scope.write_paths,
        preimage_cid=resolved.preimage.preimage_cid,
        packet_cid=resolved.packet_cid,
        tree_id=resolved.tree_id,
        obligation_ids=obligations,
        classified_incompatibility_ids=classified,
        inventory_cid=inventory_map["inventory_cid"],
        preserve_binding=preserve,
        silent_incompatibility=False,
    )


def compile_binding_compatibility_receipt(
    packet: RefactorTransformationPacket | Mapping[str, Any],
    **kwargs: Any,
) -> BindingCompatibilityReceipt:
    """Compile a SPAR-024 receipt over nominated binding adapters."""

    resolved = _coerce_packet(packet)
    preimage_cid = verify_preimages(
        resolved, claimed_preimage_cid=kwargs.get("claimed_preimage_cid") or ""
    )
    plan = compile_binding_compatibility_plan(resolved, **kwargs)
    return BindingCompatibilityReceipt(
        tree_id=resolved.tree_id,
        packet_cid=resolved.packet_cid,
        preimage_cid=preimage_cid,
        adapter_cids=plan.adapter_cids,
        plan_cid=plan.plan_cid,
        write_paths=resolved.effect_scope.write_paths,
        classified_incompatibility_ids=plan.classified_incompatibility_ids,
        inventory_cid=plan.inventory_cid,
        preserve_binding=plan.preserve_binding,
    )


def execute_binding_compatibility_adapters(
    packet: RefactorTransformationPacket | Mapping[str, Any],
    **kwargs: Any,
) -> BindingCompatibilityReceipt:
    """Execute SPAR-024 as a deterministic no-mutation dry-run."""

    if kwargs.pop("mutate", False):
        raise BindingCompatibilityError("executor cannot mutate")
    return compile_binding_compatibility_receipt(packet, **kwargs)


def execute_binding_compatibility_plan(
    packet: RefactorTransformationPacket | Mapping[str, Any],
    **kwargs: Any,
) -> BindingCompatibilityPlan:
    """Execute the unique SPAR-024 plan without mutation."""

    if kwargs.pop("mutate", False):
        raise BindingCompatibilityError("executor cannot mutate")
    return compile_binding_compatibility_plan(packet, **kwargs)


def dry_run_binding_compatibility_adapters(
    packet: RefactorTransformationPacket | Mapping[str, Any],
    **kwargs: Any,
) -> BindingCompatibilityReceipt:
    """Return a deterministic no-mutation dry-run of SPAR-024 adapters."""

    receipt = execute_binding_compatibility_adapters(packet, **kwargs)
    if receipt.mutated:
        raise BindingCompatibilityError("dry-run cannot mutate")
    return receipt


def encode_canonical_adapter(adapter: BindingCompatibilityAdapter) -> dict[str, Any]:
    return adapter.to_dict()


def decode_canonical_adapter(payload: Mapping[str, Any]) -> BindingCompatibilityAdapter:
    return BindingCompatibilityAdapter.from_dict(payload)


def encode_canonical_plan(plan: BindingCompatibilityPlan) -> dict[str, Any]:
    return plan.to_dict()


def decode_canonical_plan(payload: Mapping[str, Any]) -> BindingCompatibilityPlan:
    return BindingCompatibilityPlan.from_dict(payload)


def encode_canonical_receipt(receipt: BindingCompatibilityReceipt) -> dict[str, Any]:
    return receipt.to_dict()


def decode_canonical_receipt(
    payload: Mapping[str, Any],
) -> BindingCompatibilityReceipt:
    return BindingCompatibilityReceipt.from_dict(payload)


def provider_free_exports() -> tuple[str, ...]:
    return tuple(sorted(__all__))


def assert_not_competing_capsule_family() -> None:
    names = set(globals())
    overlap = names & _FORBIDDEN_CAPSULE_TYPE_NAMES
    if overlap:
        raise BindingCompatibilityError(
            f"binding compatibility must not define capsule types: {sorted(overlap)}"
        )


__all__ = [
    "ADAPTER_CAN_AUTHORIZE_COMPLETION",
    "ADAPTER_CAN_AUTHORIZE_TRANSITION",
    "ADAPTER_CAN_CREATE_AUTHORITY",
    "ADAPTER_CONTRACT_VERSION",
    "ADAPTER_IS_NOMINATION_ONLY",
    "ANALYZER_ID",
    "AUTHORITY",
    "AUTHORITY_OWNER",
    "BINDING_COMPATIBILITY_ADAPTER_INTERFACE",
    "BINDING_COMPATIBILITY_PLAN_INTERFACE",
    "BINDING_COMPATIBILITY_RECEIPT_INTERFACE",
    "BINDING_IDENTITY_KINDS",
    "DECLARED_BINDING_COMPATIBILITY_KINDS",
    "DECLARED_INCOMPATIBILITY_CLASSIFICATIONS",
    "DRY_RUN_IS_DETERMINISTIC",
    "DRY_RUN_MUTATES",
    "DUCKLAKE_IS_AUTHORITY",
    "EXECUTOR_IS_NOMINATION_ONLY",
    "GOAL_ID",
    "HANDLED_ADAPTER_KINDS",
    "HANDLED_MIGRATION_KINDS",
    "HANDLED_REWRITE_KINDS",
    "IDENTITY_EXCLUDED_FIELDS",
    "INTENTIONAL_INCOMPATIBILITY_MUST_BE_CLASSIFIED",
    "MARKDOWN_IS_NOT_COMPLETION",
    "MODEL_OUTPUT_IS_PROPOSAL_ONLY",
    "PLAN_IS_NOMINATION_ONLY",
    "PROGRAM",
    "PROJECTION_CLUSTERING_IS_AUTHORITY",
    "RAW_SOURCE_REQUIRED",
    "RUNTIME_OBSERVATION_IS_NOT_STATIC_FACT",
    "SILENT_INCOMPATIBILITY_REJECTED",
    "SOFT_SIGNALS_CANNOT_OVERRIDE_HARD_CONSTRAINTS",
    "TASK_ID",
    "TEST_PASS_IS_NOT_COMPLETION",
    "VECTOR_SIMILARITY_IS_AUTHORITY",
    "WORKER_SELF_APPROVAL",
    "BindingCompatibilityAdapter",
    "BindingCompatibilityError",
    "BindingCompatibilityKind",
    "BindingCompatibilityPlan",
    "BindingCompatibilityReceipt",
    "IncompatibilityClassification",
    "assert_not_competing_capsule_family",
    "binding_compatibility_cid_profile",
    "classify_intentional_incompatibility",
    "compile_binding_compatibility_adapters",
    "compile_binding_compatibility_plan",
    "compile_binding_compatibility_receipt",
    "decode_canonical_adapter",
    "decode_canonical_plan",
    "decode_canonical_receipt",
    "dry_run_binding_compatibility_adapters",
    "encode_canonical_adapter",
    "encode_canonical_plan",
    "encode_canonical_receipt",
    "execute_binding_compatibility_adapters",
    "execute_binding_compatibility_plan",
    "provider_free_exports",
    "verify_preimages",
]


assert TASK_ID == "SPAR-024"
assert BINDING_COMPATIBILITY_ADAPTER_INTERFACE == "BindingCompatibilityAdapter@1"
assert BINDING_COMPATIBILITY_PLAN_INTERFACE == "BindingCompatibilityPlan@1"
assert BINDING_COMPATIBILITY_RECEIPT_INTERFACE == "BindingCompatibilityReceipt@1"
assert EXECUTOR_IS_NOMINATION_ONLY is True
assert PLAN_IS_NOMINATION_ONLY is True
assert ADAPTER_IS_NOMINATION_ONLY is True
assert ADAPTER_CAN_AUTHORIZE_COMPLETION is False
assert DRY_RUN_MUTATES is False
assert SILENT_INCOMPATIBILITY_REJECTED is True
assert ANALYZER_ID != SPAR019_ANALYZER_ID
