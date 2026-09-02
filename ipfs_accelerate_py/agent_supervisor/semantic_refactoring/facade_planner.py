"""SPAR-018 compatibility façade and consumer migration plans.

This module extends current supervisor partition orchestration with
``CompatibilityFacadePlan@1``.  It consumes SPAR-011 public-compatibility
inventory mappings and SPAR-014 ranked partition candidates, then nominates
explicit façade, re-export, wrapper, deprecation, CLI/plugin, registry,
serialization, introspection, traceback, and patch-target migration plans
for every consumer.

The plan is nomination-only.  It cannot authorize a transition, completion,
façade retirement, or competing authority.  Vector, model, and heuristic
evidence cannot admit a façade.  SPAR-011 payloads are ingested as mappings
only; this module does not replace datasets semantic authority.
Observational metadata is excluded from identity.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Final, Mapping, Sequence
import unicodedata

from ipfs_accelerate_py.utils.cid_utils import cid_for_dag_json, validate_cid

from .partition_generators import (
    IDENTITY_EXCLUDED_FIELDS as SPAR013_IDENTITY_EXCLUDED_FIELDS,
    PartitionGenerationReceipt,
    ProgramPartitionCandidate,
)
from .partition_policy import (
    ANALYZER_ID as SPAR014_ANALYZER_ID,
    PartitionComparisonReceipt,
    compare_partition_candidates,
)


TASK_ID: Final[str] = "SPAR-018"
GOAL_ID: Final[str] = "SPAR-G033"
PROGRAM: Final[str] = "semantic-preserving-autonomous-remodularization-v1"
AUTHORITY: Final[str] = "partition orchestration"
AUTHORITY_OWNER: Final[str] = "ipfs_accelerate_py"
ANALYZER_ID: Final[str] = (
    "ipfs_accelerate_py.agent_supervisor.semantic_refactoring.facade_planner@1"
)

COMPATIBILITY_FACADE_PLAN_INTERFACE: Final[str] = "CompatibilityFacadePlan@1"
CONSUMER_MIGRATION_PLAN_INTERFACE: Final[str] = "ConsumerMigrationPlan@1"
SUBJECT_FACADE_PLAN_INTERFACE: Final[str] = "SubjectFacadePlan@1"
FACADE_PLANNING_RECEIPT_INTERFACE: Final[str] = "FacadePlanningReceipt@1"

COMPATIBILITY_FACADE_PLAN_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/compatibility-facade-plan@1"
)
CONSUMER_MIGRATION_PLAN_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/consumer-migration-plan@1"
)
SUBJECT_FACADE_PLAN_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/subject-facade-plan@1"
)
FACADE_PLANNING_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/facade-planning-receipt@1"
)

FACADE_PLAN_CONTRACT_VERSION: Final[str] = "1"

FACADE_CAN_AUTHORIZE_TRANSITION: Final[bool] = False
FACADE_CAN_AUTHORIZE_COMPLETION: Final[bool] = False
FACADE_CAN_CREATE_AUTHORITY: Final[bool] = False
FACADE_CAN_RETIRE_FACADE: Final[bool] = False
VECTOR_SIMILARITY_IS_AUTHORITY: Final[bool] = False
PROJECTION_CLUSTERING_IS_AUTHORITY: Final[bool] = False
MODEL_OUTPUT_IS_PROPOSAL_ONLY: Final[bool] = True
TEST_PASS_IS_NOT_COMPLETION: Final[bool] = True
MARKDOWN_IS_NOT_COMPLETION: Final[bool] = True
WORKER_SELF_APPROVAL: Final[bool] = False
DUCKLAKE_IS_AUTHORITY: Final[bool] = False
SOFT_SIGNALS_CANNOT_OVERRIDE_HARD_CONSTRAINTS: Final[bool] = True
PLAN_IS_NOMINATION_ONLY: Final[bool] = True
RAW_SOURCE_REQUIRED: Final[bool] = True

MAX_TEXT_CHARS: Final[int] = 16_384
MAX_MEMBERS: Final[int] = 16_384
MAX_MODULES: Final[int] = 16_384
MAX_PLANS: Final[int] = 16_384
MAX_EVIDENCE_CIDS: Final[int] = 1_024

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

_OBLIGATION_KEYS: Final[frozenset[str]] = frozenset(
    {
        "obligation_id",
        "consumer_id",
        "subject_id",
        "subject_module",
        "kind",
        "disposition",
        "required",
        "family",
    }
)

_CONSUMER_KEYS: Final[frozenset[str]] = frozenset(
    {
        "consumer_id",
        "module_name",
        "role",
        "required",
        "evidence_class",
    }
)


class FacadePlannerError(ValueError):
    """Fail-closed violation of a SPAR-018 façade-planning contract."""


class CompatibilityKind(str, Enum):
    """Closed SPAR-004/SPAR-011 public compatibility kinds."""

    IMPORT_PATH = "import_path"
    IMPORT_EFFECT = "import_effect"
    IMPORT_EAGERNESS = "import_eagerness"
    IMPORT_ORDER = "import_order"
    STAR_EXPORT = "star_export"
    MODULE_ATTRIBUTE = "module_attribute"
    SIGNATURE = "signature"
    ANNOTATION = "annotation"
    DEFAULT = "default"
    DECORATOR = "decorator"
    EXCEPTION = "exception"
    CLI = "cli"
    PLUGIN = "plugin"
    REGISTRY = "registry"
    MODULE_NAME = "module_name"
    QUALNAME = "qualname"
    PICKLE = "pickle"
    SERIALIZATION = "serialization"
    INTROSPECTION = "introspection"
    TRACEBACK = "traceback"
    DOCUMENTATION = "documentation"
    CONFIGURATION = "configuration"
    PATCH_TARGET = "patch_target"
    RESOURCE = "resource"
    RESOURCE_LIFETIME = "resource_lifetime"


class ConsumerSurface(str, Enum):
    """Closed SPAR-011 consumer surfaces."""

    IMPORT = "import"
    API = "api"
    BINDING = "binding"
    SERIALIZATION = "serialization"
    INTROSPECTION = "introspection"
    CLI = "cli"
    PLUGIN = "plugin"
    REGISTRATION = "registration"
    DOCUMENTATION = "documentation"
    PATCH = "patch"


class CompatibilityDisposition(str, Enum):
    PRESERVE = "preserve"
    MIGRATE = "migrate"
    FACADE = "facade"
    EXPLICIT_INCOMPATIBILITY = "explicit_incompatibility"
    UNSUPPORTED = "unsupported"
    UNDISPOSITIONED = "undispositioned"


class MigrationKind(str, Enum):
    """Closed SPAR-018 migration plan kinds."""

    FACADE = "facade"
    REEXPORT = "reexport"
    WRAPPER = "wrapper"
    DEPRECATION = "deprecation"
    CLI = "cli"
    PLUGIN = "plugin"
    REGISTRY = "registry"
    SERIALIZATION = "serialization"
    INTROSPECTION = "introspection"
    TRACEBACK = "traceback"
    PATCH_TARGET = "patch_target"


class ConsumerRole(str, Enum):
    MODULE = "module"
    TEST = "test"
    PROOF = "proof"
    CLI = "cli"
    PLUGIN = "plugin"
    DOCUMENTATION = "documentation"
    PATCH = "patch"
    EXTERNAL = "external"


DECLARED_CONSUMER_SURFACES: Final[frozenset[str]] = frozenset(
    surface.value for surface in ConsumerSurface
)
DECLARED_MIGRATION_KINDS: Final[frozenset[str]] = frozenset(
    kind.value for kind in MigrationKind
)
DECLARED_COMPATIBILITY_KINDS: Final[frozenset[str]] = frozenset(
    kind.value for kind in CompatibilityKind
)
DISPOSITIONED: Final[frozenset[str]] = frozenset(
    {
        CompatibilityDisposition.PRESERVE.value,
        CompatibilityDisposition.MIGRATE.value,
        CompatibilityDisposition.FACADE.value,
        CompatibilityDisposition.EXPLICIT_INCOMPATIBILITY.value,
        CompatibilityDisposition.UNSUPPORTED.value,
    }
)

KIND_SURFACE: Final[Mapping[CompatibilityKind, ConsumerSurface]] = {
    CompatibilityKind.IMPORT_PATH: ConsumerSurface.IMPORT,
    CompatibilityKind.IMPORT_EFFECT: ConsumerSurface.IMPORT,
    CompatibilityKind.IMPORT_EAGERNESS: ConsumerSurface.IMPORT,
    CompatibilityKind.IMPORT_ORDER: ConsumerSurface.IMPORT,
    CompatibilityKind.STAR_EXPORT: ConsumerSurface.IMPORT,
    CompatibilityKind.MODULE_ATTRIBUTE: ConsumerSurface.API,
    CompatibilityKind.SIGNATURE: ConsumerSurface.API,
    CompatibilityKind.ANNOTATION: ConsumerSurface.API,
    CompatibilityKind.DEFAULT: ConsumerSurface.API,
    CompatibilityKind.EXCEPTION: ConsumerSurface.API,
    CompatibilityKind.CONFIGURATION: ConsumerSurface.API,
    CompatibilityKind.MODULE_NAME: ConsumerSurface.BINDING,
    CompatibilityKind.QUALNAME: ConsumerSurface.BINDING,
    CompatibilityKind.PICKLE: ConsumerSurface.SERIALIZATION,
    CompatibilityKind.SERIALIZATION: ConsumerSurface.SERIALIZATION,
    CompatibilityKind.INTROSPECTION: ConsumerSurface.INTROSPECTION,
    CompatibilityKind.TRACEBACK: ConsumerSurface.INTROSPECTION,
    CompatibilityKind.CLI: ConsumerSurface.CLI,
    CompatibilityKind.PLUGIN: ConsumerSurface.PLUGIN,
    CompatibilityKind.REGISTRY: ConsumerSurface.REGISTRATION,
    CompatibilityKind.DECORATOR: ConsumerSurface.REGISTRATION,
    CompatibilityKind.RESOURCE: ConsumerSurface.REGISTRATION,
    CompatibilityKind.RESOURCE_LIFETIME: ConsumerSurface.REGISTRATION,
    CompatibilityKind.DOCUMENTATION: ConsumerSurface.DOCUMENTATION,
    CompatibilityKind.PATCH_TARGET: ConsumerSurface.PATCH,
}

KIND_FAMILY: Final[Mapping[CompatibilityKind, str]] = {
    CompatibilityKind.IMPORT_PATH: "import",
    CompatibilityKind.IMPORT_EFFECT: "import",
    CompatibilityKind.IMPORT_EAGERNESS: "import",
    CompatibilityKind.IMPORT_ORDER: "import",
    CompatibilityKind.STAR_EXPORT: "import",
    CompatibilityKind.REGISTRY: "registry",
    CompatibilityKind.DECORATOR: "decorator",
    CompatibilityKind.RESOURCE: "resource",
    CompatibilityKind.RESOURCE_LIFETIME: "resource",
    CompatibilityKind.SERIALIZATION: "serialization",
    CompatibilityKind.PICKLE: "serialization",
    CompatibilityKind.INTROSPECTION: "introspection",
    CompatibilityKind.MODULE_NAME: "introspection",
    CompatibilityKind.QUALNAME: "introspection",
    CompatibilityKind.SIGNATURE: "introspection",
    CompatibilityKind.ANNOTATION: "introspection",
    CompatibilityKind.DEFAULT: "introspection",
    CompatibilityKind.TRACEBACK: "introspection",
    CompatibilityKind.DOCUMENTATION: "introspection",
    CompatibilityKind.MODULE_ATTRIBUTE: "introspection",
    CompatibilityKind.EXCEPTION: "introspection",
    CompatibilityKind.CONFIGURATION: "introspection",
    CompatibilityKind.CLI: "cli_plugin",
    CompatibilityKind.PLUGIN: "cli_plugin",
    CompatibilityKind.PATCH_TARGET: "patch_target",
}

KIND_MIGRATION: Final[Mapping[CompatibilityKind, MigrationKind]] = {
    CompatibilityKind.IMPORT_PATH: MigrationKind.REEXPORT,
    CompatibilityKind.IMPORT_EFFECT: MigrationKind.REEXPORT,
    CompatibilityKind.IMPORT_EAGERNESS: MigrationKind.REEXPORT,
    CompatibilityKind.IMPORT_ORDER: MigrationKind.REEXPORT,
    CompatibilityKind.STAR_EXPORT: MigrationKind.REEXPORT,
    CompatibilityKind.MODULE_ATTRIBUTE: MigrationKind.WRAPPER,
    CompatibilityKind.SIGNATURE: MigrationKind.WRAPPER,
    CompatibilityKind.ANNOTATION: MigrationKind.WRAPPER,
    CompatibilityKind.DEFAULT: MigrationKind.WRAPPER,
    CompatibilityKind.EXCEPTION: MigrationKind.WRAPPER,
    CompatibilityKind.CONFIGURATION: MigrationKind.WRAPPER,
    CompatibilityKind.DECORATOR: MigrationKind.REGISTRY,
    CompatibilityKind.CLI: MigrationKind.CLI,
    CompatibilityKind.PLUGIN: MigrationKind.PLUGIN,
    CompatibilityKind.REGISTRY: MigrationKind.REGISTRY,
    CompatibilityKind.MODULE_NAME: MigrationKind.WRAPPER,
    CompatibilityKind.QUALNAME: MigrationKind.WRAPPER,
    CompatibilityKind.PICKLE: MigrationKind.SERIALIZATION,
    CompatibilityKind.SERIALIZATION: MigrationKind.SERIALIZATION,
    CompatibilityKind.INTROSPECTION: MigrationKind.INTROSPECTION,
    CompatibilityKind.TRACEBACK: MigrationKind.TRACEBACK,
    CompatibilityKind.DOCUMENTATION: MigrationKind.DEPRECATION,
    CompatibilityKind.PATCH_TARGET: MigrationKind.PATCH_TARGET,
    CompatibilityKind.RESOURCE: MigrationKind.REGISTRY,
    CompatibilityKind.RESOURCE_LIFETIME: MigrationKind.REGISTRY,
}

_PRESERVE_REEXPORT_KINDS: Final[frozenset[CompatibilityKind]] = frozenset(
    {
        CompatibilityKind.IMPORT_PATH,
        CompatibilityKind.IMPORT_EFFECT,
        CompatibilityKind.IMPORT_EAGERNESS,
        CompatibilityKind.IMPORT_ORDER,
        CompatibilityKind.STAR_EXPORT,
        CompatibilityKind.MODULE_ATTRIBUTE,
        CompatibilityKind.DOCUMENTATION,
    }
)


def _text(value: Any, name: str, *, empty: bool = False) -> str:
    if type(value) is not str:
        raise FacadePlannerError(f"{name} must be a string")
    if value != value.strip() or unicodedata.normalize("NFC", value) != value:
        raise FacadePlannerError(f"{name} must be trimmed NFC text")
    if not empty and not value:
        raise FacadePlannerError(f"{name} must be a nonempty string")
    if any(not char.isprintable() for char in value):
        raise FacadePlannerError(f"{name} contains invalid text")
    if len(value) > MAX_TEXT_CHARS:
        raise FacadePlannerError(f"{name} exceeds text bound")
    return value


def _cid(value: Any, name: str) -> str:
    text = _text(value, name)
    try:
        return validate_cid(text)
    except Exception as exc:
        raise FacadePlannerError(f"{name} must be a valid CID") from exc


def _optional_cid(value: Any, name: str) -> str:
    if value in (None, ""):
        return ""
    return _cid(value, name)


def _bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise FacadePlannerError(f"{name} must be a boolean")
    return value


def _tree_id(value: Any) -> str:
    text = _text(value, "tree_id")
    if len(text) not in {40, 64} or any(
        char not in "0123456789abcdef" for char in text
    ):
        raise FacadePlannerError("tree_id must be a lowercase hex Git tree identity")
    return text


def _closed(data: Mapping[str, Any], fields: frozenset[str], name: str) -> dict[str, Any]:
    if not isinstance(data, Mapping) or isinstance(data, (str, bytes, bytearray)):
        raise FacadePlannerError(f"{name} must be an object")
    extra = set(data) - fields
    missing = fields - set(data)
    if extra & IDENTITY_EXCLUDED_FIELDS:
        raise FacadePlannerError(
            f"{name} identity excludes observational fields: "
            f"{sorted(extra & IDENTITY_EXCLUDED_FIELDS)}"
        )
    if extra:
        raise FacadePlannerError(f"unknown {name} field: {sorted(extra)}")
    if missing:
        raise FacadePlannerError(f"missing {name} field: {sorted(missing)}")
    return dict(data)


def _reject_excluded(payload: Mapping[str, Any], name: str) -> None:
    present = IDENTITY_EXCLUDED_FIELDS & set(payload)
    if present:
        raise FacadePlannerError(
            f"{name} identity excludes observational fields: {sorted(present)}"
        )


def _require_dag_json(value: Any, name: str) -> None:
    try:
        cid_for_dag_json(value)
    except Exception as exc:
        raise FacadePlannerError(f"{name} must be strict DAG-JSON") from exc


def _verify_cid(claimed: Any, computed: str, name: str) -> None:
    cid = _cid(claimed, name)
    if cid != computed:
        raise FacadePlannerError(f"{name} does not verify")


def _unique_sorted_text(values: Any, name: str, *, limit: int) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        raise FacadePlannerError(f"{name} must be a list")
    ordered = tuple(sorted(_text(item, name) for item in values))
    if len(ordered) > limit:
        raise FacadePlannerError(f"{name} exceeds maximum length")
    if len(ordered) != len(set(ordered)):
        raise FacadePlannerError(f"{name} must not contain duplicates")
    return ordered


def _enum(value: Any, enum_type: type[Enum], name: str) -> str:
    if isinstance(value, enum_type):
        return value.value
    text = _text(value, name)
    try:
        return enum_type(text).value
    except ValueError as exc:
        raise FacadePlannerError(f"unknown {name}: {text}") from exc


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
    raise FacadePlannerError(f"unsupported projected type {type(value).__name__}")


def _mapping(value: Any, name: str) -> dict[str, Any]:
    projected = _project(value)
    if not isinstance(projected, dict):
        raise FacadePlannerError(f"{name} must be an object")
    _reject_excluded(projected, name)
    return projected


def facade_planner_cid_profile() -> dict[str, str]:
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
            raise FacadePlannerError(f"{name} cannot claim {flag}")


def surface_for_kind(kind: CompatibilityKind | str) -> ConsumerSurface:
    resolved = CompatibilityKind(kind) if not isinstance(kind, CompatibilityKind) else kind
    try:
        return KIND_SURFACE[resolved]
    except KeyError as exc:
        raise FacadePlannerError(f"kind has no consumer surface: {kind!r}") from exc


def family_for_kind(kind: CompatibilityKind | str) -> str:
    resolved = CompatibilityKind(kind) if not isinstance(kind, CompatibilityKind) else kind
    try:
        return KIND_FAMILY[resolved]
    except KeyError as exc:
        raise FacadePlannerError(f"kind has no family: {kind!r}") from exc


def migration_kind_for(
    kind: CompatibilityKind | str,
    disposition: CompatibilityDisposition | str,
) -> MigrationKind:
    """Map one SPAR-011 obligation onto a SPAR-018 migration kind."""

    resolved = CompatibilityKind(kind) if not isinstance(kind, CompatibilityKind) else kind
    disp = (
        disposition
        if isinstance(disposition, CompatibilityDisposition)
        else CompatibilityDisposition(disposition)
    )
    if disp is CompatibilityDisposition.FACADE:
        return MigrationKind.FACADE
    if disp is CompatibilityDisposition.EXPLICIT_INCOMPATIBILITY:
        return MigrationKind.DEPRECATION
    if disp is CompatibilityDisposition.UNSUPPORTED:
        return MigrationKind.DEPRECATION
    if disp is CompatibilityDisposition.UNDISPOSITIONED:
        return MigrationKind.FACADE
    if disp is CompatibilityDisposition.PRESERVE and resolved in _PRESERVE_REEXPORT_KINDS:
        return MigrationKind.REEXPORT
    try:
        return KIND_MIGRATION[resolved]
    except KeyError as exc:
        raise FacadePlannerError(f"kind has no migration: {kind!r}") from exc


def _pick(payload: Mapping[str, Any], keys: frozenset[str]) -> dict[str, Any]:
    return {key: payload[key] for key in keys if key in payload}


@dataclass(frozen=True, slots=True)
class CompatibilityObligationView:
    """SPAR-011 obligation projection ingested as a mapping."""

    obligation_id: str
    consumer_id: str
    subject_id: str
    subject_module: str
    kind: CompatibilityKind | str
    disposition: CompatibilityDisposition | str
    required: bool = True
    family: str = ""

    def __post_init__(self) -> None:
        kind = _enum(self.kind, CompatibilityKind, "kind")
        disposition = _enum(self.disposition, CompatibilityDisposition, "disposition")
        expected_family = family_for_kind(kind)
        family = self.family
        if family in (None, ""):
            family = expected_family
        else:
            family = _text(family, "family")
        if family != expected_family:
            raise FacadePlannerError(f"family {family} does not match kind {kind}")
        object.__setattr__(self, "obligation_id", _text(self.obligation_id, "obligation_id"))
        object.__setattr__(self, "consumer_id", _text(self.consumer_id, "consumer_id"))
        object.__setattr__(self, "subject_id", _text(self.subject_id, "subject_id"))
        object.__setattr__(
            self, "subject_module", _text(self.subject_module, "subject_module")
        )
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "disposition", disposition)
        object.__setattr__(self, "required", _bool(self.required, "required"))
        object.__setattr__(self, "family", family)

    @property
    def surface(self) -> str:
        return surface_for_kind(self.kind).value

    @property
    def migration_kind(self) -> str:
        return migration_kind_for(self.kind, self.disposition).value

    @property
    def dispositioned(self) -> bool:
        return self.disposition in DISPOSITIONED


@dataclass(frozen=True, slots=True)
class CompatibilityConsumerView:
    """SPAR-011 consumer projection ingested as a mapping."""

    consumer_id: str
    module_name: str
    role: ConsumerRole | str = ConsumerRole.MODULE
    required: bool = True
    evidence_class: str = "exact_static_fact"

    def __post_init__(self) -> None:
        object.__setattr__(self, "consumer_id", _text(self.consumer_id, "consumer_id"))
        object.__setattr__(self, "module_name", _text(self.module_name, "module_name"))
        object.__setattr__(self, "role", _enum(self.role, ConsumerRole, "role"))
        object.__setattr__(self, "required", _bool(self.required, "required"))
        object.__setattr__(
            self, "evidence_class", _text(self.evidence_class, "evidence_class")
        )
        if self.evidence_class in _NON_ADMITTING_EVIDENCE:
            raise FacadePlannerError("vector or model evidence cannot admit a consumer")


def _coerce_obligation(value: Any) -> CompatibilityObligationView:
    if isinstance(value, CompatibilityObligationView):
        return value
    if not isinstance(value, Mapping):
        to_dict = getattr(value, "to_dict", None)
        if callable(to_dict):
            value = to_dict()
        else:
            raise FacadePlannerError("obligation must be a mapping")
    payload = _pick(_mapping(value, "obligation"), _OBLIGATION_KEYS)
    if payload.get("consumer_id") in (None, ""):
        raise FacadePlannerError("obligation requires a consumer_id")
    return CompatibilityObligationView(**payload)


def _coerce_consumer(value: Any) -> CompatibilityConsumerView:
    if isinstance(value, CompatibilityConsumerView):
        return value
    if not isinstance(value, Mapping):
        to_dict = getattr(value, "to_dict", None)
        if callable(to_dict):
            value = to_dict()
        else:
            raise FacadePlannerError("consumer must be a mapping")
    payload = _pick(_mapping(value, "consumer"), _CONSUMER_KEYS)
    if "module_name" not in payload:
        payload["module_name"] = payload.get("consumer_id") or ""
    return CompatibilityConsumerView(**payload)


def _role_for_obligation(obligation: CompatibilityObligationView) -> str:
    consumer_id = obligation.consumer_id
    surface = obligation.surface
    if consumer_id.startswith("test") or ".tests." in f".{consumer_id}.":
        return ConsumerRole.TEST.value
    if consumer_id.startswith("proof") or ".proofs." in f".{consumer_id}.":
        return ConsumerRole.PROOF.value
    if surface == ConsumerSurface.CLI.value:
        return ConsumerRole.CLI.value
    if surface == ConsumerSurface.PLUGIN.value:
        return ConsumerRole.PLUGIN.value
    if surface == ConsumerSurface.DOCUMENTATION.value:
        return ConsumerRole.DOCUMENTATION.value
    if surface == ConsumerSurface.PATCH.value:
        return ConsumerRole.PATCH.value
    return ConsumerRole.MODULE.value


@dataclass(frozen=True, slots=True)
class CompatibilityInventoryView:
    """Canonical SPAR-011 inventory projection used by SPAR-018."""

    tree_id: str
    obligations: tuple[CompatibilityObligationView, ...]
    consumers: tuple[CompatibilityConsumerView, ...]
    inventory_cid: str = ""

    def __post_init__(self) -> None:
        tree_id = _tree_id(self.tree_id)
        obligations = tuple(
            sorted(self.obligations, key=lambda item: item.obligation_id)
        )
        obligation_ids = [item.obligation_id for item in obligations]
        if not obligations:
            raise FacadePlannerError("inventory requires obligations")
        if len(obligation_ids) != len(set(obligation_ids)):
            raise FacadePlannerError("obligations must have unique obligation_id")
        consumers = tuple(sorted(self.consumers, key=lambda item: item.consumer_id))
        consumer_ids = [item.consumer_id for item in consumers]
        if len(consumer_ids) != len(set(consumer_ids)):
            raise FacadePlannerError("consumers must have unique consumer_id")
        referenced = {item.consumer_id for item in obligations}
        declared = set(consumer_ids)
        missing = referenced - declared
        extra = declared - referenced
        if missing:
            raise FacadePlannerError(f"inventory is missing consumers: {sorted(missing)}")
        if extra:
            raise FacadePlannerError(f"inventory has orphan consumers: {sorted(extra)}")
        coverage: dict[tuple[str, str, str], str] = {}
        for item in obligations:
            key = (item.subject_id, item.kind, item.consumer_id)
            previous = coverage.get(key)
            if previous is None:
                coverage[key] = item.disposition
                continue
            if previous != item.disposition:
                raise FacadePlannerError(
                    "conflicting dispositions for the same subject, kind, and consumer"
                )
            raise FacadePlannerError("duplicate subject/kind/consumer coverage")
        for consumer in consumers:
            matching = [
                item for item in obligations if item.consumer_id == consumer.consumer_id
            ]
            if consumer.required and not any(item.required for item in matching):
                raise FacadePlannerError(
                    "required consumers must have a required obligation"
                )
        object.__setattr__(self, "tree_id", tree_id)
        object.__setattr__(self, "obligations", obligations)
        object.__setattr__(self, "consumers", consumers)
        object.__setattr__(
            self, "inventory_cid", _optional_cid(self.inventory_cid, "inventory_cid")
        )


def _synthesize_consumers(
    obligations: Sequence[CompatibilityObligationView],
) -> tuple[CompatibilityConsumerView, ...]:
    grouped: dict[str, list[CompatibilityObligationView]] = {}
    for item in obligations:
        grouped.setdefault(item.consumer_id, []).append(item)
    consumers: list[CompatibilityConsumerView] = []
    for consumer_id, members in grouped.items():
        required = any(item.required for item in members)
        consumers.append(
            CompatibilityConsumerView(
                consumer_id=consumer_id,
                module_name=consumer_id,
                role=_role_for_obligation(members[0]),
                required=required,
            )
        )
    return tuple(consumers)


def _coerce_inventory(value: Any) -> CompatibilityInventoryView:
    if isinstance(value, CompatibilityInventoryView):
        return value
    if isinstance(value, (list, tuple)):
        obligations = tuple(_coerce_obligation(item) for item in value)
        if not obligations:
            raise FacadePlannerError("inventory requires obligations")
        raise FacadePlannerError("inventory mapping must declare tree_id")
    if not isinstance(value, Mapping):
        to_dict = getattr(value, "to_dict", None)
        if callable(to_dict):
            value = to_dict()
        else:
            raise FacadePlannerError("inventory must be a SPAR-011 mapping")
    payload = _mapping(value, "inventory")
    raw_obligations = payload.get("obligations")
    if not isinstance(raw_obligations, (list, tuple)) or not raw_obligations:
        raise FacadePlannerError("inventory requires obligations")
    obligations = tuple(_coerce_obligation(item) for item in raw_obligations)
    raw_consumers = payload.get("consumers")
    if raw_consumers in (None, ()):
        consumers = _synthesize_consumers(obligations)
    elif isinstance(raw_consumers, (list, tuple)):
        consumers = tuple(_coerce_consumer(item) for item in raw_consumers)
    else:
        raise FacadePlannerError("consumers must be a list")
    tree_id = payload.get("tree_id")
    if not tree_id:
        raise FacadePlannerError("inventory mapping must declare tree_id")
    return CompatibilityInventoryView(
        tree_id=tree_id,
        obligations=obligations,
        consumers=consumers,
        inventory_cid=payload.get("inventory_cid") or "",
    )


@dataclass(frozen=True, slots=True)
class ConsumerMigrationPlan:
    """One nominated consumer migration. Nomination only."""

    consumer_id: str
    obligation_id: str
    subject_id: str
    subject_module: str
    kind: CompatibilityKind | str
    disposition: CompatibilityDisposition | str
    migration_kind: MigrationKind | str = ""
    required: bool = True
    target_module_id: str = ""
    role: ConsumerRole | str = ConsumerRole.MODULE

    interface: ClassVar[str] = CONSUMER_MIGRATION_PLAN_INTERFACE
    schema: ClassVar[str] = CONSUMER_MIGRATION_PLAN_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "consumer_id",
            "obligation_id",
            "subject_id",
            "subject_module",
            "kind",
            "surface",
            "disposition",
            "migration_kind",
            "required",
            "target_module_id",
            "role",
            "can_authorize_transition",
            "can_authorize_completion",
            "can_create_authority",
            "projection_is_authority",
            "plan_cid",
        }
    )

    def __post_init__(self) -> None:
        kind = _enum(self.kind, CompatibilityKind, "kind")
        disposition = _enum(self.disposition, CompatibilityDisposition, "disposition")
        expected = migration_kind_for(kind, disposition).value
        migration = self.migration_kind
        if migration in (None, ""):
            migration = expected
        else:
            migration = _enum(migration, MigrationKind, "migration_kind")
        if migration != expected:
            raise FacadePlannerError("migration_kind must remain canonical")
        object.__setattr__(self, "consumer_id", _text(self.consumer_id, "consumer_id"))
        object.__setattr__(
            self, "obligation_id", _text(self.obligation_id, "obligation_id")
        )
        object.__setattr__(self, "subject_id", _text(self.subject_id, "subject_id"))
        object.__setattr__(
            self, "subject_module", _text(self.subject_module, "subject_module")
        )
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "disposition", disposition)
        object.__setattr__(self, "migration_kind", migration)
        object.__setattr__(self, "required", _bool(self.required, "required"))
        object.__setattr__(
            self,
            "target_module_id",
            _text(self.target_module_id, "target_module_id", empty=True),
        )
        object.__setattr__(self, "role", _enum(self.role, ConsumerRole, "role"))

    @property
    def surface(self) -> str:
        return surface_for_kind(self.kind).value

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

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": CONSUMER_MIGRATION_PLAN_SCHEMA,
            "interface": CONSUMER_MIGRATION_PLAN_INTERFACE,
            "consumer_id": self.consumer_id,
            "obligation_id": self.obligation_id,
            "subject_id": self.subject_id,
            "subject_module": self.subject_module,
            "kind": self.kind,
            "surface": self.surface,
            "disposition": self.disposition,
            "migration_kind": self.migration_kind,
            "required": self.required,
            "target_module_id": self.target_module_id,
            "role": self.role,
            "can_authorize_transition": False,
            "can_authorize_completion": False,
            "can_create_authority": False,
            "projection_is_authority": False,
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
    def from_dict(cls, data: Mapping[str, Any]) -> "ConsumerMigrationPlan":
        _reject_excluded(data, cls.__name__)
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("plan_cid")
        if payload.pop("schema") != CONSUMER_MIGRATION_PLAN_SCHEMA:
            raise FacadePlannerError("unsupported ConsumerMigrationPlan schema")
        if payload.pop("interface") != CONSUMER_MIGRATION_PLAN_INTERFACE:
            raise FacadePlannerError("unsupported ConsumerMigrationPlan interface")
        _pop_authority_flags(payload, "ConsumerMigrationPlan")
        payload.pop("surface")
        result = cls(**payload)
        _verify_cid(claimed, result.plan_cid, "ConsumerMigrationPlan plan_cid")
        return result


def _coerce_migration_plan(value: Any) -> ConsumerMigrationPlan:
    if isinstance(value, ConsumerMigrationPlan):
        return value
    if isinstance(value, Mapping):
        if "plan_cid" in value:
            return ConsumerMigrationPlan.from_dict(value)
        payload = {
            key: item
            for key, item in _project(value).items()
            if key
            not in {
                "schema",
                "interface",
                "plan_cid",
                "surface",
                *_AUTHORITY_FLAG_NAMES,
            }
        }
        return ConsumerMigrationPlan(**payload)
    raise FacadePlannerError("migration plan must be a ConsumerMigrationPlan")


@dataclass(frozen=True, slots=True)
class SubjectFacadePlan:
    """Façade nomination for one original subject module."""

    subject_id: str
    subject_module: str
    facade_required: bool
    consumer_ids: Sequence[str] = ()
    undispositioned_consumer_ids: Sequence[str] = ()
    migration_kinds: Sequence[str] = ()
    target_module_id: str = ""

    interface: ClassVar[str] = SUBJECT_FACADE_PLAN_INTERFACE
    schema: ClassVar[str] = SUBJECT_FACADE_PLAN_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "subject_id",
            "subject_module",
            "facade_required",
            "consumer_ids",
            "undispositioned_consumer_ids",
            "migration_kinds",
            "target_module_id",
            "can_retire_facade",
            "can_authorize_transition",
            "can_authorize_completion",
            "can_create_authority",
            "projection_is_authority",
            "facade_cid",
        }
    )

    def __post_init__(self) -> None:
        consumers = _unique_sorted_text(
            list(self.consumer_ids), "consumer_ids", limit=MAX_MEMBERS
        )
        undispositioned = _unique_sorted_text(
            list(self.undispositioned_consumer_ids),
            "undispositioned_consumer_ids",
            limit=MAX_MEMBERS,
        )
        unknown = set(undispositioned) - set(consumers)
        if unknown:
            raise FacadePlannerError(
                "undispositioned consumers must be listed as consumers"
            )
        kinds = _unique_sorted_text(
            list(self.migration_kinds), "migration_kinds", limit=MAX_MEMBERS
        )
        unknown_kinds = [item for item in kinds if item not in DECLARED_MIGRATION_KINDS]
        if unknown_kinds:
            raise FacadePlannerError(f"unknown migration_kind: {unknown_kinds}")
        facade_required = _bool(self.facade_required, "facade_required")
        expected = bool(undispositioned) or MigrationKind.FACADE.value in kinds
        if facade_required is not expected:
            raise FacadePlannerError(
                "facade_required must match undispositioned consumers or façade plans"
            )
        object.__setattr__(self, "subject_id", _text(self.subject_id, "subject_id"))
        object.__setattr__(
            self, "subject_module", _text(self.subject_module, "subject_module")
        )
        object.__setattr__(self, "facade_required", facade_required)
        object.__setattr__(self, "consumer_ids", consumers)
        object.__setattr__(self, "undispositioned_consumer_ids", undispositioned)
        object.__setattr__(self, "migration_kinds", kinds)
        object.__setattr__(
            self,
            "target_module_id",
            _text(self.target_module_id, "target_module_id", empty=True),
        )

    @property
    def can_retire_facade(self) -> bool:
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
    def projection_is_authority(self) -> bool:
        return False

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": SUBJECT_FACADE_PLAN_SCHEMA,
            "interface": SUBJECT_FACADE_PLAN_INTERFACE,
            "subject_id": self.subject_id,
            "subject_module": self.subject_module,
            "facade_required": self.facade_required,
            "consumer_ids": list(self.consumer_ids),
            "undispositioned_consumer_ids": list(self.undispositioned_consumer_ids),
            "migration_kinds": list(self.migration_kinds),
            "target_module_id": self.target_module_id,
            "can_retire_facade": False,
            "can_authorize_transition": False,
            "can_authorize_completion": False,
            "can_create_authority": False,
            "projection_is_authority": False,
        }
        _require_dag_json(payload, self.__class__.__name__)
        return payload

    @property
    def facade_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["facade_cid"] = self.facade_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SubjectFacadePlan":
        _reject_excluded(data, cls.__name__)
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("facade_cid")
        if payload.pop("schema") != SUBJECT_FACADE_PLAN_SCHEMA:
            raise FacadePlannerError("unsupported SubjectFacadePlan schema")
        if payload.pop("interface") != SUBJECT_FACADE_PLAN_INTERFACE:
            raise FacadePlannerError("unsupported SubjectFacadePlan interface")
        _pop_authority_flags(payload, "SubjectFacadePlan")
        if payload.pop("can_retire_facade") is not False:
            raise FacadePlannerError("subject façade cannot claim can_retire_facade")
        result = cls(**payload)
        _verify_cid(claimed, result.facade_cid, "SubjectFacadePlan facade_cid")
        return result


def _coerce_subject_facade(value: Any) -> SubjectFacadePlan:
    if isinstance(value, SubjectFacadePlan):
        return value
    if isinstance(value, Mapping):
        if "facade_cid" in value:
            return SubjectFacadePlan.from_dict(value)
        payload = {
            key: item
            for key, item in _project(value).items()
            if key
            not in {
                "schema",
                "interface",
                "facade_cid",
                "can_retire_facade",
                *_AUTHORITY_FLAG_NAMES,
            }
        }
        return SubjectFacadePlan(**payload)
    raise FacadePlannerError("subject façade must be a SubjectFacadePlan")


@dataclass(frozen=True, slots=True)
class CompatibilityFacadePlan:
    """Nominated façades and consumer migrations. Predicted SPAR-018 symbol."""

    tree_id: str
    consumer_plans: Sequence[ConsumerMigrationPlan | Mapping[str, Any]]
    subject_facades: Sequence[SubjectFacadePlan | Mapping[str, Any]] = ()
    selected_candidate_cids: Sequence[str] = ()
    rejected_candidate_cids: Sequence[str] = ()
    advisory_candidate_cids: Sequence[str] = ()
    evidence_cids: Sequence[str] = ()
    comparison_receipt_cid: str = ""
    inventory_cid: str = ""
    target_api_plan_cid: str = ""
    analyzer_id: str = ANALYZER_ID

    interface: ClassVar[str] = COMPATIBILITY_FACADE_PLAN_INTERFACE
    schema: ClassVar[str] = COMPATIBILITY_FACADE_PLAN_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "tree_id",
            "consumer_plans",
            "subject_facades",
            "selected_candidate_cids",
            "rejected_candidate_cids",
            "advisory_candidate_cids",
            "negative_evidence_cids",
            "evidence_cids",
            "covered_kinds",
            "covered_surfaces",
            "covered_migration_kinds",
            "comparison_receipt_cid",
            "inventory_cid",
            "target_api_plan_cid",
            "analyzer_id",
            "can_authorize_transition",
            "can_authorize_completion",
            "can_create_authority",
            "projection_is_authority",
            "can_retire_facade",
            "plan_is_nomination_only",
            "plan_cid",
        }
    )

    def __post_init__(self) -> None:
        analyzer = _text(self.analyzer_id, "analyzer_id")
        if analyzer != ANALYZER_ID:
            raise FacadePlannerError("analyzer_id must remain the SPAR-018 analyzer")
        tree_id = _tree_id(self.tree_id)
        plans = tuple(_coerce_migration_plan(item) for item in self.consumer_plans)
        if not plans:
            raise FacadePlannerError("compatibility façade plan requires consumer_plans")
        if len(plans) > MAX_PLANS:
            raise FacadePlannerError("consumer_plans exceed maximum length")
        plan_ids = [item.obligation_id for item in plans]
        if len(plan_ids) != len(set(plan_ids)):
            raise FacadePlannerError("consumer plans must be unique per obligation")
        plans = tuple(
            sorted(plans, key=lambda item: (item.consumer_id, item.obligation_id))
        )
        facades = tuple(
            sorted(
                (_coerce_subject_facade(item) for item in self.subject_facades),
                key=lambda item: item.subject_id,
            )
        )
        if not facades:
            raise FacadePlannerError("compatibility façade plan requires subject_facades")
        subject_ids = [item.subject_id for item in facades]
        if len(subject_ids) != len(set(subject_ids)):
            raise FacadePlannerError("duplicate subject façade identity")
        planned_subjects = {item.subject_id for item in plans}
        facade_subjects = set(subject_ids)
        if planned_subjects != facade_subjects:
            raise FacadePlannerError(
                "subject façades must cover every planned subject exactly once"
            )
        for facade in facades:
            expected_consumers = tuple(
                sorted(
                    {
                        item.consumer_id
                        for item in plans
                        if item.subject_id == facade.subject_id
                    }
                )
            )
            if facade.consumer_ids != expected_consumers:
                raise FacadePlannerError(
                    "subject façade consumers must match consumer plans"
                )
        selected = _unique_sorted_text(
            list(self.selected_candidate_cids),
            "selected_candidate_cids",
            limit=MAX_MODULES,
        )
        rejected = _unique_sorted_text(
            list(self.rejected_candidate_cids),
            "rejected_candidate_cids",
            limit=MAX_MODULES,
        )
        advisory = _unique_sorted_text(
            list(self.advisory_candidate_cids),
            "advisory_candidate_cids",
            limit=MAX_MODULES,
        )
        overlap = set(selected) & set(rejected)
        if overlap:
            raise FacadePlannerError("rejected candidates cannot become façades")
        overlap = set(selected) & set(advisory)
        if overlap:
            raise FacadePlannerError("advisory candidates cannot become façades")
        evidence = tuple(
            sorted(_cid(item, "evidence_cids") for item in self.evidence_cids)
        )
        if len(evidence) != len(set(evidence)):
            raise FacadePlannerError("evidence_cids must not contain duplicates")
        if len(evidence) > MAX_EVIDENCE_CIDS:
            raise FacadePlannerError("evidence_cids exceed maximum length")
        object.__setattr__(self, "tree_id", tree_id)
        object.__setattr__(self, "consumer_plans", plans)
        object.__setattr__(self, "subject_facades", facades)
        object.__setattr__(self, "selected_candidate_cids", selected)
        object.__setattr__(self, "rejected_candidate_cids", rejected)
        object.__setattr__(self, "advisory_candidate_cids", advisory)
        object.__setattr__(self, "evidence_cids", evidence)
        object.__setattr__(
            self,
            "comparison_receipt_cid",
            _optional_cid(self.comparison_receipt_cid, "comparison_receipt_cid"),
        )
        object.__setattr__(
            self, "inventory_cid", _optional_cid(self.inventory_cid, "inventory_cid")
        )
        object.__setattr__(
            self,
            "target_api_plan_cid",
            _optional_cid(self.target_api_plan_cid, "target_api_plan_cid"),
        )
        object.__setattr__(self, "analyzer_id", analyzer)

    @property
    def covered_kinds(self) -> tuple[str, ...]:
        return tuple(sorted({item.kind for item in self.consumer_plans}))

    @property
    def covered_surfaces(self) -> tuple[str, ...]:
        return tuple(sorted({item.surface for item in self.consumer_plans}))

    @property
    def covered_migration_kinds(self) -> tuple[str, ...]:
        return tuple(sorted({item.migration_kind for item in self.consumer_plans}))

    @property
    def negative_evidence_cids(self) -> tuple[str, ...]:
        return self.rejected_candidate_cids

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
    def can_retire_facade(self) -> bool:
        return False

    @property
    def plan_is_nomination_only(self) -> bool:
        return True

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": COMPATIBILITY_FACADE_PLAN_SCHEMA,
            "interface": COMPATIBILITY_FACADE_PLAN_INTERFACE,
            "tree_id": self.tree_id,
            "consumer_plans": [item.to_dict() for item in self.consumer_plans],
            "subject_facades": [item.to_dict() for item in self.subject_facades],
            "selected_candidate_cids": list(self.selected_candidate_cids),
            "rejected_candidate_cids": list(self.rejected_candidate_cids),
            "advisory_candidate_cids": list(self.advisory_candidate_cids),
            "negative_evidence_cids": list(self.negative_evidence_cids),
            "evidence_cids": list(self.evidence_cids),
            "covered_kinds": list(self.covered_kinds),
            "covered_surfaces": list(self.covered_surfaces),
            "covered_migration_kinds": list(self.covered_migration_kinds),
            "comparison_receipt_cid": self.comparison_receipt_cid,
            "inventory_cid": self.inventory_cid,
            "target_api_plan_cid": self.target_api_plan_cid,
            "analyzer_id": self.analyzer_id,
            "can_authorize_transition": False,
            "can_authorize_completion": False,
            "can_create_authority": False,
            "projection_is_authority": False,
            "can_retire_facade": False,
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
    def from_dict(cls, data: Mapping[str, Any]) -> "CompatibilityFacadePlan":
        _reject_excluded(data, cls.__name__)
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("plan_cid")
        if payload.pop("schema") != COMPATIBILITY_FACADE_PLAN_SCHEMA:
            raise FacadePlannerError("unsupported CompatibilityFacadePlan schema")
        if payload.pop("interface") != COMPATIBILITY_FACADE_PLAN_INTERFACE:
            raise FacadePlannerError("unsupported CompatibilityFacadePlan interface")
        _pop_authority_flags(payload, "CompatibilityFacadePlan")
        if payload.pop("plan_is_nomination_only") is not True:
            raise FacadePlannerError("plan must remain nomination_only")
        if payload.pop("can_retire_facade") is not False:
            raise FacadePlannerError("plan cannot claim can_retire_facade")
        payload.pop("negative_evidence_cids")
        payload.pop("covered_kinds")
        payload.pop("covered_surfaces")
        payload.pop("covered_migration_kinds")
        result = cls(**payload)
        _verify_cid(claimed, result.plan_cid, "CompatibilityFacadePlan plan_cid")
        return result


@dataclass(frozen=True, slots=True)
class FacadePlanningReceipt:
    """Body-free planning receipt. Independent validation remains separate."""

    tree_id: str
    plan: CompatibilityFacadePlan | Mapping[str, Any]
    analyzer_id: str = ANALYZER_ID

    interface: ClassVar[str] = FACADE_PLANNING_RECEIPT_INTERFACE
    schema: ClassVar[str] = FACADE_PLANNING_RECEIPT_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "tree_id",
            "plan",
            "analyzer_id",
            "plan_cid",
            "selected_candidate_cids",
            "rejected_candidate_cids",
            "negative_evidence_cids",
            "can_authorize_transition",
            "can_authorize_completion",
            "can_create_authority",
            "projection_is_authority",
            "can_retire_facade",
            "receipt_cid",
        }
    )

    def __post_init__(self) -> None:
        analyzer = _text(self.analyzer_id, "analyzer_id")
        if analyzer != ANALYZER_ID:
            raise FacadePlannerError("analyzer_id must remain the SPAR-018 analyzer")
        plan = (
            self.plan
            if isinstance(self.plan, CompatibilityFacadePlan)
            else CompatibilityFacadePlan.from_dict(_mapping(self.plan, "plan"))
        )
        tree_id = _tree_id(self.tree_id)
        if plan.tree_id != tree_id:
            raise FacadePlannerError("receipt tree_id does not match plan")
        object.__setattr__(self, "tree_id", tree_id)
        object.__setattr__(self, "plan", plan)
        object.__setattr__(self, "analyzer_id", analyzer)

    @property
    def plan_cid(self) -> str:
        return self.plan.plan_cid

    @property
    def selected_candidate_cids(self) -> tuple[str, ...]:
        return self.plan.selected_candidate_cids

    @property
    def rejected_candidate_cids(self) -> tuple[str, ...]:
        return self.plan.rejected_candidate_cids

    @property
    def negative_evidence_cids(self) -> tuple[str, ...]:
        return self.plan.negative_evidence_cids

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
    def can_retire_facade(self) -> bool:
        return False

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": FACADE_PLANNING_RECEIPT_SCHEMA,
            "interface": FACADE_PLANNING_RECEIPT_INTERFACE,
            "tree_id": self.tree_id,
            "plan": self.plan.to_dict(),
            "analyzer_id": self.analyzer_id,
            "plan_cid": self.plan_cid,
            "selected_candidate_cids": list(self.selected_candidate_cids),
            "rejected_candidate_cids": list(self.rejected_candidate_cids),
            "negative_evidence_cids": list(self.negative_evidence_cids),
            "can_authorize_transition": False,
            "can_authorize_completion": False,
            "can_create_authority": False,
            "projection_is_authority": False,
            "can_retire_facade": False,
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
    def from_dict(cls, data: Mapping[str, Any]) -> "FacadePlanningReceipt":
        _reject_excluded(data, cls.__name__)
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("receipt_cid")
        if payload.pop("schema") != FACADE_PLANNING_RECEIPT_SCHEMA:
            raise FacadePlannerError("unsupported FacadePlanningReceipt schema")
        if payload.pop("interface") != FACADE_PLANNING_RECEIPT_INTERFACE:
            raise FacadePlannerError("unsupported FacadePlanningReceipt interface")
        _pop_authority_flags(payload, "FacadePlanningReceipt")
        if payload.pop("can_retire_facade") is not False:
            raise FacadePlannerError("receipt cannot claim can_retire_facade")
        payload.pop("plan_cid")
        payload.pop("selected_candidate_cids")
        payload.pop("rejected_candidate_cids")
        payload.pop("negative_evidence_cids")
        result = cls(**payload)
        _verify_cid(claimed, result.receipt_cid, "FacadePlanningReceipt receipt_cid")
        return result


def _coerce_candidate(
    value: ProgramPartitionCandidate | Mapping[str, Any],
) -> ProgramPartitionCandidate:
    if isinstance(value, ProgramPartitionCandidate):
        return value
    if isinstance(value, Mapping):
        if "candidate_cid" in value:
            return ProgramPartitionCandidate.from_dict(value)
        payload = {
            key: item
            for key, item in _project(value).items()
            if key
            not in {
                "schema",
                "interface",
                "candidate_cid",
                *_AUTHORITY_FLAG_NAMES,
            }
        }
        return ProgramPartitionCandidate(**payload)
    raise FacadePlannerError("candidate must be a ProgramPartitionCandidate")


def _coerce_candidates(
    value: PartitionGenerationReceipt
    | Sequence[ProgramPartitionCandidate | Mapping[str, Any]]
    | Mapping[str, Any],
) -> tuple[ProgramPartitionCandidate, ...]:
    if isinstance(value, PartitionGenerationReceipt):
        return value.candidates
    if isinstance(value, Mapping) and "candidates" in value:
        if "receipt_cid" in value:
            return PartitionGenerationReceipt.from_dict(value).candidates
        items = value.get("candidates") or ()
        return tuple(_coerce_candidate(item) for item in items)
    if isinstance(value, (list, tuple)):
        candidates = tuple(_coerce_candidate(item) for item in value)
        if not candidates:
            raise FacadePlannerError("façade planning requires SPAR-014 candidates")
        return candidates
    raise FacadePlannerError("candidates must be a SPAR-013 receipt or candidate list")


def _coerce_comparison(
    value: PartitionComparisonReceipt | Mapping[str, Any] | None,
    candidates: Sequence[ProgramPartitionCandidate],
) -> PartitionComparisonReceipt:
    if value is None:
        return compare_partition_candidates(candidates)
    if isinstance(value, PartitionComparisonReceipt):
        comparison = value
    elif isinstance(value, Mapping):
        comparison = PartitionComparisonReceipt.from_dict(value)
    else:
        raise FacadePlannerError("comparison must be a SPAR-014 receipt")
    if comparison.analyzer_id != SPAR014_ANALYZER_ID:
        raise FacadePlannerError("comparison must remain the SPAR-014 analyzer")
    return comparison


def _pairwise_disjoint(candidates: Sequence[ProgramPartitionCandidate]) -> bool:
    seen: set[str] = set()
    for candidate in candidates:
        overlap = seen.intersection(candidate.member_ids)
        if overlap:
            return False
        seen.update(candidate.member_ids)
    return True


def _select_candidates(
    candidates: Sequence[ProgramPartitionCandidate],
    comparison: PartitionComparisonReceipt,
    selected_candidate_cids: Sequence[str] | None,
) -> tuple[ProgramPartitionCandidate, ...]:
    by_cid = {item.candidate_cid: item for item in candidates}
    ranked_cids = set(comparison.ranked_candidate_cids)
    rejected_cids = set(comparison.rejected_candidate_cids)
    advisory_cids = set(comparison.advisory_candidate_cids)
    if selected_candidate_cids is None:
        ranked = tuple(
            by_cid[item] for item in comparison.ranked_candidate_cids if item in by_cid
        )
        if not ranked:
            raise FacadePlannerError("no ranked SPAR-014 candidates")
        if not _pairwise_disjoint(ranked):
            raise FacadePlannerError(
                "overlapping ranked candidates require selected_candidate_cids"
            )
        selected = ranked
    else:
        selected_ids = _unique_sorted_text(
            list(selected_candidate_cids),
            "selected_candidate_cids",
            limit=MAX_MODULES,
        )
        selected_list: list[ProgramPartitionCandidate] = []
        for cid in selected_ids:
            if cid not in by_cid:
                raise FacadePlannerError("selected candidate is not present")
            if cid not in ranked_cids:
                raise FacadePlannerError("selected candidate is not ranked by SPAR-014")
            if cid in rejected_cids:
                raise FacadePlannerError("rejected candidates cannot become façades")
            if cid in advisory_cids:
                raise FacadePlannerError("advisory candidates cannot become façades")
            selected_list.append(by_cid[cid])
        selected = tuple(selected_list)
        if not selected:
            raise FacadePlannerError("selected_candidate_cids must not be empty")
        if not _pairwise_disjoint(selected):
            raise FacadePlannerError(
                "overlapping modules cannot form a simultaneous façade plan"
            )
    tree_id = selected[0].tree_id
    for candidate in selected:
        if candidate.advisory:
            raise FacadePlannerError("advisory candidates cannot become façades")
        if candidate.hard_constraint_violations:
            raise FacadePlannerError("violating candidates cannot become façades")
        if candidate.evidence_class in _NON_ADMITTING_EVIDENCE:
            raise FacadePlannerError("vector or model evidence cannot admit a façade")
        if candidate.tree_id != tree_id:
            raise FacadePlannerError("selected candidate tree_id does not match")
    return selected


def _target_module_for(
    *,
    consumer_id: str,
    subject_id: str,
    subject_module: str,
    selected: Sequence[ProgramPartitionCandidate],
    target_modules: Mapping[str, str],
) -> str:
    if consumer_id in target_modules:
        return target_modules[consumer_id]
    if subject_id in target_modules:
        return target_modules[subject_id]
    if subject_module in target_modules:
        return target_modules[subject_module]
    for candidate in selected:
        if consumer_id in candidate.consumer_ids:
            return candidate.candidate_cid
        if subject_id in candidate.member_ids or subject_module in candidate.member_ids:
            return candidate.candidate_cid
    return ""


def _target_modules_from_api_plan(value: Any) -> tuple[dict[str, str], str]:
    if value in (None, ""):
        return {}, ""
    payload = _mapping(value, "target_api_plan")
    plan_cid = _optional_cid(payload.get("plan_cid") or payload.get("api_cid"), "plan_cid")
    modules = payload.get("modules") or ()
    if not isinstance(modules, (list, tuple)):
        raise FacadePlannerError("target_api_plan modules must be a list")
    mapping: dict[str, str] = {}
    for module in modules:
        if not isinstance(module, Mapping):
            to_dict = getattr(module, "to_dict", None)
            module = to_dict() if callable(to_dict) else module
        if not isinstance(module, Mapping):
            raise FacadePlannerError("target module must be a mapping")
        module_id = _text(module.get("module_id") or "", "module_id", empty=True)
        if not module_id:
            continue
        for export in list(module.get("public_exports") or ()):
            if isinstance(export, Mapping):
                for consumer_id in export.get("consumer_ids") or ():
                    mapping[str(consumer_id)] = module_id
                member_id = export.get("member_id")
                if member_id:
                    mapping[str(member_id)] = module_id
        for member_id in module.get("member_ids") or ():
            mapping[str(member_id)] = module_id
    return mapping, plan_cid


def _assert_required_terminals(inventory: CompatibilityInventoryView) -> None:
    for item in inventory.obligations:
        if not item.required:
            continue
        if item.disposition == CompatibilityDisposition.UNDISPOSITIONED.value:
            raise FacadePlannerError(
                "required undispositioned consumer is a typed terminal"
            )
        if item.disposition == CompatibilityDisposition.UNSUPPORTED.value:
            raise FacadePlannerError(
                "unsupported required behavior is a typed terminal"
            )


def _assert_partition_consumers_inventoried(
    selected: Sequence[ProgramPartitionCandidate],
    inventory: CompatibilityInventoryView,
) -> None:
    inventoried = {item.consumer_id for item in inventory.consumers}
    missing: set[str] = set()
    for candidate in selected:
        missing.update(set(candidate.consumer_ids) - inventoried)
    if missing:
        raise FacadePlannerError(
            f"SPAR-014 consumers missing from SPAR-011 inventory: {sorted(missing)}"
        )


def plan_compatibility_facades(
    inventory: Mapping[str, Any] | CompatibilityInventoryView | Any,
    *,
    candidates: PartitionGenerationReceipt
    | Sequence[ProgramPartitionCandidate | Mapping[str, Any]]
    | Mapping[str, Any],
    comparison: PartitionComparisonReceipt | Mapping[str, Any] | None = None,
    selected_candidate_cids: Sequence[str] | None = None,
    target_api_plan: Mapping[str, Any] | Any | None = None,
) -> CompatibilityFacadePlan:
    """Nominate façades and migrations for every SPAR-011 consumer.

    SPAR-014 ranked, non-advisory candidates bind target modules. Rejected
    comparisons remain negative evidence. Required undispositioned or
    unsupported obligations fail closed. The result cannot authorize a
    transition, completion, or façade retirement.
    """

    resolved_inventory = _coerce_inventory(inventory)
    resolved_candidates = _coerce_candidates(candidates)
    tree_ids = {item.tree_id for item in resolved_candidates}
    if len(tree_ids) != 1:
        raise FacadePlannerError("candidates must share one tree_id")
    tree_id = next(iter(tree_ids))
    if resolved_inventory.tree_id != tree_id:
        raise FacadePlannerError("inventory tree_id does not match candidates")
    resolved_comparison = _coerce_comparison(comparison, resolved_candidates)
    if resolved_comparison.tree_id != tree_id:
        raise FacadePlannerError("comparison tree_id does not match candidates")
    selected = _select_candidates(
        resolved_candidates, resolved_comparison, selected_candidate_cids
    )
    _assert_required_terminals(resolved_inventory)
    _assert_partition_consumers_inventoried(selected, resolved_inventory)
    api_targets, target_api_plan_cid = _target_modules_from_api_plan(target_api_plan)
    consumers_by_id = {
        item.consumer_id: item for item in resolved_inventory.consumers
    }
    consumer_plans = tuple(
        ConsumerMigrationPlan(
            consumer_id=item.consumer_id,
            obligation_id=item.obligation_id,
            subject_id=item.subject_id,
            subject_module=item.subject_module,
            kind=item.kind,
            disposition=item.disposition,
            required=item.required,
            role=consumers_by_id[item.consumer_id].role,
            target_module_id=_target_module_for(
                consumer_id=item.consumer_id,
                subject_id=item.subject_id,
                subject_module=item.subject_module,
                selected=selected,
                target_modules=api_targets,
            ),
        )
        for item in resolved_inventory.obligations
    )
    grouped: dict[str, list[ConsumerMigrationPlan]] = {}
    for plan in consumer_plans:
        grouped.setdefault(plan.subject_id, []).append(plan)
    subject_facades = []
    for subject_id, members in grouped.items():
        modules = {item.subject_module for item in members}
        if len(modules) != 1:
            raise FacadePlannerError(
                f"subject {subject_id} has conflicting subject_module values"
            )
        undispositioned = tuple(
            sorted(
                {
                    item.consumer_id
                    for item in members
                    if item.disposition
                    == CompatibilityDisposition.UNDISPOSITIONED.value
                }
            )
        )
        kinds = tuple(sorted({item.migration_kind for item in members}))
        facade_required = bool(undispositioned) or MigrationKind.FACADE.value in kinds
        subject_facades.append(
            SubjectFacadePlan(
                subject_id=subject_id,
                subject_module=next(iter(modules)),
                facade_required=facade_required,
                consumer_ids=tuple(sorted({item.consumer_id for item in members})),
                undispositioned_consumer_ids=undispositioned,
                migration_kinds=kinds,
                target_module_id=members[0].target_module_id,
            )
        )
    evidence = tuple(
        sorted(
            {
                *resolved_comparison.evidence_cids,
                *(cid for item in selected for cid in item.evidence_cids),
            }
        )
    )
    inventory_cid = resolved_inventory.inventory_cid
    if not inventory_cid:
        inventory_cid = cid_for_dag_json(
            {
                "tree_id": resolved_inventory.tree_id,
                "obligation_ids": [
                    item.obligation_id for item in resolved_inventory.obligations
                ],
                "consumer_ids": [
                    item.consumer_id for item in resolved_inventory.consumers
                ],
            }
        )
    return CompatibilityFacadePlan(
        tree_id=tree_id,
        consumer_plans=consumer_plans,
        subject_facades=tuple(subject_facades),
        selected_candidate_cids=tuple(item.candidate_cid for item in selected),
        rejected_candidate_cids=tuple(resolved_comparison.rejected_candidate_cids),
        advisory_candidate_cids=tuple(resolved_comparison.advisory_candidate_cids),
        evidence_cids=evidence,
        comparison_receipt_cid=resolved_comparison.receipt_cid,
        inventory_cid=inventory_cid,
        target_api_plan_cid=target_api_plan_cid,
        analyzer_id=ANALYZER_ID,
    )


def compile_facade_plan_receipt(
    plan: CompatibilityFacadePlan | Mapping[str, Any],
) -> FacadePlanningReceipt:
    resolved = (
        plan
        if isinstance(plan, CompatibilityFacadePlan)
        else CompatibilityFacadePlan.from_dict(plan)
    )
    return FacadePlanningReceipt(tree_id=resolved.tree_id, plan=resolved)


def encode_canonical_plan(plan: CompatibilityFacadePlan) -> dict[str, Any]:
    return plan.to_dict()


def decode_canonical_plan(payload: Mapping[str, Any]) -> CompatibilityFacadePlan:
    return CompatibilityFacadePlan.from_dict(payload)


def encode_canonical_receipt(receipt: FacadePlanningReceipt) -> dict[str, Any]:
    return receipt.to_dict()


def decode_canonical_receipt(payload: Mapping[str, Any]) -> FacadePlanningReceipt:
    return FacadePlanningReceipt.from_dict(payload)


def provider_free_exports() -> tuple[str, ...]:
    return tuple(sorted(__all__))


def assert_not_competing_capsule_family() -> None:
    names = set(globals())
    overlap = names & _FORBIDDEN_CAPSULE_TYPE_NAMES
    if overlap:
        raise FacadePlannerError(
            f"façade planner must not define capsule types: {sorted(overlap)}"
        )


__all__ = [
    "ANALYZER_ID",
    "AUTHORITY",
    "AUTHORITY_OWNER",
    "COMPATIBILITY_FACADE_PLAN_INTERFACE",
    "CONSUMER_MIGRATION_PLAN_INTERFACE",
    "DECLARED_COMPATIBILITY_KINDS",
    "DECLARED_CONSUMER_SURFACES",
    "DECLARED_MIGRATION_KINDS",
    "DISPOSITIONED",
    "DUCKLAKE_IS_AUTHORITY",
    "FACADE_CAN_AUTHORIZE_COMPLETION",
    "FACADE_CAN_AUTHORIZE_TRANSITION",
    "FACADE_CAN_CREATE_AUTHORITY",
    "FACADE_CAN_RETIRE_FACADE",
    "FACADE_PLANNING_RECEIPT_INTERFACE",
    "FACADE_PLAN_CONTRACT_VERSION",
    "GOAL_ID",
    "IDENTITY_EXCLUDED_FIELDS",
    "KIND_MIGRATION",
    "KIND_SURFACE",
    "MARKDOWN_IS_NOT_COMPLETION",
    "MODEL_OUTPUT_IS_PROPOSAL_ONLY",
    "PLAN_IS_NOMINATION_ONLY",
    "PROGRAM",
    "PROJECTION_CLUSTERING_IS_AUTHORITY",
    "RAW_SOURCE_REQUIRED",
    "SOFT_SIGNALS_CANNOT_OVERRIDE_HARD_CONSTRAINTS",
    "SUBJECT_FACADE_PLAN_INTERFACE",
    "TASK_ID",
    "TEST_PASS_IS_NOT_COMPLETION",
    "VECTOR_SIMILARITY_IS_AUTHORITY",
    "WORKER_SELF_APPROVAL",
    "CompatibilityDisposition",
    "CompatibilityFacadePlan",
    "CompatibilityKind",
    "ConsumerMigrationPlan",
    "ConsumerRole",
    "ConsumerSurface",
    "FacadePlannerError",
    "FacadePlanningReceipt",
    "MigrationKind",
    "SubjectFacadePlan",
    "assert_not_competing_capsule_family",
    "compile_facade_plan_receipt",
    "decode_canonical_plan",
    "decode_canonical_receipt",
    "encode_canonical_plan",
    "encode_canonical_receipt",
    "facade_planner_cid_profile",
    "family_for_kind",
    "migration_kind_for",
    "plan_compatibility_facades",
    "provider_free_exports",
    "surface_for_kind",
]
