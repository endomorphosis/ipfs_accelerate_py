"""Reviewed, capability-declared repair operator registry (PDR-050).

Interfaces:

* ``RepairOperatorRegistry@1``
* ``DoctorRepairOperatorSpec@2``

The registry is an immutable catalogue of *proposal grammars*.  It does not
hold renderer callables, source bodies, proof verdicts, permits, or mutation
handles.  Lookup therefore cannot grant semantic, proof, or write authority.
Actual rendering of the already-supported analytical transforms remains in
``deterministic_doctor_transforms`` behind its existing proof and path gates.

The v2 catalogue covers every current ``TransformKind`` plus exact move,
tracked-artifact restoration, and reviewed semantic-patch/equality-rewrite
hooks.  Resolution is fail-closed: target, value, or placement ambiguity is a
rejection; missing evidence/capability is an abstention; and dynamic,
generated, stateful, native, public-API, or dependency-changing work is
reported as approval-required without treating an approval reference as
proof.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any, ClassVar, Final

from ..analysis.change_propagation_contracts import TransformKind
from ..analysis.deterministic_doctor_contracts import (
    DoctorAuthorityRoots,
    DoctorOperatorKind,
)
from ..proof.formal_verification_contracts import CanonicalContract, content_identity
from .deterministic_doctor_transforms import (
    ANALYTICAL_TRANSFORM_OPERATOR_BINDINGS,
    DoctorRepairOperatorRegistry as LegacyDoctorRepairOperatorRegistry,
    build_default_doctor_operator_registry,
)


REPAIR_OPERATOR_REGISTRY_INTERFACE: Final[str] = "RepairOperatorRegistry@1"
DOCTOR_REPAIR_OPERATOR_SPEC_INTERFACE: Final[str] = "DoctorRepairOperatorSpec@2"
REPAIR_OPERATOR_REGISTRY_VERSION: Final[int] = 1
DOCTOR_REPAIR_OPERATOR_SPEC_VERSION: Final[int] = 2
REPAIR_OPERATOR_REGISTRY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/repair-operator-registry@1"
)
DOCTOR_REPAIR_OPERATOR_SPEC_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/doctor-repair-operator-spec@2"
)
REPAIR_OPERATOR_LOOKUP_REQUEST_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/repair-operator-lookup-request@1"
)
REPAIR_OPERATOR_LOOKUP_RESULT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/repair-operator-lookup-result@1"
)
REPAIR_OPERATOR_REGISTRY_PRODUCER: Final[str] = "repair-operator-registry@1"

MAX_OPERATOR_COUNT: Final[int] = 64
MAX_REFERENCE_COUNT: Final[int] = 256
MAX_TEXT_BYTES: Final[int] = 4_096
MAX_PATH_BYTES: Final[int] = 1_024


class RepairOperatorRegistryError(ValueError):
    """Malformed registry declaration or lookup input."""


class RepairOperatorRegistryAuthorityError(RepairOperatorRegistryError):
    """An operator declaration attempted to carry forbidden authority."""


class UnknownRepairOperatorError(RepairOperatorRegistryError):
    """A caller requested a kind outside the reviewed closed catalogue."""


class RepairOperatorKind(str, Enum):
    """Canonical v2 repair operator kinds.

    Aliases preserve the terminology used by diagnoses while serialization
    always emits the canonical value.
    """

    EXACT_RENAME = "exact_rename"
    EXACT_MOVE = "exact_move"
    ADD_ARGUMENT = "add_argument"
    MISSING_ARGUMENT = "add_argument"
    RENAME_ARGUMENT = "rename_argument"
    REORDER_ARGUMENT = "reorder_argument"
    THREAD_ARGUMENT = "thread_argument"
    VALUE_THREADING = "thread_argument"
    ADD_IMPORT = "add_import"
    EXACT_IMPORT = "add_import"
    ADD_EXPORT = "add_export"
    EXACT_EXPORT = "add_export"
    ADD_REGISTRATION = "add_registration"
    EXACT_REGISTRATION = "add_registration"
    ADD_CONSTRUCTOR_ROUTE = "add_constructor_route"
    CONSTRUCTOR = "add_constructor_route"
    ADD_FACTORY_ROUTE = "add_factory_route"
    FACTORY = "add_factory_route"
    FINITE_ADAPTER = "finite_adapter"
    ADAPTER = "finite_adapter"
    SCHEMA_PROJECTION = "schema_projection"
    SCHEMA = "schema_projection"
    SERIALIZER_UPDATE = "serializer_update"
    SERIALIZER = "serializer_update"
    FIXTURE_UPDATE = "fixture_update"
    FIXTURE = "fixture_update"
    MANIFEST_UPDATE = "manifest_update"
    MANIFEST = "manifest_update"
    RESTORE_TRACKED_ARTIFACT = "restore_tracked_artifact"
    ARTIFACT = "restore_tracked_artifact"
    SEMANTIC_PATCH = "semantic_patch"
    EQUALITY_REWRITE = "equality_rewrite"


class RepairOperatorFamily(str, Enum):
    SYMBOL = "symbol"
    MOVE = "move"
    CALL = "call"
    WIRING = "wiring"
    CONSTRUCTION = "construction"
    DATA_CONTRACT = "data_contract"
    ARTIFACT = "artifact"
    REVIEWED_REWRITE = "reviewed_rewrite"


class OperatorValueRequirement(str, Enum):
    NONE = "none"
    UNIQUE_PROVED = "unique_proved"
    TOTAL_MAPPING = "total_mapping"
    VERIFIED_PREIMAGE = "verified_preimage"
    REVIEWED_RULE = "reviewed_rule"


class ReviewedRepairHook(str, Enum):
    NONE = "none"
    EXACT_MOVE = "exact_move"
    ARTIFACT_RESTORE = "artifact_restore"
    SEMANTIC_PATCH = "semantic_patch"
    EQUALITY_REWRITE = "equality_rewrite"


class RepairOperatorCapability(str, Enum):
    """Capabilities are requirements, never authority tokens."""

    EXACT_TARGET = "exact_target"
    EXACT_PLACEMENT = "exact_placement"
    CLOSED_AST = "closed_ast"
    IDEMPOTENT_RENDER = "idempotent_render"
    SCOPE_BOUND = "scope_bound"
    PROPOSAL_ONLY = "proposal_only"
    SYMBOL_EQUIVALENCE = "symbol_equivalence"
    FILE_MOVE = "file_move"
    UNIQUE_VALUE = "unique_value"
    ROUTE_CLOSURE = "route_closure"
    IMPORT_WIRING = "import_wiring"
    EXPORT_WIRING = "export_wiring"
    REGISTRATION_WIRING = "registration_wiring"
    CONSTRUCTOR_WIRING = "constructor_wiring"
    FACTORY_WIRING = "factory_wiring"
    FINITE_ADAPTER = "finite_adapter"
    TOTAL_FIELD_MAPPING = "total_field_mapping"
    SERIALIZER_MAPPING = "serializer_mapping"
    FIXTURE_MAPPING = "fixture_mapping"
    MANIFEST_MAPPING = "manifest_mapping"
    VERIFIED_ARTIFACT = "verified_artifact"
    REVIEWED_SEMANTIC_PATCH = "reviewed_semantic_patch"
    DECLARED_EQUALITY_THEORY = "declared_equality_theory"


class RepairBehaviorClass(str, Enum):
    PURE_LOCAL = "pure_local"
    UNKNOWN = "unknown"
    DYNAMIC = "dynamic"
    GENERATED = "generated"
    STATEFUL = "stateful"
    NATIVE = "native"
    PUBLIC_API = "public_api"
    DEPENDENCY_CHANGING = "dependency_changing"


class RepairOperatorLookupDisposition(str, Enum):
    """Resolution outcome; ``PROPOSAL_ELIGIBLE`` is not admission."""

    PROPOSAL_ELIGIBLE = "proposal_eligible"
    APPROVAL_REQUIRED = "approval_required"
    ABSTAINED = "abstained"
    REJECTED = "rejected"


class RepairOperatorLookupReason(str, Enum):
    UNKNOWN_OPERATOR = "unknown_operator"
    UNKNOWN_BEHAVIOR = "unknown_behavior"
    UNSUPPORTED_LANGUAGE = "unsupported_language"
    UNSUPPORTED_AST_SHAPE = "unsupported_ast_shape"
    TARGET_MISSING = "target_missing"
    TARGET_AMBIGUOUS = "target_ambiguous"
    VALUE_MISSING = "value_missing"
    VALUE_AMBIGUOUS = "value_ambiguous"
    PLACEMENT_MISSING = "placement_missing"
    PLACEMENT_AMBIGUOUS = "placement_ambiguous"
    SCOPE_MISSING = "scope_missing"
    SCOPE_ESCAPE = "scope_escape"
    CAPABILITY_MISSING = "capability_missing"
    PROOF_REFERENCE_MISSING = "proof_reference_missing"
    REVIEW_REFERENCE_MISSING = "review_reference_missing"
    DYNAMIC_APPROVAL = "dynamic_behavior_requires_approval"
    GENERATED_APPROVAL = "generated_behavior_requires_approval"
    STATEFUL_APPROVAL = "stateful_behavior_requires_approval"
    NATIVE_APPROVAL = "native_behavior_requires_approval"
    PUBLIC_API_APPROVAL = "public_api_change_requires_approval"
    DEPENDENCY_APPROVAL = "dependency_change_requires_approval"
    CANDIDATE_ONLY = "candidate_only"


_APPROVAL_BEHAVIORS: Final[Mapping[str, RepairOperatorLookupReason]] = MappingProxyType(
    {
        RepairBehaviorClass.DYNAMIC.value: RepairOperatorLookupReason.DYNAMIC_APPROVAL,
        RepairBehaviorClass.GENERATED.value: RepairOperatorLookupReason.GENERATED_APPROVAL,
        RepairBehaviorClass.STATEFUL.value: RepairOperatorLookupReason.STATEFUL_APPROVAL,
        RepairBehaviorClass.NATIVE.value: RepairOperatorLookupReason.NATIVE_APPROVAL,
        RepairBehaviorClass.PUBLIC_API.value: RepairOperatorLookupReason.PUBLIC_API_APPROVAL,
        RepairBehaviorClass.DEPENDENCY_CHANGING.value: (
            RepairOperatorLookupReason.DEPENDENCY_APPROVAL
        ),
    }
)

_KIND_ALIASES: Final[Mapping[str, RepairOperatorKind]] = MappingProxyType(
    {
        "rename": RepairOperatorKind.EXACT_RENAME,
        "move": RepairOperatorKind.EXACT_MOVE,
        "missing_argument": RepairOperatorKind.ADD_ARGUMENT,
        "value_threading": RepairOperatorKind.THREAD_ARGUMENT,
        "thread_value": RepairOperatorKind.THREAD_ARGUMENT,
        "exact_import": RepairOperatorKind.ADD_IMPORT,
        "import": RepairOperatorKind.ADD_IMPORT,
        "exact_export": RepairOperatorKind.ADD_EXPORT,
        "export": RepairOperatorKind.ADD_EXPORT,
        "exact_registration": RepairOperatorKind.ADD_REGISTRATION,
        "registration": RepairOperatorKind.ADD_REGISTRATION,
        "constructor": RepairOperatorKind.ADD_CONSTRUCTOR_ROUTE,
        "factory": RepairOperatorKind.ADD_FACTORY_ROUTE,
        "adapter": RepairOperatorKind.FINITE_ADAPTER,
        "schema": RepairOperatorKind.SCHEMA_PROJECTION,
        "serializer": RepairOperatorKind.SERIALIZER_UPDATE,
        "fixture": RepairOperatorKind.FIXTURE_UPDATE,
        "manifest": RepairOperatorKind.MANIFEST_UPDATE,
        "artifact": RepairOperatorKind.RESTORE_TRACKED_ARTIFACT,
        "equality": RepairOperatorKind.EQUALITY_REWRITE,
    }
)


def _text(
    value: Any,
    name: str,
    *,
    required: bool = True,
    limit: int = MAX_TEXT_BYTES,
) -> str:
    if not isinstance(value, str):
        raise RepairOperatorRegistryError(f"{name} must be a string")
    result = value.strip()
    if required and not result:
        raise RepairOperatorRegistryError(f"{name} must not be empty")
    if "\x00" in result:
        raise RepairOperatorRegistryError(f"{name} must not contain NUL")
    if len(result.encode("utf-8")) > limit:
        raise RepairOperatorRegistryError(f"{name} exceeds its byte bound")
    return result


def _optional_text(value: Any, name: str) -> str:
    if value in (None, ""):
        return ""
    return _text(value, name)


def _ids(
    values: Any,
    name: str,
    *,
    required: bool = False,
    preserve_order: bool = False,
) -> tuple[str, ...]:
    if values is None:
        raw: Sequence[Any] = ()
    elif isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise RepairOperatorRegistryError(f"{name} must be a sequence")
    else:
        raw = values
    result: list[str] = []
    for item in raw:
        value = _text(item, name)
        if value not in result:
            result.append(value)
    if required and not result:
        raise RepairOperatorRegistryError(f"{name} must not be empty")
    if len(result) > MAX_REFERENCE_COUNT:
        raise RepairOperatorRegistryError(f"{name} exceeds its item bound")
    return tuple(result if preserve_order else sorted(result))


def _paths(values: Any, name: str) -> tuple[str, ...]:
    normalized: list[str] = []
    for raw in _ids(values, name, preserve_order=True):
        text = raw.replace("\\", "/")
        if len(text.encode("utf-8")) > MAX_PATH_BYTES:
            raise RepairOperatorRegistryError(f"{name} exceeds its path byte bound")
        path = PurePosixPath(text)
        if path.is_absolute() or ".." in path.parts or text in {"", "."}:
            raise RepairOperatorRegistryError(
                f"{name} must contain relative repository paths without escape"
            )
        value = path.as_posix()
        if value not in normalized:
            normalized.append(value)
    return tuple(sorted(normalized))


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise RepairOperatorRegistryError(f"{name} must be a boolean")
    return value


def _enum(value: Any, enum_type: type[Enum], name: str) -> Enum:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        raise RepairOperatorRegistryError(f"{name} has an unsupported value") from exc


def normalize_repair_operator_kind(value: Any) -> RepairOperatorKind:
    """Normalize a canonical kind, operator id, enum alias, or reviewed alias."""

    if isinstance(value, RepairOperatorKind):
        return value
    raw = str(getattr(value, "value", value)).strip().lower().replace("-", "_")
    if raw.startswith("repair_operator:"):
        raw = raw.split(":", 1)[1].split("@", 1)[0]
    elif raw.startswith("operator:"):
        raw = raw.split(":", 1)[1]
    if raw in _KIND_ALIASES:
        return _KIND_ALIASES[raw]
    try:
        return RepairOperatorKind(raw)
    except ValueError as exc:
        raise UnknownRepairOperatorError(
            RepairOperatorLookupReason.UNKNOWN_OPERATOR.value
        ) from exc


@dataclass(frozen=True)
class DoctorRepairOperatorSpec(CanonicalContract):
    """Canonical v2 proposal grammar for one reviewed repair operator."""

    SCHEMA: ClassVar[str] = DOCTOR_REPAIR_OPERATOR_SPEC_SCHEMA
    INTERFACE: ClassVar[str] = DOCTOR_REPAIR_OPERATOR_SPEC_INTERFACE

    operator_id: str
    kind: RepairOperatorKind
    family: RepairOperatorFamily
    aliases: tuple[str, ...]
    supported_languages: tuple[str, ...]
    supported_ast_shapes: tuple[str, ...]
    capability_refs: tuple[str, ...]
    precondition_refs: tuple[str, ...]
    postcondition_refs: tuple[str, ...]
    frame_condition_refs: tuple[str, ...]
    proof_requirement_refs: tuple[str, ...]
    validation_requirement_refs: tuple[str, ...]
    scope_constraints: tuple[str, ...]
    approval_classes: tuple[str, ...]
    abstain_classes: tuple[str, ...]
    value_requirement: OperatorValueRequirement = OperatorValueRequirement.NONE
    placement_required: bool = True
    analytical_transform_kind: str = ""
    reviewed_hook: ReviewedRepairHook = ReviewedRepairHook.NONE
    review_requirement_refs: tuple[str, ...] = ()
    renderer_id: str = ""
    idempotent: bool = True
    inverse_or_compensation_ref: str = ""
    proposal_only: bool = True
    semantic_authority: bool = False
    grants_proof_authority: bool = False
    grants_write_authority: bool = False

    def __post_init__(self) -> None:
        kind = normalize_repair_operator_kind(self.kind)
        object.__setattr__(self, "kind", kind)
        family = _enum(self.family, RepairOperatorFamily, "family")
        object.__setattr__(self, "family", family)
        expected_id = f"repair-operator:{kind.value}@2"
        operator_id = _text(self.operator_id, "operator_id")
        if operator_id != expected_id:
            raise RepairOperatorRegistryError(
                f"operator_id must be canonical for {kind.value}: {expected_id}"
            )
        object.__setattr__(self, "operator_id", operator_id)
        for name, required in (
            ("aliases", False),
            ("supported_languages", True),
            ("supported_ast_shapes", True),
            ("capability_refs", True),
            ("precondition_refs", True),
            ("postcondition_refs", True),
            ("frame_condition_refs", True),
            ("proof_requirement_refs", True),
            ("validation_requirement_refs", True),
            ("scope_constraints", True),
            ("approval_classes", True),
            ("abstain_classes", True),
            ("review_requirement_refs", False),
        ):
            object.__setattr__(
                self,
                name,
                _ids(getattr(self, name), name, required=required),
            )
        value_requirement = _enum(
            self.value_requirement,
            OperatorValueRequirement,
            "value_requirement",
        )
        object.__setattr__(self, "value_requirement", value_requirement)
        object.__setattr__(
            self,
            "placement_required",
            _bool(self.placement_required, "placement_required"),
        )
        analytical = _optional_text(
            self.analytical_transform_kind,
            "analytical_transform_kind",
        )
        if analytical:
            try:
                TransformKind(analytical)
            except ValueError as exc:
                raise RepairOperatorRegistryError(
                    "analytical_transform_kind is not an existing TransformKind"
                ) from exc
        object.__setattr__(self, "analytical_transform_kind", analytical)
        hook = _enum(self.reviewed_hook, ReviewedRepairHook, "reviewed_hook")
        object.__setattr__(self, "reviewed_hook", hook)
        if analytical and hook is not ReviewedRepairHook.NONE:
            raise RepairOperatorRegistryError(
                "operator cannot combine an analytical renderer with a reviewed hook"
            )
        if not analytical and hook is ReviewedRepairHook.NONE:
            raise RepairOperatorRegistryError(
                "operator must declare an analytical transform or reviewed hook"
            )
        if hook in {
            ReviewedRepairHook.EXACT_MOVE,
            ReviewedRepairHook.SEMANTIC_PATCH,
            ReviewedRepairHook.EQUALITY_REWRITE,
        } and not self.review_requirement_refs:
            raise RepairOperatorRegistryError(
                "reviewed hooks must declare review requirements"
            )
        renderer = _text(self.renderer_id, "renderer_id")
        object.__setattr__(self, "renderer_id", renderer)
        inverse = _text(
            self.inverse_or_compensation_ref,
            "inverse_or_compensation_ref",
        )
        object.__setattr__(self, "inverse_or_compensation_ref", inverse)
        if not _bool(self.idempotent, "idempotent"):
            raise RepairOperatorRegistryError("registered operators must be idempotent")
        object.__setattr__(self, "idempotent", True)
        if not _bool(self.proposal_only, "proposal_only"):
            raise RepairOperatorRegistryAuthorityError(
                "repair operators must remain proposal-only"
            )
        object.__setattr__(self, "proposal_only", True)
        authority_claims = {
            "semantic_authority": self.semantic_authority,
            "grants_proof_authority": self.grants_proof_authority,
            "grants_write_authority": self.grants_write_authority,
        }
        if any(value is not False for value in authority_claims.values()):
            raise RepairOperatorRegistryAuthorityError(
                "operator lookup cannot grant semantic, proof, or write authority"
            )
        object.__setattr__(self, "semantic_authority", False)
        object.__setattr__(self, "grants_proof_authority", False)
        object.__setattr__(self, "grants_write_authority", False)
        required_base = {
            RepairOperatorCapability.EXACT_TARGET.value,
            RepairOperatorCapability.EXACT_PLACEMENT.value,
            RepairOperatorCapability.CLOSED_AST.value,
            RepairOperatorCapability.IDEMPOTENT_RENDER.value,
            RepairOperatorCapability.SCOPE_BOUND.value,
            RepairOperatorCapability.PROPOSAL_ONLY.value,
        }
        if not required_base.issubset(set(self.capability_refs)):
            raise RepairOperatorRegistryError(
                "operator is missing canonical/scope/idempotency capabilities"
            )

    @property
    def spec_id(self) -> str:
        return self.content_id

    @property
    def requires_value(self) -> bool:
        return self.value_requirement is not OperatorValueRequirement.NONE

    @property
    def is_reviewed_hook(self) -> bool:
        return self.reviewed_hook is not ReviewedRepairHook.NONE

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": DOCTOR_REPAIR_OPERATOR_SPEC_VERSION,
            "interface": self.INTERFACE,
            "operator_id": self.operator_id,
            "kind": self.kind.value,
            "family": self.family.value,
            "aliases": list(self.aliases),
            "supported_languages": list(self.supported_languages),
            "supported_ast_shapes": list(self.supported_ast_shapes),
            "capability_refs": list(self.capability_refs),
            "precondition_refs": list(self.precondition_refs),
            "postcondition_refs": list(self.postcondition_refs),
            "frame_condition_refs": list(self.frame_condition_refs),
            "proof_requirement_refs": list(self.proof_requirement_refs),
            "validation_requirement_refs": list(self.validation_requirement_refs),
            "scope_constraints": list(self.scope_constraints),
            "approval_classes": list(self.approval_classes),
            "abstain_classes": list(self.abstain_classes),
            "value_requirement": self.value_requirement.value,
            "placement_required": self.placement_required,
            "analytical_transform_kind": self.analytical_transform_kind,
            "reviewed_hook": self.reviewed_hook.value,
            "review_requirement_refs": list(self.review_requirement_refs),
            "renderer_id": self.renderer_id,
            "idempotent": True,
            "inverse_or_compensation_ref": self.inverse_or_compensation_ref,
            "proposal_only": True,
            "semantic_authority": False,
            "grants_proof_authority": False,
            "grants_write_authority": False,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DoctorRepairOperatorSpec":
        if not isinstance(payload, Mapping):
            raise RepairOperatorRegistryError("operator spec must be an object")
        field_names = set(cls.__dataclass_fields__) - {"SCHEMA", "INTERFACE"}
        allowed = {
            "schema",
            "content_id",
            "contract_version",
            "interface",
            *field_names,
        }
        if set(payload) - allowed:
            raise RepairOperatorRegistryError("operator spec contains unknown fields")
        if payload.get("schema", cls.SCHEMA) != cls.SCHEMA:
            raise RepairOperatorRegistryError("unsupported operator spec schema")
        if payload.get(
            "contract_version",
            DOCTOR_REPAIR_OPERATOR_SPEC_VERSION,
        ) != DOCTOR_REPAIR_OPERATOR_SPEC_VERSION:
            raise RepairOperatorRegistryError("unsupported operator spec version")
        if payload.get("interface", cls.INTERFACE) != cls.INTERFACE:
            raise RepairOperatorRegistryError("unsupported operator spec interface")
        values = {
            name: payload[name]
            for name in field_names
            if name in payload
        }
        result = cls(**values)
        supplied = payload.get("content_id")
        if supplied not in (None, "", result.content_id):
            raise RepairOperatorRegistryError("operator spec content_id mismatch")
        return result


@dataclass(frozen=True)
class RepairOperatorLookupRequest(CanonicalContract):
    """Exact, body-free facts used to nominate an operator."""

    SCHEMA: ClassVar[str] = REPAIR_OPERATOR_LOOKUP_REQUEST_SCHEMA

    operator_kind: str
    repository_id: str
    tree_id: str
    target_paths: tuple[str, ...]
    placement_refs: tuple[str, ...]
    value_refs: tuple[str, ...] = ()
    capability_refs: tuple[str, ...] = ()
    proof_refs: tuple[str, ...] = ()
    review_refs: tuple[str, ...] = ()
    behavior_classes: tuple[str, ...] = (RepairBehaviorClass.PURE_LOCAL.value,)
    dependency_paths: tuple[str, ...] = ()
    requested_write_paths: tuple[str, ...] = ()
    language: str = "python"
    ast_shape: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "operator_kind",
            _text(self.operator_kind, "operator_kind"),
        )
        object.__setattr__(
            self,
            "repository_id",
            _optional_text(self.repository_id, "repository_id"),
        )
        object.__setattr__(self, "tree_id", _optional_text(self.tree_id, "tree_id"))
        object.__setattr__(self, "target_paths", _paths(self.target_paths, "target_paths"))
        object.__setattr__(
            self,
            "requested_write_paths",
            _paths(self.requested_write_paths, "requested_write_paths"),
        )
        object.__setattr__(
            self,
            "dependency_paths",
            _paths(self.dependency_paths, "dependency_paths"),
        )
        for name in (
            "placement_refs",
            "value_refs",
            "capability_refs",
            "proof_refs",
            "review_refs",
            "behavior_classes",
        ):
            object.__setattr__(self, name, _ids(getattr(self, name), name))
        object.__setattr__(self, "language", _text(self.language, "language"))
        object.__setattr__(self, "ast_shape", _optional_text(self.ast_shape, "ast_shape"))

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": REPAIR_OPERATOR_REGISTRY_VERSION,
            "operator_kind": self.operator_kind,
            "repository_id": self.repository_id,
            "tree_id": self.tree_id,
            "target_paths": list(self.target_paths),
            "placement_refs": list(self.placement_refs),
            "value_refs": list(self.value_refs),
            "capability_refs": list(self.capability_refs),
            "proof_refs": list(self.proof_refs),
            "review_refs": list(self.review_refs),
            "behavior_classes": list(self.behavior_classes),
            "dependency_paths": list(self.dependency_paths),
            "requested_write_paths": list(self.requested_write_paths),
            "language": self.language,
            "ast_shape": self.ast_shape,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RepairOperatorLookupRequest":
        if not isinstance(payload, Mapping):
            raise RepairOperatorRegistryError("lookup request must be an object")
        field_names = set(cls.__dataclass_fields__) - {"SCHEMA"}
        allowed = {"schema", "content_id", "contract_version", *field_names}
        if set(payload) - allowed:
            raise RepairOperatorRegistryError("lookup request contains unknown fields")
        if payload.get("schema", cls.SCHEMA) != cls.SCHEMA:
            raise RepairOperatorRegistryError("unsupported lookup request schema")
        if payload.get(
            "contract_version",
            REPAIR_OPERATOR_REGISTRY_VERSION,
        ) != REPAIR_OPERATOR_REGISTRY_VERSION:
            raise RepairOperatorRegistryError("unsupported lookup request version")
        result = cls(
            **{
                name: payload[name]
                for name in field_names
                if name in payload
            }
        )
        supplied = payload.get("content_id")
        if supplied not in (None, "", result.content_id):
            raise RepairOperatorRegistryError("lookup request content_id mismatch")
        return result


@dataclass(frozen=True)
class RepairOperatorLookupResult(CanonicalContract):
    """Body-free nomination result which explicitly carries no authority."""

    SCHEMA: ClassVar[str] = REPAIR_OPERATOR_LOOKUP_RESULT_SCHEMA

    request_id: str
    operator_kind: str
    operator_id: str
    spec_id: str
    disposition: RepairOperatorLookupDisposition
    reason_codes: tuple[str, ...]
    matched_capability_refs: tuple[str, ...] = ()
    proof_verification_required: bool = True
    approval_validation_required: bool = False
    proposal_only: bool = True
    semantic_authority: bool = False
    grants_proof_authority: bool = False
    grants_write_authority: bool = False

    def __post_init__(self) -> None:
        for name in ("request_id", "operator_kind"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        for name in ("operator_id", "spec_id"):
            object.__setattr__(self, name, _optional_text(getattr(self, name), name))
        disposition = _enum(
            self.disposition,
            RepairOperatorLookupDisposition,
            "disposition",
        )
        object.__setattr__(self, "disposition", disposition)
        object.__setattr__(
            self,
            "reason_codes",
            _ids(self.reason_codes, "reason_codes", required=True),
        )
        object.__setattr__(
            self,
            "matched_capability_refs",
            _ids(self.matched_capability_refs, "matched_capability_refs"),
        )
        object.__setattr__(
            self,
            "proof_verification_required",
            _bool(self.proof_verification_required, "proof_verification_required"),
        )
        object.__setattr__(
            self,
            "approval_validation_required",
            _bool(self.approval_validation_required, "approval_validation_required"),
        )
        if self.proposal_only is not True or any(
            value is not False
            for value in (
                self.semantic_authority,
                self.grants_proof_authority,
                self.grants_write_authority,
            )
        ):
            raise RepairOperatorRegistryAuthorityError(
                "lookup results are proposal-only and carry no authority"
            )
        object.__setattr__(self, "proposal_only", True)
        object.__setattr__(self, "semantic_authority", False)
        object.__setattr__(self, "grants_proof_authority", False)
        object.__setattr__(self, "grants_write_authority", False)

    @property
    def proposal_eligible(self) -> bool:
        return self.disposition is RepairOperatorLookupDisposition.PROPOSAL_ELIGIBLE

    @property
    def requires_approval(self) -> bool:
        return self.disposition is RepairOperatorLookupDisposition.APPROVAL_REQUIRED

    @property
    def admitted(self) -> bool:
        """Registry nomination is never proof/write admission."""

        return False

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": REPAIR_OPERATOR_REGISTRY_VERSION,
            "request_id": self.request_id,
            "operator_kind": self.operator_kind,
            "operator_id": self.operator_id,
            "spec_id": self.spec_id,
            "disposition": self.disposition.value,
            "reason_codes": list(self.reason_codes),
            "matched_capability_refs": list(self.matched_capability_refs),
            "proof_verification_required": self.proof_verification_required,
            "approval_validation_required": self.approval_validation_required,
            "proposal_only": True,
            "semantic_authority": False,
            "grants_proof_authority": False,
            "grants_write_authority": False,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RepairOperatorLookupResult":
        if not isinstance(payload, Mapping):
            raise RepairOperatorRegistryError("lookup result must be an object")
        field_names = set(cls.__dataclass_fields__) - {"SCHEMA"}
        allowed = {"schema", "content_id", "contract_version", *field_names}
        if set(payload) - allowed:
            raise RepairOperatorRegistryError("lookup result contains unknown fields")
        if payload.get("schema", cls.SCHEMA) != cls.SCHEMA:
            raise RepairOperatorRegistryError("unsupported lookup result schema")
        if payload.get(
            "contract_version",
            REPAIR_OPERATOR_REGISTRY_VERSION,
        ) != REPAIR_OPERATOR_REGISTRY_VERSION:
            raise RepairOperatorRegistryError("unsupported lookup result version")
        result = cls(
            **{
                name: payload[name]
                for name in field_names
                if name in payload
            }
        )
        supplied = payload.get("content_id")
        if supplied not in (None, "", result.content_id):
            raise RepairOperatorRegistryError("lookup result content_id mismatch")
        return result


@dataclass(frozen=True)
class RepairOperatorRegistry(CanonicalContract):
    """Immutable closed catalogue of v2 reviewed repair operators."""

    SCHEMA: ClassVar[str] = REPAIR_OPERATOR_REGISTRY_SCHEMA
    INTERFACE: ClassVar[str] = REPAIR_OPERATOR_REGISTRY_INTERFACE

    operators: tuple[DoctorRepairOperatorSpec, ...]
    registry_id: str = ""
    producer_id: str = REPAIR_OPERATOR_REGISTRY_PRODUCER

    def __post_init__(self) -> None:
        if not self.operators or len(self.operators) > MAX_OPERATOR_COUNT:
            raise RepairOperatorRegistryError("registry operator count is out of bounds")
        if not all(isinstance(item, DoctorRepairOperatorSpec) for item in self.operators):
            raise RepairOperatorRegistryError(
                "operators must contain DoctorRepairOperatorSpec values"
            )
        ordered = tuple(sorted(self.operators, key=lambda item: item.operator_id))
        ids = [item.operator_id for item in ordered]
        kinds = [item.kind for item in ordered]
        if len(ids) != len(set(ids)) or len(kinds) != len(set(kinds)):
            raise RepairOperatorRegistryError("operator ids and kinds must be unique")
        alias_owner: dict[str, RepairOperatorKind] = {}
        for item in ordered:
            for alias in (item.kind.value, item.operator_id, *item.aliases):
                normalized = alias.strip().lower().replace("-", "_")
                owner = alias_owner.get(normalized)
                if owner is not None and owner is not item.kind:
                    raise RepairOperatorRegistryError(
                        "operator aliases must resolve uniquely"
                    )
                alias_owner[normalized] = item.kind
        object.__setattr__(self, "operators", ordered)
        object.__setattr__(self, "producer_id", _text(self.producer_id, "producer_id"))
        calculated = content_identity(self._payload_without_registry_id())
        supplied = _optional_text(self.registry_id, "registry_id")
        if supplied and supplied != calculated:
            raise RepairOperatorRegistryError("registry_id mismatch")
        object.__setattr__(self, "registry_id", calculated)
        self._validate_coverage()

    def _validate_coverage(self) -> None:
        registered = {item.kind.value for item in self.operators}
        analytical = {
            repair_kind
            for repair_kinds in ANALYTICAL_TRANSFORM_OPERATOR_BINDINGS.values()
            for repair_kind in repair_kinds
        }
        missing = analytical - registered
        if missing:
            raise RepairOperatorRegistryError(
                "registry omits analytical transforms: " + ", ".join(sorted(missing))
            )
        required = {
            RepairOperatorKind.EXACT_RENAME.value,
            RepairOperatorKind.EXACT_MOVE.value,
            RepairOperatorKind.RESTORE_TRACKED_ARTIFACT.value,
            RepairOperatorKind.SEMANTIC_PATCH.value,
            RepairOperatorKind.EQUALITY_REWRITE.value,
        }
        if required - registered:
            raise RepairOperatorRegistryError("registry omits required reviewed operators")

    def _payload_without_registry_id(self) -> dict[str, Any]:
        return {
            "contract_version": REPAIR_OPERATOR_REGISTRY_VERSION,
            "interface": self.INTERFACE,
            "operators": [item.to_dict() for item in self.operators],
            "producer_id": self.producer_id,
            "semantic_authority": False,
            "grants_proof_authority": False,
            "grants_write_authority": False,
        }

    def _payload(self) -> dict[str, Any]:
        return {
            **self._payload_without_registry_id(),
            "registry_id": self.registry_id,
        }

    @property
    def descriptors(self) -> tuple[DoctorRepairOperatorSpec, ...]:
        return self.operators

    @property
    def semantic_authority(self) -> bool:
        return False

    @property
    def grants_proof_authority(self) -> bool:
        return False

    @property
    def grants_write_authority(self) -> bool:
        return False

    def kinds(self) -> tuple[RepairOperatorKind, ...]:
        return tuple(item.kind for item in self.operators)

    def specs(self) -> tuple[DoctorRepairOperatorSpec, ...]:
        return self.operators

    def get(self, kind_or_id: Any) -> DoctorRepairOperatorSpec:
        raw = str(getattr(kind_or_id, "value", kind_or_id)).strip()
        normalized = raw.lower().replace("-", "_")
        for item in self.operators:
            if normalized in {
                item.operator_id.lower().replace("-", "_"),
                item.kind.value,
                *(alias.lower().replace("-", "_") for alias in item.aliases),
            }:
                return item
        kind = normalize_repair_operator_kind(kind_or_id)
        for item in self.operators:
            if item.kind is kind:
                return item
        raise UnknownRepairOperatorError(
            RepairOperatorLookupReason.UNKNOWN_OPERATOR.value
        )

    def lookup(self, kind_or_id: Any) -> DoctorRepairOperatorSpec:
        """Return immutable metadata only; no proof or mutation authority."""

        return self.get(kind_or_id)

    def resolve(
        self,
        request: RepairOperatorLookupRequest,
    ) -> RepairOperatorLookupResult:
        """Resolve exact lookup facts to a proposal-only disposition."""

        if not isinstance(request, RepairOperatorLookupRequest):
            raise RepairOperatorRegistryError(
                "request must be RepairOperatorLookupRequest"
            )
        try:
            spec = self.get(request.operator_kind)
        except UnknownRepairOperatorError:
            return self._result(
                request,
                None,
                RepairOperatorLookupDisposition.ABSTAINED,
                (RepairOperatorLookupReason.UNKNOWN_OPERATOR,),
            )

        if len(request.target_paths) > 1:
            return self._result(
                request,
                spec,
                RepairOperatorLookupDisposition.REJECTED,
                (RepairOperatorLookupReason.TARGET_AMBIGUOUS,),
            )
        if len(request.value_refs) > 1:
            return self._result(
                request,
                spec,
                RepairOperatorLookupDisposition.REJECTED,
                (RepairOperatorLookupReason.VALUE_AMBIGUOUS,),
            )
        if len(request.placement_refs) > 1:
            return self._result(
                request,
                spec,
                RepairOperatorLookupDisposition.REJECTED,
                (RepairOperatorLookupReason.PLACEMENT_AMBIGUOUS,),
            )

        abstain: list[RepairOperatorLookupReason] = []
        if not request.target_paths:
            abstain.append(RepairOperatorLookupReason.TARGET_MISSING)
        if not request.repository_id or not request.tree_id:
            abstain.append(RepairOperatorLookupReason.SCOPE_MISSING)
        if spec.requires_value and not request.value_refs:
            abstain.append(RepairOperatorLookupReason.VALUE_MISSING)
        if spec.placement_required and not request.placement_refs:
            abstain.append(RepairOperatorLookupReason.PLACEMENT_MISSING)
        if request.language not in spec.supported_languages:
            abstain.append(RepairOperatorLookupReason.UNSUPPORTED_LANGUAGE)
        if request.ast_shape and request.ast_shape not in spec.supported_ast_shapes:
            abstain.append(RepairOperatorLookupReason.UNSUPPORTED_AST_SHAPE)
        missing_capabilities = set(spec.capability_refs) - set(request.capability_refs)
        if missing_capabilities:
            abstain.append(RepairOperatorLookupReason.CAPABILITY_MISSING)
        if not request.proof_refs:
            abstain.append(RepairOperatorLookupReason.PROOF_REFERENCE_MISSING)
        if spec.review_requirement_refs and not request.review_refs:
            abstain.append(RepairOperatorLookupReason.REVIEW_REFERENCE_MISSING)
        if request.requested_write_paths and (
            len(request.requested_write_paths) != 1
            or request.requested_write_paths != request.target_paths
        ):
            abstain.append(RepairOperatorLookupReason.SCOPE_ESCAPE)

        behavior = set(request.behavior_classes)
        known_behavior = {item.value for item in RepairBehaviorClass}
        if (
            not behavior
            or RepairBehaviorClass.UNKNOWN.value in behavior
            or behavior - known_behavior
        ):
            abstain.append(RepairOperatorLookupReason.UNKNOWN_BEHAVIOR)
        if abstain:
            return self._result(
                request,
                spec,
                RepairOperatorLookupDisposition.ABSTAINED,
                tuple(dict.fromkeys(abstain)),
            )

        approval: list[RepairOperatorLookupReason] = []
        for behavior_class, reason in _APPROVAL_BEHAVIORS.items():
            if behavior_class in behavior and behavior_class in spec.approval_classes:
                approval.append(reason)
        if request.dependency_paths:
            approval.append(RepairOperatorLookupReason.DEPENDENCY_APPROVAL)
        if approval:
            return self._result(
                request,
                spec,
                RepairOperatorLookupDisposition.APPROVAL_REQUIRED,
                tuple(dict.fromkeys(approval)),
                approval_validation_required=True,
            )

        return self._result(
            request,
            spec,
            RepairOperatorLookupDisposition.PROPOSAL_ELIGIBLE,
            (RepairOperatorLookupReason.CANDIDATE_ONLY,),
        )

    evaluate_lookup = resolve

    def _result(
        self,
        request: RepairOperatorLookupRequest,
        spec: DoctorRepairOperatorSpec | None,
        disposition: RepairOperatorLookupDisposition,
        reasons: tuple[RepairOperatorLookupReason, ...],
        *,
        approval_validation_required: bool = False,
    ) -> RepairOperatorLookupResult:
        return RepairOperatorLookupResult(
            request_id=request.content_id,
            operator_kind=spec.kind.value if spec is not None else request.operator_kind,
            operator_id=spec.operator_id if spec is not None else "",
            spec_id=spec.spec_id if spec is not None else "",
            disposition=disposition,
            reason_codes=tuple(item.value for item in reasons),
            matched_capability_refs=(
                tuple(
                    sorted(
                        set(request.capability_refs).intersection(spec.capability_refs)
                    )
                )
                if spec is not None
                else ()
            ),
            proof_verification_required=True,
            approval_validation_required=approval_validation_required,
            proposal_only=True,
            semantic_authority=False,
            grants_proof_authority=False,
            grants_write_authority=False,
        )

    def build_legacy_registry(
        self,
        roots: DoctorAuthorityRoots,
    ) -> LegacyDoctorRepairOperatorRegistry:
        """Build the existing root-bound renderer registry.

        This adapter deliberately delegates to the legacy factory instead of
        converting v2 lookup results.  Consequently a v2 lookup cannot smuggle
        a proof verdict, path permit, or write capability into rendering.
        """

        if not isinstance(roots, DoctorAuthorityRoots):
            raise RepairOperatorRegistryError("roots must be DoctorAuthorityRoots")
        return build_default_doctor_operator_registry(roots)

    def legacy_kind(self, kind_or_id: Any) -> DoctorOperatorKind | None:
        """Return a legacy renderer kind when one exists, otherwise ``None``."""

        kind = self.get(kind_or_id).kind
        mapping = {
            RepairOperatorKind.EXACT_RENAME: DoctorOperatorKind.EXACT_RENAME,
            RepairOperatorKind.ADD_ARGUMENT: DoctorOperatorKind.ADD_ARGUMENT,
            RepairOperatorKind.RENAME_ARGUMENT: DoctorOperatorKind.RENAME_ARGUMENT,
            RepairOperatorKind.REORDER_ARGUMENT: DoctorOperatorKind.REORDER_ARGUMENT,
            RepairOperatorKind.THREAD_ARGUMENT: DoctorOperatorKind.THREAD_ARGUMENT,
            RepairOperatorKind.ADD_IMPORT: DoctorOperatorKind.ADD_IMPORT,
            RepairOperatorKind.ADD_EXPORT: DoctorOperatorKind.ADD_EXPORT,
            RepairOperatorKind.ADD_REGISTRATION: DoctorOperatorKind.ADD_REGISTRATION,
            RepairOperatorKind.ADD_CONSTRUCTOR_ROUTE: (
                DoctorOperatorKind.ADD_CONSTRUCTOR_ROUTE
            ),
            RepairOperatorKind.ADD_FACTORY_ROUTE: DoctorOperatorKind.ADD_FACTORY_ROUTE,
            RepairOperatorKind.FINITE_ADAPTER: DoctorOperatorKind.FINITE_ADAPTER,
            RepairOperatorKind.SCHEMA_PROJECTION: DoctorOperatorKind.SCHEMA_PROJECTION,
            RepairOperatorKind.RESTORE_TRACKED_ARTIFACT: (
                DoctorOperatorKind.RESTORE_TRACKED_ARTIFACT
            ),
        }
        return mapping.get(kind)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RepairOperatorRegistry":
        if not isinstance(payload, Mapping):
            raise RepairOperatorRegistryError("registry must be an object")
        allowed = {
            "schema",
            "content_id",
            "contract_version",
            "interface",
            "operators",
            "registry_id",
            "producer_id",
            "semantic_authority",
            "grants_proof_authority",
            "grants_write_authority",
        }
        if set(payload) - allowed:
            raise RepairOperatorRegistryError("registry contains unknown fields")
        if payload.get("schema", cls.SCHEMA) != cls.SCHEMA:
            raise RepairOperatorRegistryError("unsupported registry schema")
        if payload.get(
            "contract_version",
            REPAIR_OPERATOR_REGISTRY_VERSION,
        ) != REPAIR_OPERATOR_REGISTRY_VERSION:
            raise RepairOperatorRegistryError("unsupported registry version")
        if payload.get("interface", cls.INTERFACE) != cls.INTERFACE:
            raise RepairOperatorRegistryError("unsupported registry interface")
        for authority in (
            "semantic_authority",
            "grants_proof_authority",
            "grants_write_authority",
        ):
            if payload.get(authority, False) is not False:
                raise RepairOperatorRegistryAuthorityError(
                    "serialized registry cannot claim authority"
                )
        raw_operators = payload.get("operators")
        if isinstance(raw_operators, (str, bytes, bytearray)) or not isinstance(
            raw_operators,
            Sequence,
        ):
            raise RepairOperatorRegistryError("registry operators must be a sequence")
        result = cls(
            operators=tuple(
                item
                if isinstance(item, DoctorRepairOperatorSpec)
                else DoctorRepairOperatorSpec.from_dict(item)
                for item in raw_operators
            ),
            registry_id=payload.get("registry_id", ""),
            producer_id=payload.get(
                "producer_id",
                REPAIR_OPERATOR_REGISTRY_PRODUCER,
            ),
        )
        supplied = payload.get("content_id")
        if supplied not in (None, "", result.content_id):
            raise RepairOperatorRegistryError("registry content_id mismatch")
        return result


_BASE_CAPABILITIES: Final[tuple[str, ...]] = tuple(
    item.value
    for item in (
        RepairOperatorCapability.EXACT_TARGET,
        RepairOperatorCapability.EXACT_PLACEMENT,
        RepairOperatorCapability.CLOSED_AST,
        RepairOperatorCapability.IDEMPOTENT_RENDER,
        RepairOperatorCapability.SCOPE_BOUND,
        RepairOperatorCapability.PROPOSAL_ONLY,
    )
)

_COMMON_SCOPE: Final[tuple[str, ...]] = (
    "scope:exact_repository",
    "scope:exact_tree",
    "scope:single_target_path",
    "scope:exact_before_hash",
    "scope:closed_impact",
)


def _spec(
    kind: RepairOperatorKind,
    family: RepairOperatorFamily,
    *,
    aliases: tuple[str, ...],
    shapes: tuple[str, ...],
    capability: RepairOperatorCapability,
    analytical: TransformKind | None = None,
    hook: ReviewedRepairHook = ReviewedRepairHook.NONE,
    value: OperatorValueRequirement = OperatorValueRequirement.NONE,
    review: tuple[str, ...] = (),
    languages: tuple[str, ...] = ("python",),
    pre: tuple[str, ...] = (),
    post: tuple[str, ...] = (),
    inverse: str,
) -> DoctorRepairOperatorSpec:
    renderer = (
        "AnalyticalChangeTransformer@1"
        if analytical is not None
        else f"reviewed-repair-hook:{hook.value}@1"
    )
    return DoctorRepairOperatorSpec(
        operator_id=f"repair-operator:{kind.value}@2",
        kind=kind,
        family=family,
        aliases=aliases,
        supported_languages=languages,
        supported_ast_shapes=shapes,
        capability_refs=(*_BASE_CAPABILITIES, capability.value),
        precondition_refs=pre
        or ("pre:unique_target", "pre:exact_placement", "pre:closed_scope"),
        postcondition_refs=post or (f"post:{kind.value}",),
        frame_condition_refs=(
            "frame:non_target_bytes_unchanged",
            "frame:dependencies_unchanged",
            "frame:public_api_unchanged",
        ),
        proof_requirement_refs=(
            f"proof:{kind.value}",
            "proof:current_roots",
            "proof:scope_closed",
        ),
        validation_requirement_refs=(
            "validation:parse",
            "validation:type_or_schema",
            "validation:impact_selected",
            "validation:fixed_point",
        ),
        scope_constraints=_COMMON_SCOPE,
        approval_classes=tuple(sorted(_APPROVAL_BEHAVIORS)),
        abstain_classes=(RepairBehaviorClass.UNKNOWN.value,),
        value_requirement=value,
        placement_required=True,
        analytical_transform_kind=analytical.value if analytical is not None else "",
        reviewed_hook=hook,
        review_requirement_refs=review,
        renderer_id=renderer,
        idempotent=True,
        inverse_or_compensation_ref=inverse,
        proposal_only=True,
        semantic_authority=False,
        grants_proof_authority=False,
        grants_write_authority=False,
    )


def _default_specs() -> tuple[DoctorRepairOperatorSpec, ...]:
    """Build the exhaustive reviewed operator catalogue."""

    return (
        _spec(
            RepairOperatorKind.EXACT_RENAME,
            RepairOperatorFamily.SYMBOL,
            aliases=("rename", "symbol_rename"),
            shapes=("identifier", "name", "attribute", "definition"),
            capability=RepairOperatorCapability.SYMBOL_EQUIVALENCE,
            analytical=TransformKind.RENAME_ARGUMENT,
            pre=("pre:unique_symbol", "pre:referent_equivalence", "pre:closed_callers"),
            post=("post:symbol_renamed", "post:referents_preserved"),
            inverse="compensation:exact_rename_inverse",
        ),
        _spec(
            RepairOperatorKind.EXACT_MOVE,
            RepairOperatorFamily.MOVE,
            aliases=("move", "file_move"),
            shapes=("module", "tracked_file", "package_member"),
            capability=RepairOperatorCapability.FILE_MOVE,
            hook=ReviewedRepairHook.EXACT_MOVE,
            review=("review:exact_move_patch@1",),
            pre=("pre:unique_source", "pre:unique_destination", "pre:closed_importers"),
            post=("post:artifact_moved", "post:importers_retargeted"),
            inverse="compensation:exact_move_inverse",
        ),
        _spec(
            RepairOperatorKind.ADD_ARGUMENT,
            RepairOperatorFamily.CALL,
            aliases=("missing_argument",),
            shapes=("call", "keyword_argument", "positional_argument"),
            capability=RepairOperatorCapability.UNIQUE_VALUE,
            analytical=TransformKind.ADD_ARGUMENT,
            value=OperatorValueRequirement.UNIQUE_PROVED,
            inverse="compensation:remove_argument",
        ),
        _spec(
            RepairOperatorKind.RENAME_ARGUMENT,
            RepairOperatorFamily.CALL,
            aliases=("keyword_rename", "parameter_rename"),
            shapes=("call", "function_parameter", "keyword_argument"),
            capability=RepairOperatorCapability.SYMBOL_EQUIVALENCE,
            analytical=TransformKind.RENAME_ARGUMENT,
            value=OperatorValueRequirement.UNIQUE_PROVED,
            inverse="compensation:rename_argument_inverse",
        ),
        _spec(
            RepairOperatorKind.REORDER_ARGUMENT,
            RepairOperatorFamily.CALL,
            aliases=("argument_reorder",),
            shapes=("call", "keyword_argument_list"),
            capability=RepairOperatorCapability.UNIQUE_VALUE,
            analytical=TransformKind.REORDER_ARGUMENT,
            value=OperatorValueRequirement.UNIQUE_PROVED,
            inverse="compensation:reorder_argument_inverse",
        ),
        _spec(
            RepairOperatorKind.THREAD_ARGUMENT,
            RepairOperatorFamily.CALL,
            aliases=("value_threading", "thread_value"),
            shapes=("call_route", "function_parameter", "call"),
            capability=RepairOperatorCapability.ROUTE_CLOSURE,
            analytical=TransformKind.THREAD_PARAMETER,
            value=OperatorValueRequirement.UNIQUE_PROVED,
            pre=("pre:unique_value", "pre:complete_route", "pre:finite_hops"),
            inverse="compensation:unthread_argument",
        ),
        _spec(
            RepairOperatorKind.ADD_IMPORT,
            RepairOperatorFamily.WIRING,
            aliases=("exact_import", "import"),
            shapes=("module", "import_block", "import_from"),
            capability=RepairOperatorCapability.IMPORT_WIRING,
            analytical=TransformKind.ADD_IMPORT,
            inverse="compensation:remove_import",
        ),
        _spec(
            RepairOperatorKind.ADD_EXPORT,
            RepairOperatorFamily.WIRING,
            aliases=("exact_export", "export"),
            shapes=("module", "dunder_all", "export_list"),
            capability=RepairOperatorCapability.EXPORT_WIRING,
            analytical=TransformKind.ADD_EXPORT,
            inverse="compensation:remove_export",
        ),
        _spec(
            RepairOperatorKind.ADD_REGISTRATION,
            RepairOperatorFamily.WIRING,
            aliases=("exact_registration", "registration"),
            shapes=("module", "registration_call", "registry_literal"),
            capability=RepairOperatorCapability.REGISTRATION_WIRING,
            analytical=TransformKind.ADD_REGISTRATION,
            inverse="compensation:remove_registration",
        ),
        _spec(
            RepairOperatorKind.ADD_CONSTRUCTOR_ROUTE,
            RepairOperatorFamily.CONSTRUCTION,
            aliases=("constructor", "constructor_route"),
            shapes=("constructor_call", "class_instantiation"),
            capability=RepairOperatorCapability.CONSTRUCTOR_WIRING,
            analytical=TransformKind.UPDATE_CONSTRUCTOR,
            value=OperatorValueRequirement.UNIQUE_PROVED,
            inverse="compensation:remove_constructor_route",
        ),
        _spec(
            RepairOperatorKind.ADD_FACTORY_ROUTE,
            RepairOperatorFamily.CONSTRUCTION,
            aliases=("factory", "factory_route"),
            shapes=("factory_call",),
            capability=RepairOperatorCapability.FACTORY_WIRING,
            analytical=TransformKind.UPDATE_CONSTRUCTOR,
            value=OperatorValueRequirement.UNIQUE_PROVED,
            inverse="compensation:remove_factory_route",
        ),
        _spec(
            RepairOperatorKind.FINITE_ADAPTER,
            RepairOperatorFamily.CONSTRUCTION,
            aliases=("adapter", "finite_mapping_adapter"),
            shapes=("adapter_wrap", "simple_expression"),
            capability=RepairOperatorCapability.FINITE_ADAPTER,
            analytical=TransformKind.ADD_ADAPTER,
            value=OperatorValueRequirement.UNIQUE_PROVED,
            inverse="compensation:unwrap_adapter",
        ),
        _spec(
            RepairOperatorKind.SCHEMA_PROJECTION,
            RepairOperatorFamily.DATA_CONTRACT,
            aliases=("schema", "schema_update"),
            shapes=("schema_object", "schema_literal", "json_object"),
            capability=RepairOperatorCapability.TOTAL_FIELD_MAPPING,
            analytical=TransformKind.UPDATE_SCHEMA_FIELD,
            value=OperatorValueRequirement.TOTAL_MAPPING,
            languages=("python", "json", "yaml"),
            inverse="compensation:schema_projection_inverse",
        ),
        _spec(
            RepairOperatorKind.SERIALIZER_UPDATE,
            RepairOperatorFamily.DATA_CONTRACT,
            aliases=("serializer",),
            shapes=("serializer_mapping", "serializer_literal", "json_object"),
            capability=RepairOperatorCapability.SERIALIZER_MAPPING,
            analytical=TransformKind.UPDATE_SERIALIZER,
            value=OperatorValueRequirement.TOTAL_MAPPING,
            languages=("python", "json"),
            inverse="compensation:serializer_update_inverse",
        ),
        _spec(
            RepairOperatorKind.FIXTURE_UPDATE,
            RepairOperatorFamily.DATA_CONTRACT,
            aliases=("fixture",),
            shapes=("fixture_literal", "json_object", "mapping_literal"),
            capability=RepairOperatorCapability.FIXTURE_MAPPING,
            analytical=TransformKind.UPDATE_FIXTURE,
            value=OperatorValueRequirement.TOTAL_MAPPING,
            languages=("python", "json", "yaml"),
            inverse="compensation:fixture_update_inverse",
        ),
        _spec(
            RepairOperatorKind.MANIFEST_UPDATE,
            RepairOperatorFamily.DATA_CONTRACT,
            aliases=("manifest", "generated_manifest"),
            shapes=("manifest_object", "json_object", "mapping_literal"),
            capability=RepairOperatorCapability.MANIFEST_MAPPING,
            analytical=TransformKind.UPDATE_GENERATED_MANIFEST,
            value=OperatorValueRequirement.TOTAL_MAPPING,
            languages=("python", "json", "toml", "yaml"),
            inverse="compensation:manifest_update_inverse",
        ),
        _spec(
            RepairOperatorKind.RESTORE_TRACKED_ARTIFACT,
            RepairOperatorFamily.ARTIFACT,
            aliases=("artifact", "restore_artifact"),
            shapes=("whole_file", "tracked_blob"),
            capability=RepairOperatorCapability.VERIFIED_ARTIFACT,
            hook=ReviewedRepairHook.ARTIFACT_RESTORE,
            value=OperatorValueRequirement.VERIFIED_PREIMAGE,
            languages=("binary", "json", "python", "text"),
            pre=("pre:tracked_path", "pre:verified_cid", "pre:canonical_preimage"),
            post=("post:artifact_restored", "post:cid_matches"),
            inverse="compensation:restore_previous_cid",
        ),
        _spec(
            RepairOperatorKind.SEMANTIC_PATCH,
            RepairOperatorFamily.REVIEWED_REWRITE,
            aliases=("reviewed_semantic_patch",),
            shapes=("reviewed_pattern_match", "closed_ast_capture"),
            capability=RepairOperatorCapability.REVIEWED_SEMANTIC_PATCH,
            hook=ReviewedRepairHook.SEMANTIC_PATCH,
            value=OperatorValueRequirement.REVIEWED_RULE,
            review=("review:semantic_patch_template@1",),
            pre=("pre:reviewed_patch", "pre:unique_capture", "pre:closed_metavariables"),
            post=("post:reviewed_patch_postcondition",),
            inverse="compensation:semantic_patch_inverse",
        ),
        _spec(
            RepairOperatorKind.EQUALITY_REWRITE,
            RepairOperatorFamily.REVIEWED_REWRITE,
            aliases=("equality", "egraph_rewrite"),
            shapes=("expression", "egraph_term", "closed_equation"),
            capability=RepairOperatorCapability.DECLARED_EQUALITY_THEORY,
            hook=ReviewedRepairHook.EQUALITY_REWRITE,
            value=OperatorValueRequirement.REVIEWED_RULE,
            review=("review:equality_theory@1", "review:equality_rewrite@1"),
            pre=("pre:declared_theory", "pre:oriented_rule", "pre:equivalence_proof"),
            post=("post:equivalent_under_declared_theory",),
            inverse="compensation:equality_rewrite_inverse",
        ),
    )


def build_default_repair_operator_registry(
    roots: DoctorAuthorityRoots | None = None,
) -> RepairOperatorRegistry:
    """Return the canonical reviewed v2 registry.

    ``roots`` is accepted for composition compatibility and type-checked, but
    is deliberately excluded from registry identity: descriptors are
    capability declarations, not root-bound authority.  Root binding occurs
    in lookup requests and the legacy rendering registry.
    """

    if roots is not None and not isinstance(roots, DoctorAuthorityRoots):
        raise RepairOperatorRegistryError("roots must be DoctorAuthorityRoots")
    return RepairOperatorRegistry(operators=_default_specs())


def default_repair_operator_registry_id(
    roots: DoctorAuthorityRoots | None = None,
) -> str:
    return build_default_repair_operator_registry(roots).registry_id


__all__ = (
    "DOCTOR_REPAIR_OPERATOR_SPEC_INTERFACE",
    "DOCTOR_REPAIR_OPERATOR_SPEC_SCHEMA",
    "DOCTOR_REPAIR_OPERATOR_SPEC_VERSION",
    "REPAIR_OPERATOR_LOOKUP_REQUEST_SCHEMA",
    "REPAIR_OPERATOR_LOOKUP_RESULT_SCHEMA",
    "REPAIR_OPERATOR_REGISTRY_INTERFACE",
    "REPAIR_OPERATOR_REGISTRY_PRODUCER",
    "REPAIR_OPERATOR_REGISTRY_SCHEMA",
    "REPAIR_OPERATOR_REGISTRY_VERSION",
    "DoctorRepairOperatorSpec",
    "OperatorValueRequirement",
    "RepairBehaviorClass",
    "RepairOperatorCapability",
    "RepairOperatorFamily",
    "RepairOperatorKind",
    "RepairOperatorLookupDisposition",
    "RepairOperatorLookupReason",
    "RepairOperatorLookupRequest",
    "RepairOperatorLookupResult",
    "RepairOperatorRegistry",
    "RepairOperatorRegistryAuthorityError",
    "RepairOperatorRegistryError",
    "ReviewedRepairHook",
    "UnknownRepairOperatorError",
    "build_default_repair_operator_registry",
    "default_repair_operator_registry_id",
    "normalize_repair_operator_kind",
)
