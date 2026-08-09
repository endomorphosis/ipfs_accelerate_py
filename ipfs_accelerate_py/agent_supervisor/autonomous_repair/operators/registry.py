"""Finite typed repair-operator registry for deterministic contract repair.

DCR-040 seals two closed interfaces:

* ``RepairOperator@1`` — one reviewed operator descriptor with a closed input
  schema, exact write scope, before/after predicates, preview, inverse,
  validations, and applicability proof.  Descriptors never carry source
  bodies, shell fragments, dynamic import targets, or write authority.
* ``RepairOperatorRegistry@1`` — an immutable, content-addressed catalogue of
  those descriptors.  Lookup and registration are fail-closed: unknown
  fields/operators and non-invertible or unbounded mutations are rejected
  before planning.

Later operator-family modules (DCR-041..047) implement the preview/inverse
bodies referenced here; this module is metadata authority only.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Final

from ...proof.formal_verification_contracts import (
    CanonicalContract,
    content_identity,
)

REPAIR_OPERATOR_INTERFACE: Final[str] = "RepairOperator@1"
REPAIR_OPERATOR_REGISTRY_INTERFACE: Final[str] = "RepairOperatorRegistry@1"
REPAIR_OPERATOR_VERSION: Final[int] = 1
REPAIR_OPERATOR_REGISTRY_VERSION: Final[int] = 1
REPAIR_OPERATOR_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-repair-operator@1"
)
REPAIR_OPERATOR_REGISTRY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-repair-operator-registry@1"
)
REPAIR_OPERATOR_REGISTRY_PRODUCER: Final[str] = (
    "deterministic-contract-repair-operator-registry@1"
)
OPERATOR_REGISTRY_EVIDENCE: Final[str] = "dcr/operator-registry@1"

MAX_OPERATOR_COUNT: Final[int] = 64
MAX_TEXT_BYTES: Final[int] = 4_096
MAX_PATH_BYTES: Final[int] = 1_024
MAX_WRITE_PATHS: Final[int] = 16
MAX_REF_COUNT: Final[int] = 32
MAX_WRITE_SCOPE_GLOBS: Final[int] = 8

# Fields that would smuggle authority, generation, or dynamic code into the
# registry.  Their presence is always a hard rejection.
_FORBIDDEN_DESCRIPTOR_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "source",
        "source_body",
        "source_text",
        "code",
        "code_body",
        "shell",
        "shell_fragment",
        "command",
        "script",
        "callable",
        "renderer",
        "dynamic_import",
        "import_path",
        "module_path",
        "exec",
        "eval",
        "llm_prompt",
        "prose",
        "patch_body",
        "diff_body",
    }
)

_UNSAFE_SCOPE_MARKERS: Final[tuple[str, ...]] = (
    "..",
    "*",
    "?",
    "[",
    "]",
    "{",
    "}",
    "\x00",
)
_OPERATOR_ID_RE: Final[re.Pattern[str]] = re.compile(
    r"^dcr-operator:[a-z][a-z0-9_]*(?:/[a-z][a-z0-9_]*)*@\d+$"
)
_KIND_RE: Final[re.Pattern[str]] = re.compile(r"^[a-z][a-z0-9_]*$")


class RepairOperatorRegistryError(ValueError):
    """Malformed operator descriptor, unknown field, or unsafe registry input."""


class OperatorFamily(str, Enum):  # noqa: UP042 - package supports Python 3.8
    """Closed family inventory for DCR-041 through DCR-047."""

    REGISTRY = "registry"
    PROTOCOL = "protocol"
    DISPATCH = "dispatch"
    TRANSPORT = "transport"
    UI = "ui"
    SECURITY = "security"
    CODEGEN = "codegen"
    ROOT = "root"


class OperatorKind(str, Enum):  # noqa: UP042 - package supports Python 3.8
    """Finite reviewed operator kinds.  Unknown values are inadmissible."""

    ADD_ALIAS = "add_alias"
    REMOVE_ALIAS = "remove_alias"
    RENAME_ALIAS = "rename_alias"
    BIND_REGISTRATION = "bind_registration"
    DISAMBIGUATE_ANCHOR = "disambiguate_anchor"
    REPAIR_JSONRPC_SCHEMA = "repair_jsonrpc_schema"
    REPAIR_REQUEST_ADAPTER = "repair_request_adapter"
    REPAIR_ERROR_ENVELOPE = "repair_error_envelope"
    REPAIR_PROFILE_BINDING = "repair_profile_binding"
    REPAIR_DISPATCH_BINDING = "repair_dispatch_binding"
    REPAIR_TRANSPORT_ADAPTER = "repair_transport_adapter"
    REPAIR_CAPABILITY_TRUTH = "repair_capability_truth"
    REPAIR_UI_PROJECTION = "repair_ui_projection"
    REPAIR_AUTHORIZATION_GUARD = "repair_authorization_guard"
    REGENERATE_PROJECTION = "regenerate_projection"
    UPDATE_SUBMODULE_PIN = "update_submodule_pin"


def _text(value: Any, name: str, *, required: bool = True, maximum: int = MAX_TEXT_BYTES) -> str:
    if not isinstance(value, str):
        raise RepairOperatorRegistryError(f"{name} must be a string")
    result = value.strip()
    if required and not result:
        raise RepairOperatorRegistryError(f"{name} must not be empty")
    if "\x00" in result:
        raise RepairOperatorRegistryError(f"{name} must not contain NUL")
    if len(result.encode("utf-8")) > maximum:
        raise RepairOperatorRegistryError(f"{name} exceeds its byte bound")
    return result


def _bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise RepairOperatorRegistryError(f"{name} must be a boolean")
    return value


def _positive_int(value: Any, name: str, *, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1 or value > maximum:
        raise RepairOperatorRegistryError(
            f"{name} must be an integer from 1 through {maximum}"
        )
    return value


def _enum(value: Any, enum_type: type[Enum], name: str) -> Enum:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(str(value).strip())
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(sorted(item.value for item in enum_type))
        raise RepairOperatorRegistryError(
            f"{name} must be one of: {allowed}"
        ) from exc


def _refs(
    value: Any,
    name: str,
    *,
    required: bool = True,
    maximum: int = MAX_REF_COUNT,
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise RepairOperatorRegistryError(f"{name} must be a sequence")
    if required and not value:
        raise RepairOperatorRegistryError(f"{name} must not be empty")
    if len(value) > maximum:
        raise RepairOperatorRegistryError(f"{name} exceeds its item bound")
    result: list[str] = []
    seen: set[str] = set()
    for index, item in enumerate(value):
        text = _text(item, f"{name}[{index}]")
        if text in seen:
            raise RepairOperatorRegistryError(f"{name} must not contain duplicates")
        seen.add(text)
        result.append(text)
    return tuple(result)


def _write_scope(value: Any, name: str = "write_scope") -> tuple[str, ...]:
    """Exact finite write set.  Globs and path escape markers are rejected."""

    paths = _refs(value, name, required=True, maximum=MAX_WRITE_SCOPE_GLOBS)
    for path in paths:
        if len(path.encode("utf-8")) > MAX_PATH_BYTES:
            raise RepairOperatorRegistryError(f"{name} entry exceeds path byte bound")
        if path.startswith("/") or path.startswith("\\"):
            raise RepairOperatorRegistryError(f"{name} must be relative")
        lowered = path.lower()
        for marker in _UNSAFE_SCOPE_MARKERS:
            if marker in path:
                raise RepairOperatorRegistryError(
                    f"{name} must be an exact path, not a glob or escape"
                )
        if lowered in {"", ".", "./"}:
            raise RepairOperatorRegistryError(f"{name} must not be empty or workspace-root")
        # Unbounded mutation markers.
        if lowered in {"**", "*", "/*", "/**", "repo", "repository", "workspace"}:
            raise RepairOperatorRegistryError(f"{name} is unbounded")
    return paths


def _reject_forbidden_fields(payload: Mapping[str, Any], *, label: str) -> None:
    unknown_forbidden = sorted(
        key for key in payload if key.lower() in _FORBIDDEN_DESCRIPTOR_FIELDS
    )
    if unknown_forbidden:
        raise RepairOperatorRegistryError(
            f"{label} contains forbidden fields: {', '.join(unknown_forbidden)}"
        )


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


@dataclass(frozen=True)
class OperatorDescriptor(CanonicalContract):
    """Closed metadata for one reviewed repair operator (``RepairOperator@1``)."""

    SCHEMA: ClassVar[str] = REPAIR_OPERATOR_SCHEMA
    INTERFACE: ClassVar[str] = REPAIR_OPERATOR_INTERFACE

    operator_id: str
    kind: OperatorKind
    family: OperatorFamily
    input_schema_ref: str
    write_scope: tuple[str, ...]
    before_predicates: tuple[str, ...]
    after_predicates: tuple[str, ...]
    preview_ref: str
    inverse_ref: str
    validation_refs: tuple[str, ...]
    applicability_proof_ref: str
    max_write_paths: int = 1
    version: int = REPAIR_OPERATOR_VERSION
    aliases: tuple[str, ...] = ()
    idempotent: bool = True
    invertible: bool = True
    proposal_only: bool = True
    grants_write_authority: bool = False
    grants_proof_authority: bool = False
    semantic_authority: bool = False
    allows_source_generation: bool = False

    def __post_init__(self) -> None:
        kind = _enum(self.kind, OperatorKind, "kind")
        family = _enum(self.family, OperatorFamily, "family")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "family", family)

        operator_id = _text(self.operator_id, "operator_id")
        expected_id = f"dcr-operator:{kind.value}@{REPAIR_OPERATOR_VERSION}"
        if operator_id != expected_id:
            raise RepairOperatorRegistryError(
                f"operator_id must be canonical for {kind.value}: {expected_id}"
            )
        if not _OPERATOR_ID_RE.fullmatch(operator_id):
            raise RepairOperatorRegistryError("operator_id has an invalid shape")
        object.__setattr__(self, "operator_id", operator_id)

        version = self.version
        if isinstance(version, bool) or not isinstance(version, int) or version != REPAIR_OPERATOR_VERSION:
            raise RepairOperatorRegistryError(
                f"version must be exactly {REPAIR_OPERATOR_VERSION}"
            )
        object.__setattr__(self, "version", version)

        object.__setattr__(
            self,
            "input_schema_ref",
            _text(self.input_schema_ref, "input_schema_ref"),
        )
        object.__setattr__(self, "write_scope", _write_scope(self.write_scope))
        object.__setattr__(
            self,
            "before_predicates",
            _refs(self.before_predicates, "before_predicates"),
        )
        object.__setattr__(
            self,
            "after_predicates",
            _refs(self.after_predicates, "after_predicates"),
        )
        object.__setattr__(self, "preview_ref", _text(self.preview_ref, "preview_ref"))
        inverse = _text(self.inverse_ref, "inverse_ref")
        object.__setattr__(self, "inverse_ref", inverse)
        object.__setattr__(
            self,
            "validation_refs",
            _refs(self.validation_refs, "validation_refs"),
        )
        object.__setattr__(
            self,
            "applicability_proof_ref",
            _text(self.applicability_proof_ref, "applicability_proof_ref"),
        )
        object.__setattr__(
            self,
            "max_write_paths",
            _positive_int(self.max_write_paths, "max_write_paths", maximum=MAX_WRITE_PATHS),
        )
        if self.max_write_paths < len(self.write_scope):
            raise RepairOperatorRegistryError(
                "max_write_paths must cover the declared write_scope size"
            )
        object.__setattr__(
            self,
            "aliases",
            _refs(self.aliases, "aliases", required=False),
        )
        for alias in self.aliases:
            if not _KIND_RE.fullmatch(alias):
                raise RepairOperatorRegistryError(f"alias is not a closed token: {alias}")
            if alias == kind.value:
                raise RepairOperatorRegistryError("aliases must not repeat the canonical kind")

        if not _bool(self.idempotent, "idempotent"):
            raise RepairOperatorRegistryError("registered operators must be idempotent")
        object.__setattr__(self, "idempotent", True)

        if not _bool(self.invertible, "invertible"):
            raise RepairOperatorRegistryError(
                "non-invertible operators are rejected before planning"
            )
        object.__setattr__(self, "invertible", True)
        if not inverse.startswith(("inverse:", "compensation:", "rollback:")):
            raise RepairOperatorRegistryError(
                "inverse_ref must name a reviewed inverse/compensation/rollback"
            )

        if not _bool(self.proposal_only, "proposal_only"):
            raise RepairOperatorRegistryError("operators must remain proposal-only")
        object.__setattr__(self, "proposal_only", True)

        for flag in (
            "grants_write_authority",
            "grants_proof_authority",
            "semantic_authority",
            "allows_source_generation",
        ):
            if _bool(getattr(self, flag), flag):
                raise RepairOperatorRegistryError(
                    f"{flag} cannot be true on a reviewed operator descriptor"
                )
            object.__setattr__(self, flag, False)

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": REPAIR_OPERATOR_VERSION,
            "interface": self.INTERFACE,
            "operator_id": self.operator_id,
            "kind": self.kind.value,
            "family": self.family.value,
            "version": self.version,
            "aliases": list(self.aliases),
            "input_schema_ref": self.input_schema_ref,
            "write_scope": list(self.write_scope),
            "before_predicates": list(self.before_predicates),
            "after_predicates": list(self.after_predicates),
            "preview_ref": self.preview_ref,
            "inverse_ref": self.inverse_ref,
            "validation_refs": list(self.validation_refs),
            "applicability_proof_ref": self.applicability_proof_ref,
            "max_write_paths": self.max_write_paths,
            "idempotent": True,
            "invertible": True,
            "proposal_only": True,
            "grants_write_authority": False,
            "grants_proof_authority": False,
            "semantic_authority": False,
            "allows_source_generation": False,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "OperatorDescriptor":
        if not isinstance(payload, Mapping):
            raise RepairOperatorRegistryError("operator descriptor must be an object")
        _reject_forbidden_fields(payload, label="operator descriptor")
        field_names = set(cls.__dataclass_fields__) - {"SCHEMA", "INTERFACE"}
        allowed = {
            "schema",
            "content_id",
            "contract_version",
            "interface",
            *field_names,
        }
        unknown = set(payload) - allowed
        if unknown:
            raise RepairOperatorRegistryError(
                "operator descriptor contains unknown fields: "
                + ", ".join(sorted(unknown))
            )
        if payload.get("schema", cls.SCHEMA) != cls.SCHEMA:
            raise RepairOperatorRegistryError("unsupported operator schema")
        if payload.get("contract_version", REPAIR_OPERATOR_VERSION) != REPAIR_OPERATOR_VERSION:
            raise RepairOperatorRegistryError("unsupported operator contract version")
        if payload.get("interface", cls.INTERFACE) != cls.INTERFACE:
            raise RepairOperatorRegistryError("unsupported operator interface")
        values = {name: payload[name] for name in field_names if name in payload}
        result = cls(**values)
        supplied = payload.get("content_id")
        if supplied not in (None, "", result.content_id):
            raise RepairOperatorRegistryError("operator content_id mismatch")
        return result


@dataclass(frozen=True)
class OperatorRegistry(CanonicalContract):
    """Immutable closed catalogue of reviewed repair operators."""

    SCHEMA: ClassVar[str] = REPAIR_OPERATOR_REGISTRY_SCHEMA
    INTERFACE: ClassVar[str] = REPAIR_OPERATOR_REGISTRY_INTERFACE

    operators: tuple[OperatorDescriptor, ...]
    registry_id: str = ""
    producer_id: str = REPAIR_OPERATOR_REGISTRY_PRODUCER
    evidence_id: str = OPERATOR_REGISTRY_EVIDENCE

    def __post_init__(self) -> None:
        if not self.operators:
            raise RepairOperatorRegistryError("registry must contain at least one operator")
        if len(self.operators) > MAX_OPERATOR_COUNT:
            raise RepairOperatorRegistryError("registry operator count exceeds bound")
        if not all(isinstance(item, OperatorDescriptor) for item in self.operators):
            raise RepairOperatorRegistryError(
                "operators must contain OperatorDescriptor values"
            )

        ordered = tuple(sorted(self.operators, key=lambda item: item.operator_id))
        ids = [item.operator_id for item in ordered]
        kinds = [item.kind for item in ordered]
        if len(ids) != len(set(ids)):
            raise RepairOperatorRegistryError("operator ids must be unique")
        if len(kinds) != len(set(kinds)):
            raise RepairOperatorRegistryError("operator kinds must be unique")

        alias_owner: dict[str, OperatorKind] = {}
        for item in ordered:
            tokens = (item.kind.value, item.operator_id, *item.aliases)
            for token in tokens:
                normalized = token.strip().lower().replace("-", "_")
                owner = alias_owner.get(normalized)
                if owner is not None and owner is not item.kind:
                    raise RepairOperatorRegistryError(
                        "operator aliases must resolve uniquely"
                    )
                alias_owner[normalized] = item.kind

        object.__setattr__(self, "operators", ordered)
        object.__setattr__(
            self,
            "producer_id",
            _text(self.producer_id, "producer_id"),
        )
        object.__setattr__(
            self,
            "evidence_id",
            _text(self.evidence_id, "evidence_id"),
        )
        if self.evidence_id != OPERATOR_REGISTRY_EVIDENCE:
            raise RepairOperatorRegistryError(
                f"evidence_id must be {OPERATOR_REGISTRY_EVIDENCE}"
            )

        calculated = content_identity(self._payload_without_registry_id())
        supplied = self.registry_id
        if supplied not in (None, ""):
            supplied_text = _text(supplied, "registry_id")
            if supplied_text != calculated:
                raise RepairOperatorRegistryError("registry_id mismatch")
        object.__setattr__(self, "registry_id", calculated)

    def _payload_without_registry_id(self) -> dict[str, Any]:
        return {
            "contract_version": REPAIR_OPERATOR_REGISTRY_VERSION,
            "interface": self.INTERFACE,
            "operators": [item.to_dict() for item in self.operators],
            "producer_id": self.producer_id,
            "evidence_id": self.evidence_id,
            "grants_write_authority": False,
            "grants_proof_authority": False,
            "semantic_authority": False,
            "allows_source_generation": False,
        }

    def _payload(self) -> dict[str, Any]:
        return {
            **self._payload_without_registry_id(),
            "registry_id": self.registry_id,
        }

    @property
    def descriptors(self) -> tuple[OperatorDescriptor, ...]:
        return self.operators

    @property
    def grants_write_authority(self) -> bool:
        return False

    @property
    def grants_proof_authority(self) -> bool:
        return False

    @property
    def semantic_authority(self) -> bool:
        return False

    @property
    def allows_source_generation(self) -> bool:
        return False

    def kinds(self) -> tuple[OperatorKind, ...]:
        return tuple(item.kind for item in self.operators)

    def families(self) -> tuple[OperatorFamily, ...]:
        return tuple(sorted({item.family for item in self.operators}, key=lambda f: f.value))

    def get(self, kind_or_id: Any) -> OperatorDescriptor:
        """Return the exact descriptor or raise for unknown operators."""

        raw = str(getattr(kind_or_id, "value", kind_or_id)).strip()
        if not raw:
            raise RepairOperatorRegistryError("operator kind must not be empty")
        normalized = raw.lower().replace("-", "_")
        for item in self.operators:
            tokens = {
                item.operator_id.lower().replace("-", "_"),
                item.kind.value,
                *(alias.lower().replace("-", "_") for alias in item.aliases),
            }
            if normalized in tokens:
                return item
        raise RepairOperatorRegistryError(f"unknown operator: {raw}")

    def lookup(self, kind_or_id: Any) -> OperatorDescriptor:
        """Alias of :meth:`get`; never grants proof or write authority."""

        return self.get(kind_or_id)

    def contains(self, kind_or_id: Any) -> bool:
        try:
            self.get(kind_or_id)
        except RepairOperatorRegistryError:
            return False
        return True

    def require_known(self, kind_or_id: Any) -> OperatorDescriptor:
        """Planning gate: reject unknown operators before plan admission."""

        return self.get(kind_or_id)

    def to_artifact_dict(self) -> dict[str, Any]:
        """Projection suitable for ``repair-operators.json`` materialization."""

        return {
            "schema": "ipfs_accelerate_py/agent-supervisor/deterministic-repair/repair-operators-artifact@1",
            "evidence_id": self.evidence_id,
            "interface": self.INTERFACE,
            "registry": self.to_dict(),
            "operator_count": len(self.operators),
            "kinds": [item.kind.value for item in self.operators],
            "families": [item.value for item in self.families()],
            "grants_write_authority": False,
            "allows_source_generation": False,
            "artifact_digest": "sha256:"
            + hashlib.sha256(_canonical_json_bytes(self.to_dict())).hexdigest(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "OperatorRegistry":
        if not isinstance(payload, Mapping):
            raise RepairOperatorRegistryError("registry must be an object")
        _reject_forbidden_fields(payload, label="registry")
        allowed = {
            "schema",
            "content_id",
            "contract_version",
            "interface",
            "operators",
            "registry_id",
            "producer_id",
            "evidence_id",
            "grants_write_authority",
            "grants_proof_authority",
            "semantic_authority",
            "allows_source_generation",
        }
        unknown = set(payload) - allowed
        if unknown:
            raise RepairOperatorRegistryError(
                "registry contains unknown fields: " + ", ".join(sorted(unknown))
            )
        if payload.get("schema", cls.SCHEMA) != cls.SCHEMA:
            raise RepairOperatorRegistryError("unsupported registry schema")
        if (
            payload.get("contract_version", REPAIR_OPERATOR_REGISTRY_VERSION)
            != REPAIR_OPERATOR_REGISTRY_VERSION
        ):
            raise RepairOperatorRegistryError("unsupported registry contract version")
        if payload.get("interface", cls.INTERFACE) != cls.INTERFACE:
            raise RepairOperatorRegistryError("unsupported registry interface")
        for authority in (
            "grants_write_authority",
            "grants_proof_authority",
            "semantic_authority",
            "allows_source_generation",
        ):
            if payload.get(authority, False) is not False:
                raise RepairOperatorRegistryError(
                    f"serialized registry cannot claim {authority}"
                )
        raw_operators = payload.get("operators")
        if isinstance(raw_operators, (str, bytes, bytearray)) or not isinstance(
            raw_operators, Sequence
        ):
            raise RepairOperatorRegistryError("registry operators must be a sequence")
        result = cls(
            operators=tuple(
                item
                if isinstance(item, OperatorDescriptor)
                else OperatorDescriptor.from_dict(item)
                for item in raw_operators
            ),
            registry_id=payload.get("registry_id", ""),
            producer_id=payload.get("producer_id", REPAIR_OPERATOR_REGISTRY_PRODUCER),
            evidence_id=payload.get("evidence_id", OPERATOR_REGISTRY_EVIDENCE),
        )
        supplied = payload.get("content_id")
        if supplied not in (None, "", result.content_id):
            raise RepairOperatorRegistryError("registry content_id mismatch")
        return result

    @classmethod
    def from_descriptors(
        cls,
        descriptors: Iterable[OperatorDescriptor],
        *,
        producer_id: str = REPAIR_OPERATOR_REGISTRY_PRODUCER,
    ) -> "OperatorRegistry":
        return cls(operators=tuple(descriptors), producer_id=producer_id)


def _descriptor(
    kind: OperatorKind,
    family: OperatorFamily,
    *,
    write_scope: tuple[str, ...],
    aliases: tuple[str, ...] = (),
    max_write_paths: int | None = None,
) -> OperatorDescriptor:
    """Build one canonical default-catalogue descriptor."""

    return OperatorDescriptor(
        operator_id=f"dcr-operator:{kind.value}@{REPAIR_OPERATOR_VERSION}",
        kind=kind,
        family=family,
        aliases=aliases,
        input_schema_ref=f"schema:dcr-operator/{kind.value}/input@1",
        write_scope=write_scope,
        before_predicates=(
            f"pre:{kind.value}:applicable",
            "pre:unique_anchor",
            "pre:owned_write_scope",
            "pre:exact_before_hash",
        ),
        after_predicates=(
            f"post:{kind.value}:applied",
            "post:scope_closed",
            "post:non_target_bytes_unchanged",
        ),
        preview_ref=f"preview:{kind.value}@1",
        inverse_ref=f"inverse:{kind.value}@1",
        validation_refs=(
            "validation:parse",
            "validation:schema",
            "validation:idempotence",
            "validation:inverse_roundtrip",
        ),
        applicability_proof_ref=f"proof:applicability:{kind.value}@1",
        max_write_paths=max_write_paths if max_write_paths is not None else len(write_scope),
    )


def _default_descriptors() -> tuple[OperatorDescriptor, ...]:
    """Finite reviewed catalogue for DCR-041..047 operator families."""

    return (
        # DCR-041 registry repairs
        _descriptor(
            OperatorKind.ADD_ALIAS,
            OperatorFamily.REGISTRY,
            write_scope=("scope:closed_alias_registry",),
            aliases=("register_alias",),
        ),
        _descriptor(
            OperatorKind.REMOVE_ALIAS,
            OperatorFamily.REGISTRY,
            write_scope=("scope:closed_alias_registry",),
            aliases=("unregister_alias",),
        ),
        _descriptor(
            OperatorKind.RENAME_ALIAS,
            OperatorFamily.REGISTRY,
            write_scope=("scope:closed_alias_registry",),
            aliases=("retarget_alias",),
        ),
        _descriptor(
            OperatorKind.BIND_REGISTRATION,
            OperatorFamily.REGISTRY,
            write_scope=("scope:closed_tool_registration",),
            aliases=("add_registration", "bind_tool"),
        ),
        _descriptor(
            OperatorKind.DISAMBIGUATE_ANCHOR,
            OperatorFamily.REGISTRY,
            write_scope=("scope:closed_anchor_table",),
            aliases=("unique_anchor",),
        ),
        # DCR-042 protocol/schema repairs
        _descriptor(
            OperatorKind.REPAIR_JSONRPC_SCHEMA,
            OperatorFamily.PROTOCOL,
            write_scope=("scope:closed_jsonrpc_schema",),
            aliases=("jsonrpc_schema",),
        ),
        _descriptor(
            OperatorKind.REPAIR_REQUEST_ADAPTER,
            OperatorFamily.PROTOCOL,
            write_scope=("scope:closed_request_adapter",),
            aliases=("request_adapter",),
        ),
        _descriptor(
            OperatorKind.REPAIR_ERROR_ENVELOPE,
            OperatorFamily.PROTOCOL,
            write_scope=("scope:closed_error_envelope",),
            aliases=("error_envelope",),
        ),
        _descriptor(
            OperatorKind.REPAIR_PROFILE_BINDING,
            OperatorFamily.PROTOCOL,
            write_scope=("scope:closed_profile_binding",),
            aliases=("profile_binding", "cid_profile"),
        ),
        # DCR-043 dispatch
        _descriptor(
            OperatorKind.REPAIR_DISPATCH_BINDING,
            OperatorFamily.DISPATCH,
            write_scope=("scope:closed_dispatch_table",),
            aliases=("dispatch_binding", "handler_binding"),
        ),
        # DCR-044 transport
        _descriptor(
            OperatorKind.REPAIR_TRANSPORT_ADAPTER,
            OperatorFamily.TRANSPORT,
            write_scope=("scope:closed_transport_adapter",),
            aliases=("transport_adapter",),
        ),
        _descriptor(
            OperatorKind.REPAIR_CAPABILITY_TRUTH,
            OperatorFamily.TRANSPORT,
            write_scope=("scope:closed_capability_report",),
            aliases=("capability_truth", "typed_unavailable"),
        ),
        # DCR-045 UI/ORB/IDL
        _descriptor(
            OperatorKind.REPAIR_UI_PROJECTION,
            OperatorFamily.UI,
            write_scope=("scope:closed_ui_projection",),
            aliases=("ui_projection", "orb_idl_binding"),
        ),
        # DCR-046 security
        _descriptor(
            OperatorKind.REPAIR_AUTHORIZATION_GUARD,
            OperatorFamily.SECURITY,
            write_scope=("scope:closed_authorization_guard",),
            aliases=("authorization_guard", "confirmation_check"),
        ),
        # DCR-047 codegen
        _descriptor(
            OperatorKind.REGENERATE_PROJECTION,
            OperatorFamily.CODEGEN,
            write_scope=(
                "scope:closed_generated_projection",
                "scope:closed_generated_manifest",
            ),
            aliases=("codegen_roundtrip", "regenerate_codecs"),
            max_write_paths=2,
        ),
        # Root pin update after owned change lands
        _descriptor(
            OperatorKind.UPDATE_SUBMODULE_PIN,
            OperatorFamily.ROOT,
            write_scope=("scope:closed_submodule_pin",),
            aliases=("submodule_pin",),
        ),
    )


def build_default_operator_registry() -> OperatorRegistry:
    """Return the sealed finite default catalogue."""

    return OperatorRegistry(operators=_default_descriptors())


def default_operator_registry_id() -> str:
    """Stable content identity of the default catalogue."""

    return build_default_operator_registry().registry_id


__all__ = (
    "MAX_OPERATOR_COUNT",
    "MAX_WRITE_PATHS",
    "OPERATOR_REGISTRY_EVIDENCE",
    "REPAIR_OPERATOR_INTERFACE",
    "REPAIR_OPERATOR_REGISTRY_INTERFACE",
    "REPAIR_OPERATOR_REGISTRY_SCHEMA",
    "REPAIR_OPERATOR_SCHEMA",
    "OperatorDescriptor",
    "OperatorFamily",
    "OperatorKind",
    "OperatorRegistry",
    "RepairOperatorRegistryError",
    "build_default_operator_registry",
    "default_operator_registry_id",
)
