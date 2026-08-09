"""Alias, registration, and unique-anchor repair operators (DCR-041).

Interfaces
----------
* ``RegistryRepairOperators@1`` — finite structural operators that:

  - add/remove/rename tool aliases in a closed registry;
  - bind a missing registration whose handler already exists and is uniquely
    resolved; and
  - disambiguate anchors only when a unique typed ownership edge proves
    ownership (never by lexical score).

Evidence term: ``dcr/registry-repair@1``.

Normative rules (fail-closed)
-----------------------------
* Operators are proposal-only: they mutate explicit closed tables and never
  write source, shell, or dynamic import targets.
* Ambiguous multi-anchor situations abstain unless a unique typed edge proves
  ownership.  Lexical ranking is never an admission rule.
* Stale spans (before-hash mismatch), wrong owners, and conflicting duplicate
  bindings are rejected.
* Postconditions are behavioral (resolution, ownership, reachability) — not
  mere anchor-count equality.
* Every successful apply carries an inverse that restores the prior table
  identity when the mutation was non-idempotent.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Final

from ...proof.formal_verification_contracts import (
    CanonicalContract,
    content_identity,
)
from .registry import (
    OperatorFamily,
    OperatorKind,
    build_default_operator_registry,
)

REGISTRY_REPAIR_OPERATORS_INTERFACE: Final[str] = "RegistryRepairOperators@1"
REGISTRY_REPAIR_EVIDENCE: Final[str] = "dcr/registry-repair@1"
REGISTRY_REPAIR_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-repair/registry-repairs@1"
)
ALIAS_BINDING_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-repair/alias-binding@1"
)
ALIAS_REGISTRY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-repair/alias-registry@1"
)
REGISTRATION_BINDING_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-repair/registration-binding@1"
)
REGISTRATION_TABLE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-repair/registration-table@1"
)
ANCHOR_RECORD_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-repair/anchor-record@1"
)
ANCHOR_TABLE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-repair/anchor-table@1"
)
SOURCE_SPAN_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-repair/source-span@1"
)
REGISTRY_PREVIEW_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-repair/registry-preview@1"
)
BEHAVIOR_POSTCONDITION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-repair/"
    "registry-behavior-postcondition@1"
)
REGISTRY_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-repair/registry-receipt@1"
)

MAX_ENTRIES: Final[int] = 256
MAX_TEXT_BYTES: Final[int] = 4_096
MAX_REASON_CODES: Final[int] = 32

_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"^[a-z][a-z0-9_]*(?:/[a-z][a-z0-9_]*)*$")
_ALIAS_KEY_RE: Final[re.Pattern[str]] = re.compile(
    r"^[a-z][a-z0-9_]*(?:[./:][a-z][a-z0-9_]*)*$"
)
_OWNER_REF_RE: Final[re.Pattern[str]] = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_]*"
    r"(?:\.[A-Za-z_][A-Za-z0-9_]*)+"
    r":[A-Za-z_][A-Za-z0-9_]*$"
)
_PATH_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+\.[A-Za-z0-9_.-]+$"
)
_HASH_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?:sha256:)?[0-9a-f]{64}$|^(?:bafy|bagu|bafk)[a-z0-9]{8,200}$"
)

_FORBIDDEN_BODY_FIELDS: Final[frozenset[str]] = frozenset(
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
        "handler_body",
        "dynamic_import",
        "exec",
        "eval",
        "llm_prompt",
        "prose",
        "patch_body",
        "diff_body",
    }
)
_FORBIDDEN_OWNER_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "model",
        "llm",
        "openai",
        "anthropic",
        "grok",
        "codex",
        "provider",
        "prompt",
        "residual",
    }
)


class RegistryRepairError(ValueError):
    """Malformed registry repair input or unsafe structural mutation."""


class RegistryRepairAbstention(RegistryRepairError):
    """Operator cannot proceed without inventing ownership or semantics."""


class RegistryOperatorKind(str, Enum):  # noqa: UP042 - package supports Python 3.8
    """Closed structural registry operators for DCR-041."""

    ADD_ALIAS = "add_alias"
    REMOVE_ALIAS = "remove_alias"
    RENAME_ALIAS = "rename_alias"
    BIND_REGISTRATION = "bind_registration"
    DISAMBIGUATE_ANCHOR = "disambiguate_anchor"


class AnchorKind(str, Enum):  # noqa: UP042
    """Closed AST/surface anchor kinds retained in the anchor table."""

    ALIAS = "alias"
    REGISTRATION = "registration"
    HANDLER = "handler"
    TOOL = "tool"
    SURFACE = "surface"


class OwnershipEdgeKind(str, Enum):  # noqa: UP042
    """Typed ownership edges that may admit disambiguation."""

    DECLARATION_TO_REGISTRATION = "declaration_to_registration"
    REGISTRATION_TO_HANDLER = "registration_to_handler"
    ALIAS_TO_TOOL = "alias_to_tool"
    SURFACE_TO_OWNER = "surface_to_owner"


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _text(value: Any, name: str, *, required: bool = True, maximum: int = MAX_TEXT_BYTES) -> str:
    if not isinstance(value, str):
        raise RegistryRepairError(f"{name} must be a string")
    result = value.strip()
    if required and not result:
        raise RegistryRepairError(f"{name} must not be empty")
    if "\x00" in result:
        raise RegistryRepairError(f"{name} must not contain NUL")
    if len(result.encode("utf-8")) > maximum:
        raise RegistryRepairError(f"{name} exceeds its byte bound")
    return result


def _token(value: Any, name: str) -> str:
    text = _text(value, name)
    if not _TOKEN_RE.fullmatch(text):
        raise RegistryRepairError(f"{name} must be a closed lowercase token path")
    return text


def _alias_key(value: Any, name: str = "alias_key") -> str:
    text = _text(value, name)
    if not _ALIAS_KEY_RE.fullmatch(text):
        raise RegistryRepairError(f"{name} must be a closed alias key")
    return text


def _owner_ref(value: Any, name: str = "owner_ref") -> str:
    text = _text(value, name)
    lowered = text.lower()
    for marker in _FORBIDDEN_OWNER_MARKERS:
        if marker in lowered:
            raise RegistryRepairError(
                f"{name} must not route to a model/provider surface ({marker})"
            )
    if not _OWNER_REF_RE.fullmatch(text):
        raise RegistryRepairError(
            f"{name} must be a module:callable owner reference, not a body"
        )
    return text


def _path(value: Any, name: str = "path") -> str:
    text = _text(value, name)
    if ".." in text or text.startswith("/") or text.startswith("\\"):
        raise RegistryRepairError(f"{name} must be a relative non-escaping path")
    if not _PATH_RE.fullmatch(text):
        raise RegistryRepairError(f"{name} must be a closed relative source path")
    return text


def _hash(value: Any, name: str) -> str:
    text = _text(value, name)
    if not _HASH_RE.fullmatch(text):
        raise RegistryRepairError(f"{name} must be a content hash or CID")
    if text.startswith("sha256:") or text.startswith(("bafy", "bagu", "bafk")):
        return text
    return f"sha256:{text}"


def _bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise RegistryRepairError(f"{name} must be a boolean")
    return value


def _enum(value: Any, enum_type: type[Enum], name: str) -> Enum:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(str(value).strip())
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(sorted(item.value for item in enum_type))
        raise RegistryRepairError(f"{name} must be one of: {allowed}") from exc


def _reject_body_fields(payload: Mapping[str, Any], *, label: str) -> None:
    present = sorted(key for key in payload if str(key).lower() in _FORBIDDEN_BODY_FIELDS)
    if present:
        raise RegistryRepairError(
            f"{label} contains forbidden body/generation fields: {', '.join(present)}"
        )


def _reason_codes(value: Any, name: str = "reason_codes") -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise RegistryRepairError(f"{name} must be a sequence")
    if len(value) > MAX_REASON_CODES:
        raise RegistryRepairError(f"{name} exceeds its item bound")
    return tuple(_text(item, f"{name}[{index}]") for index, item in enumerate(value))


# ---------------------------------------------------------------------------
# Closed structural models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceSpan(CanonicalContract):
    """Exact source span with a content before-hash for stale-span detection."""

    SCHEMA: ClassVar[str] = SOURCE_SPAN_SCHEMA
    INTERFACE: ClassVar[str] = REGISTRY_REPAIR_OPERATORS_INTERFACE

    path: str
    start_offset: int
    end_offset: int
    before_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _path(self.path, "path"))
        if (
            isinstance(self.start_offset, bool)
            or not isinstance(self.start_offset, int)
            or self.start_offset < 0
        ):
            raise RegistryRepairError("start_offset must be a non-negative integer")
        if (
            isinstance(self.end_offset, bool)
            or not isinstance(self.end_offset, int)
            or self.end_offset < self.start_offset
        ):
            raise RegistryRepairError("end_offset must be >= start_offset")
        object.__setattr__(self, "before_hash", _hash(self.before_hash, "before_hash"))

    def _payload(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
            "before_hash": self.before_hash,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SourceSpan":
        if not isinstance(payload, Mapping):
            raise RegistryRepairError("source span must be an object")
        _reject_body_fields(payload, label="source span")
        return cls(
            path=str(payload.get("path") or ""),
            start_offset=int(payload.get("start_offset", -1)),
            end_offset=int(payload.get("end_offset", -1)),
            before_hash=str(payload.get("before_hash") or ""),
        )


@dataclass(frozen=True)
class AliasBinding(CanonicalContract):
    """One closed alias → semantic-target binding."""

    SCHEMA: ClassVar[str] = ALIAS_BINDING_SCHEMA
    INTERFACE: ClassVar[str] = REGISTRY_REPAIR_OPERATORS_INTERFACE

    alias_key: str
    target_tool: str
    owner_ref: str
    span: SourceSpan
    registry_key: str = ""
    semantic_target: str = ""

    def __post_init__(self) -> None:
        alias_key = _alias_key(self.alias_key)
        target_tool = _token(self.target_tool, "target_tool")
        owner_ref = _owner_ref(self.owner_ref)
        if not isinstance(self.span, SourceSpan):
            raise RegistryRepairError("span must be a SourceSpan")
        registry_key = _text(self.registry_key or f"alias:{alias_key}", "registry_key")
        semantic_target = _text(
            self.semantic_target or f"tool:{target_tool}", "semantic_target"
        )
        object.__setattr__(self, "alias_key", alias_key)
        object.__setattr__(self, "target_tool", target_tool)
        object.__setattr__(self, "owner_ref", owner_ref)
        object.__setattr__(self, "registry_key", registry_key)
        object.__setattr__(self, "semantic_target", semantic_target)

    def _payload(self) -> dict[str, Any]:
        return {
            "interface": self.INTERFACE,
            "alias_key": self.alias_key,
            "target_tool": self.target_tool,
            "owner_ref": self.owner_ref,
            "span": self.span.to_dict(),
            "registry_key": self.registry_key,
            "semantic_target": self.semantic_target,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AliasBinding":
        if not isinstance(payload, Mapping):
            raise RegistryRepairError("alias binding must be an object")
        _reject_body_fields(payload, label="alias binding")
        span_raw = payload.get("span")
        if not isinstance(span_raw, Mapping):
            raise RegistryRepairError("alias binding span must be an object")
        return cls(
            alias_key=str(payload.get("alias_key") or ""),
            target_tool=str(payload.get("target_tool") or ""),
            owner_ref=str(payload.get("owner_ref") or ""),
            span=SourceSpan.from_dict(span_raw),
            registry_key=str(payload.get("registry_key") or ""),
            semantic_target=str(payload.get("semantic_target") or ""),
        )


@dataclass(frozen=True)
class AliasRegistry(CanonicalContract):
    """Closed finite alias table for proposal previews."""

    SCHEMA: ClassVar[str] = ALIAS_REGISTRY_SCHEMA
    INTERFACE: ClassVar[str] = REGISTRY_REPAIR_OPERATORS_INTERFACE

    bindings: tuple[AliasBinding, ...] = ()
    table_id: str = ""
    owner_scope: str = "scope:closed_alias_registry"

    def __post_init__(self) -> None:
        if not isinstance(self.bindings, Sequence) or isinstance(
            self.bindings, (str, bytes, bytearray)
        ):
            raise RegistryRepairError("bindings must be a sequence")
        if len(self.bindings) > MAX_ENTRIES:
            raise RegistryRepairError("bindings exceed the closed bound")
        if not all(isinstance(item, AliasBinding) for item in self.bindings):
            raise RegistryRepairError("bindings must contain AliasBinding values")
        ordered = tuple(sorted(self.bindings, key=lambda item: item.alias_key))
        keys = [item.alias_key for item in ordered]
        if len(keys) != len(set(keys)):
            raise RegistryRepairError("alias keys must be unique")
        object.__setattr__(self, "bindings", ordered)
        object.__setattr__(
            self, "owner_scope", _text(self.owner_scope, "owner_scope")
        )
        calculated = content_identity(self._payload_without_table_id())
        if self.table_id not in (None, ""):
            supplied = _text(self.table_id, "table_id")
            if supplied != calculated:
                raise RegistryRepairError("table_id mismatch")
        object.__setattr__(self, "table_id", calculated)

    def _payload_without_table_id(self) -> dict[str, Any]:
        return {
            "interface": self.INTERFACE,
            "bindings": [item.to_dict() for item in self.bindings],
            "owner_scope": self.owner_scope,
            "grants_write_authority": False,
            "allows_source_generation": False,
        }

    def _payload(self) -> dict[str, Any]:
        return {**self._payload_without_table_id(), "table_id": self.table_id}

    def keys(self) -> tuple[str, ...]:
        return tuple(item.alias_key for item in self.bindings)

    def get(self, alias_key: str) -> AliasBinding | None:
        key = _alias_key(alias_key)
        for item in self.bindings:
            if item.alias_key == key:
                return item
        return None

    def contains(self, alias_key: str) -> bool:
        return self.get(alias_key) is not None

    def with_binding(self, binding: AliasBinding) -> "AliasRegistry":
        if not isinstance(binding, AliasBinding):
            raise RegistryRepairError("binding must be an AliasBinding")
        remaining = [item for item in self.bindings if item.alias_key != binding.alias_key]
        return AliasRegistry(
            bindings=tuple(remaining) + (binding,),
            owner_scope=self.owner_scope,
        )

    def without_key(self, alias_key: str) -> "AliasRegistry":
        key = _alias_key(alias_key)
        remaining = tuple(item for item in self.bindings if item.alias_key != key)
        return AliasRegistry(bindings=remaining, owner_scope=self.owner_scope)

    def rename_key(self, old_key: str, new_key: str) -> "AliasRegistry":
        old = _alias_key(old_key, "old_key")
        new = _alias_key(new_key, "new_key")
        existing = self.get(old)
        if existing is None:
            raise RegistryRepairError(f"alias not found: {old}")
        if old != new and self.contains(new):
            raise RegistryRepairError(f"rename target already bound: {new}")
        renamed = AliasBinding(
            alias_key=new,
            target_tool=existing.target_tool,
            owner_ref=existing.owner_ref,
            span=existing.span,
            registry_key=f"alias:{new}",
            semantic_target=existing.semantic_target,
        )
        return self.without_key(old).with_binding(renamed)

    @classmethod
    def empty(cls, *, owner_scope: str = "scope:closed_alias_registry") -> "AliasRegistry":
        return cls(bindings=(), owner_scope=owner_scope)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AliasRegistry":
        if not isinstance(payload, Mapping):
            raise RegistryRepairError("alias registry must be an object")
        _reject_body_fields(payload, label="alias registry")
        raw = payload.get("bindings") or ()
        if isinstance(raw, (str, bytes, bytearray)) or not isinstance(raw, Sequence):
            raise RegistryRepairError("bindings must be a sequence")
        return cls(
            bindings=tuple(
                item if isinstance(item, AliasBinding) else AliasBinding.from_dict(item)
                for item in raw
            ),
            table_id=str(payload.get("table_id") or ""),
            owner_scope=str(payload.get("owner_scope") or "scope:closed_alias_registry"),
        )


@dataclass(frozen=True)
class RegistrationBinding(CanonicalContract):
    """One closed tool registration bound to a unique handler owner."""

    SCHEMA: ClassVar[str] = REGISTRATION_BINDING_SCHEMA
    INTERFACE: ClassVar[str] = REGISTRY_REPAIR_OPERATORS_INTERFACE

    tool_name: str
    owner_ref: str
    handler_ref: str
    input_schema_ref: str
    span: SourceSpan
    registry_key: str = ""
    semantic_target: str = ""
    handler_exists: bool = True

    def __post_init__(self) -> None:
        tool_name = _token(self.tool_name, "tool_name")
        owner_ref = _owner_ref(self.owner_ref)
        handler_ref = _owner_ref(self.handler_ref, "handler_ref")
        if not isinstance(self.span, SourceSpan):
            raise RegistryRepairError("span must be a SourceSpan")
        if not _bool(self.handler_exists, "handler_exists"):
            raise RegistryRepairError(
                "registration binding requires an existing unique handler"
            )
        object.__setattr__(self, "tool_name", tool_name)
        object.__setattr__(self, "owner_ref", owner_ref)
        object.__setattr__(self, "handler_ref", handler_ref)
        object.__setattr__(
            self, "input_schema_ref", _text(self.input_schema_ref, "input_schema_ref")
        )
        object.__setattr__(
            self,
            "registry_key",
            _text(self.registry_key or f"registration:{tool_name}", "registry_key"),
        )
        object.__setattr__(
            self,
            "semantic_target",
            _text(
                self.semantic_target or f"handler:{handler_ref}",
                "semantic_target",
            ),
        )
        object.__setattr__(self, "handler_exists", True)

    def _payload(self) -> dict[str, Any]:
        return {
            "interface": self.INTERFACE,
            "tool_name": self.tool_name,
            "owner_ref": self.owner_ref,
            "handler_ref": self.handler_ref,
            "input_schema_ref": self.input_schema_ref,
            "span": self.span.to_dict(),
            "registry_key": self.registry_key,
            "semantic_target": self.semantic_target,
            "handler_exists": True,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RegistrationBinding":
        if not isinstance(payload, Mapping):
            raise RegistryRepairError("registration binding must be an object")
        _reject_body_fields(payload, label="registration binding")
        span_raw = payload.get("span")
        if not isinstance(span_raw, Mapping):
            raise RegistryRepairError("registration binding span must be an object")
        return cls(
            tool_name=str(payload.get("tool_name") or ""),
            owner_ref=str(payload.get("owner_ref") or ""),
            handler_ref=str(payload.get("handler_ref") or ""),
            input_schema_ref=str(payload.get("input_schema_ref") or ""),
            span=SourceSpan.from_dict(span_raw),
            registry_key=str(payload.get("registry_key") or ""),
            semantic_target=str(payload.get("semantic_target") or ""),
            handler_exists=bool(payload.get("handler_exists", True)),
        )


@dataclass(frozen=True)
class RegistrationTable(CanonicalContract):
    """Closed finite tool-registration table for proposal previews."""

    SCHEMA: ClassVar[str] = REGISTRATION_TABLE_SCHEMA
    INTERFACE: ClassVar[str] = REGISTRY_REPAIR_OPERATORS_INTERFACE

    bindings: tuple[RegistrationBinding, ...] = ()
    known_handlers: tuple[str, ...] = ()
    table_id: str = ""
    owner_scope: str = "scope:closed_tool_registration"

    def __post_init__(self) -> None:
        if not isinstance(self.bindings, Sequence) or isinstance(
            self.bindings, (str, bytes, bytearray)
        ):
            raise RegistryRepairError("bindings must be a sequence")
        if len(self.bindings) > MAX_ENTRIES:
            raise RegistryRepairError("bindings exceed the closed bound")
        if not all(isinstance(item, RegistrationBinding) for item in self.bindings):
            raise RegistryRepairError("bindings must contain RegistrationBinding values")
        ordered = tuple(sorted(self.bindings, key=lambda item: item.tool_name))
        names = [item.tool_name for item in ordered]
        if len(names) != len(set(names)):
            raise RegistryRepairError("tool registrations must be unique")
        object.__setattr__(self, "bindings", ordered)

        if not isinstance(self.known_handlers, Sequence) or isinstance(
            self.known_handlers, (str, bytes, bytearray)
        ):
            raise RegistryRepairError("known_handlers must be a sequence")
        handlers = tuple(
            _owner_ref(item, f"known_handlers[{index}]")
            for index, item in enumerate(self.known_handlers)
        )
        if len(handlers) != len(set(handlers)):
            raise RegistryRepairError("known_handlers must be unique")
        object.__setattr__(self, "known_handlers", handlers)
        object.__setattr__(
            self, "owner_scope", _text(self.owner_scope, "owner_scope")
        )

        calculated = content_identity(self._payload_without_table_id())
        if self.table_id not in (None, ""):
            supplied = _text(self.table_id, "table_id")
            if supplied != calculated:
                raise RegistryRepairError("table_id mismatch")
        object.__setattr__(self, "table_id", calculated)

    def _payload_without_table_id(self) -> dict[str, Any]:
        return {
            "interface": self.INTERFACE,
            "bindings": [item.to_dict() for item in self.bindings],
            "known_handlers": list(self.known_handlers),
            "owner_scope": self.owner_scope,
            "grants_write_authority": False,
            "allows_source_generation": False,
        }

    def _payload(self) -> dict[str, Any]:
        return {**self._payload_without_table_id(), "table_id": self.table_id}

    def tools(self) -> tuple[str, ...]:
        return tuple(item.tool_name for item in self.bindings)

    def get(self, tool_name: str) -> RegistrationBinding | None:
        name = _token(tool_name, "tool_name")
        for item in self.bindings:
            if item.tool_name == name:
                return item
        return None

    def contains(self, tool_name: str) -> bool:
        return self.get(tool_name) is not None

    def handler_is_known(self, handler_ref: str) -> bool:
        return _owner_ref(handler_ref, "handler_ref") in self.known_handlers

    def with_binding(self, binding: RegistrationBinding) -> "RegistrationTable":
        if not isinstance(binding, RegistrationBinding):
            raise RegistryRepairError("binding must be a RegistrationBinding")
        remaining = [item for item in self.bindings if item.tool_name != binding.tool_name]
        handlers = self.known_handlers
        if binding.handler_ref not in handlers:
            handlers = (*handlers, binding.handler_ref)
        return RegistrationTable(
            bindings=tuple(remaining) + (binding,),
            known_handlers=handlers,
            owner_scope=self.owner_scope,
        )

    def without_tool(self, tool_name: str) -> "RegistrationTable":
        name = _token(tool_name, "tool_name")
        remaining = tuple(item for item in self.bindings if item.tool_name != name)
        return RegistrationTable(
            bindings=remaining,
            known_handlers=self.known_handlers,
            owner_scope=self.owner_scope,
        )

    @classmethod
    def empty(
        cls,
        *,
        known_handlers: Sequence[str] = (),
        owner_scope: str = "scope:closed_tool_registration",
    ) -> "RegistrationTable":
        return cls(
            bindings=(),
            known_handlers=tuple(known_handlers),
            owner_scope=owner_scope,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RegistrationTable":
        if not isinstance(payload, Mapping):
            raise RegistryRepairError("registration table must be an object")
        _reject_body_fields(payload, label="registration table")
        raw = payload.get("bindings") or ()
        if isinstance(raw, (str, bytes, bytearray)) or not isinstance(raw, Sequence):
            raise RegistryRepairError("bindings must be a sequence")
        return cls(
            bindings=tuple(
                item
                if isinstance(item, RegistrationBinding)
                else RegistrationBinding.from_dict(item)
                for item in raw
            ),
            known_handlers=tuple(payload.get("known_handlers") or ()),
            table_id=str(payload.get("table_id") or ""),
            owner_scope=str(
                payload.get("owner_scope") or "scope:closed_tool_registration"
            ),
        )


@dataclass(frozen=True)
class AnchorRecord(CanonicalContract):
    """One AST/surface anchor with ownership and span evidence."""

    SCHEMA: ClassVar[str] = ANCHOR_RECORD_SCHEMA
    INTERFACE: ClassVar[str] = REGISTRY_REPAIR_OPERATORS_INTERFACE

    anchor_id: str
    kind: AnchorKind
    registry_key: str
    owner_ref: str
    semantic_target: str
    span: SourceSpan
    ownership_edge: OwnershipEdgeKind
    selected: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "anchor_id", _text(self.anchor_id, "anchor_id"))
        object.__setattr__(self, "kind", _enum(self.kind, AnchorKind, "kind"))
        object.__setattr__(
            self, "registry_key", _text(self.registry_key, "registry_key")
        )
        object.__setattr__(self, "owner_ref", _owner_ref(self.owner_ref))
        object.__setattr__(
            self, "semantic_target", _text(self.semantic_target, "semantic_target")
        )
        if not isinstance(self.span, SourceSpan):
            raise RegistryRepairError("span must be a SourceSpan")
        object.__setattr__(
            self,
            "ownership_edge",
            _enum(self.ownership_edge, OwnershipEdgeKind, "ownership_edge"),
        )
        object.__setattr__(self, "selected", _bool(self.selected, "selected"))

    def _payload(self) -> dict[str, Any]:
        return {
            "interface": self.INTERFACE,
            "anchor_id": self.anchor_id,
            "kind": self.kind.value,
            "registry_key": self.registry_key,
            "owner_ref": self.owner_ref,
            "semantic_target": self.semantic_target,
            "span": self.span.to_dict(),
            "ownership_edge": self.ownership_edge.value,
            "selected": self.selected,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AnchorRecord":
        if not isinstance(payload, Mapping):
            raise RegistryRepairError("anchor record must be an object")
        _reject_body_fields(payload, label="anchor record")
        span_raw = payload.get("span")
        if not isinstance(span_raw, Mapping):
            raise RegistryRepairError("anchor span must be an object")
        return cls(
            anchor_id=str(payload.get("anchor_id") or ""),
            kind=payload.get("kind") or AnchorKind.SURFACE,
            registry_key=str(payload.get("registry_key") or ""),
            owner_ref=str(payload.get("owner_ref") or ""),
            semantic_target=str(payload.get("semantic_target") or ""),
            span=SourceSpan.from_dict(span_raw),
            ownership_edge=payload.get("ownership_edge")
            or OwnershipEdgeKind.SURFACE_TO_OWNER,
            selected=bool(payload.get("selected", False)),
        )


@dataclass(frozen=True)
class AnchorTable(CanonicalContract):
    """Closed finite anchor table used for unique-anchor resolution."""

    SCHEMA: ClassVar[str] = ANCHOR_TABLE_SCHEMA
    INTERFACE: ClassVar[str] = REGISTRY_REPAIR_OPERATORS_INTERFACE

    anchors: tuple[AnchorRecord, ...] = ()
    table_id: str = ""
    owner_scope: str = "scope:closed_anchor_table"

    def __post_init__(self) -> None:
        if not isinstance(self.anchors, Sequence) or isinstance(
            self.anchors, (str, bytes, bytearray)
        ):
            raise RegistryRepairError("anchors must be a sequence")
        if len(self.anchors) > MAX_ENTRIES:
            raise RegistryRepairError("anchors exceed the closed bound")
        if not all(isinstance(item, AnchorRecord) for item in self.anchors):
            raise RegistryRepairError("anchors must contain AnchorRecord values")
        ordered = tuple(sorted(self.anchors, key=lambda item: item.anchor_id))
        ids = [item.anchor_id for item in ordered]
        if len(ids) != len(set(ids)):
            raise RegistryRepairError("anchor ids must be unique")
        object.__setattr__(self, "anchors", ordered)
        object.__setattr__(
            self, "owner_scope", _text(self.owner_scope, "owner_scope")
        )
        calculated = content_identity(self._payload_without_table_id())
        if self.table_id not in (None, ""):
            supplied = _text(self.table_id, "table_id")
            if supplied != calculated:
                raise RegistryRepairError("table_id mismatch")
        object.__setattr__(self, "table_id", calculated)

    def _payload_without_table_id(self) -> dict[str, Any]:
        return {
            "interface": self.INTERFACE,
            "anchors": [item.to_dict() for item in self.anchors],
            "owner_scope": self.owner_scope,
            "grants_write_authority": False,
            "allows_source_generation": False,
        }

    def _payload(self) -> dict[str, Any]:
        return {**self._payload_without_table_id(), "table_id": self.table_id}

    def for_key(self, registry_key: str) -> tuple[AnchorRecord, ...]:
        key = _text(registry_key, "registry_key")
        return tuple(item for item in self.anchors if item.registry_key == key)

    def selected_for_key(self, registry_key: str) -> AnchorRecord | None:
        selected = [item for item in self.for_key(registry_key) if item.selected]
        if not selected:
            return None
        if len(selected) > 1:
            raise RegistryRepairError(
                f"multiple selected anchors for registry_key {registry_key}"
            )
        return selected[0]

    def with_anchors(self, anchors: Sequence[AnchorRecord]) -> "AnchorTable":
        if not isinstance(anchors, Sequence) or isinstance(
            anchors, (str, bytes, bytearray)
        ):
            raise RegistryRepairError("anchors must be a sequence")
        return AnchorTable(anchors=tuple(anchors), owner_scope=self.owner_scope)

    def replace_key(
        self,
        registry_key: str,
        anchors: Sequence[AnchorRecord],
    ) -> "AnchorTable":
        key = _text(registry_key, "registry_key")
        remaining = [item for item in self.anchors if item.registry_key != key]
        for item in anchors:
            if not isinstance(item, AnchorRecord):
                raise RegistryRepairError("anchors must contain AnchorRecord values")
            if item.registry_key != key:
                raise RegistryRepairError(
                    "replacement anchors must share the target registry_key"
                )
        return AnchorTable(
            anchors=tuple(remaining) + tuple(anchors),
            owner_scope=self.owner_scope,
        )

    @classmethod
    def empty(cls, *, owner_scope: str = "scope:closed_anchor_table") -> "AnchorTable":
        return cls(anchors=(), owner_scope=owner_scope)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AnchorTable":
        if not isinstance(payload, Mapping):
            raise RegistryRepairError("anchor table must be an object")
        _reject_body_fields(payload, label="anchor table")
        raw = payload.get("anchors") or ()
        if isinstance(raw, (str, bytes, bytearray)) or not isinstance(raw, Sequence):
            raise RegistryRepairError("anchors must be a sequence")
        return cls(
            anchors=tuple(
                item if isinstance(item, AnchorRecord) else AnchorRecord.from_dict(item)
                for item in raw
            ),
            table_id=str(payload.get("table_id") or ""),
            owner_scope=str(payload.get("owner_scope") or "scope:closed_anchor_table"),
        )


@dataclass(frozen=True)
class BehaviorPostcondition(CanonicalContract):
    """Behavioral postcondition — replaces anchor-count-only validation.

    A successful registry repair must prove:

    * the registry key resolves to exactly one semantic target;
    * that target is owned by a unique typed owner reference;
    * the retained span is fresh against the supplied before-hash; and
    * the inverse patch identity is recorded for rollback.

    Merely counting anchors (``len(anchors) == 1``) is insufficient.
    """

    SCHEMA: ClassVar[str] = BEHAVIOR_POSTCONDITION_SCHEMA
    INTERFACE: ClassVar[str] = REGISTRY_REPAIR_OPERATORS_INTERFACE

    registry_key: str
    semantic_target: str
    owner_ref: str
    resolves: bool
    unique_owner: bool
    span_fresh: bool
    inverse_patch_id: str
    before_hash: str
    after_table_id: str
    anchor_count: int = 0
    count_only_sufficient: bool = False
    behavior_satisfied: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "registry_key", _text(self.registry_key, "registry_key")
        )
        object.__setattr__(
            self, "semantic_target", _text(self.semantic_target, "semantic_target")
        )
        object.__setattr__(self, "owner_ref", _owner_ref(self.owner_ref))
        object.__setattr__(self, "resolves", _bool(self.resolves, "resolves"))
        object.__setattr__(
            self, "unique_owner", _bool(self.unique_owner, "unique_owner")
        )
        object.__setattr__(self, "span_fresh", _bool(self.span_fresh, "span_fresh"))
        object.__setattr__(
            self, "inverse_patch_id", _text(self.inverse_patch_id, "inverse_patch_id")
        )
        object.__setattr__(self, "before_hash", _hash(self.before_hash, "before_hash"))
        object.__setattr__(
            self, "after_table_id", _text(self.after_table_id, "after_table_id")
        )
        if (
            isinstance(self.anchor_count, bool)
            or not isinstance(self.anchor_count, int)
            or self.anchor_count < 0
        ):
            raise RegistryRepairError("anchor_count must be a non-negative integer")
        # Anchor-count-only validation is intentionally never sufficient.
        if self.count_only_sufficient is not False:
            raise RegistryRepairError(
                "count_only_sufficient must remain false; behavior postconditions "
                "replace anchor-count-only validation"
            )
        object.__setattr__(self, "count_only_sufficient", False)
        satisfied = bool(
            self.resolves
            and self.unique_owner
            and self.span_fresh
            and self.semantic_target
            and self.owner_ref
            and self.inverse_patch_id
        )
        # Even a single retained anchor fails if ownership/span/resolution fail.
        if self.anchor_count == 1 and not satisfied:
            satisfied = False
        object.__setattr__(self, "behavior_satisfied", satisfied)
        if type(self.behavior_satisfied) is not bool:
            raise RegistryRepairError("behavior_satisfied must be a boolean")

    def _payload(self) -> dict[str, Any]:
        return {
            "interface": self.INTERFACE,
            "registry_key": self.registry_key,
            "semantic_target": self.semantic_target,
            "owner_ref": self.owner_ref,
            "resolves": self.resolves,
            "unique_owner": self.unique_owner,
            "span_fresh": self.span_fresh,
            "inverse_patch_id": self.inverse_patch_id,
            "before_hash": self.before_hash,
            "after_table_id": self.after_table_id,
            "anchor_count": self.anchor_count,
            "count_only_sufficient": False,
            "behavior_satisfied": self.behavior_satisfied,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BehaviorPostcondition":
        if not isinstance(payload, Mapping):
            raise RegistryRepairError("behavior postcondition must be an object")
        _reject_body_fields(payload, label="behavior postcondition")
        return cls(
            registry_key=str(payload.get("registry_key") or ""),
            semantic_target=str(payload.get("semantic_target") or ""),
            owner_ref=str(payload.get("owner_ref") or ""),
            resolves=bool(payload.get("resolves", False)),
            unique_owner=bool(payload.get("unique_owner", False)),
            span_fresh=bool(payload.get("span_fresh", False)),
            inverse_patch_id=str(payload.get("inverse_patch_id") or ""),
            before_hash=str(payload.get("before_hash") or ""),
            after_table_id=str(payload.get("after_table_id") or ""),
            anchor_count=int(payload.get("anchor_count", 0)),
            count_only_sufficient=bool(payload.get("count_only_sufficient", False)),
        )


@dataclass(frozen=True)
class RegistryPreview(CanonicalContract):
    """Proposal-only preview for one structural registry operator."""

    SCHEMA: ClassVar[str] = REGISTRY_PREVIEW_SCHEMA
    INTERFACE: ClassVar[str] = REGISTRY_REPAIR_OPERATORS_INTERFACE

    operator_kind: str
    before_table_id: str
    after_table_id: str
    registry_key: str
    semantic_target: str
    owner_ref: str
    before_hash: str
    inverse_kind: str
    inverse_patch_id: str
    applicable: bool
    reason_codes: tuple[str, ...] = ()
    postcondition: BehaviorPostcondition | None = None
    proposal_only: bool = True
    grants_write_authority: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "operator_kind", _token(self.operator_kind, "operator_kind")
        )
        object.__setattr__(
            self, "before_table_id", _text(self.before_table_id, "before_table_id")
        )
        object.__setattr__(
            self, "after_table_id", _text(self.after_table_id, "after_table_id")
        )
        object.__setattr__(
            self, "registry_key", _text(self.registry_key, "registry_key")
        )
        object.__setattr__(
            self, "semantic_target", _text(self.semantic_target, "semantic_target")
        )
        object.__setattr__(self, "owner_ref", _owner_ref(self.owner_ref))
        object.__setattr__(self, "before_hash", _hash(self.before_hash, "before_hash"))
        object.__setattr__(
            self, "inverse_kind", _token(self.inverse_kind, "inverse_kind")
        )
        object.__setattr__(
            self, "inverse_patch_id", _text(self.inverse_patch_id, "inverse_patch_id")
        )
        object.__setattr__(self, "applicable", _bool(self.applicable, "applicable"))
        object.__setattr__(self, "reason_codes", _reason_codes(self.reason_codes))
        if self.postcondition is not None and not isinstance(
            self.postcondition, BehaviorPostcondition
        ):
            raise RegistryRepairError("postcondition must be a BehaviorPostcondition")
        if self.proposal_only is not True:
            raise RegistryRepairError("registry previews must remain proposal-only")
        if self.grants_write_authority is not False:
            raise RegistryRepairError("registry previews cannot grant write authority")
        object.__setattr__(self, "proposal_only", True)
        object.__setattr__(self, "grants_write_authority", False)

    def _payload(self) -> dict[str, Any]:
        return {
            "interface": self.INTERFACE,
            "operator_kind": self.operator_kind,
            "before_table_id": self.before_table_id,
            "after_table_id": self.after_table_id,
            "registry_key": self.registry_key,
            "semantic_target": self.semantic_target,
            "owner_ref": self.owner_ref,
            "before_hash": self.before_hash,
            "inverse_kind": self.inverse_kind,
            "inverse_patch_id": self.inverse_patch_id,
            "applicable": self.applicable,
            "reason_codes": list(self.reason_codes),
            "postcondition": (
                None if self.postcondition is None else self.postcondition.to_dict()
            ),
            "proposal_only": True,
            "grants_write_authority": False,
        }


@dataclass(frozen=True)
class RegistryRepairReceipt(CanonicalContract):
    """Evidence receipt for one applied registry operator (proposal scope)."""

    SCHEMA: ClassVar[str] = REGISTRY_RECEIPT_SCHEMA
    INTERFACE: ClassVar[str] = REGISTRY_REPAIR_OPERATORS_INTERFACE

    operator_kind: str
    registry_key: str
    before_table_id: str
    after_table_id: str
    before_hash: str
    inverse_patch_id: str
    postcondition: BehaviorPostcondition
    receipt_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "operator_kind", _token(self.operator_kind, "operator_kind")
        )
        object.__setattr__(
            self, "registry_key", _text(self.registry_key, "registry_key")
        )
        object.__setattr__(
            self, "before_table_id", _text(self.before_table_id, "before_table_id")
        )
        object.__setattr__(
            self, "after_table_id", _text(self.after_table_id, "after_table_id")
        )
        object.__setattr__(self, "before_hash", _hash(self.before_hash, "before_hash"))
        object.__setattr__(
            self, "inverse_patch_id", _text(self.inverse_patch_id, "inverse_patch_id")
        )
        if not isinstance(self.postcondition, BehaviorPostcondition):
            raise RegistryRepairError("postcondition must be a BehaviorPostcondition")
        if not self.postcondition.behavior_satisfied:
            raise RegistryRepairError(
                "receipt requires a satisfied behavior postcondition"
            )
        calculated = content_identity(self._payload_without_receipt_id())
        if self.receipt_id not in (None, ""):
            supplied = _text(self.receipt_id, "receipt_id")
            if supplied != calculated:
                raise RegistryRepairError("receipt_id mismatch")
        object.__setattr__(self, "receipt_id", calculated)

    def _payload_without_receipt_id(self) -> dict[str, Any]:
        return {
            "interface": self.INTERFACE,
            "evidence_id": REGISTRY_REPAIR_EVIDENCE,
            "operator_kind": self.operator_kind,
            "registry_key": self.registry_key,
            "before_table_id": self.before_table_id,
            "after_table_id": self.after_table_id,
            "before_hash": self.before_hash,
            "inverse_patch_id": self.inverse_patch_id,
            "postcondition": self.postcondition.to_dict(),
            "grants_write_authority": False,
        }

    def _payload(self) -> dict[str, Any]:
        return {**self._payload_without_receipt_id(), "receipt_id": self.receipt_id}


# ---------------------------------------------------------------------------
# Span freshness / ownership helpers
# ---------------------------------------------------------------------------


def span_is_fresh(span: SourceSpan, observed_before_hash: str) -> bool:
    """Return True when the observed before-hash matches the span evidence."""

    if not isinstance(span, SourceSpan):
        raise RegistryRepairError("span must be a SourceSpan")
    return span.before_hash == _hash(observed_before_hash, "observed_before_hash")


def _require_fresh_span(span: SourceSpan, observed_before_hash: str | None) -> None:
    if observed_before_hash is None:
        return
    if not span_is_fresh(span, observed_before_hash):
        raise RegistryRepairError("stale span: before_hash mismatch")


def evaluate_behavior_postcondition(
    *,
    registry_key: str,
    semantic_target: str,
    owner_ref: str,
    resolves: bool,
    unique_owner: bool,
    span_fresh: bool,
    inverse_patch_id: str,
    before_hash: str,
    after_table_id: str,
    anchor_count: int = 0,
) -> BehaviorPostcondition:
    """Build the behavioral postcondition that replaces count-only checks."""

    return BehaviorPostcondition(
        registry_key=registry_key,
        semantic_target=semantic_target,
        owner_ref=owner_ref,
        resolves=resolves,
        unique_owner=unique_owner,
        span_fresh=span_fresh,
        inverse_patch_id=inverse_patch_id,
        before_hash=before_hash,
        after_table_id=after_table_id,
        anchor_count=anchor_count,
        count_only_sufficient=False,
    )


def _inverse_patch_id(
    *,
    operator_kind: str,
    before_table_id: str,
    after_table_id: str,
    registry_key: str,
) -> str:
    return content_identity(
        {
            "inverse_of": operator_kind,
            "before_table_id": before_table_id,
            "after_table_id": after_table_id,
            "registry_key": registry_key,
        }
    )


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------


class _BaseRegistryOperator:
    """Shared descriptor binding for registry-family operators."""

    kind: ClassVar[RegistryOperatorKind]
    inverse_kind: ClassVar[str]
    registry_kind: ClassVar[OperatorKind]

    def __init__(self) -> None:
        registry = build_default_operator_registry()
        self.descriptor = registry.require_known(self.registry_kind)
        if self.descriptor.family is not OperatorFamily.REGISTRY:
            raise RegistryRepairError("registry family mismatch")
        if self.descriptor.kind is not self.registry_kind:
            raise RegistryRepairError("registry operator kind mismatch")

    @property
    def operator_id(self) -> str:
        return f"dcr-operator:{self.kind.value}@1"


class AddAliasOperator(_BaseRegistryOperator):
    """Add a missing alias whose owner and span are uniquely proved."""

    kind: ClassVar[RegistryOperatorKind] = RegistryOperatorKind.ADD_ALIAS
    inverse_kind: ClassVar[str] = "remove_alias"
    registry_kind: ClassVar[OperatorKind] = OperatorKind.ADD_ALIAS

    def applicability(
        self,
        table: AliasRegistry,
        binding: AliasBinding,
        *,
        observed_before_hash: str | None = None,
        expected_owner: str | None = None,
    ) -> tuple[bool, tuple[str, ...]]:
        if not isinstance(table, AliasRegistry):
            raise RegistryRepairError("table must be an AliasRegistry")
        if not isinstance(binding, AliasBinding):
            raise RegistryRepairError("binding must be an AliasBinding")
        reasons: list[str] = []
        try:
            _require_fresh_span(binding.span, observed_before_hash)
        except RegistryRepairError as exc:
            return False, (f"reject:{exc}",)
        if expected_owner not in (None, "", binding.owner_ref):
            return False, ("reject:wrong_owner",)
        existing = table.get(binding.alias_key)
        if existing is not None:
            if existing.content_id == binding.content_id:
                return True, ("already_bound", "idempotent")
            if existing.owner_ref != binding.owner_ref:
                return False, ("reject:wrong_owner", "reject:duplicate_owner_conflict")
            if existing.target_tool != binding.target_tool:
                return False, ("reject:duplicate_alias", "reject:target_conflict")
            return False, ("reject:duplicate_alias",)
        reasons.extend(("missing_alias", "unique_owner", "span_fresh"))
        return True, tuple(reasons)

    def preview(
        self,
        table: AliasRegistry,
        binding: AliasBinding,
        *,
        observed_before_hash: str | None = None,
        expected_owner: str | None = None,
    ) -> tuple[RegistryPreview, AliasRegistry]:
        applicable, reasons = self.applicability(
            table,
            binding,
            observed_before_hash=observed_before_hash,
            expected_owner=expected_owner,
        )
        if not applicable:
            raise RegistryRepairAbstention(
                f"{self.kind.value} not applicable: {', '.join(reasons)}"
            )
        after = table if "idempotent" in reasons else table.with_binding(binding)
        inverse_patch = _inverse_patch_id(
            operator_kind=self.kind.value,
            before_table_id=table.table_id,
            after_table_id=after.table_id,
            registry_key=binding.registry_key,
        )
        post = evaluate_behavior_postcondition(
            registry_key=binding.registry_key,
            semantic_target=binding.semantic_target,
            owner_ref=binding.owner_ref,
            resolves=after.contains(binding.alias_key)
            and after.get(binding.alias_key) is not None
            and after.get(binding.alias_key).target_tool == binding.target_tool,  # type: ignore[union-attr]
            unique_owner=True,
            span_fresh=span_is_fresh(
                binding.span, observed_before_hash or binding.span.before_hash
            ),
            inverse_patch_id=inverse_patch,
            before_hash=binding.span.before_hash,
            after_table_id=after.table_id,
            anchor_count=1,
        )
        if not post.behavior_satisfied:
            raise RegistryRepairError("behavior postcondition not satisfied")
        preview = RegistryPreview(
            operator_kind=self.kind.value,
            before_table_id=table.table_id,
            after_table_id=after.table_id,
            registry_key=binding.registry_key,
            semantic_target=binding.semantic_target,
            owner_ref=binding.owner_ref,
            before_hash=binding.span.before_hash,
            inverse_kind=self.inverse_kind,
            inverse_patch_id=inverse_patch,
            applicable=True,
            reason_codes=reasons,
            postcondition=post,
        )
        return preview, after

    def apply(
        self,
        table: AliasRegistry,
        binding: AliasBinding,
        **kwargs: Any,
    ) -> tuple[AliasRegistry, RegistryPreview]:
        preview, after = self.preview(table, binding, **kwargs)
        return after, preview

    def inverse(self, table: AliasRegistry, preview: RegistryPreview) -> AliasRegistry:
        if not isinstance(preview, RegistryPreview):
            raise RegistryRepairError("preview must be a RegistryPreview")
        if preview.operator_kind != self.kind.value:
            raise RegistryRepairError("preview operator_kind mismatch")
        if table.table_id != preview.after_table_id:
            raise RegistryRepairError("inverse requires the post-apply table")
        if preview.before_table_id == preview.after_table_id:
            return table
        alias_key = preview.registry_key.removeprefix("alias:")
        restored = table.without_key(alias_key)
        if restored.table_id != preview.before_table_id:
            # Fallback: strip by semantic registry key match.
            for item in table.bindings:
                if item.registry_key == preview.registry_key:
                    restored = table.without_key(item.alias_key)
                    break
        return restored


class RemoveAliasOperator(_BaseRegistryOperator):
    """Remove an existing alias binding (inverse of add)."""

    kind: ClassVar[RegistryOperatorKind] = RegistryOperatorKind.REMOVE_ALIAS
    inverse_kind: ClassVar[str] = "add_alias"
    registry_kind: ClassVar[OperatorKind] = OperatorKind.REMOVE_ALIAS

    def preview(
        self,
        table: AliasRegistry,
        alias_key: str,
        *,
        expected_owner: str | None = None,
        observed_before_hash: str | None = None,
    ) -> tuple[RegistryPreview, AliasRegistry]:
        if not isinstance(table, AliasRegistry):
            raise RegistryRepairError("table must be an AliasRegistry")
        key = _alias_key(alias_key)
        existing = table.get(key)
        if existing is None:
            # Idempotent absence: table unchanged.  A placeholder owner is used
            # only for the closed receipt shape; no ownership claim is made.
            empty_span_hash = _digest({"absent": key})
            placeholder_owner = "ipfs_accelerate_py.agent_supervisor.autonomous_repair.operators.registry_repairs:absent_alias"
            inverse_patch = _inverse_patch_id(
                operator_kind=self.kind.value,
                before_table_id=table.table_id,
                after_table_id=table.table_id,
                registry_key=f"alias:{key}",
            )
            post = evaluate_behavior_postcondition(
                registry_key=f"alias:{key}",
                semantic_target=f"absent:{key}",
                owner_ref=placeholder_owner,
                resolves=not table.contains(key),
                unique_owner=True,
                span_fresh=True,
                inverse_patch_id=inverse_patch,
                before_hash=empty_span_hash,
                after_table_id=table.table_id,
                anchor_count=0,
            )
            preview = RegistryPreview(
                operator_kind=self.kind.value,
                before_table_id=table.table_id,
                after_table_id=table.table_id,
                registry_key=f"alias:{key}",
                semantic_target=f"absent:{key}",
                owner_ref=placeholder_owner,
                before_hash=empty_span_hash,
                inverse_kind=self.inverse_kind,
                inverse_patch_id=inverse_patch,
                applicable=True,
                reason_codes=("already_absent", "idempotent"),
                postcondition=post,
            )
            return preview, table

        _require_fresh_span(existing.span, observed_before_hash)
        if expected_owner not in (None, "", existing.owner_ref):
            raise RegistryRepairError("wrong owner for alias removal")
        after = table.without_key(key)
        inverse_patch = _inverse_patch_id(
            operator_kind=self.kind.value,
            before_table_id=table.table_id,
            after_table_id=after.table_id,
            registry_key=existing.registry_key,
        )
        post = evaluate_behavior_postcondition(
            registry_key=existing.registry_key,
            semantic_target=existing.semantic_target,
            owner_ref=existing.owner_ref,
            resolves=not after.contains(key),
            unique_owner=True,
            span_fresh=span_is_fresh(
                existing.span, observed_before_hash or existing.span.before_hash
            ),
            inverse_patch_id=inverse_patch,
            before_hash=existing.span.before_hash,
            after_table_id=after.table_id,
            anchor_count=0,
        )
        preview = RegistryPreview(
            operator_kind=self.kind.value,
            before_table_id=table.table_id,
            after_table_id=after.table_id,
            registry_key=existing.registry_key,
            semantic_target=existing.semantic_target,
            owner_ref=existing.owner_ref,
            before_hash=existing.span.before_hash,
            inverse_kind=self.inverse_kind,
            inverse_patch_id=inverse_patch,
            applicable=True,
            reason_codes=("alias_present", "span_fresh"),
            postcondition=post,
        )
        return preview, after

    def apply(
        self,
        table: AliasRegistry,
        alias_key: str,
        **kwargs: Any,
    ) -> tuple[AliasRegistry, RegistryPreview]:
        preview, after = self.preview(table, alias_key, **kwargs)
        return after, preview

    def inverse(
        self,
        table: AliasRegistry,
        preview: RegistryPreview,
        *,
        restored_binding: AliasBinding | None = None,
    ) -> AliasRegistry:
        if not isinstance(preview, RegistryPreview):
            raise RegistryRepairError("preview must be a RegistryPreview")
        if preview.operator_kind != self.kind.value:
            raise RegistryRepairError("preview operator_kind mismatch")
        if table.table_id != preview.after_table_id:
            raise RegistryRepairError("inverse requires the post-apply table")
        if preview.before_table_id == preview.after_table_id:
            return table
        if restored_binding is None:
            raise RegistryRepairAbstention(
                "remove_alias inverse requires the prior AliasBinding snapshot"
            )
        return table.with_binding(restored_binding)


class RenameAliasOperator(_BaseRegistryOperator):
    """Rename an alias key without changing its semantic target/owner."""

    kind: ClassVar[RegistryOperatorKind] = RegistryOperatorKind.RENAME_ALIAS
    inverse_kind: ClassVar[str] = "rename_alias"
    registry_kind: ClassVar[OperatorKind] = OperatorKind.RENAME_ALIAS

    def preview(
        self,
        table: AliasRegistry,
        old_key: str,
        new_key: str,
        *,
        expected_owner: str | None = None,
        observed_before_hash: str | None = None,
    ) -> tuple[RegistryPreview, AliasRegistry]:
        if not isinstance(table, AliasRegistry):
            raise RegistryRepairError("table must be an AliasRegistry")
        old = _alias_key(old_key, "old_key")
        new = _alias_key(new_key, "new_key")
        existing = table.get(old)
        if existing is None:
            raise RegistryRepairAbstention(f"alias not found for rename: {old}")
        _require_fresh_span(existing.span, observed_before_hash)
        if expected_owner not in (None, "", existing.owner_ref):
            raise RegistryRepairError("wrong owner for alias rename")
        if old == new:
            inverse_patch = _inverse_patch_id(
                operator_kind=self.kind.value,
                before_table_id=table.table_id,
                after_table_id=table.table_id,
                registry_key=existing.registry_key,
            )
            post = evaluate_behavior_postcondition(
                registry_key=existing.registry_key,
                semantic_target=existing.semantic_target,
                owner_ref=existing.owner_ref,
                resolves=True,
                unique_owner=True,
                span_fresh=True,
                inverse_patch_id=inverse_patch,
                before_hash=existing.span.before_hash,
                after_table_id=table.table_id,
                anchor_count=1,
            )
            preview = RegistryPreview(
                operator_kind=self.kind.value,
                before_table_id=table.table_id,
                after_table_id=table.table_id,
                registry_key=existing.registry_key,
                semantic_target=existing.semantic_target,
                owner_ref=existing.owner_ref,
                before_hash=existing.span.before_hash,
                inverse_kind=self.inverse_kind,
                inverse_patch_id=inverse_patch,
                applicable=True,
                reason_codes=("already_named", "idempotent"),
                postcondition=post,
            )
            return preview, table
        if table.contains(new):
            raise RegistryRepairError(f"duplicate alias on rename target: {new}")
        after = table.rename_key(old, new)
        renamed = after.get(new)
        assert renamed is not None
        inverse_patch = _inverse_patch_id(
            operator_kind=self.kind.value,
            before_table_id=table.table_id,
            after_table_id=after.table_id,
            registry_key=renamed.registry_key,
        )
        post = evaluate_behavior_postcondition(
            registry_key=renamed.registry_key,
            semantic_target=renamed.semantic_target,
            owner_ref=renamed.owner_ref,
            resolves=after.contains(new) and not after.contains(old),
            unique_owner=True,
            span_fresh=span_is_fresh(
                existing.span, observed_before_hash or existing.span.before_hash
            ),
            inverse_patch_id=inverse_patch,
            before_hash=existing.span.before_hash,
            after_table_id=after.table_id,
            anchor_count=1,
        )
        preview = RegistryPreview(
            operator_kind=self.kind.value,
            before_table_id=table.table_id,
            after_table_id=after.table_id,
            registry_key=renamed.registry_key,
            semantic_target=renamed.semantic_target,
            owner_ref=renamed.owner_ref,
            before_hash=existing.span.before_hash,
            inverse_kind=self.inverse_kind,
            inverse_patch_id=inverse_patch,
            applicable=True,
            reason_codes=("renamed", "unique_owner", "span_fresh"),
            postcondition=post,
        )
        return preview, after

    def apply(
        self,
        table: AliasRegistry,
        old_key: str,
        new_key: str,
        **kwargs: Any,
    ) -> tuple[AliasRegistry, RegistryPreview]:
        preview, after = self.preview(table, old_key, new_key, **kwargs)
        return after, preview

    def inverse(
        self,
        table: AliasRegistry,
        preview: RegistryPreview,
        *,
        old_key: str,
        new_key: str,
    ) -> AliasRegistry:
        if not isinstance(preview, RegistryPreview):
            raise RegistryRepairError("preview must be a RegistryPreview")
        if preview.operator_kind != self.kind.value:
            raise RegistryRepairError("preview operator_kind mismatch")
        if table.table_id != preview.after_table_id:
            raise RegistryRepairError("inverse requires the post-apply table")
        if preview.before_table_id == preview.after_table_id:
            return table
        # Symmetric rename back.
        return table.rename_key(_alias_key(new_key, "new_key"), _alias_key(old_key, "old_key"))


class BindRegistrationOperator(_BaseRegistryOperator):
    """Bind a missing registration whose handler already exists uniquely."""

    kind: ClassVar[RegistryOperatorKind] = RegistryOperatorKind.BIND_REGISTRATION
    inverse_kind: ClassVar[str] = "unbind_registration"
    registry_kind: ClassVar[OperatorKind] = OperatorKind.BIND_REGISTRATION

    def applicability(
        self,
        table: RegistrationTable,
        binding: RegistrationBinding,
        *,
        observed_before_hash: str | None = None,
        expected_owner: str | None = None,
    ) -> tuple[bool, tuple[str, ...]]:
        if not isinstance(table, RegistrationTable):
            raise RegistryRepairError("table must be a RegistrationTable")
        if not isinstance(binding, RegistrationBinding):
            raise RegistryRepairError("binding must be a RegistrationBinding")
        try:
            _require_fresh_span(binding.span, observed_before_hash)
        except RegistryRepairError as exc:
            return False, (f"reject:{exc}",)
        if expected_owner not in (None, "", binding.owner_ref):
            return False, ("reject:wrong_owner",)
        if not table.handler_is_known(binding.handler_ref):
            return False, ("abstain:handler_not_known",)
        if binding.owner_ref != binding.handler_ref and binding.owner_ref not in table.known_handlers:
            # Owner may be a package-level owner distinct from the handler, but
            # the handler itself must still be the unique known semantic target.
            pass
        existing = table.get(binding.tool_name)
        if existing is not None:
            if existing.content_id == binding.content_id:
                return True, ("already_bound", "idempotent")
            if existing.owner_ref != binding.owner_ref:
                return False, ("reject:wrong_owner", "reject:duplicate_owner_conflict")
            if existing.handler_ref != binding.handler_ref:
                return False, ("reject:duplicate_registration", "reject:handler_conflict")
            return False, ("reject:duplicate_registration",)
        return True, ("missing_registration", "unique_handler", "span_fresh")

    def preview(
        self,
        table: RegistrationTable,
        binding: RegistrationBinding,
        *,
        observed_before_hash: str | None = None,
        expected_owner: str | None = None,
    ) -> tuple[RegistryPreview, RegistrationTable]:
        applicable, reasons = self.applicability(
            table,
            binding,
            observed_before_hash=observed_before_hash,
            expected_owner=expected_owner,
        )
        if not applicable:
            if any(code.startswith("abstain:") for code in reasons):
                raise RegistryRepairAbstention(
                    f"{self.kind.value} not applicable: {', '.join(reasons)}"
                )
            raise RegistryRepairError(
                f"{self.kind.value} not applicable: {', '.join(reasons)}"
            )
        after = table if "idempotent" in reasons else table.with_binding(binding)
        inverse_patch = _inverse_patch_id(
            operator_kind=self.kind.value,
            before_table_id=table.table_id,
            after_table_id=after.table_id,
            registry_key=binding.registry_key,
        )
        bound = after.get(binding.tool_name)
        post = evaluate_behavior_postcondition(
            registry_key=binding.registry_key,
            semantic_target=binding.semantic_target,
            owner_ref=binding.owner_ref,
            resolves=bound is not None and bound.handler_ref == binding.handler_ref,
            unique_owner=True,
            span_fresh=span_is_fresh(
                binding.span, observed_before_hash or binding.span.before_hash
            ),
            inverse_patch_id=inverse_patch,
            before_hash=binding.span.before_hash,
            after_table_id=after.table_id,
            anchor_count=1,
        )
        if not post.behavior_satisfied:
            raise RegistryRepairError("behavior postcondition not satisfied")
        preview = RegistryPreview(
            operator_kind=self.kind.value,
            before_table_id=table.table_id,
            after_table_id=after.table_id,
            registry_key=binding.registry_key,
            semantic_target=binding.semantic_target,
            owner_ref=binding.owner_ref,
            before_hash=binding.span.before_hash,
            inverse_kind=self.inverse_kind,
            inverse_patch_id=inverse_patch,
            applicable=True,
            reason_codes=reasons,
            postcondition=post,
        )
        return preview, after

    def apply(
        self,
        table: RegistrationTable,
        binding: RegistrationBinding,
        **kwargs: Any,
    ) -> tuple[RegistrationTable, RegistryPreview]:
        preview, after = self.preview(table, binding, **kwargs)
        return after, preview

    def inverse(
        self, table: RegistrationTable, preview: RegistryPreview
    ) -> RegistrationTable:
        if not isinstance(preview, RegistryPreview):
            raise RegistryRepairError("preview must be a RegistryPreview")
        if preview.operator_kind != self.kind.value:
            raise RegistryRepairError("preview operator_kind mismatch")
        if table.table_id != preview.after_table_id:
            raise RegistryRepairError("inverse requires the post-apply table")
        if preview.before_table_id == preview.after_table_id:
            return table
        tool_name = preview.registry_key.removeprefix("registration:")
        restored = table.without_tool(tool_name)
        if restored.table_id != preview.before_table_id:
            for item in table.bindings:
                if item.registry_key == preview.registry_key:
                    restored = table.without_tool(item.tool_name)
                    break
        return restored


class DisambiguateAnchorOperator(_BaseRegistryOperator):
    """Collapse duplicate anchors only under a unique typed ownership edge.

    Conflict policy (fail-closed): never choose among multiple anchors by
    lexical score.  Ambiguity abstains unless a unique typed edge proves
    ownership.  Success is measured by behavior postconditions, not by
    ``anchor_count == 1`` alone.
    """

    kind: ClassVar[RegistryOperatorKind] = RegistryOperatorKind.DISAMBIGUATE_ANCHOR
    inverse_kind: ClassVar[str] = "restore_anchors"
    registry_kind: ClassVar[OperatorKind] = OperatorKind.DISAMBIGUATE_ANCHOR

    def resolve_unique_owner(
        self,
        table: AnchorTable,
        registry_key: str,
        *,
        observed_before_hash: str | None = None,
        required_edge: OwnershipEdgeKind | None = None,
    ) -> tuple[AnchorRecord | None, tuple[str, ...]]:
        if not isinstance(table, AnchorTable):
            raise RegistryRepairError("table must be an AnchorTable")
        key = _text(registry_key, "registry_key")
        candidates = table.for_key(key)
        if not candidates:
            return None, ("abstain:no_anchors",)
        # Stale spans are hard rejects, not candidates for ranking.
        fresh: list[AnchorRecord] = []
        for item in candidates:
            if observed_before_hash is not None and not span_is_fresh(
                item.span, observed_before_hash
            ):
                continue
            fresh.append(item)
        if observed_before_hash is not None and not fresh:
            return None, ("reject:stale_span",)
        pool = fresh if fresh else list(candidates)
        if required_edge is not None:
            edge = _enum(required_edge, OwnershipEdgeKind, "required_edge")
            pool = [item for item in pool if item.ownership_edge is edge]
            if not pool:
                return None, ("abstain:no_matching_ownership_edge",)
        owners = {item.owner_ref for item in pool}
        targets = {item.semantic_target for item in pool}
        if len(owners) != 1:
            # Never fall back to lexical ranking of anchor_id / path.
            return None, ("abstain:ambiguous_owners", "reject:lexical_score_forbidden")
        if len(targets) != 1:
            return None, ("abstain:ambiguous_semantic_targets",)
        # Prefer an already-selected anchor when unique; otherwise the sole
        # ownership class may retain any member (identity is by owner/target).
        selected = [item for item in pool if item.selected]
        if len(selected) == 1:
            winner = selected[0]
            return winner, ("unique_typed_edge", "already_selected")
        if len(selected) > 1:
            return None, ("abstain:multiple_selected", "reject:lexical_score_forbidden")
        # All candidates share owner+target; pick the stable minimum by
        # content_id (content identity), never by bare lexical path score as
        # an ownership rule.  Content identity is deterministic and not an
        # ownership proof — ownership was already proved unique above.
        winner = min(pool, key=lambda item: item.content_id)
        return winner, ("unique_typed_edge", "content_stable_representative")

    def preview(
        self,
        table: AnchorTable,
        registry_key: str,
        *,
        observed_before_hash: str | None = None,
        required_edge: OwnershipEdgeKind | None = None,
        expected_owner: str | None = None,
    ) -> tuple[RegistryPreview, AnchorTable]:
        winner, reasons = self.resolve_unique_owner(
            table,
            registry_key,
            observed_before_hash=observed_before_hash,
            required_edge=required_edge,
        )
        if winner is None:
            if any(code.startswith("reject:") for code in reasons) and not any(
                code.startswith("abstain:") for code in reasons
            ):
                raise RegistryRepairError(
                    f"{self.kind.value} not applicable: {', '.join(reasons)}"
                )
            raise RegistryRepairAbstention(
                f"{self.kind.value} not applicable: {', '.join(reasons)}"
            )
        if expected_owner not in (None, "", winner.owner_ref):
            raise RegistryRepairError("wrong owner for anchor disambiguation")

        key = _text(registry_key, "registry_key")
        current_selected = table.selected_for_key(key)
        if (
            current_selected is not None
            and current_selected.owner_ref == winner.owner_ref
            and current_selected.semantic_target == winner.semantic_target
            and len(table.for_key(key)) == 1
            and current_selected.selected
        ):
            after = table
            reason_codes = ("already_unique", "idempotent", *reasons)
        else:
            retained = AnchorRecord(
                anchor_id=winner.anchor_id,
                kind=winner.kind,
                registry_key=winner.registry_key,
                owner_ref=winner.owner_ref,
                semantic_target=winner.semantic_target,
                span=winner.span,
                ownership_edge=winner.ownership_edge,
                selected=True,
            )
            after = table.replace_key(key, (retained,))
            reason_codes = ("disambiguated", *reasons)

        inverse_patch = _inverse_patch_id(
            operator_kind=self.kind.value,
            before_table_id=table.table_id,
            after_table_id=after.table_id,
            registry_key=key,
        )
        retained_after = after.for_key(key)
        post = evaluate_behavior_postcondition(
            registry_key=key,
            semantic_target=winner.semantic_target,
            owner_ref=winner.owner_ref,
            resolves=len(retained_after) == 1
            and retained_after[0].selected
            and retained_after[0].semantic_target == winner.semantic_target,
            unique_owner=len({item.owner_ref for item in retained_after}) == 1,
            span_fresh=span_is_fresh(
                winner.span, observed_before_hash or winner.span.before_hash
            ),
            inverse_patch_id=inverse_patch,
            before_hash=winner.span.before_hash,
            after_table_id=after.table_id,
            # Count is recorded but never sufficient by itself.
            anchor_count=len(retained_after),
        )
        if not post.behavior_satisfied:
            raise RegistryRepairError(
                "behavior postcondition not satisfied "
                "(anchor-count-only validation is insufficient)"
            )
        preview = RegistryPreview(
            operator_kind=self.kind.value,
            before_table_id=table.table_id,
            after_table_id=after.table_id,
            registry_key=key,
            semantic_target=winner.semantic_target,
            owner_ref=winner.owner_ref,
            before_hash=winner.span.before_hash,
            inverse_kind=self.inverse_kind,
            inverse_patch_id=inverse_patch,
            applicable=True,
            reason_codes=tuple(reason_codes),
            postcondition=post,
        )
        return preview, after

    def apply(
        self,
        table: AnchorTable,
        registry_key: str,
        **kwargs: Any,
    ) -> tuple[AnchorTable, RegistryPreview]:
        preview, after = self.preview(table, registry_key, **kwargs)
        return after, preview

    def inverse(
        self,
        table: AnchorTable,
        preview: RegistryPreview,
        *,
        prior_anchors: Sequence[AnchorRecord],
    ) -> AnchorTable:
        if not isinstance(preview, RegistryPreview):
            raise RegistryRepairError("preview must be a RegistryPreview")
        if preview.operator_kind != self.kind.value:
            raise RegistryRepairError("preview operator_kind mismatch")
        if table.table_id != preview.after_table_id:
            raise RegistryRepairError("inverse requires the post-apply table")
        if preview.before_table_id == preview.after_table_id:
            return table
        restored = table.replace_key(preview.registry_key, prior_anchors)
        return restored


@dataclass(frozen=True)
class RegistryRepairOperators:
    """Facade bundling the DCR-041 registry-family operators."""

    INTERFACE: ClassVar[str] = REGISTRY_REPAIR_OPERATORS_INTERFACE
    EVIDENCE_ID: ClassVar[str] = REGISTRY_REPAIR_EVIDENCE

    add_alias: AddAliasOperator
    remove_alias: RemoveAliasOperator
    rename_alias: RenameAliasOperator
    bind_registration: BindRegistrationOperator
    disambiguate_anchor: DisambiguateAnchorOperator

    @classmethod
    def build(cls) -> "RegistryRepairOperators":
        return cls(
            add_alias=AddAliasOperator(),
            remove_alias=RemoveAliasOperator(),
            rename_alias=RenameAliasOperator(),
            bind_registration=BindRegistrationOperator(),
            disambiguate_anchor=DisambiguateAnchorOperator(),
        )


def build_registry_repair_operators() -> RegistryRepairOperators:
    """Return the sealed DCR-041 operator bundle."""

    return RegistryRepairOperators.build()


def make_span(
    path: str,
    *,
    start_offset: int = 0,
    end_offset: int = 1,
    body: str | None = None,
    before_hash: str | None = None,
) -> SourceSpan:
    """Helper to build a SourceSpan with a deterministic before-hash."""

    digest = before_hash
    if digest is None:
        digest = _digest(
            {
                "path": path,
                "start_offset": start_offset,
                "end_offset": end_offset,
                "body": body or "",
            }
        )
    return SourceSpan(
        path=path,
        start_offset=start_offset,
        end_offset=end_offset,
        before_hash=digest,
    )


def registry_operator_vectors() -> dict[str, Any]:
    """Compact vector catalogue for admission/fixture generation."""

    ops = build_registry_repair_operators()
    span = make_span("pkg/tools/registry.py", body="alias:demo")
    alias = AliasBinding(
        alias_key="demo_alias",
        target_tool="demo_tool",
        owner_ref="pkg.tools.demo:handle_demo",
        span=span,
    )
    alias_table, alias_preview = ops.add_alias.apply(AliasRegistry.empty(), alias)
    handler = "pkg.tools.demo:handle_demo"
    reg_table = RegistrationTable.empty(known_handlers=(handler,))
    registration = RegistrationBinding(
        tool_name="demo_tool",
        owner_ref=handler,
        handler_ref=handler,
        input_schema_ref="schema:demo_tool/input@1",
        span=span,
    )
    reg_table, reg_preview = ops.bind_registration.apply(reg_table, registration)
    anchors = AnchorTable(
        anchors=(
            AnchorRecord(
                anchor_id="anchor:a",
                kind=AnchorKind.REGISTRATION,
                registry_key="registration:demo_tool",
                owner_ref=handler,
                semantic_target=f"handler:{handler}",
                span=span,
                ownership_edge=OwnershipEdgeKind.REGISTRATION_TO_HANDLER,
            ),
            AnchorRecord(
                anchor_id="anchor:b",
                kind=AnchorKind.REGISTRATION,
                registry_key="registration:demo_tool",
                owner_ref=handler,
                semantic_target=f"handler:{handler}",
                span=make_span("pkg/tools/registry.py", start_offset=10, end_offset=20, body="dup"),
                ownership_edge=OwnershipEdgeKind.REGISTRATION_TO_HANDLER,
            ),
        )
    )
    anchor_table, anchor_preview = ops.disambiguate_anchor.apply(
        anchors, "registration:demo_tool"
    )
    payload = {
        "schema": REGISTRY_REPAIR_SCHEMA,
        "interface": REGISTRY_REPAIR_OPERATORS_INTERFACE,
        "evidence_id": REGISTRY_REPAIR_EVIDENCE,
        "operators": [
            ops.add_alias.operator_id,
            ops.remove_alias.operator_id,
            ops.rename_alias.operator_id,
            ops.bind_registration.operator_id,
            ops.disambiguate_anchor.operator_id,
        ],
        "alias_table_id": alias_table.table_id,
        "registration_table_id": reg_table.table_id,
        "anchor_table_id": anchor_table.table_id,
        "alias_preview_id": alias_preview.content_id,
        "registration_preview_id": reg_preview.content_id,
        "anchor_preview_id": anchor_preview.content_id,
        "behavior_postcondition_required": True,
        "count_only_sufficient": False,
    }
    payload["vector_digest"] = _digest(payload)
    return payload


def make_registry_repair_receipt(preview: RegistryPreview) -> RegistryRepairReceipt:
    """Materialize a receipt from a satisfied preview."""

    if preview.postcondition is None or not preview.postcondition.behavior_satisfied:
        raise RegistryRepairError(
            "receipt requires a satisfied behavior postcondition on the preview"
        )
    return RegistryRepairReceipt(
        operator_kind=preview.operator_kind,
        registry_key=preview.registry_key,
        before_table_id=preview.before_table_id,
        after_table_id=preview.after_table_id,
        before_hash=preview.before_hash,
        inverse_patch_id=preview.inverse_patch_id,
        postcondition=preview.postcondition,
    )


__all__ = (
    "REGISTRY_REPAIR_EVIDENCE",
    "REGISTRY_REPAIR_OPERATORS_INTERFACE",
    "AddAliasOperator",
    "AliasBinding",
    "AliasRegistry",
    "AnchorKind",
    "AnchorRecord",
    "AnchorTable",
    "BehaviorPostcondition",
    "BindRegistrationOperator",
    "DisambiguateAnchorOperator",
    "OwnershipEdgeKind",
    "RegistrationBinding",
    "RegistrationTable",
    "RegistryOperatorKind",
    "RegistryPreview",
    "RegistryRepairAbstention",
    "RegistryRepairError",
    "RegistryRepairOperators",
    "RegistryRepairReceipt",
    "RemoveAliasOperator",
    "RenameAliasOperator",
    "SourceSpan",
    "build_registry_repair_operators",
    "evaluate_behavior_postcondition",
    "make_registry_repair_receipt",
    "make_span",
    "registry_operator_vectors",
    "span_is_fresh",
)
