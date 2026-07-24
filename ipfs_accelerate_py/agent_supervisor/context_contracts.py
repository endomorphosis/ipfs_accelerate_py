"""Shared, immutable context contracts for agent-supervisor boundaries.

The supervisor passes context between analysis, planning, implementation, and
validation stages.  These contracts deliberately carry a small invariant core
and compact evidence references, rather than an unbounded prompt, source tree,
or recursive evidence graph.

All records are versioned, frozen, content addressed, and serialized with the
repository's canonical DAG-JSON encoder.  Open-ended values are accepted only
after enforcing node-count, UTF-8 byte, and nesting-depth limits.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any, ClassVar

from .formal_verification_contracts import (
    CanonicalContract,
    ContractValidationError,
    canonical_json_bytes,
)


CONTEXT_CONTRACT_VERSION = 1
CONTRACT_VERSION = CONTEXT_CONTRACT_VERSION
SCHEMA_VERSION = CONTEXT_CONTRACT_VERSION

CONTEXT_BUDGET_SCHEMA = "ipfs_accelerate_py/agent-supervisor/context-budget@1"
CONTEXT_REFERENCE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/context-reference@1"
)
CONTEXT_CAPSULE_SCHEMA = "ipfs_accelerate_py/agent-supervisor/context-capsule@1"

ABSOLUTE_MAX_CONTEXT_BYTES = 1_048_576
ABSOLUTE_MAX_ITEM_BYTES = 65_536
ABSOLUTE_MAX_ITEMS = 4_096
ABSOLUTE_MAX_DEPTH = 32
ABSOLUTE_MAX_TEXT_BYTES = 65_536
_PATH_KEYS = frozenset(
    {
        "path",
        "paths",
        "repository_path",
        "repository_paths",
        "target_path",
        "target_paths",
        "artifact_path",
        "artifact_paths",
    }
)


class ContextContractError(ContractValidationError):
    """Base error for malformed shared context contracts."""


class ContextBoundsError(ContextContractError):
    """Raised when a context value exceeds a declared resource bound."""


class ContextIdentityError(ContextContractError):
    """Raised when a claimed context identity does not match its payload."""


class _ContextCanonicalContract(CanonicalContract):
    """Canonical mixin whose decoding failures retain the context error type."""

    @classmethod
    def from_json(cls, payload: str) -> "_ContextCanonicalContract":
        try:
            value = json.loads(payload)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ContextContractError("context contract JSON is malformed") from exc
        if not isinstance(value, Mapping):
            raise ContextContractError(
                "context contract JSON must contain an object"
            )
        decoder = getattr(cls, "from_dict", None)
        if decoder is None:
            raise ContextContractError(
                f"{cls.__name__} does not support from_dict"
            )
        return decoder(value)


class ContextTier(str, Enum):
    """Trust and disclosure tier of a context reference."""

    INVARIANT = "invariant"
    EVIDENCE = "evidence"
    SUGGESTION = "suggestion"
    EXPANSION = "expansion"


def _text(
    value: Any,
    name: str,
    *,
    required: bool = True,
    max_bytes: int = ABSOLUTE_MAX_TEXT_BYTES,
) -> str:
    if not isinstance(value, str):
        raise ContextContractError(f"{name} must be a string")
    result = value.strip()
    if required and not result:
        raise ContextContractError(f"{name} must not be empty")
    if "\x00" in result:
        raise ContextContractError(f"{name} must not contain NUL")
    if len(result.encode("utf-8")) > max_bytes:
        raise ContextBoundsError(f"{name} exceeds {max_bytes} UTF-8 bytes")
    return result


def _nonnegative(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ContextContractError(f"{name} must be a non-negative integer")
    return value


def _positive(value: Any, name: str) -> int:
    result = _nonnegative(value, name)
    if result < 1:
        raise ContextContractError(f"{name} must be at least 1")
    return result


def _enum(value: Any, enum_type: type[Enum], name: str) -> Any:
    try:
        return enum_type(str(getattr(value, "value", value)))
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(item.value for item in enum_type)
        raise ContextContractError(f"{name} must be one of: {allowed}") from exc


def _schema(payload: Mapping[str, Any], expected: str) -> None:
    if not isinstance(payload, Mapping):
        raise ContextContractError("context contract payload must be an object")
    supplied = payload.get("schema")
    if supplied not in (None, "", expected):
        raise ContextContractError(
            f"unsupported context schema {supplied!r}; expected {expected}"
        )
    version = payload.get("contract_version", payload.get("schema_version"))
    if version not in (None, CONTEXT_CONTRACT_VERSION):
        raise ContextContractError("unsupported context contract version")


def _reject_unknown(
    payload: Mapping[str, Any], allowed: Iterable[str], noun: str
) -> None:
    if set(payload).difference(allowed):
        raise ContextContractError(
            f"{noun} contains unsupported fields; rebuild its canonical payload"
        )


def _relative_path(value: Any, name: str, *, required: bool = False) -> str:
    result = _text(value, name, required=required).replace("\\", "/")
    if not result:
        return ""
    candidate = PurePosixPath(result)
    if (
        candidate.is_absolute()
        or ".." in candidate.parts
        or (len(candidate.parts) > 0 and candidate.parts[0].endswith(":"))
    ):
        raise ContextContractError(f"{name} must be repository-relative")
    normalized = candidate.as_posix()
    if normalized in ("", "."):
        if required:
            raise ContextContractError(f"{name} must not be empty")
        return ""
    return normalized.removeprefix("./")


def _freeze_canonical(
    value: Any,
    *,
    name: str,
    max_depth: int,
    max_items: int,
    max_text_bytes: int,
) -> Any:
    """Validate and deeply freeze one DAG-JSON-compatible value."""

    seen = 0

    def visit(item: Any, depth: int, key_name: str = "") -> Any:
        nonlocal seen
        seen += 1
        if seen > max_items:
            raise ContextBoundsError(f"{name} exceeds its item-count limit")
        if depth > max_depth:
            raise ContextBoundsError(f"{name} exceeds its nesting-depth limit")
        if item is None or isinstance(item, bool):
            return item
        if isinstance(item, int) and not isinstance(item, bool):
            return item
        if isinstance(item, str):
            result = _text(
                item,
                name,
                required=False,
                max_bytes=max_text_bytes,
            )
            if key_name in _PATH_KEYS and not key_name.endswith("paths"):
                return _relative_path(result, key_name, required=False)
            return result
        if isinstance(item, Enum):
            return visit(item.value, depth, key_name)
        if isinstance(item, Mapping):
            if not all(isinstance(key, str) for key in item):
                raise ContextContractError(f"{name} object keys must be strings")
            frozen: dict[str, Any] = {}
            for key in sorted(item):
                normalized_key = _text(
                    key, f"{name} key", max_bytes=max_text_bytes
                )
                raw = item[key]
                if normalized_key in _PATH_KEYS and normalized_key.endswith("paths"):
                    if isinstance(raw, str) or not isinstance(raw, Sequence):
                        raise ContextContractError(
                            f"{normalized_key} must be a sequence of paths"
                        )
                    frozen[normalized_key] = tuple(
                        _relative_path(member, normalized_key) for member in raw
                    )
                else:
                    frozen[normalized_key] = visit(
                        raw, depth + 1, normalized_key
                    )
            return MappingProxyType(frozen)
        if isinstance(item, Sequence) and not isinstance(
            item, (str, bytes, bytearray, memoryview)
        ):
            return tuple(visit(member, depth + 1, key_name) for member in item)
        raise ContextContractError(
            f"{name} contains unsupported value type {type(item).__name__}"
        )

    return visit(value, 0)


def _identity(payload: Mapping[str, Any], actual: str, noun: str) -> None:
    claimed = payload.get("content_id") or payload.get("capsule_id")
    if claimed not in (None, "") and claimed != actual:
        raise ContextIdentityError(f"{noun} identity does not match payload")


def _coerce_references(value: Any, name: str) -> tuple["ContextReference", ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes, bytearray, memoryview)) or not isinstance(
        value, Sequence
    ):
        raise ContextContractError(f"{name} must be a sequence")
    result: list[ContextReference] = []
    for item in value:
        if isinstance(item, ContextReference):
            result.append(item)
        elif isinstance(item, Mapping):
            result.append(ContextReference.from_dict(item))
        else:
            raise ContextContractError(f"{name} contains an invalid reference")
    by_id: dict[str, ContextReference] = {}
    for item in result:
        if item.reference_id in by_id:
            raise ContextContractError(f"{name} contains duplicate reference IDs")
        by_id[item.reference_id] = item
    return tuple(by_id[key] for key in sorted(by_id))


@dataclass(frozen=True)
class ContextBudget(_ContextCanonicalContract):
    """Provider-aware input and structural limits for one context capsule."""

    SCHEMA: ClassVar[str] = CONTEXT_BUDGET_SCHEMA

    max_input_tokens: int = 8_192
    reserved_output_tokens: int = 2_048
    reserved_tool_tokens: int = 512
    max_items: int = 128
    max_item_bytes: int = 16_384
    max_serialized_bytes: int = 262_144
    max_depth: int = 8
    max_text_bytes: int = 8_192

    def __post_init__(self) -> None:
        for name in (
            "max_input_tokens",
            "max_items",
            "max_item_bytes",
            "max_serialized_bytes",
            "max_depth",
            "max_text_bytes",
        ):
            object.__setattr__(self, name, _positive(getattr(self, name), name))
        for name in ("reserved_output_tokens", "reserved_tool_tokens"):
            object.__setattr__(
                self, name, _nonnegative(getattr(self, name), name)
            )
        if self.max_items > ABSOLUTE_MAX_ITEMS:
            raise ContextBoundsError("max_items exceeds the absolute limit")
        if self.max_item_bytes > ABSOLUTE_MAX_ITEM_BYTES:
            raise ContextBoundsError("max_item_bytes exceeds the absolute limit")
        if self.max_serialized_bytes > ABSOLUTE_MAX_CONTEXT_BYTES:
            raise ContextBoundsError(
                "max_serialized_bytes exceeds the absolute limit"
            )
        if self.max_depth > ABSOLUTE_MAX_DEPTH:
            raise ContextBoundsError("max_depth exceeds the absolute limit")
        if self.max_text_bytes > self.max_item_bytes:
            raise ContextBoundsError("max_text_bytes cannot exceed max_item_bytes")

    @property
    def total_token_window(self) -> int:
        return (
            self.max_input_tokens
            + self.reserved_output_tokens
            + self.reserved_tool_tokens
        )

    @property
    def input_token_limit(self) -> int:
        return self.max_input_tokens

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTEXT_CONTRACT_VERSION,
            "max_input_tokens": self.max_input_tokens,
            "reserved_output_tokens": self.reserved_output_tokens,
            "reserved_tool_tokens": self.reserved_tool_tokens,
            "max_items": self.max_items,
            "max_item_bytes": self.max_item_bytes,
            "max_serialized_bytes": self.max_serialized_bytes,
            "max_depth": self.max_depth,
            "max_text_bytes": self.max_text_bytes,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ContextBudget":
        _schema(payload, cls.SCHEMA)
        _reject_unknown(
            payload,
            {
                "schema",
                "schema_version",
                "contract_version",
                "max_input_tokens",
                "input_token_limit",
                "reserved_output_tokens",
                "reserved_tool_tokens",
                "max_items",
                "max_item_bytes",
                "max_serialized_bytes",
                "max_depth",
                "max_text_bytes",
                "content_id",
            },
            "context budget",
        )
        defaults = cls()
        result = cls(
            max_input_tokens=payload.get(
                "max_input_tokens",
                payload.get("input_token_limit", defaults.max_input_tokens),
            ),
            reserved_output_tokens=payload.get(
                "reserved_output_tokens", defaults.reserved_output_tokens
            ),
            reserved_tool_tokens=payload.get(
                "reserved_tool_tokens", defaults.reserved_tool_tokens
            ),
            max_items=payload.get("max_items", defaults.max_items),
            max_item_bytes=payload.get(
                "max_item_bytes", defaults.max_item_bytes
            ),
            max_serialized_bytes=payload.get(
                "max_serialized_bytes", defaults.max_serialized_bytes
            ),
            max_depth=payload.get("max_depth", defaults.max_depth),
            max_text_bytes=payload.get(
                "max_text_bytes", defaults.max_text_bytes
            ),
        )
        _identity(payload, result.content_id, "context budget")
        return result


@dataclass(frozen=True, init=False)
class ContextReference(_ContextCanonicalContract):
    """A compact content-addressed reference selected for a capsule."""

    SCHEMA: ClassVar[str] = CONTEXT_REFERENCE_SCHEMA

    reference_id: str
    kind: str
    tier: ContextTier = ContextTier.EVIDENCE
    referenced_content_id: str = ""
    repository_id: str = ""
    tree_id: str = ""
    path: str = ""
    summary: str = ""
    byte_count: int = 0
    token_count: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        reference_id: str,
        kind: str,
        tier: ContextTier | str = ContextTier.EVIDENCE,
        content_id: str = "",
        repository_id: str = "",
        tree_id: str = "",
        path: str = "",
        summary: str = "",
        byte_count: int = 0,
        token_count: int = 0,
        metadata: Mapping[str, Any] | None = None,
        *,
        referenced_content_id: str = "",
        target_content_id: str = "",
    ) -> None:
        values: dict[str, Any] = {
            "reference_id": reference_id,
            "kind": kind,
            "referenced_content_id": (
                referenced_content_id or target_content_id or content_id
            ),
            "repository_id": repository_id,
            "tree_id": tree_id,
            "summary": summary,
        }
        for name, required in (
            ("reference_id", True),
            ("kind", True),
            ("referenced_content_id", False),
            ("repository_id", False),
            ("tree_id", False),
            ("summary", False),
        ):
            object.__setattr__(
                self,
                name,
                _text(values[name], name, required=required),
            )
        object.__setattr__(self, "tier", _enum(tier, ContextTier, "tier"))
        object.__setattr__(
            self, "path", _relative_path(path, "path", required=False)
        )
        for name, value in (("byte_count", byte_count), ("token_count", token_count)):
            object.__setattr__(
                self, name, _nonnegative(value, name)
            )
        selected_metadata = metadata or {}
        if not isinstance(selected_metadata, Mapping):
            raise ContextContractError("metadata must be a mapping")
        frozen = _freeze_canonical(
            selected_metadata,
            name="metadata",
            max_depth=4,
            max_items=64,
            max_text_bytes=4_096,
        )
        if len(canonical_json_bytes(frozen)) > ABSOLUTE_MAX_ITEM_BYTES:
            raise ContextBoundsError("context reference metadata is too large")
        object.__setattr__(self, "metadata", frozen)

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTEXT_CONTRACT_VERSION,
            "reference_id": self.reference_id,
            "kind": self.kind,
            "tier": self.tier,
            "referenced_content_id": self.referenced_content_id,
            "repository_id": self.repository_id,
            "tree_id": self.tree_id,
            "path": self.path,
            "summary": self.summary,
            "byte_count": self.byte_count,
            "token_count": self.token_count,
            "metadata": self.metadata,
        }

    @property
    def reference_content_id(self) -> str:
        return self.content_id

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ContextReference":
        _schema(payload, cls.SCHEMA)
        _reject_unknown(
            payload,
            {
                "schema",
                "schema_version",
                "contract_version",
                "reference_id",
                "kind",
                "tier",
                "referenced_content_id",
                "target_content_id",
                "repository_id",
                "tree_id",
                "path",
                "summary",
                "byte_count",
                "token_count",
                "metadata",
                "content_id",
                "reference_content_id",
            },
            "context reference",
        )
        serialized = bool(payload.get("schema") or payload.get("contract_version"))
        result = cls(
            reference_id=payload.get("reference_id", ""),
            kind=payload.get("kind", ""),
            tier=payload.get("tier", ContextTier.EVIDENCE),
            referenced_content_id=payload.get(
                "referenced_content_id",
                payload.get(
                    "target_content_id",
                    "" if serialized else payload.get("content_id", ""),
                ),
            ),
            repository_id=payload.get("repository_id", ""),
            tree_id=payload.get("tree_id", ""),
            path=payload.get("path", ""),
            summary=payload.get("summary", ""),
            byte_count=payload.get("byte_count", 0),
            token_count=payload.get("token_count", 0),
            metadata=payload.get("metadata") or {},
        )
        claimed = payload.get("reference_content_id")
        if not claimed and serialized:
            claimed = payload.get("content_id")
        if claimed not in (None, "", result.reference_content_id):
            raise ContextIdentityError(
                "context reference identity does not match payload"
            )
        return result


@dataclass(frozen=True)
class ContextCapsule(_ContextCanonicalContract):
    """Bounded stage context whose invariant core is always present.

    ``goal``, ``authority``, ``scope``, and ``acceptance`` are separate
    mandatory fields so optional-evidence truncation can never remove or
    obscure them.
    """

    SCHEMA: ClassVar[str] = CONTEXT_CAPSULE_SCHEMA

    repository_id: str
    tree_id: str
    objective_id: str
    objective_revision: str
    policy_id: str
    policy_revision: str
    caller: str
    stage: str
    budget: ContextBudget
    goal: Any
    authority: Any
    scope: Any
    acceptance: Any
    evidence: tuple[ContextReference, ...] = ()
    expansion_references: tuple[ContextReference, ...] = ()
    input_tokens: int = 0
    truncated: bool = False
    omissions: tuple[str, ...] = ()
    parent_capsule_id: str = ""

    def __post_init__(self) -> None:
        for name in (
            "repository_id",
            "tree_id",
            "objective_id",
            "objective_revision",
            "policy_id",
            "policy_revision",
            "caller",
            "stage",
        ):
            object.__setattr__(
                self, name, _text(getattr(self, name), name, required=True)
            )
        if not isinstance(self.budget, ContextBudget):
            if not isinstance(self.budget, Mapping):
                raise ContextContractError("budget must be a ContextBudget")
            object.__setattr__(
                self, "budget", ContextBudget.from_dict(self.budget)
            )
        for name in ("goal", "authority", "scope", "acceptance"):
            value = getattr(self, name)
            if value in (None, "", (), [], {}):
                raise ContextContractError(
                    f"invariant context field {name} must not be empty"
                )
            frozen = _freeze_canonical(
                value,
                name=name,
                max_depth=self.budget.max_depth,
                max_items=self.budget.max_items,
                max_text_bytes=self.budget.max_text_bytes,
            )
            if len(canonical_json_bytes(frozen)) > self.budget.max_item_bytes:
                raise ContextBoundsError(f"{name} exceeds max_item_bytes")
            object.__setattr__(self, name, frozen)
        evidence = _coerce_references(self.evidence, "evidence")
        expansions = _coerce_references(
            self.expansion_references, "expansion_references"
        )
        if any(item.tier is ContextTier.EXPANSION for item in evidence):
            raise ContextContractError(
                "expansion references must not be embedded as selected evidence"
            )
        if any(item.tier is not ContextTier.EXPANSION for item in expansions):
            raise ContextContractError(
                "expansion_references must use the expansion tier"
            )
        identifiers = [item.reference_id for item in evidence + expansions]
        if len(identifiers) != len(set(identifiers)):
            raise ContextContractError(
                "context reference IDs must be unique across the capsule"
            )
        if len(identifiers) > self.budget.max_items:
            raise ContextBoundsError("capsule exceeds its reference-count limit")
        for item in evidence + expansions:
            if len(item.canonical_bytes()) > self.budget.max_item_bytes:
                raise ContextBoundsError(
                    "context reference exceeds max_item_bytes"
                )
            if item.repository_id and item.repository_id != self.repository_id:
                raise ContextContractError(
                    "context reference repository identity does not match capsule"
                )
            if item.tree_id and item.tree_id != self.tree_id:
                raise ContextContractError(
                    "context reference tree identity does not match capsule"
                )
        object.__setattr__(self, "evidence", evidence)
        object.__setattr__(self, "expansion_references", expansions)
        object.__setattr__(
            self, "input_tokens", _nonnegative(self.input_tokens, "input_tokens")
        )
        if self.input_tokens > self.budget.max_input_tokens:
            raise ContextBoundsError("capsule exceeds its input-token budget")
        declared_reference_tokens = sum(
            item.token_count for item in evidence
        )
        if declared_reference_tokens > self.input_tokens:
            raise ContextContractError(
                "input_tokens cannot be lower than selected reference tokens"
            )
        if not isinstance(self.truncated, bool):
            raise ContextContractError("truncated must be a boolean")
        omissions: list[str] = []
        source: Any = self.omissions
        if isinstance(source, str):
            source = (source,)
        if not isinstance(source, Sequence):
            raise ContextContractError("omissions must be a sequence")
        for item in source:
            normalized = _text(item, "omission")
            if normalized not in omissions:
                omissions.append(normalized)
        if bool(omissions) != self.truncated:
            raise ContextContractError(
                "truncated must be true exactly when omissions are recorded"
            )
        object.__setattr__(self, "omissions", tuple(sorted(omissions)))
        object.__setattr__(
            self,
            "parent_capsule_id",
            _text(
                self.parent_capsule_id,
                "parent_capsule_id",
                required=False,
            ),
        )
        if len(self.canonical_bytes()) > self.budget.max_serialized_bytes:
            raise ContextBoundsError("capsule exceeds its serialized-byte budget")

    @property
    def capsule_id(self) -> str:
        return self.content_id

    @property
    def selected_evidence(self) -> tuple[ContextReference, ...]:
        return self.evidence

    @property
    def invariant_core(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "goal": self.goal,
                "authority": self.authority,
                "scope": self.scope,
                "acceptance": self.acceptance,
            }
        )

    @property
    def is_delta(self) -> bool:
        return bool(self.parent_capsule_id)

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTEXT_CONTRACT_VERSION,
            "repository_id": self.repository_id,
            "tree_id": self.tree_id,
            "objective_id": self.objective_id,
            "objective_revision": self.objective_revision,
            "policy_id": self.policy_id,
            "policy_revision": self.policy_revision,
            "caller": self.caller,
            "stage": self.stage,
            "budget": self.budget.to_record(),
            "goal": self.goal,
            "authority": self.authority,
            "scope": self.scope,
            "acceptance": self.acceptance,
            "evidence": tuple(item.to_record() for item in self.evidence),
            "expansion_references": tuple(
                item.to_record() for item in self.expansion_references
            ),
            "input_tokens": self.input_tokens,
            "truncated": self.truncated,
            "omissions": self.omissions,
            "parent_capsule_id": self.parent_capsule_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ContextCapsule":
        _schema(payload, cls.SCHEMA)
        _reject_unknown(
            payload,
            {
                "schema",
                "schema_version",
                "contract_version",
                "repository_id",
                "tree_id",
                "objective_id",
                "objective_revision",
                "policy_id",
                "policy_revision",
                "caller",
                "stage",
                "budget",
                "goal",
                "authority",
                "scope",
                "acceptance",
                "invariant_core",
                "evidence",
                "selected_evidence",
                "expansion_references",
                "input_tokens",
                "truncated",
                "omissions",
                "parent_capsule_id",
                "content_id",
                "capsule_id",
            },
            "context capsule",
        )
        core = payload.get("invariant_core")
        if core is not None and not isinstance(core, Mapping):
            raise ContextContractError("invariant_core must be a mapping")
        core = core or {}
        budget = payload.get("budget")
        if not isinstance(budget, (ContextBudget, Mapping)):
            raise ContextContractError("capsule budget is required")
        result = cls(
            repository_id=payload.get("repository_id", ""),
            tree_id=payload.get("tree_id", ""),
            objective_id=payload.get("objective_id", ""),
            objective_revision=payload.get("objective_revision", ""),
            policy_id=payload.get("policy_id", ""),
            policy_revision=payload.get("policy_revision", ""),
            caller=payload.get("caller", ""),
            stage=payload.get("stage", ""),
            budget=(
                budget
                if isinstance(budget, ContextBudget)
                else ContextBudget.from_dict(budget)
            ),
            goal=payload.get("goal", core.get("goal")),
            authority=payload.get("authority", core.get("authority")),
            scope=payload.get("scope", core.get("scope")),
            acceptance=payload.get("acceptance", core.get("acceptance")),
            evidence=payload.get(
                "evidence", payload.get("selected_evidence", ())
            ),
            expansion_references=payload.get("expansion_references", ()),
            input_tokens=payload.get("input_tokens", 0),
            truncated=payload.get("truncated", False),
            omissions=payload.get("omissions", ()),
            parent_capsule_id=payload.get("parent_capsule_id", ""),
        )
        _identity(payload, result.content_id, "context capsule")
        return result


def canonical_context_json_bytes(value: Any) -> bytes:
    """Return the canonical wire bytes for a context contract or value."""

    return canonical_json_bytes(value)


ContextLimits = ContextBudget
ContextContractLimits = ContextBudget
ContextEvidenceReference = ContextReference
SharedContextBudget = ContextBudget
SharedContextCapsule = ContextCapsule
ContextContractValidationError = ContextContractError


__all__ = [
    "ABSOLUTE_MAX_CONTEXT_BYTES",
    "CONTEXT_BUDGET_SCHEMA",
    "CONTEXT_CAPSULE_SCHEMA",
    "CONTEXT_CONTRACT_VERSION",
    "CONTEXT_REFERENCE_SCHEMA",
    "CONTRACT_VERSION",
    "SCHEMA_VERSION",
    "ContextBoundsError",
    "ContextBudget",
    "ContextCapsule",
    "ContextContractError",
    "ContextContractLimits",
    "ContextContractValidationError",
    "ContextEvidenceReference",
    "ContextIdentityError",
    "ContextLimits",
    "ContextReference",
    "ContextTier",
    "SharedContextBudget",
    "SharedContextCapsule",
    "canonical_context_json_bytes",
]
