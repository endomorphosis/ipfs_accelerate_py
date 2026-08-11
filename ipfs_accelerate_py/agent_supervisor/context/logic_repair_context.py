"""Bounded LPR context overlays for analytical-first repair packets (LPR-016).

``LogicRepairContextBuilder`` projects admitted RPR authority
(:class:`ChangePropagationEditPacket` / :class:`ContractRepairEditPacket`)
plus prediction, behavior, value, countermodel, span, lease, and validation
bindings into a small, redacted capsule.  The capsule is a *context overlay*:
it never originates write or semantic authority, never embeds secrets or full
source/proof bodies, and marks retrieved source/comments/issues as untrusted
data so prompt text cannot become instructions.

Expansion handles are typed and path-bounded.  The model may not choose
meaning, source, owner, dependency, caller set, target, or path — those are
fixed by the admitted plan and packet before any provider sees the capsule.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any, ClassVar, Final

from ..analysis.program_logic_prediction_contracts import (
    ContextOverlayDisposition,
    CountermodelDisposition,
    CountermodelValidationReceipt,
    LogicPredictionReceipt,
    ProgramLogicAuthorityRoots,
)
from ..proof.formal_verification_contracts import (
    CanonicalContract,
    ContractValidationError,
    canonical_json_bytes,
    content_identity,
)
from ..todo_daemon.contract_packet_provider_router import redact_provider_data


# ---------------------------------------------------------------------------
# Schema / bounds
# ---------------------------------------------------------------------------

LOGIC_REPAIR_CONTEXT_INTERFACE: Final[str] = "LogicRepairContextBuilder@1"
LOGIC_REPAIR_EXPANSION_HANDLE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/logic-repair-expansion-handle@1"
)
LOGIC_REPAIR_CONTEXT_CAPSULE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/logic-repair-context-capsule@1"
)
LOGIC_REPAIR_CONTEXT_OVERLAY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/logic-repair-context-overlay@1"
)

PRODUCER_ID: Final[str] = "logic-repair-context@1"
CONTRACT_VERSION: Final[int] = 1

MAX_HANDLES: Final[int] = 64
MAX_PATHS: Final[int] = 1_024
MAX_IDS: Final[int] = 1_024
MAX_TEXT_BYTES: Final[int] = 4_096
MAX_PATH_BYTES: Final[int] = 1_024
MAX_UNTRUSTED_SNIPPET_BYTES: Final[int] = 400
MAX_CAPSULE_BYTES: Final[int] = 262_144
MAX_VALIDATIONS: Final[int] = 64

UNTRUSTED_DATA_LABEL: Final[str] = "untrusted_repository_data"
UNTRUSTED_BEGIN: Final[str] = "BEGIN_UNTRUSTED_DATA"
UNTRUSTED_END: Final[str] = "END_UNTRUSTED_DATA"
REDACTION_MARKER: Final[str] = "[REDACTED]"

# Choices the model is explicitly forbidden from making (prompt authority).
MODEL_FORBIDDEN_CHOICES: Final[tuple[str, ...]] = (
    "meaning",
    "source",
    "owner",
    "dependency",
    "caller_set",
    "target",
    "path",
)

_FORBIDDEN_HANDLE_KINDS: Final[frozenset[str]] = frozenset(
    {
        "source",
        "source_body",
        "proof_body",
        "ast_body",
        "secret",
        "secrets",
        "credential",
        "token",
        "private_key",
        "file_content",
        "repository_body",
    }
)

_BODY_FORBIDDEN_KEYS: Final[frozenset[str]] = frozenset(
    {
        "source_body",
        "source_text",
        "source_code",
        "ast_body",
        "proof_body",
        "file_content",
        "file_contents",
        "repository_body",
        "repository_content",
        "private_key",
        "secret",
        "secrets",
        "password",
        "token",
        "api_key",
        "authorization",
    }
)

_SUPPORTED_HANDLE_KINDS: Final[frozenset[str]] = frozenset(
    {
        "prediction_receipt",
        "countermodel_receipt",
        "behavior_contract",
        "value_source",
        "construction_route",
        "proof_ref",
        "validation",
        "before_hash",
        "span",
        "scc",
        "plan_step",
        "objective",
        "delta",
        "consumer",
        "index",
        "graph",
        "issue_ref",
        "comment_ref",
        "doc_ref",
        "static_finding",
        "fixed_point",
        "lease",
        "provider_config",
    }
)

_TEXT_SECRET_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"(?i)\b(bearer)\s+[A-Za-z0-9\-._~+/]+=*"),
    re.compile(r"(?i)\b(api[_-]?key|token|secret|password)\s*[:=]\s*\S+"),
    re.compile(
        r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?"
        r"-----END [A-Z0-9 ]*PRIVATE KEY-----",
        re.DOTALL,
    ),
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class LogicRepairContextError(ContractValidationError):
    """Fail-closed error for LPR context overlay construction."""


class LogicRepairContextAuthorityError(LogicRepairContextError):
    """Context would invent or broaden write/semantic authority."""


class LogicRepairContextBoundsError(LogicRepairContextError):
    """Context exceeded its declared compactness bounds."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _text(
    value: Any,
    name: str,
    *,
    required: bool = True,
    limit: int = MAX_TEXT_BYTES,
) -> str:
    if value is None:
        if required:
            raise LogicRepairContextError(f"{name} is required")
        return ""
    if not isinstance(value, str):
        raise LogicRepairContextError(f"{name} must be a string")
    text = value.strip()
    if required and not text:
        raise LogicRepairContextError(f"{name} is required")
    if len(text.encode("utf-8")) > limit:
        raise LogicRepairContextBoundsError(f"{name} exceeds text bound")
    if "\x00" in text:
        raise LogicRepairContextError(f"{name} contains a null byte")
    return text


def _identifier(value: Any, name: str) -> str:
    text = _text(value, name, required=True, limit=MAX_TEXT_BYTES)
    if any(char.isspace() for char in text):
        raise LogicRepairContextError(f"{name} must be an opaque compact identifier")
    return text


def _path(value: Any, name: str = "path") -> str:
    text = _text(value, name, required=True, limit=MAX_PATH_BYTES)
    if text.startswith("/") or text.startswith("\\"):
        raise LogicRepairContextError(f"{name} must be a repository-relative path")
    pure = PurePosixPath(text)
    if ".." in pure.parts or pure.is_absolute():
        raise LogicRepairContextError(f"{name} must not escape the repository root")
    if pure.as_posix() != text.replace("\\", "/"):
        # Normalize only after validation of the raw form.
        pass
    normalized = pure.as_posix()
    if normalized in {".", ""}:
        raise LogicRepairContextError(f"{name} must not be empty or '.'")
    return normalized


def _paths(
    values: Any,
    name: str,
    *,
    required: bool = False,
    maximum: int = MAX_PATHS,
) -> tuple[str, ...]:
    if values is None:
        if required:
            raise LogicRepairContextError(f"{name} is required")
        return ()
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise LogicRepairContextError(f"{name} must be a sequence of paths")
    result: list[str] = []
    seen: set[str] = set()
    for item in values:
        path = _path(item, name)
        if path not in seen:
            seen.add(path)
            result.append(path)
    if required and not result:
        raise LogicRepairContextError(f"{name} must not be empty")
    if len(result) > maximum:
        raise LogicRepairContextBoundsError(f"{name} exceeds path bound")
    return tuple(sorted(result))


def _ids(
    values: Any,
    name: str,
    *,
    required: bool = False,
    maximum: int = MAX_IDS,
    preserve_order: bool = False,
) -> tuple[str, ...]:
    if values is None:
        if required:
            raise LogicRepairContextError(f"{name} is required")
        return ()
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise LogicRepairContextError(f"{name} must be a sequence of identifiers")
    result: list[str] = []
    seen: set[str] = set()
    for item in values:
        ident = _identifier(item, name)
        if ident not in seen:
            seen.add(ident)
            result.append(ident)
    if required and not result:
        raise LogicRepairContextError(f"{name} must not be empty")
    if len(result) > maximum:
        raise LogicRepairContextBoundsError(f"{name} exceeds id bound")
    if preserve_order:
        return tuple(result)
    return tuple(sorted(result))


def _reject_forbidden_keys(payload: Mapping[str, Any], *, where: str) -> None:
    for key in payload:
        norm = str(key).casefold().replace("-", "_")
        if norm in _BODY_FORBIDDEN_KEYS:
            raise LogicRepairContextAuthorityError(
                f"{where} cannot embed {key} (forbidden body/secret)"
            )


def _redact_text(value: str) -> str:
    result = value
    for pattern in _TEXT_SECRET_PATTERNS:
        if "PRIVATE KEY" in pattern.pattern:
            result = pattern.sub(REDACTION_MARKER, result)
        elif pattern.pattern.startswith(r"(?i)\b(bearer)"):
            result = pattern.sub(r"\1 " + REDACTION_MARKER, result)
        else:
            result = pattern.sub(REDACTION_MARKER, result)
    return result


def redact_logic_repair_data(value: Any) -> Any:
    """Recursively redact secrets then detach via the provider redactor."""

    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise LogicRepairContextError("context keys must be strings")
            norm = key.casefold().replace("-", "_")
            if norm in _BODY_FORBIDDEN_KEYS:
                out[key] = REDACTION_MARKER
            else:
                out[key] = redact_logic_repair_data(item)
        return redact_provider_data(out)
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray, memoryview)
    ):
        return [redact_logic_repair_data(item) for item in value]
    if isinstance(value, str):
        return _redact_text(value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    raise LogicRepairContextError(
        f"context contains unsupported {type(value).__name__}"
    )


def delimit_untrusted_data(
    text: str,
    *,
    kind: str = "source",
    path: str = "",
) -> Mapping[str, Any]:
    """Wrap untrusted repository/comment/issue text as non-instruction data.

    The model must treat delimited content as data only.  Secrets and bodies
    are redacted before wrapping.
    """

    cleaned = _redact_text(_text(text, "untrusted_text", required=False, limit=MAX_TEXT_BYTES * 4))
    if len(cleaned.encode("utf-8")) > MAX_UNTRUSTED_SNIPPET_BYTES:
        cleaned = cleaned.encode("utf-8")[:MAX_UNTRUSTED_SNIPPET_BYTES].decode(
            "utf-8", errors="ignore"
        )
    kind_text = _text(kind, "untrusted_kind", required=True, limit=64)
    path_text = _path(path, "untrusted_path") if path else ""
    return MappingProxyType(
        {
            "data_label": UNTRUSTED_DATA_LABEL,
            "kind": kind_text,
            "path": path_text,
            "instruction_authority": False,
            "semantic_authority": False,
            "begin": UNTRUSTED_BEGIN,
            "end": UNTRUSTED_END,
            "payload": cleaned,
            "body_embedded": False,
            "secrets_embedded": False,
        }
    )


def _sha256_hex(payload: Mapping[str, Any] | bytes | str) -> str:
    if isinstance(payload, bytes):
        raw = payload
    elif isinstance(payload, str):
        raw = payload.encode("utf-8")
    else:
        raw = canonical_json_bytes(payload)
    return "sha256:" + hashlib.sha256(raw).hexdigest()


# ---------------------------------------------------------------------------
# Expansion handle
# ---------------------------------------------------------------------------


class LogicRepairExpansionKind(str, Enum):
    """Closed expansion handle kinds for LPR context overlays."""

    PREDICTION_RECEIPT = "prediction_receipt"
    COUNTERMODEL_RECEIPT = "countermodel_receipt"
    BEHAVIOR_CONTRACT = "behavior_contract"
    VALUE_SOURCE = "value_source"
    CONSTRUCTION_ROUTE = "construction_route"
    PROOF_REF = "proof_ref"
    VALIDATION = "validation"
    BEFORE_HASH = "before_hash"
    SPAN = "span"
    SCC = "scc"
    PLAN_STEP = "plan_step"
    OBJECTIVE = "objective"
    DELTA = "delta"
    CONSUMER = "consumer"
    INDEX = "index"
    GRAPH = "graph"
    ISSUE_REF = "issue_ref"
    COMMENT_REF = "comment_ref"
    DOC_REF = "doc_ref"
    STATIC_FINDING = "static_finding"
    FIXED_POINT = "fixed_point"
    LEASE = "lease"
    PROVIDER_CONFIG = "provider_config"


@dataclass(frozen=True)
class LogicRepairExpansionHandle:
    """Typed, path-bounded pointer to more evidence; never embeds the body."""

    handle_id: str
    kind: LogicRepairExpansionKind | str
    reference_id: str
    permitted_paths: tuple[str, ...] = ()
    budget_tokens: int = 0
    budget_bytes: int = 0
    schema: str = LOGIC_REPAIR_EXPANSION_HANDLE_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "handle_id", _identifier(self.handle_id, "handle_id"))
        kind = self.kind
        if isinstance(kind, LogicRepairExpansionKind):
            kind_value = kind.value
        else:
            kind_value = _identifier(kind, "handle.kind")
        kind_norm = kind_value.casefold().replace("-", "_")
        if kind_norm in _FORBIDDEN_HANDLE_KINDS:
            raise LogicRepairContextAuthorityError(
                "expansion handles may not name embedded bodies or secrets"
            )
        if kind_norm not in _SUPPORTED_HANDLE_KINDS:
            raise LogicRepairContextError(
                f"unsupported expansion handle kind {kind_value!r}"
            )
        object.__setattr__(self, "kind", kind_norm)
        object.__setattr__(
            self, "reference_id", _identifier(self.reference_id, "handle.reference_id")
        )
        object.__setattr__(
            self,
            "permitted_paths",
            _paths(self.permitted_paths, "handle.permitted_paths", required=False),
        )
        if not isinstance(self.budget_tokens, int) or self.budget_tokens < 0:
            raise LogicRepairContextError("budget_tokens must be a non-negative int")
        if not isinstance(self.budget_bytes, int) or self.budget_bytes < 0:
            raise LogicRepairContextError("budget_bytes must be a non-negative int")
        if self.budget_tokens > 100_000 or self.budget_bytes > MAX_CAPSULE_BYTES:
            raise LogicRepairContextBoundsError("expansion budget exceeds hard ceiling")
        if self.schema != LOGIC_REPAIR_EXPANSION_HANDLE_SCHEMA:
            raise LogicRepairContextError("unsupported expansion handle schema")

    @property
    def body_embedded(self) -> bool:
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "handle_id": self.handle_id,
            "kind": self.kind if isinstance(self.kind, str) else self.kind.value,
            "reference_id": self.reference_id,
            "permitted_paths": list(self.permitted_paths),
            "budget_tokens": self.budget_tokens,
            "budget_bytes": self.budget_bytes,
            "body_embedded": False,
            "secrets_embedded": False,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LogicRepairExpansionHandle":
        if not isinstance(payload, Mapping):
            raise LogicRepairContextError("expansion handle must be an object")
        _reject_forbidden_keys(payload, where="expansion handle")
        allowed = {
            "schema",
            "handle_id",
            "kind",
            "reference_id",
            "permitted_paths",
            "budget_tokens",
            "budget_bytes",
            "body_embedded",
            "secrets_embedded",
        }
        if set(payload).difference(allowed):
            raise LogicRepairContextError("expansion handle contains unsupported fields")
        if payload.get("body_embedded", False) is not False:
            raise LogicRepairContextAuthorityError(
                "expansion handle cannot embed a body"
            )
        if payload.get("secrets_embedded", False) is not False:
            raise LogicRepairContextAuthorityError(
                "expansion handle cannot embed secrets"
            )
        return cls(
            handle_id=payload.get("handle_id"),
            kind=payload.get("kind"),
            reference_id=payload.get("reference_id"),
            permitted_paths=tuple(payload.get("permitted_paths", ())),
            budget_tokens=int(payload.get("budget_tokens", 0) or 0),
            budget_bytes=int(payload.get("budget_bytes", 0) or 0),
            schema=payload.get("schema", LOGIC_REPAIR_EXPANSION_HANDLE_SCHEMA),
        )


# ---------------------------------------------------------------------------
# Path span / before-hash projection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LogicRepairPathSpan:
    """Exact read/write span with optional before-hash (body-free)."""

    path: str
    start: int = 0
    end: int = 0
    before_hash: str = ""
    artifact_id: str = ""
    role: str = "read"

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _path(self.path, "span.path"))
        for name in ("start", "end"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise LogicRepairContextError(f"{name} must be a non-negative int")
        if self.end and self.start > self.end:
            raise LogicRepairContextError("span start must not exceed end")
        object.__setattr__(
            self,
            "before_hash",
            _text(self.before_hash, "before_hash", required=False, limit=128),
        )
        object.__setattr__(
            self,
            "artifact_id",
            _text(self.artifact_id, "artifact_id", required=False),
        )
        role = _text(self.role, "role", required=True, limit=16).casefold()
        if role not in {"read", "write"}:
            raise LogicRepairContextError("span role must be read or write")
        object.__setattr__(self, "role", role)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "start": self.start,
            "end": self.end,
            "before_hash": self.before_hash,
            "artifact_id": self.artifact_id,
            "role": self.role,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LogicRepairPathSpan":
        if not isinstance(payload, Mapping):
            raise LogicRepairContextError("path span must be an object")
        _reject_forbidden_keys(payload, where="path span")
        return cls(
            path=payload.get("path"),
            start=int(payload.get("start", 0) or 0),
            end=int(payload.get("end", 0) or 0),
            before_hash=payload.get("before_hash", ""),
            artifact_id=payload.get("artifact_id", ""),
            role=payload.get("role", "read"),
        )


# ---------------------------------------------------------------------------
# Validation bindings
# ---------------------------------------------------------------------------


class LogicRepairValidationKind(str, Enum):
    """Closed postcondition validation families for the overlay."""

    TYPE = "type"
    EFFECT = "effect"
    RESOURCE = "resource"
    TEST = "test"
    FIXED_POINT = "fixed_point"
    LINT = "lint"
    NATIVE_BOUNDARY = "native_boundary"


@dataclass(frozen=True)
class LogicRepairValidationBinding:
    """One required validation referenced by id (never a free-form command body)."""

    validation_id: str
    kind: LogicRepairValidationKind | str
    command_ref: str = ""
    required: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "validation_id", _identifier(self.validation_id, "validation_id")
        )
        if isinstance(self.kind, LogicRepairValidationKind):
            kind_value = self.kind.value
        else:
            kind_value = _identifier(self.kind, "validation.kind")
        try:
            kind_enum = LogicRepairValidationKind(kind_value)
        except ValueError as exc:
            raise LogicRepairContextError(
                f"unsupported validation kind {kind_value!r}"
            ) from exc
        object.__setattr__(self, "kind", kind_enum)
        object.__setattr__(
            self,
            "command_ref",
            _text(self.command_ref, "command_ref", required=False),
        )
        if not isinstance(self.required, bool):
            raise LogicRepairContextError("required must be a boolean")

    def to_dict(self) -> dict[str, Any]:
        kind = self.kind.value if isinstance(self.kind, LogicRepairValidationKind) else self.kind
        return {
            "validation_id": self.validation_id,
            "kind": kind,
            "command_ref": self.command_ref,
            "required": self.required,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LogicRepairValidationBinding":
        if not isinstance(payload, Mapping):
            raise LogicRepairContextError("validation binding must be an object")
        return cls(
            validation_id=payload.get("validation_id"),
            kind=payload.get("kind"),
            command_ref=payload.get("command_ref", ""),
            required=bool(payload.get("required", True)),
        )


# ---------------------------------------------------------------------------
# Capsule / overlay records
# ---------------------------------------------------------------------------


class RprPacketInterfaceKind(str, Enum):
    """Existing RPR packet interfaces the overlay may bind (closed)."""

    CHANGE_PROPAGATION = "ChangePropagationEditPacket@1"
    CONTRACT_REPAIR = "ContractRepairEditPacket@2"


@dataclass(frozen=True)
class LogicRepairContextCapsule(CanonicalContract):
    """Body-free capsule of high-value static/proof context for one repair step.

    Write authority remains on the bound RPR packet/plan/lease.  This capsule
    only carries references and redacted untrusted data delimiters.
    """

    SCHEMA: ClassVar[str] = LOGIC_REPAIR_CONTEXT_CAPSULE_SCHEMA

    capsule_id: str
    roots: ProgramLogicAuthorityRoots
    rpr_packet_interface: RprPacketInterfaceKind | str
    rpr_packet_id: str
    rpr_plan_id: str
    rpr_plan_step_id: str
    scc_group_id: str
    writer_lease_id: str
    admitted_prediction_ids: tuple[str, ...]
    chosen_value_refs: tuple[str, ...]
    construction_route_refs: tuple[str, ...]
    admitted_behavior_ids: tuple[str, ...]
    validated_countermodel_ids: tuple[str, ...]
    read_spans: tuple[LogicRepairPathSpan, ...]
    write_spans: tuple[LogicRepairPathSpan, ...]
    before_hash_refs: tuple[str, ...]
    forbidden_path_refs: tuple[str, ...]
    forbidden_semantic_change_refs: tuple[str, ...]
    validations: tuple[LogicRepairValidationBinding, ...]
    postcondition_refs: tuple[str, ...]
    expansion_handles: tuple[LogicRepairExpansionHandle, ...]
    provider_id: str = ""
    model_id: str = ""
    config_id: str = ""
    untrusted_snippets: tuple[Mapping[str, Any], ...] = ()
    objective_id: str = ""
    delta_id: str = ""
    change_set_id: str = ""
    consumer_ids: tuple[str, ...] = ()
    proof_refs: tuple[str, ...] = ()
    static_finding_refs: tuple[str, ...] = ()
    unsupported_limits: tuple[str, ...] = ()
    disposition: ContextOverlayDisposition = ContextOverlayDisposition.MODEL_REQUIRED
    write_authority: bool = False
    semantic_authority: bool = False
    producer_id: str = PRODUCER_ID

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "capsule_id", _identifier(self.capsule_id, "capsule_id")
        )
        if not isinstance(self.roots, ProgramLogicAuthorityRoots):
            if isinstance(self.roots, Mapping):
                object.__setattr__(
                    self,
                    "roots",
                    ProgramLogicAuthorityRoots.from_dict(self.roots)
                    if "schema" in self.roots
                    else ProgramLogicAuthorityRoots(**dict(self.roots)),
                )
            else:
                raise LogicRepairContextError(
                    "roots must be ProgramLogicAuthorityRoots"
                )
        if isinstance(self.rpr_packet_interface, RprPacketInterfaceKind):
            iface = self.rpr_packet_interface
        else:
            try:
                iface = RprPacketInterfaceKind(
                    _text(self.rpr_packet_interface, "rpr_packet_interface")
                )
            except ValueError as exc:
                raise LogicRepairContextAuthorityError(
                    "rpr_packet_interface must be ChangePropagationEditPacket@1 "
                    "or ContractRepairEditPacket@2"
                ) from exc
        object.__setattr__(self, "rpr_packet_interface", iface)
        for name in (
            "rpr_packet_id",
            "rpr_plan_id",
            "rpr_plan_step_id",
            "writer_lease_id",
        ):
            object.__setattr__(
                self, name, _identifier(getattr(self, name), name)
            )
        object.__setattr__(
            self,
            "scc_group_id",
            _text(self.scc_group_id, "scc_group_id", required=False),
        )
        object.__setattr__(
            self,
            "admitted_prediction_ids",
            _ids(self.admitted_prediction_ids, "admitted_prediction_ids"),
        )
        object.__setattr__(
            self, "chosen_value_refs", _ids(self.chosen_value_refs, "chosen_value_refs")
        )
        object.__setattr__(
            self,
            "construction_route_refs",
            _ids(self.construction_route_refs, "construction_route_refs"),
        )
        object.__setattr__(
            self,
            "admitted_behavior_ids",
            _ids(self.admitted_behavior_ids, "admitted_behavior_ids"),
        )
        object.__setattr__(
            self,
            "validated_countermodel_ids",
            _ids(self.validated_countermodel_ids, "validated_countermodel_ids"),
        )
        if not isinstance(self.read_spans, Sequence) or not all(
            isinstance(item, LogicRepairPathSpan) for item in self.read_spans
        ):
            raise LogicRepairContextError("read_spans must be LogicRepairPathSpan values")
        if not isinstance(self.write_spans, Sequence) or not all(
            isinstance(item, LogicRepairPathSpan) for item in self.write_spans
        ):
            raise LogicRepairContextError(
                "write_spans must be LogicRepairPathSpan values"
            )
        object.__setattr__(
            self,
            "read_spans",
            tuple(sorted(self.read_spans, key=lambda item: (item.path, item.start))),
        )
        object.__setattr__(
            self,
            "write_spans",
            tuple(sorted(self.write_spans, key=lambda item: (item.path, item.start))),
        )
        for span in self.write_spans:
            if span.role != "write":
                raise LogicRepairContextError("write_spans must use role=write")
        for span in self.read_spans:
            if span.role != "read":
                raise LogicRepairContextError("read_spans must use role=read")
        object.__setattr__(
            self, "before_hash_refs", _ids(self.before_hash_refs, "before_hash_refs")
        )
        object.__setattr__(
            self,
            "forbidden_path_refs",
            _ids(self.forbidden_path_refs, "forbidden_path_refs"),
        )
        object.__setattr__(
            self,
            "forbidden_semantic_change_refs",
            _ids(
                self.forbidden_semantic_change_refs,
                "forbidden_semantic_change_refs",
            ),
        )
        if not isinstance(self.validations, Sequence) or not all(
            isinstance(item, LogicRepairValidationBinding) for item in self.validations
        ):
            raise LogicRepairContextError(
                "validations must be LogicRepairValidationBinding values"
            )
        validations = tuple(
            sorted(self.validations, key=lambda item: item.validation_id)
        )
        if len(validations) > MAX_VALIDATIONS:
            raise LogicRepairContextBoundsError("validations exceed bound")
        if len({item.validation_id for item in validations}) != len(validations):
            raise LogicRepairContextError("validation ids must be unique")
        object.__setattr__(self, "validations", validations)
        object.__setattr__(
            self, "postcondition_refs", _ids(self.postcondition_refs, "postcondition_refs")
        )
        if not isinstance(self.expansion_handles, Sequence) or not all(
            isinstance(item, LogicRepairExpansionHandle)
            for item in self.expansion_handles
        ):
            raise LogicRepairContextError(
                "expansion_handles must be LogicRepairExpansionHandle values"
            )
        handles = tuple(
            sorted(self.expansion_handles, key=lambda item: item.handle_id)
        )
        if len(handles) > MAX_HANDLES:
            raise LogicRepairContextBoundsError("expansion_handles exceed bound")
        if len({item.handle_id for item in handles}) != len(handles):
            raise LogicRepairContextError("expansion handle ids must be unique")
        read_paths = {item.path for item in self.read_spans} | {
            item.path for item in self.write_spans
        }
        for handle in handles:
            if not set(handle.permitted_paths).issubset(read_paths):
                raise LogicRepairContextAuthorityError(
                    "an expansion handle cannot expand read/write scope"
                )
        object.__setattr__(self, "expansion_handles", handles)
        for name in ("provider_id", "model_id", "config_id", "objective_id", "delta_id", "change_set_id"):
            object.__setattr__(
                self, name, _text(getattr(self, name), name, required=False)
            )
        if not isinstance(self.untrusted_snippets, Sequence):
            raise LogicRepairContextError("untrusted_snippets must be a sequence")
        snippets: list[Mapping[str, Any]] = []
        for item in self.untrusted_snippets:
            if not isinstance(item, Mapping):
                raise LogicRepairContextError("untrusted snippet must be a mapping")
            if item.get("data_label") != UNTRUSTED_DATA_LABEL:
                raise LogicRepairContextAuthorityError(
                    "source/comments/issues must be delimited as untrusted data"
                )
            if item.get("instruction_authority") is not False:
                raise LogicRepairContextAuthorityError(
                    "untrusted data cannot claim instruction authority"
                )
            snippets.append(MappingProxyType(dict(redact_logic_repair_data(dict(item)))))
        object.__setattr__(self, "untrusted_snippets", tuple(snippets))
        object.__setattr__(self, "consumer_ids", _ids(self.consumer_ids, "consumer_ids"))
        object.__setattr__(self, "proof_refs", _ids(self.proof_refs, "proof_refs"))
        object.__setattr__(
            self,
            "static_finding_refs",
            _ids(self.static_finding_refs, "static_finding_refs"),
        )
        object.__setattr__(
            self, "unsupported_limits", _ids(self.unsupported_limits, "unsupported_limits")
        )
        if isinstance(self.disposition, ContextOverlayDisposition):
            disposition = self.disposition
        else:
            try:
                disposition = ContextOverlayDisposition(
                    _text(self.disposition, "disposition")
                )
            except ValueError as exc:
                raise LogicRepairContextError("unsupported overlay disposition") from exc
        object.__setattr__(self, "disposition", disposition)
        if self.write_authority is not False:
            raise LogicRepairContextAuthorityError(
                "logic repair context cannot claim write authority"
            )
        object.__setattr__(self, "write_authority", False)
        if self.semantic_authority is not False:
            raise LogicRepairContextAuthorityError(
                "logic repair context cannot claim semantic authority"
            )
        object.__setattr__(self, "semantic_authority", False)
        object.__setattr__(
            self, "producer_id", _identifier(self.producer_id, "producer_id")
        )
        if (
            disposition is ContextOverlayDisposition.MODEL_REQUIRED
            and not self.model_id
        ):
            raise LogicRepairContextError(
                "model_required capsules require a model identity"
            )
        if disposition is ContextOverlayDisposition.DETERMINISTIC and self.model_id:
            raise LogicRepairContextError(
                "deterministic capsules must not bind a model identity"
            )
        encoded = canonical_json_bytes(self._payload())
        if len(encoded) > MAX_CAPSULE_BYTES:
            raise LogicRepairContextBoundsError("capsule exceeds serialized byte bound")

    def _payload(self) -> dict[str, Any]:
        iface = (
            self.rpr_packet_interface.value
            if isinstance(self.rpr_packet_interface, RprPacketInterfaceKind)
            else self.rpr_packet_interface
        )
        disposition = (
            self.disposition.value
            if isinstance(self.disposition, ContextOverlayDisposition)
            else self.disposition
        )
        return {
            "contract_version": CONTRACT_VERSION,
            "interface": LOGIC_REPAIR_CONTEXT_INTERFACE,
            "producer_id": self.producer_id,
            "capsule_id": self.capsule_id,
            "roots": self.roots.to_dict(),
            "rpr_packet_interface": iface,
            "rpr_packet_id": self.rpr_packet_id,
            "rpr_plan_id": self.rpr_plan_id,
            "rpr_plan_step_id": self.rpr_plan_step_id,
            "scc_group_id": self.scc_group_id,
            "writer_lease_id": self.writer_lease_id,
            "admitted_prediction_ids": list(self.admitted_prediction_ids),
            "chosen_value_refs": list(self.chosen_value_refs),
            "construction_route_refs": list(self.construction_route_refs),
            "admitted_behavior_ids": list(self.admitted_behavior_ids),
            "validated_countermodel_ids": list(self.validated_countermodel_ids),
            "read_spans": [item.to_dict() for item in self.read_spans],
            "write_spans": [item.to_dict() for item in self.write_spans],
            "before_hash_refs": list(self.before_hash_refs),
            "forbidden_path_refs": list(self.forbidden_path_refs),
            "forbidden_semantic_change_refs": list(self.forbidden_semantic_change_refs),
            "validations": [item.to_dict() for item in self.validations],
            "postcondition_refs": list(self.postcondition_refs),
            "expansion_handles": [item.to_dict() for item in self.expansion_handles],
            "provider_id": self.provider_id,
            "model_id": self.model_id,
            "config_id": self.config_id,
            "untrusted_snippets": [dict(item) for item in self.untrusted_snippets],
            "objective_id": self.objective_id,
            "delta_id": self.delta_id,
            "change_set_id": self.change_set_id,
            "consumer_ids": list(self.consumer_ids),
            "proof_refs": list(self.proof_refs),
            "static_finding_refs": list(self.static_finding_refs),
            "unsupported_limits": list(self.unsupported_limits),
            "disposition": disposition,
            "write_authority": False,
            "semantic_authority": False,
            "model_must_not_choose": list(MODEL_FORBIDDEN_CHOICES),
            "body_embedded": False,
            "secrets_embedded": False,
        }

    @property
    def permitted_read_paths(self) -> tuple[str, ...]:
        return tuple(sorted({item.path for item in self.read_spans}))

    @property
    def permitted_write_paths(self) -> tuple[str, ...]:
        return tuple(sorted({item.path for item in self.write_spans}))

    @property
    def expansion_handle_refs(self) -> tuple[str, ...]:
        return tuple(item.handle_id for item in self.expansion_handles)

    @property
    def validation_refs(self) -> tuple[str, ...]:
        return tuple(item.validation_id for item in self.validations)


@dataclass(frozen=True)
class LogicRepairContextOverlay:
    """Projection of a capsule onto the LPR-001 overlay identity surface."""

    capsule: LogicRepairContextCapsule
    overlay_id: str = ""
    schema: str = LOGIC_REPAIR_CONTEXT_OVERLAY_SCHEMA

    def __post_init__(self) -> None:
        if not isinstance(self.capsule, LogicRepairContextCapsule):
            raise LogicRepairContextError("capsule must be LogicRepairContextCapsule")
        overlay_id = self.overlay_id or f"overlay:{self.capsule.capsule_id}"
        object.__setattr__(self, "overlay_id", _identifier(overlay_id, "overlay_id"))
        if self.schema != LOGIC_REPAIR_CONTEXT_OVERLAY_SCHEMA:
            raise LogicRepairContextError("unsupported context overlay schema")

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "interface": LOGIC_REPAIR_CONTEXT_INTERFACE,
            "overlay_id": self.overlay_id,
            "capsule_id": self.capsule.capsule_id,
            "capsule_content_id": self.capsule.content_id,
            "rpr_packet_id": self.capsule.rpr_packet_id,
            "rpr_plan_id": self.capsule.rpr_plan_id,
            "rpr_plan_step_id": self.capsule.rpr_plan_step_id,
            "writer_lease_id": self.capsule.writer_lease_id,
            "disposition": (
                self.capsule.disposition.value
                if isinstance(self.capsule.disposition, ContextOverlayDisposition)
                else self.capsule.disposition
            ),
            "write_authority": False,
            "semantic_authority": False,
            "model_must_not_choose": list(MODEL_FORBIDDEN_CHOICES),
            "expansion_handle_refs": list(self.capsule.expansion_handle_refs),
            "validation_refs": list(self.capsule.validation_refs),
            "permitted_read_paths": list(self.capsule.permitted_read_paths),
            "permitted_write_paths": list(self.capsule.permitted_write_paths),
        }

    @property
    def content_id(self) -> str:
        return content_identity(self._identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self._identity_payload()
        payload["content_id"] = self.content_id
        return payload


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LogicRepairContextRequest:
    """Inputs for building one LPR context capsule from admitted RPR authority."""

    roots: ProgramLogicAuthorityRoots
    rpr_packet_interface: RprPacketInterfaceKind | str
    rpr_packet_id: str
    rpr_plan_id: str
    rpr_plan_step_id: str
    writer_lease_id: str
    plan_admitted: bool
    scc_group_id: str = ""
    prediction_receipts: tuple[LogicPredictionReceipt | Mapping[str, Any], ...] = ()
    admitted_prediction_ids: tuple[str, ...] = ()
    chosen_value_refs: tuple[str, ...] = ()
    construction_route_refs: tuple[str, ...] = ()
    admitted_behavior_ids: tuple[str, ...] = ()
    countermodel_receipts: tuple[
        CountermodelValidationReceipt | Mapping[str, Any], ...
    ] = ()
    validated_countermodel_ids: tuple[str, ...] = ()
    read_spans: tuple[LogicRepairPathSpan | Mapping[str, Any], ...] = ()
    write_spans: tuple[LogicRepairPathSpan | Mapping[str, Any], ...] = ()
    before_hash_refs: tuple[str, ...] = ()
    forbidden_path_refs: tuple[str, ...] = ()
    forbidden_semantic_change_refs: tuple[str, ...] = ()
    validations: tuple[LogicRepairValidationBinding | Mapping[str, Any], ...] = ()
    postcondition_refs: tuple[str, ...] = ()
    expansion_handles: tuple[LogicRepairExpansionHandle | Mapping[str, Any], ...] = ()
    provider_id: str = ""
    model_id: str = ""
    config_id: str = ""
    untrusted_source_snippets: tuple[Mapping[str, Any], ...] = ()
    untrusted_comment_snippets: tuple[Mapping[str, Any], ...] = ()
    untrusted_issue_snippets: tuple[Mapping[str, Any], ...] = ()
    objective_id: str = ""
    delta_id: str = ""
    change_set_id: str = ""
    consumer_ids: tuple[str, ...] = ()
    proof_refs: tuple[str, ...] = ()
    static_finding_refs: tuple[str, ...] = ()
    unsupported_limits: tuple[str, ...] = ()
    disposition: ContextOverlayDisposition | str = (
        ContextOverlayDisposition.MODEL_REQUIRED
    )
    capsule_id: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.plan_admitted, bool):
            raise LogicRepairContextError("plan_admitted must be a boolean")
        if not self.plan_admitted:
            raise LogicRepairContextAuthorityError(
                "exact target/atomic plan admission must precede all packet/provider work"
            )


class LogicRepairContextBuilder:
    """Build a bounded LPR context capsule from admitted RPR + prediction evidence."""

    INTERFACE: ClassVar[str] = LOGIC_REPAIR_CONTEXT_INTERFACE

    def build(self, request: LogicRepairContextRequest) -> LogicRepairContextOverlay:
        if not isinstance(request, LogicRepairContextRequest):
            raise LogicRepairContextError(
                "request must be LogicRepairContextRequest"
            )
        if not request.plan_admitted:
            raise LogicRepairContextAuthorityError(
                "exact target/atomic plan admission must precede all packet/provider work"
            )

        prediction_ids = list(request.admitted_prediction_ids)
        for receipt in request.prediction_receipts:
            if isinstance(receipt, LogicPredictionReceipt):
                prediction_ids.append(receipt.receipt_id)
            elif isinstance(receipt, Mapping):
                rid = receipt.get("receipt_id") or receipt.get("prediction_id")
                if rid:
                    prediction_ids.append(str(rid))
            else:
                raise LogicRepairContextError(
                    "prediction_receipts must be LogicPredictionReceipt values"
                )

        countermodel_ids = list(request.validated_countermodel_ids)
        for receipt in request.countermodel_receipts:
            if isinstance(receipt, CountermodelValidationReceipt):
                if receipt.disposition is not CountermodelDisposition.VALIDATED:
                    # Only independently validated countermodels bind as authority.
                    continue
                countermodel_ids.append(receipt.receipt_id)
            elif isinstance(receipt, Mapping):
                disposition = str(receipt.get("disposition") or "")
                if disposition and disposition != CountermodelDisposition.VALIDATED.value:
                    continue
                rid = receipt.get("receipt_id")
                if rid:
                    countermodel_ids.append(str(rid))
            else:
                raise LogicRepairContextError(
                    "countermodel_receipts must be CountermodelValidationReceipt values"
                )

        read_spans = tuple(
            item
            if isinstance(item, LogicRepairPathSpan)
            else LogicRepairPathSpan.from_dict(item)
            for item in request.read_spans
        )
        write_spans = tuple(
            item
            if isinstance(item, LogicRepairPathSpan)
            else LogicRepairPathSpan.from_dict({**dict(item), "role": "write"})
            for item in request.write_spans
        )
        if not write_spans and not read_spans:
            raise LogicRepairContextError(
                "context requires at least one admitted read or write span"
            )

        validations = tuple(
            item
            if isinstance(item, LogicRepairValidationBinding)
            else LogicRepairValidationBinding.from_dict(item)
            for item in request.validations
        )
        # Ensure fixed-point / type / effect / resource / test families can bind
        # when callers only pass validation refs as free-form ids.
        if not validations and request.postcondition_refs:
            validations = tuple(
                LogicRepairValidationBinding(
                    validation_id=ref,
                    kind=LogicRepairValidationKind.FIXED_POINT
                    if "fixed" in ref
                    else LogicRepairValidationKind.TEST,
                )
                for ref in request.postcondition_refs[:MAX_VALIDATIONS]
            )

        handles = tuple(
            item
            if isinstance(item, LogicRepairExpansionHandle)
            else LogicRepairExpansionHandle.from_dict(item)
            for item in request.expansion_handles
        )

        untrusted: list[Mapping[str, Any]] = []
        for kind, snippets in (
            ("source", request.untrusted_source_snippets),
            ("comment", request.untrusted_comment_snippets),
            ("issue", request.untrusted_issue_snippets),
        ):
            for snippet in snippets:
                if not isinstance(snippet, Mapping):
                    raise LogicRepairContextError(
                        f"untrusted {kind} snippets must be mappings"
                    )
                text = str(
                    snippet.get("text")
                    or snippet.get("body")
                    or snippet.get("summary")
                    or ""
                )
                path = str(snippet.get("path") or "")
                untrusted.append(
                    delimit_untrusted_data(text, kind=kind, path=path)
                    if path
                    else delimit_untrusted_data(text, kind=kind)
                )

        before_hash_refs = list(request.before_hash_refs)
        for span in (*read_spans, *write_spans):
            if span.before_hash:
                before_hash_refs.append(f"hash:{span.path}:{span.before_hash[:24]}")

        disposition = request.disposition
        if not isinstance(disposition, ContextOverlayDisposition):
            disposition = ContextOverlayDisposition(
                _text(disposition, "disposition")
            )

        capsule_id = request.capsule_id or content_identity(
            {
                "rpr_packet_id": request.rpr_packet_id,
                "rpr_plan_id": request.rpr_plan_id,
                "rpr_plan_step_id": request.rpr_plan_step_id,
                "writer_lease_id": request.writer_lease_id,
                "disposition": disposition.value,
            }
        )
        if not str(capsule_id).startswith("capsule:"):
            capsule_id = f"capsule:{capsule_id.removeprefix('sha256:')[:48]}"

        capsule = LogicRepairContextCapsule(
            capsule_id=capsule_id,
            roots=request.roots,
            rpr_packet_interface=request.rpr_packet_interface,
            rpr_packet_id=request.rpr_packet_id,
            rpr_plan_id=request.rpr_plan_id,
            rpr_plan_step_id=request.rpr_plan_step_id,
            scc_group_id=request.scc_group_id,
            writer_lease_id=request.writer_lease_id,
            admitted_prediction_ids=tuple(prediction_ids),
            chosen_value_refs=request.chosen_value_refs,
            construction_route_refs=request.construction_route_refs,
            admitted_behavior_ids=request.admitted_behavior_ids,
            validated_countermodel_ids=tuple(countermodel_ids),
            read_spans=read_spans,
            write_spans=write_spans,
            before_hash_refs=tuple(before_hash_refs),
            forbidden_path_refs=request.forbidden_path_refs,
            forbidden_semantic_change_refs=request.forbidden_semantic_change_refs,
            validations=validations,
            postcondition_refs=request.postcondition_refs,
            expansion_handles=handles,
            provider_id=request.provider_id,
            model_id=request.model_id if disposition is ContextOverlayDisposition.MODEL_REQUIRED else "",
            config_id=request.config_id,
            untrusted_snippets=tuple(untrusted),
            objective_id=request.objective_id or request.roots.objective_id,
            delta_id=request.delta_id,
            change_set_id=request.change_set_id,
            consumer_ids=request.consumer_ids,
            proof_refs=request.proof_refs,
            static_finding_refs=request.static_finding_refs,
            unsupported_limits=request.unsupported_limits,
            disposition=disposition,
        )
        return LogicRepairContextOverlay(capsule=capsule)


def build_logic_repair_context(
    request: LogicRepairContextRequest,
) -> LogicRepairContextOverlay:
    """Module-level convenience wrapper around :class:`LogicRepairContextBuilder`."""

    return LogicRepairContextBuilder().build(request)


__all__ = [
    "LOGIC_REPAIR_CONTEXT_INTERFACE",
    "LOGIC_REPAIR_EXPANSION_HANDLE_SCHEMA",
    "LOGIC_REPAIR_CONTEXT_CAPSULE_SCHEMA",
    "LOGIC_REPAIR_CONTEXT_OVERLAY_SCHEMA",
    "PRODUCER_ID",
    "CONTRACT_VERSION",
    "UNTRUSTED_DATA_LABEL",
    "UNTRUSTED_BEGIN",
    "UNTRUSTED_END",
    "REDACTION_MARKER",
    "MODEL_FORBIDDEN_CHOICES",
    "LogicRepairContextError",
    "LogicRepairContextAuthorityError",
    "LogicRepairContextBoundsError",
    "LogicRepairExpansionKind",
    "LogicRepairExpansionHandle",
    "LogicRepairPathSpan",
    "LogicRepairValidationKind",
    "LogicRepairValidationBinding",
    "RprPacketInterfaceKind",
    "LogicRepairContextCapsule",
    "LogicRepairContextOverlay",
    "LogicRepairContextRequest",
    "LogicRepairContextBuilder",
    "build_logic_repair_context",
    "delimit_untrusted_data",
    "redact_logic_repair_data",
]
