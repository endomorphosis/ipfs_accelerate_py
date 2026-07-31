"""Compact CID-addressed repair and delta-retry packets (VFS-032 / VFS-G110).

Compile the smallest sufficient model-facing repair packet from finding,
contract, call-slice, proof, and validation references.  Large source, AST,
graph, proof, and witness bodies remain behind content-addressed expansion
handles; only bounded source spans may be inlined.

Normative rules:

* Default canonical JSON is at most 16 KiB *plus* bounded source spans.
* Required authority / identity / acceptance fields never truncate under a
  provider budget — compilation fails closed or marks incomplete instead.
* A delta retry binds the prior decision/packet identity and transmits only
  changed or explicitly requested evidence.
* Model output is a proposal (``semantic_authority=false``,
  ``completion_authoritative=false``); path/proof/validation/lease/merge
  gates remain authoritative.
* Secrets and private witness material are redacted or rejected; they never
  participate in content identities.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, ClassVar, Final

from .proof.formal_verification_contracts import (
    ContractValidationError,
    canonical_json,
    canonical_json_bytes,
    content_identity,
)


# ---------------------------------------------------------------------------
# Version / schema / bounds
# ---------------------------------------------------------------------------

CONTRACT_REPAIR_PACKET_VERSION: Final[int] = 1
COMPACT_REPAIR_PACKET_EVIDENCE: Final[str] = "vfs/compact-repair-packet@1"
DELTA_REPAIR_CONTEXT_EVIDENCE: Final[str] = "vfs/delta-repair-context@1"

REPAIR_PACKET_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/contract-repair-packet@1"
)
REPAIR_PACKET_REQUEST_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/contract-repair-packet-request@1"
)
REPAIR_PACKET_DELTA_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/contract-repair-packet-delta@1"
)
REPAIR_EXPANSION_HANDLE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/contract-repair-expansion-handle@1"
)
BOUNDED_SOURCE_SPAN_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/contract-repair-source-span@1"
)
CALL_SLICE_REF_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/contract-repair-call-slice-ref@1"
)
REPAIR_PACKET_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/contract-repair-packet-receipt@1"
)

DEFAULT_MAX_PACKET_BYTES: Final[int] = 16 * 1024
DEFAULT_MAX_SPAN_BYTES: Final[int] = 2_048
DEFAULT_MAX_SPAN_LINES: Final[int] = 40
DEFAULT_MAX_SPAN_COUNT: Final[int] = 8
DEFAULT_MAX_COLLECTION_ITEMS: Final[int] = 64
DEFAULT_MAX_TEXT_CHARS: Final[int] = 512
DEFAULT_MAX_HANDLE_COUNT: Final[int] = 64
DEFAULT_MAX_CALL_SLICE_STEPS: Final[int] = 16
# Conservative deterministic token estimate (no tokenizer dependency).
BYTES_PER_TOKEN: Final[int] = 4

REDACTED: Final[str] = "<redacted>"
OMITTED: Final[str] = "<omitted>"

# Keys that must never appear as embedded full bodies in a packet payload.
_FORBIDDEN_BODY_KEYS: Final[frozenset[str]] = frozenset(
    {
        "ast",
        "ast_body",
        "body",
        "contents",
        "file_content",
        "file_contents",
        "full_ast",
        "full_graph",
        "full_proof",
        "full_receipt",
        "full_source",
        "full_trace",
        "gold_ir",
        "gold_ir_body",
        "graph",
        "graph_body",
        "hidden_witness",
        "kernel_proof_body",
        "lean_source",
        "private_witness",
        "proof_body",
        "proof_text",
        "receipt_body",
        "solver_trace",
        "source",
        "source_body",
        "source_code",
        "source_text",
        "transcript",
        "witness",
    }
)

_PRIVATE_KEY_RE = re.compile(
    r"(?:^|[_\-.])(?:password|passwd|secret|api[_-]?key|access[_-]?token|"
    r"refresh[_-]?token|session[_-]?token|credential|authorization|cookie|"
    r"private[_-]?key|private[_-]?premise|private[_-]?input|"
    r"hidden[_-]?witness|private[_-]?witness|witness)(?:$|[_\-.])",
    re.IGNORECASE,
)

# Required fields of the invariant core — never deferred as expansion handles.
REQUIRED_CORE_FIELDS: Final[tuple[str, ...]] = (
    "task_id",
    "finding_ids",
    "forest_id",
    "tree_id",
    "policy_id",
    "expected_contract_ref",
    "observed_contract_ref",
    "call_slice",
    "edit_scope",
    "effects",
    "acceptance",
    "validation_commands",
    "proof_commands",
    "risks",
    "authority",
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ContractRepairPacketError(ContractValidationError):
    """Raised when a repair packet contract is malformed or unsafe."""


class RepairPacketBudgetError(ContractRepairPacketError):
    """Raised when required fields cannot fit the configured packet budget."""


class RepairPacketIntegrityError(ContractRepairPacketError):
    """Raised on stale handles, forged identities, or reconstruction failure."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _required_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractRepairPacketError(f"{name} is required")
    text = value.strip()
    if len(text) > DEFAULT_MAX_TEXT_CHARS * 8:
        raise ContractRepairPacketError(f"{name} exceeds text bound")
    return text


def _optional_text(value: Any, name: str = "value") -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ContractRepairPacketError(f"{name} must be a string")
    return value.strip()


def _sorted_unique(
    values: Iterable[Any],
    *,
    name: str,
    required: bool = False,
    maximum: int = DEFAULT_MAX_COLLECTION_ITEMS,
) -> tuple[str, ...]:
    if values is None:
        items: Iterable[Any] = ()
    elif isinstance(values, (str, bytes, bytearray)):
        items = (values,)
    elif isinstance(values, Sequence):
        items = values
    else:
        raise ContractRepairPacketError(f"{name} must be a sequence of strings")
    out: list[str] = []
    seen: set[str] = set()
    for raw in items:
        text = _optional_text(raw, name)
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    if len(out) > maximum:
        raise ContractRepairPacketError(f"{name} exceeds {maximum} items")
    result = tuple(sorted(out))
    if required and not result:
        raise ContractRepairPacketError(f"{name} must not be empty")
    return result


def _positive_int(value: Any, name: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ContractRepairPacketError(
            f"{name} must be an integer >= {minimum}"
        )
    return value


def _enum(value: Any, enum_type: type[Enum], name: str) -> Any:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(str(value or "").strip())
    except ValueError as exc:
        raise ContractRepairPacketError(
            f"unsupported {name}: {value!r}"
        ) from exc


def _redact_inline(value: str) -> tuple[str, int]:
    """Redact obvious secret-shaped substrings without hashing them."""

    patterns = (
        re.compile(r"(?i)bearer\s+\S+"),
        re.compile(r"(?i)sk-[a-z0-9]{16,}"),
        re.compile(
            r"(?i)(api[_-]?key|access[_-]?token|password|passwd|secret|"
            r"authorization|credential)\s*[:=]\s*\S+(?:\s+\S+)*"
        ),
        re.compile(
            r"(?i)(api[_-]?key|token|password|secret|authorization)"
            r"\s*[:=]\s*\S+"
        ),
    )
    redactions = 0
    result = value
    for pattern in patterns:
        updated, count = pattern.subn(REDACTED, result)
        result = updated
        redactions += count
    return result, redactions


def _plain(value: Any, path: str = "value") -> Any:
    """Canonical plain data with forbidden body keys and private keys rejected."""

    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        raise ContractRepairPacketError(f"{path} cannot contain floats")
    if isinstance(value, (bytes, bytearray, memoryview)):
        raise ContractRepairPacketError(f"{path} cannot contain bytes")
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, str):
        redacted, _ = _redact_inline(value)
        if len(redacted) > DEFAULT_MAX_TEXT_CHARS * 16:
            return redacted[: DEFAULT_MAX_TEXT_CHARS * 16] + OMITTED
        return redacted
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for raw_key, raw_value in value.items():
            if not isinstance(raw_key, str) or not raw_key.strip():
                raise ContractRepairPacketError(
                    f"{path} keys must be non-empty strings"
                )
            key = raw_key.strip()
            lowered = key.casefold().replace("-", "_")
            if lowered in _FORBIDDEN_BODY_KEYS:
                raise ContractRepairPacketError(
                    f"{path}.{key} embeds full source/AST/graph/proof/witness; "
                    "use an expansion handle"
                )
            if _PRIVATE_KEY_RE.search(lowered):
                # Private material never enters public packet identities.
                result[key] = REDACTED
                continue
            result[key] = _plain(raw_value, f"{path}.{key}")
        return result
    if isinstance(value, Sequence):
        return [_plain(item, f"{path}[]") for item in value]
    raise ContractRepairPacketError(
        f"{path} has unsupported value type {type(value).__name__}"
    )


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray, memoryview)
    ):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray, memoryview)
    ):
        return [_thaw(item) for item in value]
    return value


def estimate_tokens(payload: Any) -> int:
    """Deterministic conservative token estimate for packet size comparisons."""

    raw = (
        payload
        if isinstance(payload, (bytes, bytearray))
        else canonical_json_bytes(payload)
    )
    return max(1, (len(raw) + BYTES_PER_TOKEN - 1) // BYTES_PER_TOKEN)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class RepairPacketStatus(str, Enum):
    COMPLETE = "complete"
    INCOMPLETE = "incomplete"
    INVALIDATED = "invalidated"


class ExpansionHandleKind(str, Enum):
    SOURCE = "source"
    AST = "ast"
    GRAPH = "graph"
    PROOF = "proof"
    WITNESS = "witness"
    COUNTEREXAMPLE = "counterexample"
    EVIDENCE = "evidence"
    CALL_SLICE = "call_slice"
    DIAGNOSTIC = "diagnostic"
    OTHER = "other"


class DeltaEvidenceKind(str, Enum):
    CHANGED = "changed"
    REQUESTED = "requested"
    RETAINED = "retained"
    DEFERRED = "deferred"


# ---------------------------------------------------------------------------
# Bounded building blocks
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RepairPacketLimits:
    """Admission limits for a compiled repair packet."""

    max_packet_bytes: int = DEFAULT_MAX_PACKET_BYTES
    max_span_bytes: int = DEFAULT_MAX_SPAN_BYTES
    max_span_lines: int = DEFAULT_MAX_SPAN_LINES
    max_span_count: int = DEFAULT_MAX_SPAN_COUNT
    max_handle_count: int = DEFAULT_MAX_HANDLE_COUNT
    max_call_slice_steps: int = DEFAULT_MAX_CALL_SLICE_STEPS
    provider_input_budget_bytes: int = 0  # 0 = unlimited for budget checks

    def __post_init__(self) -> None:
        for name in (
            "max_packet_bytes",
            "max_span_bytes",
            "max_span_lines",
            "max_span_count",
            "max_handle_count",
            "max_call_slice_steps",
        ):
            object.__setattr__(
                self, name, _positive_int(getattr(self, name), name)
            )
        budget = self.provider_input_budget_bytes
        if isinstance(budget, bool) or not isinstance(budget, int) or budget < 0:
            raise ContractRepairPacketError(
                "provider_input_budget_bytes must be a non-negative integer"
            )

    def to_dict(self) -> dict[str, int]:
        return {
            "max_packet_bytes": self.max_packet_bytes,
            "max_span_bytes": self.max_span_bytes,
            "max_span_lines": self.max_span_lines,
            "max_span_count": self.max_span_count,
            "max_handle_count": self.max_handle_count,
            "max_call_slice_steps": self.max_call_slice_steps,
            "provider_input_budget_bytes": self.provider_input_budget_bytes,
        }

    @classmethod
    def from_value(cls, value: Any) -> "RepairPacketLimits":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise ContractRepairPacketError("limits must be a mapping")
        return cls(
            max_packet_bytes=int(
                value.get("max_packet_bytes", DEFAULT_MAX_PACKET_BYTES)
            ),
            max_span_bytes=int(
                value.get("max_span_bytes", DEFAULT_MAX_SPAN_BYTES)
            ),
            max_span_lines=int(
                value.get("max_span_lines", DEFAULT_MAX_SPAN_LINES)
            ),
            max_span_count=int(
                value.get("max_span_count", DEFAULT_MAX_SPAN_COUNT)
            ),
            max_handle_count=int(
                value.get("max_handle_count", DEFAULT_MAX_HANDLE_COUNT)
            ),
            max_call_slice_steps=int(
                value.get(
                    "max_call_slice_steps", DEFAULT_MAX_CALL_SLICE_STEPS
                )
            ),
            provider_input_budget_bytes=int(
                value.get("provider_input_budget_bytes", 0) or 0
            ),
        )


@dataclass(frozen=True)
class BoundedSourceSpan:
    """A path-bound excerpt that never carries a full file body."""

    path: str
    start_line: int
    end_line: int
    excerpt: str = ""
    content_id: str = ""
    symbol: str = ""

    SCHEMA: ClassVar[str] = BOUNDED_SOURCE_SPAN_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _required_text(self.path, "path"))
        start = _positive_int(self.start_line, "start_line", minimum=1)
        end = _positive_int(self.end_line, "end_line", minimum=1)
        if end < start:
            raise ContractRepairPacketError(
                "end_line must be >= start_line"
            )
        object.__setattr__(self, "start_line", start)
        object.__setattr__(self, "end_line", end)
        object.__setattr__(
            self, "symbol", _optional_text(self.symbol, "symbol")
        )
        excerpt = _optional_text(self.excerpt, "excerpt")
        excerpt, _ = _redact_inline(excerpt)
        if len(excerpt.encode("utf-8")) > DEFAULT_MAX_SPAN_BYTES:
            # Hard truncate to bound; full body lives behind content_id.
            encoded = excerpt.encode("utf-8")[: DEFAULT_MAX_SPAN_BYTES]
            excerpt = encoded.decode("utf-8", errors="ignore") + OMITTED
        object.__setattr__(self, "excerpt", excerpt)
        object.__setattr__(
            self,
            "content_id",
            _optional_text(self.content_id, "content_id"),
        )
        line_span = end - start + 1
        if line_span > DEFAULT_MAX_SPAN_LINES and not self.content_id:
            raise ContractRepairPacketError(
                "source span exceeds line bound without a content_id handle"
            )

    @property
    def span_id(self) -> str:
        return content_identity(self.to_dict(include_span_id=False))

    @property
    def byte_count(self) -> int:
        return len(canonical_json_bytes(self.to_dict(include_span_id=False)))

    def to_dict(self, *, include_span_id: bool = True) -> dict[str, Any]:
        result = {
            "schema": self.SCHEMA,
            "path": self.path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "excerpt": self.excerpt,
            "content_id": self.content_id,
            "symbol": self.symbol,
        }
        if include_span_id:
            result["span_id"] = self.span_id
        return result

    @classmethod
    def from_dict(cls, value: Any) -> "BoundedSourceSpan":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise ContractRepairPacketError("source span must be a mapping")
        return cls(
            path=value.get("path", ""),
            start_line=int(value.get("start_line", 1) or 1),
            end_line=int(value.get("end_line", 1) or 1),
            excerpt=value.get("excerpt", "") or "",
            content_id=value.get("content_id", "") or "",
            symbol=value.get("symbol", "") or "",
        )


@dataclass(frozen=True)
class CallSliceStepRef:
    """One compact call-slice step (ids only; no source/AST bodies)."""

    step_id: str
    symbol: str = ""
    path: str = ""
    kind: str = "call"
    contract_ref: str = ""
    content_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "step_id", _required_text(self.step_id, "step_id")
        )
        object.__setattr__(
            self, "symbol", _optional_text(self.symbol, "symbol")
        )
        object.__setattr__(self, "path", _optional_text(self.path, "path"))
        object.__setattr__(
            self, "kind", _optional_text(self.kind, "kind") or "call"
        )
        object.__setattr__(
            self,
            "contract_ref",
            _optional_text(self.contract_ref, "contract_ref"),
        )
        object.__setattr__(
            self,
            "content_id",
            _optional_text(self.content_id, "content_id"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "symbol": self.symbol,
            "path": self.path,
            "kind": self.kind,
            "contract_ref": self.contract_ref,
            "content_id": self.content_id,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "CallSliceStepRef":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise ContractRepairPacketError("call slice step must be a mapping")
        return cls(
            step_id=value.get("step_id", "") or value.get("id", ""),
            symbol=value.get("symbol", "") or "",
            path=value.get("path", "") or "",
            kind=value.get("kind", "call") or "call",
            contract_ref=value.get("contract_ref", "") or "",
            content_id=value.get("content_id", "") or "",
        )


@dataclass(frozen=True)
class CallSliceRef:
    """Minimal dependency-complete call-slice reference for a repair packet."""

    slice_id: str
    steps: tuple[CallSliceStepRef, ...] = ()
    root_symbol: str = ""
    complete: bool = True

    SCHEMA: ClassVar[str] = CALL_SLICE_REF_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "slice_id", _required_text(self.slice_id, "slice_id")
        )
        object.__setattr__(
            self, "root_symbol", _optional_text(self.root_symbol, "root_symbol")
        )
        object.__setattr__(self, "complete", bool(self.complete))
        steps_raw = self.steps or ()
        if isinstance(steps_raw, Mapping):
            raise ContractRepairPacketError("call slice steps must be a sequence")
        steps: list[CallSliceStepRef] = []
        for item in steps_raw:
            if isinstance(item, CallSliceStepRef):
                steps.append(item)
            elif isinstance(item, Mapping):
                steps.append(CallSliceStepRef.from_dict(item))
            else:
                raise ContractRepairPacketError(
                    "call slice steps must be mappings or CallSliceStepRef"
                )
        if len(steps) > DEFAULT_MAX_CALL_SLICE_STEPS:
            raise ContractRepairPacketError(
                f"call slice exceeds {DEFAULT_MAX_CALL_SLICE_STEPS} steps"
            )
        object.__setattr__(self, "steps", tuple(steps))

    @property
    def content_id(self) -> str:
        return content_identity(self.to_dict(include_content_id=False))

    def to_dict(self, *, include_content_id: bool = True) -> dict[str, Any]:
        result = {
            "schema": self.SCHEMA,
            "slice_id": self.slice_id,
            "steps": [step.to_dict() for step in self.steps],
            "root_symbol": self.root_symbol,
            "complete": self.complete,
        }
        if include_content_id:
            result["content_id"] = self.content_id
        return result

    @classmethod
    def from_dict(cls, value: Any) -> "CallSliceRef":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise ContractRepairPacketError("call_slice must be a mapping")
        return cls(
            slice_id=value.get("slice_id", "") or value.get("id", ""),
            steps=tuple(value.get("steps") or ()),
            root_symbol=value.get("root_symbol", "") or "",
            complete=bool(value.get("complete", True)),
        )


@dataclass(frozen=True)
class CounterexampleSliceRef:
    """Compact counterexample/proof reference (no witness body)."""

    counterexample_id: str
    kind: str = "counterexample"
    summary: str = ""
    content_id: str = ""
    proof_receipt_ref: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "counterexample_id",
            _required_text(self.counterexample_id, "counterexample_id"),
        )
        object.__setattr__(
            self, "kind", _optional_text(self.kind, "kind") or "counterexample"
        )
        summary = _optional_text(self.summary, "summary")
        summary, _ = _redact_inline(summary)
        if len(summary) > DEFAULT_MAX_TEXT_CHARS:
            summary = summary[:DEFAULT_MAX_TEXT_CHARS] + OMITTED
        object.__setattr__(self, "summary", summary)
        object.__setattr__(
            self, "content_id", _optional_text(self.content_id, "content_id")
        )
        object.__setattr__(
            self,
            "proof_receipt_ref",
            _optional_text(self.proof_receipt_ref, "proof_receipt_ref"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "counterexample_id": self.counterexample_id,
            "kind": self.kind,
            "summary": self.summary,
            "content_id": self.content_id,
            "proof_receipt_ref": self.proof_receipt_ref,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "CounterexampleSliceRef":
        if value is None:
            return cls(counterexample_id="none", kind="none", summary="")
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise ContractRepairPacketError(
                "counterexample_slice must be a mapping"
            )
        return cls(
            counterexample_id=value.get("counterexample_id", "")
            or value.get("id", ""),
            kind=value.get("kind", "counterexample") or "counterexample",
            summary=value.get("summary", "") or "",
            content_id=value.get("content_id", "") or "",
            proof_receipt_ref=value.get("proof_receipt_ref", "") or "",
        )


@dataclass(frozen=True)
class RepairExpansionHandle:
    """Content-addressed on-demand expansion for omitted optional material."""

    handle_id: str
    kind: ExpansionHandleKind
    referenced_content_id: str
    reference_id: str = ""
    reason: str = ""
    locator: str = ""
    tree_id: str = ""
    forest_id: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    SCHEMA: ClassVar[str] = REPAIR_EXPANSION_HANDLE_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "kind",
            _enum(self.kind, ExpansionHandleKind, "expansion handle kind"),
        )
        object.__setattr__(
            self,
            "referenced_content_id",
            _required_text(
                self.referenced_content_id, "referenced_content_id"
            ),
        )
        object.__setattr__(
            self,
            "reference_id",
            _optional_text(self.reference_id, "reference_id")
            or self.referenced_content_id,
        )
        object.__setattr__(
            self, "reason", _optional_text(self.reason, "reason")
        )
        object.__setattr__(
            self, "locator", _optional_text(self.locator, "locator")
        )
        object.__setattr__(
            self, "tree_id", _optional_text(self.tree_id, "tree_id")
        )
        object.__setattr__(
            self, "forest_id", _optional_text(self.forest_id, "forest_id")
        )
        if not isinstance(self.metadata, Mapping):
            raise ContractRepairPacketError("metadata must be a mapping")
        object.__setattr__(
            self, "metadata", _freeze(_plain(self.metadata, "metadata"))
        )
        computed = content_identity(self._identity_payload())
        supplied = _optional_text(self.handle_id, "handle_id")
        if supplied and supplied != computed:
            raise RepairPacketIntegrityError(
                "expansion handle_id does not match content identity"
            )
        object.__setattr__(self, "handle_id", computed)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "kind": self.kind.value,
            "referenced_content_id": self.referenced_content_id,
            "reference_id": self.reference_id,
            "reason": self.reason,
            "locator": self.locator,
            "tree_id": self.tree_id,
            "forest_id": self.forest_id,
            "metadata": _thaw(self.metadata),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._identity_payload(), "handle_id": self.handle_id}

    @classmethod
    def from_dict(cls, value: Any) -> "RepairExpansionHandle":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise ContractRepairPacketError(
                "expansion handle must be a mapping"
            )
        return cls(
            handle_id=value.get("handle_id", "") or "",
            kind=value.get("kind", ExpansionHandleKind.OTHER.value),
            referenced_content_id=value.get("referenced_content_id", "")
            or value.get("content_id", ""),
            reference_id=value.get("reference_id", "") or "",
            reason=value.get("reason", "") or "",
            locator=value.get("locator", "") or "",
            tree_id=value.get("tree_id", "") or "",
            forest_id=value.get("forest_id", "") or "",
            metadata=value.get("metadata") or {},
        )


@dataclass(frozen=True)
class RepairAuthority:
    """Explicit trust boundary for model-facing packets."""

    mode: str = "proposal"
    semantic_authority: bool = False
    completion_authoritative: bool = False
    proof_authoritative: bool = False
    allowed_paths: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        mode = _optional_text(self.mode, "mode") or "proposal"
        if mode not in {"proposal", "proposal_only", "model_proposal"}:
            # Only proposal modes are admitted for model packets.
            raise ContractRepairPacketError(
                "repair packet authority.mode must be a proposal mode"
            )
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "semantic_authority", False)
        object.__setattr__(self, "completion_authoritative", False)
        object.__setattr__(self, "proof_authoritative", False)
        object.__setattr__(
            self,
            "allowed_paths",
            _sorted_unique(self.allowed_paths, name="allowed_paths"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "semantic_authority": False,
            "completion_authoritative": False,
            "proof_authoritative": False,
            "allowed_paths": list(self.allowed_paths),
            "model_output_is_proposal": True,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "RepairAuthority":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise ContractRepairPacketError("authority must be a mapping")
        # Reject attempts to claim authority via dict forgery.
        if value.get("semantic_authority") is True:
            raise ContractRepairPacketError(
                "repair packets cannot claim semantic_authority"
            )
        if value.get("completion_authoritative") is True:
            raise ContractRepairPacketError(
                "repair packets cannot claim completion_authoritative"
            )
        if value.get("proof_authoritative") is True:
            raise ContractRepairPacketError(
                "repair packets cannot claim proof_authoritative"
            )
        return cls(
            mode=value.get("mode", "proposal") or "proposal",
            allowed_paths=tuple(value.get("allowed_paths") or ()),
        )


# ---------------------------------------------------------------------------
# Request / packet / receipt
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RepairPacketRequest:
    """Inputs used to compile one compact repair packet."""

    task_id: str
    finding_ids: tuple[str, ...]
    forest_id: str
    tree_id: str
    policy_id: str
    expected_contract_ref: str
    observed_contract_ref: str
    call_slice: CallSliceRef
    edit_scope: tuple[str, ...]
    effects: tuple[str, ...]
    acceptance: tuple[str, ...]
    validation_commands: tuple[str, ...]
    proof_commands: tuple[str, ...]
    risks: tuple[str, ...]
    policy_revision: str = ""
    goal_id: str = ""
    symbols: tuple[str, ...] = ()
    interfaces: tuple[str, ...] = ()
    counterexample_slice: CounterexampleSliceRef | None = None
    source_spans: tuple[BoundedSourceSpan, ...] = ()
    optional_evidence: tuple[Mapping[str, Any], ...] = ()
    expansion_candidates: tuple[RepairExpansionHandle, ...] = ()
    related_finding_ids: tuple[str, ...] = ()
    superseded_finding_ids: tuple[str, ...] = ()
    authority: RepairAuthority = field(default_factory=RepairAuthority)
    limits: RepairPacketLimits = field(default_factory=RepairPacketLimits)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    repository_id: str = ""
    decision_id: str = ""

    SCHEMA: ClassVar[str] = REPAIR_PACKET_REQUEST_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "task_id", _required_text(self.task_id, "task_id")
        )
        object.__setattr__(
            self,
            "finding_ids",
            _sorted_unique(self.finding_ids, name="finding_ids", required=True),
        )
        object.__setattr__(
            self, "forest_id", _required_text(self.forest_id, "forest_id")
        )
        object.__setattr__(
            self, "tree_id", _required_text(self.tree_id, "tree_id")
        )
        object.__setattr__(
            self, "policy_id", _required_text(self.policy_id, "policy_id")
        )
        object.__setattr__(
            self,
            "expected_contract_ref",
            _required_text(
                self.expected_contract_ref, "expected_contract_ref"
            ),
        )
        object.__setattr__(
            self,
            "observed_contract_ref",
            _required_text(
                self.observed_contract_ref, "observed_contract_ref"
            ),
        )
        object.__setattr__(
            self, "call_slice", CallSliceRef.from_dict(self.call_slice)
        )
        object.__setattr__(
            self,
            "edit_scope",
            _sorted_unique(self.edit_scope, name="edit_scope", required=True),
        )
        object.__setattr__(
            self,
            "effects",
            _sorted_unique(self.effects, name="effects", required=True),
        )
        object.__setattr__(
            self,
            "acceptance",
            _sorted_unique(self.acceptance, name="acceptance", required=True),
        )
        object.__setattr__(
            self,
            "validation_commands",
            _sorted_unique(
                self.validation_commands,
                name="validation_commands",
                required=True,
            ),
        )
        object.__setattr__(
            self,
            "proof_commands",
            _sorted_unique(
                self.proof_commands, name="proof_commands", required=True
            ),
        )
        object.__setattr__(
            self,
            "risks",
            _sorted_unique(self.risks, name="risks", required=True),
        )
        object.__setattr__(
            self,
            "policy_revision",
            _optional_text(self.policy_revision, "policy_revision"),
        )
        object.__setattr__(
            self, "goal_id", _optional_text(self.goal_id, "goal_id")
        )
        object.__setattr__(
            self, "symbols", _sorted_unique(self.symbols, name="symbols")
        )
        object.__setattr__(
            self,
            "interfaces",
            _sorted_unique(self.interfaces, name="interfaces"),
        )
        if self.counterexample_slice is None:
            cex = CounterexampleSliceRef(
                counterexample_id="none", kind="none", summary=""
            )
        else:
            cex = CounterexampleSliceRef.from_dict(self.counterexample_slice)
        object.__setattr__(self, "counterexample_slice", cex)
        spans: list[BoundedSourceSpan] = []
        for item in self.source_spans or ():
            spans.append(BoundedSourceSpan.from_dict(item))
        object.__setattr__(self, "source_spans", tuple(spans))
        optional: list[Mapping[str, Any]] = []
        for index, item in enumerate(self.optional_evidence or ()):
            if not isinstance(item, Mapping):
                raise ContractRepairPacketError(
                    f"optional_evidence[{index}] must be a mapping"
                )
            optional.append(
                _freeze(_plain(item, f"optional_evidence[{index}]"))
            )
        object.__setattr__(self, "optional_evidence", tuple(optional))
        handles: list[RepairExpansionHandle] = []
        for item in self.expansion_candidates or ():
            handles.append(RepairExpansionHandle.from_dict(item))
        object.__setattr__(self, "expansion_candidates", tuple(handles))
        object.__setattr__(
            self,
            "related_finding_ids",
            _sorted_unique(
                self.related_finding_ids, name="related_finding_ids"
            ),
        )
        object.__setattr__(
            self,
            "superseded_finding_ids",
            _sorted_unique(
                self.superseded_finding_ids, name="superseded_finding_ids"
            ),
        )
        object.__setattr__(
            self, "authority", RepairAuthority.from_dict(self.authority)
        )
        # Bind edit scope into authority allowed_paths when empty.
        if not self.authority.allowed_paths:
            object.__setattr__(
                self,
                "authority",
                RepairAuthority(
                    mode=self.authority.mode,
                    allowed_paths=self.edit_scope,
                ),
            )
        object.__setattr__(
            self, "limits", RepairPacketLimits.from_value(self.limits)
        )
        if not isinstance(self.metadata, Mapping):
            raise ContractRepairPacketError("metadata must be a mapping")
        object.__setattr__(
            self, "metadata", _freeze(_plain(self.metadata, "metadata"))
        )
        object.__setattr__(
            self,
            "repository_id",
            _optional_text(self.repository_id, "repository_id"),
        )
        object.__setattr__(
            self,
            "decision_id",
            _optional_text(self.decision_id, "decision_id"),
        )
        if len(self.call_slice.steps) > self.limits.max_call_slice_steps:
            raise ContractRepairPacketError(
                "call slice exceeds limits.max_call_slice_steps"
            )
        if len(self.source_spans) > self.limits.max_span_count:
            raise ContractRepairPacketError(
                "source_spans exceed limits.max_span_count"
            )

    @property
    def request_id(self) -> str:
        return content_identity(self.to_dict(include_request_id=False))

    def to_dict(self, *, include_request_id: bool = True) -> dict[str, Any]:
        result = {
            "schema": self.SCHEMA,
            "version": CONTRACT_REPAIR_PACKET_VERSION,
            "task_id": self.task_id,
            "finding_ids": list(self.finding_ids),
            "forest_id": self.forest_id,
            "tree_id": self.tree_id,
            "policy_id": self.policy_id,
            "policy_revision": self.policy_revision,
            "goal_id": self.goal_id,
            "repository_id": self.repository_id,
            "decision_id": self.decision_id,
            "expected_contract_ref": self.expected_contract_ref,
            "observed_contract_ref": self.observed_contract_ref,
            "call_slice": self.call_slice.to_dict(),
            "counterexample_slice": self.counterexample_slice.to_dict(),
            "edit_scope": list(self.edit_scope),
            "effects": list(self.effects),
            "acceptance": list(self.acceptance),
            "validation_commands": list(self.validation_commands),
            "proof_commands": list(self.proof_commands),
            "risks": list(self.risks),
            "symbols": list(self.symbols),
            "interfaces": list(self.interfaces),
            "source_spans": [span.to_dict() for span in self.source_spans],
            "optional_evidence": [_thaw(item) for item in self.optional_evidence],
            "expansion_candidates": [
                item.to_dict() for item in self.expansion_candidates
            ],
            "related_finding_ids": list(self.related_finding_ids),
            "superseded_finding_ids": list(self.superseded_finding_ids),
            "authority": self.authority.to_dict(),
            "limits": self.limits.to_dict(),
            "metadata": _thaw(self.metadata),
        }
        if include_request_id:
            result["request_id"] = self.request_id
        return result

    @classmethod
    def from_dict(cls, value: Any) -> "RepairPacketRequest":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise ContractRepairPacketError("request must be a mapping")
        known = {
            "task_id",
            "finding_ids",
            "forest_id",
            "tree_id",
            "policy_id",
            "expected_contract_ref",
            "observed_contract_ref",
            "call_slice",
            "edit_scope",
            "effects",
            "acceptance",
            "validation_commands",
            "proof_commands",
            "risks",
            "policy_revision",
            "goal_id",
            "symbols",
            "interfaces",
            "counterexample_slice",
            "source_spans",
            "optional_evidence",
            "expansion_candidates",
            "related_finding_ids",
            "superseded_finding_ids",
            "authority",
            "limits",
            "metadata",
            "repository_id",
            "decision_id",
        }
        kwargs = {key: value[key] for key in known if key in value}
        # Normalize sequences that dataclasses expect as tuples.
        for key in (
            "finding_ids",
            "edit_scope",
            "effects",
            "acceptance",
            "validation_commands",
            "proof_commands",
            "risks",
            "symbols",
            "interfaces",
            "source_spans",
            "optional_evidence",
            "expansion_candidates",
            "related_finding_ids",
            "superseded_finding_ids",
        ):
            if key in kwargs and kwargs[key] is not None:
                kwargs[key] = tuple(kwargs[key])
        return cls(**kwargs)


@dataclass(frozen=True)
class ContractRepairPacket:
    """Canonical model-facing compact repair packet."""

    task_id: str
    finding_ids: tuple[str, ...]
    forest_id: str
    tree_id: str
    policy_id: str
    expected_contract_ref: str
    observed_contract_ref: str
    call_slice: CallSliceRef
    edit_scope: tuple[str, ...]
    effects: tuple[str, ...]
    acceptance: tuple[str, ...]
    validation_commands: tuple[str, ...]
    proof_commands: tuple[str, ...]
    risks: tuple[str, ...]
    authority: RepairAuthority
    expansion_handles: tuple[RepairExpansionHandle, ...]
    counterexample_slice: CounterexampleSliceRef
    source_spans: tuple[BoundedSourceSpan, ...] = ()
    policy_revision: str = ""
    goal_id: str = ""
    symbols: tuple[str, ...] = ()
    interfaces: tuple[str, ...] = ()
    related_finding_ids: tuple[str, ...] = ()
    superseded_finding_ids: tuple[str, ...] = ()
    optional_evidence: tuple[Mapping[str, Any], ...] = ()
    omitted_optional_ids: tuple[str, ...] = ()
    repository_id: str = ""
    decision_id: str = ""
    status: RepairPacketStatus = RepairPacketStatus.COMPLETE
    incomplete_reasons: tuple[str, ...] = ()
    packet_byte_count: int = 0
    span_byte_count: int = 0
    estimated_tokens: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    SCHEMA: ClassVar[str] = REPAIR_PACKET_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "task_id", _required_text(self.task_id, "task_id")
        )
        object.__setattr__(
            self,
            "finding_ids",
            _sorted_unique(self.finding_ids, name="finding_ids", required=True),
        )
        object.__setattr__(
            self, "forest_id", _required_text(self.forest_id, "forest_id")
        )
        object.__setattr__(
            self, "tree_id", _required_text(self.tree_id, "tree_id")
        )
        object.__setattr__(
            self, "policy_id", _required_text(self.policy_id, "policy_id")
        )
        object.__setattr__(
            self,
            "expected_contract_ref",
            _required_text(
                self.expected_contract_ref, "expected_contract_ref"
            ),
        )
        object.__setattr__(
            self,
            "observed_contract_ref",
            _required_text(
                self.observed_contract_ref, "observed_contract_ref"
            ),
        )
        object.__setattr__(
            self,
            "edit_scope",
            _sorted_unique(self.edit_scope, name="edit_scope", required=True),
        )
        object.__setattr__(
            self,
            "effects",
            _sorted_unique(self.effects, name="effects", required=True),
        )
        object.__setattr__(
            self,
            "acceptance",
            _sorted_unique(self.acceptance, name="acceptance", required=True),
        )
        object.__setattr__(
            self,
            "validation_commands",
            _sorted_unique(
                self.validation_commands,
                name="validation_commands",
                required=True,
            ),
        )
        object.__setattr__(
            self,
            "proof_commands",
            _sorted_unique(
                self.proof_commands, name="proof_commands", required=True
            ),
        )
        object.__setattr__(
            self,
            "risks",
            _sorted_unique(self.risks, name="risks", required=True),
        )
        object.__setattr__(
            self,
            "policy_revision",
            _optional_text(self.policy_revision, "policy_revision"),
        )
        object.__setattr__(
            self, "goal_id", _optional_text(self.goal_id, "goal_id")
        )
        object.__setattr__(
            self, "symbols", _sorted_unique(self.symbols, name="symbols")
        )
        object.__setattr__(
            self,
            "interfaces",
            _sorted_unique(self.interfaces, name="interfaces"),
        )
        object.__setattr__(
            self,
            "related_finding_ids",
            _sorted_unique(
                self.related_finding_ids, name="related_finding_ids"
            ),
        )
        object.__setattr__(
            self,
            "superseded_finding_ids",
            _sorted_unique(
                self.superseded_finding_ids, name="superseded_finding_ids"
            ),
        )
        object.__setattr__(
            self,
            "omitted_optional_ids",
            _sorted_unique(
                self.omitted_optional_ids, name="omitted_optional_ids"
            ),
        )
        object.__setattr__(
            self,
            "repository_id",
            _optional_text(self.repository_id, "repository_id"),
        )
        object.__setattr__(
            self,
            "decision_id",
            _optional_text(self.decision_id, "decision_id"),
        )
        object.__setattr__(
            self, "status", _enum(self.status, RepairPacketStatus, "status")
        )
        object.__setattr__(
            self,
            "incomplete_reasons",
            _sorted_unique(
                self.incomplete_reasons, name="incomplete_reasons"
            ),
        )
        object.__setattr__(
            self, "authority", RepairAuthority.from_dict(self.authority)
        )
        object.__setattr__(
            self, "call_slice", CallSliceRef.from_dict(self.call_slice)
        )
        object.__setattr__(
            self,
            "counterexample_slice",
            CounterexampleSliceRef.from_dict(self.counterexample_slice),
        )
        object.__setattr__(
            self,
            "expansion_handles",
            tuple(
                RepairExpansionHandle.from_dict(item)
                for item in self.expansion_handles or ()
            ),
        )
        object.__setattr__(
            self,
            "source_spans",
            tuple(
                BoundedSourceSpan.from_dict(item)
                for item in self.source_spans or ()
            ),
        )
        optional: list[Mapping[str, Any]] = []
        for index, item in enumerate(self.optional_evidence or ()):
            if not isinstance(item, Mapping):
                raise ContractRepairPacketError(
                    f"optional_evidence[{index}] must be a mapping"
                )
            optional.append(
                _freeze(_plain(item, f"optional_evidence[{index}]"))
            )
        object.__setattr__(self, "optional_evidence", tuple(optional))
        if not isinstance(self.metadata, Mapping):
            raise ContractRepairPacketError("metadata must be a mapping")
        object.__setattr__(
            self, "metadata", _freeze(_plain(self.metadata, "metadata"))
        )
        # Recompute size fields if not supplied (or zero).
        core = self._core_payload()
        core_bytes = len(canonical_json_bytes(core))
        span_bytes = sum(span.byte_count for span in self.source_spans)
        if self.packet_byte_count <= 0:
            object.__setattr__(self, "packet_byte_count", core_bytes)
        if self.span_byte_count <= 0:
            object.__setattr__(self, "span_byte_count", span_bytes)
        if self.estimated_tokens <= 0:
            object.__setattr__(
                self,
                "estimated_tokens",
                estimate_tokens(self.to_dict(include_packet_id=False)),
            )

    def _core_payload(self) -> dict[str, Any]:
        """Canonical payload excluding source spans (16 KiB budget base)."""

        return {
            "schema": self.SCHEMA,
            "version": CONTRACT_REPAIR_PACKET_VERSION,
            "evidence": COMPACT_REPAIR_PACKET_EVIDENCE,
            "task_id": self.task_id,
            "finding_ids": list(self.finding_ids),
            "forest_id": self.forest_id,
            "tree_id": self.tree_id,
            "policy_id": self.policy_id,
            "policy_revision": self.policy_revision,
            "goal_id": self.goal_id,
            "repository_id": self.repository_id,
            "decision_id": self.decision_id,
            "expected_contract_ref": self.expected_contract_ref,
            "observed_contract_ref": self.observed_contract_ref,
            "call_slice": self.call_slice.to_dict(),
            "counterexample_slice": self.counterexample_slice.to_dict(),
            "edit_scope": list(self.edit_scope),
            "effects": list(self.effects),
            "acceptance": list(self.acceptance),
            "validation_commands": list(self.validation_commands),
            "proof_commands": list(self.proof_commands),
            "risks": list(self.risks),
            "symbols": list(self.symbols),
            "interfaces": list(self.interfaces),
            "related_finding_ids": list(self.related_finding_ids),
            "superseded_finding_ids": list(self.superseded_finding_ids),
            "optional_evidence": [_thaw(item) for item in self.optional_evidence],
            "omitted_optional_ids": list(self.omitted_optional_ids),
            "expansion_handles": [
                item.to_dict() for item in self.expansion_handles
            ],
            "authority": self.authority.to_dict(),
            "status": self.status.value,
            "incomplete_reasons": list(self.incomplete_reasons),
            "metadata": _thaw(self.metadata),
            "embeds_full_source": False,
            "embeds_full_ast": False,
            "embeds_full_graph": False,
            "embeds_full_proof": False,
            "embeds_full_witness": False,
            "required_fields_truncated": False,
            "model_output_is_proposal": True,
        }

    @property
    def packet_id(self) -> str:
        return content_identity(self.to_dict(include_packet_id=False))

    @property
    def content_id(self) -> str:
        return self.packet_id

    @property
    def total_byte_count(self) -> int:
        return int(self.packet_byte_count) + int(self.span_byte_count)

    @property
    def required_core_present(self) -> bool:
        core = self.to_dict(include_packet_id=False)
        for field_name in REQUIRED_CORE_FIELDS:
            if field_name not in core:
                return False
            value = core[field_name]
            if value is None or value == "" or value == [] or value == {}:
                return False
        return True

    def to_dict(self, *, include_packet_id: bool = True) -> dict[str, Any]:
        result = self._core_payload()
        result["source_spans"] = [span.to_dict() for span in self.source_spans]
        result["packet_byte_count"] = int(self.packet_byte_count)
        result["span_byte_count"] = int(self.span_byte_count)
        result["total_byte_count"] = self.total_byte_count
        result["estimated_tokens"] = int(self.estimated_tokens)
        if include_packet_id:
            result["packet_id"] = self.packet_id
        return result

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    def provider_payload(self) -> dict[str, Any]:
        """Payload intended for a provider prompt (canonical, complete)."""

        return self.to_dict()

    def core_without_spans(self) -> dict[str, Any]:
        return self._core_payload()

    @classmethod
    def from_dict(cls, value: Any) -> "ContractRepairPacket":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise ContractRepairPacketError("repair packet must be a mapping")
        packet = cls(
            task_id=value.get("task_id", ""),
            finding_ids=tuple(value.get("finding_ids") or ()),
            forest_id=value.get("forest_id", ""),
            tree_id=value.get("tree_id", ""),
            policy_id=value.get("policy_id", ""),
            expected_contract_ref=value.get("expected_contract_ref", ""),
            observed_contract_ref=value.get("observed_contract_ref", ""),
            call_slice=value.get("call_slice") or {},
            edit_scope=tuple(value.get("edit_scope") or ()),
            effects=tuple(value.get("effects") or ()),
            acceptance=tuple(value.get("acceptance") or ()),
            validation_commands=tuple(value.get("validation_commands") or ()),
            proof_commands=tuple(value.get("proof_commands") or ()),
            risks=tuple(value.get("risks") or ()),
            authority=value.get("authority") or {},
            expansion_handles=tuple(value.get("expansion_handles") or ()),
            counterexample_slice=value.get("counterexample_slice"),
            source_spans=tuple(value.get("source_spans") or ()),
            policy_revision=value.get("policy_revision", "") or "",
            goal_id=value.get("goal_id", "") or "",
            symbols=tuple(value.get("symbols") or ()),
            interfaces=tuple(value.get("interfaces") or ()),
            related_finding_ids=tuple(value.get("related_finding_ids") or ()),
            superseded_finding_ids=tuple(
                value.get("superseded_finding_ids") or ()
            ),
            optional_evidence=tuple(value.get("optional_evidence") or ()),
            omitted_optional_ids=tuple(
                value.get("omitted_optional_ids") or ()
            ),
            repository_id=value.get("repository_id", "") or "",
            decision_id=value.get("decision_id", "") or "",
            status=value.get("status", RepairPacketStatus.COMPLETE.value),
            incomplete_reasons=tuple(value.get("incomplete_reasons") or ()),
            packet_byte_count=int(value.get("packet_byte_count", 0) or 0),
            span_byte_count=int(value.get("span_byte_count", 0) or 0),
            estimated_tokens=int(value.get("estimated_tokens", 0) or 0),
            metadata=value.get("metadata") or {},
        )
        supplied = value.get("packet_id") or value.get("content_id")
        if supplied and supplied != packet.packet_id:
            raise RepairPacketIntegrityError(
                "packet_id does not match content identity"
            )
        return packet


@dataclass(frozen=True)
class RepairPacketReceipt:
    """Auditable compile receipt for a repair packet."""

    packet_id: str
    request_id: str
    tree_id: str
    forest_id: str
    policy_id: str
    status: RepairPacketStatus
    packet_byte_count: int
    span_byte_count: int
    estimated_tokens: int
    required_fields: tuple[str, ...]
    expansion_handle_ids: tuple[str, ...]
    omitted_optional_ids: tuple[str, ...]
    incomplete_reasons: tuple[str, ...] = ()
    decision_id: str = ""

    SCHEMA: ClassVar[str] = REPAIR_PACKET_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "status", _enum(self.status, RepairPacketStatus, "status")
        )
        for name in (
            "packet_id",
            "request_id",
            "tree_id",
            "forest_id",
            "policy_id",
        ):
            object.__setattr__(
                self, name, _required_text(getattr(self, name), name)
            )

    @property
    def receipt_id(self) -> str:
        return content_identity(self.to_dict(include_receipt_id=False))

    def to_dict(self, *, include_receipt_id: bool = True) -> dict[str, Any]:
        result = {
            "schema": self.SCHEMA,
            "version": CONTRACT_REPAIR_PACKET_VERSION,
            "evidence": COMPACT_REPAIR_PACKET_EVIDENCE,
            "packet_id": self.packet_id,
            "request_id": self.request_id,
            "tree_id": self.tree_id,
            "forest_id": self.forest_id,
            "policy_id": self.policy_id,
            "decision_id": self.decision_id,
            "status": self.status.value,
            "packet_byte_count": self.packet_byte_count,
            "span_byte_count": self.span_byte_count,
            "estimated_tokens": self.estimated_tokens,
            "required_fields": list(self.required_fields),
            "expansion_handle_ids": list(self.expansion_handle_ids),
            "omitted_optional_ids": list(self.omitted_optional_ids),
            "incomplete_reasons": list(self.incomplete_reasons),
            "required_fields_truncated": False,
        }
        if include_receipt_id:
            result["receipt_id"] = self.receipt_id
        return result

    @classmethod
    def from_dict(cls, value: Any) -> "RepairPacketReceipt":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise ContractRepairPacketError("receipt must be a mapping")
        receipt = cls(
            packet_id=value.get("packet_id", ""),
            request_id=value.get("request_id", ""),
            tree_id=value.get("tree_id", ""),
            forest_id=value.get("forest_id", ""),
            policy_id=value.get("policy_id", ""),
            status=value.get("status", RepairPacketStatus.COMPLETE.value),
            packet_byte_count=int(value.get("packet_byte_count", 0) or 0),
            span_byte_count=int(value.get("span_byte_count", 0) or 0),
            estimated_tokens=int(value.get("estimated_tokens", 0) or 0),
            required_fields=tuple(value.get("required_fields") or ()),
            expansion_handle_ids=tuple(
                value.get("expansion_handle_ids") or ()
            ),
            omitted_optional_ids=tuple(
                value.get("omitted_optional_ids") or ()
            ),
            incomplete_reasons=tuple(value.get("incomplete_reasons") or ()),
            decision_id=value.get("decision_id", "") or "",
        )
        supplied = value.get("receipt_id")
        if supplied and supplied != receipt.receipt_id:
            raise RepairPacketIntegrityError(
                "receipt_id does not match content identity"
            )
        return receipt


@dataclass(frozen=True)
class CompiledRepairPacket:
    """Compile result binding packet + receipt."""

    packet: ContractRepairPacket
    receipt: RepairPacketReceipt
    request_id: str

    @property
    def packet_id(self) -> str:
        return self.packet.packet_id

    @property
    def status(self) -> RepairPacketStatus:
        return self.packet.status

    def to_dict(self) -> dict[str, Any]:
        return {
            "packet": self.packet.to_dict(),
            "receipt": self.receipt.to_dict(),
            "request_id": self.request_id,
            "packet_id": self.packet_id,
            "status": self.status.value,
        }


# ---------------------------------------------------------------------------
# Delta retry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DeltaEvidenceItem:
    """One changed or requested evidence item for a delta retry."""

    evidence_id: str
    kind: DeltaEvidenceKind
    content_id: str
    summary: str = ""
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "evidence_id",
            _required_text(self.evidence_id, "evidence_id"),
        )
        object.__setattr__(
            self, "kind", _enum(self.kind, DeltaEvidenceKind, "delta kind")
        )
        object.__setattr__(
            self, "content_id", _required_text(self.content_id, "content_id")
        )
        summary = _optional_text(self.summary, "summary")
        summary, _ = _redact_inline(summary)
        if len(summary) > DEFAULT_MAX_TEXT_CHARS:
            summary = summary[:DEFAULT_MAX_TEXT_CHARS] + OMITTED
        object.__setattr__(self, "summary", summary)
        if not isinstance(self.payload, Mapping):
            raise ContractRepairPacketError("delta payload must be a mapping")
        object.__setattr__(
            self, "payload", _freeze(_plain(self.payload, "payload"))
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "kind": self.kind.value,
            "content_id": self.content_id,
            "summary": self.summary,
            "payload": _thaw(self.payload),
        }

    @classmethod
    def from_dict(cls, value: Any) -> "DeltaEvidenceItem":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise ContractRepairPacketError(
                "delta evidence item must be a mapping"
            )
        return cls(
            evidence_id=value.get("evidence_id", "") or value.get("id", ""),
            kind=value.get("kind", DeltaEvidenceKind.CHANGED.value),
            content_id=value.get("content_id", ""),
            summary=value.get("summary", "") or "",
            payload=value.get("payload") or {},
        )


@dataclass(frozen=True)
class RepairPacketDelta:
    """Delta-retry packet bound to a prior decision/packet identity."""

    parent_packet_id: str
    parent_decision_id: str
    parent_tree_id: str
    parent_forest_id: str
    parent_policy_id: str
    changed_evidence: tuple[DeltaEvidenceItem, ...]
    requested_evidence: tuple[DeltaEvidenceItem, ...]
    expansion_handles: tuple[RepairExpansionHandle, ...] = ()
    status: RepairPacketStatus = RepairPacketStatus.COMPLETE
    incomplete_reasons: tuple[str, ...] = ()
    delta_byte_count: int = 0
    estimated_tokens: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    SCHEMA: ClassVar[str] = REPAIR_PACKET_DELTA_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "parent_packet_id",
            "parent_decision_id",
            "parent_tree_id",
            "parent_forest_id",
            "parent_policy_id",
        ):
            object.__setattr__(
                self, name, _required_text(getattr(self, name), name)
            )
        object.__setattr__(
            self, "status", _enum(self.status, RepairPacketStatus, "status")
        )
        changed = tuple(
            DeltaEvidenceItem.from_dict(item)
            for item in self.changed_evidence or ()
        )
        requested = tuple(
            DeltaEvidenceItem.from_dict(item)
            for item in self.requested_evidence or ()
        )
        object.__setattr__(self, "changed_evidence", changed)
        object.__setattr__(self, "requested_evidence", requested)
        object.__setattr__(
            self,
            "expansion_handles",
            tuple(
                RepairExpansionHandle.from_dict(item)
                for item in self.expansion_handles or ()
            ),
        )
        object.__setattr__(
            self,
            "incomplete_reasons",
            _sorted_unique(
                self.incomplete_reasons, name="incomplete_reasons"
            ),
        )
        if not isinstance(self.metadata, Mapping):
            raise ContractRepairPacketError("metadata must be a mapping")
        object.__setattr__(
            self, "metadata", _freeze(_plain(self.metadata, "metadata"))
        )
        # Delta must transmit something.
        if (
            not changed
            and not requested
            and self.status is not RepairPacketStatus.INVALIDATED
        ):
            raise ContractRepairPacketError(
                "delta retry must include changed or requested evidence"
            )
        payload = self.to_dict(include_delta_id=False)
        byte_count = len(canonical_json_bytes(payload))
        if self.delta_byte_count <= 0:
            object.__setattr__(self, "delta_byte_count", byte_count)
        if self.estimated_tokens <= 0:
            object.__setattr__(
                self, "estimated_tokens", estimate_tokens(payload)
            )

    @property
    def delta_id(self) -> str:
        return content_identity(self.to_dict(include_delta_id=False))

    def to_dict(self, *, include_delta_id: bool = True) -> dict[str, Any]:
        result = {
            "schema": self.SCHEMA,
            "version": CONTRACT_REPAIR_PACKET_VERSION,
            "evidence": DELTA_REPAIR_CONTEXT_EVIDENCE,
            "parent_packet_id": self.parent_packet_id,
            "parent_decision_id": self.parent_decision_id,
            "parent_tree_id": self.parent_tree_id,
            "parent_forest_id": self.parent_forest_id,
            "parent_policy_id": self.parent_policy_id,
            "changed_evidence": [
                item.to_dict() for item in self.changed_evidence
            ],
            "requested_evidence": [
                item.to_dict() for item in self.requested_evidence
            ],
            "expansion_handles": [
                item.to_dict() for item in self.expansion_handles
            ],
            "status": self.status.value,
            "incomplete_reasons": list(self.incomplete_reasons),
            "delta_byte_count": int(self.delta_byte_count),
            "estimated_tokens": int(self.estimated_tokens),
            "metadata": _thaw(self.metadata),
            "omits_inherited_invariant_core": True,
            "model_output_is_proposal": True,
        }
        if include_delta_id:
            result["delta_id"] = self.delta_id
        return result

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: Any) -> "RepairPacketDelta":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise ContractRepairPacketError("delta packet must be a mapping")
        delta = cls(
            parent_packet_id=value.get("parent_packet_id", ""),
            parent_decision_id=value.get("parent_decision_id", ""),
            parent_tree_id=value.get("parent_tree_id", ""),
            parent_forest_id=value.get("parent_forest_id", ""),
            parent_policy_id=value.get("parent_policy_id", ""),
            changed_evidence=tuple(value.get("changed_evidence") or ()),
            requested_evidence=tuple(value.get("requested_evidence") or ()),
            expansion_handles=tuple(value.get("expansion_handles") or ()),
            status=value.get("status", RepairPacketStatus.COMPLETE.value),
            incomplete_reasons=tuple(value.get("incomplete_reasons") or ()),
            delta_byte_count=int(value.get("delta_byte_count", 0) or 0),
            estimated_tokens=int(value.get("estimated_tokens", 0) or 0),
            metadata=value.get("metadata") or {},
        )
        supplied = value.get("delta_id")
        if supplied and supplied != delta.delta_id:
            raise RepairPacketIntegrityError(
                "delta_id does not match content identity"
            )
        return delta


# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------


def _optional_evidence_id(item: Mapping[str, Any], index: int) -> str:
    for key in ("evidence_id", "id", "reference_id", "content_id"):
        text = _optional_text(item.get(key), key)
        if text:
            return text
    return f"optional:{index}"


def _handle_for_optional(
    item: Mapping[str, Any],
    *,
    index: int,
    tree_id: str,
    forest_id: str,
    reason: str,
) -> RepairExpansionHandle:
    evidence_id = _optional_evidence_id(item, index)
    content = _optional_text(item.get("content_id"), "content_id") or (
        content_identity(_thaw(item))
    )
    kind_raw = _optional_text(item.get("kind"), "kind") or "evidence"
    try:
        kind = ExpansionHandleKind(kind_raw)
    except ValueError:
        kind = ExpansionHandleKind.OTHER
    return RepairExpansionHandle(
        handle_id="",
        kind=kind,
        referenced_content_id=content,
        reference_id=evidence_id,
        reason=reason,
        locator=_optional_text(item.get("locator"), "locator"),
        tree_id=tree_id,
        forest_id=forest_id,
        metadata={
            "omission_reason": reason,
            "summary": _optional_text(item.get("summary"), "summary")[:240],
        },
    )


def _measure_core_bytes(packet_fields: Mapping[str, Any]) -> int:
    return len(canonical_json_bytes(packet_fields))


def compile_repair_packet(
    request: RepairPacketRequest | Mapping[str, Any],
) -> CompiledRepairPacket:
    """Compile a compact CID-addressed repair packet from ``request``."""

    if not isinstance(request, RepairPacketRequest):
        request = RepairPacketRequest.from_dict(request)

    limits = request.limits
    decision_id = request.decision_id or content_identity(
        {
            "task_id": request.task_id,
            "tree_id": request.tree_id,
            "finding_ids": list(request.finding_ids),
            "request_id": request.request_id,
        }
    )

    # Optional evidence is VoI-ranked by declared priority then id; oversized
    # optionals become expansion handles rather than truncating required core.
    ranked_optional = sorted(
        enumerate(request.optional_evidence),
        key=lambda pair: (
            -int(pair[1].get("priority", 0) or 0),
            _optional_evidence_id(pair[1], pair[0]),
        ),
    )

    included_optional: list[Mapping[str, Any]] = []
    omitted_ids: list[str] = []
    handles: list[RepairExpansionHandle] = list(request.expansion_candidates)

    def build_core(
        optionals: Sequence[Mapping[str, Any]],
        handle_list: Sequence[RepairExpansionHandle],
        omitted: Sequence[str],
        *,
        status: RepairPacketStatus,
        reasons: Sequence[str],
    ) -> dict[str, Any]:
        return {
            "schema": REPAIR_PACKET_SCHEMA,
            "version": CONTRACT_REPAIR_PACKET_VERSION,
            "evidence": COMPACT_REPAIR_PACKET_EVIDENCE,
            "task_id": request.task_id,
            "finding_ids": list(request.finding_ids),
            "forest_id": request.forest_id,
            "tree_id": request.tree_id,
            "policy_id": request.policy_id,
            "policy_revision": request.policy_revision,
            "goal_id": request.goal_id,
            "repository_id": request.repository_id,
            "decision_id": decision_id,
            "expected_contract_ref": request.expected_contract_ref,
            "observed_contract_ref": request.observed_contract_ref,
            "call_slice": request.call_slice.to_dict(),
            "counterexample_slice": request.counterexample_slice.to_dict(),
            "edit_scope": list(request.edit_scope),
            "effects": list(request.effects),
            "acceptance": list(request.acceptance),
            "validation_commands": list(request.validation_commands),
            "proof_commands": list(request.proof_commands),
            "risks": list(request.risks),
            "symbols": list(request.symbols),
            "interfaces": list(request.interfaces),
            "related_finding_ids": list(request.related_finding_ids),
            "superseded_finding_ids": list(request.superseded_finding_ids),
            "optional_evidence": [_thaw(item) for item in optionals],
            "omitted_optional_ids": list(omitted),
            "expansion_handles": [item.to_dict() for item in handle_list],
            "authority": request.authority.to_dict(),
            "status": status.value,
            "incomplete_reasons": list(reasons),
            "metadata": _thaw(request.metadata),
            "embeds_full_source": False,
            "embeds_full_ast": False,
            "embeds_full_graph": False,
            "embeds_full_proof": False,
            "embeds_full_witness": False,
            "required_fields_truncated": False,
            "model_output_is_proposal": True,
        }

    # Required core alone must fit max_packet_bytes; otherwise fail closed.
    empty_core = build_core(
        (),
        handles,
        (),
        status=RepairPacketStatus.COMPLETE,
        reasons=(),
    )
    empty_bytes = _measure_core_bytes(empty_core)
    incomplete_reasons: list[str] = []
    status = RepairPacketStatus.COMPLETE
    if empty_bytes > limits.max_packet_bytes:
        raise RepairPacketBudgetError(
            "required repair packet core exceeds max_packet_bytes; "
            "required fields are never truncated"
        )
    if (
        limits.provider_input_budget_bytes
        and empty_bytes > limits.provider_input_budget_bytes
    ):
        raise RepairPacketBudgetError(
            "required repair packet core exceeds provider_input_budget_bytes; "
            "required fields survive provider budgets by failing closed"
        )

    for index, item in ranked_optional:
        evidence_id = _optional_evidence_id(item, index)
        candidate_optionals = included_optional + [item]
        candidate_core = build_core(
            candidate_optionals,
            handles,
            omitted_ids,
            status=status,
            reasons=incomplete_reasons,
        )
        candidate_bytes = _measure_core_bytes(candidate_core)
        over_packet = candidate_bytes > limits.max_packet_bytes
        over_provider = bool(
            limits.provider_input_budget_bytes
            and candidate_bytes > limits.provider_input_budget_bytes
        )
        if over_packet or over_provider:
            handle = _handle_for_optional(
                item,
                index=index,
                tree_id=request.tree_id,
                forest_id=request.forest_id,
                reason=(
                    "deferred_for_packet_budget"
                    if over_packet
                    else "deferred_for_provider_budget"
                ),
            )
            if len(handles) >= limits.max_handle_count:
                incomplete_reasons.append("expansion_handle_limit_exceeded")
                status = RepairPacketStatus.INCOMPLETE
                omitted_ids.append(evidence_id)
                continue
            handles.append(handle)
            omitted_ids.append(evidence_id)
            continue
        included_optional.append(item)

    # Bound source spans separately (plus budget).
    admitted_spans: list[BoundedSourceSpan] = []
    for span in request.source_spans:
        if len(admitted_spans) >= limits.max_span_count:
            handle = RepairExpansionHandle(
                handle_id="",
                kind=ExpansionHandleKind.SOURCE,
                referenced_content_id=span.content_id or span.span_id,
                reference_id=f"span:{span.path}:{span.start_line}",
                reason="deferred_for_span_count",
                tree_id=request.tree_id,
                forest_id=request.forest_id,
                metadata={"path": span.path},
            )
            handles.append(handle)
            continue
        if span.byte_count > limits.max_span_bytes:
            # Keep a path/line handle without the large excerpt.
            trimmed = BoundedSourceSpan(
                path=span.path,
                start_line=span.start_line,
                end_line=span.end_line,
                excerpt="",
                content_id=span.content_id or span.span_id,
                symbol=span.symbol,
            )
            admitted_spans.append(trimmed)
            handles.append(
                RepairExpansionHandle(
                    handle_id="",
                    kind=ExpansionHandleKind.SOURCE,
                    referenced_content_id=trimmed.content_id,
                    reference_id=f"span:{span.path}:{span.start_line}",
                    reason="excerpt_deferred_for_span_bytes",
                    tree_id=request.tree_id,
                    forest_id=request.forest_id,
                )
            )
            continue
        admitted_spans.append(span)

    if len(handles) > limits.max_handle_count:
        handles = handles[: limits.max_handle_count]
        incomplete_reasons.append("expansion_handle_limit_exceeded")
        status = RepairPacketStatus.INCOMPLETE

    if not request.call_slice.complete:
        incomplete_reasons.append("call_slice_incomplete")
        status = RepairPacketStatus.INCOMPLETE

    # Final budget squeeze: never exceed max_packet_bytes.  Prefer deferring
    # remaining optionals (and their handles) over failing after admission.
    def _finalize_core() -> tuple[dict[str, Any], int]:
        payload = build_core(
            included_optional,
            handles,
            omitted_ids,
            status=status,
            reasons=tuple(sorted(set(incomplete_reasons))),
        )
        return payload, _measure_core_bytes(payload)

    core, packet_bytes = _finalize_core()
    while packet_bytes > limits.max_packet_bytes and included_optional:
        item = included_optional.pop()
        index = len(included_optional)
        evidence_id = _optional_evidence_id(item, index)
        omitted_ids.append(evidence_id)
        if len(handles) < limits.max_handle_count:
            handles.append(
                _handle_for_optional(
                    item,
                    index=index,
                    tree_id=request.tree_id,
                    forest_id=request.forest_id,
                    reason="deferred_for_packet_budget",
                )
            )
        core, packet_bytes = _finalize_core()

    # Handles alone can still overflow a tight budget — drop trailing handles
    # while retaining omitted_optional_ids for audit.
    while packet_bytes > limits.max_packet_bytes and handles:
        handles.pop()
        incomplete_reasons.append("expansion_handle_trimmed_for_budget")
        status = RepairPacketStatus.INCOMPLETE
        core, packet_bytes = _finalize_core()

    if (
        limits.provider_input_budget_bytes
        and packet_bytes > limits.provider_input_budget_bytes
        and included_optional
    ):
        while (
            packet_bytes > limits.provider_input_budget_bytes
            and included_optional
        ):
            item = included_optional.pop()
            index = len(included_optional)
            evidence_id = _optional_evidence_id(item, index)
            omitted_ids.append(evidence_id)
            if len(handles) < limits.max_handle_count:
                handles.append(
                    _handle_for_optional(
                        item,
                        index=index,
                        tree_id=request.tree_id,
                        forest_id=request.forest_id,
                        reason="deferred_for_provider_budget",
                    )
                )
            core, packet_bytes = _finalize_core()

    span_bytes = sum(span.byte_count for span in admitted_spans)

    packet = ContractRepairPacket(
        task_id=request.task_id,
        finding_ids=request.finding_ids,
        forest_id=request.forest_id,
        tree_id=request.tree_id,
        policy_id=request.policy_id,
        expected_contract_ref=request.expected_contract_ref,
        observed_contract_ref=request.observed_contract_ref,
        call_slice=request.call_slice,
        edit_scope=request.edit_scope,
        effects=request.effects,
        acceptance=request.acceptance,
        validation_commands=request.validation_commands,
        proof_commands=request.proof_commands,
        risks=request.risks,
        authority=request.authority,
        expansion_handles=tuple(handles),
        counterexample_slice=request.counterexample_slice,
        source_spans=tuple(admitted_spans),
        policy_revision=request.policy_revision,
        goal_id=request.goal_id,
        symbols=request.symbols,
        interfaces=request.interfaces,
        related_finding_ids=request.related_finding_ids,
        superseded_finding_ids=request.superseded_finding_ids,
        optional_evidence=tuple(included_optional),
        omitted_optional_ids=tuple(sorted(set(omitted_ids))),
        repository_id=request.repository_id,
        decision_id=decision_id,
        status=status,
        incomplete_reasons=tuple(sorted(set(incomplete_reasons))),
        packet_byte_count=packet_bytes,
        span_byte_count=span_bytes,
        metadata=request.metadata,
    )

    if not packet.required_core_present:
        raise ContractRepairPacketError(
            "compiled packet is missing required core fields"
        )
    if packet.packet_byte_count > limits.max_packet_bytes:
        raise RepairPacketBudgetError(
            "compiled packet core exceeds max_packet_bytes after admission"
        )

    receipt = RepairPacketReceipt(
        packet_id=packet.packet_id,
        request_id=request.request_id,
        tree_id=request.tree_id,
        forest_id=request.forest_id,
        policy_id=request.policy_id,
        status=packet.status,
        packet_byte_count=packet.packet_byte_count,
        span_byte_count=packet.span_byte_count,
        estimated_tokens=packet.estimated_tokens,
        required_fields=REQUIRED_CORE_FIELDS,
        expansion_handle_ids=tuple(
            item.handle_id for item in packet.expansion_handles
        ),
        omitted_optional_ids=packet.omitted_optional_ids,
        incomplete_reasons=packet.incomplete_reasons,
        decision_id=decision_id,
    )
    return CompiledRepairPacket(
        packet=packet,
        receipt=receipt,
        request_id=request.request_id,
    )


def compile_repair_packet_delta(
    parent: ContractRepairPacket | CompiledRepairPacket | Mapping[str, Any],
    *,
    changed_evidence: Iterable[Any] = (),
    requested_evidence: Iterable[Any] = (),
    expansion_handles: Iterable[Any] = (),
    tree_id: str | None = None,
    forest_id: str | None = None,
    policy_id: str | None = None,
) -> RepairPacketDelta:
    """Compile a delta retry bound to ``parent`` decision/packet identity.

    Transmits only changed and/or requested evidence.  Stale parent bindings
    (tree/forest/policy drift) fail closed as ``INVALIDATED``.
    """

    if isinstance(parent, CompiledRepairPacket):
        parent_packet = parent.packet
    elif isinstance(parent, ContractRepairPacket):
        parent_packet = parent
    elif isinstance(parent, Mapping):
        parent_packet = ContractRepairPacket.from_dict(parent)
    else:
        raise ContractRepairPacketError(
            "parent must be a ContractRepairPacket or mapping"
        )

    parent_decision = parent_packet.decision_id or parent_packet.packet_id
    current_tree = tree_id if tree_id is not None else parent_packet.tree_id
    current_forest = (
        forest_id if forest_id is not None else parent_packet.forest_id
    )
    current_policy = (
        policy_id if policy_id is not None else parent_packet.policy_id
    )

    if (
        current_tree != parent_packet.tree_id
        or current_forest != parent_packet.forest_id
        or current_policy != parent_packet.policy_id
    ):
        return RepairPacketDelta(
            parent_packet_id=parent_packet.packet_id,
            parent_decision_id=parent_decision,
            parent_tree_id=parent_packet.tree_id,
            parent_forest_id=parent_packet.forest_id,
            parent_policy_id=parent_packet.policy_id,
            changed_evidence=(),
            requested_evidence=(),
            expansion_handles=(),
            status=RepairPacketStatus.INVALIDATED,
            incomplete_reasons=("stale_parent_binding",),
            metadata={
                "observed_tree_id": current_tree,
                "observed_forest_id": current_forest,
                "observed_policy_id": current_policy,
            },
        )

    changed: list[DeltaEvidenceItem] = []
    for item in changed_evidence or ():
        if isinstance(item, DeltaEvidenceItem):
            if item.kind is not DeltaEvidenceKind.CHANGED:
                item = DeltaEvidenceItem(
                    evidence_id=item.evidence_id,
                    kind=DeltaEvidenceKind.CHANGED,
                    content_id=item.content_id,
                    summary=item.summary,
                    payload=_thaw(item.payload),
                )
            changed.append(item)
        else:
            raw = dict(item) if isinstance(item, Mapping) else {}
            raw.setdefault("kind", DeltaEvidenceKind.CHANGED.value)
            changed.append(DeltaEvidenceItem.from_dict(raw))

    requested: list[DeltaEvidenceItem] = []
    for item in requested_evidence or ():
        if isinstance(item, DeltaEvidenceItem):
            if item.kind is not DeltaEvidenceKind.REQUESTED:
                item = DeltaEvidenceItem(
                    evidence_id=item.evidence_id,
                    kind=DeltaEvidenceKind.REQUESTED,
                    content_id=item.content_id,
                    summary=item.summary,
                    payload=_thaw(item.payload),
                )
            requested.append(item)
        else:
            raw = dict(item) if isinstance(item, Mapping) else {}
            raw.setdefault("kind", DeltaEvidenceKind.REQUESTED.value)
            requested.append(DeltaEvidenceItem.from_dict(raw))

    handles = [
        RepairExpansionHandle.from_dict(item)
        for item in expansion_handles or ()
    ]

    # Parent-bound handles only; reject stale tree/forest bindings.
    for handle in handles:
        if handle.tree_id and handle.tree_id != parent_packet.tree_id:
            raise RepairPacketIntegrityError(
                "expansion handle tree_id is stale relative to parent packet"
            )
        if handle.forest_id and handle.forest_id != parent_packet.forest_id:
            raise RepairPacketIntegrityError(
                "expansion handle forest_id is stale relative to parent packet"
            )

    if not changed and not requested and not handles:
        raise ContractRepairPacketError(
            "delta retry requires changed evidence, requested evidence, "
            "or expansion handles"
        )

    return RepairPacketDelta(
        parent_packet_id=parent_packet.packet_id,
        parent_decision_id=parent_decision,
        parent_tree_id=parent_packet.tree_id,
        parent_forest_id=parent_packet.forest_id,
        parent_policy_id=parent_packet.policy_id,
        changed_evidence=tuple(changed),
        requested_evidence=tuple(requested),
        expansion_handles=tuple(handles),
        status=RepairPacketStatus.COMPLETE,
    )


def reconstruct_repair_packet(
    parent: ContractRepairPacket | CompiledRepairPacket | Mapping[str, Any],
    delta: RepairPacketDelta | Mapping[str, Any],
) -> ContractRepairPacket:
    """Reconstruct a full packet from ``parent`` + ``delta``.

    Fails closed on stale parent bindings or forged delta parent identity.
    """

    if isinstance(parent, CompiledRepairPacket):
        parent_packet = parent.packet
    elif isinstance(parent, ContractRepairPacket):
        parent_packet = parent
    else:
        parent_packet = ContractRepairPacket.from_dict(parent)

    if not isinstance(delta, RepairPacketDelta):
        delta = RepairPacketDelta.from_dict(delta)

    if delta.status is RepairPacketStatus.INVALIDATED:
        raise RepairPacketIntegrityError(
            "cannot reconstruct from an invalidated delta"
        )
    if delta.parent_packet_id != parent_packet.packet_id:
        raise RepairPacketIntegrityError(
            "delta parent_packet_id does not match parent packet"
        )
    parent_decision = parent_packet.decision_id or parent_packet.packet_id
    if delta.parent_decision_id != parent_decision:
        raise RepairPacketIntegrityError(
            "delta parent_decision_id does not match parent decision"
        )
    if delta.parent_tree_id != parent_packet.tree_id:
        raise RepairPacketIntegrityError("delta parent_tree_id is stale")
    if delta.parent_forest_id != parent_packet.forest_id:
        raise RepairPacketIntegrityError("delta parent_forest_id is stale")
    if delta.parent_policy_id != parent_packet.policy_id:
        raise RepairPacketIntegrityError("delta parent_policy_id is stale")

    # Merge optional evidence: replace by evidence_id when changed/requested.
    optional_by_id: dict[str, Mapping[str, Any]] = {}
    for index, item in enumerate(parent_packet.optional_evidence):
        optional_by_id[_optional_evidence_id(item, index)] = _thaw(item)

    for item in (*delta.changed_evidence, *delta.requested_evidence):
        payload = {
            "evidence_id": item.evidence_id,
            "content_id": item.content_id,
            "summary": item.summary,
            "kind": item.kind.value,
            **_thaw(item.payload),
        }
        optional_by_id[item.evidence_id] = payload

    # Merge handles (parent deferred + delta).
    handles_by_id: dict[str, RepairExpansionHandle] = {
        item.handle_id: item for item in parent_packet.expansion_handles
    }
    for item in delta.expansion_handles:
        handles_by_id[item.handle_id] = item

    reconstructed = ContractRepairPacket(
        task_id=parent_packet.task_id,
        finding_ids=parent_packet.finding_ids,
        forest_id=parent_packet.forest_id,
        tree_id=parent_packet.tree_id,
        policy_id=parent_packet.policy_id,
        expected_contract_ref=parent_packet.expected_contract_ref,
        observed_contract_ref=parent_packet.observed_contract_ref,
        call_slice=parent_packet.call_slice,
        edit_scope=parent_packet.edit_scope,
        effects=parent_packet.effects,
        acceptance=parent_packet.acceptance,
        validation_commands=parent_packet.validation_commands,
        proof_commands=parent_packet.proof_commands,
        risks=parent_packet.risks,
        authority=parent_packet.authority,
        expansion_handles=tuple(
            handles_by_id[key] for key in sorted(handles_by_id)
        ),
        counterexample_slice=parent_packet.counterexample_slice,
        source_spans=parent_packet.source_spans,
        policy_revision=parent_packet.policy_revision,
        goal_id=parent_packet.goal_id,
        symbols=parent_packet.symbols,
        interfaces=parent_packet.interfaces,
        related_finding_ids=parent_packet.related_finding_ids,
        superseded_finding_ids=parent_packet.superseded_finding_ids,
        optional_evidence=tuple(
            optional_by_id[key] for key in sorted(optional_by_id)
        ),
        omitted_optional_ids=parent_packet.omitted_optional_ids,
        repository_id=parent_packet.repository_id,
        decision_id=parent_packet.decision_id,
        status=parent_packet.status,
        incomplete_reasons=parent_packet.incomplete_reasons,
        metadata={
            **_thaw(parent_packet.metadata),
            "reconstructed_from_delta": delta.delta_id,
            "parent_packet_id": parent_packet.packet_id,
        },
    )
    if not reconstructed.required_core_present:
        raise RepairPacketIntegrityError(
            "reconstruction lost required core fields"
        )
    return reconstructed


def expand_repair_handle(
    packet: ContractRepairPacket | CompiledRepairPacket | Mapping[str, Any],
    handle: RepairExpansionHandle | Mapping[str, Any] | str,
    *,
    store: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Resolve one content-addressed expansion handle against ``store``.

    Stale handles (wrong tree/forest or unknown id) fail closed.
    """

    if isinstance(packet, CompiledRepairPacket):
        packet_obj = packet.packet
    elif isinstance(packet, ContractRepairPacket):
        packet_obj = packet
    else:
        packet_obj = ContractRepairPacket.from_dict(packet)

    if isinstance(handle, str):
        match = next(
            (
                item
                for item in packet_obj.expansion_handles
                if item.handle_id == handle
                or item.reference_id == handle
                or item.referenced_content_id == handle
            ),
            None,
        )
        if match is None:
            raise RepairPacketIntegrityError(
                "expansion handle is not bound to the packet"
            )
        handle_obj = match
    else:
        handle_obj = RepairExpansionHandle.from_dict(handle)

    known_ids = {
        item.handle_id for item in packet_obj.expansion_handles
    } | {
        item.referenced_content_id for item in packet_obj.expansion_handles
    }
    if (
        handle_obj.handle_id not in known_ids
        and handle_obj.referenced_content_id not in known_ids
        and handle_obj.reference_id
        not in {item.reference_id for item in packet_obj.expansion_handles}
    ):
        raise RepairPacketIntegrityError(
            "expansion handle is not admitted on the packet"
        )
    if handle_obj.tree_id and handle_obj.tree_id != packet_obj.tree_id:
        raise RepairPacketIntegrityError("expansion handle tree_id is stale")
    if handle_obj.forest_id and handle_obj.forest_id != packet_obj.forest_id:
        raise RepairPacketIntegrityError(
            "expansion handle forest_id is stale"
        )
    if store is None:
        raise ContractRepairPacketError(
            "expansion requires a content-addressed store"
        )
    body = store.get(handle_obj.referenced_content_id)
    if body is None:
        raise RepairPacketIntegrityError(
            "expansion handle target is missing from store"
        )
    # Never return raw forbidden private material; re-sanitize.
    if isinstance(body, Mapping):
        return _freeze(_plain(body, "expansion_body"))
    if isinstance(body, str):
        text, _ = _redact_inline(body)
        return MappingProxyType(
            {
                "content_id": handle_obj.referenced_content_id,
                "text": text,
            }
        )
    return MappingProxyType(
        {
            "content_id": handle_obj.referenced_content_id,
            "value": _plain(body, "expansion_body"),
        }
    )


def repository_context_baseline_tokens(
    *,
    repository_files: Sequence[Mapping[str, Any]] | Sequence[str],
) -> int:
    """Token estimate for a naive full-repository baseline (benchmark aid)."""

    if not repository_files:
        return 1
    if isinstance(repository_files[0], str):
        payload = {"files": list(repository_files)}
    else:
        payload = {"files": [_thaw(item) for item in repository_files]}  # type: ignore[arg-type]
    return estimate_tokens(payload)


def packet_is_cheaper_than_baseline(
    packet: ContractRepairPacket | CompiledRepairPacket,
    *,
    baseline_tokens: int,
    minimum_reduction_ratio: float = 0.0,
) -> bool:
    """Return whether packet estimated tokens beat a repository baseline."""

    if isinstance(packet, CompiledRepairPacket):
        tokens = packet.packet.estimated_tokens
    else:
        tokens = packet.estimated_tokens
    if baseline_tokens <= 0:
        return False
    if tokens >= baseline_tokens:
        return False
    if minimum_reduction_ratio <= 0:
        return True
    reduction = 1.0 - (float(tokens) / float(baseline_tokens))
    return reduction >= minimum_reduction_ratio


class ContractRepairPacketCompiler:
    """Stateful compiler with exact-packet reuse for identical requests."""

    def __init__(self) -> None:
        self._cache: dict[str, CompiledRepairPacket] = {}

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    def compile(
        self, request: RepairPacketRequest | Mapping[str, Any]
    ) -> CompiledRepairPacket:
        if not isinstance(request, RepairPacketRequest):
            request = RepairPacketRequest.from_dict(request)
        key = request.request_id
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        result = compile_repair_packet(request)
        self._cache[key] = result
        return result

    def compile_delta(
        self,
        parent: ContractRepairPacket | CompiledRepairPacket | Mapping[str, Any],
        *,
        changed_evidence: Iterable[Any] = (),
        requested_evidence: Iterable[Any] = (),
        expansion_handles: Iterable[Any] = (),
        tree_id: str | None = None,
        forest_id: str | None = None,
        policy_id: str | None = None,
    ) -> RepairPacketDelta:
        return compile_repair_packet_delta(
            parent,
            changed_evidence=changed_evidence,
            requested_evidence=requested_evidence,
            expansion_handles=expansion_handles,
            tree_id=tree_id,
            forest_id=forest_id,
            policy_id=policy_id,
        )


_DEFAULT_COMPILER = ContractRepairPacketCompiler()


def compile_contract_repair_packet(
    request: RepairPacketRequest | Mapping[str, Any],
    *,
    compiler: ContractRepairPacketCompiler | None = None,
) -> CompiledRepairPacket:
    """Public entry: compile and optionally cache a repair packet."""

    return (compiler or _DEFAULT_COMPILER).compile(request)


__all__ = [
    "BOUNDED_SOURCE_SPAN_SCHEMA",
    "BYTES_PER_TOKEN",
    "CALL_SLICE_REF_SCHEMA",
    "COMPACT_REPAIR_PACKET_EVIDENCE",
    "CONTRACT_REPAIR_PACKET_VERSION",
    "BoundedSourceSpan",
    "CallSliceRef",
    "CallSliceStepRef",
    "CompiledRepairPacket",
    "ContractRepairPacket",
    "ContractRepairPacketCompiler",
    "ContractRepairPacketError",
    "CounterexampleSliceRef",
    "DEFAULT_MAX_CALL_SLICE_STEPS",
    "DEFAULT_MAX_HANDLE_COUNT",
    "DEFAULT_MAX_PACKET_BYTES",
    "DEFAULT_MAX_SPAN_BYTES",
    "DEFAULT_MAX_SPAN_COUNT",
    "DEFAULT_MAX_SPAN_LINES",
    "DELTA_REPAIR_CONTEXT_EVIDENCE",
    "DeltaEvidenceItem",
    "DeltaEvidenceKind",
    "ExpansionHandleKind",
    "OMITTED",
    "REDACTED",
    "REPAIR_EXPANSION_HANDLE_SCHEMA",
    "REPAIR_PACKET_DELTA_SCHEMA",
    "REPAIR_PACKET_RECEIPT_SCHEMA",
    "REPAIR_PACKET_REQUEST_SCHEMA",
    "REPAIR_PACKET_SCHEMA",
    "REQUIRED_CORE_FIELDS",
    "RepairAuthority",
    "RepairExpansionHandle",
    "RepairPacketBudgetError",
    "RepairPacketDelta",
    "RepairPacketIntegrityError",
    "RepairPacketLimits",
    "RepairPacketReceipt",
    "RepairPacketRequest",
    "RepairPacketStatus",
    "compile_contract_repair_packet",
    "compile_repair_packet",
    "compile_repair_packet_delta",
    "estimate_tokens",
    "expand_repair_handle",
    "packet_is_cheaper_than_baseline",
    "reconstruct_repair_packet",
    "repository_context_baseline_tokens",
]
