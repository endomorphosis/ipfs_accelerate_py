"""Shared helpers for external-agent handoff source adapters.

Adapters normalize documented exports into the EAAEF-010 event family.  They
never execute imported tool calls, never trust imported success claims, never
request hidden chain-of-thought, and retain only bounded unknown fields.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Final

from ...proof.formal_verification_contracts import content_identity
from ..contracts import (
    ApprovalEvent,
    ApprovalKind,
    ConversationEvent,
    ConversationRole,
    HandoffBounds,
    HandoffBoundsError,
    HandoffContractError,
    HandoffEvent,
    HandoffNormalizationReport,
    HandoffProvenance,
    HandoffVersionError,
    PatchEvent,
    PatchKind,
    SourceFamily,
    ToolInvocationEvent,
    ToolResultEvent,
    TrustClass,
    canonical_handoff_json_bytes,
    validate_event_sequence,
)


HIDDEN_CHAIN_OF_THOUGHT_KEYS: Final[frozenset[str]] = frozenset(
    {
        "chain_of_thought",
        "cot",
        "hidden_chain_of_thought",
        "hidden_cot",
        "hidden_reasoning",
        "hidden_thoughts",
        "internal_monologue",
        "model_thoughts",
        "private_reasoning",
        "private_thinking",
        "scratchpad",
        "thinking",
        "thinking_blocks",
        "thinking_private",
        "thinking_text",
    }
)
_PRIVATE_FIELD_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "access_token",
        "api_key",
        "authorization",
        "cookie",
        "credential",
        "hidden_witness",
        "password",
        "private_key",
        "private_premise",
        "private_witness",
        "refresh_token",
        "secret",
        "session_token",
        "transcript_body",
        "witness",
    }
)
_TRUNCATED_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "incomplete_export",
        "is_truncated",
        "partial_export",
        "truncated",
    }
)
_PATCH_TOOL_NAMES: Final[frozenset[str]] = frozenset(
    {
        "apply_patch",
        "applypatch",
        "create_file",
        "edit_file",
        "replace",
        "str_replace",
        "write_file",
        "write_files",
    }
)
_PATCH_PAYLOAD_KEYS: Final[frozenset[str]] = frozenset(
    {
        "contents",
        "diff",
        "new_string",
        "patch",
        "replacement",
        "unified_diff",
    }
)
_ROLE_ALIASES: Final[Mapping[str, ConversationRole]] = MappingProxyType(
    {
        "assistant": ConversationRole.ASSISTANT,
        "developer": ConversationRole.SYSTEM,
        "function": ConversationRole.TOOL,
        "gemini": ConversationRole.ASSISTANT,
        "human": ConversationRole.USER,
        "model": ConversationRole.ASSISTANT,
        "system": ConversationRole.SYSTEM,
        "tool": ConversationRole.TOOL,
        "user": ConversationRole.USER,
    }
)
_SHA256_RE: Final[re.Pattern[str]] = re.compile(r"^sha256:[0-9a-f]{64}$")
_CIDV1_RE: Final[re.Pattern[str]] = re.compile(r"^b[a-z2-7]{20,}$")
_FAMILY_ALIASES: Final[Mapping[SourceFamily, frozenset[str]]] = MappingProxyType(
    {
        SourceFamily.GEMINI_CLI: frozenset(
            {"gemini", "gemini-cli", "gemini_cli", "google-gemini", "google_gemini"}
        ),
        SourceFamily.GENERIC_MCP: frozenset(
            {"generic-mcp", "generic_mcp", "json-rpc", "jsonrpc", "mcp", "mcp_jsonrpc"}
        ),
        SourceFamily.GENERIC_JSON: frozenset(
            {"documented_json", "generic-json", "generic_json", "json"}
        ),
        SourceFamily.GENERIC_JSONL: frozenset(
            {"documented_jsonl", "generic-jsonl", "generic_jsonl", "jsonl", "ndjson"}
        ),
    }
)


def normalize_key(value: Any) -> str:
    return str(value).strip().lower().replace("-", "_")


def looks_like_content_ref(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    text = value.strip()
    return bool(_SHA256_RE.fullmatch(text) or _CIDV1_RE.fullmatch(text))


def content_ref_for(value: Any) -> str:
    if looks_like_content_ref(value):
        return str(value).strip()
    if isinstance(value, Mapping):
        return content_identity(dict(value))
    return content_identity(value)


def clip_utf8(text: str, max_bytes: int) -> str:
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    clipped = encoded[:max_bytes]
    while clipped:
        try:
            return clipped.decode("utf-8")
        except UnicodeDecodeError:
            clipped = clipped[:-1]
    return ""


def decode_utf8(raw: bytes) -> str:
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise HandoffContractError("handoff export is not valid UTF-8") from exc


def parse_json_document(text: str) -> Any:
    stripped = text.strip()
    if not stripped:
        raise HandoffContractError("handoff export JSON is malformed or truncated")
    decoder = json.JSONDecoder()
    try:
        value, index = decoder.raw_decode(stripped)
    except json.JSONDecodeError as exc:
        raise HandoffContractError("handoff export JSON is malformed or truncated") from exc
    if stripped[index:].strip():
        raise HandoffContractError("handoff export JSON is malformed or truncated")
    return value


def parse_json_object_line(text: str) -> Mapping[str, Any]:
    value = parse_json_document(text)
    if not isinstance(value, Mapping):
        raise HandoffContractError("JSONL records must be objects")
    if not all(isinstance(key, str) for key in value):
        raise HandoffContractError("JSONL object keys must be strings")
    return value


def load_json_payload(payload: Any) -> tuple[Any, bytes]:
    if isinstance(payload, Mapping):
        if not all(isinstance(key, str) for key in payload):
            raise HandoffContractError("export object keys must be strings")
        raw = canonical_handoff_json_bytes(dict(payload))
        return payload, raw
    if isinstance(payload, (bytes, bytearray, memoryview)):
        raw = bytes(payload)
        text = decode_utf8(raw)
    elif isinstance(payload, str):
        text = payload
        raw = payload.encode("utf-8")
    else:
        raise HandoffContractError(
            "export payload must be JSON text, bytes, or an object"
        )
    return parse_json_document(text), raw


def raw_export_identity(raw_bytes: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw_bytes).hexdigest()


def coerce_bounds(value: Any) -> HandoffBounds:
    if value is None:
        return HandoffBounds()
    if isinstance(value, HandoffBounds):
        return value
    if isinstance(value, Mapping):
        return HandoffBounds.from_dict(value)
    raise HandoffContractError("bounds must be a HandoffBounds object")


def require_export_bound(raw_bytes: bytes, bounds: HandoffBounds) -> None:
    if len(raw_bytes) > bounds.max_serialized_bytes:
        raise HandoffBoundsError("export exceeds max_serialized_bytes")


def mapping_or_none(value: Any, name: str) -> Mapping[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise HandoffContractError(f"{name} must be an object")
    if not all(isinstance(key, str) for key in value):
        raise HandoffContractError(f"{name} object keys must be strings")
    return value


def require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    mapping = mapping_or_none(value, name)
    if mapping is None:
        raise HandoffContractError(f"{name} must be an object")
    return mapping


def require_sequence(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray, memoryview)) or not isinstance(
        value, Sequence
    ):
        raise HandoffContractError(f"{name} must be a sequence")
    return value


def key_forbidden_reason(key: Any) -> str | None:
    normalized = normalize_key(key)
    if normalized in HIDDEN_CHAIN_OF_THOUGHT_KEYS:
        return "hidden_chain_of_thought"
    if any(
        normalized == marker or normalized.endswith("_" + marker) or marker in normalized
        for marker in _PRIVATE_FIELD_MARKERS
    ):
        return "private_material"
    return None


def reject_forbidden_export_keys(value: Any, *, name: str = "export") -> None:
    if isinstance(value, Mapping):
        thought = value.get("thought")
        if thought is True:
            raise HandoffContractError(
                f"{name} must not represent hidden chain-of-thought"
            )
        for raw_key, item in value.items():
            reason = key_forbidden_reason(raw_key)
            if reason == "hidden_chain_of_thought":
                raise HandoffContractError(
                    f"{name} must not represent hidden chain-of-thought"
                )
            if reason == "private_material":
                raise HandoffContractError(f"{name} must not contain private material")
            reject_forbidden_export_keys(item, name=name)
        return
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray, memoryview)
    ):
        for item in value:
            reject_forbidden_export_keys(item, name=name)


def reject_truncated_export(value: Any) -> None:
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            if normalize_key(raw_key) in _TRUNCATED_MARKERS and item is True:
                raise HandoffContractError("truncated handoff export is rejected")
            reject_truncated_export(item)
        return
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray, memoryview)
    ):
        for item in value:
            reject_truncated_export(item)


def reject_unsupported_export_version(version: str) -> None:
    text = version.strip()
    if not text:
        raise HandoffContractError("source_export_version is required")
    if text.endswith("@2") or text.endswith("@0") or text in {"0", "2", "2.0", "v2"}:
        raise HandoffVersionError(
            f"unsupported export version {text!r}; rebuild with @1"
        )


def resolve_export_version(payload: Mapping[str, Any], default: str) -> str:
    raw = (
        payload.get("source_export_version")
        or payload.get("export_version")
        or payload.get("version")
        or default
    )
    if isinstance(raw, bool) or (isinstance(raw, int) and not isinstance(raw, bool)):
        if raw != 1:
            raise HandoffVersionError(
                f"unsupported export version {raw!r}; rebuild with @1"
            )
        return default
    if not isinstance(raw, str):
        raise HandoffContractError("source_export_version must be a string")
    text = raw.strip() or default
    reject_unsupported_export_version(text)
    return text


def reject_family_mismatch(
    payload: Mapping[str, Any], source_family: SourceFamily
) -> None:
    declared = (
        payload.get("source_family")
        or payload.get("export_family")
        or payload.get("family")
    )
    if declared in (None, ""):
        return
    token = normalize_key(declared).replace("_", "-")
    aliases = {
        normalize_key(item).replace("_", "-")
        for item in _FAMILY_ALIASES.get(source_family, ())
    }
    aliases.add(source_family.value.replace("_", "-"))
    if token not in aliases:
        raise HandoffContractError(
            f"export family {declared!r} does not match {source_family.value}"
        )


def residual_fields(
    payload: Mapping[str, Any], known: Iterable[str]
) -> dict[str, Any]:
    known_keys = {normalize_key(item) for item in known}
    residual: dict[str, Any] = {}
    for key, value in payload.items():
        if normalize_key(key) in known_keys:
            continue
        reason = key_forbidden_reason(key)
        if reason == "hidden_chain_of_thought":
            raise HandoffContractError(
                "export must not represent hidden chain-of-thought"
            )
        if reason == "private_material":
            raise HandoffContractError("export must not contain private material")
        residual[str(key)] = value
    return residual


def coerce_role(value: Any) -> ConversationRole:
    if isinstance(value, ConversationRole):
        return value
    if value in (None, ""):
        return ConversationRole.UNKNOWN
    token = normalize_key(value)
    return _ROLE_ALIASES.get(token, ConversationRole.UNKNOWN)


def coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        for key in ("text", "content", "message", "data"):
            if key in value:
                return coerce_text(value[key])
        return ""
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray, memoryview)
    ):
        parts: list[str] = []
        for item in value:
            piece = coerce_text(item)
            if piece:
                parts.append(piece)
        return "".join(parts)
    if isinstance(value, bool) or not isinstance(value, (int,)):
        return ""
    return str(value)


def coerce_arguments(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise HandoffContractError("arguments object keys must be strings")
        return dict(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return {}
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return {"value": value}
        if isinstance(parsed, Mapping):
            return coerce_arguments(parsed)
        return {"value": parsed}
    if isinstance(value, Sequence) and not isinstance(
        value, (bytes, bytearray, memoryview)
    ):
        return {"items": list(value)}
    if isinstance(value, bool) or isinstance(value, int):
        return {"value": value}
    raise HandoffContractError("arguments must be an object")


def extract_paths(value: Any) -> tuple[str, ...]:
    if value in (None, ""):
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Mapping):
        for key in ("paths", "path", "files", "file", "file_path"):
            if key in value:
                return extract_paths(value[key])
        return ()
    if isinstance(value, Sequence) and not isinstance(
        value, (bytes, bytearray, memoryview)
    ):
        paths: list[str] = []
        seen: set[str] = set()
        for item in value:
            text = str(item).strip().replace("\\", "/")
            if not text or text in seen:
                continue
            seen.add(text)
            paths.append(text)
        return tuple(paths)
    return ()


def claimed_success_from(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if not isinstance(value, Mapping):
        return False
    if value.get("isError") is True or value.get("is_error") is True:
        return False
    if value.get("isError") is False or value.get("is_error") is False:
        return True
    for key in ("success", "ok", "claimed_success"):
        item = value.get(key)
        if item is True:
            return True
        if item is False:
            return False
    return False


def patch_tool_name(name: str) -> bool:
    return normalize_key(name) in _PATCH_TOOL_NAMES


def has_patch_payload(arguments: Mapping[str, Any]) -> bool:
    return any(normalize_key(key) in _PATCH_PAYLOAD_KEYS for key in arguments)


def first_present(payload: Mapping[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        if name in payload and payload[name] not in (None, ""):
            return payload[name]
    return None


@dataclass(frozen=True)
class HandoffAdapterResult:
    """Normalized events plus the bounded public normalization report."""

    adapter_id: str
    source_family: SourceFamily
    provenance: HandoffProvenance
    events: tuple[HandoffEvent, ...]
    report: HandoffNormalizationReport
    raw_export_id: str

    @property
    def event_ids(self) -> tuple[str, ...]:
        return tuple(event.event_id for event in self.events)


@dataclass
class HandoffNormalizationEngine:
    """Accumulate imported events without executing or trusting them."""

    adapter_id: str
    source_family: SourceFamily
    source_export_version: str
    raw_bytes: bytes
    bounds: HandoffBounds = field(default_factory=HandoffBounds)
    captured_at_ms: int = 0
    origin_uri: str = ""
    raw_export_id: str = field(init=False, default="")
    _events: list[HandoffEvent] = field(default_factory=list)
    _pending_by_id: dict[str, ToolInvocationEvent] = field(default_factory=dict)
    _pending_by_name: dict[str, list[ToolInvocationEvent]] = field(default_factory=dict)
    _success_claims: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "bounds", coerce_bounds(self.bounds))
        reject_unsupported_export_version(self.source_export_version)
        require_export_bound(self.raw_bytes, self.bounds)
        object.__setattr__(self, "raw_export_id", raw_export_identity(self.raw_bytes))

    @property
    def provenance(self) -> HandoffProvenance:
        return HandoffProvenance(
            source_family=self.source_family,
            source_export_version=self.source_export_version,
            adapter_id=self.adapter_id,
            captured_at_ms=self.captured_at_ms,
            origin_uri=self.origin_uri,
            trust_class=TrustClass.IMPORTED_UNVERIFIED,
            exportable=True,
        )

    def _next_sequence(self) -> int:
        sequence = len(self._events)
        if sequence >= self.bounds.max_events:
            raise HandoffBoundsError("event sequence exceeds max_events")
        return sequence

    def conversation(
        self,
        *,
        role: Any,
        text: str = "",
        reasoning_summary: str = "",
        residual: Mapping[str, Any] | None = None,
        created_at_ms: int | None = None,
    ) -> ConversationEvent | None:
        body = text.strip()
        summary = reasoning_summary.strip()
        if not body and not summary:
            return None
        event = ConversationEvent(
            sequence=self._next_sequence(),
            role=coerce_role(role),
            text=body,
            reasoning_summary=summary,
            residual_fields=dict(residual or {}),
            provenance=self.provenance,
            bounds=self.bounds,
            created_at_ms=self.captured_at_ms if created_at_ms is None else created_at_ms,
        )
        self._events.append(event)
        return event

    def invocation(
        self,
        *,
        tool_name: str,
        arguments: Any = None,
        residual: Mapping[str, Any] | None = None,
        call_id: str = "",
        created_at_ms: int | None = None,
        emit_patch: bool | None = None,
    ) -> ToolInvocationEvent:
        coerced = coerce_arguments(arguments)
        event = ToolInvocationEvent(
            sequence=self._next_sequence(),
            tool_name=tool_name,
            arguments=coerced,
            residual_fields=dict(residual or {}),
            provenance=self.provenance,
            bounds=self.bounds,
            created_at_ms=self.captured_at_ms if created_at_ms is None else created_at_ms,
            executed=False,
        )
        self._events.append(event)
        if call_id:
            self._pending_by_id[str(call_id)] = event
        self._pending_by_name.setdefault(event.tool_name, []).append(event)
        should_patch = (
            patch_tool_name(event.tool_name) and has_patch_payload(coerced)
            if emit_patch is None
            else emit_patch
        )
        if should_patch:
            self.patch(
                patch_body=coerced,
                paths=extract_paths(coerced),
                residual={},
                claimed_applied=False,
                created_at_ms=created_at_ms,
            )
        return event

    def result(
        self,
        *,
        tool_name: str,
        result_value: Any,
        residual: Mapping[str, Any] | None = None,
        call_id: str = "",
        claimed_success: bool | None = None,
        created_at_ms: int | None = None,
    ) -> ToolResultEvent:
        invocation = self._lookup_invocation(tool_name, call_id)
        success = (
            claimed_success_from(result_value)
            if claimed_success is None
            else claimed_success
        )
        if success:
            self._success_claims += 1
        excerpt_source = coerce_text(result_value)
        if not excerpt_source and result_value not in (None, "", {}, []):
            excerpt_source = decode_utf8(canonical_handoff_json_bytes(result_value))
        event = ToolResultEvent(
            sequence=self._next_sequence(),
            tool_name=invocation.tool_name,
            invocation_event_id=invocation.event_id,
            result_content_id=content_ref_for(
                {
                    "tool_name": invocation.tool_name,
                    "result": result_value,
                }
            ),
            result_excerpt=clip_utf8(excerpt_source, self.bounds.max_text_bytes),
            claimed_success=success,
            residual_fields=dict(residual or {}),
            provenance=self.provenance,
            bounds=self.bounds,
            created_at_ms=self.captured_at_ms if created_at_ms is None else created_at_ms,
            trusted_success=False,
        )
        self._events.append(event)
        return event

    def patch(
        self,
        *,
        patch_body: Any,
        paths: Sequence[str] | None = None,
        residual: Mapping[str, Any] | None = None,
        claimed_applied: bool = False,
        patch_kind: PatchKind | None = None,
        created_at_ms: int | None = None,
    ) -> PatchEvent:
        kind = patch_kind
        if kind is None:
            if isinstance(patch_body, Mapping) and any(
                normalize_key(key) in {"diff", "patch", "unified_diff"}
                for key in patch_body
            ):
                kind = PatchKind.UNIFIED_DIFF
            else:
                kind = PatchKind.OVERLAY_REFERENCE
        if claimed_applied:
            self._success_claims += 1
        event = PatchEvent(
            sequence=self._next_sequence(),
            patch_kind=kind,
            patch_content_id=content_ref_for(patch_body),
            paths=extract_paths(paths),
            claimed_applied=claimed_applied,
            residual_fields=dict(residual or {}),
            provenance=self.provenance,
            bounds=self.bounds,
            created_at_ms=self.captured_at_ms if created_at_ms is None else created_at_ms,
            applied=False,
        )
        self._events.append(event)
        return event

    def approval(
        self,
        *,
        decision: Any,
        subject: Any,
        residual: Mapping[str, Any] | None = None,
        created_at_ms: int | None = None,
    ) -> ApprovalEvent:
        event = ApprovalEvent(
            sequence=self._next_sequence(),
            approval_kind=ApprovalKind.IMPORTED_CLAIM,
            decision=decision,
            subject_content_id=content_ref_for(subject),
            residual_fields=dict(residual or {}),
            provenance=self.provenance,
            bounds=self.bounds,
            created_at_ms=self.captured_at_ms if created_at_ms is None else created_at_ms,
            grants_effects=False,
        )
        self._events.append(event)
        return event

    def finish(self) -> HandoffAdapterResult:
        events = tuple(self._events)
        event_ids = validate_event_sequence(events)
        unknown_fields_retained = sum(len(event.residual_fields) for event in events)
        request_id = content_identity(
            {
                "schema": (
                    "ipfs_accelerate_py/agent-supervisor/adapter-normalization-request@1"
                ),
                "adapter_id": self.adapter_id,
                "raw_export_id": self.raw_export_id,
            }
        )
        session_id = content_identity(
            {
                "schema": "ipfs_accelerate_py/agent-supervisor/adapter-session-binding@1",
                "adapter_id": self.adapter_id,
                "raw_export_id": self.raw_export_id,
                "event_content_ids": list(event_ids),
            }
        )
        report = HandoffNormalizationReport(
            request_id=request_id,
            session_id=session_id,
            source_family=self.source_family,
            raw_export_id=self.raw_export_id,
            accepted_event_ids=event_ids,
            rejected_event_count=0,
            truncated=False,
            unknown_fields_retained=unknown_fields_retained,
            hidden_chain_of_thought_rejected=0,
            imported_success_claims_untrusted=self._success_claims,
            imported_invocations_not_executed=True,
            bounds=self.bounds,
            created_at_ms=self.captured_at_ms,
        )
        return HandoffAdapterResult(
            adapter_id=self.adapter_id,
            source_family=self.source_family,
            provenance=self.provenance,
            events=events,
            report=report,
            raw_export_id=self.raw_export_id,
        )

    def _lookup_invocation(
        self, tool_name: str, call_id: str
    ) -> ToolInvocationEvent:
        if call_id:
            event = self._pending_by_id.pop(str(call_id), None)
            if event is not None:
                pending = self._pending_by_name.get(event.tool_name)
                if pending:
                    self._pending_by_name[event.tool_name] = [
                        item for item in pending if item.event_id != event.event_id
                    ]
                return event
        name = tool_name.strip()
        pending = self._pending_by_name.get(name)
        if pending:
            event = pending.pop(0)
            for key, stored in list(self._pending_by_id.items()):
                if stored.event_id == event.event_id:
                    self._pending_by_id.pop(key, None)
            return event
        raise HandoffContractError(
            "tool result is missing a matching imported invocation"
        )


def prepare_export(
    payload: Mapping[str, Any],
    *,
    source_family: SourceFamily,
    default_export_version: str,
) -> str:
    reject_forbidden_export_keys(payload)
    reject_truncated_export(payload)
    reject_family_mismatch(payload, source_family)
    return resolve_export_version(payload, default_export_version)


def start_engine(
    *,
    adapter_id: str,
    source_family: SourceFamily,
    source_export_version: str,
    raw_bytes: bytes,
    bounds: Any = None,
    captured_at_ms: int = 0,
    origin_uri: str = "",
) -> HandoffNormalizationEngine:
    return HandoffNormalizationEngine(
        adapter_id=adapter_id,
        source_family=source_family,
        source_export_version=source_export_version,
        raw_bytes=raw_bytes,
        bounds=coerce_bounds(bounds),
        captured_at_ms=captured_at_ms,
        origin_uri=origin_uri,
    )


def __getattr__(name: str) -> Any:
    if name in {
        "ADAPTER_ID",
        "GEMINI_CLI_ADAPTER_ID",
        "GeminiCliAdapter",
        "normalize_gemini_cli_export",
    }:
        from . import gemini_cli

        aliases = {
            "ADAPTER_ID": gemini_cli.ADAPTER_ID,
            "GEMINI_CLI_ADAPTER_ID": gemini_cli.ADAPTER_ID,
            "GeminiCliAdapter": gemini_cli.GeminiCliAdapter,
            "normalize_gemini_cli_export": gemini_cli.normalize_gemini_cli_export,
        }
        return aliases[name]
    if name in {
        "GENERIC_JSON_ADAPTER_ID",
        "GENERIC_JSONL_ADAPTER_ID",
        "GENERIC_MCP_ADAPTER_ID",
        "GenericJsonAdapter",
        "GenericJsonlAdapter",
        "GenericMcpAdapter",
        "normalize_generic_json_export",
        "normalize_generic_jsonl_export",
        "normalize_generic_mcp_export",
    }:
        from . import generic

        return getattr(generic, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = (
    "HIDDEN_CHAIN_OF_THOUGHT_KEYS",
    "HandoffAdapterResult",
    "HandoffNormalizationEngine",
    "claimed_success_from",
    "clip_utf8",
    "coerce_arguments",
    "coerce_bounds",
    "coerce_role",
    "coerce_text",
    "content_ref_for",
    "extract_paths",
    "first_present",
    "has_patch_payload",
    "load_json_payload",
    "looks_like_content_ref",
    "mapping_or_none",
    "normalize_key",
    "parse_json_document",
    "parse_json_object_line",
    "patch_tool_name",
    "prepare_export",
    "raw_export_identity",
    "reject_family_mismatch",
    "reject_forbidden_export_keys",
    "reject_truncated_export",
    "reject_unsupported_export_version",
    "require_mapping",
    "require_sequence",
    "residual_fields",
    "resolve_export_version",
    "start_engine",
)
