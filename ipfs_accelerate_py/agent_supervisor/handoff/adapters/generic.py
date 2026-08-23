"""Generic MCP, JSON and JSONL export adapters (EAAEF-014).

These adapters normalize documented generic MCP JSON-RPC frames and bounded
JSON/JSONL event exports.  Unknown fields are retained in ``residual_fields``
up to the handoff bounds.  Imported tool calls are never executed.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, ClassVar, Final

from . import (
    HandoffAdapterResult,
    HandoffNormalizationEngine,
    coerce_text,
    first_present,
    load_json_payload,
    mapping_or_none,
    normalize_key,
    parse_json_object_line,
    prepare_export,
    reject_forbidden_export_keys,
    reject_truncated_export,
    require_mapping,
    require_sequence,
    residual_fields,
    start_engine,
)
from ..contracts import HandoffContractError, PatchKind, SourceFamily


GENERIC_MCP_ADAPTER_ID: Final[str] = "generic-mcp@1"
GENERIC_JSON_ADAPTER_ID: Final[str] = "generic-json@1"
GENERIC_JSONL_ADAPTER_ID: Final[str] = "generic-jsonl@1"
MCP_EXPORT_VERSION: Final[str] = "mcp-export-1"
JSON_EXPORT_VERSION: Final[str] = "generic-json-export-1"
JSONL_EXPORT_VERSION: Final[str] = "generic-jsonl-export-1"

_CONVERSATION_KINDS: Final[frozenset[str]] = frozenset(
    {
        "assistant",
        "chat",
        "conversation",
        "gemini",
        "human",
        "message",
        "model",
        "system",
        "text",
        "user",
    }
)
_INVOCATION_KINDS: Final[frozenset[str]] = frozenset(
    {"function_call", "tool_call", "tool_invocation", "tool_use"}
)
_RESULT_KINDS: Final[frozenset[str]] = frozenset(
    {"function_response", "tool_response", "tool_result", "tool_result_event"}
)
_PATCH_KINDS: Final[frozenset[str]] = frozenset({"diff", "patch", "unified_diff"})
_APPROVAL_KINDS: Final[frozenset[str]] = frozenset({"approval"})
_MCP_SKIP_METHODS: Final[frozenset[str]] = frozenset(
    {
        "initialize",
        "notifications/cancelled",
        "notifications/initialized",
        "notifications/progress",
        "ping",
        "prompts/list",
        "resources/list",
        "resources/templates/list",
        "tools/list",
    }
)
_EVENT_KNOWN: Final[frozenset[str]] = frozenset(
    {
        "applied",
        "arguments",
        "call_id",
        "claimed_applied",
        "claimed_success",
        "content",
        "created_at_ms",
        "decision",
        "diff",
        "event",
        "executed",
        "id",
        "is_error",
        "kind",
        "message",
        "ok",
        "params",
        "patch",
        "patch_kind",
        "paths",
        "reasoning_summary",
        "result",
        "role",
        "sequence",
        "subject",
        "subject_content_id",
        "success",
        "text",
        "tool_call_id",
        "tool_calls",
        "tool_name",
        "trusted_success",
        "type",
        "unified_diff",
    }
)
_MCP_FRAME_KNOWN: Final[frozenset[str]] = frozenset(
    {"error", "id", "jsonrpc", "method", "params", "result"}
)
_MCP_PARAMS_KNOWN: Final[frozenset[str]] = frozenset(
    {"arguments", "data", "level", "name"}
)
_MESSAGE_KNOWN: Final[frozenset[str]] = frozenset(
    {
        "content",
        "created_at_ms",
        "role",
        "text",
        "tool_call_id",
        "tool_calls",
        "tool_name",
        "type",
    }
)


def _kind_of(record: Mapping[str, Any]) -> str:
    raw = first_present(record, ("kind", "type", "event"))
    return normalize_key(raw) if raw not in (None, "") else ""


def _tool_name_of(record: Mapping[str, Any], *, required: bool = False) -> str:
    name = first_present(record, ("tool_name", "name", "function_name"))
    if isinstance(name, str) and name.strip():
        return name.strip()
    nested = mapping_or_none(record.get("function"), "function")
    if nested is not None:
        inner = first_present(nested, ("name", "tool_name"))
        if isinstance(inner, str) and inner.strip():
            return inner.strip()
    if required:
        raise HandoffContractError("tool_name is required")
    return ""


def _call_id_of(record: Mapping[str, Any]) -> str:
    value = first_present(record, ("call_id", "tool_call_id", "id"))
    return str(value) if value not in (None, "") else ""


def _feed_documented_record(
    engine: HandoffNormalizationEngine, record: Mapping[str, Any]
) -> None:
    reject_forbidden_export_keys(record)
    reject_truncated_export(record)
    kind = _kind_of(record)
    residual = residual_fields(record, _EVENT_KNOWN)
    if kind in _INVOCATION_KINDS or (
        not kind and _tool_name_of(record) and "arguments" in record
    ):
        arguments = first_present(record, ("arguments", "args", "parameters", "input"))
        nested = mapping_or_none(record.get("function"), "function")
        if arguments is None and nested is not None:
            arguments = first_present(nested, ("arguments", "args"))
        engine.invocation(
            tool_name=_tool_name_of(record, required=True),
            arguments=arguments,
            residual=residual,
            call_id=_call_id_of(record),
        )
        return
    if kind in _RESULT_KINDS or (
        not kind and _tool_name_of(record) and "result" in record
    ):
        result_value = first_present(record, ("result", "content", "output", "text"))
        claimed = record.get("claimed_success")
        if claimed is None:
            claimed = record.get("success")
        if claimed is None:
            claimed = False if record.get("is_error") is True else None
        engine.result(
            tool_name=_tool_name_of(record, required=True),
            result_value=result_value if result_value is not None else {},
            residual=residual,
            call_id=_call_id_of(record),
            claimed_success=claimed if isinstance(claimed, bool) else None,
        )
        return
    if kind in _PATCH_KINDS:
        body = first_present(record, ("diff", "patch", "unified_diff", "contents"))
        if body is None:
            body = record
        patch_kind = record.get("patch_kind")
        engine.patch(
            patch_body=body,
            paths=record.get("paths") or record.get("path"),
            residual=residual,
            claimed_applied=record.get("claimed_applied") is True
            or record.get("applied") is True,
            patch_kind=PatchKind(patch_kind) if patch_kind else None,
        )
        return
    if kind in _APPROVAL_KINDS:
        subject = first_present(record, ("subject_content_id", "subject"))
        if subject in (None, ""):
            raise HandoffContractError("approval subject_content_id is required")
        engine.approval(
            decision=first_present(record, ("decision",)) or "defer",
            subject=subject,
            residual=residual,
        )
        return
    if kind in _CONVERSATION_KINDS or kind == "" or "role" in record:
        if record.get("tool_calls"):
            _feed_message_record(engine, record)
            return
        text = coerce_text(
            first_present(record, ("text", "content", "message")) or ""
        )
        engine.conversation(
            role=first_present(record, ("role", "author", "type")) or "unknown",
            text=text,
            reasoning_summary=str(record.get("reasoning_summary") or ""),
            residual=residual,
        )
        return
    raise HandoffContractError(f"unsupported documented export record kind {kind!r}")


def _feed_message_record(
    engine: HandoffNormalizationEngine, record: Mapping[str, Any]
) -> None:
    residual = residual_fields(record, _MESSAGE_KNOWN.union(_EVENT_KNOWN))
    role = first_present(record, ("role", "type", "author")) or "unknown"
    if normalize_key(role) in {"tool", "function"}:
        engine.result(
            tool_name=_tool_name_of(record) or "tool",
            result_value=first_present(record, ("content", "text", "result")) or "",
            residual=residual,
            call_id=_call_id_of(record),
        )
        return
    text = coerce_text(first_present(record, ("text", "content", "message")) or "")
    if text:
        engine.conversation(role=role, text=text, residual=residual)
        residual = {}
    tool_calls = record.get("tool_calls")
    if tool_calls is not None:
        for call in require_sequence(tool_calls, "tool_calls"):
            payload = require_mapping(call, "tool_call")
            function = mapping_or_none(payload.get("function"), "function") or payload
            engine.invocation(
                tool_name=_tool_name_of(function, required=True)
                if _tool_name_of(function)
                else _tool_name_of(payload, required=True),
                arguments=first_present(function, ("arguments", "args")),
                residual=residual_fields(
                    payload, {"arguments", "function", "id", "name", "type"}
                ),
                call_id=str(first_present(payload, ("id", "call_id")) or ""),
            )


def _mcp_frames_from_export(parsed: Any) -> Sequence[Any]:
    if isinstance(parsed, Sequence) and not isinstance(parsed, (str, bytes, bytearray)):
        return parsed
    document = require_mapping(parsed, "generic MCP export")
    for key in ("messages", "frames", "rpc", "calls", "records"):
        value = document.get(key)
        if value is not None:
            return require_sequence(value, key)
    if document.get("jsonrpc") not in (None, "") or document.get("method") not in (
        None,
        "",
    ):
        return (document,)
    raise HandoffContractError("generic MCP export must contain JSON-RPC frames")


def _feed_mcp_frame(
    engine: HandoffNormalizationEngine, frame: Mapping[str, Any]
) -> None:
    reject_forbidden_export_keys(frame)
    reject_truncated_export(frame)
    residual = residual_fields(frame, _MCP_FRAME_KNOWN)
    method = frame.get("method")
    params = mapping_or_none(frame.get("params"), "params") or {}
    param_residual = residual_fields(params, _MCP_PARAMS_KNOWN)
    residual.update(param_residual)
    call_id = "" if frame.get("id") in (None, "") else str(frame.get("id"))
    if isinstance(method, str) and method:
        token = method.strip()
        if token in _MCP_SKIP_METHODS or token.endswith("/list"):
            return
        if token in {"tools/call", "tools/callTool"}:
            name = first_present(params, ("name", "tool_name"))
            if not isinstance(name, str) or not name.strip():
                raise HandoffContractError("MCP tools/call name is required")
            engine.invocation(
                tool_name=name.strip(),
                arguments=first_present(params, ("arguments", "args", "input")),
                residual=residual,
                call_id=call_id,
            )
            return
        if token in {
            "logging/message",
            "notifications/message",
            "notifications/stderr",
            "sampling/createMessage",
        }:
            text = coerce_text(first_present(params, ("data", "message", "text", "content")))
            engine.conversation(
                role="system" if token.startswith("logging") else "assistant",
                text=text,
                residual=residual,
            )
            return
        engine.conversation(
            role="system",
            text=token,
            residual=residual,
        )
        return
    if "result" in frame or "error" in frame:
        tool_name = "tool"
        pending_names = [name for name, items in engine._pending_by_name.items() if items]
        if call_id and call_id in engine._pending_by_id:
            tool_name = engine._pending_by_id[call_id].tool_name
        elif len(pending_names) == 1:
            tool_name = pending_names[0]
        if "error" in frame and frame.get("error") not in (None, "", {}):
            engine.result(
                tool_name=tool_name,
                result_value=frame.get("error"),
                residual=residual,
                call_id=call_id,
                claimed_success=False,
            )
            return
        result_value = frame.get("result")
        engine.result(
            tool_name=tool_name,
            result_value=result_value if result_value is not None else {},
            residual=residual,
            call_id=call_id,
            claimed_success=claimed_success_for_mcp(result_value),
        )
        return
    raise HandoffContractError("unsupported generic MCP JSON-RPC frame")


def claimed_success_for_mcp(result_value: Any) -> bool:
    if isinstance(result_value, Mapping):
        if result_value.get("isError") is True or result_value.get("is_error") is True:
            return False
        if result_value.get("isError") is False or result_value.get("is_error") is False:
            return True
    return False


def _json_records_from_export(parsed: Any) -> Sequence[Any]:
    if isinstance(parsed, Sequence) and not isinstance(parsed, (str, bytes, bytearray)):
        return parsed
    document = require_mapping(parsed, "generic JSON export")
    for key in ("events", "records", "items"):
        value = document.get(key)
        if value is not None:
            return require_sequence(value, key)
    messages = document.get("messages")
    if messages is not None:
        return require_sequence(messages, "messages")
    if _kind_of(document) or "role" in document or "tool_name" in document:
        return (document,)
    raise HandoffContractError("generic JSON export must contain documented events")


def _jsonl_records_and_bytes(payload: Any) -> tuple[list[Mapping[str, Any]], bytes]:
    if isinstance(payload, Mapping):
        raise HandoffContractError(
            "JSONL export must be line-delimited records, not a single object"
        )
    records: list[Mapping[str, Any]] = []
    lines: list[str] = []
    if isinstance(payload, (bytes, bytearray, memoryview)):
        text = bytes(payload).decode("utf-8-sig")
        return _jsonl_records_from_text(text)
    if isinstance(payload, str):
        return _jsonl_records_from_text(payload)
    if isinstance(payload, Iterable):
        for item in payload:
            if isinstance(item, Mapping):
                record = require_mapping(item, "JSONL record")
                encoded = json_line(record)
                records.append(record)
                lines.append(encoded)
                continue
            if isinstance(item, (bytes, bytearray, memoryview)):
                line = bytes(item).decode("utf-8")
            else:
                line = str(item)
            stripped = line.strip()
            if not stripped:
                continue
            record = parse_json_object_line(stripped)
            records.append(record)
            lines.append(stripped)
        raw = ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")
        return records, raw
    raise HandoffContractError("JSONL export must be text, bytes, or a record stream")


def _jsonl_records_from_text(text: str) -> tuple[list[Mapping[str, Any]], bytes]:
    records: list[Mapping[str, Any]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        records.append(parse_json_object_line(stripped))
    return records, text.encode("utf-8")


def json_line(record: Mapping[str, Any]) -> str:
    from ..contracts import canonical_handoff_json_bytes

    return canonical_handoff_json_bytes(dict(record)).decode("utf-8")


class GenericMcpAdapter:
    """Normalize documented generic MCP JSON-RPC exports at adapter major @1."""

    adapter_id: ClassVar[str] = GENERIC_MCP_ADAPTER_ID
    source_family: ClassVar[SourceFamily] = SourceFamily.GENERIC_MCP
    source_export_version: ClassVar[str] = MCP_EXPORT_VERSION

    def normalize(
        self,
        payload: str | bytes | bytearray | Mapping[str, Any] | Sequence[Any],
        *,
        bounds: Any = None,
        captured_at_ms: int = 0,
        origin_uri: str = "",
    ) -> HandoffAdapterResult:
        if isinstance(payload, Sequence) and not isinstance(
            payload, (str, bytes, bytearray, memoryview)
        ):
            parsed = list(payload)
            from ..contracts import canonical_handoff_json_bytes

            raw_bytes = canonical_handoff_json_bytes(parsed)
            envelope: Mapping[str, Any] = {"family": "generic_mcp", "messages": parsed}
        else:
            parsed, raw_bytes = load_json_payload(payload)
            envelope = parsed if isinstance(parsed, Mapping) else {"messages": parsed}
        export_version = prepare_export(
            envelope if isinstance(envelope, Mapping) else {},
            source_family=self.source_family,
            default_export_version=self.source_export_version,
        )
        engine = start_engine(
            adapter_id=self.adapter_id,
            source_family=self.source_family,
            source_export_version=export_version,
            raw_bytes=raw_bytes,
            bounds=bounds,
            captured_at_ms=captured_at_ms,
            origin_uri=origin_uri,
        )
        for frame in _mcp_frames_from_export(parsed):
            _feed_mcp_frame(engine, require_mapping(frame, "MCP frame"))
        return engine.finish()


class GenericJsonAdapter:
    """Normalize documented generic JSON event exports at adapter major @1."""

    adapter_id: ClassVar[str] = GENERIC_JSON_ADAPTER_ID
    source_family: ClassVar[SourceFamily] = SourceFamily.GENERIC_JSON
    source_export_version: ClassVar[str] = JSON_EXPORT_VERSION

    def normalize(
        self,
        payload: str | bytes | bytearray | Mapping[str, Any] | Sequence[Any],
        *,
        bounds: Any = None,
        captured_at_ms: int = 0,
        origin_uri: str = "",
    ) -> HandoffAdapterResult:
        if isinstance(payload, Sequence) and not isinstance(
            payload, (str, bytes, bytearray, memoryview)
        ):
            parsed = list(payload)
            from ..contracts import canonical_handoff_json_bytes

            raw_bytes = canonical_handoff_json_bytes(parsed)
            envelope: Mapping[str, Any] = {"family": "generic_json", "events": parsed}
        else:
            parsed, raw_bytes = load_json_payload(payload)
            envelope = parsed if isinstance(parsed, Mapping) else {"events": parsed}
        export_version = prepare_export(
            envelope if isinstance(envelope, Mapping) else {},
            source_family=self.source_family,
            default_export_version=self.source_export_version,
        )
        engine = start_engine(
            adapter_id=self.adapter_id,
            source_family=self.source_family,
            source_export_version=export_version,
            raw_bytes=raw_bytes,
            bounds=bounds,
            captured_at_ms=captured_at_ms,
            origin_uri=origin_uri,
        )
        for record in _json_records_from_export(parsed):
            _feed_documented_record(engine, require_mapping(record, "JSON event"))
        return engine.finish()


class GenericJsonlAdapter:
    """Normalize documented generic JSONL event streams at adapter major @1."""

    adapter_id: ClassVar[str] = GENERIC_JSONL_ADAPTER_ID
    source_family: ClassVar[SourceFamily] = SourceFamily.GENERIC_JSONL
    source_export_version: ClassVar[str] = JSONL_EXPORT_VERSION

    def normalize(
        self,
        payload: str | bytes | bytearray | Iterable[Any],
        *,
        bounds: Any = None,
        captured_at_ms: int = 0,
        origin_uri: str = "",
    ) -> HandoffAdapterResult:
        records, raw_bytes = _jsonl_records_and_bytes(payload)
        for record in records:
            reject_forbidden_export_keys(record)
            reject_truncated_export(record)
            reject_family_if_present(record)
        engine = start_engine(
            adapter_id=self.adapter_id,
            source_family=self.source_family,
            source_export_version=self.source_export_version,
            raw_bytes=raw_bytes,
            bounds=bounds,
            captured_at_ms=captured_at_ms,
            origin_uri=origin_uri,
        )
        for record in records:
            _feed_documented_record(engine, record)
        return engine.finish()

    def normalize_lines(
        self,
        lines: Iterable[Any],
        **kwargs: Any,
    ) -> HandoffAdapterResult:
        return self.normalize(lines, **kwargs)


def reject_family_if_present(record: Mapping[str, Any]) -> None:
    from . import reject_family_mismatch

    if any(key in record for key in ("family", "source_family", "export_family")):
        reject_family_mismatch(record, SourceFamily.GENERIC_JSONL)


def normalize_generic_mcp_export(
    payload: str | bytes | bytearray | Mapping[str, Any] | Sequence[Any],
    **kwargs: Any,
) -> HandoffAdapterResult:
    return GenericMcpAdapter().normalize(payload, **kwargs)


def normalize_generic_json_export(
    payload: str | bytes | bytearray | Mapping[str, Any] | Sequence[Any],
    **kwargs: Any,
) -> HandoffAdapterResult:
    return GenericJsonAdapter().normalize(payload, **kwargs)


def normalize_generic_jsonl_export(
    payload: str | bytes | bytearray | Iterable[Any],
    **kwargs: Any,
) -> HandoffAdapterResult:
    return GenericJsonlAdapter().normalize(payload, **kwargs)


__all__ = (
    "GENERIC_JSONL_ADAPTER_ID",
    "GENERIC_JSON_ADAPTER_ID",
    "GENERIC_MCP_ADAPTER_ID",
    "JSONL_EXPORT_VERSION",
    "JSON_EXPORT_VERSION",
    "MCP_EXPORT_VERSION",
    "GenericJsonAdapter",
    "GenericJsonlAdapter",
    "GenericMcpAdapter",
    "normalize_generic_json_export",
    "normalize_generic_jsonl_export",
    "normalize_generic_mcp_export",
)
