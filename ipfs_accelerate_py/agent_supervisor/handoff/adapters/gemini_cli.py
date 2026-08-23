"""Gemini CLI export adapter (EAAEF-014).

Normalizes documented Gemini CLI JSON exports that use ``contents``/``parts``
with ``functionCall``/``functionResponse``.  Imported calls are recorded only;
they are never executed and success claims are never trusted.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, Final

from . import (
    HandoffAdapterResult,
    HandoffNormalizationEngine,
    coerce_arguments,
    coerce_text,
    first_present,
    load_json_payload,
    mapping_or_none,
    normalize_key,
    prepare_export,
    require_mapping,
    require_sequence,
    residual_fields,
    start_engine,
)
from ..contracts import HandoffContractError, SourceFamily


ADAPTER_ID: Final[str] = "gemini-cli@1"
SOURCE_EXPORT_VERSION: Final[str] = "gemini-cli-export-1"

_CONTENT_KNOWN: Final[frozenset[str]] = frozenset(
    {
        "content",
        "created_at_ms",
        "origin",
        "parts",
        "role",
        "timestamp_ms",
        "type",
    }
)
_PART_KNOWN: Final[frozenset[str]] = frozenset(
    {
        "codeexecutionresult",
        "code_execution_result",
        "executablecode",
        "executable_code",
        "functioncall",
        "function_call",
        "functionresponse",
        "function_response",
        "inline_data",
        "inlinedata",
        "text",
        "type",
    }
)
_CALL_KNOWN: Final[frozenset[str]] = frozenset(
    {
        "args",
        "arguments",
        "call_id",
        "id",
        "name",
        "parameters",
    }
)
_RESPONSE_KNOWN: Final[frozenset[str]] = frozenset(
    {
        "call_id",
        "id",
        "name",
        "response",
        "result",
        "success",
    }
)


def _contents_from_export(payload: Mapping[str, Any]) -> Sequence[Any]:
    for key in ("contents", "history"):
        value = payload.get(key)
        if value is not None:
            return require_sequence(value, key)
    for wrapper_key in ("session", "chat", "conversation"):
        wrapper = mapping_or_none(payload.get(wrapper_key), wrapper_key)
        if wrapper is None:
            continue
        for key in ("contents", "history", "messages"):
            value = wrapper.get(key)
            if value is not None:
                return require_sequence(value, f"{wrapper_key}.{key}")
    messages = payload.get("messages")
    if messages is not None:
        return require_sequence(messages, "messages")
    candidates = payload.get("candidates")
    if candidates is not None:
        contents: list[Any] = []
        for item in require_sequence(candidates, "candidates"):
            candidate = require_mapping(item, "candidate")
            if "content" in candidate:
                contents.append(candidate["content"])
        if contents:
            return contents
    if payload.get("parts") is not None or payload.get("role") not in (None, ""):
        return (payload,)
    raise HandoffContractError("Gemini CLI export must contain contents or parts")


def _parts_from_content(content: Mapping[str, Any]) -> Sequence[Any]:
    for key in ("parts", "content"):
        value = content.get(key)
        if isinstance(value, str):
            return ({"text": value},)
        if value is not None:
            return require_sequence(value, key)
    text = first_present(content, ("text", "message"))
    if isinstance(text, str) and text.strip():
        return ({"text": text},)
    return ()


def _content_role(content: Mapping[str, Any]) -> Any:
    return first_present(content, ("role", "type", "author")) or "unknown"


def _call_fields(payload: Mapping[str, Any]) -> tuple[str, dict[str, Any], str, dict[str, Any]]:
    name = first_present(payload, ("name", "function_name", "tool_name"))
    if not isinstance(name, str) or not name.strip():
        raise HandoffContractError("functionCall name is required")
    arguments = first_present(payload, ("args", "arguments", "parameters"))
    call_id = first_present(payload, ("id", "call_id")) or ""
    return name.strip(), coerce_arguments(arguments), str(call_id), residual_fields(payload, _CALL_KNOWN)


def _response_fields(payload: Mapping[str, Any]) -> tuple[str, Any, str, dict[str, Any]]:
    name = first_present(payload, ("name", "function_name", "tool_name"))
    if not isinstance(name, str) or not name.strip():
        raise HandoffContractError("functionResponse name is required")
    result_value = first_present(payload, ("response", "result"))
    if result_value is None:
        result_value = {}
    call_id = first_present(payload, ("id", "call_id")) or ""
    return name.strip(), result_value, str(call_id), residual_fields(payload, _RESPONSE_KNOWN)


def _mapping_part(part: Any, name: str) -> Mapping[str, Any]:
    if isinstance(part, str):
        return {"text": part}
    return require_mapping(part, name)


def _feed_function_call(
    engine: HandoffNormalizationEngine, payload: Mapping[str, Any], residual: Mapping[str, Any]
) -> None:
    name, arguments, call_id, extra = _call_fields(payload)
    merged = dict(residual)
    merged.update(extra)
    engine.invocation(
        tool_name=name,
        arguments=arguments,
        residual=merged,
        call_id=call_id,
    )


def _feed_function_response(
    engine: HandoffNormalizationEngine, payload: Mapping[str, Any], residual: Mapping[str, Any]
) -> None:
    name, result_value, call_id, extra = _response_fields(payload)
    merged = dict(residual)
    merged.update(extra)
    engine.result(
        tool_name=name,
        result_value=result_value,
        residual=merged,
        call_id=call_id,
    )


def _feed_executable_code(
    engine: HandoffNormalizationEngine, payload: Mapping[str, Any], residual: Mapping[str, Any]
) -> None:
    arguments = {
        "language": first_present(payload, ("language", "lang")) or "",
        "code": first_present(payload, ("code", "source", "text")) or "",
    }
    engine.invocation(
        tool_name="code_execution",
        arguments=arguments,
        residual=dict(residual),
        call_id=str(first_present(payload, ("id", "call_id")) or ""),
        emit_patch=False,
    )


def _feed_code_execution_result(
    engine: HandoffNormalizationEngine, payload: Mapping[str, Any], residual: Mapping[str, Any]
) -> None:
    engine.result(
        tool_name="code_execution",
        result_value=payload,
        residual=dict(residual),
        call_id=str(first_present(payload, ("id", "call_id")) or ""),
        claimed_success=normalize_key(payload.get("outcome") or "ok")
        not in {"error", "failed", "failure"},
    )


def _classify_part(part: Mapping[str, Any]) -> tuple[str, Mapping[str, Any] | str, dict[str, Any]]:
    part_residual = residual_fields(part, _PART_KNOWN)
    call = first_present(part, ("functionCall", "function_call"))
    if call is not None:
        return "call", require_mapping(call, "functionCall"), part_residual
    response = first_present(part, ("functionResponse", "function_response"))
    if response is not None:
        return "response", require_mapping(response, "functionResponse"), part_residual
    executable = first_present(part, ("executableCode", "executable_code"))
    if executable is not None:
        return "code", require_mapping(executable, "executableCode"), part_residual
    execution_result = first_present(part, ("codeExecutionResult", "code_execution_result"))
    if execution_result is not None:
        return (
            "code_result",
            require_mapping(execution_result, "codeExecutionResult"),
            part_residual,
        )
    return "text", coerce_text(part.get("text")), part_residual


def _feed_content(engine: HandoffNormalizationEngine, raw_content: Any) -> None:
    if isinstance(raw_content, str):
        engine.conversation(role="user", text=raw_content)
        return
    content = require_mapping(raw_content, "content")
    leftover = residual_fields(content, _CONTENT_KNOWN)
    role = _content_role(content)
    pending_text: list[str] = []

    def flush_text() -> None:
        nonlocal leftover
        text = "".join(pending_text)
        pending_text.clear()
        if not text:
            return
        engine.conversation(role=role, text=text, residual=leftover)
        leftover = {}

    for part in _parts_from_content(content):
        kind, payload, part_residual = _classify_part(_mapping_part(part, "part"))
        if kind == "text":
            if payload:
                pending_text.append(str(payload))
            leftover.update(part_residual)
            continue
        flush_text()
        attached = dict(leftover)
        attached.update(part_residual)
        leftover = {}
        if kind == "call":
            _feed_function_call(engine, payload, attached)  # type: ignore[arg-type]
        elif kind == "response":
            _feed_function_response(engine, payload, attached)  # type: ignore[arg-type]
        elif kind == "code":
            _feed_executable_code(engine, payload, attached)  # type: ignore[arg-type]
        else:
            _feed_code_execution_result(engine, payload, attached)  # type: ignore[arg-type]
    flush_text()


class GeminiCliAdapter:
    """Normalize Gemini CLI JSON session exports at adapter major @1."""

    adapter_id: ClassVar[str] = ADAPTER_ID
    source_family: ClassVar[SourceFamily] = SourceFamily.GEMINI_CLI
    source_export_version: ClassVar[str] = SOURCE_EXPORT_VERSION

    def normalize(
        self,
        payload: str | bytes | bytearray | Mapping[str, Any],
        *,
        bounds: Any = None,
        captured_at_ms: int = 0,
        origin_uri: str = "",
    ) -> HandoffAdapterResult:
        parsed, raw_bytes = load_json_payload(payload)
        document = require_mapping(parsed, "Gemini CLI export")
        export_version = prepare_export(
            document,
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
        for content in _contents_from_export(document):
            _feed_content(engine, content)
        return engine.finish()


def normalize_gemini_cli_export(
    payload: str | bytes | bytearray | Mapping[str, Any],
    **kwargs: Any,
) -> HandoffAdapterResult:
    return GeminiCliAdapter().normalize(payload, **kwargs)


__all__ = (
    "ADAPTER_ID",
    "SOURCE_EXPORT_VERSION",
    "GeminiCliAdapter",
    "normalize_gemini_cli_export",
)
