"""Shared CLI run metadata extraction for allocation sessions.

Walks JSON / JSONL from coding CLIs (Goose, Codex, Copilot, Grok, Claude,
Gemini, Mistral, Muse) into a bounded, secret-free dict. Thread-local last
observation is consumed by llm_router allocation recording.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Mapping, Sequence
from typing import Any, Optional

from ipfs_accelerate_py.llm_allocation.duckdb_store import sanitize_session_metadata

_LOCK = threading.Lock()
_LAST: dict[str, dict[str, Any]] = {}
_TLS = threading.local()

_ID_KEYS = (
    "session_id",
    "sessionId",
    "sessionID",
    "thread_id",
    "conversation_id",
    "run_id",
    "runId",
    "request_id",
    "requestId",
    "response_id",
    "responseId",
    "command_id",
    "commandId",
)
_MODEL_KEYS = ("model", "model_id", "modelId", "display_label")
_USAGE_KEYS = (
    "prompt_tokens",
    "input_tokens",
    "completion_tokens",
    "output_tokens",
    "total_tokens",
    "cached_tokens",
    "cache_read_input_tokens",
    "cache_write_input_tokens",
    "total_cost_usd",
    "latency_ms",
    "duration_ms",
    "time_to_first_event_ms",
    "num_turns",
    "stopReason",
    "stop_reason",
)
_CANONICAL = {
    "sessionId": "session_id",
    "sessionID": "session_id",
    "runId": "run_id",
    "requestId": "request_id",
    "responseId": "response_id",
    "commandId": "command_id",
    "modelId": "model_id",
    "model": "model_id",
    "input_tokens": "prompt_tokens",
    "output_tokens": "completion_tokens",
    "stopReason": "stop_reason",
}
_SKIP_KEYS = frozenset(
    {
        "prompt",
        "text",
        "content",
        "message",
        "delta",
        "output",
        "messages",
        "stdout",
        "stderr",
        "raw_response",
        "input",
        "user_prompt",
    }
)


def _canon(key: str) -> str:
    return _CANONICAL.get(key, key)


def _put(meta: dict[str, Any], key: str, value: Any) -> None:
    dest = _canon(str(key))
    if dest in meta and meta[dest] not in (None, "", 0, "0"):
        return
    if value is None or value == "":
        return
    if isinstance(value, bool):
        meta[dest] = value
        return
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        meta[dest] = value
        return
    text = str(value).strip()
    if text:
        meta[dest] = text[:256]


def _walk(obj: Any, meta: dict[str, Any], *, depth: int = 0) -> None:
    if depth > 6 or len(meta) >= 48:
        return
    if isinstance(obj, Mapping):
        usage = obj.get("usage")
        if isinstance(usage, Mapping):
            _walk(usage, meta, depth=depth + 1)
        model_usage = obj.get("modelUsage")
        if isinstance(model_usage, Mapping):
            _walk(model_usage, meta, depth=depth + 1)
        for key, value in obj.items():
            lk = str(key)
            if lk.lower() in _SKIP_KEYS or lk in _SKIP_KEYS:
                continue
            if lk in _ID_KEYS or lk in _MODEL_KEYS or lk in _USAGE_KEYS:
                _put(meta, lk, value)
            elif isinstance(value, (Mapping, Sequence)) and not isinstance(
                value, (str, bytes)
            ):
                _walk(value, meta, depth=depth + 1)
    elif isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
        for item in obj[:32]:
            _walk(item, meta, depth=depth + 1)


def extract_cli_metadata(
    stdout: str = "",
    stderr: str = "",
    *,
    provider: str = "",
    extra: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Extract bounded metadata from CLI stdout/stderr JSON or JSONL."""
    meta: dict[str, Any] = {}
    if provider:
        meta["cli_provider"] = str(provider)
    blobs = [stdout or "", stderr or ""]
    for blob in blobs:
        text = str(blob).strip()
        if not text:
            continue
        if text.startswith("{") or text.startswith("["):
            try:
                parsed = json.loads(text)
            except (TypeError, ValueError, json.JSONDecodeError):
                parsed = None
            if parsed is not None:
                _walk(parsed, meta)
                continue
        for line in text.splitlines():
            line = line.strip()
            if not line.startswith("{") or not line.endswith("}"):
                continue
            try:
                parsed = json.loads(line)
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            _walk(parsed, meta)
    if extra:
        for key, value in extra.items():
            _put(meta, str(key), value)
    return sanitize_session_metadata(meta)


def set_last_cli_observation(provider: str, payload: Mapping[str, Any]) -> None:
    key = str(provider or "").strip().lower().replace("-", "_")
    if not key:
        return
    clean = sanitize_session_metadata(payload)
    with _LOCK:
        _LAST[key] = dict(clean)
    _TLS.provider = key
    _TLS.payload = dict(clean)


def get_last_cli_observation(provider: str = "") -> dict[str, Any]:
    key = str(provider or "").strip().lower().replace("-", "_")
    if not key:
        key = str(getattr(_TLS, "provider", "") or "")
    if not key:
        return {}
    tls_provider = str(getattr(_TLS, "provider", "") or "")
    if tls_provider == key and isinstance(getattr(_TLS, "payload", None), dict):
        return dict(_TLS.payload)
    with _LOCK:
        return dict(_LAST.get(key) or {})


def extract_cli_assistant_text(stdout: str, *, max_chars: int = 8192) -> str:
    """Best-effort assistant text from JSON, JSONL, or plain stdout."""
    raw = str(stdout or "").strip()
    if not raw:
        return ""
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        parsed = None
    if isinstance(parsed, Mapping):
        for key in ("text", "message", "result", "output", "content"):
            value = parsed.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()[:max_chars]
    json_last = ""
    plain_last = ""
    for line in raw.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            if line:
                plain_last = line
            continue
        try:
            event = json.loads(line)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if not isinstance(event, Mapping):
            continue
        etype = str(event.get("type") or event.get("payload_type") or "").lower()
        role = str(event.get("role") or "").lower()
        payload = event.get("payload") if isinstance(event.get("payload"), Mapping) else {}
        if "input.user" in etype:
            continue
        for blob in (event, payload):
            if not isinstance(blob, Mapping):
                continue
            text = blob.get("text") or blob.get("message")
            if isinstance(text, str) and text.strip() and (
                role in {"assistant", "agent"}
                or "output" in etype
                or etype in {"agent_message", "assistant_message", "complete", "result"}
            ):
                json_last = text.strip()
    chosen = json_last or plain_last
    if chosen:
        return chosen[:max_chars]
    return raw[:max_chars] if not raw.startswith("{") else ""


def remember_cli_run(
    provider: str,
    stdout: str = "",
    stderr: str = "",
    *,
    extra: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    meta = extract_cli_metadata(stdout, stderr, provider=provider, extra=extra)
    set_last_cli_observation(provider, meta)
    return meta


__all__ = [
    "extract_cli_assistant_text",
    "extract_cli_metadata",
    "get_last_cli_observation",
    "remember_cli_run",
    "set_last_cli_observation",
]
