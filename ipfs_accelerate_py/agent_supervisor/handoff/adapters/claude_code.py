"""Claude Code export adapter (EAAEF-013).

Detects supported Claude Code session/export versions, normalizes messages,
tool_use/tool_result blocks, git diffs, and branches, and preserves bounded
unknown residual fields.  Hidden chain-of-thought is rejected.  Imported tool
calls are never executed.  Imported success claims are never trusted.
Unsupported versions and truncated or ambiguous authority claims fail closed.
Public outputs are EAAEF-010 contract objects and content identities, not raw
transcript dumps.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any, Final

from ipfs_accelerate_py.agent_supervisor.handoff.contracts import (
    ConversationEvent,
    ConversationRole,
    HandoffBounds,
    HandoffContractError,
    HandoffEvent,
    HandoffNormalizationReport,
    HandoffProvenance,
    HandoffTrustError,
    HandoffVersionError,
    PatchEvent,
    PatchKind,
    SourceFamily,
    ToolInvocationEvent,
    ToolResultEvent,
    TrustClass,
    content_identity,
    normalized_stream_identity,
    validate_event_sequence,
)


ADAPTER_ID: Final[str] = "claude-code@1"
SUPPORTED_EXPORT_VERSIONS: Final[frozenset[str]] = frozenset(
    {"claude-code-export-1", "claude-code-session-1"}
)
_DEFAULT_EXPORT_VERSION: Final[str] = "claude-code-export-1"
_HIDDEN_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "chain_of_thought",
        "cot",
        "extended_thinking",
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
_AUTHORITY_CLAIM_KEYS: Final[frozenset[str]] = frozenset(
    {
        "accepted",
        "admitted",
        "authoritative",
        "completed",
        "executed",
        "grants_effects",
        "merge_accepted",
        "self_approved",
        "trusted",
        "trusted_success",
        "worker_accepted",
    }
)
_ENVELOPE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "diff",
        "events",
        "export_version",
        "git",
        "git_diff",
        "incomplete",
        "isTruncated",
        "items",
        "messages",
        "patch",
        "schema",
        "session",
        "truncated",
        "unified_diff",
        "version",
    }
)
_KNOWN_ITEM_KEYS: Final[frozenset[str]] = frozenset(
    {
        "arguments",
        "branch",
        "branches",
        "call_id",
        "claimed_applied",
        "claimed_success",
        "content",
        "cwd",
        "diff",
        "excerpt",
        "git",
        "gitBranch",
        "git_branch",
        "id",
        "input",
        "invocation_id",
        "is_error",
        "kind",
        "message",
        "name",
        "output",
        "patch",
        "paths",
        "payload",
        "reasoning_summary",
        "result",
        "role",
        "sequence",
        "sessionId",
        "success",
        "text",
        "timestamp",
        "tool",
        "tool_name",
        "tool_result",
        "tool_use",
        "tool_use_id",
        "type",
        "uuid",
        "version",
    }
)
_SESSION_EVENT_TYPES: Final[frozenset[str]] = frozenset(
    {"assistant", "system", "tool", "user"}
)
_MESSAGE_KINDS: Final[frozenset[str]] = frozenset(
    {
        "",
        "assistant",
        "assistant_message",
        "conversation",
        "event_msg",
        "message",
        "system",
        "user",
        "user_message",
    }
)
_INVOCATION_KINDS: Final[frozenset[str]] = frozenset(
    {"function_call", "tool_call", "tool_invocation", "tool_use"}
)
_RESULT_KINDS: Final[frozenset[str]] = frozenset(
    {"function_call_output", "tool_output", "tool_result"}
)
_PATCH_KINDS: Final[frozenset[str]] = frozenset(
    {"diff", "git_patch", "patch", "unified_diff"}
)


class ClaudeCodeAdapterError(HandoffContractError):
    """Claude Code export could not be normalized under the admitted contract."""


def _as_mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ClaudeCodeAdapterError(f"{name} must be an object")
    return value


def _normalize_key(value: object) -> str:
    return str(value).strip().lower().replace("-", "_")


def _reject_hidden(payload: object, *, name: str) -> None:
    stack: list[object] = [payload]
    while stack:
        current = stack.pop()
        if isinstance(current, Mapping):
            for key, value in current.items():
                if _normalize_key(key) in _HIDDEN_MARKERS:
                    raise HandoffTrustError(
                        f"{name} must not embed hidden chain-of-thought"
                    )
                if isinstance(value, Mapping) and _normalize_key(
                    value.get("type")
                ) in _HIDDEN_MARKERS:
                    raise HandoffTrustError(
                        f"{name} must not embed hidden chain-of-thought"
                    )
                stack.append(value)
        elif isinstance(current, Sequence) and not isinstance(
            current, (str, bytes, bytearray, memoryview)
        ):
            stack.extend(current)


def _authority_keys(payload: Mapping[str, Any]) -> tuple[str, ...]:
    found: list[str] = []
    for key, value in payload.items():
        marker = _normalize_key(key)
        if marker in _AUTHORITY_CLAIM_KEYS and value not in (None, False, "", 0):
            found.append(str(key))
    return tuple(found)


def _reject_authority_claims(payload: Mapping[str, Any], *, name: str) -> None:
    keys = _authority_keys(payload)
    if keys:
        raise HandoffTrustError(
            f"{name} contains an ambiguous imported authority claim ({keys[0]})"
        )


def _looks_truncated(text: str) -> bool:
    stripped = text.strip()
    return (
        stripped.endswith("...")
        or stripped.endswith(",]")
        or stripped.endswith(",}")
        or "\ufffd" in stripped
    )


def _looks_like_session_event(document: Mapping[str, Any]) -> bool:
    kind = str(document.get("type") or document.get("kind") or "").strip().lower()
    return kind in _SESSION_EVENT_TYPES and (
        "message" in document or "content" in document or "role" in document
    )


def _parse_json_document(text: str) -> Mapping[str, Any]:
    stripped = text.strip()
    if not stripped:
        raise ClaudeCodeAdapterError("Claude Code export is empty")
    if _looks_truncated(stripped):
        raise ClaudeCodeAdapterError("Claude Code export is truncated")
    lines = [line for line in stripped.splitlines() if line.strip()]
    if len(lines) > 1:
        try:
            loaded: object = json.loads(stripped)
        except json.JSONDecodeError:
            items: list[Mapping[str, Any]] = []
            for line_no, line in enumerate(lines, start=1):
                try:
                    parsed = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ClaudeCodeAdapterError(
                        f"Claude Code JSONL line {line_no} is truncated or malformed"
                    ) from exc
                if not isinstance(parsed, Mapping):
                    raise ClaudeCodeAdapterError(
                        f"Claude Code JSONL line {line_no} must be an object"
                    )
                items.append(parsed)
            return {
                "version": _DEFAULT_EXPORT_VERSION,
                "messages": items,
            }
        if isinstance(loaded, Mapping):
            return dict(loaded)
        if isinstance(loaded, list):
            return {"version": _DEFAULT_EXPORT_VERSION, "messages": loaded}
        raise ClaudeCodeAdapterError("Claude Code JSON export must be an object")
    try:
        loaded = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ClaudeCodeAdapterError(
            "Claude Code export JSON is truncated or malformed"
        ) from exc
    if isinstance(loaded, Mapping):
        return dict(loaded)
    if isinstance(loaded, list):
        return {"version": _DEFAULT_EXPORT_VERSION, "messages": loaded}
    raise ClaudeCodeAdapterError("Claude Code JSON export must be an object")


def _document_truncated(document: Mapping[str, Any]) -> bool:
    return any(
        document.get(key) is True
        for key in ("truncated", "incomplete", "isTruncated", "is_truncated")
    )


def _resolve_version(document: Mapping[str, Any], *, inferred: bool) -> str:
    version = str(document.get("version") or "").strip()
    export_version = str(document.get("export_version") or "").strip()
    if version and export_version and version != export_version:
        if version in SUPPORTED_EXPORT_VERSIONS and export_version in SUPPORTED_EXPORT_VERSIONS:
            raise HandoffVersionError("ambiguous Claude Code export version")
        raise HandoffVersionError(
            "unsupported or ambiguous Claude Code export version; "
            f"admitted versions are {sorted(SUPPORTED_EXPORT_VERSIONS)}"
        )
    resolved = export_version or version
    if not resolved and inferred:
        return _DEFAULT_EXPORT_VERSION
    if resolved not in SUPPORTED_EXPORT_VERSIONS:
        raise HandoffVersionError(
            "unsupported or missing Claude Code export version; "
            f"admitted versions are {sorted(SUPPORTED_EXPORT_VERSIONS)}"
        )
    return resolved


def _sequence_from(value: object) -> Sequence[Any]:
    if value is None:
        return ()
    if isinstance(value, Mapping) or isinstance(value, (str, bytes, bytearray, memoryview)):
        raise ClaudeCodeAdapterError("Claude Code export messages must be an array")
    if not isinstance(value, Sequence):
        raise ClaudeCodeAdapterError("Claude Code export messages must be an array")
    return value


def _collect_items(
    document: Mapping[str, Any],
) -> tuple[list[Mapping[str, Any]], bool]:
    inferred = False
    raw_items = document.get("messages")
    if raw_items is None:
        raw_items = document.get("items")
    if raw_items is None:
        raw_items = document.get("events")
    session = document.get("session")
    if raw_items is None and isinstance(session, Mapping):
        raw_items = session.get("messages") or session.get("items") or session.get("events")
    if raw_items is None and _looks_like_session_event(document):
        return [document], True
    items: list[Mapping[str, Any]] = []
    for item in _sequence_from(raw_items):
        items.append(_as_mapping(item, "Claude Code export item"))
    if not items and _looks_like_session_event(document):
        return [document], True
    if raw_items is None and not items:
        inferred = True
    return items, inferred


def _add_branch(found: list[str], seen: set[str], value: object) -> None:
    if isinstance(value, str):
        text = value.strip()
        if text and text not in seen:
            seen.add(text)
            found.append(text)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray, memoryview)):
        for item in value:
            _add_branch(found, seen, item)


def _collect_branches(
    document: Mapping[str, Any], items: Sequence[Mapping[str, Any]]
) -> tuple[str, ...]:
    found: list[str] = []
    seen: set[str] = set()
    _add_branch(
        found,
        seen,
        document.get("branch") or document.get("git_branch") or document.get("gitBranch"),
    )
    _add_branch(found, seen, document.get("branches"))
    git = document.get("git")
    if isinstance(git, Mapping):
        _add_branch(found, seen, git.get("branch") or git.get("git_branch") or git.get("gitBranch"))
        _add_branch(found, seen, git.get("branches"))
    session = document.get("session")
    if isinstance(session, Mapping):
        _add_branch(
            found,
            seen,
            session.get("branch") or session.get("git_branch") or session.get("gitBranch"),
        )
        _add_branch(found, seen, session.get("branches"))
        nested_git = session.get("git")
        if isinstance(nested_git, Mapping):
            _add_branch(found, seen, nested_git.get("branch"))
            _add_branch(found, seen, nested_git.get("branches"))
    for item in items:
        _add_branch(
            found,
            seen,
            item.get("gitBranch") or item.get("git_branch") or item.get("branch"),
        )
        item_git = item.get("git")
        if isinstance(item_git, Mapping):
            _add_branch(found, seen, item_git.get("branch"))
            _add_branch(found, seen, item_git.get("branches"))
        message = item.get("message")
        if isinstance(message, Mapping):
            _add_branch(
                found,
                seen,
                message.get("gitBranch") or message.get("git_branch") or message.get("branch"),
            )
    return tuple(found)


def _root_diff(document: Mapping[str, Any]) -> str:
    for key in ("diff", "git_diff", "unified_diff", "patch"):
        value = document.get(key)
        if isinstance(value, str) and value.strip():
            return value
    git = document.get("git")
    if isinstance(git, Mapping):
        for key in ("diff", "patch", "unified_diff"):
            value = git.get(key)
            if isinstance(value, str) and value.strip():
                return value
    session = document.get("session")
    if isinstance(session, Mapping):
        return _root_diff(session)
    return ""


def _parse_export(
    raw: bytes | str | Mapping[str, Any],
) -> tuple[str, list[Mapping[str, Any]], Mapping[str, Any], tuple[str, ...], str]:
    inferred = False
    if isinstance(raw, Mapping):
        document = dict(raw)
    else:
        text = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
        document = dict(_parse_json_document(text))
        if "version" not in document and "export_version" not in document:
            inferred = True
    truncated = _document_truncated(document)
    authority = _authority_keys(document)
    if truncated and authority:
        raise HandoffTrustError(
            "truncated Claude Code export contains an ambiguous imported authority claim"
        )
    if truncated:
        raise ClaudeCodeAdapterError("Claude Code export is truncated")
    _reject_hidden(document, name="claude code export")
    _reject_authority_claims(document, name="claude code export")
    items, items_inferred = _collect_items(document)
    inferred = inferred or items_inferred or (
        str(document.get("version") or document.get("export_version") or "").strip() == ""
        and bool(items)
        and all(_looks_like_session_event(item) or "role" in item for item in items)
        and "messages" not in document
        and "items" not in document
        and "events" not in document
    )
    if items_inferred or (
        _looks_like_session_event(document)
        and document.get("messages") is None
        and document.get("items") is None
        and document.get("events") is None
    ):
        inferred = True
    version = _resolve_version(document, inferred=inferred)
    branches = _collect_branches(document, items)
    diff = _root_diff(document)
    residual_root = {
        key: value
        for key, value in document.items()
        if key not in _ENVELOPE_KEYS
    }
    if branches:
        residual_root["branches"] = list(branches)
        residual_root["git_branch"] = branches[0]
    return version, items, MappingProxyType(residual_root), branches, diff


def _item_type(item: Mapping[str, Any]) -> str:
    return str(item.get("type") or item.get("kind") or "").strip().lower()


def _role(value: object, *, kind: str = "") -> ConversationRole:
    text = str(value or "").strip().lower()
    if not text:
        if kind in {"user", "user_message"}:
            text = "user"
        elif kind in {"assistant", "assistant_message"}:
            text = "assistant"
        elif kind in {"system"}:
            text = "system"
        elif kind in {"tool", "tool_result"}:
            text = "tool"
    mapping = {
        "user": ConversationRole.USER,
        "human": ConversationRole.USER,
        "assistant": ConversationRole.ASSISTANT,
        "model": ConversationRole.ASSISTANT,
        "system": ConversationRole.SYSTEM,
        "tool": ConversationRole.TOOL,
    }
    return mapping.get(text, ConversationRole.UNKNOWN)


def _residual(item: Mapping[str, Any], *, extra_known: set[str] | None = None) -> dict[str, Any]:
    known = _KNOWN_ITEM_KEYS.union(extra_known or ())
    residual: dict[str, Any] = {}
    for key, value in item.items():
        if key in known:
            continue
        residual[str(key)] = value
    return residual


def _unwrap_item(item: Mapping[str, Any]) -> Mapping[str, Any]:
    message = item.get("message")
    if not isinstance(message, Mapping):
        return item
    merged = {key: value for key, value in item.items() if key != "message"}
    for key, value in message.items():
        if key == "type" and merged.get("type") in _SESSION_EVENT_TYPES:
            continue
        merged[key] = value
    return merged


def _flatten_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        inner = value.get("text")
        if isinstance(inner, str) and inner.strip():
            return inner
        inner = value.get("content")
        if isinstance(inner, str) and inner.strip():
            return inner
        return ""
    if isinstance(value, Sequence) and not isinstance(
        value, (bytes, bytearray, memoryview)
    ):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                if item.strip():
                    parts.append(item)
                continue
            if not isinstance(item, Mapping):
                continue
            block_type = str(item.get("type") or "").strip().lower()
            if block_type in {"", "text"}:
                text = _flatten_text(item)
                if text.strip():
                    parts.append(text)
        return "\n".join(parts)
    return ""


def _text_of(item: Mapping[str, Any]) -> str:
    for key in ("text", "content"):
        text = _flatten_text(item.get(key))
        if text.strip():
            return text
    message = item.get("message")
    if isinstance(message, Mapping):
        return _text_of(message)
    return ""


def _content_blocks(item: Mapping[str, Any]) -> tuple[Any, ...]:
    for key in ("content", "message"):
        value = item.get(key)
        if isinstance(value, Mapping):
            nested = value.get("content")
            if isinstance(nested, Sequence) and not isinstance(
                nested, (str, bytes, bytearray, memoryview)
            ):
                return tuple(nested)
            if key == "content":
                return (value,)
        if isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray, memoryview)
        ):
            return tuple(value)
    return ()


def _tool_name(item: Mapping[str, Any]) -> str:
    for key in ("name", "tool_name", "tool"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    payload = item.get("payload")
    if isinstance(payload, Mapping):
        return _tool_name(payload)
    raise ClaudeCodeAdapterError("tool item is missing a tool name")


def _arguments(item: Mapping[str, Any]) -> Mapping[str, Any]:
    raw = item.get("input")
    if raw is None:
        raw = item.get("arguments")
    if raw is None:
        raw = item.get("payload")
    if isinstance(raw, Mapping) and "arguments" in raw and isinstance(
        raw["arguments"], Mapping
    ):
        raw = raw["arguments"]
    if isinstance(raw, Mapping) and "input" in raw and isinstance(raw["input"], Mapping):
        raw = raw["input"]
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ClaudeCodeAdapterError("tool arguments must be an object")
    return dict(raw)


def _claimed_success(item: Mapping[str, Any]) -> bool:
    if "claimed_success" in item:
        return bool(item.get("claimed_success"))
    if "success" in item:
        return bool(item.get("success"))
    if "is_error" in item:
        return not bool(item.get("is_error"))
    return False


def _result_material(item: Mapping[str, Any]) -> tuple[str, str]:
    output = item.get("content")
    if output is None:
        output = item.get("output")
    if output is None:
        output = item.get("result")
    if isinstance(output, Mapping):
        return content_identity(dict(output)), _flatten_text(output)[:200]
    if isinstance(output, Sequence) and not isinstance(
        output, (str, bytes, bytearray, memoryview)
    ):
        return content_identity({"blocks": list(output)}), _flatten_text(output)[:200]
    text = "" if output is None else str(output)
    return content_identity({"text": text}), text[:200]


def _paths_from_diff(diff: str, explicit: object) -> tuple[str, ...]:
    paths: list[str] = []
    seen: set[str] = set()

    def add(path: object) -> None:
        text = str(path).strip().replace("\\", "/")
        if text.startswith("a/") or text.startswith("b/"):
            text = text[2:]
        if text in {"", ".", "/dev/null"}:
            return
        if text not in seen:
            seen.add(text)
            paths.append(text)

    if isinstance(explicit, str) and explicit.strip():
        add(explicit)
    elif isinstance(explicit, Sequence) and not isinstance(
        explicit, (str, bytes, bytearray, memoryview)
    ):
        for item in explicit:
            add(item)
    for line in diff.splitlines():
        if line.startswith("+++ ") or line.startswith("--- "):
            raw = line[4:].strip()
            if "\t" in raw:
                raw = raw.split("\t", 1)[0]
            add(raw)
    return tuple(paths)


class ClaudeCodeExportAdapter:
    """Normalize a Claude Code export into the EAAEF-010 contract family."""

    adapter_id = ADAPTER_ID
    source_family = SourceFamily.CLAUDE_CODE
    supported_versions = SUPPORTED_EXPORT_VERSIONS

    def normalize(
        self,
        raw: bytes | str | Mapping[str, Any],
        *,
        captured_at_ms: int,
        request_id: str | None = None,
        session_id: str | None = None,
        raw_export_id: str | None = None,
        bounds: HandoffBounds | None = None,
    ) -> tuple[tuple[HandoffEvent, ...], HandoffNormalizationReport]:
        bounds = bounds or HandoffBounds()
        version, items, root_residual, branches, diff = _parse_export(raw)
        provenance = HandoffProvenance(
            source_family=self.source_family,
            source_export_version=version,
            adapter_id=self.adapter_id,
            captured_at_ms=captured_at_ms,
            trust_class=TrustClass.IMPORTED_EXPORTABLE,
            exportable=True,
        )
        events: list[HandoffEvent] = []
        rejected = 0
        unknown_retained = 0
        success_untrusted = 0
        invocation_ids: dict[str, str] = {}
        sequence = 0

        def emit(event: HandoffEvent | None) -> None:
            nonlocal unknown_retained, success_untrusted
            if event is None:
                return
            events.append(event)
            unknown_retained += len(event.residual_fields)
            if isinstance(event, ToolResultEvent) and event.claimed_success:
                success_untrusted += 1

        for item in items:
            _reject_hidden(item, name="claude code export item")
            _reject_authority_claims(item, name="claude code export item")
            produced, skipped = self._normalize_item(
                item,
                sequence_start=sequence,
                residual_extra=dict(root_residual) if not events else {},
                provenance=provenance,
                bounds=bounds,
                captured_at_ms=captured_at_ms,
                invocation_ids=invocation_ids,
            )
            if not produced:
                rejected += skipped
                continue
            for event in produced:
                emit(event)
                sequence = event.sequence + 1
            rejected += skipped
        if diff.strip():
            patch_residual: dict[str, Any] = {}
            if branches:
                patch_residual["branches"] = list(branches)
                patch_residual["git_branch"] = branches[0]
            emit(
                self._patch_event(
                    diff=diff,
                    paths=(),
                    sequence=sequence,
                    residual=patch_residual,
                    provenance=provenance,
                    bounds=bounds,
                    captured_at_ms=captured_at_ms,
                    claimed_applied=False,
                )
            )
            sequence += 1
        if not events:
            raise ClaudeCodeAdapterError("Claude Code export produced no exportable events")
        event_ids = validate_event_sequence(events)
        stream_id = normalized_stream_identity(event_ids)
        raw_id = raw_export_id or content_identity(
            {
                "schema": "ipfs_accelerate_py/agent-supervisor/encrypted-export-reference@1",
                "adapter_id": self.adapter_id,
                "version": version,
                "event_count": len(events),
            }
        )
        session = session_id or content_identity(
            {
                "schema": "ipfs_accelerate_py/agent-supervisor/external-agent-session@1",
                "source_family": self.source_family.value,
                "normalized_stream_id": stream_id,
            }
        )
        request = request_id or content_identity(
            {
                "schema": "ipfs_accelerate_py/agent-supervisor/external-agent-handoff-request@1",
                "adapter_id": self.adapter_id,
                "session_id": session,
                "raw_export_id": raw_id,
            }
        )
        report = HandoffNormalizationReport(
            request_id=request,
            session_id=session,
            source_family=self.source_family,
            raw_export_id=raw_id,
            accepted_event_ids=event_ids,
            rejected_event_count=rejected,
            truncated=False,
            unknown_fields_retained=unknown_retained,
            hidden_chain_of_thought_rejected=0,
            imported_success_claims_untrusted=success_untrusted,
            imported_invocations_not_executed=True,
            normalized_stream_id=stream_id,
            bounds=bounds,
            created_at_ms=captured_at_ms,
        )
        return tuple(events), report

    def _normalize_item(
        self,
        item: Mapping[str, Any],
        *,
        sequence_start: int,
        residual_extra: Mapping[str, Any],
        provenance: HandoffProvenance,
        bounds: HandoffBounds,
        captured_at_ms: int,
        invocation_ids: dict[str, str],
    ) -> tuple[list[HandoffEvent], int]:
        unwrapped = _unwrap_item(item)
        kind = _item_type(unwrapped) or _item_type(item)
        residual = {**dict(residual_extra), **_residual(item), **_residual(unwrapped)}
        blocks = _content_blocks(unwrapped)
        produced: list[HandoffEvent] = []
        skipped = 0
        sequence = sequence_start
        if blocks and any(
            isinstance(block, Mapping)
            and str(block.get("type") or "").strip().lower()
            in _INVOCATION_KINDS.union(_RESULT_KINDS)
            for block in blocks
        ):
            text = _flatten_text(blocks)
            summary = str(unwrapped.get("reasoning_summary") or "").strip()
            if text.strip() or summary:
                produced.append(
                    ConversationEvent(
                        sequence=sequence,
                        role=_role(unwrapped.get("role"), kind=kind),
                        provenance=provenance,
                        text=text,
                        reasoning_summary=summary,
                        residual_fields=residual,
                        bounds=bounds,
                        created_at_ms=captured_at_ms,
                    )
                )
                residual = {}
                sequence += 1
            for block in blocks:
                if not isinstance(block, Mapping):
                    skipped += 1
                    continue
                block_kind = str(block.get("type") or "").strip().lower()
                _reject_hidden(block, name="claude code export block")
                _reject_authority_claims(block, name="claude code export block")
                if block_kind in _INVOCATION_KINDS:
                    event = self._invocation_event(
                        block,
                        sequence=sequence,
                        residual=_residual(block),
                        provenance=provenance,
                        bounds=bounds,
                        captured_at_ms=captured_at_ms,
                        invocation_ids=invocation_ids,
                    )
                    produced.append(event)
                    sequence += 1
                    continue
                if block_kind in _RESULT_KINDS:
                    produced.append(
                        self._result_event(
                            block,
                            sequence=sequence,
                            residual=_residual(block),
                            provenance=provenance,
                            bounds=bounds,
                            captured_at_ms=captured_at_ms,
                            invocation_ids=invocation_ids,
                        )
                    )
                    sequence += 1
                    continue
                if block_kind in {"", "text"}:
                    continue
                skipped += 1
            return produced, skipped
        if kind in _MESSAGE_KINDS or unwrapped.get("role") is not None:
            text = _text_of(unwrapped)
            summary = str(unwrapped.get("reasoning_summary") or "").strip()
            if not text and not summary:
                return [], 1
            produced.append(
                ConversationEvent(
                    sequence=sequence,
                    role=_role(unwrapped.get("role"), kind=kind),
                    provenance=provenance,
                    text=text,
                    reasoning_summary=summary,
                    residual_fields=residual,
                    bounds=bounds,
                    created_at_ms=captured_at_ms,
                )
            )
            return produced, 0
        if kind in _INVOCATION_KINDS:
            produced.append(
                self._invocation_event(
                    unwrapped,
                    sequence=sequence,
                    residual=residual,
                    provenance=provenance,
                    bounds=bounds,
                    captured_at_ms=captured_at_ms,
                    invocation_ids=invocation_ids,
                )
            )
            return produced, 0
        if kind in _RESULT_KINDS:
            produced.append(
                self._result_event(
                    unwrapped,
                    sequence=sequence,
                    residual=residual,
                    provenance=provenance,
                    bounds=bounds,
                    captured_at_ms=captured_at_ms,
                    invocation_ids=invocation_ids,
                )
            )
            return produced, 0
        if kind in _PATCH_KINDS:
            diff = str(
                unwrapped.get("diff")
                or unwrapped.get("patch")
                or unwrapped.get("content")
                or ""
            )
            produced.append(
                self._patch_event(
                    diff=diff,
                    paths=unwrapped.get("paths") or (),
                    sequence=sequence,
                    residual=residual,
                    provenance=provenance,
                    bounds=bounds,
                    captured_at_ms=captured_at_ms,
                    claimed_applied=bool(unwrapped.get("claimed_applied") or False),
                )
            )
            return produced, 0
        return [], 1

    def _invocation_event(
        self,
        item: Mapping[str, Any],
        *,
        sequence: int,
        residual: Mapping[str, Any],
        provenance: HandoffProvenance,
        bounds: HandoffBounds,
        captured_at_ms: int,
        invocation_ids: dict[str, str],
    ) -> ToolInvocationEvent:
        event = ToolInvocationEvent(
            sequence=sequence,
            tool_name=_tool_name(item),
            arguments=_arguments(item),
            provenance=provenance,
            residual_fields=residual,
            bounds=bounds,
            created_at_ms=captured_at_ms,
            executed=False,
        )
        if event.executed:
            raise HandoffTrustError("imported tool invocations must not be executed")
        call_id = str(item.get("id") or item.get("call_id") or event.event_id)
        invocation_ids[call_id] = event.event_id
        return event

    def _result_event(
        self,
        item: Mapping[str, Any],
        *,
        sequence: int,
        residual: Mapping[str, Any],
        provenance: HandoffProvenance,
        bounds: HandoffBounds,
        captured_at_ms: int,
        invocation_ids: Mapping[str, str],
    ) -> ToolResultEvent:
        result_id, excerpt = _result_material(item)
        call_id = str(
            item.get("tool_use_id")
            or item.get("call_id")
            or item.get("invocation_id")
            or item.get("id")
            or ""
        )
        invocation_event_id = invocation_ids.get(call_id) or (
            call_id
            if call_id.startswith("sha256:") or call_id.startswith("b")
            else content_identity(
                {"adapter": self.adapter_id, "call_id": call_id or str(sequence)}
            )
        )
        try:
            tool_name = _tool_name(item)
        except ClaudeCodeAdapterError:
            tool_name = "unknown"
        event = ToolResultEvent(
            sequence=sequence,
            tool_name=tool_name,
            invocation_event_id=invocation_event_id,
            result_content_id=result_id,
            provenance=provenance,
            result_excerpt=excerpt,
            claimed_success=_claimed_success(item),
            residual_fields=residual,
            bounds=bounds,
            created_at_ms=captured_at_ms,
            trusted_success=False,
        )
        if event.trusted_success:
            raise HandoffTrustError("imported success claims are never trusted")
        return event

    def _patch_event(
        self,
        *,
        diff: str,
        paths: object,
        sequence: int,
        residual: Mapping[str, Any],
        provenance: HandoffProvenance,
        bounds: HandoffBounds,
        captured_at_ms: int,
        claimed_applied: bool,
    ) -> PatchEvent:
        if not str(diff).strip():
            raise ClaudeCodeAdapterError("patch item is empty")
        patch_id = content_identity({"kind": "unified_diff", "diff": diff})
        event = PatchEvent(
            sequence=sequence,
            patch_kind=PatchKind.UNIFIED_DIFF,
            patch_content_id=patch_id,
            provenance=provenance,
            paths=_paths_from_diff(diff, paths),
            claimed_applied=claimed_applied,
            residual_fields=residual,
            bounds=bounds,
            created_at_ms=captured_at_ms,
            applied=False,
        )
        if event.applied:
            raise HandoffTrustError("imported patches are not marked applied")
        return event


def normalize_claude_code_export(
    raw: bytes | str | Mapping[str, Any],
    *,
    captured_at_ms: int,
    **kwargs: Any,
) -> tuple[tuple[HandoffEvent, ...], HandoffNormalizationReport]:
    """Module-level entry point used by tests and later admission."""

    return ClaudeCodeExportAdapter().normalize(
        raw, captured_at_ms=captured_at_ms, **kwargs
    )
