"""Codex export adapter (EAAEF-012).

Normalizes legitimately exportable Codex messages, tool calls/results, patches,
and explicit reasoning summaries.  Hidden chain-of-thought is rejected.
Imported tool calls are never executed.  Imported success claims are never
trusted.  Truncated or ambiguous authority claims fail closed.
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
    HandoffNormalizationReport,
    HandoffProvenance,
    HandoffTrustError,
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
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    content_identity as _cid,
)


ADAPTER_ID: Final[str] = "codex@1"
SUPPORTED_EXPORT_VERSIONS: Final[frozenset[str]] = frozenset(
    {"codex-export-1", "codex-session-1"}
)
_HIDDEN_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "chain_of_thought",
        "hidden_chain_of_thought",
        "private_reasoning",
        "thinking_private",
        "internal_monologue",
    }
)
_AUTHORITY_CLAIM_KEYS: Final[frozenset[str]] = frozenset(
    {
        "accepted",
        "admitted",
        "completed",
        "merge_accepted",
        "self_approved",
        "trusted_success",
        "worker_accepted",
    }
)
_KNOWN_ITEM_KEYS: Final[frozenset[str]] = frozenset(
    {
        "arguments",
        "call_id",
        "claimed_applied",
        "claimed_success",
        "content",
        "diff",
        "excerpt",
        "id",
        "invocation_id",
        "kind",
        "name",
        "output",
        "patch",
        "paths",
        "payload",
        "reasoning_summary",
        "result",
        "role",
        "sequence",
        "text",
        "tool",
        "tool_name",
        "type",
    }
)


class CodexAdapterError(HandoffContractError):
    """Codex export could not be normalized under the admitted contract."""


def _as_mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CodexAdapterError(f"{name} must be an object")
    return value


def _reject_hidden(payload: object, *, name: str) -> int:
    rejected = 0
    stack: list[object] = [payload]
    while stack:
        current = stack.pop()
        if isinstance(current, Mapping):
            for key, value in current.items():
                marker = str(key).strip().lower().replace("-", "_")
                if marker in _HIDDEN_MARKERS:
                    raise HandoffTrustError(
                        f"{name} must not embed hidden chain-of-thought"
                    )
                stack.append(value)
                rejected += 0
        elif isinstance(current, (list, tuple)):
            stack.extend(current)
    return rejected


def _reject_authority_claims(payload: Mapping[str, Any], *, name: str) -> None:
    for key in payload:
        marker = str(key).strip().lower().replace("-", "_")
        if marker in _AUTHORITY_CLAIM_KEYS:
            raise HandoffTrustError(
                f"{name} contains an ambiguous imported authority claim ({key})"
            )


def _parse_jsonl(stripped: str) -> list[Mapping[str, Any]]:
    items: list[Mapping[str, Any]] = []
    for line_no, line in enumerate(stripped.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CodexAdapterError(
                f"Codex JSONL line {line_no} is truncated or malformed"
            ) from exc
        if not isinstance(parsed, Mapping):
            raise CodexAdapterError(f"Codex JSONL line {line_no} must be an object")
        items.append(parsed)
    return items


def _parse_export(raw: bytes | str | Mapping[str, Any]) -> tuple[str, list[Mapping[str, Any]], Mapping[str, Any]]:
    if isinstance(raw, Mapping):
        document = dict(raw)
    else:
        text = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
        stripped = text.strip()
        if not stripped:
            raise CodexAdapterError("Codex export is empty")
        if stripped.endswith("...") or stripped.endswith(",]") or "\ufffd" in stripped:
            raise CodexAdapterError("Codex export is truncated")
        try:
            loaded = json.loads(stripped)
        except json.JSONDecodeError as exc:
            if "Extra data" in str(exc):
                document = {"version": "codex-export-1", "items": _parse_jsonl(stripped)}
            else:
                raise CodexAdapterError(
                    "Codex export JSON is truncated or malformed"
                ) from exc
        else:
            if not isinstance(loaded, Mapping):
                raise CodexAdapterError("Codex JSON export must be an object")
            document = dict(loaded)
    if document.get("truncated") is True or document.get("incomplete") is True:
        raise CodexAdapterError("Codex export is truncated")
    _reject_hidden(document, name="codex export")
    _reject_authority_claims(document, name="codex export")
    version = str(document.get("version") or document.get("export_version") or "").strip()
    if version not in SUPPORTED_EXPORT_VERSIONS:
        raise CodexAdapterError(
            "unsupported or missing Codex export version; "
            f"admitted versions are {sorted(SUPPORTED_EXPORT_VERSIONS)}"
        )
    items_raw = document.get("items")
    if items_raw is None:
        items_raw = document.get("events") or document.get("messages") or ()
    if isinstance(items_raw, Mapping):
        raise CodexAdapterError("Codex export items must be an array")
    items = []
    for item in items_raw:
        items.append(_as_mapping(item, "codex export item"))
    residual_root = {
        key: value
        for key, value in document.items()
        if key
        not in {
            "version",
            "export_version",
            "items",
            "events",
            "messages",
            "truncated",
            "incomplete",
        }
    }
    return version, items, MappingProxyType(residual_root)


def _role(value: object) -> ConversationRole:
    text = str(value or "unknown").strip().lower()
    mapping = {
        "user": ConversationRole.USER,
        "assistant": ConversationRole.ASSISTANT,
        "system": ConversationRole.SYSTEM,
        "tool": ConversationRole.TOOL,
        "developer": ConversationRole.SYSTEM,
    }
    return mapping.get(text, ConversationRole.UNKNOWN)


def _item_type(item: Mapping[str, Any]) -> str:
    return str(item.get("type") or item.get("kind") or "").strip().lower()


def _text_of(item: Mapping[str, Any]) -> str:
    for key in ("text", "content", "message"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value
        if isinstance(value, Mapping):
            inner = value.get("text") or value.get("content")
            if isinstance(inner, str) and inner.strip():
                return inner
    return ""


def _residual(item: Mapping[str, Any], *, extra_known: set[str] | None = None) -> dict[str, Any]:
    known = _KNOWN_ITEM_KEYS.union(extra_known or ())
    residual: dict[str, Any] = {}
    for key, value in item.items():
        if key in known:
            continue
        residual[str(key)] = value
    return residual


def _tool_name(item: Mapping[str, Any]) -> str:
    for key in ("tool_name", "name", "tool"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    payload = item.get("payload")
    if isinstance(payload, Mapping):
        return _tool_name(payload)
    raise CodexAdapterError("tool item is missing a tool name")


def _arguments(item: Mapping[str, Any]) -> Mapping[str, Any]:
    raw = item.get("arguments") or item.get("payload")
    if isinstance(raw, Mapping) and "arguments" in raw and isinstance(raw["arguments"], Mapping):
        raw = raw["arguments"]
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise CodexAdapterError("tool arguments must be an object")
    return dict(raw)


class CodexExportAdapter:
    """Normalize a Codex export into the EAAEF-010 contract family."""

    adapter_id = ADAPTER_ID
    source_family = SourceFamily.CODEX
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
    ) -> tuple[tuple[Any, ...], HandoffNormalizationReport]:
        bounds = bounds or HandoffBounds()
        version, items, root_residual = _parse_export(raw)
        provenance = HandoffProvenance(
            source_family=self.source_family,
            source_export_version=version,
            adapter_id=self.adapter_id,
            captured_at_ms=captured_at_ms,
            trust_class=TrustClass.IMPORTED_EXPORTABLE,
            exportable=True,
        )
        events: list[Any] = []
        rejected = 0
        unknown_retained = 0
        success_untrusted = 0
        hidden_rejected = 0
        invocation_ids: dict[str, str] = {}
        for index, item in enumerate(items):
            _reject_hidden(item, name="codex export item")
            kind = _item_type(item)
            sequence = int(item.get("sequence") or (index + 1))
            residual = _residual(item)
            if root_residual and index == 0:
                residual = {**dict(root_residual), **residual}
            unknown_retained += len(residual)
            try:
                event = self._normalize_item(
                    item,
                    kind=kind,
                    sequence=sequence,
                    residual=residual,
                    provenance=provenance,
                    bounds=bounds,
                    captured_at_ms=captured_at_ms,
                    invocation_ids=invocation_ids,
                )
            except (HandoffContractError, HandoffTrustError):
                rejected += 1
                continue
            if event is None:
                rejected += 1
                continue
            events.append(event)
            if isinstance(event, ToolResultEvent) and event.claimed_success:
                success_untrusted += 1
            if isinstance(event, ToolInvocationEvent):
                call_id = str(item.get("id") or item.get("call_id") or event.event_id)
                invocation_ids[call_id] = event.event_id
        if not events:
            raise CodexAdapterError("Codex export produced no exportable events")
        event_ids = validate_event_sequence(events)
        stream_id = normalized_stream_identity(event_ids)
        raw_id = raw_export_id or _cid(
            {
                "schema": "ipfs_accelerate_py/agent-supervisor/encrypted-export-reference@1",
                "adapter_id": self.adapter_id,
                "version": version,
                "event_count": len(events),
            }
        )
        session = session_id or _cid(
            {
                "schema": "ipfs_accelerate_py/agent-supervisor/external-agent-session@1",
                "source_family": self.source_family.value,
                "normalized_stream_id": stream_id,
            }
        )
        request = request_id or _cid(
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
            hidden_chain_of_thought_rejected=hidden_rejected,
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
        kind: str,
        sequence: int,
        residual: Mapping[str, Any],
        provenance: HandoffProvenance,
        bounds: HandoffBounds,
        captured_at_ms: int,
        invocation_ids: Mapping[str, str],
    ) -> Any | None:
        if kind in {"", "message", "conversation", "event_msg", "user_message", "assistant_message"}:
            text = _text_of(item)
            summary = str(item.get("reasoning_summary") or "").strip()
            role_value: object = item.get("role")
            if kind in {"user_message"}:
                role_value = "user"
            if kind in {"assistant_message"}:
                role_value = "assistant"
            if not text and not summary:
                payload = item.get("payload")
                if isinstance(payload, Mapping):
                    return self._normalize_item(
                        payload,
                        kind=_item_type(payload) or kind,
                        sequence=sequence,
                        residual={**dict(residual), **_residual(payload)},
                        provenance=provenance,
                        bounds=bounds,
                        captured_at_ms=captured_at_ms,
                        invocation_ids=invocation_ids,
                    )
                return None
            return ConversationEvent(
                sequence=sequence,
                role=_role(role_value),
                provenance=provenance,
                text=text,
                reasoning_summary=summary,
                residual_fields=residual,
                bounds=bounds,
                created_at_ms=captured_at_ms,
            )
        if kind in {"function_call", "tool_call", "tool_invocation", "item.function_call"}:
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
            return event
        if kind in {
            "function_call_output",
            "tool_result",
            "tool_output",
            "item.function_call_output",
        }:
            output = item.get("output") or item.get("result") or item.get("content") or ""
            if isinstance(output, Mapping):
                result_id = _cid(dict(output))
                excerpt = str(output.get("excerpt") or output.get("text") or "")[:200]
            else:
                text = str(output)
                result_id = _cid({"text": text})
                excerpt = text[:200]
            call_id = str(item.get("call_id") or item.get("invocation_id") or item.get("id") or "")
            invocation_event_id = invocation_ids.get(call_id) or (
                call_id
                if call_id.startswith("sha256:") or call_id.startswith("b")
                else _cid({"adapter": self.adapter_id, "call_id": call_id or str(sequence)})
            )
            claimed = bool(item.get("claimed_success") or item.get("success") or False)
            event = ToolResultEvent(
                sequence=sequence,
                tool_name=_tool_name(item) if item.get("name") or item.get("tool_name") or item.get("tool") else "unknown",
                invocation_event_id=invocation_event_id,
                result_content_id=result_id,
                provenance=provenance,
                result_excerpt=excerpt,
                claimed_success=claimed,
                residual_fields=residual,
                bounds=bounds,
                created_at_ms=captured_at_ms,
                trusted_success=False,
            )
            if event.trusted_success:
                raise HandoffTrustError("imported success claims are never trusted")
            return event
        if kind in {"patch", "diff", "git_patch", "unified_diff"}:
            diff = str(item.get("diff") or item.get("patch") or item.get("content") or "")
            if not diff.strip():
                raise CodexAdapterError("patch item is empty")
            patch_id = _cid({"kind": "unified_diff", "diff": diff})
            paths = item.get("paths") or ()
            if isinstance(paths, str):
                paths = (paths,)
            return PatchEvent(
                sequence=sequence,
                patch_kind=PatchKind.UNIFIED_DIFF,
                patch_content_id=patch_id,
                provenance=provenance,
                paths=tuple(str(path) for path in paths),
                claimed_applied=bool(item.get("claimed_applied") or False),
                residual_fields=residual,
                bounds=bounds,
                created_at_ms=captured_at_ms,
                applied=False,
            )
        return None


def normalize_codex_export(
    raw: bytes | str | Mapping[str, Any],
    *,
    captured_at_ms: int,
    **kwargs: Any,
) -> tuple[tuple[Any, ...], HandoffNormalizationReport]:
    """Module-level entry point used by tests and later admission."""

    return CodexExportAdapter().normalize(raw, captured_at_ms=captured_at_ms, **kwargs)


# Keep a local alias so adapters that import content_identity from contracts keep working.
content_identity = _cid
