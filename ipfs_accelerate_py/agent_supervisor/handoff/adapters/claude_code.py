"""Claude Code export adapter (EAAEF-013).

Detect supported Claude Code export versions, preserve conversation/git
branches and bounded residual fields, and reject ambiguous or truncated
authority claims.  Imported history is provenance, never authority: tool
calls are not executed, success claims are not trusted, and hidden
chain-of-thought is not requested or represented.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any, Final

from ...proof.formal_verification_contracts import content_identity
from ..contracts import (
    ApprovalDecision,
    ApprovalEvent,
    ApprovalKind,
    ConversationEvent,
    ConversationRole,
    ExternalAgentSession,
    HandoffBounds,
    HandoffBoundsError,
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
    canonical_handoff_json_bytes,
    normalized_stream_identity,
    validate_event_sequence,
)


ADAPTER_ID: Final[str] = "claude_code@1"
CANONICAL_EXPORT_VERSION: Final[str] = "claude-code-export-1"
SUPPORTED_EXPORT_VERSIONS: Final[frozenset[str]] = frozenset(
    {
        CANONICAL_EXPORT_VERSION,
        "ClaudeCodeExport@1",
        "claude_code@1",
        "claude-code@1",
        "1",
    }
)
SUPPORTED_APP_MAJOR_VERSIONS: Final[frozenset[int]] = frozenset({0, 1})

_VERSION_KEYS: Final[tuple[str, ...]] = (
    "export_version",
    "source_export_version",
    "format_version",
    "schema",
    "interface",
)
_RECORD_LIST_KEYS: Final[tuple[str, ...]] = (
    "records",
    "messages",
    "events",
    "entries",
    "items",
    "transcript_records",
)
_BRANCH_LIST_KEYS: Final[tuple[str, ...]] = ("branches", "conversation_branches")
_ENVELOPE_CONSUMED: Final[frozenset[str]] = frozenset(
    {
        *_VERSION_KEYS,
        *_RECORD_LIST_KEYS,
        *_BRANCH_LIST_KEYS,
        "source",
        "source_family",
        "truncated",
        "partial",
        "git_branch",
        "gitBranch",
        "session_id",
        "sessionId",
        "cwd",
        "origin_uri",
        "adapter_id",
        "captured_at_ms",
        "version",
        "type",
        "kind",
    }
)
_RECORD_CONSUMED: Final[frozenset[str]] = frozenset(
    {
        "type",
        "kind",
        "role",
        "message",
        "content",
        "text",
        "summary",
        "reasoning_summary",
        "timestamp",
        "created_at_ms",
        "version",
        "sessionId",
        "session_id",
        "userType",
        "user_type",
        "tool_use",
        "tool_result",
        "toolUse",
        "toolResult",
        "name",
        "input",
        "arguments",
        "id",
        "tool_use_id",
        "toolUseId",
        "is_error",
        "isError",
        "success",
        "claimed_success",
        "truncated",
        "partial",
        "permissionDecision",
        "permission_decision",
        "approval",
        "decision",
        "subject",
        "subject_content_id",
        "authority_binding_id",
        "grants_effects",
        "patch",
        "diff",
        "unified_diff",
        "file_path",
        "filePath",
        "path",
        "paths",
        "old_string",
        "oldString",
        "new_string",
        "newString",
        "export_version",
        "source_export_version",
        "schema",
        "interface",
        "source_family",
        "source",
        "records",
        "messages",
        "events",
        "branches",
    }
)
_BRANCH_RESIDUAL_KEYS: Final[tuple[tuple[str, str], ...]] = (
    ("uuid", "uuid"),
    ("parentUuid", "parent_uuid"),
    ("parent_uuid", "parent_uuid"),
    ("gitBranch", "git_branch"),
    ("git_branch", "git_branch"),
    ("isSidechain", "is_sidechain"),
    ("is_sidechain", "is_sidechain"),
    ("branch_id", "branch_id"),
    ("branchId", "branch_id"),
)
_HIDDEN_CHAIN_OF_THOUGHT_KEYS: Final[frozenset[str]] = frozenset(
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
        "redacted_thinking",
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
_TRANSCRIPT_BODY_KEYS: Final[frozenset[str]] = frozenset(
    {
        "body",
        "full_transcript",
        "raw_bytes",
        "raw_export",
        "raw_transcript",
        "transcript",
        "transcript_body",
        "transcript_text",
    }
)
_AUTHORITY_KEYS: Final[frozenset[str]] = frozenset(
    {
        "authority",
        "authority_binding_id",
        "authority_claim",
        "approval",
        "approval_decision",
        "permission",
        "permission_decision",
        "permissiondecision",
        "grants_effects",
        "bypass_permissions",
        "dangerously_skip_permissions",
        "authorized",
        "authorization_decision",
    }
)
_AUTHORITY_RECORD_TYPES: Final[frozenset[str]] = frozenset(
    {
        "approval",
        "permission",
        "permission_decision",
        "authority",
        "authority_claim",
    }
)
_DECISION_ALIASES: Final[Mapping[str, ApprovalDecision]] = MappingProxyType(
    {
        "approve": ApprovalDecision.APPROVE,
        "approved": ApprovalDecision.APPROVE,
        "allow": ApprovalDecision.APPROVE,
        "allowed": ApprovalDecision.APPROVE,
        "accept": ApprovalDecision.APPROVE,
        "accepted": ApprovalDecision.APPROVE,
        "grant": ApprovalDecision.APPROVE,
        "granted": ApprovalDecision.APPROVE,
        "reject": ApprovalDecision.REJECT,
        "rejected": ApprovalDecision.REJECT,
        "deny": ApprovalDecision.REJECT,
        "denied": ApprovalDecision.REJECT,
        "block": ApprovalDecision.REJECT,
        "blocked": ApprovalDecision.REJECT,
        "defer": ApprovalDecision.DEFER,
        "deferred": ApprovalDecision.DEFER,
        "ask": ApprovalDecision.DEFER,
        "pending": ApprovalDecision.DEFER,
    }
)
_ROLE_ALIASES: Final[Mapping[str, ConversationRole]] = MappingProxyType(
    {
        "user": ConversationRole.USER,
        "human": ConversationRole.USER,
        "assistant": ConversationRole.ASSISTANT,
        "model": ConversationRole.ASSISTANT,
        "claude": ConversationRole.ASSISTANT,
        "system": ConversationRole.SYSTEM,
        "tool": ConversationRole.TOOL,
    }
)
_PATCH_TOOLS: Final[frozenset[str]] = frozenset(
    {
        "edit",
        "write",
        "notebookedit",
        "apply_patch",
        "applypatch",
        "strreplace",
        "str_replace",
    }
)
_EPOCH: Final[datetime] = datetime(1970, 1, 1, tzinfo=timezone.utc)


@dataclass(frozen=True)
class ClaudeCodeBranch:
    """One preserved Claude Code conversation or git branch."""

    branch_id: str
    parent_uuid: str = ""
    git_branch: str = ""
    record_uuids: tuple[str, ...] = ()
    is_sidechain: bool = False


@dataclass(frozen=True)
class ClaudeCodeNormalizationResult:
    """Normalized Claude Code export.  Public form has no transcript bodies."""

    export_version: str
    provenance: HandoffProvenance
    events: tuple[HandoffEvent, ...]
    branches: tuple[ClaudeCodeBranch, ...]
    session: ExternalAgentSession
    report: HandoffNormalizationReport

    @property
    def adapter_id(self) -> str:
        return ADAPTER_ID

    @property
    def source_family(self) -> SourceFamily:
        return SourceFamily.CLAUDE_CODE


class ClaudeCodeExportAdapter:
    """Stateless adapter for supported Claude Code export versions."""

    adapter_id: str = ADAPTER_ID
    source_family: SourceFamily = SourceFamily.CLAUDE_CODE
    canonical_export_version: str = CANONICAL_EXPORT_VERSION

    def detect_version(self, payload: Any) -> str:
        return detect_claude_code_export_version(payload)

    def normalize(
        self,
        payload: Any,
        **kwargs: Any,
    ) -> ClaudeCodeNormalizationResult:
        return normalize_claude_code_export(payload, **kwargs)


def detect_claude_code_export_version(payload: Any) -> str:
    """Return the canonical supported export version, or raise."""

    envelope, records, _truncated = _parse_export(payload)
    return _detect_version(envelope, records)


def normalize_claude_code_export(
    payload: Any,
    *,
    bounds: HandoffBounds | None = None,
    raw_export_id: str = "",
    request_id: str = "",
    captured_at_ms: int = 0,
    origin_uri: str = "",
    objective_id: str = "",
    context_id: str = "",
    repository_id: str = "",
) -> ClaudeCodeNormalizationResult:
    """Normalize one supported Claude Code export into handoff contracts."""

    limits = bounds if bounds is not None else HandoffBounds()
    envelope, records, parse_truncated = _parse_export(payload)
    export_version = _detect_version(envelope, records)
    envelope_truncated = bool(
        parse_truncated
        or (isinstance(envelope, Mapping) and envelope.get("truncated") is True)
        or (isinstance(envelope, Mapping) and envelope.get("partial") is True)
    )
    source = envelope.get("source_family", envelope.get("source")) if envelope else None
    if source not in (None, "", SourceFamily.CLAUDE_CODE, SourceFamily.CLAUDE_CODE.value):
        raise HandoffContractError("Claude Code export source_family must be claude_code")

    indexed = _index_records(records, envelope)
    _reject_authority_claims(indexed, truncated_export=envelope_truncated)

    git_branch = ""
    if envelope is not None:
        git_branch = _optional_text(envelope.get("git_branch", envelope.get("gitBranch")))
    branches = _collect_branches(indexed, envelope_git_branch=git_branch)
    envelope_residual = _envelope_residual(envelope)

    provenance = HandoffProvenance(
        source_family=SourceFamily.CLAUDE_CODE,
        source_export_version=export_version,
        adapter_id=ADAPTER_ID,
        captured_at_ms=_nonnegative_int(captured_at_ms, "captured_at_ms"),
        origin_uri=_optional_text(origin_uri),
        trust_class=TrustClass.IMPORTED_EXPORTABLE,
        exportable=True,
    )
    events, stats = _normalize_records(
        indexed,
        bounds=limits,
        provenance=provenance,
        envelope_residual=envelope_residual,
        branches=branches,
    )
    event_ids = validate_event_sequence(events)
    if not raw_export_id:
        raw_export_id = content_identity(
            {
                "adapter": ADAPTER_ID,
                "export_version": export_version,
                "payload": json.loads(canonical_handoff_json_bytes(_public_payload(payload)).decode("utf-8"))
                if isinstance(payload, Mapping)
                else {"record_count": len(records)},
            }
        )
    session = ExternalAgentSession(
        source_family=SourceFamily.CLAUDE_CODE,
        raw_export_id=raw_export_id,
        event_content_ids=event_ids,
        provenance=provenance,
        objective_id=objective_id,
        context_id=context_id,
        repository_id=repository_id,
        patch_ids=tuple(
            event.patch_content_id for event in events if isinstance(event, PatchEvent)
        ),
        bounds=limits,
        created_at_ms=provenance.captured_at_ms,
    )
    if not request_id:
        request_id = content_identity(
            {
                "adapter": ADAPTER_ID,
                "raw_export_id": raw_export_id,
                "normalized_stream_id": session.normalized_stream_id,
            }
        )
    report = HandoffNormalizationReport(
        request_id=request_id,
        session_id=session.session_id,
        source_family=SourceFamily.CLAUDE_CODE,
        raw_export_id=raw_export_id,
        accepted_event_ids=event_ids,
        rejected_event_count=stats["rejected_event_count"],
        truncated=bool(envelope_truncated or stats["truncated"]),
        unknown_fields_retained=stats["unknown_fields_retained"],
        hidden_chain_of_thought_rejected=stats["hidden_chain_of_thought_rejected"],
        imported_success_claims_untrusted=stats["imported_success_claims_untrusted"],
        imported_invocations_not_executed=True,
        normalized_stream_id=normalized_stream_identity(event_ids),
        bounds=limits,
        created_at_ms=provenance.captured_at_ms,
    )
    return ClaudeCodeNormalizationResult(
        export_version=export_version,
        provenance=provenance,
        events=events,
        branches=branches,
        session=session,
        report=report,
    )


def _parse_export(payload: Any) -> tuple[Mapping[str, Any] | None, list[dict[str, Any]], bool]:
    if isinstance(payload, (bytes, bytearray, memoryview)):
        try:
            text = bytes(payload).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise HandoffContractError("Claude Code export is not valid UTF-8") from exc
        return _parse_text(text)
    if isinstance(payload, str):
        return _parse_text(payload)
    if isinstance(payload, Mapping):
        return _split_envelope(dict(payload), truncated=False)
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray, memoryview)):
        return None, [_require_object(item, "Claude Code record") for item in payload], False
    raise HandoffContractError("Claude Code export must be JSON, JSONL, an object, or a record list")


def _parse_text(text: str) -> tuple[Mapping[str, Any] | None, list[dict[str, Any]], bool]:
    if "\x00" in text:
        raise HandoffContractError("Claude Code export must not contain NUL")
    stripped = text.strip()
    if not stripped:
        raise HandoffContractError("Claude Code export is empty")
    try:
        decoded = json.loads(stripped)
    except json.JSONDecodeError:
        return _parse_jsonl(text)
    if isinstance(decoded, Mapping):
        return _split_envelope(dict(decoded), truncated=False)
    if isinstance(decoded, list):
        return None, [_require_object(item, "Claude Code record") for item in decoded], False
    raise HandoffContractError("Claude Code export JSON must be an object or array")


def _parse_jsonl(text: str) -> tuple[Mapping[str, Any] | None, list[dict[str, Any]], bool]:
    records: list[dict[str, Any]] = []
    truncated = False
    lines = text.splitlines()
    for index, raw_line in enumerate(lines):
        line = raw_line.strip()
        if not line:
            continue
        try:
            decoded = json.loads(line)
        except json.JSONDecodeError as exc:
            if index == len(lines) - 1:
                truncated = True
                continue
            raise HandoffContractError("Claude Code JSONL record is malformed") from exc
        records.append(_require_object(decoded, "Claude Code JSONL record"))
    if not records:
        raise HandoffContractError("Claude Code JSONL export contains no records")
    if len(records) == 1 and any(key in records[0] for key in _RECORD_LIST_KEYS + _BRANCH_LIST_KEYS + _VERSION_KEYS):
        return _split_envelope(records[0], truncated=truncated)
    return None, records, truncated


def _split_envelope(
    payload: Mapping[str, Any], truncated: bool
) -> tuple[Mapping[str, Any] | None, list[dict[str, Any]], bool]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for key in _RECORD_LIST_KEYS:
        value = payload.get(key)
        if value is None:
            continue
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray, memoryview)):
            raise HandoffContractError(f"{key} must be a sequence of records")
        for item in value:
            record = _require_object(item, "Claude Code record")
            marker = _record_uuid(record) or f"index:{len(records)}"
            if marker in seen:
                continue
            seen.add(marker)
            records.append(record)
    for key in _BRANCH_LIST_KEYS:
        value = payload.get(key)
        if value is None:
            continue
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray, memoryview)):
            raise HandoffContractError(f"{key} must be a sequence of branches")
        for branch in value:
            mapping = _require_object(branch, "Claude Code branch")
            nested = None
            for nested_key in _RECORD_LIST_KEYS:
                if mapping.get(nested_key) is not None:
                    nested = mapping.get(nested_key)
                    break
            if nested is None:
                continue
            if not isinstance(nested, Sequence) or isinstance(nested, (str, bytes, bytearray, memoryview)):
                raise HandoffContractError("branch records must be a sequence")
            branch_id = _optional_text(
                mapping.get("branch_id", mapping.get("id", mapping.get("branchId")))
            )
            git_branch = _optional_text(
                mapping.get("git_branch", mapping.get("gitBranch"))
            )
            parent_uuid = _optional_text(
                mapping.get("parent_uuid", mapping.get("parentUuid"))
            )
            is_sidechain = mapping.get("is_sidechain", mapping.get("isSidechain", False))
            for item in nested:
                record = dict(_require_object(item, "Claude Code branch record"))
                if branch_id and "branch_id" not in record and "branchId" not in record:
                    record["branch_id"] = branch_id
                if git_branch and "gitBranch" not in record and "git_branch" not in record:
                    record["git_branch"] = git_branch
                if parent_uuid and "parentUuid" not in record and "parent_uuid" not in record:
                    record.setdefault("parentUuid", parent_uuid)
                if is_sidechain and "isSidechain" not in record and "is_sidechain" not in record:
                    record["is_sidechain"] = True
                marker = _record_uuid(record) or f"index:{len(records)}"
                if marker in seen:
                    continue
                seen.add(marker)
                records.append(record)
    if not records:
        if _looks_like_record(payload):
            return None, [dict(payload)], truncated
        raise HandoffContractError("Claude Code export contains no records")
    return dict(payload), records, truncated or bool(payload.get("truncated") is True)


def _looks_like_record(payload: Mapping[str, Any]) -> bool:
    return any(
        key in payload
        for key in ("type", "kind", "message", "uuid", "parentUuid", "content", "role")
    )


def _detect_version(
    envelope: Mapping[str, Any] | None, records: Sequence[Mapping[str, Any]]
) -> str:
    format_candidates: list[Any] = []
    loose_candidates: list[Any] = []
    sources: list[Mapping[str, Any]] = []
    if envelope is not None:
        sources.append(envelope)
    if records:
        sources.append(records[0])
    for obj in sources:
        for key in _VERSION_KEYS:
            if obj.get(key) not in (None, ""):
                format_candidates.append(obj.get(key))
        if obj.get("version") not in (None, ""):
            loose_candidates.append(obj.get("version"))
    detected: str | None = None
    version_errors: list[HandoffVersionError] = []
    for candidate in format_candidates:
        try:
            canonical = _canonicalize_format_version(candidate, required=True)
        except HandoffVersionError as exc:
            version_errors.append(exc)
            continue
        if canonical is None:
            continue
        if detected is None:
            detected = canonical
        elif detected != canonical:
            raise HandoffVersionError("Claude Code export versions are ambiguous")
    if detected is not None and version_errors:
        raise HandoffVersionError("Claude Code export versions are ambiguous")
    if detected is not None:
        return detected
    if version_errors:
        raise version_errors[0]
    if records and _looks_like_claude_jsonl(records):
        return CANONICAL_EXPORT_VERSION
    for candidate in loose_candidates:
        canonical = _canonicalize_format_version(candidate, required=False)
        if canonical is None:
            continue
        if detected is None:
            detected = canonical
        elif detected != canonical:
            raise HandoffVersionError("Claude Code export versions are ambiguous")
    if detected is not None:
        return detected
    raise HandoffVersionError("unsupported Claude Code export version")


def _canonicalize_format_version(value: Any, *, required: bool) -> str | None:
    if isinstance(value, bool):
        raise HandoffVersionError("unsupported Claude Code export version")
    if isinstance(value, int):
        if value != 1:
            raise HandoffVersionError(
                f"unsupported Claude Code export version {value!r}; rebuild with {CANONICAL_EXPORT_VERSION}"
            )
        return CANONICAL_EXPORT_VERSION
    if not isinstance(value, str):
        raise HandoffVersionError("Claude Code export version must be a string")
    text = value.strip()
    if not text:
        return None
    lowered = text.lower().replace(" ", "")
    if lowered in {item.lower() for item in SUPPORTED_EXPORT_VERSIONS}:
        return CANONICAL_EXPORT_VERSION
    if "@" in text:
        name, suffix = text.rsplit("@", 1)
        if not suffix.isdigit():
            if not required:
                return None
            raise HandoffVersionError(f"unsupported Claude Code export version {text!r}")
        if suffix != "1":
            raise HandoffVersionError(
                f"unsupported Claude Code export version {text!r}; rebuild with {CANONICAL_EXPORT_VERSION}"
            )
        if _normalize_key(name) in {"claudecodeexport", "claude_code", "claudecode"}:
            return CANONICAL_EXPORT_VERSION
        if required:
            raise HandoffVersionError(f"unsupported Claude Code export version {text!r}")
        return None
    if lowered.startswith("claude-code-export-") or lowered.startswith("claude_code_export_"):
        suffix = lowered.rsplit("-", 1)[-1] if "-" in lowered else lowered.rsplit("_", 1)[-1]
        if suffix != "1":
            raise HandoffVersionError(
                f"unsupported Claude Code export version {text!r}; rebuild with {CANONICAL_EXPORT_VERSION}"
            )
        return CANONICAL_EXPORT_VERSION
    major = _semver_major(text)
    if major is not None:
        if major not in SUPPORTED_APP_MAJOR_VERSIONS:
            if required:
                raise HandoffVersionError(
                    f"unsupported Claude Code export version {text!r}; rebuild with {CANONICAL_EXPORT_VERSION}"
                )
            return None
        return CANONICAL_EXPORT_VERSION
    if required:
        raise HandoffVersionError(f"unsupported Claude Code export version {text!r}")
    return None


def _semver_major(value: str) -> int | None:
    head = value.split("+", 1)[0].split("-", 1)[0]
    if not head or not head[0].isdigit():
        return None
    major = head.split(".", 1)[0]
    if not major.isdigit():
        return None
    return int(major)


def _looks_like_claude_jsonl(records: Sequence[Mapping[str, Any]]) -> bool:
    sample = records[0]
    return any(
        key in sample for key in ("uuid", "parentUuid", "gitBranch", "sessionId", "message")
    ) and ("type" in sample or "kind" in sample or "role" in sample or "message" in sample)


@dataclass(frozen=True)
class _IndexedRecord:
    index: int
    record: Mapping[str, Any]
    uuid: str
    parent_uuid: str
    git_branch: str
    is_sidechain: bool
    branch_id: str
    timestamp_ms: int


def _index_records(
    records: Sequence[Mapping[str, Any]], envelope: Mapping[str, Any] | None
) -> tuple[_IndexedRecord, ...]:
    default_git = ""
    if envelope is not None:
        default_git = _optional_text(envelope.get("git_branch", envelope.get("gitBranch")))
    indexed: list[_IndexedRecord] = []
    for index, record in enumerate(records):
        mapping = _require_object(record, "Claude Code record")
        indexed.append(
            _IndexedRecord(
                index=index,
                record=mapping,
                uuid=_record_uuid(mapping),
                parent_uuid=_optional_text(
                    mapping.get("parentUuid", mapping.get("parent_uuid"))
                ),
                git_branch=_optional_text(
                    mapping.get("gitBranch", mapping.get("git_branch"))
                )
                or default_git,
                is_sidechain=bool(
                    mapping.get("isSidechain", mapping.get("is_sidechain", False))
                ),
                branch_id=_optional_text(
                    mapping.get("branch_id", mapping.get("branchId"))
                ),
                timestamp_ms=_timestamp_ms(
                    mapping.get("created_at_ms", mapping.get("timestamp"))
                ),
            )
        )
    return tuple(indexed)


def _collect_branches(
    indexed: Sequence[_IndexedRecord], *, envelope_git_branch: str
) -> tuple[ClaudeCodeBranch, ...]:
    by_uuid = {item.uuid: item for item in indexed if item.uuid}
    children: dict[str, list[_IndexedRecord]] = {}
    roots: list[_IndexedRecord] = []
    for item in indexed:
        parent = item.parent_uuid
        if parent and parent in by_uuid:
            children.setdefault(parent, []).append(item)
        else:
            roots.append(item)
    for group in children.values():
        group.sort(key=lambda item: (item.timestamp_ms, item.uuid, item.index))
    roots.sort(key=lambda item: (item.timestamp_ms, item.uuid, item.index))

    branches: list[ClaudeCodeBranch] = []
    seen_ids: set[str] = set()

    def emit(start: _IndexedRecord, *, sidechain: bool, parent_uuid: str) -> None:
        chain: list[str] = []
        current: _IndexedRecord | None = start
        git_branch = start.git_branch or envelope_git_branch
        is_sidechain = sidechain or start.is_sidechain
        while current is not None:
            if current.uuid:
                chain.append(current.uuid)
            git_branch = git_branch or current.git_branch
            is_sidechain = is_sidechain or current.is_sidechain
            kids = children.get(current.uuid, []) if current.uuid else []
            if len(kids) == 1:
                current = kids[0]
                continue
            for child in kids[1:] if kids else []:
                emit(child, sidechain=True, parent_uuid=current.uuid)
            current = kids[0] if kids else None
        branch_id = start.branch_id or start.uuid or f"branch:{start.index}"
        if branch_id in seen_ids:
            branch_id = f"{branch_id}:{start.index}"
        seen_ids.add(branch_id)
        branches.append(
            ClaudeCodeBranch(
                branch_id=branch_id,
                parent_uuid=parent_uuid,
                git_branch=git_branch,
                record_uuids=tuple(chain),
                is_sidechain=is_sidechain,
            )
        )

    if not roots:
        for item in indexed:
            emit(item, sidechain=item.is_sidechain, parent_uuid=item.parent_uuid)
        return tuple(branches)
    for root in roots:
        emit(root, sidechain=root.is_sidechain, parent_uuid=root.parent_uuid)
    return tuple(branches)


def _reject_authority_claims(
    indexed: Sequence[_IndexedRecord], *, truncated_export: bool
) -> None:
    seen_decisions: dict[str, ApprovalDecision] = {}
    for item in indexed:
        record = item.record
        if not _is_authority_claim(record):
            continue
        if truncated_export or record.get("truncated") is True or record.get("partial") is True:
            raise HandoffTrustError("truncated authority claim is rejected")
        if _authority_value_truncated(record):
            raise HandoffTrustError("truncated authority claim is rejected")
        decision = _extract_decision(record)
        if decision is None:
            raise HandoffTrustError("ambiguous authority claim is rejected")
        subject = _authority_subject(item)
        if not subject:
            raise HandoffTrustError("ambiguous authority claim is rejected")
        previous = seen_decisions.get(subject)
        if previous is not None and previous is not decision:
            raise HandoffTrustError("ambiguous authority claim is rejected")
        seen_decisions[subject] = decision
        if record.get("grants_effects") is True and not _optional_text(
            record.get("authority_binding_id")
        ):
            raise HandoffTrustError("ambiguous authority claim is rejected")


def _is_authority_claim(record: Mapping[str, Any]) -> bool:
    type_name = _normalize_key(record.get("type", record.get("kind", "")))
    if type_name in _AUTHORITY_RECORD_TYPES:
        return True
    keys = {_normalize_key(key) for key in record}
    if keys & _AUTHORITY_KEYS:
        return True
    nested = record.get("approval")
    if isinstance(nested, Mapping):
        return True
    message = record.get("message")
    if isinstance(message, Mapping):
        message_keys = {_normalize_key(key) for key in message}
        if message_keys & _AUTHORITY_KEYS:
            return True
    return False


def _authority_value_truncated(record: Mapping[str, Any]) -> bool:
    for key, value in record.items():
        if _normalize_key(key) not in _AUTHORITY_KEYS and key not in {
            "decision",
            "permissionDecision",
            "permission_decision",
            "approval",
        }:
            if isinstance(value, Mapping):
                if _authority_value_truncated(value):
                    return True
            continue
        if _is_truncated_text(value):
            return True
        if isinstance(value, Mapping) and _authority_value_truncated(value):
            return True
    return False


def _is_truncated_text(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    text = value.strip()
    if not text:
        return False
    if text.endswith(("...", "…")):
        return True
    if text.count("{") != text.count("}") or text.count("[") != text.count("]"):
        return True
    return False


def _extract_decision(record: Mapping[str, Any]) -> ApprovalDecision | None:
    candidates: list[Any] = [
        record.get("permissionDecision"),
        record.get("permission_decision"),
        record.get("decision"),
    ]
    approval = record.get("approval")
    if isinstance(approval, Mapping):
        candidates.extend(
            (
                approval.get("decision"),
                approval.get("permissionDecision"),
                approval.get("permission_decision"),
            )
        )
    elif isinstance(approval, str):
        candidates.append(approval)
    found: ApprovalDecision | None = None
    for candidate in candidates:
        if candidate in (None, ""):
            continue
        if not isinstance(candidate, str):
            raise HandoffTrustError("ambiguous authority claim is rejected")
        mapped = _DECISION_ALIASES.get(_normalize_key(candidate))
        if mapped is None:
            raise HandoffTrustError("ambiguous authority claim is rejected")
        if found is None:
            found = mapped
        elif found is not mapped:
            raise HandoffTrustError("ambiguous authority claim is rejected")
    return found


def _authority_subject(item: _IndexedRecord) -> str:
    record = item.record
    for key in ("subject_content_id", "subject", "tool_use_id", "toolUseId", "uuid"):
        text = _optional_text(record.get(key))
        if text:
            return text
    if item.uuid:
        return item.uuid
    if item.parent_uuid:
        return item.parent_uuid
    return ""


def _normalize_records(
    indexed: Sequence[_IndexedRecord],
    *,
    bounds: HandoffBounds,
    provenance: HandoffProvenance,
    envelope_residual: Mapping[str, Any],
    branches: Sequence[ClaudeCodeBranch],
) -> tuple[tuple[HandoffEvent, ...], dict[str, int]]:
    events: list[HandoffEvent] = []
    stats = {
        "rejected_event_count": 0,
        "truncated": False,
        "unknown_fields_retained": 0,
        "hidden_chain_of_thought_rejected": 0,
        "imported_success_claims_untrusted": 0,
    }
    invocation_ids: dict[str, str] = {}
    branch_by_uuid = _branch_lookup(branches)
    first_residual_attached = False

    for item in indexed:
        record = item.record
        if record.get("truncated") is True or record.get("partial") is True:
            stats["truncated"] = True
        cot_count = _count_hidden_cot(record)
        stats["hidden_chain_of_thought_rejected"] += cot_count
        residual = _record_residual(record, item, branch_by_uuid.get(item.uuid))
        pending_envelope_residual = False
        if not first_residual_attached and envelope_residual:
            merged = dict(envelope_residual)
            merged.update(dict(residual))
            residual = merged
            pending_envelope_residual = True
        residual = _freeze_residual(residual, bounds=bounds)
        type_name = _normalize_key(record.get("type", record.get("kind", "")))
        produced: list[HandoffEvent] = []
        try:
            if _is_authority_claim(record) and (
                type_name in _AUTHORITY_RECORD_TYPES or _extract_decision(record) is not None
            ):
                produced.append(
                    _approval_event(
                        item,
                        sequence=len(events),
                        bounds=bounds,
                        provenance=provenance,
                        residual=residual,
                    )
                )
            if type_name in {
                "summary",
                "progress",
                "file-history-snapshot",
                "file_history_snapshot",
                "queue-operation",
                "queue_operation",
            }:
                event = _summary_event(
                    item,
                    sequence=len(events) + len(produced),
                    bounds=bounds,
                    provenance=provenance,
                    residual=residual if not produced else MappingProxyType({}),
                )
                if event is not None:
                    produced.append(event)
                elif not produced:
                    stats["rejected_event_count"] += 1
            elif type_name not in _AUTHORITY_RECORD_TYPES or _content_blocks(record) or _message_text(record):
                produced.extend(
                    _message_events(
                        item,
                        sequence=len(events) + len(produced),
                        bounds=bounds,
                        provenance=provenance,
                        residual=residual if not produced else MappingProxyType({}),
                        invocation_ids=invocation_ids,
                    )
                )
        except (HandoffTrustError, HandoffVersionError, HandoffBoundsError):
            raise
        except HandoffContractError:
            stats["rejected_event_count"] += 1
            continue
        if not produced:
            if cot_count:
                stats["rejected_event_count"] += 1
            elif type_name not in {"summary", "progress", "file-history-snapshot", "file_history_snapshot"}:
                stats["rejected_event_count"] += 1
            continue
        if pending_envelope_residual:
            first_residual_attached = True
        for event in produced:
            if len(events) >= bounds.max_events:
                raise HandoffContractError("Claude Code export exceeds max_events")
            events.append(event)
            stats["unknown_fields_retained"] += len(event.residual_fields)
            if isinstance(event, ToolResultEvent) and event.claimed_success:
                stats["imported_success_claims_untrusted"] += 1
            if isinstance(event, PatchEvent) and event.claimed_applied:
                stats["imported_success_claims_untrusted"] += 1
            if isinstance(event, ToolInvocationEvent):
                tool_id = _optional_text(
                    event.residual_fields.get("tool_use_id") if event.residual_fields else ""
                ) or _optional_text(record.get("id", record.get("tool_use_id", record.get("toolUseId"))))
                if not tool_id:
                    blocks = _content_blocks(record)
                    for block in blocks:
                        if _normalize_key(block.get("type")) == "tool_use":
                            tool_id = _optional_text(block.get("id"))
                            if tool_id:
                                invocation_ids.setdefault(tool_id, event.event_id)
                elif tool_id:
                    invocation_ids.setdefault(tool_id, event.event_id)
                if item.uuid:
                    invocation_ids.setdefault(item.uuid, event.event_id)
    return tuple(events), stats


def _branch_lookup(branches: Sequence[ClaudeCodeBranch]) -> dict[str, ClaudeCodeBranch]:
    lookup: dict[str, ClaudeCodeBranch] = {}
    for branch in branches:
        for uuid in branch.record_uuids:
            lookup.setdefault(uuid, branch)
    return lookup


def _message_events(
    item: _IndexedRecord,
    *,
    sequence: int,
    bounds: HandoffBounds,
    provenance: HandoffProvenance,
    residual: Mapping[str, Any],
    invocation_ids: dict[str, str],
) -> list[HandoffEvent]:
    record = item.record
    produced: list[HandoffEvent] = []
    role = _conversation_role(record)
    text = _message_text(record)
    reasoning = _optional_text(record.get("reasoning_summary", record.get("summary")))
    blocks = _content_blocks(record)
    created_at_ms = item.timestamp_ms or provenance.captured_at_ms

    if text or reasoning:
        produced.append(
            ConversationEvent(
                sequence=sequence + len(produced),
                role=role,
                text=text,
                reasoning_summary=reasoning,
                residual_fields=residual if not produced else MappingProxyType({}),
                provenance=provenance,
                bounds=bounds,
                created_at_ms=created_at_ms,
            )
        )

    for block in blocks:
        block_type = _normalize_key(block.get("type", block.get("kind", "")))
        if block_type in _HIDDEN_CHAIN_OF_THOUGHT_KEYS or block_type in {"thinking", "redacted_thinking"}:
            continue
        if block_type in {"tool_use", "tooluse", "server_tool_use", "servertooluse"}:
            invocation = _invocation_from_block(
                block,
                sequence=sequence + len(produced),
                bounds=bounds,
                provenance=provenance,
                residual=residual if not produced else MappingProxyType({}),
                created_at_ms=created_at_ms,
            )
            produced.append(invocation)
            if _normalize_key(invocation.tool_name) in _PATCH_TOOLS:
                arguments = block.get("input", block.get("arguments", block.get("params", {})))
                if isinstance(arguments, Mapping):
                    try:
                        produced.append(
                            _patch_from_tool(
                                tool_name=invocation.tool_name,
                                arguments=arguments,
                                sequence=sequence + len(produced),
                                bounds=bounds,
                                provenance=provenance,
                                residual=MappingProxyType({}),
                                created_at_ms=created_at_ms,
                            )
                        )
                    except HandoffContractError:
                        pass
            continue
        if block_type in {"tool_result", "toolresult"}:
            produced.append(
                _result_from_block(
                    block,
                    sequence=sequence + len(produced),
                    bounds=bounds,
                    provenance=provenance,
                    residual=residual if not produced else MappingProxyType({}),
                    created_at_ms=created_at_ms,
                    invocation_ids=invocation_ids,
                )
            )
            continue
        if block_type in {"patch", "diff"}:
            produced.append(
                _patch_from_mapping(
                    block,
                    sequence=sequence + len(produced),
                    bounds=bounds,
                    provenance=provenance,
                    residual=residual if not produced else MappingProxyType({}),
                    created_at_ms=created_at_ms,
                )
            )

    tool_use = record.get("tool_use") or record.get("toolUse")
    if isinstance(tool_use, Mapping):
        produced.append(
            _invocation_from_block(
                tool_use,
                sequence=sequence + len(produced),
                bounds=bounds,
                provenance=provenance,
                residual=residual if not produced else MappingProxyType({}),
                created_at_ms=created_at_ms,
            )
        )
    tool_result = record.get("tool_result") or record.get("toolResult")
    if isinstance(tool_result, Mapping):
        produced.append(
            _result_from_block(
                tool_result,
                sequence=sequence + len(produced),
                bounds=bounds,
                provenance=provenance,
                residual=residual if not produced else MappingProxyType({}),
                created_at_ms=created_at_ms,
                invocation_ids=invocation_ids,
            )
        )
    if any(key in record for key in ("patch", "diff", "unified_diff")):
        produced.append(
            _patch_from_mapping(
                record,
                sequence=sequence + len(produced),
                bounds=bounds,
                provenance=provenance,
                residual=residual if not produced else MappingProxyType({}),
                created_at_ms=created_at_ms,
            )
        )
    return produced


def _invocation_from_block(
    block: Mapping[str, Any],
    *,
    sequence: int,
    bounds: HandoffBounds,
    provenance: HandoffProvenance,
    residual: Mapping[str, Any],
    created_at_ms: int,
) -> HandoffEvent:
    tool_name = _optional_text(block.get("name", block.get("tool_name", block.get("toolName"))))
    if not tool_name:
        raise HandoffContractError("tool invocation name is required")
    arguments = block.get("input", block.get("arguments", block.get("params", {})))
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, Mapping):
        raise HandoffContractError("tool arguments must be an object")
    tool_id = _optional_text(block.get("id", block.get("tool_use_id", block.get("toolUseId"))))
    extra = dict(residual)
    if tool_id:
        extra["tool_use_id"] = tool_id
    return ToolInvocationEvent(
        sequence=sequence,
        tool_name=tool_name,
        arguments=_json_safe_mapping(arguments),
        residual_fields=_freeze_residual(extra, bounds=bounds),
        provenance=provenance,
        bounds=bounds,
        created_at_ms=created_at_ms,
        executed=False,
    )


def _result_from_block(
    block: Mapping[str, Any],
    *,
    sequence: int,
    bounds: HandoffBounds,
    provenance: HandoffProvenance,
    residual: Mapping[str, Any],
    created_at_ms: int,
    invocation_ids: Mapping[str, str],
) -> ToolResultEvent:
    tool_name = _optional_text(
        block.get("name", block.get("tool_name", block.get("toolName")))
    ) or "tool"
    tool_id = _optional_text(block.get("tool_use_id", block.get("toolUseId", block.get("id"))))
    invocation_event_id = ""
    if tool_id and tool_id in invocation_ids:
        invocation_event_id = invocation_ids[tool_id]
    elif tool_id:
        invocation_event_id = content_identity(
            {"claude_code_tool_use_id": tool_id, "adapter": ADAPTER_ID}
        )
    else:
        invocation_event_id = content_identity(
            {"claude_code_tool_result": sequence, "adapter": ADAPTER_ID}
        )
    excerpt = _clip_text(_block_text(block), bounds.max_text_bytes)
    payload = _json_safe_value(block.get("content", block.get("result", excerpt)))
    result_content_id = content_identity(
        {"adapter": ADAPTER_ID, "tool_result": payload, "tool_use_id": tool_id}
    )
    claimed_success = _claimed_success(block)
    extra = dict(residual)
    if tool_id:
        extra["tool_use_id"] = tool_id
    return ToolResultEvent(
        sequence=sequence,
        tool_name=tool_name,
        invocation_event_id=invocation_event_id,
        result_content_id=result_content_id,
        result_excerpt=excerpt,
        claimed_success=claimed_success,
        residual_fields=_freeze_residual(extra, bounds=bounds),
        provenance=provenance,
        bounds=bounds,
        created_at_ms=created_at_ms,
        trusted_success=False,
    )


def _patch_from_tool(
    *,
    tool_name: str,
    arguments: Mapping[str, Any],
    sequence: int,
    bounds: HandoffBounds,
    provenance: HandoffProvenance,
    residual: Mapping[str, Any],
    created_at_ms: int,
) -> PatchEvent:
    path = _optional_text(
        arguments.get("file_path", arguments.get("filePath", arguments.get("path")))
    )
    diff = _optional_text(
        arguments.get("unified_diff", arguments.get("diff", arguments.get("patch")))
    )
    old_text = arguments.get("old_string", arguments.get("oldString"))
    new_text = arguments.get("new_string", arguments.get("newString", arguments.get("contents")))
    if not diff and (old_text is not None or new_text is not None):
        diff = (
            f"--- a/{path or 'file'}\n+++ b/{path or 'file'}\n"
            f"{old_text if isinstance(old_text, str) else ''}\n"
            f"{new_text if isinstance(new_text, str) else ''}"
        )
    if not diff and not path:
        raise HandoffContractError("patch tool arguments are incomplete")
    kind = PatchKind.UNIFIED_DIFF if diff else PatchKind.OVERLAY_REFERENCE
    payload = {"tool_name": tool_name, "path": path, "diff": diff or new_text or ""}
    return PatchEvent(
        sequence=sequence,
        patch_kind=kind,
        patch_content_id=content_identity({"adapter": ADAPTER_ID, "patch": payload}),
        paths=_safe_paths(path),
        claimed_applied=False,
        residual_fields=residual,
        provenance=provenance,
        bounds=bounds,
        created_at_ms=created_at_ms,
        applied=False,
    )


def _patch_from_mapping(
    mapping: Mapping[str, Any],
    *,
    sequence: int,
    bounds: HandoffBounds,
    provenance: HandoffProvenance,
    residual: Mapping[str, Any],
    created_at_ms: int,
) -> PatchEvent:
    nested = mapping.get("patch")
    if isinstance(nested, Mapping):
        mapping = nested
    diff = _optional_text(
        mapping.get("unified_diff", mapping.get("diff", mapping.get("patch")))
    )
    path = _optional_text(mapping.get("file_path", mapping.get("filePath", mapping.get("path"))))
    paths = mapping.get("paths")
    path_tuple = _safe_paths(path)
    if isinstance(paths, Sequence) and not isinstance(paths, (str, bytes, bytearray, memoryview)):
        collected: list[str] = []
        for item in paths:
            collected.extend(_safe_paths(_optional_text(item)))
        path_tuple = tuple(collected)
    if not diff and not path_tuple:
        raise HandoffContractError("patch payload is incomplete")
    return PatchEvent(
        sequence=sequence,
        patch_kind=PatchKind.UNIFIED_DIFF if diff else PatchKind.OVERLAY_REFERENCE,
        patch_content_id=content_identity(
            {"adapter": ADAPTER_ID, "patch": {"diff": diff, "paths": list(path_tuple)}}
        ),
        paths=path_tuple,
        claimed_applied=bool(mapping.get("claimed_applied") is True or mapping.get("applied") is True),
        residual_fields=residual,
        provenance=provenance,
        bounds=bounds,
        created_at_ms=created_at_ms,
        applied=False,
    )


def _approval_event(
    item: _IndexedRecord,
    *,
    sequence: int,
    bounds: HandoffBounds,
    provenance: HandoffProvenance,
    residual: Mapping[str, Any],
) -> ApprovalEvent:
    decision = _extract_decision(item.record)
    if decision is None:
        raise HandoffTrustError("ambiguous authority claim is rejected")
    subject = _authority_subject(item)
    subject_id = subject if _is_content_ref(subject) else content_identity(
        {"adapter": ADAPTER_ID, "approval_subject": subject or str(item.index)}
    )
    return ApprovalEvent(
        sequence=sequence,
        approval_kind=ApprovalKind.IMPORTED_CLAIM,
        decision=decision,
        subject_content_id=subject_id,
        provenance=provenance,
        authority_binding_id=_optional_text(item.record.get("authority_binding_id")),
        residual_fields=residual,
        bounds=bounds,
        created_at_ms=item.timestamp_ms or provenance.captured_at_ms,
        grants_effects=False,
    )


def _summary_event(
    item: _IndexedRecord,
    *,
    sequence: int,
    bounds: HandoffBounds,
    provenance: HandoffProvenance,
    residual: Mapping[str, Any],
) -> ConversationEvent | None:
    text = _message_text(item.record) or _optional_text(
        item.record.get("summary", item.record.get("text"))
    )
    if not text:
        return None
    return ConversationEvent(
        sequence=sequence,
        role=ConversationRole.SYSTEM,
        text=text,
        residual_fields=residual,
        provenance=provenance,
        bounds=bounds,
        created_at_ms=item.timestamp_ms or provenance.captured_at_ms,
    )


def _conversation_role(record: Mapping[str, Any]) -> ConversationRole:
    message = record.get("message")
    role_value = record.get("role")
    if isinstance(message, Mapping) and message.get("role") not in (None, ""):
        role_value = message.get("role")
    type_name = _normalize_key(record.get("type", record.get("kind", "")))
    if role_value in (None, ""):
        role_value = type_name
    mapped = _ROLE_ALIASES.get(_normalize_key(role_value))
    if mapped is not None:
        return mapped
    if type_name in {"tool", "tool_result", "toolresult"}:
        return ConversationRole.TOOL
    return ConversationRole.UNKNOWN


def _message_text(record: Mapping[str, Any]) -> str:
    message = record.get("message")
    if isinstance(message, str):
        return message.strip()
    if isinstance(message, Mapping):
        return _content_text(message.get("content", message.get("text", "")))
    return _content_text(record.get("content", record.get("text", "")))


def _content_text(content: Any) -> str:
    if content in (None, ""):
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, Mapping):
        return _block_text(content)
    if isinstance(content, Sequence) and not isinstance(content, (bytes, bytearray, memoryview)):
        parts = [_block_text(block) if isinstance(block, Mapping) else _optional_text(block) for block in content]
        return "\n".join(part for part in parts if part).strip()
    return ""


def _block_text(block: Mapping[str, Any]) -> str:
    block_type = _normalize_key(block.get("type", ""))
    if block_type in _HIDDEN_CHAIN_OF_THOUGHT_KEYS or block_type in {"thinking", "redacted_thinking"}:
        return ""
    if block_type in {"tool_use", "tool_result", "patch", "diff"}:
        return ""
    for key in ("text", "content", "result", "summary"):
        value = block.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray, memoryview)):
            nested = _content_text(value)
            if nested:
                return nested
    return ""


def _content_blocks(record: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    message = record.get("message")
    content: Any = None
    if isinstance(message, Mapping):
        content = message.get("content")
    if content is None:
        content = record.get("content")
    if isinstance(content, Mapping):
        return [content]
    if isinstance(content, Sequence) and not isinstance(content, (str, bytes, bytearray, memoryview)):
        return [item for item in content if isinstance(item, Mapping)]
    return []


def _count_hidden_cot(value: Any) -> int:
    count = 0
    if isinstance(value, Mapping):
        for key, item in value.items():
            if _normalize_key(key) in _HIDDEN_CHAIN_OF_THOUGHT_KEYS:
                count += 1
            else:
                count += _count_hidden_cot(item)
        type_name = _normalize_key(value.get("type", value.get("kind", "")))
        if type_name in _HIDDEN_CHAIN_OF_THOUGHT_KEYS or type_name in {"thinking", "redacted_thinking"}:
            count += 1
        return count
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray, memoryview)):
        return sum(_count_hidden_cot(item) for item in value)
    return 0


def _record_residual(
    record: Mapping[str, Any], item: _IndexedRecord, branch: ClaudeCodeBranch | None
) -> Mapping[str, Any]:
    residual: dict[str, Any] = {}
    for source_key, dest_key in _BRANCH_RESIDUAL_KEYS:
        if source_key in record and record[source_key] not in (None, ""):
            residual[dest_key] = _json_safe_value(record[source_key])
    if item.git_branch and "git_branch" not in residual:
        residual["git_branch"] = item.git_branch
    if item.is_sidechain:
        residual["is_sidechain"] = True
    if branch is not None:
        residual.setdefault("branch_id", branch.branch_id)
        if branch.git_branch:
            residual.setdefault("git_branch", branch.git_branch)
        residual["is_sidechain"] = bool(residual.get("is_sidechain") or branch.is_sidechain)
    for key, value in record.items():
        if key in _RECORD_CONSUMED:
            continue
        if any(key == source for source, _dest in _BRANCH_RESIDUAL_KEYS):
            continue
        reason = _forbidden_reason(key)
        if reason == "hidden_chain_of_thought":
            continue
        if reason is not None:
            raise HandoffContractError(f"Claude Code export must not contain {reason.replace('_', ' ')}")
        residual[key] = _json_safe_value(value)
    return residual


def _envelope_residual(envelope: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if envelope is None:
        return {}
    residual: dict[str, Any] = {}
    for key, value in envelope.items():
        if key in _ENVELOPE_CONSUMED:
            continue
        reason = _forbidden_reason(key)
        if reason == "hidden_chain_of_thought":
            continue
        if reason is not None:
            raise HandoffContractError(f"Claude Code export must not contain {reason.replace('_', ' ')}")
        residual[key] = _json_safe_value(value)
    return residual


def _freeze_residual(value: Mapping[str, Any], *, bounds: HandoffBounds) -> Mapping[str, Any]:
    if not value:
        return MappingProxyType({})
    cleaned: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key.strip():
            continue
        reason = _forbidden_reason(key)
        if reason == "hidden_chain_of_thought":
            continue
        if reason is not None:
            raise HandoffContractError(f"residual_fields must not contain {reason.replace('_', ' ')}")
        cleaned[key.strip()] = item
    if len(cleaned) > bounds.max_unknown_fields:
        raise HandoffBoundsError("residual_fields exceeds its field-count limit")
    return cleaned


def _forbidden_reason(key: str) -> str | None:
    normalized = _normalize_key(key)
    if normalized in _HIDDEN_CHAIN_OF_THOUGHT_KEYS:
        return "hidden_chain_of_thought"
    if normalized in _TRANSCRIPT_BODY_KEYS:
        return "transcript_body"
    if any(
        normalized == marker or normalized.endswith("_" + marker) or marker in normalized
        for marker in _PRIVATE_FIELD_MARKERS
    ):
        return "private_material"
    return None


def _json_safe_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _json_safe_value(item) for key, item in value.items()}


def _json_safe_value(value: Any) -> Any:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        raise HandoffContractError("Claude Code export cannot contain floats")
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return _json_safe_mapping(value)
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, memoryview)):
        return [_json_safe_value(item) for item in value]
    raise HandoffContractError(
        f"Claude Code export contains unsupported value type {type(value).__name__}"
    )


def _claimed_success(block: Mapping[str, Any]) -> bool:
    if block.get("is_error") is True or block.get("isError") is True:
        return False
    if block.get("claimed_success") is True or block.get("success") is True:
        return True
    status = _normalize_key(block.get("status", block.get("subtype", "")))
    return status in {"success", "ok", "completed"}


def _safe_paths(path: str) -> tuple[str, ...]:
    text = path.strip().replace("\\", "/")
    if not text or text in {".", "/"}:
        return ()
    if text.startswith("/") or ".." in text.split("/") or (":" in text.split("/", 1)[0]):
        return ()
    return (text.removeprefix("./"),)


def _record_uuid(record: Mapping[str, Any]) -> str:
    return _optional_text(record.get("uuid", record.get("id")))


def _optional_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return ""
    if isinstance(value, str):
        return value.strip()
    return ""


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise HandoffContractError(f"{name} must be a non-negative integer")
    return value


def _timestamp_ms(value: Any) -> int:
    if value in (None, ""):
        return 0
    if isinstance(value, bool):
        raise HandoffContractError("timestamp must be a non-negative integer or ISO-8601 string")
    if isinstance(value, int):
        if value < 0:
            raise HandoffContractError("timestamp must be a non-negative integer")
        if value < 1_000_000_000_000:
            return value * 1000
        return value
    if not isinstance(value, str):
        raise HandoffContractError("timestamp must be a non-negative integer or ISO-8601 string")
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise HandoffContractError("timestamp must be ISO-8601") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    delta = parsed.astimezone(timezone.utc) - _EPOCH
    return delta.days * 86_400_000 + delta.seconds * 1000 + delta.microseconds // 1000


def _clip_text(text: str, max_bytes: int) -> str:
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


def _is_content_ref(value: str) -> bool:
    if value.startswith("sha256:") and len(value) == 71:
        return all(char in "0123456789abcdef" for char in value[7:])
    return value.startswith("b") and value.islower() and len(value) >= 21


def _require_object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise HandoffContractError(f"{name} must be an object")
    if not all(isinstance(key, str) for key in value):
        raise HandoffContractError(f"{name} object keys must be strings")
    return dict(value)


def _public_payload(payload: Any) -> Any:
    if isinstance(payload, Mapping):
        return {key: payload[key] for key in payload if _forbidden_reason(str(key)) is None}
    return payload


def _normalize_key(value: Any) -> str:
    return str(value).strip().lower().replace("-", "_")


__all__ = (
    "ADAPTER_ID",
    "CANONICAL_EXPORT_VERSION",
    "SUPPORTED_EXPORT_VERSIONS",
    "ClaudeCodeBranch",
    "ClaudeCodeExportAdapter",
    "ClaudeCodeNormalizationResult",
    "detect_claude_code_export_version",
    "normalize_claude_code_export",
)
