"""Deterministic tests for the EAAEF-013 Claude Code export adapter."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError

import pytest

from ipfs_accelerate_py.agent_supervisor.handoff.adapters.claude_code import (
    ADAPTER_ID,
    CANONICAL_EXPORT_VERSION,
    SUPPORTED_EXPORT_VERSIONS,
    ClaudeCodeExportAdapter,
    detect_claude_code_export_version,
    normalize_claude_code_export,
)
from ipfs_accelerate_py.agent_supervisor.handoff.contracts import (
    ApprovalDecision,
    ApprovalEvent,
    ApprovalKind,
    ConversationEvent,
    ConversationRole,
    EventKind,
    HandoffBounds,
    HandoffBoundsError,
    HandoffContractError,
    HandoffTrustError,
    HandoffVersionError,
    PatchEvent,
    SourceFamily,
    ToolInvocationEvent,
    ToolResultEvent,
    TrustClass,
)


FIXED_MS = 1_700_000_000_000
SHA_A = "sha256:" + ("a" * 64)
SHA_B = "sha256:" + ("b" * 64)
SHA_C = "sha256:" + ("c" * 64)


def _normalize(payload: object, **changes: object):
    values: dict[str, object] = {
        "raw_export_id": SHA_A,
        "request_id": SHA_B,
        "captured_at_ms": FIXED_MS,
    }
    values.update(changes)
    return normalize_claude_code_export(payload, **values)


def _user(text: str, **fields: object) -> dict[str, object]:
    record: dict[str, object] = {
        "type": "user",
        "uuid": fields.pop("uuid", "user-1"),
        "parentUuid": fields.pop("parentUuid", None),
        "timestamp": "2023-11-14T22:13:20+00:00",
        "gitBranch": fields.pop("gitBranch", "main"),
        "message": {"role": "user", "content": text},
        "version": "1.0.80",
    }
    record.update(fields)
    return record


def _assistant(text: str, **fields: object) -> dict[str, object]:
    content: object = fields.pop("content", [{"type": "text", "text": text}])
    record: dict[str, object] = {
        "type": "assistant",
        "uuid": fields.pop("uuid", "assistant-1"),
        "parentUuid": fields.pop("parentUuid", "user-1"),
        "timestamp": "2023-11-14T22:13:21+00:00",
        "gitBranch": fields.pop("gitBranch", "main"),
        "message": {"role": "assistant", "content": content},
        "version": "1.0.80",
    }
    record.update(fields)
    return record


def _export(*records: dict[str, object], **envelope: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "export_version": "claude-code-export-1",
        "source_family": "claude_code",
        "records": list(records),
    }
    payload.update(envelope)
    return payload


def test_detects_supported_export_versions() -> None:
    records = [_user("hello")]
    assert detect_claude_code_export_version(_export(*records)) == CANONICAL_EXPORT_VERSION
    assert (
        detect_claude_code_export_version(_export(*records, export_version="ClaudeCodeExport@1"))
        == CANONICAL_EXPORT_VERSION
    )
    assert detect_claude_code_export_version(_export(*records, export_version=1)) == CANONICAL_EXPORT_VERSION
    assert detect_claude_code_export_version(_export(*records, export_version="1")) == CANONICAL_EXPORT_VERSION
    jsonl = json.dumps(_user("hello"), separators=(",", ":")) + "\n" + json.dumps(
        _assistant("hi"), separators=(",", ":")
    )
    assert detect_claude_code_export_version(jsonl) == CANONICAL_EXPORT_VERSION
    assert detect_claude_code_export_version(jsonl.encode("utf-8")) == CANONICAL_EXPORT_VERSION
    assert "claude-code-export-1" in SUPPORTED_EXPORT_VERSIONS


def test_unsupported_and_ambiguous_export_versions_are_rejected() -> None:
    with pytest.raises(HandoffVersionError, match="unsupported Claude Code export version"):
        detect_claude_code_export_version(
            _export(_user("hello"), export_version="claude-code-export-2")
        )
    with pytest.raises(HandoffVersionError, match="unsupported"):
        detect_claude_code_export_version(_export(_user("hello"), export_version="ClaudeCodeExport@2"))
    with pytest.raises(HandoffVersionError, match="ambiguous"):
        detect_claude_code_export_version(
            {
                "export_version": "claude-code-export-1",
                "schema": "ClaudeCodeExport@2",
                "records": [_user("hello")],
            }
        )
    with pytest.raises(HandoffVersionError):
        detect_claude_code_export_version({"records": [{"text": "not claude"}]})


def test_adapter_normalizes_messages_tools_patches_and_approvals() -> None:
    export = _export(
        _user("continue the work"),
        _assistant(
            "editing",
            content=[
                {"type": "text", "text": "I will edit the file"},
                {
                    "type": "tool_use",
                    "id": "tool-1",
                    "name": "Edit",
                    "input": {
                        "file_path": "src/example.py",
                        "old_string": "a = 1",
                        "new_string": "a = 2",
                    },
                },
            ],
        ),
        {
            "type": "user",
            "uuid": "tool-result-1",
            "parentUuid": "assistant-1",
            "timestamp": "2023-11-14T22:13:22+00:00",
            "gitBranch": "main",
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tool-1",
                        "content": "updated src/example.py",
                        "is_error": False,
                        "success": True,
                    }
                ],
            },
        },
        {
            "type": "approval",
            "uuid": "approval-1",
            "parentUuid": "assistant-1",
            "decision": "approve",
            "subject_content_id": SHA_C,
            "gitBranch": "main",
        },
    )
    result = _normalize(export)
    assert result.export_version == CANONICAL_EXPORT_VERSION
    assert result.adapter_id == ADAPTER_ID
    assert result.source_family is SourceFamily.CLAUDE_CODE
    assert result.provenance.source_family is SourceFamily.CLAUDE_CODE
    assert result.provenance.source_export_version == CANONICAL_EXPORT_VERSION
    assert result.provenance.adapter_id == ADAPTER_ID
    assert result.provenance.trust_class is TrustClass.IMPORTED_EXPORTABLE
    kinds = [event.kind for event in result.events]
    assert EventKind.CONVERSATION in kinds
    assert EventKind.TOOL_INVOCATION in kinds
    assert EventKind.PATCH in kinds
    assert EventKind.TOOL_RESULT in kinds
    assert EventKind.APPROVAL in kinds
    conversation = next(event for event in result.events if isinstance(event, ConversationEvent))
    assert conversation.role is ConversationRole.USER
    assert conversation.text == "continue the work"
    invocation = next(event for event in result.events if isinstance(event, ToolInvocationEvent))
    assert invocation.tool_name == "Edit"
    assert invocation.executed is False
    assert invocation.arguments["file_path"] == "src/example.py"
    patch = next(event for event in result.events if isinstance(event, PatchEvent))
    assert patch.applied is False
    assert "src/example.py" in patch.paths
    tool_result = next(event for event in result.events if isinstance(event, ToolResultEvent))
    assert tool_result.claimed_success is True
    assert tool_result.trusted_success is False
    assert tool_result.invocation_event_id == invocation.event_id
    approval = next(event for event in result.events if isinstance(event, ApprovalEvent))
    assert approval.approval_kind is ApprovalKind.IMPORTED_CLAIM
    assert approval.decision is ApprovalDecision.APPROVE
    assert approval.grants_effects is False
    assert result.report.imported_invocations_not_executed is True
    assert result.report.imported_success_claims_untrusted >= 1
    assert result.session.source_family is SourceFamily.CLAUDE_CODE
    assert result.session.raw_export_id == SHA_A
    assert result.session.normalized_stream_id == result.report.normalized_stream_id
    assert result.session.raw_export_id != result.session.normalized_stream_id


def test_preserves_conversation_branches_and_git_branches() -> None:
    export = _export(
        _user("start", uuid="root", parentUuid=None, gitBranch="main"),
        _assistant("path a", uuid="a1", parentUuid="root", gitBranch="main"),
        _assistant(
            "path b",
            uuid="b1",
            parentUuid="root",
            gitBranch="feat/alt",
            isSidechain=True,
        ),
        branches=[
            {
                "branch_id": "explicit-side",
                "git_branch": "feat/explicit",
                "parentUuid": "root",
                "is_sidechain": True,
                "records": [
                    _assistant(
                        "path c",
                        uuid="c1",
                        parentUuid="root",
                        gitBranch="feat/explicit",
                        isSidechain=True,
                    )
                ],
            }
        ],
    )
    result = _normalize(export)
    texts = [event.text for event in result.events if isinstance(event, ConversationEvent)]
    assert "start" in texts
    assert "path a" in texts
    assert "path b" in texts
    assert "path c" in texts
    git_branches = {branch.git_branch for branch in result.branches}
    assert "main" in git_branches
    assert "feat/alt" in git_branches
    assert "feat/explicit" in git_branches
    assert any(branch.is_sidechain for branch in result.branches)
    side = next(
        event
        for event in result.events
        if isinstance(event, ConversationEvent) and event.text == "path b"
    )
    assert side.residual_fields["parent_uuid"] == "root"
    assert side.residual_fields["git_branch"] == "feat/alt"
    assert side.residual_fields["is_sidechain"] is True
    assert side.residual_fields["uuid"] == "b1"
    assert "branch_id" in side.residual_fields


def test_preserves_bounded_residual_fields() -> None:
    export = _export(
        _user("hello", cwd="/workspace", client_tag="visible", extra_flag=True),
        client_label="envelope-tag",
    )
    result = _normalize(export)
    conversation = next(event for event in result.events if isinstance(event, ConversationEvent))
    assert conversation.residual_fields["client_tag"] == "visible"
    assert conversation.residual_fields["extra_flag"] is True
    assert conversation.residual_fields["cwd"] == "/workspace"
    assert conversation.residual_fields["client_label"] == "envelope-tag"
    assert result.report.unknown_fields_retained >= 3
    restored = ConversationEvent.from_json(conversation.to_json())
    assert restored.residual_fields == conversation.residual_fields
    too_many = {f"field_{index}": "x" for index in range(HandoffBounds().max_unknown_fields + 1)}
    with pytest.raises(HandoffBoundsError):
        _normalize(_export(_user("hello", **too_many)))


def test_rejects_truncated_authority_claims() -> None:
    truncated_record = {
        "type": "approval",
        "uuid": "approval-trunc",
        "decision": "approve",
        "subject_content_id": SHA_C,
        "truncated": True,
    }
    with pytest.raises(HandoffTrustError, match="truncated authority claim"):
        _normalize(_export(_user("hello"), truncated_record))
    with pytest.raises(HandoffTrustError, match="truncated authority claim"):
        _normalize(
            _export(
                _user("hello"),
                {
                    "type": "approval",
                    "uuid": "approval-ellipsis",
                    "decision": "approve...",
                    "subject_content_id": SHA_C,
                },
            )
        )
    jsonl = (
        json.dumps(_user("hello"), separators=(",", ":"))
        + "\n"
        + json.dumps(
            {
                "type": "approval",
                "uuid": "approval-complete",
                "decision": "approve",
                "subject_content_id": SHA_C,
            },
            separators=(",", ":"),
        )
        + "\n"
        + '{"type":"approval","decision":"appr'
    )
    with pytest.raises(HandoffTrustError, match="truncated authority claim"):
        _normalize(jsonl)


def test_rejects_ambiguous_authority_claims() -> None:
    with pytest.raises(HandoffTrustError, match="ambiguous authority claim"):
        _normalize(
            _export(
                {
                    "type": "approval",
                    "uuid": "approval-empty",
                    "decision": "",
                    "subject_content_id": SHA_C,
                }
            )
        )
    with pytest.raises(HandoffTrustError, match="ambiguous authority claim"):
        _normalize(
            _export(
                {
                    "type": "approval",
                    "uuid": "approval-maybe",
                    "decision": "maybe",
                    "subject_content_id": SHA_C,
                }
            )
        )
    with pytest.raises(HandoffTrustError, match="ambiguous authority claim"):
        _normalize(
            _export(
                {
                    "type": "approval",
                    "uuid": "approval-a",
                    "decision": "approve",
                    "subject": "same-subject",
                },
                {
                    "type": "approval",
                    "uuid": "approval-b",
                    "decision": "reject",
                    "subject": "same-subject",
                },
            )
        )
    with pytest.raises(HandoffTrustError, match="ambiguous authority claim"):
        _normalize(
            _export(
                {
                    "type": "approval",
                    "uuid": "approval-grant",
                    "decision": "approve",
                    "subject_content_id": SHA_C,
                    "grants_effects": True,
                }
            )
        )


def test_hidden_chain_of_thought_is_not_represented() -> None:
    export = _export(
        _assistant(
            "",
            content=[
                {"type": "thinking", "thinking": "secret scratchpad"},
                {"type": "text", "text": "visible answer"},
            ],
        )
    )
    result = _normalize(export)
    conversation = next(event for event in result.events if isinstance(event, ConversationEvent))
    assert conversation.text == "visible answer"
    assert "thinking" not in conversation.residual_fields
    assert "secret scratchpad" not in conversation.to_json()
    assert result.report.hidden_chain_of_thought_rejected >= 1
    with pytest.raises(HandoffContractError, match="hidden chain-of-thought"):
        ConversationEvent.from_dict(
            {**conversation.to_dict(), "residual_fields": {"thinking": "no"}}
        )


def test_private_material_and_transcript_bodies_are_rejected() -> None:
    with pytest.raises(HandoffContractError, match="private material"):
        _normalize(_export(_user("hello", api_key="sk-secret")))
    with pytest.raises(HandoffContractError, match="transcript"):
        _normalize(_export(_user("hello"), transcript_body="full dump"))


def test_normalization_is_deterministic_and_frozen() -> None:
    export = _export(
        _user("hello", uuid="root"),
        _assistant("hi", uuid="child", parentUuid="root"),
    )
    first = _normalize(export)
    second = _normalize(json.loads(json.dumps(export)))
    assert first.session.content_id == second.session.content_id
    assert first.report.content_id == second.report.content_id
    assert [event.content_id for event in first.events] == [
        event.content_id for event in second.events
    ]
    assert first.session.to_json() == json.dumps(
        json.loads(first.session.to_json()),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    with pytest.raises(FrozenInstanceError):
        first.session.source_family = SourceFamily.CODEX  # type: ignore[misc]


def test_class_adapter_matches_module_functions() -> None:
    adapter = ClaudeCodeExportAdapter()
    payload = _export(_user("hello"))
    assert adapter.detect_version(payload) == detect_claude_code_export_version(payload)
    via_class = adapter.normalize(payload, raw_export_id=SHA_A, request_id=SHA_B, captured_at_ms=FIXED_MS)
    via_function = _normalize(payload)
    assert via_class.session.content_id == via_function.session.content_id
    assert via_class.report.content_id == via_function.report.content_id


def test_malformed_and_empty_exports_fail_closed() -> None:
    with pytest.raises(HandoffContractError):
        _normalize("")
    with pytest.raises(HandoffContractError):
        _normalize("not json and not jsonl {")
    with pytest.raises(HandoffContractError):
        _normalize(b"\xff\xfe")
    with pytest.raises(HandoffContractError):
        _normalize({"export_version": "claude-code-export-1", "records": "nope"})
    with pytest.raises(HandoffContractError, match="floats"):
        _normalize(_export(_user("hello", score=1.5)))
    with pytest.raises(HandoffContractError, match="source_family"):
        _normalize(_export(_user("hello"), source_family="codex"))


def test_truncated_jsonl_without_authority_is_marked_truncated() -> None:
    jsonl = json.dumps(_user("hello"), separators=(",", ":")) + "\n" + '{"type":"assistant","message":'
    result = _normalize(jsonl)
    assert result.report.truncated is True
    texts = [event.text for event in result.events if isinstance(event, ConversationEvent)]
    assert "hello" in texts


def test_complete_approval_does_not_grant_effects() -> None:
    result = _normalize(
        _export(
            _user("please approve"),
            {
                "type": "approval",
                "uuid": "approval-ok",
                "decision": "allow",
                "subject_content_id": SHA_C,
                "authority_binding_id": "binding:imported",
            },
        )
    )
    approval = next(event for event in result.events if isinstance(event, ApprovalEvent))
    assert approval.grants_effects is False
    assert approval.decision is ApprovalDecision.APPROVE
    assert approval.authority_binding_id == "binding:imported"
