"""Deterministic tests for the EAAEF-013 Claude Code export adapter."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.handoff.adapters.claude_code import (
    ADAPTER_ID,
    ClaudeCodeAdapterError,
    ClaudeCodeExportAdapter,
    normalize_claude_code_export,
)
from ipfs_accelerate_py.agent_supervisor.handoff.contracts import (
    ConversationEvent,
    ConversationRole,
    EventKind,
    HandoffTrustError,
    HandoffVersionError,
    PatchEvent,
    SourceFamily,
    ToolInvocationEvent,
    ToolResultEvent,
    decode_handoff_event,
    normalized_stream_identity,
    validate_event_sequence,
)


FIXED_MS = 1_700_000_000_000
SHA_A = "sha256:" + ("a" * 64)
SHA_B = "sha256:" + ("b" * 64)
SHA_C = "sha256:" + ("c" * 64)


def _export() -> dict:
    return {
        "version": "claude-code-export-1",
        "git_branch": "feat/claude-handoff",
        "branches": ["feat/claude-handoff", "main"],
        "git": {
            "branch": "feat/claude-handoff",
            "diff": "--- a/src/example.py\n+++ b/src/example.py\n@@\n-old\n+new\n",
        },
        "client_tag": "visible",
        "messages": [
            {"role": "user", "content": "fix the overlay", "custom_meta": "keep-me"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "I'll inspect the files."},
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "Read",
                        "input": {"path": "src/example.py"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": "ok",
                        "is_error": False,
                    }
                ],
            },
        ],
    }


def _normalize(payload: object, **changes: object):
    return normalize_claude_code_export(
        payload,
        captured_at_ms=FIXED_MS,
        request_id=SHA_A,
        session_id=SHA_B,
        raw_export_id=SHA_C,
        **changes,
    )


def test_normalize_conversation_tools_and_patch() -> None:
    events, report = _normalize(_export())
    assert report.source_family is SourceFamily.CLAUDE_CODE
    assert report.request_id == SHA_A
    assert report.session_id == SHA_B
    assert report.raw_export_id == SHA_C
    assert report.imported_invocations_not_executed is True
    assert report.imported_success_claims_untrusted == 1
    assert report.truncated is False
    assert report.normalized_stream_id == normalized_stream_identity(
        report.accepted_event_ids
    )
    assert "transcript" not in report.to_dict()
    assert "transcript_body" not in report.to_dict()
    kinds = [type(event) for event in events]
    assert kinds == [
        ConversationEvent,
        ConversationEvent,
        ToolInvocationEvent,
        ToolResultEvent,
        PatchEvent,
    ]
    assert events[0].role is ConversationRole.USER
    assert events[0].text == "fix the overlay"
    assert events[1].role is ConversationRole.ASSISTANT
    assert events[1].text == "I'll inspect the files."
    invocation = events[2]
    assert invocation.kind is EventKind.TOOL_INVOCATION
    assert invocation.executed is False
    assert invocation.tool_name == "Read"
    assert dict(invocation.arguments) == {"path": "src/example.py"}
    result = events[3]
    assert result.kind is EventKind.TOOL_RESULT
    assert result.claimed_success is True
    assert result.trusted_success is False
    assert result.invocation_event_id == invocation.event_id
    patch = events[4]
    assert patch.kind is EventKind.PATCH
    assert patch.applied is False
    assert patch.paths == ("src/example.py",)
    assert validate_event_sequence(events) == report.accepted_event_ids
    decoded = decode_handoff_event(invocation.to_dict())
    assert isinstance(decoded, ToolInvocationEvent)
    assert decoded.event_id == invocation.event_id
    assert ADAPTER_ID == "claude-code@1"
    assert ADAPTER_ID == ClaudeCodeExportAdapter.adapter_id
    assert events[0].provenance.adapter_id == ADAPTER_ID
    assert events[0].provenance.source_family is SourceFamily.CLAUDE_CODE
    assert events[0].provenance.source_export_version == "claude-code-export-1"


def test_unsupported_version_is_rejected() -> None:
    with pytest.raises(HandoffVersionError, match="unsupported"):
        _normalize(
            {
                "version": "claude-code-export-99",
                "messages": [{"role": "user", "content": "x"}],
            }
        )
    with pytest.raises(HandoffVersionError, match="ambiguous"):
        _normalize(
            {
                "version": "claude-code-export-1",
                "export_version": "claude-code-session-1",
                "messages": [{"role": "user", "content": "x"}],
            }
        )


def test_truncated_export_is_rejected() -> None:
    with pytest.raises(ClaudeCodeAdapterError, match="truncated"):
        _normalize(
            {
                "version": "claude-code-export-1",
                "truncated": True,
                "messages": [{"role": "user", "content": "x"}],
            }
        )
    with pytest.raises(ClaudeCodeAdapterError, match="truncated"):
        _normalize('{"version":"claude-code-export-1","messages":[')
    with pytest.raises(HandoffTrustError, match="authority"):
        _normalize(
            {
                "version": "claude-code-export-1",
                "truncated": True,
                "accepted": True,
                "messages": [{"role": "user", "content": "x"}],
            }
        )


def test_branches_are_preserved() -> None:
    events, _report = _normalize(_export())
    first = events[0]
    assert first.residual_fields["git_branch"] == "feat/claude-handoff"
    assert tuple(first.residual_fields["branches"]) == (
        "feat/claude-handoff",
        "main",
    )
    patch = next(event for event in events if isinstance(event, PatchEvent))
    assert patch.residual_fields["git_branch"] == "feat/claude-handoff"
    assert "feat/claude-handoff" in tuple(patch.residual_fields["branches"])


def test_residual_fields_are_preserved() -> None:
    events, report = _normalize(_export())
    assert events[0].residual_fields["client_tag"] == "visible"
    assert events[0].residual_fields["custom_meta"] == "keep-me"
    assert report.unknown_fields_retained >= 3
    restored = ConversationEvent.from_json(events[0].to_json())
    assert restored.residual_fields["client_tag"] == "visible"
    assert restored.residual_fields["custom_meta"] == "keep-me"


def test_hidden_chain_of_thought_is_rejected() -> None:
    payload = _export()
    payload["messages"][1]["thinking"] = "secret scratchpad"
    with pytest.raises(HandoffTrustError, match="hidden"):
        _normalize(payload)
    payload = _export()
    payload["messages"][1]["content"].append(
        {"type": "thinking", "thinking": "hidden"}
    )
    with pytest.raises(HandoffTrustError, match="hidden"):
        _normalize(payload)


def test_imported_success_is_untrusted() -> None:
    events, report = _normalize(_export())
    result = next(event for event in events if isinstance(event, ToolResultEvent))
    assert result.claimed_success is True
    assert result.trusted_success is False
    assert report.imported_success_claims_untrusted == 1
    with pytest.raises(HandoffTrustError):
        ToolResultEvent(
            sequence=99,
            tool_name="Read",
            invocation_event_id=SHA_A,
            result_content_id=SHA_B,
            provenance=result.provenance,
            claimed_success=True,
            trusted_success=True,
        )


def test_imported_invocations_are_not_executed() -> None:
    events, report = _normalize(_export())
    invocation = next(event for event in events if isinstance(event, ToolInvocationEvent))
    assert invocation.executed is False
    assert report.imported_invocations_not_executed is True
    with pytest.raises(HandoffTrustError):
        ToolInvocationEvent(
            sequence=99,
            tool_name="Read",
            arguments={"path": "src/example.py"},
            provenance=invocation.provenance,
            executed=True,
        )


def test_imported_authority_claim_is_rejected() -> None:
    payload = _export()
    payload["accepted"] = True
    with pytest.raises(HandoffTrustError, match="authority"):
        _normalize(payload)
