"""Deterministic tests for the EAAEF-012 Codex export adapter."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.handoff.adapters.codex import (
    ADAPTER_ID,
    CodexAdapterError,
    CodexExportAdapter,
    normalize_codex_export,
)
from ipfs_accelerate_py.agent_supervisor.handoff.contracts import (
    ConversationEvent,
    ConversationRole,
    HandoffTrustError,
    PatchEvent,
    SourceFamily,
    ToolInvocationEvent,
    ToolResultEvent,
    validate_event_sequence,
)


FIXED_MS = 1_700_000_000_000


def _export() -> dict:
    return {
        "version": "codex-export-1",
        "items": [
            {"type": "message", "role": "user", "text": "fix the overlay"},
            {
                "type": "message",
                "role": "assistant",
                "text": "I'll inspect the files.",
                "reasoning_summary": "Need the overlay source.",
            },
            {
                "type": "function_call",
                "id": "call-1",
                "name": "read_file",
                "arguments": {"path": "handoff/contracts.py"},
            },
            {
                "type": "function_call_output",
                "call_id": "call-1",
                "name": "read_file",
                "output": "ok",
                "claimed_success": True,
            },
            {
                "type": "patch",
                "diff": "--- a/x\n+++ b/x\n@@\n-old\n+new\n",
                "paths": ["x"],
            },
        ],
        "client": "codex-cli",
    }


def test_normalize_messages_tools_patches_and_reasoning_summary() -> None:
    events, report = normalize_codex_export(_export(), captured_at_ms=FIXED_MS)
    assert report.source_family is SourceFamily.CODEX
    assert report.imported_invocations_not_executed is True
    assert report.imported_success_claims_untrusted == 1
    assert report.truncated is False
    kinds = [type(event) for event in events]
    assert kinds == [
        ConversationEvent,
        ConversationEvent,
        ToolInvocationEvent,
        ToolResultEvent,
        PatchEvent,
    ]
    assert events[0].role is ConversationRole.USER
    assert events[1].reasoning_summary == "Need the overlay source."
    invocation = events[2]
    assert invocation.executed is False
    assert invocation.tool_name == "read_file"
    result = events[3]
    assert result.claimed_success is True
    assert result.trusted_success is False
    patch = events[4]
    assert patch.applied is False
    assert patch.paths == ("x",)
    assert validate_event_sequence(events) == report.accepted_event_ids
    assert ADAPTER_ID == CodexExportAdapter.adapter_id


def test_jsonl_roundtrip() -> None:
    raw = "\n".join(
        [
            '{"type":"message","role":"user","text":"hello"}',
            '{"type":"message","role":"assistant","text":"hi"}',
        ]
    )
    events, report = normalize_codex_export(raw, captured_at_ms=FIXED_MS)
    assert len(events) == 2
    assert report.accepted_event_ids[0] != report.accepted_event_ids[1]


def test_unsupported_version_is_rejected() -> None:
    with pytest.raises(CodexAdapterError, match="unsupported"):
        normalize_codex_export(
            {"version": "codex-export-99", "items": [{"type": "message", "role": "user", "text": "x"}]},
            captured_at_ms=FIXED_MS,
        )


def test_truncated_export_is_rejected() -> None:
    with pytest.raises(CodexAdapterError, match="truncated"):
        normalize_codex_export(
            {"version": "codex-export-1", "truncated": True, "items": []},
            captured_at_ms=FIXED_MS,
        )
    with pytest.raises(CodexAdapterError, match="truncated"):
        normalize_codex_export('{"version":"codex-export-1","items":[', captured_at_ms=FIXED_MS)


def test_hidden_chain_of_thought_is_rejected() -> None:
    payload = _export()
    payload["items"][1]["chain_of_thought"] = "secret"
    with pytest.raises(HandoffTrustError, match="hidden"):
        normalize_codex_export(payload, captured_at_ms=FIXED_MS)


def test_imported_authority_claim_is_rejected() -> None:
    payload = _export()
    payload["accepted"] = True
    with pytest.raises(HandoffTrustError, match="authority"):
        normalize_codex_export(payload, captured_at_ms=FIXED_MS)


def test_imported_invocations_are_not_executed() -> None:
    events, _report = normalize_codex_export(_export(), captured_at_ms=FIXED_MS)
    invocation = next(event for event in events if isinstance(event, ToolInvocationEvent))
    assert invocation.executed is False
    with pytest.raises(HandoffTrustError):
        ToolInvocationEvent(
            sequence=99,
            tool_name="read_file",
            arguments={"path": "x"},
            provenance=invocation.provenance,
            executed=True,
        )
