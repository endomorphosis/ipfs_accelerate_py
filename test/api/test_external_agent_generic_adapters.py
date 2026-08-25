"""Deterministic tests for EAAEF-014 Gemini CLI and generic JSON/MCP adapters."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.handoff.adapters.gemini_cli import (
    ADAPTER_ID as GEMINI_CLI_ADAPTER_ID,
    GeminiCliAdapter,
    normalize_gemini_cli_export,
)
from ipfs_accelerate_py.agent_supervisor.handoff.adapters.generic import (
    GENERIC_JSON_ADAPTER_ID,
    GENERIC_JSONL_ADAPTER_ID,
    GENERIC_MCP_ADAPTER_ID,
    GenericJsonAdapter,
    GenericJsonlAdapter,
    GenericMcpAdapter,
    normalize_generic_json_export,
    normalize_generic_jsonl_export,
    normalize_generic_mcp_export,
)
from ipfs_accelerate_py.agent_supervisor.handoff.contracts import (
    ConversationEvent,
    EventKind,
    HandoffBounds,
    HandoffBoundsError,
    HandoffContractError,
    HandoffVersionError,
    PatchEvent,
    SourceFamily,
    ToolInvocationEvent,
    ToolResultEvent,
    decode_handoff_event,
    validate_event_sequence,
)


FIXED_MS = 1_700_000_000_000
ADAPTER_DIR = (
    Path(__file__).resolve().parents[2]
    / "ipfs_accelerate_py"
    / "agent_supervisor"
    / "handoff"
    / "adapters"
)


def _gemini_export() -> dict[str, object]:
    return {
        "family": "gemini_cli",
        "export_version": "gemini-cli-export-1",
        "contents": [
            {
                "role": "user",
                "parts": [{"text": "patch src/example.py"}],
                "client_tag": "turn-0",
            },
            {
                "role": "model",
                "parts": [
                    {"text": "applying the patch"},
                    {
                        "functionCall": {
                            "name": "apply_patch",
                            "args": {
                                "path": "src/example.py",
                                "diff": "--- a/src/example.py\n+++ b/src/example.py\n",
                            },
                            "id": "call-1",
                        }
                    },
                ],
            },
            {
                "role": "user",
                "parts": [
                    {
                        "functionResponse": {
                            "name": "apply_patch",
                            "id": "call-1",
                            "response": {"ok": True, "excerpt": "patched"},
                        }
                    }
                ],
            },
            {"role": "model", "parts": [{"text": "done"}]},
        ],
    }


def _mcp_export() -> dict[str, object]:
    return {
        "family": "generic_mcp",
        "version": "mcp-export-1",
        "messages": [
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "read_file",
                    "arguments": {"path": "src/example.py"},
                    "client_tag": "mcp-call",
                },
            },
            {
                "jsonrpc": "2.0",
                "id": 1,
                "result": {
                    "content": [{"type": "text", "text": "ok"}],
                    "isError": False,
                },
                "trace_id": "trace-9",
            },
        ],
    }


def _json_export() -> dict[str, object]:
    return {
        "family": "generic_json",
        "export_version": "generic-json-export-1",
        "events": [
            {
                "kind": "conversation",
                "role": "user",
                "text": "continue",
                "client_tag": "json-0",
            },
            {
                "kind": "tool_invocation",
                "tool_name": "read_file",
                "arguments": {"path": "src/example.py"},
                "executed": True,
                "call_id": "call-json",
            },
            {
                "kind": "tool_result",
                "tool_name": "read_file",
                "call_id": "call-json",
                "result": "ok",
                "success": True,
                "trusted_success": True,
            },
            {
                "kind": "patch",
                "paths": ["src/example.py"],
                "diff": "--- a/src/example.py\n+++ b/src/example.py\n",
                "claimed_applied": True,
            },
        ],
    }


def _jsonl_lines() -> list[str]:
    return [
        json.dumps(
            {
                "kind": "conversation",
                "role": "user",
                "text": "stream me",
                "client_tag": "jsonl-0",
            }
        ),
        json.dumps(
            {
                "kind": "tool_invocation",
                "tool_name": "read_file",
                "arguments": {"path": "src/example.py"},
                "call_id": "call-jsonl",
            }
        ),
        json.dumps(
            {
                "kind": "tool_result",
                "tool_name": "read_file",
                "call_id": "call-jsonl",
                "result": "ok",
                "success": True,
            }
        ),
    ]


def _assert_untrusted(result) -> None:
    for event in result.events:
        if isinstance(event, ToolInvocationEvent):
            assert event.executed is False
        if isinstance(event, ToolResultEvent):
            assert event.trusted_success is False
        if isinstance(event, PatchEvent):
            assert event.applied is False
    assert result.report.imported_invocations_not_executed is True
    assert result.report.truncated is False
    validate_event_sequence(result.events)
    for event in result.events:
        restored = decode_handoff_event(event.to_dict())
        assert restored.event_id == event.event_id


def test_gemini_cli_normalizes_contents_parts_and_function_calls() -> None:
    result = normalize_gemini_cli_export(_gemini_export(), captured_at_ms=FIXED_MS)
    assert result.adapter_id == "gemini-cli@1"
    assert result.adapter_id == GEMINI_CLI_ADAPTER_ID
    assert result.source_family is SourceFamily.GEMINI_CLI
    assert result.provenance.adapter_id == "gemini-cli@1"
    kinds = [event.kind for event in result.events]
    assert kinds[0] is EventKind.CONVERSATION
    assert kinds[1] is EventKind.CONVERSATION
    assert EventKind.TOOL_INVOCATION in kinds
    assert EventKind.TOOL_RESULT in kinds
    assert EventKind.PATCH in kinds
    user = result.events[0]
    assert isinstance(user, ConversationEvent)
    assert user.text == "patch src/example.py"
    invocation = next(
        event for event in result.events if isinstance(event, ToolInvocationEvent)
    )
    assert invocation.tool_name == "apply_patch"
    assert invocation.arguments["path"] == "src/example.py"
    result_event = next(
        event for event in result.events if isinstance(event, ToolResultEvent)
    )
    assert result_event.claimed_success is True
    assert result_event.invocation_event_id == invocation.event_id
    _assert_untrusted(result)


def test_gemini_cli_never_executes_or_trusts_success() -> None:
    payload = _gemini_export()
    result = GeminiCliAdapter().normalize(payload, captured_at_ms=FIXED_MS)
    invocation = next(
        event for event in result.events if isinstance(event, ToolInvocationEvent)
    )
    tool_result = next(
        event for event in result.events if isinstance(event, ToolResultEvent)
    )
    patch = next(event for event in result.events if isinstance(event, PatchEvent))
    assert invocation.executed is False
    assert tool_result.claimed_success is True
    assert tool_result.trusted_success is False
    assert patch.applied is False
    assert result.report.imported_success_claims_untrusted >= 1


def test_gemini_cli_retains_bounded_unknown_fields() -> None:
    result = normalize_gemini_cli_export(_gemini_export())
    user = result.events[0]
    assert user.residual_fields["client_tag"] == "turn-0"
    assert result.report.unknown_fields_retained >= 1


def test_gemini_cli_rejects_hidden_chain_of_thought() -> None:
    payload = _gemini_export()
    payload["contents"][1]["thinking"] = "secret scratchpad"
    with pytest.raises(HandoffContractError, match="hidden chain-of-thought"):
        normalize_gemini_cli_export(payload)
    thought_payload = {
        "contents": [
            {"role": "model", "parts": [{"text": "visible", "thought": True}]}
        ]
    }
    with pytest.raises(HandoffContractError, match="hidden chain-of-thought"):
        normalize_gemini_cli_export(thought_payload)


def test_gemini_cli_rejects_malformed_and_truncated_exports() -> None:
    with pytest.raises(HandoffContractError, match="malformed or truncated"):
        normalize_gemini_cli_export("{")
    with pytest.raises(HandoffContractError, match="malformed or truncated"):
        normalize_gemini_cli_export('{"contents":[{"role":"user"}')
    with pytest.raises(HandoffContractError, match="truncated"):
        normalize_gemini_cli_export(
            {"contents": [{"role": "user", "parts": [{"text": "hi"}], "truncated": True}]}
        )
    with pytest.raises(HandoffContractError, match="must contain contents or parts"):
        normalize_gemini_cli_export({"family": "gemini_cli", "notes": "empty"})
    with pytest.raises(HandoffVersionError):
        normalize_gemini_cli_export(
            {"export_version": "gemini-cli-export@2", "contents": []}
        )


def test_generic_mcp_normalizes_tools_call_frames() -> None:
    result = normalize_generic_mcp_export(_mcp_export(), captured_at_ms=FIXED_MS)
    assert result.adapter_id == GENERIC_MCP_ADAPTER_ID
    assert result.source_family is SourceFamily.GENERIC_MCP
    kinds = [event.kind for event in result.events]
    assert kinds == [EventKind.TOOL_INVOCATION, EventKind.TOOL_RESULT]
    invocation = result.events[0]
    tool_result = result.events[1]
    assert isinstance(invocation, ToolInvocationEvent)
    assert isinstance(tool_result, ToolResultEvent)
    assert invocation.tool_name == "read_file"
    assert invocation.executed is False
    assert invocation.residual_fields["client_tag"] == "mcp-call"
    assert tool_result.claimed_success is True
    assert tool_result.trusted_success is False
    assert tool_result.residual_fields["trace_id"] == "trace-9"
    assert result.report.unknown_fields_retained >= 2
    _assert_untrusted(result)


def test_generic_mcp_retains_unknown_fields_and_untrusted_results() -> None:
    result = GenericMcpAdapter().normalize(_mcp_export())
    assert result.report.imported_success_claims_untrusted == 1
    assert result.events[0].executed is False  # type: ignore[attr-defined]
    with pytest.raises(HandoffContractError, match="hidden chain-of-thought"):
        normalize_generic_mcp_export(
            {
                "family": "generic_mcp",
                "messages": [
                    {
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "tools/call",
                        "params": {"name": "read_file", "arguments": {}},
                        "scratchpad": "nope",
                    }
                ],
            }
        )


def test_generic_json_documented_export() -> None:
    result = normalize_generic_json_export(_json_export(), captured_at_ms=FIXED_MS)
    assert result.adapter_id == GENERIC_JSON_ADAPTER_ID
    assert result.source_family is SourceFamily.GENERIC_JSON
    kinds = [event.kind for event in result.events]
    assert kinds == [
        EventKind.CONVERSATION,
        EventKind.TOOL_INVOCATION,
        EventKind.TOOL_RESULT,
        EventKind.PATCH,
    ]
    conversation, invocation, tool_result, patch = result.events
    assert isinstance(conversation, ConversationEvent)
    assert conversation.residual_fields["client_tag"] == "json-0"
    assert isinstance(invocation, ToolInvocationEvent)
    assert invocation.executed is False
    assert isinstance(tool_result, ToolResultEvent)
    assert tool_result.claimed_success is True
    assert tool_result.trusted_success is False
    assert isinstance(patch, PatchEvent)
    assert patch.claimed_applied is True
    assert patch.applied is False
    assert patch.paths == ("src/example.py",)
    assert result.report.unknown_fields_retained >= 1
    _assert_untrusted(result)


def test_generic_jsonl_line_streaming() -> None:
    lines = _jsonl_lines()
    streamed = GenericJsonlAdapter().normalize_lines(iter(lines), captured_at_ms=FIXED_MS)
    joined = normalize_generic_jsonl_export("\n".join(lines) + "\n")
    assert streamed.adapter_id == GENERIC_JSONL_ADAPTER_ID
    assert streamed.source_family is SourceFamily.GENERIC_JSONL
    assert [event.kind for event in streamed.events] == [
        EventKind.CONVERSATION,
        EventKind.TOOL_INVOCATION,
        EventKind.TOOL_RESULT,
    ]
    assert streamed.events[0].residual_fields["client_tag"] == "jsonl-0"
    assert streamed.events[1].executed is False  # type: ignore[attr-defined]
    assert streamed.events[2].trusted_success is False  # type: ignore[attr-defined]
    assert streamed.events[2].claimed_success is True  # type: ignore[attr-defined]
    assert [event.kind for event in joined.events] == [
        event.kind for event in streamed.events
    ]
    assert joined.report.unknown_fields_retained >= 1
    _assert_untrusted(streamed)


def test_generic_jsonl_rejects_truncated_and_malformed_lines() -> None:
    truncated = (
        json.dumps({"kind": "conversation", "role": "user", "text": "ok"})
        + "\n"
        + '{"kind":"conversation","role":"assistant","text":"cut'
    )
    with pytest.raises(HandoffContractError, match="malformed or truncated"):
        normalize_generic_jsonl_export(truncated)
    with pytest.raises(HandoffContractError, match="malformed or truncated"):
        GenericJsonlAdapter().normalize_lines(["{"])
    with pytest.raises(HandoffContractError, match="line-delimited"):
        normalize_generic_jsonl_export({"kind": "conversation", "role": "user", "text": "x"})


def test_generic_families_reject_chain_of_thought() -> None:
    with pytest.raises(HandoffContractError, match="hidden chain-of-thought"):
        normalize_generic_json_export(
            {
                "events": [
                    {
                        "kind": "conversation",
                        "role": "assistant",
                        "text": "visible",
                        "chain_of_thought": "hidden",
                    }
                ]
            }
        )
    with pytest.raises(HandoffContractError, match="hidden chain-of-thought"):
        normalize_generic_jsonl_export(
            json.dumps(
                {
                    "kind": "conversation",
                    "role": "user",
                    "text": "hi",
                    "cot": "hidden",
                }
            )
            + "\n"
        )


def test_adapter_ids_and_source_families() -> None:
    gemini = GeminiCliAdapter()
    mcp = GenericMcpAdapter()
    documented = GenericJsonAdapter()
    jsonl = GenericJsonlAdapter()
    assert gemini.adapter_id == "gemini-cli@1"
    assert mcp.adapter_id == "generic-mcp@1"
    assert documented.adapter_id == "generic-json@1"
    assert jsonl.adapter_id == "generic-jsonl@1"
    assert gemini.source_family is SourceFamily.GEMINI_CLI
    assert mcp.source_family is SourceFamily.GENERIC_MCP
    assert documented.source_family is SourceFamily.GENERIC_JSON
    assert jsonl.source_family is SourceFamily.GENERIC_JSONL


def test_unknown_fields_exceeding_bound_are_rejected() -> None:
    too_many = {f"field_{index}": "x" for index in range(HandoffBounds().max_unknown_fields + 1)}
    event = {"kind": "conversation", "role": "user", "text": "hello", **too_many}
    with pytest.raises(HandoffBoundsError):
        normalize_generic_json_export({"events": [event]})


def test_family_mismatch_and_empty_payloads_fail_closed() -> None:
    with pytest.raises(HandoffContractError, match="does not match"):
        normalize_gemini_cli_export({"family": "generic_json", "contents": []})
    with pytest.raises(HandoffContractError, match="malformed or truncated"):
        normalize_generic_json_export("")
    with pytest.raises(HandoffContractError):
        normalize_generic_mcp_export({"family": "generic_mcp"})


def test_adapters_never_execute_imported_calls() -> None:
    forbidden = ("subprocess", "os.system", "Popen", "eval(", "exec(")
    for name in ("gemini_cli.py", "generic.py"):
        text = (ADAPTER_DIR / name).read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text
    result = normalize_generic_json_export(_json_export())
    assert all(
        not getattr(event, "executed", False) for event in result.events
    )
    assert all(
        not getattr(event, "trusted_success", False) for event in result.events
    )
