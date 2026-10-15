"""Closed provider retry hints; authored CLI failures never call a model."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

from ipfs_accelerate_py import llm_router
from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner as runner
from ipfs_accelerate_py.llm_allocation.observations import CallErrorKind
from test.api.test_supervisor_router_execution_contract import native_router


def event(value=15, *, key="retry_after_seconds", status=429, error_type="rate_limit_exceeded", nested=True):
    body = {"type": "error", "status": status, "error": {
        "type": error_type, "message": "rate limit reached", key: value,
    }}
    return {"type": "error", "message": json.dumps(body)} if nested else body


def failure(row):
    return llm_router._codex_cli_failure("codex exec failed", stdout=json.dumps(row), stderr="")


@pytest.mark.parametrize("key", ["retry_after_seconds", "retry_after", "resets_in_seconds", "reset_in_seconds"])
@pytest.mark.parametrize("value", [1, 0.25, 604800])
def test_native_error_relative_second_fields_survive_closed_projection(key, value):
    error = failure(event(value, key=key))
    assert error.codex_retry_after_seconds == value
    assert not hasattr(error, "retry_after_seconds")
    assert not hasattr(error, "status_code")
    assert runner._provider_failure_diagnostic(error, provider="codex_cli") == {
        "phase": "provider_invocation", "reason_code": "rate_limit", "http_status": 429,
        "retry_after_seconds": value,
    }


@pytest.mark.parametrize("value", [None, True, False, "12", "1.25", "Wed, 21 Oct 2026 07:28:00 GMT",
                                   0, -1, -0.25, float("nan"), float("inf"), float("-inf"), [], {}])
def test_malformed_relative_hint_never_becomes_wait_authority(value):
    error = failure(event(value))
    assert not hasattr(error, "codex_retry_after_seconds")
    assert not hasattr(error, "codex_retry_after_unbounded")
    observed = runner._provider_failure_diagnostic(error, provider="codex_cli")
    assert "retry_after_seconds" not in observed
    assert "retry_after_unbounded" not in observed


@pytest.mark.parametrize("value", [604801, 604800.5, 10**200])
def test_valid_long_reset_is_explicitly_unbounded_and_never_shortened(value):
    error = failure(event(value))
    assert not hasattr(error, "codex_retry_after_seconds")
    assert error.codex_retry_after_unbounded is True
    assert runner._provider_failure_diagnostic(error, provider="codex_cli") == {
        "phase": "provider_invocation", "reason_code": "rate_limit", "http_status": 429,
        "retry_after_unbounded": True,
    }


@pytest.mark.parametrize("row", [
    {"type": "item.completed", "item": {"type": "command_execution", "output": event()}},
    {"type": "item.completed", "item": {"type": "agent_message", "text": json.dumps(event())}},
    {"type": "turn.completed", "usage": {"retry_after_seconds": 12}},
    {"type": "tool_result", "error": event()},
    {"type": "error", "message": "rate limit: retry_after_seconds=12"},
    {"type": "error", "error": {"type": "not_a_provider_rate_error", "retry_after_seconds": 12}},
    {"type": "error", "status": 429, "error": {"type": "rate_limit_exceeded", "metadata": {"retry_after_seconds": 12}}},
    {"type": "error", "status": 429, "headers": {"retry-after": "12"}},
    {"type": "error", "status": 429, "reset_at": 1790000000},
])
def test_tool_model_freeform_and_arbitrary_nested_fields_are_not_hints(row):
    assert llm_router._codex_retry_after_hint(json.dumps(row)) == {}


@pytest.mark.parametrize("raw", [
    '{"type":"error","status":429,"retry_after_seconds":1,"retry_after_seconds":15}',
    '{"type":"error","status":429,"retry_after_seconds":15} {"extra":true}',
    '{"type":"error","status":429,"retry_after_seconds":1e309}',
    '{"type":"error","status":429,"retry_after_seconds":NaN}',
    '[{"type":"error","status":429,"retry_after_seconds":15}]',
    '{"type":"error","status":429,"retry_after_seconds":15,"message":"' + "x" * 16384 + '"}',
    "x" * 262145,
    "\n" * 1025,
    '{"type":[],"status":429,"retry_after_seconds":15}',
    '{"type":{},"status":429,"retry_after_seconds":15}',
    '{"type":"error","status":429,"retry_after_seconds":15,"message":"\ud800"}',
])
def test_malformed_oversized_duplicate_extra_or_nonfinite_json_is_ignored(raw):
    assert llm_router._codex_retry_after_hint(raw) == {}
    assert llm_router._codex_retry_after_hint(json.dumps({"type": "error", "message": raw})) == {}


@pytest.mark.parametrize("status,error_type,reason", [
    (401, "rate_limit_exceeded", "authentication"),
    (402, "rate_limit_exceeded", "billing"),
    (429, "invalid_api_key", "authentication"),
    (429, "authentication_error", "authentication"),
    (429, "insufficient_quota", "billing"),
    (429, "billing_error", "billing"),
    (403, "rate_limit_exceeded", "policy"),
])
@pytest.mark.parametrize("shape", ["nested", "direct", "turn_failed"])
def test_explicit_auth_or_billing_is_not_reclassified_by_rate_prose_or_hint(status, error_type, reason, shape):
    row = event(12, status=status, error_type=error_type, nested=shape == "nested")
    if shape == "turn_failed":
        row["type"] = "turn.failed"
    error = failure(row)
    observed = runner._provider_failure_diagnostic(error, provider="codex_cli")
    assert observed == {"phase": "provider_invocation", "reason_code": reason, "http_status": status}
    assert not hasattr(error, "codex_retry_after_seconds")


def test_longest_valid_relative_hint_wins_without_shortening():
    row = event(3, nested=False)
    row["reset_in_seconds"] = 11.5
    assert llm_router._codex_retry_after_hint(json.dumps(row)) == {"retry_after_seconds": 11.5}
    row["resets_in_seconds"] = 604801
    assert llm_router._codex_retry_after_hint(json.dumps(row)) == {"retry_after_unbounded": True}


def test_turn_failed_accepts_only_direct_error_or_one_nested_native_error():
    direct = {"type": "turn.failed", "error": {"type": "rate_limit_exceeded", "retry_after_seconds": 15}}
    nested = {"type": "turn.failed", "error": {"message": event()["message"]}}
    assert llm_router._codex_retry_after_hint(json.dumps(direct)) == {"retry_after_seconds": 15}
    assert llm_router._codex_retry_after_hint(json.dumps(nested)) == {"retry_after_seconds": 15}
    wrapped_twice = {"type": "turn.failed", "error": {"message": json.dumps(event())}}
    assert llm_router._codex_retry_after_hint(json.dumps(wrapped_twice)) == {}


@pytest.mark.parametrize("status,kind", [(401, "authentication"), (402, "billing"), (403, "policy"), (400, "invalid_request")])
def test_last_native_terminal_error_supersedes_earlier_transient_reset(status, kind):
    last = {"type": "error", "status": status, "error": {"type": "invalid_request_error", "message": "refused"}}
    stdout = json.dumps(event(12)) + "\n" + json.dumps(last)
    error = llm_router._codex_cli_failure("codex exec failed", stdout=stdout, stderr="")
    assert runner._provider_failure_diagnostic(error, provider="codex_cli") == {
        "phase": "provider_invocation", "reason_code": kind, "http_status": status}
    assert llm_router._codex_retry_after_hint(stdout) == {}


def test_tool_output_reset_beside_native_rate_error_is_not_promoted():
    stdout = "\n".join(json.dumps(row) for row in [
        {"type": "item.completed", "item": {"type": "command_execution", "output": event(45)}},
        {"type": "error", "status": 429, "error": {"type": "rate_limit_exceeded", "message": "rate limit"}},
        {"type": "item.completed", "item": {"type": "agent_message", "text": json.dumps(event(55))}},
    ])
    error = llm_router._codex_cli_failure("codex exec failed", stdout=stdout, stderr="")
    assert runner._provider_failure_diagnostic(error, provider="codex_cli") == {
        "phase": "provider_invocation", "reason_code": "rate_limit", "http_status": 429}


@pytest.mark.parametrize("status,kind", [(401, "authentication"), (402, "billing"), (403, "policy")])
def test_outer_explicit_hard_stop_cannot_be_overridden_by_nested_transient_error(status, kind):
    row = event(12)
    row["status"] = status
    observed = runner._provider_failure_diagnostic(failure(row), provider="codex_cli")
    assert observed == {"phase": "provider_invocation", "reason_code": kind, "http_status": status}
    assert llm_router._codex_retry_after_hint(json.dumps(row)) == {}


@pytest.mark.parametrize("delays,expected", [([300], {"retry_after_seconds": 300}),
    ([300, 12], {"retry_after_seconds": 300}), ([12, 300], {"retry_after_seconds": 300}),
    ([604801, 12], {"retry_after_unbounded": True})])
def test_generic_final_rate_wrapper_preserves_longest_typed_invocation_hint(delays, expected):
    rows = [event(delay) for delay in delays]
    rows.append({"type": "turn.failed", "error": {"message": "usage limit reached"}})
    stdout = "\n".join(json.dumps(row) for row in rows)
    error = llm_router._codex_cli_failure("codex exec failed", stdout=stdout, stderr="")
    assert llm_router._codex_retry_after_hint(stdout) == expected
    assert runner._provider_failure_diagnostic(error, provider="codex_cli") == {
        "phase": "provider_invocation", "reason_code": "rate_limit", **expected}


@pytest.mark.parametrize("tail,expected", [
    ({"type": "turn.failed", "error": {"type": "authentication_error", "message": "usage limit reached"}}, "authentication"),
    ({"type": "turn.failed", "error": {"type": "billing_error", "message": "usage limit reached"}}, "billing"),
    ({"type": "turn.failed", "error": {"message": "unclassified final failure"}}, "unknown"),
])
def test_final_non_rate_stop_discards_all_earlier_invocation_hints(tail, expected):
    stdout = "\n".join(json.dumps(row) for row in [event(604801), event(300), tail])
    error = llm_router._codex_cli_failure("codex exec failed", stdout=stdout, stderr="")
    assert llm_router._codex_retry_after_hint(stdout) == {}
    assert runner._provider_failure_diagnostic(error, provider="codex_cli") == {
        "phase": "provider_invocation", "reason_code": expected}


def test_generic_final_wrapper_cannot_supply_its_own_untyped_reset_fields():
    stdout = "\n".join(json.dumps(row) for row in [event(15),
        {"type": "turn.failed", "error": {"message": "usage limit reached; retry in 300 seconds", "retry_after_seconds": 300}}])
    assert llm_router._codex_retry_after_hint(stdout) == {"retry_after_seconds": 15}


@pytest.mark.parametrize("provider", ["grok_cli", "meta_ai"])
@pytest.mark.parametrize("value,expected", [(2.5, {"retry_after_seconds": 2.5}),
    (604801, {"retry_after_unbounded": True}), (True, {}), ("12", {}), (float("inf"), {})])
def test_other_provider_hint_requires_typed_attribute_and_rate_limit(provider, value, expected):
    error = llm_router.LLMRouterError("rate limit; retry after 99 seconds")
    assert runner._provider_failure_diagnostic(error, provider=provider) == {
        "phase": "provider_invocation", "reason_code": "rate_limit"}
    error.retry_after_seconds = value
    assert runner._provider_failure_diagnostic(error, provider=provider) == {
        "phase": "provider_invocation", "reason_code": "rate_limit", **expected}


@pytest.mark.parametrize("reason", [CallErrorKind.AUTHENTICATION, CallErrorKind.BILLING,
                                    CallErrorKind.UNKNOWN, CallErrorKind.INVALID_REQUEST])
def test_typed_retry_hint_does_not_override_non_rate_failure(reason):
    error = llm_router.LLMRouterError("rate limit explanatory prose")
    error.codex_error_kind = reason
    error.codex_retry_after_seconds = 15
    error.codex_retry_after_unbounded = True
    assert runner._provider_failure_diagnostic(error, provider="codex_cli") == {
        "phase": "provider_invocation", "reason_code": reason.value}


def test_explicit_unbounded_marker_takes_precedence_over_shorter_numeric_hint():
    error = llm_router.LLMRouterError("rate limit")
    error.retry_after_seconds = 1
    error.retry_after_unbounded = True
    assert runner._provider_failure_diagnostic(error, provider="grok_cli")["retry_after_unbounded"] is True
    assert "retry_after_seconds" not in runner._provider_failure_diagnostic(error, provider="grok_cli")


def test_optional_timing_property_failure_does_not_mask_native_error():
    class BrokenTiming(llm_router.LLMRouterError):
        @property
        def retry_after_seconds(self):
            raise ValueError("private-broken-timing-diagnostic")

    error = BrokenTiming("rate limit")
    assert runner._provider_failure_diagnostic(error, provider="grok_cli") == {
        "phase": "provider_invocation", "reason_code": "rate_limit"}


@pytest.mark.parametrize("purpose", ["coding", "planning"])
@pytest.mark.parametrize("generic_tail", [False, True])
def test_real_authored_native_rate_limit_has_one_dispatch_no_fallback_and_unknown_usage(native_router, capsys, purpose, generic_tail):
    workspace, cache, _ = native_router
    executable = Path(llm_router.shutil.which("codex"))
    native_event = event(2.5)
    rows = [native_event]
    if generic_tail:
        rows.append({"type": "turn.failed", "error": {"message": "usage limit reached"}})
    executable.write_text("#!" + sys.executable + "\nimport pathlib,sys\nsys.stdin.read()\n"
        + "with pathlib.Path('physical-dispatches.txt').open('a') as stream:stream.write('call\\n')\n"
        + "pathlib.Path('partial-effect.txt').write_text('retained candidate edit')\n"
        + "print(" + repr("\n".join(json.dumps(row) for row in rows)) + ")\nraise SystemExit(1)\n")
    with pytest.raises(llm_router.LLMRouterError):
        runner.run(prompt="private-authored-task-instruction", provider="codex_cli", model="gpt-6.1-sol",
                   timeout=15, max_output_tokens=128, purpose=purpose)
    captured = capsys.readouterr().out
    receipts = [json.loads(line) for line in captured.splitlines() if line.startswith("{")]
    receipt = next(row for row in receipts if row.get("schema") == "router-implementation-invocation@1")
    assert (workspace / "physical-dispatches.txt").read_text() == "call\n"
    assert (workspace / "partial-effect.txt").is_file()
    assert (cache.reads, cache.writes) == (0, 0)
    assert receipt["status"] == "failed"
    assert receipt["provider_failure"] == {"phase": "provider_invocation", "reason_code": "rate_limit",
        "retry_after_seconds": 2.5, **({} if generic_tail else {"http_status": 429})}
    assert receipt["usage"]["exit_code"] == 1
    assert "prompt_tokens" not in receipt["usage"]
    assert "completion_tokens" not in receipt["usage"]
    assert "private-authored-task-instruction" not in captured
    assert "rate limit reached" not in captured
    assert receipt["completion_authority"] is False
