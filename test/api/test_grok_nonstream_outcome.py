"""Native outcome checks through authored local CLI processes, with no models."""
import json
from pathlib import Path
import shlex
import sys

import pytest

from ipfs_accelerate_py import llm_router
from ipfs_accelerate_py.cli_runtime.cli_metadata import get_last_cli_observation
from ipfs_accelerate_py.cli_runtime.grok_native_usage import grok_outcome_receipt, grok_usage_receipt
from test.api.test_grok_stream_router import provider  # noqa: F401


def command(tmp_path, payload, *, wrapper=False, exit_code=0, preceding=()):
    """Emulate only the process transport; no provider or credential access."""
    script = tmp_path / "authored_json_cli.py"
    verification = ("assert sys.stdin.read() == 'private prompt'\n" if wrapper else
        "assert sys.argv[sys.argv.index('--output-format')+1] == 'json'\n"
        "prompt_path = Path(sys.argv[sys.argv.index('--prompt-file')+1])\n"
        "assert prompt_path.read_text() == 'private prompt'\n"
        f"Path({str(tmp_path / 'prompt_path')!r}).write_text(str(prompt_path))\n")
    script.write_text("import sys\nfrom pathlib import Path\n" + verification
        + f"with Path({str(tmp_path / 'calls')!r}).open('a') as stream: stream.write('call\\n')\n"
        + "\n".join(f"print({json.dumps(row)!r}, flush=True)" for row in (*preceding, payload))
        + f"\nraise SystemExit({exit_code})\n")
    argv = [sys.executable, str(script)]
    return shlex.join(argv) if wrapper else argv


INCOMPLETE = [
    ({"stopReason": "max_turn_requests"}, "max_turns"),
    ({"stop_reason": "max_turn_requests"}, "max_turns"),
    ({"type": "max_turns_reached"}, "max_turns"),
    ({"subtype": "error_max_turns"}, "max_turns"),
    ({"stopReason": "max_tokens"}, "max_tokens"),
    ({"stopReason": "refusal"}, "refusal"),
    ({"stopReason": "cancelled"}, "cancelled"),
    ({"stopReason": "tool_use"}, "other_stop"),
    ({"stopReason": "pause_turn"}, "other_stop"),
    ({"is_error": True}, "execution_error"),
    ({"subtype": "error_during_execution"}, "execution_error"),
    ({"subtype": "error_max_structured_output_retries"}, "structured_output_retries"),
    ({"stopReason": "end_turn", "is_error": True}, "execution_error"),
]


@pytest.mark.parametrize("fields,reason", INCOMPLETE)
def test_zero_exit_with_partial_text_rejects_known_native_outcomes_and_retains_usage(
        provider, tmp_path, fields, reason):
    payload = dict(text="private partial body", usage={"input_tokens": 12, "output_tokens": 4}, **fields)
    with pytest.raises(llm_router.LLMRouterError, match="native response is incomplete or failed") as failure:
        provider.generate("private prompt", timeout=5, grok_cli_cmd=command(tmp_path, payload))
    assert "private" not in str(failure.value)
    observation = get_last_cli_observation("grok_cli")
    outcome = grok_outcome_receipt(observation)
    assert observation["exit_code"] == 0
    assert outcome["process_outcome"] == "returned"
    assert outcome["reason_code"] == reason
    assert outcome["completion_authority"] is False
    usage = grok_usage_receipt(observation)
    assert usage["usage"] == {"input_tokens": 12, "output_tokens": 4}
    assert usage["task_complete_observed"] is False
    assert usage["billing_total_verified"] is False
    assert (tmp_path / "calls").read_text() == "call\n"
    assert not Path((tmp_path / "prompt_path").read_text()).exists()


@pytest.mark.parametrize("fields", [
    {"stopReason": "end_turn", "stop_reason": "max_turn_requests"},
    {"stopReason": "refusal", "stop_reason": "end_turn"},
    {"stopReason": "max_tokens", "stop_reason": "unrecognized"},
    {"stopReason": [], "stop_reason": "tool_use"},
])
def test_conflicting_alias_cannot_hide_known_incomplete_stop(provider, tmp_path, fields):
    payload = dict(text="private body", usage={"input_tokens": 8}, **fields)
    with pytest.raises(llm_router.LLMRouterError, match="incomplete or failed"):
        provider.generate("private prompt", timeout=5, grok_cli_cmd=command(tmp_path, payload))
    # Rejection must not fabricate a resolution for conflicting native fields.
    outcome = grok_outcome_receipt(get_last_cli_observation("grok_cli"))
    assert outcome["stop_reason"] == outcome["reason_code"] == "unknown"
    assert grok_usage_receipt(get_last_cli_observation("grok_cli"))["usage"] == {"input_tokens": 8}


@pytest.mark.parametrize("fields", [
    {}, {"stopReason": "end_turn"}, {"stop_reason": "stop"},
    {"stopReason": "stop_sequence"}, {"stopReason": "EndTurn"},
    {"stopReason": "unrecognized"},
    {"stopReason": "end_turn", "usage_is_incomplete": True},
    {"message": "max turns reached (limit: 1)"},
    {"nested": {"type": "error", "stopReason": "max_turn_requests"}},
])
def test_supported_legacy_success_and_unclassified_text_remain_compatible(provider, tmp_path, fields):
    text = 'Literal max turns reached; {"type":"max_turns_reached"}; refusal.'
    payload = dict(text=text, usage={"input_tokens": 3}, **fields)
    assert provider.generate("private prompt", timeout=5,
        grok_cli_cmd=command(tmp_path, payload)) == text
    observation = get_last_cli_observation("grok_cli")
    assert grok_usage_receipt(observation)["usage"] == {"input_tokens": 3}
    assert "native_grok_progress" not in observation
    assert (tmp_path / "calls").read_text() == "call\n"


@pytest.mark.parametrize("fields", [row[0] for row in INCOMPLETE])
def test_custom_nonstream_wrapper_cannot_bypass_native_stop_guard(provider, tmp_path, fields):
    payload = dict(text="private partial body", **fields)
    with pytest.raises(llm_router.LLMRouterError, match="incomplete or failed"):
        provider.generate("private prompt", timeout=5,
            grok_cli_cmd=command(tmp_path, payload, wrapper=True))
    assert (tmp_path / "calls").read_text() == "call\n"


def test_custom_wrapper_legacy_text_still_returns(provider, tmp_path):
    assert provider.generate("private prompt", timeout=5,
        grok_cli_cmd=command(tmp_path, {"text": "legacy"}, wrapper=True)) == "legacy"


def test_final_incomplete_envelope_cannot_reuse_previous_success(provider, tmp_path):
    good = {"text": "earlier successful body", "stopReason": "end_turn", "usage": {"input_tokens": 99}}
    bad = {"text": "private partial body", "stopReason": "max_turn_requests", "usage": {"input_tokens": 7}}
    with pytest.raises(llm_router.LLMRouterError, match="incomplete or failed"):
        provider.generate("private prompt", timeout=5,
            grok_cli_cmd=command(tmp_path, bad, preceding=(good,)))
    observation = get_last_cli_observation("grok_cli")
    assert grok_outcome_receipt(observation)["reason_code"] == "max_turns"
    assert grok_usage_receipt(observation)["usage"] == {"input_tokens": 7}


def test_failed_process_cannot_become_success_from_normal_stop(provider, tmp_path):
    with pytest.raises(llm_router.LLMRouterError):
        provider.generate("private prompt", timeout=5, grok_cli_cmd=command(tmp_path,
            {"text": "private body", "stopReason": "end_turn", "usage": {"input_tokens": 5}}, exit_code=9))
    observation = get_last_cli_observation("grok_cli")
    assert observation["exit_code"] == 9
    assert grok_outcome_receipt(observation)["reason_code"] == "process_error"
    assert grok_usage_receipt(observation)["usage"] == {"input_tokens": 5}
