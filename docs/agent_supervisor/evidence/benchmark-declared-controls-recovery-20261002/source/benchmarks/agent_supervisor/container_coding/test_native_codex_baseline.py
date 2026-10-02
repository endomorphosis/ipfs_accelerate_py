"""Receipt accounting preserves native cumulative totals and missing fields."""

import json
import os
from pathlib import Path
import shutil

import pytest

from benchmarks.agent_supervisor.container_coding.native_codex_baseline import (
    TOKEN_FIELDS,
    config_for,
    rollout_usage,
    _task_hashes,
)


@pytest.mark.parametrize("kind", ["linked_directory", "broken_link", "fifo", "linked_root"])
def test_task_inventory_cannot_silently_omit_nonregular_inputs(tmp_path, kind):
    task = tmp_path / "task"
    task.mkdir()
    (task / "instruction.md").write_text("Fix the defect.\n")
    (task / "task.toml").write_text("version = 1\n")
    (task / "environment").mkdir()
    (task / "tests").mkdir()
    (task / "tests/test.sh").write_text("exit 0\n")
    assert set(_task_hashes(task)) == {"instruction.md", "task.toml", "tests/test.sh"}
    if kind == "linked_directory":
        (task / "environment/linked").symlink_to(task / "tests", target_is_directory=True)
    elif kind == "broken_link":
        (task / "environment/broken").symlink_to(tmp_path / "missing")
    elif kind == "fifo":
        os.mkfifo(task / "environment/pipe")
    else:
        link = tmp_path / "linked-task"
        link.symlink_to(task, target_is_directory=True)
        task = link
    with pytest.raises(ValueError):
        _task_hashes(task)


def write_session(agent, identity, totals):
    path = agent / "sessions" / identity / ("rollout-" + identity + ".jsonl")
    path.parent.mkdir(parents=True)
    rows = [
        {"type": "session_meta", "payload": {"id": identity, "cli_version": "0.158.0"}},
        {"type": "turn_context", "payload": {"model": "gpt-5.6-sol", "effort": "high"}},
    ]
    rows += [
        {
            "type": "event_msg",
            "payload": {"type": "token_count", "info": {"total_token_usage": value}},
        }
        for value in totals
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return path


def append_records(path, *rows):
    with path.open("a") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")


COMPLETE = {"type": "event_msg", "payload": {"type": "task_complete"}}


def test_profile_uses_native_agent_and_exact_independent_bounds(tmp_path):
    config = config_for(Path("/dataset"), tmp_path)
    assert config["tasks"] == [{"path": "/dataset/fix-code-vulnerability"}]
    assert config["agents"][0]["name"] == "codex"
    assert config["agents"][0]["model_name"] == "gpt-5.6-sol"
    assert config["agents"][0]["kwargs"] == {"version": "0.158.0", "reasoning_effort": "high"}
    assert (
        config["agents"][0]["override_timeout_sec"] == config["agents"][0]["max_timeout_sec"] == 300
    )
    assert config["n_attempts"] == config["n_concurrent_trials"] == 1
    assert config["retry"]["max_retries"] == 0
    assert config["verifier"] == {"disable": False}


def test_no_native_transcript_means_unknown_tokens_not_zero(tmp_path):
    result = rollout_usage(tmp_path)
    assert result["sessions"] == []
    assert result["usage"] == {name: None for name in TOKEN_FIELDS}
    assert result["reported_cost_usd"] is None
    assert result["task_complete_observed"] is False
    assert result["billing_total_verified"] is False


def test_cumulative_updates_not_summed_and_explicit_zero_preserved(tmp_path):
    write_session(
        tmp_path,
        "a",
        [
            {"input_tokens": 10},
            {
                "input_tokens": 25,
                "cached_input_tokens": 0,
                "output_tokens": 7,
                "reasoning_output_tokens": 3,
                "total_tokens": 32,
            },
        ],
    )
    result = rollout_usage(tmp_path)
    assert result["usage"]["input_tokens"] == 25
    assert result["usage"]["cached_input_tokens"] == 0
    assert result["usage"]["cache_write_input_tokens"] is None
    assert result["usage"]["reasoning_output_tokens"] == 3
    assert result["usage"]["total_tokens"] == 32
    assert result["sessions"][0]["token_count_events"] == 2
    assert result["sessions"][0]["observed_reasoning_efforts"] == ["high"]
    assert result["task_complete_observed"] is False


def test_exact_native_completion_marks_last_counters_without_billing_claim(tmp_path):
    path = write_session(tmp_path, "a", [{"input_tokens": 5, "total_tokens": 7},
                                           {"input_tokens": 8, "total_tokens": 11}])
    append_records(path, COMPLETE)
    result = rollout_usage(tmp_path)
    assert result["usage"]["total_tokens"] == 11
    assert result["task_complete_observed"] is True
    assert result["sessions"][0]["task_complete_event_observed"] is True
    assert result["sessions"][0]["task_complete_observed"] is True
    assert result["sessions"][0]["malformed_lines"] == 0
    assert result["billing_total_verified"] is False
    assert result["sessions"][0]["billing_total_verified"] is False


def test_model_text_reward_and_exit_are_not_completion_events(tmp_path):
    path = write_session(tmp_path, "a", [{"total_tokens": 11}])
    append_records(path,
        {"type": "response_item", "payload": {"type": "message", "role": "assistant",
            "content": [{"type": "output_text", "text": json.dumps(COMPLETE)}]}},
        {"type": "response_item", "payload": {"type": "task_complete"}},
        {"type": "event_msg", "payload": {"type": "agent_message", "message": "task_complete"}},
        {"type": "event_msg", "payload": {"type": "agent_exit", "exit_code": 0}},
        {"type": "event_msg", "payload": {"type": "reward", "reward": 1}})
    result = rollout_usage(tmp_path)
    assert result["usage"]["total_tokens"] == 11
    assert result["task_complete_observed"] is False
    assert result["sessions"][0]["task_complete_event_observed"] is False


@pytest.mark.parametrize("bad", ['{"type":', '[]', 'null', '{"type":"event_msg","payload":null}'])
@pytest.mark.parametrize("after_completion", [False, True])
def test_malformed_records_preserve_lower_bound_but_invalidate_completion(tmp_path, bad, after_completion):
    path = write_session(tmp_path, "a", [{"total_tokens": 11}])
    if after_completion:
        append_records(path, COMPLETE)
    with path.open("a") as stream:
        stream.write(bad + "\n")
    if not after_completion:
        append_records(path, COMPLETE)
    result = rollout_usage(tmp_path)
    assert result["usage"]["total_tokens"] == 11
    assert result["sessions"][0]["malformed_lines"] == 1
    assert result["sessions"][0]["task_complete_event_observed"] is True
    assert result["sessions"][0]["task_complete_observed"] is False
    assert result["task_complete_observed"] is False


@pytest.mark.parametrize("event", [
    {"type": "turn_context", "payload": {"model": "pinned"}},
    *[{"type": "event_msg", "payload": {"type": name}} for name in
      ("task_started", "turn_started", "turn_aborted", "task_aborted", "user_message")],
    {"type": "event_msg", "payload": {"type": "token_count",
        "info": {"total_token_usage": {"total_tokens": 17}}}},
])
def test_work_after_complete_requires_a_new_completion_event(tmp_path, event):
    path = write_session(tmp_path, "a", [{"total_tokens": 11}])
    append_records(path, COMPLETE, event)
    observed = rollout_usage(tmp_path)
    assert observed["task_complete_observed"] is False
    assert observed["sessions"][0]["task_complete_event_observed"] is True
    assert observed["usage"]["total_tokens"] == (17 if event["payload"].get("type") == "token_count" else 11)
    append_records(path, COMPLETE)
    assert rollout_usage(tmp_path)["task_complete_observed"] is True


def test_every_identified_session_must_complete(tmp_path):
    first = write_session(tmp_path, "a", [{"total_tokens": 11}])
    second = write_session(tmp_path, "b", [{"total_tokens": 7}])
    append_records(first, COMPLETE)
    result = rollout_usage(tmp_path)
    assert result["usage"]["total_tokens"] == 18
    assert result["task_complete_observed"] is False
    append_records(second, COMPLETE)
    assert rollout_usage(tmp_path)["task_complete_observed"] is True
    first.write_text("\n".join(first.read_text().splitlines()[1:]) + "\n")
    result = rollout_usage(tmp_path)
    assert result["usage"]["total_tokens"] == 18
    assert result["task_complete_observed"] is False
    assert any(row["session_id"].startswith("unidentified:") for row in result["sessions"])


def test_synthesized_native_completion_flows_into_comparison(tmp_path):
    from benchmarks.agent_supervisor.container_coding.benchmark_comparison import BASELINE, collect
    path = write_session(tmp_path, "a", [{"input_tokens": 8, "total_tokens": 11}])
    receipt = tmp_path / "receipt.json"
    for complete in (False, True):
        if complete:
            append_records(path, COMPLETE)
        receipt.write_text(json.dumps({"schema": BASELINE, "task": "authored-task", "trial_count": 1,
            "trials": [{"trial": "authored-trial", "reward": {"reward": 1},
                        "raw_usage": rollout_usage(tmp_path)}]}))
        row = collect(baseline=receipt, supervisors=[])["rows"][0]
        assert row["native_usage_complete_observed"] is complete
        assert row["tokens"]["total_tokens"] == (11 if complete else None)
        assert row["known_token_subtotals"]["total_tokens"] == 11


def test_multi_session_partial_field_remains_unknown(tmp_path):
    write_session(tmp_path, "a", [{"input_tokens": 25, "output_tokens": 7}])
    write_session(tmp_path, "b", [{"input_tokens": 8, "output_tokens": True}])
    result = rollout_usage(tmp_path)
    assert result["usage"]["input_tokens"] == 33
    assert result["usage"]["output_tokens"] is None


def test_identical_copied_session_is_counted_once(tmp_path):
    path = write_session(tmp_path, "a", [{"input_tokens": 25}])
    shutil.copyfile(path, path.with_name("rollout-copy.jsonl"))
    assert rollout_usage(tmp_path)["usage"]["input_tokens"] == 25
    assert len(rollout_usage(tmp_path)["sessions"]) == 1


def test_conflicting_session_identity_is_not_silently_double_counted(tmp_path):
    path = write_session(tmp_path, "a", [{"input_tokens": 25}])
    path.with_name("rollout-copy.jsonl").write_text(path.read_text().replace("25", "35"))
    with pytest.raises(ValueError, match="same native session identity"):
        rollout_usage(tmp_path)


def test_known_nonsecret_flag_recovery_preserves_raw_files(tmp_path):
    path = write_session(tmp_path, 'session-1', [{'input_tokens': 123, 'cached_input_tokens': 100, 'output_tokens': 11, 'total_tokens': 134}])
    path.write_text(path.read_text().replace('1', '[REDACTED]'))
    original = path.read_bytes()
    result = rollout_usage(tmp_path, known_redacted_literal='1')
    assert result['usage']['input_tokens'] == 123
    assert result['usage']['cached_input_tokens'] == 100
    assert result['usage']['output_tokens'] == 11
    assert result['usage']['total_tokens'] == 134
    assert result['sessions'][0]['session_id'] == 'session-1'
    assert result['sessions'][0]['cli_version'] == '0.158.0'
    assert path.read_bytes() == original
    with pytest.raises(ValueError, match='unsupported'):
        rollout_usage(tmp_path, known_redacted_literal='credential')
