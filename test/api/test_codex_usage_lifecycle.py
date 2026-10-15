"""Native lifecycle boundaries are observations, not billing or task authority."""

from __future__ import annotations

import json

import pytest
from ipfs_accelerate_py.agent_supervisor.runtime.codex_usage_receipt import (
    USAGE_LIFECYCLE_COMPLETION_BASIS,
    recover_codex_usage,
)

THREAD = "authored-thread-lifecycle-123456"
COUNTERS = {
    "input_tokens": 50,
    "cached_input_tokens": 30,
    "output_tokens": 5,
    "reasoning_output_tokens": 2,
    "total_tokens": 55,
}


def event(name, **payload):
    return {"type": "event_msg", "payload": {"type": name, **payload}}


def count(usage=None):
    return event("token_count", info={"total_token_usage": COUNTERS if usage is None else usage})


def observe(tmp_path, records):
    home = tmp_path / "authored-codex-home"
    folder = home / "sessions/2026/10/12"
    folder.mkdir(parents=True)
    workspace = tmp_path / "authored-workspace"
    workspace.mkdir()
    rows = [{"type": "session_meta", "payload": {"id": THREAD, "cwd": str(workspace)}}]
    rows.extend(records)
    raw = "\n".join(row if isinstance(row, str) else json.dumps(row) for row in rows)
    (folder / f"rollout-{THREAD}.jsonl").write_text(raw + "\n")
    result = recover_codex_usage(home=home, thread_id=THREAD, workspace=workspace)
    assert result is not None
    assert result["completion_basis"] == USAGE_LIFECYCLE_COMPLETION_BASIS
    assert result["billing_total_verified"] is False and result["dollar_cost"] is None
    assert result["cache_included_in_input"] is True
    # The returned receipt contains no request/model/tool text or local paths.
    assert str(workspace) not in json.dumps(result)
    return result


def test_completion_closes_latest_usage_without_adding_cache_or_reasoning(tmp_path):
    result = observe(tmp_path, [count(), event("task_complete")])
    assert result["usage"] == COUNTERS
    assert result["usage_available"] is True
    assert result["observed_token_count_records"] == 1
    assert result["task_complete_event_observed"] is True
    assert result["task_complete_observed"] is True
    assert result["usage_lifecycle_complete"] is True
    assert result["lifecycle_malformed_records"] == 0


@pytest.mark.parametrize("resumed", [
    event("token_count", info={"total_token_usage": {**COUNTERS, "input_tokens": 60, "total_tokens": 65}}),
    {"type": "turn_context", "payload": {}},
    event("task_started"), event("turn_started"), event("turn_aborted"),
    event("task_aborted"), event("user_message"),
])
def test_later_work_invalidates_earlier_completion(tmp_path, resumed):
    result = observe(tmp_path, [count(), event("task_complete"), resumed])
    assert result["usage_available"] is True
    assert result["task_complete_event_observed"] is True
    assert result["task_complete_observed"] is False
    assert result["usage_lifecycle_complete"] is False
    assert result["lifecycle_malformed_records"] == 0


def test_resumed_usage_requires_its_own_later_completion(tmp_path):
    latest = {**COUNTERS, "input_tokens": 80, "output_tokens": 8, "total_tokens": 88}
    result = observe(tmp_path, [count(), event("task_complete"), count(latest), event("task_complete")])
    assert result["usage"] == latest
    assert result["observed_token_count_records"] == 2
    assert result["usage_lifecycle_complete"] is True


@pytest.mark.parametrize("malformed", [
    "{incomplete", "null", "[]",
    {"type": "event_msg", "payload": []},
    event([]), event(None),
    event("token_count", info=None),
    event("token_count", info={}),
    event("token_count", info={"total_token_usage": []}),
    count({}),
    count({"input_tokens": True, "output_tokens": 5}),
    count({"input_tokens": -1, "output_tokens": 5}),
])
def test_malformed_records_cannot_become_complete_after_a_later_marker(tmp_path, malformed):
    result = observe(tmp_path, [count(), malformed, event("task_complete")])
    assert result["usage_available"] is True
    assert result["task_complete_event_observed"] is True
    assert result["task_complete_observed"] is False
    assert result["usage_lifecycle_complete"] is False
    assert result["lifecycle_malformed_records"] == 1
    assert result["tool_diagnostics"]["tool_event_counts_complete"] is False


def test_regressing_counters_retain_reported_observation_with_incomplete_lifecycle(tmp_path):
    latest = {**COUNTERS, "input_tokens": 40, "total_tokens": 45}
    result = observe(tmp_path, [count(), count(latest), event("task_complete")])
    assert result["usage"] == latest
    assert result["observed_token_count_records"] == 2
    assert result["lifecycle_malformed_records"] == 1
    assert result["usage_lifecycle_complete"] is False


@pytest.mark.parametrize("latest", [
    {**COUNTERS, "input_tokens": 70, "output_tokens": 4, "total_tokens": 74},
    {**COUNTERS, "input_tokens": 70, "cached_input_tokens": 29, "total_tokens": 75},
    {**COUNTERS, "input_tokens": 70, "total_tokens": 54},
])
def test_partial_snapshots_cannot_hide_category_regressions(tmp_path, latest):
    result = observe(tmp_path, [count(), count({"input_tokens": 60}), count(latest), event("task_complete")])
    assert result["usage"] == latest
    assert result["observed_token_count_records"] == 3
    assert result["lifecycle_malformed_records"] == 1
    assert result["usage_lifecycle_complete"] is False


def test_partial_snapshots_do_not_invent_returned_categories_or_prevent_valid_growth(tmp_path):
    latest = {**COUNTERS, "input_tokens": 70, "output_tokens": 8, "total_tokens": 78}
    result = observe(tmp_path, [count(), count({"input_tokens": 60}), count(latest), event("task_complete")])
    assert result["usage"] == latest
    assert result["observed_token_count_records"] == 3
    assert result["lifecycle_malformed_records"] == 0
    assert result["usage_lifecycle_complete"] is True


def test_completion_without_usage_never_invents_zero_totals(tmp_path):
    result = observe(tmp_path, [event("task_complete")])
    assert result["task_complete_observed"] is True
    assert result["usage_lifecycle_complete"] is False
    assert result["usage_available"] is False
    assert result["usage"] == {} and result["observed_token_count_records"] == 0


def test_missing_categories_remain_missing_with_complete_lifecycle(tmp_path):
    result = observe(tmp_path, [count({"input_tokens": 50}), event("task_complete")])
    assert result["usage"] == {"input_tokens": 50}
    assert result["usage_lifecycle_complete"] is True
    assert "output_tokens" not in result["usage"] and "total_tokens" not in result["usage"]
