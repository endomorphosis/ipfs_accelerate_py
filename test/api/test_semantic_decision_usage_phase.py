"""Log rereads preserve owner phases without discarding changed native data."""

from copy import deepcopy

from ipfs_accelerate_py.agent_supervisor.runtime.semantic_decision_usage import (
    INVOCATION_SCHEMA, collect_provider_usage_observation,
)


def _native(**changes):
    return {
        "schema": INVOCATION_SCHEMA, "invocation_id": "authored-invocation",
        "provider": "codex_cli", "model": "authored-model", "status": "provider_returned",
        "usage": {"prompt_tokens": 10, "completion_tokens": 2}, **changes,
    }


def test_owner_labeled_planning_receipt_deduplicates_raw_log_without_purpose():
    native = _native()
    prior = {"phase": "planning", **deepcopy(native)}
    report = {"provider_invocations": [prior]}
    assert collect_provider_usage_observation(report, deepcopy(native)) is True
    assert report["provider_invocations"] == [prior]
    assert report["provider_invocations"][0] is prior
    assert prior["phase"] == "planning" and "purpose" not in prior
    assert not any(row["phase"] == "coding" for row in report["provider_invocations"])


def test_explicit_native_planning_purpose_standalone_is_preserved_and_deduplicated():
    native = _native(purpose="planning")
    report = {}
    assert collect_provider_usage_observation(report, deepcopy(native)) is True
    assert collect_provider_usage_observation(report, deepcopy(native)) is True
    assert report["provider_invocations"] == [{"phase": "planning", **native}]


def test_changed_counters_same_identity_remain_observations_with_known_owner_phase():
    first = _native()
    prior = {"phase": "planning", **deepcopy(first)}
    second = _native(usage={"prompt_tokens": 12, "completion_tokens": 3})
    report = {"provider_invocations": [prior]}
    assert collect_provider_usage_observation(report, deepcopy(second)) is True
    assert collect_provider_usage_observation(report, deepcopy(second)) is True
    rows = report["provider_invocations"]
    assert rows == [prior, {"phase": "planning", **second}]
    assert [row["usage"]["prompt_tokens"] for row in rows] == [10, 12]
    assert {row["invocation_id"] for row in rows} == {"authored-invocation"}
    assert all(row["phase"] == "planning" for row in rows)


def test_changed_native_body_and_explicit_purpose_conflict_are_retained():
    first = _native(purpose="planning")
    second = _native(purpose="coding", status="failed")
    report = {}
    assert collect_provider_usage_observation(report, deepcopy(first)) is True
    assert collect_provider_usage_observation(report, deepcopy(second)) is True
    assert report["provider_invocations"] == [
        {"phase": "planning", **first}, {"phase": "coding", **second},
    ]
