"""Authored native counters test observation/coverage, never provider efficacy."""
from copy import deepcopy

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_decision_usage_projection as projection
from test.api.test_terminal_decision_usage import _native


@pytest.mark.parametrize("key", ["semantic_decision_dialogue_expected",
    "semantic_decision_owner_binding_expected", "semantic_decision_typed_nomination_expected"])
def test_non_boolean_expected_mode_cannot_authorize_complete_native_total(key):
    report = {"provider_invocations": [_native()], key: 1}
    original = deepcopy(report)
    measured = projection.measured_decision_usage(report)
    projection.finalize_usage_accounting(report)
    assert measured["known_provider_token_subtotal"] == 120
    assert measured["total_tokens"] is None
    assert measured["harbor_usage_exportable"] is False
    assert report["provider_usage_accounting"]["report_provider_total_tokens"] is None
    assert "invalid_expected_decision_mode" in {
        issue["reason"] for issue in measured["semantic_decision_coverage"]["issues"]}
    assert {key: value for key, value in report.items() if key != "provider_usage_accounting"} == original


def test_complete_grok_native_scope_remains_separate_from_decision_coverage():
    native = _native()
    native.update(provider="grok_cli", model="grok-4.7")
    native["native_rollout_usage"] = {
        "schema": "native-grok-final-usage@1", "task_complete_observed": True,
        "usage_complete_observed": True, "usage_available": True,
        "totals_scope": "native_final_envelope", "cache_included_in_input": False,
        "usage": {"input_tokens": 11, "cached_input_tokens": 7,
            "cache_write_input_tokens": 3, "output_tokens": 5, "total_tokens": 26},
        "provider_transcript": "authored-private-marker",
    }
    report = {"provider_invocations": [native], "semantic_decision_dialogue_expected": False}
    before = deepcopy(report)
    measured = projection.measured_decision_usage(report)
    projection.finalize_usage_accounting(report)
    assert measured["total_tokens"] is measured["known_provider_token_subtotal"] is None
    assert measured["harbor_usage_exportable"] is False
    observed = measured["other_provider_native_observations"]
    assert observed == report["provider_usage_accounting"]["other_provider_native_observations"]
    assert observed[0]["native_scope_complete_observed"] is True
    assert observed[0]["decision_usage_coverage_qualified"] is False
    assert observed[0]["cache_included_in_input"] is False
    assert [observed[0][key] for key in ("input_tokens", "cached_input_tokens",
        "cache_write_input_tokens", "output_tokens", "total_tokens")] == [11, 7, 3, 5, 26]
    assert "authored-private-marker" not in repr(measured)
    assert "authored-private-marker" not in repr(report["provider_usage_accounting"])
    assert {key: value for key, value in report.items() if key != "provider_usage_accounting"} == before
