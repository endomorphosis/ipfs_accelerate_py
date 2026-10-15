"""Retained input variants stay distinct without inventing matched savings."""
import json

import pytest

from benchmarks.agent_supervisor.container_coding import benchmark_comparison as comparison
from benchmarks.agent_supervisor.container_coding.test_benchmark_comparison import (
    _baseline, _invocation, _receipt,
)


def variant_receipt(tmp_path, name, *, options=None, reward=1, invocation=None):
    options = {} if options is None else options
    observation = {
        "prompt_comparison_options": options,
        "prompt_comparison_options_valid": True,
        "prompt_comparison_config_unchanged": True,
    }
    path = _receipt(tmp_path, name + ".json", comparison.SUPERVISOR, [{
        "trial": name, "reward": {"reward": reward}, **observation,
        "supervisor": {"task_completed": reward == 1,
                       "provider_invocations": [invocation or _invocation(name)]},
    }], "full")
    value = json.loads(path.read_text())
    value.update(observation)
    path.write_text(json.dumps(value))
    return path


def test_two_full_variants_need_no_unrelated_native_harness(tmp_path):
    native = variant_receipt(tmp_path, "native", options={"semantic_native_input": True})
    packet = variant_receipt(tmp_path, "packet", options={
        "semantic_decision_nomination_source": "/owner/pinned.json"})
    result = comparison.collect(supervisors=[native, packet])
    assert result["native_harness_baseline_supplied"] is False
    assert result["retained_trial_count"] == 2
    assert [row["configured_input_variant"] for row in result["rows"]] == [
        "native-input", "decision-packet"]
    assert all(row["comparison_profile_matches"] for row in result["rows"])
    assert all(row["input_variant_is_provider_wire_evidence"] is False for row in result["rows"])
    totals = result["configured_input_variant_totals"]
    assert set(totals) == {"full/native-input", "full/decision-packet"}
    assert all(row["tokens"]["total_tokens"] == 22 for row in totals.values())
    assert result["benchmark_advantage_claimed"] is result["causal_claim"] is False
    rendered = comparison.markdown(result)
    assert "full/native-input" in rendered and "full/decision-packet" in rendered
    assert "MUST NOT APPEAR" not in json.dumps(result)


def test_default_is_automatic_codec_and_native_harness_stays_separate(tmp_path):
    auto = variant_receipt(tmp_path, "automatic")
    result = comparison.collect(baseline=_baseline(tmp_path), supervisors=[auto])
    assert result["native_harness_baseline_supplied"] is True
    assert [row["configured_input_variant"] for row in result["rows"]] == [
        "native-harness", "automatic-codec"]
    assert result["rows"][1]["label"] == comparison.LABELS["full"]
    assert result["arm_totals"]["native-codex"]["tokens"]["total_tokens"] == 110


@pytest.mark.parametrize("mutation", ["missing", "changed", "invalid", "conflict", "wrong_arm"])
def test_unverified_or_conflicting_configuration_is_not_a_native_control(tmp_path, mutation):
    path = variant_receipt(tmp_path, "unverified", options={"semantic_native_input": True})
    value = json.loads(path.read_text())
    if mutation == "missing":
        value.pop("prompt_comparison_options_valid")
    elif mutation == "changed":
        value["trials"][0]["prompt_comparison_options"] = {}
    elif mutation == "invalid":
        value["prompt_comparison_options_valid"] = False
    elif mutation == "conflict":
        value["prompt_comparison_options"]["semantic_decision_policy_artifact"] = "/owner/policy"
        value["trials"][0]["prompt_comparison_options"] = value["prompt_comparison_options"]
    else:
        value["arm"] = "no-index"
    path.write_text(json.dumps(value))
    row = comparison.collect(supervisors=[path])["rows"][0]
    assert row["configured_input_variant"] == "unverified"
    assert row["tokens"]["total_tokens"] == 22


def test_shared_session_between_variants_withholds_variant_totals(tmp_path):
    call = _invocation("shared")
    native = variant_receipt(tmp_path, "native", options={"semantic_native_input": True}, invocation=call)
    packet = variant_receipt(tmp_path, "packet", options={
        "semantic_decision_policy_artifact": "/owner/policy"}, invocation=call)
    result = comparison.collect(supervisors=[native, packet])
    assert result["input_variant_usage_guard"]["native_session_overlap_detected"] is True
    assert all(row["tokens"]["total_tokens"] == 22 for row in result["rows"])
    for totals in result["configured_input_variant_totals"].values():
        assert totals["tokens"]["total_tokens"] is None
        assert totals["known_token_subtotals"]["total_tokens"] is None
        assert totals["provider_calls"] is None


def test_failed_and_interrupted_variant_keeps_observed_costs(tmp_path):
    call = _invocation("interrupted")
    call["native_rollout_usage"]["usage_lifecycle_complete"] = False
    call["native_rollout_usage"]["task_complete_observed"] = False
    path = variant_receipt(tmp_path, "failed", options={"semantic_native_input": True},
                           reward=0, invocation=call)
    result = comparison.collect(supervisors=[path])
    totals = result["configured_input_variant_totals"]["full/native-input"]
    assert totals["observed_official_passes"] == 0
    assert totals["tokens"]["total_tokens"] is None
    assert totals["known_token_subtotals"]["total_tokens"] == 22
    assert result["unsuccessful_attempt_costs"]["full"]["trials"] == 1


def test_empty_population_refuses(tmp_path):
    with pytest.raises(ValueError, match="at least one"):
        comparison.collect(supervisors=[])
