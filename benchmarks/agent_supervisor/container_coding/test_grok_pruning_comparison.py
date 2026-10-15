"""Authored paired declarations; no provider, container, or retained run input."""
from copy import deepcopy
import json

import pytest

from ipfs_accelerate_py.cli_runtime import grok_pruning as pruning
from benchmarks.agent_supervisor.container_coding import benchmark_comparison as comparison
from benchmarks.agent_supervisor.container_coding import benchmark_controls as controls
from benchmarks.agent_supervisor.container_coding import full_supervisor_benchmark as benchmark
from benchmarks.agent_supervisor.container_coding.benchmark_provider_profile import GROK_PROFILE, resolve_provider_profile
from benchmarks.agent_supervisor.container_coding.test_benchmark_comparison import _grok_invocation, _baseline, _receipt


OFF, ON = "grok-pruning-off@1", "grok-pruning-on@1"
IDENTITY = {"task": benchmark.TASK, **{key: value for key, value in resolve_provider_profile(GROK_PROFILE).items()
    if key in {"model", "reasoning_effort", "cli_version"}}}


def preflight(profile):
    value = {key: [] for key in ("agents", "hooks", "lspServers", "marketplaces", "mcpServers",
        "plugins", "projectInstructions", "skills")}
    value.update(configSources={"layers": []}, externalCompat={}, loginPolicy={}, permissions={},
        channel="authored", cwd="/authored", grokVersion="authored", projectRoot=None, projectTrusted=False)
    effective = pruning.observe_grok_pruning_inspection(profile, stdout=json.dumps(value).encode(),
        stderr=b"", returncode=0)
    return pruning.build_grok_pruning_preflight(profile,
        file_observation=pruning._expected_file_observation(profile, 0, 0), effective_observation=effective)


def reseal(receipt, config):
    previous = receipt.get("comparison_controls", {}).get("declared", {})
    declared = controls._control_record(config,
        task_input_sha256=previous.get("task_input_sha256", {"instruction.md": "a" * 64}),
        **{key: receipt[key] for key in IDENTITY})
    receipt["comparison_controls"] = {"status": "observed", "reason": None,
        "declared": declared, "configuration_unchanged": True}


@pytest.fixture
def pair(tmp_path):
    sides = []
    for profile, name in ((OFF, "off"), (ON, "on")):
        config = benchmark.config_for(tmp_path / "dataset", tmp_path / name, tmp_path / "archive", "full",
            provider_profile=GROK_PROFILE, planning_mode="declared-contract@1", grok_pruning_profile=profile)
        call = _grok_invocation(name)
        call.update(purpose="coding", model=IDENTITY["model"], reasoning_effort=IDENTITY["reasoning_effort"],
            provider_invocation_policy={"grok_pruning": preflight(profile)}, prompt="PRIVATE-PROMPT")
        trial = {"trial": name, "task": benchmark.TASK, "exact_trial_task_matches": True,
            "reward": None, "durations_seconds": {"verifier": None},
            "agent_context": {"metadata": {"runtime_archive_sha256": "b" * 64}},
            "supervisor": {"grok_pruning_profile": profile, "task_completed": None,
                "provider_invocations": [call]}}
        receipt = {"schema": comparison.SUPERVISOR, "arm": "full", **IDENTITY,
            "grok_pruning_profile": profile, "original_task_inputs_unchanged": True,
            "complete_single_trial_receipt": True, "native_job_result_present": True,
            "grok_pruning_config_unchanged": True, "grok_pruning_runtime_declared_matches": True,
            "grok_pruning_runtime_policy_matches": True, "trial_count": 1, "trials": [trial]}
        reseal(receipt, config)
        sides.append([receipt, config])
    return sides


def compare(pair):
    return comparison.compare_grok_pruning_pair(pair[0][0], pair[1][0],
        off_config=pair[0][1], on_config=pair[1][1])


def call_for(pair, index=1):
    return pair[index][0]["trials"][0]["supervisor"]["provider_invocations"][0]


def test_configuration_eligibility_does_not_invent_correctness_usage_or_effectiveness(pair):
    original = deepcopy(pair)
    result = compare(pair)
    assert result["eligible_configuration_comparison"] is True
    assert result["ineligible_reasons"] == []
    assert result["correctness_observations_complete"] is False
    assert result["native_usage_observations_complete"] is False
    assert result["effective_settings_verified"] is result["config_file_consumption_verified"] is False
    assert result["native_behavior"] == "unknown"
    assert result["billing_total_verified"] is result["token_savings_established"] is False
    for side in result["profiles"].values():
        assert side["usage"]["tokens"]["total_tokens"] is None
        assert side["usage"]["known_token_subtotals"]["total_tokens"] == 42
    assert result["profiles"]["off"]["configuration_sha256"] != result["profiles"]["on"]["configuration_sha256"]
    assert result["profiles"]["off"]["normalized_configuration_sha256"] == result["profiles"]["on"]["normalized_configuration_sha256"]
    assert "PRIVATE-PROMPT" not in json.dumps(result)
    assert pair == original


def test_observed_matching_success_and_complete_native_usage_still_are_not_savings(pair):
    for index, (receipt, _) in enumerate(pair):
        receipt["trials"][0].update(reward={"reward": 1}, durations_seconds={"verifier": index + 1.0})
        receipt["trials"][0]["supervisor"]["task_completed"] = True
        call_for(pair, index)["native_rollout_usage"]["usage_complete_observed"] = True
    result = compare(pair)
    assert result["eligible_configuration_comparison"] is True
    assert result["correctness_observations_complete"] is result["native_usage_observations_complete"] is True
    assert result["profiles"]["on"]["usage"]["tokens"]["total_tokens"] == 42
    assert result["token_savings_established"] is result["benchmark_advantage_claimed"] is False


@pytest.mark.parametrize("field", ["reward", "task_completed", "verifier", "reward_extra"])
def test_different_outcome_observations_do_not_invalidate_configuration_comparison(pair, field):
    trial = pair[1][0]["trials"][0]
    if field == "reward": trial["reward"] = {"reward": 1}
    elif field == "verifier": trial["durations_seconds"]["verifier"] = 0.1
    elif field == "task_completed": trial["supervisor"][field] = True
    else: trial["reward"] = {"PRIVATE-KEY": "PRIVATE-VALUE"}
    result = compare(pair)
    assert result["eligible_configuration_comparison"] is True
    assert result["ineligible_reasons"] == []
    assert result["correctness_observations_match"] is False
    assert ("reward_observations_differ" if field == "reward_extra" else
        "correctness_observations_differ") in result["outcome_difference_reasons"]
    assert "PRIVATE-KEY" not in json.dumps(result) and "PRIVATE-VALUE" not in json.dumps(result)


@pytest.mark.parametrize("change", ["unknown_kwarg", "dataset", "archive", "job_name", "environment", "turns",
    "task_hash", "effort", "model", "cli"])
def test_only_two_explicit_configuration_fields_may_differ(pair, change):
    receipt, config = pair[1]
    kwargs = config["agents"][0]["kwargs"]
    if change == "unknown_kwarg": kwargs["PRIVATE-KEY"] = "PRIVATE-VALUE"
    elif change == "dataset": config["tasks"][0]["path"] = "/different-dataset"
    elif change == "archive": kwargs["runtime_archive"] = "/different-archive"
    elif change == "job_name": config["job_name"] = "different-job"
    elif change == "environment": config["environment"]["kwargs"] = {"extra": True}
    elif change == "turns": kwargs["grok_coding_max_turns"] = 16
    elif change == "task_hash": receipt["comparison_controls"]["declared"]["task_input_sha256"]["instruction.md"] = "c" * 64
    elif change == "effort":
        receipt["reasoning_effort"] = "medium"
        call_for(pair)["reasoning_effort"] = "medium"
    elif change == "model":
        receipt["model"] = config["agents"][0]["model_name"] = "different-model"
        call_for(pair)["model"] = "different-model"
    elif change == "cli": receipt["cli_version"] = "1.0.47"
    reseal(receipt, config)
    result = compare(pair)
    assert result["eligible_configuration_comparison"] is False
    assert "PRIVATE-KEY" not in json.dumps(result) and "PRIVATE-VALUE" not in json.dumps(result)


@pytest.mark.parametrize("change", ["config_unsealed", "config_hash", "status", "archive_digest", "profile",
    "runtime_profile", "provider", "model", "effort", "planning", "no_calls", "unreceipted", "missing_preflight",
    "wrong_preflight", "unknown_preflight_field", "claims_effective", "incomplete_receipt", "baseline", "arm_mismatch"])
def test_invalid_unbound_or_bypassed_runs_cannot_qualify(pair, change):
    receipt, config = pair[1]
    trial = receipt["trials"][0]
    report = trial["supervisor"]
    call = call_for(pair)
    if change == "config_unsealed": config["extra"] = "unsealed"
    elif change == "config_hash": receipt["comparison_controls"]["declared"]["configuration_sha256"] = "f" * 64
    elif change == "status": receipt["comparison_controls"]["status"] = "mismatch"
    elif change == "archive_digest": trial["agent_context"]["metadata"]["runtime_archive_sha256"] = "d" * 64
    elif change == "profile": receipt["grok_pruning_profile"] = OFF
    elif change == "runtime_profile": report["grok_pruning_profile"] = OFF
    elif change == "provider": call["provider"] = "codex_cli"
    elif change == "model": call["model"] = "different"
    elif change == "effort": call["reasoning_effort"] = "different"
    elif change == "planning": call["purpose"] = call["phase"] = "planning"
    elif change == "no_calls": report["provider_invocations"] = []
    elif change == "unreceipted": report["unreceipted_provider_attempt"] = True
    elif change == "missing_preflight": call.pop("provider_invocation_policy")
    elif change == "wrong_preflight": call["provider_invocation_policy"]["grok_pruning"] = preflight(OFF)
    elif change == "unknown_preflight_field": call["provider_invocation_policy"]["grok_pruning"]["SECRET"] = "DO-NOT-EXPORT"
    elif change == "claims_effective": call["provider_invocation_policy"]["grok_pruning"]["effective_settings_verified"] = True
    elif change == "incomplete_receipt": receipt["complete_single_trial_receipt"] = False
    elif change == "baseline": receipt["schema"] = comparison.BASELINE
    else: receipt["arm"] = "no-index"
    result = compare(pair)
    assert result["eligible_configuration_comparison"] is False
    assert "SECRET" not in json.dumps(result) and "DO-NOT-EXPORT" not in json.dumps(result)


@pytest.mark.parametrize("both", [False, True])
def test_same_arm_no_index_comparison_is_supported_and_mixed_arms_rejected(pair, both):
    for index in ([0, 1] if both else [1]):
        receipt, config = pair[index]
        receipt["arm"] = config["agents"][0]["kwargs"]["arm"] = "no-index"
        config["agents"][0]["kwargs"]["model_revision"] = ""
        config["job_name"] = "supervisor-no-index-" + benchmark.TASK
        reseal(receipt, config)
    result = compare(pair)
    assert result["eligible_configuration_comparison"] is both
    if not both:
        assert "configurations_differ" in result["ineligible_reasons"]


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), "\ud800", 1 << 4097, [0] * 100001])
def test_pair_input_bounds_fail_closed(pair, bad):
    pair[1][0]["extra"] = bad
    assert compare(pair)["ineligible_reasons"] == ["on_receipt_or_configuration_invalid"]


def test_legacy_invocation_output_is_unchanged_and_native_usage_is_independent():
    raw = _grok_invocation(complete=True)
    before = comparison._invocations({"provider_invocations": [raw]})[0]
    assert "grok_pruning" not in before
    raw.update(purpose="coding", provider_invocation_policy={"grok_pruning": preflight(ON)})
    after = comparison._invocations({"provider_invocations": [raw]})[0]
    assert after.pop("grok_pruning") == preflight(ON)
    assert after == before
    raw["provider_invocation_policy"]["grok_pruning"]["PRIVATE"] = "NEVER-EXPORT"
    invalid = comparison._invocations({"provider_invocations": [raw]})[0]
    assert invalid.pop("grok_pruning") is None
    assert invalid == before


@pytest.mark.parametrize("invalid", [False, True])
def test_collect_and_markdown_keep_validated_closed_profile_or_unknown(tmp_path, invalid):
    call = _grok_invocation(complete=True)
    call.update(purpose="coding", provider_invocation_policy={"grok_pruning": preflight(ON)})
    if invalid: call["provider_invocation_policy"]["grok_pruning"]["PRIVATE"] = "NEVER-EXPORT"
    supervisor = _receipt(tmp_path, "supervisor.json", comparison.SUPERVISOR,
        [{"trial": "profiled", "supervisor": {"provider_invocations": [call]}}], "full")
    result = comparison.collect(baseline=_baseline(tmp_path), supervisors=[supervisor])
    text = comparison.markdown(result)
    assert "Exact file observed" in text and "actual pruning behavior and token savings remain unknown" in text
    assert "NEVER-EXPORT" not in text and "PRIVATE" not in json.dumps(result)
    row = next(row for row in result["rows"] if row["arm"] == "full")
    assert row["tokens"]["total_tokens"] == 42
    assert (row["invocations"][0]["grok_pruning"] is None) is invalid


def test_matching_native_failure_is_an_observation_not_success(pair):
    for receipt, _ in pair:
        receipt["trials"][0]["reward"] = {"reward": 0}
        receipt["trials"][0]["durations_seconds"]["verifier"] = 1
        receipt["trials"][0]["supervisor"]["task_completed"] = False
    result = compare(pair)
    assert result["eligible_configuration_comparison"] is True
    assert result["correctness_observations_complete"] is True
    assert result["profiles"]["on"]["correctness"]["official_reward"] == 0
