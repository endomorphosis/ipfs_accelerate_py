"""Configuration controls are independent of model usage and reward accounting."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from benchmarks.agent_supervisor.container_coding.benchmark_controls import (
    build_controls, compare_controls, observe_controls, validate_controls,
)
from benchmarks.agent_supervisor.container_coding.native_codex_baseline import config_for
from benchmarks.agent_supervisor.container_coding.full_supervisor_benchmark import config_for as supervisor_config
from benchmarks.agent_supervisor.container_coding.test_benchmark_comparison import (
    _baseline, _receipt, SUPERVISOR,
)
from benchmarks.agent_supervisor.container_coding.benchmark_comparison import collect, markdown


HASHES = {"instruction.md": "a" * 64, "task.toml": "b" * 64,
          "environment/Dockerfile": "c" * 64, "tests/test.sh": "d" * 64}


def declaration(config, hashes=HASHES):
    return build_controls(config, task_input_sha256=hashes, task="fix-code-vulnerability",
                          model="gpt-6.1-sol", reasoning_effort="high", cli_version="0.160.0")


def observation(config, hashes=HASHES):
    return observe_controls({"comparison_controls": declaration(config, hashes)}, config,
                            current_task_hashes=hashes)


def test_existing_native_and_both_supervisor_configs_have_equal_common_controls():
    baseline = config_for(Path("/dataset"), Path("/out/base"))
    for arm in ("full", "no-index"):
        candidate = supervisor_config(Path("/dataset"), Path("/out") / arm, Path("/archive"), arm)
        result = compare_controls(observation(baseline), observation(candidate))
        assert result["matches"] is True
        assert result["differences"] == []
    assert baseline["agents"][0]["override_setup_timeout_sec"] == 1800


def test_actual_harbor_normalized_configs_preserve_equal_controls():
    from harbor.models.job.config import JobConfig

    def normalized(config):
        return JobConfig.model_validate(config, extra="forbid").model_dump(mode="json")

    baseline = normalized(config_for(Path("/dataset"), Path("/out/base")))
    for arm in ("full", "no-index"):
        candidate = normalized(supervisor_config(Path("/dataset"), Path("/out") / arm,
                                                Path("/archive"), arm))
        assert compare_controls(observation(baseline), observation(candidate))["matches"] is True


@pytest.mark.parametrize("field,value,section,key", [
    ("n_concurrent_trials", 4, "job_controls_sha256", "n_concurrent_trials"),
    ("n_attempts", 2, "job_controls_sha256", "n_attempts"),
    ("retry", {"max_retries": 2}, "job_controls_sha256", "retry"),
    ("environment", {"type": "docker", "override_cpus": 8}, "job_controls_sha256", "environment"),
    ("verifier", {"disable": True}, "job_controls_sha256", "verifier"),
    ("extra_instructions", ["extra hint"], "job_controls_sha256", "extra_instructions"),
    ("override_timeout_sec", 600, "agent_controls_sha256", "override_timeout_sec"),
    ("override_setup_timeout_sec", None, "agent_controls_sha256", "override_setup_timeout_sec"),
    ("mcp_servers", [{"name": "extra tool"}], "agent_controls_sha256", "mcp_servers"),
])
def test_independent_control_mismatches_are_visible(field, value, section, key):
    original = config_for(Path("/dataset"), Path("/out"))
    changed = deepcopy(original)
    target = changed["agents"][0] if section == "agent_controls_sha256" else changed
    target[field] = value
    result = compare_controls(observation(original), observation(changed))
    assert result["matches"] is False
    assert result["differences"] == [f"{section}.{key}"]


def test_changed_oracle_hash_is_compared_without_exporting_oracle_content():
    config = config_for(Path("/dataset"), Path("/out"))
    result = compare_controls(observation(config), observation(config, {**HASHES, "tests/test.sh": "e" * 64}))
    assert result["matches"] is False
    assert result["differences"] == ["task_input_sha256.tests/test.sh"]


def test_post_preparation_change_cannot_reseal_the_declaration():
    config = config_for(Path("/dataset"), Path("/out"))
    prepared = {"comparison_controls": declaration(config)}
    config["n_attempts"] = 9
    seen = observe_controls(prepared, config, current_task_hashes=HASHES)
    assert seen["status"] == "mismatch"
    assert seen["declared"] == prepared["comparison_controls"]
    assert compare_controls(seen, seen)["matches"] is False


@pytest.mark.parametrize("change", ["model", "agent_count", "cli", "inputs"])
def test_invalid_changed_configuration_retains_failed_observation(change):
    config = config_for(Path("/dataset"), Path("/out"))
    prepared = {"comparison_controls": declaration(config)}
    hashes = HASHES
    if change == "model":
        config["agents"][0]["model_name"] = "other"
    elif change == "agent_count":
        config["agents"].append(deepcopy(config["agents"][0]))
    elif change == "cli":
        config["agents"][0]["kwargs"]["version"] = "other"
    else:
        hashes = None
    seen = observe_controls(prepared, config, current_task_hashes=hashes)
    assert seen["status"] == "mismatch"
    assert seen["configuration_unchanged"] is False
    assert compare_controls(seen, seen)["matches"] is False


def test_arm_or_task_path_drift_is_refused_even_though_unequal_between_arms():
    config = supervisor_config(Path("/dataset"), Path("/out"), Path("/archive"), "full")
    prepared = {"comparison_controls": declaration(config)}
    changed = deepcopy(config)
    changed["agents"][0]["kwargs"]["arm"] = "no-index"
    assert observe_controls(prepared, changed, current_task_hashes=HASHES)["status"] == "mismatch"
    changed = deepcopy(config)
    changed["tasks"][0]["path"] = "/different-task"
    assert observe_controls(prepared, changed, current_task_hashes=HASHES)["status"] == "mismatch"


def test_legacy_preparation_and_missing_observation_remain_unknown():
    seen = observe_controls({}, {}, current_task_hashes={})
    assert seen["status"] == "unavailable"
    assert compare_controls(seen, seen)["matches"] is None
    assert compare_controls(None, None)["matches"] is None


@pytest.mark.parametrize("corruption", ["digest", "missing", "extra", "wrong_type"])
def test_malformed_frozen_controls_never_match(corruption):
    config = config_for(Path("/dataset"), Path("/out"))
    declared = declaration(config)
    if corruption == "digest":
        declared["sha256"] = "f" * 64
    elif corruption == "missing":
        declared["job_controls_sha256"].pop("retry")
    elif corruption == "extra":
        declared["unchecked"] = True
    else:
        declared["identity"]["model"] = False
    assert not validate_controls(declared)
    seen = observe_controls({"comparison_controls": declared}, config, current_task_hashes=HASHES)
    assert compare_controls(seen, seen)["matches"] is False


def test_no_config_secret_or_prompt_text_exported():
    config = config_for(Path("/dataset"), Path("/out"))
    config["agents"][0]["env"] = {"CODEX_AUTH_JSON_PATH": "/PRIVATE CREDENTIAL PATH"}
    config["extra_instructions"] = ["PRIVATE PROMPT"]
    encoded = json.dumps(declaration(config))
    assert "PRIVATE" not in encoded and "SECRET" not in encoded


@pytest.mark.parametrize("field", ["user_agent", "extra_instruction_paths", "mounts", "extra_docker_compose", "env"])
def test_unsealed_external_configuration_inputs_are_refused(field):
    config = config_for(Path("/dataset"), Path("/out"))
    target = config["environment"] if field in {"mounts", "extra_docker_compose"} else config
    if field == "env":
        config["agents"][0]["env"]["API_ENDPOINT"] = "changed"
    else:
        target[field] = ["/external/input"]
    with pytest.raises(ValueError, match="comparison profile"):
        declaration(config)


@pytest.mark.parametrize("key,value", [("version", "other-cli"), ("reasoning_effort", "low")])
def test_native_kwargs_cannot_contradict_declared_identity(key, value):
    config = config_for(Path("/dataset"), Path("/out"))
    config["agents"][0]["kwargs"][key] = value
    with pytest.raises(ValueError, match="native model controls"):
        declaration(config)


def test_supervisor_identity_must_match_source_bound_profile():
    config = supervisor_config(Path("/dataset"), Path("/out"), Path("/archive"), "full")
    with pytest.raises(ValueError, match="fixed source profile"):
        build_controls(config, task_input_sha256=HASHES, task="fix-code-vulnerability",
                       model="gpt-6.1-sol", reasoning_effort="low", cli_version="0.160.0")
    config["agents"][0]["kwargs"]["reasoning_effort"] = "low"
    with pytest.raises(ValueError, match="unsupported supervisor kwargs"):
        declaration(config)


def test_report_keeps_legacy_trials_without_claiming_matched_controls(tmp_path):
    baseline = _baseline(tmp_path)
    supervised = _receipt(tmp_path, "supervised.json", SUPERVISOR,
                          [{"trial": "legacy", "reward": {"reward": 0}}], "full")
    result = collect(baseline=baseline, supervisors=[supervised])
    assert len(result["rows"]) == 2
    assert all(row["reported_identity_matches"] is True for row in result["rows"])
    assert all(row["comparison_profile_matches"] is None for row in result["rows"])
    assert not result["all_declared_controls_match"]
    assert "Declared controls" in markdown(result)


def test_report_rejects_changed_baseline_even_when_candidate_declaration_matches(tmp_path):
    baseline = _baseline(tmp_path)
    supervised = _receipt(tmp_path, "supervised.json", SUPERVISOR,
                          [{"trial": "new", "reward": {"reward": 1}}], "no-index")
    controls = observation(config_for(Path("/dataset"), Path("/out")))
    for path in (baseline, supervised):
        value = json.loads(path.read_text())
        value["comparison_controls"] = controls
        if path == baseline:
            value["original_task_inputs_unchanged"] = False
        path.write_text(json.dumps(value))
    result = collect(baseline=baseline, supervisors=[supervised])
    assert all(row["comparison_profile_matches"] is False for row in result["rows"])
    assert not result["all_declared_controls_match"]
    assert not result["runtime_control_enforcement_verified"]
