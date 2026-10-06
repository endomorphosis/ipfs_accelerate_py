"""Pure declaration support for the existing closed setup-cache selection."""
from copy import deepcopy
from pathlib import Path

import pytest

from benchmarks.agent_supervisor.container_coding.benchmark_controls import (
    _control_record, compare_controls, observe_controls, validate_controls,
)
from benchmarks.agent_supervisor.container_coding.benchmark_resource_profile import SOURCE384_PROFILE
from benchmarks.agent_supervisor.container_coding.full_supervisor_benchmark import config_for as supervisor_config
from benchmarks.agent_supervisor.container_coding.native_codex_baseline import config_for as native_config
from benchmarks.agent_supervisor.container_coding.test_benchmark_controls import HASHES, declaration, observation
from benchmarks.agent_supervisor.container_coding import terminal_setup_cache_advice as advice

SELECTION = dict(schema="terminal-setup-cache-selection@1", policy=advice.POLICY, manifest_sha256="e"*64)


def selected():
    return supervisor_config(Path("/dataset"),Path("/out/full"),Path("/archive"),"full",
        resource_profile=SOURCE384_PROFILE,setup_cache_selection=SELECTION)


def test_normalized_full_selection_is_declared_without_archive_or_auth_reads(monkeypatch):
    from harbor.models.job.config import JobConfig
    def forbidden(*args,**kwargs):pytest.fail("pure comparison controls read deployment assets")
    monkeypatch.setattr(advice,"_regular_bytes",forbidden)
    monkeypatch.setattr(advice,"validate_setup_cache_prerequisites",forbidden)
    config=JobConfig.model_validate(selected(),extra="forbid").model_dump(mode="json")
    controls=declaration(config)
    assert validate_controls(controls) and controls["schema"]=="terminal-benchmark-declared-controls@1"
    assert observation(config)["status"]=="observed"


def _historical_configuration(arm, profile):
    """Fixed declaration from 6968b0ae6, never a current executable profile.

    The retained hashes include this authored path string; no auth file is read.
    Keep the old model, CLI and inert retrieval revision explicit so changes to
    current defaults do not silently redefine the historical hash contract.
    """
    agent = dict(model_name="gpt-5.6-sol", override_timeout_sec=300.0,
                 max_timeout_sec=300.0, override_setup_timeout_sec=1800.0)
    if arm == "native":
        agent.update(name="codex", kwargs={"version": "0.158.0", "reasoning_effort": "high"},
                     env={"CODEX_AUTH_JSON_PATH": "/home/barberb/.codex/auth.json"})
    else:
        agent.update(import_path="benchmarks.agent_supervisor.container_coding.full_supervisor_harbor_agent:FullSupervisorAgent",
                     kwargs={"runtime_archive": "/archive", "arm": arm,
                             "model_revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"})
        if profile is not None:
            agent["kwargs"]["resource_profile"] = profile
    config = dict(job_name=("native-codex" if arm == "native" else "supervisor-" + arm) + "-fix-code-vulnerability",
                  jobs_dir="/out/" + ("base" if arm == "native" else arm) + "/jobs",
                  n_attempts=1, n_concurrent_trials=1, timeout_multiplier=1.0,
                  retry={"max_retries": 0}, environment={"type": "docker", "force_build": True, "delete": True},
                  verifier={"disable": False}, agents=[agent], tasks=[{"path": "/dataset/fix-code-vulnerability"}])
    if profile is not None:
        assert profile == SOURCE384_PROFILE
        config["environment"].update(override_cpus=5, override_memory_mb=12288,
                                     cpu_enforcement_policy="limit", memory_enforcement_policy="limit")
    return config


@pytest.mark.parametrize("arm,profile,expected",[
    ("native",None,"66af6d6afb639666d6bc61014ce289fe25f55816a7ce4747db90f34ee63e43a1"),
    ("full",None,"5e1a5d434d0287a8ea6d2fe2a5a7fff6a2de4c6d97c51583f3562f68d718cc8c"),
    ("no-index",None,"3796aefc6a391cb5dd02812a2e5eac1997ac12ff8b40b931a786a74d5aee5bc5"),
    ("native",SOURCE384_PROFILE,"e72a33f06b429631d3a2cbe9bccf548d418fece7c6eea45dfa12de1db157aa1e"),
    ("full",SOURCE384_PROFILE,"9c27af01c108ae793ce725bdc71f726ba24d5f58cd50969cd50a57da392e7e39"),
    ("no-index",SOURCE384_PROFILE,"700ccd9dcdea021fecd4cada6af606fb48a74d35dcd218963b36999ef198cdc8"),
])
def test_absent_selection_historical_hash_replays_and_current_profile_delta_is_explicit(arm,profile,expected,monkeypatch):
    historical = _historical_configuration(arm, profile)
    frozen = _control_record(historical, task_input_sha256=HASHES, task="fix-code-vulnerability",
                             model="gpt-5.6-sol", reasoning_effort="high", cli_version="0.158.0")
    assert frozen["sha256"] == expected
    observed = observe_controls({"comparison_controls": frozen}, historical, current_task_hashes=HASHES)
    assert observed["status"] == "observed" and observed["configuration_unchanged"] is True
    # Pin the path only for this pure configuration comparison, not credential
    # discovery. Current provider/retrieval defaults intentionally differ.
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: Path("/home/barberb")))
    config=(native_config(Path("/dataset"),Path("/out/base"),resource_profile=profile) if arm=="native"
        else supervisor_config(Path("/dataset"),Path("/out")/arm,Path("/archive"),arm,resource_profile=profile))
    assert "setup_cache_selection" not in config["agents"][0]["kwargs"]
    expected_current = deepcopy(historical)
    expected_current["agents"][0]["model_name"] = "gpt-6.1-sol"
    expected_current["agents"][0]["kwargs"].update(
        {"version": "0.160.0"} if arm == "native" else {"model_revision": ""})
    assert config == expected_current
    current = declaration(config)
    assert current["identity"] == {**frozen["identity"], "model": "gpt-6.1-sol", "cli_version": "0.160.0"}
    assert current["sha256"] != expected
    assert not compare_controls(observed, observation(config))["matches"]


@pytest.mark.parametrize("change",["none","policy","hash","uppercase_hash","schema","missing","extra","wrong_type"])
def test_malformed_selection_refused(change):
    config=selected();kwargs=config["agents"][0]["kwargs"];value=kwargs["setup_cache_selection"]
    if change=="none":kwargs["setup_cache_selection"]=None
    elif change=="policy":value["policy"]="unreviewed-policy"
    elif change=="hash":value["manifest_sha256"]="short"
    elif change=="uppercase_hash":value["manifest_sha256"]="A"*64
    elif change=="schema":value["schema"]="terminal-setup-cache-selection@2"
    elif change=="missing":value.pop("policy")
    elif change=="extra":value["private_auth_path"]="private"
    else:kwargs["setup_cache_selection"]=[value]
    with pytest.raises(ValueError):declaration(config)


@pytest.mark.parametrize("change",["no-index","missing_arm","missing_profile","wrong_profile","none_profile","changed_limits"])
def test_selection_requires_full_arm_and_exact_common_resources(change):
    config=selected();kwargs=config["agents"][0]["kwargs"]
    if change=="no-index":kwargs["arm"]="no-index"
    elif change=="missing_arm":kwargs.pop("arm")
    elif change=="missing_profile":kwargs.pop("resource_profile")
    elif change=="wrong_profile":kwargs["resource_profile"]="source384-5cpu-8gib@1"
    elif change=="none_profile":kwargs["resource_profile"]=None
    else:config["environment"]["override_memory_mb"]=8192
    with pytest.raises(ValueError):declaration(config)


@pytest.mark.parametrize("change",["different_manifest","dropped","malformed"])
def test_selection_change_cannot_reseal_frozen_observation(change):
    config=selected();prepared={"comparison_controls":declaration(config)};kwargs=config["agents"][0]["kwargs"]
    if change=="different_manifest":kwargs["setup_cache_selection"]["manifest_sha256"]="d"*64
    elif change=="dropped":kwargs.pop("setup_cache_selection")
    else:kwargs["setup_cache_selection"]=None
    seen=observe_controls(prepared,config,current_task_hashes=HASHES)
    assert seen["status"]=="mismatch" and seen["configuration_unchanged"] is False
    assert seen["declared"]==prepared["comparison_controls"]
    assert compare_controls(seen,seen)["matches"] is False


def test_common_controls_compare_without_claiming_equal_setup_or_runtime():
    from harbor.models.job.config import JobConfig
    baseline=JobConfig.model_validate(native_config(Path("/dataset"),Path("/out/base"),
        resource_profile=SOURCE384_PROFILE),extra="forbid").model_dump(mode="json")
    candidate=JobConfig.model_validate(selected(),extra="forbid").model_dump(mode="json")
    left,right=observation(baseline),observation(candidate)
    assert left["declared"]["configuration_sha256"]!=right["declared"]["configuration_sha256"]
    result=compare_controls(left,right)
    assert result["matches"] is True and result["differences"]==[]
    assert result["basis"]=="predeclared_configuration_only; runtime_enforcement_not_established"
