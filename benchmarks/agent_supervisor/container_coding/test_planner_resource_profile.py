"""Planner time changes only under an explicit signed development profile."""
import json
from types import SimpleNamespace

import pytest

from benchmarks.agent_supervisor.container_coding import benchmark_resource_profile as profile
from benchmarks.agent_supervisor.container_coding import terminal_container_supervisor as driver
from benchmarks.agent_supervisor.container_coding import terminal_indexed_preparation as prep
from benchmarks.agent_supervisor.container_coding.test_terminal_indexed_preparation import original, _proposal_json
from benchmarks.agent_supervisor.container_coding.test_terminal_initial_context import _version


NEW = profile.PLANNER180_SOURCE384_PROFILE


def test_explicit_planner_profile_preserves_existing_resource_and_execution_dictionaries():
    legacy = dict(driver_seconds=285, cleanup_seconds=40, source384_seconds=90,
        harbor_seconds=300, exec_seconds=295, qualification_seconds=270,
        qualification_exec_seconds=300, native_start_seconds=20)
    extended = dict(driver_seconds=900, cleanup_seconds=60, source384_seconds=180,
        harbor_seconds=960, exec_seconds=910, qualification_seconds=600,
        qualification_exec_seconds=630, native_start_seconds=120)
    assert profile.execution_budget() == profile.execution_budget(profile.SOURCE384_PROFILE) == legacy
    assert profile.execution_budget(profile.EXTENDED_SOURCE384_PROFILE) == profile.execution_budget(NEW) == extended
    assert extended["driver_seconds"] - extended["cleanup_seconds"] == 840
    assert profile.resource_environment(NEW) == profile.resource_environment(profile.EXTENDED_SOURCE384_PROFILE)
    assert profile.resource_environment(NEW)["override_memory_mb"] == 16384
    assert profile.resource_environment(NEW)["override_cpus"] == 5
    assert profile.admission_environment(NEW) == profile.admission_environment(profile.EXTENDED_SOURCE384_PROFILE)
    assert profile.native_start_timeout_ms(NEW, remaining_work_seconds=840) == 120000
    for selected in (None, profile.SOURCE384_PROFILE, profile.EXTENDED_SOURCE384_PROFILE):
        assert profile.planner_timeout_seconds(selected) == 90
    assert profile.planner_timeout_seconds(NEW) == 180
    with pytest.raises(ValueError, match="unknown"):
        profile.planner_timeout_seconds("unreviewed")


@pytest.mark.parametrize("selected,caller_timeout,expected", [
    (None, None, 90), (profile.SOURCE384_PROFILE, None, 90),
    (profile.EXTENDED_SOURCE384_PROFILE, None, 90), (NEW, None, 180), (NEW, 37, 37),
])
def test_signed_request_and_real_planner_config_deliver_only_selected_timeout(
        original, monkeypatch, selected, caller_timeout, expected):
    root, instruction, state = original
    prepared = prep.prepare(repository=root, instruction=instruction, state=state,
        resource_profile=selected, disable_intent_autoencoder=True)
    limit = profile.planner_timeout_seconds(selected)
    assert prepared["request"]["budget"]["max_latency_ms"] == limit * 1000
    assert prepared["manifest"]["payload"]["planning_inputs"]["request"] == prepared["request"]
    assert prepared["planner_timeout_seconds"] == limit
    assert prepared["max_total_agent_seconds"] == profile.execution_budget(selected)["harbor_seconds"]
    assert prep._load_prepared(state) == prepared
    _version(monkeypatch)
    observed = []
    native = prep.generate_prompt_goal_graph
    def generate(*args, **kwargs):
        observed.append(("config", kwargs["config"].timeout_seconds))
        return native(*args, **kwargs)
    monkeypatch.setattr(prep, "generate_prompt_goal_graph", generate)
    def router(prompt, **kwargs):
        observed.append(("router", kwargs["timeout"]))
        return dict(text=_proposal_json(prepared), observation={}, execution_receipt=None)
    result = prep.plan(state, provider_callable=router, timeout_seconds=caller_timeout)
    assert result["qualified"], result
    assert result["planner_timeout_seconds"] == expected
    assert observed == [("config", expected), ("router", expected)]


@pytest.mark.parametrize("selected,timeout", [
    (None, 91), (profile.SOURCE384_PROFILE, 180), (profile.EXTENDED_SOURCE384_PROFILE, 180),
    (NEW, 181), (NEW, True), (NEW, 180.0), (NEW, 0),
])
def test_caller_cannot_extend_signed_planner_cap(original, selected, timeout):
    root, instruction, state = original
    prep.prepare(repository=root, instruction=instruction, state=state,
        resource_profile=selected, disable_intent_autoencoder=True)
    with pytest.raises(ValueError, match="signed overall trial budget"):
        prep.plan(state, provider_callable=lambda *args, **kwargs: pytest.fail("provider reached"),
            timeout_seconds=timeout)
    assert not (state / "planner-invoked.json").exists()


@pytest.mark.parametrize("damage", ["recorded_timeout", "resource_selection", "signed_request"])
def test_unsigned_preparation_changes_cannot_increase_planning_allowance(original, damage):
    root, instruction, state = original
    prepared = prep.prepare(repository=root, instruction=instruction, state=state,
        disable_intent_autoencoder=True)
    if damage == "recorded_timeout":
        prepared["planner_timeout_seconds"] = 180
    elif damage == "resource_selection":
        prepared.update(resource_profile=NEW, planner_timeout_seconds=180)
    else:
        prepared["request"]["budget"]["max_latency_ms"] = 180000
    (state / "prepared.json").write_text(json.dumps(prepared))
    with pytest.raises(ValueError):
        prep.plan(state, provider_callable=lambda *args, **kwargs: pytest.fail("provider reached"))
    assert not (state / "planner-invoked.json").exists()


@pytest.mark.parametrize("selected,left,expected", [
    (None, 200, 90), (profile.EXTENDED_SOURCE384_PROFILE, 400, 90),
    (NEW, 400, 180), (NEW, 100, 70), (NEW, 31, 1),
])
def test_driver_planner_uses_remaining_work_and_keeps_original_cleanup(
        tmp_path, monkeypatch, selected, left, expected):
    budget = profile.execution_budget(selected)
    now = [1000.]
    alarms, calls = [], []
    monkeypatch.delenv("IPFS_DATASETS_PROOF_RESOURCE_PROFILE", raising=False)
    for key, value in profile.admission_environment(selected).items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(driver, "ROOT", tmp_path)
    monkeypatch.setattr(driver.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(driver.os, "umask", lambda *args: None)
    monkeypatch.setattr(driver.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(driver.signal, "signal", lambda *args: None)
    monkeypatch.setattr(driver.signal, "setitimer", lambda *args: alarms.append(args))
    monkeypatch.setattr(driver, "_failure_diagnostics", lambda *args, **kwargs: {})
    monkeypatch.setattr(driver, "_final_context_audit", lambda *args, **kwargs: None)
    monkeypatch.setattr(driver.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=0))
    def prepare(**kwargs):
        now[0] += budget["driver_seconds"] - budget["cleanup_seconds"] - left
        return {"intent_preplanning": {}}
    class PlanningBoundaryReached(RuntimeError):
        pass
    def plan(**kwargs):
        calls.append(kwargs["timeout_seconds"])
        raise PlanningBoundaryReached("authored planner boundary")
    monkeypatch.setattr(driver.preparation, "prepare", prepare)
    monkeypatch.setattr(driver.preparation, "plan", plan)
    state = tmp_path / "state/run"
    state.mkdir(parents=True)
    result = driver.run(instruction=tmp_path / "instruction", state=state, arm="no-index",
        resource_profile=selected)
    assert calls == [expected]
    assert result["error"]["type"] == "PlanningBoundaryReached"
    assert result["error_phase"] == "planning"
    assert result["max_total_agent_seconds"] == budget["driver_seconds"]
    assert result["reserved_cleanup_seconds"] == budget["cleanup_seconds"]
    assert result["work_cutoff_seconds"] == budget["driver_seconds"] - budget["cleanup_seconds"]
    assert (driver.signal.ITIMER_REAL, result["work_cutoff_seconds"]) in alarms
    assert result["provider_invocations"] == []


def test_new_planner_profile_cannot_expand_driver_or_select_unqualified_reuse(tmp_path, monkeypatch):
    from benchmarks.agent_supervisor.container_coding import terminal_setup_cache_advice as cache
    from benchmarks.agent_supervisor.container_coding import terminal_source384_warm_recovery as warm
    monkeypatch.setattr(driver.os, "umask", lambda *args: pytest.fail("invalid cap reached setup"))
    with pytest.raises(ValueError, match="bounded arm"):
        driver.run(instruction=tmp_path / "instruction", state=tmp_path / "state", arm="full",
            resource_profile=NEW, timeout_seconds=901)
    assert warm.validate_selection(NEW, None) is None
    with pytest.raises(ValueError, match="explicit policy"):
        warm.validate_selection(NEW, warm.POLICY)
    assert cache.validate_setup_cache_prerequisites(None, install_codex=True, auth_json=None,
        resource_profile=NEW) is None
    with pytest.raises(ValueError, match="selected cache policy"):
        cache.validate_setup_cache_prerequisites(dict(schema="terminal-setup-cache-selection@1",
            policy=cache.POLICY_V2, manifest_sha256="a" * 64), install_codex=True,
            auth_json=tmp_path / "unused-auth", resource_profile=NEW)
