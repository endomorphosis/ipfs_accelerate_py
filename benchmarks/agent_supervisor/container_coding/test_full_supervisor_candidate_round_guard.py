"""Refuse unsupported full-suite continuations before setup or model work.

The ordinary owner can qualify two rounds separately. The full benchmark still
has Doctor obligations, so accepting its choice here would silently promise an
unavailable route. No dataset, credentials, container or provider is read.
"""
import asyncio
from copy import deepcopy
import json
from pathlib import Path

import pytest

from benchmarks.agent_supervisor.container_coding import benchmark_controls as controls
from benchmarks.agent_supervisor.container_coding import full_supervisor_benchmark as benchmark
from benchmarks.agent_supervisor.container_coding import full_supervisor_harbor_agent as harbor
from benchmarks.agent_supervisor.container_coding import terminal_container_supervisor as driver
from benchmarks.agent_supervisor.container_coding.test_benchmark_controls import HASHES, declaration


@pytest.mark.parametrize("arm", ["full", "no-index"])
def test_default_round_has_identical_configuration_and_declared_controls(arm):
    arguments = (Path("/authored-dataset"), Path("/authored-output"), Path("/authored-archive"), arm)
    original = benchmark.config_for(*arguments)
    explicit = benchmark.config_for(*arguments, candidate_rounds=1)
    assert explicit == original
    assert "candidate_rounds" not in original["agents"][0]["kwargs"]
    assert declaration(original) == declaration(explicit)


@pytest.mark.parametrize("value", [True, False, "1", "2", 1.0, 2.0, None, 0, 3, -1])
def test_round_count_is_exact_integer(value):
    with pytest.raises(ValueError, match="exactly one or two"):
        controls.validate_full_supervisor_candidate_rounds(value, arm="full")


@pytest.mark.parametrize("arm,reason", [("full", "Doctor constraints"), ("no-index", "semantic source broker")])
def test_two_round_preparation_refuses_before_assets_and_provider_selection(tmp_path, monkeypatch, arm, reason):
    monkeypatch.setattr(benchmark, "resolve_provider_profile", lambda *a, **kw: pytest.fail("provider selected before refusal"))
    with pytest.raises(ValueError, match=reason):
        benchmark.prepare(dataset=tmp_path/"missing-data", output=tmp_path/"out",
            archive=tmp_path/"missing-archive", arm=arm, candidate_rounds=2)
    with pytest.raises(ValueError, match=reason):
        benchmark.config_for(tmp_path/"missing-data", tmp_path/"out", tmp_path/"missing-archive",
            arm, candidate_rounds=2)
    assert not (tmp_path/"out").exists()


@pytest.mark.parametrize("arm", ["full", "no-index"])
def test_direct_driver_refuses_before_instruction_state_or_planning(tmp_path, monkeypatch, arm):
    monkeypatch.setattr(driver.preparation, "prepare", lambda *a, **kw: pytest.fail("planning reached"))
    with pytest.raises(ValueError, match="candidate rounds=2"):
        driver.run(instruction=tmp_path/"missing-instruction", state=tmp_path/"state", arm=arm,
            candidate_rounds=2)
    assert not (tmp_path/"state").exists()


@pytest.mark.parametrize("arm", ["full", "no-index"])
def test_harbor_constructor_refuses_before_base_setup_or_archive(tmp_path, monkeypatch, arm):
    monkeypatch.setattr(harbor.BaseAgent, "__init__", lambda *a, **kw: pytest.fail("base agent setup reached"))
    with pytest.raises(ValueError, match="candidate rounds=2"):
        harbor.FullSupervisorAgent(runtime_archive=str(tmp_path/"missing-archive"), arm=arm,
            candidate_rounds=2)


@pytest.mark.parametrize("operation", ["setup", "run"])
@pytest.mark.parametrize("value", [2, True, "2"])
def test_harbor_rechecks_selection_before_environment_access(operation, value):
    agent = object.__new__(harbor.FullSupervisorAgent)
    agent.arm = "full"
    agent.candidate_rounds = value
    agent._frozen_candidate_rounds = 1
    # Missing archive/context/environment attributes prove refusal comes first.
    coroutine = agent.setup(None) if operation == "setup" else agent.run("authored instruction", None, None)
    with pytest.raises(ValueError):
        asyncio.run(coroutine)


@pytest.mark.parametrize("operation", ["execute", "collect"])
@pytest.mark.parametrize("location", ["prepared", "config"])
def test_unsupported_round_cannot_be_injected_into_saved_trial(tmp_path, monkeypatch, operation, location):
    prepared = dict(arm="full")
    config = dict(agents=[dict(kwargs=dict(arm="full"))])
    selected = prepared if location == "prepared" else config["agents"][0]["kwargs"]
    selected["candidate_rounds"] = 2
    (tmp_path/"preparation.json").write_text(json.dumps(prepared))
    (tmp_path/"config.json").write_text(json.dumps(config))
    monkeypatch.setattr(benchmark, "_prepared_task", lambda *a, **kw: pytest.fail("trial execution/read reached"))
    with pytest.raises(ValueError, match="Doctor constraints"):
        getattr(benchmark, operation)(tmp_path)
    assert not (tmp_path/"receipt.json").exists()


def test_closed_controls_cannot_declare_two_round_full_trial():
    config = benchmark.config_for(Path("/authored-data"), Path("/authored-out"), Path("/authored-archive"), "full")
    frozen = declaration(config)
    config["agents"][0]["kwargs"]["candidate_rounds"] = 2
    with pytest.raises(ValueError, match="Doctor constraints"):
        declaration(config)
    observed = controls.observe_controls(dict(comparison_controls=frozen), config,
        current_task_hashes=HASHES)
    assert observed["status"] == "mismatch" and observed["configuration_unchanged"] is False


def test_exact_explicit_one_remains_bound_by_the_existing_full_config_hash():
    config = benchmark.config_for(Path("/authored-data"), Path("/authored-out"), Path("/authored-archive"), "full")
    explicit = deepcopy(config)
    explicit["agents"][0]["kwargs"]["candidate_rounds"] = 1
    frozen = declaration(explicit)
    assert controls.validate_controls(frozen)
    assert frozen["schema"] == controls.SCHEMA
    assert frozen["configuration_sha256"] != declaration(config)["configuration_sha256"]
    assert controls.validate_frozen_candidate_rounds(dict(arm="full", candidate_rounds=1), explicit) == 1


def test_ordinary_owner_selection_still_requires_no_doctor_constraints():
    from benchmarks.agent_supervisor.container_coding.terminal_doctor_dispatch import implementation_argv
    settings = dict(router=Path("/owner"), model="authored-model", reasoning="high", timeout=90,
        semantic_repository=Path("/canonical"), source_view_profile="identity@1", candidate_rounds=2)
    ordinary = implementation_argv(**settings, doctor=None)
    assert ordinary[ordinary.index("--candidate-rounds")+1] == "2"
    for doctor in (dict(route="model_router", status="residual"),
                   dict(route="doctor_candidate", status="candidate_ready"),
                   dict(route="doctor_contract_candidate", status="candidate_ready")):
        with pytest.raises(ValueError, match="ordinary semantic source coding"):
            implementation_argv(**settings, doctor=doctor)
