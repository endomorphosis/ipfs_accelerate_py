"""Explicit child-local decision policy controls survive host and Harbor routing.

These authored configuration and process fixtures do not stage a policy,
launch providers or Docker, or claim Terminal Bench acceptance or savings.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import shlex
import sys
from types import SimpleNamespace

import pytest
from harbor.models.agent.context import AgentContext
from harbor.models.job.config import JobConfig

from benchmarks.agent_supervisor.container_coding import full_supervisor_benchmark as benchmark
from benchmarks.agent_supervisor.container_coding import full_supervisor_harbor_agent as adapter
from test.api.test_benchmark_prompt_options import DEFAULTS
from test.api.test_terminal_timeout_configuration import (
    _authored_prepare_dataset,
)

POLICY_PATH = "/opt/ipfs-supervisor/task-policy/decision.json"
POLICY_SHA = "d" * 64
DECISION = {
    "semantic_decision_policy_artifact": POLICY_PATH,
    "semantic_decision_policy_sha256": POLICY_SHA,
    "semantic_decision_max_calls": 3,
    "semantic_decision_max_total_input_tokens": None,
}


@pytest.mark.parametrize("arm", ["full", "no-index"])
def test_default_decision_controls_preserve_legacy_configuration_shape(tmp_path, arm):
    assert benchmark._prompt_options(arm) == DEFAULTS
    assert benchmark._prompt_options(arm, semantic_decision_max_calls=3,
                                     semantic_decision_max_total_input_tokens=None) == DEFAULTS
    config = benchmark.config_for(tmp_path / "dataset", tmp_path / "output", tmp_path / "archive", arm)
    assert not set(DECISION) & set(config["agents"][0]["kwargs"])
    observation = benchmark._prompt_option_observation(
        {"arm": arm, "prompt_comparison_options": DEFAULTS}, config["agents"][0]["kwargs"])
    assert observation["prompt_comparison_options"] == DEFAULTS
    assert observation["prompt_comparison_config_unchanged"] is True


@pytest.mark.parametrize("limits", [{}, {"semantic_decision_max_calls": 2,
                                       "semantic_decision_max_total_input_tokens": 30000}])
def test_explicit_decision_configuration_retains_all_controls_and_original_resources(tmp_path, limits):
    options = {**DECISION, **limits, "semantic_decision_policy_artifact": Path(POLICY_PATH)}
    expected = {**DECISION, **limits}
    default = benchmark.config_for(tmp_path / "dataset", tmp_path / "output", tmp_path / "archive", "full")
    config = benchmark.config_for(tmp_path / "dataset", tmp_path / "output", tmp_path / "archive", "full", **options)
    wire = JobConfig.model_validate(config, extra="forbid").model_dump(mode="json")
    assert wire["agents"][0]["kwargs"] == {**default["agents"][0]["kwargs"], **expected}
    assert benchmark._prompt_options("full", **options) == {**DEFAULTS, **expected}
    for key in ("tasks", "environment", "verifier", "n_attempts", "n_concurrent_trials", "timeout_multiplier", "retry"):
        assert config[key] == default[key]
    assert wire["agents"][0]["override_timeout_sec"] == default["agents"][0]["override_timeout_sec"]
    assert wire["agents"][0]["override_setup_timeout_sec"] == 1800


@pytest.mark.parametrize("arm,options", [
    ("no-index", DECISION),
    ("full", {"semantic_decision_policy_artifact": POLICY_PATH}),
    ("full", {"semantic_decision_policy_sha256": POLICY_SHA}),
    ("full", {"semantic_decision_max_calls": 2}),
    ("full", {"semantic_decision_max_total_input_tokens": 1000}),
    ("full", {**DECISION, "semantic_decision_max_calls": True}),
    ("full", {**DECISION, "semantic_decision_max_total_input_tokens": 0}),
    ("full", {**DECISION, "semantic_slot_policy_revision": "closed-typed-identifiers@2"}),
    ("full", {**DECISION, "semantic_slot_policy_revision": "closed-typed-identifiers@2",
              "semantic_context_projection": True}),
])
def test_invalid_or_conflicting_decision_controls_refuse_before_preparation_or_adapter_io(tmp_path, monkeypatch, arm, options):
    def unexpected(*_args, **_kwargs):
        pytest.fail("invalid decision controls reached archive or process work")
    unread = SimpleNamespace(resolve=unexpected)
    monkeypatch.setattr(benchmark.subprocess, "run", unexpected)
    with pytest.raises(ValueError):
        benchmark.config_for(unread, tmp_path / "output", unread, arm, **options)
    with pytest.raises(ValueError):
        benchmark.prepare(dataset=unread, output=tmp_path / "output", archive=unread, arm=arm, **options)
    with pytest.raises(ValueError):
        adapter.FullSupervisorAgent(
            logs_dir=tmp_path / "logs", model_name="gpt-5.6-sol", runtime_archive=str(tmp_path / "missing"),
            arm=arm, **options)
    assert not (tmp_path / "output").exists() and not (tmp_path / "logs").exists()


def test_preparation_retains_child_local_policy_without_loading_or_staging_it(tmp_path, monkeypatch):
    from test.api.test_terminal_decision_archive import _decision_runtime_archive

    archive, _ = _decision_runtime_archive(tmp_path)
    dataset = _authored_prepare_dataset(tmp_path)
    output = tmp_path / "prepared"
    calls = []
    def dry(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout="Authored dry-run double.\n", stderr="")
    monkeypatch.setattr(benchmark.subprocess, "run", dry)
    result = benchmark.prepare(dataset=dataset, output=output, archive=archive, arm="full", **DECISION)
    config = json.loads((output / "config.json").read_text())
    assert result["prompt_comparison_options"] == {**DEFAULTS, **DECISION}
    assert config["agents"][0]["kwargs"]["semantic_decision_policy_artifact"] == POLICY_PATH
    assert result["provider_calls"] == 0 and result["benchmark_advantage_claimed"] is False
    assert len(calls) == 1 and calls[0][-1] == "--dry-run"
    assert not (output / "decision.json").exists()
    observation = benchmark._prompt_option_observation(result, config["agents"][0]["kwargs"])
    assert observation["prompt_comparison_config_unchanged"] is True


@pytest.mark.parametrize("change", ["policy", "calls", "total", "removed", "incomplete_pin"])
def test_decision_observation_detects_configuration_or_pin_changes(change):
    selected = {**DECISION, "semantic_decision_max_total_input_tokens": 30000}
    prepared = {"arm": "full", "prompt_comparison_options": {**DEFAULTS, **selected}}
    configured = dict(selected)
    if change == "policy":
        configured["semantic_decision_policy_sha256"] = "e" * 64
    elif change == "calls":
        configured["semantic_decision_max_calls"] = 2
    elif change == "total":
        configured["semantic_decision_max_total_input_tokens"] = 29000
    elif change == "removed":
        configured = {}
    else:
        prepared["prompt_comparison_options"].pop("semantic_decision_max_calls")
    observation = benchmark._prompt_option_observation(prepared, configured)
    assert observation["prompt_comparison_options_valid"] is True
    assert observation["prompt_comparison_config_unchanged"] is False


def test_host_cli_forwards_explicit_decision_controls(tmp_path, monkeypatch, capsys):
    calls = []
    def prepare(**kwargs):
        calls.append(kwargs)
        return {"prepared": False, "provider_calls": 0}
    monkeypatch.setattr(benchmark, "prepare", prepare)
    monkeypatch.setattr(sys, "argv", [
        "full-supervisor", "prepare", "--output", str(tmp_path / "output"),
        "--dataset", str(tmp_path / "dataset"), "--archive", str(tmp_path / "archive"),
        "--semantic-decision-policy-artifact", POLICY_PATH,
        "--semantic-decision-policy-sha256", POLICY_SHA,
        "--semantic-decision-max-calls", "2",
        "--semantic-decision-max-total-input-tokens", "30000",
    ])
    benchmark.main()
    assert json.loads(capsys.readouterr().out)["provider_calls"] == 0
    assert len(calls) == 1
    assert {key: calls[0][key] for key in DECISION} == {
        **DECISION, "semantic_decision_max_calls": 2, "semantic_decision_max_total_input_tokens": 30000,
    }


@pytest.mark.parametrize("selected", [False, True])
def test_harbor_run_forwards_policy_only_when_selected_without_uploading_it(tmp_path, monkeypatch, selected):
    archive = tmp_path / "archive"
    archive.mkdir()
    (archive / "manifest.json").write_text(json.dumps({"archive_sha256": "authored", "learned_requirements": []}))
    commands, uploads = [], []
    class Environment:
        async def upload_file(self, source, target):
            uploads.append(target)
        async def exec(self, **kwargs):
            commands.append(shlex.split(kwargs["command"]))
            return SimpleNamespace(stdout="", stderr="", return_code=0)
        async def download_file(self, source, target):
            raise FileNotFoundError(source)
    async def capture(*args, **kwargs):
        return {"status": "authored"}
    monkeypatch.setattr(adapter, "capture_public_inputs", capture)
    monkeypatch.setattr(adapter, "export_public_outputs", capture)
    options = {**DECISION, "semantic_decision_max_calls": 2,
               "semantic_decision_max_total_input_tokens": 30000} if selected else {}
    agent = adapter.FullSupervisorAgent(
        logs_dir=tmp_path / "logs", model_name="gpt-5.6-sol", runtime_archive=str(archive), arm="full", **options)
    context = AgentContext()
    asyncio.run(agent.run("Authored public instruction.", Environment(), context))
    assert len(commands) == 1
    command = commands[0]
    if selected:
        for key, value in options.items():
            flag = "--" + key.replace("_", "-")
            assert command.count(flag) == 1 and command[command.index(flag) + 1] == str(value)
    else:
        assert not any(argument.startswith("--semantic-decision-") for argument in command)
    assert POLICY_PATH not in uploads
    assert context.metadata["task_completed"] is False
