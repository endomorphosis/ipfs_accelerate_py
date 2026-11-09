"""Nomination forwarding preserves separate owner binding and legacy shapes.

Authored archives and mock process/Harbor environments do not stage a
nomination, invoke a provider, or establish benchmark success or savings.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import shlex
import sys
from types import SimpleNamespace

import pytest
from benchmarks.agent_supervisor.container_coding.test_terminal_deployment import _supervisor_manifest
from harbor.models.agent.context import AgentContext
from harbor.models.job.config import JobConfig

from benchmarks.agent_supervisor.container_coding import full_supervisor_benchmark as benchmark
from benchmarks.agent_supervisor.container_coding import full_supervisor_harbor_agent as adapter
from test.api.test_benchmark_prompt_options import DEFAULTS
from test.api.test_current_decision_configuration import _archive as _current_decision_archive
from test.api.test_terminal_decision_configuration import DECISION
from test.api.test_current_terminal_archive_fixtures import (
    _authored_prepare_dataset, _timeout_runtime_archive,
)

NOMINATION_PATH = "/opt/ipfs-supervisor/task-policy/nomination.json"
NOMINATION_SHA = "e" * 64
NOMINATION = {
    "semantic_decision_nomination_artifact": NOMINATION_PATH,
    "semantic_decision_nomination_sha256": NOMINATION_SHA,
    "semantic_decision_max_calls": 3,
    "semantic_decision_max_total_input_tokens": None,
    "disable_semantic_typed_nomination": False,
}


@pytest.mark.parametrize("limits", [{}, {"semantic_decision_max_calls": 2,
                                       "semantic_decision_max_total_input_tokens": 30000}])
def test_nomination_configuration_has_exact_variant_shape_and_preserves_legacy(tmp_path, limits):
    expected = {**NOMINATION, **limits, "disable_semantic_typed_nomination": False}
    options = {**expected, "semantic_decision_nomination_artifact": Path(NOMINATION_PATH)}
    assert benchmark._prompt_options("full") == DEFAULTS
    assert benchmark._prompt_options("full", **DECISION) == {**DEFAULTS, **DECISION}
    selected = benchmark._prompt_options("full", **options)
    assert selected == {**DEFAULTS, **expected} and len(selected) == 10
    assert not {"semantic_decision_policy_artifact", "semantic_decision_policy_sha256"} & set(selected)
    default = benchmark.config_for(tmp_path / "dataset", tmp_path / "output", tmp_path / "archive", "full")
    config = benchmark.config_for(tmp_path / "dataset", tmp_path / "output", tmp_path / "archive", "full", **options)
    wire = JobConfig.model_validate(config, extra="forbid").model_dump(mode="json")
    assert wire["agents"][0]["kwargs"] == {**default["agents"][0]["kwargs"], **expected}
    for key in ("tasks", "environment", "verifier", "n_attempts", "n_concurrent_trials", "timeout_multiplier", "retry"):
        assert config[key] == default[key]
    assert not any("task_id" in key or "task_key" in key for key in wire["agents"][0]["kwargs"])


@pytest.mark.parametrize("arm,options", [
    ("no-index", NOMINATION),
    ("full", {"semantic_decision_nomination_artifact": NOMINATION_PATH}),
    ("full", {"semantic_decision_nomination_sha256": NOMINATION_SHA}),
    ("full", {**NOMINATION, **DECISION}),
    ("full", {**NOMINATION, "semantic_decision_policy_artifact": DECISION["semantic_decision_policy_artifact"]}),
    ("full", {**NOMINATION, "semantic_decision_nomination_artifact": "relative/nomination.json"}),
    ("full", {**NOMINATION, "semantic_decision_max_calls": 0}),
    ("full", {**NOMINATION, "semantic_slot_policy_revision": "closed-typed-identifiers@2"}),
    ("full", {**NOMINATION, "semantic_slot_policy_revision": "closed-typed-identifiers@2",
              "semantic_context_projection": True}),
])
def test_invalid_nomination_or_mixed_variants_refuse_before_preparation_or_adapter_io(tmp_path, monkeypatch, arm, options):
    def unexpected(*args, **kwargs):
        pytest.fail("invalid nomination options reached filesystem or process work")
    unread = SimpleNamespace(resolve=unexpected)
    monkeypatch.setattr(benchmark.subprocess, "run", unexpected)
    with pytest.raises(ValueError):
        benchmark.config_for(unread, tmp_path / "output", unread, arm, **options)
    with pytest.raises(ValueError):
        benchmark.prepare(dataset=unread, output=tmp_path / "output", archive=unread, arm=arm, **options)
    with pytest.raises(ValueError):
        adapter.FullSupervisorAgent(
            logs_dir=tmp_path / "logs", model_name=adapter.resolve_provider_profile()["model"], runtime_archive=str(tmp_path / "missing"),
            arm=arm, **options)
    assert not (tmp_path / "output").exists() and not (tmp_path / "logs").exists()


@pytest.mark.parametrize("change", ["incomplete_pin", "mixed_pin", "variant", "sha", "budget"])
def test_nomination_observation_detects_incomplete_mixed_or_changed_pins(change):
    prepared = {"arm": "full", "prompt_comparison_options": {**DEFAULTS, **NOMINATION}}
    configured = dict(NOMINATION)
    if change == "incomplete_pin":
        prepared["prompt_comparison_options"].pop("semantic_decision_nomination_sha256")
    elif change == "mixed_pin":
        prepared["prompt_comparison_options"].update(DECISION)
    elif change == "variant":
        configured = dict(DECISION)
    elif change == "sha":
        configured["semantic_decision_nomination_sha256"] = "f" * 64
    else:
        configured["semantic_decision_max_calls"] = 2
    observation = benchmark._prompt_option_observation(prepared, configured)
    assert observation["prompt_comparison_options_valid"] is True
    assert observation["prompt_comparison_config_unchanged"] is False


def test_host_cli_forwards_nomination_without_exposing_owner_task_identity(tmp_path, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(benchmark, "prepare", lambda **kwargs: calls.append(kwargs) or {"provider_calls": 0})
    monkeypatch.setattr(sys, "argv", [
        "full-supervisor", "prepare", "--output", str(tmp_path / "output"),
        "--dataset", str(tmp_path / "dataset"), "--archive", str(tmp_path / "archive"),
        "--semantic-decision-nomination-artifact", NOMINATION_PATH,
        "--semantic-decision-nomination-sha256", NOMINATION_SHA,
        "--semantic-decision-max-calls", "2",
        "--semantic-decision-max-total-input-tokens", "30000",
    ])
    benchmark.main()
    assert json.loads(capsys.readouterr().out)["provider_calls"] == 0
    assert len(calls) == 1
    assert {key: calls[0][key] for key in NOMINATION} == {
        **NOMINATION, "semantic_decision_max_calls": 2, "semantic_decision_max_total_input_tokens": 30000,
    }
    assert calls[0].get("semantic_decision_policy_artifact") is None
    assert calls[0].get("semantic_decision_policy_sha256") is None
    assert not any("task_id" in key or "task_key" in key for key in calls[0])


def test_nomination_preparation_retains_owner_capability_pins_without_loading_or_staging_input(tmp_path, monkeypatch):
    from test.api.test_terminal_decision_archive import _decision_runtime_archive

    archive, _ = _decision_runtime_archive(tmp_path)
    dataset = _authored_prepare_dataset(tmp_path)
    output = tmp_path / "prepared"
    calls = []
    def dry(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout="Authored dry run.", stderr="")
    monkeypatch.setattr(benchmark.subprocess, "run", dry)
    prepared = benchmark.prepare(dataset=dataset, output=output, archive=archive, arm="full", **NOMINATION)
    config = json.loads((output / "config.json").read_text())
    assert prepared["prompt_comparison_options"] == {**DEFAULTS, **NOMINATION}
    assert len(calls) == 1 and calls[0][-1] == "--dry-run"
    assert benchmark.DECISION_RUNTIME_ARCHIVE_PREFIX + "semantic_decision_owner.py" in prepared["runtime_decision_source_sha256"]
    assert prepared["provider_calls"] == 0 and prepared["benchmark_advantage_claimed"] is False
    assert not (output / "nomination.json").exists()
    assert benchmark._prompt_option_observation(prepared, config["agents"][0]["kwargs"])["prompt_comparison_config_unchanged"] is True


def test_nomination_preparation_refuses_old_archive_before_process(tmp_path, monkeypatch):
    archive, _ = _timeout_runtime_archive(tmp_path)
    dataset = _authored_prepare_dataset(tmp_path)
    output = tmp_path / "output"
    monkeypatch.setattr(benchmark.subprocess, "run", lambda *args, **kwargs:
                        pytest.fail("missing nomination owner capability reached a process"))
    with pytest.raises(ValueError, match="selected decision"):
        benchmark.prepare(dataset=dataset, output=output, archive=archive, arm="full", **NOMINATION)
    assert not output.exists()


def test_nomination_execute_requires_current_owner_source_pin(tmp_path, monkeypatch):
    from test.api.test_terminal_decision_archive import _decision_runtime_archive

    archive, _ = _decision_runtime_archive(tmp_path)
    dataset = _authored_prepare_dataset(tmp_path)
    output = tmp_path / "prepared"
    monkeypatch.setattr(benchmark.subprocess, "run", lambda *args, **kwargs:
                        SimpleNamespace(returncode=0, stdout="Authored dry run.", stderr=""))
    prepared = benchmark.prepare(dataset=dataset, output=output, archive=archive, arm="full", **NOMINATION)
    prepared["runtime_decision_source_sha256"].pop(benchmark.DECISION_RUNTIME_ARCHIVE_PREFIX + "semantic_decision_owner.py")
    benchmark._json(output / "preparation.json", prepared)
    monkeypatch.setattr(benchmark.subprocess, "run", lambda *args, **kwargs:
                        pytest.fail("missing nomination source pin reached process invocation"))
    with pytest.raises(ValueError, match="decision source pins"):
        benchmark.execute(output)
    assert not (output / "invocation.json").exists()


def test_harbor_forwards_nomination_pair_and_budgets_without_upload_or_task_override(tmp_path, monkeypatch):
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
    options = {**NOMINATION, "semantic_decision_max_calls": 2, "semantic_decision_max_total_input_tokens": 30000}
    archive, _ = _current_decision_archive(tmp_path, options)
    agent = adapter.FullSupervisorAgent(
        logs_dir=tmp_path / "logs", model_name=adapter.resolve_provider_profile()["model"], runtime_archive=str(archive), arm="full", **options)
    context = AgentContext()
    asyncio.run(agent.run("Authored public instruction.", Environment(), context))
    assert len(commands) == 1
    command = commands[0]
    for key, value in options.items():
        flag = "--" + key.replace("_", "-")
        if key == "disable_semantic_typed_nomination":
            assert value is False and flag not in command
        else:
            assert command.count(flag) == 1 and command[command.index(flag) + 1] == str(value)
    assert not any(argument.startswith("--semantic-decision-policy-") for argument in command)
    assert "--task-id" not in command and "--task-key" not in command
    assert NOMINATION_PATH not in uploads
    assert context.metadata["task_completed"] is False
