"""Pure native input keeps the full indexed arm and pins its deployed runtime.

Authored task/archive bytes and local process doubles qualify configuration and
preflight refusal. These checks invoke no provider, Docker or task verifier.
"""

from __future__ import annotations

import asyncio
import io
import json
from pathlib import Path
import shlex
import sys
import tarfile
from types import SimpleNamespace

import pytest
from harbor.models.agent.context import AgentContext
from harbor.models.job.config import JobConfig

from benchmarks.agent_supervisor.container_coding import full_supervisor_benchmark as benchmark
from benchmarks.agent_supervisor.container_coding import full_supervisor_harbor_agent as adapter
from test.api.test_benchmark_prompt_options import DEFAULTS
from test.api.test_terminal_decision_archive import _decision_runtime_archive
from test.api.test_terminal_timeout_configuration import (
    _authored_prepare_dataset, _timeout_runtime_archive,
)

NATIVE = {"semantic_native_input": True}


@pytest.mark.parametrize("arm", ["full", "no-index"])
def test_false_native_control_preserves_default_option_and_configuration_shapes(tmp_path, arm):
    assert benchmark._prompt_options(arm) == DEFAULTS
    assert benchmark._prompt_options(arm, semantic_native_input=False) == DEFAULTS
    arguments = (tmp_path / "dataset", tmp_path / "output", tmp_path / "archive", arm)
    ordinary = benchmark.config_for(*arguments)
    assert benchmark.config_for(*arguments, semantic_native_input=False) == ordinary
    assert "semantic_native_input" not in ordinary["agents"][0]["kwargs"]
    observation = benchmark._prompt_option_observation(
        {"arm": arm, "prompt_comparison_options": DEFAULTS}, ordinary["agents"][0]["kwargs"])
    assert observation["prompt_comparison_config_unchanged"] is True
    assert "semantic_native_input_requested" not in observation


def test_native_control_is_a_separate_closed_full_index_configuration(tmp_path):
    arguments = (tmp_path / "dataset", tmp_path / "output", tmp_path / "archive", "full")
    ordinary = benchmark.config_for(*arguments)
    configured = benchmark.config_for(*arguments, **NATIVE)
    wire = JobConfig.model_validate(configured, extra="forbid").model_dump(mode="json")
    assert wire["agents"][0]["kwargs"] == {**ordinary["agents"][0]["kwargs"], **NATIVE}
    assert benchmark._prompt_options("full", **NATIVE) == {**DEFAULTS, **NATIVE}
    for field in ("tasks", "environment", "verifier", "n_attempts", "n_concurrent_trials",
                  "timeout_multiplier", "retry"):
        assert configured[field] == ordinary[field]
    observed = benchmark._prompt_option_observation(
        {"arm": "full", "prompt_comparison_options": {**DEFAULTS, **NATIVE}}, wire["agents"][0]["kwargs"])
    assert observed["prompt_comparison_config_unchanged"] is True
    assert observed["semantic_native_input_requested"] is True


@pytest.mark.parametrize("arm,options", [
    ("no-index", NATIVE),
    ("full", {"semantic_native_input": 1}),
    ("full", {"semantic_native_input": "false"}),
    ("full", {"semantic_native_input": None}),
    ("full", {**NATIVE, "semantic_context_projection": True}),
    ("full", {**NATIVE, "semantic_slot_policy_revision": "closed-typed-identifiers@2"}),
    ("full", {**NATIVE, "semantic_decision_policy_artifact": "/child/policy.json",
              "semantic_decision_policy_sha256": "d" * 64}),
    ("full", {**NATIVE, "semantic_decision_nomination_artifact": "/child/nomination.json",
              "semantic_decision_nomination_sha256": "e" * 64}),
    ("full", {**NATIVE, "semantic_decision_nomination_source": "/host/nomination.json",
              "semantic_decision_nomination_sha256": "f" * 64}),
], ids=["arm", "integer", "string", "none", "projection", "codec", "policy", "nomination", "host-source"])
def test_invalid_native_control_refuses_before_preparation_or_adapter_io(tmp_path, monkeypatch, arm, options):
    def unexpected(*args, **kwargs):
        pytest.fail("invalid native control reached source/archive/process work")

    unread = SimpleNamespace(resolve=unexpected)
    monkeypatch.setattr(benchmark, "read_nomination_source", unexpected)
    monkeypatch.setattr(benchmark.subprocess, "run", unexpected)
    with pytest.raises(ValueError):
        benchmark.config_for(unread, tmp_path / "output", unread, arm, **options)
    with pytest.raises(ValueError):
        benchmark.prepare(dataset=unread, output=tmp_path / "output", archive=unread, arm=arm, **options)
    with pytest.raises(ValueError):
        adapter.FullSupervisorAgent(logs_dir=tmp_path / "logs", model_name="gpt-5.6-sol",
            runtime_archive=str(tmp_path / "missing"), arm=arm, **options)
    assert not (tmp_path / "output").exists() and not (tmp_path / "logs").exists()


@pytest.mark.parametrize("change", ["removed", "added", "false-pin", "mixed-pin", "invalid-config"])
def test_native_control_observation_detects_changed_configuration_or_nonclosed_pin(change):
    prepared = {"arm": "full", "prompt_comparison_options": {**DEFAULTS, **NATIVE}}
    configured = dict(NATIVE)
    if change == "removed":
        configured = {}
    elif change == "added":
        prepared["prompt_comparison_options"] = dict(DEFAULTS)
    elif change == "false-pin":
        prepared["prompt_comparison_options"]["semantic_native_input"] = False
    elif change == "mixed-pin":
        prepared["prompt_comparison_options"]["semantic_decision_max_calls"] = 3
    else:
        configured["semantic_native_input"] = 1
    observation = benchmark._prompt_option_observation(prepared, configured)
    assert observation["prompt_comparison_config_unchanged"] is False
    assert observation["prompt_comparison_options_valid"] is (change != "invalid-config")


def test_cli_forwards_explicit_native_control(tmp_path, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(benchmark, "prepare", lambda **kwargs:
                        calls.append(kwargs) or {"provider_calls": 0})
    monkeypatch.setattr(sys, "argv", ["full-supervisor", "prepare", "--output", str(tmp_path / "output"),
        "--dataset", str(tmp_path / "dataset"), "--archive", str(tmp_path / "archive"),
        "--semantic-native-input"])
    benchmark.main()
    assert json.loads(capsys.readouterr().out) == {"provider_calls": 0}
    assert len(calls) == 1 and calls[0]["semantic_native_input"] is True


def _prepared_native_trial(tmp_path, monkeypatch):
    archive, _ = _decision_runtime_archive(tmp_path)
    dataset = _authored_prepare_dataset(tmp_path)
    output = tmp_path / "prepared"
    calls = []

    def dry(command, **kwargs):
        assert command[-1] == "--dry-run"
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout="Authored dry run.\n", stderr="")

    monkeypatch.setattr(benchmark.subprocess, "run", dry)
    prepared = benchmark.prepare(dataset=dataset, output=output, archive=archive, arm="full", **NATIVE)
    assert len(calls) == 1
    return output, prepared


def test_native_preparation_pins_current_router_and_runtime_under_distinct_receipt_field(tmp_path, monkeypatch):
    output, prepared = _prepared_native_trial(tmp_path, monkeypatch)
    expected = {name: benchmark._hash(source) for name, source in benchmark._decision_runtime_sources().items()}
    assert prepared["runtime_native_input_source_sha256"] == expected
    assert benchmark.DECISION_RUNTIME_ARCHIVE_PREFIX + "router_implementation_runner.py" in expected
    assert "runtime_decision_source_sha256" not in prepared
    assert prepared["semantic_native_input_requested"] is True
    assert prepared["prompt_comparison_options"] == {**DEFAULTS, **NATIVE}
    config = json.loads((output / "config.json").read_text())
    assert config["agents"][0]["kwargs"]["semantic_native_input"] is True
    assert prepared["provider_calls"] == 0 and prepared["benchmark_advantage_claimed"] is False


def test_native_preparation_refuses_deadline_only_archive_before_output(tmp_path, monkeypatch):
    archive, _ = _timeout_runtime_archive(tmp_path)
    dataset = _authored_prepare_dataset(tmp_path)
    output = tmp_path / "output"
    monkeypatch.setattr(benchmark.subprocess, "run", lambda *args, **kwargs:
                        pytest.fail("missing native capability reached Harbor dry-run"))
    with pytest.raises(ValueError, match="selected decision"):
        benchmark.prepare(dataset=dataset, output=output, archive=archive, arm="full", **NATIVE)
    assert not output.exists()


@pytest.mark.parametrize("fault", ["missing", "incomplete", "changed", "local-source"])
def test_native_execute_refuses_changed_or_missing_runtime_pins_before_invocation(tmp_path, monkeypatch, fault):
    output, prepared = _prepared_native_trial(tmp_path, monkeypatch)
    pins = prepared["runtime_native_input_source_sha256"]
    target = benchmark.DECISION_RUNTIME_ARCHIVE_PREFIX + "router_implementation_runner.py"
    if fault == "missing":
        prepared.pop("runtime_native_input_source_sha256")
    elif fault == "incomplete":
        pins.pop(target)
    elif fault == "changed":
        pins[target] = "0" * 64
    else:
        original = benchmark._hash
        selected = benchmark._decision_runtime_sources()[target]
        monkeypatch.setattr(benchmark, "_hash", lambda path:
                            "0" * 64 if Path(path) == selected else original(path))
    benchmark._json(output / "preparation.json", prepared)
    monkeypatch.setattr(benchmark.subprocess, "run", lambda *args, **kwargs:
                        pytest.fail("changed native input pins reached invocation"))
    with pytest.raises(ValueError, match="native input source pins"):
        benchmark.execute(output)
    assert not (output / "invocation.json").exists()


def test_native_execute_replays_actual_router_archive_member_even_if_outer_hashes_repaired(tmp_path, monkeypatch):
    output, prepared = _prepared_native_trial(tmp_path, monkeypatch)
    archive = Path(prepared["archive"])
    manifest = json.loads((archive / "manifest.json").read_text())
    target = benchmark.DECISION_RUNTIME_ARCHIVE_PREFIX + "router_implementation_runner.py"
    members = []
    with tarfile.open(archive / "runtime.tar.gz", "r:gz") as original:
        for member in original:
            if member.name != target:
                members.append((member.name, original.extractfile(member).read()))
    with tarfile.open(archive / "runtime.tar.gz", "w:gz") as bundle:
        for name, raw in members:
            member = tarfile.TarInfo(name)
            member.size, member.mode = len(raw), 0o644
            bundle.addfile(member, io.BytesIO(raw))
    manifest["archive_sha256"] = benchmark._hash(archive / "runtime.tar.gz")
    benchmark._json(archive / "manifest.json", manifest)
    prepared.update(archive_sha256=manifest["archive_sha256"],
                    manifest_sha256=benchmark._hash(archive / "manifest.json"))
    benchmark._json(output / "preparation.json", prepared)
    monkeypatch.setattr(benchmark.subprocess, "run", lambda *args, **kwargs:
                        pytest.fail("missing router archive member reached invocation"))
    with pytest.raises(ValueError, match="selected decision"):
        benchmark.execute(output)
    assert not (output / "invocation.json").exists()


@pytest.mark.parametrize("selected", [False, True])
def test_harbor_driver_forwards_native_switch_only_when_explicit_and_keeps_indexed_arm(tmp_path, monkeypatch, selected):
    archive = tmp_path / "archive"
    archive.mkdir()
    (archive / "manifest.json").write_text(json.dumps({"archive_sha256": "authored", "learned_requirements": []}))
    commands = []

    class Environment:
        async def upload_file(self, source, target):
            pass

        async def exec(self, **kwargs):
            commands.append(shlex.split(kwargs["command"]))
            return SimpleNamespace(stdout="", stderr="", return_code=0)

        async def download_file(self, source, target):
            raise FileNotFoundError(source)

    async def capture(*args, **kwargs):
        return {"status": "authored"}

    monkeypatch.setattr(adapter, "capture_public_inputs", capture)
    monkeypatch.setattr(adapter, "export_public_outputs", capture)
    agent = adapter.FullSupervisorAgent(logs_dir=tmp_path / "logs", model_name="gpt-5.6-sol",
        runtime_archive=str(archive), arm="full", semantic_native_input=selected)
    context = AgentContext()
    asyncio.run(agent.run("Authored public instruction.", Environment(), context))
    assert len(commands) == 1
    command = commands[0]
    assert command.count("--semantic-native-input") == int(selected)
    assert command[command.index("--arm") + 1] == "full"
    assert not any(argument.startswith("--semantic-decision-") for argument in command)
    assert "--semantic-context-projection" not in command
    assert "--semantic-slot-policy-revision" not in command
    assert context.metadata.get("semantic_native_input_requested") is (True if selected else None)
    assert context.metadata["task_completed"] is False
