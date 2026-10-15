"""Host configurations retain requested prompt controls without claiming savings.

Authored archives and retained trial records exercise the real Harbor config
schema and host preparation/collection boundaries. The dry-run is replaced by
an owned process double; no provider, Docker task, or verifier is executed.
"""

from __future__ import annotations

import builtins
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from benchmarks.agent_supervisor.container_coding import (
    full_supervisor_benchmark as supervisor,
)
from benchmarks.agent_supervisor.container_coding import (
    native_codex_baseline as native,
)
from harbor.models.job.config import JobConfig

from test.api.test_terminal_timeout_configuration import (
    _authored_prepare_dataset,
    _timeout_runtime_archive,
)

DESCRIPTOR = "/opt/ipfs-supervisor/prompt-tokenizer/descriptor.json"
DIGEST = "a" * 64
V2 = "closed-typed-identifiers@2"
DEFAULTS = {
    "prompt_tokenizer_artifact": None,
    "prompt_tokenizer_sha256": None,
    "max_input_tokens": None,
    "semantic_slot_policy_revision": "legacy@1",
    "semantic_context_projection": False,
}
TOKENIZER = {
    "prompt_tokenizer_artifact": DESCRIPTOR,
    "prompt_tokenizer_sha256": DIGEST,
    "max_input_tokens": 8192,
}


def _canonical(options):
    return {**DEFAULTS, **{
        key: str(value) if key == "prompt_tokenizer_artifact" and value is not None else value
        for key, value in options.items()
    }}


def _config(tmp_path, arm="full", **options):
    return supervisor.config_for(tmp_path / "dataset", tmp_path / "output",
                                 tmp_path / "archive", arm, **options)


@pytest.mark.parametrize("arm", ["full", "no-index"])
def test_default_harbor_configuration_preserves_existing_adapter_kwargs(tmp_path, arm):
    config = _config(tmp_path, arm)
    wire = JobConfig.model_validate(config, extra="forbid").model_dump(mode="json")
    from benchmarks.agent_supervisor.container_coding.benchmark_resource_profile import execution_budget
    assert wire["agents"][0]["kwargs"] == {
        "runtime_archive": str(tmp_path / "archive"), "arm": arm,
        "model_revision": "",
    }
    assert wire["agents"][0]["override_timeout_sec"] == execution_budget()["harbor_seconds"]


@pytest.mark.parametrize("arm,options", [
    ("full", {"semantic_slot_policy_revision": V2}),
    ("full", {"semantic_slot_policy_revision": V2, "semantic_context_projection": True}),
    ("full", {**TOKENIZER, "prompt_tokenizer_artifact": Path(DESCRIPTOR)}),
    ("no-index", TOKENIZER),
    ("full", {"max_input_tokens": 8192}),
    ("no-index", {"max_input_tokens": 8192}),
])
def test_explicit_options_reach_real_harbor_schema_without_changing_task_or_resources(tmp_path, arm, options):
    default = _config(tmp_path, arm)
    config = _config(tmp_path, arm, **options)
    wire = JobConfig.model_validate(config, extra="forbid").model_dump(mode="json")
    expected = {**default["agents"][0]["kwargs"], **{
        key: str(value) if isinstance(value, Path) else value for key, value in options.items()
    }}
    assert wire["agents"][0]["kwargs"] == expected
    for key in ("tasks", "environment", "verifier", "n_attempts", "n_concurrent_trials",
                "timeout_multiplier", "retry"):
        assert config[key] == default[key]
    assert wire["agents"][0]["override_timeout_sec"] == default["agents"][0]["override_timeout_sec"]
    assert wire["agents"][0]["override_setup_timeout_sec"] == 1800


INVALID = [
    {"semantic_slot_policy_revision": True},
    {"semantic_slot_policy_revision": "unknown"},
    {"semantic_context_projection": 1},
    {"semantic_context_projection": "true"},
    {"semantic_context_projection": True},
    {"prompt_tokenizer_artifact": DESCRIPTOR},
    {"prompt_tokenizer_sha256": DIGEST},
    {**TOKENIZER, "max_input_tokens": True},
    {**TOKENIZER, "max_input_tokens": 0},
    {**TOKENIZER, "max_input_tokens": "8192"},
    {**TOKENIZER, "prompt_tokenizer_artifact": "relative/descriptor.json"},
    {**TOKENIZER, "prompt_tokenizer_artifact": "/opt/../descriptor.json"},
    {**TOKENIZER, "prompt_tokenizer_sha256": "A" * 64},
    {**TOKENIZER, "prompt_tokenizer_sha256": "a" * 63},
]


@pytest.mark.parametrize("options", INVALID)
def test_invalid_prompt_options_refuse_before_prepare_reads_paths_or_imports_harbor(tmp_path, monkeypatch, options):
    def unexpected(*_args, **_kwargs):
        raise AssertionError("invalid options reached filesystem or process work")

    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "harbor" or name.startswith("harbor."):
            raise AssertionError("invalid options reached Harbor import")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(supervisor.subprocess, "run", unexpected)
    unread = SimpleNamespace(resolve=unexpected)
    with pytest.raises(ValueError):
        supervisor.prepare(dataset=unread, output=tmp_path / "output", archive=unread,
                           arm="full", **options)
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("options", [
    {"semantic_slot_policy_revision": V2},
    {"semantic_slot_policy_revision": V2, "semantic_context_projection": True},
])
def test_no_index_semantic_options_refuse_before_preparation_io(tmp_path, monkeypatch, options):
    def unexpected(*_args, **_kwargs):
        raise AssertionError("no-index semantic options reached filesystem or process work")

    unread = SimpleNamespace(resolve=unexpected)
    monkeypatch.setattr(supervisor.subprocess, "run", unexpected)
    with pytest.raises(ValueError, match="full"):
        supervisor.config_for(unread, tmp_path / "output", unread, "no-index", **options)
    with pytest.raises(ValueError, match="full"):
        supervisor.prepare(dataset=unread, output=tmp_path / "output", archive=unread,
                           arm="no-index", **options)
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("arm,options", [
    ("full", {}),
    ("full", {**TOKENIZER, "semantic_slot_policy_revision": V2,
              "semantic_context_projection": True}),
    ("no-index", TOKENIZER),
])
def test_prepare_retains_canonical_requested_options_and_child_local_paths(tmp_path, monkeypatch, arm, options):
    archive, _ = _timeout_runtime_archive(tmp_path)
    dataset = _authored_prepare_dataset(tmp_path)
    output = tmp_path / "prepared"
    calls = []

    def dry(command, **_kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout="Authored dry-run double.\n", stderr="")

    monkeypatch.setattr(supervisor.subprocess, "run", dry)
    result = supervisor.prepare(dataset=dataset, output=output, archive=archive, arm=arm, **options)
    assert len(calls) == 1 and calls[0][-1] == "--dry-run"
    assert result["prompt_comparison_options"] == _canonical(options)
    assert result["provider_calls"] == 0
    assert result["benchmark_advantage_claimed"] is False
    prepared = json.loads((output / "preparation.json").read_text())
    config = json.loads((output / "config.json").read_text())
    assert prepared["prompt_comparison_options"] == _canonical(options)
    observation = supervisor._prompt_option_observation(prepared, config["agents"][0]["kwargs"])
    assert observation["prompt_comparison_options"] == _canonical(options)
    assert observation["prompt_comparison_options_valid"] is True
    assert observation["prompt_comparison_basis"] == "preparation"
    assert observation["prompt_comparison_config_unchanged"] is True


def _retained_trial(tmp_path, *, options, pin=True):
    # Authored retained task using the current resource-profile configuration.
    dataset = _authored_prepare_dataset(tmp_path)
    task = dataset / native.TASK
    output = tmp_path / "trial"
    output.mkdir()
    config = supervisor.config_for(dataset, output, tmp_path / "archive", "full", **options)
    native._json(output / "config.json", config)
    prepared = {
        "prepared": True, "dataset": str(dataset), "arm": "full",
        "config_sha256": native._hash(output / "config.json"),
        "task_input_sha256": native._task_hashes(task),
        "collector_source_sha256": native._hash(Path(supervisor.__file__)),
        "native_adapter_source_sha256": {}, "command": ["authored-harbor"],
    }
    if pin:
        prepared["prompt_comparison_options"] = _canonical(options)
    native._json(output / "preparation.json", prepared)
    record = {
        "trial_name": "authored-trial", "task_name": task.name,
        "config": {"task": {"path": str(task)}, "agent": config["agents"][0],
                   "verifier": {"disable": False}},
        "verifier_result": {"rewards": {"reward": 0}},
    }
    job = Path(config["jobs_dir"]) / config["job_name"]
    trial = job / "authored-trial"
    trial.mkdir(parents=True)
    native._json(trial / "result.json", record)
    native._json(job / "result.json", {"stats": {}})
    return output, prepared, record, trial


@pytest.mark.parametrize("pin", [True, False])
def test_collection_preserves_advanced_retained_options_and_marks_independent_pin_status(tmp_path, pin):
    options = {**TOKENIZER, "semantic_slot_policy_revision": V2, "semantic_context_projection": True}
    output, _, _, _ = _retained_trial(tmp_path, options=options, pin=pin)
    original = (output / "preparation.json").read_bytes()
    result = supervisor.collect(output)
    for observed in (result, result["trials"][0]):
        assert observed["prompt_comparison_options"] == _canonical(options)
        assert observed["prompt_comparison_options_valid"] is True
        assert observed["prompt_comparison_basis"] == (
            "preparation" if pin else "retained_configuration_without_preparation_pin")
        assert observed["prompt_comparison_config_unchanged"] is (True if pin else None)
    assert result["benchmark_advantage_claimed"] is False
    assert (output / "preparation.json").read_bytes() == original


def test_collection_distinguishes_changed_trial_options_from_unchanged_prepared_configuration(tmp_path):
    output, _, record, trial = _retained_trial(tmp_path, options=TOKENIZER)
    record["config"]["agent"]["kwargs"]["max_input_tokens"] = 4096
    native._json(trial / "result.json", record)
    result = supervisor.collect(output)
    assert result["prompt_comparison_config_unchanged"] is True
    observed = result["trials"][0]
    assert observed["prompt_comparison_options_valid"] is True
    assert observed["prompt_comparison_config_unchanged"] is False
    assert observed["prompt_comparison_options"]["max_input_tokens"] == 4096


def test_malformed_trial_options_are_reported_invalid_without_independent_selection_claim(tmp_path):
    output, _, record, trial = _retained_trial(tmp_path, options=TOKENIZER)
    record["config"]["agent"]["kwargs"]["max_input_tokens"] = True
    native._json(trial / "result.json", record)
    result = supervisor.collect(output)
    observed = result["trials"][0]
    assert observed["prompt_comparison_options_valid"] is False
    assert observed["prompt_comparison_config_unchanged"] is False


@pytest.mark.parametrize("missing", ["config", "agent", "kwargs"])
def test_absent_trial_configuration_cannot_confirm_pinned_default_options(tmp_path, missing):
    output, _, record, trial = _retained_trial(tmp_path, options={})
    if missing == "config":
        record.pop("config")
    elif missing == "agent":
        record["config"].pop("agent")
    else:
        record["config"]["agent"].pop("kwargs")
    native._json(trial / "result.json", record)
    result = supervisor.collect(output)
    assert result["prompt_comparison_config_unchanged"] is True
    observed = result["trials"][0]
    assert observed["prompt_comparison_options"] is None
    assert observed["prompt_comparison_options_valid"] is None
    assert observed["prompt_comparison_config_unchanged"] is None
    assert observed["prompt_comparison_basis"] == "unavailable_configuration"


@pytest.mark.parametrize("malformed", [[], "", 0])
def test_non_mapping_trial_kwargs_cannot_be_coerced_into_valid_default_options(tmp_path, malformed):
    output, _, record, trial = _retained_trial(tmp_path, options={})
    record["config"]["agent"]["kwargs"] = malformed
    native._json(trial / "result.json", record)
    result = supervisor.collect(output)
    observed = result["trials"][0]
    assert observed["prompt_comparison_options"] is None
    assert observed["prompt_comparison_options_valid"] is False
    assert observed["prompt_comparison_config_unchanged"] is False
    assert observed["prompt_comparison_basis"] == "unavailable_configuration"


def test_execution_refuses_missing_actual_kwargs_even_when_config_hash_is_updated(tmp_path, monkeypatch):
    output, prepared, _, _ = _retained_trial(tmp_path, options={})
    config = json.loads((output / "config.json").read_text())
    config["agents"][0].pop("kwargs")
    native._json(output / "config.json", config)
    prepared["config_sha256"] = native._hash(output / "config.json")
    native._json(output / "preparation.json", prepared)
    monkeypatch.setattr(supervisor.subprocess, "run", lambda *_args, **_kwargs: pytest.fail("Harbor launched"))
    with pytest.raises(ValueError, match="prompt"):
        supervisor.execute(output)
    assert not (output / "invocation.json").exists()


@pytest.mark.parametrize("change", ["value", "missing_key"])
def test_execution_refuses_changed_preparation_selection_before_launch_or_runtime_reads(tmp_path, monkeypatch, change):
    output, prepared, _, _ = _retained_trial(tmp_path, options=TOKENIZER)
    if change == "value":
        prepared["prompt_comparison_options"]["max_input_tokens"] = 4096
    else:
        prepared["prompt_comparison_options"].pop("max_input_tokens")
    native._json(output / "preparation.json", prepared)
    monkeypatch.setattr(supervisor.subprocess, "run", lambda *_args, **_kwargs: pytest.fail("Harbor launched"))
    with pytest.raises(ValueError, match="prompt"):
        supervisor.execute(output)
    assert not (output / "invocation.json").exists()


@pytest.mark.parametrize("options,arguments", [
    ({}, []),
    ({"semantic_slot_policy_revision": V2, "semantic_context_projection": True},
     ["--semantic-slot-policy-revision", V2, "--semantic-context-projection"]),
    (TOKENIZER, ["--prompt-tokenizer-artifact", DESCRIPTOR, "--prompt-tokenizer-sha256", DIGEST,
                 "--max-input-tokens", "8192"]),
])
def test_prepare_cli_forwards_explicit_prompt_options_to_host_preparation(tmp_path, monkeypatch, capsys, options, arguments):
    observed = {}
    monkeypatch.setattr(supervisor, "prepare", lambda **kwargs: observed.update(kwargs) or {"prepared": True})
    monkeypatch.setattr(sys, "argv", ["benchmark", "prepare", "--dataset", str(tmp_path / "dataset"),
                                     "--output", str(tmp_path / "output"), "--archive", str(tmp_path / "archive"),
                                     *arguments])
    supervisor.main()
    selected = {key: observed.get(key, value) for key, value in DEFAULTS.items()}
    assert _canonical(selected) == _canonical(options)
    assert json.loads(capsys.readouterr().out) == {"prepared": True}


@pytest.mark.parametrize("operation", ["execute", "collect"])
def test_existing_preparation_cli_operations_do_not_override_retained_prompt_selection(tmp_path, monkeypatch, capsys, operation):
    observed = []
    monkeypatch.setattr(supervisor, operation, lambda output: observed.append(output) or {"operation": operation})
    monkeypatch.setattr(sys, "argv", ["benchmark", operation, "--output", str(tmp_path / "output"),
                                     "--semantic-slot-policy-revision", V2, "--semantic-context-projection"])
    supervisor.main()
    assert observed == [tmp_path / "output"]
    assert json.loads(capsys.readouterr().out) == {"operation": operation}
