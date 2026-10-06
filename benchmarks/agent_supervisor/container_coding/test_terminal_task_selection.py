"""Task selection and immutable runner binding without Docker or model calls."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.agent_supervisor.container_coding import benchmark_controls
from benchmarks.agent_supervisor.container_coding import full_supervisor_benchmark as supervisor
from benchmarks.agent_supervisor.container_coding import native_codex_baseline as native


SELECTED = "another-coding-task"


def _task(dataset, name):
    task = dataset / name
    task.mkdir(parents=True)
    (task / "instruction.md").write_text("Create the requested example.\n")
    (task / "task.toml").write_text('version = "1.0"\n')
    (task / "environment").mkdir()
    (task / "tests").mkdir()
    # Authored test data, never the benchmark's withheld verifier.
    (task / "environment/Dockerfile").write_text("FROM scratch\n")
    return task


def _prepared(tmp_path, runner, *, task_name=SELECTED):
    dataset, output, archive = (tmp_path / name for name in ("dataset", "output", "archive"))
    _task(dataset, native.TASK)
    _task(dataset, SELECTED)
    output.mkdir()
    archive.mkdir()
    (archive / "runtime.tar.gz").write_bytes(b"authored archive fixture")
    manifest = {"archive_sha256": native._hash(archive / "runtime.tar.gz"), "codex_version": native.CLI_VERSION}
    if runner is not native:
        from benchmarks.agent_supervisor.container_coding.test_terminal_deployment import _supervisor_manifest
        manifest = _supervisor_manifest(manifest)
    native._json(archive / "manifest.json", manifest)
    if runner is native:
        config = native.config_for(dataset, output, task_name=task_name)
    else:
        config = supervisor.config_for(dataset, output, archive, "full", task_name=task_name)
    native._json(output / "config.json", config)
    hashes = native._task_hashes(dataset / task_name)
    prepared = dict(prepared=True, task=task_name, dataset=str(dataset), arm="full",
        task_input_sha256=hashes, config_sha256=native._hash(output / "config.json"),
        command=["authored-harbor", "run", "--config", str(output / "config.json"), "--yes"],
        collector_source_sha256=native._hash(Path(native.__file__)),
        controls_source_sha256=native._hash(Path(benchmark_controls.__file__)),
        provider_profile_source_sha256=native._hash(Path(native.benchmark_provider_profile.__file__)),
        native_adapter_source_sha256={}, host_source_sha256={}, archive=str(archive),
        archive_sha256=native._hash(archive / "runtime.tar.gz"),
        manifest_sha256=native._hash(archive / "manifest.json"),
        comparison_controls=benchmark_controls.build_controls(config, task_input_sha256=hashes,
            task=task_name, model=native.MODEL, reasoning_effort=native.REASONING, cli_version=native.CLI_VERSION))
    native._json(output / "preparation.json", prepared)
    return output, prepared, config


@pytest.mark.parametrize("runner", [native, supervisor])
def test_selected_task_is_bound_through_config_execution_collection_and_controls(tmp_path, monkeypatch, runner):
    output, prepared, config = _prepared(tmp_path, runner)
    assert config["tasks"] == [{"path": str(tmp_path / "dataset" / SELECTED)}]
    assert config["job_name"].endswith(SELECTED)
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(runner.subprocess, "run", run)
    receipt = runner.execute(output, task_name=SELECTED)
    assert calls == [prepared["command"]]
    assert receipt["task"] == SELECTED
    assert receipt["original_task_inputs_unchanged"] is True
    assert receipt["comparison_controls"]["status"] == "observed"
    assert receipt["comparison_controls"]["declared"]["identity"]["task"] == SELECTED
    assert receipt["trial_count"] == 0
    if runner is native:
        assert receipt["native_job_usage"]["input_tokens"] is None
    else:
        assert receipt["dollar_cost"] is None


@pytest.mark.parametrize("runner", [native, supervisor])
def test_default_task_preserves_original_runner_profile(tmp_path, runner):
    output, _, config = _prepared(tmp_path, runner, task_name=native.TASK)
    assert config["tasks"][0]["path"].endswith("/" + native.TASK)
    assert runner.collect(output)["task"] == native.TASK


@pytest.mark.parametrize("name", ["", ".", "..", "../escape", "/absolute", "a/b", "a\\b", "a*", "a\n", "a" * 129, None, 1])
@pytest.mark.parametrize("runner", [native, supervisor])
def test_paths_globs_and_non_names_are_rejected_before_configuration(tmp_path, runner, name):
    with pytest.raises(ValueError, match="simple dataset child"):
        if runner is native:
            runner.config_for(tmp_path, tmp_path / "out", task_name=name)
        else:
            runner.config_for(tmp_path, tmp_path / "out", tmp_path / "archive", "full", task_name=name)


def test_task_directory_and_dataset_parent_cannot_be_symlink_aliases(tmp_path):
    dataset = tmp_path / "dataset"
    target = _task(dataset, SELECTED)
    (dataset / "linked").symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="canonical non-symlink"):
        native._task_path(dataset, "linked")
    alias = tmp_path / "alias"
    alias.symlink_to(dataset, target_is_directory=True)
    with pytest.raises(ValueError, match="canonical non-symlink"):
        native._task_path(alias, SELECTED)


@pytest.mark.parametrize("runner", [native, supervisor])
@pytest.mark.parametrize("operation", ["execute", "collect"])
@pytest.mark.parametrize("changed", ["prepared_task", "config_task", "job_name", "jobs_dir", "controls_task", "requested_task"])
def test_task_binding_mismatch_fails_before_provider_or_result_read(tmp_path, monkeypatch, runner, operation, changed):
    output, prepared, config = _prepared(tmp_path, runner)
    kwargs = {}
    if changed == "prepared_task":
        prepared["task"] = native.TASK
    elif changed == "config_task":
        config["tasks"][0]["path"] = str(tmp_path / "dataset" / native.TASK)
    elif changed == "job_name":
        config["job_name"] += "-other"
    elif changed == "jobs_dir":
        config["jobs_dir"] = str(tmp_path / "unrelated-jobs")
    elif changed == "controls_task":
        prepared["comparison_controls"] = benchmark_controls.build_controls(config,
            task_input_sha256=prepared["task_input_sha256"], task=native.TASK,
            model=native.MODEL, reasoning_effort=native.REASONING, cli_version=native.CLI_VERSION)
    else:
        kwargs["task_name"] = native.TASK
    native._json(output / "config.json", config)
    # Even an updated config checksum cannot legitimize a changed task binding.
    prepared["config_sha256"] = native._hash(output / "config.json")
    native._json(output / "preparation.json", prepared)
    def forbidden(*args, **kwargs):
        pytest.fail("mismatched selection reached a provider boundary")
    monkeypatch.setattr(runner.subprocess, "run", forbidden)
    with pytest.raises(ValueError, match="task|configuration"):
        getattr(runner, operation)(output, **kwargs)
    assert not (output / "invocation.json").exists()


@pytest.mark.parametrize("runner", [native, supervisor])
def test_selected_task_input_drift_prevents_execution(tmp_path, monkeypatch, runner):
    output, _, _ = _prepared(tmp_path, runner)
    (tmp_path / "dataset" / SELECTED / "instruction.md").write_text("Changed public input.\n")
    def forbidden(*args, **kwargs):
        pytest.fail("changed task reached a provider boundary")
    monkeypatch.setattr(runner.subprocess, "run", forbidden)
    with pytest.raises(ValueError, match="task changed"):
        runner.execute(output)
    assert not (output / "invocation.json").exists()


@pytest.mark.parametrize("runner", [native, supervisor])
@pytest.mark.parametrize("identity,expected", [("matching", True), ("wrong_path", False),
    ("wrong_name", False), ("missing", None), ("missing_name", None)])
def test_collected_trial_must_match_selected_task_before_scoring(tmp_path, runner, identity, expected):
    output, _, config = _prepared(tmp_path, runner)
    task = tmp_path / "dataset" / SELECTED
    job = output / "jobs" / config["job_name"]
    trial = job / "authored-trial"
    trial.mkdir(parents=True)
    result = dict(task_name=SELECTED, trial_name=trial.name,
        config={"task": {"path": str(task)}, "agent": config["agents"][0], "verifier": config["verifier"]},
        verifier_result={"rewards": {"reward": 1.0}})
    if identity == "wrong_path":
        result["config"]["task"]["path"] = str(task.parent / native.TASK)
    elif identity == "wrong_name":
        result["task_name"] = native.TASK
    elif identity == "missing":
        result.pop("task_name")
        result["config"].pop("task")
    elif identity == "missing_name":
        result.pop("task_name")
    native._json(trial / "result.json", result)
    native._json(job / "result.json", {"stats": {}})
    receipt = runner.collect(output)
    assert receipt["trials"][0]["exact_trial_task_matches"] is expected
    assert receipt["complete_single_trial_receipt"] is (expected is True)
    # Retain the observed verifier value even if attribution cannot be established.
    assert receipt["trials"][0]["reward"] == {"reward": 1.0}


def test_trial_uses_public_task_package_name_when_declared(tmp_path):
    task = _task(tmp_path / "dataset", SELECTED)
    (task / "task.toml").write_text('[task]\nname = "terminal-bench/another-coding-task"\n')
    result = {"task_name": "terminal-bench/" + SELECTED, "config": {"task": {"path": str(task)}}}
    assert native._trial_task_matches(result, task) is True
    result["task_name"] = SELECTED
    assert native._trial_task_matches(result, task) is False


def test_nondefault_supervisor_task_needs_explicit_profile(tmp_path):
    task = _task(tmp_path / "dataset", SELECTED)
    with pytest.raises(ValueError, match="requires an explicit task profile"):
        supervisor._load_task_profile(None, task=task)
    original = _task(tmp_path / "dataset", native.TASK)
    assert supervisor._load_task_profile(None, task=original) is None


def _profile(instruction):
    from benchmarks.agent_supervisor.container_coding.terminal_task_profile import instruction_sha256, SCHEMA
    return {"schema": SCHEMA, "instruction_sha256": instruction_sha256(instruction),
        "input_paths": ["source.py"], "outputs": [
            {"path": "source.py", "effect": "modify", "media_type": "text/x-python"}]}


def test_task_profile_binds_normalized_instruction_and_reaches_harbor_kwargs(tmp_path):
    task = _task(tmp_path / "dataset", SELECTED)
    instruction = "Repair the source.\nPreserve its interface.\n"
    (task / "instruction.md").write_bytes(instruction.replace("\n", "\r\n").encode())
    profile = _profile(instruction)
    path = tmp_path / "profile.json"
    native._json(path, profile)
    selected = supervisor._load_task_profile(path, task=task)
    config = supervisor.config_for(task.parent, tmp_path / "out", tmp_path / "archive", "full",
                                    task_name=SELECTED, task_profile=selected)
    assert config["agents"][0]["kwargs"]["task_profile"] == profile
    # Detached transport prevents a caller changing a selection after validation.
    selected["outputs"][0]["path"] = "changed.py"
    assert config["agents"][0]["kwargs"]["task_profile"] == profile
    (task / "instruction.md").write_text("A different task.\n")
    with pytest.raises(ValueError, match="instruction digest differs"):
        supervisor._load_task_profile(path, task=task)


@pytest.mark.parametrize("kind", ["duplicate", "nonfinite", "oversize", "symlink", "instruction_symlink"])
def test_task_profile_loader_rejects_ambiguous_unbounded_or_aliased_inputs(tmp_path, kind):
    task = _task(tmp_path / "dataset", SELECTED)
    path = tmp_path / "profile.json"
    profile = _profile((task / "instruction.md").read_text())
    native._json(path, profile)
    if kind == "duplicate":
        path.write_text('{"schema":"a","schema":"b"}')
    elif kind == "nonfinite":
        path.write_text('{"unknown":NaN}')
    elif kind == "oversize":
        path.write_bytes(b" " * (supervisor.MAX_TASK_PROFILE_BYTES + 1))
    elif kind == "symlink":
        alias = tmp_path / "profile-alias.json"
        alias.symlink_to(path)
        path = alias
    else:
        original = task / "instruction.md"
        original.rename(tmp_path / "instruction.txt")
        original.symlink_to(tmp_path / "instruction.txt")
    with pytest.raises(ValueError):
        supervisor._load_task_profile(path, task=task)
