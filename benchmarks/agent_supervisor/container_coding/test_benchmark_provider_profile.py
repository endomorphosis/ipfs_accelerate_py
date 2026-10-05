"""Current CLI parity and historical archive refusal before provider dispatch."""
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.agent_supervisor.container_coding import benchmark_provider_profile as profile
from benchmarks.agent_supervisor.container_coding import full_supervisor_benchmark as supervisor
from benchmarks.agent_supervisor.container_coding import native_codex_baseline as native
from benchmarks.agent_supervisor.container_coding import terminal_deployment as deployment
from benchmarks.agent_supervisor.container_coding import terminal_indexed_preparation as preparation
from benchmarks.agent_supervisor.container_coding.test_terminal_deployment import _inputs
from benchmarks.agent_supervisor.container_coding.test_terminal_task_selection import _prepared, _task
from benchmarks.agent_supervisor.container_coding.test_terminal_indexed_preparation import original


def test_current_native_and_supervisor_paths_select_the_same_cli(tmp_path):
    assert profile.CLI_VERSION == "0.160.0"
    assert native.CLI_VERSION == deployment.CODEX_VERSION == preparation.CLI_VERSION == profile.CLI_VERSION
    config = native.config_for(Path("/dataset"), tmp_path)
    assert config["agents"][0]["kwargs"]["version"] == profile.CLI_VERSION
    manifest = deployment.build_runtime_archive(output=tmp_path / "bundle", **_inputs(tmp_path))
    assert manifest["codex_version"] == profile.CLI_VERSION
    namespace = {}
    # The script's literal pin is the same one checked by the actual native
    # executable deployment tests; no provider binary executes here.
    exec(deployment.native_codex_exposure_script().split("import hashlib", 1)[0], namespace)
    assert namespace["expected_version"] == profile.CLI_VERSION


@pytest.mark.parametrize("version", ["0.158.0", "0.159.0", "0.161.0", None, 160])
def test_old_or_malformed_archive_refused_before_container_setup(tmp_path, version):
    manifest = deployment.build_runtime_archive(output=tmp_path / "bundle", **_inputs(tmp_path))
    manifest["codex_version"] = version
    (tmp_path / "bundle/manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Codex version differs"):
        asyncio.run(deployment.deploy_supervisor(None, archive_dir=tmp_path / "bundle",
            output=tmp_path / "deployment"))
    assert not (tmp_path / "deployment").exists()


def test_old_archive_refused_before_harbor_preparation(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    _task(dataset, native.TASK)
    manifest = deployment.build_runtime_archive(output=tmp_path / "bundle", **_inputs(tmp_path))
    manifest["codex_version"] = "0.158.0"
    (tmp_path / "bundle/manifest.json").write_text(json.dumps(manifest))
    def forbidden(*args, **kwargs):
        pytest.fail("old archive reached Harbor or model setup")
    monkeypatch.setattr(supervisor.subprocess, "run", forbidden)
    with pytest.raises(ValueError, match="Codex version differs"):
        supervisor.prepare(dataset=dataset, output=tmp_path / "trial", archive=tmp_path / "bundle", arm="full")
    assert not (tmp_path / "trial").exists()


def test_rehashed_old_archive_cannot_reach_harbor_execution(tmp_path, monkeypatch):
    output, prepared, _ = _prepared(tmp_path, supervisor)
    manifest_path = Path(prepared["archive"]) / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["codex_version"] = "0.158.0"
    native._json(manifest_path, manifest)
    prepared["manifest_sha256"] = native._hash(manifest_path)
    native._json(output / "preparation.json", prepared)
    def forbidden(*args, **kwargs):
        pytest.fail("old archive reached Harbor execution")
    monkeypatch.setattr(supervisor.subprocess, "run", forbidden)
    with pytest.raises(ValueError, match="Codex version differs"):
        supervisor.execute(output)
    assert not (output / "invocation.json").exists()


def test_direct_adapter_refuses_old_archive_before_task_bootstrap(tmp_path):
    from benchmarks.agent_supervisor.container_coding.full_supervisor_harbor_agent import FullSupervisorAgent
    (tmp_path / "manifest.json").write_text(json.dumps({"codex_version": "0.158.0"}))
    agent = object.__new__(FullSupervisorAgent)
    agent.runtime_archive = tmp_path
    with pytest.raises(ValueError, match="Codex version differs"):
        asyncio.run(agent.setup(None))


def test_old_planner_cli_refused_before_model_dispatch(original, monkeypatch):
    repository, instruction, state = original
    preparation.prepare(repository=repository, instruction=instruction, state=state)
    actual_run = preparation.subprocess.run
    def command(argv, *args, **kwargs):
        if argv == ["codex", "--version"]:
            return SimpleNamespace(stdout="codex-cli 0.158.0\n")
        return actual_run(argv, *args, **kwargs)
    def forbidden(*args, **kwargs):
        pytest.fail("old CLI reached planner provider")
    monkeypatch.setattr(preparation.subprocess, "run", command)
    with pytest.raises(ValueError, match="native baseline parity requires codex-cli 0.160.0"):
        preparation.plan(state=state, provider_callable=forbidden)
    assert not (state / "planner-invoked.json").exists()


def _historical_preparation(tmp_path, runner, monkeypatch):
    output, prepared, config = _prepared(tmp_path, runner)
    historical = {"model": "gpt-5.6-sol", "reasoning_effort": "high", "cli_version": "0.158.0"}
    config["agents"][0]["model_name"] = historical["model"]
    if runner is native:
        config["agents"][0]["kwargs"]["version"] = historical["cli_version"]
    # Author the historical control with its historical source profile, then
    # collect it with today's untouched profile. No actual task/model executes.
    with monkeypatch.context() as frozen:
        frozen.setattr(native, "MODEL", historical["model"])
        frozen.setattr(native, "CLI_VERSION", historical["cli_version"])
        prepared["comparison_controls"] = native.benchmark_controls.build_controls(config,
            task_input_sha256=prepared["task_input_sha256"], task=prepared["task"], **historical)
    native._json(output / "config.json", config)
    prepared.update(historical, config_sha256=native._hash(output / "config.json"))
    native._json(output / "preparation.json", prepared)
    job = output / "jobs" / config["job_name"]
    trial = job / "authored-trial"
    trial.mkdir(parents=True)
    result = {"task_name": prepared["task"], "trial_name": trial.name,
        "config": {"task": config["tasks"][0], "agent": config["agents"][0], "verifier": config["verifier"]},
        "verifier_result": {"rewards": {"reward": 1.0}}}
    native._json(trial / "result.json", result)
    native._json(job / "result.json", {"stats": {}})
    return output, prepared, config, historical


@pytest.mark.parametrize("runner", [native, supervisor])
@pytest.mark.parametrize("has_controls", [False, True])
def test_historical_recollection_preserves_frozen_model_and_cli(tmp_path, monkeypatch, runner, has_controls):
    output, prepared, _, historical = _historical_preparation(tmp_path, runner, monkeypatch)
    if not has_controls:
        prepared.pop("comparison_controls")
        native._json(output / "preparation.json", prepared)
    assert native.CLI_VERSION == profile.CLI_VERSION == "0.160.0"
    receipt = runner.collect(output)
    assert {key: receipt[key] for key in historical} == historical
    assert receipt["trials"][0]["reward"] == {"reward": 1.0}
    assert receipt["complete_single_trial_receipt"] is True
    if runner is native:
        assert receipt["trials"][0]["exact_trial_profile_matches"] is True
    assert receipt["comparison_controls"]["status"] == ("observed" if has_controls else "unavailable")
    assert not (output / "invocation.json").exists()


@pytest.mark.parametrize("runner", [native, supervisor])
@pytest.mark.parametrize("mutation", ["metadata_conflict", "config_conflict", "missing_identity"])
def test_recollection_refuses_missing_or_conflicting_frozen_profile(tmp_path, monkeypatch, runner, mutation):
    output, prepared, config, _ = _historical_preparation(tmp_path, runner, monkeypatch)
    if mutation == "metadata_conflict":
        prepared["cli_version"] = profile.CLI_VERSION
    elif mutation == "config_conflict":
        config["agents"][0]["model_name"] = native.MODEL
        native._json(output / "config.json", config)
        prepared["config_sha256"] = native._hash(output / "config.json")
    else:
        for key in ("comparison_controls", "model", "reasoning_effort", "cli_version"):
            prepared.pop(key)
    native._json(output / "preparation.json", prepared)
    with pytest.raises(ValueError, match="provider|profile"):
        runner.collect(output)
    assert not (output / "receipt.json").exists()


def test_native_execution_refuses_changed_profile_source_before_invocation(tmp_path, monkeypatch):
    copied_profile = tmp_path / "benchmark_provider_profile.py"
    copied_profile.write_bytes(Path(profile.__file__).read_bytes())
    monkeypatch.setattr(profile, "__file__", str(copied_profile))
    output, _, _ = _prepared(tmp_path, native)
    copied_profile.write_text(copied_profile.read_text() + "\n# changed after preparation\n")
    def forbidden(*args, **kwargs):
        pytest.fail("changed provider profile reached a paid invocation")
    monkeypatch.setattr(native.subprocess, "run", forbidden)
    with pytest.raises(ValueError, match="provider profile changed"):
        native.execute(output)
    assert not (output / "invocation.json").exists()


@pytest.mark.parametrize("config", [None, [], {}, {"agents": []}, {"agents": [None]}, {"agents": [{}]}])
def test_malformed_config_is_a_control_mismatch_and_cannot_supply_provider_identity(tmp_path, config):
    _, prepared, _ = _prepared(tmp_path, native)
    observed = native.benchmark_controls.observe_controls(prepared, config,
        current_task_hashes=prepared["task_input_sha256"])
    assert observed["status"] == "mismatch"
    legacy = {"model": "gpt-5.6-sol", "reasoning_effort": "high", "cli_version": "0.158.0"}
    with pytest.raises(ValueError, match="provider"):
        profile.prepared_provider_identity(legacy, config)
