"""Fresh host budgets agree across arms; retained trials keep their own bounds.

These authored configurations and receipt fixtures do not launch Harbor,
Docker, providers, or the benchmark verifier.
"""

from __future__ import annotations

import io
import json
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from benchmarks.agent_supervisor.container_coding import (
    full_supervisor_benchmark as supervisor,
)
from benchmarks.agent_supervisor.container_coding import (
    native_codex_baseline as native,
)
from benchmarks.agent_supervisor.container_coding import (
    timeout_policy,
)
from benchmarks.agent_supervisor.container_coding import benchmark_controls
from benchmarks.agent_supervisor.container_coding.benchmark_resource_profile import (
    PROFILES, FORMALIZATION_SOURCE384_PROFILE, execution_budget,
)
from harbor.models.job.config import JobConfig
from harbor.trial.trial import Trial


def _configs(tmp_path, profile=None):
    options = {"resource_profile": profile}
    return [
        native.config_for(tmp_path / "dataset", tmp_path / "native", **options),
        *[supervisor.config_for(tmp_path / "dataset", tmp_path / arm,
                               tmp_path / "archive", arm, **options)
          for arm in ("full", "no-index")],
    ]


def _assert_common(configs, profile=None):
    seconds = 6480 if profile == FORMALIZATION_SOURCE384_PROFILE else 960 if profile in PROFILES[1:] else 300
    baseline = configs[0]
    for config in configs:
        agent = config["agents"][0]
        assert agent["override_timeout_sec"] == agent["max_timeout_sec"] == seconds
        assert agent["override_setup_timeout_sec"] == 1800
        validated = JobConfig.model_validate(config, extra="forbid")
        trial = SimpleNamespace(config=SimpleNamespace(
            agent=validated.agents[0], agent_setup_timeout_multiplier=None,
            timeout_multiplier=validated.timeout_multiplier,
        ))
        trial._resolve_timeout_sec = Trial._resolve_timeout_sec.__get__(trial)
        assert Trial._compute_agent_setup_timeout_sec(trial) == 1800
        for key in ("tasks", "environment", "verifier", "n_attempts", "n_concurrent_trials",
                    "timeout_multiplier", "retry"):
            assert config[key] == baseline[key]
    assert baseline["agents"][0]["kwargs"] == {
        "version": native.CLI_VERSION, "reasoning_effort": native.REASONING,
    }
    for config in configs[1:]:
        assert "agent_timeout_seconds" not in config["agents"][0]["kwargs"]
        assert config["agents"][0]["kwargs"].get("resource_profile") == profile


def test_fresh_default_budget_is_shared_for_every_arm(tmp_path):
    _assert_common(_configs(tmp_path))


@pytest.mark.parametrize("seconds", [120, 300, 450, 600])
def test_numeric_timeout_validator_retains_its_supported_boundaries(tmp_path, seconds):
    assert timeout_policy.validate_agent_timeout(seconds) == seconds
    # Public benchmark budgets are selected through resource_profile; numeric
    # timeout helper values do not create an unsupported public override.
    with pytest.raises(TypeError, match="agent_timeout_seconds"):
        native.config_for(tmp_path, tmp_path, agent_timeout_seconds=seconds)


@pytest.mark.parametrize("profile", [None, *PROFILES])
def test_supported_resource_budgets_are_shared_without_changing_task_resources(tmp_path, profile):
    _assert_common(_configs(tmp_path, profile), profile)


@pytest.mark.parametrize("seconds", [True, False, "600", 600.0, None, 119, 601])
@pytest.mark.parametrize("arm", ["native", "full", "no-index"])
def test_invalid_config_budget_is_rejected(tmp_path, seconds, arm):
    with pytest.raises(ValueError, match="agent timeout"):
        timeout_policy.validate_agent_timeout(seconds)
    with pytest.raises(TypeError, match="agent_timeout_seconds"):
        if arm == "native":
            native.config_for(tmp_path, tmp_path, agent_timeout_seconds=seconds)
        else:
            supervisor.config_for(tmp_path, tmp_path, tmp_path, arm,
                                  agent_timeout_seconds=seconds)


@pytest.mark.parametrize("seconds", [True, "600", 600.0, None, 119, 601])
@pytest.mark.parametrize("module", [native, supervisor])
def test_invalid_prepare_budget_is_rejected_before_filesystem_or_process_work(tmp_path, seconds, module, monkeypatch):
    def unexpected(*_args, **_kwargs):
        raise AssertionError("invalid budget reached filesystem or process work")

    monkeypatch.setattr(module.subprocess, "run", unexpected)
    missing_dataset = SimpleNamespace(resolve=unexpected, __fspath__=unexpected)
    options = {"dataset": missing_dataset, "output": tmp_path / "output",
               "agent_timeout_seconds": seconds}
    if module is supervisor:
        options.update(archive=tmp_path / "missing-archive", arm="full")
    with pytest.raises(TypeError, match="agent_timeout_seconds"):
        module.prepare(**options)


def _native_trial(tmp_path, *, seconds, omit_timeout=False, pin_policy=False):
    dataset = tmp_path / "dataset"
    task = dataset / native.TASK
    task.mkdir(parents=True)
    (task / "instruction.md").write_text("Authored public instruction.\n")
    (task / "task.toml").write_text('version = "1.0"\n')
    (task / "environment").mkdir()
    (task / "tests").mkdir()
    output = tmp_path / "trial"
    output.mkdir()
    config = native.config_for(dataset, output)
    # Retained configurations can contain historical budgets; the current
    # profile matcher must classify those rather than relabel them as current.
    config["agents"][0]["override_timeout_sec"] = float(seconds)
    config["agents"][0]["max_timeout_sec"] = float(seconds)
    native._json(output / "config.json", config)
    prepared = {
        "prepared": True, "dataset": str(dataset), "config_sha256": native._hash(output / "config.json"),
        "task_input_sha256": native._task_hashes(task),
        "collector_source_sha256": native._hash(Path(native.__file__)),
        "native_adapter_source_sha256": {}, "command": ["authored-harbor"],
        "controls_source_sha256": native._hash(Path(benchmark_controls.__file__)),
        "provider_profile_source_sha256": native._hash(Path(native.benchmark_provider_profile.__file__)),
        "model": native.MODEL, "reasoning_effort": native.REASONING, "cli_version": native.CLI_VERSION,
    }
    if not omit_timeout:
        prepared["agent_timeout_seconds"] = seconds
    if pin_policy:
        prepared["timeout_policy_source_sha256"] = native._hash(Path(timeout_policy.__file__))
    native._json(output / "preparation.json", prepared)
    job = output / "jobs" / ("native-codex-" + native.TASK)
    trial = job / "authored-trial"
    trial.mkdir(parents=True)
    record = {
        "trial_name": "authored-trial", "task_name": native.TASK,
        "config": {"task": {"path": str(task)}, "agent": config["agents"][0],
                   "verifier": {"disable": False}},
        "verifier_result": {"rewards": {"reward": 0}},
    }
    native._json(trial / "result.json", record)
    native._json(job / "result.json", {"stats": {}})
    return output, prepared, record


@pytest.mark.parametrize("seconds,omit_timeout,pin_policy", [
    (300, True, False), (300, False, False), (450, False, True), (600, False, True),
])
def test_native_collection_classifies_each_immutable_retained_budget(tmp_path, seconds, omit_timeout, pin_policy):
    output, _, _ = _native_trial(tmp_path, seconds=seconds, omit_timeout=omit_timeout, pin_policy=pin_policy)
    original = (output / "preparation.json").read_bytes()
    result = native.collect(output)
    assert result["trials"][0]["exact_trial_profile_matches"] is (seconds == 300)
    assert result["complete_single_trial_receipt"] is (seconds == 300)
    assert json.loads((output / "config.json").read_text())["agents"][0]["override_timeout_sec"] == seconds
    assert "agent_timeout_config_unchanged" not in result
    assert "timeout_policy_source_unchanged" not in result
    assert result["benchmark_advantage_claimed"] is False
    assert (output / "preparation.json").read_bytes() == original


def test_missing_policy_pin_cannot_report_policy_source_verified(tmp_path):
    output, _, _ = _native_trial(tmp_path, seconds=600)
    result = native.collect(output)
    assert "timeout_policy_source_sha256" not in result
    assert "timeout_policy_source_unchanged" not in result


def test_existing_controls_owner_detects_changed_setup_allowance(tmp_path):
    output, prepared, _ = _native_trial(tmp_path, seconds=300, pin_policy=True)
    original = (output / "preparation.json").read_bytes()
    result = native.collect(output)
    assert result["complete_single_trial_receipt"] is True
    config = json.loads((output / "config.json").read_text())
    declared = benchmark_controls.build_controls(config, task_input_sha256=prepared['task_input_sha256'],
        task=native.TASK, model=native.MODEL, reasoning_effort=native.REASONING, cli_version=native.CLI_VERSION)
    config["agents"][0]["override_setup_timeout_sec"] = 360
    observed = benchmark_controls.observe_controls({'comparison_controls': declared}, config,
        current_task_hashes=prepared['task_input_sha256'])
    assert observed['status'] == 'mismatch' and observed['configuration_unchanged'] is False
    assert "agent_setup_timeout_config_unchanged" not in result
    assert (output / "preparation.json").read_bytes() == original


def test_legacy_native_collection_does_not_invent_setup_allowance(tmp_path):
    output, prepared, record = _native_trial(tmp_path, seconds=300, omit_timeout=True)
    config = json.loads((output / "config.json").read_text())
    del config["agents"][0]["override_setup_timeout_sec"]
    del record["config"]["agent"]["override_setup_timeout_sec"]
    native._json(output / "config.json", config)
    prepared["config_sha256"] = native._hash(output / "config.json")
    native._json(output / "preparation.json", prepared)
    native._json(output / "jobs" / ("native-codex-" + native.TASK) / "authored-trial/result.json", record)
    original = (output / "preparation.json").read_bytes()
    result = native.collect(output)
    assert "agent_setup_timeout_seconds" not in result
    assert "agent_setup_timeout_basis" not in result
    assert "agent_setup_timeout_config_unchanged" not in result
    assert result["comparison_controls"]["status"] == "unavailable"
    assert result["trials"][0]["exact_trial_profile_matches"] is True
    assert (output / "preparation.json").read_bytes() == original


@pytest.mark.parametrize("field", ["override_timeout_sec", "max_timeout_sec"])
def test_native_collection_marks_trial_budget_mismatch_false(tmp_path, field):
    output, _, record = _native_trial(tmp_path, seconds=300, pin_policy=True)
    record["config"]["agent"][field] = 120
    native._json(output / "jobs" / ("native-codex-" + native.TASK) / "authored-trial/result.json", record)
    result = native.collect(output)
    assert result["trials"][0]["exact_trial_profile_matches"] is False
    assert result["complete_single_trial_receipt"] is False


def test_native_collection_does_not_promote_unbound_preparation_budget_metadata(tmp_path):
    output, prepared, _ = _native_trial(tmp_path, seconds=600, pin_policy=True)
    prepared["agent_timeout_seconds"] = 450
    native._json(output / "preparation.json", prepared)
    result = native.collect(output)
    assert "agent_timeout_config_unchanged" not in result
    assert result["trials"][0]["exact_trial_profile_matches"] is False
    assert result["complete_single_trial_receipt"] is False
    assert json.loads((output / "config.json").read_text())["agents"][0]["override_timeout_sec"] == 600


def test_native_collection_rejects_modified_immutable_configuration(tmp_path):
    output, _, _ = _native_trial(tmp_path, seconds=600, pin_policy=True)
    config = json.loads((output / "config.json").read_text())
    config["agents"][0]["override_timeout_sec"] = 300
    native._json(output / "config.json", config)
    with pytest.raises(ValueError, match="prepared task binding differs"):
        native.collect(output)


def test_native_execution_rejects_changed_pinned_budget_controls_owner_before_launch(tmp_path, monkeypatch):
    output, prepared, _ = _native_trial(tmp_path, seconds=600, pin_policy=True)
    prepared["controls_source_sha256"] = "0" * 64
    native._json(output / "preparation.json", prepared)
    monkeypatch.setattr(native.subprocess, "run", lambda *_args, **_kwargs: pytest.fail("Harbor launched"))
    with pytest.raises(ValueError, match="benchmark controls owner changed"):
        native.execute(output)
    assert not (output / "invocation.json").exists()


def test_native_execution_rejects_frozen_configuration_budget_mismatch_before_launch(tmp_path, monkeypatch):
    output, _, _ = _native_trial(tmp_path, seconds=600, pin_policy=True)
    config = json.loads((output / "config.json").read_text())
    config["agents"][0]["override_timeout_sec"] = 450
    native._json(output / "config.json", config)
    monkeypatch.setattr(native.subprocess, "run", lambda *_args, **_kwargs: pytest.fail("Harbor launched"))
    with pytest.raises(ValueError, match="prepared task binding differs"):
        native.execute(output)
    assert not (output / "invocation.json").exists()


@pytest.mark.parametrize("module", [native, supervisor])
@pytest.mark.parametrize("mismatch", ["preparation", "configuration"])
def test_execution_rejects_setup_allowance_binding_mismatch_before_launch(tmp_path, monkeypatch, module, mismatch):
    output, prepared, _ = _native_trial(tmp_path, seconds=600, pin_policy=True)
    config = json.loads((output / "config.json").read_text())
    if module is supervisor:
        prepared["arm"] = "full"
        config = supervisor.config_for(Path(prepared["dataset"]), output, tmp_path / "archive", "full")
    native._json(output / "config.json", config)
    prepared["config_sha256"] = native._hash(output / "config.json")
    if mismatch == "configuration":
        config["agents"][0]["override_setup_timeout_sec"] = 360
        native._json(output / "config.json", config)
    else:
        prepared["config_sha256"] = "0" * 64
    native._json(output / "preparation.json", prepared)
    monkeypatch.setattr(module.subprocess, "run", lambda *_args, **_kwargs: pytest.fail("Harbor launched"))
    with pytest.raises(ValueError, match="prepared task binding differs"):
        module.execute(output)
    assert not (output / "invocation.json").exists()


def test_legacy_native_execution_preserves_retained_budget_without_policy_pin(tmp_path, monkeypatch):
    output, _, _ = _native_trial(tmp_path, seconds=300, omit_timeout=True)
    original = (output / "config.json").read_bytes()
    calls = []
    def run(command, **_kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(native.subprocess, "run", run)
    result = native.execute(output)
    assert calls == [["authored-harbor"]]
    assert result["trials"][0]["exact_trial_profile_matches"] is True
    assert "agent_timeout_config_unchanged" not in result
    assert (output / "config.json").read_bytes() == original


@pytest.mark.parametrize("seconds", [300, 450, 600])
def test_supervisor_collection_preserves_frozen_historical_budget_without_invented_projection(tmp_path, seconds):
    output, prepared, _ = _native_trial(tmp_path, seconds=seconds)
    prepared["arm"] = "full"
    config = supervisor.config_for(Path(prepared["dataset"]), output, tmp_path / "archive", "full")
    config["agents"][0]["override_timeout_sec"] = seconds
    config["agents"][0]["max_timeout_sec"] = seconds
    native._json(output / "config.json", config)
    prepared["config_sha256"] = native._hash(output / "config.json")
    native._json(output / "preparation.json", prepared)
    original = (output / "preparation.json").read_bytes()
    result = supervisor.collect(output)
    assert "agent_timeout_config_unchanged" not in result
    assert result["comparison_controls"]["status"] == "unavailable"
    assert result["complete_single_trial_receipt"] is False
    assert result["resource_profile"] is None
    assert (output / "preparation.json").read_bytes() == original
    config["agents"][0]["override_timeout_sec"] = 120
    native._json(output / "config.json", config)
    with pytest.raises(ValueError, match="prepared task binding differs"):
        supervisor.collect(output)


def test_legacy_supervisor_collection_keeps_implicit_five_minute_adapter_budget(tmp_path):
    output, prepared, _ = _native_trial(tmp_path, seconds=300, omit_timeout=True)
    prepared["arm"] = "no-index"
    config = supervisor.config_for(Path(prepared["dataset"]), output, tmp_path / "archive", "no-index")
    native._json(output / "config.json", config)
    prepared["config_sha256"] = native._hash(output / "config.json")
    native._json(output / "preparation.json", prepared)
    result = supervisor.collect(output)
    assert config["agents"][0]["override_timeout_sec"] == 300
    assert "agent_timeout_seconds" not in config["agents"][0]["kwargs"]
    assert "agent_setup_timeout_basis" not in result
    assert result["comparison_controls"]["status"] == "unavailable"
    assert result["complete_single_trial_receipt"] is False


def test_supervisor_execution_rejects_frozen_budget_mismatch_before_launch(tmp_path, monkeypatch):
    output, prepared, _ = _native_trial(tmp_path, seconds=600, pin_policy=True)
    prepared["arm"] = "full"
    config = supervisor.config_for(Path(prepared["dataset"]), output, tmp_path / "archive", "full")
    native._json(output / "config.json", config)
    prepared["config_sha256"] = native._hash(output / "config.json")
    native._json(output / "preparation.json", prepared)
    config["agents"][0]["override_timeout_sec"] = 450
    native._json(output / "config.json", config)
    monkeypatch.setattr(supervisor.subprocess, "run", lambda *_args, **_kwargs: pytest.fail("Harbor launched"))
    with pytest.raises(ValueError, match="prepared task binding differs"):
        supervisor.execute(output)
    assert not (output / "invocation.json").exists()


@pytest.mark.parametrize("module", [native, supervisor])
@pytest.mark.parametrize("profile", [None, PROFILES[1]])
def test_prepare_cli_forwards_default_or_explicit_resource_profile(module, profile, tmp_path, monkeypatch, capsys):
    observed = {}
    monkeypatch.setattr(module, "prepare", lambda **kwargs: observed.update(kwargs) or {"prepared": True})
    argv = ["benchmark", "prepare", "--dataset", str(tmp_path / "dataset"), "--output", str(tmp_path / "output")]
    if module is supervisor:
        argv += ["--archive", str(tmp_path / "archive")]
    if profile is not None:
        argv += ["--resource-profile", profile]
    monkeypatch.setattr(sys, "argv", argv)
    module.main()
    assert observed["resource_profile"] == profile
    assert json.loads(capsys.readouterr().out) == {"prepared": True}


@pytest.mark.parametrize("module", [native, supervisor])
def test_execute_cli_uses_retained_preparation_without_budget_override(module, tmp_path, monkeypatch, capsys):
    observed = []
    monkeypatch.setattr(module, "execute", lambda output, **kwargs: observed.append(output) or {"executed": True})
    monkeypatch.setattr(sys, "argv", ["benchmark", "execute", "--output", str(tmp_path / "output")])
    module.main()
    assert observed == [tmp_path / "output"]
    assert json.loads(capsys.readouterr().out) == {"executed": True}


@pytest.mark.parametrize("module", [native, supervisor])
def test_removed_numeric_timeout_cli_override_refuses_before_dispatch(module, tmp_path, monkeypatch):
    monkeypatch.setattr(module, "execute", lambda *args, **kwargs: pytest.fail("unsupported override dispatched"))
    monkeypatch.setattr(sys, "argv", ["benchmark", "execute", "--output", str(tmp_path / "output"),
                                    "--agent-timeout-seconds", "450"])
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 2


def _timeout_runtime_archive(tmp_path, *, stale_payload=None, omit_payload=None, extra_member=None):
    from benchmarks.agent_supervisor.container_coding.terminal_worker_capability import (
        KEY, _current_capability,
    )
    archive = tmp_path / "runtime"
    archive.mkdir()
    sources = supervisor._timeout_runtime_sources()
    capability = _current_capability()
    checkout = Path(supervisor.__file__).resolve().parents[3]
    # Real source bodies support current preparation capability guards. The
    # fixture remains model-free and every process call is explicitly mocked.
    sources.update({name: checkout / name.removeprefix("source/")
                    for name in capability["source_sha256"]})
    inventory = []
    with tarfile.open(archive / "runtime.tar.gz", "w:gz") as bundle:
        for path, source in sources.items():
            name = source.name
            raw = source.read_bytes()
            inventory.append({"path": path, "bytes": len(raw),
                              "sha256": supervisor._hash(source),
                              "mode": 0o644})
            if name == omit_payload:
                continue
            payload = (b"#" if raw[0:1] != b"#" else b" ") + raw[1:] if name == stale_payload else raw
            entry = tarfile.TarInfo(path)
            entry.size, entry.mode = len(payload), 0o644
            bundle.addfile(entry, io.BytesIO(payload))
        if extra_member is not None:
            entry, raw = extra_member
            bundle.addfile(entry, io.BytesIO(raw) if raw is not None else None)
    manifest = {"schema": "terminal-supervisor-runtime-archive@1", "files": inventory,
                KEY: capability, "codex_version": supervisor.CLI_VERSION, "learned_requirements": [],
                "archive_sha256": supervisor._hash(archive / "runtime.tar.gz")}
    supervisor._json(archive / "manifest.json", manifest)
    return archive, manifest


def test_current_deadline_runtime_manifest_and_actual_member_bytes_agree(tmp_path):
    archive, manifest = _timeout_runtime_archive(tmp_path)
    verified = supervisor._require_current_timeout_runtime(archive, manifest)
    assert verified == {name: supervisor._hash(source)
                        for name, source in supervisor._timeout_runtime_sources().items()}


@pytest.mark.parametrize("name", supervisor.TIMEOUT_RUNTIME_SOURCE_NAMES)
def test_timeout_runtime_requires_each_current_manifest_source_pin(tmp_path, name):
    archive, manifest = _timeout_runtime_archive(tmp_path)
    required = supervisor.TIMEOUT_RUNTIME_ARCHIVE_PREFIX + name
    manifest["files"] = [row for row in manifest["files"] if row["path"] != required]
    with pytest.raises(ValueError, match="rebuild the runtime archive"):
        supervisor._require_current_timeout_runtime(archive, manifest)


@pytest.mark.parametrize("name", supervisor.TIMEOUT_RUNTIME_SOURCE_NAMES)
def test_timeout_runtime_rejects_stale_manifest_source_hash(tmp_path, name):
    archive, manifest = _timeout_runtime_archive(tmp_path)
    next(row for row in manifest["files"] if row["path"].endswith("/" + name))["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="rebuild the runtime archive"):
        supervisor._require_current_timeout_runtime(archive, manifest)


@pytest.mark.parametrize("name", supervisor.TIMEOUT_RUNTIME_SOURCE_NAMES)
def test_timeout_runtime_manifest_claim_cannot_hide_stale_actual_source_bytes(tmp_path, name):
    archive, manifest = _timeout_runtime_archive(tmp_path, stale_payload=name)
    assert supervisor._hash(archive / "runtime.tar.gz") == manifest["archive_sha256"]
    with pytest.raises(ValueError, match="rebuild the runtime archive"):
        supervisor._require_current_timeout_runtime(archive, manifest)


def test_timeout_runtime_rejects_missing_actual_source_member(tmp_path):
    archive, manifest = _timeout_runtime_archive(tmp_path, omit_payload="timeout_policy.py")
    with pytest.raises(ValueError, match="rebuild the runtime archive"):
        supervisor._require_current_timeout_runtime(archive, manifest)


def test_timeout_runtime_rejects_duplicate_actual_source_member(tmp_path):
    path = Path(supervisor.__file__).with_name("timeout_policy.py")
    raw = path.read_bytes()
    entry = tarfile.TarInfo(supervisor.TIMEOUT_RUNTIME_ARCHIVE_PREFIX + path.name)
    entry.size = len(raw)
    archive, manifest = _timeout_runtime_archive(tmp_path, extra_member=(entry, raw))
    with pytest.raises(ValueError, match="rebuild the runtime archive"):
        supervisor._require_current_timeout_runtime(archive, manifest)


@pytest.mark.parametrize("kind", ["symlink", "path_alias"])
def test_timeout_runtime_rejects_extraction_aliases_and_links(tmp_path, kind):
    entry = tarfile.TarInfo("source/alias")
    raw = None
    if kind == "symlink":
        entry.type, entry.linkname = tarfile.SYMTYPE, "benchmarks/agent_supervisor/container_coding"
    else:
        entry.name = supervisor.TIMEOUT_RUNTIME_ARCHIVE_PREFIX + "../container_coding/timeout_policy.py"
        raw = b"authored stale alias"
        entry.size = len(raw)
    archive, manifest = _timeout_runtime_archive(tmp_path, extra_member=(entry, raw))
    with pytest.raises(ValueError, match="rebuild the runtime archive"):
        supervisor._require_current_timeout_runtime(archive, manifest)


@pytest.mark.parametrize("field,value", [("bytes", True), ("path", []), ("bytes", 1)])
def test_timeout_runtime_rejects_malformed_required_manifest_entries(tmp_path, field, value):
    archive, manifest = _timeout_runtime_archive(tmp_path)
    manifest["files"][0][field] = value
    with pytest.raises(ValueError, match="rebuild the runtime archive"):
        supervisor._require_current_timeout_runtime(archive, manifest)


@pytest.mark.parametrize("failure", ["missing_manifest_pin", "stale_payload"])
def test_prepare_refuses_old_runtime_before_output_creation_or_harbor_dry_run(tmp_path, monkeypatch, failure):
    archive, manifest = _timeout_runtime_archive(tmp_path,
        stale_payload="terminal_container_supervisor.py" if failure == "stale_payload" else None)
    if failure == "missing_manifest_pin":
        manifest["files"] = [row for row in manifest["files"]
                             if not row["path"].endswith("/timeout_policy.py")]
        supervisor._json(archive / "manifest.json", manifest)
    dataset = _authored_prepare_dataset(tmp_path)
    output = tmp_path / "fresh-output"
    monkeypatch.setattr(supervisor.subprocess, "run", lambda *_args, **_kwargs: pytest.fail("Harbor launched"))
    with pytest.raises(ValueError, match="rebuild the runtime archive"):
        supervisor.prepare(dataset=dataset, output=output, archive=archive, arm="full")
    assert not output.exists()


def test_timeout_runtime_rejects_declared_archive_member_bound(tmp_path, monkeypatch):
    archive, manifest = _timeout_runtime_archive(tmp_path)
    monkeypatch.setattr(supervisor, "MAX_RUNTIME_ARCHIVE_MEMBERS", 3)
    with pytest.raises(ValueError, match="rebuild the runtime archive"):
        supervisor._require_current_timeout_runtime(archive, manifest)


def test_timeout_runtime_rejects_declared_archive_byte_bound(tmp_path, monkeypatch):
    archive, manifest = _timeout_runtime_archive(tmp_path)
    monkeypatch.setattr(supervisor, "MAX_RUNTIME_ARCHIVE_UNCOMPRESSED_BYTES", 1)
    with pytest.raises(ValueError, match="rebuild the runtime archive"):
        supervisor._require_current_timeout_runtime(archive, manifest)


def _authored_prepare_dataset(tmp_path):
    dataset = tmp_path / "dataset"
    task = dataset / supervisor.TASK
    task.mkdir(parents=True)
    (task / "instruction.md").write_text("Authored preparation transport fixture.\n")
    (task / "task.toml").write_text('version = "1.0"\n')
    (task / "environment").mkdir()
    (task / "tests").mkdir()
    return dataset


def test_prepare_retains_verified_archive_source_pins_and_actual_configured_budget(tmp_path, monkeypatch):
    archive, manifest = _timeout_runtime_archive(tmp_path)
    dataset = _authored_prepare_dataset(tmp_path)
    output = tmp_path / "prepared-output"
    calls = []

    def dry(command, **_kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout="Authored dry-run double.\n", stderr="")

    monkeypatch.setattr(supervisor.subprocess, "run", dry)
    result = supervisor.prepare(dataset=dataset, output=output, archive=archive, arm="full",
                                resource_profile=PROFILES[1])
    assert len(calls) == 1 and calls[0][-1] == "--dry-run"
    assert result["agent_timeout_seconds"] == 960
    assert result["resource_profile"] == PROFILES[1]
    assert result["provider_calls"] == 0
    assert result["runtime_timeout_source_sha256"] == {
        name: supervisor._hash(source) for name, source in supervisor._timeout_runtime_sources().items()
    }
    assert result["host_source_sha256"][str(Path(timeout_policy.__file__))] == native._hash(Path(timeout_policy.__file__))
    assert (output / "preparation.json").is_file()


@pytest.mark.parametrize("source_name", ["timeout_policy.py", "codex_usage_receipt.py"])
def test_prepare_rejects_deadline_source_drift_between_archive_check_and_host_snapshot(tmp_path, monkeypatch, source_name):
    archive, _ = _timeout_runtime_archive(tmp_path)
    dataset = _authored_prepare_dataset(tmp_path)
    output = tmp_path / "prepared-output"
    after_dry = False
    original_hash = supervisor._hash
    drifted = next(source for source in supervisor._timeout_runtime_sources().values()
                   if source.name == source_name)

    def dry(*_args, **_kwargs):
        nonlocal after_dry
        after_dry = True
        return SimpleNamespace(returncode=0, stdout="Authored dry-run double.\n", stderr="")

    def observed_hash(path):
        if after_dry and Path(path) == drifted:
            return "0" * 64
        return original_hash(path)

    monkeypatch.setattr(supervisor.subprocess, "run", dry)
    monkeypatch.setattr(supervisor, "_hash", observed_hash)
    with pytest.raises(ValueError, match="deadline sources changed during preflight"):
        supervisor.prepare(dataset=dataset, output=output, archive=archive, arm="full")
    assert not (output / "preparation.json").exists()


@pytest.mark.parametrize("required", [name for name in supervisor._timeout_runtime_sources()
                                    if not name.endswith(tuple('/' + part for part in supervisor.TIMEOUT_RUNTIME_SOURCE_NAMES))])
@pytest.mark.parametrize("kind", ["missing-pin", "duplicate-pin", "stale-pin", "stale-body", "missing-body", "duplicate-body"])
def test_usage_owners_require_current_unique_pins_and_actual_member_bytes(tmp_path, required, kind):
    source = supervisor._timeout_runtime_sources()[required]
    options = {}
    if kind == "stale-body":
        options['stale_payload'] = source.name
    elif kind == "missing-body":
        options['omit_payload'] = source.name
    elif kind == "duplicate-body":
        raw = source.read_bytes()
        entry = tarfile.TarInfo(required)
        entry.size = len(raw)
        options['extra_member'] = (entry, raw)
    archive, manifest = _timeout_runtime_archive(tmp_path, **options)
    row = next(item for item in manifest['files'] if item['path'] == required)
    if kind == "missing-pin":
        manifest['files'].remove(row)
    elif kind == "duplicate-pin":
        manifest['files'].append(dict(row))
    elif kind == "stale-pin":
        row['sha256'] = '0' * 64
        manifest['router_worker_capability']['source_sha256'][required] = row['sha256']
    supervisor._json(archive / 'manifest.json', manifest)
    assert supervisor._hash(archive / 'runtime.tar.gz') == manifest['archive_sha256']
    with pytest.raises(ValueError, match="rebuild the runtime archive"):
        supervisor._require_current_timeout_runtime(archive, manifest)


def test_selected_scope_cannot_replace_an_unconditional_usage_owner_with_a_copy(tmp_path):
    archive, manifest = _timeout_runtime_archive(tmp_path)
    required = 'source/ipfs_accelerate_py/agent_supervisor/runtime/cumulative_usage_reconciliation.py'
    original = supervisor._timeout_runtime_sources()[required]
    copied = tmp_path / original.name
    copied.write_bytes(original.read_bytes())
    with pytest.raises(ValueError, match="rebuild the runtime archive"):
        supervisor._require_current_timeout_runtime(archive, manifest, selected_sources={required: copied})
