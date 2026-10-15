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
from harbor.models.job.config import JobConfig
from harbor.trial.trial import Trial


def _configs(tmp_path, seconds=None):
    options = {} if seconds is None else {"agent_timeout_seconds": seconds}
    return [
        native.config_for(tmp_path / "dataset", tmp_path / "native", **options),
        *[supervisor.config_for(tmp_path / "dataset", tmp_path / arm,
                               tmp_path / "archive", arm, **options)
          for arm in ("full", "no-index")],
    ]


def _assert_common(configs, seconds):
    baseline = configs[0]
    for config in configs:
        agent = config["agents"][0]
        assert agent["override_timeout_sec"] == agent["max_timeout_sec"] == seconds
        assert agent["override_setup_timeout_sec"] == native.AGENT_SETUP_TIMEOUT_SECONDS == 1800
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
        assert config["agents"][0]["kwargs"]["agent_timeout_seconds"] == seconds


def test_fresh_default_is_ten_minutes_for_every_arm(tmp_path):
    _assert_common(_configs(tmp_path), 600)


@pytest.mark.parametrize("seconds", [120, 300, 450, 600])
def test_explicit_budget_is_shared_without_changing_task_resources(tmp_path, seconds):
    _assert_common(_configs(tmp_path, seconds), seconds)


@pytest.mark.parametrize("seconds", [True, False, "600", 600.0, None, 119, 601])
@pytest.mark.parametrize("arm", ["native", "full", "no-index"])
def test_invalid_config_budget_is_rejected(tmp_path, seconds, arm):
    with pytest.raises(ValueError, match="agent timeout"):
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
    with pytest.raises(ValueError, match="agent timeout"):
        module.prepare(**options)


def _native_trial(tmp_path, *, seconds, omit_timeout=False, pin_policy=False):
    dataset = tmp_path / "dataset"
    task = dataset / native.TASK
    task.mkdir(parents=True)
    (task / "instruction.md").write_text("Authored public instruction.\n")
    (task / "task.toml").write_text('version = "1.0"\n')
    output = tmp_path / "trial"
    output.mkdir()
    config = native.config_for(dataset, output, agent_timeout_seconds=seconds)
    native._json(output / "config.json", config)
    prepared = {
        "prepared": True, "dataset": str(dataset), "config_sha256": native._hash(output / "config.json"),
        "task_input_sha256": native._task_hashes(task),
        "collector_source_sha256": native._hash(Path(native.__file__)),
        "native_adapter_source_sha256": {}, "command": ["authored-harbor"],
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
        "trial_name": "authored-trial", "task_name": "terminal-bench/" + native.TASK,
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
def test_native_collection_uses_each_immutable_preparation_budget(tmp_path, seconds, omit_timeout, pin_policy):
    output, _, _ = _native_trial(tmp_path, seconds=seconds, omit_timeout=omit_timeout,
                                 pin_policy=pin_policy)
    original = (output / "preparation.json").read_bytes()
    result = native.collect(output)
    assert result["agent_timeout_seconds"] == seconds
    assert result["agent_timeout_config_unchanged"] is True
    assert result["trials"][0]["exact_trial_profile_matches"] is True
    assert result["complete_single_trial_receipt"] is True
    assert result["timeout_policy_source_unchanged"] is (True if pin_policy else None)
    assert result["benchmark_advantage_claimed"] is False
    assert (output / "preparation.json").read_bytes() == original


def test_missing_policy_pin_cannot_report_policy_source_verified(tmp_path):
    output, _, _ = _native_trial(tmp_path, seconds=600)
    result = native.collect(output)
    assert result["timeout_policy_source_sha256"] is None
    assert result["timeout_policy_source_unchanged"] is None


def test_native_collection_requires_new_preparation_setup_allowance_to_match_trial(tmp_path):
    output, prepared, record = _native_trial(tmp_path, seconds=600, pin_policy=True)
    prepared["agent_setup_timeout_seconds"] = 1800
    native._json(output / "preparation.json", prepared)
    original = (output / "preparation.json").read_bytes()
    result = native.collect(output)
    assert result["agent_setup_timeout_seconds"] == 1800
    assert result["agent_setup_timeout_basis"] == "preparation"
    assert result["agent_setup_timeout_config_unchanged"] is True
    assert result["trials"][0]["agent_setup_timeout_config_unchanged"] is True
    assert result["complete_single_trial_receipt"] is True
    record["config"]["agent"]["override_setup_timeout_sec"] = 360
    native._json(output / "jobs" / ("native-codex-" + native.TASK) / "authored-trial/result.json", record)
    result = native.collect(output)
    assert result["agent_setup_timeout_config_unchanged"] is True
    assert result["trials"][0]["agent_setup_timeout_config_unchanged"] is False
    assert result["trials"][0]["exact_trial_profile_matches"] is False
    assert result["complete_single_trial_receipt"] is False
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
    assert result["agent_setup_timeout_seconds"] is None
    assert result["agent_setup_timeout_basis"] == "unknown_framework_default"
    assert result["agent_setup_timeout_config_unchanged"] is None
    assert result["trials"][0]["agent_setup_timeout_seconds"] is None
    assert result["trials"][0]["exact_trial_profile_matches"] is True
    assert result["complete_single_trial_receipt"] is True
    assert (output / "preparation.json").read_bytes() == original


@pytest.mark.parametrize("field", ["override_timeout_sec", "max_timeout_sec"])
def test_native_collection_marks_trial_budget_mismatch_false(tmp_path, field):
    output, _, record = _native_trial(tmp_path, seconds=600, pin_policy=True)
    record["config"]["agent"][field] = 300
    path = output / "jobs" / ("native-codex-" + native.TASK) / "authored-trial/result.json"
    native._json(path, record)
    result = native.collect(output)
    assert result["agent_timeout_config_unchanged"] is True
    assert result["trials"][0]["exact_trial_profile_matches"] is False
    assert result["complete_single_trial_receipt"] is False


def test_native_collection_marks_prepared_budget_config_mismatch_false(tmp_path):
    output, prepared, _ = _native_trial(tmp_path, seconds=600, pin_policy=True)
    prepared["agent_timeout_seconds"] = 450
    native._json(output / "preparation.json", prepared)
    result = native.collect(output)
    assert result["agent_timeout_config_unchanged"] is False
    assert result["trials"][0]["exact_trial_profile_matches"] is False
    assert result["complete_single_trial_receipt"] is False


def test_native_collection_rejects_modified_immutable_configuration(tmp_path):
    output, _, _ = _native_trial(tmp_path, seconds=600, pin_policy=True)
    config = json.loads((output / "config.json").read_text())
    config["agents"][0]["override_timeout_sec"] = 300
    native._json(output / "config.json", config)
    with pytest.raises(ValueError, match="baseline config changed"):
        native.collect(output)


def test_native_execution_rejects_changed_pinned_timeout_policy_before_launch(tmp_path, monkeypatch):
    output, prepared, _ = _native_trial(tmp_path, seconds=600, pin_policy=True)
    prepared["timeout_policy_source_sha256"] = "0" * 64
    native._json(output / "preparation.json", prepared)
    monkeypatch.setattr(native.subprocess, "run", lambda *_args, **_kwargs: pytest.fail("Harbor launched"))
    with pytest.raises(ValueError, match="timeout policy changed"):
        native.execute(output)
    assert not (output / "invocation.json").exists()


def test_native_execution_rejects_preparation_budget_mismatch_before_launch(tmp_path, monkeypatch):
    output, prepared, _ = _native_trial(tmp_path, seconds=600, pin_policy=True)
    prepared["agent_timeout_seconds"] = 450
    native._json(output / "preparation.json", prepared)
    monkeypatch.setattr(native.subprocess, "run", lambda *_args, **_kwargs: pytest.fail("Harbor launched"))
    with pytest.raises(ValueError, match="timeout differs"):
        native.execute(output)
    assert not (output / "invocation.json").exists()


@pytest.mark.parametrize("module", [native, supervisor])
@pytest.mark.parametrize("mismatch", ["preparation", "configuration"])
def test_execution_rejects_setup_allowance_mismatch_before_launch(tmp_path, monkeypatch, module, mismatch):
    output, prepared, _ = _native_trial(tmp_path, seconds=600, pin_policy=True)
    if module is supervisor:
        prepared["arm"] = "full"
        config = supervisor.config_for(Path(prepared["dataset"]), output, tmp_path / "archive", "full")
    else:
        config = json.loads((output / "config.json").read_text())
    prepared["agent_setup_timeout_seconds"] = 360 if mismatch == "preparation" else 1800
    if mismatch == "configuration":
        config["agents"][0]["override_setup_timeout_sec"] = 360
    native._json(output / "config.json", config)
    prepared["config_sha256"] = native._hash(output / "config.json")
    native._json(output / "preparation.json", prepared)
    monkeypatch.setattr(module.subprocess, "run", lambda *_args, **_kwargs: pytest.fail("Harbor launched"))
    with pytest.raises(ValueError, match="setup timeout differs"):
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
    assert result["agent_timeout_seconds"] == 300
    assert (output / "config.json").read_bytes() == original


@pytest.mark.parametrize("seconds", [300, 450, 600])
def test_supervisor_collection_records_configured_adapter_and_harbor_budget(tmp_path, seconds):
    output, prepared, _ = _native_trial(tmp_path, seconds=seconds)
    prepared["arm"] = "full"
    native._json(output / "preparation.json", prepared)
    config = supervisor.config_for(Path(prepared["dataset"]), output, tmp_path / "archive", "full",
                                   agent_timeout_seconds=seconds)
    native._json(output / "config.json", config)
    result = supervisor.collect(output)
    assert result["agent_timeout_seconds"] == seconds
    assert result["agent_timeout_config_unchanged"] is True
    config["agents"][0]["kwargs"]["agent_timeout_seconds"] = 120
    native._json(output / "config.json", config)
    assert supervisor.collect(output)["agent_timeout_config_unchanged"] is False


def test_legacy_supervisor_collection_keeps_implicit_five_minute_adapter_budget(tmp_path):
    output, prepared, _ = _native_trial(tmp_path, seconds=300, omit_timeout=True)
    prepared["arm"] = "no-index"
    native._json(output / "preparation.json", prepared)
    config = supervisor.config_for(Path(prepared["dataset"]), output, tmp_path / "archive", "no-index",
                                   agent_timeout_seconds=300)
    del config["agents"][0]["kwargs"]["agent_timeout_seconds"]
    native._json(output / "config.json", config)
    result = supervisor.collect(output)
    assert result["agent_timeout_seconds"] == 300
    assert result["agent_timeout_config_unchanged"] is True
    assert result["agent_setup_timeout_seconds"] == 1800
    assert result["agent_setup_timeout_basis"] == "retained_configuration_override"
    assert result["agent_setup_timeout_config_unchanged"] is None


def test_supervisor_execution_rejects_preparation_budget_mismatch_before_launch(tmp_path, monkeypatch):
    output, prepared, _ = _native_trial(tmp_path, seconds=600, pin_policy=True)
    prepared.update(arm="full", agent_timeout_seconds=450)
    config = supervisor.config_for(Path(prepared["dataset"]), output, tmp_path / "archive", "full")
    native._json(output / "config.json", config)
    prepared["config_sha256"] = native._hash(output / "config.json")
    native._json(output / "preparation.json", prepared)
    monkeypatch.setattr(supervisor.subprocess, "run", lambda *_args, **_kwargs: pytest.fail("Harbor launched"))
    with pytest.raises(ValueError, match="timeout differs"):
        supervisor.execute(output)
    assert not (output / "invocation.json").exists()


@pytest.mark.parametrize("module", [native, supervisor])
@pytest.mark.parametrize("seconds", [None, 450])
def test_prepare_cli_forwards_default_or_explicit_budget(module, seconds, tmp_path, monkeypatch, capsys):
    observed = {}
    monkeypatch.setattr(module, "prepare", lambda **kwargs: observed.update(kwargs) or {"prepared": True})
    argv = ["benchmark", "prepare", "--dataset", str(tmp_path / "dataset"),
            "--output", str(tmp_path / "output")]
    if module is supervisor:
        argv += ["--archive", str(tmp_path / "archive")]
    if seconds is not None:
        argv += ["--agent-timeout-seconds", str(seconds)]
    monkeypatch.setattr(sys, "argv", argv)
    module.main()
    assert observed["agent_timeout_seconds"] == (600 if seconds is None else seconds)
    assert json.loads(capsys.readouterr().out) == {"prepared": True}


@pytest.mark.parametrize("module", [native, supervisor])
def test_execute_cli_uses_retained_preparation_instead_of_overriding_it(module, tmp_path, monkeypatch, capsys):
    observed = []
    monkeypatch.setattr(module, "execute", lambda output: observed.append(output) or {"executed": True})
    monkeypatch.setattr(sys, "argv", ["benchmark", "execute", "--output", str(tmp_path / "output"),
                                     "--agent-timeout-seconds", "450"])
    module.main()
    assert observed == [tmp_path / "output"]
    assert json.loads(capsys.readouterr().out) == {"executed": True}


def _timeout_runtime_archive(tmp_path, *, stale_payload=None, omit_payload=None, extra_member=None):
    archive = tmp_path / "runtime"
    archive.mkdir()
    inventory = []
    with tarfile.open(archive / "runtime.tar.gz", "w:gz") as bundle:
        for name in supervisor.TIMEOUT_RUNTIME_SOURCE_NAMES:
            raw = Path(supervisor.__file__).with_name(name).read_bytes()
            path = supervisor.TIMEOUT_RUNTIME_ARCHIVE_PREFIX + name
            inventory.append({"path": path, "bytes": len(raw),
                              "sha256": supervisor._hash(Path(supervisor.__file__).with_name(name)),
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
                "archive_sha256": supervisor._hash(archive / "runtime.tar.gz")}
    supervisor._json(archive / "manifest.json", manifest)
    return archive, manifest


def test_current_deadline_runtime_manifest_and_actual_member_bytes_agree(tmp_path):
    archive, manifest = _timeout_runtime_archive(tmp_path)
    verified = supervisor._require_current_timeout_runtime(archive, manifest)
    assert verified == {row["path"]: row["sha256"] for row in manifest["files"]}


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
        manifest["files"].pop()
        supervisor._json(archive / "manifest.json", manifest)
    dataset = tmp_path / "dataset"
    dataset.mkdir()
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
                                agent_timeout_seconds=450)
    assert len(calls) == 1 and calls[0][-1] == "--dry-run"
    assert result["agent_timeout_seconds"] == 450
    assert result["provider_calls"] == 0
    assert result["runtime_timeout_source_sha256"] == {
        row["path"]: row["sha256"] for row in manifest["files"]
    }
    assert result["host_source_sha256"][str(Path(timeout_policy.__file__))] == native._hash(Path(timeout_policy.__file__))
    assert (output / "preparation.json").is_file()


def test_prepare_rejects_deadline_source_drift_between_archive_check_and_host_snapshot(tmp_path, monkeypatch):
    archive, _ = _timeout_runtime_archive(tmp_path)
    dataset = _authored_prepare_dataset(tmp_path)
    output = tmp_path / "prepared-output"
    after_dry = False
    original_hash = supervisor._hash

    def dry(*_args, **_kwargs):
        nonlocal after_dry
        after_dry = True
        return SimpleNamespace(returncode=0, stdout="Authored dry-run double.\n", stderr="")

    def observed_hash(path):
        if after_dry and Path(path) == Path(timeout_policy.__file__):
            return "0" * 64
        return original_hash(path)

    monkeypatch.setattr(supervisor.subprocess, "run", dry)
    monkeypatch.setattr(supervisor, "_hash", observed_hash)
    with pytest.raises(ValueError, match="deadline sources changed during preflight"):
        supervisor.prepare(dataset=dataset, output=output, archive=archive, arm="full")
    assert not (output / "preparation.json").exists()
