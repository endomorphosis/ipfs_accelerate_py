"""Host nomination snapshots preserve bytes and refuse drift before invocation.

Authored public datasets and runtime archives exercise the actual preparation
and retained-configuration guards. Harbor dry-run is an explicit process
double. These checks invoke no provider, benchmark task, Docker, or verifier.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import sys
from types import SimpleNamespace

import pytest
from harbor.models.job.config import JobConfig

from benchmarks.agent_supervisor.container_coding import full_supervisor_benchmark as benchmark
from test.api.test_benchmark_prompt_options import DEFAULTS
from test.api.test_terminal_decision_archive import _decision_runtime_archive
from test.api.test_terminal_decision_configuration import DECISION
from test.api.test_terminal_decision_nomination_configuration import NOMINATION
from test.api.test_current_terminal_archive_fixtures import (
    _authored_prepare_dataset, _timeout_runtime_archive,
)

HOST_PATH = "/authored-host/decision/nomination.json"
RAW = b'''  {\r\n  "schema": "supervisor-semantic-decision-nomination@1",\r\n  "policy": {"required_capsule_ids": ["authored-required"],\r\n             "optional_capsule_ids": ["authored-optional"],\r\n             "selected_optional_capsule_ids": []}\r\n}\n\t'''
SOURCE = {
    "semantic_decision_nomination_source": HOST_PATH,
    "semantic_decision_nomination_sha256": hashlib.sha256(RAW).hexdigest(),
    "semantic_decision_max_calls": 3,
    "semantic_decision_max_total_input_tokens": None,
    "disable_semantic_typed_nomination": False,
}


def _source(tmp_path, raw=RAW):
    tmp_path.chmod(0o755)
    source = tmp_path / "authored-nomination.json"
    source.write_bytes(raw)
    source.chmod(0o644)
    return source, {**SOURCE, "semantic_decision_nomination_source": str(source),
                    "semantic_decision_nomination_sha256": hashlib.sha256(raw).hexdigest()}


def _prepared(tmp_path, monkeypatch, *, selected=SOURCE):
    archive, _ = _decision_runtime_archive(tmp_path)
    archive.chmod(0o755)
    dataset = _authored_prepare_dataset(tmp_path)
    dataset.chmod(0o755)
    (dataset / benchmark.TASK).chmod(0o755)
    output = tmp_path / "prepared"
    calls = []

    def dry(command, **kwargs):
        assert command[-1] == "--dry-run"
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout="Authored dry-run only.\n", stderr="")

    monkeypatch.setattr(benchmark.subprocess, "run", dry)
    if selected is SOURCE:
        source, options = _source(tmp_path)
    else:
        source, options = None, selected
    prepared = benchmark.prepare(dataset=dataset, output=output, archive=archive,
                                 arm="full", **options)
    assert len(calls) == 1
    return output, prepared, source, calls


@pytest.mark.parametrize("limits", [
    {}, {"semantic_decision_max_calls": 2, "semantic_decision_max_total_input_tokens": 24000},
], ids=["defaults", "bounded"])
def test_source_configuration_is_separate_closed_variant_and_preserves_defaults(tmp_path, monkeypatch, limits):
    def unexpected(*args, **kwargs):
        pytest.fail("lexical configuration validation performed source I/O")

    monkeypatch.setattr(benchmark, "read_nomination_source", unexpected)
    monkeypatch.setattr(benchmark, "capture_nomination_snapshot", unexpected)
    options = {**SOURCE, **limits, "disable_semantic_typed_nomination": False}
    assert benchmark._prompt_options("full") == DEFAULTS
    assert benchmark._prompt_options("full", **DECISION) == {**DEFAULTS, **DECISION}
    assert benchmark._prompt_options("full", **NOMINATION) == {**DEFAULTS, **NOMINATION, "disable_semantic_typed_nomination": False}
    selected = benchmark._prompt_options("full", **{
        **options, "semantic_decision_nomination_source": Path(HOST_PATH),
    })
    assert selected == {**DEFAULTS, **options} and len(selected) == 10
    assert "semantic_decision_nomination_artifact" not in selected
    default = benchmark.config_for(tmp_path / "dataset", tmp_path / "output", tmp_path / "archive", "full")
    config = benchmark.config_for(tmp_path / "dataset", tmp_path / "output", tmp_path / "archive", "full", **options)
    wire = JobConfig.model_validate(config, extra="forbid").model_dump(mode="json")
    assert wire["agents"][0]["kwargs"] == {**default["agents"][0]["kwargs"], **options}
    for key in ("tasks", "environment", "verifier", "n_attempts", "n_concurrent_trials", "timeout_multiplier", "retry"):
        assert config[key] == default[key]
    assert not any("task_id" in key or "task_key" in key or "destination" in key
                   for key in wire["agents"][0]["kwargs"])


@pytest.mark.parametrize("arm,options", [
    ("no-index", SOURCE),
    ("full", {"semantic_decision_nomination_source": HOST_PATH}),
    ("full", {"semantic_decision_nomination_sha256": SOURCE["semantic_decision_nomination_sha256"]}),
    ("full", {**SOURCE, "semantic_decision_nomination_artifact": NOMINATION["semantic_decision_nomination_artifact"]}),
    ("full", {**SOURCE, **DECISION}),
    ("full", {**SOURCE, "semantic_decision_policy_sha256": "c" * 64}),
    ("full", {**SOURCE, "semantic_decision_max_calls": 0}),
    ("full", {**SOURCE, "semantic_decision_max_total_input_tokens": 0}),
    ("full", {**SOURCE, "semantic_slot_policy_revision": "closed-typed-identifiers@2"}),
    ("full", {**SOURCE, "semantic_slot_policy_revision": "closed-typed-identifiers@2",
              "semantic_context_projection": True}),
], ids=["arm", "missing-sha", "orphan-sha", "child-mixed", "policy-mixed", "partial-policy",
        "calls", "total", "codec", "projection"])
def test_invalid_source_variant_refuses_before_io(tmp_path, monkeypatch, arm, options):
    def unexpected(*args, **kwargs):
        pytest.fail("invalid source variant reached preparation I/O")

    unread = SimpleNamespace(resolve=unexpected)
    monkeypatch.setattr(benchmark, "read_nomination_source", unexpected)
    monkeypatch.setattr(benchmark.subprocess, "run", unexpected)
    with pytest.raises(ValueError):
        benchmark.config_for(unread, tmp_path / "output", unread, arm, **options)
    with pytest.raises(ValueError):
        benchmark.prepare(dataset=unread, output=tmp_path / "output", archive=unread,
                          arm=arm, **options)
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("source", [
    "", "relative/nomination.json", "/host/./nomination.json", "/host/../nomination.json",
    "/host//nomination.json", "/host/nomination\\file.json", "/host/nomination\x00.json",
    "/" + "a" * 4096, "file:///host/nomination.json", 123,
], ids=["empty", "relative", "dot", "parent", "slash", "backslash", "nul", "long", "uri", "type"])
def test_invalid_source_spelling_refuses_without_reads(monkeypatch, source):
    monkeypatch.setattr(benchmark, "read_nomination_source", lambda **kwargs:
                        pytest.fail("invalid lexical path reached source read"))
    with pytest.raises(ValueError):
        benchmark._prompt_options("full", **{**SOURCE, "semantic_decision_nomination_source": source})


@pytest.mark.parametrize("digest", ["A" * 64, "a" * 63, "", 123],
                         ids=["uppercase", "short", "empty", "type"])
def test_invalid_source_digest_refuses_without_reads(monkeypatch, digest):
    monkeypatch.setattr(benchmark, "read_nomination_source", lambda **kwargs:
                        pytest.fail("invalid digest reached source read"))
    with pytest.raises(ValueError):
        benchmark._prompt_options("full", **{**SOURCE, "semantic_decision_nomination_sha256": digest})


@pytest.mark.parametrize("change", ["incomplete", "mixed", "child-variant", "source", "sha", "budget"],
                         ids=["incomplete", "mixed", "child-variant", "source", "sha", "budget"])
def test_source_observation_detects_changed_or_nonclosed_pins(change):
    prepared = {"arm": "full", "prompt_comparison_options": {**DEFAULTS, **SOURCE}}
    configured = dict(SOURCE)
    if change == "incomplete":
        prepared["prompt_comparison_options"].pop("semantic_decision_nomination_sha256")
    elif change == "mixed":
        prepared["prompt_comparison_options"].update(NOMINATION)
    elif change == "child-variant":
        configured = dict(NOMINATION)
    elif change == "source":
        configured["semantic_decision_nomination_source"] = "/other-host/nomination.json"
    elif change == "sha":
        configured["semantic_decision_nomination_sha256"] = "f" * 64
    else:
        configured["semantic_decision_max_calls"] = 2
    observed = benchmark._prompt_option_observation(prepared, configured)
    assert observed["prompt_comparison_options_valid"] is True
    assert observed["prompt_comparison_config_unchanged"] is False


def test_cli_forwards_source_pair_without_child_path_or_task_identity(tmp_path, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(benchmark, "prepare", lambda **kwargs: calls.append(kwargs) or {"provider_calls": 0})
    monkeypatch.setattr(sys, "argv", [
        "full-supervisor", "prepare", "--output", str(tmp_path / "output"),
        "--dataset", str(tmp_path / "dataset"), "--archive", str(tmp_path / "archive"),
        "--semantic-decision-nomination-source", HOST_PATH,
        "--semantic-decision-nomination-sha256", SOURCE["semantic_decision_nomination_sha256"],
        "--semantic-decision-max-calls", "2", "--semantic-decision-max-total-input-tokens", "24000",
    ])
    benchmark.main()
    assert json.loads(capsys.readouterr().out) == {"provider_calls": 0}
    assert len(calls) == 1
    assert {key: calls[0][key] for key in SOURCE} == {
        **SOURCE, "semantic_decision_max_calls": 2, "semantic_decision_max_total_input_tokens": 24000,
    }
    assert calls[0].get("semantic_decision_nomination_artifact") is None
    assert calls[0].get("semantic_decision_policy_artifact") is None
    assert not any("task_id" in key or "task_key" in key or "destination" in key for key in calls[0])


def test_preparation_snapshots_exact_raw_bytes_and_rewires_only_host_source(tmp_path, monkeypatch):
    output, prepared, source, calls = _prepared(tmp_path, monkeypatch)
    snapshot = prepared["semantic_decision_nomination_snapshot"]
    assert set(snapshot) == {"schema", "artifact", "sha256", "bytes"}
    assert snapshot["schema"] == "terminal-semantic-decision-nomination-snapshot@1"
    captured = Path(snapshot["artifact"])
    assert captured.parent == output and captured != source
    assert captured.read_bytes() == RAW == source.read_bytes()
    assert snapshot["bytes"] == len(RAW) and snapshot["sha256"] == SOURCE["semantic_decision_nomination_sha256"]
    assert stat.S_IMODE(captured.stat().st_mode) == 0o444 and captured.stat().st_nlink == 1
    expected = {**DEFAULTS, **SOURCE, "semantic_decision_nomination_source": str(captured)}
    assert prepared["prompt_comparison_options"] == expected
    config = json.loads((output / "config.json").read_text())
    kwargs = config["agents"][0]["kwargs"]
    assert kwargs["semantic_decision_nomination_source"] == str(captured)
    assert kwargs["semantic_decision_nomination_sha256"] == SOURCE["semantic_decision_nomination_sha256"]
    assert "semantic_decision_nomination_artifact" not in kwargs
    assert not any("task_id" in key or "task_key" in key or "destination" in key for key in kwargs)
    assert benchmark._prompt_option_observation(prepared, kwargs)["prompt_comparison_config_unchanged"] is True
    assert len(calls) == 1 and calls[0][0][-1] == "--dry-run"
    assert prepared["provider_calls"] == 0 and prepared["benchmark_advantage_claimed"] is False
    source.unlink()
    assert benchmark.verify_nomination_snapshot(snapshot, directory=output) == RAW


def test_source_execution_accepts_relative_output_with_original_source_removed(tmp_path, monkeypatch):
    output, prepared, source, _ = _prepared(tmp_path, monkeypatch)
    source.unlink()
    monkeypatch.chdir(tmp_path)
    calls, collected = [], []

    def invocation(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0)

    def collect(path):
        collected.append(path)
        return {"provider_calls": 0, "authored_collection": True}

    monkeypatch.setattr(benchmark.subprocess, "run", invocation)
    monkeypatch.setattr(benchmark, "collect", collect)
    relative = Path(output.name)
    result = benchmark.execute(relative)
    assert calls == [prepared["command"]] and collected == [relative.resolve()]
    assert result["harbor_returncode"] == 0 and result["provider_calls"] == 0
    assert (output / "invocation.json").is_file()


@pytest.mark.parametrize("fault", ["missing", "directory", "symlink", "changed", "malformed", "foreign-schema"],
                         ids=["missing", "directory", "symlink", "changed", "malformed", "schema"])
def test_bad_source_refuses_before_output_or_any_process(tmp_path, monkeypatch, fault):
    source, options = _source(tmp_path)
    if fault == "missing":
        source.unlink()
    elif fault == "directory":
        source.unlink()
        source.mkdir(mode=0o755)
    elif fault == "symlink":
        target = source.with_name("other.json")
        source.rename(target)
        source.symlink_to(target)
    elif fault == "changed":
        source.write_bytes(RAW + b" ")
    elif fault == "malformed":
        source.write_bytes(b"not json")
        options["semantic_decision_nomination_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    else:
        source.write_bytes(RAW.replace(b"supervisor-semantic-decision-nomination@1", b"foreign-semantic-decision-nomination@1"))
        options["semantic_decision_nomination_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()

    def unexpected(*args, **kwargs):
        pytest.fail("invalid source reached output or process work")

    unread = SimpleNamespace(resolve=unexpected)
    monkeypatch.setattr(benchmark.subprocess, "run", unexpected)
    with pytest.raises((ValueError, OSError)):
        benchmark.prepare(dataset=unread, output=tmp_path / "output", archive=unread,
                          arm="full", **options)
    assert not (tmp_path / "output").exists()


def test_preparation_rechecks_captured_snapshot_before_dry_run(tmp_path, monkeypatch):
    original = benchmark.capture_nomination_snapshot

    def changed(**kwargs):
        snapshot = original(**kwargs)
        path = Path(snapshot["artifact"])
        path.chmod(0o644)
        path.write_bytes(RAW + b" ")
        path.chmod(0o444)
        return snapshot

    monkeypatch.setattr(benchmark, "capture_nomination_snapshot", changed)
    monkeypatch.setattr(benchmark.subprocess, "run", lambda *args, **kwargs:
                        pytest.fail("changed captured snapshot reached Harbor dry-run"))
    source, options = _source(tmp_path)
    archive, _ = _decision_runtime_archive(tmp_path)
    dataset = _authored_prepare_dataset(tmp_path)
    output = tmp_path / "output"
    with pytest.raises(ValueError):
        benchmark.prepare(dataset=dataset, output=output, archive=archive, arm="full", **options)
    assert not (output / "dry-run.stdout").exists()


@pytest.mark.parametrize("fault", ["changed", "deleted", "mode", "symlink", "hardlink"],
                         ids=["changed", "deleted", "mode", "symlink", "hardlink"])
def test_snapshot_drift_refuses_execute_before_invocation(tmp_path, monkeypatch, fault):
    output, prepared, source, _ = _prepared(tmp_path, monkeypatch)
    captured = Path(prepared["semantic_decision_nomination_snapshot"]["artifact"])
    if fault == "changed":
        captured.chmod(0o644)
        captured.write_bytes(RAW + b" ")
        captured.chmod(0o444)
    elif fault == "deleted":
        captured.unlink()
    elif fault == "mode":
        captured.chmod(0o644)
    elif fault == "symlink":
        captured.unlink()
        captured.symlink_to(source)
    else:
        os.link(captured, output / "snapshot-alias.json")
    monkeypatch.setattr(benchmark.subprocess, "run", lambda *args, **kwargs:
                        pytest.fail("drifted snapshot reached process invocation"))
    with pytest.raises((ValueError, OSError)):
        benchmark.execute(output)
    assert not (output / "invocation.json").exists()


@pytest.mark.parametrize("fault", ["missing", "foreign-path", "foreign-sha", "extra-field", "foreign-schema"],
                         ids=["missing", "path", "sha", "extra", "schema"])
def test_source_execution_requires_its_exact_snapshot_descriptor(tmp_path, monkeypatch, fault):
    output, prepared, source, _ = _prepared(tmp_path, monkeypatch)
    snapshot = prepared["semantic_decision_nomination_snapshot"]
    if fault == "missing":
        prepared.pop("semantic_decision_nomination_snapshot")
    elif fault == "foreign-path":
        snapshot["artifact"] = str(source)
    elif fault == "foreign-sha":
        snapshot["sha256"] = "f" * 64
    elif fault == "extra-field":
        snapshot["task_id"] = "caller-must-not-override"
    else:
        snapshot["schema"] = "foreign@1"
    benchmark._json(output / "preparation.json", prepared)
    monkeypatch.setattr(benchmark.subprocess, "run", lambda *args, **kwargs:
                        pytest.fail("foreign snapshot descriptor reached invocation"))
    with pytest.raises(ValueError):
        benchmark.execute(output)
    assert not (output / "invocation.json").exists()


@pytest.mark.parametrize("selected", [DECISION, NOMINATION], ids=["prebound", "child-nomination"])
def test_snapshot_descriptor_is_forbidden_without_host_source_variant(tmp_path, monkeypatch, selected):
    output, prepared, _, _ = _prepared(tmp_path, monkeypatch, selected=selected)
    assert "semantic_decision_nomination_snapshot" not in prepared
    prepared["semantic_decision_nomination_snapshot"] = {
        "schema": "terminal-semantic-decision-nomination-snapshot@1", "artifact": HOST_PATH,
        "sha256": SOURCE["semantic_decision_nomination_sha256"], "bytes": len(RAW),
    }
    benchmark._json(output / "preparation.json", prepared)
    monkeypatch.setattr(benchmark.subprocess, "run", lambda *args, **kwargs:
                        pytest.fail("unexpected host snapshot reached invocation"))
    with pytest.raises(ValueError, match="host nomination snapshot requires"):
        benchmark.execute(output)
    assert not (output / "invocation.json").exists()


def test_source_variant_refuses_deadline_only_archive_before_output(tmp_path, monkeypatch):
    source, options = _source(tmp_path)
    archive, _ = _timeout_runtime_archive(tmp_path)
    dataset = _authored_prepare_dataset(tmp_path)
    output = tmp_path / "output"
    monkeypatch.setattr(benchmark.subprocess, "run", lambda *args, **kwargs:
                        pytest.fail("old archive reached Harbor dry-run"))
    with pytest.raises(ValueError, match="selected decision"):
        benchmark.prepare(dataset=dataset, output=output, archive=archive, arm="full", **options)
    assert not output.exists() and source.read_bytes() == RAW


def test_selected_source_pins_include_host_delivery_and_harbor_adapter():
    sources = benchmark._decision_runtime_sources()
    for name in ("semantic_decision_nomination_delivery.py", "full_supervisor_harbor_agent.py", "terminal_deployment.py"):
        key = benchmark.TIMEOUT_RUNTIME_ARCHIVE_PREFIX + name
        assert key in sources and sources[key] == Path(benchmark.__file__).with_name(name)
    assert len(sources) == 32


@pytest.mark.parametrize("name", ["semantic_decision_nomination_delivery.py", "full_supervisor_harbor_agent.py", "terminal_deployment.py"],
                         ids=["delivery", "harbor", "deployment"])
def test_source_execution_requires_each_delivery_capability_pin(tmp_path, monkeypatch, name):
    output, prepared, _, _ = _prepared(tmp_path, monkeypatch)
    prepared["runtime_decision_source_sha256"].pop(benchmark.TIMEOUT_RUNTIME_ARCHIVE_PREFIX + name)
    benchmark._json(output / "preparation.json", prepared)
    monkeypatch.setattr(benchmark.subprocess, "run", lambda *args, **kwargs:
                        pytest.fail("missing delivery capability reached invocation"))
    with pytest.raises(ValueError, match="decision source pins"):
        benchmark.execute(output)
    assert not (output / "invocation.json").exists()
