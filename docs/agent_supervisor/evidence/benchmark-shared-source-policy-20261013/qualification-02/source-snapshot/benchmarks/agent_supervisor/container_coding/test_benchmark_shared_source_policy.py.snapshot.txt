"""Frozen shared display recipes reach the benchmark without provider/Docker IO."""
import asyncio
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import shlex
import sys
import tarfile
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchmarks.agent_supervisor.container_coding import benchmark_controls as controls
from benchmarks.agent_supervisor.container_coding import full_supervisor_benchmark as benchmark
from benchmarks.agent_supervisor.container_coding import full_supervisor_harbor_agent as adapter
from benchmarks.agent_supervisor.container_coding import terminal_source_view_policy as policy
from benchmarks.agent_supervisor.container_coding.benchmark_provider_profile import CLI_VERSION
from benchmarks.agent_supervisor.container_coding.test_terminal_deployment import _supervisor_manifest
from benchmarks.agent_supervisor.container_coding.test_terminal_semantic_metadata_policy import authored_task
from ipfs_accelerate_py.agent_supervisor.runtime.source_view_policy import SourceViewPolicy


PROFILE = "python-comments@1"


def recipe():
    return SourceViewPolicy.from_mapping(profile=PROFILE,
        protected_lines={b"bottle.py".hex(): [2, 4], b"types.pyi".hex(): [2]}).to_dict()


def authored_manifest():
    return _supervisor_manifest({"archive_sha256": "a" * 64,
        "codex_version": CLI_VERSION, "files": [], "learned_requirements": []})


def declaration(config):
    return controls.build_controls(config, task_input_sha256={"instruction.md": "a" * 64},
        task=benchmark.TASK, model=benchmark.MODEL, reasoning_effort="high", cli_version=CLI_VERSION)


@pytest.fixture
def prepared_case(tmp_path, monkeypatch):
    dataset, task = authored_task(tmp_path)
    archive = tmp_path / "archive"
    archive.mkdir()
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz"):
        pass
    raw = stream.getvalue()
    (archive / "runtime.tar.gz").write_bytes(raw)
    manifest = authored_manifest()
    manifest["archive_sha256"] = hashlib.sha256(raw).hexdigest()
    (archive / "manifest.json").write_text(json.dumps(manifest))
    calls = []
    monkeypatch.setattr(benchmark.subprocess, "run", lambda argv, **kwargs:
        calls.append(argv) or SimpleNamespace(returncode=0, stdout="Authored dry run", stderr=""))
    output = tmp_path / "output"
    selected = recipe()
    prepared = benchmark.prepare(dataset=dataset, output=output, archive=archive, arm="full",
        source_view_profile=PROFILE, source_view_policy=selected)
    return SimpleNamespace(dataset=dataset, task=task, archive=archive, output=output,
        selected=selected, prepared=prepared, calls=calls)


def write(path, value):
    path.write_text(json.dumps(value, sort_keys=True))


@pytest.mark.parametrize("arm", ["full", "no-index"])
def test_absent_recipe_preserves_legacy_configuration_and_observations(tmp_path, arm):
    args = (tmp_path, tmp_path / "output", tmp_path / "archive", arm)
    assert benchmark.config_for(*args) == benchmark.config_for(*args, source_view_policy=None)
    assert "source_view_policy" not in benchmark.config_for(*args)["agents"][0]["kwargs"]
    assert policy.source_view_selection("disabled", None) == {}
    assert policy.source_view_observation("disabled", None) == {}


def test_configuration_detaches_recipe_and_control_digest_binds_each_protected_line(tmp_path):
    selected = recipe()
    args = (tmp_path, tmp_path / "output", tmp_path / "archive", "full")
    config = benchmark.config_for(*args, source_view_profile=PROFILE, source_view_policy=selected)
    frozen = deepcopy(config)
    selected["protected_lines"][b"bottle.py".hex()].append(5)
    assert config == frozen
    other = deepcopy(config)
    other["agents"][0]["kwargs"]["source_view_policy"]["protected_lines"][b"bottle.py".hex()].append(5)
    assert declaration(config)["configuration_sha256"] != declaration(other)["configuration_sha256"]
    observed = controls.observe_controls({"comparison_controls": declaration(config)}, other,
        current_task_hashes={"instruction.md": "a" * 64})
    assert observed["configuration_unchanged"] is False
    assert "protected_lines" not in json.dumps(declaration(config))


@pytest.mark.parametrize("change", ["disabled", "profile", "no-index", "path", "unknown", "bool_line"])
def test_invalid_or_conflicting_recipe_refuses_before_preparation_io(tmp_path, monkeypatch, change):
    selected, profile, arm = recipe(), PROFILE, "full"
    if change == "disabled": profile = "disabled"
    elif change == "profile": profile = "identity@1"
    elif change == "no-index": arm = "no-index"
    elif change == "path": selected = tmp_path / "recipe.json"
    elif change == "unknown": selected["owner_path"] = "/private/index"
    else: selected["protected_lines"][b"bottle.py".hex()] = [True]
    monkeypatch.setattr(benchmark.subprocess, "run", lambda *args, **kwargs: pytest.fail("invalid policy spawned process"))
    for function in (benchmark.config_for, benchmark.prepare):
        arguments = dict(dataset=tmp_path / "missing", output=tmp_path / "output",
            archive=tmp_path / "missing", arm=arm, source_view_profile=profile, source_view_policy=selected)
        with pytest.raises(ValueError):
            function(**arguments)


@pytest.mark.parametrize("change", ["profile_missing", "profile_conflict", "arm", "malformed"])
def test_controls_allow_only_valid_shared_recipe_kwargs(tmp_path, change):
    config = benchmark.config_for(tmp_path, tmp_path / "output", tmp_path / "archive", "full",
        source_view_profile=PROFILE, source_view_policy=recipe())
    kwargs = config["agents"][0]["kwargs"]
    if change == "profile_missing": kwargs.pop("source_view_profile")
    elif change == "profile_conflict": kwargs["source_view_profile"] = "python-whitespace@1"
    elif change == "arm": kwargs["arm"] = "no-index"
    else: kwargs["source_view_policy"]["protected_lines"] = []
    with pytest.raises(ValueError): declaration(config)


def test_prepare_freezes_recipe_and_both_hashes_without_mutating_caller(prepared_case):
    case = prepared_case
    frozen = recipe()
    case.selected["protected_lines"][b"bottle.py".hex()].append(9)
    configured = json.loads((case.output / "config.json").read_text())["agents"][0]["kwargs"]
    assert configured["source_view_policy"] == case.prepared["source_view_policy"] == frozen
    expected = policy.source_view_observation(PROFILE, frozen)
    assert all(case.prepared[key] == value for key, value in expected.items())
    assert len(case.calls) == 1 and "--dry-run" in case.calls[0]
    assert case.prepared["provider_calls"] == 0
    collected = benchmark.collect(case.output)
    assert collected["source_view_policy_config_unchanged"] is True
    assert collected["source_view_policy"] == frozen
    assert collected["source_view_policy_binding_sha256"] == expected["source_view_policy_binding_sha256"]
    assert collected["source_view_token_savings_claimed"] is False


@pytest.mark.parametrize("change", ["drop_prepared", "drop_config", "edit_prepared", "edit_config",
    "recipe_digest", "binding_digest", "missing_binding", "malformed_prepared", "malformed_config",
    "null_prepared", "null_config", "config_sha", "rehashed_config", "drop_both_rehashed"])
def test_execute_and_collection_refuse_recipe_or_binding_drift_before_process(
        prepared_case, monkeypatch, change):
    case = prepared_case
    prepared = json.loads((case.output / "preparation.json").read_text())
    config = json.loads((case.output / "config.json").read_text())
    kwargs = config["agents"][0]["kwargs"]
    if change in {"drop_prepared", "drop_both_rehashed"}:
        for key in ("source_view_policy", "source_view_policy_sha256", "source_view_policy_binding_sha256"):
            prepared.pop(key)
        if change == "drop_both_rehashed": kwargs.pop("source_view_policy")
    elif change == "drop_config": kwargs.pop("source_view_policy")
    elif change == "edit_prepared": prepared["source_view_policy"]["protected_lines"][b"bottle.py".hex()].append(6)
    elif change in {"edit_config", "rehashed_config"}:
        kwargs["source_view_policy"]["protected_lines"][b"bottle.py".hex()].append(6)
    elif change == "recipe_digest": prepared["source_view_policy_sha256"] = "0" * 64
    elif change == "binding_digest": prepared["source_view_policy_binding_sha256"] = "0" * 64
    elif change == "missing_binding": prepared.pop("source_view_policy_binding_sha256")
    elif change == "malformed_prepared": prepared["source_view_policy"]["protected_lines"] = []
    elif change == "malformed_config": kwargs["source_view_policy"]["protected_lines"] = []
    elif change == "null_prepared": prepared["source_view_policy"] = None
    elif change == "null_config": kwargs["source_view_policy"] = None
    else: prepared["config_sha256"] = "0" * 64
    # Preserve original bytes unless the particular scenario edits the config.
    if change in {"drop_config", "edit_config", "rehashed_config", "malformed_config", "null_config", "drop_both_rehashed"}:
        write(case.output / "config.json", config)
    if change in {"rehashed_config", "drop_both_rehashed"}:
        prepared["config_sha256"] = benchmark._hash(case.output / "config.json")
    write(case.output / "preparation.json", prepared)
    monkeypatch.setattr(benchmark.subprocess, "run", lambda *args, **kwargs: pytest.fail("drift reached execution"))
    changed_file = change in {"drop_config", "edit_config", "malformed_config", "null_config", "config_sha"}
    expected_refusal = ("prepared task binding differs from the frozen configuration" if changed_file
                        else "prepared source view policy changed")
    with pytest.raises(ValueError, match="^" + expected_refusal + "$"):
        benchmark.execute(case.output)
    if changed_file or change in {"rehashed_config", "drop_both_rehashed"}:
        # Outer task/provider custody refuses an edited config before policy
        # observation. Do not weaken that boundary to manufacture a receipt.
        expected_collection_refusal = (expected_refusal if changed_file else
            "frozen provider profile differs from prepared configuration")
        with pytest.raises(ValueError, match="^" + expected_collection_refusal + "$"):
            benchmark.collect(case.output)
        assert not (case.output / "receipt.json").exists()
    else:
        result = benchmark.collect(case.output)
        assert result["source_view_policy_config_unchanged"] is False
        assert result["complete_single_trial_receipt"] is False
    assert not (case.output / "invocation.json").exists()


def test_unchanged_preparation_can_execute_through_authored_process_seam(prepared_case):
    case = prepared_case
    result = benchmark.execute(case.output)
    assert len(case.calls) == 2
    assert "--dry-run" in case.calls[0] and "--dry-run" not in case.calls[1]
    assert result["harbor_returncode"] == 0
    assert result["source_view_policy_config_unchanged"] is True
    assert result["source_view_policy"] == recipe()
    assert (case.output / "invocation.json").is_file()


@pytest.mark.parametrize("operation", ["execute", "collect"])
def test_cli_policy_selection_is_preparation_only_before_opening_file(tmp_path, monkeypatch, operation):
    monkeypatch.setattr(sys, "argv", ["benchmark", operation, "--output", str(tmp_path / "output"),
        "--source-view-policy", str(tmp_path / "must-not-read.json")])
    monkeypatch.setattr(benchmark, "_load_source_view_policy", lambda *args: pytest.fail("late policy file read"))
    with pytest.raises(SystemExit) as error: benchmark.main()
    assert error.value.code == 2


def test_cli_preparation_reads_exact_recipe_once_and_passes_detached_object(tmp_path, monkeypatch, capsys):
    path = tmp_path / "policy.json"
    path.write_text(SourceViewPolicy.from_dict(recipe()).to_json())
    calls = []
    monkeypatch.setattr(benchmark, "prepare", lambda **kwargs: calls.append(kwargs) or {"prepared": True})
    monkeypatch.setattr(sys, "argv", ["benchmark", "prepare", "--output", str(tmp_path / "output"),
        "--dataset", str(tmp_path / "dataset"), "--archive", str(tmp_path / "archive"),
        "--source-view-profile", PROFILE, "--source-view-policy", str(path)])
    benchmark.main()
    assert calls[0]["source_view_policy"] == recipe()
    assert type(calls[0]["source_view_policy"]) is dict
    assert calls[0]["source_view_profile"] == PROFILE
    assert json.loads(capsys.readouterr().out) == {"prepared": True}


@pytest.mark.parametrize("bad", ["duplicate", "oversized", "non_utf8", "foreign_field", "symlink"])
def test_cli_loader_refuses_unbounded_or_noncanonical_recipe_files(tmp_path, bad):
    path = tmp_path / "policy.json"
    raw = SourceViewPolicy.from_dict(recipe()).to_json().encode()
    if bad == "duplicate": raw = raw[:-1] + b',"profile":"python-comments@1"}'
    elif bad == "oversized": raw = b" " * (benchmark.MAX_SOURCE_VIEW_POLICY_BYTES + 1)
    elif bad == "non_utf8": raw = b"\xff"
    elif bad == "foreign_field": raw = json.dumps(recipe() | {"owner_path": "/private/cache"}).encode()
    path.write_bytes(raw)
    if bad == "symlink":
        alias = tmp_path / "alias.json"
        alias.symlink_to(path)
        path = alias
    with pytest.raises((ValueError, UnicodeError)):
        benchmark._load_source_view_policy(path)


@pytest.fixture
def environment(monkeypatch):
    monkeypatch.setattr(adapter, "capture_public_inputs", AsyncMock(return_value={"status": "captured"}))
    monkeypatch.setattr(adapter, "export_public_outputs", AsyncMock(return_value={"status": "captured"}))
    calls = []

    class Environment:
        async def upload_file(self, source, target): pass
        async def exec(self, **kwargs):
            calls.append(shlex.split(kwargs["command"]))
            return SimpleNamespace(return_code=0, stdout="", stderr="")
        async def download_file(self, source, target):
            Path(target).write_text(json.dumps({"task_completed": False, "seconds": 1,
                "phases": {}, "provider_invocations": []} if source.endswith("-result.json") else {}))

    return Environment(), calls


def test_agent_forwards_one_canonical_recipe_argument_and_records_its_bindings(tmp_path, environment):
    from harbor.models.agent.context import AgentContext

    (tmp_path / "manifest.json").write_text(json.dumps(authored_manifest()))
    selected = recipe()
    agent = adapter.FullSupervisorAgent(logs_dir=tmp_path / "logs", model_name=benchmark.MODEL,
        runtime_archive=str(tmp_path), arm="full", source_view_profile=PROFILE, source_view_policy=selected)
    selected["protected_lines"][b"bottle.py".hex()].append(6)
    context = AgentContext()
    env, calls = environment
    asyncio.run(agent.run("Authored instruction", env, context))
    assert len(calls) == 1 and calls[0].count("--source-view-policy-json") == 1
    raw = calls[0][calls[0].index("--source-view-policy-json") + 1]
    assert raw == SourceViewPolicy.from_dict(recipe()).to_json()
    assert context.metadata["source_view_policy"] == recipe()
    expected = policy.source_view_observation(PROFILE, recipe())
    assert all(context.metadata[key] == value for key, value in expected.items())


@pytest.mark.parametrize("change", ["lines", "malformed", "drop", "profile", "binding"])
def test_agent_setup_and_run_refuse_changed_policy_before_environment_actions(tmp_path, monkeypatch, change):
    (tmp_path / "manifest.json").write_text(json.dumps(authored_manifest()))
    agent = adapter.FullSupervisorAgent(logs_dir=tmp_path / "logs", model_name=benchmark.MODEL,
        runtime_archive=str(tmp_path), arm="full", source_view_profile=PROFILE, source_view_policy=recipe())
    if change == "lines": agent.source_view_policy["protected_lines"][b"bottle.py".hex()].append(6)
    elif change == "malformed": agent.source_view_policy["protected_lines"] = []
    elif change == "drop": agent.source_view_policy = None
    elif change == "profile": agent.source_view_profile = "identity@1"
    else:
        from ipfs_accelerate_py.agent_supervisor.runtime import source_view_policy as runtime_policy
        monkeypatch.setattr(runtime_policy, "current_codec_binding", lambda: {"changed": True})
    env = SimpleNamespace(upload_file=AsyncMock(), exec=AsyncMock(), download_file=AsyncMock())
    for operation in (agent.setup(env), agent.run("Authored instruction", env, SimpleNamespace())):
        with pytest.raises(ValueError): asyncio.run(operation)
    env.upload_file.assert_not_awaited()
    env.exec.assert_not_awaited()
    env.download_file.assert_not_awaited()
