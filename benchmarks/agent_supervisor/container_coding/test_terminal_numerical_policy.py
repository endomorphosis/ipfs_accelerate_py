"""Frozen numerical selection and archive joins, without provider/Docker calls."""
import asyncio
import hashlib
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
from benchmarks.agent_supervisor.container_coding import terminal_numerical_policy as policy
from benchmarks.agent_supervisor.container_coding.benchmark_provider_profile import CLI_VERSION
from benchmarks.agent_supervisor.container_coding.test_terminal_deployment import _supervisor_manifest
from benchmarks.agent_supervisor.container_coding.test_terminal_semantic_metadata_policy import authored_task


def inventory(profile=policy.PROFILE):
    return {"files": [{"path": name, "bytes": path.stat().st_size,
                       "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                      for name, path in policy._producer_paths(profile).items()]}


def test_function_unit_extractor_producers_are_bound_by_archive_and_consumer():
    from ipfs_datasets_py.logic.formalization.autoencoder import source_function_units
    from ipfs_accelerate_py.agent_supervisor.runtime import numerical_formalization_context

    expected = source_function_units.pins()
    consumer_names = {
        "source_function_units.py": "function_units",
        "security_formula_corpus.py": "unit_span_validation",
        "security_formalization_evaluation.py": "unit_span_extraction",
        "source_screening.py": "unit_source_screening",
    }
    assert set(expected) == set(consumer_names)
    actual = numerical_formalization_context._producers()
    archive = policy._producer_paths()
    for basename, digest in expected.items():
        paths = [path for name, path in archive.items()
                 if name.startswith("datasets/") and path.name == basename]
        assert len(paths) == 1
        assert hashlib.sha256(paths[0].read_bytes()).hexdigest() == digest
        assert actual[consumer_names[basename]] == digest


def seal(config):
    return controls.build_controls(config, task_input_sha256={"instruction.md": "a" * 64},
        task=benchmark.TASK, model=benchmark.MODEL, reasoning_effort="high", cli_version=CLI_VERSION)


def observed(config):
    return dict(status="observed", configuration_unchanged=True, declared=seal(config))


@pytest.mark.parametrize("arm", ["full", "no-index"])
def test_default_preserves_historical_configuration_and_controls(tmp_path, arm):
    args = (tmp_path, tmp_path / "out", tmp_path / "archive", arm)
    legacy = benchmark.config_for(*args)
    assert benchmark.config_for(*args, numerical_formalization_profile=None) == legacy
    assert "numerical_formalization_profile" not in legacy["agents"][0]["kwargs"]
    assert seal(legacy)["schema"] == controls.SCHEMA
    assert "experimental_controls_sha256" not in seal(legacy)
    assert policy.numerical_observation(None, arm) == {}
    policy.require_numerical_archive(None, None)


@pytest.mark.parametrize("profile", policy.PROFILES)
def test_new_controls_compare_selected_profile_across_arms(tmp_path, profile):
    args = (tmp_path, tmp_path / "out", tmp_path / "archive")
    full = benchmark.config_for(*args, "full", numerical_formalization_profile=profile)
    no_index = benchmark.config_for(*args, "no-index", numerical_formalization_profile=profile)
    legacy = benchmark.config_for(*args, "no-index")
    for config in (full, no_index):
        assert seal(config)["schema"] == controls.NUMERICAL_SCHEMA
        assert controls.validate_controls(seal(config))
    assert controls.compare_controls(observed(full), observed(no_index))["matches"] is True
    mismatch = controls.compare_controls(observed(full), observed(legacy))
    assert mismatch["matches"] is False
    assert mismatch["differences"] == ["experimental_controls_sha256.numerical_formalization_profile"]
    assert policy.numerical_observation(profile, "no-index")["numerical_formalization_disabled_reason"] == "no_index_ablation"
    assert not policy.numerical_observation(profile, "full")["numerical_formalization_proof_authority"]


@pytest.mark.parametrize("mutation", ["missing", "unknown", "bool", "digest", "legacy_schema"])
def test_new_control_record_rejects_edited_or_incomplete_experiment(tmp_path, mutation):
    result = seal(benchmark.config_for(tmp_path, tmp_path / "out", tmp_path / "archive", "full",
                                      numerical_formalization_profile=policy.PROFILE))
    if mutation == "missing": del result["experimental_controls_sha256"]
    elif mutation == "unknown": result["experimental_controls_sha256"]["unknown"] = "0" * 64
    elif mutation == "bool": result["experimental_controls_sha256"]["numerical_formalization_profile"] = True
    elif mutation == "legacy_schema": result["schema"] = controls.SCHEMA
    else: result["experimental_controls_sha256"]["numerical_formalization_profile"] = "0" * 64
    assert not controls.validate_controls(result)


@pytest.mark.parametrize("value", [True, {}, 1, "", "python-numerical-structure@2"])
def test_unknown_selection_fails_before_input_reads(tmp_path, value):
    with pytest.raises(ValueError, match="supported numerical formalization profile"):
        benchmark.prepare(dataset=tmp_path / "missing", output=tmp_path / "out",
            archive=tmp_path / "missing", arm="full", numerical_formalization_profile=value)


@pytest.mark.parametrize("name", [*("source/" + n for n in policy.SOURCE_PATHS),
                                  *("datasets/" + n for n in policy.DATASET_PATHS)])
def test_every_current_numerical_producer_is_joined_to_archive(name):
    manifest = inventory()
    next(row for row in manifest["files"] if row["path"] == name)["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="newly qualified runtime archive"):
        policy.require_numerical_archive(manifest, policy.PROFILE)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "bool_bytes", "wrong_bytes"])
def test_archive_inventory_cannot_omit_duplicate_or_coerce_producer_metadata(mutation):
    manifest = inventory(); row = manifest["files"][0]
    if mutation == "missing": manifest["files"].remove(row)
    elif mutation == "duplicate": manifest["files"].append(dict(row))
    elif mutation == "bool_bytes": row["bytes"] = True
    else: row["bytes"] += 1
    with pytest.raises(ValueError, match="newly qualified runtime archive"):
        policy.require_numerical_archive(manifest, policy.PROFILE)


@pytest.mark.parametrize("manifest", [None, True, {}, {"files": {}}, {"files": "other"}])
def test_malformed_inventory_fails_closed(manifest):
    with pytest.raises(ValueError, match="current runtime source inventory"):
        policy.require_numerical_archive(manifest, policy.PROFILE)


def test_exact_current_inventory_passes_as_compatibility_only():
    policy.require_numerical_archive(inventory(), policy.PROFILE)


def test_guarded_archive_adds_only_selected_modules_and_binds_reused_bases():
    from ipfs_accelerate_py.agent_supervisor.runtime import numerical_formalization_context as numerical

    legacy = policy._producer_paths()
    guarded = policy._producer_paths(policy.GUARDED_PROFILE)
    assert set(guarded) - set(legacy) == {"datasets/" + p for p in policy.GUARDED_DATASET_PATHS}
    assert {key: guarded[key] for key in legacy} == legacy
    assert policy.PROFILES == numerical.PROFILES
    pins = numerical._producers(policy.GUARDED_PROFILE)
    for key, basename in (("grammar", "numerical_source_v2.py"), ("projections", "numerical_logic_v2.py"),
                          ("base_grammar", "numerical_source.py"), ("base_projections", "numerical_logic.py")):
        paths = [p for p in guarded.values() if p.name == basename]
        assert len(paths) == 1 and pins[key] == hashlib.sha256(paths[0].read_bytes()).hexdigest()
    policy.require_numerical_archive(inventory(policy.GUARDED_PROFILE), policy.GUARDED_PROFILE)


@pytest.mark.parametrize("name", policy.GUARDED_DATASET_PATHS)
@pytest.mark.parametrize("mutation", ["missing", "changed"])
def test_guarded_profile_requires_exact_new_producers_but_v1_does_not(name, mutation):
    manifest = inventory(policy.GUARDED_PROFILE)
    row = next(row for row in manifest["files"] if row["path"] == "datasets/" + name)
    if mutation == "missing":
        manifest["files"].remove(row)
    else:
        row["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="newly qualified runtime archive"):
        policy.require_numerical_archive(manifest, policy.GUARDED_PROFILE)
    policy.require_numerical_archive(manifest, policy.PROFILE)


def test_v1_and_v2_controls_are_distinct_even_with_identical_task_inputs(tmp_path):
    configurations = [benchmark.config_for(tmp_path, tmp_path / "out", tmp_path / "archive", "full",
        numerical_formalization_profile=profile) for profile in policy.PROFILES]
    comparison = controls.compare_controls(*(observed(config) for config in configurations))
    assert comparison["matches"] is False
    assert comparison["differences"] == ["experimental_controls_sha256.numerical_formalization_profile"]


@pytest.mark.parametrize("arm", ["full", "no-index"])
@pytest.mark.parametrize("profile", policy.PROFILES)
def test_prepare_freezes_same_selection_and_opposite_consumption(tmp_path, monkeypatch, arm, profile):
    dataset, _ = authored_task(tmp_path); archive = tmp_path / "archive"; archive.mkdir()
    # The unselected learned-model verifier still checks the archive's absence
    # of undeclared model members. Give it an actual empty archive to inspect.
    with tarfile.open(archive / "runtime.tar.gz", "w:gz"):
        pass
    raw = (archive / "runtime.tar.gz").read_bytes()
    (archive / "manifest.json").write_text(json.dumps(_supervisor_manifest({**inventory(profile),
        "archive_sha256": hashlib.sha256(raw).hexdigest(), "codex_version": CLI_VERSION})))
    calls = []
    def dry(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=0, stdout="Authored dry-run", stderr="")
    monkeypatch.setattr(benchmark.subprocess, "run", dry)
    result = benchmark.prepare(dataset=dataset, output=tmp_path / "out", archive=archive, arm=arm,
                               numerical_formalization_profile=profile)
    assert result["numerical_formalization_profile"] == profile
    assert result["numerical_formalization_enabled"] is (arm == "full")
    assert result["comparison_controls"]["schema"] == controls.NUMERICAL_SCHEMA
    assert result["provider_calls"] == 0 and all("--dry-run" in c for c in calls)


@pytest.mark.parametrize("arm", ["full", "no-index"])
def test_adapter_rejects_old_bundle_before_setup_upload_or_run(tmp_path, arm):
    from benchmarks.agent_supervisor.container_coding.full_supervisor_harbor_agent import FullSupervisorAgent
    (tmp_path / "manifest.json").write_text(json.dumps(_supervisor_manifest({
        "archive_sha256": "a" * 64, "codex_version": CLI_VERSION, "files": []})))
    agent = FullSupervisorAgent(logs_dir=tmp_path / "logs", model_name=benchmark.MODEL,
        runtime_archive=str(tmp_path), arm=arm, numerical_formalization_profile=policy.PROFILE)
    env = SimpleNamespace(upload_file=AsyncMock(), exec=AsyncMock(), download_file=AsyncMock())
    for call in (agent.setup(env), agent.run("Authored instruction", env, SimpleNamespace())):
        with pytest.raises(ValueError, match="newly qualified runtime archive"):
            asyncio.run(call)
    env.upload_file.assert_not_awaited(); env.exec.assert_not_awaited(); env.download_file.assert_not_awaited()


@pytest.mark.parametrize("arm", ["full", "no-index"])
@pytest.mark.parametrize("selected", [None, *policy.PROFILES])
def test_harbor_passes_frozen_profile_and_records_no_index_disable(tmp_path, monkeypatch, arm, selected):
    from harbor.models.agent.context import AgentContext
    from benchmarks.agent_supervisor.container_coding import full_supervisor_harbor_agent as adapter
    (tmp_path / "manifest.json").write_text(json.dumps(_supervisor_manifest({**inventory(selected or policy.PROFILE),
        "archive_sha256": "a" * 64, "codex_version": CLI_VERSION, "learned_requirements": []})))
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
    agent = adapter.FullSupervisorAgent(logs_dir=tmp_path / "logs", model_name=benchmark.MODEL,
        runtime_archive=str(tmp_path), arm=arm, numerical_formalization_profile=selected)
    context = AgentContext()
    asyncio.run(agent.run("Authored instruction", Environment(), context))
    assert len(calls) == 1 and context.metadata["planning_strategy"] == "direct"
    flag = "--numerical-formalization-profile"
    if selected is None:
        assert flag not in calls[0]
        assert "numerical_formalization_profile" not in context.metadata
    else:
        assert calls[0].count(flag) == 1 and calls[0][calls[0].index(flag) + 1] == selected
        assert context.metadata["numerical_formalization_enabled"] is (arm == "full")


@pytest.mark.parametrize("profile", policy.PROFILES)
def test_collect_and_execute_refuse_numerical_selection_drift(tmp_path, monkeypatch, profile):
    dataset, task = authored_task(tmp_path); output = tmp_path / "out"; output.mkdir()
    config = benchmark.config_for(dataset, output, tmp_path / "archive", "full",
                                   numerical_formalization_profile=profile)
    (output / "config.json").write_text(json.dumps(config))
    prepared = {"arm": "full", "dataset": str(dataset), "prepared": True,
        "config_sha256": benchmark._hash(output / "config.json"), "task_input_sha256": benchmark._task_hashes(task),
        "model": benchmark.MODEL, "reasoning_effort": "high", "cli_version": CLI_VERSION,
        "numerical_formalization_profile": profile}
    (output / "preparation.json").write_text(json.dumps(prepared))
    assert benchmark.collect(output)["numerical_formalization_config_unchanged"] is True
    del prepared["numerical_formalization_profile"]
    (output / "preparation.json").write_text(json.dumps(prepared))
    collected = benchmark.collect(output)
    assert not collected["numerical_formalization_config_unchanged"]
    assert not collected["complete_single_trial_receipt"]
    calls = []; monkeypatch.setattr(benchmark.subprocess, "run", lambda *a, **k: calls.append(a))
    with pytest.raises(ValueError, match="prepared numerical formalization selection changed"):
        benchmark.execute(output)
    assert calls == [] and not (output / "invocation.json").exists()


@pytest.mark.parametrize("profile", policy.PROFILES)
def test_cli_selection_is_frozen_by_prepare(tmp_path, monkeypatch, profile):
    calls = []; monkeypatch.setattr(benchmark, "prepare", lambda **kw: calls.append(kw) or {})
    monkeypatch.setattr(sys, "argv", ["benchmark", "prepare", "--dataset", str(tmp_path),
        "--archive", str(tmp_path), "--output", str(tmp_path / "out"),
        "--numerical-formalization-profile", profile])
    benchmark.main()
    assert calls[0]["numerical_formalization_profile"] == profile
    for operation in ("collect", "execute"):
        monkeypatch.setattr(sys, "argv", ["benchmark", operation, "--output", str(tmp_path),
            "--numerical-formalization-profile", profile])
        with pytest.raises(SystemExit) as error: benchmark.main()
        assert error.value.code == 2
