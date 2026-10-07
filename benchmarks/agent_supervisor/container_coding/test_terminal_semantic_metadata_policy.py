"""Explicit metadata coding treatment is frozen without provider/Docker calls."""
import asyncio
import hashlib
import json
from pathlib import Path
import shlex
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchmarks.agent_supervisor.container_coding import full_supervisor_benchmark as benchmark
from benchmarks.agent_supervisor.container_coding import terminal_semantic_metadata_policy as policy
from benchmarks.agent_supervisor.container_coding.benchmark_controls import build_controls
from benchmarks.agent_supervisor.container_coding.benchmark_provider_profile import CLI_VERSION, GROK_PROFILE
from benchmarks.agent_supervisor.container_coding.test_terminal_deployment import _supervisor_manifest

MODE = policy.COMMON_BINDINGS_METADATA_VIEW
TRANSPORT = "supervisor-semantic-router-input@1"
DICTIONARY = "supervisor-semantic-router-input@2"


def inventory():
    root = Path(policy.__file__).resolve().parents[3]
    return {"files": [{"path": "source/" + name,
                       "sha256": hashlib.sha256((root / name).read_bytes()).hexdigest(),
                       "bytes": (root / name).stat().st_size} for name in policy.SOURCE_PATHS]}


def authored_task(tmp_path):
    dataset = tmp_path / "dataset"
    task = dataset / benchmark.TASK
    for directory in (task, task / "environment", task / "tests"):
        directory.mkdir(parents=True, exist_ok=True)
    for name, text in {"instruction.md": "Authored public fixture", "task.toml": 'version = "1.0"\n',
        "environment/Dockerfile": "FROM scratch\n", "tests/test.sh": "exit 0\n"}.items():
        (task / name).write_text(text)
    return dataset, task


@pytest.mark.parametrize("arm,transport", [("full", TRANSPORT), ("full", DICTIONARY), ("no-index", TRANSPORT)])
def test_legacy_omits_selection_and_preserves_configuration_shape(tmp_path, arm, transport):
    args = (tmp_path, tmp_path / "out", tmp_path / "archive", arm)
    original = benchmark.config_for(*args, semantic_transport_schema=transport)
    explicit = benchmark.config_for(*args, semantic_transport_schema=transport, semantic_metadata_view="legacy")
    assert json.dumps(original, sort_keys=True) == json.dumps(explicit, sort_keys=True)
    assert "semantic_metadata_view" not in original["agents"][0]["kwargs"]
    assert policy.semantic_metadata_selection("legacy") == {}
    assert policy.semantic_metadata_observation("legacy") == {}
    policy.require_semantic_metadata_archive(True, "legacy")


def test_opt_in_configuration_and_full_controls_are_bound(tmp_path):
    args = (tmp_path, tmp_path / "out", tmp_path / "archive", "full")
    legacy = benchmark.config_for(*args)
    selected = benchmark.config_for(*args, semantic_metadata_view=MODE)
    assert selected["agents"][0]["kwargs"] == {**legacy["agents"][0]["kwargs"], "semantic_metadata_view": MODE}
    assert "intent_requirement_contract" not in selected["agents"][0]["kwargs"]
    def controls(config):
        return build_controls(config, task_input_sha256={"instruction.md": "a" * 64}, task=benchmark.TASK,
            model=benchmark.MODEL, reasoning_effort="high", cli_version=CLI_VERSION)
    assert controls(legacy)["configuration_sha256"] != controls(selected)["configuration_sha256"]
    assert policy.semantic_metadata_observation(MODE) == dict(semantic_metadata_view=MODE,
        semantic_metadata_token_savings_claimed=False, semantic_metadata_authority=False)


@pytest.mark.parametrize("value", [None, True, {}, "common-bindings@2", "controller-dictionary@1"])
def test_unknown_mode_rejected_before_preparation_file_reads(tmp_path, value):
    with pytest.raises(ValueError, match="supported semantic metadata view"):
        benchmark.prepare(dataset=tmp_path / "missing", output=tmp_path / "out",
            archive=tmp_path / "missing", arm="full", semantic_metadata_view=value)


@pytest.mark.parametrize("options", [{"arm": "no-index"}, {"arm": "full", "semantic_transport_schema": DICTIONARY}])
def test_unqualified_arm_or_dictionary_rejected_before_reads(tmp_path, options):
    with pytest.raises(ValueError, match="metadata view requires"):
        benchmark.prepare(dataset=tmp_path / "missing", output=tmp_path / "out",
            archive=tmp_path / "missing", semantic_metadata_view=MODE, **options)


@pytest.mark.parametrize("provider", ["doctor_candidate", "provider_free", False])
def test_provider_free_selection_is_never_qualified(provider):
    with pytest.raises(ValueError, match="signed model-router"):
        policy.validate_semantic_metadata_view(MODE, arm="full", provider=provider)


def test_explicit_grok_model_route_preserves_existing_profile(tmp_path):
    result = benchmark.config_for(tmp_path, tmp_path / "out", tmp_path / "archive", "full",
                                  provider_profile=GROK_PROFILE, semantic_metadata_view=MODE)
    assert result["agents"][0]["kwargs"]["provider_profile"] == GROK_PROFILE
    assert result["agents"][0]["kwargs"]["semantic_metadata_view"] == MODE


@pytest.mark.parametrize("relative", policy.SOURCE_PATHS)
def test_each_opt_in_helper_requires_its_exact_current_archive_body(relative):
    manifest = inventory()
    row = next(item for item in manifest["files"] if item["path"] == "source/" + relative)
    row["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="newly qualified runtime archive"):
        policy.require_semantic_metadata_archive(manifest, MODE)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "bool_bytes", "wrong_bytes"])
def test_missing_duplicate_or_boolean_archive_metadata_rejected(mutation):
    manifest = inventory()
    row = next(item for item in manifest["files"] if item["path"].endswith("semantic_metadata_view.py"))
    if mutation == "missing": manifest["files"].remove(row)
    elif mutation == "duplicate": manifest["files"].append(dict(row))
    elif mutation == "bool_bytes": row["bytes"] = True
    elif mutation == "wrong_bytes": row["bytes"] += 1
    with pytest.raises(ValueError, match="newly qualified runtime archive"):
        policy.require_semantic_metadata_archive(manifest, MODE)


@pytest.mark.parametrize("manifest", [None, True, {"files": True}, {"files": {}}, {"files": "foreign"}])
def test_malformed_archive_inventory_rejected(manifest):
    with pytest.raises(ValueError, match="current runtime source inventory"):
        policy.require_semantic_metadata_archive(manifest, MODE)


def test_current_inventory_passes_without_claiming_actual_archive_execution():
    policy.require_semantic_metadata_archive(inventory(), MODE)


def test_prepare_freezes_selection_and_default_receipt_shape(tmp_path, monkeypatch):
    dataset, _ = authored_task(tmp_path)
    archive = tmp_path / "archive"
    archive.mkdir()
    raw = b"AUTHORED-no-executable-archive"
    (archive / "runtime.tar.gz").write_bytes(raw)
    (archive / "manifest.json").write_text(json.dumps(_supervisor_manifest({**inventory(),
        "archive_sha256": hashlib.sha256(raw).hexdigest(), "codex_version": CLI_VERSION})))
    commands = []
    def dry_run(argv, **kwargs):
        commands.append(argv)
        return SimpleNamespace(returncode=0, stdout="Authored dry-run fixture", stderr="")
    monkeypatch.setattr(benchmark.subprocess, "run", dry_run)
    selected = benchmark.prepare(dataset=dataset, output=tmp_path / "selected", archive=archive,
                                  arm="full", semantic_metadata_view=MODE)
    assert selected["semantic_metadata_view"] == MODE
    assert selected["semantic_metadata_token_savings_claimed"] is False
    assert selected["semantic_metadata_authority"] is False
    assert selected["provider_calls"] == 0
    assert json.loads((tmp_path / "selected/config.json").read_text())["agents"][0]["kwargs"]["semantic_metadata_view"] == MODE
    default = benchmark.prepare(dataset=dataset, output=tmp_path / "legacy", archive=archive, arm="full")
    assert not any(key.startswith("semantic_metadata") for key in default)
    assert all("--dry-run" in argv for argv in commands)
    assert len(commands) == 2


def test_adapter_rejects_unbound_archive_before_upload_setup_or_run(tmp_path):
    from benchmarks.agent_supervisor.container_coding.full_supervisor_harbor_agent import FullSupervisorAgent
    (tmp_path / "manifest.json").write_text(json.dumps(_supervisor_manifest({
        "archive_sha256": "a" * 64, "codex_version": CLI_VERSION, "files": []})))
    agent = FullSupervisorAgent(logs_dir=tmp_path / "logs", model_name=benchmark.MODEL,
        runtime_archive=str(tmp_path), arm="full", semantic_metadata_view=MODE)
    environment = SimpleNamespace(upload_file=AsyncMock(), exec=AsyncMock(), download_file=AsyncMock())
    for call in (agent.setup(environment), agent.run("Authored instruction", environment, SimpleNamespace())):
        with pytest.raises(ValueError, match="newly qualified runtime archive"):
            asyncio.run(call)
    environment.upload_file.assert_not_awaited()
    environment.exec.assert_not_awaited()
    environment.download_file.assert_not_awaited()


@pytest.mark.parametrize("mode", ["legacy", MODE])
def test_adapter_driver_pass_through_preserves_direct_planning(tmp_path, monkeypatch, mode):
    from harbor.models.agent.context import AgentContext
    from benchmarks.agent_supervisor.container_coding import full_supervisor_harbor_agent as adapter
    (tmp_path / "manifest.json").write_text(json.dumps(_supervisor_manifest({**inventory(),
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
            Path(target).write_text(json.dumps({"task_completed": False, "seconds": 1, "phases": {},
                "provider_invocations": []} if source.endswith("-result.json") else {}))
    agent = adapter.FullSupervisorAgent(logs_dir=tmp_path / "logs", model_name=benchmark.MODEL,
        runtime_archive=str(tmp_path), arm="full", semantic_metadata_view=mode)
    context = AgentContext()
    asyncio.run(agent.run("Authored instruction", Environment(), context))
    assert len(calls) == 1
    assert "--intent-requirement-contract" not in calls[0]
    assert context.metadata["planning_strategy"] == "direct"
    if mode == "legacy":
        assert "--semantic-metadata-view" not in calls[0]
        assert not any(key.startswith("semantic_metadata") for key in context.metadata)
    else:
        assert calls[0].count("--semantic-metadata-view") == 1
        assert calls[0][calls[0].index("--semantic-metadata-view") + 1] == MODE
        assert context.metadata["semantic_metadata_token_savings_claimed"] is False
        assert context.metadata["semantic_metadata_authority"] is False


def test_collect_and_execute_refuse_qualification_on_selection_drift(tmp_path, monkeypatch):
    dataset, task = authored_task(tmp_path)
    output = tmp_path / "out"
    output.mkdir()
    config = benchmark.config_for(dataset, output, tmp_path / "archive", "full", semantic_metadata_view=MODE)
    (output / "config.json").write_text(json.dumps(config))
    prepared = {"arm": "full", "dataset": str(dataset), "prepared": True,
        "config_sha256": benchmark._hash(output / "config.json"), "task_input_sha256": benchmark._task_hashes(task),
        "model": benchmark.MODEL, "reasoning_effort": "high", "cli_version": CLI_VERSION,
        "semantic_metadata_view": MODE}
    (output / "preparation.json").write_text(json.dumps(prepared))
    assert benchmark.collect(output)["semantic_metadata_config_unchanged"] is True
    prepared["semantic_metadata_view"] = "legacy"
    (output / "preparation.json").write_text(json.dumps(prepared))
    result = benchmark.collect(output)
    assert result["semantic_metadata_config_unchanged"] is False
    assert result["complete_single_trial_receipt"] is False
    calls = []
    monkeypatch.setattr(benchmark.subprocess, "run", lambda *args, **kwargs: calls.append(args))
    with pytest.raises(ValueError, match="prepared semantic metadata view selection changed"):
        benchmark.execute(output)
    assert calls == []
    assert not (output / "invocation.json").exists()


def test_cli_selection_is_allowed_only_during_prepare(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(benchmark, "prepare", lambda **kwargs: calls.append(kwargs) or {"prepared": True})
    monkeypatch.setattr(sys, "argv", ["benchmark", "prepare", "--dataset", str(tmp_path),
        "--archive", str(tmp_path / "archive"), "--output", str(tmp_path / "out"), "--semantic-metadata-view", MODE])
    benchmark.main()
    assert calls[0]["semantic_metadata_view"] == MODE
    assert calls[0]["intent_requirement_contract"] is None
    for operation in ("collect", "execute"):
        monkeypatch.setattr(sys, "argv", ["benchmark", operation, "--output", str(tmp_path), "--semantic-metadata-view", MODE])
        with pytest.raises(SystemExit) as error: benchmark.main()
        assert error.value.code == 2
