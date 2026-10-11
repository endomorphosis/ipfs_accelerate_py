"""Tiny authored transport assets exercise real archive and Harbor boundaries.

Only native checkpoint compatibility is a double; no neural inference, Docker
launch, training, model download or provider call occurs in this module.
"""
import asyncio
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import shlex
import tarfile
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_deployment as deployment
from benchmarks.agent_supervisor.container_coding import terminal_numerical_selection_deployment as transport
from benchmarks.agent_supervisor.container_coding import full_supervisor_harbor_agent as harbor
from benchmarks.agent_supervisor.container_coding.benchmark_resource_profile import SOURCE384_PROFILE
from benchmarks.agent_supervisor.container_coding.test_terminal_deployment import _inputs


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture
def selection(tmp_path, monkeypatch):
    from ipfs_datasets_py.logic.formalization.autoencoder import numerical_selection_384 as native
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer import autoencoder_embedding_runtime as embedding
    from ipfs_accelerate_py.agent_supervisor.runtime import numerical_selection_config as owner

    snapshot = tmp_path / "models--thenlper--gte-small" / "snapshots" / embedding.PINNED_REVISION
    snapshot.mkdir(parents=True)
    model = b"authored numerical transport model, not usable weights"
    (snapshot / "model.safetensors").write_bytes(model)
    (snapshot / "1_Pooling").mkdir()
    (snapshot / "1_Pooling/config.json").write_bytes(b"{}")
    pins = {"model.safetensors": (len(model), _sha(model)), "1_Pooling/config.json": (2, _sha(b"{}"))}
    monkeypatch.setattr(embedding, "_PINNED_ASSETS", pins)
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_bytes(b'{"authored_transport_fixture":true}')
    monkeypatch.setattr(owner, "PUBLISHED_CHECKPOINT_SHA256", _sha(checkpoint.read_bytes()))
    calls = []
    def compatibility(path, *, expected_sha256):
        assert _sha(Path(path).read_bytes()) == expected_sha256
        calls.append(str(path))
    monkeypatch.setattr(native, "load_selection_checkpoint", compatibility)
    config = owner.published_numerical_selection_config(checkpoint_path=checkpoint, embedding_snapshot=snapshot)
    return config, calls


def _build(tmp_path, selection):
    inputs = _inputs(tmp_path)
    for name, source in transport._producer_paths().items():
        prefix, relative = name.split("/", 1)
        target = inputs[prefix] / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    return deployment.build_runtime_archive(output=tmp_path / "bundle",
        numerical_selection_config=selection[0], **inputs)


def test_numerical_assets_are_offline_separate_and_only_consumed_by_full_arm(tmp_path, selection):
    manifest = _build(tmp_path, selection)
    binding = transport.verify_numerical_selection_archive(tmp_path / "bundle/runtime.tar.gz", manifest)
    assert selection[1]
    assert manifest["source384_requirements"] == []
    assert manifest["numerical_selection_requirements"] == list(deployment.LEARNED_REQUIREMENTS)
    assert manifest.get("source384") is None and manifest["task_inputs_in_archive"] is False
    assert binding["source_training_data_included"] is False
    assert binding["config"]["checkpoint_path"] == deployment.ROOT + "/" + transport.CHECKPOINT_PATH
    with tarfile.open(tmp_path / "bundle/runtime.tar.gz") as bundle:
        assert json.load(bundle.extractfile(transport.CONFIG_PATH)) == binding["config"]
        assert bundle.extractfile(transport.CHECKPOINT_PATH).read() == Path(selection[0]["checkpoint_path"]).read_bytes()
    calls = len(selection[1])
    args, observation = transport.numerical_selection_asset_arguments(manifest, "full", SOURCE384_PROFILE)
    assert args == ["--numerical-selection-config", deployment.ROOT + "/" + transport.CONFIG_PATH]
    assert observation["enabled"] is True and observation["proof_authority"] is False
    args, observation = transport.numerical_selection_asset_arguments(manifest, "no-index", SOURCE384_PROFILE)
    assert args == [] and observation["enabled"] is False
    assert observation["disabled_reason"] == "no_index_ablation"
    assert len(selection[1]) == calls  # Neither arm's metadata decision opens a model.
    with pytest.raises(ValueError, match="common resource"):
        transport.numerical_selection_asset_arguments(manifest, "full")


@pytest.mark.parametrize("mutation", ["missing_binding", "extra_binding", "wrong_config", "missing_model",
    "extra_model", "duplicate", "dependency", "setup_cache", "missing_producer", "stale_producer", "rehashed_foreign_checkpoint"])
def test_numerical_archive_requires_exact_selection_and_current_producer_inventory(tmp_path, selection, mutation):
    manifest = _build(tmp_path, selection)
    if mutation == "missing_binding": del manifest[transport.KEY]
    elif mutation == "extra_binding": manifest[transport.KEY]["proof_authority"] = True
    elif mutation == "wrong_config": manifest[transport.KEY]["config_path"] = "models/source384/config.json"
    elif mutation == "missing_model":
        manifest["files"] = [r for r in manifest["files"] if r["path"] != transport.CHECKPOINT_PATH]
    elif mutation == "extra_model":
        manifest["files"].append(dict(path=transport.PREFIX + "unselected.json", bytes=1, sha256="0" * 64))
    elif mutation == "duplicate": manifest["files"].append(deepcopy(manifest["files"][-1]))
    elif mutation == "dependency": manifest["numerical_selection_requirements"] = []
    elif mutation == "setup_cache": manifest["setup_cache"] = {}
    elif mutation == "rehashed_foreign_checkpoint":
        binding = manifest[transport.KEY]
        binding["config"]["checkpoint_sha256"] = "0" * 64
        config_bytes = transport._raw(binding["config"])
        binding["config_sha256"] = _sha(config_bytes)
        next(r for r in manifest["files"] if r["path"] == transport.CONFIG_PATH).update(
            sha256=binding["config_sha256"], bytes=len(config_bytes))
        next(r for r in manifest["files"] if r["path"] == transport.CHECKPOINT_PATH)["sha256"] = "0" * 64
    else:
        name = "datasets/ipfs_datasets_py/logic/software_contracts/codebase_numerical_selection_384_worker.py"
        if mutation == "missing_producer": manifest["files"] = [r for r in manifest["files"] if r["path"] != name]
        else: next(r for r in manifest["files"] if r["path"] == name)["sha256"] = "0" * 64
    with pytest.raises(ValueError):
        transport.numerical_selection_asset_arguments(manifest, "full", SOURCE384_PROFILE)


@pytest.mark.parametrize("member", [transport.CHECKPOINT_PATH,
    "datasets/ipfs_datasets_py/logic/software_contracts/codebase_numerical_selection_384_worker.py"])
def test_rehashed_archive_cannot_hide_changed_model_or_native_worker_bytes(tmp_path, selection, member):
    manifest = _build(tmp_path, selection)
    path = tmp_path / "bundle/runtime.tar.gz"
    with tarfile.open(path) as bundle:
        rows = [(m, bundle.extractfile(m).read()) for m in bundle]
    with tarfile.open(path, "w:gz") as bundle:
        for item, raw in rows:
            if item.name == member:
                raw = b"x" * len(raw)
            bundle.addfile(item, io.BytesIO(raw))
    manifest["archive_sha256"] = deployment._sha(path)
    (tmp_path / "bundle/manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="digest"):
        asyncio.run(deployment.deploy_supervisor(None, archive_dir=tmp_path / "bundle", output=tmp_path / "deployment"))


def test_absent_selection_refuses_hidden_uninventoried_numerical_tar_member(tmp_path):
    manifest = deployment.build_runtime_archive(output=tmp_path / "bundle", **_inputs(tmp_path))
    archive = tmp_path / "hidden.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        entry = tarfile.TarInfo(transport.CHECKPOINT_PATH)
        entry.size = 1
        bundle.addfile(entry, io.BytesIO(b"x"))
    with pytest.raises(ValueError, match="unexpected"):
        transport.verify_numerical_selection_archive(archive, manifest)


def test_setup_cache_rejected_before_numerical_asset_access(tmp_path, selection, monkeypatch):
    monkeypatch.setattr(transport, "numerical_selection_assets",
        lambda *args: pytest.fail("unsupported setup cache reached model asset loading"))
    with pytest.raises(ValueError, match="setup cache"):
        deployment.build_runtime_archive(output=tmp_path / "bundle", numerical_selection_config=selection[0],
            setup_cache_policy="authored-policy", **_inputs(tmp_path))
    assert not (tmp_path / "bundle").exists()


@pytest.mark.parametrize("arm", ["full", "no-index"])
def test_actual_harbor_runner_forwards_numerical_configuration_only_to_full(tmp_path, selection, monkeypatch, arm):
    from harbor.models.agent.context import AgentContext

    _build(tmp_path, selection)
    monkeypatch.setattr(harbor, "capture_public_inputs", AsyncMock(return_value={}))
    monkeypatch.setattr(harbor, "export_public_outputs", AsyncMock(return_value={}))
    commands = []
    class Environment:
        async def upload_file(self, *args): pass
        async def exec(self, **kwargs):
            commands.append(shlex.split(kwargs["command"]))
            return SimpleNamespace(stdout="", stderr="", return_code=0)
        async def download_file(self, *args): raise FileNotFoundError

    agent = harbor.FullSupervisorAgent(logs_dir=tmp_path / "logs", model_name="gpt-6.1-sol",
        runtime_archive=str(tmp_path / "bundle"), arm=arm, resource_profile=SOURCE384_PROFILE)
    context = AgentContext()
    count = len(selection[1])
    asyncio.run(agent.run("Public instruction.", Environment(), context))
    assert ("--numerical-selection-config" in commands[0]) is (arm == "full")
    assert context.metadata["numerical_selection_assets"]["enabled"] is (arm == "full")
    assert len(selection[1]) == count


def test_required_archive_inventory_covers_every_declared_consumer_producer():
    from ipfs_accelerate_py.agent_supervisor.runtime.numerical_selection_context import _producers

    # The explicit packaging list must cover all byte-level pins declared by
    # the owner and shared decoder, including their imported compatibility code.
    def digests(value):
        if isinstance(value, dict):
            return {h for item in value.values() for h in digests(item)}
        if isinstance(value, (list, tuple)):
            return {h for item in value for h in digests(item)}
        return {value} if type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value) else set()
    pinned = {_sha(path.read_bytes()) for path in transport._producer_paths().values()}
    assert digests(_producers()) <= pinned
