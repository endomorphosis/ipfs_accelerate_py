"""The selected archive, prepared job and offline retrieval loader use one pin."""
import asyncio
import json
import shlex
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchmarks.agent_supervisor.container_coding import full_supervisor_benchmark as benchmark
from benchmarks.agent_supervisor.container_coding import full_supervisor_harbor_agent as adapter
from benchmarks.agent_supervisor.container_coding import learned_vector_preflight
from benchmarks.agent_supervisor.container_coding.terminal_deployment import ROOT, build_runtime_archive
from benchmarks.agent_supervisor.container_coding.terminal_retrieval_selection import (
    selected_retrieval_revision, require_retrieval_revision,
)
from benchmarks.agent_supervisor.container_coding.test_terminal_deployment import _supervisor_inputs

GTE_REVISION = "17e1f347d17fe144873b1201da91788898c639cd"
LEGACY_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


@pytest.fixture
def packaged(tmp_path):
    snapshot = tmp_path / GTE_REVISION
    snapshot.mkdir()
    # Packaging fixture only: neural inference is not claimed by these bytes.
    (snapshot / "config.json").write_text('{"fixture":true}')
    archive = tmp_path / "bundle"
    manifest = build_runtime_archive(output=archive, model_snapshot=snapshot, **_supervisor_inputs(tmp_path))
    dataset = tmp_path / "dataset"
    task = dataset / benchmark.TASK
    task.mkdir(parents=True)
    (task / "instruction.md").write_text("Repair the authored program.")
    (task / "task.toml").write_text('version = "1.0"')
    (task / "environment").mkdir()
    (task / "tests").mkdir()
    return archive, manifest, snapshot, dataset


@pytest.mark.parametrize("arm", ["full", "no-index"])
def test_actual_preparation_derives_revision_from_selected_archive(packaged, tmp_path, monkeypatch, arm):
    archive, manifest, snapshot, dataset = packaged
    calls = []
    monkeypatch.setattr(benchmark.subprocess, "run", lambda *args, **kwargs:
        calls.append(args) or SimpleNamespace(returncode=0, stdout="", stderr=""))
    output = tmp_path / "trial"
    prepared = benchmark.prepare(dataset=dataset, output=output, archive=archive, arm=arm)
    config = json.loads((output / "config.json").read_text())
    actual = config["agents"][0]["kwargs"]["model_revision"]
    assert actual == (GTE_REVISION if arm == "full" else "")
    assert actual != LEGACY_REVISION
    assert require_retrieval_revision(manifest, arm, actual) == actual
    assert prepared["manifest_sha256"] == benchmark._hash(archive / "manifest.json")
    assert prepared["config_sha256"] == benchmark._hash(output / "config.json")
    assert len(calls) == 1 and "--dry-run" in calls[0][0]
    assert prepared["provider_calls"] == 0


@pytest.mark.parametrize("revision", [None, "", "main", "A" * 40, "1" * 39, True, []])
def test_malformed_selected_revision_refused_before_preflight(packaged, tmp_path, monkeypatch, revision):
    archive, manifest, _snapshot, dataset = packaged
    manifest["model_snapshot_revision"] = revision
    (archive / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(benchmark.subprocess, "run", lambda *a, **k: pytest.fail("unexpected Harbor call"))
    with pytest.raises(ValueError, match="revision"):
        benchmark.prepare(dataset=dataset, output=tmp_path / "trial", archive=archive, arm="full")
    assert not (tmp_path / "trial").exists()


@pytest.mark.parametrize("requirements", [None, "requirements", [None], [""]])
def test_malformed_learned_selection_is_not_silently_disabled(requirements):
    with pytest.raises(ValueError, match="selection"):
        selected_retrieval_revision({"learned_requirements": requirements, "model_snapshot_revision": GTE_REVISION}, "full")


def test_lexical_and_no_index_legacy_config_defaults_remain_inert():
    legacy = {"learned_requirements": []}
    assert require_retrieval_revision(legacy, "full", LEGACY_REVISION) == ""
    assert require_retrieval_revision({"learned_requirements": ["pinned"], "model_snapshot_revision": GTE_REVISION}, "no-index", LEGACY_REVISION) == ""
    assert benchmark.config_for(Path('/dataset'), Path('/output'), Path('/archive'), 'full')['agents'][0]['kwargs']['model_revision'] == ""
    with pytest.raises(ValueError, match="lacks selected"):
        selected_retrieval_revision({"learned_requirements": [], "model_snapshot_revision": GTE_REVISION}, "full")


class Environment:
    def __init__(self):
        self.commands, self.uploads = [], []

    async def upload_file(self, source, target):
        self.uploads.append(target)

    async def exec(self, **kwargs):
        self.commands.append(shlex.split(kwargs["command"]))
        return SimpleNamespace(return_code=0, stdout="", stderr="")

    async def download_file(self, *args):
        raise FileNotFoundError


def _agent(packaged, tmp_path, monkeypatch, revision, arm="full"):
    monkeypatch.setattr(adapter, "capture_public_inputs", AsyncMock(return_value={}))
    monkeypatch.setattr(adapter, "export_public_outputs", AsyncMock(return_value={}))
    return adapter.FullSupervisorAgent(logs_dir=tmp_path / "logs", model_name=benchmark.MODEL,
        runtime_archive=str(packaged[0]), arm=arm, model_revision=revision)


@pytest.mark.parametrize("stage", ["setup", "run"])
@pytest.mark.parametrize("revision", [LEGACY_REVISION, "", None, "main"])
def test_adapter_rejects_stale_pin_before_deployment_or_upload(packaged, tmp_path, monkeypatch, stage, revision):
    from harbor.models.agent.context import AgentContext
    agent = _agent(packaged, tmp_path, monkeypatch, revision)
    environment = Environment()
    monkeypatch.setattr(adapter, "deploy_supervisor", AsyncMock(side_effect=AssertionError("unexpected deployment")))
    with pytest.raises(ValueError, match="revision"):
        asyncio.run(agent.setup(environment) if stage == "setup" else agent.run("Authored task", environment, AgentContext()))
    assert environment.commands == environment.uploads == []


@pytest.mark.parametrize("arm", ["full", "no-index"])
def test_container_argv_matches_prepared_revision_without_model_call(packaged, tmp_path, monkeypatch, arm):
    from harbor.models.agent.context import AgentContext
    selected = selected_retrieval_revision(packaged[1], arm)
    config = benchmark.config_for(packaged[3], tmp_path / "trial", packaged[0], arm, model_revision=selected)
    agent = _agent(packaged, tmp_path, monkeypatch, config['agents'][0]['kwargs']['model_revision'], arm)
    environment = Environment()
    asyncio.run(agent.run("Authored task", environment, AgentContext()))
    argv = environment.commands[0]
    if arm == "full":
        assert argv.count('--model-revision') == argv.count('--model-snapshot') == 1
        assert argv[argv.index('--model-revision') + 1] == GTE_REVISION
        assert argv[argv.index('--model-snapshot') + 1] == ROOT + '/models/embedding'
    else:
        assert '--model-revision' not in argv and '--model-snapshot' not in argv


def test_existing_loader_rejects_stale_revision_before_model_load(packaged, tmp_path, monkeypatch):
    archive, manifest, snapshot, dataset = packaged
    monkeypatch.setattr(learned_vector_preflight, '_LocalRouterModel', lambda *args:
                        pytest.fail('unexpected model load'))
    with pytest.raises(ValueError, match='exact local snapshot directory'):
        learned_vector_preflight.qualify(tmp_path, tmp_path / 'retrieval', ['authored.py'],
            'authored query', snapshot, LEGACY_REVISION)


def test_execute_revalidates_active_revision_before_harbor(packaged, tmp_path, monkeypatch):
    archive, _manifest, _snapshot, dataset = packaged
    monkeypatch.setattr(benchmark.subprocess, 'run', lambda *args, **kwargs:
        SimpleNamespace(returncode=0, stdout='', stderr=''))
    output = tmp_path / 'trial'
    prepared = benchmark.prepare(dataset=dataset, output=output, archive=archive, arm='full')
    config_path = output / 'config.json'
    config = json.loads(config_path.read_text())
    config['agents'][0]['kwargs']['model_revision'] = LEGACY_REVISION
    config_path.write_text(json.dumps(config))
    # Even a newly recorded outer config hash cannot make the stale model pin
    # agree with the independently frozen archive manifest.
    prepared['config_sha256'] = benchmark._hash(config_path)
    (output / 'preparation.json').write_text(json.dumps(prepared))
    monkeypatch.setattr(benchmark.subprocess, 'run', lambda *a, **k: pytest.fail('unexpected Harbor execution'))
    with pytest.raises(ValueError, match='differs from runtime archive'):
        benchmark.execute(output)
    assert not (output / 'invocation.json').exists()
