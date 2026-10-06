"""Extended coding budgets require the actual packaged installer capability."""
import asyncio
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchmarks.agent_supervisor.container_coding import full_supervisor_benchmark as benchmark
from benchmarks.agent_supervisor.container_coding import full_supervisor_harbor_agent as adapter
from benchmarks.agent_supervisor.container_coding import terminal_worker_capability as capability
from benchmarks.agent_supervisor.container_coding.benchmark_resource_profile import (
    CODING600_SOURCE384_PROFILE, PLANNER180_20GIB_SOURCE384_PROFILE,
)
from benchmarks.agent_supervisor.container_coding.terminal_deployment import build_runtime_archive
from benchmarks.agent_supervisor.container_coding.test_terminal_deployment import _inputs

GTE_REVISION = '17e1f347d17fe144873b1201da91788898c639cd'


@pytest.fixture
def packaged(tmp_path):
    inputs = _inputs(tmp_path)
    root = Path(__file__).resolve().parents[3]
    for name in capability.SOURCE_PATHS:
        target = inputs['source'] / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((root / name).read_bytes())
    archive = tmp_path / 'bundle'
    snapshot = tmp_path / GTE_REVISION
    snapshot.mkdir()
    # Packaging selection fixture; actual GTE inference has a separate probe.
    (snapshot / 'config.json').write_text('{"fixture":true}')
    manifest = build_runtime_archive(output=archive, model_snapshot=snapshot, **inputs)
    dataset = tmp_path / 'dataset'
    task = dataset / benchmark.TASK
    task.mkdir(parents=True)
    (task / 'instruction.md').write_text('Repair the authored program.')
    (task / 'task.toml').write_text('version = "1.0"')
    (task / 'environment').mkdir()
    (task / 'tests').mkdir()
    return archive, manifest, dataset


def test_builder_binds_exact_installer_template_and_shared_router_cap(packaged):
    from benchmarks.agent_supervisor.container_coding.container_worker_deployment import WORKER_ENTRY
    from ipfs_accelerate_py.agent_supervisor.runtime.router_implementation_runner import MAX_IMPLEMENTATION_TIMEOUT_SECONDS
    value = capability.require_worker_capability(packaged[1], CODING600_SOURCE384_PROFILE)
    assert value['max_timeout_seconds'] == MAX_IMPLEMENTATION_TIMEOUT_SECONDS == 600
    assert value['worker_entry_sha256'] == hashlib.sha256(WORKER_ENTRY.encode()).hexdigest()
    assert len(value['source_sha256']) == 2


def test_partial_and_other_source_archives_cannot_advertise_current_capability(tmp_path, packaged):
    assert capability.worker_capability_for_inventory([]) is None
    inventory = copy.deepcopy(packaged[1]['files'])
    next(row for row in inventory if row['path'].endswith('container_worker_deployment.py'))['sha256'] = 'a' * 64
    assert capability.worker_capability_for_inventory(inventory) is None


@pytest.mark.parametrize('profile', [None, PLANNER180_20GIB_SOURCE384_PROFILE])
def test_legacy_archives_remain_usable_at_original_budgets(profile):
    assert capability.require_worker_capability({}, profile) is None


def _mutate(manifest, kind):
    value = copy.deepcopy(manifest)
    if kind == 'legacy':
        del value[capability.KEY]
    elif kind == 'old-cap':
        value[capability.KEY]['max_timeout_seconds'] = 300
    elif kind == 'false-cap':
        value[capability.KEY]['max_timeout_seconds'] = True
    elif kind == 'template':
        value[capability.KEY]['worker_entry_sha256'] = 'b' * 64
    elif kind == 'inventory':
        next(row for row in value['files'] if row['path'].endswith('container_worker_deployment.py'))['sha256'] = 'c' * 64
    elif kind == 'duplicate':
        value['files'].append(next(row for row in value['files'] if row['path'].endswith('router_implementation_runner.py')))
    else:
        value[capability.KEY]['authority'] = True
    return value


@pytest.mark.parametrize('kind', ['legacy', 'old-cap', 'false-cap', 'template', 'inventory', 'duplicate', 'extra'])
def test_prepare_rejects_stale_or_mismatched_archive_before_harbor(packaged, tmp_path, monkeypatch, kind):
    archive, manifest, dataset = packaged
    (archive / 'manifest.json').write_text(json.dumps(_mutate(manifest, kind)))
    monkeypatch.setattr(benchmark.subprocess, 'run', lambda *a, **k: pytest.fail('unexpected Harbor call'))
    with pytest.raises(ValueError, match='worker archive capability'):
        benchmark.prepare(dataset=dataset, output=tmp_path / 'trial', archive=archive,
                          arm='full', resource_profile=CODING600_SOURCE384_PROFILE)
    assert not (tmp_path / 'trial').exists()


@pytest.mark.parametrize('arm', ['full', 'no-index'])
def test_actual_prepare_accepts_matching_capability_without_downgrade(packaged, tmp_path, monkeypatch, arm):
    archive, manifest, dataset = packaged
    calls = []
    monkeypatch.setattr(benchmark.subprocess, 'run', lambda *args, **kwargs:
        calls.append(args) or SimpleNamespace(returncode=0, stdout='', stderr=''))
    output = tmp_path / 'trial'
    result = benchmark.prepare(dataset=dataset, output=output, archive=archive,
        arm=arm, resource_profile=CODING600_SOURCE384_PROFILE)
    config = json.loads((output / 'config.json').read_text())
    assert config['agents'][0]['kwargs']['resource_profile'] == CODING600_SOURCE384_PROFILE
    assert config['agents'][0]['kwargs']['model_revision'] == (GTE_REVISION if arm == 'full' else '')
    assert result['prepared'] and len(calls) == 1 and '--dry-run' in calls[0][0]


def test_actual_legacy300_preparation_still_accepts_manifest_without_capability(packaged, tmp_path, monkeypatch):
    archive, manifest, dataset = packaged
    del manifest[capability.KEY]
    (archive / 'manifest.json').write_text(json.dumps(manifest))
    monkeypatch.setattr(benchmark.subprocess, 'run', lambda *a, **k:
        SimpleNamespace(returncode=0, stdout='', stderr=''))
    result = benchmark.prepare(dataset=dataset, output=tmp_path / 'trial', archive=archive,
        arm='full', resource_profile=PLANNER180_20GIB_SOURCE384_PROFILE)
    assert result['prepared']


class Environment:
    def __init__(self):
        self.commands, self.uploads = [], []

    async def upload_file(self, source, target):
        self.uploads.append(target)

    async def exec(self, **kwargs):
        self.commands.append(kwargs['command'])
        return SimpleNamespace(return_code=0, stdout='', stderr='')

    async def download_file(self, *args):
        raise FileNotFoundError


def _agent(packaged, tmp_path, monkeypatch):
    monkeypatch.setattr(adapter, 'capture_public_inputs', AsyncMock(return_value={}))
    monkeypatch.setattr(adapter, 'export_public_outputs', AsyncMock(return_value={}))
    return adapter.FullSupervisorAgent(logs_dir=tmp_path / 'logs', model_name=benchmark.MODEL,
        runtime_archive=str(packaged[0]), arm='full', resource_profile=CODING600_SOURCE384_PROFILE,
        model_revision=GTE_REVISION)


@pytest.mark.parametrize('stage', ['setup', 'run'])
@pytest.mark.parametrize('kind', ['legacy', 'old-cap', 'inventory'])
def test_adapter_rejects_before_upload_deploy_or_planning(packaged, tmp_path, monkeypatch, stage, kind):
    from harbor.models.agent.context import AgentContext
    archive, manifest, _ = packaged
    (archive / 'manifest.json').write_text(json.dumps(_mutate(manifest, kind)))
    agent = _agent(packaged, tmp_path, monkeypatch)
    environment = Environment()
    monkeypatch.setattr(adapter, 'deploy_supervisor', AsyncMock(side_effect=AssertionError('unexpected deployment')))
    with pytest.raises(ValueError, match='worker archive capability'):
        asyncio.run(agent.setup(environment) if stage == 'setup' else agent.run('Authored task', environment, AgentContext()))
    assert not environment.commands and not environment.uploads


def test_valid_capability_reaches_setup_and_retains_coding600_profile(packaged, tmp_path, monkeypatch):
    deploy, boundary = AsyncMock(), AsyncMock()
    monkeypatch.setattr(adapter, 'deploy_supervisor', deploy)
    monkeypatch.setattr(adapter, 'deploy_worker_boundary', boundary)
    agent = _agent(packaged, tmp_path, monkeypatch)
    asyncio.run(agent.setup(Environment()))
    assert deploy.await_count == boundary.await_count == 1
    assert agent.resource_profile == CODING600_SOURCE384_PROFILE


def test_valid_capability_and_gte_revision_reach_driver_argv(packaged, tmp_path, monkeypatch):
    from harbor.models.agent.context import AgentContext
    environment = Environment()
    agent = _agent(packaged, tmp_path, monkeypatch)
    asyncio.run(agent.run('Authored task', environment, AgentContext()))
    assert len(environment.commands) == 1
    command = environment.commands[0]
    assert '--resource-profile ' + CODING600_SOURCE384_PROFILE in command
    assert '--model-revision ' + GTE_REVISION in command


def test_execute_rechecks_even_when_outer_manifest_hash_is_updated(packaged, tmp_path, monkeypatch):
    archive, manifest, dataset = packaged
    monkeypatch.setattr(benchmark.subprocess, 'run', lambda *a, **k: SimpleNamespace(returncode=0, stdout='', stderr=''))
    output = tmp_path / 'trial'
    prepared = benchmark.prepare(dataset=dataset, output=output, archive=archive,
        arm='full', resource_profile=CODING600_SOURCE384_PROFILE)
    (archive / 'manifest.json').write_text(json.dumps(_mutate(manifest, 'legacy')))
    prepared['manifest_sha256'] = benchmark._hash(archive / 'manifest.json')
    (output / 'preparation.json').write_text(json.dumps(prepared))
    monkeypatch.setattr(benchmark.subprocess, 'run', lambda *a, **k: pytest.fail('unexpected Harbor execution'))
    with pytest.raises(ValueError, match='worker archive capability'):
        benchmark.execute(output)
    assert not (output / 'invocation.json').exists()
