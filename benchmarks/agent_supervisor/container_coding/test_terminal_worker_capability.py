"""All supervisor profiles require the packaged worker child custody capability."""
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
    CODING600_SOURCE384_PROFILE, PROFILES,
)
from benchmarks.agent_supervisor.container_coding.terminal_deployment import build_runtime_archive
from benchmarks.agent_supervisor.container_coding.test_terminal_deployment import _supervisor_inputs

GTE_REVISION = '17e1f347d17fe144873b1201da91788898c639cd'


@pytest.fixture
def packaged(tmp_path):
    inputs = _supervisor_inputs(tmp_path)
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
    assert value['schema'] == 'terminal-router-worker-capability@2'
    assert value['worker_child_custody_schema'] == 'terminal-worker-child-custody@1'
    assert len(value['source_sha256']) == 3
    assert any(name.endswith('/native_cli_subreaper.py') for name in value['source_sha256'])


@pytest.mark.parametrize('kind', ['installer', 'missing-custody', 'stale-custody'])
def test_partial_and_other_source_archives_cannot_advertise_current_capability(packaged, kind):
    assert capability.worker_capability_for_inventory([]) is None
    inventory = copy.deepcopy(packaged[1]['files'])
    if kind == 'missing-custody':
        inventory = [row for row in inventory if not row['path'].endswith('native_cli_subreaper.py')]
    else:
        name = 'native_cli_subreaper.py' if kind == 'stale-custody' else 'container_worker_deployment.py'
        next(row for row in inventory if row['path'].endswith(name))['sha256'] = 'a' * 64
    assert capability.worker_capability_for_inventory(inventory) is None


@pytest.mark.parametrize('profile', [None, *PROFILES])
def test_all_profiles_require_rebuilt_archive_with_child_custody(profile, packaged):
    with pytest.raises(ValueError, match='rebuild runtime archive'):
        capability.require_worker_capability({}, profile)
    assert capability.require_worker_capability(packaged[1], profile) == packaged[1][capability.KEY]


def _mutate(manifest, kind):
    value = copy.deepcopy(manifest)
    if kind == 'legacy':
        del value[capability.KEY]
    elif kind == 'v1':
        value[capability.KEY]['schema'] = 'terminal-router-worker-capability@1'
        value[capability.KEY].pop('worker_child_custody_schema')
        value[capability.KEY]['source_sha256'].pop('source/' + capability.SOURCE_PATHS[-1])
    elif kind == 'missing-custody-feature':
        value[capability.KEY].pop('worker_child_custody_schema')
    elif kind == 'custody-feature':
        value[capability.KEY]['worker_child_custody_schema'] = 'terminal-worker-child-custody@2'
    elif kind == 'missing-custody-inventory':
        value['files'] = [row for row in value['files'] if not row['path'].endswith('native_cli_subreaper.py')]
    elif kind == 'custody-inventory':
        next(row for row in value['files'] if row['path'].endswith('native_cli_subreaper.py'))['sha256'] = 'd' * 64
    elif kind == 'rehashed-custody':
        name = 'source/' + capability.SOURCE_PATHS[-1]
        value[capability.KEY]['source_sha256'][name] = 'd' * 64
        next(row for row in value['files'] if row['path'] == name)['sha256'] = 'd' * 64
    elif kind == 'duplicate-custody':
        value['files'].append(next(row for row in value['files'] if row['path'].endswith('native_cli_subreaper.py')))
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


@pytest.mark.parametrize('kind', ['legacy', 'v1', 'old-cap', 'false-cap', 'template', 'inventory', 'duplicate', 'extra',
    'missing-custody-feature', 'custody-feature', 'missing-custody-inventory', 'custody-inventory', 'rehashed-custody', 'duplicate-custody'])
@pytest.mark.parametrize('profile', [None, CODING600_SOURCE384_PROFILE])
def test_prepare_rejects_stale_or_mismatched_archive_before_harbor(packaged, tmp_path, monkeypatch, kind, profile):
    archive, manifest, dataset = packaged
    (archive / 'manifest.json').write_text(json.dumps(_mutate(manifest, kind)))
    monkeypatch.setattr(benchmark.subprocess, 'run', lambda *a, **k: pytest.fail('unexpected Harbor call'))
    with pytest.raises(ValueError, match='worker archive capability'):
        benchmark.prepare(dataset=dataset, output=tmp_path / 'trial', archive=archive,
                          arm='full', resource_profile=profile)
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


@pytest.mark.parametrize('profile', [None, *PROFILES])
def test_rebuilt_archive_prepares_all_profiles_with_original_budget(packaged, tmp_path, monkeypatch, profile):
    archive, manifest, dataset = packaged
    monkeypatch.setattr(benchmark.subprocess, 'run', lambda *a, **k:
        SimpleNamespace(returncode=0, stdout='', stderr=''))
    result = benchmark.prepare(dataset=dataset, output=tmp_path / 'trial', archive=archive,
        arm='no-index', resource_profile=profile)
    config = json.loads((tmp_path / 'trial' / 'config.json').read_text())
    assert result['prepared']
    assert config['agents'][0]['kwargs'].get('resource_profile') == profile
    from benchmarks.agent_supervisor.container_coding.benchmark_resource_profile import coding_timeout_seconds
    assert coding_timeout_seconds(profile) == (600 if profile == CODING600_SOURCE384_PROFILE else 300)


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


def _agent(packaged, tmp_path, monkeypatch, profile=CODING600_SOURCE384_PROFILE):
    monkeypatch.setattr(adapter, 'capture_public_inputs', AsyncMock(return_value={}))
    monkeypatch.setattr(adapter, 'export_public_outputs', AsyncMock(return_value={}))
    return adapter.FullSupervisorAgent(logs_dir=tmp_path / 'logs', model_name=benchmark.MODEL,
        runtime_archive=str(packaged[0]), arm='full', resource_profile=profile,
        model_revision=GTE_REVISION)


@pytest.mark.parametrize('stage', ['setup', 'run'])
@pytest.mark.parametrize('kind', ['legacy', 'v1', 'missing-custody-inventory', 'rehashed-custody'])
@pytest.mark.parametrize('profile', [None, CODING600_SOURCE384_PROFILE])
def test_adapter_rejects_before_upload_deploy_or_planning(packaged, tmp_path, monkeypatch, stage, kind, profile):
    from harbor.models.agent.context import AgentContext
    archive, manifest, _ = packaged
    (archive / 'manifest.json').write_text(json.dumps(_mutate(manifest, kind)))
    agent = _agent(packaged, tmp_path, monkeypatch, profile)
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


@pytest.mark.parametrize('profile', [None, CODING600_SOURCE384_PROFILE])
@pytest.mark.parametrize('kind', ['legacy', 'rehashed-custody'])
def test_execute_rechecks_even_when_outer_manifest_hash_is_updated(packaged, tmp_path, monkeypatch, profile, kind):
    archive, manifest, dataset = packaged
    monkeypatch.setattr(benchmark.subprocess, 'run', lambda *a, **k: SimpleNamespace(returncode=0, stdout='', stderr=''))
    output = tmp_path / 'trial'
    prepared = benchmark.prepare(dataset=dataset, output=output, archive=archive,
        arm='full', resource_profile=profile)
    (archive / 'manifest.json').write_text(json.dumps(_mutate(manifest, kind)))
    prepared['manifest_sha256'] = benchmark._hash(archive / 'manifest.json')
    (output / 'preparation.json').write_text(json.dumps(prepared))
    monkeypatch.setattr(benchmark.subprocess, 'run', lambda *a, **k: pytest.fail('unexpected Harbor execution'))
    with pytest.raises(ValueError, match='worker archive capability'):
        benchmark.execute(output)
    assert not (output / 'invocation.json').exists()
