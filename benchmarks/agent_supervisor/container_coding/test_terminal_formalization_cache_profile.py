"""Reserved formalization budgets reuse exact cache pins without new authority.

Native manifest/helper validation and resource validators run normally. The
transport returns authored public receipts; no container or model runs here.
"""
import asyncio
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_setup_cache_advice as cache
from benchmarks.agent_supervisor.container_coding import terminal_setup_cache_libraries as libraries
from benchmarks.agent_supervisor.container_coding import terminal_setup_cache_codex as native
from benchmarks.agent_supervisor.container_coding import benchmark_resource_profile as resources
from benchmarks.agent_supervisor.container_coding.full_supervisor_benchmark import config_for
from benchmarks.agent_supervisor.container_coding.test_benchmark_controls import declaration, observation
from benchmarks.agent_supervisor.container_coding.test_terminal_setup_cache_advice import (
    selected_cache, selected_cache_v2, _native_receipt,  # noqa: F401
)

PROFILE = resources.FORMALIZATION_SOURCE384_PROFILE


def configured(selection, profile=PROFILE):
    return config_for(Path('/dataset'), Path('/output'), Path('/archive'), 'full',
        resource_profile=profile, setup_cache_selection=selection)


def test_formalization_cache_uses_exact_shared_resources_and_independent_phase_budgets(selected_cache_v2, monkeypatch):
    from harbor.models.job.config import JobConfig
    root, manifest, selection, auth = selected_cache_v2
    assert PROFILE == 'source384-5cpu-20gib-formalization@1'
    assert PROFILE in cache.RESOURCE_PROFILES
    with monkeypatch.context() as guard:
        def forbidden(*args, **kwargs):
            pytest.fail('cache prerequisites must not read credentials')
        for method in ('open', 'read_bytes', 'read_text'):
            guard.setattr(Path, method, forbidden)
        cache.validate_setup_cache_prerequisites(selection, install_codex=True, auth_json=auth,
            arm='full', resource_profile=PROFILE)
    normalized = JobConfig.model_validate(configured(selection), extra='forbid').model_dump(mode='json')
    assert normalized['environment']['override_cpus'] == 5
    assert normalized['environment']['override_memory_mb'] == 20480
    assert normalized['environment']['cpu_enforcement_policy'] == normalized['environment']['memory_enforcement_policy'] == 'limit'
    assert normalized['agents'][0]['override_timeout_sec'] == normalized['agents'][0]['max_timeout_sec'] == 6480
    assert normalized['agents'][0]['kwargs']['setup_cache_selection'] == selection
    assert declaration(normalized)['schema'] == 'terminal-benchmark-declared-controls@1'
    assert observation(normalized)['status'] == 'observed'
    assert cache.validate_setup_cache_selection(root, selection) == (manifest, manifest['setup_cache'])
    # The existing formalization profile reserves later work after long prep.
    budget = resources.ReservedPhaseBudget(started=0., profile=PROFILE)
    budget.enter('preparation', now=0.)
    budget.enter('planning', now=3300.)
    budget.enter('preparation', now=3500.)
    budget.enter('native_setup', now=3550.)
    budget.enter('implementation', now=3600.)
    assert budget.remaining(now=3600.) == 1860
    assert resources.coding_timeout_seconds(PROFILE) == 1800
    assert manifest['setup_cache']['codex_sha256'] == cache.CODEX_PINS_V2
    assert len(manifest['setup_cache']['native_libraries']['rows']) == 131


@pytest.mark.parametrize('section,key,value', [
    ('environment', 'override_cpus', 4), ('environment', 'override_cpus', True),
    ('environment', 'override_memory_mb', 16384),
    ('environment', 'cpu_enforcement_policy', 'none'),
    ('environment', 'memory_enforcement_policy', 'none'),
    ('agent', 'override_timeout_sec', 960), ('agent', 'max_timeout_sec', 960),
])
def test_formalization_cache_controls_reject_tampered_limits(selected_cache_v2, section, key, value):
    config = configured(selected_cache_v2[2])
    (config['environment'] if section == 'environment' else config['agents'][0])[key] = value
    with pytest.raises(ValueError, match='resource limits|time limits'):
        declaration(config)


@pytest.mark.parametrize('profile', [resources.CODING600_SOURCE384_PROFILE, resources.PLANNER180_SOURCE384_PROFILE])
def test_other_valid_budget_profiles_cannot_declare_an_unsupported_cache_route(selected_cache_v2, profile):
    _, _, selection, auth = selected_cache_v2
    with pytest.raises(ValueError, match='supported resource profile'):
        declaration(configured(selection, profile))
    with pytest.raises(ValueError, match='common resources'):
        cache.validate_setup_cache_prerequisites(selection, install_codex=True, auth_json=auth,
            resource_profile=profile)


@pytest.mark.parametrize('mutation', ['helper_pin', 'codex_pin', 'library_pin'])
def test_formalization_cache_refuses_rehashed_mutated_asset_bindings_before_transport(selected_cache_v2, tmp_path, mutation):
    root, manifest, selection, _ = selected_cache_v2
    if mutation == 'helper_pin': manifest['setup_cache']['helpers']['terminal_setup_cache_advice.py'] = '0' * 64
    elif mutation == 'codex_pin': manifest['setup_cache']['codex_sha256']['codex'] = '0' * 64
    else: manifest['setup_cache']['native_libraries']['rows'][0]['sha256'] = '0' * 64
    raw = json.dumps(manifest).encode()
    (root/'manifest.json').write_bytes(raw)
    selection = {**selection, 'manifest_sha256': hashlib.sha256(raw).hexdigest()}
    env = SimpleNamespace(upload_file=AsyncMock(), exec=AsyncMock())
    with pytest.raises(ValueError, match='binding|pins'):
        asyncio.run(cache.apply_setup_cache_advice(env, archive_dir=root, expected=selection,
            boundary_output=tmp_path/'boundary', output=tmp_path/'unused', resource_profile=PROFILE))
    env.upload_file.assert_not_awaited(); env.exec.assert_not_awaited()
    assert not (tmp_path/'unused').exists()


@pytest.mark.parametrize('mutation', [None, 'cpu', 'memory', 'detected_cpu'])
def test_formalization_cache_transport_preserves_payload_pins_and_checks_exact_cgroup_receipt(
        selected_cache_v2, tmp_path, mutation):
    root, manifest, selection, _ = selected_cache_v2
    boundary = tmp_path/'boundary'; (boundary/'installation').mkdir(parents=True)
    raw = json.dumps(_native_receipt('0.160.0', native.PINS_V2)).encode()
    (boundary/'installation/native-codex-binary.log').write_bytes(raw)
    size = sum(row['expected_bytes'] for row in libraries.ROWS)
    observations = [
        dict(schema='manifest-cache-advice@1', manifest_sha256=selection['manifest_sha256'],
            archive_sha256=manifest['archive_sha256'], selected_files=len(manifest['files']),
            body_reads=0, metadata_unchanged=True, freed_bytes_claimed=False),
        dict(schema='pinned-native-codex-cache-advice@1', codex_version='0.160.0',
            post_boundary_receipt_sha256=hashlib.sha256(raw).hexdigest(), selected_files=4,
            hashed_files=4, body_reads_after_advice=0, body_read_bytes=100, selected_bytes=100,
            metadata_unchanged=True, freed_bytes_claimed=False),
        dict(schema='pinned-native-libraries-cache-advice@1', wheel_sha256=libraries.WHEEL_SHA256,
            torch_requirement=libraries.TORCH_REQUIREMENT, source_pins_sha256=libraries.SOURCE_PINS_SHA256,
            source_selection=libraries.SOURCE_SELECTION, installer_mode_policy=libraries.INSTALLER_MODE_POLICY,
            selected_files=131, hashed_files=131, selected_bytes=size, body_read_bytes=size,
            body_reads_after_advice=0, metadata_unchanged=True, freed_bytes_claimed=False,
            files=deepcopy(libraries.ROWS)),
        dict(schema='terminal-source384-cgroup-observation@1', cgroup_path='/authored/cgroup',
            cpu_max='500000 100000', memory_max=str(20480*1024*1024),
            detected_cpu_slots=5, detected_total_memory_mb=20480, available_memory_mb=16000),
    ]
    if mutation == 'cpu': observations[-1]['cpu_max'] = '400000 100000'
    elif mutation == 'memory': observations[-1]['memory_max'] = str(16384*1024*1024)
    elif mutation == 'detected_cpu': observations[-1]['detected_cpu_slots'] = 4
    env = SimpleNamespace(upload_file=AsyncMock(), exec=AsyncMock(side_effect=[
        SimpleNamespace(return_code=0, stdout=json.dumps(value), stderr='') for value in observations]))
    output = tmp_path/'advice'
    def apply():
        return asyncio.run(cache.apply_setup_cache_advice(env, archive_dir=root, expected=selection,
            boundary_output=boundary, output=output, resource_profile=PROFILE))
    if mutation is not None:
        with pytest.raises(ValueError, match='actual cgroup limits'):
            apply()
        report = json.loads((output/'setup-cache-advice.json').read_text())
        assert report['completed'] is False
    else:
        report = apply()
        assert report['completed'] is True
        assert report['resources'] == observations[-1]
        assert report['native_libraries']['files'] == libraries.ROWS
        estimate = json.loads((output/'admission-estimate.json').read_text())
        assert estimate['minimum_available_memory_mb'] == 10240
        assert estimate['exact_admission_decision'] is False and estimate['actual_native_admission_required'] is True
    assert env.exec.await_count == 4
    assert report['admission_authority'] is False and report['freed_bytes_claimed'] is False
    assert report['global_drop_caches'] is False and report['credential_contents_recorded'] is False
