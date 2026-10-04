"""A larger development run must remain bounded and comparable across arms."""
from copy import deepcopy
from pathlib import Path

import pytest
from harbor.models.job.config import JobConfig
from benchmarks.agent_supervisor.container_coding import benchmark_resource_profile as profile
from benchmarks.agent_supervisor.container_coding import full_supervisor_benchmark as supervisor
from benchmarks.agent_supervisor.container_coding import native_codex_baseline as baseline
from benchmarks.agent_supervisor.container_coding import terminal_source384_qualification as qualification
from benchmarks.agent_supervisor.container_coding.benchmark_controls import compare_controls
from benchmarks.agent_supervisor.container_coding.test_terminal_source384_transport import observation


@pytest.mark.parametrize('selected', profile.PROFILES)
def test_all_three_arms_share_harbor_limits_and_different_profiles_do_not_match(selected):
    def normalize(value):
        return JobConfig.model_validate(value, extra='forbid').model_dump(mode='json')
    native = normalize(baseline.config_for(Path('/dataset'), Path('/base'), resource_profile=selected))
    expected = profile.execution_budget(selected)
    assert native['agents'][0]['override_timeout_sec'] == expected['harbor_seconds']
    assert native['environment']['override_memory_mb'] == profile.resource_environment(selected)['override_memory_mb']
    for arm in ('full', 'no-index'):
        other = normalize(supervisor.config_for(Path('/dataset'), Path('/supervisor'), Path('/archive'), arm,
                                                resource_profile=selected))
        assert compare_controls(observation(native), observation(other))['matches']
    original = normalize(baseline.config_for(Path('/dataset'), Path('/base'),
                                             resource_profile=profile.SOURCE384_PROFILE))
    assert compare_controls(observation(native), observation(original))['matches'] is (selected == profile.SOURCE384_PROFILE)


def test_extended_budget_leaves_cleanup_and_transport_inside_harbor_deadline():
    budget = profile.execution_budget(profile.EXTENDED_SOURCE384_PROFILE)
    assert budget['driver_seconds'] - budget['cleanup_seconds'] == 840
    assert budget['source384_seconds'] == 180
    assert budget['native_start_seconds'] == 120
    assert budget['cleanup_seconds'] >= 2 * 20  # native STOP and accounting reserve
    assert budget['driver_seconds'] < budget['exec_seconds'] < budget['harbor_seconds']
    assert budget['qualification_seconds'] < budget['qualification_exec_seconds']
    assert profile.execution_budget()['driver_seconds'] == 285
    assert profile.execution_budget()['native_start_seconds'] == 20
    assert profile.execution_budget(profile.SOURCE384_PROFILE)['native_start_seconds'] == 20
    assert profile.admission_environment() == profile.admission_environment(profile.SOURCE384_PROFILE) == {}
    admission = profile.admission_environment(profile.EXTENDED_SOURCE384_PROFILE)
    assert admission['IPFS_DATASETS_PROOF_RESOURCE_PROFILE'] == 'local-benchmark@1'
    assert admission['IPFS_DATASETS_RESOURCE_SCHEDULER_PATH'].startswith('/opt/ipfs-supervisor/state/')


@pytest.mark.parametrize('seconds,expected', [(840, 120000), (120, 120000), (5, 5000), (2.5, 2500), (2, 2000)])
def test_extended_start_allowance_is_bounded_by_remaining_work(seconds, expected):
    assert profile.native_start_timeout_ms(profile.EXTENDED_SOURCE384_PROFILE,
        remaining_work_seconds=seconds) == expected


@pytest.mark.parametrize('seconds', [0, 1, 1.999])
def test_extended_start_refuses_insufficient_time_instead_of_spending_cleanup(seconds):
    with pytest.raises(TimeoutError, match='insufficient work budget'):
        profile.native_start_timeout_ms(profile.EXTENDED_SOURCE384_PROFILE, remaining_work_seconds=seconds)


@pytest.mark.parametrize('selected', [None, profile.SOURCE384_PROFILE])
def test_legacy_profiles_do_not_override_native_start(selected):
    assert profile.native_start_timeout_ms(selected, remaining_work_seconds=120) is None


@pytest.mark.parametrize('seconds', [True, -1, float('inf'), float('nan'), '120'])
def test_start_budget_requires_finite_exact_numeric_work_time(seconds):
    with pytest.raises(ValueError, match='remaining work'):
        profile.native_start_timeout_ms(profile.EXTENDED_SOURCE384_PROFILE, remaining_work_seconds=seconds)


def test_unknown_start_profile_is_rejected():
    with pytest.raises(ValueError, match='unknown'):
        profile.native_start_timeout_ms('unknown-profile', remaining_work_seconds=120)


@pytest.mark.parametrize('field', ['override_timeout_sec', 'max_timeout_sec'])
def test_profile_rejects_old_outer_timeout_after_extended_selection(field):
    config = supervisor.config_for(Path('/dataset'), Path('/output'), Path('/archive'), 'full',
                                   resource_profile=profile.EXTENDED_SOURCE384_PROFILE)
    config['agents'][0][field] = 300
    with pytest.raises(ValueError, match='time limits'):
        profile.validate_resource_profile(config, profile.EXTENDED_SOURCE384_PROFILE)


def test_actual_container_memory_must_match_selected_profile():
    observed = dict(schema='terminal-source384-cgroup-observation@1', cgroup_path='/sys/fs/cgroup',
                    cpu_max='500000 100000', memory_max=str(16384*1024*1024),
                    detected_cpu_slots=5, detected_total_memory_mb=16384, available_memory_mb=14000)
    assert qualification.validate_resource_observation(observed, profile.EXTENDED_SOURCE384_PROFILE) == observed
    with pytest.raises(ValueError, match='limits differ'):
        qualification.validate_resource_observation(observed, profile.SOURCE384_PROFILE)


@pytest.mark.parametrize('function', [profile.execution_budget, profile.admission_environment,
                                     profile.resource_environment])
def test_unknown_profile_is_never_silently_accepted(function):
    with pytest.raises(ValueError):
        function('unreviewed-profile')


@pytest.mark.parametrize('reported_seconds, matches', [(960, True), (300, False)])
def test_baseline_collector_uses_selected_extended_deadline(tmp_path, monkeypatch, reported_seconds, matches):
    import json
    config = JobConfig.model_validate(baseline.config_for(Path('/dataset'), tmp_path,
        resource_profile=profile.EXTENDED_SOURCE384_PROFILE), extra='forbid').model_dump(mode='json')
    config_path = tmp_path / 'config.json'
    config_path.write_text(json.dumps(config))
    hashes = {'instruction.md': 'a' * 64}
    from benchmarks.agent_supervisor.container_coding import benchmark_controls
    prepared = dict(dataset='/dataset', config_sha256=baseline._hash(config_path),
                    resource_profile=profile.EXTENDED_SOURCE384_PROFILE, task_input_sha256=hashes,
                    comparison_controls=benchmark_controls.build_controls(config, task_input_sha256=hashes,
                        task=baseline.TASK, model=baseline.MODEL, reasoning_effort=baseline.REASONING,
                        cli_version=baseline.CLI_VERSION))
    (tmp_path / 'preparation.json').write_text(json.dumps(prepared))
    job = tmp_path / 'jobs' / ('native-codex-' + baseline.TASK)
    trial = job / 'authored-trial'
    trial.mkdir(parents=True)
    agent = deepcopy(config['agents'][0])
    agent['override_timeout_sec'] = agent['max_timeout_sec'] = reported_seconds
    result = dict(task_name='terminal-bench/' + baseline.TASK, trial_name='authored-trial',
                  config=dict(task={'path': '/dataset/' + baseline.TASK}, agent=agent, verifier={'disable': False}))
    (trial / 'result.json').write_text(json.dumps(result))
    (job / 'result.json').write_text('{}')
    monkeypatch.setattr(baseline, '_task_hashes', lambda path: hashes)
    receipt = baseline.collect(tmp_path)
    assert receipt['trials'][0]['exact_trial_profile_matches'] is matches
    assert receipt['complete_single_trial_receipt'] is matches


@pytest.mark.parametrize('selected, total, expected', [(False, 120, 30), (True, 120, 90), (True, 12, 12)])
def test_source_currentness_wait_uses_profile_without_extending_operation(monkeypatch, selected, total, expected):
    from contextlib import contextmanager
    from ipfs_accelerate_py.agent_supervisor.runtime import source384_repository_context as context
    from ipfs_datasets_py.logic.software_contracts import codebase_resources
    if selected:
        monkeypatch.setenv('IPFS_DATASETS_PROOF_RESOURCE_PROFILE', 'local-benchmark@1')
    else:
        monkeypatch.delenv('IPFS_DATASETS_PROOF_RESOURCE_PROFILE', raising=False)
    waits = []
    @contextmanager
    def acquire(**kwargs):
        waits.append(kwargs['timeout_seconds'])
        yield object()
    monkeypatch.setattr(codebase_resources, 'acquire_codebase_resources', acquire)
    monkeypatch.setattr(context, '_validate_source384_context', lambda **kwargs: {'checked': True})
    assert context.validate_source384_context(repository='/authored', expected_receipt={}, timeout_seconds=total) == {'checked': True}
    assert expected - .1 <= waits[0] <= expected
