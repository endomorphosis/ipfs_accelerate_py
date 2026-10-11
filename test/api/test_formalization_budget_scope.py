"""An extended time profile does not relax legacy or replay authority bounds."""
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import header_intent_applicability as scope
from ipfs_accelerate_py.agent_supervisor.runtime.benchmark_budget_policy import FORMALIZATION_BUDGET_PROFILE as PROFILE


def test_long_work_scope_retains_profile_without_renewing_captured_or_nested_deadlines(monkeypatch):
    now = [1000.]
    monkeypatch.setattr(scope.time, 'monotonic', lambda: now[0])
    monkeypatch.setenv('IPFS_DATASETS_PROOF_RESOURCE_PROFILE', 'local-benchmark@1')
    with scope.local_benchmark_applicability_budget(deadline_monotonic=7300., budget_profile=PROFILE):
        captured = scope.capture_applicability_budget()
        assert captured.budget_profile == PROFILE
        assert scope.applicability_replay_timeout() == 120.
        with scope.applicability_budget(600):
            assert scope.capture_applicability_budget().deadline_monotonic == 1600.
    now[0] = 7200.
    with scope.resume_applicability_budget(captured):
        with scope.applicability_budget(600):
            assert scope.capture_applicability_budget().deadline_monotonic == 7300.
    now[0] = 7300.
    with pytest.raises(TimeoutError, match='expired'):
        with scope.resume_applicability_budget(captured): pass


@pytest.mark.parametrize('profile, duration', [(None, 901), (PROFILE, 6301), ('unknown', 600), (True, 600)])
def test_scope_refuses_unknown_profiles_and_excessive_deadlines(monkeypatch, profile, duration):
    monkeypatch.setattr(scope.time, 'monotonic', lambda: 1000.)
    monkeypatch.setenv('IPFS_DATASETS_PROOF_RESOURCE_PROFILE', 'local-benchmark@1')
    with pytest.raises(ValueError):
        with scope.local_benchmark_applicability_budget(deadline_monotonic=1000. + duration,
                budget_profile=profile): pass


def test_legacy_aggregate_deadline_remains_300_seconds():
    with pytest.raises(ValueError, match='aggregate'):
        with scope.applicability_budget(301): pass


@pytest.mark.parametrize('profile, lifetime', [(None, 901), (None, 2160), (PROFILE, 2161),
    (PROFILE, 2160.), (PROFILE, True), (PROFILE, 900), ('unknown', 2160)])
def test_native_long_lifetime_requires_exact_explicit_profile_before_state_creation(tmp_path, profile, lifetime):
    from ipfs_accelerate_py.agent_supervisor.entrypoints.admitted_benchmark_runtime import AdmittedBenchmarkRuntime
    with pytest.raises(ValueError):
        AdmittedBenchmarkRuntime.create(tmp_path / 'launch', admission=None, server=None, source=None,
            lifetime_seconds=lifetime, budget_profile=profile)
    assert not (tmp_path / 'launch').exists()


def test_native_launch_signs_profile_watchdog_and_owner_lifetime_and_rejects_tampering(tmp_path, monkeypatch):
    from benchmarks.agent_supervisor.container_coding.native_quack_qualification import open_existing_native_owner
    from ipfs_accelerate_py.agent_supervisor.control import profile_authority
    from ipfs_accelerate_py.agent_supervisor.entrypoints.admitted_benchmark_runtime import AdmittedBenchmarkRuntime
    from ipfs_accelerate_py.agent_supervisor.runtime.local_planning_admission import verify_local_benchmark_admission
    from ipfs_accelerate_py.agent_supervisor.task_sources.task_execution_route_policy import GROK_CODEX_EXECUTION_MODE
    from test.integration.test_admitted_benchmark_runtime import _prepare_implementation_fixture
    monkeypatch.setattr(profile_authority, '_LIFECYCLE_REGISTRY_ROOT_OVERRIDE', tmp_path / 'account')
    monkeypatch.delenv('IPFS_DATASETS_PROOF_RESOURCE_PROFILE', raising=False)
    prepared = _prepare_implementation_fixture(tmp_path / 'task')
    verified = verify_local_benchmark_admission(prepared['admission'])
    with open_existing_native_owner(database=Path(prepared['intent_database']),
            checkout=Path(prepared['repository']), state_dir=tmp_path / 'owner',
            repository_id=verified['manifest']['repository_cid'],
            execution_routes={prepared['task_id']: GROK_CODEX_EXECUTION_MODE}) as owner:
        runtime = AdmittedBenchmarkRuntime.create(tmp_path / 'launch', admission=prepared['admission'],
            server=owner.server, source=owner.source, implement=True, implementation_command='/usr/bin/true',
            implementation_timeout_seconds=1860, lifetime_seconds=2160, budget_profile=PROFILE)
        try:
            assert runtime.manifest['budget_profile'] == PROFILE
            assert runtime.manifest['lifetime_seconds'] == 2160
            assert runtime.manifest['implementation_timeout_seconds'] == 1860
            runtime._verify()
            runtime.budget_profile = None
            with pytest.raises(ValueError, match='budget profile differs'):
                runtime._verify()
            runtime.budget_profile = PROFILE
            runtime.manifest['lifetime_seconds'] = 2161
            with pytest.raises(ValueError, match='2160-second lifetime'):
                runtime._verify()
        finally:
            runtime.budget_profile = PROFILE
            runtime.manifest['lifetime_seconds'] = 2160
            assert not runtime.process.snapshot(runtime.profile).members
            runtime.close()


@pytest.mark.parametrize('profile, timeout', [(None, 181), (PROFILE, 3601),
    (PROFILE, True), (PROFILE, float('inf')), ('unknown', 180)])
def test_source384_extended_deadline_is_explicit_before_any_capture(tmp_path, profile, timeout):
    from ipfs_accelerate_py.agent_supervisor.runtime.source384_repository_context import prepare_source384_context
    with pytest.raises(ValueError):
        prepare_source384_context(repository=tmp_path / 'absent', source_hashes={}, output=tmp_path / 'out',
            config_path=tmp_path / 'config', timeout_seconds=timeout, budget_profile=profile)
    assert not (tmp_path / 'out').exists()


def test_source384_explicit_budget_reaches_capture_without_relaxing_replay(tmp_path, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.runtime import source384_repository_context as source
    class CaptureBoundaryReached(RuntimeError): pass
    def boundary(*args, **kwargs): raise CaptureBoundaryReached()
    monkeypatch.setattr(source, '_output', boundary)
    with pytest.raises(CaptureBoundaryReached):
        source.prepare_source384_context(repository=tmp_path, source_hashes={}, output=tmp_path / 'out',
            config_path=tmp_path / 'config', timeout_seconds=3599., budget_profile=PROFILE)
    with pytest.raises(ValueError, match='observation deadline'):
        source.validate_source384_context(repository=tmp_path, expected_receipt={}, timeout_seconds=181)
