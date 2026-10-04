"""Signed subprocess maintenance and catalog limits at the factory boundary."""
import os
import subprocess

import pytest

from ipfs_accelerate_py.agent_supervisor.control.control_contracts import Operation, get_operation_catalog
from ipfs_accelerate_py.agent_supervisor.entrypoints.admitted_benchmark_runtime import (
    AdmittedBenchmarkRuntime, _bounded_git_environment,
)
from ipfs_accelerate_py.agent_supervisor.entrypoints.isolated_benchmark_runtime import IsolatedBenchmarkRuntime


@pytest.mark.parametrize('factory', [AdmittedBenchmarkRuntime, IsolatedBenchmarkRuntime])
@pytest.mark.parametrize('timeout', [True, 1999, 30000.0, 30001, 90000, 120000])
def test_factory_rejects_unexecutable_start_stop_budget_before_state_creation(tmp_path, factory, timeout):
    maximum = min(get_operation_catalog().by_name[op.value].bounds.timeout_ms
                  for op in (Operation.START, Operation.STOP))
    assert maximum == 30000
    kwargs = {'admission': None, 'server': None, 'source': None} if factory is AdmittedBenchmarkRuntime else {}
    with pytest.raises(ValueError, match='START/STOP catalog'):
        factory.create(tmp_path/'launch', timeout_ms=timeout, **kwargs)
    assert not (tmp_path/'launch').exists()


@pytest.mark.parametrize('candidate_runner', [False, True])
@pytest.mark.parametrize('inherited_parameters', ['', "'gc.auto'='1' 'gc.autoDetach'='true' 'maintenance.auto'='true'"])
def test_signed_git_configuration_overrides_repository_auto_maintenance(tmp_path, candidate_runner, inherited_parameters, monkeypatch):
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    for key, value in (('gc.auto','1'), ('gc.autoDetach','true'), ('maintenance.auto','true')):
        subprocess.run(['git', '-C', str(tmp_path), 'config', key, value], check=True)
    monkeypatch.setenv('GIT_CONFIG_PARAMETERS', inherited_parameters)
    environment = {**os.environ, **_bounded_git_environment(candidate_runner=candidate_runner)}
    for key, expected in (('gc.auto','0'), ('gc.autoDetach','false'), ('maintenance.auto','false')):
        actual = subprocess.check_output(['git','-C',str(tmp_path),'config','--get',key], env=environment, text=True).strip()
        assert actual == expected
    # The exact launch config suppresses detached optional gc without bypassing
    # Git itself or ignoring any background process in the lifecycle snapshot.
    subprocess.run(['git', '-C', str(tmp_path), 'gc', '--auto'], env=environment,
                   check=True, capture_output=True, timeout=5)
    assert not (tmp_path/'.git/gc.pid').exists()
