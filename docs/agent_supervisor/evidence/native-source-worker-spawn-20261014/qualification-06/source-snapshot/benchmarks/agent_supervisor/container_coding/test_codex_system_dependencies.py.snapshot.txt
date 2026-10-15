"""Finite system dependency allowance and fail-fast deployment, without containers."""
import asyncio
from types import SimpleNamespace

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_deployment as deployment
from benchmarks.agent_supervisor.container_coding.test_terminal_deployment import _inputs
from benchmarks.agent_supervisor.container_coding.test_terminal_uv_layout import RecordingEnvironment


class DependencyEnvironment(RecordingEnvironment):
    def __init__(self, failure=None):
        super().__init__()
        self.failure = failure

    async def exec(self, command, **kwargs):
        result = await super().exec(command, **kwargs)
        if command.endswith('install -y --no-install-recommends nodejs npm'):
            if isinstance(self.failure, Exception):
                raise self.failure
            if self.failure is not None:
                return SimpleNamespace(return_code=self.failure, stdout='authored dependency stopped', stderr='')
        if command == 'codex --version':
            return SimpleNamespace(return_code=0, stdout='codex-cli ' + deployment.CODEX_VERSION, stderr='')
        return result


def _deploy(tmp_path, environment, *, install_codex=True):
    deployment.build_runtime_archive(output=tmp_path / 'bundle', **_inputs(tmp_path))
    auth = tmp_path / 'auth.json'
    auth.write_text('{"authored_fixture":true}')
    return asyncio.run(deployment.deploy_supervisor(environment, archive_dir=tmp_path / 'bundle',
        output=tmp_path / 'deployed', install_codex=install_codex, auth_json=auth))


def test_dependency_allowance_reaches_harbor_without_extending_native_deadline(tmp_path):
    environment = DependencyEnvironment()
    report = _deploy(tmp_path, environment)
    dependencies = [(command, kwargs) for command, kwargs in environment.commands
                    if command.endswith('install -y --no-install-recommends nodejs npm')]
    assert len(dependencies) == 1
    command, options = dependencies[0]
    assert command == (
        'timeout --signal=TERM --kill-after=5s 300s env DEBIAN_FRONTEND=noninteractive '
        'apt-get -o Acquire::Retries=1 -o Acquire::http::Timeout=30 '
        '-o Acquire::https::Timeout=30 install -y --no-install-recommends nodejs npm')
    assert options['timeout_sec'] == 330
    assert options['user'] == 'root'
    native = [(command, kwargs) for command, kwargs in environment.commands if 'install_timeout=180' in command]
    assert len(native) == 1
    assert native[0][1]['timeout_sec'] == 210
    assert native[0][1]['user'] == 'supervisor'
    assert environment.commands.index(dependencies[0]) < environment.commands.index(native[0])
    assert report['qualified'] is True and report['provider_calls'] == 0
    assert report['task_source_preserved'] is True
    assert environment.default_user == 'root'


@pytest.mark.parametrize('failure', [100, 124, 137, TimeoutError('authored Harbor timeout')])
def test_failed_dependency_stops_before_native_install_or_credential_transfer(tmp_path, failure):
    environment = DependencyEnvironment(failure)
    exception = TimeoutError if isinstance(failure, Exception) else RuntimeError
    with pytest.raises(exception):
        _deploy(tmp_path, environment)
    commands = [command for command, _ in environment.commands]
    assert commands[-1].endswith('install -y --no-install-recommends nodejs npm')
    assert not any('install_timeout=' in command or 'codex login status' in command
                   or 'provider-bin' in command or ' uv python install ' in command for command in commands)
    assert len(environment.uploads) == 1
    assert str(environment.uploads[0][0]).endswith('/runtime.tar.gz')
    assert environment.default_user == 'root'
    assert not (tmp_path / 'deployed/deployment.json').exists()
    if not isinstance(failure, Exception):
        assert (tmp_path / 'deployed/codex-system-dependencies.log').read_text() == 'authored dependency stopped'


def test_deployment_without_codex_does_not_install_its_system_dependencies(tmp_path):
    environment = DependencyEnvironment()
    report = _deploy(tmp_path, environment, install_codex=False)
    assert not any('nodejs npm' in command or 'install_timeout=' in command for command, _ in environment.commands)
    assert report['codex_installed'] is False
    assert report['provider_calls'] == 0
