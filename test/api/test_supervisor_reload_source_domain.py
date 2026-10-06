"""Source reload observes supervisor code and preserves the native launch."""
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon import implementation_supervisor as module


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()


def repository(root, *, supervisor=False):
    root.mkdir()
    git(root, 'init', '-q')
    path = root / (module.CONTROL_PLANE_SOURCE_PATHS[0] if supervisor else 'target.py')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('# authored source\n')
    git(root, 'add', '.')
    git(root, '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid',
        'commit', '-qm', 'Authored source')
    return root


def test_unrelated_commits_do_not_nominate_a_supervisor_reload(tmp_path, monkeypatch):
    installed = repository(tmp_path / 'installed', supervisor=True)
    target = repository(tmp_path / 'target')
    loaded = module._read_control_plane_source_snapshot(installed)
    monkeypatch.setattr(module, 'IMPORTED_CONTROL_PLANE_SOURCE', loaded)
    selected = module._control_plane_probe_root(target)
    assert selected == installed
    assert not module._control_plane_update_is_pending(loaded, module._read_control_plane_source_snapshot(selected))
    (target / 'target.py').write_text('# unrelated candidate\n')
    git(target, 'add', '.')
    git(target, '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid', 'commit', '-qm', 'Candidate')
    assert not module._control_plane_update_is_pending(loaded, module._read_control_plane_source_snapshot(selected))
    (installed / module.CONTROL_PLANE_SOURCE_PATHS[0]).write_text('# changed supervisor\n')
    assert module._control_plane_update_is_pending(loaded, module._read_control_plane_source_snapshot(selected))


def test_operator_supervisor_checkout_stays_bound_when_its_files_are_deleted(tmp_path, monkeypatch):
    installed = repository(tmp_path / 'installed', supervisor=True)
    target = repository(tmp_path / 'operator', supervisor=True)
    loaded = module._read_control_plane_source_snapshot(target)
    monkeypatch.setattr(module, 'IMPORTED_CONTROL_PLANE_SOURCE', module._read_control_plane_source_snapshot(installed))
    selected = module._control_plane_probe_root(target)
    assert selected == target
    (target / module.CONTROL_PLANE_SOURCE_PATHS[0]).unlink()
    assert module._control_plane_probe_root(target) == target  # Tracked deletion at startup.
    assert module._control_plane_update_is_pending(loaded, module._read_control_plane_source_snapshot(selected))


@pytest.mark.parametrize('entry', ['implementation_supervisor', 'implementation_supervisor_runner'])
def test_native_reload_preserves_original_interpreter_flags_module_and_arguments(tmp_path, monkeypatch, entry):
    config = SimpleNamespace(repo_root=tmp_path, supervisor_script_path=None)
    tail = ['--state-prefix', 'accepted-lane', '--implementation-command', 'authored command']
    original = [sys.executable, '-P', '-B', '-X', 'utf8', '-m',
        'ipfs_accelerate_py.agent_supervisor.todo_daemon.' + entry, *tail]
    monkeypatch.setattr(sys, 'orig_argv', original)
    monkeypatch.setattr(sys, 'argv', ['entry.py', *tail])
    captured = module._captured_supervisor_reload_arguments(config)
    assert captured == tuple(original)
    runtime = object.__new__(module.TodoImplementationSupervisor)
    runtime.config = config
    runtime._control_plane_reload_arguments = captured
    monkeypatch.setattr(sys, 'argv', ['unrelated', '--changed'])
    calls = []
    class Reloaded(Exception):
        pass
    def execute(executable, args):
        calls.append((executable, args))
        raise Reloaded
    monkeypatch.setattr(module.os, 'execv', execute)
    with pytest.raises(Reloaded):
        runtime._reload_for_control_plane_update()
    assert calls == [(sys.executable, original)]


def test_configured_wrapper_is_preserved_but_unrelated_invocations_are_not(tmp_path, monkeypatch):
    config = SimpleNamespace(repo_root=tmp_path, supervisor_script_path='wrapper.py')
    wrapper = tmp_path / 'wrapper.py'
    original = [sys.executable, '-P', str(wrapper), '--state-prefix', 'lane']
    monkeypatch.setattr(sys, 'orig_argv', original)
    monkeypatch.setattr(sys, 'argv', [str(wrapper), '--state-prefix', 'lane'])
    assert module._captured_supervisor_reload_arguments(config) == tuple(original)
    monkeypatch.setattr(sys, 'orig_argv', [sys.executable, '-m', 'pytest', '--state-prefix', 'lane'])
    assert module._captured_supervisor_reload_arguments(config) is None
    monkeypatch.setattr(sys, 'orig_argv', [sys.executable, '-P', '-m',
        'ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor_runner',
        '--state-prefix', 'lane'])
    assert module._captured_supervisor_reload_arguments(config) is None  # Explicit wrapper wins.
    monkeypatch.setattr(sys, 'orig_argv', original)
    monkeypatch.setattr(sys, 'argv', [str(wrapper), '--state-prefix', 'different'])
    assert module._captured_supervisor_reload_arguments(config) is None
