"""Actual SDK descendant producer and bounded real pytest collection.

These observations do not dispatch a provider, restart a supervisor, modify a
campaign, or claim completion. Legacy row controls qualify normalization only.
"""
from contextlib import contextmanager
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon import supervisor


@pytest.mark.parametrize('row,expected', [
    ({'cmdline':'python -m pytest --collect-only'}, True),
    ({'cmdline':['python','-m','pytest','--collect-only']}, True),
    ({'cmdline':('node','playwright','test')}, True),
    ({'cmdline':('bash','scripts/ops/legal_ir/run_legal_ir_8h_canary.sh')}, True),
    ({'argv':('node','vitest'), 'cmdline':'unrelated diagnostic'}, True),
    ({'argv':('sleep','10'), 'cmdline':'pytest incorrect diagnostic'}, False),
    ({'cmdline':['pytest',17]}, False),
    ({'cmdline':None}, False),
    ({'cmdline':'sleep 10'}, False),
    ({}, False),
])
def test_explicit_legacy_row_types_do_not_join_scalar_characters(monkeypatch,row,expected):
    monkeypatch.setattr(supervisor,'descendant_processes',lambda _pid:[row])
    assert supervisor.active_validation_subprocess_exists(123) is expected


@contextmanager
def _real_collect_owner(tmp_path):
    owner = tmp_path/'inert_owner.py'
    collector = tmp_path/'work'; collector.mkdir()
    (collector/'test_sample.py').write_text('def test_sample(): assert True\n')
    (collector/'conftest.py').write_text('''from pathlib import Path
import time
def pytest_sessionstart(session):
 root=Path(__file__).parent
 (root/'ready').touch()
 while not (root/'release').exists(): time.sleep(.01)
''')
    owner.write_text('''from pathlib import Path
import json,os,subprocess,sys,time
root=Path(sys.argv[1]);workspace=root/'work'
with (root/'collector.log').open('wb') as output:
 child=subprocess.Popen([sys.executable,'-P','-B','-m','pytest','--collect-only','-q','-p','no:cacheprovider',
  '--basetemp',str(root/'pytest-temp'),'test_sample.py'],cwd=workspace,
  env=dict(os.environ,PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'),stdout=output,stderr=subprocess.STDOUT)
 try:
  (root/'child.json').write_text(json.dumps({'pid':child.pid}))
  while not (root/'owner-release').exists(): time.sleep(.01)
 finally:
  (workspace/'release').touch()
  try:child.wait(timeout=10)
  except subprocess.TimeoutExpired:child.kill();child.wait(timeout=5)
''')
    with (tmp_path/'owner.log').open('wb') as output:
        process=subprocess.Popen([sys.executable,'-P','-B',str(owner),str(tmp_path)],
            stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            deadline=time.monotonic()+20
            while not (collector/'ready').exists() and process.poll() is None and time.monotonic()<deadline:
                time.sleep(.01)
            assert (collector/'ready').exists(), (tmp_path/'owner.log').read_text()[-1000:]
            yield process,json.loads((tmp_path/'child.json').read_text())['pid']
        finally:
            (tmp_path/'owner-release').touch()
            try:process.wait(timeout=15)
            except subprocess.TimeoutExpired:process.kill();process.wait(timeout=5)


def test_actual_sdk_string_producer_recognizes_only_current_owned_collector(tmp_path):
    assert Path(supervisor.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[2])
    with _real_collect_owner(tmp_path) as (owner,child_pid):
        rows=supervisor.descendant_processes(owner.pid)
        child=next(row for row in rows if row['pid']==child_pid)
        assert type(child['cmdline']) is str and 'pytest' in child['cmdline']
        assert isinstance(child['argv'],tuple) and 'pytest' in child['argv']
        # The former expression deterministically misses this genuine row.
        assert 'pytest' not in ' '.join(child['cmdline'])
        assert supervisor.active_validation_subprocess_exists(owner.pid) is True
        (tmp_path/'work/release').touch()
        deadline=time.monotonic()+5
        while any(row['pid']==child_pid and row['argv'] for row in supervisor.descendant_processes(owner.pid)) and time.monotonic()<deadline:
            time.sleep(.01)
        unrelated=subprocess.Popen([sys.executable,'-I','-B','-c','import time;time.sleep(15)','--pytest-inert-sentinel'])
        try:
            assert unrelated.poll() is None
            assert supervisor.active_validation_subprocess_exists(owner.pid) is False
            assert unrelated.poll() is None
        finally:
            unrelated.terminate();unrelated.wait(timeout=5)
        (tmp_path/'owner-release').touch();owner.wait(timeout=15)
        assert supervisor.active_validation_subprocess_exists(owner.pid) is False
    assert supervisor.active_validation_subprocess_exists(True) is False
    assert supervisor.active_validation_subprocess_exists(0) is False


def test_full_supervisor_delegates_after_missing_owner_refusal_without_other_changes():
    """Source-only delegation contract; do not fake an unavailable SDK import."""
    path=Path(supervisor.__file__).with_name('implementation_supervisor.py')
    module=ast.parse(path.read_text())
    cls=next(n for n in module.body if isinstance(n,ast.ClassDef) and n.name=='PortalImplementationSupervisor')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_active_validation_subprocess_exists')
    # Keep the original missing-daemon check and call the actual tested owner.
    assert any(isinstance(n,ast.If) and isinstance(n.test,ast.UnaryOp) for n in method.body)
    returned=method.body[-1]
    assert isinstance(returned,ast.Return) and isinstance(returned.value,ast.Call)
    assert isinstance(returned.value.func,ast.Name) and returned.value.func.id=='active_validation_subprocess_exists'
    assert isinstance(returned.value.args[0],ast.Name) and returned.value.args[0].id=='daemon_pid'


def test_actual_full_supervisor_wrapper_observes_real_child_without_starting_a_daemon(tmp_path):
    from ipfs_accelerate_py.agent_supervisor.todo_daemon import implementation_supervisor as native
    assert Path(native.__file__).resolve() == Path(supervisor.__file__).with_name('implementation_supervisor.py').resolve()
    state = tmp_path/'state'; state.mkdir()
    config = native.PortalSupervisorConfig(todo_path=tmp_path/'synthetic-board.md',
        state_path=state/'task_state.json', strategy_path=state/'strategy.json',
        events_path=state/'events.jsonl', state_dir=state, repo_root=tmp_path, implement=False)
    instance = object.__new__(native.PortalImplementationSupervisor)
    instance.config = config
    assert instance._active_validation_subprocess_exists() is False
    scope = tmp_path/'owner'; scope.mkdir()
    with _real_collect_owner(scope) as (owner,_child):
        path = instance._managed_daemon_pid_path()
        path.write_text(str(owner.pid)); path.chmod(0o600)
        assert instance._active_validation_subprocess_exists() is True
        (scope/'owner-release').touch(); owner.wait(timeout=15)
        assert instance._active_validation_subprocess_exists() is False

