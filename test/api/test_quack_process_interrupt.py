"""Native DuckDB errors must not turn a process stop into a retryable read."""
import os
from pathlib import Path
import subprocess
import sys

import pytest

from ipfs_accelerate_py.agent_supervisor.task_sources import duckdb_state


@pytest.mark.parametrize('boundary', ['attach', 'consume'])
def test_real_sigterm_during_duckdb_query_closes_without_retry(tmp_path, boundary):
    script = tmp_path / 'interrupt.py'
    script.write_text('''
import os, signal, threading
import duckdb
from ipfs_accelerate_py.agent_supervisor.task_sources import duckdb_state as state
boundary = os.environ['TEST_INTERRUPT_BOUNDARY']
raw = duckdb.connect(':memory:', config={'threads': 1, 'memory_limit': '256MB'})
calls = []
def stop(signum, frame):
    raise SystemExit(128 + signum)
signal.signal(signal.SIGTERM, stop)
def query():
    calls.append('query')
    timer = threading.Timer(.05, lambda: os.kill(os.getpid(), signal.SIGTERM))
    timer.start()
    try:
        return raw.execute('SELECT sum(i) FROM range(10000000000) t(i)')
    finally:
        timer.cancel()
        timer.join()
class Connection:
    def execute(self, *args, **kwargs):
        return query()
    def fetchall(self):
        return query()
    def close(self):
        calls.append('close')
        raw.close()
    def interrupt(self):
        raw.interrupt()
connection = Connection()
try:
    if boundary == 'attach':
        duckdb.connect = lambda *a, **k: connection
        native = getattr(state, '_open_quack_transport_connection_once', None)
        if native:
            native('quack:127.0.0.1:45123', token='test_token_123')
        else:
            state._attach_quack_once('quack:127.0.0.1:45123', 'test_token_123')
    elif boundary == 'consume':
        state._consume_duckdb_result(connection)
    else:
        from ipfs_accelerate_py.agent_supervisor.task_sources.quack_read_continuity import owned_connection_deadline
        import time
        with owned_connection_deadline(connection, time.monotonic()+3):
            query()
except SystemExit as error:
    assert error.code == 143
    assert calls.count('query') == 1
    if boundary == 'attach':
        assert calls.count('close') == 1
    print('native-stop-preserved')
else:
    raise AssertionError('stop was swallowed')
finally:
    raw.close()
''')
    root = Path(duckdb_state.__file__).resolve().parents[3]
    env = {**os.environ, 'PYTHONPATH': str(root), 'TEST_INTERRUPT_BOUNDARY': boundary,
           'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
    result = subprocess.run([sys.executable, str(script)], cwd=root, env=env,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'native-stop-preserved' in result.stdout


@pytest.mark.parametrize('stop', [SystemExit(143), KeyboardInterrupt()])
def test_restore_exact_process_exception_and_exit_code(stop):
    from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_interrupts import reraise_process_interrupt
    wrapper = RuntimeError('Query interrupted')
    wrapper.__cause__ = stop
    with pytest.raises(type(stop)) as caught:
        reraise_process_interrupt(wrapper)
    assert caught.value is stop


def test_query_text_and_suppressed_context_are_not_stop_authority():
    from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_interrupts import reraise_process_interrupt
    ordinary = RuntimeError('Query interrupted')
    reraise_process_interrupt(ordinary)
    ordinary.__context__ = SystemExit(143)
    ordinary.__suppress_context__ = True
    reraise_process_interrupt(ordinary)
    ordinary.__cause__ = ordinary
    reraise_process_interrupt(ordinary)

