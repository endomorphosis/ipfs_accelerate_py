"""Disposable actual CLI loop and native DuckDB signal boundary qualification."""
import hashlib
import importlib.machinery
import json
import os
from pathlib import Path
import signal
import sys
import types

source, output, mode = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
package = types.ModuleType('ipfs_accelerate_py')
package.__path__ = [str(source / 'ipfs_accelerate_py')]
package.__spec__ = importlib.machinery.ModuleSpec(package.__name__, loader=None, is_package=True)
package.__spec__.submodule_search_locations = package.__path__
sys.modules[package.__name__] = package
from ipfs_accelerate_py.agent_supervisor.todo_daemon import implementation_daemon as module
import duckdb

records = {'native_query_completed': False, 'closed': False, 'passes': 0,
           'module': module.__file__, 'source_sha256': hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()}


class ActualNativeCall:
    def __init__(self, **_kwargs):
        self.connection = duckdb.connect(':memory:', config={'threads': 1, 'memory_limit': '128MB'})
        def deliver(value: int) -> int:
            os.kill(os.getpid(), signal.SIGTERM)
            return value
        self.connection.create_function('deliver_shutdown', deliver)

    def run_once(self):
        records['passes'] += 1
        if mode == 'query':
            assert self.connection.execute('SELECT deliver_shutdown(23)').fetchall() == [(23,)]
            records['native_query_completed'] = True
        return {'unchanged': True, 'selection_idle_reason': 'private_signal_qualification'}

    def close(self):
        if mode == 'close':
            assert self.connection.execute('SELECT deliver_shutdown(23)').fetchall() == [(23,)]
            records['native_query_completed'] = True
        self.connection.close()
        records['closed'] = True


module.DatabaseImplementationDaemon = ActualNativeCall
module.bind_database_portal_execution_from_args = lambda *_args, **_kwargs: None
module.publish_database_daemon_pass_heartbeat = lambda **_kwargs: None
try:
    module.main(['--task-source-kind', 'duckdb', '--authority-mode', 'embedded',
                 '--database-path', str(output.parent / 'unused.duckdb'),
                 '--state-dir', str(output.parent), '--state-prefix', 'private-shutdown', '--once'])
except BaseException as exc:
    records['exception_type'] = type(exc).__name__
    records['exit_code'] = exc.code if isinstance(exc, SystemExit) else None
output.write_text(json.dumps(records, sort_keys=True))
