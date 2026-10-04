"""Run one retained stale API control against exact pre-fix owners."""
import importlib.abc
import importlib.util
from pathlib import Path
import sys
A = Path("/home/barberb/lift_coding/.worktrees/ir-release-accelerate-20261002")
B = Path(__file__).resolve().parent / "owner-original"
class Original(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    paths = {"ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon": "todo_daemon/implementation_daemon.py",
             "ipfs_accelerate_py.agent_supervisor.todo_daemon.database_portal_bridge": "todo_daemon/database_portal_bridge.py",
             "ipfs_accelerate_py.agent_supervisor.runtime.task_context_bundle": "runtime/task_context_bundle.py"}
    def find_spec(self, fullname, path=None, target=None):
        if fullname in self.paths:
            owner = A / "ipfs_accelerate_py/agent_supervisor" / self.paths[fullname]
            return importlib.util.spec_from_file_location(fullname, owner, loader=self)
    def create_module(self, spec):
        return None
    def exec_module(self, module):
        old = B / Path(self.paths[module.__name__]).name
        exec(compile(old.read_bytes(), module.__file__, "exec"), module.__dict__)
sys.meta_path.insert(0, Original())
import pytest
raise SystemExit(pytest.main(sys.argv[1:]))
