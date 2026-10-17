"""Read-only denial before the original report-only public pass effects.

This deliberately stops the whole private legacy lane on an unresolved deny.
Returning None does not grant a pass: the original actual native-host guard is
still mandatory. It creates no journal, connection, lease, owner or subprocess.
"""
import importlib
from pathlib import Path
import sys
from types import ModuleType

TODO = 'ipfs_accelerate_py.agent_supervisor.todo_daemon.'
DAEMON = TODO+'implementation_daemon'
LIFECYCLE = 'ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle'
DISPOSITION = TODO+'attempt_disposition'
DAEMON_SHA = '67f4959c391776a1361a98f2a5f8d137e86ea5651e9ced0ab96fca3eb91f15a0'
LIFECYCLE_SHA = 'b7f8508b4b4b50d5bc259c186b55c4c6f87e938d99689a913383a8a35130249f'
DISPOSITION_SHA = '980597e97d337aaa90c0a93527229be37974db1f6351b0eb33a98c95cc8fe43b'


def require_clear_report_only_disposition(daemon):
    # Imports only the reviewed stdlib helper. The original daemon and store
    # must already be the actual canonical modules constructed by the caller.
    disposition = importlib.import_module(DISPOSITION)
    disposition.require(type(disposition) is ModuleType
        and sys.modules.get(DISPOSITION) is disposition,
        'actual_loaded_disposition_module_required')
    dm, lm = sys.modules.get(DAEMON), sys.modules.get(LIFECYCLE)
    disposition.require(type(dm) is ModuleType and type(lm) is ModuleType
        and type(daemon) is dm.PortalImplementationDaemon
        and daemon.report_only_candidate_freezing is True,
        'actual_original_report_only_daemon_entry_required')
    pins = {}
    for module, name, digest in ((dm, DAEMON, DAEMON_SHA),
            (lm, LIFECYCLE, LIFECYCLE_SHA), (disposition, DISPOSITION, DISPOSITION_SHA)):
        path = Path(module.__file__).absolute()
        disposition.require(str(path).endswith('/'+name.replace('.', '/')+'.py'),
            'exact_original_entry_module_origin_required')
        pins[str(path)] = digest
    scope = disposition.OriginalDispositionScope(dm, lm, daemon, source_sha256=pins)
    scope.assert_no_denials_for_state()
    return None  # Absence of a denial grants no native or productive authority.
