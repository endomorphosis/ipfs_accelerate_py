"""Source-only exact H3 successor assembler; no imports/jobs/provider execution."""
from pathlib import Path
import hashlib
import difflib

HERE = Path(__file__).absolute().parent
BASE = Path('/home/barberb/lift_coding/artifacts/legal-ir-callable-handoff-20261017-01/handoff-v03/proposed/ipfs_accelerate_py/agent_supervisor/runtime/ordinary_success_handoff.py')
EXPECTED = '47464be42c3c11e15dc9144c807badcb383bfd8d6ba9f8f4a0445c8b570d298a'


def assemble(raw):
    assert hashlib.sha256(raw).hexdigest() == EXPECTED
    text = raw.decode()
    def one(before, after):
        nonlocal text
        assert text.count(before) == 1, before[:100]
        text = text.replace(before, after, 1)
    one('from contextvars import ContextVar\n', 'from contextvars import ContextVar\nfrom contextlib import contextmanager\nimport threading\n')
    one('        self.private_allocations = []\n', '        self.private_allocations = []\n        self.original_groups = {}\n        self.owner_thread = threading.get_ident()\n        self.primary_native, self.watchdog_native = owner.native_lease, watchdog_owner.native_lease\n        self.callback_capture, self.freezer_scope, self.stopped_pending = None, None, None\n')
    one('    def query(self, arguments):\n', '    def query(self, arguments):\n        _ordinary_admin_capacity(self)\n')
    one('        self.watchdog_owner.check_usage(self.watchdog_owner.directory,\n', '        _capture_native_group(self, self.watchdog_owner, lease._watchdog)\n        self.watchdog_owner.check_usage(self.watchdog_owner.directory,\n')
    one('        self.owner.check_usage(self.owner.directory, child_pid=process.pid, full=True)\n', '        _capture_native_group(self, self.owner, process)\n        self.owner.check_usage(self.owner.directory, child_pid=process.pid, full=True)\n')
    one('    def reap_observed_transport_after_failure(self):\n', '    def reap_observed_transport_after_failure(self, *, stopped=False):\n')
    one('        if self.processes or self.allocated_leases:\n            for owner in (self.owner, self.watchdog_owner):\n', '        if stopped:\n            try:\n                _original_native_groups_stopped(self)\n            except Exception:\n                pending = True\n        elif self.processes or self.allocated_leases:\n            for owner in (self.owner, self.watchdog_owner):\n')
    one('            raise OrdinaryCustodyPending(self.owner, self.watchdog_owner, self.allocated_leases,\n                                         self.processes, self.fence, self.daemon_id, self.runtime,\n                                         custody_hold=self.custody_hold, private_allocations=self.private_allocations)\n', '            raise _retain_ordinary_pending(self)\n')
    one('    current.reap_observed_transport_after_failure()\n\n\ndef _pending_custody(current):', '    if current.freezer_scope is None:\n        current.reap_observed_transport_after_failure()\n    else:\n        scope = current.freezer_scope\n        capture = scope["capture"]\n        entry = _AUTHENTICATED_CALLBACKS.get(capture)\n        need(entry is not None and entry["current"] is current and entry["outcome"] is outcome\n             and current.callback_capture is capture and entry["pid"] == os.getpid()\n             and entry["thread"] == threading.get_ident() and entry["used"]\n             and not scope["lease"].released, "actual_active_authenticated_freezer_scope_required")\n        _original_native_groups_stopped(current)\n\n\ndef _pending_custody(current):')
    one('    return OrdinaryCustodyPending(current.owner, current.watchdog_owner, current.allocated_leases,\n        current.processes, current.fence, current.daemon_id, current.runtime,\n        custody_hold=current.custody_hold, private_allocations=current.private_allocations)\n', '    return _retain_ordinary_pending(current)\n')
    one('        current.custody_module.forget_acknowledged_custody_hold(current.custody_hold)\n', '        current.custody_module.forget_acknowledged_custody_hold(current.custody_hold)\n        _forget_unused_callback(current)\n')
    return text + '\n\n' + (HERE / 'handoff_v04_additions.py.txt').read_text()


if __name__ == '__main__':
    original = BASE.read_bytes()
    successor = assemble(original)
    output = HERE / 'proposed/ipfs_accelerate_py/agent_supervisor/runtime/ordinary_success_handoff.py'
    output.write_text(successor)
    (HERE / 'ordinary_success_handoff-v04.patch').write_text(''.join(difflib.unified_diff(
        original.decode().splitlines(True), successor.splitlines(True), fromfile='handoff-v03/ordinary_success_handoff.py',
        tofile='native-host-v01/ordinary_success_handoff.py')))
