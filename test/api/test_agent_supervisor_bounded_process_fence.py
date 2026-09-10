"""A stopped vfork child must not consume the whole native fence deadline."""
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon import core
from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import read_process_birth


def test_unstoppable_parent_reserves_time_to_kill_stopped_child(monkeypatch):
    now = [0.0]
    signals = []
    killed = [False]
    parent, child = 45451, 45452
    def snapshot():
        return core.ProcessIdentitySnapshot.observed({} if killed[0] else {
            parent: ('D', 1, parent, parent, '123'),
            child: ('T', parent, parent, parent, '124'),
        })
    def send(pid, sig):
        signals.append((pid, sig))
        if sig == signal.SIGKILL:
            killed[0] = True
    monkeypatch.setattr(core, '_process_identity_snapshot', snapshot)
    monkeypatch.setattr(core.time, 'monotonic', lambda: now[0])
    monkeypatch.setattr(core.time, 'sleep', lambda seconds: now.__setitem__(0, now[0] + seconds))
    monkeypatch.setattr(core.os, 'kill', send)
    monkeypatch.setattr(core.os, 'killpg', send)
    assert core.terminate_pid_tree(
        parent, grace_seconds=1.0, freeze_first=True, require_gone=True,
        owned_process_group_id=parent, expected_root_start_time_ticks=123,
        strict_timeout_seconds=1.0,
    )
    assert any(sig == signal.SIGKILL for _, sig in signals)
    assert now[0] < 1.0


@pytest.mark.parametrize("failure", ["permission", "uninterruptible", "observation_lost"])
def test_failed_fence_never_advertises_capacity(monkeypatch, failure):
    now = [0.0]
    signals = []
    parent = 45451
    def snapshot():
        if failure == "observation_lost" and signals:
            return core.ProcessIdentitySnapshot.unavailable("procfs unavailable")
        return core.ProcessIdentitySnapshot.observed({
            parent: ('D', 1, parent, parent, '123'),
        })
    def send(pid, sig):
        signals.append((pid, sig))
        if failure == "permission":
            raise PermissionError()
    monkeypatch.setattr(core, '_process_identity_snapshot', snapshot)
    monkeypatch.setattr(core.time, 'monotonic', lambda: now[0])
    monkeypatch.setattr(core.time, 'sleep', lambda seconds: now.__setitem__(0, now[0] + seconds))
    monkeypatch.setattr(core.os, 'kill', send)
    monkeypatch.setattr(core.os, 'killpg', send)
    assert not core.terminate_pid_tree(
        parent, grace_seconds=1.0, freeze_first=True, require_gone=True,
        owned_process_group_id=parent, expected_root_start_time_ticks=123,
        strict_timeout_seconds=1.0,
    )
    if failure != "observation_lost":
        assert any(sig == signal.SIGKILL for _, sig in signals)
    assert now[0] <= 1.03


def test_vfork_parent_and_stopped_child_are_fenced_within_bound(tmp_path):
    cc = shutil.which('cc')
    if not cc or not Path('/proc').is_dir():
        pytest.skip('Linux procfs and C compiler required for real vfork regression')
    source = tmp_path / 'vfork.c'
    binary = tmp_path / 'vfork'
    source.write_text('''#include <unistd.h>
#include <signal.h>
#include <sys/wait.h>
int main(void) {
    pid_t child = vfork();
    if (child == 0) { raise(SIGSTOP); _exit(0); }
    if (child < 0) return 2;
    waitpid(child, 0, 0);
    return 0;
}
''')
    subprocess.run([cc, '-O0', str(source), '-o', str(binary)], check=True, capture_output=True)
    process = subprocess.Popen([str(binary)], start_new_session=True)
    try:
        birth = read_process_birth(process.pid)
        assert birth is not None
        until = time.monotonic() + 3
        while time.monotonic() < until:
            table = core._process_identity_snapshot().processes
            root = table.get(process.pid)
            children = [row for row in table.values() if row[1] == process.pid]
            if root and root[0] == 'D' and any(row[0] == 'T' for row in children):
                break
            time.sleep(.01)
        else:
            pytest.fail('fixture did not reach vfork wait with a stopped child')
        assert core.terminate_pid_tree(
            process.pid, grace_seconds=1.0, freeze_first=True, require_gone=True,
            owned_process_group_id=process.pid,
            expected_root_start_time_ticks=birth.start_time_ticks,
            strict_timeout_seconds=1.0,
        )
        process.wait(timeout=2)
    finally:
        # Test-owned session only; reap even if the regression leaves it frozen.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=2)
