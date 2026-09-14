"""Disposable four-lane process/FLOCK qualification; no task or board APIs."""
import contextlib
from dataclasses import replace
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import native_phased_graceful_recovery as phased
from ipfs_accelerate_py.agent_supervisor.runtime.native_launcher_quiescence import LauncherQuiescence

P = phased.process
pytestmark = pytest.mark.skipif(sys.platform != 'linux' or not hasattr(os, 'pidfd_open') or not hasattr(signal, 'pidfd_send_signal'), reason='Linux pidfd custody required')


@contextlib.contextmanager
def retained_fence(context, path):
    with context:
        before = path.stat()
        candidates = []
        for item in Path('/proc/self/fd').iterdir():
            try:
                current = os.fstat(int(item.name))
                if (current.st_dev,current.st_ino)==(before.st_dev,before.st_ino):
                    candidates.append(int(item.name))
            except OSError: pass
        assert len(candidates)==1
        fd=candidates[0]
        def check():
            current=os.fstat(fd)
            assert (current.st_dev,current.st_ino)==(before.st_dev,before.st_ino)
            assert path.stat().st_ino==before.st_ino
            rows=[row.split() for row in Path('/proc/self/fdinfo',str(fd)).read_text().splitlines() if row.startswith('lock:')]
            if not (len(rows)==1 and rows[0][2:6]==['FLOCK','ADVISORY','WRITE',str(os.getpid())]):
                raise P.GracefulRecoveryUnverified('positive_lock_missing')
        check()
        yield check


def test_reusable_mechanism_refuses_to_freeze_its_own_custodian(monkeypatch):
    own=P.observe_process(os.getpid())
    monkeypatch.setattr(signal,'pidfd_send_signal',lambda *_:pytest.fail('self-pause must refuse before signals'))
    with pytest.raises(P.GracefulRecoveryUnverified,match='cannot_freeze_itself'):
        LauncherQuiescence(controller=own,supervisors=[replace(own,pid=own.pid+1,parent=own.pid)],
                           lane_fence=lambda _:None,effect_gate=lambda:None,record_phase=lambda _:None)

def wait(predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(.01)
    raise AssertionError('private launcher fixture timeout')

@contextlib.contextmanager
def tree(root, mode='launch'):
    (root/'opaque-outcome.json').write_text('{"outcome":"unknown","attempt":"retained"}\n')
    log = (root/'actors.log').open('wb')
    child = subprocess.Popen([sys.executable, '-I', '-S',
                              str(Path(__file__).with_name('native_launcher_fixture.py')),
                              str(root), 'master', mode], cwd=root,start_new_session=True, stdout=log, stderr=log)
    original = []
    try:
        wait(lambda: all((root/f'ready-{i}').exists() for i in range(4)))
        roster = json.loads((root/'roster.json').read_text())
        master = P.observe_process(child.pid)
        wrappers = [P.observe_process(pid) for pid in roster['wrappers']]
        original = [master, *wrappers]
        yield child, master, wrappers
    finally:
        marker = root/'daemon.json'
        if marker.exists():
            pid = json.loads(marker.read_text())['pid']
            try:
                observed = P.observe_process(pid)
                if observed.parent in {b.pid for b in original} and str(root).encode() in P._read_proc(Path('/proc')/str(pid)/'cmdline'):
                    original.append(observed)
            except (FileNotFoundError, ProcessLookupError, P.GracefulRecoveryUnverified):
                pass
        for binding in reversed(original):
            try:
                P.require_exact_process(binding)
                with P._exact_pidfd(binding) as fd:
                    signal.pidfd_send_signal(fd, signal.SIGCONT)
                    signal.pidfd_send_signal(fd, signal.SIGTERM)
            except (FileNotFoundError, ProcessLookupError, P.GracefulRecoveryUnverified):
                pass
        try:
            child.wait(timeout=8)
        finally:
            log.close()

@contextlib.contextmanager
def flock(path):
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        deadline = time.monotonic()+1
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError('private launch fence remained held')
                time.sleep(.01)
        yield
    finally:
        os.close(fd)

def custody(root, master, wrappers, phases, *, gate=lambda:None):
    def fence(index):
        if index == 2:
            assert all(P._stat(b.pid)[0] not in {'T','t'} for b in wrappers)
            (root/'request-peer-progress').touch()
        return retained_fence(flock(root/f'lane{index}.lock'),root/f'lane{index}.lock')
    return LauncherQuiescence(controller=master, supervisors=wrappers,
                              lane_fence=fence, effect_gate=gate,
                              record_phase=phases.append, timeout_seconds=2)

def test_real_late_launch_is_nominated_once_after_all_fences_and_closed_with_original_handles(tmp_path):
    with tree(tmp_path) as (child,master,wrappers):
        assert not (tmp_path/'daemon.json').exists()
        before=(tmp_path/'opaque-outcome.json').read_bytes(); phases=[]
        with custody(tmp_path,master,wrappers,phases) as owned:
            daemon=P.observe_process(json.loads((tmp_path/'daemon.json').read_text())['pid'])
            assert daemon.parent==wrappers[2].pid
            owned.nominate_daemons([daemon])
            original_handles=dict(owned.descriptors)
            def population():
                for wrapper in wrappers:
                    if not Path('/proc',str(wrapper.pid)).exists():continue
                    for tid in Path('/proc',str(wrapper.pid),'task').iterdir():
                        for pid in P._read_proc(tid/'children').decode().split():
                            fields=P._stat(int(pid))
                            assert fields[0] in {'Z','X'} or int(pid)==daemon.pid
            result=phased.gracefully_close_native_lanes(
                controller=master,lanes=[P.LaneBinding(wrappers[2],daemon)],
                idle_supervisors=[wrappers[i] for i in [0,1,3]],launcher_custody=owned,
                lane_fence=lambda _:pytest.fail('original launch fences must transfer without reacquisition'),
                effect_gate=lambda:None,population_gate=population,population_refusal=RuntimeError,
                lane_children_gate=lambda _:None,closed_children_gate=lambda:None,
                record_phase=phases.append,timeout_seconds=3)
            assert result['controller_exited']
            assert owned.descriptors==original_handles
            assert owned.suspended==[]
            with pytest.raises(P.GracefulRecoveryUnverified,match='nomination_not_admitted'):
                owned.consume_for_closure(master,wrappers,[daemon])
        assert child.wait(timeout=3)==0
        assert (tmp_path/'opaque-outcome.json').read_bytes()==before
        assert phases.count('original_launcher_custody_consumed_once')==1
        assert phases.index('all_native_launch_fences_acquired_before_nomination') < phases.index('supervisor_suspend_prepared')

def test_persistent_fence_refuses_before_pausing_any_wrapper_and_restores_master(tmp_path):
    with tree(tmp_path,'never') as (_child,master,wrappers):
        phases=[]
        with pytest.raises(TimeoutError):
            with custody(tmp_path,master,wrappers,phases):pytest.fail('persistent fence cannot admit')
        assert all(P._stat(b.pid)[0] not in {'T','t'} for b in [master,*wrappers])
        assert 'supervisor_suspend_prepared' not in phases

def test_capture_refusal_resumes_only_original_launchers(tmp_path):
    with tree(tmp_path) as (_child,master,wrappers):
        with pytest.raises(ValueError,match='captured owner uncertainty'):
            with custody(tmp_path,master,wrappers,[]):
                raise ValueError('captured owner uncertainty')
        assert all(P._stat(b.pid)[0] not in {'T','t'} for b in [master,*wrappers])

def test_wrong_binding_cannot_consume_frozen_custody(tmp_path):
    with tree(tmp_path) as (_child,master,wrappers):
        with custody(tmp_path,master,wrappers,[]) as owned:
            with pytest.raises(P.GracefulRecoveryUnverified,match='closure_binding_changed'):
                owned.consume_for_closure(replace(master,birth=master.birth+1),wrappers,[])
            owned.require_frozen()

def test_original_daemon_handle_is_retained_and_nomination_cannot_refresh(tmp_path):
    with tree(tmp_path) as (_child,master,wrappers):
        with custody(tmp_path,master,wrappers,[]) as owned:
            daemon=P.observe_process(json.loads((tmp_path/'daemon.json').read_text())['pid'])
            owned.nominate_daemons([daemon])
            fd=owned.descriptors[daemon.pid]
            with pytest.raises(P.GracefulRecoveryUnverified,match='nomination_replayed'):
                owned.nominate_daemons([daemon])
            assert owned.descriptors[daemon.pid]==fd
            with pytest.raises(P.GracefulRecoveryUnverified,match='nomination_changed'):
                owned.consume_for_closure(master,wrappers,[])

def test_nominated_daemon_exit_refuses_without_replacement_adoption(tmp_path):
    with tree(tmp_path) as (_child,master,wrappers):
        with custody(tmp_path,master,wrappers,[]) as owned:
            daemon=P.observe_process(json.loads((tmp_path/'daemon.json').read_text())['pid'])
            owned.nominate_daemons([daemon])
            signal.pidfd_send_signal(owned.descriptors[daemon.pid],signal.SIGTERM)
            wait(lambda:P._exited(owned.descriptors[daemon.pid]))
            with pytest.raises((FileNotFoundError,P.GracefulRecoveryUnverified)):
                owned.consume_for_closure(master,wrappers,[daemon])
            assert owned.state=='frozen'

def test_lost_pause_and_foreign_pidfd_refuse_before_closure(tmp_path):
    with tree(tmp_path) as (_child,master,wrappers):
        with custody(tmp_path,master,wrappers,[]) as owned:
            signal.pidfd_send_signal(owned.descriptors[wrappers[0].pid],signal.SIGCONT)
            wait(lambda:P._stat(wrappers[0].pid)[0] not in {'T','t'})
            with pytest.raises(P.GracefulRecoveryUnverified,match='suspension_lost'):
                owned.require_frozen()
            signal.pidfd_send_signal(owned.descriptors[wrappers[0].pid],signal.SIGSTOP)
            wait(lambda:P.all_threads_stopped(wrappers[0]))
            original=owned.descriptors[master.pid]
            owned.descriptors[master.pid]=owned.descriptors[wrappers[0].pid]
            try:
                with pytest.raises(P.GracefulRecoveryUnverified,match='pidfd_(replaced|target_changed)'):
                    owned.require_frozen()
            finally:
                owned.descriptors[master.pid]=original

def test_forked_controller_cannot_consume_parent_custody(tmp_path):
    with tree(tmp_path) as (_child,master,wrappers):
        with custody(tmp_path,master,wrappers,[]) as owned:
            readfd,writefd=os.pipe()
            pid=os.fork()
            if pid==0:
                try:
                    os.close(readfd)
                    try:owned.consume_for_closure(master,wrappers,[])
                    except P.GracefulRecoveryUnverified as error:os.write(writefd,str(error).encode())
                    else:os.write(writefd,b'UNSAFE')
                finally:os._exit(0)
            os.close(writefd)
            try:assert os.read(readfd,1024)==b'launcher_custody_foreign_process'
            finally:os.close(readfd);os.waitpid(pid,0)
            owned.require_frozen()

@pytest.mark.parametrize('at',['controller_all_threads_stopped','supervisor_all_threads_stopped'])
def test_phase_audit_failure_restores_every_original_paused_launcher(tmp_path,at):
    with tree(tmp_path) as (_child,master,wrappers):
        owned=custody(tmp_path,master,wrappers,[])
        def phase(name):
            if name==at:raise OSError('fixture audit ENOSPC')
        owned.record_phase=phase
        with pytest.raises(OSError,match='ENOSPC'):owned.__enter__()
        assert owned.closed and not owned.suspended
        assert all(P._stat(b.pid)[0] not in {'T','t'} for b in [master,*wrappers])

def test_real_after_close_callback_error_restores_original_launchers_without_fd_retry(tmp_path):
    with tree(tmp_path) as (_child,master,wrappers):
        owned=custody(tmp_path,master,wrappers,[])
        ordinary=owned.lane_fence;callbacks=[]
        @contextlib.contextmanager
        def fence(index):
            try:
                with ordinary(index) as check:yield check
            finally:
                callbacks.append(index)
                if index==1:raise OSError('fixture close callback failed after owned close')
        owned.lane_fence=fence
        owned.__enter__()
        with pytest.raises(OSError,match='close callback'):owned.close()
        assert owned.closed and owned.suspended==[] and owned.normal_exit_required
        assert sorted(callbacks)==[0,1,2,3]
        owned.close()
        assert len(callbacks)==4
        assert all(P._stat(b.pid)[0] not in {'T','t'} for b in [master,*wrappers])

def test_released_native_lock_is_not_reacquired_by_verification(tmp_path):
    path=tmp_path/'native.lock'
    with retained_fence(flock(path),path) as check:
        info=path.stat()
        descriptors=[]
        for p in Path('/proc/self/fd').iterdir():
            try:
                if os.fstat(int(p.name)).st_ino==info.st_ino:descriptors.append(int(p.name))
            except OSError:pass
        assert len(descriptors)==1
        fcntl.flock(descriptors[0],fcntl.LOCK_UN)
        with pytest.raises(P.GracefulRecoveryUnverified,match='positive_lock_missing'):check()
        with flock(path):pass

def test_early_fence_release_cannot_be_consumed_as_frozen_custody(tmp_path):
    with tree(tmp_path) as (_child,master,wrappers):
        with custody(tmp_path,master,wrappers,[]) as owned:
            owned.release_fences()
            with pytest.raises(P.GracefulRecoveryUnverified,match='fences_not_retained'):
                owned.nominate_daemons([])


@pytest.mark.parametrize('value', [None, {}, object()])
def test_phased_entry_refuses_unowned_custody_before_any_signal(monkeypatch, value):
    own = P.observe_process(os.getpid())
    monkeypatch.setattr(signal, 'pidfd_send_signal', lambda *_: pytest.fail('no unowned signal'))
    with pytest.raises(P.GracefulRecoveryUnverified, match='original_launcher_custody_required'):
        phased.gracefully_close_native_lanes(
            controller=own, lanes=[], idle_supervisors=[replace(own, pid=own.pid+1, parent=own.pid)],
            launcher_custody=value, lane_fence=lambda _: None, effect_gate=lambda: None,
            population_gate=lambda: None, population_refusal=RuntimeError,
            lane_children_gate=lambda _: None, closed_children_gate=lambda: None,
            record_phase=lambda _: None)


def test_direct_phased_self_population_refuses_before_signals(monkeypatch):
    own = P.observe_process(os.getpid())
    value = object.__new__(LauncherQuiescence)
    monkeypatch.setattr(signal, 'pidfd_send_signal', lambda *_: pytest.fail('no self signal'))
    with pytest.raises(P.GracefulRecoveryUnverified, match='recovery_custodian_cannot_freeze_itself'):
        phased.gracefully_close_native_lanes(
            controller=own, lanes=[], idle_supervisors=[replace(own, pid=own.pid+1, parent=own.pid)],
            launcher_custody=value, lane_fence=lambda _: None, effect_gate=lambda: None,
            population_gate=lambda: None, population_refusal=RuntimeError,
            lane_children_gate=lambda _: None, closed_children_gate=lambda: None,
            record_phase=lambda _: None)


def test_custodian_cannot_be_nominated_as_daemon(tmp_path):
    with tree(tmp_path) as (_child, master, wrappers):
        with custody(tmp_path, master, wrappers, []) as owned:
            with pytest.raises(P.GracefulRecoveryUnverified, match='cannot_nominate_itself'):
                owned.nominate_daemons([P.observe_process(os.getpid())])
            assert owned.daemon_bindings is None
            owned.require_frozen()


@pytest.mark.parametrize('mode', ['launch', 'pipe'])
def test_unreported_child_blocks_idle_closure_without_signal_or_adoption(tmp_path, monkeypatch, mode):
    with tree(tmp_path, mode) as (_child, master, wrappers):
        with custody(tmp_path, master, wrappers, []) as owned:
            daemon = P.observe_process(json.loads((tmp_path/'daemon.json').read_text())['pid'])
            owned.nominate_daemons([])
            signals = []
            original_signal = signal.pidfd_send_signal
            def observe(fd, sig, *args):
                signals.append((fd, sig))
                return original_signal(fd, sig, *args)
            monkeypatch.setattr(signal, 'pidfd_send_signal', observe)
            with pytest.raises(P.GracefulRecoveryUnverified, match='idle_supervisor_has_live_child'):
                phased.gracefully_close_native_lanes(
                    controller=master, lanes=[], idle_supervisors=wrappers, launcher_custody=owned,
                    lane_fence=lambda _: pytest.fail('no fence reacquisition'), effect_gate=lambda: None,
                    population_gate=lambda: None, population_refusal=RuntimeError,
                    lane_children_gate=lambda _: None, closed_children_gate=lambda: None,
                    record_phase=lambda _: None, timeout_seconds=1)
            assert daemon.pid not in owned.descriptors
            assert all(sig != signal.SIGTERM for _, sig in signals)
            P.require_exact_process(daemon)
            assert P._stat(daemon.pid)[0] not in {'T', 't', 'Z', 'X'}


@pytest.mark.parametrize('refusal_at', ['population', 'stop_delivery'])
def test_daemon_cont_failure_keeps_original_handle_for_same_object_recovery(tmp_path, monkeypatch, refusal_at):
    with tree(tmp_path) as (_child, master, wrappers):
        before = (tmp_path/'opaque-outcome.json').read_bytes()
        phases = []
        owned = custody(tmp_path, master, wrappers, phases)
        owned.__enter__()
        daemon = P.observe_process(json.loads((tmp_path/'daemon.json').read_text())['pid'])
        owned.nominate_daemons([daemon])
        daemon_fd = owned.descriptors[daemon.pid]
        original_signal = signal.pidfd_send_signal
        blocked = True
        signals = []

        def failure(fd, sig, *args):
            signals.append((fd, sig))
            if fd == daemon_fd and sig == signal.SIGCONT and blocked:
                raise OSError('private original daemon CONT EIO')
            result = original_signal(fd, sig, *args)
            if fd == daemon_fd and sig == signal.SIGSTOP and refusal_at == 'stop_delivery':
                raise OSError('private STOP delivered before observation EIO')
            return result

        def population():
            raise ValueError('private pre-terminal census refusal')

        monkeypatch.setattr(signal, 'pidfd_send_signal', failure)
        # This is the caller's independent native guard, deliberately retained
        # until its original actor cleanup completes; no native adapter exists.
        path = tmp_path/'caller-native-guard.lock'
        try:
            with retained_fence(flock(path), path) as guard:
                with pytest.raises(P.GracefulRecoveryUnverified, match='actor_resume_unverified'):
                    phased.gracefully_close_native_lanes(
                        controller=master, lanes=[P.LaneBinding(wrappers[2], daemon)],
                        idle_supervisors=[wrappers[i] for i in [0, 1, 3]], launcher_custody=owned,
                        lane_fence=lambda _: pytest.fail('original fences cannot reacquire'),
                        effect_gate=lambda: None, population_gate=population, population_refusal=RuntimeError,
                        lane_children_gate=lambda _: None, closed_children_gate=lambda: None,
                        record_phase=phases.append, timeout_seconds=2)
                wait(lambda: P.all_threads_stopped(daemon))
                assert owned.suspended == [daemon]
                assert owned.descriptors[daemon.pid] == daemon_fd
                with pytest.raises(P.GracefulRecoveryUnverified, match='original_resume_failed'):
                    owned.close()
                assert not owned.closed and owned.suspended == [daemon]
                owned.require_owner()
                guard()
                competitor = os.open(path, os.O_RDWR)
                try:
                    with pytest.raises(BlockingIOError):
                        fcntl.flock(competitor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                finally:
                    os.close(competitor)
                assert all(sig != signal.SIGTERM for _, sig in signals)
                assert 'daemon_graceful_exit_prepared' not in phases
                assert phases.count('original_launcher_custody_consumed_once') == 1
                with pytest.raises(P.GracefulRecoveryUnverified, match='nomination_not_admitted'):
                    owned.consume_for_closure(master, wrappers, [daemon])
                blocked = False
                owned.close()
                assert owned.closed and not owned.suspended
                wait(lambda: P._stat(daemon.pid)[0] not in {'T', 't'})
            with flock(path):
                pass
            assert (tmp_path/'opaque-outcome.json').read_bytes() == before
        finally:
            monkeypatch.setattr(signal, 'pidfd_send_signal', original_signal)
            if not owned.closed:
                owned.close()


def test_runtime_import_has_no_process_signal_or_optional_stack_effects(tmp_path):
    root = Path(__file__).resolve().parents[2]
    code = '''
import os,signal,subprocess,sys
def forbidden(*args, **kwargs): raise AssertionError("import performed process effect")
os.pidfd_open=forbidden
signal.pidfd_send_signal=forbidden
subprocess.Popen=forbidden
from ipfs_accelerate_py.agent_supervisor.runtime import native_launcher_quiescence, native_phased_graceful_recovery, native_process_custody
assert not any(name.startswith(("duckdb", "torch", "transformers")) for name in sys.modules)
print(native_launcher_quiescence.__file__)
'''
    result = subprocess.run([sys.executable, '-B', '-c', code], cwd=tmp_path,
                            env={**os.environ, 'PYTHONPATH': str(root), 'PYTHONDONTWRITEBYTECODE': '1'},
                            capture_output=True, text=True, timeout=10, check=True)
    assert result.stdout.strip() == str(root/'ipfs_accelerate_py/agent_supervisor/runtime/native_launcher_quiescence.py')
