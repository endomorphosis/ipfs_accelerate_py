"""Real diagnostic threads/subprocesses; no native admission or model work."""
import io
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import threading
import time

import pytest

import ipfs_accelerate_py.agent_supervisor.todo_daemon.native_owner_startup_watchdog as subject


def private_logger():
    stream = io.StringIO()
    logger = logging.Logger('private-startup-watchdog-regression', logging.INFO)
    logger.addHandler(logging.StreamHandler(stream))
    return logger, stream


def test_default_cadence_stop_before_interval_proves_thread_closure():
    logger, stream = private_logger()
    watchdog = subject.NativeOwnerStartupWatchdog(logger=logger)
    assert watchdog.interval_seconds == 10.0
    watchdog.start()
    watchdog.close()
    watchdog.close()
    assert not watchdog.is_alive and stream.getvalue() == ''


def test_multiple_real_samples_stop_and_no_poststop_output():
    logger, stream = private_logger()
    watchdog = subject.NativeOwnerStartupWatchdog(logger=logger, interval_seconds=.01)
    try:
        watchdog.start()
        deadline = time.monotonic() + 2
        while stream.getvalue().count('Native owner startup stack metadata: ') < 3:
            assert time.monotonic() < deadline
            time.sleep(.005)
    finally:
        watchdog.close()
    before = stream.getvalue()
    time.sleep(.04)
    assert not watchdog.is_alive and stream.getvalue() == before
    for line in before.splitlines():
        payload = line.split('Native owner startup stack metadata: ', 1)[1]
        assert len(payload.encode('ascii')) <= subject.MAX_METADATA_BYTES
        assert json.loads(payload)['schema'] == 'native-startup-stack-metadata/v1'


def test_context_early_return_stops_actual_thread():
    logger, stream = private_logger()
    watchdog = subject.NativeOwnerStartupWatchdog(logger=logger)
    def early_return():
        with watchdog:
            return 37
    assert early_return() == 37
    assert not watchdog.is_alive and stream.getvalue() == ''


def test_real_frame_metadata_is_bounded_and_does_not_read_local_values():
    local_secret = 'do-not-serialize-private-local-value-4fce'
    stop = threading.Event()
    children = [threading.Thread(target=stop.wait) for _ in range(subject.MAX_THREADS + 2)]
    try:
        for child in children:
            child.start()
        value = subject.startup_stack_metadata()
        raw = json.dumps(value, ensure_ascii=True, separators=(',', ':'))
        assert local_secret not in raw
        assert len(raw.encode('ascii')) <= subject.MAX_METADATA_BYTES
        assert len(value['threads']) <= subject.MAX_THREADS and value['threads_truncated']
        assert value['threads'][0]['thread_id'] == threading.main_thread().ident
        for row in value['threads']:
            assert set(row) == {'thread_id', 'frames', 'frames_truncated'}
            assert len(row['frames']) <= subject.MAX_FRAMES_PER_THREAD
            for frame in row['frames']:
                assert set(frame) == {'filename', 'function', 'line'}
                assert len(frame['filename']) <= 192 and len(frame['function']) <= 96
    finally:
        stop.set()
        for child in children:
            if child.ident is not None:
                child.join(timeout=2)
                assert not child.is_alive()


@pytest.mark.parametrize('generation', [1, 2])
def test_real_subprocess_busy_path_and_waiting_threads_survive_multiple_cadences(generation):
    script = textwrap.dedent('''
        import io, json, logging, pathlib, threading, time
        from ipfs_accelerate_py.agent_supervisor.todo_daemon.native_owner_startup_watchdog import NativeOwnerStartupWatchdog
        stream=io.StringIO()
        logger=logging.Logger('real-child-startup', logging.INFO)
        logger.addHandler(logging.StreamHandler(stream))
        stop=threading.Event()
        children=[threading.Thread(target=stop.wait) for _ in range(2)]
        watchdog=NativeOwnerStartupWatchdog(logger=logger, interval_seconds=.01)
        try:
            for child in children: child.start()
            watchdog.start()
            until=time.monotonic()+1.2
            while time.monotonic()<until:
                str(pathlib.Path('/tmp')/'ordinary-startup-diagnostic-path')
        finally:
            watchdog.close()
            stop.set()
            for child in children:
                child.join(timeout=2)
                assert not child.is_alive()
        count=stream.getvalue().count('Native owner startup stack metadata: ')
        assert count>=3 and not watchdog.is_alive
        before=stream.getvalue()
        time.sleep(.04)
        assert stream.getvalue()==before
        print(json.dumps(dict(samples=count, watchdog_stopped=True, waiters_stopped=True)))
    ''')
    child = subprocess.Popen([sys.executable, '-B', '-c', script], stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=dict(os.environ))
    try:
        stdout, stderr = child.communicate(timeout=20)
    finally:
        if child.poll() is None:
            child.kill()
            child.communicate(timeout=5)
    assert child.returncode == 0, stderr.decode('utf-8', 'replace')[-4096:]
    assert len(stdout) < 4096 and len(stderr) < 65536
    assert json.loads(stdout) == dict(samples=json.loads(stdout)['samples'],
                                    watchdog_stopped=True, waiters_stopped=True)

@pytest.mark.parametrize('descriptor', [999999, -1], ids=['unavailable-inherited-fd', 'missing-inherited-fd'])
def test_real_main_bootstrap_refusal_closes_started_watchdog_without_database(tmp_path, descriptor):
    """Run real parser/main/bootstrap in a child; grant no authority or socket."""
    private = tmp_path / 'real-main-refusal'
    private.mkdir(mode=0o700)
    # Do not forward admitted credentials, model/provider tokens or unrelated
    # application config. Keep the normal package roots selected by the owner.
    environment = {key: os.environ[key] for key in
                   ('PATH', 'HOME', 'LANG', 'LC_ALL', 'TZ', 'USER', 'LOGNAME', 'PYTHONPATH')
                   if key in os.environ}
    source_root = str(Path(subject.__file__).resolve(strict=True).parents[3])
    roots = [source_root, *environment.get('PYTHONPATH', '').split(os.pathsep)]
    environment['PYTHONPATH'] = os.pathsep.join(dict.fromkeys(value for value in roots if value))
    environment.update(PYTHONDONTWRITEBYTECODE='1',
                       IPFS_DATASETS_PY_MINIMAL_IMPORTS='1', IPFS_DATASETS_AUTO_INSTALL='0',
                       IPFS_AUTO_INSTALL='false', IPFS_ACCEL_SKIP_CORE='1',
                       IPFS_DATASETS_PY_LAZY_INSTALL_ERGOAI='0', IPFS_TEST_PROOF_REUSE_MODE='off',
                       GIT_NO_LAZY_FETCH='1', GIT_OPTIONAL_LOCKS='0',
                       OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                       HF_HUB_OFFLINE='1',
                       TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1',
                       TOKENIZERS_PARALLELISM='false', CUDA_VISIBLE_DEVICES='',
                       IPFS_ACCELERATE_PY_TASK_P2P_BOOTSTRAP_PEERS='0',
                       IPFS_ACCELERATE_PY_TASK_P2P_DHT='0',
                       IPFS_ACCELERATE_PY_TASK_P2P_RENDEZVOUS='0',
                       TMPDIR=str(private), XDG_CACHE_HOME=str(private/'cache'))
    script = textwrap.dedent(r"""
        import errno, json, logging, os, pathlib, sys, threading
        from ipfs_accelerate_py.agent_supervisor.todo_daemon import implementation_daemon as daemon
        from ipfs_accelerate_py.agent_supervisor.runtime.process_security import state_authority_credentials_present
        from ipfs_accelerate_py.agent_supervisor.task_sources.state_owner_bootstrap import StateOwnerBootstrapError
        from ipfs_accelerate_py.agent_supervisor.todo_daemon import native_owner_startup_watchdog as watchdog_module
        private=pathlib.Path(sys.argv[1]); descriptor=int(sys.argv[2])
        assert pathlib.Path(daemon.__file__).resolve().parents[3] == pathlib.Path(watchdog_module.__file__).resolve().parents[3]
        assert not state_authority_credentials_present()
        if descriptor >= 0:
            try: os.fstat(descriptor)
            except OSError as error: assert error.errno == errno.EBADF
            else: raise AssertionError('fixture descriptor unexpectedly open')
        before_threads=set(threading.enumerate())
        assert not any(row.name=='native-startup-metadata' for row in before_threads)
        observed=[]; stages=[]
        class Observer(logging.Handler):
            def emit(self, record):
                message=record.getMessage()
                if message.startswith('Native owner startup stage: '): stages.append(message)
                if message=='Native owner startup stage: main entered':
                    actual=[row for row in threading.enumerate() if row.name=='native-startup-metadata']
                    assert len(actual)==1 and actual[0].is_alive()
                    observed.append(actual[0])
        observer=Observer(); old_level=daemon.logger.level
        daemon.logger.addHandler(observer); daemon.logger.setLevel(logging.INFO)
        database=private/'must-not-be-created.duckdb'; state=private/'must-not-be-created-state'
        argv=['--authority-mode','quack','--task-source-kind','duckdb',
              '--database-path',str(database),'--state-dir',str(state),
              '--state-owner-client-id','portable-watchdog-no-grants',
              '--state-owner-bootstrap-fd',str(descriptor),
              '--endpoint-secret-handle','env://PORTABLE_WATCHDOG_UNUSED',
              '--quack-endpoint','quack:127.0.0.1:1',
              '--state-store-id','portable-watchdog-no-grants',
              '--state-store-generation','portable-watchdog-generation',
              '--state-schema-revision','portable-watchdog-schema', '--once']
        caught=None
        try:
            try: daemon.main(argv)
            except BaseException as error: caught=error
        finally:
            daemon.logger.removeHandler(observer); daemon.logger.setLevel(old_level)
        if descriptor == -1:
            assert type(caught) is ValueError
            assert str(caught)=='native owner bootstrap requires its descriptor, client and Quack program'
            assert observed==[] and stages==[]
        else:
            assert type(caught) is StateOwnerBootstrapError
            assert str(caught)=='state-owner bootstrap descriptor is unavailable'
            assert isinstance(caught.__cause__, OSError) and caught.__cause__.errno==errno.EBADF
            assert len(observed)==1 and not observed[0].is_alive()
            assert stages==['Native owner startup stage: main entered',
                            'Native owner startup stage: bootstrap starting']
        assert not any(row.name=='native-startup-metadata' for row in threading.enumerate())
        assert not database.exists() and not state.exists()
        assert not list(private.rglob('*.duckdb')) and not list(private.rglob('*.wal'))
        assert not state_authority_credentials_present()
        print(json.dumps(dict(descriptor=descriptor, actual_main_refused=True,
            observed_started_threads=len(observed), started_thread_stopped=not any(t.is_alive() for t in observed),
            database_and_state_absent=True)))
    """)
    child = subprocess.Popen([sys.executable, '-B', '-c', script, str(private), str(descriptor)],
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env=environment, cwd=private, start_new_session=True, close_fds=True)
    try:
        stdout, stderr = child.communicate(timeout=60)
    finally:
        if child.poll() is None:
            try:
                os.killpg(child.pid, 9)
            except ProcessLookupError:
                pass
            child.communicate(timeout=5)
    assert child.returncode == 0, stderr.decode('utf-8', 'replace')[-8192:]
    assert len(stdout) < 16384 and len(stderr) < 262144
    # Genuine normal imports may print package notices before the final record.
    observed = json.loads(stdout.decode('utf-8').splitlines()[-1])
    assert observed == dict(descriptor=descriptor, actual_main_refused=True,
        observed_started_threads=int(descriptor != -1), started_thread_stopped=True,
        database_and_state_absent=True)
