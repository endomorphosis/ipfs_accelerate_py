"""A caught signal exception must not restart a database daemon pass."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    implementation_daemon as runtime,
)


def _native_daemon(tmp_path):
    return runtime.DatabaseImplementationDaemon(
        database_path=tmp_path / "control.duckdb",
        coordination_path=tmp_path / "coordination.duckdb",
        execution_path=tmp_path / "execution.duckdb",
        authority_mode="embedded",
        task_source_kind="duckdb",
    ).open()


def _main_args(tmp_path):
    return [
        "--authority-mode",
        "embedded",
        "--task-source-kind",
        "duckdb",
        "--database-path",
        str(tmp_path / "control.duckdb"),
        "--state-dir",
        str(tmp_path / "state"),
        "--interval",
        "300",
    ]


@pytest.fixture
def native_main(tmp_path, monkeypatch):
    pytest.importorskip("duckdb")
    daemon = _native_daemon(tmp_path)
    calls = []
    original_close = daemon.close_event_runtime

    def close():
        calls.append("close")
        original_close()

    monkeypatch.setattr(daemon, "close_event_runtime", close)
    monkeypatch.setattr(
        runtime, "DatabaseImplementationDaemon", lambda **kwargs: daemon
    )
    monkeypatch.setattr(
        runtime, "database_program_from_daemon_namespace", lambda args: None
    )
    monkeypatch.setattr(
        runtime, "bind_database_portal_execution_from_args", lambda *a, **kw: None
    )
    previous = {signal.SIGTERM: object(), signal.SIGINT: object()}
    handlers = dict(previous)

    def install(signum, handler):
        old = handlers[signum]
        handlers[signum] = handler
        return old

    monkeypatch.setattr(runtime.signal, "signal", install)
    try:
        yield daemon, calls, handlers, previous, _main_args(tmp_path)
    finally:
        if not daemon._closed:
            original_close()


def _assert_native_closed(daemon, calls, handlers, previous):
    assert calls.count("close") == 1
    assert daemon._closed
    assert daemon._connection is None
    assert daemon._coordinator is None
    assert daemon._task_source is None
    assert daemon._embedded_writer_lock_handle is None
    assert handlers == previous


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGINT])
@pytest.mark.parametrize("phase", ["run_once", "log", "timeout", "wait"])
def test_caught_signal_remains_latched_through_native_cleanup(
    native_main, monkeypatch, signum, phase
):
    daemon, calls, handlers, previous, args = native_main

    def swallowed():
        calls.append("signal")
        try:
            handlers[signum](signum, None)
        except SystemExit as exc:
            assert exc.code == 128 + signum
            calls.append("caught")

    def run_once():
        calls.append("pass")
        if calls.count("pass") != 1:
            # Main retries ordinary pass errors. Bound the negative test with
            # a distinct terminal code, which cannot satisfy the signal check.
            raise SystemExit(99)
        if phase == "run_once":
            swallowed()
        return {"reason": "no_ready_tasks", "implemented": False}

    def log(*a, **kw):
        calls.append("log")
        if phase == "log":
            swallowed()

    def timeout(*a, **kw):
        calls.append("timeout")
        if phase == "timeout":
            swallowed()
        return 300

    def wait(**kw):
        calls.append("wait")
        assert phase == "wait", "entered idle wait after a caught stop request"
        swallowed()

    monkeypatch.setattr(daemon, "run_once", run_once)
    monkeypatch.setattr(daemon, "wait_for_wake", wait)
    monkeypatch.setattr(runtime, "log_daemon_pass_result", log)
    monkeypatch.setattr(runtime, "bounded_daemon_wait_timeout", timeout)
    with pytest.raises(SystemExit) as stopped:
        runtime.main(args)
    assert stopped.value.code == 128 + signum
    assert calls.count("pass") == 1
    if phase == "run_once":
        assert "log" not in calls
    _assert_native_closed(daemon, calls, handlers, previous)


@pytest.mark.parametrize("idle_method", ["wake", "sleep"])
def test_idle_sigterm_unwinds_and_closes_once(native_main, monkeypatch, idle_method):
    daemon, calls, handlers, previous, args = native_main
    monkeypatch.setattr(
        daemon, "run_once", lambda: calls.append("pass") or {"reason": "no_ready_tasks"}
    )

    def wait(*a, **kw):
        calls.append("wait")
        handlers[signal.SIGTERM](signal.SIGTERM, None)

    if idle_method == "wake":
        monkeypatch.setattr(daemon, "wait_for_wake", wait)
    else:
        monkeypatch.setattr(daemon, "wait_for_wake", None)
        monkeypatch.setattr(runtime.time, "sleep", wait)
    with pytest.raises(SystemExit) as stopped:
        runtime.main(args)
    assert stopped.value.code == 128 + signal.SIGTERM
    assert calls.count("pass") == 1
    _assert_native_closed(daemon, calls, handlers, previous)


def test_first_signal_keeps_exit_identity_when_both_are_caught(
    native_main, monkeypatch
):
    daemon, calls, handlers, previous, args = native_main

    def run_once():
        for signum in (signal.SIGTERM, signal.SIGINT):
            try:
                handlers[signum](signum, None)
            except SystemExit as exc:
                assert exc.code == 128 + signal.SIGTERM
        return {"reason": "no_ready_tasks"}

    monkeypatch.setattr(daemon, "run_once", run_once)
    with pytest.raises(SystemExit) as stopped:
        runtime.main(args + ["--once"])
    assert stopped.value.code == 128 + signal.SIGTERM
    _assert_native_closed(daemon, calls, handlers, previous)


@pytest.mark.parametrize(
    "failure", [RuntimeError("pass failed"), KeyboardInterrupt(), SystemExit(19)]
)
def test_non_signal_failures_keep_original_exception_and_cleanup(
    native_main, monkeypatch, failure
):
    daemon, calls, handlers, previous, args = native_main

    def run_once():
        raise failure

    monkeypatch.setattr(daemon, "run_once", run_once)
    with pytest.raises(type(failure)) as caught:
        runtime.main(args + ["--once"])
    assert caught.value is failure
    _assert_native_closed(daemon, calls, handlers, previous)


def test_ordinary_once_still_closes_without_signal(native_main, monkeypatch):
    daemon, calls, handlers, previous, args = native_main
    monkeypatch.setattr(
        daemon, "run_once", lambda: calls.append("pass") or {"reason": "no_ready_tasks"}
    )
    runtime.main(args + ["--once"])
    assert calls.count("pass") == 1
    _assert_native_closed(daemon, calls, handlers, previous)


@pytest.mark.parametrize("once", [False, True])
def test_latched_stop_preempts_translated_retryable_pass_error(
    native_main, monkeypatch, once
):
    daemon, calls, handlers, previous, args = native_main

    def run_once():
        calls.append("pass")
        if calls.count("pass") != 1:
            raise SystemExit(99)
        try:
            handlers[signal.SIGTERM](signal.SIGTERM, None)
        except SystemExit:
            raise RuntimeError("FatalException: interrupted native query") from None

    monkeypatch.setattr(daemon, "run_once", run_once)
    monkeypatch.setattr(daemon, "wait_for_wake", lambda **kwargs: None)
    with pytest.raises(SystemExit) as stopped:
        runtime.main(args + (["--once"] if once else []))
    assert stopped.value.code == 128 + signal.SIGTERM
    assert calls.count("pass") == 1
    _assert_native_closed(daemon, calls, handlers, previous)


def test_latched_stop_after_compatibility_projection_prevents_idle_wait(
    native_main, monkeypatch
):
    daemon, calls, handlers, previous, args = native_main
    monkeypatch.setattr(daemon, "run_once", lambda: {"reason": "no_ready_tasks"})

    def project(*args, **kwargs):
        try:
            handlers[signal.SIGTERM](signal.SIGTERM, None)
        except SystemExit:
            pass

    monkeypatch.setattr(runtime, "materialize_database_task_state_compatibility_projection", project)
    monkeypatch.setattr(daemon, "wait_for_wake", lambda **kwargs: (_ for _ in ()).throw(SystemExit(99)))
    with pytest.raises(SystemExit) as stopped:
        runtime.main(args)
    assert stopped.value.code == 128 + signal.SIGTERM
    _assert_native_closed(daemon, calls, handlers, previous)


def test_latched_stop_before_loop_does_not_start_first_pass(native_main, monkeypatch):
    daemon, calls, handlers, previous, args = native_main
    install = runtime.signal.signal

    def install_and_catch(signum, handler):
        old = install(signum, handler)
        if signum == signal.SIGTERM and handler is not previous[signum]:
            try:
                handler(signum, None)
            except SystemExit:
                pass
        return old

    monkeypatch.setattr(runtime.signal, "signal", install_and_catch)
    monkeypatch.setattr(
        daemon, "run_once", lambda: (_ for _ in ()).throw(SystemExit(99))
    )
    with pytest.raises(SystemExit) as stopped:
        runtime.main(args)
    assert stopped.value.code == 128 + signal.SIGTERM
    _assert_native_closed(daemon, calls, handlers, previous)


def test_cleanup_failure_restores_handlers_without_retrying_close(
    native_main, monkeypatch
):
    daemon, calls, handlers, previous, args = native_main
    close = daemon.close_event_runtime
    failure = RuntimeError("native close response lost")

    def fail_after_native_close():
        close()
        raise failure

    def stopped_pass():
        try:
            handlers[signal.SIGTERM](signal.SIGTERM, None)
        except SystemExit:
            pass
        return {"reason": "no_ready_tasks"}

    monkeypatch.setattr(daemon, "run_once", stopped_pass)
    monkeypatch.setattr(daemon, "close_event_runtime", fail_after_native_close)
    with pytest.raises(RuntimeError) as caught:
        runtime.main(args + ["--once"])
    assert caught.value is failure
    _assert_native_closed(daemon, calls, handlers, previous)


def _child_signal_run(path, phase):
    path = Path(path)
    daemon = _native_daemon(path)
    native_close = daemon.close_event_runtime
    calls = []
    original_handlers = {
        s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)
    }

    def close():
        calls.append("close")
        native_close()

    def run_once():
        calls.append("pass")
        if calls.count("pass") != 1:
            raise SystemExit(99)
        if phase == "run_once":
            try:
                os.kill(os.getpid(), signal.SIGTERM)
            except SystemExit:
                calls.append("caught_system_exit")
        return {"reason": "no_ready_tasks", "implemented": False}

    def wait(**kw):
        calls.append("wait")
        os.kill(os.getpid(), signal.SIGTERM)

    runtime.DatabaseImplementationDaemon = lambda **kwargs: daemon
    runtime.database_program_from_daemon_namespace = lambda args: None
    runtime.bind_database_portal_execution_from_args = lambda *a, **kw: None
    daemon.run_once = run_once
    daemon.wait_for_wake = wait
    daemon.close_event_runtime = close
    try:
        runtime.main(_main_args(path))
    except SystemExit as exc:
        result = {
            "exit_code": exc.code,
            "calls": calls,
            "closed": daemon._closed,
            "connection_detached": daemon._connection is None,
            "writer_released": daemon._embedded_writer_lock_handle is None,
            "handlers_restored": all(
                signal.getsignal(s) is h for s, h in original_handlers.items()
            ),
        }
        (path / "result.json").write_text(json.dumps(result))
        raise
    finally:
        if not daemon._closed:
            native_close()


@pytest.mark.parametrize("phase", ["run_once", "idle"])
def test_actual_disposable_process_sigterm_and_native_close(tmp_path, phase):
    pytest.importorskip("duckdb")
    code = "import runpy,sys; runpy.run_path(sys.argv[1])['_child_signal_run'](sys.argv[2], sys.argv[3])"
    env = {k: v for k, v in os.environ.items() if not k.startswith("IPFS_ACCELERATE_")}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = str(Path(runtime.__file__).resolve().parents[3])
    result = subprocess.run(
        [sys.executable, "-B", "-c", code, __file__, str(tmp_path), phase],
        env=env,
        text=True,
        capture_output=True,
        timeout=45,
        check=False,
    )
    assert result.returncode == 128 + signal.SIGTERM, result.stderr
    observed = json.loads((tmp_path / "result.json").read_text())
    assert observed["calls"].count("pass") == 1
    assert observed["calls"].count("close") == 1
    assert observed["closed"] and observed["connection_detached"]
    assert observed["writer_released"] and observed["handlers_restored"]
    if phase == "run_once":
        assert "caught_system_exit" in observed["calls"]
        assert "wait" not in observed["calls"]
