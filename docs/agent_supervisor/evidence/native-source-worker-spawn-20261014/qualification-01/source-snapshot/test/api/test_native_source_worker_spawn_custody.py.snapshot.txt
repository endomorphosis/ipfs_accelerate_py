"""Exact process custody at the optional native source worker handoff.

Native claim/Portal/lifecycle, Git/CAS and inert direct Python process birth are
real. Quack listener and deployment metadata remain authored host seams. The
same-UID inert child reads one stdin byte; it executes no source or model and
does not establish separate worker-UID access or a Terminal coding deployment.
"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from copy import deepcopy
import os
import select
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import source_lookup_worker_client as protocol
from test.api.test_current_source_task_admission import _update_body
from test.api.test_current_source_worktree_allocation import _allocation_case
from test.api.test_native_nominated_source_allocation_custody import bind, source_files, source_session


@pytest.fixture
def native_spawn(tmp_path, monkeypatch):
    with _allocation_case(tmp_path, monkeypatch, defer_source_binding=True,
                          source_files=source_files()) as case:
        bind(case)
        with source_session(case) as session:
            yield case, session


@contextmanager
def inert_child(workspace, *, process_type=subprocess.Popen):
    code = "import sys; print('inert-ready',flush=True); sys.stdin.buffer.read(1)"
    process = process_type([sys.executable, "-I", "-B", "-c", code], cwd=workspace,
        env={"PATH": os.defpath, "PYTHONDONTWRITEBYTECODE": "1"},
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        assert process.stdout is not None
        assert select.select([process.stdout], [], [], 10)[0], "inert child did not become ready"
        assert process.stdout.readline() == b"inert-ready\n"
        assert process.poll() is None
        yield process
    finally:
        if process.poll() is None:
            process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None:
                stream.close()


def assert_retired_after_birth(session, process):
    assert session._process is process
    assert not session._registered.is_set()
    assert session._closed and session._grant_revoked
    assert process.poll() is None
    # Retirement leaves termination and reap with the actual spawning caller.
    # It does not kill an unrelated process or replace a consumed registration.
    with pytest.raises((ValueError, RuntimeError)):
        session.require_before_spawn()
    with pytest.raises((ValueError, RuntimeError)):
        session.note_spawned(process)


def callback_on_first_cas(case, monkeypatch, action):
    original = case.native.retained.cas.get
    seen = []

    def callback(cid):
        value = original(cid)
        if not seen:
            seen.append(cid)
            action()
        return value

    monkeypatch.setattr(case.native.retained.cas, "get", callback)
    return seen


def callback_on_first_clock(case, monkeypatch, action):
    original = type(case.native.clock).__call__
    seen = []

    def callback(clock):
        value = original(clock)
        if clock is case.native.clock and not seen:
            seen.append(True)
            action()
        return value

    monkeypatch.setattr(type(case.native.clock), "__call__", callback)
    return seen


@pytest.mark.parametrize("callback", ["cas", "clock"])
def test_prepare_callbacks_cannot_reenter_preparation_or_register_a_duck_process(
        native_spawn, monkeypatch, callback):
    case, session = native_spawn

    def reenter():
        with pytest.raises((ValueError, RuntimeError)):
            session.require_before_spawn()
        fake = SimpleNamespace(pid=os.getpid(), poll=lambda: None)
        with pytest.raises((ValueError, RuntimeError)):
            session.note_spawned(fake)

    seen = (callback_on_first_cas(case, monkeypatch, reenter) if callback == "cas"
            else callback_on_first_clock(case, monkeypatch, reenter))
    assert session.require_before_spawn() is None
    assert len(seen) == 1
    assert not session._registered.is_set() and session._process is None
    assert not session._closed
    with inert_child(case.native.workspace) as process:
        session.note_spawned(process)
        assert session._registered.is_set() and session._process is process


@pytest.mark.parametrize("callback", ["cas", "clock"])
def test_registration_callbacks_cannot_reenter_either_spawn_stage(native_spawn, monkeypatch, callback):
    case, session = native_spawn
    session.require_before_spawn()
    with inert_child(case.native.workspace) as process:

        def reenter():
            with pytest.raises((ValueError, RuntimeError)):
                session.require_before_spawn()
            with pytest.raises((ValueError, RuntimeError)):
                session.note_spawned(process)

        seen = (callback_on_first_cas(case, monkeypatch, reenter) if callback == "cas"
                else callback_on_first_clock(case, monkeypatch, reenter))
        session.note_spawned(process)
        assert len(seen) == 1
        assert session._registered.is_set() and session._process is process
        assert not session._closed


@pytest.mark.parametrize("replacement", ["equal_tuple", "serialized", "object"])
def test_detached_preparation_representation_cannot_authorize_actual_child(native_spawn, replacement):
    case, session = native_spawn
    session.require_before_spawn()
    original = session._spawn_prepared
    if replacement == "equal_tuple":
        substituted = tuple(list(original))
        assert substituted is not original
    elif replacement == "serialized":
        substituted = {"prepared": True, "allocation": case.guard.require_current(),
                       "scope": case.guard.require_nominated_source_scope()}
    else:
        substituted = object()
    session._spawn_prepared = substituted
    with inert_child(case.native.workspace) as process:
        with pytest.raises((ValueError, RuntimeError)):
            session.note_spawned(process)
        assert_retired_after_birth(session, process)


@pytest.mark.parametrize("tamper", ["descriptor_bytes", "socket_mode"])
def test_endpoint_tampering_after_preparation_prevents_child_registration(native_spawn, tamper):
    case, session = native_spawn
    session.require_before_spawn()
    if tamper == "descriptor_bytes":
        raw = session.descriptor_path.read_bytes()
        session.descriptor_path.chmod(0o600)
        session.descriptor_path.write_bytes(raw + b"\n")
        session.descriptor_path.chmod(0o440)
    else:
        session.socket_path.chmod(0o600)
    with inert_child(case.native.workspace) as process:
        with pytest.raises((ValueError, RuntimeError)):
            session.note_spawned(process)
        assert_retired_after_birth(session, process)


@pytest.mark.parametrize("change", ["selected", "omitted", "task_body", "daemon_grant", "claim_lease"])
def test_cas_callback_custody_loss_after_actual_birth_retires_proxy_before_registration(
        native_spawn, monkeypatch, change):
    case, session = native_spawn
    session.require_before_spawn()

    def mutate():
        if change in {"selected", "omitted"}:
            name = "module.py" if change == "selected" else "unselected.py"
            path = case.native.workspace / name
            path.write_bytes(path.read_bytes() + b"\n# changed during birth replay\n")
        elif change == "task_body":
            body = deepcopy(dict(case.native.native_task.body))
            body["title"] = "changed during actual child replay"
            _update_body(case.native, body)
        elif change == "daemon_grant":
            case.native.server._command_gateway.revoke_grant(case.native.daemon_grant_id)
        else:
            claim = case.native.daemon.coordinator.get_task_claim(case.native.attempt.claim_id)
            lease = case.native.daemon.coordinator.protect_task_claim(claim)
            case.native.daemon.coordinator.release(lease, reason="authored inert child custody loss")

    seen = callback_on_first_cas(case, monkeypatch, mutate)
    with inert_child(case.native.workspace) as process:
        with pytest.raises((ValueError, RuntimeError)):
            session.note_spawned(process)
        assert len(seen) == 1
        assert_retired_after_birth(session, process)


def test_source_edit_at_closing_native_clock_cannot_escape_birth_replay(native_spawn, monkeypatch):
    case, session = native_spawn
    session.require_before_spawn()
    seen = callback_on_first_cas(case, monkeypatch,
        lambda: setattr(case.native.clock, "reads_until_edit", 1))
    with inert_child(case.native.workspace) as process:
        with pytest.raises((ValueError, RuntimeError)):
            session.note_spawned(process)
        assert len(seen) == 1 and case.native.clock.edited
        assert_retired_after_birth(session, process)


def concurrent_stage(action):
    barrier = threading.Barrier(2)

    def invoke():
        barrier.wait(timeout=10)
        try:
            action()
            return "accepted"
        except (ValueError, RuntimeError):
            return "refused"

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(invoke) for _ in range(2)]
        return sorted(future.result(timeout=20) for future in futures)


def test_concurrent_duplicate_preparation_has_one_original_winner(native_spawn):
    case, session = native_spawn
    assert concurrent_stage(session.require_before_spawn) == ["accepted", "refused"]
    assert not session._closed and not session._registered.is_set()
    assert session._spawn_prepared is session._original_spawn_prepared
    with inert_child(case.native.workspace) as process:
        session.note_spawned(process)
        assert session._registered.is_set() and session._process is process


def test_concurrent_duplicate_registration_keeps_the_first_actual_process(native_spawn):
    case, session = native_spawn
    session.require_before_spawn()
    with inert_child(case.native.workspace) as process:
        assert concurrent_stage(lambda: session.note_spawned(process)) == ["accepted", "refused"]
        assert session._registered.is_set() and session._process is process
        assert session._child == protocol.process_identity(process.pid)
        assert not session._closed


@pytest.mark.parametrize("representation", ["duck", "unconstructed", "subclass"])
def test_malformed_process_representation_never_becomes_registered_child(native_spawn, representation):
    case, session = native_spawn
    session.require_before_spawn()
    if representation == "subclass":
        class DerivedPopen(subprocess.Popen):
            pass

        with inert_child(case.native.workspace, process_type=DerivedPopen) as process:
            with pytest.raises((ValueError, RuntimeError)):
                session.note_spawned(process)
            assert not session._registered.is_set()
    else:
        process = (SimpleNamespace(pid=os.getpid(), poll=lambda: None) if representation == "duck"
                   else object.__new__(subprocess.Popen))
        with pytest.raises((ValueError, RuntimeError)):
            session.note_spawned(process)
        assert not session._registered.is_set()
    assert session._child is None


def test_child_birth_recapture_substitution_refuses_without_losing_caller_cleanup(native_spawn, monkeypatch):
    case, session = native_spawn
    session.require_before_spawn()
    with inert_child(case.native.workspace) as process:
        original = protocol.process_identity
        reads = []

        def changed_identity(pid):
            identity = original(pid)
            if pid == process.pid:
                reads.append(pid)
                if len(reads) == 2:
                    identity = {**identity, "start_time_ticks": identity["start_time_ticks"] + 1}
            return identity

        monkeypatch.setattr(protocol, "process_identity", changed_identity)
        with pytest.raises((ValueError, RuntimeError)):
            session.note_spawned(process)
        assert len(reads) >= 2
        assert_retired_after_birth(session, process)
