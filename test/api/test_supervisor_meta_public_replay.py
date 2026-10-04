"""Real context/PID suppression with an explicit activation-protocol double.

No database owners or training/inference are opened. The optional fork is a
tiny joined pipe control of inherited context, not a worker qualification.
"""
import os
import select
import signal
import threading

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import supervisor_meta_index as meta


@pytest.fixture
def activation_probe(monkeypatch):
    sentinel, calls = object(), []
    monkeypatch.setenv(meta.ENV_DUCKDB, "/authored/unopened/configured-meta.duckdb")

    def from_env(cls):
        calls.append((os.getpid(), threading.get_ident()))
        return sentinel

    def forbidden(*_, **__):
        pytest.fail("public replay suppression control opened metadata storage")

    monkeypatch.setattr(meta.SupervisorMetaIndex, "from_env", classmethod(from_env))
    monkeypatch.setattr(meta, "_connect", forbidden)
    return sentinel, calls


def test_normal_context_preserves_existing_activation_protocol(activation_probe):
    sentinel, calls = activation_probe
    assert meta._active() is sentinel
    assert meta._active() is sentinel
    assert calls == [(os.getpid(), threading.get_ident())] * 2


def test_configured_public_replay_skips_activation_and_real_mirror_hooks(activation_probe):
    sentinel, calls = activation_probe
    with meta.public_replay_without_metadata():
        assert meta._active() is None
        registered = meta.register_catalog(kind="metadata", locator_ref="authored-metadata")
        mirrored = meta.mirror_work_record(catalog_kind="proof_cache", record_kind="authored-control",
            record_ref="authored:record", subject_kind="record_cid", subject_ref="authored:record")
        assert registered["status"] == mirrored["status"] == "skip"
        assert calls == []
    assert meta._active() is sentinel
    assert len(calls) == 1


def test_nested_scopes_restore_outer_suppression_then_default(activation_probe):
    sentinel, calls = activation_probe
    with meta.public_replay_without_metadata():
        assert meta._active() is None
        with meta.public_replay_without_metadata():
            assert meta._active() is None
        assert meta._active() is None
        assert calls == []
    assert meta._active() is sentinel
    assert len(calls) == 1


def test_exception_resets_public_replay_context(activation_probe):
    sentinel, calls = activation_probe
    with pytest.raises(RuntimeError, match="authored callback fault"):
        with meta.public_replay_without_metadata():
            assert meta._active() is None
            raise RuntimeError("authored callback fault")
    assert calls == []
    assert meta._active() is sentinel
    assert len(calls) == 1


def test_inner_exception_does_not_release_outer_scope(activation_probe):
    sentinel, calls = activation_probe
    with meta.public_replay_without_metadata():
        with pytest.raises(ValueError, match="inner callback fault"):
            with meta.public_replay_without_metadata():
                raise ValueError("inner callback fault")
        assert meta._active() is None
        assert calls == []
    assert meta._active() is sentinel
    assert len(calls) == 1


def test_another_thread_retains_its_own_normal_and_scoped_context(activation_probe):
    sentinel, calls = activation_probe
    observed = []

    def separate_thread():
        observed.append(meta._active())
        with meta.public_replay_without_metadata():
            observed.append(meta._active())
        observed.append(meta._active())

    with meta.public_replay_without_metadata():
        thread = threading.Thread(target=separate_thread, daemon=True)
        thread.start()
        thread.join(timeout=5)
        assert not thread.is_alive()
        assert meta._active() is None
        assert observed == [sentinel, None, sentinel]
        assert len(calls) == 2
        assert all(pid == os.getpid() and identity == thread.ident for pid, identity in calls)
    assert meta._active() is sentinel
    assert len(calls) == 3


def test_explicit_copied_context_cannot_suppress_another_thread(activation_probe):
    from contextvars import copy_context

    sentinel, calls = activation_probe
    observed = []
    with meta.public_replay_without_metadata():
        copied = copy_context()
        thread = threading.Thread(target=lambda: copied.run(lambda: observed.append(meta._active())),
            daemon=True)
        thread.start()
        thread.join(timeout=5)
        assert not thread.is_alive()
        assert observed == [sentinel]
        assert len(calls) == 1 and calls[0] == (os.getpid(), thread.ident)
        assert meta._active() is None
    assert meta._active() is sentinel
    assert len(calls) == 2


@pytest.mark.skipif(not hasattr(os, "fork"), reason="PID inheritance control requires fork")
def test_inherited_fork_context_does_not_suppress_another_pid(activation_probe):
    sentinel, calls = activation_probe
    # Real PID boundary; the child only probes the activation double, writes
    # one tiny pipe message and exits. No owner, job, Git or numerical work.
    with meta.public_replay_without_metadata():
        reader, writer = os.pipe()
        try:
            child = os.fork()
        except BaseException:
            os.close(reader)
            os.close(writer)
            raise
        if child == 0:
            os.close(reader)
            try:
                observed = meta._active()
                status = b"active" if observed is sentinel else b"suppressed"
                os.write(writer, status + b":" + str(len(calls)).encode("ascii"))
            except BaseException as error:
                os.write(writer, b"error:" + type(error).__name__.encode("ascii"))
            finally:
                os.close(writer)
                os._exit(0)
        os.close(writer)
        try:
            ready, _, _ = select.select([reader], [], [], 5)
            if not ready:
                try:
                    os.kill(child, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                pytest.fail("fork context control exceeded its bounded pipe wait")
            response = os.read(reader, 128)
        finally:
            os.close(reader)
            waited, status = os.waitpid(child, 0)
        assert waited == child and os.waitstatus_to_exitcode(status) == 0
        assert response == b"active:1"
        assert calls == []
        assert meta._active() is None
    assert meta._active() is sentinel
    assert len(calls) == 1
