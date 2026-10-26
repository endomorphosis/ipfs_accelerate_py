"""Bounded MCP cancellation and private delegation; no alternate allocator.

Authored sessions isolate transport/lifetime semantics. Native subprocess
coverage requires the enclosing admitted runner's actual parent lease.
"""
from contextlib import contextmanager
import builtins
import io
import json
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_mcp as mcp
from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import (
    LeaseCancelledError, LeaseTimeoutError,
)
from test.api.test_repository_source_mcp import configuration, initialization
from test.api.test_repository_source_parent_admission import native_parent


class AuthoredSession:
    def __init__(self, workspace, state_dir, **options):
        self.options, self.closed = options, False

    def close(self):
        self.closed = True


@contextmanager
def pipe_pair():
    incoming, sender = os.pipe()
    receiver, outgoing = os.pipe()
    handles = [os.fdopen(incoming, "rb", buffering=0), os.fdopen(sender, "wb", buffering=0),
               os.fdopen(receiver, "rb", buffering=0), os.fdopen(outgoing, "wb", buffering=0)]
    try:
        yield handles
    finally:
        for stream in handles:
            stream.close()


def start_serve(server, incoming, outgoing):
    results = []
    def run():
        try:
            results.append(mcp.serve(server, incoming, outgoing))
        except BaseException as error:
            results.append(error)
    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread, results


def finish_thread(thread, signal):
    signal.set()
    thread.join(2)
    assert not thread.is_alive(), "bounded transport did not observe cancellation"


@pytest.mark.parametrize("value", [True, False, "deadline", float("nan"), float("inf"), 10**400])
def test_invalid_deadline_refuses_before_audit_creation(tmp_path, value):
    config, _, _ = configuration(tmp_path)
    with pytest.raises(ValueError):
        mcp.RepositoryMCPServer(config, session_factory=AuthoredSession, deadline_monotonic=value)
    assert not Path(config["audit_store"]).exists()


def test_expired_deadline_refuses_before_audit_creation(tmp_path):
    config, _, _ = configuration(tmp_path)
    with pytest.raises(LeaseTimeoutError):
        mcp.RepositoryMCPServer(config, session_factory=AuthoredSession,
                                deadline_monotonic=time.monotonic() - 1)
    assert not Path(config["audit_store"]).exists()


def test_deadline_is_forwarded_without_retrying_an_incompatible_factory(tmp_path):
    config, _, _ = configuration(tmp_path)
    deadline, calls = time.monotonic() + 30, []
    def legacy_factory(workspace, state_dir, *, representation):
        calls.append("unsafe retry")
    server = mcp.RepositoryMCPServer(config, session_factory=legacy_factory, deadline_monotonic=deadline)
    try:
        assert server.handle(initialization())["error"]["message"] == "repository_session_unavailable"
        assert calls == []
    finally:
        server.close()


def test_explicit_controls_forwarded_once_and_cleanup_remains_available(tmp_path):
    config, _, _ = configuration(tmp_path)
    signal, deadline = threading.Event(), time.monotonic() + 30
    server = mcp.RepositoryMCPServer(config, session_factory=AuthoredSession,
        cancel_event=signal, deadline_monotonic=deadline)
    try:
        assert "result" in server.handle(initialization())
        assert server.session.options == {"representation": "compact", "cancel_event": signal,
                                          "deadline_monotonic": deadline}
        signal.set()
        with pytest.raises(LeaseCancelledError):
            server.handle(dict(jsonrpc="2.0", id=2, method="ping"))
    finally:
        server.close()
    assert server.session.closed
    assert "deadline_monotonic" not in Path(config["audit_store"]).read_text()


@pytest.mark.parametrize("partial", [False, True])
def test_idle_and_partial_request_cancellation_restore_fd_modes(tmp_path, monkeypatch, partial):
    config, _, _ = configuration(tmp_path)
    signal, waiting = threading.Event(), threading.Event()
    server = mcp.RepositoryMCPServer(config, session_factory=AuthoredSession, cancel_event=signal)
    original = mcp._InterruptibleStdio._ready
    def ready(self, descriptor, **kwargs):
        if not kwargs.get("writing") and (not partial or self.pending):
            waiting.set()
        return original(self, descriptor, **kwargs)
    monkeypatch.setattr(mcp._InterruptibleStdio, "_ready", ready)
    try:
        with pipe_pair() as (incoming, sender, receiver, outgoing):
            if partial:
                sender.write(b'{"jsonrpc":"2.0",')
            thread, results = start_serve(server, incoming, outgoing)
            try:
                assert waiting.wait(1)
            finally:
                finish_thread(thread, signal)
            assert len(results) == 1 and isinstance(results[0], LeaseCancelledError)
            assert os.get_blocking(incoming.fileno()) and os.get_blocking(outgoing.fileno())
    finally:
        server.close()


def test_blocked_output_observes_cancellation_without_a_background_writer(tmp_path, monkeypatch):
    config, _, _ = configuration(tmp_path)
    signal, waiting = threading.Event(), threading.Event()
    server = mcp.RepositoryMCPServer(config, session_factory=AuthoredSession, cancel_event=signal)
    original = mcp._InterruptibleStdio._ready
    def ready(self, descriptor, **kwargs):
        if kwargs.get("writing"):
            waiting.set()
        return original(self, descriptor, **kwargs)
    monkeypatch.setattr(mcp._InterruptibleStdio, "_ready", ready)
    try:
        with pipe_pair() as (incoming, sender, receiver, outgoing):
            os.set_blocking(outgoing.fileno(), False)
            try:
                while True:
                    os.write(outgoing.fileno(), b"x" * 4096)
            except BlockingIOError:
                pass
            os.set_blocking(outgoing.fileno(), True)
            sender.write(b'{"jsonrpc":"2.0","id":1,"method":"ping"}\n')
            thread, results = start_serve(server, incoming, outgoing)
            try:
                assert waiting.wait(1)
            finally:
                finish_thread(thread, signal)
            assert len(results) == 1 and isinstance(results[0], LeaseCancelledError)
            assert os.get_blocking(outgoing.fileno())
    finally:
        server.close()


def test_idle_deadline_ends_without_another_client_request(tmp_path):
    config, _, _ = configuration(tmp_path)
    signal = threading.Event()
    server = mcp.RepositoryMCPServer(config, session_factory=AuthoredSession,
        cancel_event=signal, deadline_monotonic=time.monotonic() + .2)
    try:
        with pipe_pair() as (incoming, sender, receiver, outgoing):
            thread, results = start_serve(server, incoming, outgoing)
            try:
                thread.join(2)
                assert not thread.is_alive()
            finally:
                finish_thread(thread, signal)
            assert len(results) == 1 and isinstance(results[0], LeaseTimeoutError)
    finally:
        server.close()


def test_cancel_during_synchronous_mutation_retains_audit_truth_before_transport_exit(tmp_path):
    config, _, _ = configuration(tmp_path)
    signal, entered, release = threading.Event(), threading.Event(), threading.Event()
    class Session(AuthoredSession):
        def replace_symbol(self, **kwargs):
            entered.set()
            assert release.wait(2)
            return dict(generation=1, canonical_changed=True, refresh_required=True,
                        proof_authority=False, error="index_refresh_failed")
    server = mcp.RepositoryMCPServer(config, session_factory=Session, cancel_event=signal)
    assert "result" in server.handle(initialization())
    server.handle(dict(jsonrpc="2.0", method="notifications/initialized"))
    request = dict(jsonrpc="2.0", id=2, method="tools/call", params=dict(name="repository_replace_symbol",
        arguments=dict(generation=1, path="calc.py", symbol="add", expected_sha256="a" * 64,
                       replacement="def add(value): return value + 2\n")))
    try:
        with pipe_pair() as (incoming, sender, receiver, outgoing):
            sender.write(json.dumps(request).encode() + b"\n")
            thread, results = start_serve(server, incoming, outgoing)
            try:
                assert entered.wait(1)
                signal.set()
                assert thread.is_alive() and not server.session.closed
            finally:
                release.set()
                finish_thread(thread, signal)
            assert isinstance(results[0], LeaseCancelledError)
    finally:
        server.close()
    rows = [json.loads(line) for line in Path(config["audit_store"]).read_text().splitlines()]
    assert rows[-1]["canonical_changed"] is True and rows[-1]["success"] is False
    assert server.session.closed


def test_delegated_stdio_refuses_unpollable_streams_and_legacy_bytesio_still_works(tmp_path):
    config, _, _ = configuration(tmp_path)
    signal = threading.Event()
    server = mcp.RepositoryMCPServer(config, session_factory=AuthoredSession, cancel_event=signal)
    try:
        with pytest.raises((OSError, ValueError)):
            mcp.serve(server, io.BytesIO(), io.BytesIO())
    finally:
        server.close()
    # The unconfigured route retains the old file-like transport contract.
    legacy = SimpleNamespace(handle=lambda request: {"jsonrpc": "2.0", "id": 1, "result": {}})
    output = io.BytesIO()
    assert mcp.serve(legacy, io.BytesIO(b'{"jsonrpc":"2.0","id":1,"method":"ping"}\n'), output) == 0
    assert json.loads(output.getvalue())["result"] == {}


def test_main_requires_private_grant_pair_before_config_or_session(tmp_path, monkeypatch):
    monkeypatch.setattr(mcp, "load_config", lambda *args: pytest.fail("incomplete grant reached config"))
    with pytest.raises(SystemExit) as error:
        mcp.main(["--config", str(tmp_path / "config"), "--config-sha256", "a" * 64,
                  "--resource-grant", str(tmp_path / "grant")])
    assert error.value.code == 2


def _block_datasets_imports(monkeypatch):
    original, attempts = builtins.__import__, []
    def blocked(name, *args, **kwargs):
        if name == "ipfs_datasets_py" or name.startswith("ipfs_datasets_py."):
            attempts.append(name)
            raise ModuleNotFoundError("authored private missing dependency diagnostic", name=name)
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", blocked)
    return attempts


def test_main_help_does_not_import_datasets(monkeypatch, capsys):
    attempts = _block_datasets_imports(monkeypatch)
    with pytest.raises(SystemExit) as error:
        mcp.main(["--help"])
    assert error.value.code == 0 and attempts == []
    output = capsys.readouterr()
    assert "--resource-grant" in output.out and output.err == ""


@pytest.mark.parametrize("operation", ["eof", "ping", "initialize"])
def test_main_without_grant_preserves_lazy_missing_dependency_behavior(
        tmp_path, monkeypatch, capsys, operation):
    config, path, digest = configuration(tmp_path)
    request = (initialization() if operation == "initialize" else
               dict(jsonrpc="2.0", id=1, method="ping"))
    incoming = io.BytesIO(b"" if operation == "eof" else json.dumps(request).encode() + b"\n")
    outgoing = io.BytesIO()
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(buffer=incoming))
    monkeypatch.setattr(sys, "stdout", SimpleNamespace(buffer=outgoing))
    attempts = _block_datasets_imports(monkeypatch)
    assert mcp.main(["--config", str(path), "--config-sha256", digest]) == 0
    if operation == "initialize":
        # Initialization genuinely needs datasets; failure stays a closed MCP
        # response rather than preventing the ordinary CLI from starting.
        assert attempts
        assert json.loads(outgoing.getvalue())["error"]["message"] == "repository_session_unavailable"
        assert json.loads(Path(config["audit_store"]).read_text())["success"] is False
    else:
        assert attempts == []
        if operation == "eof":
            assert outgoing.getvalue() == b""
        else:
            assert json.loads(outgoing.getvalue()) == {"jsonrpc": "2.0", "id": 1, "result": {}}
    assert capsys.readouterr().err == ""


def test_main_delegated_missing_dependency_refuses_without_fallback(
        tmp_path, monkeypatch, capsys):
    config, path, digest = configuration(tmp_path)
    attempts = _block_datasets_imports(monkeypatch)
    monkeypatch.setattr(mcp, "RepositoryMCPServer", lambda *a, **k: pytest.fail("unsafe fallback"))
    assert mcp.main(["--config", str(path), "--config-sha256", digest,
        "--resource-grant", str(tmp_path / "grant"), "--resource-grant-sha256", "b" * 64]) == 2
    assert attempts and not Path(config["audit_store"]).exists()
    output = capsys.readouterr()
    assert output.out == "" and output.err == "repository_source_mcp_unavailable\n"


@pytest.mark.parametrize("failure", [None, "serve", "close"])
def test_main_closes_server_inside_delegated_scope(tmp_path, monkeypatch, capsys, failure):
    from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_resource_handoff as handoff
    config, path, digest = configuration(tmp_path)
    order, borrowed = [], object()
    deadline = time.monotonic() + 30
    @contextmanager
    def delegated(**kwargs):
        assert kwargs == dict(artifact=tmp_path / "grant", expected_sha256="b" * 64,
                             config_sha256=digest, workspace=Path(config["workspace"]))
        order.append("acquire")
        print("delegation setup diagnostics")
        try:
            yield SimpleNamespace(native_options=lambda: {"parent_lease": borrowed,
                "deadline_monotonic": deadline})
        finally:
            order.append("release")
    class Server:
        def __init__(self, cfg, **kwargs):
            assert cfg == config and kwargs == {"parent_lease": borrowed, "deadline_monotonic": deadline}
            order.append("server")
        def close(self):
            order.append("close")
            if failure == "close": raise ValueError("private cleanup diagnostic")
    def serve(*args):
        order.append("serve")
        if failure == "serve": raise LeaseCancelledError("private lease diagnostic")
        return 0
    monkeypatch.setattr(handoff, "delegated_repository_source_resources", delegated)
    monkeypatch.setattr(mcp, "RepositoryMCPServer", Server)
    monkeypatch.setattr(mcp, "serve", serve)
    result = mcp.main(["--config", str(path), "--config-sha256", digest,
        "--resource-grant", str(tmp_path / "grant"), "--resource-grant-sha256", "b" * 64])
    assert result == (0 if failure is None else 2)
    assert order == ["acquire", "server", "serve", "close", "release"]
    output = capsys.readouterr()
    assert not output.out
    assert "private" not in output.err and "lease_key" not in output.err
    assert "delegation setup diagnostics" in output.err


def test_private_grant_refusal_never_falls_back_to_a_default_root(tmp_path, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_resource_handoff as handoff
    config, path, digest = configuration(tmp_path)
    @contextmanager
    def refused(**kwargs):
        raise LeaseTimeoutError("authored admission refusal")
        yield
    monkeypatch.setattr(handoff, "delegated_repository_source_resources", refused)
    monkeypatch.setattr(mcp, "RepositoryMCPServer", lambda *a, **k: pytest.fail("unsafe fallback"))
    assert mcp.main(["--config", str(path), "--config-sha256", digest,
        "--resource-grant", str(tmp_path / "grant"), "--resource-grant-sha256", "b" * 64]) == 2
    assert not Path(config["audit_store"]).exists()


def _receive_frame(stream, pending, timeout=30):
    """Bound both bytes and time, including partial frames after readiness."""
    deadline = time.monotonic() + timeout
    descriptor = stream.fileno()
    previous = os.get_blocking(descriptor)
    os.set_blocking(descriptor, False)
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(descriptor, selectors.EVENT_READ)
            while True:
                newline = pending.find(b"\n")
                if newline >= 0:
                    assert newline <= mcp.MAX_RESPONSE_BYTES, "delegated MCP response exceeded byte bound"
                    raw = bytes(pending[:newline])
                    del pending[:newline + 1]
                    return json.loads(raw)
                assert len(pending) < mcp.MAX_RESPONSE_BYTES + 1, "delegated MCP response exceeded byte bound"
                remaining = deadline - time.monotonic()
                assert remaining > 0 and selector.select(remaining), "bounded delegated MCP response unavailable"
                try:
                    raw = os.read(descriptor, min(65536, mcp.MAX_RESPONSE_BYTES + 1 - len(pending)))
                except BlockingIOError:
                    continue
                assert raw, "delegated MCP exited before a complete response"
                pending.extend(raw)
    finally:
        os.set_blocking(descriptor, previous)


def _drain_owned_process_group(process, timeout=5):
    """Terminate the new session's group, reap its leader, and require drain.

    Descendants are not waitpid children of pytest. Their group must disappear
    before the borrowed native parent scope can be left; we never change the
    runner's subreaper state or signal its own process group.
    """
    assert process.pid != os.getpgrp()
    deadline = time.monotonic() + timeout
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=max(.01, deadline - time.monotonic()))
    while True:
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            return
        assert time.monotonic() < deadline, "owned delegated MCP process group did not drain"
        time.sleep(.01)


@pytest.mark.parametrize("closed", [False, True])
def test_native_cli_reader_bounds_partial_frames_and_restores_fd_mode(closed):
    with pipe_pair() as (incoming, sender, receiver, outgoing):
        sender.write(b'{"jsonrpc":')
        if closed:
            sender.close()
        with pytest.raises(AssertionError, match="complete response" if closed else "unavailable"):
            _receive_frame(incoming, bytearray(), timeout=.02)
        assert os.get_blocking(incoming.fileno())


def test_native_cli_reader_retains_already_buffered_next_frame():
    with pipe_pair() as (incoming, sender, receiver, outgoing):
        sender.write(b'{"id":1}\n{"id":2}\n')
        pending = bytearray()
        assert _receive_frame(incoming, pending, timeout=.2) == {"id": 1}
        assert _receive_frame(incoming, pending, timeout=.2) == {"id": 2}
        assert pending == bytearray() and os.get_blocking(incoming.fileno())


def test_native_cli_reader_enforces_response_byte_bound(monkeypatch):
    monkeypatch.setattr(mcp, "MAX_RESPONSE_BYTES", 32)
    with pipe_pair() as (incoming, sender, receiver, outgoing):
        sender.write(b"x" * 33)
        with pytest.raises(AssertionError, match="byte bound"):
            _receive_frame(incoming, bytearray(), timeout=.2)
        assert os.get_blocking(incoming.fileno())


@pytest.mark.parametrize("stop", ["eof", "cancel"])
def test_real_delegated_cli_uses_supplied_native_parent_and_drains_before_cleanup(
        tmp_path, native_parent, monkeypatch, stop):
    from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_policy as policy
    from ipfs_accelerate_py.agent_supervisor.runtime.repository_source_resource_handoff import install_repository_source_resources
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer.existing_resource_scheduler import policy_sha256

    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_SEMANTIC_TRUTH_AUTHORITY", "ipfs_datasets_py")
    config, _, _ = configuration(tmp_path)
    binding = policy.create_binding(workspace=Path(config["workspace"]), requested_policy=policy.POLICIES[2])
    identity = policy.create_runtime_store(binding)
    child = None
    try:
        # This descendant is owned by this test; cancellation never touches the
        # admitted runner parent or another consumer. Its context remains live
        # until the actual helper process has exited and been reaped.
        with native_parent.acquire_child(cpu_slots=1, memory_mb=512, child_process_slots=1,
                timeout=10, request_id="test-repository-mcp-delegation") as owned:
            install_repository_source_resources(binding, parent_lease=owned,
                expected_policy_sha256=policy_sha256(owned._scheduler.config.persisted_dict()),
                deadline_monotonic=time.monotonic() + 30)
            grant = binding["resource_grant"]
            command = [sys.executable, "-B", mcp.__file__, "--config", binding["config_path"],
                "--config-sha256", binding["config_sha256"], "--resource-grant", grant["path"],
                "--resource-grant-sha256", grant["sha256"]]
            with (tmp_path / "delegated-stderr.log").open("wb") as stderr:
                child = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                    stderr=stderr, bufsize=0, cwd="/tmp", start_new_session=True)
                pending = bytearray()
                def send(value):
                    child.stdin.write(json.dumps(value).encode() + b"\n")
                def receive():
                    return _receive_frame(child.stdout, pending)
                try:
                    send(initialization())
                    assert "result" in receive()
                    send(dict(jsonrpc="2.0", method="notifications/initialized"))
                    send(dict(jsonrpc="2.0", id=2, method="tools/call",
                        params=dict(name="repository_read", arguments=dict(generation=1, path="calc.py"))))
                    reply = receive()
                    assert reply["result"]["isError"] is False
                    assert "def add" in json.loads(reply["result"]["content"][0]["text"])["content"]
                    if stop == "cancel":
                        owned.cancel()
                    else:
                        child.stdin.close()
                    child.wait(timeout=5)
                    assert child.returncode == (2 if stop == "cancel" else 0)
                    assert not owned.released
                    assert not any(row.get("parent_lease_id") == owned.lease_id
                                   for row in owned._scheduler.active_leases())
                finally:
                    _drain_owned_process_group(child)
                    for stream in (child.stdin, child.stdout):
                        stream.close()
        assert not native_parent.released and not native_parent.cancelled
        audit = Path(binding["audit_store"]).read_text()
        assert "parent_lease" not in audit and "lease_key" not in audit and "resource_grant" not in audit
    finally:
        assert child is None or child.poll() is not None
        usage = policy.finish_runtime_store(binding, expected_identity=identity)
        assert usage["store_cleanup_complete"] is True
