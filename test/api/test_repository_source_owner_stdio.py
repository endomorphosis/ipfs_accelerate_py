"""Worker transport fixtures use real pipes; no scheduler, provider or private grant."""
from contextlib import contextmanager
import copy
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_owner_stdio as stdio
from test.api.test_repository_source_owner_protocol import descriptor


@pytest.fixture
def authored(monkeypatch, descriptor, tmp_path):
    descriptor["workspace"] = str(tmp_path)
    state = SimpleNamespace(descriptor=descriptor, loaded=[], calls=[], response=None, called=threading.Event())
    def load(path, digest, *, workspace=None):
        state.loaded.append((path, digest, workspace))
        assert path == "/owner/source.json" and digest == "a" * 64 and workspace == tmp_path
        return stdio.protocol.validate_descriptor(copy.deepcopy(state.descriptor))
    def request(path, digest, *, message, workspace=None, deadline_monotonic=None, cancel_event=None):
        state.calls.append(copy.deepcopy(message))
        state.called.set()
        result = None if "id" not in message else {"jsonrpc": "2.0", "id": message["id"], "result": {}}
        value = {"schema": stdio.protocol.RESPONSE_SCHEMA, "session_id": descriptor["session_id"],
                 "request_id": "2" * 32, "ok": True, "message": result, "reason_code": None}
        return value if state.response is None else state.response(message, value)
    monkeypatch.setattr(stdio.protocol, "load_descriptor", load)
    monkeypatch.setattr(stdio.protocol, "request_message", request)
    return state


def ping(identity=1):
    return {"jsonrpc": "2.0", "id": identity, "method": "ping"}


def line(value):
    return stdio.protocol.canonical_json(value) + b"\n"


@contextmanager
def pipes():
    input_fd, sender_fd = os.pipe()
    receiver_fd, output_fd = os.pipe()
    incoming, sender, receiver, outgoing = [os.fdopen(fd, mode, buffering=0) for fd, mode in
        ((input_fd, "rb"), (sender_fd, "wb"), (receiver_fd, "rb"), (output_fd, "wb"))]
    try:
        yield incoming, sender, receiver, outgoing
    finally:
        for stream in (incoming, sender, receiver, outgoing):
            stream.close()


def start_bridge(tmp_path, incoming, outgoing, **controls):
    outcomes = []
    def run():
        try:
            outcomes.append(stdio.serve("/owner/source.json", "a" * 64, workspace=tmp_path,
                incoming=incoming, outgoing=outgoing, **controls))
        except BaseException as error:
            outcomes.append(error)
    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread, outcomes


def finish(thread, *, cancel=None):
    if cancel is not None:
        cancel.set()
    thread.join(3)
    assert not thread.is_alive(), "worker stdio did not stop within the bounded control"


def drain(receiver):
    os.set_blocking(receiver.fileno(), False)
    raw = bytearray()
    try:
        while True:
            try:
                part = os.read(receiver.fileno(), 65536)
            except BlockingIOError:
                break
            if not part:
                break
            raw.extend(part)
    finally:
        os.set_blocking(receiver.fileno(), True)
    return bytes(raw)


def exchange(tmp_path, raw, **controls):
    with pipes() as (incoming, sender, receiver, outgoing):
        thread, outcomes = start_bridge(tmp_path, incoming, outgoing, **controls)
        try:
            sender.write(raw)
            sender.close()
            finish(thread)
            received = drain(receiver)
            assert os.get_blocking(incoming.fileno()) and os.get_blocking(outgoing.fileno())
        finally:
            if thread.is_alive():
                finish(thread, cancel=controls.get("cancel_event"))
        return outcomes, [json.loads(row) for row in received.splitlines()], received


def test_eof_without_request_is_clean_and_never_calls_owner(authored, tmp_path):
    outcomes, replies, _ = exchange(tmp_path, b"")
    assert outcomes == [0] and replies == [] and authored.calls == []
    assert len(authored.loaded) == 1


@pytest.mark.parametrize("identity", [1, -1, "exact-id"])
def test_jsonrpc_response_preserves_exact_id_type(authored, tmp_path, identity):
    outcomes, replies, raw = exchange(tmp_path, line(ping(identity)))
    assert outcomes == [0] and authored.calls == [ping(identity)]
    assert replies == [{"jsonrpc": "2.0", "id": identity, "result": {}}]
    assert raw == line(replies[0])


def test_initialized_notification_receives_no_output(authored, tmp_path):
    message = {"jsonrpc": "2.0", "method": "notifications/initialized"}
    outcomes, replies, _ = exchange(tmp_path, line(message) + line(ping()))
    assert outcomes == [0] and len(authored.calls) == 2
    assert replies == [{"jsonrpc": "2.0", "id": 1, "result": {}}]


@pytest.mark.parametrize("message", [
    {"jsonrpc": "2.0", "method": "notifications/unknown"},
    {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {"command": "forbidden"}},
    {"jsonrpc": "2.0", "method": "ping"},
])
def test_invalid_notifications_never_acquire_a_synthetic_reply(authored, tmp_path, message):
    outcomes, replies, _ = exchange(tmp_path, line(message))
    assert outcomes == [0] and replies == [] and authored.calls == []


@pytest.mark.parametrize("message", [
    {"jsonrpc": "2.0", "id": "edit", "method": "tools/call", "params": {
        "name": "repository_replace_symbol", "arguments": {}}},
    {"jsonrpc": "2.0", "id": "edit", "method": "tools/call", "params": {
        "name": "repository_status", "arguments": {"workspace": "/secret"}}},
    {"jsonrpc": "2.0", "id": "edit", "method": "resources/read"},
])
def test_disallowed_requests_never_reach_owner(authored, tmp_path, message):
    outcomes, replies, raw = exchange(tmp_path, line(message))
    assert outcomes == [0] and authored.calls == []
    assert replies == [{"jsonrpc": "2.0", "id": "edit", "error": {"code": -32600, "message": "invalid_request"}}]
    assert b"/secret" not in raw and b"repository_replace_symbol" not in raw


@pytest.mark.parametrize("raw", [b'{"jsonrpc":"2.0","id":1,"id":2}\n', b'{"a":NaN}\n',
    b'{"a":1e999}\n', b'\xff\n', b'{bad}\n'])
def test_malformed_json_has_closed_parse_error_and_can_continue(authored, tmp_path, raw):
    outcomes, replies, _ = exchange(tmp_path, raw + line(ping()))
    assert outcomes == [0] and authored.calls == [ping()]
    assert replies[0] == {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "invalid_json"}}
    assert replies[1] == {"jsonrpc": "2.0", "id": 1, "result": {}}


@pytest.mark.parametrize("message", [[], None, {"jsonrpc": "2.0", "id": True, "method": "ping"},
    {"jsonrpc": "2.0", "id": None, "method": "ping"}])
def test_invalid_request_shape_uses_null_error_identity(authored, tmp_path, message):
    outcomes, replies, _ = exchange(tmp_path, line(message))
    assert outcomes == [0] and authored.calls == []
    assert replies[0]["id"] is None and replies[0]["error"]["code"] == -32600


def test_owner_jsonrpc_error_passes_without_rewriting_id(authored, tmp_path):
    def error(message, value):
        value["message"] = {"jsonrpc": "2.0", "id": message["id"],
                            "error": {"code": -32002, "message": "not_initialized"}}
        return value
    authored.response = error
    outcomes, replies, _ = exchange(tmp_path, line(ping("current")))
    assert outcomes == [0]
    assert replies == [{"jsonrpc": "2.0", "id": "current", "error": {"code": -32002, "message": "not_initialized"}}]


@pytest.mark.parametrize("reason", sorted(stdio.protocol.REASON_CODES))
def test_owner_refusal_is_closed_and_ends_without_retry(authored, tmp_path, reason):
    def refuse(message, value):
        value.update(ok=False, message=None, reason_code=reason)
        return value
    authored.response = refuse
    outcomes, replies, _ = exchange(tmp_path, line(ping("id")) + line(ping(2)))
    assert outcomes == [2] and len(authored.calls) == 1
    assert replies == [{"jsonrpc": "2.0", "id": "id", "error": {"code": -32000, "message": reason}}]


def test_refused_notification_is_silent_and_fatal(authored, tmp_path):
    authored.response = lambda message, value: {**value, "ok": False, "message": None, "reason_code": "session_closed"}
    outcomes, replies, _ = exchange(tmp_path, line({"jsonrpc": "2.0", "method": "notifications/initialized"}))
    assert outcomes == [2] and replies == [] and len(authored.calls) == 1


@pytest.mark.parametrize("kind", ["wrong_id", "wrong_session", "private_field", "source_exception", "invalid_reason"])
def test_unusable_owner_result_never_leaks_source_or_retries(authored, tmp_path, kind):
    def invalid(message, value):
        if kind == "wrong_id":
            value["message"]["id"] = "different"
        elif kind == "wrong_session":
            value["session_id"] = "f" * 32
        elif kind == "private_field":
            value["private_index"] = "/do-not-print"
        elif kind == "source_exception":
            raise RuntimeError("secret-source-fragment")
        else:
            value.update(ok=False, message=None, reason_code="secret-source-fragment")
        return value
    authored.response = invalid
    outcomes, replies, raw = exchange(tmp_path, line(ping()) + line(ping(2)))
    assert outcomes == [2] and len(authored.calls) == 1
    assert replies == [{"jsonrpc": "2.0", "id": 1, "error": {"code": -32000, "message": "owner_unavailable"}}]
    assert b"secret" not in raw and b"do-not-print" not in raw


def test_request_count_is_enforced_even_when_owner_would_allow_more(authored, tmp_path):
    authored.descriptor["max_requests"] = 2
    outcomes, replies, _ = exchange(tmp_path, line(ping(1)) + line(ping(2)) + line(ping(3)))
    assert outcomes == [0] and len(authored.calls) == 2 and len(replies) == 2


def test_invalid_lines_consume_local_request_count(authored, tmp_path):
    authored.descriptor["max_requests"] = 2
    outcomes, replies, _ = exchange(tmp_path, b"invalid\ninvalid\n" + line(ping()))
    assert outcomes == [0] and len(replies) == 2 and authored.calls == []


def test_local_aggregate_budget_counts_full_owner_envelopes(authored, tmp_path):
    authored.descriptor["max_total_response_bytes"] = 1024
    def text(message, value):
        value["message"]["result"] = {"source": "x" * 400}
        return value
    authored.response = text
    outcomes, replies, raw = exchange(tmp_path, b"".join(line(ping(i)) for i in range(4)))
    assert outcomes == [2] and len(authored.calls) == 2 and len(replies) == 1
    assert len(raw) <= 1024 and replies[0]["result"]["source"] == "x" * 400


def test_notification_envelopes_consume_budget_without_stdout(authored, tmp_path):
    authored.descriptor["max_total_response_bytes"] = 1024
    notification = {"jsonrpc": "2.0", "method": "notifications/initialized"}
    outcomes, replies, _ = exchange(tmp_path, line(notification) * 20)
    assert outcomes == [2] and replies == [] and 1 < len(authored.calls) < 20


def test_local_error_bytes_also_consume_budget(authored, tmp_path):
    authored.descriptor["max_total_response_bytes"] = 1024
    outcomes, replies, raw = exchange(tmp_path, b"bad\n" * 64)
    assert outcomes == [2] and authored.calls == [] and 1 < len(replies) < 64 and len(raw) <= 1024


@pytest.mark.parametrize("raw", [b'{"jsonrpc":"2.0"}', b"partial"])
def test_partial_eof_is_a_closed_frame_failure(authored, tmp_path, raw):
    outcomes, replies, _ = exchange(tmp_path, raw)
    assert outcomes == [2] and authored.calls == []
    assert replies == [{"jsonrpc": "2.0", "id": None, "error": {"code": -32600, "message": "request_limit"}}]


def test_exact_payload_limit_allows_one_extra_newline(authored, tmp_path):
    raw = line(ping()).rstrip(b"\n")
    padded = raw + b" " * (stdio.MAX_REQUEST_BYTES - len(raw)) + b"\n"
    outcomes, replies, _ = exchange(tmp_path, padded)
    assert outcomes == [0] and authored.calls == [ping()] and len(replies) == 1


def test_oversized_line_does_not_forward_a_truncated_request(authored, tmp_path):
    raw = b" " * (stdio.MAX_REQUEST_BYTES + 1) + b"\n"
    outcomes, replies, _ = exchange(tmp_path, raw)
    assert outcomes == [2] and authored.calls == [] and replies[0]["error"]["message"] == "request_limit"


@pytest.mark.parametrize("partial", [b"", b'{"jsonrpc":"2.0",'])
def test_idle_and_partial_input_cancellation_returns_two(authored, tmp_path, partial):
    signal = threading.Event()
    with pipes() as (incoming, sender, receiver, outgoing):
        sender.write(partial)
        thread, outcomes = start_bridge(tmp_path, incoming, outgoing, cancel_event=signal)
        finish(thread, cancel=signal)
        assert outcomes == [2] and drain(receiver) == b"" and authored.calls == []
        assert os.get_blocking(incoming.fileno()) and os.get_blocking(outgoing.fileno())


@pytest.mark.parametrize("partial", [b"", b'{"jsonrpc":"2.0",'])
def test_idle_and_partial_input_honor_absolute_caller_deadline(authored, tmp_path, partial):
    with pipes() as (incoming, sender, receiver, outgoing):
        sender.write(partial)
        thread, outcomes = start_bridge(tmp_path, incoming, outgoing, deadline_monotonic=time.monotonic() + .05)
        finish(thread)
        assert outcomes == [2] and drain(receiver) == b"" and authored.calls == []


def test_descriptor_expiry_bounds_idle_without_caller_deadline(authored, tmp_path):
    authored.descriptor["expires_monotonic_ns"] = time.monotonic_ns() + 50_000_000
    with pipes() as (incoming, sender, receiver, outgoing):
        thread, outcomes = start_bridge(tmp_path, incoming, outgoing)
        finish(thread)
        assert outcomes == [2] and drain(receiver) == b""


def test_cancel_after_owner_read_suppresses_delivery_without_retry(authored, tmp_path):
    signal = threading.Event()
    def stop(message, value):
        value["message"]["result"] = {"source": "retained-source-not-delivered"}
        signal.set()
        return value
    authored.response = stop
    outcomes, replies, _ = exchange(tmp_path, line(ping()) + line(ping(2)), cancel_event=signal)
    assert outcomes == [2] and replies == [] and len(authored.calls) == 1


def test_blocked_stdout_cancellation_restores_fd_modes(authored, tmp_path):
    signal = threading.Event()
    with pipes() as (incoming, sender, receiver, outgoing):
        os.set_blocking(outgoing.fileno(), False)
        filled = 0
        try:
            while True:
                filled += os.write(outgoing.fileno(), b"x" * 4096)
        except BlockingIOError:
            pass
        os.set_blocking(outgoing.fileno(), True)
        sender.write(line(ping()))
        thread, outcomes = start_bridge(tmp_path, incoming, outgoing, cancel_event=signal)
        assert authored.called.wait(1)
        timer = threading.Timer(.05, signal.set)
        timer.start()
        try:
            finish(thread)
        finally:
            timer.join(1)
        assert outcomes == [2] and len(authored.calls) == 1
        assert drain(receiver) == b"x" * filled
        assert os.get_blocking(incoming.fileno()) and os.get_blocking(outgoing.fileno())


def test_blocked_stdout_honors_absolute_deadline(authored, tmp_path):
    with pipes() as (incoming, sender, receiver, outgoing):
        os.set_blocking(outgoing.fileno(), False)
        try:
            while True:
                os.write(outgoing.fileno(), b"x" * 4096)
        except BlockingIOError:
            pass
        os.set_blocking(outgoing.fileno(), True)
        sender.write(line(ping()))
        thread, outcomes = start_bridge(tmp_path, incoming, outgoing, deadline_monotonic=time.monotonic() + .05)
        finish(thread)
        assert outcomes == [2] and len(authored.calls) == 1


@pytest.mark.parametrize("deadline", [True, "deadline", float("nan"), float("inf"), 10**400, -1])
def test_invalid_or_expired_caller_deadline_refuses_before_descriptor(authored, tmp_path, deadline):
    outcomes, replies, _ = exchange(tmp_path, b"", deadline_monotonic=deadline)
    assert outcomes == [2] and replies == [] and authored.loaded == [] and authored.calls == []


def test_initial_descriptor_refusal_has_no_stdout_or_owner_request(authored, tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise ValueError("private path must not be exposed")
    monkeypatch.setattr(stdio.protocol, "load_descriptor", fail)
    outcomes, replies, _ = exchange(tmp_path, line(ping()))
    assert outcomes == [2] and replies == [] and authored.calls == []


def test_same_input_output_fd_refused(authored, tmp_path):
    stream, peer = socket.socketpair()
    with stream, peer:
        assert stdio.serve("/owner/source.json", "a" * 64, workspace=tmp_path,
                           incoming=stream, outgoing=stream) == 2


def test_isolated_direct_import_has_no_accelerator_or_native_modules():
    source = str(Path(stdio.__file__).resolve())
    script = (
        "import importlib.util,sys;"
        "s=importlib.util.spec_from_file_location('worker_stdio',sys.argv[1]);"
        "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
        "assert not any(k.startswith(('ipfs_accelerate_py','ipfs_datasets_py')) for k in sys.modules);"
        "assert m.protocol.DESCRIPTOR_FLAG=='--repository-source-owner-descriptor'"
    )
    result = subprocess.run([sys.executable, "-I", "-B", "-c", script, source],
                            capture_output=True, timeout=5, check=False)
    assert result.returncode == 0 and result.stdout == b"" and result.stderr == b""


def test_direct_cli_descriptor_failure_uses_only_closed_diagnostic(tmp_path):
    result = subprocess.run([sys.executable, "-I", "-B", str(Path(stdio.__file__).resolve()),
        stdio.protocol.DESCRIPTOR_FLAG, str(tmp_path / "private-marker-missing.json"),
        stdio.protocol.DIGEST_FLAG, "a" * 64, "--workspace", str(tmp_path)],
        input=b"", capture_output=True, timeout=5, check=False, cwd=tmp_path)
    assert result.returncode == 2 and result.stdout == b""
    assert result.stderr == (stdio.UNAVAILABLE + "\n").encode()


@pytest.mark.parametrize("flag", ["--config", "--resource-grant", "--helper", "--repository-source-owner-desc"])
def test_cli_has_no_capability_helper_or_abbreviated_selector(tmp_path, flag):
    result = subprocess.run([sys.executable, "-I", "-B", str(Path(stdio.__file__).resolve()),
        stdio.protocol.DESCRIPTOR_FLAG, str(tmp_path / "missing.json"),
        stdio.protocol.DIGEST_FLAG, "a" * 64, "--workspace", str(tmp_path), flag, "forbidden"],
        input=b"", capture_output=True, timeout=5, check=False, cwd=tmp_path)
    assert result.returncode == 2 and result.stdout == b"" and b"unrecognized arguments" in result.stderr


def test_local_parse_errors_leave_too_little_for_owner_reply_so_no_request_starts(authored, tmp_path):
    authored.descriptor["max_total_response_bytes"] = 1024
    budget = stdio._Budget(1024)
    error_bytes = len(line(stdio._error(None, code=-32700, reason="invalid_json")))
    malformed_count = 0
    while budget.can_request(authored.descriptor, ping()):
        budget.accounted += error_bytes
        malformed_count += 1
    assert budget.accounted < 1024
    outcomes, replies, raw = exchange(tmp_path, b"bad\n" * malformed_count + line(ping()))
    assert outcomes == [2] and authored.calls == []
    assert len(replies) == malformed_count and len(raw) == budget.accounted
