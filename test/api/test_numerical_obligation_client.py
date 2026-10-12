"""Offline protocol/peer tests; no providers or owner databases are opened."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import time

import pytest

MODULE = Path(__file__).resolve().parents[2] / "ipfs_accelerate_py/agent_supervisor/runtime/numerical_obligation_client.py"
spec = importlib.util.spec_from_file_location("numerical_obligation_client_standalone", MODULE)
client = importlib.util.module_from_spec(spec)
spec.loader.exec_module(client)


def request(**query):
    fields = dict(path="sample.py", function_id=None, kind=None, cursor=None, limit=8, max_bytes=8192)
    fields.update(query)
    return dict(schema=client.REQUEST_SCHEMA, session_id="a" * 32, request_id="b" * 32, query=fields)


def descriptor():
    identity = client.process_identity(os.getpid())
    now = time.monotonic_ns()
    return dict(schema=client.DESCRIPTOR_SCHEMA, session_id="a" * 32,
        owner_uid=identity["uid"], worker_uid=identity["uid"] + 1,
        owner_pid=os.getpid(), owner_start_time_ticks=identity["start_time_ticks"],
        workspace=str(Path.cwd()), task_id="task-1", task_cid="sha256:task",
        context_bundle_sha256="c" * 64, context_receipt_sha256="d" * 64,
        source_tree_id="sha256:tree", boundary_manifest_sha256="e" * 64,
        socket_path="/owner/socket", socket_device=1, socket_inode=2,
        issued_monotonic_ns=now - 1_000_000_000, expires_monotonic_ns=now + 30_000_000_000,
        max_requests=64, max_total_response_bytes=524288,
        proof_authority=False, execution_authority=False, completion_authority=False)


def response(*, ok=True):
    value = dict(schema=client.RESPONSE_SCHEMA, session_id="a" * 32,
        request_id="b" * 32, ok=ok, page=None, reason_code="stale_context" if not ok else None)
    if ok:
        value["page"] = dict(schema="supervisor-numerical-obligation-page@1",
            binding=dict(path="sample.py"), context_receipt_sha256="d" * 64,
            source_tree_id="sha256:tree", source_obligations_discharged=0, rows=[], returned_count=0,
            query=dict(function_id=None, kind=None, limit=8, max_bytes=8000),
            **{key: False for key in ("proof_authority", "execution_authority", "completion_authority",
                "formalization_authority", "source_semantics_verified", "whole_program_semantics_verified", "source_executed")})
    return value


def test_standalone_source_imports_only_standard_library():
    import ast
    import sys
    tree = ast.parse(MODULE.read_text())
    names = {node.module.split(".")[0] for node in ast.walk(tree)
             if isinstance(node, ast.ImportFrom)}
    names.update(alias.name.split(".")[0] for node in ast.walk(tree)
                 if isinstance(node, ast.Import) for alias in node.names)
    assert names <= sys.stdlib_module_names


@pytest.mark.parametrize("raw", [b'{"a":1,"a":2}', b'{"a":{"x":1,"x":2}}',
    b'{"a":NaN}', b'{"a":Infinity}', b'{"a":-Infinity}', b'{"a":1e999}',
    b'{"a":\xff}', b'{', b'{}x', b' ' * 8193], ids=['duplicate', 'nested-duplicate', 'nan', 'infinity', 'negative-infinity', 'overflow', 'utf8', 'truncated', 'suffix', 'oversized'])
def test_strict_json_refuses_ambiguous_nonfinite_malformed_or_oversized_input(raw):
    with pytest.raises(ValueError):
        client.strict_json_loads(raw)


def test_unicode_canonical_roundtrip_and_frame():
    value = {"path": "café/λ.py", "count": 2}
    raw = client.canonical_json(value)
    assert b"\\u" not in raw
    assert client.strict_json_loads(raw) == value
    left, right = socket.socketpair()
    with left, right:
        client.send_frame(left, value)
        assert client.receive_frame(right) == value


@pytest.mark.parametrize("wire", [b"\0\0\x20\x01", b"\0\0\0\x01", b"\0\0\0\x05{}",
    (len(b'{"x":1,"x":2}')).to_bytes(4, "big") + b'{"x":1,"x":2}',
    (len(b'{ "x": 1 }')).to_bytes(4, "big") + b'{ "x": 1 }'])
def test_receive_refuses_oversized_truncated_duplicate_or_noncanonical_frame(wire):
    left, right = socket.socketpair()
    with left, right:
        left.sendall(wire)
        left.shutdown(socket.SHUT_WR)
        with pytest.raises(ValueError):
            client.receive_frame(right)


def test_outgoing_frame_bound_is_utf8_and_rejects_boolean_budget():
    left, right = socket.socketpair()
    with left, right:
        with pytest.raises(ValueError):
            client.send_frame(left, {"x": "é" * 4096})
        with pytest.raises(ValueError):
            client.send_frame(left, {}, max_bytes=True)


@pytest.mark.parametrize("key,value", [("limit", True), ("limit", 0), ("limit", 9),
    ("max_bytes", 1023), ("max_bytes", 8193), ("max_bytes", True),
    ("path", "../secret"), ("path", "/owner/lake"), ("path", "./sample.py"),
    ("path", "."), ("kind", ""), ("function_id", False), ("cursor", False),
    ("cursor", {"owner": "forged"})])
def test_query_refuses_unbounded_or_ambiguous_selection(key, value):
    body = request()
    body["query"][key] = value
    with pytest.raises(ValueError):
        client.validate_request(body)


def test_closed_request_rejects_owner_locator_and_boolean_cursor():
    body = request()
    body["repository"] = "/owner"
    with pytest.raises(ValueError):
        client.validate_request(body)
    body = request()
    body["query"]["cursor"] = dict(context_receipt_sha256="d" * 64,
        cursor=dict(schema="numerical-obligation-cursor@1", request_sha256="f" * 64, offset=1))
    assert client.validate_request(body) == body
    body["query"]["cursor"]["cursor"]["offset"] = True
    with pytest.raises(ValueError):
        client.validate_request(body)


@pytest.mark.parametrize("key,value", [("worker_uid", True), ("owner_start_time_ticks", 0),
    ("socket_inode", False), ("socket_device", -1), ("max_requests", 65),
    ("max_total_response_bytes", 524289), ("proof_authority", 0),
    ("boundary_manifest_sha256", "foreign"), ("workspace", "/app/../other")])
def test_descriptor_schema_rejects_type_confusion_and_unbounded_declarations(key, value):
    body = descriptor()
    body[key] = value
    with pytest.raises(ValueError):
        client.validate_descriptor(body)


def test_descriptor_lifetime_is_independent_and_bounded():
    body = descriptor()
    assert client.validate_descriptor(body) == body
    body["expires_monotonic_ns"] = body["issued_monotonic_ns"] + 1800_000_000_001
    with pytest.raises(ValueError):
        client.validate_descriptor(body)


@pytest.fixture
def owner_files(monkeypatch):
    # Exercise real files and socket metadata while emulating the reader UID.
    # The separate container qualification supplies actual UID separation.
    with tempfile.TemporaryDirectory(prefix="num-client-", dir=Path.home()) as name:
        parent = Path(name)
        parent.chmod(0o750)
        workspace = parent / "workspace"
        workspace.mkdir()
        monkeypatch.chdir(workspace)
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        endpoint = parent / "owner.sock"
        listener.bind(str(endpoint))
        endpoint.chmod(0o660)
        listener.listen(1)
        value = descriptor()
        value["socket_path"] = str(endpoint)
        info = endpoint.lstat()
        value.update(socket_device=info.st_dev, socket_inode=info.st_ino)
        path = parent / "descriptor.json"
        def write():
            raw = client.canonical_json(value)
            if path.exists():
                path.chmod(0o600)
            path.write_bytes(raw)
            path.chmod(0o440)
            return hashlib.sha256(raw).hexdigest()
        digest = write()
        monkeypatch.setattr(client.os, "getuid", lambda: value["worker_uid"])
        monkeypatch.setattr(client.os, "geteuid", lambda: value["worker_uid"])
        monkeypatch.setattr(client.os, "access", lambda *args: False)
        try:
            yield path, digest, value, write, listener
        finally:
            listener.close()


def test_load_descriptor_checks_actual_socket_files_owner_birth_and_workspace(owner_files):
    path, digest, value, _, _ = owner_files
    assert client.load_descriptor(path, digest) == value


@pytest.mark.parametrize("change", ["digest", "expired", "future", "workspace", "file_mode",
    "socket_mode", "socket_inode", "owner_birth", "worker_uid", "symlink", "ancestor_mode"])
def test_load_descriptor_refuses_stale_foreign_or_writable_boundary(owner_files, change):
    path, digest, value, write, _ = owner_files
    if change == "digest":
        digest = "0" * 64
    elif change == "expired":
        value["issued_monotonic_ns"] = time.monotonic_ns() - 60_000_000_000
        value["expires_monotonic_ns"] = time.monotonic_ns() - 1
        digest = write()
    elif change == "future":
        value["issued_monotonic_ns"] = time.monotonic_ns() + 10_000_000_000
        value["expires_monotonic_ns"] = value["issued_monotonic_ns"] + 10_000_000_000
        digest = write()
    elif change == "workspace":
        value["workspace"] = str(path.parent)
        digest = write()
    elif change == "file_mode":
        path.chmod(0o660)
    elif change == "socket_mode":
        Path(value["socket_path"]).chmod(0o666)
    elif change == "socket_inode":
        value["socket_inode"] += 1
        digest = write()
    elif change == "owner_birth":
        value["owner_start_time_ticks"] += 1
        digest = write()
    elif change == "worker_uid":
        value["worker_uid"] = value["owner_uid"]
        digest = write()
    elif change == "symlink":
        other = path.parent / "link.json"
        other.symlink_to(path)
        path = other
    else:
        path.parent.chmod(0o770)
    with pytest.raises(ValueError):
        client.load_descriptor(path, digest)


def test_worker_writable_parent_refused_even_with_expected_modes(owner_files, monkeypatch):
    path, digest, _, _, _ = owner_files
    monkeypatch.setattr(client.os, "access", lambda *args: True)
    with pytest.raises(ValueError):
        client.load_descriptor(path, digest)


@pytest.mark.parametrize("change", ["pid", "uid", "birth"])
def test_server_peer_must_match_kernel_credentials_and_birth(change):
    value = descriptor()
    value[{"pid": "owner_pid", "uid": "owner_uid", "birth": "owner_start_time_ticks"}[change]] += 1
    left, right = socket.socketpair()
    with left, right:
        with pytest.raises(ValueError):
            client._verify_peer(right, value)


def test_actual_socket_peer_and_process_identity_are_observed():
    value = descriptor()
    left, right = socket.socketpair()
    with left, right:
        client._verify_peer(right, value)
    identity = client.process_identity(os.getpid())
    assert identity["uid"] == os.geteuid()
    assert identity["parent_pid"] == os.getppid()
    assert identity["start_time_ticks"] > 0


@pytest.mark.parametrize("change", ["session", "request", "budget", "proof", "bool_zero", "owner", "path", "reason",
                                    "filter", "bool_limit", "row_count"])
def test_response_is_bound_to_request_owner_scope_and_complete_byte_budget(change):
    body = response()
    if change == "session": body["session_id"] = "f" * 32
    elif change == "request": body["request_id"] = "f" * 32
    elif change == "budget": body["page"]["padding"] = "x" * 8192
    elif change == "proof": body["page"]["proof_authority"] = True
    elif change == "bool_zero": body["page"]["source_obligations_discharged"] = False
    elif change == "owner": body["page"]["context_receipt_sha256"] = "f" * 64
    elif change == "path": body["page"]["binding"]["path"] = "other.py"
    elif change == "filter": body["page"]["query"]["kind"] = "other"
    elif change == "bool_limit": body["page"]["query"]["limit"] = True
    elif change == "row_count": body["page"]["returned_count"] = 1
    else:
        body = response(ok=False)
        body["reason_code"] = "raw exception or private path"
    with pytest.raises(ValueError):
        client.validate_response(body, request=request(), descriptor=descriptor())


def test_valid_success_and_closed_refusal_do_not_gain_authority():
    for ok in (True, False):
        body = response(ok=ok)
        assert client.validate_response(body, request=request(), descriptor=descriptor()) == body


def test_cli_error_never_prints_exception_or_user_paths(capsys, monkeypatch):
    monkeypatch.setattr(client, "request_page", lambda *a, **k: (_ for _ in ()).throw(ValueError("secret /owner/private")))
    code = client.main([client.DESCRIPTOR_FLAG, "/owner/private", client.DIGEST_FLAG, "a" * 64, "--path", "sample.py"])
    captured = capsys.readouterr()
    assert code == 2
    assert captured.out == ""
    assert captured.err == '{"reason_code":"transport_error"}\n'


def test_cli_reads_only_explicit_descriptor_environment(capsys, monkeypatch):
    calls = []
    monkeypatch.setenv(client.DESCRIPTOR_ENV, "/owner/descriptor")
    monkeypatch.setenv(client.DIGEST_ENV, "a" * 64)
    def invoke(*args, **kwargs):
        calls.append((args, kwargs))
        return response(ok=False)
    monkeypatch.setattr(client, "request_page", invoke)
    assert client.main(["--path", "sample.py", "--kind", "nonempty"]) == 2
    assert calls[0][0] == ("/owner/descriptor", "a" * 64)
    assert calls[0][1]["kind"] == "nonempty"
    assert json.loads(capsys.readouterr().out)["reason_code"] == "stale_context"


@pytest.mark.parametrize("outcome", ["page", "refused", "oversized"])
def test_request_page_uses_real_framing_and_peer_checks(owner_files, outcome):
    path, digest, value, _, listener = owner_files
    listener.settimeout(3)
    observed = []
    errors = []
    def serve():
        try:
            channel, _ = listener.accept()
            with channel:
                channel.settimeout(3)
                incoming = client.validate_request(client.receive_frame(channel))
                observed.append(incoming)
                outgoing = response(ok=outcome != "refused")
                outgoing["request_id"] = incoming["request_id"]
                if outcome == "oversized":
                    outgoing["page"]["padding"] = "x" * 4096
                client.send_frame(channel, outgoing)
        except BaseException as error:
            errors.append(error)
    worker = threading.Thread(target=serve)
    worker.start()
    try:
        if outcome == "oversized":
            with pytest.raises(ValueError, match="frame exceeds"):
                client.request_page(path, digest, path="sample.py", max_bytes=1024)
        else:
            result = client.request_page(path, digest, path="sample.py")
            assert result["ok"] is (outcome == "page")
        assert len(observed) == 1
        assert observed[0]["session_id"] == value["session_id"]
        assert observed[0]["query"]["path"] == "sample.py"
    finally:
        worker.join(4)
    assert not worker.is_alive()
    assert not errors


def test_dead_or_zombie_pid_cannot_supply_peer_identity():
    process = subprocess.Popen([sys.executable, "-I", "-c", "pass"])
    pid = process.pid
    try:
        os.waitid(os.P_PID, pid, os.WEXITED | os.WNOWAIT)
        with pytest.raises(ValueError, match="live process"):
            client.process_identity(pid)
    finally:
        process.wait(timeout=3)
    with pytest.raises(OSError):
        client.process_identity(pid)
