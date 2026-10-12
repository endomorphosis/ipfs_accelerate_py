"""Native owner proxy and actual host process lineage, without a worker UID claim.

The deployment manifest/parent adapter is authored in these host tests. The
owner source reader, retained CAS, allocated Git worktree, DuckDB task and grant
are genuine. Owner-side proxy calls do not qualify separate-UID worker access;
the independent ancestry predicate uses real same-UID kernel observations.
"""

from contextlib import contextmanager
from copy import deepcopy
import base64
import hashlib
import json
import os
from pathlib import Path
import socket
import struct
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import uuid

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import source_formalization_context as guard
from ipfs_accelerate_py.agent_supervisor.runtime import source_lookup_worker_broker as broker
from ipfs_accelerate_py.agent_supervisor.runtime import source_lookup_worker_client as protocol
from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner as runner
from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import FakeQuackTransport, build_server
from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_inventory as sources
from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import OwnerLiveness
from ipfs_accelerate_py.agent_supervisor.task_sources import typed_state_owner as typed
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import canonical_json_bytes, content_identity
from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import open_duckdb_connection
from ipfs_datasets_py.logic.software_contracts.cache import ImmutableCAS
from ipfs_datasets_py.logic.software_contracts.content import cid_for_bytes, cid_for_structured
from test.api.causal_federation.test_bootstrap_runtime import _capability, _migrate
from test.api.test_repository_source_inventory import _repository, _retain, _git


SOURCE = ('VALUE = "' + '"\\\t\u03bb' * 300 + '"\n').encode()
PROMPT = "Read the admitted authored source without executing it."


@contextmanager
def _native_case(tmp_path, monkeypatch, *, bind_allocated=True, max_slice_bytes=4096):
    canonical = _repository(tmp_path, {b"module.py": SOURCE})
    retained = _retain(canonical, tmp_path, limits=sources.SourceInventoryLimits(frame_bytes=4096))
    workspace = tmp_path / "allocated"
    _git(canonical, "worktree", "add", "--detach", "-q", str(workspace))
    body = dict(task_id=retained.request.task_id, title="Read the authored source", read_paths=[b"module.py".hex()])
    task_cid = content_identity({"authored_task": body})
    server = build_server(database_path=tmp_path / "control.duckdb", state_dir=tmp_path / "owner",
        repository_id=retained.request.repository_id, store_id="source-worker-tests",
        transport=FakeQuackTransport(), capability_probe=_capability, migrate=_migrate,
        connection_factory=open_duckdb_connection, owner_liveness_probe=lambda _birth: OwnerLiveness.DEAD,
        typed_command_socket_path=tmp_path / "native.sock")
    server.start()
    session, control = None, None
    try:
        connection, gateway = server._connection, server._command_gateway
        connection.execute("INSERT INTO goals (goal_cid,goal_alias,objective_id,parent_goal_cid,ordinal,title,status,created_at,updated_at,revision,body_json) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            ["goal:source-worker", "SOURCE-GOAL", "objective:source-worker", "", 1, "Source lookup", "open",
             "1970-01-01T00:00:00Z", "1970-01-01T00:00:00Z", 1, "{}"])
        connection.execute("INSERT INTO tasks (task_cid,task_alias,goal_cid,plan_cid,objective_id,ordinal,status,revision,priority,created_at,updated_at,identity_json,body_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [task_cid, "SOURCE-TASK", "goal:source-worker", "", "objective:source-worker", 1, "ready", 3, "P0",
             "1970-01-01T00:00:00Z", "1970-01-01T00:00:00Z", "{}", canonical_json_bytes(body).decode()])
        config = dict(repository=workspace if bind_allocated else canonical, cas=retained.cas,
            expected_inventory_cid=retained.ref.inventory_cid, request=retained.request, limits=retained.limits,
            task_cid=task_cid, task_revision=3, expected_task_body_cid=content_identity(body),
            context_cid=cid_for_structured({"authored_context": task_cid}),
            allowed_paths=(b"module.py".hex(),), max_response_bytes=4096, max_slice_bytes=max_slice_bytes)
        binding = server.bind_current_source_lookup(**config)
        # Keep the session socket below Linux's pathname bound independently
        # of pytest's long native-owner artifact path.
        control = tempfile.TemporaryDirectory(prefix="slb-")
        boundary = Path(control.name) / "boundary.json"
        boundary.write_bytes(b"authored host deployment adapter\n")
        (boundary.parent / "worker-observations").mkdir(mode=0o750)
        boundary_sha = hashlib.sha256(boundary.read_bytes()).hexdigest()
        # No real worker UID is admitted by this adapter. Native proxy tests
        # stay owner-side; _answer must still reject the actual host peer UID.
        monkeypatch.setattr(guard, "_owner_boundary", lambda _path: (
            dict(owner_uid=os.getuid(), worker_uid=os.getuid() + 1), boundary_sha))
        monkeypatch.setattr(guard, "_protected_directory", lambda *args, **kwargs: None)
        factory = dict(server=server, canonical_repository=canonical, workspace=workspace,
            boundary=boundary, prompt=PROMPT, timeout_seconds=90)
        issued, issue = [], server.issue_typed_client_grant_record
        def recorded_issue(**kwargs):
            token, grant = issue(**kwargs)
            issued.append((token, grant))
            return token, grant
        monkeypatch.setattr(server, "issue_typed_client_grant_record", recorded_issue)
        if bind_allocated:
            session = broker.start_source_lookup_worker_broker(**factory)
        yield SimpleNamespace(canonical=canonical, workspace=workspace, retained=retained,
            cas=retained.cas, server=server, gateway=gateway, connection=connection,
            task_cid=task_cid, body=body, binding=binding, factory=factory, session=session, issued=issued)
    finally:
        try:
            if session is not None:
                session.close()
        finally:
            try:
                server.stop()
            finally:
                if control is not None:
                    control.cleanup()


@pytest.fixture
def native(tmp_path, monkeypatch, request):
    with _native_case(tmp_path, monkeypatch, max_slice_bytes=getattr(request, "param", 4096)) as case:
        yield case


def _payload(case, *, offset=0, maximum=None, continuation=None):
    return dict(schema="supervisor-current-source-request@1", binding_cid=case.binding["binding_cid"],
        raw_path_hex=b"module.py".hex(), source_cid=cid_for_bytes(SOURCE), offset=offset,
        max_bytes=case.binding["max_slice_bytes"] if maximum is None else maximum, continuation=continuation)


@contextmanager
def _actual_descendant():
    with tempfile.TemporaryDirectory(prefix="slp-") as temporary:
        path = Path(temporary) / "peer.sock"
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.bind(str(path))
        listener.listen(1)
        listener.settimeout(10)
        helper = "import socket,sys; s=socket.socket(socket.AF_UNIX); s.connect(sys.argv[1]); s.recv(1); s.close()"
        orchestrator = "import subprocess,sys; sys.stdin.readline(); subprocess.run([sys.executable,'-I','-c',sys.argv[2],sys.argv[1]],check=True)"
        process = subprocess.Popen([sys.executable, "-I", "-c", orchestrator, str(path), helper],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        channel = None
        try:
            root = protocol.process_identity(process.pid)
            process.stdin.write(b"go\n")
            process.stdin.flush()
            channel, _ = listener.accept()
            channel.settimeout(5)
            peer_pid, peer_uid, _ = struct.unpack("3i", channel.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
            yield SimpleNamespace(process=process, root=root, channel=channel, pid=peer_pid, uid=peer_uid)
        finally:
            if channel is not None:
                channel.sendall(b"x")
                channel.close()
            listener.close()
            if process.poll() is None:
                try:
                    process.communicate(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.communicate(timeout=5)


def test_real_socket_peer_is_an_observed_descendant_of_one_registered_birth():
    with _actual_descendant() as peer:
        observed = broker._require_descendant(peer.pid, uid=peer.uid,
            root_child=peer.root, owner_pid=os.getpid())
        assert observed == protocol.process_identity(peer.pid)
        assert observed["parent_pid"] == peer.process.pid and observed["uid"] == os.getuid()


@pytest.mark.parametrize("change", ["uid", "unrelated", "birth"])
def test_same_uid_parent_or_foreign_registered_birth_does_not_admit_helper(change):
    with _actual_descendant() as peer:
        uid, root = peer.uid, deepcopy(peer.root)
        if change == "uid":
            uid += 1
        elif change == "unrelated":
            root = protocol.process_identity(os.getpid())
        else:
            root["start_time_ticks"] += 1
        with pytest.raises(ValueError):
            broker._require_descendant(peer.pid, uid=uid, root_child=root, owner_pid=os.getpid())
        with pytest.raises(ValueError):
            broker._require_descendant(os.getpid(), uid=peer.uid,
                root_child=peer.root, owner_pid=os.getpid())


def _request(case, payload):
    return dict(schema=protocol.REQUEST_SCHEMA, session_id=case.session.descriptor["session_id"],
        request_id=uuid.uuid4().hex, payload=payload)


def _bytes(result):
    return (result["content"].encode("utf-8") if result["encoding"] == "utf-8"
            else base64.b64decode(result["content"], validate=True))


@pytest.mark.parametrize("native", [4096, 512], indirect=True, ids=["default-4096", "smaller-512"])
def test_owner_proxy_keeps_one_native_grant_and_exact_full_response_costs(native, monkeypatch):
    """This is an owner API read, never a claimed admitted worker read."""
    session, received = native.session, []
    assert session.descriptor["source_refs"] == [dict(raw_path_hex=b"module.py".hex(),
        source_cid=cid_for_bytes(SOURCE), size_bytes=len(SOURCE), git_mode="100644")]
    admitted_bound = native.binding["max_slice_bytes"]
    assert session.descriptor["max_slice_bytes"] == admitted_bound
    original = typed.TypedStateOwnerConnection.source_get
    def observed(self, payload):
        response = original(self, payload)
        if self is session._client:
            received.append(deepcopy(response))
        return response
    monkeypatch.setattr(typed.TypedStateOwnerConnection, "source_get", observed)
    monkeypatch.setattr(typed.TypedStateOwnerConnection, "__init__", lambda *a, **k: pytest.fail("proxy constructed a new native client"))
    monkeypatch.setattr(ImmutableCAS, "__init__", lambda *a, **k: pytest.fail("proxy constructed a new CAS"))
    payload, parts, costs = _payload(native), [], []
    for _ in range(64):
        result = session._read_source(payload)
        request = _request(native, payload)
        wrapper = session._response(request, result=result)
        protocol.validate_response(wrapper, descriptor=session.descriptor,
            payload=payload, request_id=request["request_id"])
        raw, backend = _bytes(result), received[-1]
        assert payload["max_bytes"] == admitted_bound and len(raw) <= admitted_bound
        assert result["slice_cid"] == cid_for_bytes(raw)
        assert result["request_cid"] == cid_for_structured(payload)
        assert result["backend_response_sha256"] == hashlib.sha256(canonical_json_bytes(backend)).hexdigest()
        assert result["backend_response_bytes"] == len(canonical_json_bytes(backend)) + 1
        assert result["backend_response_wire_bytes"] == result["backend_response_bytes"] + 3
        assert wrapper["response_bytes"] == len(protocol.canonical_json(wrapper)) + 1 <= session.descriptor["max_response_bytes"]
        assert wrapper["response_wire_bytes"] == wrapper["response_bytes"] + 3
        public = protocol.canonical_json(wrapper) + protocol.canonical_json(session.descriptor)
        assert "binding" not in result and "owner_identity" not in result and "caller_grant_id" not in result
        assert str(native.server.config.database_path).encode() not in public
        assert str(native.server.typed_command_socket_path()).encode() not in public
        assert all(public.find(token.encode()) == -1 for token, _ in native.issued)
        forged = deepcopy(wrapper)
        forged["result"]["owner_database_path"] = "/unadmitted/private-owner.duckdb"
        broker._count_response(forged)
        with pytest.raises(ValueError):
            protocol.validate_response(forged, descriptor=session.descriptor,
                payload=payload, request_id=request["request_id"])
        parts.append(raw)
        costs.append(result["backend_response_bytes"])
        if result["source_complete"]:
            break
        assert raw and result["next_continuation"] is not None
        payload = _payload(native, offset=result["end_offset"], continuation=result["next_continuation"])
    else:
        pytest.fail("bounded source continuation did not terminate")
    assert len(parts) > 1 and b"".join(parts) == SOURCE
    assert len(native.issued) == 1
    grant = native.issued[0][1]
    assert grant.allowed_operations == frozenset({"source.get"}) and grant.peer_pid == os.getpid()
    assert {item["result"]["caller_grant_id"] for item in received} == {grant.grant_id}
    assert session._native_response_bytes == sum(costs)
    assert session._native_wire_bytes == sum(size + 3 for size in costs)
    assert all(result[name] is False for name in
        ("repository_complete", "proof_authority", "completion_authority", "edit_authority", "omission_authority"))
    # Frontend scope refusal uses the real admitted public refs. Descriptor UID
    # admission is authored for this narrow pre-connect test, never _peer.
    with monkeypatch.context() as scope_patch:
        scope_patch.setattr(protocol, "load_descriptor", lambda *a, **k: deepcopy(session.descriptor))
        scope_patch.setattr(runner, "_source_lookup_executable", lambda _descriptor:
            (Path("/opt/ipfs-supervisor/bin/source-lookup"), "9" * 64))
        observation = runner._source_lookup_broker_observation(path=session.descriptor_path,
            digest=session.descriptor_sha256, workspace=native.workspace, prompt=PROMPT,
            boundary_sha256=session.descriptor["boundary_manifest_sha256"], formalization_receipt=None)
        assert observation["max_slice_bytes"] == admitted_bound
        assert '"max_bytes":' + str(min(4096, admitted_bound)) + "," in runner._source_lookup_guidance(observation)
        scope_patch.setattr(protocol.socket, "socket", lambda *a, **k: pytest.fail("foreign source pair opened a worker connection"))
        for field, value in (("source_cid", cid_for_bytes(b"foreign raw source")),
                             ("raw_path_hex", b"outside-admitted-scope.py".hex()),
                             ("max_bytes", admitted_bound + 1)):
            foreign = _payload(native)
            foreign[field] = value
            with pytest.raises(ValueError):
                protocol.request_source(path=session.descriptor_path,
                    expected_sha256=session.descriptor_sha256, payload=foreign)
        for invalid in (True, 0, 16385):
            descriptor = deepcopy(session.descriptor)
            descriptor["max_slice_bytes"] = invalid
            with pytest.raises(ValueError):
                protocol.validate_descriptor(descriptor)


@pytest.mark.parametrize("root", ["canonical", "workspace"])
def test_proxy_refuses_either_checkout_overlay_before_returning_source(native, root):
    (getattr(native, root) / "module.py").write_bytes(b"changed authored overlay\n")
    with pytest.raises((ValueError, RuntimeError)):
        native.session._read_source(_payload(native))


def test_allocated_edit_after_verified_native_reply_cannot_escape_proxy(native, monkeypatch):
    original, calls = typed.TypedStateOwnerConnection.source_get, []
    def edited(self, payload):
        response = original(self, payload)
        if self is native.session._client:
            calls.append((response["result"]["response_bytes"], response["result"]["response_wire_bytes"]))
            (native.workspace / "module.py").write_bytes(b"edit after native response\n")
        return response
    monkeypatch.setattr(typed.TypedStateOwnerConnection, "source_get", edited)
    with pytest.raises((ValueError, RuntimeError)):
        native.session._read_source(_payload(native))
    # Refusal at the worker boundary cannot erase an already received native
    # success envelope from the backend cost ledger.
    assert len(calls) == 1 and native.session._native_response_bytes == calls[0][0] > 0
    assert native.session._native_wire_bytes == calls[0][1]


def test_canonical_gateway_binding_cannot_be_silently_rebound_to_allocation(tmp_path, monkeypatch):
    with _native_case(tmp_path, monkeypatch, bind_allocated=False) as case:
        original = case.gateway._current_source_lookup
        with pytest.raises(ValueError, match="allocation"):
            broker.start_source_lookup_worker_broker(**case.factory)
        assert case.gateway._current_source_lookup is original
        assert original._root == case.canonical and case.issued == []


def test_real_host_socket_uid_cannot_use_public_descriptor_as_worker_authority(native, monkeypatch):
    process = subprocess.Popen([sys.executable, "-I", "-c", "import time; time.sleep(30)"])
    try:
        native.session.note_spawned(process)
        with pytest.raises(ValueError):
            native.session.note_spawned(process)
        monkeypatch.setattr(native.session, "_read_source", lambda payload: pytest.fail("unrelated host peer reached source proxy"))
        left, right = socket.socketpair()
        with left, right:
            request = _request(native, _payload(native))
            response = native.session._answer(request, right)
        assert response["ok"] is False and response["result"] is None
        assert response["reason_code"] == "peer_refused"
        protocol.validate_response(response, descriptor=native.session.descriptor,
            payload=request["payload"], request_id=request["request_id"])
    finally:
        process.terminate()
        process.wait(timeout=5)


def test_session_close_revokes_original_native_grant_and_removes_channel(native):
    session = native.session
    session._read_source(_payload(native, maximum=1))
    grant_id = session._grant.grant_id
    session.close()
    session.close()
    assert grant_id in native.gateway._revoked_grants
    assert not session.socket_path.exists() and not session.descriptor_path.exists()
    audit = json.loads(session.audit_path.read_bytes())
    assert audit["native_proxy_grant_revoked"] is audit["service_stopped"] is audit["accounting_complete"] is True
    assert audit["accepted_requests"] == 0 and audit["requests"] == []
    assert audit["backend_response_bytes"] > 0 and audit["retrieval_provider_calls"] == 0
    assert all(audit[name] is False for name in ("proof_authority", "execution_authority", "completion_authority"))
    public = canonical_json_bytes(audit)
    assert all(public.find(token.encode()) == -1 for token, _ in native.issued)
    with pytest.raises((ValueError, RuntimeError)):
        session._read_source(_payload(native, maximum=1))


def test_source_worker_transport_preserves_large_canonical_body_and_refuses_header_before_body():
    # This is framing qualification, not an authenticated source response.
    body = {"authored": "\u03bb" * 6000}
    assert 8192 < len(protocol.canonical_json(body)) < protocol.MAX_RESPONSE_BYTES
    left, right = socket.socketpair()
    with left, right:
        protocol.send_frame(left, body)
        assert protocol.receive_frame(right, max_bytes=protocol.MAX_RESPONSE_BYTES) == body
        left.sendall((protocol.MAX_FRAME_BYTES + 1).to_bytes(4, "big"))
        right.settimeout(0.5)
        with pytest.raises(ValueError):
            protocol.receive_frame(right)
