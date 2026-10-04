"""Native claim fencing and bounded Unix protocol controls.

These host tests qualify the real typed claim/grant validator and receiving
refusals. They do not fabricate a successful proof broker or isolated UID1001
worker. The two-hop live broker/worker transition is qualified separately by
the genuine container experiment.
"""
from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import socket
import tempfile

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import finite_proof_query_worker_dispatch as dispatch
from ipfs_accelerate_py.agent_supervisor.runtime import finite_repository_execution as execution
from ipfs_accelerate_py.agent_supervisor.task_sources import typed_state_owner as typed
from ipfs_accelerate_py.agent_supervisor.todo_daemon import supervisor_runtime
from test.api.test_finite_integer_codebase import finite_tools  # noqa: F401
from test.api.test_finite_proof_query_worker_context import _native_proof_query_worker_case
from test.api.test_finite_repository_execution import _daemon


def _inert_request():
    return {"schema": dispatch.SCHEMA, "operation": "prepare", "context_cid": "cid:inert-context",
        "command": ["/literal/owner-worker", "--finite-proof-query-context", "/literal/context.json",
                    "--finite-proof-query-sha256", "a" * 64,
                    "--finite-proof-query-context-cid", "cid:inert-context"],
        "worktree": "/literal/worktree", "task_id": "OFFSET-TASK",
        "database_task_cid": "cid:inert-task", "database_attempt_id": "attempt:inert",
        "database_claim_id": "claim:inert", "database_attempt_number": 1}


@pytest.fixture(scope="module")
def native_dispatch_claim_case(tmp_path_factory, finite_tools):
    with _native_proof_query_worker_case(
            tmp_path_factory.mktemp("finite-proof-query-dispatch-claim") / "native", finite_tools) as case:
        server = case["native"].server
        with server._lock:
            population = execution._physical_native(server._connection)
            population["selected_task_cids"] = [case["offset_cid"]]
        driver = _daemon(case, "actual-dispatch-claim")
        try:
            attempt = driver.claim_next()
            assert attempt is not None and attempt.task_cid == case["offset_cid"]
            task = case["native"].source.get_task(attempt.task_cid)
            receipt = dict(task.body["completion_receipt"])
            assert task.status == "in_progress"
            assert receipt["operation"] == "database_attempt_admitted"
            request = _inert_request()
            request.update(database_task_cid=attempt.task_cid, database_attempt_id=attempt.attempt_id,
                database_claim_id=attempt.claim_id, database_attempt_number=receipt["attempt_number"],
                context_cid=case["worker_context"]["context_cid"],
                command=["/literal/owner-worker", "--finite-proof-query-context",
                         case["worker_context"]["artifact"], "--finite-proof-query-sha256",
                         case["worker_context"]["sha256"], "--finite-proof-query-context-cid",
                         case["worker_context"]["context_cid"]])
            # SO_PEERCRED is actually obtained from the kernel, not supplied by
            # a fake transport or a caller-authored successful attestation.
            left, right = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                peer = typed._kernel_peer_identity(left)
            finally:
                left.close()
                right.close()
            case.update(dispatch_population=population, dispatch_request=request, dispatch_peer=peer,
                        dispatch_attempt=attempt, dispatch_receipt=receipt)
            yield case
        finally:
            driver.close()


def _claim(case, *, request=None, peer=None):
    native = case["native"]
    with native.server._lock:
        return dispatch._require_native_dispatch_claim(server=native.server,
            expected_population=case["dispatch_population"],
            request=case["dispatch_request"] if request is None else request,
            peer_identity=case["dispatch_peer"] if peer is None else peer,
            client_id=native.credentials.client_id, store_id=native.identity.store_id)


def test_actual_native_admitted_claim_matches_kernel_peer_and_original_population(native_dispatch_claim_case):
    case = native_dispatch_claim_case
    result = _claim(case)
    assert result["task_cid"] == case["offset_cid"]
    assert result["attempt_id"] == case["dispatch_attempt"].attempt_id
    assert result["claim_id"] == case["dispatch_attempt"].claim_id
    assert result["peer_pid"] == os.getpid()
    assert result["peer_uid"] == os.geteuid()
    assert result["peer_start_time_ticks"] == typed._process_start_time_ticks(os.getpid())
    assert result["grant_id"] == case["dispatch_receipt"]["claim_process_attestation"]["grant_id"]
    assert result["task_revision"] == case["candidate"]["task_revision"] + 2
    # This is a claim validator result, not a successful sealed runtime/proof gate.
    assert "proof_authority" not in result and "convergence_proved" not in result


@pytest.mark.parametrize("field", ["database_task_cid", "task_id", "database_attempt_id",
                                  "database_claim_id", "database_attempt_number"])
def test_changed_native_binding_scalar_refuses(native_dispatch_claim_case, field):
    case = native_dispatch_claim_case
    request = deepcopy(case["dispatch_request"])
    request[field] = request[field] + 1 if field == "database_attempt_number" else "unbound:changed"
    with pytest.raises((ValueError, RuntimeError)):
        _claim(case, request=request)
    assert _claim(case)["claim_id"] == case["dispatch_attempt"].claim_id


@pytest.mark.parametrize("field", [0, 1, 2])
def test_claim_grant_refuses_wrong_pid_uid_or_start_ticks(native_dispatch_claim_case, field):
    case = native_dispatch_claim_case
    peer = list(case["dispatch_peer"])
    peer[field] += 1
    with pytest.raises((ValueError, RuntimeError), match="peer identity"):
        _claim(case, peer=tuple(peer))


def test_real_native_grant_revocation_refuses_same_claim(native_dispatch_claim_case):
    case = native_dispatch_claim_case
    gateway = case["native"].server._command_gateway
    grant_id = _claim(case)["grant_id"]
    with gateway._grants_lock:
        assert grant_id not in gateway._revoked_grants
        gateway._revoked_grants.add(grant_id)
    try:
        with pytest.raises((ValueError, RuntimeError), match="revoked"):
            _claim(case)
    finally:
        # Restore only this deliberate negative control, not the private token table.
        with gateway._grants_lock:
            gateway._revoked_grants.discard(grant_id)
    assert _claim(case)["grant_id"] == grant_id


@pytest.mark.parametrize("mutation", ["status", "revision", "body", "claim-birth", "claim-phase"])
def test_detached_current_native_row_corruption_refuses(native_dispatch_claim_case, mutation):
    case = native_dispatch_claim_case
    server, cid = case["native"].server, case["offset_cid"]
    with server._lock:
        before_row = server._connection.execute(
            "SELECT status,revision,body_json FROM tasks WHERE task_cid=?", [cid]).fetchone()
        # Native DuckDBRow iteration yields field names; positional access
        # preserves exact original scalar values for mutation and restoration.
        before = tuple(before_row[position] for position in range(3))
        body = json.loads(before[2])
        changed = list(before)
        if mutation == "status":
            changed[0] = "completed"
        elif mutation == "revision":
            changed[1] += 1
        elif mutation == "body":
            body["extra_output_semantics"] = "changed"
        elif mutation == "claim-birth":
            body["completion_receipt"]["claim_process_attestation"]["process_birth_id"] = "birth:changed"
        else:
            body["completion_receipt"]["attempt_execution_phase"] = "provider_running"
        changed[2] = json.dumps(body, sort_keys=True)
        server._connection.execute("UPDATE tasks SET status=?,revision=?,body_json=? WHERE task_cid=?",
                                   [*changed, cid])
    try:
        with pytest.raises((ValueError, RuntimeError)):
            _claim(case)
    finally:
        with server._lock:
            server._connection.execute("UPDATE tasks SET status=?,revision=?,body_json=? WHERE task_cid=?",
                                       [*before, cid])
    assert _claim(case)["claim_id"] == case["dispatch_attempt"].claim_id


def test_full_prerequisite_native_completion_rows_remain_required(native_dispatch_claim_case):
    case = native_dispatch_claim_case
    server, cid = case["native"].server, case["type_cid"]
    with server._lock:
        native_rows = server._connection.execute("SELECT * FROM completion_receipts WHERE task_cid=?", [cid]).fetchall()
        rows = [tuple(row[position] for position in range(len(row))) for row in native_rows]
        assert rows
        server._connection.execute("DELETE FROM completion_receipts WHERE task_cid=?", [cid])
    try:
        with pytest.raises(dispatch.FiniteProofQueryDispatchError, match="prerequisite population"):
            _claim(case)
    finally:
        with server._lock:
            server._connection.executemany("INSERT INTO completion_receipts VALUES ("
                + ",".join("?" for _ in rows[0]) + ")", [tuple(row) for row in rows])
    assert _claim(case)["claim_id"] == case["dispatch_attempt"].claim_id


def test_closed_inert_unix_frames_preserve_exact_data_and_kernel_peer():
    left, right = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        request = _inert_request()
        dispatch._send(left, request)
        assert dispatch._request(dispatch._receive(right)) == request
        pid, uid, ticks = typed._kernel_peer_identity(right)
        assert (pid, uid, ticks) == (os.getpid(), os.geteuid(), typed._process_start_time_ticks(os.getpid()))
    finally:
        left.close()
        right.close()


@pytest.mark.parametrize("raw", [b'{"schema":"a","schema":"b"}', b'{"value":NaN}', b'[]'])
def test_duplicate_nonfinite_or_nonobject_wire_frames_refuse(raw):
    left, right = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        left.sendall(len(raw).to_bytes(4, "big") + raw)
        with pytest.raises(dispatch.FiniteProofQueryDispatchError):
            dispatch._receive(right)
    finally:
        left.close()
        right.close()


@pytest.mark.parametrize("field,value", [("owner_key", "forbidden"), ("sql", "SELECT 1"),
                                       ("callback", "execute"), ("peer_pid", 1),
                                       ("database_attempt_number", True)])
def test_request_accepts_only_closed_native_binding_scalars(field, value):
    request = _inert_request()
    request[field] = value
    with pytest.raises(dispatch.FiniteProofQueryDispatchError):
        dispatch._request(request)


def test_incomplete_context_command_refuses_at_literal_popen_boundary(tmp_path, monkeypatch):
    reached = []
    def forbidden_popen(*args, **kwargs):
        reached.append(True)
        pytest.fail("mandatory context refusal reached literal Popen")
    monkeypatch.setattr(supervisor_runtime.subprocess, "Popen", forbidden_popen)
    with pytest.raises(dispatch.FiniteProofQueryDispatchError):
        supervisor_runtime.launch_process_child(
            ["/literal/owner-worker", "--finite-proof-query-context", "/literal/context.json"],
            cwd=tmp_path, env={}, inherit_environment=False)
    assert not reached


def test_missing_native_binding_cannot_use_complete_context(tmp_path):
    with pytest.raises(dispatch.FiniteProofQueryDispatchError, match="native database attempt authority"):
        dispatch.require_finite_proof_query_worker_dispatch(command=_inert_request()["command"],
            worktree=tmp_path, environment={}, dispatch_binding=None)


def test_owner_socket_group_access_refuses_before_connection(tmp_path):
    # Retained pytest paths exceed Linux's Unix socket pathname bound. Use an
    # actual short owner temporary endpoint and retain its control metadata.
    control = {"schema": "inert-unix-socket-permission-control@1",
               "scope": "protocol-only; no native proof owner or worker launch", "refused": False}
    with tempfile.TemporaryDirectory(prefix="pq-socket-", dir="/tmp") as short:
        endpoint = Path(short) / "dispatch.sock"
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            listener.bind(str(endpoint))
            endpoint.chmod(0o660)
            before = endpoint.lstat()
            control.update(path=str(endpoint), before_mode=before.st_mode & 0o777,
                           before_uid=before.st_uid, before_dev=before.st_dev, before_inode=before.st_ino)
            with pytest.raises(dispatch.FiniteProofQueryDispatchError, match="private owner dispatch socket"):
                dispatch.require_finite_proof_query_worker_dispatch(command=_inert_request()["command"],
                    worktree=tmp_path, environment={dispatch.SOCKET_ENV: str(endpoint),
                        dispatch.CONTEXT_ENV: "cid:inert-context"}, dispatch_binding=_inert_request())
            control["refused"] = True
        finally:
            listener.close()
            endpoint.unlink(missing_ok=True)
            control["socket_removed"] = not endpoint.exists()
    control["temporary_directory_removed"] = not Path(short).exists()
    (tmp_path / "unix-socket-permission-control.json").write_text(
        json.dumps(control, sort_keys=True, indent=2) + "\n")


def test_serialized_scope_and_runtime_cannot_create_broker():
    with pytest.raises(dispatch.FiniteProofQueryDispatchError, match="exact bound native runtime"):
        dispatch.start_owner_finite_proof_query_dispatch_broker(scope={"scope": "inert"}, runtime={})


def test_unsealed_or_subclass_broker_cannot_install_transport():
    class Substitute(dispatch.OwnerFiniteProofQueryDispatchBroker):
        pass
    with pytest.raises(dispatch.FiniteProofQueryDispatchError):
        Substitute(object(), scope=None, runtime=None)
    with pytest.raises(dispatch.FiniteProofQueryDispatchError):
        dispatch.OwnerFiniteProofQueryDispatchBroker(object(), scope=None, runtime=None)


def test_isolated_worker_cannot_use_daemon_binding_without_pid_bound_wrapper_ticket(tmp_path):
    command = ["/usr/bin/sudo", "-n", "-u", "benchmarkworker", "--", "/literal/worker-entry",
               *_inert_request()["command"][1:]]
    with pytest.raises(dispatch.FiniteProofQueryDispatchError):
        dispatch.require_finite_proof_query_isolated_worker_dispatch(
            command=command, worktree=tmp_path, environment={dispatch.CONTEXT_ENV: "cid:inert-context"})


def test_ordinary_command_uses_no_new_dispatch_capability(tmp_path):
    assert dispatch.require_finite_proof_query_worker_dispatch(
        command=["/literal/ordinary"], worktree=tmp_path, environment={}, dispatch_binding=None) is None
    assert dispatch.require_finite_proof_query_isolated_worker_dispatch(
        command=["/literal/ordinary"], worktree=tmp_path, environment={}) is None



def assert_genuine_broker_private_custody_controls(broker):
    """Custody controls on an actual constructed, unlaunched native broker.

    This helper requires the genuine native runtime factory result. A host
    constructor test may supply explicit deployment launcher and sudo fixtures,
    but must not fabricate the runtime, proof owner or successful checker.
    Record these controls separately from actual Docker execution, worker birth,
    formal proof or authenticated process-origin claims.
    """
    assert type(broker) is dispatch.OwnerFiniteProofQueryDispatchBroker
    assert not broker._scope._spawned and not broker._spawned and not broker._pending
    original_environment = broker.daemon_environment()
    with broker._server._lock:
        population = execution._physical_native(broker._server._connection)
    receipts = []
    original_fields = broker._fields
    try:
        broker._fields = ()
        with pytest.raises(dispatch.FiniteProofQueryDispatchError, match="inputs were rebound"):
            broker.daemon_environment()
        receipts.append({"control": "truncated-native-fields", "refused_before_birth": True})
    finally:
        broker._fields = original_fields
    assert broker.daemon_environment() == original_environment
    for field in ("_population", "_command", "_context", "socket_path"):
        original = getattr(broker, field)
        replacement = deepcopy(original)
        if field == "_population":
            replacement["selected_task_cids"] = ["unbound:changed"]
        elif field == "_command":
            replacement.append("--unbound")
        elif field == "_context":
            replacement = "unbound:changed-context"
        else:
            replacement = original.with_name("unbound-alias.sock")
        try:
            setattr(broker, field, replacement)
            with pytest.raises(dispatch.FiniteProofQueryDispatchError, match="cached signed inputs"):
                broker.daemon_environment()
            receipts.append({"control": field, "refused_before_birth": True})
        finally:
            setattr(broker, field, original)
        assert broker.daemon_environment() == original_environment
    endpoint, state = broker.socket_path, Path(broker._runtime.state)
    for target, mode, control in ((endpoint, 0o660, "socket-mode"), (state, 0o750, "state-mode")):
        original_mode = target.stat().st_mode & 0o777
        try:
            target.chmod(mode)
            with pytest.raises(dispatch.FiniteProofQueryDispatchError):
                broker.daemon_environment()
            receipts.append({"control": control, "refused_before_birth": True})
        finally:
            target.chmod(original_mode)
        assert broker.daemon_environment() == original_environment
    displaced = endpoint.with_name(endpoint.name + ".retained-original")
    replacement_listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    original_inode = (endpoint.stat().st_dev, endpoint.stat().st_ino)
    endpoint.rename(displaced)
    try:
        replacement_listener.bind(str(endpoint))
        endpoint.chmod(0o600)
        assert (endpoint.stat().st_dev, endpoint.stat().st_ino) != original_inode
        with pytest.raises(dispatch.FiniteProofQueryDispatchError, match="socket inode"):
            broker.daemon_environment()
        receipts.append({"control": "socket-inode-replacement", "refused_before_birth": True})
    finally:
        replacement_listener.close()
        endpoint.unlink(missing_ok=True)
        displaced.rename(endpoint)
    assert (endpoint.stat().st_dev, endpoint.stat().st_ino) == original_inode
    assert broker.daemon_environment() == original_environment
    with broker._server._lock:
        assert execution._physical_native(broker._server._connection) == population
    assert not broker._scope._spawned and not broker._spawned and not broker._pending
    return {"schema": dispatch.SCHEMA, "controls": receipts, "actual_broker_constructed": True,
            "all_original_bindings_restored": True, "native_task_population_unchanged": True,
            "worker_birth_qualification": False, "authenticated_process_origin": False,
            "proof_authority": False, "convergence_proved": False}
