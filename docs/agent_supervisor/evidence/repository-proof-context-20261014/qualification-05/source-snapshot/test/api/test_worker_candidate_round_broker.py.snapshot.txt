"""Failed child -> owner snapshot -> fresh candidate-only source session.

Native Git, source capture, retained capsules/formulas, processes and sockets run
normally. The UID seam is authored; this does not qualify Docker isolation or a
provider continuation. No model, solver or benchmark task is invoked.
"""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
import socket
import struct
import subprocess
import sys
import time
from types import SimpleNamespace
import uuid

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import worker_candidate_round_broker as module
from ipfs_accelerate_py.agent_supervisor.runtime import worker_candidate_round_client as protocol
from ipfs_accelerate_py.agent_supervisor.runtime.owner_candidate_round_context import prepare_candidate_round_context
from test.api.test_owner_candidate_round_context import source_service, native, SOURCE, CHANGED  # noqa: F401


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture
def case(source_service):
    script = ("import sys;from pathlib import Path;sys.stdin.read();"
        "Path('mod.py').write_bytes(bytes.fromhex(sys.argv[1]));"
        "p=Path('harness.py');p.write_bytes(p.read_bytes()+b'\\n#'+b'x'*12000+b'\\n');"
        "raise SystemExit(7)")
    child = subprocess.Popen([sys.executable, "-c", script, CHANGED.hex()],
        cwd=source_service.workspace, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL)
    source_service.note_spawned(child)
    child.communicate(b"authored edit", timeout=10)
    assert child.returncode == 7
    source_service.close()
    deadline = time.monotonic() + 120
    snapshot = prepare_candidate_round_context(source_broker=source_service,
        prior_returncode=7, round_number=2, absolute_deadline=deadline)
    services = []
    def new():
        service = module.WorkerCandidateRoundBroker(snapshot)
        services.append(service)
        return service
    yield SimpleNamespace(original=source_service, snapshot=snapshot, new=new,
        generation=snapshot.metadata["workspace_generation"], deadline=deadline)
    for service in services:
        if service._process is not None and service._process.poll() is None:
            service._process.kill()
            service._process.wait()
        service.close()


def request(service, *, artifact=None, **kwargs):
    if artifact is None:
        query = dict(path="mod.py", view="original", cursor=None, max_bytes=8192,
            expected_source_sha256=None, expected_view_sha256=None,
            source_state="working-candidate", expected_workspace_generation=service._generation)
        schema = protocol.REQUEST_SCHEMA_V2
    else:
        query = dict(artifact=artifact, cursor=None, max_bytes=8192,
            expected_workspace_generation=service._generation)
        schema = "supervisor-worker-candidate-context-request@1"
    query.update(kwargs)
    return protocol.validate_request(dict(schema=schema, session_id=service.descriptor["session_id"],
        request_id=uuid.uuid4().hex, query=query))


def answer(service, query=None, *, delivered=True, **kwargs):
    query = request(service, **kwargs) if query is None else query
    response = service._answer(query, None)
    protocol.validate_response(response, request=query, descriptor=service.descriptor)
    service._response_finished(query, response, delivered=delivered)
    return response


def body(page):
    return page["content"].encode() if page["encoding"] == "utf-8" else base64.b64decode(page["content"])


def artifact(service, name):
    parts, cursor = [], None
    for _ in range(64):
        response = answer(service, artifact=name, cursor=cursor)
        assert response["ok"], response
        parts.append(body(response["page"]))
        cursor = response["page"]["next_cursor"]
        if cursor is None:
            raw = b"".join(parts)
            assert sha(raw) == response["page"]["artifact_sha256"]
            return raw
    pytest.fail("candidate artifact paging exceeded bound")


def test_new_session_has_separate_candidate_binding_and_keeps_baseline_stale(case, monkeypatch):
    service = case.new()
    monkeypatch.setattr(service, "_peer", lambda channel: None)
    descriptor = service.descriptor
    assert descriptor["schema"] == protocol.DESCRIPTOR_SCHEMA
    assert descriptor["session_id"] != case.original.descriptor["session_id"]
    assert service.descriptor_path != case.original.descriptor_path
    assert service.socket_path != case.original.socket_path
    assert descriptor["prior_session_id"] == case.original.descriptor["session_id"]
    assert descriptor["prior_descriptor_sha256"] == case.original.descriptor_sha256
    assert descriptor["round_context_sha256"] == sha(case.snapshot.metadata_bytes)
    assert descriptor["candidate_semantic_root_cid"] == case.snapshot.metadata["semantic_root_cid"]
    assert descriptor["semantic_root_cid"] == case.original.descriptor["semantic_root_cid"]
    assert descriptor["expires_monotonic_ns"] <= int(case.deadline*1_000_000_000)
    assert service.views is case.original.views
    assert service._process is None and not service._registered.is_set()
    assert answer(service)["page"]["source_sha256"] == sha(CHANGED)
    assert body(answer(service)["page"]) == CHANGED
    assert (service.repository / "mod.py").read_bytes() == SOURCE
    with pytest.raises(ValueError, match="stale"):
        service.views.admission()
    with pytest.raises(ValueError):
        protocol.source.validate_descriptor(descriptor)
    with pytest.raises(ValueError, match="baseline"):
        service._current_context(working=False)
    with pytest.raises(ValueError):
        service._current_context(working=True, generation="a" * 64)


def test_serves_retained_native_artifacts_without_rebuilding_or_promoting(case, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.runtime import worker_candidate_artifacts as native_artifacts
    service = case.new()
    monkeypatch.setattr(service, "_peer", lambda channel: None)
    monkeypatch.setattr(native_artifacts, "build_candidate_artifacts",
        lambda **kwargs: pytest.fail("candidate session rebuilt retained native artifacts"))
    for name, raw in case.snapshot.artifacts.items():
        assert artifact(service, name) == raw
    formal = json.loads(case.snapshot.artifacts["formal"])
    report = formal["reports"]["mod.py"]
    definitions = {row["definition"] for row in report["projections"]["lean"]["bindings"]}
    assert "SelectionMin" in definitions and "SelectionMax" not in definitions
    assert formal["proof_authority"] is formal["completion_authority"] is False
    assert formal["source_obligations_discharged"] == 0
    assert formal["program_paths"] == ["mod.py"]
    assert "harness.py" not in formal["reports"]


def test_source_cursors_are_bound_to_delivery_and_this_fresh_session(case, monkeypatch):
    first, second = case.new(), case.new()
    for service in (first, second):
        monkeypatch.setattr(service, "_peer", lambda channel: None)
    probe = request(first, path="harness.py", max_bytes=2400)
    native_budget = 2400 - (len(protocol.canonical_json(first._response(probe, page={}))) - 2)
    prior = case.original.views.working_page(path="harness.py", view="original", max_bytes=native_budget,
        expected_workspace_generation=case.generation)
    assert prior["next_cursor"] is not None
    old_query = request(first, path="harness.py", max_bytes=2400, cursor=prior["next_cursor"])
    assert answer(first, old_query)["reason_code"] == "invalid_request"
    # The old cursor is valid in the retained view for this exact page budget;
    # its refusal above comes from the new session boundary, not a query typo.
    old_continuation = case.original.views.working_page(path="harness.py", view="original",
        max_bytes=native_budget, expected_workspace_generation=case.generation,
        cursor=prior["next_cursor"])
    assert old_continuation["offset"] == prior["end_offset"]
    page = answer(first, path="harness.py", max_bytes=2400)["page"]
    cursor = page["next_cursor"]
    assert cursor is not None
    assert answer(second, path="harness.py", max_bytes=2400, cursor=cursor)["reason_code"] == "invalid_request"
    continuation = request(first, path="harness.py", max_bytes=2400, cursor=cursor)
    advanced = answer(first, continuation)
    assert advanced["ok"] and advanced["page"]["offset"] == page["end_offset"]
    assert answer(first, path="harness.py", max_bytes=2400, cursor=cursor)["reason_code"] == "invalid_request"
    undelivered = answer(second, path="harness.py", max_bytes=2400, delivered=False)["page"]["next_cursor"]
    assert undelivered is not None
    assert answer(second, path="harness.py", max_bytes=2400, cursor=undelivered)["reason_code"] == "invalid_request"


def test_candidate_artifact_cursors_cannot_cross_new_sessions(case, monkeypatch):
    first, second = case.new(), case.new()
    for service in (first, second):
        monkeypatch.setattr(service, "_peer", lambda channel: None)
    page = answer(first, artifact="summary", max_bytes=2400)["page"]
    assert page["next_cursor"] is not None
    assert answer(second, artifact="summary", max_bytes=2400,
        cursor=page["next_cursor"])["ok"] is False
    assert answer(first, artifact="summary", max_bytes=2400, cursor=page["next_cursor"])["ok"] is True


@pytest.mark.parametrize("kind", ["foreign_generation", "unpinned", "baseline", "discovery", "edit", "semantic"])
def test_server_refuses_other_contracts_before_source_work(case, monkeypatch, kind):
    service = case.new()
    query = request(service)
    if kind == "foreign_generation":
        query["query"]["expected_workspace_generation"] = "0" * 64
    elif kind == "unpinned":
        query["query"]["expected_workspace_generation"] = None
    elif kind == "baseline":
        query["query"]["source_state"] = "baseline"
    elif kind == "discovery":
        query["schema"] = protocol.REQUEST_SCHEMA_V3
        query["query"] = dict(operation="discover", cursor=None, max_bytes=8192, view_policy_binding_sha256=None)
    else:
        query["schema"] = ("supervisor-worker-source-edit-request@1" if kind == "edit"
                            else "supervisor-worker-semantic-evidence-request@1")
    monkeypatch.setattr(service, "_peer", lambda *_: pytest.fail("invalid request reached child admission"))
    monkeypatch.setattr(service.views, "working_page", lambda **_: pytest.fail("invalid request reached source"))
    assert service._answer(query, None)["reason_code"] == "invalid_request"


@pytest.mark.parametrize("target", ["candidate", "canonical", "support", "deleted", "producer", "descriptor"])
def test_retained_snapshot_drift_refuses_delivery(case, monkeypatch, target):
    service = case.new()
    monkeypatch.setattr(service, "_peer", lambda channel: None)
    assert answer(service)["ok"]
    if target in ("candidate", "canonical", "support", "deleted"):
        root = service.repository if target == "canonical" else service.workspace
        path = root / ("harness.py" if target == "support" else "mod.py")
        if target == "deleted":
            path.unlink()
        else:
            path.write_bytes(path.read_bytes()+b"\n# changed since source admission\n")
    elif target == "producer":
        from ipfs_accelerate_py.agent_supervisor.runtime import owner_candidate_round_context as snapshot_module
        monkeypatch.setattr(snapshot_module, "_producers", lambda: b'{"changed":"producer"}')
    else:
        service.descriptor["candidate_scope_cid"] += "changed"
    response = service._answer(request(service), None)
    assert response["ok"] is False and response["page"] is None


@pytest.mark.parametrize("serialized", [None, {}, b"{}", "checkpoint.json", Path("checkpoint.json")])
def test_checkpoint_metadata_is_not_owner_admission(serialized):
    with pytest.raises(ValueError, match="exact retained owner"):
        module.WorkerCandidateRoundBroker(serialized)


def test_expiring_budget_does_not_create_session_or_refresh_deadline(case, monkeypatch):
    paths = set(case.original.parent.iterdir())
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: case.deadline - .5))
    with pytest.raises(ValueError, match="one second"):
        case.new()
    assert set(case.original.parent.iterdir()) == paths


def test_distant_finite_deadline_does_not_overflow_or_extend_native_session(case):
    snapshot = prepare_candidate_round_context(source_broker=case.original,
        prior_returncode=7, round_number=2, absolute_deadline=sys.float_info.max)
    service = module.WorkerCandidateRoundBroker(snapshot)
    try:
        assert service._deadline_ns == (1 << 63) - 1
        assert (service.descriptor["expires_monotonic_ns"] - service.descriptor["issued_monotonic_ns"]
            == module.MAX_SESSION_SECONDS * 1_000_000_000)
    finally:
        service.close()


def test_session_close_cleans_transport_and_preserves_advisory_accounting(case, monkeypatch):
    service = case.new()
    monkeypatch.setattr(service, "_peer", lambda channel: None)
    query = request(service, artifact="summary", max_bytes=2400)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as channel:
        channel.settimeout(30)
        channel.connect(str(service.socket_path))
        protocol.send_frame(channel, query)
        response = protocol.receive_frame(channel)
    assert protocol.validate_response(response, request=query, descriptor=service.descriptor)["ok"]
    service.close()
    assert not service.descriptor_path.exists() and not service.socket_path.exists()
    audit = json.loads(service.audit_path.read_bytes())
    assert audit["schema"] == "supervisor-worker-candidate-round-session@1"
    assert audit["service_stopped"] is audit["accounting_complete"] is True
    assert audit["accepted_requests"] == len(audit["requests"]) == 1
    assert audit["requests"][0]["candidate_context_page"] is True
    assert audit["retrieval_provider_calls"] == audit["source_obligations_discharged"] == 0
    assert audit["proof_authority"] is audit["completion_authority"] is False
    assert service._source_continuations == {}


def test_real_registered_descendant_retrieves_fresh_source_and_candidate_summary(case, monkeypatch):
    service = case.new()
    # Author only the UID distinction. Real kernel peer PID, birth times, parent
    # chain, socket framing, child registration, wait and cleanup remain active.
    original_identity, original_peer = protocol.process_identity, service._peer
    owner_pid = module.os.getpid()
    def identity(pid):
        actual = original_identity(pid)
        return actual if pid == owner_pid else {**actual, "uid": service.worker_uid}
    monkeypatch.setattr(protocol, "process_identity", identity)
    class ChannelUID:
        def __init__(self, channel):
            self.channel = channel
        def getsockopt(self, *args):
            pid, uid, gid = struct.unpack("3i", self.channel.getsockopt(*args))
            return struct.pack("3i", pid, service.worker_uid, gid)
    monkeypatch.setattr(service, "_peer", lambda channel: original_peer(ChannelUID(channel)))
    script = r'''
import importlib.util,json,socket,sys
spec=importlib.util.spec_from_file_location('authored_candidate_client',sys.argv[1])
client=importlib.util.module_from_spec(spec);spec.loader.exec_module(client)
requests=json.loads(sys.stdin.read()); responses=[]
for request in requests:
    with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as channel:
        channel.settimeout(30);channel.connect(sys.argv[2]);client.send_frame(channel,request)
        responses.append(client.receive_frame(channel))
print(json.dumps(responses))
'''
    requests = [request(service), request(service, artifact="summary")]
    child = subprocess.Popen([sys.executable, "-I", "-c", script, str(Path(protocol.__file__)),
        str(service.socket_path)], cwd=service.workspace, stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    service.note_spawned(child)
    out, err = child.communicate(json.dumps(requests), timeout=60)
    assert child.returncode == 0, err
    responses = json.loads(out)
    for query, response in zip(requests, responses):
        assert protocol.validate_response(response, request=query, descriptor=service.descriptor)["ok"]
    assert body(responses[0]["page"]) == CHANGED
    assert responses[1]["page"]["artifact_sha256"] == sha(case.snapshot.artifacts["summary"])
    assert responses[1]["page"]["workspace_generation"] == case.generation
    assert not service._live()
    with pytest.raises(ValueError, match="one live spawned child"):
        service.note_spawned(child)
    service.close()
    audit = json.loads(service.audit_path.read_bytes())
    assert audit["service_stopped"] is audit["accounting_complete"] is True
    assert audit["accepted_requests"] == 2
    assert (service.repository / "mod.py").read_bytes() == SOURCE
