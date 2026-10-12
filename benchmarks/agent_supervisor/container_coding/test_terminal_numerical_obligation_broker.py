"""Native lakes/pager/task selection with authored container identity in unit tests.

Cross-UID permissions and the deployed launcher are qualified separately in
Docker. These tests do not claim the current user's peers are isolated workers.
"""
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid

import pytest

from benchmarks.agent_supervisor.container_coding.test_terminal_numerical_context import context  # noqa: F401
from ipfs_accelerate_py.agent_supervisor.runtime import numerical_obligation_broker as broker
from ipfs_accelerate_py.agent_supervisor.runtime import numerical_obligation_client as client
from ipfs_accelerate_py.agent_supervisor.runtime import numerical_formalization_context as numerical
from ipfs_accelerate_py.agent_supervisor.runtime import source_formalization_context as guard
from ipfs_accelerate_py.agent_supervisor.runtime import task_context_bundle as bundles
from test.api.test_source_formalization_context import _wire


@pytest.fixture
def service(context, monkeypatch):
    root, receipt = context
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "sample.py"], check=True)
    subprocess.run(["git", "-C", str(root), "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                    "commit", "-qm", "sample"], check=True)
    workspace = root.parent / "allocated"
    subprocess.run(["git", "-C", str(root), "worktree", "add", "--detach", "-q", str(workspace)], check=True)
    binding = bundles.write_task_context_bundle(repository=root, output=root / ".runtime/context.json",
        prepared=[dict(schema="supervisor-task-context-preparation@1", task_cid="task:sample", task_id="sample",
                       metadata={}, numerical_formalization=receipt)])
    selected = bundles.load_task_context_selection(repository=root, artifact=binding["artifact"],
        expected_sha256=binding["sha256"], task_cid="task:sample", task_id="sample")
    payload = selected["worker_evidence"]
    with tempfile.TemporaryDirectory(prefix="nob-") as temporary:
        boundary = Path(temporary) / "boundary.json"
        boundary.write_text("authored boundary")
        (boundary.parent / "worker-observations").mkdir(mode=0o750)
        # The local test user cannot create root-owned container manifests. Only
        # this boundary/protected-parent adapter is authored; Git, selected
        # receipt files, all native lakes and numerical page replay are real.
        monkeypatch.setattr(guard, "_owner_boundary", lambda path: (
            dict(owner_uid=os.getuid(), worker_uid=os.getuid() + 1), "b" * 64))
        monkeypatch.setattr(guard, "_protected_directory", lambda *args, **kwargs: None)
        value = broker.NumericalObligationBroker(repository=root, workspace=workspace, boundary=boundary,
            payload=payload, evidence_text=guard._json(payload), selection=selected, timeout_seconds=90)
        try:
            yield value
        finally:
            value.close()


def _request(value, **query):
    return client.validate_request(dict(schema=client.REQUEST_SCHEMA,
        session_id=value.descriptor["session_id"], request_id=uuid.uuid4().hex,
        query=dict(path="sample.py", function_id=None, kind=None, cursor=None, limit=8, max_bytes=8192, **query)))


def _query(value, **changed):
    request = _request(value)
    request["query"].update(changed)
    return client.validate_request(request)


def test_real_pages_fit_entire_transport_and_continuations_preserve_exact_rows(service, monkeypatch):
    monkeypatch.setattr(service, "_peer", lambda channel: None)
    rows, cursor = [], None
    for _ in range(64):
        request = _query(service, limit=2, max_bytes=4096, cursor=cursor)
        response = service._answer(request, None)
        client.validate_response(response, request=request, descriptor=service.descriptor)
        assert response["ok"], response
        assert len(client.canonical_json(response)) <= 4096
        rows += response["page"]["rows"]
        cursor = response["page"]["next_cursor"]
        if cursor is None:
            break
    else:
        pytest.fail("pagination did not terminate")
    report = json.loads((Path(service.receipt["output"]) / "report-0000.json").read_bytes())
    assert rows == report["pending_obligations"]
    assert len({r["obligation_id"] for r in rows}) == len(rows)


@pytest.mark.parametrize("target", ["repository", "workspace", "bundle", "receipt"])
def test_source_and_context_mutations_refused_before_native_retrieval(service, monkeypatch, target):
    monkeypatch.setattr(service, "_peer", lambda channel: None)
    path = {"repository": service.repository / "sample.py", "workspace": service.workspace / "sample.py",
        "bundle": service.repository / service.payload["context_bundle"]["artifact"],
        "receipt": Path(service.receipt["output"]) / "receipt.json"}[target]
    path.write_bytes(path.read_bytes() + b" ")
    monkeypatch.setattr(numerical, "retrieve_numerical_obligations", lambda **kwargs: pytest.fail("stale owner reached pager"))
    response = service._answer(_request(service), None)
    assert response["reason_code"] == "stale_context" and response["page"] is None


def test_concurrent_allocated_edit_is_fenced_after_native_page(service, monkeypatch):
    monkeypatch.setattr(service, "_peer", lambda channel: None)
    original = numerical.retrieve_numerical_obligations
    def changed(**kwargs):
        result = original(**kwargs)
        path = service.workspace / "sample.py"
        path.write_bytes(path.read_bytes() + b"\n# edit during replay\n")
        return result
    monkeypatch.setattr(numerical, "retrieve_numerical_obligations", changed)
    response = service._answer(_request(service), None)
    assert response["reason_code"] == "stale_context" and response["page"] is None


@pytest.mark.parametrize("change", ["session", "cursor", "path", "budget", "expired", "peer"])
def test_closed_scope_refusals(service, monkeypatch, change):
    monkeypatch.setattr(service, "_peer", lambda channel: None)
    request = _request(service)
    expected = "retrieval_refused"
    if change == "session":
        request["session_id"] = "0" * 32
        expected = "invalid_request"
    elif change == "cursor":
        request["query"]["cursor"] = dict(context_receipt_sha256="0" * 64, cursor={})
    elif change == "path":
        request["query"]["path"] = "not-selected.py"
    elif change == "budget":
        service._response_bytes = broker.MAX_TOTAL_RESPONSE_BYTES
        expected = "budget_exhausted"
    elif change == "expired":
        service.descriptor["expires_monotonic_ns"] = time.monotonic_ns() - 1
        expected = "session_closed"
    else:
        def denied(channel):
            raise ValueError("unrelated peer")
        monkeypatch.setattr(service, "_peer", denied)
        expected = "peer_refused"
    response = service._answer(request, None)
    assert response["reason_code"] == expected and response["page"] is None


def test_real_socket_exchange_replay_and_shutdown_accounting(service, monkeypatch):
    # This exercises the real framing/socket/thread, with only peer admission
    # authored. Cross-UID ancestry is exercised by the Docker qualification.
    monkeypatch.setattr(service, "_peer", lambda channel: None)
    request = _request(service)
    replies = []
    for _ in range(2):
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as channel:
            channel.settimeout(30)
            channel.connect(str(service.socket_path))
            client.send_frame(channel, request)
            replies.append(client.receive_frame(channel))
    assert replies[0]["ok"] is True and replies[1]["reason_code"] == "invalid_request"
    service.close()
    service.close()
    assert not service.socket_path.exists() and not service.descriptor_path.exists()
    audit = json.loads(service.audit_path.read_bytes())
    assert audit["service_stopped"] is True and audit["accepted_requests"] == 2
    assert audit["accounting_complete"] is True and audit["inflight_request"] is None
    assert audit["response_bytes"] == sum(len(client.canonical_json(r)) for r in replies)
    assert all(row["delivered"] for row in audit["requests"])
    assert "rows" not in audit["requests"][0] and audit["source_obligations_discharged"] == 0


def test_inflight_shutdown_cannot_claim_complete_accounting_or_send_late_rows(service, monkeypatch):
    started, release = threading.Event(), threading.Event()
    def pending(request, channel):
        started.set()
        assert release.wait(timeout=10)
        return service._response(request, reason="retrieval_refused")
    monkeypatch.setattr(service, "_answer", pending)
    channel = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    channel.settimeout(10)
    try:
        channel.connect(str(service.socket_path))
        client.send_frame(channel, _request(service))
        assert started.wait(timeout=5)
        service.close()
        audit = json.loads(service.audit_path.read_bytes())
        assert audit["accepted_requests"] == audit["inflight_request"] == 1
        assert audit["accounting_complete"] is audit["service_stopped"] is False
        assert audit["response_bytes"] == 0 and audit["requests"] == []
        release.set()
        service._thread.join(timeout=5)
        assert not service._thread.is_alive()
        assert channel.recv(1) == b""
    finally:
        release.set()
        channel.close()


def test_failed_exclusive_descriptor_write_removes_only_its_created_inode(tmp_path, monkeypatch):
    target = tmp_path / "descriptor.json"
    def full(fd):
        raise OSError("authored disk full")
    monkeypatch.setattr(os, "fsync", full)
    with pytest.raises(OSError, match="disk full"):
        broker._write_new(target, {"bounded": True}, 0o440)
    assert not target.exists()
    target.write_bytes(b"existing owner data")
    with pytest.raises(FileExistsError):
        broker._write_new(target, {"bounded": True}, 0o440)
    assert target.read_bytes() == b"existing owner data"


def test_peer_uid_and_exact_birth_ancestry_are_both_required(service):
    child = subprocess.Popen([sys.executable, "-I", "-c", "import time; time.sleep(30)"])
    try:
        service.note_spawned(child)
        with pytest.raises(ValueError, match="one live"):
            service.note_spawned(child)
        left, right = socket.socketpair()
        try:
            with pytest.raises(ValueError, match="uid"):
                service._peer(right)
            service.worker_uid = os.getuid()
            # Matching UID still cannot admit this socket's owner test process:
            # it is not beneath the actual launched child.
            with pytest.raises(ValueError, match="outside"):
                service._peer(right)
        finally:
            left.close()
            right.close()
    finally:
        child.terminate()
        child.wait(timeout=5)


def test_factory_uses_sealed_prompt_without_consuming_worker_stdin(service):
    prompt = guard._json(_wire(service.payload)).encode()
    fd = os.memfd_create("numerical-test-prompt", os.MFD_ALLOW_SEALING | os.MFD_CLOEXEC)
    argv = ["--semantic-repository", str(service.repository), "--purpose", "coding", "--timeout", "60"]
    other = None
    try:
        os.write(fd, prompt)
        os.lseek(fd, 0, os.SEEK_SET)
        with pytest.raises(ValueError, match="sealed owner prompt"):
            broker.prepare_numerical_obligation_broker(argv=argv, boundary=service.boundary,
                workspace=service.workspace, input_fd=fd)
        fcntl.fcntl(fd, fcntl.F_ADD_SEALS, fcntl.F_SEAL_SEAL | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_GROW | fcntl.F_SEAL_WRITE)
        other = broker.prepare_numerical_obligation_broker(argv=argv, boundary=service.boundary,
            workspace=service.workspace, input_fd=fd)
        assert os.lseek(fd, 0, os.SEEK_CUR) == 0 and os.read(fd, len(prompt) + 1) == prompt
        assert other.descriptor["context_receipt_sha256"] == service.receipt["receipt_sha256"]
        assert other.descriptor["task_id"] == service.payload["task_id"]
        assert other.descriptor["expires_monotonic_ns"] - other.descriptor["issued_monotonic_ns"] == 60_000_000_000
        with pytest.raises(ValueError, match="independently"):
            broker.prepare_numerical_obligation_broker(argv=argv + other.extra_argv,
                boundary=service.boundary, workspace=service.workspace, input_fd=fd)
    finally:
        os.close(fd)
        if other is not None:
            other.close()
