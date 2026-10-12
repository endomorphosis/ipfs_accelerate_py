"""Real semantic producer/worktree/source views; authored container admission.

Kernel cross-UID admission is qualified separately in Docker. These tests use
actual source replay, sealed stdin, socket framing and lifecycle accounting.
"""
import base64
import fcntl
import json
import os
from pathlib import Path
import socket
import tempfile
import uuid

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import worker_source_broker as broker
from ipfs_accelerate_py.agent_supervisor.runtime import worker_source_client as client
from ipfs_accelerate_py.agent_supervisor.runtime import source_formalization_context as guard
from ipfs_accelerate_py.agent_supervisor.runtime.reversible_source_view import build_source_view
from ipfs_accelerate_py.agent_supervisor.validation.validation_runtime import _validation_memfd_create
from test.api.test_semantic_router_integration import allocated, native  # noqa: F401


@pytest.fixture
def service(allocated, monkeypatch):
    repository, workspace, prompt, source = allocated
    with tempfile.TemporaryDirectory(prefix="source-broker-") as directory:
        boundary = Path(directory) / "boundary.json"
        boundary.write_text("authored boundary")
        (boundary.parent / "worker-observations").mkdir(mode=0o750)
        monkeypatch.setattr(guard, "_owner_boundary", lambda path: (
            dict(owner_uid=os.getuid(), worker_uid=os.getuid() + 1), "b" * 64))
        monkeypatch.setattr(guard, "_protected_directory", lambda *args, **kwargs: None)
        value = broker.WorkerSourceBroker(repository=repository, workspace=workspace,
            boundary=boundary, prompt=prompt, profile="python-comments@1", timeout_seconds=90)
        try:
            yield value
        finally:
            value.close()


def _request(value, **changed):
    query = dict(path="mod.py", view="compact", cursor=None, max_bytes=8192,
        expected_source_sha256=None, expected_view_sha256=None)
    query.update(changed)
    return client.validate_request(dict(schema=client.REQUEST_SCHEMA,
        session_id=value.descriptor["session_id"], request_id=uuid.uuid4().hex, query=query))


def _bytes(page):
    return page["content"].encode() if page["encoding"] == "utf-8" else base64.b64decode(page["content"])


def test_compact_and_pinned_original_without_numerical_context(service, monkeypatch):
    monkeypatch.setattr(service, "_peer", lambda channel: None)
    assert "numerical-formalization-context" not in service.prompt
    request = _request(service)
    compact = service._answer(request, None)
    assert compact["ok"], compact
    page = compact["page"]
    original_bytes = (service.repository / "mod.py").read_bytes()
    expected = build_source_view(original_bytes, profile="python-comments@1")
    assert page["exhausted"] and _bytes(page) == expected.view
    assert page["proof_authority"] is page["omission_authority"] is False
    request = _request(service, view="original", expected_source_sha256=page["source_sha256"],
        expected_view_sha256=page["view_sha256"])
    original = service._answer(request, None)
    assert original["ok"] and original["page"]["exhausted"]
    assert _bytes(original["page"]) == original_bytes
    assert len(client.canonical_json(original)) <= request["query"]["max_bytes"]
    assert all(key not in service.descriptor for key in ("task_cid", "context_bundle_sha256", "source_tree_id"))


@pytest.mark.parametrize("target", ["canonical", "allocated", "artifact", "binding", "boundary"])
def test_stale_source_or_context_never_reaches_view_page(service, monkeypatch, target):
    monkeypatch.setattr(service, "_peer", lambda channel: None)
    if target in {"canonical", "allocated"}:
        root = service.repository if target == "canonical" else service.workspace
        path = root / "mod.py"
        path.write_bytes(path.read_bytes() + b"\n# changed\n")
    elif target == "artifact":
        path = service.repository / json.loads(service._table_bytes)["semantic_artifact"]
        path.write_bytes(path.read_bytes() + b" ")
    elif target == "binding":
        service.descriptor["semantic_sha256"] = "0" * 64
    else:
        monkeypatch.setattr(guard, "_owner_boundary", lambda path: (
            dict(owner_uid=os.getuid(), worker_uid=os.getuid() + 1), "f" * 64))
    monkeypatch.setattr(service.views, "page", lambda **kw: pytest.fail("stale context reached source page"))
    response = service._answer(_request(service), None)
    assert response["reason_code"] == "stale_context" and response["page"] is None


def test_edit_after_view_preparation_refuses_response(service, monkeypatch):
    monkeypatch.setattr(service, "_peer", lambda channel: None)
    original = service.views.page
    def changed(**kwargs):
        result = original(**kwargs)
        path = service.workspace / "mod.py"
        path.write_bytes(path.read_bytes() + b"\n# racing edit\n")
        return result
    monkeypatch.setattr(service.views, "page", changed)
    response = service._answer(_request(service), None)
    assert response["reason_code"] == "stale_context" and response["page"] is None


@pytest.mark.parametrize("change,reason", [("foreign_path", "retrieval_refused"),
    ("pins", "retrieval_refused"), ("cursor", "retrieval_refused"), ("session", "invalid_request"),
    ("replay", "invalid_request"), ("peer", "peer_refused"), ("budget", "budget_exhausted")])
def test_bounded_scope_refusals(service, monkeypatch, change, reason):
    monkeypatch.setattr(service, "_peer", lambda channel: None)
    request = _request(service)
    if change == "foreign_path": request["query"]["path"] = "private.db"
    elif change == "pins":
        request["query"].update(expected_source_sha256="0" * 64, expected_view_sha256="1" * 64)
    elif change == "cursor": request["query"]["cursor"] = {}
    elif change == "session": request["session_id"] = "0" * 32
    elif change == "replay": assert service._answer(request, None)["ok"]
    elif change == "peer":
        def refuse(channel): raise ValueError("authored foreign peer")
        monkeypatch.setattr(service, "_peer", refuse)
    else: service._response_bytes = broker.MAX_TOTAL_RESPONSE_BYTES
    response = service._answer(request, None)
    assert response["reason_code"] == reason and response["page"] is None


def test_socket_lifecycle_retains_source_accounting_not_numerical_rows(service, monkeypatch):
    monkeypatch.setattr(service, "_peer", lambda channel: None)
    request = _request(service, view="original")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as channel:
        channel.settimeout(30)
        channel.connect(str(service.socket_path))
        client.send_frame(channel, request)
        response = client.receive_frame(channel)
    assert response["ok"]
    service.close()
    assert not service.socket_path.exists() and not service.descriptor_path.exists()
    audit = json.loads(service.audit_path.read_bytes())
    assert audit["schema"] == "supervisor-worker-source-session@1"
    assert audit["service_stopped"] is audit["accounting_complete"] is True
    assert audit["accepted_requests"] == len(audit["requests"]) == 1
    assert audit["requests"][0]["returned_source_bytes"] == len(_bytes(response["page"]))
    assert "returned_rows" not in audit["requests"][0]
    assert audit["response_bytes"] == len(client.canonical_json(response))
    assert audit["requests"][0]["delivered"] is True
    assert audit["retrieval_provider_calls"] == audit["source_obligations_discharged"] == 0


def test_factory_requires_sealed_stdin_and_does_not_consume_it(service):
    fd = _validation_memfd_create("source-view-prompt", getattr(os, "MFD_ALLOW_SEALING", 2) | getattr(os, "MFD_CLOEXEC", 1))
    other = None
    try:
        raw = service.prompt.encode()
        os.write(fd, raw)
        os.lseek(fd, 0, os.SEEK_SET)
        argv = ["--purpose", "coding", "--semantic-repository", str(service.repository),
            "--source-view-profile", "python-comments@1", "--timeout", "60"]
        with pytest.raises(ValueError, match="sealed owner prompt"):
            broker.prepare_worker_source_broker(argv=argv, boundary=service.boundary,
                workspace=service.workspace, input_fd=fd)
        fcntl.fcntl(fd, getattr(fcntl, "F_ADD_SEALS", 1033), 15)
        other = broker.prepare_worker_source_broker(argv=argv, boundary=service.boundary,
            workspace=service.workspace, input_fd=fd)
        assert os.read(fd, len(raw)) == raw
        assert other.descriptor["native_prompt_sha256"] == service.descriptor["native_prompt_sha256"]
        assert other.descriptor["expires_monotonic_ns"] - other.descriptor["issued_monotonic_ns"] == 60_000_000_000
        with pytest.raises(ValueError, match="independently"):
            broker.prepare_worker_source_broker(argv=argv + other.extra_argv,
                boundary=service.boundary, workspace=service.workspace, input_fd=fd)
    finally:
        os.close(fd)
        if other is not None: other.close()


@pytest.mark.parametrize("argv", [[], ["--source-view-profile", "disabled"]])
def test_default_does_not_read_stdin_or_open_sources(argv, monkeypatch):
    monkeypatch.setattr(os, "pread", lambda *args: pytest.fail("disabled source view read stdin"))
    assert broker.prepare_worker_source_broker(argv=argv, boundary="/unavailable",
        workspace="/unavailable", input_fd=-1) is None


@pytest.mark.parametrize("argv", [["--source-view-profile", "unknown"],
    ["--source-view-profile", "identity@1", "--purpose", "planning"],
    ["--source-view-profile", "identity@1"],
    ["--source-view-profile", "disabled", "--source-view-profile", "identity@1"],
    ["--source-view-descriptor", "/foreign"]])
def test_invalid_optin_refuses_before_stdin(argv, monkeypatch):
    monkeypatch.setattr(os, "pread", lambda *args: pytest.fail("invalid route read stdin"))
    with pytest.raises(ValueError):
        broker.prepare_worker_source_broker(argv=argv, boundary="/unavailable", workspace="/unavailable", input_fd=-1)
