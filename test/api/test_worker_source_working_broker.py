"""Working-source requests across actual semantic admission and broker framing."""
import json
import socket
import uuid

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import worker_source_client as client
from test.api.test_worker_source_broker import allocated, native, service, _bytes  # noqa: F401


def request(value, **changed):
    query = dict(path="mod.py", view="compact", cursor=None, max_bytes=8192,
        expected_source_sha256=None, expected_view_sha256=None,
        source_state="working-candidate", expected_workspace_generation=None)
    query.update(changed)
    return client.validate_request(dict(schema="supervisor-worker-source-request@2",
        session_id=value.descriptor["session_id"], request_id=uuid.uuid4().hex, query=query))


def answer(service, **changed):
    query = request(service, **changed)
    result = service._answer(query, None)
    return client.validate_response(result, request=query, descriptor=service.descriptor)


def pin(page):
    return dict(expected_source_sha256=page["source_sha256"],
        expected_view_sha256=page["view_sha256"],
        expected_workspace_generation=page["workspace_generation"])


def test_edited_allocated_source_refreshes_without_changing_canonical_semantic_admission(service, monkeypatch):
    monkeypatch.setattr(service, "_peer", lambda channel: None)
    before = answer(service)
    assert before["ok"], before
    baseline, descriptor = service._table_bytes, dict(service.descriptor)
    canonical = (service.repository / "mod.py").read_bytes()
    changed = canonical + b"\nEXTRA = 42  # edited in allocated worktree\n"
    (service.workspace / "mod.py").write_bytes(changed)
    current = answer(service)
    assert current["ok"], current
    page = current["page"]
    assert page["workspace_generation"] != before["page"]["workspace_generation"]
    assert page["baseline_evidence_applicable"] is False
    expanded = answer(service, view="original", **pin(page))
    assert expanded["ok"] and _bytes(expanded["page"]) == changed
    assert service._table_bytes == baseline and service.descriptor == descriptor
    assert (service.repository / "mod.py").read_bytes() == canonical
    from test.api.test_worker_source_broker import _request
    refused = service._answer(_request(service), None)
    assert refused["ok"] is False and refused["page"] is None


@pytest.mark.parametrize("target", ["canonical", "artifact", "binding"])
def test_working_refresh_cannot_adopt_stale_canonical_or_semantic_context(service, monkeypatch, target):
    monkeypatch.setattr(service, "_peer", lambda channel: None)
    assert answer(service)["ok"]
    if target == "canonical":
        path = service.repository / "mod.py"
        path.write_bytes(path.read_bytes() + b"\n# canonical drift\n")
    elif target == "artifact":
        path = service.repository / json.loads(service._table_bytes)["semantic_artifact"]
        path.write_bytes(path.read_bytes() + b" ")
    else:
        service.descriptor["semantic_sha256"] = "0" * 64
    monkeypatch.setattr(service.views, "working_page", lambda **kwargs: pytest.fail("stale baseline reached candidate projection"))
    result = service._answer(request(service), None)
    assert result["reason_code"] == "stale_context" and result["page"] is None


def test_working_page_changed_after_preparation_is_not_delivered(service, monkeypatch):
    monkeypatch.setattr(service, "_peer", lambda channel: None)
    original = service.views.working_page
    def changed(**kwargs):
        result = original(**kwargs)
        path = service.workspace / "mod.py"
        path.write_bytes(path.read_bytes() + b"\n# racing allocated edit\n")
        return result
    monkeypatch.setattr(service.views, "working_page", changed)
    response = service._answer(request(service), None)
    assert response["ok"] is False and response["page"] is None
    assert response["reason_code"] == "stale_context"


def test_old_pinned_generation_refuses_without_auto_refresh_then_fresh_read_recovers(service, monkeypatch):
    monkeypatch.setattr(service, "_peer", lambda channel: None)
    first = answer(service)["page"]
    path = service.workspace / "mod.py"
    original = path.read_bytes()
    path.write_bytes(original + b"\nVALUE_2 = 2\n")
    refused = answer(service, view="original", **pin(first))
    assert refused["ok"] is False and refused["page"] is None
    fresh = answer(service)
    assert fresh["ok"] and fresh["page"]["workspace_generation"] != first["workspace_generation"]
    refused = answer(service, view="original", **pin(first))
    assert refused["ok"] is False
    assert answer(service, view="original", **pin(fresh["page"]))["ok"]


def test_peer_refusal_precedes_working_snapshot_capture(service, monkeypatch):
    def refuse(channel):
        raise ValueError("foreign peer")
    monkeypatch.setattr(service, "_peer", refuse)
    monkeypatch.setattr(service.views, "working_page", lambda **kwargs: pytest.fail("unadmitted peer reached working source"))
    response = service._answer(request(service), None)
    assert response["reason_code"] == "peer_refused" and response["page"] is None


def test_replayed_request_does_not_refresh_again(service, monkeypatch):
    monkeypatch.setattr(service, "_peer", lambda channel: None)
    query = request(service)
    first = service._answer(query, None)
    assert first["ok"]
    path = service.workspace / "mod.py"
    path.write_bytes(path.read_bytes() + b"\nVALUE_2 = 2\n")
    monkeypatch.setattr(service.views, "working_page", lambda **kwargs: pytest.fail("replayed request reached working source"))
    result = service._answer(query, None)
    assert result["reason_code"] == "invalid_request" and result["page"] is None


def test_socket_working_source_bytes_are_accounted_and_cleanup_retires_service(service, monkeypatch):
    monkeypatch.setattr(service, "_peer", lambda channel: None)
    path = service.workspace / "mod.py"
    changed = path.read_bytes() + b"\nVALUE_2 = 2\n"
    path.write_bytes(changed)
    query = request(service, view="original")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as channel:
        channel.settimeout(30)
        channel.connect(str(service.socket_path))
        client.send_frame(channel, query)
        response = client.receive_frame(channel)
    client.validate_response(response, request=query, descriptor=service.descriptor)
    assert response["ok"] and _bytes(response["page"]) == changed
    service.close()
    assert not service.socket_path.exists() and not service.descriptor_path.exists()
    audit = json.loads(service.audit_path.read_bytes())
    assert audit["service_stopped"] is audit["accounting_complete"] is True
    assert audit["accepted_requests"] == len(audit["requests"]) == 1
    assert audit["requests"][0]["returned_source_bytes"] == len(changed)
    assert audit["requests"][0]["delivered"] is True
    assert audit["response_bytes"] == len(client.canonical_json(response))
    assert audit["retrieval_provider_calls"] == audit["source_obligations_discharged"] == 0
