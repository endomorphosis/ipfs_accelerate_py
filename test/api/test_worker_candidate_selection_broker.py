"""Selected native artifacts use the real source broker admission and accounting."""
import base64
import json
import socket
import uuid

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import worker_source_client as protocol
from test.api.test_worker_source_broker import allocated, native, service  # noqa: F401
from test.api.test_worker_source_working_broker import answer as source_answer


def selection(**changes):
    return dict(dict(path=None, symbol_id=None, function_id=None, kind=None), **changes)


def request(service, generation, artifact="catalog", selected=None, cursor=None):
    return protocol.validate_request(dict(schema="supervisor-worker-candidate-context-request@2",
        session_id=service.descriptor["session_id"], request_id=uuid.uuid4().hex,
        query=dict(artifact=artifact, selection=selection() if selected is None else selected,
            expected_workspace_generation=generation, cursor=cursor, max_bytes=8192)))


def answer(service, generation, **kwargs):
    query = request(service, generation, **kwargs)
    result = service._answer(query, None)
    return protocol.validate_response(result, request=query, descriptor=service.descriptor)


def content(page):
    return page["content"].encode() if page["encoding"] == "utf-8" else base64.b64decode(page["content"])


def test_native_selected_query_delivers_exact_pinned_subset_and_omission_counts(service, monkeypatch):
    monkeypatch.setattr(service, "_peer", lambda channel: None)
    generation = source_answer(service)["page"]["workspace_generation"]
    result = answer(service, generation)
    assert result["ok"] and result["page"]["schema"] == "supervisor-worker-candidate-context-page@2"
    assert result["page"]["selection"] == selection()
    catalog_raw = content(result["page"])
    while result["page"]["next_cursor"] is not None:
        result = answer(service, generation, cursor=result["page"]["next_cursor"])
        assert result["ok"]
        catalog_raw += content(result["page"])
    assert json.loads(catalog_raw)["source_state"] == "working-candidate"
    query = selection(path="mod.py", kind="not_a_real_obligation_kind")
    result = answer(service, generation, artifact="obligations", selected=query)
    assert result["ok"] and result["page"]["selection"] == query
    data = content(result["page"])
    while result["page"]["next_cursor"] is not None:
        result = answer(service, generation, artifact="obligations", selected=query,
            cursor=result["page"]["next_cursor"])
        assert result["ok"]
        data += content(result["page"])
    payload = json.loads(data)
    assert payload["counts"]["matched_count"] == 0 and payload["rows"] == []
    assert payload["source_obligations_discharged"] == 0 and payload["proof_authority"] is False
    assert answer(service, generation, artifact="capsules", selected=selection(path="private.py"))["ok"] is False


def test_selected_socket_response_counts_candidate_bytes_never_source_bytes(service, monkeypatch):
    monkeypatch.setattr(service, "_peer", lambda channel: None)
    generation = source_answer(service)["page"]["workspace_generation"]
    query = request(service, generation)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as channel:
        channel.settimeout(30)
        channel.connect(str(service.socket_path))
        protocol.send_frame(channel, query)
        response = protocol.receive_frame(channel)
    protocol.validate_response(response, request=query, descriptor=service.descriptor)
    assert response["ok"]
    service.close()
    audit = json.loads(service.audit_path.read_bytes())
    row, = audit["requests"]
    assert audit["accounting_complete"] is True
    assert row["returned_source_bytes"] == 0 and row["candidate_context_page"] is True
    assert row["returned_candidate_context_bytes"] == len(content(response["page"])) > 0
    assert row["response_bytes"] == len(protocol.canonical_json(response))
    assert b"lengthy_first_function" not in service.audit_path.read_bytes()


@pytest.mark.parametrize("change", ["working", "canonical", "after_page", "peer"])
def test_selected_delivery_retains_owner_pre_post_checks(service, monkeypatch, change):
    monkeypatch.setattr(service, "_peer", lambda channel: None)
    generation = source_answer(service)["page"]["workspace_generation"]
    assert answer(service, generation)["ok"]
    def edit(root):
        path = root / "mod.py"
        path.write_bytes(path.read_bytes() + b"\n# drift\n")
    if change in ("working", "canonical"):
        edit(service.workspace if change == "working" else service.repository)
    elif change == "after_page":
        original = service.candidate_context.page
        def altered(**kwargs):
            page = original(**kwargs)
            edit(service.workspace)
            return page
        monkeypatch.setattr(service.candidate_context, "page", altered)
    else:
        def refuse(channel):
            raise ValueError("unrelated peer")
        monkeypatch.setattr(service, "_peer", refuse)
    response = answer(service, generation)
    assert response["ok"] is False and response["page"] is None
