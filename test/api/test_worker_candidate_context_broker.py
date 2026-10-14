"""Candidate artifact routing under the existing source session admission."""
import json
import socket
import uuid

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import worker_source_client as protocol
from test.api.test_worker_source_broker import allocated, native, service  # noqa: F401
from test.api.test_worker_source_working_broker import answer as source_answer


def request(service, generation, **options):
    query = dict(artifact="summary", expected_workspace_generation=generation,
        cursor=None, max_bytes=8192)
    query.update(options)
    return protocol.validate_request(dict(schema="supervisor-worker-candidate-context-request@1",
        session_id=service.descriptor["session_id"], request_id=uuid.uuid4().hex, query=query))


def answer(service, generation, **options):
    query = request(service, generation, **options)
    result = service._answer(query, None)
    return protocol.validate_response(result, request=query, descriptor=service.descriptor)


def fixture_builder(**kwargs):
    raw = protocol.canonical_json(dict(generation=kwargs["workspace_generation"],
        paths=sorted(kwargs["sources"]), support_excluded=kwargs["program_paths"]))
    return {key: raw for key in ("summary", "capsules", "formal")}


def install_builder(service, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.runtime.worker_candidate_context import WorkerCandidateContext
    monkeypatch.setattr(service, "_peer", lambda channel: None)
    service.candidate_context = WorkerCandidateContext(service.views, ("mod.py",), builder=fixture_builder)


def test_context_requires_existing_current_generation(service, monkeypatch):
    install_builder(service, monkeypatch)
    assert answer(service, "0" * 64)["reason_code"] == "stale_context"
    generation = source_answer(service)["page"]["workspace_generation"]
    result = answer(service, generation)
    assert result["ok"] and result["page"]["baseline_evidence_applicable"] is False
    path = service.workspace / "mod.py"
    path.write_bytes(path.read_bytes() + b"\n# edit\n")
    assert answer(service, generation)["reason_code"] == "stale_context"
    current = source_answer(service)["page"]["workspace_generation"]
    assert current != generation and answer(service, current)["ok"]
    assert not answer(service, generation)["ok"]


@pytest.mark.parametrize("change", ["peer", "post_edit", "closed", "replay"])
def test_candidate_broker_rechecks_owner_lifecycle(service, monkeypatch, change):
    install_builder(service, monkeypatch)
    generation = source_answer(service)["page"]["workspace_generation"]
    query = request(service, generation)
    if change == "peer":
        def refuse(channel):
            raise ValueError("not descendant")
        monkeypatch.setattr(service, "_peer", refuse)
    elif change == "closed":
        service.close()
    elif change == "replay":
        assert service._answer(query, None)["ok"]
    else:
        original = service.candidate_context.page
        def edit(**kwargs):
            result = original(**kwargs)
            path = service.workspace / "mod.py"
            path.write_bytes(path.read_bytes() + b"\n# edit after artifact\n")
            return result
        monkeypatch.setattr(service.candidate_context, "page", edit)
    result = service._answer(query, None)
    assert result["ok"] is False and result["page"] is None


def test_actual_socket_candidate_accounting_is_separate_from_source(service, monkeypatch):
    install_builder(service, monkeypatch)
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
    assert audit["schema"] == "supervisor-worker-source-session@5"
    assert audit["service_stopped"] and audit["accounting_complete"]
    row, = audit["requests"]
    assert (row["request_operation"], row["request_source_state"], row["request_view"]) == (
        "candidate_context", "working-candidate", "not_applicable")
    assert row["returned_source_bytes"] == 0 and row["candidate_context_page"] is True
    assert row["returned_candidate_context_bytes"] == response["page"]["end_offset"]
    assert row["response_bytes"] == len(protocol.canonical_json(response))
    assert b"support_excluded" not in service.audit_path.read_bytes()
