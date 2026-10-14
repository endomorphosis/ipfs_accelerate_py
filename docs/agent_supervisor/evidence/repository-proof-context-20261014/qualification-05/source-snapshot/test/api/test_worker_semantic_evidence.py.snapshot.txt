"""Exact baseline capsule expansion; no providers, inference, or hidden tasks.

The native semantic producer, Git worktrees and registered child/socket are
real. Only the container UID/protected-directory admission is authored here.
"""
import base64
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import uuid

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import worker_semantic_evidence_client as client
from ipfs_accelerate_py.agent_supervisor.runtime import worker_source_client as source
from ipfs_accelerate_py.agent_supervisor.runtime import worker_source_broker as broker
from ipfs_accelerate_py.agent_supervisor.runtime import source_formalization_context as guard
from ipfs_accelerate_py.agent_supervisor.runtime.worker_semantic_evidence import WorkerSemanticEvidence
from test.api.test_semantic_router_integration import allocated, native  # noqa: F401


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _bytes(page):
    return page["content"].encode() if page["encoding"] == "utf-8" else base64.b64decode(page["content"], validate=True)


def _request(service, **kwargs):
    return source.validate_request(dict(schema=client.REQUEST_SCHEMA,
        session_id=service.descriptor["session_id"], request_id=uuid.uuid4().hex,
        query=dict(capsule_index=0, cursor=None, max_bytes=4096) | kwargs))


CHILD = r'''
import hashlib, importlib.util, json, pathlib, socket, sys
spec = importlib.util.spec_from_file_location("semantic_client", sys.argv[1])
client = importlib.util.module_from_spec(spec)
spec.loader.exec_module(client)
# Authored same-UID fixture only; exact descriptor bytes/workspace remain checked.
def descriptor(path, digest, *, workspace=None):
    raw = pathlib.Path(path).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == digest
    value = client.source.validate_descriptor(json.loads(raw))
    assert client.canonical_json(value) == raw
    assert workspace is None or value["workspace"] == str(workspace)
    return value
client.source.load_descriptor = descriptor
for line in sys.stdin:
    value = json.loads(line)
    if "client" in value:
        response = client.request_page(sys.argv[2], sys.argv[3],
            workspace=pathlib.Path.cwd(), **value["client"])
    else:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as channel:
            channel.settimeout(30)
            channel.connect(sys.argv[4])
            client.source.send_frame(channel, value)
            response = client.source.receive_frame(channel)
    print(json.dumps(response), flush=True)
'''


def exchange(service, request):
    process = service.test_process
    process.stdin.write(json.dumps(request) + "\n")
    process.stdin.flush()
    line = process.stdout.readline()
    assert line, "registered evidence worker exited without a response"
    return json.loads(line)


@pytest.fixture
def evidence_service(allocated, monkeypatch):
    root, workspace, prompt, _ = allocated
    with tempfile.TemporaryDirectory(prefix="semantic-evidence-") as directory:
        boundary = Path(directory) / "boundary.json"
        boundary.write_text("authored boundary")
        (boundary.parent / "worker-observations").mkdir(mode=0o750)
        monkeypatch.setattr(guard, "_owner_boundary", lambda path:
            (dict(owner_uid=os.getuid(), worker_uid=os.getuid() + 1), "b" * 64))
        monkeypatch.setattr(guard, "_protected_directory", lambda *args, **kwargs: None)
        service = broker.WorkerSourceBroker(repository=root, workspace=workspace, boundary=boundary,
            prompt=prompt, profile="identity@1", timeout_seconds=120,
            semantic_evidence_mode=client.MODE)
        service.worker_uid = os.getuid()
        process = subprocess.Popen([sys.executable, "-I", "-c", CHILD, client.__file__,
            str(service.descriptor_path), service.descriptor_sha256, str(service.socket_path)],
            cwd=workspace, text=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        service.note_spawned(process)
        service.test_process = process
        try:
            yield service
        finally:
            process.stdin.close()
            process.wait(timeout=10)
            service.close()


@pytest.fixture
def pager():
    binding = dict(task_id="task:fixture", scope_cid="scope:fixture", semantic_root_cid="root:fixture",
        semantic_sha256="a" * 64, native_prompt_sha256="b" * 64,
        source_manifest_sha256="c" * 64, profile="identity@1")
    capsules = [dict(stable_symbol_id=f"symbol:{index}", text="λ🙂\\\"\n" * 500) for index in range(2)]
    return WorkerSemanticEvidence(capsules=capsules, binding=binding), capsules


def test_unit_paging_restores_exact_canonical_native_bytes_and_retains_input_copy(pager):
    service, capsules = pager
    expected = client.canonical_json(capsules[0])
    capsules[0]["text"] = "caller mutation"
    cursor, chunks, encodings = None, [], set()
    for _ in range(100):
        page = service.page(capsule_index=0, cursor=cursor, max_bytes=1800)
        assert len(client.canonical_json(page)) <= 1800
        chunks.append(_bytes(page))
        encodings.add(page["encoding"])
        assert all(page[key] is False for key in client.AUTHORITY_FLAGS)
        assert page["source_obligations_discharged"] == 0
        cursor = page["next_cursor"]
        if cursor is None:
            break
    else:
        pytest.fail("bounded capsule pagination did not terminate")
    assert b"".join(chunks) == expected
    assert page["artifact_sha256"] == _sha(expected)
    assert "base64" in encodings


@pytest.mark.parametrize("limit", ["count", "single_bytes", "aggregate_bytes"])
def test_owner_refuses_capsule_population_outside_initial_expansion_bound(pager, limit):
    binding = client.strict_json_loads(pager[0]._binding_bytes)
    if limit == "count":
        capsules = [{} for _ in range(client.MAX_SELECTED_CAPSULES + 1)]
    elif limit == "single_bytes":
        capsules = [dict(text="x" * client.MAX_SELECTED_CAPSULE_BYTES)]
    else:
        capsules = [dict(text="x" * (client.MAX_SELECTED_CAPSULE_BYTES // 2)) for _ in range(2)]
    with pytest.raises(ValueError):
        WorkerSemanticEvidence(capsules=capsules, binding=binding)


def test_owner_accepts_exact_aggregate_boundary(pager):
    binding = client.strict_json_loads(pager[0]._binding_bytes)
    row = dict(text="")
    row["text"] = "x" * (client.MAX_SELECTED_CAPSULE_BYTES - len(client.canonical_json(row)))
    service = WorkerSemanticEvidence(capsules=[row], binding=binding)
    assert len(service._capsules[0]) == client.MAX_SELECTED_CAPSULE_BYTES


@pytest.mark.parametrize("mutation", ["index", "budget", "token", "offset", "digest", "replay", "cleared"])
def test_cursor_is_bound_to_exact_selection_budget_and_single_use(pager, mutation):
    service, _ = pager
    first = service.page(capsule_index=0, max_bytes=1800)
    query = dict(capsule_index=0, max_bytes=1800, cursor=deepcopy(first["next_cursor"]))
    assert query["cursor"] is not None
    if mutation == "index": query["capsule_index"] = 1
    elif mutation == "budget": query["max_bytes"] = 1801
    elif mutation == "token": query["cursor"]["token"] = "0" * 32
    elif mutation == "offset": query["cursor"]["offset"] += 1
    elif mutation == "digest": query["cursor"]["request_sha256"] = "0" * 64
    elif mutation == "replay": service.page(**query)
    else: service.clear()
    with pytest.raises(ValueError):
        service.page(**query)
    assert service._continuations == {}


@pytest.mark.parametrize("query", [dict(capsule_index=True), dict(capsule_index=-1),
    dict(capsule_index=256), dict(path="owner.duckdb"), dict(artifact="world"),
    dict(expected_workspace_generation="a" * 64), dict(max_bytes=2047)])
def test_closed_request_rejects_wrong_types_and_arbitrary_locators(query):
    with pytest.raises(ValueError):
        client.validate_request(dict(schema=client.REQUEST_SCHEMA, session_id="a" * 32,
            request_id="b" * 32, query=dict(capsule_index=0, cursor=None, max_bytes=4096) | query))


def test_registered_child_stdlib_client_restores_selected_native_capsule_and_accounts(evidence_service):
    service = evidence_service
    table = json.loads(service._table_bytes)
    payload = json.loads((service.repository / table["semantic_artifact"]).read_bytes())
    expected = client.canonical_json(payload["capsules"][0])
    cursor, chunks = None, []
    while True:
        response = exchange(service, {"client": dict(capsule_index=0, cursor=cursor, max_bytes=4096)})
        assert response["ok"], response
        page = response["page"]
        assert page["binding"] == {key: service.descriptor[key] for key in source.BINDING_FIELDS}
        chunks.append(_bytes(page))
        cursor = page["next_cursor"]
        if cursor is None:
            break
    assert b"".join(chunks) == expected and page["artifact_sha256"] == _sha(expected)
    service.close()
    audit = json.loads(service.audit_path.read_bytes())
    assert audit["schema"] == "supervisor-worker-source-session@6"
    assert audit["accounting_complete"] is audit["service_stopped"] is True
    assert sum(row["returned_semantic_evidence_bytes"] for row in audit["requests"]) == len(expected)
    assert all(row["semantic_evidence_page"] and row["delivered"]
        and row["returned_source_bytes"] == row["returned_candidate_context_bytes"] == 0
        and row["request_operation"] == "semantic_evidence" for row in audit["requests"])
    assert audit["retrieval_provider_calls"] == audit["source_obligations_discharged"] == 0


@pytest.mark.parametrize("target", ["canonical", "workspace", "artifact", "producer", "descriptor", "boundary"])
def test_stale_baseline_never_reaches_pager(evidence_service, monkeypatch, target):
    service = evidence_service
    if target in {"canonical", "workspace"}:
        path = (service.repository if target == "canonical" else service.workspace) / "mod.py"
        path.write_bytes(path.read_bytes() + b"\n# changed\n")
    elif target in {"artifact", "producer"}:
        table = json.loads(service._table_bytes)
        artifact = service.repository / table["semantic_artifact"]
        path = artifact if target == "artifact" else artifact.parent / "blocks" / table["semantic_root_cid"]
        path.write_bytes(path.read_bytes() + b" ")
    elif target == "descriptor": service.descriptor["semantic_sha256"] = "0" * 64
    else:
        monkeypatch.setattr(guard, "_owner_boundary", lambda path:
            (dict(owner_uid=os.getuid(), worker_uid=os.getuid() + 1), "f" * 64))
    monkeypatch.setattr(service.semantic_evidence, "page", lambda **kwargs: pytest.fail("stale reached pager"))
    response = exchange(service, _request(service))
    assert response["reason_code"] == "stale_context" and response["page"] is None


def test_edit_during_page_preparation_refuses_and_revokes_continuations(evidence_service, monkeypatch):
    service = evidence_service
    original = service.semantic_evidence.page
    def changed(**kwargs):
        page = original(**kwargs)
        path = service.workspace / "mod.py"
        path.write_bytes(path.read_bytes() + b"\n# racing edit\n")
        return page
    monkeypatch.setattr(service.semantic_evidence, "page", changed)
    response = exchange(service, _request(service))
    assert response["reason_code"] == "stale_context" and response["page"] is None
    assert service.semantic_evidence._continuations == {}


@pytest.mark.parametrize("field", ["task_id", "scope_cid", "semantic_root_cid", "semantic_sha256", "native_prompt_sha256"])
def test_client_rejects_foreign_page_bindings(evidence_service, field):
    request = _request(evidence_service, max_bytes=client.MAX_FRAME_BYTES)
    response = exchange(evidence_service, request)
    assert response["ok"]
    response["page"]["binding"][field] = "f" * 64
    with pytest.raises(ValueError, match="binding differs"):
        source.validate_response(response, request=request, descriptor=evidence_service.descriptor)


def test_capability_is_explicit_and_edit_clients_accept_enabled_v4(evidence_service):
    from ipfs_accelerate_py.agent_supervisor.runtime import worker_source_edit_client as edits
    value = deepcopy(evidence_service.descriptor)
    assert client.validate_capability(value) == value
    with pytest.raises(ValueError):
        edits.validate_descriptor(value)
    value["source_edit_mode"] = "mapped-span@2"
    assert edits.validate_descriptor(value) == value
    del value["semantic_evidence_mode"]
    value["schema"] = source.DESCRIPTOR_SCHEMA_V3
    assert source.validate_descriptor(value) == value
    with pytest.raises(ValueError, match="not advertised"):
        client.validate_capability(value)


@pytest.mark.parametrize("change,reason", [("index", "retrieval_refused"),
    ("session", "invalid_request"), ("replay", "invalid_request"), ("budget", "budget_exhausted")])
def test_closed_owner_scope_refusals(evidence_service, change, reason):
    service = evidence_service
    request = _request(service)
    if change == "index": request["query"]["capsule_index"] = client.MAX_CAPSULES - 1
    elif change == "session": request["session_id"] = "0" * 32
    elif change == "replay": assert exchange(service, request)["ok"]
    else: service._response_bytes = broker.MAX_TOTAL_RESPONSE_BYTES
    response = exchange(service, request)
    assert response["reason_code"] == reason and response["page"] is None


def test_edit_refuses_retained_baseline_cursor_even_after_source_is_restored(evidence_service):
    service = evidence_service
    first = exchange(service, _request(service, max_bytes=2048))
    assert first["ok"] and first["page"]["next_cursor"] is not None
    cursor = first["page"]["next_cursor"]
    path = service.workspace / "mod.py"
    original = path.read_bytes()
    path.write_bytes(original + b"\n# changed\n")
    stale = exchange(service, _request(service, cursor=cursor, max_bytes=2048))
    assert stale["reason_code"] == "stale_context" and stale["page"] is None
    path.write_bytes(original)
    revoked = exchange(service, _request(service, cursor=cursor, max_bytes=2048))
    assert revoked["reason_code"] == "retrieval_refused" and revoked["page"] is None


def test_factory_requires_selected_source_route_before_prompt_or_repository_access(tmp_path):
    for argv in (["--semantic-working-set", "capsule-references@1"],
                 ["--semantic-working-set", "unknown", "--source-view-profile", "identity@1"]):
        with pytest.raises(ValueError):
            broker.prepare_worker_source_broker(argv=argv, boundary=tmp_path / "missing",
                workspace=tmp_path / "absent", input_fd=-1)


def test_factory_binds_explicit_working_set_to_sealed_native_prompt(evidence_service):
    import fcntl
    from ipfs_accelerate_py.agent_supervisor.validation.validation_runtime import _validation_memfd_create
    service, other = evidence_service, None
    fd = _validation_memfd_create("semantic-evidence-fixture", os.MFD_ALLOW_SEALING)
    try:
        raw = service.prompt.encode()
        os.write(fd, raw)
        os.lseek(fd, 0, os.SEEK_SET)
        fcntl.fcntl(fd, getattr(fcntl, "F_ADD_SEALS", 1033), 15)
        other = broker.prepare_worker_source_broker(argv=["--purpose", "coding", "--semantic-repository",
            str(service.repository), "--source-view-profile", "identity@1",
            "--semantic-working-set", "capsule-references@1"], boundary=service.boundary,
            workspace=service.workspace, input_fd=fd)
        assert client.validate_capability(other.descriptor)["native_prompt_sha256"] == _sha(raw)
        assert other.semantic_evidence_mode == client.MODE
        assert os.read(fd, len(raw)) == raw
    finally:
        os.close(fd)
        if other is not None:
            other.close()


def test_client_entrypoint_runs_in_isolation_without_project_import_path():
    result = subprocess.run([sys.executable, "-I", str(Path(client.__file__)), "--help"],
        capture_output=True, text=True, check=True)
    assert "--capsule-index" in result.stdout
