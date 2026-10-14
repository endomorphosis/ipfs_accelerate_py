"""Discovery/read joins over real task admission, sockets and an isolated CLI.

The authored UID seam qualifies protocol and descendant transport, not the
separate container's owner/worker filesystem boundary. No provider is called.
"""
import base64
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import uuid

import pytest

from ipfs_accelerate_py.agent_supervisor.context.context_compiler import (
    ContextCompiler, build_text_context_references, render_context_capsule,
)
from ipfs_accelerate_py.agent_supervisor.context.context_contracts import ContextBudget
from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner as runner
from ipfs_accelerate_py.agent_supervisor.runtime import source_formalization_context as guard
from ipfs_accelerate_py.agent_supervisor.runtime import worker_candidate_context_client as candidate
from ipfs_accelerate_py.agent_supervisor.runtime import worker_source_broker as broker
from ipfs_accelerate_py.agent_supervisor.runtime import worker_source_client as client
from ipfs_accelerate_py.agent_supervisor.runtime.semantic_context_runtime import prepare_semantic_context
from ipfs_accelerate_py.agent_supervisor.runtime.semantic_router_translation import encode_semantic_router_prompt
from ipfs_accelerate_py.agent_supervisor.runtime.source_view_policy import SourceViewPolicy
from ipfs_accelerate_py.agent_supervisor.validation.validation_runtime import _validation_memfd_create
from test.api.test_worker_candidate_context_broker import fixture_builder
from test.api.test_worker_source_views import git, sha


DESCRIPTOR_V2 = "supervisor-worker-source-descriptor@2"
REQUEST_V3 = "supervisor-worker-source-request@3"
PRIVATE_COMMENT = b"# ordinary-body-marker-not-a-discovery-field\n"
MOD = (b"VALUE: int = 1          \r\n"
       b"# blue policy annotation          \r\n"
       b"def target(value):\r\n    return value + VALUE          \r\n"
       + PRIVATE_COMMENT * 70)
STUB = (b"VALUE: int          \r\n"
        b"# blue stub annotation          \r\n"
        b"def target(value: int) -> int: ...          \r\n"
        + PRIVATE_COMMENT * 20)


@pytest.fixture
def admitted(tmp_path, monkeypatch):
    """The real semantic producer, compiler and a real detached Git worktree."""
    root = tmp_path / "repository"
    root.mkdir()
    sources = {"mod.py": MOD, "types.pyi": STUB}
    sources.update({f"part_{index:02d}.py": f"VALUE_{index} = {index}\n".encode()
                    for index in range(12)})
    for name, raw in sources.items():
        (root / name).write_bytes(raw)
    output = root / ".runtime/semantic"
    prepare_semantic_context(repository=root, paths=sorted(sources), required_raw_paths=["mod.py"],
        program_paths=sorted(name for name in sources if name.endswith(".py")),
        objective="Inspect target and its value.", task_id="DISCOVERY-1", output=output,
        worker_query="target", worker_capsule_limit=2)
    artifact = output / "worker-context.json"
    refs = build_text_context_references(artifact.read_text(), reference_prefix="semantic-context",
        kind="semantic-context", path=artifact.relative_to(root).as_posix(),
        repository_id="repo:discovery", tree_id="tree:discovery", required=True, chunk_bytes=1201)
    compiled = ContextCompiler(ContextBudget(max_input_tokens=32768, max_items=128,
        max_item_bytes=16384, max_serialized_bytes=262144)).compile(
        repository_id="repo:discovery", tree_id="tree:discovery", objective_id="DISCOVERY-1",
        objective_revision="sha256:task", policy_id="policy:discovery", policy_revision="sha256:policy",
        caller="supervisor:discovery", stage="implementation", goal={"id": "DISCOVERY-1"},
        authority={"mode": "candidate_only", "completion_authority": False},
        scope={"allowed_paths": sorted(sources)}, acceptance={"criteria": ["native validation pending"]},
        evidence=refs)
    prompt = render_context_capsule(compiled.capsule) + "\nAuthored discovery guidance.\n"
    git(root, "init", "-q")
    git(root, "add", *sources)
    git(root, "-c", "user.name=Qualification", "-c", "user.email=qualification@example.invalid",
        "commit", "-qm", "discovery fixture")
    workspace = tmp_path / "allocated"
    git(root, "worktree", "add", "--detach", "-q", str(workspace))
    monkeypatch.chdir(workspace)
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "empty-codex-home"))
    return dict(root=root, workspace=workspace, prompt=prompt, sources=sources, artifact=artifact)


@pytest.fixture
def service_factory(admitted, monkeypatch):
    services = []
    with tempfile.TemporaryDirectory(prefix="discovery-wire-") as directory:
        monkeypatch.setattr(guard, "_owner_boundary", lambda path:
            (dict(owner_uid=os.getuid(), worker_uid=os.getuid() + 1), "b" * 64))
        monkeypatch.setattr(guard, "_protected_directory", lambda *args, **kwargs: None)

        def create(shared=True):
            parent = Path(directory) / str(len(services))
            parent.mkdir()
            boundary = parent / "boundary.json"
            boundary.write_text("authored boundary")
            (parent / "worker-observations").mkdir(mode=0o750)
            policy = SourceViewPolicy.from_mapping(profile="python-comments@1",
                protected_lines={b"mod.py".hex(): [2], b"types.pyi".hex(): [2]})
            value = broker.WorkerSourceBroker(repository=admitted["root"],
                workspace=admitted["workspace"], boundary=boundary, prompt=admitted["prompt"],
                timeout_seconds=90, **(dict(view_policy=policy) if shared else
                                       dict(profile="python-comments@1")))
            services.append(value)
            return value

        yield create
        for value in services:
            value.close()


def query(service, operation="discover", **changed):
    shared = service.descriptor["schema"] == DESCRIPTOR_V2
    if operation == "discover":
        selected = dict(operation="discover", view_policy_binding_sha256=
            service.descriptor.get("view_policy_binding_sha256"), cursor=None, max_bytes=8192)
        schema = REQUEST_V3
    else:
        selected = dict(path="mod.py", view="compact", cursor=None, max_bytes=8192,
            expected_source_sha256=None, expected_view_sha256=None)
        if shared:
            selected.update(operation="read", view_policy_binding_sha256=
                service.descriptor["view_policy_binding_sha256"], source_state="baseline",
                expected_workspace_generation=None)
            schema = REQUEST_V3
        elif changed.get("source_state") == "working-candidate":
            selected.update(source_state="working-candidate", expected_workspace_generation=None)
            schema = client.REQUEST_SCHEMA_V2
        else:
            schema = client.REQUEST_SCHEMA
    return dict(schema=schema, session_id=service.descriptor["session_id"],
        request_id=uuid.uuid4().hex, query=selected | changed)


def answer(service, operation="discover", **changed):
    request = client.validate_request(query(service, operation, **changed))
    response = service._answer(request, None)
    return client.validate_response(response, request=request, descriptor=service.descriptor)


def authored_reader(service, monkeypatch):
    monkeypatch.setattr(service, "_peer", lambda channel: None)


def decoded(page):
    return (page["content"].encode() if page["encoding"] == "utf-8"
            else base64.b64decode(page["content"], validate=True))


def pinned(row):
    return dict(expected_source_sha256=row["source_sha256"], expected_view_sha256=row["view_sha256"])


def socket_answer(service, request):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as channel:
        channel.settimeout(5)
        channel.connect(str(service.socket_path))
        client.send_frame(channel, request)
        response = client.receive_frame(channel)
    return client.validate_response(response, request=request, descriptor=service.descriptor)


@pytest.mark.parametrize("shared", [False, True])
def test_real_discovery_pages_admit_every_path_and_bind_exact_pinned_reads(
        service_factory, admitted, monkeypatch, shared):
    service = service_factory(shared)
    authored_reader(service, monkeypatch)
    rows, cursor, pages = [], None, []
    for _ in range(32):
        response = answer(service, cursor=cursor, max_bytes=2048)
        assert response["ok"] is True
        page = response["page"]
        assert page["offset"] == len(rows)
        assert len(client.canonical_json(response)) <= 2048
        assert page["schema"] == f"supervisor-worker-source-view-discovery@{2 if shared else 1}"
        assert page["proof_authority"] is page["omission_authority"] is False
        assert page["source_obligations_discharged"] == 0
        rows.extend(page["rows"])
        pages.append(page)
        cursor = page["next_cursor"]
        if cursor is None:
            break
    else:
        pytest.fail("discovery did not terminate")
    assert len(pages) > 1
    assert [row["path"] for row in rows] == sorted(admitted["sources"])
    assert len(rows) == pages[-1]["total_count"] == len(admitted["sources"])
    for row in rows:
        assert set(row) == {"path", "source_sha256", "view_sha256", "manifest_sha256",
                            "source_bytes", "view_bytes", "effective_profile"}
        assert row["source_sha256"] == sha(admitted["sources"][row["path"]])
    wire = client.canonical_json(pages)
    assert PRIVATE_COMMENT.strip() not in wire
    assert not {"content", "pieces", "codec_binding", "restoration", "original"} & set(pages[0])
    selected = next(row for row in rows if row["path"] == "mod.py")
    compact = answer(service, "read", **pinned(selected))["page"]
    original = answer(service, "read", view="original", **pinned(selected))["page"]
    assert compact["source_sha256"] == original["source_sha256"] == selected["source_sha256"]
    assert decoded(original) == MOD
    assert PRIVATE_COMMENT.strip() not in decoded(compact)
    if shared:
        assert b"blue policy annotation" in decoded(compact)
        assert pages[0]["scope_sha256"] == service.descriptor["source_scope_sha256"]
        assert pages[0]["view_policy_binding_sha256"] == compact["view_policy_binding_sha256"]
        stub = next(row for row in rows if row["path"] == "types.pyi")
        stub_page = answer(service, "read", path="types.pyi", **pinned(stub))["page"]
        assert decoded(stub_page) == service.views._view_policy.project(
            STUB, raw_path=b"types.pyi", regular_file=True).view
        assert b"blue stub annotation" in decoded(stub_page)
        assert PRIVATE_COMMENT.strip() not in decoded(stub_page)
    else:
        assert "view_policy_binding_sha256" not in pages[0]
        assert "source_scope_sha256" not in service.descriptor


@pytest.mark.parametrize("change", ["extra", "row_extra", "content", "row_count", "offset_bool",
    "end", "total_bool", "duplicate", "unsorted", "path", "row_bool", "profile", "scope", "policy",
    "proof", "omission", "discharged_bool", "read_only", "current", "schema"])
def test_shared_discovery_response_tampering_is_not_a_valid_catalog(service_factory, monkeypatch, change):
    service = service_factory()
    authored_reader(service, monkeypatch)
    request = query(service)
    response = service._answer(client.validate_request(request), None)
    assert response["ok"] is True
    response = deepcopy(response)
    page = response["page"]
    if change == "extra": page["owner_locator"] = "/private/index.duckdb"
    elif change == "row_extra": page["rows"][0]["pieces"] = []
    elif change == "content": page["rows"][0]["content"] = "private source"
    elif change == "row_count": page["returned_count"] -= 1
    elif change == "offset_bool": page["offset"] = False
    elif change == "end": page["end_offset"] -= 1
    elif change == "total_bool": page["total_count"] = True
    elif change == "duplicate": page["rows"][1] = deepcopy(page["rows"][0])
    elif change == "unsorted": page["rows"][0], page["rows"][1] = page["rows"][1], page["rows"][0]
    elif change == "path": page["rows"][0]["path"] = "../private"
    elif change == "row_bool": page["rows"][0]["source_bytes"] = True
    elif change == "profile": page["rows"][0]["effective_profile"] = "foreign@1"
    elif change == "scope": page["scope_sha256"] = "0" * 64
    elif change == "policy": page["view_policy_binding_sha256"] = "0" * 64
    elif change == "proof": page["proof_authority"] = True
    elif change == "omission": page["omission_authority"] = True
    elif change == "discharged_bool": page["source_obligations_discharged"] = False
    elif change == "read_only": page["read_only"] = False
    elif change == "current": page["workspace_source_current"] = False
    else: page["schema"] = "supervisor-worker-source-view-discovery@1"
    with pytest.raises(ValueError):
        client.validate_response(response, request=request, descriptor=service.descriptor)


@pytest.mark.parametrize("change", ["missing_operation", "extra_path", "extra_state", "null_policy_read",
    "bool_policy", "operation", "cursor_schema", "max_bool", "max_small", "baseline_generation"])
def test_discriminated_requests_refuse_ambiguous_operations_before_transport(service_factory, change):
    service = service_factory()
    request = query(service, "read") if change in {"null_policy_read", "baseline_generation"} else query(service)
    selected = request["query"]
    if change == "missing_operation": selected.pop("operation")
    elif change == "extra_path": selected["path"] = "mod.py"
    elif change == "extra_state": selected["source_state"] = "working-candidate"
    elif change == "null_policy_read": selected["view_policy_binding_sha256"] = None
    elif change == "bool_policy": selected["view_policy_binding_sha256"] = True
    elif change == "operation": selected["operation"] = "write"
    elif change == "cursor_schema": selected["cursor"] = dict(schema="supervisor-worker-candidate-context-cursor@1",
        request_sha256="a" * 64, offset=1, token="b" * 32)
    elif change == "max_bool": selected["max_bytes"] = True
    elif change == "max_small": selected["max_bytes"] = 2047
    else: selected["expected_workspace_generation"] = "a" * 64
    with pytest.raises(ValueError):
        client.validate_request(request)


@pytest.mark.parametrize("change", ["null", "foreign"])
def test_shared_owner_refuses_mixed_policy_before_projection(service_factory, monkeypatch, change):
    service = service_factory()
    authored_reader(service, monkeypatch)
    monkeypatch.setattr(service.views, "discovery", lambda **kwargs: pytest.fail("mixed policy reached catalog"))
    request = client.validate_request(query(service,
        view_policy_binding_sha256=None if change == "null" else "0" * 64))
    response = service._answer(request, None)
    assert response["ok"] is False and response["reason_code"] == "invalid_request"
    with pytest.raises(ValueError):
        client.validate_response(response, request=request, descriptor=service.descriptor)


@pytest.mark.parametrize("working", [False, True])
def test_shared_owner_and_client_refuse_legacy_read_downgrade(service_factory, monkeypatch, working):
    service = service_factory()
    authored_reader(service, monkeypatch)
    modern = query(service, "read", source_state="working-candidate" if working else "baseline")
    good = service._answer(client.validate_request(modern), None)
    assert good["ok"] is True
    old = deepcopy(modern)
    old["schema"] = client.REQUEST_SCHEMA_V2 if working else client.REQUEST_SCHEMA
    old["query"].pop("operation")
    old["query"].pop("view_policy_binding_sha256")
    if not working:
        old["query"].pop("source_state")
        old["query"].pop("expected_workspace_generation")
    client.validate_request(old)
    response = service._answer(old, None)
    assert response["ok"] is False and response["reason_code"] == "invalid_request"
    with pytest.raises(ValueError):
        client.validate_response(good, request=old, descriptor=service.descriptor)


@pytest.mark.parametrize("change", ["missing_policy", "missing_scope", "bool_scope", "extra", "legacy_schema"])
def test_shared_descriptor_is_closed_and_cannot_shed_admission_pins(service_factory, change):
    descriptor = deepcopy(service_factory().descriptor)
    if change == "missing_policy": descriptor.pop("view_policy_binding_sha256")
    elif change == "missing_scope": descriptor.pop("source_scope_sha256")
    elif change == "bool_scope": descriptor["source_scope_sha256"] = True
    elif change == "extra": descriptor["inventory_path"] = "/private/custody"
    else: descriptor["schema"] = client.DESCRIPTOR_SCHEMA
    with pytest.raises(ValueError):
        client.validate_descriptor(descriptor)


@pytest.mark.parametrize("change", ["policy", "scope", "generation", "legacy_schema"])
def test_shared_read_response_joins_policy_scope_and_working_generation(service_factory, monkeypatch, change):
    service = service_factory()
    authored_reader(service, monkeypatch)
    working = change == "generation"
    first = answer(service, "read", source_state="working-candidate" if working else "baseline")["page"]
    request = query(service, "read", source_state="working-candidate" if working else "baseline",
        **({"expected_workspace_generation": first["workspace_generation"]} if working else {}))
    response = service._answer(client.validate_request(request), None)
    client.validate_response(response, request=request, descriptor=service.descriptor)
    if change == "policy": response["page"]["view_policy_binding_sha256"] = "0" * 64
    elif change == "scope": response["page"]["scope_sha256"] = "0" * 64
    elif change == "generation": response["page"]["workspace_generation"] = "0" * 64
    else: response["page"]["schema"] = "supervisor-worker-source-view-page@1"
    with pytest.raises(ValueError):
        client.validate_response(response, request=request, descriptor=service.descriptor)


@pytest.mark.parametrize("change", ["budget", "offset", "token", "read_cursor", "replay"])
def test_discovery_continuations_are_one_use_and_query_bound(service_factory, monkeypatch, change):
    service = service_factory()
    authored_reader(service, monkeypatch)
    first = answer(service, max_bytes=4096)["page"]
    cursor = deepcopy(first["next_cursor"])
    assert cursor is not None
    budget = 4096
    if change == "budget": budget += 1
    elif change == "offset": cursor["offset"] += 1
    elif change == "token": cursor["token"] = "0" * 32
    elif change == "read_cursor":
        cursor = answer(service, "read", view="original", max_bytes=4096)["page"]["next_cursor"]
        assert cursor is not None
        with pytest.raises(ValueError):
            client.validate_request(query(service, cursor=cursor, max_bytes=budget))
        # Source byte offsets cannot be used as discovery row offsets. Smaller
        # syntactically valid offsets are separately bound by the owner query.
        cursor["offset"] = 1
        response = answer(service, cursor=cursor, max_bytes=budget)
        assert response["ok"] is False and response["reason_code"] == "retrieval_refused"
        return
    else:
        assert answer(service, cursor=cursor, max_bytes=budget)["ok"] is True
    response = answer(service, cursor=cursor, max_bytes=budget)
    assert response["ok"] is False and response["reason_code"] == "retrieval_refused"


@pytest.mark.parametrize("change", ["canonical", "workspace", "artifact", "policy", "cached_source", "cached_view"])
def test_discovery_never_exports_pins_from_stale_or_corrupted_admission(
        service_factory, admitted, monkeypatch, change):
    service = service_factory()
    authored_reader(service, monkeypatch)
    if change in {"canonical", "workspace"}:
        root = admitted["root"] if change == "canonical" else admitted["workspace"]
        (root / "mod.py").write_bytes(MOD + b"# concurrent edit\n")
    elif change == "artifact": admitted["artifact"].write_text("{}")
    elif change == "policy": object.__setattr__(service.views._view_policy, "protected_lines", ())
    elif change == "cached_source":
        changed = MOD.replace(b"VALUE: int = 1", b"VALUE: int = 9")
        service.views._originals["mod.py"] = changed
        projected = service.views._view_policy.project(changed, raw_path=b"mod.py", regular_file=True)
        manifest = client.canonical_json(projected.manifest)
        service.views._views["mod.py"] = projected.view, manifest, sha(manifest)
    else:
        compact, raw, _ = service.views._views["mod.py"]
        manifest = json.loads(raw)
        compact += b"# fabricated compact body\n"
        manifest["view_sha256"] = sha(compact)
        manifest["view_bytes"] = len(compact)
        raw = client.canonical_json(manifest)
        service.views._views["mod.py"] = compact, raw, sha(raw)
    response = answer(service)
    assert response["ok"] is False
    assert response["reason_code"] in {"stale_context", "retrieval_refused"}
    assert response["page"] is None


def test_peer_replay_and_session_budget_refuse_before_discovery(service_factory, monkeypatch):
    service = service_factory()
    monkeypatch.setattr(service, "_peer", lambda channel: (_ for _ in ()).throw(ValueError("foreign peer")))
    monkeypatch.setattr(service.views, "discovery", lambda **kwargs: pytest.fail("refusal reached catalog"))
    request = client.validate_request(query(service))
    assert service._answer(request, None)["reason_code"] == "peer_refused"
    authored_reader(service, monkeypatch)
    assert service._answer(request, None)["reason_code"] == "invalid_request"
    service._response_bytes = broker.MAX_TOTAL_RESPONSE_BYTES
    assert answer(service)["reason_code"] == "budget_exhausted"


@pytest.mark.parametrize("field", ["view_policy_binding_sha256", "source_scope_sha256"])
def test_owner_session_descriptor_cannot_change_its_policy_or_scope_pin(service_factory, monkeypatch, field):
    service = service_factory()
    authored_reader(service, monkeypatch)
    request = client.validate_request(query(service))
    service.descriptor[field] = "0" * 64
    response = service._answer(request, None)
    assert response["ok"] is False and response["page"] is None
    assert response["reason_code"] in {"invalid_request", "stale_context"}


def test_discovery_post_projection_edit_refuses_the_prepared_rows(service_factory, admitted, monkeypatch):
    service = service_factory()
    authored_reader(service, monkeypatch)
    original = service.views.discovery

    def raced(**kwargs):
        page = original(**kwargs)
        (admitted["workspace"] / "mod.py").write_bytes(MOD + b"# after projection\n")
        return page

    monkeypatch.setattr(service.views, "discovery", raced)
    response = answer(service)
    assert response["ok"] is False and response["page"] is None
    assert response["reason_code"] == "stale_context"


def test_discovery_pins_to_working_generation_to_existing_candidate_context(
        service_factory, admitted, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.runtime.worker_candidate_context import WorkerCandidateContext

    service = service_factory()
    authored_reader(service, monkeypatch)
    service.candidate_context = WorkerCandidateContext(service.views,
        tuple(name for name in sorted(admitted["sources"]) if name.endswith(".py")), builder=fixture_builder)
    row = next(row for row in answer(service)["page"]["rows"] if row["path"] == "mod.py")
    baseline = answer(service, "read", **pinned(row))["page"]
    old = answer(service, "read", source_state="working-candidate")["page"]
    changed = MOD.replace(b"VALUE: int = 1", b"VALUE: int = 9")
    (admitted["workspace"] / "mod.py").write_bytes(changed)
    fresh = answer(service, "read", source_state="working-candidate")["page"]
    generation = fresh["workspace_generation"]
    assert generation != old["workspace_generation"]
    assert fresh["schema"] == "supervisor-worker-source-view-page@4"
    assert fresh["baseline_source_sha256"] == baseline["source_sha256"]
    assert fresh["baseline_evidence_applicable"] is False
    assert fresh["view_policy_binding_sha256"] == service.descriptor["view_policy_binding_sha256"]
    assert b"blue policy annotation" in decoded(fresh)
    expanded = answer(service, "read", source_state="working-candidate", view="original",
        expected_workspace_generation=generation, **pinned(fresh))["page"]
    assert decoded(expanded) == changed
    refused = answer(service, "read", source_state="working-candidate",
        expected_workspace_generation=old["workspace_generation"], **pinned(old))
    assert refused["ok"] is False
    request = candidate.validate_request(dict(schema=candidate.REQUEST_SCHEMA,
        session_id=service.descriptor["session_id"], request_id=uuid.uuid4().hex,
        query=dict(artifact="summary", expected_workspace_generation=generation, cursor=None, max_bytes=8192)))
    response = service._answer(request, None)
    candidate.validate_response(response, request=request, descriptor=service.descriptor)
    assert response["ok"] is True
    payload = json.loads(decoded(response["page"]))
    assert payload["generation"] == generation
    assert payload["paths"] == sorted(admitted["sources"])
    assert (admitted["root"] / "mod.py").read_bytes() == MOD
    assert answer(service)["reason_code"] == "stale_context"


@pytest.mark.parametrize("shared", [False, True])
def test_real_socket_discovery_counts_rows_separately_from_source_bytes(
        service_factory, monkeypatch, shared):
    service = service_factory(shared)
    authored_reader(service, monkeypatch)
    request = query(service, max_bytes=2048)
    response = socket_answer(service, request)
    assert response["ok"] is True
    service.close()
    audit = json.loads(service.audit_path.read_text())
    assert audit["schema"] == "supervisor-worker-source-session@5"
    assert audit["accepted_requests"] == len(audit["requests"]) == 1
    assert audit["response_bytes"] == len(client.canonical_json(response))
    entry = audit["requests"][0]
    assert entry["delivered"] is True
    assert entry["discovery_page"] is True
    assert entry["returned_discovery_rows"] == response["page"]["returned_count"]
    assert entry["returned_source_bytes"] == entry["returned_candidate_context_bytes"] == 0
    assert entry["candidate_context_page"] is False
    assert (entry["request_operation"], entry["request_source_state"], entry["request_view"]) == (
        "discover", "not_applicable", "not_applicable")
    assert PRIVATE_COMMENT.strip() not in client.canonical_json(audit)
    assert not service.socket_path.exists() and not service.descriptor_path.exists()
    assert audit["retrieval_provider_calls"] == audit["source_obligations_discharged"] == 0


def test_shared_dispatch_observation_joins_actual_descriptor_and_semantic_profile(
        service_factory, admitted, monkeypatch):
    service = service_factory()
    encoded = encode_semantic_router_prompt(prompt=admitted["prompt"], repository=admitted["root"])
    monkeypatch.setattr(client, "load_descriptor", lambda *args, **kwargs:
        client.validate_descriptor(deepcopy(service.descriptor)))
    observed = runner._source_broker_observation(path=service.descriptor_path,
        digest=service.descriptor_sha256, workspace=admitted["workspace"], encoded=encoded,
        boundary_sha256="b" * 64, profile="python-comments@1")
    assert observed["schema"] == "worker-source-dispatch-observation@2"
    assert observed["view_policy_binding_sha256"] == service.descriptor["view_policy_binding_sha256"]
    assert observed["source_scope_sha256"] == service.descriptor["source_scope_sha256"]
    assert observed["requests_observed"] is False
    assert observed["proof_authority"] is observed["execution_authority"] is observed["completion_authority"] is False
    with pytest.raises(ValueError, match="semantic evidence"):
        runner._source_broker_observation(path=service.descriptor_path, digest=service.descriptor_sha256,
            workspace=admitted["workspace"], encoded=encoded, boundary_sha256="b" * 64,
            profile="python-whitespace@1")


def test_owner_factory_binds_shared_policy_before_consuming_sealed_prompt(
        service_factory, admitted, monkeypatch):
    service = service_factory()
    selected = service.views._view_policy
    fd = _validation_memfd_create("discovery-qualification",
        getattr(os, "MFD_ALLOW_SEALING", 2) | getattr(os, "MFD_CLOEXEC", 1))
    try:
        os.write(fd, admitted["prompt"].encode())
        fcntl.fcntl(fd, getattr(fcntl, "F_ADD_SEALS", 1033), 15)
        position = os.lseek(fd, 0, os.SEEK_CUR)
        argv = ["--source-view-profile", "python-comments@1", "--semantic-repository", str(admitted["root"])]
        prepared = broker.prepare_worker_source_broker(argv=argv, boundary=service.boundary,
            workspace=admitted["workspace"], input_fd=fd, view_policy=selected)
        try:
            assert prepared.descriptor["schema"] == DESCRIPTOR_V2
            assert prepared.descriptor["view_policy_binding_sha256"] == service.descriptor["view_policy_binding_sha256"]
            assert prepared.descriptor["source_scope_sha256"] == service.descriptor["source_scope_sha256"]
            assert os.lseek(fd, 0, os.SEEK_CUR) == position
        finally:
            prepared.close()
        monkeypatch.setattr(os, "pread", lambda *args: pytest.fail("mismatched policy consumed prompt"))
        for selected_argv in ([], ["--source-view-profile", "python-whitespace@1"]):
            with pytest.raises(ValueError, match="policy"):
                broker.prepare_worker_source_broker(argv=selected_argv, boundary=service.boundary,
                    workspace=admitted["workspace"], input_fd=fd, view_policy=selected)
    finally:
        os.close(fd)


@pytest.mark.parametrize("mutation", [None, "disabled", "resolver"])
def test_owner_policy_disabled_or_resolver_drift_refuses_before_session(
        service_factory, admitted, monkeypatch, mutation):
    service = service_factory()
    selected = service.views._view_policy
    monkeypatch.setattr(os, "pread", lambda *args: pytest.fail("invalid owner policy consumed prompt"))
    if mutation == "disabled":
        object.__setattr__(selected, "profile", "disabled")
    elif mutation == "resolver":
        from ipfs_accelerate_py.agent_supervisor.runtime import source_view_policy as policy_module
        monkeypatch.setattr(policy_module, "current_codec_binding", lambda: {"changed": True})
        authored_reader(service, monkeypatch)
        assert answer(service)["reason_code"] == "stale_context"
        return
    with pytest.raises(ValueError):
        broker.prepare_worker_source_broker(argv=["--source-view-profile", "disabled"],
            boundary=service.boundary, workspace=admitted["workspace"], input_fd=-1, view_policy=selected)


# Only descriptor filesystem ownership is authored here. The executable still
# loads its fixed standard-library sibling under -I, checks canonical bytes and
# digest, and performs real client peer checks and framed broker IO.
ISOLATED_CLI = r'''
import hashlib, importlib.util, pathlib, sys
spec = importlib.util.spec_from_file_location("isolated_source_reader", sys.argv[1])
reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader)
def authored_descriptor(path, digest, *, workspace=None):
    raw = pathlib.Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError("descriptor digest mismatch")
    value = reader.validate_descriptor(reader.strict_json_loads(raw))
    if raw != reader.canonical_json(value) or str(pathlib.Path.cwd()) != value["workspace"]:
        raise ValueError("descriptor workspace mismatch")
    return value
reader.load_descriptor = authored_descriptor
sys.stdin.buffer.read(1)
raise SystemExit(reader.main(sys.argv[2:]))
'''


def isolated_cli(service, flags):
    # Retain the production registered-child birth/ancestry checks; author only
    # the worker UID because this provider-free fixture runs under one UID.
    service.worker_uid = os.getuid()
    process = subprocess.Popen([sys.executable, "-I", "-c", ISOLATED_CLI,
        str(Path(client.__file__)), *service.extra_argv, *flags], cwd=service.workspace,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        service.note_spawned(process)
        stdout, stderr = process.communicate("x", timeout=15)
        return process.returncode, stdout, stderr
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()


@pytest.mark.parametrize("shared,raw", [(False, False), (True, False), (True, True)])
def test_isolated_cli_discovers_through_real_registered_child_socket(service_factory, shared, raw):
    service = service_factory(shared)
    code, stdout, stderr = isolated_cli(service, ["--discover", "--max-bytes", "2048"]
        + (["--raw-response"] if raw else []))
    assert code == 0 and not stderr
    value = json.loads(stdout)
    page = value["page"] if raw else value
    assert page["returned_count"] == len(page["rows"]) > 0
    assert page["total_count"] == 14
    assert page["next_cursor"] is not None
    if not raw:
        assert set(page) == {"ok", "total_count", "returned_count", "offset", "end_offset", "rows",
                             "exhausted", "next_cursor"} | ({"view_policy_binding_sha256"} if shared else set())
    assert PRIVATE_COMMENT.strip().decode() not in stdout
    service.close()
    entry = json.loads(service.audit_path.read_text())["requests"][0]
    assert entry["delivered"] is entry["discovery_page"] is True
    assert entry["returned_source_bytes"] == 0


@pytest.mark.parametrize("working", [False, True])
def test_isolated_cli_automatically_reads_shared_stub_with_protected_comment(service_factory, working):
    service = service_factory()
    code, stdout, stderr = isolated_cli(service,
        ["--path", "types.pyi", "--raw-response"] + (["--working"] if working else []))
    assert code == 0 and not stderr
    response = json.loads(stdout)
    page = response["page"]
    assert page["schema"] == f"supervisor-worker-source-view-page@{4 if working else 3}"
    assert page["view_policy_binding_sha256"] == service.descriptor["view_policy_binding_sha256"]
    assert b"blue stub annotation" in decoded(page)
    assert PRIVATE_COMMENT.strip() not in decoded(page)
    if working:
        assert page["source_state"] == "working-candidate"
        assert page["baseline_evidence_applicable"] is False
    service.close()
    entry = json.loads(service.audit_path.read_text())["requests"][0]
    assert entry["discovery_page"] is False and entry["returned_discovery_rows"] == 0
    assert entry["returned_source_bytes"] == len(decoded(page))


@pytest.mark.parametrize("flags", [["--view", "compact"], ["--working"],
    ["--expected-source-sha256", "a" * 64], ["--expected-view-sha256", "a" * 64],
    ["--expected-workspace-generation", "a" * 64]])
def test_cli_discovery_rejects_source_only_flags_before_loading_descriptor(monkeypatch, capsys, flags):
    monkeypatch.setattr(client, "load_descriptor", lambda *args, **kwargs: pytest.fail("ambiguous CLI reached owner"))
    code = client.main(["--source-view-descriptor", "/not-read", "--source-view-descriptor-sha256",
        "a" * 64, "--discover", *flags])
    captured = capsys.readouterr()
    assert code == 2 and captured.out == ""
    assert captured.err == '{"reason_code":"transport_error"}\n'


def test_standalone_cli_help_exposes_exclusive_discovery_without_package_loading():
    result = subprocess.run([sys.executable, "-I", str(Path(client.__file__)), "--help"],
        capture_output=True, text=True, timeout=10)
    assert result.returncode == 0 and not result.stderr
    assert "--discover" in result.stdout and "--path" in result.stdout
    result = subprocess.run([sys.executable, "-I", str(Path(client.__file__)),
        "--discover", "--path", "mod.py"], capture_output=True, text=True, timeout=10)
    assert result.returncode == 2 and not result.stdout
    assert "not allowed with argument" in result.stderr
