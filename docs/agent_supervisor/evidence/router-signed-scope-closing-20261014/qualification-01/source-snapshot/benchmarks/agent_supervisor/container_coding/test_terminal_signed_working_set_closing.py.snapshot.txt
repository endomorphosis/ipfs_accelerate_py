"""Signed full-source dispatch closes after an authenticated-shaped page reply.

Git, signatures, native owners, compiled capsules, complete signed inventories,
linked allocation, source views and capsule paging are real. Boundary/UID and
same-process endpoint routing are authored seams: no socket, deployed worker,
provider, model, solver, target program or Docker invocation is established.
"""
from contextlib import contextmanager
import json
import os
from types import SimpleNamespace

import pytest

from benchmarks.agent_supervisor.container_coding.test_terminal_worker_signed_scope import (
    invoke, sha, worker,  # noqa: F401
)
from benchmarks.agent_supervisor.container_coding.test_terminal_task_profile import git
from ipfs_accelerate_py.agent_supervisor.runtime import container_worker_boundary as boundary
from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner as runner
from ipfs_accelerate_py.agent_supervisor.runtime import router_public_instruction as instruction
from ipfs_accelerate_py.agent_supervisor.runtime import semantic_router_translation as codec
from ipfs_accelerate_py.agent_supervisor.runtime import terminal_task_profile as profiles
from ipfs_accelerate_py.agent_supervisor.runtime import worker_semantic_evidence_client as evidence
from ipfs_accelerate_py.agent_supervisor.runtime import worker_source_client as source
from ipfs_accelerate_py.agent_supervisor.runtime.worker_semantic_evidence import WorkerSemanticEvidence
from ipfs_accelerate_py.agent_supervisor.runtime.worker_source_views import WorkerSourceViews
from test.api.test_semantic_router_integration import provider  # noqa: F401


@pytest.fixture
def working(worker, monkeypatch):
    encoded = codec.encode_semantic_router_prompt(prompt=worker.prompt, repository=worker.case.root)
    table = encoded.table.to_dict()
    binding = source.semantic_binding(encoded, "python-whitespace@1")
    views = WorkerSourceViews(repository=worker.case.root, workspace=worker.workspace,
        source_hashes={name: row["sha256"] for name, row in table["source_manifest"].items()}, **binding)
    descriptor_path = worker.case.state / "authored-working-set-descriptor.json"
    descriptor = dict(schema=source.DESCRIPTOR_SCHEMA_V4, session_id="1" * 32,
        owner_uid=1000, worker_uid=1001, owner_pid=os.getpid(), owner_start_time_ticks=1,
        workspace=str(worker.workspace), socket_path=str(worker.case.state / "authored-unused.sock"),
        socket_device=0, socket_inode=1, issued_monotonic_ns=1, expires_monotonic_ns=1000000001,
        max_requests=16, max_total_response_bytes=65536, boundary_manifest_sha256="b" * 64,
        source_edit_mode="disabled", semantic_evidence_mode=evidence.MODE,
        source_scope_sha256=views.admission()["scope_sha256"], view_policy_binding_sha256=None,
        proof_authority=False, execution_authority=False, completion_authority=False, **binding)
    source.validate_descriptor(descriptor)
    raw = source.canonical_json(descriptor)
    descriptor_path.write_bytes(raw)
    descriptor_path.chmod(0o400)
    digest = sha(raw)
    # Payload bytes are the original native producer artifact, never invented
    # capsules or a replacement public-instruction admission.
    semantic = codec._parse(codec._read(worker.case.root, table["semantic_artifact"]).decode())
    pager = WorkerSemanticEvidence(capsules=semantic["capsules"], binding=binding)
    state = SimpleNamespace(encoded=encoded, descriptor=descriptor, pager=pager, views=views,
        events=[], after_page=None, load_events=[], descriptor_path=descriptor_path, digest=digest)

    def load(path, expected_digest, *, workspace=None):
        assert path == descriptor_path and expected_digest == digest and workspace == worker.workspace
        current = path.read_bytes()
        assert sha(current) == digest
        value = source.validate_descriptor(source.strict_json_loads(current))
        assert source.canonical_json(value) == current
        return value

    def page(path, expected_digest, *, workspace, capsule_index, max_bytes):
        current = load(path, expected_digest, workspace=workspace)
        views.admission()
        request = evidence.validate_request(dict(schema=evidence.REQUEST_SCHEMA,
            session_id=current["session_id"], request_id="2" * 32,
            query=dict(capsule_index=capsule_index, cursor=None, max_bytes=max_bytes)))
        response = dict(schema=source.RESPONSE_SCHEMA, session_id=current["session_id"],
            request_id=request["request_id"], ok=True,
            page=pager.page(**request["query"]), reason_code=None)
        views.admission()
        evidence.validate_response(response, request=request, descriptor=current)
        state.events.append("valid_page")
        if state.after_page is not None:
            state.after_page()
            state.events.append("changed_after_page")
        return response

    original_instruction_loader = instruction.load_public_instruction

    def load_instruction(**kwargs):
        state.load_events.append(dict(after_page=bool(state.events),
            current=kwargs.get("require_current_source", True)))
        return original_instruction_loader(**kwargs)

    monkeypatch.setattr(source, "load_descriptor", load)
    monkeypatch.setattr(boundary, "verify_container_worker_boundary",
        lambda **kwargs: {"authored_uid_boundary": True})
    monkeypatch.setattr(evidence, "request_page", page)
    monkeypatch.setattr(instruction, "load_public_instruction", load_instruction)
    state.options = dict(container_boundary=worker.case.state / "authored-boundary.json",
        container_boundary_sha256="b" * 64, source_view_profile=binding["profile"],
        source_view_descriptor=descriptor_path, source_view_descriptor_sha256=digest,
        semantic_working_set="capsule-references@1")
    yield state
    pager.clear()


def _failed_receipt(capsys):
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")]
    return next(row for row in rows if row.get("schema") == "router-implementation-invocation@1")


def _mutate(worker, monkeypatch, location, change):
    root = worker.case.root if location == "canonical" else worker.workspace
    if change == "head":
        git(root, "-c", "user.name=Closing check fixture", "-c", "user.email=close@example.invalid",
            "commit", "--allow-empty", "-qm", "Authored callback baseline drift")
    elif change == "artifact-bytes":
        path = worker.request["artifact"]
        path.chmod(0o600)
        path.write_bytes(path.read_bytes() + b"\n")
        path.chmod(0o400)
    elif change == "artifact-mode":
        worker.request["artifact"].chmod(0o600)
    elif change == "cwd":
        monkeypatch.chdir(worker.case.root)
    else:
        name = {"selected-bytes": worker.case.selected[0], "selected-mode": worker.case.selected[0],
                "support-bytes": profiles.INSTRUCTION, "profile-bytes": profiles.PROFILE,
                "omitted-bytes": "module_069.py"}[change]
        path = root / name
        if change == "selected-mode":
            path.chmod(path.stat().st_mode ^ 0o111)
        else:
            path.write_bytes(path.read_bytes() + b"\n# Authored post-page drift\n")


def test_signed_working_set_dispatch_rechecks_complete_scope_after_page(worker, working, provider):
    from ipfs_accelerate_py.agent_supervisor.runtime.semantic_working_set import restore_semantic_working_set
    _, observed = provider
    before = {name: (worker.case.root / name).read_bytes() for name in worker.case.prepared["worker_inputs"]}
    output, receipt = invoke(worker, **working.options)
    assert output == "literal summary" and len(observed) == 1
    assert working.events == ["valid_page"]
    assert working.load_events == [dict(after_page=False, current=True), dict(after_page=True, current=True)]
    assert receipt["terminal_semantic_scope_dispatch_rechecked"] is True and receipt["router_calls"] == 1
    scope = receipt["public_instruction"]["terminal_semantic_scope"]
    assert (scope["selected_count"], scope["full_count"], scope["frontier_count"]) == (5, 73, 68)
    assert scope["program_count"] == 2 and scope["required_count"] == 4
    assert all(scope[key] is False for key in ("native_claim_verified", "native_allocation_verified",
        "process_identity_verified", "proof_authority", "execution_authority", "completion_authority", "omission_authority"))
    selected = receipt["semantic_working_set"]
    assert selected["selected_mode"] == "capsule-references@1"
    assert receipt["semantic_working_set_preflight"]["requests"] == 1
    actual = observed[0][0]
    wire, _ = json.JSONDecoder().raw_decode(actual[receipt["workspace_advisory_bytes"]:])
    native = json.loads(working.encoded.provider_prompt)
    assert wire["native_context"] == native["native_context"]
    assert wire["native_suffix"] == native["native_suffix"]
    assert wire["translated_semantic"]["raw_sources"] == native["translated_semantic"]["raw_sources"]
    assert (worker.case.root / profiles.INSTRUCTION).read_text() in actual

    def resolve(reference):
        cursor, chunks = None, []
        while True:
            page = working.pager.page(capsule_index=reference["capsule_index"], cursor=cursor, max_bytes=4096)
            import base64
            chunks.append(page["content"].encode() if page["encoding"] == "utf-8"
                          else base64.b64decode(page["content"], validate=True))
            cursor = page["next_cursor"]
            if cursor is None:
                break
        raw = b"".join(chunks)
        assert sha(raw) == reference["capsule_sha256"]
        return raw

    restored = restore_semantic_working_set(provider_prompt=source.canonical_json(wire).decode(),
        receipt=selected["view_receipt"], encoded=working.encoded, descriptor=working.descriptor, resolver=resolve)
    assert restored == working.encoded.provider_prompt
    assert all((root / name).read_bytes() == raw for root in (worker.case.root, worker.workspace)
               for name, raw in before.items())


_DRIFTS = [(location, change) for location in ("canonical", "allocated") for change in (
    "selected-bytes", "selected-mode", "support-bytes", "profile-bytes", "omitted-bytes", "head")]
_DRIFTS += [("canonical", change) for change in ("artifact-bytes", "artifact-mode", "cwd")]


@pytest.mark.parametrize("location,change", _DRIFTS, ids=[a + "-" + b for a, b in _DRIFTS])
def test_post_page_drift_refuses_signed_dispatch(worker, working, provider, monkeypatch, capsys, location, change):
    _, observed = provider
    working.after_page = lambda: _mutate(worker, monkeypatch, location, change)
    capsys.readouterr()
    with pytest.raises(ValueError):
        invoke(worker, **working.options)
    receipt = _failed_receipt(capsys)
    assert working.events == ["valid_page", "changed_after_page"] and observed == []
    assert working.load_events == [dict(after_page=False, current=True), dict(after_page=True, current=True)]
    assert receipt["semantic_working_set"]["selected_mode"] == "capsule-references@1"
    assert receipt["semantic_working_set_preflight"]["requests"] == 1
    assert receipt["status"] == "failed" and receipt["failure_phase"] == "terminal_semantic_scope_currentness"
    assert receipt["terminal_semantic_scope_dispatch_rechecked"] is False
    assert receipt["router_calls"] == 0 and receipt["completion_authority"] is False


@pytest.mark.parametrize("change", ["omitted-bytes", "cwd"])
def test_descriptor_context_entry_drift_refuses_signed_dispatch(worker, working, provider, monkeypatch, capsys, change):
    _, observed = provider
    original = runner._source_descriptor_environment

    @contextmanager
    def enter(path, digest):
        with original(path, digest):
            _mutate(worker, monkeypatch, "canonical", change)
            working.events.append("changed_at_descriptor_entry")
            yield

    monkeypatch.setattr(runner, "_source_descriptor_environment", enter)
    capsys.readouterr()
    with pytest.raises(ValueError):
        invoke(worker, **working.options)
    receipt = _failed_receipt(capsys)
    assert working.events == ["valid_page", "changed_at_descriptor_entry"] and observed == []
    assert receipt["failure_phase"] == "terminal_semantic_scope_currentness"
    assert receipt["terminal_semantic_scope_dispatch_rechecked"] is False and receipt["router_calls"] == 0
