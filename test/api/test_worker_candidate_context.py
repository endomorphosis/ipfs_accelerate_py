"""Exact candidate custody and bounded paging, without model or database calls."""
import base64
from copy import deepcopy
import hashlib
import json
import socket
import subprocess
import threading
import uuid

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import worker_candidate_context as module
from ipfs_accelerate_py.agent_supervisor.runtime.worker_source_views import WorkerSourceViews, canonical_json
from test.api.test_worker_source_broker import allocated, native, service as broker_service  # noqa: F401


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def git(root, *args):
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)


@pytest.fixture
def case(tmp_path):
    root, workspace = tmp_path / "canonical", tmp_path / "allocated"
    root.mkdir()
    sources = {"code.py": b"def increment(value):\n    return value + 1\n",
        "other.py": b"def decrease(value):\n    return value - 1\n", "support.txt": b"public data\n"}
    for path, raw in sources.items():
        (root / path).write_bytes(raw)
    git(root, "init", "-q")
    git(root, "add", ".")
    git(root, "-c", "user.name=Authored Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "fixture")
    git(root, "worktree", "add", "--detach", "-q", str(workspace))
    views = WorkerSourceViews(repository=root, workspace=workspace,
        source_hashes={name: sha(raw) for name, raw in sources.items()}, task_id="authored-task",
        scope_cid="scope:authored", semantic_root_cid="semantic:authored", semantic_sha256="a" * 64,
        native_prompt_sha256="b" * 64, source_manifest_sha256="c" * 64)
    generation = views.working_page(path="code.py")["workspace_generation"]
    calls = []

    def builder(**kwargs):
        calls.append(deepcopy(kwargs))
        return {artifact: canonical_json(dict(schema="authored-artifact@1", artifact=artifact,
            workspace_generation=kwargs["workspace_generation"],
            source_hashes={path: sha(raw) for path, raw in kwargs["sources"].items()},
            content="candidate λ🙂\\\"\n" * 500)) for artifact in module.ARTIFACTS}

    service = module.WorkerCandidateContext(views, ("code.py", "other.py"), builder=builder)
    return dict(root=root, workspace=workspace, sources=sources, views=views, generation=generation,
        calls=calls, builder=builder, service=service)


def request(case, **kwargs):
    return case["service"].page(**dict(artifact="summary",
        expected_workspace_generation=case["generation"], max_bytes=2400) | kwargs)


def decode(page):
    raw = (page["content"].encode() if page["encoding"] == "utf-8"
        else base64.b64decode(page["content"], validate=True))
    assert sha(raw) == page["slice_sha256"]
    assert len(raw) == page["end_offset"] - page["offset"]
    return raw


def assert_discarded(case):
    assert case["service"]._generation is None
    assert case["service"]._artifacts == case["service"]._continuations == {}


def test_owner_accessor_returns_isolated_exact_byte_map_without_refresh(case):
    before = case["views"]._working_counter
    captured = case["views"].working_sources(expected_workspace_generation=case["generation"])
    assert captured == case["sources"]
    captured["code.py"] = b"changed only caller dictionary"
    assert case["views"].working_sources(expected_workspace_generation=case["generation"]) == case["sources"]
    assert case["views"]._working_counter == before


@pytest.mark.parametrize("generation", [None, "", "a" * 64, 1, True])
def test_owner_accessor_refuses_missing_foreign_generation_and_discards(case, generation):
    with pytest.raises(ValueError):
        case["views"].working_sources(expected_workspace_generation=generation)
    assert case["views"]._working is None


def test_owner_accessor_requires_explicit_generation(case):
    with pytest.raises(TypeError):
        case["views"].working_sources()


def test_owner_accessor_rejects_corrupted_retained_bytes(case):
    case["views"]._working._originals["code.py"] = b"corrupted retained bytes"
    with pytest.raises(ValueError, match="captured working source differs"):
        case["views"].working_sources(expected_workspace_generation=case["generation"])
    assert case["views"]._working is None


@pytest.mark.parametrize("artifact", sorted(module.ARTIFACTS))
def test_exact_artifact_reconstruction_and_complete_envelope_bound(case, artifact):
    chunks, pages, cursor, offset = [], [], None, 0
    for _ in range(100):
        page = request(case, artifact=artifact, cursor=cursor)
        assert page["offset"] == offset
        assert len(canonical_json(page)) <= 2400
        chunks.append(decode(page))
        pages.append(page)
        offset = page["end_offset"]
        cursor = page["next_cursor"]
        assert page["exhausted"] is (cursor is None)
        assert page["source_state"] == "working-candidate"
        assert page["baseline_evidence_applicable"] is page["proof_authority"] is page["completion_authority"] is False
        assert type(page["source_obligations_discharged"]) is int and page["source_obligations_discharged"] == 0
        if cursor is None:
            break
    else:
        pytest.fail("candidate pagination did not terminate")
    result = b"".join(chunks)
    assert len(pages) > 1
    assert len(result) == pages[0]["artifact_bytes"] and sha(result) == pages[0]["artifact_sha256"]
    assert result == case["service"]._artifacts[artifact]
    assert json.loads(result)["workspace_generation"] == case["generation"]
    assert len(case["calls"]) == 1
    assert case["calls"][0]["program_paths"] == ("code.py", "other.py")
    assert case["calls"][0]["sources"] == case["sources"]
    assert set(pages[0]) == {"schema", "binding", "workspace_generation", "source_state",
        "baseline_evidence_applicable", "proof_authority", "completion_authority", "source_obligations_discharged",
        "artifact", "artifact_sha256", "artifact_bytes", "offset", "end_offset", "encoding", "content",
        "slice_sha256", "exhausted", "next_cursor"}


def test_multiple_artifacts_and_first_pages_build_once(case):
    for artifact in ["summary", "capsules", "formal", "summary"]:
        assert request(case, artifact=artifact)["artifact"] == artifact
    assert len(case["calls"]) == 1
    case["service"].clear()
    assert_discarded(case)
    request(case)
    assert len(case["calls"]) == 2


@pytest.mark.parametrize("location", ["workspace", "root"])
@pytest.mark.parametrize("path", ["code.py", "other.py", "support.txt"])
def test_complete_scope_mutation_rejects_even_cached_page(case, location, path):
    first = request(case)
    (case[location] / path).write_bytes(b"changed source\n")
    with pytest.raises(ValueError):
        request(case, cursor=first["next_cursor"])
    assert_discarded(case)
    assert len(case["calls"]) == 1


def test_cross_file_mutation_during_build_is_not_published(case):
    def builder(**kwargs):
        result = case["builder"](**kwargs)
        (case["workspace"] / "other.py").write_bytes(b"VALUE = 999\n")
        return result
    case["service"] = module.WorkerCandidateContext(case["views"], ("code.py",), builder=builder)
    with pytest.raises(ValueError):
        request(case)
    assert_discarded(case)
    assert case["views"]._working is None


def test_cache_hit_checks_again_after_initial_capture(case, monkeypatch):
    request(case)
    original = case["views"].working_sources
    calls = []
    def racing(**kwargs):
        value = original(**kwargs)
        calls.append(True)
        if len(calls) == 1:
            (case["workspace"] / "support.txt").write_bytes(b"changed support\n")
        return value
    monkeypatch.setattr(case["views"], "working_sources", racing)
    with pytest.raises(ValueError):
        request(case)
    assert_discarded(case)
    assert len(case["calls"]) == 1


def test_edit_needs_explicit_working_source_refresh_then_builds_new_generation(case):
    first = request(case)
    (case["workspace"] / "code.py").write_bytes(b"def increment(value):\n    return value + 2\n")
    with pytest.raises(ValueError):
        request(case)
    assert_discarded(case)
    current = case["views"].working_page(path="code.py")["workspace_generation"]
    assert current != case["generation"]
    page = request(case, expected_workspace_generation=current)
    assert page["workspace_generation"] == current
    assert page["artifact_sha256"] != first["artifact_sha256"]
    assert len(case["calls"]) == 2
    assert case["calls"][1]["sources"]["code.py"].endswith(b"value + 2\n")


@pytest.mark.parametrize("change", ["artifact", "budget", "offset", "token", "digest", "schema", "extra"])
def test_foreign_or_forged_continuation_discards_cache(case, change):
    first = request(case)
    cursor = deepcopy(first["next_cursor"])
    kwargs = dict(cursor=cursor)
    if change == "artifact":
        kwargs["artifact"] = "formal"
    elif change == "budget":
        kwargs["max_bytes"] = 2500
    elif change == "offset":
        cursor["offset"] += 1
    elif change == "token":
        cursor["token"] = "0" * 32
    elif change == "digest":
        cursor["request_sha256"] = "0" * 64
    elif change == "schema":
        cursor["schema"] = "foreign"
    else:
        cursor["extra"] = True
    with pytest.raises(ValueError):
        request(case, **kwargs)
    assert_discarded(case)


def test_consumed_continuation_and_foreign_session_are_rejected(case):
    cursor = request(case)["next_cursor"]
    request(case, cursor=cursor)
    with pytest.raises(ValueError):
        request(case, cursor=cursor)
    assert_discarded(case)
    case["service"] = module.WorkerCandidateContext(case["views"], ("code.py",), builder=case["builder"])
    with pytest.raises(ValueError):
        request(case, cursor=cursor)
    assert_discarded(case)


def test_continuations_from_previous_generation_are_never_adopted(case):
    cursor = request(case)["next_cursor"]
    (case["workspace"] / "code.py").write_bytes(b"VALUE = 2\n")
    new = case["views"].working_page(path="code.py")["workspace_generation"]
    with pytest.raises(ValueError):
        request(case, expected_workspace_generation=new, cursor=cursor)
    assert_discarded(case)
    assert len(case["calls"]) == 1


@pytest.mark.parametrize("value", [True, 1, 1023, 65537, "8192", None])
def test_invalid_page_budget_clears_cache(case, value):
    request(case)
    with pytest.raises(ValueError):
        request(case, max_bytes=value)
    assert_discarded(case)


def test_page_that_cannot_fit_content_refuses_without_continuation(case):
    with pytest.raises(ValueError, match="cannot fit"):
        request(case, max_bytes=1024)
    assert_discarded(case)


@pytest.mark.parametrize("paths", [["code.py", "code.py"], ["../code.py"], ["/code.py"], [".git/config"], "code.py"])
def test_invalid_program_population_refused(case, paths):
    with pytest.raises(ValueError):
        module.WorkerCandidateContext(case["views"], paths, builder=case["builder"])


def test_unadmitted_program_path_never_reaches_builder(case):
    case["service"] = module.WorkerCandidateContext(case["views"], ("private.py",), builder=case["builder"])
    with pytest.raises(ValueError):
        request(case)
    assert case["calls"] == []
    assert_discarded(case)


@pytest.mark.parametrize("kind", ["extra", "missing", "text", "empty", "list", "pretty", "duplicate", "nan", "large", "raised"])
def test_builder_failure_or_invalid_artifact_discards_everything(case, kind):
    def builder(**kwargs):
        result = case["builder"](**kwargs)
        if kind == "extra": result["other"] = b"{}"
        elif kind == "missing": del result["formal"]
        elif kind == "text": result["summary"] = "{}"
        elif kind == "empty": result["summary"] = b""
        elif kind == "list": result["summary"] = b"[]"
        elif kind == "pretty": result["summary"] = b'{ "x": 1 }'
        elif kind == "duplicate": result["summary"] = b'{"x":1,"x":1}'
        elif kind == "nan": result["summary"] = b'{"x":NaN}'
        elif kind == "large": result["summary"] = b"x" * (module.MAX_ARTIFACT_BYTES + 1)
        else: raise RuntimeError("authored builder failure")
        return result
    case["service"] = module.WorkerCandidateContext(case["views"], ("code.py",), builder=builder)
    with pytest.raises((ValueError, RuntimeError)):
        request(case)
    assert_discarded(case)


def test_cursor_population_bound_and_single_use_replacement(case, monkeypatch):
    monkeypatch.setattr(module, "MAX_CONTINUATIONS", 2)
    one = request(case)
    request(case, artifact="formal")
    # Consuming an existing cursor frees exactly the slot for its successor.
    request(case, cursor=one["next_cursor"])
    assert len(case["service"]._continuations) == 2
    with pytest.raises(ValueError, match="population"):
        request(case, artifact="capsules")
    assert_discarded(case)


def test_support_only_generation_preserves_scope_without_promoting_source_to_program(case):
    case["service"] = module.WorkerCandidateContext(case["views"], ())
    page = request(case, max_bytes=65536)
    assert page["exhausted"] is True
    summary = json.loads(decode(page))
    assert summary["program_paths"] == [] and summary["program_files"] == 0
    assert summary["support_files"] == summary["source_files"] == len(case["sources"])
    assert summary["capsule_count"] == summary["formal_report_count"] == summary["pending_obligation_count"] == 0
    assert summary["source_hashes"] == {name: sha(raw) for name, raw in case["sources"].items()}
    assert all(row["role"] == "support" and row["formal_status"] == "not_selected_support"
        for row in summary["coverage"])
    assert case["calls"] == []


@pytest.mark.parametrize("when", ["before", "after"])
def test_cached_native_artifact_refuses_producer_drift_before_and_after_read(case, monkeypatch, when):
    from ipfs_accelerate_py.agent_supervisor.runtime import worker_candidate_artifacts as artifacts
    case["service"] = module.WorkerCandidateContext(case["views"], ())
    request(case, max_bytes=65536)
    original = artifacts._producer_binding()
    changed = dict(original)
    changed[next(iter(changed))] = "0" * 64
    calls = []
    def binding():
        calls.append(True)
        return changed if when == "before" or len(calls) > 1 else original
    monkeypatch.setattr(artifacts, "_producer_binding", binding)
    with pytest.raises(ValueError, match="producer sources changed"):
        request(case)
    assert_discarded(case)
    assert case["service"]._producers is None


def test_broker_stop_during_candidate_build_refuses_completed_artifact_and_clears_cache(broker_service, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.runtime import worker_source_client as protocol
    value = broker_service
    generation = value.views.working_page(path="mod.py")["workspace_generation"]
    entered, release = threading.Event(), threading.Event()
    def blocked_builder(**kwargs):
        entered.set()
        assert release.wait(timeout=10), "authored build was not released"
        return {key: canonical_json(dict(schema="authored-artifact@1", artifact=key)) for key in module.ARTIFACTS}
    value.candidate_context = module.WorkerCandidateContext(value.views, ("mod.py",), builder=blocked_builder)
    monkeypatch.setattr(value, "_peer", lambda channel: None)
    request_value = protocol.validate_request(dict(schema="supervisor-worker-candidate-context-request@1",
        session_id=value.descriptor["session_id"], request_id=uuid.uuid4().hex,
        query=dict(artifact="summary", expected_workspace_generation=generation, cursor=None, max_bytes=8192)))
    responses, answer = [], value._answer
    def observe(request_value, channel):
        response = answer(request_value, channel)
        responses.append(response)
        return response
    monkeypatch.setattr(value, "_answer", observe)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as channel:
        channel.settimeout(10)
        channel.connect(str(value.socket_path))
        protocol.send_frame(channel, request_value)
        try:
            assert entered.wait(timeout=10), "candidate build did not start"
            value.close()
            audit = json.loads(value.audit_path.read_bytes())
            assert audit["service_stopped"] is audit["accounting_complete"] is False
            assert audit["accepted_requests"] == audit["inflight_request"] == 1
            assert audit["requests"] == [] and audit["response_bytes"] == 0
            assert not value.socket_path.exists() and not value.descriptor_path.exists()
        finally:
            release.set()
            value._thread.join(timeout=10)
        assert channel.recv(1) == b""
    assert not value._thread.is_alive()
    assert len(responses) == 1 and responses[0]["ok"] is False
    assert responses[0]["page"] is None and responses[0]["reason_code"] == "stale_context"
    assert value.candidate_context._generation is None
    assert value.candidate_context._artifacts == value.candidate_context._continuations == {}
    assert len(value._records) == 1 and value._records[0]["delivered"] is False
    assert value._records[0]["response_bytes"] == value._records[0]["returned_candidate_context_bytes"] == 0
