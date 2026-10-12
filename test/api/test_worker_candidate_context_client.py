"""Closed candidate wire contracts, isolated CLI and real framed socket reads."""
import base64
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import threading

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import worker_candidate_context_client as client
from test.api.test_numerical_obligation_client import descriptor as numeric_descriptor, owner_files  # noqa: F401


BINDING = dict(task_id="candidate-task", scope_cid="scope:1", semantic_root_cid="semantic:1",
    semantic_sha256="c" * 64, native_prompt_sha256="d" * 64,
    source_manifest_sha256="e" * 64, profile="python-comments@1")
GENERATION = "f" * 64


def descriptor():
    value = numeric_descriptor()
    for key in ("task_cid", "context_bundle_sha256", "context_receipt_sha256", "source_tree_id"):
        value.pop(key)
    return dict(value, **{key: item for key, item in BINDING.items() if key not in value},
        schema=client.source.DESCRIPTOR_SCHEMA) | BINDING


def request(**query):
    return dict(schema=client.REQUEST_SCHEMA, session_id="a" * 32, request_id="b" * 32,
        query=dict(artifact="summary", expected_workspace_generation=GENERATION,
            cursor=None, max_bytes=8192) | query)


def cursor(offset):
    return dict(schema=client.CURSOR_SCHEMA, request_sha256="1" * 64, offset=offset, token="2" * 32)


def response(query=None, *, raw=None, end=None):
    query = request() if query is None else query
    raw = client.canonical_json({"candidate": "λ", "pending": True}) if raw is None else raw
    offset = 0 if query["query"]["cursor"] is None else query["query"]["cursor"]["offset"]
    end = len(raw) if end is None else end
    part = raw[offset:end]
    try:
        content, encoding = part.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        content, encoding = base64.b64encode(part).decode("ascii"), "base64"
    page = dict(schema=client.PAGE_SCHEMA, binding=dict(BINDING), workspace_generation=GENERATION,
        source_state="working-candidate", baseline_evidence_applicable=False,
        proof_authority=False, completion_authority=False, source_obligations_discharged=0,
        artifact=query["query"]["artifact"], artifact_sha256=hashlib.sha256(raw).hexdigest(),
        artifact_bytes=len(raw), offset=offset, end_offset=end, encoding=encoding, content=content,
        slice_sha256=hashlib.sha256(part).hexdigest(), exhausted=end == len(raw),
        next_cursor=None if end == len(raw) else cursor(end))
    return dict(schema=client.RESPONSE_SCHEMA, session_id=query["session_id"], request_id=query["request_id"],
        ok=True, page=page, reason_code=None)


@pytest.mark.parametrize("artifact", client.ARTIFACTS)
def test_complete_exact_artifact_pages_bind_advisory_generation(artifact):
    query = request(artifact=artifact)
    value = response(query)
    assert client.validate_response(value, request=query, descriptor=descriptor()) == value
    assert json.loads(value["page"]["content"])["pending"] is True


def test_multibyte_boundary_pages_reconstruct_exact_artifact_bytes():
    raw = client.canonical_json({"candidate": "λ", "pending": True})
    split = raw.index("λ".encode()) + 1
    first = request(artifact="formal")
    first_value = response(first, raw=raw, end=split)
    client.validate_response(first_value, request=first, descriptor=descriptor())
    second = request(artifact="formal", cursor=first_value["page"]["next_cursor"])
    second_value = response(second, raw=raw)
    client.validate_response(second_value, request=second, descriptor=descriptor())
    assert first_value["page"]["encoding"] == second_value["page"]["encoding"] == "base64"
    reconstructed = b"".join(base64.b64decode(value["page"]["content"])
        for value in (first_value, second_value))
    assert reconstructed == raw
    assert hashlib.sha256(reconstructed).hexdigest() == second_value["page"]["artifact_sha256"]


@pytest.mark.parametrize("field,value", [("artifact", "source"), ("artifact", False),
    ("expected_workspace_generation", None), ("expected_workspace_generation", "f" * 63),
    ("expected_workspace_generation", "F" * 64), ("max_bytes", 2047), ("max_bytes", 8193),
    ("max_bytes", True), ("cursor", False), ("cursor", {}),
    ("cursor", dict(schema="supervisor-worker-source-view-cursor@1", request_sha256="1" * 64,
        offset=1, token="2" * 32)), ("cursor", cursor(True)), ("cursor", cursor(0)),
    ("cursor", cursor(client.MAX_ARTIFACT_BYTES + 1))])
def test_invalid_selections_refuse_before_wire(field, value):
    with pytest.raises(ValueError):
        client.validate_request(request(**{field: value}))


@pytest.mark.parametrize("target", ["envelope", "query", "cursor"])
def test_closed_requests_reject_additional_fields(target):
    query = request(cursor=cursor(1))
    container = query if target == "envelope" else query["query"] if target == "query" else query["query"]["cursor"]
    container["private_path"] = "/owner/private"
    with pytest.raises(ValueError): client.validate_request(query)


@pytest.mark.parametrize("change", ["envelope", "page", "binding", "schema", "page_schema",
    "session", "request", "ok_type", "success_reason", "artifact", "generation", "source_state",
    "baseline", "proof", "completion", "discharged", "discharged_type", "artifact_digest",
    "slice_digest", "content", "artifact_size", "oversized_artifact", "bool_size", "offset",
    "bool_offset", "end_offset", "encoding", "content_type", "exhausted", "exhausted_type",
    "cursor", "semantic", "owner_session", "bad_descriptor"])
def test_tampered_or_cross_context_pages_fail_closed(change):
    query, value, selected = request(), response(), descriptor()
    page = value["page"]
    if change == "envelope": value["extra"] = True
    elif change == "page": page["private_path"] = "/owner/private"
    elif change == "binding": page["binding"]["task_cid"] = "foreign"
    elif change == "schema": value["schema"] = "foreign"
    elif change == "page_schema": page["schema"] = "supervisor-worker-source-view-page@2"
    elif change == "session": value["session_id"] = "0" * 32
    elif change == "request": value["request_id"] = "0" * 32
    elif change == "ok_type": value["ok"] = 1
    elif change == "success_reason": value["reason_code"] = "stale_context"
    elif change == "artifact": page["artifact"] = "capsules"
    elif change == "generation": page["workspace_generation"] = "0" * 64
    elif change == "source_state": page["source_state"] = "accepted"
    elif change == "baseline": page["baseline_evidence_applicable"] = True
    elif change in {"proof", "completion"}: page[change + "_authority"] = True
    elif change == "discharged": page["source_obligations_discharged"] = 1
    elif change == "discharged_type": page["source_obligations_discharged"] = False
    elif change == "artifact_digest": page["artifact_sha256"] = "0" * 64
    elif change == "slice_digest": page["slice_sha256"] = "0" * 64
    elif change == "content": page["content"] += "x"
    elif change == "artifact_size": page["artifact_bytes"] += 1
    elif change == "oversized_artifact": page["artifact_bytes"] = client.MAX_ARTIFACT_BYTES + 1
    elif change == "bool_size": page["artifact_bytes"] = True
    elif change == "offset": page["offset"] = 1
    elif change == "bool_offset": page["offset"] = False
    elif change == "end_offset": page["end_offset"] += 1
    elif change == "encoding": page["encoding"] = "latin1"
    elif change == "content_type": page["content"] = []
    elif change == "exhausted": page["exhausted"] = False
    elif change == "exhausted_type": page["exhausted"] = 1
    elif change == "cursor": page["next_cursor"] = cursor(page["end_offset"])
    elif change == "semantic": page["binding"]["semantic_sha256"] = "0" * 64
    elif change == "owner_session": selected["session_id"] = "0" * 32
    else: selected["owner_uid"] = True
    with pytest.raises(ValueError): client.validate_response(value, request=query, descriptor=selected)


def test_binding_types_are_checked_even_without_descriptor():
    value = response()
    value["page"]["binding"]["scope_cid"] = False
    with pytest.raises(ValueError): client.validate_response(value, request=request())


def test_full_wire_budget_counts_metadata_and_content():
    query = request(max_bytes=2048)
    value = response(query, raw=b"x" * 2048)
    with pytest.raises(ValueError, match="budget"):
        client.validate_response(value, request=query, descriptor=descriptor())


def test_noncanonical_base64_pad_bits_are_refused():
    value = response(raw=b"a")
    value["page"].update(encoding="base64", content="YR==")
    with pytest.raises(ValueError, match="noncanonical"):
        client.validate_response(value, request=request(), descriptor=descriptor())


@pytest.mark.parametrize("change", ["offset", "query", "no_progress", "foreign_cursor"])
def test_continuation_must_progress_with_same_query_and_cursor_schema(change):
    raw = b"0123456789"
    query = request(cursor=cursor(2))
    value = response(query, raw=raw, end=4)
    assert client.validate_response(value, request=query, descriptor=descriptor()) == value
    if change == "offset": value["page"]["next_cursor"]["offset"] = 5
    elif change == "query": value["page"]["next_cursor"]["request_sha256"] = "0" * 64
    elif change == "no_progress": value = response(query, raw=raw, end=2)
    else: value["page"]["next_cursor"]["schema"] = "supervisor-worker-source-view-cursor@1"
    with pytest.raises(ValueError): client.validate_response(value, request=query, descriptor=descriptor())


def test_refusal_is_closed_and_contains_no_candidate_artifact():
    value = response()
    value.update(ok=False, page=None, reason_code="stale_context")
    assert client.validate_response(value, request=request(), descriptor=descriptor()) == value
    assert client.render_page_for_model(value) == {"ok": False, "reason_code": "stale_context"}
    value["reason_code"] = "private traceback"
    with pytest.raises(ValueError): client.validate_response(value, request=request())


def test_model_projection_retains_exact_bytes_identity_and_candidate_warning():
    value = response()
    client.validate_response(value, request=request(), descriptor=descriptor())
    projected = client.render_page_for_model(value)
    assert all(projected[key] == value["page"][key] for key in projected if key != "ok")
    assert projected["source_state"] == "working-candidate"
    assert projected["proof_authority"] is projected["baseline_evidence_applicable"] is False
    assert projected["artifact_sha256"] and projected["workspace_generation"]
    assert len(client.canonical_json(projected)) < len(client.canonical_json(value))
    assert "binding" not in projected and "session_id" not in projected


def test_request_uses_real_source_descriptor_and_authenticated_frame(owner_files):
    path, _, value, write, listener = owner_files
    replacement = descriptor()
    for key in ("owner_uid", "worker_uid", "owner_pid", "owner_start_time_ticks", "workspace",
                "socket_path", "socket_device", "socket_inode"):
        replacement[key] = value[key]
    value.clear(); value.update(replacement)
    digest = write()
    observed, errors = [], []
    def answer():
        try:
            channel, _ = listener.accept()
            with channel:
                query = client.source.receive_frame(channel)
                observed.append(client.validate_request(query))
                client.source.send_frame(channel, response(query))
        except BaseException as exc:
            errors.append(exc)
    thread = threading.Thread(target=answer, daemon=True)
    thread.start()
    result = client.request_page(path, digest, expected_workspace_generation=GENERATION, artifact="capsules")
    thread.join(timeout=3)
    assert not thread.is_alive() and not errors
    assert observed[0]["schema"] == client.REQUEST_SCHEMA
    assert observed[0]["query"]["artifact"] == result["page"]["artifact"] == "capsules"


def test_standalone_isolated_cli_has_no_package_requirement():
    result = subprocess.run([sys.executable, "-I", str(Path(client.__file__)), "--help"],
        capture_output=True, text=True, timeout=10)
    assert result.returncode == 0
    assert "--expected-workspace-generation" in result.stdout and "--artifact" in result.stdout


def test_cli_sanitizes_malformed_cursor_without_querying_owner(monkeypatch, capsys):
    monkeypatch.setattr(client, "request_page", lambda *a, **k: pytest.fail("ambiguous cursor reached owner"))
    code = client.main(["--source-view-descriptor", "/selected",
        "--source-view-descriptor-sha256", "a" * 64, "--expected-workspace-generation", GENERATION,
        "--cursor", '{"token":"private","token":"duplicate"}'])
    captured = capsys.readouterr()
    assert code == 2 and not captured.out
    assert json.loads(captured.err) == {"reason_code": "transport_error"}


@pytest.mark.parametrize("raw", [False, True])
def test_cli_default_projection_or_requested_raw_envelope(monkeypatch, capsys, raw):
    value = response()
    observed = []
    def read(*args, **kwargs):
        observed.append(kwargs)
        return client.validate_response(deepcopy(value), request=request(), descriptor=descriptor())
    monkeypatch.setattr(client, "request_page", read)
    argv = ["--source-view-descriptor", "/selected", "--source-view-descriptor-sha256", "a" * 64,
        "--expected-workspace-generation", GENERATION]
    if raw: argv.append("--raw-response")
    assert client.main(argv) == 0
    captured = capsys.readouterr()
    expected = value if raw else client.render_page_for_model(value)
    assert captured.out == client.canonical_json(expected).decode() + "\n" and not captured.err
    assert observed[0]["expected_workspace_generation"] == GENERATION


def test_cli_guidance_names_separate_command_and_keeps_authority_limits():
    text = client.guidance()
    assert "worker_candidate_context_client.py" in text and "--expected-workspace-generation" in text
    assert "--artifact formal" in text and "baseline evidence is inapplicable" in text
    assert "no proof or task completion" in text
