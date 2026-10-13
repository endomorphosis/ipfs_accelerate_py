"""Working-candidate client contracts against real bounded source snapshots."""
import base64
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import worker_source_client as client
from test.api.test_worker_source_client import descriptor, request as baseline_request, response
from test.api.test_worker_source_views import case, service  # noqa: F401


def request(**changed):
    query = baseline_request(**changed)
    query["schema"] = client.REQUEST_SCHEMA_V2
    query["query"] = dict(source_state="working-candidate", expected_workspace_generation=None) | query["query"]
    return query


def read(svc, query):
    args = dict(query["query"])
    assert args.pop("source_state") == "working-candidate"
    args["max_bytes"] -= len(client.canonical_json(response({}, query))) - 2
    page = svc.working_page(**args)
    value = response(page, query)
    return client.validate_response(value, request=query, descriptor=descriptor(page["binding"]))


@pytest.fixture
def sample(case):
    svc = service(case)
    (case["workspace"] / "other.py").write_bytes(b"VALUE = 12   # candidate change\n")
    query = request(view="compact")
    value = read(svc, query)
    return svc, query, value, descriptor(value["page"]["binding"])


def test_working_compact_and_pinned_original_round_trip_after_edit(case):
    svc = service(case)
    name = "pkg/sample.py"
    edited = case["sources"][name].replace(b"VALUE_0 = 0", b"VALUE_0 = 100")
    (case["workspace"] / name).write_bytes(edited)
    first = read(svc, request(path=name, view="compact", max_bytes=4096))["page"]
    assert first["source_sha256"] == hashlib.sha256(edited).hexdigest()
    assert first["baseline_source_sha256"] == hashlib.sha256(case["sources"][name]).hexdigest()
    cursor, joined = None, bytearray()
    for _ in range(32):
        query = request(path=name, view="original", max_bytes=4096, cursor=cursor,
            expected_workspace_generation=first["workspace_generation"],
            expected_source_sha256=first["source_sha256"], expected_view_sha256=first["view_sha256"])
        page = read(svc, query)["page"]
        joined.extend(page["content"].encode() if page["encoding"] == "utf-8"
                      else base64.b64decode(page["content"], validate=True))
        assert page["baseline_evidence_applicable"] is False
        cursor = page["next_cursor"]
        if cursor is None:
            break
    else:
        pytest.fail("candidate continuation did not finish")
    assert joined == edited
    assert (case["root"] / name).read_bytes() == case["sources"][name]


@pytest.mark.parametrize("path,view", [("empty.py", "original"), ("binary.dat", "compact"),
                                      ("other.py", "compact")])
def test_empty_binary_and_unmodified_candidate_remain_explicitly_unproven(case, path, view):
    value = read(service(case), request(path=path, view=view))
    assert value["page"]["baseline_evidence_applicable"] is False
    assert value["page"]["source_sha256"] == value["page"]["baseline_source_sha256"]


def test_generation_only_pin_accepts_current_snapshot_and_refuses_later_edit(case, sample):
    svc, _, value, _ = sample
    generation = value["page"]["workspace_generation"]
    pinned = request(expected_workspace_generation=generation)
    assert read(svc, pinned)["page"]["workspace_generation"] == generation
    (case["workspace"] / "other.py").write_bytes(b"VALUE = 13\n")
    with pytest.raises(ValueError):
        read(svc, pinned)
    fresh = read(svc, request())["page"]
    assert fresh["workspace_generation"] != generation


@pytest.mark.parametrize("field,bad", [("source_state", "baseline"), ("source_state", None),
    ("source_state", True), ("expected_workspace_generation", ""),
    ("expected_workspace_generation", "a" * 63), ("expected_workspace_generation", "A" * 64),
    ("expected_workspace_generation", True)])
def test_invalid_candidate_request_fields_refuse(field, bad):
    with pytest.raises(ValueError):
        client.validate_request(request(**{field: bad}))


@pytest.mark.parametrize("change", ["missing_state", "missing_generation", "extra", "v1_candidate",
    "v2_baseline", "pins_without_generation", "cursor_without_generation", "half_pins"])
def test_closed_request_versions_and_generation_requirements(change):
    query = request()
    if change == "missing_state": query["query"].pop("source_state")
    elif change == "missing_generation": query["query"].pop("expected_workspace_generation")
    elif change == "extra": query["query"]["refresh_proofs"] = True
    elif change == "v1_candidate": query["schema"] = client.REQUEST_SCHEMA
    elif change == "v2_baseline": query["query"] = baseline_request()["query"]
    elif change == "pins_without_generation":
        query["query"].update(expected_source_sha256="a" * 64, expected_view_sha256="b" * 64)
    elif change == "cursor_without_generation":
        query["query"]["cursor"] = dict(schema="supervisor-worker-source-view-cursor@1",
            request_sha256="a" * 64, offset=5, token="b" * 32)
    else: query["query"].update(expected_source_sha256="a" * 64, expected_workspace_generation="b" * 64)
    with pytest.raises(ValueError):
        client.validate_request(query)


@pytest.mark.parametrize("changed", [{}, {"expected_workspace_generation": "a" * 64},
    {"expected_workspace_generation": "a" * 64, "expected_source_sha256": "b" * 64,
     "expected_view_sha256": "c" * 64},
    {"expected_workspace_generation": "a" * 64, "cursor": {
        "schema": "supervisor-worker-source-view-cursor@1", "request_sha256": "b" * 64,
        "offset": 5, "token": "c" * 32}}])
def test_first_page_generation_only_expansion_and_continuation_requests_are_supported(changed):
    value = request(**changed)
    assert client.validate_request(value) == value


@pytest.mark.parametrize("field,bad", [("source_state", "baseline"),
    ("workspace_generation", True), ("workspace_generation", "a" * 63),
    ("baseline_source_sha256", None), ("baseline_source_sha256", "A" * 64),
    ("baseline_evidence_applicable", True), ("baseline_evidence_applicable", 0),
    ("source_check_profile", "owner-admitted-exact-source@1"),
    ("canonical_source_current", False), ("workspace_source_current", False),
    ("proof_authority", True), ("completion_authority", True), ("execution_authority", True),
    ("mutation_authority", True), ("omission_authority", True),
    ("semantic_equivalence_claimed", True), ("source_obligations_discharged", 1),
    ("schema", "supervisor-worker-source-view-page@1")])
def test_candidate_identity_authority_and_freshness_are_strict(sample, field, bad):
    _, query, value, selected = sample
    value = deepcopy(value)
    value["page"][field] = bad
    with pytest.raises(ValueError):
        client.validate_response(value, request=query, descriptor=selected)


@pytest.mark.parametrize("field", ["source_state", "workspace_generation", "baseline_source_sha256",
                                  "baseline_evidence_applicable"])
def test_candidate_page_requires_all_extra_fields(sample, field):
    _, query, value, selected = sample
    value = deepcopy(value)
    value["page"].pop(field)
    with pytest.raises(ValueError):
        client.validate_response(value, request=query, descriptor=selected)


def test_candidate_extra_fields_and_cross_version_responses_refuse(sample):
    _, query, value, selected = sample
    extra = deepcopy(value)
    extra["page"]["new_proof"] = True
    with pytest.raises(ValueError): client.validate_response(extra, request=query, descriptor=selected)
    old = baseline_request(view="compact")
    with pytest.raises(ValueError): client.validate_response(value, request=old, descriptor=selected)
    baseline = deepcopy(value)
    for key in ("source_state", "workspace_generation", "baseline_source_sha256", "baseline_evidence_applicable"):
        baseline["page"].pop(key)
    baseline["page"].update(schema="supervisor-worker-source-view-page@1",
                            source_check_profile="owner-admitted-exact-source@1")
    with pytest.raises(ValueError): client.validate_response(baseline, request=query, descriptor=selected)


def test_candidate_response_must_match_requested_generation_and_content_pins(sample):
    _, _, value, selected = sample
    page = value["page"]
    valid = request(view="compact", expected_workspace_generation=page["workspace_generation"],
        expected_source_sha256=page["source_sha256"], expected_view_sha256=page["view_sha256"])
    assert client.validate_response(value, request=valid, descriptor=selected) == value
    for field in ("expected_workspace_generation", "expected_source_sha256", "expected_view_sha256"):
        altered = deepcopy(valid)
        altered["query"][field] = "0" * 64
        with pytest.raises(ValueError): client.validate_response(value, request=altered, descriptor=selected)


def test_default_candidate_projection_preserves_status_and_all_read_pins(sample):
    _, _, value, _ = sample
    projected = client.render_page_for_model(value)
    expected = {"ok", "path", "view", "source_sha256", "view_sha256", "offset", "end_offset",
        "exhausted", "next_cursor", "encoding", "content", "source_state", "workspace_generation",
        "baseline_source_sha256", "baseline_evidence_applicable"}
    assert set(projected) == expected and projected["ok"] is True
    assert all(projected[key] == value["page"][key] for key in expected - {"ok"})
    assert projected["baseline_evidence_applicable"] is False
    assert len(client.canonical_json(projected)) < len(client.canonical_json(value))


@pytest.mark.parametrize("working", [False, True])
def test_request_page_emits_exact_selected_version_and_validates_real_core_reply(case, tmp_path, monkeypatch, working):
    svc = service(case)
    selected = descriptor(svc.page(path="other.py")["binding"])
    observed = []
    class Channel:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def settimeout(self, seconds): assert 0 < seconds <= 30
        def connect(self, path): assert path == selected["socket_path"]
    monkeypatch.setattr(client, "load_descriptor", lambda *args, **kwargs: selected)
    monkeypatch.setattr(client.socket, "socket", lambda *args: Channel())
    monkeypatch.setattr(client.transport, "_verify_peer", lambda *args: None)
    monkeypatch.setattr(client.transport, "_socket_identity", lambda *args: None)
    monkeypatch.setattr(client, "send_frame", lambda channel, query: observed.append(query))
    def receive(channel, **kwargs):
        query = observed[-1]
        if working: return read(svc, query)
        return response(svc.page(**query["query"]), query)
    monkeypatch.setattr(client, "receive_frame", receive)
    result = client.request_page(tmp_path / "descriptor.json", "a" * 64, path="other.py", working=working)
    assert result["ok"] is True and len(observed) == 1
    expected = request(path="other.py", view="compact") if working else baseline_request(path="other.py", view="compact")
    assert observed[0]["schema"] == expected["schema"] and observed[0]["query"] == expected["query"]


@pytest.mark.parametrize("changed", [{"working": False, "expected_workspace_generation": "a" * 64},
                                    {"working": 1}, {"working": "working"}])
def test_api_refuses_generation_without_working_or_nonboolean_selection_before_transport(monkeypatch, changed):
    monkeypatch.setattr(client, "load_descriptor", lambda *args, **kwargs: pytest.fail("invalid selection reached owner"))
    with pytest.raises(ValueError): client.request_page("/unused", "a" * 64, path="other.py", **changed)


@pytest.mark.parametrize("raw", [False, True])
def test_cli_forwards_working_generation_and_preserves_candidate_labels(sample, monkeypatch, capsys, raw):
    _, _, value, _ = sample
    page = value["page"]
    observed = []
    def fetch(*args, **kwargs):
        observed.append(kwargs)
        return value
    monkeypatch.setattr(client, "request_page", fetch)
    monkeypatch.setattr(client, "load_descriptor", lambda *args, **kwargs:
        pytest.fail("ordinary CLI reads must not add an edit descriptor preload"))
    argv = ["--source-view-descriptor", "/selected", "--source-view-descriptor-sha256", "a" * 64,
        "--path", "other.py", "--working", "--expected-workspace-generation", page["workspace_generation"],
        "--expected-source-sha256", page["source_sha256"], "--expected-view-sha256", page["view_sha256"]]
    if raw: argv.append("--raw-response")
    assert client.main(argv) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out) == (value if raw else client.render_page_for_model(value))
    assert not captured.err and observed[0]["working"] is True
    assert observed[0]["expected_workspace_generation"] == page["workspace_generation"]


def test_cli_edit_ready_adds_only_manifest_pin_for_enabled_working_session(sample, monkeypatch, capsys):
    _, _, value, selected = sample
    page = value["page"]
    selected = client.validate_descriptor(selected | {
        "schema": client.DESCRIPTOR_SCHEMA_V3, "source_edit_mode": "mapped-span@1",
        "source_scope_sha256": page["scope_sha256"], "view_policy_binding_sha256": None})
    loaded, fetched = [], []
    def load(*args):
        loaded.append(args)
        return selected
    def fetch(*args, **kwargs):
        fetched.append((args, kwargs))
        return value
    monkeypatch.setattr(client, "load_descriptor", load)
    monkeypatch.setattr(client, "request_page", fetch)
    argv = ["--source-view-descriptor", "/selected", "--source-view-descriptor-sha256", "a" * 64,
        "--path", "other.py", "--view", "compact", "--working", "--edit-ready"]
    assert client.main(argv) == 0
    captured = capsys.readouterr()
    expected = client.render_page_for_model(value) | {"manifest_sha256": page["manifest_sha256"]}
    assert captured.out == client.canonical_json(expected).decode("utf-8") + "\n"
    assert not captured.err and loaded == [("/selected", "a" * 64)]
    assert len(fetched) == 1 and fetched[0][0] == ("/selected", "a" * 64)
    assert fetched[0][1]["working"] is True and fetched[0][1]["view"] == "compact"
    assert "edit_ready" not in fetched[0][1]


@pytest.mark.parametrize("selection", ["disabled_descriptor", "discovery", "nonworking"])
def test_cli_edit_ready_refuses_unsupported_selection_before_request(sample, monkeypatch, capsys, selection):
    _, _, _, selected = sample
    client.validate_descriptor(selected)
    loaded = []
    def load(*args):
        assert selection == "disabled_descriptor", "invalid CLI selection reached descriptor loading"
        loaded.append(args)
        return selected
    def refused(*args, **kwargs):
        pytest.fail("invalid edit-ready selection reached a source request")
    monkeypatch.setattr(client, "load_descriptor", load)
    monkeypatch.setattr(client, "request_page", refused)
    monkeypatch.setattr(client, "request_discovery", refused)
    argv = ["--source-view-descriptor", "/selected", "--source-view-descriptor-sha256", "a" * 64,
        "--edit-ready"]
    argv += ["--discover", "--working"] if selection == "discovery" else ["--path", "other.py"]
    if selection == "disabled_descriptor":
        argv.append("--working")
    assert client.main(argv) == 2
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == '{"reason_code":"transport_error"}\n'
    assert loaded == ([("/selected", "a" * 64)] if selection == "disabled_descriptor" else [])


def test_standalone_help_and_guidance_document_candidate_evidence_boundary():
    result = subprocess.run([sys.executable, "-I", str(Path(client.__file__)), "--help"],
        capture_output=True, text=True, timeout=10)
    assert result.returncode == 0 and "--working" in result.stdout
    assert "--expected-workspace-generation" in result.stdout and "--edit-ready" in result.stdout
    guidance = client.guidance("python-comments@1")
    assert "Edit original workspace files" in guidance
    assert "Working snapshots do not refresh or prove the accepted semantic index" in guidance
    assert "worker_candidate_context_client.py" in guidance
    assert "runs no solver or Lake build" in guidance
    assert "--working" in guidance and "--expected-workspace-generation" in guidance
