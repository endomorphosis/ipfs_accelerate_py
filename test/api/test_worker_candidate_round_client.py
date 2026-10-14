"""Candidate-only protocol, native source pages and authenticated framed reads."""
from copy import deepcopy
import base64
import json
import subprocess
import sys
import threading
import time

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import worker_candidate_round_client as client
from ipfs_accelerate_py.agent_supervisor.runtime.source_view_policy import SourceViewPolicy
from test.api.test_numerical_obligation_client import descriptor as numerical_descriptor, owner_files  # noqa: F401
from test.api.test_worker_candidate_context_client import BINDING, GENERATION, response as candidate_response
from test.api.test_worker_source_views import case, service  # noqa: F401


def descriptor(binding=None, generation=GENERATION, policy=None, scope="8" * 64):
    value = numerical_descriptor()
    for name in ("task_cid", "context_bundle_sha256", "context_receipt_sha256", "source_tree_id"):
        value.pop(name)
    return value | (BINDING if binding is None else binding) | dict(
        schema=client.DESCRIPTOR_SCHEMA, view_policy_binding_sha256=policy,
        source_scope_sha256=scope, workspace_generation=generation,
        round_context_sha256="1" * 64, prior_session_id="9" * 32,
        prior_descriptor_sha256="2" * 64, round_number=2,
        candidate_scope_cid="scope:candidate", candidate_semantic_root_cid="semantic:candidate")


def source_request(generation=GENERATION, *, policy=None, **changes):
    query = dict(path="other.py", view="original", cursor=None, max_bytes=8192,
        expected_source_sha256=None, expected_view_sha256=None,
        source_state="working-candidate", expected_workspace_generation=generation)
    if policy is not None:
        query.update(operation="read", view_policy_binding_sha256=policy)
    query.update(changes)
    return dict(schema=client.REQUEST_SCHEMA_V3 if policy else client.REQUEST_SCHEMA_V2,
        session_id="a" * 32, request_id="b" * 32, query=query)


def context_request(**changes):
    return dict(schema=client.candidate.REQUEST_SCHEMA, session_id="a" * 32, request_id="b" * 32,
        query=dict(artifact="summary", expected_workspace_generation=GENERATION,
            cursor=None, max_bytes=8192) | changes)


def wrap(page, request):
    return dict(schema=client.RESPONSE_SCHEMA, session_id=request["session_id"],
        request_id=request["request_id"], ok=True, page=page, reason_code=None)


@pytest.mark.parametrize("policy", [None, "7" * 64])
def test_distinct_descriptor_retains_baseline_namespaces_and_is_not_ordinary_source(policy):
    selected = descriptor(policy=policy)
    assert client.validate_descriptor(selected) == selected
    assert selected["scope_cid"] == BINDING["scope_cid"]
    assert selected["candidate_scope_cid"] != selected["scope_cid"]
    with pytest.raises(ValueError):
        client.source.validate_descriptor(selected)
    old = {key: value for key, value in selected.items() if key in client.source._DESCRIPTOR_FIELDS}
    old["schema"] = client.source.DESCRIPTOR_SCHEMA
    assert client.source.validate_descriptor(old) == old
    with pytest.raises(ValueError):
        client.validate_descriptor(old)


@pytest.mark.parametrize("field,value", [("round_number", True), ("round_number", 1),
    ("round_number", 3), ("prior_session_id", "a" * 32), ("prior_session_id", "z" * 32),
    ("workspace_generation", "F" * 64), ("round_context_sha256", None),
    ("prior_descriptor_sha256", "2" * 63), ("source_scope_sha256", ""),
    ("view_policy_binding_sha256", False), ("view_policy_binding_sha256", ""),
    ("candidate_scope_cid", "foreign\nidentity"), ("candidate_semantic_root_cid", False),
    ("profile", "unsafe"), ("proof_authority", True), ("execution_authority", 0),
    ("completion_authority", True)])
def test_invalid_candidate_descriptor_fields_refuse(field, value):
    selected = descriptor()
    selected[field] = value
    with pytest.raises(ValueError):
        client.validate_descriptor(selected)


@pytest.mark.parametrize("schema", [client.source.REQUEST_SCHEMA,
    "supervisor-worker-source-edit-request@1", "supervisor-worker-semantic-evidence-request@1",
    "supervisor-numerical-obligation-request@1", "unknown"])
def test_other_capabilities_cannot_enter_candidate_round(schema):
    request = source_request()
    request["schema"] = schema
    with pytest.raises(ValueError):
        client.validate_request(request)


@pytest.mark.parametrize("changes", [dict(expected_workspace_generation=None),
    dict(expected_workspace_generation="0" * 63), dict(source_state="baseline"),
    dict(path="../owner"), dict(path=".git/config"), dict(max_bytes=True),
    dict(max_bytes=8193), dict(cursor={}), dict(expected_source_sha256="1" * 64)])
def test_working_requests_require_exact_generation_and_bounded_selection(changes):
    with pytest.raises(ValueError):
        client.validate_request(source_request(**changes))


def test_discovery_and_baseline_shared_requests_are_explicitly_refused():
    discovery = dict(schema=client.REQUEST_SCHEMA_V3, session_id="a" * 32, request_id="b" * 32,
        query=dict(operation="discover", view_policy_binding_sha256="7" * 64, cursor=None, max_bytes=8192))
    baseline = source_request(policy="7" * 64, source_state="baseline", expected_workspace_generation=None)
    for request in (discovery, baseline):
        assert client.source.validate_request(request) == request
        with pytest.raises(ValueError):
            client.validate_request(request)


@pytest.mark.parametrize("shared", [False, True])
def test_actual_working_source_pages_bind_baseline_and_fixed_candidate(case, shared):
    policy = SourceViewPolicy.from_mapping(profile="python-comments@1") if shared else None
    svc = service(case, **({"view_policy": policy} if policy else {}))
    (case["workspace"] / "other.py").write_text("VALUE = 3\n")
    page = svc.working_page(path="other.py", view="original", max_bytes=7600)
    generation = page["workspace_generation"]
    policy_hash = page.get("view_policy_binding_sha256")
    request = source_request(generation, policy=policy_hash)
    selected = descriptor(page["binding"], generation, policy_hash, svc.baseline_admission()["scope_sha256"])
    value = wrap(page, request)
    assert client.validate_response(value, request=request, descriptor=selected) == value
    assert value["page"]["content"] == "VALUE = 3\n"
    assert value["page"]["baseline_evidence_applicable"] is False
    assert (case["root"] / "other.py").read_text() == "VALUE = 2\n"
    for field in client.BINDING_FIELDS:
        wrong = deepcopy(selected)
        wrong[field] = "identity:other" if field in ("task_id", "scope_cid", "semantic_root_cid") else (
            "identity@1" if field == "profile" else "0" * 64)
        with pytest.raises(ValueError):
            client.validate_response(value, request=request, descriptor=wrong)
    wrong = deepcopy(selected)
    wrong["workspace_generation"] = "0" * 64
    with pytest.raises(ValueError):
        client.validate_response(value, request=request, descriptor=wrong)
    wrong = deepcopy(selected)
    wrong["view_policy_binding_sha256"] = "6" * 64 if policy_hash is None else None
    with pytest.raises(ValueError):
        client.validate_response(value, request=request, descriptor=wrong)


@pytest.mark.parametrize("artifact", client.candidate.ARTIFACTS)
def test_candidate_artifacts_keep_advisory_generation_and_integrity(artifact):
    request = context_request(artifact=artifact)
    value = candidate_response(request)
    assert client.validate_response(value, request=request, descriptor=descriptor()) == value
    projected = client.render_page_for_model(value)
    assert projected["baseline_evidence_applicable"] is False
    assert projected["proof_authority"] is False
    for changed in ("generation", "binding", "content", "proof", "baseline"):
        wrong = deepcopy(value)
        if changed == "generation": wrong["page"]["workspace_generation"] = "0" * 64
        elif changed == "binding": wrong["page"]["binding"]["scope_cid"] = "scope:foreign"
        elif changed == "content": wrong["page"]["content"] += "x"
        elif changed == "proof": wrong["page"]["proof_authority"] = True
        else: wrong["page"]["baseline_evidence_applicable"] = True
        with pytest.raises(ValueError):
            client.validate_response(wrong, request=request, descriptor=descriptor())


def test_selected_candidate_pages_are_validated_without_baseline_descriptor_relabeling():
    request = context_request(artifact="catalog")
    request["schema"] = client.candidate.REQUEST_SCHEMA_V2
    request["query"]["selection"] = dict(path=None, symbol_id=None, function_id=None, kind=None)
    value = candidate_response(request)
    value["page"].update(schema=client.candidate.PAGE_SCHEMA_V2, selection=request["query"]["selection"])
    assert client.validate_response(value, request=request, descriptor=descriptor()) == value
    wrong = deepcopy(value)
    wrong["page"]["selection"]["path"] = "other.py"
    with pytest.raises(ValueError):
        client.validate_response(wrong, request=request, descriptor=descriptor())


def test_complete_candidate_multibyte_pages_reconstruct_exact_bytes():
    raw = client.canonical_json({"candidate": "λ", "pending": True})
    split = raw.index("λ".encode()) + 1
    first = context_request(artifact="formal")
    first_value = candidate_response(first, raw=raw, end=split)
    second = context_request(artifact="formal", cursor=first_value["page"]["next_cursor"])
    second_value = candidate_response(second, raw=raw)
    for request, value in ((first, first_value), (second, second_value)):
        client.validate_response(value, request=request, descriptor=descriptor())
    assert b"".join(base64.b64decode(value["page"]["content"])
        for value in (first_value, second_value)) == raw


def _replace_owner_descriptor(owner_files):
    path, _, value, write, listener = owner_files
    selected = descriptor()
    for key in ("owner_uid", "worker_uid", "owner_pid", "owner_start_time_ticks", "workspace",
            "socket_path", "socket_device", "socket_inode"):
        selected[key] = value[key]
    value.clear(); value.update(selected)
    return path, write(), value, write, listener


def test_real_descriptor_socket_peer_and_candidate_frames(owner_files):
    path, digest, selected, _, listener = _replace_owner_descriptor(owner_files)
    observed, errors = [], []
    def answer():
        try:
            channel, _ = listener.accept()
            with channel:
                request = client.validate_request(client.receive_frame(channel))
                observed.append(request)
                client.send_frame(channel, candidate_response(request))
        except BaseException as error:
            errors.append(error)
    thread = threading.Thread(target=answer, daemon=True)
    thread.start()
    result = client.request_candidate_page(path, digest, expected_workspace_generation=GENERATION,
        artifact="formal")
    thread.join(timeout=3)
    assert not thread.is_alive() and not errors
    assert len(observed) == 1 and result["page"]["artifact"] == "formal"
    assert client.load_descriptor(path, digest) == selected
    with pytest.raises(ValueError):
        client.source.load_descriptor(path, digest)


@pytest.mark.parametrize("change", ["digest", "expired", "future", "workspace", "mode",
    "socket_inode", "owner_birth", "authority"])
def test_real_owner_descriptor_refuses_lifecycle_or_integrity_changes(owner_files, change):
    path, digest, value, write, _ = _replace_owner_descriptor(owner_files)
    if change == "digest": digest = "0" * 64
    elif change == "expired":
        value["issued_monotonic_ns"] = time.monotonic_ns() - 60_000_000_000
        value["expires_monotonic_ns"] = time.monotonic_ns() - 1
        digest = write()
    elif change == "future":
        value["issued_monotonic_ns"] = time.monotonic_ns() + 10_000_000_000
        value["expires_monotonic_ns"] = value["issued_monotonic_ns"] + 30_000_000_000
        digest = write()
    elif change == "mode": path.chmod(0o660)
    else:
        if change == "workspace": value["workspace"] = str(path.parent)
        elif change == "socket_inode": value["socket_inode"] += 1
        elif change == "owner_birth": value["owner_start_time_ticks"] += 1
        else: value["execution_authority"] = True
        digest = write()
    with pytest.raises(ValueError):
        client.load_descriptor(path, digest)


def test_wrong_generation_is_rejected_before_opening_socket(monkeypatch):
    monkeypatch.setattr(client, "load_descriptor", lambda *args, **kwargs: descriptor())
    monkeypatch.setattr(client.socket, "socket", lambda *args: pytest.fail("unbound generation reached socket"))
    for request in (lambda: client.request_page("/owner/descriptor", "0" * 64, path="other.py",
                      expected_workspace_generation="0" * 64),
                    lambda: client.request_candidate_page("/owner/descriptor", "0" * 64,
                      expected_workspace_generation="0" * 64)):
        with pytest.raises(ValueError): request()


def test_standalone_isolated_cli_imports_fixed_siblings_without_package_or_credentials():
    result = subprocess.run([sys.executable, "-I", client.__file__, "--help"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--candidate-round-descriptor" in result.stdout and "--artifact" in result.stdout
    # The package-independent candidate branch must also resolve the sibling
    # validator without invoking source client's package-relative import.
    code = """import importlib.util,sys
spec=importlib.util.spec_from_file_location('isolated_candidate_round',sys.argv[1])
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
request=module.strict_json_loads(sys.argv[2]);response=module.strict_json_loads(sys.argv[3])
module.validate_response(response,request=request)
assert not any(name.startswith('ipfs_accelerate_py') for name in sys.modules)
"""
    request = context_request()
    result = subprocess.run([sys.executable, "-I", "-c", code, client.__file__,
        client.canonical_json(request).decode(), client.canonical_json(candidate_response(request)).decode()],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("args", [["--path", "other.py"], ["--artifact", "summary"],
    ["--select", "--artifact", "catalog"], ["--select", "--artifact", "capsules", "--path", "other.py"]])
def test_cli_uses_separate_candidate_environment_and_generation(monkeypatch, capsys, args):
    monkeypatch.setenv(client.DESCRIPTOR_ENV, "/owner/candidate.json")
    monkeypatch.setenv(client.DIGEST_ENV, "0" * 64)
    observed = []
    def read(descriptor_path, digest, **kwargs):
        observed.append((descriptor_path, digest, kwargs))
        return dict(ok=False, page=None, reason_code="stale_context")
    monkeypatch.setattr(client, "request_page", read)
    monkeypatch.setattr(client, "request_context_page", read)
    assert client.main(["--expected-workspace-generation", GENERATION, *args]) == 2
    assert len(observed) == 1
    assert observed[0][0:2] == ("/owner/candidate.json", "0" * 64)
    assert observed[0][2]["expected_workspace_generation"] == GENERATION
    assert json.loads(capsys.readouterr().out) == dict(ok=False, reason_code="stale_context")
