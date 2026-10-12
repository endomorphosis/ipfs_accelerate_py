"""Source client contracts against actual core pages; no providers or owner SQL."""
import base64
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import worker_source_client as client
from test.api.test_numerical_obligation_client import descriptor as numerical_descriptor, owner_files  # noqa: F401
from test.api.test_worker_source_views import case, service  # noqa: F401


def descriptor(binding):
    value = numerical_descriptor()
    for key in ("task_cid", "context_bundle_sha256", "context_receipt_sha256", "source_tree_id"):
        value.pop(key)
    return {**value, **binding, "schema": client.DESCRIPTOR_SCHEMA}


def request(**changed):
    return dict(schema=client.REQUEST_SCHEMA, session_id="a" * 32, request_id="b" * 32,
        query=dict(path="other.py", view="original", cursor=None, max_bytes=8192,
            expected_source_sha256=None, expected_view_sha256=None) | changed)


def response(page, query):
    return dict(schema=client.RESPONSE_SCHEMA, session_id=query["session_id"],
        request_id=query["request_id"], ok=True, page=page, reason_code=None)


@pytest.fixture
def sample(case):
    svc = service(case)
    query = request()
    page = svc.page(**query["query"])
    return svc, query, response(page, query), descriptor(page["binding"])


def test_real_core_pages_match_closed_client_contract_and_continue_exactly(case):
    svc = service(case)
    cursor, raw = None, bytearray()
    for _ in range(32):
        query = request(path="pkg/sample.py", max_bytes=4096, cursor=cursor)
        overhead = len(client.canonical_json(response({}, query))) - 2
        page = svc.page(**(query["query"] | {"max_bytes": 4096 - overhead}))
        value = response(page, query)
        assert client.validate_response(value, request=query, descriptor=descriptor(page["binding"])) == value
        raw.extend(page["content"].encode() if page["encoding"] == "utf-8" else base64.b64decode(page["content"]))
        cursor = page["next_cursor"]
        if cursor is None: break
    else: pytest.fail("source continuation did not finish")
    assert raw == case["sources"]["pkg/sample.py"]


@pytest.mark.parametrize("path,view", [("empty.py", "original"), ("binary.dat", "compact"), ("other.py", "compact")])
def test_real_empty_binary_and_compact_pages(case, path, view):
    query = request(path=path, view=view)
    page = service(case).page(**(query["query"] | {"max_bytes": 7800}))
    value = response(page, query)
    assert client.validate_response(value, request=query, descriptor=descriptor(page["binding"])) == value


@pytest.mark.parametrize("field,bad", [("path", "../secret"), ("path", "/secret"), ("path", "a/.git/config"),
    ("view", "raw"), ("max_bytes", True), ("max_bytes", 2047), ("max_bytes", 8193),
    ("cursor", {}), ("expected_source_sha256", "a" * 64)])
def test_invalid_query_and_partial_pins_refuse(field, bad):
    with pytest.raises(ValueError): client.validate_request(request(**{field: bad}))


@pytest.mark.parametrize("change", ["extra_envelope", "extra_page", "extra_binding", "schema", "path", "view",
    "bool_size", "bool_offset", "content", "slice_sha", "full_sha", "representation_size", "offset",
    "exhausted", "cursor", "proof", "mutation", "omission", "discharged", "current", "read_only",
    "semantic", "source_scope", "native_prompt", "manifest", "profile", "owner_session", "bad_descriptor"])
def test_cross_context_or_malformed_responses_fail_closed(sample, change):
    _, query, value, selected = sample
    value, selected = deepcopy(value), deepcopy(selected)
    page = value["page"]
    if change == "extra_envelope": value["extra"] = True
    elif change == "extra_page": page["private_path"] = "/owner/private"
    elif change == "extra_binding": page["binding"]["task_cid"] = "not-a-native-task"
    elif change == "schema": page["schema"] = "wrong"
    elif change == "path": page["path"] = "private.db"
    elif change == "view": page["view"] = "compact"
    elif change == "bool_size": page["source_bytes"] = True
    elif change == "bool_offset": page["offset"] = False
    elif change == "content": page["content"] += "x"
    elif change == "slice_sha": page["slice_sha256"] = "0" * 64
    elif change == "full_sha": page["source_sha256"] = page["representation_sha256"] = "0" * 64
    elif change == "representation_size": page["representation_bytes"] += 1
    elif change == "offset": page["offset"] = 1
    elif change == "exhausted": page["exhausted"] = False
    elif change == "cursor": page["next_cursor"] = {}
    elif change in {"proof", "mutation", "omission"}: page[change + "_authority"] = True
    elif change == "discharged": page["source_obligations_discharged"] = False
    elif change == "current": page["workspace_source_current"] = False
    elif change == "read_only": page["read_only"] = False
    elif change == "semantic": page["binding"]["semantic_root_cid"] = "foreign:semantic"
    elif change == "source_scope": page["binding"]["scope_cid"] = "foreign:scope"
    elif change == "native_prompt": page["binding"]["native_prompt_sha256"] = "0" * 64
    elif change == "manifest": page["binding"]["source_manifest_sha256"] = "0" * 64
    elif change == "profile": page["binding"]["profile"] = page["requested_profile"] = "identity@1"
    elif change == "owner_session": selected["session_id"] = "f" * 32
    else: selected["worker_uid"] = True
    with pytest.raises(ValueError): client.validate_response(value, request=query, descriptor=selected)


def test_noncanonical_base64_pad_bits_refuse(sample):
    _, query, value, selected = sample
    page = value["page"]
    # YQ== and YR== both decode to b'a'; only the first is canonical.
    digest = hashlib.sha256(b"a").hexdigest()
    page.update(encoding="base64", content="YR==", slice_sha256=digest,
        source_sha256=digest, representation_sha256=digest, source_bytes=1,
        representation_bytes=1, offset=0, end_offset=1)
    with pytest.raises(ValueError, match="noncanonical"):
        client.validate_response(value, request=query, descriptor=selected)


def test_unknown_receipt_is_not_a_zero_byte_success(sample):
    _, query, value, selected = sample
    value.update(ok=False, page=None, reason_code="stale_context")
    assert client.validate_response(value, request=query, descriptor=selected) == value
    value["reason_code"] = "owner private exception body"
    with pytest.raises(ValueError): client.validate_response(value, request=query, descriptor=selected)


def test_shared_loader_is_explicit_source_validator_and_numeric_default_stays_closed(owner_files):
    path, _, numeric, write, _ = owner_files
    source = descriptor(dict(task_id="source-task", scope_cid="scope:1", semantic_root_cid="semantic:1",
        semantic_sha256="a" * 64, native_prompt_sha256="b" * 64,
        source_manifest_sha256="c" * 64, profile="python-whitespace@1"))
    # Keep this real file/socket/owner/workspace boundary from the shared fixture.
    for key in ("owner_uid", "worker_uid", "owner_pid", "owner_start_time_ticks", "workspace",
                "socket_path", "socket_device", "socket_inode"):
        source[key] = numeric[key]
    numeric.clear(); numeric.update(source)
    digest = write()
    assert client.load_descriptor(path, digest) == source
    with pytest.raises(ValueError): client.transport.load_descriptor(path, digest)


def test_cli_standalone_isolated_help_needs_no_package_imports():
    result = subprocess.run([sys.executable, "-I", str(Path(client.__file__)), "--help"],
        capture_output=True, text=True, timeout=10)
    assert result.returncode == 0 and "--expected-source-sha256" in result.stdout


def test_cli_sanitizes_invalid_cursor_without_invoking_reader(monkeypatch, capsys):
    monkeypatch.setattr(client, "request_page", lambda *a, **k: pytest.fail("invalid cursor reached socket"))
    code = client.main(["--source-view-descriptor", "/selected", "--source-view-descriptor-sha256", "a" * 64,
        "--path", "other.py", "--cursor", '{"token":"secret","token":"secret2"}'])
    output = capsys.readouterr()
    assert code == 2 and not output.out
    assert json.loads(output.err) == {"reason_code": "transport_error"}


def test_default_tool_projection_preserves_exact_content_pins_and_continuation(case):
    query = request(path="pkg/sample.py", max_bytes=4096)
    page = service(case).page(**(query["query"] | {"max_bytes": 3800}))
    value = response(page, query)
    client.validate_response(value, request=query, descriptor=descriptor(page["binding"]))
    projected = client.render_page_for_model(value)
    assert set(projected) == {"ok", "path", "view", "source_sha256", "view_sha256", "offset", "end_offset",
        "exhausted", "next_cursor", "encoding", "content"}
    assert projected["ok"] is True and projected["next_cursor"] is not None
    assert all(projected[key] == page[key] for key in set(projected) - {"ok"})
    assert len(client.canonical_json(projected)) + 1 < len(client.canonical_json(value)) + 1
    assert "binding" not in projected and "descriptor" not in projected


@pytest.mark.parametrize("raw", [False, True])
def test_cli_defaults_to_compact_projection_after_validated_read(sample, monkeypatch, capsys, raw):
    _, query, value, selected = sample
    observed = []
    def read(*args, **kwargs):
        observed.append(kwargs)
        return client.validate_response(value, request=query, descriptor=selected)
    monkeypatch.setattr(client, "request_page", read)
    argv = ["--source-view-descriptor", "/selected", "--source-view-descriptor-sha256", "a" * 64,
            "--path", "other.py", "--view", "original"]
    if raw: argv.append("--raw-response")
    assert client.main(argv) == 0
    captured = capsys.readouterr()
    expected = value if raw else client.render_page_for_model(value)
    assert captured.out == client.canonical_json(expected).decode() + "\n" and not captured.err
    assert len(observed) == 1


def test_compact_refusal_contains_no_source_or_success_claim():
    value = dict(schema=client.RESPONSE_SCHEMA, session_id="a" * 32, request_id="b" * 32,
        ok=False, page=None, reason_code="stale_context")
    assert client.render_page_for_model(value) == {"ok": False, "reason_code": "stale_context"}
