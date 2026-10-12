"""Exact candidate filters stay bound through wire validation and model output."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import threading

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import worker_candidate_context_client as client
from test.api import test_worker_candidate_context_client as legacy
from test.api.test_numerical_obligation_client import owner_files  # noqa: F401


def selection(**filters):
    return dict(path=None, symbol_id=None, function_id=None, kind=None) | filters


def request(artifact="catalog", **filters):
    value = legacy.request(artifact=artifact)
    value["schema"] = client.REQUEST_SCHEMA_V2
    value["query"]["selection"] = selection(**filters)
    return value


def response(query, **kwargs):
    value = legacy.response(query, **kwargs)
    value["page"].update(schema=client.PAGE_SCHEMA_V2,
        selection=deepcopy(query["query"]["selection"]))
    return value


@pytest.mark.parametrize("artifact,filters", [
    ("catalog", {}), ("catalog", dict(path="src/math.py")),
    ("catalog", dict(path="λ" * 2048)),
    ("capsules", dict(path="src/math.py")),
    ("capsules", dict(path="src/math.py", symbol_id="symbol:math.apply")),
    ("capsules", dict(path="src/math.py", symbol_id="λ" * 256)),
    ("obligations", dict(path="src/math.py")),
    ("obligations", dict(path="src/math.py", function_id="apply")),
    ("obligations", dict(path="src/math.py", kind="call_requirement")),
    ("obligations", dict(path="src/math.py", function_id="apply", kind="call_requirement")),
])
def test_selected_artifacts_echo_closed_filters_and_generation(artifact, filters):
    query = request(artifact, **filters)
    value = response(query)
    assert client.validate_request(query) == query
    assert client.validate_response(value, request=query, descriptor=legacy.descriptor()) == value
    projected = client.render_page_for_model(value)
    assert projected["selection"] == query["query"]["selection"]
    assert projected["workspace_generation"] == legacy.GENERATION
    assert projected["source_obligations_discharged"] == 0
    assert all(projected[key] is False for key in (
        "baseline_evidence_applicable", "proof_authority", "completion_authority"))


@pytest.mark.parametrize("artifact,filters", [
    ("summary", {}), ("formal", {}), (False, {}),
    ("catalog", dict(symbol_id="symbol:1")), ("catalog", dict(function_id="f")),
    ("catalog", dict(kind="call")), ("capsules", {}), ("obligations", {}),
    ("capsules", dict(path="x.py", function_id="f")),
    ("capsules", dict(path="x.py", kind="call")),
    ("obligations", dict(path="x.py", symbol_id="symbol:1")),
])
def test_cross_artifact_filters_are_rejected(artifact, filters):
    with pytest.raises(ValueError):
        client.validate_request(request(artifact, **filters))


@pytest.mark.parametrize("path", ["", ".", "..", "../x.py", "x/../y.py", "x/./y.py",
    "/x.py", "//x.py", "x//y.py", "x.py/", ".git/config", "src/.git/file", "x\\y.py",
    "x\x00y.py", "x\ny.py", "x\ry.py", "a" * 4097, "λ" * 2049, True, 7, [], {}])
def test_paths_are_bounded_canonical_relative_names(path):
    with pytest.raises(ValueError):
        client.validate_request(request("catalog", path=path))


@pytest.mark.parametrize("key,artifact", [("symbol_id", "capsules"),
    ("function_id", "obligations"), ("kind", "obligations")])
@pytest.mark.parametrize("value", ["", "x" * 513, "λ" * 257, "x\ny", "x\ry", "x\x00y", True, 4, [], {}])
def test_filter_values_are_nonempty_bounded_utf8(key, artifact, value):
    with pytest.raises(ValueError):
        client.validate_request(request(artifact, path="x.py", **{key: value}))


@pytest.mark.parametrize("key,artifact", [("path", "catalog"), ("symbol_id", "capsules"),
    ("function_id", "obligations"), ("kind", "obligations")])
@pytest.mark.parametrize("control", ["\t", "\x01", "\x1f", "\x7f"])
def test_selection_control_characters_match_owner_refusal(key, artifact, control):
    filters = dict(path="x.py", **{key: "x" + control + "y"}) if key != "path" else {key: "x" + control + "y.py"}
    with pytest.raises(ValueError):
        client.validate_request(request(artifact, **filters))


@pytest.mark.parametrize("change", ["missing_selection", "null_selection", "extra_selection",
    "missing_filter", "extra_query", "extra_envelope", "v1_schema", "missing_generation",
    "short_generation", "upper_generation"])
def test_selected_requests_fail_closed(change):
    query = request("capsules", path="x.py")
    if change == "missing_selection": query["query"].pop("selection")
    elif change == "null_selection": query["query"]["selection"] = None
    elif change == "extra_selection": query["query"]["selection"]["private"] = True
    elif change == "missing_filter": query["query"]["selection"].pop("kind")
    elif change == "extra_query": query["query"]["private"] = True
    elif change == "extra_envelope": query["private"] = True
    elif change == "v1_schema": query["schema"] = client.REQUEST_SCHEMA
    elif change == "missing_generation": query["query"]["expected_workspace_generation"] = None
    elif change == "short_generation": query["query"]["expected_workspace_generation"] = "f" * 63
    else: query["query"]["expected_workspace_generation"] = "F" * 64
    with pytest.raises(ValueError): client.validate_request(query)


@pytest.mark.parametrize("change", ["missing_selection", "extra_selection", "missing_filter",
    "extra_page", "extra_query", "path", "symbol", "artifact", "generation", "schema",
    "proof", "baseline", "completion", "discharged"])
def test_selection_tampering_and_query_mixups_are_rejected(change):
    query = request("capsules", path="x.py", symbol_id="symbol:x.f")
    value = response(query)
    page = value["page"]
    if change == "missing_selection": page.pop("selection")
    elif change == "extra_selection": page["selection"]["private"] = True
    elif change == "missing_filter": page["selection"].pop("kind")
    elif change == "extra_page": page["private"] = True
    elif change == "extra_query": query["query"]["private"] = True
    elif change == "path": page["selection"]["path"] = "other.py"
    elif change == "symbol": page["selection"]["symbol_id"] = "symbol:x.g"
    elif change == "artifact": page["artifact"] = "obligations"
    elif change == "generation": page["workspace_generation"] = "0" * 64
    elif change == "schema": page["schema"] = client.PAGE_SCHEMA
    elif change == "proof": page["proof_authority"] = True
    elif change == "baseline": page["baseline_evidence_applicable"] = True
    elif change == "completion": page["completion_authority"] = True
    else: page["source_obligations_discharged"] = 1
    with pytest.raises(ValueError):
        client.validate_response(value, request=query, descriptor=legacy.descriptor())


def test_selected_continuations_keep_the_owner_query_hash_and_exact_selection():
    query = request("obligations", path="x.py", function_id="f", kind="call_requirement")
    query["query"]["cursor"] = legacy.cursor(2)
    value = response(query, raw=b"0123456789", end=4)
    assert client.validate_response(value, request=query) == value
    value["page"]["next_cursor"]["request_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="query differs"):
        client.validate_response(value, request=query)


def test_v1_and_v2_pages_cannot_be_interchanged_even_for_capsules():
    old_query = legacy.request(artifact="capsules")
    new_query = request("capsules", path="x.py")
    with pytest.raises(ValueError):
        client.validate_response(response(new_query), request=old_query)
    with pytest.raises(ValueError):
        client.validate_response(legacy.response(old_query), request=new_query)
    old_value = legacy.response(old_query)
    assert client.validate_response(old_value, request=old_query) == old_value
    assert "selection" not in client.render_page_for_model(old_value)


def test_selected_read_uses_real_source_descriptor_and_authenticated_socket(owner_files):
    path, _, value, write, listener = owner_files
    replacement = legacy.descriptor()
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
    filters = selection(path="x.py", function_id="f", kind="call_requirement")
    result = client.request_page(path, digest, expected_workspace_generation=legacy.GENERATION,
        artifact="obligations", selection=filters)
    thread.join(timeout=3)
    assert not thread.is_alive() and not errors
    assert observed[0]["schema"] == client.REQUEST_SCHEMA_V2
    assert observed[0]["query"]["selection"] == result["page"]["selection"] == filters


def cli_args():
    return ["--source-view-descriptor", "/selected", "--source-view-descriptor-sha256", "a" * 64,
        "--expected-workspace-generation", legacy.GENERATION]


@pytest.mark.parametrize("extra,artifact,filters", [([], "catalog", {}),
    (["--path", "x.py"], "catalog", dict(path="x.py")),
    (["--artifact", "capsules", "--path", "x.py", "--symbol-id", "symbol:x.f"],
        "capsules", dict(path="x.py", symbol_id="symbol:x.f")),
    (["--artifact", "obligations", "--path", "x.py", "--function-id", "f", "--kind", "call_requirement"],
        "obligations", dict(path="x.py", function_id="f", kind="call_requirement"))])
def test_selected_cli_defaults_to_catalog_and_keeps_filters(monkeypatch, capsys, extra, artifact, filters):
    observed = []
    def read(*args, **kwargs):
        observed.append(kwargs)
        return response(request(artifact, **filters))
    monkeypatch.setattr(client, "request_page", read)
    assert client.main(cli_args() + ["--select"] + extra) == 0
    displayed = json.loads(capsys.readouterr().out)
    assert observed[0]["artifact"] == artifact
    assert observed[0]["selection"] == displayed["selection"] == selection(**filters)


@pytest.mark.parametrize("extra", [["--path", "x.py"], ["--symbol-id", "s"],
    ["--function-id", "f"], ["--kind", "call"], ["--artifact", "catalog"],
    ["--artifact", "obligations"], ["--select", "--artifact", "summary"],
    ["--select", "--artifact", "formal"], ["--select", "--artifact", "capsules"],
    ["--select", "--kind", "call"], ["--select", "--path", "../private.py"]])
def test_cli_refuses_ambiguous_or_invalid_selection_before_request(monkeypatch, capsys, extra):
    monkeypatch.setattr(client, "request_page", lambda *a, **k: pytest.fail("invalid selection reached owner"))
    assert client.main(cli_args() + extra) == 2
    captured = capsys.readouterr()
    assert not captured.out and json.loads(captured.err) == {"reason_code": "transport_error"}


def test_standalone_isolated_cli_exposes_selected_reads():
    value = subprocess.run([sys.executable, "-I", str(Path(client.__file__)), "--help"],
        capture_output=True, text=True, timeout=10)
    assert value.returncode == 0
    assert all(flag in value.stdout for flag in (
        "--select", "--path", "--symbol-id", "--function-id", "--kind"))


def test_guidance_prefers_catalog_then_exact_selection_and_retains_authority_limits():
    text = client.guidance()
    assert text.index("--select --artifact catalog") < text.index("--select --artifact capsules")
    assert "--select --artifact obligations" in text and "all selection filters" in text
    assert "not proof coverage" in text and "baseline evidence is inapplicable" in text
    assert "no proof or task completion" in text
