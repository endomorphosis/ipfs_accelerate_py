"""Prepared source-read exposure, without source text or delivery claims.

The real child uses native repository custody. There is no remote provider,
model-attention, displaced-context, proof or whole-task savings observation.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_mcp as mcp
from test.api.test_repository_source_mcp import configuration, real_client


def payload(scope="full_file", *, representation="original", content=None):
    value = dict(generation=1, path="private-module.py", source_sha256="1" * 64,
        representation=representation, encoding="utf-8", proof_authority=False,
        content="private source text\n" if content is None else content)
    if scope == "definition":
        value.update(scope="python_definition", symbol_id="private-symbol-id",
                     symbol="private_local_name", edit_supported=True)
    elif scope == "line_range":
        value.update(scope="line_range", start_line=2, end_line=3, total_lines=7, omitted_lines=5,
                     content="private line two\nprivate line three\n" if content is None else content)
    return value


def arguments(value):
    result = {key: value[key] for key in ("generation", "path", "representation")}
    result.update({key: value[key] for key in ("symbol_id", "start_line", "end_line") if key in value})
    return result


def journal(tmp_path):
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    path = private / "operations.jsonl"
    audit = mcp._Audit(path)
    audit.record("initialize")
    return path, audit


def record(audit, value=None, *, success=True, legacy=False, request=None, compact=None):
    value = payload() if value is None else value
    request = arguments(value) if request is None else request
    compact = value.get("representation") == "compact" if compact is None else compact
    audit.record("tool", tool="repository_read", success=success, returned_bytes=len(mcp._json(value)),
        compact=compact, read=None if legacy else mcp._read_event(request, value, success))


def validate(summary, value=None, **changes):
    inputs = {key: summary[key] for key in ("tool_calls", "errors", "successful_reads", "compact_reads")}
    return mcp.validate_read_observation(summary["read_observation"] if value is None else value,
                                         **{**inputs, **changes})


def test_disjoint_read_scopes_exposure_and_bindings_never_retain_source_text(tmp_path):
    path, audit = journal(tmp_path)
    values = [payload(content=""), payload("definition", representation="compact"), payload("line_range")]
    for value in values:
        record(audit, value)
    for error in ("line_range_requires_text", "line_range_out_of_bounds", "stale_source", "stale_generation"):
        record(audit, {"error": error}, success=False, request={"generation": 1, "path": "private-module.py"})
    audit.close()
    summary = mcp.summarize_repository_audit(path)
    observed = validate(summary)
    assert observed["complete"] is True
    assert observed["read_calls"] == 7 and observed["successful_reads"] == 3 and observed["failed_reads"] == 4
    assert [observed[key] for key in ("full_file_reads", "definition_reads", "line_range_reads")] == [1, 1, 1]
    assert observed["original_reads"] == 2 and observed["compact_reads"] == 1
    assert observed["generation_bindings_observed"] == 3 and observed["unbound_successes"] == 0
    assert observed["exposed_content_utf8_bytes"] == sum(len(value["content"].encode()) for value in values)
    assert (observed["line_range_selected_lines"], observed["line_range_total_lines"],
            observed["line_range_omitted_lines"]) == (2, 7, 5)
    assert all(observed[key] is False for key in mcp.READ_FALSE)
    retained = path.read_text()
    for text in ("private-module.py", "private source text", "private line two", "private-symbol-id", "private_local_name"):
        assert text not in retained and text not in json.dumps(observed)
    rows = [json.loads(line) for line in retained.splitlines()]
    assert all(row["schema"] == "repository-source-mcp-audit@3" for row in rows[1:])
    chain = hashlib.sha256(b"").digest()
    for row in rows[1:]:
        chain = hashlib.sha256(chain + mcp._json(row)).digest()
    assert observed["event_chain_sha256"] == chain.hex()


@pytest.mark.parametrize("mixed", [False, True])
def test_legacy_reads_remain_unknown_or_explicitly_incomplete(tmp_path, mixed):
    path, audit = journal(tmp_path)
    record(audit, payload(representation="compact"), legacy=True)
    record(audit, {"error": "stale_source"}, legacy=True, success=False, request={})
    if mixed:
        record(audit, payload("line_range"))
    audit.close()
    summary = mcp.summarize_repository_audit(path)
    assert summary["status"] == "observed"
    if mixed:
        observed = validate(summary)
        assert observed["complete"] is False and observed["line_range_reads"] == 1
        assert observed["legacy_read_calls"] == 2 and observed["legacy_successful_reads"] == 1
        assert observed["legacy_compact_reads"] == 1 and observed["read_calls"] == 1
    else:
        assert "read_observation" not in summary


def test_unbound_successes_remain_in_the_exact_aggregate_population(tmp_path):
    path, audit = journal(tmp_path)
    value = payload(representation="compact")
    value["source_sha256"] = "unbound"
    record(audit, value)
    audit.close()
    observed = validate(mcp.summarize_repository_audit(path))
    assert observed["complete"] is False and observed["successful_reads"] == observed["unbound_successes"] == 1
    assert observed["unbound_compact_reads"] == 1 and observed["compact_reads"] == 0
    assert observed["generation_bindings_observed"] == observed["exposed_content_utf8_bytes"] == 0
    assert observed["line_range_reads"] == observed["full_file_reads"] == 0


@pytest.mark.parametrize("field,value", [
    ("successful_reads", True), ("read_calls", 0), ("read_calls", 513),
    ("failed_reads", 1), ("full_file_reads", 1), ("generation_bindings_observed", 0),
    ("original_reads", 0), ("compact_reads", 1), ("unbound_successes", 1),
    ("unbound_compact_reads", 1), ("legacy_successful_reads", 1), ("legacy_compact_reads", 1),
    ("line_range_selected_lines", 0), ("line_range_total_lines", 1), ("line_range_omitted_lines", 0),
    ("exposed_content_utf8_bytes", 1), ("exposed_content_utf8_bytes", 2**31),
    ("source_displacement_verified", True), ("provider_usage_measured", True),
    ("model_attention_verified", True), ("token_savings_measured", True),
    ("complete", False), ("event_chain_sha256", "foreign"),
])
def test_impossible_or_authoritative_read_summaries_refuse(tmp_path, field, value):
    path, audit = journal(tmp_path)
    record(audit, payload("line_range"))
    audit.close()
    summary = mcp.summarize_repository_audit(path)
    changed = deepcopy(summary["read_observation"])
    changed[field] = value
    with pytest.raises(ValueError, match="invalid_read_observation"):
        validate(summary, changed)


@pytest.mark.parametrize("changes", [
    {"tool_calls": True}, {"tool_calls": 0}, {"tool_calls": 513},
    {"errors": -1}, {"errors": True}, {"errors": 3}, {"errors": 2},
    {"successful_reads": True}, {"successful_reads": 0}, {"successful_reads": 2},
    {"compact_reads": True}, {"compact_reads": 1},
])
def test_read_summary_must_match_global_calls_errors_successes_and_compaction(tmp_path, changes):
    path, audit = journal(tmp_path)
    record(audit, payload("line_range"))
    audit.close()
    with pytest.raises(ValueError, match="invalid_read_observation"):
        validate(mcp.summarize_repository_audit(path), **changes)


@pytest.mark.parametrize("change", ["bool_line", "missing_binding", "foreign_field", "wrong_tool",
                                    "compact_range", "omission", "content_bytes", "zero_generation"])
def test_malformed_read_journals_are_unavailable(tmp_path, change):
    path, audit = journal(tmp_path)
    record(audit, payload("line_range"))
    audit.close()
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    event = rows[-1]["read"]
    if change == "bool_line": event["start_line"] = True
    elif change == "missing_binding": event["binding_sha256"] = None
    elif change == "foreign_field": event["source_text"] = "cannot be retained"
    elif change == "wrong_tool": rows[-1]["tool"] = "repository_refresh"
    elif change == "compact_range": event["representation"] = "compact"; rows[-1]["compact"] = True
    elif change == "omission": event["omitted_lines"] += 1
    elif change == "content_bytes": event["content_utf8_bytes"] = rows[-1]["returned_bytes"] + 1
    else: event["requested_generation"] = 0
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    summary = mcp.summarize_repository_audit(path)
    assert summary["status"] == "unavailable" and "read_observation" not in summary


def test_read_and_retrieval_observations_coexist_without_reusing_calls(tmp_path):
    path, audit = journal(tmp_path)
    record(audit, payload("line_range"))
    data = dict(generation=1, semantic_root_cid="private-root", semantic_state_cid="private-state",
                source_tree_sha256="1" * 64, candidates=[{}])
    audit.record("tool", tool="repository_context_query", returned_bytes=128,
        retrieval=mcp._retrieval_event("repository_context_query", {"generation": 1}, data, True))
    audit.close()
    summary = mcp.summarize_repository_audit(path)
    assert validate(summary)["read_calls"] == 1
    assert summary["retrieval_observation"]["query_calls"] == 1
    assert summary["tool_calls"] == 2


def test_real_child_retains_exact_scopes_binary_empty_refusals_and_stale_generations(tmp_path):
    config, path, digest = configuration(tmp_path)
    source = Path(config["workspace"])
    raw = b"// authored context\nfunction value(x) {\n  return x + 1;\n}\n"
    (source / "app.js").write_bytes(raw)
    (source / "empty.txt").write_bytes(b"")
    (source / "opaque.bin").write_bytes(b"\xff\x00\n")
    exposed = 0
    with real_client(path, digest) as client:
        def read(name, generation=1, **options):
            nonlocal exposed
            value, failed = client.call("repository_read", generation=generation, path=name,
                                        representation="original", **options)
            assert not failed, value
            exposed += len(value["content"].encode("utf-8"))
            return value
        whole = read("app.js")
        window = read("app.js", start_line=2, end_line=3)
        assert whole["source_sha256"] == window["source_sha256"] == hashlib.sha256(raw).hexdigest()
        assert read("empty.txt")["content"] == ""
        assert read("opaque.bin")["encoding"] == "base64"
        for name, error in (("opaque.bin", "line_range_requires_text"), ("empty.txt", "line_range_out_of_bounds")):
            failed, is_error = client.call("repository_read", generation=1, path=name,
                representation="original", start_line=1, end_line=1)
            assert is_error and failed == {"error": error}
        context, failed = client.call("repository_context_query", generation=1, query="add", limit=1)
        assert not failed and len(context["candidates"]) == 1
        read("calc.py", symbol_id=context["candidates"][0]["symbol_id"])
        dependency = source / "calc.py"
        dependency.write_bytes(dependency.read_bytes() + b"# external authored change\n")
        failed, is_error = client.call("repository_read", generation=1, path="app.js",
            representation="original", start_line=2, end_line=3)
        assert is_error and failed == {"error": "stale_source"}
        current, failed = client.call("repository_refresh", generation=1)
        assert not failed and current["generation"] == 2
        read("app.js", generation=2, start_line=2, end_line=3)
        failed, is_error = client.call("repository_read", generation=1, path="app.js",
            representation="original", start_line=2, end_line=3)
        assert is_error and failed == {"error": "stale_generation"}
    summary = mcp.summarize_repository_audit(config["audit_store"])
    observed = validate(summary)
    assert observed["complete"] is True
    assert observed["read_calls"] == 10 and observed["successful_reads"] == 6 and observed["failed_reads"] == 4
    assert (observed["full_file_reads"], observed["definition_reads"], observed["line_range_reads"]) == (3, 1, 2)
    assert observed["original_reads"] == observed["generation_bindings_observed"] == 6
    assert observed["compact_reads"] == 0 and observed["exposed_content_utf8_bytes"] == exposed
    assert (observed["line_range_selected_lines"], observed["line_range_total_lines"],
            observed["line_range_omitted_lines"]) == (4, 8, 4)
    assert summary["retrieval_observation"]["successful_queries"] == 1
    assert all(observed[key] is False for key in mcp.READ_FALSE)
    retained = Path(config["audit_store"]).read_text()
    assert "app.js" not in retained and "calc.py" not in retained and "return x" not in retained
