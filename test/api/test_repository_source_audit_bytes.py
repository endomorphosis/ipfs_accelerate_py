"""Authored retained-audit bytes; no native sessions or provider consumption.

File parity uses only fresh caller-owned temporary files. The bytes API proves
parsing/accounting, not the provenance or delivery of the supplied records.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_mcp as mcp


COUNTERS = ("initializations", "tool_calls", "successful_reads", "compact_reads",
            "canonical_edits", "uncertain_edits", "refreshes", "errors", "returned_bytes")


def _row(sequence=1, *, tool="repository_status", **changes):
    row = dict(schema="repository-source-mcp-audit@1", sequence=sequence,
        event="initialize" if tool is None else "tool", tool=tool, success=True,
        returned_bytes=0 if tool is None else 128, compact=False,
        canonical_changed=False, canonical_uncertain=False, refreshed=False)
    row.update(changes)
    return row


def _line(row):
    return json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode() + b"\n"


def _raw(rows):
    return b"".join(_line(row) for row in rows)


def _read(sequence=1, *, generation=1, representation="compact", bound=True, success=True):
    content = b"value = 1\n"
    event = dict(schema="repository-source-read-event@1", generation=generation,
        source_identity_sha256="1" * 64, scope="file", representation=representation,
        encoding="utf-8", content_sha256=hashlib.sha256(content).hexdigest(), content_bytes=len(content))
    return _row(sequence, tool="repository_read", schema="repository-source-mcp-audit@3",
        compact=representation == "compact", success=success,
        read_observation=event if bound else None)


def _retrieval(sequence=1, *, tool="repository_context_query", count=2,
               bound=True, success=True, error=None):
    return _row(sequence, tool=tool, schema="repository-source-mcp-audit@2", success=success,
        retrieval=dict(requested_generation=1, binding_sha256="2" * 64 if bound else None,
            returned_items=count if bound else None, error=error))


def _read_rows():
    return [_row(tool=None), _read(2), _read(3),
            _row(4, tool="repository_refresh", refreshed=True), _read(5, generation=2)]


def _capsule_rows():
    return [_row(tool=None), _retrieval(2),
        _retrieval(3, tool="repository_capsule_expand", count=1),
        _retrieval(4, tool="repository_capsule_expand", bound=False, success=False, error="root_mismatch"),
        _retrieval(5, bound=False), _row(6, tool="repository_context_query"),
        _retrieval(7, bound=False, success=False, error="stale_generation"),
        _retrieval(8, tool="repository_capsule_expand", bound=False, success=False, error="stale_source")]


def _unavailable(value):
    assert value["status"] == "unavailable"
    assert all(value[name] is None for name in COUNTERS)
    assert all(item is None for name, item in value.items() if name != "status")


def _private_audit(tmp_path, raw):
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    private.chmod(0o700)
    path = private / "operations.jsonl"
    path.write_bytes(raw)
    path.chmod(0o600)
    return path


def test_bytes_empty_is_not_observed_with_known_zero_population():
    assert mcp.summarize_repository_audit_bytes(b"") == {
        "status": "not_observed", **dict.fromkeys(COUNTERS, 0)}


@pytest.mark.parametrize("raw", [None, "", 0, True, [], {}, bytearray(), memoryview(b"")],
                         ids=["none", "text", "integer", "boolean", "list", "dict", "bytearray", "memoryview"])
def test_bytes_wrong_type_is_unavailable_without_coercion(raw):
    _unavailable(mcp.summarize_repository_audit_bytes(raw))


def test_bytes_subclass_is_not_treated_as_exact_retained_bytes():
    class Derived(bytes):
        pass
    _unavailable(mcp.summarize_repository_audit_bytes(Derived(b"")))


def test_bytes_parser_performs_no_file_or_directory_access(monkeypatch):
    raw = _raw(_read_rows() + [_retrieval(6)])
    def refused(*args, **kwargs):
        pytest.fail("pure bytes summary attempted filesystem access")
    with monkeypatch.context() as patch:
        patch.setattr(mcp, "_private_directory", refused)
        patch.setattr(mcp.os, "open", refused)
        patch.setattr(mcp.os, "fstat", refused)
        patch.setattr(Path, "open", refused)
        result = mcp.summarize_repository_audit_bytes(raw)
    assert result["status"] == "observed"
    assert result["successful_reads"] == 3
    assert result["retrieval_observation"]["query_calls"] == 1


@pytest.mark.parametrize("size,observed", [(mcp.MAX_AUDIT_BYTES - 1, True),
    (mcp.MAX_AUDIT_BYTES, True), (mcp.MAX_AUDIT_BYTES + 1, False)],
    ids=["below-limit", "at-limit", "over-limit"])
def test_bytes_total_size_limit_includes_json_whitespace(size, observed):
    line = _line(_row())[:-1]
    raw = line + b" " * (size - len(line) - 1) + b"\n"
    assert len(raw) == size
    result = mcp.summarize_repository_audit_bytes(raw)
    if observed:
        assert result["status"] == "observed" and result["tool_calls"] == 1
    else:
        _unavailable(result)


@pytest.mark.parametrize("raw", [b"\n", b" \n", b"{}\n", b"[]\n", b"null\n", b"{\n",
    b"\xff\n", b'{"schema":NaN}\n', b'{"schema":Infinity}\n',
    b'{"schema":"first","schema":"second"}\n',
    b'{"nested":{"duplicate":0,"duplicate":1}}\n'],
    ids=["blank", "whitespace", "empty-object", "array", "null", "truncated", "utf8",
         "nan", "infinity", "duplicate-top-level", "duplicate-nested"])
def test_bytes_malformed_json_is_unavailable(raw):
    _unavailable(mcp.summarize_repository_audit_bytes(raw))


@pytest.mark.parametrize("suffix", [b"", b"\r", b" ", b"\n\n"],
                         ids=["missing-newline", "carriage-only", "space", "blank-trailing-row"])
def test_bytes_refuses_truncation_or_extra_blank_row(suffix):
    _unavailable(mcp.summarize_repository_audit_bytes(_line(_row())[:-1] + suffix))


def test_bytes_preserves_existing_crlf_line_handling():
    assert mcp.summarize_repository_audit_bytes(_line(_row())[:-1] + b"\r\n") == \
        mcp.summarize_repository_audit_bytes(_line(_row()))


@pytest.mark.parametrize("field,value", [
    ("schema", "repository-source-mcp-audit@4"), ("schema", []),
    ("sequence", True), ("sequence", 2), ("event", "other"), ("event", []),
    ("tool", "unknown_tool"), ("tool", []), ("success", 1), ("compact", 0),
    ("returned_bytes", True), ("returned_bytes", -1), ("returned_bytes", mcp.MAX_RESPONSE_BYTES + 1),
    ("extra", "private content"),
])
def test_bytes_malformed_schema_and_counters_invalidate_whole_summary(field, value):
    row = _row()
    row[field] = value
    _unavailable(mcp.summarize_repository_audit_bytes(_line(row)))


@pytest.mark.parametrize("mutation", ["missing", "duplicate_initialize", "sequence_gap", "tool_for_initialize"])
def test_bytes_record_population_checks_are_retained(mutation):
    rows = [_row(tool=None), _row(2)]
    if mutation == "missing":
        del rows[1]["success"]
    elif mutation == "duplicate_initialize":
        rows[1] = _row(2, tool=None)
    elif mutation == "sequence_gap":
        rows[1]["sequence"] = 3
    else:
        rows[0]["tool"] = "repository_status"
    _unavailable(mcp.summarize_repository_audit_bytes(_raw(rows)))


@pytest.mark.parametrize("tools,initialize,observed", [
    (512, False, True), (512, True, True), (513, False, False), (513, True, False)])
def test_bytes_operation_and_row_limits_remain_distinct(tools, initialize, observed):
    rows = [_row(tool=None)] if initialize else []
    rows.extend(_row(sequence) for sequence in range(len(rows) + 1, len(rows) + tools + 1))
    result = mcp.summarize_repository_audit_bytes(_raw(rows))
    if observed:
        assert result["status"] == "observed"
        assert result["tool_calls"] == tools and result["initializations"] == int(initialize)
    else:
        _unavailable(result)


def test_bytes_legacy_operations_preserve_counts_without_invented_observations():
    rows = [_row(tool=None), _row(2, tool="repository_read", compact=True),
        _row(3, tool="repository_replace_symbol", canonical_changed=True),
        _row(4, tool="repository_replace_symbol", canonical_uncertain=True, success=False),
        _row(5, tool="repository_refresh", refreshed=True)]
    assert mcp.summarize_repository_audit_bytes(_raw(rows)) == dict(
        status="observed", initializations=1, tool_calls=4, successful_reads=1,
        compact_reads=1, canonical_edits=1, uncertain_edits=1, refreshes=1, errors=1,
        returned_bytes=4 * 128)


def test_bytes_read_repeats_preserve_content_and_payload_accounting():
    result = mcp.summarize_repository_audit_bytes(_raw(_read_rows()))
    assert result["status"] == "observed"
    assert (result["initializations"], result["tool_calls"], result["successful_reads"],
            result["compact_reads"], result["refreshes"], result["returned_bytes"]) == (1, 4, 3, 3, 1, 512)
    observed = result["read_observation"]
    assert observed["complete"] is True and observed["identified_reads"] == 3
    assert observed["first_identity_reads"] == observed["same_generation_repeat_reads"] == \
        observed["cross_generation_repeat_reads"] == 1
    assert observed["content_bytes"] == 3 * len(b"value = 1\n")
    assert observed["successful_read_payload_bytes"] == 384
    assert observed["other_operation_payload_bytes"] == 128
    assert observed["provider_ingestion_verified"] is None and observed["provider_tokens"] is None
    assert observed["token_savings"] is None and observed["proof_authority"] is False


def test_bytes_unknown_successful_reads_keep_full_content_totals_null():
    rows = [_read(), _read(2, bound=False), _read(3, bound=False, success=False),
            _row(4, tool="repository_read", compact=True)]
    result = mcp.summarize_repository_audit_bytes(_raw(rows))
    assert result["status"] == "observed" and result["errors"] == 1
    observed = result["read_observation"]
    assert observed["complete"] is False
    assert observed["identified_reads"] == observed["unbound_successful_reads"] == \
        observed["legacy_successful_reads"] == observed["read_error_count"] == 1
    assert observed["content_bytes"] is None
    assert observed["known_content_bytes"] == len(b"value = 1\n")


@pytest.mark.parametrize("mutation", ["wrong_tool", "failed_bound", "compact_mismatch", "bad_digest",
    "boolean_generation", "oversize_content", "payload_too_small", "extra_event_field"])
def test_bytes_modern_read_validation_is_shared(mutation):
    row = _read()
    event = row["read_observation"]
    if mutation == "wrong_tool": row["tool"] = "repository_status"
    elif mutation == "failed_bound": row["success"] = False
    elif mutation == "compact_mismatch": row["compact"] = False
    elif mutation == "bad_digest": event["source_identity_sha256"] = "A" * 64
    elif mutation == "boolean_generation": event["generation"] = True
    elif mutation == "oversize_content": event["content_bytes"] = 1024 * 1024 + 1
    elif mutation == "payload_too_small": row["returned_bytes"] = 0
    else: event["source_text"] = "not an audit field"
    _unavailable(mcp.summarize_repository_audit_bytes(_line(row)))


def test_bytes_capsule_outcomes_preserve_unknowns_errors_and_ordered_commitment():
    rows = _capsule_rows()
    result = mcp.summarize_repository_audit_bytes(_raw(rows))
    assert result["status"] == "observed" and result["tool_calls"] == 7 and result["errors"] == 3
    observed = result["retrieval_observation"]
    expected = dict(query_calls=3, successful_queries=2, empty_queries=0, nonempty_queries=1,
        candidates_returned=2, expansion_calls=3, successful_expansions=1, empty_expansions=0,
        nonempty_expansions=1, capsules_returned=1, errors=3, stale_generation_errors=1,
        stale_source_errors=1, root_mismatch_errors=1, legacy_retrieval_calls=1,
        unbound_successes=1, generation_bindings_observed=2)
    assert {key: observed[key] for key in expected} == expected
    assert observed["complete"] is False
    assert all(observed[key] is False for key in mcp.RETRIEVAL_FALSE)
    chain = hashlib.sha256(b"").hexdigest()
    for row in rows:
        if row["schema"] == "repository-source-mcp-audit@2":
            chain = hashlib.sha256(bytes.fromhex(chain) + _line(row)[:-1]).hexdigest()
    assert observed["event_chain_sha256"] == chain


@pytest.mark.parametrize("tool", ["repository_context_query", "repository_capsule_expand"])
def test_bytes_empty_bound_retrieval_is_known_empty_not_unbound(tool):
    result = mcp.summarize_repository_audit_bytes(_line(_retrieval(tool=tool, count=0)))
    observed = result["retrieval_observation"]
    assert observed["complete"] is True and observed["unbound_successes"] == 0
    assert observed["generation_bindings_observed"] == 1
    assert observed["empty_queries" if tool == "repository_context_query" else "empty_expansions"] == 1


@pytest.mark.parametrize("mutation", ["wrong_tool", "missing", "extra", "bad_digest", "boolean_count",
    "oversize_count", "unbound_count", "failed_bound", "error_on_success", "failed_without_error", "bad_error"])
def test_bytes_capsule_validation_is_shared(mutation):
    row = _retrieval()
    event = row["retrieval"]
    if mutation == "wrong_tool": row["tool"] = "repository_status"
    elif mutation == "missing": del event["error"]
    elif mutation == "extra": event["query"] = "private query"
    elif mutation == "bad_digest": event["binding_sha256"] = "x" * 64
    elif mutation == "boolean_count": event["returned_items"] = True
    elif mutation == "oversize_count": event["returned_items"] = 33
    elif mutation == "unbound_count": event["binding_sha256"] = None
    elif mutation == "failed_bound": row["success"] = False; event["error"] = "other"
    elif mutation == "error_on_success": event["error"] = "other"
    elif mutation == "failed_without_error":
        row["success"] = False
        event.update(binding_sha256=None, returned_items=None)
    else: event["error"] = []
    _unavailable(mcp.summarize_repository_audit_bytes(_line(row)))


@pytest.mark.parametrize("kind", ["empty", "legacy", "reads", "capsules", "mixed", "malformed",
    "duplicate", "schema", "missing_newline", "oversize"])
def test_bytes_and_secure_path_summaries_are_identical(tmp_path, kind):
    samples = dict(empty=b"", legacy=_line(_row()), reads=_raw(_read_rows()),
        capsules=_raw(_capsule_rows()), mixed=_raw(_read_rows() + [_retrieval(6)]),
        malformed=b"{\n", duplicate=b'{"schema":1,"schema":2}\n',
        schema=_line(_row(schema="unknown")), missing_newline=_line(_row())[:-1],
        oversize=b" " * mcp.MAX_AUDIT_BYTES + b"\n")
    raw = samples[kind]
    path = _private_audit(tmp_path, raw)
    assert mcp.summarize_repository_audit(path) == mcp.summarize_repository_audit_bytes(raw)
    assert path.read_bytes() == raw and path.stat().st_mode & 0o777 == 0o600


def test_path_missing_file_and_missing_directory_remain_not_observed(tmp_path):
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    expected = {"status": "not_observed", **dict.fromkeys(COUNTERS, 0)}
    assert mcp.summarize_repository_audit(private / "missing.jsonl") == expected
    assert mcp.summarize_repository_audit(tmp_path / "missing" / "missing.jsonl") == expected


@pytest.mark.parametrize("mutation", ["read_only_copy", "public_file", "public_directory",
    "symlink", "hardlink", "directory", "relative"])
def test_path_permissions_and_link_checks_remain_stricter_than_supplied_bytes(tmp_path, mutation):
    raw = _line(_row())
    path = _private_audit(tmp_path, raw)
    if mutation == "read_only_copy": path.chmod(0o400)
    elif mutation == "public_file": path.chmod(0o644)
    elif mutation == "public_directory": path.parent.chmod(0o755)
    elif mutation == "symlink":
        link = path.with_name("alias.jsonl")
        link.symlink_to(path)
        path = link
    elif mutation == "hardlink": os.link(path, path.with_name("alias.jsonl"))
    elif mutation == "directory":
        path.unlink()
        path.mkdir(mode=0o600)
    else: path = Path(os.path.relpath(path))
    try:
        _unavailable(mcp.summarize_repository_audit(path))
        assert mcp.summarize_repository_audit_bytes(raw)["status"] == "observed"
    finally:
        if mutation == "directory": path.chmod(0o700)
        if mutation == "public_directory": path.parent.chmod(0o700)


def test_bytes_summary_is_detached_between_calls():
    raw = _raw(_read_rows())
    before = mcp.summarize_repository_audit_bytes(raw)
    expected = deepcopy(before)
    before["read_observation"]["groups"][0]["read_count"] = 99
    before["status"] = "unavailable"
    assert mcp.summarize_repository_audit_bytes(raw) == expected
