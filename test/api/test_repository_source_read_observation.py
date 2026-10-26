"""Pure prepared-read accounting; no source execution or provider consumption."""
from copy import deepcopy
import base64
import hashlib
import json

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_read_observation as reads


def _bound(*, generation=1, content="value = 'Ω'\n", path="calc.py", source="1" * 64,
           representation="compact", symbol=None, encoding="utf-8"):
    arguments = dict(generation=generation, path=path)
    payload = dict(generation=generation, path=path, source_sha256=source,
        representation=representation, encoding=encoding, content=content, proof_authority=False)
    if symbol is not None:
        arguments["symbol_id"] = symbol
        payload.update(symbol_id=symbol, scope="python_definition", symbol="f", edit_supported=True)
    return arguments, payload


def _event(**kwargs):
    arguments, payload = _bound(**kwargs)
    event = reads.make_read_event(arguments, payload, True)
    assert event is not None
    return event


def _row(sequence=1, *, event=None, legacy=False, success=True, tool="repository_read", size=256,
         compact=None):
    row = dict(schema="repository-source-mcp-audit@1" if legacy else "repository-source-mcp-audit@3",
        sequence=sequence, event="tool", tool=tool, success=success, returned_bytes=size,
        compact=(event is not None and event["representation"] == "compact") if compact is None else compact,
        canonical_changed=False, canonical_uncertain=False, refreshed=False)
    if not legacy:
        row["read_observation"] = deepcopy(event)
    return row


def _validate(value, rows):
    return reads.validate_read_observation(value,
        successful_reads=sum(row["tool"] == "repository_read" and row["success"] for row in rows),
        compact_reads=sum(row["tool"] == "repository_read" and row["success"] and row["compact"] for row in rows),
        returned_bytes=sum(row["returned_bytes"] for row in rows),
        tool_calls=sum(row["event"] == "tool" for row in rows),
        errors=sum(not row["success"] for row in rows))


def test_event_uses_bound_generation_source_identity_and_returned_utf8_content():
    arguments, payload = _bound()
    event = reads.make_read_event(arguments, payload, True)
    identity = dict(path="calc.py", source_sha256="1" * 64, scope="file", symbol_id=None)
    canonical = json.dumps(identity, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    assert event["source_identity_sha256"] == hashlib.sha256(canonical).hexdigest()
    assert event["content_sha256"] == hashlib.sha256(payload["content"].encode()).hexdigest()
    assert event["content_bytes"] == len(payload["content"].encode())
    assert event["generation"] == 1 and event["scope"] == "file"
    assert "calc.py" not in json.dumps(event) and "value" not in json.dumps(event)
    assert reads.validate_read_event(event) == event
    assert reads.validate_read_event(event) is not event


def test_binary_count_covers_base64_response_text_not_decoded_source():
    content = base64.b64encode(b"\x00\xff\x01\x02").decode()
    event = _event(content=content, encoding="base64")
    assert event["content_bytes"] == len(content.encode()) == 8
    assert event["content_sha256"] == hashlib.sha256(content.encode()).hexdigest()


def test_empty_content_has_a_real_identity_and_zero_bytes():
    event = _event(content="")
    assert event["content_bytes"] == 0 and event["content_sha256"] == hashlib.sha256(b"").hexdigest()
    rows = [_row(event=event), _row(2, event=event)]
    summary = reads.summarize_read_observation(rows)
    assert summary["first_identity_reads"] == summary["same_generation_repeat_reads"] == 1
    assert summary["content_bytes"] == summary["same_generation_repeat_bytes"] == 0


def test_symbol_normalization_keeps_file_and_other_symbol_identities_separate():
    file_event = _event()
    first, second = _event(symbol="symbol:calc.f"), _event(symbol="symbol:calc.g")
    assert first["scope"] == second["scope"] == "symbol"
    assert len({event["source_identity_sha256"] for event in (file_event, first, second)}) == 3
    assert len({event["content_sha256"] for event in (file_event, first, second)}) == 1


def test_rpc_ids_and_unrelated_response_fields_never_enter_identity():
    arguments, payload = _bound()
    before = reads.make_read_event(arguments, payload, True)
    arguments.update(request_id="private-first-rpc")
    payload.update(request_id="private-other-rpc", private_error="DO NOT EXPORT")
    assert reads.make_read_event(arguments, payload, True) == before


@pytest.mark.parametrize("success", [False, None, 0, 1, "true"])
def test_only_exact_success_observes_bound_output(success):
    assert reads.make_read_event(*_bound(), success) is None


@pytest.mark.parametrize("target,field,value", [
    ("arguments", "generation", True), ("arguments", "generation", 0),
    ("arguments", "generation", 9), ("payload", "generation", 2),
    ("payload", "generation", True), ("payload", "path", "other.py"),
    ("arguments", "path", "../calc.py"), ("arguments", "path", "calc.py\n"),
    ("payload", "source_sha256", True), ("payload", "source_sha256", "A" * 64),
    ("payload", "proof_authority", True), ("payload", "proof_authority", 0),
    ("payload", "representation", []), ("payload", "encoding", "bytes"),
    ("payload", "content", b"not a string"), ("payload", "content", "\ud800"),
    ("payload", "scope", "python_definition"), ("payload", "symbol_id", "unrequested"),
    ("arguments", "symbol_id", None), ("arguments", "representation", []),
])
def test_unbound_or_malformed_output_is_unknown_not_a_partial_identity(target, field, value):
    arguments, payload = _bound()
    (arguments if target == "arguments" else payload)[field] = value
    assert reads.make_read_event(arguments, payload, True) is None


@pytest.mark.parametrize("path", ["/calc.py", "dir//calc.py", "dir/./calc.py", ".git/config",
                                  "dir\\calc.py", "a" * 1025, "a/" * 64 + "calc.py"])
def test_noncanonical_or_unbounded_paths_do_not_create_an_identity(path):
    assert reads.make_read_event(*_bound(path=path), True) is None


def test_utf8_bound_is_bytes_not_characters():
    assert reads.make_read_event(*_bound(content="Ω" * (reads.MAX_CONTENT_BYTES // 2 + 1)), True) is None
    event = _event(content="x" * reads.MAX_CONTENT_BYTES)
    assert event["content_bytes"] == reads.MAX_CONTENT_BYTES


@pytest.mark.parametrize("content", ["not base64!", "YQ=", "YQ==\n", "YR=="])
def test_malformed_or_noncanonical_base64_is_unbound(content):
    assert reads.make_read_event(*_bound(content=content, encoding="base64"), True) is None


def test_explicit_original_cannot_be_replaced_by_compact_but_compact_can_fallback():
    arguments, payload = _bound()
    arguments["representation"] = "original"
    assert reads.make_read_event(arguments, payload, True) is None
    arguments["representation"] = "compact"
    payload["representation"] = "original"
    assert reads.make_read_event(arguments, payload, True)["representation"] == "original"


@pytest.mark.parametrize("field,value", [
    ("generation", True), ("generation", 9), ("content_bytes", True), ("content_bytes", -1),
    ("content_bytes", reads.MAX_CONTENT_BYTES + 1), ("content_sha256", "z" * 64),
    ("source_identity_sha256", []), ("scope", "python_definition"),
    ("encoding", ["utf-8"]), ("representation", "other"), ("schema", "other"),
    ("proof_authority", True), ("content_bytes", 0),
    ("content_sha256", hashlib.sha256(b"").hexdigest()),
])
def test_strict_event_validation_refuses_forged_types_fields_or_empty_digest(field, value):
    event = _event()
    event[field] = value
    with pytest.raises(ValueError, match="invalid_read_event"):
        reads.validate_read_event(event)


def test_first_same_generation_and_cross_generation_partitions_are_exact():
    first, refreshed = _event(generation=1), _event(generation=2)
    rows = [_row(1, event=first), _row(2, event=first), _row(3, event=refreshed),
            _row(4, event=refreshed), _row(5, event=refreshed)]
    summary = reads.summarize_read_observation(rows)
    size = first["content_bytes"]
    assert summary["complete"] is True
    assert summary["successful_reads"] == summary["identified_reads"] == 5
    assert summary["first_identity_reads"] == 1
    assert summary["same_generation_repeat_reads"] == 3
    assert summary["cross_generation_repeat_reads"] == 1
    assert summary["content_bytes"] == 5 * size
    assert summary["first_identity_bytes"] == size
    assert summary["same_generation_repeat_bytes"] == 3 * size
    assert summary["cross_generation_repeat_bytes"] == size
    assert [group["read_count"] for group in summary["groups"]] == [2, 3]
    assert _validate(summary, rows) == summary


@pytest.mark.parametrize("change", [{"source": "2" * 64}, {"path": "other.py"},
                                   {"symbol": "symbol:calc.f"}, {"representation": "original"}])
def test_changed_identity_or_representation_is_not_same_source_repeat(change):
    first, changed = _event(), _event(**change)
    assert first["content_sha256"] == changed["content_sha256"]
    rows = [_row(event=first), _row(2, event=changed)]
    summary = reads.summarize_read_observation(rows)
    assert summary["first_identity_reads"] == 2
    assert summary["same_generation_repeat_reads"] == summary["cross_generation_repeat_reads"] == 0


def test_generation_group_classification_is_not_invented_temporal_order():
    first, second = _event(generation=1), _event(generation=2)
    forward = reads.summarize_read_observation([_row(event=first), _row(2, event=second)])
    reverse = reads.summarize_read_observation([_row(event=second), _row(2, event=first)])
    assert forward["groups"] == reverse["groups"]
    assert forward["cross_generation_repeat_reads"] == reverse["cross_generation_repeat_reads"] == 1
    assert forward["event_chain_sha256"] != reverse["event_chain_sha256"]


@pytest.mark.parametrize("legacy", [False, True])
def test_unidentified_success_keeps_known_partition_but_all_content_totals_unknown(legacy):
    event = _event()
    rows = [_row(event=event), _row(2, event=event), _row(3, legacy=legacy)]
    summary = reads.summarize_read_observation(rows)
    assert summary["complete"] is False
    assert summary["successful_reads"] == 3 and summary["identified_reads"] == 2
    assert summary["legacy_successful_reads"] == int(legacy)
    assert summary["unbound_successful_reads"] == int(not legacy)
    assert summary["known_content_bytes"] == 2 * event["content_bytes"]
    assert summary["known_first_identity_reads"] == summary["known_same_generation_repeat_reads"] == 1
    for name in reads._TOTALS:
        assert summary[name] is None
    assert _validate(summary, rows) == summary


def test_failed_read_does_not_make_successful_content_population_incomplete():
    rows = [_row(event=_event()), _row(2, success=False, size=30)]
    summary = reads.summarize_read_observation(rows)
    assert summary["complete"] is True
    assert summary["read_call_count"] == 2 and summary["read_error_count"] == 1
    assert summary["successful_read_payload_bytes"] == 256
    assert summary["failed_read_payload_bytes"] == 30


def test_modern_failed_only_population_has_known_zero_successful_content():
    summary = reads.summarize_read_observation([_row(success=False, size=30)])
    assert summary["complete"] is True and summary["groups"] == []
    assert all(summary[name] == 0 for name in reads._TOTALS)
    assert summary["read_error_count"] == 1


def test_modern_unbound_only_population_is_not_hidden_as_absent():
    summary = reads.summarize_read_observation([_row()])
    assert summary["complete"] is False and summary["identified_reads"] == 0
    assert summary["unbound_successful_reads"] == 1 and summary["content_bytes"] is None


@pytest.mark.parametrize("rows", [[], [_row(legacy=True)], [_row(legacy=True, success=False)],
                                 [_row(legacy=True, tool="repository_status")]])
def test_no_modern_read_instrumentation_returns_none(rows):
    assert reads.summarize_read_observation(rows) is None


def test_operation_payload_partition_keeps_other_tools_and_errors_separate():
    rows = [_row(event=_event(), size=200), _row(2, success=False, size=30),
            _row(3, legacy=True, tool="repository_list", size=100)]
    summary = reads.summarize_read_observation(rows)
    assert summary["payload_byte_scope"] == "inner_payload_json_utf8"
    assert summary["successful_read_payload_bytes"] == 200
    assert summary["failed_read_payload_bytes"] == 30
    assert summary["other_operation_payload_bytes"] == 100
    assert summary["content_bytes"] < 200
    assert _validate(summary, rows) == summary


def test_maximum_read_population_fits_existing_private_audit_and_summary_bounds():
    initialize = _row(legacy=True, tool=None, size=0)
    initialize["event"] = "initialize"
    rows = [initialize] + [_row(index + 2, event=_event(path=f"module_{index}.py"))
                           for index in range(reads.MAX_READS)]
    audit_bytes = b"".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode() + b"\n" for row in rows)
    # The production MCP journal cap is 512 KiB; this checks actual row framing,
    # including initialize, for the maximum all-modern distinct-read population.
    assert len(audit_bytes) <= 512 * 1024
    summary = reads.summarize_read_observation(rows)
    assert len(summary["groups"]) == summary["first_identity_reads"] == reads.MAX_READS
    assert len(json.dumps(summary).encode()) <= reads.MAX_SUMMARY_BYTES
    assert _validate(summary, rows) == summary


@pytest.mark.parametrize("mutation", ["extra_read", "excess_rows", "sequence_gap", "boolean_sequence",
    "event_on_failure", "event_on_other_tool", "representation_mismatch", "content_exceeds_payload",
    "unknown_schema", "row_extra", "boolean_payload", "payload_overflow", "duplicate_initialize"])
def test_invalid_or_unbounded_audit_populations_are_refused(mutation):
    event = _event()
    rows = [_row(event=event)]
    if mutation == "extra_read": rows = [_row(i + 1, event=event) for i in range(reads.MAX_READS + 1)]
    elif mutation == "excess_rows": rows *= reads.MAX_READS + 2
    elif mutation == "sequence_gap": rows[0]["sequence"] = 2
    elif mutation == "boolean_sequence": rows[0]["sequence"] = True
    elif mutation == "event_on_failure": rows[0]["success"] = False
    elif mutation == "event_on_other_tool": rows[0]["tool"] = "repository_list"
    elif mutation == "representation_mismatch": rows[0]["compact"] = False
    elif mutation == "content_exceeds_payload": rows[0]["returned_bytes"] = event["content_bytes"] - 1
    elif mutation == "unknown_schema": rows[0]["schema"] = "unknown"
    elif mutation == "row_extra": rows[0]["private"] = "DO NOT EXPORT"
    elif mutation == "boolean_payload": rows[0]["returned_bytes"] = True
    elif mutation == "payload_overflow": rows[0]["returned_bytes"] = reads.MAX_PAYLOAD_BYTES + 1
    elif mutation == "duplicate_initialize":
        rows = [_row(i + 1, legacy=True, tool=None, size=0) for i in range(2)]
        for row in rows: row["event"] = "initialize"
    with pytest.raises(ValueError):
        reads.summarize_read_observation(rows)


@pytest.mark.parametrize("field,value", [
    ("complete", False), ("complete", 1), ("successful_reads", 2), ("identified_reads", 0),
    ("legacy_successful_reads", 1), ("unbound_successful_reads", True), ("read_error_count", 1),
    ("read_call_count", 0), ("known_content_bytes", 0), ("known_first_identity_reads", True),
    ("known_same_generation_repeat_reads", 1), ("known_cross_generation_repeat_reads", 1),
    ("first_identity_bytes", 0), ("successful_read_payload_bytes", 255),
    ("failed_read_payload_bytes", 1), ("other_operation_payload_bytes", 1),
    ("event_chain_sha256", []), ("provider_ingestion_verified", True), ("provider_tokens", 0),
    ("inline_overlap_bytes", 0), ("token_savings", False), ("proof_authority", True),
    ("completion_authority", 0), ("extra", "DO NOT EXPORT"),
    ("scope", "provider_consumed"), ("payload_byte_scope", "complete_wire_frame"),
    ("population_scope", "entire_provider_session"), ("journal_closure_verified", True),
])
def test_summary_rejects_forged_arithmetic_types_authority_and_extra_fields(field, value):
    rows = [_row(event=_event())]
    summary = reads.summarize_read_observation(rows)
    summary[field] = value
    with pytest.raises(ValueError):
        _validate(summary, rows)


@pytest.mark.parametrize("field,value", [("successful_reads", 0), ("successful_reads", True),
    ("compact_reads", 0), ("returned_bytes", 1), ("returned_bytes", True),
    ("tool_calls", 0), ("tool_calls", reads.MAX_READS + 1), ("errors", True),
    ("errors", -1), ("errors", 3)])
def test_summary_must_join_parent_counters(field, value):
    rows = [_row(event=_event())]
    summary = reads.summarize_read_observation(rows)
    parents = dict(successful_reads=1, compact_reads=1, returned_bytes=256, tool_calls=1, errors=0)
    parents[field] = value
    with pytest.raises(ValueError):
        reads.validate_read_observation(summary, **parents)


def test_failed_read_count_cannot_exceed_the_parent_error_population():
    rows = [_row(event=_event()), _row(2, success=False, size=30)]
    summary = reads.summarize_read_observation(rows)
    with pytest.raises(ValueError):
        reads.validate_read_observation(summary, successful_reads=1, compact_reads=1,
            returned_bytes=286, tool_calls=2, errors=0)


def test_initialization_failure_is_part_of_parent_errors_not_read_errors():
    initialize = _row(legacy=True, tool=None, success=False, size=0)
    initialize["event"] = "initialize"
    rows = [initialize, _row(2, success=False, size=30)]
    summary = reads.summarize_read_observation(rows)
    assert summary["read_error_count"] == 1
    assert _validate(summary, rows) == summary


def test_incomplete_population_cannot_export_known_subtotal_as_complete_total():
    rows = [_row(event=_event()), _row(2, legacy=True)]
    summary = reads.summarize_read_observation(rows)
    summary["content_bytes"] = summary["known_content_bytes"]
    with pytest.raises(ValueError):
        _validate(summary, rows)


@pytest.mark.parametrize("mutation", ["duplicate", "reverse", "boolean_count", "too_many", "extra",
                                       "conflicting_content_size", "conflicting_scope", "conflicting_source_content"])
def test_group_projection_refuses_contradictory_or_forged_identities(mutation):
    rows = [_row(event=_event(generation=1)), _row(2, event=_event(generation=2))]
    summary = reads.summarize_read_observation(rows)
    groups = summary["groups"]
    if mutation == "duplicate": groups[1] = deepcopy(groups[0])
    elif mutation == "reverse": groups.reverse()
    elif mutation == "boolean_count": groups[0]["read_count"] = True
    elif mutation == "too_many": groups.extend(deepcopy(groups[0]) for _ in range(reads.MAX_READS))
    elif mutation == "extra": groups[0]["path"] = "private.py"
    elif mutation == "conflicting_content_size": groups[1]["content_bytes"] += 1
    elif mutation == "conflicting_scope": groups[1]["scope"] = "symbol"
    elif mutation == "conflicting_source_content": groups[1]["content_sha256"] = "f" * 64
    with pytest.raises(ValueError):
        _validate(summary, rows)


def test_summary_output_is_detached_content_free_and_keeps_unknown_boundaries():
    arguments, payload = _bound(path="private_location.py", content="PRIVATE_SOURCE_MARKER")
    event = reads.make_read_event(arguments, payload, True)
    rows = [_row(event=event)]
    before = deepcopy(rows)
    summary = reads.summarize_read_observation(rows)
    copied = _validate(summary, rows)
    assert rows == before and copied == summary
    assert copied is not summary and copied["groups"] is not summary["groups"]
    assert copied["groups"][0] is not summary["groups"][0]
    encoded = json.dumps(summary)
    assert "PRIVATE_SOURCE_MARKER" not in encoded and "private_location.py" not in encoded
    for field in ("provider_ingestion_verified", "provider_tokens", "token_savings", "inline_overlap_bytes"):
        assert summary[field] is None
    assert summary["proof_authority"] is summary["completion_authority"] is False
    assert summary["population_scope"] == "supplied_validated_audit_rows"
    assert summary["journal_closure_verified"] is False
