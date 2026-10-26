"""Pure source-slot observations; fixtures author input, not provider delivery."""
from copy import deepcopy
import hashlib
import json

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import inline_source_observation as inline
from ipfs_accelerate_py.agent_supervisor.runtime.repository_source_read_observation import make_read_event


def _sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _wire(sources=None, *, schema="supervisor-semantic-router-input@1"):
    sources = {"calc.py": "value = 'Ω'\r\n"} if sources is None else sources
    return {"schema": schema, "translated_semantic": {"raw_sources": sources,
        "manifest": {path: {"sha256": _sha(content), "source_cid": "authored-source-cid"}
                     for path, content in sources.items()}}}


def _observe(wire=None, *, advisory="Workspace advisory:\n", suffix="\nOpaque suffix", identity="invocation-1"):
    wire = _wire() if wire is None else wire
    model = advisory + json.dumps(wire, ensure_ascii=True) + suffix
    return inline.observe_inline_sources(model_prompt=model, workspace_advisory=advisory,
                                        invocation_id=identity), model


def _validate(value, model, identity="invocation-1"):
    return inline.validate_inline_sources(value, model_prompt_sha256=_sha(model),
        model_prompt_bytes=len(model.encode()), invocation_id_sha256=_sha(identity))


@pytest.mark.parametrize("schema", sorted(inline.SUPPORTED_TRANSPORTS))
def test_supported_literal_slot_observation_uses_final_prompt_pins_and_read_identity(schema):
    wire = _wire(schema=schema)
    value, model = _observe(wire)
    assert value["status"] == "observed" and value["reason"] is None
    assert value["transport_schema"] == schema
    assert value["model_prompt_sha256"] == _sha(model)
    assert value["model_prompt_bytes"] == len(model.encode())
    assert value["invocation_id_sha256"] == _sha("invocation-1")
    source = wire["translated_semantic"]["raw_sources"]["calc.py"]
    event = make_read_event({"generation": 1, "path": "calc.py"},
        dict(generation=1, path="calc.py", source_sha256=_sha(source), representation="original",
             encoding="utf-8", content=source, proof_authority=False), True)
    assert value["entries"] == [{key: item for key, item in event.items() if key not in {"schema", "generation"}}]
    assert value["source_count"] == 1 and value["logical_source_bytes"] == len(source.encode())
    assert _validate(value, model) == value


def test_logical_bytes_preserve_unicode_crlf_and_do_not_count_json_escaping_or_wrappers():
    source = 'value = "λ\\Ω"\r\n'
    value, model = _observe(_wire({"calc.py": source}))
    assert value["logical_byte_scope"] == "decoded_utf8_source_text"
    assert value["logical_source_bytes"] == len(source.encode())
    assert value["logical_source_bytes"] < len(json.dumps(source).encode()) < len(model.encode())
    assert value["entries"][0]["content_sha256"] == _sha(source)


def test_manifest_only_sources_and_source_looking_suffixes_are_not_inline_slots():
    wire = _wire()
    semantic = wire["translated_semantic"]
    semantic["manifest"]["omitted.py"] = {"sha256": _sha("OMITTED_SOURCE")}
    semantic["required_raw_paths"] = ["omitted.py"]
    semantic["worker_projection"] = {"raw_source_fetch_required": {"omitted.py": {"sha256": _sha("OMITTED_SOURCE")}}}
    value, model = _observe(wire, suffix='\nraw_sources: {"suffix.py": "EXTRA_SOURCE"}')
    assert value["source_count"] == 1
    assert value["full_prompt_source_coverage"] is None
    assert "omitted.py" not in json.dumps(value) and "EXTRA_SOURCE" not in json.dumps(value)
    assert _validate(value, model) == value


def test_empty_literal_map_is_observed_zero_for_supported_slot_only():
    value, model = _observe(_wire({}))
    assert value["status"] == "observed" and value["entries"] == []
    assert value["source_count"] == value["logical_source_bytes"] == 0
    assert value["scope"] == "translated_semantic.raw_sources"
    assert value["full_prompt_source_coverage"] is None
    assert _validate(value, model) == value


def test_empty_source_has_exact_empty_digest_and_zero_logical_bytes():
    value, model = _observe(_wire({"empty.py": ""}))
    assert value["source_count"] == 1 and value["logical_source_bytes"] == 0
    assert value["entries"][0]["content_sha256"] == _sha("")
    assert _validate(value, model) == value


@pytest.mark.parametrize("schema", [None, [], True, "unknown-private-schema",
    "native-context@1", "repository-world-packet@1", "supervisor-semantic-query-reference-view@1"])
def test_unknown_native_and_referenced_transports_are_unavailable_not_empty(schema):
    value, model = _observe(_wire(schema=schema))
    assert value["status"] == "unavailable" and value["reason"] == "unsupported_transport"
    assert value["entries"] is value["source_count"] is value["logical_source_bytes"] is None
    assert value["transport_schema"] is None
    assert _validate(value, model) == value
    assert "unknown-private-schema" not in json.dumps(value)


@pytest.mark.parametrize("mutation", ["reference", "list_content", "wrong_hash", "boolean_hash",
    "missing_binding", "non_dict_binding", "bad_path", "surrogate_content", "missing_raw",
    "raw_list", "missing_manifest", "manifest_list", "missing_semantic", "semantic_list"])
def test_any_malformed_or_nonliteral_source_invalidates_whole_slot_population(mutation):
    wire = _wire({"good.py": "VALID_SOURCE", "bad.py": "BAD_SOURCE"})
    semantic = wire["translated_semantic"]
    if mutation == "reference": semantic["raw_sources"]["bad.py"] = {"$public_instruction": {"source_path": "bad.py"}}
    elif mutation == "list_content": semantic["raw_sources"]["bad.py"] = ["BAD_SOURCE"]
    elif mutation == "wrong_hash": semantic["manifest"]["bad.py"]["sha256"] = "f" * 64
    elif mutation == "boolean_hash": semantic["manifest"]["bad.py"]["sha256"] = True
    elif mutation == "missing_binding": del semantic["manifest"]["bad.py"]
    elif mutation == "non_dict_binding": semantic["manifest"]["bad.py"] = "PRIVATE_LOCATION"
    elif mutation == "bad_path":
        semantic["raw_sources"]["../bad.py"] = semantic["raw_sources"].pop("bad.py")
        semantic["manifest"]["../bad.py"] = semantic["manifest"].pop("bad.py")
    elif mutation == "surrogate_content": semantic["raw_sources"]["bad.py"] = "\ud800"
    elif mutation == "missing_raw": del semantic["raw_sources"]
    elif mutation == "raw_list": semantic["raw_sources"] = []
    elif mutation == "missing_manifest": del semantic["manifest"]
    elif mutation == "manifest_list": semantic["manifest"] = []
    elif mutation == "missing_semantic": del wire["translated_semantic"]
    elif mutation == "semantic_list": wire["translated_semantic"] = []
    value, model = _observe(wire)
    assert value["status"] == "unavailable"
    assert value["reason"] in {"invalid_source_population", "invalid_source_binding"}
    assert value["entries"] is None and value["source_count"] is None
    assert "VALID_SOURCE" not in json.dumps(value) and "PRIVATE_LOCATION" not in json.dumps(value)
    assert _validate(value, model) == value


@pytest.mark.parametrize("body", [
    '{"schema":"x","schema":"y"}', '{"value":NaN}', '{"value":Infinity}',
    '{"value":' + '[' * 40 + '1' + ']' * 40 + '}', '{"value":', 'not JSON',
    '{"items":[' + ','.join('0' for _ in range(20_001)) + ']}',
])
def test_existing_bounded_parser_rejects_duplicate_fields_nonfinite_depth_and_nodes(body):
    model = "advice" + body
    value = inline.observe_inline_sources(model_prompt=model, workspace_advisory="advice", invocation_id="invocation-1")
    assert value["status"] == "unavailable" and value["reason"] == "invalid_json"
    assert _validate(value, model) == value


def test_advisory_must_match_exact_leading_text_without_substring_search():
    model = json.dumps(_wire()) + "TRAILING_ADVISORY"
    value = inline.observe_inline_sources(model_prompt=model, workspace_advisory="TRAILING_ADVISORY", invocation_id="invocation-1")
    assert value["status"] == "unavailable" and value["reason"] == "advisory_mismatch"
    assert _validate(value, model) == value


@pytest.mark.parametrize("field,bad", [("model_prompt", None), ("model_prompt", ""),
    ("model_prompt", "x" * (inline.MAX_INPUT_BYTES + 1)), ("model_prompt", "\ud800"),
    ("workspace_advisory", None), ("workspace_advisory", "x" * 8193),
    ("invocation_id", ""), ("invocation_id", "x" * 513), ("invocation_id", "\ud800")])
def test_invalid_input_returns_closed_unavailable_pins_without_private_errors(field, bad):
    arguments = dict(model_prompt=json.dumps(_wire()), workspace_advisory="", invocation_id="invocation-1")
    arguments[field] = bad
    value = inline.observe_inline_sources(**arguments)
    assert value["status"] == "unavailable" and value["reason"] == "invalid_input"
    assert value["entries"] is None
    assert inline.validate_inline_sources(value, model_prompt_sha256=value["model_prompt_sha256"],
        model_prompt_bytes=value["model_prompt_bytes"], invocation_id_sha256=value["invocation_id_sha256"]) == value


def test_exact_input_bound_and_source_count_bound_are_supported_without_truncation():
    wire = _wire({f"file_{index}.py": "x" for index in range(inline.MAX_SOURCES)})
    body = json.dumps(wire)
    model = body + " " * (inline.MAX_INPUT_BYTES - len(body.encode()))
    value = inline.observe_inline_sources(model_prompt=model, workspace_advisory="", invocation_id="invocation-1")
    assert value["status"] == "observed" and value["source_count"] == inline.MAX_SOURCES
    assert value["model_prompt_bytes"] == inline.MAX_INPUT_BYTES
    assert len(json.dumps(value).encode()) <= inline.MAX_RESULT_BYTES
    assert value["entries"] == sorted(value["entries"], key=lambda entry: entry["source_identity_sha256"])
    assert _validate(value, model) == value
    wire["translated_semantic"]["manifest"]["excess.py"] = {"sha256": _sha("x")}
    wire["translated_semantic"]["raw_sources"]["excess.py"] = "x"
    too_many, _ = _observe(wire)
    assert too_many["status"] == "unavailable" and too_many["reason"] == "invalid_source_population"


def test_broader_manifest_does_not_inflate_or_refuse_small_inline_population():
    wire = _wire()
    manifest = wire["translated_semantic"]["manifest"]
    for index in range(64):
        manifest[f"deferred_{index}.py"] = {"sha256": _sha("deferred")}
    assert len(manifest) == 65
    value, model = _observe(wire)
    assert value["status"] == "observed" and value["source_count"] == 1
    assert _validate(value, model) == value


def test_deferred_manifest_still_has_an_independent_bounded_population():
    wire = _wire()
    manifest = wire["translated_semantic"]["manifest"]
    # Deferred bodies are not claimed inline or inspected; bounded empty
    # records keep this oversized manifest below the separate prompt byte cap.
    for index in range(inline.MAX_MANIFEST_ITEMS):
        manifest[f"deferred_{index}.py"] = {}
    value, model = _observe(wire)
    assert len(model.encode()) < inline.MAX_INPUT_BYTES
    assert value["status"] == "unavailable" and value["reason"] == "invalid_source_population"


@pytest.mark.parametrize("field,bad", [("model_prompt_sha256", "b" * 64), ("model_prompt_bytes", 1),
    ("model_prompt_bytes", True), ("invocation_id_sha256", "c" * 64),
    ("source_count", True), ("source_count", 2), ("logical_source_bytes", 0),
    ("status", []), ("reason", "private_exception"), ("transport_schema", "unknown"),
    ("scope", "all_prompt_sources"), ("logical_byte_scope", "serialized_json"),
    ("full_prompt_source_coverage", True), ("provider_tokens", 0), ("provider_ingestion_verified", True),
    ("wire_overlap_bytes", 0), ("proof_authority", True), ("completion_authority", 0),
    ("raw_content_exported", True), ("private_path", "PRIVATE_PATH")])
def test_validator_rejects_binding_arithmetic_authority_and_extra_fields(field, bad):
    value, model = _observe()
    value[field] = bad
    with pytest.raises(ValueError):
        _validate(value, model)


@pytest.mark.parametrize("mutation", ["duplicate", "reversed", "too_many", "path", "symbol_scope",
    "compact", "base64", "boolean_bytes", "empty_digest_mismatch", "same_digest_different_size"])
def test_entry_groups_are_strict_sorted_unique_whole_original_utf8_sources(mutation):
    value, model = _observe(_wire({"one.py": "same source", "two.py": "same source"}))
    entries = value["entries"]
    if mutation == "duplicate": entries[1] = deepcopy(entries[0])
    elif mutation == "reversed": entries.reverse()
    elif mutation == "too_many": entries.extend(deepcopy(entries[0]) for _ in range(inline.MAX_SOURCES))
    elif mutation == "path": entries[0]["path"] = "PRIVATE_PATH"
    elif mutation == "symbol_scope": entries[0]["scope"] = "symbol"
    elif mutation == "compact": entries[0]["representation"] = "compact"
    elif mutation == "base64": entries[0]["encoding"] = "base64"
    elif mutation == "boolean_bytes": entries[0]["content_bytes"] = True
    elif mutation == "empty_digest_mismatch": entries[0]["content_sha256"] = _sha("")
    elif mutation == "same_digest_different_size": entries[1]["content_bytes"] += 1
    with pytest.raises(ValueError):
        _validate(value, model)


def test_unavailable_cannot_be_relabelled_as_zero_source_coverage():
    value, model = _observe(_wire(schema="native"))
    value.update(entries=[], source_count=0, logical_source_bytes=0)
    with pytest.raises(ValueError):
        _validate(value, model)


def test_unavailable_constructor_closes_reason_and_preserves_only_valid_pins():
    model = json.dumps(_wire())
    value = inline.unavailable_inline_sources(model_prompt=model, workspace_advisory="PRIVATE_ADVISORY",
        invocation_id="invocation-1", reason="PRIVATE_EXCEPTION")
    assert value["reason"] == "observer_unavailable"
    assert "PRIVATE" not in json.dumps(value)
    assert _validate(value, model) == value


def test_result_is_detached_and_does_not_leak_raw_text_path_or_invocation():
    wire = _wire({"private/location.py": "PRIVATE_SOURCE"})
    before = deepcopy(wire)
    value, model = _observe(wire)
    detached = _validate(value, model)
    assert wire == before and detached == value
    assert detached is not value and detached["entries"] is not value["entries"]
    assert detached["entries"][0] is not value["entries"][0]
    encoded = json.dumps(value)
    for private in ("private/location.py", "PRIVATE_SOURCE", "invocation-1"):
        assert private not in encoded
    for unknown in ("full_prompt_source_coverage", "provider_tokens", "provider_ingestion_verified", "wire_overlap_bytes"):
        assert value[unknown] is None
    assert value["raw_content_exported"] is value["proof_authority"] is value["completion_authority"] is False
