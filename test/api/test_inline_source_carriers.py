"""Pure codec query and explicitly authored AST-wrapper source observations.

AST fixtures express the public wrapper layout, not native producer validation.
No source-read display row is counted as an original whole-file source.
"""
from copy import deepcopy
import hashlib
import json

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import inline_source_carriers as carriers
from ipfs_accelerate_py.agent_supervisor.runtime import inline_source_observation as inline
from ipfs_accelerate_py.agent_supervisor.runtime import semantic_query_reference as query


def _sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _native(sources, *, schema=query.TRANSPORT_SCHEMA):
    semantic = {"schema": "supervisor-semantic-worker-context@1", "task_id": "authored-task",
        "scope_cid": "authored-scope", "semantic_root_cid": "authored-root",
        "manifest": {path: {"sha256": _sha(text), "source_cid": "authored-source:" + path}
                     for path, text in sources.items()}, "raw_sources": dict(sources),
        "required_raw_paths": list(sources), "capsules": [], "admissions": [],
        "worker_projection": {"schema": "supervisor-semantic-worker-projection@1",
                              "query": next(iter(sources.values()), "authored objective")},
        "completion_authority": False}
    wire = {"schema": schema, "translation_cid": "authored-translation", "translation_table": {},
        "native_context": {"objective_id": "authored-task", "evidence": []},
        "translated_semantic": semantic, "native_suffix": "authored native suffix",
        "instructions": "authored representation instructions", "execution_authority": False,
        "completion_authority": False}
    if schema == query.COMPACT_TRANSPORT_SCHEMA:
        del wire["translation_table"]
        wire.update({key: semantic[key] for key in ("task_id", "scope_cid", "semantic_root_cid")})
    elif schema == query.METADATA_SCHEMA:
        wire.update(source_transport_schema=query.TRANSPORT_SCHEMA,
                    common_bindings={"capsules": {}, "admissions": {}, "admission_refs": {}},
                    metadata_instructions="authored metadata instructions")
    return wire


def query_view(*, schema=query.TRANSPORT_SCHEMA):
    """Run the actual pure query codec over authored, self-contained input."""
    source = "Preserve all authored requirements and the Unicode character Ω.\n" * 48
    return query.project_semantic_query_reference(_json(_native({"instruction.txt": source}, schema=schema)))


def source_wrapper(*, sources=None, schema=query.TRANSPORT_SCHEMA):
    """Authored public AST-wrapper shape; does not invoke a native producer."""
    sources = {"calc.py": "value = 'Ω'\r\n"} if sources is None else sources
    inner = _native(sources, schema=schema)
    semantic = inner["translated_semantic"]
    display_source = "# authored comment\ndef answer():\n    return 42\n"
    semantic["manifest"]["display.py"] = {"sha256": _sha(display_source), "source_cid": "authored-display"}
    semantic["capsules"] = [{"metadata": {"facets": [{"version_evidence": {"$source_input_ast": "v0"}}]}}]
    block = {"schema": "supervisor-source-read-input@1", "instructions": "authored display instructions",
        "views": [{"id": "v0", "path": "display.py", "symbol": "answer", "format": "compact",
                   "source_sha256": _sha(display_source), "content": "def answer():\n return 42\n"}],
        "edit_authority": False, "proof_authority": False, "completion_authority": False}
    return {"schema": carriers.SOURCE_INPUT_SCHEMA, "semantic_transport": inner,
            "source_read_view": block, "factored_ast": True,
            "execution_authority": False, "completion_authority": False}


def _observe(wire, *, suffix="\nOpaque suffix"):
    prompt = "Workspace advisory:\n" + (_json(wire) if type(wire) is dict else wire) + suffix
    value = inline.observe_inline_sources(model_prompt=prompt, workspace_advisory="Workspace advisory:\n",
                                         invocation_id="authored-invocation")
    return value, prompt


def _validate(value, prompt):
    return inline.validate_inline_sources(value, model_prompt_sha256=_sha(prompt),
        model_prompt_bytes=len(prompt.encode()), invocation_id_sha256=_sha("authored-invocation"))


@pytest.mark.parametrize("schema", [query.TRANSPORT_SCHEMA, query.COMPACT_TRANSPORT_SCHEMA, query.METADATA_SCHEMA])
def test_actual_pure_query_codec_counts_retained_literal_once_without_restoration(schema, monkeypatch):
    view = query_view(schema=schema)
    wire = json.loads(view.provider_prompt)
    source = wire["transport"]["translated_semantic"]["raw_sources"]["instruction.txt"]
    def forbidden(**kwargs):
        raise AssertionError("observer must not reconstruct the referenced query")
    monkeypatch.setattr(query, "restore_semantic_query_reference", forbidden)
    value, prompt = _observe(view.provider_prompt)
    assert value["schema"] == inline.CARRIER_SCHEMA and value["status"] == "observed"
    assert value["scope"] == "transport.translated_semantic.raw_sources"
    assert value["transport_schema"] == query.VIEW_SCHEMA
    assert value["source_count"] == 1 and value["logical_source_bytes"] == len(source.encode())
    assert value["entries"][0]["content_sha256"] == _sha(source)
    assert value["full_prompt_source_coverage"] is value["wire_overlap_bytes"] is None
    assert _validate(value, prompt) == value
    assert "instruction.txt" not in _json(value) and source not in _json(value)


@pytest.mark.parametrize("schema", [query.TRANSPORT_SCHEMA, query.COMPACT_TRANSPORT_SCHEMA])
@pytest.mark.parametrize("display_format", ["raw", "compact"])
def test_authored_ast_wrapper_counts_only_literal_sources_not_supplemental_views(schema, display_format):
    wire = source_wrapper(schema=schema)
    wire["source_read_view"]["views"][0]["format"] = display_format
    wire["source_read_view"]["views"][0]["symbol"] = "module"
    value, prompt = _observe(wire)
    source = wire["semantic_transport"]["translated_semantic"]["raw_sources"]["calc.py"]
    assert value["schema"] == inline.CARRIER_SCHEMA and value["status"] == "observed"
    assert value["scope"] == "semantic_transport.translated_semantic.raw_sources"
    assert value["source_count"] == 1 and value["logical_source_bytes"] == len(source.encode())
    assert value["entries"][0]["content_sha256"] == _sha(source)
    assert _validate(value, prompt) == value
    assert "display.py" not in _json(value) and "return 42" not in _json(value)


def test_ast_wrapper_empty_literal_slot_is_scope_qualified_zero():
    value, prompt = _observe(source_wrapper(sources={}))
    assert value["status"] == "observed" and value["entries"] == []
    assert value["source_count"] == value["logical_source_bytes"] == 0
    assert value["full_prompt_source_coverage"] is None
    assert _validate(value, prompt) == value


def test_source_wrapper_allows_broader_deferred_manifest_but_preserves_inline_cap():
    wire = source_wrapper(sources={f"f{i}.py": "x" for i in range(64)})
    semantic = wire["semantic_transport"]["translated_semantic"]
    semantic["manifest"].update({f"deferred{i}.py": {"sha256": _sha("deferred")} for i in range(65)})
    value, _ = _observe(wire)
    assert value["status"] == "observed" and value["source_count"] == 64
    semantic["raw_sources"]["excess.py"] = "x"
    semantic["manifest"]["excess.py"] = {"sha256": _sha("x")}
    value, _ = _observe(wire)
    assert value["status"] == "unavailable" and value["entries"] is None


@pytest.mark.parametrize("mutation", ["extra_outer", "instructions", "authority", "wrong_marker", "extra_marker",
    "missing_target", "nonliteral", "wrong_hash", "ambiguous_required", "nested_wrapper", "inner_extra"])
def test_query_wrapper_refuses_invalid_or_ambiguous_carrier_whole_population(mutation):
    wire = json.loads(query_view().provider_prompt)
    semantic = wire["transport"]["translated_semantic"]
    if mutation == "extra_outer": wire["private_extra"] = "PRIVATE"
    elif mutation == "instructions": wire["instructions"] = "changed"
    elif mutation == "authority": wire["execution_authority"] = True
    elif mutation == "wrong_marker": semantic["worker_projection"]["query"] = {"$other": "instruction.txt"}
    elif mutation == "extra_marker": semantic["extra"] = {query.REF_KEY: "instruction.txt"}
    elif mutation == "missing_target": semantic["worker_projection"]["query"][query.REF_KEY] = "missing.txt"
    elif mutation == "nonliteral": semantic["raw_sources"]["instruction.txt"] = {"$reference": "PRIVATE"}
    elif mutation == "wrong_hash": semantic["manifest"]["instruction.txt"]["sha256"] = "f" * 64
    elif mutation == "ambiguous_required":
        semantic["raw_sources"]["duplicate.txt"] = semantic["raw_sources"]["instruction.txt"]
        semantic["manifest"]["duplicate.txt"] = dict(semantic["manifest"]["instruction.txt"])
        semantic["required_raw_paths"].append("duplicate.txt")
    elif mutation == "nested_wrapper": wire["transport"] = source_wrapper()
    elif mutation == "inner_extra": wire["transport"]["extra"] = 1
    value, prompt = _observe(wire)
    assert value["status"] == "unavailable" and value["schema"] == inline.SCHEMA
    assert value["entries"] is None and value["reason"] == "invalid_source_population"
    assert _validate(value, prompt) == value and "PRIVATE" not in _json(value)


@pytest.mark.parametrize("mutation", ["extra_outer", "factored_false", "factored_integer", "outer_authority",
    "nested_wrapper", "inner_schema", "inner_extra", "inner_authority", "block_extra", "block_schema",
    "block_authority", "block_views", "empty_views", "view_extra", "view_identity", "view_format",
    "view_nonliteral", "view_path", "view_sha", "view_unbound", "view_overlap", "block_limit",
    "raw_reference", "raw_hash"])
def test_authored_ast_wrapper_refuses_invalid_closed_shape_and_bindings(mutation):
    wire = source_wrapper()
    inner, block = wire["semantic_transport"], wire["source_read_view"]
    semantic, row = inner["translated_semantic"], block["views"][0]
    if mutation == "extra_outer": wire["extra"] = 1
    elif mutation == "factored_false": wire["factored_ast"] = False
    elif mutation == "factored_integer": wire["factored_ast"] = 1
    elif mutation == "outer_authority": wire["completion_authority"] = True
    elif mutation == "nested_wrapper": wire["semantic_transport"] = json.loads(query_view().provider_prompt)
    elif mutation == "inner_schema": inner["schema"] = query.METADATA_SCHEMA
    elif mutation == "inner_extra": inner["extra"] = 1
    elif mutation == "inner_authority": inner["execution_authority"] = True
    elif mutation == "block_extra": block["extra"] = 1
    elif mutation == "block_schema": block["schema"] = "unknown"
    elif mutation == "block_authority": block["proof_authority"] = True
    elif mutation == "block_views": block["views"] = {}
    elif mutation == "empty_views": block["views"] = []
    elif mutation == "view_extra": row["extra"] = 1
    elif mutation == "view_identity": row["id"] = "v2"
    elif mutation == "view_format": row["format"] = "original"
    elif mutation == "view_nonliteral": row["content"] = {"$source": "PRIVATE"}
    elif mutation == "view_path": row["path"] = "../display.py"
    elif mutation == "view_sha": row["source_sha256"] = True
    elif mutation == "view_unbound": row["source_sha256"] = "f" * 64
    elif mutation == "view_overlap":
        row["path"] = "calc.py"
        row["source_sha256"] = semantic["manifest"]["calc.py"]["sha256"]
    elif mutation == "block_limit": row["content"] = "x" * 32768
    elif mutation == "raw_reference": semantic["raw_sources"]["calc.py"] = {"$reference": "PRIVATE"}
    elif mutation == "raw_hash": semantic["manifest"]["calc.py"]["sha256"] = "f" * 64
    value, prompt = _observe(wire)
    assert value["status"] == "unavailable" and value["schema"] == inline.SCHEMA
    assert value["entries"] is None
    assert _validate(value, prompt) == value and "PRIVATE" not in _json(value)


def test_wrapper_in_opaque_suffix_cannot_replace_unsupported_final_carrier():
    value, prompt = _observe({"schema": "unknown"}, suffix=query_view().provider_prompt)
    assert value["status"] == "unavailable" and value["reason"] == "unsupported_transport"
    assert _validate(value, prompt) == value


@pytest.mark.parametrize("field,bad", [("schema", inline.SCHEMA), ("schema", "inline-source-observation@3"),
    ("scope", "translated_semantic.raw_sources"),
    ("scope", "translated_semantic.raw_sources_and_bound_public_instruction"),
    ("transport_schema", query.TRANSPORT_SCHEMA), ("transport_schema", carriers.SOURCE_INPUT_SCHEMA),
    ("proof_authority", True), ("provider_ingestion_verified", True)])
def test_v2_validator_rejects_version_scope_transport_and_authority_drift(field, bad):
    value, prompt = _observe(query_view().provider_prompt)
    value[field] = bad
    with pytest.raises(ValueError):
        _validate(value, prompt)


def test_direct_v1_cannot_be_relabelled_v2_and_retains_exact_old_fields():
    wire = _native({"calc.py": "x"})
    old, prompt = _observe(wire)
    newer, _ = _observe(source_wrapper())
    assert old["schema"] == inline.SCHEMA and set(old) == set(newer)
    old["schema"] = inline.CARRIER_SCHEMA
    with pytest.raises(ValueError):
        _validate(old, prompt)


def test_world_reference_unavailable_reason_remains_v1_with_no_inferred_zero():
    prompt = "bounded world reference"
    value = inline.unavailable_inline_sources(model_prompt=prompt, workspace_advisory="",
        invocation_id="authored-invocation", reason="unsupported_world_reference_source_carrier")
    assert value["schema"] == inline.SCHEMA and value["status"] == "unavailable"
    assert value["entries"] is value["source_count"] is value["logical_source_bytes"] is None
    assert _validate(value, prompt) == value
    value["schema"] = inline.CARRIER_SCHEMA
    with pytest.raises(ValueError):
        _validate(value, prompt)


def test_carrier_and_validation_do_not_modify_input_or_share_output_entries():
    wire = source_wrapper()
    retained = deepcopy(wire)
    value, prompt = _observe(wire)
    detached = _validate(value, prompt)
    detached["entries"][0]["content_bytes"] += 1
    assert wire == retained and detached != value
