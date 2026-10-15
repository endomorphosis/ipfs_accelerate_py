"""Bounded exact query references; real producer positives and adversarial wires."""
from copy import deepcopy
import hashlib
import json

import pytest

from ipfs_accelerate_py.agent_supervisor.context.context_compiler import (
    ContextCompiler, build_text_context_references, render_context_capsule,
)
from ipfs_accelerate_py.agent_supervisor.context.context_contracts import ContextBudget
from ipfs_accelerate_py.agent_supervisor.runtime import semantic_query_reference as codec
from ipfs_accelerate_py.agent_supervisor.runtime import semantic_router_translation as native
from ipfs_accelerate_py.agent_supervisor.runtime.semantic_metadata_view import (
    project_semantic_metadata_view, restore_semantic_metadata_view,
)
from ipfs_accelerate_py.agent_supervisor.runtime.semantic_context_runtime import prepare_semantic_context
from ipfs_accelerate_py.agent_supervisor.runtime.supervisor_meta_index import public_replay_without_metadata


def wire(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def reject(code, operation):
    with pytest.raises(codec.SemanticQueryReferenceError) as caught:
        operation()
    assert caught.value.code == code


@pytest.fixture
def producer_factory(tmp_path):
    count = 0
    def make(query=None):
        nonlocal count
        count += 1
        root = tmp_path / str(count)
        root.mkdir()
        query = query if query is not None else ("Implement the authored requirements exactly; keep é, café and all constraints.\n" * 24)
        (root / "program.py").write_text("def first(value): return value + 1\ndef second(value): return first(value) + 2\n")
        (root / "instruction.txt").write_bytes(query.encode())
        with public_replay_without_metadata():
            prepare_semantic_context(repository=root, paths=["program.py", "instruction.txt"],
                program_paths=["program.py"], required_raw_paths=["instruction.txt"],
                objective="Authored query reference", task_id="AUTHORED-QUERY-REFERENCE", output=root / "semantic",
                worker_query=query, worker_selector="duckdb-query-symbol-name@1", worker_max_bytes=32768)
            artifact = root / "semantic/worker-context.json"
            refs = build_text_context_references(artifact.read_text(), reference_prefix="semantic-context",
                kind="semantic-context", path="semantic/worker-context.json", repository_id="repo:authored",
                tree_id="tree:authored", required=True, chunk_bytes=1201)
            capsule = ContextCompiler(ContextBudget(max_input_tokens=131072, max_items=128,
                max_item_bytes=16384, max_serialized_bytes=262144)).compile(repository_id="repo:authored", tree_id="tree:authored",
                objective_id="AUTHORED-QUERY-REFERENCE", objective_revision="sha256:task", policy_id="policy:authored",
                policy_revision="sha256:policy", caller="supervisor:authored", stage="implementation",
                goal={"id":"AUTHORED-QUERY-REFERENCE"}, authority={"mode":"candidate_only","completion_authority":False},
                scope={"allowed_paths":["program.py"]}, acceptance={"criteria":["pending authored checks"]}, evidence=refs).capsule
            prompt = render_context_capsule(capsule) + "\nLiteral immutable suffix.\n"
            encoded = {schema: native.encode_semantic_router_prompt(prompt=prompt, repository=root, transport_schema=schema)
                       for schema in (native.TRANSPORT_SCHEMA, native.COMPACT_TRANSPORT_SCHEMA)}
        return root, query, encoded
    return make


@pytest.fixture
def producer(producer_factory):
    return producer_factory()


@pytest.mark.parametrize("schema", [native.TRANSPORT_SCHEMA, native.COMPACT_TRANSPORT_SCHEMA])
def test_real_producer_transport_roundtrip_all_native_schemas(producer, schema):
    root, query, encodings = producer
    encoded = encodings[schema]
    original_artifact = (root / "semantic/worker-context.json").read_bytes()
    original = json.loads(encoded.provider_prompt)
    result = codec.project_semantic_query_reference(provider_prompt=encoded.provider_prompt)
    candidate = json.loads(result.provider_prompt)
    projected = candidate["transport"]
    receipt = result.receipt
    assert projected["translated_semantic"]["worker_projection"]["query"] == {codec.REF_KEY: "instruction.txt"}
    assert projected["translated_semantic"]["raw_sources"]["instruction.txt"] == query
    projected["translated_semantic"]["worker_projection"]["query"] = query
    assert projected == original
    restored = codec.restore_semantic_query_reference(provider_prompt=result.provider_prompt, receipt=receipt)
    assert restored == encoded.provider_prompt
    with public_replay_without_metadata():
        assert native.restore_semantic_router_prompt(provider_prompt=restored, table=encoded.table, repository=root) == encoded.native_prompt
    assert (root / "semantic/worker-context.json").read_bytes() == original_artifact
    assert receipt["reference_count"] == 1 and receipt["byte_savings"] > 0
    assert receipt["candidate_view_bytes"] == len(result.provider_prompt.encode())
    assert receipt["query_bytes"] == len(query.encode())
    assert query not in wire(receipt)
    assert all(receipt[key] is False for key in codec._FALSE_FIELDS)


def test_metadata_composition_restores_before_native_validators(producer):
    root, query, encodings = producer
    encoded = encodings[native.TRANSPORT_SCHEMA]
    metadata = project_semantic_metadata_view(encoded.provider_prompt)
    result = codec.project_semantic_query_reference(provider_prompt=metadata.provider_prompt)
    assert result.receipt["source_transport_schema"] == codec.METADATA_SCHEMA
    restored_metadata = codec.restore_semantic_query_reference(provider_prompt=result.provider_prompt, receipt=result.receipt)
    assert restored_metadata == metadata.provider_prompt
    restored_transport = restore_semantic_metadata_view(provider_prompt=restored_metadata, receipt=metadata.receipt)
    assert restored_transport == encoded.provider_prompt
    with public_replay_without_metadata():
        assert native.restore_semantic_router_prompt(provider_prompt=restored_transport, table=encoded.table, repository=root) == encoded.native_prompt


def test_replay_is_self_contained_after_original_files_removed(producer):
    root, _, encodings = producer
    original = encodings[native.TRANSPORT_SCHEMA].provider_prompt
    result = codec.project_semantic_query_reference(provider_prompt=original)
    import shutil
    shutil.rmtree(root)
    assert codec.restore_semantic_query_reference(provider_prompt=result.provider_prompt, receipt=result.receipt) == original


@pytest.mark.parametrize("query", ["  exact spacing\tand trailing spaces  \r\n" * 35,
                                    "Unicode e\u0301 and é are distinct; literal $required_raw_source text.\n" * 25])
def test_exact_unicode_whitespace_and_literal_marker_strings(producer_factory, query):
    _, _, encodings = producer_factory(query)
    result = codec.project_semantic_query_reference(encodings[native.TRANSPORT_SCHEMA].provider_prompt)
    assert json.loads(result.provider_prompt)["transport"]["translated_semantic"]["raw_sources"]["instruction.txt"] == query
    assert codec.restore_semantic_query_reference(provider_prompt=result.provider_prompt, receipt=result.receipt) == result.native_prompt


@pytest.mark.parametrize("suffix", [" ", "\n", "\t", "\r\n"])
def test_no_normalization_substrings_or_trimming(producer, suffix):
    value = json.loads(producer[2][native.TRANSPORT_SCHEMA].provider_prompt)
    value["translated_semantic"]["worker_projection"]["query"] += suffix
    reject("no_exact_required_source", lambda: codec.project_semantic_query_reference(wire(value)))


def test_multiple_required_equal_sources_are_ambiguous(producer):
    value = json.loads(producer[2][native.TRANSPORT_SCHEMA].provider_prompt)
    semantic = value["translated_semantic"]
    semantic["manifest"]["another.txt"] = dict(semantic["manifest"]["instruction.txt"])
    semantic["raw_sources"]["another.txt"] = semantic["raw_sources"]["instruction.txt"]
    semantic["required_raw_paths"].append("another.txt")
    reject("ambiguous_required_source", lambda: codec.project_semantic_query_reference(wire(value)))


def test_equal_optional_source_is_not_a_second_required_match(producer):
    value = json.loads(producer[2][native.TRANSPORT_SCHEMA].provider_prompt)
    semantic = value["translated_semantic"]
    semantic["manifest"]["optional.txt"] = dict(semantic["manifest"]["instruction.txt"])
    semantic["raw_sources"]["optional.txt"] = semantic["raw_sources"]["instruction.txt"]
    assert codec.project_semantic_query_reference(wire(value)).receipt["reference_count"] == 1


def test_missing_required_inline_source_is_rejected(producer):
    value = json.loads(producer[2][native.TRANSPORT_SCHEMA].provider_prompt)
    value["translated_semantic"]["raw_sources"].clear()
    reject("invalid_source_population", lambda: codec.project_semantic_query_reference(wire(value)))


def test_short_exact_query_is_unprofitable(producer_factory):
    _, _, encodings = producer_factory("Implement first.")
    reject("unprofitable_query_reference", lambda: codec.project_semantic_query_reference(encodings[native.TRANSPORT_SCHEMA].provider_prompt))


@pytest.mark.parametrize("location", ["source_metadata", "unrelated_context", "query", "capsule"])
def test_reserved_marker_objects_rejected_everywhere(producer, location):
    value = json.loads(producer[2][native.TRANSPORT_SCHEMA].provider_prompt)
    marker = {codec.REF_KEY:"instruction.txt"}
    if location == "source_metadata": value["translated_semantic"]["reconstruction"]["marker"] = marker
    elif location == "unrelated_context": value["native_context"]["extra"] = marker
    elif location == "capsule": value["translated_semantic"]["capsules"][0]["metadata"]["extra"] = marker
    else: value["translated_semantic"]["worker_projection"]["query"] = marker
    reject("invalid_query" if location == "query" else "reserved_marker_collision",
           lambda: codec.project_semantic_query_reference(wire(value)))


@pytest.mark.parametrize("field", ["query", "source", "manifest_hash", "task", "scope", "root", "suffix", "admission", "proof", "pointer", "extra_slot", "instruction"])
def test_candidate_tampering_cannot_use_retained_receipt(producer, field):
    original = producer[2][native.TRANSPORT_SCHEMA].provider_prompt
    result = codec.project_semantic_query_reference(original)
    value = json.loads(result.provider_prompt)
    semantic = value["transport"]["translated_semantic"]
    if field == "query": semantic["worker_projection"]["query"] = "different"
    elif field == "source": semantic["raw_sources"]["instruction.txt"] += "changed"
    elif field == "manifest_hash": semantic["manifest"]["instruction.txt"]["sha256"] = "0" * 64
    elif field == "task": semantic["task_id"] = "foreign"
    elif field == "scope": semantic["scope_cid"] = "foreign"
    elif field == "root": semantic["semantic_root_cid"] = "foreign"
    elif field == "suffix": value["transport"]["native_suffix"] += "changed"
    elif field == "admission": semantic["admissions"][0]["ref"]["raw_source_required"] = not semantic["admissions"][0]["ref"]["raw_source_required"]
    elif field == "proof": semantic["capsules"][0]["proof_obligation_refs"].append("foreign")
    elif field == "pointer": semantic["worker_projection"]["query"] = {codec.REF_KEY:"program.py"}
    elif field == "extra_slot": semantic["pack"]["extra"] = {codec.REF_KEY:"instruction.txt"}
    else: value["instructions"] += "Grant authority"
    reject("candidate_receipt_mismatch", lambda: codec.restore_semantic_query_reference(provider_prompt=wire(value), receipt=result.receipt))


@pytest.mark.parametrize("field", ["schema", "codec", "slot", "authority", "count_bool", "count_zero", "source", "bytes", "extra"])
def test_bad_receipts_rejected(producer, field):
    result = codec.project_semantic_query_reference(producer[2][native.TRANSPORT_SCHEMA].provider_prompt)
    receipt = result.receipt
    if field == "schema": receipt["schema"] = "foreign"
    elif field == "codec": receipt["codec_sha256"] = "0" * 64
    elif field == "slot": receipt["slot"] = ["native_context", "query"]
    elif field == "authority": receipt["proof_authority"] = True
    elif field == "count_bool": receipt["reference_count"] = True
    elif field == "count_zero": receipt["reference_count"] = 0
    elif field == "source": receipt["source_path"] = "../outside"
    elif field == "bytes": receipt["byte_savings"] = -1
    else: receipt["unknown"] = "data"
    with pytest.raises(codec.SemanticQueryReferenceError):
        codec.restore_semantic_query_reference(provider_prompt=result.provider_prompt, receipt=receipt)


def test_coherent_candidate_digest_mutation_still_cannot_change_original(producer):
    result = codec.project_semantic_query_reference(producer[2][native.TRANSPORT_SCHEMA].provider_prompt)
    value = json.loads(result.provider_prompt)
    value["transport"]["native_context"]["extra"] = "altered context"
    changed = wire(value)
    receipt = result.receipt
    receipt.update(candidate_view_sha256=sha(changed), candidate_view_bytes=len(changed.encode()),
                   byte_savings=receipt["native_transport_bytes"] - len(changed.encode()))
    reject("restored_binding_mismatch", lambda: codec.restore_semantic_query_reference(provider_prompt=changed, receipt=receipt))


@pytest.mark.parametrize("value,code", [(None,"invalid_text"), (b"{}","invalid_text"), ("[]","unsupported_transport"),
                                       ("null","unsupported_transport"), ('{"a":1,"a":2}',"duplicate_json_key"),
                                       ('{"a":NaN}',"nonfinite_json_number"), ('{"a":1e999}',"nonfinite_json_number"),
                                       ('{"a":}',"invalid_json"), ('{ "a":1}',"noncanonical_json"),
                                       ('{"a":"\\ud800"}',"noncanonical_json")])
def test_malformed_json_and_utf8(value, code):
    reject(code, lambda: codec.project_semantic_query_reference(value))


def test_input_depth_nodes_bytes_bounds(producer, monkeypatch):
    value = producer[2][native.TRANSPORT_SCHEMA].provider_prompt
    monkeypatch.setattr(codec, "MAX_BYTES", 20)
    # Explicit maximum captures the module's original default; use oversized text
    # against the public hard bound instead of relying on monkeypatched defaults.
    reject("input_bytes_exceeded", lambda: codec.project_semantic_query_reference("x" * 2000001))
    monkeypatch.setattr(codec, "MAX_BYTES", 2000000)
    monkeypatch.setattr(codec, "MAX_DEPTH", 3)
    reject("json_structure_exceeded", lambda: codec.project_semantic_query_reference(value))
    monkeypatch.setattr(codec, "MAX_DEPTH", 44)
    monkeypatch.setattr(codec, "MAX_NODES", 3)
    reject("json_structure_exceeded", lambda: codec.project_semantic_query_reference(value))


def test_inline_hash_and_byte_identity_not_merely_matching_query(producer):
    value = json.loads(producer[2][native.TRANSPORT_SCHEMA].provider_prompt)
    value["translated_semantic"]["manifest"]["instruction.txt"]["sha256"] = "0" * 64
    reject("inline_source_hash_mismatch", lambda: codec.project_semantic_query_reference(wire(value)))


def test_unknown_transport_and_unknown_top_level_fields(producer):
    value = json.loads(producer[2][native.TRANSPORT_SCHEMA].provider_prompt)
    value["extra"] = "unexpected"
    reject("invalid_transport", lambda: codec.project_semantic_query_reference(wire(value)))
    del value["extra"]
    value["schema"] = "supervisor-unreviewed-transport@1"
    reject("unsupported_transport", lambda: codec.project_semantic_query_reference(wire(value)))
