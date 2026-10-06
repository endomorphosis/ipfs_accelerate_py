"""Real producer capsules cross the reversible router representation boundary."""
import copy
import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.context.context_compiler import (
    ContextCompiler, build_text_context_references, render_context_capsule,
)
from ipfs_accelerate_py.agent_supervisor.context.context_contracts import ContextBudget
from ipfs_accelerate_py.agent_supervisor.runtime.semantic_context_runtime import prepare_semantic_context
from ipfs_accelerate_py.agent_supervisor.runtime import semantic_router_translation as codec


@pytest.fixture
def native(tmp_path):
    repository = tmp_path / "repository"
    repository.mkdir()
    source = ("# Literal aliases s0 and $semantic_ref are not identifiers to translate.\n"
        "def lengthy_first_function(value):\n"
        "    '''Keep comment and literal spelling s0 exactly.'''\n"
        "    return value + 1\n\n"
        "def lengthy_second_function(value):\n"
        "    return lengthy_first_function(value)\n\n"
        "class FirstNamespace:\n"
        "    def shared_method_name(self):\n"
        "        return 's0'\n\n"
        "class SecondNamespace:\n"
        "    def shared_method_name(self):\n"
        "        return '$semantic_ref'\n")
    (repository / "mod.py").write_text(source)
    output = repository / ".runtime/semantic"
    prepared = prepare_semantic_context(repository=repository, paths=["mod.py"],
        required_raw_paths=["mod.py"], objective="Inspect both functions.", task_id="TASK-1", output=output)
    artifact = output / "worker-context.json"
    text = artifact.read_text()
    refs = build_text_context_references(text, reference_prefix="semantic-context", kind="semantic-context",
        path=artifact.relative_to(repository).as_posix(), repository_id="repo:test", tree_id="tree:test",
        required=True, chunk_bytes=1201)
    compiled = ContextCompiler(ContextBudget(max_input_tokens=32768, max_items=128,
        max_item_bytes=16384, max_serialized_bytes=262144)).compile(
        repository_id="repo:test", tree_id="tree:test", objective_id="TASK-1",
        objective_revision="sha256:task", policy_id="policy:test", policy_revision="sha256:policy",
        caller="supervisor:test", stage="implementation", goal={"id": "TASK-1"},
        authority={"mode": "candidate_only", "completion_authority": False},
        scope={"allowed_paths": ["mod.py"]}, acceptance={"criteria": ["pending native validation"]},
        evidence=refs)
    prompt = render_context_capsule(compiled.capsule) + "\nAuthorized guidance remains literal: s0.\n"
    return repository, artifact, prompt, source


def test_real_capsules_round_trip_source_literals_and_immutable_core(native):
    root, artifact, prompt, source = native
    encoded = codec.encode_semantic_router_prompt(prompt=prompt, repository=root)
    assert codec.restore_semantic_router_prompt(provider_prompt=encoded.provider_prompt,
        table=encoded.table, repository=root) == prompt
    table = codec.SemanticTranslationTable.from_dict(encoded.table.to_dict())
    assert table == encoded.table
    original, _ = json.JSONDecoder().raw_decode(prompt)
    transport = json.loads(encoded.provider_prompt)
    semantic = transport["translated_semantic"]
    assert semantic["raw_sources"]["mod.py"] == source
    assert transport["native_context"]["authority"] == original["authority"]
    assert transport["native_context"]["scope"] == original["scope"]
    assert transport["native_suffix"] == "\nAuthorized guidance remains literal: s0.\n"
    assert semantic["capsules"][0]["signature"] == json.loads(artifact.read_text())["capsules"][0]["signature"]
    assert encoded.receipt["identifier_mappings"] > 0
    assert encoded.receipt["identifier_occurrences"] > encoded.receipt["identifier_mappings"]
    assert encoded.receipt["freshness_checked"] is True
    assert encoded.receipt["execution_authority"] is False
    assert encoded.receipt["completion_authority"] is False
    assert encoded.receipt["semantic_equivalence_claimed"] is False
    entries = table.to_dict()["entries"]
    assert len({row["source_symbol_id"] for row in entries}) == len(entries)
    assert len({row["target_symbol_ids"][0] for row in entries}) == len(entries)
    # Equal source spellings in separate namespaces retain distinct producer
    # identities; neither executable spelling nor its literals are renamed.
    assert source.count("def shared_method_name") == 2
    assert len(table.to_dict()["symbol_ids"]) >= 6


def test_plain_response_is_literal_even_when_alias_text_is_present(native):
    root, _, prompt, _ = native
    encoded = codec.encode_semantic_router_prompt(prompt=prompt, repository=root)
    text = '  Retained s0 and {"$semantic_ref":"s999"} as literal prose.\n'
    decoded = codec.decode_semantic_router_response(response=text, encoded=encoded, repository=root)
    assert decoded.text == text
    assert decoded.receipt["structured_translation"] is False


def test_compact_dictionary_round_trip_keeps_native_table_and_literal_contract(native):
    root, artifact, prompt, source = native
    default = codec.encode_semantic_router_prompt(prompt=prompt, repository=root)
    legacy = codec.encode_semantic_router_prompt(prompt=prompt, repository=root,
        transport_schema=codec.TRANSPORT_SCHEMA)
    compact = codec.encode_semantic_router_prompt(prompt=prompt, repository=root,
        transport_schema=codec.COMPACT_TRANSPORT_SCHEMA)
    assert default == legacy
    assert "transport_schema" not in legacy.receipt
    assert legacy.receipt["schema"] == "supervisor-semantic-router-encoding@1"
    assert compact.table == legacy.table
    transport = json.loads(compact.provider_prompt)
    original = json.loads(legacy.provider_prompt)
    assert "translation_table" not in transport
    assert compact.table.to_dict()["entries"]
    for key in ("native_context", "translated_semantic", "native_suffix", "translation_cid",
                "execution_authority", "completion_authority"):
        assert transport[key] == original[key]
    for key in ("task_id", "scope_cid", "semantic_root_cid"):
        assert transport[key] == compact.table.to_dict()[key]
    assert transport["translated_semantic"]["raw_sources"]["mod.py"] == source
    assert transport["translated_semantic"]["capsules"][0]["signature"] == json.loads(artifact.read_text())["capsules"][0]["signature"]
    assert compact.receipt["schema"] == "supervisor-semantic-router-encoding@2"
    assert compact.receipt["transport_schema"] == codec.COMPACT_TRANSPORT_SCHEMA
    assert codec.restore_semantic_router_prompt(provider_prompt=compact.provider_prompt,
        table=compact.table, repository=root) == prompt
    replay = codec.replay_semantic_router_prompt_for_audit(prompt=prompt, repository=root,
        transport_schema=codec.COMPACT_TRANSPORT_SCHEMA)
    assert replay.provider_prompt == compact.provider_prompt and replay.table == compact.table
    assert replay.freshness_checked is False


@pytest.mark.parametrize("schema", ["supervisor-semantic-router-input@3", "", None, {}, []])
def test_unsupported_transport_schema_refuses_live_and_historical_encoding(native, schema):
    root, _, prompt, _ = native
    for encode in (codec.encode_semantic_router_prompt, codec.replay_semantic_router_prompt_for_audit):
        with pytest.raises(codec.SemanticTranslationError, match="unsupported semantic transport schema"):
            encode(prompt=prompt, repository=root, transport_schema=schema)


def test_compact_transport_rejects_structured_marker_collisions_but_keeps_literal_text(native):
    root, _, prompt, _ = native
    wire, end = json.JSONDecoder().raw_decode(prompt)
    wire["goal"]["literal_marker_text"] = '{"$semantic_ref":"s0"}'
    literal = codec._json(wire) + prompt[end:]
    compact = codec.encode_semantic_router_prompt(prompt=literal, repository=root,
        transport_schema=codec.COMPACT_TRANSPORT_SCHEMA)
    assert codec.restore_semantic_router_prompt(provider_prompt=compact.provider_prompt,
        table=compact.table, repository=root) == literal
    wire["goal"]["literal_marker_text"] = {codec.REF_KEY: "s0"}
    collision = codec._json(wire) + prompt[end:]
    # The legacy transport remains unchanged; only the opt-in protocol reserves
    # actual structured marker keys outside its registered identifier paths.
    codec.encode_semantic_router_prompt(prompt=collision, repository=root)
    with pytest.raises(codec.SemanticTranslationError, match="reserved semantic reference marker collision"):
        codec.encode_semantic_router_prompt(prompt=collision, repository=root,
            transport_schema=codec.COMPACT_TRANSPORT_SCHEMA)


def _reply(encoded):
    table = encoded.table.to_dict()
    symbol = table["symbol_ids"][0]
    matches = [row for row in table["entries"] if row["source_symbol_id"] == symbol]
    assert matches
    alias = matches[0]["target_symbol_ids"][0]
    body = {"output_class": "PATCH_SKETCH", "structured_payload": {
        "files": ["mod.py"], "symbol_ids": [{codec.REF_KEY: alias}],
        "operations": ["replace_function"], "maximum_changed_lines": 20,
        "validation_ids": ["pytest:focused"]},
        "confidence_or_score": 900000, "calibration_group": "patch:python:R2:fixture",
        "abstained": False, "reason_codes": [], "evidence_references": [], "candidate_only": True}
    from ipfs_accelerate_py.agent_supervisor.residual_intelligence.contracts import ResidualTaskFamily
    return {"schema": codec.REPLY_SCHEMA, "translation_cid": encoded.table.translation_cid,
        "task_id": table["task_id"], "scope_cid": table["scope_cid"],
        "semantic_root_cid": table["semantic_root_cid"],
        "task_family": ResidualTaskFamily.PATCH_SKETCH_GENERATION.value, "response": body}, symbol


def test_structured_symbol_reply_enters_existing_native_candidate_grammar(native):
    root, _, prompt, _ = native
    encoded = codec.encode_semantic_router_prompt(prompt=prompt, repository=root)
    reply, symbol = _reply(encoded)
    decoded = codec.decode_semantic_router_response(response=json.dumps(reply), encoded=encoded, repository=root)
    body = json.loads(decoded.text)
    assert body["structured_payload"]["symbol_ids"] == [symbol]
    assert body["candidate_only"] is True
    assert decoded.receipt["native_grammar_validated"] is True
    assert decoded.receipt["completion_authority"] is False


def test_compact_symbol_reply_uses_trusted_dictionary_and_native_grammar(native):
    root, _, prompt, _ = native
    encoded = codec.encode_semantic_router_prompt(prompt=prompt, repository=root,
        transport_schema=codec.COMPACT_TRANSPORT_SCHEMA)
    reply, symbol = _reply(encoded)
    decoded = codec.decode_semantic_router_response(response=json.dumps(reply), encoded=encoded, repository=root)
    assert json.loads(decoded.text)["structured_payload"]["symbol_ids"] == [symbol]
    assert decoded.receipt["native_grammar_validated"] is True
    assert decoded.receipt["completion_authority"] is False
    literal = '  Literal {"$semantic_ref":"s999"} and s0 stay unchanged.\n'
    assert codec.decode_semantic_router_response(response=literal, encoded=encoded, repository=root).text == literal


@pytest.mark.parametrize("mutation", ["task_id", "scope_cid", "semantic_root_cid", "translation_cid",
    "unknown_alias", "map", "literal", "reserved_marker", "unsupported_schema", "table"])
def test_compact_transport_rejects_foreign_bindings_and_tampering(native, mutation):
    root, _, prompt, _ = native
    encoded = codec.encode_semantic_router_prompt(prompt=prompt, repository=root,
        transport_schema=codec.COMPACT_TRANSPORT_SCHEMA)
    wire = json.loads(encoded.provider_prompt)
    table = encoded.table
    if mutation in {"task_id", "scope_cid", "semantic_root_cid", "translation_cid"}:
        wire[mutation] = "foreign"
    elif mutation == "unknown_alias":
        codec._set(wire["translated_semantic"], table.to_dict()["replacement_paths"][0],
            {codec.REF_KEY: "s99999"})
    elif mutation == "map":
        wire["translation_table"] = {"s0": "foreign"}
    elif mutation == "literal":
        wire["translated_semantic"]["raw_sources"]["mod.py"] += "# changed\n"
    elif mutation == "reserved_marker":
        wire["native_context"]["goal"]["collision"] = {codec.REF_KEY: "s0"}
    elif mutation == "unsupported_schema":
        wire["schema"] = "supervisor-semantic-router-input@3"
    else:
        payload = json.loads(table.payload_json)
        payload["entries"][0]["source_symbol_id"] = "foreign"
        table = codec.SemanticTranslationTable(codec._json(payload), codec.cid_for_payload(payload))
    with pytest.raises(codec.SemanticTranslationError):
        codec.restore_semantic_router_prompt(provider_prompt=codec._json(wire), table=table, repository=root)


@pytest.mark.parametrize("mutation", ["task_id", "scope_cid", "semantic_root_cid", "translation_cid",
    "unknown_alias", "authority"])
def test_compact_structured_reply_rejects_foreign_binding_and_unknown_alias(native, mutation):
    root, _, prompt, _ = native
    encoded = codec.encode_semantic_router_prompt(prompt=prompt, repository=root,
        transport_schema=codec.COMPACT_TRANSPORT_SCHEMA)
    reply, _ = _reply(encoded)
    if mutation == "unknown_alias":
        reply["response"]["structured_payload"]["symbol_ids"] = [{codec.REF_KEY: "s99999"}]
    elif mutation == "authority":
        reply["response"]["candidate_only"] = False
    else:
        reply[mutation] = "foreign"
    with pytest.raises(codec.SemanticTranslationError):
        codec.decode_semantic_router_response(response=json.dumps(reply), encoded=encoded, repository=root)


def test_compact_stale_source_blocks_operational_use_but_historical_replay_matches(native):
    root, _, prompt, source = native
    encoded = codec.encode_semantic_router_prompt(prompt=prompt, repository=root,
        transport_schema=codec.COMPACT_TRANSPORT_SCHEMA)
    reply, _ = _reply(encoded)
    (root / "mod.py").write_text(source + "# changed after dispatch\n")
    with pytest.raises(codec.SemanticTranslationError, match="stale"):
        codec.restore_semantic_router_prompt(provider_prompt=encoded.provider_prompt,
            table=encoded.table, repository=root)
    with pytest.raises(codec.SemanticTranslationError, match="stale"):
        codec.decode_semantic_router_response(response=json.dumps(reply), encoded=encoded, repository=root)
    replay = codec.replay_semantic_router_prompt_for_audit(prompt=prompt, repository=root,
        transport_schema=codec.COMPACT_TRANSPORT_SCHEMA)
    assert replay.provider_prompt == encoded.provider_prompt and replay.table == encoded.table
    assert replay.receipt["freshness_checked"] is False
    with pytest.raises(codec.SemanticTranslationError, match="historical"):
        codec.decode_semantic_router_response(response=json.dumps(reply), encoded=replay, repository=root)


def test_duplicate_reserved_reply_keys_are_not_treated_as_plain_text(native):
    root, _, prompt, _ = native
    encoded = codec.encode_semantic_router_prompt(prompt=prompt, repository=root)
    reply, _ = _reply(encoded)
    text = json.dumps(reply)[:-1] + ',"schema":' + json.dumps(codec.REPLY_SCHEMA) + '}'
    with pytest.raises(codec.SemanticTranslationError, match="malformed reserved"):
        codec.decode_semantic_router_response(response=text, encoded=encoded, repository=root)


def test_mapping_alias_to_non_symbol_cannot_become_a_symbol_proposal(native):
    root, _, prompt, _ = native
    encoded = codec.encode_semantic_router_prompt(prompt=prompt, repository=root)
    reply, _ = _reply(encoded)
    table = encoded.table.to_dict()
    non_symbol = next(row for row in table["entries"] if row["source_symbol_id"] not in table["symbol_ids"])
    reply["response"]["structured_payload"]["symbol_ids"] = [{codec.REF_KEY: non_symbol["target_symbol_ids"][0]}]
    with pytest.raises(codec.SemanticTranslationError, match="unknown semantic response identifier"):
        codec.decode_semantic_router_response(response=json.dumps(reply), encoded=encoded, repository=root)


@pytest.mark.parametrize("mutation", ["unknown_alias", "foreign_task", "foreign_root", "authority", "untyped_authority"])
def test_structured_alias_or_authority_tampering_is_rejected(native, mutation):
    root, _, prompt, _ = native
    encoded = codec.encode_semantic_router_prompt(prompt=prompt, repository=root)
    reply, _ = _reply(encoded)
    if mutation == "unknown_alias":
        reply["response"]["structured_payload"]["symbol_ids"] = [{codec.REF_KEY: "s99999"}]
    elif mutation == "foreign_task":
        reply["task_id"] = "TASK-2"
    elif mutation == "foreign_root":
        reply["semantic_root_cid"] = "foreign"
    elif mutation == "authority":
        reply["response"]["candidate_only"] = False
    else:
        reply["response"]["structured_payload"]["completed"] = True
    with pytest.raises(ValueError):
        codec.decode_semantic_router_response(response=json.dumps(reply), encoded=encoded, repository=root)


def test_stale_sources_refuse_dispatch_and_structured_decode_but_allow_historical_audit(native):
    root, _, prompt, source = native
    encoded = codec.encode_semantic_router_prompt(prompt=prompt, repository=root)
    reply, _ = _reply(encoded)
    (root / "mod.py").write_text(source + "# newly edited candidate\n")
    with pytest.raises(ValueError, match="stale"):
        codec.encode_semantic_router_prompt(prompt=prompt, repository=root)
    with pytest.raises(ValueError, match="stale"):
        codec.decode_semantic_router_response(response=json.dumps(reply), encoded=encoded, repository=root)
    replay = codec.replay_semantic_router_prompt_for_audit(prompt=prompt, repository=root)
    assert replay.provider_prompt == encoded.provider_prompt
    assert replay.table == encoded.table
    assert replay.freshness_checked is replay.receipt["freshness_checked"] is False
    with pytest.raises(ValueError, match="historical"):
        codec.decode_semantic_router_response(response=json.dumps(reply), encoded=replay, repository=root)


@pytest.mark.parametrize("mutation", ["map", "literal", "core", "unknown", "table"])
def test_transport_round_trip_rejects_tampering(native, mutation):
    root, _, prompt, _ = native
    encoded = codec.encode_semantic_router_prompt(prompt=prompt, repository=root)
    payload, table = json.loads(encoded.provider_prompt), encoded.table
    if mutation == "map":
        payload["translation_table"]["s0"] = payload["translation_table"].get("s1", "foreign")
    elif mutation == "literal":
        payload["translated_semantic"]["raw_sources"]["mod.py"] += "# changed\n"
    elif mutation == "core":
        payload["native_context"]["authority"]["completion_authority"] = True
    elif mutation == "unknown":
        path = table.to_dict()["replacement_paths"][0]
        codec._set(payload["translated_semantic"], path, {codec.REF_KEY: "unknown"})
    else:
        record = table.to_dict()
        record["entries"][0]["target_symbol_ids"] = ["foreign"]
        with pytest.raises(ValueError):
            codec.SemanticTranslationTable.from_dict(record)
        return
    with pytest.raises(ValueError):
        codec.restore_semantic_router_prompt(provider_prompt=codec._json(payload), table=table, repository=root)


def test_tampered_producer_block_and_symlink_artifact_are_refused(native):
    root, artifact, prompt, _ = native
    data = json.loads(artifact.read_text())
    block = artifact.parent / "blocks" / data["semantic_root_cid"]
    raw = block.read_bytes()
    block.write_bytes(raw + b" ")
    with pytest.raises(ValueError):
        codec.encode_semantic_router_prompt(prompt=prompt, repository=root)
    block.write_bytes(raw)
    saved = artifact.with_suffix(".original")
    artifact.rename(saved)
    artifact.symlink_to(saved)
    with pytest.raises((ValueError, OSError)):
        codec.encode_semantic_router_prompt(prompt=prompt, repository=root)


@pytest.mark.parametrize("mutation", ["scope", "omitted_source", "foreign_manifest_source", "unknown_raw_source"])
def test_rebound_prompt_cannot_narrow_or_substitute_producer_source_inventory(native, mutation):
    root, artifact, prompt, source = native
    (root / "dependency.py").write_text("def dependency_function(value):\n    return value * 2\n")
    output = root / ".runtime/full-inventory"
    prepare_semantic_context(repository=root, paths=["mod.py", "dependency.py"],
        required_raw_paths=["mod.py", "dependency.py"], objective="Inspect both functions.",
        task_id="TASK-1", output=output)
    artifact = output / "worker-context.json"
    payload = json.loads(artifact.read_text())
    if mutation == "scope":
        payload["scope_cid"] = codec.cid_for_payload({"foreign": "scope"})
    elif mutation == "omitted_source":
        del payload["manifest"]["dependency.py"]
        del payload["raw_sources"]["dependency.py"]
        payload["scope_cid"] = codec.cid_for_payload({"schema": "supervisor-source-scope@1", "sources": payload["manifest"]})
        (root / "dependency.py").write_text("# changed outside the narrowed nominated manifest\n")
    elif mutation == "foreign_manifest_source":
        payload["manifest"]["dependency.py"] = dict(payload["manifest"]["mod.py"])
        payload["scope_cid"] = codec.cid_for_payload({"schema": "supervisor-source-scope@1", "sources": payload["manifest"]})
    else:
        payload["raw_sources"]["not-in-producer.py"] = "unverified source text"
    text = codec._json(payload)
    artifact.write_text(text)
    # Even a new, internally consistent native nomination cannot manufacture
    # equality to the independently verified producer snapshot inventory.
    wire, end = json.JSONDecoder().raw_decode(prompt)
    refs = build_text_context_references(text, reference_prefix="semantic-context", kind="semantic-context",
        path=artifact.relative_to(root).as_posix(), repository_id="repo:test", tree_id="tree:test",
        required=True, chunk_bytes=1201)
    wire["evidence"] = [reference.to_dict() for reference in refs]
    rebound = codec._json(wire) + prompt[end:]
    expected = "scope identity" if mutation == "scope" else "producer.*inventory"
    with pytest.raises(codec.SemanticTranslationError, match=expected):
        codec.encode_semantic_router_prompt(prompt=rebound, repository=root)
    with pytest.raises(codec.SemanticTranslationError, match=expected):
        codec.replay_semantic_router_prompt_for_audit(prompt=rebound, repository=root)


def test_mixed_public_source_inventory_with_selected_capsules_round_trips(native):
    root, _, prompt, source = native
    (root / "README.md").write_text("# Public instruction\nInspect identifiers; preserve literal s0.\n")
    (root / "validation.py").write_text("from mod import lengthy_first_function\nassert lengthy_first_function(2) == 3\n")
    paths = ["mod.py", "README.md", "validation.py"]
    output = root / ".runtime/mixed-context"
    prepared = prepare_semantic_context(repository=root, paths=paths, required_raw_paths=paths,
        objective="Inspect both functions.", task_id="TASK-1", output=output,
        max_symbols=16, worker_query="lengthy_first_function", worker_capsule_limit=1,
        worker_max_bytes=32768)
    artifact = output / "worker-context.json"
    text = artifact.read_text()
    wire, end = json.JSONDecoder().raw_decode(prompt)
    wire["evidence"] = [reference.to_dict() for reference in build_text_context_references(
        text, reference_prefix="semantic-context", kind="semantic-context",
        path=artifact.relative_to(root).as_posix(), repository_id="repo:test", tree_id="tree:test",
        required=True, chunk_bytes=1201)]
    mixed_prompt = codec._json(wire) + prompt[end:]
    encoded = codec.encode_semantic_router_prompt(prompt=mixed_prompt, repository=root)
    assert set(encoded.table.to_dict()["source_manifest"]) == set(paths)
    assert prepared["worker_capsules"] <= 1
    assert prepared["capsules"] > prepared["worker_capsules"]
    translated = json.loads(encoded.provider_prompt)["translated_semantic"]
    assert translated["raw_sources"] == {path: (root / path).read_text() for path in paths}
    assert codec.restore_semantic_router_prompt(provider_prompt=encoded.provider_prompt,
        table=encoded.table, repository=root) == mixed_prompt


def _program_prompt(native, *, program):
    root, _, prompt, _ = native
    (root / "harness.py").write_text("def excluded_harness_symbol(): return unresolved_dependency()\n")
    output = root / ".runtime/program-context"
    prepare_semantic_context(repository=root, paths=["mod.py", "harness.py"],
        required_raw_paths=["harness.py"], program_paths=program,
        objective="Inspect the exact program.", task_id="TASK-1", output=output)
    artifact = output / "worker-context.json"
    wire, end = json.JSONDecoder().raw_decode(prompt)
    wire["evidence"] = [reference.to_dict() for reference in build_text_context_references(
        artifact.read_text(), reference_prefix="semantic-context", kind="semantic-context",
        path=artifact.relative_to(root).as_posix(), repository_id="repo:test", tree_id="tree:test",
        required=True, chunk_bytes=1201)]
    return root, artifact, codec._json(wire) + prompt[end:]


@pytest.mark.parametrize("program", [[], ["mod.py"]])
def test_explicit_program_translation_round_trips_real_subset_and_support(native, program):
    root, artifact, prompt = _program_prompt(native, program=program)
    encoded = codec.encode_semantic_router_prompt(prompt=prompt, repository=root)
    assert codec.restore_semantic_router_prompt(provider_prompt=encoded.provider_prompt,
        table=encoded.table, repository=root) == prompt
    translated = json.loads(encoded.provider_prompt)["translated_semantic"]
    assert translated["schema"] == "supervisor-semantic-worker-context@2"
    assert translated["program_paths"] == program
    assert translated["raw_sources"]["harness.py"] == (root / "harness.py").read_text()
    assert set(encoded.table.to_dict()["source_manifest"]) == {"harness.py", "mod.py"}
    assert "excluded_harness_symbol" not in json.dumps(translated["capsules"])
    if not program:
        assert encoded.table.to_dict()["symbol_ids"] == []
        assert translated["capsules"] == translated["admissions"] == []
    else:
        reply, symbol = _reply(encoded)
        decoded = codec.decode_semantic_router_response(response=json.dumps(reply), encoded=encoded, repository=root)
        assert json.loads(decoded.text)["structured_payload"]["symbol_ids"] == [symbol]
    assert encoded.receipt["completion_authority"] is False
    (root / "harness.py").write_text("# support changed after nomination\n")
    with pytest.raises(ValueError, match="stale"):
        codec.encode_semantic_router_prompt(prompt=prompt, repository=root)
    replay = codec.replay_semantic_router_prompt_for_audit(prompt=prompt, repository=root)
    assert replay.provider_prompt == encoded.provider_prompt
    assert replay.receipt["freshness_checked"] is False


@pytest.mark.parametrize("mutation", ["subset", "dropped_capsule"])
def test_explicit_program_historical_replay_rejects_rehashed_nominations(native, mutation):
    root, artifact, prompt = _program_prompt(native, program=["mod.py"])
    payload = json.loads(artifact.read_text())
    if mutation == "subset":
        payload["program_paths"] = []
        payload["scope_cid"] = codec.cid_for_payload({"schema": "supervisor-source-scope@2",
            "sources": payload["manifest"], "program_paths": []})
        payload["reconstruction"].update(scope_cid=payload["scope_cid"], program_paths=[],
            doctor_source_paths=[], program_source_count=0, support_source_count=2)
        payload["raw_sources"]["mod.py"] = (root / "mod.py").read_text()
    else:
        payload["capsules"].pop()
        payload["admissions"].pop()
    artifact.write_text(codec._json(payload))
    wire, end = json.JSONDecoder().raw_decode(prompt)
    wire["evidence"] = [reference.to_dict() for reference in build_text_context_references(
        artifact.read_text(), reference_prefix="semantic-context", kind="semantic-context",
        path=artifact.relative_to(root).as_posix(), repository_id="repo:test", tree_id="tree:test",
        required=True, chunk_bytes=1201)]
    rebound = codec._json(wire) + prompt[end:]
    for translate in (codec.encode_semantic_router_prompt, codec.replay_semantic_router_prompt_for_audit):
        with pytest.raises(codec.SemanticTranslationError):
            translate(prompt=rebound, repository=root)
