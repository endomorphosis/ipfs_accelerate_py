"""Real producer legacy/v2 identifier transport; literals are never source edits."""

import copy
import json

import pytest

from ipfs_accelerate_py.agent_supervisor.context.context_compiler import (
    ContextCompiler,
    build_text_context_references,
    render_context_capsule,
)
from ipfs_accelerate_py.agent_supervisor.context.context_contracts import ContextBudget
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.contracts import ResidualTaskFamily
from ipfs_accelerate_py.agent_supervisor.runtime import semantic_router_translation as codec
from ipfs_accelerate_py.agent_supervisor.runtime.semantic_context_projection import (
    SemanticProjectionBindings,
    build_semantic_context_projection,
    semantic_context_projection_references,
)
from ipfs_accelerate_py.agent_supervisor.runtime.semantic_context_runtime import (
    prepare_semantic_context,
    resolve_semantic_worker_context,
)

SOURCE = (
    "# CID-looking text, aliases s0, and $semantic_ref stay literal.\n"
    "def add_offset(value):\n"
    "    '''Literal b" + "a" * 58 + " and s0.'''\n"
    "    return value + 1\n\n"
    "def consume(value):\n"
    "    return add_offset(value)\n"
)


def compile_prompt(root, artifact, text, *, references=None):
    if references is None:
        references = build_text_context_references(
            text,
            reference_prefix="semantic-context",
            kind="semantic-context",
            path=artifact.relative_to(root).as_posix(),
            repository_id="repo:slots",
            tree_id="tree:slots",
            required=True,
            chunk_bytes=1201,
        )
    result = ContextCompiler(
        ContextBudget(
            max_input_tokens=32768, max_items=128, max_item_bytes=16384, max_serialized_bytes=262144
        )
    ).compile(
        repository_id="repo:slots",
        tree_id="tree:slots",
        objective_id="TASK-SLOTS",
        objective_revision="sha256:task",
        policy_id="policy:slots",
        policy_revision="sha256:policy",
        caller="supervisor:slots",
        stage="implementation",
        goal={"id": "TASK-SLOTS"},
        authority={"mode": "candidate_only", "completion_authority": False},
        scope={"allowed_paths": ["mod.py"]},
        acceptance={"criteria": ["native validation pending"]},
        evidence=references,
    )
    return render_context_capsule(result.capsule) + "\nLiteral suffix s0 $semantic_ref.\n"


@pytest.fixture(scope="module")
def producer(tmp_path_factory):
    root = tmp_path_factory.mktemp("slot-producer") / "repository"
    root.mkdir()
    (root / "mod.py").write_text(SOURCE)
    (root / "dependency.py").write_text("def dependency(value):\n    return value * 2\n")
    prepared = prepare_semantic_context(
        repository=root,
        paths=["mod.py", "dependency.py"],
        required_raw_paths=["mod.py"],
        objective="Inspect add_offset.",
        task_id="TASK-SLOTS",
        output=root / ".runtime/semantic",
        worker_query="add_offset",
        worker_capsule_limit=4,
        worker_max_bytes=65536,
    )
    artifact = root / ".runtime/semantic/worker-context.json"
    text = artifact.read_text()
    return root, artifact, compile_prompt(root, artifact, text), json.loads(text), prepared


@pytest.fixture
def native(producer):
    root, artifact, *_ = producer
    original = {
        path: path.read_bytes() for path in (root / "mod.py", root / "dependency.py", artifact)
    }
    yield producer
    for path, raw in original.items():
        if path.read_bytes() != raw:
            path.write_bytes(raw)


def encode(native, revision=codec.SLOT_POLICY_V2):
    root, _, prompt, _, _ = native
    return codec.encode_semantic_router_prompt(
        prompt=prompt, repository=root, slot_policy_revision=revision
    )


def role(native):
    root, artifact, _, payload, _ = native
    original = artifact.read_text()
    bindings = SemanticProjectionBindings(
        artifact=artifact.relative_to(root).as_posix(),
        expected_sha256=codec._sha(original),
        task_id="TASK-SLOTS",
        semantic_root_cid=payload["semantic_root_cid"],
        scope_cid=payload["scope_cid"],
        repository_id="repo:slots",
        tree_id="tree:slots",
    )
    view = build_semantic_context_projection(original_text=original, bindings=bindings)
    references = semantic_context_projection_references(
        view=view, original_text=original, bindings=bindings, chunk_bytes=1201
    )
    return view, compile_prompt(root, artifact, codec._json(view), references=references), bindings


def reply(encoded):
    table = encoded.table.to_dict()
    symbol = table["symbol_ids"][0]
    alias = next(
        row["target_symbol_ids"][0] for row in table["entries"] if row["source_symbol_id"] == symbol
    )
    return {
        "schema": codec.REPLY_SCHEMA,
        "translation_cid": table["translation_cid"],
        "task_id": table["task_id"],
        "semantic_root_cid": table["semantic_root_cid"],
        "scope_cid": table["scope_cid"],
        "task_family": ResidualTaskFamily.PATCH_SKETCH_GENERATION.value,
        "response": {
            "output_class": "PATCH_SKETCH",
            "candidate_only": True,
            "confidence_or_score": 900000,
            "calibration_group": "patch:python:R2:fixture",
            "abstained": False,
            "reason_codes": [],
            "evidence_references": [],
            "structured_payload": {
                "files": ["mod.py"],
                "symbol_ids": [{codec.REF_KEY: alias}],
                "operations": ["replace_function"],
                "maximum_changed_lines": 20,
                "validation_ids": ["pytest:focused"],
            },
        },
    }


def test_explicit_v2_adds_closed_slots_and_round_trips_all_native_bytes(native):
    root, artifact, prompt, payload, _ = native
    legacy, v2 = encode(native, codec.SLOT_POLICY_LEGACY), encode(native)
    old_paths = {tuple(path) for path in legacy.table.to_dict()["replacement_paths"]}
    new_paths = {tuple(path) for path in v2.table.to_dict()["replacement_paths"]}
    assert old_paths < new_paths
    for path in [
        ("scope_cid",),
        ("semantic_root_cid",),
        ("manifest", "mod.py", "source_cid"),
        ("pack", "pack", "target_source_cid"),
        ("pack", "pack", "surrounding_source_cid"),
        ("pack", "pack", "test_source_cid"),
        ("reconstruction", "scope_cid"),
        ("reconstruction", "semantic_root_cid"),
        ("worker_projection", "selected_symbols", 0, "stable_symbol_id"),
        ("worker_projection", "raw_source_fetch_required", "dependency.py", "source_cid"),
    ]:
        assert path in new_paths, path
    transport = json.loads(v2.provider_prompt)
    translated = transport["translated_semantic"]
    assert translated["raw_sources"] == payload["raw_sources"]
    for before, after in zip(payload["capsules"], translated["capsules"], strict=True):
        assert before["metadata"] == after["metadata"]
        assert before["signature"] == after["signature"]
    assert translated["pack"]["decisions"] == payload["pack"]["decisions"]
    assert translated["worker_projection"]["query"] == payload["worker_projection"]["query"]
    assert (
        translated["worker_projection"]["raw_required_caveat"]
        == payload["worker_projection"]["raw_required_caveat"]
    )
    assert transport["native_suffix"] == "\nLiteral suffix s0 $semantic_ref.\n"
    assert (
        codec.restore_semantic_router_prompt(
            provider_prompt=v2.provider_prompt, table=v2.table, repository=root
        )
        == prompt
    )
    assert artifact.read_text() == codec._json(payload)
    assert (root / "mod.py").read_text() == SOURCE
    assert v2.table.slot_policy_revision == codec.SLOT_POLICY_V2
    assert v2.table.to_dict()["schema"] == codec.TABLE_SCHEMA_V2
    assert v2.receipt["slot_policy_revision"] == codec.SLOT_POLICY_V2
    assert encode(native) == v2


def test_default_retains_legacy_schema_fields_identity_and_serialized_replay(native):
    root, _, prompt, _, _ = native
    default = codec.encode_semantic_router_prompt(prompt=prompt, repository=root)
    legacy = encode(native, codec.SLOT_POLICY_LEGACY)
    assert default == legacy
    record = legacy.table.to_dict()
    assert set(record) == codec.TABLE_FIELDS | {"translation_cid"}
    assert record["schema"] == codec.TABLE_SCHEMA and "slot_policy_revision" not in record
    assert "slot_policy_revision" not in legacy.receipt
    table = codec.SemanticTranslationTable.from_dict(json.loads(codec._json(record)))
    assert table == legacy.table and table.slot_policy_revision == codec.SLOT_POLICY_LEGACY
    replayed = codec.replay_semantic_router_prompt_for_audit(prompt=prompt, repository=root)
    assert replayed.table == table and replayed.provider_prompt == legacy.provider_prompt
    assert (
        codec.restore_semantic_router_prompt(
            provider_prompt=legacy.provider_prompt, table=table, repository=root
        )
        == prompt
    )


@pytest.mark.parametrize("revision", [None, False, 2, "", "closed-typed-identifiers@3", "legacy@2"])
def test_unsupported_policy_refuses_before_producer_read(native, revision, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("unsupported policy reached source or producer read")

    monkeypatch.setattr(codec, "_read", unexpected)
    with pytest.raises(codec.SemanticTranslationError, match="unsupported.*policy"):
        encode(native, revision)
    with pytest.raises(codec.SemanticTranslationError, match="unsupported.*policy"):
        codec.replay_semantic_router_prompt_for_audit(
            prompt=native[2], repository=native[0], slot_policy_revision=revision
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "unknown_schema",
        "unknown_revision",
        "missing_revision",
        "legacy_with_revision",
        "v2_with_legacy_revision",
        "extra_field",
    ],
)
def test_table_revision_is_explicit_and_closed(native, mutation):
    record = encode(native).table.to_dict()
    if mutation == "unknown_schema":
        record["schema"] = "supervisor-semantic-translation-table@3"
    elif mutation == "unknown_revision":
        record["slot_policy_revision"] = "closed-typed-identifiers@3"
    elif mutation == "missing_revision":
        del record["slot_policy_revision"]
    elif mutation == "legacy_with_revision":
        record["schema"] = codec.TABLE_SCHEMA
    elif mutation == "v2_with_legacy_revision":
        record["slot_policy_revision"] = codec.SLOT_POLICY_LEGACY
    else:
        record["future_authority"] = False
    record["translation_cid"] = codec.cid_for_payload(
        {k: v for k, v in record.items() if k != "translation_cid"}
    )
    with pytest.raises(codec.SemanticTranslationError):
        codec.SemanticTranslationTable.from_dict(record)


@pytest.mark.parametrize("policy", [codec.SLOT_POLICY_LEGACY, codec.SLOT_POLICY_V2])
def test_current_structured_decode_uses_the_table_policy(native, policy):
    encoded = encode(native, policy)
    response = reply(encoded)
    decoded = codec.decode_semantic_router_response(
        response=codec._json(response), encoded=encoded, repository=native[0]
    )
    assert json.loads(decoded.text)["structured_payload"]["symbol_ids"] == [
        encoded.table.to_dict()["symbol_ids"][0]
    ]
    assert decoded.receipt["candidate_only"] is True
    literal = 'Literal s0 and {"$semantic_ref":"s999"}.'
    assert (
        codec.decode_semantic_router_response(
            response=literal, encoded=encoded, repository=native[0]
        ).text
        == literal
    )


@pytest.mark.parametrize("policy", [codec.SLOT_POLICY_LEGACY, codec.SLOT_POLICY_V2])
def test_stale_current_sources_refuse_but_historical_policy_replays_exactly(native, policy):
    root, _, prompt, _, _ = native
    encoded = encode(native, policy)
    (root / "mod.py").write_text(SOURCE + "# fresh candidate\n")
    with pytest.raises(ValueError, match="stale"):
        encode(native, policy)
    with pytest.raises(ValueError, match="stale"):
        codec.decode_semantic_router_response(
            response=codec._json(reply(encoded)), encoded=encoded, repository=root
        )
    historical = codec.replay_semantic_router_prompt_for_audit(
        prompt=prompt, repository=root, slot_policy_revision=policy
    )
    assert (
        historical.table == encoded.table and historical.provider_prompt == encoded.provider_prompt
    )
    assert historical.freshness_checked is False
    with pytest.raises(ValueError, match="historical"):
        codec.decode_semantic_router_response(
            response=codec._json(reply(encoded)), encoded=historical, repository=root
        )


@pytest.mark.parametrize("field", ["pack", "reconstruction"])
def test_unknown_record_schema_remains_literal_without_recursive_aliasing(native, field):
    payload = copy.deepcopy(native[3])
    payload[field]["schema"] = "future-record@1"
    slots = codec._slots(payload, slot_policy_revision=codec.SLOT_POLICY_V2)
    assert not any(path[0] == field for path, _ in slots)
    assert payload == {**native[3], field: payload[field]}


@pytest.mark.parametrize(
    "path",
    [
        ("manifest", "mod.py", "source_cid"),
        ("pack", "pack", "delta_cid"),
        ("reconstruction", "scope_cid"),
    ],
)
def test_known_typed_slot_invalid_identifier_refuses(native, path):
    payload = copy.deepcopy(native[3])
    codec._set(payload, path, "arbitrary prose identifier")
    with pytest.raises(codec.SemanticTranslationError, match="typed semantic slot"):
        codec._slots(payload, slot_policy_revision=codec.SLOT_POLICY_V2)


@pytest.mark.parametrize("path", [("pack",), ("worker_projection",), ("reconstruction",)])
def test_known_shape_cannot_add_unknown_identifier_slots(native, path):
    payload = copy.deepcopy(native[3])
    record = codec._get(payload, path)
    record["arbitrary_capsule_cid"] = payload["semantic_root_cid"]
    with pytest.raises(codec.SemanticTranslationError, match="typed semantic slot"):
        codec._slots(payload, slot_policy_revision=codec.SLOT_POLICY_V2)


def test_unknown_reference_metadata_and_capsule_ast_remain_literal(native):
    payload = copy.deepcopy(native[3])
    reference = payload["pack"]["references"][0]
    reference["metadata"]["arbitrary_capsule_cid"] = payload["semantic_root_cid"]
    # Rebind the native descriptor so this is a valid typed reference with
    # unknown literal metadata, rather than a stale descriptor failure.
    from ipfs_accelerate_py.agent_supervisor.context.context_contracts import ContextReference

    reference.pop("content_id", None)
    reference.pop("reference_content_id", None)
    payload["pack"]["references"][0] = ContextReference.from_dict(reference).to_dict()
    slots = codec._slots(payload, slot_policy_revision=codec.SLOT_POLICY_V2)
    assert not any(path[:4] == ("pack", "references", 0, "metadata") for path, _ in slots)
    assert not any(path[0] == "capsules" and "metadata" in path for path, _ in slots)
    assert not any(
        path[0] in {"raw_sources", "objective", "required_raw_paths"} for path, _ in slots
    )


def test_actual_refresh_lineage_closed_identifiers_round_trip(native):
    root, artifact, _, _, _ = native
    old = json.loads(artifact.read_text())
    (root / "mod.py").write_text(SOURCE + "# changed after nomination\n")
    resolved = resolve_semantic_worker_context(
        repository=root,
        artifact=artifact.relative_to(root).as_posix(),
        expected_sha256=codec._sha(artifact.read_bytes()),
        task_id="TASK-SLOTS",
        refresh_output=root / ".runtime/refresh",
        attempt_id="TASK-SLOTS:2",
    )
    refreshed = root / resolved["artifact"]
    text = resolved["text"]
    prompt = compile_prompt(root, refreshed, text)
    encoded = codec.encode_semantic_router_prompt(
        prompt=prompt, repository=root, slot_policy_revision=codec.SLOT_POLICY_V2
    )
    paths = {
        tuple(path)
        for path, _ in codec._slots(json.loads(text), slot_policy_revision=codec.SLOT_POLICY_V2)
    }
    assert ("refresh_lineage", "previous_scope_cid") in paths
    assert ("refresh_lineage", "previous_semantic_root_cid") in paths
    assert ("refresh_lineage", "source_delta", "mod.py", "before", "source_cid") in paths
    assert json.loads(text)["refresh_lineage"]["previous_scope_cid"] == old["scope_cid"]
    assert (
        codec.restore_semantic_router_prompt(
            provider_prompt=encoded.provider_prompt, table=encoded.table, repository=root
        )
        == prompt
    )


def test_role_view_verifies_full_parent_and_restores_projected_prompt(native):
    root, artifact, _, payload, _ = native
    before = artifact.read_bytes()
    view, prompt, _ = role(native)
    encoded = codec.encode_semantic_router_prompt(
        prompt=prompt, repository=root, slot_policy_revision=codec.SLOT_POLICY_V2
    )
    paths = {tuple(p) for p in encoded.table.to_dict()["replacement_paths"]}
    assert ("required_context", "scope_cid") in paths
    assert ("origin", "scope_cid") in paths
    assert ("deferred_references", 0, "metadata", "semantic_root_cid") in paths
    assert (
        codec.restore_semantic_router_prompt(
            provider_prompt=encoded.provider_prompt, table=encoded.table, repository=root
        )
        == prompt
    )
    assert encoded.table.to_dict()["source_manifest"] == payload["manifest"]
    assert view["required_context"]["raw_sources"] == payload["raw_sources"]
    assert artifact.read_bytes() == before
    decoded = codec.decode_semantic_router_response(
        response=codec._json(reply(encoded)), encoded=encoded, repository=root
    )
    assert decoded.receipt["freshness_checked"] is True
    with pytest.raises(codec.SemanticTranslationError, match="requires.*v2"):
        codec.encode_semantic_router_prompt(prompt=prompt, repository=root)


@pytest.mark.parametrize(
    "mutation",
    [
        "parent_sha",
        "parent_path",
        "source",
        "capsule",
        "root",
        "scope",
        "repository",
        "tree",
        "coverage",
        "handle",
        "authority",
        "telemetry",
    ],
)
def test_repaired_role_view_chunks_cannot_escape_exact_parent_or_coverage(native, mutation):
    root, artifact, _, _, _ = native
    view, _, _ = role(native)
    if mutation == "parent_sha":
        view["origin"]["expected_sha256"] = "a" * 64
    elif mutation == "parent_path":
        view["origin"]["artifact"] = ".runtime/foreign.json"
    elif mutation == "source":
        view["required_context"]["raw_sources"]["mod.py"] += "# omitted literal\n"
    elif mutation == "capsule":
        view["required_context"]["capsules"].pop()
    elif mutation == "root":
        view["origin"]["semantic_root_cid"] = "b" + "a" * 58
    elif mutation == "scope":
        view["origin"]["scope_cid"] = "b" + "a" * 58
    elif mutation == "repository":
        view["origin"]["repository_id"] = "repo:foreign"
    elif mutation == "tree":
        view["origin"]["tree_id"] = "tree:foreign"
    elif mutation == "coverage":
        view["required_coverage_sha256"] = "a" * 64
    elif mutation == "handle":
        view["deferred_references"][0]["metadata"]["selector_policy"] = "all-content@1"
    elif mutation == "authority":
        view["execution_authority"] = True
    else:
        view["required_context"]["pack"]["token_estimate"] = {}
    # Fresh compiled reference hashes do not bypass parent-backed regeneration.
    prompt = compile_prompt(root, artifact, codec._json(view))
    with pytest.raises(ValueError):
        codec.encode_semantic_router_prompt(
            prompt=prompt, repository=root, slot_policy_revision=codec.SLOT_POLICY_V2
        )


def test_role_view_stale_original_current_refuses_and_historical_replays(native):
    root, _, _, _, _ = native
    _, prompt, _ = role(native)
    encoded = codec.encode_semantic_router_prompt(
        prompt=prompt, repository=root, slot_policy_revision=codec.SLOT_POLICY_V2
    )
    (root / "mod.py").write_text(SOURCE + "# subsequent edit\n")
    with pytest.raises(ValueError, match="stale"):
        codec.restore_semantic_router_prompt(
            provider_prompt=encoded.provider_prompt, table=encoded.table, repository=root
        )
    replay = codec.replay_semantic_router_prompt_for_audit(
        prompt=prompt, repository=root, slot_policy_revision=codec.SLOT_POLICY_V2
    )
    assert replay.provider_prompt == encoded.provider_prompt and replay.table == encoded.table
    with pytest.raises(ValueError, match="historical"):
        codec.decode_semantic_router_response(
            response=codec._json(reply(encoded)), encoded=replay, repository=root
        )


def test_rebound_replacement_path_to_literal_cannot_become_valid_v2_table(native):
    root, _, _, _, _ = native
    encoded = encode(native)
    record = encoded.table.to_dict()
    record["replacement_paths"][0] = ["raw_sources", "mod.py"]
    record["translation_cid"] = codec.cid_for_payload(
        {k: v for k, v in record.items() if k != "translation_cid"}
    )
    altered = codec.SemanticTranslationTable.from_dict(record)
    wire = json.loads(encoded.provider_prompt)
    wire["translation_cid"] = altered.translation_cid
    with pytest.raises(ValueError):
        codec.restore_semantic_router_prompt(
            provider_prompt=codec._json(wire), table=altered, repository=root
        )


@pytest.mark.parametrize("payload", [None, [], [{}], "not a table"])
def test_nonobject_translation_table_rejects_through_the_contract(payload):
    with pytest.raises(codec.SemanticTranslationError, match="translation table"):
        codec.SemanticTranslationTable(codec._json(payload), "b" + "a" * 58)


def test_native_exact_byte_repacking_flag_is_a_supported_literal(native):
    payload = copy.deepcopy(native[3])
    payload["worker_projection"]["exact_byte_budget_repacked"] = True
    paths = codec._slots(payload, slot_policy_revision=codec.SLOT_POLICY_V2)
    assert ("worker_projection", "capsule_index_cid") in {path for path, _ in paths}
    assert not any("exact_byte_budget_repacked" in path for path, _ in paths)


@pytest.mark.parametrize("flag", [False, 0, 1])
def test_native_repacking_flag_cannot_change_type_or_meaning(native, flag):
    payload = copy.deepcopy(native[3])
    payload["worker_projection"]["exact_byte_budget_repacked"] = flag
    with pytest.raises(codec.SemanticTranslationError, match="repacking flag"):
        codec._slots(payload, slot_policy_revision=codec.SLOT_POLICY_V2)


@pytest.mark.parametrize("field", ["scope_cid", "semantic_root_cid", "doctor_manifest_cid"])
def test_required_typed_slots_do_not_accept_null(native, field):
    payload = copy.deepcopy(native[3])
    payload[field] = None
    with pytest.raises(codec.SemanticTranslationError, match="typed semantic slot"):
        codec._slots(payload, slot_policy_revision=codec.SLOT_POLICY_V2)


def test_added_v2_slots_still_obey_the_complete_slot_population_bound(native, monkeypatch):
    payload = native[3]
    legacy = codec._slots(payload)
    assert len(codec._slots(payload, slot_policy_revision=codec.SLOT_POLICY_V2)) > len(legacy)
    monkeypatch.setattr(codec, "MAX_ITEMS", len(legacy))
    assert codec._slots(payload) == legacy
    with pytest.raises(codec.SemanticTranslationError, match="scope exceeds bound"):
        codec._slots(payload, slot_policy_revision=codec.SLOT_POLICY_V2)


def test_unknown_admission_schema_retains_its_identifiers_as_literal_data(native):
    payload = copy.deepcopy(native[3])
    payload["admissions"][0]["schema"] = "future-admission@1"
    slots = codec._slots(payload, slot_policy_revision=codec.SLOT_POLICY_V2)
    assert not any(path[:2] == ("admissions", 0) for path, _ in slots)
