"""Independent business census and exact inverse for pure packet presentation.

These fixtures use the published native transport shape, not a live producer.
Native issuance, compiler/source freshness and worker delivery are separately
qualified by the binding tests. No model, tokenizer, DB or target code runs here.
"""
from copy import copy, deepcopy
import hashlib
import json

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import repository_world_packet_presentation as codec


PREFIX = "Observe every original business/source fact; no execution or completion authority."
SOURCE = "# Exact authored source; never imported.\n" + "def untouched(value):\n    return value + 1\n" * 35


def wire(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def sha(raw): return hashlib.sha256(raw).hexdigest()


def native_prompt(*, payload_extra=None):
    task = {"task_cid": "native-task:original", "revision": 7, "body": {"completion": "auto", "title": "Preserve λ. " * 25},
        "outputs": [{"ordinal": 0, "path": "module.py", "new_policy": {"nullable": None, "empty": []}}],
        "acceptance": [{"criterion": "Every occurrence and unknown field remains unreviewed."}], "validations": [],
        "dependencies": [{"dependency_task_cid": "native:dependency", "kind": "requires"}]}
    task_string = wire(task).decode()
    reference = {"reference_id": "source:module.py", "kind": "exact-selected-source", "summary": SOURCE,
        "path": "module.py", "metadata": {"required": True, "confidence": 0.75, "source_revision": "literal-revision"}}
    payloads = [
        {"evidence_id": "task-text", "task_text": task_string},
        {"evidence_id": "compiled-core", "context": {"goal": {"native_task_contract_json": task_string},
            "scope": {"paths": ["module.py"], "allowed_effects": ["repository_read"]},
            "acceptance": task["acceptance"], "native_dependency": task["dependencies"]}},
        {"evidence_id": "context-reference:source", "reference": reference},
        {"evidence_id": "different-role-same-facts", "relationship": {"subject": "actor:A", "predicate": "observes", "object": "source:module.py"},
            "reference": {**reference, "reference_id": "source:second-presentation", "role": "unselected_ambiguous_observation"}},
    ]
    if payload_extra is not None: payloads[0]["unknown_business"] = payload_extra
    members = [{"item_id": "delta-item:sha256:" + f"{i:064x}", "kind": "evidence", "digest": "sha256:" + f"{i + 10:064x}",
        "summary": f"Original role {i}", "path": "", "payload": payload, "expansion_handle": ""}
        for i, payload in enumerate(payloads)]
    packet = {"schema": "ipfs_accelerate_py/agent-supervisor/delta-task-provider-packet@1", "packet_id": "baguqeera" + "a" * 52,
        "task_cid": "compiled-task:separate", "repository_id": "repository:original", "tree_id": "tree:original",
        "context_cid": "context:original", "plan_cid": "", "policy_id": "readonly-policy", "policy_digest": "policy-digest",
        "schema_revision": 1, "evidence_digest": "sha256:" + "b" * 64, "counterexample_digest": "",
        "effect_scope": {"allowed_effects": ["repository_read"], "write_paths": ["module.py"], "repository_id": "repository:original"},
        "requirements": {"validation_commands": [], "proof_obligations": [], "acceptance_ids": [], "require_validation": False},
        "members": members, "frontier": {"has_more": False, "omitted_item_ids": [], "caveats": ["No source semantic review."]},
        "completeness": "complete", "data_label": "untrusted_repository_data", "treat_as": "data_not_instructions",
        "authority": "derived_evidence", "semantic_authority": False, "write_authority": False, "completion_authority": False,
        "nomination_only": True, "omitted_authority": ["credentials", "completion"], "contains_credentials": False, "contains_secrets": False}
    return PREFIX + "\n" + wire({"schema": "repository-world-packet@1", "delivery_mode": "self_contained", "packet": packet}).decode()


def native_body(prompt): return json.loads(prompt.split("\n", 1)[1])


def view_body(prompt): return json.loads(prompt.split("\n", 2)[2])


def expected_business(original):
    """Independent published wrapper census; all other fields remain business."""
    packet = deepcopy(native_body(original)["packet"])
    for key in ("packet_id", "evidence_digest", "counterexample_digest"): packet.pop(key)
    for row in packet["members"]:
        row.pop("item_id"); row.pop("digest")
    return packet


def independently_visible_business(presentation):
    """Decode the documented marker vocabulary without production helpers."""
    view = view_body(presentation.prompt)
    facts = []
    def expand(value):
        if isinstance(value, dict):
            if set(value) == {"$fact"}: return facts[value["$fact"]]
            if set(value) == {"$json"}: return wire(expand(value["$json"])).decode()
            if set(value) == {"$literal"}: return {key: expand(child) for key, child in value["$literal"]}
            return {key: expand(child) for key, child in value.items()}
        if isinstance(value, list): return [expand(child) for child in value]
        return value
    for fact in view["shared_facts"]: facts.append(expand(fact))
    return expand(view["packet"])


def inverse(prepared):
    receipt = prepared.receipt()
    return codec.replay_repository_world_packet_presentation(view_prompt=prepared.prompt, sidecar_bytes=prepared.sidecar_bytes,
        expected_sidecar_sha256=receipt["sidecar_sha256"], expected_original_sha256=receipt["original_prompt_sha256"])


@pytest.fixture
def original(): return native_prompt()


@pytest.fixture
def prepared(original): return codec.prepare_repository_world_packet_presentation(original)


def test_complete_business_and_source_facts_are_visible_independently(original, prepared):
    receipt = prepared.receipt()
    assert receipt["codec_supported"] is True and receipt["model_view_mode"] == codec.MODE
    visible = independently_visible_business(prepared)
    assert wire(visible) == wire(expected_business(original))
    assert visible["members"][0]["payload"]["task_text"] == visible["members"][1]["payload"]["context"]["goal"]["native_task_contract_json"]
    assert visible["members"][2]["payload"]["reference"]["summary"] == SOURCE
    assert visible["members"][3]["payload"]["reference"]["summary"] == SOURCE
    assert visible["members"][2]["payload"]["reference"]["reference_id"] != visible["members"][3]["payload"]["reference"]["reference_id"]
    assert visible["members"][3]["payload"]["relationship"]["predicate"] == "observes"
    assert len(visible["members"]) == 4 and visible["members"][3]["payload"]["reference"]["role"] == "unselected_ambiguous_observation"
    assert inverse(prepared) == original and prepared.validate_original(original) == receipt
    assert prepared.prompt.startswith(PREFIX + "\n")
    assert receipt["shared_fact_count"] > 0 and receipt["shared_reference_count"] >= 2
    assert receipt["host_custody_field_count"] == 3 + 2 * 4
    assert receipt["presented_prompt_bytes"] < receipt["original_prompt_bytes"]


def test_custody_is_only_reviewed_wrappers_not_business_ids(original, prepared):
    sidecar = json.loads(prepared.sidecar_bytes)
    assert sidecar["original_prompt"] == original
    assert {tuple(row[:2]) for row in sidecar["custody"] if row[0] == "packet"} == {
        ("packet", "packet_id"), ("packet", "evidence_digest"), ("packet", "counterexample_digest")}
    assert all(row[2] in {"item_id", "digest"} for row in sidecar["custody"] if row[0] == "member")
    visible = independently_visible_business(prepared)
    for key in ("task_cid", "context_cid", "tree_id", "repository_id", "policy_id", "policy_digest"):
        assert visible[key] == native_body(original)["packet"][key]
    assert visible["members"][2]["payload"]["reference"]["reference_id"] == "source:module.py"


def test_equal_business_member_rows_keep_distinct_wrapper_occurrences_and_integrity():
    body = native_body(native_prompt())
    first = deepcopy(body["packet"]["members"][0])
    second = deepcopy(first)
    second["item_id"] = "delta-item:sha256:" + "f" * 64
    second["digest"] = "sha256:" + "e" * 64
    body["packet"]["members"] = [first, second]
    original = PREFIX + "\n" + wire(body).decode()
    prepared = codec.prepare_repository_world_packet_presentation(original)
    visible = independently_visible_business(prepared)
    assert len(visible["members"]) == 2 and wire(visible["members"][0]) == wire(visible["members"][1])
    restored = native_body(inverse(prepared))["packet"]["members"]
    assert restored == [first, second]
    assert restored[0]["item_id"] != restored[1]["item_id"]
    assert restored[0]["digest"] != restored[1]["digest"]
    assert inverse(prepared) == original


def test_nested_business_member_literal_does_not_acquire_another_rows_custody():
    body = native_body(native_prompt())
    target = deepcopy(body["packet"]["members"][0])
    literal = {key: value for key, value in target.items() if key not in {"item_id", "digest"}}
    observer = deepcopy(body["packet"]["members"][1])
    observer["payload"] = {"evidence_id": "nested-row-observation", "original_row_literal": literal}
    body["packet"]["members"] = [observer, target]
    original = PREFIX + "\n" + wire(body).decode()
    prepared = codec.prepare_repository_world_packet_presentation(original)
    visible = independently_visible_business(prepared)
    assert visible["members"][0]["payload"]["original_row_literal"] == visible["members"][1]
    restored = native_body(inverse(prepared))["packet"]["members"]
    nested = restored[0]["payload"]["original_row_literal"]
    assert nested == literal and "item_id" not in nested and "digest" not in nested
    assert restored[1] == target and inverse(prepared) == original


@pytest.mark.parametrize("data", [
    {"$fact": 0}, {"$json": {"$fact": 4}}, {"$literal": [["key", "value"]]},
    {"$fact": None, "$json": [], "other": "x"}, {"nested": [{"$fact": True}, {"$literal": None}]},
])
def test_literal_marker_objects_never_become_codec_references(data):
    original = native_prompt(payload_extra=data)
    prepared = codec.prepare_repository_world_packet_presentation(original)
    assert independently_visible_business(prepared)["members"][0]["payload"]["unknown_business"] == data
    assert inverse(prepared) == original


def test_literal_marker_collisions_inside_canonical_json_data_string_are_escaped():
    data = wire({"$fact": 3, "other": "原文 λ " * 60, "nested": {"$json": [1, 1.0, True, None]}}).decode()
    original = native_prompt(payload_extra={"canonical_data": data})
    prepared = codec.prepare_repository_world_packet_presentation(original)
    assert independently_visible_business(prepared)["members"][0]["payload"]["unknown_business"]["canonical_data"] == data
    assert inverse(prepared) == original


def test_absent_null_empty_order_multiplicity_and_numeric_types_remain_distinct():
    data = {"ordered": [2, 1, 2, None, [], {}, "", False, 0, 0.0, -0.0, "0", "0.0"],
        "missing_is_not_null": {"present": None}, "confidence": 0.95, "unicode": "é ≠ é; 😀;\n"}
    original = native_prompt(payload_extra=data)
    prepared = codec.prepare_repository_world_packet_presentation(original)
    visible = independently_visible_business(prepared)["members"][0]["payload"]["unknown_business"]
    assert wire(visible) == wire(data)
    assert type(visible["ordered"][8]) is int and type(visible["ordered"][9]) is float
    assert inverse(prepared) == original


@pytest.mark.parametrize("data", ['{"duplicates":1,"duplicates":2}' + " " * 150,
    '{"x":NaN,"literal":"' + "x" * 150 + '"}', ' {"noncanonical":"' + "x" * 150 + '"} ', '{invalid' + "x" * 150])
def test_json_like_business_strings_are_opaque_unless_exact_canonical_data(data):
    original = native_prompt(payload_extra=data)
    prepared = codec.prepare_repository_world_packet_presentation(original)
    assert independently_visible_business(prepared)["members"][0]["payload"]["unknown_business"] == data
    assert inverse(prepared) == original


@pytest.mark.parametrize("change", ["outer", "outer_schema", "packet", "packet_schema", "member", "member_id"])
def test_unknown_transport_fields_and_schemas_fallback_to_entire_original(change):
    body = native_body(native_prompt())
    if change == "outer": body["unknown_transport"] = {"must_keep": [1, 2]}
    elif change == "outer_schema": body["schema"] = "future@2"
    elif change == "packet": body["packet"]["unknown_transport"] = {"must_keep": True}
    elif change == "packet_schema": body["packet"]["schema"] = "future-packet@2"
    elif change == "member": body["packet"]["members"][0]["unknown_wrapper"] = {"must_keep": None}
    else: body["packet"]["members"][0]["item_id"] = "business:unreviewed-role-id"
    original = PREFIX + "\n" + wire(body).decode()
    prepared = codec.prepare_repository_world_packet_presentation(original)
    assert prepared.prompt == original and prepared.receipt()["model_view_mode"] == "original_fallback"
    assert prepared.receipt()["codec_supported"] is False and prepared.receipt()["fallback_reason"]
    assert inverse(prepared) == original and json.loads(prepared.sidecar_bytes)["custody"] == []


@pytest.mark.parametrize("original", ["Plain exact source text π.\nKeep this line.", "No newline.",
    PREFIX + '\n{"schema":"unknown","packet":null}', PREFIX + '\n  {"unknown":true}'])
def test_unrecognized_input_is_a_named_lossless_original_fallback(original):
    prepared = codec.prepare_repository_world_packet_presentation(original)
    assert prepared.prompt == inverse(prepared) == original
    assert prepared.receipt()["model_view_mode"] == "original_fallback"


@pytest.mark.parametrize("body", ['{"schema":1,"schema":2}', '{"x":NaN}', '{"x":Infinity}', '{"x":1e309}', '{"x":[}'])
def test_strict_transport_json_refuses_duplicate_nonfinite_overflow_and_malformed(body):
    with pytest.raises(ValueError): codec.prepare_repository_world_packet_presentation(PREFIX + "\n" + body)


def test_noncanonical_native_transport_keeps_original_exact_whitespace():
    original = PREFIX + "\n" + json.dumps(native_body(native_prompt()), indent=2, ensure_ascii=False)
    prepared = codec.prepare_repository_world_packet_presentation(original)
    assert prepared.prompt == original and prepared.receipt()["fallback_reason"] == "noncanonical_transport"
    assert inverse(prepared) == original


def test_original_guidance_is_byte_exact_and_not_part_of_shared_fact_data(original, prepared):
    assert prepared.prompt.partition("\n")[0] == original.partition("\n")[0]
    assert prepared.prompt.split("\n", 2)[1] == codec.EXPLANATION
    assert inverse(prepared).partition("\n")[0] == PREFIX


def test_short_scalar_and_empty_values_do_not_create_a_fact_pool():
    body = native_body(native_prompt())
    body["packet"]["members"] = [body["packet"]["members"][0]]
    body["packet"]["members"][0]["payload"] = {"values": [False, 0, None, [], {}, ""] * 10}
    original = PREFIX + "\n" + wire(body).decode()
    prepared = codec.prepare_repository_world_packet_presentation(original)
    assert prepared.receipt()["shared_fact_count"] == 0
    assert independently_visible_business(prepared) == expected_business(original)
    assert inverse(prepared) == original


@pytest.mark.parametrize("change", ["view", "sidecar", "original_pin", "sidecar_pin", "missing_pin", "bool_pin"])
def test_independent_original_and_sidecar_and_view_pins_are_required(original, prepared, change):
    row = prepared.receipt()
    args = dict(view_prompt=prepared.prompt, sidecar_bytes=prepared.sidecar_bytes,
        expected_sidecar_sha256=row["sidecar_sha256"], expected_original_sha256=row["original_prompt_sha256"])
    if change == "view": args["view_prompt"] += " "
    elif change == "sidecar": args["sidecar_bytes"] += b" "
    elif change == "original_pin": args["expected_original_sha256"] = "0" * 64
    elif change == "sidecar_pin": args["expected_sidecar_sha256"] = "0" * 64
    elif change == "missing_pin": args["expected_original_sha256"] = None
    else: args["expected_sidecar_sha256"] = True
    with pytest.raises(ValueError): codec.replay_repository_world_packet_presentation(**args)


@pytest.mark.parametrize("change", ["drop_business", "drop_member", "role", "facts_drop", "facts_duplicate", "facts_unused",
    "facts_reorder", "cycle", "dangling", "bool_index", "custody_role"])
def test_resealed_view_and_sidecar_cannot_drop_or_retarget_original_population(original, prepared, change):
    view, sidecar = view_body(prepared.prompt), json.loads(prepared.sidecar_bytes)
    if change == "drop_business": view["packet"].pop("effect_scope")
    elif change == "drop_member": view["packet"]["members"].pop()
    elif change == "role": view["packet"]["members"][0]["kind"] = "approved-proof"
    elif change == "facts_drop": view["shared_facts"].pop()
    elif change == "facts_duplicate": view["shared_facts"].append(deepcopy(view["shared_facts"][0]))
    elif change == "facts_unused": view["shared_facts"].append({"new": "unreferenced"})
    elif change == "facts_reorder": view["shared_facts"].reverse()
    elif change == "cycle": view["shared_facts"][0] = {"$fact": 0}
    elif change == "dangling": view["packet"]["members"][0]["payload"] = {"$fact": 999999}
    elif change == "bool_index": view["packet"]["members"][0]["payload"] = {"$fact": True}
    else: sidecar["custody"][0][2] = "not-original"
    presented = PREFIX + "\n" + codec.EXPLANATION + "\n" + wire(view).decode()
    sidecar.update(presented_prompt_sha256=sha(presented.encode()), presented_prompt_bytes=len(presented.encode()))
    raw_sidecar = wire(sidecar)
    with pytest.raises(ValueError): codec.replay_repository_world_packet_presentation(view_prompt=presented,
        sidecar_bytes=raw_sidecar, expected_sidecar_sha256=sha(raw_sidecar), expected_original_sha256=sha(original.encode()))


@pytest.mark.parametrize("field", ["prompt", "sidecar_bytes", "_receipt_bytes"])
def test_equal_byte_instance_replacement_refuses_original_issuance(prepared, original, field):
    old = getattr(prepared, field)
    clone = old.encode().decode() if isinstance(old, str) else bytes(bytearray(old))
    assert clone == old and clone is not old
    object.__setattr__(prepared, field, clone)
    try:
        with pytest.raises(ValueError, match="original_presentation_replaced"): prepared.validate_original(original)
    finally: object.__setattr__(prepared, field, old)


@pytest.mark.parametrize("method", ["receipt", "validate_original", "_retained", "_seal_material"])
def test_issued_presentation_is_immutable_and_methods_cannot_be_shadowed(prepared, method):
    with pytest.raises(AttributeError): setattr(prepared, method, lambda *_a: {})


@pytest.mark.parametrize("kind", ["copy", "empty", "partial", "subclass", "receipt"])
def test_detached_or_unissued_object_is_not_presentation_authority(prepared, kind):
    if kind == "copy":
        with pytest.raises((ValueError, AttributeError)):
            bad = copy(prepared)
            codec.PreparedRepositoryWorldPacketPresentation._retained(bad)
        return
    elif kind == "empty": bad = object.__new__(codec.PreparedRepositoryWorldPacketPresentation)
    elif kind == "partial":
        bad = object.__new__(codec.PreparedRepositoryWorldPacketPresentation); object.__setattr__(bad, "_seal", codec._SEAL)
    elif kind == "subclass":
        class Other(codec.PreparedRepositoryWorldPacketPresentation): pass
        bad = object.__new__(Other)
    else: bad = prepared.receipt()
    with pytest.raises(ValueError): codec.PreparedRepositoryWorldPacketPresentation._retained(bad)


@pytest.mark.parametrize("field", ["prompt", "sidecar_bytes", "_receipt_bytes", "_seal"])
def test_ordinary_field_deletion_is_denied_and_bypass_has_closed_refusal(prepared, original, field):
    old = getattr(prepared, field)
    with pytest.raises(AttributeError): delattr(prepared, field)
    assert prepared.validate_original(original)["codec_supported"] is True
    object.__delattr__(prepared, field)
    try:
        with pytest.raises(ValueError): codec.PreparedRepositoryWorldPacketPresentation.validate_original(prepared, original)
    finally: object.__setattr__(prepared, field, old)


def test_closed_receipt_has_only_hashes_counts_profile_and_false_authorities(prepared):
    row = prepared.receipt()
    assert all(row[key] is False for key in codec.FALSE_FIELDS)
    assert row["sidecar_contains_original_prompt"] is row["presentation_structural_only"] is True
    assert SOURCE not in json.dumps(row) and PREFIX not in json.dumps(row)
    assert row["sidecar_sha256"] == sha(prepared.sidecar_bytes)
    assert row["presented_prompt_sha256"] == sha(prepared.prompt.encode())
    assert type(row["shared_fact_count"]) is int and row["shared_fact_count"] > 0
    row["proof_authority"] = True
    assert prepared.receipt()["proof_authority"] is False


@pytest.mark.parametrize("value", [None, {}, b"bytes", "", "x" * (codec.MAX_BYTES + 1), "\ud800"])
def test_original_input_type_utf8_and_size_are_bounded(value):
    with pytest.raises(ValueError): codec.prepare_repository_world_packet_presentation(value)


def test_lexical_depth_refusal_precedes_json_materialization(monkeypatch):
    called = []
    monkeypatch.setattr(codec, "MAX_DEPTH", 4)
    monkeypatch.setattr(codec.json, "loads", lambda *_a, **_kw: called.append(True))
    with pytest.raises(ValueError): codec.prepare_repository_world_packet_presentation(PREFIX + '\n{"deep":' + '[' * 8 + '0' + ']' * 8 + '}')
    assert called == []


def test_lexical_node_refusal_precedes_json_materialization(monkeypatch):
    called = []
    monkeypatch.setattr(codec, "MAX_NODES", 10)
    monkeypatch.setattr(codec.json, "loads", lambda *_a, **_kw: called.append(True))
    with pytest.raises(ValueError): codec.prepare_repository_world_packet_presentation(PREFIX + '\n{"wide":[' + ','.join('0' for _ in range(20)) + ']}')
    assert called == []


def test_shared_fact_expansion_bomb_is_measured_before_any_materialization(monkeypatch):
    called = []
    monkeypatch.setattr(codec, "_unpack", lambda *_a: called.append(True))
    measures = []
    with pytest.raises(ValueError):
        for index in range(25):
            value = "x" * 100 if index == 0 else [{"$fact": index - 1}, {"$fact": index - 1}]
            measures.append(codec._unpack_measure(value, measures, index))
    assert called == []


def test_fact_bound_uses_complete_original_fallback_without_truncation(original, monkeypatch):
    monkeypatch.setattr(codec, "MAX_FACTS", 0)
    prepared = codec.prepare_repository_world_packet_presentation(original)
    assert prepared.prompt == original and prepared.receipt()["model_view_mode"] == "original_fallback"
    assert prepared.receipt()["fallback_reason"] == "shared_fact_bound"
    assert inverse(prepared) == original
