"""Lossless public columns, refusal contracts, and bounded virtual expansion."""

from __future__ import annotations

import copy
import hashlib
import json

import pytest
from ipfs_accelerate_py.agent_supervisor.semantic_refactoring import python_code_view as engine
from ipfs_accelerate_py.agent_supervisor.semantic_refactoring import python_code_view_wire as wire

SOURCE = (
    '# café λ: quantity_total is literal commentary.\n'
    '"""Semantic module documentation: quantity_total."""\n'
    'def calculate_total(quantity_total, tax_amount):\n'
    '    """Keep quantity_total and tax_amount literally."""\n'
    '    subtotal_amount = quantity_total + tax_amount * 0xA\n'
    '    return subtotal_amount // 3  # café λ\n'
    'def retain_other(other_quantity):\n'
    '    return (\n'
    '        other_quantity + 2\n'
    '    )\n'
)


def make(source=SOURCE):
    return engine.build_python_code_view(
        source=source, path="src/calculation.py", repository_id="repository:fixture",
        tree_id="tree:fixture", task_id="task:fixture",
    )


def public():
    return make().model_packet()


def packed():
    return wire.pack_python_code_view_packet(public())[0]


def at(value, path):
    for key in path:
        value = value[key]
    return value


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


@pytest.mark.parametrize("ending", ["\n", "\r\n"])
@pytest.mark.parametrize("final_newline", [True, False])
def test_exact_roundtrip_preserves_every_field_literal_and_semantic_name(ending, final_newline):
    source = SOURCE.replace("\n", ending)
    if not final_newline:
        source = source.removesuffix(ending)
    view = make(source)
    packet = view.model_packet()
    original = copy.deepcopy(packet)
    transport, stats = wire.pack_python_code_view_packet(packet)
    restored = wire.unpack_python_code_view_packet(transport)
    assert restored == packet == original
    assert encoded(restored) == encoded(packet)
    assert transport["source"].encode() == view.compact_source.encode()
    assert "café λ" in transport["source"] and "0xA" in transport["source"]
    assert restored["functions"][0]["aliases"][0]["original"] == "quantity_total"
    assert restored["functions"][1]["name"] == "retain_other"
    assert transport["slots"][-1][1] == 1
    assert restored["slots"][-1]["editable"] is False
    assert restored["slots"][-1]["refusal_reason"] == wire.REFUSAL_REASON
    assert restored["authority"] and all(value is False for value in restored["authority"].values())
    assert stats["public_packet_bytes"] == len(encoded(packet))
    assert stats["wire_packet_bytes"] == len(encoded(transport))
    assert stats["input_tokens"] is None and stats["lossless"] is True
    assert stats["byte_delta"] == len(encoded(transport)) - len(encoded(packet))
    assert stats["wire_smaller"] is (stats["byte_delta"] < 0)
    assert wire.require_bound_python_code_view_packet(transport, view=view) == packet
    assert wire.require_bound_python_code_view_packet(packet, view=view) == packet


def test_wire_generation_profile_has_fixed_columns_and_no_private_native_material():
    packet = public()
    transport, _ = wire.pack_python_code_view_packet(packet)
    assert set(transport) == wire.WIRE_FIELDS
    assert transport["schema"] == "supervisor-python-code-view-wire@1"
    assert transport["profile"] == "python-arithmetic-model-packet-columns@1"
    assert transport["bindings"] == [packet["bindings"][key] for key in wire.BINDING_COLUMNS]
    assert transport["functions"][0][2][0] == ["quantity_total", "a", 0]
    assert transport["functions"][0][2][-1] == ["subtotal_amount", "c", 1]
    assert transport["slots"][0][2] == 0 and transport["slots"][1][2] == 1
    serialized = encoded(transport)
    for private in ("original_source", "source_span", "token_map", "source_sha256", "original_expression"):
        assert private.encode() not in serialized
    assert SOURCE.encode() not in serialized


def test_public_and_wire_outputs_have_no_mutable_shared_collections():
    packet = public()
    transport, _ = wire.pack_python_code_view_packet(packet)
    original_wire = copy.deepcopy(transport)
    restored = wire.unpack_python_code_view_packet(transport)
    packet["functions"][0]["aliases"][0]["original"] = "changed_public_name"
    restored["functions"][0]["aliases"][0]["original"] = "changed_restored_name"
    restored["bindings"]["path"] = "different.py"
    assert transport == original_wire
    transport["functions"][0][2][0][0] = "changed_wire_name"
    assert packet["functions"][0]["aliases"][0]["original"] == "changed_public_name"
    assert restored["functions"][0]["aliases"][0]["original"] == "changed_restored_name"


@pytest.mark.parametrize("field,value", [
    ("schema", "unknown-model@1"), ("schema", True),
    ("revision", "python-arithmetic-source-view@2"), ("revision", 1),
    ("view_id", "not-a-cid"), ("view_id", True),
    ("compact_source", 1), ("functions", ()), ("slots", ()),
])
def test_unknown_public_schema_revision_and_top_level_types_refuse(field, value):
    packet = public()
    packet[field] = value
    with pytest.raises(wire.PythonCodeViewWireError):
        wire.pack_python_code_view_packet(packet)


@pytest.mark.parametrize("field,value", [
    ("schema", "unknown-wire@1"), ("schema", False),
    ("profile", "python-arithmetic-model-packet-columns@2"), ("profile", 1),
    ("view_id", "not-a-cid"), ("source", 1),
    ("bindings", {}), ("functions", {}), ("slots", {}),
])
def test_unknown_wire_profile_and_top_level_types_refuse(field, value):
    transport = packed()
    transport[field] = value
    with pytest.raises(wire.PythonCodeViewWireError):
        wire.unpack_python_code_view_packet(transport)


@pytest.mark.parametrize("path", [(), ("bindings",), ("authority",), ("functions", 0), ("functions", 0, "aliases", 0), ("slots", 0)])
@pytest.mark.parametrize("operation", ["add", "remove"])
def test_unknown_and_missing_public_fields_are_never_silently_omitted(path, operation):
    packet = public()
    item = at(packet, path)
    if operation == "add":
        item["unexpected"] = "literal"
    else:
        item.pop(next(iter(item)))
    with pytest.raises(wire.PythonCodeViewWireError):
        wire.pack_python_code_view_packet(packet)


@pytest.mark.parametrize("field", ["columns", "authority", "revision", "compact_source", "original_source"])
def test_wire_cannot_add_implicit_or_private_fields(field):
    transport = packed()
    transport[field] = "unexpected"
    with pytest.raises(wire.PythonCodeViewWireError):
        wire.unpack_python_code_view_packet(transport)


@pytest.mark.parametrize("field", sorted(wire.WIRE_FIELDS))
def test_wire_requires_all_profile_fields(field):
    transport = packed()
    del transport[field]
    with pytest.raises(wire.PythonCodeViewWireError):
        wire.unpack_python_code_view_packet(transport)


@pytest.mark.parametrize("field", wire.AUTHORITY_COLUMNS)
@pytest.mark.parametrize("value", [True, 0, None, "false"])
def test_no_truthiness_or_integer_conversion_can_grant_or_forge_authority(field, value):
    packet = public()
    packet["authority"][field] = value
    with pytest.raises(wire.PythonCodeViewWireError):
        wire.pack_python_code_view_packet(packet)


@pytest.mark.parametrize("path", [("bindings",), ("functions", 0), ("functions", 0, 2, 0), ("slots", 0)])
@pytest.mark.parametrize("operation", ["short", "long", "tuple"])
def test_wire_positional_rows_require_exact_width_and_list_type(path, operation):
    transport = packed()
    parent, key = at(transport, path[:-1]), path[-1]
    item = parent[key]
    if operation == "short":
        parent[key] = item[:-1]
    elif operation == "long":
        parent[key] = [*item, "extra"]
    else:
        parent[key] = tuple(item)
    with pytest.raises(wire.PythonCodeViewWireError):
        wire.unpack_python_code_view_packet(transport)


@pytest.mark.parametrize("path", [("slots", 0, 1), ("slots", 0, 2), ("functions", 0, 2, 0, 2)])
@pytest.mark.parametrize("value", [True, False, -1, 1000, "0", 0.0, None])
def test_indices_are_exact_in_range_integers_and_never_booleans(path, value):
    transport = packed()
    at(transport, path[:-1])[path[-1]] = value
    with pytest.raises(wire.PythonCodeViewWireError):
        wire.unpack_python_code_view_packet(transport)


@pytest.mark.parametrize("kind", ["functions", "slots", "aliases"])
@pytest.mark.parametrize("representation", ["public", "wire"])
def test_reordered_generated_rows_refuse_instead_of_changing_scope(kind, representation):
    packet = public() if representation == "public" else packed()
    rows = packet[kind] if kind != "aliases" else (
        packet["functions"][0]["aliases"] if representation == "public" else packet["functions"][0][2]
    )
    rows[0], rows[1] = rows[1], rows[0]
    function = wire.pack_python_code_view_packet if representation == "public" else wire.unpack_python_code_view_packet
    with pytest.raises(wire.PythonCodeViewWireError):
        function(packet)


@pytest.mark.parametrize("kind", ["function_id", "function_name", "slot_id", "alias", "original"])
@pytest.mark.parametrize("representation", ["public", "wire"])
def test_duplicate_functions_slots_and_scoped_aliases_refuse(kind, representation):
    packet = public() if representation == "public" else packed()
    is_wire = representation == "wire"
    if kind in {"function_id", "function_name"}:
        key = (0 if kind == "function_id" else 1) if is_wire else ("function_id" if kind == "function_id" else "name")
        packet["functions"][1][key] = packet["functions"][0][key]
    elif kind == "slot_id":
        key = 0 if is_wire else "slot_id"
        packet["slots"][1][key] = packet["slots"][0][key]
    else:
        aliases = packet["functions"][0][2] if is_wire else packet["functions"][0]["aliases"]
        key = (1 if kind == "alias" else 0) if is_wire else kind
        aliases[1][key] = aliases[0][key]
    function = wire.unpack_python_code_view_packet if is_wire else wire.pack_python_code_view_packet
    with pytest.raises(wire.PythonCodeViewWireError):
        function(packet)


@pytest.mark.parametrize("value", ["unknown_function", True, None])
def test_public_slot_cannot_reference_an_unknown_function(value):
    packet = public()
    packet["slots"][0]["function"] = value
    with pytest.raises(wire.PythonCodeViewWireError):
        wire.pack_python_code_view_packet(packet)


@pytest.mark.parametrize("value", [1, 0, "true", None])
def test_slot_editability_is_an_exact_boolean(value):
    transport = packed()
    transport["slots"][0][4] = value
    with pytest.raises(wire.PythonCodeViewWireError):
        wire.unpack_python_code_view_packet(transport)


@pytest.mark.parametrize("editable,reason", [(True, "not-null"), (False, None), (False, "invented"), (False, True)])
def test_refusal_reason_is_not_dropped_or_invented(editable, reason):
    transport = packed()
    transport["slots"][0][4:6] = [editable, reason]
    with pytest.raises(wire.PythonCodeViewWireError):
        wire.unpack_python_code_view_packet(transport)


@pytest.mark.parametrize("value", ["0" * 63, "A" * 64, "z" * 64, 0, None])
def test_full_lowercase_preimage_hash_is_required(value):
    transport = packed()
    transport["slots"][0][3] = value
    with pytest.raises(wire.PythonCodeViewWireError):
        wire.unpack_python_code_view_packet(transport)


@pytest.mark.parametrize("value", ["../escape.py", "/absolute.py", "src//file.py", "src\\file.py", "has space.py", ""])
def test_canonical_bound_path_contract_survives_wire(value):
    transport = packed()
    transport["bindings"][0] = value
    with pytest.raises(wire.PythonCodeViewWireError):
        wire.unpack_python_code_view_packet(transport)


@pytest.mark.parametrize("mutation", ["legend", "hash", "source", "binding"])
def test_structural_roundtrip_does_not_authenticate_forged_public_values(mutation):
    view = make()
    packet = view.model_packet()
    if mutation == "legend":
        packet["functions"][0]["aliases"][0]["original"] = "different_long_semantic_name"
    elif mutation == "hash":
        packet["slots"][0]["before_sha256"] = "0" * 64
    elif mutation == "source":
        packet["compact_source"] = packet["compact_source"].replace("//3", "//7")
    else:
        packet["bindings"]["task_id"] = "task:other"
    transport, _ = wire.pack_python_code_view_packet(packet)
    assert wire.unpack_python_code_view_packet(transport) == packet
    assert packet["view_id"] == view.view_id
    with pytest.raises(wire.PythonCodeViewWireError, match="captured view"):
        wire.require_bound_python_code_view_packet(transport, view=view)
    with pytest.raises(wire.PythonCodeViewWireError, match="captured view"):
        wire.require_bound_python_code_view_packet(packet, view=view)


def test_bound_verification_requires_an_exact_captured_engine_view():
    with pytest.raises(wire.PythonCodeViewWireError, match="captured PythonCodeView"):
        wire.require_bound_python_code_view_packet(public(), view=object())


@pytest.mark.parametrize("representation", ["public", "wire"])
@pytest.mark.parametrize("kind", ["bytes", "nodes", "depth", "surrogate", "integer"])
def test_literal_structure_limits_run_before_source_parse_or_copy(monkeypatch, representation, kind):
    packet = public() if representation == "public" else packed()
    field = "compact_source" if representation == "public" else "source"
    if kind == "bytes":
        packet[field] = "x" * (engine.MAX_ARTIFACT_BYTES + 1)
    elif kind == "nodes":
        packet["unexpected"] = [None] * engine.MAX_ARTIFACT_NODES
    elif kind == "depth":
        nested = None
        for _ in range(engine.MAX_ARTIFACT_DEPTH + 1):
            nested = [nested]
        packet["unexpected"] = nested
    elif kind == "surrogate":
        packet[field] = "\ud800"
    else:
        packet["unexpected"] = 1 << (engine.MAX_INTEGER_DIGITS * 4 + 1)
    monkeypatch.setattr(wire, "_inventory", lambda *args: pytest.fail("source parser reached before resource refusal"))
    function = wire.pack_python_code_view_packet if representation == "public" else wire.unpack_python_code_view_packet
    with pytest.raises(wire.PythonCodeViewWireError):
        function(packet)


@pytest.mark.parametrize("kind", ["functions", "aliases", "slots"])
def test_row_population_is_bounded_before_inventory_parse(monkeypatch, kind):
    transport = packed()
    if kind == "functions":
        transport["functions"] = [["f0", "name", []]] * (engine.MAX_FUNCTIONS + 1)
    elif kind == "aliases":
        transport["functions"][0][2] = [["original", "a", 0]] * (wire.MAX_ALIASES + 1)
    else:
        transport["slots"] = [[]] * (wire.MAX_SLOTS + 1)
    monkeypatch.setattr(wire, "_inventory", lambda *args: pytest.fail("source parser reached before row refusal"))
    with pytest.raises(wire.PythonCodeViewWireError):
        wire.unpack_python_code_view_packet(transport)


def test_repeated_long_function_names_cannot_expand_past_byte_budget(monkeypatch):
    # The source and wire are small enough; repeating a 40 KB function name in
    # 100 expanded slot objects would exceed the existing 2 MB artifact cap.
    function_name = "f" + "x" * 40_000
    source = f"def {function_name}():\n" + "".join(f"    v{index}=1\n" for index in range(99)) + "    return v98\n"
    transport = {
        "schema": wire.WIRE_SCHEMA, "profile": wire.WIRE_PROFILE,
        "view_id": public()["view_id"],
        "bindings": ["test.py", "repository:fixture", "tree:fixture", "task:fixture"],
        "source": source,
        "functions": [["f0", function_name, [[f"v{index}", f"v{index}", 1] for index in range(99)]]],
        "slots": [[f"s{index}", 0, 0 if index < 99 else 1, "0" * 64, True, None] for index in range(100)],
    }
    assert len(encoded(transport)) < engine.MAX_ARTIFACT_BYTES
    assert len(source.encode()) < engine.MAX_SOURCE_BYTES
    monkeypatch.setattr(wire, "_checked_model", lambda *args: pytest.fail("expanded packet materialized before virtual bound"))
    with pytest.raises(wire.PythonCodeViewWireError, match="virtual packet expansion"):
        wire.unpack_python_code_view_packet(transport)


def test_complete_public_packet_bytes_are_measured_without_forcing_selection(record_property):
    packet = public()
    transport, stats = wire.pack_python_code_view_packet(packet)
    literal = encoded(packet)
    compact = encoded(transport)
    assert wire.unpack_python_code_view_packet(transport) == packet
    probe = {
        "schema": "supervisor-python-code-view-wire-probe@1",
        "public_packet_bytes": len(literal), "wire_packet_bytes": len(compact),
        "public_sha256": hashlib.sha256(literal).hexdigest(),
        "wire_sha256": hashlib.sha256(compact).hexdigest(),
        "byte_delta": len(compact) - len(literal), "input_tokens": None,
        "provider_calls": 0, "lean_admits": 0, "selection_forced": False,
    }
    assert stats["public_packet_bytes"] == probe["public_packet_bytes"]
    assert stats["wire_packet_bytes"] == probe["wire_packet_bytes"]
    record_property("python_code_view_wire_probe", json.dumps(probe, sort_keys=True))
