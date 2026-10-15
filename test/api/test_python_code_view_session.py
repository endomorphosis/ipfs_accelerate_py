"""Session metadata reuse preserves exact public semantics without authority."""

from __future__ import annotations

import copy
import json

import pytest
from ipfs_accelerate_py.agent_supervisor.semantic_refactoring import python_code_view as engine
from ipfs_accelerate_py.agent_supervisor.semantic_refactoring import (
    python_code_view_session as session,
)
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


def packet(source=SOURCE):
    return wire.pack_python_code_view_packet(make(source).model_packet())[0]


def split(source=SOURCE):
    return session.split_python_code_view_session(packet(source))[:2]


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def cid(legend):
    return engine.cid_for_bytes(encoded({key: value for key, value in legend.items() if key != "legend_id"}), codec="dag-json")


def rebound(legend, state):
    legend["legend_id"] = state["legend_id"] = cid(legend)


def at(value, path):
    for key in path:
        value = value[key]
    return value


@pytest.mark.parametrize("ending", ["\n", "\r\n"])
@pytest.mark.parametrize("final_newline", [True, False])
def test_lossless_exact_source_names_slots_identity_and_refusal(ending, final_newline):
    source = SOURCE.replace("\n", ending)
    if not final_newline:
        source = source.removesuffix(ending)
    original = packet(source)
    original_bytes = encoded(original)
    legend, state, stats = session.split_python_code_view_session(original)
    restored = session.restore_python_code_view_session(legend, state)
    assert restored == original and encoded(restored) == original_bytes
    assert state["source"].encode() == original["source"].encode()
    assert "café λ" in state["source"] and "0xA" in state["source"]
    assert legend["functions"] == original["functions"]
    assert legend["functions"][0][2][0] == ["quantity_total", "a", 0]
    assert legend["slots"][-1] == ["s2", 1, 1, False, wire.REFUSAL_REASON]
    assert state["hashes"] == [slot[3] for slot in original["slots"]]
    assert state["view_id"] == original["view_id"]
    assert legend["legend_id"] == state["legend_id"] == cid(legend)
    assert stats["wire_packet_bytes"] == len(original_bytes)
    assert stats["legend_bytes"] == len(encoded(legend))
    assert stats["state_bytes"] == len(encoded(state))
    assert stats["initial_session_bytes"] == len(encoded(legend)) + len(encoded(state))
    assert stats["initial_byte_delta"] == stats["initial_session_bytes"] - len(original_bytes)
    assert stats["lossless"] is True and stats["input_tokens"] is None
    assert stats["execution_authority"] is stats["proof_authority"] is stats["qualification_authority"] is False
    assert wire.require_bound_python_code_view_packet(restored, view=make(source)) == make(source).model_packet()


def test_real_explicit_edit_reuses_exact_legend_and_updates_full_state():
    initial = make()
    first_packet = wire.pack_python_code_view_packet(initial.model_packet())[0]
    first_legend, first_state, _ = session.split_python_code_view_session(first_packet)
    slot = initial.to_dict()["slots"][0]
    edit = engine.replace_python_expression(
        view=initial, slot_id=slot["slot_id"], before_sha256=slot["before_sha256"],
        expression="a + b * 0xB + 12345", current_source=SOURCE,
    )
    assert "subtotal_amount = quantity_total + tax_amount * 0xB + 12345" in edit.after_source
    changed = make(edit.after_source)
    second_packet = wire.pack_python_code_view_packet(changed.model_packet())[0]
    second_legend, second_state, _ = session.split_python_code_view_session(second_packet)
    assert first_legend == second_legend
    assert first_state["legend_id"] == second_state["legend_id"]
    assert first_state["view_id"] != second_state["view_id"]
    assert first_state["source"] != second_state["source"]
    assert first_state["hashes"][0] != second_state["hashes"][0]
    assert first_state["hashes"][1:] == second_state["hashes"][1:]
    assert session.restore_python_code_view_session(first_legend, first_state) == first_packet
    assert session.restore_python_code_view_session(first_legend, second_state) == second_packet
    assert wire.require_bound_python_code_view_packet(
        session.restore_python_code_view_session(first_legend, second_state), view=changed,
    ) == changed.model_packet()


def test_metadata_change_gets_different_full_legend_and_old_state_refuses():
    first_legend, first_state = split()
    source = SOURCE.replace("quantity_total", "input_quantity_total")
    changed_legend, changed_state = split(source)
    assert first_legend["legend_id"] != changed_legend["legend_id"]
    assert session.restore_python_code_view_session(changed_legend, changed_state) == packet(source)
    with pytest.raises(session.PythonCodeViewSessionError, match="different full legend"):
        session.restore_python_code_view_session(first_legend, changed_state)
    with pytest.raises(session.PythonCodeViewSessionError, match="different full legend"):
        session.restore_python_code_view_session(changed_legend, first_state)


def test_no_private_source_authority_or_compact_identity_substitution():
    legend, state = split()
    assert set(legend) == session.LEGEND_FIELDS
    assert set(state) == session.STATE_FIELDS
    assert legend["schema"] == session.LEGEND_SCHEMA
    assert state["schema"] == session.STATE_SCHEMA
    assert legend["profile"] == state["profile"] == session.SESSION_PROFILE
    serialized = encoded([legend, state])
    assert SOURCE.encode() not in serialized
    for field in ("original_source", "source_span", "token_map", "original_expression", "authority"):
        assert field.encode() not in serialized
    assert len(state["legend_id"]) == len(state["view_id"])


def test_outputs_and_restored_packet_are_independent_of_caller_mutation():
    original = packet()
    legend, state, _ = session.split_python_code_view_session(original)
    captured_legend, captured_state = copy.deepcopy(legend), copy.deepcopy(state)
    restored = session.restore_python_code_view_session(legend, state)
    original["functions"][0][2][0][0] = "changed_input"
    original["slots"][0][3] = "0" * 64
    assert legend == captured_legend and state == captured_state
    restored["functions"][0][2][0][0] = "changed_output"
    restored["slots"][0][3] = "1" * 64
    assert legend == captured_legend and state == captured_state
    legend["functions"][0][2][0][0] = "changed_legend"
    state["hashes"][0] = "2" * 64
    assert restored["functions"][0][2][0][0] == "changed_output"
    assert restored["slots"][0][3] == "1" * 64


@pytest.mark.parametrize("which", ["legend", "state"])
@pytest.mark.parametrize("field,value", [
    ("schema", "unknown@1"), ("schema", True),
    ("profile", "python-arithmetic-model-packet-session-columns@2"), ("profile", 1),
    ("legend_id", "not-a-cid"), ("legend_id", True),
])
def test_unknown_schema_profile_and_identity_types_refuse(which, field, value):
    legend, state = split()
    (legend if which == "legend" else state)[field] = value
    with pytest.raises(session.PythonCodeViewSessionError):
        session.restore_python_code_view_session(legend, state)


@pytest.mark.parametrize("which,fields", [("legend", session.LEGEND_FIELDS), ("state", session.STATE_FIELDS)])
def test_every_field_is_required(which, fields):
    for field in fields:
        legend, state = split()
        del (legend if which == "legend" else state)[field]
        with pytest.raises(session.PythonCodeViewSessionError):
            session.restore_python_code_view_session(legend, state)


@pytest.mark.parametrize("which", ["legend", "state"])
@pytest.mark.parametrize("field", ["authority", "source_write_authority", "columns", "original_source", "extensions"])
def test_closed_profiles_reject_extra_fields_even_if_they_are_false(which, field):
    legend, state = split()
    (legend if which == "legend" else state)[field] = False
    with pytest.raises(session.PythonCodeViewSessionError):
        session.restore_python_code_view_session(legend, state)


@pytest.mark.parametrize("path", [("bindings",), ("functions", 0), ("functions", 0, 2, 0), ("slots", 0)])
@pytest.mark.parametrize("operation", ["short", "long", "tuple"])
def test_rows_require_exact_list_type_and_width(path, operation):
    legend, state = split()
    parent, key = at(legend, path[:-1]), path[-1]
    row = parent[key]
    parent[key] = row[:-1] if operation == "short" else ([*row, "extra"] if operation == "long" else tuple(row))
    with pytest.raises(session.PythonCodeViewSessionError):
        session.restore_python_code_view_session(legend, state)


@pytest.mark.parametrize("path", [("slots", 0, 1), ("slots", 0, 2), ("functions", 0, 2, 0, 2)])
@pytest.mark.parametrize("value", [True, False, -1, 1000, "0", 0.0, None])
def test_boolean_or_out_of_range_indices_never_change_semantics(path, value):
    legend, state = split()
    at(legend, path[:-1])[path[-1]] = value
    with pytest.raises(session.PythonCodeViewSessionError):
        session.restore_python_code_view_session(legend, state)


@pytest.mark.parametrize("change", ["semantic_name", "binding", "refusal", "slot_order"])
def test_legend_content_cannot_change_under_an_existing_cid(change):
    legend, state = split()
    if change == "semantic_name":
        legend["functions"][0][2][0][0] = "different_long_original_name"
    elif change == "binding":
        legend["bindings"][0] = "src/different.py"
    elif change == "refusal":
        legend["slots"][-1][3:] = [True, None]
    else:
        legend["slots"][0], legend["slots"][1] = legend["slots"][1], legend["slots"][0]
    with pytest.raises(session.PythonCodeViewSessionError):
        session.restore_python_code_view_session(legend, state)


@pytest.mark.parametrize("value", ["0" * 63, "A" * 64, "z" * 64, 1, None])
def test_state_retains_full_exact_lowercase_preimages(value):
    legend, state = split()
    state["hashes"][0] = value
    with pytest.raises(session.PythonCodeViewSessionError):
        session.restore_python_code_view_session(legend, state)


@pytest.mark.parametrize("operation", ["short", "long", "tuple", "empty"])
def test_hash_population_never_omits_or_adds_slots(operation):
    legend, state = split()
    hashes = state["hashes"]
    state["hashes"] = hashes[:-1] if operation == "short" else (
        [*hashes, hashes[0]] if operation == "long" else (tuple(hashes) if operation == "tuple" else [])
    )
    with pytest.raises(session.PythonCodeViewSessionError):
        session.restore_python_code_view_session(legend, state)


def test_structural_transport_never_authenticates_forged_hashes_or_view_identity():
    view = make()
    legend, state = split()
    state["hashes"][0], state["hashes"][1] = state["hashes"][1], state["hashes"][0]
    state["view_id"] = engine.cid_for_dag_json({"not": "a captured view"})
    restored = session.restore_python_code_view_session(legend, state)
    assert [slot[3] for slot in restored["slots"]] == state["hashes"]
    with pytest.raises(wire.PythonCodeViewWireError, match="captured view"):
        wire.require_bound_python_code_view_packet(restored, view=view)


@pytest.mark.parametrize("change", ["function_name", "alias", "slot_reference", "slot_kind", "source"])
def test_recomputed_cid_still_requires_source_inventory_and_scope(change):
    legend, state = split()
    if change == "function_name":
        legend["functions"][0][1] = "different_function"
    elif change == "alias":
        legend["functions"][0][2][0][1] = "z"
    elif change == "slot_reference":
        legend["slots"][0][1] = 1
    elif change == "slot_kind":
        legend["slots"][0][2] = 1
    else:
        state["source"] = state["source"].replace("def calculate_total", "def other_name")
    rebound(legend, state)
    with pytest.raises(session.PythonCodeViewSessionError):
        session.restore_python_code_view_session(legend, state)


@pytest.mark.parametrize("which", ["legend", "state"])
def test_json_parsers_preserve_canonical_bytes_and_detect_duplicate_fields(which):
    legend, state = split()
    value = legend if which == "legend" else state
    parser = session.parse_python_code_view_session_legend if which == "legend" else session.parse_python_code_view_session_state
    text = encoded(value).decode()
    parsed = parser(text)
    assert parsed == value and encoded(parsed).decode() == text
    parsed["profile"] = "changed"
    assert value["profile"] == session.SESSION_PROFILE
    duplicate = '{"schema":"discarded",' + text[1:]
    with pytest.raises(session.PythonCodeViewSessionError):
        parser(duplicate)
    with pytest.raises(session.PythonCodeViewSessionError, match="canonical"):
        parser(" " + text)
    with pytest.raises(session.PythonCodeViewSessionError, match="canonical"):
        parser(json.dumps(value, ensure_ascii=False, indent=2))


@pytest.mark.parametrize("which", ["legend", "state"])
@pytest.mark.parametrize("suffix", ["=", "a", "\n"])
def test_full_cids_reject_noncanonical_encodings_and_extra_bytes(which, suffix):
    legend, state = split()
    target = legend if which == "legend" else state
    target["legend_id"] += suffix
    with pytest.raises(session.PythonCodeViewSessionError):
        session.restore_python_code_view_session(legend, state)


def test_non_json_subclasses_cycles_depth_and_node_excess_refuse_before_copy(monkeypatch):
    class Impostor(dict):
        pass

    legend, state = split()
    broken_values = [
        Impostor(state), {**state, "hashes": state["hashes"] * wire.MAX_SLOTS},
        {**state, "hashes": [None] * engine.MAX_ARTIFACT_NODES},
    ]
    deep = []
    for _ in range(engine.MAX_ARTIFACT_DEPTH + 1):
        deep = [deep]
    cycle = []
    cycle.append(cycle)
    broken_values.extend([{**state, "hashes": deep}, {**state, "hashes": cycle}])

    def forbidden_copy(*args, **kwargs):
        raise AssertionError("unbounded or malformed state was copied")

    monkeypatch.setattr(session.copy, "deepcopy", forbidden_copy)
    for broken in broken_values:
        with pytest.raises(session.PythonCodeViewSessionError):
            session.restore_python_code_view_session(legend, broken)


@pytest.mark.parametrize("source", ["a" * (engine.MAX_SOURCE_BYTES + 1), "λ" * engine.MAX_SOURCE_BYTES, "\ud800"])
def test_literal_source_bytes_are_bounded_before_copy(source, monkeypatch):
    legend, state = split()
    state["source"] = source

    def forbidden_copy(*args, **kwargs):
        raise AssertionError("unbounded literal source was copied")

    monkeypatch.setattr(session.copy, "deepcopy", forbidden_copy)
    with pytest.raises(session.PythonCodeViewSessionError):
        session.restore_python_code_view_session(legend, state)


def test_virtual_model_expansion_is_bounded_before_repeated_metadata_copy(monkeypatch):
    name = "f" * 60000
    count = 80
    source = f"def {name}(z):\n" + "".join(f" a{index} = z\n" for index in range(count)) + f" return a{count - 1}\n"
    identity = engine.cid_for_dag_json({"synthetic": "transport only"})
    legend = {
        "schema": session.LEGEND_SCHEMA, "profile": session.SESSION_PROFILE,
        "bindings": ["calc.py", "repo:fixture", "tree:fixture", "task:fixture"],
        "functions": [["f0", name, [["z", "z", 0], *[[f"a{index}", f"a{index}", 1] for index in range(count)]]]],
        "slots": [[f"s{index}", 0, 0 if index < count else 1, True, None] for index in range(count + 1)],
    }
    legend["legend_id"] = cid(legend)
    state = {
        "schema": session.STATE_SCHEMA, "profile": session.SESSION_PROFILE,
        "view_id": identity, "legend_id": legend["legend_id"], "source": source,
        "hashes": ["0" * 64] * (count + 1),
    }
    assert len(source.encode()) < engine.MAX_SOURCE_BYTES
    assert len(encoded(legend)) < engine.MAX_ARTIFACT_BYTES
    assert len(encoded(state)) < engine.MAX_ARTIFACT_BYTES

    def forbidden_copy(*args, **kwargs):
        raise AssertionError("virtual expansion copied repeated metadata")

    monkeypatch.setattr(session.copy, "deepcopy", forbidden_copy)
    with pytest.raises(session.PythonCodeViewSessionError, match="virtual packet expansion"):
        session.restore_python_code_view_session(legend, state)


@pytest.mark.parametrize("field,value", [("schema", "unknown@1"), ("profile", "unknown@1"), ("source", 1)])
def test_split_requires_valid_closed_wire_input(field, value):
    value_packet = packet()
    value_packet[field] = value
    with pytest.raises(session.PythonCodeViewSessionError):
        session.split_python_code_view_session(value_packet)
