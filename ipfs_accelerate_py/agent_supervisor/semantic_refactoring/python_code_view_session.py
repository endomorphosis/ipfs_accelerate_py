"""Pure, bounded legend reuse for the closed Python code-view wire profile.

The legend's full CID binds its exact generation profile and ordered metadata.
Each state retains the full view identity, literal source and every slot's
original-expression hash. Content addressing is transport integrity, never
native source visibility, semantic equivalence, execution or qualification.
No session cache, filesystem access, source writes or model authority live here.
"""

from __future__ import annotations

import copy
from typing import Any

from . import python_code_view as engine
from . import python_code_view_wire as wire

LEGEND_SCHEMA = "supervisor-python-code-view-session-legend@1"
STATE_SCHEMA = "supervisor-python-code-view-session-state@1"
SESSION_PROFILE = "python-arithmetic-model-packet-session-columns@1"
LEGEND_FIELDS = frozenset({
    "schema", "profile", "legend_id", "bindings", "functions", "slots",
})
STATE_FIELDS = frozenset({
    "schema", "profile", "view_id", "legend_id", "source", "hashes",
})
LEGEND_SLOT_WIDTH = 5


class PythonCodeViewSessionError(ValueError):
    """A closed session packet, identity or expansion bound failed."""


def _fail(message: str) -> None:
    raise PythonCodeViewSessionError(message)


def _call(function: Any, *args: Any, **kwargs: Any) -> Any:
    try:
        return function(*args, **kwargs)
    except (engine.PythonCodeViewError, wire.PythonCodeViewWireError) as error:
        raise PythonCodeViewSessionError(str(error)) from error


def _canonical(value: Any) -> bytes:
    return _call(engine._canonical, value)


def _schema(packet: Any, fields: Any, schema: str, label: str) -> None:
    _call(engine._profile, packet)
    _call(wire._object, packet, fields, label)
    if (
        type(packet["schema"]) is not str or packet["schema"] != schema
        or type(packet["profile"]) is not str or packet["profile"] != SESSION_PROFILE
    ):
        _fail(f"unsupported {label} schema or profile")


def _legend_cid(legend: dict) -> str:
    # The self-identity is deliberately excluded; all other closed fields bind.
    body = {key: value for key, value in legend.items() if key != "legend_id"}
    return engine.cid_for_bytes(_canonical(body), codec="dag-json")


def _checked_legend(legend: Any) -> None:
    _schema(legend, LEGEND_FIELDS, LEGEND_SCHEMA, "session legend")
    _call(wire._identity, legend["legend_id"])
    _call(wire._row, legend["bindings"], len(wire.BINDING_COLUMNS), "bindings")
    _call(wire._bindings, dict(zip(wire.BINDING_COLUMNS, legend["bindings"], strict=True)))
    _call(wire._rows, legend["functions"], engine.MAX_FUNCTIONS, "function rows")
    _call(wire._rows, legend["slots"], wire.MAX_SLOTS, "legend slot rows")
    if not legend["functions"] or not legend["slots"]:
        _fail("the closed profile requires functions and slots")
    names: set[str] = set()
    alias_count = 0
    for index, function in enumerate(legend["functions"]):
        _call(wire._row, function, len(wire.FUNCTION_COLUMNS), "function row")
        function_id, name, aliases = function
        if type(function_id) is not str or function_id != f"f{index}":
            _fail("function IDs must preserve the generated order")
        _call(wire._name, name)
        if name in names:
            _fail("function names must be unique")
        names.add(name)
        _call(wire._rows, aliases, wire.MAX_ALIASES, "alias rows")
        alias_count += len(aliases)
        if alias_count > wire.MAX_ALIASES:
            _fail("total alias population exceeds the source token bound")
        originals: set[str] = set()
        spellings: set[str] = set()
        for alias in aliases:
            _call(wire._row, alias, len(wire.ALIAS_COLUMNS), "alias row")
            original, spelling, kind = alias
            _call(wire._name, original)
            _call(wire._name, spelling)
            _call(wire._index, kind, len(wire.ALIAS_KINDS), "alias kind")
            if original in originals or spelling in spellings or len(spelling) > len(original):
                _fail("alias legends must be scoped nonexpanding bijections")
            originals.add(original)
            spellings.add(spelling)
    for index, slot in enumerate(legend["slots"]):
        _call(wire._row, slot, LEGEND_SLOT_WIDTH, "legend slot row")
        slot_id, reference, kind, editable, reason = slot
        if type(slot_id) is not str or slot_id != f"s{index}":
            _fail("slot IDs must preserve the generated order")
        _call(wire._index, reference, len(legend["functions"]), "slot function reference")
        _call(wire._index, kind, len(wire.SLOT_KINDS), "slot kind")
        _call(wire._editable, editable, reason)
    if _legend_cid(legend) != legend["legend_id"]:
        _fail("legend identity differs from its exact canonical content")


def _checked_state(state: Any) -> None:
    _schema(state, STATE_FIELDS, STATE_SCHEMA, "session state")
    _call(wire._identity, state["view_id"])
    _call(wire._identity, state["legend_id"])
    _call(engine._raw, state["source"], engine.MAX_SOURCE_BYTES, "compact source")
    _call(wire._rows, state["hashes"], wire.MAX_SLOTS, "slot preimage hashes")
    if not state["hashes"]:
        _fail("the closed profile requires slot preimage hashes")
    for before_hash in state["hashes"]:
        _call(wire._hash, before_hash)


def _measure_session_input(packet: dict) -> None:
    """Measure both new objects before copying metadata or serializing them."""
    legend = wire._Measure()
    legend.obj(LEGEND_FIELDS, 0)
    for value in (LEGEND_SCHEMA, SESSION_PROFILE, packet["view_id"]):
        # Every CID in this closed profile has the same full encoded width.
        legend.value(value, 1)
    legend.array(len(wire.BINDING_COLUMNS), 1)
    for value in packet["bindings"]:
        legend.value(value, 2)
    _measure_functions(legend, packet["functions"], wire_form=True)
    legend.array(len(packet["slots"]), 1)
    for slot_id, function, kind, _, editable, reason in packet["slots"]:
        legend.array(LEGEND_SLOT_WIDTH, 2)
        for value in (slot_id, function, kind, editable, reason):
            legend.value(value, 3)
    state = wire._Measure()
    state.obj(STATE_FIELDS, 0)
    for value in (STATE_SCHEMA, SESSION_PROFILE, packet["view_id"], packet["view_id"], packet["source"]):
        state.value(value, 1)
    state.array(len(packet["slots"]), 1)
    for slot in packet["slots"]:
        state.value(slot[3], 2)


def _measure_functions(budget: wire._Measure, functions: list, *, wire_form: bool) -> None:
    budget.array(len(functions), 1)
    for function_id, name, aliases in functions:
        if wire_form:
            budget.array(len(wire.FUNCTION_COLUMNS), 2)
        else:
            budget.obj(wire.FUNCTION_COLUMNS, 2)
        budget.value(function_id, 3)
        budget.value(name, 3)
        budget.array(len(aliases), 3)
        for original, alias, kind in aliases:
            if wire_form:
                budget.array(len(wire.ALIAS_COLUMNS), 4)
            else:
                budget.obj(wire.ALIAS_COLUMNS, 4)
            for value in (original, alias, kind if wire_form else wire.ALIAS_KINDS[kind]):
                budget.value(value, 5)


def _measure_expansion(legend: dict, state: dict) -> None:
    """Measure virtual WIRE and MODEL outputs before building repeated fields."""
    transport = wire._Measure()
    transport.obj(wire.WIRE_FIELDS, 0)
    for value in (wire.WIRE_SCHEMA, wire.WIRE_PROFILE, state["view_id"], state["source"]):
        transport.value(value, 1)
    transport.array(len(wire.BINDING_COLUMNS), 1)
    for value in legend["bindings"]:
        transport.value(value, 2)
    _measure_functions(transport, legend["functions"], wire_form=True)
    transport.array(len(legend["slots"]), 1)
    for slot, before_hash in zip(legend["slots"], state["hashes"], strict=True):
        slot_id, function, kind, editable, reason = slot
        transport.array(len(wire.SLOT_COLUMNS), 2)
        for value in (slot_id, function, kind, before_hash, editable, reason):
            transport.value(value, 3)
    model = wire._Measure()
    model.obj(wire.MODEL_FIELDS, 0)
    for value in (wire.MODEL_SCHEMA, engine.VIEW_REVISION, state["view_id"], state["source"]):
        model.value(value, 1)
    model.obj(wire.BINDING_COLUMNS, 1)
    for value in legend["bindings"]:
        model.value(value, 2)
    _measure_functions(model, legend["functions"], wire_form=False)
    model.array(len(legend["slots"]), 1)
    for slot, before_hash in zip(legend["slots"], state["hashes"], strict=True):
        slot_id, function, kind, editable, reason = slot
        model.obj(wire.SLOT_COLUMNS, 2)
        for value in (slot_id, legend["functions"][function][1], wire.SLOT_KINDS[kind], before_hash, editable, reason):
            model.value(value, 3)
    model.obj(wire.AUTHORITY_COLUMNS, 1)
    for _ in wire.AUTHORITY_COLUMNS:
        model.value(False, 2)


def _checked_inventory(legend: dict, state: dict) -> None:
    inventory = _call(wire._inventory, state["source"])
    if len(inventory) != len(legend["functions"]):
        _fail("function inventory differs from compact source")
    expected_slots: list[tuple[int, str]] = []
    for index, (function, actual) in enumerate(zip(legend["functions"], inventory, strict=True)):
        if function[1] != actual["node"].name:
            _fail("function name differs from compact source")
        _call(wire._aliases, function[2], actual, wire=True)
        expected_slots.extend((index, slot[2]) for slot in actual["slots"])
    if len(expected_slots) != len(legend["slots"]):
        _fail("slot inventory differs from compact source")
    for slot, (function, kind) in zip(legend["slots"], expected_slots, strict=True):
        if slot[1] != function or wire.SLOT_KINDS[slot[2]] != kind:
            _fail("slot order or function reference differs from compact source")


def split_python_code_view_session(wire_packet: dict) -> tuple[dict, dict, dict]:
    """Split a valid exact WIRE packet; independent outputs omit no values."""
    _call(wire.unpack_python_code_view_packet, wire_packet)
    _call(_measure_session_input, wire_packet)
    legend = {
        "schema": LEGEND_SCHEMA, "profile": SESSION_PROFILE,
        "bindings": list(wire_packet["bindings"]),
        "functions": copy.deepcopy(wire_packet["functions"]),
        "slots": [[slot_id, function, kind, editable, reason]
                  for slot_id, function, kind, _, editable, reason in wire_packet["slots"]],
    }
    legend["legend_id"] = _legend_cid(legend)
    state = {
        "schema": STATE_SCHEMA, "profile": SESSION_PROFILE,
        "view_id": wire_packet["view_id"], "legend_id": legend["legend_id"],
        "source": wire_packet["source"], "hashes": [slot[3] for slot in wire_packet["slots"]],
    }
    _checked_legend(legend)
    _checked_state(state)
    wire_bytes, legend_bytes, state_bytes = map(len, (
        _canonical(wire_packet), _canonical(legend), _canonical(state),
    ))
    return legend, state, {
        "schema": "supervisor-python-code-view-session-stats@1",
        "profile": SESSION_PROFILE, "legend_id": legend["legend_id"],
        "wire_packet_bytes": wire_bytes, "legend_bytes": legend_bytes,
        "state_bytes": state_bytes, "initial_session_bytes": legend_bytes + state_bytes,
        "initial_byte_delta": legend_bytes + state_bytes - wire_bytes,
        "lossless": True, "input_tokens": None,
        "execution_authority": False, "proof_authority": False,
        "qualification_authority": False,
    }


def restore_python_code_view_session(legend: dict, state: dict) -> dict:
    """Restore exact WIRE semantics only under the state's full legend CID."""
    _checked_legend(legend)
    _checked_state(state)
    if legend["legend_id"] != state["legend_id"]:
        _fail("state refers to a different full legend identity")
    if len(legend["slots"]) != len(state["hashes"]):
        _fail("slot hashes must preserve the complete ordered slot population")
    _checked_inventory(legend, state)
    _call(_measure_expansion, legend, state)
    restored = {
        "schema": wire.WIRE_SCHEMA, "profile": wire.WIRE_PROFILE,
        "view_id": state["view_id"], "bindings": list(legend["bindings"]),
        "source": state["source"], "functions": copy.deepcopy(legend["functions"]),
        "slots": [[slot_id, function, kind, before_hash, editable, reason]
                  for (slot_id, function, kind, editable, reason), before_hash
                  in zip(legend["slots"], state["hashes"], strict=True)],
    }
    _call(wire.unpack_python_code_view_packet, restored)
    return restored


def _parse(text: str, checker: Any) -> dict:
    value = _call(engine._read_json, text)
    checker(value)
    if _canonical(value) != _call(engine._raw, text, engine.MAX_ARTIFACT_BYTES, "session JSON"):
        _fail("session JSON must use the exact canonical encoding")
    return value


def parse_python_code_view_session_legend(text: str) -> dict:
    """Read canonical duplicate-free JSON, preserving the full legend CID."""
    return _parse(text, _checked_legend)


def parse_python_code_view_session_state(text: str) -> dict:
    """Read canonical duplicate-free JSON; restoration checks legend binding."""
    return _parse(text, _checked_state)
