"""Lossless, bounded columns for the closed Python code-view public packet.

Only representation changes: literal source, semantic alias legends, slot
identities, and every public value survive expansion. A view ID alone does not
authenticate a packet. Native source visibility, live leases, edit inversion,
execution, and qualification remain the existing controller's responsibility.
"""

from __future__ import annotations

from typing import Any

from ipfs_accelerate_py.utils.cid_utils import validate_cid

from . import python_code_view as engine

WIRE_SCHEMA = "supervisor-python-code-view-wire@1"
WIRE_PROFILE = "python-arithmetic-model-packet-columns@1"
MODEL_SCHEMA = "supervisor-python-code-view-model-packet@1"
BINDING_COLUMNS = ("path", "repository_id", "tree_id", "task_id")
FUNCTION_COLUMNS = ("function_id", "name", "aliases")
ALIAS_COLUMNS = ("original", "alias", "kind")
SLOT_COLUMNS = ("slot_id", "function", "kind", "before_sha256", "editable", "refusal_reason")
ALIAS_KINDS = ("parameter", "local")
SLOT_KINDS = ("assignment", "return")
AUTHORITY_COLUMNS = (
    "source_write_authority", "execution_authority", "proof_authority",
    "qualification_authority", "completion_authority",
)
MODEL_FIELDS = frozenset({
    "schema", "revision", "view_id", "bindings", "compact_source",
    "functions", "slots", "authority",
})
WIRE_FIELDS = frozenset({
    "schema", "profile", "view_id", "bindings", "source", "functions", "slots",
})
MAX_ALIASES = engine.MAX_TOKENS
MAX_SLOTS = engine.MAX_AST_NODES
REFUSAL_REASON = "multiline_or_protected_expression_trivia"


class PythonCodeViewWireError(ValueError):
    """A closed packet, positional row, or expansion bound was violated."""


def _fail(message: str) -> None:
    raise PythonCodeViewWireError(message)


def _profile(value: Any) -> None:
    try:
        engine._profile(value)
    except engine.PythonCodeViewError as error:
        raise PythonCodeViewWireError(str(error)) from error


def _raw(value: Any, limit: int, label: str) -> bytes:
    try:
        return engine._raw(value, limit, label)
    except engine.PythonCodeViewError as error:
        raise PythonCodeViewWireError(str(error)) from error


def _canonical(value: Any) -> bytes:
    try:
        return engine._canonical(value)
    except engine.PythonCodeViewError as error:
        raise PythonCodeViewWireError(str(error)) from error


def _object(value: Any, fields: Any, label: str) -> None:
    if type(value) is not dict or set(value) != set(fields):
        _fail(f"{label} must have exactly the closed fields")


def _rows(value: Any, maximum: int, label: str) -> None:
    if type(value) is not list or len(value) > maximum:
        _fail(f"{label} must be a bounded exact list")


def _row(value: Any, width: int, label: str) -> None:
    if type(value) is not list or len(value) != width:
        _fail(f"{label} has an invalid positional width")


def _name(value: Any) -> None:
    _raw(value, engine.MAX_SOURCE_BYTES, "binding name")
    try:
        engine._name(value)
    except engine.PythonCodeViewError as error:
        raise PythonCodeViewWireError(str(error)) from error


def _identity(value: Any) -> None:
    _raw(value, 128, "view identity")
    try:
        validate_cid(value, codecs=("dag-json",))
    except (ValueError, TypeError) as error:
        raise PythonCodeViewWireError("view identity must be a canonical DAG-JSON CID") from error


def _hash(value: Any) -> None:
    if (
        type(value) is not str or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        _fail("slot preimage must be an exact lowercase SHA-256")


def _editable(editable: Any, reason: Any) -> None:
    if type(editable) is not bool:
        _fail("editable must be an exact boolean")
    if (editable and reason is not None) or (
        not editable and (type(reason) is not str or reason != REFUSAL_REASON)
    ):
        _fail("editable and refusal reason violate the closed generation profile")


def _index(value: Any, length: int, label: str) -> None:
    if type(value) is not int or not 0 <= value < length:
        _fail(f"{label} must be an exact in-range integer")


def _inventory(source: Any) -> list[dict[str, Any]]:
    _raw(source, engine.MAX_SOURCE_BYTES, "compact source")
    try:
        if "\r" in source.replace("\r\n", ""):
            _fail("standalone carriage returns are outside the source profile")
        engine._tokens(source)
        return engine._functions(engine._tree(source))
    except engine.PythonCodeViewError as error:
        raise PythonCodeViewWireError(str(error)) from error


def _bindings(value: Any) -> None:
    try:
        engine._bindings(value)
    except engine.PythonCodeViewError as error:
        raise PythonCodeViewWireError(str(error)) from error


def _aliases(records: list, inventory: dict, *, wire: bool) -> None:
    _rows(records, MAX_ALIASES, "alias rows")
    expected = [
        *((name, "parameter") for name in inventory["parameters"]),
        *((name, "local") for name in inventory["locals"]),
    ]
    if len(records) != len(expected):
        _fail("alias rows differ from compact source binding inventory")
    originals: set[str] = set()
    aliases: set[str] = set()
    for record, (expected_alias, expected_kind) in zip(records, expected, strict=True):
        if wire:
            _row(record, len(ALIAS_COLUMNS), "alias row")
            original, alias, kind_index = record
            _index(kind_index, len(ALIAS_KINDS), "alias kind")
            kind = ALIAS_KINDS[kind_index]
        else:
            _object(record, ALIAS_COLUMNS, "alias row")
            original, alias, kind = (record[key] for key in ALIAS_COLUMNS)
            if type(kind) is not str or kind not in ALIAS_KINDS:
                _fail("alias kind is outside the closed profile")
        _name(original)
        _name(alias)
        if original in originals or alias in aliases:
            _fail("alias legends must be scoped bijections")
        originals.add(original)
        aliases.add(alias)
        if alias != expected_alias or kind != expected_kind or len(alias) > len(original):
            _fail("alias order, kind, or spelling differs from the compact source")


def _population(functions: Any, slots: Any, *, wire: bool) -> None:
    _rows(functions, engine.MAX_FUNCTIONS, "function rows")
    _rows(slots, MAX_SLOTS, "slot rows")
    if not functions or not slots:
        _fail("the closed profile requires functions and slots")
    alias_count = 0
    for function in functions:
        if wire:
            _row(function, len(FUNCTION_COLUMNS), "function row")
            aliases = function[2]
        else:
            _object(function, FUNCTION_COLUMNS, "function row")
            aliases = function["aliases"]
        _rows(aliases, MAX_ALIASES, "alias rows")
        alias_count += len(aliases)
        if alias_count > MAX_ALIASES:
            _fail("total alias population exceeds the source token bound")


def _records(functions: list, slots: list, inventory: list, *, wire: bool) -> None:
    if len(functions) != len(inventory):
        _fail("function inventory differs from compact source")
    names: dict[str, int] = {}
    expected_slots: list[tuple[int, str]] = []
    for index, (function, actual) in enumerate(zip(functions, inventory, strict=True)):
        if wire:
            function_id, name, aliases = function
        else:
            function_id, name, aliases = (function[key] for key in FUNCTION_COLUMNS)
        if type(function_id) is not str or function_id != f"f{index}":
            _fail("function IDs must preserve the generated order")
        _name(name)
        if name in names or name != actual["node"].name:
            _fail("function names must be unique and match compact source")
        names[name] = index
        _aliases(aliases, actual, wire=wire)
        expected_slots.extend((index, slot[2]) for slot in actual["slots"])
    if len(slots) != len(expected_slots):
        _fail("slot inventory differs from compact source")
    for index, (slot, (function_index, expected_kind)) in enumerate(zip(slots, expected_slots, strict=True)):
        if wire:
            _row(slot, len(SLOT_COLUMNS), "slot row")
            slot_id, reference, kind_index, before_hash, editable, reason = slot
            _index(reference, len(functions), "slot function reference")
            _index(kind_index, len(SLOT_KINDS), "slot kind")
            kind = SLOT_KINDS[kind_index]
        else:
            _object(slot, SLOT_COLUMNS, "slot row")
            slot_id, function_name, kind, before_hash, editable, reason = (slot[key] for key in SLOT_COLUMNS)
            if type(function_name) is not str or function_name not in names:
                _fail("slot refers to an unknown function")
            reference = names[function_name]
            if type(kind) is not str or kind not in SLOT_KINDS:
                _fail("slot kind is outside the closed profile")
        if type(slot_id) is not str or slot_id != f"s{index}":
            _fail("slot IDs must preserve the generated order")
        if reference != function_index or kind != expected_kind:
            _fail("slot order or function reference differs from compact source")
        _hash(before_hash)
        _editable(editable, reason)


class _Measure:
    """Count a virtual output before constructing repeated fields or JSON."""

    def __init__(self) -> None:
        self.nodes = 0
        self.bytes = 0

    def _add(self, amount: int, depth: int) -> None:
        self.nodes += 1
        self.bytes += amount
        if (
            self.nodes > engine.MAX_ARTIFACT_NODES
            or self.bytes > engine.MAX_ARTIFACT_BYTES
            or depth > engine.MAX_ARTIFACT_DEPTH
        ):
            _fail("virtual packet expansion exceeds artifact bounds")

    def obj(self, fields: Any, depth: int) -> None:
        self._add(2 + max(0, len(fields) - 1) + len(fields), depth)
        for key in fields:
            self.bytes += engine._json_string_size(key)
            if self.bytes > engine.MAX_ARTIFACT_BYTES:
                _fail("virtual packet expansion exceeds artifact bounds")

    def array(self, length: int, depth: int) -> None:
        self._add(2 + max(0, length - 1), depth)

    def value(self, value: Any, depth: int) -> None:
        if type(value) is str:
            amount = engine._json_string_size(value)
        elif type(value) is bool:
            amount = 4 if value else 5
        elif value is None:
            amount = 4
        elif type(value) is int:
            amount = len(str(value))
        else:
            _fail("invalid virtual scalar")
        self._add(amount, depth)


def _measure_model(wire: dict) -> None:
    budget = _Measure()
    budget.obj(MODEL_FIELDS, 0)
    for value in (MODEL_SCHEMA, engine.VIEW_REVISION, wire["view_id"], wire["source"]):
        budget.value(value, 1)
    budget.obj(BINDING_COLUMNS, 1)
    for value in wire["bindings"]:
        budget.value(value, 2)
    budget.array(len(wire["functions"]), 1)
    for function_id, name, aliases in wire["functions"]:
        budget.obj(FUNCTION_COLUMNS, 2)
        budget.value(function_id, 3)
        budget.value(name, 3)
        budget.array(len(aliases), 3)
        for original, alias, kind in aliases:
            budget.obj(ALIAS_COLUMNS, 4)
            for value in (original, alias, ALIAS_KINDS[kind]):
                budget.value(value, 5)
    budget.array(len(wire["slots"]), 1)
    for slot_id, function, kind, before_hash, editable, reason in wire["slots"]:
        budget.obj(SLOT_COLUMNS, 2)
        for value in (slot_id, wire["functions"][function][1], SLOT_KINDS[kind], before_hash, editable, reason):
            budget.value(value, 3)
    budget.obj(AUTHORITY_COLUMNS, 1)
    for _ in AUTHORITY_COLUMNS:
        budget.value(False, 2)


def _measure_wire(packet: dict) -> None:
    budget = _Measure()
    names = {function["name"]: index for index, function in enumerate(packet["functions"])}
    budget.obj(WIRE_FIELDS, 0)
    for value in (WIRE_SCHEMA, WIRE_PROFILE, packet["view_id"], packet["compact_source"]):
        budget.value(value, 1)
    budget.array(len(BINDING_COLUMNS), 1)
    for key in BINDING_COLUMNS:
        budget.value(packet["bindings"][key], 2)
    budget.array(len(packet["functions"]), 1)
    for function in packet["functions"]:
        budget.array(len(FUNCTION_COLUMNS), 2)
        budget.value(function["function_id"], 3)
        budget.value(function["name"], 3)
        budget.array(len(function["aliases"]), 3)
        for alias in function["aliases"]:
            budget.array(len(ALIAS_COLUMNS), 4)
            budget.value(alias["original"], 5)
            budget.value(alias["alias"], 5)
            budget.value(ALIAS_KINDS.index(alias["kind"]), 5)
    budget.array(len(packet["slots"]), 1)
    for slot in packet["slots"]:
        budget.array(len(SLOT_COLUMNS), 2)
        for value in (slot["slot_id"], names[slot["function"]], SLOT_KINDS.index(slot["kind"]), slot["before_sha256"], slot["editable"], slot["refusal_reason"]):
            budget.value(value, 3)


def _checked_model(packet: Any) -> None:
    _profile(packet)
    _object(packet, MODEL_FIELDS, "model packet")
    if (
        type(packet["schema"]) is not str or packet["schema"] != MODEL_SCHEMA
        or type(packet["revision"]) is not str or packet["revision"] != engine.VIEW_REVISION
    ):
        _fail("unsupported model packet schema or revision")
    _identity(packet["view_id"])
    _bindings(packet["bindings"])
    _object(packet["authority"], AUTHORITY_COLUMNS, "authority")
    if any(type(value) is not bool or value for value in packet["authority"].values()):
        _fail("wire packets have no authority")
    _population(packet["functions"], packet["slots"], wire=False)
    inventory = _inventory(packet["compact_source"])
    _records(packet["functions"], packet["slots"], inventory, wire=False)


def _pack_checked(packet: dict) -> dict:
    _measure_wire(packet)
    names = {function["name"]: index for index, function in enumerate(packet["functions"])}
    return {
        "schema": WIRE_SCHEMA, "profile": WIRE_PROFILE, "view_id": packet["view_id"],
        "bindings": [packet["bindings"][key] for key in BINDING_COLUMNS],
        "source": packet["compact_source"],
        "functions": [[
            function["function_id"], function["name"],
            [[alias["original"], alias["alias"], ALIAS_KINDS.index(alias["kind"])] for alias in function["aliases"]],
        ] for function in packet["functions"]],
        "slots": [[
            slot["slot_id"], names[slot["function"]], SLOT_KINDS.index(slot["kind"]),
            slot["before_sha256"], slot["editable"], slot["refusal_reason"],
        ] for slot in packet["slots"]],
    }


def pack_python_code_view_packet(packet: dict) -> tuple[dict, dict]:
    """Pack a closed public packet without forcing adoption or omitting values."""
    _checked_model(packet)
    packed = _pack_checked(packet)
    public_bytes, wire_bytes = len(_canonical(packet)), len(_canonical(packed))
    return packed, {
        "schema": "supervisor-python-code-view-wire-stats@1", "profile": WIRE_PROFILE,
        "public_packet_bytes": public_bytes, "wire_packet_bytes": wire_bytes,
        "byte_delta": wire_bytes - public_bytes, "wire_smaller": wire_bytes < public_bytes,
        "function_count": len(packet["functions"]),
        "alias_count": sum(len(function["aliases"]) for function in packet["functions"]),
        "slot_count": len(packet["slots"]), "lossless": True,
        "input_tokens": None, "execution_authority": False,
        "proof_authority": False, "qualification_authority": False,
    }


def unpack_python_code_view_packet(wire_packet: dict) -> dict:
    """Expand only the fixed generation profile, before any native edit gate."""
    _profile(wire_packet)
    _object(wire_packet, WIRE_FIELDS, "wire packet")
    if (
        type(wire_packet["schema"]) is not str or wire_packet["schema"] != WIRE_SCHEMA
        or type(wire_packet["profile"]) is not str or wire_packet["profile"] != WIRE_PROFILE
    ):
        _fail("unsupported wire schema or generation profile")
    _identity(wire_packet["view_id"])
    _row(wire_packet["bindings"], len(BINDING_COLUMNS), "bindings")
    _bindings(dict(zip(BINDING_COLUMNS, wire_packet["bindings"], strict=True)))
    _population(wire_packet["functions"], wire_packet["slots"], wire=True)
    inventory = _inventory(wire_packet["source"])
    _records(wire_packet["functions"], wire_packet["slots"], inventory, wire=True)
    _measure_model(wire_packet)
    restored = {
        "schema": MODEL_SCHEMA, "revision": engine.VIEW_REVISION,
        "view_id": wire_packet["view_id"],
        "bindings": dict(zip(BINDING_COLUMNS, wire_packet["bindings"], strict=True)),
        "compact_source": wire_packet["source"],
        "functions": [{
            "function_id": function_id, "name": name,
            "aliases": [{"original": original, "alias": alias, "kind": ALIAS_KINDS[kind]} for original, alias, kind in aliases],
        } for function_id, name, aliases in wire_packet["functions"]],
        "slots": [{
            "slot_id": slot_id, "function": wire_packet["functions"][function][1],
            "kind": SLOT_KINDS[kind], "before_sha256": before_hash,
            "editable": editable, "refusal_reason": reason,
        } for slot_id, function, kind, before_hash, editable, reason in wire_packet["slots"]],
        "authority": dict.fromkeys(AUTHORITY_COLUMNS, False),
    }
    _checked_model(restored)
    if _canonical(_pack_checked(restored)) != _canonical(wire_packet):
        _fail("wire packet is not the deterministic encoding of its expansion")
    return restored


def require_bound_python_code_view_packet(packet: dict, *, view: engine.PythonCodeView) -> dict:
    """Require exact public bytes of a real captured view, without granting authority."""
    if type(view) is not engine.PythonCodeView:
        _fail("an exact captured PythonCodeView is required")
    _profile(packet)
    if type(packet) is not dict:
        _fail("an exact packet dictionary is required")
    if packet.get("schema") == WIRE_SCHEMA:
        restored = unpack_python_code_view_packet(packet)
    else:
        _checked_model(packet)
        # Packing/expansion makes the returned value independent of caller mutation.
        restored = unpack_python_code_view_packet(_pack_checked(packet))
    if _canonical(restored) != _canonical(view.model_packet()):
        _fail("packet differs from the captured view's exact public packet")
    return restored
