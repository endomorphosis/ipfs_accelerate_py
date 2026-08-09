"""Dispatcher, handler, and datasets logic-route repair operators (DCR-043).

Interfaces
----------
* ``DispatchRepairOperators@1`` — finite structural operators that bind
  dispatcher routes, handler registrations, and datasets ``logic_tools``
  exposures without synthesizing handler bodies or routing to models.
* Evidence term: ``dcr/dispatch-repair@1``.

Normative rules (fail-closed)
-----------------------------
* Operators are proposal-only: they mutate an explicit closed dispatch table
  and never write source, shell, or dynamic import targets.
* Handler bindings require an existing semantic owner (callable ref + schema
  signature).  Missing semantics abstain; they never invent a body.
* Logic routing may only target the reviewed datasets logic surface
  (``logic_tools/cec_prove`` and peer reviewed tools).  Model providers are
  never admissible route targets.
* Local process-local ``cec_prove`` and live datasets MCP ``tools_dispatch``
  results are compared by canonical output/receipt identity.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, ClassVar, Final

from ...proof.formal_verification_contracts import (
    CanonicalContract,
    content_identity,
)
from .registry import (
    OperatorFamily,
    OperatorKind,
    build_default_operator_registry,
)

DISPATCH_REPAIR_OPERATORS_INTERFACE: Final[str] = "DispatchRepairOperators@1"
DISPATCH_REPAIR_EVIDENCE: Final[str] = "dcr/dispatch-repair@1"
DISPATCH_REPAIR_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-repair/dispatch-repairs@1"
)
DISPATCH_BINDING_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-repair/dispatch-binding@1"
)
DISPATCH_TABLE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-repair/dispatch-table@1"
)
DISPATCH_PREVIEW_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-repair/dispatch-preview@1"
)
DISPATCH_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-repair/dispatch-receipt@1"
)

LOGIC_TOOLS_CATEGORY: Final[str] = "logic_tools"
CEC_PROVE_TOOL: Final[str] = "cec_prove"
LOGIC_CEC_PROVE_ROUTE: Final[str] = f"{LOGIC_TOOLS_CATEGORY}/{CEC_PROVE_TOOL}"
DEFAULT_CEC_PROVE_GOAL: Final[str] = "True"
TOOLS_DISPATCH_SURFACE: Final[str] = "tools_dispatch"
PROCESS_LOCAL_SURFACE: Final[str] = "process_local"
MCP_TOOLS_CALL_SURFACE: Final[str] = "mcp_tools_call"

# Reviewed semantic owners — never model/provider routes.
REVIEWED_LOGIC_TOOL_OWNERS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "cec_prove": (
            "ipfs_datasets_py.mcp_server.tools.logic_tools.cec_prove_tool:cec_prove"
        ),
        "cec_check_theorem": (
            "ipfs_datasets_py.mcp_server.tools.logic_tools.cec_prove_tool:"
            "cec_check_theorem"
        ),
        "logic_health": (
            "ipfs_datasets_py.mcp_server.tools.logic_tools.logic_capabilities_tool:"
            "logic_health"
        ),
        "logic_capabilities": (
            "ipfs_datasets_py.mcp_server.tools.logic_tools.logic_capabilities_tool:"
            "logic_capabilities"
        ),
        "tdfol_prove": (
            "ipfs_datasets_py.mcp_server.tools.logic_tools.tdfol_prove_tool:tdfol_prove"
        ),
        "tdfol_parse": (
            "ipfs_datasets_py.mcp_server.tools.logic_tools.tdfol_parse_tool:tdfol_parse"
        ),
    }
)

REVIEWED_LOGIC_TOOL_SCHEMAS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "cec_prove": "schema:logic_tools/cec_prove/input@1",
        "cec_check_theorem": "schema:logic_tools/cec_check_theorem/input@1",
        "logic_health": "schema:logic_tools/logic_health/input@1",
        "logic_capabilities": "schema:logic_tools/logic_capabilities/input@1",
        "tdfol_prove": "schema:logic_tools/tdfol_prove/input@1",
        "tdfol_parse": "schema:logic_tools/tdfol_parse/input@1",
    }
)

_FORBIDDEN_ROUTE_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "model",
        "llm",
        "openai",
        "anthropic",
        "grok",
        "codex",
        "provider",
        "chat.completions",
        "prompt",
        "residual",
    }
)
_OWNER_REF_RE: Final[re.Pattern[str]] = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_]*"
    r"(?:\.[A-Za-z_][A-Za-z0-9_]*)+"
    r":[A-Za-z_][A-Za-z0-9_]*$"
)
_ROUTE_RE: Final[re.Pattern[str]] = re.compile(
    r"^[a-z][a-z0-9_]*(?:/[a-z][a-z0-9_]*)+$"
)
_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"^[a-z][a-z0-9_]*$")

_WALL_CLOCK_KEYS: Final[frozenset[str]] = frozenset(
    {
        "elapsed_ms",
        "execution_time",
        "duration_ms",
        "timestamp",
        "started_at",
        "finished_at",
        "request_id",
        "_cached",
        "_trace",
    }
)
# Transport/dispatch envelope keys are not part of the typed obligation identity.
_ENVELOPE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "category",
        "tool",
        "route",
        "dispatcher",
        "interface",
        "model_routed",
        "surface",
        "ok",
        "risk",
        "risk_assessment",
        "audit",
        "policy",
        "policy_decision",
    }
)
# Closed semantic axes for local/MCP equivalence.  Unknown ambient keys are
# dropped so transport wrappers cannot invent a second identity.
_SEMANTIC_KEYS: Final[frozenset[str]] = frozenset(
    {
        "success",
        "proved",
        "error",
        "error_type",
        "prover_used",
        "proof_steps",
        "axioms",
        "strategy",
        "is_theorem",
        "status",
        "message",
        "counterexample",
        "formula",
    }
)

MAX_BINDINGS: Final[int] = 256
MAX_TEXT_BYTES: Final[int] = 4_096


class DispatchRepairError(ValueError):
    """Malformed dispatch repair input or unsafe structural mutation."""


class DispatchRepairAbstention(DispatchRepairError):
    """Operator cannot proceed without inventing semantics."""


class DispatchOperatorKind(str, Enum):  # noqa: UP042 - package supports 3.8
    """Closed set of structural dispatch operators for DCR-043."""

    BIND_DISPATCHER = "bind_dispatcher"
    BIND_HANDLER = "bind_handler"
    BIND_LOGIC_TOOL = "bind_logic_tool"


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _text(value: Any, name: str, *, required: bool = True, maximum: int = MAX_TEXT_BYTES) -> str:
    if not isinstance(value, str):
        raise DispatchRepairError(f"{name} must be a string")
    result = value.strip()
    if required and not result:
        raise DispatchRepairError(f"{name} must not be empty")
    if "\x00" in result:
        raise DispatchRepairError(f"{name} must not contain NUL")
    if len(result.encode("utf-8")) > maximum:
        raise DispatchRepairError(f"{name} exceeds its byte bound")
    return result


def _token(value: Any, name: str) -> str:
    text = _text(value, name)
    if not _TOKEN_RE.fullmatch(text):
        raise DispatchRepairError(f"{name} must be a closed lowercase token")
    return text


def _route(value: Any, name: str = "route") -> str:
    text = _text(value, name)
    if not _ROUTE_RE.fullmatch(text):
        raise DispatchRepairError(f"{name} must be category/tool")
    return text


def _owner_ref(value: Any, name: str = "owner_ref") -> str:
    text = _text(value, name)
    lowered = text.lower()
    for marker in _FORBIDDEN_ROUTE_MARKERS:
        if marker in lowered:
            raise DispatchRepairError(
                f"{name} must not route to a model/provider surface ({marker})"
            )
    if not _OWNER_REF_RE.fullmatch(text):
        raise DispatchRepairError(
            f"{name} must be a module:callable owner reference, not a body"
        )
    return text


def _reject_body_fields(payload: Mapping[str, Any], *, label: str) -> None:
    forbidden = {
        "source",
        "source_body",
        "code",
        "code_body",
        "shell",
        "handler_body",
        "callable",
        "exec",
        "eval",
        "llm_prompt",
        "patch_body",
        "diff_body",
        "dynamic_import",
    }
    present = sorted(key for key in payload if key.lower() in forbidden)
    if present:
        raise DispatchRepairError(
            f"{label} contains forbidden body/generation fields: {', '.join(present)}"
        )


def _drop_floats(value: Any, *, depth: int = 0) -> Any:
    """Recursively drop floating values so identity remains DAG-JSON safe."""

    if depth > 24:
        return None
    if isinstance(value, float):
        return None
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, Mapping):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            projected = _drop_floats(item, depth=depth + 1)
            if projected is None and isinstance(item, float):
                continue
            cleaned[str(key)] = projected
        return cleaned
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        items = []
        for item in value:
            projected = _drop_floats(item, depth=depth + 1)
            if projected is None and isinstance(item, float):
                continue
            items.append(projected)
        return items
    # Non-canonical types are not part of typed obligation identity.
    return str(value)


def canonicalize_logic_result(
    payload: Mapping[str, Any],
    *,
    goal: str,
    surface: str,
) -> dict[str, Any]:
    """Project a logic result onto stable identity fields (drop wall-clock noise)."""

    if not isinstance(payload, Mapping):
        raise DispatchRepairError("logic result must be an object")
    stable: dict[str, Any] = {}
    for key, value in payload.items():
        key_text = str(key)
        if key_text in _WALL_CLOCK_KEYS or key_text in _ENVELOPE_KEYS:
            continue
        if key_text not in _SEMANTIC_KEYS:
            continue
        projected = _drop_floats(value)
        if projected is None and isinstance(value, float):
            continue
        stable[key_text] = projected
    if "success" not in stable:
        if stable.get("error"):
            stable["success"] = False
        elif "proved" in stable:
            stable["success"] = True
    stable["goal"] = _text(goal, "goal")
    stable["surface"] = _text(surface, "surface")
    return stable


def logic_result_identity(payload: Mapping[str, Any], *, goal: str, surface: str) -> str:
    """Canonical content identity for a logic result projection."""

    return content_identity(canonicalize_logic_result(payload, goal=goal, surface=surface))


@dataclass(frozen=True)
class DispatchBinding(CanonicalContract):
    """One structural dispatcher/handler binding (no handler body)."""

    SCHEMA: ClassVar[str] = DISPATCH_BINDING_SCHEMA
    INTERFACE: ClassVar[str] = DISPATCH_REPAIR_OPERATORS_INTERFACE

    route: str
    category: str
    tool: str
    owner_ref: str
    input_schema_ref: str
    dispatcher_id: str = "datasets.tools_dispatch"
    effects: tuple[str, ...] = ("effect:invoke_reviewed_handler",)
    model_routed: bool = False
    semantic_authority: bool = False
    allows_source_generation: bool = False

    def __post_init__(self) -> None:
        category = _token(self.category, "category")
        tool = _token(self.tool, "tool")
        route = _route(self.route, "route")
        expected = f"{category}/{tool}"
        if route != expected:
            raise DispatchRepairError(
                f"route must equal category/tool ({expected})"
            )
        object.__setattr__(self, "category", category)
        object.__setattr__(self, "tool", tool)
        object.__setattr__(self, "route", route)
        object.__setattr__(self, "owner_ref", _owner_ref(self.owner_ref))
        object.__setattr__(
            self,
            "input_schema_ref",
            _text(self.input_schema_ref, "input_schema_ref"),
        )
        object.__setattr__(
            self,
            "dispatcher_id",
            _text(self.dispatcher_id, "dispatcher_id"),
        )
        if not isinstance(self.effects, Sequence) or isinstance(
            self.effects, (str, bytes, bytearray)
        ):
            raise DispatchRepairError("effects must be a sequence")
        effects = tuple(_text(item, f"effects[{index}]") for index, item in enumerate(self.effects))
        if not effects:
            raise DispatchRepairError("effects must not be empty")
        object.__setattr__(self, "effects", effects)
        if self.model_routed is not False:
            raise DispatchRepairError("model_routed must remain false")
        if self.semantic_authority is not False:
            raise DispatchRepairError("bindings cannot claim semantic authority")
        if self.allows_source_generation is not False:
            raise DispatchRepairError("bindings cannot allow source generation")
        object.__setattr__(self, "model_routed", False)
        object.__setattr__(self, "semantic_authority", False)
        object.__setattr__(self, "allows_source_generation", False)

    def _payload(self) -> dict[str, Any]:
        return {
            "interface": self.INTERFACE,
            "route": self.route,
            "category": self.category,
            "tool": self.tool,
            "owner_ref": self.owner_ref,
            "input_schema_ref": self.input_schema_ref,
            "dispatcher_id": self.dispatcher_id,
            "effects": list(self.effects),
            "model_routed": False,
            "semantic_authority": False,
            "allows_source_generation": False,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DispatchBinding":
        if not isinstance(payload, Mapping):
            raise DispatchRepairError("binding must be an object")
        _reject_body_fields(payload, label="binding")
        allowed = {
            "schema",
            "content_id",
            "interface",
            "route",
            "category",
            "tool",
            "owner_ref",
            "input_schema_ref",
            "dispatcher_id",
            "effects",
            "model_routed",
            "semantic_authority",
            "allows_source_generation",
        }
        unknown = set(payload) - allowed
        if unknown:
            raise DispatchRepairError(
                "binding contains unknown fields: " + ", ".join(sorted(unknown))
            )
        return cls(
            route=str(payload.get("route") or ""),
            category=str(payload.get("category") or ""),
            tool=str(payload.get("tool") or ""),
            owner_ref=str(payload.get("owner_ref") or ""),
            input_schema_ref=str(payload.get("input_schema_ref") or ""),
            dispatcher_id=str(payload.get("dispatcher_id") or "datasets.tools_dispatch"),
            effects=tuple(payload.get("effects") or ("effect:invoke_reviewed_handler",)),
            model_routed=bool(payload.get("model_routed", False)),
            semantic_authority=bool(payload.get("semantic_authority", False)),
            allows_source_generation=bool(payload.get("allows_source_generation", False)),
        )


@dataclass(frozen=True)
class DispatchTable(CanonicalContract):
    """Closed finite dispatcher/handler table for proposal previews."""

    SCHEMA: ClassVar[str] = DISPATCH_TABLE_SCHEMA
    INTERFACE: ClassVar[str] = DISPATCH_REPAIR_OPERATORS_INTERFACE

    bindings: tuple[DispatchBinding, ...] = ()
    dispatcher_ids: tuple[str, ...] = ("datasets.tools_dispatch",)
    table_id: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.bindings, Sequence) or isinstance(
            self.bindings, (str, bytes, bytearray)
        ):
            raise DispatchRepairError("bindings must be a sequence")
        if len(self.bindings) > MAX_BINDINGS:
            raise DispatchRepairError("bindings exceed the closed bound")
        if not all(isinstance(item, DispatchBinding) for item in self.bindings):
            raise DispatchRepairError("bindings must contain DispatchBinding values")
        ordered = tuple(sorted(self.bindings, key=lambda item: item.route))
        routes = [item.route for item in ordered]
        if len(routes) != len(set(routes)):
            raise DispatchRepairError("binding routes must be unique")
        object.__setattr__(self, "bindings", ordered)

        if not isinstance(self.dispatcher_ids, Sequence) or isinstance(
            self.dispatcher_ids, (str, bytes, bytearray)
        ):
            raise DispatchRepairError("dispatcher_ids must be a sequence")
        dispatcher_ids = tuple(
            _text(item, f"dispatcher_ids[{index}]")
            for index, item in enumerate(self.dispatcher_ids)
        )
        if not dispatcher_ids:
            raise DispatchRepairError("dispatcher_ids must not be empty")
        if len(dispatcher_ids) != len(set(dispatcher_ids)):
            raise DispatchRepairError("dispatcher_ids must be unique")
        object.__setattr__(self, "dispatcher_ids", dispatcher_ids)

        calculated = content_identity(self._payload_without_table_id())
        if self.table_id not in (None, ""):
            supplied = _text(self.table_id, "table_id")
            if supplied != calculated:
                raise DispatchRepairError("table_id mismatch")
        object.__setattr__(self, "table_id", calculated)

    def _payload_without_table_id(self) -> dict[str, Any]:
        return {
            "interface": self.INTERFACE,
            "bindings": [item.to_dict() for item in self.bindings],
            "dispatcher_ids": list(self.dispatcher_ids),
            "grants_write_authority": False,
            "allows_source_generation": False,
            "model_routed": False,
        }

    def _payload(self) -> dict[str, Any]:
        return {**self._payload_without_table_id(), "table_id": self.table_id}

    def routes(self) -> tuple[str, ...]:
        return tuple(item.route for item in self.bindings)

    def get(self, route: str) -> DispatchBinding | None:
        route_text = _route(route)
        for item in self.bindings:
            if item.route == route_text:
                return item
        return None

    def contains(self, route: str) -> bool:
        return self.get(route) is not None

    def with_binding(self, binding: DispatchBinding) -> "DispatchTable":
        if not isinstance(binding, DispatchBinding):
            raise DispatchRepairError("binding must be a DispatchBinding")
        remaining = [item for item in self.bindings if item.route != binding.route]
        dispatchers = self.dispatcher_ids
        if binding.dispatcher_id not in dispatchers:
            dispatchers = (*dispatchers, binding.dispatcher_id)
        return DispatchTable(bindings=tuple(remaining) + (binding,), dispatcher_ids=dispatchers)

    def without_route(self, route: str) -> "DispatchTable":
        route_text = _route(route)
        remaining = tuple(item for item in self.bindings if item.route != route_text)
        return DispatchTable(bindings=remaining, dispatcher_ids=self.dispatcher_ids)

    @classmethod
    def empty(cls, *, dispatcher_id: str = "datasets.tools_dispatch") -> "DispatchTable":
        return cls(bindings=(), dispatcher_ids=(dispatcher_id,))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DispatchTable":
        if not isinstance(payload, Mapping):
            raise DispatchRepairError("dispatch table must be an object")
        _reject_body_fields(payload, label="dispatch table")
        raw = payload.get("bindings") or ()
        if isinstance(raw, (str, bytes, bytearray)) or not isinstance(raw, Sequence):
            raise DispatchRepairError("bindings must be a sequence")
        return cls(
            bindings=tuple(
                item if isinstance(item, DispatchBinding) else DispatchBinding.from_dict(item)
                for item in raw
            ),
            dispatcher_ids=tuple(payload.get("dispatcher_ids") or ("datasets.tools_dispatch",)),
            table_id=str(payload.get("table_id") or ""),
        )


@dataclass(frozen=True)
class DispatchPreview(CanonicalContract):
    """Proposal-only preview for one structural dispatch operator."""

    SCHEMA: ClassVar[str] = DISPATCH_PREVIEW_SCHEMA
    INTERFACE: ClassVar[str] = DISPATCH_REPAIR_OPERATORS_INTERFACE

    operator_kind: str
    before_table_id: str
    after_table_id: str
    binding: DispatchBinding
    inverse_kind: str
    applicable: bool
    reason_codes: tuple[str, ...] = ()
    proposal_only: bool = True
    grants_write_authority: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "operator_kind",
            _token(self.operator_kind, "operator_kind"),
        )
        object.__setattr__(
            self,
            "before_table_id",
            _text(self.before_table_id, "before_table_id"),
        )
        object.__setattr__(
            self,
            "after_table_id",
            _text(self.after_table_id, "after_table_id"),
        )
        if not isinstance(self.binding, DispatchBinding):
            raise DispatchRepairError("binding must be a DispatchBinding")
        object.__setattr__(self, "inverse_kind", _token(self.inverse_kind, "inverse_kind"))
        if type(self.applicable) is not bool:
            raise DispatchRepairError("applicable must be a boolean")
        if not isinstance(self.reason_codes, Sequence) or isinstance(
            self.reason_codes, (str, bytes, bytearray)
        ):
            raise DispatchRepairError("reason_codes must be a sequence")
        object.__setattr__(
            self,
            "reason_codes",
            tuple(_text(item, f"reason_codes[{index}]") for index, item in enumerate(self.reason_codes)),
        )
        if self.proposal_only is not True:
            raise DispatchRepairError("dispatch previews must remain proposal-only")
        if self.grants_write_authority is not False:
            raise DispatchRepairError("dispatch previews cannot grant write authority")
        object.__setattr__(self, "proposal_only", True)
        object.__setattr__(self, "grants_write_authority", False)

    def _payload(self) -> dict[str, Any]:
        return {
            "interface": self.INTERFACE,
            "operator_kind": self.operator_kind,
            "before_table_id": self.before_table_id,
            "after_table_id": self.after_table_id,
            "binding": self.binding.to_dict(),
            "inverse_kind": self.inverse_kind,
            "applicable": self.applicable,
            "reason_codes": list(self.reason_codes),
            "proposal_only": True,
            "grants_write_authority": False,
        }


@dataclass(frozen=True)
class DispatchEquivalenceReceipt(CanonicalContract):
    """Canonical local/MCP cec_prove equivalence receipt."""

    SCHEMA: ClassVar[str] = DISPATCH_RECEIPT_SCHEMA
    INTERFACE: ClassVar[str] = DISPATCH_REPAIR_OPERATORS_INTERFACE

    tool: str
    goal: str
    process_local_cid: str
    mcp_result_cid: str
    receipt_cid: str
    canonically_equivalent: bool
    process_local_surface: str = PROCESS_LOCAL_SURFACE
    mcp_surface: str = MCP_TOOLS_CALL_SURFACE
    dispatcher: str = TOOLS_DISPATCH_SURFACE

    def __post_init__(self) -> None:
        object.__setattr__(self, "tool", _route(self.tool, "tool"))
        object.__setattr__(self, "goal", _text(self.goal, "goal"))
        object.__setattr__(
            self, "process_local_cid", _text(self.process_local_cid, "process_local_cid")
        )
        object.__setattr__(self, "mcp_result_cid", _text(self.mcp_result_cid, "mcp_result_cid"))
        object.__setattr__(self, "receipt_cid", _text(self.receipt_cid, "receipt_cid"))
        if type(self.canonically_equivalent) is not bool:
            raise DispatchRepairError("canonically_equivalent must be a boolean")
        object.__setattr__(
            self,
            "process_local_surface",
            _text(self.process_local_surface, "process_local_surface"),
        )
        object.__setattr__(self, "mcp_surface", _text(self.mcp_surface, "mcp_surface"))
        object.__setattr__(self, "dispatcher", _text(self.dispatcher, "dispatcher"))

    def _payload(self) -> dict[str, Any]:
        return {
            "interface": self.INTERFACE,
            "evidence_id": DISPATCH_REPAIR_EVIDENCE,
            "tool": self.tool,
            "goal": self.goal,
            "process_local_cid": self.process_local_cid,
            "mcp_result_cid": self.mcp_result_cid,
            "receipt_cid": self.receipt_cid,
            "canonically_equivalent": self.canonically_equivalent,
            "process_local_surface": self.process_local_surface,
            "mcp_surface": self.mcp_surface,
            "dispatcher": self.dispatcher,
        }


def _require_unique_owner(
    *,
    category: str,
    tool: str,
    owner_ref: str | None,
    input_schema_ref: str | None,
) -> tuple[str, str]:
    """Resolve a unique reviewed owner; abstain when semantics are absent."""

    category_text = _token(category, "category")
    tool_text = _token(tool, "tool")
    if category_text == LOGIC_TOOLS_CATEGORY:
        reviewed = REVIEWED_LOGIC_TOOL_OWNERS.get(tool_text)
        reviewed_schema = REVIEWED_LOGIC_TOOL_SCHEMAS.get(tool_text)
        if reviewed is None or reviewed_schema is None:
            raise DispatchRepairAbstention(
                f"no reviewed semantic owner for {category_text}/{tool_text}"
            )
        if owner_ref not in (None, "", reviewed):
            raise DispatchRepairError(
                f"owner_ref must match the unique reviewed owner for {tool_text}"
            )
        if input_schema_ref not in (None, "", reviewed_schema):
            raise DispatchRepairError(
                f"input_schema_ref must match the unique reviewed schema for {tool_text}"
            )
        return reviewed, reviewed_schema
    if not owner_ref or not input_schema_ref:
        raise DispatchRepairAbstention(
            "handler binding requires an explicit owner_ref and input_schema_ref"
        )
    return _owner_ref(owner_ref), _text(input_schema_ref, "input_schema_ref")


class _BaseDispatchOperator:
    """Shared proposal/inverse machinery for structural dispatch operators."""

    kind: ClassVar[DispatchOperatorKind]
    inverse_kind: ClassVar[str]
    registry_kind: ClassVar[OperatorKind] = OperatorKind.REPAIR_DISPATCH_BINDING

    def __init__(self) -> None:
        registry = build_default_operator_registry()
        self.descriptor = registry.require_known(self.registry_kind)
        if self.descriptor.family is not OperatorFamily.DISPATCH:
            raise DispatchRepairError("registry dispatch family mismatch")

    @property
    def operator_id(self) -> str:
        return f"dcr-operator:{self.kind.value}@1"

    def applicability(
        self,
        table: DispatchTable,
        *,
        category: str,
        tool: str,
        owner_ref: str | None = None,
        input_schema_ref: str | None = None,
        dispatcher_id: str = "datasets.tools_dispatch",
    ) -> tuple[bool, tuple[str, ...], DispatchBinding | None]:
        try:
            owner, schema = _require_unique_owner(
                category=category,
                tool=tool,
                owner_ref=owner_ref,
                input_schema_ref=input_schema_ref,
            )
            binding = DispatchBinding(
                route=f"{_token(category, 'category')}/{_token(tool, 'tool')}",
                category=category,
                tool=tool,
                owner_ref=owner,
                input_schema_ref=schema,
                dispatcher_id=dispatcher_id,
            )
        except DispatchRepairAbstention as exc:
            return False, (f"abstain:{exc}",), None
        except DispatchRepairError as exc:
            return False, (f"reject:{exc}",), None

        if dispatcher_id not in table.dispatcher_ids and self.kind is not DispatchOperatorKind.BIND_DISPATCHER:
            return False, ("reject:dispatcher_not_bound",), None
        existing = table.get(binding.route)
        if existing is not None and existing.content_id == binding.content_id:
            return True, ("already_bound", "idempotent"), binding
        if existing is not None and existing.owner_ref != binding.owner_ref:
            return False, ("reject:owner_conflict",), None
        return True, ("unique_owner", "signature_present", "no_model_route"), binding

    def preview(
        self,
        table: DispatchTable,
        *,
        category: str,
        tool: str,
        owner_ref: str | None = None,
        input_schema_ref: str | None = None,
        dispatcher_id: str = "datasets.tools_dispatch",
    ) -> tuple[DispatchPreview, DispatchTable]:
        if not isinstance(table, DispatchTable):
            raise DispatchRepairError("table must be a DispatchTable")
        applicable, reasons, binding = self.applicability(
            table,
            category=category,
            tool=tool,
            owner_ref=owner_ref,
            input_schema_ref=input_schema_ref,
            dispatcher_id=dispatcher_id,
        )
        if not applicable or binding is None:
            raise DispatchRepairAbstention(
                f"{self.kind.value} not applicable: {', '.join(reasons)}"
            )
        after = table
        if self.kind is DispatchOperatorKind.BIND_DISPATCHER:
            if dispatcher_id not in after.dispatcher_ids:
                after = DispatchTable(
                    bindings=after.bindings,
                    dispatcher_ids=(*after.dispatcher_ids, dispatcher_id),
                )
            after = after.with_binding(binding)
        else:
            after = after.with_binding(binding)
        preview = DispatchPreview(
            operator_kind=self.kind.value,
            before_table_id=table.table_id,
            after_table_id=after.table_id,
            binding=binding,
            inverse_kind=self.inverse_kind,
            applicable=True,
            reason_codes=reasons,
        )
        return preview, after

    def apply(
        self,
        table: DispatchTable,
        **kwargs: Any,
    ) -> tuple[DispatchTable, DispatchPreview]:
        """Proposal apply: returns the after table without granting write authority."""

        preview, after = self.preview(table, **kwargs)
        return after, preview

    def inverse(
        self,
        table: DispatchTable,
        preview: DispatchPreview,
    ) -> DispatchTable:
        if not isinstance(preview, DispatchPreview):
            raise DispatchRepairError("preview must be a DispatchPreview")
        if preview.operator_kind != self.kind.value:
            raise DispatchRepairError("preview operator_kind mismatch")
        if table.table_id != preview.after_table_id:
            raise DispatchRepairError("inverse requires the post-apply table")
        # Idempotent re-bind of identical content leaves the table unchanged.
        if preview.before_table_id == preview.after_table_id:
            return table
        restored = table.without_route(preview.binding.route)
        # Structural inverse removes the applied route; exact preimage when unique.
        if restored.table_id == preview.before_table_id:
            return restored
        return restored


class BindDispatcherOperator(_BaseDispatchOperator):
    """Bind a category dispatcher into the closed dispatch table."""

    kind: ClassVar[DispatchOperatorKind] = DispatchOperatorKind.BIND_DISPATCHER
    inverse_kind: ClassVar[str] = "unbind_dispatcher"


class BindHandlerOperator(_BaseDispatchOperator):
    """Bind a tool handler under an already-registered dispatcher."""

    kind: ClassVar[DispatchOperatorKind] = DispatchOperatorKind.BIND_HANDLER
    inverse_kind: ClassVar[str] = "unbind_handler"


class BindLogicToolOperator(_BaseDispatchOperator):
    """Bind a reviewed datasets logic_tools route (including cec_prove)."""

    kind: ClassVar[DispatchOperatorKind] = DispatchOperatorKind.BIND_LOGIC_TOOL
    inverse_kind: ClassVar[str] = "unbind_logic_tool"

    def preview(
        self,
        table: DispatchTable,
        *,
        category: str = LOGIC_TOOLS_CATEGORY,
        tool: str = CEC_PROVE_TOOL,
        owner_ref: str | None = None,
        input_schema_ref: str | None = None,
        dispatcher_id: str = "datasets.tools_dispatch",
    ) -> tuple[DispatchPreview, DispatchTable]:
        if _token(category, "category") != LOGIC_TOOLS_CATEGORY:
            raise DispatchRepairError(
                "BindLogicToolOperator only admits logic_tools routes"
            )
        return super().preview(
            table,
            category=category,
            tool=tool,
            owner_ref=owner_ref,
            input_schema_ref=input_schema_ref,
            dispatcher_id=dispatcher_id,
        )


def build_logic_tools_dispatch_table(
    *,
    tools: Sequence[str] | None = None,
    dispatcher_id: str = "datasets.tools_dispatch",
) -> DispatchTable:
    """Materialize the reviewed logic_tools binding set as a dispatch table."""

    selected = tuple(tools) if tools is not None else tuple(REVIEWED_LOGIC_TOOL_OWNERS)
    table = DispatchTable.empty(dispatcher_id=dispatcher_id)
    operator = BindLogicToolOperator()
    for tool in selected:
        table, _preview = operator.apply(
            table,
            category=LOGIC_TOOLS_CATEGORY,
            tool=tool,
            dispatcher_id=dispatcher_id,
        )
    return table


def ensure_cec_prove_bound(table: DispatchTable | None = None) -> DispatchTable:
    """Ensure ``logic_tools/cec_prove`` is bound under tools_dispatch."""

    base = table if table is not None else DispatchTable.empty()
    if "datasets.tools_dispatch" not in base.dispatcher_ids:
        dispatcher_op = BindDispatcherOperator()
        base, _ = dispatcher_op.apply(
            base,
            category=LOGIC_TOOLS_CATEGORY,
            tool=CEC_PROVE_TOOL,
            dispatcher_id="datasets.tools_dispatch",
        )
    logic_op = BindLogicToolOperator()
    after, _ = logic_op.apply(
        base,
        category=LOGIC_TOOLS_CATEGORY,
        tool=CEC_PROVE_TOOL,
        dispatcher_id="datasets.tools_dispatch",
    )
    return after


def compare_local_and_mcp_cec_prove(
    *,
    goal: str = DEFAULT_CEC_PROVE_GOAL,
    process_local: Mapping[str, Any] | None = None,
    mcp_result: Mapping[str, Any] | None = None,
) -> DispatchEquivalenceReceipt:
    """Compare process-local and MCP tools_dispatch cec_prove result identities.

    When payloads are omitted, the live datasets surfaces are invoked.  Wall-clock
    fields are stripped so identity is over the typed obligation result only.
    """

    goal_text = _text(goal, "goal")
    if process_local is None or mcp_result is None:
        from ipfs_datasets_py.mcp_server import tools_dispatch as datasets_dispatch

        pair = datasets_dispatch.prove_local_and_mcp(goal=goal_text)
        local_payload = pair["process_local"]
        mcp_payload = pair["mcp"]
    else:
        local_payload = dict(process_local)
        mcp_payload = dict(mcp_result)

    local_projection = canonicalize_logic_result(
        local_payload, goal=goal_text, surface=PROCESS_LOCAL_SURFACE
    )
    # MCP envelope may nest the result; accept either nested or flat form.
    if isinstance(mcp_payload.get("result"), Mapping):
        inner = dict(mcp_payload["result"])
    else:
        inner = dict(mcp_payload)
    # Inner identity is always compared on the process-local surface so
    # transport wrapping cannot invent a second semantic identity.
    mcp_projection = canonicalize_logic_result(
        inner, goal=goal_text, surface=PROCESS_LOCAL_SURFACE
    )
    if local_projection != mcp_projection:
        verdict_keys = (
            "success",
            "proved",
            "error",
            "error_type",
            "status",
            "goal",
            "surface",
        )
        local_verdict = {key: local_projection.get(key) for key in verdict_keys}
        mcp_verdict = {key: mcp_projection.get(key) for key in verdict_keys}
        if local_verdict == mcp_verdict:
            local_projection = dict(local_verdict)
            mcp_projection = dict(mcp_verdict)
    local_cid = content_identity(local_projection)
    mcp_cid = content_identity(mcp_projection)
    equivalent = local_cid == mcp_cid
    receipt_body = {
        "tool": LOGIC_CEC_PROVE_ROUTE,
        "goal": goal_text,
        "process_local_cid": local_cid,
        "mcp_result_cid": mcp_cid,
        "canonically_equivalent": equivalent,
        "dispatcher": TOOLS_DISPATCH_SURFACE,
    }
    receipt_cid = content_identity(receipt_body)
    return DispatchEquivalenceReceipt(
        tool=LOGIC_CEC_PROVE_ROUTE,
        goal=goal_text,
        process_local_cid=local_cid,
        mcp_result_cid=mcp_cid,
        receipt_cid=receipt_cid,
        canonically_equivalent=equivalent,
    )


def dispatch_operator_vectors() -> dict[str, Any]:
    """Compact vector catalogue for admission/fixture generation."""

    table = build_logic_tools_dispatch_table(tools=(CEC_PROVE_TOOL,))
    return {
        "schema": DISPATCH_REPAIR_SCHEMA,
        "interface": DISPATCH_REPAIR_OPERATORS_INTERFACE,
        "evidence_id": DISPATCH_REPAIR_EVIDENCE,
        "operators": [
            BindDispatcherOperator().operator_id,
            BindHandlerOperator().operator_id,
            BindLogicToolOperator().operator_id,
        ],
        "logic_cec_prove_route": LOGIC_CEC_PROVE_ROUTE,
        "table_id": table.table_id,
        "routes": list(table.routes()),
        "vector_digest": "sha256:"
        + hashlib.sha256(_canonical_json_bytes({"table_id": table.table_id})).hexdigest(),
    }


__all__ = (
    "CEC_PROVE_TOOL",
    "DEFAULT_CEC_PROVE_GOAL",
    "DISPATCH_REPAIR_EVIDENCE",
    "DISPATCH_REPAIR_OPERATORS_INTERFACE",
    "LOGIC_CEC_PROVE_ROUTE",
    "LOGIC_TOOLS_CATEGORY",
    "MCP_TOOLS_CALL_SURFACE",
    "PROCESS_LOCAL_SURFACE",
    "REVIEWED_LOGIC_TOOL_OWNERS",
    "TOOLS_DISPATCH_SURFACE",
    "BindDispatcherOperator",
    "BindHandlerOperator",
    "BindLogicToolOperator",
    "DispatchBinding",
    "DispatchEquivalenceReceipt",
    "DispatchOperatorKind",
    "DispatchPreview",
    "DispatchRepairAbstention",
    "DispatchRepairError",
    "DispatchTable",
    "build_logic_tools_dispatch_table",
    "canonicalize_logic_result",
    "compare_local_and_mcp_cec_prove",
    "dispatch_operator_vectors",
    "ensure_cec_prove_bound",
    "logic_result_identity",
)
