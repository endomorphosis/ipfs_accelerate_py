"""Explicit task-owned MCP discovery/dispatch for bounded arithmetic views.

Use an otherwise empty canonical HierarchicalToolManager directly. This module
does not bootstrap a server, register provider tools, alter a default registry,
or opt into server result caching. Every response must come from a live handler.
The existing manager registration API publishes static schema/name metadata to
its process-global metadata catalog; no bound function, session, worktree,
visibility callback, candidate, source, or authority is put in that catalog.

Model tools can prepare bounded private candidates, never apply them. A trusted
controller supplies real transport visibility, consumes a freshly rechecked
candidate, and retains its existing single native apply/qualification stage.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import re
import threading
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import anyio
from ipfs_accelerate_py.mcp_server import hierarchical_tool_manager as manager_module
from ipfs_accelerate_py.mcp_server import tool_registry as registry_module
from ipfs_accelerate_py.utils.cid_utils import cid_for_dag_json, validate_cid

from ..semantic_refactoring.python_code_view import _canonical, _raw, _read_json
from .python_code_view_session_broker import PythonCodeViewSessionBroker
from .python_code_view_wire_worktree import (
    EDIT_WIRE_SCHEMA,
    MAX_EDIT_WIRE_BYTES,
    expand_python_code_view_wire_edit,
    parse_python_code_view_wire_edit,
)
from .python_code_view_worktree import NativeViewPatch

CATEGORY = "python_code_view"
MAX_ARGUMENT_BYTES = 64 * 1024
MAX_CANDIDATES = 4
MAX_CANDIDATE_ENCODED_BYTES = 8 * 1024 * 1024
MAX_CANDIDATE_ALLOCATIONS = 1_000_000
ERROR_SCHEMA = "supervisor-python-code-view-mcp-error@1"
CANDIDATE_SCHEMA = "supervisor-python-code-view-mcp-candidate@1"
_AUTHORITY = {
    "source_write_authority": False, "execution_authority": False,
    "proof_authority": False, "qualification_authority": False,
    "completion_authority": False,
}
_SID_SCHEMA = {"type": "string", "minLength": 1, "maxLength": 128}
_CID_SCHEMA = {"type": "string", "minLength": 61, "maxLength": 61, "pattern": "^b[a-z2-7]{60}$"}
_PATH_SCHEMA = {"type": "string", "minLength": 1, "maxLength": 1024}
_WIRE_EDIT_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "schema": {"const": EDIT_WIRE_SCHEMA},
        "edit": {
            "type": "array", "minItems": 4, "maxItems": 4,
            "prefixItems": [
                _CID_SCHEMA,
                {"type": "string", "minLength": 2, "maxLength": 1024, "pattern": "^s(?:0|[1-9][0-9]*)$"},
                {"type": "string", "minLength": 64, "maxLength": 64, "pattern": "^[0-9a-f]{64}$"},
                {"type": "string", "minLength": 1, "maxLength": 8192},
            ],
            "items": False,
        },
    },
    "required": ["schema", "edit"],
}
_WIRE_REQUEST_SCHEMA = {
    "oneOf": [_WIRE_EDIT_SCHEMA, {"type": "string", "minLength": 1, "maxLength": MAX_EDIT_WIRE_BYTES}],
}
_OPERATIONS = {
    "read_view": ({"path"}, {"known_read_id"}),
    "acknowledge_legend": ({"legend_id", "read_id"}, set()),
    "acknowledge_read": ({"read_id", "view_id"}, set()),
    "deliver_native_source": ({"view_id"}, {"known_delivery_id"}),
    "acknowledge_native_source": ({"delivery_id", "view_id"}, set()),
    "propose_edit": ({"wire_request"}, set()),
    "close_session": (set(), set()),
}
_DESCRIPTIONS = {
    "read_view": "Read a live bounded arithmetic Python view; references require explicit legend/read acknowledgements.",
    "acknowledge_legend": "Explicitly acknowledge a delivered semantic legend for this session/read; this describes transport only.",
    "acknowledge_read": "Explicitly acknowledge a delivered current view/read after its legend acknowledgement; this describes transport only.",
    "deliver_native_source": "Deliver the exact current native source; acknowledged exact delivery is required before proposing native edits.",
    "acknowledge_native_source": "Explicitly acknowledge an actual current native source delivery; a private retained original is never visibility.",
    "propose_edit": "Prepare a private hash-bound arithmetic expression candidate only; the trusted controller supplies visibility and consumes/applies it once.",
    "close_session": "Release session and candidate custody; old handlers refuse further calls.",
}
_LIVE_DESCRIPTION = (
    " Always live and not cacheable: use only the dedicated manager's direct dispatch, never a server result cache."
    " No response grants source writes, execution, proof, qualification, task completion or Lean admission."
)


class PythonCodeViewMCPError(ValueError):
    """A controller binding, private registration or candidate contract failed."""


class _InvalidRequest(ValueError):
    pass


def _failure(code: str) -> dict:
    return {"schema": ERROR_SCHEMA, "is_error": True, "error_code": code,
            "candidate_only": True, **_AUTHORITY, "lean_admits": 0}


def _request_cid(value: Any) -> None:
    _raw(value, 128, "identity")
    validate_cid(value, codecs=("dag-json",))


def _properties(fields: set[str]) -> dict:
    return {
        field: copy.deepcopy(
            _SID_SCHEMA if field == "session_id" else
            _PATH_SCHEMA if field == "path" else
            _WIRE_REQUEST_SCHEMA if field == "wire_request" else _CID_SCHEMA
        ) for field in sorted(fields)
    }


def _descriptor(name: str) -> dict:
    required, optional = _OPERATIONS[name]
    required = required | {"session_id"}
    return {
        "category": CATEGORY, "name": name,
        "description": _DESCRIPTIONS[name] + _LIVE_DESCRIPTION,
        "input_schema": {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object", "additionalProperties": False,
            "properties": _properties(required | optional), "required": sorted(required),
        },
    }


class PythonCodeViewMCPTools:
    """One explicit task/session adapter; handlers hold no independent source cache."""

    def __init__(self, broker: PythonCodeViewSessionBroker, session_id: str,
                 visibility_resolver: Callable[[dict], Mapping] | None = None):
        if type(broker) is not PythonCodeViewSessionBroker:
            raise PythonCodeViewMCPError("an exact task-owned session broker is required")
        try:
            _raw(session_id, 128, "session identity")
            if not session_id or any(char.isspace() for char in session_id):
                raise ValueError
        except Exception:
            raise PythonCodeViewMCPError("a bounded explicit session identity is required") from None
        if session_id != broker._session_id or broker._closed:
            raise PythonCodeViewMCPError("adapter and live broker session must match")
        if visibility_resolver is not None and (
            not callable(visibility_resolver) or inspect.iscoroutinefunction(visibility_resolver)
            or inspect.iscoroutinefunction(visibility_resolver.__call__)
        ):
            raise PythonCodeViewMCPError("visibility resolver must be an explicit synchronous controller callback")
        self._broker = broker
        self._session_id = session_id
        self._resolver = visibility_resolver
        self._proposals_enabled = visibility_resolver is not None
        self._adapter_id = "code-view-mcp:" + uuid.uuid4().hex
        self._allocation = 0
        self._candidates = {}
        self._lock = threading.RLock()
        self._closed = False
        self._registered = False

    def descriptors(self) -> list[dict]:
        """Return independent static schemas without bound objects or session data."""
        return [_descriptor(name) for name in _OPERATIONS
                if name != "propose_edit" or self._proposals_enabled]

    def _parameters(self, name: str, parameters: dict) -> dict:
        try:
            required, optional = _OPERATIONS[name]
            required = required | {"session_id"}
            if type(parameters) is not dict or not required <= set(parameters) <= required | optional:
                raise ValueError
            for field, value in parameters.items():
                if field == "wire_request":
                    if type(value) is str:
                        _raw(value, MAX_EDIT_WIRE_BYTES, "wire request")
                        parse_python_code_view_wire_edit(value)
                    elif type(value) is dict:
                        expand_python_code_view_wire_edit(value)
                    else:
                        raise ValueError
                elif field == "session_id":
                    _raw(value, 128, "session identity")
                    if value != self._session_id:
                        raise ValueError
                elif field == "path":
                    _raw(value, 1024, "path")
                    if (re.fullmatch(r"[A-Za-z0-9_./-]+\.py", value) is None
                            or value.startswith("/") or any(part in {"", ".", ".."} for part in value.split("/"))):
                        raise ValueError
                else:
                    _request_cid(value)
            # Field/row/type bounds above run before serialization or copying.
            raw = _canonical(parameters)
            if len(raw) > MAX_ARGUMENT_BYTES:
                raise ValueError
            result = _read_json(raw.decode("utf-8"))
            if "wire_request" in result:
                request = result["wire_request"]
                if type(request) is str:
                    request = parse_python_code_view_wire_edit(request)
                expanded = expand_python_code_view_wire_edit(request)
                _request_cid(expanded["view_id"])
                if re.fullmatch(r"s(?:0|[1-9][0-9]*)", expanded["slot_id"]) is None:
                    raise ValueError
                result["wire_request"] = request
            return result
        except Exception:
            raise _InvalidRequest from None

    @staticmethod
    def _visibility(value: Any) -> None:
        # This is a trusted controller port, never model-provided arguments.
        if not isinstance(value, Mapping) or len(value) > 4:
            raise PythonCodeViewMCPError("bounded actual transport visibility is required")
        for path, source in value.items():
            _raw(path, 1024, "visible path")
            if type(source) is str:
                _raw(source, 64 * 1024, "visible native source")
            elif type(source) is not bytes or len(source) > 64 * 1024:
                raise PythonCodeViewMCPError("exact actual transport source bytes are required")

    def _release(self) -> None:
        self._candidates.clear()
        self._broker = None
        self._resolver = None
        self._closed = True

    def _candidate_public(self, candidate_id: str, request: dict) -> dict:
        edit = expand_python_code_view_wire_edit(request)
        return {"schema": CANDIDATE_SCHEMA, "candidate_id": candidate_id,
                "view_id": edit["view_id"], "slot_id": edit["slot_id"],
                "candidate_only": True, **_AUTHORITY, "lean_admits": 0}

    def _propose(self, request: dict) -> dict:
        broker = self._broker
        edit = expand_python_code_view_wire_edit(request)
        # Validate the current owner/view before invoking the controller port.
        broker._check(self._session_id)
        broker._view(edit["view_id"])
        if (self._resolver is None or len(self._candidates) >= MAX_CANDIDATES
                or self._allocation >= MAX_CANDIDATE_ALLOCATIONS):
            raise PythonCodeViewMCPError("candidate preparation is unavailable or bounded")
        visible = self._resolver(_read_json(_canonical(request).decode("utf-8")))
        self._visibility(visible)
        prepared = broker.prepare_edit(wire_request=request, visible_sources=visible, session_id=self._session_id)
        size = (len(prepared.view.artifact_json.encode("utf-8"))
                + len(prepared.native_edit.artifact_json.encode("utf-8"))
                + len(prepared.patch_text.encode("utf-8")) + len(_canonical(request))
                + len(_canonical(prepared.receipt())))
        allocation = self._allocation + 1
        candidate_id = cid_for_dag_json({
            "schema": "supervisor-python-code-view-mcp-candidate-binding@1",
            "adapter_id": self._adapter_id, "allocation": allocation,
            "broker_id": broker._broker_id, "session_id": self._session_id,
            "wire_request": request, "native_edit_id": prepared.native_edit.edit_id,
            "native_patch_sha256": hashlib.sha256(prepared.patch_text.encode("utf-8")).hexdigest(),
        })
        size += len(candidate_id.encode("utf-8"))
        if sum(candidate[2] for candidate in self._candidates.values()) + size > MAX_CANDIDATE_ENCODED_BYTES:
            raise PythonCodeViewMCPError("encoded candidate retention exceeds bounds")
        self._candidates[candidate_id] = (request, prepared, size)
        self._allocation = allocation
        return self._candidate_public(candidate_id, request)

    def _execute(self, name: str, parameters: dict) -> dict:
        with self._lock:
            if self._closed or self._broker is None:
                return _failure("session_closed")
            try:
                arguments = self._parameters(name, parameters)
                if name == "propose_edit":
                    return self._propose(arguments["wire_request"])
                if name == "close_session":
                    try:
                        return self._broker.close_session(session_id=self._session_id)
                    finally:
                        self._release()
                return getattr(self._broker, name)(**arguments)
            except _InvalidRequest:
                return _failure("invalid_request")
            except Exception:
                if self._broker is not None and self._broker._closed:
                    self._release()
                    return _failure("session_closed")
                return _failure("operation_refused")

    def _handler(self, name: str) -> Callable:
        async def handler(**parameters):
            # No blocking broker/resolver/Git work runs on the event-loop thread.
            return await anyio.to_thread.run_sync(self._execute, name, parameters, abandon_on_cancel=False)
        handler.__name__ = name
        handler.__doc__ = _DESCRIPTIONS[name] + _LIVE_DESCRIPTION
        return handler

    def consume_candidate(self, candidate_id: str, *, visible_sources: Mapping) -> NativeViewPatch:
        """Controller only: freshly prepare, compare and consume once before native apply."""
        with self._lock:
            try:
                _request_cid(candidate_id)
                if self._closed or self._broker is None or candidate_id not in self._candidates:
                    raise ValueError
                self._visibility(visible_sources)
                request, retained, _ = self._candidates[candidate_id]
                fresh = self._broker.prepare_edit(wire_request=request, visible_sources=visible_sources,
                                                 session_id=self._session_id)
                if fresh != retained:
                    raise ValueError
                del self._candidates[candidate_id]
                return fresh
            except Exception:
                if self._broker is not None and self._broker._closed:
                    self._release()
                raise PythonCodeViewMCPError("candidate consumption requires current owned source and explicit native visibility") from None

    def close(self) -> dict:
        """Controller cleanup releases all references even after a failed/expired session."""
        with self._lock:
            try:
                if self._broker is not None:
                    self._broker.close_session(session_id=self._session_id)
            except Exception:
                # Cleanup owns no model-facing exception/log text.
                pass
            finally:
                self._release()
            return {"schema": "supervisor-python-code-view-mcp-tools-closed@1", "closed": True,
                    "retained_candidates": 0, **_AUTHORITY, "lean_admits": 0}


def register_python_code_view_tools(manager: manager_module.HierarchicalToolManager,
                                   adapter: PythonCodeViewMCPTools) -> tuple[str, ...]:
    """Register once through the real API on an explicit empty private manager.

    Static public metadata is also published by that API. The controller must
    retain exclusive manager/registry ownership and use direct live dispatch;
    this function never attaches a default/global server or enables caching.
    """
    if (type(manager) is not manager_module.HierarchicalToolManager
            or type(adapter) is not PythonCodeViewMCPTools):
        raise PythonCodeViewMCPError("exact canonical manager and adapter are required")
    package = Path(__file__).resolve().parents[2]
    if (Path(manager_module.__file__).resolve() != package / "mcp_server/hierarchical_tool_manager.py"
            or Path(registry_module.__file__).resolve() != package / "mcp_server/tool_registry.py"):
        raise PythonCodeViewMCPError("canonical manager and registry module origins are required")
    with adapter._lock:
        if adapter._closed or adapter._registered or manager.runtime_router is not None:
            raise PythonCodeViewMCPError("an unregistered live adapter and private unrouted manager are required")
        registry = manager.registry
        if (type(registry) is not registry_module.ToolRegistry
                or registry is registry_module._global_registry
                or getattr(manager.register_tool, "__func__", None) is not manager_module.HierarchicalToolManager.register_tool
                or getattr(registry.register_function, "__func__", None) is not registry_module.ToolRegistry.register_function
                or type(registry.total_executions) is not int or registry.total_executions != 0
                or any(type(value) is not dict or value for value in (
                    manager._categories, manager._category_loaders, manager._category_loaded, manager._breakers,
                    registry._tools, registry._categories, registry._tags,
                ))):
            raise PythonCodeViewMCPError("empty dedicated manager/registry state is required before registration")
        descriptors = adapter.descriptors()
        handlers = {item["name"]: adapter._handler(item["name"]) for item in descriptors}
        # All caller bindings, collisions, descriptors and handlers are checked
        # before entering the canonical registration API's mutations.
        adapter._registered = True
        for descriptor in descriptors:
            manager.register_tool(category=CATEGORY, name=descriptor["name"],
                func=handlers[descriptor["name"]], description=descriptor["description"],
                input_schema=copy.deepcopy(descriptor["input_schema"]), runtime=None, tags=[])
            # The manager API gives its wrapper an inferred **kwargs schema.
            # Override only this newly registered public wrapper schema so its
            # registry discovery agrees with the manager's explicit schema.
            registry.get_tool(f"{CATEGORY}.{descriptor['name']}").input_schema = copy.deepcopy(descriptor["input_schema"])
        return tuple(f"{CATEGORY}.{item['name']}" for item in descriptors)
