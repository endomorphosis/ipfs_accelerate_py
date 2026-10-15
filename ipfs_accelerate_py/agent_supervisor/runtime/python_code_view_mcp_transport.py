"""Bounded task-private JSON-RPC transport for live arithmetic code-view tools.

This is a controller-owned byte-channel boundary, not a default server, a CLI
configuration or a provider loop. A trusted asynchronous sender must finish
writing the exact frame before native bytes enter transport visibility. Model
acknowledgement remains separate. Candidate handoff never applies source.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import threading
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

import anyio

from ...mcp_server.hierarchical_tool_manager import HierarchicalToolManager
from ...mcp_server.tool_registry import ToolRegistry
from ..semantic_refactoring.python_code_view import _canonical, _json_pairs, _profile
from .python_code_view_mcp_tools import (
    CATEGORY,
    PythonCodeViewMCPTools,
    register_python_code_view_tools,
)
from .python_code_view_session_broker import PythonCodeViewSessionBroker
from .python_code_view_worktree import NativeViewPatch

PROTOCOL_VERSION = "2024-11-05"
MAX_FRAME_BYTES = 96 * 1024
MAX_RESPONSE_BYTES = 512 * 1024
MAX_SESSION_BYTES = 8 * 1024 * 1024
MAX_MESSAGES = 256
MAX_JSON_DEPTH = 32
MAX_JSON_NODES = 16_384


class PythonCodeViewMCPTransportError(ValueError):
    """Fixed controller-side transport failure; contains no source or request."""


class _RPCError(Exception):
    def __init__(self, code: int):
        self.code = code


@dataclass(frozen=True)
class PythonCodeViewCandidateHandoff:
    """Private consumed candidate and actual native visibility for one writer.

    This object is never a protocol response. The existing controller must
    revalidate current owner, fence, source, scope and patch before its one
    native apply. It carries no qualification or completion verdict.
    """

    prepared: NativeViewPatch
    visible_sources: Mapping[str, str]


def _bounded_text(value, maximum=128):
    if type(value) is not str or not value or len(value) > maximum:
        raise _RPCError(-32602)
    try:
        if len(value.encode("utf-8")) > maximum:
            raise _RPCError(-32602)
    except UnicodeError:
        raise _RPCError(-32602) from None


def _error(code, identity=None):
    return {"jsonrpc": "2.0", "id": identity, "error": {
        "code": code,
        "message": {
            -32700: "Parse error", -32600: "Invalid request",
            -32601: "Method not found", -32602: "Invalid params",
            -32000: "Session unavailable",
        }[code],
    }}


def _parse(raw):
    if type(raw) is not bytes or not raw or len(raw) > MAX_FRAME_BYTES:
        raise _RPCError(-32700)
    # One stdio-style frame. JSON strings must escape their own line breaks.
    body = raw[:-1] if raw.endswith(b"\n") else raw
    if b"\n" in body or b"\r" in body:
        raise _RPCError(-32700)
    try:
        value = json.loads(body.decode("utf-8", errors="strict"), object_pairs_hook=_json_pairs)
        _profile(value)
        pending, count = [(value, 0)], 0
        while pending:
            item, depth = pending.pop()
            count += 1
            if count > MAX_JSON_NODES or depth > MAX_JSON_DEPTH:
                raise ValueError
            if type(item) is dict:
                pending.extend((child, depth + 1) for child in item.values())
            elif type(item) is list:
                pending.extend((child, depth + 1) for child in item)
        return value
    except Exception:
        raise _RPCError(-32700) from None


def _envelope(value):
    if type(value) is not dict or value.get("jsonrpc") != "2.0":
        raise _RPCError(-32600)
    if not {"jsonrpc", "method"} <= set(value) <= {"jsonrpc", "id", "method", "params"}:
        raise _RPCError(-32600)
    if type(value["method"]) is not str or len(value["method"]) > 128:
        raise _RPCError(-32600)
    if "id" in value:
        identity = value["id"]
        if type(identity) is int:
            if abs(identity) > 2**53 - 1:
                raise _RPCError(-32600)
        elif type(identity) is str:
            try:
                _bounded_text(identity)
            except _RPCError:
                raise _RPCError(-32600) from None
        else:
            raise _RPCError(-32600)
    if "params" in value and type(value["params"]) is not dict:
        raise _RPCError(-32602)
    return value


class PythonCodeViewMCPTransport:
    """One sequential, cacheless channel bound to one existing native broker.

    ``send`` is a trusted controller I/O callback. It must complete only after
    the entire byte frame is written/flushed to the intended task channel. Its
    successful return proves transport write, not model reading or consent.
    Reentrant/concurrent exchange, consumption and closure are refused.
    """

    def __init__(self, *, broker: PythonCodeViewSessionBroker, session_id: str,
                 editable: bool = False):
        if type(editable) is not bool:
            raise PythonCodeViewMCPTransportError("an explicit boolean edit policy is required")
        self._visible = {}
        self._adapter = PythonCodeViewMCPTools(
            broker=broker, session_id=session_id,
            visibility_resolver=self._visibility if editable else None,
        )
        self._manager = HierarchicalToolManager(registry=ToolRegistry())
        names = register_python_code_view_tools(self._manager, self._adapter)
        self._names = frozenset(names)
        self._session_id = session_id
        self._state = "new"
        self._busy = False
        self._closed = False
        self._messages = 0
        self._bytes = 0
        self._ids = set()
        self._thread = threading.get_ident()
        self._loop = None

    def _enter(self):
        if threading.get_ident() != self._thread:
            raise PythonCodeViewMCPTransportError("one controller thread and event loop are required")
        loop = asyncio.get_running_loop()
        if self._loop is not None and loop is not self._loop:
            raise PythonCodeViewMCPTransportError("one controller thread and event loop are required")
        if self._busy:
            raise PythonCodeViewMCPTransportError("a sequential controller-owned channel is required")
        self._loop = loop
        self._busy = True

    def _visibility(self, request):
        return dict(self._visible)

    async def _release(self):
        # Cancellation must not leave a partially delivered session reusable.
        self._closed = True
        self._state = "closed"
        self._visible.clear()
        self._ids.clear()
        adapter, self._adapter = self._adapter, None
        self._manager = None
        if adapter is not None:
            with anyio.CancelScope(shield=True):
                await anyio.to_thread.run_sync(adapter.close, abandon_on_cancel=False)

    async def _dispatch(self, request):
        method, params = request["method"], request.get("params", {})
        notification = "id" not in request
        if self._closed:
            raise _RPCError(-32000)
        if notification:
            # Unsupported notifications are silent, never dispatched as tools.
            if method != "notifications/initialized":
                return None
            if params:
                raise _RPCError(-32602)
            if self._state != "initializing":
                raise _RPCError(-32000)
            self._state = "ready"
            return None
        if method == "initialize":
            if self._state != "new":
                raise _RPCError(-32000)
            if set(params) != {"protocolVersion", "capabilities", "clientInfo"}:
                raise _RPCError(-32602)
            _bounded_text(params["protocolVersion"])
            client = params["clientInfo"]
            if (type(params["capabilities"]) is not dict or params["capabilities"]
                    or type(client) is not dict or set(client) != {"name", "version"}):
                raise _RPCError(-32602)
            _bounded_text(client["name"])
            _bounded_text(client["version"])
            return {"protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": "task-private-python-code-view", "version": "1"}}
        if self._state != "ready":
            raise _RPCError(-32000)
        if method == "tools/list":
            if params:
                raise _RPCError(-32602)
            return {"tools": [
                {"name": f"{CATEGORY}.{item['name']}", "description": item["description"],
                 "inputSchema": item["input_schema"]}
                for item in self._adapter.descriptors()
            ]}
        if method != "tools/call":
            raise _RPCError(-32601)
        if set(params) != {"name", "arguments"}:
            raise _RPCError(-32602)
        name, arguments = params["name"], params["arguments"]
        if type(name) is not str or name not in self._names or type(arguments) is not dict:
            raise _RPCError(-32602)
        # The only execution route is the real private manager's live handler.
        result = await self._manager.dispatch(CATEGORY, name.split(".", 1)[1], arguments)
        return {"content": [{"type": "text", "text": _canonical(result).decode("utf-8")}],
                "isError": result.get("is_error", False) is True}

    def _native_delivery(self, result):
        # Read only the actual successful tool reply, never a private original.
        if not result or "content" not in result or result.get("isError"):
            return None
        body = json.loads(result["content"][0]["text"])
        if body.get("schema") != "supervisor-python-code-view-native-delivery@1":
            return None
        if body["session_id"] != self._session_id:
            raise ValueError
        self._adapter._visibility({body["path"]: body["source"]})
        if body["path"] not in self._visible and len(self._visible) >= 4:
            raise ValueError
        return body["path"], body["source"]

    async def exchange(self, raw: bytes, send: Callable[[bytes], Awaitable[None]]) -> None:
        """Dispatch one complete frame and await its exact outbound delivery.

        No response bytes or private custody objects are returned to the model
        caller. Invalid notifications are silent. Requests receive only fixed
        protocol errors or actual live tool responses.
        """
        if not callable(send) or not (
            inspect.iscoroutinefunction(send) or inspect.iscoroutinefunction(send.__call__)
        ):
            raise PythonCodeViewMCPTransportError("an asynchronous controller sender is required")
        self._enter()
        identity, notification, result, success = None, False, None, False
        try:
            if self._messages >= MAX_MESSAGES or (
                type(raw) is bytes and self._bytes + len(raw) > MAX_SESSION_BYTES
            ):
                await self._release()
                raise PythonCodeViewMCPTransportError("task channel budget exhausted")
            self._messages += 1
            self._bytes += len(raw) if type(raw) is bytes else 0
            try:
                parsed = _parse(raw)
                notification = (type(parsed) is dict and "id" not in parsed and parsed.get("jsonrpc") == "2.0"
                                and type(parsed.get("method")) is str)
                request = _envelope(parsed)
                identity = request.get("id")
                notification = "id" not in request
                if not notification:
                    key = (type(identity), identity)
                    if key in self._ids:
                        raise _RPCError(-32600)
                    self._ids.add(key)
                result = await self._dispatch(request)
                success = True
                response = {"jsonrpc": "2.0", "id": identity, "result": result}
            except _RPCError as exc:
                response = _error(exc.code, identity)
            except Exception:
                # Internal failures cannot expose manager/SDK exception text.
                await self._release()
                response = _error(-32000, identity)
            if notification:
                return
            delivery = self._native_delivery(result) if success else None
            frame = _canonical(response) + b"\n"
            if len(frame) > MAX_RESPONSE_BYTES or self._bytes + len(frame) > MAX_SESSION_BYTES:
                await self._release()
                raise PythonCodeViewMCPTransportError("task channel budget exhausted")
            self._bytes += len(frame)
            # No custody/visibility advance before the entire real write.
            await send(frame)
            if delivery is not None:
                self._visible[delivery[0]] = delivery[1]
            if success and request["method"] == "initialize":
                self._state = "initializing"
            if self._adapter is not None and self._adapter._closed:
                await self._release()
        except BaseException as exc:
            await self._release()
            if isinstance(exc, anyio.get_cancelled_exc_class()):
                raise
            raise PythonCodeViewMCPTransportError("task channel delivery or custody failed") from None
        finally:
            self._busy = False

    async def consume_candidate(self, candidate_id: str) -> PythonCodeViewCandidateHandoff:
        """Controller only: recheck, consume once, seal channel; never apply.

        The immutable visible map accompanies the private prepared patch for
        the existing native validation/apply stage. Closing the broker does
        not close or replace its native worktree/lifecycle owner.
        """
        if self._closed or self._state != "ready":
            raise PythonCodeViewMCPTransportError("a ready idle owned candidate channel is required")
        self._enter()
        try:
            visible = dict(self._visible)
            prepared = await anyio.to_thread.run_sync(
                lambda: self._adapter.consume_candidate(candidate_id, visible_sources=visible),
                abandon_on_cancel=False,
            )
            handoff = PythonCodeViewCandidateHandoff(prepared, MappingProxyType(visible))
            await self._release()
            return handoff
        except BaseException as exc:
            await self._release()
            if isinstance(exc, anyio.get_cancelled_exc_class()):
                raise
            raise PythonCodeViewMCPTransportError("current owned candidate handoff required") from None
        finally:
            self._busy = False

    async def aclose(self) -> None:
        """Controller cleanup; an active exchange must finish or be cancelled."""
        self._enter()
        try:
            await self._release()
        finally:
            self._busy = False
