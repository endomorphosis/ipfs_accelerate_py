"""Runtime MCP dual-binding adapter for the accelerate MCP++ server.

Interface: RuntimeBindingAdapter@1
Task: MCPP-023

Implements path selection and advertisement for:

* ``mcp-binding/legacy-2024-11-05`` — initialize-era session lifecycle
* ``mcp-binding/2026-07-28`` — stateless per-request ``_meta`` (no initialize)

A peer advertises only the bindings it implements and rejects all others
fail-closed. The current path never silently accepts ``initialize``.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Identity pins (BindingAndA2ADecision@1 / BindingCompatibilityMatrix@1)
# ---------------------------------------------------------------------------

INTERFACE_LABEL = "RuntimeBindingAdapter@1"

LEGACY_BINDING_ID = "mcp-binding/legacy-2024-11-05"
CURRENT_BINDING_ID = "mcp-binding/2026-07-28"

LEGACY_PROTOCOL_VERSION = "2024-11-05"
CURRENT_PROTOCOL_VERSION = "2026-07-28"

KNOWN_BINDING_IDS = frozenset({LEGACY_BINDING_ID, CURRENT_BINDING_ID})
KNOWN_PROTOCOL_VERSIONS = frozenset({LEGACY_PROTOCOL_VERSION, CURRENT_PROTOCOL_VERSION})

# Current-path per-request _meta keys
META_PROTOCOL_VERSION = "io.modelcontextprotocol/protocolVersion"
META_CLIENT_CAPS = "io.modelcontextprotocol/clientCapabilities"
META_CLIENT_INFO = "io.modelcontextprotocol/clientInfo"
META_SERVER_INFO = "io.modelcontextprotocol/serverInfo"
META_BINDING_ID = "io.mcplusplus/bindingId"

EXT_TASKS = "io.modelcontextprotocol/tasks"
EXT_PROFILES = "io.mcplusplus/profiles"

# JSON-RPC error codes
ERR_METHOD_NOT_FOUND = -32601
ERR_INVALID_PARAMS = -32602
ERR_INVALID_REQUEST = -32600
ERR_UNSUPPORTED_PROTOCOL_VERSION = -32022
ERR_NOT_INITIALIZED = -32000

# Matrix reason codes
REASON_FORGED_VERSION = "forged_version"
REASON_BINDING_MISMATCH = "binding_id_mismatch"
REASON_SILENT_DOWNGRADE = "silent_downgrade_rejected"
REASON_INIT_AS_CURRENT = "initialize_as_current_rejected"
REASON_BINDING_NOT_OFFERED = "binding_not_offered"
REASON_PATH_AMBIGUOUS = "path_ambiguous"
REASON_VERSION_BINDING_MISMATCH = "version_binding_mismatch"
REASON_NOT_INITIALIZED = "not_initialized"
REASON_BINDING_NAME_REQUIRED = "binding_name_required"

DEFAULT_PROFILES: Tuple[str, ...] = (
    "mcp++/mcp-idl",
    "mcp++/cid-envelope",
    "mcp++/ucan",
    "mcp++/deontic-policy",
    "mcp++/p2p-transport",
    "mcp++/event-dag",
    "mcp++/risk-scheduling",
    "mcp++/x402-payments",
)

LEGACY_LIFECYCLE_METHODS = frozenset(
    {"initialize", "notifications/initialized", "initialized"}
)


class PeerMode(str, Enum):
    """Which bindings this runtime peer implements and advertises."""

    LEGACY_ONLY = "legacy-only"
    CURRENT_ONLY = "current-only"
    DUAL = "dual"


class SessionPhase(Enum):
    """Legacy session phases under mcp-binding/legacy-2024-11-05."""

    UNINITIALIZED = auto()
    INITIALIZED = auto()  # InitializeResult sent; awaiting notifications/initialized
    READY = auto()


@dataclass
class BindingResponse:
    """JSON-RPC style response from the runtime binding adapter."""

    id: Any
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None
    is_notification_ack: bool = False
    path: Optional[str] = None  # "legacy" | "current" | None

    @property
    def ok(self) -> bool:
        if self.is_notification_ack:
            return self.error is None
        return self.error is None and self.result is not None

    def as_jsonrpc(self) -> Dict[str, Any]:
        body: Dict[str, Any] = {"jsonrpc": "2.0", "id": self.id}
        if self.error is not None:
            body["error"] = self.error
        elif self.result is not None:
            body["result"] = self.result
        return body


def mode_to_bindings(mode: PeerMode) -> Tuple[str, ...]:
    """Return the binding ids honestly implemented for ``mode``."""
    if mode is PeerMode.LEGACY_ONLY:
        return (LEGACY_BINDING_ID,)
    if mode is PeerMode.CURRENT_ONLY:
        return (CURRENT_BINDING_ID,)
    return (CURRENT_BINDING_ID, LEGACY_BINDING_ID)


def mode_to_versions(mode: PeerMode) -> Tuple[str, ...]:
    """Return the protocol versions honestly supported for ``mode``."""
    if mode is PeerMode.LEGACY_ONLY:
        return (LEGACY_PROTOCOL_VERSION,)
    if mode is PeerMode.CURRENT_ONLY:
        return (CURRENT_PROTOCOL_VERSION,)
    return (CURRENT_PROTOCOL_VERSION, LEGACY_PROTOCOL_VERSION)


def extract_binding_and_profiles(
    capabilities: Dict[str, Any],
) -> Tuple[Optional[str], List[str], bool]:
    """Extract binding id, profile keys, and whether the client claims MCP++."""
    binding: Optional[str] = None
    profiles: List[str] = []
    claim_mcpp = False

    nested = capabilities.get("mcp++")
    if isinstance(nested, dict):
        claim_mcpp = True
        raw_binding = nested.get("bindingId")
        if isinstance(raw_binding, str):
            binding = raw_binding
        raw_profiles = nested.get("profiles")
        if isinstance(raw_profiles, list):
            profiles.extend(str(p) for p in raw_profiles)
        elif isinstance(raw_profiles, dict):
            profiles.extend(str(k) for k, v in raw_profiles.items() if v)

    experimental = capabilities.get("experimental")
    if isinstance(experimental, dict):
        exp_binding = experimental.get("mcp++/bindingId")
        if isinstance(exp_binding, str):
            binding = binding or exp_binding
            claim_mcpp = True
        for key, value in experimental.items():
            if key.startswith("mcp++/") and key != "mcp++/bindingId" and value:
                claim_mcpp = True
                profiles.append(key)

    seen: Set[str] = set()
    ordered: List[str] = []
    for p in profiles:
        if p not in seen:
            seen.add(p)
            ordered.append(p)
    return binding, ordered, claim_mcpp


def _default_tools() -> Dict[str, Dict[str, Any]]:
    return {
        "echo": {
            "name": "echo",
            "description": "Echo arguments (runtime binding smoke tool)",
            "inputSchema": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
            },
        }
    }


@dataclass
class RuntimeBindingAdapter:
    """RuntimeBindingAdapter@1 for accelerate / datasets MCP++ servers.

    Advertises only the bindings configured via ``mode`` and rejects every
    other binding or forged version/id pair fail-closed. Current-path
    requests never require or silently perform ``initialize``.
    """

    mode: PeerMode = PeerMode.DUAL
    runtime: str = "accelerate"
    server_name: str = "ipfs-accelerate-mcp++"
    server_version: str = "1.0.0"
    profiles: Set[str] = field(default_factory=lambda: set(DEFAULT_PROFILES))
    tools: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Connection-scoped negotiation state
    phase: SessionPhase = SessionPhase.UNINITIALIZED
    active_binding: Optional[str] = None
    negotiated_version: Optional[str] = None
    negotiated_client_binding: Optional[str] = None
    negotiated_profiles: Set[str] = field(default_factory=set)
    client_info: Optional[Dict[str, Any]] = None

    # Observability
    initialize_calls: int = 0
    initialized_notifications: int = 0
    request_count: int = 0
    legacy_successes: int = 0
    current_successes: int = 0
    rejected_downgrades: int = 0
    rejected_forgeries: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.mode, str):
            self.mode = PeerMode(self.mode)
        if not self.tools:
            self.tools = _default_tools()
        if not self.profiles:
            self.profiles = set(DEFAULT_PROFILES)

    # -- advertisement ------------------------------------------------------

    @property
    def offers_legacy(self) -> bool:
        return self.mode in (PeerMode.LEGACY_ONLY, PeerMode.DUAL)

    @property
    def offers_current(self) -> bool:
        return self.mode in (PeerMode.CURRENT_ONLY, PeerMode.DUAL)

    def implemented_bindings(self) -> List[str]:
        """Binding ids this runtime actually implements (honest advertisement)."""
        return list(mode_to_bindings(self.mode))

    def supported_versions(self) -> List[str]:
        return list(mode_to_versions(self.mode))

    def implements(self, binding_id: str) -> bool:
        return binding_id in mode_to_bindings(self.mode)

    def advertisement(self) -> Dict[str, Any]:
        """Capability / discovery advertisement for the configured mode."""
        bindings = self.implemented_bindings()
        versions = self.supported_versions()
        primary = (
            CURRENT_BINDING_ID
            if self.offers_current
            else LEGACY_BINDING_ID
        )
        return {
            "interface": INTERFACE_LABEL,
            "runtime": self.runtime,
            "mode": self.mode.value,
            "supportedBindings": bindings,
            "supportedVersions": versions,
            "bindingId": primary,
            "capabilities": {
                "mcp++": {
                    "bindingId": primary,
                    "bindingIds": bindings,
                    "profiles": sorted(self.profiles),
                    "supportedVersions": versions,
                }
            },
            "serverInfo": {
                "name": self.server_name,
                "version": self.server_version,
            },
        }

    def reset(self) -> None:
        """Reset connection-scoped negotiation state."""
        self.phase = SessionPhase.UNINITIALIZED
        self.active_binding = None
        self.negotiated_version = None
        self.negotiated_client_binding = None
        self.negotiated_profiles = set()
        self.client_info = None
        self.initialize_calls = 0
        self.initialized_notifications = 0

    # -- dispatch (BindingPathSelect@1) -------------------------------------

    def handle(self, message: Dict[str, Any]) -> BindingResponse:
        """Select path and enforce dual-binding fail-closed rules."""
        self.request_count += 1
        msg_id = message.get("id")
        method = message.get("method")

        if message.get("jsonrpc") != "2.0":
            return self._error(msg_id, ERR_INVALID_PARAMS, "jsonrpc must be '2.0'")

        if not method or not isinstance(method, str):
            return self._error(msg_id, ERR_INVALID_PARAMS, "missing method")

        # Priority 1: initialize-family → legacy path (or reject if not offered).
        if method in LEGACY_LIFECYCLE_METHODS:
            return self._handle_legacy_lifecycle(message)

        params = message.get("params") or {}
        if params is not None and not isinstance(params, dict):
            return self._error(msg_id, ERR_INVALID_PARAMS, "params must be an object")
        if not isinstance(params, dict):
            params = {}

        meta = params.get("_meta")
        has_current_meta = isinstance(meta, dict) and (
            META_PROTOCOL_VERSION in meta or META_BINDING_ID in meta
        )

        # Priority 2: current-shaped _meta → current path.
        if has_current_meta:
            return self._handle_current_path(message, meta)

        # Priority 3: open legacy session, bare application methods.
        if self.offers_legacy and self.phase is not SessionPhase.UNINITIALIZED:
            if self.active_binding == CURRENT_BINDING_ID:
                return self._reject_silent_downgrade(
                    msg_id,
                    detail="bare legacy application method after current path",
                )
            return self._handle_legacy_application(message, params)

        # No usable path.
        if self.offers_legacy and not self.offers_current:
            return self._error(
                msg_id,
                ERR_NOT_INITIALIZED,
                "session not initialized; send initialize first",
                data={
                    "bindingId": LEGACY_BINDING_ID,
                    "reason": REASON_NOT_INITIALIZED,
                    "supportedVersions": self.supported_versions(),
                    "supportedBindings": self.implemented_bindings(),
                },
                path="legacy",
            )

        if self.offers_current and not has_current_meta:
            return self._error(
                msg_id,
                ERR_INVALID_PARAMS,
                "current path requires params._meta protocol version",
                data={
                    "reason": (
                        REASON_BINDING_NOT_OFFERED
                        if not self.offers_legacy
                        else REASON_PATH_AMBIGUOUS
                    ),
                    "supportedBindings": self.implemented_bindings(),
                    "supportedVersions": self.supported_versions(),
                    "missing": [META_PROTOCOL_VERSION],
                },
            )

        return self._error(
            msg_id,
            ERR_METHOD_NOT_FOUND,
            f"no binding path for method {method!r}",
            data={
                "reason": REASON_PATH_AMBIGUOUS,
                "supportedBindings": self.implemented_bindings(),
            },
        )

    # -- legacy path --------------------------------------------------------

    def _handle_legacy_lifecycle(self, message: Dict[str, Any]) -> BindingResponse:
        msg_id = message.get("id")
        method = message.get("method")

        if not self.offers_legacy:
            # Current-only: initialize-as-current rejected (no silent initialize).
            return self._reject_initialize_as_current(msg_id, method=str(method))

        # Silent downgrade: active current → legacy initialize family.
        if self.active_binding == CURRENT_BINDING_ID and method == "initialize":
            return self._reject_silent_downgrade(
                msg_id,
                detail="initialize after current binding became active",
            )

        if method == "initialize":
            return self._handle_initialize(msg_id, message)

        if method in ("notifications/initialized", "initialized"):
            return self._handle_initialized_notification(msg_id)

        return self._error(
            msg_id, ERR_METHOD_NOT_FOUND, f"Method not found: {method}", path="legacy"
        )

    def _handle_initialize(
        self, msg_id: Any, message: Dict[str, Any]
    ) -> BindingResponse:
        self.initialize_calls += 1
        params = message.get("params") or {}
        if not isinstance(params, dict):
            return self._error(
                msg_id, ERR_INVALID_PARAMS, "params must be an object", path="legacy"
            )

        version = params.get("protocolVersion")
        if version is None:
            return self._error(
                msg_id,
                ERR_INVALID_PARAMS,
                "missing required params.protocolVersion",
                data={"missing": ["protocolVersion"]},
                path="legacy",
            )

        # Modern version on initialize is never promotion to current.
        if version == CURRENT_PROTOCOL_VERSION:
            self.rejected_forgeries += 1
            return self._error(
                msg_id,
                ERR_UNSUPPORTED_PROTOCOL_VERSION,
                "modern protocol version is not valid on initialize path",
                data={
                    "reason": REASON_VERSION_BINDING_MISMATCH,
                    "requested": version,
                    "supported": (
                        [LEGACY_PROTOCOL_VERSION]
                        if self.mode is PeerMode.LEGACY_ONLY
                        else self.supported_versions()
                    ),
                    "path": "legacy",
                    "bindingId": LEGACY_BINDING_ID,
                },
                path="legacy",
            )

        if version != LEGACY_PROTOCOL_VERSION:
            self.rejected_forgeries += 1
            return self._error(
                msg_id,
                ERR_UNSUPPORTED_PROTOCOL_VERSION,
                "Unsupported protocol version",
                data={
                    "supported": (
                        [LEGACY_PROTOCOL_VERSION]
                        if self.mode is PeerMode.LEGACY_ONLY
                        else self.supported_versions()
                    ),
                    "requested": version,
                    "bindingId": LEGACY_BINDING_ID,
                    "reason": REASON_FORGED_VERSION,
                },
                path="legacy",
            )

        capabilities = params.get("capabilities")
        if capabilities is None:
            capabilities = {}
        if not isinstance(capabilities, dict):
            return self._error(
                msg_id,
                ERR_INVALID_PARAMS,
                "params.capabilities must be an object",
                path="legacy",
            )

        client_binding, client_profiles, claim_mcpp = extract_binding_and_profiles(
            capabilities
        )

        # MCP++ 1.0: binding name is mandatory when the client claims MCP++.
        if claim_mcpp:
            if not client_binding:
                return self._error(
                    msg_id,
                    ERR_INVALID_PARAMS,
                    "binding name is mandatory in capability advertisement",
                    data={
                        "expected": LEGACY_BINDING_ID,
                        "reason": REASON_BINDING_NAME_REQUIRED,
                        "bindingId": LEGACY_BINDING_ID,
                    },
                    path="legacy",
                )
            if client_binding == CURRENT_BINDING_ID:
                self.rejected_forgeries += 1
                return self._error(
                    msg_id,
                    ERR_INVALID_PARAMS,
                    "current binding id is not valid on initialize path",
                    data={
                        "reason": REASON_BINDING_MISMATCH,
                        "expected": LEGACY_BINDING_ID,
                        "requested": client_binding,
                        "path": "legacy",
                    },
                    path="legacy",
                )
            if client_binding != LEGACY_BINDING_ID:
                if not self.implements(client_binding):
                    self.rejected_forgeries += 1
                    return self._error(
                        msg_id,
                        ERR_INVALID_PARAMS,
                        "binding is not implemented by this runtime",
                        data={
                            "reason": REASON_BINDING_NOT_OFFERED,
                            "requested": client_binding,
                            "supportedBindings": self.implemented_bindings(),
                        },
                        path="legacy",
                    )
                self.rejected_forgeries += 1
                return self._error(
                    msg_id,
                    ERR_INVALID_PARAMS,
                    "binding id does not match legacy binding",
                    data={
                        "expected": LEGACY_BINDING_ID,
                        "requested": client_binding,
                        "reason": REASON_BINDING_MISMATCH,
                    },
                    path="legacy",
                )

        client_info = params.get("clientInfo")
        if client_info is not None and not isinstance(client_info, dict):
            return self._error(
                msg_id,
                ERR_INVALID_PARAMS,
                "params.clientInfo must be an object",
                path="legacy",
            )

        self.phase = SessionPhase.INITIALIZED
        self.negotiated_version = version
        self.negotiated_client_binding = client_binding or LEGACY_BINDING_ID
        self.client_info = copy.deepcopy(client_info) if client_info else None
        if client_profiles:
            self.negotiated_profiles = set(client_profiles) & set(self.profiles)
        else:
            self.negotiated_profiles = set(self.profiles)
        self.active_binding = LEGACY_BINDING_ID
        self.legacy_successes += 1

        return BindingResponse(
            id=msg_id,
            result=self._initialize_result(),
            path="legacy",
        )

    def _handle_initialized_notification(self, msg_id: Any) -> BindingResponse:
        self.initialized_notifications += 1
        if self.phase is SessionPhase.UNINITIALIZED:
            return self._error(
                msg_id if msg_id is not None else None,
                ERR_NOT_INITIALIZED,
                "notifications/initialized without prior initialize",
                data={"reason": REASON_NOT_INITIALIZED, "bindingId": LEGACY_BINDING_ID},
                path="legacy",
            )
        self.phase = SessionPhase.READY
        self.active_binding = LEGACY_BINDING_ID
        return BindingResponse(id=msg_id, is_notification_ack=True, path="legacy")

    def _handle_legacy_application(
        self, message: Dict[str, Any], params: Dict[str, Any]
    ) -> BindingResponse:
        msg_id = message.get("id")
        method = message.get("method")

        if method == "tools/list":
            self.legacy_successes += 1
            self.active_binding = LEGACY_BINDING_ID
            return BindingResponse(
                id=msg_id,
                result={
                    "tools": list(self.tools.values()),
                    "bindingId": LEGACY_BINDING_ID,
                },
                path="legacy",
            )
        if method == "tools/call":
            return self._tools_call_legacy(msg_id, params)
        if method == "ping":
            self.legacy_successes += 1
            self.active_binding = LEGACY_BINDING_ID
            return BindingResponse(id=msg_id, result={}, path="legacy")

        return self._error(
            msg_id, ERR_METHOD_NOT_FOUND, f"Method not found: {method}", path="legacy"
        )

    def _tools_call_legacy(
        self, msg_id: Any, params: Dict[str, Any]
    ) -> BindingResponse:
        name = params.get("name")
        if not name or name not in self.tools:
            return self._error(
                msg_id, ERR_INVALID_PARAMS, f"unknown tool: {name!r}", path="legacy"
            )
        arguments = params.get("arguments") or {}
        text = arguments.get("text", "")
        self.legacy_successes += 1
        self.active_binding = LEGACY_BINDING_ID
        return BindingResponse(
            id=msg_id,
            result={
                "content": [{"type": "text", "text": str(text)}],
                "bindingId": LEGACY_BINDING_ID,
            },
            path="legacy",
        )

    def _initialize_result(self) -> Dict[str, Any]:
        mcpp: Dict[str, Any] = {
            "bindingId": LEGACY_BINDING_ID,
            "profiles": sorted(self.profiles),
        }
        if self.mode is PeerMode.DUAL:
            mcpp["bindingIds"] = self.implemented_bindings()
            mcpp["supportedVersions"] = self.supported_versions()
        result: Dict[str, Any] = {
            "protocolVersion": LEGACY_PROTOCOL_VERSION,
            "capabilities": {
                "tools": {"listChanged": True},
                "experimental": {},
                "mcp++": mcpp,
            },
            "serverInfo": {
                "name": self.server_name,
                "version": self.server_version,
            },
        }
        if self.mode is PeerMode.DUAL:
            result["supportedVersions"] = self.supported_versions()
            result["supportedBindings"] = self.implemented_bindings()
        return result

    # -- current path -------------------------------------------------------

    def _handle_current_path(
        self, message: Dict[str, Any], meta: Dict[str, Any]
    ) -> BindingResponse:
        msg_id = message.get("id")
        method = message.get("method")

        if not self.offers_current:
            self.rejected_forgeries += 1
            return self._error(
                msg_id,
                ERR_UNSUPPORTED_PROTOCOL_VERSION,
                "current binding is not offered by this peer",
                data={
                    "reason": REASON_BINDING_NOT_OFFERED,
                    "requested": meta.get(META_PROTOCOL_VERSION),
                    "requestedBinding": meta.get(META_BINDING_ID),
                    "supported": self.supported_versions(),
                    "supportedBindings": self.implemented_bindings(),
                },
            )

        version = meta.get(META_PROTOCOL_VERSION)
        claimed_binding = meta.get(META_BINDING_ID)

        if version is None:
            return self._error(
                msg_id,
                ERR_INVALID_PARAMS,
                f"missing required _meta key {META_PROTOCOL_VERSION}",
                data={"missing": [META_PROTOCOL_VERSION]},
                path="current",
            )

        if version == LEGACY_PROTOCOL_VERSION:
            self.rejected_forgeries += 1
            return self._error(
                msg_id,
                ERR_UNSUPPORTED_PROTOCOL_VERSION,
                "legacy protocol version is not valid on current path",
                data={
                    "reason": REASON_FORGED_VERSION,
                    "requested": version,
                    "supported": (
                        [CURRENT_PROTOCOL_VERSION]
                        if self.mode is PeerMode.CURRENT_ONLY
                        else self.supported_versions()
                    ),
                    "path": "current",
                    "expectedBinding": CURRENT_BINDING_ID,
                },
                path="current",
            )

        if version != CURRENT_PROTOCOL_VERSION:
            self.rejected_forgeries += 1
            return self._error(
                msg_id,
                ERR_UNSUPPORTED_PROTOCOL_VERSION,
                "Unsupported protocol version",
                data={
                    "supported": (
                        [CURRENT_PROTOCOL_VERSION]
                        if self.mode is PeerMode.CURRENT_ONLY
                        else self.supported_versions()
                    ),
                    "requested": version,
                    "reason": REASON_FORGED_VERSION,
                },
                path="current",
            )

        caps = meta.get(META_CLIENT_CAPS)
        if caps is None or not isinstance(caps, dict):
            return self._error(
                msg_id,
                ERR_INVALID_PARAMS,
                f"missing required _meta key {META_CLIENT_CAPS}",
                data={"missing": [META_CLIENT_CAPS]},
                path="current",
            )

        if (
            claimed_binding == CURRENT_BINDING_ID
            and version != CURRENT_PROTOCOL_VERSION
        ):
            self.rejected_forgeries += 1
            return self._error(
                msg_id,
                ERR_INVALID_PARAMS,
                "forged protocol version for current binding id",
                data={
                    "reason": REASON_FORGED_VERSION,
                    "requested": version,
                    "requestedBinding": claimed_binding,
                    "expected": CURRENT_PROTOCOL_VERSION,
                },
                path="current",
            )

        if (
            claimed_binding == LEGACY_BINDING_ID
            and version == CURRENT_PROTOCOL_VERSION
        ):
            self.rejected_forgeries += 1
            return self._error(
                msg_id,
                ERR_INVALID_PARAMS,
                "legacy binding id is not valid on current path",
                data={
                    "reason": REASON_BINDING_MISMATCH,
                    "expected": CURRENT_BINDING_ID,
                    "requested": claimed_binding,
                    "path": "current",
                },
                path="current",
            )

        if claimed_binding is not None and claimed_binding not in KNOWN_BINDING_IDS:
            self.rejected_forgeries += 1
            return self._error(
                msg_id,
                ERR_INVALID_PARAMS,
                "binding is not implemented by this runtime",
                data={
                    "reason": REASON_BINDING_NOT_OFFERED,
                    "requested": claimed_binding,
                    "supportedBindings": self.implemented_bindings(),
                },
                path="current",
            )

        if (
            claimed_binding is not None
            and claimed_binding != CURRENT_BINDING_ID
            and not self.implements(claimed_binding)
        ):
            self.rejected_forgeries += 1
            return self._error(
                msg_id,
                ERR_INVALID_PARAMS,
                "binding is not implemented by this runtime",
                data={
                    "reason": REASON_BINDING_NOT_OFFERED,
                    "requested": claimed_binding,
                    "supportedBindings": self.implemented_bindings(),
                },
                path="current",
            )

        # Application methods on current path (no initialize ever required).
        if method == "server/discover":
            result = self._discover_result()
            self.active_binding = CURRENT_BINDING_ID
            self.current_successes += 1
            return BindingResponse(id=msg_id, result=result, path="current")

        if method == "tools/list":
            self.active_binding = CURRENT_BINDING_ID
            self.current_successes += 1
            return BindingResponse(
                id=msg_id,
                result={
                    "resultType": "complete",
                    "tools": list(self.tools.values()),
                    "_meta": self._server_meta(),
                },
                path="current",
            )

        if method == "tools/call":
            return self._tools_call_current(msg_id, message.get("params") or {})

        if method == "ping":
            self.active_binding = CURRENT_BINDING_ID
            self.current_successes += 1
            return BindingResponse(
                id=msg_id,
                result={"_meta": self._server_meta()},
                path="current",
            )

        return self._error(
            msg_id,
            ERR_METHOD_NOT_FOUND,
            f"Method not found: {method}",
            path="current",
        )

    def _tools_call_current(
        self, msg_id: Any, params: Dict[str, Any]
    ) -> BindingResponse:
        name = params.get("name")
        if not name or name not in self.tools:
            return self._error(
                msg_id, ERR_INVALID_PARAMS, f"unknown tool: {name!r}", path="current"
            )
        arguments = params.get("arguments") or {}
        text = arguments.get("text", "")
        self.active_binding = CURRENT_BINDING_ID
        self.current_successes += 1
        return BindingResponse(
            id=msg_id,
            result={
                "resultType": "complete",
                "content": [{"type": "text", "text": str(text)}],
                "_meta": self._server_meta(),
            },
            path="current",
        )

    def _server_meta(self) -> Dict[str, Any]:
        return {
            META_SERVER_INFO: {
                "name": self.server_name,
                "version": self.server_version,
            },
            META_BINDING_ID: CURRENT_BINDING_ID,
        }

    def _discover_result(self) -> Dict[str, Any]:
        profile_map = {k: True for k in sorted(self.profiles)}
        mcpp: Dict[str, Any] = {
            "bindingId": CURRENT_BINDING_ID,
            "profiles": sorted(self.profiles),
        }
        if self.mode is PeerMode.DUAL:
            mcpp["bindingIds"] = self.implemented_bindings()
        result: Dict[str, Any] = {
            "resultType": "complete",
            "supportedVersions": self.supported_versions(),
            "capabilities": {
                "tools": {},
                "extensions": {
                    EXT_TASKS: {},
                    EXT_PROFILES: profile_map,
                },
                "mcp++": mcpp,
            },
            "_meta": self._server_meta(),
        }
        if self.mode is PeerMode.DUAL:
            result["supportedBindings"] = self.implemented_bindings()
        return result

    # -- shared rejections --------------------------------------------------

    def _reject_initialize_as_current(
        self, msg_id: Any, *, method: str
    ) -> BindingResponse:
        """Reject initialize-family methods on current-only (no silent init)."""
        self.initialize_calls += 1
        if method == "initialize":
            message = f"initialize is not supported under {CURRENT_BINDING_ID}"
        else:
            message = (
                f"{method} is not supported under {CURRENT_BINDING_ID}"
            )
        return self._error(
            msg_id,
            ERR_METHOD_NOT_FOUND,
            message,
            data={
                "bindingId": CURRENT_BINDING_ID,
                "supportedVersions": [CURRENT_PROTOCOL_VERSION],
                "supportedBindings": self.implemented_bindings(),
                "reason": REASON_INIT_AS_CURRENT,
            },
            path="current",
        )

    def _reject_silent_downgrade(
        self, msg_id: Any, *, detail: str
    ) -> BindingResponse:
        self.rejected_downgrades += 1
        return self._error(
            msg_id,
            ERR_INVALID_PARAMS,
            f"silent downgrade rejected: {detail}",
            data={
                "reason": REASON_SILENT_DOWNGRADE,
                "activeBinding": self.active_binding,
                "supportedBindings": self.implemented_bindings(),
                "detail": detail,
            },
        )

    def _error(
        self,
        msg_id: Any,
        code: int,
        message: str,
        data: Optional[Dict[str, Any]] = None,
        path: Optional[str] = None,
    ) -> BindingResponse:
        err: Dict[str, Any] = {"code": code, "message": message}
        if data is not None:
            err["data"] = data
        return BindingResponse(id=msg_id, error=err, path=path)


# ---------------------------------------------------------------------------
# Client helpers (for tests and runtime callers)
# ---------------------------------------------------------------------------


def legacy_initialize_params(
    *,
    protocol_version: str = LEGACY_PROTOCOL_VERSION,
    client_name: str = "legacy-runtime-client",
    client_version: str = "1.0.0",
    profiles: Optional[List[str]] = None,
    include_binding_id: bool = True,
    binding_id: str = LEGACY_BINDING_ID,
    form: str = "nested",
) -> Dict[str, Any]:
    """Build initialize params for a legacy MCP++ client."""
    capabilities: Dict[str, Any] = {"tools": {}, "experimental": {}}
    profile_list = list(profiles) if profiles is not None else ["mcp++/cid-envelope"]

    if form == "nested":
        mcpp: Dict[str, Any] = {"profiles": profile_list}
        if include_binding_id:
            mcpp["bindingId"] = binding_id
        capabilities["mcp++"] = mcpp
    elif form == "experimental":
        experimental: Dict[str, Any] = {}
        if include_binding_id:
            experimental["mcp++/bindingId"] = binding_id
        for p in profile_list:
            experimental[p] = True
        capabilities["experimental"] = experimental
    elif form == "baseline_only":
        pass
    else:
        raise ValueError(f"unknown form: {form}")

    return {
        "protocolVersion": protocol_version,
        "capabilities": capabilities,
        "clientInfo": {"name": client_name, "version": client_version},
    }


def current_request_meta(
    *,
    client_name: str = "current-runtime-client",
    client_version: str = "1.0.0",
    capabilities: Optional[Dict[str, Any]] = None,
    profiles: Optional[List[str]] = None,
    include_binding_id: bool = True,
    protocol_version: str = CURRENT_PROTOCOL_VERSION,
    binding_id: str = CURRENT_BINDING_ID,
) -> Dict[str, Any]:
    """Build per-request _meta for a modern current client (no initialize)."""
    caps: Dict[str, Any] = dict(capabilities or {})
    if profiles:
        extensions = dict(caps.get("extensions") or {})
        extensions[EXT_PROFILES] = {p: True for p in profiles}
        caps["extensions"] = extensions
        caps.setdefault("mcp++", {})
        if isinstance(caps["mcp++"], dict):
            caps["mcp++"] = dict(caps["mcp++"])
            caps["mcp++"]["profiles"] = list(profiles)
    meta: Dict[str, Any] = {
        META_PROTOCOL_VERSION: protocol_version,
        META_CLIENT_CAPS: caps,
        META_CLIENT_INFO: {"name": client_name, "version": client_version},
    }
    if include_binding_id:
        meta[META_BINDING_ID] = binding_id
    return meta


def make_legacy_request(
    method: str,
    *,
    req_id: Any = 1,
    params: Optional[Dict[str, Any]] = None,
    notification: bool = False,
) -> Dict[str, Any]:
    body: Dict[str, Any] = {"jsonrpc": "2.0", "method": method}
    if not notification:
        body["id"] = req_id
    if params is not None:
        body["params"] = params
    elif method == "initialize":
        body["params"] = legacy_initialize_params()
    return body


def make_current_request(
    method: str,
    *,
    req_id: Any = 1,
    params: Optional[Dict[str, Any]] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    body_params = dict(params or {})
    if meta is not None:
        body_params["_meta"] = meta
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "method": method,
        "params": body_params,
    }


def open_legacy_session(adapter: RuntimeBindingAdapter) -> BindingResponse:
    """Run initialize + notifications/initialized; return initialize response."""
    init_resp = adapter.handle(
        make_legacy_request("initialize", req_id=1, params=legacy_initialize_params())
    )
    if not init_resp.ok:
        return init_resp
    adapter.handle(
        make_legacy_request(
            "notifications/initialized",
            notification=True,
            params={},
        )
    )
    return init_resp


def create_runtime_binding_adapter(
    *,
    mode: PeerMode | str = PeerMode.DUAL,
    runtime: str = "accelerate",
    server_name: Optional[str] = None,
    server_version: str = "1.0.0",
    profiles: Optional[Set[str]] = None,
) -> RuntimeBindingAdapter:
    """Factory for RuntimeBindingAdapter@1 used by accelerate runtime."""
    if server_name is None:
        server_name = (
            "ipfs-accelerate-mcp++"
            if runtime == "accelerate"
            else f"ipfs-{runtime}-mcp++"
        )
    return RuntimeBindingAdapter(
        mode=PeerMode(mode) if isinstance(mode, str) else mode,
        runtime=runtime,
        server_name=server_name,
        server_version=server_version,
        profiles=set(profiles) if profiles is not None else set(DEFAULT_PROFILES),
    )


__all__ = [
    "INTERFACE_LABEL",
    "LEGACY_BINDING_ID",
    "CURRENT_BINDING_ID",
    "LEGACY_PROTOCOL_VERSION",
    "CURRENT_PROTOCOL_VERSION",
    "KNOWN_BINDING_IDS",
    "META_PROTOCOL_VERSION",
    "META_CLIENT_CAPS",
    "META_CLIENT_INFO",
    "META_SERVER_INFO",
    "META_BINDING_ID",
    "ERR_METHOD_NOT_FOUND",
    "ERR_INVALID_PARAMS",
    "ERR_UNSUPPORTED_PROTOCOL_VERSION",
    "ERR_NOT_INITIALIZED",
    "REASON_FORGED_VERSION",
    "REASON_BINDING_MISMATCH",
    "REASON_SILENT_DOWNGRADE",
    "REASON_INIT_AS_CURRENT",
    "REASON_BINDING_NOT_OFFERED",
    "REASON_PATH_AMBIGUOUS",
    "REASON_VERSION_BINDING_MISMATCH",
    "PeerMode",
    "SessionPhase",
    "BindingResponse",
    "RuntimeBindingAdapter",
    "mode_to_bindings",
    "mode_to_versions",
    "extract_binding_and_profiles",
    "legacy_initialize_params",
    "current_request_meta",
    "make_legacy_request",
    "make_current_request",
    "open_legacy_session",
    "create_runtime_binding_adapter",
]
