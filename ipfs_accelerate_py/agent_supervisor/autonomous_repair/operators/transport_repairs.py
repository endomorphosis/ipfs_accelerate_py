"""Transport, lifecycle, and browser mediation repair operators (DCR-044).

Interfaces
----------
* ``TransportRepairOperators@1`` — finite structural operators that bind
  endpoint/transport adapters, lifecycle/capability truth reports, and
  desktop same-origin mediation policy without inventing handlers or
  authority.
* ``GovernedMcpMediator@1`` — the single browser-facing mutation surface;
  raw service proxies are read-only allowlisted or rejected.

Evidence term: ``dcr/transport-repair@1``.

Predicted symbols
-----------------
* :class:`TransportBindingOperator`
* :class:`LifecycleBindingOperator`
* :class:`BrowserMediationOperator`

Normative rules (fail-closed)
-----------------------------
* Mutation must traverse one governed mediator; raw service proxies may only
  forward reviewed read-only methods/paths.
* Health and ``initialize`` alone never establish capability availability —
  they yield ``typed_unavailable`` / ``initialized_not_available``.
* Operators remain proposal-only: they never grant write, proof, or semantic
  authority and never mutate production trees.
* Evidence subset: endpoint identity, route kind, method/effect class,
  middleware transcript, rollback.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, ClassVar, Final

from ...proof.formal_verification_contracts import (
    CanonicalContract,
    ContractValidationError,
    content_identity,
)
from .registry import (
    OperatorFamily,
    OperatorKind,
    build_default_operator_registry,
)


# ---------------------------------------------------------------------------
# Closed interface / evidence constants
# ---------------------------------------------------------------------------

TRANSPORT_REPAIR_OPERATORS_INTERFACE: Final[str] = "TransportRepairOperators@1"
GOVERNED_MCP_MEDIATOR_INTERFACE: Final[str] = "GovernedMcpMediator@1"
TRANSPORT_REPAIR_EVIDENCE: Final[str] = "dcr/transport-repair@1"
TRANSPORT_REPAIR_VERSION: Final[int] = 1

TRANSPORT_REPAIR_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/transport-repair@1"
)
TRANSPORT_BINDING_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/transport-endpoint-binding@1"
)
LIFECYCLE_BINDING_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/transport-lifecycle-binding@1"
)
BROWSER_MEDIATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/browser-mediation-policy@1"
)
CAPABILITY_REPORT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/transport-capability-report@1"
)
MIDDLEWARE_TRANSCRIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/transport-middleware-transcript@1"
)
TRANSPORT_REQUEST_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/transport-repair-request@1"
)
TRANSPORT_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/transport-repair-receipt@1"
)
TRANSPORT_PREVIEW_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/transport-repair-preview@1"
)
TRANSPORT_OPERATOR_VECTORS_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/transport-operator-vectors@1"
)

GOVERNED_MUTATION_ROUTE: Final[str] = "/mcp/tools/call"
SERVICE_PROXY_PREFIX: Final[str] = "/mcp/services/"

# Read-only JSON-RPC methods admissible on the raw same-origin service proxy.
SERVICE_PROXY_READ_ONLY_JSONRPC_METHODS: Final[frozenset[str]] = frozenset(
    {
        "initialize",
        "ping",
        "tools/list",
        "tools/get",
        "resources/list",
        "resources/read",
        "resources/templates/list",
        "prompts/list",
        "prompts/get",
        "interfaces/list",
        "interfaces/get",
        "interfaces/compat",
        "interfaces/select",
        "mcp++/p2p/peers",
        "mcp++/dag/frontier",
        "mcp++/dag/archives",
        "mcp++/dag/history",
        "mcp++/dag/provenance",
        "mcp++/dag/certificate/get",
        "mcp++/dag/certificate/verify",
        "mcp++/dag/inclusion",
        "mcp++/risk/profile",
        "mcp++/risk/evidence",
        "mcp++/risk/history",
        "mcp++/goals/list",
        "mcp++/goals/get",
        "mcp++/tasks/list",
        "mcp++/tasks/ready",
        "mcp++/tasks/get",
        "mcp++/schedule/frontier",
        "mcp++/schedule/status",
        "mcp++/neighborhood/query",
        "mcp++/artifacts/get",
        "mcp++/ucan/validate",
        "mcp++/ucan/identity",
        "mcp++/policy/evaluate",
    }
)

# Explicit mutation / write methods that must never traverse the raw proxy.
SERVICE_PROXY_MUTATION_JSONRPC_METHODS: Final[frozenset[str]] = frozenset(
    {
        "tools/call",
        "mcp++/execute",
        "mcp++/ucan/delegate",
        "mcp++/ucan/revoke",
        "mcp++/dag/append",
        "mcp++/dag/compact",
        "mcp++/dag/archive",
        "mcp++/goals/create",
        "mcp++/goals/decompose",
        "mcp++/goals/select",
        "mcp++/tasks/create",
        "mcp++/risk/assess",
        "mcp++/neighborhood/attest",
        "mcp++/schedule/propose",
        "mcp++/schedule/claim",
        "mcp++/schedule/renew",
        "mcp++/schedule/release",
        "mcp++/schedule/resolve",
        "mcp++/schedule/reconcile",
        "mcp++/artifacts/put",
    }
)

SERVICE_PROXY_READ_ONLY_GET_SUFFIXES: Final[frozenset[str]] = frozenset(
    {
        "/mcp/health",
        "/mcp/helia/status",
        "/mcp",
    }
)

MAX_TEXT_BYTES: Final[int] = 4_096
MAX_ID_BYTES: Final[int] = 512
MAX_COLLECTION: Final[int] = 256
MAX_REASON_CODES: Final[int] = 32
MAX_TRANSCRIPT_ROWS: Final[int] = 64

_IDENTIFIER_RE: Final[re.Pattern[str]] = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}$"
)
_CID_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?:bafy|bagu|bafk|sha256:)[A-Za-z0-9:_-]{8,200}$"
)

_FORBIDDEN_PAYLOAD_KEYS: Final[frozenset[str]] = frozenset(
    {
        "source_body",
        "source_text",
        "code",
        "code_body",
        "shell",
        "shell_fragment",
        "command",
        "script",
        "callable",
        "dynamic_import",
        "exec",
        "eval",
        "llm_prompt",
        "prose",
        "patch_body",
        "diff_body",
        "handler_body",
        "private_key",
        "secret",
        "password",
    }
)

REVIEWED_SERVICE_OWNERS: Final[frozenset[str]] = frozenset(
    {
        "ipfs_kit_py",
        "ipfs_datasets_py",
        "ipfs_accelerate_py",
    }
)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class TransportRepairError(ContractValidationError):
    """Malformed transport repair input or closed-boundary violation."""


class TransportRepairAbstention(TransportRepairError):
    """Operator cannot proceed without inventing transport/lifecycle semantics."""


class RepairDisposition(str, Enum):  # noqa: UP042 - package supports Python 3.8
    """Closed outcomes for one transport repair attempt."""

    PREVIEW_READY = "preview_ready"
    ALREADY_ALIGNED = "already_aligned"
    ABSTAIN = "abstain"
    REJECTED = "rejected"


class OperatorRole(str, Enum):  # noqa: UP042
    """Closed operator roles for DCR-044."""

    TRANSPORT_BINDING = "transport_binding"
    LIFECYCLE_BINDING = "lifecycle_binding"
    BROWSER_MEDIATION = "browser_mediation"


class TransportProfile(str, Enum):  # noqa: UP042
    """Closed transport profiles admitted by repair operators."""

    IN_PROCESS = "in_process"
    STDIO = "stdio"
    HTTP = "http"
    HTTP_SSE = "http_sse"
    WEBSOCKET = "websocket"
    LIBP2P = "libp2p"


class RouteKind(str, Enum):  # noqa: UP042
    """Closed route kinds for endpoint bindings."""

    HEALTH = "health"
    INITIALIZE = "initialize"
    TOOLS_LIST = "tools_list"
    TOOLS_CALL = "tools_call"
    GOVERNED_MEDIATOR = "governed_mediator"
    SERVICE_PROXY = "service_proxy"
    DISCOVERY = "discovery"
    CAPABILITY = "capability"


class MethodEffectClass(str, Enum):  # noqa: UP042
    """Closed method/effect classes used in mediation and capability reports."""

    READ = "read"
    WRITE = "write"
    MUTATE = "mutate"
    NO_EFFECT = "no_effect"
    UNKNOWN = "unknown"


class CapabilityState(str, Enum):  # noqa: UP042
    """Closed capability truth states.  Health/initialize alone are not available."""

    UNREACHABLE = "unreachable"
    HEALTH_ONLY = "health_only"
    INITIALIZED_NOT_AVAILABLE = "initialized_not_available"
    TYPED_UNAVAILABLE = "typed_unavailable"
    TOOLS_DISCOVERED = "tools_discovered"
    AVAILABLE = "available"


class LifecyclePhase(str, Enum):  # noqa: UP042
    """Explicit lifecycle phases for timeout/cancellation glue."""

    UNBOUND = "unbound"
    HEALTH_PROBED = "health_probed"
    INITIALIZED = "initialized"
    CAPABILITY_READY = "capability_ready"
    AVAILABLE = "available"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"
    FAILED = "failed"


class MediationPathClass(str, Enum):  # noqa: UP042
    """Closed mediation path classes."""

    GOVERNED_MEDIATOR = "governed_mediator"
    RAW_SERVICE_PROXY = "raw_service_proxy"
    DIRECT_PROXY = "direct_proxy"  # rejected


class ProxyDecision(str, Enum):  # noqa: UP042
    """Closed decisions for raw service-proxy classification."""

    ALLOW_READ = "allow_read"
    REJECT_MUTATION = "reject_mutation"
    REJECT_UNKNOWN = "reject_unknown"
    REQUIRE_GOVERNED_MEDIATOR = "require_governed_mediator"


class AuthoritySource(str, Enum):  # noqa: UP042
    """Authority retained on transport repair artifacts."""

    REVIEWED = "reviewed"
    PRODUCTION = "production"
    FIXTURE = "fixture"
    MANIFEST = "manifest"
    INVENTED = "invented"
    PROSE_INFERRED = "prose_inferred"
    MISSING = "missing"

    @property
    def authorizes_transport_source(self) -> bool:
        return self in {
            AuthoritySource.REVIEWED,
            AuthoritySource.PRODUCTION,
            AuthoritySource.FIXTURE,
            AuthoritySource.MANIFEST,
        }

    @property
    def is_abstaining_source(self) -> bool:
        return self in {
            AuthoritySource.INVENTED,
            AuthoritySource.PROSE_INFERRED,
            AuthoritySource.MISSING,
        }


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _text(
    value: Any,
    name: str,
    *,
    required: bool = True,
    maximum: int = MAX_TEXT_BYTES,
    identifier: bool = False,
) -> str:
    if not isinstance(value, str):
        raise TransportRepairError(f"{name} must be a string")
    result = value.strip()
    if required and not result:
        raise TransportRepairError(f"{name} must not be empty")
    if "\x00" in result:
        raise TransportRepairError(f"{name} must not contain NUL")
    if len(result.encode("utf-8")) > maximum:
        raise TransportRepairError(f"{name} exceeds its byte bound")
    if identifier and result and not _IDENTIFIER_RE.fullmatch(result):
        raise TransportRepairError(f"{name} must be a closed identifier")
    return result


def _bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise TransportRepairError(f"{name} must be a boolean")
    return value


def _nonnegative(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TransportRepairError(f"{name} must be a non-negative integer")
    return value


def _enum(value: Any, kind: type[Enum], name: str) -> Any:
    try:
        return kind(str(getattr(value, "value", value)))
    except (TypeError, ValueError) as exc:
        raise TransportRepairError(f"unsupported {name}: {value!r}") from exc


def _cid(value: Any, name: str, *, required: bool = True) -> str:
    text = _text(value, name, required=required, maximum=MAX_ID_BYTES)
    if text and not _CID_RE.fullmatch(text):
        if not text.startswith("sha256:") and not text.startswith("b"):
            raise TransportRepairError(f"{name} must be a content identity")
    return text


def _string_tuple(
    values: Any,
    name: str,
    *,
    required: bool = False,
    maximum: int = MAX_COLLECTION,
    ordered: bool = False,
    identifier: bool = False,
) -> tuple[str, ...]:
    if values is None:
        items: Sequence[Any] = ()
    elif isinstance(values, str):
        items = (values,)
    elif isinstance(values, Sequence) and not isinstance(values, (bytes, bytearray)):
        items = values
    else:
        raise TransportRepairError(f"{name} must be a sequence of strings")
    if len(items) > maximum:
        raise TransportRepairError(f"{name} exceeds its item bound")
    result: list[str] = []
    seen: set[str] = set()
    for index, item in enumerate(items):
        text = _text(item, f"{name}[{index}]", identifier=identifier)
        if text in seen:
            raise TransportRepairError(f"{name} must not contain duplicates")
        seen.add(text)
        result.append(text)
    if required and not result:
        raise TransportRepairError(f"{name} must not be empty")
    if ordered:
        return tuple(result)
    return tuple(sorted(result))


def _reject_forbidden_fields(payload: Mapping[str, Any], *, label: str) -> None:
    for key in payload:
        lowered = str(key).strip().lower()
        if lowered in _FORBIDDEN_PAYLOAD_KEYS:
            raise TransportRepairError(
                f"{label} contains forbidden field {lowered!r}"
            )


def _same_origin_service_base(owner: str) -> str:
    return f"{SERVICE_PROXY_PREFIX}{owner}"


# ---------------------------------------------------------------------------
# Capability truth and service-proxy classification (shared with browser)
# ---------------------------------------------------------------------------


def classify_capability_state(
    *,
    health_ok: bool,
    initialize_ok: bool,
    tools_ok: bool = False,
    interfaces_ok: bool = False,
) -> CapabilityState:
    """Return capability truth from probe outcomes.

    Health and initialize alone never establish availability.
    """

    if not health_ok:
        return CapabilityState.UNREACHABLE
    if not initialize_ok:
        return CapabilityState.HEALTH_ONLY
    if tools_ok or interfaces_ok:
        if tools_ok:
            return CapabilityState.AVAILABLE
        return CapabilityState.TOOLS_DISCOVERED
    # Protocol handshake without a capability surface is typed unavailable.
    return CapabilityState.INITIALIZED_NOT_AVAILABLE


def capability_claims_available(state: CapabilityState | str) -> bool:
    """Whether a capability state may claim service availability."""

    state_enum = (
        state
        if isinstance(state, CapabilityState)
        else _enum(state, CapabilityState, "state")
    )
    return state_enum in {
        CapabilityState.AVAILABLE,
        CapabilityState.TOOLS_DISCOVERED,
    }


def classify_jsonrpc_effect(method: str | None) -> MethodEffectClass:
    """Classify a JSON-RPC method as read, mutate, or unknown."""

    if method is None or not str(method).strip():
        return MethodEffectClass.UNKNOWN
    text = str(method).strip()
    if text in SERVICE_PROXY_MUTATION_JSONRPC_METHODS:
        return MethodEffectClass.MUTATE
    if text in SERVICE_PROXY_READ_ONLY_JSONRPC_METHODS:
        return MethodEffectClass.READ
    # Heuristic closed markers for unlisted methods — still fail closed.
    lowered = text.lower()
    for marker in (
        "create",
        "append",
        "claim",
        "write",
        "mutate",
        "delete",
        "revoke",
        "delegate",
        "execute",
        "call",
        "put",
        "compact",
        "archive",
        "propose",
        "renew",
        "release",
        "resolve",
        "reconcile",
        "assess",
        "attest",
        "decompose",
        "select",
    ):
        if marker in lowered.split("/")[-1] or lowered.endswith(f"/{marker}"):
            return MethodEffectClass.MUTATE
    return MethodEffectClass.UNKNOWN


def classify_service_proxy_access(
    *,
    http_method: str,
    service_path: str,
    jsonrpc_method: str | None = None,
) -> dict[str, Any]:
    """Classify a browser request against the raw same-origin service proxy.

    Mutations are rejected and must use :data:`GOVERNED_MUTATION_ROUTE`.
    """

    method = _text(http_method, "http_method").upper()
    path = _text(service_path, "service_path")
    # Normalize path for suffix checks.
    path_only = path.split("?", 1)[0]
    if not path_only.startswith("/"):
        path_only = f"/{path_only}"

    if method == "GET":
        for suffix in SERVICE_PROXY_READ_ONLY_GET_SUFFIXES:
            if path_only == suffix or path_only.endswith(suffix):
                return {
                    "allowed": True,
                    "decision": ProxyDecision.ALLOW_READ.value,
                    "effect_class": MethodEffectClass.READ.value,
                    "mediation": MediationPathClass.RAW_SERVICE_PROXY.value,
                    "reason": "read_only_get_allowlisted",
                    "governed_route": GOVERNED_MUTATION_ROUTE,
                }
        return {
            "allowed": False,
            "decision": ProxyDecision.REJECT_UNKNOWN.value,
            "effect_class": MethodEffectClass.UNKNOWN.value,
            "mediation": MediationPathClass.RAW_SERVICE_PROXY.value,
            "reason": "get_path_not_allowlisted",
            "governed_route": GOVERNED_MUTATION_ROUTE,
        }

    if method != "POST":
        return {
            "allowed": False,
            "decision": ProxyDecision.REJECT_UNKNOWN.value,
            "effect_class": MethodEffectClass.UNKNOWN.value,
            "mediation": MediationPathClass.RAW_SERVICE_PROXY.value,
            "reason": f"http_method_not_admitted:{method}",
            "governed_route": GOVERNED_MUTATION_ROUTE,
        }

    effect = classify_jsonrpc_effect(jsonrpc_method)
    if effect is MethodEffectClass.MUTATE:
        return {
            "allowed": False,
            "decision": ProxyDecision.REQUIRE_GOVERNED_MEDIATOR.value,
            "effect_class": effect.value,
            "mediation": MediationPathClass.GOVERNED_MEDIATOR.value,
            "reason": "mutation_requires_governed_mediator",
            "jsonrpc_method": jsonrpc_method or "",
            "governed_route": GOVERNED_MUTATION_ROUTE,
        }
    if effect is MethodEffectClass.READ and jsonrpc_method in SERVICE_PROXY_READ_ONLY_JSONRPC_METHODS:
        return {
            "allowed": True,
            "decision": ProxyDecision.ALLOW_READ.value,
            "effect_class": effect.value,
            "mediation": MediationPathClass.RAW_SERVICE_PROXY.value,
            "reason": "read_only_jsonrpc_allowlisted",
            "jsonrpc_method": jsonrpc_method or "",
            "governed_route": GOVERNED_MUTATION_ROUTE,
        }
    return {
        "allowed": False,
        "decision": ProxyDecision.REJECT_UNKNOWN.value,
        "effect_class": effect.value,
        "mediation": MediationPathClass.RAW_SERVICE_PROXY.value,
        "reason": "jsonrpc_method_not_allowlisted",
        "jsonrpc_method": jsonrpc_method or "",
        "governed_route": GOVERNED_MUTATION_ROUTE,
    }


def assert_no_browser_mutation_bypass(
    *,
    http_method: str,
    service_path: str,
    jsonrpc_method: str | None = None,
    mediation_path: MediationPathClass | str | None = None,
) -> dict[str, Any]:
    """Fail closed when a browser mutation attempts to bypass the mediator."""

    classification = classify_service_proxy_access(
        http_method=http_method,
        service_path=service_path,
        jsonrpc_method=jsonrpc_method,
    )
    path = (
        mediation_path
        if isinstance(mediation_path, MediationPathClass)
        else (
            _enum(mediation_path, MediationPathClass, "mediation_path")
            if mediation_path
            else None
        )
    )
    if path is MediationPathClass.DIRECT_PROXY:
        raise TransportRepairError(
            "direct_proxy mediation is rejected; mutations must use GovernedMcpMediator"
        )
    if classification["effect_class"] == MethodEffectClass.MUTATE.value:
        if path is not None and path is not MediationPathClass.GOVERNED_MEDIATOR:
            raise TransportRepairError(
                "browser mutation bypass rejected: raw service proxies are read-only"
            )
        if classification["allowed"]:
            raise TransportRepairError(
                "invariant violation: mutation classified as allowed on raw proxy"
            )
    return classification


# ---------------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TransportEndpointBinding(CanonicalContract):
    """One reviewed endpoint/transport adapter binding."""

    SCHEMA: ClassVar[str] = TRANSPORT_BINDING_SCHEMA
    INTERFACE: ClassVar[str] = TRANSPORT_REPAIR_OPERATORS_INTERFACE

    binding_id: str
    owner: str
    endpoint_identity: str
    transport_profile: TransportProfile
    route_kind: RouteKind
    method: str
    effect_class: MethodEffectClass
    same_origin_path: str
    correlation_required: bool = True
    framing: str = "jsonrpc-2.0"
    authority: AuthoritySource = AuthoritySource.REVIEWED

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "binding_id", _text(self.binding_id, "binding_id", identifier=True)
        )
        owner = _text(self.owner, "owner", identifier=True)
        if owner not in REVIEWED_SERVICE_OWNERS:
            raise TransportRepairError(
                f"owner must be a reviewed service owner, got {owner!r}"
            )
        object.__setattr__(self, "owner", owner)
        object.__setattr__(
            self,
            "endpoint_identity",
            _text(self.endpoint_identity, "endpoint_identity", identifier=True),
        )
        object.__setattr__(
            self,
            "transport_profile",
            _enum(self.transport_profile, TransportProfile, "transport_profile"),
        )
        object.__setattr__(
            self, "route_kind", _enum(self.route_kind, RouteKind, "route_kind")
        )
        object.__setattr__(self, "method", _text(self.method, "method", identifier=True))
        object.__setattr__(
            self,
            "effect_class",
            _enum(self.effect_class, MethodEffectClass, "effect_class"),
        )
        path = _text(self.same_origin_path, "same_origin_path")
        expected_prefix = _same_origin_service_base(owner)
        # Absolute endpoints fail closed before same-origin prefix checks so
        # leaky backend URLs surface an absolute-backend reason code.
        if "://" in path or "localhost" in path or "127.0.0.1" in path:
            raise TransportRepairError(
                "same_origin_path must not embed absolute backend endpoints"
            )
        if not path.startswith(expected_prefix) and path != GOVERNED_MUTATION_ROUTE:
            raise TransportRepairError(
                "same_origin_path must be a same-origin mediated route"
            )
        object.__setattr__(self, "same_origin_path", path)
        object.__setattr__(
            self,
            "correlation_required",
            _bool(self.correlation_required, "correlation_required"),
        )
        object.__setattr__(self, "framing", _text(self.framing, "framing"))
        object.__setattr__(
            self, "authority", _enum(self.authority, AuthoritySource, "authority")
        )
        if self.authority.is_abstaining_source:
            raise TransportRepairError(
                "TransportEndpointBinding cannot carry invented authority"
            )
        if (
            self.effect_class is MethodEffectClass.MUTATE
            and self.route_kind is not RouteKind.GOVERNED_MEDIATOR
            and path != GOVERNED_MUTATION_ROUTE
        ):
            raise TransportRepairError(
                "mutate effect_class requires governed mediator route"
            )

    def _payload(self) -> dict[str, Any]:
        return {
            "interface": self.INTERFACE,
            "binding_id": self.binding_id,
            "owner": self.owner,
            "endpoint_identity": self.endpoint_identity,
            "transport_profile": self.transport_profile.value,
            "route_kind": self.route_kind.value,
            "method": self.method,
            "effect_class": self.effect_class.value,
            "same_origin_path": self.same_origin_path,
            "correlation_required": self.correlation_required,
            "framing": self.framing,
            "authority": self.authority.value,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TransportEndpointBinding":
        if not isinstance(payload, Mapping):
            raise TransportRepairError("transport binding must be an object")
        _reject_forbidden_fields(payload, label="transport binding")
        return cls(
            binding_id=payload.get("binding_id", ""),
            owner=payload.get("owner", ""),
            endpoint_identity=payload.get("endpoint_identity", ""),
            transport_profile=payload.get("transport_profile", TransportProfile.HTTP),
            route_kind=payload.get("route_kind", RouteKind.SERVICE_PROXY),
            method=payload.get("method", ""),
            effect_class=payload.get("effect_class", MethodEffectClass.READ),
            same_origin_path=payload.get("same_origin_path", ""),
            correlation_required=payload.get("correlation_required", True),
            framing=payload.get("framing", "jsonrpc-2.0"),
            authority=payload.get("authority", AuthoritySource.REVIEWED),
        )


@dataclass(frozen=True)
class LifecycleBinding(CanonicalContract):
    """Explicit lifecycle binding with timeout/cancellation transitions."""

    SCHEMA: ClassVar[str] = LIFECYCLE_BINDING_SCHEMA
    INTERFACE: ClassVar[str] = TRANSPORT_REPAIR_OPERATORS_INTERFACE

    binding_id: str
    owner: str
    phase: LifecyclePhase
    capability_state: CapabilityState
    health_ok: bool
    initialize_ok: bool
    tools_ok: bool = False
    interfaces_ok: bool = False
    timeout_ms: int = 8_000
    cancellation_supported: bool = True
    claims_available: bool = False
    authority: AuthoritySource = AuthoritySource.REVIEWED

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "binding_id", _text(self.binding_id, "binding_id", identifier=True)
        )
        owner = _text(self.owner, "owner", identifier=True)
        if owner not in REVIEWED_SERVICE_OWNERS:
            raise TransportRepairError(
                f"owner must be a reviewed service owner, got {owner!r}"
            )
        object.__setattr__(self, "owner", owner)
        object.__setattr__(self, "phase", _enum(self.phase, LifecyclePhase, "phase"))
        derived = classify_capability_state(
            health_ok=_bool(self.health_ok, "health_ok"),
            initialize_ok=_bool(self.initialize_ok, "initialize_ok"),
            tools_ok=_bool(self.tools_ok, "tools_ok"),
            interfaces_ok=_bool(self.interfaces_ok, "interfaces_ok"),
        )
        state = _enum(self.capability_state, CapabilityState, "capability_state")
        if state is not derived:
            # Repair operators restore truth: capability_state must match probes.
            raise TransportRepairError(
                "capability_state must match health/initialize/tools/interfaces probes "
                f"(expected {derived.value}, got {state.value})"
            )
        object.__setattr__(self, "capability_state", state)
        object.__setattr__(self, "health_ok", bool(self.health_ok))
        object.__setattr__(self, "initialize_ok", bool(self.initialize_ok))
        object.__setattr__(self, "tools_ok", bool(self.tools_ok))
        object.__setattr__(self, "interfaces_ok", bool(self.interfaces_ok))
        object.__setattr__(
            self, "timeout_ms", _nonnegative(self.timeout_ms, "timeout_ms")
        )
        object.__setattr__(
            self,
            "cancellation_supported",
            _bool(self.cancellation_supported, "cancellation_supported"),
        )
        claims = _bool(self.claims_available, "claims_available")
        may_claim = capability_claims_available(state)
        if claims and not may_claim:
            raise TransportRepairError(
                "claims_available cannot be true from health/initialize alone"
            )
        if claims != may_claim:
            # Normalize to truth: claim flag must equal derived availability.
            raise TransportRepairError(
                "claims_available must equal capability_claims_available(state)"
            )
        object.__setattr__(self, "claims_available", claims)
        object.__setattr__(
            self, "authority", _enum(self.authority, AuthoritySource, "authority")
        )
        if self.authority.is_abstaining_source:
            raise TransportRepairError(
                "LifecycleBinding cannot carry invented authority"
            )
        # Phase consistency with capability state.
        if state is CapabilityState.AVAILABLE and self.phase not in {
            LifecyclePhase.AVAILABLE,
            LifecyclePhase.CAPABILITY_READY,
        }:
            raise TransportRepairError(
                "available capability_state requires capability_ready/available phase"
            )
        if state is CapabilityState.INITIALIZED_NOT_AVAILABLE and self.phase is LifecyclePhase.AVAILABLE:
            raise TransportRepairError(
                "initialized_not_available cannot claim available lifecycle phase"
            )

    def _payload(self) -> dict[str, Any]:
        return {
            "interface": self.INTERFACE,
            "binding_id": self.binding_id,
            "owner": self.owner,
            "phase": self.phase.value,
            "capability_state": self.capability_state.value,
            "health_ok": self.health_ok,
            "initialize_ok": self.initialize_ok,
            "tools_ok": self.tools_ok,
            "interfaces_ok": self.interfaces_ok,
            "timeout_ms": self.timeout_ms,
            "cancellation_supported": self.cancellation_supported,
            "claims_available": self.claims_available,
            "authority": self.authority.value,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LifecycleBinding":
        if not isinstance(payload, Mapping):
            raise TransportRepairError("lifecycle binding must be an object")
        _reject_forbidden_fields(payload, label="lifecycle binding")
        return cls(
            binding_id=payload.get("binding_id", ""),
            owner=payload.get("owner", ""),
            phase=payload.get("phase", LifecyclePhase.UNBOUND),
            capability_state=payload.get(
                "capability_state", CapabilityState.UNREACHABLE
            ),
            health_ok=payload.get("health_ok", False),
            initialize_ok=payload.get("initialize_ok", False),
            tools_ok=payload.get("tools_ok", False),
            interfaces_ok=payload.get("interfaces_ok", False),
            timeout_ms=payload.get("timeout_ms", 8_000),
            cancellation_supported=payload.get("cancellation_supported", True),
            claims_available=payload.get("claims_available", False),
            authority=payload.get("authority", AuthoritySource.REVIEWED),
        )


@dataclass(frozen=True)
class BrowserMediationPolicy(CanonicalContract):
    """Closed browser mediation policy bound to GovernedMcpMediator@1."""

    SCHEMA: ClassVar[str] = BROWSER_MEDIATION_SCHEMA
    INTERFACE: ClassVar[str] = GOVERNED_MCP_MEDIATOR_INTERFACE

    policy_id: str
    governed_mutation_route: str = GOVERNED_MUTATION_ROUTE
    read_only_jsonrpc_methods: tuple[str, ...] = ()
    mutation_jsonrpc_methods: tuple[str, ...] = ()
    allow_raw_proxy_mutations: bool = False
    require_correlation_id: bool = True
    authority: AuthoritySource = AuthoritySource.REVIEWED

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "policy_id", _text(self.policy_id, "policy_id", identifier=True)
        )
        route = _text(self.governed_mutation_route, "governed_mutation_route")
        if route != GOVERNED_MUTATION_ROUTE:
            raise TransportRepairError(
                f"governed_mutation_route must be exactly {GOVERNED_MUTATION_ROUTE}"
            )
        object.__setattr__(self, "governed_mutation_route", route)
        if self.read_only_jsonrpc_methods in (None, ()):
            object.__setattr__(
                self,
                "read_only_jsonrpc_methods",
                tuple(sorted(SERVICE_PROXY_READ_ONLY_JSONRPC_METHODS)),
            )
        else:
            object.__setattr__(
                self,
                "read_only_jsonrpc_methods",
                _string_tuple(
                    self.read_only_jsonrpc_methods,
                    "read_only_jsonrpc_methods",
                    ordered=True,
                ),
            )
        if self.mutation_jsonrpc_methods in (None, ()):
            object.__setattr__(
                self,
                "mutation_jsonrpc_methods",
                tuple(sorted(SERVICE_PROXY_MUTATION_JSONRPC_METHODS)),
            )
        else:
            object.__setattr__(
                self,
                "mutation_jsonrpc_methods",
                _string_tuple(
                    self.mutation_jsonrpc_methods,
                    "mutation_jsonrpc_methods",
                    ordered=True,
                ),
            )
        if self.allow_raw_proxy_mutations is not False:
            raise TransportRepairError(
                "allow_raw_proxy_mutations must remain false"
            )
        object.__setattr__(self, "allow_raw_proxy_mutations", False)
        object.__setattr__(
            self,
            "require_correlation_id",
            _bool(self.require_correlation_id, "require_correlation_id"),
        )
        object.__setattr__(
            self, "authority", _enum(self.authority, AuthoritySource, "authority")
        )
        if self.authority.is_abstaining_source:
            raise TransportRepairError(
                "BrowserMediationPolicy cannot carry invented authority"
            )

    def classify(
        self,
        *,
        http_method: str,
        service_path: str,
        jsonrpc_method: str | None = None,
    ) -> dict[str, Any]:
        return classify_service_proxy_access(
            http_method=http_method,
            service_path=service_path,
            jsonrpc_method=jsonrpc_method,
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "interface": self.INTERFACE,
            "policy_id": self.policy_id,
            "governed_mutation_route": self.governed_mutation_route,
            "read_only_jsonrpc_methods": list(self.read_only_jsonrpc_methods),
            "mutation_jsonrpc_methods": list(self.mutation_jsonrpc_methods),
            "allow_raw_proxy_mutations": False,
            "require_correlation_id": self.require_correlation_id,
            "authority": self.authority.value,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BrowserMediationPolicy":
        if not isinstance(payload, Mapping):
            raise TransportRepairError("browser mediation policy must be an object")
        _reject_forbidden_fields(payload, label="browser mediation policy")
        return cls(
            policy_id=payload.get("policy_id", ""),
            governed_mutation_route=payload.get(
                "governed_mutation_route", GOVERNED_MUTATION_ROUTE
            ),
            read_only_jsonrpc_methods=tuple(
                payload.get("read_only_jsonrpc_methods") or ()
            ),
            mutation_jsonrpc_methods=tuple(
                payload.get("mutation_jsonrpc_methods") or ()
            ),
            allow_raw_proxy_mutations=payload.get("allow_raw_proxy_mutations", False),
            require_correlation_id=payload.get("require_correlation_id", True),
            authority=payload.get("authority", AuthoritySource.REVIEWED),
        )


@dataclass(frozen=True)
class CapabilityReport(CanonicalContract):
    """Truthful capability report — never upgrades health/init to available."""

    SCHEMA: ClassVar[str] = CAPABILITY_REPORT_SCHEMA
    INTERFACE: ClassVar[str] = TRANSPORT_REPAIR_OPERATORS_INTERFACE

    owner: str
    state: CapabilityState
    health_ok: bool
    initialize_ok: bool
    tools_ok: bool = False
    interfaces_ok: bool = False
    available: bool = False
    typed_unavailable_reason: str = ""

    def __post_init__(self) -> None:
        owner = _text(self.owner, "owner", identifier=True)
        if owner not in REVIEWED_SERVICE_OWNERS:
            raise TransportRepairError(
                f"owner must be a reviewed service owner, got {owner!r}"
            )
        object.__setattr__(self, "owner", owner)
        derived = classify_capability_state(
            health_ok=_bool(self.health_ok, "health_ok"),
            initialize_ok=_bool(self.initialize_ok, "initialize_ok"),
            tools_ok=_bool(self.tools_ok, "tools_ok"),
            interfaces_ok=_bool(self.interfaces_ok, "interfaces_ok"),
        )
        state = _enum(self.state, CapabilityState, "state")
        if state is not derived:
            raise TransportRepairError(
                f"capability report state must match probes (expected {derived.value})"
            )
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "health_ok", bool(self.health_ok))
        object.__setattr__(self, "initialize_ok", bool(self.initialize_ok))
        object.__setattr__(self, "tools_ok", bool(self.tools_ok))
        object.__setattr__(self, "interfaces_ok", bool(self.interfaces_ok))
        available = _bool(self.available, "available")
        may_claim = capability_claims_available(state)
        if available != may_claim:
            raise TransportRepairError(
                "available must not be claimed from health/initialize alone"
            )
        object.__setattr__(self, "available", available)
        if not available and state in {
            CapabilityState.HEALTH_ONLY,
            CapabilityState.INITIALIZED_NOT_AVAILABLE,
            CapabilityState.TYPED_UNAVAILABLE,
        }:
            reason = self.typed_unavailable_reason or (
                "health_and_initialize_do_not_establish_availability"
                if state is CapabilityState.INITIALIZED_NOT_AVAILABLE
                else state.value
            )
            object.__setattr__(
                self,
                "typed_unavailable_reason",
                _text(reason, "typed_unavailable_reason", required=False),
            )
        else:
            object.__setattr__(
                self,
                "typed_unavailable_reason",
                _text(
                    self.typed_unavailable_reason,
                    "typed_unavailable_reason",
                    required=False,
                ),
            )

    def _payload(self) -> dict[str, Any]:
        return {
            "interface": self.INTERFACE,
            "owner": self.owner,
            "state": self.state.value,
            "health_ok": self.health_ok,
            "initialize_ok": self.initialize_ok,
            "tools_ok": self.tools_ok,
            "interfaces_ok": self.interfaces_ok,
            "available": self.available,
            "typed_unavailable_reason": self.typed_unavailable_reason,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CapabilityReport":
        if not isinstance(payload, Mapping):
            raise TransportRepairError("capability report must be an object")
        _reject_forbidden_fields(payload, label="capability report")
        return cls(
            owner=payload.get("owner", ""),
            state=payload.get("state", CapabilityState.UNREACHABLE),
            health_ok=payload.get("health_ok", False),
            initialize_ok=payload.get("initialize_ok", False),
            tools_ok=payload.get("tools_ok", False),
            interfaces_ok=payload.get("interfaces_ok", False),
            available=payload.get("available", False),
            typed_unavailable_reason=payload.get("typed_unavailable_reason", ""),
        )


@dataclass(frozen=True)
class MiddlewareTranscriptRow(CanonicalContract):
    """One middleware decision row for preview evidence."""

    SCHEMA: ClassVar[str] = MIDDLEWARE_TRANSCRIPT_SCHEMA

    http_method: str
    service_path: str
    jsonrpc_method: str
    decision: ProxyDecision
    effect_class: MethodEffectClass
    allowed: bool
    reason: str
    mediation: MediationPathClass = MediationPathClass.RAW_SERVICE_PROXY

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "http_method", _text(self.http_method, "http_method").upper()
        )
        object.__setattr__(
            self, "service_path", _text(self.service_path, "service_path")
        )
        object.__setattr__(
            self,
            "jsonrpc_method",
            _text(self.jsonrpc_method, "jsonrpc_method", required=False),
        )
        object.__setattr__(
            self, "decision", _enum(self.decision, ProxyDecision, "decision")
        )
        object.__setattr__(
            self,
            "effect_class",
            _enum(self.effect_class, MethodEffectClass, "effect_class"),
        )
        object.__setattr__(self, "allowed", _bool(self.allowed, "allowed"))
        object.__setattr__(self, "reason", _text(self.reason, "reason"))
        object.__setattr__(
            self, "mediation", _enum(self.mediation, MediationPathClass, "mediation")
        )
        if self.mediation is MediationPathClass.DIRECT_PROXY:
            raise TransportRepairError("direct_proxy rows are inadmissible")
        if (
            self.effect_class is MethodEffectClass.MUTATE
            and self.allowed
            and self.mediation is not MediationPathClass.GOVERNED_MEDIATOR
        ):
            raise TransportRepairError(
                "mutation rows cannot be allowed on raw service proxy"
            )

    def _payload(self) -> dict[str, Any]:
        return {
            "http_method": self.http_method,
            "service_path": self.service_path,
            "jsonrpc_method": self.jsonrpc_method,
            "decision": self.decision.value,
            "effect_class": self.effect_class.value,
            "allowed": self.allowed,
            "reason": self.reason,
            "mediation": self.mediation.value,
            "governed_route": GOVERNED_MUTATION_ROUTE,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MiddlewareTranscriptRow":
        if not isinstance(payload, Mapping):
            raise TransportRepairError("middleware transcript row must be an object")
        return cls(
            http_method=payload.get("http_method", ""),
            service_path=payload.get("service_path", ""),
            jsonrpc_method=payload.get("jsonrpc_method", ""),
            decision=payload.get("decision", ProxyDecision.REJECT_UNKNOWN),
            effect_class=payload.get("effect_class", MethodEffectClass.UNKNOWN),
            allowed=payload.get("allowed", False),
            reason=payload.get("reason", ""),
            mediation=payload.get(
                "mediation", MediationPathClass.RAW_SERVICE_PROXY
            ),
        )


def build_middleware_transcript(
    cases: Sequence[Mapping[str, Any]],
) -> tuple[MiddlewareTranscriptRow, ...]:
    """Build a closed middleware transcript from request cases."""

    if len(cases) > MAX_TRANSCRIPT_ROWS:
        raise TransportRepairError("middleware transcript exceeds closed bound")
    rows: list[MiddlewareTranscriptRow] = []
    for index, case in enumerate(cases):
        if not isinstance(case, Mapping):
            raise TransportRepairError(f"cases[{index}] must be an object")
        classification = classify_service_proxy_access(
            http_method=str(case.get("http_method") or ""),
            service_path=str(case.get("service_path") or ""),
            jsonrpc_method=case.get("jsonrpc_method"),
        )
        rows.append(
            MiddlewareTranscriptRow(
                http_method=str(case.get("http_method") or ""),
                service_path=str(case.get("service_path") or ""),
                jsonrpc_method=str(case.get("jsonrpc_method") or ""),
                decision=classification["decision"],
                effect_class=classification["effect_class"],
                allowed=bool(classification["allowed"]),
                reason=str(classification["reason"]),
                mediation=classification["mediation"],
            )
        )
    return tuple(rows)


# ---------------------------------------------------------------------------
# Repair request / receipt / preview
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TransportRepairRequest(CanonicalContract):
    """Closed input for one transport/lifecycle/mediation repair."""

    SCHEMA: ClassVar[str] = TRANSPORT_REQUEST_SCHEMA

    role: OperatorRole
    reviewed_transport: TransportEndpointBinding | None = None
    reviewed_lifecycle: LifecycleBinding | None = None
    reviewed_mediation: BrowserMediationPolicy | None = None
    current_transport: TransportEndpointBinding | None = None
    current_lifecycle: LifecycleBinding | None = None
    current_mediation: BrowserMediationPolicy | None = None
    middleware_cases: tuple[Mapping[str, Any], ...] = ()
    authority: AuthoritySource = AuthoritySource.REVIEWED

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", _enum(self.role, OperatorRole, "role"))
        object.__setattr__(
            self, "authority", _enum(self.authority, AuthoritySource, "authority")
        )

        def _opt_transport(value: Any, name: str) -> TransportEndpointBinding | None:
            if value is None:
                return None
            if isinstance(value, TransportEndpointBinding):
                return value
            if isinstance(value, Mapping):
                return TransportEndpointBinding.from_dict(value)
            raise TransportRepairError(f"{name} must be a TransportEndpointBinding")

        def _opt_lifecycle(value: Any, name: str) -> LifecycleBinding | None:
            if value is None:
                return None
            if isinstance(value, LifecycleBinding):
                return value
            if isinstance(value, Mapping):
                return LifecycleBinding.from_dict(value)
            raise TransportRepairError(f"{name} must be a LifecycleBinding")

        def _opt_mediation(value: Any, name: str) -> BrowserMediationPolicy | None:
            if value is None:
                return None
            if isinstance(value, BrowserMediationPolicy):
                return value
            if isinstance(value, Mapping):
                return BrowserMediationPolicy.from_dict(value)
            raise TransportRepairError(f"{name} must be a BrowserMediationPolicy")

        object.__setattr__(
            self,
            "reviewed_transport",
            _opt_transport(self.reviewed_transport, "reviewed_transport"),
        )
        object.__setattr__(
            self,
            "reviewed_lifecycle",
            _opt_lifecycle(self.reviewed_lifecycle, "reviewed_lifecycle"),
        )
        object.__setattr__(
            self,
            "reviewed_mediation",
            _opt_mediation(self.reviewed_mediation, "reviewed_mediation"),
        )
        object.__setattr__(
            self,
            "current_transport",
            _opt_transport(self.current_transport, "current_transport"),
        )
        object.__setattr__(
            self,
            "current_lifecycle",
            _opt_lifecycle(self.current_lifecycle, "current_lifecycle"),
        )
        object.__setattr__(
            self,
            "current_mediation",
            _opt_mediation(self.current_mediation, "current_mediation"),
        )
        if self.middleware_cases is None:
            object.__setattr__(self, "middleware_cases", ())
        elif not isinstance(self.middleware_cases, Sequence) or isinstance(
            self.middleware_cases, (str, bytes, bytearray)
        ):
            raise TransportRepairError("middleware_cases must be a sequence")
        else:
            cases = tuple(
                MappingProxyType(dict(item))
                if isinstance(item, Mapping)
                else (_ for _ in ()).throw(
                    TransportRepairError("middleware_cases items must be objects")
                )
                for item in self.middleware_cases
            )
            if len(cases) > MAX_TRANSCRIPT_ROWS:
                raise TransportRepairError("middleware_cases exceeds closed bound")
            object.__setattr__(self, "middleware_cases", cases)

    def _payload(self) -> dict[str, Any]:
        return {
            "role": self.role.value,
            "reviewed_transport": (
                None
                if self.reviewed_transport is None
                else self.reviewed_transport.to_dict()
            ),
            "reviewed_lifecycle": (
                None
                if self.reviewed_lifecycle is None
                else self.reviewed_lifecycle.to_dict()
            ),
            "reviewed_mediation": (
                None
                if self.reviewed_mediation is None
                else self.reviewed_mediation.to_dict()
            ),
            "current_transport": (
                None
                if self.current_transport is None
                else self.current_transport.to_dict()
            ),
            "current_lifecycle": (
                None
                if self.current_lifecycle is None
                else self.current_lifecycle.to_dict()
            ),
            "current_mediation": (
                None
                if self.current_mediation is None
                else self.current_mediation.to_dict()
            ),
            "middleware_cases": [dict(item) for item in self.middleware_cases],
            "authority": self.authority.value,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TransportRepairRequest":
        if not isinstance(payload, Mapping):
            raise TransportRepairError("transport repair request must be an object")
        _reject_forbidden_fields(payload, label="transport repair request")
        return cls(
            role=payload.get("role", OperatorRole.TRANSPORT_BINDING),
            reviewed_transport=payload.get("reviewed_transport"),
            reviewed_lifecycle=payload.get("reviewed_lifecycle"),
            reviewed_mediation=payload.get("reviewed_mediation"),
            current_transport=payload.get("current_transport"),
            current_lifecycle=payload.get("current_lifecycle"),
            current_mediation=payload.get("current_mediation"),
            middleware_cases=tuple(payload.get("middleware_cases") or ()),
            authority=payload.get("authority", AuthoritySource.REVIEWED),
        )


@dataclass(frozen=True)
class TransportRepairReceipt(CanonicalContract):
    """Non-authoritative preview/inverse receipt for one transport repair."""

    SCHEMA: ClassVar[str] = TRANSPORT_RECEIPT_SCHEMA
    INTERFACE: ClassVar[str] = TRANSPORT_REPAIR_OPERATORS_INTERFACE

    disposition: RepairDisposition
    role: OperatorRole
    operator_kind: str
    reason_codes: tuple[str, ...]
    preview_transport: TransportEndpointBinding | None = None
    preview_lifecycle: LifecycleBinding | None = None
    preview_mediation: BrowserMediationPolicy | None = None
    inverse_transport: TransportEndpointBinding | None = None
    inverse_lifecycle: LifecycleBinding | None = None
    inverse_mediation: BrowserMediationPolicy | None = None
    capability_report: CapabilityReport | None = None
    middleware_transcript: tuple[MiddlewareTranscriptRow, ...] = ()
    rollback_identity: str = ""
    proposal_only: bool = True
    grants_write_authority: bool = False
    grants_proof_authority: bool = False
    semantic_authority: bool = False
    evidence_id: str = TRANSPORT_REPAIR_EVIDENCE

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "disposition",
            _enum(self.disposition, RepairDisposition, "disposition"),
        )
        object.__setattr__(self, "role", _enum(self.role, OperatorRole, "role"))
        object.__setattr__(
            self, "operator_kind", _text(self.operator_kind, "operator_kind")
        )
        object.__setattr__(
            self,
            "reason_codes",
            _string_tuple(
                self.reason_codes, "reason_codes", required=True, ordered=True
            ),
        )
        if len(self.reason_codes) > MAX_REASON_CODES:
            raise TransportRepairError("reason_codes exceeds its item bound")
        if self.proposal_only is not True:
            raise TransportRepairError("receipts must remain proposal-only")
        if self.grants_write_authority is not False:
            raise TransportRepairError("receipts cannot grant write authority")
        if self.grants_proof_authority is not False:
            raise TransportRepairError("receipts cannot grant proof authority")
        if self.semantic_authority is not False:
            raise TransportRepairError("receipts cannot claim semantic authority")
        object.__setattr__(self, "proposal_only", True)
        object.__setattr__(self, "grants_write_authority", False)
        object.__setattr__(self, "grants_proof_authority", False)
        object.__setattr__(self, "semantic_authority", False)
        object.__setattr__(
            self, "evidence_id", _text(self.evidence_id, "evidence_id")
        )
        object.__setattr__(
            self,
            "rollback_identity",
            _text(self.rollback_identity, "rollback_identity", required=False),
        )
        if self.middleware_transcript is None:
            object.__setattr__(self, "middleware_transcript", ())
        elif not isinstance(self.middleware_transcript, Sequence) or isinstance(
            self.middleware_transcript, (str, bytes, bytearray)
        ):
            raise TransportRepairError("middleware_transcript must be a sequence")
        else:
            rows = tuple(
                item
                if isinstance(item, MiddlewareTranscriptRow)
                else MiddlewareTranscriptRow.from_dict(item)
                for item in self.middleware_transcript
            )
            object.__setattr__(self, "middleware_transcript", rows)

    def _payload(self) -> dict[str, Any]:
        return {
            "interface": self.INTERFACE,
            "disposition": self.disposition.value,
            "role": self.role.value,
            "operator_kind": self.operator_kind,
            "reason_codes": list(self.reason_codes),
            "preview_transport": (
                None
                if self.preview_transport is None
                else self.preview_transport.to_dict()
            ),
            "preview_lifecycle": (
                None
                if self.preview_lifecycle is None
                else self.preview_lifecycle.to_dict()
            ),
            "preview_mediation": (
                None
                if self.preview_mediation is None
                else self.preview_mediation.to_dict()
            ),
            "inverse_transport": (
                None
                if self.inverse_transport is None
                else self.inverse_transport.to_dict()
            ),
            "inverse_lifecycle": (
                None
                if self.inverse_lifecycle is None
                else self.inverse_lifecycle.to_dict()
            ),
            "inverse_mediation": (
                None
                if self.inverse_mediation is None
                else self.inverse_mediation.to_dict()
            ),
            "capability_report": (
                None
                if self.capability_report is None
                else self.capability_report.to_dict()
            ),
            "middleware_transcript": [
                item.to_dict() for item in self.middleware_transcript
            ],
            "rollback_identity": self.rollback_identity,
            "proposal_only": True,
            "grants_write_authority": False,
            "grants_proof_authority": False,
            "semantic_authority": False,
            "evidence_id": self.evidence_id,
            "version": TRANSPORT_REPAIR_VERSION,
            "evidence_subset": [
                "endpoint_identity",
                "route_kind",
                "method_effect_class",
                "middleware_transcript",
                "rollback",
            ],
        }


# ---------------------------------------------------------------------------
# Operator implementations
# ---------------------------------------------------------------------------


def _transport_descriptor():
    registry = build_default_operator_registry()
    return registry.require_known(OperatorKind.REPAIR_TRANSPORT_ADAPTER)


def _capability_descriptor():
    registry = build_default_operator_registry()
    return registry.require_known(OperatorKind.REPAIR_CAPABILITY_TRUTH)


def _base_receipt(
    request: TransportRepairRequest,
    *,
    disposition: RepairDisposition,
    role: OperatorRole,
    operator_kind: str,
    reasons: Sequence[str],
    preview_transport: TransportEndpointBinding | None = None,
    preview_lifecycle: LifecycleBinding | None = None,
    preview_mediation: BrowserMediationPolicy | None = None,
    inverse_transport: TransportEndpointBinding | None = None,
    inverse_lifecycle: LifecycleBinding | None = None,
    inverse_mediation: BrowserMediationPolicy | None = None,
    capability_report: CapabilityReport | None = None,
    middleware_transcript: Sequence[MiddlewareTranscriptRow] = (),
    rollback_identity: str = "",
) -> TransportRepairReceipt:
    return TransportRepairReceipt(
        disposition=disposition,
        role=role,
        operator_kind=operator_kind,
        reason_codes=tuple(reasons) or (disposition.value,),
        preview_transport=preview_transport,
        preview_lifecycle=preview_lifecycle,
        preview_mediation=preview_mediation,
        inverse_transport=inverse_transport,
        inverse_lifecycle=inverse_lifecycle,
        inverse_mediation=inverse_mediation,
        capability_report=capability_report,
        middleware_transcript=tuple(middleware_transcript),
        rollback_identity=rollback_identity,
    )


def _guard_authority(
    role: OperatorRole,
    request: TransportRepairRequest,
    operator_kind: str,
) -> TransportRepairReceipt | None:
    if not request.authority.authorizes_transport_source:
        return _base_receipt(
            request,
            disposition=RepairDisposition.ABSTAIN,
            role=role,
            operator_kind=operator_kind,
            reasons=(
                "transport_source_not_reviewed",
                f"authority:{request.authority.value}",
                "conflict_policy_abstain",
            ),
        )
    return None


def build_capability_report(
    *,
    owner: str,
    health_ok: bool,
    initialize_ok: bool,
    tools_ok: bool = False,
    interfaces_ok: bool = False,
) -> CapabilityReport:
    """Build a truthful capability report from probe outcomes."""

    state = classify_capability_state(
        health_ok=health_ok,
        initialize_ok=initialize_ok,
        tools_ok=tools_ok,
        interfaces_ok=interfaces_ok,
    )
    available = capability_claims_available(state)
    reason = ""
    if not available:
        if state is CapabilityState.INITIALIZED_NOT_AVAILABLE:
            reason = "health_and_initialize_do_not_establish_availability"
        elif state is CapabilityState.HEALTH_ONLY:
            reason = "initialize_required_before_capability_claim"
        elif state is CapabilityState.UNREACHABLE:
            reason = "health_probe_failed"
        else:
            reason = state.value
    return CapabilityReport(
        owner=owner,
        state=state,
        health_ok=health_ok,
        initialize_ok=initialize_ok,
        tools_ok=tools_ok,
        interfaces_ok=interfaces_ok,
        available=available,
        typed_unavailable_reason=reason,
    )


def build_lifecycle_binding(
    *,
    binding_id: str,
    owner: str,
    health_ok: bool,
    initialize_ok: bool,
    tools_ok: bool = False,
    interfaces_ok: bool = False,
    timeout_ms: int = 8_000,
    cancellation_supported: bool = True,
    authority: AuthoritySource = AuthoritySource.REVIEWED,
) -> LifecycleBinding:
    """Derive a truthful lifecycle binding from probes."""

    state = classify_capability_state(
        health_ok=health_ok,
        initialize_ok=initialize_ok,
        tools_ok=tools_ok,
        interfaces_ok=interfaces_ok,
    )
    if state is CapabilityState.UNREACHABLE:
        phase = LifecyclePhase.UNBOUND
    elif state is CapabilityState.HEALTH_ONLY:
        phase = LifecyclePhase.HEALTH_PROBED
    elif state is CapabilityState.INITIALIZED_NOT_AVAILABLE:
        phase = LifecyclePhase.INITIALIZED
    elif state is CapabilityState.TOOLS_DISCOVERED:
        phase = LifecyclePhase.CAPABILITY_READY
    else:
        phase = LifecyclePhase.AVAILABLE
    return LifecycleBinding(
        binding_id=binding_id,
        owner=owner,
        phase=phase,
        capability_state=state,
        health_ok=health_ok,
        initialize_ok=initialize_ok,
        tools_ok=tools_ok,
        interfaces_ok=interfaces_ok,
        timeout_ms=timeout_ms,
        cancellation_supported=cancellation_supported,
        claims_available=capability_claims_available(state),
        authority=authority,
    )


class TransportBindingOperator:
    """Repair endpoint/transport adapter bindings (REPAIR_TRANSPORT_ADAPTER)."""

    ROLE: ClassVar[OperatorRole] = OperatorRole.TRANSPORT_BINDING
    INTERFACE: ClassVar[str] = TRANSPORT_REPAIR_OPERATORS_INTERFACE

    def __init__(self) -> None:
        self.descriptor = _transport_descriptor()
        if self.descriptor.family is not OperatorFamily.TRANSPORT:
            raise TransportRepairError("registry transport family mismatch")

    @property
    def operator_id(self) -> str:
        return f"dcr-operator:{self.ROLE.value}@1"

    def apply(self, request: TransportRepairRequest) -> TransportRepairReceipt:
        if not isinstance(request, TransportRepairRequest):
            raise TransportRepairError("request must be a TransportRepairRequest")
        kind = OperatorKind.REPAIR_TRANSPORT_ADAPTER.value
        blocked = _guard_authority(self.ROLE, request, kind)
        if blocked is not None:
            return blocked
        reviewed = request.reviewed_transport
        if reviewed is None:
            return _base_receipt(
                request,
                disposition=RepairDisposition.ABSTAIN,
                role=self.ROLE,
                operator_kind=kind,
                reasons=("missing_reviewed_transport", "conflict_policy_abstain"),
            )
        current = request.current_transport
        if current is not None and current.content_id == reviewed.content_id:
            return _base_receipt(
                request,
                disposition=RepairDisposition.ALREADY_ALIGNED,
                role=self.ROLE,
                operator_kind=kind,
                reasons=("already_aligned", "idempotent"),
                preview_transport=reviewed,
                inverse_transport=current,
                rollback_identity=current.content_id,
            )
        # Evidence subset fields must be present on the reviewed binding.
        reasons = [
            "endpoint_identity",
            "route_kind",
            "method_effect_class",
            "preview_ready",
        ]
        return _base_receipt(
            request,
            disposition=RepairDisposition.PREVIEW_READY,
            role=self.ROLE,
            operator_kind=kind,
            reasons=reasons,
            preview_transport=reviewed,
            inverse_transport=current,
            rollback_identity=(
                current.content_id if current is not None else reviewed.content_id
            ),
        )

    def preview(
        self, request: TransportRepairRequest
    ) -> TransportRepairReceipt:
        return self.apply(request)

    def inverse(
        self, receipt: TransportRepairReceipt
    ) -> TransportEndpointBinding | None:
        if not isinstance(receipt, TransportRepairReceipt):
            raise TransportRepairError("receipt must be a TransportRepairReceipt")
        return receipt.inverse_transport


class LifecycleBindingOperator:
    """Repair lifecycle/capability-truth bindings (REPAIR_CAPABILITY_TRUTH)."""

    ROLE: ClassVar[OperatorRole] = OperatorRole.LIFECYCLE_BINDING
    INTERFACE: ClassVar[str] = TRANSPORT_REPAIR_OPERATORS_INTERFACE

    def __init__(self) -> None:
        self.descriptor = _capability_descriptor()
        if self.descriptor.family is not OperatorFamily.TRANSPORT:
            raise TransportRepairError("registry transport family mismatch")

    @property
    def operator_id(self) -> str:
        return f"dcr-operator:{self.ROLE.value}@1"

    def apply(self, request: TransportRepairRequest) -> TransportRepairReceipt:
        if not isinstance(request, TransportRepairRequest):
            raise TransportRepairError("request must be a TransportRepairRequest")
        kind = OperatorKind.REPAIR_CAPABILITY_TRUTH.value
        blocked = _guard_authority(self.ROLE, request, kind)
        if blocked is not None:
            return blocked
        reviewed = request.reviewed_lifecycle
        if reviewed is None:
            return _base_receipt(
                request,
                disposition=RepairDisposition.ABSTAIN,
                role=self.ROLE,
                operator_kind=kind,
                reasons=("missing_reviewed_lifecycle", "conflict_policy_abstain"),
            )
        # Reject false availability claims on the reviewed artifact.
        if reviewed.claims_available and not capability_claims_available(
            reviewed.capability_state
        ):
            return _base_receipt(
                request,
                disposition=RepairDisposition.REJECTED,
                role=self.ROLE,
                operator_kind=kind,
                reasons=(
                    "availability_claim_from_health_initialize_alone",
                    "typed_unavailable_required",
                ),
            )
        report = build_capability_report(
            owner=reviewed.owner,
            health_ok=reviewed.health_ok,
            initialize_ok=reviewed.initialize_ok,
            tools_ok=reviewed.tools_ok,
            interfaces_ok=reviewed.interfaces_ok,
        )
        current = request.current_lifecycle
        if current is not None and current.content_id == reviewed.content_id:
            return _base_receipt(
                request,
                disposition=RepairDisposition.ALREADY_ALIGNED,
                role=self.ROLE,
                operator_kind=kind,
                reasons=("already_aligned", "idempotent", "capability_truth"),
                preview_lifecycle=reviewed,
                inverse_lifecycle=current,
                capability_report=report,
                rollback_identity=current.content_id,
            )
        return _base_receipt(
            request,
            disposition=RepairDisposition.PREVIEW_READY,
            role=self.ROLE,
            operator_kind=kind,
            reasons=(
                "lifecycle_binding",
                "capability_truth",
                "no_health_initialize_availability_claim",
                "preview_ready",
            ),
            preview_lifecycle=reviewed,
            inverse_lifecycle=current,
            capability_report=report,
            rollback_identity=(
                current.content_id if current is not None else reviewed.content_id
            ),
        )

    def preview(
        self, request: TransportRepairRequest
    ) -> TransportRepairReceipt:
        return self.apply(request)

    def inverse(self, receipt: TransportRepairReceipt) -> LifecycleBinding | None:
        if not isinstance(receipt, TransportRepairReceipt):
            raise TransportRepairError("receipt must be a TransportRepairReceipt")
        return receipt.inverse_lifecycle


class BrowserMediationOperator:
    """Repair desktop same-origin mediation (GovernedMcpMediator@1)."""

    ROLE: ClassVar[OperatorRole] = OperatorRole.BROWSER_MEDIATION
    INTERFACE: ClassVar[str] = GOVERNED_MCP_MEDIATOR_INTERFACE

    def __init__(self) -> None:
        # Browser mediation is transport-family; prefer capability truth scope
        # for write_scope documentation while still registering under transport.
        self.descriptor = _transport_descriptor()

    @property
    def operator_id(self) -> str:
        return f"dcr-operator:{self.ROLE.value}@1"

    def apply(self, request: TransportRepairRequest) -> TransportRepairReceipt:
        if not isinstance(request, TransportRepairRequest):
            raise TransportRepairError("request must be a TransportRepairRequest")
        kind = OperatorKind.REPAIR_TRANSPORT_ADAPTER.value
        blocked = _guard_authority(self.ROLE, request, kind)
        if blocked is not None:
            return blocked
        reviewed = request.reviewed_mediation
        if reviewed is None:
            return _base_receipt(
                request,
                disposition=RepairDisposition.ABSTAIN,
                role=self.ROLE,
                operator_kind=kind,
                reasons=("missing_reviewed_mediation", "conflict_policy_abstain"),
            )
        if reviewed.allow_raw_proxy_mutations:
            return _base_receipt(
                request,
                disposition=RepairDisposition.REJECTED,
                role=self.ROLE,
                operator_kind=kind,
                reasons=("raw_proxy_mutations_forbidden",),
            )
        cases = request.middleware_cases or (
            MappingProxyType(
                {
                    "http_method": "GET",
                    "service_path": "/mcp/health",
                    "jsonrpc_method": "",
                }
            ),
            MappingProxyType(
                {
                    "http_method": "POST",
                    "service_path": "/mcp",
                    "jsonrpc_method": "initialize",
                }
            ),
            MappingProxyType(
                {
                    "http_method": "POST",
                    "service_path": "/mcp",
                    "jsonrpc_method": "tools/call",
                }
            ),
            MappingProxyType(
                {
                    "http_method": "POST",
                    "service_path": "/mcp",
                    "jsonrpc_method": "mcp++/execute",
                }
            ),
        )
        transcript = build_middleware_transcript(cases)
        # Prove no mutation bypass in the transcript.
        for row in transcript:
            if row.effect_class is MethodEffectClass.MUTATE and row.allowed:
                return _base_receipt(
                    request,
                    disposition=RepairDisposition.REJECTED,
                    role=self.ROLE,
                    operator_kind=kind,
                    reasons=("browser_mutation_bypass", row.reason),
                    preview_mediation=reviewed,
                    middleware_transcript=transcript,
                )
            if (
                row.effect_class is MethodEffectClass.MUTATE
                and row.decision
                not in {
                    ProxyDecision.REQUIRE_GOVERNED_MEDIATOR,
                    ProxyDecision.REJECT_MUTATION,
                }
            ):
                return _base_receipt(
                    request,
                    disposition=RepairDisposition.REJECTED,
                    role=self.ROLE,
                    operator_kind=kind,
                    reasons=("mutation_not_redirected_to_governed_mediator",),
                    preview_mediation=reviewed,
                    middleware_transcript=transcript,
                )
        current = request.current_mediation
        if current is not None and current.content_id == reviewed.content_id:
            disposition = RepairDisposition.ALREADY_ALIGNED
            reasons = (
                "already_aligned",
                "idempotent",
                "no_browser_mutation_bypass",
                "middleware_transcript",
            )
        else:
            disposition = RepairDisposition.PREVIEW_READY
            reasons = (
                "browser_mediation",
                "governed_mediator",
                "no_browser_mutation_bypass",
                "middleware_transcript",
                "preview_ready",
            )
        return _base_receipt(
            request,
            disposition=disposition,
            role=self.ROLE,
            operator_kind=kind,
            reasons=reasons,
            preview_mediation=reviewed,
            inverse_mediation=current,
            middleware_transcript=transcript,
            rollback_identity=(
                current.content_id if current is not None else reviewed.content_id
            ),
        )

    def preview(
        self, request: TransportRepairRequest
    ) -> TransportRepairReceipt:
        return self.apply(request)

    def inverse(
        self, receipt: TransportRepairReceipt
    ) -> BrowserMediationPolicy | None:
        if not isinstance(receipt, TransportRepairReceipt):
            raise TransportRepairError("receipt must be a TransportRepairReceipt")
        return receipt.inverse_mediation


@dataclass(frozen=True)
class TransportRepairOperators:
    """Closed bundle of DCR-044 transport/lifecycle/mediation operators."""

    INTERFACE: ClassVar[str] = TRANSPORT_REPAIR_OPERATORS_INTERFACE
    EVIDENCE_ID: ClassVar[str] = TRANSPORT_REPAIR_EVIDENCE
    MEDIATOR_INTERFACE: ClassVar[str] = GOVERNED_MCP_MEDIATOR_INTERFACE

    transport_binding: TransportBindingOperator
    lifecycle_binding: LifecycleBindingOperator
    browser_mediation: BrowserMediationOperator

    def apply(self, request: TransportRepairRequest) -> TransportRepairReceipt:
        if request.role is OperatorRole.TRANSPORT_BINDING:
            return self.transport_binding.apply(request)
        if request.role is OperatorRole.LIFECYCLE_BINDING:
            return self.lifecycle_binding.apply(request)
        if request.role is OperatorRole.BROWSER_MEDIATION:
            return self.browser_mediation.apply(request)
        raise TransportRepairError(f"unsupported role: {request.role!r}")


def build_transport_repair_operators() -> TransportRepairOperators:
    """Construct the sealed DCR-044 operator bundle."""

    return TransportRepairOperators(
        transport_binding=TransportBindingOperator(),
        lifecycle_binding=LifecycleBindingOperator(),
        browser_mediation=BrowserMediationOperator(),
    )


def default_browser_mediation_policy(
    *,
    policy_id: str = "policy:desktop-same-origin-mediator",
) -> BrowserMediationPolicy:
    return BrowserMediationPolicy(policy_id=policy_id)


def materialize_transport_operator_vectors() -> dict[str, Any]:
    """Content-addressed operator vectors for DCR-044 evidence."""

    ops = build_transport_repair_operators()
    transport = TransportEndpointBinding(
        binding_id="binding:transport:ipfs_accelerate_py:http",
        owner="ipfs_accelerate_py",
        endpoint_identity="endpoint:ipfs_accelerate_py:same-origin",
        transport_profile=TransportProfile.HTTP,
        route_kind=RouteKind.SERVICE_PROXY,
        method="initialize",
        effect_class=MethodEffectClass.READ,
        same_origin_path="/mcp/services/ipfs_accelerate_py/mcp",
    )
    mutation = TransportEndpointBinding(
        binding_id="binding:transport:governed-tools-call",
        owner="ipfs_accelerate_py",
        endpoint_identity="endpoint:governed-mediator",
        transport_profile=TransportProfile.HTTP,
        route_kind=RouteKind.GOVERNED_MEDIATOR,
        method="tools/call",
        effect_class=MethodEffectClass.MUTATE,
        same_origin_path=GOVERNED_MUTATION_ROUTE,
    )
    lifecycle_unavailable = build_lifecycle_binding(
        binding_id="binding:lifecycle:health-init-only",
        owner="ipfs_accelerate_py",
        health_ok=True,
        initialize_ok=True,
        tools_ok=False,
        interfaces_ok=False,
    )
    lifecycle_available = build_lifecycle_binding(
        binding_id="binding:lifecycle:tools-ready",
        owner="ipfs_accelerate_py",
        health_ok=True,
        initialize_ok=True,
        tools_ok=True,
        interfaces_ok=True,
    )
    mediation = default_browser_mediation_policy()
    cases = (
        {
            "http_method": "GET",
            "service_path": "/mcp/health",
            "jsonrpc_method": "",
        },
        {
            "http_method": "POST",
            "service_path": "/mcp",
            "jsonrpc_method": "initialize",
        },
        {
            "http_method": "POST",
            "service_path": "/mcp",
            "jsonrpc_method": "tools/list",
        },
        {
            "http_method": "POST",
            "service_path": "/mcp",
            "jsonrpc_method": "tools/call",
        },
        {
            "http_method": "POST",
            "service_path": "/mcp",
            "jsonrpc_method": "mcp++/goals/create",
        },
    )
    transport_receipt = ops.transport_binding.apply(
        TransportRepairRequest(
            role=OperatorRole.TRANSPORT_BINDING,
            reviewed_transport=transport,
            current_transport=None,
        )
    )
    lifecycle_receipt = ops.lifecycle_binding.apply(
        TransportRepairRequest(
            role=OperatorRole.LIFECYCLE_BINDING,
            reviewed_lifecycle=lifecycle_unavailable,
        )
    )
    mediation_receipt = ops.browser_mediation.apply(
        TransportRepairRequest(
            role=OperatorRole.BROWSER_MEDIATION,
            reviewed_mediation=mediation,
            middleware_cases=cases,
        )
    )
    payload = {
        "schema": TRANSPORT_OPERATOR_VECTORS_SCHEMA,
        "interface": TRANSPORT_REPAIR_OPERATORS_INTERFACE,
        "mediator_interface": GOVERNED_MCP_MEDIATOR_INTERFACE,
        "evidence_id": TRANSPORT_REPAIR_EVIDENCE,
        "version": TRANSPORT_REPAIR_VERSION,
        "operators": {
            "transport_binding": ops.transport_binding.operator_id,
            "lifecycle_binding": ops.lifecycle_binding.operator_id,
            "browser_mediation": ops.browser_mediation.operator_id,
        },
        "bindings": {
            "transport": transport.to_dict(),
            "mutation_transport": mutation.to_dict(),
            "lifecycle_unavailable": lifecycle_unavailable.to_dict(),
            "lifecycle_available": lifecycle_available.to_dict(),
            "mediation": mediation.to_dict(),
        },
        "receipts": {
            "transport": transport_receipt.to_dict(),
            "lifecycle": lifecycle_receipt.to_dict(),
            "mediation": mediation_receipt.to_dict(),
        },
        "capability_truth": {
            "health_initialize_only_available": False,
            "health_initialize_only_state": lifecycle_unavailable.capability_state.value,
            "tools_ready_available": lifecycle_available.claims_available,
        },
        "governed_mutation_route": GOVERNED_MUTATION_ROUTE,
        "read_only_jsonrpc_methods": sorted(SERVICE_PROXY_READ_ONLY_JSONRPC_METHODS),
        "mutation_jsonrpc_methods": sorted(SERVICE_PROXY_MUTATION_JSONRPC_METHODS),
    }
    payload["vector_digest"] = _digest(payload)
    return payload


__all__ = (
    "TRANSPORT_REPAIR_OPERATORS_INTERFACE",
    "GOVERNED_MCP_MEDIATOR_INTERFACE",
    "TRANSPORT_REPAIR_EVIDENCE",
    "GOVERNED_MUTATION_ROUTE",
    "SERVICE_PROXY_READ_ONLY_JSONRPC_METHODS",
    "SERVICE_PROXY_MUTATION_JSONRPC_METHODS",
    "TransportRepairError",
    "TransportRepairAbstention",
    "RepairDisposition",
    "OperatorRole",
    "TransportProfile",
    "RouteKind",
    "MethodEffectClass",
    "CapabilityState",
    "LifecyclePhase",
    "MediationPathClass",
    "ProxyDecision",
    "AuthoritySource",
    "TransportEndpointBinding",
    "LifecycleBinding",
    "BrowserMediationPolicy",
    "CapabilityReport",
    "MiddlewareTranscriptRow",
    "TransportRepairRequest",
    "TransportRepairReceipt",
    "TransportBindingOperator",
    "LifecycleBindingOperator",
    "BrowserMediationOperator",
    "TransportRepairOperators",
    "build_transport_repair_operators",
    "build_capability_report",
    "build_lifecycle_binding",
    "build_middleware_transcript",
    "classify_capability_state",
    "capability_claims_available",
    "classify_jsonrpc_effect",
    "classify_service_proxy_access",
    "assert_no_browser_mutation_bypass",
    "default_browser_mediation_policy",
    "materialize_transport_operator_vectors",
)
