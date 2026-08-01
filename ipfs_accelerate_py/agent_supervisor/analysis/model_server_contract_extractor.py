"""Cold extraction of model-server route and inference contracts (SCA-171).

Interface: ``ModelServerContractExtractor@1`` / ``ModelServerContractCatalog@1``

Consumes structured route and inference premises (optionally bound to a
:class:`RuntimeComponentCatalog` model-server root) and emits a
content-addressed catalog of:

* launcher, connector, capability-registry, CLI, Flask/integrated/MCP++,
  compatibility-adapter, HF, MCP AI-model, and native-tool route identities;
* launcher↔connector route-table agreement or exact counterexamples;
* invocation mode classification (canonical JSON-RPC, reviewed adapter,
  compatibility, mock/degraded, synthesized alias);
* model id / revision / generation parameters and result / error / provenance
  preservation across consumer→handler boundaries; and
* proof eligibility — synthesized aliases and mock/degraded transports can
  never prove success or reachability.

The extractor is deliberately static and fail-closed.  It never imports model
servers, never opens transports, and never elevates mock, degraded, or
synthesized evidence into a success proof.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Final

from ..proof.formal_verification_contracts import content_identity
from .runtime_component_catalog import (
    RuntimeComponentCatalog,
    RuntimeComponentKind,
    RuntimeRouteKind,
)


MODEL_SERVER_CONTRACT_EXTRACTOR_INTERFACE: Final = "ModelServerContractExtractor@1"
MODEL_SERVER_CONTRACT_CATALOG_INTERFACE: Final = "ModelServerContractCatalog@1"
MODEL_SERVER_CONTRACT_EXTRACTOR_VERSION: Final = "1"

MODEL_SERVER_CATALOG_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/model-server-contract-catalog@1"
)
MODEL_SERVER_ROUTE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/model-server-route@1"
)
MODEL_SERVER_ROUTE_AGREEMENT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/model-server-route-agreement@1"
)
MODEL_SERVER_COUNTEREXAMPLE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/model-server-counterexample@1"
)
MODEL_SERVER_INVOCATION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/model-server-invocation@1"
)
MODEL_SERVER_FIELD_PRESERVATION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/model-server-field-preservation@1"
)
MODEL_SERVER_INFERENCE_CONTRACT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/model-server-inference-contract@1"
)
MODEL_SERVER_REVIEWED_ADAPTER_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/model-server-reviewed-adapter@1"
)

# Fields that must survive consumer → handler translation for inference.
REQUIRED_INFERENCE_FIELDS: Final[tuple[str, ...]] = (
    "model_id",
    "model_revision",
    "parameters",
    "result",
    "error",
    "provenance",
)

CANONICAL_JSON_RPC_SELECTORS: Final[frozenset[str]] = frozenset(
    {"tools/list", "tools/call"}
)

_NON_PROVING_MODES: Final[frozenset[str]] = frozenset(
    {
        "mock_transport",
        "degraded_transport",
        "synthesized_alias",
    }
)

_FASTAPI_ROUTE_RE: Final[re.Pattern[str]] = re.compile(
    r"""@(?P<target>[\w.]+)\.(?P<method>get|post|put|delete|patch|websocket)\(\s*['"](?P<path>[^'"]+)['"]""",
    re.MULTILINE,
)

_TS_JSON_RPC_RE: Final[re.Pattern[str]] = re.compile(
    r"""(?:jsonRpc|jsonrpc|request)\(\s*['"](?P<method>tools/(?:list|call)|[^'"]+)['"]""",
    re.MULTILINE | re.IGNORECASE,
)

_TS_HEALTH_RE: Final[re.Pattern[str]] = re.compile(
    r"""['"](?P<path>/health(?:/[^'"]*)?)['"]""",
    re.MULTILINE,
)

_MCP_TOOL_DECORATOR_RE: Final[re.Pattern[str]] = re.compile(
    r"""@(?P<owner>\w+)\.tool\(\s*(?:name\s*=\s*['"](?P<name>[^'"]+)['"])?""",
    re.MULTILINE,
)


class ModelServerContractExtractorError(ValueError):
    """Malformed or unsafe model-server contract extraction input."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str = "model_server_contract_error",
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.details = dict(details or {})


class RouteSurface(str, Enum):
    """Where a model-server route declaration was observed."""

    LAUNCHER = "launcher"
    CONNECTOR = "connector"
    CAPABILITY_REGISTRY = "capability_registry"
    CLI_LAUNCHER = "cli_launcher"
    FLASK_SERVER = "flask_server"
    INTEGRATED_SERVER = "integrated_server"
    MCP_PLUS_PLUS_SERVER = "mcp_plus_plus_server"
    COMPATIBILITY_ADAPTER = "compatibility_adapter"
    HF_MODEL_SERVER = "hf_model_server"
    MCP_AI_MODEL_SERVER = "mcp_ai_model_server"
    NATIVE_MODEL_TOOL = "native_model_tool"
    PACKAGE_HANDLER = "package_handler"


class ModelServerRouteKind(str, Enum):
    """Route kinds retained for model-server assurance."""

    CONNECTOR = "connector"
    LAUNCHER = "launcher"
    HEALTH = "health"
    LIST = "list"
    CALL = "call"
    COMPLETIONS = "completions"
    CHAT = "chat"
    EMBEDDINGS = "embeddings"
    MODELS = "models"
    AUTH = "auth"
    QUEUE = "queue"
    BATCH = "batch"
    CACHE = "cache"
    STREAM = "stream"
    ERROR = "error"
    PROVENANCE = "provenance"
    SCHEMA = "schema"
    OTHER = "other"


class InvocationMode(str, Enum):
    """How the consumer reaches a model-server handler."""

    CANONICAL_JSON_RPC = "canonical_json_rpc"
    REVIEWED_ADAPTER = "reviewed_adapter"
    COMPATIBILITY = "compatibility"
    DIRECT_REST = "direct_rest"
    SYNTHESIZED_ALIAS = "synthesized_alias"
    MOCK_TRANSPORT = "mock_transport"
    DEGRADED_TRANSPORT = "degraded_transport"
    UNKNOWN = "unknown"


class ProofEligibility(str, Enum):
    """Whether evidence may participate in a success proof."""

    PROOF_ELIGIBLE = "proof_eligible"
    NON_PROVING = "non_proving"


class AgreementState(str, Enum):
    """Launcher/connector (or peer surface) table comparison outcome."""

    AGREED = "agreed"
    REFUTED = "refuted"
    UNKNOWN = "unknown"


class PreservationState(str, Enum):
    """Field survival across a consumer→handler boundary."""

    PRESERVED = "preserved"
    REFUTED = "refuted"
    UNKNOWN = "unknown"


def _text(value: Any, name: str, *, required: bool = True) -> str:
    if value is None:
        text = ""
    elif isinstance(value, str):
        text = value
    else:
        raise ModelServerContractExtractorError(
            f"{name} must be text",
            reason_code="invalid_text_field",
            details={"field": name},
        )
    if "\x00" in text:
        raise ModelServerContractExtractorError(
            f"{name} must not contain NUL",
            reason_code="invalid_text_field",
            details={"field": name},
        )
    if required and not text.strip():
        raise ModelServerContractExtractorError(
            f"{name} is required",
            reason_code="missing_text_field",
            details={"field": name},
        )
    return text


def _plain(value: Any, *, depth: int = 0) -> Any:
    if depth > 32:
        raise ModelServerContractExtractorError(
            "model-server contract evidence exceeds nesting bound",
            reason_code="nesting_bound",
        )
    if isinstance(value, Enum):
        return value.value
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        raise ModelServerContractExtractorError(
            "floating values are not canonical model-server contract evidence",
            reason_code="non_canonical_float",
        )
    if isinstance(value, Mapping):
        if len(value) > 4_096:
            raise ModelServerContractExtractorError(
                "contract object is oversized",
                reason_code="object_oversized",
            )
        return {
            str(key): _plain(value[key], depth=depth + 1)
            for key in sorted(value, key=lambda item: str(item))
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if len(value) > 65_536:
            raise ModelServerContractExtractorError(
                "contract sequence is oversized",
                reason_code="sequence_oversized",
            )
        return [_plain(item, depth=depth + 1) for item in value]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _plain(to_dict(), depth=depth + 1)
    raise ModelServerContractExtractorError(
        f"unsupported contract value: {type(value).__name__}",
        reason_code="unsupported_value_type",
    )


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ModelServerContractExtractorError(
            f"{name} must be an object",
            reason_code="invalid_object_field",
            details={"field": name},
        )
    return value


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ModelServerContractExtractorError(
            f"{name} must be an array",
            reason_code="invalid_array_field",
            details={"field": name},
        )
    return value


def _strings(value: Any, name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ModelServerContractExtractorError(
            f"{name} must be a sequence of strings",
            reason_code="invalid_string_sequence",
            details={"field": name},
        )
    return tuple(sorted({_text(str(item), name, required=False) for item in value if str(item)}))


def _enum(value: Any, enum_type: type[Enum], name: str) -> Any:
    if isinstance(value, enum_type):
        return value
    raw = getattr(value, "value", value)
    try:
        return enum_type(str(raw))
    except (TypeError, ValueError) as exc:
        raise ModelServerContractExtractorError(
            f"unknown {name}: {value!r}",
            reason_code="invalid_enum",
            details={"field": name, "value": value},
        ) from exc


def _cid(value: Any) -> str:
    return content_identity(_plain(value))


def _source_digest(text: str) -> str:
    return "sha256:" + hashlib.sha256(
        text.encode("utf-8", errors="surrogatepass")
    ).hexdigest()


def _route_kind_from_selector(selector: str, declared: str | None = None) -> ModelServerRouteKind:
    if declared:
        try:
            return ModelServerRouteKind(declared)
        except ValueError:
            pass
    selector_l = selector.lower()
    if selector_l in {"tools/list", "list_tools", "list"}:
        return ModelServerRouteKind.LIST
    if selector_l in {"tools/call", "call_tool", "call"}:
        return ModelServerRouteKind.CALL
    if "health" in selector_l:
        return ModelServerRouteKind.HEALTH
    if "chat" in selector_l:
        return ModelServerRouteKind.CHAT
    if "completion" in selector_l:
        return ModelServerRouteKind.COMPLETIONS
    if "embedding" in selector_l:
        return ModelServerRouteKind.EMBEDDINGS
    if selector_l.endswith("/models") or selector_l == "/v1/models" or selector_l == "models":
        return ModelServerRouteKind.MODELS
    if "auth" in selector_l:
        return ModelServerRouteKind.AUTH
    if "queue" in selector_l:
        return ModelServerRouteKind.QUEUE
    if "batch" in selector_l:
        return ModelServerRouteKind.BATCH
    if "cache" in selector_l:
        return ModelServerRouteKind.CACHE
    if "stream" in selector_l:
        return ModelServerRouteKind.STREAM
    if "provenance" in selector_l:
        return ModelServerRouteKind.PROVENANCE
    if "error" in selector_l:
        return ModelServerRouteKind.ERROR
    if "schema" in selector_l:
        return ModelServerRouteKind.SCHEMA
    if selector_l in {"startmcpserver", "launcher"}:
        return ModelServerRouteKind.LAUNCHER
    if "connector" in selector_l:
        return ModelServerRouteKind.CONNECTOR
    return ModelServerRouteKind.OTHER


def _invocation_mode_from_raw(
    mode: Any,
    *,
    transport: str,
    selector: str,
    is_mock: bool = False,
    is_degraded: bool = False,
    is_synthesized: bool = False,
) -> InvocationMode:
    if is_mock:
        return InvocationMode.MOCK_TRANSPORT
    if is_degraded:
        return InvocationMode.DEGRADED_TRANSPORT
    if is_synthesized:
        return InvocationMode.SYNTHESIZED_ALIAS
    if mode is not None and str(mode):
        return _enum(mode, InvocationMode, "invocation_mode")
    transport_l = transport.lower()
    selector_l = selector.lower()
    if transport_l in {"json-rpc", "jsonrpc", "mcp"} and selector_l in CANONICAL_JSON_RPC_SELECTORS:
        return InvocationMode.CANONICAL_JSON_RPC
    if transport_l in {"json-rpc", "jsonrpc"} and selector_l.startswith("tools/"):
        return InvocationMode.CANONICAL_JSON_RPC
    if "mock" in transport_l:
        return InvocationMode.MOCK_TRANSPORT
    if "degraded" in transport_l:
        return InvocationMode.DEGRADED_TRANSPORT
    if "compat" in transport_l or "legacy" in transport_l or "shim" in transport_l:
        return InvocationMode.COMPATIBILITY
    if transport_l in {"http", "https", "rest"} and selector_l.startswith("/"):
        return InvocationMode.DIRECT_REST
    if transport_l in {"adapter", "reviewed-adapter"}:
        return InvocationMode.REVIEWED_ADAPTER
    return InvocationMode.UNKNOWN


def _proof_eligibility(mode: InvocationMode) -> ProofEligibility:
    if mode.value in _NON_PROVING_MODES or mode is InvocationMode.UNKNOWN:
        return ProofEligibility.NON_PROVING
    if mode is InvocationMode.CANONICAL_JSON_RPC:
        return ProofEligibility.PROOF_ELIGIBLE
    if mode is InvocationMode.REVIEWED_ADAPTER:
        return ProofEligibility.PROOF_ELIGIBLE
    # Compatibility and direct REST remain visible but cannot alone prove
    # MCP++-mediated success without a reviewed adapter binding.
    return ProofEligibility.NON_PROVING


@dataclass(frozen=True, slots=True)
class ModelServerCounterexample:
    """Exact expected-vs-actual witness for a route or field disagreement."""

    reason_code: str
    boundary_id: str
    path: str
    expected: Any
    actual: Any
    source_ids: tuple[str, ...] = ()
    counterexample_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "reason_code", _text(self.reason_code, "reason_code")
        )
        object.__setattr__(
            self, "boundary_id", _text(self.boundary_id, "boundary_id")
        )
        object.__setattr__(
            self, "path", _text(self.path, "path", required=False)
        )
        object.__setattr__(self, "expected", _plain(self.expected))
        object.__setattr__(self, "actual", _plain(self.actual))
        object.__setattr__(
            self, "source_ids", _strings(self.source_ids, "source_ids")
        )
        derived = _cid(self._identity_payload())
        if self.counterexample_id and self.counterexample_id != derived:
            raise ModelServerContractExtractorError(
                "counterexample_id does not match content",
                reason_code="counterexample_id_mismatch",
            )
        object.__setattr__(self, "counterexample_id", derived)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": MODEL_SERVER_COUNTEREXAMPLE_SCHEMA,
            "reason_code": self.reason_code,
            "boundary_id": self.boundary_id,
            "path": self.path,
            "expected": self.expected,
            "actual": self.actual,
            "source_ids": list(self.source_ids),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "counterexample_id": self.counterexample_id,
            **self._identity_payload(),
        }


@dataclass(frozen=True, slots=True)
class ReviewedAdapter:
    """A versioned, reviewed adapter that may authorize non-canonical routes."""

    adapter_id: str
    from_surface: str
    to_surface: str
    version: str
    review_id: str
    source_ids: tuple[str, ...]
    maps: Mapping[str, str] = field(default_factory=dict)
    identity: str = ""

    def __post_init__(self) -> None:
        for name in ("adapter_id", "from_surface", "to_surface", "version", "review_id"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        object.__setattr__(self, "source_ids", _strings(self.source_ids, "source_ids"))
        if not self.source_ids:
            raise ModelServerContractExtractorError(
                "reviewed adapter requires authority-bearing source_ids",
                reason_code="adapter_missing_sources",
            )
        maps = {
            _text(str(key), "maps.key"): _text(str(value), "maps.value")
            for key, value in dict(self.maps or {}).items()
        }
        object.__setattr__(self, "maps", MappingProxyType(maps))
        derived = _cid(self._identity_payload())
        if self.identity and self.identity != derived:
            raise ModelServerContractExtractorError(
                "reviewed adapter identity mismatch",
                reason_code="adapter_identity_mismatch",
            )
        object.__setattr__(self, "identity", derived)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": MODEL_SERVER_REVIEWED_ADAPTER_SCHEMA,
            "adapter_id": self.adapter_id,
            "from_surface": self.from_surface,
            "to_surface": self.to_surface,
            "version": self.version,
            "review_id": self.review_id,
            "source_ids": list(self.source_ids),
            "maps": dict(self.maps),
        }

    def to_dict(self) -> dict[str, Any]:
        return {"identity": self.identity, **self._identity_payload()}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ReviewedAdapter":
        raw = _mapping(value, "reviewed_adapter")
        return cls(
            adapter_id=str(raw.get("adapter_id") or raw.get("adapterId") or ""),
            from_surface=str(raw.get("from_surface") or raw.get("fromSurface") or ""),
            to_surface=str(raw.get("to_surface") or raw.get("toSurface") or ""),
            version=str(raw.get("version") or ""),
            review_id=str(raw.get("review_id") or raw.get("reviewId") or ""),
            source_ids=tuple(raw.get("source_ids") or raw.get("sourceIds") or ()),
            maps=dict(raw.get("maps") or {}),
            identity=str(raw.get("identity") or ""),
        )


@dataclass(frozen=True, slots=True)
class ModelServerRoute:
    """One exact route / schema / function identity on a model-server surface."""

    surface: RouteSurface
    kind: ModelServerRouteKind
    transport: str
    selector: str
    source_path: str
    function_symbol: str = ""
    schema_id: str = ""
    invocation_mode: InvocationMode = InvocationMode.UNKNOWN
    proof_eligibility: ProofEligibility = ProofEligibility.NON_PROVING
    component_id: str = ""
    component_root_cid: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)
    source_ids: tuple[str, ...] = ()
    route_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "surface", _enum(self.surface, RouteSurface, "surface")
        )
        object.__setattr__(
            self, "kind", _enum(self.kind, ModelServerRouteKind, "kind")
        )
        object.__setattr__(self, "transport", _text(self.transport, "transport"))
        object.__setattr__(self, "selector", _text(self.selector, "selector"))
        object.__setattr__(
            self, "source_path", _text(self.source_path, "source_path")
        )
        object.__setattr__(
            self,
            "function_symbol",
            _text(self.function_symbol, "function_symbol", required=False),
        )
        object.__setattr__(
            self, "schema_id", _text(self.schema_id, "schema_id", required=False)
        )
        mode = _enum(self.invocation_mode, InvocationMode, "invocation_mode")
        object.__setattr__(self, "invocation_mode", mode)
        eligibility = _enum(
            self.proof_eligibility, ProofEligibility, "proof_eligibility"
        )
        # Non-proving modes always win over a claimed eligibility.
        if mode.value in _NON_PROVING_MODES:
            eligibility = ProofEligibility.NON_PROVING
        elif eligibility is ProofEligibility.PROOF_ELIGIBLE:
            eligibility = _proof_eligibility(mode)
        object.__setattr__(self, "proof_eligibility", eligibility)
        object.__setattr__(
            self,
            "component_id",
            _text(self.component_id, "component_id", required=False),
        )
        object.__setattr__(
            self,
            "component_root_cid",
            _text(self.component_root_cid, "component_root_cid", required=False),
        )
        object.__setattr__(
            self, "metadata", MappingProxyType(dict(_plain(self.metadata or {})))
        )
        object.__setattr__(self, "source_ids", _strings(self.source_ids, "source_ids"))
        derived = _cid(self._identity_payload())
        if self.route_id and self.route_id != derived:
            raise ModelServerContractExtractorError(
                "route_id does not match content",
                reason_code="route_id_mismatch",
            )
        object.__setattr__(self, "route_id", derived)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": MODEL_SERVER_ROUTE_SCHEMA,
            "surface": self.surface.value,
            "kind": self.kind.value,
            "transport": self.transport,
            "selector": self.selector,
            "source_path": self.source_path,
            "function_symbol": self.function_symbol,
            "schema_id": self.schema_id,
            "invocation_mode": self.invocation_mode.value,
            "proof_eligibility": self.proof_eligibility.value,
            "component_id": self.component_id,
            "component_root_cid": self.component_root_cid,
            "metadata": dict(self.metadata),
            "source_ids": list(self.source_ids),
        }

    def to_dict(self) -> dict[str, Any]:
        return {"route_id": self.route_id, **self._identity_payload()}

    def comparison_key(self) -> tuple[str, str, str]:
        """Kind + transport + selector used for launcher/connector agreement."""

        return (self.kind.value, self.transport, self.selector)


@dataclass(frozen=True, slots=True)
class RouteTableAgreement:
    """Agreement or refutation between two route tables (typically launcher vs connector)."""

    left_surface: RouteSurface
    right_surface: RouteSurface
    state: AgreementState
    matched_route_ids: tuple[str, ...]
    counterexamples: tuple[ModelServerCounterexample, ...]
    component_id: str = ""
    agreement_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "left_surface", _enum(self.left_surface, RouteSurface, "left_surface")
        )
        object.__setattr__(
            self,
            "right_surface",
            _enum(self.right_surface, RouteSurface, "right_surface"),
        )
        object.__setattr__(self, "state", _enum(self.state, AgreementState, "state"))
        object.__setattr__(
            self,
            "matched_route_ids",
            tuple(sorted({_text(item, "matched_route_ids") for item in self.matched_route_ids})),
        )
        items = tuple(
            item
            if isinstance(item, ModelServerCounterexample)
            else ModelServerCounterexample(**item)  # type: ignore[arg-type]
            for item in self.counterexamples
        )
        by_id = {item.counterexample_id: item for item in items}
        object.__setattr__(
            self,
            "counterexamples",
            tuple(by_id[key] for key in sorted(by_id)),
        )
        object.__setattr__(
            self,
            "component_id",
            _text(self.component_id, "component_id", required=False),
        )
        if self.state is AgreementState.REFUTED and not self.counterexamples:
            raise ModelServerContractExtractorError(
                "refuted route agreement requires exact counterexamples",
                reason_code="refutation_missing_counterexample",
            )
        if self.state is AgreementState.AGREED and self.counterexamples:
            raise ModelServerContractExtractorError(
                "agreed route tables cannot retain counterexamples",
                reason_code="agreed_with_counterexamples",
            )
        derived = _cid(self._identity_payload())
        if self.agreement_id and self.agreement_id != derived:
            raise ModelServerContractExtractorError(
                "agreement_id does not match content",
                reason_code="agreement_id_mismatch",
            )
        object.__setattr__(self, "agreement_id", derived)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": MODEL_SERVER_ROUTE_AGREEMENT_SCHEMA,
            "left_surface": self.left_surface.value,
            "right_surface": self.right_surface.value,
            "state": self.state.value,
            "matched_route_ids": list(self.matched_route_ids),
            "counterexamples": [item.to_dict() for item in self.counterexamples],
            "component_id": self.component_id,
        }

    def to_dict(self) -> dict[str, Any]:
        return {"agreement_id": self.agreement_id, **self._identity_payload()}


@dataclass(frozen=True, slots=True)
class InvocationContract:
    """Normalized invocation path with proof eligibility."""

    operation_id: str
    mode: InvocationMode
    proof_eligibility: ProofEligibility
    transport: str
    selector: str
    surface: RouteSurface
    adapter_identity: str = ""
    can_prove_success: bool = False
    reason_codes: tuple[str, ...] = ()
    source_ids: tuple[str, ...] = ()
    invocation_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "operation_id", _text(self.operation_id, "operation_id")
        )
        mode = _enum(self.mode, InvocationMode, "mode")
        object.__setattr__(self, "mode", mode)
        eligibility = _proof_eligibility(mode)
        if mode is InvocationMode.REVIEWED_ADAPTER and not self.adapter_identity:
            eligibility = ProofEligibility.NON_PROVING
        object.__setattr__(self, "proof_eligibility", eligibility)
        object.__setattr__(self, "transport", _text(self.transport, "transport"))
        object.__setattr__(self, "selector", _text(self.selector, "selector"))
        object.__setattr__(
            self, "surface", _enum(self.surface, RouteSurface, "surface")
        )
        object.__setattr__(
            self,
            "adapter_identity",
            _text(self.adapter_identity, "adapter_identity", required=False),
        )
        # Explicit non-proving modes can never prove success, regardless of
        # any caller-supplied can_prove_success claim.
        can_prove = (
            eligibility is ProofEligibility.PROOF_ELIGIBLE
            and mode.value not in _NON_PROVING_MODES
            and (
                mode is not InvocationMode.REVIEWED_ADAPTER
                or bool(self.adapter_identity)
            )
        )
        object.__setattr__(self, "can_prove_success", can_prove)
        object.__setattr__(
            self, "reason_codes", _strings(self.reason_codes, "reason_codes")
        )
        object.__setattr__(self, "source_ids", _strings(self.source_ids, "source_ids"))
        derived = _cid(self._identity_payload())
        if self.invocation_id and self.invocation_id != derived:
            raise ModelServerContractExtractorError(
                "invocation_id does not match content",
                reason_code="invocation_id_mismatch",
            )
        object.__setattr__(self, "invocation_id", derived)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": MODEL_SERVER_INVOCATION_SCHEMA,
            "operation_id": self.operation_id,
            "mode": self.mode.value,
            "proof_eligibility": self.proof_eligibility.value,
            "transport": self.transport,
            "selector": self.selector,
            "surface": self.surface.value,
            "adapter_identity": self.adapter_identity,
            "can_prove_success": self.can_prove_success,
            "reason_codes": list(self.reason_codes),
            "source_ids": list(self.source_ids),
        }

    def to_dict(self) -> dict[str, Any]:
        return {"invocation_id": self.invocation_id, **self._identity_payload()}


@dataclass(frozen=True, slots=True)
class FieldPreservation:
    """Whether one inference field survives a consumer→handler boundary."""

    field_path: str
    state: PreservationState
    consumer_value: Any
    handler_value: Any
    counterexamples: tuple[ModelServerCounterexample, ...] = ()
    source_ids: tuple[str, ...] = ()
    preservation_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "field_path", _text(self.field_path, "field_path"))
        object.__setattr__(
            self, "state", _enum(self.state, PreservationState, "state")
        )
        object.__setattr__(self, "consumer_value", _plain(self.consumer_value))
        object.__setattr__(self, "handler_value", _plain(self.handler_value))
        items = tuple(
            item
            if isinstance(item, ModelServerCounterexample)
            else ModelServerCounterexample(**item)  # type: ignore[arg-type]
            for item in self.counterexamples
        )
        by_id = {item.counterexample_id: item for item in items}
        object.__setattr__(
            self,
            "counterexamples",
            tuple(by_id[key] for key in sorted(by_id)),
        )
        object.__setattr__(self, "source_ids", _strings(self.source_ids, "source_ids"))
        if self.state is PreservationState.REFUTED and not self.counterexamples:
            raise ModelServerContractExtractorError(
                "refuted field preservation requires a counterexample",
                reason_code="preservation_missing_counterexample",
            )
        derived = _cid(self._identity_payload())
        if self.preservation_id and self.preservation_id != derived:
            raise ModelServerContractExtractorError(
                "preservation_id does not match content",
                reason_code="preservation_id_mismatch",
            )
        object.__setattr__(self, "preservation_id", derived)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": MODEL_SERVER_FIELD_PRESERVATION_SCHEMA,
            "field_path": self.field_path,
            "state": self.state.value,
            "consumer_value": self.consumer_value,
            "handler_value": self.handler_value,
            "counterexamples": [item.to_dict() for item in self.counterexamples],
            "source_ids": list(self.source_ids),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "preservation_id": self.preservation_id,
            **self._identity_payload(),
        }


@dataclass(frozen=True, slots=True)
class InferenceContract:
    """Model selection and generation contract bound to one operation."""

    operation_id: str
    model_id: str
    model_revision: str
    parameters: Mapping[str, Any]
    result_fields: tuple[str, ...]
    error_fields: tuple[str, ...]
    provenance_fields: tuple[str, ...]
    consumer_fields: Mapping[str, Any]
    handler_fields: Mapping[str, Any]
    preservations: tuple[FieldPreservation, ...]
    source_ids: tuple[str, ...] = ()
    contract_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "operation_id", _text(self.operation_id, "operation_id")
        )
        object.__setattr__(self, "model_id", _text(self.model_id, "model_id"))
        object.__setattr__(
            self,
            "model_revision",
            _text(self.model_revision, "model_revision", required=False),
        )
        object.__setattr__(
            self,
            "parameters",
            MappingProxyType(dict(_plain(self.parameters or {}))),
        )
        object.__setattr__(
            self, "result_fields", _strings(self.result_fields, "result_fields")
        )
        object.__setattr__(
            self, "error_fields", _strings(self.error_fields, "error_fields")
        )
        object.__setattr__(
            self,
            "provenance_fields",
            _strings(self.provenance_fields, "provenance_fields"),
        )
        object.__setattr__(
            self,
            "consumer_fields",
            MappingProxyType(dict(_plain(self.consumer_fields or {}))),
        )
        object.__setattr__(
            self,
            "handler_fields",
            MappingProxyType(dict(_plain(self.handler_fields or {}))),
        )
        items = tuple(
            item
            if isinstance(item, FieldPreservation)
            else FieldPreservation(**item)  # type: ignore[arg-type]
            for item in self.preservations
        )
        object.__setattr__(
            self,
            "preservations",
            tuple(sorted(items, key=lambda item: item.field_path)),
        )
        object.__setattr__(self, "source_ids", _strings(self.source_ids, "source_ids"))
        derived = _cid(self._identity_payload())
        if self.contract_id and self.contract_id != derived:
            raise ModelServerContractExtractorError(
                "contract_id does not match content",
                reason_code="inference_contract_id_mismatch",
            )
        object.__setattr__(self, "contract_id", derived)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": MODEL_SERVER_INFERENCE_CONTRACT_SCHEMA,
            "operation_id": self.operation_id,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "parameters": dict(self.parameters),
            "result_fields": list(self.result_fields),
            "error_fields": list(self.error_fields),
            "provenance_fields": list(self.provenance_fields),
            "consumer_fields": dict(self.consumer_fields),
            "handler_fields": dict(self.handler_fields),
            "preservations": [item.to_dict() for item in self.preservations],
            "source_ids": list(self.source_ids),
        }

    def to_dict(self) -> dict[str, Any]:
        return {"contract_id": self.contract_id, **self._identity_payload()}

    @property
    def all_fields_preserved(self) -> bool:
        return bool(self.preservations) and all(
            item.state is PreservationState.PRESERVED for item in self.preservations
        )


@dataclass(frozen=True, slots=True)
class ModelServerContractCatalog:
    """Content-addressed model-server route and inference contract catalog."""

    component_id: str
    component_root_cid: str
    routes: tuple[ModelServerRoute, ...]
    agreements: tuple[RouteTableAgreement, ...]
    invocations: tuple[InvocationContract, ...]
    inference_contracts: tuple[InferenceContract, ...]
    reviewed_adapters: tuple[ReviewedAdapter, ...]
    extractor_version: str = MODEL_SERVER_CONTRACT_EXTRACTOR_VERSION
    catalog_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "component_id", _text(self.component_id, "component_id")
        )
        object.__setattr__(
            self,
            "component_root_cid",
            _text(self.component_root_cid, "component_root_cid", required=False),
        )
        object.__setattr__(
            self,
            "routes",
            tuple(sorted(self.routes, key=lambda item: item.route_id)),
        )
        object.__setattr__(
            self,
            "agreements",
            tuple(sorted(self.agreements, key=lambda item: item.agreement_id)),
        )
        object.__setattr__(
            self,
            "invocations",
            tuple(sorted(self.invocations, key=lambda item: item.invocation_id)),
        )
        object.__setattr__(
            self,
            "inference_contracts",
            tuple(
                sorted(self.inference_contracts, key=lambda item: item.contract_id)
            ),
        )
        object.__setattr__(
            self,
            "reviewed_adapters",
            tuple(sorted(self.reviewed_adapters, key=lambda item: item.identity)),
        )
        object.__setattr__(
            self,
            "extractor_version",
            _text(self.extractor_version, "extractor_version"),
        )
        derived = _cid(self._identity_payload())
        if self.catalog_id and self.catalog_id != derived:
            raise ModelServerContractExtractorError(
                "catalog_id does not match content",
                reason_code="catalog_id_mismatch",
            )
        object.__setattr__(self, "catalog_id", derived)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": MODEL_SERVER_CATALOG_SCHEMA,
            "interface": MODEL_SERVER_CONTRACT_CATALOG_INTERFACE,
            "extractor_interface": MODEL_SERVER_CONTRACT_EXTRACTOR_INTERFACE,
            "extractor_version": self.extractor_version,
            "component_id": self.component_id,
            "component_root_cid": self.component_root_cid,
            "routes": [item.to_dict() for item in self.routes],
            "agreements": [item.to_dict() for item in self.agreements],
            "invocations": [item.to_dict() for item in self.invocations],
            "inference_contracts": [
                item.to_dict() for item in self.inference_contracts
            ],
            "reviewed_adapters": [item.to_dict() for item in self.reviewed_adapters],
        }

    def to_dict(self) -> dict[str, Any]:
        return {"catalog_id": self.catalog_id, **self._identity_payload()}

    def routes_for_surface(self, surface: RouteSurface | str) -> tuple[ModelServerRoute, ...]:
        surface_e = _enum(surface, RouteSurface, "surface")
        return tuple(route for route in self.routes if route.surface is surface_e)

    def launcher_connector_agreement(self) -> RouteTableAgreement | None:
        for agreement in self.agreements:
            if (
                agreement.left_surface is RouteSurface.LAUNCHER
                and agreement.right_surface is RouteSurface.CONNECTOR
            ) or (
                agreement.left_surface is RouteSurface.CONNECTOR
                and agreement.right_surface is RouteSurface.LAUNCHER
            ):
                return agreement
        return None

    def proof_eligible_invocations(self) -> tuple[InvocationContract, ...]:
        return tuple(
            item
            for item in self.invocations
            if item.can_prove_success
            and item.proof_eligibility is ProofEligibility.PROOF_ELIGIBLE
        )

    def non_proving_invocations(self) -> tuple[InvocationContract, ...]:
        return tuple(
            item for item in self.invocations if not item.can_prove_success
        )


def compare_route_tables(
    left: Sequence[ModelServerRoute],
    right: Sequence[ModelServerRoute],
    *,
    left_surface: RouteSurface,
    right_surface: RouteSurface,
    component_id: str = "",
) -> RouteTableAgreement:
    """Agree or refute two route tables with exact counterexamples."""

    left_by_kind: dict[str, list[ModelServerRoute]] = {}
    right_by_kind: dict[str, list[ModelServerRoute]] = {}
    for route in left:
        left_by_kind.setdefault(route.kind.value, []).append(route)
    for route in right:
        right_by_kind.setdefault(route.kind.value, []).append(route)

    kinds = sorted(set(left_by_kind) | set(right_by_kind))
    matched: list[str] = []
    counterexamples: list[ModelServerCounterexample] = []

    for kind in kinds:
        left_routes = left_by_kind.get(kind, [])
        right_routes = right_by_kind.get(kind, [])
        if not left_routes:
            for route in right_routes:
                counterexamples.append(
                    ModelServerCounterexample(
                        reason_code="route_missing_on_left",
                        boundary_id=f"{left_surface.value}:{right_surface.value}",
                        path=f"routes.{kind}",
                        expected=None,
                        actual={
                            "transport": route.transport,
                            "selector": route.selector,
                            "source_path": route.source_path,
                        },
                        source_ids=route.source_ids or (route.route_id,),
                    )
                )
            continue
        if not right_routes:
            for route in left_routes:
                counterexamples.append(
                    ModelServerCounterexample(
                        reason_code="route_missing_on_right",
                        boundary_id=f"{left_surface.value}:{right_surface.value}",
                        path=f"routes.{kind}",
                        expected={
                            "transport": route.transport,
                            "selector": route.selector,
                            "source_path": route.source_path,
                        },
                        actual=None,
                        source_ids=route.source_ids or (route.route_id,),
                    )
                )
            continue

        left_keys = {route.comparison_key(): route for route in left_routes}
        right_keys = {route.comparison_key(): route for route in right_routes}
        for key, left_route in sorted(left_keys.items()):
            right_route = right_keys.get(key)
            if right_route is None:
                # Same kind present but transport/selector disagree.
                sample = next(iter(right_keys.values()))
                counterexamples.append(
                    ModelServerCounterexample(
                        reason_code="route_selector_mismatch",
                        boundary_id=f"{left_surface.value}:{right_surface.value}",
                        path=f"routes.{kind}",
                        expected={
                            "transport": left_route.transport,
                            "selector": left_route.selector,
                            "source_path": left_route.source_path,
                        },
                        actual={
                            "transport": sample.transport,
                            "selector": sample.selector,
                            "source_path": sample.source_path,
                        },
                        source_ids=tuple(
                            sorted(
                                {
                                    *(left_route.source_ids or (left_route.route_id,)),
                                    *(sample.source_ids or (sample.route_id,)),
                                }
                            )
                        ),
                    )
                )
            else:
                matched.append(left_route.route_id)
                matched.append(right_route.route_id)
        for key, right_route in sorted(right_keys.items()):
            if key not in left_keys:
                sample = next(iter(left_keys.values()))
                counterexamples.append(
                    ModelServerCounterexample(
                        reason_code="route_selector_mismatch",
                        boundary_id=f"{left_surface.value}:{right_surface.value}",
                        path=f"routes.{kind}",
                        expected={
                            "transport": sample.transport,
                            "selector": sample.selector,
                            "source_path": sample.source_path,
                        },
                        actual={
                            "transport": right_route.transport,
                            "selector": right_route.selector,
                            "source_path": right_route.source_path,
                        },
                        source_ids=tuple(
                            sorted(
                                {
                                    *(sample.source_ids or (sample.route_id,)),
                                    *(
                                        right_route.source_ids
                                        or (right_route.route_id,)
                                    ),
                                }
                            )
                        ),
                    )
                )

    # Deduplicate counterexamples that mirror each other for the same kind.
    unique_cx: dict[str, ModelServerCounterexample] = {}
    for item in counterexamples:
        unique_cx[item.counterexample_id] = item
    counterexamples = list(unique_cx.values())

    if counterexamples:
        state = AgreementState.REFUTED
    elif matched:
        state = AgreementState.AGREED
    else:
        state = AgreementState.UNKNOWN

    return RouteTableAgreement(
        left_surface=left_surface,
        right_surface=right_surface,
        state=state,
        matched_route_ids=tuple(sorted(set(matched))),
        counterexamples=tuple(counterexamples),
        component_id=component_id,
    )


def _field_lookup(fields: Mapping[str, Any], path: str) -> Any:
    if path in fields:
        return fields[path]
    current: Any = fields
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def preserve_inference_fields(
    *,
    operation_id: str,
    consumer_fields: Mapping[str, Any],
    handler_fields: Mapping[str, Any],
    required_fields: Sequence[str] = REQUIRED_INFERENCE_FIELDS,
    reviewed_adapters: Sequence[ReviewedAdapter] = (),
    source_ids: Sequence[str] = (),
) -> tuple[FieldPreservation, ...]:
    """Compare consumer and handler field bags for exact preservation."""

    adapter_maps: dict[str, str] = {}
    for adapter in reviewed_adapters:
        adapter_maps.update(dict(adapter.maps))

    results: list[FieldPreservation] = []
    consumer = dict(consumer_fields)
    handler = dict(handler_fields)
    sources = _strings(source_ids, "source_ids")

    for field_path in required_fields:
        consumer_value = _field_lookup(consumer, field_path)
        # Allow reviewed rename of the leaf field.
        handler_key = adapter_maps.get(field_path, field_path)
        handler_value = _field_lookup(handler, handler_key)
        if handler_value is None and handler_key != field_path:
            handler_value = _field_lookup(handler, field_path)

        if consumer_value is None and handler_value is None:
            results.append(
                FieldPreservation(
                    field_path=field_path,
                    state=PreservationState.UNKNOWN,
                    consumer_value=None,
                    handler_value=None,
                    counterexamples=(),
                    source_ids=sources,
                )
            )
            continue

        if consumer_value is None or handler_value is None:
            results.append(
                FieldPreservation(
                    field_path=field_path,
                    state=PreservationState.REFUTED,
                    consumer_value=consumer_value,
                    handler_value=handler_value,
                    counterexamples=(
                        ModelServerCounterexample(
                            reason_code="field_missing",
                            boundary_id=operation_id,
                            path=field_path,
                            expected=consumer_value,
                            actual=handler_value,
                            source_ids=sources,
                        ),
                    ),
                    source_ids=sources,
                )
            )
            continue

        if _plain(consumer_value) != _plain(handler_value):
            results.append(
                FieldPreservation(
                    field_path=field_path,
                    state=PreservationState.REFUTED,
                    consumer_value=consumer_value,
                    handler_value=handler_value,
                    counterexamples=(
                        ModelServerCounterexample(
                            reason_code="field_value_mismatch",
                            boundary_id=operation_id,
                            path=field_path,
                            expected=consumer_value,
                            actual=handler_value,
                            source_ids=sources,
                        ),
                    ),
                    source_ids=sources,
                )
            )
            continue

        results.append(
            FieldPreservation(
                field_path=field_path,
                state=PreservationState.PRESERVED,
                consumer_value=consumer_value,
                handler_value=handler_value,
                counterexamples=(),
                source_ids=sources,
            )
        )
    return tuple(results)


def _parse_route_entry(
    raw: Mapping[str, Any],
    *,
    default_surface: RouteSurface | None = None,
    component_id: str = "",
    component_root_cid: str = "",
) -> ModelServerRoute:
    if raw.get("surface"):
        surface = _enum(raw.get("surface"), RouteSurface, "surface")
    elif default_surface is not None:
        surface = default_surface
    else:
        surface = RouteSurface.PACKAGE_HANDLER

    selector = _text(
        raw.get("selector") or raw.get("path") or raw.get("method") or "",
        "selector",
    )
    transport = _text(raw.get("transport") or "unknown", "transport")
    kind = _route_kind_from_selector(
        selector, str(raw.get("kind") or "") or None
    )
    is_mock = bool(raw.get("mock") or raw.get("is_mock"))
    is_degraded = bool(raw.get("degraded") or raw.get("is_degraded"))
    is_synthesized = bool(
        raw.get("synthesized")
        or raw.get("is_synthesized")
        or raw.get("synthesized_alias")
    )
    mode = _invocation_mode_from_raw(
        raw.get("invocation_mode") or raw.get("mode"),
        transport=transport,
        selector=selector,
        is_mock=is_mock,
        is_degraded=is_degraded,
        is_synthesized=is_synthesized,
    )
    eligibility = _proof_eligibility(mode)
    source_ids = tuple(raw.get("source_ids") or raw.get("sourceIds") or ())
    if not source_ids and raw.get("source_path"):
        source_ids = (str(raw.get("source_path")),)
    return ModelServerRoute(
        surface=surface,
        kind=kind,
        transport=transport,
        selector=selector,
        source_path=_text(
            raw.get("source_path") or raw.get("sourcePath") or "unknown",
            "source_path",
        ),
        function_symbol=str(
            raw.get("function_symbol")
            or raw.get("functionSymbol")
            or raw.get("symbol")
            or ""
        ),
        schema_id=str(raw.get("schema_id") or raw.get("schemaId") or ""),
        invocation_mode=mode,
        proof_eligibility=eligibility,
        component_id=str(raw.get("component_id") or component_id or ""),
        component_root_cid=str(
            raw.get("component_root_cid") or component_root_cid or ""
        ),
        metadata=dict(raw.get("metadata") or {}),
        source_ids=source_ids,
        route_id=str(raw.get("route_id") or ""),
    )


def extract_fastapi_routes_from_source(
    source: str,
    *,
    source_path: str,
    surface: RouteSurface = RouteSurface.HF_MODEL_SERVER,
    component_id: str = "",
    component_root_cid: str = "",
) -> tuple[ModelServerRoute, ...]:
    """Statically extract FastAPI route decorators from Python source text."""

    routes: list[ModelServerRoute] = []
    for match in _FASTAPI_ROUTE_RE.finditer(source):
        path = match.group("path")
        method = match.group("method").upper()
        kind = _route_kind_from_selector(path)
        mode = (
            InvocationMode.DIRECT_REST
            if path.startswith("/")
            else InvocationMode.UNKNOWN
        )
        routes.append(
            ModelServerRoute(
                surface=surface,
                kind=kind,
                transport="http",
                selector=path,
                source_path=source_path,
                function_symbol="",
                invocation_mode=mode,
                proof_eligibility=_proof_eligibility(mode),
                component_id=component_id,
                component_root_cid=component_root_cid,
                metadata={"http_method": method},
                source_ids=(f"{source_path}:{path}",),
            )
        )
    return tuple(routes)


def extract_mcp_tools_from_source(
    source: str,
    *,
    source_path: str,
    surface: RouteSurface = RouteSurface.MCP_AI_MODEL_SERVER,
    component_id: str = "",
    component_root_cid: str = "",
) -> tuple[ModelServerRoute, ...]:
    """Statically extract ``@mcp.tool`` registrations and following def names."""

    routes: list[ModelServerRoute] = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        # Fall back to regex-only extraction when source is a partial fixture.
        for match in _MCP_TOOL_DECORATOR_RE.finditer(source):
            name = match.group("name") or "unnamed_tool"
            routes.append(
                ModelServerRoute(
                    surface=surface,
                    kind=ModelServerRouteKind.CALL,
                    transport="mcp",
                    selector=name,
                    source_path=source_path,
                    function_symbol=name,
                    invocation_mode=InvocationMode.CANONICAL_JSON_RPC,
                    proof_eligibility=ProofEligibility.PROOF_ELIGIBLE,
                    component_id=component_id,
                    component_root_cid=component_root_cid,
                    source_ids=(f"{source_path}:{name}",),
                )
            )
        return tuple(routes)

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            is_tool = False
            tool_name: str | None = None
            if isinstance(decorator, ast.Call):
                func = decorator.func
                if isinstance(func, ast.Attribute) and func.attr == "tool":
                    is_tool = True
                elif isinstance(func, ast.Name) and func.id == "tool":
                    is_tool = True
                for keyword in decorator.keywords:
                    if keyword.arg == "name" and isinstance(
                        keyword.value, ast.Constant
                    ):
                        tool_name = str(keyword.value.value)
            elif isinstance(decorator, ast.Attribute) and decorator.attr == "tool":
                is_tool = True
            if not is_tool:
                continue
            name = tool_name or node.name
            routes.append(
                ModelServerRoute(
                    surface=surface,
                    kind=ModelServerRouteKind.CALL,
                    transport="mcp",
                    selector=name,
                    source_path=source_path,
                    function_symbol=node.name,
                    invocation_mode=InvocationMode.CANONICAL_JSON_RPC,
                    proof_eligibility=ProofEligibility.PROOF_ELIGIBLE,
                    component_id=component_id,
                    component_root_cid=component_root_cid,
                    source_ids=(f"{source_path}:{name}",),
                )
            )
    return tuple(routes)


def extract_typescript_jsonrpc_routes_from_source(
    source: str,
    *,
    source_path: str,
    surface: RouteSurface,
    component_id: str = "",
    component_root_cid: str = "",
) -> tuple[ModelServerRoute, ...]:
    """Extract literal JSON-RPC method and health paths from TypeScript text."""

    routes: list[ModelServerRoute] = []
    seen: set[tuple[str, str]] = set()
    for match in _TS_JSON_RPC_RE.finditer(source):
        method = match.group("method")
        key = ("json-rpc", method)
        if key in seen:
            continue
        seen.add(key)
        kind = _route_kind_from_selector(method)
        mode = (
            InvocationMode.CANONICAL_JSON_RPC
            if method in CANONICAL_JSON_RPC_SELECTORS
            else InvocationMode.UNKNOWN
        )
        routes.append(
            ModelServerRoute(
                surface=surface,
                kind=kind,
                transport="json-rpc",
                selector=method,
                source_path=source_path,
                invocation_mode=mode,
                proof_eligibility=_proof_eligibility(mode),
                component_id=component_id,
                component_root_cid=component_root_cid,
                source_ids=(f"{source_path}:{method}",),
            )
        )
    for match in _TS_HEALTH_RE.finditer(source):
        path = match.group("path")
        key = ("http", path)
        if key in seen:
            continue
        seen.add(key)
        routes.append(
            ModelServerRoute(
                surface=surface,
                kind=ModelServerRouteKind.HEALTH,
                transport="http",
                selector=path,
                source_path=source_path,
                invocation_mode=InvocationMode.DIRECT_REST,
                proof_eligibility=ProofEligibility.NON_PROVING,
                component_id=component_id,
                component_root_cid=component_root_cid,
                source_ids=(f"{source_path}:{path}",),
            )
        )
    return tuple(routes)


def _build_invocation(
    route: ModelServerRoute,
    *,
    operation_id: str | None = None,
    reviewed_adapters: Sequence[ReviewedAdapter] = (),
) -> InvocationContract:
    adapter_identity = ""
    mode = route.invocation_mode
    if mode is InvocationMode.REVIEWED_ADAPTER or (
        mode is InvocationMode.COMPATIBILITY and reviewed_adapters
    ):
        for adapter in reviewed_adapters:
            if (
                adapter.from_surface == route.surface.value
                or route.selector in adapter.maps
                or route.transport in adapter.maps.values()
            ):
                adapter_identity = adapter.identity
                mode = InvocationMode.REVIEWED_ADAPTER
                break
    reasons: list[str] = []
    if mode is InvocationMode.CANONICAL_JSON_RPC:
        reasons.append("canonical_json_rpc")
    elif mode is InvocationMode.REVIEWED_ADAPTER and adapter_identity:
        reasons.append("reviewed_adapter")
    elif mode is InvocationMode.MOCK_TRANSPORT:
        reasons.append("mock_transport_non_proving")
    elif mode is InvocationMode.DEGRADED_TRANSPORT:
        reasons.append("degraded_transport_non_proving")
    elif mode is InvocationMode.SYNTHESIZED_ALIAS:
        reasons.append("synthesized_alias_non_proving")
    elif mode is InvocationMode.COMPATIBILITY:
        reasons.append("compatibility_requires_reviewed_adapter")
    elif mode is InvocationMode.DIRECT_REST:
        reasons.append("direct_rest_non_mcp_proof")
    else:
        reasons.append("unknown_invocation_mode")

    return InvocationContract(
        operation_id=operation_id or f"{route.surface.value}:{route.selector}",
        mode=mode,
        proof_eligibility=_proof_eligibility(mode),
        transport=route.transport,
        selector=route.selector,
        surface=route.surface,
        adapter_identity=adapter_identity,
        can_prove_success=True,  # constructor enforces mode gate
        reason_codes=tuple(reasons),
        source_ids=route.source_ids or (route.route_id,),
    )


def _bind_component(
    payload: Mapping[str, Any],
    runtime_catalog: RuntimeComponentCatalog | None,
) -> tuple[str, str]:
    component_id = str(
        payload.get("component_id") or payload.get("componentId") or "model-server"
    )
    component_root_cid = str(
        payload.get("component_root_cid")
        or payload.get("componentRootCid")
        or ""
    )
    if runtime_catalog is not None:
        model_components = [
            component
            for component in runtime_catalog.components
            if component.kind is RuntimeComponentKind.MODEL_SERVER
            and (
                component.component_id == component_id
                or component.authority.kind.value == "primary"
                and component_id in {"model-server", component.component_id}
            )
        ]
        if component_id:
            matches = [
                component
                for component in runtime_catalog.components
                if component.component_id == component_id
            ]
            if matches:
                model_components = matches
        if not model_components:
            primaries = [
                component
                for component in runtime_catalog.components
                if component.kind is RuntimeComponentKind.MODEL_SERVER
                and component.authority.kind.value == "primary"
            ]
            model_components = primaries
        if model_components:
            component = model_components[0]
            component_id = component.component_id
            component_root_cid = component.root_cid
    return component_id, component_root_cid


def _routes_from_runtime_catalog(
    runtime_catalog: RuntimeComponentCatalog,
    *,
    component_id: str,
) -> list[ModelServerRoute]:
    routes: list[ModelServerRoute] = []
    for route in runtime_catalog.routes:
        if route.component_id != component_id:
            continue
        surface = {
            RuntimeRouteKind.CONNECTOR: RouteSurface.CONNECTOR,
            RuntimeRouteKind.LAUNCHER: RouteSurface.LAUNCHER,
            RuntimeRouteKind.HEALTH: RouteSurface.CONNECTOR,
            RuntimeRouteKind.LIST: RouteSurface.CONNECTOR,
            RuntimeRouteKind.CALL: RouteSurface.CONNECTOR,
        }.get(route.kind, RouteSurface.CONNECTOR)
        if route.kind is RuntimeRouteKind.LAUNCHER:
            surface = RouteSurface.LAUNCHER
        mode = _invocation_mode_from_raw(
            None, transport=route.transport, selector=route.selector
        )
        routes.append(
            ModelServerRoute(
                surface=surface,
                kind=_enum(route.kind.value, ModelServerRouteKind, "kind")
                if route.kind.value
                in {item.value for item in ModelServerRouteKind}
                else _route_kind_from_selector(route.selector, route.kind.value),
                transport=route.transport,
                selector=route.selector,
                source_path=route.source_path,
                function_symbol=route.selector
                if route.kind in {RuntimeRouteKind.CONNECTOR, RuntimeRouteKind.LAUNCHER}
                else "",
                invocation_mode=mode,
                proof_eligibility=_proof_eligibility(mode),
                component_id=component_id,
                component_root_cid=route.component_root_cid,
                source_ids=(route.route_id,),
            )
        )
    return routes


class ModelServerContractExtractor:
    """Extract and normalize model-server route and inference contracts."""

    def extract(
        self,
        payload: Mapping[str, Any],
        *,
        runtime_catalog: RuntimeComponentCatalog | None = None,
    ) -> ModelServerContractCatalog:
        """Extract a model-server contract catalog from structured premises.

        Accepted payload keys (all optional except where noted):

        * ``component_id`` / ``component_root_cid``
        * ``launcher_routes``, ``connector_routes``, ``capability_registry_routes``
        * ``cli_routes``, ``flask_routes``, ``integrated_routes``,
          ``mcp_plus_plus_routes``, ``compatibility_adapter_routes``
        * ``hf_routes``, ``mcp_ai_routes``, ``native_tool_routes``
        * ``routes`` — free-form list with explicit ``surface``
        * ``sources`` — ``{path: {language, text, surface?}}`` for static scan
        * ``reviewed_adapters``
        * ``inference_contracts`` — list of consumer/handler field bags
        * ``invocations`` — optional explicit invocation records
        """

        if not isinstance(payload, Mapping):
            raise ModelServerContractExtractorError(
                "payload must be an object",
                reason_code="invalid_payload",
            )

        component_id, component_root_cid = _bind_component(payload, runtime_catalog)
        routes: list[ModelServerRoute] = []

        if runtime_catalog is not None:
            routes.extend(
                _routes_from_runtime_catalog(
                    runtime_catalog, component_id=component_id
                )
            )

        surface_lists: tuple[tuple[str, RouteSurface], ...] = (
            ("launcher_routes", RouteSurface.LAUNCHER),
            ("connector_routes", RouteSurface.CONNECTOR),
            ("capability_registry_routes", RouteSurface.CAPABILITY_REGISTRY),
            ("cli_routes", RouteSurface.CLI_LAUNCHER),
            ("flask_routes", RouteSurface.FLASK_SERVER),
            ("integrated_routes", RouteSurface.INTEGRATED_SERVER),
            ("mcp_plus_plus_routes", RouteSurface.MCP_PLUS_PLUS_SERVER),
            ("compatibility_adapter_routes", RouteSurface.COMPATIBILITY_ADAPTER),
            ("hf_routes", RouteSurface.HF_MODEL_SERVER),
            ("mcp_ai_routes", RouteSurface.MCP_AI_MODEL_SERVER),
            ("native_tool_routes", RouteSurface.NATIVE_MODEL_TOOL),
        )
        for key, surface in surface_lists:
            if key not in payload:
                continue
            for item in _sequence(payload.get(key), key):
                routes.append(
                    _parse_route_entry(
                        _mapping(item, key),
                        default_surface=surface,
                        component_id=component_id,
                        component_root_cid=component_root_cid,
                    )
                )

        if "routes" in payload:
            for item in _sequence(payload.get("routes"), "routes"):
                routes.append(
                    _parse_route_entry(
                        _mapping(item, "routes[]"),
                        component_id=component_id,
                        component_root_cid=component_root_cid,
                    )
                )

        sources = payload.get("sources")
        if sources is not None:
            source_map = _mapping(sources, "sources")
            for path, entry in source_map.items():
                entry_map = (
                    entry
                    if isinstance(entry, Mapping)
                    else {"text": entry, "language": "auto"}
                )
                text = str(entry_map.get("text") or entry_map.get("source") or "")
                language = str(
                    entry_map.get("language") or entry_map.get("lang") or "auto"
                ).lower()
                surface_raw = entry_map.get("surface")
                surface = (
                    _enum(surface_raw, RouteSurface, "source.surface")
                    if surface_raw
                    else None
                )
                path_s = str(path)
                if language in {"python", "py", "auto"} and (
                    path_s.endswith(".py") or language in {"python", "py"}
                ):
                    if surface in {
                        RouteSurface.MCP_AI_MODEL_SERVER,
                        RouteSurface.NATIVE_MODEL_TOOL,
                    } or "ai_model" in path_s or "mcp" in path_s:
                        routes.extend(
                            extract_mcp_tools_from_source(
                                text,
                                source_path=path_s,
                                surface=surface or RouteSurface.MCP_AI_MODEL_SERVER,
                                component_id=component_id,
                                component_root_cid=component_root_cid,
                            )
                        )
                    if surface in {
                        RouteSurface.HF_MODEL_SERVER,
                        RouteSurface.FLASK_SERVER,
                        RouteSurface.INTEGRATED_SERVER,
                        None,
                    } or "hf_model" in path_s or "server.py" in path_s:
                        routes.extend(
                            extract_fastapi_routes_from_source(
                                text,
                                source_path=path_s,
                                surface=surface or RouteSurface.HF_MODEL_SERVER,
                                component_id=component_id,
                                component_root_cid=component_root_cid,
                            )
                        )
                if language in {"typescript", "ts", "javascript", "js", "auto"} and (
                    path_s.endswith((".ts", ".tsx", ".js", ".mjs"))
                    or language in {"typescript", "ts", "javascript", "js"}
                ):
                    default_surface = surface or (
                        RouteSurface.LAUNCHER
                        if "entrypoint" in path_s or path_s.endswith("mcp.ts")
                        else RouteSurface.CONNECTOR
                    )
                    routes.extend(
                        extract_typescript_jsonrpc_routes_from_source(
                            text,
                            source_path=path_s,
                            surface=default_surface,
                            component_id=component_id,
                            component_root_cid=component_root_cid,
                        )
                    )

        # Deduplicate routes by identity.
        by_id: dict[str, ModelServerRoute] = {}
        for route in routes:
            by_id[route.route_id] = route
        unique_routes = list(by_id.values())

        reviewed_adapters = tuple(
            ReviewedAdapter.from_dict(_mapping(item, "reviewed_adapters[]"))
            for item in _sequence(
                payload.get("reviewed_adapters") or (), "reviewed_adapters"
            )
        )

        launcher = [
            route
            for route in unique_routes
            if route.surface is RouteSurface.LAUNCHER
        ]
        connector = [
            route
            for route in unique_routes
            if route.surface is RouteSurface.CONNECTOR
        ]
        # For agreement, compare operational kinds shared by both tables
        # (health/list/call) even if launcher only declares launcher identity.
        # If launcher only has LAUNCHER kind, fold catalog-profile launcher
        # health/list/call from connector-bound runtime routes that share
        # the launcher source is not automatic — callers must declare
        # operational routes under launcher_routes for parity checks.
        agreements: list[RouteTableAgreement] = []
        if launcher and connector:
            agreements.append(
                compare_route_tables(
                    launcher,
                    connector,
                    left_surface=RouteSurface.LAUNCHER,
                    right_surface=RouteSurface.CONNECTOR,
                    component_id=component_id,
                )
            )
        # Also compare capability registry vs connector when both present.
        capability = [
            route
            for route in unique_routes
            if route.surface is RouteSurface.CAPABILITY_REGISTRY
        ]
        if capability and connector:
            agreements.append(
                compare_route_tables(
                    capability,
                    connector,
                    left_surface=RouteSurface.CAPABILITY_REGISTRY,
                    right_surface=RouteSurface.CONNECTOR,
                    component_id=component_id,
                )
            )

        invocations: list[InvocationContract] = []
        # Always derive invocations from operational routes so mock/degraded
        # surface flags cannot be omitted from the proof-eligibility ledger.
        for route in unique_routes:
            if route.kind in {
                ModelServerRouteKind.LIST,
                ModelServerRouteKind.CALL,
                ModelServerRouteKind.COMPLETIONS,
                ModelServerRouteKind.CHAT,
                ModelServerRouteKind.EMBEDDINGS,
                ModelServerRouteKind.HEALTH,
            } or route.invocation_mode in {
                InvocationMode.CANONICAL_JSON_RPC,
                InvocationMode.REVIEWED_ADAPTER,
                InvocationMode.MOCK_TRANSPORT,
                InvocationMode.DEGRADED_TRANSPORT,
                InvocationMode.SYNTHESIZED_ALIAS,
                InvocationMode.COMPATIBILITY,
                InvocationMode.DIRECT_REST,
            }:
                invocations.append(
                    _build_invocation(
                        route, reviewed_adapters=reviewed_adapters
                    )
                )
        if "invocations" in payload:
            for item in _sequence(payload.get("invocations"), "invocations"):
                raw = _mapping(item, "invocations[]")
                mode = _invocation_mode_from_raw(
                    raw.get("mode") or raw.get("invocation_mode"),
                    transport=str(raw.get("transport") or ""),
                    selector=str(raw.get("selector") or ""),
                    is_mock=bool(raw.get("mock")),
                    is_degraded=bool(raw.get("degraded")),
                    is_synthesized=bool(
                        raw.get("synthesized") or raw.get("synthesized_alias")
                    ),
                )
                adapter_identity = str(raw.get("adapter_identity") or "")
                if mode is InvocationMode.REVIEWED_ADAPTER and not adapter_identity:
                    for adapter in reviewed_adapters:
                        if adapter.adapter_id == raw.get("adapter_id"):
                            adapter_identity = adapter.identity
                            break
                        if adapter.review_id == raw.get("review_id"):
                            adapter_identity = adapter.identity
                            break
                surface = _enum(
                    raw.get("surface") or RouteSurface.CONNECTOR,
                    RouteSurface,
                    "invocation.surface",
                )
                reasons = tuple(raw.get("reason_codes") or ())
                if not reasons:
                    if mode is InvocationMode.MOCK_TRANSPORT:
                        reasons = ("mock_transport_non_proving",)
                    elif mode is InvocationMode.DEGRADED_TRANSPORT:
                        reasons = ("degraded_transport_non_proving",)
                    elif mode is InvocationMode.SYNTHESIZED_ALIAS:
                        reasons = ("synthesized_alias_non_proving",)
                    elif mode is InvocationMode.CANONICAL_JSON_RPC:
                        reasons = ("canonical_json_rpc",)
                    elif mode is InvocationMode.REVIEWED_ADAPTER:
                        reasons = ("reviewed_adapter",)
                invocations.append(
                    InvocationContract(
                        operation_id=_text(
                            raw.get("operation_id")
                            or raw.get("operationId")
                            or "op",
                            "operation_id",
                        ),
                        mode=mode,
                        proof_eligibility=_proof_eligibility(mode),
                        transport=_text(
                            raw.get("transport") or "unknown", "transport"
                        ),
                        selector=_text(
                            raw.get("selector") or "unknown", "selector"
                        ),
                        surface=surface,
                        adapter_identity=adapter_identity,
                        can_prove_success=True,
                        reason_codes=reasons,
                        source_ids=tuple(raw.get("source_ids") or ()),
                    )
                )

        # Deduplicate invocations.
        inv_by_id: dict[str, InvocationContract] = {
            item.invocation_id: item for item in invocations
        }
        unique_invocations = list(inv_by_id.values())

        inference_contracts: list[InferenceContract] = []
        for item in _sequence(
            payload.get("inference_contracts") or (), "inference_contracts"
        ):
            raw = _mapping(item, "inference_contracts[]")
            operation_id = _text(
                raw.get("operation_id") or raw.get("operationId") or "inference",
                "operation_id",
            )
            consumer_provided = "consumer_fields" in raw or "consumer" in raw
            handler_provided = "handler_fields" in raw or "handler" in raw
            consumer_fields = dict(
                raw.get("consumer_fields") or raw.get("consumer") or {}
            )
            handler_fields = dict(
                raw.get("handler_fields") or raw.get("handler") or {}
            )
            # Promote top-level model fields only into bags that were not
            # explicitly supplied.  An explicit handler bag that omits a field
            # is a preservation counterexample, not a cue to back-fill.
            for key in (
                "model_id",
                "model_revision",
                "parameters",
                "result",
                "error",
                "provenance",
            ):
                if key in raw and key not in consumer_fields and not consumer_provided:
                    consumer_fields[key] = raw[key]
                if f"handler_{key}" in raw:
                    handler_fields[key] = raw[f"handler_{key}"]
                elif key in raw and key not in handler_fields and not handler_provided:
                    handler_fields[key] = raw[key]
            # Still allow sparse top-level convenience when consumer bag is
            # present but omits a declared identity field.
            if consumer_provided:
                for key in ("model_id", "model_revision", "parameters"):
                    if key in raw and key not in consumer_fields:
                        consumer_fields[key] = raw[key]
            model_id = str(
                raw.get("model_id")
                or consumer_fields.get("model_id")
                or handler_fields.get("model_id")
                or ""
            )
            model_revision = str(
                raw.get("model_revision")
                or consumer_fields.get("model_revision")
                or handler_fields.get("model_revision")
                or ""
            )
            parameters = dict(
                raw.get("parameters")
                or consumer_fields.get("parameters")
                or {}
            )
            if "parameters" not in consumer_fields and parameters:
                consumer_fields["parameters"] = parameters
            if (
                "parameters" not in handler_fields
                and parameters
                and not handler_provided
            ):
                handler_fields["parameters"] = parameters
            if "model_id" not in consumer_fields and model_id:
                consumer_fields["model_id"] = model_id
            if "model_id" not in handler_fields and model_id and not handler_provided:
                handler_fields["model_id"] = model_id
            if "model_revision" not in consumer_fields and model_revision:
                consumer_fields["model_revision"] = model_revision
            if (
                "model_revision" not in handler_fields
                and model_revision
                and not handler_provided
            ):
                handler_fields["model_revision"] = model_revision

            required = tuple(
                raw.get("required_fields") or REQUIRED_INFERENCE_FIELDS
            )
            preservations = preserve_inference_fields(
                operation_id=operation_id,
                consumer_fields=consumer_fields,
                handler_fields=handler_fields,
                required_fields=required,
                reviewed_adapters=reviewed_adapters,
                source_ids=tuple(raw.get("source_ids") or ()),
            )
            result_fields = tuple(
                raw.get("result_fields")
                or (
                    list(consumer_fields.get("result") or [])
                    if isinstance(consumer_fields.get("result"), Sequence)
                    and not isinstance(consumer_fields.get("result"), (str, bytes))
                    else list((consumer_fields.get("result") or {}).keys())
                    if isinstance(consumer_fields.get("result"), Mapping)
                    else ()
                )
            )
            error_fields = tuple(
                raw.get("error_fields")
                or (
                    list((consumer_fields.get("error") or {}).keys())
                    if isinstance(consumer_fields.get("error"), Mapping)
                    else ()
                )
            )
            provenance_fields = tuple(
                raw.get("provenance_fields")
                or (
                    list((consumer_fields.get("provenance") or {}).keys())
                    if isinstance(consumer_fields.get("provenance"), Mapping)
                    else ()
                )
            )
            inference_contracts.append(
                InferenceContract(
                    operation_id=operation_id,
                    model_id=model_id or "unknown",
                    model_revision=model_revision,
                    parameters=parameters,
                    result_fields=result_fields,
                    error_fields=error_fields,
                    provenance_fields=provenance_fields,
                    consumer_fields=consumer_fields,
                    handler_fields=handler_fields,
                    preservations=preservations,
                    source_ids=tuple(raw.get("source_ids") or ()),
                )
            )

        return ModelServerContractCatalog(
            component_id=component_id,
            component_root_cid=component_root_cid,
            routes=tuple(unique_routes),
            agreements=tuple(agreements),
            invocations=tuple(unique_invocations),
            inference_contracts=tuple(inference_contracts),
            reviewed_adapters=reviewed_adapters,
        )


def extract_model_server_contracts(
    payload: Mapping[str, Any],
    *,
    runtime_catalog: RuntimeComponentCatalog | None = None,
) -> ModelServerContractCatalog:
    """Module-level convenience wrapper for :class:`ModelServerContractExtractor`."""

    return ModelServerContractExtractor().extract(
        payload, runtime_catalog=runtime_catalog
    )


__all__ = [
    "MODEL_SERVER_CONTRACT_EXTRACTOR_INTERFACE",
    "MODEL_SERVER_CONTRACT_CATALOG_INTERFACE",
    "MODEL_SERVER_CONTRACT_EXTRACTOR_VERSION",
    "MODEL_SERVER_CATALOG_SCHEMA",
    "REQUIRED_INFERENCE_FIELDS",
    "CANONICAL_JSON_RPC_SELECTORS",
    "ModelServerContractExtractorError",
    "RouteSurface",
    "ModelServerRouteKind",
    "InvocationMode",
    "ProofEligibility",
    "AgreementState",
    "PreservationState",
    "ModelServerCounterexample",
    "ReviewedAdapter",
    "ModelServerRoute",
    "RouteTableAgreement",
    "InvocationContract",
    "FieldPreservation",
    "InferenceContract",
    "ModelServerContractCatalog",
    "ModelServerContractExtractor",
    "compare_route_tables",
    "preserve_inference_fields",
    "extract_fastapi_routes_from_source",
    "extract_mcp_tools_from_source",
    "extract_typescript_jsonrpc_routes_from_source",
    "extract_model_server_contracts",
]
