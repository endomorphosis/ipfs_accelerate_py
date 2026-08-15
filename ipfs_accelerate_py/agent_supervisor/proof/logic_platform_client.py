"""SupervisorLogicPlatformClient@1 — lazy supervisor boundary to the logic platform.

LPC-110 / LPC-G110.  One handshake + typed invocation surface for:

* platform handshake (LogicPlatformManifest@1)
* catalog snapshot reads
* formalization (FormalizationArtifact@3)
* domain slice / obligation / plan construction
* capability discovery
* typed provider invocation (LogicProviderProtocol@2)
* reconstruction and verification
* receipt projection
* counterexample projection
* cache-key construction and freshness checks

Importing this module never imports ``ipfs_datasets_py``.  Datasets packages
load only for an explicit client method.  Callers cannot overclaim authority:
request ceilings are checked against evidence kind before dispatch.

Requests bind task, tree, policy, plan, budget, network, cancellation,
deadline, correlation, evidence, and authority.  Semantic identities remain
datasets-owned; the supervisor owns scheduling, isolation, and placement.
"""

from __future__ import annotations

import hashlib
import importlib
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Final

from .canonical_logic_adapter import (
    SUPERVISOR_CANONICAL_LOGIC_ADAPTER_INTERFACE,
    SupervisorCanonicalLogicAdapter,
    get_canonical_logic_adapter,
)
from .formal_verification_contracts import EvidenceFreshness


# ---------------------------------------------------------------------------
# Interface / schema identities
# ---------------------------------------------------------------------------

SUPERVISOR_LOGIC_PLATFORM_CLIENT_INTERFACE: Final = "SupervisorLogicPlatformClient@1"
SUPERVISOR_LOGIC_PLATFORM_CLIENT_VERSION: Final = "1.0.0"
SUPERVISOR_LOGIC_PLATFORM_CLIENT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/logic-platform-client@1"
)
SUPERVISOR_LOGIC_PLATFORM_CLIENT_TASK_ID: Final = "LPC-110"
SUPERVISOR_LOGIC_PLATFORM_CLIENT_GOAL_ID: Final = "LPC-G110"

CLIENT_REQUEST_CONTEXT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/logic-platform-client-request@1"
)
CLIENT_INVOCATION_RESULT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/logic-platform-client-result@1"
)
CLIENT_CACHE_FRESHNESS_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/logic-platform-cache-freshness@1"
)
CLIENT_RECEIPT_VIEW_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/logic-platform-receipt-view@1"
)
CLIENT_COUNTEREXAMPLE_VIEW_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/logic-platform-counterexample-view@1"
)

# Lazy datasets module paths (never imported at module load).
_MANIFEST_MODULE: Final = "ipfs_datasets_py.logic.platform.manifest"
_CATALOG_MODULE: Final = "ipfs_datasets_py.logic.families.canonical_catalog"
_ARTIFACTS_V3_MODULE: Final = "ipfs_datasets_py.logic.formalization.artifacts_v3"
_REQUESTS_V2_MODULE: Final = "ipfs_datasets_py.logic.backends.requests_v2"
_PROTOCOL_V2_MODULE: Final = "ipfs_datasets_py.logic.backends.protocol_v2"
_RESPONSE_V2_MODULE: Final = "ipfs_datasets_py.logic.backends.response_v2"
_CACHE_KEY_MODULE: Final = "ipfs_datasets_py.logic.common.canonical_cache_key"
_AXES_MODULE: Final = "ipfs_datasets_py.logic.ir_core.axes"
_TACTICIAN_MODULE: Final = (
    "ipfs_datasets_py.logic.software_verification.tactician.contracts"
)
_NAMESPACES_MODULE: Final = "ipfs_datasets_py.logic.families.namespaces"
_COUNTEREXAMPLE_MODULE: Final = (
    "ipfs_datasets_py.logic.software_verification.counterexamples.contracts"
)
_RECEIPTS_MODULE: Final = "ipfs_datasets_py.logic.software_verification.receipts"

# Closed client operation vocabulary (acceptance surface).
CLIENT_OPERATIONS: Final[tuple[str, ...]] = (
    "handshake",
    "catalog",
    "formalize",
    "create_slice",
    "create_obligation",
    "create_plan",
    "discover_capabilities",
    "invoke",
    "reconstruct",
    "verify",
    "receipt",
    "counterexample",
    "cache_freshness",
)

# Request context fields required by LPC-G110 acceptance.
REQUIRED_CONTEXT_FIELDS: Final[tuple[str, ...]] = (
    "task_id",
    "tree_id",
    "policy_id",
    "plan_id",
    "budget",
    "network_allowed",
    "cancellation",
    "deadline_unix_ms",
    "correlation_id",
    "evidence_kind",
    "authority_ceiling",
)

# Authority ranks for overclaim checks when datasets is not yet loaded.
_AUTHORITY_RANK: Final[Mapping[str, int]] = MappingProxyType(
    {
        "none": 0,
        "advisory": 1,
        "candidate": 2,
        "bounded": 3,
        "finite_trace": 4,
        "authorization": 5,
        "satisfiability": 6,
        "protocol": 7,
        "reconstruction": 8,
        "kernel": 9,
        "attestation": 10,
    }
)

# Maximum request ceiling each evidence kind may claim (mirrors BackendRequest@2).
_EVIDENCE_AUTHORITY_CEILING: Final[Mapping[str, str]] = MappingProxyType(
    {
        "parse": "none",
        "advisory": "advisory",
        "candidate": "candidate",
        "model": "satisfiability",
        "core": "satisfiability",
        "trace": "finite_trace",
        "monitor": "finite_trace",
        "attack": "protocol",
        "proof": "reconstruction",
        "kernel": "kernel",
        "kernel_receipt": "kernel",
        "attestation": "attestation",
        "authorization": "authorization",
        "bounded": "bounded",
    }
)

_NON_KERNEL_EVIDENCE: Final[frozenset[str]] = frozenset(
    {
        "parse",
        "advisory",
        "candidate",
        "model",
        "core",
        "trace",
        "monitor",
        "attack",
        "authorization",
        "bounded",
    }
)

_NON_THEOREM_EVIDENCE: Final[frozenset[str]] = frozenset(
    {
        "parse",
        "advisory",
        "candidate",
        "model",
        "core",
        "trace",
        "monitor",
        "attack",
        "authorization",
        "bounded",
    }
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class LogicPlatformClientError(RuntimeError):
    """Raised when the supervisor logic-platform client cannot proceed."""


class LogicPlatformClientAdmissionError(LogicPlatformClientError):
    """Raised when a request fails closed before platform dispatch."""


class LogicPlatformClientAuthorityError(LogicPlatformClientAdmissionError):
    """Raised when a caller overclaims evidence authority."""


class LogicPlatformClientHandshakeError(LogicPlatformClientError):
    """Raised when an operation is attempted without a successful handshake."""


class LogicPlatformClientFreshnessError(LogicPlatformClientAdmissionError):
    """Raised when a cache entry fails freshness admission."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _text(value: object, field_name: str, *, optional: bool = False) -> str:
    if value is None:
        if optional:
            return ""
        raise LogicPlatformClientAdmissionError(f"{field_name} is required")
    if not isinstance(value, str):
        raise LogicPlatformClientAdmissionError(f"{field_name} must be a string")
    text = value.strip()
    if "\x00" in text:
        raise LogicPlatformClientAdmissionError(
            f"{field_name} must not contain NUL bytes"
        )
    if not text:
        if optional:
            return ""
        raise LogicPlatformClientAdmissionError(f"{field_name} is required")
    return text


def _optional_int(value: object, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise LogicPlatformClientAdmissionError(
            f"{field_name} must be an int or None"
        )
    if value < 0:
        raise LogicPlatformClientAdmissionError(
            f"{field_name} must be non-negative"
        )
    return value


def _digest(label: str) -> str:
    """Return a bare lowercase 64-hex sha256 digest (syntax-core form)."""

    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _digest_prefaced(label: str) -> str:
    """Return a ``sha256:<hex>`` digest (cache-key / content-id form)."""

    return "sha256:" + _digest(label)


def _bare_digest(value: object, field_name: str) -> str:
    """Normalize ``sha256:<hex>`` or bare hex digests to bare 64-hex form."""

    text = _text(value, field_name)
    if text.startswith("sha256:"):
        text = text[len("sha256:") :]
    text = text.strip().lower()
    if len(text) != 64 or any(ch not in "0123456789abcdef" for ch in text):
        raise LogicPlatformClientAdmissionError(
            f"{field_name} must be a lowercase 64-hex sha256 digest"
        )
    return text


def _new_id(prefix: str) -> str:
    return f"{prefix}:{uuid.uuid4().hex}"


def _token(value: object) -> str:
    if value is None:
        return ""
    # Prefer explicit .value (Enum / LogicIdentity) over str(object).
    if hasattr(value, "value") and not isinstance(value, type):
        raw = getattr(value, "value")
        if not isinstance(raw, (str, int, float, bool)) and hasattr(raw, "value"):
            raw = getattr(raw, "value")
        return str(raw).strip()
    return str(value).strip()


def _require_mapping(value: object, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise LogicPlatformClientAdmissionError(f"{field_name} must be a mapping")
    return value


def _authority_token(value: object) -> str:
    return _token(value).lower()


def _evidence_token(value: object) -> str:
    text = _token(value)
    # Accept qualified identities (evidence:candidate) and bare values.
    if ":" in text:
        text = text.rsplit(":", 1)[-1]
    if "/" in text:
        text = text.rsplit("/", 1)[-1]
    return text.lower()


def check_authority_overclaim(
    authority_ceiling: object,
    evidence_kind: object,
    *,
    field_name: str = "authority_ceiling",
) -> None:
    """Fail closed when a request ceiling exceeds evidence-kind support."""

    ceiling = _authority_token(authority_ceiling)
    evidence = _evidence_token(evidence_kind)
    if ceiling not in _AUTHORITY_RANK:
        raise LogicPlatformClientAuthorityError(
            f"{field_name} is not a known authority ceiling: {ceiling!r}"
        )
    max_ceiling = _EVIDENCE_AUTHORITY_CEILING.get(evidence, "advisory")
    if _AUTHORITY_RANK[ceiling] > _AUTHORITY_RANK[max_ceiling]:
        raise LogicPlatformClientAuthorityError(
            f"{field_name} {ceiling!r} overclaims evidence kind {evidence!r} "
            f"(max admitted ceiling {max_ceiling!r}); fail closed before dispatch"
        )
    if ceiling == "kernel" and evidence in _NON_KERNEL_EVIDENCE:
        raise LogicPlatformClientAuthorityError(
            f"kernel authority cannot be claimed with evidence {evidence!r}"
        )
    if evidence in _NON_THEOREM_EVIDENCE and ceiling in {
        "kernel",
        "reconstruction",
    }:
        raise LogicPlatformClientAuthorityError(
            f"proof/reconstruction authority cannot be claimed with evidence "
            f"{evidence!r}"
        )


# ---------------------------------------------------------------------------
# Request context and result carriers
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ClientRequestContext:
    """Operational bindings for one client call (LPC-G110 acceptance fields)."""

    task_id: str
    tree_id: str
    policy_id: str
    plan_id: str = ""
    budget: Mapping[str, Any] = field(default_factory=dict)
    network_allowed: bool = False
    cancellation: Mapping[str, Any] | None = None
    deadline_unix_ms: int | None = None
    correlation_id: str = ""
    evidence_kind: str = "candidate"
    authority_ceiling: str = "advisory"
    schema_version: str = CLIENT_REQUEST_CONTEXT_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", _text(self.task_id, "task_id"))
        object.__setattr__(self, "tree_id", _text(self.tree_id, "tree_id"))
        object.__setattr__(self, "policy_id", _text(self.policy_id, "policy_id"))
        object.__setattr__(
            self, "plan_id", _text(self.plan_id, "plan_id", optional=True)
        )
        if not isinstance(self.budget, Mapping):
            raise LogicPlatformClientAdmissionError("budget must be a mapping")
        object.__setattr__(self, "budget", MappingProxyType(dict(self.budget)))
        if not isinstance(self.network_allowed, bool):
            raise LogicPlatformClientAdmissionError(
                "network_allowed must be a bool"
            )
        if self.cancellation is not None and not isinstance(
            self.cancellation, Mapping
        ):
            raise LogicPlatformClientAdmissionError(
                "cancellation must be a mapping or None"
            )
        if self.cancellation is not None:
            object.__setattr__(
                self,
                "cancellation",
                MappingProxyType(dict(self.cancellation)),
            )
        object.__setattr__(
            self,
            "deadline_unix_ms",
            _optional_int(self.deadline_unix_ms, "deadline_unix_ms"),
        )
        object.__setattr__(
            self,
            "correlation_id",
            _text(self.correlation_id, "correlation_id", optional=True)
            or _new_id("corr"),
        )
        object.__setattr__(
            self, "evidence_kind", _evidence_token(self.evidence_kind)
        )
        object.__setattr__(
            self,
            "authority_ceiling",
            _authority_token(self.authority_ceiling),
        )
        object.__setattr__(
            self,
            "schema_version",
            _text(self.schema_version, "schema_version"),
        )
        if self.schema_version != CLIENT_REQUEST_CONTEXT_SCHEMA:
            raise LogicPlatformClientAdmissionError(
                f"unsupported client request schema {self.schema_version!r}"
            )
        check_authority_overclaim(self.authority_ceiling, self.evidence_kind)

    def to_dict(self) -> dict[str, Any]:
        return {
            "authority_ceiling": self.authority_ceiling,
            "budget": dict(self.budget),
            "cancellation": (
                None if self.cancellation is None else dict(self.cancellation)
            ),
            "correlation_id": self.correlation_id,
            "deadline_unix_ms": self.deadline_unix_ms,
            "evidence_kind": self.evidence_kind,
            "network_allowed": self.network_allowed,
            "plan_id": self.plan_id,
            "policy_id": self.policy_id,
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "tree_id": self.tree_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ClientRequestContext":
        payload = dict(_require_mapping(value, "ClientRequestContext"))
        return cls(
            task_id=str(payload.get("task_id") or ""),
            tree_id=str(payload.get("tree_id") or ""),
            policy_id=str(payload.get("policy_id") or ""),
            plan_id=str(payload.get("plan_id") or ""),
            budget=dict(payload.get("budget") or {}),
            network_allowed=bool(payload.get("network_allowed", False)),
            cancellation=payload.get("cancellation"),
            deadline_unix_ms=payload.get("deadline_unix_ms"),
            correlation_id=str(payload.get("correlation_id") or ""),
            evidence_kind=payload.get("evidence_kind") or "candidate",
            authority_ceiling=payload.get("authority_ceiling") or "advisory",
            schema_version=str(
                payload.get("schema_version") or CLIENT_REQUEST_CONTEXT_SCHEMA
            ),
        )


@dataclass(frozen=True, slots=True)
class ClientInvocationResult:
    """Typed client result wrapping a provider response and request context."""

    operation: str
    context: ClientRequestContext
    response: Any
    request: Any = None
    schema_version: str = CLIENT_INVOCATION_RESULT_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "operation", _text(self.operation, "operation")
        )
        if not isinstance(self.context, ClientRequestContext):
            raise LogicPlatformClientError(
                "context must be a ClientRequestContext"
            )
        object.__setattr__(
            self,
            "schema_version",
            _text(self.schema_version, "schema_version"),
        )

    def to_dict(self) -> dict[str, Any]:
        response = self.response
        if hasattr(response, "to_dict"):
            response_payload = response.to_dict()
        elif isinstance(response, Mapping):
            response_payload = dict(response)
        else:
            response_payload = {"value": repr(response)}
        request_payload = None
        if self.request is not None:
            if hasattr(self.request, "to_dict"):
                request_payload = self.request.to_dict()
            elif isinstance(self.request, Mapping):
                request_payload = dict(self.request)
        return {
            "context": self.context.to_dict(),
            "operation": self.operation,
            "request": request_payload,
            "response": response_payload,
            "schema_version": self.schema_version,
        }


@dataclass(frozen=True, slots=True)
class CacheFreshnessReport:
    """Outcome of a cache freshness check against a canonical key binding."""

    fresh: bool
    freshness: str
    key_id: str
    reasons: tuple[str, ...] = ()
    observed_environment: str = ""
    expected_environment: str = ""
    schema_version: str = CLIENT_CACHE_FRESHNESS_SCHEMA

    def __post_init__(self) -> None:
        if not isinstance(self.fresh, bool):
            raise LogicPlatformClientError("fresh must be a bool")
        object.__setattr__(
            self, "freshness", _text(self.freshness, "freshness")
        )
        object.__setattr__(self, "key_id", _text(self.key_id, "key_id"))
        object.__setattr__(self, "reasons", tuple(self.reasons or ()))
        object.__setattr__(
            self,
            "observed_environment",
            _text(
                self.observed_environment,
                "observed_environment",
                optional=True,
            ),
        )
        object.__setattr__(
            self,
            "expected_environment",
            _text(
                self.expected_environment,
                "expected_environment",
                optional=True,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "expected_environment": self.expected_environment,
            "fresh": self.fresh,
            "freshness": self.freshness,
            "key_id": self.key_id,
            "observed_environment": self.observed_environment,
            "reasons": list(self.reasons),
            "schema_version": self.schema_version,
        }


@dataclass(frozen=True, slots=True)
class ClientReceiptView:
    """Supervisor-facing receipt projection (not a second semantic authority)."""

    receipt_id: str
    request_id: str
    operation: str
    provider_id: str
    evidence_kind: str
    evidence_authority: str
    verdict: str
    operation_status: str
    context: ClientRequestContext
    simulated: bool = False
    translation_ids: tuple[str, ...] = ()
    artifact_ids: tuple[str, ...] = ()
    content_digest: str = ""
    schema_version: str = CLIENT_RECEIPT_VIEW_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "receipt_id", _text(self.receipt_id, "receipt_id")
        )
        object.__setattr__(
            self, "request_id", _text(self.request_id, "request_id")
        )
        object.__setattr__(
            self, "operation", _text(self.operation, "operation")
        )
        object.__setattr__(
            self, "provider_id", _text(self.provider_id, "provider_id")
        )
        object.__setattr__(
            self, "evidence_kind", _text(self.evidence_kind, "evidence_kind")
        )
        object.__setattr__(
            self,
            "evidence_authority",
            _text(self.evidence_authority, "evidence_authority"),
        )
        object.__setattr__(self, "verdict", _text(self.verdict, "verdict"))
        object.__setattr__(
            self,
            "operation_status",
            _text(self.operation_status, "operation_status"),
        )
        if not isinstance(self.context, ClientRequestContext):
            raise LogicPlatformClientError(
                "context must be a ClientRequestContext"
            )
        if not isinstance(self.simulated, bool):
            raise LogicPlatformClientError("simulated must be a bool")
        object.__setattr__(
            self, "translation_ids", tuple(self.translation_ids or ())
        )
        object.__setattr__(
            self, "artifact_ids", tuple(self.artifact_ids or ())
        )
        if not self.content_digest:
            payload = {
                "artifact_ids": list(self.artifact_ids),
                "evidence_authority": self.evidence_authority,
                "evidence_kind": self.evidence_kind,
                "operation": self.operation,
                "operation_status": self.operation_status,
                "provider_id": self.provider_id,
                "request_id": self.request_id,
                "simulated": self.simulated,
                "translation_ids": list(self.translation_ids),
                "verdict": self.verdict,
            }
            object.__setattr__(
                self,
                "content_digest",
                _digest_prefaced(repr(sorted(payload.items()))),
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_ids": list(self.artifact_ids),
            "content_digest": self.content_digest,
            "context": self.context.to_dict(),
            "evidence_authority": self.evidence_authority,
            "evidence_kind": self.evidence_kind,
            "operation": self.operation,
            "operation_status": self.operation_status,
            "provider_id": self.provider_id,
            "receipt_id": self.receipt_id,
            "request_id": self.request_id,
            "schema_version": self.schema_version,
            "simulated": self.simulated,
            "translation_ids": list(self.translation_ids),
            "verdict": self.verdict,
        }


@dataclass(frozen=True, slots=True)
class ClientCounterexampleView:
    """Public-safe counterexample projection for supervisor consumers."""

    counterexample_id: str
    kind: str
    summary: str
    context: ClientRequestContext
    semantic_id: str = ""
    property_id: str = ""
    authority: str = "advisory"
    payload: Mapping[str, Any] = field(default_factory=dict)
    redacted: bool = True
    schema_version: str = CLIENT_COUNTEREXAMPLE_VIEW_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "counterexample_id",
            _text(self.counterexample_id, "counterexample_id"),
        )
        object.__setattr__(self, "kind", _text(self.kind, "kind"))
        object.__setattr__(
            self, "summary", _text(self.summary, "summary", optional=True)
        )
        if not isinstance(self.context, ClientRequestContext):
            raise LogicPlatformClientError(
                "context must be a ClientRequestContext"
            )
        object.__setattr__(
            self,
            "semantic_id",
            _text(self.semantic_id, "semantic_id", optional=True),
        )
        object.__setattr__(
            self,
            "property_id",
            _text(self.property_id, "property_id", optional=True),
        )
        object.__setattr__(
            self,
            "authority",
            _authority_token(self.authority or "advisory"),
        )
        if not isinstance(self.payload, Mapping):
            raise LogicPlatformClientError("payload must be a mapping")
        object.__setattr__(
            self, "payload", MappingProxyType(dict(self.payload))
        )
        if not isinstance(self.redacted, bool):
            raise LogicPlatformClientError("redacted must be a bool")
        # Never retain private markers in the public view.
        for key in self.payload:
            lowered = str(key).lower()
            if any(
                marker in lowered
                for marker in (
                    "hidden_witness",
                    "credential",
                    "api_key",
                    "raw_output",
                    "prover_output",
                    "source_code",
                )
            ):
                raise LogicPlatformClientAdmissionError(
                    f"counterexample view rejects private key {key!r}"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "authority": self.authority,
            "context": self.context.to_dict(),
            "counterexample_id": self.counterexample_id,
            "kind": self.kind,
            "payload": dict(self.payload),
            "property_id": self.property_id,
            "redacted": self.redacted,
            "schema_version": self.schema_version,
            "semantic_id": self.semantic_id,
            "summary": self.summary,
        }


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class SupervisorLogicPlatformClient:
    """Lazy supervisor-side client for the datasets logic platform.

    Interface: ``SupervisorLogicPlatformClient@1``.

    Construction and import never load ``ipfs_datasets_py``.  The first
    :meth:`handshake` (or any operation that needs datasets) loads the
    package-neutral manifest and sealed catalog.  Typed provider dispatch uses
    an injectable protocol-v2 provider; when absent, only non-provider
    operations (handshake, catalog, formalize, slice/obligation/plan, cache,
    receipt/counterexample projection) succeed.
    """

    interface: Final = SUPERVISOR_LOGIC_PLATFORM_CLIENT_INTERFACE
    version: Final = SUPERVISOR_LOGIC_PLATFORM_CLIENT_VERSION
    schema_version: Final = SUPERVISOR_LOGIC_PLATFORM_CLIENT_SCHEMA
    task_id: Final = SUPERVISOR_LOGIC_PLATFORM_CLIENT_TASK_ID
    goal_id: Final = SUPERVISOR_LOGIC_PLATFORM_CLIENT_GOAL_ID

    def __init__(
        self,
        *,
        adapter: SupervisorCanonicalLogicAdapter | None = None,
        provider: Any | None = None,
        provider_id: str = "",
        provider_version: str = "",
        require_handshake: bool = True,
        module_importer: Callable[[str], Any] | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._adapter = adapter
        self._provider = provider
        self._provider_id = (
            _text(provider_id, "provider_id", optional=True) or "unbound"
        )
        self._provider_version = (
            _text(provider_version, "provider_version", optional=True) or "0"
        )
        self._require_handshake = bool(require_handshake)
        self._import = module_importer or importlib.import_module
        self._clock = clock or time.time
        self._handshake_result: Any | None = None
        self._manifest: Any | None = None
        self._lock = threading.RLock()
        self._module_cache: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Lazy loaders
    # ------------------------------------------------------------------

    def _load(self, module_name: str) -> Any:
        cached = self._module_cache.get(module_name)
        if cached is not None:
            return cached
        module = self._import(module_name)
        self._module_cache[module_name] = module
        return module

    def _get_adapter(self) -> SupervisorCanonicalLogicAdapter:
        if self._adapter is None:
            self._adapter = get_canonical_logic_adapter()
        return self._adapter

    def _ensure_handshake(self) -> None:
        if not self._require_handshake:
            return
        if self._handshake_result is None or not getattr(
            self._handshake_result, "compatible", False
        ):
            raise LogicPlatformClientHandshakeError(
                "handshake() must succeed before this operation; "
                "pass require_handshake=False only for pure offline probes"
            )

    def _resolve_context(
        self,
        context: ClientRequestContext | Mapping[str, Any] | None,
        **overrides: Any,
    ) -> ClientRequestContext:
        if context is None:
            base: dict[str, Any] = {
                "task_id": overrides.pop("task_id", "task:unbound"),
                "tree_id": overrides.pop("tree_id", "tree:unbound"),
                "policy_id": overrides.pop("policy_id", "policy:unbound"),
            }
        elif isinstance(context, ClientRequestContext):
            base = context.to_dict()
        else:
            base = dict(_require_mapping(context, "context"))
        for key, value in overrides.items():
            if value is not None:
                base[key] = value
        return ClientRequestContext.from_dict(base)

    def _require_provider(self) -> Any:
        if self._provider is None:
            raise LogicPlatformClientError(
                "no protocol-v2 provider bound; pass provider= to the client "
                "constructor for capability/invoke/reconstruct/verify"
            )
        return self._provider

    def _dispatch_operation(self, operation: str, request: Any) -> Any:
        provider = self._require_provider()
        method = getattr(provider, operation, None)
        if not callable(method):
            raise LogicPlatformClientError(
                f"provider does not implement operation {operation!r}"
            )
        return method(request)

    def _normalize_response(
        self,
        *,
        operation: str,
        request: Any,
        raw: Any,
    ) -> Any:
        response_mod = self._load(_RESPONSE_V2_MODULE)
        if isinstance(raw, response_mod.ProviderResponseV2):
            return response_mod.admit_provider_response_v2(raw)
        if isinstance(raw, Mapping):
            payload = dict(raw)
            payload.setdefault("request_id", getattr(request, "request_id", ""))
            payload.setdefault("operation", operation)
            payload.setdefault("provider_id", self._provider_id)
            payload.setdefault("provider_version", self._provider_version)
            if "operation_status" not in payload and "ok" in payload:
                # Lift LogicProvider@1 style into @2 with untrusted defaults.
                if payload.get("ok"):
                    return response_mod.ProviderResponseV2.succeeded(
                        request_id=str(payload["request_id"]),
                        operation=operation,
                        provider_id=str(
                            payload.get("provider_id") or self._provider_id
                        ),
                        provider_version=str(
                            payload.get("provider_version")
                            or self._provider_version
                        ),
                        metadata={
                            k: v
                            for k, v in payload.items()
                            if k
                            not in {
                                "ok",
                                "result",
                                "error",
                                "request_id",
                                "operation",
                                "provider_id",
                                "provider_version",
                            }
                        },
                    )
                error = payload.get("error") or {}
                return response_mod.ProviderResponseV2.failed(
                    request_id=str(payload["request_id"]),
                    operation=operation,
                    provider_id=str(
                        payload.get("provider_id") or self._provider_id
                    ),
                    provider_version=str(
                        payload.get("provider_version")
                        or self._provider_version
                    ),
                    code=str(
                        getattr(error, "get", lambda *_: "provider_error")(
                            "code", "provider_error"
                        )
                        if isinstance(error, Mapping)
                        else getattr(error, "code", "provider_error")
                    ),
                    message=str(
                        error.get("message", "provider failure")
                        if isinstance(error, Mapping)
                        else getattr(error, "message", "provider failure")
                    ),
                )
            return response_mod.admit_provider_response_v2(payload)
        raise LogicPlatformClientError(
            f"provider returned unsupported response type for {operation}: "
            f"{type(raw).__name__}"
        )

    def _resource_budget_from_context(
        self, context: ClientRequestContext
    ) -> Any:
        provider_mod = self._load("ipfs_datasets_py.logic.backends.provider")
        budget = dict(context.budget)
        kwargs: dict[str, Any] = {"network_allowed": context.network_allowed}
        mapping = (
            ("wall_time_ms", "wall_time_ms"),
            ("timeout_ms", "wall_time_ms"),
            ("cpu_time_ms", "cpu_time_ms"),
            ("memory_bytes", "memory_bytes"),
            ("max_memory_bytes", "memory_bytes"),
            ("disk_bytes", "disk_bytes"),
            ("max_output_bytes", "max_output_bytes"),
            ("model_token_limit", "model_token_limit"),
            ("max_processes", "max_processes"),
            ("max_premises", "max_premises"),
            ("provider_quota", "provider_quota"),
        )
        for src, dst in mapping:
            if src in budget and dst not in kwargs:
                try:
                    kwargs[dst] = int(budget[src])
                except (TypeError, ValueError):
                    continue
        return provider_mod.ProviderResourceBudget(**kwargs)

    def _cancellation_from_context(
        self, context: ClientRequestContext
    ) -> Any | None:
        if context.cancellation is None:
            return None
        provider_mod = self._load("ipfs_datasets_py.logic.backends.provider")
        payload = dict(context.cancellation)
        return provider_mod.ProviderCancellation(
            cancellation_id=str(
                payload.get("cancellation_id")
                or f"request:{context.correlation_id}"
            ),
            cancelled=bool(payload.get("cancelled", False)),
            reason=str(payload.get("reason") or ""),
        )

    # ------------------------------------------------------------------
    # Handshake / catalog
    # ------------------------------------------------------------------

    def handshake(
        self,
        requirements: Any | Mapping[str, Any] | None = None,
        *,
        manifest: Any | None = None,
    ) -> Any:
        """Negotiate platform compatibility (LogicPlatformManifest@1).

        Default requirements admit the installed wheel without Git or sibling
        layout.  Typed incompatibilities are returned, not raised.
        """

        manifest_mod = self._load(_MANIFEST_MODULE)
        if manifest is None:
            manifest = manifest_mod.build_logic_platform_manifest()
        if requirements is None:
            req = manifest_mod.HandshakeRequirements(
                required_adapter_versions=(
                    SUPERVISOR_CANONICAL_LOGIC_ADAPTER_INTERFACE,
                    SUPERVISOR_LOGIC_PLATFORM_CLIENT_INTERFACE,
                )
            )
        elif isinstance(requirements, manifest_mod.HandshakeRequirements):
            req = requirements
        else:
            payload = dict(_require_mapping(requirements, "requirements"))
            req = manifest_mod.HandshakeRequirements(**payload)
        result = manifest_mod.handshake(req, manifest=manifest)
        with self._lock:
            self._handshake_result = result
            self._manifest = result.manifest
        return result

    @property
    def last_handshake(self) -> Any | None:
        return self._handshake_result

    @property
    def manifest(self) -> Any | None:
        return self._manifest

    def catalog(self, *, require_handshake: bool | None = None) -> Any:
        """Return the sealed CanonicalLogicCatalogSnapshot@1."""

        if require_handshake is None:
            require_handshake = self._require_handshake
        if require_handshake:
            self._ensure_handshake()
        catalog_mod = self._load(_CATALOG_MODULE)
        return catalog_mod.DEFAULT_CANONICAL_CATALOG_SNAPSHOT

    def catalog_root(self) -> str:
        """Return the sealed catalog content root (CIDv1)."""

        snapshot = self.catalog(require_handshake=False)
        return str(snapshot.content_root)

    def catalog_digest(self) -> str:
        """Return the sealed catalog content digest."""

        snapshot = self.catalog(require_handshake=False)
        return str(snapshot.content_digest)

    # ------------------------------------------------------------------
    # Formalization / slice / obligation / plan
    # ------------------------------------------------------------------

    def formalize(
        self,
        *,
        artifact_id: str = "",
        sample_id: str = "",
        domain: str,
        document_id: str,
        source_digest: str,
        expression_id: str,
        expression_digest: str,
        family: Any,
        profile: Any,
        view: Any = "source",
        notation: Any = "canonical_text",
        slices: Sequence[Any] = (),
        status: str = "ok",
        context: ClientRequestContext | Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        **context_overrides: Any,
    ) -> Any:
        """Construct a FormalizationArtifact@3 bound to source and expression."""

        self._ensure_handshake()
        ctx = self._resolve_context(context, **context_overrides)
        artifacts = self._load(_ARTIFACTS_V3_MODULE)
        slice_items: list[Any] = []
        for item in slices:
            if isinstance(item, artifacts.DomainLogicSliceV2):
                slice_items.append(item)
            else:
                slice_items.append(
                    artifacts.DomainLogicSliceV2.from_dict(
                        _require_mapping(item, "slice")
                    )
                )
        meta = dict(metadata or {})
        meta.setdefault("task_id", ctx.task_id)
        meta.setdefault("tree_id", ctx.tree_id)
        meta.setdefault("policy_id", ctx.policy_id)
        meta.setdefault("correlation_id", ctx.correlation_id)
        return artifacts.FormalizationArtifactV3(
            artifact_id=artifact_id or _new_id("artifact"),
            sample_id=sample_id or _new_id("sample"),
            domain=domain,
            document_id=document_id,
            source_digest=_bare_digest(source_digest, "source_digest"),
            expression_id=expression_id,
            expression_digest=_bare_digest(
                expression_digest, "expression_digest"
            ),
            family=family,
            profile=profile,
            view=view,
            notation=notation,
            status=status,
            slices=tuple(slice_items),
            metadata=meta,
        )

    def create_slice(
        self,
        *,
        slice_id: str = "",
        domain: str,
        document_id: str,
        source_digest: str,
        expression_id: str,
        expression_digest: str,
        family: Any,
        profile: Any,
        property: Any,
        view: Any = "source",
        notation: Any = "canonical_text",
        status: str = "admitted",
        features: Sequence[str] = (),
        assumption_ids: Sequence[str] = (),
        formalization_artifact_id: str = "",
        context: ClientRequestContext | Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        **context_overrides: Any,
    ) -> Any:
        """Construct an admitted DomainLogicSlice@2."""

        self._ensure_handshake()
        ctx = self._resolve_context(context, **context_overrides)
        artifacts = self._load(_ARTIFACTS_V3_MODULE)
        meta = dict(metadata or {})
        meta.setdefault("task_id", ctx.task_id)
        meta.setdefault("tree_id", ctx.tree_id)
        meta.setdefault("policy_id", ctx.policy_id)
        meta.setdefault("correlation_id", ctx.correlation_id)
        return artifacts.DomainLogicSliceV2(
            slice_id=slice_id or _new_id("slice"),
            domain=domain,
            document_id=document_id,
            source_digest=_bare_digest(source_digest, "source_digest"),
            expression_id=expression_id,
            expression_digest=_bare_digest(
                expression_digest, "expression_digest"
            ),
            family=family,
            profile=profile,
            property=property,
            view=view,
            notation=notation,
            status=status,
            features=tuple(features),
            assumption_ids=tuple(assumption_ids),
            formalization_artifact_id=formalization_artifact_id,
            metadata=meta,
        )

    def create_obligation(
        self,
        *,
        obligation_id: str = "",
        statement: str,
        document_id: str,
        source_digest: str,
        expression_id: str,
        expression_digest: str,
        family: Any,
        profile: Any,
        property: Any,
        view: Any = "source",
        notation: Any = "canonical_text",
        encoding: Any,
        evidence_kind: Any | None = None,
        bounds: Any,
        authority_ceiling: Any | None = None,
        features: Sequence[str] = (),
        assumption_ids: Sequence[str] = (),
        slice_id: str = "",
        slice_digest: str = "",
        context: ClientRequestContext | Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        **context_overrides: Any,
    ) -> Any:
        """Construct a LogicObligation@2 (fail closed on authority overclaim)."""

        self._ensure_handshake()
        ctx = self._resolve_context(
            context,
            evidence_kind=evidence_kind,
            authority_ceiling=authority_ceiling,
            **context_overrides,
        )
        # Re-check after context resolution (context may have filled defaults).
        check_authority_overclaim(ctx.authority_ceiling, ctx.evidence_kind)
        requests = self._load(_REQUESTS_V2_MODULE)
        namespaces = self._load(_NAMESPACES_MODULE)
        evidence = evidence_kind or namespaces.evidence_id(ctx.evidence_kind)
        ceiling = authority_ceiling or ctx.authority_ceiling
        meta = dict(metadata or {})
        meta.setdefault("task_id", ctx.task_id)
        meta.setdefault("tree_id", ctx.tree_id)
        meta.setdefault("policy_id", ctx.policy_id)
        meta.setdefault("plan_id", ctx.plan_id)
        meta.setdefault("correlation_id", ctx.correlation_id)
        try:
            return requests.LogicObligationV2(
                obligation_id=obligation_id or _new_id("obl"),
                statement=statement,
                document_id=document_id,
                source_digest=_bare_digest(source_digest, "source_digest"),
                expression_id=expression_id,
                expression_digest=_bare_digest(
                    expression_digest, "expression_digest"
                ),
                family=family,
                profile=profile,
                property=property,
                view=view,
                notation=notation,
                encoding=encoding,
                evidence_kind=evidence,
                bounds=bounds,
                authority_ceiling=ceiling,
                features=tuple(features),
                assumption_ids=tuple(assumption_ids),
                slice_id=slice_id,
                slice_digest=(
                    _bare_digest(slice_digest, "slice_digest")
                    if slice_digest
                    else ""
                ),
                metadata=meta,
            )
        except Exception as error:
            # Surface datasets authority overclaim as the client gate.
            if type(error).__name__ == "AuthorityOverclaimError":
                raise LogicPlatformClientAuthorityError(str(error)) from error
            raise

    def create_plan(
        self,
        *,
        plan_id: str = "",
        formal_goal_id: str,
        graph_id: str,
        tree_id: str | None = None,
        candidates: Sequence[Any] = (),
        step_order: Sequence[str] = (),
        provider_ids: Sequence[str] = (),
        bounds: Any | None = None,
        context: ClientRequestContext | Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        **context_overrides: Any,
    ) -> Any:
        """Construct a draft GoalDirectedProofPlan@1 (proposal only)."""

        self._ensure_handshake()
        ctx = self._resolve_context(
            context, tree_id=tree_id, **context_overrides
        )
        tactician = self._load(_TACTICIAN_MODULE)
        meta = dict(metadata or {})
        meta.setdefault("task_id", ctx.task_id)
        meta.setdefault("policy_id", ctx.policy_id)
        meta.setdefault("correlation_id", ctx.correlation_id)
        resolved_bounds = bounds
        if resolved_bounds is None:
            budget = dict(ctx.budget)
            resolved_bounds = tactician.ResourceBounds(
                wall_time_ms=int(budget.get("timeout_ms") or budget.get("wall_time_ms") or 0),
                memory_bytes=int(
                    budget.get("max_memory_bytes")
                    or budget.get("memory_bytes")
                    or 0
                ),
                max_steps=int(budget.get("max_steps") or 0),
                network_allowed=ctx.network_allowed,
            )
        plan = tactician.GoalDirectedProofPlan(
            plan_id=plan_id or ctx.plan_id or _new_id("plan"),
            formal_goal_id=formal_goal_id,
            graph_id=graph_id,
            tree_id=tree_id or ctx.tree_id,
            candidates=tuple(candidates),
            step_order=tuple(step_order),
            status=tactician.PlanStatus.DRAFT,
            bounds=resolved_bounds,
            provider_ids=tuple(provider_ids),
            authority=tactician.AuthorityCeiling.CANDIDATE,
            proof_claimed=False,
            completion_claimed=False,
            metadata=meta,
        )
        return plan

    def create_backend_request(
        self,
        *,
        request_id: str = "",
        obligation: Any | None = None,
        obligation_id: str = "",
        obligation_digest: str = "",
        document_id: str = "",
        source_digest: str = "",
        expression_id: str = "",
        expression_digest: str = "",
        family: Any = None,
        profile: Any = None,
        property: Any = None,
        view: Any = "source",
        notation: Any = "canonical_text",
        encoding: Any = None,
        evidence_kind: Any | None = None,
        bounds: Any = None,
        authority_ceiling: Any | None = None,
        features: Sequence[str] = (),
        assumption_ids: Sequence[str] = (),
        slice_id: str = "",
        slice_digest: str = "",
        requested_provider: Any | None = None,
        context: ClientRequestContext | Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        **context_overrides: Any,
    ) -> Any:
        """Construct a BackendRequest@2, optionally from a LogicObligation@2."""

        self._ensure_handshake()
        requests = self._load(_REQUESTS_V2_MODULE)
        namespaces = self._load(_NAMESPACES_MODULE)
        if obligation is not None:
            obligation_id = obligation_id or obligation.obligation_id
            obligation_digest = (
                obligation_digest or obligation.content_digest
            )
            document_id = document_id or obligation.document_id
            source_digest = source_digest or obligation.source_digest
            expression_id = expression_id or obligation.expression_id
            expression_digest = (
                expression_digest or obligation.expression_digest
            )
            family = family if family is not None else obligation.family
            profile = profile if profile is not None else obligation.profile
            property = (
                property if property is not None else obligation.property
            )
            view = view if view is not None else obligation.view
            notation = (
                notation if notation is not None else obligation.notation
            )
            encoding = (
                encoding if encoding is not None else obligation.encoding
            )
            evidence_kind = (
                evidence_kind
                if evidence_kind is not None
                else obligation.evidence_kind
            )
            bounds = bounds if bounds is not None else obligation.bounds
            authority_ceiling = (
                authority_ceiling
                if authority_ceiling is not None
                else obligation.authority_ceiling
            )
            features = features or obligation.features
            assumption_ids = assumption_ids or obligation.assumption_ids
            slice_id = slice_id or obligation.slice_id
            slice_digest = slice_digest or obligation.slice_digest

        ctx = self._resolve_context(
            context,
            evidence_kind=_evidence_token(
                evidence_kind or "candidate"
            ),
            authority_ceiling=_authority_token(
                authority_ceiling or "advisory"
            ),
            **context_overrides,
        )
        check_authority_overclaim(ctx.authority_ceiling, ctx.evidence_kind)
        if bounds is None:
            raise LogicPlatformClientAdmissionError(
                "BackendRequest@2 requires finite bounds"
            )
        evidence = evidence_kind or namespaces.evidence_id(ctx.evidence_kind)
        ceiling = authority_ceiling or ctx.authority_ceiling
        meta = dict(metadata or {})
        meta.setdefault("task_id", ctx.task_id)
        meta.setdefault("tree_id", ctx.tree_id)
        meta.setdefault("policy_id", ctx.policy_id)
        meta.setdefault("plan_id", ctx.plan_id)
        meta.setdefault("correlation_id", ctx.correlation_id)
        resolved_obligation_id = obligation_id or _new_id("obl")
        resolved_obligation_digest = (
            _bare_digest(obligation_digest, "obligation_digest")
            if obligation_digest
            else _digest(resolved_obligation_id)
        )
        try:
            return requests.BackendRequestV2(
                request_id=request_id or _new_id("req"),
                obligation_id=resolved_obligation_id,
                obligation_digest=resolved_obligation_digest,
                document_id=document_id,
                source_digest=_bare_digest(source_digest, "source_digest"),
                expression_id=expression_id,
                expression_digest=_bare_digest(
                    expression_digest, "expression_digest"
                ),
                family=family,
                profile=profile,
                property=property,
                view=view,
                notation=notation,
                encoding=encoding,
                evidence_kind=evidence,
                bounds=bounds,
                authority_ceiling=ceiling,
                features=tuple(features),
                assumption_ids=tuple(assumption_ids),
                slice_id=slice_id,
                slice_digest=(
                    _bare_digest(slice_digest, "slice_digest")
                    if slice_digest
                    else ""
                ),
                requested_provider=requested_provider,
                metadata=meta,
            )
        except Exception as error:
            if type(error).__name__ == "AuthorityOverclaimError":
                raise LogicPlatformClientAuthorityError(str(error)) from error
            raise

    # ------------------------------------------------------------------
    # Capability discovery / typed invocation
    # ------------------------------------------------------------------

    def discover_capabilities(
        self,
        *,
        provider_id: str = "",
        feature_query: Sequence[str] = (),
        include_versions: bool = False,
        context: ClientRequestContext | Mapping[str, Any] | None = None,
        **context_overrides: Any,
    ) -> ClientInvocationResult:
        """Capability discovery (non-executable; never mints proof authority)."""

        self._ensure_handshake()
        ctx = self._resolve_context(context, **context_overrides)
        protocol = self._load(_PROTOCOL_V2_MODULE)
        request = protocol.CapabilityRequestV2(
            request_id=_new_id("cap"),
            provider_id=provider_id or self._provider_id,
            feature_query=tuple(feature_query),
            include_versions=include_versions,
            resource_budget=self._resource_budget_from_context(ctx),
            cancellation=self._cancellation_from_context(ctx),
            network_allowed=ctx.network_allowed,
            deadline_unix_ms=ctx.deadline_unix_ms,
            metadata={
                "task_id": ctx.task_id,
                "tree_id": ctx.tree_id,
                "policy_id": ctx.policy_id,
                "plan_id": ctx.plan_id,
                "correlation_id": ctx.correlation_id,
            },
        )
        raw = self._dispatch_operation("capability", request)
        response = self._normalize_response(
            operation="capability", request=request, raw=raw
        )
        return ClientInvocationResult(
            operation="capability",
            context=ctx,
            response=response,
            request=request,
        )

    def invoke(
        self,
        operation: str,
        *,
        bounds: Any | None = None,
        backend_request: Any | None = None,
        statement: str = "",
        goal_digest: str = "",
        mode: str | None = None,
        source_encoding: str = "",
        target_encoding: str = "",
        source_artifact_digest: str = "",
        preservation_claim: str = "",
        candidate_digest: str = "",
        candidate_artifact_id: str = "",
        kernel_id: str = "",
        evidence_digest: str = "",
        evidence_kind: str = "",
        verifier_id: str = "",
        statement_digest: str = "",
        subject_id: str = "",
        attestation_profile: str = "",
        context: ClientRequestContext | Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        **context_overrides: Any,
    ) -> ClientInvocationResult:
        """Typed LogicProviderProtocol@2 invocation for a closed operation."""

        self._ensure_handshake()
        op = _text(operation, "operation").lower()
        protocol = self._load(_PROTOCOL_V2_MODULE)
        if op not in protocol.PROTOCOL_V2_OPERATIONS:
            raise LogicPlatformClientAdmissionError(
                f"unknown protocol operation {op!r}; admitted: "
                + ", ".join(sorted(protocol.PROTOCOL_V2_OPERATIONS))
            )
        ctx = self._resolve_context(context, **context_overrides)
        check_authority_overclaim(ctx.authority_ceiling, ctx.evidence_kind)
        meta = {
            "task_id": ctx.task_id,
            "tree_id": ctx.tree_id,
            "policy_id": ctx.policy_id,
            "plan_id": ctx.plan_id,
            "correlation_id": ctx.correlation_id,
            "evidence_kind": ctx.evidence_kind,
            "authority_ceiling": ctx.authority_ceiling,
        }
        if metadata:
            meta.update(dict(metadata))

        if op == "capability":
            return self.discover_capabilities(
                context=ctx,
                provider_id=self._provider_id,
            )

        if backend_request is None:
            raise LogicPlatformClientAdmissionError(
                f"{op} requires an admitted BackendRequest@2"
            )
        if bounds is None:
            bounds = getattr(backend_request, "bounds", None)
        if bounds is None:
            raise LogicPlatformClientAdmissionError(
                f"{op} requires positive finite bounds"
            )

        common = {
            "bounds": bounds,
            "backend_request": backend_request,
            "request_id": _new_id(op),
            "resource_budget": self._resource_budget_from_context(ctx),
            "cancellation": self._cancellation_from_context(ctx),
            "network_allowed": ctx.network_allowed,
            "deadline_unix_ms": ctx.deadline_unix_ms,
            "metadata": meta,
        }

        if op in {"prove", "check"}:
            request = protocol.ProveCheckRequestV2(
                mode=mode or op,
                statement=statement,
                goal_digest=goal_digest,
                **common,
            )
            dispatch_op = op
        elif op == "translate":
            request = protocol.TranslationRequestV2(
                source_encoding=source_encoding or "source",
                target_encoding=target_encoding or "target",
                source_artifact_digest=source_artifact_digest,
                preservation_claim=preservation_claim,
                **common,
            )
            dispatch_op = "translate"
        elif op == "reconstruct":
            digest = candidate_digest or _digest(
                candidate_artifact_id or "candidate"
            )
            if digest.startswith("sha256:"):
                digest = digest[len("sha256:") :]
            request = protocol.ReconstructRequestV2(
                candidate_digest=digest,
                candidate_artifact_id=candidate_artifact_id,
                kernel_id=kernel_id,
                **common,
            )
            dispatch_op = "reconstruct"
        elif op == "verify":
            digest = evidence_digest or _digest("evidence")
            if digest.startswith("sha256:"):
                digest = digest[len("sha256:") :]
            request = protocol.VerifyRequestV2(
                evidence_digest=digest,
                evidence_kind=evidence_kind or ctx.evidence_kind,
                verifier_id=verifier_id or self._provider_id,
                **common,
            )
            dispatch_op = "verify"
        elif op == "attest":
            digest = statement_digest or _digest(statement or "statement")
            if digest.startswith("sha256:"):
                digest = digest[len("sha256:") :]
            request = protocol.AttestRequestV2(
                statement_digest=digest,
                subject_id=subject_id or ctx.task_id,
                attestation_profile=attestation_profile,
                **common,
            )
            dispatch_op = "attest"
        else:
            raise LogicPlatformClientAdmissionError(
                f"operation {op!r} is not dispatchable via invoke()"
            )

        # Re-admit through protocol gate so free-form bypass cannot sneak in.
        admitted = protocol.admit_provider_request_v2(request)
        raw = self._dispatch_operation(dispatch_op, admitted)
        response = self._normalize_response(
            operation=dispatch_op, request=admitted, raw=raw
        )
        return ClientInvocationResult(
            operation=dispatch_op,
            context=ctx,
            response=response,
            request=admitted,
        )

    def reconstruct(
        self,
        *,
        bounds: Any,
        backend_request: Any,
        candidate_digest: str = "",
        candidate_artifact_id: str = "",
        kernel_id: str = "",
        context: ClientRequestContext | Mapping[str, Any] | None = None,
        **context_overrides: Any,
    ) -> ClientInvocationResult:
        """Typed reconstruction under positive finite bounds."""

        return self.invoke(
            "reconstruct",
            bounds=bounds,
            backend_request=backend_request,
            candidate_digest=candidate_digest,
            candidate_artifact_id=candidate_artifact_id,
            kernel_id=kernel_id,
            context=context,
            **context_overrides,
        )

    def verify(
        self,
        *,
        bounds: Any,
        backend_request: Any,
        evidence_digest: str = "",
        evidence_kind: str = "",
        verifier_id: str = "",
        context: ClientRequestContext | Mapping[str, Any] | None = None,
        **context_overrides: Any,
    ) -> ClientInvocationResult:
        """Typed independent verification under positive finite bounds."""

        return self.invoke(
            "verify",
            bounds=bounds,
            backend_request=backend_request,
            evidence_digest=evidence_digest,
            evidence_kind=evidence_kind,
            verifier_id=verifier_id,
            context=context,
            **context_overrides,
        )

    # ------------------------------------------------------------------
    # Receipts / counterexamples
    # ------------------------------------------------------------------

    def project_receipt(
        self,
        result: ClientInvocationResult | Mapping[str, Any] | Any,
        *,
        context: ClientRequestContext | Mapping[str, Any] | None = None,
        simulated: bool = False,
        **context_overrides: Any,
    ) -> ClientReceiptView:
        """Project a provider response into a supervisor receipt view.

        Success never upgrades authority.  Simulated evidence is marked and
        remains non-authoritative for completion/merge (LPC-111 admission).
        """

        self._ensure_handshake()
        if isinstance(result, ClientInvocationResult):
            ctx = result.context
            response = result.response
            operation = result.operation
        else:
            ctx = self._resolve_context(context, **context_overrides)
            response = result
            operation = _token(getattr(response, "operation", "unknown"))

        if simulated:
            # Simulated evidence cannot influence completion (plan §8 item 9).
            pass

        request_id = str(getattr(response, "request_id", "") or _new_id("req"))
        provider_id = str(
            getattr(response, "provider_id", "") or self._provider_id
        )
        evidence_kind = _token(
            getattr(response, "evidence_kind", ctx.evidence_kind)
        )
        evidence_authority = _token(
            getattr(response, "evidence_authority", "advisory")
        )
        verdict = _token(getattr(response, "verdict", "unknown"))
        operation_status = _token(
            getattr(response, "operation_status", "unknown")
        )
        translations = getattr(response, "translations", ()) or ()
        translation_ids = tuple(
            str(getattr(item, "translation_id", item))
            for item in translations
        )
        artifacts = getattr(response, "artifacts", ()) or ()
        artifact_ids = tuple(
            str(getattr(item, "artifact_id", item)) for item in artifacts
        )
        return ClientReceiptView(
            receipt_id=_new_id("receipt"),
            request_id=request_id,
            operation=operation,
            provider_id=provider_id,
            evidence_kind=evidence_kind,
            evidence_authority=evidence_authority,
            verdict=verdict,
            operation_status=operation_status,
            context=ctx if isinstance(ctx, ClientRequestContext) else self._resolve_context(ctx),
            simulated=bool(simulated),
            translation_ids=translation_ids,
            artifact_ids=artifact_ids,
        )

    def project_counterexample(
        self,
        raw: Mapping[str, Any] | Any,
        *,
        context: ClientRequestContext | Mapping[str, Any] | None = None,
        kind: str = "generic_failure",
        summary: str = "",
        property_id: str = "",
        **context_overrides: Any,
    ) -> ClientCounterexampleView:
        """Project raw failure material into a public-safe counterexample view.

        Private keys (credentials, hidden witnesses, raw prover output, source
        blobs) are stripped.  Authority remains advisory unless the caller
        supplies a weaker ceiling.
        """

        self._ensure_handshake()
        ctx = self._resolve_context(context, **context_overrides)
        if hasattr(raw, "to_dict"):
            payload = dict(raw.to_dict())
        elif isinstance(raw, Mapping):
            payload = dict(raw)
        else:
            payload = {"value": str(raw)}

        private_markers = (
            "hidden_witness",
            "private_witness",
            "private_inputs",
            "credential",
            "access_token",
            "api_key",
            "raw_output",
            "prover_output",
            "stdout",
            "stderr",
            "source_excerpt",
            "source_code",
            "source_text",
        )
        public_payload: dict[str, Any] = {}
        for key, value in payload.items():
            lowered = str(key).lower()
            if any(marker in lowered for marker in private_markers):
                continue
            if isinstance(value, (str, int, float, bool)) or value is None:
                public_payload[str(key)] = value
            elif isinstance(value, (list, tuple)):
                public_payload[str(key)] = [
                    item
                    for item in value
                    if isinstance(item, (str, int, float, bool)) or item is None
                ]
            elif isinstance(value, Mapping):
                # Keep one level of public scalars only.
                nested = {
                    str(nk): nv
                    for nk, nv in value.items()
                    if (
                        (isinstance(nv, (str, int, float, bool)) or nv is None)
                        and not any(
                            m in str(nk).lower() for m in private_markers
                        )
                    )
                }
                if nested:
                    public_payload[str(key)] = nested

        counterexample_id = str(
            payload.get("counterexample_id")
            or payload.get("id")
            or _new_id("cex")
        )
        resolved_kind = str(payload.get("kind") or kind)
        resolved_summary = str(payload.get("summary") or summary or resolved_kind)
        semantic_id = str(
            payload.get("semantic_id") or payload.get("content_id") or ""
        )
        resolved_property = str(
            payload.get("property_id")
            or payload.get("violated_property")
            or property_id
            or ""
        )
        return ClientCounterexampleView(
            counterexample_id=counterexample_id,
            kind=resolved_kind,
            summary=resolved_summary,
            context=ctx,
            semantic_id=semantic_id,
            property_id=resolved_property,
            authority="advisory",
            payload=public_payload,
            redacted=True,
        )

    # ------------------------------------------------------------------
    # Cache key / freshness
    # ------------------------------------------------------------------

    def build_cache_key(
        self,
        *,
        source: Any,
        expression: Any,
        formalization: Any,
        slice: Any,
        obligation: Any,
        assumptions: Any = (),
        bounds: Any = None,
        translation: Any = None,
        provider: str,
        environment: Any,
        policy: Any,
        schema: Any,
        checker: str,
        network_policy: Any,
        evidence_kind: Any,
        authority_ceiling: Any,
        source_cid: str = "",
    ) -> Any:
        """Build a CanonicalProofCacheKey@1 (datasets-owned semantics).

        Candidate-as-kernel and digest validity are enforced by the datasets
        cache-key contract (``LogicEvidenceAuthority`` / ``LogicEvidenceKind``),
        not by the request-ceiling map used for BackendRequest@2.
        """

        self._ensure_handshake()
        cache_mod = self._load(_CACHE_KEY_MODULE)
        # Pre-check with the datasets helper so callers get a clear error
        # before full key construction when axes are incompatible.
        cache_mod.reject_candidate_as_kernel(evidence_kind, authority_ceiling)
        return cache_mod.CanonicalProofCacheKey.build(
            source=source,
            expression=expression,
            formalization=formalization,
            slice=slice,
            obligation=obligation,
            assumptions=assumptions,
            bounds=bounds,
            translation=translation,
            provider=provider,
            environment=environment,
            policy=policy,
            schema=schema,
            checker=checker,
            network_policy=network_policy,
            evidence_kind=evidence_kind,
            authority_ceiling=authority_ceiling,
            source_cid=source_cid,
        )

    def check_cache_freshness(
        self,
        key: Any | Mapping[str, Any],
        *,
        observed_environment: Any,
        entry_freshness: EvidenceFreshness | str = EvidenceFreshness.CURRENT,
        now_unix_ms: int | None = None,
        expires_unix_ms: int | None = None,
        simulated: bool = False,
    ) -> CacheFreshnessReport:
        """Admit a cache hit only when environment and freshness bind.

        Fail closed on:

        * environment mismatch (cross-environment hit)
        * stale / unknown freshness
        * expired TTL
        * simulated evidence
        """

        self._ensure_handshake()
        cache_mod = self._load(_CACHE_KEY_MODULE)
        if isinstance(key, cache_mod.CanonicalProofCacheKey):
            resolved = key
        else:
            resolved = cache_mod.CanonicalProofCacheKey.from_dict(
                _require_mapping(key, "key")
            )

        reasons: list[str] = []
        expected_env = resolved.environment
        observed = cache_mod.digest_of(observed_environment)

        freshness_token = _token(entry_freshness).lower()
        if freshness_token not in {
            EvidenceFreshness.CURRENT.value,
            EvidenceFreshness.STALE.value,
            EvidenceFreshness.UNKNOWN.value,
        }:
            reasons.append("unknown_freshness_label")
            freshness_token = EvidenceFreshness.UNKNOWN.value

        if observed != expected_env:
            reasons.append("environment_mismatch")
            try:
                cache_mod.reject_cross_environment  # type: ignore[attr-defined]
            except AttributeError:
                pass
            # Cross-environment hits are always rejected.
            try:
                raise cache_mod.CrossEnvironmentHitError(
                    "cache hit spans mismatched environment identities"
                )
            except cache_mod.CrossEnvironmentHitError:
                reasons.append("cross_environment_hit")

        if freshness_token == EvidenceFreshness.STALE.value:
            reasons.append("stale_entry")
        if freshness_token == EvidenceFreshness.UNKNOWN.value:
            reasons.append("unknown_freshness")
        if simulated:
            reasons.append("simulated_evidence")

        if expires_unix_ms is not None:
            now = (
                now_unix_ms
                if now_unix_ms is not None
                else int(self._clock() * 1000)
            )
            if now > int(expires_unix_ms):
                reasons.append("ttl_expired")

        fresh = not reasons and freshness_token == EvidenceFreshness.CURRENT.value
        report = CacheFreshnessReport(
            fresh=fresh,
            freshness=(
                EvidenceFreshness.CURRENT.value
                if fresh
                else (
                    EvidenceFreshness.STALE.value
                    if "stale_entry" in reasons or "ttl_expired" in reasons
                    else EvidenceFreshness.UNKNOWN.value
                    if reasons
                    else freshness_token
                )
            ),
            key_id=resolved.key_id,
            reasons=tuple(reasons),
            observed_environment=observed,
            expected_environment=expected_env,
        )
        if not report.fresh:
            # Callers may inspect the report; raising is reserved for strict mode.
            return report
        return report

    def require_cache_fresh(
        self,
        key: Any | Mapping[str, Any],
        *,
        observed_environment: Any,
        entry_freshness: EvidenceFreshness | str = EvidenceFreshness.CURRENT,
        now_unix_ms: int | None = None,
        expires_unix_ms: int | None = None,
        simulated: bool = False,
    ) -> CacheFreshnessReport:
        """Like :meth:`check_cache_freshness` but raises on stale/mismatched hits."""

        report = self.check_cache_freshness(
            key,
            observed_environment=observed_environment,
            entry_freshness=entry_freshness,
            now_unix_ms=now_unix_ms,
            expires_unix_ms=expires_unix_ms,
            simulated=simulated,
        )
        if not report.fresh:
            raise LogicPlatformClientFreshnessError(
                "cache entry failed freshness admission: "
                + ", ".join(report.reasons)
            )
        return report

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def supported_operations(self) -> tuple[str, ...]:
        return CLIENT_OPERATIONS

    def required_context_fields(self) -> tuple[str, ...]:
        return REQUIRED_CONTEXT_FIELDS

    def to_dict(self) -> dict[str, Any]:
        return {
            "compatible_adapter": SUPERVISOR_CANONICAL_LOGIC_ADAPTER_INTERFACE,
            "goal_id": self.goal_id,
            "handshake_compatible": bool(
                self._handshake_result is not None
                and getattr(self._handshake_result, "compatible", False)
            ),
            "interface": self.interface,
            "provider_id": self._provider_id,
            "provider_version": self._provider_version,
            "require_handshake": self._require_handshake,
            "schema_version": self.schema_version,
            "supported_operations": list(CLIENT_OPERATIONS),
            "task_id": self.task_id,
            "version": self.version,
        }


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_default_client: SupervisorLogicPlatformClient | None = None
_default_client_lock = threading.Lock()


def get_logic_platform_client(
    **kwargs: Any,
) -> SupervisorLogicPlatformClient:
    """Return a process-wide client, or a fresh one when overrides are supplied."""

    if kwargs:
        return SupervisorLogicPlatformClient(**kwargs)
    global _default_client
    client = _default_client
    if client is None:
        with _default_client_lock:
            client = _default_client
            if client is None:
                client = SupervisorLogicPlatformClient()
                _default_client = client
    return client


__all__ = [
    "CLIENT_CACHE_FRESHNESS_SCHEMA",
    "CLIENT_COUNTEREXAMPLE_VIEW_SCHEMA",
    "CLIENT_INVOCATION_RESULT_SCHEMA",
    "CLIENT_OPERATIONS",
    "CLIENT_RECEIPT_VIEW_SCHEMA",
    "CLIENT_REQUEST_CONTEXT_SCHEMA",
    "CacheFreshnessReport",
    "ClientCounterexampleView",
    "ClientInvocationResult",
    "ClientReceiptView",
    "ClientRequestContext",
    "LogicPlatformClientAdmissionError",
    "LogicPlatformClientAuthorityError",
    "LogicPlatformClientError",
    "LogicPlatformClientFreshnessError",
    "LogicPlatformClientHandshakeError",
    "REQUIRED_CONTEXT_FIELDS",
    "SUPERVISOR_LOGIC_PLATFORM_CLIENT_GOAL_ID",
    "SUPERVISOR_LOGIC_PLATFORM_CLIENT_INTERFACE",
    "SUPERVISOR_LOGIC_PLATFORM_CLIENT_SCHEMA",
    "SUPERVISOR_LOGIC_PLATFORM_CLIENT_TASK_ID",
    "SUPERVISOR_LOGIC_PLATFORM_CLIENT_VERSION",
    "SupervisorLogicPlatformClient",
    "check_authority_overclaim",
    "get_logic_platform_client",
]
