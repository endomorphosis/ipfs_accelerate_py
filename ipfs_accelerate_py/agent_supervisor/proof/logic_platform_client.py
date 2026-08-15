"""SupervisorLogicPlatformClient@1 — lazy supervisor client for the logic platform.

LPC-110 provides one handshake + typed invocation surface for catalog access,
formalization, slice/obligation/plan creation, capability discovery, provider
operations (capability/translate/prove/reconstruct/verify/attest), receipts,
counterexamples, and cache freshness.

Design invariants
-----------------
* Importing this module never imports ``ipfs_datasets_py``. Datasets packages
  load only for explicit operations.
* Every request binds task, tree, policy, plan, budget, network, cancellation,
  deadline, correlation, evidence, and authority axes.
* Callers cannot overclaim authority relative to the bound evidence kind.
* Semantic identities come from the catalog / adapter / platform contracts;
  this client never redefines them.
* Handshake is the first lazy step. Typed operations fail closed until a
  compatible handshake has been recorded.
* Transport or provider success never upgrades evidence authority.

Interface: ``SupervisorLogicPlatformClient@1``
"""

from __future__ import annotations

import hashlib
import importlib
import json
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Final

from .canonical_logic_adapter import (
    SupervisorCanonicalLogicAdapter,
    get_canonical_logic_adapter,
)
from .formal_verification_capabilities import ProofProviderOperation
from .formal_verification_contracts import ResourceBudget
from .formal_verification_provider import (
    CancellationToken,
    ProviderRequest,
    ProviderResponse,
)
from .logic_provider_contract import SupervisorLogicProviderFacade


# ---------------------------------------------------------------------------
# Interface / schema identities
# ---------------------------------------------------------------------------

SUPERVISOR_LOGIC_PLATFORM_CLIENT_INTERFACE: Final = (
    "SupervisorLogicPlatformClient@1"
)
SUPERVISOR_LOGIC_PLATFORM_CLIENT_VERSION: Final = "1.0.0"
CLIENT_SCHEMA_VERSION: Final = (
    "ipfs_accelerate_py/agent-supervisor/logic-platform-client@1"
)
CLIENT_REQUEST_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/logic-platform-client-request@1"
)
CLIENT_RESULT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/logic-platform-client-result@1"
)
CLIENT_TASK_ID: Final = "LPC-110"
CLIENT_GOAL_ID: Final = "LPC-G110"

MANIFEST_MODULE: Final = "ipfs_datasets_py.logic.platform.manifest"
CATALOG_MODULE: Final = "ipfs_datasets_py.logic.families.canonical_catalog"
ARTIFACTS_MODULE: Final = "ipfs_datasets_py.logic.formalization.artifacts_v3"
REQUESTS_V2_MODULE: Final = "ipfs_datasets_py.logic.backends.requests_v2"
NAMESPACES_MODULE: Final = "ipfs_datasets_py.logic.families.namespaces"
VERIFICATION_API_MODULE: Final = "ipfs_datasets_py.logic.verification_api"
PROOF_REPOSITORY_MODULE: Final = "ipfs_datasets_py.logic.common.proof_repository"
TACTICIAN_CONTRACTS_MODULE: Final = (
    "ipfs_datasets_py.logic.software_verification.tactician.contracts"
)

# Closed client operation vocabulary (acceptance LPC-110).
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

TYPED_PROVIDER_OPERATIONS: Final[frozenset[str]] = frozenset(
    {
        "capability",
        "translate",
        "prove",
        "reconstruct",
        "verify",
        "attest",
    }
)

# Maximum authority ceiling each evidence kind may claim (mirrors BackendRequest@2).
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

_NON_KERNEL_EVIDENCE: Final[frozenset[str]] = frozenset(
    {
        "parse",
        "model",
        "trace",
        "attack",
        "monitor",
        "candidate",
        "advisory",
    }
)

_REQUIRED_REQUEST_BINDINGS: Final[tuple[str, ...]] = (
    "task_id",
    "tree_id",
    "policy_id",
    "plan_id",
    "resource_budget",
    "network_allowed",
    "cancellation",
    "deadline_unix_ms",
    "correlation_id",
    "evidence_kind",
    "authority_ceiling",
)


# ---------------------------------------------------------------------------
# Errors / enums
# ---------------------------------------------------------------------------


class LogicPlatformClientError(RuntimeError):
    """Raised when a client request or operation fails closed."""


class LogicPlatformClientBindingError(LogicPlatformClientError, ValueError):
    """Raised when a request is missing required bindings or is malformed."""


class LogicPlatformClientAuthorityError(LogicPlatformClientError, ValueError):
    """Raised when a caller overclaims authority relative to evidence kind."""


class LogicPlatformClientHandshakeError(LogicPlatformClientError):
    """Raised when an operation requires a compatible handshake."""


class CacheFreshnessStatus(str, Enum):
    """Closed cache freshness vocabulary for client results."""

    CURRENT = "current"
    STALE = "stale"
    UNKNOWN = "unknown"
    MISS = "miss"
    INVALIDATED = "invalidated"


class ClientOperationStatus(str, Enum):
    """Lifecycle status of a client operation (not a semantic verdict)."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    INVALID = "invalid"
    UNAVAILABLE = "unavailable"
    CANCELLED = "cancelled"
    DECLARATIVE = "declarative"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _text(value: object, field_name: str, *, required: bool = True) -> str:
    if value is None:
        if required:
            raise LogicPlatformClientBindingError(
                f"{field_name} is required; fail closed"
            )
        return ""
    if not isinstance(value, str):
        raise LogicPlatformClientBindingError(
            f"{field_name} must be a string; fail closed"
        )
    text = value.strip()
    if required and not text:
        raise LogicPlatformClientBindingError(
            f"{field_name} must be a non-empty string; fail closed"
        )
    if "\x00" in text:
        raise LogicPlatformClientBindingError(
            f"{field_name} must not contain NUL bytes; fail closed"
        )
    return text


def _optional_int(value: object, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise LogicPlatformClientBindingError(
            f"{field_name} must be an integer or null; fail closed"
        )
    if value < 0:
        raise LogicPlatformClientBindingError(
            f"{field_name} must be non-negative; fail closed"
        )
    return value


def _json_object(value: object, *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise LogicPlatformClientBindingError(
            f"{field_name} must be a mapping; fail closed"
        )
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        decoded = json.loads(encoded)
    except (TypeError, ValueError) as error:
        raise LogicPlatformClientBindingError(
            f"{field_name} must be strict JSON; fail closed"
        ) from error
    if not isinstance(decoded, dict):
        raise LogicPlatformClientBindingError(
            f"{field_name} must decode to an object; fail closed"
        )
    return MappingProxyType(decoded)


def _hex_digest(payload: Mapping[str, Any] | Sequence[Any] | str | bytes) -> str:
    """Return a bare 64-hex SHA-256 digest (datasets artifact contract)."""

    if isinstance(payload, bytes):
        body = payload
    elif isinstance(payload, str):
        body = payload.encode("utf-8")
    else:
        body = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            default=str,
        ).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def _digest_of(payload: Mapping[str, Any] | Sequence[Any] | str) -> str:
    return f"sha256:{_hex_digest(payload)}"


def _artifact_digest(value: object, *, fallback: str) -> str:
    """Normalize a digest for FormalizationArtifact / DomainLogicSlice fields."""

    if value is None or value == "":
        return _hex_digest(fallback)
    text = _text(value, "digest")
    if text.startswith("sha256:"):
        text = text[len("sha256:") :]
    if len(text) == 64 and all(ch in "0123456789abcdef" for ch in text):
        return text
    return _hex_digest(text)


def _resource_budget(value: object) -> ResourceBudget:
    if isinstance(value, ResourceBudget):
        return value
    if value is None:
        return ResourceBudget()
    if isinstance(value, Mapping):
        payload = dict(value)
        # Accept both schema-tagged and bare resource maps.
        if "schema" not in payload and "schema_version" not in payload:
            try:
                return ResourceBudget(
                    wall_time_ms=int(payload.get("wall_time_ms", 0) or 0),
                    cpu_time_ms=int(payload.get("cpu_time_ms", 0) or 0),
                    memory_bytes=int(payload.get("memory_bytes", 0) or 0),
                    disk_bytes=int(payload.get("disk_bytes", 0) or 0),
                    max_processes=int(payload.get("max_processes", 0) or 0),
                    max_premises=int(payload.get("max_premises", 0) or 0),
                    max_output_bytes=int(payload.get("max_output_bytes", 0) or 0),
                    model_token_limit=int(payload.get("model_token_limit", 0) or 0),
                    provider_quota=int(payload.get("provider_quota", 0) or 0),
                    network_allowed=bool(payload.get("network_allowed", False)),
                )
            except Exception as error:
                raise LogicPlatformClientBindingError(
                    f"resource_budget is malformed: {error}"
                ) from error
        try:
            return ResourceBudget.from_dict(payload)
        except Exception as error:
            raise LogicPlatformClientBindingError(
                f"resource_budget is malformed: {error}"
            ) from error
    raise LogicPlatformClientBindingError(
        "resource_budget must be a ResourceBudget or mapping; fail closed"
    )


def _authority_token(value: object) -> str:
    if isinstance(value, Enum):
        token = str(value.value)
    else:
        token = _text(value, "authority_ceiling")
    if token not in _AUTHORITY_RANK:
        raise LogicPlatformClientAuthorityError(
            f"unknown authority_ceiling {token!r}; fail closed"
        )
    return token


def _evidence_token(value: object) -> str:
    if isinstance(value, Enum):
        return str(value.value)
    if isinstance(value, Mapping):
        raw = value.get("value") or value.get("id") or value.get("evidence_kind")
        return _text(raw, "evidence_kind")
    text = _text(value, "evidence_kind")
    # Accept qualified forms like "evidence:candidate".
    if ":" in text:
        prefix, _, rest = text.partition(":")
        if prefix in {"evidence", "ev"} and rest:
            return rest
    return text


def check_authority_overclaim(
    *,
    evidence_kind: object,
    authority_ceiling: object,
) -> None:
    """Fail closed when authority exceeds what the evidence kind supports."""

    evidence = _evidence_token(evidence_kind)
    ceiling = _authority_token(authority_ceiling)
    max_ceiling = _EVIDENCE_AUTHORITY_CEILING.get(evidence, "advisory")
    if _AUTHORITY_RANK[ceiling] > _AUTHORITY_RANK[max_ceiling]:
        raise LogicPlatformClientAuthorityError(
            f"authority_ceiling {ceiling!r} overclaims evidence kind "
            f"{evidence!r} (max admitted {max_ceiling!r}); fail closed"
        )
    if ceiling == "kernel" and evidence in _NON_KERNEL_EVIDENCE:
        raise LogicPlatformClientAuthorityError(
            f"kernel authority cannot be claimed with evidence {evidence!r}"
        )
    if evidence in _NON_KERNEL_EVIDENCE and ceiling in {
        "kernel",
        "reconstruction",
    }:
        raise LogicPlatformClientAuthorityError(
            f"proof/reconstruction authority cannot be claimed with evidence "
            f"{evidence!r}"
        )


def _lazy_import(module_name: str) -> Any:
    return importlib.import_module(module_name)


def _to_dict(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return value.to_dict()
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (list, tuple)):
        return [_to_dict(item) for item in value]
    return value


# ---------------------------------------------------------------------------
# Request / result envelopes
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LogicPlatformClientRequest:
    """Bound supervisor request for a logic-platform client operation.

    Required bindings (LPC-G110 acceptance):

    * ``task_id`` / ``tree_id`` / ``policy_id`` / ``plan_id``
    * ``resource_budget`` / ``network_allowed``
    * ``cancellation`` / ``deadline_unix_ms``
    * ``correlation_id``
    * ``evidence_kind`` / ``authority_ceiling``
    """

    task_id: str
    tree_id: str
    policy_id: str
    plan_id: str
    correlation_id: str
    evidence_kind: str
    authority_ceiling: str
    resource_budget: ResourceBudget = field(default_factory=ResourceBudget)
    network_allowed: bool = False
    cancellation: CancellationToken | None = None
    deadline_unix_ms: int | None = None
    operation: str = ""
    payload: Mapping[str, Any] = field(default_factory=dict)
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    schema_version: str = CLIENT_REQUEST_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", _text(self.task_id, "task_id"))
        object.__setattr__(self, "tree_id", _text(self.tree_id, "tree_id"))
        object.__setattr__(self, "policy_id", _text(self.policy_id, "policy_id"))
        object.__setattr__(self, "plan_id", _text(self.plan_id, "plan_id"))
        object.__setattr__(
            self,
            "correlation_id",
            _text(self.correlation_id, "correlation_id"),
        )
        evidence = _evidence_token(self.evidence_kind)
        object.__setattr__(self, "evidence_kind", evidence)
        ceiling = _authority_token(self.authority_ceiling)
        object.__setattr__(self, "authority_ceiling", ceiling)
        check_authority_overclaim(
            evidence_kind=evidence,
            authority_ceiling=ceiling,
        )
        object.__setattr__(
            self, "resource_budget", _resource_budget(self.resource_budget)
        )
        if not isinstance(self.network_allowed, bool):
            raise LogicPlatformClientBindingError(
                "network_allowed must be a boolean; fail closed"
            )
        # Resource budget may not authorize more network than the request.
        budget_network = bool(
            getattr(self.resource_budget, "network_allowed", False)
        )
        if self.network_allowed and not budget_network:
            raise LogicPlatformClientBindingError(
                "network_allowed exceeds resource_budget.network_allowed; "
                "fail closed"
            )
        object.__setattr__(
            self,
            "deadline_unix_ms",
            _optional_int(self.deadline_unix_ms, "deadline_unix_ms"),
        )
        if self.cancellation is not None and not isinstance(
            self.cancellation, CancellationToken
        ):
            raise LogicPlatformClientBindingError(
                "cancellation must be a CancellationToken or None; fail closed"
            )
        operation = _text(self.operation, "operation", required=False)
        object.__setattr__(self, "operation", operation)
        object.__setattr__(
            self,
            "payload",
            _json_object(self.payload, field_name="payload"),
        )
        request_id = _text(self.request_id, "request_id")
        if len(request_id) > 128:
            raise LogicPlatformClientBindingError(
                "request_id must be at most 128 characters; fail closed"
            )
        object.__setattr__(self, "request_id", request_id)
        if self.schema_version != CLIENT_REQUEST_SCHEMA:
            raise LogicPlatformClientBindingError(
                f"unsupported client request schema {self.schema_version!r}"
            )

    @property
    def cancelled(self) -> bool:
        return self.cancellation is not None and self.cancellation.is_cancelled()

    @property
    def expired(self) -> bool:
        return (
            self.deadline_unix_ms is not None
            and int(time.time() * 1000) >= self.deadline_unix_ms
        )

    def binding_digest(self) -> str:
        return _digest_of(
            {
                "authority_ceiling": self.authority_ceiling,
                "correlation_id": self.correlation_id,
                "deadline_unix_ms": self.deadline_unix_ms,
                "evidence_kind": self.evidence_kind,
                "network_allowed": self.network_allowed,
                "plan_id": self.plan_id,
                "policy_id": self.policy_id,
                "resource_budget": self.resource_budget.to_dict(),
                "task_id": self.task_id,
                "tree_id": self.tree_id,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "authority_ceiling": self.authority_ceiling,
            "binding_digest": self.binding_digest(),
            "cancellation": {
                "cancelled": self.cancelled,
            },
            "correlation_id": self.correlation_id,
            "deadline_unix_ms": self.deadline_unix_ms,
            "evidence_kind": self.evidence_kind,
            "network_allowed": self.network_allowed,
            "operation": self.operation,
            "payload": dict(self.payload),
            "plan_id": self.plan_id,
            "policy_id": self.policy_id,
            "request_id": self.request_id,
            "resource_budget": self.resource_budget.to_dict(),
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "tree_id": self.tree_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LogicPlatformClientRequest":
        if not isinstance(payload, Mapping):
            raise LogicPlatformClientBindingError(
                "client request must be a mapping; fail closed"
            )
        cancellation = payload.get("cancellation")
        token: CancellationToken | None = None
        if isinstance(cancellation, CancellationToken):
            token = cancellation
        elif isinstance(cancellation, Mapping) and bool(
            cancellation.get("cancelled")
        ):
            token = CancellationToken()
            token.cancel()
        elif cancellation is not None and not isinstance(cancellation, Mapping):
            raise LogicPlatformClientBindingError(
                "cancellation must be a mapping, CancellationToken, or null"
            )
        return cls(
            task_id=str(payload.get("task_id") or ""),
            tree_id=str(payload.get("tree_id") or ""),
            policy_id=str(payload.get("policy_id") or ""),
            plan_id=str(payload.get("plan_id") or ""),
            correlation_id=str(payload.get("correlation_id") or ""),
            evidence_kind=str(payload.get("evidence_kind") or ""),
            authority_ceiling=str(payload.get("authority_ceiling") or ""),
            resource_budget=payload.get("resource_budget") or {},
            network_allowed=bool(payload.get("network_allowed", False)),
            cancellation=token,
            deadline_unix_ms=payload.get("deadline_unix_ms"),
            operation=str(payload.get("operation") or ""),
            payload=payload.get("payload") or {},
            request_id=str(payload.get("request_id") or uuid.uuid4().hex),
            schema_version=str(
                payload.get("schema_version") or CLIENT_REQUEST_SCHEMA
            ),
        )


@dataclass(frozen=True, slots=True)
class LogicPlatformClientResult:
    """Typed client result; never upgrades authority above the request ceiling."""

    operation: str
    status: ClientOperationStatus | str
    request_id: str
    correlation_id: str
    result: Mapping[str, Any] | None = None
    error: str | None = None
    evidence_kind: str = "candidate"
    authority_ceiling: str = "candidate"
    cache_freshness: CacheFreshnessStatus | str = CacheFreshnessStatus.UNKNOWN
    binding_digest: str = ""
    duration_ms: int = 0
    schema_version: str = CLIENT_RESULT_SCHEMA
    interface: str = SUPERVISOR_LOGIC_PLATFORM_CLIENT_INTERFACE

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "operation", _text(self.operation, "operation")
        )
        if isinstance(self.status, ClientOperationStatus):
            status = self.status
        else:
            try:
                status = ClientOperationStatus(
                    _text(self.status, "status")
                )
            except ValueError as error:
                raise LogicPlatformClientError(
                    f"unknown client operation status {self.status!r}"
                ) from error
        object.__setattr__(self, "status", status)
        object.__setattr__(
            self, "request_id", _text(self.request_id, "request_id")
        )
        object.__setattr__(
            self,
            "correlation_id",
            _text(self.correlation_id, "correlation_id"),
        )
        if self.result is not None:
            object.__setattr__(
                self,
                "result",
                _json_object(self.result, field_name="result"),
            )
        if self.error is not None:
            object.__setattr__(
                self, "error", _text(self.error, "error", required=False)
            )
        object.__setattr__(
            self, "evidence_kind", _evidence_token(self.evidence_kind)
        )
        object.__setattr__(
            self,
            "authority_ceiling",
            _authority_token(self.authority_ceiling),
        )
        check_authority_overclaim(
            evidence_kind=self.evidence_kind,
            authority_ceiling=self.authority_ceiling,
        )
        if isinstance(self.cache_freshness, CacheFreshnessStatus):
            freshness = self.cache_freshness
        else:
            try:
                freshness = CacheFreshnessStatus(
                    _text(self.cache_freshness, "cache_freshness")
                )
            except ValueError as error:
                raise LogicPlatformClientError(
                    f"unknown cache freshness {self.cache_freshness!r}"
                ) from error
        object.__setattr__(self, "cache_freshness", freshness)
        object.__setattr__(
            self,
            "binding_digest",
            _text(self.binding_digest, "binding_digest", required=False),
        )
        if (
            isinstance(self.duration_ms, bool)
            or not isinstance(self.duration_ms, int)
            or self.duration_ms < 0
        ):
            raise LogicPlatformClientError(
                "duration_ms must be a non-negative integer"
            )
        if self.schema_version != CLIENT_RESULT_SCHEMA:
            raise LogicPlatformClientError(
                f"unsupported client result schema {self.schema_version!r}"
            )
        if status is ClientOperationStatus.SUCCEEDED and self.error:
            raise LogicPlatformClientError(
                "successful client results cannot carry an error"
            )
        if status is not ClientOperationStatus.SUCCEEDED and self.result is not None:
            # Declarative/partial results may still carry a body without success.
            pass

    @property
    def ok(self) -> bool:
        return self.status in {
            ClientOperationStatus.SUCCEEDED,
            ClientOperationStatus.DECLARATIVE,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "authority_ceiling": self.authority_ceiling,
            "binding_digest": self.binding_digest,
            "cache_freshness": (
                self.cache_freshness.value
                if isinstance(self.cache_freshness, CacheFreshnessStatus)
                else str(self.cache_freshness)
            ),
            "correlation_id": self.correlation_id,
            "duration_ms": self.duration_ms,
            "error": self.error,
            "evidence_kind": self.evidence_kind,
            "interface": self.interface,
            "ok": self.ok,
            "operation": self.operation,
            "request_id": self.request_id,
            "result": dict(self.result) if self.result is not None else None,
            "schema_version": self.schema_version,
            "status": (
                self.status.value
                if isinstance(self.status, ClientOperationStatus)
                else str(self.status)
            ),
        }


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class SupervisorLogicPlatformClient:
    """Lazy supervisor-side client for the datasets logic platform.

    Interface: ``SupervisorLogicPlatformClient@1``.
    """

    interface: Final = SUPERVISOR_LOGIC_PLATFORM_CLIENT_INTERFACE
    version: Final = SUPERVISOR_LOGIC_PLATFORM_CLIENT_VERSION
    schema_version: Final = CLIENT_SCHEMA_VERSION
    task_id: Final = CLIENT_TASK_ID
    goal_id: Final = CLIENT_GOAL_ID
    operations: Final = CLIENT_OPERATIONS

    def __init__(
        self,
        *,
        adapter: SupervisorCanonicalLogicAdapter | None = None,
        provider_facade: SupervisorLogicProviderFacade | None = None,
        module_importer: Callable[[str], Any] | None = None,
        require_handshake: bool = True,
    ) -> None:
        self._adapter = adapter
        self._provider_facade = provider_facade
        self._import = module_importer or _lazy_import
        self._require_handshake = bool(require_handshake)
        self._lock = threading.RLock()
        self._handshake_result: Any | None = None
        self._manifest: Any | None = None
        self._catalog_snapshot: Any | None = None

    # ------------------------------------------------------------------
    # Lazy dependencies
    # ------------------------------------------------------------------

    @property
    def adapter(self) -> SupervisorCanonicalLogicAdapter:
        if self._adapter is None:
            self._adapter = get_canonical_logic_adapter()
        return self._adapter

    @property
    def handshaken(self) -> bool:
        result = self._handshake_result
        if result is None:
            return False
        return bool(getattr(result, "compatible", False))

    def datasets_import_is_lazy(self) -> bool:
        """True when this client has not yet imported datasets modules."""

        # Handshake/catalog/formalization load datasets; until then stay cold.
        return self._manifest is None and self._catalog_snapshot is None

    def _load(self, module_name: str) -> Any:
        return self._import(module_name)

    def _ensure_handshake(self, *, operation: str) -> None:
        if not self._require_handshake:
            return
        if self.handshaken:
            return
        raise LogicPlatformClientHandshakeError(
            f"operation {operation!r} requires a compatible handshake first; "
            "fail closed"
        )

    def _bind_request(
        self,
        request: LogicPlatformClientRequest | Mapping[str, Any] | None,
        *,
        operation: str,
        payload: Mapping[str, Any] | None = None,
        **binding_overrides: Any,
    ) -> LogicPlatformClientRequest:
        if request is None:
            if not binding_overrides:
                raise LogicPlatformClientBindingError(
                    f"{operation} requires a bound LogicPlatformClientRequest"
                )
            body = dict(binding_overrides)
            if payload is not None:
                body["payload"] = payload
            body.setdefault("operation", operation)
            return LogicPlatformClientRequest.from_dict(body)
        if isinstance(request, LogicPlatformClientRequest):
            bound = request
        elif isinstance(request, Mapping):
            body = dict(request)
            body.update(binding_overrides)
            if payload is not None:
                body["payload"] = payload
            body.setdefault("operation", operation)
            bound = LogicPlatformClientRequest.from_dict(body)
        else:
            raise LogicPlatformClientBindingError(
                "request must be LogicPlatformClientRequest, mapping, or None"
            )
        if bound.cancelled:
            raise LogicPlatformClientError(
                f"{operation} cancelled before execution"
            )
        if bound.expired:
            raise LogicPlatformClientError(
                f"{operation} deadline expired before execution"
            )
        return bound

    def _result(
        self,
        *,
        operation: str,
        status: ClientOperationStatus,
        request: LogicPlatformClientRequest,
        result: Mapping[str, Any] | None = None,
        error: str | None = None,
        cache_freshness: CacheFreshnessStatus = CacheFreshnessStatus.UNKNOWN,
        duration_ms: int = 0,
        evidence_kind: str | None = None,
        authority_ceiling: str | None = None,
    ) -> LogicPlatformClientResult:
        return LogicPlatformClientResult(
            operation=operation,
            status=status,
            request_id=request.request_id,
            correlation_id=request.correlation_id,
            result=result,
            error=error,
            evidence_kind=evidence_kind or request.evidence_kind,
            authority_ceiling=authority_ceiling or request.authority_ceiling,
            cache_freshness=cache_freshness,
            binding_digest=request.binding_digest(),
            duration_ms=duration_ms,
        )

    # ------------------------------------------------------------------
    # Handshake
    # ------------------------------------------------------------------

    def handshake(
        self,
        requirements: Mapping[str, Any] | Any | None = None,
        *,
        manifest: Any | None = None,
    ) -> Any:
        """Perform package-neutral platform handshake (first lazy step)."""

        started = time.time()
        module = self._load(MANIFEST_MODULE)
        handshake_fn = getattr(module, "handshake")
        if requirements is None:
            # Pin this client interface as a required adapter version.
            req_cls = getattr(module, "HandshakeRequirements")
            requirements = req_cls(
                required_adapter_versions=(
                    SUPERVISOR_LOGIC_PLATFORM_CLIENT_INTERFACE,
                )
            )
        result = handshake_fn(requirements, manifest=manifest)
        with self._lock:
            self._handshake_result = result
            self._manifest = getattr(result, "manifest", manifest)
        # Touch duration for diagnostics without changing result type.
        _ = int((time.time() - started) * 1000)
        return result

    def last_handshake(self) -> Any | None:
        return self._handshake_result

    # ------------------------------------------------------------------
    # Catalog
    # ------------------------------------------------------------------

    def catalog(
        self,
        request: LogicPlatformClientRequest | Mapping[str, Any] | None = None,
        **binding: Any,
    ) -> LogicPlatformClientResult:
        """Return sealed catalog identity and declarative inventory."""

        self._ensure_handshake(operation="catalog")
        bound = self._bind_request(request, operation="catalog", **binding)
        started = time.time()
        try:
            catalog_mod = self._load(CATALOG_MODULE)
            snapshot = getattr(catalog_mod, "DEFAULT_CANONICAL_CATALOG_SNAPSHOT")
            with self._lock:
                self._catalog_snapshot = snapshot
            inventory = self.adapter.vocabulary_inventory()
            family_ids = self.adapter.list_canonical_family_ids()
            payload = {
                "catalog_interface": getattr(
                    snapshot, "interface", "CanonicalLogicCatalogSnapshot@1"
                ),
                "catalog_root": getattr(snapshot, "content_root", ""),
                "catalog_digest": getattr(snapshot, "content_digest", ""),
                "family_ids": list(family_ids),
                "family_count": len(family_ids),
                "vocabulary": inventory,
                "adapter_interface": self.adapter.interface,
            }
            return self._result(
                operation="catalog",
                status=ClientOperationStatus.DECLARATIVE,
                request=bound,
                result=payload,
                cache_freshness=CacheFreshnessStatus.CURRENT,
                duration_ms=int((time.time() - started) * 1000),
            )
        except LogicPlatformClientError:
            raise
        except Exception as error:
            return self._result(
                operation="catalog",
                status=ClientOperationStatus.UNAVAILABLE,
                request=bound,
                error=f"{type(error).__name__}: {error}",
                duration_ms=int((time.time() - started) * 1000),
            )

    # ------------------------------------------------------------------
    # Formalization / slice / obligation / plan
    # ------------------------------------------------------------------

    def formalize(
        self,
        request: LogicPlatformClientRequest | Mapping[str, Any] | None = None,
        *,
        document_id: str = "",
        source_digest: str = "",
        expression_id: str = "",
        expression_digest: str = "",
        domain: str = "software",
        family: str = "first_order",
        profile: str = "default",
        view: str = "surface",
        notation: str = "text",
        statement: str = "",
        **binding: Any,
    ) -> LogicPlatformClientResult:
        """Build a FormalizationArtifact@3 candidate bound to the request."""

        self._ensure_handshake(operation="formalize")
        payload = {
            "document_id": document_id,
            "source_digest": source_digest,
            "expression_id": expression_id,
            "expression_digest": expression_digest,
            "domain": domain,
            "family": family,
            "profile": profile,
            "view": view,
            "notation": notation,
            "statement": statement,
        }
        bound = self._bind_request(
            request, operation="formalize", payload=payload, **binding
        )
        started = time.time()
        try:
            artifacts = self._load(ARTIFACTS_MODULE)
            namespaces = self._load(NAMESPACES_MODULE)
            body = dict(bound.payload)
            doc_id = _text(
                body.get("document_id") or document_id or "document:client",
                "document_id",
            )
            src = _artifact_digest(
                body.get("source_digest") or source_digest,
                fallback=statement or doc_id,
            )
            expr_id = _text(
                body.get("expression_id") or expression_id or "expression:client",
                "expression_id",
            )
            expr_digest = _artifact_digest(
                body.get("expression_digest") or expression_digest,
                fallback=statement or expr_id,
            )
            artifact_cls = getattr(artifacts, "FormalizationArtifactV3")
            artifact = artifact_cls(
                artifact_id=f"formalization:{bound.request_id}",
                sample_id=f"sample:{bound.task_id}",
                domain=_text(body.get("domain") or domain, "domain"),
                document_id=doc_id,
                source_digest=src,
                expression_id=expr_id,
                expression_digest=expr_digest,
                family=namespaces.family_id(
                    _text(body.get("family") or family, "family")
                ),
                profile=namespaces.profile_id(
                    _text(body.get("profile") or profile, "profile")
                ),
                view=namespaces.view_id(_text(body.get("view") or view, "view")),
                notation=namespaces.notation_id(
                    _text(body.get("notation") or notation, "notation")
                ),
                metadata={
                    "binding_digest": bound.binding_digest(),
                    "correlation_id": bound.correlation_id,
                    "plan_id": bound.plan_id,
                    "policy_id": bound.policy_id,
                    "task_id": bound.task_id,
                    "tree_id": bound.tree_id,
                    "statement": _text(
                        body.get("statement") or statement,
                        "statement",
                        required=False,
                    ),
                },
            )
            return self._result(
                operation="formalize",
                status=ClientOperationStatus.SUCCEEDED,
                request=bound,
                result={
                    "artifact": _to_dict(artifact),
                    "interface": getattr(
                        artifact, "interface", "FormalizationArtifact@3"
                    ),
                    "content_digest": getattr(artifact, "content_digest", ""),
                },
                duration_ms=int((time.time() - started) * 1000),
            )
        except LogicPlatformClientError:
            raise
        except Exception as error:
            return self._result(
                operation="formalize",
                status=ClientOperationStatus.FAILED,
                request=bound,
                error=f"{type(error).__name__}: {error}",
                duration_ms=int((time.time() - started) * 1000),
            )

    def create_slice(
        self,
        request: LogicPlatformClientRequest | Mapping[str, Any] | None = None,
        *,
        slice_id: str = "",
        document_id: str = "",
        source_digest: str = "",
        expression_id: str = "",
        expression_digest: str = "",
        domain: str = "software",
        family: str = "first_order",
        profile: str = "default",
        property_id: str = "theorem",
        view: str = "surface",
        notation: str = "text",
        **binding: Any,
    ) -> LogicPlatformClientResult:
        """Create an admitted DomainLogicSlice@2 bound to the request."""

        self._ensure_handshake(operation="create_slice")
        payload = {
            "slice_id": slice_id,
            "document_id": document_id,
            "source_digest": source_digest,
            "expression_id": expression_id,
            "expression_digest": expression_digest,
            "domain": domain,
            "family": family,
            "profile": profile,
            "property": property_id,
            "view": view,
            "notation": notation,
        }
        bound = self._bind_request(
            request, operation="create_slice", payload=payload, **binding
        )
        started = time.time()
        try:
            artifacts = self._load(ARTIFACTS_MODULE)
            namespaces = self._load(NAMESPACES_MODULE)
            body = dict(bound.payload)
            src = _artifact_digest(
                body.get("source_digest") or source_digest,
                fallback=bound.correlation_id,
            )
            expr_digest = _artifact_digest(
                body.get("expression_digest") or expression_digest,
                fallback=bound.request_id,
            )
            slice_cls = getattr(artifacts, "DomainLogicSliceV2")
            slice_obj = slice_cls(
                slice_id=_text(
                    body.get("slice_id") or slice_id or f"slice:{bound.request_id}",
                    "slice_id",
                ),
                domain=_text(body.get("domain") or domain, "domain"),
                document_id=_text(
                    body.get("document_id") or document_id or "document:client",
                    "document_id",
                ),
                source_digest=src,
                expression_id=_text(
                    body.get("expression_id")
                    or expression_id
                    or "expression:client",
                    "expression_id",
                ),
                expression_digest=expr_digest,
                family=namespaces.family_id(
                    _text(body.get("family") or family, "family")
                ),
                profile=namespaces.profile_id(
                    _text(body.get("profile") or profile, "profile")
                ),
                property=namespaces.property_id(
                    _text(body.get("property") or property_id, "property")
                ),
                view=namespaces.view_id(_text(body.get("view") or view, "view")),
                notation=namespaces.notation_id(
                    _text(body.get("notation") or notation, "notation")
                ),
                metadata={
                    "binding_digest": bound.binding_digest(),
                    "task_id": bound.task_id,
                    "tree_id": bound.tree_id,
                    "policy_id": bound.policy_id,
                    "plan_id": bound.plan_id,
                },
            )
            return self._result(
                operation="create_slice",
                status=ClientOperationStatus.SUCCEEDED,
                request=bound,
                result={
                    "slice": _to_dict(slice_obj),
                    "interface": getattr(slice_obj, "interface", "DomainLogicSlice@2"),
                    "content_digest": getattr(slice_obj, "content_digest", ""),
                },
                duration_ms=int((time.time() - started) * 1000),
            )
        except LogicPlatformClientError:
            raise
        except Exception as error:
            return self._result(
                operation="create_slice",
                status=ClientOperationStatus.FAILED,
                request=bound,
                error=f"{type(error).__name__}: {error}",
                duration_ms=int((time.time() - started) * 1000),
            )

    def create_obligation(
        self,
        request: LogicPlatformClientRequest | Mapping[str, Any] | None = None,
        *,
        slice_payload: Mapping[str, Any] | Any | None = None,
        obligation_id: str = "",
        statement: str = "",
        encoding: str = "smt_lib",
        evidence_kind: str | None = None,
        bounds: Mapping[str, Any] | None = None,
        **binding: Any,
    ) -> LogicPlatformClientResult:
        """Create LogicObligation@2 from an admitted domain slice."""

        self._ensure_handshake(operation="create_obligation")
        payload = {
            "slice": _to_dict(slice_payload) if slice_payload is not None else {},
            "obligation_id": obligation_id,
            "statement": statement,
            "encoding": encoding,
            "evidence_kind": evidence_kind or "",
            "bounds": dict(bounds or {}),
        }
        bound = self._bind_request(
            request, operation="create_obligation", payload=payload, **binding
        )
        started = time.time()
        try:
            artifacts = self._load(ARTIFACTS_MODULE)
            requests_v2 = self._load(REQUESTS_V2_MODULE)
            namespaces = self._load(NAMESPACES_MODULE)
            body = dict(bound.payload)
            raw_slice = body.get("slice") or slice_payload
            if raw_slice is None:
                raise LogicPlatformClientBindingError(
                    "create_obligation requires slice_payload"
                )
            slice_cls = getattr(artifacts, "DomainLogicSliceV2")
            if isinstance(raw_slice, slice_cls):
                slice_obj = raw_slice
            elif isinstance(raw_slice, Mapping):
                slice_obj = slice_cls.from_dict(dict(raw_slice))
            elif hasattr(raw_slice, "to_dict") and callable(raw_slice.to_dict):
                slice_obj = slice_cls.from_dict(dict(raw_slice.to_dict()))
            else:
                raise LogicPlatformClientBindingError(
                    "slice_payload must be DomainLogicSliceV2 or a mapping"
                )
            bound_evidence = _text(
                body.get("evidence_kind") or evidence_kind or bound.evidence_kind,
                "evidence_kind",
            )
            # Obligation authority cannot exceed the client request ceiling.
            check_authority_overclaim(
                evidence_kind=bound_evidence,
                authority_ceiling=bound.authority_ceiling,
            )
            request_bounds = body.get("bounds") or bounds
            if not request_bounds:
                budget = bound.resource_budget
                request_bounds = {
                    "timeout_ms": max(1, int(getattr(budget, "wall_time_ms", 0) or 1_000)),
                    "max_steps": max(1, int(getattr(budget, "max_premises", 0) or 1_024)),
                    "max_memory_bytes": max(
                        1, int(getattr(budget, "memory_bytes", 0) or 16 * 1024 * 1024)
                    ),
                    "max_output_bytes": max(
                        1, int(getattr(budget, "max_output_bytes", 0) or 65_536)
                    ),
                }
            obligation_cls = getattr(requests_v2, "LogicObligationV2")
            obligation = obligation_cls.from_slice(
                slice_obj,
                obligation_id=_text(
                    body.get("obligation_id")
                    or obligation_id
                    or f"obligation:{bound.request_id}",
                    "obligation_id",
                ),
                statement=_text(
                    body.get("statement") or statement or "client obligation",
                    "statement",
                ),
                encoding=namespaces.encoding_id(
                    _text(body.get("encoding") or encoding, "encoding")
                ),
                evidence_kind=namespaces.evidence_id(bound_evidence),
                bounds=request_bounds,
                authority_ceiling=bound.authority_ceiling,
                metadata={
                    "binding_digest": bound.binding_digest(),
                    "correlation_id": bound.correlation_id,
                    "task_id": bound.task_id,
                    "tree_id": bound.tree_id,
                    "policy_id": bound.policy_id,
                    "plan_id": bound.plan_id,
                },
            )
            return self._result(
                operation="create_obligation",
                status=ClientOperationStatus.SUCCEEDED,
                request=bound,
                result={
                    "obligation": _to_dict(obligation),
                    "interface": getattr(
                        obligation, "interface", "LogicObligation@2"
                    ),
                    "content_digest": getattr(obligation, "content_digest", ""),
                },
                duration_ms=int((time.time() - started) * 1000),
            )
        except LogicPlatformClientError:
            raise
        except Exception as error:
            return self._result(
                operation="create_obligation",
                status=ClientOperationStatus.FAILED,
                request=bound,
                error=f"{type(error).__name__}: {error}",
                duration_ms=int((time.time() - started) * 1000),
            )

    def create_plan(
        self,
        request: LogicPlatformClientRequest | Mapping[str, Any] | None = None,
        *,
        formal_goal_id: str = "",
        graph_id: str = "",
        candidates: Sequence[Mapping[str, Any]] | None = None,
        **binding: Any,
    ) -> LogicPlatformClientResult:
        """Create a GoalDirectedProofPlan@1 draft bound to task/tree/policy."""

        self._ensure_handshake(operation="create_plan")
        payload = {
            "formal_goal_id": formal_goal_id,
            "graph_id": graph_id,
            "candidates": list(candidates or ()),
        }
        bound = self._bind_request(
            request, operation="create_plan", payload=payload, **binding
        )
        started = time.time()
        try:
            contracts = self._load(TACTICIAN_CONTRACTS_MODULE)
            body = dict(bound.payload)
            plan_cls = getattr(contracts, "GoalDirectedProofPlan")
            plan = plan_cls(
                plan_id=bound.plan_id,
                formal_goal_id=_text(
                    body.get("formal_goal_id")
                    or formal_goal_id
                    or f"goal:{bound.task_id}",
                    "formal_goal_id",
                ),
                graph_id=_text(
                    body.get("graph_id") or graph_id or f"graph:{bound.tree_id}",
                    "graph_id",
                ),
                tree_id=bound.tree_id,
                candidates=tuple(body.get("candidates") or candidates or ()),
                status=getattr(contracts, "PlanStatus").DRAFT,
                bounds=getattr(contracts, "ResourceBounds")(
                    wall_time_ms=int(
                        getattr(bound.resource_budget, "wall_time_ms", 0) or 0
                    ),
                    memory_bytes=int(
                        getattr(bound.resource_budget, "memory_bytes", 0) or 0
                    ),
                    network_allowed=bound.network_allowed,
                ),
                authority=getattr(contracts, "AuthorityCeiling").CANDIDATE,
                proof_claimed=False,
                completion_claimed=False,
                metadata={
                    "binding_digest": bound.binding_digest(),
                    "correlation_id": bound.correlation_id,
                    "policy_id": bound.policy_id,
                    "task_id": bound.task_id,
                    "client_interface": self.interface,
                },
            )
            return self._result(
                operation="create_plan",
                status=ClientOperationStatus.SUCCEEDED,
                request=bound,
                result={
                    "plan": _to_dict(plan),
                    "interface": getattr(
                        plan, "INTERFACE", "GoalDirectedProofPlan@1"
                    ),
                    "proof_claimed": False,
                    "completion_claimed": False,
                },
                # Plans are candidate-only; never raise authority.
                evidence_kind="candidate",
                authority_ceiling="candidate",
                duration_ms=int((time.time() - started) * 1000),
            )
        except LogicPlatformClientError:
            raise
        except Exception as error:
            return self._result(
                operation="create_plan",
                status=ClientOperationStatus.FAILED,
                request=bound,
                error=f"{type(error).__name__}: {error}",
                duration_ms=int((time.time() - started) * 1000),
            )

    # ------------------------------------------------------------------
    # Capability discovery / typed invocation
    # ------------------------------------------------------------------

    def discover_capabilities(
        self,
        request: LogicPlatformClientRequest | Mapping[str, Any] | None = None,
        *,
        provider_facade: SupervisorLogicProviderFacade | None = None,
        **binding: Any,
    ) -> LogicPlatformClientResult:
        """Discover provider capabilities without claiming availability as proof."""

        self._ensure_handshake(operation="discover_capabilities")
        bound = self._bind_request(
            request, operation="discover_capabilities", **binding
        )
        started = time.time()
        facade = provider_facade or self._provider_facade
        try:
            inventory = self.adapter.vocabulary_inventory()
            providers: list[dict[str, Any]] = []
            diagnostics: list[str] = []
            if facade is not None:
                provider_request = ProviderRequest(
                    operation=ProofProviderOperation.CAPABILITY,
                    request_id=f"capability:{bound.request_id}",
                    payload={
                        "binding_digest": bound.binding_digest(),
                        "task_id": bound.task_id,
                        "tree_id": bound.tree_id,
                    },
                    resource_budget=bound.resource_budget,
                    network_allowed=bound.network_allowed,
                    deadline_unix_ms=bound.deadline_unix_ms,
                )
                response = facade.capability(provider_request)
                providers.append(
                    {
                        "provider_id": facade.provider_id,
                        "provider_version": facade.provider_version,
                        "loaded": facade.loaded,
                        "ok": response.ok,
                        "result": _to_dict(response.result)
                        if response.ok
                        else None,
                        "error": _to_dict(response.error)
                        if not response.ok
                        else None,
                    }
                )
            else:
                # Declarative discovery via verification API when no facade set.
                try:
                    api_mod = self._load(VERIFICATION_API_MODULE)
                    api = getattr(api_mod, "LogicVerificationAPI")()
                    listed = api.list_providers()
                    result_body = getattr(listed, "result", None) or {}
                    for item in result_body.get("providers") or ():
                        if isinstance(item, Mapping):
                            providers.append(dict(item))
                except Exception as error:  # pragma: no cover - optional path
                    diagnostics.append(
                        f"verification_api_unavailable:{type(error).__name__}"
                    )
            return self._result(
                operation="discover_capabilities",
                status=ClientOperationStatus.DECLARATIVE,
                request=bound,
                result={
                    "providers": providers,
                    "provider_count": len(providers),
                    "provider_operations": list(TYPED_PROVIDER_OPERATIONS),
                    "client_operations": list(CLIENT_OPERATIONS),
                    "vocabulary_domains": list(inventory.get("domains") or ()),
                    "diagnostics": diagnostics,
                    "availability_is_not_proof": True,
                },
                duration_ms=int((time.time() - started) * 1000),
            )
        except LogicPlatformClientError:
            raise
        except Exception as error:
            return self._result(
                operation="discover_capabilities",
                status=ClientOperationStatus.UNAVAILABLE,
                request=bound,
                error=f"{type(error).__name__}: {error}",
                duration_ms=int((time.time() - started) * 1000),
            )

    def invoke(
        self,
        operation: str | ProofProviderOperation,
        request: LogicPlatformClientRequest | Mapping[str, Any] | None = None,
        *,
        provider_facade: SupervisorLogicProviderFacade | None = None,
        payload: Mapping[str, Any] | None = None,
        **binding: Any,
    ) -> LogicPlatformClientResult:
        """Typed provider invocation through the lazy logic-provider facade."""

        op_token = (
            operation.value
            if isinstance(operation, ProofProviderOperation)
            else _text(operation, "operation")
        )
        if op_token not in TYPED_PROVIDER_OPERATIONS:
            raise LogicPlatformClientBindingError(
                f"unsupported provider operation {op_token!r}; expected one of "
                f"{sorted(TYPED_PROVIDER_OPERATIONS)}"
            )
        self._ensure_handshake(operation=f"invoke:{op_token}")
        bound = self._bind_request(
            request,
            operation=op_token,
            payload=payload,
            **binding,
        )
        started = time.time()
        facade = provider_facade or self._provider_facade
        if facade is None:
            return self._result(
                operation=op_token,
                status=ClientOperationStatus.UNAVAILABLE,
                request=bound,
                error="provider_facade is required for typed invocation",
                duration_ms=int((time.time() - started) * 1000),
            )
        try:
            provider_request = ProviderRequest(
                operation=ProofProviderOperation(op_token),
                request_id=bound.request_id,
                payload={
                    **dict(bound.payload),
                    "binding_digest": bound.binding_digest(),
                    "correlation_id": bound.correlation_id,
                    "task_id": bound.task_id,
                    "tree_id": bound.tree_id,
                    "policy_id": bound.policy_id,
                    "plan_id": bound.plan_id,
                    "evidence_kind": bound.evidence_kind,
                    "authority_ceiling": bound.authority_ceiling,
                },
                resource_budget=bound.resource_budget,
                network_allowed=bound.network_allowed,
                deadline_unix_ms=bound.deadline_unix_ms,
            )
            response = facade.invoke(
                provider_request, cancellation=bound.cancellation
            )
            return self._provider_response_to_result(
                operation=op_token,
                request=bound,
                response=response,
                duration_ms=int((time.time() - started) * 1000),
            )
        except LogicPlatformClientError:
            raise
        except Exception as error:
            return self._result(
                operation=op_token,
                status=ClientOperationStatus.FAILED,
                request=bound,
                error=f"{type(error).__name__}: {error}",
                duration_ms=int((time.time() - started) * 1000),
            )

    def _provider_response_to_result(
        self,
        *,
        operation: str,
        request: LogicPlatformClientRequest,
        response: ProviderResponse,
        duration_ms: int,
    ) -> LogicPlatformClientResult:
        if response.ok:
            # Provider success never upgrades authority above the request.
            return self._result(
                operation=operation,
                status=ClientOperationStatus.SUCCEEDED,
                request=request,
                result={
                    "provider_id": response.provider_id,
                    "provider_version": response.provider_version,
                    "provider_result": _to_dict(response.result) or {},
                    "authority_upgraded": False,
                    "duration_ms": response.duration_ms,
                },
                duration_ms=duration_ms,
            )
        error = response.error
        message = (
            error.message
            if error is not None and hasattr(error, "message")
            else str(error or "provider failure")
        )
        return self._result(
            operation=operation,
            status=ClientOperationStatus.FAILED,
            request=request,
            error=message,
            result={
                "provider_id": response.provider_id,
                "provider_version": response.provider_version,
                "provider_error": _to_dict(error),
            },
            duration_ms=duration_ms,
        )

    def reconstruct(
        self,
        request: LogicPlatformClientRequest | Mapping[str, Any] | None = None,
        *,
        provider_facade: SupervisorLogicProviderFacade | None = None,
        payload: Mapping[str, Any] | None = None,
        **binding: Any,
    ) -> LogicPlatformClientResult:
        return self.invoke(
            "reconstruct",
            request,
            provider_facade=provider_facade,
            payload=payload,
            **binding,
        )

    def verify(
        self,
        request: LogicPlatformClientRequest | Mapping[str, Any] | None = None,
        *,
        provider_facade: SupervisorLogicProviderFacade | None = None,
        payload: Mapping[str, Any] | None = None,
        **binding: Any,
    ) -> LogicPlatformClientResult:
        return self.invoke(
            "verify",
            request,
            provider_facade=provider_facade,
            payload=payload,
            **binding,
        )

    # ------------------------------------------------------------------
    # Receipts / counterexamples / cache freshness
    # ------------------------------------------------------------------

    def receipt(
        self,
        request: LogicPlatformClientRequest | Mapping[str, Any] | None = None,
        *,
        validation_result: Mapping[str, Any] | Any | None = None,
        **binding: Any,
    ) -> LogicPlatformClientResult:
        """Project a translation/validation receipt without authority upgrade."""

        self._ensure_handshake(operation="receipt")
        payload = {
            "validation_result": _to_dict(validation_result)
            if validation_result is not None
            else {},
        }
        bound = self._bind_request(
            request, operation="receipt", payload=payload, **binding
        )
        started = time.time()
        try:
            body = dict(bound.payload)
            raw = body.get("validation_result") or validation_result
            if raw is None:
                raise LogicPlatformClientBindingError(
                    "receipt requires validation_result"
                )
            projected = self.adapter.project_translation_validation_receipt(raw)
            # Adapter projects authority="none"; client may retain request ceiling
            # for binding but never elevates projected receipt authority.
            return self._result(
                operation="receipt",
                status=ClientOperationStatus.SUCCEEDED,
                request=bound,
                result={
                    "receipt": projected,
                    "authority": projected.get("authority", "none"),
                    "proof_success": bool(projected.get("proof_success", False)),
                    "valid": bool(projected.get("valid", False)),
                },
                # Receipt projection itself is non-authoritative.
                evidence_kind="candidate",
                authority_ceiling="candidate",
                duration_ms=int((time.time() - started) * 1000),
            )
        except LogicPlatformClientError:
            raise
        except Exception as error:
            return self._result(
                operation="receipt",
                status=ClientOperationStatus.FAILED,
                request=bound,
                error=f"{type(error).__name__}: {error}",
                duration_ms=int((time.time() - started) * 1000),
            )

    def counterexample(
        self,
        request: LogicPlatformClientRequest | Mapping[str, Any] | None = None,
        *,
        kind: str = "generic_failure",
        statement: str = "",
        witness: Mapping[str, Any] | None = None,
        **binding: Any,
    ) -> LogicPlatformClientResult:
        """Build a canonical supervisor counterexample (never a positive proof)."""

        self._ensure_handshake(operation="counterexample")
        payload = {
            "kind": kind,
            "statement": statement,
            "witness": dict(witness or {}),
        }
        bound = self._bind_request(
            request, operation="counterexample", payload=payload, **binding
        )
        started = time.time()
        try:
            from .formal_counterexamples import (
                CounterexampleBindings,
                CounterexampleKind,
                normalize_counterexample,
            )

            body = dict(bound.payload)
            kind_token = _text(body.get("kind") or kind, "kind")
            try:
                cex_kind = CounterexampleKind(kind_token)
            except ValueError:
                cex_kind = CounterexampleKind.GENERIC_FAILURE
            statement_text = _text(
                body.get("statement") or statement or "counterexample",
                "statement",
            )
            witness_body = body.get("witness") or witness or {}
            if not isinstance(witness_body, Mapping):
                raise LogicPlatformClientBindingError(
                    "witness must be a mapping; fail closed"
                )
            raw_cex: dict[str, Any] = {
                "kind": cex_kind.value,
                "summary": statement_text,
                "violated_property": statement_text,
                "property_class": "client_counterexample",
                "task_id": bound.task_id,
                "plan_id": bound.plan_id,
                "tree_id": bound.tree_id,
            }
            # Promote common witness shapes to top-level keys the normalizer
            # already understands (assignment/model/trace/etc.).
            raw_cex.update(dict(witness_body))
            if "assignment" not in raw_cex and "model" not in raw_cex:
                raw_cex["assignment"] = dict(witness_body)
            counterexample = normalize_counterexample(
                raw_cex,
                kind=cex_kind,
                bindings=CounterexampleBindings(
                    task_ids=(bound.task_id,),
                    plan_ids=(bound.plan_id,),
                    tree_ids=(bound.tree_id,),
                    policy_ids=(bound.policy_id,),
                ),
                property_class="client_counterexample",
                violated_property=statement_text,
                summary=statement_text,
            )
            return self._result(
                operation="counterexample",
                status=ClientOperationStatus.SUCCEEDED,
                request=bound,
                result={
                    "counterexample": _to_dict(counterexample),
                    "is_proof": False,
                    "kind": cex_kind.value,
                },
                evidence_kind="candidate",
                authority_ceiling="candidate",
                duration_ms=int((time.time() - started) * 1000),
            )
        except LogicPlatformClientError:
            raise
        except Exception as error:
            return self._result(
                operation="counterexample",
                status=ClientOperationStatus.FAILED,
                request=bound,
                error=f"{type(error).__name__}: {error}",
                duration_ms=int((time.time() - started) * 1000),
            )

    def cache_freshness(
        self,
        request: LogicPlatformClientRequest | Mapping[str, Any] | None = None,
        *,
        cache_key: Mapping[str, Any] | Any | None = None,
        stored_at_unix_ms: int | None = None,
        ttl_ms: int | None = None,
        invalidated: bool = False,
        now_unix_ms: int | None = None,
        repository: Any | None = None,
        **binding: Any,
    ) -> LogicPlatformClientResult:
        """Evaluate cache freshness; stale/invalidated/unknown fail closed."""

        self._ensure_handshake(operation="cache_freshness")
        payload = {
            "cache_key": _to_dict(cache_key) if cache_key is not None else {},
            "stored_at_unix_ms": stored_at_unix_ms,
            "ttl_ms": ttl_ms,
            "invalidated": invalidated,
        }
        bound = self._bind_request(
            request, operation="cache_freshness", payload=payload, **binding
        )
        started = time.time()
        try:
            body = dict(bound.payload)
            if repository is not None and cache_key is not None:
                report = repository.freshness(cache_key)
                report_dict = _to_dict(report) or {}
                is_fresh = bool(getattr(report, "is_fresh", report_dict.get("is_fresh")))
                disposition = str(
                    getattr(report, "disposition", None)
                    or report_dict.get("disposition")
                    or ""
                )
                if "invalid" in disposition.lower():
                    freshness = CacheFreshnessStatus.INVALIDATED
                elif is_fresh:
                    freshness = CacheFreshnessStatus.CURRENT
                elif disposition.lower() in {"miss", ""}:
                    freshness = CacheFreshnessStatus.MISS
                else:
                    freshness = CacheFreshnessStatus.STALE
                return self._result(
                    operation="cache_freshness",
                    status=ClientOperationStatus.SUCCEEDED,
                    request=bound,
                    result={
                        "freshness": freshness.value,
                        "is_fresh": is_fresh,
                        "report": report_dict,
                        "source": "proof_repository",
                    },
                    cache_freshness=freshness,
                    duration_ms=int((time.time() - started) * 1000),
                )

            if bool(body.get("invalidated") or invalidated):
                freshness = CacheFreshnessStatus.INVALIDATED
                is_fresh = False
                reason = "explicitly invalidated"
            else:
                stored = body.get("stored_at_unix_ms")
                if stored is None:
                    stored = stored_at_unix_ms
                ttl = body.get("ttl_ms")
                if ttl is None:
                    ttl = ttl_ms
                now = now_unix_ms if now_unix_ms is not None else int(time.time() * 1000)
                if stored is None or ttl is None:
                    freshness = CacheFreshnessStatus.UNKNOWN
                    is_fresh = False
                    reason = "missing stored_at_unix_ms or ttl_ms"
                else:
                    stored_i = _optional_int(stored, "stored_at_unix_ms")
                    ttl_i = _optional_int(ttl, "ttl_ms")
                    assert stored_i is not None and ttl_i is not None
                    age = max(0, int(now) - stored_i)
                    if age > ttl_i:
                        freshness = CacheFreshnessStatus.STALE
                        is_fresh = False
                        reason = f"age_ms={age} exceeds ttl_ms={ttl_i}"
                    else:
                        freshness = CacheFreshnessStatus.CURRENT
                        is_fresh = True
                        reason = f"age_ms={age} within ttl_ms={ttl_i}"

            return self._result(
                operation="cache_freshness",
                status=ClientOperationStatus.SUCCEEDED,
                request=bound,
                result={
                    "freshness": freshness.value,
                    "is_fresh": is_fresh,
                    "reason": reason,
                    "cache_key": body.get("cache_key") or _to_dict(cache_key) or {},
                    "source": "client_local",
                },
                cache_freshness=freshness,
                duration_ms=int((time.time() - started) * 1000),
            )
        except LogicPlatformClientError:
            raise
        except Exception as error:
            return self._result(
                operation="cache_freshness",
                status=ClientOperationStatus.FAILED,
                request=bound,
                error=f"{type(error).__name__}: {error}",
                cache_freshness=CacheFreshnessStatus.UNKNOWN,
                duration_ms=int((time.time() - started) * 1000),
            )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "interface": self.interface,
            "version": self.version,
            "task_id": self.task_id,
            "goal_id": self.goal_id,
            "operations": list(self.operations),
            "typed_provider_operations": sorted(TYPED_PROVIDER_OPERATIONS),
            "required_request_bindings": list(_REQUIRED_REQUEST_BINDINGS),
            "handshaken": self.handshaken,
            "require_handshake": self._require_handshake,
            "adapter_interface": SUPERVISOR_LOGIC_PLATFORM_CLIENT_INTERFACE
            if self._adapter is None
            else self.adapter.interface,
        }


def get_logic_platform_client(
    **kwargs: Any,
) -> SupervisorLogicPlatformClient:
    """Construct a supervisor logic-platform client."""

    return SupervisorLogicPlatformClient(**kwargs)


__all__ = [
    "CLIENT_GOAL_ID",
    "CLIENT_OPERATIONS",
    "CLIENT_REQUEST_SCHEMA",
    "CLIENT_RESULT_SCHEMA",
    "CLIENT_SCHEMA_VERSION",
    "CLIENT_TASK_ID",
    "CacheFreshnessStatus",
    "ClientOperationStatus",
    "LogicPlatformClientAuthorityError",
    "LogicPlatformClientBindingError",
    "LogicPlatformClientError",
    "LogicPlatformClientHandshakeError",
    "LogicPlatformClientRequest",
    "LogicPlatformClientResult",
    "SUPERVISOR_LOGIC_PLATFORM_CLIENT_INTERFACE",
    "SUPERVISOR_LOGIC_PLATFORM_CLIENT_VERSION",
    "SupervisorLogicPlatformClient",
    "TYPED_PROVIDER_OPERATIONS",
    "check_authority_overclaim",
    "get_logic_platform_client",
]
