"""SupervisorLogicPlatformClient@1 — lazy supervisor-side logic platform client.

One handshake + typed invocation surface for the datasets logic platform.
Importing this module never loads ``ipfs_datasets_py``; datasets packages are
reached only on explicit boundary calls (handshake, catalog, formalization,
slice/obligation/plan, capability, typed provider ops, receipts,
counterexamples, cache freshness).

The client does not create another supervisor, does not reimplement family
provers, and never upgrades provider success into proof authority (LPC-032).
Receipt admission (ten-point gate) is owned by LPC-111.
"""

from __future__ import annotations

import importlib
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Final

from .canonical_logic_adapter import (
    SUPERVISOR_CANONICAL_LOGIC_ADAPTER_INTERFACE,
    SupervisorCanonicalLogicAdapter,
    get_canonical_logic_adapter,
)
from .formal_verification_capabilities import ProofProviderOperation
from .formal_verification_contracts import (
    AssuranceLevel,
    EvidenceFreshness,
    EvidenceKind,
    ResourceBudget,
)
from .formal_verification_provider import (
    CancellationToken,
    ProviderFailure,
    ProviderFailureCode,
    ProviderRequest,
    ProviderResponse,
)
from .logic_provider_contract import SupervisorLogicProviderFacade


SUPERVISOR_LOGIC_PLATFORM_CLIENT_INTERFACE: Final = (
    "SupervisorLogicPlatformClient@1"
)
SUPERVISOR_LOGIC_PLATFORM_CLIENT_VERSION: Final = "1.0.0"
CLIENT_SCHEMA_VERSION: Final = (
    "ipfs_accelerate_py/agent-supervisor/logic-platform-client@1"
)
CLIENT_RESULT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/logic-platform-client-result@1"
)
CLIENT_REQUEST_CONTEXT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/logic-platform-client-request@1"
)
CLIENT_TASK_ID: Final = "LPC-110"
CLIENT_GOAL_ID: Final = "LPC-G110"

CANONICAL_MANIFEST_MODULE: Final = "ipfs_datasets_py.logic.platform.manifest"
CANONICAL_CATALOG_MODULE: Final = (
    "ipfs_datasets_py.logic.families.canonical_catalog"
)
CANONICAL_CACHE_MODULE: Final = "ipfs_datasets_py.logic.backends.cache_protocol"
CANONICAL_VERIFICATION_API_MODULE: Final = (
    "ipfs_datasets_py.logic.verification_api"
)

# Provider-protocol operations that cross SupervisorLogicProviderFacade.
_PROVIDER_OPERATIONS: Final[frozenset[str]] = frozenset(
    {
        ProofProviderOperation.CAPABILITY.value,
        ProofProviderOperation.TRANSLATE.value,
        ProofProviderOperation.PROVE.value,
        ProofProviderOperation.RECONSTRUCT.value,
        ProofProviderOperation.VERIFY.value,
        ProofProviderOperation.ATTEST.value,
    }
)

# Closed authority lattice for client non-overclaim checks.
_AUTHORITY_RANK: Final[Mapping[str, int]] = MappingProxyType(
    {
        AssuranceLevel.UNVERIFIED.value: 0,
        "unknown": 0,
        AssuranceLevel.CANDIDATE.value: 1,
        "advisory": 1,
        "simulated": 1,
        AssuranceLevel.SOLVER_CHECKED.value: 2,
        AssuranceLevel.KERNEL_VERIFIED.value: 3,
        AssuranceLevel.ATTESTED.value: 4,
    }
)

# Evidence kinds that may never claim kernel-or-above authority alone.
_NON_KERNEL_EVIDENCE: Final[frozenset[str]] = frozenset(
    {
        EvidenceKind.UNKNOWN.value,
        EvidenceKind.LLM_OUTPUT.value,
        EvidenceKind.ATP_CANDIDATE.value,
        EvidenceKind.SMT_CANDIDATE.value,
        EvidenceKind.SOLVER_RESULT.value,
        EvidenceKind.TEST_RESULT.value,
        EvidenceKind.STATIC_ANALYSIS.value,
        EvidenceKind.CACHE_ENTRY.value,
        "simulated",
        "advisory",
        "candidate",
    }
)


class LogicPlatformClientError(RuntimeError):
    """Raised when a client request is structurally invalid or overclaims."""


class ClientOperation(str, Enum):
    """Closed operation vocabulary for SupervisorLogicPlatformClient@1."""

    HANDSHAKE = "handshake"
    CATALOG = "catalog"
    FORMALIZE = "formalize"
    SLICE = "slice"
    OBLIGATION = "obligation"
    PLAN = "plan"
    CAPABILITY = "capability"
    TRANSLATE = "translate"
    PROVE = "prove"
    RECONSTRUCT = "reconstruct"
    VERIFY = "verify"
    ATTEST = "attest"
    RECEIPTS = "receipts"
    COUNTEREXAMPLES = "counterexamples"
    CACHE_FRESHNESS = "cache_freshness"


def _token(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise LogicPlatformClientError(f"{field_name} must be a non-empty string")
    token = value.strip()
    if not token:
        raise LogicPlatformClientError(f"{field_name} must be a non-empty string")
    return token


def _optional_token(value: Any, *, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise LogicPlatformClientError(f"{field_name} must be a string or null")
    token = value.strip()
    return token or None


def _json_object(value: Any, *, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise LogicPlatformClientError(f"{field_name} must be an object")
    if not all(isinstance(key, str) for key in value):
        raise LogicPlatformClientError(f"{field_name} keys must be strings")
    return dict(value)


def _authority_rank(value: Any) -> int:
    token = str(getattr(value, "value", value) or "unknown").strip().lower()
    if token not in _AUTHORITY_RANK:
        raise LogicPlatformClientError(
            f"unknown authority ceiling: {token!r}"
        )
    return _AUTHORITY_RANK[token]


def _normalize_authority(value: Any) -> str:
    token = str(getattr(value, "value", value) or "unknown").strip().lower()
    if not token:
        return AssuranceLevel.UNVERIFIED.value
    # Accept AssuranceLevel members and a few client-local aliases.
    if token in _AUTHORITY_RANK:
        if token in {"unknown", "advisory", "simulated"}:
            return (
                AssuranceLevel.UNVERIFIED.value
                if token == "unknown"
                else AssuranceLevel.CANDIDATE.value
            )
        return token
    try:
        return AssuranceLevel(token).value
    except ValueError as error:
        raise LogicPlatformClientError(
            f"unknown authority ceiling: {token!r}"
        ) from error


def _normalize_evidence_kind(value: Any | None) -> str | None:
    if value is None:
        return None
    token = str(getattr(value, "value", value)).strip().lower()
    if not token:
        return None
    try:
        return EvidenceKind(token).value
    except ValueError:
        # Residual / advisory labels stay as lower-case tokens; overclaim is
        # enforced against authority ceilings, not by rejecting unknown kinds.
        return token


def _resource_budget(
    value: ResourceBudget | Mapping[str, Any] | None,
) -> ResourceBudget:
    if value is None:
        return ResourceBudget()
    if isinstance(value, ResourceBudget):
        return value
    if isinstance(value, Mapping):
        return ResourceBudget.from_dict(value)
    raise LogicPlatformClientError(
        "resource_budget must be a ResourceBudget or object"
    )


def _operation(value: ClientOperation | str) -> ClientOperation:
    try:
        return ClientOperation(str(getattr(value, "value", value)))
    except ValueError as error:
        raise LogicPlatformClientError(
            f"unsupported client operation: {value!r}"
        ) from error


@dataclass(frozen=True, slots=True)
class ClientRequestContext:
    """Immutable binding for every non-handshake client call.

    Missing or contradictory bindings fail closed; they never soft-succeed.
    """

    task_id: str
    repository_tree_id: str
    policy_id: str
    resource_budget: ResourceBudget = field(default_factory=ResourceBudget)
    network_allowed: bool = False
    cancellation: CancellationToken | None = None
    cancelled: bool = False
    deadline_unix_ms: int | None = None
    correlation_id: str = ""
    plan_id: str | None = None
    policy_revision: str = ""
    evidence_kind: str | None = None
    authority_ceiling: str = AssuranceLevel.UNVERIFIED.value
    schema_version: str = CLIENT_REQUEST_CONTEXT_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", _token(self.task_id, field_name="task_id"))
        object.__setattr__(
            self,
            "repository_tree_id",
            _token(self.repository_tree_id, field_name="repository_tree_id"),
        )
        object.__setattr__(
            self, "policy_id", _token(self.policy_id, field_name="policy_id")
        )
        object.__setattr__(
            self,
            "policy_revision",
            (
                self.policy_revision.strip()
                if isinstance(self.policy_revision, str)
                else ""
            ),
        )
        plan_id = _optional_token(self.plan_id, field_name="plan_id")
        object.__setattr__(self, "plan_id", plan_id)
        object.__setattr__(self, "resource_budget", _resource_budget(self.resource_budget))
        if not isinstance(self.network_allowed, bool):
            raise LogicPlatformClientError("network_allowed must be a boolean")
        if not isinstance(self.cancelled, bool):
            raise LogicPlatformClientError("cancelled must be a boolean")
        if self.cancellation is not None and not hasattr(
            self.cancellation, "is_cancelled"
        ):
            raise LogicPlatformClientError(
                "cancellation must provide is_cancelled()"
            )
        if self.deadline_unix_ms is not None and (
            isinstance(self.deadline_unix_ms, bool)
            or not isinstance(self.deadline_unix_ms, int)
            or self.deadline_unix_ms < 0
        ):
            raise LogicPlatformClientError(
                "deadline_unix_ms must be a non-negative integer or null"
            )
        correlation = self.correlation_id
        if not isinstance(correlation, str):
            raise LogicPlatformClientError("correlation_id must be a string")
        correlation = correlation.strip()
        if not correlation:
            correlation = f"corr:{uuid.uuid4().hex}"
        object.__setattr__(self, "correlation_id", correlation)
        object.__setattr__(
            self,
            "evidence_kind",
            _normalize_evidence_kind(self.evidence_kind),
        )
        object.__setattr__(
            self,
            "authority_ceiling",
            _normalize_authority(self.authority_ceiling),
        )
        if self.schema_version != CLIENT_REQUEST_CONTEXT_SCHEMA:
            raise LogicPlatformClientError(
                f"unsupported request context schema: {self.schema_version!r}"
            )
        # Network policy: default deny; budget cannot grant network unless
        # the context also allows it.
        if self.network_allowed is False and self.resource_budget.network_allowed:
            raise LogicPlatformClientError(
                "resource_budget.network_allowed cannot exceed context network policy"
            )
        # Evidence kind cannot claim kernel authority without a kernel-grade kind.
        if (
            self.evidence_kind in _NON_KERNEL_EVIDENCE
            and _authority_rank(self.authority_ceiling)
            >= _authority_rank(AssuranceLevel.KERNEL_VERIFIED.value)
        ):
            raise LogicPlatformClientError(
                "evidence_kind cannot support the requested authority ceiling"
            )

    @property
    def is_cancelled(self) -> bool:
        if self.cancelled:
            return True
        if self.cancellation is not None and bool(self.cancellation.is_cancelled()):
            return True
        return False

    @property
    def is_expired(self) -> bool:
        return (
            self.deadline_unix_ms is not None
            and int(time.time() * 1000) >= self.deadline_unix_ms
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "repository_tree_id": self.repository_tree_id,
            "policy_id": self.policy_id,
            "policy_revision": self.policy_revision,
            "plan_id": self.plan_id,
            "resource_budget": self.resource_budget.to_dict(),
            "network_allowed": self.network_allowed,
            "cancelled": self.is_cancelled,
            "deadline_unix_ms": self.deadline_unix_ms,
            "correlation_id": self.correlation_id,
            "evidence_kind": self.evidence_kind,
            "authority_ceiling": self.authority_ceiling,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ClientRequestContext:
        if not isinstance(payload, Mapping):
            raise LogicPlatformClientError("request context must be an object")
        return cls(
            task_id=str(payload.get("task_id", "")),
            repository_tree_id=str(payload.get("repository_tree_id", "")),
            policy_id=str(payload.get("policy_id", "")),
            policy_revision=str(payload.get("policy_revision", "") or ""),
            plan_id=payload.get("plan_id"),
            resource_budget=_resource_budget(payload.get("resource_budget")),
            network_allowed=bool(payload.get("network_allowed", False)),
            cancelled=bool(payload.get("cancelled", False)),
            deadline_unix_ms=payload.get("deadline_unix_ms"),
            correlation_id=str(payload.get("correlation_id", "") or ""),
            evidence_kind=payload.get("evidence_kind"),
            authority_ceiling=payload.get(
                "authority_ceiling", AssuranceLevel.UNVERIFIED.value
            ),
            schema_version=str(
                payload.get("schema_version", CLIENT_REQUEST_CONTEXT_SCHEMA)
            ),
        )


@dataclass(frozen=True, slots=True)
class ClientResult:
    """Typed client envelope: ok/error without success⇒proof promotion."""

    operation: ClientOperation | str
    ok: bool
    payload: Mapping[str, Any] | None = None
    error: Mapping[str, Any] | None = None
    residual_identity: Mapping[str, Any] = field(default_factory=dict)
    authority_ceiling: str = AssuranceLevel.UNVERIFIED.value
    freshness: Mapping[str, Any] = field(default_factory=dict)
    simulated: bool = False
    correlation_id: str = ""
    request_id: str = ""
    provider_response: ProviderResponse | None = None
    schema_version: str = CLIENT_RESULT_SCHEMA

    def __post_init__(self) -> None:
        operation = _operation(self.operation)
        object.__setattr__(self, "operation", operation)
        if not isinstance(self.ok, bool):
            raise LogicPlatformClientError("ok must be a boolean")
        if not isinstance(self.simulated, bool):
            raise LogicPlatformClientError("simulated must be a boolean")
        payload = None if self.payload is None else _json_object(
            self.payload, field_name="payload"
        )
        error = None if self.error is None else _json_object(
            self.error, field_name="error"
        )
        if self.ok and error is not None:
            raise LogicPlatformClientError("successful result cannot carry error")
        if not self.ok and error is None:
            raise LogicPlatformClientError("failed result requires error")
        object.__setattr__(self, "payload", payload)
        object.__setattr__(self, "error", error)
        object.__setattr__(
            self,
            "residual_identity",
            MappingProxyType(
                _json_object(self.residual_identity, field_name="residual_identity")
            ),
        )
        object.__setattr__(
            self,
            "freshness",
            MappingProxyType(_json_object(self.freshness, field_name="freshness")),
        )
        object.__setattr__(
            self, "authority_ceiling", _normalize_authority(self.authority_ceiling)
        )
        if self.schema_version != CLIENT_RESULT_SCHEMA:
            raise LogicPlatformClientError(
                f"unsupported client result schema: {self.schema_version!r}"
            )
        # Simulated / advisory results cannot claim kernel authority.
        if self.simulated and _authority_rank(self.authority_ceiling) >= _authority_rank(
            AssuranceLevel.KERNEL_VERIFIED.value
        ):
            raise LogicPlatformClientError(
                "simulated results cannot claim kernel authority"
            )

    @property
    def semantic_verdict(self) -> str:
        """Semantic verdict is never inferred from operation success alone."""

        if not self.ok:
            code = ""
            if isinstance(self.error, Mapping):
                code = str(self.error.get("code", "") or "")
            if code in {"cancelled", ProviderFailureCode.CANCELLED.value}:
                return "cancelled"
            if code in {"timed_out", ProviderFailureCode.TIMED_OUT.value}:
                return "error"
            return "error"
        if self.payload is None:
            return "unknown"
        # Never promote provider-claimed authority or ok=True into proved.
        explicit = self.payload.get("semantic_verdict")
        if explicit is not None:
            return str(getattr(explicit, "value", explicit))
        return "unknown"

    @property
    def operation_status(self) -> str:
        if not self.ok:
            if isinstance(self.error, Mapping):
                return str(self.error.get("code", "failed") or "failed")
            return "failed"
        return "succeeded"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "interface": SUPERVISOR_LOGIC_PLATFORM_CLIENT_INTERFACE,
            "operation": self.operation.value
            if isinstance(self.operation, ClientOperation)
            else str(self.operation),
            "ok": self.ok,
            "payload": None if self.payload is None else dict(self.payload),
            "error": None if self.error is None else dict(self.error),
            "residual_identity": dict(self.residual_identity),
            "authority_ceiling": self.authority_ceiling,
            "freshness": dict(self.freshness),
            "simulated": self.simulated,
            "correlation_id": self.correlation_id,
            "request_id": self.request_id,
            "operation_status": self.operation_status,
            "semantic_verdict": self.semantic_verdict,
            "provider_response": (
                self.provider_response.to_dict()
                if self.provider_response is not None
                and hasattr(self.provider_response, "to_dict")
                else None
            ),
        }

    @classmethod
    def success(
        cls,
        operation: ClientOperation | str,
        payload: Mapping[str, Any] | None = None,
        *,
        residual_identity: Mapping[str, Any] | None = None,
        authority_ceiling: str = AssuranceLevel.UNVERIFIED.value,
        freshness: Mapping[str, Any] | None = None,
        simulated: bool = False,
        correlation_id: str = "",
        request_id: str = "",
        provider_response: ProviderResponse | None = None,
    ) -> ClientResult:
        return cls(
            operation=operation,
            ok=True,
            payload=payload or {},
            residual_identity=residual_identity or {},
            authority_ceiling=authority_ceiling,
            freshness=freshness or {"status": EvidenceFreshness.CURRENT.value},
            simulated=simulated,
            correlation_id=correlation_id,
            request_id=request_id,
            provider_response=provider_response,
        )

    @classmethod
    def failure(
        cls,
        operation: ClientOperation | str,
        code: str,
        message: str,
        *,
        details: Mapping[str, Any] | None = None,
        residual_identity: Mapping[str, Any] | None = None,
        authority_ceiling: str = AssuranceLevel.UNVERIFIED.value,
        freshness: Mapping[str, Any] | None = None,
        simulated: bool = False,
        correlation_id: str = "",
        request_id: str = "",
        provider_response: ProviderResponse | None = None,
        retryable: bool = False,
    ) -> ClientResult:
        return cls(
            operation=operation,
            ok=False,
            error={
                "code": str(code),
                "message": str(message)[:4096],
                "retryable": bool(retryable),
                "details": dict(details or {}),
            },
            residual_identity=residual_identity or {},
            authority_ceiling=authority_ceiling,
            freshness=freshness or {"status": EvidenceFreshness.UNKNOWN.value},
            simulated=simulated,
            correlation_id=correlation_id,
            request_id=request_id,
            provider_response=provider_response,
        )


class SupervisorLogicPlatformClient:
    """Lazy supervisor-side client for the datasets logic platform.

    Interface: ``SupervisorLogicPlatformClient@1``.

    Declared identity is stable before any datasets package is loaded.  The
    client never owns scheduling, merge, or daemon loops.
    """

    interface: Final = SUPERVISOR_LOGIC_PLATFORM_CLIENT_INTERFACE
    version: Final = SUPERVISOR_LOGIC_PLATFORM_CLIENT_VERSION
    schema_version: Final = CLIENT_SCHEMA_VERSION
    task_id: Final = CLIENT_TASK_ID
    goal_id: Final = CLIENT_GOAL_ID

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
        self._import = module_importer or importlib.import_module
        self._require_handshake = bool(require_handshake)
        self._handshake_result: Any | None = None
        self._manifest: Any | None = None
        self._load_lock = threading.Lock()
        self._datasets_loaded = False

    # ------------------------------------------------------------------
    # Identity / lazy state
    # ------------------------------------------------------------------

    @property
    def loaded_datasets(self) -> bool:
        """Whether any datasets package has been imported through this client."""

        return self._datasets_loaded

    def datasets_import_is_lazy(self) -> bool:
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "interface": self.interface,
            "version": self.version,
            "task_id": self.task_id,
            "goal_id": self.goal_id,
            "adapter_interface": SUPERVISOR_CANONICAL_LOGIC_ADAPTER_INTERFACE,
            "require_handshake": self._require_handshake,
            "handshaken": self._handshake_result is not None,
            "datasets_loaded": self._datasets_loaded,
            "operations": [op.value for op in ClientOperation],
        }

    def _lazy_import(self, module_name: str) -> Any:
        module = self._import(module_name)
        if module_name.startswith("ipfs_datasets_py"):
            self._datasets_loaded = True
        return module

    def _get_adapter(self) -> SupervisorCanonicalLogicAdapter:
        if self._adapter is None:
            self._adapter = get_canonical_logic_adapter()
        return self._adapter

    def _get_provider_facade(self) -> SupervisorLogicProviderFacade:
        if self._provider_facade is None:
            raise LogicPlatformClientError(
                "provider_facade is required for typed provider invocation"
            )
        return self._provider_facade

    def _residual_identity(
        self,
        *,
        context: ClientRequestContext | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        identity: dict[str, Any] = {
            "client_interface": self.interface,
            "adapter_interface": SUPERVISOR_CANONICAL_LOGIC_ADAPTER_INTERFACE,
        }
        if context is not None:
            identity.update(
                {
                    "task_id": context.task_id,
                    "repository_tree_id": context.repository_tree_id,
                    "policy_id": context.policy_id,
                    "policy_revision": context.policy_revision,
                    "plan_id": context.plan_id,
                    "correlation_id": context.correlation_id,
                    "authority_ceiling": context.authority_ceiling,
                    "evidence_kind": context.evidence_kind,
                }
            )
        if extra:
            identity.update(dict(extra))
        return identity

    def _ensure_handshaken(self) -> None:
        if not self._require_handshake:
            return
        if self._handshake_result is None:
            raise LogicPlatformClientError(
                "handshake is required before semantic client operations"
            )
        compatible = getattr(self._handshake_result, "compatible", None)
        if compatible is False:
            raise LogicPlatformClientError(
                "platform handshake is incompatible; refuse semantic operations"
            )

    def _guard_context(
        self,
        context: ClientRequestContext,
        *,
        operation: ClientOperation,
        require_plan: bool = False,
    ) -> ClientResult | None:
        if not isinstance(context, ClientRequestContext):
            raise LogicPlatformClientError(
                "context must be a ClientRequestContext"
            )
        residual = self._residual_identity(context=context)
        if context.is_cancelled:
            return ClientResult.failure(
                operation,
                "cancelled",
                "client request was cancelled",
                residual_identity=residual,
                authority_ceiling=context.authority_ceiling,
                correlation_id=context.correlation_id,
            )
        if context.is_expired:
            return ClientResult.failure(
                operation,
                "timed_out",
                "client request deadline has expired",
                residual_identity=residual,
                authority_ceiling=context.authority_ceiling,
                correlation_id=context.correlation_id,
            )
        if require_plan and not context.plan_id:
            return ClientResult.failure(
                operation,
                "missing_plan",
                "plan_id is required for plan-scoped operations",
                residual_identity=residual,
                authority_ceiling=context.authority_ceiling,
                correlation_id=context.correlation_id,
            )
        return None

    # ------------------------------------------------------------------
    # Handshake (LPC-100)
    # ------------------------------------------------------------------

    def handshake(
        self,
        requirements: Any | None = None,
        *,
        manifest: Any | None = None,
    ) -> ClientResult:
        """First semantic step: LogicPlatformManifest@1 compatibility check."""

        try:
            manifest_mod = self._lazy_import(CANONICAL_MANIFEST_MODULE)
        except Exception as error:
            return ClientResult.failure(
                ClientOperation.HANDSHAKE,
                "unavailable",
                f"logic platform manifest unavailable: {type(error).__name__}",
                retryable=True,
                residual_identity=self._residual_identity(),
            )

        try:
            if manifest is not None:
                platform_manifest = manifest
            else:
                platform_manifest = manifest_mod.build_logic_platform_manifest()
            result = manifest_mod.handshake(
                requirements, manifest=platform_manifest
            )
        except Exception as error:
            return ClientResult.failure(
                ClientOperation.HANDSHAKE,
                "malformed_request",
                f"handshake failed structurally: {str(error)[:512]}",
                residual_identity=self._residual_identity(),
            )

        self._handshake_result = result
        self._manifest = getattr(result, "manifest", platform_manifest)

        # Confirm this client interface is listed as compatible (informational
        # when requirements did not pin adapters).
        adapters = tuple(
            getattr(self._manifest, "compatible_adapter_versions", ()) or ()
        )
        listed = SUPERVISOR_LOGIC_PLATFORM_CLIENT_INTERFACE in adapters
        payload = {
            "compatible": bool(getattr(result, "compatible", False)),
            "incompatibilities": [
                item.to_dict() if hasattr(item, "to_dict") else dict(item)
                for item in (getattr(result, "incompatibilities", ()) or ())
            ],
            "manifest": (
                self._manifest.to_dict()
                if hasattr(self._manifest, "to_dict")
                else {}
            ),
            "client_interface_listed": listed,
            "requires_git": bool(
                getattr(self._manifest, "requires_git", lambda: False)()
            ),
            "requires_sibling_repos": bool(
                getattr(self._manifest, "requires_sibling_repos", lambda: False)()
            ),
            "requires_repository_layout": bool(
                getattr(
                    self._manifest, "requires_repository_layout", lambda: False
                )()
            ),
        }
        if not payload["compatible"]:
            return ClientResult.failure(
                ClientOperation.HANDSHAKE,
                "incompatible",
                "logic platform handshake reported incompatibilities",
                details={"handshake": payload},
                residual_identity=self._residual_identity(
                    extra={"client_interface_listed": listed}
                ),
                freshness={"status": EvidenceFreshness.CURRENT.value},
            )
        return ClientResult.success(
            ClientOperation.HANDSHAKE,
            payload,
            residual_identity=self._residual_identity(
                extra={"client_interface_listed": listed}
            ),
            authority_ceiling=AssuranceLevel.UNVERIFIED.value,
            freshness={"status": EvidenceFreshness.CURRENT.value},
        )

    # ------------------------------------------------------------------
    # Catalog
    # ------------------------------------------------------------------

    def catalog(
        self,
        context: ClientRequestContext,
        *,
        payload: Mapping[str, Any] | None = None,
    ) -> ClientResult:
        """Read the sealed catalog / content root (declaration only)."""

        self._ensure_handshaken()
        guarded = self._guard_context(context, operation=ClientOperation.CATALOG)
        if guarded is not None:
            return guarded
        _ = _json_object(payload, field_name="payload")
        residual = self._residual_identity(context=context)
        try:
            catalog_mod = self._lazy_import(CANONICAL_CATALOG_MODULE)
            snapshot = getattr(catalog_mod, "DEFAULT_CANONICAL_CATALOG_SNAPSHOT")
        except Exception as error:
            return ClientResult.failure(
                ClientOperation.CATALOG,
                "unavailable",
                f"canonical catalog unavailable: {type(error).__name__}",
                residual_identity=residual,
                authority_ceiling=context.authority_ceiling,
                correlation_id=context.correlation_id,
                retryable=True,
            )

        body: dict[str, Any] = {
            "content_root": getattr(snapshot, "content_root", ""),
            "content_digest": getattr(snapshot, "content_digest", ""),
            "interface": getattr(
                snapshot,
                "INTERFACE",
                getattr(snapshot, "interface", "CanonicalLogicCatalogSnapshot@1"),
            ),
            # Catalog presence never implies provider availability.
            "provider_availability_claimed": False,
            "executable": False,
        }
        if hasattr(snapshot, "to_dict"):
            body["snapshot"] = snapshot.to_dict()
        return ClientResult.success(
            ClientOperation.CATALOG,
            body,
            residual_identity=residual,
            authority_ceiling=AssuranceLevel.UNVERIFIED.value,
            correlation_id=context.correlation_id,
            freshness={
                "status": EvidenceFreshness.CURRENT.value,
                "catalog_digest": body.get("content_digest", ""),
            },
        )

    # ------------------------------------------------------------------
    # Formalization / slice / obligation / plan
    # ------------------------------------------------------------------

    def _platform_write(
        self,
        *,
        operation: ClientOperation,
        context: ClientRequestContext,
        payload: Mapping[str, Any] | None,
        require_plan: bool = False,
        kind: str,
    ) -> ClientResult:
        self._ensure_handshaken()
        guarded = self._guard_context(
            context, operation=operation, require_plan=require_plan
        )
        if guarded is not None:
            return guarded
        body = _json_object(payload, field_name="payload")
        residual = self._residual_identity(context=context)

        # Project residual supervisor vocabulary through the lazy adapter so
        # unknown values fail closed before any write path.
        adapter = self._get_adapter()
        projections: dict[str, Any] = {}
        if "logic_family" in body or "family" in body:
            family = body.get("logic_family", body.get("family"))
            try:
                projections["logic_family"] = adapter.project_analysis_family(
                    family
                ).to_dict()
            except Exception as error:
                return ClientResult.failure(
                    operation,
                    "unknown_vocabulary",
                    f"residual vocabulary rejected: {str(error)[:512]}",
                    residual_identity=residual,
                    authority_ceiling=context.authority_ceiling,
                    correlation_id=context.correlation_id,
                )
        if "property_kind" in body:
            try:
                projections["property_kind"] = adapter.project_property_kind(
                    body["property_kind"]
                ).to_dict()
            except Exception as error:
                return ClientResult.failure(
                    operation,
                    "unknown_vocabulary",
                    f"residual vocabulary rejected: {str(error)[:512]}",
                    residual_identity=residual,
                    authority_ceiling=context.authority_ceiling,
                    correlation_id=context.correlation_id,
                )

        artifact_id = str(
            body.get("artifact_id")
            or body.get("slice_id")
            or body.get("obligation_id")
            or body.get("plan_id")
            or f"{kind}:{uuid.uuid4().hex}"
        )
        result_payload = {
            "kind": kind,
            "artifact_id": artifact_id,
            "admitted": True,
            "authority_ceiling": context.authority_ceiling,
            "bindings": {
                "task_id": context.task_id,
                "repository_tree_id": context.repository_tree_id,
                "policy_id": context.policy_id,
                "plan_id": context.plan_id,
                "correlation_id": context.correlation_id,
            },
            "projections": projections,
            "request": body,
            # Write path admission is structural only; not proof.
            "semantic_verdict": "unknown",
            "proof_attempted": False,
            "proof_success": False,
        }
        return ClientResult.success(
            operation,
            result_payload,
            residual_identity=residual,
            authority_ceiling=context.authority_ceiling,
            correlation_id=context.correlation_id,
            request_id=artifact_id,
        )

    def formalize(
        self,
        context: ClientRequestContext,
        payload: Mapping[str, Any] | None = None,
    ) -> ClientResult:
        return self._platform_write(
            operation=ClientOperation.FORMALIZE,
            context=context,
            payload=payload,
            kind="formalization_artifact",
        )

    def slice(
        self,
        context: ClientRequestContext,
        payload: Mapping[str, Any] | None = None,
    ) -> ClientResult:
        return self._platform_write(
            operation=ClientOperation.SLICE,
            context=context,
            payload=payload,
            kind="domain_logic_slice",
        )

    def obligation(
        self,
        context: ClientRequestContext,
        payload: Mapping[str, Any] | None = None,
    ) -> ClientResult:
        return self._platform_write(
            operation=ClientOperation.OBLIGATION,
            context=context,
            payload=payload,
            kind="logic_obligation",
        )

    def plan(
        self,
        context: ClientRequestContext,
        payload: Mapping[str, Any] | None = None,
    ) -> ClientResult:
        return self._platform_write(
            operation=ClientOperation.PLAN,
            context=context,
            payload=payload,
            kind="goal_directed_proof_plan",
        )

    # ------------------------------------------------------------------
    # Typed provider invocation
    # ------------------------------------------------------------------

    def invoke(
        self,
        context: ClientRequestContext,
        operation: ClientOperation | str,
        payload: Mapping[str, Any] | None = None,
        *,
        request_id: str | None = None,
    ) -> ClientResult:
        """Typed platform operation dispatch (provider + helper ops)."""

        op = _operation(operation)
        if op is ClientOperation.HANDSHAKE:
            return self.handshake(payload)
        if op is ClientOperation.CATALOG:
            return self.catalog(context, payload=payload)
        if op is ClientOperation.FORMALIZE:
            return self.formalize(context, payload)
        if op is ClientOperation.SLICE:
            return self.slice(context, payload)
        if op is ClientOperation.OBLIGATION:
            return self.obligation(context, payload)
        if op is ClientOperation.PLAN:
            return self.plan(context, payload)
        if op is ClientOperation.RECEIPTS:
            return self.receipts(context, payload)
        if op is ClientOperation.COUNTEREXAMPLES:
            return self.counterexamples(context, payload)
        if op is ClientOperation.CACHE_FRESHNESS:
            return self.cache_freshness(context, payload)
        if op.value in _PROVIDER_OPERATIONS:
            return self._invoke_provider(
                context,
                op,
                payload,
                request_id=request_id,
            )
        raise LogicPlatformClientError(f"unsupported client operation: {op.value}")

    def _invoke_provider(
        self,
        context: ClientRequestContext,
        operation: ClientOperation,
        payload: Mapping[str, Any] | None,
        *,
        request_id: str | None = None,
    ) -> ClientResult:
        self._ensure_handshaken()
        guarded = self._guard_context(context, operation=operation)
        if guarded is not None:
            return guarded
        body = _json_object(payload, field_name="payload")
        residual = self._residual_identity(context=context)

        # Project residual vocabulary present in the payload.
        adapter = self._get_adapter()
        if "logic_family" in body:
            try:
                projection = adapter.project_analysis_family(body["logic_family"])
                body = dict(body)
                body["logic_family"] = projection.canonical_id
                body["logic_family_residual"] = projection.supervisor_id
            except Exception as error:
                return ClientResult.failure(
                    operation,
                    "unknown_vocabulary",
                    f"residual vocabulary rejected: {str(error)[:512]}",
                    residual_identity=residual,
                    authority_ceiling=context.authority_ceiling,
                    correlation_id=context.correlation_id,
                )

        # Caller cannot request a budget that exceeds the bound context.
        budget = context.resource_budget
        network_allowed = context.network_allowed and budget.network_allowed
        provider_request = ProviderRequest(
            operation=ProofProviderOperation(operation.value),
            payload=body,
            request_id=(
                request_id.strip()
                if isinstance(request_id, str) and request_id.strip()
                else f"lpc:{uuid.uuid4().hex}"
            ),
            resource_budget=budget,
            network_allowed=network_allowed,
            deadline_unix_ms=context.deadline_unix_ms,
        )

        try:
            facade = self._get_provider_facade()
        except LogicPlatformClientError as error:
            return ClientResult.failure(
                operation,
                "unavailable",
                str(error),
                residual_identity=residual,
                authority_ceiling=context.authority_ceiling,
                correlation_id=context.correlation_id,
                request_id=provider_request.request_id,
            )

        response = facade.invoke(
            provider_request, cancellation=context.cancellation
        )
        return self._provider_response_to_client_result(
            context=context,
            operation=operation,
            response=response,
            residual=residual,
        )

    def _provider_response_to_client_result(
        self,
        *,
        context: ClientRequestContext,
        operation: ClientOperation,
        response: ProviderResponse,
        residual: Mapping[str, Any],
    ) -> ClientResult:
        # LPC-032: operation success never becomes semantic proof.
        simulated = False
        if response.ok and isinstance(response.result, Mapping):
            simulated = bool(
                response.result.get("simulated")
                or response.result.get("simulation")
                or response.result.get("advisory")
            )
            # Strip any provider-claimed authority upgrades.
            safe_payload = dict(response.result)
            safe_payload.pop("provider_claimed_authority", None)
            safe_payload["operation_status"] = "succeeded"
            if "semantic_verdict" not in safe_payload:
                safe_payload["semantic_verdict"] = "unknown"
            # Authority ceiling is the min of context ceiling and non-proof floor
            # unless the payload honestly declares a lower ceiling.
            claimed = str(
                safe_payload.get("authority_ceiling")
                or context.authority_ceiling
            )
            try:
                claimed_norm = _normalize_authority(claimed)
            except LogicPlatformClientError:
                claimed_norm = AssuranceLevel.UNVERIFIED.value
            if simulated:
                ceiling = AssuranceLevel.CANDIDATE.value
            else:
                ceiling = (
                    claimed_norm
                    if _authority_rank(claimed_norm)
                    <= _authority_rank(context.authority_ceiling)
                    else context.authority_ceiling
                )
            # Kernel authority requires kernel-grade evidence kind.
            if (
                context.evidence_kind in _NON_KERNEL_EVIDENCE
                or context.evidence_kind is None
            ) and _authority_rank(ceiling) >= _authority_rank(
                AssuranceLevel.KERNEL_VERIFIED.value
            ):
                ceiling = AssuranceLevel.SOLVER_CHECKED.value
                safe_payload["authority_ceiling_reduced"] = True
            safe_payload["authority_ceiling"] = ceiling
            safe_payload["proof_success"] = False
            if "proof_attempted" not in safe_payload:
                safe_payload["proof_attempted"] = operation in {
                    ClientOperation.PROVE,
                    ClientOperation.RECONSTRUCT,
                    ClientOperation.VERIFY,
                    ClientOperation.ATTEST,
                }
            return ClientResult.success(
                operation,
                safe_payload,
                residual_identity=residual,
                authority_ceiling=ceiling,
                simulated=simulated,
                correlation_id=context.correlation_id,
                request_id=response.request_id,
                provider_response=response,
                freshness={
                    "status": EvidenceFreshness.CURRENT.value,
                    "source": "provider_response",
                },
            )

        error_code = "provider_error"
        error_message = "provider invocation failed"
        retryable = False
        details: dict[str, Any] = {}
        if response.error is not None:
            if isinstance(response.error, ProviderFailure):
                error_code = response.error.code.value
                error_message = response.error.message
                retryable = response.error.retryable
                details = dict(response.error.details)
            elif isinstance(response.error, Mapping):
                error_code = str(response.error.get("code", error_code))
                error_message = str(response.error.get("message", error_message))
                retryable = bool(response.error.get("retryable", False))
                details = dict(response.error.get("details") or {})
        return ClientResult.failure(
            operation,
            error_code,
            error_message,
            details=details,
            residual_identity=residual,
            authority_ceiling=context.authority_ceiling,
            correlation_id=context.correlation_id,
            request_id=response.request_id,
            provider_response=response,
            retryable=retryable,
        )

    def capability(
        self,
        context: ClientRequestContext,
        payload: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> ClientResult:
        return self._invoke_provider(
            context, ClientOperation.CAPABILITY, payload, **kwargs
        )

    def translate(
        self,
        context: ClientRequestContext,
        payload: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> ClientResult:
        return self._invoke_provider(
            context, ClientOperation.TRANSLATE, payload, **kwargs
        )

    def prove(
        self,
        context: ClientRequestContext,
        payload: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> ClientResult:
        return self._invoke_provider(
            context, ClientOperation.PROVE, payload, **kwargs
        )

    def reconstruct(
        self,
        context: ClientRequestContext,
        payload: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> ClientResult:
        return self._invoke_provider(
            context, ClientOperation.RECONSTRUCT, payload, **kwargs
        )

    def verify(
        self,
        context: ClientRequestContext,
        payload: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> ClientResult:
        return self._invoke_provider(
            context, ClientOperation.VERIFY, payload, **kwargs
        )

    def attest(
        self,
        context: ClientRequestContext,
        payload: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> ClientResult:
        return self._invoke_provider(
            context, ClientOperation.ATTEST, payload, **kwargs
        )

    # ------------------------------------------------------------------
    # Receipts / counterexamples / cache freshness
    # ------------------------------------------------------------------

    def receipts(
        self,
        context: ClientRequestContext,
        payload: Mapping[str, Any] | None = None,
    ) -> ClientResult:
        """Project receipt envelopes as untrusted until LPC-111 admission."""

        self._ensure_handshaken()
        guarded = self._guard_context(context, operation=ClientOperation.RECEIPTS)
        if guarded is not None:
            return guarded
        body = _json_object(payload, field_name="payload")
        residual = self._residual_identity(context=context)

        receipts_in = body.get("receipts", body.get("receipt"))
        items: list[Any]
        if receipts_in is None:
            items = []
        elif isinstance(receipts_in, Mapping):
            items = [receipts_in]
        elif isinstance(receipts_in, Sequence) and not isinstance(
            receipts_in, (str, bytes, bytearray)
        ):
            items = list(receipts_in)
        else:
            return ClientResult.failure(
                ClientOperation.RECEIPTS,
                "malformed_request",
                "receipts must be an object or list of objects",
                residual_identity=residual,
                authority_ceiling=context.authority_ceiling,
                correlation_id=context.correlation_id,
            )

        projected: list[dict[str, Any]] = []
        for index, item in enumerate(items):
            if not isinstance(item, Mapping):
                return ClientResult.failure(
                    ClientOperation.RECEIPTS,
                    "malformed_request",
                    f"receipts[{index}] must be an object",
                    residual_identity=residual,
                    authority_ceiling=context.authority_ceiling,
                    correlation_id=context.correlation_id,
                )
            entry = dict(item)
            simulated = bool(entry.get("simulated", False))
            ceiling = _normalize_authority(
                entry.get("authority_ceiling", context.authority_ceiling)
            )
            if simulated and _authority_rank(ceiling) >= _authority_rank(
                AssuranceLevel.KERNEL_VERIFIED.value
            ):
                ceiling = AssuranceLevel.CANDIDATE.value
            projected.append(
                {
                    "receipt": entry,
                    "admitted": False,  # LPC-111 owns admission
                    "trusted": False,
                    "simulated": simulated,
                    "authority_ceiling": ceiling,
                    "ten_point_gate": "deferred_to_lpc_111",
                }
            )

        return ClientResult.success(
            ClientOperation.RECEIPTS,
            {
                "receipts": projected,
                "count": len(projected),
                "admitted": False,
                "semantic_verdict": "unknown",
            },
            residual_identity=residual,
            authority_ceiling=context.authority_ceiling,
            correlation_id=context.correlation_id,
            simulated=any(item["simulated"] for item in projected),
        )

    def counterexamples(
        self,
        context: ClientRequestContext,
        payload: Mapping[str, Any] | None = None,
    ) -> ClientResult:
        """Project counterexample evidence with kind constraints."""

        self._ensure_handshaken()
        guarded = self._guard_context(
            context, operation=ClientOperation.COUNTEREXAMPLES
        )
        if guarded is not None:
            return guarded
        body = _json_object(payload, field_name="payload")
        residual = self._residual_identity(context=context)

        evidence_kind = _normalize_evidence_kind(
            body.get("evidence_kind", context.evidence_kind)
        )
        if evidence_kind is None:
            evidence_kind = EvidenceKind.SOLVER_RESULT.value

        raw = body.get("counterexamples", body.get("counterexample"))
        if raw is None:
            items: list[Any] = []
        elif isinstance(raw, Mapping):
            items = [raw]
        elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
            items = list(raw)
        else:
            return ClientResult.failure(
                ClientOperation.COUNTEREXAMPLES,
                "malformed_request",
                "counterexamples must be an object or list of objects",
                residual_identity=residual,
                authority_ceiling=context.authority_ceiling,
                correlation_id=context.correlation_id,
            )

        projected: list[dict[str, Any]] = []
        for index, item in enumerate(items):
            if not isinstance(item, Mapping):
                return ClientResult.failure(
                    ClientOperation.COUNTEREXAMPLES,
                    "malformed_request",
                    f"counterexamples[{index}] must be an object",
                    residual_identity=residual,
                    authority_ceiling=context.authority_ceiling,
                    correlation_id=context.correlation_id,
                )
            entry = dict(item)
            projected.append(
                {
                    "counterexample": entry,
                    "evidence_kind": evidence_kind,
                    "authority_ceiling": context.authority_ceiling,
                    "semantic_verdict": str(
                        entry.get("semantic_verdict", "disproved")
                    ),
                    "trusted": False,
                }
            )

        return ClientResult.success(
            ClientOperation.COUNTEREXAMPLES,
            {
                "counterexamples": projected,
                "count": len(projected),
                "evidence_kind": evidence_kind,
                "semantic_verdict": (
                    "disproved" if projected else "unknown"
                ),
            },
            residual_identity=residual,
            authority_ceiling=context.authority_ceiling,
            correlation_id=context.correlation_id,
        )

    def cache_freshness(
        self,
        context: ClientRequestContext,
        payload: Mapping[str, Any] | None = None,
    ) -> ClientResult:
        """Report cache freshness / invalidation signals.

        Datasets owns semantic key identity; the supervisor owns placement and
        single-flight.  This helper projects freshness without inventing keys.
        """

        self._ensure_handshaken()
        guarded = self._guard_context(
            context, operation=ClientOperation.CACHE_FRESHNESS
        )
        if guarded is not None:
            return guarded
        body = _json_object(payload, field_name="payload")
        residual = self._residual_identity(context=context)

        # Optional live key digest when datasets cache protocol is available.
        key_digest = str(body.get("cache_key_digest") or body.get("digest") or "")
        entry_digest = str(body.get("entry_digest") or "")
        declared_status = str(
            body.get("status") or body.get("freshness") or ""
        ).strip().lower()
        expected_tree = str(body.get("expected_tree_id") or "")
        expected_policy = str(body.get("expected_policy_id") or "")

        reasons: list[str] = []
        status = EvidenceFreshness.CURRENT.value

        if declared_status in {
            EvidenceFreshness.STALE.value,
            "invalid",
            "expired",
        }:
            status = EvidenceFreshness.STALE.value
            reasons.append(declared_status or "declared_stale")
        elif declared_status in {EvidenceFreshness.UNKNOWN.value, "unknown"}:
            status = EvidenceFreshness.UNKNOWN.value
            reasons.append("declared_unknown")

        if expected_tree and expected_tree != context.repository_tree_id:
            status = EvidenceFreshness.STALE.value
            reasons.append("tree_mismatch")
        if expected_policy and expected_policy != context.policy_id:
            status = EvidenceFreshness.STALE.value
            reasons.append("policy_mismatch")

        # When a structured key is provided, validate digest shape without
        # requiring datasets import when the caller already supplied digests.
        if not key_digest and body.get("cache_key") is not None:
            try:
                cache_mod = self._lazy_import(CANONICAL_CACHE_MODULE)
                key_obj = body["cache_key"]
                if hasattr(cache_mod, "VerificationCacheKey") and isinstance(
                    key_obj, Mapping
                ):
                    key = cache_mod.VerificationCacheKey.from_dict(key_obj)
                    key_digest = str(getattr(key, "digest", "") or "")
                elif hasattr(key_obj, "digest"):
                    key_digest = str(key_obj.digest)
            except Exception as error:
                return ClientResult.failure(
                    ClientOperation.CACHE_FRESHNESS,
                    "malformed_request",
                    f"cache key rejected: {str(error)[:512]}",
                    residual_identity=residual,
                    authority_ceiling=context.authority_ceiling,
                    correlation_id=context.correlation_id,
                )

        is_fresh = status == EvidenceFreshness.CURRENT.value
        result_payload = {
            "status": status,
            "is_fresh": is_fresh,
            "reasons": reasons,
            "cache_key_digest": key_digest,
            "entry_digest": entry_digest,
            "bindings": {
                "repository_tree_id": context.repository_tree_id,
                "policy_id": context.policy_id,
                "task_id": context.task_id,
            },
            # Freshness is not proof authority.
            "semantic_verdict": "unknown",
            "proof_success": False,
        }
        return ClientResult.success(
            ClientOperation.CACHE_FRESHNESS,
            result_payload,
            residual_identity=residual,
            authority_ceiling=context.authority_ceiling,
            correlation_id=context.correlation_id,
            freshness={
                "status": status,
                "is_fresh": is_fresh,
                "reasons": reasons,
                "cache_key_digest": key_digest,
            },
        )

    # ------------------------------------------------------------------
    # Vocabulary projection helpers (via LPC-090 adapter)
    # ------------------------------------------------------------------

    def project_residual(
        self,
        domain: str,
        value: Any,
    ) -> Mapping[str, Any]:
        """Project a residual supervisor value through the canonical adapter."""

        adapter = self._get_adapter()
        domain_token = _token(domain, field_name="domain")
        projectors = {
            "analysis_family": adapter.project_analysis_family,
            "logic_family": adapter.project_analysis_family,
            "property_kind": adapter.project_property_kind,
            "cache_scope": adapter.project_cache_scope,
        }
        projector = projectors.get(domain_token)
        if projector is None:
            raise LogicPlatformClientError(
                f"unsupported residual projection domain: {domain_token!r}"
            )
        projection = projector(value)
        return projection.to_dict() if hasattr(projection, "to_dict") else dict(
            projection
        )


_default_client: SupervisorLogicPlatformClient | None = None
_default_client_lock = threading.Lock()


def get_supervisor_logic_platform_client(
    **kwargs: Any,
) -> SupervisorLogicPlatformClient:
    """Return a process-local client, or a fresh one when overrides are supplied.

    Does not create a second supervisor runtime.
    """

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


def _clear_default_client_for_tests() -> None:
    """Test helper: drop the process-local client singleton."""

    global _default_client
    with _default_client_lock:
        _default_client = None


__all__ = [
    "CANONICAL_CACHE_MODULE",
    "CANONICAL_CATALOG_MODULE",
    "CANONICAL_MANIFEST_MODULE",
    "CANONICAL_VERIFICATION_API_MODULE",
    "CLIENT_GOAL_ID",
    "CLIENT_REQUEST_CONTEXT_SCHEMA",
    "CLIENT_RESULT_SCHEMA",
    "CLIENT_SCHEMA_VERSION",
    "CLIENT_TASK_ID",
    "ClientOperation",
    "ClientRequestContext",
    "ClientResult",
    "LogicPlatformClientError",
    "SUPERVISOR_LOGIC_PLATFORM_CLIENT_INTERFACE",
    "SUPERVISOR_LOGIC_PLATFORM_CLIENT_VERSION",
    "SupervisorLogicPlatformClient",
    "get_supervisor_logic_platform_client",
]
