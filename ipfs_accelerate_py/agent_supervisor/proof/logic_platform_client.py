"""SupervisorLogicPlatformClient@1 — lazy supervisor-side logic platform client.

LPC-110 provides one handshake + typed invocation surface for catalog access,
formalization, slice/obligation/plan creation, capability discovery, provider
invocation (including reconstruction and verification), receipt projection,
counterexamples, and cache freshness.

Importing this module never imports ``ipfs_datasets_py``.  Datasets packages
are loaded only for an explicit handshake or operation that needs them.
The supervisor still owns scheduling, isolation, resources, cancellation, and
admission policy; datasets still owns semantic identity.
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
from .formal_counterexamples import (
    CounterexampleBindings,
    CounterexampleKind,
    FormalCounterexample,
    normalize_counterexample,
)
from .formal_verification_capabilities import ProofProviderOperation
from .formal_verification_contracts import ResourceBudget
from .formal_verification_provider import (
    CancellationToken,
    ProviderRequest,
    ProviderResponse,
)
from .logic_provider_contract import SupervisorLogicProviderFacade
from .logic_translation_validation import TranslationValidationResult


# ---------------------------------------------------------------------------
# Interface / schema identities
# ---------------------------------------------------------------------------

SUPERVISOR_LOGIC_PLATFORM_CLIENT_INTERFACE: Final = (
    "SupervisorLogicPlatformClient@1"
)
LOGIC_PLATFORM_CLIENT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/logic-platform-client@1"
)
LOGIC_PLATFORM_CLIENT_REQUEST_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/logic-platform-client-request@1"
)
LOGIC_PLATFORM_CLIENT_RESULT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/logic-platform-client-result@1"
)
LOGIC_PLATFORM_CLIENT_VERSION: Final = "1.0.0"
LOGIC_PLATFORM_CLIENT_TASK_ID: Final = "LPC-110"
LOGIC_PLATFORM_CLIENT_GOAL_ID: Final = "LPC-G110"

MANIFEST_MODULE: Final = "ipfs_datasets_py.logic.platform.manifest"
CATALOG_MODULE: Final = "ipfs_datasets_py.logic.families.canonical_catalog"
ARTIFACTS_MODULE: Final = "ipfs_datasets_py.logic.formalization.artifacts_v3"
REQUESTS_V2_MODULE: Final = "ipfs_datasets_py.logic.backends.requests_v2"
PLAN_CONTRACTS_MODULE: Final = (
    "ipfs_datasets_py.logic.software_verification.tactician.contracts"
)
VERIFICATION_API_MODULE: Final = "ipfs_datasets_py.logic.verification_api"
NAMESPACES_MODULE: Final = "ipfs_datasets_py.logic.families.namespaces"

TYPED_PROVIDER_OPERATIONS: Final[frozenset[str]] = frozenset(
    {
        ProofProviderOperation.CAPABILITY.value,
        ProofProviderOperation.TRANSLATE.value,
        ProofProviderOperation.PROVE.value,
        ProofProviderOperation.RECONSTRUCT.value,
        ProofProviderOperation.VERIFY.value,
        ProofProviderOperation.ATTEST.value,
    }
)

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

# Evidence kind → maximum authority ceiling (mirrors datasets RequestAuthority).
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
        "reconstruction": "reconstruction",
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


# ---------------------------------------------------------------------------
# Errors / enums
# ---------------------------------------------------------------------------


class LogicPlatformClientError(RuntimeError):
    """Raised when a client operation cannot proceed fail-closed."""


class LogicPlatformClientRequestError(ValueError, LogicPlatformClientError):
    """Raised when a bound request envelope is structurally invalid."""


class LogicPlatformClientAuthorityError(LogicPlatformClientRequestError):
    """Raised when authority ceiling exceeds evidence-kind support."""


class LogicPlatformClientHandshakeError(LogicPlatformClientError):
    """Raised when an operation requires a compatible handshake first."""


class CacheFreshnessStatus(str, Enum):
    """Closed cache freshness vocabulary for client results."""

    CURRENT = "current"
    STALE = "stale"
    UNKNOWN = "unknown"
    MISS = "miss"
    INVALIDATED = "invalidated"


class ClientResultStatus(str, Enum):
    """Lifecycle status for a client operation (never a proof verdict)."""

    OK = "ok"
    FAILED = "failed"
    UNAVAILABLE = "unavailable"
    REJECTED = "rejected"
    DECLARATIVE = "declarative"
    INCOMPATIBLE = "incompatible"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise LogicPlatformClientRequestError(
            f"{field_name} must be a non-empty trimmed string"
        )
    if "\x00" in value:
        raise LogicPlatformClientRequestError(
            f"{field_name} must not contain NUL bytes"
        )
    return value


def _optional_text(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    return _text(value, field_name)


def _nonnegative_int_or_none(value: object, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise LogicPlatformClientRequestError(
            f"{field_name} must be a non-negative integer or null"
        )
    return value


def _json_safe(value: Any) -> Any:
    """Round-trip through strict JSON so free-form floats cannot escape."""

    def validate(item: Any) -> None:
        if item is None or isinstance(item, (str, bool, int)):
            return
        if isinstance(item, float):
            raise LogicPlatformClientRequestError(
                "client payloads cannot contain floating-point values"
            )
        if isinstance(item, Mapping):
            if not all(isinstance(key, str) for key in item):
                raise LogicPlatformClientRequestError(
                    "client payload object keys must be strings"
                )
            for nested in item.values():
                validate(nested)
            return
        if isinstance(item, (list, tuple)):
            for nested in item:
                validate(nested)
            return
        raise LogicPlatformClientRequestError(
            f"client payload contains unsupported value {type(item).__name__}"
        )

    validate(value)
    return json.loads(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    )


def _mapping(value: object, field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise LogicPlatformClientRequestError(f"{field_name} must be an object")
    return {str(key): item for key, item in value.items()}


def _authority_token(value: object, field_name: str) -> str:
    token = _text(str(getattr(value, "value", value)), field_name).lower()
    if token not in _AUTHORITY_RANK:
        raise LogicPlatformClientRequestError(
            f"{field_name} must be a closed authority ceiling token"
        )
    return token


def _evidence_token(value: object, field_name: str) -> str:
    raw = getattr(value, "value", value)
    if hasattr(raw, "local_id"):
        raw = raw.local_id
    elif isinstance(raw, Mapping):
        raw = raw.get("local_id") or raw.get("id") or raw.get("value") or ""
    token = _text(str(raw), field_name).lower()
    # Accept catalog-style evidence ids such as ``evidence.candidate``.
    if "." in token:
        token = token.rsplit(".", 1)[-1]
    return token


def check_authority_overclaim(
    evidence_kind: object,
    authority_ceiling: object,
) -> str:
    """Fail closed when authority exceeds evidence-kind support.

    Returns the normalized authority ceiling token when the claim is admitted.
    """

    evidence = _evidence_token(evidence_kind, "evidence_kind")
    ceiling = _authority_token(authority_ceiling, "authority_ceiling")
    max_ceiling = _EVIDENCE_AUTHORITY_CEILING.get(evidence, "advisory")
    if _AUTHORITY_RANK[ceiling] > _AUTHORITY_RANK[max_ceiling]:
        raise LogicPlatformClientAuthorityError(
            f"authority ceiling {ceiling!r} exceeds evidence kind "
            f"{evidence!r} maximum {max_ceiling!r}"
        )
    if ceiling in {"kernel", "reconstruction"} and evidence in _NON_KERNEL_EVIDENCE:
        raise LogicPlatformClientAuthorityError(
            f"authority ceiling {ceiling!r} cannot be claimed from "
            f"evidence kind {evidence!r}"
        )
    return ceiling


def _binding_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        _json_safe(dict(payload)),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _to_dict(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return value.to_dict()
    if isinstance(value, Mapping):
        return {str(key): _to_dict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_dict(item) for item in value]
    return str(value)


# ---------------------------------------------------------------------------
# Request / result envelopes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LogicPlatformClientRequest:
    """Bound request envelope with every LPC-G110 axis required at construction."""

    task_id: str
    tree_id: str
    policy_id: str
    plan_id: str
    resource_budget: ResourceBudget
    network_allowed: bool
    correlation_id: str
    evidence_kind: str
    authority_ceiling: str
    cancellation: CancellationToken | None = None
    deadline_unix_ms: int | None = None
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    binding_digest: str = ""
    schema_version: str = LOGIC_PLATFORM_CLIENT_REQUEST_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", _text(self.task_id, "task_id"))
        object.__setattr__(self, "tree_id", _text(self.tree_id, "tree_id"))
        object.__setattr__(self, "policy_id", _text(self.policy_id, "policy_id"))
        object.__setattr__(self, "plan_id", _text(self.plan_id, "plan_id"))
        object.__setattr__(
            self, "correlation_id", _text(self.correlation_id, "correlation_id")
        )
        object.__setattr__(
            self, "request_id", _text(self.request_id, "request_id")
        )
        if not isinstance(self.resource_budget, ResourceBudget):
            if isinstance(self.resource_budget, Mapping):
                object.__setattr__(
                    self,
                    "resource_budget",
                    ResourceBudget.from_dict(self.resource_budget),
                )
            else:
                raise LogicPlatformClientRequestError(
                    "resource_budget must be a ResourceBudget"
                )
        if not isinstance(self.network_allowed, bool):
            raise LogicPlatformClientRequestError(
                "network_allowed must be a boolean"
            )
        if self.network_allowed and not self.resource_budget.network_allowed:
            raise LogicPlatformClientRequestError(
                "network_allowed cannot exceed resource_budget.network_allowed"
            )
        if self.cancellation is not None and not isinstance(
            self.cancellation, CancellationToken
        ):
            raise LogicPlatformClientRequestError(
                "cancellation must be a CancellationToken or null"
            )
        object.__setattr__(
            self,
            "deadline_unix_ms",
            _nonnegative_int_or_none(self.deadline_unix_ms, "deadline_unix_ms"),
        )
        evidence = _evidence_token(self.evidence_kind, "evidence_kind")
        ceiling = check_authority_overclaim(evidence, self.authority_ceiling)
        object.__setattr__(self, "evidence_kind", evidence)
        object.__setattr__(self, "authority_ceiling", ceiling)
        if self.schema_version != LOGIC_PLATFORM_CLIENT_REQUEST_SCHEMA:
            raise LogicPlatformClientRequestError(
                f"unsupported request schema {self.schema_version!r}"
            )
        digest = _binding_digest(
            {
                "task_id": self.task_id,
                "tree_id": self.tree_id,
                "policy_id": self.policy_id,
                "plan_id": self.plan_id,
                "resource_budget": self.resource_budget.to_dict(),
                "network_allowed": self.network_allowed,
                "deadline_unix_ms": self.deadline_unix_ms,
                "correlation_id": self.correlation_id,
                "evidence_kind": self.evidence_kind,
                "authority_ceiling": self.authority_ceiling,
            }
        )
        if self.binding_digest and self.binding_digest != digest:
            raise LogicPlatformClientRequestError(
                "binding_digest does not match bound request axes"
            )
        object.__setattr__(self, "binding_digest", digest)

    def binding_payload(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "tree_id": self.tree_id,
            "policy_id": self.policy_id,
            "plan_id": self.plan_id,
            "resource_budget": self.resource_budget.to_dict(),
            "network_allowed": self.network_allowed,
            "deadline_unix_ms": self.deadline_unix_ms,
            "correlation_id": self.correlation_id,
            "evidence_kind": self.evidence_kind,
            "authority_ceiling": self.authority_ceiling,
            "binding_digest": self.binding_digest,
            "request_id": self.request_id,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "interface": SUPERVISOR_LOGIC_PLATFORM_CLIENT_INTERFACE,
            **self.binding_payload(),
        }


@dataclass(frozen=True)
class LogicPlatformClientResult:
    """Typed client operation result. Lifecycle status is not a proof verdict."""

    operation: str
    status: ClientResultStatus | str
    ok: bool
    request_id: str
    binding_digest: str
    authority_ceiling: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    error: str | None = None
    cache_freshness: CacheFreshnessStatus | str | None = None
    authority_upgraded: bool = False
    proof_claimed: bool = False
    is_proof: bool = False
    availability_is_not_proof: bool = True
    schema_version: str = LOGIC_PLATFORM_CLIENT_RESULT_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "operation", _text(self.operation, "operation"))
        status = self.status
        if not isinstance(status, ClientResultStatus):
            status = ClientResultStatus(str(status))
        object.__setattr__(self, "status", status)
        if not isinstance(self.ok, bool):
            raise LogicPlatformClientError("ok must be a boolean")
        object.__setattr__(self, "request_id", _text(self.request_id, "request_id"))
        object.__setattr__(
            self, "binding_digest", _text(self.binding_digest, "binding_digest")
        )
        object.__setattr__(
            self,
            "authority_ceiling",
            _authority_token(self.authority_ceiling, "authority_ceiling"),
        )
        if not isinstance(self.payload, Mapping):
            raise LogicPlatformClientError("payload must be an object")
        object.__setattr__(self, "payload", MappingProxyType(dict(self.payload)))
        if self.error is not None:
            object.__setattr__(self, "error", _text(self.error, "error"))
        if self.cache_freshness is not None and not isinstance(
            self.cache_freshness, CacheFreshnessStatus
        ):
            object.__setattr__(
                self,
                "cache_freshness",
                CacheFreshnessStatus(str(self.cache_freshness)),
            )
        for flag_name in (
            "authority_upgraded",
            "proof_claimed",
            "is_proof",
            "availability_is_not_proof",
        ):
            if not isinstance(getattr(self, flag_name), bool):
                raise LogicPlatformClientError(f"{flag_name} must be a boolean")
        if self.authority_upgraded:
            raise LogicPlatformClientError(
                "client results cannot claim authority_upgraded=true"
            )
        if self.schema_version != LOGIC_PLATFORM_CLIENT_RESULT_SCHEMA:
            raise LogicPlatformClientError(
                f"unsupported result schema {self.schema_version!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "interface": SUPERVISOR_LOGIC_PLATFORM_CLIENT_INTERFACE,
            "operation": self.operation,
            "status": self.status.value
            if isinstance(self.status, ClientResultStatus)
            else str(self.status),
            "ok": self.ok,
            "request_id": self.request_id,
            "binding_digest": self.binding_digest,
            "authority_ceiling": self.authority_ceiling,
            "payload": dict(self.payload),
            "error": self.error,
            "cache_freshness": (
                self.cache_freshness.value
                if isinstance(self.cache_freshness, CacheFreshnessStatus)
                else self.cache_freshness
            ),
            "authority_upgraded": self.authority_upgraded,
            "proof_claimed": self.proof_claimed,
            "is_proof": self.is_proof,
            "availability_is_not_proof": self.availability_is_not_proof,
        }


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class SupervisorLogicPlatformClient:
    """Lazy supervisor-side client for the datasets logic platform.

    Interface: ``SupervisorLogicPlatformClient@1``.
    """

    interface: Final = SUPERVISOR_LOGIC_PLATFORM_CLIENT_INTERFACE
    schema_version: Final = LOGIC_PLATFORM_CLIENT_SCHEMA
    version: Final = LOGIC_PLATFORM_CLIENT_VERSION
    task_id: Final = LOGIC_PLATFORM_CLIENT_TASK_ID
    goal_id: Final = LOGIC_PLATFORM_CLIENT_GOAL_ID

    def __init__(
        self,
        *,
        adapter: SupervisorCanonicalLogicAdapter | None = None,
        provider_facade: SupervisorLogicProviderFacade | None = None,
        require_handshake: bool = True,
        datasets_importer: Callable[[str], Any] | None = None,
    ) -> None:
        self._adapter = adapter or get_canonical_logic_adapter()
        self._provider_facade = provider_facade
        self._require_handshake = bool(require_handshake)
        self._datasets_importer = datasets_importer or importlib.import_module
        self._lock = threading.RLock()
        self._handshake_result: Any | None = None
        self._manifest: Any | None = None
        self._import_cache: dict[str, Any] = {}

    # -- identity ------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "interface": self.interface,
            "version": self.version,
            "task_id": self.task_id,
            "goal_id": self.goal_id,
            "operations": list(CLIENT_OPERATIONS),
            "typed_provider_operations": sorted(TYPED_PROVIDER_OPERATIONS),
            "require_handshake": self._require_handshake,
            "handshake_compatible": self.handshake_compatible,
            "datasets_import_is_lazy": True,
        }

    @property
    def handshake_compatible(self) -> bool:
        result = self._handshake_result
        return bool(result is not None and getattr(result, "compatible", False))

    def datasets_import_is_lazy(self) -> bool:
        return True

    # -- lazy datasets import ------------------------------------------------

    def _import_datasets(self, module_name: str) -> Any:
        cached = self._import_cache.get(module_name)
        if cached is not None:
            return cached
        with self._lock:
            cached = self._import_cache.get(module_name)
            if cached is not None:
                return cached
            module = self._datasets_importer(module_name)
            self._import_cache[module_name] = module
            return module

    def _require_compatible_handshake(self, operation: str) -> None:
        if not self._require_handshake:
            return
        if self.handshake_compatible:
            return
        raise LogicPlatformClientHandshakeError(
            f"{operation} requires a compatible handshake first"
        )

    def _result(
        self,
        *,
        operation: str,
        request: LogicPlatformClientRequest,
        status: ClientResultStatus,
        ok: bool,
        payload: Mapping[str, Any] | None = None,
        error: str | None = None,
        cache_freshness: CacheFreshnessStatus | None = None,
        authority_ceiling: str | None = None,
        proof_claimed: bool = False,
        is_proof: bool = False,
    ) -> LogicPlatformClientResult:
        ceiling = authority_ceiling or request.authority_ceiling
        return LogicPlatformClientResult(
            operation=operation,
            status=status,
            ok=ok,
            request_id=request.request_id,
            binding_digest=request.binding_digest,
            authority_ceiling=ceiling,
            payload=dict(payload or {}),
            error=error,
            cache_freshness=cache_freshness,
            authority_upgraded=False,
            proof_claimed=proof_claimed,
            is_proof=is_proof,
            availability_is_not_proof=True,
        )

    def _failed(
        self,
        *,
        operation: str,
        request: LogicPlatformClientRequest,
        error: str,
        status: ClientResultStatus = ClientResultStatus.FAILED,
        payload: Mapping[str, Any] | None = None,
    ) -> LogicPlatformClientResult:
        return self._result(
            operation=operation,
            request=request,
            status=status,
            ok=False,
            error=error,
            payload=payload,
        )

    # -- handshake -----------------------------------------------------------

    def handshake(
        self,
        requirements: Any | Mapping[str, Any] | None = None,
        *,
        manifest: Any | None = None,
        request: LogicPlatformClientRequest | None = None,
    ) -> LogicPlatformClientResult | Any:
        """Package-neutral LogicPlatformManifest@1 compatibility check.

        When ``request`` is provided, returns a :class:`LogicPlatformClientResult`.
        Without a request, returns the raw datasets ``HandshakeResult`` so
        callers that only need the typed handshake can avoid fabricating
        binding axes.
        """

        manifest_mod = self._import_datasets(MANIFEST_MODULE)
        HandshakeRequirements = manifest_mod.HandshakeRequirements
        handshake_fn = manifest_mod.handshake

        if requirements is None:
            req = HandshakeRequirements(
                required_adapter_versions=(self.interface,),
            )
        elif isinstance(requirements, HandshakeRequirements):
            req = requirements
        elif isinstance(requirements, Mapping):
            payload = dict(requirements)
            if "required_adapter_versions" not in payload:
                payload["required_adapter_versions"] = (self.interface,)
            req = HandshakeRequirements(**payload)
        else:
            req = requirements

        result = handshake_fn(req, manifest=manifest)
        with self._lock:
            self._handshake_result = result
            if getattr(result, "compatible", False):
                self._manifest = result.manifest
            else:
                # Keep last compatible manifest if re-handshake fails.
                pass

        if request is None:
            return result

        payload = {
            "compatible": bool(result.compatible),
            "manifest": _to_dict(result.manifest),
            "incompatibilities": [
                _to_dict(item) for item in (result.incompatibilities or ())
            ],
            "requires_git": False,
            "requires_sibling_repos": False,
            "requires_repository_layout": False,
        }
        if result.compatible:
            return self._result(
                operation="handshake",
                request=request,
                status=ClientResultStatus.OK,
                ok=True,
                payload=payload,
                cache_freshness=CacheFreshnessStatus.CURRENT,
            )
        return self._result(
            operation="handshake",
            request=request,
            status=ClientResultStatus.INCOMPATIBLE,
            ok=False,
            payload=payload,
            error="logic platform handshake incompatible",
        )

    # -- catalog -------------------------------------------------------------

    def catalog(
        self,
        request: LogicPlatformClientRequest,
    ) -> LogicPlatformClientResult:
        self._require_compatible_handshake("catalog")
        try:
            catalog_mod = self._import_datasets(CATALOG_MODULE)
            snapshot = catalog_mod.DEFAULT_CANONICAL_CATALOG_SNAPSHOT
            inventory = self._adapter.vocabulary_inventory()
            family_ids: list[str] = []
            if hasattr(snapshot, "family_ids"):
                family_ids = list(snapshot.family_ids)
            elif hasattr(snapshot, "families"):
                family_ids = sorted(
                    str(getattr(item, "family_id", item))
                    for item in snapshot.families
                )
            content_root = getattr(snapshot, "content_root", "") or getattr(
                snapshot, "catalog_root", ""
            )
            content_digest = getattr(snapshot, "content_digest", "") or getattr(
                snapshot, "catalog_digest", ""
            )
            payload = {
                "status": "declarative",
                "catalog_interface": getattr(
                    snapshot, "interface", "CanonicalLogicCatalogSnapshot@1"
                ),
                "catalog_root": content_root,
                "catalog_digest": content_digest,
                "family_ids": family_ids,
                "adapter_vocabulary": inventory,
                "availability_is_not_proof": True,
                "catalog_presence_is_not_proof": True,
                "binding": request.binding_payload(),
            }
            return self._result(
                operation="catalog",
                request=request,
                status=ClientResultStatus.DECLARATIVE,
                ok=True,
                payload=payload,
                cache_freshness=CacheFreshnessStatus.CURRENT,
            )
        except Exception as error:  # noqa: BLE001 - project to typed result
            return self._failed(
                operation="catalog",
                request=request,
                error=f"catalog access failed: {type(error).__name__}: {error}",
            )

    # -- formalize / slice / obligation / plan -------------------------------

    def formalize(
        self,
        request: LogicPlatformClientRequest,
        *,
        artifact_id: str,
        sample_id: str,
        domain: str,
        document_id: str,
        source_digest: str,
        expression_id: str,
        expression_digest: str,
        family: Any,
        profile: Any,
        view: Any,
        notation: Any,
        slices: Sequence[Any] = (),
        status: str | Any = "ok",
        metadata: Mapping[str, Any] | None = None,
    ) -> LogicPlatformClientResult:
        self._require_compatible_handshake("formalize")
        try:
            artifacts = self._import_datasets(ARTIFACTS_MODULE)
            FormalizationArtifactV3 = artifacts.FormalizationArtifactV3
            meta = {
                "task_id": request.task_id,
                "tree_id": request.tree_id,
                "policy_id": request.policy_id,
                "plan_id": request.plan_id,
                "binding_digest": request.binding_digest,
                "evidence_kind": request.evidence_kind,
                "authority_ceiling": request.authority_ceiling,
                **dict(metadata or {}),
            }
            artifact = FormalizationArtifactV3(
                artifact_id=artifact_id,
                sample_id=sample_id,
                domain=domain,
                document_id=document_id,
                source_digest=source_digest,
                expression_id=expression_id,
                expression_digest=expression_digest,
                family=family,
                profile=profile,
                view=view,
                notation=notation,
                status=status,
                slices=tuple(slices),
                metadata=meta,
            )
            return self._result(
                operation="formalize",
                request=request,
                status=ClientResultStatus.OK,
                ok=True,
                payload={
                    "artifact": _to_dict(artifact),
                    "interface": "FormalizationArtifact@3",
                    "candidate": True,
                    "proof_claimed": False,
                },
                authority_ceiling="candidate",
                proof_claimed=False,
            )
        except Exception as error:  # noqa: BLE001
            return self._failed(
                operation="formalize",
                request=request,
                error=f"formalize failed: {type(error).__name__}: {error}",
            )

    def create_slice(
        self,
        request: LogicPlatformClientRequest,
        *,
        slice_id: str,
        domain: str,
        document_id: str,
        source_digest: str,
        expression_id: str,
        expression_digest: str,
        family: Any,
        profile: Any,
        property: Any,
        view: Any,
        notation: Any,
        status: str | Any = "admitted",
        features: Sequence[str] = (),
        assumption_ids: Sequence[str] = (),
        formalization_artifact_id: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> LogicPlatformClientResult:
        self._require_compatible_handshake("create_slice")
        try:
            artifacts = self._import_datasets(ARTIFACTS_MODULE)
            DomainLogicSliceV2 = artifacts.DomainLogicSliceV2
            meta = {
                "task_id": request.task_id,
                "tree_id": request.tree_id,
                "policy_id": request.policy_id,
                "plan_id": request.plan_id,
                "binding_digest": request.binding_digest,
                **dict(metadata or {}),
            }
            slice_obj = DomainLogicSliceV2(
                slice_id=slice_id,
                domain=domain,
                document_id=document_id,
                source_digest=source_digest,
                expression_id=expression_id,
                expression_digest=expression_digest,
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
            return self._result(
                operation="create_slice",
                request=request,
                status=ClientResultStatus.OK,
                ok=True,
                payload={
                    "slice": _to_dict(slice_obj),
                    "interface": "DomainLogicSlice@2",
                    "status": str(
                        getattr(slice_obj.status, "value", slice_obj.status)
                    ),
                },
            )
        except Exception as error:  # noqa: BLE001
            return self._failed(
                operation="create_slice",
                request=request,
                error=f"create_slice failed: {type(error).__name__}: {error}",
            )

    def create_obligation(
        self,
        request: LogicPlatformClientRequest,
        *,
        slice_: Any,
        obligation_id: str,
        statement: str,
        encoding: Any,
        bounds: Any,
        evidence_kind: Any | None = None,
        authority_ceiling: Any | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> LogicPlatformClientResult:
        self._require_compatible_handshake("create_obligation")
        try:
            requests_v2 = self._import_datasets(REQUESTS_V2_MODULE)
            LogicObligationV2 = requests_v2.LogicObligationV2
            evidence = evidence_kind if evidence_kind is not None else request.evidence_kind
            ceiling = (
                authority_ceiling
                if authority_ceiling is not None
                else request.authority_ceiling
            )
            # Re-check against request binding so callers cannot overclaim.
            check_authority_overclaim(evidence, ceiling)
            meta = {
                "task_id": request.task_id,
                "tree_id": request.tree_id,
                "policy_id": request.policy_id,
                "plan_id": request.plan_id,
                "binding_digest": request.binding_digest,
                **dict(metadata or {}),
            }
            obligation = LogicObligationV2.from_slice(
                slice_,
                obligation_id=obligation_id,
                statement=statement,
                encoding=encoding,
                evidence_kind=evidence,
                bounds=bounds,
                authority_ceiling=ceiling,
                metadata=meta,
            )
            return self._result(
                operation="create_obligation",
                request=request,
                status=ClientResultStatus.OK,
                ok=True,
                payload={
                    "obligation": _to_dict(obligation),
                    "interface": "LogicObligation@2",
                },
                authority_ceiling=_authority_token(
                    getattr(obligation, "authority_ceiling", ceiling),
                    "authority_ceiling",
                ),
            )
        except Exception as error:  # noqa: BLE001
            return self._failed(
                operation="create_obligation",
                request=request,
                error=f"create_obligation failed: {type(error).__name__}: {error}",
            )

    def create_plan(
        self,
        request: LogicPlatformClientRequest,
        *,
        plan_id: str | None = None,
        formal_goal_id: str,
        graph_id: str,
        candidates: Sequence[Any] = (),
        step_order: Sequence[str] = (),
        provider_ids: Sequence[str] = (),
        bounds: Any | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> LogicPlatformClientResult:
        self._require_compatible_handshake("create_plan")
        try:
            contracts = self._import_datasets(PLAN_CONTRACTS_MODULE)
            GoalDirectedProofPlan = contracts.GoalDirectedProofPlan
            PlanStatus = contracts.PlanStatus
            AuthorityCeiling = contracts.AuthorityCeiling
            ResourceBounds = contracts.ResourceBounds

            plan_bounds = bounds
            if plan_bounds is None:
                plan_bounds = ResourceBounds()
            meta = {
                "task_id": request.task_id,
                "policy_id": request.policy_id,
                "binding_digest": request.binding_digest,
                "correlation_id": request.correlation_id,
                **dict(metadata or {}),
            }
            plan = GoalDirectedProofPlan(
                plan_id=plan_id or request.plan_id,
                formal_goal_id=formal_goal_id,
                graph_id=graph_id,
                tree_id=request.tree_id,
                candidates=tuple(candidates),
                step_order=tuple(step_order),
                status=PlanStatus.DRAFT,
                bounds=plan_bounds,
                provider_ids=tuple(provider_ids),
                authority=AuthorityCeiling.CANDIDATE,
                proof_claimed=False,
                completion_claimed=False,
                metadata=meta,
            )
            # Force candidate authority on the client envelope regardless of
            # any provider-side claim that might appear later.
            return self._result(
                operation="create_plan",
                request=request,
                status=ClientResultStatus.OK,
                ok=True,
                payload={
                    "plan": _to_dict(plan),
                    "interface": "GoalDirectedProofPlan@1",
                    "proof_claimed": False,
                    "completion_claimed": False,
                },
                authority_ceiling="candidate",
                proof_claimed=False,
            )
        except Exception as error:  # noqa: BLE001
            return self._failed(
                operation="create_plan",
                request=request,
                error=f"create_plan failed: {type(error).__name__}: {error}",
            )

    # -- capability discovery / typed invocation -----------------------------

    def discover_capabilities(
        self,
        request: LogicPlatformClientRequest,
        *,
        include_verification_api: bool = True,
    ) -> LogicPlatformClientResult:
        self._require_compatible_handshake("discover_capabilities")
        try:
            providers: list[dict[str, Any]] = []
            facade_meta: dict[str, Any] | None = None
            if self._provider_facade is not None:
                facade_meta = {
                    "provider_id": self._provider_facade.provider_id,
                    "provider_version": self._provider_facade.provider_version,
                    "protocol_version": self._provider_facade.protocol_version,
                    "loaded": self._provider_facade.loaded,
                    "operations": sorted(TYPED_PROVIDER_OPERATIONS),
                }
                providers.append(dict(facade_meta))

            verification_providers: list[Any] = []
            if include_verification_api:
                try:
                    api_mod = self._import_datasets(VERIFICATION_API_MODULE)
                    response = api_mod.list_providers()
                    raw = getattr(response, "result", None)
                    if raw is None and hasattr(response, "to_dict"):
                        raw = response.to_dict()
                    if isinstance(raw, Mapping):
                        items = raw.get("providers") or raw.get("items") or []
                        if isinstance(items, Sequence) and not isinstance(
                            items, (str, bytes)
                        ):
                            verification_providers = list(items)
                    elif isinstance(raw, Sequence) and not isinstance(
                        raw, (str, bytes)
                    ):
                        verification_providers = list(raw)
                except Exception:  # noqa: BLE001 - discovery is declarative
                    verification_providers = []

            payload = {
                "status": "declarative",
                "availability_is_not_proof": True,
                "facade": facade_meta,
                "providers": providers,
                "verification_api_providers": [
                    _to_dict(item) for item in verification_providers
                ],
                "typed_provider_operations": sorted(TYPED_PROVIDER_OPERATIONS),
                "binding": request.binding_payload(),
            }
            return self._result(
                operation="discover_capabilities",
                request=request,
                status=ClientResultStatus.DECLARATIVE,
                ok=True,
                payload=payload,
            )
        except Exception as error:  # noqa: BLE001
            return self._failed(
                operation="discover_capabilities",
                request=request,
                error=(
                    "discover_capabilities failed: "
                    f"{type(error).__name__}: {error}"
                ),
            )

    def invoke(
        self,
        request: LogicPlatformClientRequest,
        *,
        operation: str | ProofProviderOperation,
        payload: Mapping[str, Any] | None = None,
        provider_facade: SupervisorLogicProviderFacade | None = None,
    ) -> LogicPlatformClientResult:
        self._require_compatible_handshake("invoke")
        op = str(getattr(operation, "value", operation)).strip().lower()
        if op not in TYPED_PROVIDER_OPERATIONS:
            return self._failed(
                operation="invoke",
                request=request,
                status=ClientResultStatus.REJECTED,
                error=f"unsupported typed provider operation: {op!r}",
                payload={"allowed_operations": sorted(TYPED_PROVIDER_OPERATIONS)},
            )
        facade = provider_facade or self._provider_facade
        if facade is None:
            return self._failed(
                operation="invoke",
                request=request,
                status=ClientResultStatus.UNAVAILABLE,
                error="no provider facade bound for typed invocation",
            )
        if request.cancellation is not None and request.cancellation.is_cancelled():
            return self._failed(
                operation="invoke",
                request=request,
                status=ClientResultStatus.REJECTED,
                error="request was cancelled before invocation",
            )

        try:
            body = dict(payload or {})
            body.setdefault("binding", request.binding_payload())
            body.setdefault("binding_digest", request.binding_digest)
            body.setdefault("task_id", request.task_id)
            body.setdefault("tree_id", request.tree_id)
            body.setdefault("policy_id", request.policy_id)
            body.setdefault("plan_id", request.plan_id)
            body.setdefault("evidence_kind", request.evidence_kind)
            body.setdefault("authority_ceiling", request.authority_ceiling)
            body.setdefault("correlation_id", request.correlation_id)
            provider_request = ProviderRequest(
                operation=op,
                payload=_json_safe(body),
                request_id=request.request_id,
                resource_budget=request.resource_budget,
                network_allowed=request.network_allowed,
                deadline_unix_ms=request.deadline_unix_ms,
            )
            response: ProviderResponse = facade.invoke(
                provider_request,
                cancellation=request.cancellation,
            )
            if response.ok:
                result_payload = dict(response.result or {})
                # Never upgrade authority from provider-claimed values.
                return self._result(
                    operation="invoke",
                    request=request,
                    status=ClientResultStatus.OK,
                    ok=True,
                    payload={
                        "provider_operation": op,
                        "provider_id": response.provider_id,
                        "provider_version": response.provider_version,
                        "duration_ms": response.duration_ms,
                        "result": result_payload,
                        "authority_upgraded": False,
                    },
                    authority_ceiling=request.authority_ceiling,
                )
            error = response.error
            message = (
                error.message
                if error is not None
                else "provider invocation failed"
            )
            code = error.code.value if error is not None else "provider_error"
            return self._failed(
                operation="invoke",
                request=request,
                status=ClientResultStatus.FAILED,
                error=message,
                payload={
                    "provider_operation": op,
                    "provider_id": response.provider_id,
                    "provider_version": response.provider_version,
                    "failure_code": code,
                    "retryable": bool(
                        getattr(error, "retryable", False) if error else False
                    ),
                },
            )
        except Exception as error:  # noqa: BLE001
            return self._failed(
                operation="invoke",
                request=request,
                error=f"invoke failed: {type(error).__name__}: {error}",
            )

    def reconstruct(
        self,
        request: LogicPlatformClientRequest,
        *,
        payload: Mapping[str, Any] | None = None,
        provider_facade: SupervisorLogicProviderFacade | None = None,
    ) -> LogicPlatformClientResult:
        result = self.invoke(
            request,
            operation=ProofProviderOperation.RECONSTRUCT,
            payload=payload,
            provider_facade=provider_facade,
        )
        # Preserve invoke semantics but surface operation name for callers.
        return LogicPlatformClientResult(
            operation="reconstruct",
            status=result.status,
            ok=result.ok,
            request_id=result.request_id,
            binding_digest=result.binding_digest,
            authority_ceiling=result.authority_ceiling,
            payload=dict(result.payload),
            error=result.error,
            cache_freshness=result.cache_freshness,
            authority_upgraded=False,
            proof_claimed=False,
            is_proof=False,
            availability_is_not_proof=True,
        )

    def verify(
        self,
        request: LogicPlatformClientRequest,
        *,
        payload: Mapping[str, Any] | None = None,
        provider_facade: SupervisorLogicProviderFacade | None = None,
    ) -> LogicPlatformClientResult:
        result = self.invoke(
            request,
            operation=ProofProviderOperation.VERIFY,
            payload=payload,
            provider_facade=provider_facade,
        )
        return LogicPlatformClientResult(
            operation="verify",
            status=result.status,
            ok=result.ok,
            request_id=result.request_id,
            binding_digest=result.binding_digest,
            authority_ceiling=result.authority_ceiling,
            payload=dict(result.payload),
            error=result.error,
            cache_freshness=result.cache_freshness,
            authority_upgraded=False,
            proof_claimed=False,
            is_proof=False,
            availability_is_not_proof=True,
        )

    # -- receipts ------------------------------------------------------------

    def receipt(
        self,
        request: LogicPlatformClientRequest,
        *,
        validation_result: TranslationValidationResult | Mapping[str, Any],
    ) -> LogicPlatformClientResult:
        self._require_compatible_handshake("receipt")
        try:
            projected = self._adapter.project_translation_validation_receipt(
                validation_result
            )
            # Force non-upgrade invariants even if adapter payload drifts.
            projected = dict(projected)
            projected["authority"] = "none"
            projected["proof_success"] = False
            projected["proof_attempted"] = bool(
                projected.get("proof_attempted", False)
            )
            projected["binding_digest"] = request.binding_digest
            projected["task_id"] = request.task_id
            projected["tree_id"] = request.tree_id
            projected["policy_id"] = request.policy_id
            projected["plan_id"] = request.plan_id
            return self._result(
                operation="receipt",
                request=request,
                status=ClientResultStatus.OK,
                ok=True,
                payload={
                    "receipt": projected,
                    "authority": "none",
                    "proof_success": False,
                },
                authority_ceiling="candidate",
                proof_claimed=False,
            )
        except Exception as error:  # noqa: BLE001
            return self._failed(
                operation="receipt",
                request=request,
                error=f"receipt projection failed: {type(error).__name__}: {error}",
            )

    # -- counterexamples -----------------------------------------------------

    def counterexample(
        self,
        request: LogicPlatformClientRequest,
        value: Any,
        *,
        kind: CounterexampleKind | str | None = None,
        property_class: str = "unknown",
        violated_property: str = "unknown",
        summary: str = "",
        assumption_ids: Sequence[str] = (),
        finite_bounds: Mapping[str, Any] | None = None,
        obligation_ids: Sequence[str] = (),
        provider_ids: Sequence[str] = (),
    ) -> LogicPlatformClientResult:
        self._require_compatible_handshake("counterexample")
        try:
            bindings = CounterexampleBindings(
                plan_ids=(request.plan_id,),
                task_ids=(request.task_id,),
                tree_ids=(request.tree_id,),
                policy_ids=(request.policy_id,),
                obligation_ids=tuple(obligation_ids),
                provider_ids=tuple(provider_ids),
                assumption_ids=tuple(assumption_ids),
            )
            normalized: FormalCounterexample = normalize_counterexample(
                value,
                kind=kind,
                bindings=bindings,
                property_class=property_class or "unknown",
                violated_property=violated_property or "unknown",
                summary=summary,
                assumption_ids=assumption_ids,
                finite_bounds=finite_bounds,
            )
            payload = {
                "counterexample": _to_dict(normalized),
                "is_proof": False,
                "proof_claimed": False,
            }
            return self._result(
                operation="counterexample",
                request=request,
                status=ClientResultStatus.OK,
                ok=True,
                payload=payload,
                authority_ceiling="candidate",
                proof_claimed=False,
                is_proof=False,
            )
        except Exception as error:  # noqa: BLE001
            return self._failed(
                operation="counterexample",
                request=request,
                error=(
                    "counterexample normalization failed: "
                    f"{type(error).__name__}: {error}"
                ),
            )

    # -- cache freshness -----------------------------------------------------

    def cache_freshness(
        self,
        request: LogicPlatformClientRequest,
        *,
        cache_key: str | None = None,
        stored_at_unix_ms: int | None = None,
        ttl_ms: int | None = None,
        now_unix_ms: int | None = None,
        invalidated: bool = False,
        repository: Any | None = None,
    ) -> LogicPlatformClientResult:
        self._require_compatible_handshake("cache_freshness")
        try:
            if invalidated:
                status = CacheFreshnessStatus.INVALIDATED
            elif repository is not None and cache_key is not None:
                status = self._freshness_from_repository(
                    repository, cache_key=cache_key
                )
            elif stored_at_unix_ms is None or ttl_ms is None:
                status = CacheFreshnessStatus.UNKNOWN
            else:
                if (
                    isinstance(stored_at_unix_ms, bool)
                    or not isinstance(stored_at_unix_ms, int)
                    or stored_at_unix_ms < 0
                ):
                    raise LogicPlatformClientRequestError(
                        "stored_at_unix_ms must be a non-negative integer"
                    )
                if (
                    isinstance(ttl_ms, bool)
                    or not isinstance(ttl_ms, int)
                    or ttl_ms < 0
                ):
                    raise LogicPlatformClientRequestError(
                        "ttl_ms must be a non-negative integer"
                    )
                now = (
                    now_unix_ms
                    if now_unix_ms is not None
                    else int(time.time() * 1000)
                )
                if isinstance(now, bool) or not isinstance(now, int) or now < 0:
                    raise LogicPlatformClientRequestError(
                        "now_unix_ms must be a non-negative integer"
                    )
                age = now - stored_at_unix_ms
                status = (
                    CacheFreshnessStatus.STALE
                    if age > ttl_ms
                    else CacheFreshnessStatus.CURRENT
                )

            payload = {
                "status": status.value,
                "cache_key": cache_key,
                "stored_at_unix_ms": stored_at_unix_ms,
                "ttl_ms": ttl_ms,
                "invalidated": bool(invalidated),
                "binding_digest": request.binding_digest,
            }
            return self._result(
                operation="cache_freshness",
                request=request,
                status=ClientResultStatus.OK,
                ok=True,
                payload=payload,
                cache_freshness=status,
            )
        except Exception as error:  # noqa: BLE001
            return self._failed(
                operation="cache_freshness",
                request=request,
                error=f"cache_freshness failed: {type(error).__name__}: {error}",
            )

    def _freshness_from_repository(
        self, repository: Any, *, cache_key: str
    ) -> CacheFreshnessStatus:
        if hasattr(repository, "freshness") and callable(repository.freshness):
            report = repository.freshness(cache_key)
        elif hasattr(repository, "get_freshness") and callable(
            repository.get_freshness
        ):
            report = repository.get_freshness(cache_key)
        else:
            return CacheFreshnessStatus.UNKNOWN

        if report is None:
            return CacheFreshnessStatus.MISS
        if isinstance(report, CacheFreshnessStatus):
            return report
        if isinstance(report, str):
            try:
                return CacheFreshnessStatus(report.lower())
            except ValueError:
                return CacheFreshnessStatus.UNKNOWN
        if isinstance(report, Mapping):
            if report.get("miss") or report.get("status") == "miss":
                return CacheFreshnessStatus.MISS
            if report.get("invalidated") or report.get("status") == "invalidated":
                return CacheFreshnessStatus.INVALIDATED
            token = str(
                report.get("status")
                or report.get("freshness")
                or report.get("state")
                or "unknown"
            ).lower()
            try:
                return CacheFreshnessStatus(token)
            except ValueError:
                return CacheFreshnessStatus.UNKNOWN
        token = str(getattr(report, "value", report)).lower()
        try:
            return CacheFreshnessStatus(token)
        except ValueError:
            return CacheFreshnessStatus.UNKNOWN


def get_logic_platform_client(
    **kwargs: Any,
) -> SupervisorLogicPlatformClient:
    """Construct a :class:`SupervisorLogicPlatformClient`."""

    return SupervisorLogicPlatformClient(**kwargs)


__all__ = [
    "CLIENT_OPERATIONS",
    "CacheFreshnessStatus",
    "ClientResultStatus",
    "LOGIC_PLATFORM_CLIENT_GOAL_ID",
    "LOGIC_PLATFORM_CLIENT_REQUEST_SCHEMA",
    "LOGIC_PLATFORM_CLIENT_RESULT_SCHEMA",
    "LOGIC_PLATFORM_CLIENT_SCHEMA",
    "LOGIC_PLATFORM_CLIENT_TASK_ID",
    "LOGIC_PLATFORM_CLIENT_VERSION",
    "LogicPlatformClientAuthorityError",
    "LogicPlatformClientError",
    "LogicPlatformClientHandshakeError",
    "LogicPlatformClientRequest",
    "LogicPlatformClientRequestError",
    "LogicPlatformClientResult",
    "SUPERVISOR_LOGIC_PLATFORM_CLIENT_INTERFACE",
    "TYPED_PROVIDER_OPERATIONS",
    "SupervisorLogicPlatformClient",
    "check_authority_overclaim",
    "get_logic_platform_client",
]
