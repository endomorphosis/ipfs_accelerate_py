"""SupervisorLogicPlatformClient@1 — lazy supervisor boundary for the logic platform.

Provides one handshake + typed invocation surface for supervisors.  Semantic
identities stay datasets-owned.  The supervisor retains scheduling, isolation,
resources, cancellation, leases, and placement.

Importing this module never imports ``ipfs_datasets_py``.  Datasets packages are
loaded only for an explicit handshake, catalog, formalization, obligation, plan,
provider, reconstruction, verification, receipt, counterexample, or cache call.
"""

from __future__ import annotations

import importlib
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Final

from .canonical_logic_adapter import SupervisorCanonicalLogicAdapter


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
CLIENT_TASK_ID: Final = "LPC-110"
CLIENT_GOAL_ID: Final = "LPC-G110"

DEFAULT_REQUIRED_ADAPTER_VERSIONS: Final[tuple[str, ...]] = (
    "SupervisorCanonicalLogicAdapter@1",
    SUPERVISOR_LOGIC_PLATFORM_CLIENT_INTERFACE,
)

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

# Datasets module targets (loaded only after an explicit boundary call).
_MANIFEST_MODULE: Final = "ipfs_datasets_py.logic.platform.manifest"
_CATALOG_MODULE: Final = "ipfs_datasets_py.logic.families.canonical_catalog"
_ARTIFACTS_MODULE: Final = "ipfs_datasets_py.logic.formalization.artifacts_v3"
_REQUESTS_MODULE: Final = "ipfs_datasets_py.logic.backends.requests_v2"
_PROTOCOL_MODULE: Final = "ipfs_datasets_py.logic.backends.protocol_v2"
_RESPONSE_MODULE: Final = "ipfs_datasets_py.logic.backends.response_v2"
_CACHE_KEY_MODULE: Final = "ipfs_datasets_py.logic.common.canonical_cache_key"
_PLAN_MODULE: Final = (
    "ipfs_datasets_py.logic.software_verification.tactician.contracts"
)
_NAMESPACES_MODULE: Final = "ipfs_datasets_py.logic.families.namespaces"

# Private material stripped from counterexample projections.
_PRIVATE_COUNTEREXAMPLE_KEYS: Final[frozenset[str]] = frozenset(
    {
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
    }
)

_import_lock: Final = threading.Lock()
_import_cache: dict[str, Any] = {}
_client_singleton_lock: Final = threading.Lock()
_client_singleton: Any | None = None


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class LogicPlatformClientError(RuntimeError):
    """Base error for SupervisorLogicPlatformClient@1 failures."""


class LogicPlatformClientHandshakeError(LogicPlatformClientError):
    """Raised when an operation requires a successful handshake first."""


class LogicPlatformClientAuthorityError(LogicPlatformClientError, ValueError):
    """Raised when a request ceiling overclaims its evidence kind."""


class LogicPlatformClientFreshnessError(LogicPlatformClientError, ValueError):
    """Raised when a cache entry fails fail-closed freshness admission."""


class LogicPlatformClientProviderError(LogicPlatformClientError):
    """Raised when a provider is required but unbound or returns invalid output."""


# ---------------------------------------------------------------------------
# Lazy import helpers
# ---------------------------------------------------------------------------


def _lazy_import(module_name: str) -> Any:
    """Import a datasets module only after an explicit boundary call."""

    cached = _import_cache.get(module_name)
    if cached is not None:
        return cached
    with _import_lock:
        cached = _import_cache.get(module_name)
        if cached is not None:
            return cached
        module = importlib.import_module(module_name)
        _import_cache[module_name] = module
        return module


def _clear_import_cache_for_tests() -> None:
    """Test helper: drop cached datasets imports without unloading modules."""

    with _import_lock:
        _import_cache.clear()


def _text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise LogicPlatformClientError(
            f"{field_name} must be a non-empty trimmed string"
        )
    if "\x00" in value:
        raise LogicPlatformClientError(f"{field_name} must not contain NUL bytes")
    return value


def _optional_text(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise LogicPlatformClientError(f"{field_name} must be a string or None")
    stripped = value.strip()
    if not stripped:
        return None
    if "\x00" in stripped:
        raise LogicPlatformClientError(f"{field_name} must not contain NUL bytes")
    return stripped


def _enum_value(value: object) -> str:
    return str(getattr(value, "value", value))


def _new_correlation_id() -> str:
    return f"corr:{uuid.uuid4().hex}"


# ---------------------------------------------------------------------------
# Context / result views
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ClientRequestContext:
    """Request bindings for LPC-G110 operational calls.

    Authority ceilings are checked against evidence kind at construction.
    """

    task_id: str
    tree_id: str
    policy_id: str
    plan_id: str | None = None
    budget: Mapping[str, Any] = field(default_factory=dict)
    network_allowed: bool = False
    cancellation: Mapping[str, Any] | None = None
    deadline_unix_ms: int | None = None
    correlation_id: str = ""
    evidence_kind: str = "candidate"
    authority_ceiling: str = "advisory"

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", _text(self.task_id, "task_id"))
        object.__setattr__(self, "tree_id", _text(self.tree_id, "tree_id"))
        object.__setattr__(self, "policy_id", _text(self.policy_id, "policy_id"))
        object.__setattr__(
            self, "plan_id", _optional_text(self.plan_id, "plan_id")
        )
        if not isinstance(self.budget, Mapping):
            raise LogicPlatformClientError("budget must be a mapping")
        object.__setattr__(
            self, "budget", MappingProxyType(dict(self.budget))
        )
        if not isinstance(self.network_allowed, bool):
            raise LogicPlatformClientError("network_allowed must be a bool")
        if self.cancellation is not None and not isinstance(
            self.cancellation, Mapping
        ):
            raise LogicPlatformClientError("cancellation must be a mapping or None")
        if self.cancellation is not None:
            object.__setattr__(
                self,
                "cancellation",
                MappingProxyType(dict(self.cancellation)),
            )
        if self.deadline_unix_ms is not None:
            if (
                isinstance(self.deadline_unix_ms, bool)
                or not isinstance(self.deadline_unix_ms, int)
                or self.deadline_unix_ms < 0
            ):
                raise LogicPlatformClientError(
                    "deadline_unix_ms must be a non-negative int or None"
                )
        correlation = self.correlation_id.strip() if self.correlation_id else ""
        if not correlation:
            correlation = _new_correlation_id()
        object.__setattr__(self, "correlation_id", _text(correlation, "correlation_id"))
        evidence = _enum_value(self.evidence_kind).strip()
        ceiling = _enum_value(self.authority_ceiling).strip()
        object.__setattr__(self, "evidence_kind", _text(evidence, "evidence_kind"))
        object.__setattr__(
            self, "authority_ceiling", _text(ceiling, "authority_ceiling")
        )
        check_authority_overclaim(self.authority_ceiling, self.evidence_kind)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "tree_id": self.tree_id,
            "policy_id": self.policy_id,
            "plan_id": self.plan_id,
            "budget": dict(self.budget),
            "network_allowed": self.network_allowed,
            "cancellation": (
                None if self.cancellation is None else dict(self.cancellation)
            ),
            "deadline_unix_ms": self.deadline_unix_ms,
            "correlation_id": self.correlation_id,
            "evidence_kind": self.evidence_kind,
            "authority_ceiling": self.authority_ceiling,
        }


@dataclass(frozen=True, slots=True)
class ClientInvocationResult:
    """Typed wrap of a protocol-v2 response plus request context."""

    request_id: str
    operation: str
    response: Any
    context: ClientRequestContext
    provider_id: str = ""
    provider_version: str = ""
    simulated: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_id", _text(self.request_id, "request_id"))
        object.__setattr__(self, "operation", _text(self.operation, "operation"))
        if not isinstance(self.context, ClientRequestContext):
            raise LogicPlatformClientError(
                "context must be a ClientRequestContext"
            )
        object.__setattr__(
            self, "provider_id", str(self.provider_id or "").strip()
        )
        object.__setattr__(
            self, "provider_version", str(self.provider_version or "").strip()
        )
        if not isinstance(self.simulated, bool):
            raise LogicPlatformClientError("simulated must be a bool")
        if not isinstance(self.metadata, Mapping):
            raise LogicPlatformClientError("metadata must be a mapping")
        object.__setattr__(
            self, "metadata", MappingProxyType(dict(self.metadata))
        )

    @property
    def evidence_kind(self) -> str:
        response = self.response
        kind = getattr(response, "evidence_kind", None)
        if kind is not None:
            return _enum_value(kind)
        if isinstance(response, Mapping):
            return _enum_value(response.get("evidence_kind", "candidate"))
        return "candidate"

    @property
    def evidence_authority(self) -> str:
        response = self.response
        authority = getattr(response, "evidence_authority", None)
        if authority is not None:
            return _enum_value(authority)
        if isinstance(response, Mapping):
            return _enum_value(response.get("evidence_authority", "advisory"))
        return "advisory"

    @property
    def operation_status(self) -> str:
        response = self.response
        status = getattr(response, "operation_status", None)
        if status is not None:
            return _enum_value(status)
        if isinstance(response, Mapping):
            return _enum_value(response.get("operation_status", "unknown"))
        return "unknown"

    @property
    def verdict(self) -> str:
        response = self.response
        verdict = getattr(response, "verdict", None)
        if verdict is not None:
            return _enum_value(verdict)
        if isinstance(response, Mapping):
            return _enum_value(response.get("verdict", "unknown"))
        return "unknown"

    def to_dict(self) -> dict[str, Any]:
        response = self.response
        if hasattr(response, "to_dict") and callable(response.to_dict):
            response_payload: Any = response.to_dict()
        elif isinstance(response, Mapping):
            response_payload = dict(response)
        else:
            response_payload = response
        return {
            "request_id": self.request_id,
            "operation": self.operation,
            "response": response_payload,
            "context": self.context.to_dict(),
            "provider_id": self.provider_id,
            "provider_version": self.provider_version,
            "simulated": self.simulated,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class ClientReceiptView:
    """Supervisor receipt projection (untrusted by default; not LPC-111 admission)."""

    request_id: str
    operation: str
    provider_id: str
    evidence_kind: str
    evidence_authority: str
    verdict: str
    operation_status: str
    context: ClientRequestContext
    translation_ids: tuple[str, ...] = ()
    artifact_ids: tuple[str, ...] = ()
    simulated: bool = False
    authority: str = "advisory"

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "operation": self.operation,
            "provider_id": self.provider_id,
            "evidence_kind": self.evidence_kind,
            "evidence_authority": self.evidence_authority,
            "verdict": self.verdict,
            "operation_status": self.operation_status,
            "context": self.context.to_dict(),
            "translation_ids": list(self.translation_ids),
            "artifact_ids": list(self.artifact_ids),
            "simulated": self.simulated,
            "authority": self.authority,
        }


@dataclass(frozen=True, slots=True)
class ClientCounterexampleView:
    """Public-safe counterexample projection with private material stripped."""

    request_id: str
    public_fields: Mapping[str, Any]
    redacted: bool = True
    authority: str = "advisory"
    stripped_keys: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "public_fields": dict(self.public_fields),
            "redacted": self.redacted,
            "authority": self.authority,
            "stripped_keys": list(self.stripped_keys),
        }


@dataclass(frozen=True, slots=True)
class CacheFreshnessReport:
    """Cache hit admission outcome under fail-closed freshness rules."""

    fresh: bool
    reason: str
    key_id: str = ""
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.fresh, bool):
            raise LogicPlatformClientError("fresh must be a bool")
        object.__setattr__(self, "reason", _text(self.reason, "reason"))
        object.__setattr__(self, "key_id", str(self.key_id or ""))
        if not isinstance(self.details, Mapping):
            raise LogicPlatformClientError("details must be a mapping")
        object.__setattr__(
            self, "details", MappingProxyType(dict(self.details))
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "fresh": self.fresh,
            "reason": self.reason,
            "key_id": self.key_id,
            "details": dict(self.details),
        }


# ---------------------------------------------------------------------------
# Authority overclaim (fail closed)
# ---------------------------------------------------------------------------


# Local mirror of datasets request ceilings so pure-client probes work offline.
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
        # LogicEvidenceAuthority spellings used on cache keys / responses.
        "independently_checkable": 9,
        "authoritative": 9,
        "unknown": 1,
    }
)

_EVIDENCE_AUTHORITY_CEILING: Final[Mapping[str, str]] = MappingProxyType(
    {
        "parse": "none",
        "advisory": "advisory",
        "candidate": "candidate",
        "atp_candidate": "candidate",
        "smt_candidate": "candidate",
        "llm_output": "candidate",
        "model_output": "candidate",
        "declaration": "candidate",
        "review": "candidate",
        "model": "satisfiability",
        "core": "satisfiability",
        "trace": "finite_trace",
        "monitor": "finite_trace",
        "attack": "protocol",
        "proof": "reconstruction",
        "checked_proof": "reconstruction",
        "proof_certificate": "reconstruction",
        "kernel": "kernel",
        "kernel_receipt": "kernel",
        "kernel_checked_proof": "kernel",
        "attestation": "attestation",
        "authorization": "authorization",
        "bounded": "bounded",
        "unknown": "advisory",
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
        "atp_candidate",
        "smt_candidate",
        "llm_output",
        "model_output",
        "declaration",
        "review",
    }
)

_NON_THEOREM_EVIDENCE: Final[frozenset[str]] = frozenset(
    {
        "parse",
        "model",
        "trace",
        "attack",
        "monitor",
        "candidate",
        "advisory",
        "core",
        "atp_candidate",
        "smt_candidate",
        "llm_output",
        "model_output",
        "declaration",
        "review",
    }
)


def check_authority_overclaim(
    authority_ceiling: object,
    evidence_kind: object,
    *,
    field_name: str = "authority_ceiling",
) -> None:
    """Reject ceilings above the evidence-kind max (fail closed).

    Pure client-side gate used before dispatch.  Datasets
    ``AuthorityOverclaimError`` remains the semantic gate on obligations and
    backend requests.
    """

    ceiling = _enum_value(authority_ceiling).strip().lower()
    evidence = _enum_value(evidence_kind).strip().lower()
    if not ceiling or not evidence:
        raise LogicPlatformClientAuthorityError(
            f"{field_name} and evidence_kind must be non-empty"
        )
    max_ceiling = _EVIDENCE_AUTHORITY_CEILING.get(evidence, "advisory")
    ceiling_rank = _AUTHORITY_RANK.get(ceiling)
    max_rank = _AUTHORITY_RANK.get(max_ceiling, 1)
    if ceiling_rank is None:
        raise LogicPlatformClientAuthorityError(
            f"{field_name} {ceiling!r} is not an admitted authority ceiling"
        )
    if ceiling_rank > max_rank:
        raise LogicPlatformClientAuthorityError(
            f"{field_name} {ceiling!r} overclaims evidence kind {evidence!r} "
            f"(max admitted ceiling {max_ceiling!r}); fail closed before dispatch"
        )
    if ceiling in {"kernel", "authoritative", "independently_checkable"} and (
        evidence in _NON_KERNEL_EVIDENCE
    ):
        raise LogicPlatformClientAuthorityError(
            f"kernel authority cannot be claimed with evidence {evidence!r}"
        )
    if evidence in _NON_THEOREM_EVIDENCE and ceiling in {
        "kernel",
        "reconstruction",
        "authoritative",
        "independently_checkable",
    }:
        raise LogicPlatformClientAuthorityError(
            f"proof/reconstruction authority cannot be claimed with evidence "
            f"{evidence!r}"
        )


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

    def __init__(
        self,
        *,
        provider: Any | None = None,
        adapter: SupervisorCanonicalLogicAdapter | None = None,
        module_importer: Callable[[str], Any] | None = None,
        require_handshake: bool = True,
        default_context: ClientRequestContext | Mapping[str, Any] | None = None,
    ) -> None:
        self._provider = provider
        self._adapter = adapter
        self._import = module_importer or _lazy_import
        self._require_handshake = bool(require_handshake)
        self._handshake_result: Any | None = None
        self._handshake_compatible = False
        self._manifest: Any | None = None
        self._lock = threading.RLock()
        self._default_context = self._coerce_context(
            default_context,
            allow_none=True,
        )

    # -- identity ------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "interface": self.interface,
            "version": self.version,
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "goal_id": self.goal_id,
            "require_handshake": self._require_handshake,
            "handshake_compatible": self._handshake_compatible,
            "provider_bound": self._provider is not None,
        }

    @property
    def handshake_compatible(self) -> bool:
        return self._handshake_compatible

    @property
    def datasets_import_is_lazy(self) -> bool:
        """True when no datasets modules have been loaded through this client."""

        return not any(
            name.startswith("ipfs_datasets_py") for name in _import_cache
        )

    def adapter(self) -> SupervisorCanonicalLogicAdapter:
        """Return the lazy vocabulary adapter (created on first use)."""

        if self._adapter is None:
            self._adapter = SupervisorCanonicalLogicAdapter(
                module_importer=self._import
            )
        return self._adapter

    # -- context -------------------------------------------------------------

    def _coerce_context(
        self,
        context: ClientRequestContext | Mapping[str, Any] | None,
        *,
        allow_none: bool = False,
        **overrides: Any,
    ) -> ClientRequestContext | None:
        if context is None and not overrides:
            if allow_none:
                return None
            if self._default_context is not None:
                base = self._default_context.to_dict()
            else:
                raise LogicPlatformClientError(
                    "ClientRequestContext is required for operational calls"
                )
        elif isinstance(context, ClientRequestContext):
            base = context.to_dict()
        elif isinstance(context, Mapping):
            base = dict(context)
        elif context is None:
            base = (
                self._default_context.to_dict()
                if self._default_context is not None
                else {}
            )
        else:
            raise LogicPlatformClientError(
                "context must be ClientRequestContext, mapping, or None"
            )
        base.update({k: v for k, v in overrides.items() if v is not None})
        if allow_none and not base:
            return None
        # Fill required identity fields with safe offline defaults only when
        # explicitly allowed (construction-time default_context=None path).
        if allow_none and not base.get("task_id"):
            return None
        return ClientRequestContext(
            task_id=str(base.get("task_id") or ""),
            tree_id=str(base.get("tree_id") or ""),
            policy_id=str(base.get("policy_id") or ""),
            plan_id=base.get("plan_id"),
            budget=base.get("budget") or {},
            network_allowed=bool(base.get("network_allowed", False)),
            cancellation=base.get("cancellation"),
            deadline_unix_ms=base.get("deadline_unix_ms"),
            correlation_id=str(base.get("correlation_id") or ""),
            evidence_kind=str(base.get("evidence_kind") or "candidate"),
            authority_ceiling=str(base.get("authority_ceiling") or "advisory"),
        )

    def bind_context(
        self,
        context: ClientRequestContext | Mapping[str, Any] | None = None,
        **overrides: Any,
    ) -> ClientRequestContext:
        """Build or inherit a fail-closed request context."""

        resolved = self._coerce_context(context, allow_none=False, **overrides)
        assert resolved is not None
        return resolved

    def _ensure_handshake(self, *, require_handshake: bool | None = None) -> None:
        required = (
            self._require_handshake
            if require_handshake is None
            else bool(require_handshake)
        )
        if not required:
            return
        if not self._handshake_compatible:
            raise LogicPlatformClientHandshakeError(
                "compatible handshake required before this operation; "
                "call handshake() first"
            )

    # -- handshake / catalog -------------------------------------------------

    def handshake(
        self,
        requirements: Any | None = None,
        *,
        manifest: Any | None = None,
        require_adapter_versions: Sequence[str] | None = None,
    ) -> Any:
        """First lazy datasets load: package-neutral platform handshake."""

        manifest_mod = self._import(_MANIFEST_MODULE)
        if requirements is None:
            adapters = tuple(
                require_adapter_versions
                if require_adapter_versions is not None
                else DEFAULT_REQUIRED_ADAPTER_VERSIONS
            )
            requirements = manifest_mod.HandshakeRequirements(
                required_adapter_versions=adapters,
            )
        elif require_adapter_versions is not None:
            # Merge explicit adapter requirements into a mapping-built request.
            if hasattr(requirements, "required_adapter_versions"):
                existing = tuple(requirements.required_adapter_versions or ())
                merged = existing + tuple(
                    item
                    for item in require_adapter_versions
                    if item not in existing
                )
                requirements = manifest_mod.HandshakeRequirements(
                    required_manifest_interface=getattr(
                        requirements,
                        "required_manifest_interface",
                        manifest_mod.LOGIC_PLATFORM_MANIFEST_INTERFACE,
                    ),
                    required_package_name=getattr(
                        requirements, "required_package_name", None
                    ),
                    min_package_version=getattr(
                        requirements, "min_package_version", None
                    ),
                    exact_package_version=getattr(
                        requirements, "exact_package_version", None
                    ),
                    required_interface_versions=dict(
                        getattr(requirements, "required_interface_versions", {})
                        or {}
                    ),
                    required_schema_roots=dict(
                        getattr(requirements, "required_schema_roots", {}) or {}
                    ),
                    required_operation_versions=dict(
                        getattr(requirements, "required_operation_versions", {})
                        or {}
                    ),
                    required_receipt_versions=dict(
                        getattr(requirements, "required_receipt_versions", {})
                        or {}
                    ),
                    required_plan_versions=dict(
                        getattr(requirements, "required_plan_versions", {}) or {}
                    ),
                    required_adapter_versions=merged,
                    required_catalog_root=getattr(
                        requirements, "required_catalog_root", None
                    ),
                    required_catalog_digest=getattr(
                        requirements, "required_catalog_digest", None
                    ),
                    required_source_commit=getattr(
                        requirements, "required_source_commit", None
                    ),
                    require_source_commit=bool(
                        getattr(requirements, "require_source_commit", False)
                    ),
                )
            elif isinstance(requirements, Mapping):
                payload = dict(requirements)
                existing = tuple(payload.get("required_adapter_versions") or ())
                payload["required_adapter_versions"] = existing + tuple(
                    item
                    for item in require_adapter_versions
                    if item not in existing
                )
                requirements = manifest_mod.HandshakeRequirements(**payload)

        if manifest is None:
            manifest = getattr(
                manifest_mod, "DEFAULT_LOGIC_PLATFORM_MANIFEST", None
            )
            if manifest is None:
                manifest = manifest_mod.build_logic_platform_manifest()
        result = manifest_mod.handshake(requirements, manifest=manifest)
        with self._lock:
            self._handshake_result = result
            self._manifest = manifest
            self._handshake_compatible = bool(getattr(result, "compatible", False))
        return result

    def catalog(self, *, require_handshake: bool | None = None) -> Any:
        """Return the sealed ``CanonicalLogicCatalogSnapshot@1``."""

        self._ensure_handshake(require_handshake=require_handshake)
        catalog_mod = self._import(_CATALOG_MODULE)
        return catalog_mod.DEFAULT_CANONICAL_CATALOG_SNAPSHOT

    def catalog_root(self, *, require_handshake: bool | None = None) -> str:
        snapshot = self.catalog(require_handshake=require_handshake)
        root = getattr(snapshot, "content_root", None)
        if not isinstance(root, str) or not root:
            raise LogicPlatformClientError("catalog content_root is missing")
        return root

    def catalog_digest(self, *, require_handshake: bool | None = None) -> str:
        snapshot = self.catalog(require_handshake=require_handshake)
        digest = getattr(snapshot, "content_digest", None)
        if not isinstance(digest, str) or not digest.startswith("sha256:"):
            raise LogicPlatformClientError("catalog content_digest is missing")
        return digest

    # -- formalize / slice / obligation / plan -------------------------------

    def formalize(
        self,
        artifact: Any | Mapping[str, Any],
        *,
        context: ClientRequestContext | Mapping[str, Any] | None = None,
        require_handshake: bool | None = None,
    ) -> Any:
        """Admit a ``FormalizationArtifact@3`` (datasets-owned semantics)."""

        self._ensure_handshake(require_handshake=require_handshake)
        if context is not None:
            self.bind_context(context)
        artifacts = self._import(_ARTIFACTS_MODULE)
        if isinstance(artifact, artifacts.FormalizationArtifactV3):
            return artifact
        if isinstance(artifact, Mapping):
            return artifacts.FormalizationArtifactV3.from_dict(artifact)
        if hasattr(artifact, "to_dict") and callable(artifact.to_dict):
            return artifacts.FormalizationArtifactV3.from_dict(artifact.to_dict())
        raise LogicPlatformClientError(
            "formalize requires FormalizationArtifact@3 or a mapping body"
        )

    def create_slice(
        self,
        slice_: Any | Mapping[str, Any],
        *,
        context: ClientRequestContext | Mapping[str, Any] | None = None,
        require_handshake: bool | None = None,
    ) -> Any:
        """Admit a ``DomainLogicSlice@2``."""

        self._ensure_handshake(require_handshake=require_handshake)
        if context is not None:
            self.bind_context(context)
        artifacts = self._import(_ARTIFACTS_MODULE)
        if isinstance(slice_, artifacts.DomainLogicSliceV2):
            admitted = slice_
        elif isinstance(slice_, Mapping):
            admitted = artifacts.DomainLogicSliceV2.from_dict(slice_)
        elif hasattr(slice_, "to_dict") and callable(slice_.to_dict):
            admitted = artifacts.DomainLogicSliceV2.from_dict(slice_.to_dict())
        else:
            raise LogicPlatformClientError(
                "create_slice requires DomainLogicSlice@2 or a mapping body"
            )
        status = _enum_value(getattr(admitted, "status", ""))
        if status and status != "admitted":
            raise LogicPlatformClientError(
                f"create_slice rejects non-admitted slice status {status!r}"
            )
        return admitted

    def create_obligation(
        self,
        obligation: Any | Mapping[str, Any] | None = None,
        *,
        slice_: Any | Mapping[str, Any] | None = None,
        bounds: Any | Mapping[str, Any] | None = None,
        evidence_kind: Any | None = None,
        authority_ceiling: Any | None = None,
        obligation_id: str | None = None,
        statement: str | None = None,
        encoding: Any | None = None,
        context: ClientRequestContext | Mapping[str, Any] | None = None,
        require_handshake: bool | None = None,
        metadata: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Admit a ``LogicObligation@2`` (optionally elevated from a slice)."""

        self._ensure_handshake(require_handshake=require_handshake)
        ctx = self.bind_context(context) if context is not None else (
            self._default_context
        )
        requests = self._import(_REQUESTS_MODULE)
        if obligation is not None:
            if isinstance(obligation, requests.LogicObligationV2):
                admitted = obligation
            elif isinstance(obligation, Mapping):
                admitted = requests.LogicObligationV2.from_dict(obligation)
            elif hasattr(obligation, "to_dict") and callable(obligation.to_dict):
                admitted = requests.LogicObligationV2.from_dict(
                    obligation.to_dict()
                )
            else:
                raise LogicPlatformClientError(
                    "create_obligation requires LogicObligation@2 or mapping"
                )
        elif slice_ is not None:
            admitted_slice = self.create_slice(
                slice_,
                context=context,
                require_handshake=False,
            )
            if bounds is None:
                raise LogicPlatformClientError(
                    "create_obligation from slice requires finite bounds"
                )
            if not isinstance(bounds, requests.RequestBounds):
                bounds = requests.RequestBounds.from_dict(
                    bounds if isinstance(bounds, Mapping) else dict(bounds)
                )
            evidence = evidence_kind or (
                ctx.evidence_kind if ctx is not None else "candidate"
            )
            ceiling = authority_ceiling or (
                ctx.authority_ceiling if ctx is not None else "advisory"
            )
            check_authority_overclaim(ceiling, evidence)
            namespaces = self._import(_NAMESPACES_MODULE)
            evidence_identity = (
                evidence
                if hasattr(evidence, "namespace")
                else namespaces.evidence_id(_enum_value(evidence))
            )
            encoding_identity = encoding
            if encoding_identity is None:
                encoding_identity = namespaces.encoding_id("tptp")
            elif isinstance(encoding_identity, str):
                encoding_identity = namespaces.encoding_id(encoding_identity)
            admitted = requests.LogicObligationV2.from_slice(
                admitted_slice,
                obligation_id=obligation_id
                or f"obligation:{getattr(admitted_slice, 'slice_id', 'anon')}",
                statement=statement
                or f"obligation for {getattr(admitted_slice, 'slice_id', 'slice')}",
                encoding=encoding_identity,
                evidence_kind=evidence_identity,
                bounds=bounds,
                authority_ceiling=ceiling,
                metadata=metadata,
                **kwargs,
            )
        else:
            raise LogicPlatformClientError(
                "create_obligation requires obligation or slice_"
            )
        evidence_obj = getattr(admitted, "evidence_kind", None)
        ceiling_obj = getattr(admitted, "authority_ceiling", None)
        if evidence_obj is not None and ceiling_obj is not None:
            check_authority_overclaim(
                ceiling_obj,
                getattr(evidence_obj, "value", evidence_obj),
            )
        return admitted

    def create_plan(
        self,
        plan: Any | Mapping[str, Any],
        *,
        context: ClientRequestContext | Mapping[str, Any] | None = None,
        require_handshake: bool | None = None,
    ) -> Any:
        """Admit a draft ``GoalDirectedProofPlan@1`` (proposal only)."""

        self._ensure_handshake(require_handshake=require_handshake)
        if context is not None:
            self.bind_context(context)
        plan_mod = self._import(_PLAN_MODULE)
        if isinstance(plan, plan_mod.GoalDirectedProofPlan):
            admitted = plan
        elif isinstance(plan, Mapping):
            admitted = plan_mod.GoalDirectedProofPlan.from_dict(plan)
        elif hasattr(plan, "to_dict") and callable(plan.to_dict):
            admitted = plan_mod.GoalDirectedProofPlan.from_dict(plan.to_dict())
        else:
            raise LogicPlatformClientError(
                "create_plan requires GoalDirectedProofPlan@1 or a mapping body"
            )
        if bool(getattr(admitted, "proof_claimed", False)) or bool(
            getattr(admitted, "completion_claimed", False)
        ):
            raise LogicPlatformClientError(
                "create_plan rejects plans that claim proof or completion"
            )
        return admitted

    def create_backend_request(
        self,
        *,
        obligation: Any | Mapping[str, Any] | None = None,
        slice_: Any | Mapping[str, Any] | None = None,
        bounds: Any | Mapping[str, Any] | None = None,
        request_id: str | None = None,
        requested_provider: Any | None = None,
        context: ClientRequestContext | Mapping[str, Any] | None = None,
        require_handshake: bool | None = None,
        metadata: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Elevate an obligation (or slice) into an admitted ``BackendRequest@2``."""

        self._ensure_handshake(require_handshake=require_handshake)
        requests = self._import(_REQUESTS_MODULE)
        if obligation is None:
            obligation = self.create_obligation(
                slice_=slice_,
                bounds=bounds,
                context=context,
                require_handshake=False,
                metadata=metadata,
                **{
                    key: value
                    for key, value in kwargs.items()
                    if key
                    in {
                        "evidence_kind",
                        "authority_ceiling",
                        "statement",
                        "encoding",
                        "obligation_id",
                    }
                },
            )
        elif not isinstance(obligation, requests.LogicObligationV2):
            obligation = self.create_obligation(
                obligation,
                context=context,
                require_handshake=False,
            )
        if not isinstance(obligation, requests.LogicObligationV2):
            raise LogicPlatformClientError(
                "create_backend_request requires an admitted LogicObligation@2"
            )
        return requests.BackendRequestV2.from_obligation(
            obligation,
            request_id=request_id
            or f"request:{getattr(obligation, 'obligation_id', uuid.uuid4().hex)}",
            requested_provider=requested_provider,
            metadata=metadata,
        )

    # -- capability / invoke / reconstruct / verify --------------------------

    def discover_capabilities(
        self,
        *,
        provider_id: str = "",
        feature_query: Sequence[str] = (),
        include_versions: bool = False,
        context: ClientRequestContext | Mapping[str, Any] | None = None,
        require_handshake: bool | None = None,
        **kwargs: Any,
    ) -> ClientInvocationResult:
        """Non-executable capability probe; never mints proof authority."""

        protocol = self._import(_PROTOCOL_MODULE)
        request = protocol.CapabilityRequestV2(
            provider_id=provider_id,
            feature_query=tuple(feature_query),
            include_versions=include_versions,
            **kwargs,
        )
        return self.invoke(
            request,
            context=context,
            require_handshake=require_handshake,
        )

    def invoke(
        self,
        operation: Any | Mapping[str, Any] | str,
        *,
        context: ClientRequestContext | Mapping[str, Any] | None = None,
        require_handshake: bool | None = None,
        provider: Any | None = None,
        simulated: bool = False,
        **kwargs: Any,
    ) -> ClientInvocationResult:
        """Typed ``LogicProviderProtocol@2`` dispatch with fail-closed admission."""

        self._ensure_handshake(require_handshake=require_handshake)
        ctx = self.bind_context(context) if context is not None else self.bind_context(
            self._default_context
        )
        protocol = self._import(_PROTOCOL_MODULE)
        response_mod = self._import(_RESPONSE_MODULE)

        if isinstance(operation, str) and kwargs:
            # Build a minimal request body from operation name + kwargs.
            body: dict[str, Any] = {"operation": operation, **kwargs}
            admitted = protocol.admit_provider_request_v2(body)
        elif isinstance(operation, str):
            raise LogicPlatformClientError(
                "invoke(operation: str) requires typed request fields as kwargs"
            )
        else:
            admitted = protocol.admit_provider_request_v2(operation)

        op_name = _enum_value(getattr(admitted, "operation", operation))
        if protocol.is_executable_operation(op_name):
            check_authority_overclaim(ctx.authority_ceiling, ctx.evidence_kind)
            if getattr(admitted, "bounds", None) is None:
                raise LogicPlatformClientError(
                    f"executable operation {op_name!r} requires positive finite bounds"
                )
            if getattr(admitted, "backend_request", None) is None:
                raise LogicPlatformClientError(
                    f"executable operation {op_name!r} requires admitted "
                    "BackendRequest@2"
                )

        # Re-admit after authority checks (fail closed before dispatch).
        admitted = protocol.admit_provider_request_v2(
            admitted.to_dict() if hasattr(admitted, "to_dict") else admitted
        )

        bound_provider = provider if provider is not None else self._provider
        started = time.monotonic()
        if bound_provider is None:
            # Offline / unbound path: capability may still succeed as advisory.
            if op_name == "capability":
                response = response_mod.ProviderResponseV2.succeeded(
                    request_id=str(getattr(admitted, "request_id", "")),
                    operation=op_name,
                    provider_id="unbound",
                    provider_version="0",
                    evidence_kind="candidate",
                    evidence_authority="advisory",
                    metadata={
                        "unbound": True,
                        "correlation_id": ctx.correlation_id,
                    },
                )
            else:
                raise LogicPlatformClientProviderError(
                    f"invoke({op_name!r}) requires a bound protocol-v2 provider"
                )
        else:
            invoke_fn = getattr(bound_provider, "invoke", None)
            handle_fn = getattr(bound_provider, "handle", None)
            call_fn = invoke_fn or handle_fn or bound_provider
            if not callable(call_fn):
                raise LogicPlatformClientProviderError(
                    "provider must be callable or expose invoke/handle"
                )
            raw = call_fn(admitted)
            if isinstance(raw, response_mod.ProviderResponseV2):
                response = raw
            elif isinstance(raw, Mapping):
                response = response_mod.admit_provider_response_v2(raw)
            elif hasattr(raw, "to_dict") and callable(raw.to_dict):
                response = response_mod.admit_provider_response_v2(raw.to_dict())
            else:
                raise LogicPlatformClientProviderError(
                    "provider returned an unsupported response type"
                )

        duration_ms = int((time.monotonic() - started) * 1000)
        provider_id = str(
            getattr(response, "provider_id", "")
            or getattr(bound_provider, "provider_id", "")
            or ""
        )
        provider_version = str(
            getattr(response, "provider_version", "")
            or getattr(bound_provider, "provider_version", "")
            or ""
        )
        return ClientInvocationResult(
            request_id=str(
                getattr(response, "request_id", None)
                or getattr(admitted, "request_id", "")
            ),
            operation=op_name,
            response=response,
            context=ctx,
            provider_id=provider_id,
            provider_version=provider_version,
            simulated=bool(simulated),
            metadata={
                "duration_ms": duration_ms,
                "correlation_id": ctx.correlation_id,
            },
        )

    def reconstruct(
        self,
        request: Any | Mapping[str, Any],
        *,
        context: ClientRequestContext | Mapping[str, Any] | None = None,
        require_handshake: bool | None = None,
        provider: Any | None = None,
        **kwargs: Any,
    ) -> ClientInvocationResult:
        """Executable reconstruction under finite bounds."""

        protocol = self._import(_PROTOCOL_MODULE)
        if isinstance(request, protocol.ReconstructRequestV2):
            admitted = request
        elif isinstance(request, Mapping):
            body = dict(request)
            body.setdefault("operation", "reconstruct")
            admitted = protocol.admit_provider_request_v2(body)
        else:
            admitted = protocol.admit_provider_request_v2(request)
        return self.invoke(
            admitted,
            context=context,
            require_handshake=require_handshake,
            provider=provider,
            **kwargs,
        )

    def verify(
        self,
        request: Any | Mapping[str, Any],
        *,
        context: ClientRequestContext | Mapping[str, Any] | None = None,
        require_handshake: bool | None = None,
        provider: Any | None = None,
        **kwargs: Any,
    ) -> ClientInvocationResult:
        """Independent verification under finite bounds."""

        protocol = self._import(_PROTOCOL_MODULE)
        if isinstance(request, protocol.VerifyRequestV2):
            admitted = request
        elif isinstance(request, Mapping):
            body = dict(request)
            body.setdefault("operation", "verify")
            admitted = protocol.admit_provider_request_v2(body)
        else:
            admitted = protocol.admit_provider_request_v2(request)
        return self.invoke(
            admitted,
            context=context,
            require_handshake=require_handshake,
            provider=provider,
            **kwargs,
        )

    # -- receipts / counterexamples ------------------------------------------

    def project_receipt(
        self,
        result: ClientInvocationResult | Mapping[str, Any],
        *,
        simulated: bool | None = None,
    ) -> ClientReceiptView:
        """Project an invocation result into an untrusted receipt view."""

        if isinstance(result, Mapping):
            # Minimal mapping admission for tests / offline probes.
            ctx_payload = result.get("context") or {}
            context = (
                ctx_payload
                if isinstance(ctx_payload, ClientRequestContext)
                else ClientRequestContext(
                    task_id=str(ctx_payload.get("task_id") or "task:unknown"),
                    tree_id=str(ctx_payload.get("tree_id") or "tree:unknown"),
                    policy_id=str(ctx_payload.get("policy_id") or "policy:unknown"),
                    plan_id=ctx_payload.get("plan_id"),
                    budget=ctx_payload.get("budget") or {},
                    network_allowed=bool(ctx_payload.get("network_allowed", False)),
                    cancellation=ctx_payload.get("cancellation"),
                    deadline_unix_ms=ctx_payload.get("deadline_unix_ms"),
                    correlation_id=str(ctx_payload.get("correlation_id") or ""),
                    evidence_kind=str(
                        ctx_payload.get("evidence_kind") or "candidate"
                    ),
                    authority_ceiling=str(
                        ctx_payload.get("authority_ceiling") or "advisory"
                    ),
                )
            )
            return ClientReceiptView(
                request_id=str(result.get("request_id") or "request:unknown"),
                operation=str(result.get("operation") or "unknown"),
                provider_id=str(result.get("provider_id") or ""),
                evidence_kind=str(result.get("evidence_kind") or "candidate"),
                evidence_authority=str(
                    result.get("evidence_authority") or "advisory"
                ),
                verdict=str(result.get("verdict") or "unknown"),
                operation_status=str(result.get("operation_status") or "unknown"),
                context=context,
                translation_ids=tuple(result.get("translation_ids") or ()),
                artifact_ids=tuple(result.get("artifact_ids") or ()),
                simulated=bool(
                    result.get("simulated") if simulated is None else simulated
                ),
                authority="advisory",
            )

        if not isinstance(result, ClientInvocationResult):
            raise LogicPlatformClientError(
                "project_receipt requires ClientInvocationResult or mapping"
            )

        response = result.response
        translation_ids: list[str] = []
        artifact_ids: list[str] = []
        translations = getattr(response, "translations", ()) or ()
        for item in translations:
            tid = getattr(item, "translation_id", None)
            if tid is None and isinstance(item, Mapping):
                tid = item.get("translation_id")
            if tid:
                translation_ids.append(str(tid))
        artifacts = getattr(response, "artifacts", ()) or ()
        for item in artifacts:
            aid = getattr(item, "artifact_id", None)
            if aid is None and isinstance(item, Mapping):
                aid = item.get("artifact_id")
            if aid:
                artifact_ids.append(str(aid))

        is_simulated = result.simulated if simulated is None else bool(simulated)
        return ClientReceiptView(
            request_id=result.request_id,
            operation=result.operation,
            provider_id=result.provider_id,
            evidence_kind=result.evidence_kind,
            evidence_authority=result.evidence_authority,
            verdict=result.verdict,
            operation_status=result.operation_status,
            context=result.context,
            translation_ids=tuple(translation_ids),
            artifact_ids=tuple(artifact_ids),
            simulated=is_simulated,
            authority="advisory",
        )

    def project_counterexample(
        self,
        payload: Mapping[str, Any] | ClientInvocationResult,
        *,
        request_id: str | None = None,
    ) -> ClientCounterexampleView:
        """Strip private/raw/source material from a counterexample payload."""

        if isinstance(payload, ClientInvocationResult):
            rid = payload.request_id
            raw = payload.to_dict().get("response") or {}
            if not isinstance(raw, Mapping):
                raw = {"value": raw}
        elif isinstance(payload, Mapping):
            rid = request_id or str(payload.get("request_id") or "request:unknown")
            raw = dict(payload)
        else:
            raise LogicPlatformClientError(
                "project_counterexample requires a mapping or ClientInvocationResult"
            )

        public: dict[str, Any] = {}
        stripped: list[str] = []

        def _walk(source: Mapping[str, Any], dest: dict[str, Any]) -> None:
            for key, value in source.items():
                key_text = str(key)
                if key_text in _PRIVATE_COUNTEREXAMPLE_KEYS:
                    stripped.append(key_text)
                    continue
                if isinstance(value, Mapping):
                    nested: dict[str, Any] = {}
                    _walk(value, nested)
                    dest[key_text] = nested
                else:
                    dest[key_text] = value

        _walk(raw, public)
        return ClientCounterexampleView(
            request_id=rid,
            public_fields=MappingProxyType(public),
            redacted=True,
            authority="advisory",
            stripped_keys=tuple(sorted(set(stripped))),
        )

    # -- cache freshness -----------------------------------------------------

    def build_cache_key(self, **fields: Any) -> Any:
        """Build a datasets-owned ``CanonicalProofCacheKey@1``."""

        cache_mod = self._import(_CACHE_KEY_MODULE)
        # Prefer .build when raw values are supplied; fall through to ctor.
        if all(
            name in fields
            for name in cache_mod.REQUIRED_IDENTITY_FIELDS
        ) and any(
            not (
                isinstance(fields[name], str)
                and (
                    fields[name].startswith("sha256:")
                    or name in {"provider", "checker", "evidence_kind", "authority_ceiling"}
                )
            )
            for name in cache_mod.REQUIRED_IDENTITY_FIELDS
            if name not in {"provider", "checker", "evidence_kind", "authority_ceiling"}
        ):
            try:
                return cache_mod.CanonicalProofCacheKey.build(**fields)
            except TypeError:
                pass
        try:
            return cache_mod.CanonicalProofCacheKey.build(**fields)
        except (TypeError, cache_mod.CanonicalCacheKeyError):
            return cache_mod.CanonicalProofCacheKey(**fields)

    def check_cache_freshness(
        self,
        *,
        request_key: Any | Mapping[str, Any],
        stored_key: Any | Mapping[str, Any] | None = None,
        stored_entry: Mapping[str, Any] | None = None,
        now_unix_s: int | None = None,
    ) -> CacheFreshnessReport:
        """Report cache freshness without raising."""

        cache_mod = self._import(_CACHE_KEY_MODULE)
        try:
            request = cache_mod.admit_canonical_cache_key(request_key)
        except cache_mod.CandidateAsKernelError as error:
            return CacheFreshnessReport(
                fresh=False,
                reason="candidate_as_kernel",
                details={"error": str(error)},
            )
        except Exception as error:  # noqa: BLE001 - fail-closed report
            return CacheFreshnessReport(
                fresh=False,
                reason="unknown_freshness",
                details={"error": str(error)},
            )

        entry = dict(stored_entry or {})
        if bool(entry.get("simulated")) or bool(entry.get("simulated_evidence")):
            return CacheFreshnessReport(
                fresh=False,
                reason="simulated_evidence",
                key_id=request.key_id,
            )

        if stored_key is not None:
            try:
                stored = cache_mod.admit_canonical_cache_key(stored_key)
            except Exception as error:  # noqa: BLE001
                return CacheFreshnessReport(
                    fresh=False,
                    reason="unknown_freshness",
                    key_id=request.key_id,
                    details={"error": str(error)},
                )
            if stored.environment != request.environment:
                return CacheFreshnessReport(
                    fresh=False,
                    reason="environment_mismatch",
                    key_id=request.key_id,
                    details={
                        "stored_environment": stored.environment,
                        "request_environment": request.environment,
                        "cross_environment_hit": True,
                    },
                )
            try:
                cache_mod.admit_cache_hit(stored, request)
            except cache_mod.CrossEnvironmentHitError:
                return CacheFreshnessReport(
                    fresh=False,
                    reason="cross_environment_hit",
                    key_id=request.key_id,
                )
            except Exception as error:  # noqa: BLE001
                return CacheFreshnessReport(
                    fresh=False,
                    reason="stale_entry",
                    key_id=request.key_id,
                    details={"error": str(error)},
                )

        expires_at = entry.get("expires_at_unix_s")
        if expires_at is not None:
            now = int(time.time() if now_unix_s is None else now_unix_s)
            try:
                exp = int(expires_at)
            except (TypeError, ValueError):
                return CacheFreshnessReport(
                    fresh=False,
                    reason="unknown_freshness",
                    key_id=request.key_id,
                    details={"expires_at_unix_s": expires_at},
                )
            if now > exp:
                return CacheFreshnessReport(
                    fresh=False,
                    reason="ttl_expired",
                    key_id=request.key_id,
                    details={"expires_at_unix_s": exp, "now_unix_s": now},
                )

        if bool(entry.get("stale")):
            return CacheFreshnessReport(
                fresh=False,
                reason="stale_entry",
                key_id=request.key_id,
            )

        if stored_key is None and not entry:
            return CacheFreshnessReport(
                fresh=False,
                reason="unknown_freshness",
                key_id=request.key_id,
                details={"note": "no stored key or entry provided"},
            )

        return CacheFreshnessReport(
            fresh=True,
            reason="fresh",
            key_id=request.key_id,
        )

    def require_cache_fresh(
        self,
        *,
        request_key: Any | Mapping[str, Any],
        stored_key: Any | Mapping[str, Any] | None = None,
        stored_entry: Mapping[str, Any] | None = None,
        now_unix_s: int | None = None,
    ) -> CacheFreshnessReport:
        """Raise ``LogicPlatformClientFreshnessError`` when freshness fails."""

        report = self.check_cache_freshness(
            request_key=request_key,
            stored_key=stored_key,
            stored_entry=stored_entry,
            now_unix_s=now_unix_s,
        )
        if not report.fresh:
            raise LogicPlatformClientFreshnessError(
                f"cache entry not fresh: {report.reason}"
            )
        return report


def get_logic_platform_client(
    *,
    reset: bool = False,
    **kwargs: Any,
) -> SupervisorLogicPlatformClient:
    """Return a process-local ``SupervisorLogicPlatformClient`` singleton."""

    global _client_singleton
    with _client_singleton_lock:
        if reset or _client_singleton is None or kwargs:
            _client_singleton = SupervisorLogicPlatformClient(**kwargs)
        return _client_singleton


__all__ = [
    "SUPERVISOR_LOGIC_PLATFORM_CLIENT_INTERFACE",
    "SUPERVISOR_LOGIC_PLATFORM_CLIENT_VERSION",
    "CLIENT_SCHEMA_VERSION",
    "CLIENT_TASK_ID",
    "CLIENT_GOAL_ID",
    "DEFAULT_REQUIRED_ADAPTER_VERSIONS",
    "REQUIRED_CONTEXT_FIELDS",
    "CacheFreshnessReport",
    "ClientCounterexampleView",
    "ClientInvocationResult",
    "ClientReceiptView",
    "ClientRequestContext",
    "LogicPlatformClientAuthorityError",
    "LogicPlatformClientError",
    "LogicPlatformClientFreshnessError",
    "LogicPlatformClientHandshakeError",
    "LogicPlatformClientProviderError",
    "SupervisorLogicPlatformClient",
    "check_authority_overclaim",
    "get_logic_platform_client",
]
