"""Lazy, fail-closed registry for pinned shared-IR artifacts.

The registry is deliberately a small verification boundary.  Capability
discovery only inspects immutable declarations; optional modules and analysis
providers are activated only by an explicit load.  Loaded bodies are returned
to the caller and are never copied into a registry cache.

Artifacts use :class:`~.decision_contracts.PinnedArtifactRef`, so their CIDv1
and supervisor SHA-256 digest independently bind the same canonical DAG-JSON
bytes.  This module additionally verifies the semantic envelope (schema,
producer/configuration, provenance, review/trust, authority, freshness, and
root membership) before an adapter may consume it.
"""

from __future__ import annotations

import importlib
import hashlib
import inspect
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

from ..context.decision_contracts import (
    PinnedArtifactRef,
    ReferenceAuthority,
    canonical_artifact_bytes,
)


IR_REGISTRY_VERSION: Final[int] = 1
IR_CAPABILITY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/ir-capability@1"
)
IR_LOAD_REQUEST_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/ir-load-request@1"
)
IR_LOAD_RESULT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/ir-load-result@1"
)
IR_ARTIFACT_ENVELOPE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/ir-artifact-envelope@1"
)

DEFAULT_MAX_IR_ARTIFACT_BYTES: Final[int] = 4 * 1024 * 1024
DEFAULT_MAX_IR_ITEMS: Final[int] = 16_384
DEFAULT_MAX_IR_DEPTH: Final[int] = 32
DEFAULT_MAX_IR_TEXT_BYTES: Final[int] = 256 * 1024


class IRRegistryError(ValueError):
    """A registry declaration or request is malformed."""


class IRFamily(str, Enum):
    IR_CORE = "ir_core"
    FORMALIZATION = "formalization"
    INTENT = "intent_ir"
    LEGAL = "legal_ir"
    SECURITY = "security_ir"


class IROperation(str, Enum):
    DISCOVER = "discover"
    LOAD = "load"
    LOAD_ARTIFACT = "load"
    VERIFY = "verify"
    VERIFY_ARTIFACT = "verify"
    NORMALIZE = "normalize"


class IRFailureCode(str, Enum):
    """Closed failure vocabulary used at every required-input boundary."""

    UNSUPPORTED = "unsupported"
    UNAVAILABLE = "unavailable"
    PARTIAL = "partial"
    STALE = "stale"
    QUARANTINED = "quarantined"
    AMBIGUOUS = "ambiguous"
    CONTRADICTION = "contradiction"
    BOUNDS = "bounds"
    BOUNDS_EXCEEDED = "bounds"


class IRLoadStatus(str, Enum):
    VERIFIED = "verified"
    UNSUPPORTED = IRFailureCode.UNSUPPORTED.value
    UNAVAILABLE = IRFailureCode.UNAVAILABLE.value
    PARTIAL = IRFailureCode.PARTIAL.value
    STALE = IRFailureCode.STALE.value
    QUARANTINED = IRFailureCode.QUARANTINED.value
    AMBIGUOUS = IRFailureCode.AMBIGUOUS.value
    CONTRADICTION = IRFailureCode.CONTRADICTION.value
    BOUNDS = IRFailureCode.BOUNDS.value
    BOUNDS_EXCEEDED = IRFailureCode.BOUNDS.value

    @property
    def successful(self) -> bool:
        return self is IRLoadStatus.VERIFIED


IRStatus = IRLoadStatus


class IRReviewState(str, Enum):
    REVIEWED = "reviewed"
    APPROVED = "approved"
    VERIFIED = "verified"
    UNREVIEWED = "unreviewed"
    REJECTED = "rejected"

    @property
    def accepted(self) -> bool:
        return self in {
            IRReviewState.REVIEWED,
            IRReviewState.APPROVED,
            IRReviewState.VERIFIED,
        }


class IRTrustState(str, Enum):
    TRUSTED = "trusted"
    VERIFIED = "verified"
    REVIEWED = "reviewed"
    UNTRUSTED = "untrusted"
    UNKNOWN = "unknown"
    QUARANTINED = "quarantined"

    @property
    def accepted(self) -> bool:
        return self in {
            IRTrustState.TRUSTED,
            IRTrustState.VERIFIED,
            IRTrustState.REVIEWED,
        }


class IRDeclaredAuthority(str, Enum):
    AUTHORITATIVE = "authoritative"
    VERIFIED = "verified"
    ADVISORY = "advisory"
    PROPOSAL = "proposal"
    CONTEXT_ONLY = "context_only"
    UNTRUSTED = "untrusted"
    NONE = "none"


_FAMILY_ALIASES: Final[Mapping[str, IRFamily]] = MappingProxyType(
    {
        "core": IRFamily.IR_CORE,
        "shared_ir_core": IRFamily.IR_CORE,
        "formal": IRFamily.FORMALIZATION,
        "intent": IRFamily.INTENT,
        "intentir": IRFamily.INTENT,
        "legal": IRFamily.LEGAL,
        "legalir": IRFamily.LEGAL,
        "security": IRFamily.SECURITY,
        "securityir": IRFamily.SECURITY,
    }
)


def normalize_ir_family(value: IRFamily | str) -> IRFamily:
    if isinstance(value, IRFamily):
        return value
    raw = str(getattr(value, "value", value)).strip().lower().replace("-", "_")
    if raw in _FAMILY_ALIASES:
        return _FAMILY_ALIASES[raw]
    try:
        return IRFamily(raw)
    except ValueError as exc:
        raise IRRegistryError(f"unsupported IR family: {value!r}") from exc


def _text(value: Any, name: str, *, required: bool = True, maximum: int = 8192) -> str:
    if not isinstance(value, str):
        raise IRRegistryError(f"{name} must be a string")
    if value != value.strip():
        raise IRRegistryError(f"{name} must not have surrounding whitespace")
    if required and not value:
        raise IRRegistryError(f"{name} must not be empty")
    if "\x00" in value:
        raise IRRegistryError(f"{name} must not contain NUL")
    if len(value.encode("utf-8")) > maximum:
        raise IRRegistryError(f"{name} exceeds {maximum} UTF-8 bytes")
    return value


def _enum(value: Any, kind: type[Enum], name: str) -> Any:
    if isinstance(value, kind):
        return value
    try:
        return kind(str(getattr(value, "value", value)))
    except (TypeError, ValueError) as exc:
        raise IRRegistryError(
            f"{name} must be one of: " + ", ".join(item.value for item in kind)
        ) from exc


def _strings(
    values: Any, name: str, *, required: bool = False, maximum: int = 256
) -> tuple[str, ...]:
    if isinstance(values, str) or not isinstance(values, Sequence):
        raise IRRegistryError(f"{name} must be a sequence")
    if len(values) > maximum:
        raise IRRegistryError(f"{name} exceeds its count bound")
    result = tuple(_text(item, name) for item in values)
    if required and not result:
        raise IRRegistryError(f"{name} must not be empty")
    if len(result) != len(set(result)):
        raise IRRegistryError(f"{name} contains duplicates")
    return tuple(sorted(result))


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _deep_freeze(item) for key, item in sorted(value.items())}
        )
    if isinstance(value, list):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _count_and_bound(
    value: Any,
    *,
    bounds: "IRRegistryBounds",
    depth: int = 0,
    counter: list[int] | None = None,
) -> None:
    current = counter if counter is not None else [0]
    current[0] += 1
    if current[0] > bounds.max_items:
        raise OverflowError("artifact exceeds max_items")
    if depth > bounds.max_depth:
        raise OverflowError("artifact exceeds max_depth")
    if isinstance(value, str):
        if len(value.encode("utf-8")) > bounds.max_text_bytes:
            raise OverflowError("artifact text exceeds max_text_bytes")
    elif isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("artifact object keys must be strings")
            _count_and_bound(key, bounds=bounds, depth=depth + 1, counter=current)
            _count_and_bound(item, bounds=bounds, depth=depth + 1, counter=current)
    elif isinstance(value, list):
        for item in value:
            _count_and_bound(item, bounds=bounds, depth=depth + 1, counter=current)
    elif value is not None and not isinstance(value, (bool, int)):
        raise TypeError(f"unsupported artifact value {type(value).__name__}")


@dataclass(frozen=True)
class IRRegistryBounds:
    max_artifact_bytes: int = DEFAULT_MAX_IR_ARTIFACT_BYTES
    max_items: int = DEFAULT_MAX_IR_ITEMS
    max_depth: int = DEFAULT_MAX_IR_DEPTH
    max_text_bytes: int = DEFAULT_MAX_IR_TEXT_BYTES

    def __post_init__(self) -> None:
        limits = {
            "max_artifact_bytes": (self.max_artifact_bytes, 64 * 1024 * 1024),
            "max_items": (self.max_items, 1_000_000),
            "max_depth": (self.max_depth, 128),
            "max_text_bytes": (self.max_text_bytes, 4 * 1024 * 1024),
        }
        for name, (value, maximum) in limits.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
                or value > maximum
            ):
                raise IRRegistryError(f"{name} must be an integer from 1 through {maximum}")

    def to_dict(self) -> dict[str, int]:
        return {
            "max_artifact_bytes": self.max_artifact_bytes,
            "max_items": self.max_items,
            "max_depth": self.max_depth,
            "max_text_bytes": self.max_text_bytes,
        }


@dataclass(frozen=True, order=True)
class IRSchemaSupport:
    family: IRFamily
    schema: str
    version: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "family", normalize_ir_family(self.family))
        object.__setattr__(self, "schema", _text(self.schema, "schema", maximum=512))
        object.__setattr__(self, "version", _text(self.version, "version", maximum=128))

    def matches(self, reference: PinnedArtifactRef, family: IRFamily) -> bool:
        return (
            self.family is family
            and self.schema == reference.artifact_schema
            and self.version == reference.artifact_schema_version
        )


def _default_schema_support() -> tuple[IRSchemaSupport, ...]:
    names = {
        IRFamily.IR_CORE: (
            "ipfs_datasets_py/logic/ir-core@1",
            "ipfs_datasets_py/logic/ir_core@1",
            "ipfs_datasets_py.logic.ir_core@1",
            "ir-core@1",
        ),
        IRFamily.FORMALIZATION: (
            "ipfs_datasets_py/logic/formalization@1",
            "ipfs_datasets_py.logic.formalization@1",
            "formalization@1",
        ),
        IRFamily.INTENT: (
            "ipfs_datasets_py/logic/intent-ir@1",
            "ipfs_datasets_py/logic/intent_ir@1",
            "ipfs_datasets_py.logic.intent_ir@1",
            "intent-ir@1",
        ),
        IRFamily.LEGAL: (
            "ipfs_datasets_py/logic/legal-ir@1",
            "ipfs_datasets_py/logic/legal_ir@1",
            "ipfs_datasets_py.logic.legal_ir@1",
            "legal-ir@1",
        ),
        IRFamily.SECURITY: (
            "ipfs_datasets_py/logic/security-ir@1",
            "ipfs_datasets_py/logic/security_ir@1",
            "ipfs_datasets_py.logic.security_ir@1",
            "security-ir@1",
        ),
    }
    return tuple(
        sorted(
            (
                IRSchemaSupport(family=family, schema=schema, version="1")
                for family, schemas in names.items()
                for schema in schemas
            ),
            key=lambda item: (item.family.value, item.schema, item.version),
        )
    )


SUPPORTED_IR_SCHEMAS: Final[tuple[IRSchemaSupport, ...]] = _default_schema_support()


@dataclass(frozen=True)
class IRCapability:
    """Static capability declaration; constructing or inspecting it does no I/O."""

    provider_id: str
    capability_revision: str
    schemas: tuple[IRSchemaSupport, ...] = SUPPORTED_IR_SCHEMAS
    operations: tuple[IROperation, ...] = tuple(IROperation)
    provider_version: str = "unknown"
    remote: bool = False

    def __post_init__(self) -> None:
        for name in ("provider_id", "capability_revision", "provider_version"):
            object.__setattr__(
                self, name, _text(getattr(self, name), name, maximum=512)
            )
        if isinstance(self.schemas, (str, bytes)) or not isinstance(self.schemas, Sequence):
            raise IRRegistryError("schemas must be a sequence")
        schemas = tuple(
            item if isinstance(item, IRSchemaSupport) else IRSchemaSupport(**item)
            for item in self.schemas
        )
        if not schemas or len(schemas) != len(set(schemas)):
            raise IRRegistryError("schemas must be non-empty and unique")
        object.__setattr__(
            self,
            "schemas",
            tuple(sorted(schemas, key=lambda item: (item.family.value, item.schema, item.version))),
        )
        if isinstance(self.operations, str) or not isinstance(self.operations, Sequence):
            raise IRRegistryError("operations must be a sequence")
        operations = tuple(
            sorted(
                {_enum(item, IROperation, "operation") for item in self.operations},
                key=lambda item: item.value,
            )
        )
        if not operations:
            raise IRRegistryError("operations must not be empty")
        object.__setattr__(self, "operations", operations)
        if not isinstance(self.remote, bool):
            raise IRRegistryError("remote must be a boolean")

    @property
    def families(self) -> tuple[IRFamily, ...]:
        return tuple(sorted({item.family for item in self.schemas}, key=lambda item: item.value))

    @property
    def lazy(self) -> bool:
        return True

    @property
    def completion_authority(self) -> bool:
        return False

    def supports(
        self,
        reference: PinnedArtifactRef,
        *,
        family: IRFamily,
        operation: IROperation = IROperation.LOAD,
    ) -> bool:
        return operation in self.operations and any(
            item.matches(reference, family) for item in self.schemas
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": IR_CAPABILITY_SCHEMA,
            "registry_version": IR_REGISTRY_VERSION,
            "provider_id": self.provider_id,
            "provider_version": self.provider_version,
            "capability_revision": self.capability_revision,
            "families": [item.value for item in self.families],
            "schemas": [
                {
                    "family": item.family.value,
                    "schema": item.schema,
                    "version": item.version,
                }
                for item in self.schemas
            ],
            "operations": [item.value for item in self.operations],
            "remote": self.remote,
            "lazy": True,
        }


@dataclass(frozen=True)
class IRLoadRequest:
    reference: PinnedArtifactRef
    family: IRFamily
    root_reference: PinnedArtifactRef | None = None
    required: bool = True
    provider_id: str = ""
    producer_configuration_id: str = ""
    bounds: IRRegistryBounds = field(default_factory=IRRegistryBounds)

    def __post_init__(self) -> None:
        if not isinstance(self.reference, PinnedArtifactRef):
            raise IRRegistryError("reference must be a PinnedArtifactRef")
        object.__setattr__(self, "family", normalize_ir_family(self.family))
        if self.root_reference is not None and not isinstance(
            self.root_reference, PinnedArtifactRef
        ):
            raise IRRegistryError("root_reference must be a PinnedArtifactRef")
        if not isinstance(self.required, bool):
            raise IRRegistryError("required must be a boolean")
        for name in ("provider_id", "producer_configuration_id"):
            object.__setattr__(
                self,
                name,
                _text(getattr(self, name), name, required=False, maximum=512),
            )
        if not isinstance(self.bounds, IRRegistryBounds):
            raise IRRegistryError("bounds must be IRRegistryBounds")

    @property
    def effective_root(self) -> PinnedArtifactRef:
        return self.root_reference or self.reference

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": IR_LOAD_REQUEST_SCHEMA,
            "registry_version": IR_REGISTRY_VERSION,
            "reference": self.reference.to_dict(),
            "family": self.family.value,
            "root_reference": (
                self.root_reference.to_dict()
                if self.root_reference is not None
                else None
            ),
            "required": self.required,
            "provider_id": self.provider_id,
            "producer_configuration_id": self.producer_configuration_id,
            "bounds": self.bounds.to_dict(),
        }

    @property
    def content_id(self) -> str:
        encoded = canonical_artifact_bytes(self.to_dict())
        return "ir-load-request:sha256:" + hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class IRFailure:
    code: IRFailureCode
    reason: str
    required: bool
    artifact_id: str
    provider_id: str = ""
    details: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "code", _enum(self.code, IRFailureCode, "failure code"))
        for name in ("reason", "artifact_id", "provider_id"):
            object.__setattr__(
                self,
                name,
                _text(
                    getattr(self, name),
                    name,
                    required=name != "provider_id",
                    maximum=2048,
                ),
            )
        if not isinstance(self.required, bool):
            raise IRRegistryError("failure.required must be a boolean")
        object.__setattr__(
            self, "details", _strings(self.details, "details", maximum=256)
        )

    @property
    def fail_closed(self) -> bool:
        return self.required

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code.value,
            "reason": self.reason,
            "required": self.required,
            "artifact_id": self.artifact_id,
            "provider_id": self.provider_id,
            "details": list(self.details),
            "fail_closed": self.fail_closed,
        }


@dataclass(frozen=True)
class VerifiedIRArtifact:
    """Ephemeral verified artifact; registry instances never retain this body."""

    reference: PinnedArtifactRef
    root_reference: PinnedArtifactRef
    family: IRFamily
    payload: Mapping[str, Any]
    producer_configuration_id: str
    provenance: tuple[Mapping[str, Any], ...]
    review_state: IRReviewState
    trust_state: IRTrustState
    declared_authority: IRDeclaredAuthority
    provider_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.payload, Mapping):
            raise IRRegistryError("verified payload must be an object")
        object.__setattr__(self, "payload", _deep_freeze(dict(self.payload)))
        object.__setattr__(
            self,
            "provenance",
            tuple(_deep_freeze(dict(item)) for item in self.provenance),
        )

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_artifact_bytes(self.payload)

    @property
    def body_cached(self) -> bool:
        return False

    def to_dict(self) -> dict[str, Any]:
        """Return a body-free verification receipt."""

        reference_fields = {
            "artifact_id",
            "capability_id",
            "cid",
            "cid_v1",
            "configuration_id",
            "digest",
            "evidence_id",
            "producer_id",
            "record_id",
            "reference_id",
            "revision",
            "source_id",
            "span_id",
            "supervisor_digest",
            "uri",
        }
        compact_provenance = [
            {
                key: item
                for key, item in provenance.items()
                if key in reference_fields
                and (item is None or isinstance(item, (str, bool, int)))
            }
            for provenance in self.provenance
        ]
        return {
            "schema": IR_ARTIFACT_ENVELOPE_SCHEMA,
            "registry_version": IR_REGISTRY_VERSION,
            "reference": self.reference.to_dict(),
            "root_reference": self.root_reference.to_dict(),
            "family": self.family.value,
            "producer_configuration_id": self.producer_configuration_id,
            "provenance": compact_provenance,
            "review_state": self.review_state.value,
            "trust_state": self.trust_state.value,
            "declared_authority": self.declared_authority.value,
            "provider_id": self.provider_id,
            "canonical_bytes_verified": True,
            "root_membership_verified": True,
            "body_cached": False,
        }


@dataclass(frozen=True)
class IRLoadResult:
    status: IRLoadStatus
    request: IRLoadRequest
    artifact: VerifiedIRArtifact | None = None
    failure: IRFailure | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", _enum(self.status, IRLoadStatus, "status"))
        if self.status.successful:
            if self.artifact is None or self.failure is not None:
                raise IRRegistryError("verified results require only an artifact")
        elif self.failure is None or self.artifact is not None:
            raise IRRegistryError("failed results require only a typed failure")
        if self.failure is not None and self.failure.code.value != self.status.value:
            raise IRRegistryError("failure code must match result status")

    @property
    def successful(self) -> bool:
        return self.status.successful

    @property
    def usable(self) -> bool:
        return self.successful

    @property
    def accepted(self) -> bool:
        return self.successful

    @property
    def failure_code(self) -> IRFailureCode | None:
        return self.failure.code if self.failure is not None else None

    @property
    def fail_closed(self) -> bool:
        return bool(self.failure and self.failure.fail_closed)

    def require_artifact(self) -> VerifiedIRArtifact:
        if self.artifact is None:
            assert self.failure is not None
            raise IRRegistryError(
                f"required IR artifact failed closed: {self.failure.code.value}: "
                f"{self.failure.reason}"
            )
        return self.artifact

    def __bool__(self) -> bool:
        raise TypeError("IRLoadResult has no truth value; inspect status explicitly")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": IR_LOAD_RESULT_SCHEMA,
            "registry_version": IR_REGISTRY_VERSION,
            "status": self.status.value,
            "request_id": self.request.content_id,
            "artifact": self.artifact.to_dict() if self.artifact is not None else None,
            "failure": self.failure.to_dict() if self.failure is not None else None,
            "successful": self.successful,
            "fail_closed": self.fail_closed,
        }

    @property
    def content_id(self) -> str:
        encoded = canonical_artifact_bytes(self.to_dict())
        return "ir-load-result:sha256:" + hashlib.sha256(encoded).hexdigest()


@dataclass
class _ProviderRegistration:
    capability: IRCapability
    loader: Callable[[IRLoadRequest], Any] | None = None
    factory: Callable[[], Any] | None = None
    instance: Any = None
    activated: bool = False


def _metadata_object(payload: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = payload.get(name)
    if not isinstance(value, Mapping):
        raise KeyError(name)
    return value


def _metadata_text(
    payload: Mapping[str, Any],
    name: str,
    *,
    aliases: tuple[str, ...] = (),
) -> str:
    for candidate in (name, *aliases):
        value = payload.get(candidate)
        if isinstance(value, str) and value:
            return value
    raise KeyError(name)


def _authority_allowed(
    declared: IRDeclaredAuthority, reference: ReferenceAuthority
) -> bool:
    if declared is IRDeclaredAuthority.AUTHORITATIVE:
        return reference is ReferenceAuthority.AUTHORITATIVE
    if declared is IRDeclaredAuthority.VERIFIED:
        return reference in {
            ReferenceAuthority.AUTHORITATIVE,
            ReferenceAuthority.VERIFIED,
        }
    if declared is IRDeclaredAuthority.UNTRUSTED:
        return reference is ReferenceAuthority.UNTRUSTED
    return True


class IRRegistry:
    """Capability-negotiated registry with lazy providers and no body cache."""

    def __init__(self, *, bounds: IRRegistryBounds | None = None) -> None:
        self.bounds = bounds or IRRegistryBounds()
        self._providers: dict[str, _ProviderRegistration] = {}
        self._provider_order: list[str] = []
        self._local_artifacts: dict[tuple[str, str], bytes | Path] = {}
        self.register_provider(
            IRCapability(
                provider_id="supervisor-local-ir",
                provider_version="1",
                capability_revision="builtin@1",
            ),
            loader=self._load_local,
        )

    def register_provider(
        self,
        capability: IRCapability,
        *,
        loader: Callable[[IRLoadRequest], Any],
        replace_existing: bool = False,
    ) -> None:
        if not isinstance(capability, IRCapability):
            raise IRRegistryError("capability must be IRCapability")
        if not callable(loader):
            raise IRRegistryError("loader must be callable")
        self._register(
            _ProviderRegistration(
                capability=capability,
                loader=loader,
                instance=loader,
                activated=True,
            ),
            replace_existing=replace_existing,
        )

    def register_lazy_provider(
        self,
        capability: IRCapability,
        *,
        factory: Callable[[], Any],
        replace_existing: bool = False,
    ) -> None:
        if not isinstance(capability, IRCapability):
            raise IRRegistryError("capability must be IRCapability")
        if not callable(factory):
            raise IRRegistryError("factory must be callable")
        self._register(
            _ProviderRegistration(capability=capability, factory=factory),
            replace_existing=replace_existing,
        )

    def register_optional_module(
        self,
        capability: IRCapability,
        *,
        module_name: str = "ipfs_datasets_py",
        attribute: str = "ir_artifact_provider",
        replace_existing: bool = False,
    ) -> None:
        """Declare an optional provider without importing or probing it."""

        module = _text(module_name, "module_name", maximum=512)
        member = _text(attribute, "attribute", maximum=256)

        def factory() -> Any:
            imported = importlib.import_module(module)
            provider = getattr(imported, member)
            return provider() if inspect.isclass(provider) else provider

        self.register_lazy_provider(
            capability, factory=factory, replace_existing=replace_existing
        )

    register_ipfs_datasets_provider = register_optional_module

    def register_analysis_transport(
        self,
        capability: IRCapability,
        *,
        transport: Any,
        analysis_provider_id: str = "",
        operation: str = "load_pinned_ir",
        resolver: Callable[[Mapping[str, Any], IRLoadRequest], Any] | None = None,
        replace_existing: bool = False,
    ) -> None:
        """Route optional remote location discovery through analysis transport.

        The transport is used only to obtain a compact location/reference.  It
        cannot carry the IR body or confer authority.  Exact bytes are read
        from the returned absolute path, or supplied by an explicit resolver,
        and pass through the same local canonical verification afterward.
        """

        operation_name = _text(operation, "analysis operation", maximum=256)
        preferred = _text(
            analysis_provider_id,
            "analysis_provider_id",
            required=False,
            maximum=256,
        )
        if not callable(getattr(transport, "dispatch", None)):
            raise IRRegistryError("transport must provide async dispatch")
        if resolver is not None and not callable(resolver):
            raise IRRegistryError("resolver must be callable")

        async def load_through_transport(request: IRLoadRequest) -> bytes:
            # Keep the dependency lazy: registering and discovering this
            # capability does not import or initialize the transport module.
            from ..analysis.analysis_transport import (
                AnalysisRequest,
                AnalysisTransportStatus,
            )

            root = request.effective_root
            transport_request = AnalysisRequest(
                operation=operation_name,
                question="Resolve the exact pinned IR artifact location.",
                preferred_provider_id=preferred,
                artifact_references=(
                    {
                        "artifact_id": request.reference.artifact_id,
                        "cid": request.reference.cid_v1,
                        "digest": request.reference.supervisor_digest,
                        "byte_count": request.reference.size_bytes,
                        "kind": request.reference.artifact_kind,
                        "producer_id": request.reference.producer_id,
                    },
                    {
                        "artifact_id": root.artifact_id,
                        "cid": root.cid_v1,
                        "digest": root.supervisor_digest,
                        "byte_count": root.size_bytes,
                        "kind": "semantic_root",
                        "producer_id": root.producer_id,
                    },
                ),
                metadata={
                    "family": request.family.value,
                    "artifact_schema": request.reference.artifact_schema,
                    "artifact_schema_version": (
                        request.reference.artifact_schema_version
                    ),
                },
            )
            result = await transport.dispatch(transport_request)
            if result.status not in {
                AnalysisTransportStatus.COMPLETED,
                AnalysisTransportStatus.FALLBACK,
            }:
                raise FileNotFoundError(
                    f"analysis transport: {result.status.value}"
                )
            if result.truncated:
                raise FileNotFoundError("analysis transport result was truncated")
            matches = tuple(
                item
                for item in result.evidence_references
                if (
                    item.get("artifact_id") == request.reference.artifact_id
                    and item.get("digest") == request.reference.supervisor_digest
                    and item.get("cid", request.reference.cid_v1)
                    == request.reference.cid_v1
                )
            )
            if len(matches) != 1:
                raise FileNotFoundError(
                    "analysis transport did not return one exact pinned location"
                )
            location = matches[0]
            if resolver is not None:
                value = resolver(location, request)
                if inspect.isawaitable(value):
                    value = await value
                if not isinstance(value, bytes):
                    raise TypeError("IR location resolver must return bytes")
                return value
            path_value = location.get("path")
            if not isinstance(path_value, str):
                raise FileNotFoundError(
                    "remote reference requires an explicit byte resolver"
                )
            path = Path(path_value)
            if not path.is_absolute():
                raise FileNotFoundError("transport artifact path must be absolute")
            with path.open("rb") as stream:
                return stream.read(
                    min(
                        self.bounds.max_artifact_bytes,
                        request.bounds.max_artifact_bytes,
                    )
                    + 1
                )

        self.register_lazy_provider(
            capability,
            factory=lambda: load_through_transport,
            replace_existing=replace_existing,
        )

    register_analysis_transport_provider = register_analysis_transport

    def _register(
        self, registration: _ProviderRegistration, *, replace_existing: bool
    ) -> None:
        provider_id = registration.capability.provider_id
        if provider_id in self._providers and not replace_existing:
            raise IRRegistryError(f"provider already registered: {provider_id}")
        if provider_id not in self._providers:
            self._provider_order.append(provider_id)
        self._providers[provider_id] = registration

    def discover_capabilities(
        self,
        family: IRFamily | str | None = None,
        operation: IROperation | str | None = None,
    ) -> tuple[IRCapability, ...]:
        """Return declarations only; never import, instantiate, probe, or read."""

        normalized_family = normalize_ir_family(family) if family is not None else None
        normalized_operation = (
            _enum(operation, IROperation, "operation") if operation is not None else None
        )
        return tuple(
            registration.capability
            for provider_id in self._provider_order
            for registration in (self._providers[provider_id],)
            if (
                normalized_family is None
                or normalized_family in registration.capability.families
            )
            and (
                normalized_operation is None
                or normalized_operation in registration.capability.operations
            )
        )

    discover = discover_capabilities

    def supported_schemas(
        self, family: IRFamily | str | None = None
    ) -> tuple[IRSchemaSupport, ...]:
        normalized = normalize_ir_family(family) if family is not None else None
        return tuple(
            sorted(
                {
                    schema
                    for capability in self.discover_capabilities(normalized)
                    for schema in capability.schemas
                    if normalized is None or schema.family is normalized
                },
                key=lambda item: (item.family.value, item.schema, item.version),
            )
        )

    def register_local_artifact(
        self, reference: PinnedArtifactRef, canonical_bytes: bytes
    ) -> None:
        """Register deterministic immutable fixture bytes after exact verification."""

        if not isinstance(reference, PinnedArtifactRef):
            raise IRRegistryError("reference must be a PinnedArtifactRef")
        if not reference.verify_canonical_bytes(canonical_bytes):
            raise IRRegistryError("local artifact bytes do not match pinned reference")
        self._local_artifacts[
            (reference.cid_v1, reference.supervisor_digest)
        ] = canonical_bytes

    def register_local_path(
        self, reference: PinnedArtifactRef, path: str | Path
    ) -> None:
        """Register an exact file path; bytes are read and verified only on load."""

        if not isinstance(reference, PinnedArtifactRef):
            raise IRRegistryError("reference must be a PinnedArtifactRef")
        candidate = Path(path)
        if not candidate.is_absolute():
            raise IRRegistryError("local artifact path must be absolute")
        self._local_artifacts[
            (reference.cid_v1, reference.supervisor_digest)
        ] = candidate

    def _load_local(self, request: IRLoadRequest) -> bytes:
        source = self._local_artifacts.get(
            (request.reference.cid_v1, request.reference.supervisor_digest)
        )
        if source is None:
            raise FileNotFoundError(request.reference.artifact_id)
        if isinstance(source, bytes):
            return source
        with source.open("rb") as stream:
            return stream.read(
                min(
                    self.bounds.max_artifact_bytes,
                    request.bounds.max_artifact_bytes,
                )
                + 1
            )

    def _candidate_registrations(
        self, request: IRLoadRequest
    ) -> tuple[_ProviderRegistration, ...]:
        return tuple(
            self._providers[provider_id]
            for provider_id in self._provider_order
            if (
                not request.provider_id or provider_id == request.provider_id
            )
            and self._providers[provider_id].capability.supports(
                request.reference, family=request.family
            )
        )

    def _failure(
        self,
        request: IRLoadRequest,
        code: IRFailureCode,
        reason: str,
        *,
        provider_id: str = "",
        details: Sequence[str] = (),
    ) -> IRLoadResult:
        return IRLoadResult(
            status=IRLoadStatus(code.value),
            request=request,
            failure=IRFailure(
                code=code,
                reason=reason,
                required=request.required,
                artifact_id=request.reference.artifact_id,
                provider_id=provider_id,
                details=tuple(details),
            ),
        )

    def load(self, request: IRLoadRequest) -> IRLoadResult:
        """Synchronously load and verify an artifact from the first exact provider."""

        if not isinstance(request, IRLoadRequest):
            raise IRRegistryError("request must be IRLoadRequest")
        preflight = self._preflight_bounds(request)
        if preflight is not None:
            return preflight
        candidates = self._candidate_registrations(request)
        if not candidates:
            return self._failure(
                request,
                IRFailureCode.UNSUPPORTED,
                "no declared provider supports the exact family/schema/version",
                provider_id=request.provider_id,
            )
        unavailable: list[str] = []
        for registration in candidates:
            try:
                loader = self._activate(registration)
                value = loader(request)
                if inspect.isawaitable(value):
                    close = getattr(value, "close", None)
                    if callable(close):
                        close()
                    unavailable.append(
                        f"{registration.capability.provider_id}: asynchronous loader"
                    )
                    continue
                return self._verify(
                    request, value, provider_id=registration.capability.provider_id
                )
            except (FileNotFoundError, ModuleNotFoundError, ImportError, AttributeError) as exc:
                unavailable.append(
                    f"{registration.capability.provider_id}: {type(exc).__name__}"
                )
            except Exception as exc:
                unavailable.append(
                    f"{registration.capability.provider_id}: {type(exc).__name__}"
                )
        return self._failure(
            request,
            IRFailureCode.UNAVAILABLE,
            "all supporting providers were unavailable",
            provider_id=request.provider_id,
            details=unavailable,
        )

    def load_required(
        self,
        reference: PinnedArtifactRef,
        family: IRFamily | str,
        **kwargs: Any,
    ) -> VerifiedIRArtifact:
        """Load one mandatory input or raise a typed fail-closed error."""

        request = IRLoadRequest(
            reference=reference,
            family=normalize_ir_family(family),
            required=True,
            **kwargs,
        )
        return self.load(request).require_artifact()

    async def load_async(self, request: IRLoadRequest) -> IRLoadResult:
        """Async load facade supporting both sync and async lazy providers."""

        if not isinstance(request, IRLoadRequest):
            raise IRRegistryError("request must be IRLoadRequest")
        preflight = self._preflight_bounds(request)
        if preflight is not None:
            return preflight
        candidates = self._candidate_registrations(request)
        if not candidates:
            return self._failure(
                request,
                IRFailureCode.UNSUPPORTED,
                "no declared provider supports the exact family/schema/version",
                provider_id=request.provider_id,
            )
        unavailable: list[str] = []
        for registration in candidates:
            try:
                loader = self._activate(registration)
                value = loader(request)
                if inspect.isawaitable(value):
                    value = await value
                return self._verify(
                    request, value, provider_id=registration.capability.provider_id
                )
            except (FileNotFoundError, ModuleNotFoundError, ImportError, AttributeError) as exc:
                unavailable.append(
                    f"{registration.capability.provider_id}: {type(exc).__name__}"
                )
            except Exception as exc:
                unavailable.append(
                    f"{registration.capability.provider_id}: {type(exc).__name__}"
                )
        return self._failure(
            request,
            IRFailureCode.UNAVAILABLE,
            "all supporting providers were unavailable",
            provider_id=request.provider_id,
            details=unavailable,
        )

    def _preflight_bounds(self, request: IRLoadRequest) -> IRLoadResult | None:
        maximum = min(
            self.bounds.max_artifact_bytes, request.bounds.max_artifact_bytes
        )
        if request.reference.size_bytes > maximum:
            return self._failure(
                request,
                IRFailureCode.BOUNDS,
                "pinned artifact size exceeds the effective pre-load bound",
                provider_id=request.provider_id,
            )
        return None

    def _activate(self, registration: _ProviderRegistration) -> Callable[[IRLoadRequest], Any]:
        if not registration.activated:
            assert registration.factory is not None
            registration.instance = registration.factory()
            registration.activated = True
        provider = registration.instance
        if callable(provider):
            return provider
        for name in ("load_ir_artifact", "load_artifact", "load"):
            loader = getattr(provider, name, None)
            if callable(loader):
                return loader
        raise AttributeError("provider has no IR artifact load operation")

    def _verify(
        self, request: IRLoadRequest, value: Any, *, provider_id: str
    ) -> IRLoadResult:
        if isinstance(value, Mapping):
            value = canonical_artifact_bytes(value)
        if not isinstance(value, bytes):
            return self._failure(
                request,
                IRFailureCode.UNAVAILABLE,
                "provider did not return canonical artifact bytes",
                provider_id=provider_id,
            )
        effective_max = min(
            self.bounds.max_artifact_bytes, request.bounds.max_artifact_bytes
        )
        if len(value) > effective_max or len(value) != request.reference.size_bytes:
            return self._failure(
                request,
                IRFailureCode.BOUNDS,
                "artifact byte size exceeds a bound or its pinned size",
                provider_id=provider_id,
            )
        if not request.reference.verify_canonical_bytes(value):
            return self._failure(
                request,
                IRFailureCode.QUARANTINED,
                "canonical bytes, CIDv1, or supervisor digest verification failed",
                provider_id=provider_id,
            )
        try:
            payload = json.loads(value)
            _count_and_bound(
                payload,
                bounds=IRRegistryBounds(
                    max_artifact_bytes=effective_max,
                    max_items=min(self.bounds.max_items, request.bounds.max_items),
                    max_depth=min(self.bounds.max_depth, request.bounds.max_depth),
                    max_text_bytes=min(
                        self.bounds.max_text_bytes,
                        request.bounds.max_text_bytes,
                    ),
                ),
            )
        except (json.JSONDecodeError, UnicodeDecodeError, TypeError, OverflowError) as exc:
            return self._failure(
                request,
                IRFailureCode.BOUNDS,
                f"artifact structure is outside bounds: {type(exc).__name__}",
                provider_id=provider_id,
            )
        if not isinstance(payload, Mapping):
            return self._failure(
                request,
                IRFailureCode.QUARANTINED,
                "IR artifact must be a canonical object",
                provider_id=provider_id,
            )
        return self._verify_envelope(request, payload, provider_id=provider_id)

    def _verify_envelope(
        self,
        request: IRLoadRequest,
        payload: Mapping[str, Any],
        *,
        provider_id: str,
    ) -> IRLoadResult:
        ref = request.reference
        try:
            schema = _metadata_text(payload, "schema", aliases=("artifact_schema",))
            version = _metadata_text(
                payload, "schema_version", aliases=("artifact_schema_version", "version")
            )
            family = normalize_ir_family(
                _metadata_text(payload, "family", aliases=("ir_family", "artifact_kind"))
            )
        except (KeyError, IRRegistryError):
            return self._failure(
                request,
                IRFailureCode.PARTIAL,
                "artifact omits schema, version, or IR family",
                provider_id=provider_id,
            )
        if (
            schema != ref.artifact_schema
            or version != ref.artifact_schema_version
            or family is not request.family
        ):
            return self._failure(
                request,
                IRFailureCode.UNSUPPORTED,
                "artifact envelope does not match the pinned schema/version/family",
                provider_id=provider_id,
            )

        producer = payload.get("producer")
        configuration_id = ""
        if isinstance(producer, Mapping):
            producer_id = producer.get("id", producer.get("producer_id"))
            configuration_id = producer.get(
                "configuration_id",
                producer.get(
                    "configuration",
                    producer.get(
                        "configuration_digest",
                        producer.get("configuration_revision"),
                    ),
                ),
            )
        else:
            producer_id = producer or payload.get("producer_id")
            configuration_id = payload.get("producer_configuration_id")
        if producer_id != ref.producer_id or not isinstance(configuration_id, str) or not configuration_id:
            return self._failure(
                request,
                IRFailureCode.QUARANTINED,
                "producer or producer configuration is missing or mismatched",
                provider_id=provider_id,
            )
        if (
            request.producer_configuration_id
            and configuration_id != request.producer_configuration_id
        ):
            return self._failure(
                request,
                IRFailureCode.STALE,
                "producer configuration does not match the pinned request",
                provider_id=provider_id,
            )

        provenance = payload.get("provenance")
        if isinstance(provenance, Mapping):
            provenance = (provenance,)
        if (
            isinstance(provenance, (str, bytes))
            or not isinstance(provenance, Sequence)
            or not provenance
            or any(not isinstance(item, Mapping) or not item for item in provenance)
        ):
            return self._failure(
                request,
                IRFailureCode.PARTIAL,
                "artifact requires non-empty structured provenance",
                provider_id=provider_id,
            )

        try:
            review_value = payload.get("review")
            review_raw = (
                review_value.get("status")
                if isinstance(review_value, Mapping)
                else review_value or payload.get("review_state")
            )
            trust_value = payload.get("trust")
            trust_raw = (
                trust_value.get("state")
                if isinstance(trust_value, Mapping)
                else trust_value or payload.get("trust_state")
            )
            review = _enum(review_raw, IRReviewState, "review state")
            trust = _enum(trust_raw, IRTrustState, "trust state")
            authority_value = payload.get(
                "authority", payload.get("declared_authority")
            )
            if isinstance(authority_value, Mapping):
                authority_value = authority_value.get(
                    "class",
                    authority_value.get(
                        "authority", authority_value.get("declared_authority")
                    ),
                )
            authority = _enum(
                authority_value,
                IRDeclaredAuthority,
                "declared authority",
            )
        except IRRegistryError:
            return self._failure(
                request,
                IRFailureCode.PARTIAL,
                "review, trust, or declared authority is missing or unsupported",
                provider_id=provider_id,
            )
        if trust is IRTrustState.QUARANTINED or review is IRReviewState.REJECTED:
            return self._failure(
                request,
                IRFailureCode.QUARANTINED,
                "artifact is quarantined or rejected",
                provider_id=provider_id,
            )
        if not review.accepted or not trust.accepted:
            return self._failure(
                request,
                IRFailureCode.QUARANTINED,
                "artifact review/trust state is not accepted",
                provider_id=provider_id,
            )
        if not _authority_allowed(authority, ref.authority):
            return self._failure(
                request,
                IRFailureCode.QUARANTINED,
                "declared authority exceeds the pinned reference authority",
                provider_id=provider_id,
            )

        freshness = payload.get("freshness", "fresh")
        if freshness not in {"fresh", "current"}:
            return self._failure(
                request,
                IRFailureCode.STALE,
                "artifact freshness is not current",
                provider_id=provider_id,
            )
        if payload.get("partial") is True or payload.get("truncated") is True:
            return self._failure(
                request,
                IRFailureCode.PARTIAL,
                "artifact declares partial or truncated coverage",
                provider_id=provider_id,
            )
        ambiguities = payload.get("ambiguities", ())
        if isinstance(ambiguities, Sequence) and not isinstance(ambiguities, str) and ambiguities:
            return self._failure(
                request,
                IRFailureCode.AMBIGUOUS,
                "artifact contains unresolved ambiguities",
                provider_id=provider_id,
            )
        contradictions = payload.get("contradictions", ())
        if (
            isinstance(contradictions, Sequence)
            and not isinstance(contradictions, str)
            and contradictions
        ):
            return self._failure(
                request,
                IRFailureCode.CONTRADICTION,
                "artifact contains unresolved contradictions",
                provider_id=provider_id,
            )

        root = request.effective_root
        if ref != root:
            membership = payload.get(
                "root_membership", payload.get("semantic_root")
            )
            if not isinstance(membership, Mapping):
                return self._failure(
                    request,
                    IRFailureCode.PARTIAL,
                    "non-root artifact omits root membership",
                    provider_id=provider_id,
                )
            if (
                membership.get("root_cid_v1", membership.get("cid_v1")) != root.cid_v1
                or membership.get(
                    "root_supervisor_digest", membership.get("supervisor_digest")
                )
                != root.supervisor_digest
                or membership.get("member_cid_v1", ref.cid_v1) != ref.cid_v1
            ):
                return self._failure(
                    request,
                    IRFailureCode.STALE,
                    "artifact is not a member of the requested pinned root",
                    provider_id=provider_id,
                )

        artifact = VerifiedIRArtifact(
            reference=ref,
            root_reference=root,
            family=request.family,
            payload=payload,
            producer_configuration_id=configuration_id,
            provenance=tuple(provenance),
            review_state=review,
            trust_state=trust,
            declared_authority=authority,
            provider_id=provider_id,
        )
        return IRLoadResult(
            status=IRLoadStatus.VERIFIED,
            request=request,
            artifact=artifact,
        )


def deterministic_ir_fixture(
    family: IRFamily | str,
    *,
    declarations: Sequence[Mapping[str, Any]] = (),
    formal_views: Sequence[Mapping[str, Any]] = (),
    claims: Sequence[Mapping[str, Any]] = (),
    assumptions: Sequence[Mapping[str, Any]] = (),
    obligations: Sequence[Mapping[str, Any]] = (),
    result_authority: Mapping[str, Any] | Sequence[Mapping[str, Any]] | str = (),
    artifact_id: str | None = None,
    producer_id: str = "producer:deterministic-ir-fixture",
    producer_configuration_id: str = "configuration:deterministic@1",
    authority: IRDeclaredAuthority | str = IRDeclaredAuthority.VERIFIED,
    reference_authority: ReferenceAuthority | str = ReferenceAuthority.VERIFIED,
    root_reference: PinnedArtifactRef | None = None,
    schema: str | None = None,
    schema_version: str = "1",
    updates: Mapping[str, Any] | None = None,
) -> tuple[PinnedArtifactRef, bytes]:
    """Build deterministic, production-shaped local fixture bytes and reference."""

    normalized_family = normalize_ir_family(family)
    declared = _enum(authority, IRDeclaredAuthority, "authority")
    selected_schema = schema or next(
        item.schema for item in SUPPORTED_IR_SCHEMAS if item.family is normalized_family
    )
    payload: dict[str, Any] = {
        "schema": selected_schema,
        "schema_version": schema_version,
        "family": normalized_family.value,
        "producer": {
            "id": producer_id,
            "configuration_id": producer_configuration_id,
        },
        "provenance": [
            {
                "source_id": "source:deterministic-fixture",
                "span_id": "span:0",
            }
        ],
        "review": {"status": IRReviewState.REVIEWED.value, "reviewer_id": "reviewer:fixture"},
        "trust": {"state": IRTrustState.TRUSTED.value},
        "authority": declared.value,
        "freshness": "fresh",
        "declarations": list(declarations),
        "formal_views": list(formal_views),
        "claims": list(claims),
        "assumptions": list(assumptions),
        "obligations": list(obligations),
        "result_authority": (
            dict(result_authority)
            if isinstance(result_authority, Mapping)
            else (
                result_authority
                if isinstance(result_authority, str)
                else list(result_authority)
            )
        ),
        "ambiguities": [],
        "contradictions": [],
        "partial": False,
        "truncated": False,
    }
    if root_reference is not None:
        payload["root_membership"] = {
            "root_cid_v1": root_reference.cid_v1,
            "root_supervisor_digest": root_reference.supervisor_digest,
        }
    if updates:
        payload.update(dict(updates))
    encoded = canonical_artifact_bytes(payload)
    reference = PinnedArtifactRef.from_canonical_bytes(
        encoded,
        artifact_id=artifact_id or f"fixture:{normalized_family.value}",
        artifact_kind=normalized_family.value,
        artifact_schema=selected_schema,
        artifact_schema_version=schema_version,
        producer_id=producer_id,
        authority=reference_authority,
    )
    return reference, encoded


def create_default_ir_registry(
    *, include_optional_ipfs_datasets: bool = False
) -> IRRegistry:
    """Create the local registry, optionally declaring (not importing) datasets."""

    registry = IRRegistry()
    if include_optional_ipfs_datasets:
        registry.register_optional_module(
            IRCapability(
                provider_id="ipfs-datasets-ir",
                provider_version="optional",
                capability_revision="declared@1",
                remote=True,
            )
        )
    return registry


def verify_ir_artifact(
    request: IRLoadRequest, canonical_bytes: bytes
) -> IRLoadResult:
    """Verify exact caller-supplied bytes without retaining them in shared state."""

    if not isinstance(request, IRLoadRequest):
        raise IRRegistryError("request must be IRLoadRequest")
    registry = IRRegistry(bounds=request.bounds)
    if request.reference.verify_canonical_bytes(canonical_bytes):
        registry.register_local_artifact(request.reference, canonical_bytes)
    else:
        # Verification still returns a typed quarantined result instead of
        # failing during fixture registration.
        return registry._verify(
            request, canonical_bytes, provider_id="caller-supplied-ir"
        )
    return registry.load(replace(request, provider_id="supervisor-local-ir"))


__all__ = [
    "DEFAULT_MAX_IR_ARTIFACT_BYTES",
    "IR_ARTIFACT_ENVELOPE_SCHEMA",
    "IR_CAPABILITY_SCHEMA",
    "IR_LOAD_REQUEST_SCHEMA",
    "IR_LOAD_RESULT_SCHEMA",
    "IR_REGISTRY_VERSION",
    "IRCapability",
    "IRDeclaredAuthority",
    "IRFailure",
    "IRFailureCode",
    "IRFamily",
    "IRLoadRequest",
    "IRLoadResult",
    "IRLoadStatus",
    "IROperation",
    "IRRegistry",
    "IRRegistryBounds",
    "IRRegistryError",
    "IRReviewState",
    "IRSchemaSupport",
    "IRStatus",
    "IRTrustState",
    "SUPPORTED_IR_SCHEMAS",
    "VerifiedIRArtifact",
    "create_default_ir_registry",
    "deterministic_ir_fixture",
    "normalize_ir_family",
    "verify_ir_artifact",
]
