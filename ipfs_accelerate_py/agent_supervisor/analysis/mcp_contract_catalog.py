"""Reviewed MCP++ contract catalog (SCA-040).

Interface: ``McpContractCatalog@1``

Defines versioned contract source records, explicit authority classes, review
state, closed claim families, contradiction retention, and complete source /
schema-version invalidators for MCP++ declarations and implementations.

Normative rules:

* Canonical IDs are content-addressed; supplied IDs are re-derived, never trusted.
* Source kinds retain their authority class; documentation and inferred prose
  may only nominate contracts and cannot silently become reviewed.
* Conflicting sources produce explicit contradiction records and are never
  silently resolved to a single winner.
* Unknown claim families and unreviewed prose fail closed.
* Every contract and source carries complete source-version and schema-version
  invalidators.

Conflict policy: adapt :mod:`..proof.code_property_catalog` and interface
contract patterns; retain one assurance lattice (no parallel authority model).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Final

from ..proof.formal_verification_contracts import content_identity


MCP_CONTRACT_CATALOG_INTERFACE: Final = "McpContractCatalog@1"
MCP_CONTRACT_CATALOG_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/mcp-contract-catalog@1"
)
MCP_CONTRACT_SOURCE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/mcp-contract-source@1"
)
MCP_CONTRACT_RECORD_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/mcp-contract-record@1"
)
MCP_CONTRACT_CONTRADICTION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/mcp-contract-contradiction@1"
)
MCP_CLAIM_FAMILY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/mcp-claim-family@1"
)
MCP_INVALIDATOR_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/mcp-contract-invalidator@1"
)
CATALOG_VERSION: Final = "1"
CONTRACT_SCHEMA_VERSION: Final = "1"


class McpContractCatalogError(ValueError):
    """Catalog input is malformed or violates the closed-registration policy."""


class UnknownMcpContractError(LookupError):
    """Raised when a contract id is not present in the catalog."""


class UnknownMcpClaimFamilyError(LookupError):
    """Raised when a claim family id is not in the closed reviewed set."""


class UnreviewedContractError(McpContractCatalogError):
    """Raised when unreviewed or prose material is treated as authoritative."""


class ContractSourceKind(str, Enum):
    """Closed set of MCP++ contract evidence source kinds.

    Authority order (plan § expected behavior):

    1. Versioned MCP-IDL, JSON Schema, typed public interface, reviewed policy
    2. Canonical conformance vector / executable contract test
    3. Package tool registration and schema publication
    4. Reviewed manifest or release gate
    5. Documentation or inferred behavior (nominating only until reviewed)
    """

    MCP_IDL = "mcp_idl"
    JSON_SCHEMA = "json_schema"
    TYPED_INTERFACE = "typed_interface"
    POLICY_CONTRACT = "policy_contract"
    CONFORMANCE_TEST = "conformance_test"
    REGISTRATION = "registration"
    MANIFEST = "manifest"
    DOCUMENTATION = "documentation"
    INFERRED_PROSE = "inferred_prose"


class SourceAuthorityClass(str, Enum):
    """Authority class retained on every source record.

    Lower :meth:`rank` is higher precedence.  Precedence is used only to
    *describe* effective authority; conflicting sources are never collapsed.
    """

    AUTHORITATIVE = "authoritative"
    CONFORMANCE = "conformance"
    REGISTRATION = "registration"
    MANIFEST = "manifest"
    NOMINATING = "nominating"
    NONE = "none"

    @property
    def rank(self) -> int:
        return {
            SourceAuthorityClass.AUTHORITATIVE: 1,
            SourceAuthorityClass.CONFORMANCE: 2,
            SourceAuthorityClass.REGISTRATION: 3,
            SourceAuthorityClass.MANIFEST: 4,
            SourceAuthorityClass.NOMINATING: 5,
            SourceAuthorityClass.NONE: 6,
        }[self]

    @property
    def may_authorize_reviewed_contract(self) -> bool:
        """Whether this class can bind a reviewed contract alone."""

        return self in {
            SourceAuthorityClass.AUTHORITATIVE,
            SourceAuthorityClass.CONFORMANCE,
            SourceAuthorityClass.REGISTRATION,
            SourceAuthorityClass.MANIFEST,
        }


class ReviewState(str, Enum):
    """Explicit review state for sources and contracts."""

    UNREVIEWED = "unreviewed"
    NOMINATED = "nominated"
    UNDER_REVIEW = "under_review"
    REVIEWED = "reviewed"
    REJECTED = "rejected"
    CONTRADICTED = "contradicted"

    @property
    def is_reviewed(self) -> bool:
        return self is ReviewState.REVIEWED

    @property
    def is_fail_closed(self) -> bool:
        """States that must not mint reviewed contract authority."""

        return self in {
            ReviewState.UNREVIEWED,
            ReviewState.NOMINATED,
            ReviewState.UNDER_REVIEW,
            ReviewState.REJECTED,
            ReviewState.CONTRADICTED,
        }


class McpClaimFamily(str, Enum):
    """Closed reviewed MCP++ claim families (plan § Contract claim families)."""

    DECLARED_TOOL_EXISTS = "DeclaredToolExists"
    DESCRIPTOR_SCHEMA_MATCHES = "DescriptorSchemaMatches"
    INVOCATION_REACHABLE = "InvocationReachable"
    ARGUMENTS_PRESERVED = "ArgumentsPreserved"
    RESULT_ENVELOPE_PRESERVED = "ResultEnvelopePreserved"
    POLICY_BEFORE_EFFECT = "PolicyBeforeEffect"
    NO_COMPATIBILITY_BYPASS = "NoCompatibilityBypass"
    TRANSPORT_PARITY = "TransportParity"
    DISCOVERY_EXECUTION_PARITY = "DiscoveryExecutionParity"
    FAILURE_PARITY = "FailureParity"
    SNAPSHOT_FRESHNESS = "SnapshotFreshness"
    NO_DYNAMIC_AUTHORITY = "NoDynamicAuthority"


class ContractInvalidationKind(str, Enum):
    """Machine-readable reasons a catalog entry becomes stale."""

    SOURCE_VERSION = "source_version"
    SCHEMA_VERSION = "schema_version"
    SOURCE_CONTENT = "source_content"
    CATALOG_VERSION = "catalog_version"
    CLAIM_FAMILY = "claim_family"
    REVIEW_STATE = "review_state"
    AUTHORITY_CLASS = "authority_class"
    REPOSITORY_TREE = "repository_tree"
    POLICY = "policy"
    TOOLCHAIN = "toolchain"
    SUBJECT = "subject"


# Default authority class for each closed source kind (plan precedence).
_SOURCE_KIND_AUTHORITY: Mapping[ContractSourceKind, SourceAuthorityClass] = (
    MappingProxyType(
        {
            ContractSourceKind.MCP_IDL: SourceAuthorityClass.AUTHORITATIVE,
            ContractSourceKind.JSON_SCHEMA: SourceAuthorityClass.AUTHORITATIVE,
            ContractSourceKind.TYPED_INTERFACE: SourceAuthorityClass.AUTHORITATIVE,
            ContractSourceKind.POLICY_CONTRACT: SourceAuthorityClass.AUTHORITATIVE,
            ContractSourceKind.CONFORMANCE_TEST: SourceAuthorityClass.CONFORMANCE,
            ContractSourceKind.REGISTRATION: SourceAuthorityClass.REGISTRATION,
            ContractSourceKind.MANIFEST: SourceAuthorityClass.MANIFEST,
            ContractSourceKind.DOCUMENTATION: SourceAuthorityClass.NOMINATING,
            ContractSourceKind.INFERRED_PROSE: SourceAuthorityClass.NONE,
        }
    )
)

# Human-readable descriptions for seeded claim families.
_CLAIM_FAMILY_TITLES: Mapping[McpClaimFamily, str] = MappingProxyType(
    {
        McpClaimFamily.DECLARED_TOOL_EXISTS: (
            "SwissKnife-declared tool exists in the package MCP registry"
        ),
        McpClaimFamily.DESCRIPTOR_SCHEMA_MATCHES: (
            "MCP++ descriptor I/O compatible with package-published schema"
        ),
        McpClaimFamily.INVOCATION_REACHABLE: (
            "Declared call path reaches a concrete handler via allowlisted dispatch"
        ),
        McpClaimFamily.ARGUMENTS_PRESERVED: (
            "Required arguments, defaults, names, and types survive translation"
        ),
        McpClaimFamily.RESULT_ENVELOPE_PRESERVED: (
            "Success, error, streaming, and provenance envelope semantics preserved"
        ),
        McpClaimFamily.POLICY_BEFORE_EFFECT: (
            "Auth, UCAN/deontic policy, lease/fence checks dominate mutation effects"
        ),
        McpClaimFamily.NO_COMPATIBILITY_BYPASS: (
            "Compatibility endpoints cannot bypass MCP++ policy or schema gates"
        ),
        McpClaimFamily.TRANSPORT_PARITY: (
            "HTTP, stdio, WebSocket, and libp2p expose only reviewed differences"
        ),
        McpClaimFamily.DISCOVERY_EXECUTION_PARITY: (
            "tools/list discovery agrees with tools/call reachability"
        ),
        McpClaimFamily.FAILURE_PARITY: (
            "Unsupported, denied, timed-out, and partial states stay distinguishable"
        ),
        McpClaimFamily.SNAPSHOT_FRESHNESS: (
            "Claims bind exact repository, schema, policy, and capability roots"
        ),
        McpClaimFamily.NO_DYNAMIC_AUTHORITY: (
            "Unresolved dynamic dispatch cannot be reported as proved"
        ),
    }
)

_PROSE_KINDS: Final[frozenset[ContractSourceKind]] = frozenset(
    {
        ContractSourceKind.DOCUMENTATION,
        ContractSourceKind.INFERRED_PROSE,
    }
)

_NL_MARKERS: Final[tuple[str, ...]] = (
    "natural_language",
    "nl_claim",
    "freeform",
    "free_text",
    "prose_claim",
    "arbitrary_claim",
    "inferred_prose",
    "unreviewed_prose",
)


def _norm_text(value: Any, *, field_name: str, required: bool = False) -> str:
    if value is None:
        text = ""
    elif isinstance(value, str):
        text = value.strip()
    else:
        raise McpContractCatalogError(f"{field_name} must be a string")
    if required and not text:
        raise McpContractCatalogError(f"{field_name} is required")
    return text


def _norm_enum(value: Any, enum_cls: type[Enum], *, field_name: str) -> Enum:
    if isinstance(value, enum_cls):
        return value
    if isinstance(value, str) and value.strip():
        try:
            return enum_cls(value.strip())
        except ValueError as exc:
            raise McpContractCatalogError(
                f"unknown {field_name}: {value!r}"
            ) from exc
    raise McpContractCatalogError(f"{field_name} must be a valid {enum_cls.__name__}")


def _sorted_unique_strings(
    values: Iterable[Any], *, field_name: str, required: bool = False
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        values = (values,)
    result = tuple(sorted({str(v).strip() for v in values if str(v).strip()}))
    if required and not result:
        raise McpContractCatalogError(f"{field_name} must not be empty")
    return result


def _mapping_proxy(value: Any, *, field_name: str) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise McpContractCatalogError(f"{field_name} must be a mapping")
    return MappingProxyType(dict(value))


def authority_for_source_kind(
    kind: ContractSourceKind | str,
) -> SourceAuthorityClass:
    """Return the fixed default authority class for a closed source kind."""

    kind_e = _norm_enum(kind, ContractSourceKind, field_name="kind")
    return _SOURCE_KIND_AUTHORITY[kind_e]  # type: ignore[index]


def is_prose_or_unreviewed_source(
    *,
    kind: ContractSourceKind | str,
    review_state: ReviewState | str,
    authority: SourceAuthorityClass | str | None = None,
) -> bool:
    """Whether a source is prose/nominating and must fail closed for review."""

    kind_e = _norm_enum(kind, ContractSourceKind, field_name="kind")
    state_e = _norm_enum(review_state, ReviewState, field_name="review_state")
    auth_e = (
        _norm_enum(authority, SourceAuthorityClass, field_name="authority")
        if authority is not None
        else authority_for_source_kind(kind_e)
    )
    if kind_e in _PROSE_KINDS:
        return True
    if auth_e in {SourceAuthorityClass.NOMINATING, SourceAuthorityClass.NONE}:
        return True
    if state_e.is_fail_closed and not (
        state_e is ReviewState.REVIEWED and auth_e.may_authorize_reviewed_contract
    ):
        return state_e is not ReviewState.REVIEWED
    return False


def reject_natural_language_claim(payload: Mapping[str, Any] | str | None) -> None:
    """Fail closed when freeform / natural-language claim markers are present."""

    if payload is None:
        return
    if isinstance(payload, str):
        lowered = payload.strip().lower()
        for marker in _NL_MARKERS:
            if marker in lowered.replace("-", "_"):
                raise UnreviewedContractError(
                    f"natural-language / unreviewed prose fails closed: {marker}"
                )
        return
    if not isinstance(payload, Mapping):
        raise McpContractCatalogError("claim payload must be a mapping or string")
    blob_parts: list[str] = []
    for key, value in payload.items():
        blob_parts.append(str(key).lower())
        if isinstance(value, str):
            blob_parts.append(value.lower())
        elif value is True or value == 1:
            blob_parts.append(str(key).lower())
    blob = " ".join(blob_parts)
    for marker in _NL_MARKERS:
        if marker in blob.replace("-", "_"):
            raise UnreviewedContractError(
                f"natural-language / unreviewed prose fails closed: {marker}"
            )
    # Explicit freeform statement without a reviewed family fails closed.
    if payload.get("freeform_statement") or payload.get("prose"):
        raise UnreviewedContractError(
            "freeform statement without a reviewed claim family fails closed"
        )


@dataclass(frozen=True)
class ContractInvalidator:
    """One machine-readable binding that can invalidate a catalog entry."""

    kind: ContractInvalidationKind
    value: str
    reason_code: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "kind",
            _norm_enum(self.kind, ContractInvalidationKind, field_name="kind"),
        )
        object.__setattr__(
            self, "value", _norm_text(self.value, field_name="value", required=True)
        )
        object.__setattr__(
            self,
            "reason_code",
            _norm_text(self.reason_code, field_name="reason_code"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": MCP_INVALIDATOR_SCHEMA,
            "kind": self.kind.value,
            "value": self.value,
            "reason_code": self.reason_code,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ContractInvalidator":
        if not isinstance(payload, Mapping):
            raise McpContractCatalogError("invalidator must be an object")
        schema = payload.get("schema")
        if schema not in (None, "", MCP_INVALIDATOR_SCHEMA):
            raise McpContractCatalogError("unsupported invalidator schema")
        return cls(
            kind=payload.get("kind", ContractInvalidationKind.SOURCE_VERSION),
            value=str(payload.get("value") or ""),
            reason_code=str(payload.get("reason_code") or ""),
        )

    def matches(
        self, *, kind: ContractInvalidationKind | str, value: str
    ) -> bool:
        kind_e = _norm_enum(kind, ContractInvalidationKind, field_name="kind")
        return self.kind is kind_e and self.value == str(value).strip()


def build_source_invalidators(
    *,
    source_version: str,
    schema_version: str,
    source_content_id: str = "",
    catalog_version: str = CATALOG_VERSION,
    repository_tree_id: str = "",
    policy_id: str = "",
    toolchain_id: str = "",
    subject: str = "",
    claim_family: str = "",
    review_state: str = "",
    authority_class: str = "",
) -> tuple[ContractInvalidator, ...]:
    """Build the complete closed invalidator set for a source or contract.

    ``source_version`` and ``schema_version`` invalidators are **always**
    present (acceptance: source and schema version invalidators are complete).
    """

    source_version = _norm_text(
        source_version, field_name="source_version", required=True
    )
    schema_version = _norm_text(
        schema_version, field_name="schema_version", required=True
    )
    selectors: list[ContractInvalidator] = [
        ContractInvalidator(
            kind=ContractInvalidationKind.SOURCE_VERSION,
            value=source_version,
            reason_code="source_version_drift",
        ),
        ContractInvalidator(
            kind=ContractInvalidationKind.SCHEMA_VERSION,
            value=schema_version,
            reason_code="schema_version_drift",
        ),
    ]
    if source_content_id:
        selectors.append(
            ContractInvalidator(
                kind=ContractInvalidationKind.SOURCE_CONTENT,
                value=source_content_id,
                reason_code="source_content_changed",
            )
        )
    if catalog_version:
        selectors.append(
            ContractInvalidator(
                kind=ContractInvalidationKind.CATALOG_VERSION,
                value=_norm_text(catalog_version, field_name="catalog_version"),
                reason_code="catalog_version_drift",
            )
        )
    if repository_tree_id:
        selectors.append(
            ContractInvalidator(
                kind=ContractInvalidationKind.REPOSITORY_TREE,
                value=repository_tree_id,
                reason_code="stale_repository_tree",
            )
        )
    if policy_id:
        selectors.append(
            ContractInvalidator(
                kind=ContractInvalidationKind.POLICY,
                value=policy_id,
                reason_code="policy_drift",
            )
        )
    if toolchain_id:
        selectors.append(
            ContractInvalidator(
                kind=ContractInvalidationKind.TOOLCHAIN,
                value=toolchain_id,
                reason_code="toolchain_drift",
            )
        )
    if subject:
        selectors.append(
            ContractInvalidator(
                kind=ContractInvalidationKind.SUBJECT,
                value=subject,
                reason_code="subject_changed",
            )
        )
    if claim_family:
        selectors.append(
            ContractInvalidator(
                kind=ContractInvalidationKind.CLAIM_FAMILY,
                value=claim_family,
                reason_code="claim_family_changed",
            )
        )
    if review_state:
        selectors.append(
            ContractInvalidator(
                kind=ContractInvalidationKind.REVIEW_STATE,
                value=review_state,
                reason_code="review_state_changed",
            )
        )
    if authority_class:
        selectors.append(
            ContractInvalidator(
                kind=ContractInvalidationKind.AUTHORITY_CLASS,
                value=authority_class,
                reason_code="authority_class_changed",
            )
        )
    # Stable order by kind then value.
    return tuple(sorted(selectors, key=lambda item: (item.kind.value, item.value)))


def evaluate_invalidation(
    invalidators: Sequence[ContractInvalidator],
    *,
    current: Mapping[str, str],
) -> tuple[ContractInvalidator, ...]:
    """Return invalidators whose bound value no longer matches ``current``.

    ``current`` keys are :class:`ContractInvalidationKind` values.
    """

    if not isinstance(current, Mapping):
        raise McpContractCatalogError("current bindings must be a mapping")
    matched: list[ContractInvalidator] = []
    for inv in invalidators:
        if not isinstance(inv, ContractInvalidator):
            raise McpContractCatalogError(
                "invalidators must be ContractInvalidator instances"
            )
        key = inv.kind.value
        if key not in current:
            continue
        if str(current[key]).strip() != inv.value:
            matched.append(inv)
    return tuple(matched)


def require_complete_version_invalidators(
    invalidators: Sequence[ContractInvalidator],
) -> None:
    """Fail closed when source_version or schema_version invalidators are missing."""

    kinds = {
        inv.kind
        for inv in invalidators
        if isinstance(inv, ContractInvalidator)
    }
    missing: list[str] = []
    if ContractInvalidationKind.SOURCE_VERSION not in kinds:
        missing.append(ContractInvalidationKind.SOURCE_VERSION.value)
    if ContractInvalidationKind.SCHEMA_VERSION not in kinds:
        missing.append(ContractInvalidationKind.SCHEMA_VERSION.value)
    if missing:
        raise McpContractCatalogError(
            "incomplete version invalidators; missing: " + ", ".join(missing)
        )


@dataclass(frozen=True)
class ContractSourceRecord:
    """One versioned contract source with explicit authority and review state."""

    kind: ContractSourceKind
    authority_class: SourceAuthorityClass
    review_state: ReviewState
    source_version: str
    schema_version: str
    subject: str
    path: str = ""
    content_digest: str = ""
    payload_fingerprint: str = ""
    metadata: Mapping[str, Any] = MappingProxyType({})
    invalidators: tuple[ContractInvalidator, ...] = ()
    source_id: str = ""

    def __post_init__(self) -> None:
        kind_e = _norm_enum(self.kind, ContractSourceKind, field_name="kind")
        object.__setattr__(self, "kind", kind_e)
        default_auth = authority_for_source_kind(kind_e)
        auth_e = _norm_enum(
            self.authority_class, SourceAuthorityClass, field_name="authority_class"
        )
        # Authority class must not exceed the default for the kind (fail closed
        # against silent promotion of documentation/prose).
        if auth_e.rank < default_auth.rank:
            raise McpContractCatalogError(
                f"authority_class {auth_e.value!r} exceeds allowed class "
                f"{default_auth.value!r} for source kind {kind_e.value!r}"
            )
        object.__setattr__(self, "authority_class", auth_e)
        state_e = _norm_enum(
            self.review_state, ReviewState, field_name="review_state"
        )
        object.__setattr__(self, "review_state", state_e)
        object.__setattr__(
            self,
            "source_version",
            _norm_text(self.source_version, field_name="source_version", required=True),
        )
        object.__setattr__(
            self,
            "schema_version",
            _norm_text(self.schema_version, field_name="schema_version", required=True),
        )
        object.__setattr__(
            self, "subject", _norm_text(self.subject, field_name="subject", required=True)
        )
        object.__setattr__(self, "path", _norm_text(self.path, field_name="path"))
        object.__setattr__(
            self,
            "content_digest",
            _norm_text(self.content_digest, field_name="content_digest"),
        )
        object.__setattr__(
            self,
            "payload_fingerprint",
            _norm_text(self.payload_fingerprint, field_name="payload_fingerprint"),
        )
        object.__setattr__(
            self, "metadata", _mapping_proxy(self.metadata, field_name="metadata")
        )
        if not isinstance(self.invalidators, tuple):
            object.__setattr__(self, "invalidators", tuple(self.invalidators))
        for inv in self.invalidators:
            if not isinstance(inv, ContractInvalidator):
                raise McpContractCatalogError(
                    "invalidators must be ContractInvalidator instances"
                )
        if not self.invalidators:
            object.__setattr__(
                self,
                "invalidators",
                build_source_invalidators(
                    source_version=self.source_version,
                    schema_version=self.schema_version,
                    source_content_id=self.content_digest or self.payload_fingerprint,
                    subject=self.subject,
                    review_state=state_e.value,
                    authority_class=auth_e.value,
                ),
            )
        require_complete_version_invalidators(self.invalidators)
        object.__setattr__(
            self,
            "invalidators",
            tuple(
                sorted(self.invalidators, key=lambda item: (item.kind.value, item.value))
            ),
        )
        derived = self._derive_source_id()
        claimed = _norm_text(self.source_id, field_name="source_id")
        if claimed and claimed != derived:
            raise McpContractCatalogError("source_id does not match content")
        object.__setattr__(self, "source_id", derived)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": MCP_CONTRACT_SOURCE_SCHEMA,
            "kind": self.kind.value,
            "authority_class": self.authority_class.value,
            "review_state": self.review_state.value,
            "source_version": self.source_version,
            "schema_version": self.schema_version,
            "subject": self.subject,
            "path": self.path,
            "content_digest": self.content_digest,
            "payload_fingerprint": self.payload_fingerprint,
            "metadata": dict(self.metadata),
            "invalidators": [inv.to_dict() for inv in self.invalidators],
        }

    def _derive_source_id(self) -> str:
        return content_identity(self._identity_payload())

    @property
    def may_authorize_contract(self) -> bool:
        return (
            self.review_state is ReviewState.REVIEWED
            and self.authority_class.may_authorize_reviewed_contract
            and self.kind not in _PROSE_KINDS
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._identity_payload(),
            "source_id": self.source_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ContractSourceRecord":
        if not isinstance(payload, Mapping):
            raise McpContractCatalogError("source record must be an object")
        schema = payload.get("schema")
        if schema not in (None, "", MCP_CONTRACT_SOURCE_SCHEMA):
            raise McpContractCatalogError("unsupported contract-source schema")
        raw_inv = payload.get("invalidators") or ()
        invalidators = tuple(ContractInvalidator.from_dict(item) for item in raw_inv)
        return cls(
            kind=payload.get("kind", ContractSourceKind.DOCUMENTATION),
            authority_class=payload.get(
                "authority_class", SourceAuthorityClass.NOMINATING
            ),
            review_state=payload.get("review_state", ReviewState.UNREVIEWED),
            source_version=str(payload.get("source_version") or ""),
            schema_version=str(payload.get("schema_version") or ""),
            subject=str(payload.get("subject") or ""),
            path=str(payload.get("path") or ""),
            content_digest=str(payload.get("content_digest") or ""),
            payload_fingerprint=str(payload.get("payload_fingerprint") or ""),
            metadata=dict(payload.get("metadata") or {}),
            invalidators=invalidators,
            source_id=str(payload.get("source_id") or ""),
        )


@dataclass(frozen=True)
class ClaimFamilyDescriptor:
    """One closed, reviewed MCP claim family entry."""

    family: McpClaimFamily
    title: str = ""
    review_state: ReviewState = ReviewState.REVIEWED
    catalog_version: str = CATALOG_VERSION
    family_id: str = ""

    def __post_init__(self) -> None:
        family_e = _norm_enum(self.family, McpClaimFamily, field_name="family")
        object.__setattr__(self, "family", family_e)
        title = _norm_text(self.title, field_name="title")
        if not title:
            title = _CLAIM_FAMILY_TITLES[family_e]
        object.__setattr__(self, "title", title)
        object.__setattr__(
            self,
            "review_state",
            _norm_enum(self.review_state, ReviewState, field_name="review_state"),
        )
        if self.review_state is not ReviewState.REVIEWED:
            raise McpContractCatalogError(
                "seed claim families must remain review_state=reviewed"
            )
        object.__setattr__(
            self,
            "catalog_version",
            _norm_text(
                self.catalog_version, field_name="catalog_version", required=True
            ),
        )
        derived = content_identity(
            {
                "schema": MCP_CLAIM_FAMILY_SCHEMA,
                "family": family_e.value,
                "title": title,
                "review_state": self.review_state.value,
                "catalog_version": self.catalog_version,
            }
        )
        claimed = _norm_text(self.family_id, field_name="family_id")
        if claimed and claimed != derived:
            raise McpContractCatalogError("family_id does not match content")
        object.__setattr__(self, "family_id", derived)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": MCP_CLAIM_FAMILY_SCHEMA,
            "family": self.family.value,
            "family_id": self.family_id,
            "title": self.title,
            "review_state": self.review_state.value,
            "catalog_version": self.catalog_version,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ClaimFamilyDescriptor":
        if not isinstance(payload, Mapping):
            raise McpContractCatalogError("claim family must be an object")
        schema = payload.get("schema")
        if schema not in (None, "", MCP_CLAIM_FAMILY_SCHEMA):
            raise McpContractCatalogError("unsupported claim-family schema")
        return cls(
            family=payload.get("family", McpClaimFamily.NO_DYNAMIC_AUTHORITY),
            title=str(payload.get("title") or ""),
            review_state=payload.get("review_state", ReviewState.REVIEWED),
            catalog_version=str(payload.get("catalog_version") or CATALOG_VERSION),
            family_id=str(payload.get("family_id") or ""),
        )


@dataclass(frozen=True)
class ContradictionRecord:
    """Explicit, unresolved contradiction between sources.

    Contradictory sources **remain** contradictory: this record never picks a
    winner and ``resolved`` is always ``False`` at construction.
    """

    subject: str
    field_name: str
    source_ids: tuple[str, ...]
    values: tuple[str, ...]
    reason_code: str = "conflicting_sources"
    resolved: bool = False
    contradiction_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "subject", _norm_text(self.subject, field_name="subject", required=True)
        )
        object.__setattr__(
            self,
            "field_name",
            _norm_text(self.field_name, field_name="field_name", required=True),
        )
        object.__setattr__(
            self,
            "source_ids",
            _sorted_unique_strings(
                self.source_ids, field_name="source_ids", required=True
            ),
        )
        if isinstance(self.values, (str, bytes, bytearray)):
            values = (str(self.values),)
        else:
            values = tuple(str(v) for v in self.values)
        if len(values) < 2:
            raise McpContractCatalogError(
                "contradiction requires at least two distinct values"
            )
        # Preserve order of first appearance but require distinctness.
        unique = tuple(dict.fromkeys(values))
        if len(unique) < 2:
            raise McpContractCatalogError(
                "contradiction values must not collapse to one value"
            )
        object.__setattr__(self, "values", unique)
        object.__setattr__(
            self,
            "reason_code",
            _norm_text(self.reason_code, field_name="reason_code") or "conflicting_sources",
        )
        # Fail closed: contradictions cannot be marked resolved in-catalog.
        if self.resolved:
            raise McpContractCatalogError(
                "contradictory sources must remain unresolved in the catalog"
            )
        object.__setattr__(self, "resolved", False)
        derived = content_identity(
            {
                "schema": MCP_CONTRACT_CONTRADICTION_SCHEMA,
                "subject": self.subject,
                "field_name": self.field_name,
                "source_ids": list(self.source_ids),
                "values": list(self.values),
                "reason_code": self.reason_code,
                "resolved": False,
            }
        )
        claimed = _norm_text(self.contradiction_id, field_name="contradiction_id")
        if claimed and claimed != derived:
            raise McpContractCatalogError("contradiction_id does not match content")
        object.__setattr__(self, "contradiction_id", derived)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": MCP_CONTRACT_CONTRADICTION_SCHEMA,
            "contradiction_id": self.contradiction_id,
            "subject": self.subject,
            "field_name": self.field_name,
            "source_ids": list(self.source_ids),
            "values": list(self.values),
            "reason_code": self.reason_code,
            "resolved": False,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ContradictionRecord":
        if not isinstance(payload, Mapping):
            raise McpContractCatalogError("contradiction must be an object")
        schema = payload.get("schema")
        if schema not in (None, "", MCP_CONTRACT_CONTRADICTION_SCHEMA):
            raise McpContractCatalogError("unsupported contradiction schema")
        if payload.get("resolved") is True:
            raise McpContractCatalogError(
                "contradictory sources must remain unresolved in the catalog"
            )
        return cls(
            subject=str(payload.get("subject") or ""),
            field_name=str(payload.get("field_name") or ""),
            source_ids=tuple(payload.get("source_ids") or ()),
            values=tuple(payload.get("values") or ()),
            reason_code=str(payload.get("reason_code") or "conflicting_sources"),
            resolved=False,
            contradiction_id=str(payload.get("contradiction_id") or ""),
        )


@dataclass(frozen=True)
class ContractRecord:
    """One versioned MCP++ contract entry bound to reviewed sources."""

    claim_family: McpClaimFamily
    subject: str
    source_ids: tuple[str, ...]
    authority_class: SourceAuthorityClass
    review_state: ReviewState
    source_version: str
    schema_version: str
    tool_name: str = ""
    package_id: str = ""
    contradiction_ids: tuple[str, ...] = ()
    invalidators: tuple[ContractInvalidator, ...] = ()
    metadata: Mapping[str, Any] = MappingProxyType({})
    contract_id: str = ""

    def __post_init__(self) -> None:
        family_e = _norm_enum(
            self.claim_family, McpClaimFamily, field_name="claim_family"
        )
        object.__setattr__(self, "claim_family", family_e)
        object.__setattr__(
            self, "subject", _norm_text(self.subject, field_name="subject", required=True)
        )
        object.__setattr__(
            self,
            "source_ids",
            _sorted_unique_strings(
                self.source_ids, field_name="source_ids", required=True
            ),
        )
        auth_e = _norm_enum(
            self.authority_class, SourceAuthorityClass, field_name="authority_class"
        )
        object.__setattr__(self, "authority_class", auth_e)
        state_e = _norm_enum(
            self.review_state, ReviewState, field_name="review_state"
        )
        object.__setattr__(self, "review_state", state_e)
        object.__setattr__(
            self,
            "source_version",
            _norm_text(self.source_version, field_name="source_version", required=True),
        )
        object.__setattr__(
            self,
            "schema_version",
            _norm_text(self.schema_version, field_name="schema_version", required=True),
        )
        object.__setattr__(
            self, "tool_name", _norm_text(self.tool_name, field_name="tool_name")
        )
        object.__setattr__(
            self, "package_id", _norm_text(self.package_id, field_name="package_id")
        )
        object.__setattr__(
            self,
            "contradiction_ids",
            _sorted_unique_strings(
                self.contradiction_ids, field_name="contradiction_ids"
            ),
        )
        object.__setattr__(
            self, "metadata", _mapping_proxy(self.metadata, field_name="metadata")
        )
        if not isinstance(self.invalidators, tuple):
            object.__setattr__(self, "invalidators", tuple(self.invalidators))
        for inv in self.invalidators:
            if not isinstance(inv, ContractInvalidator):
                raise McpContractCatalogError(
                    "invalidators must be ContractInvalidator instances"
                )
        if not self.invalidators:
            object.__setattr__(
                self,
                "invalidators",
                build_source_invalidators(
                    source_version=self.source_version,
                    schema_version=self.schema_version,
                    subject=self.subject,
                    claim_family=family_e.value,
                    review_state=state_e.value,
                    authority_class=auth_e.value,
                ),
            )
        require_complete_version_invalidators(self.invalidators)
        object.__setattr__(
            self,
            "invalidators",
            tuple(
                sorted(self.invalidators, key=lambda item: (item.kind.value, item.value))
            ),
        )
        # Reviewed contracts require authorizing authority; prose fails closed.
        if state_e is ReviewState.REVIEWED:
            if not auth_e.may_authorize_reviewed_contract:
                raise UnreviewedContractError(
                    "reviewed contracts require an authorizing authority class; "
                    f"got {auth_e.value!r}"
                )
            if self.contradiction_ids:
                raise McpContractCatalogError(
                    "reviewed contracts cannot bind unresolved contradiction_ids; "
                    "mark review_state=contradicted instead"
                )
        if state_e is ReviewState.CONTRADICTED and not self.contradiction_ids:
            raise McpContractCatalogError(
                "contradicted contracts must reference contradiction_ids"
            )
        reject_natural_language_claim(self.metadata)
        derived = self._derive_contract_id()
        claimed = _norm_text(self.contract_id, field_name="contract_id")
        if claimed and claimed != derived:
            raise McpContractCatalogError("contract_id does not match content")
        object.__setattr__(self, "contract_id", derived)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": MCP_CONTRACT_RECORD_SCHEMA,
            "claim_family": self.claim_family.value,
            "subject": self.subject,
            "source_ids": list(self.source_ids),
            "authority_class": self.authority_class.value,
            "review_state": self.review_state.value,
            "source_version": self.source_version,
            "schema_version": self.schema_version,
            "tool_name": self.tool_name,
            "package_id": self.package_id,
            "contradiction_ids": list(self.contradiction_ids),
            "invalidators": [inv.to_dict() for inv in self.invalidators],
            "metadata": dict(self.metadata),
        }

    def _derive_contract_id(self) -> str:
        return content_identity(self._identity_payload())

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._identity_payload(),
            "contract_id": self.contract_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ContractRecord":
        if not isinstance(payload, Mapping):
            raise McpContractCatalogError("contract record must be an object")
        schema = payload.get("schema")
        if schema not in (None, "", MCP_CONTRACT_RECORD_SCHEMA):
            raise McpContractCatalogError("unsupported contract-record schema")
        raw_inv = payload.get("invalidators") or ()
        invalidators = tuple(ContractInvalidator.from_dict(item) for item in raw_inv)
        return cls(
            claim_family=payload.get(
                "claim_family", McpClaimFamily.DECLARED_TOOL_EXISTS
            ),
            subject=str(payload.get("subject") or ""),
            source_ids=tuple(payload.get("source_ids") or ()),
            authority_class=payload.get(
                "authority_class", SourceAuthorityClass.AUTHORITATIVE
            ),
            review_state=payload.get("review_state", ReviewState.UNREVIEWED),
            source_version=str(payload.get("source_version") or ""),
            schema_version=str(payload.get("schema_version") or ""),
            tool_name=str(payload.get("tool_name") or ""),
            package_id=str(payload.get("package_id") or ""),
            contradiction_ids=tuple(payload.get("contradiction_ids") or ()),
            invalidators=invalidators,
            metadata=dict(payload.get("metadata") or {}),
            contract_id=str(payload.get("contract_id") or ""),
        )


# Source kinds whose schema_version binds the same version lattice and may
# contradict each other when they disagree.
_SCHEMA_BEARING_KINDS: Final[frozenset[ContractSourceKind]] = frozenset(
    {
        ContractSourceKind.MCP_IDL,
        ContractSourceKind.JSON_SCHEMA,
        ContractSourceKind.TYPED_INTERFACE,
        ContractSourceKind.POLICY_CONTRACT,
    }
)


def detect_source_contradictions(
    sources: Sequence[ContractSourceRecord],
    *,
    fields: Sequence[str] = ("payload_fingerprint", "schema_version"),
) -> tuple[ContradictionRecord, ...]:
    """Detect explicit contradictions among sources sharing a subject.

    When two or more sources for the same subject disagree on a compared field,
    a :class:`ContradictionRecord` is emitted.  No source is selected as winner.

    Default comparison:

    * ``payload_fingerprint`` / content digest — any kinds with non-empty values
    * ``schema_version`` — only among schema-bearing kinds (IDL/schema/type/policy)
      or among sources of the same kind

    Distinct ``source_version`` strings across *different* kinds are normal
    (package vs IDL release trains) and are not treated as contradictions unless
    ``source_version`` is requested explicitly for same-kind groups.
    """

    by_subject: dict[str, list[ContractSourceRecord]] = {}
    for source in sources:
        if not isinstance(source, ContractSourceRecord):
            raise McpContractCatalogError(
                "sources must be ContractSourceRecord instances"
            )
        by_subject.setdefault(source.subject, []).append(source)

    contradictions: list[ContradictionRecord] = []
    field_names = tuple(str(f).strip() for f in fields if str(f).strip())
    for subject, group in sorted(by_subject.items()):
        if len(group) < 2:
            continue
        for field_name in field_names:
            if field_name == "payload_fingerprint":
                candidates = group
            elif field_name == "schema_version":
                # Compare schema-bearing kinds together; also same-kind groups.
                schema_group = [
                    s for s in group if s.kind in _SCHEMA_BEARING_KINDS
                ]
                if len(schema_group) >= 2:
                    candidates = schema_group
                else:
                    # Fall back to per-kind comparison for non-schema kinds.
                    candidates = group
            elif field_name == "source_version":
                # Only same-kind source_version disagreements count.
                by_kind: dict[ContractSourceKind, list[ContractSourceRecord]] = {}
                for source in group:
                    by_kind.setdefault(source.kind, []).append(source)
                for kind_group in by_kind.values():
                    if len(kind_group) < 2:
                        continue
                    contradictions.extend(
                        _contradictions_for_field(
                            subject=subject,
                            field_name=field_name,
                            group=kind_group,
                        )
                    )
                continue
            else:
                candidates = group
            contradictions.extend(
                _contradictions_for_field(
                    subject=subject,
                    field_name=field_name,
                    group=candidates,
                )
            )
    return tuple(
        sorted(
            contradictions,
            key=lambda item: (item.subject, item.field_name, item.contradiction_id),
        )
    )


def _field_value(source: ContractSourceRecord, field_name: str) -> str:
    if field_name == "payload_fingerprint":
        return source.payload_fingerprint or source.content_digest
    if field_name == "schema_version":
        return source.schema_version
    if field_name == "source_version":
        return source.source_version
    if field_name == "authority_class":
        return source.authority_class.value
    if field_name == "kind":
        return source.kind.value
    return str(source.metadata.get(field_name, "") or "")


def _contradictions_for_field(
    *,
    subject: str,
    field_name: str,
    group: Sequence[ContractSourceRecord],
) -> list[ContradictionRecord]:
    if field_name == "schema_version" and len(group) >= 2:
        kinds = {s.kind for s in group}
        # When mixing schema-bearing with non-schema kinds only, skip unless
        # all candidates are schema-bearing or all share one kind.
        if not kinds.issubset(_SCHEMA_BEARING_KINDS) and len(kinds) > 1:
            # Split into same-kind buckets.
            out: list[ContradictionRecord] = []
            by_kind: dict[ContractSourceKind, list[ContractSourceRecord]] = {}
            for source in group:
                by_kind.setdefault(source.kind, []).append(source)
            for kind_group in by_kind.values():
                out.extend(
                    _contradictions_for_field(
                        subject=subject,
                        field_name=field_name,
                        group=kind_group,
                    )
                )
            return out

    value_map: dict[str, list[str]] = {}
    for source in group:
        value = _field_value(source, field_name)
        if not value:
            continue
        value_map.setdefault(value, []).append(source.source_id)
    if len(value_map) < 2:
        return []
    values = tuple(sorted(value_map.keys()))
    source_ids = tuple(sorted({sid for sids in value_map.values() for sid in sids}))
    return [
        ContradictionRecord(
            subject=subject,
            field_name=field_name,
            source_ids=source_ids,
            values=values,
            reason_code=f"conflicting_{field_name}",
            resolved=False,
        )
    ]



def effective_authority_for_sources(
    sources: Sequence[ContractSourceRecord],
    *,
    contradictions: Sequence[ContradictionRecord] = (),
) -> tuple[SourceAuthorityClass, ReviewState]:
    """Describe effective authority without collapsing contradictions.

    If any contradiction covers the sources' subject, review state is
    ``contradicted`` and authority is the best *candidate* rank only (still
    non-authorizing when contradicted).
    """

    if not sources:
        return SourceAuthorityClass.NONE, ReviewState.UNREVIEWED
    for source in sources:
        if not isinstance(source, ContractSourceRecord):
            raise McpContractCatalogError(
                "sources must be ContractSourceRecord instances"
            )
    subjects = {s.subject for s in sources}
    contradicted_subjects = {c.subject for c in contradictions}
    if subjects & contradicted_subjects:
        best = min(sources, key=lambda s: s.authority_class.rank)
        return best.authority_class, ReviewState.CONTRADICTED

    authorizing = [s for s in sources if s.may_authorize_contract]
    if authorizing:
        best = min(authorizing, key=lambda s: s.authority_class.rank)
        return best.authority_class, ReviewState.REVIEWED

    # Nominating / unreviewed only.
    best = min(sources, key=lambda s: s.authority_class.rank)
    if best.review_state is ReviewState.NOMINATED or best.kind in _PROSE_KINDS:
        return best.authority_class, ReviewState.NOMINATED
    return best.authority_class, best.review_state


def make_source_record(
    *,
    kind: ContractSourceKind | str,
    subject: str,
    source_version: str,
    schema_version: str = CONTRACT_SCHEMA_VERSION,
    review_state: ReviewState | str | None = None,
    authority_class: SourceAuthorityClass | str | None = None,
    path: str = "",
    content_digest: str = "",
    payload_fingerprint: str = "",
    metadata: Mapping[str, Any] | None = None,
) -> ContractSourceRecord:
    """Construct a source record with default authority for ``kind``."""

    kind_e = _norm_enum(kind, ContractSourceKind, field_name="kind")
    default_auth = authority_for_source_kind(kind_e)
    auth = (
        default_auth
        if authority_class is None
        else _norm_enum(
            authority_class, SourceAuthorityClass, field_name="authority_class"
        )
    )
    if review_state is None:
        if kind_e in _PROSE_KINDS:
            state: ReviewState | str = ReviewState.NOMINATED
        elif auth.may_authorize_reviewed_contract:
            state = ReviewState.REVIEWED
        else:
            state = ReviewState.UNREVIEWED
    else:
        state = review_state
    return ContractSourceRecord(
        kind=kind_e,
        authority_class=auth,
        review_state=state,
        source_version=source_version,
        schema_version=schema_version,
        subject=subject,
        path=path,
        content_digest=content_digest,
        payload_fingerprint=payload_fingerprint,
        metadata=dict(metadata or {}),
    )


def nominate_from_prose(
    *,
    subject: str,
    prose: str,
    source_version: str = "prose:unversioned",
    schema_version: str = CONTRACT_SCHEMA_VERSION,
    path: str = "",
    kind: ContractSourceKind | str = ContractSourceKind.INFERRED_PROSE,
) -> ContractSourceRecord:
    """Nominate a subject from documentation/prose without granting review.

    The returned source is always non-authorizing.  Callers that attempt to
    register it as a reviewed contract must fail closed.
    """

    prose_text = _norm_text(prose, field_name="prose", required=True)
    kind_e = _norm_enum(kind, ContractSourceKind, field_name="kind")
    if kind_e not in _PROSE_KINDS:
        raise McpContractCatalogError(
            "nominate_from_prose requires documentation or inferred_prose kind"
        )
    fingerprint = content_identity(
        {"prose": prose_text, "subject": subject, "path": path}
    )
    return make_source_record(
        kind=kind_e,
        subject=subject,
        source_version=source_version,
        schema_version=schema_version,
        review_state=ReviewState.NOMINATED,
        authority_class=authority_for_source_kind(kind_e),
        path=path,
        payload_fingerprint=fingerprint,
        metadata={"nomination": True, "prose_length": len(prose_text)},
    )


def build_contract_from_sources(
    *,
    claim_family: McpClaimFamily | str,
    subject: str,
    sources: Sequence[ContractSourceRecord],
    tool_name: str = "",
    package_id: str = "",
    source_version: str | None = None,
    schema_version: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    require_reviewed: bool = True,
) -> tuple[ContractRecord, tuple[ContradictionRecord, ...]]:
    """Build a contract from sources, retaining contradictions explicitly.

    When ``require_reviewed`` is true (default), pure prose / unreviewed
    sources fail closed and do not produce a reviewed contract.
    """

    if not sources:
        raise McpContractCatalogError("sources must not be empty")
    family_e = _norm_enum(claim_family, McpClaimFamily, field_name="claim_family")
    reject_natural_language_claim(metadata)
    contradictions = detect_source_contradictions(sources)
    authority, review_state = effective_authority_for_sources(
        sources, contradictions=contradictions
    )
    if require_reviewed:
        if review_state is ReviewState.CONTRADICTED:
            # Explicit contradicted contract is allowed; not silently resolved.
            pass
        elif review_state is not ReviewState.REVIEWED:
            raise UnreviewedContractError(
                "unknown/unreviewed prose fails closed: no authorizing reviewed source"
            )
        if not any(s.may_authorize_contract for s in sources):
            if review_state is not ReviewState.CONTRADICTED:
                raise UnreviewedContractError(
                    "inferred natural language cannot silently become a reviewed contract"
                )
    source_ids = tuple(sorted({s.source_id for s in sources}))
    # Prefer versions from highest-authority non-conflicting source; when
    # contradicted, still record versions but keep contradicted state.
    ordered = sorted(sources, key=lambda s: s.authority_class.rank)
    primary = ordered[0]
    sv = source_version if source_version is not None else primary.source_version
    scv = schema_version if schema_version is not None else primary.schema_version
    contradiction_ids = tuple(c.contradiction_id for c in contradictions)
    if contradictions and review_state is not ReviewState.CONTRADICTED:
        review_state = ReviewState.CONTRADICTED
    contract = ContractRecord(
        claim_family=family_e,
        subject=subject,
        source_ids=source_ids,
        authority_class=authority,
        review_state=review_state,
        source_version=sv,
        schema_version=scv,
        tool_name=tool_name,
        package_id=package_id,
        contradiction_ids=contradiction_ids,
        metadata=dict(metadata or {}),
    )
    return contract, contradictions


@dataclass(frozen=True)
class McpContractCatalog:
    """Immutable, content-addressed MCP++ contract catalog."""

    claim_families: tuple[ClaimFamilyDescriptor, ...]
    sources: tuple[ContractSourceRecord, ...] = ()
    contracts: tuple[ContractRecord, ...] = ()
    contradictions: tuple[ContradictionRecord, ...] = ()
    catalog_version: str = CATALOG_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.claim_families, tuple):
            object.__setattr__(self, "claim_families", tuple(self.claim_families))
        if not isinstance(self.sources, tuple):
            object.__setattr__(self, "sources", tuple(self.sources))
        if not isinstance(self.contracts, tuple):
            object.__setattr__(self, "contracts", tuple(self.contracts))
        if not isinstance(self.contradictions, tuple):
            object.__setattr__(self, "contradictions", tuple(self.contradictions))

        families = tuple(
            sorted(self.claim_families, key=lambda item: item.family.value)
        )
        seen_families: set[str] = set()
        for fam in families:
            if not isinstance(fam, ClaimFamilyDescriptor):
                raise McpContractCatalogError(
                    "claim_families must be ClaimFamilyDescriptor instances"
                )
            if fam.family.value in seen_families:
                raise McpContractCatalogError(
                    f"duplicate claim family: {fam.family.value}"
                )
            seen_families.add(fam.family.value)
        object.__setattr__(self, "claim_families", families)

        sources = tuple(sorted(self.sources, key=lambda item: item.source_id))
        seen_sources: set[str] = set()
        for source in sources:
            if not isinstance(source, ContractSourceRecord):
                raise McpContractCatalogError(
                    "sources must be ContractSourceRecord instances"
                )
            if source.source_id in seen_sources:
                raise McpContractCatalogError(
                    f"duplicate source_id: {source.source_id}"
                )
            seen_sources.add(source.source_id)
        object.__setattr__(self, "sources", sources)

        contracts = tuple(sorted(self.contracts, key=lambda item: item.contract_id))
        seen_contracts: set[str] = set()
        for contract in contracts:
            if not isinstance(contract, ContractRecord):
                raise McpContractCatalogError(
                    "contracts must be ContractRecord instances"
                )
            if contract.contract_id in seen_contracts:
                raise McpContractCatalogError(
                    f"duplicate contract_id: {contract.contract_id}"
                )
            seen_contracts.add(contract.contract_id)
            for sid in contract.source_ids:
                if sid not in seen_sources:
                    raise McpContractCatalogError(
                        f"contract {contract.contract_id} references unknown source {sid}"
                    )
            require_complete_version_invalidators(contract.invalidators)
        object.__setattr__(self, "contracts", contracts)

        contradictions = tuple(
            sorted(
                self.contradictions,
                key=lambda item: (item.subject, item.field_name, item.contradiction_id),
            )
        )
        seen_ctr: set[str] = set()
        for ctr in contradictions:
            if not isinstance(ctr, ContradictionRecord):
                raise McpContractCatalogError(
                    "contradictions must be ContradictionRecord instances"
                )
            if ctr.resolved:
                raise McpContractCatalogError(
                    "contradictory sources must remain unresolved in the catalog"
                )
            if ctr.contradiction_id in seen_ctr:
                raise McpContractCatalogError(
                    f"duplicate contradiction_id: {ctr.contradiction_id}"
                )
            seen_ctr.add(ctr.contradiction_id)
        object.__setattr__(self, "contradictions", contradictions)

        object.__setattr__(
            self,
            "catalog_version",
            _norm_text(
                self.catalog_version, field_name="catalog_version", required=True
            ),
        )
        object.__setattr__(
            self,
            "_source_index",
            MappingProxyType({s.source_id: s for s in sources}),
        )
        object.__setattr__(
            self,
            "_contract_index",
            MappingProxyType({c.contract_id: c for c in contracts}),
        )
        object.__setattr__(
            self,
            "_family_index",
            MappingProxyType({f.family.value: f for f in families}),
        )
        object.__setattr__(
            self,
            "_contradiction_index",
            MappingProxyType({c.contradiction_id: c for c in contradictions}),
        )

    @property
    def catalog_id(self) -> str:
        return content_identity(
            {
                "schema": MCP_CONTRACT_CATALOG_SCHEMA,
                "catalog_version": self.catalog_version,
                "claim_families": [f.to_dict() for f in self.claim_families],
                "sources": [s.to_dict() for s in self.sources],
                "contracts": [c.to_dict() for c in self.contracts],
                "contradictions": [c.to_dict() for c in self.contradictions],
            }
        )

    def get_source(self, source_id: str) -> ContractSourceRecord | None:
        return getattr(self, "_source_index").get(str(source_id).strip())

    def get_contract(self, contract_id: str) -> ContractRecord | None:
        return getattr(self, "_contract_index").get(str(contract_id).strip())

    def get_family(
        self, family: McpClaimFamily | str
    ) -> ClaimFamilyDescriptor | None:
        key = family.value if isinstance(family, McpClaimFamily) else str(family).strip()
        return getattr(self, "_family_index").get(key)

    def require_contract(self, contract_id: str) -> ContractRecord:
        contract = self.get_contract(contract_id)
        if contract is None:
            raise UnknownMcpContractError(f"unknown contract id: {contract_id!r}")
        return contract

    def require_family(
        self, family: McpClaimFamily | str
    ) -> ClaimFamilyDescriptor:
        desc = self.get_family(family)
        if desc is None:
            key = family.value if isinstance(family, McpClaimFamily) else family
            raise UnknownMcpClaimFamilyError(f"unknown claim family: {key!r}")
        return desc

    def contract_ids(self) -> tuple[str, ...]:
        return tuple(c.contract_id for c in self.contracts)

    def source_ids(self) -> tuple[str, ...]:
        return tuple(s.source_id for s in self.sources)

    def family_ids(self) -> tuple[str, ...]:
        return tuple(f.family.value for f in self.claim_families)

    def contradictions_for(self, subject: str) -> tuple[ContradictionRecord, ...]:
        subject_n = str(subject).strip()
        return tuple(c for c in self.contradictions if c.subject == subject_n)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": MCP_CONTRACT_CATALOG_SCHEMA,
            "interface": MCP_CONTRACT_CATALOG_INTERFACE,
            "catalog_version": self.catalog_version,
            "catalog_id": self.catalog_id,
            "claim_families": [f.to_dict() for f in self.claim_families],
            "sources": [s.to_dict() for s in self.sources],
            "contracts": [c.to_dict() for c in self.contracts],
            "contradictions": [c.to_dict() for c in self.contradictions],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "McpContractCatalog":
        if not isinstance(payload, Mapping):
            raise McpContractCatalogError("catalog must be an object")
        schema = payload.get("schema")
        if schema not in (None, "", MCP_CONTRACT_CATALOG_SCHEMA):
            raise McpContractCatalogError("unsupported catalog schema")
        catalog = cls(
            claim_families=tuple(
                ClaimFamilyDescriptor.from_dict(item)
                for item in (payload.get("claim_families") or ())
            ),
            sources=tuple(
                ContractSourceRecord.from_dict(item)
                for item in (payload.get("sources") or ())
            ),
            contracts=tuple(
                ContractRecord.from_dict(item)
                for item in (payload.get("contracts") or ())
            ),
            contradictions=tuple(
                ContradictionRecord.from_dict(item)
                for item in (payload.get("contradictions") or ())
            ),
            catalog_version=str(payload.get("catalog_version") or CATALOG_VERSION),
        )
        claimed = payload.get("catalog_id")
        if claimed is not None and str(claimed) != catalog.catalog_id:
            raise McpContractCatalogError("catalog_id does not match content")
        return catalog


def build_seed_claim_families(
    *,
    catalog_version: str = CATALOG_VERSION,
) -> tuple[ClaimFamilyDescriptor, ...]:
    """Build one reviewed descriptor per closed MCP claim family."""

    families = tuple(
        ClaimFamilyDescriptor(
            family=family,
            title=_CLAIM_FAMILY_TITLES[family],
            review_state=ReviewState.REVIEWED,
            catalog_version=catalog_version,
        )
        for family in sorted(McpClaimFamily, key=lambda item: item.value)
    )
    if len(families) != len(McpClaimFamily):
        raise McpContractCatalogError("seed claim families incomplete")
    return families


def build_default_mcp_contract_catalog() -> McpContractCatalog:
    """Return the sealed default catalog of reviewed claim families."""

    return McpContractCatalog(
        claim_families=build_seed_claim_families(),
        sources=(),
        contracts=(),
        contradictions=(),
        catalog_version=CATALOG_VERSION,
    )


def admit_source(
    catalog: McpContractCatalog,
    source: ContractSourceRecord,
) -> McpContractCatalog:
    """Return a new catalog with ``source`` admitted (no authority upgrade)."""

    if not isinstance(catalog, McpContractCatalog):
        raise McpContractCatalogError("catalog must be a McpContractCatalog")
    if not isinstance(source, ContractSourceRecord):
        raise McpContractCatalogError("source must be a ContractSourceRecord")
    if catalog.get_source(source.source_id) is not None:
        raise McpContractCatalogError(
            f"source_id already registered: {source.source_id}"
        )
    return McpContractCatalog(
        claim_families=catalog.claim_families,
        sources=catalog.sources + (source,),
        contracts=catalog.contracts,
        contradictions=catalog.contradictions,
        catalog_version=catalog.catalog_version,
    )


def register_contract(
    catalog: McpContractCatalog,
    contract: ContractRecord,
    *,
    contradictions: Sequence[ContradictionRecord] = (),
    allow_contradicted: bool = True,
) -> McpContractCatalog:
    """Register a contract if its claim family is reviewed and sources exist.

    Unreviewed prose contracts fail closed.  Contradicted contracts remain
    contradicted and are stored with their contradiction records.
    """

    if not isinstance(catalog, McpContractCatalog):
        raise McpContractCatalogError("catalog must be a McpContractCatalog")
    if not isinstance(contract, ContractRecord):
        raise McpContractCatalogError("contract must be a ContractRecord")
    catalog.require_family(contract.claim_family)
    if catalog.get_contract(contract.contract_id) is not None:
        raise McpContractCatalogError(
            f"contract_id already registered: {contract.contract_id}"
        )
    for sid in contract.source_ids:
        if catalog.get_source(sid) is None:
            raise McpContractCatalogError(
                f"contract references unknown source_id: {sid}"
            )
    if contract.review_state is ReviewState.REVIEWED:
        if not contract.authority_class.may_authorize_reviewed_contract:
            raise UnreviewedContractError(
                "reviewed registration requires authorizing authority class"
            )
        for sid in contract.source_ids:
            source = catalog.get_source(sid)
            assert source is not None
            if source.kind in _PROSE_KINDS and len(contract.source_ids) == 1:
                raise UnreviewedContractError(
                    "inferred natural language cannot silently become a reviewed contract"
                )
    if contract.review_state is ReviewState.CONTRADICTED and not allow_contradicted:
        raise McpContractCatalogError("contradicted contracts are not allowed")
    if contract.review_state.is_fail_closed and contract.review_state not in {
        ReviewState.CONTRADICTED,
        ReviewState.NOMINATED,
    }:
        if contract.review_state is ReviewState.UNREVIEWED:
            raise UnreviewedContractError(
                "unknown/unreviewed prose fails closed for contract registration"
            )

    new_contradictions = list(catalog.contradictions)
    seen = {c.contradiction_id for c in new_contradictions}
    for ctr in contradictions:
        if not isinstance(ctr, ContradictionRecord):
            raise McpContractCatalogError(
                "contradictions must be ContradictionRecord instances"
            )
        if ctr.contradiction_id not in seen:
            new_contradictions.append(ctr)
            seen.add(ctr.contradiction_id)
    for cid in contract.contradiction_ids:
        if cid not in seen:
            raise McpContractCatalogError(
                f"contract references unknown contradiction_id: {cid}"
            )

    return McpContractCatalog(
        claim_families=catalog.claim_families,
        sources=catalog.sources,
        contracts=catalog.contracts + (contract,),
        contradictions=tuple(new_contradictions),
        catalog_version=catalog.catalog_version,
    )


def register_sources_and_contract(
    catalog: McpContractCatalog,
    *,
    claim_family: McpClaimFamily | str,
    subject: str,
    sources: Sequence[ContractSourceRecord],
    tool_name: str = "",
    package_id: str = "",
    metadata: Mapping[str, Any] | None = None,
    require_reviewed: bool = True,
) -> McpContractCatalog:
    """Admit sources, detect contradictions, and register the derived contract."""

    working = catalog
    for source in sources:
        if working.get_source(source.source_id) is None:
            working = admit_source(working, source)
    contract, contradictions = build_contract_from_sources(
        claim_family=claim_family,
        subject=subject,
        sources=sources,
        tool_name=tool_name,
        package_id=package_id,
        metadata=metadata,
        require_reviewed=require_reviewed,
    )
    return register_contract(
        working, contract, contradictions=contradictions, allow_contradicted=True
    )


DEFAULT_MCP_CONTRACT_CATALOG = build_default_mcp_contract_catalog()


__all__ = [
    "MCP_CONTRACT_CATALOG_INTERFACE",
    "MCP_CONTRACT_CATALOG_SCHEMA",
    "MCP_CONTRACT_SOURCE_SCHEMA",
    "MCP_CONTRACT_RECORD_SCHEMA",
    "MCP_CONTRACT_CONTRADICTION_SCHEMA",
    "MCP_CLAIM_FAMILY_SCHEMA",
    "MCP_INVALIDATOR_SCHEMA",
    "CATALOG_VERSION",
    "CONTRACT_SCHEMA_VERSION",
    "McpContractCatalogError",
    "UnknownMcpContractError",
    "UnknownMcpClaimFamilyError",
    "UnreviewedContractError",
    "ContractSourceKind",
    "SourceAuthorityClass",
    "ReviewState",
    "McpClaimFamily",
    "ContractInvalidationKind",
    "ContractInvalidator",
    "ContractSourceRecord",
    "ClaimFamilyDescriptor",
    "ContradictionRecord",
    "ContractRecord",
    "McpContractCatalog",
    "authority_for_source_kind",
    "is_prose_or_unreviewed_source",
    "reject_natural_language_claim",
    "build_source_invalidators",
    "evaluate_invalidation",
    "require_complete_version_invalidators",
    "detect_source_contradictions",
    "effective_authority_for_sources",
    "make_source_record",
    "nominate_from_prose",
    "build_contract_from_sources",
    "build_seed_claim_families",
    "build_default_mcp_contract_catalog",
    "admit_source",
    "register_contract",
    "register_sources_and_contract",
    "DEFAULT_MCP_CONTRACT_CATALOG",
]
