"""Canonical MCP contract, schema, method, and CID identity (DCR-020).

Interfaces
----------
* ``CanonicalContractIdentity@1``
* ``SemanticContractKey@1``

This module is the relocation-stable identity boundary for SwissKnife and
MCP++ contract declarations before graph construction (DCR-021).  Normative
rules:

* Canonical bytes use deterministic DAG-JSON (sorted keys, compact separators,
  no NaN/Infinity, no non-string map keys).
* Local CIDs are recomputed as CIDv1 + ``dag-json`` + ``sha2-256`` and are
  never taken from a claimed field without revalidation.
* Semantic keys bind package, operation, direction, schema root, profile,
  transport, and optional runtime instance.  Absolute host paths are rejected
  so identities remain relocation-stable.
* Equivalent declarations converge to one identity; altered bytes, direction
  or profile changes, pseudo-CIDs, and duplicate aliases stay distinct and
  typed.
* No standalone vector artifact is emitted.  Conformance vectors remain
  inline or test-local.

Conflict policy: preserve normative MCP++ / supervisor ``content_identity``
canonicalization; never trust claimed CIDs; never promote digests to CIDs.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Final

from ..proof.formal_verification_contracts import (
    ContractValidationError,
    canonical_json_bytes,
    content_identity,
)

CANONICAL_CONTRACT_IDENTITY_INTERFACE: Final = "CanonicalContractIdentity@1"
SEMANTIC_CONTRACT_KEY_INTERFACE: Final = "SemanticContractKey@1"
CANONICAL_CONTRACT_IDENTITY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/canonical-contract-identity@1"
)
SEMANTIC_CONTRACT_KEY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/semantic-contract-key@1"
)
CONTRACT_IDENTITY_EVIDENCE_TERM: Final = "dcr/identity@1"
CONTRACT_VERSION: Final[int] = 1

# Multiformat CIDv1 (dag-json / sha2-256) is lowercase base32 without padding.
_CID_V1_BASE32_RE: Final = re.compile(r"\Ab[a-z2-7]{50,}\Z")
_DIGEST_LABEL_RE: Final = re.compile(r"\Asha256:[0-9a-fA-F]{64}\Z")
_BARE_HEX_DIGEST_RE: Final = re.compile(r"\A[0-9a-fA-F]{64}\Z")
_ABSOLUTE_PATH_RE: Final = re.compile(r"\A(?:/|[A-Za-z]:[\\/]|\\\\)")

_MAX_FIELD_BYTES: Final[int] = 4_096
_MAX_ALIASES: Final[int] = 64
_MAX_SOURCE_ROOTS: Final[int] = 64


class McpContractIdentityError(ValueError):
    """Raised when a contract identity declaration is malformed or unsafe."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str = "contract_identity_error",
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.details = dict(details or {})


class PseudoCidError(McpContractIdentityError):
    """Raised when a claimed CID is digest-shaped or not a real multiformat CID."""

    def __init__(
        self,
        message: str = "claimed identity is a pseudo-CID, not a multiformat CID",
        *,
        reason_code: str = "pseudo_cid",
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message, reason_code=reason_code, details=details)


class ClaimedCidMismatchError(McpContractIdentityError):
    """Raised when a claimed CID does not match locally recomputed identity."""

    def __init__(
        self,
        *,
        claimed_cid: str,
        local_cid: str,
    ) -> None:
        super().__init__(
            "claimed CID does not match locally recomputed identity",
            reason_code="claimed_cid_mismatch",
            details={"claimed_cid": claimed_cid, "local_cid": local_cid},
        )
        self.claimed_cid = claimed_cid
        self.local_cid = local_cid


class ContractDirection(str, Enum):
    """Closed set of contract call/response directions."""

    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    BIDIRECTIONAL = "bidirectional"
    SCHEMA = "schema"
    METHOD = "method"


class ContractIdentityDisposition(str, Enum):
    """Typed identity comparison outcomes (never silent aliases)."""

    CONVERGENT = "convergent"
    DISTINCT = "distinct"
    CLAIMED_CID_MISMATCH = "claimed_cid_mismatch"
    PSEUDO_CID = "pseudo_cid"
    DUPLICATE_ALIAS = "duplicate_alias"
    INVALID_DECLARATION = "invalid_declaration"


def _norm_text(
    value: Any,
    *,
    field_name: str,
    required: bool = False,
    allow_empty: bool = False,
) -> str:
    if value is None:
        text = ""
    elif isinstance(value, str):
        text = value.strip()
    else:
        raise McpContractIdentityError(
            f"{field_name} must be a string",
            reason_code="invalid_field_type",
            details={"field": field_name, "type": type(value).__name__},
        )
    if required and not text and not allow_empty:
        raise McpContractIdentityError(
            f"{field_name} is required",
            reason_code="missing_required_field",
            details={"field": field_name},
        )
    if len(text.encode("utf-8")) > _MAX_FIELD_BYTES:
        raise McpContractIdentityError(
            f"{field_name} exceeds the {_MAX_FIELD_BYTES}-byte limit",
            reason_code="field_too_large",
            details={"field": field_name},
        )
    return text


def _norm_enum(value: Any, enum_cls: type[Enum], *, field_name: str) -> Enum:
    if isinstance(value, enum_cls):
        return value
    if isinstance(value, str) and value.strip():
        try:
            return enum_cls(value.strip())
        except ValueError as exc:
            raise McpContractIdentityError(
                f"unknown {field_name}: {value!r}",
                reason_code="unknown_enum_value",
                details={"field": field_name, "value": value},
            ) from exc
    raise McpContractIdentityError(
        f"{field_name} must be a valid {enum_cls.__name__}",
        reason_code="invalid_enum",
        details={"field": field_name},
    )


def _sorted_unique_strings(
    values: Iterable[Any] | None,
    *,
    field_name: str,
    maximum: int,
) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, (str, bytes, bytearray)):
        values = (values,)
    cleaned: list[str] = []
    seen: set[str] = set()
    for raw in values:
        text = _norm_text(raw, field_name=field_name)
        if not text or text in seen:
            continue
        seen.add(text)
        cleaned.append(text)
    cleaned.sort()
    if len(cleaned) > maximum:
        raise McpContractIdentityError(
            f"{field_name} exceeds its item limit ({maximum})",
            reason_code="field_item_limit",
            details={"field": field_name, "count": len(cleaned)},
        )
    return tuple(cleaned)


def _reject_absolute_path(value: str, *, field_name: str) -> str:
    """Reject host-absolute paths so identities stay relocation-stable."""

    text = _norm_text(value, field_name=field_name)
    if not text:
        return text
    if _ABSOLUTE_PATH_RE.match(text) or ".." in text.split("/"):
        raise McpContractIdentityError(
            f"{field_name} must be repository-relative and relocation-stable",
            reason_code="non_relocation_stable_path",
            details={"field": field_name, "value": text},
        )
    return text


def is_digest_shaped(value: Any) -> bool:
    """Return True when *value* looks like a labeled or bare SHA-256 digest."""

    if not isinstance(value, str) or not value:
        return False
    text = value.strip()
    return bool(_DIGEST_LABEL_RE.fullmatch(text) or _BARE_HEX_DIGEST_RE.fullmatch(text))


def is_pseudo_cid(value: Any) -> bool:
    """Return True for digest-shaped or obviously non-CIDv1 strings."""

    if not isinstance(value, str) or not value.strip():
        return True
    text = value.strip()
    if is_digest_shaped(text):
        return True
    if text.startswith("sha256:") or text.startswith("repository:sha256:"):
        return True
    if not _CID_V1_BASE32_RE.fullmatch(text):
        return True
    return False


def validate_multiformat_cid(value: Any, *, field_name: str = "cid") -> str:
    """Validate a claimed multiformat CIDv1 string without trusting its preimage.

    Digests and other pseudo-CIDs fail closed.  Structural CIDv1 base32 shape is
    required; preimage agreement is checked separately via local recomputation.
    """

    text = _norm_text(value, field_name=field_name, required=True)
    if is_digest_shaped(text):
        raise PseudoCidError(
            f"{field_name} must be a multiformat CID, not a digest-shaped string",
            reason_code="digest_labeled_as_cid",
            details={"field": field_name, "value_prefix": text[:24]},
        )
    if is_pseudo_cid(text):
        raise PseudoCidError(
            f"{field_name} is not a canonical lowercase base32 CIDv1",
            reason_code="pseudo_cid",
            details={"field": field_name, "value_prefix": text[:24]},
        )
    # Round-trip through content_identity preimage is impossible without bytes;
    # require the same CIDv1 dag-json/sha2-256 prefix used by content_identity.
    # Decode base32 and check the multicodec prefix when possible.
    try:
        import base64

        padding = "=" * ((8 - len(text[1:]) % 8) % 8)
        decoded = base64.b32decode((text[1:].upper() + padding).encode("ascii"))
    except (ValueError, UnicodeEncodeError) as exc:
        raise PseudoCidError(
            f"{field_name} is not valid base32 CIDv1",
            reason_code="malformed_cid",
            details={"field": field_name},
        ) from exc
    # CIDv1 + dag-json (0x0129 varint as \x01\xa9\x02) + sha2-256 (\x12\x20)
    expected_prefix = b"\x01\xa9\x02\x12\x20"
    if len(decoded) != len(expected_prefix) + 32 or not decoded.startswith(
        expected_prefix
    ):
        raise PseudoCidError(
            f"{field_name} must be CIDv1 dag-json sha2-256",
            reason_code="unsupported_cid_codec",
            details={"field": field_name},
        )
    canonical = "b" + base64.b32encode(decoded).decode("ascii").rstrip("=").lower()
    if canonical != text:
        raise PseudoCidError(
            f"{field_name} is not canonically encoded",
            reason_code="non_canonical_cid_encoding",
            details={"field": field_name},
        )
    return text


def canonical_json_cid(value: Any) -> str:
    """Return a CIDv1 dag-json/sha2-256 identity for canonical JSON bytes.

    This is the normative MCP++ / supervisor identity function for contract
    bodies.  Claimed CIDs must be compared against this local recomputation.
    """

    try:
        return content_identity(value)
    except ContractValidationError as exc:
        raise McpContractIdentityError(
            "value is not canonical-JSON encodable",
            reason_code="non_canonical_json",
            details={"cause": str(exc)},
        ) from exc


def digest_for_canonical_bytes(data: bytes) -> str:
    """Return ``sha256:<hex>`` for exact canonical bytes (never a CID)."""

    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise McpContractIdentityError(
            "canonical bytes must be bytes-like",
            reason_code="invalid_field_type",
        )
    return "sha256:" + hashlib.sha256(bytes(data)).hexdigest()


@dataclass(frozen=True, slots=True)
class SemanticContractKey:
    """Relocation-stable semantic key for one contract surface.

    Interface: ``SemanticContractKey@1``
    """

    package: str
    operation: str
    direction: ContractDirection
    schema_root: str
    profile: str
    transport: str
    runtime_instance: str = ""
    schema: str = SEMANTIC_CONTRACT_KEY_SCHEMA
    interface: str = SEMANTIC_CONTRACT_KEY_INTERFACE

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "package",
            _norm_text(self.package, field_name="package", required=True),
        )
        object.__setattr__(
            self,
            "operation",
            _norm_text(self.operation, field_name="operation", required=True),
        )
        object.__setattr__(
            self,
            "direction",
            _norm_enum(self.direction, ContractDirection, field_name="direction"),
        )
        object.__setattr__(
            self,
            "schema_root",
            _reject_absolute_path(
                _norm_text(self.schema_root, field_name="schema_root", required=True),
                field_name="schema_root",
            ),
        )
        object.__setattr__(
            self,
            "profile",
            _norm_text(self.profile, field_name="profile", required=True),
        )
        object.__setattr__(
            self,
            "transport",
            _norm_text(self.transport, field_name="transport", required=True),
        )
        object.__setattr__(
            self,
            "runtime_instance",
            _norm_text(self.runtime_instance, field_name="runtime_instance"),
        )
        object.__setattr__(
            self,
            "schema",
            _norm_text(self.schema, field_name="schema", required=True),
        )
        object.__setattr__(
            self,
            "interface",
            _norm_text(self.interface, field_name="interface", required=True),
        )
        if self.schema != SEMANTIC_CONTRACT_KEY_SCHEMA:
            raise McpContractIdentityError(
                "unsupported semantic contract key schema",
                reason_code="unsupported_schema",
            )
        if self.interface != SEMANTIC_CONTRACT_KEY_INTERFACE:
            raise McpContractIdentityError(
                "unsupported semantic contract key interface",
                reason_code="unsupported_interface",
            )

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "interface": self.interface,
            "package": self.package,
            "operation": self.operation,
            "direction": self.direction.value
            if isinstance(self.direction, ContractDirection)
            else str(self.direction),
            "schema_root": self.schema_root,
            "profile": self.profile,
            "transport": self.transport,
            "runtime_instance": self.runtime_instance,
        }

    @property
    def key_id(self) -> str:
        """Content-addressed identity of the semantic key alone."""

        return canonical_json_cid(self._identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self._identity_payload()
        payload["key_id"] = self.key_id
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SemanticContractKey":
        if not isinstance(payload, Mapping):
            raise McpContractIdentityError(
                "semantic contract key must be an object",
                reason_code="invalid_field_type",
            )
        allowed = {
            "schema",
            "interface",
            "package",
            "operation",
            "direction",
            "schema_root",
            "profile",
            "transport",
            "runtime_instance",
            "key_id",
        }
        unknown = set(payload).difference(allowed)
        if unknown:
            raise McpContractIdentityError(
                "semantic contract key contains unsupported fields: "
                + ", ".join(sorted(unknown)),
                reason_code="unknown_fields",
            )
        result = cls(
            package=str(payload.get("package") or ""),
            operation=str(payload.get("operation") or ""),
            direction=str(payload.get("direction") or ""),
            schema_root=str(payload.get("schema_root") or ""),
            profile=str(payload.get("profile") or ""),
            transport=str(payload.get("transport") or ""),
            runtime_instance=str(payload.get("runtime_instance") or ""),
            schema=str(payload.get("schema") or SEMANTIC_CONTRACT_KEY_SCHEMA),
            interface=str(payload.get("interface") or SEMANTIC_CONTRACT_KEY_INTERFACE),
        )
        claimed = payload.get("key_id")
        if claimed not in (None, "") and claimed != result.key_id:
            raise McpContractIdentityError(
                "stored semantic key_id does not match recomputed identity",
                reason_code="forged_key_id",
                details={"claimed": claimed, "local": result.key_id},
            )
        return result


def semantic_contract_key(
    *,
    package: str,
    operation: str,
    direction: ContractDirection | str,
    schema_root: str,
    profile: str,
    transport: str,
    runtime_instance: str = "",
) -> SemanticContractKey:
    """Build one relocation-stable semantic contract key."""

    return SemanticContractKey(
        package=package,
        operation=operation,
        direction=direction,  # type: ignore[arg-type]
        schema_root=schema_root,
        profile=profile,
        transport=transport,
        runtime_instance=runtime_instance,
    )


@dataclass(frozen=True, slots=True)
class CanonicalContractIdentity:
    """Full content-addressed identity for one contract declaration.

    Interface: ``CanonicalContractIdentity@1``

    ``local_cid`` is always recomputed from canonical declaration bytes.
    ``claimed_cid`` is retained for audit and may only agree after local
    recomputation; mismatches and pseudo-CIDs are typed dispositions.
    """

    semantic_key: SemanticContractKey
    declaration: Mapping[str, Any]
    local_cid: str
    claimed_cid: str = ""
    canonical_digest: str = ""
    source_roots: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()
    disposition: ContractIdentityDisposition = (
        ContractIdentityDisposition.CONVERGENT
    )
    reason_codes: tuple[str, ...] = ()
    schema: str = CANONICAL_CONTRACT_IDENTITY_SCHEMA
    interface: str = CANONICAL_CONTRACT_IDENTITY_INTERFACE
    evidence_term: str = CONTRACT_IDENTITY_EVIDENCE_TERM

    def __post_init__(self) -> None:
        if not isinstance(self.semantic_key, SemanticContractKey):
            raise McpContractIdentityError(
                "semantic_key must be a SemanticContractKey",
                reason_code="invalid_field_type",
            )
        if not isinstance(self.declaration, Mapping):
            raise McpContractIdentityError(
                "declaration must be a mapping",
                reason_code="invalid_field_type",
            )
        object.__setattr__(
            self, "declaration", MappingProxyType(dict(self.declaration))
        )
        # Normalize provenance and labels before any identity recomputation so
        # sorted unique roots contribute deterministically to local_cid.
        object.__setattr__(
            self,
            "source_roots",
            tuple(
                _reject_absolute_path(item, field_name="source_roots")
                for item in _sorted_unique_strings(
                    self.source_roots,
                    field_name="source_roots",
                    maximum=_MAX_SOURCE_ROOTS,
                )
            ),
        )
        object.__setattr__(
            self,
            "aliases",
            _sorted_unique_strings(
                self.aliases, field_name="aliases", maximum=_MAX_ALIASES
            ),
        )
        object.__setattr__(
            self,
            "disposition",
            _norm_enum(
                self.disposition,
                ContractIdentityDisposition,
                field_name="disposition",
            ),
        )
        object.__setattr__(
            self,
            "reason_codes",
            _sorted_unique_strings(
                self.reason_codes, field_name="reason_codes", maximum=64
            ),
        )
        object.__setattr__(
            self,
            "schema",
            _norm_text(self.schema, field_name="schema", required=True),
        )
        object.__setattr__(
            self,
            "interface",
            _norm_text(self.interface, field_name="interface", required=True),
        )
        object.__setattr__(
            self,
            "evidence_term",
            _norm_text(self.evidence_term, field_name="evidence_term", required=True),
        )
        if self.schema != CANONICAL_CONTRACT_IDENTITY_SCHEMA:
            raise McpContractIdentityError(
                "unsupported canonical contract identity schema",
                reason_code="unsupported_schema",
            )
        if self.interface != CANONICAL_CONTRACT_IDENTITY_INTERFACE:
            raise McpContractIdentityError(
                "unsupported canonical contract identity interface",
                reason_code="unsupported_interface",
            )
        claimed = _norm_text(self.claimed_cid, field_name="claimed_cid")
        object.__setattr__(self, "claimed_cid", claimed)
        # Recompute and refuse forged local_cid / digest pairs.
        expected_cid = canonical_json_cid(self._declaration_payload())
        object.__setattr__(
            self,
            "local_cid",
            validate_multiformat_cid(self.local_cid, field_name="local_cid"),
        )
        if expected_cid != self.local_cid:
            raise McpContractIdentityError(
                "local_cid does not match recomputed declaration identity",
                reason_code="forged_local_cid",
                details={"claimed": self.local_cid, "local": expected_cid},
            )
        expected_digest = digest_for_canonical_bytes(
            canonical_json_bytes(self._declaration_payload())
        )
        digest = _norm_text(self.canonical_digest, field_name="canonical_digest")
        if not digest:
            digest = expected_digest
        elif not is_digest_shaped(digest) or not digest.startswith("sha256:"):
            raise McpContractIdentityError(
                "canonical_digest must be sha256:<64 hex>",
                reason_code="invalid_digest",
            )
        object.__setattr__(self, "canonical_digest", digest)
        if expected_digest != self.canonical_digest:
            raise McpContractIdentityError(
                "canonical_digest does not match recomputed declaration bytes",
                reason_code="forged_canonical_digest",
            )

    def _declaration_payload(self) -> dict[str, Any]:
        # Aliases and claimed CIDs are audit metadata only: they must not alter
        # the local declaration CID, or equivalent bodies would diverge when
        # labeled differently.
        return {
            "semantic_key": self.semantic_key._identity_payload(),
            "declaration": dict(self.declaration),
            "source_roots": list(self.source_roots),
        }

    @property
    def content_id(self) -> str:
        """Identity of the complete identity record (excluding disposition)."""

        return canonical_json_cid(
            {
                "schema": self.schema,
                "interface": self.interface,
                "evidence_term": self.evidence_term,
                "semantic_key": self.semantic_key._identity_payload(),
                "declaration": dict(self.declaration),
                "local_cid": self.local_cid,
                "claimed_cid": self.claimed_cid,
                "canonical_digest": self.canonical_digest,
                "source_roots": list(self.source_roots),
                "aliases": list(self.aliases),
                "disposition": self.disposition.value
                if isinstance(self.disposition, ContractIdentityDisposition)
                else str(self.disposition),
                "reason_codes": list(self.reason_codes),
            }
        )

    @property
    def claimed_matches_local(self) -> bool:
        return bool(self.claimed_cid) and self.claimed_cid == self.local_cid

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "interface": self.interface,
            "evidence_term": self.evidence_term,
            "content_id": self.content_id,
            "semantic_key": self.semantic_key.to_dict(),
            "declaration": dict(self.declaration),
            "local_cid": self.local_cid,
            "claimed_cid": self.claimed_cid,
            "canonical_digest": self.canonical_digest,
            "source_roots": list(self.source_roots),
            "aliases": list(self.aliases),
            "disposition": self.disposition.value
            if isinstance(self.disposition, ContractIdentityDisposition)
            else str(self.disposition),
            "reason_codes": list(self.reason_codes),
            "policies": {
                "trust_claimed_cid": False,
                "digest_labeled_as_cid_allowed": False,
                "absolute_paths_allowed": False,
                "undeclared_vector_artifact_allowed": False,
                "cross_direction_alias_allowed": False,
                "cross_profile_alias_allowed": False,
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CanonicalContractIdentity":
        if not isinstance(payload, Mapping):
            raise McpContractIdentityError(
                "canonical contract identity must be an object",
                reason_code="invalid_field_type",
            )
        allowed = {
            "schema",
            "interface",
            "evidence_term",
            "content_id",
            "semantic_key",
            "declaration",
            "local_cid",
            "claimed_cid",
            "canonical_digest",
            "source_roots",
            "aliases",
            "disposition",
            "reason_codes",
            "policies",
        }
        unknown = set(payload).difference(allowed)
        if unknown:
            raise McpContractIdentityError(
                "canonical contract identity contains unsupported fields: "
                + ", ".join(sorted(unknown)),
                reason_code="unknown_fields",
            )
        key_payload = payload.get("semantic_key")
        if not isinstance(key_payload, Mapping):
            raise McpContractIdentityError(
                "semantic_key must be an object",
                reason_code="invalid_field_type",
            )
        declaration = payload.get("declaration")
        if not isinstance(declaration, Mapping):
            raise McpContractIdentityError(
                "declaration must be a mapping",
                reason_code="invalid_field_type",
            )
        result = cls(
            semantic_key=SemanticContractKey.from_dict(key_payload),
            declaration=dict(declaration),
            local_cid=str(payload.get("local_cid") or ""),
            claimed_cid=str(payload.get("claimed_cid") or ""),
            canonical_digest=str(payload.get("canonical_digest") or ""),
            source_roots=tuple(payload.get("source_roots") or ()),
            aliases=tuple(payload.get("aliases") or ()),
            disposition=str(
                payload.get("disposition")
                or ContractIdentityDisposition.CONVERGENT.value
            ),
            reason_codes=tuple(payload.get("reason_codes") or ()),
            schema=str(payload.get("schema") or CANONICAL_CONTRACT_IDENTITY_SCHEMA),
            interface=str(
                payload.get("interface") or CANONICAL_CONTRACT_IDENTITY_INTERFACE
            ),
            evidence_term=str(
                payload.get("evidence_term") or CONTRACT_IDENTITY_EVIDENCE_TERM
            ),
        )
        claimed_content = payload.get("content_id")
        if claimed_content not in (None, "") and claimed_content != result.content_id:
            raise McpContractIdentityError(
                "stored content_id does not match recomputed identity",
                reason_code="forged_content_id",
            )
        return result


def identify_contract_declaration(
    *,
    package: str,
    operation: str,
    direction: ContractDirection | str,
    schema_root: str,
    profile: str,
    transport: str,
    declaration: Mapping[str, Any],
    runtime_instance: str = "",
    claimed_cid: str = "",
    source_roots: Sequence[str] = (),
    aliases: Sequence[str] = (),
    require_claimed_match: bool = False,
) -> CanonicalContractIdentity:
    """Derive a canonical identity for one declaration, never trusting claims.

    When *claimed_cid* is provided it is classified:

    * empty — no claim; disposition stays convergent for the local identity
    * equals local recomputation — retained and disposition convergent
    * pseudo-CID — typed ``PSEUDO_CID`` (raises when *require_claimed_match*)
    * real CID mismatch — typed ``CLAIMED_CID_MISMATCH`` (raises when required)
    """

    if not isinstance(declaration, Mapping):
        raise McpContractIdentityError(
            "declaration must be a mapping",
            reason_code="invalid_field_type",
        )
    key = semantic_contract_key(
        package=package,
        operation=operation,
        direction=direction,
        schema_root=schema_root,
        profile=profile,
        transport=transport,
        runtime_instance=runtime_instance,
    )
    roots = tuple(
        _reject_absolute_path(item, field_name="source_roots")
        for item in _sorted_unique_strings(
            source_roots, field_name="source_roots", maximum=_MAX_SOURCE_ROOTS
        )
    )
    alias_values = _sorted_unique_strings(
        aliases, field_name="aliases", maximum=_MAX_ALIASES
    )
    body = {
        "semantic_key": key._identity_payload(),
        "declaration": dict(declaration),
        "source_roots": list(roots),
    }
    local_cid = canonical_json_cid(body)
    canonical_digest = digest_for_canonical_bytes(canonical_json_bytes(body))
    claimed = _norm_text(claimed_cid, field_name="claimed_cid")
    disposition = ContractIdentityDisposition.CONVERGENT
    reasons: list[str] = []

    if claimed:
        if is_pseudo_cid(claimed) or is_digest_shaped(claimed):
            disposition = ContractIdentityDisposition.PSEUDO_CID
            reasons.append("pseudo_cid")
            if require_claimed_match:
                raise PseudoCidError(
                    details={"claimed_cid": claimed, "local_cid": local_cid}
                )
        else:
            try:
                validated_claim = validate_multiformat_cid(
                    claimed, field_name="claimed_cid"
                )
            except PseudoCidError:
                disposition = ContractIdentityDisposition.PSEUDO_CID
                reasons.append("pseudo_cid")
                validated_claim = claimed
                if require_claimed_match:
                    raise
            else:
                claimed = validated_claim
                if claimed != local_cid:
                    disposition = ContractIdentityDisposition.CLAIMED_CID_MISMATCH
                    reasons.append("claimed_cid_mismatch")
                    if require_claimed_match:
                        raise ClaimedCidMismatchError(
                            claimed_cid=claimed, local_cid=local_cid
                        )

    return CanonicalContractIdentity(
        semantic_key=key,
        declaration=dict(declaration),
        local_cid=local_cid,
        claimed_cid=claimed,
        canonical_digest=canonical_digest,
        source_roots=roots,
        aliases=alias_values,
        disposition=disposition,
        reason_codes=tuple(reasons),
    )


def identities_converge(
    left: CanonicalContractIdentity,
    right: CanonicalContractIdentity,
) -> bool:
    """Return True when two declarations share the same local CID and key."""

    if not isinstance(left, CanonicalContractIdentity) or not isinstance(
        right, CanonicalContractIdentity
    ):
        raise McpContractIdentityError(
            "identities_converge requires CanonicalContractIdentity values",
            reason_code="invalid_field_type",
        )
    return (
        left.local_cid == right.local_cid
        and left.semantic_key.key_id == right.semantic_key.key_id
    )


def compare_contract_identities(
    left: CanonicalContractIdentity,
    right: CanonicalContractIdentity,
) -> ContractIdentityDisposition:
    """Classify the relationship between two identities without collapsing them.

    * ``CONVERGENT`` — same local CID and semantic key
    * ``DUPLICATE_ALIAS`` — same local CID but differing alias sets or claims
    * ``DISTINCT`` — different semantic keys or declaration bytes
    * retained typed dispositions when either side already failed closed
    """

    if not isinstance(left, CanonicalContractIdentity) or not isinstance(
        right, CanonicalContractIdentity
    ):
        raise McpContractIdentityError(
            "compare_contract_identities requires CanonicalContractIdentity values",
            reason_code="invalid_field_type",
        )
    for side in (left, right):
        if side.disposition in {
            ContractIdentityDisposition.PSEUDO_CID,
            ContractIdentityDisposition.CLAIMED_CID_MISMATCH,
            ContractIdentityDisposition.INVALID_DECLARATION,
        }:
            return side.disposition
    if identities_converge(left, right):
        if left.aliases != right.aliases or left.claimed_cid != right.claimed_cid:
            return ContractIdentityDisposition.DUPLICATE_ALIAS
        return ContractIdentityDisposition.CONVERGENT
    # Same semantic key with different declaration bytes is distinct, not an
    # alias; direction/profile/operation changes also stay distinct.
    return ContractIdentityDisposition.DISTINCT


def classify_alias_collision(
    identities: Sequence[CanonicalContractIdentity],
) -> tuple[ContractIdentityDisposition, tuple[str, ...]]:
    """Detect typed duplicate-alias collisions across a set of identities.

    Returns ``(disposition, colliding_alias_names)``.  Distinct semantic keys
    that share a surface alias name are reported as ``DUPLICATE_ALIAS`` so
    graph construction cannot silently merge them.
    """

    if not identities:
        return ContractIdentityDisposition.CONVERGENT, ()
    by_alias: dict[str, list[CanonicalContractIdentity]] = {}
    for identity in identities:
        if not isinstance(identity, CanonicalContractIdentity):
            raise McpContractIdentityError(
                "classify_alias_collision requires CanonicalContractIdentity values",
                reason_code="invalid_field_type",
            )
        if identity.disposition in {
            ContractIdentityDisposition.PSEUDO_CID,
            ContractIdentityDisposition.CLAIMED_CID_MISMATCH,
            ContractIdentityDisposition.INVALID_DECLARATION,
        }:
            return identity.disposition, ()
        for alias in identity.aliases:
            by_alias.setdefault(alias, []).append(identity)
    collisions: list[str] = []
    for alias, group in sorted(by_alias.items()):
        key_ids = {item.semantic_key.key_id for item in group}
        local_cids = {item.local_cid for item in group}
        if len(key_ids) > 1 or len(local_cids) > 1:
            collisions.append(alias)
    if collisions:
        return ContractIdentityDisposition.DUPLICATE_ALIAS, tuple(collisions)
    return ContractIdentityDisposition.CONVERGENT, ()


__all__ = [
    "CANONICAL_CONTRACT_IDENTITY_INTERFACE",
    "CANONICAL_CONTRACT_IDENTITY_SCHEMA",
    "CONTRACT_IDENTITY_EVIDENCE_TERM",
    "CONTRACT_VERSION",
    "SEMANTIC_CONTRACT_KEY_INTERFACE",
    "SEMANTIC_CONTRACT_KEY_SCHEMA",
    "CanonicalContractIdentity",
    "ClaimedCidMismatchError",
    "ContractDirection",
    "ContractIdentityDisposition",
    "McpContractIdentityError",
    "PseudoCidError",
    "SemanticContractKey",
    "canonical_json_cid",
    "classify_alias_collision",
    "compare_contract_identities",
    "digest_for_canonical_bytes",
    "identify_contract_declaration",
    "identities_converge",
    "is_digest_shaped",
    "is_pseudo_cid",
    "semantic_contract_key",
    "validate_multiformat_cid",
]
