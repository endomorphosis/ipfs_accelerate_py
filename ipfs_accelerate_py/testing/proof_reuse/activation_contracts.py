"""Runtime activation and candidate-context contracts (PTR-131).

This module seals the fail-closed composition boundary for automatic
proof-backed pytest reuse.  It does not authorize skips, import pytest, open a
network socket, or install packages.  Later composition steps (default identity
services, candidate-context store, revalidation, deferred issuer, plugin
wiring) must implement against these typed records.

Authority doctrine carried by this contract:

* A **locator hint** is mutable retrieval narrowing only.
* An **immutable candidate context** retains exact pass-time canonical bytes
  (execution key, static/runtime traces, forest/environment/policy, receipt).
* A **fresh current context** is rebuilt from live state for comparison and is
  never a historical trace relabeled as current.
* A **trusted pass receipt** is post-pass admitted evidence of one complete
  execution; it is not a skip by itself.
* A **deferred proof request** is a public-only issuance envelope; missing
  proving capability yields ``DEFERRED`` / ``RUN`` while retaining the receipt.
* An **authoritative certificate** is the only artifact class that, after exact
  current-context comparison and local verification, may authorize ``SKIP``.

At every content-addressed boundary, retained canonical bytes must decode,
re-canonicalize, and rehash to the claimed CID.  Before ``SKIP``, current
AST, static, runtime, environment, and policy identities must match the
candidate.  Post-pass runtime observations are recorded after the single real
setup/call/teardown lifecycle without re-invoking the test body.  Every
missing, malformed, incompatible, timed-out, or exceptional optional capability
maps to ``RUN`` or ``DEFERRED`` and must never fail pytest collection.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar, Dict, Final, Iterable, NoReturn, Tuple

from ...agent_supervisor.proof.formal_verification_contracts import (
    CanonicalContract,
    ContractValidationError,
    _contains_private_material,
    _enum,
    _mapping,
    _nonnegative_int,
    _reject_unknown_fields,
    _text,
    bounded_rejection_reason,
    canonical_json_bytes,
    content_identity,
)


# ---------------------------------------------------------------------------
# Schema / interface constants
# ---------------------------------------------------------------------------

ACTIVATION_CONTRACT_VERSION: Final = 1
SCHEMA_VERSION: Final = ACTIVATION_CONTRACT_VERSION

PROOF_REUSE_ACTIVATION_CONTRACT_INTERFACE: Final = (
    "ProofReuseActivationContract@1"
)
CANDIDATE_EXECUTION_CONTEXT_INTERFACE: Final = "CandidateExecutionContext@1"
CURRENT_EXECUTION_CONTEXT_INTERFACE: Final = "CurrentExecutionContext@1"
RUNTIME_REUSE_DISPOSITION_INTERFACE: Final = "RuntimeReuseDisposition@1"
LOCATOR_HINT_INTERFACE: Final = "LocatorHint@1"
DEFERRED_PROOF_REQUEST_INTERFACE: Final = "DeferredProofRequest@1"
TRUSTED_PASS_RECEIPT_BINDING_INTERFACE: Final = "TrustedPassReceiptBinding@1"
AUTHORITATIVE_CERTIFICATE_BINDING_INTERFACE: Final = (
    "AuthoritativeCertificateBinding@1"
)
POST_PASS_RUNTIME_OBSERVATION_INTERFACE: Final = (
    "PostPassRuntimeObservation@1"
)
CONTENT_ADDRESSED_BOUNDARY_INTERFACE: Final = "ContentAddressedBoundary@1"

PROOF_REUSE_ACTIVATION_CONTRACT_SCHEMA: Final = (
    "ipfs_accelerate_py/testing/proof-reuse/activation-contract@1"
)
CANDIDATE_EXECUTION_CONTEXT_SCHEMA: Final = (
    "ipfs_accelerate_py/testing/proof-reuse/candidate-execution-context@1"
)
CURRENT_EXECUTION_CONTEXT_SCHEMA: Final = (
    "ipfs_accelerate_py/testing/proof-reuse/current-execution-context@1"
)
RUNTIME_REUSE_DISPOSITION_SCHEMA: Final = (
    "ipfs_accelerate_py/testing/proof-reuse/runtime-reuse-disposition@1"
)
LOCATOR_HINT_SCHEMA: Final = (
    "ipfs_accelerate_py/testing/proof-reuse/locator-hint@1"
)
DEFERRED_PROOF_REQUEST_SCHEMA: Final = (
    "ipfs_accelerate_py/testing/proof-reuse/deferred-proof-request@1"
)
TRUSTED_PASS_RECEIPT_BINDING_SCHEMA: Final = (
    "ipfs_accelerate_py/testing/proof-reuse/trusted-pass-receipt-binding@1"
)
AUTHORITATIVE_CERTIFICATE_BINDING_SCHEMA: Final = (
    "ipfs_accelerate_py/testing/proof-reuse/authoritative-certificate-binding@1"
)
POST_PASS_RUNTIME_OBSERVATION_SCHEMA: Final = (
    "ipfs_accelerate_py/testing/proof-reuse/post-pass-runtime-observation@1"
)
CONTENT_ADDRESSED_BOUNDARY_SCHEMA: Final = (
    "ipfs_accelerate_py/testing/proof-reuse/content-addressed-boundary@1"
)

MAX_TEXT_CHARS: Final = 4_096
MAX_REASON_CHARS: Final = 256
MAX_DIAGNOSTIC_KEYS: Final = 32
MAX_DIAGNOSTIC_VALUE_CHARS: Final = 512
MAX_SEQUENCE_ITEMS: Final = 256
MAX_COMPONENT_ENTRIES: Final = 128
MAX_CANONICAL_BYTES: Final = 8 * 1_048_576
MAX_NODE_ID_CHARS: Final = 2_048

# Fixed authority sequence for automatic runtime activation (plan §13.2).
ACTIVATION_AUTHORITY_SEQUENCE: Final[Tuple[str, ...]] = (
    "compute_stable_locator",
    "resolve_bounded_candidate_descriptor",
    "load_retained_candidate_bytes_and_rehash",
    "rebuild_current_dependency_frontier",
    "compare_current_and_verify_authoritative_certificate",
    "execute_once_or_emit_skip",
    "record_post_pass_observations_without_duplicate_call",
    "request_deferred_issuance_public_only",
    "publish_candidate_certificate_state_atomically",
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ActivationContractError(ContractValidationError):
    """Raised when an activation/candidate-context contract is malformed."""

    __test__ = False


def _raise_contract(message: str, cause: BaseException | None = None) -> NoReturn:
    if cause is None:
        raise ActivationContractError(message)
    raise ActivationContractError(message) from cause


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ArtifactRole(str, Enum):
    """Closed vocabulary distinguishing non-interchangeable activation artifacts.

    These roles are deliberately distinct types of evidence.  A locator hint
    cannot be relabeled as a candidate context, a historical trace cannot be
    promoted to a current context, and a deferred request cannot become an
    authoritative certificate.
    """

    LOCATOR_HINT = "locator_hint"
    IMMUTABLE_CANDIDATE_CONTEXT = "immutable_candidate_context"
    FRESH_CURRENT_CONTEXT = "fresh_current_context"
    TRUSTED_PASS_RECEIPT = "trusted_pass_receipt"
    DEFERRED_PROOF_REQUEST = "deferred_proof_request"
    AUTHORITATIVE_CERTIFICATE = "authoritative_certificate"


class RuntimeReuseAction(str, Enum):
    """Closed disposition set at the activation boundary.

    ``SKIP`` requires exact current-context comparison plus a locally verified
    authoritative certificate.  ``DEFERRED`` retains a trusted pass receipt and
    waits for later issuance.  Everything else is ``RUN``.
    """

    RUN = "RUN"
    SKIP = "SKIP"
    DEFERRED = "DEFERRED"


class OptionalCapabilityFaultKind(str, Enum):
    """Closed fault classes for optional proving / transport / store capabilities."""

    MISSING = "missing"
    MALFORMED = "malformed"
    INCOMPATIBLE = "incompatible"
    TIMED_OUT = "timed_out"
    EXCEPTIONAL = "exceptional"


class SkipComparisonDimension(str, Enum):
    """Dimensions that must match before an authoritative SKIP is legal."""

    AST = "ast"
    STATIC = "static"
    RUNTIME = "runtime"
    ENVIRONMENT = "environment"
    POLICY = "policy"


SKIP_REQUIRED_COMPARISON_DIMENSIONS: Final[Tuple[SkipComparisonDimension, ...]] = (
    SkipComparisonDimension.AST,
    SkipComparisonDimension.STATIC,
    SkipComparisonDimension.RUNTIME,
    SkipComparisonDimension.ENVIRONMENT,
    SkipComparisonDimension.POLICY,
)

# Roles that may never authorize SKIP on their own.
_NON_AUTHORIZING_ROLES: Final = frozenset(
    {
        ArtifactRole.LOCATOR_HINT,
        ArtifactRole.IMMUTABLE_CANDIDATE_CONTEXT,
        ArtifactRole.FRESH_CURRENT_CONTEXT,
        ArtifactRole.TRUSTED_PASS_RECEIPT,
        ArtifactRole.DEFERRED_PROOF_REQUEST,
    }
)

# Capability faults that prefer DEFERRED when a trusted receipt was retained.
_DEFERRED_PREFER_FAULTS: Final = frozenset(
    {
        OptionalCapabilityFaultKind.MISSING,
        OptionalCapabilityFaultKind.INCOMPATIBLE,
        OptionalCapabilityFaultKind.TIMED_OUT,
    }
)


# ---------------------------------------------------------------------------
# Shared normalizers
# ---------------------------------------------------------------------------


def _bounded_text(
    value: Any,
    *,
    field_name: str,
    required: bool = False,
    max_chars: int = MAX_TEXT_CHARS,
) -> str:
    try:
        text = _text(value, field_name=field_name, required=required)
    except ContractValidationError as exc:
        _raise_contract(str(exc), exc)
    if len(text) > max_chars:
        _raise_contract(
            "%s exceeds bounded length of %d characters" % (field_name, max_chars)
        )
    return text


def _bounded_ids(
    values: Any,
    *,
    field_name: str,
    required: bool = False,
    max_items: int = MAX_SEQUENCE_ITEMS,
) -> Tuple[str, ...]:
    if values is None:
        items: Tuple[str, ...] = ()
    elif isinstance(values, str):
        items = (values,)
    elif isinstance(values, Sequence) and not isinstance(values, (bytes, bytearray)):
        items = tuple(values)
    else:
        _raise_contract("%s must be a sequence of strings" % field_name)
        raise AssertionError("unreachable")
    normalized: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = _bounded_text(item, field_name=field_name, required=True)
        if text in seen:
            continue
        seen.add(text)
        normalized.append(text)
    if len(normalized) > max_items:
        _raise_contract(
            "%s exceeds bounded item count of %d" % (field_name, max_items)
        )
    if required and not normalized:
        _raise_contract("%s must not be empty" % field_name)
    return tuple(sorted(normalized))


def _bounded_mapping(
    value: Any,
    *,
    field_name: str,
    max_keys: int = MAX_DIAGNOSTIC_KEYS,
    max_value_chars: int = MAX_DIAGNOSTIC_VALUE_CHARS,
) -> Dict[str, Any]:
    if value is None:
        normalized: Dict[str, Any] = {}
    elif not isinstance(value, Mapping):
        _raise_contract("%s must be a mapping" % field_name)
        raise AssertionError("unreachable")
    else:
        try:
            normalized = _mapping(value, field_name=field_name)
        except ContractValidationError as exc:
            _raise_contract(str(exc), exc)
    if len(normalized) > max_keys:
        _raise_contract(
            "%s exceeds bounded key count of %d" % (field_name, max_keys)
        )
    for key, item in normalized.items():
        if not isinstance(key, str) or not key:
            _raise_contract("%s keys must be non-empty strings" % field_name)
        if len(key) > MAX_TEXT_CHARS:
            _raise_contract("%s key exceeds bounded length" % field_name)
        if isinstance(item, float):
            _raise_contract("%s must not contain floating-point values" % field_name)
        if isinstance(item, str) and len(item) > max_value_chars:
            _raise_contract(
                "%s value exceeds bounded length of %d characters"
                % (field_name, max_value_chars)
            )
    return normalized


def _component_map(value: Any, *, field_name: str) -> Dict[str, str]:
    raw = _bounded_mapping(
        value,
        field_name=field_name,
        max_keys=MAX_COMPONENT_ENTRIES,
        max_value_chars=MAX_TEXT_CHARS,
    )
    result: Dict[str, str] = {}
    for key, item in raw.items():
        if not isinstance(item, str) or not item.strip():
            _raise_contract(
                "%s values must be non-empty identity strings" % field_name
            )
        result[key] = item.strip()
    return dict(sorted(result.items()))


def _safe_enum(value: Any, enum_type: type, *, field_name: str) -> Any:
    try:
        return _enum(value, enum_type, field_name=field_name)
    except ContractValidationError as exc:
        _raise_contract(str(exc), exc)
        raise AssertionError("unreachable") from exc


def _safe_nonnegative_int(value: Any, *, field_name: str) -> int:
    try:
        return _nonnegative_int(value, field_name=field_name)
    except ContractValidationError as exc:
        _raise_contract(str(exc), exc)
        raise AssertionError("unreachable") from exc


def _safe_reject_unknown_fields(
    payload: Mapping[str, Any],
    allowed: Iterable[str],
    *,
    artifact_name: str,
) -> None:
    try:
        _reject_unknown_fields(payload, allowed, artifact_name=artifact_name)
    except ContractValidationError as exc:
        _raise_contract(str(exc), exc)


def _require_versioned_interface(
    payload: Mapping[str, Any],
    *,
    expected_interface: str,
    artifact_name: str,
) -> None:
    interface = payload.get("interface")
    version = payload.get("contract_version")
    if interface in (None, "") and version in (None, ""):
        _raise_contract(
            "%s is versionless; require interface %s or contract_version %s"
            % (artifact_name, expected_interface, ACTIVATION_CONTRACT_VERSION)
        )
    if interface not in (None, "", expected_interface):
        _raise_contract(
            "%s interface must be %s" % (artifact_name, expected_interface)
        )
    if version not in (None, "", ACTIVATION_CONTRACT_VERSION):
        _raise_contract(
            "%s contract_version must be %s"
            % (artifact_name, ACTIVATION_CONTRACT_VERSION)
        )


def _schema_or_raise(
    payload: Mapping[str, Any], expected: str, *, artifact_name: str
) -> None:
    if not isinstance(payload, Mapping):
        _raise_contract("%s must be an object" % artifact_name)
    supplied = payload.get("schema")
    if supplied not in (None, "", expected):
        _raise_contract(
            "unsupported %s schema; use %s" % (artifact_name, expected)
        )


def _bool(value: Any, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        _raise_contract("%s must be a boolean" % field_name)
    return value


def _diagnostics(value: Any) -> Dict[str, Any]:
    return _bounded_mapping(
        value,
        field_name="diagnostics",
        max_keys=MAX_DIAGNOSTIC_KEYS,
        max_value_chars=MAX_DIAGNOSTIC_VALUE_CHARS,
    )


# ---------------------------------------------------------------------------
# Content-addressed boundary: canonical bytes + CID rehash
# ---------------------------------------------------------------------------


def rehash_retained_canonical_bytes(data: bytes) -> str:
    """Rehash retained exact canonical DAG-JSON bytes to CIDv1/base32/dag-json.

    The retained bytes must already be in exact canonical form.  Decode, require
    re-canonicalization equality, then derive the content identity.  Callers
    use this at every content-addressed store/load/publication boundary.
    """

    if type(data) is not bytes:
        _raise_contract("canonical bytes must be exact bytes")
    if not data:
        _raise_contract("canonical bytes must be nonempty")
    if len(data) > MAX_CANONICAL_BYTES:
        _raise_contract(
            "canonical bytes exceed bounded length of %d" % MAX_CANONICAL_BYTES
        )
    try:
        text = data.decode("utf-8")
        payload = json.loads(text)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        _raise_contract("canonical bytes are not valid UTF-8 DAG-JSON", exc)
    if not isinstance(payload, (dict, list)):
        _raise_contract("canonical bytes must encode a JSON object or array")
    try:
        if _contains_private_material(payload):
            _raise_contract(bounded_rejection_reason("private_material"))
        recomputed = canonical_json_bytes(payload)
    except ContractValidationError as exc:
        _raise_contract(str(exc), exc)
    except Exception as exc:  # pragma: no cover - defensive
        _raise_contract("canonical bytes failed re-canonicalization", exc)
    if recomputed != data:
        _raise_contract("retained bytes are not canonical DAG-JSON form")
    try:
        return content_identity(payload)
    except Exception as exc:  # pragma: no cover - defensive
        _raise_contract("failed to derive content identity for retained bytes", exc)


def cid_for_public_payload(payload: Mapping[str, Any] | Sequence[Any] | Any) -> str:
    """Derive the CIDv1 identity of a public structured payload."""

    try:
        if _contains_private_material(payload):
            _raise_contract(bounded_rejection_reason("private_material"))
        return content_identity(payload)
    except ActivationContractError:
        raise
    except ContractValidationError as exc:
        _raise_contract(str(exc), exc)
    except Exception as exc:  # pragma: no cover - defensive
        _raise_contract("failed to derive content identity", exc)


@dataclass(frozen=True)
class ContentAddressedBoundaryAdmission(CanonicalContract):
    """Result of admitting one content-addressed boundary.

    ``admitted`` is true only when retained canonical bytes rehash to the
    claimed CID.  A failed admission never becomes skip authority.
    """

    __test__: ClassVar[bool] = False
    SCHEMA: ClassVar[str] = CONTENT_ADDRESSED_BOUNDARY_SCHEMA

    role: ArtifactRole
    claimed_cid: str
    actual_cid: str = ""
    admitted: bool = False
    byte_length: int = 0
    reason_code: str = ""
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "role",
            _safe_enum(self.role, ArtifactRole, field_name="role"),
        )
        object.__setattr__(
            self,
            "claimed_cid",
            _bounded_text(self.claimed_cid, field_name="claimed_cid"),
        )
        object.__setattr__(
            self,
            "actual_cid",
            _bounded_text(self.actual_cid, field_name="actual_cid"),
        )
        object.__setattr__(
            self, "admitted", _bool(self.admitted, field_name="admitted")
        )
        object.__setattr__(
            self,
            "byte_length",
            _safe_nonnegative_int(self.byte_length, field_name="byte_length"),
        )
        object.__setattr__(
            self,
            "reason_code",
            _bounded_text(
                self.reason_code, field_name="reason_code", max_chars=MAX_REASON_CHARS
            ),
        )
        object.__setattr__(
            self, "diagnostics", _diagnostics(self.diagnostics)
        )
        if self.admitted:
            if not self.claimed_cid or not self.actual_cid:
                _raise_contract(
                    "admitted content-addressed boundary requires claimed and actual CID"
                )
            if self.claimed_cid != self.actual_cid:
                _raise_contract(
                    "admitted content-addressed boundary requires CID agreement"
                )
            if self.byte_length <= 0:
                _raise_contract(
                    "admitted content-addressed boundary requires positive byte length"
                )

    @property
    def interface(self) -> str:
        return CONTENT_ADDRESSED_BOUNDARY_INTERFACE

    @property
    def boundary_id(self) -> str:
        return self.content_id

    def _payload(self) -> Dict[str, Any]:
        return {
            "contract_version": ACTIVATION_CONTRACT_VERSION,
            "interface": CONTENT_ADDRESSED_BOUNDARY_INTERFACE,
            "role": self.role,
            "claimed_cid": self.claimed_cid,
            "actual_cid": self.actual_cid,
            "admitted": self.admitted,
            "byte_length": self.byte_length,
            "reason_code": self.reason_code,
            "diagnostics": dict(self.diagnostics),
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "ContentAddressedBoundaryAdmission":
        _schema_or_raise(
            payload, cls.SCHEMA, artifact_name="content-addressed boundary"
        )
        _require_versioned_interface(
            payload,
            expected_interface=CONTENT_ADDRESSED_BOUNDARY_INTERFACE,
            artifact_name="content-addressed boundary",
        )
        _safe_reject_unknown_fields(
            payload,
            {
                "schema",
                "contract_version",
                "interface",
                "role",
                "claimed_cid",
                "actual_cid",
                "admitted",
                "byte_length",
                "reason_code",
                "diagnostics",
                "content_id",
                "boundary_id",
            },
            artifact_name="content-addressed boundary",
        )
        result = cls(
            role=payload.get("role", ArtifactRole.LOCATOR_HINT),
            claimed_cid=payload.get("claimed_cid", ""),
            actual_cid=payload.get("actual_cid", ""),
            admitted=payload.get("admitted", False),
            byte_length=payload.get("byte_length", 0),
            reason_code=payload.get("reason_code", ""),
            diagnostics=payload.get("diagnostics") or {},
        )
        claimed = payload.get("boundary_id") or payload.get("content_id")
        if claimed and claimed != result.boundary_id:
            _raise_contract(
                "content-addressed boundary content identity does not match payload"
            )
        return result


def admit_content_addressed_boundary(
    *,
    role: ArtifactRole | str,
    claimed_cid: str,
    canonical_bytes: bytes,
) -> ContentAddressedBoundaryAdmission:
    """Require canonical bytes plus CID rehash at one content-addressed boundary.

    Failures return a non-admitted result rather than raising, so optional
    store/transport faults never abort collection.  Construction of other
    contracts still raises on malformed caller inputs via
    :class:`ActivationContractError`.
    """

    role_value = (
        role
        if isinstance(role, ArtifactRole)
        else _safe_enum(role, ArtifactRole, field_name="role")
    )
    claimed = _bounded_text(claimed_cid, field_name="claimed_cid")
    if type(canonical_bytes) is not bytes:
        return ContentAddressedBoundaryAdmission(
            role=role_value,
            claimed_cid=claimed,
            admitted=False,
            reason_code="malformed_artifact",
            diagnostics={"stage": "type_check"},
        )
    if not claimed:
        return ContentAddressedBoundaryAdmission(
            role=role_value,
            claimed_cid="",
            admitted=False,
            byte_length=len(canonical_bytes),
            reason_code="malformed_artifact",
            diagnostics={"stage": "missing_claimed_cid"},
        )
    try:
        actual = rehash_retained_canonical_bytes(canonical_bytes)
    except ActivationContractError as exc:
        return ContentAddressedBoundaryAdmission(
            role=role_value,
            claimed_cid=claimed,
            admitted=False,
            byte_length=len(canonical_bytes),
            reason_code="candidate_integrity_failed",
            diagnostics={
                "stage": "rehash",
                "detail": str(exc)[:MAX_DIAGNOSTIC_VALUE_CHARS],
            },
        )
    if actual != claimed:
        return ContentAddressedBoundaryAdmission(
            role=role_value,
            claimed_cid=claimed,
            actual_cid=actual,
            admitted=False,
            byte_length=len(canonical_bytes),
            reason_code="candidate_integrity_failed",
            diagnostics={"stage": "cid_mismatch"},
        )
    return ContentAddressedBoundaryAdmission(
        role=role_value,
        claimed_cid=claimed,
        actual_cid=actual,
        admitted=True,
        byte_length=len(canonical_bytes),
        reason_code="boundary_admitted",
    )


def require_content_addressed_boundary(
    *,
    role: ArtifactRole | str,
    claimed_cid: str,
    canonical_bytes: bytes,
) -> ContentAddressedBoundaryAdmission:
    """Strict form of :func:`admit_content_addressed_boundary` that raises on miss."""

    admission = admit_content_addressed_boundary(
        role=role,
        claimed_cid=claimed_cid,
        canonical_bytes=canonical_bytes,
    )
    if not admission.admitted:
        _raise_contract(
            "content-addressed boundary rejected: %s" % (admission.reason_code or "unknown")
        )
    return admission


# ---------------------------------------------------------------------------
# LocatorHint@1 — mutable retrieval narrowing only
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LocatorHint(CanonicalContract):
    """Non-authoritative mutable index entry pointing at retained candidate bytes.

    A locator hint may find candidates.  It cannot reconstruct what a
    certificate attests and cannot authorize ``SKIP``.
    """

    __test__: ClassVar[bool] = False
    SCHEMA: ClassVar[str] = LOCATOR_HINT_SCHEMA

    locator_cid: str
    candidate_context_cid: str = ""
    certificate_cid: str = ""
    index_generation: int = 0
    repository_id: str = ""
    node_id: str = ""
    selection_semantics: str = "exact_node"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "locator_cid",
            _bounded_text(self.locator_cid, field_name="locator_cid", required=True),
        )
        object.__setattr__(
            self,
            "candidate_context_cid",
            _bounded_text(
                self.candidate_context_cid, field_name="candidate_context_cid"
            ),
        )
        object.__setattr__(
            self,
            "certificate_cid",
            _bounded_text(self.certificate_cid, field_name="certificate_cid"),
        )
        object.__setattr__(
            self,
            "index_generation",
            _safe_nonnegative_int(
                self.index_generation, field_name="index_generation"
            ),
        )
        object.__setattr__(
            self,
            "repository_id",
            _bounded_text(self.repository_id, field_name="repository_id"),
        )
        object.__setattr__(
            self,
            "node_id",
            _bounded_text(
                self.node_id, field_name="node_id", max_chars=MAX_NODE_ID_CHARS
            ),
        )
        object.__setattr__(
            self,
            "selection_semantics",
            _bounded_text(
                self.selection_semantics,
                field_name="selection_semantics",
                required=True,
                max_chars=64,
            ),
        )
        object.__setattr__(
            self, "metadata", _bounded_mapping(self.metadata, field_name="metadata")
        )

    @property
    def interface(self) -> str:
        return LOCATOR_HINT_INTERFACE

    @property
    def role(self) -> ArtifactRole:
        return ArtifactRole.LOCATOR_HINT

    @property
    def hint_id(self) -> str:
        return self.content_id

    @property
    def may_authorize_skip(self) -> bool:
        return False

    def _payload(self) -> Dict[str, Any]:
        return {
            "contract_version": ACTIVATION_CONTRACT_VERSION,
            "interface": LOCATOR_HINT_INTERFACE,
            "role": ArtifactRole.LOCATOR_HINT,
            "locator_cid": self.locator_cid,
            "candidate_context_cid": self.candidate_context_cid,
            "certificate_cid": self.certificate_cid,
            "index_generation": self.index_generation,
            "repository_id": self.repository_id,
            "node_id": self.node_id,
            "selection_semantics": self.selection_semantics,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LocatorHint":
        _schema_or_raise(payload, cls.SCHEMA, artifact_name="locator hint")
        _require_versioned_interface(
            payload,
            expected_interface=LOCATOR_HINT_INTERFACE,
            artifact_name="locator hint",
        )
        _safe_reject_unknown_fields(
            payload,
            {
                "schema",
                "contract_version",
                "interface",
                "role",
                "locator_cid",
                "candidate_context_cid",
                "certificate_cid",
                "index_generation",
                "repository_id",
                "node_id",
                "selection_semantics",
                "metadata",
                "content_id",
                "hint_id",
            },
            artifact_name="locator hint",
        )
        role = payload.get("role")
        if role not in (None, "", ArtifactRole.LOCATOR_HINT, ArtifactRole.LOCATOR_HINT.value):
            _raise_contract("locator hint role must be locator_hint")
        result = cls(
            locator_cid=payload.get("locator_cid", ""),
            candidate_context_cid=payload.get("candidate_context_cid", ""),
            certificate_cid=payload.get("certificate_cid", ""),
            index_generation=payload.get("index_generation", 0),
            repository_id=payload.get("repository_id", ""),
            node_id=payload.get("node_id", ""),
            selection_semantics=payload.get("selection_semantics", "exact_node"),
            metadata=payload.get("metadata") or {},
        )
        claimed = payload.get("hint_id") or payload.get("content_id")
        if claimed and claimed != result.hint_id:
            _raise_contract("locator hint content identity does not match payload")
        return result


# ---------------------------------------------------------------------------
# CandidateExecutionContext@1 — immutable retained pass-time context
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CandidateExecutionContext(CanonicalContract):
    """Immutable canonical candidate context retained after a complete pass.

    Retained fields pin the exact execution key, source/static closure,
    observed runtime trace, forest/environment/policy facts, and pass receipt
    so a later warm run can reconstruct what a certificate attests.  Bytes are
    content-addressed; the mutable locator index is only a hint to these bytes.
    """

    __test__: ClassVar[bool] = False
    SCHEMA: ClassVar[str] = CANDIDATE_EXECUTION_CONTEXT_SCHEMA

    locator_cid: str
    execution_key_cid: str
    pass_receipt_cid: str
    repository_forest_cid: str
    test_ast_cid: str
    static_trace_root_cid: str
    runtime_trace_root_cid: str
    environment_cid: str
    policy_cid: str
    dependency_lock_cid: str = ""
    installed_distributions_cid: str = ""
    platform_cid: str = ""
    capability_root_cid: str = ""
    certificate_cid: str = ""
    component_cids: Mapping[str, str] = field(default_factory=dict)
    external_snapshot_cids: Tuple[str, ...] = ()
    retained_at_ms: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "locator_cid",
            "execution_key_cid",
            "pass_receipt_cid",
            "repository_forest_cid",
            "test_ast_cid",
            "static_trace_root_cid",
            "runtime_trace_root_cid",
            "environment_cid",
            "policy_cid",
        ):
            object.__setattr__(
                self,
                name,
                _bounded_text(getattr(self, name), field_name=name, required=True),
            )
        for name in (
            "dependency_lock_cid",
            "installed_distributions_cid",
            "platform_cid",
            "capability_root_cid",
            "certificate_cid",
        ):
            object.__setattr__(
                self, name, _bounded_text(getattr(self, name), field_name=name)
            )
        object.__setattr__(
            self,
            "component_cids",
            _component_map(self.component_cids, field_name="component_cids"),
        )
        object.__setattr__(
            self,
            "external_snapshot_cids",
            _bounded_ids(
                self.external_snapshot_cids, field_name="external_snapshot_cids"
            ),
        )
        object.__setattr__(
            self,
            "retained_at_ms",
            _safe_nonnegative_int(self.retained_at_ms, field_name="retained_at_ms"),
        )
        object.__setattr__(
            self, "metadata", _bounded_mapping(self.metadata, field_name="metadata")
        )

    @property
    def interface(self) -> str:
        return CANDIDATE_EXECUTION_CONTEXT_INTERFACE

    @property
    def role(self) -> ArtifactRole:
        return ArtifactRole.IMMUTABLE_CANDIDATE_CONTEXT

    @property
    def candidate_context_id(self) -> str:
        return self.content_id

    @property
    def may_authorize_skip(self) -> bool:
        return False

    def comparison_identities(self) -> Dict[str, str]:
        """Identities required for pre-SKIP current-context comparison."""

        return {
            SkipComparisonDimension.AST.value: self.test_ast_cid,
            SkipComparisonDimension.STATIC.value: self.static_trace_root_cid,
            SkipComparisonDimension.RUNTIME.value: self.runtime_trace_root_cid,
            SkipComparisonDimension.ENVIRONMENT.value: self.environment_cid,
            SkipComparisonDimension.POLICY.value: self.policy_cid,
        }

    def _payload(self) -> Dict[str, Any]:
        return {
            "contract_version": ACTIVATION_CONTRACT_VERSION,
            "interface": CANDIDATE_EXECUTION_CONTEXT_INTERFACE,
            "role": ArtifactRole.IMMUTABLE_CANDIDATE_CONTEXT,
            "locator_cid": self.locator_cid,
            "execution_key_cid": self.execution_key_cid,
            "pass_receipt_cid": self.pass_receipt_cid,
            "repository_forest_cid": self.repository_forest_cid,
            "test_ast_cid": self.test_ast_cid,
            "static_trace_root_cid": self.static_trace_root_cid,
            "runtime_trace_root_cid": self.runtime_trace_root_cid,
            "environment_cid": self.environment_cid,
            "policy_cid": self.policy_cid,
            "dependency_lock_cid": self.dependency_lock_cid,
            "installed_distributions_cid": self.installed_distributions_cid,
            "platform_cid": self.platform_cid,
            "capability_root_cid": self.capability_root_cid,
            "certificate_cid": self.certificate_cid,
            "component_cids": dict(self.component_cids),
            "external_snapshot_cids": list(self.external_snapshot_cids),
            "retained_at_ms": self.retained_at_ms,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CandidateExecutionContext":
        _schema_or_raise(
            payload, cls.SCHEMA, artifact_name="candidate execution context"
        )
        _require_versioned_interface(
            payload,
            expected_interface=CANDIDATE_EXECUTION_CONTEXT_INTERFACE,
            artifact_name="candidate execution context",
        )
        _safe_reject_unknown_fields(
            payload,
            {
                "schema",
                "contract_version",
                "interface",
                "role",
                "locator_cid",
                "execution_key_cid",
                "pass_receipt_cid",
                "repository_forest_cid",
                "test_ast_cid",
                "static_trace_root_cid",
                "runtime_trace_root_cid",
                "environment_cid",
                "policy_cid",
                "dependency_lock_cid",
                "installed_distributions_cid",
                "platform_cid",
                "capability_root_cid",
                "certificate_cid",
                "component_cids",
                "external_snapshot_cids",
                "retained_at_ms",
                "metadata",
                "content_id",
                "candidate_context_id",
            },
            artifact_name="candidate execution context",
        )
        role = payload.get("role")
        if role not in (
            None,
            "",
            ArtifactRole.IMMUTABLE_CANDIDATE_CONTEXT,
            ArtifactRole.IMMUTABLE_CANDIDATE_CONTEXT.value,
        ):
            _raise_contract(
                "candidate execution context role must be immutable_candidate_context"
            )
        result = cls(
            locator_cid=payload.get("locator_cid", ""),
            execution_key_cid=payload.get("execution_key_cid", ""),
            pass_receipt_cid=payload.get("pass_receipt_cid", ""),
            repository_forest_cid=payload.get("repository_forest_cid", ""),
            test_ast_cid=payload.get("test_ast_cid", ""),
            static_trace_root_cid=payload.get("static_trace_root_cid", ""),
            runtime_trace_root_cid=payload.get("runtime_trace_root_cid", ""),
            environment_cid=payload.get("environment_cid", ""),
            policy_cid=payload.get("policy_cid", ""),
            dependency_lock_cid=payload.get("dependency_lock_cid", ""),
            installed_distributions_cid=payload.get(
                "installed_distributions_cid", ""
            ),
            platform_cid=payload.get("platform_cid", ""),
            capability_root_cid=payload.get("capability_root_cid", ""),
            certificate_cid=payload.get("certificate_cid", ""),
            component_cids=payload.get("component_cids") or {},
            external_snapshot_cids=tuple(
                payload.get("external_snapshot_cids") or ()
            ),
            retained_at_ms=payload.get("retained_at_ms", 0),
            metadata=payload.get("metadata") or {},
        )
        claimed = payload.get("candidate_context_id") or payload.get("content_id")
        if claimed and claimed != result.candidate_context_id:
            _raise_contract(
                "candidate execution context content identity does not match payload"
            )
        return result


# ---------------------------------------------------------------------------
# CurrentExecutionContext@1 — freshly rebuilt live context
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CurrentExecutionContext(CanonicalContract):
    """Freshly rebuilt current execution context for warm admission comparison.

    Historical runtime traces may name a dependency frontier to resolve, but
    this record must be produced from live source, AST, fixtures, locks,
    environment, capabilities, and policy.  Relabeling a prior trace as current
    is forbidden by construction (``rebuild_source`` is closed).
    """

    __test__: ClassVar[bool] = False
    SCHEMA: ClassVar[str] = CURRENT_EXECUTION_CONTEXT_SCHEMA

    locator_cid: str
    execution_key_cid: str
    repository_forest_cid: str
    test_ast_cid: str
    static_trace_root_cid: str
    runtime_trace_root_cid: str
    environment_cid: str
    policy_cid: str
    dependency_lock_cid: str = ""
    installed_distributions_cid: str = ""
    platform_cid: str = ""
    capability_root_cid: str = ""
    component_cids: Mapping[str, str] = field(default_factory=dict)
    external_snapshot_cids: Tuple[str, ...] = ()
    rebuild_source: str = "fresh_live_rebuild"
    rebuilt_at_ms: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    _ALLOWED_REBUILD_SOURCES: ClassVar[frozenset[str]] = frozenset(
        {
            "fresh_live_rebuild",
            "controlled_preflight",
        }
    )

    def __post_init__(self) -> None:
        for name in (
            "locator_cid",
            "execution_key_cid",
            "repository_forest_cid",
            "test_ast_cid",
            "static_trace_root_cid",
            "runtime_trace_root_cid",
            "environment_cid",
            "policy_cid",
        ):
            object.__setattr__(
                self,
                name,
                _bounded_text(getattr(self, name), field_name=name, required=True),
            )
        for name in (
            "dependency_lock_cid",
            "installed_distributions_cid",
            "platform_cid",
            "capability_root_cid",
        ):
            object.__setattr__(
                self, name, _bounded_text(getattr(self, name), field_name=name)
            )
        object.__setattr__(
            self,
            "component_cids",
            _component_map(self.component_cids, field_name="component_cids"),
        )
        object.__setattr__(
            self,
            "external_snapshot_cids",
            _bounded_ids(
                self.external_snapshot_cids, field_name="external_snapshot_cids"
            ),
        )
        rebuild_source = _bounded_text(
            self.rebuild_source,
            field_name="rebuild_source",
            required=True,
            max_chars=64,
        )
        if rebuild_source not in self._ALLOWED_REBUILD_SOURCES:
            _raise_contract(
                "current execution context rebuild_source must be a fresh source; "
                "historical traces cannot be relabeled as current"
            )
        object.__setattr__(self, "rebuild_source", rebuild_source)
        object.__setattr__(
            self,
            "rebuilt_at_ms",
            _safe_nonnegative_int(self.rebuilt_at_ms, field_name="rebuilt_at_ms"),
        )
        object.__setattr__(
            self, "metadata", _bounded_mapping(self.metadata, field_name="metadata")
        )

    @property
    def interface(self) -> str:
        return CURRENT_EXECUTION_CONTEXT_INTERFACE

    @property
    def role(self) -> ArtifactRole:
        return ArtifactRole.FRESH_CURRENT_CONTEXT

    @property
    def current_context_id(self) -> str:
        return self.content_id

    @property
    def may_authorize_skip(self) -> bool:
        return False

    def comparison_identities(self) -> Dict[str, str]:
        return {
            SkipComparisonDimension.AST.value: self.test_ast_cid,
            SkipComparisonDimension.STATIC.value: self.static_trace_root_cid,
            SkipComparisonDimension.RUNTIME.value: self.runtime_trace_root_cid,
            SkipComparisonDimension.ENVIRONMENT.value: self.environment_cid,
            SkipComparisonDimension.POLICY.value: self.policy_cid,
        }

    def _payload(self) -> Dict[str, Any]:
        return {
            "contract_version": ACTIVATION_CONTRACT_VERSION,
            "interface": CURRENT_EXECUTION_CONTEXT_INTERFACE,
            "role": ArtifactRole.FRESH_CURRENT_CONTEXT,
            "locator_cid": self.locator_cid,
            "execution_key_cid": self.execution_key_cid,
            "repository_forest_cid": self.repository_forest_cid,
            "test_ast_cid": self.test_ast_cid,
            "static_trace_root_cid": self.static_trace_root_cid,
            "runtime_trace_root_cid": self.runtime_trace_root_cid,
            "environment_cid": self.environment_cid,
            "policy_cid": self.policy_cid,
            "dependency_lock_cid": self.dependency_lock_cid,
            "installed_distributions_cid": self.installed_distributions_cid,
            "platform_cid": self.platform_cid,
            "capability_root_cid": self.capability_root_cid,
            "component_cids": dict(self.component_cids),
            "external_snapshot_cids": list(self.external_snapshot_cids),
            "rebuild_source": self.rebuild_source,
            "rebuilt_at_ms": self.rebuilt_at_ms,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CurrentExecutionContext":
        _schema_or_raise(
            payload, cls.SCHEMA, artifact_name="current execution context"
        )
        _require_versioned_interface(
            payload,
            expected_interface=CURRENT_EXECUTION_CONTEXT_INTERFACE,
            artifact_name="current execution context",
        )
        _safe_reject_unknown_fields(
            payload,
            {
                "schema",
                "contract_version",
                "interface",
                "role",
                "locator_cid",
                "execution_key_cid",
                "repository_forest_cid",
                "test_ast_cid",
                "static_trace_root_cid",
                "runtime_trace_root_cid",
                "environment_cid",
                "policy_cid",
                "dependency_lock_cid",
                "installed_distributions_cid",
                "platform_cid",
                "capability_root_cid",
                "component_cids",
                "external_snapshot_cids",
                "rebuild_source",
                "rebuilt_at_ms",
                "metadata",
                "content_id",
                "current_context_id",
            },
            artifact_name="current execution context",
        )
        role = payload.get("role")
        if role not in (
            None,
            "",
            ArtifactRole.FRESH_CURRENT_CONTEXT,
            ArtifactRole.FRESH_CURRENT_CONTEXT.value,
        ):
            _raise_contract(
                "current execution context role must be fresh_current_context"
            )
        result = cls(
            locator_cid=payload.get("locator_cid", ""),
            execution_key_cid=payload.get("execution_key_cid", ""),
            repository_forest_cid=payload.get("repository_forest_cid", ""),
            test_ast_cid=payload.get("test_ast_cid", ""),
            static_trace_root_cid=payload.get("static_trace_root_cid", ""),
            runtime_trace_root_cid=payload.get("runtime_trace_root_cid", ""),
            environment_cid=payload.get("environment_cid", ""),
            policy_cid=payload.get("policy_cid", ""),
            dependency_lock_cid=payload.get("dependency_lock_cid", ""),
            installed_distributions_cid=payload.get(
                "installed_distributions_cid", ""
            ),
            platform_cid=payload.get("platform_cid", ""),
            capability_root_cid=payload.get("capability_root_cid", ""),
            component_cids=payload.get("component_cids") or {},
            external_snapshot_cids=tuple(
                payload.get("external_snapshot_cids") or ()
            ),
            rebuild_source=payload.get("rebuild_source", "fresh_live_rebuild"),
            rebuilt_at_ms=payload.get("rebuilt_at_ms", 0),
            metadata=payload.get("metadata") or {},
        )
        claimed = payload.get("current_context_id") or payload.get("content_id")
        if claimed and claimed != result.current_context_id:
            _raise_contract(
                "current execution context content identity does not match payload"
            )
        return result


# ---------------------------------------------------------------------------
# Trusted pass receipt / authoritative certificate bindings
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrustedPassReceiptBinding(CanonicalContract):
    """Activation-boundary binding to an admitted complete-pass receipt.

    The receipt itself is defined by ``TestPassReceipt@1``.  This binding pins
    the receipt CID into the activation lifecycle and records that it is not
    skip authority by itself.
    """

    __test__: ClassVar[bool] = False
    SCHEMA: ClassVar[str] = TRUSTED_PASS_RECEIPT_BINDING_SCHEMA

    receipt_cid: str
    execution_key_cid: str
    locator_cid: str
    candidate_context_cid: str = ""
    runtime_trace_root_cid: str = ""
    admitted: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("receipt_cid", "execution_key_cid", "locator_cid"):
            object.__setattr__(
                self,
                name,
                _bounded_text(getattr(self, name), field_name=name, required=True),
            )
        for name in ("candidate_context_cid", "runtime_trace_root_cid"):
            object.__setattr__(
                self, name, _bounded_text(getattr(self, name), field_name=name)
            )
        object.__setattr__(
            self, "admitted", _bool(self.admitted, field_name="admitted")
        )
        object.__setattr__(
            self, "metadata", _bounded_mapping(self.metadata, field_name="metadata")
        )

    @property
    def interface(self) -> str:
        return TRUSTED_PASS_RECEIPT_BINDING_INTERFACE

    @property
    def role(self) -> ArtifactRole:
        return ArtifactRole.TRUSTED_PASS_RECEIPT

    @property
    def binding_id(self) -> str:
        return self.content_id

    @property
    def may_authorize_skip(self) -> bool:
        return False

    def _payload(self) -> Dict[str, Any]:
        return {
            "contract_version": ACTIVATION_CONTRACT_VERSION,
            "interface": TRUSTED_PASS_RECEIPT_BINDING_INTERFACE,
            "role": ArtifactRole.TRUSTED_PASS_RECEIPT,
            "receipt_cid": self.receipt_cid,
            "execution_key_cid": self.execution_key_cid,
            "locator_cid": self.locator_cid,
            "candidate_context_cid": self.candidate_context_cid,
            "runtime_trace_root_cid": self.runtime_trace_root_cid,
            "admitted": self.admitted,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TrustedPassReceiptBinding":
        _schema_or_raise(
            payload, cls.SCHEMA, artifact_name="trusted pass receipt binding"
        )
        _require_versioned_interface(
            payload,
            expected_interface=TRUSTED_PASS_RECEIPT_BINDING_INTERFACE,
            artifact_name="trusted pass receipt binding",
        )
        _safe_reject_unknown_fields(
            payload,
            {
                "schema",
                "contract_version",
                "interface",
                "role",
                "receipt_cid",
                "execution_key_cid",
                "locator_cid",
                "candidate_context_cid",
                "runtime_trace_root_cid",
                "admitted",
                "metadata",
                "content_id",
                "binding_id",
            },
            artifact_name="trusted pass receipt binding",
        )
        result = cls(
            receipt_cid=payload.get("receipt_cid", ""),
            execution_key_cid=payload.get("execution_key_cid", ""),
            locator_cid=payload.get("locator_cid", ""),
            candidate_context_cid=payload.get("candidate_context_cid", ""),
            runtime_trace_root_cid=payload.get("runtime_trace_root_cid", ""),
            admitted=payload.get("admitted", True),
            metadata=payload.get("metadata") or {},
        )
        claimed = payload.get("binding_id") or payload.get("content_id")
        if claimed and claimed != result.binding_id:
            _raise_contract(
                "trusted pass receipt binding content identity does not match payload"
            )
        return result


@dataclass(frozen=True)
class AuthoritativeCertificateBinding(CanonicalContract):
    """Activation-boundary binding to a locally verifiable authoritative certificate.

    Only this role may authorize ``SKIP``, and only after exact current-context
    comparison and local cryptographic verification.  Simulated / non-attested
    certificates must set ``authoritative=False`` and cannot skip.
    """

    __test__: ClassVar[bool] = False
    SCHEMA: ClassVar[str] = AUTHORITATIVE_CERTIFICATE_BINDING_SCHEMA

    certificate_cid: str
    receipt_cid: str
    execution_key_cid: str
    candidate_context_cid: str
    statement_cid: str
    circuit_cid: str
    verifying_key_cid: str
    policy_cid: str
    issuer_id: str
    epoch: str = ""
    authoritative: bool = True
    simulated: bool = False
    locally_verified: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "certificate_cid",
            "receipt_cid",
            "execution_key_cid",
            "candidate_context_cid",
            "statement_cid",
            "circuit_cid",
            "verifying_key_cid",
            "policy_cid",
            "issuer_id",
        ):
            object.__setattr__(
                self,
                name,
                _bounded_text(getattr(self, name), field_name=name, required=True),
            )
        object.__setattr__(
            self, "epoch", _bounded_text(self.epoch, field_name="epoch")
        )
        object.__setattr__(
            self, "authoritative", _bool(self.authoritative, field_name="authoritative")
        )
        object.__setattr__(
            self, "simulated", _bool(self.simulated, field_name="simulated")
        )
        object.__setattr__(
            self,
            "locally_verified",
            _bool(self.locally_verified, field_name="locally_verified"),
        )
        object.__setattr__(
            self, "metadata", _bounded_mapping(self.metadata, field_name="metadata")
        )
        if self.simulated and self.authoritative:
            _raise_contract(
                "illegal-authority: simulated certificate cannot be authoritative"
            )
        if self.simulated:
            object.__setattr__(self, "authoritative", False)

    @property
    def interface(self) -> str:
        return AUTHORITATIVE_CERTIFICATE_BINDING_INTERFACE

    @property
    def role(self) -> ArtifactRole:
        return ArtifactRole.AUTHORITATIVE_CERTIFICATE

    @property
    def binding_id(self) -> str:
        return self.content_id

    @property
    def may_authorize_skip(self) -> bool:
        return (
            self.authoritative
            and not self.simulated
            and self.locally_verified
        )

    def _payload(self) -> Dict[str, Any]:
        return {
            "contract_version": ACTIVATION_CONTRACT_VERSION,
            "interface": AUTHORITATIVE_CERTIFICATE_BINDING_INTERFACE,
            "role": ArtifactRole.AUTHORITATIVE_CERTIFICATE,
            "certificate_cid": self.certificate_cid,
            "receipt_cid": self.receipt_cid,
            "execution_key_cid": self.execution_key_cid,
            "candidate_context_cid": self.candidate_context_cid,
            "statement_cid": self.statement_cid,
            "circuit_cid": self.circuit_cid,
            "verifying_key_cid": self.verifying_key_cid,
            "policy_cid": self.policy_cid,
            "issuer_id": self.issuer_id,
            "epoch": self.epoch,
            "authoritative": self.authoritative,
            "simulated": self.simulated,
            "locally_verified": self.locally_verified,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "AuthoritativeCertificateBinding":
        _schema_or_raise(
            payload,
            cls.SCHEMA,
            artifact_name="authoritative certificate binding",
        )
        _require_versioned_interface(
            payload,
            expected_interface=AUTHORITATIVE_CERTIFICATE_BINDING_INTERFACE,
            artifact_name="authoritative certificate binding",
        )
        _safe_reject_unknown_fields(
            payload,
            {
                "schema",
                "contract_version",
                "interface",
                "role",
                "certificate_cid",
                "receipt_cid",
                "execution_key_cid",
                "candidate_context_cid",
                "statement_cid",
                "circuit_cid",
                "verifying_key_cid",
                "policy_cid",
                "issuer_id",
                "epoch",
                "authoritative",
                "simulated",
                "locally_verified",
                "metadata",
                "content_id",
                "binding_id",
            },
            artifact_name="authoritative certificate binding",
        )
        result = cls(
            certificate_cid=payload.get("certificate_cid", ""),
            receipt_cid=payload.get("receipt_cid", ""),
            execution_key_cid=payload.get("execution_key_cid", ""),
            candidate_context_cid=payload.get("candidate_context_cid", ""),
            statement_cid=payload.get("statement_cid", ""),
            circuit_cid=payload.get("circuit_cid", ""),
            verifying_key_cid=payload.get("verifying_key_cid", ""),
            policy_cid=payload.get("policy_cid", ""),
            issuer_id=payload.get("issuer_id", ""),
            epoch=payload.get("epoch", ""),
            authoritative=payload.get("authoritative", True),
            simulated=payload.get("simulated", False),
            locally_verified=payload.get("locally_verified", False),
            metadata=payload.get("metadata") or {},
        )
        claimed = payload.get("binding_id") or payload.get("content_id")
        if claimed and claimed != result.binding_id:
            _raise_contract(
                "authoritative certificate binding content identity does not match payload"
            )
        return result


# ---------------------------------------------------------------------------
# DeferredProofRequest@1 — public-only deferred issuance envelope
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DeferredProofRequest(CanonicalContract):
    """Public-only deferred certificate issuance request.

    Workers may transport this envelope.  Private witness material is forbidden.
    Missing packages, keys, circuits, endpoints, binaries, caches, or transports
    retain the pass receipt and surface a typed ``DEFERRED``/``RUN`` result.
    """

    __test__: ClassVar[bool] = False
    SCHEMA: ClassVar[str] = DEFERRED_PROOF_REQUEST_SCHEMA

    receipt_cid: str
    execution_key_cid: str
    candidate_context_cid: str
    statement_cid: str
    policy_cid: str
    circuit_cid: str
    verifying_key_cid: str
    issuer_id: str
    epoch: str = ""
    locator_cid: str = ""
    public_inputs: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "receipt_cid",
            "execution_key_cid",
            "candidate_context_cid",
            "statement_cid",
            "policy_cid",
            "circuit_cid",
            "verifying_key_cid",
            "issuer_id",
        ):
            object.__setattr__(
                self,
                name,
                _bounded_text(getattr(self, name), field_name=name, required=True),
            )
        for name in ("epoch", "locator_cid"):
            object.__setattr__(
                self, name, _bounded_text(getattr(self, name), field_name=name)
            )
        object.__setattr__(
            self,
            "public_inputs",
            _bounded_mapping(
                self.public_inputs,
                field_name="public_inputs",
                max_keys=MAX_COMPONENT_ENTRIES,
                max_value_chars=MAX_TEXT_CHARS,
            ),
        )
        object.__setattr__(
            self, "metadata", _bounded_mapping(self.metadata, field_name="metadata")
        )

    @property
    def interface(self) -> str:
        return DEFERRED_PROOF_REQUEST_INTERFACE

    @property
    def role(self) -> ArtifactRole:
        return ArtifactRole.DEFERRED_PROOF_REQUEST

    @property
    def request_id(self) -> str:
        return self.content_id

    @property
    def may_authorize_skip(self) -> bool:
        return False

    def _payload(self) -> Dict[str, Any]:
        return {
            "contract_version": ACTIVATION_CONTRACT_VERSION,
            "interface": DEFERRED_PROOF_REQUEST_INTERFACE,
            "role": ArtifactRole.DEFERRED_PROOF_REQUEST,
            "receipt_cid": self.receipt_cid,
            "execution_key_cid": self.execution_key_cid,
            "candidate_context_cid": self.candidate_context_cid,
            "statement_cid": self.statement_cid,
            "policy_cid": self.policy_cid,
            "circuit_cid": self.circuit_cid,
            "verifying_key_cid": self.verifying_key_cid,
            "issuer_id": self.issuer_id,
            "epoch": self.epoch,
            "locator_cid": self.locator_cid,
            "public_inputs": dict(self.public_inputs),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DeferredProofRequest":
        _schema_or_raise(payload, cls.SCHEMA, artifact_name="deferred proof request")
        _require_versioned_interface(
            payload,
            expected_interface=DEFERRED_PROOF_REQUEST_INTERFACE,
            artifact_name="deferred proof request",
        )
        _safe_reject_unknown_fields(
            payload,
            {
                "schema",
                "contract_version",
                "interface",
                "role",
                "receipt_cid",
                "execution_key_cid",
                "candidate_context_cid",
                "statement_cid",
                "policy_cid",
                "circuit_cid",
                "verifying_key_cid",
                "issuer_id",
                "epoch",
                "locator_cid",
                "public_inputs",
                "metadata",
                "content_id",
                "request_id",
            },
            artifact_name="deferred proof request",
        )
        result = cls(
            receipt_cid=payload.get("receipt_cid", ""),
            execution_key_cid=payload.get("execution_key_cid", ""),
            candidate_context_cid=payload.get("candidate_context_cid", ""),
            statement_cid=payload.get("statement_cid", ""),
            policy_cid=payload.get("policy_cid", ""),
            circuit_cid=payload.get("circuit_cid", ""),
            verifying_key_cid=payload.get("verifying_key_cid", ""),
            issuer_id=payload.get("issuer_id", ""),
            epoch=payload.get("epoch", ""),
            locator_cid=payload.get("locator_cid", ""),
            public_inputs=payload.get("public_inputs") or {},
            metadata=payload.get("metadata") or {},
        )
        claimed = payload.get("request_id") or payload.get("content_id")
        if claimed and claimed != result.request_id:
            _raise_contract(
                "deferred proof request content identity does not match payload"
            )
        return result


# ---------------------------------------------------------------------------
# Context comparison before SKIP
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ContextComparisonResult:
    """Exact comparison of candidate vs current activation dimensions."""

    __test__: ClassVar[bool] = False

    matched: bool
    mismatched_dimensions: Tuple[str, ...] = ()
    missing_dimensions: Tuple[str, ...] = ()
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "matched", bool(self.matched))
        object.__setattr__(
            self,
            "mismatched_dimensions",
            tuple(str(item) for item in self.mismatched_dimensions),
        )
        object.__setattr__(
            self,
            "missing_dimensions",
            tuple(str(item) for item in self.missing_dimensions),
        )
        object.__setattr__(
            self,
            "diagnostics",
            dict(self.diagnostics) if self.diagnostics else {},
        )


def compare_contexts_for_skip(
    candidate: CandidateExecutionContext,
    current: CurrentExecutionContext,
) -> ContextComparisonResult:
    """Require exact AST/static/runtime/environment/policy agreement before SKIP.

    Additional execution-key / forest agreement is also required.  A mismatch
    or incomplete dimension is a hard ``RUN``; it never becomes skip authority.
    """

    if not isinstance(candidate, CandidateExecutionContext):
        return ContextComparisonResult(
            matched=False,
            missing_dimensions=tuple(
                dim.value for dim in SKIP_REQUIRED_COMPARISON_DIMENSIONS
            ),
            diagnostics={"stage": "candidate_type"},
        )
    if not isinstance(current, CurrentExecutionContext):
        return ContextComparisonResult(
            matched=False,
            missing_dimensions=tuple(
                dim.value for dim in SKIP_REQUIRED_COMPARISON_DIMENSIONS
            ),
            diagnostics={"stage": "current_type"},
        )

    mismatched: list[str] = []
    missing: list[str] = []

    if candidate.locator_cid != current.locator_cid:
        mismatched.append("locator")
    if candidate.execution_key_cid != current.execution_key_cid:
        mismatched.append("execution_key")
    if candidate.repository_forest_cid != current.repository_forest_cid:
        mismatched.append("repository_forest")

    candidate_ids = candidate.comparison_identities()
    current_ids = current.comparison_identities()
    for dimension in SKIP_REQUIRED_COMPARISON_DIMENSIONS:
        key = dimension.value
        left = candidate_ids.get(key, "")
        right = current_ids.get(key, "")
        if not left or not right:
            missing.append(key)
        elif left != right:
            mismatched.append(key)

    matched = not mismatched and not missing
    return ContextComparisonResult(
        matched=matched,
        mismatched_dimensions=tuple(mismatched),
        missing_dimensions=tuple(missing),
        diagnostics={
            "required_dimensions": [
                dim.value for dim in SKIP_REQUIRED_COMPARISON_DIMENSIONS
            ]
        },
    )


# ---------------------------------------------------------------------------
# Post-pass runtime observations (no duplicate test call)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PostPassRuntimeObservation(CanonicalContract):
    """Runtime frontier recorded after one real setup/call/teardown pass.

    Capture is post-pass only.  ``test_call_count`` must be exactly 1 for the
    lifecycle that produced this observation; the contract forbids recording
    by re-invoking the test body.
    """

    __test__: ClassVar[bool] = False
    SCHEMA: ClassVar[str] = POST_PASS_RUNTIME_OBSERVATION_SCHEMA

    locator_cid: str
    execution_key_cid: str
    runtime_trace_root_cid: str
    pass_receipt_cid: str
    test_call_count: int = 1
    setup_call_count: int = 1
    teardown_call_count: int = 1
    duplicate_test_call_forbidden: bool = True
    observed_at_ms: int = 0
    observation_source: str = "post_pass_lifecycle"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "locator_cid",
            "execution_key_cid",
            "runtime_trace_root_cid",
            "pass_receipt_cid",
        ):
            object.__setattr__(
                self,
                name,
                _bounded_text(getattr(self, name), field_name=name, required=True),
            )
        for name in (
            "test_call_count",
            "setup_call_count",
            "teardown_call_count",
            "observed_at_ms",
        ):
            object.__setattr__(
                self,
                name,
                _safe_nonnegative_int(getattr(self, name), field_name=name),
            )
        object.__setattr__(
            self,
            "duplicate_test_call_forbidden",
            _bool(
                self.duplicate_test_call_forbidden,
                field_name="duplicate_test_call_forbidden",
            ),
        )
        object.__setattr__(
            self,
            "observation_source",
            _bounded_text(
                self.observation_source,
                field_name="observation_source",
                required=True,
                max_chars=64,
            ),
        )
        object.__setattr__(
            self, "metadata", _bounded_mapping(self.metadata, field_name="metadata")
        )
        if self.observation_source != "post_pass_lifecycle":
            _raise_contract(
                "runtime observations must be captured post-pass; "
                "pre-execution prediction is forbidden"
            )
        if not self.duplicate_test_call_forbidden:
            _raise_contract(
                "post-pass observation must forbid duplicate test call capture"
            )
        if self.test_call_count != 1:
            _raise_contract(
                "post-pass observation requires exactly one test call "
                "(no duplicate body execution)"
            )
        if self.setup_call_count != 1 or self.teardown_call_count != 1:
            _raise_contract(
                "post-pass observation requires exactly one setup and teardown"
            )

    @property
    def interface(self) -> str:
        return POST_PASS_RUNTIME_OBSERVATION_INTERFACE

    @property
    def observation_id(self) -> str:
        return self.content_id

    def _payload(self) -> Dict[str, Any]:
        return {
            "contract_version": ACTIVATION_CONTRACT_VERSION,
            "interface": POST_PASS_RUNTIME_OBSERVATION_INTERFACE,
            "locator_cid": self.locator_cid,
            "execution_key_cid": self.execution_key_cid,
            "runtime_trace_root_cid": self.runtime_trace_root_cid,
            "pass_receipt_cid": self.pass_receipt_cid,
            "test_call_count": self.test_call_count,
            "setup_call_count": self.setup_call_count,
            "teardown_call_count": self.teardown_call_count,
            "duplicate_test_call_forbidden": self.duplicate_test_call_forbidden,
            "observed_at_ms": self.observed_at_ms,
            "observation_source": self.observation_source,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PostPassRuntimeObservation":
        _schema_or_raise(
            payload, cls.SCHEMA, artifact_name="post-pass runtime observation"
        )
        _require_versioned_interface(
            payload,
            expected_interface=POST_PASS_RUNTIME_OBSERVATION_INTERFACE,
            artifact_name="post-pass runtime observation",
        )
        _safe_reject_unknown_fields(
            payload,
            {
                "schema",
                "contract_version",
                "interface",
                "locator_cid",
                "execution_key_cid",
                "runtime_trace_root_cid",
                "pass_receipt_cid",
                "test_call_count",
                "setup_call_count",
                "teardown_call_count",
                "duplicate_test_call_forbidden",
                "observed_at_ms",
                "observation_source",
                "metadata",
                "content_id",
                "observation_id",
            },
            artifact_name="post-pass runtime observation",
        )
        result = cls(
            locator_cid=payload.get("locator_cid", ""),
            execution_key_cid=payload.get("execution_key_cid", ""),
            runtime_trace_root_cid=payload.get("runtime_trace_root_cid", ""),
            pass_receipt_cid=payload.get("pass_receipt_cid", ""),
            test_call_count=payload.get("test_call_count", 1),
            setup_call_count=payload.get("setup_call_count", 1),
            teardown_call_count=payload.get("teardown_call_count", 1),
            duplicate_test_call_forbidden=payload.get(
                "duplicate_test_call_forbidden", True
            ),
            observed_at_ms=payload.get("observed_at_ms", 0),
            observation_source=payload.get(
                "observation_source", "post_pass_lifecycle"
            ),
            metadata=payload.get("metadata") or {},
        )
        claimed = payload.get("observation_id") or payload.get("content_id")
        if claimed and claimed != result.observation_id:
            _raise_contract(
                "post-pass runtime observation content identity does not match payload"
            )
        return result


def record_post_pass_runtime_observation(
    *,
    locator_cid: str,
    execution_key_cid: str,
    runtime_trace_root_cid: str,
    pass_receipt_cid: str,
    test_call_count: int = 1,
    setup_call_count: int = 1,
    teardown_call_count: int = 1,
    observed_at_ms: int = 0,
    metadata: Mapping[str, Any] | None = None,
) -> PostPassRuntimeObservation:
    """Build a post-pass observation after the single real test lifecycle.

    This helper never invokes the test body.  Callers must pass the observed
    call counts from the completed lifecycle; values other than one call each
    for setup/call/teardown are rejected.
    """

    return PostPassRuntimeObservation(
        locator_cid=locator_cid,
        execution_key_cid=execution_key_cid,
        runtime_trace_root_cid=runtime_trace_root_cid,
        pass_receipt_cid=pass_receipt_cid,
        test_call_count=test_call_count,
        setup_call_count=setup_call_count,
        teardown_call_count=teardown_call_count,
        duplicate_test_call_forbidden=True,
        observed_at_ms=observed_at_ms,
        observation_source="post_pass_lifecycle",
        metadata=metadata or {},
    )


# ---------------------------------------------------------------------------
# RuntimeReuseDisposition@1
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RuntimeReuseDisposition(CanonicalContract):
    """Typed activation disposition: ``RUN``, ``SKIP``, or ``DEFERRED``.

    Collection must never fail because of an optional capability fault.  Valid
    dispositions always set ``collection_failed=False``.  ``SKIP`` is only
    constructible when every pre-skip gate is satisfied.
    """

    __test__: ClassVar[bool] = False
    SCHEMA: ClassVar[str] = RUNTIME_REUSE_DISPOSITION_SCHEMA

    action: RuntimeReuseAction
    reason_code: str
    artifact_role: ArtifactRole | None = None
    certificate_cid: str = ""
    receipt_cid: str = ""
    candidate_context_cid: str = ""
    collection_failed: bool = False
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "action",
            _safe_enum(self.action, RuntimeReuseAction, field_name="action"),
        )
        object.__setattr__(
            self,
            "reason_code",
            _bounded_text(
                self.reason_code,
                field_name="reason_code",
                required=True,
                max_chars=MAX_REASON_CHARS,
            ),
        )
        if self.artifact_role is not None:
            object.__setattr__(
                self,
                "artifact_role",
                _safe_enum(
                    self.artifact_role, ArtifactRole, field_name="artifact_role"
                ),
            )
        for name in ("certificate_cid", "receipt_cid", "candidate_context_cid"):
            object.__setattr__(
                self, name, _bounded_text(getattr(self, name), field_name=name)
            )
        object.__setattr__(
            self,
            "collection_failed",
            _bool(self.collection_failed, field_name="collection_failed"),
        )
        object.__setattr__(
            self, "diagnostics", _diagnostics(self.diagnostics)
        )
        if self.collection_failed:
            _raise_contract(
                "activation disposition must never fail collection; "
                "map optional capability faults to RUN or DEFERRED"
            )
        if self.action is RuntimeReuseAction.SKIP:
            if not self.certificate_cid:
                _raise_contract("SKIP disposition requires certificate_cid")
            if not self.receipt_cid:
                _raise_contract("SKIP disposition requires receipt_cid")
            if self.artifact_role not in (
                None,
                ArtifactRole.AUTHORITATIVE_CERTIFICATE,
            ):
                _raise_contract(
                    "SKIP disposition may only bind authoritative_certificate role"
                )
            object.__setattr__(
                self, "artifact_role", ArtifactRole.AUTHORITATIVE_CERTIFICATE
            )
        if self.action is RuntimeReuseAction.DEFERRED:
            if not self.receipt_cid:
                _raise_contract(
                    "DEFERRED disposition retains a trusted pass receipt_cid"
                )

    @property
    def interface(self) -> str:
        return RUNTIME_REUSE_DISPOSITION_INTERFACE

    @property
    def disposition_id(self) -> str:
        return self.content_id

    @property
    def is_run(self) -> bool:
        return self.action is RuntimeReuseAction.RUN

    @property
    def is_skip(self) -> bool:
        return self.action is RuntimeReuseAction.SKIP

    @property
    def is_deferred(self) -> bool:
        return self.action is RuntimeReuseAction.DEFERRED

    def _payload(self) -> Dict[str, Any]:
        return {
            "contract_version": ACTIVATION_CONTRACT_VERSION,
            "interface": RUNTIME_REUSE_DISPOSITION_INTERFACE,
            "action": self.action,
            "reason_code": self.reason_code,
            "artifact_role": (
                self.artifact_role if self.artifact_role is not None else ""
            ),
            "certificate_cid": self.certificate_cid,
            "receipt_cid": self.receipt_cid,
            "candidate_context_cid": self.candidate_context_cid,
            "collection_failed": self.collection_failed,
            "diagnostics": dict(self.diagnostics),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RuntimeReuseDisposition":
        _schema_or_raise(
            payload, cls.SCHEMA, artifact_name="runtime reuse disposition"
        )
        _require_versioned_interface(
            payload,
            expected_interface=RUNTIME_REUSE_DISPOSITION_INTERFACE,
            artifact_name="runtime reuse disposition",
        )
        _safe_reject_unknown_fields(
            payload,
            {
                "schema",
                "contract_version",
                "interface",
                "action",
                "reason_code",
                "artifact_role",
                "certificate_cid",
                "receipt_cid",
                "candidate_context_cid",
                "collection_failed",
                "diagnostics",
                "content_id",
                "disposition_id",
            },
            artifact_name="runtime reuse disposition",
        )
        role_raw = payload.get("artifact_role")
        artifact_role: ArtifactRole | None
        if role_raw in (None, ""):
            artifact_role = None
        else:
            artifact_role = _safe_enum(
                role_raw, ArtifactRole, field_name="artifact_role"
            )
        result = cls(
            action=payload.get("action", RuntimeReuseAction.RUN),
            reason_code=payload.get("reason_code", "unknown"),
            artifact_role=artifact_role,
            certificate_cid=payload.get("certificate_cid", ""),
            receipt_cid=payload.get("receipt_cid", ""),
            candidate_context_cid=payload.get("candidate_context_cid", ""),
            collection_failed=payload.get("collection_failed", False),
            diagnostics=payload.get("diagnostics") or {},
        )
        claimed = payload.get("disposition_id") or payload.get("content_id")
        if claimed and claimed != result.disposition_id:
            _raise_contract(
                "runtime reuse disposition content identity does not match payload"
            )
        return result


def disposition_run(
    reason_code: str,
    *,
    artifact_role: ArtifactRole | None = None,
    diagnostics: Mapping[str, Any] | None = None,
    receipt_cid: str = "",
    candidate_context_cid: str = "",
) -> RuntimeReuseDisposition:
    return RuntimeReuseDisposition(
        action=RuntimeReuseAction.RUN,
        reason_code=reason_code,
        artifact_role=artifact_role,
        receipt_cid=receipt_cid,
        candidate_context_cid=candidate_context_cid,
        collection_failed=False,
        diagnostics=diagnostics or {},
    )


def disposition_deferred(
    reason_code: str,
    *,
    receipt_cid: str,
    candidate_context_cid: str = "",
    diagnostics: Mapping[str, Any] | None = None,
) -> RuntimeReuseDisposition:
    return RuntimeReuseDisposition(
        action=RuntimeReuseAction.DEFERRED,
        reason_code=reason_code,
        artifact_role=ArtifactRole.DEFERRED_PROOF_REQUEST,
        receipt_cid=receipt_cid,
        candidate_context_cid=candidate_context_cid,
        collection_failed=False,
        diagnostics=diagnostics or {},
    )


def disposition_skip(
    *,
    certificate_cid: str,
    receipt_cid: str,
    candidate_context_cid: str = "",
    reason_code: str = "proof_cache_hit",
    diagnostics: Mapping[str, Any] | None = None,
) -> RuntimeReuseDisposition:
    return RuntimeReuseDisposition(
        action=RuntimeReuseAction.SKIP,
        reason_code=reason_code,
        artifact_role=ArtifactRole.AUTHORITATIVE_CERTIFICATE,
        certificate_cid=certificate_cid,
        receipt_cid=receipt_cid,
        candidate_context_cid=candidate_context_cid,
        collection_failed=False,
        diagnostics=diagnostics or {},
    )


def disposition_for_optional_capability_fault(
    fault: OptionalCapabilityFaultKind | str,
    *,
    capability: str,
    receipt_retained: bool = False,
    receipt_cid: str = "",
    candidate_context_cid: str = "",
    diagnostics: Mapping[str, Any] | None = None,
) -> RuntimeReuseDisposition:
    """Map every optional capability fault class to RUN or DEFERRED.

    Never raises for known fault kinds, never sets ``collection_failed``, and
    never returns ``SKIP``.  When a trusted pass receipt was retained and the
    fault is missing/incompatible/timed-out proving infrastructure, prefer
    ``DEFERRED`` so issuance can proceed later without re-running collection.
    """

    try:
        fault_kind = (
            fault
            if isinstance(fault, OptionalCapabilityFaultKind)
            else _safe_enum(
                fault, OptionalCapabilityFaultKind, field_name="fault"
            )
        )
    except ActivationContractError:
        return disposition_run(
            "exception_fail_open_to_run",
            diagnostics={
                "capability": str(capability)[:MAX_DIAGNOSTIC_VALUE_CHARS],
                "fault": "unknown",
                **dict(diagnostics or {}),
            },
        )

    capability_text = _bounded_text(
        capability, field_name="capability", max_chars=MAX_REASON_CHARS
    )
    reason = "optional_capability_%s" % fault_kind.value
    diag = {
        "capability": capability_text,
        "fault": fault_kind.value,
        **dict(diagnostics or {}),
    }

    if (
        receipt_retained
        and receipt_cid
        and fault_kind in _DEFERRED_PREFER_FAULTS
    ):
        return disposition_deferred(
            reason,
            receipt_cid=receipt_cid,
            candidate_context_cid=candidate_context_cid,
            diagnostics=diag,
        )
    return disposition_run(
        reason,
        receipt_cid=receipt_cid if receipt_retained else "",
        candidate_context_cid=candidate_context_cid,
        diagnostics=diag,
    )


# ---------------------------------------------------------------------------
# ProofReuseActivationContract@1 — sealed composition doctrine
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProofReuseActivationContract(CanonicalContract):
    """Sealed doctrine object for automatic runtime activation composition.

    Implementations of identity services, candidate stores, revalidation,
    deferred issuance, and plugin wiring must honor this contract.  It is
    intentionally pure data plus pure decision helpers: no I/O on import.
    """

    __test__: ClassVar[bool] = False
    SCHEMA: ClassVar[str] = PROOF_REUSE_ACTIVATION_CONTRACT_SCHEMA

    authority_sequence: Tuple[str, ...] = ACTIVATION_AUTHORITY_SEQUENCE
    skip_required_dimensions: Tuple[str, ...] = tuple(
        dim.value for dim in SKIP_REQUIRED_COMPARISON_DIMENSIONS
    )
    content_addressed_rehash_required: bool = True
    post_pass_observation_without_duplicate_call: bool = True
    optional_capability_collection_failure_forbidden: bool = True
    non_authorizing_roles: Tuple[str, ...] = tuple(
        role.value for role in sorted(_NON_AUTHORIZING_ROLES, key=lambda r: r.value)
    )
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "authority_sequence",
            tuple(
                _bounded_text(item, field_name="authority_sequence", required=True)
                for item in (self.authority_sequence or ())
            ),
        )
        if tuple(self.authority_sequence) != ACTIVATION_AUTHORITY_SEQUENCE:
            _raise_contract(
                "authority_sequence must match the sealed activation sequence"
            )
        object.__setattr__(
            self,
            "skip_required_dimensions",
            tuple(
                _bounded_text(item, field_name="skip_required_dimensions", required=True)
                for item in (self.skip_required_dimensions or ())
            ),
        )
        expected_dims = tuple(
            dim.value for dim in SKIP_REQUIRED_COMPARISON_DIMENSIONS
        )
        if tuple(self.skip_required_dimensions) != expected_dims:
            _raise_contract(
                "skip_required_dimensions must be ast/static/runtime/environment/policy"
            )
        for name in (
            "content_addressed_rehash_required",
            "post_pass_observation_without_duplicate_call",
            "optional_capability_collection_failure_forbidden",
        ):
            object.__setattr__(
                self, name, _bool(getattr(self, name), field_name=name)
            )
        if not self.content_addressed_rehash_required:
            _raise_contract("content-addressed rehash is mandatory")
        if not self.post_pass_observation_without_duplicate_call:
            _raise_contract("post-pass observation without duplicate call is mandatory")
        if not self.optional_capability_collection_failure_forbidden:
            _raise_contract(
                "optional capability faults must never fail collection"
            )
        object.__setattr__(
            self,
            "non_authorizing_roles",
            tuple(
                sorted(
                    {
                        _bounded_text(item, field_name="non_authorizing_roles", required=True)
                        for item in (self.non_authorizing_roles or ())
                    }
                )
            ),
        )
        expected_roles = tuple(
            role.value for role in sorted(_NON_AUTHORIZING_ROLES, key=lambda r: r.value)
        )
        if tuple(self.non_authorizing_roles) != expected_roles:
            _raise_contract("non_authorizing_roles vocabulary is sealed")
        object.__setattr__(
            self, "metadata", _bounded_mapping(self.metadata, field_name="metadata")
        )

    @property
    def interface(self) -> str:
        return PROOF_REUSE_ACTIVATION_CONTRACT_INTERFACE

    @property
    def contract_id(self) -> str:
        return self.content_id

    def role_may_authorize_skip(self, role: ArtifactRole | str) -> bool:
        role_value = (
            role
            if isinstance(role, ArtifactRole)
            else _safe_enum(role, ArtifactRole, field_name="role")
        )
        return role_value is ArtifactRole.AUTHORITATIVE_CERTIFICATE

    def evaluate_skip_admission(
        self,
        *,
        candidate: CandidateExecutionContext,
        current: CurrentExecutionContext,
        certificate: AuthoritativeCertificateBinding,
        candidate_bytes: bytes | None = None,
        certificate_bytes: bytes | None = None,
    ) -> RuntimeReuseDisposition:
        """Gate SKIP on rehash, context comparison, and certificate authority.

        Any incomplete step returns ``RUN``.  This method never raises for
        ordinary mismatch / integrity faults and never fails collection.
        """

        if candidate_bytes is not None:
            admission = admit_content_addressed_boundary(
                role=ArtifactRole.IMMUTABLE_CANDIDATE_CONTEXT,
                claimed_cid=candidate.candidate_context_id,
                canonical_bytes=candidate_bytes,
            )
            if not admission.admitted:
                return disposition_run(
                    admission.reason_code or "candidate_integrity_failed",
                    artifact_role=ArtifactRole.IMMUTABLE_CANDIDATE_CONTEXT,
                    diagnostics={"stage": "candidate_rehash"},
                )

        if certificate_bytes is not None:
            admission = admit_content_addressed_boundary(
                role=ArtifactRole.AUTHORITATIVE_CERTIFICATE,
                claimed_cid=certificate.certificate_cid,
                canonical_bytes=certificate_bytes,
            )
            if not admission.admitted:
                return disposition_run(
                    admission.reason_code or "candidate_integrity_failed",
                    artifact_role=ArtifactRole.AUTHORITATIVE_CERTIFICATE,
                    diagnostics={"stage": "certificate_rehash"},
                )

        comparison = compare_contexts_for_skip(candidate, current)
        if not comparison.matched:
            return disposition_run(
                "execution_key_mismatch"
                if "execution_key" in comparison.mismatched_dimensions
                else "policy_mismatch"
                if "policy" in comparison.mismatched_dimensions
                else "invalidation",
                diagnostics={
                    "mismatched_dimensions": list(comparison.mismatched_dimensions),
                    "missing_dimensions": list(comparison.missing_dimensions),
                },
            )

        if (
            certificate.execution_key_cid != candidate.execution_key_cid
            or certificate.candidate_context_cid != candidate.candidate_context_id
            or certificate.receipt_cid != candidate.pass_receipt_cid
        ):
            return disposition_run(
                "receipt_mismatch",
                diagnostics={"stage": "certificate_binding"},
            )

        if not certificate.may_authorize_skip:
            if certificate.simulated:
                return disposition_run(
                    "certificate_non_attested",
                    diagnostics={"stage": "simulated_certificate"},
                )
            return disposition_run(
                "certificate_non_attested"
                if not certificate.authoritative
                else "trust_policy_rejected",
                diagnostics={"stage": "certificate_not_skip_capable"},
            )

        return disposition_skip(
            certificate_cid=certificate.certificate_cid,
            receipt_cid=certificate.receipt_cid,
            candidate_context_cid=candidate.candidate_context_id,
            diagnostics={"stage": "activation_skip_admitted"},
        )

    def _payload(self) -> Dict[str, Any]:
        return {
            "contract_version": ACTIVATION_CONTRACT_VERSION,
            "interface": PROOF_REUSE_ACTIVATION_CONTRACT_INTERFACE,
            "authority_sequence": list(self.authority_sequence),
            "skip_required_dimensions": list(self.skip_required_dimensions),
            "content_addressed_rehash_required": (
                self.content_addressed_rehash_required
            ),
            "post_pass_observation_without_duplicate_call": (
                self.post_pass_observation_without_duplicate_call
            ),
            "optional_capability_collection_failure_forbidden": (
                self.optional_capability_collection_failure_forbidden
            ),
            "non_authorizing_roles": list(self.non_authorizing_roles),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProofReuseActivationContract":
        _schema_or_raise(
            payload, cls.SCHEMA, artifact_name="proof reuse activation contract"
        )
        _require_versioned_interface(
            payload,
            expected_interface=PROOF_REUSE_ACTIVATION_CONTRACT_INTERFACE,
            artifact_name="proof reuse activation contract",
        )
        _safe_reject_unknown_fields(
            payload,
            {
                "schema",
                "contract_version",
                "interface",
                "authority_sequence",
                "skip_required_dimensions",
                "content_addressed_rehash_required",
                "post_pass_observation_without_duplicate_call",
                "optional_capability_collection_failure_forbidden",
                "non_authorizing_roles",
                "metadata",
                "content_id",
                "contract_id",
            },
            artifact_name="proof reuse activation contract",
        )
        result = cls(
            authority_sequence=tuple(
                payload.get("authority_sequence") or ACTIVATION_AUTHORITY_SEQUENCE
            ),
            skip_required_dimensions=tuple(
                payload.get("skip_required_dimensions")
                or tuple(dim.value for dim in SKIP_REQUIRED_COMPARISON_DIMENSIONS)
            ),
            content_addressed_rehash_required=payload.get(
                "content_addressed_rehash_required", True
            ),
            post_pass_observation_without_duplicate_call=payload.get(
                "post_pass_observation_without_duplicate_call", True
            ),
            optional_capability_collection_failure_forbidden=payload.get(
                "optional_capability_collection_failure_forbidden", True
            ),
            non_authorizing_roles=tuple(
                payload.get("non_authorizing_roles")
                or tuple(
                    role.value
                    for role in sorted(_NON_AUTHORIZING_ROLES, key=lambda r: r.value)
                )
            ),
            metadata=payload.get("metadata") or {},
        )
        claimed = payload.get("contract_id") or payload.get("content_id")
        if claimed and claimed != result.contract_id:
            _raise_contract(
                "proof reuse activation contract content identity does not match payload"
            )
        return result

    @classmethod
    def sealed(cls) -> "ProofReuseActivationContract":
        """Return the single sealed activation contract instance payload."""

        return cls()


def artifact_role_of(value: Any) -> ArtifactRole | None:
    """Return the sealed artifact role of a known activation record."""

    if isinstance(value, LocatorHint):
        return ArtifactRole.LOCATOR_HINT
    if isinstance(value, CandidateExecutionContext):
        return ArtifactRole.IMMUTABLE_CANDIDATE_CONTEXT
    if isinstance(value, CurrentExecutionContext):
        return ArtifactRole.FRESH_CURRENT_CONTEXT
    if isinstance(value, TrustedPassReceiptBinding):
        return ArtifactRole.TRUSTED_PASS_RECEIPT
    if isinstance(value, DeferredProofRequest):
        return ArtifactRole.DEFERRED_PROOF_REQUEST
    if isinstance(value, AuthoritativeCertificateBinding):
        return ArtifactRole.AUTHORITATIVE_CERTIFICATE
    role = getattr(value, "role", None)
    if isinstance(role, ArtifactRole):
        return role
    if isinstance(role, str):
        try:
            return ArtifactRole(role)
        except ValueError:
            return None
    return None


def roles_are_distinct() -> Mapping[str, str]:
    """Return the sealed role vocabulary for documentation and tests."""

    return {role.name: role.value for role in ArtifactRole}


__all__ = [
    "ACTIVATION_AUTHORITY_SEQUENCE",
    "ACTIVATION_CONTRACT_VERSION",
    "AUTHORITATIVE_CERTIFICATE_BINDING_INTERFACE",
    "ActivationContractError",
    "ArtifactRole",
    "AuthoritativeCertificateBinding",
    "CANDIDATE_EXECUTION_CONTEXT_INTERFACE",
    "CONTENT_ADDRESSED_BOUNDARY_INTERFACE",
    "CURRENT_EXECUTION_CONTEXT_INTERFACE",
    "CandidateExecutionContext",
    "ContentAddressedBoundaryAdmission",
    "ContextComparisonResult",
    "CurrentExecutionContext",
    "DEFERRED_PROOF_REQUEST_INTERFACE",
    "DeferredProofRequest",
    "LOCATOR_HINT_INTERFACE",
    "LocatorHint",
    "OptionalCapabilityFaultKind",
    "POST_PASS_RUNTIME_OBSERVATION_INTERFACE",
    "PROOF_REUSE_ACTIVATION_CONTRACT_INTERFACE",
    "PostPassRuntimeObservation",
    "ProofReuseActivationContract",
    "RUNTIME_REUSE_DISPOSITION_INTERFACE",
    "RuntimeReuseAction",
    "RuntimeReuseDisposition",
    "SKIP_REQUIRED_COMPARISON_DIMENSIONS",
    "SkipComparisonDimension",
    "TRUSTED_PASS_RECEIPT_BINDING_INTERFACE",
    "TrustedPassReceiptBinding",
    "admit_content_addressed_boundary",
    "artifact_role_of",
    "cid_for_public_payload",
    "compare_contexts_for_skip",
    "disposition_deferred",
    "disposition_for_optional_capability_fault",
    "disposition_run",
    "disposition_skip",
    "record_post_pass_runtime_observation",
    "rehash_retained_canonical_bytes",
    "require_content_addressed_boundary",
    "roles_are_distinct",
]
